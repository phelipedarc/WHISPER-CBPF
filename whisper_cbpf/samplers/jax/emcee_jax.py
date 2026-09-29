"""The two emcee arms of the benchmark.

Arm A(1) — ``fit_emcee_jax``: emcee driving the SAME jitted JAX log-density the NUTS arm uses,
in ``vectorize=True`` mode so each step is one batched GPU call over the whole walker ensemble
rather than one Python call per walker. CPU-orchestrated, GPU-compute.

Arm A(2) — ``fit_emcee_numpy``: the realistic baseline. A pure-numpy log-posterior with no JAX
anywhere, parallelized across processes like ``whisper_cbpf.samplers.mcmc``'s ``n_jobs``. This
is what a scientist runs today with no GPU/JAX infrastructure, and it must NOT benefit from any
of the JAX work — that's the whole point of having it as a separate arm.

Both return a ``whisper_cbpf.samplers.base.SamplerResult`` built through the same helpers every
whisper_cbpf sampler uses, so all four arms of the benchmark share one output schema.
"""
from __future__ import annotations

import sys
import time

import emcee
import numpy as np
import pandas as pd

# Imported relatively -- the CPU core and this sampler now live in one package. In the legacy
# split (whisper-GPU @ 10796a0, discontinued; superseded by whisper_cbpf) this file sat in a
# separate repo cloned INTO a checkout of the CPU package, and reached that package absolutely
# off sys.path; the merge removes that indirection.

from ...models import get_model  # noqa: E402
from ...samplers.base import (  # noqa: E402
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    _pre_event_plan,
    check_burnin,
    check_not_empty,
    summarize_posterior,
)

#: Walkers per compiled vmap call in :func:`fit_emcee_jax`. ``"half"`` (the default) is half the
#: ensemble, rounded up: emcee's stretch move scores one half of the walkers per call, so every call
#: is ONE vmapped block. An int fixes the width; ``None`` is a single vmap over whatever emcee
#: passes (the whole ensemble on its first call, a half on every step).
#:
#: It was a fixed 16, which split the 30-walker half of a 60-walker ensemble into two blocks run
#: one after the other. Measured on one A6000, float64, per call at 30 walkers (compile at 60, then
#: at 30): the TDE at ``n_time=5000`` 50.6 ms at 16 against 22.5 ms at 30 (8.3 / 8.8 s compile),
#: at ``n_time=500`` 6.0 against 2.7 ms, a kilonova or an arnett supernova 0.6-0.8 against
#: 0.56-0.60 ms (1.3-3.5 s compile either way). For the default ``nwalkers=32`` the half IS 16.
#: In float32 the kilonovae compile in 1.9-2.9 s at 16 and 1.8-1.9 s at 30 (the 71 s once measured
#: at 16 in float32, and 16 s in float64, came with the old 2000-point photometry grid), and a
#: 30-walker call takes 0.47-0.58 against 0.29 ms.
#:
#: Compile time is erratic in the width on the GPU, and the width is now a user-scaled count: the
#: kilonovae's vmap compiles in 1.5-3.5 s at 16-100, 250, 500 and 1000 walkers, but takes 6-13 s at
#: 128, 21-54 s at 256 and more than 300 s at 1024 (``_diagnostics.SCORE_BLOCK``). If a large
#: power-of-two ensemble compiles for minutes, pass a nearby width, e.g. ``walker_chunk=250``.
DEFAULT_WALKER_CHUNK = "half"


def _make_walker_mapper(jax, jnp, batched, width):
    """Build a JITTED ``theta -> logp`` that ``lax.map``s ``batched`` over blocks of ``width``.

    Constant memory, regardless of ``nwalkers``. Padding keeps every block the same shape so there
    is exactly one compiled kernel — the construction ``abc_gpu`` uses.

    **The jit is not optional.** emcee calls this once PER STEP, and an un-jitted ``lax.map`` is
    re-traced and re-lowered on every one of those calls. Measured, ``nwalkers=64, nsteps=60``:
    0.462 s with a wide vmap against **22.587 s** chunked-without-jit, for a byte-identical
    ``aic = -128.451599`` — a 49x regression introduced by the chunking itself. Per call:
    0.11-0.14 ms wide, 189-207 ms chunked-unjitted, 0.232 ms chunked-and-jitted. ``abc_gpu``, which
    this mirrors, issues one ``lax.scan`` per *fit*, not per step, which is why it never showed.
    """
    @jax.jit
    def mapped(theta):
        b = theta.shape[0]
        n_blocks = -(-b // width)
        pad = n_blocks * width - b
        t = jnp.concatenate([theta, jnp.repeat(theta[-1:], pad, axis=0)], axis=0) if pad else theta
        out = jax.lax.map(batched, t.reshape(n_blocks, width, t.shape[-1]))
        return out.reshape(-1)[:b]

    return mapped


from ...models.jax._flare_spec import PARAMETERS as _PARAMS  # noqa: E402
from ...models.jax._flare_spec import flare_flux_numpy  # noqa: E402


def _flux_numpy(theta, times):
    """Arm A(2)'s forward model: pure numpy, no JAX anywhere on this path.

    Delegates to ``models.flare_spec.flare_flux_numpy`` — that module imports numpy only, so
    the no-JAX-infrastructure property of this arm is preserved while guaranteeing it evaluates
    the same function the rest of the benchmark does.
    """
    return flare_flux_numpy(dict(zip(_PARAMS, theta)), times)


def _log_prob_numpy(theta, times, values, sigmas, lows, highs):
    """Module-level (picklable, for multiprocessing) numpy log-posterior. Flat prior box."""
    if np.any(theta < lows) or np.any(theta > highs):
        return -np.inf
    mf = _flux_numpy(theta, times)
    if not np.all(np.isfinite(mf)):
        return -np.inf
    return float(-0.5 * np.sum(((values - mf) / sigmas) ** 2 + np.log(2 * np.pi * sigmas ** 2)))


def _finish(sampler_name, lc, model, names, flat, ll_flat, runtime, info, space,
            compile_time_s=0.0, fixed=None, all_names=None):
    """Shared tail: build the SamplerResult exactly like every whisper_cbpf sampler does.

    ``names`` are the sampled columns of ``flat``; ``fixed`` / ``all_names`` put a Fixed prior's
    parameters back as constant columns (not counted in AIC/BIC)."""
    from ._diagnostics import fill_fixed

    fixed = dict(fixed or {})
    all_names = list(names) if all_names is None else list(all_names)
    samples = pd.DataFrame(fill_fixed(flat, list(names), fixed, all_names), columns=all_names)
    k, n = len(names), int(len(lc.time))

    best_idx = int(np.nanargmax(ll_flat))
    best_params = {nm: float(flat[best_idx, j]) for j, nm in enumerate(names)}
    best_params = {nm: best_params.get(nm, fixed.get(nm)) for nm in all_names}
    max_log_likelihood = float(ll_flat[best_idx])

    info["compile_time_s"] = float(compile_time_s)
    attach_band_metrics(info, lc, model, best_params, space)
    aic, bic = aic_bic(max_log_likelihood, k, n)
    result = SamplerResult(
        sampler=sampler_name, model=model.name, parameters=all_names, samples=samples,
        summary=summarize_posterior(samples, all_names), best_params=best_params,
        n_data=n, n_params=k, runtime_s=float(runtime), info=info,
        max_log_likelihood=max_log_likelihood,
        aic=aic, bic=bic,
    )
    attach_predictive_metrics(result, lc, space, model=model)
    return result


def _init_walkers(nwalkers, lows, highs, seed):
    """Spread walkers uniformly across the prior box, in LINEAR coordinates -- ``init="box"``.

    The start this module used for every fit until ``init=`` existed. Uniform in linear space is not
    the prior for a LogUniform parameter (``LogUniform(100, 6000)`` puts half its mass below 775;
    this puts half below 3050), and a walker drawn where the model is dark at every epoch, or in a
    separate optimum, can stay there for the whole run: on SN2026jkr 54 of 90 GPU fits had stranded
    walkers, and on a Gaussian-bump mock 78 of 100.
    """
    rng = np.random.default_rng(int(seed))
    return rng.uniform(lows, highs, size=(nwalkers, len(lows)))


#: Width of ``init=point`` / ``init=(point, scale)``'s ball, as a fraction of each parameter's prior
#: width in its own coordinate (log width for a LogUniform). Shared with every MCMC-type sampler.
from ._diagnostics import DEFAULT_BALL_SCALE  # noqa: E402


def _in_slices(log_density, theta, width):
    """Evaluate ``log_density`` on ``theta`` in slices of exactly ``width`` rows (the last padded).

    One compiled shape, the one emcee's own first call uses, so scoring candidate starts does not
    compile a second program.
    """
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    pad = (-n) % width
    t = np.concatenate([theta, np.repeat(theta[-1:], pad, axis=0)]) if pad else theta
    return np.concatenate([log_density(t[i:i + width]) for i in range(0, len(t), width)])[:n]


def _walker_starts(init, prior, names, nwalkers, seed, log_density, sampler, scalar_density=None,
                   dtype=None):
    """Resolve ``init=`` -> ``(p0, label, detail)``, every start at a finite density.

    ``log_density`` maps an ``(m, ndim)`` array to ``(m,)`` sampled log-densities. The default start
    is ``_diagnostics.scan_starts`` -- scan, climb, keep the starts that reach the best basin -- as
    for the NUTS samplers: climbed by gradient when ``scalar_density`` (a JAX ``theta -> scalar``,
    the emcee_jax case) is given, by :func:`_diagnostics.climb_numpy` otherwise (the numpy arm).
    A point, ``(point, scale)``, one start per walker and a previous result go through
    :func:`_diagnostics.explicit_starts`. See :func:`fit_emcee_jax` for the accepted forms.
    """
    from . import _diagnostics as _dg

    rng = np.random.default_rng([int(seed), 1])

    def score(theta):
        return _in_slices(log_density, theta, int(nwalkers))

    if init is None or (isinstance(init, str) and init == "prior_scan"):
        sc = _dg.scan_starts(scalar_density, prior, names, int(nwalkers), seed, dtype,
                             n_scan=max(_dg.N_PRIOR_SCAN, 4 * int(nwalkers)),
                             score=None if scalar_density is not None else score)
        p0 = _checked(sc["starts"], score(sc["starts"]), sampler, "init='prior_scan'")
        return p0, "prior_scan", {"n_draws": int(len(sc["draws"])), "n_climbed": sc["n_climbed"],
                                  "n_reaching_best": sc["n_good"],
                                  "plateau_fraction": sc["plateau_fraction"],
                                  "climb_error": sc["climb_error"],
                                  "best_log_prob": sc["reference"],
                                  "worst_start_log_prob": float(np.min(sc["start_scores"]))}
    if isinstance(init, str) and init == "box":
        lows, highs = _dg._box(prior, names)
        unbounded = [nm for nm, lo, hi in zip(names, lows, highs) if not np.isfinite(hi - lo)]
        if unbounded:
            raise ValueError(f"{sampler}: init='box' spreads walkers uniformly over the prior box, "
                             f"and {unbounded} have no finite box. Use 'prior_scan' or 'prior'.")
        return _init_walkers(nwalkers, lows, highs, seed), "box", {}
    if isinstance(init, str) and init == "prior":
        # Independent prior draws, each redrawn while its density is -inf or NaN.
        p0 = _dg.prior_draws(prior, names, int(nwalkers), int(rng.integers(2**31)))
        lp = score(p0)
        for _ in range(100):
            bad = ~np.isfinite(lp)
            if not bad.any():
                break
            p0[bad] = _dg.prior_draws(prior, names, int(bad.sum()), int(rng.integers(2**31)))
            lp[bad] = score(p0[bad])
        return _checked(p0, lp, sampler, "init='prior'"), "prior", {}
    if isinstance(init, str):
        raise ValueError(f"{sampler}: unknown init {init!r}. Use 'prior_scan' (default), 'prior', "
                         f"'box', a point, (point, scale), one start per walker, or a result.")
    p0, _lp, label, detail = _dg.explicit_starts(init, prior, names, int(nwalkers), seed, score,
                                                 sampler, what="walker")
    return p0, label, detail


def _checked(p0, lp, sampler, what):
    """Refuse starts at a non-finite density: emcee would carry such a walker unmoved."""
    from ._diagnostics import refuse_non_finite
    return refuse_non_finite(p0, lp, sampler, what)


def _walker_diagnostics(sampler, burnin, thin, nsteps):
    """``walker_health`` on an emcee sampler: stuck walkers, and nsteps vs the LARGEST tau."""
    from . import _diagnostics as _dg

    try:
        taus = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
    except Exception:                               # noqa: BLE001 - emcee raises on short chains
        taus = np.array([np.nan])
    lp = sampler.get_log_prob(discard=int(burnin), thin=int(thin))        # (steps, walkers)
    return taus, _dg.walker_health(np.asarray(lp).T, taus, nsteps)


def fit_emcee_jax(lc, model, log_prob_fn=None, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000,
                  thin=10, seed=0, progress=False, space="auto", likelihood="auto",
                  walker_chunk=DEFAULT_WALKER_CHUNK, init="prior_scan",
                  walker_coordinates="own") -> SamplerResult:
    """Arm A(1): emcee vectorized against a jitted JAX log-density.

    emcee's ``vectorize=True`` hands the whole ``(nwalkers, ndim)`` ensemble to the callable at
    once, so one jitted+vmapped GPU call replaces ``nwalkers`` Python-level calls per step. The
    numpy<->JAX conversion happens only at that boundary; emcee itself never sees a JAX type, so
    there's no interaction with JAX tracing.

    **emcee needs a log-POSTERIOR, unlike ``nuts_gpu``.** This sampler has no mechanism for a prior:
    it samples whatever density it is given, so its stationary distribution is exactly
    ``exp(log_prob_fn)``. ``nuts_gpu`` and ``pymc_gpu`` add the prior through their own sample
    sites and therefore want a pure log-*likelihood*. Handing the same object to both — which this
    module's docstring recommends, and which ``fit_emcee_jax`` assumes with the
    comment "flat prior => log-posterior == log-likelihood + const" — is correct **only if every
    prior is Uniform**. It is not: ``kilonova_model`` gives ``temperature_floor`` a
    ``LogUniform(100, 6000)``, whose median is 775 K against a uniform's 3050 K. So the auto-built
    density here is built with ``include_prior=True``, and draws are ranked for AIC/BIC by the
    likelihood twin rather than by ``sampler.get_log_prob``.

    If you pass ``log_prob_fn`` yourself, pass a log-**posterior**; a pure likelihood silently
    imposes a flat box.

    ``likelihood`` selects the registered likelihood ``kind`` for the auto-built density —
    ``"auto"``, ``"gaussian"``, ``"upper_limits"`` (flux space only), ``"gaussian_scatter"``. It is
    ignored when ``log_prob_fn`` is supplied. With ``"gaussian_scatter"`` the prior's
    ``scatter_param`` column (default ``"sigma"``) becomes a sampled walker dimension routed to the
    likelihood instead of to ``model.predict``. The resolved class lands in ``info["likelihood"]``.

    ``init`` says where the walkers start; ``info["init"]`` records which form ran:

    ``"prior_scan"`` (default)
        Score ``max(1000, 4 * nwalkers)`` draws from the prior -- each in its own coordinate, log for
        a LogUniform -- climb the best ``nwalkers`` a short way uphill, and start the walkers ~2
        posterior sd around the climbed points that reach the best basin, exactly as ``nuts_gpu``
        starts its chains (``_diagnostics.scan_starts``). A start that is -inf, or
        whose model is dark at the data's epochs (it scores the zero-flux model's likelihood, below
        any start that puts flux where the data are), is never chosen. It replaced ``"box"``.
    ``"prior"``
        ``nwalkers`` independent prior draws, each redrawn while its density is -inf or NaN.
    ``"box"``
        Uniform in LINEAR coordinates over the prior box: the old default, kept to reproduce old
        runs. On a Gaussian-bump mock it left stuck walkers in 78 of 100 fits.
    a point -- a dict ``{name: value}`` or a ``(ndim,)`` array -- or ``(point, scale)``
        A ball around it, ``scale`` (default ``DEFAULT_BALL_SCALE`` = 1e-3) times each prior's
        width in its own coordinate, clipped into the box, bad starts redrawn.
    a ``(nwalkers, ndim)`` array
        One start per walker, as given (must be strictly inside the box, at a finite density).
    a ``SamplerResult``
        A previous fit: to continue it, to hand over from ABC, or to fit an alert's next cut (3 ->
        6 detections -> +10 d). With at least ``nwalkers`` usable draws (inside the box, distinct,
        at a finite density here) the walkers start on a random subset of them; with fewer, in a
        ball around its optimised likelihood maximum (``result.info["likelihood_max_opt"]``) or its
        ``best_params``, sized by the sd of its draws (``_diagnostics.result_starts``).
        ``info["init_detail"]["mode"]`` says which ("draws" or "ball").

    ``walker_coordinates`` says where the walkers move: ``"own"`` (default) in each prior's own
    coordinate -- the natural log of a LogUniform parameter, the parameter itself otherwise, with
    the log-Jacobian added to the density, so the posterior is the same -- or ``"linear"`` in the
    parameters themselves (whisper 0.1.x). The stretch move is affine-invariant only in the
    coordinates it moves in, and in linear ones the supernova and TDE posteriors came out too
    narrow. ``info["walker_coordinates"]`` names the log columns. The draws, the summary and
    ``samples_by_chain`` are always in parameter units; ``result.emcee_sampler`` holds the chain in
    walker coordinates.

    ``converged`` requires no stuck walker (median log-posterior, in walker coordinates, more than
    10 nats below the best walker's) and ``nsteps >= 50 x`` the LARGEST autocorrelation time. It
    used to be the autocorrelation test alone, on the MEAN time, and read True on 55 of 78 bump fits
    with stuck walkers. ``info["convergence_problems"]`` names each failed check, and a non-empty
    list warns.
    Like ``nuts_gpu``, the fit warns before sampling when float32 cannot resolve ``lc.time`` or a
    Uniform prior bound (absolute value >= 1e3), and under ``"prior_scan"`` when most of the prior
    is a flat zero-signal plateau (a time prior much wider than the data window).

    Timing: ``info["compile_time_s"]`` is the compile at every ensemble shape emcee calls with (the
    whole ensemble and both halves), ``runtime_s`` the start plus the sampling, and
    ``info["postprocess_s"]`` the likelihood re-scan and walker checks after it. ``walker_chunk``
    (default ``"half"``: one vmapped block per emcee call) is recorded as the width it resolved to
    in ``info["walker_chunk"]``; :data:`DEFAULT_WALKER_CHUNK` has the measurements.

    The prior may mix ``Uniform``, ``LogUniform``, ``Normal``, ``TruncatedNormal`` (the density is
    ``priors.log_prob_jax``'s, truncation normalisation included) and ``Fixed`` (held at its value,
    not a walker dimension, not counted in AIC/BIC; ``info["fixed"]``).

    Pre-event rows are left out before anything is built, as in every sampler
    (:func:`whisper_cbpf.samplers.base.prepare_lc`; ``info["excluded_pre_event"]``), and a model
    with a free ``t_exp`` gets the explosion-time prior the data define unless ``prior`` names one
    (``info["t_exp_prior"]``). A caller's ``log_prob_fn`` must be built on the same rows.

    Examples
    --------
    An ABC fit handed over to emcee (a ball around ABC's best point if it accepted fewer draws than
    there are walkers):

    >>> import numpy as np, whisper_cbpf as wp
    >>> from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30,
    ...                    flux=wp.get_model("flare_jax").predict(truth, t),
    ...                    flux_err=np.full(30, 0.1))
    >>> abc = wp.fit(lc, "flare_jax", sampler="abc", n_simulations=5000, quantile=0.01)
    >>> r = fit_emcee_jax(lc, "flare_jax", init=abc, nsteps=1000, burnin=300)
    >>> r.info["init"]
    'result'
    """
    check_burnin(burnin, nsteps)

    import jax
    import jax.numpy as jnp

    from . import _diagnostics as _dg
    from ._adapters import float_dtype, resolve_density as _resolve_density
    from ._adapters import free_density as _free_density
    from ._adapters import resolve_sampling_names as _resolve_sampling_names

    model = get_model(model) if isinstance(model, str) else model
    # The pre-event rule, for a direct call (through EmceeJAXSampler.fit it has already run and
    # this finds nothing left to do).
    pre_event = _pre_event_plan(lc, model, prior, sampler="emcee_jax",
                                own_density=log_prob_fn is not None, likelihood=likelihood)
    pre_event.warn()
    lc, prior = pre_event.lc, pre_event.prior
    prior = prior if prior is not None else model.default_prior
    if prior is None:
        raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
    check_not_empty(lc)                 # a caller's log_prob_fn never meets the likelihood's check
    names, likelihood = _resolve_sampling_names(lc, model, prior, log_prob_fn, space, likelihood)
    # A Fixed parameter is held at its value: `prior` / `names` are the FREE ones from here on.
    all_names, full_prior = names, prior
    prior, names, fixed = _dg.split_fixed(full_prior, all_names, "emcee_jax")
    ndim = len(names)

    log_prob_fn, space, space_source, fn_source = _resolve_density(
        lc, model, full_prior, log_prob_fn, space, all_names,
        include_prior=True, sampler="emcee_jax", likelihood=likelihood)
    log_prob_fn, _, _ = _free_density(log_prob_fn, full_prior, all_names)
    # Rank draws by LIKELIHOOD for AIC/BIC even though we SAMPLE the posterior.
    loglik_fn = getattr(log_prob_fn, "log_likelihood", None)

    # The walkers move in each prior's own coordinate (log for a LogUniform): emcee samples
    # `walker_density`, the same posterior in those coordinates (Jacobian included).
    is_log = _dg.walker_log_columns(prior, names, walker_coordinates, "emcee_jax")
    walker_density = _dg.jax_walker_density(log_prob_fn, is_log) if is_log.any() else log_prob_fn
    batched = jax.jit(jax.vmap(walker_density))
    dt = float_dtype()          # was a hard jnp.float32, which threw away x64 when it was enabled

    # BOUND THE VMAP EXTENT, ONCE, at one block per emcee call by default ("half": see
    # DEFAULT_WALKER_CHUNK for the measurements, and the widths whose compile runs to minutes).
    # `walker_chunk=None` restores the single wide vmap. The mapper is built ONCE here rather than
    # per step -- see _make_walker_mapper for what per-step costs.
    if isinstance(walker_chunk, str):
        if walker_chunk != "half":
            raise ValueError(f"emcee_jax: walker_chunk must be 'half', a positive int or None; got "
                             f"{walker_chunk!r}.")
        walker_chunk = -(-int(nwalkers) // 2)
    _chunked = (_make_walker_mapper(jax, jnp, batched, int(walker_chunk))
                if walker_chunk and nwalkers > walker_chunk else None)

    def walker_batch(walkers):          # emcee's density: rows of walker coordinates
        theta = jnp.asarray(walkers, dtype=dt)
        out = _chunked(theta) if _chunked is not None else batched(theta)
        return np.asarray(jax.block_until_ready(out), dtype=float)

    def log_prob_batch(theta_batch):    # rows of parameters -> log posterior (the starts' score)
        if not is_log.any():
            return walker_batch(theta_batch)
        y = _dg.to_walker(theta_batch, is_log)
        return walker_batch(y) - _dg.walker_log_jacobian(y, is_log)

    f32_hazard = _dg.float32_hazard(lc.time, prior, names, "emcee_jax",
                                    bool(jax.config.jax_enable_x64))

    # Compile on a throwaway ensemble so JIT time is reported separately from sampling time, at
    # EVERY shape emcee calls with: the whole ensemble (its first evaluation, and the start scoring)
    # and each half (every step: the stretch move updates one half against the other, the halves
    # sized n // 2 and n - n // 2). Warming the whole ensemble alone left the half-ensemble compile
    # inside runtime_s: 0.67-0.93 s on the supernovae and the kilonova, 8.89 s on the TDE at
    # n_time=5000.
    t_c = time.perf_counter()
    warm = _dg.prior_draws(prior, names, int(nwalkers), seed)      # any finite rows: compile only
    for m in sorted({int(nwalkers), int(nwalkers) // 2, int(nwalkers) - int(nwalkers) // 2}):
        log_prob_batch(warm[:m])
    compile_time = time.perf_counter() - t_c

    t_i = time.perf_counter()
    p0, init_label, init_detail = _walker_starts(init, prior, names, nwalkers, seed,
                                                 log_prob_batch, "emcee_jax",
                                                 scalar_density=log_prob_fn, dtype=dt)
    y0 = _dg.refuse_collapsed(_dg.to_walker(p0, is_log), "emcee_jax", f"init={init_label!r}")
    init_time = time.perf_counter() - t_i
    sampler = emcee.EnsembleSampler(nwalkers, ndim, walker_batch, vectorize=True)
    sampler._random.seed(int(seed))

    t0 = time.perf_counter()
    sampler.run_mcmc(y0, int(nsteps), progress=progress)
    runtime = time.perf_counter() - t0 + init_time
    t_post = time.perf_counter()

    flat_w = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    flat = _dg.from_walker(flat_w, is_log)
    # AIC/BIC are defined on the maximum log-LIKELIHOOD, and what emcee stores is whatever density
    # it sampled -- a log-POSTERIOR on the auto-built path. Re-evaluate the likelihood twin over
    # the thinned chain rather than reusing sampler.get_log_prob(), which would fold the prior into
    # both. In fixed-width blocks (`_diagnostics.block_scorer`): not one vmap, whose extent would be
    # a draw count, and not one draw per step (`lax.map`), which cost the TDE at n_time=5000
    # 50.9 ms a draw -- ~672 s on 13,200 draws against 156 s of sampling, outside runtime_s.
    if loglik_fn is not None:
        ll_flat = _dg.block_scorer(loglik_fn, dt)(flat)
    else:
        # A hand-built density: we cannot separate prior from likelihood, so report what was
        # sampled (in parameter units) and say so in info["max_log_likelihood_is"].
        ll_flat = (sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
                   - _dg.walker_log_jacobian(flat_w, is_log))

    taus, health = _walker_diagnostics(sampler, burnin, thin, nsteps)
    _dg.warn_if_unconverged("emcee_jax", health)
    postprocess_time = time.perf_counter() - t_post
    info = {
        "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
        "space": space, "seed": int(seed),
        "init": init_label, "init_detail": init_detail, "init_time_s": float(init_time),
        "walker_coordinates": {"coordinates": walker_coordinates,
                               "log": [nm for nm, lg in zip(names, is_log) if lg]},
        # After sampling, outside runtime_s: the likelihood re-scan and the walker checks.
        "postprocess_s": float(postprocess_time),
        "float32_hazard": f32_hazard,
        "space_source": space_source, "log_prob_fn": fn_source,
        # `type(likelihood).__name__` -- the key `samplers.base._LIKELIHOOD_KINDS` maps back to a
        # registry name so `waic`/`predictive_metrics` re-score under the density that was fitted.
        # None when the caller supplied their own log_prob_fn: we do not know then.
        "likelihood": getattr(log_prob_fn, "likelihood", None),
        "scatter_param": getattr(log_prob_fn, "scatter_param", None),
        "x64": bool(jax.config.jax_enable_x64),
        "fixed": dict(fixed),
        "walker_chunk": (int(walker_chunk) if walker_chunk else None),
        "max_log_likelihood_is": ("likelihood" if loglik_fn is not None else "sampled density"),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "mean_autocorr_time": float(np.nanmean(taus)) if np.isfinite(taus).any() else float("nan"),
        **health,
        "backend": "emcee + jitted JAX log_prob (vectorized)",
    }
    result = _finish("emcee_jax", lc, model, names, flat, ll_flat, runtime, info, space,
                     compile_time_s=compile_time, fixed=fixed, all_names=all_names)
    pre_event.stamp(result)
    # (chains, draws, k) for ArviZ ESS — emcee walkers act as the chain axis
    chain = _dg.from_walker(sampler.get_chain(discard=int(burnin), thin=int(thin)),
                            is_log)                                    # (steps, walkers, k)
    result.samples_by_chain = _dg.fill_fixed(np.swapaxes(chain, 0, 1), names, fixed, all_names)
    result.emcee_sampler = sampler
    return result


def fit_emcee_numpy(lc, model, times, values, sigmas, *, prior=None, nwalkers=32, nsteps=5000,
                    burnin=1000, thin=10, seed=0, n_jobs=8, progress=False,
                    space="flux", init="prior_scan") -> SamplerResult:
    """Arm A(2): plain-numpy emcee on N CPU cores — the realistic no-GPU baseline.

    No JAX involvement anywhere: its own numpy forward model, its own numpy log-posterior,
    parallelized over processes exactly like ``whisper_cbpf.samplers.mcmc``'s ``n_jobs`` path.
    ``init`` and ``converged`` mean what they mean for :func:`fit_emcee_jax` (the candidate starts
    are scored serially here, with the numpy density, and the default start climbs them with the
    derivative-free :func:`_diagnostics.climb_numpy`).

    **Run this in a process that has never imported JAX.** The pool uses ``fork`` (whisper_cbpf's
    MCMCSampler now spawns instead), and forking a process with JAX's threads already running
    risks deadlock — JAX itself warns about exactly this. ``run_benchmark.py`` therefore
    dispatches this arm as its own subprocess. Running it inside a JAX process may appear to
    work and then hang unpredictably, which is worse than failing outright.
    """
    check_burnin(burnin, nsteps)

    import multiprocessing

    from . import _diagnostics as _dg                   # numpy-only: keeps this arm JAX-free

    if "jax" in sys.modules:
        raise RuntimeError(
            "fit_emcee_numpy must run in a JAX-free process: forking with JAX's threads live "
            "risks deadlock, and a JAX-warmed process would also distort this arm's timing "
            "(it is meant to be the no-JAX-infrastructure baseline). Dispatch it as its own "
            "subprocess -- see run_benchmark.py.")

    check_not_empty(lc)
    prior = prior if prior is not None else model.default_prior
    names = list(model.parameters)
    lows = np.array([prior.distributions[nm].bounds[0] for nm in names], dtype=float)
    highs = np.array([prior.distributions[nm].bounds[1] for nm in names], dtype=float)
    ndim = len(names)

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    p0, init_label, init_detail = _walker_starts(
        init, prior, names, nwalkers, seed,
        lambda th: np.array([_log_prob_numpy(t, times, values, sigmas, lows, highs) for t in th]),
        "emcee_numpy")

    pool = multiprocessing.get_context("fork").Pool(int(n_jobs)) if n_jobs and n_jobs > 1 else None
    try:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, _log_prob_numpy, pool=pool,
            args=(times, values, sigmas, lows, highs))
        sampler._random.seed(int(seed))
        t0 = time.perf_counter()
        sampler.run_mcmc(p0, int(nsteps), progress=progress)
        runtime = time.perf_counter() - t0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    ll_flat = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)

    taus, health = _walker_diagnostics(sampler, burnin, thin, nsteps)
    _dg.warn_if_unconverged("emcee_numpy", health)
    info = {
        "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
        "space": space, "seed": int(seed), "n_jobs": int(n_jobs),
        "init": init_label, "init_detail": init_detail,
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "mean_autocorr_time": float(np.nanmean(taus)) if np.isfinite(taus).any() else float("nan"),
        **health,
        "backend": f"emcee + numpy log_prob ({n_jobs} processes)",
    }
    result = _finish("emcee_numpy", lc, model, names, flat, ll_flat, runtime, info, space,
                     compile_time_s=0.0)     # nothing to compile — that's the point of this arm
    chain = sampler.get_chain(discard=int(burnin), thin=int(thin))
    result.samples_by_chain = np.swapaxes(chain, 0, 1)
    result.emcee_sampler = sampler
    return result


class EmceeJAXSampler(BaseSampler):
    """whisper registry adapter for :func:`fit_emcee_jax`.

    The function form predates the registry and is kept as the public entry point (the flare
    benchmark calls it directly). This class is the thin ``BaseSampler`` shell whisper needs:
    ``get_sampler(name)`` constructs with zero arguments, so every option travels through ``fit``.

    Registered because this sampler is genuinely model-agnostic -- it consumes a ``log_prob_fn`` and
    never touches a specific forward model. ``fit_emcee_numpy`` in this same module is NOT
    registered for the opposite reason: it hard-wires the flare spec. (``snpe_gpu`` was once the
    other example here; it became model-agnostic and is now registered too.)
    """

    name = "emcee_jax"

    def fit(self, lc, model, prior=None, *, log_prob_fn=None, init="prior_scan",
            walker_coordinates="own", **kwargs) -> SamplerResult:
        """Fit via emcee against a jitted JAX log-density. See :func:`fit_emcee_jax`.

        ``log_prob_fn`` is optional: omit it and one is built from ``model.predict_jax`` and the
        likelihood ``likelihood=``/``space=`` imply. If you do pass one, pass a log-**POSTERIOR** --
        emcee has no prior mechanism of its own, so a pure log-likelihood silently imposes a flat
        box. That differs from ``nuts_gpu``, which wants a pure likelihood; see
        :func:`fit_emcee_jax`. ``init`` says where the walkers start (the prior scan by default,
        or a point, one start per walker, or a previous result such as an ABC fit), and
        ``walker_coordinates`` where they move (``"own"``: log for a LogUniform parameter).
        """
        return fit_emcee_jax(lc, model, log_prob_fn, prior=prior, init=init,
                             walker_coordinates=walker_coordinates, **kwargs)
