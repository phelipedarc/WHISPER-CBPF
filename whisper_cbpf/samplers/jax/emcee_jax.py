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

import os
import sys
import time

import emcee
import numpy as np
import pandas as pd

# Imported relatively -- the CPU core and this sampler now live in one package. In the legacy
# split this file sat in whisper-GPU, cloned INTO a whisper-labia checkout as `whisper_gpu/`,
# and reached whisper_labia absolutely off sys.path; the merge removes that indirection.

from ...models import get_model  # noqa: E402
from ...samplers.base import (  # noqa: E402
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)

#: Walkers per compiled vmap call in :func:`fit_emcee_jax`.
#:
#: Mirrors ``abc_gpu.DEFAULT_CHUNK`` and for the same reason: the extent of a vmap must never be a
#: user-scaled count on a model with a wavelength axis, because XLA's compile time — not memory —
#: is what gives out. ``None`` restores the single wide vmap this sampler used to do
#: unconditionally, which is fine for an analytic density.
DEFAULT_WALKER_CHUNK = 16


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
            compile_time_s=0.0):
    """Shared tail: build the SamplerResult exactly like every whisper_cbpf sampler does."""
    samples = pd.DataFrame(flat, columns=names)
    k, n = len(names), int(len(lc.time))

    best_idx = int(np.nanargmax(ll_flat))
    best_params = {nm: float(flat[best_idx, j]) for j, nm in enumerate(names)}
    max_log_likelihood = float(ll_flat[best_idx])

    info["compile_time_s"] = float(compile_time_s)
    attach_band_metrics(info, lc, model.name, best_params, space)
    result = SamplerResult(
        sampler=sampler_name, model=model.name, parameters=list(names), samples=samples,
        summary=summarize_posterior(samples, names), best_params=best_params,
        n_data=n, n_params=k, runtime_s=float(runtime), info=info,
        max_log_likelihood=max_log_likelihood,
        aic=float(-2.0 * max_log_likelihood + 2 * k),
        bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
    )
    attach_predictive_metrics(result, lc, space)
    return result


def _init_walkers(nwalkers, lows, highs, seed):
    """Spread walkers uniformly across the prior box (no initial guess needed — matches how
    whisper_cbpf's MCMCSampler initializes when ``initial_guess`` is None)."""
    rng = np.random.default_rng(int(seed))
    return rng.uniform(lows, highs, size=(nwalkers, len(lows)))


def fit_emcee_jax(lc, model, log_prob_fn=None, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000,
                  thin=10, seed=0, progress=False, space="auto", likelihood="auto",
                  walker_chunk=DEFAULT_WALKER_CHUNK) -> SamplerResult:
    """Arm A(1): emcee vectorized against a jitted JAX log-density.

    emcee's ``vectorize=True`` hands the whole ``(nwalkers, ndim)`` ensemble to the callable at
    once, so one jitted+vmapped GPU call replaces ``nwalkers`` Python-level calls per step. The
    numpy<->JAX conversion happens only at that boundary; emcee itself never sees a JAX type, so
    there's no interaction with JAX tracing.

    **emcee needs a log-POSTERIOR, unlike ``nuts_gpu``.** This sampler has no mechanism for a prior:
    it samples whatever density it is given, so its stationary distribution is exactly
    ``exp(log_prob_fn)``. ``nuts_gpu`` and ``pymc_gpu`` add the prior through their own sample
    sites and therefore want a pure log-*likelihood*. Handing the same object to both — which this
    module's docstring used to recommend, and which ``fit_emcee_jax`` used to assume with the
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
    """
    import jax
    import jax.numpy as jnp

    from ._adapters import float_dtype, resolve_density as _resolve_density
    from ._adapters import resolve_sampling_names as _resolve_sampling_names

    model = get_model(model) if isinstance(model, str) else model
    prior = prior if prior is not None else model.default_prior
    if prior is None:
        raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
    names, likelihood = _resolve_sampling_names(lc, model, prior, log_prob_fn, space, likelihood)
    lows = np.array([prior.distributions[nm].bounds[0] for nm in names], dtype=float)
    highs = np.array([prior.distributions[nm].bounds[1] for nm in names], dtype=float)
    ndim = len(names)

    log_prob_fn, space, space_source, fn_source = _resolve_density(
        lc, model, prior, log_prob_fn, space, names,
        include_prior=True, sampler="emcee_jax", likelihood=likelihood)
    # Rank draws by LIKELIHOOD for AIC/BIC even though we SAMPLE the posterior.
    loglik_fn = getattr(log_prob_fn, "log_likelihood", None)

    batched = jax.jit(jax.vmap(log_prob_fn))
    dt = float_dtype()          # was a hard jnp.float32, which threw away x64 when it was enabled

    # BOUND THE VMAP EXTENT, ONCE. `nwalkers` is user-scaled, and a vmap sized by a user-scaled
    # count is the trap that has stalled this project's compiles five times (see GPU_SETUP.md). It
    # was harmless while the only density here was the analytic flare; it is not once a photometric
    # model can reach this path. `walker_chunk=None` restores the single wide vmap. The mapper is
    # built ONCE here rather than per step -- see _make_walker_mapper for what per-step costs.
    _chunked = (_make_walker_mapper(jax, jnp, batched, int(walker_chunk))
                if walker_chunk and nwalkers > walker_chunk else None)

    def log_prob_batch(theta_batch):
        theta = jnp.asarray(theta_batch, dtype=dt)
        out = _chunked(theta) if _chunked is not None else batched(theta)
        return np.asarray(jax.block_until_ready(out), dtype=float)

    # Compile on a throwaway ensemble so JIT time is reported separately from sampling time.
    t_c = time.perf_counter()
    log_prob_batch(_init_walkers(nwalkers, lows, highs, seed))
    compile_time = time.perf_counter() - t_c

    p0 = _init_walkers(nwalkers, lows, highs, seed)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_batch, vectorize=True)
    sampler._random.seed(int(seed))

    t0 = time.perf_counter()
    sampler.run_mcmc(p0, int(nsteps), progress=progress)
    runtime = time.perf_counter() - t0

    flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    # AIC/BIC are defined on the maximum log-LIKELIHOOD, and what emcee stores is whatever density
    # it sampled -- a log-POSTERIOR on the auto-built path. Re-evaluate the likelihood twin over
    # the thinned chain rather than reusing sampler.get_log_prob(), which would fold the prior into
    # both. `lax.map`, not vmap: the extent is a draw count.
    if loglik_fn is not None:
        ll_flat = np.asarray(jax.block_until_ready(
            jax.lax.map(loglik_fn, jnp.asarray(flat, dtype=dt))), dtype=float)
    else:
        # A hand-built density: we cannot separate prior from likelihood, so report what was
        # sampled and say so in info["max_log_likelihood_is"].
        ll_flat = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)

    try:
        autocorr = float(np.nanmean(sampler.get_autocorr_time(tol=0)))
    except Exception:
        autocorr = float("nan")
    info = {
        "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
        "space": space, "seed": int(seed),
        "space_source": space_source, "log_prob_fn": fn_source,
        # `type(likelihood).__name__` -- the key `samplers.base._LIKELIHOOD_KINDS` maps back to a
        # registry name so `waic`/`predictive_metrics` re-score under the density that was fitted.
        # None when the caller supplied their own log_prob_fn: we do not know then.
        "likelihood": getattr(log_prob_fn, "likelihood", None),
        "scatter_param": getattr(log_prob_fn, "scatter_param", None),
        "x64": bool(jax.config.jax_enable_x64),
        "walker_chunk": (int(walker_chunk) if walker_chunk else None),
        "max_log_likelihood_is": ("likelihood" if loglik_fn is not None else "sampled density"),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "mean_autocorr_time": autocorr,
        "converged": bool(np.isfinite(autocorr) and autocorr > 0 and nsteps >= 50.0 * autocorr),
        "backend": "emcee + jitted JAX log_prob (vectorized)",
    }
    result = _finish("emcee_jax", lc, model, names, flat, ll_flat, runtime, info, space,
                     compile_time_s=compile_time)
    # (chains, draws, k) for ArviZ ESS — emcee walkers act as the chain axis
    chain = sampler.get_chain(discard=int(burnin), thin=int(thin))       # (steps, walkers, k)
    result.samples_by_chain = np.swapaxes(chain, 0, 1)
    result.emcee_sampler = sampler
    return result


def fit_emcee_numpy(lc, model, times, values, sigmas, *, prior=None, nwalkers=32, nsteps=5000,
                    burnin=1000, thin=10, seed=0, n_jobs=8, progress=False,
                    space="flux") -> SamplerResult:
    """Arm A(2): plain-numpy emcee on N CPU cores — the realistic no-GPU baseline.

    No JAX involvement anywhere: its own numpy forward model, its own numpy log-posterior,
    parallelized over processes exactly like ``whisper_cbpf.samplers.mcmc``'s ``n_jobs`` path.

    **Run this in a process that has never imported JAX.** The pool uses ``fork`` (matching
    whisper_cbpf's MCMCSampler), and forking a process with JAX's threads already running
    risks deadlock — JAX itself warns about exactly this. ``run_benchmark.py`` therefore
    dispatches this arm as its own subprocess. Running it inside a JAX process may appear to
    work and then hang unpredictably, which is worse than failing outright.
    """
    import multiprocessing

    if "jax" in sys.modules:
        raise RuntimeError(
            "fit_emcee_numpy must run in a JAX-free process: forking with JAX's threads live "
            "risks deadlock, and a JAX-warmed process would also distort this arm's timing "
            "(it is meant to be the no-JAX-infrastructure baseline). Dispatch it as its own "
            "subprocess -- see run_benchmark.py.")

    prior = prior if prior is not None else model.default_prior
    names = list(model.parameters)
    lows = np.array([prior.distributions[nm].bounds[0] for nm in names], dtype=float)
    highs = np.array([prior.distributions[nm].bounds[1] for nm in names], dtype=float)
    ndim = len(names)

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    p0 = _init_walkers(nwalkers, lows, highs, seed)

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

    try:
        autocorr = float(np.nanmean(sampler.get_autocorr_time(tol=0)))
    except Exception:
        autocorr = float("nan")
    info = {
        "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
        "space": space, "seed": int(seed), "n_jobs": int(n_jobs),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "mean_autocorr_time": autocorr,
        "converged": bool(np.isfinite(autocorr) and autocorr > 0 and nsteps >= 50.0 * autocorr),
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
    registered for the opposite reason: it hard-wires the flare spec. (``snpe_gpu`` used to be the
    other example here; it became model-agnostic and is now registered too.)
    """

    name = "emcee_jax"

    def fit(self, lc, model, prior=None, *, log_prob_fn=None, **kwargs) -> SamplerResult:
        """Fit via emcee against a jitted JAX log-density. See :func:`fit_emcee_jax`.

        ``log_prob_fn`` is optional: omit it and one is built from ``model.predict_jax`` and the
        likelihood ``likelihood=``/``space=`` imply. If you do pass one, pass a log-**POSTERIOR** --
        emcee has no prior mechanism of its own, so a pure log-likelihood silently imposes a flat
        box. That differs from ``nuts_gpu``, which wants a pure likelihood; see
        :func:`fit_emcee_jax`.
        """
        return fit_emcee_jax(lc, model, log_prob_fn, prior=prior, **kwargs)
