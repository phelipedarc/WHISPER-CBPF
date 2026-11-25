"""emcee's ensemble sampler on the device (``emcee_jax``), and its numpy baseline.

``fit_emcee_jax`` (registered as ``emcee_jax``): emcee's stretch move run entirely on the device.
Every step of K ensembles runs inside one compiled ``lax`` loop -- the
:func:`~whisper_cbpf.samplers.jax.batch.fit_batch` engine, for one light curve -- with an optional
pilot that restarts every ensemble from the best pilot ensemble. It used to drive the emcee
library from Python, one host round trip per half-step; the draws for a given seed changed with the move to the device
(different random streams), the ``SamplerResult`` did not.

``fit_emcee_numpy``: the realistic baseline. A pure-numpy log-posterior with no JAX anywhere,
run by the emcee library and parallelized across processes like ``whisper_cbpf.samplers.mcmc``'s
``n_jobs``. It must NOT benefit from any of the JAX work, which is why it is a separate arm.

Both return a ``whisper_cbpf.samplers.base.SamplerResult``, the schema every whisper_cbpf sampler
shares.
"""
from __future__ import annotations

import sys
import time
import types

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
    _zero_flux_given,
    check_burnin,
    check_not_empty,
    summarize_posterior,
)

#: How :func:`fit_emcee_jax` evaluates each half-ensemble on the device. ``"half"`` (the default) is
#: half the ensemble, rounded up: the stretch move scores one half of each ensemble's walkers per
#: half-step, so every half-step is ONE vmapped block (``info["walker_chunk"]`` records the width).
#: An int splits each half into ``lax.map`` blocks of that many walkers (padded), which bounds the
#: vmap width the compiler sees; ``None`` is one vmap, as ``"half"``, recorded as ``None``.
#:
#: The width matters for compile time, not for the numbers. Compile time is erratic in the width on
#: the GPU: the kilonovae's vmap compiles in 1.5-3.5 s at 16-100, 250, 500 and 1000 walkers, but
#: takes 6-13 s at 128, 21-54 s at 256 and more than 300 s at 1024 (``_diagnostics.SCORE_BLOCK``).
#: If a large power-of-two ensemble compiles for minutes, pass a nearby width, e.g.
#: ``walker_chunk=250``.
DEFAULT_WALKER_CHUNK = "half"

#: Ensembles :func:`fit_emcee_jax` runs by default. Their agreement is the split R-hat between
#: them (``info["ensemble_rhat"]``): between independent runs only without a pilot, which
#: restarts them all from one pool.
DEFAULT_N_ENSEMBLES = 4

#: ``pilot_steps="auto"`` is ``nsteps // PILOT_FRACTION``: the speed study's R2 runs used pilots
#: of 0.2-0.375 of the main run (stream 4000 + 20 000 steps, analytical 3000 + 15 000, mosfit
#: 1500 + 4000).
PILOT_FRACTION = 5

#: The pilot keeps at most this many states per walker (from its second half) to restart from.
PILOT_KEEP = 64


#: ``until_converged=True`` with ``max_steps="auto"``: the main run stops at this many times
#: ``nsteps`` (200 000 steps at the default 5000). Measured on the five JAX TDE models, each on its
#: own light curve (35 epochs x 5 bands, one RTX A6000, 4 x 32 walkers): the runs that converged
#: needed 7 000-9 000 steps (tde_stream_stream) and 33 000-73 000 (tde_mosfit). Within
#: :data:`UNTIL_MAX_TIME_S` only the fastest models reach this cap, so it bounds memory and the
#: cost of the final checks rather than the slow models.
UNTIL_MAX_STEPS_FACTOR = 40

#: With ``until_converged=True`` the checks that rank every draw (R-hat and ESS) run again only
#: once the kept chain has grown by this fraction since they last ran; the cheap checks (stuck and
#: frozen walkers, chain length over the autocorrelation time) run after every chunk. On a TDE
#: light curve the ranking checks cost 10 s at 60 000 steps against 2.2 s per 1000-step chunk.
FULL_CHECK_GROWTH = 0.25

#: ``until_converged=True`` with ``max_time_s="auto"``: no new chunk starts that would end after
#: this many seconds from the call (compile included). Measured as for
#: :data:`UNTIL_MAX_STEPS_FACTOR`: every converged run finished within 600 s with a cold compile
#: (tde_stream_stream 25-29 s, tde_mosfit 291-556 s) and none converged between 600 and 900 s;
#: tde_cooling_envelope, tde_analytical and tde_fallback had not converged at 900 s (the chain
#: length over the autocorrelation time, N/tau >= 50, fails last), so they stop here and warn.
UNTIL_MAX_TIME_S = 600.0


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


def ensemble_seeds(seed, n_ensembles):
    """The integer seed of each of :func:`fit_emcee_jax`'s ensembles, derived from ``seed``.

    Each ensemble draws its start and its moves from its own stream. The seeds come from
    ``numpy.random.SeedSequence([seed, ...])``, not ``seed + i``, so ``seed=0`` and ``seed=1`` share
    no ensemble.

    Parameters
    ----------
    seed : int
        The fit's seed.
    n_ensembles : int
        How many ensembles.

    Returns
    -------
    list of int
        One non-negative seed below ``2**31`` per ensemble; the first ``n`` of a longer list are
        the same.

    Examples
    --------
    >>> from whisper_cbpf.samplers.jax.emcee_jax import ensemble_seeds
    >>> a = ensemble_seeds(0, 4)
    >>> len(a), len(set(a)), ensemble_seeds(0, 2) == a[:2]
    (4, 4, True)
    >>> set(a) & set(ensemble_seeds(1, 4))
    set()
    """
    ss = np.random.SeedSequence([int(seed), 0x656D6365])
    return [int(v) >> 1 for v in ss.generate_state(int(n_ensembles), dtype=np.uint32)]


def pilot_plan(pilot, pilot_steps, n_ensembles, nsteps):
    """The pilot :func:`fit_emcee_jax` runs before its main run, or ``None`` for no pilot.

    Parameters
    ----------
    pilot : {"auto", True, False}
        ``"auto"``: a pilot when there are at least two ensembles to choose between.
    pilot_steps : "auto" or int
        ``"auto"`` is ``nsteps // PILOT_FRACTION``; an int is the pilot's length in steps.
    n_ensembles, nsteps : int
        The fit's ensembles and main-run steps.

    Returns
    -------
    dict or None
        ``steps``, ``burnin`` (half the pilot), ``keep`` and ``thin``: the pilot keeps ``keep``
        states per walker, one every ``thin`` steps after ``burnin``, to restart from. ``None``
        when there is no pilot (``pilot=False``; ``"auto"`` with one ensemble, or with an
        ``"auto"`` length under 2 steps).

    Raises
    ------
    ValueError
        ``pilot`` is not one of the three, or an explicit ``pilot_steps`` is below 2.

    Examples
    --------
    >>> from whisper_cbpf.samplers.jax.emcee_jax import pilot_plan
    >>> pilot_plan("auto", "auto", 4, 5000)
    {'steps': 1000, 'burnin': 500, 'keep': 64, 'thin': 7}
    >>> pilot_plan("auto", "auto", 1, 5000) is None, pilot_plan(False, 300, 4, 5000) is None
    (True, True)
    """
    if isinstance(pilot, str) and pilot == "auto":
        on = int(n_ensembles) > 1
    elif isinstance(pilot, (bool, np.bool_)):
        on = bool(pilot)
    else:
        raise ValueError(f"emcee_jax: pilot must be 'auto', True or False; got {pilot!r}.")
    if not on:
        return None
    if isinstance(pilot_steps, str) and pilot_steps == "auto":
        steps = int(nsteps) // PILOT_FRACTION
        if steps < 2:
            return None
    else:
        steps = int(pilot_steps)
        if steps < 2:
            raise ValueError(f"emcee_jax: pilot_steps={pilot_steps!r}: a pilot needs at least 2 "
                             f"steps (it keeps states from its second half). Pass pilot=False "
                             f"for none.")
    burn = steps // 2
    keep = min(steps - burn, PILOT_KEEP)
    return {"steps": steps, "burnin": burn, "keep": keep, "thin": (steps - burn) // keep}


def until_plan(nsteps, burnin, thin, check_every="auto", max_steps="auto", max_time_s="auto"):
    """The chunks :func:`fit_emcee_jax` runs with ``until_converged=True``.

    The first chunk is the pilot, the burn-in and the first ``nsteps - burnin`` steps (rounded
    down to a multiple of ``thin``); every later chunk is ``check_every`` more steps, and the
    convergence checks run after each one.

    Parameters
    ----------
    nsteps, burnin, thin : int
        The fit's main-run steps, burn-in and thinning.
    check_every : "auto" or int
        Steps per chunk after the first, a multiple of ``thin``. ``"auto"``: half of
        ``nsteps - burnin``, rounded down to a multiple of ``thin`` and at least ``thin``.
    max_steps : "auto", int or None
        The most main-run steps (burn-in included, pilot not). ``"auto"``:
        :data:`UNTIL_MAX_STEPS_FACTOR` x ``nsteps``; ``None``: no cap.
    max_time_s : "auto", float or None
        No chunk starts that would end more than this many seconds after the call began.
        ``"auto"``: :data:`UNTIL_MAX_TIME_S`; ``None``: no cap.

    Returns
    -------
    dict
        ``first_keep`` (kept states of the first chunk), ``every_keep`` (of each later chunk),
        ``check_every``, ``max_steps`` and ``max_time_s``.

    Raises
    ------
    ValueError
        ``check_every`` not a positive multiple of ``thin``, ``max_steps`` below ``nsteps``,
        ``max_time_s`` not positive, or a value that is neither ``"auto"``, a number nor ``None``.

    Examples
    --------
    >>> from whisper_cbpf.samplers.jax.emcee_jax import until_plan
    >>> until_plan(5000, 1000, 10)
    {'first_keep': 400, 'every_keep': 200, 'check_every': 2000, 'max_steps': 200000, 'max_time_s': 600.0}
    >>> until_plan(300, 100, 7, max_steps=None, max_time_s=None)["check_every"]
    98
    """
    nsteps, burnin, thin = int(nsteps), int(burnin), int(thin)
    first_keep = (nsteps - burnin) // thin
    if isinstance(check_every, str):
        if check_every != "auto":
            raise ValueError(f"emcee_jax: check_every must be 'auto' or a number of steps; got "
                             f"{check_every!r}.")
        every = max(thin, ((nsteps - burnin) // 2) // thin * thin)
    else:
        every = int(check_every)
        if every < thin or every % thin:
            raise ValueError(f"emcee_jax: check_every={check_every!r} must be a positive multiple "
                             f"of thin={thin}, so every chunk keeps whole thinned draws.")
    if isinstance(max_steps, str):
        if max_steps != "auto":
            raise ValueError(f"emcee_jax: max_steps must be 'auto', a number of steps or None; "
                             f"got {max_steps!r}.")
        cap = UNTIL_MAX_STEPS_FACTOR * nsteps
    elif max_steps is None:
        cap = None
    else:
        cap = int(max_steps)
        if cap < nsteps:
            raise ValueError(f"emcee_jax: max_steps={max_steps!r} is below nsteps={nsteps}; the "
                             f"first chunk alone runs nsteps.")
    if isinstance(max_time_s, str):
        if max_time_s != "auto":
            raise ValueError(f"emcee_jax: max_time_s must be 'auto', a number of seconds or None; "
                             f"got {max_time_s!r}.")
        secs = float(UNTIL_MAX_TIME_S)
    elif max_time_s is None:
        secs = None
    else:
        secs = float(max_time_s)
        if not secs > 0:
            raise ValueError(f"emcee_jax: max_time_s={max_time_s!r} must be positive or None.")
    return {"first_keep": first_keep, "every_keep": every // thin, "check_every": every,
            "max_steps": cap, "max_time_s": secs}


def _convergence_check(chain_w, lp_c, is_log, names, k, thin, main_steps, init_label, full):
    """The convergence checks of ``diagnostics()`` on a kept chain, between chunks, cheap first.

    Stage 1 (cheap): stuck and frozen walkers, and the chain at least 50 of its largest
    autocorrelation times long (the sampler's own walker checks). Stage 2: R-hat between the
    ensembles. Stage 3: the report's chain rows (``results._ensemble_rows``: R-hat across walkers,
    bulk and tail ESS, and the rest again). A stage runs only when the ones before it passed, and
    stages 2 and 3, which rank every draw, only when ``full`` is True. Returns ``(passed, stage,
    record)``: ``stage`` is the last stage run, ``record`` the per-chunk entry of
    ``info["until_converged"]["chunks"]`` (``None`` for a metric not computed)."""
    from ... import results as _R
    from . import _diagnostics as _dg
    from .batch import _kept_chain_health

    _taus, _by, health = _kept_chain_health(chain_w, lp_c, thin, main_steps, n_ensembles=k)
    tau = health["max_autocorr_time"]
    n_over_tau = float(main_steps / tau) if np.isfinite(tau) and tau > 0 else float("nan")
    record = {"main_steps": int(main_steps), "kept_draws": int(chain_w.shape[0]), "stage": 1,
              "passed": False, "failed_checks": [], "max_autocorr_time": float(tau),
              "n_over_tau": n_over_tau, "ensemble_rhat_max": None, "walker_rhat_max": None,
              "ess_bulk_min": None, "ess_tail_min": None,
              "stuck_walkers": len(health["stuck_walkers"]),
              "frozen_walkers": len(health["frozen_walkers"])}
    failed = record["failed_checks"]
    if health["stuck_walkers"]:
        failed.append("stuck walkers")
    if health["frozen_walkers"]:
        failed.append("frozen walkers")
    if not (n_over_tau >= _R.N_OVER_TAU_MIN):
        failed.append("chain length / autocorrelation time")
    if failed or not full:
        return False, 1, record
    chain = _dg.from_walker(chain_w, is_log)
    record["stage"] = 2
    ens = {"rhat": {}}
    if k > 1:
        ens = _ensemble_rhat(chain, k, names, ess=False)          # the R-hat only
        vals = [float(v) for v in ens["rhat"].values()]
        record["ensemble_rhat_max"] = (max(vals) if vals and np.all(np.isfinite(vals))
                                       else float("nan"))
        if not record["ensemble_rhat_max"] < _dg.RHAT_MAX:
            failed.append("R-hat between independent ensembles (largest)")
            return False, 2, record
    record["stage"] = 3
    info = {"stuck_walkers": health["stuck_walkers"], "frozen_walkers": health["frozen_walkers"],
            "nwalkers": int(chain_w.shape[1]), "init": init_label, "max_autocorr_time": tau,
            "nsteps": int(main_steps), "ensemble_rhat": ens["rhat"], "n_ensembles": int(k)}
    view = types.SimpleNamespace(info=info, parameters=list(names),
                                 samples=pd.DataFrame(columns=list(names)))
    rows = _R._ensemble_rows(view, np.swapaxes(chain, 0, 1))
    failed += [r.check for r in rows if r.passed is False]
    # an R-hat or ESS that could not be computed is not a pass
    failed += [r.check for r in rows if r.passed is None and ("R-hat" in r.check or "ESS" in
                                                              r.check)]
    for key, check in (("walker_rhat_max", "split R-hat across walkers (largest)"),
                       ("ess_bulk_min", "bulk ESS (smallest)"),
                       ("ess_tail_min", "tail ESS (smallest)")):
        record[key] = next((r.value for r in rows if r.check == check), None)
    record["passed"] = not failed
    return not failed, 3, record


def _walker_width(walker_chunk, nwalkers):
    """``(width recorded in info, width of the loop's lax.map blocks or None)``."""
    if isinstance(walker_chunk, str):
        if walker_chunk != "half":
            raise ValueError(f"emcee_jax: walker_chunk must be 'half', a positive int or None; got "
                             f"{walker_chunk!r}.")
        return -(-int(nwalkers) // 2), None
    if walker_chunk is None:
        return None, None
    width = int(walker_chunk)
    if width < 1:
        raise ValueError(f"emcee_jax: walker_chunk must be 'half', a positive int or None; got "
                         f"{walker_chunk!r}.")
    return width, width


def _ensemble_inits(init, prior, names, nwalkers, k, seeds):
    """``(mode, inits)`` for :func:`batch._run_ensembles`: ``inits`` is ``None`` for the scans,
    else one ``init=`` per ensemble."""
    from . import _diagnostics as _dg

    if init is None or (isinstance(init, str) and init == "prior_scan"):
        return "prior_scan", None
    if isinstance(init, str) and init == "prior":
        return "prior", None
    if isinstance(init, str) and init == "box":
        lows, highs = _dg._box(prior, names)
        unbounded = [nm for nm, lo, hi in zip(names, lows, highs) if not np.isfinite(hi - lo)]
        if unbounded:
            raise ValueError(f"emcee_jax: init='box' spreads walkers uniformly over the prior box, "
                             f"and {unbounded} have no finite box. Use 'prior_scan' or 'prior'.")
        return "box", [_init_walkers(nwalkers, lows, highs, s) for s in seeds]
    if isinstance(init, str):
        raise ValueError(f"emcee_jax: unknown init {init!r}. Use 'prior_scan' (default), 'prior', "
                         f"'box', a point, (point, scale), one start per walker, or a result.")
    if not (_dg._is_result(init) or isinstance(init, (dict, tuple))):
        try:
            arr = np.asarray(init, dtype=float)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.ndim == 3:
            if arr.shape[:2] != (k, nwalkers):
                raise ValueError(f"emcee_jax: an init array of starts per ensemble must be "
                                 f"(n_ensembles, nwalkers, ndim) = ({k}, {nwalkers}, "
                                 f"{len(names)}); got {arr.shape}.")
            return "explicit", list(arr)
    return "explicit", [init] * k


def _ensemble_rhat(chain, n_ensembles, names, ll=None, ess=True):
    """Rank-normalised split R-hat between the ensembles of a kept chain, and the ESS.

    ``chain`` is ``(n_keep, n_ensembles * W, D)`` (ensemble-major walkers). For the R-hat each
    ensemble is one chain of its draws in step order, so the split compares each ensemble's early
    and late halves as well as the ensembles. The bulk and tail ESS count every walker as a chain:
    in the step-order sequence the W walkers of one step sit side by side, nearly independent, which
    hides the autocorrelation along each walker and inflates the ESS by about W. ``ess=False``
    skips that second ranking pass (the ESS entries are then NaN)."""
    from . import _diagnostics as _dg

    n_keep, n_w, d = chain.shape
    per = n_w // n_ensembles
    by = np.stack([chain[:, e * per:(e + 1) * per].reshape(n_keep * per, d)
                   for e in range(n_ensembles)])
    ll_by = (None if ll is None else
             np.stack([ll[:, e * per:(e + 1) * per].reshape(-1) for e in range(n_ensembles)]))
    out = _dg.rank_diagnostics(by, names, ll_by)
    if ess:
        walkers = _dg.rank_diagnostics(np.swapaxes(chain, 0, 1), names)
        out["ess_bulk"], out["ess_tail"] = walkers["ess_bulk"], walkers["ess_tail"]
    else:
        out["ess_bulk"] = {nm: float("nan") for nm in names}
        out["ess_tail"] = {nm: float("nan") for nm in names}
    return out


def fit_emcee_jax(lc, model, log_prob_fn=None, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000,
                  thin=10, seed=0, progress=False, space="auto", likelihood="auto",
                  walker_chunk=DEFAULT_WALKER_CHUNK, init="prior_scan",
                  walker_coordinates="own", n_ensembles=DEFAULT_N_ENSEMBLES, pilot="auto",
                  pilot_steps="auto", until_converged=False, max_steps="auto",
                  max_time_s="auto", check_every="auto") -> SamplerResult:
    """emcee's stretch move, run entirely on the device: K ensembles in one compiled loop.

    The sampler is emcee's ``StretchMove`` (Goodman & Weare 2010, a = 2, emcee's randomised
    red-blue split), run by the :func:`~whisper_cbpf.samplers.jax.batch.fit_batch` engine: every
    step of every ensemble runs inside ONE compiled ``lax.fori_loop`` on the device, with no
    return to Python between steps. ``n_ensembles`` ensembles of ``nwalkers`` walkers each
    advance together, every half-step one batched evaluation of ``n_ensembles x nwalkers / 2``
    models. Each ensemble has its own random stream (:func:`ensemble_seeds`) and its own start.
    Without a pilot the split R-hat between them (``info["ensemble_rhat"]``) compares independent
    runs; with a pilot it does not (see below).

    With a pilot (the default for two or more ensembles), the ensembles first run
    ``pilot_steps`` steps; then EVERY ensemble, the best one included, restarts on its own random
    draws, without replacement, of the kept second-half pilot states of the ONE ensemble that
    reached the highest log likelihood, and the main run of ``nsteps`` steps follows, all in the
    same compiled loop. This is the speed study's recipe R2: it rescues an ensemble stranded in a
    lower mode (on the TDE light curves, 17-49 nats below the best). It follows the best ensemble
    whatever the gap, so it pools all the ensembles into one peak:

    - A second peak of comparable probability is dropped and the fit can report "converged"
      without it (on MOSFiT, a second ``starmass``-``efficiency`` peak 0-0.5 nats lower).
    - After the restart the R-hat between the ensembles tests mixing within that one peak, not
      that independent starts agree: the ensembles are not independent runs
      (``info["pilot"]["independent"]`` is False).
    - The pilot's own R-hat between the ensembles (``info["pilot"]["rhat"]``, before the restart)
      is the only record of disagreement between the starts; no check reads it.
    - If the best ensemble was itself still leaving a lower trap, trap states enter the pool and
      restarted walkers can be stuck there.

    ``info["pilot"]`` records the pilot's best log likelihood per ensemble, the gaps to the best
    (``gap_each``), the ensembles restarted (``restarted_ensembles``: all of them) and the pilot
    R-hat. When the posterior may have several peaks, check ``info["pilot"]["rhat"]`` and
    ``["gap_each"]``, or run with ``pilot=False`` (each ensemble keeps its own start, and the
    R-hat compares independent runs) or with nested sampling.

    **The density is a log-POSTERIOR.** emcee has no prior mechanism of its own: it samples
    ``exp(log_prob_fn)``. With ``log_prob_fn`` omitted, the density is
    :func:`~whisper_cbpf.samplers.jax._adapters.log_density`'s, the prior plus the likelihood
    ``wp.fit`` uses, with the data as arguments of one compiled program, so light curves of the
    same model and size share the compile. If you pass ``log_prob_fn``, pass a log-posterior (a
    pure likelihood silently imposes a flat box); it may carry ``.log_likelihood`` for AIC/BIC.
    A model whose ``log_prob_jax`` is a pure likelihood is refused.

    Parameters
    ----------
    lc : LightCurve
        The data. Pre-event rows are left out before anything is built, as in every sampler
        (``info["excluded_pre_event"]``), and a free ``t_exp`` gets the explosion-time prior the
        data define unless ``prior`` names one (``info["t_exp_prior"]``).
    model : str or Model
        A JAX model (``predict_jax``), or any model with ``log_prob_fn``.
    log_prob_fn : callable, optional
        ``theta -> log posterior`` (a JAX scalar), ``theta`` in ``model.parameters`` order.
    prior : Prior, optional
        Default: the model's. ``Uniform``, ``LogUniform``, ``Normal``, ``TruncatedNormal`` and
        ``Fixed`` (held at its value, not a walker dimension, not counted in AIC/BIC;
        ``info["fixed"]``).
    nwalkers : int
        Walkers per ensemble (at least twice the free parameters).
    nsteps, burnin, thin : int
        The main run's steps; the kept draws are the states after steps
        ``burnin + thin - 1, burnin + 2 thin - 1, ...``, as emcee's ``get_chain``.
    seed : int
        Seeds every ensemble's start and moves. A fixed seed gives identical draws on the CPU; on
        a GPU only with XLA's deterministic flags (``XLA_FLAGS="--xla_gpu_deterministic_ops=true
        --xla_gpu_autotune_level=0"``, set before JAX starts). Without them two GPU runs with the
        same seed draw from the same posterior but differ draw by draw, and with
        ``until_converged`` they can stop at different lengths.
    progress : bool
        Ignored: the loop runs on the device without returning to Python. Kept for old calls.
    space, likelihood : str
        As for ``wp.fit``: the residual space and the likelihood kind (``"auto"``,
        ``"gaussian"``, ``"upper_limits"``, ``"gaussian_scatter"``, whose ``scatter_param``
        column becomes a walker dimension). Ignored for the density when ``log_prob_fn`` is given.
    walker_chunk : "half", int or None
        How each half-ensemble is evaluated; see :data:`DEFAULT_WALKER_CHUNK`.
    init : str, point, array or SamplerResult
        Where each ensemble starts (``info["init"]``; see Notes).
    walker_coordinates : {"own", "linear"}
        ``"own"`` (default): the walkers move in each prior's own coordinate, the natural log of a
        LogUniform parameter, with the log-Jacobian added, so the posterior is the same.
        ``"linear"``: in the parameters themselves (whisper 0.1.x).
    n_ensembles : int
        Independent ensembles (default :data:`DEFAULT_N_ENSEMBLES`).
    pilot : {"auto", True, False}
        The R2 pilot and restart; ``"auto"`` runs it for two or more ensembles.
    pilot_steps : "auto" or int
        The pilot's length; ``"auto"`` is ``nsteps // 5`` (:func:`pilot_plan`).
    until_converged : bool, default False
        Run until the convergence checks pass, in chunks (:func:`until_plan`): the pilot, the
        burn-in and the main run to ``nsteps`` first, then ``check_every`` more steps at a time,
        each chunk one call of the SAME compiled program (the chain's state stays on the device;
        only the kept draws come back for the checks). After every chunk the checks of
        ``diagnostics()`` run on all the draws kept so far -- no stuck or frozen walker, the chain
        at least 50 of its largest autocorrelation times long, R-hat between the ensembles
        < 1.01 and across walkers < 1.05, bulk and tail ESS >= 400 -- and the run stops when they
        all pass, or when the next chunk would pass ``max_steps`` or ``max_time_s``. The checks
        run cheap first: the R-hat and ESS checks, which rank every draw, run only once the walker
        and chain-length checks pass, and again only after the chain grew by
        :data:`FULL_CHECK_GROWTH`. ``info["until_converged"]`` records the steps run, the stop
        reason and each chunk's checks. ``False``: exactly ``nsteps`` steps, in one call.
        Off by default: on the JAX TDE models a run until converged takes 25-30 s
        (tde_stream_stream) to 5-10 min (tde_mosfit), and the slower-mixing models reach the
        :data:`UNTIL_MAX_TIME_S` cap without converging, so the fixed-length run stays the quick
        default and ``wp.fit``'s verdict names ``until_converged=True`` when the run was short.
    max_steps : "auto", int or None
        With ``until_converged``: the most main-run steps (burn-in included, pilot not);
        ``"auto"`` is :data:`UNTIL_MAX_STEPS_FACTOR` x ``nsteps``.
    max_time_s : "auto", float or None
        With ``until_converged``: no chunk starts that would end more than this many seconds after
        the call began (compile included); ``"auto"`` is :data:`UNTIL_MAX_TIME_S`. The first chunk
        always runs. It is not a hard wall-time limit: the last checks and the post-processing
        come after it (10-55 s more on the TDE models at 600-900 s).
    check_every : "auto" or int
        With ``until_converged``: steps per chunk after the first, a multiple of ``thin``;
        ``"auto"`` is half of ``nsteps - burnin``.

    Returns
    -------
    SamplerResult
        ``sampler="emcee_jax"``. ``samples`` holds every ensemble's kept draws (step-major, as
        emcee's flat chain); ``samples_by_chain`` is ``(n_ensembles * nwalkers, draws,
        parameters)``, the walkers of ensemble ``e`` at rows ``e * nwalkers`` to
        ``(e + 1) * nwalkers - 1``. ``info`` has ``nwalkers`` (all walkers), ``n_ensembles``,
        ``walkers_per_ensemble``, ``ensemble_seeds``, ``ensemble_rhat`` / ``ensemble_ess_bulk``
        / ``ensemble_ess_tail`` (per parameter; the ESS counts every walker as a chain) and
        ``ensemble_rhat_log_likelihood``, ``pilot``,
        the walker checks (``converged``, ``convergence_problems``, ``stuck_walkers``,
        ``frozen_walkers``, ``max_autocorr_time``, the largest over the ensembles), and the
        timings: ``compile_time_s`` (the loop and the start's scorer), ``runtime_s`` (start and
        run), ``postprocess_s``.

    Raises
    ------
    ValueError
        ``burnin >= nsteps``, a thinning that keeps no draw, fewer than ``2 * ndim`` walkers,
        fewer than one ensemble, an unknown ``init``, ``walker_chunk`` or ``pilot``, a start at a
        non-finite density, or a model whose ``log_prob_jax`` is a pure likelihood.

    Notes
    -----
    ``init`` forms, each applied to every ensemble with that ensemble's own seed:

    ``"prior_scan"`` (default)
        Score ``max(1000, 4 * nwalkers)`` prior draws, each in its own coordinate, climb the best
        32 with the derivative-free climb (one batched evaluation per round for all ensembles),
        and spread the walkers two conditional sd around the climbed points that reach the best
        basin (``_diagnostics.scan_starts``). A start whose model is dark at the data's epochs is
        never chosen.
    ``"prior"``
        Independent prior draws, each redrawn while its density is -inf or NaN.
    ``"box"``
        Uniform in LINEAR coordinates over the prior box: the old default, kept to reproduce old
        runs. On a Gaussian-bump mock it left stuck walkers in 78 of 100 fits.
    a point -- a dict ``{name: value}`` or a ``(ndim,)`` array -- or ``(point, scale)``
        A ball around it, ``scale`` (default ``DEFAULT_BALL_SCALE`` = 1e-3) times each prior's
        width in its own coordinate.
    a ``(nwalkers, ndim)`` array, or ``(n_ensembles, nwalkers, ndim)``
        The starts, the same for every ensemble or one set per ensemble.
    a ``SamplerResult``
        A previous fit (ABC, an earlier cut of the alert): each ensemble starts on its own random
        subset of its usable draws, or with too few, in a ball around its likelihood maximum
        (``info["init_detail"]["mode"]``: "draws" or "ball").

    ``info["init_detail"]`` is the first ensemble's, with every ensemble's under
    ``"per_ensemble"``. ``converged`` requires no stuck walker (median log-posterior more than 10
    nats below the best walker's, over all ensembles), no frozen walker, ``nsteps >= 50 x`` the
    largest autocorrelation time, and an R-hat between the ensembles below 1.01 on every
    parameter; ``info["convergence_problems"]`` names each failed check, and a non-empty list
    warns. The fit also warns before sampling when float32 cannot resolve ``lc.time`` or a Uniform
    prior bound.

    Examples
    --------
    >>> import numpy as np, whisper_cbpf as wp
    >>> from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30,
    ...                    flux=wp.get_model("flare_jax").predict(truth, t),
    ...                    flux_err=np.full(30, 0.1))
    >>> r = fit_emcee_jax(lc, "flare_jax", nsteps=2000, burnin=1000, seed=0)
    >>> r.info["n_ensembles"], r.info["walkers_per_ensemble"], r.samples_by_chain.shape[:2]
    (4, 32, (128, 100))
    >>> round(r.summary["t0"]["median"])
    10

    An ABC fit handed over (each ensemble starts on its own subset of ABC's draws):

    >>> abc = wp.fit(lc, "flare_jax", sampler="abc", n_simulations=5000, quantile=0.01)
    >>> fit_emcee_jax(lc, "flare_jax", init=abc, nsteps=1000, burnin=300).info["init"]
    'result'

    Until the checks pass, at most 20 000 steps:

    >>> r = fit_emcee_jax(lc, "flare_jax", nsteps=2000, burnin=1000, until_converged=True,
    ...                   max_steps=20000)
    >>> r.info["until_converged"]["stop_reason"] in ("converged", "max_steps", "max_time_s")
    True
    """
    t_call = time.perf_counter()
    check_burnin(burnin, nsteps)
    nwalkers, nsteps, burnin, thin = int(nwalkers), int(nsteps), int(burnin), int(thin)
    k = int(n_ensembles)
    if k < 1:
        raise ValueError(f"emcee_jax: n_ensembles must be at least 1; got {n_ensembles!r}.")
    if thin < 1:
        raise ValueError(f"emcee_jax: thin must be >= 1; got {thin}.")
    n_keep = (nsteps - burnin) // thin
    if n_keep < 1:
        raise ValueError(f"emcee_jax: thin={thin} keeps no draw of the {nsteps - burnin} steps "
                         f"after burnin={burnin}. Lower thin or raise nsteps.")
    tail = (nsteps - burnin) - n_keep * thin
    plan = pilot_plan(pilot, pilot_steps, k, nsteps)
    until = (until_plan(nsteps, burnin, thin, check_every, max_steps, max_time_s)
             if until_converged else None)
    chunk_info, width = _walker_width(walker_chunk, nwalkers)

    import jax
    import jax.numpy as jnp

    from . import _diagnostics as _dg
    from ._adapters import float_dtype, log_density
    from ._adapters import free_density as _free_density
    from ._adapters import resolve_density as _resolve_density
    from ._adapters import resolve_sampling_names as _resolve_sampling_names
    from .batch import _kept_chain_health, _result, _run_ensembles

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
    if nwalkers < 2 * ndim:
        raise ValueError(f"emcee_jax: nwalkers={nwalkers} is fewer than twice the {ndim} sampled "
                         f"parameters; the stretch move needs at least {2 * ndim} per ensemble.")
    dt = float_dtype()

    auto = (log_prob_fn is None and getattr(model, "log_prob_jax", None) is None
            and getattr(model, "predict_jax", None) is not None)
    if auto:
        # The fit_batch density: the data are arguments of one compiled program.
        with _zero_flux_given():        # given once above, by pre_event.warn()
            ld = log_density(lc, model, space=space, likelihood=likelihood, prior=full_prior)
        if list(ld.names) != list(names):
            raise RuntimeError(f"emcee_jax: the density samples {ld.names}, expected {names}.")
        program, one_data = ld.shared, ld.data
        space, space_source, fn_source = ld.space, "density", "auto"
        like_name = ld.likelihood
        scatter = next((nm for nm in ld.names if nm not in model.parameters), None)
        ll_is, n_data = "likelihood", int(ld.n_data)
        program_info = ("data as arguments" if ld.data_as_argument else
                        "epochs closed over: " + ld.reason)
    else:
        fn, space, space_source, fn_source = _resolve_density(
            lc, model, full_prior, log_prob_fn, space, all_names,
            include_prior=True, sampler="emcee_jax", likelihood=likelihood)
        fn, _, _ = _free_density(fn, full_prior, all_names)
        loglik_fn = getattr(fn, "log_likelihood", None)

        def program(theta, d):
            lp = fn(theta)
            # AIC/BIC rank draws by the likelihood twin when there is one; a hand-built density
            # without one reports what was sampled (info["max_log_likelihood_is"]).
            return lp, (loglik_fn(theta) if loglik_fn is not None else lp)

        one_data = {"unused": jnp.zeros((1,), dt)}
        like_name = getattr(fn, "likelihood", None)
        scatter = getattr(fn, "scatter_param", None)
        ll_is = "likelihood" if loglik_fn is not None else "sampled density"
        n_data = int(len(lc.time))
        program_info = "a closure over this light curve"

    # The walkers move in each prior's own coordinate (log for a LogUniform).
    is_log = _dg.walker_log_columns(prior, names, walker_coordinates, "emcee_jax")
    f32_hazard = _dg.float32_hazard(lc.time, prior, names, "emcee_jax",
                                    bool(jax.config.jax_enable_x64))
    seeds = ensemble_seeds(seed, k)
    mode, inits = _ensemble_inits(init, prior, names, nwalkers, k, seeds)
    data = jax.tree_util.tree_map(lambda x: jnp.stack([x] * k), one_data)
    who = [f"ensemble {e}" for e in range(k)]
    init_label = "box" if mode == "box" else ("explicit" if mode == "explicit" else mode)
    chunks, stop = [], {"reason": None}
    if until is not None:
        last_full = {"draws": 0}

        def keep_going(xs, lps, lls, main_steps, call_s):
            """Check the draws kept so far; the next chunk's kept states, or 0 to stop."""
            t_k = time.perf_counter()
            cw = np.concatenate(xs).astype(float).reshape(-1, k * nwalkers, ndim)
            lp = np.concatenate(lps).astype(float).reshape(-1, k * nwalkers)
            n_draws = cw.shape[0]

            def check(full):
                return _convergence_check(cw, lp, is_log, names, k, thin, main_steps,
                                          init_label, full)

            full = n_draws >= (1.0 + FULL_CHECK_GROWTH) * last_full["draws"]
            ok, stage, rec = check(full)
            step_s = call_s / max(1, xs[-1].shape[0] * thin)
            if not ok:
                if until["max_steps"] is not None and \
                        main_steps + until["check_every"] > until["max_steps"]:
                    stop["reason"] = "max_steps"
                elif until["max_time_s"] is not None and (
                        time.perf_counter() - t_call + step_s * until["check_every"]
                        + (time.perf_counter() - t_k) > until["max_time_s"]):
                    stop["reason"] = "max_time_s"
                if stop["reason"] and stage == 1 and not full and not rec["failed_checks"]:
                    ok, stage, rec = check(True)             # the last chunk: check it all
                    full = True
            if full and stage > 1:
                last_full["draws"] = n_draws
            if ok:
                stop["reason"] = "converged"
            rec.update(call_s=float(call_s), check_s=float(time.perf_counter() - t_k),
                       elapsed_s=float(time.perf_counter() - t_call))
            chunks.append(rec)
            return 0 if stop["reason"] else until["every_keep"]
    g = _run_ensembles(program, data, [one_data] * k, [prior] * k, names, seeds, inits,
                       "explicit" if mode in ("box", "explicit") else mode, is_log, nwalkers,
                       burnin, max(n_keep, until["every_keep"]) if until else n_keep, thin, tail,
                       dt, "emcee_jax", who, width=width, pilot=plan,
                       **({"until": keep_going, "first_keep": n_keep} if until else {}))
    labels = ["box"] * k if mode == "box" else g["labels"]
    runtime = g["start_s"] + g["run_s"]
    nsteps_requested = nsteps
    if until is not None:
        nsteps = int(g["main_steps"])
        n_keep = int(np.asarray(g["xs"]).shape[0])

    t_post = time.perf_counter()
    n_w = k * nwalkers
    chain_w = np.asarray(g["xs"], dtype=float).reshape(n_keep, n_w, ndim)  # walker coordinates
    lp_c = np.asarray(g["lps"], dtype=float).reshape(n_keep, n_w)
    ll_c = np.asarray(g["lls"], dtype=float).reshape(n_keep, n_w)
    chain = _dg.from_walker(chain_w, is_log)
    taus, taus_by, health = _kept_chain_health(chain_w, lp_c, thin, nsteps, n_ensembles=k)
    ens = {}
    if k > 1:
        ens = _ensemble_rhat(chain, k, names, ll_c)
        worst = {nm: v for nm, v in ens["rhat"].items()
                 if not (np.isfinite(v) and v < _dg.RHAT_MAX)}
        if worst:
            shown = ", ".join(f"{nm} {v:.3g}" for nm, v in list(worst.items())[:5])
            health["convergence_problems"].append(
                f"the {k} ensembles disagree: R-hat between them {shown} (needs < "
                f"{_dg.RHAT_MAX:g}). Run longer, or check for a second mode")
            health["converged"] = False
    pilot_info = None
    if plan:
        p = g["pilot"]
        pll = np.asarray(p["lls"], dtype=float)                     # (keep, K, W)
        best_each = [float(v) for v in np.max(pll, axis=(0, 2))]
        top = max(best_each)
        p_chain = _dg.from_walker(np.asarray(p["xs"], dtype=float).reshape(plan["keep"], n_w, ndim),
                                  is_log)
        prh = (_ensemble_rhat(p_chain, k, names, pll.reshape(plan["keep"], n_w), ess=False)
               if k > 1 else {"rhat": {}, "rhat_log_likelihood": float("nan")})
        pilot_info = {**plan, "best_ensemble": int(p["best"]),
                      "max_log_likelihood_each": best_each,
                      "gap_each": [top - v for v in best_each],
                      "n_ensembles_dropped": int(sum(top - v > _dg.STRANDED_NATS
                                                     for v in best_each)),
                      # informational: every ensemble, the best one included, restarts from the
                      # best ensemble's kept states, so the main-run ensembles are not
                      # independent runs
                      "restarted_ensembles": list(range(k)),
                      "independent": False,
                      "mean_acceptance_fraction": float(np.mean(p["n_acc"]) / plan["steps"]),
                      "rhat": prh["rhat"], "rhat_log_likelihood": prh["rhat_log_likelihood"],
                      "restart": "each ensemble: its own random draws, without replacement, of "
                                 "the best pilot ensemble's kept states"}
    _dg.warn_if_unconverged("emcee_jax", health)
    n_acc = np.asarray(g["n_acc"]).reshape(-1)
    details = g["details"]
    info = {
        "nwalkers": n_w, "walkers_per_ensemble": nwalkers, "n_ensembles": k,
        "nsteps": nsteps, "burnin": burnin, "thin": thin,
        "space": space, "seed": int(seed), "ensemble_seeds": seeds,
        "init": labels[0],
        "init_detail": (dict(details[0]) if k == 1 else
                        {**details[0], "per_ensemble": list(details)}),
        "init_time_s": float(g["start_s"]),
        "walker_coordinates": {"coordinates": walker_coordinates,
                               "log": [nm for nm, lg in zip(names, is_log) if lg]},
        "float32_hazard": f32_hazard,
        "space_source": space_source, "log_prob_fn": fn_source,
        # `type(likelihood).__name__` -- the key `samplers.base._LIKELIHOOD_KINDS` maps back to a
        # registry name so `waic`/`predictive_metrics` re-score under the density that was fitted.
        # None when the caller supplied their own log_prob_fn: we do not know then.
        "likelihood": like_name, "scatter_param": scatter,
        "x64": bool(jax.config.jax_enable_x64),
        "fixed": dict(fixed),
        "walker_chunk": chunk_info,
        "max_log_likelihood_is": ll_is,
        "mean_acceptance_fraction": float(np.mean(n_acc) / nsteps),
        "acceptance_fraction": [float(v) for v in n_acc / nsteps],
        "mean_autocorr_time": (float(np.nanmean(taus)) if np.isfinite(taus).any()
                               else float("nan")),
        "autocorr_time": [float(v) for v in taus],
        "autocorr_time_by_ensemble": [[float(v) for v in t] for t in taus_by],
        "autocorr_from": f"kept chain (post-burn-in, thinned by {thin}) x {thin}, per ensemble",
        "n_nan_proposals": int(np.sum(g["n_nan"])),
        "ensemble_rhat": ens.get("rhat", {}),
        "ensemble_ess_bulk": ens.get("ess_bulk", {}),
        "ensemble_ess_tail": ens.get("ess_tail", {}),
        "ensemble_rhat_log_likelihood": ens.get("rhat_log_likelihood", float("nan")),
        "ensemble_rhat_method": ens.get("rhat_method"),
        "pilot": pilot_info,
        "until_converged": (
            {"enabled": False, "steps_run": nsteps, "stop_reason": "nsteps"} if until is None else
            {"enabled": True, "steps_run": nsteps, "nsteps_requested": nsteps_requested,
             "stop_reason": stop["reason"], "n_chunks": len(chunks),
             "check_every": until["check_every"], "max_steps": until["max_steps"],
             "max_time_s": until["max_time_s"], "chunks": chunks,
             "compiled_calls": len(g["calls"]), "calls": g["calls"]}),
        "program": program_info,
        **health,
        "backend": (f"emcee stretch move on the device (one compiled lax loop): {k} ensembles x "
                    f"{nwalkers} walkers"),
        "device_timing": {"compile_s": float(g["compile_s"]), "start_s": float(g["start_s"]),
                          "run_s": float(g["run_s"])},
        "compile_time_s": float(g["compile_s"]),
    }
    shim = types.SimpleNamespace(names=list(names), fixed=dict(fixed), n_data=n_data, space=space)
    result = _result(lc, model, shim, chain, lp_c, ll_c, info, runtime, True, sampler="emcee_jax",
                     all_names=all_names)
    pre_event.stamp(result)
    result.info["postprocess_s"] = float(time.perf_counter() - t_post)
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
    """whisper registry adapter for :func:`fit_emcee_jax` (``sampler="emcee_jax"``).

    The function form is kept as the public entry point. This class is the thin ``BaseSampler``
    shell the registry needs: ``get_sampler(name)`` constructs with zero arguments, so every option
    travels through ``fit``. Registered because the sampler is model-agnostic: it takes any JAX
    model, or a ``log_prob_fn``. ``fit_emcee_numpy`` in this module is not registered: it
    hard-wires the flare spec.

    Examples
    --------
    >>> import numpy as np, whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30,
    ...                    flux=wp.get_model("flare_jax").predict(truth, t),
    ...                    flux_err=np.full(30, 0.1))
    >>> r = wp.fit(lc, "flare_jax", sampler="emcee_jax", nsteps=2000, burnin=1000)
    >>> r.sampler, r.info["n_ensembles"]
    ('emcee_jax', 4)
    """

    name = "emcee_jax"

    def fit(self, lc, model, prior=None, *, log_prob_fn=None, init="prior_scan",
            walker_coordinates="own", **kwargs) -> SamplerResult:
        """Fit with emcee's stretch move on the device. See :func:`fit_emcee_jax`.

        ``log_prob_fn`` is optional: omit it and the density is built from ``model.predict_jax``
        and the likelihood ``likelihood=`` / ``space=`` imply. If you do pass one, pass a
        log-**POSTERIOR**: the sampler has no prior mechanism of its own, so a pure log-likelihood
        silently imposes a flat box (``nuts_gpu``, by contrast, wants a pure likelihood). ``init``
        says where the walkers start, ``walker_coordinates`` where they move, and ``kwargs`` are
        :func:`fit_emcee_jax`'s (``nwalkers``, ``nsteps``, ``n_ensembles``, ``pilot``, ...).

        Returns
        -------
        SamplerResult
            As :func:`fit_emcee_jax`.

        Examples
        --------
        >>> import numpy as np, whisper_cbpf as wp
        >>> from whisper_cbpf.samplers.jax.emcee_jax import EmceeJAXSampler
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30,
        ...                    flux=wp.get_model("flare_jax").predict(truth, t),
        ...                    flux_err=np.full(30, 0.1))
        >>> r = EmceeJAXSampler().fit(lc, "flare_jax", nsteps=1000, burnin=500, n_ensembles=2)
        >>> r.info["nwalkers"]
        64
        """
        return fit_emcee_jax(lc, model, log_prob_fn, prior=prior, init=init,
                             walker_coordinates=walker_coordinates, **kwargs)
