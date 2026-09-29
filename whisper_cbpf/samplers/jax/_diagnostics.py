"""Chain health and chain starts for the likelihood samplers, in one place so they cannot drift apart.

Used by ``nuts_gpu``, ``pymc_jax_gpu_*``, ``emcee_jax``, the CPU ``mcmc`` sampler and
:mod:`whisper_cbpf.metrics._jax` (ESS). numpy only at module scope -- arviz and numpyro are imported
lazily -- so the CPU sampler can use it on an install without the ``[gpu]`` extra.

WHY THIS EXISTS
---------------
Every NUTS run on arviz >= 1.0 read ``rhat={}``, ``max_rhat=None`` and ``converged=False``, good or
bad: ``nuts_gpu`` called ``az.convert_to_inference_data``, which arviz 1.x removed, inside a bare
``except Exception: pass``. So the flag users are told to check carried no information. Behind it
sat real failures that a working R-hat would only partly have caught. A mock-problem matrix
(2 672 fits before this module existed) found, in float64
with a relative clock and each model's default prior, on CPU and GPU alike:

* **Stranded chains.** A Gaussian bump fitted by ``nuts_gpu``: 22 / 100 runs had 1-3 of 4 chains in
  a local optimum -- the model shrunk to its width floor and hidden between epochs, at the zero-flux
  model's likelihood -- so the pooled t0 was a median 18.6 sd off with an sd 32x too wide. The
  one-component kilonova: 7 / 13 runs with chains on a prior-box corner, 972-65 618 nats below the
  best chain. ``pymc_jax_gpu_vectorized`` failed the same way.
* **All chains wrong.** Two pymc bump runs had every chain in the wrong mode, with R-hat 1.002:
  no between-chain statistic can see that. Only an independent optimum can.
* **Frozen chains.** float32 with an MJD clock: t0 can only take values 0.0039 d apart, the
  potential becomes a staircase, and a chain adapts its step to ~1e-5 and stops (19 / 100 flare
  runs, 25 / 100 bump runs, mostly with zero divergences).
* **Stuck emcee walkers**, in 24-100 % of ``emcee_jax`` runs, while its autocorrelation-only rule
  said ``converged=True`` on 55 of 78 stuck bump runs.

SBC ranks and coverage stayed nominal in all of these cells -- a stranded chain widens the pooled
intervals, so the truth still lands inside them. What exposes them is per-run evidence, which is
what this module computes, and where the chains start, which is what prevents most of them.

WHERE THE CHAINS START (``scan_starts``: ``init_strategy="prior_scan"``, ``init="prior_scan"``)
--------------------------------------------------------------------------------------------
Score 1000 prior draws in one device program; climb the best 32 a short way uphill (never
downhill), which tells which basin each belongs to; keep those that end within 10 nats of the best,
farthest-first; start each chain ~2 conditional posterior sd from one of them. A time prior 50x
wider than the data (t0 ~ U(-1000, 1000), plan mechanism 2) stranded every chain of the old start
in 4 / 4 seeds and none of this one; when most of the prior is such a zero-signal plateau the scan
also warns. The same mocks, seeds and budgets as the matrix, re-run on this code: broken runs
22 -> 0 of 100 (bump, ``nuts_gpu``, float64), 50 -> 0 (bump, float32 MJD clock), 29 -> 1 (flare,
float32 MJD: one frozen chain, flagged), 12 -> 0 (bump, ``pymc_jax_gpu_vectorized``), 78 -> 0 and
57 -> 0 (bump and flare, ``emcee_jax``), 7 -> 0 of 13 (kilonova, CPU), 3 -> 0 of 9 (arnett
supernova). With the OLD start and these diagnostics, every broken run was flagged: 22 / 22,
50 / 50, 4 / 4 and 29 / 29 (``nuts_gpu``), 12 / 12 (pymc) and 78 / 78, 57 / 57 (``emcee_jax``).

The other ``init=`` forms every MCMC-type sampler shares -- a point (a ball), one start per chain,
and a previous result, the ABC -> MCMC handoff and the staging of an alert's fits -- resolve in
:func:`explicit_starts`. A density with no JAX gradient (CPU ``mcmc``, ``fit_emcee_numpy``) gets
the same scan, climbed by the derivative-free :func:`climb_numpy`: on redback ``arnett`` /
SN2025pgp, 60 walkers x 10 000 steps, stuck walkers 20 of 60 from independent prior draws -> 0,
for a 26-29 s start.

WHAT ``converged`` MEANS NOW (NUTS samplers)
-------------------------------------------
No divergence, rank-normalised R-hat < 1.01 on every parameter AND on the log-likelihood, bulk and
tail ESS >= 100 per chain (Vehtari et al. 2021, arXiv:1903.08008), no stranded chain, no frozen
chain, and no independent optimum beating every chain. Each failed condition is written, as a
sentence, into ``info["convergence_problems"]`` and repeated in one warning.
"""
from __future__ import annotations

import numpy as np

from ...priors._numpy import family


def _warn(message, category=UserWarning):
    """Warn at the user's own line (see :func:`whisper_cbpf.samplers.base._warn_user`)."""
    from ..base import _warn_user
    _warn_user(message, category)


#: Rank-normalised R-hat must be below this on every parameter and on the log-likelihood. Classical
#: split R-hat already reaches 1.010-1.012 on healthy 4 x 500 runs, which is why the rank-normalised
#: statistic is used (Vehtari et al. 2021).
RHAT_MAX = 1.01
#: Bulk and tail ESS must reach this many per chain (Vehtari et al. 2021: "at least 100 per chain").
ESS_PER_CHAIN_MIN = 100
#: A chain (or walker) whose median log-likelihood is more than this many nats below the best
#: chain's is stranded. A healthy chain's median sits ~D/2 below the peak, the same for every
#: chain; a gap of 10 nats is a region carrying e^-10 of the posterior mass. The Step 0 matrix used
#: the same rule, and every stranded chain it found sat 11-65 618 nats below.
STRANDED_NATS = 10.0
#: A chain is frozen when its adapted step is below this fraction of the median chain's AND it
#: barely moves (its own sd of some parameter < ``FROZEN_SD_RATIO`` of the pooled sd). The step
#: alone over-flags: flare sim 53 adapted a step of 2e-4 against 0.2 and still mixed (chain sd 0.96
#: of pooled). 1e-3 missed a stranded chain at 1.2e-3; 1e-2 did not.
FROZEN_STEP_RATIO = 1e-2
FROZEN_SD_RATIO = 0.1
#: Prior draws scored before sampling: they choose where the chains start (``"prior_scan"``) and
#: give the independent optimum every run is checked against. Scored in blocks of
#: :data:`SCORE_BLOCK` (:func:`block_scorer`).
N_PRIOR_SCAN = 1000
#: Draws per vmapped block wherever one density is scored at many draws: the prior scan, and the
#: post-fit re-scan of every kept draw by ``nuts_gpu``, ``pymc_jax_gpu_*`` and ``emcee_jax``
#: (:func:`block_scorer`). Measured on one A6000, float64, fresh process per width
#: One draw per step, as the samplers did: 51.3 ms a
#: draw for the TDE at ``n_time=5000``, 5.2 ms at 500, 0.17 ms for a kilonova or an arnett
#: supernova -- the TDE pays its ~20 ms of ODE latency on every draw. In blocks of 250: 0.10, 0.012,
#: 0.006-0.010 and 0.007-0.014 ms, for a compile of 17 s, 6 s and 1.5-2.4 s. Not a power of two:
#: XLA's GPU compile time is erratic in the width, and the kilonovae take 6-13 s at 128, 21-54 s at
#: 256 and more than 300 s at 1024 (killed), against 1.5-2.8 s at 100, 250, 500 and 1000. On the CPU
#: there is no such spike and blocks still help, 0.46-0.74 -> 0.15-0.33 ms a draw.
SCORE_BLOCK = 250
#: The best-scoring scan draws are then climbed by a short gradient ascent (``REFINE_STEPS`` Adam
#: steps in the logit of each prior's own coordinate), which tells which basin each one belongs to.
#: The raw score cannot: on one-component kilonova sim 3 the 4 best draws scored 18.9, -18.3, -99
#: and -102, and climbed to 38.9, 38.9, 38.9 and 1.1 -- the 4th start was the chain that stranded,
#: 38 nats below. The whole start (scan, climb, curvature) took, compile included, on the CPU
#: 1-2 s for the bump, 7-13 s for a kilonova or an arnett supernova and 96 s for the TDE at
#: ``n_time=500``; on one A6000 7-12 s for those, 99 s for the TDE at 500 and 150 s at 5000
#: Most of the TDE's was
#: compiling the curvature, ``jax.hessian`` through the ODE: 76 s of the CPU's 96 and 57 s of the
#: GPU's 99. The curvature is now a second difference through the scan's compiled scorer
#: (``_conditional_sd``): on one A6000, at the same climbed centres, 9.4 s -> 0.02 s (arnett),
#: 17 s -> 0.03 s (two-component kilonova), 110 s -> 0.02 s (TDE 500, a loaded machine), with the
#: same sd as the Hessian's to within 2x on 97-100 % of the coordinates at 4 starts and 83-98 % at
#: 60 (where it more often reads a coordinate as flat: 59 against 26 of the TDE's 300, capped and
#: then bounded by the start's cost limit). A whole ``emcee_jax`` TDE-500 fit, 60 x 300, one
#: A6000, the two back to back at load 230-360: ``init_time_s`` 96-97 s against 219-361 s, max
#: log-likelihood -89.70 against -90.48. What is
#: left is the scan and, mostly, the climb's compile; the start still dominates such a short run.
N_REFINE = 32
REFINE_STEPS = 200
#: Starts climbed together, as one vmapped block: the climb's 200
#: steps are sequential, so one start at a time made a latency-bound model pay its per-call cost
#: 32 x 200 times. Measured on one A6000 (compile included), one start at a time against blocks of
#: 32: the TDE at ``n_time=500`` 198 s against 46 s, at 5000 ~30 min (extrapolated) against 92 s;
#: a kilonova 7.1 against 9.0 s, an arnett supernova 5.2 against 5.1 s. On the CPU 69 against 16 s
#: (TDE), 17 against 9.5 s (kilonova), 3.7 against 5.8 s (arnett). The climbed points agree to
#: ~1e-13 relative on the smooth models; on the TDE a few differ by up to 0.8 % (its envelope has
#: kinks, so a last-bit difference can flip one accept/reject step), the best point to 1e-12.
CLIMB_BLOCK = 32
#: Starts are the climbed candidates that end within ``STRANDED_NATS`` of the best one, taken
#: farthest-first (cycled when fewer than the chains), each moved ``START_SPREAD`` conditional
#: posterior sd at random -- the sd read off the curvature there, capped at ``MAX_SPREAD_U`` logit
#: units along a flat direction. Without the spread every chain started within ~0.01 sd of the same
#: maximum (bump sims 0 and 13), which leaves R-hat nothing to compare; with it the starts are
#: overdispersed but in the basin.
START_SPREAD = 2.0
MAX_SPREAD_U = 1.0
#: Rounds of that spread: a refused move is retried in a fresh direction and its mirror image at
#: half the length, down to 2^-19 of the first. Six rounds of one fixed direction, halved, left
#: every walker of a magnetar and a TDE injection on one point (their optimum is pressed against
#: a constraint wall), and emcee refused to start.
SPREAD_ROUNDS = 20
#: The curvature is a second difference through the scan's compiled scorer (``_conditional_sd``),
#: first taken ``FD_STEP0`` logit units either side, then ``FD_ROUNDS - 1`` more times at the sd
#: the last one implied: 2 k probes per start per round.
FD_STEP0 = 0.1
FD_ROUNDS = 3
#: Warn when at least this fraction of the scan's draws score the same log-density (within 1 nat of
#: the median) and some draw does not: the prior is mostly a zero-signal plateau, typically a time
#: prior much wider than the data window (plan U3 (iv)). A 3 d bump seen over [-20, 20] d with
#: t0 ~ U(-L, L): 0.68 at L = 100, 0.97 at L = 1000, 0.997 at L = 10000, where 1 of 4 seeds had no
#: draw near the event and its chains wandered the plateau. The Step 0 mocks' default priors reach
#: at most 0.004 (bump), 0.08 (flare) and 0.0 (kilonova).
PLATEAU_FRACTION = 0.9
#: float32 spacing at 1e3 is 6.1e-5; at MJD 6e4 it is 3.9e-3 d, which made the t0 potential a
#: staircase with 6.3-nat steps. Keyed on the absolute value, not on bound/width: U(50000, 70000)
#: has a small ratio and a 0.0078 d quantum.
F32_ABS_LIMIT = 1.0e3


# ------------------------------------------------------------------------------------------ R-hat
def _arviz_rank(x):
    """Rank-normalised split R-hat, bulk ESS and tail ESS of a (chains, draws) array.

    The ndarray path of ``az.rhat`` / ``az.ess`` exists, with the same meaning, in arviz 0.x and in
    arviz >= 1.0 (checked on 1.3.0); the ``InferenceData`` converters around it do not
    (``convert_to_inference_data`` is gone in 1.x, ``from_dict`` changed its signature). The tail
    ESS gets its quantiles explicitly: 0.x defaulted them to (0.05, 0.95), while 1.3's ndarray
    path has no default and raised ``TypeError: _ess_tail() missing ... 'prob'``.
    """
    import arviz as az
    return (float(az.rhat(x)), float(az.ess(x, method="bulk")),
            float(az.ess(x, method="tail", prob=(0.05, 0.95))))


def _numpyro_split(x):
    """Fallback: numpyro's classical split R-hat and ESS. Not rank-normalised; no tail ESS."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    return float(split_gelman_rubin(x)), float(effective_sample_size(x)), float("nan")


def _finite_or_floor(x):
    """Non-finite entries (a -inf log-likelihood) -> just below the finite minimum, so they rank
    last instead of turning the statistic into NaN."""
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x)
    if bad.any():
        good = x[~bad]
        x = np.where(bad, (good.min() - 1.0e3) if good.size else 0.0, x)
    return x


def rank_diagnostics(samples_by_chain, names, ll_by_chain=None):
    """Per-parameter R-hat, bulk ESS and tail ESS, plus the R-hat of the log-likelihood.

    ``samples_by_chain`` is ``(chains, draws, k)``; ``ll_by_chain`` is ``(chains, draws)``. Returns a
    dict with ``rhat``, ``ess_bulk``, ``ess_tail`` (``{name: float}``), ``rhat_log_likelihood``,
    ``rhat_method`` and ``rhat_error``. arviz is tried first; if it is missing or raises, numpyro's
    split R-hat is used and ``rhat_error`` says why -- never a silent pass. NaN where neither can
    compute a value.
    """
    x = np.asarray(samples_by_chain, dtype=float)
    columns = [x[:, :, j] for j in range(x.shape[2])]
    if ll_by_chain is not None:
        ll = _finite_or_floor(ll_by_chain)
        columns.append(ll)
    method, error = None, None
    try:
        import arviz as az
        stats = [_arviz_rank(c) for c in columns]
        method = f"arviz {az.__version__}: rank-normalised split R-hat, bulk/tail ESS"
    except Exception as exc:                        # noqa: BLE001 - recorded, then the fallback
        error = f"arviz: {type(exc).__name__}: {exc}"
        try:
            stats = [_numpyro_split(c) for c in columns]
            method = "numpyro split_gelman_rubin (classical, not rank-normalised; no tail ESS)"
        except Exception as exc2:                   # noqa: BLE001 - both failed, say so
            error += f"; numpyro: {type(exc2).__name__}: {exc2}"
            stats = [(float("nan"),) * 3 for _ in columns]
            method = "unavailable"
    out = {"rhat": {}, "ess_bulk": {}, "ess_tail": {}}
    for nm, (r, eb, et) in zip(names, stats):
        out["rhat"][nm], out["ess_bulk"][nm], out["ess_tail"][nm] = r, eb, et
    rll = float("nan")
    if ll_by_chain is not None:
        # A log-likelihood that is identical in every draw (a flat likelihood) cannot differ between
        # chains; its R-hat is 0/0. Report "no between-chain difference" rather than a NaN failure.
        rll = 1.0 if np.ptp(ll) == 0.0 else stats[-1][0]
    out["rhat_log_likelihood"] = rll
    out["rhat_method"], out["rhat_error"] = method, error
    return out


# ----------------------------------------------------------------------------------- chain health
def _fmt_chains(idx):
    return ", ".join(str(int(i)) for i in idx)


def chain_health(samples_by_chain, names, ll_by_chain, *, n_divergences,
                 step_size_by_chain=None, reference_ll=None,
                 reference_label="the best point the prior scan found"):
    """Everything ``info`` needs to say whether a multi-chain NUTS posterior can be used.

    Parameters
    ----------
    samples_by_chain : (chains, draws, k) array, in parameter units.
    ll_by_chain : (chains, draws) array
        Log-likelihood of every draw -- the samplers compute it anyway for the best draw.
    n_divergences : int
        ``-1`` means "not recorded", which fails closed.
    step_size_by_chain : sequence, optional
        Adapted step size per chain, for the frozen-chain check.
    reference_ll : float, optional
        An independent optimum (the best log-likelihood the prior scan and its climb found). When
        every chain's best draw is more than ``STRANDED_NATS`` below it, all chains missed the
        best region -- the case R-hat is blind to, because nothing differs *between* chains.

    Returns the ``info`` entries: ``rhat``, ``max_rhat`` (float, NaN when unknown -- never None),
    ``ess_bulk``, ``ess_tail``, ``min_ess``, ``rhat_log_likelihood``, ``rhat_method``,
    ``rhat_error``, ``chain_median_log_likelihood``, ``stranded_chains``, ``frozen_chains``,
    ``step_size_by_chain``, ``convergence_problems`` and ``converged``.
    """
    x = np.asarray(samples_by_chain, dtype=float)
    ll = np.asarray(ll_by_chain, dtype=float)
    n_chains = int(x.shape[0])
    d = rank_diagnostics(x, names, ll)
    problems = []

    if n_divergences < 0:
        problems.append("divergences were not recorded, so they cannot be ruled out")
    elif n_divergences > 0:
        problems.append(f"{int(n_divergences)} divergent transition(s): the sampler could not "
                        f"follow the posterior's curvature somewhere, so the draws may be biased")

    finite = {nm: v for nm, v in d["rhat"].items() if np.isfinite(v)}
    max_rhat = float(max(finite.values())) if finite else float("nan")
    undefined = [nm for nm, v in d["rhat"].items() if not np.isfinite(v)]
    rll = d["rhat_log_likelihood"]
    if n_chains < 2:
        # arviz 1.x returns NaN for one chain; either way there is nothing to compare it with.
        problems.append("a single chain: R-hat and the stranded-chain check need at least 2 "
                        "chains (4 recommended), so convergence cannot be established")
    elif undefined:
        problems.append(f"R-hat could not be computed for {undefined} ({d['rhat_method']}; "
                        f"{d['rhat_error'] or 'every draw identical?'})")
    if finite and max_rhat >= RHAT_MAX:
        worst = max(finite, key=finite.get)
        problems.append(f"R-hat {max_rhat:.3f} on {worst!r} (must be < {RHAT_MAX}): the chains "
                        f"disagree about where the posterior is")
    if n_chains >= 2 and not np.isfinite(rll):
        problems.append("the log-likelihood R-hat could not be computed")
    elif rll >= RHAT_MAX:
        problems.append(f"log-likelihood R-hat {rll:.3f} (must be < {RHAT_MAX}): the chains sit "
                        f"at different likelihood levels")
    ess = [v for v in list(d["ess_bulk"].values()) + list(d["ess_tail"].values())
           if np.isfinite(v)]
    min_ess = float(min(ess)) if ess else float("nan")
    need = ESS_PER_CHAIN_MIN * n_chains
    if not np.isfinite(min_ess) or min_ess < need:
        problems.append(f"smallest bulk/tail ESS {min_ess:.0f} < {need} ({ESS_PER_CHAIN_MIN} per "
                        f"chain): too few effective draws for R-hat and intervals to be reliable")

    # Stranded: median log-likelihood far below the best chain's, or best draw far below an
    # independent optimum. -inf (outside the box) ranks as the lowest possible level.
    med = np.array([np.median(np.where(np.isfinite(c), c, -np.inf)) for c in ll])
    top = np.array([np.max(np.where(np.isfinite(c), c, -np.inf)) for c in ll])
    best_chain = int(np.argmax(med))
    gap = med[best_chain] - med
    stranded = set(np.flatnonzero(gap > STRANDED_NATS).tolist())
    if stranded:
        problems.append(
            f"chain(s) {_fmt_chains(sorted(stranded))} stranded: median log-likelihood "
            f"{', '.join(f'{gap[c]:.1f}' for c in sorted(stranded))} nats below chain "
            f"{best_chain}'s. They sit in a local optimum and their draws are pooled into the "
            f"posterior")
    if reference_ll is not None and np.isfinite(reference_ll):
        below = top < reference_ll - STRANDED_NATS
        if below.all():
            problems.append(
                f"every chain's best draw is at least {reference_ll - top.max():.1f} nats below "
                f"{reference_label} (log-likelihood {reference_ll:.1f}): all chains missed the "
                f"best region, which R-hat cannot see")
        elif below.any():
            extra = sorted(set(np.flatnonzero(below).tolist()) - stranded)
            if extra:
                problems.append(f"chain(s) {_fmt_chains(extra)} never came within "
                                f"{STRANDED_NATS:.0f} nats of {reference_label}")
        stranded |= set(np.flatnonzero(below).tolist())

    frozen = []
    steps = None
    if step_size_by_chain is not None:
        steps = np.atleast_1d(np.asarray(step_size_by_chain, dtype=float))
        if steps.size == n_chains and n_chains > 1:
            med_step = float(np.median(steps))
            pooled_sd = x.reshape(-1, x.shape[-1]).std(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_sd = x.std(axis=1) / pooled_sd[None, :]          # (chains, k)
            collapsed = np.nanmin(np.where(np.isfinite(rel_sd), rel_sd, np.inf), axis=1)
            frozen = [int(c) for c in range(n_chains)
                      if steps[c] < FROZEN_STEP_RATIO * med_step and collapsed[c] < FROZEN_SD_RATIO]
            if frozen:
                problems.append(
                    f"chain(s) {_fmt_chains(frozen)} frozen: adapted step "
                    f"{', '.join(f'{steps[c]:.1e}' for c in frozen)} against a median of "
                    f"{med_step:.1e}, and it barely moves. In float32 this is the signature of a "
                    f"quantised parameter or clock (MJD-scale times): enable x64 or shift the clock")

    return {
        "rhat": d["rhat"], "max_rhat": max_rhat,
        "ess_bulk": d["ess_bulk"], "ess_tail": d["ess_tail"], "min_ess": min_ess,
        "rhat_log_likelihood": float(rll),
        "rhat_method": d["rhat_method"], "rhat_error": d["rhat_error"],
        "chain_median_log_likelihood": [float(v) for v in med],
        "stranded_chains": sorted(int(c) for c in stranded),
        "frozen_chains": frozen,
        "step_size_by_chain": (None if steps is None else [float(s) for s in steps]),
        "reference_log_likelihood": (None if reference_ll is None else float(reference_ll)),
        "convergence_problems": problems,
        "converged": not problems,
    }


def walker_health(logp_by_walker, autocorr_times, nsteps):
    """Convergence for an emcee ensemble: stuck walkers and the LARGEST autocorrelation time.

    ``logp_by_walker`` is the sampled log-density after burn-in, ``(walkers, draws)``. A walker
    whose median is more than ``STRANDED_NATS`` below the best walker's is stuck: the stretch move
    builds its proposals from the other walkers, and a walker parked in a separate optimum sees
    them only across a valley, so it rejects forever. The autocorrelation rule used to be the only
    test and it passed stuck ensembles (``converged=True`` on 55 of 78 stuck bump runs); it also
    used the MEAN tau, so one slow parameter could hide behind fast ones. ``nsteps >= 50 * max(tau)``
    now, and an unknown tau is not a pass.
    """
    lp = np.asarray(logp_by_walker, dtype=float)
    med = np.array([np.median(np.where(np.isnan(w), -np.inf, w)) for w in lp])
    best = int(np.argmax(med))
    gap = med[best] - med
    stuck = np.flatnonzero(gap > STRANDED_NATS)
    tau = np.asarray(autocorr_times, dtype=float).ravel()
    max_tau = float(np.nanmax(tau)) if tau.size and np.isfinite(tau).any() else float("nan")
    problems = []
    if stuck.size:
        problems.append(
            f"{stuck.size} of {lp.shape[0]} walker(s) stuck (walkers {_fmt_chains(stuck[:10])}"
            f"{', ...' if stuck.size > 10 else ''}): median log-probability up to "
            f"{gap[stuck].max():.1f} nats below walker {best}'s. Their draws are pooled into the "
            f"posterior and widen it")
    if not np.isfinite(max_tau) or max_tau <= 0:
        problems.append("the autocorrelation time could not be estimated")
    elif nsteps < 50.0 * max_tau:
        problems.append(f"nsteps={int(nsteps)} < 50 x the largest autocorrelation time "
                        f"({max_tau:.0f}): the chain is too short to trust")
    return {"stuck_walkers": [int(w) for w in stuck],
            "walker_median_log_prob": [float(v) for v in med],
            "max_autocorr_time": max_tau,
            "convergence_problems": problems,
            "converged": not problems}


def warn_if_unconverged(sampler, health):
    """One warning naming every problem, so an unusable posterior is never silent."""
    if health["convergence_problems"]:
        _warn(f"{sampler}: this posterior is not converged -- "
                      + "; ".join(health["convergence_problems"])
                      + ". result.info['convergence_problems'] lists these.")


# ---------------------------------------------------------------------------------- chain starts
def prior_draws(prior, names, n, seed):
    """``(n, k)`` independent prior draws, in each prior's own coordinate (log for LogUniform).

    Through ``rescale`` (the prior's quantile function), so any whisper prior works. ``u`` stays a
    hair inside (0, 1) so no draw lands exactly on a bound, where the samplers' logit transforms
    are infinite.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=(int(n), len(names)))
    dists = [prior.distributions[nm] for nm in names]
    return np.array([[d.rescale(u[i, j]) for j, d in enumerate(dists)] for i in range(u.shape[0])],
                    dtype=float)


def block_scorer(log_prob_fn, dtype, width=None):
    """``score(theta) -> (n,)`` numpy: a scalar JAX density at every row of ``theta``, in blocks.

    One jitted ``vmap`` over exactly ``width`` rows (default :data:`SCORE_BLOCK`), called once per
    block, the last block padded with copies of the last row. So there is ONE compiled program, of
    one width, whatever the number of draws, and the returned function reuses it on every call: the
    prior scan, the start trials and the post-fit re-scan of every kept draw share one compile.

    It replaced ``jax.lax.map(log_prob_fn, theta)``, one draw per step, which pays a latency-bound
    model's fixed cost per draw (:data:`SCORE_BLOCK` has the numbers). Not
    ``lax.map(..., batch_size=width)`` either: that vmaps the remainder separately, a second compile
    at an arbitrary width (and compile time is erratic in the width), and a new draw count is a new
    program. Not one wide ``vmap``: its extent would be the draw count, the trap that once stalled
    ``nuts_gpu``'s re-scan for over an hour.
    """
    import jax
    import jax.numpy as jnp

    width = SCORE_BLOCK if width is None else int(width)
    batched = jax.jit(jax.vmap(log_prob_fn))

    def score(theta):
        theta = np.asarray(theta, dtype=float)
        n = theta.shape[0]
        pad = (-n) % width
        if pad:
            theta = np.concatenate([theta, np.repeat(theta[-1:], pad, axis=0)])
        # Sliced and joined on the host: a device-side slice per block would compile once per
        # offset, and a device-side concatenate once per block count. Every block is dispatched
        # before the first is fetched, so the device never waits on the host.
        out = [batched(jnp.asarray(theta[i:i + width], dtype=dtype))
               for i in range(0, theta.shape[0], width)]
        return np.concatenate([np.asarray(o, dtype=float) for o in jax.device_get(out)])[:n]

    return score


def prior_scan(log_prob_fn, prior, names, seed, dtype, n=N_PRIOR_SCAN, score=None):
    """Score ``n`` prior draws with a scalar JAX density -> ``(draws, log_density)``.

    Through :func:`block_scorer` (pass ``score`` to reuse one already compiled for this density),
    so the compile does not grow with ``n``.
    """
    draws = prior_draws(prior, names, n, seed)
    score = block_scorer(log_prob_fn, dtype) if score is None else score
    return draws, score(draws)


#: The climb's coordinate for each prior family. A box prior (Uniform, LogUniform) is walked in the
#: logit of its own coordinate (log10 for a LogUniform), clipped at +-7; a Normal in its
#: standardised value ``(x - mu) / sigma``, unbounded; a TruncatedNormal in the logit of its own
#: CDF (the probability integral transform the NUTS samplers sample it by), clipped at +-7.
_BOX_KINDS = ("Uniform", "LogUniform")


def _kind(prior, nm):
    return family(prior.distributions[nm])


def _site_box(prior, names):
    """Each parameter's own coordinate (log10 for a LogUniform) and its box there.

    Columns that are not a box prior get a dummy (0, 1): their coordinate is handled column by
    column in :func:`_to_logit` / :func:`_from_logit`.
    """
    is_log = np.array([_kind(prior, nm) == "LogUniform" for nm in names])
    box = np.array([_kind(prior, nm) in _BOX_KINDS for nm in names])
    lo = np.array([float(prior.distributions[nm].bounds[0]) if b else 0.0
                   for nm, b in zip(names, box)])
    hi = np.array([float(prior.distributions[nm].bounds[1]) if b else 1.0
                   for nm, b in zip(names, box)])
    return (is_log, np.where(is_log, np.log10(np.where(is_log, lo, 1.0)), lo),
            np.where(is_log, np.log10(np.where(is_log, hi, 1.0)), hi))


def _special_columns(prior, names):
    """``[(j, distribution)]`` for the columns that are not a box prior."""
    return [(j, prior.distributions[nm]) for j, nm in enumerate(names)
            if _kind(prior, nm) not in _BOX_KINDS]


def _logit_limits(prior, names):
    """``(lo, hi)`` of the climb's coordinate per column: +-7 where it is a logit (within ~1e-3 of a
    bound, so a start never sits where a sampler's Jacobian vanishes), unbounded for a Normal."""
    free = np.array([_kind(prior, nm) == "Normal" for nm in names])
    return np.where(free, -np.inf, -7.0), np.where(free, np.inf, 7.0)


def _to_logit(x, prior, names):
    x = np.asarray(x, dtype=float)
    is_log, lo_s, hi_s = _site_box(prior, names)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(is_log, np.log10(np.where(is_log, x, 1.0)), x)
        p = np.clip((s - lo_s) / (hi_s - lo_s), 1e-9, 1.0 - 1e-9)
    u = np.log(p) - np.log1p(-p)
    for j, d in _special_columns(prior, names):
        if type(d).__name__ == "Normal":
            u[..., j] = (x[..., j] - d.mu) / d.sigma
        else:                                            # the logit of the prior's own CDF
            q = np.clip(d.cdf(x[..., j]), 1e-9, 1.0 - 1e-9)
            u[..., j] = np.log(q) - np.log1p(-q)
    return u


def _from_logit(u, prior, names):
    u = np.asarray(u, dtype=float)
    is_log, lo_s, hi_s = _site_box(prior, names)
    with np.errstate(over="ignore"):                   # exp(-u) at a clipped -7 logit is fine
        s = lo_s + (hi_s - lo_s) / (1.0 + np.exp(-u))
    x = np.where(is_log, 10.0 ** np.where(is_log, s, 0.0), s)   # no 10**MJD on a linear column
    for j, d in _special_columns(prior, names):
        if type(d).__name__ == "Normal":
            x[..., j] = d.mu + d.sigma * u[..., j]
        else:
            x[..., j] = d.ppf(1.0 / (1.0 + np.exp(-u[..., j])))
    return x


def _jax_from_logit(prior, names, dtype):
    """The JAX twin of :func:`_from_logit`: logit of each prior's own coordinate -> parameters."""
    import jax
    import jax.numpy as jnp

    is_log, lo_s, hi_s = _site_box(prior, names)
    il, los, his = jnp.asarray(is_log), jnp.asarray(lo_s, dtype), jnp.asarray(hi_s, dtype)
    special = []
    for j, d in _special_columns(prior, names):
        if type(d).__name__ == "Normal":
            mu, sd = float(d.mu), float(d.sigma)
            special.append((j, lambda v, mu=mu, sd=sd: mu + sd * v))
        else:
            from ...priors._jax import ppf_jax
            ppf = ppf_jax(d)
            special.append((j, lambda v, ppf=ppf: ppf(jax.nn.sigmoid(v))))

    def to_x(u):
        s = los + (his - los) * jax.nn.sigmoid(u)
        # 10 ** s only where it is taken: on a linear column at an MJD it is inf, and the gradient
        # of the branch where() does not take (inf * 0) would be NaN there.
        x = jnp.where(il, 10.0 ** jnp.where(il, s, 0.0), s)
        for j, f in special:
            x = x.at[..., j].set(f(u[..., j]))
        return x

    return to_x


def _vmap_in_blocks(fn, x, width=None):
    """``fn`` over the rows of ``x``, at most :data:`CLIMB_BLOCK` rows per vmapped block, blocks in
    a ``lax.map``; the last block padded with copies of the last row (dropped from the output). A
    bounded vmap width whatever the number of rows (``emcee_jax`` climbs one start per walker),
    and no padding below one block."""
    import jax
    import jax.numpy as jnp

    n = x.shape[0]
    width = min(CLIMB_BLOCK if width is None else int(width), n)
    pad = (-n) % width
    if pad:
        x = jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0)
    out = jax.lax.map(jax.vmap(fn), x.reshape(-1, width, x.shape[-1]))
    return jax.tree_util.tree_map(lambda a: a.reshape((-1,) + a.shape[2:])[:n], out)


def _conditional_sd(score, points, prior, names):
    """Posterior sd of each logit coordinate with the others held at ``points``: 1/sqrt(-d2 logL).

    The CONDITIONAL width, narrower than the marginal on a correlated posterior, so a start moved
    by a few of these stays on the ridge instead of stepping off it. ``MAX_SPREAD_U`` where the
    curvature is flat, positive or undefined.

    Read off a central second difference, ``(f(u+h) - 2 f(u) + f(u-h)) / h^2``, through ``score``:
    the scan's already-compiled :func:`block_scorer`, so it compiles nothing. It replaced
    ``jax.hessian`` through the model, whose compile was most of the TDE's start (57 of 99 s on
    one A6000, 76 of 96 s on the CPU, at ``n_time=500``). The step is the posterior's own width,
    not a tiny ``h``: starting from :data:`FD_STEP0` logit units, each of :data:`FD_ROUNDS` rounds
    moves ``h`` to the sd the last difference implies (exact in one round on a Gaussian), so the
    difference spans the peak rather than a float32 staircase (an MJD-valued t0) or a kink. A
    probe at ``-inf`` or NaN (a constraint wall) shrinks that step 10x for the next round, and a
    coordinate still walled after the last round gets that shrunk step: an optimum pressed against
    a constraint wall (a magnetar's rotational-energy bound) has no curvature to read there, and
    the full ``MAX_SPREAD_U`` would throw every start through the wall.
    """
    u0 = _to_logit(np.asarray(points, float), prior, names)
    n, k = u0.shape
    f0 = score(_from_logit(u0, prior, names))[:, None]
    eye = np.eye(k)
    h = np.full((n, k), FD_STEP0)
    for _ in range(FD_ROUNDS):
        probes = np.concatenate([u0[:, None, :] + h[:, :, None] * eye,
                                 u0[:, None, :] - h[:, :, None] * eye]).reshape(-1, k)
        f = score(_from_logit(probes, prior, names)).reshape(2, n, k)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            curv = (2.0 * f0 - f[0] - f[1]) / h ** 2
            ok = np.isfinite(curv) & (curv > 0.0)
            h = np.clip(np.where(ok, 1.0 / np.sqrt(curv),
                                 np.where(np.isfinite(curv), MAX_SPREAD_U, 0.1 * h)),
                        1e-9, MAX_SPREAD_U)
    return np.where(ok | ~np.isfinite(curv), h, MAX_SPREAD_U)


def climb(log_density, starts, prior, names, dtype, n_steps=REFINE_STEPS):
    """A short, monotone Adam ascent of ``log_density`` from each start -> ``(points, values)``.

    In the logit of each prior's own coordinate, so every step stays inside the box, clipped to
    +-7 (within ~1e-3 of a bound) so a maximum on a box face does not park a chain where the
    sampler's own Jacobian is vanishing. The step shrinks from 0.1 to 0.002 logit units: far enough
    to find which basin a start belongs to, not so far that every start collapses onto one point --
    under a flat likelihood nothing moves and the starts stay the independent draws they were.
    The starts climb together, :data:`CLIMB_BLOCK` per vmapped block, so the 200 sequential steps
    are paid once per block rather than once per start, and the compile is one program whatever
    their number.

    **Monotone:** a step that lowers the density is refused and the next one halved (a doubling
    brings it back after a success), so a climbed point is never worse than its draw. Adam's step
    is ~``rate`` logit units whatever the gradient, and in a wide box that can be a long way: with
    t0 ~ U(-1000, 1000) and a 3 d bump in 40 d of data, 0.1 logit units is 50 d, and the free
    ascent threw 17 of the 32 best draws off the bump onto the zero-flux plateau (best draw -4097,
    every climbed point -7537.5), so every chain started stranded.
    """
    import jax
    import jax.numpy as jnp

    to_x = _jax_from_logit(prior, names, dtype)
    value_and_grad = jax.value_and_grad(lambda u: log_density(to_x(u)))
    lo_u, hi_u = (jnp.asarray(v, dtype) for v in _logit_limits(prior, names))

    def one(u0):
        def step(carry, i):
            u, f, g, m, v, shrink = carry
            g = jnp.where(jnp.isfinite(g), g, 0.0)      # a NaN corner must not derail the rest
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            rate = shrink * 0.1 * 0.02 ** (i / n_steps)
            trial = jnp.clip(u + rate * (m / (1.0 - 0.9 ** (i + 1)))
                             / (jnp.sqrt(v / (1.0 - 0.999 ** (i + 1))) + 1e-8), lo_u, hi_u)
            f_t, g_t = value_and_grad(trial)
            up = jnp.isfinite(f_t) & (f_t >= f)
            return (jnp.where(up, trial, u), jnp.where(up, f_t, f), jnp.where(up, g_t, g), m, v,
                    jnp.where(up, jnp.minimum(2.0 * shrink, 1.0), 0.5 * shrink)), None
        f0, g0 = value_and_grad(u0)
        z = jnp.zeros_like(u0)
        (u, f, _, _, _, _), _ = jax.lax.scan(step, (u0, f0, g0, z, z, jnp.ones((), u0.dtype)),
                                             jnp.arange(n_steps, dtype=u0.dtype))
        return u, f

    u, val = _vmap_in_blocks(one, jnp.asarray(_to_logit(np.asarray(starts, float), prior, names),
                                              dtype))
    return np.asarray(to_x(u), dtype=float), np.asarray(val, dtype=float)


#: :func:`climb_numpy`: rounds of the derivative-free climb, and the gain (nats) below which a
#: start that has not improved for two rounds stops climbing.
FD_CLIMB_ROUNDS = 30
FD_CLIMB_TOL = 1e-3


def climb_numpy(score, starts, prior, names, n_rounds=FD_CLIMB_ROUNDS, tol=FD_CLIMB_TOL):
    """:func:`climb` for a density with no JAX gradient (a CPU model) -> ``(points, values)``.

    ``score`` maps ``(m, k)`` parameter rows to ``(m,)`` log-densities (serial, or through a worker
    pool), and every start climbs in the same coordinate :func:`climb` uses. Each round probes every
    coordinate at +-h, the conditional posterior sd the last round's second difference implied
    (:data:`FD_STEP0` logit units at first), and moves to the best of: the start, its 2k probes, and
    a diagonal Newton step ``g / c`` (at most one logit unit, tried at full, half and a quarter
    length). A probe at ``-inf`` or NaN shrinks that coordinate's ``h`` tenfold. Monotone, like
    :func:`climb`, and a start stops once it has gained less than ``tol`` nats in two rounds.

    ``2 k + 3`` evaluations per climbing start per round, so on the redback ``arnett`` (6
    parameters, ~2 ms a call) 32 starts cost at most ~29 s.
    """
    lo_u, hi_u = _logit_limits(prior, names)
    u = np.clip(_to_logit(np.asarray(starts, float), prior, names), lo_u, hi_u)
    n, k = u.shape

    def f_of(uu):
        v = np.asarray(score(_from_logit(uu, prior, names)), dtype=float)
        return np.where(np.isfinite(v), v, -np.inf)

    f = f_of(u)
    h = np.full((n, k), FD_STEP0)
    eye = np.eye(k)
    active = np.isfinite(f)
    stall = np.zeros(n, dtype=int)
    for _ in range(int(n_rounds)):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        ua, fa, ha = u[idx], f[idx], h[idx]
        m = idx.size
        probes = np.clip(np.stack([ua[:, None, :] + ha[:, :, None] * eye,
                                   ua[:, None, :] - ha[:, :, None] * eye]), lo_u, hi_u)
        fp = f_of(probes.reshape(-1, k)).reshape(2, m, k)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            g = (fp[0] - fp[1]) / (2.0 * ha)
            c = (2.0 * fa[:, None] - fp[0] - fp[1]) / ha ** 2
        known = np.isfinite(g) & np.isfinite(c)
        curved = known & (c > 0.0)
        step = np.where(curved, g / np.where(curved, c, 1.0),
                        np.where(np.isfinite(g), np.sign(g) * ha, 0.0))
        step = np.clip(step, -1.0, 1.0)
        trials = np.clip(ua[None] + np.array([1.0, 0.5, 0.25])[:, None, None] * step[None],
                         lo_u, hi_u)                                            # (3, m, k)
        ft = f_of(trials.reshape(-1, k)).reshape(3, m)
        cand_u = np.concatenate([ua[:, None, :], probes.transpose(1, 0, 2, 3).reshape(m, 2 * k, k),
                                 trials.transpose(1, 0, 2)], axis=1)          # (m, 1+2k+3, k)
        cand_f = np.concatenate([fa[:, None], fp.transpose(1, 0, 2).reshape(m, 2 * k), ft.T],
                                axis=1)
        best = np.argmax(cand_f, axis=1)
        new_f = cand_f[np.arange(m), best]
        u[idx] = cand_u[np.arange(m), best]
        gain = new_f - fa
        f[idx] = new_f
        # The next probe distance: the conditional sd where the curvature is informative, 10x
        # shorter across a wall (-inf / NaN probe), twice as long where the density is flat.
        h[idx] = np.clip(np.where(curved, 1.0 / np.sqrt(np.where(curved, c, 1.0)),
                                  np.where(known, 2.0 * ha, 0.1 * ha)), 1e-6, MAX_SPREAD_U)
        stall[idx] = np.where(gain < tol, stall[idx] + 1, 0)
        active[idx[stall[idx] >= 2]] = False
    return _from_logit(u, prior, names), f


def _farthest_first(u, n):
    """Indices of ``n`` rows of ``u``: row 0, then each time the row farthest from those taken.

    Cycles through the chosen rows when ``u`` has fewer than ``n`` rows.
    """
    chosen = [0]
    dmin = np.linalg.norm(u - u[0], axis=1)
    dmin[0] = -1.0
    while len(chosen) < min(n, len(u)):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(u - u[j], axis=1))
        dmin[chosen] = -1.0
    return np.asarray(chosen)[np.arange(n) % len(chosen)]


def scan_starts(log_density, prior, names, n_starts, seed, dtype, n_scan=N_PRIOR_SCAN,
                n_climb=None, score=None):
    """The default start (``"prior_scan"``) and the independent optimum, in one pass.

    1. Score ``n_scan`` prior draws (:func:`prior_scan`).
    2. Climb the best ``n_climb`` (default ``max(N_REFINE, n_starts)``) with :func:`climb`; with
       ``log_density=None`` and a numpy ``score`` (a CPU model, no JAX), the best ``N_REFINE``
       with :func:`climb_numpy` instead.
    3. Of the climbed points that end within ``STRANDED_NATS`` of the best -- the best basin, or
       competing optima of nearly the same height -- take the best and then, each time, the one
       farthest from those taken (:func:`_farthest_first`), cycling if there are fewer than
       ``n_starts``; move each ``START_SPREAD`` conditional sd at random
       (:func:`_conditional_sd`, shortened where that costs more than ``max(10, 2k)`` nats) so
       that no two chains start at the same point.

    Farthest-first matters when the posterior has two optima of similar height: flare mock 7 has
    t0 modes near 5 and 22 d, 2.5 nats apart, and chains started only in the higher one returned
    t0 sd 3.0 d against the nested-sampling 5.1 d with R-hat 1.011. Chains started in both cannot
    mix between them -- NUTS is local -- so R-hat says so plainly, which is the honest answer.

    Warns when at least ``PLATEAU_FRACTION`` of the draws, but not all, sit on one flat level
    (plan U3 (iv): a time prior much wider than the data window).

    Returns a dict: ``draws`` and ``scores`` (the scan), ``starts`` ``(n_starts, k)`` and their
    ``start_scores``, ``n_good`` (candidates that reached the best basin), ``n_climbed``,
    ``climb_error`` (why the climb fell back to raw scores, else ``None``), ``plateau_fraction``,
    ``reference`` -- the best log-density found, which every run is checked against -- and
    ``score``, the compiled :func:`block_scorer` of ``log_density``, for the sampler's post-fit
    re-scan of the same density.
    """
    if log_density is None:
        draws = prior_draws(prior, names, n_scan, seed)
        scores = np.asarray(score(draws), dtype=float)
    else:
        score = block_scorer(log_density, dtype) if score is None else score
        draws, scores = prior_scan(log_density, prior, names, seed, dtype, n_scan, score=score)
    ok = np.isfinite(scores)
    if not ok.any():
        raise ValueError(f"the log-density is -inf or NaN at all {len(scores)} prior draws. Check "
                         f"the density (a NaN from the physics?) or pass explicit starts.")
    median = float(np.median(scores[ok]))
    on_plateau = np.abs(scores[ok] - median) < 1.0
    plateau = float(np.mean(on_plateau))
    # At least one draw off it: a likelihood flat over the WHOLE box is a different case (a density
    # that ignores the parameters). The draws off it may all score LOWER -- a bump half over the
    # data fits worse than none -- and the case with no draw above is the one that most needs
    # saying: with t0 ~ U(-1e4, 1e4), 2 of 4 seeds had none, and one never found the event.
    if plateau >= PLATEAU_FRACTION and not on_plateau.all():
        _warn(
            f"{100 * plateau:.1f} % of {int(ok.sum())} prior draws score the same log-density, "
            f"{median:.1f} to within 1 nat: over most of the prior box the model puts no signal "
            f"where the data are -- typically a time parameter (t0, an explosion or peak epoch) "
            f"whose prior is much wider than the data window. A chain there has no gradient to "
            f"follow, and the start rests on the {int((~on_plateau).sum())} draws off that "
            f"plateau. Narrow that prior to a box around the data.")
    if n_climb is None:
        n_climb = N_REFINE if log_density is None else max(N_REFINE, int(n_starts))
    cand = np.argsort(-np.where(ok, scores, -np.inf), kind="stable")[:min(int(n_climb),
                                                                          int(ok.sum()))]
    climb_error = None
    try:
        x, val = (climb_numpy(score, draws[cand], prior, names) if log_density is None
                  else climb(log_density, draws[cand], prior, names, dtype))
    except Exception as exc:                          # noqa: BLE001 - recorded, then raw scores
        # emcee_jax needs no gradient, so its density may be one JAX cannot differentiate; the
        # start then falls back to the best raw scores, and says so.
        climb_error = f"{type(exc).__name__}: {exc}"
        x, val = draws[cand], scores[cand]
    val = np.where(np.isfinite(val), val, -np.inf)
    if not np.isfinite(val).any():                   # the ascent failed everywhere: raw scores
        x, val = draws[cand], scores[cand]
    good = np.flatnonzero(val >= val.max() - STRANDED_NATS)
    good = good[np.argsort(-val[good], kind="stable")]       # best first
    pick = good[_farthest_first(_to_logit(x[good], prior, names), int(n_starts))]
    centres = x[pick]
    sd_u = _conditional_sd(score, centres, prior, names)
    rng = np.random.default_rng([int(seed), 2])
    u0 = _to_logit(centres, prior, names)
    # A 2-sd move in each of k coordinates costs ~2k nats on a Gaussian; where the curvature misled
    # (a kink, a ridge) it can cost hundreds -- flare mock 7 put a start 260 nats down. Such a move
    # is refused when it costs more than this, and the start tries again: a fresh direction and its
    # mirror image (one of the two stays on the allowed side of a constraint wall the optimum is
    # pressed against), at half the length, for up to SPREAD_ROUNDS rounds.
    limit = max(STRANDED_NATS, 2.0 * len(names))
    starts, start_scores = centres.copy(), val[pick].copy()
    todo = np.ones(len(pick), dtype=bool)
    lo_u, hi_u = _logit_limits(prior, names)
    scale = START_SPREAD * sd_u
    for _ in range(SPREAD_ROUNDS):
        offset = scale * rng.standard_normal(centres.shape)
        for sign in (1.0, -1.0):
            rows = np.flatnonzero(todo)
            if rows.size == 0:
                break
            trial = _from_logit(np.clip(u0[rows] + sign * offset[rows], lo_u, hi_u), prior, names)
            trial_score = np.asarray(score(trial), dtype=float)
            keep = np.isfinite(trial_score) & (trial_score >= val[pick][rows] - limit)
            starts[rows[keep]], start_scores[rows[keep]] = trial[keep], trial_score[keep]
            todo[rows[keep]] = False
        if not todo.any():
            break
        scale = 0.5 * scale
    return {"draws": draws, "scores": scores, "starts": starts, "start_scores": start_scores,
            "n_good": int(len(good)), "n_climbed": int(len(cand)), "climb_error": climb_error,
            "plateau_fraction": plateau, "score": score,
            "reference": float(max(np.max(val), np.max(np.where(ok, scores, -np.inf)),
                                   np.max(np.where(np.isfinite(start_scores), start_scores,
                                                   -np.inf))))}


def check_starts(starts, prior, names, n, sampler, what="chain"):
    """A caller's ``(n, k)`` starts, in parameter units: the right shape, strictly inside the box.

    Strictly, because the NUTS samplers walk a logit of each bounded coordinate, which is infinite
    on the bound itself.
    """
    arr = np.asarray(starts, dtype=float)
    if arr.shape != (int(n), len(names)):
        raise ValueError(
            f"{sampler}: starts of shape {arr.shape} given; need one row per {what} and one column "
            f"per parameter, i.e. ({int(n)}, {len(names)}), columns in {list(names)} order, in "
            f"parameter units.")
    lo = np.array([float(prior.distributions[nm].bounds[0]) for nm in names])
    hi = np.array([float(prior.distributions[nm].bounds[1]) for nm in names])
    bad = ~np.isfinite(arr) | (arr <= lo) | (arr >= hi)
    if bad.any():
        i, j = (int(v) for v in np.argwhere(bad)[0])
        raise ValueError(
            f"{sampler}: start {i} has {names[j]} = {arr[i, j]!r}, which is not strictly inside "
            f"its prior ({lo[j]}, {hi[j]}). A start there has zero prior density.")
    return arr


#: Width of the ball ``init=point`` gives (``init=(point, scale)`` sets it), as a fraction of each
#: parameter's prior width in its own coordinate: the box width (its log width for a LogUniform);
#: ``sqrt(12)`` prior sd for a Normal or TruncatedNormal, which is the box width for a Uniform.
DEFAULT_BALL_SCALE = 1e-3


def _own_coordinate(prior, names):
    """``(is_log, width)``: each parameter's own coordinate (natural log for a LogUniform, linear
    otherwise) and the prior's width there (see :data:`DEFAULT_BALL_SCALE`)."""
    is_log = np.array([_kind(prior, nm) == "LogUniform" for nm in names])
    width = []
    for nm, log in zip(names, is_log):
        d = prior.distributions[nm]
        lo, hi = (float(b) for b in d.bounds)
        if _kind(prior, nm) in _BOX_KINDS:
            width.append(np.log(hi / lo) if log else hi - lo)
        else:
            width.append(np.sqrt(12.0) * float(d.std))
    return is_log, np.asarray(width, dtype=float)


def _to_own(x, is_log):
    return np.where(is_log, np.log(np.where(is_log, x, 1.0)), x)


def _box(prior, names):
    lo = np.array([float(prior.distributions[nm].bounds[0]) for nm in names])
    hi = np.array([float(prior.distributions[nm].bounds[1]) for nm in names])
    return lo, hi


def refuse_non_finite(p0, lp, sampler, what):
    """Refuse starts at a non-finite density: emcee carries such a walker unmoved, and a NUTS chain
    started there may never leave it. Inside the prior box (:func:`check_starts`) is not enough: a
    constraint wall or the caller's own density can still be -inf there."""
    bad = np.flatnonzero(~np.isfinite(np.asarray(lp, dtype=float)))
    if bad.size:
        raise ValueError(f"{sampler}: {what} puts start(s) {bad[:10].tolist()} where the "
                         f"log-density is -inf or NaN (outside the prior, behind a constraint wall, "
                         f"or where the model cannot be evaluated). Move them inside the region the "
                         f"density allows, or use the default init='prior_scan'.")
    return p0


def refuse_collapsed(walkers, sampler, what):
    """Refuse an ensemble start that does not span every parameter (in walker coordinates).

    The stretch move proposes along the line between two walkers, so an ensemble started in a
    lower-dimensional subspace (all walkers on one point, or on a few collinear ones) never leaves
    it. emcee refuses such a start with "Initial state has a large condition number"; this names the
    cause and the way out first, for ``emcee_jax``, ``mcmc`` and ``fit_batch`` alike. The test is
    emcee's own: finite, no constant column, and a column-normalised condition number <= 1e8.
    """
    w = np.array(walkers, dtype=float)
    ok = np.all(np.isfinite(w))
    if ok:
        c = w - w.mean(axis=0)
        colmax = np.max(np.abs(c), axis=0)
        ok = bool(np.all(colmax > 0))
        if ok:
            c = c / colmax
            c = c / np.sqrt(np.sum(c ** 2, axis=0))
            ok = bool(np.linalg.cond(c) <= 1e8)
    if not ok:
        n_rows = len(np.unique(w, axis=0))
        raise ValueError(
            f"{sampler}: {what} put the {w.shape[0]} walkers on {n_rows} distinct point(s), which "
            f"do not span all {w.shape[1]} parameters, and the stretch move cannot leave that "
            f"subspace. Start from independent prior draws (init='prior') or pass starts that "
            f"differ in every parameter.")
    return walkers


# ------------------------------------------------------------------------------ walker coordinates
#: The coordinates the ensemble samplers (``emcee_jax``, ``mcmc``, ``fit_batch``) move their
#: walkers in. ``"own"`` (default): each prior's own coordinate -- the natural log of a
#: LogUniform parameter, the parameter itself otherwise -- with the log-Jacobian added to the
#: density, so the posterior is the same. ``"linear"``: the parameters themselves (whisper 0.1.x).
#: The stretch move is affine-invariant only in the coordinates it moves in: in linear ones a
#: parameter spanning eight decades (a LogUniform ``kappa_gamma``) or a ridge that is straight in
#: log(nickel mass) and log(ejecta mass) is explored slowly, and the supernova posteriors came out
#: too narrow (68 % intervals holding the truth 47-60 % of the time on simulated LSST alerts).
WALKER_COORDINATES = ("own", "linear")


def walker_log_columns(prior, names, coordinates, sampler):
    """``(k,)`` bool: which walker coordinates are the natural log of their parameter."""
    if coordinates not in WALKER_COORDINATES:
        raise ValueError(f"{sampler}: walker_coordinates must be one of {WALKER_COORDINATES}; got "
                         f"{coordinates!r}.")
    if coordinates == "linear":
        return np.zeros(len(names), dtype=bool)
    return np.array([_kind(prior, nm) == "LogUniform" for nm in names], dtype=bool)


def to_walker(x, is_log):
    """Parameters -> walker coordinates (natural log on the ``is_log`` columns)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(is_log, np.log(np.where(is_log, x, 1.0)), x)


def from_walker(y, is_log):
    """Walker coordinates -> parameters."""
    y = np.asarray(y, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(is_log, np.exp(np.where(is_log, y, 0.0)), y)


def walker_log_jacobian(y, is_log):
    """``ln |dx / dy|`` per row of walker coordinates ``y``: the sum of the log columns."""
    return np.sum(np.where(is_log, np.asarray(y, dtype=float), 0.0), axis=-1)


def jax_walker_density(density, is_log):
    """``density(theta)`` as a density of walker coordinates: ``density(x(y)) + ln |dx/dy|``."""
    import jax.numpy as jnp

    il = jnp.asarray(is_log)

    def walker_density(y):
        x = jnp.where(il, jnp.exp(jnp.where(il, y, 0.0)), y)
        return density(x) + jnp.sum(jnp.where(il, y, 0.0), axis=-1)

    return walker_density


def ball_starts(centre, sd_own, prior, names, n, rng, score, sampler, what="walker",
                label="ball"):
    """``n`` starts ``~ N(centre, sd_own)`` in each parameter's own coordinate -> ``(starts, lp)``.

    Clipped a hair inside the box, and every start whose density is -inf or NaN (a constraint wall,
    a region the model cannot evaluate) redrawn, up to 100 times, before it is refused. Then, as
    :func:`scan_starts` does with its spread, a start that costs more than ``max(10, 2k)`` nats
    against the centre is pulled halfway back to it, up to 6 times: a ball sized by a broad ABC
    posterior must not put a walker on a zero-signal plateau, where it would stay.
    """
    centre = np.asarray(centre, dtype=float)
    check_starts(centre[None, :], prior, names, 1, sampler, what=f"{label} centre")
    is_log, _ = _own_coordinate(prior, names)
    lo, hi = _box(prior, names)
    with np.errstate(invalid="ignore"):
        eps = np.where(np.isfinite(hi - lo), 1e-9 * (hi - lo), 0.0)
    c_own = _to_own(centre, is_log)
    k = len(names)
    sd_own = np.asarray(sd_own, dtype=float)

    def place(offset):
        x = c_own + offset
        x = np.where(is_log, np.exp(x), x)
        return np.clip(x, lo + eps, hi - eps)

    off = sd_own * rng.standard_normal((int(n), k))
    p0 = place(off)
    lp = np.asarray(score(p0), dtype=float)
    for _ in range(100):
        bad = ~np.isfinite(lp)
        if not bad.any():
            break
        off[bad] = sd_own * rng.standard_normal((int(bad.sum()), k))
        p0[bad] = place(off[bad])
        lp[bad] = score(p0[bad])
    lp_centre = float(np.asarray(score(centre[None, :]), dtype=float)[0])
    limit = max(STRANDED_NATS, 2.0 * k)
    for _ in range(6 if np.isfinite(lp_centre) else 0):
        low = np.flatnonzero(np.isfinite(lp) & (lp < lp_centre - limit))
        if not low.size:
            break
        trial_off = 0.5 * off[low]
        trial = place(trial_off)
        tl = np.asarray(score(trial), dtype=float)
        ok = np.isfinite(tl)
        off[low[ok]], p0[low[ok]], lp[low[ok]] = trial_off[ok], trial[ok], tl[ok]
    return refuse_non_finite(p0, lp, sampler, f"init={label!r}"), lp


def _is_result(obj):
    """A :class:`~whisper_cbpf.samplers.base.SamplerResult`, or anything shaped like one (a result
    reloaded from disk): posterior ``samples`` and ``best_params``."""
    return hasattr(obj, "samples") and hasattr(obj, "best_params") and hasattr(obj, "info")


def _result_centre(result, names):
    """The ball fallback's centre: the optimised likelihood maximum when
    ``result.info["likelihood_max_opt"]`` carries one (a ``LikelihoodMaxOptResult`` or a dict with
    ``"params"``) for every name, else ``result.best_params``."""
    peak = (result.info or {}).get("likelihood_max_opt") if isinstance(result.info, dict) else None
    params = getattr(peak, "params", None)
    if params is None and isinstance(peak, dict):
        params = peak.get("params")
    if isinstance(params, dict) and all(nm in params for nm in names):
        c = np.array([float(params[nm]) for nm in names])
        if np.all(np.isfinite(c)):
            return c, "likelihood_max_opt"
    best = dict(result.best_params or {})
    return np.array([float(best.get(nm, np.nan)) for nm in names]), "best_params"


def result_starts(result, prior, names, n, seed, score, sampler, what="walker"):
    """Start from a previous fit -> ``(starts, lp, "result", detail)``.

    The ABC -> MCMC handoff, a continuation, or an alert's next cut (3 -> 6 detections -> +10 d):

    * If the result has at least ``n`` usable draws -- finite, strictly inside the box, distinct,
      and at a finite density here (inside every constraint wall) -- the starts are a random subset
      of them (``detail["mode"] = "draws"``).
    * Otherwise a ball around its optimised likelihood maximum
      (``result.info["likelihood_max_opt"]``) or else its ``best_params``, sized by its own
      spread: the sd of its draws in each parameter's own coordinate, or :data:`DEFAULT_BALL_SCALE` of the prior width with fewer than 2 draws
      (``detail["mode"] = "ball"``). An ABC fit that accepted fewer draws than there are walkers
      (0-43 on real supernovae, against 60 walkers) lands here.

    Every start is checked against the box, the constraint wall and a finite density.
    """
    rng = np.random.default_rng([int(seed), 3])
    k = len(names)
    samples = getattr(result, "samples", None)
    cols = [] if samples is None else list(samples.columns)
    best = dict(result.best_params or {})
    missing = [nm for nm in names if nm not in cols and nm not in best]
    src = getattr(result, "sampler", "?")
    if missing:
        raise ValueError(
            f"{sampler}: init=<{src} result> has no {missing}; it holds "
            f"{list(getattr(result, 'parameters', cols))}. Seed from a fit of the same model and "
            f"prior, or pass a point: init={{name: value, ...}}.")
    have_draws = samples is not None and len(samples) and all(nm in cols for nm in names)
    draws = samples[names].to_numpy(dtype=float) if have_draws else np.empty((0, k))
    lo, hi = _box(prior, names)
    inside = np.all(np.isfinite(draws) & (draws > lo) & (draws < hi), axis=1)
    cand = np.unique(draws[inside], axis=0) if inside.any() else np.empty((0, k))
    cand = cand[rng.permutation(len(cand))]
    usable, lp_ok = [], []
    block = max(4 * int(n), SCORE_BLOCK)
    for i in range(0, len(cand), block):
        blk = cand[i:i + block]
        lp = np.asarray(score(blk), dtype=float)
        ok = np.isfinite(lp)
        usable.extend(blk[ok])
        lp_ok.extend(lp[ok])
        if len(usable) >= n:
            break
    detail = {"source": src, "n_draws": int(len(draws)), "n_distinct_in_box": int(len(cand)),
              "n_usable_found": int(len(usable))}
    if len(usable) >= n:
        return (np.asarray(usable[:n]), np.asarray(lp_ok[:n]), "result",
                {**detail, "mode": "draws"})

    centre, centre_src = _result_centre(result, names)
    if not np.all(np.isfinite(centre)):
        raise ValueError(
            f"{sampler}: init=<{src} result> has {len(usable)} usable draws for {n} {what}s and no "
            f"finite best_params to centre a ball on. Use init='prior_scan', or pass a point.")
    is_log, width = _own_coordinate(prior, names)
    fallback = DEFAULT_BALL_SCALE * width
    if len(cand) >= 2:
        sd = np.std(_to_own(cand, is_log), axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0.0), sd, fallback)
        spread = "sd of its draws"
    else:
        sd, spread = fallback, f"{DEFAULT_BALL_SCALE:g} x prior width (fewer than 2 draws)"
    p0, lp = ball_starts(centre, sd, prior, names, n, rng, score, sampler, what,
                         label="result")
    return p0, lp, "result", {**detail, "mode": "ball", "centre": centre_src, "spread": spread,
                              "ball_sd_own": {nm: float(s) for nm, s in zip(names, sd)}}


def split_fixed(prior, names, sampler):
    """``(prior over the free names, free names, {name: value})``: a sampler moves only the free
    parameters and holds each :class:`~whisper_cbpf.priors.Fixed` one at its value."""
    fixed = {nm: float(prior.distributions[nm].value) for nm in names
             if _kind(prior, nm) == "Fixed"}
    if not fixed:
        return prior, list(names), {}
    free = [nm for nm in names if nm not in fixed]
    if not free:
        raise ValueError(f"{sampler}: every parameter is Fixed ({fixed}), so there is nothing to "
                         f"sample. Evaluate the model at those values directly.")
    return type(prior)({nm: prior.distributions[nm] for nm in free}), free, fixed


def fill_fixed(x, free, fixed, names):
    """The last axis of ``x`` (the free parameters, in ``free`` order) -> ``names`` order, each
    Fixed parameter a constant column at its value."""
    if not fixed:
        return x
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape[:-1] + (len(names),))
    for j, nm in enumerate(names):
        out[..., j] = fixed[nm] if nm in fixed else x[..., free.index(nm)]
    return out


def merge_init(init, init_strategy, default, sampler):
    """``init=`` is the name every MCMC-type sampler shares; ``init_strategy=`` the older name the
    NUTS samplers keep. Either may be given, not both."""
    if init is None:
        return init_strategy
    if not (isinstance(init_strategy, str) and init_strategy == default):
        raise ValueError(f"{sampler}: pass the start as init= or as init_strategy=, not both "
                         f"(init={type(init).__name__}, init_strategy={init_strategy!r:.60}).")
    return init


def explicit_starts(init, prior, names, n, seed, score, sampler, what="walker"):
    """The ``init=`` forms every MCMC-type sampler shares -> ``(starts, lp, label, detail)``.

    * a previous result (a ``SamplerResult``, ABC included): :func:`result_starts`, label
      ``"result"``;
    * a point -- a dict ``{name: value}`` or a ``(k,)`` array -- or ``(point, scale)``: a ball of
      ``scale`` (default :data:`DEFAULT_BALL_SCALE`) times each prior's width, label ``"ball"``;
    * a ``(n, k)`` array: one start per chain or walker, as given, label ``"per_chain"`` /
      ``"per_walker"``.

    ``score`` maps ``(m, k)`` rows to log-densities; every start must be finite there. ``names`` are
    the sampled parameters, in the order the columns of an array are read.
    """
    if _is_result(init):
        return result_starts(init, prior, names, n, seed, score, sampler, what)
    k = len(names)
    scale = DEFAULT_BALL_SCALE
    if (isinstance(init, tuple) and len(init) == 2 and np.ndim(init[1]) == 0
            and (isinstance(init[0], dict) or np.ndim(init[0]) >= 1)):
        init, scale = init
        if not (np.isfinite(float(scale)) and float(scale) > 0.0):
            raise ValueError(f"{sampler}: init=(point, scale) needs a positive scale; got {scale!r}.")
    if isinstance(init, dict):
        missing = [nm for nm in names if nm not in init]
        if missing:
            raise ValueError(f"{sampler}: init point has no value for {missing}; it needs one per "
                             f"sampled parameter {list(names)}.")
        init = [float(init[nm]) for nm in names]
    try:
        arr = np.asarray(init, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{sampler}: init must be 'prior_scan', 'prior', a point (dict or "
                        f"({k},) array), (point, scale), one start per {what} ({n}, {k}), or a "
                        f"previous result; got {type(init).__name__}.") from None
    if arr.ndim == 2:
        p0 = check_starts(arr, prior, names, n, sampler, what=what)
        lp = np.asarray(score(p0), dtype=float)
        label = f"per_{what}"
        return refuse_non_finite(p0.copy(), lp, sampler, f"init=<{label} starts>"), lp, label, {}
    if arr.shape != (k,):
        raise ValueError(f"{sampler}: init point has shape {arr.shape}; need ({k},) in {list(names)} "
                         f"order, a dict, or ({n}, {k}) starts, one per {what}.")
    _, width = _own_coordinate(prior, names)
    rng = np.random.default_rng([int(seed), 1])
    p0, lp = ball_starts(arr, float(scale) * width, prior, names, n, rng, score, sampler, what)
    return p0, lp, "ball", {"scale": float(scale)}


def prior_starts(sc, n, sampler):
    """``init_strategy="prior"``: the first ``n`` of :func:`scan_starts`' prior draws at which the
    density is finite -> ``(starts, scores)``. Independent draws, not ranked by the density, but
    never a start at -inf -- the rule emcee_jax's ``"prior"`` and NumPyro's ``init_to_uniform``
    keep by redrawing. Taking the first ``n`` draws whatever their score put 2 of 4 chains at -inf
    on the JAX ``arnett`` (seed 0), whose constraint wall excludes 61 % of its default prior."""
    ok = np.flatnonzero(np.isfinite(sc["scores"]))
    if ok.size < int(n):
        raise ValueError(
            f"{sampler}: init_strategy='prior' found {ok.size} of {len(sc['scores'])} prior draws "
            f"where the log-density is finite, for {int(n)} chains. Use the default 'prior_scan' "
            f"or pass one start per chain.")
    idx = ok[:int(n)]
    return sc["draws"][idx], sc["scores"][idx]


# ---------------------------------------------------------------------------------------- guards
def float32_hazard(times, prior, names, sampler, x64):
    """Warn before sampling when float32 cannot resolve the clock or a Uniform parameter.

    Keyed on the ABSOLUTE size of ``lc.time`` and of every Uniform bound (see ``F32_ABS_LIMIT``).
    LogUniform parameters are scale parameters, whose relative precision float32 keeps, and the
    NUTS samplers walk them in log10 anyway. Returns the message (for ``info``) or ``None``.
    """
    if x64:
        return None
    offenders = []
    t = np.asarray(times, dtype=float)
    if t.size and np.nanmax(np.abs(t)) >= F32_ABS_LIMIT:
        v = float(np.nanmax(np.abs(t)))
        offenders.append(f"lc.time reaches {v:.6g} (float32 spacing {np.spacing(np.float32(v)):.2g})")
    for nm in names:
        d = prior.distributions.get(nm)
        kind = family(d)
        if d is not None and kind in ("Uniform", "Normal", "TruncatedNormal"):
            # Where the parameter lives: a Uniform's bounds, a Normal's mean, a TruncatedNormal's
            # mean and finite bounds -- all sampled in linear units.
            vals = ([abs(float(b)) for b in d.bounds if np.isfinite(b)]
                    + ([abs(float(d.mu))] if kind != "Uniform" else []))
            v = max(vals)
            if v >= F32_ABS_LIMIT:
                offenders.append(f"{nm!r} ~ {d!r} (float32 spacing {np.spacing(np.float32(v)):.2g})")
    if not offenders:
        return None
    msg = (f"{sampler} runs in float32 (jax_enable_x64 is off) and " + "; ".join(offenders)
           + ". At that size float32 turns the posterior into a staircase: a time parameter near MJD "
             "60000 can only move in 0.0039 d steps, and chains freeze on it (19 of 100 flare fits "
             "froze this way). Enable float64 before importing JAX (JAX_ENABLE_X64=1), or measure "
             "time from an epoch near the data: lc.set_explosion_date(mjd) and a t0 prior in days.")
    _warn(msg)
    return msg


def check_likelihood_contract(log_prob_fn, prior, names, sampler):
    """Refuse a log-POSTERIOR where these samplers add the prior themselves.

    ``nuts_gpu`` and ``pymc_jax_gpu_*`` contribute the prior density through their own sample
    sites, so a density that already includes it counts it twice: 2.5 dex off on a LogUniform
    parameter, silently. Only detectable when the density says so (``.includes_prior``, set by
    ``make_log_prob_jax``). With every prior Uniform the doubled term is a constant inside the box
    and changes nothing -- the fairness benchmark shares one object that way -- so that is a warning.
    """
    if not getattr(log_prob_fn, "includes_prior", False):
        return
    non_uniform = {nm: type(prior.distributions[nm]).__name__ for nm in names
                   if type(prior.distributions[nm]).__name__ != "Uniform"}
    if non_uniform:
        raise ValueError(
            f"{sampler} was given a log-POSTERIOR (log_prob_fn.includes_prior is True), but it adds "
            f"the prior density itself, so the prior would be counted twice -- on {non_uniform} "
            f"that is a different posterior (measured: 2.5 dex off on a LogUniform parameter). "
            f"Pass the pure log-likelihood: make_log_prob_jax(..., include_prior=False), or the "
            f"density's .log_likelihood twin.")
    _warn(
        f"{sampler} was given a log-POSTERIOR (includes_prior=True). Every prior here is Uniform, "
        f"so the doubled prior term is a constant and the posterior is unchanged; pass the pure "
        f"log-likelihood to silence this.")


#: NumPyro strategies that put every chain on (nearly) the same point. ``init_to_median`` takes the
#: median of 15 prior draws per chain, which lands within a few percent of the same point.
POINT_STRATEGIES = ("init_to_median", "init_to_mean", "init_to_feasible", "init_to_value")


def warn_point_start(label, num_chains, sampler):
    """Chains that start at one point cannot disagree about where they started, which is half of
    what R-hat measures."""
    if int(num_chains) > 1 and label in POINT_STRATEGIES:
        _warn(
            f"{sampler}: init_strategy {label!r} starts all {int(num_chains)} chains at (nearly) "
            f"the same point, so R-hat can no longer tell a chain that never left a local optimum "
            f"from a converged one. Use the default 'prior_scan' (distinct high-density prior "
            f"draws), 'prior', or pass one start per chain.")
