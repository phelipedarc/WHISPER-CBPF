"""Markov Chain Monte Carlo (MCMC) sampling via ``emcee`` -- likelihood-based posterior inference.

Same pluggable contract as every other sampler: ``fit(lc, model, prior=None, ...) -> SamplerResult``.
The log-posterior is

    log P(theta | data) = log prior(theta) + log L(theta)

where the prior is Whisper's :class:`~whisper_cbpf.priors.Prior` and **the likelihood is Whisper's own
likelihood layer** (:func:`~whisper_cbpf.likelihood.make_likelihood` /
:class:`~whisper_cbpf.likelihood.GaussianLikelihood`). So MCMC uses the *same physically consistent
likelihood* as ABC / ABC-SMC / SNPE and automatically respects the light curve's ``data_mode`` — it
compares in **flux** space for flux data and **magnitude** space for magnitude data (the model always
predicts flux; the likelihood converts). All four samplers should therefore converge to the same
posterior.

``emcee`` is a core dependency (no extra needed). Sampling is seeded for reproducibility.
"""
from __future__ import annotations

import functools
import multiprocessing
import time

import numpy as np
import pandas as pd

from ..likelihood import make_likelihood
from ..models import get_model
from .base import (
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    check_burnin,
    summarize_posterior,
)
from .jax import _diagnostics as _dg                                # numpy-only; no JAX import
from .jax._diagnostics import walker_health, warn_if_unconverged

#: Worker start method for the ``n_jobs > 1`` pool: "spawn", as in ``samplers/abc.py`` and
#: ``samplers/nested.py``, never the Linux default "fork". A process that has run any JAX model
#: holds a live XLA runtime (hundreds of threads), and ``fork()`` copies only the calling thread, so
#: a lock held by any other stays locked in every child: a JAX-on-CPU model under a forked emcee
#: pool hung for 48 minutes at 0 % CPU. Spawned workers start clean. The
#: result does not depend on the start method -- emcee draws every proposal in the parent and the
#: pool only evaluates densities -- but spawn pays a fresh interpreter per worker, which is why
#: ``n_jobs`` stays opt-in.
_MP_CONTEXT = multiprocessing.get_context("spawn")


def _log_prob(theta, names, prior, predict, times, bands, likelihood, scatter_name=None, *,
              fixed=None, reject_dark=False):
    """log-posterior for one parameter vector ``theta`` (ordered like ``names``).

    ``scatter_name`` routes that prior parameter to the likelihood as its extra-scatter term
    (:class:`~whisper_cbpf.likelihood.GaussianLikelihoodWithScatter`) instead of the model —
    the model's ``predict`` still receives the full params dict (models ignore unknown keys).
    ``fixed`` (``{name: value}``) holds a Fixed prior's parameters at their values: ``theta``
    carries only the free ones.

    ``reject_dark=True`` is the test a START must pass, not the sampled density: ``-inf`` where the
    model predicts no flux at any epoch -- a draw behind a constraint wall (the redback adapter and
    the JAX supernova and TDE models return zero flux there) or with its event outside the data. A
    walker started there sees a flat density and can stay for the whole run.
    """
    params = {nm: float(v) for nm, v in zip(names, theta)}
    if fixed:
        params.update(fixed)
    lp = prior.log_prob(params)
    if not np.isfinite(lp):                       # outside the prior support -> forbidden
        return -np.inf
    model_flux = np.asarray(predict(params, times, bands), dtype=float)
    if reject_dark and not np.any(model_flux != 0.0):
        return -np.inf
    if scatter_name is not None:
        ll = likelihood.log_likelihood(model_flux, sigma_extra=params[scatter_name])
    else:
        ll = likelihood.log_likelihood(model_flux)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def _log_prob_walker(walker, is_log, *args, **kwargs):
    """:func:`_log_prob` of the walker coordinates (natural log on the ``is_log`` columns), plus
    the log-Jacobian: the same posterior, in the coordinates the walkers move in."""
    walker = np.asarray(walker, dtype=float)
    lp = _log_prob(_dg.from_walker(walker, is_log), *args, **kwargs)
    return lp + float(_dg.walker_log_jacobian(walker, is_log)) if np.isfinite(lp) else lp


def _start_scorer(args, pool, n_jobs, fixed=None):
    """``score(theta_2d) -> (m,)``: a start's log-posterior (``-inf`` where it is dark or walled),
    row by row, through the sampler's worker pool when there is one."""
    fn = functools.partial(_log_prob, names=args[0], prior=args[1], predict=args[2],
                           times=args[3], bands=args[4], likelihood=args[5],
                           scatter_name=args[6], fixed=fixed, reject_dark=True)

    def score(theta):
        rows = list(np.asarray(theta, dtype=float))
        if pool is None:
            return np.array([fn(r) for r in rows], dtype=float)
        chunk = max(1, len(rows) // (4 * int(n_jobs)))
        return np.asarray(pool.map(fn, rows, chunksize=chunk), dtype=float)

    return score


def _walker_starts(init, initial_guess, initial_scatter, prior, names, nwalkers, seed, score):
    """Resolve ``init=`` (or the older ``initial_guess=``) -> ``(p0, label, detail)``."""
    rng = np.random.default_rng(int(seed))
    ndim = len(names)
    if initial_guess is not None:
        g = np.array([float(initial_guess[nm]) if isinstance(initial_guess, dict)
                      else float(initial_guess[i]) for i, nm in enumerate(names)], dtype=float)
        p0 = g + initial_scatter * rng.standard_normal((nwalkers, ndim))
        return (_dg.refuse_non_finite(p0, score(p0), "mcmc", "initial_guess"), "initial_guess",
                {"initial_scatter": float(initial_scatter)})
    if init is None or (isinstance(init, str) and init == "prior_scan"):
        sc = _dg.scan_starts(None, prior, names, int(nwalkers), seed, None,
                             n_scan=max(_dg.N_PRIOR_SCAN, 4 * int(nwalkers)), score=score)
        p0 = _dg.refuse_non_finite(sc["starts"], sc["start_scores"], "mcmc", "init='prior_scan'")
        return p0, "prior_scan", {"n_draws": int(len(sc["draws"])),
                                  "n_climbed": sc["n_climbed"], "n_reaching_best": sc["n_good"],
                                  "plateau_fraction": sc["plateau_fraction"],
                                  "climb_error": sc["climb_error"],
                                  "best_log_prob": sc["reference"],
                                  "worst_start_log_prob": float(np.min(sc["start_scores"]))}
    if isinstance(init, str) and init == "prior":
        # The start every fit used before "prior_scan": one prior draw per walker, the same draws
        # for the same seed. Now each is redrawn (from the same stream) while it is -inf or dark.
        def draw(m):
            return np.array([[prior.distributions[nm].sample(rng) for nm in names]
                             for _ in range(m)], dtype=float)

        p0 = draw(int(nwalkers))
        lp = score(p0)
        for _ in range(100):
            bad = ~np.isfinite(lp)
            if not bad.any():
                break
            p0[bad] = draw(int(bad.sum()))
            lp[bad] = score(p0[bad])
        return _dg.refuse_non_finite(p0, lp, "mcmc", "init='prior'"), "prior", {}
    if isinstance(init, str):
        raise ValueError(f"mcmc: unknown init {init!r}. Use 'prior_scan' (default), 'prior', a "
                         f"point, (point, scale), one start per walker, or a previous result.")
    p0, _lp, label, detail = _dg.explicit_starts(init, prior, names, int(nwalkers), seed, score,
                                                 "mcmc", what="walker")
    return p0, label, detail


class MCMCSampler(BaseSampler):
    """Affine-invariant ensemble MCMC (``emcee``). See the module docstring."""

    name = "mcmc"

    def fit(self, lc, model, prior=None, *, nwalkers=None, nsteps=5000, burnin=1000, thin=10,
            init="prior_scan", initial_guess=None, initial_scatter=1e-3, space="auto",
            likelihood="auto", seed=0, progress=False, moves=None, n_jobs=None,
            walker_coordinates="own") -> SamplerResult:
        """Fit ``lc`` with ``model`` via emcee MCMC, returning a :class:`SamplerResult`.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry errors (``flux_err`` / ``magnitude_err``) for the likelihood.
        model : str or Model
            Registered model name or a :class:`~whisper_cbpf.models.Model`.
        prior : Prior, optional
            Parameter prior; defaults to the model's ``default_prior`` (``ValueError`` if neither).
        nwalkers : int, optional
            Number of ensemble walkers (default ``max(2*ndim+2, 4*ndim)``, forced even; must be ``>= 2*ndim``).
        nsteps : int, default 5000
            MCMC iterations per walker.
        burnin : int, default 1000
            Initial steps discarded before convergence.
        thin : int, default 10
            Keep every ``thin``-th sample (reduces autocorrelation).
        init : str, dict, array, tuple or SamplerResult, default "prior_scan"
            Where the walkers start, as for ``emcee_jax`` and the NUTS samplers:

            ``"prior_scan"`` (default)
                Score ``max(1000, 4 * nwalkers)`` prior draws (each in its own coordinate, log for
                a LogUniform), climb the best 32 by a derivative-free ascent
                (``_diagnostics.climb_numpy``), and start the walkers ~2 posterior sd around the
                climbed points that reach the best basin. With ``n_jobs > 1`` the pool scores them.
                It replaced independent prior draws, which on redback ``arnett`` / SN2025pgp left
                12-15 of 60 walkers stuck in lower modes and N/tau 11-12 after 30 000 steps.
            ``"prior"``
                The old start: one independent prior draw per walker (the same draws as before for
                a given ``seed``), each redrawn while its density is -inf.
            a point -- a dict or a ``(ndim,)`` array -- or ``(point, scale)``
                A ball of ``scale`` (default 1e-3) times each prior's width in its own coordinate.
            a ``(nwalkers, ndim)`` array
                One start per walker, in ``prior.names`` order.
            a ``SamplerResult``
                A previous fit -- an ABC fit, a continuation, an alert's earlier cut: a random
                subset of its usable draws, or with fewer than ``nwalkers`` of them a ball around its
                optimised likelihood maximum or ``best_params``, sized by its own spread.

            Every start is checked: inside the box, at a finite density, and with the model
            predicting some flux (not behind a constraint wall). ``info["init"]`` records the kind
            that ran, ``info["init_detail"]`` how, and ``info["init_time_s"]`` its cost (inside
            ``runtime_s``).
        initial_guess : dict or array, optional
            The older way to give a start: walkers at ``initial_guess`` plus ``initial_scatter``
            Gaussian noise in linear units. Cannot be combined with a non-default ``init``.
        initial_scatter : float, default 1e-3
            Gaussian spread of the initial walker cloud around ``initial_guess``.
        space : {'auto', 'flux', 'magnitude'}, default 'auto'
            Comparison space passed to the likelihood (``'auto'`` follows the data's ``data_mode``).
        likelihood : str, default 'auto'
            Likelihood ``kind`` for :func:`make_likelihood` (e.g. ``'gaussian'``,
            ``'gaussian_upper_limits'``, ``'mixture'``); ``'auto'`` picks by data.
        seed : int, default 0
            RNG seed; the chain is reproducible given the seed (walker init + emcee proposals are seeded).
        progress : bool, default False
            Show the emcee progress bar.
        moves : optional
            An ``emcee`` move (or list of (move, weight)); default is emcee's stretch move. It acts
            in the walker coordinates (``walker_coordinates``).
        walker_coordinates : {"own", "linear"}, default "own"
            Where the walkers move: ``"own"`` in each prior's own coordinate -- the natural log of
            a LogUniform parameter, the parameter itself otherwise, with the log-Jacobian added,
            so the posterior is the same -- or ``"linear"`` in the parameters themselves (whisper
            0.1.x). The stretch move is affine-invariant only in the coordinates it moves in; in
            linear ones a LogUniform parameter spanning decades is explored slowly and the
            supernova posteriors came out too narrow. ``info["walker_coordinates"]`` names the log
            columns; the draws are in parameter units, ``result.emcee_sampler``'s chain in walker
            coordinates.
        n_jobs : int, optional
            Worker processes for parallel likelihood evaluation (emcee ``pool``). Worth it only when
            one likelihood call is expensive (e.g. a ~0.1 s radiative-transfer model); for cheap
            analytic models the process overhead dominates. Default ``None`` = serial. Workers are
            started with ``spawn`` (see ``_MP_CONTEXT``), so the model ``predict``, ``prior`` and
            likelihood must be picklable, and a script passing ``n_jobs > 1`` needs an
            ``if __name__ == "__main__":`` guard, because ``spawn`` re-imports ``__main__``. The
            draws are identical to a serial run with the same ``seed``.

        Returns
        -------
        SamplerResult
            Posterior samples + summary, ``best_params`` (max-posterior draw), exact Gaussian
            ``max_log_likelihood`` / ``aic`` / ``bic``, and diagnostics in ``info`` (acceptance fraction,
            autocorrelation times). ``converged`` requires no stuck walker -- one whose median
            log-posterior is more than 10 nats below the best walker's (``stuck_walkers``) -- and
            ``nsteps >= 50 x`` the LARGEST autocorrelation time (``max_autocorr_time``); each failed
            check is a sentence in ``convergence_problems`` and one warning. The rule used to be
            the MEAN time alone, and passed an unknown time: a flare fit with a stuck walker
            reported t0 9.7 sd off, sd 37x too wide, and ``converged=True``. The trained
            ``emcee.EnsembleSampler`` is attached as ``result.emcee_sampler``. A ``Fixed`` prior's
            parameters are held at their values (not sampled, not counted in AIC/BIC,
            ``info["fixed"]``).

        Examples
        --------
        Fit, then fit again from the first posterior (an alert's next cut, or a longer run):

        >>> import numpy as np, whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=wp.get_model("flare").predict(truth, t),
        ...                    flux_err=np.full(30, 0.1))
        >>> first = wp.fit_MCMC(lc, "flare", nsteps=1500, burnin=500)
        >>> first.info["init"]
        'prior_scan'
        >>> again = wp.fit_MCMC(lc, "flare", nsteps=1500, burnin=500, init=first)
        >>> again.info["init"], again.info["init_detail"]["mode"]
        ('result', 'draws')
        """
        import emcee

        check_burnin(burnin, nsteps)
        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        lik = make_likelihood(lc, kind=likelihood, space=space)   # reuse the shared likelihood layer
        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        predict = model.predict
        # A Fixed parameter is held at its value: the walkers move only the free ones (`names`,
        # `free_prior`); the density sees every parameter through `fixed`.
        all_names = list(prior.names)
        free_prior, names, fixed = _dg.split_fixed(prior, all_names, "mcmc")
        ndim = len(names)
        k, n = ndim, int(len(times))

        nwalkers = int(nwalkers) if nwalkers else max(2 * ndim + 2, 4 * ndim)
        nwalkers += nwalkers % 2                                   # emcee's red-blue split needs even
        if nwalkers < 2 * ndim:
            raise ValueError(f"nwalkers ({nwalkers}) must be >= 2*ndim ({2 * ndim}).")

        if initial_guess is not None and not (isinstance(init, str) and init == "prior_scan"):
            raise ValueError("mcmc: pass the start as init= or as initial_guess=, not both.")

        # A prior parameter named after the likelihood's scatter term (GaussianLikelihoodWithScatter)
        # is a LIKELIHOOD parameter: routed to log_likelihood(sigma_extra=...), sampled like the rest.
        scatter_name = getattr(lik, "scatter_param", None)
        scatter_name = scatter_name if (scatter_name and scatter_name in names) else None
        args = (names, prior, predict, times, bands, lik, scatter_name)
        # The walkers move in each prior's own coordinate (log for a LogUniform) by default.
        is_log = _dg.walker_log_columns(free_prior, names, walker_coordinates, "mcmc")

        pool = _MP_CONTEXT.Pool(int(n_jobs)) if n_jobs and int(n_jobs) > 1 else None
        try:
            t_i = time.perf_counter()
            p0, init_label, init_detail = _walker_starts(
                init, initial_guess, initial_scatter, free_prior, names, nwalkers, seed,
                _start_scorer(args, pool, n_jobs, fixed))
            y0 = _dg.refuse_collapsed(_dg.to_walker(p0, is_log), "mcmc", f"init={init_label!r}")
            init_time = time.perf_counter() - t_i

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, _log_prob_walker, moves=moves, pool=pool, args=(is_log,) + args,
                kwargs={"fixed": fixed} if fixed else None)
            sampler._random.seed(int(seed))                       # seed emcee's proposal RNG

            t0 = time.perf_counter()
            sampler.run_mcmc(y0, int(nsteps), progress=progress)
            runtime = time.perf_counter() - t0 + init_time
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        flat_w = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
        flat = _dg.from_walker(flat_w, is_log)
        # the log posterior in parameter units: the sampled density less the log-Jacobian
        flat_lp = (sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
                   - _dg.walker_log_jacobian(flat_w, is_log))
        full = _dg.fill_fixed(flat, names, fixed, all_names)        # every parameter, as columns
        samples = pd.DataFrame(full, columns=all_names)

        # AIC/BIC use the maximum *likelihood*, not the maximum *posterior*. log_prob = prior + ll, so
        # recover the per-draw log-likelihood ll = log_prob - prior.log_prob and take its argmax. (For a
        # flat/Uniform prior these coincide; for a LogUniform prior they differ, which would bias AIC/BIC.)
        prior_lp = np.array([prior.log_prob({nm: float(full[i, j]) for j, nm in enumerate(all_names)})
                             for i in range(full.shape[0])], dtype=float)
        flat_ll = flat_lp - prior_lp
        best_idx = int(np.nanargmax(flat_ll)) if np.any(np.isfinite(flat_ll)) else int(np.argmax(flat_lp))
        best_params = {nm: float(full[best_idx, j]) for j, nm in enumerate(all_names)}
        max_log_likelihood = float(flat_ll[best_idx])

        try:
            taus = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
        except Exception:                                         # pragma: no cover - short chains
            taus = np.array([np.nan])
        # Convergence guard: CIs and AIC/BIC from an unconverged chain are unreliable. A stuck walker
        # (median log-posterior far below the best walker's) fails it as well as a short chain does,
        # and an unknown autocorrelation time is not a pass.
        health = walker_health(sampler.get_log_prob(discard=int(burnin), thin=int(thin)).T,
                               taus, nsteps)
        warn_if_unconverged("mcmc", health)
        info = {
            "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
            "space": lik.space, "likelihood": type(lik).__name__,
            "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "mean_autocorr_time": (float(np.nanmean(taus)) if np.isfinite(taus).any()
                                   else float("nan")),
            "n_samples_per_walker": int(flat.shape[0] // nwalkers),
            "init": init_label, "init_detail": init_detail, "init_time_s": float(init_time),
            "walker_coordinates": {"coordinates": walker_coordinates,
                                   "log": [nm for nm, lg in zip(names, is_log) if lg]},
            "fixed": dict(fixed),
            **health, "seed": int(seed),
        }
        attach_band_metrics(info, lc, model, best_params, space)
        aic, bic = aic_bic(max_log_likelihood, k, n)
        result = SamplerResult(
            sampler="mcmc", model=model.name, parameters=list(all_names), samples=samples,
            summary=summarize_posterior(samples, all_names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=aic, bic=bic,
        )
        result.emcee_sampler = sampler
        attach_predictive_metrics(result, lc, space, model=model)
        return result


def fit_MCMC(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via emcee MCMC. See :meth:`MCMCSampler.fit` for options.

    Parameters
    ----------
    lc : LightCurve
        The data.
    model : str or Model, default "flare"
    prior : Prior, optional
        Default: the model's.
    **kwargs
        The sampler's settings (``nwalkers``, ``nsteps``, ``burnin``, ``init``, ``seed``, ...).

    Returns
    -------
    SamplerResult

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=wp.get_model("flare").predict(truth, t),
    ...                    flux_err=np.full(30, 0.2))
    >>> res = wp.fit_MCMC(lc, "flare", nwalkers=16, nsteps=1500, burnin=500, seed=0)
    >>> res.sampler, bool(abs(res.summary["amplitude"]["median"] - 5.0) < 0.5)
    ('mcmc', True)
    """
    return MCMCSampler().fit(lc, model, prior=prior, **kwargs)
