"""Markov Chain Monte Carlo (MCMC) sampling via ``emcee`` -- likelihood-based posterior inference.

Same pluggable contract as every other sampler: ``fit(lc, model, prior=None, ...) -> SamplerResult``.
The log-posterior is

    log P(theta | data) = log prior(theta) + log L(theta)

where the prior is Whisper's :class:`~whisper_labia.priors.Prior` and **the likelihood is Whisper's own
likelihood layer** (:func:`~whisper_labia.likelihood.make_likelihood` /
:class:`~whisper_labia.likelihood.GaussianLikelihood`). So MCMC uses the *same physically consistent
likelihood* as ABC / ABC-SMC / SNPE and automatically respects the light curve's ``data_mode`` — it
compares in **flux** space for flux data and **magnitude** space for magnitude data (the model always
predicts flux; the likelihood converts). All four samplers should therefore converge to the same
posterior.

``emcee`` is a core dependency (no extra needed). Sampling is seeded for reproducibility.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from ..likelihood import make_likelihood
from ..models import get_model
from .base import (
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)


def _log_prob(theta, names, prior, predict, times, bands, likelihood, scatter_name=None):
    """log-posterior for one parameter vector ``theta`` (ordered like ``names``).

    ``scatter_name`` routes that prior parameter to the likelihood as its extra-scatter term
    (:class:`~whisper_labia.likelihood.GaussianLikelihoodWithScatter`) instead of the model —
    the model's ``predict`` still receives the full params dict (models ignore unknown keys).
    """
    params = {nm: float(v) for nm, v in zip(names, theta)}
    lp = prior.log_prob(params)
    if not np.isfinite(lp):                       # outside the prior support -> forbidden
        return -np.inf
    model_flux = np.asarray(predict(params, times, bands), dtype=float)
    if scatter_name is not None:
        ll = likelihood.log_likelihood(model_flux, sigma_extra=params[scatter_name])
    else:
        ll = likelihood.log_likelihood(model_flux)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


class MCMCSampler(BaseSampler):
    """Affine-invariant ensemble MCMC (``emcee``). See the module docstring."""

    name = "mcmc"

    def fit(self, lc, model, prior=None, *, nwalkers=None, nsteps=5000, burnin=1000, thin=10,
            initial_guess=None, initial_scatter=1e-3, space="auto", likelihood="auto",
            seed=0, progress=False, moves=None, n_jobs=None) -> SamplerResult:
        """Fit ``lc`` with ``model`` via emcee MCMC, returning a :class:`SamplerResult`.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry errors (``flux_err`` / ``magnitude_err``) for the likelihood.
        model : str or Model
            Registered model name or a :class:`~whisper_labia.models.Model`.
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
        initial_guess : dict or array, optional
            Starting point; walkers are perturbed around it by ``initial_scatter``. If ``None``, walkers
            are initialised by sampling the prior (no guess needed -- like ABC/SNPE).
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
            An ``emcee`` move (or list of (move, weight)); default is emcee's stretch move.
        n_jobs : int, optional
            Worker processes for parallel likelihood evaluation (emcee ``pool``). Worth it only when
            one likelihood call is expensive (e.g. a ~0.1 s radiative-transfer model); for cheap
            analytic models the process overhead dominates. Default ``None`` = serial.

        Returns
        -------
        SamplerResult
            Posterior samples + summary, ``best_params`` (max-posterior draw), exact Gaussian
            ``max_log_likelihood`` / ``aic`` / ``bic``, and diagnostics in ``info`` (acceptance fraction,
            autocorrelation time). The trained ``emcee.EnsembleSampler`` is attached as
            ``result.emcee_sampler``.
        """
        import emcee

        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        lik = make_likelihood(lc, kind=likelihood, space=space)   # reuse the shared likelihood layer
        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        predict = model.predict
        names = list(prior.names)
        ndim = len(names)
        k, n = ndim, int(len(times))

        nwalkers = int(nwalkers) if nwalkers else max(2 * ndim + 2, 4 * ndim)
        nwalkers += nwalkers % 2                                   # emcee's red-blue split needs even
        if nwalkers < 2 * ndim:
            raise ValueError(f"nwalkers ({nwalkers}) must be >= 2*ndim ({2 * ndim}).")

        rng = np.random.default_rng(int(seed))
        if initial_guess is not None:
            g = np.array([float(initial_guess[nm]) if isinstance(initial_guess, dict)
                          else float(initial_guess[i]) for i, nm in enumerate(names)], dtype=float)
            p0 = g + initial_scatter * rng.standard_normal((nwalkers, ndim))
        else:                                                     # spread walkers across the prior
            p0 = np.array([[prior.distributions[nm].sample(rng) for nm in names]
                           for _ in range(nwalkers)], dtype=float)

        # A prior parameter named after the likelihood's scatter term (GaussianLikelihoodWithScatter)
        # is a LIKELIHOOD parameter: routed to log_likelihood(sigma_extra=...), sampled like the rest.
        scatter_name = getattr(lik, "scatter_param", None)
        scatter_name = scatter_name if (scatter_name and scatter_name in names) else None

        pool = None
        if n_jobs and int(n_jobs) > 1:
            import multiprocessing
            pool = multiprocessing.get_context("fork").Pool(int(n_jobs))
        try:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, _log_prob, moves=moves, pool=pool,
                args=(names, prior, predict, times, bands, lik, scatter_name))
            sampler._random.seed(int(seed))                       # seed emcee's proposal RNG

            t0 = time.perf_counter()
            sampler.run_mcmc(p0, int(nsteps), progress=progress)
            runtime = time.perf_counter() - t0
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
        flat_lp = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
        samples = pd.DataFrame(flat, columns=names)

        # AIC/BIC use the maximum *likelihood*, not the maximum *posterior*. log_prob = prior + ll, so
        # recover the per-draw log-likelihood ll = log_prob - prior.log_prob and take its argmax. (For a
        # flat/Uniform prior these coincide; for a LogUniform prior they differ, which would bias AIC/BIC.)
        prior_lp = np.array([prior.log_prob({nm: float(flat[i, j]) for j, nm in enumerate(names)})
                             for i in range(flat.shape[0])], dtype=float)
        flat_ll = flat_lp - prior_lp
        best_idx = int(np.nanargmax(flat_ll)) if np.any(np.isfinite(flat_ll)) else int(np.argmax(flat_lp))
        best_params = {nm: float(flat[best_idx, j]) for j, nm in enumerate(names)}
        max_log_likelihood = float(flat_ll[best_idx])

        try:
            autocorr = float(np.nanmean(sampler.get_autocorr_time(tol=0)))
        except Exception:                                         # pragma: no cover - short chains
            autocorr = float("nan")
        # Convergence guard: warn if the chain is too short relative to its autocorrelation time, since
        # CIs and AIC/BIC from an unconverged / heavily autocorrelated chain are unreliable.
        converged = True
        if np.isfinite(autocorr) and autocorr > 0 and nsteps < 50.0 * autocorr:
            converged = False
            warnings.warn(f"MCMC may be under-converged: nsteps={nsteps} < 50*tau (tau~{autocorr:.0f}); "
                          "increase nsteps/burnin and re-check posterior CIs and AIC/BIC.", stacklevel=2)
        info = {
            "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
            "space": lik.space, "likelihood": type(lik).__name__,
            "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "mean_autocorr_time": autocorr, "n_samples_per_walker": int(flat.shape[0] // nwalkers),
            "converged": bool(converged), "seed": int(seed),
        }
        attach_band_metrics(info, lc, model.name, best_params, space)
        result = SamplerResult(
            sampler="mcmc", model=model.name, parameters=list(names), samples=samples,
            summary=summarize_posterior(samples, names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )
        result.emcee_sampler = sampler
        attach_predictive_metrics(result, lc, space)
        return result


def fit_MCMC(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via emcee MCMC. See :meth:`MCMCSampler.fit` for options."""
    return MCMCSampler().fit(lc, model, prior=prior, **kwargs)
