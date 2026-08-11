"""Approximate Bayesian Computation -- parallel rejection sampler.

Workflow: sample prior -> simulate light curve -> compute distance to data -> accept the closest.
Acceptance is by ``quantile`` (keep the best fraction; robust, default) or a fixed ``threshold``.
Simulations are split across processes (``n_jobs``); the model ``predict``, ``prior`` and
``distance`` must be picklable for ``n_jobs > 1`` (use ``n_jobs=1`` for closures/lambdas).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import warnings

from ..distance import chi2_distance, get_distance
from ..likelihood import make_likelihood
from ..models import get_model
from .base import (
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)


def _simulate_batch(predict, prior, distance, times, bands, obs_y, obs_err, indices, seed,
                    simulate_noise, model_in_space, scatter_name):
    """Simulate the given **global** simulation indices.

    Each index ``i`` draws from its own RNG stream ``default_rng([seed, i])``, so the set of draws is
    identical no matter how the indices are chunked across workers. This makes the result fully
    reproducible for a fixed ``seed`` **regardless of ``n_jobs``** (the machine's core count).

    ``model_in_space`` maps the model's flux prediction into the comparison space (identity for flux,
    flux→AB magnitude for magnitude space), so data, simulation, noise and distance all live in ONE
    space. With ``simulate_noise`` the simulation matches the generative model of the data — per-point
    white noise ``N(0, obs_err)`` (plus, when ``scatter_name`` is set, that prior parameter's extra
    scatter in quadrature, Villar+2017-style) is added to the prediction from this simulation's own
    stream, preserving reproducibility. Comparing *noisy* simulations to the noisy data is what makes
    ABC exact in the small-epsilon limit.
    """
    thetas = []
    distances = np.empty(len(indices), dtype=float)
    for j, idx in enumerate(indices):
        rng = np.random.default_rng([int(seed), int(idx)])
        theta = prior.sample(rng)
        sim = np.asarray(model_in_space(predict(theta, times, bands)), dtype=float)
        if simulate_noise:
            err = obs_err if scatter_name is None else np.sqrt(
                obs_err ** 2 + float(theta[scatter_name]) ** 2)
            sim = sim + rng.normal(0.0, err)
        distances[j] = distance(obs_y, obs_err, sim, bands)
        thetas.append(theta)
    return thetas, distances


def _worker(args):
    return _simulate_batch(*args)


class ABCSampler(BaseSampler):
    """Approximate Bayesian Computation by parallel rejection (see the module docstring)."""

    name = "abc"

    def fit(self, lc, model, prior=None, *, n_simulations=10000, quantile=0.01, threshold=None,
            distance=chi2_distance, simulate_noise=True, space="auto", scatter_param=None,
            n_jobs=None, seed=0):
        """Fit ``lc`` with ``model`` by rejection ABC, returning a :class:`SamplerResult`.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry flux errors (``flux_err``) for the chi-square distance.
        model : str or Model
            A registered model name or a :class:`~whisper_labia.models.Model`.
        prior : Prior, optional
            Parameter prior; defaults to the model's ``default_prior`` (``ValueError`` if neither).
        n_simulations : int, default 10000
            Number of prior draws / simulations.
        quantile : float, default 0.01
            Acceptance fraction — keep the closest ``quantile`` of draws (robust default).
        threshold : float, optional
            Fixed acceptance distance epsilon; overrides ``quantile`` when given.
            **Scale warning:** with ``simulate_noise=True`` (default) distances include the simulation
            noise — ``E[D] ≈ χ² + n_points`` with roughly doubled per-point variance — so a threshold
            calibrated against the old noiseless χ² scale will accept (near) nothing. Re-derive fixed
            thresholds, or use ``quantile`` (which adapts automatically).
        distance : callable, default :func:`chi2_distance`
            ``f(obs_flux, obs_err, sim_flux, bands) -> float``. Must be picklable for ``n_jobs > 1``.
        simulate_noise : bool, default True
            Add per-point white noise ``N(0, flux_err)`` to each simulation so it matches the
            generative model of the data (measurement noise included). This is what makes ABC exact
            as epsilon → 0 and keeps the posterior width **calibrated**; ``False`` restores the old
            noiseless-simulator behaviour (a hard cut on a likelihood shell, mis-shaped width).
        space : {'auto', 'flux', 'magnitude'}, default 'auto'
            Comparison space: data, simulations, noise and distance all live here (``'auto'`` follows
            the data's ``data_mode``, like the likelihood-based samplers).
        scatter_param : str, optional
            Name of a prior parameter to treat as a free **extra-scatter** term (Villar+2017): each
            simulation's noise becomes ``N(0, sqrt(err² + scatter²))`` with that draw's value, so ABC
            fits the same scatter-augmented generative model as
            :class:`~whisper_labia.likelihood.GaussianLikelihoodWithScatter`. Requires
            ``simulate_noise=True``; the parameter is ignored by ``model.predict`` (models read only
            the keys they know) and appears in the posterior like any other.
        n_jobs : int, optional
            Worker processes (default ``min(os.cpu_count(), 8)``). **The result is independent of
            ``n_jobs``** for a fixed ``seed`` — parallelism affects speed only, not the science.
        seed : int, default 0
            Base RNG seed; the full posterior is reproducible given ``(seed, n_simulations)``.

        Returns
        -------
        SamplerResult
            Posterior samples + summary, ``best_params``, and ``max_log_likelihood``/``aic``/``bic``
            (chi-square = -2 ln L for a Gaussian likelihood).
        """
        model = get_model(model)
        distance = get_distance(distance)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")

        # The comparison space: data, simulations, noise and distance all live here. The Gaussian
        # likelihood object supplies the space-resolved observation (y, err) and the flux->space map.
        from ..likelihood import GaussianLikelihood
        lik_space = GaussianLikelihood(lc, space=space)
        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        obs_y = np.asarray(lik_space.y, dtype=float)
        obs_err = np.asarray(lik_space.sigma, dtype=float)
        predict = model.predict
        names = list(prior.names)               # sampled parameters (may include a scatter nuisance)
        if scatter_param is not None:
            if scatter_param not in names:
                raise ValueError(f"scatter_param {scatter_param!r} is not in the prior ({names}).")
            if not simulate_noise:
                raise ValueError("scatter_param requires simulate_noise=True (the scatter enters "
                                 "the generative noise).")

        n_jobs = n_jobs or min(os.cpu_count() or 1, 8)
        n_jobs = max(1, min(int(n_jobs), n_simulations))
        # Split the GLOBAL simulation indices into contiguous chunks; each index keeps its own RNG
        # stream, so the union of draws (and their order) is identical for any n_jobs.
        index_chunks = [c for c in np.array_split(np.arange(n_simulations), n_jobs) if len(c)]

        t0 = time.perf_counter()
        thetas = []
        dist_parts = []
        if n_jobs == 1:
            th, ds = _simulate_batch(predict, prior, distance, times, bands,
                                     obs_y, obs_err, index_chunks[0], seed, simulate_noise,
                                     lik_space.model_in_space, scatter_param)
            thetas.extend(th)
            dist_parts.append(ds)
        else:
            args = [(predict, prior, distance, times, bands, obs_y, obs_err, idx, seed,
                     simulate_noise, lik_space.model_in_space, scatter_param)
                    for idx in index_chunks]
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for th, ds in executor.map(_worker, args):
                    thetas.extend(th)
                    dist_parts.append(ds)
        runtime = time.perf_counter() - t0

        distances = np.concatenate(dist_parts)
        epsilon = float(np.quantile(distances, quantile)) if threshold is None else float(threshold)
        keep = distances <= epsilon
        if not keep.any():
            warnings.warn(
                f"ABC accepted 0 of {n_simulations} draws at epsilon={epsilon:g}; the posterior is "
                "empty. best_params / AIC / BIC reflect the single closest draw only.", stacklevel=2)
        accepted = [thetas[i] for i in np.nonzero(keep)[0]]
        samples = pd.DataFrame(accepted, columns=names)     # ALL sampled params (incl. any scatter)
        samples["distance"] = distances[keep]

        # Model-selection metrics. The chi-square distance drops the Gaussian normalisation; to make
        # AIC/BIC **comparable across samplers** (MCMC/SNPE use the exact Gaussian log-likelihood in
        # the data's natural space), evaluate that same likelihood here too — the scatter-augmented
        # one when a scatter parameter is fitted, with each draw's own scatter value. With
        # simulate_noise the noisy-distance argmin is the *luckiest simulation-noise draw*, not the
        # best theta — so the best fit is chosen by exact log-likelihood over the accepted draws
        # (deterministically capped scan), never by the noisy distance.
        chi2_min = float(distances.min())     # noisy scale when simulate_noise: E[D] ~ chi2 + n_points
        k, n = len(names), lc.n_points
        lik = (make_likelihood(lc, kind="gaussian_scatter", space=space,
                               scatter_param=scatter_param) if scatter_param
               else make_likelihood(lc, space=space))
        cand = accepted if accepted else [thetas[int(np.argmin(distances))]]
        scan = cand[:2000]

        def _logl(th):
            mf = np.asarray(predict(th, times, bands), dtype=float)
            if scatter_param:
                return lik.log_likelihood(mf, sigma_extra=float(th[scatter_param]))
            return lik.log_likelihood(mf)

        logls = np.array([_logl(th) for th in scan], dtype=float)
        best_idx = int(np.nanargmax(logls)) if np.any(np.isfinite(logls)) else 0
        best = {p: float(scan[best_idx][p]) for p in names}
        max_log_likelihood = float(logls[best_idx])
        info = {
            "n_simulations": int(n_simulations),
            "n_accepted": int(keep.sum()),
            "acceptance_rate": float(keep.mean()),
            "epsilon": epsilon,
            "quantile": None if threshold is not None else float(quantile),
            "simulate_noise": bool(simulate_noise),
            "space": lik_space.space,
            "scatter_param": scatter_param,
            "n_jobs": int(n_jobs),
            "distance": getattr(distance, "__name__", str(distance)),
            "likelihood_space": lik.space,
        }
        attach_band_metrics(info, lc, model.name, best, space)
        result = SamplerResult(
            sampler="abc", model=model.name, parameters=names,
            samples=samples, summary=summarize_posterior(samples, names),
            best_params=best, n_data=n, n_params=k, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )
        attach_predictive_metrics(result, lc, space)
        return result


def fit_ABC(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via rejection ABC. See :meth:`ABCSampler.fit` for options."""
    return ABCSampler().fit(lc, model, prior=prior, **kwargs)
