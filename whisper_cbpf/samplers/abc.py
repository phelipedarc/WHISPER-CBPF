"""Approximate Bayesian Computation -- parallel rejection sampler.

Workflow: sample prior -> simulate light curve -> compute distance to data -> accept the closest.
Acceptance is by ``quantile`` (keep the best fraction; robust, default) or a fixed ``threshold``.
Simulations run serially by default. ``n_jobs > 1`` splits them across ``spawn``ed processes and
needs the model ``predict``, ``prior`` and ``distance`` to be picklable -- and is measurably slower
at ordinary budgets, so it is opt-in. See the ``n_jobs`` parameter.
"""
from __future__ import annotations

import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

#: Worker start method for the ``n_jobs > 1`` pools below. MUST be "spawn", not the Linux default
#: "fork".
#:
#: Any photometric JAX model has an initialised XLA runtime in the parent -- measured at 254 threads
#: -- and ``fork()`` carries only the calling thread into the child. A mutex held by any of the other
#: 253 at fork time stays locked forever, so every worker blocks in ``futex_wait_queue`` and the fit
#: hangs rather than failing. Measured: 8 workers at 0.0% CPU holding 373 MB each, indefinitely, on
#: a fit that takes 3.4 s at ``n_jobs=1``.
#:
#: "spawn" starts each worker from a fresh interpreter, so there is no inherited lock to deadlock on.
#: It also *requires* the payload to be picklable, which is why the photometric factories build their
#: ``predict`` as a module-level class rather than a closure.
_MP_CONTEXT = multiprocessing.get_context("spawn")


from ..distance import chi2_distance, get_distance
from ..likelihood import make_likelihood
from ..models import get_model
from .jax import _diagnostics as _dg                                # numpy-only; no JAX import
from .base import (
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
    _warn_user,
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


def _logl_batch(predict, lik, times, bands, thetas, scatter_name):
    """Exact log-likelihood of each theta -- the best-fit scan, split across workers like the
    simulations. ``scatter_name`` routes that draw's own scatter value to ``sigma_extra``."""
    logls = np.empty(len(thetas), dtype=float)
    for j, th in enumerate(thetas):
        mf = np.asarray(predict(th, times, bands), dtype=float)
        logls[j] = (lik.log_likelihood(mf, sigma_extra=float(th[scatter_name])) if scatter_name
                    else lik.log_likelihood(mf))
    return logls


def _logl_worker(args):
    return _logl_batch(*args)


class ABCSampler(BaseSampler):
    """Approximate Bayesian Computation by parallel rejection (see the module docstring)."""

    name = "abc"

    def fit(self, lc, model, prior=None, *, n_simulations=10000, quantile=0.01, threshold=None,
            distance=chi2_distance, simulate_noise=True, space="auto", scatter_param=None,
            n_jobs=None, seed=0, max_logl_scan=None):
        """Fit ``lc`` with ``model`` by rejection ABC, returning a :class:`SamplerResult`.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry flux errors (``flux_err``) for the chi-square distance.
        model : str or Model
            A registered model name or a :class:`~whisper_cbpf.models.Model`.
        prior : Prior, optional
            Parameter prior; defaults to the model's ``default_prior`` (``ValueError`` if neither).
            Any distribution with ``sample(rng)`` works (``Normal``, ``TruncatedNormal`` included).
            A ``Fixed`` parameter is drawn as its value: a constant column in ``samples``, listed
            in ``info["fixed"]``, and not counted in AIC/BIC (``n_params``).
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
        distance : callable or str, default :func:`chi2_distance`
            ``f(obs_flux, obs_err, sim_flux, bands) -> float``, or a registered name. Must be
            picklable for ``n_jobs > 1``. ``"max_abs_z"`` with ``threshold=k`` accepts a draw only
            if every point is within ``k`` of its own error (the v3 rule is ``threshold=5,
            simulate_noise=False, space="magnitude"``); see
            :func:`whisper_cbpf.distance.max_abs_z_distance`.
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
            :class:`~whisper_cbpf.likelihood.GaussianLikelihoodWithScatter`. Requires
            ``simulate_noise=True``; the parameter is ignored by ``model.predict`` (models read only
            the keys they know) and appears in the posterior like any other.
        n_jobs : int, optional
            Worker processes. **Default 1 (serial), deliberately** — workers are started with
            ``spawn``, which pays a fresh interpreter, a JAX import and an XLA compile each, and at
            ordinary budgets that costs more than it saves: measured 3.5x to 8.3x SLOWER than serial
            on AT2017GFO, with bit-identical AICs. Raise it only when the simulation count is large
            enough to amortise that. A script passing ``n_jobs > 1`` needs an
            ``if __name__ == "__main__":`` guard, because ``spawn`` re-imports ``__main__``.
            **The result is independent of ``n_jobs``** for a fixed ``seed`` — parallelism affects
            speed only, not the science.
        seed : int, default 0
            Base RNG seed; the full posterior is reproducible given ``(seed, n_simulations)``.
        max_logl_scan : int, optional
            Cap on the accepted draws re-scored by exact log-likelihood to pick ``best_params``
            (one forward-model call each, on the same ``n_jobs`` workers as the simulations).
            **Default ``None`` scores every accepted draw.** When a cap binds, the draws with the
            LOWEST distance are scored, never the first ones accepted. ``info['logl_scan_n']`` and
            ``info['logl_scan_capped']`` record what was done.

        Returns
        -------
        SamplerResult
            Posterior samples + summary, ``best_params``, and ``max_log_likelihood``/``aic``/``bic``
            (chi-square = -2 ln L for a Gaussian likelihood). If no draw was accepted,
            ``best_params`` is the closest *rejected* draw (``info['best_params_source']``) and the
            predictive metrics are skipped (``info['predictive_metrics_skipped']``).
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
        # A Fixed parameter is drawn as its value (Fixed.sample) and kept as a constant column, but
        # it is not a fitted parameter, so it is not counted in AIC/BIC.
        _, free_names, fixed = _dg.split_fixed(prior, names, "abc")
        if scatter_param is not None:
            if scatter_param not in names:
                raise ValueError(f"scatter_param {scatter_param!r} is not in the prior ({names}).")
            if not simulate_noise:
                raise ValueError("scatter_param requires simulate_noise=True (the scatter enters "
                                 "the generative noise).")

        # DEFAULT IS SERIAL, and that is a measured choice rather than timidity.
        #
        # Every photometric JAX model must now be sent to workers by `spawn` (see _MP_CONTEXT), and
        # spawn pays a fresh interpreter, a JAX import and an XLA compile PER WORKER. Measured on
        # AT2017GFO at these budgets, n_jobs=8 was slower than serial in every case tried:
        #
        #     kilonova_one_jax  abc      14.3 s vs  4.1 s   (0.28x)
        #     tde_gaussianrise  abc      12.3 s vs  1.7 s   (0.14x)
        #     tde_gaussianrise  abc_smc  23.3 s vs  2.8 s   (0.12x)
        #
        # with AICs bit-identical, so the parallelism is correct and simply not worth its startup.
        # It wins only once the simulation count is large enough to amortise that, which is a
        # judgement about YOUR budget -- hence opt-in. Pass n_jobs explicitly to use it, and note
        # that spawn re-imports __main__, so a script doing so needs an
        # `if __name__ == "__main__":` guard.
        n_jobs = 1 if n_jobs is None else int(n_jobs)
        n_jobs = max(1, min(int(n_jobs), n_simulations))
        # Split the GLOBAL simulation indices into contiguous chunks; each index keeps its own RNG
        # stream, so the union of draws (and their order) is identical for any n_jobs.
        index_chunks = [c for c in np.array_split(np.arange(n_simulations), n_jobs) if len(c)]

        t0 = time.perf_counter()
        thetas = []
        dist_parts = []
        # One pool for the simulations AND the best-fit scan below: spawn's start-up is the cost
        # that makes n_jobs > 1 opt-in (see above), so the scan reuses the workers instead of
        # paying it twice.
        pool = (ProcessPoolExecutor(max_workers=n_jobs, mp_context=_MP_CONTEXT)
                if n_jobs > 1 else None)
        try:
            if pool is None:
                th, ds = _simulate_batch(predict, prior, distance, times, bands,
                                         obs_y, obs_err, index_chunks[0], seed, simulate_noise,
                                         lik_space.model_in_space, scatter_param)
                thetas.extend(th)
                dist_parts.append(ds)
            else:
                args = [(predict, prior, distance, times, bands, obs_y, obs_err, idx, seed,
                         simulate_noise, lik_space.model_in_space, scatter_param)
                        for idx in index_chunks]
                for th, ds in pool.map(_worker, args):
                    thetas.extend(th)
                    dist_parts.append(ds)
            runtime = time.perf_counter() - t0

            distances = np.concatenate(dist_parts)
            epsilon = (float(np.quantile(distances, quantile)) if threshold is None
                       else float(threshold))
            keep = distances <= epsilon
            if not keep.any():
                _warn_user(
                    f"ABC accepted 0 of {n_simulations} draws at epsilon={epsilon:g}; the "
                    "posterior is empty. best_params / AIC / BIC reflect the single closest draw, "
                    "which was NOT accepted, and the predictive metrics are skipped.")
            accepted = [thetas[i] for i in np.nonzero(keep)[0]]
            samples = pd.DataFrame(accepted, columns=names)  # ALL sampled params (incl. scatter)
            samples["distance"] = distances[keep]

            # Model-selection metrics. The chi-square distance drops the Gaussian normalisation; to
            # make AIC/BIC **comparable across samplers** (MCMC/SNPE use the exact Gaussian
            # log-likelihood in the data's natural space), evaluate that same likelihood here too —
            # the scatter-augmented one when a scatter parameter is fitted, with each draw's own
            # scatter value. With simulate_noise the noisy-distance argmin is the *luckiest
            # simulation-noise draw*, not the best theta — so the best fit is chosen by exact
            # log-likelihood over the accepted draws, never by the noisy distance.
            chi2_min = float(distances.min())  # noisy when simulate_noise: E[D] ~ chi2 + n_points
            k, n = len(free_names), lc.n_points
            lik = (make_likelihood(lc, kind="gaussian_scatter", space=space,
                                   scatter_param=scatter_param) if scatter_param
                   else make_likelihood(lc, space=space))
            # EVERY accepted draw by default. This used to be the first 2000 in acceptance order,
            # an arbitrary subset: on real supernova fits ln L_max came out up to 3.4 low and BIC
            # up to 6.9 high. A cap, when given, keeps the lowest-distance draws -- the most
            # promising -- and at least one, so the ranking can never come back empty.
            cand = np.nonzero(keep)[0] if keep.any() else np.array([int(np.argmin(distances))])
            n_cand = len(cand)
            cap = n_cand if max_logl_scan is None else max(int(max_logl_scan), 1)
            if n_cand > cap:
                cand = cand[np.argsort(distances[cand], kind="stable")[:cap]]
            scan = [thetas[i] for i in cand]
            if pool is None:
                logls = _logl_batch(predict, lik, times, bands, scan, scatter_param)
            else:
                parts = [c for c in np.array_split(np.arange(len(scan)), n_jobs) if len(c)]
                logls = np.concatenate(list(pool.map(
                    _logl_worker, [(predict, lik, times, bands, [scan[j] for j in c], scatter_param)
                                   for c in parts])))
        finally:
            if pool is not None:
                pool.shutdown()
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
            "logl_scan_n": int(len(scan)),
            "logl_scan_capped": bool(len(scan) < n_cand),
            "best_params_source": "accepted_draws" if accepted else "closest_rejected_draw",
            "fixed": dict(fixed),
        }
        attach_band_metrics(info, lc, model, best, space)
        aic, bic = aic_bic(max_log_likelihood, k, n)
        result = SamplerResult(
            sampler="abc", model=model.name, parameters=names,
            samples=samples, summary=summarize_posterior(samples, names),
            best_params=best, n_data=n, n_params=k, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=max_log_likelihood,
            aic=aic, bic=bic,
        )
        attach_predictive_metrics(result, lc, space, model=model)
        return result


def fit_ABC(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via rejection ABC. See :meth:`ABCSampler.fit` for options.

    Parameters
    ----------
    lc : LightCurve
        The data (detections only: ABC cannot use an upper limit).
    model : str or Model, default "flare"
    prior : Prior, optional
        Default: the model's.
    **kwargs
        The sampler's settings (``n_simulations``, ``quantile``, ``threshold``, ``distance``,
        ``seed``, ...).

    Returns
    -------
    SamplerResult

    Examples
    --------
    Accept a draw only if every point is within 5 sigma (``distance="max_abs_z"``):

    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t, b = np.linspace(0.5, 30.0, 30), np.array(["r"] * 30)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, b)
    >>> lc = wp.LightCurve(time=t, band=b, flux=flux, flux_err=np.full(30, 0.25))
    >>> res = wp.fit_ABC(lc, "flare", n_simulations=5000, distance="max_abs_z", threshold=5.0,
    ...                  simulate_noise=False)
    >>> res.info["n_accepted"] > 0 and bool((res.samples["distance"] <= 5.0).all())
    True
    """
    return ABCSampler().fit(lc, model, prior=prior, **kwargs)
