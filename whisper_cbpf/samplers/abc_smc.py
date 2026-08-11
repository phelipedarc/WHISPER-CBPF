"""ABC-SMC (Sequential Monte Carlo) — **importance-weighted** (Beaumont et al. 2009; Toni et al. 2009).

Refine a particle population over rounds of shrinking acceptance threshold (epsilon):

* round 0 samples the prior (uniform weights);
* each later round resamples a parent from the **weighted** population, perturbs it with a Gaussian
  kernel whose (diagonal) covariance is twice the weighted population variance (the Toni 2009 choice),
  rejects perturbed particles outside the prior, and keeps those under that round's (tighter) epsilon;
* each kept particle gets an **importance weight** ``w_i ∝ π(θ_i) / Σ_j w_j^{prev} K(θ_i | θ_j^{prev})``.
  These weights correct the bias the perturbation kernel would otherwise introduce — without them the
  population converges to a distorted (not the ABC-posterior) distribution.

The returned ``samples`` are the final weighted population **resampled to equal weights**, so downstream
summaries/plots need no weight handling. Far more simulation-efficient than flat rejection at tight
thresholds.

Epsilon schedule: pass an explicit ``epsilon_schedule`` list, or leave it ``None`` for an **adaptive**
schedule where the next round's epsilon = the ``quantile`` (default 0.5) of the current round's
accepted distances.

Reproducibility: each proposal is seeded by its global ``(seed, round, attempt)`` index, so the
accepted population (and the final resample) is identical for any ``n_jobs``.
"""
from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import logsumexp

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

#: Floor on a parameter's perturbation std so a fixed/zero-variance parameter still has a valid kernel.
_PERTURB_FLOOR = 1e-12


def _kernel_std(parents_arr, weights):
    """Diagonal Gaussian kernel std per parameter = sqrt(2 * weighted population variance) (Toni 2009)."""
    mean = np.average(parents_arr, axis=0, weights=weights)
    var = np.average((parents_arr - mean) ** 2, axis=0, weights=weights)
    std = np.sqrt(2.0 * var)
    return np.where(std > 0, std, _PERTURB_FLOOR)


def _smc_batch(args):
    """Try the given **global attempt indices** for one round; each index owns its RNG stream.

    Round 0 draws from the prior; later rounds resample a parent from the *weighted* previous population
    and perturb it with the diagonal Gaussian kernel. Seeding by the global attempt index (not the
    worker) makes the proposals — and hence the population — independent of ``n_jobs``.
    """
    (round_idx, parents_arr, parent_weights, param_names, kernel_std, prior,
     predict, distance, times, bands, obs_y, obs_err, epsilon, indices, seed,
     simulate_noise, model_in_space, scatter_name) = args
    accepted = []
    for a in indices:
        rng = np.random.default_rng([int(seed), int(round_idx), int(a)])
        if round_idx == 0:
            theta = prior.sample(rng)
        else:
            j = int(rng.choice(len(parent_weights), p=parent_weights))   # weighted resample
            vec = parents_arr[j] + rng.normal(0.0, kernel_std)           # diagonal Gaussian perturb
            theta = {nm: float(v) for nm, v in zip(param_names, vec)}
            if not np.isfinite(prior.log_prob(theta)):                   # outside prior support -> reject
                continue
        sim = np.asarray(model_in_space(predict(theta, times, bands)), dtype=float)
        if simulate_noise:                       # generative match: same per-point noise as the data
            err = obs_err if scatter_name is None else np.sqrt(
                obs_err ** 2 + float(theta[scatter_name]) ** 2)   # + fitted extra scatter (Villar+17)
            sim = sim + rng.normal(0.0, err)
        d = distance(obs_y, obs_err, sim, bands)
        if d < epsilon:
            rec = dict(theta)
            rec["distance"] = d
            rec["_attempt"] = int(a)
            accepted.append(rec)
    return accepted, len(indices)


def _importance_log_weights(theta_new, param_names, parents_arr, parent_log_weights, kernel_std, prior):
    """Log importance weights ``ln w_i = ln π(θ_i) - ln Σ_j w_j K(θ_i|θ_j)`` (normalised), in log-space."""
    new_arr = np.array([[t[p] for p in param_names] for t in theta_new], dtype=float)   # (M, D)
    log_k_norm = -np.sum(np.log(kernel_std)) - 0.5 * len(param_names) * np.log(2.0 * np.pi)
    log_w = np.empty(len(theta_new), dtype=float)
    for i in range(len(theta_new)):
        diff = (new_arr[i] - parents_arr) / kernel_std                  # (N, D)
        log_kernel = log_k_norm - 0.5 * np.sum(diff * diff, axis=1)     # (N,)  K(θ_i | θ_j)
        denom = logsumexp(parent_log_weights + log_kernel)             # ln Σ_j w_j K(θ_i|θ_j)
        log_w[i] = float(prior.log_prob(theta_new[i])) - denom
    return log_w - logsumexp(log_w)                                     # normalise


class ABCSMCSampler(BaseSampler):
    """Importance-weighted Sequential Monte Carlo ABC over rounds of shrinking epsilon."""

    name = "abc_smc"

    def fit(self, lc, model, prior=None, *, n_particles=500, n_rounds=5, epsilon_schedule=None,
            quantile=0.5, min_epsilon=None, simulate_noise=True, space="auto", scatter_param=None,
            perturbation_scale=0.1, distance=chi2_distance, n_jobs=None, seed=0,
            max_attempts_per_round=None):
        """Fit ``lc`` with ``model`` via importance-weighted ABC-SMC, returning a :class:`SamplerResult`.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry ``flux_err`` for the chi-square distance.
        model : str or Model
            Registered model name or a :class:`~whisper_labia.models.Model`.
        prior : Prior, optional
            Parameter prior; defaults to the model's ``default_prior`` (``ValueError`` if neither).
        n_particles : int, default 500
            Accepted-particle population size kept each round.
        n_rounds : int, default 5
            Number of SMC rounds (overridden by ``len(epsilon_schedule)`` when that is given).
        epsilon_schedule : list of float, optional
            Explicit per-round acceptance thresholds; if ``None``, epsilon is adaptive
            (next epsilon = ``quantile`` of the current round's accepted distances).
            **Scale warning:** with ``simulate_noise=True`` (default) distances include the simulation
            noise (``E[D] ≈ χ² + n_points``); schedules or float ``min_epsilon`` values calibrated
            against the old noiseless χ² scale sit below the achievable floor and will exhaust
            ``max_attempts_per_round``. Re-derive them, or rely on the adaptive quantile schedule.
        quantile : float, default 0.5
            Adaptive-epsilon quantile (ignored when ``epsilon_schedule`` is given).
        min_epsilon : float or ``"auto"``, optional
            Floor on the adaptive epsilon so it is not driven all the way to the best achievable
            distance ``chi2_min``. A hard chi-square cutoff at ``chi2_min`` accepts only a razor-thin
            shell around the best fit, collapsing the posterior to (essentially) the MLE and yielding an
            **overconfident** posterior whose width is far below the true parameter uncertainty.
            ``"auto"`` (recommended) floors epsilon at ``chi2_min + 2·(k+2)`` (``k`` = #parameters),
            which reproduces the Gaussian posterior width; a float sets a fixed floor. Ignored when
            ``epsilon_schedule`` is given (default ``None`` = no floor, epsilon → ``chi2_min``).
            With ``simulate_noise=True`` the distances carry an irreducible noise floor (E[χ²] ≈ 2N at
            the truth), so epsilon self-regulates and the floor is a harmless extra guard.
        simulate_noise : bool, default True
            Add per-point white noise ``N(0, flux_err)`` to each simulation so it matches the
            generative model of the data — the smooth acceptance kernel this induces is what makes
            ABC-SMC's posterior **width calibrated** (a noiseless simulator under a hard cut targets a
            likelihood shell). ``False`` restores the old behaviour.
        space : {'auto', 'flux', 'magnitude'}, default 'auto'
            Comparison space: data, simulations, noise and distance all live here (``'auto'`` follows
            the data's ``data_mode``, like the likelihood-based samplers).
        scatter_param : str, optional
            Prior parameter treated as a free **extra-scatter** term (Villar+2017): each simulation's
            noise becomes ``N(0, sqrt(err² + scatter²))`` with that particle's value, matching
            :class:`~whisper_labia.likelihood.GaussianLikelihoodWithScatter`. It is perturbed and
            weighted like every other particle dimension. Requires ``simulate_noise=True``.
        perturbation_scale : float, default 0.1
            Retained for backward compatibility; the kernel covariance is now set adaptively from the
            weighted population (Toni 2009), so this value is unused.
        distance : callable, default :func:`chi2_distance`
            ``f(obs_flux, obs_err, sim_flux, bands) -> float``; picklable for ``n_jobs > 1``.
        n_jobs : int, optional
            Worker processes (default ``min(os.cpu_count(), 8)``). The accepted population is
            **independent of ``n_jobs``** for a fixed ``seed`` (parallelism affects speed only).
        seed : int, default 0
            Base RNG seed; the result is reproducible given ``(seed, n_particles, schedule)``.
        max_attempts_per_round : int, optional
            Safety cap on proposals per round (default ``max(200*n_particles, 200000)``); a round that
            hits it warns and proceeds with however many particles it accepted.

        Returns
        -------
        SamplerResult
            Equal-weight posterior samples + summary, ``best_params``, ``aic``/``bic`` (from the exact
            Gaussian log-likelihood at the best fit — comparable across samplers), and per-round
            diagnostics in ``info['rounds']``.
        """
        model = get_model(model)
        distance = get_distance(distance)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        # Comparison space: data, simulations, noise and distance all live here (see ABCSampler).
        from ..likelihood import GaussianLikelihood
        lik_space = GaussianLikelihood(lc, space=space)
        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        obs_y = np.asarray(lik_space.y, dtype=float)
        obs_err = np.asarray(lik_space.sigma, dtype=float)
        predict = model.predict
        params = list(prior.names)              # particle dimensions (may include a scatter nuisance)
        if scatter_param is not None:
            if scatter_param not in params:
                raise ValueError(f"scatter_param {scatter_param!r} is not in the prior ({params}).")
            if not simulate_noise:
                raise ValueError("scatter_param requires simulate_noise=True (the scatter enters "
                                 "the generative noise).")

        # Epsilon floor. A hard chi-square cutoff at epsilon = chi2_min + c accepts the ellipsoid
        # {Δθ : Δθ'·Hess(chi2)·Δθ < c}; matching its marginal width to the Gaussian posterior needs
        # c ≈ 2·(k+2) (uniform-in-ellipsoid, k params). Driving epsilon to chi2_min (c→0) instead
        # collapses the posterior to the MLE -> overconfident. ``"auto"`` floors epsilon adaptively at
        # (running best distance) + 2·(k+2); a float sets a fixed floor.
        auto_floor = min_epsilon == "auto"
        eps_floor = None if (auto_floor or min_epsilon is None) else float(min_epsilon)
        floor_c = 2.0 * (len(params) + 2)
        if epsilon_schedule is not None:
            epsilon_schedule = [float(e) for e in epsilon_schedule]
            n_rounds = len(epsilon_schedule)
        n_jobs = max(1, int(n_jobs or min(os.cpu_count() or 1, 8)))
        if max_attempts_per_round is None:
            max_attempts_per_round = max(200 * n_particles, 200_000)
        batch = max(n_particles // n_jobs, 200)

        t0 = time.perf_counter()
        parents_arr = None            # (N, D) population parameters of the previous round
        parent_weights = None         # (N,)   normalised linear weights (for resampling)
        parent_log_weights = None     # (N,)   normalised log weights (for the weight recursion)
        kernel_std = None
        population = []
        round_info = []
        total_attempts = 0
        epsilon = epsilon_schedule[0] if epsilon_schedule is not None else np.inf
        pool = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        try:
            for round_idx in range(n_rounds):
                if epsilon_schedule is not None:
                    epsilon = epsilon_schedule[round_idx]
                accepted, next_attempt = [], 0
                while len(accepted) < n_particles and next_attempt < max_attempts_per_round:
                    block = np.arange(next_attempt, next_attempt + n_jobs * batch)
                    idx_chunks = [c for c in np.array_split(block, n_jobs) if len(c)]
                    args = [(round_idx, parents_arr, parent_weights, params, kernel_std, prior,
                             predict, distance, times, bands, obs_y, obs_err, epsilon, idx, seed,
                             simulate_noise, lik_space.model_in_space, scatter_param)
                            for idx in idx_chunks]
                    results = [_smc_batch(args[0])] if pool is None else list(pool.map(_smc_batch, args))
                    for acc, _ in results:
                        accepted.extend(acc)
                    next_attempt += n_jobs * batch
                attempts = next_attempt
                total_attempts += attempts
                # Order by global attempt index so the kept population is identical for any n_jobs.
                accepted.sort(key=lambda r: r["_attempt"])
                if len(accepted) < n_particles:
                    warnings.warn(
                        f"ABC-SMC round {round_idx + 1}: only {len(accepted)}/{n_particles} accepted "
                        f"after {attempts} attempts (epsilon={epsilon:g}).", stacklevel=2)
                population = [{k: v for k, v in r.items() if k != "_attempt"}
                             for r in accepted[:n_particles]]
                theta_pop = [{k: p[k] for k in params} for p in population]
                pop_arr = np.array([[t[p] for p in params] for t in theta_pop], dtype=float) \
                    if theta_pop else np.empty((0, len(params)))

                # Importance weights: uniform at round 0, otherwise the SMC recursion (corrects the kernel).
                if round_idx == 0 or parents_arr is None or len(theta_pop) == 0:
                    log_w = np.full(len(theta_pop), -np.log(max(len(theta_pop), 1)))
                else:
                    log_w = _importance_log_weights(theta_pop, params, parents_arr,
                                                    parent_log_weights, kernel_std, prior)
                weights = np.exp(log_w - logsumexp(log_w)) if len(log_w) else np.array([])

                dists = (np.array([p["distance"] for p in population]) if population
                         else np.array([np.inf]))
                ess = float(1.0 / np.sum(weights ** 2)) if len(weights) else 0.0
                round_info.append({
                    "round": round_idx + 1,
                    "epsilon": (float(epsilon) if np.isfinite(epsilon) else None),
                    "n_accepted": len(population),
                    "attempts": int(attempts),
                    "acceptance_rate": float(len(population) / attempts) if attempts else 0.0,
                    "best_distance": float(dists.min()),
                    "effective_sample_size": ess,
                })
                # Prepare the next round: weighted population + adaptive kernel covariance.
                parents_arr = pop_arr
                parent_log_weights = log_w
                parent_weights = weights
                kernel_std = _kernel_std(pop_arr, weights) if len(weights) else None
                if epsilon_schedule is None and round_idx + 1 < n_rounds:
                    epsilon = float(np.quantile(dists, quantile))
                    if auto_floor:                     # keep epsilon above chi2_min + 2(k+2): match width
                        epsilon = max(epsilon, float(dists.min()) + floor_c)
                    elif eps_floor is not None:        # fixed user floor
                        epsilon = max(epsilon, eps_floor)
        finally:
            if pool is not None:
                pool.shutdown()

        # Resample the final weighted population to EQUAL weights so downstream code needs no weights.
        if len(population):
            ridx = np.random.default_rng([int(seed), 10_000]).choice(
                len(population), size=len(population), replace=True, p=parent_weights)
            resampled = [{k: population[i][k] for k in params + ["distance"]} for i in ridx]
        else:
            resampled = []
        runtime = time.perf_counter() - t0

        samples = pd.DataFrame(resampled, columns=params + ["distance"])
        distances = (np.array([p["distance"] for p in population], dtype=float)
                     if population else np.array([np.inf]))
        chi2_min = float(distances.min())
        k, n = len(params), lc.n_points
        # Best fit by the exact Gaussian log-likelihood over the final population — scatter-augmented
        # (with each particle's own scatter value) when a scatter parameter is fitted — so AIC/BIC are
        # COMPARABLE ACROSS SAMPLERS. (With simulate_noise the noisy-distance argmin is the luckiest
        # simulation-noise draw, not the best theta — never select by it.)
        best, max_log_likelihood = {}, float("nan")
        if population:
            lik = (make_likelihood(lc, kind="gaussian_scatter", space=space,
                                   scatter_param=scatter_param) if scatter_param
                   else make_likelihood(lc, space=space))

            def _logl(pt):
                mf = np.asarray(predict({p: float(pt[p]) for p in params}, times, bands), dtype=float)
                if scatter_param:
                    return lik.log_likelihood(mf, sigma_extra=float(pt[scatter_param]))
                return lik.log_likelihood(mf)

            logls = np.array([_logl(pt) for pt in population[:2000]], dtype=float)
            bi = int(np.nanargmax(logls)) if np.any(np.isfinite(logls)) else 0
            best = {p: float(population[bi][p]) for p in params}
            max_log_likelihood = float(logls[bi])
        info = {
            "n_particles": int(n_particles), "n_rounds": int(n_rounds),
            "rounds": round_info, "total_simulations": int(total_attempts),
            "weighted": True, "kernel": "diagonal_gaussian_2x_weighted_var",
            "quantile": None if epsilon_schedule is not None else float(quantile),
            "min_epsilon": ("auto" if auto_floor else eps_floor),
            "simulate_noise": bool(simulate_noise),
            "space": lik_space.space,
            "scatter_param": scatter_param,
            "n_jobs": int(n_jobs), "distance": getattr(distance, "__name__", str(distance)),
            "likelihood_space": lik_space.space,
        }
        attach_band_metrics(info, lc, model.name, best, space)
        result = SamplerResult(
            sampler="abc_smc", model=model.name, parameters=params,
            samples=samples, summary=summarize_posterior(samples, params), best_params=best,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )
        attach_predictive_metrics(result, lc, space)
        return result


def fit_ABC_SMC(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via ABC-SMC. See :meth:`ABCSMCSampler.fit` for options."""
    return ABCSMCSampler().fit(lc, model, prior=prior, **kwargs)
