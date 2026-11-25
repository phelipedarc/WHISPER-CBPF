"""Importance-weighted ABC-SMC on the GPU -- a JAX port of :mod:`whisper_cbpf.samplers.abc_smc`.

Only the propose -> simulate -> distance step runs on the device, because that step is essentially
all of the cost: it evaluates the forward model once per attempt, and a round needs thousands of
attempts. The importance-weight recursion is O(M*N*D) -- two million flops at 500 particles and 8
parameters, microseconds on a CPU -- so ``_kernel_std`` and ``_importance_log_weights`` are
imported from the CPU module and called unchanged. That recursion is also the part of the algorithm
easiest to get subtly wrong: drop it and the population converges to a distorted distribution that
still looks like a posterior. The expensive part is on the GPU; the delicate part is shared.

Semantics match ``abc_smc.py``:

* **Acceptance is strict** (``d < epsilon``). Rejection ABC uses ``d <= epsilon``; ABC-SMC does
  not. The difference shows only when a distance lands exactly on the threshold -- which is exactly
  what a schedule setting epsilon to a previous round's accepted distance produces.
* Round 0 draws the prior with uniform weights. Later rounds resample a parent from the *weighted*
  population and perturb it with a diagonal Gaussian of std ``sqrt(2 * weighted variance)``
  (Toni 2009), in native parameter space.
* A perturbed particle outside the prior's support is rejected. JAX has no ``continue``, so the
  attempt is still evaluated and its distance forced to ``+inf``: same outcome, since it can never
  beat a finite epsilon, and the attempt indexing stays dense so the population is independent of
  blocking.
* The adaptive schedule sets the next epsilon to the ``quantile`` of this round's accepted
  distances; ``min_epsilon="auto"`` floors it at ``best + 2*(k+2)``, because driving epsilon to the
  minimum distance collapses the posterior onto the MLE.
* The final weighted population is resampled to equal weights, so downstream code never sees
  weights.

One thing cannot match: numpy's Philox and JAX's threefry are different bit generators, so the same
seed gives different draws. Agreement with the CPU sampler is statistical.

**Attempts are issued in waves, not one scan per round.** A round runs until ``n_particles`` are
accepted, so its length is data-dependent and cannot be a fixed-trip-count scan. Each wave is one
``lax.scan`` over ``wave // chunk`` vmapped blocks, evaluated entirely on device, with a single host
transfer at the end. A round typically needs one or two waves, so a fit synchronises of order ten
times rather than once per block -- the trap that cost ``abc_gpu`` 53% of its runtime before it was
fused.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ...likelihood import GaussianLikelihood, make_likelihood
from ...models import get_model
from ...samplers.abc_smc import _importance_log_weights, _kernel_std, _warn_auto_floor_scale
from ...samplers.base import (
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
    _warn_user,
)

from ...backends._env import require_jax
from ...distance._jax import jnp_distance
from .abc_gpu import DEFAULT_CHUNK, _best_by_logl, _model_in_space_jnp, _PriorDraw

#: Attempts issued per host synchronisation, as a multiple of ``n_particles``.
#:
#: Sized so a round usually finishes in one wave. Too small and the fit syncs constantly; too large
#: and a round that accepts easily wastes work on attempts it never needed. Acceptance rates in
#: ABC-SMC start near 1 and fall as epsilon tightens, so 8x is generous early and about right late.
WAVE_MULTIPLIER = 8


class ABCSMCGPUSampler(BaseSampler):
    """Importance-weighted ABC-SMC with a vmapped JAX simulator. See the module docstring."""

    name = "abc_smc_gpu"

    def fit(self, lc, model, prior=None, *, predict_jax=None, n_particles=500, n_rounds=5,
            epsilon_schedule=None, quantile=0.5, min_epsilon=None, simulate_noise=True,
            space="auto", scatter_param=None, distance="chi2", seed=0, chunk=DEFAULT_CHUNK,
            attempts_per_wave=None, max_attempts_per_round=None,
            max_logl_scan=None) -> SamplerResult:
        """Fit ``lc`` with ``model`` by importance-weighted ABC-SMC on the GPU.

        Parameters
        ----------
        prior : Prior, optional
            Defaults to ``model.default_prior``. ``Uniform``, ``LogUniform``, ``Normal``,
            ``TruncatedNormal`` and ``Fixed``; round 0 draws each through its inverse CDF on the
            device (as ``abc_gpu`` does). A ``Fixed`` parameter is never perturbed: a constant
            column in ``samples``, listed in ``info["fixed"]``, left out of the kernel and the
            importance weights, and not counted in AIC/BIC (``n_params``).
        predict_jax : callable, optional
            ``predict_jax(theta_2d, times) -> flux_2d`` mapping ``(B, D)`` parameters and ``(n,)``
            times to ``(B, n)`` model **flux**, on device. Columns are in ``prior.names`` order.
            **Optional**: when omitted it is built from ``model.predict_jax`` by
            :func:`whisper_cbpf.samplers.jax._adapters.make_batched_predict_jax`. An explicitly
            passed callable always wins.
        n_particles, n_rounds, epsilon_schedule, quantile, min_epsilon, simulate_noise, space,
        scatter_param, seed
            As :meth:`whisper_cbpf.samplers.abc_smc.ABCSMCSampler.fit`.
        distance : str
            ``chi2`` (default), ``mse``, ``rmse``, ``mae``, ``wmse``, ``wmae``, ``max_abs_z``
            (every point within epsilon sigma, accepted strictly below epsilon).
            ``min_epsilon="auto"`` is derived for ``chi2`` and warns with any other distance.
        chunk : int
            Particles per vmapped block inside the scan. **Bounded by XLA compile time, not by
            memory**, and that compile time is erratic in the width: see ``abc_gpu.fit``'s
            ``chunk`` and :data:`~whisper_cbpf.samplers.jax.abc_gpu.DEFAULT_CHUNK` for the
            measurements behind the default, 250. Prefer a width that is not a power of two
            (256 and 1024 compile for minutes on the kilonovae).
        attempts_per_wave : int, optional
            Attempts per host synchronisation. Defaults to ``WAVE_MULTIPLIER * n_particles``.
        max_logl_scan : int, optional
            Cap on the final-population particles re-scored by exact log-likelihood to pick
            ``best_params``. **Default ``None`` scores the whole population** -- the particles
            themselves, before the equal-weight resample. When a cap binds, the particles with the
            LOWEST distance are scored. ``info['logl_scan_n']`` and ``info['logl_scan_capped']``
            record what was done.
        """
        jax, jnp = require_jax("abc_smc_gpu")

        dist_fn = jnp_distance(distance)
        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        names = list(prior.names)
        k = len(names)

        predict_jax_source = "caller"
        if predict_jax is None:
            if getattr(model, "predict_jax", None) is None:
                raise ValueError(
                    f"abc_smc_gpu needs predict_jax(theta_2d, times) -> flux_2d, a batched JAX "
                    f"forward model, and model {model.name!r} carries no predict_jax to build one "
                    f"from. Register it through a JAX factory, pass predict_jax= yourself, or use "
                    f"sampler='abc_smc' instead.")
            from ._adapters import make_batched_predict_jax
            predict_jax = make_batched_predict_jax(lc, model, names=names,
                                                   chunk=None, max_batch=None)
            predict_jax_source = "auto"

        if scatter_param is not None:
            if scatter_param not in names:
                raise ValueError(f"scatter_param {scatter_param!r} is not in the prior ({names}).")
            if not simulate_noise:
                raise ValueError("scatter_param requires simulate_noise=True (the scatter enters "
                                 "the generative noise).")

        lik_space = GaussianLikelihood(lc, space=space)
        sp, zp = lik_space.space, lik_space.zeropoint_jy
        times = np.asarray(lc.time, dtype=float)
        obs_y = np.asarray(lik_space.y, dtype=float)
        obs_err = np.asarray(lik_space.sigma, dtype=float)

        # Round 0 draws the prior through each parameter's inverse CDF (`_PriorDraw`). The
        # perturbation kernel and the support test live in NATIVE space, matching abc_smc.py. The
        # kernel moves the FREE parameters only: a Fixed one has kernel width 0, so it keeps its
        # value exactly, and it is left out of the importance weights and of AIC/BIC.
        spec = _PriorDraw(prior, names, "abc_smc_gpu")
        free_idx = spec.free
        free_names = [names[i] for i in free_idx]
        free_prior = type(prior)({nm: prior.distributions[nm] for nm in free_names})

        t_j = jnp.asarray(times)
        y_j, e_j = jnp.asarray(obs_y), jnp.asarray(obs_err)
        draw = spec.build(jnp)
        nlo_j, nhi_j = jnp.asarray(spec.nat_lo), jnp.asarray(spec.nat_hi)
        scatter_idx = names.index(scatter_param) if scatter_param else None

        def _simulate(theta, key):
            """theta (D,) -> distance. Shared by every round; only the proposal differs."""
            flux = predict_jax(theta[None, :], t_j)[0]
            sim = _model_in_space_jnp(jnp, flux, sp, zp)
            if simulate_noise:
                sig = e_j if scatter_idx is None else jnp.sqrt(e_j ** 2 + theta[scatter_idx] ** 2)
                sim = sim + jax.random.normal(key, sim.shape) * sig
            return dist_fn(y_j, e_j, sim)

        def make_round(round_idx, parents_j, weights_j, kstd_j):
            """Build the per-attempt proposal+score for one round, closed over its population."""

            def one(key):
                k_prop, k_noise = jax.random.split(key)
                if round_idx == 0:
                    u = jax.random.uniform(k_prop, (k,))
                    theta = draw(u)
                    return theta, _simulate(theta, k_noise)
                k_pick, k_pert = jax.random.split(k_prop)
                j = jax.random.choice(k_pick, parents_j.shape[0], p=weights_j)
                theta = parents_j[j] + jax.random.normal(k_pert, (k,)) * kstd_j
                # Outside the prior's support -> rejected. No `continue` in JAX, so the attempt is
                # evaluated and its distance forced to +inf: it can never beat a finite epsilon, and
                # the attempt still consumes its index so the population stays blocking-independent.
                inside = jnp.all((theta >= nlo_j) & (theta <= nhi_j))
                d = _simulate(theta, k_noise)
                return theta, jnp.where(inside, d, jnp.inf)

            return one

        base_key = jax.random.PRNGKey(int(seed))
        wave = int(attempts_per_wave or WAVE_MULTIPLIER * n_particles)
        width = max(min(int(chunk), wave), 1)
        n_blocks = -(-wave // width)
        if max_attempts_per_round is None:
            max_attempts_per_round = max(200 * n_particles, 200_000)
        if epsilon_schedule is not None:
            epsilon_schedule = [float(e) for e in epsilon_schedule]
            n_rounds = len(epsilon_schedule)

        auto_floor = min_epsilon == "auto"
        eps_floor = None if (auto_floor or min_epsilon is None) else float(min_epsilon)
        floor_c = 2.0 * (k + 2)
        if (auto_floor and epsilon_schedule is None
                and str(distance).lower() not in ("chi2", "chi_square")):
            _warn_auto_floor_scale(str(distance), floor_c)

        t0 = time.perf_counter()
        parents_arr = None
        parent_weights = None
        parent_log_weights = None
        kernel_std = None
        population, round_info = [], []
        total_attempts = 0
        epsilon = epsilon_schedule[0] if epsilon_schedule is not None else np.inf

        for round_idx in range(n_rounds):
            if epsilon_schedule is not None:
                epsilon = epsilon_schedule[round_idx]

            if round_idx == 0:
                parents_j = jnp.zeros((1, k))
                weights_j = jnp.ones((1,))
                kstd_j = jnp.ones((k,))
            else:
                parents_j = jnp.asarray(parents_arr)
                weights_j = jnp.asarray(parent_weights)
                kstd_j = jnp.asarray(kernel_std)

            one = make_round(round_idx, parents_j, weights_j, kstd_j)
            round_key = jax.random.fold_in(base_key, round_idx)

            def block(_, block_idx, _one=one, _rk=round_key):
                keys = jax.vmap(lambda i: jax.random.fold_in(_rk, i))(block_idx)
                return None, jax.vmap(_one)(keys)

            acc_theta, acc_dist, acc_idx = [], [], []
            n_acc, next_attempt = 0, 0
            while n_acc < n_particles and next_attempt < max_attempts_per_round:
                idx = jnp.arange(next_attempt, next_attempt + n_blocks * width).reshape(n_blocks, width)
                _, (th, d) = jax.lax.scan(block, None, idx)          # one wave, one sync
                th = spec.exact_fixed(np.asarray(th, dtype=float).reshape(-1, k))
                d = np.asarray(d, dtype=float).reshape(-1)
                keep = d < epsilon                                    # STRICT, matching abc_smc.py
                if keep.any():
                    acc_theta.append(th[keep])
                    acc_dist.append(d[keep])
                    acc_idx.append(np.nonzero(keep)[0] + next_attempt)
                    n_acc += int(keep.sum())
                next_attempt += n_blocks * width

            attempts = next_attempt
            total_attempts += attempts
            if n_acc < n_particles:
                _warn_user(
                    f"abc_smc_gpu round {round_idx + 1}: only {n_acc}/{n_particles} accepted after "
                    f"{attempts} attempts (epsilon={epsilon:g}).")

            if acc_theta:
                th_all = np.concatenate(acc_theta, axis=0)
                d_all = np.concatenate(acc_dist)
                i_all = np.concatenate(acc_idx)
                order = np.argsort(i_all)            # by global attempt index, as abc_smc.py does
                th_all, d_all = th_all[order][:n_particles], d_all[order][:n_particles]
            else:
                th_all, d_all = np.empty((0, k)), np.empty(0)

            theta_pop = [{nm: float(v) for nm, v in zip(names, row)} for row in th_all]
            population = [dict(t, distance=float(dd)) for t, dd in zip(theta_pop, d_all)]

            # Uniform weights at round 0, otherwise whisper's own SMC recursion -- imported, not
            # reimplemented, because this is the part that silently distorts the posterior if wrong.
            if round_idx == 0 or parents_arr is None or not theta_pop:
                log_w = np.full(len(theta_pop), -np.log(max(len(theta_pop), 1)))
            else:
                log_w = _importance_log_weights(theta_pop, free_names, parents_arr[:, free_idx],
                                                parent_log_weights, kernel_std[free_idx],
                                                free_prior)
            weights = np.exp(log_w - logsumexp(log_w)) if len(log_w) else np.array([])

            dists = d_all if len(d_all) else np.array([np.inf])
            round_info.append({
                "round": round_idx + 1,
                "epsilon": (float(epsilon) if np.isfinite(epsilon) else None),
                "n_accepted": len(population),
                "attempts": int(attempts),
                "acceptance_rate": float(len(population) / attempts) if attempts else 0.0,
                "best_distance": float(dists.min()),
                "effective_sample_size": float(1.0 / np.sum(weights ** 2)) if len(weights) else 0.0,
            })

            parents_arr = th_all
            parent_log_weights = log_w
            parent_weights = weights
            kernel_std = None
            if len(weights):                         # width 0 on a Fixed column: never moved
                kernel_std = np.zeros(k)
                kernel_std[free_idx] = _kernel_std(th_all[:, free_idx], weights)
            if epsilon_schedule is None and round_idx + 1 < n_rounds:
                epsilon = float(np.quantile(dists, quantile))
                if auto_floor:
                    epsilon = max(epsilon, float(dists.min()) + floor_c)
                elif eps_floor is not None:
                    epsilon = max(epsilon, eps_floor)

        runtime = time.perf_counter() - t0

        # Resample to EQUAL weights so downstream summaries need no weight handling.
        rng = np.random.default_rng(int(seed))
        if population and len(parent_weights):
            pick = rng.choice(len(population), size=len(population), replace=True, p=parent_weights)
            resampled = [population[i] for i in pick]
        else:
            resampled = population
        samples = pd.DataFrame(resampled) if resampled else pd.DataFrame(columns=names + ["distance"])

        n_data = int(lc.n_points)
        distances_final = (np.asarray(samples["distance"], dtype=float) if len(samples)
                           else np.array([np.inf]))
        chi2_min = float(distances_final.min())

        lik = (make_likelihood(lc, kind="gaussian_scatter", space=space,
                               scatter_param=scatter_param) if scatter_param
               else make_likelihood(lc, space=space))
        # Best fit over the final POPULATION, as abc_smc.py does -- not over the head of the
        # resampled `samples`, which held duplicates and could miss the best particle entirely.
        n_scanned, capped = 0, False
        if population:
            b, max_log_likelihood, n_scanned, capped = _best_by_logl(
                jax, jnp, predict_jax, t_j, th_all, d_all, lik, scatter_idx, chunk, max_logl_scan)
            best = {nm: float(th_all[b, j]) for j, nm in enumerate(names)}
        else:
            best, max_log_likelihood = {nm: float("nan") for nm in names}, float("nan")

        info = {
            "n_particles": int(n_particles),
            "n_rounds": int(n_rounds),
            "rounds": round_info,
            "total_attempts": int(total_attempts),
            "final_epsilon": round_info[-1]["epsilon"] if round_info else None,
            "quantile": float(quantile),
            "simulate_noise": bool(simulate_noise),
            "space": lik_space.space,
            "scatter_param": scatter_param,
            "distance": str(distance),
            "chunk": int(width),
            "attempts_per_wave": int(wave),
            "device": str(jax.devices()[0]),
            "backend": "jax-vmap-scan",
            "x64": bool(jax.config.jax_enable_x64),
            "predict_jax": predict_jax_source,
            "logl_scan_n": int(n_scanned),
            "logl_scan_capped": bool(capped),
            # points simulated at mag_floor. None = not counted: this sampler runs the
            # forward map inside one compiled scan, and a traced map counts nothing.
            "mag_floor_stats": (dict(fs) if (fs := getattr(predict_jax, "floor_stats", None))
                                and fs["n_points"] else None),
            "fixed": dict(spec.fixed),
        }
        attach_band_metrics(info, lc, model, best, lik_space.space)

        k_free = len(free_idx)                       # a Fixed parameter is not counted in AIC/BIC
        aic = (2.0 * k_free - 2.0 * max_log_likelihood if np.isfinite(max_log_likelihood)
               else float("nan"))
        bic = (k_free * np.log(max(n_data, 1)) - 2.0 * max_log_likelihood
               if np.isfinite(max_log_likelihood) else float("nan"))

        result = SamplerResult(
            sampler=self.name, model=model.name, parameters=names, samples=samples,
            summary=summarize_posterior(samples, names), best_params=best, n_data=n_data,
            n_params=k_free, runtime_s=runtime, info=info, min_distance=chi2_min,
            max_log_likelihood=max_log_likelihood, aic=aic, bic=bic)
        attach_predictive_metrics(result, lc, lik_space.space, model=model)
        return result


def fit_ABCSMCGPU(lc, model, prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via ABC-SMC on GPU. See :meth:`ABCSMCGPUSampler.fit`.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models import get_model
    >>> from whisper_cbpf.samplers.jax import fit_ABCSMCGPU
    >>> t = np.linspace(1.0, 29.0, 30)
    >>> truth = {"log_amp": 0.5, "log_sigma": 0.3, "log_tau": 1.5, "t0": 8.0}
    >>> flux = np.asarray(get_model("flare_jax").predict(truth, t, None), float)
    >>> lc = wp.LightCurve(time=t, band=np.array(["r"] * 30), flux=flux,
    ...                    flux_err=np.full(30, 0.05 * flux.max()))
    >>> res = fit_ABCSMCGPU(lc, "flare_jax", n_particles=100, n_rounds=3, seed=0)
    >>> res.n_samples, len(res.info["rounds"])
    (100, 3)
    """
    return ABCSMCGPUSampler().fit(lc, model, prior=prior, **kwargs)
