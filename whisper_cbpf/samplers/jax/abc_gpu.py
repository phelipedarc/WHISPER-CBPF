"""Rejection ABC on the GPU -- a JAX port of ``whisper_cbpf.samplers.abc``.

ABC is the case where a GPU forward model pays off most. It has no network to train and no chain to
advance: the entire cost is *simulate, measure distance, keep the closest*. Measured on the same
AT2017GFO problem, the JAX kilonova simulator ran ~33x faster per simulation than redback's; SNPE
converted that into only 2.05x end-to-end because neural training then dominated. ABC has no such
Amdahl term, so the end-to-end speedup should land far closer to the raw simulator ratio.

Semantics are matched to the CPU sampler deliberately and in detail, because a posterior that
differs for an unnoticed reason is worse than no port at all:

* **The simulator is generative.** ``x = model_in_space(predict(theta)) + N(0, sigma)``, with the
  noise added *after* the flux->space map, per point, using that draw's own scatter in quadrature
  when ``scatter_param`` is set. That is what makes ABC exact as epsilon -> 0 and keeps the
  posterior width calibrated rather than merely peaked.
* **The distance denominator is the RAW observational error**, even when a scatter parameter is
  fitted. The scatter enters the simulated noise only. Putting it in the denominator too would
  double-count it and shrink the recovered scatter.
* **Acceptance is inclusive** (``d <= epsilon``), matching ``abc.py``; ABC-SMC uses a strict ``<``.
* **``best_params`` is the argmax of the exact Gaussian log-likelihood over the accepted draws, not
  the argmin-distance draw.** With ``simulate_noise=True`` the closest draw is the luckiest noise
  realisation, not the best parameter vector.
* **No NaN scrubbing.** ``abc.py`` lets non-finite values propagate into the distance (whisper's
  SNPE path does scrub them, and that difference is intentional). A NaN distance therefore poisons
  the quantile, exactly as on CPU -- which is a signal worth seeing, not hiding.

The GPU-specific parts are: prior draws by inverse-CDF (``rescale``) on ``jax.random.uniform`` so
sampling vectorises, one PRNG key folded per simulation index (matching the CPU's per-index
streams), and the simulate/distance step vmapped over particles and evaluated in chunks.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

# Imported relatively -- the CPU core and this sampler now live in one package. In the legacy
# split this file sat in whisper-GPU, cloned INTO a whisper-labia checkout as `whisper_gpu/`,
# and reached whisper_labia absolutely off sys.path; the merge removes that indirection.
from ...likelihood import GaussianLikelihood, make_likelihood
from ...models import get_model
from ...samplers.base import (
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)

from .._env import require_jax
from ..distances import jnp_distance

#: Particles simulated per compiled call.
#:
#: Sized by COMPILE TIME, not by memory and not by throughput. The distance reduction is only
#: (B, n_points) -> (B,) and would tolerate thousands, but a photometric model builds a
#: (B, n_comp, n_obs, n_wave) spectrum on the way, and XLA's compile time for that fused reduction
#: grows far faster than its throughput improves. Measured on the two-component kilonova
#: (n_obs=126, n_wave=300, float64, A6000), against redback's 16.7 ms/sim:
#:
#:     B      per-sim      speedup    compile
#:     1     0.3197 ms       52x       0.9 s
#:    16     0.1251 ms      134x       4.0 s
#:    64     0.1162 ms      144x      59.3 s
#:   256          --          --      did not finish in 6 minutes
#:
#: Throughput saturates by B=16; past it you buy 7% more speed for 15x the compile. Solving for
#: where B=64 overtakes B=16 gives ~6.2 MILLION simulations -- far beyond any realistic ABC budget,
#: and at a typical n_sim=256000 the choice is 36 s versus 89 s end to end. Hence 16.
#:
#: Since the sweep became a single on-device ``lax.scan``, this is now purely a compile-versus-
#: occupancy trade: the per-block host synchronisation that used to make small chunks expensive is
#: gone, and the measured cost at chunk=16 (0.219 ms/sim) is already below what the same model
#: benchmarked at standalone (0.272 ms/sim) -- because that figure was itself paying per-call
#: dispatch. There is no longer a latency reason to raise it.
#:
#: Raise it for a cheap analytic model with no wavelength axis (there the compile stays small and
#: bigger batches do help); lower it only if you OOM, which is unlikely to be the binding limit.
DEFAULT_CHUNK = 16

_MIN_FLUX_JY = 1e-300          # matches whisper_cbpf.likelihood._MIN_FLUX_JY
AB_ZEROPOINT_JY = 3631.0


def _model_in_space_jnp(jnp, flux, space, zeropoint_jy):
    """jnp counterpart of ``GaussianLikelihood.model_in_space``.

    Identity in flux space; in magnitude space ``-2.5 log10(clip(F, 1e-300) / zp)``. The clip is
    load-bearing and is copied verbatim: it maps a non-positive model flux to a finite ~759 mag
    (hence a huge chi2) instead of a NaN that would silently poison the acceptance quantile.
    """
    if space == "flux":
        return flux
    return -2.5 * jnp.log10(jnp.maximum(flux, _MIN_FLUX_JY) / zeropoint_jy)


def _prior_rescale_arrays(prior, names):
    """Vectorised inverse-CDF for the prior, built from whisper's own ``rescale`` hooks.

    ``Prior.sample`` draws one variate per parameter through numpy; that is inherently serial. Every
    whisper distribution also exposes ``rescale(u)`` mapping the unit cube to the parameter -- which
    is exactly what a vmapped sampler needs. Uniform and LogUniform are handled analytically here;
    anything else falls back to a per-draw python call, which still works but does not vectorise.
    """
    lows, highs, is_log = [], [], []
    for nm in names:
        d = prior.distributions[nm]
        kind = type(d).__name__
        if kind not in ("Uniform", "LogUniform"):
            raise TypeError(
                f"abc_gpu can only vectorise Uniform/LogUniform priors; parameter {nm!r} is "
                f"{kind}. Use the CPU sampler `abc`, or extend _prior_rescale_arrays.")
        lo, hi = d.bounds
        is_log.append(kind == "LogUniform")
        lows.append(np.log(lo) if kind == "LogUniform" else lo)
        highs.append(np.log(hi) if kind == "LogUniform" else hi)
    return np.asarray(lows), np.asarray(highs), np.asarray(is_log)


class ABCGPUSampler(BaseSampler):
    """Rejection ABC with a vmapped JAX simulator. See the module docstring for matched semantics."""

    name = "abc_gpu"

    def fit(self, lc, model, prior=None, *, predict_jax=None, n_simulations=10000, quantile=0.01,
            threshold=None, distance="chi2", simulate_noise=True, space="auto", scatter_param=None,
            seed=0, chunk=DEFAULT_CHUNK, max_logl_scan=2000) -> SamplerResult:
        """Fit ``lc`` with ``model`` by rejection ABC on the GPU.

        Parameters
        ----------
        predict_jax : callable
            ``predict_jax(theta_2d, times) -> flux_2d`` mapping ``(B, D)`` parameters and ``(n,)``
            times to ``(B, n)`` model **flux**, on device. Required: this sampler exists to
            vectorise the forward model, and falling back to the numpy ``model.predict`` in a
            python loop would just be the CPU sampler with extra steps. Parameter columns are in
            ``prior.names`` order.
        distance : str
            One of ``chi2`` (default, matching whisper's CPU ABC), ``mse``, ``rmse``, ``mae``,
            ``wmse``, ``wmae``. See :mod:`whisper_cbpf.distance` for what each implies -- the
            distance defines what "close" means, so it defines the posterior.
        n_simulations, quantile, threshold, simulate_noise, space, scatter_param, seed
            As ``whisper_cbpf.samplers.abc.ABCSampler.fit``. ``threshold`` overrides ``quantile``.
            ``scatter_param`` adds that draw's scatter to the SIMULATED noise; it does not make the
            parameter fittable (see the warning raised below).
        chunk : int
            Particles per compiled call.
        """
        jax, jnp = require_jax("abc_gpu")
        if predict_jax is None:
            raise ValueError(
                "abc_gpu needs predict_jax(theta_2d, times) -> flux_2d, a batched JAX forward "
                "model. Without it there is nothing to vectorise -- use sampler='abc' instead.")

        dist_fn = jnp_distance(distance)
        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        names = list(prior.names)
        if scatter_param is not None:
            if scatter_param not in names:
                raise ValueError(f"scatter_param {scatter_param!r} is not in the prior ({names}).")
            # A noise scale is a LIKELIHOOD parameter and is not identifiable by a rejection
            # distance: extra simulated noise only ever raises the expected residual, so every
            # distance here is monotonically penalised by it and the posterior rails to the
            # smallest allowed scatter. Fit it with a likelihood-based sampler, or leave it out of
            # the ABC prior (which is what whisper's own AT2017GFO analysis does).
            warnings.warn(
                f"abc_gpu: {scatter_param!r} is in the ABC prior, but a noise scale cannot be "
                "inferred from a rejection distance -- its marginal will rail to the prior's lower "
                "edge and is not a measurement. Drop it from the ABC prior and fit it with mcmc / "
                "nuts_gpu instead.", stacklevel=2)

        lik_space = GaussianLikelihood(lc, space=space)
        times = np.asarray(lc.time, dtype=float)
        obs_y = np.asarray(lik_space.y, dtype=float)
        obs_err = np.asarray(lik_space.sigma, dtype=float)
        sp, zp = lik_space.space, lik_space.zeropoint_jy

        lows, highs, is_log = _prior_rescale_arrays(prior, names)
        scatter_idx = names.index(scatter_param) if scatter_param else None

        t_j = jnp.asarray(times)
        y_j, e_j = jnp.asarray(obs_y), jnp.asarray(obs_err)
        lo_j, hi_j, log_j = jnp.asarray(lows), jnp.asarray(highs), jnp.asarray(is_log)

        def one(key):
            """One simulation: prior draw -> forward model -> noise -> chi2. Vmapped below."""
            k_theta, k_noise = jax.random.split(key)
            u = jax.random.uniform(k_theta, (lo_j.size,))
            raw = lo_j + u * (hi_j - lo_j)              # inverse CDF in the parameter's own space
            theta = jnp.where(log_j, jnp.exp(raw), raw)  # LogUniform is uniform in log, then exp
            flux = predict_jax(theta[None, :], t_j)[0]
            sim = _model_in_space_jnp(jnp, flux, sp, zp)
            if simulate_noise:
                # scatter enters the SIMULATED noise only, never the distance denominator
                sig = e_j if scatter_idx is None else jnp.sqrt(e_j ** 2 + theta[scatter_idx] ** 2)
                sim = sim + jax.random.normal(k_noise, sim.shape) * sig
            # the distance denominator is the RAW observational error even when a scatter
            # parameter is set -- the scatter enters the simulated noise only. Weighting the
            # residual by the inflated error too would double-count it.
            return theta, dist_fn(y_j, e_j, sim)

        base_key = jax.random.PRNGKey(int(seed))
        n_total = int(n_simulations)
        width = max(min(int(chunk), n_total), 1)
        n_blocks = -(-n_total // width)                  # ceil; the last block is padded, see below

        def block(_, block_idx):
            """One vmapped block of `width` simulations. The scan body -- compiled ONCE."""
            # fold_in per GLOBAL simulation index, mirroring the CPU's default_rng([seed, idx]):
            # the draw for index i depends only on (seed, i), never on how the run was blocked. That
            # property is what makes chunk a pure performance knob, and it is asserted by a test.
            keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(block_idx)
            return None, jax.vmap(one)(keys)

        # ONE on-device scan, not a Python loop with a host round trip per block.
        #
        # The loop this replaces did `block_until_ready` + `np.asarray` on every block, so a
        # 512,000-simulation run at chunk=16 made 32,000 device->host round trips and 32,000 python
        # dispatches. Measured on the two-component kilonova, that scaffolding cost MORE than the
        # physics: 0.5800 ms/sim total against a forward model of only 0.2723 ms/sim, i.e. 53% of
        # the runtime was overhead. Enough to turn a 61x-faster forward model into a dead heat with
        # 48 redback CPU cores.
        #
        # lax.scan keeps the whole sweep -- RNG, forward model, noise, distance -- resident on the
        # device and transfers once at the end. Compile is still governed by `width` (the vmap
        # extent inside the body), which is why DEFAULT_CHUNK stays modest; what disappears is the
        # per-block synchronisation, not the per-block compile.
        #
        # PADDING: indices run to n_blocks*width, so up to width-1 simulations past n_total are
        # computed and then discarded. That is deliberate -- a ragged final block would be a second
        # shape and therefore a second XLA compile, which on this model costs far more than the
        # handful of wasted draws. The padded draws are dropped before the quantile, so they cannot
        # influence epsilon or acceptance.
        # No explicit dtype: `fold_in` keys off this VALUE, so the default integer width must match
        # what the previous `jnp.arange(n_done, n_done + m)` produced or every draw changes.
        idx = jnp.arange(n_blocks * width).reshape(n_blocks, width)

        t0 = time.perf_counter()
        _, (theta_dev, dist_dev) = jax.lax.scan(block, None, idx)
        jax.block_until_ready(dist_dev)
        runtime = time.perf_counter() - t0

        theta_all = np.asarray(theta_dev, dtype=float).reshape(-1, len(names))[:n_total]
        distances = np.asarray(dist_dev, dtype=float).reshape(-1)[:n_total]

        epsilon = float(np.quantile(distances, quantile)) if threshold is None else float(threshold)
        keep = distances <= epsilon                      # inclusive, matching abc.py
        if not keep.any():
            warnings.warn(
                f"abc_gpu accepted 0 of {n_simulations} draws at epsilon={epsilon:g}; the posterior "
                "is empty. best_params / AIC / BIC reflect the single closest draw only.",
                stacklevel=2)
        acc_idx = np.nonzero(keep)[0]
        samples = pd.DataFrame(theta_all[acc_idx], columns=names)
        samples["distance"] = distances[acc_idx]

        # Best fit by EXACT likelihood, not by distance: with simulate_noise the argmin-distance
        # draw is the luckiest noise realisation. Scanned on CPU over a bounded head, as abc.py does.
        chi2_min = float(distances.min())               # over ALL draws, matching abc.py
        k, n = len(names), lc.n_points
        lik = (make_likelihood(lc, kind="gaussian_scatter", space=space,
                               scatter_param=scatter_param) if scatter_param
               else make_likelihood(lc, space=space))
        cand = acc_idx if len(acc_idx) else np.array([int(np.argmin(distances))])
        # At least one candidate, always. `max_logl_scan=0` is a reasonable thing for a caller to
        # pass ("skip the ranking, I only want timings"), and it used to leave `scan` empty, walk
        # past the `np.any(np.isfinite(...))` guard with b=0, and die on `scan[0]` with
        # "index 0 is out of bounds for axis 0 with size 0" -- after all the sampling was done.
        scan = cand[:int(max_logl_scan)] if int(max_logl_scan) > 0 else cand[:1]

        # The model fluxes come from the BATCHED path, not a per-draw call to the numpy
        # `model.predict`. This sampler already requires `predict_jax`, so a Python loop over up to
        # `max_logl_scan` draws re-entering an unjitted forward model was pure waste -- measured at
        # ~150 ms per draw on the two-component kilonova, i.e. ~48 s to rank 320 accepted draws,
        # which dwarfed the simulation loop it was ranking. Chunked with the same `chunk` used
        # above, for the same compile-time reason.
        #
        # The likelihood itself is still whisper's own `lik.log_likelihood`, evaluated in numpy on
        # the precomputed fluxes: that part is cheap array algebra, and reimplementing it in jnp
        # would risk the two arms scoring "best fit" by subtly different rules.
        flux_rows = []
        for s in range(0, len(scan), max(int(chunk), 1)):
            block = jnp.asarray(theta_all[scan[s:s + int(chunk)]])
            flux_rows.append(np.asarray(jax.block_until_ready(predict_jax(block, t_j)), dtype=float))
        fluxes = np.concatenate(flux_rows, axis=0) if flux_rows else np.zeros((0, len(obs_y)))

        logls = []
        for row, i in zip(fluxes, scan):
            th = {nm: float(theta_all[i, j]) for j, nm in enumerate(names)}
            logls.append(lik.log_likelihood(row, sigma_extra=th[scatter_param]) if scatter_param
                         else lik.log_likelihood(row))
        logls = np.asarray(logls, dtype=float)
        b = int(np.nanargmax(logls)) if np.any(np.isfinite(logls)) else 0
        best = {nm: float(theta_all[scan[b], j]) for j, nm in enumerate(names)}
        max_log_likelihood = float(logls[b])

        info = {
            "n_simulations": int(n_simulations),
            "n_accepted": int(keep.sum()),
            "acceptance_rate": float(keep.mean()),
            "epsilon": epsilon,
            "quantile": None if threshold is not None else float(quantile),
            "simulate_noise": bool(simulate_noise),
            "space": lik_space.space,
            "scatter_param": scatter_param,
            "distance": str(distance),
            "likelihood_space": lik.space,
            "backend": "jax-vmap",
            "chunk": int(chunk),
            "device": str(jax.devices()[0]),
            "x64": bool(jax.config.jax_enable_x64),
            "seed": int(seed),
        }
        attach_band_metrics(info, lc, model.name, best, space)
        result = SamplerResult(
            sampler="abc_gpu", model=model.name, parameters=names, samples=samples,
            summary=summarize_posterior(samples, names), best_params=best,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )
        # (n_chains, n_draws, n_params) for the shared ESS helper; ABC draws are independent.
        result.samples_by_chain = samples[names].to_numpy()[None, :, :]
        attach_predictive_metrics(result, lc, space)
        return result


def fit_ABC_GPU(lc, model, prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` by rejection ABC on the GPU. See :meth:`ABCGPUSampler.fit`."""
    return ABCGPUSampler().fit(lc, model, prior=prior, **kwargs)
