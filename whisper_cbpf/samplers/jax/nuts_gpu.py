"""GPU NUTS sampling via NumPyro, following whisper_cbpf's ``BaseSampler``/``SamplerResult``
contract exactly (same shape as ``whisper_cbpf.samplers.mcmc.MCMCSampler``).

Registered into whisper's own sampler registry as ``nuts_gpu`` when :mod:`whisper_cbpf` is
imported, so ``wp.fit(lc, model, sampler="nuts_gpu", ...)`` works through the normal dispatcher.
Whisper instantiates samplers with zero arguments, so every GPU option travels through ``fit()``.

Takes an already-built, jitted ``log_prob_fn`` (e.g. from
``whisper_cbpf.models.jax.flare.make_log_prob_jax``) rather than building its own from ``model`` —
this is what lets the emcee-vectorized-JAX arm and this NUTS arm share the *literal same*
Python object for the fairness comparison, instead of just "the same code written twice".
"""
from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

# Imported relatively -- the CPU core and this sampler now live in one package. In the legacy
# split this file sat in whisper-GPU, cloned INTO a whisper-labia checkout as `whisper_gpu/`,
# and reached whisper_labia absolutely off sys.path; the merge removes that indirection.

from ...samplers.base import (  # noqa: E402
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)


def _numpyro_priors(prior, names):
    """Map whisper's prior onto NumPyro distributions, ONE PER PARAMETER.

    This used to read ``.bounds`` for every parameter and build a single vectorised
    ``dist.Uniform(lows, highs)``. That silently substituted a uniform density for any parameter
    whose prior was not uniform -- a ``LogUniform(100, 6000)`` temperature floor became
    ``Uniform(100, 6000)``, which is a *different posterior*, not a different parameterisation:
    log-uniform puts half its mass below 775 K and uniform puts half below 3050 K. Nothing warned,
    because ``.bounds`` is identical for both.

    Per-parameter sites also let NumPyro pick the right bijector for each support. For LogUniform
    that is a log transform, which is what a scale parameter spanning ~2 decades wants anyway --
    NUTS explores it far better in log space than a uniform reparameterisation would.

    Only the distributions whisper's ABC/NUTS paths actually support are mapped; anything else
    raises rather than being quietly approximated.
    """
    out = []
    for nm in names:
        d = prior.distributions[nm]
        kind = type(d).__name__
        lo, hi = (float(x) for x in d.bounds)
        if kind == "Uniform":
            out.append(dist.Uniform(lo, hi))
        elif kind == "LogUniform":
            out.append(dist.LogUniform(lo, hi))
        else:
            raise TypeError(
                f"nuts_gpu cannot express prior {kind!r} on parameter {nm!r} as a NumPyro "
                f"distribution. Supported: Uniform, LogUniform. Extend _numpyro_priors rather "
                f"than falling back to a uniform box -- that changes the posterior silently.")
    return out


def _float_dtype():
    """Match JAX's configured precision instead of forcing float32.

    The kilonova likelihood is evaluated in float64 for production runs (``JAX_ENABLE_X64=1``);
    hard-coding float32 here downcast the sampled theta and the final argmax scan back to single
    precision, quietly discarding the very thing x64 was turned on for.
    """
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class NUTSGPUSampler(BaseSampler):
    """NUTS (NumPyro) on GPU. See module docstring for the shared-``log_prob_fn`` design."""

    name = "nuts_gpu"

    def fit(self, lc, model, prior=None, *, log_prob_fn, num_warmup=1000, num_samples=2000,
            num_chains=4, space="flux", seed=0, progress=False,
            target_accept_prob=0.8) -> SamplerResult:
        """Fit ``lc`` with ``model`` via NumPyro NUTS.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve (only used for ``n_data``/``attach_*`` metrics here — the actual
            likelihood is baked into ``log_prob_fn``).
        model : whisper_cbpf.models.Model-like
            Needs ``.name`` and ``.parameters`` (order must match ``log_prob_fn``'s expected
            flat theta vector).
        prior : Prior, optional
            Defaults to ``model.default_prior``. Must be a box of ``Uniform`` per parameter
            (bounds read from ``prior.distributions[name].bounds``).
        log_prob_fn : Callable[[jnp.ndarray], jnp.ndarray]
            Required. A ``jax.jit``-compiled function mapping a flat ``theta`` vector (order =
            ``model.parameters``) to a scalar log-likelihood, ``-inf`` outside the prior box
            (see ``flare_jax.make_log_prob_jax``) — i.e. pure log-likelihood, not log-posterior;
            the flat prior's constant density is added separately via NumPyro's own ``Uniform``
            sample site, so no double-counting.
        num_warmup, num_samples, num_chains : int
            NUTS budget. Chains run via ``chain_method="vectorized"`` (all chains batched on the
            one visible GPU via vmap — the standard single-device multi-chain NumPyro pattern).
        space : str
            Recorded in ``info``/passed to ``attach_predictive_metrics`` — not used to rebuild the
            likelihood (that's already fixed inside ``log_prob_fn``).

        Returns
        -------
        SamplerResult
            Warmup/sampling time reported separately in ``info``, alongside divergences and
            per-parameter r-hat (via arviz). The raw NumPyro ``MCMC`` object is attached as
            ``result.numpyro_mcmc``.
        """
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        names = list(model.parameters)
        k, n = len(names), int(len(lc.time))
        site_dists = _numpyro_priors(prior, names)

        def numpyro_model():
            # One site per parameter so each keeps its OWN density and support transform; stacked
            # back into the flat theta vector that log_prob_fn expects, in model.parameters order.
            theta = jnp.stack([numpyro.sample(nm, d) for nm, d in zip(names, site_dists)])
            numpyro.factor("log_likelihood", log_prob_fn(theta))

        kernel = NUTS(numpyro_model, target_accept_prob=float(target_accept_prob))
        mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples),
                    num_chains=int(num_chains), progress_bar=progress, chain_method="vectorized")

        rng_key = jax.random.PRNGKey(int(seed))

        # NOTE ON TIMING: XLA compiles the NUTS kernel on the first warmup() call, so
        # `warmup_time_s` below is compile + warmup on a cold process. Compilation is a
        # one-time cost that does NOT scale with the number of curves fitted, and reporting
        # it as sampler cost badly misrepresents throughput (measured: ~8.4 s compile vs
        # ~4.8 s of actual warmup). The benchmark driver therefore fits each configuration
        # twice in one process -- XLA caches the executable, so run 2 is compile-free and
        # (run1 - run2) isolates compile time. See run_benchmark.py.
        t0 = time.perf_counter()
        mcmc.warmup(rng_key, collect_warmup=False, extra_fields=("diverging",))
        jax.block_until_ready(mcmc.last_state)
        warmup_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        # extra_fields is REQUIRED for divergences to be collected at all. Without it
        # get_extra_fields() comes back empty and n_divergences falls through to its -1 sentinel --
        # which reads like "unknown" but in practice got treated as "none", silently disabling the
        # single most important NUTS health check. A divergent chain is biased, not merely noisy.
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("diverging",))
        # Per-parameter sites come back as a dict of (chains, draws); restack in parameter order.
        by_site = mcmc.get_samples(group_by_chain=True)
        samples_by_chain = jnp.stack([by_site[nm] for nm in names], axis=-1)  # (chains, draws, k)
        extra = mcmc.get_extra_fields(group_by_chain=True)
        jax.block_until_ready(samples_by_chain)
        sampling_time = time.perf_counter() - t1
        runtime = warmup_time + sampling_time

        n_chains_run, n_draws = samples_by_chain.shape[0], samples_by_chain.shape[1]
        theta_flat = np.asarray(samples_by_chain).reshape(-1, k)
        samples = pd.DataFrame(theta_flat, columns=names)

        n_divergences = int(np.asarray(extra["diverging"]).sum()) if "diverging" in extra else -1

        rhat_by_param = {}
        try:
            import arviz as az
            idata = az.convert_to_inference_data({"theta": np.asarray(samples_by_chain)})
            rhat_vals = np.atleast_1d(np.asarray(az.rhat(idata)["theta"]))
            rhat_by_param = {nm: float(v) for nm, v in zip(names, rhat_vals)}
        except Exception:
            pass  # arviz optional / single-chain rhat undefined either way

        # log_prob_fn is pure log-likelihood (the prior's density is contributed separately by
        # NumPyro's own sample sites), so the best draw is the argmax of log_prob_fn itself.
        #
        # SCANNED WITH lax.map, NOT vmap. This was `jax.vmap(log_prob_fn)(theta_flat)` -- every
        # post-warmup sample batched into a single call -- which makes the vmap extent equal to the
        # draw count. XLA's compile time for a fused reduction grows steeply with that extent: on
        # the two-component kilonova a mere B=256 did not finish compiling in six minutes, so at
        # 16,000 draws the sampler sat at 0% GPU utilisation, single-threaded, for over an hour
        # AFTER sampling had finished. It scaled with the very axis a user turns up.
        #
        # lax.map compiles the SCALAR log_prob_fn once -- the same program NUTS itself just ran --
        # and loops it on device, so compile is constant in the number of draws and the cost is a
        # predictable ~0.3 ms per draw. Slower per element than a wide vmap would be if that vmap
        # could ever be compiled; unboundedly faster than one that cannot.
        ll_flat = np.asarray(jax.block_until_ready(
            jax.lax.map(log_prob_fn, jnp.asarray(theta_flat, dtype=_float_dtype()))))
        best_idx = int(np.nanargmax(ll_flat))
        best_params = {nm: float(theta_flat[best_idx, j]) for j, nm in enumerate(names)}
        max_log_likelihood = float(ll_flat[best_idx])

        max_rhat = max(rhat_by_param.values()) if rhat_by_param else float("nan")
        # FAIL CLOSED. This used to read `not np.isfinite(max_rhat) or max_rhat < 1.01`, so a run
        # where arviz was missing or r-hat could not be computed (max_rhat = NaN) was reported as
        # CONVERGED on the strength of zero divergences alone. A convergence flag that defaults to
        # True when the evidence is unavailable is worse than no flag: unknown is not converged.
        converged = (n_divergences == 0) and np.isfinite(max_rhat) and max_rhat < 1.01

        info = {
            "num_warmup": int(num_warmup), "num_samples": int(num_samples),
            "num_chains": int(n_chains_run), "space": space, "seed": int(seed),
            "target_accept_prob": float(target_accept_prob),
            "warmup_time_s": float(warmup_time), "sampling_time_s": float(sampling_time),
            "n_divergences": n_divergences, "rhat": rhat_by_param,
            "max_rhat": (float(max_rhat) if np.isfinite(max_rhat) else None),
            "converged": bool(converged),
        }
        attach_band_metrics(info, lc, model.name, best_params, space)
        result = SamplerResult(
            sampler="nuts_gpu", model=model.name, parameters=names, samples=samples,
            summary=summarize_posterior(samples, names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )
        result.numpyro_mcmc = mcmc
        result.samples_by_chain = np.asarray(samples_by_chain)   # (chains, draws, k) -- for ESS/sec
        attach_predictive_metrics(result, lc, space)
        return result


def fit_NUTSGPU(lc, model, prior=None, *, log_prob_fn, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via NumPyro NUTS on GPU. See :meth:`NUTSGPUSampler.fit`."""
    return NUTSGPUSampler().fit(lc, model, prior=prior, log_prob_fn=log_prob_fn, **kwargs)
