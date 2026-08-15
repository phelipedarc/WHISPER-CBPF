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

from ...models import get_model  # noqa: E402
from ...samplers.base import (  # noqa: E402
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)
from ._adapters import resolve_density as _resolve_density  # noqa: E402
from ._adapters import resolve_sampling_names as _resolve_sampling_names  # noqa: E402


def _numpyro_priors(prior, names):
    """Map whisper's prior onto NumPyro sample sites, ONE PER PARAMETER.

    Returns one ``(site_name, numpyro_distribution, to_linear)`` triple per entry of ``names``,
    in that order. ``to_linear`` is ``None`` when the site *is* the model parameter, and a callable
    when the site is a reparameterisation that has to be pushed back to the parameter's own units.

    This used to read ``.bounds`` for every parameter and build a single vectorised
    ``dist.Uniform(lows, highs)``. That silently substituted a uniform density for any parameter
    whose prior was not uniform -- a ``LogUniform(100, 6000)`` temperature floor became
    ``Uniform(100, 6000)``, which is a *different posterior*, not a different parameterisation:
    log-uniform puts half its mass below 775 K and uniform puts half below 3050 K. Nothing warned,
    because ``.bounds`` is identical for both.

    WHY LOG-UNIFORM IS SAMPLED AS A UNIFORM ON log10, NOT AS ``dist.LogUniform``
    ---------------------------------------------------------------------------
    ``dist.LogUniform`` carries the correct *density*, but not the geometry NUTS needs, and the two
    are independent. NumPyro walks a site in the unconstrained space that ``biject_to(d.support)``
    defines, and ``dist.LogUniform(lo, hi).support`` is ``constraints.interval(lo, hi)`` -- exactly
    what a plain ``Uniform`` declares. So ``biject_to`` returns (measured, NumPyro 0.21)::

        biject_to(dist.LogUniform(100., 6000.).support)
        -> ComposeTransform([SigmoidTransform(), AffineTransform()])

    a sigmoid on the LINEAR value. There is no log anywhere in it. This module's docstring used to
    claim "NumPyro picks the right bijector ... For LogUniform that is a log transform"; it is not,
    and the consequences are not cosmetic:

    * The unconstrained origin lands on the ARITHMETIC midpoint. For (100, 6000), ``u = 0`` maps to
      x = 3050 K -- the 83rd percentile of the log-uniform prior -- instead of the prior median
      ``sqrt(100 * 6000) = 774.6`` K.
    * NumPyro's default ``init_to_uniform(radius=2)`` draws u ~ U(-2, 2), which under that transform
      is x in [803.3, 5296.7] K -- the whole init distribution sits ABOVE the prior median (measured
      over 400 seeds: lowest init 818.0 K). A temperature floor whose true value is 107 K cannot be
      initialised anywhere near.
    * The step size cannot serve both ends. Measured, ``dx/du`` at the init edge u = -2 is 619.5
      against 6.99 at the truth -- a factor 88.6. In log10 coordinates the same two points differ by
      9.84, so one adapted step size covers the traverse instead of being wrong at one end or the
      other.

    ``Uniform(log10 lo, log10 hi)`` pushed through ``10**u`` is the identical measure -- it is the
    definition of log-uniform -- and puts ``u = 0`` on the prior median by construction. It is also
    exactly what ``whisper_cbpf.samplers.jax.pymc_gpu._pymc_prior`` does, so the two GPU NUTS arms
    now walk the SAME geometry and are comparable as parameterisations rather than differing in one.

    Simulation-based calibration on data generated FROM the model is what exposed this;
    ``tests/test_nuts_gpu_loguniform_geometry.py`` pins the geometry directly, without sampling, so
    the regression cannot come back silently.

    Only the distributions whisper's ABC/NUTS paths actually support are mapped; anything else
    raises rather than being quietly approximated.
    """
    out = []
    for nm in names:
        d = prior.distributions[nm]
        kind = type(d).__name__
        lo, hi = (float(x) for x in d.bounds)
        if kind == "Uniform":
            out.append((nm, dist.Uniform(lo, hi), None))
        elif kind == "LogUniform":
            out.append((f"log10_{nm}", dist.Uniform(np.log10(lo), np.log10(hi)),
                        lambda u: 10.0 ** u))
        else:
            raise TypeError(
                f"nuts_gpu cannot express prior {kind!r} on parameter {nm!r} as a NumPyro "
                f"distribution. Supported: Uniform, LogUniform. Extend _numpyro_priors rather "
                f"than falling back to a uniform box -- that changes the posterior silently.")
    return out


#: ``init_strategy=`` accepts these names as well as any NumPyro init callable. Spelled out rather
#: than resolved with ``getattr(numpyro.infer, ...)`` so a typo names the valid options instead of
#: reaching an arbitrary attribute.
_INIT_STRATEGIES = {
    "uniform": numpyro.infer.init_to_uniform,
    "median": numpyro.infer.init_to_median,
    "sample": numpyro.infer.init_to_sample,
    "feasible": numpyro.infer.init_to_feasible,
    "mean": numpyro.infer.init_to_mean,
}

#: What NumPyro's ``MCMC`` accepts. Checked here so a typo raises before XLA spends a minute
#: compiling the kernel, and with a message naming the alternatives.
_CHAIN_METHODS = ("vectorized", "parallel", "sequential")


def _resolve_init_strategy(init_strategy):
    """``None`` -> NumPyro's own default; a short name -> the matching callable; a callable -> itself.

    Returns ``(callable_or_None, label)``; the label is what goes into ``result.info`` so a saved
    run records which strategy actually ran rather than a repr of a function object.
    """
    if init_strategy is None:
        return None, "numpyro_default(init_to_uniform)"
    if isinstance(init_strategy, str):
        key = init_strategy.strip().lower()
        key = key[len("init_to_"):] if key.startswith("init_to_") else key
        if key not in _INIT_STRATEGIES:
            raise ValueError(
                f"nuts_gpu: unknown init_strategy {init_strategy!r}. Use one of "
                f"{sorted(_INIT_STRATEGIES)} (with or without the 'init_to_' prefix), or pass a "
                f"NumPyro init callable such as numpyro.infer.init_to_value(values=...).")
        return _INIT_STRATEGIES[key], f"init_to_{key}"
    if not callable(init_strategy):
        raise TypeError(
            f"nuts_gpu: init_strategy must be None, one of {sorted(_INIT_STRATEGIES)}, or a "
            f"NumPyro init callable; got {type(init_strategy).__name__}.")
    return init_strategy, getattr(init_strategy, "__name__", repr(init_strategy))


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

    def fit(self, lc, model, prior=None, *, log_prob_fn=None, num_warmup=1000, num_samples=2000,
            num_chains=4, space="auto", likelihood="auto", seed=0, progress=False,
            target_accept_prob=0.8, init_strategy=None, dense_mass=False, max_tree_depth=10,
            step_size=1.0, chain_method="vectorized") -> SamplerResult:
        """Fit ``lc`` with ``model`` via NumPyro NUTS.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve. Used for ``n_data``/``attach_*`` metrics, and — when
            ``log_prob_fn`` is not supplied — to build the density.
        model : whisper_cbpf.models.Model or str
            A registered model, by object or by name.
        prior : Prior, optional
            Defaults to ``model.default_prior``. Each parameter must be ``Uniform`` or
            ``LogUniform``; anything else raises rather than being approximated by its bounds.
        log_prob_fn : Callable[[jnp.ndarray], jnp.ndarray], optional
            A ``jax.jit``-compiled function mapping a flat ``theta`` vector (order =
            ``model.parameters``) to a scalar log-**likelihood**, ``-inf`` outside the prior box —
            not a log-posterior. The prior density is added separately by NumPyro's own per-
            parameter sample sites, so passing a posterior double-counts it.

            **Optional since the adapter layer landed.** When omitted, ``model.log_prob_jax`` is
            used if the factory filled it, and otherwise one is built from ``model.predict_jax``
            plus the likelihood implied by ``space`` (see
            :func:`whisper_cbpf.samplers.jax._adapters.make_log_prob_jax`). An explicitly passed
            callable always wins.
        num_warmup, num_samples, num_chains : int
            NUTS budget.
        init_strategy : str or callable, optional
            Where the chains start, in unconstrained space. ``None`` (default) leaves NumPyro's own
            ``init_to_uniform(radius=2)``. Accepts ``"uniform"``, ``"median"``, ``"sample"``,
            ``"feasible"``, ``"mean"`` (the ``init_to_`` prefix is optional) or any NumPyro init
            callable, e.g. ``numpyro.infer.init_to_value(values={"mej": 0.03})``. Worth reaching
            for: the default draws u ~ U(-2, 2) in unconstrained space, which for a bounded
            parameter is a *fraction of the prior*, not the prior — on ``LogUniform(100, 6000)``
            it covers only x in [163, 3683] K.
        dense_mass : bool, default False
            ``False`` adapts a diagonal mass matrix (NumPyro's default), ``True`` a full one.
            Dense costs O(k^2) memory and a slower warmup, and pays for itself only when the
            posterior has strong linear correlations — which the kilonova's mej/vej/kappa do.
        max_tree_depth : int, default 10
            Doubling cap per NUTS trajectory: at most ``2**max_tree_depth`` leapfrog steps. Raise
            it when the sampler saturates (a badly conditioned posterior needs longer
            trajectories); every increment doubles the worst-case cost per draw.
        step_size : float, default 1.0
            *Initial* leapfrog step size. Adaptation is on, so this is a starting guess, not the
            step actually used.
        chain_method : {"vectorized", "parallel", "sequential"}, default "vectorized"
            How chains are placed. ``"vectorized"`` batches all chains onto one device via vmap —
            the standard single-GPU multi-chain NumPyro pattern, and the only one that uses a
            single card fully. ``"parallel"`` needs one device per chain and NumPyro **silently
            downgrades it to "sequential"** otherwise (see
            ``whisper_cbpf.samplers.jax.pymc_gpu._effective_chain_method``), so what you asked for
            and what runs can differ; ``info["chain_method"]`` records the request.
        space : {"auto", "flux", "magnitude"}
            **Live on the auto-built path**, where it selects the likelihood: ``"auto"`` follows
            ``lc.data_mode``, the same rule ``abc``/``abc_smc``/``mcmc`` use, so magnitude data is
            fitted in magnitude space. It used to default to ``"flux"`` and be recorded but never
            used, which meant a magnitude light curve was silently scored in flux by every
            predictive metric.

            When *you* pass ``log_prob_fn``, the space is fixed inside your density and this
            argument cannot change it — it is then only a label for the metrics, and
            ``info["space_source"]`` records which case applies.

        Returns
        -------
        SamplerResult
            Warmup/sampling time reported separately in ``info``, alongside divergences and
            per-parameter r-hat (via arviz). The raw NumPyro ``MCMC`` object is attached as
            ``result.numpyro_mcmc``.
        """
        # Resolve a NAME through the registry, but pass any Model-LIKE object straight through:
        # the pre-audit docstring advertised "Needs .name and .parameters", and get_model() on a
        # duck-typed object raises KeyError (or TypeError if it is unhashable). Only strings need
        # looking up.
        model = get_model(model) if isinstance(model, str) else model
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        names, likelihood = _resolve_sampling_names(lc, model, prior, log_prob_fn, space,
                                                   likelihood)
        k, n = len(names), int(len(lc.time))
        # Validate the prior and the kernel options BEFORE building anything: an unsupported
        # distribution or a mistyped option should report itself, not hide behind a filter-set
        # build or a minute of XLA compilation.
        site_specs = _numpyro_priors(prior, names)
        init_fn, init_label = _resolve_init_strategy(init_strategy)
        if chain_method not in _CHAIN_METHODS:
            raise ValueError(f"nuts_gpu: chain_method must be one of {_CHAIN_METHODS}, got "
                             f"{chain_method!r}.")
        log_prob_fn, space, space_source, fn_source = _resolve_density(
            lc, model, prior, log_prob_fn, space, names, include_prior=False,
            sampler=self.name, likelihood=likelihood)

        def numpyro_model():
            # One site per parameter so each keeps its OWN density and support transform. A
            # reparameterised site (log-uniform -> uniform on log10) is pushed back to the
            # parameter's own units by a `deterministic`, which NumPyro collects into
            # `get_samples()` alongside the sampled sites -- so `samples["temperature_floor"]` is
            # kelvin, exactly as before, while the chain walks log10 kelvin.
            values = []
            for nm, (site, d, to_linear) in zip(names, site_specs):
                v = numpyro.sample(site, d)
                values.append(v if to_linear is None else numpyro.deterministic(nm, to_linear(v)))
            # Stacked back into the flat theta vector log_prob_fn expects, in model.parameters order
            numpyro.factor("log_likelihood", log_prob_fn(jnp.stack(values)))

        kernel_kw = {} if init_fn is None else {"init_strategy": init_fn}
        kernel = NUTS(numpyro_model, target_accept_prob=float(target_accept_prob),
                      dense_mass=bool(dense_mass), max_tree_depth=int(max_tree_depth),
                      step_size=float(step_size), **kernel_kw)
        mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples),
                    num_chains=int(num_chains), progress_bar=progress, chain_method=chain_method)

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
            "space_source": space_source,
            # `_LIKELIHOOD_KINDS` in samplers/base.py maps this back to a registry name, so WAIC
            # and predictive_metrics re-score under the density that was actually fitted. None
            # when the caller supplied their own log_prob_fn: we cannot know then.
            "likelihood": getattr(log_prob_fn, "likelihood", None),
            "scatter_param": getattr(log_prob_fn, "scatter_param", None),
            "log_prob_fn": fn_source,
            "x64": bool(jax.config.jax_enable_x64),
            "target_accept_prob": float(target_accept_prob),
            # The kernel hyper-parameters that actually ran. Without these a saved run cannot be
            # reproduced or compared -- two fits differing only in dense_mass or max_tree_depth
            # were previously indistinguishable in the record.
            "init_strategy": init_label, "dense_mass": bool(dense_mass),
            "max_tree_depth": int(max_tree_depth), "step_size": float(step_size),
            "chain_method": chain_method,
            # The site each parameter is actually sampled on. Differs from the parameter name
            # exactly where a reparameterisation is in play, which is the thing worth recording.
            "sample_sites": {nm: site for nm, (site, _d, _f) in zip(names, site_specs)},
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
            # max(n, 1) matches pymc_gpu. Unguarded, an empty light curve gives np.log(0) = -inf
            # (with a RuntimeWarning that nothing reads), so BIC came back -inf -- which compares
            # as the BEST possible model in any argmin over BIC, rather than as the error it is.
            bic=float(-2.0 * max_log_likelihood + k * np.log(max(n, 1))),
        )
        result.numpyro_mcmc = mcmc
        result.samples_by_chain = np.asarray(samples_by_chain)   # (chains, draws, k) -- for ESS/sec
        attach_predictive_metrics(result, lc, space)
        return result


def fit_NUTSGPU(lc, model, prior=None, *, log_prob_fn=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via NumPyro NUTS on GPU. See :meth:`NUTSGPUSampler.fit`.

    ``log_prob_fn`` is optional here for the same reason it is on the method: omit it and one is
    built from ``model.predict_jax``. It stayed *required* on this wrapper after the method relaxed
    it, so the public function could not reach the auto-build path the method advertises --
    ``NUTSGPUSampler().fit(...)`` and ``wp.fit(...)`` worked while ``fit_NUTSGPU(...)`` raised
    ``TypeError: missing 1 required keyword-only argument``.
    """
    return NUTSGPUSampler().fit(lc, model, prior=prior, log_prob_fn=log_prob_fn, **kwargs)
