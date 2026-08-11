"""The two emcee arms of the benchmark.

Arm A(1) — ``fit_emcee_jax``: emcee driving the SAME jitted JAX log-density the NUTS arm uses,
in ``vectorize=True`` mode so each step is one batched GPU call over the whole walker ensemble
rather than one Python call per walker. CPU-orchestrated, GPU-compute.

Arm A(2) — ``fit_emcee_numpy``: the realistic baseline. A pure-numpy log-posterior with no JAX
anywhere, parallelized across processes like ``whisper_labia.samplers.mcmc``'s ``n_jobs``. This
is what a scientist runs today with no GPU/JAX infrastructure, and it must NOT benefit from any
of the JAX work — that's the whole point of having it as a separate arm.

Both return a ``whisper_labia.samplers.base.SamplerResult`` built through the same helpers every
whisper_labia sampler uses, so all four arms of the benchmark share one output schema.
"""
from __future__ import annotations

import os
import sys
import time

import emcee
import numpy as np
import pandas as pd

# whisper_labia is imported directly: whisper-GPU is cloned INTO the whisper checkout as
# `whisper_gpu/`, so the repo root that holds both is already on sys.path. The previous
# three-level dirname() walk-up hard-coded this file's nesting depth and broke on any move.

from whisper_labia.samplers.base import (  # noqa: E402
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)


from ..models._flare_spec import PARAMETERS as _PARAMS  # noqa: E402
from ..models._flare_spec import flare_flux_numpy  # noqa: E402


def _flux_numpy(theta, times):
    """Arm A(2)'s forward model: pure numpy, no JAX anywhere on this path.

    Delegates to ``models.flare_spec.flare_flux_numpy`` — that module imports numpy only, so
    the no-JAX-infrastructure property of this arm is preserved while guaranteeing it evaluates
    the same function the rest of the benchmark does.
    """
    return flare_flux_numpy(dict(zip(_PARAMS, theta)), times)


def _log_prob_numpy(theta, times, values, sigmas, lows, highs):
    """Module-level (picklable, for multiprocessing) numpy log-posterior. Flat prior box."""
    if np.any(theta < lows) or np.any(theta > highs):
        return -np.inf
    mf = _flux_numpy(theta, times)
    if not np.all(np.isfinite(mf)):
        return -np.inf
    return float(-0.5 * np.sum(((values - mf) / sigmas) ** 2 + np.log(2 * np.pi * sigmas ** 2)))


def _finish(sampler_name, lc, model, names, flat, ll_flat, runtime, info, space,
            compile_time_s=0.0):
    """Shared tail: build the SamplerResult exactly like every whisper_labia sampler does."""
    samples = pd.DataFrame(flat, columns=names)
    k, n = len(names), int(len(lc.time))

    best_idx = int(np.nanargmax(ll_flat))
    best_params = {nm: float(flat[best_idx, j]) for j, nm in enumerate(names)}
    max_log_likelihood = float(ll_flat[best_idx])

    info["compile_time_s"] = float(compile_time_s)
    attach_band_metrics(info, lc, model.name, best_params, space)
    result = SamplerResult(
        sampler=sampler_name, model=model.name, parameters=list(names), samples=samples,
        summary=summarize_posterior(samples, names), best_params=best_params,
        n_data=n, n_params=k, runtime_s=float(runtime), info=info,
        max_log_likelihood=max_log_likelihood,
        aic=float(-2.0 * max_log_likelihood + 2 * k),
        bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
    )
    attach_predictive_metrics(result, lc, space)
    return result


def _init_walkers(nwalkers, lows, highs, seed):
    """Spread walkers uniformly across the prior box (no initial guess needed — matches how
    whisper_labia's MCMCSampler initializes when ``initial_guess`` is None)."""
    rng = np.random.default_rng(int(seed))
    return rng.uniform(lows, highs, size=(nwalkers, len(lows)))


def fit_emcee_jax(lc, model, log_prob_fn, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000,
                  thin=10, seed=0, progress=False, space="flux") -> SamplerResult:
    """Arm A(1): emcee vectorized against the shared jitted JAX log-density.

    ``log_prob_fn`` is the SAME object the NUTS arm receives — passed by reference, not rebuilt,
    so the two arms provably evaluate an identical log-density and the comparison isolates the
    sampling algorithm rather than NumPy-vs-XLA.

    emcee's ``vectorize=True`` hands the whole ``(nwalkers, ndim)`` ensemble to the callable at
    once, so one jitted+vmapped GPU call replaces ``nwalkers`` Python-level calls per step. The
    numpy<->JAX conversion happens only at that boundary; emcee itself never sees a JAX type, so
    there's no interaction with JAX tracing.
    """
    import jax
    import jax.numpy as jnp

    prior = prior if prior is not None else model.default_prior
    names = list(model.parameters)
    lows = np.array([prior.distributions[nm].bounds[0] for nm in names], dtype=float)
    highs = np.array([prior.distributions[nm].bounds[1] for nm in names], dtype=float)
    ndim = len(names)

    batched = jax.jit(jax.vmap(log_prob_fn))

    def log_prob_batch(theta_batch):
        out = batched(jnp.asarray(theta_batch, dtype=jnp.float32))
        return np.asarray(jax.block_until_ready(out), dtype=float)

    # Compile on a throwaway ensemble so JIT time is reported separately from sampling time.
    t_c = time.perf_counter()
    log_prob_batch(_init_walkers(nwalkers, lows, highs, seed))
    compile_time = time.perf_counter() - t_c

    p0 = _init_walkers(nwalkers, lows, highs, seed)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_batch, vectorize=True)
    sampler._random.seed(int(seed))

    t0 = time.perf_counter()
    sampler.run_mcmc(p0, int(nsteps), progress=progress)
    runtime = time.perf_counter() - t0

    flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    # Flat prior => log-posterior == log-likelihood + const, and log_prob_fn is already pure
    # log-likelihood (-inf outside the box), so its values are directly the log-likelihoods.
    ll_flat = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)

    try:
        autocorr = float(np.nanmean(sampler.get_autocorr_time(tol=0)))
    except Exception:
        autocorr = float("nan")
    info = {
        "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
        "space": space, "seed": int(seed),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "mean_autocorr_time": autocorr,
        "converged": bool(np.isfinite(autocorr) and autocorr > 0 and nsteps >= 50.0 * autocorr),
        "backend": "emcee + jitted JAX log_prob (vectorized)",
    }
    result = _finish("emcee_jax", lc, model, names, flat, ll_flat, runtime, info, space,
                     compile_time_s=compile_time)
    # (chains, draws, k) for ArviZ ESS — emcee walkers act as the chain axis
    chain = sampler.get_chain(discard=int(burnin), thin=int(thin))       # (steps, walkers, k)
    result.samples_by_chain = np.swapaxes(chain, 0, 1)
    result.emcee_sampler = sampler
    return result


def fit_emcee_numpy(lc, model, times, values, sigmas, *, prior=None, nwalkers=32, nsteps=5000,
                    burnin=1000, thin=10, seed=0, n_jobs=8, progress=False,
                    space="flux") -> SamplerResult:
    """Arm A(2): plain-numpy emcee on N CPU cores — the realistic no-GPU baseline.

    No JAX involvement anywhere: its own numpy forward model, its own numpy log-posterior,
    parallelized over processes exactly like ``whisper_labia.samplers.mcmc``'s ``n_jobs`` path.

    **Run this in a process that has never imported JAX.** The pool uses ``fork`` (matching
    whisper_labia's MCMCSampler), and forking a process with JAX's threads already running
    risks deadlock — JAX itself warns about exactly this. ``run_benchmark.py`` therefore
    dispatches this arm as its own subprocess. Running it inside a JAX process may appear to
    work and then hang unpredictably, which is worse than failing outright.
    """
    import multiprocessing

    if "jax" in sys.modules:
        raise RuntimeError(
            "fit_emcee_numpy must run in a JAX-free process: forking with JAX's threads live "
            "risks deadlock, and a JAX-warmed process would also distort this arm's timing "
            "(it is meant to be the no-JAX-infrastructure baseline). Dispatch it as its own "
            "subprocess -- see run_benchmark.py.")

    prior = prior if prior is not None else model.default_prior
    names = list(model.parameters)
    lows = np.array([prior.distributions[nm].bounds[0] for nm in names], dtype=float)
    highs = np.array([prior.distributions[nm].bounds[1] for nm in names], dtype=float)
    ndim = len(names)

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    p0 = _init_walkers(nwalkers, lows, highs, seed)

    pool = multiprocessing.get_context("fork").Pool(int(n_jobs)) if n_jobs and n_jobs > 1 else None
    try:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, _log_prob_numpy, pool=pool,
            args=(times, values, sigmas, lows, highs))
        sampler._random.seed(int(seed))
        t0 = time.perf_counter()
        sampler.run_mcmc(p0, int(nsteps), progress=progress)
        runtime = time.perf_counter() - t0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
    ll_flat = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)

    try:
        autocorr = float(np.nanmean(sampler.get_autocorr_time(tol=0)))
    except Exception:
        autocorr = float("nan")
    info = {
        "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
        "space": space, "seed": int(seed), "n_jobs": int(n_jobs),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "mean_autocorr_time": autocorr,
        "converged": bool(np.isfinite(autocorr) and autocorr > 0 and nsteps >= 50.0 * autocorr),
        "backend": f"emcee + numpy log_prob ({n_jobs} processes)",
    }
    result = _finish("emcee_numpy", lc, model, names, flat, ll_flat, runtime, info, space,
                     compile_time_s=0.0)     # nothing to compile — that's the point of this arm
    chain = sampler.get_chain(discard=int(burnin), thin=int(thin))
    result.samples_by_chain = np.swapaxes(chain, 0, 1)
    result.emcee_sampler = sampler
    return result


class EmceeJAXSampler(BaseSampler):
    """whisper registry adapter for :func:`fit_emcee_jax`.

    The function form predates the registry and is kept as the public entry point (the flare
    benchmark calls it directly). This class is the thin ``BaseSampler`` shell whisper needs:
    ``get_sampler(name)`` constructs with zero arguments, so every option travels through ``fit``.

    Registered, unlike ``snpe_gpu``, because this sampler is genuinely model-agnostic -- it consumes
    a ``log_prob_fn`` and never touches a specific forward model. ``fit_emcee_numpy`` in this same
    module is NOT registered for the opposite reason: it hard-wires the flare spec.
    """

    name = "emcee_jax"

    def fit(self, lc, model, prior=None, *, log_prob_fn, **kwargs) -> SamplerResult:
        """Fit via emcee against a shared jitted JAX log-density. See :func:`fit_emcee_jax`.

        ``log_prob_fn`` is required and keyword-only, exactly as for ``nuts_gpu`` -- the two arms
        are meant to receive the *same object* so a comparison between them isolates the sampling
        algorithm rather than the log-density implementation.
        """
        from whisper_labia.models import get_model
        return fit_emcee_jax(lc, get_model(model), log_prob_fn, prior=prior, **kwargs)
