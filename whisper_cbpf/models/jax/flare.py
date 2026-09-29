"""JAX-native flare model: Gaussian rise, exponential decay, log-space parameters.

    F(t) = A * exp(-(t-t0)^2 / (2*sigma^2))   for t <  t0   (rise)
    F(t) = A * exp(-(t-t0)/tau)                for t >= t0   (decay)

Parameters, in this fixed order everywhere in this module: ``log_amp, log_sigma,
log_tau, t0`` (amplitude/sigma/tau are strictly positive by construction, so they're
sampled/optimized in log space; t0 stays linear).

It is JAX-native, so it lives in whisper_cbpf.models.jax, is resolved lazily, and needs the
[gpu] extra to run. It mirrors whisper_cbpf's `Model.predict(params,
times, bands) -> flux` contract via `predict_numpy` below, so it drops into
existing plotting/data-generation helpers that expect that shape.

**Double-where, NaN-safe gradients.** ``jnp.where(cond, f(x), g(x))`` evaluates
BOTH ``f`` and ``g`` on the raw ``x`` for every element (that's how its VJP is
defined), then masks. If the *unselected* branch would overflow on the raw
input (e.g. evaluating the decay branch's ``exp(-dt/tau)`` at a very negative
``dt`` from deep in the rise region blows up to +inf), the resulting NaN/Inf
survives being multiplied by a zero mask in the backward pass (``0 * inf =
nan``) and contaminates the gradient even though it never appears in the
forward output. Fix: sanitize each branch's input to a safe value (0) on the
side where it isn't used, *before* calling the branch function, so neither
branch ever sees an out-of-domain input regardless of which side is selected.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Parameter order and prior box live in the JAX-free spec module so the numpy baseline arm
# (which must never import JAX) and this module cannot drift apart.
from ._flare_spec import PARAMETERS, PRIOR_BOUNDS, default_prior  # noqa: F401


def _flux_scalar(log_amp, log_sigma, log_tau, t0, t):
    """F(t) for one time point and one parameter set. Pure JAX, no Python control flow."""
    amp = jnp.exp(log_amp)
    sigma = jnp.exp(log_sigma)
    tau = jnp.exp(log_tau)
    dt = t - t0
    is_rise = dt < 0.0
    dt_rise = jnp.where(is_rise, dt, 0.0)     # sanitized for the rise branch
    dt_decay = jnp.where(is_rise, 0.0, dt)    # sanitized for the decay branch
    rise = amp * jnp.exp(-(dt_rise ** 2) / (2.0 * sigma ** 2))
    decay = amp * jnp.exp(-dt_decay / tau)
    return jnp.where(is_rise, rise, decay)


def predict_jax(params, times):
    """Batched/vmapped flux.

    ``params``: dict with keys ``log_amp, log_sigma, log_tau, t0``, each either a
    JAX scalar (single parameter set) or a 1-D array of shape ``(n_draws,)`` (a
    batch of parameter sets evaluated against the SAME ``times`` grid — the
    per-observation case: one grid, many theta draws, whether from NUTS chains
    or SNPE's simulate-for-training batch).
    ``times``: 1-D array, shape ``(n_points,)``.

    Returns flux of shape ``(n_points,)`` for scalar params, or
    ``(n_draws, n_points)`` for batched params.
    """
    log_amp, log_sigma, log_tau, t0 = (
        params["log_amp"], params["log_sigma"], params["log_tau"], params["t0"]
    )
    over_times = jax.vmap(_flux_scalar, in_axes=(None, None, None, None, 0))
    if jnp.ndim(log_amp) == 0:
        return over_times(log_amp, log_sigma, log_tau, t0, times)
    over_theta = jax.vmap(over_times, in_axes=(0, 0, 0, 0, None))
    return over_theta(log_amp, log_sigma, log_tau, t0, times)


def _session_dtype():
    """``jnp.float64`` under ``jax_enable_x64``, else ``jnp.float32``: the session's precision."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def predict_numpy(parameters, times, bands=None):
    """whisper_cbpf's ``Model.predict(parameters: dict, times, bands) -> flux`` contract.

    The epochs follow the session's precision. They were cast to float32, so an x64 session still
    evaluated the flare in single precision, where an MJD-scale clock moves in 0.0039 d steps:
    2.3e-2 relative off the float64 NumPy twin on a 0.3-d rise at MJD 59000.
    """
    params = {k: jnp.asarray(float(parameters[k])) for k in PARAMETERS}
    return np.asarray(predict_jax(params, jnp.asarray(times, dtype=_session_dtype())))


def make_log_prob_jax(times, values, sigmas, prior_bounds):
    """Build a ``jax.jit``-compiled ``log_prob(theta_vec) -> scalar`` log-posterior.

    ``theta_vec``: flat array ``[log_amp, log_sigma, log_tau, t0]`` (matches
    ``PARAMETERS`` order). ``prior_bounds``: sequence of ``(low, high)`` per
    parameter, same order — a flat/uniform prior over that box.

    This is the ONE object both the vectorized-emcee arm and the NUTS arm call,
    by reference, so the sampler comparison isolates the sampling algorithm
    rather than any difference in how the log-density is computed.

    Arrays follow the session's precision (float64 under ``jax_enable_x64``). They were
    hard-coded float32, so an x64 session still evaluated the flare in single precision, where
    an MJD-scale ``t0`` moves in 0.0039 d steps and chains freeze on the staircase.
    """
    dt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    times = jnp.asarray(times, dtype=dt)
    values = jnp.asarray(values, dtype=dt)
    sigmas = jnp.asarray(sigmas, dtype=dt)
    lows = jnp.array([b[0] for b in prior_bounds], dtype=dt)
    highs = jnp.array([b[1] for b in prior_bounds], dtype=dt)

    def log_prob(theta_vec):
        in_bounds = jnp.all((theta_vec >= lows) & (theta_vec <= highs))
        params = {
            "log_amp": theta_vec[0], "log_sigma": theta_vec[1],
            "log_tau": theta_vec[2], "t0": theta_vec[3],
        }
        model_flux = predict_jax(params, times)
        resid = (values - model_flux) / sigmas
        ll = -0.5 * jnp.sum(resid ** 2 + jnp.log(2.0 * jnp.pi * sigmas ** 2))
        return jnp.where(in_bounds, ll, -jnp.inf)

    return jax.jit(log_prob)


def predict_torch(theta_batch, times, device="cuda"):
    """GPU-batched simulator for SNPE: ``(B, 4)`` torch params -> ``(B, n)`` torch flux.

    Runs the JAX model on GPU and hands the result to torch via the zero-copy
    ``__dlpack__`` protocol (``torch.from_dlpack`` / ``jnp.from_dlpack``), so
    SNPE's simulate-for-training step never leaves the GPU and never round-trips
    through a CPU numpy array — this is what makes simulation GPU-bound instead
    of the CPU-bound per-row loop ``whisper_cbpf``'s existing SNPE sampler falls
    back to when no ``predict_torch`` is supplied.

    Parameters, epochs and the returned flux follow the session's precision (float64 under
    ``jax_enable_x64``), as :func:`predict_numpy` does; they were cast to float32.
    """
    import torch

    want = torch.float64 if jax.config.jax_enable_x64 else torch.float32
    theta_jax = jnp.from_dlpack(theta_batch.contiguous().to(want))
    times_np = times.detach().cpu().numpy() if isinstance(times, torch.Tensor) else np.asarray(times)
    times_jax = jnp.asarray(times_np, dtype=_session_dtype())

    params = {
        "log_amp": theta_jax[:, 0], "log_sigma": theta_jax[:, 1],
        "log_tau": theta_jax[:, 2], "t0": theta_jax[:, 3],
    }
    flux_jax = predict_jax(params, times_jax)              # (B, n_points), computed on GPU
    flux_jax = jax.block_until_ready(flux_jax)              # pin down compute time before handoff
    flux_jax = jnp.nan_to_num(flux_jax, nan=0.0, posinf=0.0, neginf=0.0)

    flux_torch = torch.from_dlpack(flux_jax)                # zero-copy GPU->GPU
    return flux_torch.to(device)


def prior_bounds_flat():
    """``[(low, high), ...]`` in PARAMETERS order (from the shared JAX-free spec)."""
    return list(PRIOR_BOUNDS)


def get_model():
    """A ``whisper_cbpf.models.Model`` instance for this model — NOT registered in whisper_cbpf's
    registry by this function -- :mod:`whisper_cbpf` registers it as ``flare_jax`` on import.
    It satisfies the same ``.name``/``.predict``/``.parameters``/``.default_prior`` contract every
    whisper_cbpf model does.
    """
    from ...models import Model

    return Model(
        name="flare_jax",
        predict=predict_numpy,
        parameters=list(PARAMETERS),
        default_prior=default_prior(),
        description="JAX Gaussian-rise/exponential-decay flare, log-space parameters "
                     "(JAX/GPU; registered by whisper_cbpf as 'flare_jax').",
    )
