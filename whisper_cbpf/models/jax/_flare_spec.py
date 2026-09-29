"""JAX-free definition of the flare model: parameter names, prior box, and a numpy forward model.

Kept separate from ``flare_jax`` so arm A(2) (the pure-numpy emcee baseline) can build a model
object **without importing JAX at all** — that arm exists to represent what a scientist runs
today with no GPU/JAX infrastructure, and emcee's ``fork`` pool can deadlock in a process where
JAX's threads are live.

``flare_jax`` imports the parameter order and prior box from here, so the two paths cannot drift
apart on the one thing they must agree on.
"""
from __future__ import annotations


import numpy as np

PARAMETERS = ["log_amp", "log_sigma", "log_tau", "t0"]

# Uniform on each LOG parameter (plus linear t0). Deliberately not whisper_cbpf's LogUniform:
# that samples a *linear* value under a log-uniform density, whereas these parameters are
# already in log space, so a plain Uniform on the log parameter is the correct prior.
PRIOR_BOUNDS = [(-2.0, 3.0),    # log_amp   -> amp   in ~[0.14, 20.1]
                (-2.0, 3.0),    # log_sigma -> sigma in ~[0.14, 20.1] days
                (-2.0, 4.0),    # log_tau   -> tau   in ~[0.14, 54.6] days
                (0.0, 30.0)]    # t0, days (matches gaussian_rise.py's existing bound)


def _whisper():
    import whisper_cbpf.priors as pr
    from ...models import Model
    return pr, Model


def default_prior():
    """The model's prior as a ``whisper_cbpf.priors.Prior``."""
    pr, _ = _whisper()
    return pr.Prior({nm: pr.Uniform(lo, hi) for nm, (lo, hi) in zip(PARAMETERS, PRIOR_BOUNDS)})


def flare_flux_numpy(parameters, times, bands=None):
    """Gaussian rise / exponential decay in pure numpy (float64).

    Module-level so it stays picklable for multiprocessing pools. Follows whisper_cbpf's
    ``predict(parameters: dict, times, bands) -> flux`` contract; ``bands`` is accepted and
    ignored because the model is band-independent by design.
    """
    la = float(parameters["log_amp"])
    ls = float(parameters["log_sigma"])
    lt = float(parameters["log_tau"])
    t0 = float(parameters["t0"])
    amp, sig, tau = np.exp(la), np.exp(ls), np.exp(lt)
    dt = np.asarray(times, dtype=np.float64) - t0
    out = np.empty_like(dt)
    rise = dt < 0
    # clamp the exponent so float64 never overflows/underflows to inf or a hard zero
    out[rise] = amp * np.exp(-np.minimum((dt[rise] ** 2) / (2.0 * sig ** 2), 80.0))
    out[~rise] = amp * np.exp(-np.minimum(dt[~rise] / tau, 80.0))
    return out


def get_model_numpy():
    """A ``whisper_cbpf.models.Model`` backed by the numpy forward model. No JAX imported.

    Not registered in whisper_cbpf's model registry — this is a standalone pre-integration
    registry by this function; whisper_cbpf registers the JAX variant. Constructed directly so it
    satisfies the same
    ``.name`` / ``.predict`` / ``.parameters`` / ``.default_prior`` contract.
    """
    _, Model = _whisper()
    return Model(
        name="flare_jax",          # same name as the JAX model: same model, different backend
        predict=flare_flux_numpy,
        parameters=list(PARAMETERS),
        default_prior=default_prior(),
        description="Gaussian-rise/exponential-decay flare, log-space parameters (numpy backend).",
    )
