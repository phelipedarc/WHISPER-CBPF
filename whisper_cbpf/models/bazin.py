"""Bazin (2009) supernova light-curve model -- the classic rise + fall.

``flux(t) = A * exp(-(t - t0) / tau_fall) / (1 + exp(-(t - t0) / tau_rise))``

Band-independent and vectorized; computed in log-space for numerical stability, so extreme prior
draws stay finite and (for ``tau_rise < tau_fall``) the flux correctly decays toward 0 far before
the peak. ``t`` is days since explosion.
"""
from __future__ import annotations

import numpy as np

from ..priors import Prior, Uniform

PARAMETERS = ["amplitude", "t0", "tau_rise", "tau_fall"]
DESCRIPTION = "Bazin (2009): A*exp(-(t-t0)/tau_fall) / (1 + exp(-(t-t0)/tau_rise))."

#: Clip the log-flux before ``exp()`` just under float64's exp overflow (exp(709) is the last finite value).
_LOG_FLUX_CLIP = 700.0

PRIOR = Prior({
    "amplitude": Uniform(0.0, 10.0),
    "t0": Uniform(-10.0, 30.0),
    "tau_rise": Uniform(0.1, 20.0),
    "tau_fall": Uniform(0.5, 60.0),
})


def bazin_flux(parameters, times, bands=None):
    """A * exp(-dt/tau_fall) / (1 + exp(-dt/tau_rise)) with dt = t - t0, evaluated stably in log-space."""
    amplitude = parameters["amplitude"]
    t0 = parameters["t0"]
    tau_rise = parameters["tau_rise"]
    tau_fall = parameters["tau_fall"]
    dt = np.asarray(times, dtype=float) - t0
    log_flux = -dt / tau_fall - np.logaddexp(0.0, -dt / tau_rise)
    return amplitude * np.exp(np.clip(log_flux, -_LOG_FLUX_CLIP, _LOG_FLUX_CLIP))
