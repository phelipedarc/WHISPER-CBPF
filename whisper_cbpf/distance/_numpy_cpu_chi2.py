"""Distance metrics for ABC (and pseudo-likelihoods for model selection).

A distance has the signature ``f(obs_flux, obs_flux_err, sim_flux, bands) -> float`` and is treated
as a black box by the sampler -- plug in any custom metric.
"""
from __future__ import annotations

import numpy as np


def chi2_distance(obs_flux, obs_flux_err, sim_flux, bands=None):
    """Multi-band chi-square: ``sum(((obs - sim) / err)**2)``.

    Summing over all points is equivalent to summing per band and adding. Numerically this equals
    ``-2 ln L`` for an independent Gaussian likelihood, which is what lets ABC report AIC/BIC.
    """
    obs_flux = np.asarray(obs_flux, dtype=float)
    sim_flux = np.asarray(sim_flux, dtype=float)
    obs_flux_err = np.asarray(obs_flux_err, dtype=float)
    residual = (obs_flux - sim_flux) / obs_flux_err
    return float(np.sum(residual * residual))


# --- registry (so distances are discoverable / usable by name, like models/samplers/likelihoods) ---
_DISTANCES = {"chi2": chi2_distance, "chi_square": chi2_distance}


def register_distance(name, fn, *, overwrite=False):
    """Register a distance ``f(obs_flux, obs_flux_err, sim_flux, bands) -> float`` under ``name``."""
    key = str(name).lower()
    if key in _DISTANCES and not overwrite:
        raise ValueError(f"Distance {name!r} already registered (pass overwrite=True).")
    _DISTANCES[key] = fn


def get_distance(distance):
    """Resolve ``distance``: a callable passes through; a registered name is looked up."""
    if callable(distance):
        return distance
    key = str(distance).lower()
    if key not in _DISTANCES:
        raise KeyError(f"Unknown distance {distance!r}. Available: {list_distances()}")
    return _DISTANCES[key]


def list_distances():
    """Sorted list of registered distance names."""
    return sorted(_DISTANCES)
