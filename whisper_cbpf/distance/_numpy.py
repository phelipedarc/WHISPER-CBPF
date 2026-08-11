"""ABC distance metrics — numpy backend (the default).

Assembled from two sources, function bodies unchanged:

* ``chi2_distance`` — WHISPER_AI @ 8ba3843, ``whisper_labia/distance.py``
* ``mse_distance``, ``rmse_distance``, ``mae_distance``, ``wmse_distance``, ``wmae_distance``
  — whisper-GPU @ 10796a0, ``distances.py``

The registry lives in this subpackage's ``__init__``; this module holds implementations only.
The jnp versions are in ``_jax.py`` and are never imported from here.

Which metric to use, and why it matters more than it looks
----------------------------------------------------------
The distance defines what "close" means, so it defines the posterior. The three families here are
not interchangeable:

* **chi2 / wmse / wmae** divide by the per-point error, so each point contributes in units of its
  own uncertainty. ``chi2`` additionally equals ``-2 ln L`` for an independent Gaussian, which is
  what lets ABC report an AIC/BIC on the same scale as the likelihood-based samplers.
* **mse / rmse / mae** are unweighted. A bright, well-measured epoch and a faint, noisy one count
  the same, so these follow the *shape* of the light curve rather than its statistical agreement.
  In magnitude space that is often what you want (magnitudes are already a log scale); in flux
  space, where a kilonova spans several decades, an unweighted metric is dominated by peak epochs
  and will effectively ignore the tail.
* **mae / wmae** use absolute rather than squared residuals, so a single bad epoch cannot dominate.
  Worth reaching for when the photometry has outliers a Gaussian likelihood would fight.

A note on the scatter parameter
-------------------------------
None of these can *fit* a noise-scale parameter. Extra simulated noise only ever increases the
expected residual, so a rejection distance is monotonically penalised by it and the posterior
rails to the smallest allowed scatter -- the parameter is not identifiable by distance-based ABC at
all. It is a *likelihood* parameter. Fit it with a likelihood-based sampler, or omit it from the
ABC prior (which is what whisper's own AT2017GFO analysis does). ``abc_gpu`` warns if you try.
"""
from __future__ import annotations

import numpy as np

__all__ = ["chi2_distance", "mse_distance", "rmse_distance", "mae_distance",
           "wmse_distance", "wmae_distance", "NUMPY_DISTANCES"]


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


# Signature is whisper's: distance(obs, obs_err, sim, bands=None) -> float. `bands` is accepted for
# API compatibility and ignored, exactly as chi2_distance does -- a flat sum over all points is
# equivalent to summing per band and adding.

def mse_distance(obs, obs_err, sim, bands=None):
    """Mean squared error, unweighted: ``mean((obs - sim)**2)``."""
    r = np.asarray(obs, float) - np.asarray(sim, float)
    return float(np.mean(r * r))


def rmse_distance(obs, obs_err, sim, bands=None):
    """Root mean squared error. Monotone in ``mse``, so it selects the SAME draws at a matched
    quantile -- it changes only the numeric scale of epsilon, which matters if you set an absolute
    ``threshold`` rather than a quantile."""
    r = np.asarray(obs, float) - np.asarray(sim, float)
    return float(np.sqrt(np.mean(r * r)))


def mae_distance(obs, obs_err, sim, bands=None):
    """Mean absolute error: ``mean(|obs - sim|)``. Outlier-tolerant."""
    return float(np.mean(np.abs(np.asarray(obs, float) - np.asarray(sim, float))))


def wmse_distance(obs, obs_err, sim, bands=None):
    """Error-weighted MSE: ``mean(((obs - sim)/err)**2)``. This is ``chi2 / n_points``."""
    r = (np.asarray(obs, float) - np.asarray(sim, float)) / np.asarray(obs_err, float)
    return float(np.mean(r * r))


def wmae_distance(obs, obs_err, sim, bands=None):
    """Error-weighted MAE: ``mean(|obs - sim| / err)``. Outlier-tolerant and error-aware."""
    r = (np.asarray(obs, float) - np.asarray(sim, float)) / np.asarray(obs_err, float)
    return float(np.mean(np.abs(r)))


NUMPY_DISTANCES = {
    "chi2": chi2_distance, "mse": mse_distance, "rmse": rmse_distance,
    "mae": mae_distance, "wmse": wmse_distance, "wmae": wmae_distance,
}
