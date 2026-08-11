"""ABC distance metrics, in numpy (for whisper's CPU ABC) and jnp (for ``abc_gpu``).

whisper ships exactly one registered distance, ``chi2`` (aliased ``chi_square``), and an unused
``register_distance`` extension point. This module fills that in: it defines the common metrics in
both backends, keeps the two bit-comparable, and registers the numpy versions into whisper's own
registry so the CPU sampler gains them too. Importing :mod:`whisper_gpu` performs the registration.

    from whisper_labia.distance import list_distances
    list_distances()      # -> ['chi2', 'chi_square', 'mae', 'mse', 'rmse', 'wmae', 'wmse']

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
           "wmse_distance", "wmae_distance", "NUMPY_DISTANCES", "jnp_distance",
           "register_into_whisper"]


# --------------------------------------------------------------------------- numpy (CPU ABC)
# Signature is whisper's: distance(obs, obs_err, sim, bands=None) -> float. `bands` is accepted for
# API compatibility and ignored, exactly as chi2_distance does -- a flat sum over all points is
# equivalent to summing per band and adding.

def chi2_distance(obs, obs_err, sim, bands=None):
    """``sum(((obs - sim) / err)**2)`` -- identical to whisper's built-in, re-exported for symmetry."""
    r = (np.asarray(obs, float) - np.asarray(sim, float)) / np.asarray(obs_err, float)
    return float(np.sum(r * r))


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


def register_into_whisper(overwrite=True):
    """Add these metrics to whisper's own distance registry, so the CPU ABC can use them too."""
    from whisper_labia.distance import list_distances, register_distance

    known = set(list_distances())
    for name, fn in NUMPY_DISTANCES.items():
        if name in known and not overwrite:
            continue
        register_distance(name, fn, overwrite=True)
    return sorted(NUMPY_DISTANCES)


# --------------------------------------------------------------------------- jnp (GPU ABC)
def jnp_distance(name):
    """Return ``f(obs, err, sim) -> scalar`` in jnp, for use inside a vmapped simulate step.

    Deliberately mirrors the numpy versions term for term, including what they do NOT do: no
    masking of non-finite values, no clipping of the error. whisper's ABC lets a NaN propagate into
    the distance (and thence into the acceptance quantile) rather than hiding it, and the GPU port
    matches that so the two produce the same diagnosis on bad input.
    """
    import jax.numpy as jnp

    def _chi2(o, e, s):
        r = (o - s) / e
        return jnp.sum(r * r)

    def _mse(o, e, s):
        r = o - s
        return jnp.mean(r * r)

    def _rmse(o, e, s):
        r = o - s
        return jnp.sqrt(jnp.mean(r * r))

    def _mae(o, e, s):
        return jnp.mean(jnp.abs(o - s))

    def _wmse(o, e, s):
        r = (o - s) / e
        return jnp.mean(r * r)

    def _wmae(o, e, s):
        return jnp.mean(jnp.abs((o - s) / e))

    table = {"chi2": _chi2, "chi_square": _chi2, "mse": _mse, "rmse": _rmse,
             "mae": _mae, "wmse": _wmse, "wmae": _wmae}
    key = str(name).lower()
    if key not in table:
        raise KeyError(f"abc_gpu has no jnp distance {name!r}. Available: {sorted(table)}")
    return table[key]
