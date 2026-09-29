"""ABC distance metrics — numpy backend (the default).

Assembled from two sources (both discontinued; superseded by whisper_cbpf), function bodies
unchanged:

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
* **max_abs_z** is the opposite choice: the WORST point decides. A draw is close only if every
  point is close, in units of its own error, so a model that fits the peak and misses one late epoch
  by 6 sigma is rejected at ``threshold=5`` however good its chi-square is. The rule reads
  "every point within 5 sigma".

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
           "wmse_distance", "wmae_distance", "max_abs_z_distance", "NUMPY_DISTANCES"]


def chi2_distance(obs_flux, obs_flux_err, sim_flux, bands=None):
    """Multi-band chi-square: ``sum(((obs - sim) / err)**2)``.

    Summing over all points is equivalent to summing per band and adding. Numerically this equals
    ``-2 ln L`` for an independent Gaussian likelihood, up to its normalising constant, which is
    what lets ABC report AIC/BIC.

    Parameters
    ----------
    obs_flux, obs_flux_err, sim_flux : array_like
        Observed values, their errors, and simulated values, in the same space.
    bands : array_like, optional
        Accepted for the distance signature; not used.

    Returns
    -------
    float

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.chi2_distance([1.0, 2.0], [0.5, 0.5], [1.5, 2.0])
    1.0
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


def max_abs_z_distance(obs, obs_err, sim, bands=None):
    """Worst-point distance: ``max_i |obs_i - sim_i| / err_i``, in units of each point's error.

    The value is the deviation of the worst-fitting point in sigma, so an ABC ``threshold`` of
    ``k`` accepts a draw only if **every** point lies within ``k`` of its own error bar. Registered
    as ``"max_abs_z"`` (``"max_abs_z_jax"`` for the jnp twin that ``abc_gpu`` uses).

    Parameters
    ----------
    obs, obs_err : array_like
        Observed values and their 1-sigma errors, in the comparison space the sampler chose
        (magnitudes or flux). The errors must be positive; the samplers refuse data where they
        are not.
    sim : array_like
        Simulated values at the same points, in the same space.
    bands : array_like, optional
        Accepted for the common distance signature and ignored: the maximum is taken over every
        point of every band.

    Returns
    -------
    float
        ``max |z|``. A point whose simulated value is not finite (NaN or inf) counts as infinitely
        far, so such a draw is never accepted at a finite threshold and cannot poison a quantile;
        this is the only distance that maps NaN to ``inf``, because "every point within k sigma"
        is simply false for a point with no value. An empty input returns ``inf``.

    Notes
    -----
    **Threshold semantics.** ``fit_ABC(..., distance="max_abs_z", threshold=k)`` and ``abc_gpu``
    accept ``d <= k`` (inclusive); ``abc_smc`` / ``abc_smc_gpu`` accept ``d < epsilon`` (strict).
    With ``quantile=q`` instead, the ``q`` fraction of draws with the smallest worst point is kept.
    ``min_distance`` on the result is the closest draw's worst point in sigma, which is how a fit
    that accepted nothing reports how far it got (for example "closest 5.81 sigma").

    **The v3 rule** is ``distance="max_abs_z", threshold=5, simulate_noise=False,
    space="magnitude"``. With ``simulate_noise=True`` (the samplers' default) each simulation
    carries its own noise draw, so at the true parameters a residual has standard deviation
    ``sqrt(2)`` errors and ``threshold=5`` is a looser cut than the v3 rule in those units.

    **Not a likelihood.** Unlike ``chi2`` it is not ``-2 ln L``; the samplers compute AIC/BIC from
    the exact Gaussian likelihood whatever the distance. ABC-SMC's ``min_epsilon="auto"`` floor
    (``best + 2 (k + 2)``) is derived for ``chi2`` and is far too loose on this scale.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.distance import max_abs_z_distance
    >>> obs = np.array([20.0, 20.5, 21.0])
    >>> err = np.array([0.25, 0.25, 0.125])
    >>> max_abs_z_distance(obs, err, np.array([20.25, 20.5, 21.5]))  # the last point, 4 sigma
    4.0
    >>> max_abs_z_distance(obs, err, np.array([20.0, np.nan, 21.0]))
    inf

    The same rule by name, as the v3 fits used it::

        import whisper_cbpf as wp
        res = wp.fit_ABC(lc, "flare", distance="max_abs_z", threshold=5.0,
                         simulate_noise=False, space="magnitude")
    """
    z = np.abs(np.asarray(obs, float) - np.asarray(sim, float)) / np.asarray(obs_err, float)
    if z.size == 0:
        return float("inf")
    return float(np.max(np.where(np.isnan(z), np.inf, z)))


NUMPY_DISTANCES = {
    "chi2": chi2_distance, "mse": mse_distance, "rmse": rmse_distance,
    "mae": mae_distance, "wmse": wmse_distance, "wmae": wmae_distance,
    "max_abs_z": max_abs_z_distance,
}
