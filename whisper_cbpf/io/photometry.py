"""Magnitude <-> flux-density conversions (AB system).

Flux densities are in janskys (Jy); the AB zeropoint is 3631 Jy, i.e.
``m_AB = -2.5 * log10(f_nu / 3631 Jy)``. (An optional physical-model backend such as redback emits
``flux_density`` in mJy, 1 Jy = 1e3 mJy; the Phase-2 forward model reconciles units.) For ingestion we
only need internally consistent, invertible conversions with correct error propagation.
"""
from __future__ import annotations

import warnings

import numpy as np

AB_ZEROPOINT_JY = 3631.0
_LN10 = np.log(10.0)
POGSON = 2.5 / _LN10  # ~1.0857; magnitude error <-> SNR via sigma_m = POGSON / SNR


def mag_to_flux_density(magnitude, magnitude_err=None, zeropoint_jy=AB_ZEROPOINT_JY):
    """Convert AB magnitude to flux density (Jy). Returns ``flux`` or ``(flux, flux_err)``."""
    magnitude = np.asarray(magnitude, dtype=float)
    flux = zeropoint_jy * np.power(10.0, -0.4 * magnitude)
    if magnitude_err is None:
        return flux
    magnitude_err = np.asarray(magnitude_err, dtype=float)
    flux_err = 0.4 * _LN10 * flux * magnitude_err
    return flux, flux_err


def flux_density_to_mag(flux, flux_err=None, zeropoint_jy=AB_ZEROPOINT_JY):
    """Convert flux density (Jy) to AB magnitude. Returns ``mag`` or ``(mag, mag_err)``.

    Non-positive flux densities have no AB magnitude (``log10`` of <= 0): they map to ``NaN`` (with a
    warning) rather than a silent NaN/negative-error pair. Keep such data in flux space instead.
    """
    flux = np.asarray(flux, dtype=float)
    bad = ~(flux > 0)
    if np.any(bad):
        warnings.warn(f"flux_density_to_mag: {int(np.sum(bad))} non-positive flux value(s) have no AB "
                      "magnitude -> NaN (keep such data in flux space).", stacklevel=2)
    safe = np.where(bad, np.nan, flux)
    with np.errstate(invalid="ignore", divide="ignore"):
        magnitude = -2.5 * np.log10(safe / zeropoint_jy)
        if flux_err is None:
            return magnitude
        magnitude_err = (2.5 / _LN10) * (np.asarray(flux_err, dtype=float) / safe)
    return magnitude, magnitude_err


def mag_err_to_snr(magnitude_err):
    """Per-point SNR from an AB magnitude error: ``SNR = (2.5 / ln 10) / sigma_m`` (Pogson)."""
    magnitude_err = np.asarray(magnitude_err, dtype=float)
    return POGSON / magnitude_err
