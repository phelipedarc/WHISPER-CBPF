import numpy as np

from whisper_labia.io.photometry import (
    AB_ZEROPOINT_JY,
    flux_density_to_mag,
    mag_err_to_snr,
    mag_to_flux_density,
)


def test_ab_zeropoint():
    assert np.isclose(mag_to_flux_density(0.0), AB_ZEROPOINT_JY)


def test_mag_flux_roundtrip():
    mags = np.array([16.0, 18.5, 20.0, 22.3])
    back = flux_density_to_mag(mag_to_flux_density(mags))
    assert np.allclose(back, mags)


def test_error_roundtrip():
    mags = np.array([18.0, 19.0, 20.0])
    errs = np.array([0.02, 0.1, 0.3])
    flux, flux_err = mag_to_flux_density(mags, errs)
    _, back_err = flux_density_to_mag(flux, flux_err)
    assert np.allclose(back_err, errs)


def test_brighter_is_more_flux():
    assert mag_to_flux_density(18.0) > mag_to_flux_density(20.0)


def test_mag_err_to_snr_matches_flux_snr():
    # SNR from a magnitude error equals flux/flux_err under linear error propagation.
    mag, err = 19.0, 0.05
    flux, flux_err = mag_to_flux_density(mag, err)
    assert np.isclose(flux / flux_err, mag_err_to_snr(err))
    assert np.isclose(mag_err_to_snr(0.1), (2.5 / np.log(10)) / 0.1)
