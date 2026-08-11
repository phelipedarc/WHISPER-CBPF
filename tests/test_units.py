"""Astropy-unit validation + conversion for the three data modes (whisper_labia.io.units)."""
import numpy as np
import pytest
import astropy.units as u

from whisper_labia.io.units import (
    check_magnitude_unit,
    to_canonical,
    to_flux_cgs,
    to_flux_density_jy,
)


def test_fnu_mjy_to_jy():
    assert np.allclose(to_flux_density_jy([1.0, 2.0], "mJy"), [1e-3, 2e-3])
    assert np.allclose(to_flux_density_jy([3631.0], "Jy"), [3631.0])


def test_flux_density_roundtrip_mjy_vs_flambda():
    """The same physical flux as F_nu (mJy) and as F_lambda (erg/s/cm^2/AA) must give equal Jy."""
    lam = 6000.0
    fnu_jy = to_flux_density_jy([1.0], "mJy")                      # 1 mJy -> 1e-3 Jy
    flam_val = (1.0 * u.mJy).to_value(
        u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(lam * u.AA))
    flam_jy = to_flux_density_jy([flam_val], "erg/(s cm2 AA)", lambda_eff=[lam])
    assert np.allclose(fnu_jy, flam_jy)


def test_flambda_requires_lambda_eff():
    with pytest.raises(ValueError, match="lambda_eff"):
        to_flux_density_jy([1e-15], "erg/(s cm2 AA)", lambda_eff=None)


def test_unrecognised_flux_density_unit_errors():
    with pytest.raises(ValueError, match="flux-density"):
        to_flux_density_jy([1.0], "erg/(s cm2)")   # band-integrated, not a density


def test_magnitude_rejects_flux_unit():
    with pytest.raises(ValueError, match="magnitude"):
        check_magnitude_unit("Jy")
    with pytest.raises(ValueError, match="magnitude"):
        to_canonical([20.0], "Jy", "magnitude")


def test_magnitude_accepts_dimensionless_and_mag():
    check_magnitude_unit("")        # no unit
    check_magnitude_unit("mag")     # explicit magnitude
    assert np.allclose(to_canonical([20.0, 21.0], "mag", "magnitude"), [20.0, 21.0])


def test_flux_band_integrated_dimensionality():
    assert np.allclose(to_flux_cgs([1.0], "erg/(s cm2)"), [1.0])
    with pytest.raises(ValueError, match="band-integrated"):
        to_flux_cgs([1.0], "Jy")


def test_no_unit_metadata_warns_and_defaults():
    with pytest.warns(UserWarning, match="no unit metadata"):
        out = to_canonical([1.0], None, "flux_density")
    assert np.allclose(out, [1.0])   # default Jy -> unchanged


def test_unknown_data_mode_errors():
    with pytest.raises(ValueError, match="data_mode"):
        to_canonical([1.0], "Jy", "luminosity")


def test_flambda_uses_per_point_wavelength():
    """Same F_lambda value at different lambda_eff must give different Jy (Jy ∝ lambda^2)."""
    out = to_flux_density_jy([1e-15, 1e-15], "erg/(s cm2 AA)", lambda_eff=[4000.0, 8000.0])
    assert not np.isclose(out[0], out[1])
    assert np.isclose(out[1] / out[0], (8000.0 / 4000.0) ** 2, rtol=1e-6)


def test_flambda_nan_wavelength_with_finite_value_errors():
    """An unresolved band (NaN lambda_eff) with a real flux must error, not silently NaN."""
    with pytest.raises(ValueError, match="unresolved band"):
        to_flux_density_jy([1e-15, 2e-15], "erg/(s cm2 AA)", lambda_eff=[6000.0, np.nan])


def test_to_canonical_flux_band_integrated_dispatch():
    assert np.allclose(to_canonical([1.0], "erg/(s cm2)", "flux"), [1.0])
    with pytest.raises(ValueError, match="band-integrated"):
        to_canonical([1.0], "Jy", "flux")
