"""Redshift handling, data_mode, the __call__ dataframe, and loader unit/band integration."""
import warnings

import numpy as np
import pytest
import astropy.units as u

from whisper_labia import LightCurve, load_lightcurve
from whisper_labia.io import svo


# --------------------------------------------------------------------------- data_mode
def test_data_mode_inferred():
    lc_f = LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[0.1])
    assert lc_f.data_mode == "flux_density" and lc_f.output_format == "flux_density"
    lc_m = LightCurve(time=[1.0], band=["r"], magnitude=[20.0], magnitude_err=[0.1])
    assert lc_m.data_mode == "magnitude" and lc_m.output_format == "magnitude"


def test_data_mode_explicit_flux_maps_to_flux_density_output():
    lc = LightCurve(time=[1.0], band=["r"], flux=[1e-12], flux_err=[1e-13], data_mode="flux")
    assert lc.data_mode == "flux" and lc.output_format == "flux_density"


def test_data_mode_invalid_raises():
    with pytest.raises(ValueError, match="data_mode"):
        LightCurve(time=[1.0], band=["r"], flux=[1.0], data_mode="luminosity")


def test_data_mode_survives_subset():
    lc = LightCurve(time=[1.0, 2.0], band=["r", "g"], flux=[1.0, 2.0], flux_err=[0.1, 0.1],
                    data_mode="flux_density")
    assert lc.select_bands("r").data_mode == "flux_density"


# --------------------------------------------------------------------------- lc() is the table
def test_call_returns_self_table():
    """lc() returns the LightCurve itself (an astropy Table) -- assign/compute columns directly."""
    lc = LightCurve(time=[1.0, 2.0], band=["r", "g"], magnitude=[20.0, 21.0],
                    magnitude_err=[0.1, 0.1])
    assert lc() is lc
    lc()["shifted"] = lc()["magnitude"] + 5          # griffin-style column assignment
    assert list(lc["shifted"]) == [25.0, 26.0]
    assert "shifted" in lc.colnames


def test_add_mag_fills_magnitude_column():
    lc = LightCurve(time=[1.0], band=["r"], flux=[1e-3], flux_err=[1e-4])
    out = lc.add_mag()
    assert "magnitude" in out.colnames and np.isfinite(out["magnitude"][0])


# --------------------------------------------------------------------------- redshift
def _mag_csv(tmp_path, extra_col="", extra_val=""):
    p = tmp_path / "lc.csv"
    p.write_text(
        f"time,magnitude,e_magnitude,band{extra_col}\n"
        f"1.0,20.0,0.1,g{extra_val}\n"
        f"2.0,20.5,0.1,r{extra_val}\n")
    return p


def test_redshift_from_argument(tmp_path):
    lc = load_lightcurve(_mag_csv(tmp_path), redshift=0.12)
    assert lc.redshift == 0.12 and lc.redshift_known


def test_redshift_from_column(tmp_path):
    lc = load_lightcurve(_mag_csv(tmp_path, ",redshift", ",0.34"))
    assert lc.redshift == 0.34 and lc.redshift_known


def test_redshift_argument_overrides_column(tmp_path):
    lc = load_lightcurve(_mag_csv(tmp_path, ",redshift", ",0.34"), redshift=0.99)
    assert lc.redshift == 0.99


def test_redshift_unknown_warns_and_sets_prior(tmp_path):
    with pytest.warns(UserWarning, match="redshift"):
        lc = load_lightcurve(_mag_csv(tmp_path))
    assert lc.redshift is None and not lc.redshift_known
    assert lc.redshift_prior is not None and lc.redshift_prior["type"] == "Uniform"


def test_redshift_zero_requires_distance():
    with pytest.raises(ValueError, match="luminosity_distance"):
        LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[0.1], redshift=0.0)
    lc = LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[0.1],
                    redshift=0.0, luminosity_distance=40.0)
    assert lc.redshift == 0.0 and lc.luminosity_distance == 40.0


def test_redshift_negative_and_nan_are_fatal():
    with pytest.raises(ValueError, match="redshift"):
        LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[0.1], redshift=-0.5)
    with pytest.raises(ValueError, match="finite"):
        LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[0.1], redshift=float("nan"))


# --------------------------------------------------------------------------- loader units + bands
def _flux_csv(tmp_path, flux_vals, band="r"):
    p = tmp_path / "flux.csv"
    rows = "\n".join(f"{i+1}.0,{v},{abs(v)*0.1},{band}" for i, v in enumerate(flux_vals))
    p.write_text("time,flux,flux_err,band\n" + rows + "\n")
    return p


def test_loader_no_unit_warns_and_assumes_jy(tmp_path):
    with pytest.warns(UserWarning, match="no unit metadata"):
        lc = load_lightcurve(_flux_csv(tmp_path, [1e-3, 2e-3]), redshift=0.1)
    assert np.allclose(lc.flux, [1e-3, 2e-3])   # assumed Jy, unchanged


def test_loader_flambda_converts_with_band_wavelength(tmp_path):
    """F_lambda flux + resolved band wavelength -> Jy (matches the F_nu equivalent)."""
    lam = 6215.0   # r-band LSST anchor
    flam = (1.0 * u.mJy).to_value(
        u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(lam * u.AA))
    lc = load_lightcurve(_flux_csv(tmp_path, [flam], band="r"), redshift=0.1,
                         flux_unit="erg/(s cm2 AA)")
    assert np.allclose(lc.flux, [1e-3], rtol=1e-3)   # 1 mJy
    assert np.isfinite(lc.lambda_eff[0])


def test_loader_flambda_without_wavelength_errors(tmp_path):
    with pytest.raises(ValueError, match="lambda_eff"):
        load_lightcurve(_flux_csv(tmp_path, [1e-15], band="r"), redshift=0.1,
                        flux_unit="erg/(s cm2 AA)", resolve_band_info=False)


def test_loader_magnitude_given_flux_unit_errors(tmp_path):
    with pytest.raises(ValueError, match="magnitude"):
        load_lightcurve(_mag_csv(tmp_path), redshift=0.1, magnitude_unit="Jy")


def test_loader_known_bands_skip_svo(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("SVO must not be queried for known bands")
    monkeypatch.setattr(svo, "_svo_fetch_metadata", boom)
    monkeypatch.setattr(svo, "_svo_fetch_index", boom)
    lc = load_lightcurve(_mag_csv(tmp_path), redshift=0.1)   # bands g, r
    assert lc.zero_point is not None and np.all(np.isfinite(lc.zero_point))
    assert lc.lambda_eff is not None and np.all(np.isfinite(lc.lambda_eff))


def test_loader_magnitude_no_unit_warns(tmp_path):
    with pytest.warns(UserWarning, match="no unit metadata"):
        load_lightcurve(_mag_csv(tmp_path), redshift=0.1)


def test_redshift_column_all_nan_degrades_to_unknown(tmp_path):
    p = tmp_path / "z.csv"
    p.write_text("time,magnitude,e_magnitude,band,redshift\n"
                 "1.0,20.0,0.1,g,nan\n2.0,20.5,0.1,r,\n")
    with pytest.warns(UserWarning, match="redshift"):
        lc = load_lightcurve(p)
    assert lc.redshift is None and not lc.redshift_known and lc.redshift_prior is not None


def test_redshift_column_negative_is_fatal(tmp_path):
    p = tmp_path / "z.csv"
    p.write_text("time,magnitude,e_magnitude,band,redshift\n"
                 "1.0,20.0,0.1,g,-0.5\n2.0,20.5,0.1,r,-0.5\n")
    with pytest.raises(ValueError, match="redshift"):
        load_lightcurve(p)


def test_subset_preserves_redshift_state_without_rewarning():
    lc = LightCurve(time=[1.0, 2.0], band=["r", "g"], flux=[1.0, 2.0], flux_err=[0.1, 0.1])
    assert not lc.redshift_known and lc.redshift_prior is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any warning fails the test
        sub = lc.select_bands("r")                # boolean-mask slicing (Table)
        cp = lc.copy()
    assert not sub.redshift_known and sub.redshift_prior == lc.redshift_prior
    assert not cp.redshift_known
    lz = LightCurve(time=[1.0, 2.0], band=["r", "g"], flux=[1.0, 2.0], flux_err=[0.1, 0.1],
                    redshift=0.0, luminosity_distance=40.0)
    s = lz.select_bands("r")
    assert s.redshift == 0.0 and s.luminosity_distance == 40.0


def test_add_flux_uses_constant_ab_zeropoint_with_per_band_opt_in():
    """add_flux() stays on AB 3631 (modelling); pass the per-band zero point explicitly to opt in."""
    lc = LightCurve(time=[1.0], band=["myJ"], magnitude=[20.0], magnitude_err=[0.1],
                    zero_point=[1594.0], lambda_eff=[12350.0])
    assert np.isclose(lc.add_flux()["flux"][0], 3631.0 * 10 ** (-0.4 * 20.0))               # AB constant
    assert np.isclose(lc.add_flux(zeropoint_jy=lc.zero_point)["flux"][0],
                      1594.0 * 10 ** (-0.4 * 20.0))                                          # per-band


def test_add_flux_mag_reject_band_integrated_flux():
    lc = LightCurve(time=[1.0], band=["r"], flux=[1e-12], flux_err=[1e-13], data_mode="flux")
    with pytest.raises(ValueError, match="band-integrated"):
        lc.add_mag()
    with pytest.raises(ValueError, match="band-integrated"):
        lc.add_flux()


def test_loader_flux_mode_band_integrated(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("time,flux,flux_err,band\n1.0,1e-12,1e-13,r\n2.0,2e-12,2e-13,r\n")
    lc = load_lightcurve(p, redshift=0.1, data_mode="flux", flux_unit="erg/(s cm2)")
    assert lc.data_mode == "flux" and lc.output_format == "flux_density"
    assert np.allclose(lc.flux, [1e-12, 2e-12])
    df = lc()
    assert "flux" in df.columns and "magnitude" not in df.columns
    with pytest.raises(ValueError, match="band-integrated"):
        load_lightcurve(p, redshift=0.1, data_mode="flux", flux_unit="Jy")


def test_loader_flambda_unresolved_band_errors(tmp_path):
    """Valid F_lambda data for an unresolvable band must error clearly, not silently empty out."""
    p = tmp_path / "f.csv"
    p.write_text("time,flux,flux_err,band\n1.0,1e-15,1e-16,WeirdBand\n")
    with pytest.raises(ValueError, match="unresolved band"):
        load_lightcurve(p, redshift=0.1, flux_unit="erg/(s cm2 AA)", svo_fallback=False)
