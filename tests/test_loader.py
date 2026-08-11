import numpy as np
import pytest

from whisper_labia import LightCurve, load_lightcurve
from whisper_labia.io.bands import unmapped_bands


def test_load_at2017gfo(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv)
    assert isinstance(lc, LightCurve)
    assert lc.name == "at2017gfo"
    assert lc.data_mode == "magnitude"
    assert lc.n_points > 400
    assert "g" in lc.bands and "Ks" in lc.bands
    assert 57982.0 <= lc.time.min() and lc.time.max() <= 58008.0
    assert lc.magnitude_err is not None


def test_time_window(at2017gfo_csv):
    full = load_lightcurve(at2017gfo_csv)
    lc = load_lightcurve(at2017gfo_csv, time_max=57985.0)
    assert lc.time.max() <= 57985.0
    assert lc.n_points < full.n_points


def test_band_subset(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv, bands=["g", "r"])
    assert set(lc.bands) <= {"g", "r"} and lc.n_points > 0


def test_system_recorded(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv)
    systems = set(lc.system.tolist())
    assert {"AB", "Vega", "unknown"} <= systems   # incl. blank system cells


def test_add_flux(at2017gfo_csv):
    lc = load_lightcurve(at2017gfo_csv).add_flux()
    assert lc.flux is not None and np.all(np.isfinite(lc.flux))


def test_min_snr_cut(at2017gfo_csv):
    full = load_lightcurve(at2017gfo_csv)
    cut = load_lightcurve(at2017gfo_csv, min_snr=5.0)
    assert cut.n_points < full.n_points
    assert np.all(cut.snr >= 5.0)


def test_explosion_date(at2017gfo_csv):
    full = load_lightcurve(at2017gfo_csv)
    lc = load_lightcurve(at2017gfo_csv, explosion_date=57982.0)
    assert lc.meta["explosion_mjd"] == 57982.0
    assert np.isclose(lc.time.min(), full.time.min() - 57982.0)


def test_upper_limit_kept_through_quality_cut(tmp_path):
    p = tmp_path / "ul.csv"
    p.write_text(
        "time,magnitude,e_magnitude,band,upper_limit\n"
        "1.0,20.0,0.1,g,False\n"
        "2.0,22.5,nan,g,True\n"      # upper limit, no error -> kept despite quality_cuts
    )
    lc = load_lightcurve(p)
    assert lc.n_points == 2 and lc.upper_limit.sum() == 1


def test_band_grouping_at2017gfo(at2017gfo_csv):
    full = load_lightcurve(at2017gfo_csv)
    grouped = load_lightcurve(at2017gfo_csv, band_lookup=True)
    assert {"g-band", "r-band", "i-band", "z-band", "K-band"} <= set(grouped.bands)
    assert not ({"V", "Ks", "B", "y"} & set(grouped.bands))   # collapsed away
    assert len(grouped.bands) < len(full.bands)
    # HST/J1 now mapped; only clear/white-light bands remain ungrouped.
    assert "F336W" not in grouped.bands and "J1" not in grouped.bands
    assert set(unmapped_bands(full.band)) == {"C", "W", "w"}
    assert "C" in grouped.bands   # passthrough, not dropped


def test_missing_time_column_raises(tmp_path):
    p = tmp_path / "no_time.csv"
    p.write_text("mag,band\n18.0,g\n19.0,r\n")
    with pytest.raises(ValueError, match="time"):
        load_lightcurve(p)


def test_quality_cuts(tmp_path):
    p = tmp_path / "dirty.csv"
    p.write_text(
        "time,magnitude,e_magnitude,band\n"
        "1.0,18.0,0.1,g\n"
        "2.0,nan,0.1,g\n"      # NaN magnitude -> dropped
        "3.0,19.0,-1.0,g\n"    # non-positive error -> dropped
        "4.0,19.5,0.2,r\n"
    )
    lc = load_lightcurve(p)
    assert lc.n_points == 2
    assert np.allclose(np.sort(lc.time), [1.0, 4.0])


def test_default_band(tmp_path):
    p = tmp_path / "noband.csv"
    p.write_text("mjd,mag,magerr\n1.0,18.0,0.1\n2.0,18.5,0.1\n")
    lc = load_lightcurve(p, default_band="ztfg")
    assert lc.bands == ["ztfg"] and lc.n_points == 2
