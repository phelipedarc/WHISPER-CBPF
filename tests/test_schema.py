import numpy as np
import pytest

from whisper_labia.io.photometry import flux_density_to_mag
from whisper_labia.io.schema import LightCurve


def _toy():
    return LightCurve(
        time=np.array([1.0, 2.0, 3.0]),
        band=np.array(["g", "r", "g"]),
        magnitude=np.array([18.0, 18.5, 19.0]),
        magnitude_err=np.array([0.1, 0.1, 0.2]),
        name="toy",
    )


def test_basic():
    lc = _toy()
    assert lc.n_points == 3 and len(lc) == 3
    assert lc.bands == ["g", "r"]
    assert lc.data_mode == "magnitude"


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        LightCurve(time=np.array([1.0, 2.0]), band=np.array(["g"]),
                   magnitude=np.array([18.0, 19.0]))


def test_requires_mag_or_flux():
    with pytest.raises(ValueError):
        LightCurve(time=np.array([1.0]), band=np.array(["g"]))


def test_select_time_window():
    lc = _toy().select_time_window(time_min=1.5)
    assert lc.n_points == 2 and lc.time.min() >= 1.5


def test_select_bands():
    lc = _toy().select_bands("r")
    assert lc.bands == ["r"] and lc.n_points == 1


def test_add_flux():
    lc = _toy().add_flux()
    assert lc.flux is not None and lc.flux_err is not None
    assert np.allclose(flux_density_to_mag(lc.flux), _toy().magnitude)


def test_add_mag():
    base = _toy().add_flux()
    flux_only = LightCurve(time=base.time, band=base.band, flux=base.flux, flux_err=base.flux_err)
    out = flux_only.add_mag()
    assert out.magnitude is not None
    assert np.allclose(out.magnitude, _toy().magnitude)


def test_snr_from_magnitude():
    lc = _toy()
    assert np.allclose(lc.snr, (2.5 / np.log(10)) / np.array([0.1, 0.1, 0.2]))


def test_select_snr():
    # snr ~ [10.86, 10.86, 5.43]; min_snr=8 keeps the two well-measured points.
    cut = _toy().select_snr(min_snr=8.0)
    assert cut.n_points == 2 and np.all(cut.snr >= 8.0)


def test_set_explosion_date():
    lc = _toy().set_explosion_date(2.0)   # times [1, 2, 3] -> [-1, 0, 1]
    assert lc.meta["explosion_mjd"] == 2.0
    assert np.allclose(lc.time, [-1.0, 0.0, 1.0])


def test_upper_limit_field():
    lc = LightCurve(time=[1.0, 2.0], band=["g", "g"],
                    magnitude=[20.0, 21.0], magnitude_err=[0.1, 0.1],
                    upper_limit=[False, True])
    assert lc.upper_limit.dtype == bool and lc.upper_limit.tolist() == [False, True]
    assert lc.select_bands("g").upper_limit.tolist() == [False, True]
