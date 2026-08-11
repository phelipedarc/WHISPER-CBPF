"""LightCurve as an astropy.table.Table: table semantics, where(), phase, absolute magnitude."""
import numpy as np
import pytest
from astropy.table import Table

from whisper_labia import LightCurve


def _lc(**kw):
    return LightCurve(time=[1.0, 2.0, 3.0, 4.0], band=["r", "g", "r", "g"],
                      magnitude=[20.0, 20.5, 21.0, 21.5], magnitude_err=[0.1, 0.1, 0.2, 0.2], **kw)


def test_lightcurve_is_a_table():
    lc = _lc(redshift=0.1, name="t")
    assert isinstance(lc, Table) and lc() is lc
    # griffin-style column ops
    lc["m+5"] = lc["magnitude"] + 5
    assert list(lc["m+5"]) == [25.0, 25.5, 26.0, 26.5]
    # native Table methods work
    lc.sort("magnitude")
    assert lc["magnitude"][0] == 20.0
    assert len(lc.group_by("band").groups) == 2


def test_properties_and_setters():
    lc = _lc()
    assert np.allclose(lc.time, [1, 2, 3, 4]) and lc.flux is None and lc.magnitude is not None
    assert lc.n_points == 4 and lc.bands == ["g", "r"] and lc.data_mode == "magnitude"
    lc.flux = [1.0, 2.0, 3.0, 4.0]                 # property setter -> adds the column
    assert "flux" in lc.colnames and np.allclose(lc["flux"], [1, 2, 3, 4])


def test_slicing_keeps_subclass_and_meta():
    lc = _lc(redshift=0.2, name="sn")
    sub = lc[lc["magnitude"] < 21.0]
    assert isinstance(sub, LightCurve) and sub.n_points == 2
    assert sub.redshift == 0.2 and sub.name == "sn" and sub.data_mode == "magnitude"


def test_where():
    lc = _lc()
    assert lc.where(band="r").n_points == 2
    assert lc.where(band="r", time_min=2.0).n_points == 1
    assert lc.where(time_min=2.0, time_max=3.0).n_points == 2
    assert lc.where(band_not="r").n_points == 2
    assert lc.where(band=["r", "g"]).n_points == 4


def test_calc_phase_restframe():
    lc = _lc(redshift=0.5)
    ph = lc.set_explosion_date(0.5).calc_phase()           # time already explosion-relative; z=0.5 -> /1.5
    assert np.allclose(ph["phase"], (np.array([1, 2, 3, 4]) - 0.5) / 1.5)
    assert ph.meta["refmjd"] == 0.0 and ph.meta["redshift_for_phase"] == 0.5
    # peak-relative (brightest = smallest magnitude, here the first point)
    pk = lc.calc_phase(peak=True)
    assert pk.meta["peakdate"] == 1.0 and np.isclose(pk["phase"][0], 0.0)


def test_calc_absmag_distance_and_extinction():
    lc = _lc(redshift=0.05)
    # distance modulus from redshift; no extinction
    ab = lc.calc_absmag()
    assert "absmag" in ab.colnames and ab.meta["dm"] > 30
    assert np.allclose(ab["absmag"], lc["magnitude"].data - ab.meta["dm"])
    # explicit per-band extinction dict
    ab2 = lc.calc_absmag(dm=36.0, extinction={"r": 0.3, "g": 0.5})
    expected = lc["magnitude"].data - 36.0 - np.array([0.3, 0.5, 0.3, 0.5])
    assert np.allclose(ab2["absmag"], expected)
    # MW extinction from E(B-V) via CCM89 (needs resolved wavelengths); A_band > 0
    ab3 = lc.resolve_bands(svo_fallback=False).calc_absmag(dm=36.0, ebv=0.2)
    a_band = (lc["magnitude"].data - 36.0) - ab3["absmag"].data
    assert np.all(a_band > 0) and ab3.meta["ebv"] == 0.2


def test_luminosity_distance_dm():
    lc = LightCurve(time=[1.0], band=["r"], magnitude=[20.0], magnitude_err=[0.1],
                    redshift=0.0, luminosity_distance=40.0)               # z=0 with LD
    ab = lc.calc_absmag()
    assert np.isclose(ab.meta["dm"], 5 * np.log10(40.0 * 1e6) - 5)
