"""Survey ingest presets: ``load_lightcurve(x, survey="ztf" | "lsst")``.

Two groups. The unit tests build alert records in the brokers' own field names (ALeRCE, the ZTF
alert packet, Fink's prefixed columns, the Rubin alert packet) and check the photometry rules:
difference-imaging PSF photometry only (``magpsf`` / ``psfFlux``, never ``magpsf_corr`` /
``scienceFlux``), negative subtractions dropped, 5-sigma limits from ``diffmaglim`` and the forced
photometry, survey-prefixed band labels.

The gate tests rebuild a reference set of cut light curves (five ZTF and LSST transients) row for
row from their raw alert data. They need that data (``$WHISPER_REALWORLD_DATA``) and are
skipped without it.
"""
import doctest
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.io import SURVEY_BANDS, surveys

POGSON = 2.5 / np.log(10.0)


def _load(x, **kw):
    kw.setdefault("redshift", 0.05)
    return wp.load_lightcurve(x, **kw)


# ----------------------------------------------------------------------------------------- ZTF
def _alerce():
    """An ALeRCE ``/lightcurve`` response: detections and non_detections."""
    det = [
        # magpsf_corr is a decoy: the preset must never read it
        {"mjd": 60001.0, "fid": 1, "magpsf": 19.20, "sigmapsf": 0.08, "isdiffpos": 1,
         "magpsf_corr": 16.0, "sigmapsf_corr": 0.01, "diffmaglim": 20.6, "candid": 11},
        {"mjd": 60000.5, "fid": 2, "magpsf": 19.50, "sigmapsf": 0.10, "isdiffpos": 1,
         "magpsf_corr": 16.1, "sigmapsf_corr": 0.01, "diffmaglim": 20.7, "candid": 12},
        {"mjd": 60003.0, "fid": 3, "magpsf": 18.90, "sigmapsf": 0.12, "isdiffpos": 1,
         "magpsf_corr": 16.2, "sigmapsf_corr": 0.01, "diffmaglim": 20.1, "candid": 13},
        # a negative subtraction: magpsf is |flux difference|, it has no AB magnitude
        {"mjd": 60004.0, "fid": 1, "magpsf": 20.00, "sigmapsf": 0.20, "isdiffpos": -1,
         "magpsf_corr": 16.3, "sigmapsf_corr": 0.01, "diffmaglim": 20.5, "candid": 14},
    ]
    nondet = [
        {"mjd": 59998.0, "fid": 1, "diffmaglim": 20.9},
        {"mjd": 59999.0, "fid": 2, "diffmaglim": 20.8},
        # the same exposure as a detection: not a non-detection
        {"mjd": 60001.0, "fid": 1, "diffmaglim": 20.6},
    ]
    return {"detections": det, "non_detections": nondet}


def test_ztf_alerce_response_uses_magpsf_and_diffmaglim():
    with pytest.warns(UserWarning, match="negative"):
        lc = _load(_alerce(), survey="ztf")
    assert lc.data_mode == "magnitude"
    assert list(lc.time) == [59998.0, 59999.0, 60000.5, 60001.0, 60003.0]        # sorted
    assert list(lc.band) == ["ztfg", "ztfr", "ztfr", "ztfg", "ztfi"]
    det = ~lc.upper_limit
    assert list(lc.magnitude[det]) == [19.50, 19.20, 18.90]                     # magpsf, not _corr
    assert list(lc.magnitude_err[det]) == [0.10, 0.08, 0.12]
    assert list(lc.magnitude[~det]) == [20.9, 20.8]                             # diffmaglim
    assert np.all(np.isnan(lc.magnitude_err[~det]))
    assert lc.meta["survey"] == "ztf" and lc.meta["upper_limit_sigma"] == 5.0
    assert lc.meta["first_detection_mjd"] == 60000.5
    dropped = lc.meta["n_dropped"]
    assert dropped["negative_flux"] == 1 and dropped["limit_at_detection"] == 1


def test_ztf_alert_packets_jd_and_dedupe():
    """Two ZTF alert packets of one object: candidate + prv_candidates, jd clock, 't'/'f' signs."""
    def cand(jd, fid, mag, err, pos="t", candid=None):
        return {"jd": jd, "fid": fid, "magpsf": mag, "sigmapsf": err, "isdiffpos": pos,
                "diffmaglim": 20.5, "candid": candid}

    nd = {"jd": 2460000.0, "fid": 2, "magpsf": None, "sigmapsf": None, "isdiffpos": None,
          "diffmaglim": 20.7, "candid": None}
    p1 = {"objectId": "ZTF26aaaaaaa", "candidate": cand(2460001.5, 1, 19.0, 0.1, "t", 1),
          "prv_candidates": [nd]}
    p2 = {"objectId": "ZTF26aaaaaaa", "candidate": cand(2460002.5, 2, 18.8, 0.1, "1", 2),
          "prv_candidates": [nd, cand(2460001.5, 1, 19.0, 0.1, "t", 1),
                             cand(2460002.0, 1, 19.9, 0.2, "f", 3)]}
    with pytest.warns(UserWarning, match="negative"):
        lc = _load([p1, p2], survey="ztf")
    assert lc.name == "ZTF26aaaaaaa"
    assert np.allclose(lc.time, [59999.5, 60001.0, 60002.0])                  # jd - 2400000.5
    assert list(lc.upper_limit) == [True, False, False]
    assert list(lc.band) == ["ztfr", "ztfg", "ztfr"]
    assert lc.meta["n_dropped"]["duplicate"] == 2                            # nd and cand 1 twice


def test_fink_ztf_prefixed_columns_and_tags():
    df = pd.DataFrame({
        "i:objectId": ["ZTF26b"] * 4,
        "i:jd": [2460000.5, 2460001.5, 2460002.5, 2460003.5],
        "i:fid": [1, 2, 1, 2],
        "i:magpsf": [np.nan, 19.0, 18.7, 18.5],
        "i:sigmapsf": [np.nan, 0.1, 0.1, 0.1],
        "i:isdiffpos": [None, "t", "t", "t"],
        "i:diffmaglim": [20.4, 20.5, 20.5, 20.5],
        "d:tag": ["upperlim", "valid", "badquality", "valid"],
    })
    lc = _load(df, survey="ztf")
    assert lc.name == "ZTF26b"
    assert np.allclose(lc.time, [60000.0, 60001.0, 60003.0])
    assert list(lc.upper_limit) == [True, False, False]
    assert lc.meta["n_dropped"]["bad_quality"] == 1


def test_ztf_refuses_magpsf_corr():
    recs = [{"mjd": 60000.0, "fid": 1, "magpsf_corr": 17.0, "sigmapsf_corr": 0.01}]
    with pytest.raises(ValueError, match="magpsf_corr"):
        _load(recs, survey="ztf")


def test_ztf_unknown_fid_raises():
    recs = [{"mjd": 60000.0, "fid": 4, "magpsf": 19.0, "sigmapsf": 0.1, "isdiffpos": "t"}]
    with pytest.raises(ValueError, match="fid"):
        _load(recs, survey="ztf")


def test_ztf_without_isdiffpos_warns():
    recs = [{"mjd": 60000.0, "fid": 1, "magpsf": 19.0, "sigmapsf": 0.1}]
    with pytest.warns(UserWarning, match="isdiffpos"):
        lc = _load(recs, survey="ztf")
    assert lc.n_points == 1


# ---------------------------------------------------------------------------------------- LSST
def _fink_sources():
    return [
        {"r:diaSourceId": 1, "r:midpointMjdTai": 61001.0, "r:band": "r", "r:visit": 101,
         "r:psfFlux": 2000.0, "r:psfFluxErr": 200.0, "r:isNegative": False, "r:psfFlux_flag": False,
         "r:scienceFlux": 90000.0, "r:scienceFluxErr": 300.0},
        {"r:diaSourceId": 2, "r:midpointMjdTai": 61000.0, "r:band": "g", "r:visit": 100,
         "r:psfFlux": 1000.0, "r:psfFluxErr": 150.0, "r:isNegative": False, "r:psfFlux_flag": False,
         "r:scienceFlux": 80000.0, "r:scienceFluxErr": 300.0},
        {"r:diaSourceId": 3, "r:midpointMjdTai": 61002.0, "r:band": "i", "r:visit": 102,
         "r:psfFlux": -900.0, "r:psfFluxErr": 150.0, "r:isNegative": True, "r:psfFlux_flag": False,
         "r:scienceFlux": 70000.0, "r:scienceFluxErr": 300.0},
        {"r:diaSourceId": 4, "r:midpointMjdTai": 61003.0, "r:band": "z", "r:visit": 103,
         "r:psfFlux": 3000.0, "r:psfFluxErr": 250.0, "r:isNegative": False, "r:psfFlux_flag": True,
         "r:scienceFlux": 70000.0, "r:scienceFluxErr": 300.0},
    ]


def _fink_forced():
    return [
        # same visit as detection 1: not a non-detection
        {"r:diaForcedSourceId": 11, "r:midpointMjdTai": 61001.0, "r:band": "r", "r:visit": 101,
         "r:psfFlux": 1990.0, "r:psfFluxErr": 210.0},
        {"r:diaForcedSourceId": 12, "r:midpointMjdTai": 61004.0, "r:band": "u", "r:visit": 104,
         "r:psfFlux": 30.0, "r:psfFluxErr": 120.0},
        {"r:diaForcedSourceId": 13, "r:midpointMjdTai": 60999.0, "r:band": "y", "r:visit": 99,
         "r:psfFlux": -10.0, "r:psfFluxErr": 400.0},
        {"r:diaForcedSourceId": 14, "r:midpointMjdTai": 61005.0, "r:band": "g", "r:visit": 105,
         "r:psfFlux": 5.0, "r:psfFluxErr": 100.0, "r:timeWithdrawnMjdTai": 61006.0},
    ]


def test_lsst_psfflux_to_ab_and_forced_limits():
    with pytest.warns(UserWarning, match="negative"):
        lc = _load(_fink_sources(), survey="lsst", limits=_fink_forced())
    assert list(lc.time) == [60999.0, 61000.0, 61001.0, 61004.0]
    assert list(lc.band) == ["lssty", "lsstg", "lsstr", "lsstu"]
    assert list(lc.upper_limit) == [True, False, False, True]
    det = ~lc.upper_limit
    f, e = np.array([1000.0, 2000.0]), np.array([150.0, 200.0])
    np.testing.assert_allclose(lc.magnitude[det], 31.4 - 2.5 * np.log10(f), rtol=0, atol=1e-12)
    np.testing.assert_allclose(lc.magnitude_err[det], POGSON * e / f, rtol=1e-14)
    lim = 31.4 - 2.5 * np.log10(5.0 * np.array([400.0, 120.0]))
    np.testing.assert_allclose(lc.magnitude[~det], lim, rtol=0, atol=1e-12)
    d = lc.meta["n_dropped"]
    assert d["negative_flux"] == 1 and d["flagged"] == 1
    assert d["withdrawn"] == 1 and d["limit_at_detection"] == 1
    assert lc.meta["survey"] == "lsst" and lc.meta["first_detection_mjd"] == 61000.0


def test_lsst_rubin_alert_packets():
    src = [{k[2:]: v for k, v in r.items()} for r in _fink_sources()[:2]]
    fp = [{k[2:]: v for k, v in r.items()} for r in _fink_forced()[1:3]]
    p1 = {"diaObject": {"diaObjectId": 170230797176406143}, "diaSource": src[1],
          "prvDiaSources": [], "prvDiaForcedSources": [fp[1]]}
    p2 = {"diaObject": {"diaObjectId": 170230797176406143}, "diaSource": src[0],
          "prvDiaSources": [src[1]], "prvDiaForcedSources": [fp[1], fp[0]]}
    lc = _load([p1, p2], survey="lsst")
    assert lc.name == "170230797176406143"
    assert list(lc.band) == ["lssty", "lsstg", "lsstr", "lsstu"]
    assert list(lc.upper_limit) == [True, False, False, True]
    assert lc.meta["n_dropped"]["duplicate"] == 2


def test_lsst_refuses_scienceflux():
    recs = [{"midpointMjdTai": 61000.0, "band": "g", "scienceFlux": 1e5, "scienceFluxErr": 300.0}]
    with pytest.raises(ValueError, match="scienceFlux"):
        _load(recs, survey="lsst")


def test_lsst_forced_needs_psffluxerr():
    with pytest.raises(ValueError, match="psfFluxErr"):
        _load(_fink_sources()[:2], survey="lsst",
              limits=[{"midpointMjdTai": 61004.0, "band": "u", "m_lim": 24.0}])
    with pytest.raises(ValueError, match="limits="):             # forced photometry as detections
        _load(_fink_forced(), survey="lsst")


# ----------------------------------------------------------------------------------- bands etc.
def test_survey_bands_are_prefixed_and_known_to_the_models():
    assert SURVEY_BANDS["lsst"] == ("lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty")
    assert SURVEY_BANDS["ztf"] == ("ztfg", "ztfr", "ztfi")
    from whisper_cbpf.synphot.labels import redback_filter_table

    rb = redback_filter_table()
    for survey, labels in SURVEY_BANDS.items():
        for b in labels:
            assert wp.resolve_filter(b) == b                 # a filter of its own, not a group
            assert np.isfinite(wp.resolve_band(b)["lambda_eff"])
            if rb:                                           # redback reads it as the same filter
                assert rb[b] == b
            assert surveys.survey_band(b[-1], survey) == b   # bare letter -> survey label
    assert surveys.survey_band("zg", "ztf") == "ztfg"
    assert surveys.survey_band("ZTF_r", "ztf") == "ztfr"
    with pytest.raises(ValueError, match="ztfg, ztfr, ztfi"):
        surveys.survey_band("y", "ztf")
    with pytest.raises(ValueError, match="not one of the LSST bands"):
        surveys.survey_band("R", "lsst")                     # Cousins R is not LSST r


def test_bands_selection_takes_survey_letters():
    with pytest.warns(UserWarning, match="negative"):
        lc = _load(_fink_sources(), survey="lsst", limits=_fink_forced(), bands=["g", "lsstr"])
    assert lc.bands == ["lsstg", "lsstr"] and not lc.upper_limit.any()
    with pytest.raises(ValueError, match="ztfg, ztfr, ztfi"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _load(_alerce(), survey="ztf", bands=["y"])


def test_min_snr_cuts_detections_and_keeps_limits():
    recs = _alerce()
    recs["detections"][2]["sigmapsf"] = 0.5                     # SNR 2.2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = _load(recs, survey="ztf", min_snr=3)
    assert lc.upper_limit.sum() == 2
    assert np.all(lc.snr[~lc.upper_limit] >= 3) and (~lc.upper_limit).sum() == 2
    assert lc.meta["n_dropped"]["low_snr"] == 1


def test_window_is_applied_in_mjd_before_the_explosion_date():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = _load(_alerce(), survey="ztf", time_min=59998.5, time_max=60002.0,
                   explosion_date=59998.0)
    np.testing.assert_allclose(lc.time, [1.0, 2.5, 3.0])       # 59999, 60000.5, 60001 - 59998
    assert lc.meta["explosion_mjd"] == 59998.0
    assert lc.meta["first_detection_mjd"] == 60000.5          # stays on the MJD clock


def test_not_enough_data_raises():
    with pytest.raises(ValueError, match="no detection"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _load(_alerce(), survey="ztf", time_max=59999.5)       # limits only


def test_bad_arguments_name_the_fix():
    with pytest.raises(ValueError, match="'ztf', 'lsst'"):
        _load(_alerce(), survey="sdss")
    with pytest.raises(ValueError, match="band_lookup"):
        _load(_alerce(), survey="ztf", band_lookup=True)
    with pytest.raises(ValueError, match="survey="):
        _load(pd.DataFrame({"time": [1.0], "band": ["g"], "magnitude": [19.0],
                            "magnitude_err": [0.1]}), limits=[{"mjd": 0.0}])


def test_survey_reads_json_and_csv_paths(tmp_path):
    js = tmp_path / "alerce.json"
    js.write_text(json.dumps(_alerce()))
    with pytest.warns(UserWarning, match="negative"):
        a = _load(js, survey="ztf")
    csv = tmp_path / "fink.csv"
    pd.DataFrame(_fink_sources()).to_csv(csv, index=False)
    with pytest.warns(UserWarning, match="negative"):
        b = _load(csv, survey="lsst")
    assert a.n_points == 5 and b.bands == ["lsstg", "lsstr"]
    assert a.meta["source_file"] == str(js)
    with open(csv) as fh, pytest.warns(UserWarning, match="negative"):
        c = _load(fh, survey="lsst")                           # an open file works too
    assert c.bands == b.bands and np.array_equal(c.magnitude, b.magnitude)


# ----------------------------------------------------------------- generic path (no survey=)
def test_generic_bands_selection_is_normalised(tmp_path):
    """``bands=`` goes through the same label normalisation as the band column."""
    p = tmp_path / "zg.csv"
    p.write_text("time,band,magnitude,magnitude_err\n1.0,zg,19.0,0.1\n2.0,zr,19.1,0.1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = wp.load_lightcurve(p, bands=["zg", "zi"])
        assert lc.bands == ["ztfg"]                            # 0 points before the fix
        grouped = wp.load_lightcurve(p, bands=["zr"], band_lookup=True)
        assert grouped.bands == ["r-band"]
        with pytest.raises(ValueError, match=r"\['ztfg', 'ztfr'\]"):
            wp.load_lightcurve(p, bands=["lsstg"])


def test_generic_path_takes_dataframe_and_records():
    rows = [{"time": 1.0, "band": "g", "magnitude": 19.0, "magnitude_err": 0.1},
            {"time": 2.0, "band": "r", "magnitude": 19.2, "magnitude_err": 0.1}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = wp.load_lightcurve(pd.DataFrame(rows))
        b = wp.load_lightcurve(rows)
    assert a.n_points == b.n_points == 2 and a.bands == b.bands == ["g", "r"]


def test_docstring_examples_run():
    from whisper_cbpf.io import loader

    for mod in (loader, surveys):
        res = doctest.testmod(mod, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert res.attempted > 0 and res.failed == 0, mod.__name__


# ------------------------------------------------------------------ gate: the demo's cuts
DEMO = Path(os.environ.get("WHISPER_REALWORLD_DATA", ""))
needs_demo = pytest.mark.skipif(not (DEMO / "decision_points.json").exists(),
                                reason="reference data not found; set $WHISPER_REALWORLD_DATA")

# The reference settings: fitted bands, min_snr=3, a 30-day window
# from the first detection, and stages "first N detections" / "detections up to D days".
TARGETS = {"sn2025pgp": ("ztf", "ZTF25aaxwrva", 0.051),
           "ztf20achncvv": ("ztf", "ZTF20achncvv", 0.128),
           "tde2020neh": ("ztf", "ZTF20abgwfek", 0.062),
           "sn2025ajnc": ("ztf", "ZTF25acldfsf", 0.025),
           "sn2026jkr": ("lsst", "170230797176406143", 0.350)}
FIT_BANDS = {"ztf": ["g", "r", "i"], "lsst": ["g", "r", "i", "z"]}
WINDOW_DAYS, MIN_DET_EMCEE = 30.0, 6


def _demo_raw(key):
    survey, sid, z = TARGETS[key]
    raw = DEMO / "raw"
    if survey == "ztf":   # ALeRCE detections (isdiffpos=1 kept) and non-detections, as cached
        return survey, raw / f"{sid}.csv", raw / f"{sid}_nondet.csv", z
    return survey, raw / f"{sid}_sources.json", raw / f"{sid}_fp.json", z   # Fink, untouched


def _stage_rows(det, t_fd, point):
    if point["method"] == "abc":
        return det[:point["n"]]
    return det[det["time"] - t_fd <= float(point["stage"][1:])]


@needs_demo
@pytest.mark.parametrize("key", list(TARGETS))
def test_gate_demo_cuts_row_for_row(key):
    survey, x, lim, z = _demo_raw(key)
    lc = wp.load_lightcurve(x, survey=survey, limits=lim, redshift=z, bands=FIT_BANDS[survey],
                            min_snr=3)
    t_fd = lc.meta["first_detection_mjd"]
    # the window is cut on the MJD clock, before any time reference is set
    win = lc.select_time_window(None, t_fd + WINDOW_DAYS)
    det = win.where(upper_limit=False).to_pandas()
    points = json.loads((DEMO / "decision_points.json").read_text())[key]
    stages = sorted(p["stage"] for p in points["decision_points"])
    assert stages and stages == sorted(p.stem for p in (DEMO / "cuts" / key).glob("*.csv")), key
    for point in points["decision_points"]:
        cut = pd.read_csv(DEMO / "cuts" / key / f"{point['stage']}.csv")
        mine = _stage_rows(det, t_fd, point).reset_index(drop=True)
        assert len(mine) == len(cut) == point["n"], (key, point["stage"])
        assert list(mine["band"]) == list(cut["band"])
        np.testing.assert_array_equal(mine["time"].to_numpy(), cut["time"].to_numpy())
        np.testing.assert_allclose(mine["magnitude"], cut["magnitude"], rtol=1e-13, atol=0)
        # the demo wrote sigma = 1.0857362 psfFluxErr / psfFlux, a 4e-9 truncation of 2.5/ln 10
        np.testing.assert_allclose(mine["magnitude_err"], cut["magnitude_err"], rtol=5e-9, atol=0)
        assert not cut["upper_limit"].any()
    for skipped in points["skipped"]:           # stages the demo left out: too few detections
        stage = skipped["stage"]
        if stage.startswith("abc"):
            assert len(det) < int(stage[3:]), (key, stage)
        else:
            assert int((det["time"] - t_fd <= float(stage[1:])).sum()) < MIN_DET_EMCEE, (key, stage)


@needs_demo
@pytest.mark.parametrize("key", [k for k, v in TARGETS.items() if v[0] == "ztf"])
def test_gate_ztf_alert_fields_equal_the_cached_csv(key):
    """The demo cached ALeRCE's detections as a whisper CSV; ALeRCE's own field names load the same."""
    _, x, lim, z = _demo_raw(key)
    d, nd = pd.read_csv(x), pd.read_csv(lim)
    fid = {"zg": 1, "zr": 2, "zi": 3}
    det = [{"mjd": r.time, "fid": fid[r.band], "magpsf": r.magnitude, "sigmapsf": r.magnitude_err,
            "isdiffpos": 1, "magpsf_corr": r.magnitude - 1.0} for r in d.itertuples()]
    det.append({"mjd": float(d.time.iloc[3]) + 0.01, "fid": 1, "magpsf": 18.0, "sigmapsf": 0.05,
                "isdiffpos": -1, "magpsf_corr": 17.0})       # a negative subtraction: dropped
    nondet = [{"mjd": r.time, "fid": fid[r.band], "diffmaglim": r.diffmaglim}
              for r in nd.itertuples()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = wp.load_lightcurve(x, survey="ztf", limits=lim, redshift=z, min_snr=3)
        b = wp.load_lightcurve({"detections": det, "non_detections": nondet}, survey="ztf",
                               redshift=z, min_snr=3)
    for col in ("time", "band", "magnitude", "upper_limit"):
        np.testing.assert_array_equal(np.asarray(a[col]), np.asarray(b[col]))
    np.testing.assert_array_equal(np.asarray(a["magnitude_err"]), np.asarray(b["magnitude_err"]))
    assert b.meta["n_dropped"]["negative_flux"] == 1
