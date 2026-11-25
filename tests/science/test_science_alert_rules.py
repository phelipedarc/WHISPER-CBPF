"""(e) The alert rules, end to end through ``compare``: pre-event rows, upper limits, survey fields.

- **Pre-event rows are never fitted.** Adding forced photometry and even a detection from before
  the explosion (or merger) to an alert changes the whole comparison by exactly zero: every
  table number and every posterior draw, at the same seed.
- **Upper limits are fitted by default.** An LSST alert's forced-photometry non-detections after
  the event enter the likelihood (the censored flux likelihood at the preset's 5 sigma) with no
  argument; leaving them out changes the answer.
- **Survey fields.** A ZTF alert packet is read from ``magpsf``/``sigmapsf``/``diffmaglim`` (never
  ``magpsf_corr``), an LSST one from ``psfFlux``/``psfFluxErr`` (never ``scienceFlux``), and both go
  straight into ``compare``.

The fast tests use the numpy ``flare`` and ``bazin`` models on the CPU; the kilonova version needs
the JAX models and is slow.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp

import _sim  # noqa: E402

MJD0 = _sim.MJD_EVENT
FAST = dict(nsteps=600, burnin=200, evidence_check=False, seed=3)
NUMERIC = ["n_params", "n_data", "max_log_likelihood", "aic", "bic", "delta", "weight"]


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _toy_alert(seed=11, *, pre_event_detection=False):
    """A toy LSST alert, and the same alert plus pre-event rows."""
    rng = np.random.default_rng(seed)
    t, b, _m5, flux, err, det = _sim.toy_visits(rng)
    post = t > 0
    clean = _sim.alert_packet(MJD0 + t[post], b[post], flux[post], err[post], det[post])
    det_full = det.copy()
    if pre_event_detection:                      # a pre-explosion precursor or artefact at 6 sigma
        k = int(np.flatnonzero(~post)[-1])
        flux = flux.copy()
        flux[k] = 6.0 * err[k]
        det_full[k] = True
    full = _sim.alert_packet(MJD0 + t, b, flux, err, det_full)
    return clean, full, int((~post).sum())


def _load(packet, **kw):
    return _quiet(wp.load_lightcurve, packet, survey="lsst", **kw).set_explosion_date(MJD0)


def _compare(lc, **kw):
    return _quiet(wp.compare, lc, ["flare", "bazin"], prior=_sim.toy_priors(), **{**FAST, **kw})


def test_pre_event_rows_change_the_comparison_by_exactly_zero():
    clean, full, n_pre = _toy_alert(pre_event_detection=True)
    lc_clean, lc_full = _load(clean), _load(full)
    assert len(lc_full) == len(lc_clean) + n_pre and n_pre >= 3
    assert (~lc_full.upper_limit & (lc_full.time <= 0)).sum() == 1      # the pre-event detection
    a, b = _compare(lc_clean), _compare(lc_full)
    pd.testing.assert_frame_equal(a.table[["model"] + NUMERIC], b.table[["model"] + NUMERIC])
    assert a.winner == b.winner
    for m in ("flare", "bazin"):
        ra, rb = a.results[m], b.results[m]
        assert rb.info["excluded_pre_event"] == n_pre and ra.info["excluded_pre_event"] == 0
        np.testing.assert_array_equal(ra.samples.to_numpy(), rb.samples.to_numpy())
        assert a.peaks[m].max_log_likelihood == b.peaks[m].max_log_likelihood


def test_upper_limits_after_the_event_are_fitted_by_default():
    clean, _full, _ = _toy_alert()
    lc = _load(clean)
    n_lim, n_det = int(lc.upper_limit.sum()), int((~lc.upper_limit).sum())
    assert n_lim >= 3, "the toy alert needs non-detections after the event"
    cmp = _compare(lc)
    for m in ("flare", "bazin"):
        res = cmp.results[m]
        assert res.n_data == n_det + n_lim                    # every limit is a data point
        assert "UpperLimits" in str(res.info.get("likelihood"))
        assert res.info.get("space") == "flux"
    det_only = _compare(lc.where(upper_limit=False))
    assert det_only.results["flare"].n_data == n_det
    assert det_only.peaks["flare"].max_log_likelihood != cmp.peaks["flare"].max_log_likelihood


def test_ztf_and_lsst_alert_fields_go_straight_into_compare():
    rng = np.random.default_rng(5)
    bright = dict(_sim.TOY_TRUTH, amplitude=4e-4)             # ~17.9 mag: ZTF detects it
    t, b, _m5, flux, err, det = _sim.toy_visits(rng, bands=("ztfg", "ztfr"), truth=bright,
                                                gap=(1.0, 2.0))
    ztf = _sim.ztf_packet(MJD0 + t, b, flux, err, det)
    lc = _quiet(wp.load_lightcurve, ztf, survey="ztf").set_explosion_date(MJD0)
    got = lc.magnitude[~lc.upper_limit]
    want = [r["magpsf"] for r in [ztf["candidate"]] + ztf["prv_candidates"]
            if r["magpsf"] is not None]
    np.testing.assert_allclose(np.sort(got), np.sort(want), rtol=0, atol=1e-12)   # not _corr
    assert set(lc.band) <= {"ztfg", "ztfr"}
    cmp = _compare(lc)
    assert cmp.winner in ("flare", "bazin")
    assert cmp.n_data == int((lc.time > 0).sum())            # the pre-explosion limits set no row

    clean, _full, _ = _toy_alert(seed=12)
    for s in clean["prvDiaSources"] + [clean["diaSource"]]:
        s["scienceFlux"] = 50.0 * s["psfFlux"]                # a decoy: host light included
    lc = _load(clean)
    det_rows = [clean["diaSource"]] + clean["prvDiaSources"]
    want = sorted(31.4 - 2.5 * np.log10(r["psfFlux"]) for r in det_rows)
    np.testing.assert_allclose(np.sort(lc.magnitude[~lc.upper_limit]), want, rtol=0, atol=1e-12)
    assert set(lc.band) <= set(_sim.WFD_BANDS)
    assert _compare(lc).winner in ("flare", "bazin")


@pytest.mark.slow
def test_pre_merger_rows_change_a_kilonova_comparison_by_exactly_zero():
    """The kilonova version: forced photometry and a detection before the merger change nothing."""
    pytest.importorskip("jax")
    rng = np.random.default_rng(21)
    alert = _sim.simulate_alert("kilonova", rng, min_detections=8)
    packet = alert["packet"]
    rows = [packet["diaSource"]] + packet["prvDiaSources"]
    forced = packet["prvDiaForcedSources"]
    t_all = np.array([f["midpointMjdTai"] for f in forced]) - MJD0
    pre = [f for f, t in zip(forced, t_all) if t <= 0]
    assert len(pre) >= 2
    clean = dict(packet, prvDiaForcedSources=[f for f, t in zip(forced, t_all) if t > 0])
    art = dict(rows[0], diaSourceId=999999, visit=999999,
               midpointMjdTai=pre[-1]["midpointMjdTai"], band=pre[-1]["band"],
               psfFlux=6.0 * pre[-1]["psfFluxErr"], psfFluxErr=pre[-1]["psfFluxErr"])
    full = dict(packet, prvDiaSources=packet["prvDiaSources"] + [art])
    z = alert["redshift"]

    def run(p):
        lc = _load(p, redshift=z)
        models = [_sim.build_model(f, lc.band, z) for f in ("kilonova", "kilonova_two")]
        # init="prior": the default start's gradient climb is not bitwise reproducible on a GPU
        # (test_science_devices.py), which would hide what this test is about.
        return lc, _quiet(wp.compare, lc, models, evidence_check=False, seed=1, init="prior")

    (lc_a, a), (lc_b, b) = run(clean), run(full)
    assert len(lc_b) == len(lc_a) + len(pre)         # the artefact replaces its epoch's limit
    assert (~lc_b.upper_limit & (lc_b.time <= 0)).sum() == 1
    pd.testing.assert_frame_equal(a.table[["model"] + NUMERIC], b.table[["model"] + NUMERIC])
    for m in a.results:
        np.testing.assert_array_equal(a.results[m].samples.to_numpy(),
                                      b.results[m].samples.to_numpy())
