"""(b) Model selection: the right class wins, few points say so, and the likelihood maximum is
steady.

- **Confusion matrix.** Alerts simulated from a supernova (Arnett), a TDE and a kilonova, all on
  one cadence (visits every 1-2 days from 10 days before to 60 days after the event) and at good
  SNR (at least 12 detections in 3 filters), are compared with ``wp.compare`` over the three
  classes with the explosion time unknown. The true class must rank first in at least 75 % of the
  alerts of each class.
- **Few detections.** The same alerts cut at their third detection must not produce a confident
  ranking: either "not enough data" (every model has at least as many parameters as points) or a
  grade no stronger than "substantial".
- **Why the ranking uses the likelihood maximum.** One alert is compared with several seeds. The
  sampler's best draw falls short of the likelihood peak by a different amount in every run and
  for every model; the optimised likelihood maximum of a model with one peak is the same in every
  run. So BIC at the likelihood maximum carries no sampler noise for such a model.

The physical studies need a GPU and are ``slow``; the toy versions at the bottom run on the CPU.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import whisper_cbpf as wp

import _sim  # noqa: E402
import _studies  # noqa: E402

N_PER_CLASS = int(os.environ.get("WHISPER_SCIENCE_N") or 8)
SN_FAMILIES = ("arnett", "magnetar", "shock_cooling_arnett", "csm_shock_arnett")
SEEDS = tuple(range(6))


@pytest.mark.slow
@pytest.mark.parametrize("truth", _studies.SELECTION_CANDIDATES)
def test_the_true_class_ranks_first_at_good_snr(truth, needs_gpu):
    recs = _studies.selection(truth, N_PER_CLASS)
    wins = [r["winner"] == truth for r in recs]
    _studies.record("selection_summary", {"truth": truth, "n": len(recs),
                                          "winners": [r["winner"] for r in recs],
                                          "grades": [_winner_grade(r) for r in recs],
                                          "true_class_first": float(np.mean(wins))})
    assert np.mean(wins) >= 0.75, [r["winner"] for r in recs]


@pytest.mark.slow
@pytest.mark.parametrize("truth", _studies.SELECTION_CANDIDATES)
def test_an_alert_at_its_third_detection_gets_no_confident_ranking(truth, needs_gpu):
    recs = _studies.selection(truth, 3, early=3)
    for r in recs:
        g = _winner_grade(r)
        assert r["winner"] is None or g in (None, "inconclusive", "substantial"), r["headline"]
        if r["winner"] is None:
            assert "not enough data" in r["headline"]


@pytest.mark.slow
@pytest.mark.parametrize("truth", ("arnett", "csm_shock_arnett"))
def test_the_likelihood_maximum_removes_the_seed_noise_of_the_best_draw(truth, needs_gpu):
    """Six seeds, explosion date known, the four supernova families.

    - the optimised likelihood maximum is never below the sampler's best draw;
    - the winner by BIC at the likelihood maximum is the same for every seed;
    - where a model's peak is unique (its optimised ln L agrees across seeds to 0.05), the
      sampler's best draw varies more than that: that seed noise is what the maximum-likelihood
      optimisation removes.
    A model with several local peaks (the CSM family here) can climb to different ones from
    different runs; docs/VALIDATION.md reports it.
    """
    recs = _studies.likelihood_max_stability(truth, SN_FAMILIES, SEEDS)
    s = _studies.summarize_likelihood_max_stability(recs)
    _studies.record("likelihood_max_stability_summary", {"truth": truth, **s})
    for m, v in s["models"].items():
        assert v["min_gain"] >= -1e-6, (m, v)
    assert len(s["winners"]["optimised"]) == 1, s["winners"]
    unique = {m: v for m, v in s["models"].items() if v["optimised_range"] < 0.05}
    assert unique, s["models"]
    for m, v in unique.items():
        assert v["sampler_range"] > v["optimised_range"], (m, v)


def _winner_grade(rec):
    for row in rec["table"]:
        if row["model"] == rec["winner"]:
            return row["grade"]
    return None


# ----------------------------------------------------------------------- fast CPU versions
def _toy_lc(seed, n_detections=None):
    rng = np.random.default_rng(seed)
    t, b, _m5, flux, err, det = _sim.toy_visits(rng)
    packet = _sim.alert_packet(_sim.MJD_EVENT + t, b, flux, err, det)
    if n_detections is not None:
        packet = _sim.truncate_packet(packet, n_detections)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = wp.load_lightcurve(packet, survey="lsst")
    return lc.set_explosion_date(_sim.MJD_EVENT)


def _toy_compare(lc, seed, **kw):
    return _studies.compare_quiet(lc, ["flare", "bazin"], prior=_sim.toy_priors(),
                                  evidence_check=False, seed=seed, **kw)


def test_toy_likelihood_maximum_is_steadier_than_the_best_draw():
    lc = _toy_lc(31)
    recs = [_studies.comparison_record(_toy_compare(lc, s, nsteps=300, burnin=100))
            for s in range(5)]
    s = _studies.summarize_likelihood_max_stability(recs)
    for m, v in s["models"].items():
        assert v["optimised_sd"] < v["sampler_sd"], (m, v)
        assert v["optimised_sd"] < 0.05, (m, v)
    assert s["orders"]["optimised"] == 1, s


def test_toy_alert_with_three_points_says_not_enough_data():
    lc = _toy_lc(32, n_detections=3)
    lc = lc.where(upper_limit=False)                       # three points: k >= n for both
    cmp = _toy_compare(lc, 0, nsteps=300, burnin=100)
    assert cmp.winner is None
    assert "not enough data" in repr(cmp)
    assert set(cmp.table["status"]) == {"left out"}
    assert all("not enough data" in r for r in cmp.table["left_out_reason"])
