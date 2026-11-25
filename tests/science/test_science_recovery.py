"""(a) Known-answer recovery: are the posteriors calibrated at the recommended settings?

For every model family (kilonova one- and two-component, Arnett, magnetar, shock cooling + Arnett,
CSM + Arnett, TDE), parameter sets are drawn from the family's own prior (inside its constraint
walls), observed as LSST alerts (``_sim``: wide-fast-deep cadence, single-visit depths, LSST's
photometric error model; a kilonova gets nightly follow-up) with the explosion or merger date
known, read with ``load_lightcurve(..., survey="lsst")`` and fitted with ``wp.fit(...,
sampler="auto")`` at its default settings.

If the posteriors are calibrated, the central 68 % and 95 % intervals hold the true value 68 % and
95 % of the time (pooled over parameters, within 3 binomial standard errors), and the rank of the
truth among 49 posterior draws is uniform (simulation-based calibration, chi-square p > 0.001).

These studies need a GPU (about 30 s per supernova or kilonova fit, 2-3 min per TDE fit) and are
``slow``. Set ``WHISPER_VALIDATION_OUT`` to keep every injection's record (a rerun then resumes)
and ``WHISPER_SCIENCE_N`` to change the number of injections. The fast test at the bottom checks
the same machinery on a toy model on the CPU.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import _sim  # noqa: E402
import _studies  # noqa: E402

N_INJECTIONS = {"arnett": 30, "magnetar": 30, "shock_cooling_arnett": 30,
                "csm_shock_arnett": 30, "kilonova": 30, "kilonova_two": 30, "tde": 20}


def _n(family):
    return int(os.environ.get("WHISPER_SCIENCE_N") or N_INJECTIONS[family])


def _check(summary):
    for lev, c in summary["coverage"].items():
        lo, hi = c["band"]
        assert lo <= c["rate"] <= hi, (f"{int(float(lev) * 100)} % intervals hold the truth "
                                       f"{c['rate']:.2f} of the time over {c['n']} intervals; "
                                       f"calibrated is {lo:.2f}-{hi:.2f}")
    assert summary["sbc"]["p"] > 1e-3, f"SBC ranks are not uniform: {summary['sbc']}"


@pytest.mark.slow
@pytest.mark.parametrize("family", list(N_INJECTIONS))
def test_known_answer_recovery_at_the_default_settings(family, needs_gpu):
    recs = _studies.recovery(family, _n(family))
    summary = _studies.summarize_recovery(recs)
    _studies.record("recovery_summary", {"family": family, "settings": "default", **summary})
    _check(summary)


@pytest.mark.slow
@pytest.mark.parametrize("family, i", [("tde", 0), ("magnetar", 17)])
def test_the_default_start_gives_the_walkers_room_to_move(family, i, needs_gpu):
    """Injections on which the default start (``init="prior_scan"``) put every walker on one
    point and emcee refused to run ("Initial state has a large condition number")."""
    alert = _sim.simulate_alert(family, _studies.rng_for("recovery", family, i), object_id=i + 1)
    lc = _studies.alert_lc(alert, explosion_known=True)
    model = _sim.build_model(family, lc.band, alert["redshift"])
    res = _studies.fit_auto(lc, model, seed=i)
    assert res.n_samples > 0


@pytest.mark.slow
def test_explosion_time_prior_from_the_data_window_covers_the_truth(needs_gpu):
    """With the explosion date unknown, the default prior runs from the last non-detection to
    the first detection. The injections say how often the truth lies inside that window."""
    recs = _studies.recovery("arnett", _n("arnett"), study="recovery_texp_free",
                             explosion_known=False)
    inside = [r["t_exp_prior"][0] <= r["params"]["t_exp"]["truth"] <= r["t_exp_prior"][1]
              for r in recs]
    summary = _studies.summarize_recovery(recs)
    _studies.record("recovery_summary", {"family": "arnett", "settings": "t_exp free",
                                         "truth_in_window": float(np.mean(inside)), **summary})
    assert np.mean(inside) >= 0.95, (f"the data-window explosion-time prior excludes the true "
                                     f"explosion in {1 - np.mean(inside):.0%} of the alerts")
    _check(summary)


# ------------------------------------------------------------------------ the fast CPU version
def test_toy_known_answer_recovery_is_calibrated_on_the_cpu():
    """The same bookkeeping on the numpy flare model, which the CPU sampler mixes well."""
    import whisper_cbpf as wp
    from whisper_cbpf.priors import Prior, Uniform

    model = wp.get_model("flare")
    prior = Prior({"amplitude": Uniform(1.0, 10.0), "rise_time": Uniform(1.0, 10.0),
                   "decay_time": Uniform(5.0, 30.0)})
    t = np.linspace(0.5, 40.0, 25)
    hits, ranks = {0.68: [], 0.95: []}, []
    rng = np.random.default_rng(2026)
    for i in range(12):
        truth = {k: float(d.sample(rng)) for k, d in prior.distributions.items()}
        flux = model.predict(truth, t, None)
        err = 0.05 * flux.max() + 0.02 * flux
        lc = wp.LightCurve(time=t, band=["r"] * t.size, flux=flux + rng.normal(0, err),
                           flux_err=err)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = wp.fit(lc, model, sampler="mcmc", prior=prior, nsteps=1500, burnin=500, seed=i)
        for k, v in truth.items():
            s = res.samples[k].to_numpy()
            for lev in hits:
                hits[lev].append(_sim.interval_hits(s, v, lev))
            ranks.append(_sim.sbc_rank(s, v))
    for lev, h in hits.items():
        lo, hi = _sim.binomial_band(lev, len(h))
        assert lo <= np.mean(h) <= hi, (lev, np.mean(h))
    p, _counts = _sim.rank_uniformity_p(ranks)
    assert p > 1e-3
