"""The facts file: every number recomputed here by an independent route.

Percentiles by explicit linear interpolation of the sorted draws, prior widths with scipy.stats,
distance moduli with astropy's Planck18, slopes with numpy.polyfit, colours by brute force over all
pairs, ranking weights and grades by hand. ``prior_dominated`` is checked on a flat-likelihood toy
(a parameter the model ignores) fitted by nested sampling and by emcee.
"""
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import whisper_cbpf as wp
from whisper_cbpf import facts as F
from whisper_cbpf.priors import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform
from whisper_cbpf.results import data_hash, prior_record
from whisper_cbpf.samplers.base import SamplerResult


# ============================================================================== independent tools
def pct(x, q):
    """Percentile by linear interpolation between order statistics (numpy's default rule)."""
    x = np.sort(np.asarray(x, dtype=float))
    pos = (len(x) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(x) - 1)
    return x[lo] + (pos - lo) * (x[hi] - x[lo])


def distmod(z):
    from astropy.cosmology import Planck18
    return float(Planck18.distmod(z).value)


def mag_lc(rows, **kw):
    """A magnitude light curve from (time, band, mag, err, upper_limit) rows."""
    t, b, m, e, ul = (list(c) for c in zip(*rows))
    return wp.LightCurve(time=t, band=b, magnitude=m, magnitude_err=e, upper_limit=ul, **kw)


def simple_lc(**kw):
    rows = [(0.0, "ztfg", 19.0, 0.05, False), (0.4, "ztfr", 19.2, 0.05, False),
            (2.0, "ztfg", 18.5, 0.05, False), (2.3, "ztfr", 18.7, 0.05, False),
            (5.0, "ztfg", 18.9, 0.06, False), (5.2, "ztfr", 18.8, 0.06, False)]
    return mag_lc(rows, **kw)


def hand_result(samples, *, prior=None, lc=None, n_data=30, n_params=None, sampler="custom",
                info=None, max_ll=-10.0, model="toy"):
    """A SamplerResult built by hand, with the prior (and data hash) recorded as a fit records it."""
    df = pd.DataFrame(samples)
    k = len(df.columns) if n_params is None else n_params
    res = SamplerResult(sampler, model, list(df.columns), df, {},
                        {c: float(np.median(df[c])) if len(df) else float("nan") for c in df},
                        n_data, k, 0.5, dict(info or {}), max_log_likelihood=max_ll)
    res.provenance = {"model": {"prior": prior_record(prior) if prior is not None else None},
                      "data": {"hash": data_hash(lc) if lc is not None else None}}
    return res


# ======================================================================= parameters and widths
def test_percentiles_and_width_ratios_are_recomputed():
    rng = np.random.default_rng(1)
    n = 5000
    prior = Prior({"a": Uniform(0.5, 1.5), "m": LogUniform(1e-3, 1.0),
                   "z": TruncatedNormal(0.1, 0.02, 0.001, 0.3), "s": Normal(2.0, 0.5)})
    draws = {"a": rng.normal(1.0, 0.05, n), "m": 10 ** rng.uniform(-2.0, -1.0, n),
             "z": rng.normal(0.1, 0.004, n), "s": rng.normal(2.0, 0.1, n)}
    lc = simple_lc()
    f = F.result_facts(hand_result(draws, prior=prior, lc=lc), lc)
    scipy_prior = {"a": stats.uniform(0.5, 1.0), "m": stats.loguniform(1e-3, 1.0),
                   "z": stats.truncnorm((0.001 - 0.1) / 0.02, (0.3 - 0.1) / 0.02, 0.1, 0.02),
                   "s": stats.norm(2.0, 0.5)}
    for name, x in draws.items():
        rec = f["parameters"][name]
        for key, q in (("p16", 16), ("median", 50), ("p84", 84)):
            assert rec[key] == pytest.approx(pct(x, q), rel=1e-12, abs=1e-15)
        coord = np.log10 if name == "m" else (lambda v: v)
        assert rec["coordinate"] == ("log10" if name == "m" else "linear")
        post = coord(pct(x, 84)) - coord(pct(x, 16))
        pri = coord(scipy_prior[name].ppf(0.84)) - coord(scipy_prior[name].ppf(0.16))
        assert rec["posterior_width"] == pytest.approx(post, rel=1e-10)
        assert rec["prior_width"] == pytest.approx(pri, rel=1e-8)
        assert rec["width_ratio"] == pytest.approx(post / pri, rel=1e-8)
        assert rec["prior"]["median"] == pytest.approx(scipy_prior[name].ppf(0.5), rel=1e-8)
        assert rec["prior_dominated"] is bool(post / pri > 1 / math.sqrt(2))
    assert f["parameters"]["m"]["prior_dominated"] is False        # 1 of 3 decades: ratio 0.49
    assert f["parameters"]["s"]["at_prior_edge"] is False           # a Normal has no bound
    assert f["inputs"]["prior_source"] == "recorded with the fit"


def test_prior_dominated_fires_when_the_posterior_equals_the_prior():
    rng = np.random.default_rng(2)
    prior = Prior({"a": Uniform(0.0, 10.0), "b": Uniform(-1.0, 3.0)})
    draws = {"a": rng.normal(5.0, 0.1, 20000), "b": rng.uniform(-1.0, 3.0, 20000)}
    lc = simple_lc()
    f = F.result_facts(hand_result(draws, prior=prior, lc=lc), lc)
    b = f["parameters"]["b"]
    assert b["width_ratio"] == pytest.approx(1.0, abs=0.03)
    assert b["prior_dominated"] is True
    assert b["reading"].startswith("not enough data to measure b")
    assert f["parameters"]["a"]["prior_dominated"] is False
    assert f["caveats"]["prior_dominated"] is True
    assert f["caveats"]["prior_dominated_parameters"] == ["b"]


@pytest.fixture(scope="module")
def flat_toy():
    """A model that ignores its parameter ``b``: the likelihood is flat in b."""
    def predict(params, t, bands):
        return params["a"] * np.exp(-np.asarray(t, dtype=float) / 10.0)

    prior = Prior({"a": Uniform(0.5, 2.0), "b": Uniform(0.0, 1.0)})
    wp.register_model("facts_flat_toy", predict, ["a", "b"], prior=prior, overwrite=True)
    rng = np.random.default_rng(3)
    t = np.linspace(0.5, 20.0, 25)
    flux = np.exp(-t / 10.0) * (1.0 + 0.02 * rng.normal(size=t.size))
    return wp.LightCurve(time=t, band=["ztfr"] * t.size, flux=flux, flux_err=np.full(t.size, 0.02))


@pytest.mark.parametrize("sampler,kw", [("nested", {"nlive": 150}),
                                        ("mcmc", {"nsteps": 2500, "burnin": 500})])
def test_prior_dominated_on_a_flat_likelihood_toy(flat_toy, sampler, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit(flat_toy, "facts_flat_toy", sampler=sampler, seed=0, **kw)
    f = F.result_facts(res, flat_toy)
    b, a = f["parameters"]["b"], f["parameters"]["a"]
    assert 0.8 < b["width_ratio"] < 1.2, b["width_ratio"]
    assert b["prior_dominated"] is True
    assert a["prior_dominated"] is False and a["width_ratio"] < 0.1
    assert f["caveats"]["prior_dominated_parameters"] == ["b"]
    assert b["width_ratio"] == pytest.approx(
        (pct(res.samples["b"], 84) - pct(res.samples["b"], 16)) / 0.68, rel=1e-9)


DEMO = Path(os.environ.get("WHISPER_REALWORLD_DATA", ""))
V3_ABC6 = DEMO.parent / "versions" / "v3" / "runs" / "abc" / "sn2026jkr" / "abc6"


@pytest.mark.skipif(not V3_ABC6.is_dir(), reason="needs the reference fits "
                    "($WHISPER_REALWORLD_DATA)")
def test_plan_gate_prior_dominated_redshift_on_sn2026jkr_abc6():
    """The gate: v3's five families return the redshift prior as the posterior."""
    z0 = json.loads((DEMO / "targets.json").read_text())["sn2026jkr"]["redshift"]
    sigma = 0.05 * (1 + z0)                                   # v3's photo-z prior
    tn = TruncatedNormal(z0, sigma, max(0.001, z0 - 3 * sigma), z0 + 3 * sigma)
    cut = pd.read_csv(DEMO / "cuts" / "sn2026jkr" / "abc6.csv")
    lc = wp.LightCurve(time=cut.time, band=cut.band, magnitude=cut.magnitude,
                       magnitude_err=cut.magnitude_err, upper_limit=cut.upper_limit.astype(bool))
    ratios = {}
    for fam in ("arnett", "csm_shock_arnett", "magnetar", "shock_cooling_arnett",
                "tde_gaussianrise"):
        z_u = pd.read_csv(V3_ABC6 / fam / "samples.csv")["z_u"].to_numpy()
        z = tn.ppf(z_u)                                       # v3 sampled the CDF coordinate
        res = hand_result({"redshift": z}, prior=Prior({"redshift": tn}), lc=lc,
                          n_data=len(cut), sampler="abc", info={"n_accepted": len(z)})
        f = F.result_facts(res, lc)
        ratios[fam] = f["narrowing"]["redshift"]["width_ratio"]
        assert f["narrowing"]["redshift"]["prior_dominated"] is True, fam
        assert f["parameters"]["redshift"]["reading"].startswith("not enough data")
    assert all(0.95 < r < 1.05 for r in ratios.values()), ratios


def test_at_prior_edge_matches_the_diagnostics_rule():
    rng = np.random.default_rng(4)
    n = 4000
    c = rng.uniform(0.0, 1.0, n)
    c[: int(0.3 * n)] = 0.9995                          # 30 % piled at the upper bound (1.0)
    m = 10 ** rng.uniform(-3.0, -1.0, n)
    m[: int(0.2 * n)] = 1.0005e-3                        # 20 % at the lower bound, in log10
    prior = Prior({"c": Uniform(0.0, 1.0), "m": LogUniform(1e-3, 1.0)})
    lc = simple_lc()
    res = hand_result({"c": c, "m": m}, prior=prior, lc=lc)
    f = F.result_facts(res, lc)
    rc, rm = f["parameters"]["c"], f["parameters"]["m"]
    assert rc["at_prior_edge"] is True and rc["edge_side"] == "upper"
    assert rc["edge_fraction_upper"] == pytest.approx(np.mean(c >= 1.0 - 0.01))
    assert rm["at_prior_edge"] is True and rm["edge_side"] == "lower"
    lo, hi = -3.0, 0.0
    assert rm["edge_fraction_lower"] == pytest.approx(np.mean(np.log10(m) <= lo + 0.01 * (hi - lo)))
    assert "piles up against its upper prior bound" in rc["reading"]
    assert f["caveats"]["at_prior_edge_parameters"] == ["c", "m"]
    edge_row = [r for r in f["diagnostics"]["rows"] if r["check"] == "prior-edge pile-up"][0]
    assert edge_row["passed"] is False                   # the diagnostics' own rule agrees


# ======================================================================================= narrowing
def test_redshift_and_explosion_time_narrowing():
    rng = np.random.default_rng(5)
    prior = Prior({"a": Uniform(0.0, 2.0), "t_exp": Uniform(-10.0, 0.0),
                   "redshift": TruncatedNormal(0.1, 0.02, 0.001, 0.3)})
    draws = {"a": rng.normal(1.0, 0.1, 6000), "t_exp": rng.uniform(-3.0, -2.0, 6000),
             "redshift": rng.normal(0.1, 0.005, 6000)}
    lc = simple_lc()
    f = F.result_facts(hand_result(draws, prior=prior, lc=lc), lc)
    tn = stats.truncnorm((0.001 - 0.1) / 0.02, (0.3 - 0.1) / 0.02, 0.1, 0.02)
    z = f["narrowing"]["redshift"]
    ratio = (pct(draws["redshift"], 84) - pct(draws["redshift"], 16)) / (tn.ppf(0.84) - tn.ppf(0.16))
    assert z["fitted"] is True and z["width_ratio"] == pytest.approx(ratio, rel=1e-8)
    assert z["narrowing_factor"] == pytest.approx(1.0 / ratio, rel=1e-8)
    assert z["prior"]["p84"] == pytest.approx(tn.ppf(0.84), rel=1e-9)
    t = f["narrowing"]["t_exp"]
    ratio_t = (pct(draws["t_exp"], 84) - pct(draws["t_exp"], 16)) / 6.8
    assert t["width_ratio"] == pytest.approx(ratio_t, rel=1e-9)
    assert t["prior_dominated"] is False


def test_narrowing_when_not_fitted_says_where_the_value_came_from():
    rng = np.random.default_rng(6)
    lc = simple_lc(redshift=0.042)
    prior = Prior({"a": Uniform(0.0, 2.0), "t_exp": Fixed(-2.5)})
    res = hand_result({"a": rng.normal(1.0, 0.1, 500), "t_exp": np.full(500, -2.5)},
                      prior=prior, lc=lc, n_params=1)
    f = F.result_facts(res, lc)
    assert f["narrowing"]["redshift"] == {"fitted": False, "value": 0.042,
                                          "source": "the light curve's redshift"}
    assert f["narrowing"]["t_exp"]["value"] == -2.5 and f["fixed"] == {"t_exp": -2.5}
    assert "t_exp" not in f["parameters"]


# ============================================================================= absolute magnitude
def test_absolute_magnitude_range_is_asymmetric_from_a_redshift_prior():
    hint = {"type": "TruncatedNormal", "mu": 0.05, "sigma": 0.03, "low": 0.001, "high": 0.2}
    lc = simple_lc(redshift_prior=hint)
    res = hand_result({"a": np.random.default_rng(7).normal(1.0, 0.1, 500)},
                      prior=Prior({"a": Uniform(0.0, 2.0)}), lc=lc)
    f = F.result_facts(res, lc)
    am = f["absolute_magnitude"]
    tn = stats.truncnorm((0.001 - 0.05) / 0.03, (0.2 - 0.05) / 0.03, 0.05, 0.03)
    z16, z50, z84 = tn.ppf([0.16, 0.5, 0.84])
    assert am["redshift_kind"] == "prior"
    assert am["redshift"]["median"] == pytest.approx(z50, rel=1e-9)
    dm16, dm50, dm84 = distmod(z16), distmod(z50), distmod(z84)
    m = 18.5                                              # the brightest ztfg detection
    g = am["per_band"]["ztfg"]
    assert g["apparent_magnitude"] == m
    assert g["value"] == pytest.approx(m - dm50, abs=1e-5)
    assert g["brighter_by"] == pytest.approx(dm84 - dm50, abs=1e-5)
    assert g["fainter_by"] == pytest.approx(dm50 - dm16, abs=1e-5)
    assert g["fainter_by"] > 1.5 * g["brighter_by"]        # not symmetric (the demos' I8)
    assert g["range_at_prior_bounds"] == pytest.approx([m - distmod(0.2), m - distmod(0.001)],
                                                       abs=1e-5)


def test_absolute_magnitude_from_a_fitted_redshift_and_from_a_known_distance():
    rng = np.random.default_rng(8)
    lc = simple_lc()
    z = rng.normal(0.08, 0.01, 4000)
    res = hand_result({"redshift": z}, prior=Prior({"redshift": Uniform(0.01, 0.3)}), lc=lc)
    am = F.result_facts(res, lc)["absolute_magnitude"]
    assert am["redshift_kind"] == "posterior"
    r = am["per_band"]["ztfr"]
    assert r["value"] == pytest.approx(18.7 - distmod(pct(z, 50)), abs=1e-5)
    assert r["brighter_by"] == pytest.approx(distmod(pct(z, 84)) - distmod(pct(z, 50)), abs=1e-5)
    known = simple_lc(redshift=0.03)
    am = F.result_facts(hand_result({"a": np.ones(10)}, lc=known), known)["absolute_magnitude"]
    assert am["redshift_kind"] == "known" and am["per_band"]["ztfr"]["fainter_by"] == 0.0
    assert am["per_band"]["ztfr"]["value"] == pytest.approx(18.7 - distmod(0.03), abs=1e-5)
    ld = simple_lc(luminosity_distance=100.0, redshift=0.02)
    am = F.result_facts(hand_result({"a": np.ones(10)}, lc=ld), ld)["absolute_magnitude"]
    assert am["distance_modulus"]["value"] == pytest.approx(5 * math.log10(100e6 / 10.0))


# ========================================================================================== data
def lsst_like():
    rows = [(-3.0, "lsstr", 23.8, float("nan"), True), (-1.0, "lsstg", 24.0, float("nan"), True),
            (0.0, "lsstg", 21.5, 0.08, False), (0.02, "lsstr", 21.7, 0.09, False),
            (1.1, "lsstg", 21.0, 0.06, False), (1.9, "lsstr", 21.1, 0.07, False),
            (3.0, "lsstg", 20.6, 0.05, False), (3.4, "lsstr", 20.7, 0.05, False),
            (6.0, "lsstg", 20.9, 0.05, False), (6.6, "lsstr", 20.75, 0.05, False),
            (6.8, "lssti", 20.9, 0.07, False), (9.0, "lsstg", 21.4, 0.06, False),
            (9.1, "lssti", 20.8, 0.05, False)]
    return mag_lc(rows, name="toy LSST alert")


def test_data_block_is_recomputed_by_brute_force():
    lc = lsst_like()
    d = F.result_facts(hand_result({"a": np.ones(10)}, lc=lc), lc)["data"]
    assert d["bands"] == ["lsstg", "lsstr", "lssti"]                   # blue to red
    assert d["n_detections"] == 11 and d["n_upper_limits"] == 2
    assert d["last_nondetection_before_first_detection"] == {
        "time": -1.0, "band": "lsstg", "limiting_magnitude": 24.0}
    t, b, m = (np.asarray(lc[c]) for c in ("time", "band", "magnitude"))
    e, ul = np.asarray(lc["magnitude_err"]), np.asarray(lc["upper_limit"])
    for band in ("lsstg", "lsstr", "lssti"):
        sel = (b == band) & ~ul
        tb, mb, eb = t[sel], m[sel], e[sel]
        k = int(np.argmin(mb))
        rec = d["per_band"][band]
        assert rec["brightest"] == {"time": tb[k], "magnitude": mb[k], "error": eb[k]}
        for key, side, sign in (("rise_rate", slice(0, k + 1), -1), ("decline_rate",
                                                                     slice(k, None), 1)):
            ts, ms, es = tb[side], mb[side], eb[side]
            if len(ts) < 2:
                assert rec[key]["value"] is None
                assert rec[key]["reason"].startswith("not enough data")
                continue
            coef, cov = np.polyfit(ts, ms, 1, w=1.0 / es, cov="unscaled")
            assert rec[key]["value"] == pytest.approx(sign * coef[0], rel=1e-9)
            assert rec[key]["error"] == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-9)
    assert d["per_band"]["lsstg"]["state"] == "peaked"
    assert d["per_band"]["lssti"]["state"] == "rising"
    # colours by brute force: of all pairs within 1 d, the latest later point, then the closest
    got = {tuple(c["bands"]): c for c in d["colours"]}
    assert set(got) == {("lsstg", "lsstr"), ("lsstr", "lssti")}
    for (blue, red), c in got.items():
        pairs = [(max(t1, t2), -abs(t1 - t2), m1 - m2, math.hypot(e1, e2))
                 for t1, b1, m1, e1, u1 in zip(t, b, m, e, ul) if b1 == blue and not u1
                 for t2, b2, m2, e2, u2 in zip(t, b, m, e, ul) if b2 == red and not u2
                 if abs(t1 - t2) <= 1.0]
        best = max(pairs)
        assert c["time"] == best[0] and c["colour"] == pytest.approx(best[2])
        assert c["error"] == pytest.approx(best[3]) and c["separation_days"] == -best[1]


def test_few_detections_say_not_enough_data():
    lc = mag_lc([(0.0, "lsstg", 22.0, 0.1, False), (3.0, "lsstr", 21.5, 0.1, False)])
    d = F.result_facts(hand_result({"a": np.ones(5)}, lc=lc), lc)["data"]
    g = d["per_band"]["lsstg"]
    assert g["state"] == "single detection" and g["peak_observed"] is False
    assert g["rise_rate"]["value"] is None and "not enough data" in g["rise_rate"]["reason"]
    c = d["colours"][0]
    assert c["colour"] is None and c["reason"].startswith("not enough data")


def test_flux_light_curve_converts_and_skips_negative_flux():
    lc = wp.LightCurve(time=[0.0, 1.0, 2.0, 3.0], band=["ztfg"] * 4,
                       flux=[1e-4, 2e-4, -1e-5, 1.5e-4], flux_err=[1e-5] * 4)
    d = F.result_facts(hand_result({"a": np.ones(5)}, lc=lc), lc)["data"]
    assert d["n_detections"] == 3 and d["n_unusable"] == 1
    br = d["per_band"]["ztfg"]["brightest"]
    assert br["magnitude"] == pytest.approx(-2.5 * math.log10(2e-4 / 3631.0))
    assert br["error"] == pytest.approx(2.5 / math.log(10) * 1e-5 / 2e-4, rel=1e-9)


# ===================================================================================== fit block
def test_fit_block_uses_the_likelihood_maximum_and_flags_a_large_gain():
    lc = simple_lc()
    peak = {"params": {"a": 1.0}, "max_log_likelihood": 5.0, "start_log_likelihood": 3.0,
            "gain": 2.0, "at_edge": ["a"], "aic": 0.0, "bic": 0.0, "n_data": 30, "n_params": 1,
            "n_evals": 10, "runtime_s": 0.1, "method": "test"}
    res = hand_result({"a": np.random.default_rng(9).normal(1.0, 0.1, 800)},
                      prior=Prior({"a": Uniform(0.0, 2.0)}), lc=lc, max_ll=3.0,
                      info={"likelihood_max_opt": [dict(peak, max_log_likelihood=4.0), peak]})
    f = F.result_facts(res, lc)
    fit = f["fit"]
    assert fit["max_log_likelihood"] == 5.0 and fit["likelihood_max_gain"] == 2.0
    assert fit["optimised_max_log_likelihood"] == 5.0
    assert fit["aic"] == pytest.approx(-2 * 5.0 + 2 * 1)
    assert fit["bic"] == pytest.approx(-2 * 5.0 + 1 * math.log(30))
    assert f["caveats"]["large_likelihood_max_gain"] is True
    assert f["parameters"]["a"]["peak"] == 1.0 and f["parameters"]["a"]["peak_at_prior_edge"]
    small = F.result_facts(res, lc, thresholds={"likelihood_max_gain_large": 3.0})
    assert small["caveats"]["large_likelihood_max_gain"] is False
    assert small["thresholds"]["likelihood_max_gain_large"] == 3.0


def test_k_at_least_n_gives_no_aic_bic_but_a_statement():
    lc = simple_lc()
    res = hand_result({"a": np.ones(50), "b": np.ones(50) * 2, "c": np.ones(50) * 3},
                      lc=lc, n_data=3)
    f = F.result_facts(res, lc)
    assert f["fit"]["aic"] is None and f["fit"]["bic"] is None
    assert f["caveats"]["not_enough_data"] is True
    assert any(r.startswith("not enough data: 3 data points for 3") for r in f["readings"])


def test_no_posterior_draws():
    lc = simple_lc()
    res = hand_result({"a": np.array([], dtype=float)}, lc=lc, sampler="abc",
                      info={"n_accepted": 0})
    f = F.result_facts(res, lc)
    assert f["parameters"] == {} and f["caveats"]["no_posterior_draws"] is True
    assert f["readings"][0].startswith("not enough data: the fit returned no posterior draws")


def test_converged_and_stranded_walkers_come_from_the_diagnostics():
    lc = simple_lc()
    rng = np.random.default_rng(10)
    info = {"nwalkers": 8, "nsteps": 2000, "max_autocorr_time": 10.0, "stuck_walkers": [2, 5]}
    res = hand_result({"a": rng.normal(1.0, 0.1, 800)}, prior=Prior({"a": Uniform(0.0, 2.0)}),
                      lc=lc, sampler="mcmc", info=info)
    f = F.result_facts(res, lc)
    assert f["caveats"]["stranded_walkers"] is True
    assert f["caveats"]["converged"] is False
    report = res.diagnostics()
    assert f["diagnostics"]["reasons"] == report.reasons
    res.info["stuck_walkers"] = []
    f = F.result_facts(res, lc)
    assert f["caveats"]["stranded_walkers"] is False
    other = hand_result({"a": rng.normal(1.0, 0.1, 50)}, lc=lc)       # no convergence row at all
    assert F.result_facts(other, lc)["caveats"]["converged"] is None


# ==================================================================================== comparison
class FakeComparison:
    def __init__(self, table, results, criterion, winner, lc, **kw):
        self.table, self.results, self.criterion, self.winner, self.lc = (
            table, results, criterion, winner, lc)
        self.peaks = {}
        self.__dict__.update(kw)


def _fake_results(names, lc):
    rng = np.random.default_rng(11)
    prior = Prior({"a": Uniform(0.0, 2.0)})
    return {n: hand_result({"a": rng.normal(1.0, 0.1, 300)}, prior=prior, lc=lc, model=n)
            for n in names}


def test_comparison_ranking_is_recomputed_by_hand():
    lc = simple_lc()
    bic = {"m1": 100.0, "m2": 103.1, "m3": 112.0, "m4": 99.0}
    table = pd.DataFrame([
        {"model": "m1", "bic": bic["m1"], "status": "ranked", "left_out_reason": None},
        {"model": "m2", "bic": bic["m2"], "status": "ranked", "left_out_reason": None},
        {"model": "m3", "bic": bic["m3"], "status": "ranked", "left_out_reason": None},
        {"model": "m4", "bic": bic["m4"], "status": "left out",
         "left_out_reason": "not enough data: k = 6 >= n = 6"}])
    cmp = FakeComparison(table, _fake_results(["m1", "m2", "m3", "m4"], lc), "BIC", "m1", lc)
    f = F.comparison_facts(cmp)
    # no redshift given: the data's own absolute magnitude comes from the default redshift prior
    assert f["absolute_magnitude"]["redshift_kind"] == "prior"
    assert f["absolute_magnitude"]["redshift"]["median"] == pytest.approx(0.5005)
    assert list(f["models"]) == ["m1", "m2", "m3", "m4"]            # rank order, then the rest
    r = f["ranking"]
    assert r["winner"] == "m1" and r["runner_up"] == "m2"
    assert r["left_out"] == [{"model": "m4", "reason": "not enough data: k = 6 >= n = 6"}]
    w = {m: math.exp(-bic[m] / 2) for m in ("m1", "m2", "m3")}
    total = sum(w.values())
    cuts = (math.log(10 ** 0.5), math.log(10), math.log(100))
    grades = ("inconclusive", "substantial", "strong", "decisive")
    for row in r["models"]:
        m = row["model"]
        assert row["delta_bic"] == pytest.approx(bic[m] - 100.0)
        assert row["weight"] == pytest.approx(w[m] / total, rel=1e-12)
        if m != "m1":
            ln_b = (bic[m] - 100.0) / 2
            assert row["ln_b_winner_over_this"] == pytest.approx(ln_b)
            assert row["grade_winner_over_this"] == grades[sum(ln_b >= c for c in cuts)]
    assert r["grade"] == "substantial"                         # ln B = 1.55
    assert r["reading"].startswith("m1 is preferred over m2 by delta BIC = 3.1")


def test_comparison_by_ln_z_and_evidence_check_disagreement():
    lc = simple_lc()
    table = pd.DataFrame([{"model": "a", "bic": 10.0, "log_evidence": -5.0, "status": "ranked"},
                          {"model": "b", "bic": 12.0, "log_evidence": -9.0, "status": "ranked"}])
    cmp = FakeComparison(table, _fake_results(["a", "b"], lc), "ln Z", "a", lc)
    r = F.comparison_facts(cmp)["ranking"]
    assert r["criterion"] == "ln Z" and r["models"][1]["delta"] == 4.0
    assert r["models"][0]["weight"] == pytest.approx(1 / (1 + math.exp(-4.0)))
    assert r["grade"] == "strong"                              # ln B = 4 < ln 100
    check = {"ran": True, "models": ["a", "b"], "agrees": False, "ln_b": -1.2, "ln_b_err": 0.3,
             "reason": "requested"}
    cmp = FakeComparison(table, _fake_results(["a", "b"], lc), "BIC", "a", lc,
                         evidence_check=check, problems=["the evidence check disagrees"])
    f = F.comparison_facts(cmp)
    assert f["ranking"]["grade"] == "inconclusive"
    assert f["caveats"]["evidence_check_disagrees"] is True
    assert f["caveats"]["comparison_problems"] == ["the evidence check disagrees"]
    wrong = FakeComparison(table, _fake_results(["a", "b"], lc), "BIC", "b", lc)
    assert F.comparison_facts(wrong)["caveats"]["winner_differs_from_comparison"] is True


def test_comparison_facts_agree_with_compare():
    t = np.linspace(0.5, 30.0, 30)
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
                                          "decay_time": 15.0}, t, None)
    lc = wp.LightCurve(time=t, band=["ztfr"] * 30, flux=flux, flux_err=np.full(30, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500, evidence_check=False)
    f = cmp.facts()
    r = f["ranking"]
    assert r["winner"] == cmp.winner and r["criterion"] == cmp.criterion
    by = {row["model"]: row for row in r["models"]}
    for row in cmp.table.itertuples(index=False):
        mine = by[row.model]
        assert mine["delta"] == pytest.approx(row.delta, abs=1e-9)
        assert mine["weight"] == pytest.approx(row.weight, rel=1e-9)
        assert mine["diagnostics_passed"] == row.converged
        if row.model != cmp.winner:
            assert mine["grade_winner_over_this"] == row.grade
        assert f["models"][row.model]["fit"]["max_log_likelihood"] == pytest.approx(
            cmp.peaks[row.model].max_log_likelihood)
    assert r["grade"] == cmp.table.iloc[0]["grade"]
    assert list(f["models"]) == [row["model"] for row in r["models"]]


# =========================================================================== determinism, output
def test_facts_are_deterministic_sealed_and_nan_free(tmp_path):
    lc = lsst_like()
    rng = np.random.default_rng(12)
    res = hand_result({"a": rng.normal(1, 0.1, 400)}, prior=Prior({"a": Uniform(0, 2)}), lc=lc)
    f1, f2 = F.result_facts(res, lc), F.result_facts(res, lc)
    assert f1 == f2
    body = {k: v for k, v in f1.items() if k != "sha256"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False,
                       ensure_ascii=False).encode()
    assert f1["sha256"] == hashlib.sha256(canon).hexdigest()
    p1 = F.write_facts(f1, tmp_path / "a")
    p2 = F.write_facts(f2, tmp_path / "b.json")
    assert p1.name == "facts.json" and p1.read_bytes() == p2.read_bytes()

    def no_constants(name):
        raise AssertionError(f"non-finite number {name} in the facts file")
    assert json.loads(p1.read_text(), parse_constant=no_constants)["sha256"] == f1["sha256"]
    assert f1["inputs"]["data_sha256"] == data_hash(lc)
    assert set(f1["thresholds"]) == set(F.DEFAULT_THRESHOLDS)
    assert "prior_dominated" in f1["rules"]


def test_saved_and_loaded_result_gives_the_same_facts(tmp_path):
    t = np.linspace(0.5, 30.0, 30)
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
                                          "decay_time": 15.0}, t, None)
    lc = wp.LightCurve(time=t, band=["ztfr"] * 30, flux=flux, flux_err=np.full(30, 0.1),
                       redshift=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit(lc, "flare", sampler="mcmc", nsteps=800, burnin=200, seed=0)
    again = wp.load_result(res.save(tmp_path / "fit"))
    assert F.result_facts(again, lc) == F.result_facts(res, lc)


def test_data_that_differ_from_the_fit_are_flagged():
    lc = simple_lc()
    res = hand_result({"a": np.ones(20)}, lc=lc)
    assert F.result_facts(res, lc)["caveats"]["data_differs_from_fit"] is False
    other = lc.copy()
    other["magnitude"][0] = 19.5
    assert F.result_facts(res, other)["caveats"]["data_differs_from_fit"] is True


# ========================================================================================= errors
def test_errors_name_the_cause_and_the_fix():
    lc = simple_lc()
    res = hand_result({"a": np.ones(5)}, lc=lc)
    with pytest.raises(ValueError, match="prior_dominated_ratio"):
        F.result_facts(res, lc, thresholds={"prior_dominated": 0.9})
    with pytest.raises(ValueError, match="finite number > 0"):
        F.result_facts(res, lc, thresholds={"edge_band": -1})
    with pytest.raises(ValueError, match="three increasing"):
        F.result_facts(res, lc, thresholds={"grade_cuts_ln_b": (2.0, 1.0, 3.0)})
    with pytest.raises(TypeError, match="must be a dict"):
        F.result_facts(res, lc, thresholds=0.9)
    with pytest.raises(TypeError, match="pass lc="):
        F.result_facts(res, None)
    with pytest.raises(TypeError, match="expected a fit"):
        F.result_facts({"a": 1}, lc)
    with pytest.raises(TypeError, match="Comparison from wp.compare"):
        F.comparison_facts(object(), lc)
    with pytest.raises(TypeError, match="pass lc="):
        F.comparison_facts(FakeComparison(pd.DataFrame(), {}, "BIC", None, None))
    with pytest.raises(TypeError, match="write_facts writes a dict"):
        F.write_facts([1, 2], "x.json")
