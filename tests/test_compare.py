"""``compare``: fit several models to one light curve, find each likelihood maximum, rank, weigh and
grade.

The claims, each a test:

1. **The ranking rule** (pure, no fitting): BIC order, dBIC, weights ``exp(-dBIC/2)``, Jeffreys'
   grade at its cuts; ln Z only when every ranked model has a trustworthy one; the demos' rule for
   the number of points (only the most common ``n`` is compared); ``k >= n``, failed fits, empty
   posteriors and non-finite BIC are left out, each with its reason.
2. **It reproduces the demos' rankings** from their own numbers (``ranking.json`` of
   two earlier releases of a follow-up study, read-only): the same order, ln B and grade wherever every family has
   ``k < n``; where some have ``k >= n``, the same order over the rest.
3. **End to end** on a toy (CPU ``mcmc``): the right winner, likelihood maxima never below the
   sampler's, a table with the documented columns (gaps are ``None``, not NaN, under pandas 3),
   the convergence report attached.
4. **The evidence check** runs nested sampling on the top two and says whether ln Z agrees; a
   disagreement makes the grade "inconclusive" with the reason.
5. **Save / load** gives back the same table, peaks, data and results; a cache directory resumes.
6. **Same data for every model**: upper limits reach every fit (flux space, censored likelihood);
   a model whose fit left out pre-event rows is not ranked against one that kept them.
7. **Family names are bound to the data**: bands, redshift fixed or free, the explosion-time window
   from the last non-detection before the first detection (the pre-event rule), and a caveat when
   only the fallback window exists; an LSST alert compared by family names alone (slow).
8. **Errors name the cause and the fix.**
"""
from __future__ import annotations

import glob
import importlib
import json
import math
import os
import warnings

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.compare import COLUMNS, Comparison, compare

C = importlib.import_module("whisper_cbpf.compare")     # wp.compare is the function

ANCHORS = os.environ.get("WHISPER_REFERENCE_RANKINGS", "")
TRUTH = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
FAST = {"nsteps": 1500, "burnin": 500}


def _toy(n=30, seed=1, name="toy"):
    t = np.linspace(0.5, 30.0, n)
    flux = wp.get_model("flare").predict(TRUTH, t, None)
    noisy = flux + np.random.default_rng(seed).normal(0.0, 0.1, n)
    return wp.LightCurve(time=t, band=["r"] * n, flux=noisy, flux_err=np.full(n, 0.1), name=name)


def _row(label, k, n, bic, *, lnz=float("nan"), lnz_ok=False, error=None, n_samples=100):
    return {"label": label, "error": error, "n_samples": n_samples, "n_params": k, "n_data": n,
            "max_ll": float("nan"), "aic": float("nan"), "bic": bic, "lnz": lnz,
            "lnz_err": 0.1, "lnz_ok": lnz_ok}


@pytest.fixture(scope="module")
def toy_cmp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compare(_toy(), ["flare", "bazin", "gaussian_rise"], evidence_check=False, **FAST)


# --- 1. the ranking rule ---------------------------------------------------------------------------
def test_bic_order_delta_and_weights():
    r = C._rank([_row("a", 3, 30, 10.0), _row("b", 4, 30, 4.0), _row("c", 3, 30, 7.0)])
    assert r["criterion"] == "BIC" and r["order"] == ["b", "c", "a"]
    assert r["delta"] == {"b": 0.0, "c": 3.0, "a": 6.0}
    w = np.exp(-np.array([0.0, 3.0, 6.0]) / 2.0)
    assert np.allclose([r["weight"][k] for k in "bca"], w / w.sum(), rtol=0, atol=1e-15)
    assert r["grade"]["b"] == C._grade(1.5) == "substantial"         # winner vs runner-up
    assert r["grade"]["a"] == C._grade(3.0) == "strong"              # winner vs that model


@pytest.mark.parametrize("ln_b,grade", [(0.0, "inconclusive"), (1.1512, "inconclusive"),
                                        (1.1513, "substantial"), (2.3025, "substantial"),
                                        (2.3026, "strong"), (4.6051, "strong"),
                                        (4.6052, "decisive"), (float("nan"), None)])
def test_jeffreys_grade_cuts(ln_b, grade):
    assert C._grade(ln_b) == grade


def test_ln_z_only_when_every_ranked_model_has_one():
    rows = [_row("a", 3, 30, 10.0, lnz=5.0, lnz_ok=True), _row("b", 4, 30, 4.0, lnz=9.0,
                                                              lnz_ok=True)]
    r = C._rank(rows)
    assert r["criterion"] == "ln Z" and r["order"] == ["b", "a"]
    assert r["delta"]["a"] == pytest.approx(4.0)
    assert r["weight"]["b"] == pytest.approx(1.0 / (1.0 + math.exp(-4.0)))
    rows[0]["lnz_ok"] = False                                        # one untrusted: BIC
    assert C._rank(rows)["criterion"] == "BIC"


def test_left_out_models_each_with_a_reason():
    rows = [_row("ok1", 3, 20, 1.0), _row("ok2", 3, 20, 2.0),
            _row("other_n", 3, 19, -50.0),                          # fewer points
            _row("too_many", 20, 20, -99.0),                        # k >= n
            _row("failed", None, None, float("nan"), error="RuntimeError: boom", n_samples=0),
            _row("empty", 3, 20, float("nan"), n_samples=0),
            _row("dark", 3, 20, float("nan"))]
    r = C._rank(rows)
    assert r["order"] == ["ok1", "ok2"] and r["ref_n"] == 20
    why = r["left_out"]
    assert "n = 19 points, the others to n = 20" in why["other_n"]
    assert "not enough data: k = 20 free parameters >= n = 20" in why["too_many"]
    assert why["failed"] == "the fit failed: RuntimeError: boom"
    assert "no posterior draws" in why["empty"]
    assert "no finite BIC" in why["dark"]


def test_most_common_n_first_on_a_tie():
    r = C._rank([_row("a", 2, 10, 5.0), _row("b", 2, 12, 1.0)])
    assert r["ref_n"] == 10 and r["order"] == ["a"]


def test_nothing_rankable_gives_no_order():
    r = C._rank([_row("a", 12, 10, 1.0)])
    assert r["order"] == [] and "not enough data" in r["left_out"]["a"]


# --- 2. the demos' rankings ------------------------------------------------------------------------
def _anchor_files():
    return sorted(glob.glob(f"{ANCHORS}/v[23]/stages/*/*/ranking.json"))


@pytest.mark.skipif(not _anchor_files(),
                    reason="reference rankings not found; set $WHISPER_REFERENCE_RANKINGS")
def test_reproduces_the_demos_rankings_from_their_numbers():
    same, filtered = 0, 0
    for path in _anchor_files():
        doc = json.load(open(path))
        fams = doc["families"]
        rows = [_row(f, r["k"], r["n"], r["bic"]) for f, r in fams.items() if r["status"] == "ok"]
        r = C._rank(rows)
        if all(fr["k"] < fr["n"] for fr in fams.values()):
            assert r["order"] == doc["order"], path
            assert r["delta"][r["order"][1]] / 2 == pytest.approx(doc["ln_bf_best_vs_runner_up"],
                                                                  abs=1e-9), path
            assert r["grade"][r["order"][0]] == doc["strength"], path
            same += 1
        else:
            assert r["order"] == [f for f in doc["order"] if fams[f]["k"] < fams[f]["n"]], path
            filtered += 1
    assert same >= 20 and filtered >= 20          # 34 and 26 of the 60 v2/v3 decision points


# --- 3. end to end -----------------------------------------------------------------------------------
def test_toy_comparison_end_to_end(toy_cmp):
    cmp = toy_cmp
    assert cmp.winner == "flare" and cmp.criterion == "BIC" and cmp.n_data == 30
    assert list(cmp.table.columns) == COLUMNS
    assert list(cmp.table["model"]) == ["flare", "bazin", "gaussian_rise"]
    assert (cmp.table["status"] == "ranked").all()
    assert cmp.table["weight"].sum() == pytest.approx(1.0)
    assert cmp.table["delta"].iloc[0] == 0.0 and (cmp.table["delta"].diff().iloc[1:] >= 0).all()
    for lab, res in cmp.results.items():
        peak = cmp.peaks[lab]
        assert peak.max_log_likelihood >= res.max_log_likelihood - 1e-9
        row = cmp.table.set_index("model").loc[lab]
        assert row["bic"] == pytest.approx(peak.bic) and row["sampler"] == "mcmc"
        assert row["converged"] is cmp.diagnostics[lab].passed
        assert res.info["likelihood_max_opt"]["max_log_likelihood"] == peak.max_log_likelihood
        assert set(cmp.timing[lab]) == {"fit_s", "likelihood_max_opt_s", "evidence_s"}
    text = cmp.summary()
    assert text.startswith("Comparison of 3 models on 30 points by BIC: winner 'flare'")
    assert "Read next" in text and "read next" in repr(cmp)


def test_likelihood_max_opt_false_ranks_on_the_samplers_best_draws():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare(_toy(), ["flare", "bazin"], likelihood_max_opt=False, evidence_check=False,
                      **FAST)
    assert cmp.peaks == {}
    row = cmp.table.set_index("model").loc["flare"]
    assert row["bic"] == cmp.results["flare"].bic
    assert any("likelihood_max_opt=False" in p for p in cmp.problems)


def test_a_failed_fit_is_left_out_and_all_failed_raises():
    def boom(params, t, bands):
        raise RuntimeError("model exploded")

    bad = wp.Model(name="exploding", predict=boom, parameters=["a"],
                   default_prior=wp.Prior({"a": wp.Uniform(0.0, 1.0)}))
    with pytest.warns(UserWarning, match="'exploding' failed"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="(?!compare:)")
            cmp = compare(_toy(), ["flare", bad], evidence_check=False, **FAST)
    row = cmp.table.set_index("model").loc["exploding"]
    assert row["status"] == "left out" and "model exploded" in row["left_out_reason"]
    with pytest.raises(RuntimeError, match="model exploded"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compare(_toy(), [bad], evidence_check=False, **FAST)


def test_not_enough_data_says_so_instead_of_a_number():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare(_toy(n=4), ["flare", "bazin"], evidence_check=False, nsteps=600,
                      burnin=200)
    row = cmp.table.set_index("model").loc["bazin"]
    assert row["status"] == "left out" and row["left_out_reason"].startswith("not enough data")
    assert math.isnan(row["weight"]) and row["grade"] is None
    assert any("only 'flare' could be ranked" in p for p in cmp.problems)


def test_gaps_in_text_columns_are_none_and_the_summary_prints_them(toy_cmp):
    # pandas >= 3 stores the gaps of a string column as NaN; a graded ranked row next to a left-out
    # row then gave a NaN grade that crashed summary() (the sn2025ajnc/d30 gate run).
    rows = C._records(toy_cmp.table)
    rows[2].update(status="left out", grade=None, bic=float("nan"), delta=float("nan"),
                   weight=float("nan"), left_out_reason="not enough data: k = 13 >= n = 13")
    table = C._table(rows)
    assert table.loc[2, "grade"] is None and table.loc[0, "left_out_reason"] is None
    assert isinstance(table.loc[0, "grade"], str) and table.loc[0, "converged"] in (True, False)
    cmp = Comparison.__new__(Comparison)
    cmp.__dict__.update(toy_cmp.__dict__)
    cmp.table = table
    text = cmp.summary()
    assert "Left out:\n  - gaussian_rise: not enough data" in text


# --- 4. the evidence check ---------------------------------------------------------------------------
def test_evidence_check_runs_nested_on_the_top_two():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare(_toy(n=20), ["flare", "bazin"], evidence_check=True, **FAST)
    ec = cmp.evidence_check
    assert ec["ran"] and ec["models"] == ["flare", "bazin"] and ec["agrees"] is True
    assert set(cmp.evidence) == {"flare", "bazin"}
    assert cmp.table["log_evidence"].notna().all()
    assert cmp.criterion == "ln Z"               # the two are all the ranked models
    assert f"agrees with BIC: ln Z(flare) - ln Z(bazin) = {ec['ln_b']:.2f} +/- " in cmp.summary()


def test_auto_evidence_check_is_skipped_on_a_clear_gap(toy_cmp):
    ec = toy_cmp.evidence_check
    assert ec["ran"] is False and ec["reason"].startswith("not requested")
    rk = C._rank([_row("a", 3, 30, 0.0), _row("b", 3, 30, 20.0)])
    assert not C._should_check("auto", rk) and C._should_check(True, rk)
    close = C._rank([_row("a", 3, 30, 0.0), _row("b", 3, 30, 4.0)])      # ln B = 2 < ln 10
    assert C._should_check("auto", close)
    assert "not needed" in C._no_check_reason("auto", rk)


def test_a_disagreeing_evidence_check_makes_the_grade_inconclusive(toy_cmp):
    recs = {lab: {"label": lab, "model": toy_cmp.models[lab], "result": toy_cmp.results[lab],
                  "peak": toy_cmp.peaks[lab], "diagnostics": toy_cmp.diagnostics[lab],
                  "evidence": None, "problems": [], "error": None, "sampler": "mcmc",
                  "timing": {}, "fit_prior": None}
            for lab in ("flare", "bazin")}
    ranking = C._rank([C._rank_row(r) for r in recs.values()])
    check = {"requested": True, "ran": True, "models": ["flare", "bazin"], "agrees": False,
             "ln_b": -1.5, "ln_b_err": 0.4, "reason": "requested"}
    cmp = C._build(recs, ranking, check, _toy(), True, {})
    assert cmp.table.set_index("model").loc["flare", "grade"] == "inconclusive"
    assert any("prefers 'bazin' over 'flare' by ln B = 1.50" in p for p in cmp.problems)


# --- 5. save / load / resume ---------------------------------------------------------------------------
def test_save_and_load_round_trip(toy_cmp, tmp_path):
    toy_cmp.save(tmp_path / "cmp")
    back = Comparison.load(tmp_path / "cmp")
    assert back.table.equals(toy_cmp.table)
    assert back.winner == toy_cmp.winner and back.criterion == toy_cmp.criterion
    assert {k: v.to_dict() for k, v in back.peaks.items()} == \
        {k: v.to_dict() for k, v in toy_cmp.peaks.items()}
    assert all(back.results[k].summary == toy_cmp.results[k].summary for k in back.results)
    assert np.array_equal(back.lc.time, toy_cmp.lc.time)
    assert np.array_equal(back.lc.flux, toy_cmp.lc.flux)
    assert back.models["flare"] is wp.get_model("flare")
    assert back.diagnostics["flare"].passed == toy_cmp.diagnostics["flare"].passed
    assert back.summary() == toy_cmp.summary()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        toy_cmp.save(tmp_path / "cmp")
    with pytest.raises(FileNotFoundError, match="comparison.json is missing"):
        Comparison.load(tmp_path / "nothing")


def test_cache_dir_resumes(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = compare(_toy(), ["flare", "bazin"], cache_dir=tmp_path, evidence_check=False,
                        **FAST)
        again = compare(_toy(), ["flare", "bazin"], cache_dir=tmp_path, evidence_check=False,
                        **FAST)
    assert all(getattr(r, "loaded_from", None) for r in again.results.values())
    assert again.table.drop(columns="problems").equals(first.table.drop(columns="problems"))


# --- 6. same data for every model ------------------------------------------------------------------------
def test_upper_limits_reach_every_fit():
    lc = _toy(n=24)
    ul = np.zeros(24, dtype=bool)
    ul[-3:] = True                                            # late limits above the model
    lc["upper_limit"] = ul
    lc["flux"][ul] = 1.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare(lc, ["flare", "bazin"], evidence_check=False, **FAST)
    for res in cmp.results.values():
        assert res.info["space"] == "flux" and res.n_data == 24
        assert res.info["likelihood"] == "GaussianLikelihoodWithUpperLimits"
    assert cmp.n_data == 24


def test_a_fit_with_pre_event_rows_left_out_is_not_ranked_against_one_without():
    rows = [_row("keeps_all", 3, 20, 1.0), _row("drops_pre_event", 4, 17, -9.0),
            _row("also_all", 4, 20, 2.0)]
    rows[1]["excluded"] = 3
    r = C._rank(rows)
    assert r["order"] == ["keeps_all", "also_all"]
    assert "n = 17 points (3 pre-event row(s) left out by its fit)" in \
        r["left_out"]["drops_pre_event"]


# --- 7. family names bound to the data -------------------------------------------------------------------
@pytest.fixture
def x64():
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _alert(redshift=None, with_limit=True):
    t = [60000.0, 60002.0, 60003.0, 60004.5, 60006.0]
    ul = [with_limit, False, False, False, False]
    return wp.LightCurve(time=t, band=["lsstg", "lsstr", "lsstg", "lsstr", "lssti"],
                         magnitude=[22.5, 21.0, 20.6, 20.4, 20.5],
                         magnitude_err=[np.nan, 0.1, 0.1, 0.1, 0.1], upper_limit=ul,
                         redshift=redshift, name="alert")


def test_family_bound_to_the_alert(x64):
    m = C._bind_family("magnetar", _alert(), None)
    assert m.parameters[-2:] == ["t_exp", "redshift"]
    assert m.default_prior.distributions["t_exp"].bounds == (60000.0, 60002.0)
    assert m.default_prior.distributions["redshift"].bounds == (0.001, 1.0)
    fixed = C._bind_family("arnett", _alert(redshift=0.05), None)
    assert fixed.parameters[-1] == "t_exp" and "redshift" not in fixed.parameters
    exploded = _alert(redshift=0.05).set_explosion_date(59999.0)
    assert "t_exp" not in C._bind_family("arnett", exploded, None).parameters
    tde = C._bind_family("tde", _alert(), None)
    assert tde.parameters[-2:] == ["t_exp", "redshift"] and "peak_time" in tde.parameters
    assert C._bind_family("no_such_family", _alert(), None) is None
    custom = C._bind_family("arnett", _alert(), wp.Prior({"t_exp": wp.Uniform(59990.0,
                                                                              60002.0)}))
    assert custom.default_prior.distributions["t_exp"].bounds == (59990.0, 60002.0)


def test_a_family_explosion_time_prior_given_to_compare_is_used_as_given(x64):
    """prior={family: Prior({"t_exp": ...})} reaches the fit as given; left to the bound model's
    default prior, the fit's pre-event rule cut it back to the data window."""
    from whisper_cbpf.samplers.base import _t_exp_prior

    lc = _alert(redshift=0.05)
    wide = wp.Prior({"t_exp": wp.Uniform(59970.0, 60002.0)})
    entry = C._resolve_models(lc, ["arnett"], {"arnett": wide})["arnett"]
    assert entry["prior"].distributions["t_exp"].bounds == (59970.0, 60002.0)
    dist, rec = _t_exp_prior(lc, entry["model"], entry["prior"], entry["prior"])
    assert dist.bounds == (59970.0, 60002.0) and rec["source"] == "passed to fit"
    default = C._resolve_models(lc, ["arnett"], {})["arnett"]
    assert default["prior"] is None                       # the fit applies the window itself


def test_a_family_explosion_time_reference_does_not_come_from_the_prior(x64):
    """The epochs are shifted by the first detection, whatever the prior: a TruncatedNormal
    or Uniform prior predicts the same light curve, and a Normal one (no lower bound) is refused
    by the supernova's fixed epochs instead of predicting the magnitude floor everywhere."""
    lc = _alert(redshift=0.05)
    point = {"t_exp": 60001.0, "f_nickel": 0.1, "mej": 2.0, "vej": 1e4, "kappa": 0.2,
             "kappa_gamma": 10.0, "temperature_floor": 5e3}
    t, b = np.asarray(lc.time[1:]), np.asarray(lc.band[1:])
    flux = []
    for d in (wp.Uniform(59990.0, 60002.0), wp.TruncatedNormal(60001.0, 2.0, 59990.0, 60002.0)):
        m = C._bind_family("arnett", lc, wp.Prior({"t_exp": d}))
        flux.append(np.asarray(m.predict(point, t, b), float))
    np.testing.assert_allclose(flux[0], flux[1], rtol=1e-12)
    assert np.all(flux[0] > 1e-9)                          # bright, not the 3.6e-13 Jy floor
    with pytest.raises(ValueError, match="no lower bound"):
        C._bind_family("arnett", lc, wp.Prior({"t_exp": wp.Normal(60001.0, 2.0)}))
    tde = C._bind_family("tde", lc, wp.Prior({"t_exp": wp.Normal(60001.0, 2.0)}))
    assert "t_exp" in tde.parameters


def test_a_family_bound_to_a_long_light_curve_covers_its_epochs(x64):
    """A supernova followed for 300 d: the fixed diffusion epochs reach the last epoch at the
    earliest explosion allowed, instead of the 200 d a model built without times= covers."""
    t = np.array([61000.0, 61001.0, 61005.0, 61020.0, 61060.0, 61150.0, 61260.0, 61330.0])
    ul = np.array([True, False, False, False, False, False, True, True])
    lc = wp.LightCurve(time=t, band=["lsstg", "lsstr"] * 4,
                       magnitude=[24.5, 23.0, 21.5, 20.5, 21.5, 23.0, 24.6, 24.7],
                       magnitude_err=np.where(ul, np.nan, 0.05), upper_limit=ul, redshift=0.05)
    m = C._bind_family("arnett", lc, None)
    assert m.predict_jax.ctx.fixed_kw["max_phase_days"] >= (61330.0 - 61000.0) / 1.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert wp.log_density(lc, m).n_data == 7            # the phase check passes


def test_family_without_a_pre_detection_limit_gets_the_fallback_and_a_caveat(x64):
    lc = _alert(with_limit=False)
    entries = C._resolve_models(lc, ["arnett"], {})
    t_exp = entries["arnett"]["model"].default_prior.distributions["t_exp"]
    assert t_exp.bounds == lc.explosion_time_prior().bounds
    assert "fallback window" in entries["arnett"]["problems"][0]
    assert "prior={'arnett': Prior({'t_exp': ...})}" in entries["arnett"]["problems"][0]
    assert "problems" not in C._resolve_models(_alert(), ["arnett"], {})["arnett"]


@pytest.mark.slow
def test_an_lsst_alert_compared_by_family_names_alone(x64):
    """Data and model names only: an Arnett alert in three LSST bands, one limit before it."""
    from whisper_cbpf.models.cosmology import luminosity_distance_cm

    bands, z, t_expl = ["lsstg", "lsstr", "lssti"], 0.05, 60001.0
    truth = wp.supernova_model("arnett", bands, redshift=z,
                               dl_cm=float(luminosity_distance_cm(z)))
    t = 60000.0 + np.array([0.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 12.0, 14.0, 16.0, 19.0, 22.0,
                            25.0])
    b = np.array(["lsstr"] + [bands[i % 3] for i in range(len(t) - 1)])
    flux = np.asarray(truth.predict({"f_nickel": 0.1, "mej": 2.0, "vej": 1e4, "kappa": 1.0,
                                     "kappa_gamma": 1.0, "temperature_floor": 5e3},
                                    t - t_expl, b), float)
    mag = -2.5 * np.log10(flux / 3631.0) + np.random.default_rng(0).normal(0.0, 0.05, len(t))
    err = np.full(len(t), 0.05)
    mag[0], err[0] = 23.5, np.nan                            # the non-detection before it
    lc = wp.LightCurve(time=t, band=b, magnitude=mag, magnitude_err=err,
                       upper_limit=np.arange(len(t)) == 0, redshift=z, name="alert")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare(lc, ["arnett", "magnetar"], evidence_check=False, nsteps=1000, burnin=400)
    assert cmp.winner == "arnett" and cmp.n_data == 12
    assert (cmp.table["status"] == "ranked").all()
    assert set(cmp.table["sampler"]) == {C._choose_sampler("auto", cmp.models["arnett"])}
    for res in cmp.results.values():                 # the pre-event rule, the same for both
        assert res.info["excluded_pre_event"] == 1
        assert (res.info["t_exp_prior"]["low"], res.info["t_exp_prior"]["high"]) == \
            (60000.0, 60003.0)
    assert abs(float(np.median(cmp.results["arnett"].samples["t_exp"])) - t_expl) < 0.5
    assert not any("fitted redshift" in p or "fits the redshift" in p for p in cmp.problems)


# --- 8. errors -------------------------------------------------------------------------------------------
def test_errors_name_the_cause_and_the_fix():
    lc = _toy()
    with pytest.raises(ValueError, match="unknown model 'nope'.*registered model"):
        compare(lc, ["flare", "nope"])
    with pytest.raises(ValueError, match="listed twice"):
        compare(lc, ["flare", "flare"])
    with pytest.raises(ValueError, match="evidence_check must be"):
        compare(lc, ["flare"], evidence_check="sometimes")
    with pytest.raises(ValueError, match="sampler must be 'auto' or one of"):
        compare(lc, ["flare"], sampler="gibbs")
    with pytest.raises(ValueError, match=r"prior= names \['bazin'\]"):
        compare(lc, ["flare"], prior={"bazin": wp.Prior({"a": wp.Uniform(0, 1)})})
    with pytest.raises(TypeError, match="does not take \\['nlive'\\]"):
        compare(lc, ["flare"], sampler="mcmc", nlive=100)
    with pytest.raises(TypeError, match="list of model names"):
        compare(lc, "flare")
    with pytest.raises(ValueError, match="models is empty"):
        compare(lc, [])


def test_an_evidence_gap_the_likelihood_peaks_do_not_explain_is_put_down_to_the_priors():
    """The README's old example: two empirical shapes a whole 0.96 ln L apart at their peaks, and
    ln Z 14.84 apart, all of it from their default priors' volumes."""
    rows = {"gaussian_rise": {"max_ll": -133.75}, "bazin": {"max_ll": -134.71}}
    check = {"ran": True, "models": ["gaussian_rise", "bazin"], "ln_b": -14.84}
    note = C._prior_volume_note(check, rows)
    assert "differ by 0.96 ln L" in note and "comes from the priors" in note
    assert C._prior_volume_note(dict(check, ln_b=1.5), rows) == ""        # the peaks explain it
    assert C._prior_volume_note(dict(check, ln_b=None), rows) == ""       # no ln Z
    assert C._prior_volume_note({"ran": False}, rows) == ""


def test_a_family_name_without_jax_says_to_install_it(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "jax" else real(name, *a))
    monkeypatch.setattr(C, "_bind_family", lambda *a: None)       # as without JAX
    with pytest.raises(ValueError, match="unknown model 'arnett'.*JAX is not installed.*pip "
                                         "install jax"):
        compare(_toy(), ["flare", "arnett"])


def test_auto_sampler_choice(monkeypatch):
    assert C._choose_sampler("auto", wp.get_model("flare")) == "mcmc"
    jax_model = wp.get_model("flare_jax")
    monkeypatch.setattr(C, "_gpu_visible", lambda: True)
    assert C._choose_sampler("auto", jax_model) == "emcee_jax"
    monkeypatch.setattr(C, "_gpu_visible", lambda: False)
    assert C._choose_sampler("auto", jax_model) == "mcmc"
    assert C._choose_sampler("nested", jax_model) == "nested"


def test_likelihood_max_opt_and_diagnostics_use_the_prior_the_fit_ran_under():
    class _Res:
        info = {"t_exp_prior": {"type": "Uniform", "low": 2.0, "high": 5.0}}
    model = wp.Model(name="m", predict=lambda p, t, b: t, parameters=["a", "t_exp"],
                     default_prior=wp.Prior({"a": wp.Uniform(0, 1),
                                             "t_exp": wp.Uniform(-10.0, 5.0)}))
    used = C._prior_used({"model": model, "prior": None}, _Res())
    assert used.distributions["t_exp"].bounds == (2.0, 5.0)          # cut by the pre-event rule
    assert used.distributions["a"] is model.default_prior.distributions["a"]
    _Res.info = {}
    assert C._prior_used({"model": model, "prior": None}, _Res()) is model.default_prior


def test_delegated_modules_name_what_is_missing():
    with pytest.raises(ImportError, match="whisper_cbpf.no_such_module module"):
        C._module("no_such_module", "Comparison.thing()")


def test_discriminate_needs_two_ranked_models(toy_cmp):
    one = Comparison.__new__(Comparison)
    one.__dict__.update(toy_cmp.__dict__)
    one.table = toy_cmp.table.assign(status=["ranked", "left out", "left out"])
    with pytest.raises(ValueError, match="needs two ranked models"):
        one.discriminate([31.0], ["r"])


# --- 9. the same density for every ranked model; a saved comparison rebinds its families --------------
def test_a_fit_in_another_space_or_on_other_rows_is_not_ranked_against_the_others():
    rows = [_row("a", 3, 20, 1.0), _row("b", 3, 20, 2.0), _row("mag", 3, 20, -50.0),
            _row("rows", 3, 20, -60.0)]
    for r, key, space in zip(rows, ["flux:h1", "flux:h1", "magnitude:h1", "flux:h2"],
                             ["flux", "flux", "magnitude", "flux"]):
        r.update(data_key=key, space=space)
    r = C._rank(rows)
    assert r["order"] == ["a", "b"]
    assert "in magnitude space, the others in flux space" in r["left_out"]["mag"]
    assert "other rows of the light curve" in r["left_out"]["rows"]


def test_the_toy_comparison_records_what_each_fit_scored(toy_cmp):
    keys = {C._data_key(res, toy_cmp.lc) for res in toy_cmp.results.values()}
    assert len(keys) == 1 and None not in keys and next(iter(keys)).startswith("flux:")


def test_load_binds_a_family_again_from_the_saved_data_and_prior(x64):
    from whisper_cbpf.results import model_record

    lc = _alert(redshift=0.05)
    model = C._bind_family("arnett", lc, None)

    class _Saved:                                   # what load_result gives back: names only
        parameters = list(model.parameters)
        provenance = {"model": model_record(model)}

    again = C._rebind_family("arnett", lc, _Saved())
    assert again is not None and again is not model and again.parameters == model.parameters
    point = {}
    for nm, d in model.default_prior.distributions.items():
        lo, hi = d.bounds
        point[nm] = lo + 0.3 * (hi - lo)
    t, b = np.asarray(lc.time), np.asarray(lc.band)
    np.testing.assert_array_equal(np.asarray(again.predict(point, t, b)),
                                  np.asarray(model.predict(point, t, b)))
    assert C._rebind_family("no_such_family", lc, _Saved()) is None
    _Saved.parameters = ["other"]                   # not the family's parameters: keep the name
    assert C._rebind_family("arnett", lc, _Saved()) is None
