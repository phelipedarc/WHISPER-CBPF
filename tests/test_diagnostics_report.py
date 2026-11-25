"""One convergence report for every sampler: each check, pass or fail, and why.

Chain results are built from the samplers' own health functions (``_diagnostics.chain_health`` /
``walker_health``) on synthetic chains, so the report is checked against exactly what the samplers
record; a few real fits check the wiring.
"""
from __future__ import annotations

import json
import warnings
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf import results as R
from whisper_cbpf.models.flare import flare_flux
from whisper_cbpf.priors import LogUniform, Prior, Uniform
from whisper_cbpf.samplers.base import SamplerResult
from whisper_cbpf.samplers.jax import _diagnostics as dg

NAMES = ["a", "b"]


def _result(sampler, samples, info, *, chains=None, n_data=40, n_params=2, max_ll=0.0,
            prior=None):
    res = SamplerResult(sampler=sampler, model="toy", parameters=list(samples.columns),
                        samples=samples, summary={}, best_params={}, n_data=n_data,
                        n_params=n_params, runtime_s=1.0, info=info, max_log_likelihood=max_ll)
    if chains is not None:
        res.samples_by_chain = chains
    if prior is not None:
        res.provenance = {"model": {"prior": R.prior_record(prior)}}
    return res


def _nuts(shift_chain=None, n_divergences=0, scan_gap=0.5, seed=0):
    """A 4-chain NUTS-like result on a 2-D standard normal, with the info nuts_gpu records."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 500, 2))
    if shift_chain is not None:
        x[shift_chain] += 8.0                              # a chain in another, worse, optimum
    ll = -0.5 * (x ** 2).sum(-1)
    info = dg.chain_health(x, NAMES, ll, n_divergences=n_divergences,
                           reference_ll=float(ll.max()) + scan_gap)
    info.update(num_chains=4, n_divergences=n_divergences,
                prior_scan={"best_log_likelihood": float(ll.max()) + scan_gap})
    return _result("nuts_gpu", pd.DataFrame(x.reshape(-1, 2), columns=NAMES), info, chains=x,
                   max_ll=float(ll.max()))


def _failed(report):
    return [r.check for r in report.rows if r.passed is False]


# ------------------------------------------------------------------------------------------ NUTS
def test_a_healthy_nuts_run_passes_every_check():
    pytest.importorskip("arviz")        # R-hat and tail ESS need it (the [analysis] extra)
    report = _nuts().diagnostics()
    assert report.passed and report.reasons == []
    assert [r.check for r in report.rows] == [
        "posterior draws", "data points", "divergences", "split R-hat (largest)",
        "bulk ESS (smallest)", "tail ESS (smallest)", "log-likelihood R-hat", "stranded chains",
        "frozen chains", "gap to the prior scan's optimum", "prior-edge pile-up"]
    rhat = next(r for r in report.rows if r.check == "split R-hat (largest)")
    assert rhat.value == pytest.approx(_nuts().info["max_rhat"])
    assert next(r for r in report.rows if r.check == "bulk ESS (smallest)").threshold == ">= 400"


def test_a_stranded_chain_fails_with_its_reason():
    report = _nuts(shift_chain=3).diagnostics()
    assert not report.passed
    assert {"split R-hat (largest)", "log-likelihood R-hat", "stranded chains"} <= set(_failed(report))
    stranded = next(r for r in report.rows if r.check == "stranded chains")
    assert stranded.value == 1 and "chain(s) 3" in stranded.reason and "local" in stranded.reason


def test_divergences_fail_and_unrecorded_divergences_fail_closed():
    div = next(r for r in _nuts(n_divergences=5).diagnostics().rows if r.check == "divergences")
    assert div.passed is False and div.value == 5 and "divergent" in div.reason
    res = _nuts()
    res.info["n_divergences"] = -1
    div = next(r for r in res.diagnostics().rows if r.check == "divergences")
    assert div.passed is False and "cannot be ruled out" in div.reason


def test_chains_that_all_missed_the_scan_optimum_fail():
    report = _nuts(scan_gap=25.0).diagnostics()
    gap = next(r for r in report.rows if r.check == "gap to the prior scan's optimum")
    assert gap.passed is False and gap.value == pytest.approx(25.0)
    assert "R-hat cannot see" in gap.reason


@pytest.mark.parametrize("case", [{}, {"shift_chain": 1}, {"n_divergences": 2},
                                  {"scan_gap": 30.0}, {"seed": 4}])
def test_the_report_agrees_with_the_samplers_own_verdict(case):
    res = _nuts(**case)
    assert res.diagnostics().passed == res.info["converged"]


# -------------------------------------------------------------------------------------- ensembles
def _ensemble(stuck=False, nsteps=5000, tau=40.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((32, 400, 2))
    logp = -0.5 * (x ** 2).sum(-1)
    if stuck:
        x[5] += 6.0
        logp[5] -= 30.0
    info = dg.walker_health(logp, [tau, tau / 2], nsteps)
    info.update(nwalkers=32, nsteps=nsteps, burnin=1000, thin=10)
    return _result("emcee_jax", pd.DataFrame(x.reshape(-1, 2), columns=NAMES), info, chains=x)


def test_a_healthy_ensemble_passes():
    report = _ensemble().diagnostics()
    assert report.passed, report
    ratio = next(r for r in report.rows if r.check == "chain length / autocorrelation time")
    assert ratio.value == pytest.approx(5000 / 40.0)          # from the LARGEST tau, not the mean
    rhat = next(r for r in report.rows if r.check.startswith("split R-hat across walkers"))
    assert rhat.threshold == f"< {R.ENSEMBLE_RHAT_MAX}"


def test_a_stuck_walker_and_a_short_chain_fail():
    report = _ensemble(stuck=True, nsteps=1000).diagnostics()
    assert {"stuck walkers", "chain length / autocorrelation time"} <= set(_failed(report))
    if find_spec("arviz") is not None:  # R-hat needs it (the [analysis] extra); else not checked
        assert "split R-hat across walkers (largest)" in _failed(report)
    ratio = next(r for r in report.rows if r.check == "chain length / autocorrelation time")
    assert "Raise nsteps to at least 2000" in ratio.reason
    assert "1 of 32 walkers" in next(r for r in report.rows if r.check == "stuck walkers").reason


def test_rhat_that_cannot_be_computed_is_not_a_failure_for_an_ensemble(monkeypatch):
    def unavailable(x, names, ll=None):
        nan = {n: float("nan") for n in names}
        return {"rhat": nan, "ess_bulk": nan, "ess_tail": nan, "rhat_log_likelihood": float("nan"),
                "rhat_method": "unavailable", "rhat_error": "arviz: missing"}
    monkeypatch.setattr(dg, "rank_diagnostics", unavailable)
    report = _ensemble().diagnostics()
    rhat = next(r for r in report.rows if r.check.startswith("split R-hat"))
    assert rhat.passed is None and "[analysis]" in rhat.reason
    assert report.passed


def test_cpu_mcmc_report_uses_its_walkers():
    t = np.linspace(0.5, 30.0, 40)
    flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit(lc, "flare", sampler="mcmc", nsteps=600, burnin=200, seed=0)
    report = res.diagnostics()
    assert report.kind == "ensemble" and not report.passed
    ratio = next(r for r in report.rows if r.check == "chain length / autocorrelation time")
    assert ratio.value == pytest.approx(600 / res.info["max_autocorr_time"])
    assert ratio.passed is False
    assert (next(r for r in report.rows if r.check == "stuck walkers").passed
            is (not res.info["stuck_walkers"]))
    edge = next(r for r in report.rows if r.check == "prior-edge pile-up")
    assert edge.passed is True                             # the prior recorded with the fit


# ----------------------------------------------------------------------- samplers without chains
def test_abc_without_accepted_draws_says_there_is_no_posterior():
    empty = pd.DataFrame({"a": [], "b": []})
    report = _result("abc", empty, {"n_accepted": 0}).diagnostics()
    assert set(_failed(report)) == {"posterior draws", "accepted draws"}
    accepted = next(r for r in report.rows if r.check == "accepted draws")
    assert "no posterior" in accepted.reason and "closest REJECTED" in accepted.reason
    few = _result("abc", pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0)}),
                  {"n_accepted": 20}).diagnostics()
    assert _failed(few) == ["accepted draws"] and "only 20" in few.reasons[0]


def test_more_parameters_than_points_is_not_enough_data():
    samples = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0)})
    report = _result("abc", samples, {"n_accepted": 200}, n_data=2, n_params=2).diagnostics()
    assert _failed(report) == ["data points"]
    assert report.reasons[0].startswith("data points: not enough data")


def test_nested_snpe_and_smc_rows():
    s = pd.DataFrame({"a": np.random.default_rng(0).random(500), "b": np.zeros(500)})
    bad = _result("nested", s, {"log_evidence": -1.0, "converged": False, "n_effective": 50.0})
    assert set(_failed(bad.diagnostics())) == {"stopped on dlogz", "effective sample size"}
    good = _result("nested", s, {"log_evidence": -1.0, "converged": True, "n_effective": 400.0})
    assert good.diagnostics().passed
    snpe = _result("snpe", s, {"converged": False, "leakage": True, "x_o_min_rms_z": 3.2})
    reason = snpe.diagnostics().reasons[0]
    assert "fell back to MCMC" in reason and "leakage=True" in reason and "3.2 sigma" in reason
    few = pd.DataFrame({"a": np.repeat(np.arange(10.0), 50), "b": np.zeros(500)})
    smc = _result("abc_smc", few, {"n_particles": 500}).diagnostics()
    assert _failed(smc) == ["distinct particles"]


def test_an_unknown_sampler_gets_its_own_flag_and_the_common_checks():
    s = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    res = _result("my_sampler", s, {"converged": False, "convergence_problems": ["it wandered"]})
    report = res.diagnostics()
    assert report.kind == "other" and report.reasons == ["sampler's own convergence flag: it wandered"]


# ---------------------------------------------------------------------------------- prior edges
def test_draws_piled_on_a_bound_fail_and_name_the_parameter():
    rng = np.random.default_rng(1)
    prior = Prior({"a": Uniform(0.0, 10.0), "b": Uniform(0.0, 1.0)})
    s = pd.DataFrame({"a": 10.0 - np.abs(rng.normal(0, 0.3, 2000)), "b": rng.random(2000)})
    report = _result("abc", s, {"n_accepted": 2000}, prior=prior).diagnostics()
    edge = next(r for r in report.rows if r.check == "prior-edge pile-up")
    assert edge.passed is False and "'a' at its upper bound (10)" in edge.reason
    assert "'b'" not in edge.reason                        # a flat posterior is not a pile-up


def test_loguniform_edges_are_judged_in_log_space():
    rng = np.random.default_rng(2)
    prior = Prior({"a": LogUniform(1e-3, 10.0), "b": Uniform(0.0, 1.0)})
    s = pd.DataFrame({"a": 10 ** rng.uniform(-3, 1, 5000), "b": rng.random(5000)})
    res = _result("abc", s, {"n_accepted": 5000}, prior=prior)
    assert res.diagnostics().passed                         # 1% per edge in log10, as the prior
    linear = Prior({"a": Uniform(1e-3, 10.0), "b": Uniform(0.0, 1.0)})
    assert not res.diagnostics(prior=linear).passed         # an explicit prior= wins


def test_without_a_recorded_prior_the_edge_check_says_so():
    s = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0)})
    edge = next(r for r in _result("abc", s, {"n_accepted": 200}).diagnostics().rows
                if r.check == "prior-edge pile-up")
    assert edge.passed is None and "pass prior=" in edge.reason


# -------------------------------------------------------------------------- likelihood maximum
def test_a_likelihood_maximum_far_above_the_best_draw_fails():
    class Peak:
        max_log_likelihood = 7.0
    res = _nuts()
    res.max_log_likelihood = 0.0
    check = "gap to the likelihood maximum"
    far = next(r for r in res.diagnostics(likelihood_max_opt=Peak()).rows if r.check == check)
    assert far.passed is False and far.value == pytest.approx(7.0)
    near = next(r for r in res.diagnostics(likelihood_max_opt=1.0).rows if r.check == check)
    assert near.passed is True


# ------------------------------------------------------------------------------------ the report
def test_repr_table_reasons_and_json():
    report = _nuts(shift_chain=2).diagnostics()
    text = repr(report)
    assert text.startswith("Diagnostics of the nuts_gpu fit of 'toy': FAILED")
    assert "check" in text.splitlines()[1] and "threshold" in text.splitlines()[1]
    assert "Why:" in text and "report.rows" in text
    assert all(r.split(":")[0] in _failed(report) for r in report.reasons)
    d = json.loads(json.dumps(report.to_dict()))
    assert d["passed"] is False and len(d["rows"]) == len(report.rows)


def test_the_report_is_the_same_after_save_and_load(tmp_path):
    res = _nuts(shift_chain=1)
    res.provenance = {"model": {"prior": R.prior_record(Prior({"a": Uniform(-20, 20),
                                                               "b": Uniform(-20, 20)}))}}
    back = wp.load_result(res.save(tmp_path / "r"))
    assert back.diagnostics().to_dict() == res.diagnostics().to_dict()
    assert wp.DiagnosticsReport is R.DiagnosticsReport


@pytest.mark.slow
def test_real_nuts_and_emcee_jax_fits_report_what_they_recorded():
    pytest.importorskip("numpyro")
    t = np.linspace(0.5, 30.0, 40)
    lc = wp.LightCurve(time=t, band=["r"] * 40, flux=np.exp(-0.5 * ((t - 12) / 3) ** 2) + 0.01,
                       flux_err=np.full(40, 0.05))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nuts = wp.fit(lc, "flare_jax", sampler="nuts_gpu", num_warmup=300, num_samples=300,
                      num_chains=2, seed=0)
        ens = wp.fit(lc, "flare_jax", sampler="emcee_jax", nwalkers=16, nsteps=400, burnin=100,
                     seed=0)
    rep = nuts.diagnostics()
    rhat = next(r for r in rep.rows if r.check == "split R-hat (largest)")
    assert rhat.value == pytest.approx(nuts.info["max_rhat"])
    chain_checks = [r for r in rep.rows if r.check not in ("prior-edge pile-up", "data points")]
    assert all(r.passed is not False for r in chain_checks) == nuts.info["converged"]
    erep = ens.diagnostics()
    assert erep.kind == "ensemble"
    assert (next(r for r in erep.rows if r.check == "stuck walkers").passed
            is (not ens.info["stuck_walkers"]))
