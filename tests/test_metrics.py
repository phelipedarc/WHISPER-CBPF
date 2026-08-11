"""Tests for WAIC (`whisper_labia.metrics.waic`) and the pointwise log-likelihood it relies on."""
import numpy as np
import pandas as pd
import pytest

import whisper_labia as wp
from whisper_labia.likelihood import GaussianLikelihood, GaussianLikelihoodWithUpperLimits
from whisper_labia.models import get_model

TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}


def _lc():
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 50)
    flux = m.predict(TRUE, t, None)
    obs = flux + np.random.default_rng(0).normal(0, 0.1, flux.shape)
    return wp.LightCurve(time=t, band=["r"] * 50, flux=obs, flux_err=np.full_like(flux, 0.1), name="syn")


def test_pointwise_sums_to_total_loglik():
    lc = _lc()
    lik = GaussianLikelihood(lc, space="flux")
    mf = get_model("gaussian_rise").predict(TRUE, np.asarray(lc.time), None)
    pw = lik.log_likelihood_pointwise(mf)
    assert pw.shape == (lc.n_points,)
    assert np.isclose(pw.sum(), lik.log_likelihood(mf))


def test_pointwise_upper_limits_sums_to_total():
    """The upper-limit likelihood's pointwise terms also sum to its total."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.1, 30, 20)
    flux = get_model("gaussian_rise").predict(TRUE, t, None)
    ul = np.zeros(20, dtype=bool); ul[::5] = True
    lc = wp.LightCurve(time=t, band=["r"] * 20, flux=flux + rng.normal(0, 0.1, 20),
                       flux_err=np.full(20, 0.1), upper_limit=ul, name="ul")
    lik = GaussianLikelihoodWithUpperLimits(lc, space="flux")
    mf = get_model("gaussian_rise").predict(TRUE, t, None)
    pw = lik.log_likelihood_pointwise(mf)
    assert pw.shape == (20,)
    assert np.isclose(pw.sum(), lik.log_likelihood(mf))


def test_waic_keys_finite_and_ordering():
    """WAIC returns the expected fields; a better-fitting posterior has the lower WAIC."""
    lc = _lc()
    rng = np.random.default_rng(1)
    names = list(TRUE)
    good = pd.DataFrame({n: rng.normal(TRUE[n], 0.02 * abs(TRUE[n]), 300) for n in names})
    bad = pd.DataFrame({n: rng.normal(TRUE[n] * 1.5, 0.02 * abs(TRUE[n]), 300) for n in names})
    wg = wp.waic(good, lc, "gaussian_rise", space="flux", max_samples=300, seed=0)
    wb = wp.waic(bad, lc, "gaussian_rise", space="flux", max_samples=300, seed=0)
    assert set(wg) >= {"waic", "lppd", "p_waic", "se", "n_samples", "n_data"}
    assert np.isfinite(wg["waic"]) and wg["p_waic"] > 0 and wg["n_data"] == lc.n_points
    assert wg["waic"] < wb["waic"]


def test_waic_fixed_parameters_and_subsampling():
    """`fixed=` supplies params absent from the posterior columns; `max_samples` caps the draws."""
    lc = _lc()
    rng = np.random.default_rng(2)
    df = pd.DataFrame({n: rng.normal(TRUE[n], 0.02 * abs(TRUE[n]), 500)
                       for n in ["amplitude", "t0", "sigma_rise"]})       # tau_decay omitted
    w = wp.waic(df, lc, "gaussian_rise", space="flux", fixed={"tau_decay": 15.0},
                max_samples=120, seed=0)
    assert np.isfinite(w["waic"]) and w["n_samples"] == 120


def test_per_band_metrics_zero_at_truth_and_keys():
    """At the exact truth the residuals vanish (MSE/MAE ~ 0); output has per-band + overall stats."""
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 40)
    times = np.concatenate([t, t])
    bands = np.array(["g"] * 40 + ["r"] * 40)
    flux = m.predict(TRUE, times, bands)
    lc = wp.LightCurve(time=times, band=bands, flux=flux,
                       flux_err=np.full_like(flux, 0.1), name="syn")
    pbm = wp.per_band_metrics(lc, "gaussian_rise", TRUE, space="flux")
    assert pbm["space"] == "flux" and pbm["unit"] == "Jy"
    assert set(pbm["bands"]) == {"g", "r"}
    for b in ("g", "r"):
        assert pbm["bands"][b]["n"] == 40
        assert pbm["bands"][b]["mse"] < 1e-12 and pbm["bands"][b]["mae"] < 1e-6
        assert pbm["bands"][b]["rmse"] == pytest.approx(pbm["bands"][b]["mse"] ** 0.5)
    assert pbm["overall"]["n"] == 80


def test_per_band_metrics_detects_offset():
    """A constant flux offset raises MAE by ~that offset (per band and overall)."""
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 40)
    flux = m.predict(TRUE, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux + 0.5,   # data 0.5 Jy brighter than model
                       flux_err=np.full_like(flux, 0.1), name="syn")
    pbm = wp.per_band_metrics(lc, "gaussian_rise", TRUE, space="flux")
    assert pbm["bands"]["r"]["mae"] == pytest.approx(0.5, abs=1e-6)


def test_fit_reports_band_metrics_in_json():
    """Every sampler attaches info['band_metrics'], so it lands in to_json/to_dict."""
    import json
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 40)
    flux = m.predict(TRUE, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux + np.random.default_rng(0).normal(0, 0.1, 40),
                       flux_err=np.full_like(flux, 0.1), name="syn")
    prior = wp.Prior({k: wp.Uniform(0.5 * v, 1.5 * v) for k, v in TRUE.items()})
    res = wp.fit_ABC(lc, "gaussian_rise", prior=prior, n_simulations=2000, quantile=0.05,
                     n_jobs=1, seed=0)
    assert "band_metrics" in res.info
    bm = json.loads(res.to_json())["info"]["band_metrics"]
    assert "r" in bm["bands"] and bm["bands"]["r"]["n"] == 40


def test_predictive_metrics_block_and_autoattach():
    """predictive_metrics returns RMSE/LPD/ELPD-LOO/WAIC/AIC/BIC/coverage; samplers auto-attach it."""
    import json
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 40)
    times = np.concatenate([t, t])
    bands = np.array(["g"] * 40 + ["r"] * 40)
    flux = m.predict(TRUE, times, bands)
    lc = wp.LightCurve(time=times, band=bands, flux=flux + np.random.default_rng(0).normal(0, 0.1, 80),
                       flux_err=np.full_like(flux, 0.1), name="syn")
    prior = wp.Prior({k: wp.Uniform(0.5 * v, 1.5 * v) for k, v in TRUE.items()})
    res = wp.fit_MCMC(lc, "gaussian_rise", prior=prior, nsteps=1200, burnin=300, seed=0)

    pm = wp.predictive_metrics(res, lc, space="flux", n_draws=300)
    # RMSE per band + overall
    assert set(pm["rmse"]["bands"]) == {"g", "r"} and np.isfinite(pm["rmse"]["overall"])
    # LPD
    assert np.isfinite(pm["lpd"]["total"]) and np.isfinite(pm["lpd"]["per_point"])
    # ELPD (PSIS-LOO via arviz) — present with the k diagnostic
    assert pm["elpd_loo"] is not None
    assert {"elpd_loo", "p_loo", "se", "looic", "pareto_k_max"} <= set(pm["elpd_loo"])
    # WAIC deviance vs elpd sign convention; AIC/BIC carried through
    assert pm["waic"]["waic"] == pytest.approx(-2.0 * pm["waic"]["elpd_waic"], rel=1e-6)
    assert np.isfinite(pm["aic"]) and np.isfinite(pm["bic"])
    # coverage-calibration curve at all requested levels, empirical in [0,1]
    levels = [c["nominal"] for c in pm["coverage"]["overall"]]
    assert levels == [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    assert all(0.0 <= c["empirical"] <= 1.0 for c in pm["coverage"]["overall"])

    # auto-attached to the fit's JSON
    info = json.loads(res.to_json())["info"]
    assert "predictive_metrics" in info and "coverage" in info["predictive_metrics"]


def test_predictive_metrics_coverage_calibrated_for_good_fit():
    """A correct model + Gaussian noise gives empirical coverage close to nominal at each level."""
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 120)
    flux = m.predict(TRUE, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 120, flux=flux + np.random.default_rng(1).normal(0, 0.1, 120),
                       flux_err=np.full_like(flux, 0.1), name="syn")
    prior = wp.Prior({k: wp.Uniform(0.5 * v, 1.5 * v) for k, v in TRUE.items()})
    res = wp.fit_MCMC(lc, "gaussian_rise", prior=prior, nsteps=2000, burnin=500, seed=0)
    pm = wp.predictive_metrics(res, lc, space="flux", n_draws=400)
    for c in pm["coverage"]["overall"]:                    # empirical within 0.15 of nominal
        assert abs(c["empirical"] - c["nominal"]) < 0.15


def test_predictive_metrics_scatter_aware():
    """With a fitted extra-scatter sigma, the scatter-augmented predictive gives sane WAIC/p_waic and
    near-nominal coverage; dropping sigma mis-specifies it (p_waic explodes, coverage collapses)."""
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 80)
    clean = m.predict(TRUE, t, None)
    rng = np.random.default_rng(0)
    err = np.full_like(clean, 0.02)                       # tiny reported errors
    obs = clean + rng.normal(0, np.sqrt(err ** 2 + 0.3 ** 2))   # true intrinsic scatter 0.3
    lc = wp.LightCurve(time=t, band=["r"] * 80, flux=obs, flux_err=err, name="syn")
    prior = wp.Prior({**{k: wp.Uniform(0.5 * v, 1.5 * v) for k, v in TRUE.items()},
                      "sigma": wp.LogUniform(1e-3, 3.0)})
    res = wp.fit_MCMC(lc, "gaussian_rise", prior=prior, nsteps=2500, burnin=800,
                      likelihood="gaussian_scatter", seed=0)

    sa = wp.predictive_metrics(res, lc, space="flux", n_draws=400)          # auto -> scatter-aware
    assert sa["scatter_param"] == "sigma"
    assert sa["waic"]["p_waic"] < 50                      # sane effective #params (truth ~5)
    assert sa["elpd_loo"] is not None and sa["elpd_loo"]["pareto_k_max"] < 0.7
    cov95 = next(c["empirical"] for c in sa["coverage"]["overall"] if c["nominal"] == 0.95)
    assert cov95 > 0.85                                   # near-nominal (calibrated)

    no = wp.predictive_metrics(res, lc, space="flux", n_draws=400, scatter_param=None)
    assert no["waic"]["p_waic"] > sa["waic"]["p_waic"] * 100   # mis-specified -> explodes
    cov95_no = next(c["empirical"] for c in no["coverage"]["overall"] if c["nominal"] == 0.95)
    assert cov95_no < cov95                                # under-covers without sigma
