"""Tests for WAIC (`whisper_cbpf.metrics.waic`) and the pointwise log-likelihood it relies on."""
import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.likelihood import GaussianLikelihood, GaussianLikelihoodWithUpperLimits
from whisper_cbpf.models import get_model

TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}

#: PSIS-LOO is the only part of `predictive_metrics` that needs an optional dependency (the `loo`
#: extra). The documented contract is that it degrades to None rather than failing, so both branches
#: are asserted -- a bare `assert pm["elpd_loo"] is not None` fails the suite on exactly the minimal
#: install the package promises to support, which is what CI runs.
try:
    import arviz  # noqa: F401

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


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


def _censored_lc(n_ul=8, limit=3.0):
    """24 flux points in two bands, ``n_ul`` non-detections carrying NaN ``flux_err``.

    NaN in the error column is what whisper's loader writes for an upper limit, and the limit sits
    far ABOVE the model so a correct fit is maximally penalised if the censored rows get scored.
    """
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 24)
    bands = np.array(["g"] * 12 + ["r"] * 12)
    flux = np.asarray(m.predict(TRUE, t, bands), dtype=float)
    obs = flux + np.random.default_rng(0).normal(0, 0.1, flux.shape)
    err = np.full(24, 0.1)
    ul = np.zeros(24, dtype=bool)
    ul[:: 24 // n_ul] = True
    obs[ul], err[ul] = limit, np.nan
    return wp.LightCurve(time=t, band=bands, flux=obs, flux_err=err, upper_limit=ul, name="censored")


def test_per_band_metrics_survives_a_censored_light_curve():
    """It must not need a full likelihood just to convert flux into the comparison space.

    ``GaussianLikelihood`` refuses a light curve with ``upper_limit=True`` rows (their error bar is
    NaN, which would make the log-likelihood NaN everywhere). Building one here as a space converter
    therefore raised on exactly the data that needs per-band metrics most: measured on a 24-point
    flux curve with 8 non-detections fitted with ``likelihood='upper_limits'``, every result carried
    ``info['band_metrics_error']`` instead of ``info['band_metrics']``.
    """
    lc = _censored_lc()
    with pytest.raises(ValueError, match="censoring"):                 # what the old code did first
        GaussianLikelihood(lc, space="flux")

    pbm = wp.per_band_metrics(lc, "gaussian_rise", TRUE, space="flux")
    assert np.isfinite([pbm["overall"][k] for k in ("mse", "rmse", "mae")]).all()
    for b in ("g", "r"):
        assert np.isfinite([pbm["bands"][b][k] for k in ("mse", "rmse", "mae")]).all()


def test_per_band_metrics_scores_detections_only_and_reports_the_exclusions():
    """A non-detection's ``y`` is a LIMIT, so ``observed - model`` on that row is not a residual.

    Scoring it makes a model that correctly sits BELOW the limit look worse the deeper (the more
    constraining) the limit is — measured here as a 100x-plus inflation of the MSE. Every ``n``
    counts detections, and ``n_upper_limits_excluded`` says how many rows that dropped.
    """
    lc = _censored_lc()
    pbm = wp.per_band_metrics(lc, "gaussian_rise", TRUE, space="flux")
    assert pbm["n_upper_limits_excluded"] == 8
    assert pbm["overall"]["n"] == 16                                   # 24 rows - 8 limits
    assert sum(s["n"] for s in pbm["bands"].values()) == 16

    # Identical to dropping the limits up front, and far below the all-rows number it replaced.
    ref = wp.per_band_metrics(lc.where(upper_limit=False), "gaussian_rise", TRUE, space="flux")
    assert ref["overall"]["mse"] == pytest.approx(pbm["overall"]["mse"])
    assert ref["n_upper_limits_excluded"] == 0
    resid_all = np.asarray(lc.flux, float) - np.asarray(
        get_model("gaussian_rise").predict(TRUE, np.asarray(lc.time, float),
                                           np.asarray(lc.band)), float)
    assert float(np.mean(resid_all ** 2)) > 100 * pbm["overall"]["mse"]


def test_per_band_metrics_keeps_a_band_that_is_entirely_non_detections():
    """``n = 0`` is a result; a band silently vanishing from the report is not."""
    base = _censored_lc()
    ul = np.zeros(24, dtype=bool)
    ul[12:] = True                                                     # all of r
    lc = wp.LightCurve(time=base.time, band=base.band, flux=base.flux,
                       flux_err=np.where(ul, np.nan, 0.1), upper_limit=ul, name="r-blind")
    pbm = wp.per_band_metrics(lc, "gaussian_rise", TRUE, space="flux")
    assert set(pbm["bands"]) == {"g", "r"}
    assert pbm["bands"]["r"]["n"] == 0 and np.isnan(pbm["bands"]["r"]["mse"])
    assert pbm["bands"]["g"]["n"] == 12 and pbm["n_upper_limits_excluded"] == 12


def test_per_band_metrics_is_unchanged_without_upper_limits():
    """No ``upper_limit`` column -> every row scored, and the new key reports zero exclusions."""
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 24)
    flux = m.predict(TRUE, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 24, flux=flux + 0.5,
                       flux_err=np.full_like(flux, 0.1), name="syn")
    pbm = wp.per_band_metrics(lc, "gaussian_rise", TRUE, space="flux")
    assert pbm["n_upper_limits_excluded"] == 0 and pbm["overall"]["n"] == 24
    assert pbm["bands"]["r"]["mae"] == pytest.approx(0.5, abs=1e-6)


def test_flux_to_space_matches_the_bound_method_and_needs_a_resolved_space():
    """The converter `per_band_metrics` calls is the one the likelihood uses, floor included."""
    from whisper_cbpf.likelihood import flux_to_space

    lik = GaussianLikelihood(_censored_lc().where(upper_limit=False), space="magnitude")
    mf = np.array([1e-3, 1.0, 0.0, -5.0])                              # incl. non-positive fluxes
    assert np.allclose(flux_to_space(mf, "magnitude", lik.zeropoint_jy), lik.model_in_space(mf))
    assert np.isfinite(flux_to_space(mf, "magnitude")).all()           # floored, not NaN
    assert np.array_equal(flux_to_space(mf, "flux"), mf)               # identity in flux space
    with pytest.raises(ValueError, match="resolved space"):
        flux_to_space(mf, "mag")                                       # unresolved spelling


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
    # ELPD (PSIS-LOO via arviz) — present with the k diagnostic, or None without the `loo` extra
    if HAS_ARVIZ:
        assert pm["elpd_loo"] is not None
        assert {"elpd_loo", "p_loo", "se", "looic", "pareto_k_max"} <= set(pm["elpd_loo"])
    else:
        assert pm["elpd_loo"] is None, "without arviz elpd_loo must degrade to None, not fail"
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


def _tight_posterior(n=300, seed=3):
    """A posterior clustered tightly around ``TRUE`` — enough to score, cheaper than a real fit."""
    rng = np.random.default_rng(seed)
    names = list(TRUE)
    return pd.DataFrame(
        np.array([[TRUE[k] * (1 + 0.01 * z) for k, z in zip(names, row)]
                  for row in rng.normal(0, 1, size=(n, len(names)))]), columns=names)


def test_predictive_metrics_rmse_and_coverage_score_detections_only():
    """``rmse`` and ``coverage`` compare POINTS, so a limit does not belong in either.

    ``observed - model`` on a non-detection is not a residual — a model correctly below an upper
    limit still contributes ``(limit - model)**2``, so the deeper the limit the worse a correct fit
    looks — and asking whether a bound falls inside a central predictive interval is not a
    calibration question. Both blocks used to run over every row: measured here, the RMSE was 10x+
    its true value. The fix must reproduce the same light curve with the limits dropped up front.
    """
    lc = _censored_lc()
    post = _tight_posterior()
    pm = wp.predictive_metrics(post, lc, model="gaussian_rise", space="flux", n_draws=300, seed=0)
    assert pm["n_upper_limits_excluded"] == 8

    det = np.asarray(lc.upper_limit, dtype=bool) == False        # noqa: E712 - explicit mask
    det_lc = wp.LightCurve(time=np.asarray(lc.time)[det], band=np.asarray(lc.band)[det],
                           flux=np.asarray(lc.flux)[det], flux_err=np.asarray(lc.flux_err)[det],
                           name="det-only")
    ref = wp.predictive_metrics(post, det_lc, model="gaussian_rise", space="flux",
                                n_draws=300, seed=0)
    assert pm["rmse"]["overall"] == pytest.approx(ref["rmse"]["overall"])
    for b in pm["rmse"]["bands"]:
        assert pm["rmse"]["bands"][b] == pytest.approx(ref["rmse"]["bands"][b])
    for a, c in zip(pm["coverage"]["overall"], ref["coverage"]["overall"]):
        assert a["empirical"] == pytest.approx(c["empirical"])

    # The all-rows RMSE this replaced is far larger, which is the whole point.
    resid_all = np.asarray(lc.flux, float) - np.asarray(
        get_model("gaussian_rise").predict(TRUE, np.asarray(lc.time, float),
                                           np.asarray(lc.band)), float)
    assert float(np.sqrt(np.mean(resid_all ** 2))) > 10 * pm["rmse"]["overall"]


def test_predictive_metrics_density_blocks_still_count_the_limits():
    """The asymmetry is deliberate: LPD/WAIC/LOO keep every row, rmse/coverage do not.

    A censored row's contribution to a predictive DENSITY is its survival term
    ``log P(flux < limit)`` — real information that ``log_likelihood_pointwise`` supplies. Dropping
    it would both discard that and make the totals incomparable with a fit whose likelihood counted
    it, so ``lpd`` on the censored curve must NOT equal ``lpd`` with the limits deleted.
    """
    lc = _censored_lc()
    post = _tight_posterior()
    pm = wp.predictive_metrics(post, lc, model="gaussian_rise", space="flux", n_draws=300, seed=0)

    det = ~np.asarray(lc.upper_limit, dtype=bool)
    det_lc = wp.LightCurve(time=np.asarray(lc.time)[det], band=np.asarray(lc.band)[det],
                           flux=np.asarray(lc.flux)[det], flux_err=np.asarray(lc.flux_err)[det],
                           name="det-only")
    ref = wp.predictive_metrics(post, det_lc, model="gaussian_rise", space="flux",
                                n_draws=300, seed=0)
    assert pm["lpd"]["total"] != pytest.approx(ref["lpd"]["total"])
    assert pm["waic"]["n_data"] == lc.n_points                   # every row scored by the density
    assert ref["waic"]["n_data"] == int(det.sum())


def test_predictive_metrics_unchanged_without_upper_limits():
    """No limits -> identical numbers whether the column is absent or present-and-all-False.

    Guards the exclusion against changing the uncensored path — the coverage block draws its
    predictive replications from an RNG, so masking the wrong array would silently reshuffle them.
    """
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 24)
    bands = np.array(["g"] * 12 + ["r"] * 12)
    flux = np.asarray(m.predict(TRUE, t, bands), dtype=float)
    obs = flux + np.random.default_rng(0).normal(0, 0.1, 24)
    post = _tight_posterior()
    kw = dict(model="gaussian_rise", space="flux", n_draws=300, seed=0)

    bare = wp.LightCurve(time=t, band=bands, flux=obs, flux_err=np.full(24, 0.1), name="bare")
    flagged = wp.LightCurve(time=t, band=bands, flux=obs, flux_err=np.full(24, 0.1),
                            upper_limit=np.zeros(24, dtype=bool), name="flagged")
    a, b = wp.predictive_metrics(post, bare, **kw), wp.predictive_metrics(post, flagged, **kw)
    assert a["n_upper_limits_excluded"] == b["n_upper_limits_excluded"] == 0
    assert a["rmse"] == b["rmse"] and a["coverage"] == b["coverage"]
    assert a["lpd"] == b["lpd"]


def test_predictive_metrics_keeps_an_all_censored_band_as_nan():
    """A band with no detection reports NaN in rmse and coverage rather than disappearing."""
    base = _censored_lc()
    ul = np.zeros(24, dtype=bool)
    ul[12:] = True                                               # all of r
    lc = wp.LightCurve(time=base.time, band=base.band, flux=base.flux,
                       flux_err=np.where(ul, np.nan, 0.1), upper_limit=ul, name="r-blind")
    pm = wp.predictive_metrics(_tight_posterior(), lc, model="gaussian_rise", space="flux",
                               n_draws=300, seed=0)
    assert pm["n_upper_limits_excluded"] == 12
    assert set(pm["rmse"]["bands"]) == {"g", "r"} and np.isnan(pm["rmse"]["bands"]["r"])
    assert np.isfinite(pm["rmse"]["bands"]["g"])
    assert set(pm["coverage"]["bands"]) == {"g", "r"}
    assert all(np.isnan(c["empirical"]) for c in pm["coverage"]["bands"]["r"])
    assert all(np.isfinite(c["empirical"]) for c in pm["coverage"]["bands"]["g"])


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
    if HAS_ARVIZ:
        assert sa["elpd_loo"] is not None and sa["elpd_loo"]["pareto_k_max"] < 0.7
    cov95 = next(c["empirical"] for c in sa["coverage"]["overall"] if c["nominal"] == 0.95)
    assert cov95 > 0.85                                   # near-nominal (calibrated)

    no = wp.predictive_metrics(res, lc, space="flux", n_draws=400, scatter_param=None)
    assert no["waic"]["p_waic"] > sa["waic"]["p_waic"] * 100   # mis-specified -> explodes
    cov95_no = next(c["empirical"] for c in no["coverage"]["overall"] if c["nominal"] == 0.95)
    assert cov95_no < cov95                                # under-covers without sigma


# ---------------------------------------------------------------------------------------------
# WAIC pinned to a value. Every other WAIC assertion in this suite is relational or structural --
# key presence, finiteness, `wg["waic"] < wb["waic"]`, the internal identity waic == -2*elpd_waic,
# a loose `p_waic < 50`. All of those pass unchanged if the estimator is off by a constant, uses
# the wrong variance convention, or forgets the 1/S in the lppd. These two pin the number.
# ---------------------------------------------------------------------------------------------

#: parameters of the closed-form fixture: S draws split evenly between theta = -A and theta = +A.
CF_S, CF_A, CF_SD = 200, 0.3, 1.0
CF_Y = np.array([-2.0, -1.3, -0.6, -0.1, 0.4, 0.9, 1.5, 2.0])


def _closed_form_ll():
    """The (CF_S, 8) log-likelihood matrix of the two-point posterior described below."""
    theta = np.repeat([-CF_A, CF_A], CF_S // 2)
    return (-0.5 * np.log(2 * np.pi * CF_SD ** 2)
            - (CF_Y[None, :] - theta[:, None]) ** 2 / (2 * CF_SD ** 2))


def test_waic_matches_closed_form_for_a_two_point_gaussian_posterior():
    r"""WAIC pinned against an exact analytic value, not against itself.

    **Setup.** Data :math:`y_i` observed with known :math:`\sigma`, model :math:`p(y_i\mid\theta) =
    \mathcal N(\theta, \sigma^2)`. The posterior is the two-point measure putting S/2 draws at
    :math:`\theta = -a` and S/2 at :math:`+a`, so every posterior expectation is a finite sum.
    Write :math:`c = -\tfrac12\ln 2\pi\sigma^2`; then
    :math:`\ell_i^{\pm} = c - (y_i \mp a)^2 / 2\sigma^2`.

    **lppd.** :math:`\text{lppd}_i = \ln\frac1S\sum_s e^{\ell_i(\theta_s)}
    = c + \ln\tfrac12\big(e^{-(y_i-a)^2/2\sigma^2} + e^{-(y_i+a)^2/2\sigma^2}\big)`.
    Factor out :math:`e^{-(y_i^2+a^2)/2\sigma^2}`; the bracket becomes
    :math:`\tfrac12(e^{y_i a/\sigma^2} + e^{-y_i a/\sigma^2}) = \cosh(y_i a/\sigma^2)`, so

    .. math:: \text{lppd}_i = c - \frac{y_i^2 + a^2}{2\sigma^2} + \ln\cosh\!\frac{y_i a}{\sigma^2}.

    **p_waic.** :math:`\ell_i^{+} - \ell_i^{-} = 2y_i a/\sigma^2 \equiv \Delta_i`. For a balanced
    two-valued sample of size S the deviations from the mean are all :math:`\pm\Delta_i/2`, so the
    sample variance (Vehtari/Gelman/Gabry 2017 Eq. 11 defines V with the 1/(S-1) divisor) is
    :math:`S\Delta_i^2 / 4(S-1)`, i.e.

    .. math:: p_{\text{waic},i} = \frac{S\,y_i^2 a^2}{(S-1)\,\sigma^4}.

    **WAIC** :math:`= -2\sum_i(\text{lppd}_i - p_{\text{waic},i})`. With S=200, a=0.3, sigma=1 and
    the eight y in ``CF_Y`` that is lppd = -13.779272569482, p_waic = 1.201206030151,
    WAIC = 29.960957199265 -- reproduced to 1e-12 below.
    """
    from whisper_cbpf.metrics._numpy import _loo_waic

    ll = _closed_form_ll()
    _, w, _ = _loo_waic(ll)

    c = -0.5 * np.log(2 * np.pi * CF_SD ** 2)
    lppd_i = c - (CF_Y ** 2 + CF_A ** 2) / (2 * CF_SD ** 2) + np.log(np.cosh(CF_Y * CF_A / CF_SD ** 2))
    p_waic_i = CF_S * CF_Y ** 2 * CF_A ** 2 / ((CF_S - 1) * CF_SD ** 4)

    assert w["lppd"] == pytest.approx(float(lppd_i.sum()), rel=1e-12)
    assert w["p_waic"] == pytest.approx(float(p_waic_i.sum()), rel=1e-12)
    assert w["elpd_waic"] == pytest.approx(float((lppd_i - p_waic_i).sum()), rel=1e-12)
    assert w["waic"] == pytest.approx(float(-2.0 * (lppd_i - p_waic_i).sum()), rel=1e-12)
    # the literal values, so a change of convention cannot slip through by editing both sides
    assert w["lppd"] == pytest.approx(-13.779272569482, abs=1e-10)
    assert w["p_waic"] == pytest.approx(1.201206030151, abs=1e-10)
    assert w["waic"] == pytest.approx(29.960957199265, abs=1e-10)
    assert w["n_data"] == 8 and w["p_waic_reliable"] is True


def test_public_waic_matches_closed_form_end_to_end():
    """The same pinning through the public `wp.waic`, so the model call and the Gaussian
    normalisation are inside the test, not just the WAIC arithmetic.

    Posterior = S/2 draws at parameter set A and S/2 at B, so the (S, n) log-likelihood matrix has
    exactly two distinct rows ``llA``, ``llB`` written out longhand from the Gaussian density.
    Then ``lppd_i = logaddexp(llA_i, llB_i) - ln 2`` and, by the balanced two-point variance above,
    ``p_waic_i = S/(S-1) * ((llA_i - llB_i)/2)**2``. Measured: lppd = 10.960685229720,
    p_waic = 3.655884848458, WAIC = -14.609600762525.
    """
    m = get_model("gaussian_rise")
    t = np.linspace(0.5, 30.0, 25)
    err = np.full(25, 0.15)
    obs = m.predict(TRUE, t, None) + np.random.default_rng(7).normal(0, err)
    lc = wp.LightCurve(time=t, band=["r"] * 25, flux=obs, flux_err=err, name="an")

    A, B = dict(TRUE), {**TRUE, "amplitude": 5.2, "t0": 8.1}
    S = 100
    df = pd.DataFrame([A if i % 2 == 0 else B for i in range(S)])[list(TRUE)]
    got = wp.waic(df, lc, "gaussian_rise", space="flux", max_samples=S)

    def gaussian_ll(p):                                    # -0.5[(o-m)^2/s^2 + ln 2*pi*s^2]
        mf = np.asarray(m.predict(p, t, np.array(["r"] * 25)), float)
        return -0.5 * np.log(2 * np.pi * err ** 2) - (obs - mf) ** 2 / (2 * err ** 2)

    ll_a, ll_b = gaussian_ll(A), gaussian_ll(B)
    lppd_i = np.logaddexp(ll_a, ll_b) - np.log(2.0)
    p_waic_i = S / (S - 1) * ((ll_a - ll_b) / 2.0) ** 2
    elpd_i = lppd_i - p_waic_i

    assert got["n_samples"] == S and got["subsampled"] is False
    assert got["lppd"] == pytest.approx(float(lppd_i.sum()), rel=1e-12)
    assert got["p_waic"] == pytest.approx(float(p_waic_i.sum()), rel=1e-12)
    assert got["waic"] == pytest.approx(float(-2.0 * elpd_i.sum()), rel=1e-12)
    assert got["se"] == pytest.approx(float(np.sqrt(25 * np.var(-2.0 * elpd_i, ddof=1))), rel=1e-12)
    assert got["waic"] == pytest.approx(-14.609600762525, abs=1e-9)


@pytest.mark.skipif(not HAS_ARVIZ or not hasattr(__import__("arviz"), "waic"),
                    reason="cross-check needs arviz with az.waic (removed in arviz 1.0)")
def test_waic_cross_checks_against_arviz_on_the_same_log_likelihood_matrix():
    """`arviz.waic` on the identical (draws, data) matrix, built the way `_loo_waic` builds it.

    The two agree on ``lppd`` exactly and differ on ``p_waic`` by exactly the factor ``(S-1)/S``:
    arviz computes the posterior variance of the pointwise log-likelihood with the **population**
    divisor (``xarray.var`` defaults to ddof=0, arviz/stats/stats.py ``vars_lpd =
    log_likelihood.var(dim="__sample__")``), while Watanabe 2010 / Vehtari, Gelman & Gabry 2017
    Eq. 11 and BDA3 Eq. 7.12 all define V as the **sample** variance with 1/(S-1) -- which is what
    this package uses. Measured on a real 200x40 matrix: arviz p_waic 3.601809979809 against ours
    3.619909527446, i.e. 0.5% at S=200 and O(1/S) in general. The relation below is exact, so it
    pins our numbers to arviz's *and* documents the one deliberate difference.
    """
    import warnings

    import arviz as az
    from whisper_cbpf.metrics._numpy import _idata_from_ll, _loo_waic

    ll = _closed_form_ll()
    _, w, _ = _loo_waic(ll)
    idata = _idata_from_ll(az, ll)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # arviz's own p_waic > 0.4 notice
        az_w = az.waic(idata, pointwise=True)

    az_elpd, az_p = float(az_w.elpd_waic), float(az_w.p_waic)
    assert w["lppd"] == pytest.approx(az_elpd + az_p, rel=1e-12)          # identical lppd
    assert w["p_waic"] == pytest.approx(az_p * CF_S / (CF_S - 1), rel=1e-12)
    assert w["elpd_waic"] == pytest.approx(az_elpd - az_p / (CF_S - 1), rel=1e-12)
    assert w["waic"] == pytest.approx(-2.0 * (az_elpd - az_p / (CF_S - 1)), rel=1e-12)
    # and the two are the same number for practical purposes at this S
    assert w["waic"] == pytest.approx(-2.0 * az_elpd, rel=1e-3)


def test_manual_and_attached_waic_agree_on_a_scatter_fit():
    """`wp.waic(result, lc)` and `result.waic` are the same number on a fit with free scatter.

    They were not. `wp.waic` defaulted to ``likelihood="auto"`` (a plain Gaussian on the reported
    errors, with the fitted ``sigma`` column fed to ``model.predict`` and then ignored) on
    ``max_samples=2000`` draws, while the auto-attached block used the fit's own
    ``gaussian_scatter`` density on 200 draws. Measured on this fixture before the fix:
    ``wp.waic`` gave 310496.50 with ``p_waic = 152995.77`` flagged UNRELIABLE, against the block's
    33.82 with ``p_waic = 4.39`` flagged reliable -- a factor of 9182, with nothing to say which
    was meant. `wp.waic` now adopts the space, likelihood kind, scatter column and draw count the
    block recorded, so the two agree exactly.
    """
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 80)
    clean = m.predict(TRUE, t, None)
    err = np.full_like(clean, 0.02)                        # tiny reported errors
    obs = clean + np.random.default_rng(0).normal(0, np.sqrt(err ** 2 + 0.3 ** 2))
    lc = wp.LightCurve(time=t, band=["r"] * 80, flux=obs, flux_err=err, name="syn")
    prior = wp.Prior({**{k: wp.Uniform(0.5 * v, 1.5 * v) for k, v in TRUE.items()},
                      "sigma": wp.LogUniform(1e-3, 3.0)})
    res = wp.fit_MCMC(lc, "gaussian_rise", prior=prior, nsteps=2500, burnin=800,
                      likelihood="gaussian_scatter", seed=0)

    block = res.info["predictive_metrics"]
    assert block["scatter_param"] == "sigma"
    manual = wp.waic(res, lc)
    assert manual["waic"] == pytest.approx(block["waic"]["waic"], rel=1e-12)
    assert manual["p_waic"] == pytest.approx(block["waic"]["p_waic"], rel=1e-12)
    assert manual["p_waic_reliable"] is block["waic"]["p_waic_reliable"]
    assert manual["scatter_param"] == "sigma"              # followed the fit, not "auto" = Gaussian
    assert manual["n_samples"] == block["n_draws"]
    assert manual["waic"] == pytest.approx(res.waic, rel=1e-12)
    # ...and an explicit argument still overrides the fit: dropping sigma mis-specifies the density
    n_req = block["n_draws_requested"]
    assert len(res.samples) > n_req                        # the agreement survived a subsample
    plain = wp.waic(res, lc, scatter_param=None, max_samples=n_req)
    assert plain["p_waic"] > 100 * manual["p_waic"]


def test_result_exposes_metrics_as_properties_and_never_raises():
    """`result.waic` / `.waic_reliable` / `.elpd_loo` / `.rmse` instead of a four-hop dict walk.

    The block is best-effort, so on a failed fit ``info["predictive_metrics"]`` is ABSENT and the
    naive ``result.info["predictive_metrics"]["waic"]["waic"]`` raises ``KeyError``. The properties
    return ``None`` at every level instead, and stay out of ``to_dict()`` so the JSON is unchanged.
    """
    import json

    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 40)
    flux = m.predict(TRUE, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 40,
                       flux=flux + np.random.default_rng(0).normal(0, 0.1, 40),
                       flux_err=np.full_like(flux, 0.1), name="syn")
    prior = wp.Prior({k: wp.Uniform(0.5 * v, 1.5 * v) for k, v in TRUE.items()})
    res = wp.fit_MCMC(lc, "gaussian_rise", prior=prior, nsteps=1200, burnin=300, seed=0)

    block = res.info["predictive_metrics"]
    assert res.waic == block["waic"]["waic"]
    assert res.waic_reliable == block["waic"]["p_waic_reliable"]
    assert res.rmse == block["rmse"]["overall"]
    assert res.elpd_loo == (block["elpd_loo"]["elpd_loo"] if HAS_ARVIZ else None)
    assert isinstance(res.waic, float) and isinstance(res.rmse, float)

    # they are views, not new serialised fields -- the JSON shape must not have moved
    d = json.loads(res.to_json())
    assert {"waic", "rmse", "elpd_loo", "waic_reliable"}.isdisjoint(d)
    assert d["info"]["predictive_metrics"]["waic"]["waic"] == res.waic

    # a failed / missing block: None everywhere, no KeyError
    res.info.pop("predictive_metrics")
    res.info["predictive_metrics_error"] = "RuntimeError: forward model exploded"
    assert (res.waic, res.waic_reliable, res.elpd_loo, res.rmse) == (None, None, None, None)
    with pytest.raises(KeyError):                          # what users had to write before
        res.info["predictive_metrics"]["waic"]["waic"]


@pytest.mark.skipif(not HAS_ARVIZ, reason="needs arviz installed to make it fail")
def test_loo_waic_distinguishes_arviz_failing_from_arviz_absent(monkeypatch):
    """A broken `az.loo` used to be indistinguishable from no arviz at all: one bare
    `except Exception: pass` returned ``elpd_loo = None`` for both. The reason is now recorded."""
    import arviz as az
    from whisper_cbpf.metrics._numpy import _loo_waic

    # Healthy arviz: no error recorded. On a continuous posterior -- arviz >= 1.0's PSIS refuses the
    # two-valued closed-form matrix below outright ("All tail values are the same"), which is
    # recorded as an error, correctly, and is not what this line tests.
    theta = np.random.default_rng(0).normal(0.0, 0.3, CF_S)
    healthy = (-0.5 * np.log(2 * np.pi * CF_SD ** 2)
               - (CF_Y[None, :] - theta[:, None]) ** 2 / (2 * CF_SD ** 2))
    loo, w, err = _loo_waic(healthy)
    assert loo is not None and err is None
    assert np.isfinite(loo["elpd_loo"]) and np.isfinite(loo["pareto_k_max"])

    ll = _closed_form_ll()

    def boom(*a, **k):
        raise RuntimeError("simulated arviz breakage")

    monkeypatch.setattr(az, "loo", boom)
    loo, w, err = _loo_waic(ll)
    assert loo is None and "simulated arviz breakage" in err and "RuntimeError" in err
    assert w["waic"] == pytest.approx(29.960957199265, abs=1e-10)   # WAIC unaffected by LOO failing


@pytest.mark.skipif(not HAS_ARVIZ, reason="ESS needs arviz")
def test_ess_is_finite_on_every_arviz():
    """``ess_by_parameter`` called ``az.convert_to_inference_data``, which arviz 1.x removed: every
    ESS, and every ESS/sec built on it, was NaN. It now shares the NUTS samplers' routine."""
    from whisper_cbpf.metrics import ess_by_parameter, ess_summary

    x = np.random.default_rng(0).normal(size=(4, 500, 2))
    bulk = ess_by_parameter(x, ["a", "b"])
    tail = ess_by_parameter(x, ["a", "b"], method="tail")
    assert all(1000 < v < 4000 for v in bulk.values()), bulk     # ~2000 independent draws
    assert all(np.isfinite(v) for v in tail.values())
    s = ess_summary(x, ["a", "b"], wall_clock_s=2.0)
    assert np.isfinite(s["ess_worst_per_sec"]) and s["ess_worst_param"] in ("a", "b")
    with pytest.raises(ValueError, match="bulk"):
        ess_by_parameter(x, ["a", "b"], method="quantile")
