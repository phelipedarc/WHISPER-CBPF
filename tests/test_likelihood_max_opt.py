"""``likelihood_max_opt``: the peak of a fit's own likelihood, behind AIC and BIC.

A sampler's best draw is a lower bound on the maximum likelihood, short by a different amount for
each model, so a BIC ranking built on it carries sampler noise. These tests pin down what ``likelihood_max_opt`` promises:

* it reaches a peak known in closed form (a noiseless Gaussian bump, a weighted straight line);
* it reaches an optimum that sits ON a prior edge exactly (box coordinates, not logit);
* a LogUniform parameter is climbed, and judged at an edge, in log10;
* it never returns less than the sampler's best draw, even when the optimiser misbehaves;
* it starts from an earlier peak kept in ``result.info["likelihood_max_opt"]``;
* on redback ``arnett`` fitted to SN2025pgp's 30-day cut it climbs above the chain's best, respects
  the constraint wall, and agrees with an earlier independent likelihood maximum like for like.
"""
from __future__ import annotations

import io
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.likelihood import make_likelihood
from whisper_cbpf.samplers.base import SamplerResult

# wp.likelihood_max_opt is the function, not the module
lmo_mod = sys.modules["whisper_cbpf.likelihood_max_opt"]


# --- toy models ---------------------------------------------------------------------------------
def _bump(parameters, times, bands=None):
    t = np.asarray(times, dtype=float)
    return parameters["A"] * np.exp(-0.5 * ((t - parameters["t0"]) / parameters["w"]) ** 2)


def _line(parameters, times, bands=None):
    return parameters["a"] + parameters["b"] * np.asarray(times, dtype=float)


def _decay(parameters, times, bands=None):
    return parameters["A"] * np.exp(-np.asarray(times, dtype=float) / parameters["tau"])


BUMP_PRIOR = wp.Prior({"A": wp.Uniform(0.1, 10.0), "t0": wp.Uniform(0.0, 20.0),
                       "w": wp.Uniform(0.5, 10.0)})
LINE_PRIOR = wp.Prior({"a": wp.Uniform(-10.0, 10.0), "b": wp.Uniform(-1.0, 1.0)})
DECAY_PRIOR = wp.Prior({"A": wp.LogUniform(1e-3, 1e3), "tau": wp.Uniform(1.0, 50.0)})

wp.register_model("_lmax_bump", _bump, ["A", "t0", "w"], prior=BUMP_PRIOR, overwrite=True)
wp.register_model("_lmax_line", _line, ["a", "b"], prior=LINE_PRIOR, overwrite=True)
wp.register_model("_lmax_decay", _decay, ["A", "tau"], prior=DECAY_PRIOR, overwrite=True)


def _lc(model, truth, n=25, sigma=0.05, noise_seed=None, t=None):
    t = np.linspace(0.5, 20.0, n) if t is None else np.asarray(t, dtype=float)
    y = np.asarray(wp.get_model(model).predict(truth, t, None), dtype=float)
    if noise_seed is not None:
        y = y + np.random.default_rng(noise_seed).normal(0.0, sigma, y.size)
    return wp.LightCurve(time=t, band=["r"] * t.size, flux=y, flux_err=np.full(t.size, sigma))


def _result(model, lc, draws, likelihood="GaussianLikelihood", info=None):
    """A SamplerResult whose draws are ``draws`` and whose best draw is their argmax, scored the
    way the samplers score it (the fit's likelihood through ``model.predict``)."""
    m = wp.get_model(model)
    names = list(draws.columns)
    lik = make_likelihood(lc, space="flux")
    t, b = np.asarray(lc.time, float), np.asarray(lc.band)
    ll = np.array([lik.log_likelihood(m.predict(r, t, b)) for r in draws.to_dict("records")])
    i = int(np.argmax(ll))
    return SamplerResult(
        sampler="test", model=m.name, parameters=names, samples=draws, summary={},
        best_params={nm: float(draws[nm].iloc[i]) for nm in names}, n_data=lc.n_points,
        n_params=len(names), runtime_s=0.0,
        info={"space": "flux", "likelihood": likelihood, **(info or {})},
        max_log_likelihood=float(ll[i]))


def _draws(center, spread, n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: v + s * rng.standard_normal(n)
                         for (k, v), s in zip(center.items(), spread)})


def _log_norm(lc):
    return float(-0.5 * np.sum(np.log(2 * np.pi) + 2 * np.log(np.asarray(lc.flux_err, float))))


def _wls(lc):
    """Closed-form weighted least squares of a straight line: (a, b), ln L_max."""
    t, y, s = (np.asarray(v, float) for v in (lc.time, lc.flux, lc.flux_err))
    X, w = np.column_stack([np.ones_like(t), t]), 1.0 / s ** 2
    beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    return beta, -0.5 * float(np.sum(w * (y - X @ beta) ** 2)) + _log_norm(lc)


# --- peaks known in closed form -----------------------------------------------------------------
def test_noiseless_bump_reaches_the_truth_and_the_normalisation():
    """Noiseless data: the peak is the truth and ln L_max is the Gaussian normalisation."""
    truth = {"A": 3.0, "t0": 9.0, "w": 2.5}
    lc = _lc("_lmax_bump", truth)
    res = _result("_lmax_bump", lc, _draws(truth, [0.3, 0.8, 0.4], 200))
    assert res.max_log_likelihood < _log_norm(lc) - 1.0            # the sampler stopped short

    pr = wp.likelihood_max_opt(res, lc)
    assert isinstance(pr, wp.LikelihoodMaxOptResult)
    for k, v in truth.items():
        assert pr.params[k] == pytest.approx(v, rel=1e-6)
    assert pr.max_log_likelihood == pytest.approx(_log_norm(lc), abs=1e-8)
    assert pr.start_log_likelihood == res.max_log_likelihood
    assert pr.gain == pytest.approx(pr.max_log_likelihood - res.max_log_likelihood)
    assert pr.at_edge == [] and pr.n_data == 25 and pr.n_params == 3
    aic, bic = -2 * pr.max_log_likelihood + 6, -2 * pr.max_log_likelihood + 3 * np.log(25)
    assert (pr.aic, pr.bic) == (pytest.approx(aic), pytest.approx(bic))
    assert pr.n_evals > 0 and pr.runtime_s > 0 and "scipy" in pr.method


def test_noisy_line_reaches_the_weighted_least_squares_peak():
    lc = _lc("_lmax_line", {"a": 1.0, "b": 0.2}, noise_seed=3, sigma=0.1)
    (a, b), ll_max = _wls(lc)
    res = _result("_lmax_line", lc, _draws({"a": 1.2, "b": 0.19}, [0.1, 0.01], 100))
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.params["a"] == pytest.approx(a, abs=1e-6)
    assert pr.params["b"] == pytest.approx(b, abs=1e-7)
    assert pr.max_log_likelihood == pytest.approx(ll_max, abs=1e-9)


def test_a_real_fit_is_optimised_to_the_peak_and_the_posterior_is_untouched():
    truth = {"A": 3.0, "t0": 9.0, "w": 2.5}
    lc = _lc("_lmax_bump", truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit_ABC(lc, "_lmax_bump", n_simulations=3000, seed=1)
    before = res.samples.copy(), dict(res.summary), res.bic
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.max_log_likelihood == pytest.approx(_log_norm(lc), abs=1e-8)
    assert pr.max_log_likelihood >= res.max_log_likelihood and pr.bic <= res.bic
    pd.testing.assert_frame_equal(res.samples, before[0])
    assert res.summary == before[1] and res.bic == before[2]
    assert "likelihood_max_opt" not in res.info


# --- edges and log coordinates ------------------------------------------------------------------
def test_an_optimum_on_a_prior_edge_is_reached_exactly():
    """The slope's box stops below the least-squares slope, so the constrained peak sits on the
    face b = 0.15 with a at its conditional optimum. A logit map cannot reach that face (S14)."""
    lc = _lc("_lmax_line", {"a": 1.0, "b": 0.2}, noise_seed=3, sigma=0.1)
    hi = 0.15
    prior = wp.Prior({"a": wp.Uniform(-10.0, 10.0), "b": wp.Uniform(-1.0, hi)})
    (_, b_free), _ = _wls(lc)
    assert b_free > hi + 0.02                                     # the free peak is outside the box
    t, y, s = (np.asarray(v, float) for v in (lc.time, lc.flux, lc.flux_err))
    w = 1.0 / s ** 2
    a_edge = float(np.sum(w * (y - hi * t)) / np.sum(w))
    ll_edge = -0.5 * float(np.sum(w * (y - a_edge - hi * t) ** 2)) + _log_norm(lc)

    res = _result("_lmax_line", lc, _draws({"a": 1.5, "b": 0.1}, [0.2, 0.01], 100))
    pr = wp.likelihood_max_opt(res, lc, prior=prior)
    assert pr.params["b"] == hi                                   # exactly on the face
    assert pr.params["a"] == pytest.approx(a_edge, abs=1e-6)
    assert pr.max_log_likelihood == pytest.approx(ll_edge, abs=1e-9)
    assert pr.at_edge == ["b"]


def test_log_uniform_parameters_are_climbed_and_judged_in_log10():
    """A = 0.9 in LogUniform(1e-3, 1e3) is mid-box in log10 (u = 0.49) but within 0.1 % of the
    lower bound in linear units (u = 9e-4). It must be reached, and not reported at an edge."""
    truth = {"A": 0.9, "tau": 6.0}
    lc = _lc("_lmax_decay", truth, sigma=0.01)
    res = _result("_lmax_decay", lc, _draws({"A": 0.6, "tau": 8.0}, [0.05, 0.5], 100))
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.params["A"] == pytest.approx(0.9, rel=1e-6)
    assert pr.params["tau"] == pytest.approx(6.0, rel=1e-6)
    assert pr.at_edge == []

    box = lmo_mod._Box(DECAY_PRIOR, ["A", "tau"])
    assert box.to_u(np.array([[1.0, 25.5]]))[0] == pytest.approx([0.5, 0.5])   # geometric mid
    assert box.to_x(np.array([[1.0, 1.0]]))[0].tolist() == [1e3, 50.0]          # faces exact


# --- never below the best draw ------------------------------------------------------------------
def test_never_below_the_best_draw_even_when_the_optimiser_misbehaves(monkeypatch):
    """Every optimiser call is replaced by one that probes a bad corner and claims success. The
    peak must then be the best candidate, never the optimiser's claim."""
    import scipy.optimize
    from scipy.optimize import OptimizeResult

    def sabotage(fun, x0, *args, **kwargs):
        bad = np.zeros_like(np.asarray(x0, dtype=float))
        out = fun(bad)
        return OptimizeResult(x=bad, fun=out[0] if isinstance(out, tuple) else out,
                              success=True, nfev=1)

    monkeypatch.setattr(scipy.optimize, "minimize", sabotage)
    truth = {"A": 3.0, "t0": 9.0, "w": 2.5}
    lc = _lc("_lmax_bump", truth)
    res = _result("_lmax_bump", lc, _draws(truth, [0.3, 0.8, 0.4], 50))
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.max_log_likelihood == res.max_log_likelihood
    assert pr.params == res.best_params and pr.gain == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_never_below_the_best_draw_on_real_abc_fits(seed):
    lc = _lc("_lmax_bump", {"A": 3.0, "t0": 9.0, "w": 2.5}, noise_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit_ABC(lc, "_lmax_bump", n_simulations=1000, seed=seed)
    pr = wp.likelihood_max_opt(res, lc, n_starts=2)
    assert pr.start_log_likelihood == res.max_log_likelihood
    assert pr.max_log_likelihood >= res.max_log_likelihood and pr.gain >= 0.0


def test_an_earlier_peak_in_info_is_a_start(monkeypatch):
    """With the climbs switched off, the answer can only be a candidate: the earlier peak (kept
    as ``result.info["likelihood_max_opt"]``), which no draw is near."""
    monkeypatch.setattr(lmo_mod._Objective, "refine", lambda self, p, tol: p)
    monkeypatch.setattr(lmo_mod._Objective, "restart", lambda self, p, tol: p)
    truth = {"A": 3.0, "t0": 9.0, "w": 2.5}
    lc = _lc("_lmax_bump", truth)
    res = _result("_lmax_bump", lc, _draws({"A": 2.0, "t0": 12.0, "w": 4.0}, [0.1] * 3, 50))
    assert wp.likelihood_max_opt(res, lc).max_log_likelihood < _log_norm(lc) - 1.0
    earlier = dict(truth, A=3.0 + 1e-9)
    res.info["likelihood_max_opt"] = {"params": earlier, "max_log_likelihood": 0.0}
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.params == earlier
    assert pr.max_log_likelihood == pytest.approx(_log_norm(lc), abs=1e-8)
    res.info["likelihood_max_opt"] = wp.LikelihoodMaxOptResult.from_dict(
        dict(pr.to_dict(), params=truth))
    assert wp.likelihood_max_opt(res, lc).params == truth    # a LikelihoodMaxOptResult works too


# --- the fit's own density ----------------------------------------------------------------------
def test_the_fit_scatter_likelihood_is_used():
    """A free-scatter fit is optimised under GaussianLikelihoodWithScatter: sigma moves."""
    lc = _lc("_lmax_line", {"a": 1.0, "b": 0.2}, noise_seed=5, sigma=0.05)
    lc = wp.LightCurve(time=lc.time, band=lc.band, flux=lc.flux + np.random.default_rng(9).normal(
        0.0, 0.2, lc.n_points), flux_err=lc.flux_err)
    prior = wp.Prior({"a": wp.Uniform(-10.0, 10.0), "b": wp.Uniform(-1.0, 1.0),
                      "sigma": wp.Uniform(0.0, 2.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit_MCMC(lc, "_lmax_line", prior=prior, likelihood="gaussian_scatter",
                          nwalkers=12, nsteps=300, burnin=100, seed=0,
                          initial_guess={"a": 1.0, "b": 0.2, "sigma": 0.5})
    pr = wp.likelihood_max_opt(res, lc, prior=prior)
    assert pr.start_log_likelihood == pytest.approx(res.max_log_likelihood, abs=1e-9)
    assert pr.max_log_likelihood >= res.max_log_likelihood
    assert 0.1 < pr.params["sigma"] < 0.4                         # the injected 0.2, not zero
    # the MLE of a Gaussian scatter: its profile derivative vanishes at the peak
    lik = make_likelihood(lc, kind="gaussian_scatter", space="flux")
    f = np.asarray(_line(pr.params, lc.time))
    d = [lik.log_likelihood(f, sigma_extra=pr.params["sigma"] + h) for h in (-1e-4, 0.0, 1e-4)]
    assert d[1] >= d[0] and d[1] >= d[2]


def test_a_density_mismatch_warns():
    lc = _lc("_lmax_bump", {"A": 3.0, "t0": 9.0, "w": 2.5})
    res = _result("_lmax_bump", lc, _draws({"A": 3.0, "t0": 9.0, "w": 2.5}, [0.3] * 3, 50))
    res.max_log_likelihood += 5.0
    with pytest.warns(UserWarning, match="another density"):
        wp.likelihood_max_opt(res, lc)


# --- errors and LSST-sized data -----------------------------------------------------------------
def test_not_enough_data_gives_nan_aic_bic_and_says_so():
    truth = {"A": 3.0, "t0": 9.0, "w": 2.5}
    lc = _lc("_lmax_bump", truth, n=3)
    res = _result("_lmax_bump", lc, _draws(truth, [0.3, 0.8, 0.4], 50))
    with pytest.warns(UserWarning, match="not enough data"):
        pr = wp.likelihood_max_opt(res, lc)
    assert np.isnan(pr.aic) and np.isnan(pr.bic) and not pr.enough_data
    assert "not enough data" in repr(pr)


def test_errors_name_the_cause_and_the_fix():
    truth = {"A": 3.0, "t0": 9.0, "w": 2.5}
    lc = _lc("_lmax_bump", truth)
    res = _result("_lmax_bump", lc, _draws(truth, [0.3, 0.8, 0.4], 50))
    with pytest.raises(ValueError, match="after the same selection steps"):
        wp.likelihood_max_opt(res, _lc("_lmax_bump", truth, n=20))
    narrow = wp.Prior({"A": wp.Uniform(0.1, 1.0), "t0": wp.Uniform(0.0, 20.0),
                       "w": wp.Uniform(0.5, 10.0)})
    with pytest.raises(ValueError, match="Pass prior="):
        wp.likelihood_max_opt(res, lc, prior=narrow)
    res.model = "_lmax_not_registered"
    with pytest.raises(ValueError, match="Pass model="):
        wp.likelihood_max_opt(res, lc)
    with pytest.raises(ValueError, match="neither parameters of model"):
        wp.likelihood_max_opt(res, lc, model="_lmax_line", prior=BUMP_PRIOR)
    with pytest.raises(ValueError, match="backend='jax' is not available"):
        wp.likelihood_max_opt(res, lc, model="_lmax_bump", backend="jax")
    with pytest.raises(ValueError, match="backend must be"):
        wp.likelihood_max_opt(res, lc, model="_lmax_bump", backend="gpu")
    with pytest.raises(ValueError, match="tol must be"):
        wp.likelihood_max_opt(res, lc, model="_lmax_bump", tol=0.0)


def test_upper_limits_are_scored_as_the_fit_scored_them():
    """A few detections plus non-detections (the LSST case): optimised under the censored
    likelihood the fit recorded, in flux space."""
    truth = {"A": 3.0, "t0": 9.0, "w": 2.5}
    t = np.linspace(0.5, 20.0, 12)
    y = _bump(truth, t)
    ul = y < 0.5
    lc = wp.LightCurve(time=t, band=["r"] * 12, flux=np.where(ul, 0.3, y),
                       flux_err=np.where(ul, np.nan, 0.05), upper_limit=ul)
    lik = make_likelihood(lc, space="flux")
    assert type(lik).__name__ == "GaussianLikelihoodWithUpperLimits" and ul.sum() >= 4
    draws = _draws(truth, [0.2, 0.5, 0.3], 60)
    ll = np.array([lik.log_likelihood(_bump(r, t)) for r in draws.to_dict("records")])
    i = int(np.argmax(ll))
    res = SamplerResult(sampler="test", model="_lmax_bump", parameters=list(draws), samples=draws,
                        summary={}, best_params=draws.iloc[i].to_dict(), n_data=12, n_params=3,
                        runtime_s=0.0, max_log_likelihood=float(ll[i]),
                        info={"space": "flux", "likelihood": "GaussianLikelihoodWithUpperLimits"})
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.max_log_likelihood >= res.max_log_likelihood
    best = lik.log_likelihood(_bump(pr.params, t))
    assert pr.max_log_likelihood == pytest.approx(best, abs=1e-12)


# --- the JAX backend ----------------------------------------------------------------------------
@pytest.fixture
def x64():
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", was)


def _line_jax(theta, times, band_idx=None):
    import jax.numpy as jnp
    return theta[0] + theta[1] * jnp.asarray(times)


def test_jax_backend_reaches_the_same_edge_optimum_exactly(x64):
    lc = _lc("_lmax_line", {"a": 1.0, "b": 0.2}, noise_seed=3, sigma=0.1)
    hi = 0.15
    prior = wp.Prior({"a": wp.Uniform(-10.0, 10.0), "b": wp.Uniform(-1.0, hi)})
    wp.register_model("_lmax_line_jax", _line, ["a", "b"], prior=prior, overwrite=True,
                      predict_jax=_line_jax)
    res = _result("_lmax_line_jax", lc, _draws({"a": 1.5, "b": 0.1}, [0.2, 0.01], 100))
    pj = wp.likelihood_max_opt(res, lc)                     # auto: the model has predict_jax
    pc = wp.likelihood_max_opt(res, lc, backend="cpu")
    assert "JAX" in pj.method and "scipy" in pc.method
    assert pj.params["b"] == hi and pc.params["b"] == hi
    assert pj.params["a"] == pytest.approx(pc.params["a"], abs=1e-7)
    assert pj.max_log_likelihood == pytest.approx(pc.max_log_likelihood, abs=1e-9)
    assert pj.at_edge == ["b"] and pj.max_log_likelihood >= res.max_log_likelihood


def test_the_jax_backend_compiles_once_for_light_curves_of_one_bucket(x64):
    """An alert stream fitted with one model: the second light curve (same size bucket) reuses
    the first one's compiled programs, and each peak is still its own data's."""
    wp.register_model("_lmax_line_jax3", _line, ["a", "b"], prior=LINE_PRIOR, overwrite=True,
                      predict_jax=_line_jax)
    lmo_mod._PROGRAMS.clear()
    peaks = []
    for seed in (3, 4):
        lc = _lc("_lmax_line", {"a": 1.0, "b": 0.2}, noise_seed=seed, sigma=0.1)
        res = _result("_lmax_line_jax3", lc,
                      _draws({"a": 1.0, "b": 0.2}, [0.1, 0.01], 50, seed=seed))
        peaks.append((wp.likelihood_max_opt(res, lc, backend="jax"), _wls(lc)[1]))
    assert "reused" not in peaks[0][0].method and "reused" in peaks[1][0].method
    for peak, ln_l_max in peaks:
        assert peak.max_log_likelihood == pytest.approx(ln_l_max, abs=1e-6)


def test_a_float32_optimisation_says_so():
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        wp.register_model("_lmax_line_jax4", _line, ["a", "b"], prior=LINE_PRIOR,
                          overwrite=True, predict_jax=_line_jax)
        lc = _lc("_lmax_line", {"a": 1.0, "b": 0.2}, noise_seed=3, sigma=0.1)
        res = _result("_lmax_line_jax4", lc, _draws({"a": 1.0, "b": 0.2}, [0.1, 0.01], 50))
        with pytest.warns(UserWarning, match="float32"):
            wp.likelihood_max_opt(res, lc, backend="jax")
    finally:
        jax.config.update("jax_enable_x64", was)


def test_auto_backend_falls_back_to_the_cpu_for_a_likelihood_jax_lacks(x64):
    lc = _lc("_lmax_line", {"a": 1.0, "b": 0.2}, noise_seed=3, sigma=0.1)
    wp.register_model("_lmax_line_jax2", _line, ["a", "b"], prior=LINE_PRIOR, overwrite=True,
                      predict_jax=_line_jax)
    res = _result("_lmax_line_jax2", lc, _draws({"a": 1.0, "b": 0.2}, [0.1, 0.01], 50),
                  likelihood="MixtureGaussianLikelihood")
    lik = make_likelihood(lc, kind="mixture", space="flux")
    t = np.asarray(lc.time, float)
    res.max_log_likelihood = lik.log_likelihood(_line(res.best_params, t))
    pr = wp.likelihood_max_opt(res, lc)
    assert "scipy" in pr.method
    assert pr.max_log_likelihood == pytest.approx(lik.log_likelihood(_line(pr.params, t)),
                                                  abs=1e-12)
    with pytest.raises(ValueError, match="does not implement MixtureGaussianLikelihood"):
        wp.likelihood_max_opt(res, lc, backend="jax")


# --- redback arnett, SN2025pgp ------------------------------------------------------------------
# SN2025pgp, 30-day cut: 28 ZTF detections, MJD, AB mag. Fitted on the clock days since first
# detection + 3 d, at z = 0.051.
SN2025PGP_D30 = """\
time,band,magnitude,magnitude_err
60853.30425930023,ztfg,20.121428,0.18801062
60855.287835599855,ztfg,19.396336,0.12833908
60855.34975690022,ztfr,19.62281,0.15517399
60857.32444440015,ztfg,18.658733,0.0813545
60859.33033559984,ztfr,18.72278,0.0760506
60859.35039350018,ztfg,18.38077,0.057640616
60861.3628934999,ztfr,18.441578,0.061664153
60861.413611100055,ztfg,18.237764,0.0520921
60863.2532060002,ztfg,18.280054,0.060048737
60863.30518520018,ztfr,18.37345,0.08445486
60865.299016200006,ztfr,18.54368,0.09427341
60865.34564810013,ztfg,18.388763,0.07113685
60867.29686339991,ztfr,18.481213,0.08425352
60867.34576389985,ztfg,18.501478,0.08157507
60869.19906250015,ztfg,18.666647,0.07660789
60869.27834489988,ztfr,18.634205,0.1035424
60871.248425900005,ztfg,18.869564,0.093682125
60871.28538190015,ztfr,18.7424,0.13346128
60873.24048609985,ztfg,18.921078,0.10473108
60873.28320599999,ztfr,18.817572,0.11878267
60876.28289349983,ztfr,18.983622,0.101359196
60876.32490740018,ztfg,19.213766,0.108030796
60878.276493099984,ztfr,19.183887,0.083077274
60878.34434030019,ztfg,19.521996,0.1286301
60880.26185189979,ztfr,19.323874,0.097067565
60880.320763899945,ztfg,19.76436,0.14210327
60882.22098380001,ztfg,20.039885,0.17725715
60882.26236109994,ztfr,19.643427,0.13453755
"""
T_FD, Z_PGP = 60853.30425930023, 0.051
#: The best draw of a 60 x 30 000 CPU emcee reference on this cut
#: (1.8 M evaluations, ln L 31.3826), and the peak likelihood_max_opt finds from it: ln L 31.49396,
#: with f_nickel and kappa on their prior edges.
REF_BEST = {"f_nickel": 0.9813512249856495, "mej": 0.45026143953444986, "vej": 8207.61852161502,
            "kappa": 0.052541421376849515, "kappa_gamma": 0.011243343382461652,
            "temperature_floor": 6850.259466620045}
REF_PEAK_LNL = 31.49396


def _pgp(dt=3.0):
    d = pd.read_csv(io.StringIO(SN2025PGP_D30))
    return wp.LightCurve(time=d["time"].to_numpy() - T_FD + dt, band=d["band"].to_numpy(),
                         magnitude=d["magnitude"].to_numpy(),
                         magnitude_err=d["magnitude_err"].to_numpy(), name="sn2025pgp_d30")


@pytest.fixture(scope="module")
def arnett_cpu():
    pytest.importorskip("redback")
    return wp.register_redback("arnett", band_names=["ztfg", "ztfr"], redshift=Z_PGP,
                               name="_lmax_arnett_redback", overwrite=True)


def test_redback_arnett_likelihood_max_climbs_above_the_chain(arnett_cpu):
    """A short CPU emcee run on redback arnett, started beside the reference chain's best draw;
    the optimisation must beat the chain, reach the reference peak, and respect the constraint
    wall."""
    lc = _pgp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit_MCMC(lc, arnett_cpu.name, nwalkers=16, nsteps=150, burnin=75, seed=0,
                          initial_guess=REF_BEST, initial_scatter=1e-4)
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.start_log_likelihood == res.max_log_likelihood        # same density, same number
    assert pr.max_log_likelihood >= res.max_log_likelihood
    assert pr.max_log_likelihood == pytest.approx(REF_PEAK_LNL, abs=2e-3)
    assert {"f_nickel", "kappa"} <= set(pr.at_edge)
    assert pr.params["f_nickel"] == pytest.approx(1.0, rel=1e-9)
    assert pr.params["kappa"] == pytest.approx(0.05, rel=1e-9)
    assert arnett_cpu.predict.physical({**pr.params, **arnett_cpu.predict.pinned})
    assert pr.bic == pytest.approx(-2 * pr.max_log_likelihood + 6 * np.log(28))


def test_redback_arnett_like_for_like_with_the_v3_likelihood_maximum(arnett_cpu):
    """An earlier independent fit ("v3") took this cut with the redshift and the explosion time
    free, and 0.1.0's monochromatic photometry, and optimised it to a likelihood maximum of
    ln L 31.569594. Pinning z and dt at v3's peak and rebuilding its photometry reproduces that
    number, and the optimisation from there must not fall below it (within 0.5)."""
    from scipy.special import ndtri

    v3 = {"f_nickel": 0.19278610835467436, "mej": 0.03158794465205126, "vej": 1000.0004704471487,
          "kappa": 0.08592906724845734, "kappa_gamma": 0.002563840680746869,
          "temperature_floor": 6619.32993964552}
    v3_lnl, z_u, dt = 31.569593984211917, 0.19642137253787847, 2.8186899712147673
    sig = 0.05 * (1 + Z_PGP)
    z = float(np.clip(Z_PGP + sig * ndtri(z_u), 0.001, Z_PGP + 3 * sig))
    m = wp.register_redback("arnett", band_names=["ztfg", "ztfr"], redshift=z,
                            photometry="monochromatic", name="_lmax_arnett_v3", overwrite=True)
    lc = _pgp(dt)
    lik = make_likelihood(lc, space="magnitude")
    ll0 = lik.log_likelihood(m.predict(v3, np.asarray(lc.time, float), np.asarray(lc.band)))
    assert ll0 == pytest.approx(v3_lnl, abs=1e-6)
    res = SamplerResult(sampler="v3", model=m.name, parameters=list(m.parameters),
                        samples=pd.DataFrame(columns=m.parameters), summary={}, best_params=v3,
                        n_data=28, n_params=6, runtime_s=0.0, max_log_likelihood=ll0,
                        info={"space": "magnitude", "likelihood": "GaussianLikelihood"})
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.max_log_likelihood >= ll0
    assert abs(pr.max_log_likelihood - v3_lnl) < 0.5


@pytest.mark.slow
def test_redback_arnett_gpu_fit_optimised_on_both_backends(arnett_cpu, x64):
    """``emcee_jax`` on the JAX arnett twin, a short run from the default prior-scan start. The JAX
    optimisation and a CPU optimisation of the same draws with the redback model reach the same
    peak, the reference one, far above the chain's best; redback scores the JAX peak the same."""
    from whisper_cbpf.models import redback_adapter as ra
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    lc = _pgp()
    dl = ra.redback_luminosity_distance_cm(Z_PGP, model="arnett")
    twin = wp.register_supernova("arnett", ["ztfg", "ztfr"], Z_PGP, dl, name="_lmax_arnett_jax")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_emcee_jax(lc, twin, nwalkers=32, nsteps=600, burnin=300, thin=5, seed=0)
    pj = wp.likelihood_max_opt(res, lc)
    pc = wp.likelihood_max_opt(res, lc, model=arnett_cpu.name)
    assert "JAX" in pj.method and "scipy" in pc.method
    for pr in (pj, pc):
        assert pr.max_log_likelihood >= res.max_log_likelihood - 1e-9
        assert pr.max_log_likelihood == pytest.approx(REF_PEAK_LNL, abs=2e-3)
    lik = make_likelihood(lc, space="magnitude")
    t, b = np.asarray(lc.time, float), np.asarray(lc.band)
    assert lik.log_likelihood(arnett_cpu.predict(pj.params, t, b)) == pytest.approx(
        pj.max_log_likelihood, abs=1e-6)
    theta = np.array([pj.params[n] for n in twin.parameters])
    assert bool(twin.predict_jax.constraint_ok(theta))


def test_an_mjd_valued_parameter_beside_a_log_uniform_one_raises_no_overflow_warning():
    """``10 ** s`` was taken on every column and kept only on the log ones: on an MJD-valued
    explosion time it overflowed, and a RuntimeWarning reached the user in every compare."""
    from whisper_cbpf.likelihood_max_opt import _Box

    box = _Box(wp.Prior({"t_exp": wp.Uniform(60990.0, 61003.0),
                         "mej": wp.LogUniform(0.01, 10.0)}), ["t_exp", "mej"])
    u = np.array([[0.3, 0.5], [0.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = box.to_x(u)
        d = box.dx_du(u)
    np.testing.assert_allclose(x[0], [60990.0 + 0.3 * 13.0, 10.0 ** -0.5])
    assert x[1].tolist() == [60990.0, 10.0] and np.all(np.isfinite(d))
