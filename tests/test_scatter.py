"""Tests for the Villar+17 extra-scatter likelihood and its sampler routing."""
import numpy as np
import pytest

import whisper_labia as wp
from whisper_labia.models import get_model
from whisper_labia.priors import LogUniform, Prior, Uniform

TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}


def _lc(n=40, err=0.1, extra=0.3, seed=3):
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, n)
    obs = m.predict(TRUE, t, None) + np.random.default_rng(seed).normal(
        0, np.sqrt(err ** 2 + extra ** 2), t.shape)
    return wp.LightCurve(time=t, band=["r"] * n, flux=obs, flux_err=np.full_like(t, err))


def _prior():
    return Prior({"amplitude": Uniform(1, 10), "t0": Uniform(2, 15), "sigma_rise": Uniform(0.5, 8),
                  "tau_decay": Uniform(5, 40), "sigma": LogUniform(0.01, 2.0)})


def test_scatter_likelihood_formula_and_zero_limit():
    lc = _lc()
    base = wp.GaussianLikelihood(lc, space="flux")
    sc = wp.GaussianLikelihoodWithScatter(lc, space="flux")
    m = get_model("gaussian_rise")
    flux = m.predict(TRUE, np.asarray(lc.time, float), None)
    # sigma_extra=0 reduces exactly to the plain Gaussian
    assert sc.log_likelihood(flux, sigma_extra=0.0) == pytest.approx(base.log_likelihood(flux))
    # hand-computed Villar form at sigma_extra=0.3
    var = np.asarray(lc.flux_err, float) ** 2 + 0.3 ** 2
    expected = -0.5 * np.sum((np.asarray(lc.flux, float) - flux) ** 2 / var
                             + np.log(2 * np.pi * var))
    assert sc.log_likelihood(flux, sigma_extra=0.3) == pytest.approx(expected)
    # a mis-reported error budget prefers nonzero scatter (true data noise 0.32 > reported 0.1)
    assert sc.log_likelihood(flux, sigma_extra=0.3) > sc.log_likelihood(flux, sigma_extra=0.0)
    assert "gaussian_scatter" in wp.list_likelihoods() and "villar" in wp.list_likelihoods()
    assert sc.log_likelihood_pointwise(flux, sigma_extra=0.3).sum() == pytest.approx(
        sc.log_likelihood(flux, sigma_extra=0.3))


def test_mcmc_recovers_extra_scatter():
    r = wp.fit_MCMC(_lc(), get_model("gaussian_rise"), prior=_prior(), nsteps=1500, burnin=500,
                    thin=3, space="flux", likelihood="gaussian_scatter", seed=0)
    s = r.summary["sigma"]
    assert 0.15 < s["median"] < 0.55                    # true extra scatter 0.3
    assert r.n_params == 5                              # sigma counted in AIC/BIC
    assert "sigma" in r.samples.columns


def test_abc_scatter_validation_and_columns():
    lc, prior = _lc(), _prior()
    with pytest.raises(ValueError, match="not in the prior"):
        wp.fit_ABC(lc, "gaussian_rise", prior=prior, n_simulations=100, scatter_param="bogus")
    with pytest.raises(ValueError, match="simulate_noise"):
        wp.fit_ABC(lc, "gaussian_rise", prior=prior, n_simulations=100, scatter_param="sigma",
                   simulate_noise=False)
    r = wp.fit_ABC(lc, "gaussian_rise", prior=prior, n_simulations=800, quantile=0.05,
                   scatter_param="sigma", n_jobs=2, seed=0)
    assert "sigma" in r.samples.columns and r.info["scatter_param"] == "sigma"
    assert r.n_params == 5


def test_abc_magnitude_space():
    # near-peak window keeps the noisy flux strictly positive -> finite magnitudes
    m = get_model("gaussian_rise")
    t = np.linspace(4, 20, 40)
    obs = m.predict(TRUE, t, None) + np.random.default_rng(3).normal(0, 0.1, t.shape)
    lc = wp.LightCurve(time=t, band=["r"] * 40, flux=obs, flux_err=np.full_like(t, 0.1)).add_mag()
    r = wp.fit_ABC(lc, "gaussian_rise", prior=_prior(), n_simulations=600, quantile=0.05,
                   space="magnitude", scatter_param="sigma", n_jobs=1, seed=0)
    assert r.info["space"] == "magnitude" and np.isfinite(r.max_log_likelihood)
