"""Regression tests for the scientific-review fixes (physical + Bayesian correctness)."""
import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_labia as wp
from whisper_labia.io.photometry import flux_density_to_mag
from whisper_labia.io.schema import _ccm89_a_lambda
from whisper_labia.likelihood import make_likelihood
from whisper_labia.models import get_model, register_model

TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}


def _flux_lc(n=60, seed=0):
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, n)
    flux = m.predict(TRUE, t, None)
    obs = flux + np.random.default_rng(seed).normal(0, 0.1, flux.shape)
    return wp.LightCurve(time=t, band=["r"] * n, flux=obs, flux_err=np.full_like(flux, 0.1), name="syn")


# --- #2: ABC / ABC-SMC AIC are from the EXACT Gaussian likelihood (comparable across samplers) ------

def test_abc_aic_uses_exact_likelihood():
    lc = _flux_lc()
    r = wp.fit_ABC(lc, "gaussian_rise", n_simulations=4000, quantile=0.03, n_jobs=1, seed=0)
    exact = make_likelihood(lc).log_likelihood(
        get_model("gaussian_rise").predict(r.best_params, np.asarray(lc.time), None))
    assert np.isclose(r.max_log_likelihood, exact)            # exact Gaussian, incl. normalisation
    assert not np.isclose(r.max_log_likelihood, -0.5 * r.min_distance)   # NOT the bare chi2/-2
    assert np.isclose(r.aic, -2.0 * exact + 2 * 4)
    assert r.info["likelihood_space"] == "flux"


# --- #7: ABC-SMC is importance-weighted ------------------------------------------------------------

def test_abc_smc_is_importance_weighted():
    lc = _flux_lc()
    r = wp.fit_ABC_SMC(lc, "gaussian_rise", n_particles=200, n_rounds=3, n_jobs=1, seed=0)
    assert r.info["weighted"] is True
    assert all("effective_sample_size" in rd for rd in r.info["rounds"])
    assert r.info["rounds"][-1]["effective_sample_size"] > 1.0
    # recovery still holds (median within ~40% of truth)
    assert abs(r.summary["amplitude"]["median"] - 5.0) / 5.0 < 0.4


# --- #1: WAIC drops non-finite DRAWS, never data points -------------------------------------------

def _spiky_predict(params, times, bands=None):
    """gaussian_rise, but inject an inf at the first point when amplitude is large (one bad draw)."""
    out = get_model("gaussian_rise").predict(params, times, bands).copy()
    if float(params["amplitude"]) > 50.0:
        out[0] = np.inf
    return out


def test_waic_drops_draws_not_data_points():
    register_model("_spiky", _spiky_predict, ["amplitude", "t0", "sigma_rise", "tau_decay"],
                   overwrite=True)
    lc = _flux_lc(n=40)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({k: rng.normal(TRUE[k], 0.02 * abs(TRUE[k]), 50) for k in TRUE})
    df.loc[0, "amplitude"] = 200.0                            # one pathological draw -> inf at point 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = wp.waic(df, lc, "_spiky", space="flux", max_samples=50)
    assert w["n_data"] == lc.n_points                         # ALL data points retained
    assert w["n_draws_dropped"] >= 1                          # the bad draw was dropped


# --- #3: mck19 blackbody carries the factor of pi -------------------------------------------------

def test_mck19_blackbody_has_pi_factor():
    m = get_model("mck19")
    p = {"v_kick": 300.0, "M_smbh": 1e8, "M_bh": 80.0, "r_bh": 700.0, "redshift": 0.28}
    t = np.linspace(0, 60, 400)
    peak_mag = -2.5 * np.log10(m.predict(p, t, np.array(["g"] * 400)).max() / 3631.0)
    # pi-corrected peak is exactly 2.5*log10(pi) ~ 1.24 mag brighter than the (wrong) no-pi value 27.14
    assert peak_mag == pytest.approx(27.143 - 2.5 * np.log10(np.pi), abs=0.05)


# --- #4: CCM89 clamps out-of-range bands (does not silently return A=0) ----------------------------

def test_ccm89_clamps_out_of_range_and_warns():
    # F444W ~ 44040 A -> x ~ 0.227 um^-1, below the 0.3 valid edge
    with pytest.warns(UserWarning, match="0.3-8"):
        a = _ccm89_a_lambda(np.array([44040.0]), ebv=0.1, rv=3.1)
    assert np.all(np.isfinite(a)) and a[0] > 0.0             # clamped -> finite, non-zero extinction
    # an in-range optical band is unaffected and needs no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a_opt = _ccm89_a_lambda(np.array([5500.0]), ebv=0.1, rv=3.1)
    assert a_opt[0] > 0.0


# --- #8: flux_density_to_mag guards non-positive flux ---------------------------------------------

def test_flux_to_mag_nonpositive_is_nan_with_warning():
    with pytest.warns(UserWarning, match="non-positive"):
        mag, mag_err = flux_density_to_mag(np.array([1e-3, 0.0, -1e-4]),
                                           flux_err=np.array([1e-4, 1e-4, 1e-4]))
    assert np.isfinite(mag[0]) and np.isnan(mag[1]) and np.isnan(mag[2])
    assert np.isfinite(mag_err[0]) and np.isnan(mag_err[1])
