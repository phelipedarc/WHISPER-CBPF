"""Tests for the inference-validation tools (recovery, PPC, SBC)."""
import numpy as np
import pandas as pd
import pytest

import whisper_labia as wp
from whisper_labia.models import get_model

TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}


def _gaussian_posterior(truth, std_frac=0.05, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: rng.normal(v, std_frac * abs(v), n) for k, v in truth.items()})


def test_recovery_metrics_zscore_and_coverage():
    # posterior centred on the truth -> z ~ 0, both intervals cover
    post = _gaussian_posterior(TRUE)
    r = wp.recovery_metrics(post, TRUE)
    for p in TRUE:
        assert abs(r[p]["z_score"]) < 0.5 and r[p]["within_68"] and r[p]["within_95"]
    assert r["_summary"]["coverage95"] == 1.0 and r["_summary"]["max_abs_z"] < 0.5
    # a posterior offset by ~4 sigma is flagged (large |z|, 68% interval misses)
    off = _gaussian_posterior({k: v * 1.2 for k, v in TRUE.items()}, std_frac=0.05)
    r2 = wp.recovery_metrics(off, TRUE)
    assert r2["_summary"]["max_abs_z"] > 2.0
    assert r2["_summary"]["coverage68"] < 1.0


def test_posterior_predictive_check_well_fit():
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 60)
    flux = m.predict(TRUE, t, None)
    sigma = 0.1
    obs = flux + np.random.default_rng(0).normal(0, sigma, flux.shape)
    lc = wp.LightCurve(time=t, band=["r"] * 60, flux=obs, flux_err=np.full_like(flux, sigma), name="syn")
    post = _gaussian_posterior(TRUE, std_frac=0.01)         # tight, correct posterior
    ppc = wp.posterior_predictive_check(post, lc, "gaussian_rise", n_draws=300)
    assert 0.7 < ppc["reduced_chi2"] < 1.4                  # good fit -> reduced chi2 ~ 1
    assert ppc["ppc_coverage95"] > 0.85                     # predictive band contains ~most data
    assert 0.1 < ppc["bayesian_p_value"] < 0.9              # tight, correct posterior -> p ~ 0.5
    assert ppc["median"].shape == ppc["time_grid"].shape and ppc["dof"] == 60 - 4


def test_sbc_ranks_uniform_vs_biased():
    rng = np.random.default_rng(0)
    M, L = 100, 400
    # calibrated: ranks uniform on {0..M} -> high uniformity p
    uniform = {"a": rng.integers(0, M + 1, L)}
    su = wp.sbc_ranks(uniform, n_bins=20)
    assert su["a"]["uniformity_p"] > 0.05 and su["_summary"]["calibrated"] is True
    # mis-calibrated (overconfident): ranks pushed to the edges -> low uniformity p
    edges = {"a": np.where(rng.random(L) < 0.5, rng.integers(0, M // 10, L),
                           rng.integers(M - M // 10, M + 1, L))}
    se = wp.sbc_ranks(edges, n_bins=20)
    assert se["a"]["uniformity_p"] < 0.05 and se["_summary"]["calibrated"] is False


def test_sbc_rank_basic():
    s = np.arange(100.0)
    assert wp.sbc_rank(s, 49.5) == 50 and wp.sbc_rank(s, -1) == 0 and wp.sbc_rank(s, 1000) == 100
