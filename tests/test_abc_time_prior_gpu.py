"""``ABC_time_prior=True`` on a GPU: the ABC step runs on the card, fast, and holds the truth.

The claims, each a test (skipped without a GPU):

1. **On the card**: the ABC step of :func:`whisper_cbpf.abc_time_prior` runs on the GPU, and the
   same seed gives the same window there.
2. **A JAX TDE model** (``tde_stream_stream``, ``t_exp`` 12 days before the first point, two LSST
   bands, 5 % + 1 %-of-peak errors): one million draws take well under a minute, enough of them
   are accepted, and the window holds the true ``t_exp``; ``wp.fit(..., ABC_time_prior=True)``
   then fits inside that window.
"""
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.models.jax import tde_stream_stream as TS  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402

from test_abc_time_prior import MODEL, T_TRUE, _lc  # noqa: E402

BANDS = ["lsstg", "lsstr"]
Z, DL = 0.05, 7.093375016731583e26
TRUTH = dict(mbh_6=1.5, mstar=1.0, sigma_t=20.0, peak_time=30.0)
TDE_T_EXP = -12.0


def _gpu_or_skip():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("needs a GPU: the ABC step is timed on the card")


def _stream_lc():
    m = TS.tde_stream_stream_model(BANDS, Z, DL, free=["t_exp"],
                                   prior=Prior({"t_exp": Uniform(-60.0, 0.0)}))
    t = np.repeat(np.arange(0.0, 120.0, 4.0), 2)
    b = np.array(BANDS * (t.size // 2))
    flux = np.asarray(m.predict(dict(TRUTH, t_exp=TDE_T_EXP), t, b), float)
    peak = {band: flux[b == band].max() for band in BANDS}
    err = 0.05 * flux + 0.01 * np.array([peak[x] for x in b])
    noisy = flux + np.random.default_rng(0).normal(0.0, err)
    return m, wp.LightCurve(time=t, band=b, flux=noisy, flux_err=err, redshift=Z)


def test_the_abc_step_runs_on_the_gpu_and_repeats_for_a_seed():
    _gpu_or_skip()
    a_win, a = wp.abc_time_prior(_lc(), MODEL, n_simulations=200_000, seed=7)
    b_win, b = wp.abc_time_prior(_lc(), MODEL, n_simulations=200_000, seed=7)
    assert "cuda" in a["device"].lower() or "gpu" in a["device"].lower()
    assert a_win.bounds == b_win.bounds and a["n_accepted"] == b["n_accepted"]
    assert a_win.low < T_TRUE < a_win.high and not a["fell_back"]


def test_a_jax_tde_model_one_million_draws_in_under_a_minute():
    _gpu_or_skip()
    m, lc = _stream_lc()
    window, rec = wp.abc_time_prior(lc, m, n_simulations=1_000_000)
    assert rec["runtime_s"] < 60.0, rec["runtime_s"]
    assert rec["default_window"] == [-30.0, 0.0]
    assert not rec["fell_back"] and rec["n_accepted"] >= 50, rec
    assert window.low <= TDE_T_EXP <= window.high
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit(lc, m, sampler="emcee_jax", nsteps=600, burnin=200, check_convergence=False,
                     ABC_time_prior=True, ABC_time_prior_kwargs={"n_simulations": 1_000_000})
    assert res.info["abc_time_prior"]["window"] == rec["window"]
    t = np.asarray(res.samples["t_exp"], float)
    assert np.all((t >= window.low) & (t <= window.high))
