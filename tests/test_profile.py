"""``profile`` and ``capacity``: the cost of a model on this hardware, and alerts per night.

The claims, each a test:

1. ``profile`` reports compile and call time per batch size for the value and the gradient,
   XLA's scratch memory, the fraction of prior draws with a finite density, and whether the
   gradient is finite; a program that would not fit in memory is skipped, with the reason.
2. ``capacity`` reproduces measured nights: 636.7 s per alert, one at a time, is ~68 alerts in
   12 h; 106.16 GPU s per alert on 5 GPUs is 2 035; a batched Arnett chain's 13.928 GPU s per
   alert is 258.5 alerts per GPU-hour; later compiled alerts (3.3-4.2 s each) and the TDE's GPU
   path (110-144 s) bracket the counts below, and a full-budget Arnett fit (167 s on one GPU,
   215 s on 30 cores) gives 258 / 200.
3. A ``ProfileReport`` converts to seconds per alert as ``(nsteps x nwalkers + scan) x`` the lowest
   cost per evaluation.
4. Errors name the cause and the fix.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
import whisper_cbpf.models as M  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402
from whisper_cbpf.profile import ProfileReport, capacity, profile  # noqa: E402

TRUTH = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _flare_lc(n=30):
    flare = wp.get_model("flare_jax")
    t = np.linspace(0.5, 30.0, n)
    return wp.LightCurve(time=t, band=["r"] * n, flux=flare.predict(TRUTH, t),
                         flux_err=np.full(n, 0.1))


# --- 1. profile --------------------------------------------------------------------------------

def test_profile_times_value_and_gradient_by_batch_size():
    report = profile("flare_jax", _flare_lc(), batch_sizes=(1, 16), repeats=2)
    assert [(r["batch"], r["kind"]) for r in report.rows] == [
        (1, "value"), (1, "value_and_grad"), (16, "value"), (16, "value_and_grad")]
    for r in report.rows:
        assert r["skipped"] is None and r["compile_s"] > 0 and r["call_s"] > 0
        assert r["per_eval_us"] == pytest.approx(1e6 * r["call_s"] / r["batch"])
    assert report.finite_fraction == 1.0
    assert report.gradient["finite_fraction"] == 1.0 and report.gradient["zero_fraction"] == 0.0
    assert report.gradient["n"] == 16 and len(report.gradient["median_abs"]) == 4
    assert report.x64 and report.n_data == 30 and report.bucket == 32
    assert report.names == ["log_amp", "log_sigma", "log_tau", "t0"]
    json.dumps(report.to_dict())                              # JSON-able
    text = repr(report)
    assert "us / eval" in text and "capacity(report" in text
    assert report.seconds_per_eval() == pytest.approx(
        min(r["per_eval_us"] for r in report.rows if r["kind"] == "value") * 1e-6)


def test_profile_counts_non_finite_densities_and_gradients():
    def predict_jax(theta, t):                   # NaN physics for a < 0: half the prior box
        return jnp.where(theta[0] > 0, theta[0] * jnp.exp(-t / theta[1]), jnp.nan)

    model = M.Model(name="half_nan_toy", predict=lambda p, t, b=None: p["a"] * np.exp(-t / p["tau"]),
                    parameters=["a", "tau"],
                    default_prior=Prior({"a": Uniform(-1.0, 1.0), "tau": Uniform(1.0, 10.0)}),
                    predict_jax=predict_jax)
    t = np.linspace(0.0, 10.0, 20)
    lc = wp.LightCurve(time=t, band=["r"] * 20, flux=0.5 * np.exp(-t / 4.0),
                       flux_err=np.full(20, 0.05))
    report = profile(model, lc, batch_sizes=(64,), repeats=1)
    assert 0.3 < report.finite_fraction < 0.7
    assert report.gradient["finite_fraction"] == 1.0          # where the density is finite
    only_value = profile(model, lc, batch_sizes=(8,), grad=False, repeats=1)
    assert [r["kind"] for r in only_value.rows] == ["value"] and only_value.gradient == {}


def test_a_program_that_would_not_fit_in_memory_is_skipped_with_the_reason(monkeypatch):
    import sys

    # the module, not the function `whisper_cbpf.profile` that shadows its name on the package
    P = sys.modules[profile.__module__]
    monkeypatch.setattr(P, "_free_bytes", lambda device: 0)
    report = profile("flare_jax", _flare_lc(), batch_sizes=(64,), repeats=1)
    skipped = [r for r in report.rows if r["skipped"]]
    if not any(r["temp_bytes"] for r in report.rows):
        pytest.skip("this backend reports no scratch memory")
    assert skipped and "GB of scratch memory" in skipped[0]["skipped"]
    assert "skipped" in repr(report)


# --- 2. capacity reproduces the demos' arithmetic ----------------------------------------------

def test_capacity_reproduces_the_demos_nights():
    v3 = capacity(636.7, hours=12, n_gpus=1)                     # one alert at a time
    assert round(v3["alerts_exact"]) == 68 and v3["alerts"] == 67
    now = capacity(106.16439285714287, hours=12, n_gpus=5)       # 5 GPUs, measured per-alert cost
    assert round(now["alerts_exact"]) == 2035
    s17 = capacity(13.928)                                       # batched Arnett, 256 per call
    assert round(s17["alerts_per_gpu_hour"], 1) == 258.5
    s12 = [capacity(s)["alerts"] for s in (4.2, 3.3)]            # later, compiled alerts
    assert s12 == [10285, 13090]
    s15 = [capacity(s)["alerts"] for s in (144.0, 110.0)]        # the TDE's GPU path
    assert s15 == [300, 392]
    # full budget: 600 000 Arnett draws (60 walkers) in 167 s on one GPU, 215 s on 30 cores
    i10 = [capacity(s, hours=12)["alerts"] for s in (167.0, 215.0)]
    assert i10 == [258, 200]
    assert capacity(167.0)["alerts_per_gpu_hour"] == pytest.approx(21.557, abs=1e-3)
    both = capacity([13.928, 11.994], hours=12, n_gpus=5)        # one alert, two models
    assert both["seconds_per_alert"] == pytest.approx(25.922)
    assert both["alerts"] == math.floor(12 * 3600 * 5 / 25.922)
    assert len(both["basis"]) == 2 and both["assumptions"]


def test_a_profile_converts_to_seconds_per_alert():
    rep = ProfileReport("m", "gpu", True, 59, 64, ["a"],
                        rows=[{"batch": 1, "kind": "value", "per_eval_us": 314.0},
                              {"batch": 4096, "kind": "value", "per_eval_us": 8.9},
                              {"batch": 4096, "kind": "value_and_grad", "per_eval_us": 24.2}])
    assert rep.seconds_per_eval() == pytest.approx(8.9e-6)
    assert rep.seconds_per_eval("value_and_grad") == pytest.approx(24.2e-6)
    s = rep.seconds_per_alert(nwalkers=60, nsteps=10000)
    assert s == pytest.approx((60 * 10000 + 1000) * 8.9e-6)
    out = capacity(rep, hours=12, n_gpus=1, nwalkers=60, nsteps=10000)
    assert out["seconds_per_alert"] == pytest.approx(s)
    assert "8.90 us per evaluation" in out["basis"][0]


# --- 4. errors ---------------------------------------------------------------------------------

def test_errors_name_the_cause_and_the_fix():
    with pytest.raises(ValueError, match="not a positive number"):
        capacity(0.0)
    with pytest.raises(ValueError, match="not a positive number"):
        capacity(float("nan"))
    with pytest.raises(TypeError, match="seconds per alert"):
        capacity("fast")
    with pytest.raises(ValueError, match="hours must be > 0"):
        capacity(10.0, hours=0)
    with pytest.raises(ValueError, match="no cost"):
        capacity([])
    with pytest.raises(ValueError, match="positive integers"):
        profile("flare_jax", _flare_lc(), batch_sizes=(0,))
    empty = ProfileReport("m", "cpu", True, 1, 16, ["a"])
    with pytest.raises(ValueError, match="no measured 'value' row"):
        empty.seconds_per_eval()
