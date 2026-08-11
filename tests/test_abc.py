import json

import numpy as np

from whisper_labia import (
    LightCurve,
    Prior,
    Uniform,
    fit,
    fit_ABC,
    list_samplers,
    register_model,
)
from whisper_labia.models.flare import flare_flux


def _synthetic(truth, n=50, noise_frac=0.01, seed=0):
    times = np.linspace(0.5, 30, n)
    bands = np.array(["r"] * n)
    flux = flare_flux(truth, times, bands)
    err = np.full_like(flux, noise_frac * flux.max()) + 1e-9
    noisy = flux + np.random.default_rng(seed).normal(0, err)
    return LightCurve(time=times, band=bands, flux=noisy, flux_err=err, name="synth")


def test_abc_recovers_amplitude():
    truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    lc = _synthetic(truth)
    res = fit_ABC(lc, "flare", n_simulations=30000, quantile=0.01, n_jobs=4, seed=1)
    assert res.n_samples > 0
    assert abs(res.summary["amplitude"]["median"] - 5.0) < 1.5
    assert abs(res.best_params["amplitude"] - 5.0) < 1.0
    assert res.min_distance < res.n_data * 5          # a good fit was found
    assert np.isfinite(res.aic) and np.isfinite(res.bic)


def test_abc_reproducible_and_njobs_independent():
    """Same (seed) -> identical posterior, regardless of n_jobs (the scientific reproducibility contract)."""
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=30)
    r1 = fit_ABC(lc, "flare", n_simulations=2000, quantile=0.05, n_jobs=1, seed=3)
    r1b = fit_ABC(lc, "flare", n_simulations=2000, quantile=0.05, n_jobs=1, seed=3)
    r4 = fit_ABC(lc, "flare", n_simulations=2000, quantile=0.05, n_jobs=4, seed=3)
    assert r1.n_samples > 0
    assert r1.samples.equals(r1b.samples) and r1.best_params == r1b.best_params   # determinism
    assert r1.samples.equals(r4.samples) and r1.best_params == r4.best_params      # n_jobs-independent
    assert fit_ABC(lc, "flare", n_simulations=2000, quantile=0.05, n_jobs=1, seed=4).best_params \
        != r1.best_params                                                          # different seed -> different


def test_abc_acceptance_count():
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=20)
    res = fit_ABC(lc, "flare", n_simulations=10000, quantile=0.1, n_jobs=2, seed=2)
    assert abs(res.info["n_accepted"] - 1000) <= 100   # ~10% accepted


def test_abc_result_json():
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=20)
    res = fit_ABC(lc, "flare", n_simulations=2000, quantile=0.05, n_jobs=2)
    d = json.loads(res.to_json())
    assert d["sampler"] == "abc" and d["model"] == "flare"
    assert "aic" in d and "summary" in d and "best_params" in d


def test_fit_dispatch_and_list_samplers():
    assert "abc" in list_samplers()
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=20)
    res = fit(lc, "flare", sampler="abc", n_simulations=1000, quantile=0.1, n_jobs=1)
    assert res.sampler == "abc"


def test_abc_custom_model_serial():
    # A closure model is not picklable, so parallel would fail -> use n_jobs=1.
    def line(params, times, bands=None):
        return params["a"] * np.asarray(times, float) + params["b"]

    register_model("line_abc", line, ["a", "b"],
                   prior=Prior({"a": Uniform(0, 5), "b": Uniform(0, 5)}), overwrite=True)
    times = np.linspace(0, 10, 30)
    bands = np.array(["g"] * 30)
    flux = line({"a": 2.0, "b": 1.0}, times)
    err = np.full_like(flux, 0.1) + 1e-9
    noisy = flux + np.random.default_rng(0).normal(0, err)
    lc = LightCurve(time=times, band=bands, flux=noisy, flux_err=err)
    res = fit_ABC(lc, "line_abc", n_simulations=5000, quantile=0.02, n_jobs=1, seed=0)
    assert abs(res.best_params["a"] - 2.0) < 0.5
