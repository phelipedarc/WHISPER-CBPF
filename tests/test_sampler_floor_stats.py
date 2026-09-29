"""The simulation-based GPU samplers report how much of what they simulated sat at ``mag_floor``.

the JAX models cap magnitudes at ``mag_floor`` (40 by default) and redback does not, so a
sampler that trains on prior draws can learn from a wall of identical points. The batched forward
map (``_adapters.make_batched_predict_jax``) counts them into ``.floor_stats`` and warns past 10 %;
``info["mag_floor_stats"]`` puts that count on the result, where a saved run keeps it.

``snpe_gpu`` calls the map on concrete arrays, so it counts every simulated point. ``abc_gpu`` and
``abc_smc_gpu`` run it inside one compiled scan, where the map sees tracers and counts nothing; they
then report ``None`` ("not counted") rather than a dict reading 0 of 0 points, which a reader would
take for "nothing at the floor".
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.io.photometry import AB_ZEROPOINT_JY  # noqa: E402

MAG_FLOOR = 40.0
FLOOR_JY = AB_ZEROPOINT_JY * 10 ** (-0.4 * MAG_FLOOR)
N_OBS = 20


def _predict_jax(theta, times, band_idx=None):
    """An exponential decay clamped at the floor, as the JAX factories clamp their magnitudes."""
    theta = jnp.atleast_1d(theta)
    return jnp.maximum(theta[0] * jnp.exp(-jnp.asarray(times) / theta[1]), FLOOR_JY)


_predict_jax.mag_floor = MAG_FLOOR


def _predict(p, times, bands=None):
    return np.maximum(p["amp"] * np.exp(-np.asarray(times, float) / p["tau"]), FLOOR_JY)


@pytest.fixture(scope="module")
def model():
    # tau down to 0.05 d: a good share of the prior is dark long before the last epoch.
    prior = wp.Prior({"amp": wp.Uniform(1e-3, 1e-2), "tau": wp.LogUniform(0.05, 20.0)})
    return wp.register_model("floor_stats_toy", _predict, ["amp", "tau"], prior=prior,
                             overwrite=True, predict_jax=_predict_jax)


@pytest.fixture(scope="module")
def lc():
    t = np.linspace(0.5, 30.0, N_OBS)
    return wp.LightCurve(time=t, band=np.array(["ztfg"] * N_OBS), flux=5e-3 * np.exp(-t / 6.0),
                         flux_err=np.full(N_OBS, 1e-4), name="floor_stats")


def test_snpe_gpu_reports_the_floored_fraction_of_its_simulations(model, lc):
    pytest.importorskip("sbi")
    n_sim = 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wp.fit(lc, model.name, sampler="snpe_gpu", device="cpu", space="flux", seed=0,
                   num_simulations=n_sim, num_rounds=1, num_samples=100, max_num_epochs=5)
    fs = r.info["mag_floor_stats"]
    assert fs["mag_floor"] == MAG_FLOOR
    # every simulated point was counted (the map's padding rows are not simulations)
    assert fs["n_points"] >= n_sim * N_OBS
    assert 0.10 < fs["fraction"] < 1.0 and fs["n_floored"] == round(fs["fraction"] * fs["n_points"])


@pytest.mark.parametrize("sampler,kw", [
    ("abc_gpu", dict(n_simulations=2000, quantile=0.1)),
    ("abc_smc_gpu", dict(n_particles=300, n_rounds=2)),
])
def test_abc_gpu_samplers_never_report_an_uncounted_zero(model, lc, sampler, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wp.fit(lc, model.name, sampler=sampler, space="flux", seed=0, **kw)
    fs = r.info["mag_floor_stats"]
    assert fs is None or fs["n_points"] > 0, fs
