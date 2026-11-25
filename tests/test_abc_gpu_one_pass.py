"""``abc_gpu`` evaluates each draw once, keeps the draws on the device, and runs float32 where safe.

0.1.1's ``abc_gpu`` transferred every draw to the host and then re-ran the forward model over the
accepted ones, in a second compiled scan, to find the best fit. Now the
sweep returns every draw's exact log-likelihood next to its distance, from the one model call, and
only the accepted rows leave the device. These tests pin down:

* the forward model is traced ONCE per fit (0.1.1 traced it twice: sweep, then rescoring);
* the best fit is still the maximum of the exact likelihood over the accepted draws, recomputed
  independently in numpy, including the scatter-augmented likelihood and magnitude space;
* ``precision="float32"`` runs single precision in any session, ``"float64"`` double, ``None``
  follows the session; float32 agrees with float64 on a kilonova within the S11/S18 gate (best
  ln L within 0.5), and the float64-only supernova and TDE engines are refused by name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.likelihood import make_likelihood  # noqa: E402
from whisper_cbpf.priors import LogUniform, Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax.abc_gpu import ABCGPUSampler  # noqa: E402

N_OBS = 24
PRIOR = Prior({"amp": Uniform(0.5, 2.0), "tau": LogUniform(1.0, 20.0)})
NAMES = list(PRIOR.names)
MODEL = "_abc_gpu_one_pass_toy"
_TOL = 1e-8 if jax.config.jax_enable_x64 else 1e-4


def _predict(parameters, times, bands=None):
    return parameters["amp"] * np.exp(-np.asarray(times, float) / parameters["tau"])


def _predict_jax(theta_2d, times):
    return jax.vmap(lambda th: th[0] * jnp.exp(-times / th[1]))(theta_2d)


@pytest.fixture(scope="module")
def lc():
    rng = np.random.default_rng(0)
    t = np.linspace(0.5, 20.0, N_OBS)
    flux = 1.2 * np.exp(-t / 6.0)
    err = np.full_like(flux, 0.05 * flux.max())
    return wp.LightCurve(time=t, flux=flux + err * rng.standard_normal(N_OBS), flux_err=err,
                         band=np.array(["r"] * N_OBS))


@pytest.fixture(scope="module", autouse=True)
def model():
    wp.register_model(MODEL, _predict, NAMES, prior=PRIOR, overwrite=True)
    return MODEL


def _logls(lc, samples, names=NAMES, **lik_kw):
    lik = make_likelihood(lc, **lik_kw)
    scatter = lik_kw.get("scatter_param")
    out = []
    for r in samples[names].to_dict("records"):
        f = _predict(r, lc.time)
        out.append(lik.log_likelihood(f, sigma_extra=r[scatter]) if scatter
                   else lik.log_likelihood(f))
    return np.asarray(out)


# --------------------------------------------------------------------------- one evaluation
@pytest.mark.parametrize("threshold", [None, 0.0])
def test_the_forward_model_is_traced_once_per_fit(lc, threshold):
    """One trace = one model call per draw, inside the sweep. 0.1.1 traced it a second time for
    the best-fit scan -- a second XLA compile of the model on every fit. ``threshold=0`` covers
    the fit that accepts nothing (its best fit is the closest rejected draw)."""
    calls = []

    def counted(theta_2d, times):
        calls.append(tuple(theta_2d.shape))
        return _predict_jax(theta_2d, times)

    kw = dict(quantile=0.05) if threshold is None else dict(threshold=threshold)
    res = ABCGPUSampler().fit(lc, MODEL, prior=PRIOR, predict_jax=counted, n_simulations=3000,
                              space="flux", seed=0, **kw)
    assert len(calls) == 1, calls
    assert res.info["postprocess_s"] >= 0.0
    if threshold == 0.0:
        assert res.info["n_accepted"] == 0
        assert res.info["best_params_source"] == "closest_rejected_draw"
    # the ln L reported for the best fit is the exact one, whichever draw it is
    best = _logls(lc, pd.DataFrame([res.best_params]), space="flux")[0]
    assert res.max_log_likelihood == pytest.approx(best, rel=0, abs=_TOL)


def test_best_fit_in_magnitude_space_is_the_max_over_accepted_draws(lc):
    res = ABCGPUSampler().fit(lc, MODEL, prior=PRIOR, predict_jax=_predict_jax,
                              n_simulations=6000, quantile=0.2, space="magnitude", seed=4)
    ll = _logls(lc, res.samples, space="magnitude")
    assert res.info["likelihood_space"] == "magnitude"
    assert res.max_log_likelihood == pytest.approx(ll.max(), rel=0, abs=_TOL)
    assert res.best_params == {nm: float(res.samples[nm].iloc[int(np.argmax(ll))])
                               for nm in NAMES}


def test_scatter_likelihood_uses_each_draws_own_scatter(lc):
    prior = Prior({**PRIOR.distributions, "sigma": Uniform(1e-3, 0.2)})
    names = list(prior.names)

    def predict_jax(theta_2d, times):                 # the scatter column is not a model input
        return _predict_jax(theta_2d[:, :2], times)

    with pytest.warns(UserWarning, match="noise scale cannot be"):
        res = ABCGPUSampler().fit(lc, MODEL, prior=prior, predict_jax=predict_jax,
                                  n_simulations=4000, quantile=0.1, space="flux", seed=2,
                                  scatter_param="sigma")
    ll = _logls(lc, res.samples, names=names, kind="gaussian_scatter", space="flux",
                scatter_param="sigma")
    assert res.max_log_likelihood == pytest.approx(ll.max(), rel=0, abs=_TOL)


# --------------------------------------------------------------------------- precision
def test_precision_follows_the_session_by_default(lc):
    res = ABCGPUSampler().fit(lc, MODEL, prior=PRIOR, predict_jax=_predict_jax,
                              n_simulations=1000, quantile=0.1, space="flux")
    session = bool(jax.config.jax_enable_x64)
    assert res.info["x64"] is session and res.info["x64_session"] is session
    assert res.info["precision"] == ("float64" if session else "float32")


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_precision_is_honoured_in_any_session(lc, precision):
    seen = []

    def spy(theta_2d, times):
        seen.append((theta_2d.dtype, times.dtype))
        return _predict_jax(theta_2d, times)

    res = ABCGPUSampler().fit(lc, MODEL, prior=PRIOR, predict_jax=spy, n_simulations=3000,
                              quantile=0.1, space="flux", seed=1, precision=precision)
    assert seen == [(np.dtype(precision), np.dtype(precision))]
    assert res.info["precision"] == precision and res.info["x64"] is (precision == "float64")
    assert res.info["x64_session"] is bool(jax.config.jax_enable_x64)
    # the session is left as it was
    assert jnp.asarray(1.0).dtype == (jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    tol = 1e-8 if precision == "float64" else 1e-4
    ll = _logls(lc, res.samples, space="flux")
    assert res.max_log_likelihood == pytest.approx(ll.max(), rel=0, abs=tol)


def test_unknown_precision_is_refused(lc):
    with pytest.raises(ValueError, match="precision must be None.*'float32' or 'float64'"):
        ABCGPUSampler().fit(lc, MODEL, prior=PRIOR, predict_jax=_predict_jax,
                            n_simulations=100, precision="half")


def _lsst_toy(model, times, truth):
    from whisper_cbpf.io.photometry import AB_ZEROPOINT_JY

    bands = ["lsstg", "lsstr", "lssti"]
    t = np.repeat(times, 3)
    b = np.tile(bands, len(times))
    with jax.enable_x64(True):
        flux = np.asarray(model.predict(truth, t, b), float)
    mag = -2.5 * np.log10(flux / AB_ZEROPOINT_JY)
    mag = mag + 0.08 * np.random.default_rng(0).standard_normal(mag.size)
    return wp.LightCurve(time=t, band=b, magnitude=mag, magnitude_err=np.full(mag.size, 0.08))


@pytest.mark.parametrize("family", ["supernova", "tde"])
def test_float32_is_refused_for_the_float64_only_engines(family):
    from whisper_cbpf.models.jax import supernova_model, tde_model

    bands = ["lsstg", "lsstr", "lssti"]
    if family == "supernova":
        m = supernova_model("arnett", bands, 0.05, 7.1e26, name="_abc_f32_arnett")
    else:
        m = tde_model(bands, 0.05, 7.1e26, rise="gaussian", name="_abc_f32_tde")
    t = np.repeat(np.linspace(2.0, 40.0, 6), 3)
    lc = wp.LightCurve(time=t, band=np.tile(bands, 6), magnitude=np.full(18, 20.0),
                       magnitude_err=np.full(18, 0.1))
    with pytest.raises(ValueError) as err:
        ABCGPUSampler().fit(lc, m, n_simulations=100, precision="float32", space="magnitude")
    msg = str(err.value)
    assert "runs only in float64" in msg and m.name in msg
    assert "precision='float64'" in msg and "kilonovae" in msg
    assert isinstance(err.value.__cause__, RuntimeError)          # the engine's own guard


@pytest.mark.slow
def test_float32_kilonova_agrees_with_float64():
    """float32 changes nothing a user reads. On the float32 run's own accepted draws, rescored in
    float64 numpy: every worst point within 1e-3 sigma of the device's and inside the threshold,
    and the reported best ln L within 0.02 of the float64 maximum over those draws. A float64 run
    of the same rule accepts a Poisson-consistent number (the draws differ: float32 and float64
    uniforms from one key are different numbers)."""
    from whisper_cbpf.distance import max_abs_z_distance
    from whisper_cbpf.models import redback_adapter as ra

    dl = ra.redback_luminosity_distance_cm(0.01, model="one_component_kilonova_model")
    kn = wp.register_kilonova(["lsstg", "lsstr", "lssti"], 0.01, dl, name="_abc_f32_kilonova")
    truth = dict(mej=0.03, vej=0.25, kappa=8.0, temperature_floor=2500.0)
    lc = _lsst_toy(kn, np.linspace(0.3, 8.0, 8), truth)
    # a box around the truth, so a few percent of draws pass (the full prior passes 15 in 40000)
    prior = Prior({"mej": Uniform(0.02, 0.04), "vej": Uniform(0.18, 0.32),
                   "kappa": Uniform(5.0, 11.0), "temperature_floor": LogUniform(1700.0, 3700.0)})
    kw = dict(prior=prior, n_simulations=20000, distance="max_abs_z", threshold=5.0,
              simulate_noise=False, space="magnitude", seed=0)
    r32 = ABCGPUSampler().fit(lc, kn.name, precision="float32", **kw)
    r64 = ABCGPUSampler().fit(lc, kn.name, precision="float64", **kw)
    assert r32.info["precision"] == "float32" and r64.info["precision"] == "float64"
    n32, n64 = r32.info["n_accepted"], r64.info["n_accepted"]
    assert n32 > 20 and n64 > 20
    assert abs(n32 - n64) <= 3.0 * np.sqrt(n32 + n64)

    lik = make_likelihood(lc, space="magnitude")
    with jax.enable_x64(True):
        fluxes = [np.asarray(kn.predict(r, lc.time, lc.band), float)
                  for r in r32.samples[list(kn.parameters)].to_dict("records")]
    d64 = np.array([max_abs_z_distance(lik.y, lik.sigma, lik.model_in_space(f)) for f in fluxes])
    ll64 = np.array([lik.log_likelihood(f) for f in fluxes])
    np.testing.assert_allclose(r32.samples["distance"].to_numpy(), d64, rtol=0, atol=1e-3)
    assert np.all(d64 <= 5.0 + 1e-3)
    assert r32.max_log_likelihood == pytest.approx(ll64.max(), abs=0.02)
