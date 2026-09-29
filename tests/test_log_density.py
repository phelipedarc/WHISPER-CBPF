"""``log_density``: the public log-posterior of one light curve, with the data as arguments.

The claims, each a test:

1. **It is the density the samplers use.** Value and likelihood equal ``make_log_prob_jax``'s (the
   density ``emcee_jax`` samples) to rounding, for a Gaussian in flux and in magnitude space, upper
   limits, a free scatter, and every prior family including Fixed.
2. **Padding changes nothing.** The same light curve at bucket None, "auto" or 96 gives the same
   numbers, also for the supernova whose value depends on the call's last epoch.
3. **One compile serves many light curves** of one bucket: ``.shared`` is one object and its jit
   cache holds one program after two different light curves.
4. **Falls back, and says so**, for a model that needs concrete epochs and for a float32 session
   on an MJD clock.
5. **Errors name the cause and the fix.**
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.priors import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax import _diagnostics as dg  # noqa: E402
from whisper_cbpf.samplers.jax._adapters import (  # noqa: E402
    bucket_size,
    free_density,
    log_density,
    make_log_prob_jax,
)

TRUTH = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _flare_lc(n=30, t0=10.0, seed=0, upper=None):
    flare = wp.get_model("flare_jax")
    t = np.linspace(0.5, 30.0, n)
    rng = np.random.default_rng(seed)
    flux = flare.predict({**TRUTH, "t0": t0}, t) + rng.normal(0, 0.1, n)
    return wp.LightCurve(time=t, band=["r"] * n, flux=flux, flux_err=np.full(n, 0.1),
                         upper_limit=upper)


def _compare(ld, ref, prior, names, n=40, seed=3):
    """Max |difference| of the posterior and of the likelihood at ``n`` prior draws."""
    draws = dg.prior_draws(prior, names, n, seed)
    lp = np.array([float(ld.fn(d)) for d in draws])
    lp_ref = np.array([float(ref(d)) for d in draws])
    ll = np.array([float(ld.log_likelihood(d)) for d in draws])
    ll_ref = np.array([float(ref.log_likelihood(d)) for d in draws])
    assert np.array_equal(np.isfinite(lp), np.isfinite(lp_ref))
    fin = np.isfinite(lp_ref)
    assert fin.any()
    scale = max(1.0, float(np.max(np.abs(lp_ref[fin]))))
    return (float(np.max(np.abs(lp[fin] - lp_ref[fin]))) / scale,
            float(np.max(np.abs(ll[fin] - ll_ref[fin]))) / scale)


# --- 0. no JAX at import -----------------------------------------------------------------------

def test_the_new_modules_import_without_jax():
    code = ("import sys, importlib;"
            "importlib.import_module('whisper_cbpf.samplers.jax._adapters');"
            "importlib.import_module('whisper_cbpf.samplers.jax.batch');"
            "importlib.import_module('whisper_cbpf.profile');"
            "sys.exit(1 if 'jax' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_bucket_sizes_are_powers_of_two_and_one_and_a_half_times_them():
    assert [bucket_size(n) for n in (0, 1, 16, 17, 24, 25, 33, 48, 49, 64, 65, 97, 200)] == \
        [16, 16, 16, 24, 24, 32, 48, 48, 64, 64, 96, 128, 256]
    for n in range(1, 600):
        b = bucket_size(n)
        assert b >= n and (b <= 16 or b <= 1.5 * n + 1e-9)       # at most a third wasted
    with pytest.raises(ValueError, match=">= 0"):
        bucket_size(-1)


# --- 1. the density the samplers use -----------------------------------------------------------

def test_flux_gaussian_equals_the_samplers_density():
    lc = _flare_lc()
    model = wp.get_model("flare_jax")
    ld = log_density(lc, model)
    ref = make_log_prob_jax(lc, model, include_prior=True)
    d_lp, d_ll = _compare(ld, ref, model.default_prior, ld.names)
    assert d_lp < 1e-12 and d_ll < 1e-12
    assert ld.names == model.parameters and ld.n_data == 30 and ld.bucket == 32
    assert ld.data_as_argument and ld.reason == "" and ld.space == "flux"
    assert ld.likelihood == "GaussianLikelihood"
    np.testing.assert_array_equal(ld.lows, [-2.0, -2.0, -2.0, 0.0])
    np.testing.assert_array_equal(ld.highs, [3.0, 3.0, 4.0, 30.0])


def test_upper_limits_in_flux_space_equal_the_samplers_density():
    upper = np.zeros(30, dtype=bool)
    upper[[0, 1, 2, 28, 29]] = True
    lc = _flare_lc(upper=upper)
    flux = np.asarray(lc["flux"], dtype=float)
    flux[upper] = 0.3                            # a limiting flux on each censored row
    lc["flux"] = flux
    err = np.asarray(lc["flux_err"], dtype=float)
    err[upper] = np.nan                          # what the loader stores for a non-detection
    lc["flux_err"] = err
    model = wp.get_model("flare_jax")
    ld = log_density(lc, model, space="flux", likelihood="upper_limits")
    ref = make_log_prob_jax(lc, model, include_prior=True, space="flux", likelihood="upper_limits")
    d_lp, d_ll = _compare(ld, ref, model.default_prior, ld.names)
    assert d_lp < 1e-12 and d_ll < 1e-12
    assert ld.likelihood == "GaussianLikelihoodWithUpperLimits"
    assert all(np.isfinite(np.asarray(v)).all() for v in ld.data.values())   # no NaN on device


def test_free_scatter_equals_the_samplers_density():
    lc = _flare_lc()
    model = wp.get_model("flare_jax")
    prior = Prior({**model.default_prior.distributions, "sigma": LogUniform(1e-3, 1.0)})
    ld = log_density(lc, model, prior=prior, likelihood="gaussian_scatter")
    assert ld.names == model.parameters + ["sigma"]
    ref = make_log_prob_jax(lc, model, prior, include_prior=True, likelihood="gaussian_scatter",
                            names=ld.names)
    d_lp, d_ll = _compare(ld, ref, prior, ld.names)
    assert d_lp < 1e-12 and d_ll < 1e-12


def test_every_prior_family_and_fixed_equal_the_samplers_density():
    lc = _flare_lc()
    model = wp.get_model("flare_jax")
    prior = Prior({"log_amp": Normal(1.0, 0.5), "log_sigma": TruncatedNormal(0.5, 0.3, -1.0, 2.0),
                   "log_tau": Fixed(1.5), "t0": Uniform(0.0, 30.0)})
    ld = log_density(lc, model, prior=prior)
    assert ld.names == ["log_amp", "log_sigma", "t0"] and ld.fixed == {"log_tau": 1.5}
    assert ld.lows[0] == -np.inf and ld.highs[0] == np.inf
    full = make_log_prob_jax(lc, model, prior, include_prior=True)
    ref, free, _ = free_density(full, prior, model.parameters)
    assert free == ld.names
    free_prior = dg.split_fixed(prior, model.parameters, "test")[0]
    d_lp, d_ll = _compare(ld, ref, free_prior, ld.names)
    assert d_lp < 1e-12 and d_ll < 1e-12


def test_magnitude_space_supernova_with_free_explosion_and_redshift():
    F = pytest.importorskip("whisper_cbpf.models.jax._factories")
    bands = ["lsstg", "lsstr", "lssti"]
    prior = Prior({"t_exp": Uniform(-15.0, 0.0), "redshift": TruncatedNormal(0.1, 0.02, 0.04, 0.2)})
    model = F.supernova_model("arnett", bands, free=["t_exp", "redshift"], prior=prior)
    truth = {"f_nickel": 0.1, "mej": 2.0, "vej": 1e4, "kappa": 0.1, "kappa_gamma": 0.1,
             "temperature_floor": 4000.0, "t_exp": -5.0, "redshift": 0.1}
    t = np.sort(np.random.default_rng(1).uniform(0.0, 30.0, 21))
    b = np.array(bands * 7)
    mag = -2.5 * np.log10(model.predict(truth, t, b) / 3631.0)
    lc = wp.LightCurve(time=t, band=b, magnitude=mag, magnitude_err=np.full(21, 0.05))
    ld = log_density(lc, model)
    assert ld.data_as_argument and ld.space == "magnitude" and ld.bucket == 24
    ref = make_log_prob_jax(lc, model, include_prior=True)
    d_lp, d_ll = _compare(ld, ref, model.default_prior, ld.names, n=12)
    assert d_lp < 1e-12 and d_ll < 1e-12
    # padding repeats the latest epoch, so this model (whose value depends on the last epoch) is
    # unchanged by it -- also when the rows are not in time order
    order = np.random.default_rng(2).permutation(21)
    shuffled = wp.LightCurve(time=t[order], band=b[order], magnitude=mag[order],
                             magnitude_err=np.full(21, 0.05))
    x = np.array([truth[k] for k in ld.names])
    values = [float(log_density(lc_, model, bucket=bk).fn(x))
              for lc_ in (lc, shuffled) for bk in (None, "auto", 96)]
    assert max(values) - min(values) < 1e-9 * abs(values[0]), values
    g = jax.grad(ld.fn)(jnp.asarray(x))
    assert np.all(np.isfinite(np.asarray(g)))


# --- 2 / 3. padding and one compile ------------------------------------------------------------

def test_padding_changes_nothing_and_one_program_serves_two_light_curves():
    model = wp.get_model("flare_jax")
    lc_a, lc_b = _flare_lc(n=30, seed=1), _flare_lc(n=27, t0=14.0, seed=2)
    x = np.array([1.0, 0.5, 1.5, 12.0])
    exact = float(log_density(lc_a, model, bucket=None).fn(x))
    for bk in ("auto", 32, 100):
        assert abs(float(log_density(lc_a, model, bucket=bk).fn(x)) - exact) < 1e-12
    ld_a, ld_b = log_density(lc_a, model), log_density(lc_b, model)
    assert ld_a.shared is ld_b.shared and ld_a.bucket == ld_b.bucket == 32
    before = ld_a.shared._cache_size()
    float(ld_a.fn(x)), float(ld_b.fn(x))
    float(ld_a.shared(x, ld_b.data)[0])
    assert ld_a.shared._cache_size() == max(before, 1)
    assert float(ld_a.fn(x)) != float(ld_b.fn(x))           # and the data really differ
    # usable under jit / vmap / grad
    v = jax.jit(jax.vmap(ld_a.fn))(jnp.asarray(np.stack([x, x + 0.1])))
    assert abs(float(v[0]) - exact) < 1e-12
    assert np.all(np.isfinite(np.asarray(jax.grad(ld_a.fn)(jnp.asarray(x)))))


# --- 4. the fall-backs -------------------------------------------------------------------------

def test_a_model_that_needs_concrete_epochs_is_closed_over_and_says_so():
    import whisper_cbpf.models as M

    def predict_jax(theta, times):
        t = np.asarray(times, dtype=float)                   # refuses a traced array
        return theta[0] + theta[1] * jnp.asarray(t)

    model = M.Model(name="concrete_toy", predict=lambda p, t, b=None: p["a"] + p["b"] * t,
                    parameters=["a", "b"], default_prior=Prior({"a": Uniform(-5, 5),
                                                                "b": Uniform(-1, 1)}),
                    predict_jax=predict_jax)
    t = np.linspace(0, 10, 12)
    lc = wp.LightCurve(time=t, band=["r"] * 12, flux=1.0 + 0.2 * t, flux_err=np.full(12, 0.1))
    ld = log_density(lc, model)
    assert not ld.data_as_argument and "needs concrete epochs" in ld.reason
    assert ld.bucket == 12
    ref = make_log_prob_jax(lc, model, include_prior=True)
    d_lp, d_ll = _compare(ld, ref, model.default_prior, ld.names)
    assert d_lp < 1e-12 and d_ll < 1e-12
    assert "closed over" in repr(ld)


def test_float32_on_an_mjd_clock_closes_the_epochs_over_as_float64():
    jax.config.update("jax_enable_x64", False)
    try:
        model = wp.get_model("flare_jax")
        lc = _flare_lc()
        lc["time"] = np.asarray(lc["time"], dtype=float) + 60000.0
        ld = log_density(lc, model, prior=Prior({**model.default_prior.distributions,
                                                 "t0": Uniform(60000.0, 60030.0)}))
        assert not ld.data_as_argument and "float32" in ld.reason
        assert np.isfinite(float(ld.fn(np.array([1.0, 0.5, 1.5, 60010.0]))))
    finally:
        jax.config.update("jax_enable_x64", True)


# --- 5. errors ---------------------------------------------------------------------------------

def test_errors_name_the_cause_and_the_fix():
    lc = _flare_lc()
    with pytest.raises(ValueError, match="has no predict_jax"):
        log_density(lc, "flare")
    with pytest.raises(ValueError, match="smaller than the light curve"):
        log_density(lc, "flare_jax", bucket=10)
    with pytest.raises(NotImplementedError, match="gaussian_scatter"):
        log_density(lc, "flare_jax", likelihood="mixture")
    with pytest.raises(ValueError, match="no distribution for"):
        log_density(lc, "flare_jax", prior=Prior({"log_amp": Uniform(0, 1)}))
    with pytest.raises(ValueError, match="not in the prior"):
        log_density(lc, "flare_jax", likelihood="gaussian_scatter")
    with pytest.raises(ValueError, match="every parameter is Fixed"):
        log_density(lc, "flare_jax", prior=Prior({k: Fixed(v) for k, v in TRUTH.items()}))
    with pytest.raises(ValueError, match="no data points"):
        log_density(lc.where(time_min=1e9), "flare_jax")
