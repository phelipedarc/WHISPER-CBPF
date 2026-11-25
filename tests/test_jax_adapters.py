"""Regression tests for the JAX adapter layer (``samplers/jax/_adapters.py``) and the fixes
that landed with it.

The two risks worth the most here are the ones that produce a **plausible wrong number with no
error**: an auto-built density that quietly adds the prior on top of what NumPyro already
contributes, and a parameter order taken from the prior being fed to a model that expects its own.
Both are tested first and directly.

Most of these need JAX but **not a GPU** — they run on CPU-JAX with the ``[gpu]`` extra installed.
The three at the top need no JAX at all and run on the minimal install and in CI.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest


# --------------------------------------------------------------------- JAX required (CPU is fine)
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_adapters_module_imports_without_pulling_in_jax():
    """Importing the adapter must not import jax — the CPU-only install promise.

    Run in a SUBPROCESS. Purging ``jax`` from ``sys.modules`` in-process and re-importing it leaves
    every already-bound ``jax``/``jnp`` name in this file pointing at a dead module object, which
    fails the rest of the file for a reason that has nothing to do with the code under test.
    """
    import subprocess

    code = ("import sys, importlib;"
            "importlib.import_module('whisper_cbpf.samplers.jax._adapters');"
            "sys.exit(1 if 'jax' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"importing _adapters pulled in jax at module scope (rc={proc.returncode})\n{proc.stderr}")


def _toy_lc(n=24, seed=0):
    """A small flux-space light curve in two bands."""
    import whisper_cbpf as wp

    t = np.linspace(1.0, 30.0, n)
    band = np.array(["ztfg" if i % 2 else "ztfr" for i in range(n)])
    flux = 2.0 * np.exp(-t / 12.0) + 0.5
    rng = np.random.default_rng(seed)
    return wp.LightCurve(time=t, band=band, flux=flux + rng.normal(0, 0.02, n),
                         flux_err=np.full(n, 0.02), name="toy", redshift=0.01)


def _rtol():
    """Tolerance matching the session's JAX precision.

    ``predict`` is numpy float64 while ``predict_jax`` runs at ``float_dtype()``, so in a session
    without ``jax_enable_x64`` the two agree only to float32. Pinning 1e-10 unconditionally would
    make these tests assert the session flag rather than the adapter.
    """
    return 1e-10 if jax.config.jax_enable_x64 else 2e-6


def _toy_model(name="adapter_toy", reversed_prior=False):
    """A 2-parameter analytic model with BOTH ``predict`` and ``predict_jax``.

    Band-independent on purpose, so ``predict_jax`` carries no ``.band_index`` and the
    self-contained branch of the adapter is exercised too.
    """
    import whisper_cbpf as wp

    params = ["amp", "tau"]

    def predict(parameters, times, bands=None):
        t = np.asarray(times, dtype=float)
        return np.asarray(parameters["amp"]) * np.exp(-t / np.asarray(parameters["tau"])) + 0.5

    def predict_jax(theta, times, band_idx=None):
        return theta[0] * jnp.exp(-jnp.asarray(times) / theta[1]) + 0.5

    dists = {"tau": wp.Uniform(4.0, 30.0), "amp": wp.Uniform(0.5, 5.0)} if reversed_prior \
        else {"amp": wp.Uniform(0.5, 5.0), "tau": wp.Uniform(4.0, 30.0)}
    return wp.register_model(name, predict, params, prior=wp.Prior(dists), overwrite=True,
                             predict_jax=predict_jax)


def test_auto_density_is_the_pure_likelihood_not_the_posterior():
    """R1 — the risk that silently tightens every NUTS/PyMC posterior by one extra prior factor.

    ``nuts_gpu``/``pymc_gpu`` contribute the prior through their own sample sites, so the density
    they are handed must be a pure log-likelihood.
    """
    import whisper_cbpf as wp
    from whisper_cbpf.priors import log_prob_jax as prior_log_prob_jax
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    lc, m = _toy_lc(), _toy_model()
    f = make_log_prob_jax(lc, m, space="flux")
    theta = jnp.asarray([2.0, 12.0])

    expected = wp.make_likelihood(lc, space="flux").log_likelihood(
        m.predict({"amp": 2.0, "tau": 12.0}, lc.time, lc.band))
    assert float(f(theta)) == pytest.approx(expected, rel=_rtol())

    # and it must NOT equal the log-posterior
    lp = float(prior_log_prob_jax(m.default_prior, list(m.parameters))(theta))
    assert lp != pytest.approx(0.0), "toy prior is flat with density 0; test cannot discriminate"
    assert float(f(theta)) != pytest.approx(expected + lp, rel=_rtol())


def test_include_prior_adds_exactly_the_prior_and_nothing_else():
    """The emcee path: it samples the density directly, so it needs the posterior."""
    from whisper_cbpf.priors import log_prob_jax as prior_log_prob_jax
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    lc, m = _toy_lc(), _toy_model()
    theta = jnp.asarray([2.0, 12.0])
    fl = make_log_prob_jax(lc, m, space="flux", include_prior=False)
    fp = make_log_prob_jax(lc, m, space="flux", include_prior=True)
    lp = float(prior_log_prob_jax(m.default_prior, list(m.parameters))(theta))
    assert float(fp(theta)) - float(fl(theta)) == pytest.approx(lp, rel=_rtol(), abs=1e-4)
    assert fp.includes_prior is True and fl.includes_prior is False
    # the likelihood twin is carried, so AIC/BIC can be computed from a posterior chain
    assert float(fp.log_likelihood(theta)) == pytest.approx(float(fl(theta)), rel=_rtol())


def test_parameter_order_is_the_callers_not_the_models():
    """R5 — ``abc_gpu`` orders theta by ``prior.names``; ``predict_jax`` expects
    ``model.parameters``. A user-supplied prior in another order would evaluate the wrong
    parameter at the wrong slot, converge, and be wrong with no error."""
    from whisper_cbpf.samplers.jax._adapters import make_batched_predict_jax

    lc = _toy_lc()
    m = _toy_model(name="adapter_toy_rev", reversed_prior=True)
    names = list(m.default_prior.names)
    assert names == ["tau", "amp"] != list(m.parameters), "fixture must exercise a reordering"

    bp = make_batched_predict_jax(lc, m, names=names)
    theta = jnp.asarray([[12.0, 2.0]])                         # (tau, amp) in the caller's order
    expected = m.predict({"amp": 2.0, "tau": 12.0}, lc.time, lc.band)
    assert np.allclose(np.asarray(bp(theta))[0], expected, rtol=_rtol())


def test_batched_predict_matches_predict_row_by_row():
    from whisper_cbpf.samplers.jax._adapters import make_batched_predict_jax

    lc, m = _toy_lc(), _toy_model()
    rows = np.array([[1.0, 6.0], [2.5, 15.0], [4.0, 28.0]])
    got = np.asarray(make_batched_predict_jax(lc, m)(jnp.asarray(rows)))
    for i, (amp, tau) in enumerate(rows):
        assert np.allclose(got[i], m.predict({"amp": amp, "tau": tau}, lc.time, lc.band), rtol=_rtol())


def test_a_different_times_array_is_rejected():
    """R4 — the epochs are closed over (the supernova family compiles its grid from them), so a
    silently-ignored ``times`` argument would fit the wrong times."""
    from whisper_cbpf.samplers.jax._adapters import make_batched_predict_jax

    lc, m = _toy_lc(), _toy_model()
    bp = make_batched_predict_jax(lc, m)
    theta = jnp.asarray([[2.0, 12.0]])
    bp(theta, np.asarray(lc.time))                                    # the same array is fine
    with pytest.raises(ValueError, match="shape"):
        bp(theta, np.linspace(0, 1, 7))
    with pytest.raises(ValueError, match="differs from the array"):
        bp(theta, np.asarray(lc.time) + 1.0)


def test_chunked_batches_equal_unchunked_including_the_padded_tail():
    """R10 — ``chunk`` must stay a performance knob, never a semantic one."""
    from whisper_cbpf.samplers.jax._adapters import make_batched_predict_jax

    lc, m = _toy_lc(), _toy_model()
    theta = jnp.asarray(np.random.default_rng(1).uniform([0.5, 4.0], [5.0, 30.0], size=(7, 2)))
    plain = np.asarray(make_batched_predict_jax(lc, m)(theta))
    for width in (1, 2, 3, 7, 8):                                     # 7 is not a multiple of most
        chunked = np.asarray(make_batched_predict_jax(lc, m, chunk=width)(theta))
        assert np.allclose(plain, chunked, rtol=0, atol=0), f"chunk={width} changed the answer"


def test_chunked_forward_map_compiles_once_not_on_every_call():
    """The chunked path ran ``lax.map`` outside ``jit``, so every call
    traced and compiled the scan again: ``snpe_gpu`` (which always chunks) paid 0.6-7.1 s per
    simulation round on one GPU (float64, B = 1000, arnett / kilonova / TDE) for a kernel it had
    already built. Built once inside ``jax.jit``, a second call of the same shape compiles nothing,
    and a new batch size compiles once."""
    import jax.monitoring as monitoring

    from whisper_cbpf.samplers.jax._adapters import make_batched_predict_jax

    lc, m = _toy_lc(), _toy_model()
    rng = np.random.default_rng(2)
    rows = [rng.uniform([0.5, 4.0], [5.0, 30.0], size=(n, 2)) for n in (7, 7, 5, 5)]
    f = make_batched_predict_jax(lc, m, chunk=3)
    first = np.asarray(f(jnp.asarray(rows[0])))              # compiles
    compiles = []

    def listener(event, duration, **kw):
        if event == "/jax/core/compile/backend_compile_duration":
            compiles[-1] += 1

    monitoring.register_event_duration_secs_listener(listener)
    try:
        out = []
        for r in rows[1:]:
            compiles.append(0)
            out.append(np.asarray(f(jnp.asarray(r))))
    finally:
        monitoring.unregister_event_duration_listener(listener)
    assert compiles[0] == 0, f"{compiles[0]} XLA compile(s) on a repeated call of the same shape"
    assert compiles[2] == 0, "the new batch size compiled again on its second call"
    plain = make_batched_predict_jax(lc, m)
    for r, o in zip(rows[1:], out):
        assert np.array_equal(o, np.asarray(plain(jnp.asarray(r)))), "chunking changed the answer"
    assert first.shape == (7, lc.time.size)


def test_unbounded_vmap_guard_fires_and_names_the_knob():
    """R12 — the trap that has stalled this project's compiles five times."""
    from whisper_cbpf.samplers.jax._adapters import MAX_UNCHUNKED_BATCH, make_batched_predict_jax

    lc, m = _toy_lc(), _toy_model()
    bp = make_batched_predict_jax(lc, m)
    with pytest.raises(ValueError, match="chunk="):
        bp(jnp.zeros((MAX_UNCHUNKED_BATCH + 1, 2)))


def test_density_is_minus_inf_outside_the_box_with_a_finite_gradient():
    """R13 — ``jnp.where`` alone leaves ``0 * NaN`` in the backward pass, which poisons a whole
    NUTS trajectory. The clip-before / where-after construction keeps the cotangent finite."""
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    lc, m = _toy_lc(), _toy_model()
    f = make_log_prob_jax(lc, m, space="flux")
    outside = jnp.asarray([1e6, 1e6])
    assert float(f(outside)) == -np.inf
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(outside))))


def test_a_non_box_prior_is_refused_rather_than_approximated():
    """Substituting a uniform for a non-box prior is a different posterior, not a different
    parameterisation — the same refusal ``nuts_gpu._numpyro_priors`` makes."""
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    class Weird:
        bounds = (0.0, 1.0)

    import whisper_cbpf as wp
    lc, m = _toy_lc(), _toy_model()
    bad = wp.Prior({"amp": Weird(), "tau": wp.Uniform(4.0, 30.0)})
    with pytest.raises(TypeError, match="cannot express prior"):
        make_log_prob_jax(lc, m, bad, space="flux")


def test_a_model_without_predict_jax_is_refused_by_name():
    from whisper_cbpf.samplers.jax._adapters import make_batched_predict_jax

    import whisper_cbpf as wp
    lc = _toy_lc()
    with pytest.raises(ValueError, match="has no predict_jax"):
        make_batched_predict_jax(lc, wp.get_model("gaussian_rise"))


@pytest.mark.parametrize("sampler,kwargs", [
    ("abc_gpu", dict(n_simulations=64, quantile=0.25)),
    ("abc_smc_gpu", dict(n_particles=16, n_rounds=2)),
    ("emcee_jax", dict(nwalkers=8, nsteps=60, burnin=20, thin=2)),
])
def test_wp_fit_reaches_the_gpu_samplers_with_nothing_hand_built(sampler, kwargs):
    """The headline: these five used to raise unless the caller wrote a density or a forward map."""
    import whisper_cbpf as wp

    lc, m = _toy_lc(), _toy_model()
    r = wp.fit(lc, m.name, sampler=sampler, seed=0, **kwargs)
    assert np.isfinite(r.aic) and np.isfinite(r.bic)
    assert r.info["space"] == "flux"
    assert (r.info.get("log_prob_fn") or r.info.get("predict_jax")) == "auto"


def test_an_explicitly_passed_forward_map_still_wins():
    """R11 — the auto-build must fire only where the code used to raise."""
    import whisper_cbpf as wp

    lc, m = _toy_lc(), _toy_model()
    sentinel = {"called": False}

    def my_predict(theta_2d, times=None):
        sentinel["called"] = True
        t = jnp.asarray(lc.time)
        return jax.vmap(lambda th: th[0] * jnp.exp(-t / th[1]) + 0.5)(jnp.asarray(theta_2d))

    r = wp.fit(lc, m.name, sampler="abc_gpu", predict_jax=my_predict, n_simulations=64,
               quantile=0.25, seed=0)
    assert sentinel["called"], "the caller's predict_jax was not used"
    assert r.info["predict_jax"] == "caller"


def test_space_auto_selects_magnitude_for_magnitude_data():
    """B4's other half: ``space=`` used to be inert on the NUTS family and defaulted to flux, so a
    magnitude light curve was scored in flux by every predictive metric."""
    import whisper_cbpf as wp
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    lc = wp.load_lightcurve("tests/data/at2017gfo.csv", redshift=0.0098).select_bands(["r"])
    assert lc.data_mode == "magnitude"
    m = _toy_model()
    assert make_log_prob_jax(lc, m, space="auto").space == "magnitude"
    assert make_log_prob_jax(lc, m, space="flux").space == "flux"


def test_emcee_does_not_downcast_theta_when_x64_is_on():
    """B9 — it hard-cast to ``jnp.float32``, throwing away the precision x64 was enabled for."""
    from whisper_cbpf.samplers.jax._adapters import float_dtype

    was = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        assert float_dtype() == jnp.float64
        jax.config.update("jax_enable_x64", False)
        assert float_dtype() == jnp.float32
    finally:
        jax.config.update("jax_enable_x64", was)
