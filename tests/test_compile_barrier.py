"""The band integral compiles in seconds on a GPU, and a zero temperature is zero flux.

Four claims, each a test:

1. **The barrier changes no number.** Every band-integral path holds the Planck array behind
   ``jax.lax.optimization_barrier`` before the band sum (a trace of each model shows it), and the
   magnitudes equal those of the same model with the barrier replaced by the identity to 1e-12 mag
   on prior draws (the arithmetic is the same; the barrier only stops XLA fusing the exponentials
   into the reduction).
2. **A batched evaluation compiles in under a minute on a GPU.** ``jit(vmap(predict_jax))`` over 480
   parameter sets and 480 observations in six bands took ten minutes to compile before the barrier
   and about a second after. Needs a GPU, so it is skipped without one.
3. **A zero temperature is zero flux**, as in redback, so the magnitude is ``mag_floor`` and not NaN:
   in the TDE and kilonova blackbody and in the supernova band integral behind ``shock_cooling``,
   whose photosphere is exactly cold at late times.
4. **The float32 gradient with respect to a free redshift is finite** for the kilonovae: XLA
   reassociated ``(R / dl)^2`` and the backward pass cubed ``dl``, 1e78 against float32's 3.4e38.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.models.cosmology import luminosity_distance_cm  # noqa: E402

BANDS = ["lsstg", "lsstr", "lssti"]
Z = 0.05
DL = float(luminosity_distance_cm(Z))
EPOCHS = np.linspace(2.0, 90.0, 24)
BAND_NAMES = np.array([BANDS[i % 3] for i in range(EPOCHS.size)])


@pytest.fixture(scope="module", autouse=True)
def x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _models():
    """One model per band-integral path: every supernova name (blackbody, cut-off, line and
    synchrotron SEDs, and the models with their own photosphere), the three TDEs and the four
    kilonova families."""
    out = {n: (lambda n=n: wp.supernova_model(n, BANDS, Z, DL, constraint=None))
           for n in wp.supernova_models()}
    out.update({
        "tde": lambda: wp.tde_model(BANDS, Z, DL, constraint=None),
        "tde_fallback": lambda: wp.tde_fallback_model(BANDS, Z, DL),
        "tde_mosfit": lambda: wp.tde_mosfit_model(BANDS, Z, DL),
        "tde_analytical": lambda: wp.tde_analytical_model(BANDS, Z, DL, constraint=None),
        "tde_stream_stream": lambda: wp.tde_stream_stream_model(BANDS, Z, DL),
        "kilonova_one": lambda: wp.kilonova_model(BANDS, Z, DL),
        "kilonova_two": lambda: wp.kilonova_two_model(BANDS, Z, DL),
        "kilonova_three": lambda: wp.kilonova_three_model(BANDS, Z, DL),
        "metzger": lambda: wp.metzger_kilonova_model(BANDS, Z, DL),
        "magnetar_boosted": lambda: wp.magnetar_boosted_kilonova_model(BANDS, Z, DL),
    })
    return out


MODELS = _models()


def _barriers(jaxpr, found):
    """The input shapes of every ``optimization_barrier`` in ``jaxpr`` and the jaxprs inside it."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "optimization_barrier":
            found.extend(tuple(v.aval.shape) for v in eqn.invars if hasattr(v, "aval"))
        stack = list(eqn.params.values())
        while stack:
            p = stack.pop()
            if hasattr(p, "jaxpr"):
                _barriers(p.jaxpr, found)
            elif hasattr(p, "eqns"):
                _barriers(p, found)
            elif isinstance(p, (list, tuple)):
                stack.extend(p)


def _draw(model, seed):
    rng = np.random.default_rng(seed)
    pr = model.default_prior
    return np.array([float(pr.distributions[k].sample(rng)) for k in model.parameters])


def _mags(model, theta, times=EPOCHS):
    bidx = model.predict_jax.band_index(BAND_NAMES)
    flux = np.asarray(jax.jit(lambda th: model.predict_jax(th, times, bidx))(jnp.asarray(theta)))
    return -2.5 * np.log10(flux / 3631.0)


# --- 1. the barrier is there, and changes no number ------------------------------------------------

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_band_integral_holds_the_planck_array_behind_a_barrier(name):
    model = MODELS[name]()
    n_wave = model.predict.__self__.weights.shape[1]
    bidx = model.predict_jax.band_index(BAND_NAMES)
    theta = jnp.asarray(_draw(model, 0))
    closed = jax.make_jaxpr(lambda th: model.predict_jax(th, EPOCHS, bidx))(theta)
    shapes = []
    _barriers(closed.jaxpr, shapes)
    assert any(s[-2:] == (EPOCHS.size, n_wave) for s in shapes), (
        f"{name}: no optimization_barrier on an (observations, wavelengths) array; found {shapes}")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", ["arnett", "kilonova_one"])
def test_the_barrier_changes_no_magnitude(name, monkeypatch):
    """Against the same model with ``optimization_barrier`` replaced by the identity: |dmag| <=
    1e-12 on 12 prior draws (the models are rebuilt, so neither jit cache is shared)."""
    draws = [_draw(MODELS[name](), s) for s in range(12)]
    with_barrier = np.stack([_mags(MODELS[name](), th) for th in draws])
    jax.clear_caches()
    monkeypatch.setattr(jax.lax, "optimization_barrier", lambda x: x)
    without = np.stack([_mags(MODELS[name](), th) for th in draws])
    monkeypatch.undo()
    jax.clear_caches()
    assert np.all(np.isfinite(with_barrier))
    assert np.max(np.abs(with_barrier - without)) <= 1e-12, np.max(np.abs(with_barrier - without))


# --- 2. compile time on a GPU -----------------------------------------------------------------------

def _gpu_or_skip():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("needs a GPU: this is the XLA GPU fusion autotuner")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["arnett", "kilonova_one"])
def test_a_batch_of_480_over_480_observations_compiles_in_under_a_minute_on_a_gpu(name):
    device = _gpu_or_skip()
    bands = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]
    model = (wp.supernova_model("arnett", bands, Z, DL, constraint=None) if name == "arnett"
             else wp.kilonova_model(bands, Z, DL))
    rng = np.random.default_rng(0)
    theta = np.stack([_draw(model, s) for s in range(480)])
    times = np.sort(rng.uniform(1.0, 60.0, 480))
    bidx = model.predict_jax.band_index(np.array([bands[i % 6] for i in range(480)]))
    batched = jax.jit(jax.vmap(lambda th: model.predict_jax(th, times, bidx)))  # times: concrete
    arg = jax.device_put(jnp.asarray(theta), device)
    start = time.perf_counter()
    compiled = batched.lower(arg).compile()
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"{name}: compiling the batch took {elapsed:.0f} s (about 1 s expected)"
    assert np.all(np.isfinite(np.asarray(compiled(arg))))


# --- 3. a zero temperature is zero flux --------------------------------------------------------------

def _filters(model):
    ctx = model.predict.__self__
    return (jnp.asarray(ctx.weights), jnp.asarray(ctx.norms), jnp.asarray(ctx.lam))


def test_a_zero_temperature_tde_blackbody_is_at_mag_floor_not_nan():
    from whisper_cbpf.models.jax import tde

    weights, norms, lam = _filters(MODELS["tde"]())
    temperature = jnp.asarray([0.0, 5000.0, 0.0])
    radius = jnp.asarray([1e14, 1e14, 0.0])
    band = jnp.asarray([0, 1, 2])

    def mags(t):
        return tde.ab_magnitude(t, radius, band, weights, norms, lam, Z, DL, mag_floor=40.0)

    for fn in (mags, jax.jit(mags)):
        out = np.asarray(fn(temperature))
        np.testing.assert_allclose(out[[0, 2]], 40.0, rtol=0, atol=1e-9)
        assert 20.0 < out[1] < 30.0, out
    grad = np.asarray(jax.grad(lambda t: jnp.sum(mags(t)))(temperature))
    assert np.all(np.isfinite(grad))
    flux = np.asarray(tde._flux_nu_tde(temperature, radius, DL, jnp.asarray(5e14), Z))
    assert flux[0] == 0.0 and flux[1] > 0.0


def test_a_zero_temperature_kilonova_blackbody_is_zero_flux_not_nan():
    from whisper_cbpf.models.jax import kilonova as kn

    temperature = jnp.asarray([0.0, 5000.0])
    flux = np.asarray(kn._flux_nu(temperature, jnp.asarray([1e14, 1e14]), DL,
                                  jnp.asarray([5e14, 5e14]), Z))
    assert flux[0] == 0.0 and flux[1] > 0.0
    grad = np.asarray(jax.grad(lambda t: jnp.sum(kn._flux_nu(t, 1e14, DL, 5e14, Z)))(temperature))
    assert np.all(np.isfinite(grad))


@pytest.mark.filterwarnings("ignore")
def test_shock_cooling_at_late_times_is_at_mag_floor_not_nan():
    """The Piro photosphere is exactly cold once ``exp(-t^2 / 2 td^2)`` underflows, 40 diffusion
    times after the peak."""
    from whisper_cbpf.models.jax import supernova as sn

    mass, radius, energy = 1.0, 1e13, 1e51
    td_days = float(sn.shock_cooling(1.0, mass, radius, energy)[3]) / 86400.0
    late = 60.0 * td_days
    _, _, temperature, _ = sn.shock_cooling(jnp.asarray([late * 86400.0]), mass, radius, energy)
    assert float(temperature[0]) == 0.0, "the test needs a photosphere that is exactly cold"
    model = wp.supernova_model("shock_cooling", BANDS, Z, DL, constraint=None)
    theta = {"log10_mass": 0.0, "log10_radius": 13.0, "log10_energy": 51.0, "nn": 10.0,
             "delta": 1.1}
    flux = model.predict(theta, np.array([late, late]), np.array(["lsstg", "lsstr"]))
    assert np.all(np.isfinite(flux))
    np.testing.assert_allclose(-2.5 * np.log10(np.asarray(flux) / 3631.0), 40.0, atol=1e-9)
    weights, norms, lam = _filters(model)
    zero = jnp.zeros(2)
    out = sn.sn_ab_magnitude(None, None, None, None, jnp.asarray([0, 1]), weights, norms, lam, Z,
                             DL, phot=(zero, jnp.full(2, 1e14)))
    assert np.all(np.asarray(out) == 40.0)


# --- 4. the float32 gradient with respect to a free redshift ------------------------------------------

_F32 = textwrap.dedent("""
    import json, numpy as np, jax, jax.numpy as jnp
    import whisper_cbpf as wp
    assert not jax.config.jax_enable_x64
    prior = wp.Uniform(0.001, 0.1)
    makers = {"kilonova_one": wp.kilonova_model, "metzger": wp.metzger_kilonova_model}
    out = {}
    for name, make in makers.items():
        m = make(["lsstg", "lsstr"], free=["redshift"], redshift_prior=prior)
        theta = jnp.asarray([float(np.mean(m.default_prior.distributions[k].bounds))
                             if k != "redshift" else 0.02 for k in m.parameters], jnp.float32)
        t = jnp.asarray([1.0, 2.0, 4.0, 6.0], jnp.float32)
        b = m.predict_jax.band_index(np.array(["lsstg", "lsstr", "lsstg", "lsstr"]))
        g = jax.grad(lambda th: jnp.sum(m.predict_jax(th, t, b)))(theta)
        out[name] = {"parameters": m.parameters, "grad": [float(x) for x in np.asarray(g)]}
    print(json.dumps(out))
""")


def test_float32_gradient_with_respect_to_a_free_redshift_is_finite():
    env = dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS="cpu")
    run = subprocess.run([sys.executable, "-c", _F32], capture_output=True, text=True, env=env,
                         timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    result = json.loads(run.stdout.strip().splitlines()[-1])
    for name, r in result.items():
        grad = np.asarray(r["grad"], dtype=float)
        assert np.all(np.isfinite(grad)), (name, r)
        assert grad[r["parameters"].index("redshift")] != 0.0, (name, r)
