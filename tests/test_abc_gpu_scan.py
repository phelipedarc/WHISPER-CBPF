"""`chunk` must be a performance knob, never a semantic one.

``abc_gpu`` runs its sweep as a single ``lax.scan`` over blocks of ``chunk`` simulations, and two
details of that arrangement are load-bearing:

* The PRNG key is ``fold_in(base_key, GLOBAL simulation index)``, so draw *i* depends only on
  ``(seed, i)`` and not on how the run was blocked. Fold in a *within-block* index instead and every
  result silently becomes a function of a tuning parameter -- the kind of bug that produces
  irreproducible science and no error message.
* The final block is PADDED up to ``chunk`` (a ragged block would be a second XLA shape and so a
  second compile, which on a photometric model costs far more than a few wasted draws). Those padded
  draws must be discarded before the acceptance quantile, or ``epsilon`` depends on the padding.

A cheap analytic forward model is used deliberately: this test is about the sampler's bookkeeping,
and a photometric model would spend minutes in XLA proving nothing extra.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf  # noqa: E402,F401  registers abc_gpu
import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.priors import LogUniform, Prior, Uniform  # noqa: E402

N_OBS = 24
PRIOR = Prior({"amp": Uniform(0.5, 2.0), "tau": LogUniform(1.0, 20.0)})
NAMES = list(PRIOR.names)


def _decay(theta, times):
    return theta[0] * jnp.exp(-times / theta[1])


def _predict_jax(theta_2d, times):
    return jax.vmap(lambda th: _decay(th, times))(theta_2d)


@pytest.fixture(scope="module")
def lc():
    rng = np.random.default_rng(0)
    t = np.linspace(0.5, 20.0, N_OBS)
    flux = 1.2 * np.exp(-t / 6.0)
    err = np.full_like(flux, 0.05 * flux.max())
    return wp.LightCurve(time=t, flux=flux + err * rng.standard_normal(N_OBS), flux_err=err,
                         band=np.array(["r"] * N_OBS))


@pytest.fixture(scope="module")
def model(lc):
    def predict(parameters, times, bands=None):
        th = jnp.asarray([float(parameters[k]) for k in NAMES])
        return np.asarray(_decay(th, jnp.asarray(np.asarray(times, float))), dtype=float)

    wp.register_model("_abc_scan_toy", predict, NAMES, prior=PRIOR, overwrite=True)
    return "_abc_scan_toy"


def _fit(lc, model, n_simulations, chunk, seed=0):
    return whisper_cbpf.samplers.jax.abc_gpu.ABCGPUSampler().fit(
        lc, model, prior=PRIOR, predict_jax=_predict_jax, n_simulations=n_simulations,
        quantile=0.05, distance="chi2", space="flux", seed=seed, chunk=chunk, max_logl_scan=1)


#: Tolerance for quantities that pass through the distance reduction.
#:
#: Parameters are exactly invariant to `chunk`; epsilon is not, and cannot be. It is a quantile over
#: a 24-term (here) sum whose fusion XLA chooses based on the vmap width, and floating-point
#: addition is not associative -- so a different chunk gives a different summation order and a
#: last-bit-different distance. Measured on the real two-component kilonova in float64: max 7.95e-16
#: relative, ~3.6 ULPs, with the accepted set unchanged. The bound therefore has to track the
#: working precision rather than being a fixed constant.
_EPS_RTOL = 1e-11 if jax.config.jax_enable_x64 else 1e-5


@pytest.mark.parametrize("chunk", [7, 16, 64, 300])
def test_results_are_invariant_to_chunk(lc, model, chunk):
    """Same seed, same budget, different blocking -> the same posterior.

    The chunk values cover every awkward case: 7 and 64 do NOT divide 300, so the final block is
    padded; 300 equals the budget (one block, no padding at all); 16 is the default.
    """
    ref = _fit(lc, model, 300, 16)
    got = _fit(lc, model, 300, chunk)

    # epsilon comes from a quantile over ALL distances, so padded draws leaking into the acceptance
    # step would move it by far more than reduction-order rounding.
    assert got.info["epsilon"] == pytest.approx(ref.info["epsilon"], rel=_EPS_RTOL)
    # A draw sitting within rounding of the acceptance boundary can legitimately flip in or out.
    assert abs(got.info["n_accepted"] - ref.info["n_accepted"]) <= 1

    for nm in NAMES:
        a = np.sort(np.asarray(ref.samples[nm], float))
        b = np.sort(np.asarray(got.samples[nm], float))
        # Parameters are a short deterministic function of the PRNG key, never touched by the
        # reduction, so every draw common to both runs must match EXACTLY -- no tolerance. Only the
        # membership of the accepted set may differ, and only by a boundary draw.
        common = np.intersect1d(a, b)
        assert common.size >= min(a.size, b.size) - 1, (
            f"{nm}: chunk={chunk} shares only {common.size} of {a.size} draws with chunk=16 -- "
            f"that is a semantic change, not a boundary flip")


def test_chunk_larger_than_budget_does_not_overrun(lc, model):
    """A block wider than the whole run must still yield exactly n_simulations draws."""
    res = _fit(lc, model, 50, 4096)
    assert res.info["n_simulations"] == 50
    # quantile 0.05 of 50 draws; the padded 4046 must never reach the acceptance step
    assert 1 <= res.info["n_accepted"] <= 50
    assert len(res.samples) == res.info["n_accepted"]


def test_seed_controls_the_draws(lc, model):
    """Different seed -> different draws; same seed -> identical. Guards the fold_in wiring."""
    a = _fit(lc, model, 300, 16, seed=0)
    b = _fit(lc, model, 300, 16, seed=0)
    c = _fit(lc, model, 300, 16, seed=1)
    assert np.array_equal(np.sort(np.asarray(a.samples["amp"], float)),
                          np.sort(np.asarray(b.samples["amp"], float)))
    assert a.info["epsilon"] != c.info["epsilon"]


def test_default_chunk_is_the_re_measured_width(lc, model):
    """16 was sized for a compile time the current models no longer have, and it made
    a latency-bound model pay its per-call cost 16 simulations at a time (the TDE at n_time=5000:
    1.6 ms a simulation against 0.10 ms at 250). 250 is not a power of two on purpose: the
    kilonovae compile in seconds at 250 and in minutes at 256 or 1024. ``abc_smc_gpu`` shares it."""
    from whisper_cbpf.samplers.jax import abc_gpu, abc_smc_gpu

    assert abc_gpu.DEFAULT_CHUNK == 250
    assert abc_smc_gpu.DEFAULT_CHUNK == abc_gpu.DEFAULT_CHUNK
    res = abc_gpu.ABCGPUSampler().fit(lc, model, prior=PRIOR, predict_jax=_predict_jax,
                                      n_simulations=600, quantile=0.05, space="flux", seed=0)
    assert res.info["chunk"] == 250 and res.info["n_simulations"] == 600
