"""``abc_smc_gpu`` must reproduce whisper's ``abc_smc``, not merely run.

The SMC recursion is the part of this algorithm that fails quietly. Drop the importance weights, or
get the perturbation kernel's covariance wrong, and the population still converges to something that
looks like a posterior — just not the right one. So these tests check the things that would be
silently wrong rather than the things that would raise:

* the population actually contracts round over round (epsilon falls, best distance falls);
* the recovered posterior agrees with the CPU sampler's on a problem with a known answer;
* acceptance is STRICT (``d < epsilon``), which is where ABC-SMC differs from rejection ABC and
  where a copy-paste from ``abc_gpu`` would introduce an off-by-one-particle bug;
* a particle perturbed outside the prior is rejected rather than clipped back into the box, which
  would pile probability mass onto the boundary.

A cheap analytic model is used on purpose — this is about the sampler's bookkeeping, and a
photometric model would spend the whole test budget in XLA.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf  # noqa: E402,F401  registers abc_smc_gpu
import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.priors import LogUniform, Prior, Uniform  # noqa: E402

N_OBS = 24
TRUE = {"amp": 1.20, "tau": 6.0}
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
    flux = TRUE["amp"] * np.exp(-t / TRUE["tau"])
    err = np.full_like(flux, 0.03 * flux.max())
    return wp.LightCurve(time=t, flux=flux + err * rng.standard_normal(N_OBS), flux_err=err,
                         band=np.array(["r"] * N_OBS))


@pytest.fixture(scope="module")
def model(lc):
    def predict(parameters, times, bands=None):
        th = jnp.asarray([float(parameters[k]) for k in NAMES])
        return np.asarray(_decay(th, jnp.asarray(np.asarray(times, float))), dtype=float)

    wp.register_model("_abcsmc_toy", predict, NAMES, prior=PRIOR, overwrite=True)
    return "_abcsmc_toy"


def _fit_gpu(lc, model, **kw):
    opts = dict(n_particles=120, n_rounds=3, quantile=0.5, distance="chi2",
                space="flux", seed=0, chunk=32)
    opts.update(kw)
    return whisper_cbpf.samplers.jax.abc_smc_gpu.ABCSMCGPUSampler().fit(
        lc, model, prior=PRIOR, predict_jax=_predict_jax, **opts)


def test_population_contracts_over_rounds(lc, model):
    """Epsilon and the best distance must both fall — that IS the algorithm."""
    res = _fit_gpu(lc, model)
    rounds = res.info["rounds"]
    assert len(rounds) == 3
    eps = [r["epsilon"] for r in rounds]
    best = [r["best_distance"] for r in rounds]

    # Round 0 runs at epsilon=inf (recorded as None) so every draw is accepted; later rounds tighten.
    later = [e for e in eps if e is not None]
    assert later == sorted(later, reverse=True), f"epsilon did not decrease: {eps}"
    assert best[-1] <= best[0] + 1e-9, f"best distance got worse: {best}"
    assert all(r["n_accepted"] == 120 for r in rounds), [r["n_accepted"] for r in rounds]
    # A degenerate population (all weight on one particle) means the weight recursion is broken.
    assert rounds[-1]["effective_sample_size"] > 5.0, rounds[-1]


def test_agrees_with_the_cpu_abc_smc(lc, model):
    """Same problem, same budget, different backend and different bit generator.

    numpy's Philox and JAX's threefry give different draws from the same seed, so this is a
    statistical comparison: the two posteriors must agree to within their own Monte-Carlo error.
    """
    gpu = _fit_gpu(lc, model)
    cpu = wp.fit_ABC_SMC(lc, model, prior=PRIOR, n_particles=120, n_rounds=3, quantile=0.5,
                        space="flux", seed=0, n_jobs=1)
    worst = 0.0
    for nm in NAMES:
        a = np.asarray(gpu.samples[nm], float)
        b = np.asarray(cpu.samples[nm], float)
        mc = np.hypot(a.std() / np.sqrt(a.size), b.std() / np.sqrt(b.size))
        shift = abs(np.median(a) - np.median(b)) / mc if mc > 0 else 0.0
        worst = max(worst, shift)
    assert worst < 5.0, f"worst median shift {worst:.2f} MC sigma between backends"


def test_acceptance_is_strict(lc, model):
    """``d < epsilon``, not ``d <= epsilon`` — the one place ABC-SMC differs from rejection ABC.

    Driven with an explicit schedule so epsilon is known exactly, then every accepted distance must
    be strictly below it. With an inherited ``<=`` a draw landing on the threshold slips through.
    """
    res = _fit_gpu(lc, model, epsilon_schedule=[np.inf, 5000.0, 500.0])
    d = np.asarray(res.samples["distance"], float)
    assert np.all(d < 500.0), f"accepted a distance >= epsilon: max {d.max()}"


def test_perturbed_particles_stay_inside_the_prior(lc, model):
    """Out-of-support proposals must be REJECTED, never clipped.

    Clipping would pile mass on the boundary and quietly bias every marginal; rejection is what
    abc_smc.py does (`if not np.isfinite(prior.log_prob(theta)): continue`).
    """
    res = _fit_gpu(lc, model, n_rounds=4)
    for nm in NAMES:
        v = np.asarray(res.samples[nm], float)
        lo, hi = PRIOR.distributions[nm].bounds
        assert v.min() >= lo and v.max() <= hi, f"{nm} outside [{lo}, {hi}]"
        # A clipping bug shows up as a spike exactly ON a bound, not merely near it.
        assert np.mean(np.isclose(v, lo)) < 0.05 and np.mean(np.isclose(v, hi)) < 0.05, (
            f"{nm}: suspicious pile-up on a prior bound (clipping instead of rejecting?)")
