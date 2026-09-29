"""The worst-point ABC rule, ``distance="max_abs_z"``: every point within k sigma.

A real-data follow-up study accepted a draw only if EVERY observation lay within 5 sigma of the model
magnitude (``simulate_noise=False``, magnitude space), through a distance written outside the
package, and the GPU sampler could not use it at all. These tests pin the rule down:

* on the CPU, the accepted set is EXACTLY the set of prior draws whose worst point is within k
  sigma -- rebuilt here from the sampler's documented per-index RNG streams;
* on the GPU, every accepted draw passes the rule when rescored on the CPU, and the accepted count
  agrees with the CPU sampler's within 2 sigma (Poisson) in at least 9 of 10 fits -- the rule
  used to validate the GPU sampler. The two backends use different bit generators, so
  agreement is statistical;
* ABC-SMC accepts by it with its own strict ``d < epsilon``;
* a fit that accepts nothing reports how far the closest draw got, in sigma.

The model is an analytic decay in magnitude space, so the tests are about the rule, not a
photometric model's compile time.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.distance import max_abs_z_distance
from whisper_cbpf.likelihood import GaussianLikelihood
from whisper_cbpf.priors import LogUniform, Prior, Uniform

N_OBS = 24
SCALE_JY = 1e-4                       # amp = 1 is ~18.9 AB mag
PRIOR = Prior({"amp": Uniform(0.5, 2.0), "tau": LogUniform(1.0, 20.0)})
NAMES = list(PRIOR.names)
MODEL = "_abc_worst_point_toy"
RULE = dict(distance="max_abs_z", simulate_noise=False, space="magnitude")


def _predict(parameters, times, bands=None):
    return SCALE_JY * parameters["amp"] * np.exp(-np.asarray(times, float) / parameters["tau"])


@pytest.fixture(scope="module")
def lc():
    t = np.linspace(0.5, 20.0, N_OBS)
    mag = -2.5 * np.log10(_predict({"amp": 1.2, "tau": 6.0}, t) / 3631.0)
    err = np.full(N_OBS, 0.1)
    mag = mag + err * np.random.default_rng(0).standard_normal(N_OBS)
    return wp.LightCurve(time=t, band=np.array(["r"] * N_OBS), magnitude=mag, magnitude_err=err,
                         name="worst_point_toy")


@pytest.fixture(scope="module", autouse=True)
def model():
    wp.register_model(MODEL, _predict, NAMES, prior=PRIOR, overwrite=True)
    return MODEL


def _worst_point(lc, rows):
    """max |z| of each parameter dict, in magnitude space, computed independently of any sampler."""
    lik = GaussianLikelihood(lc, space="magnitude")
    t = np.asarray(lc.time, float)
    return np.array([max_abs_z_distance(lik.y, lik.sigma, lik.model_in_space(_predict(r, t)))
                     for r in rows])


# --------------------------------------------------------------------------- CPU: the exact rule
@pytest.mark.parametrize("k", [3.0, 5.0])
def test_cpu_accepts_exactly_the_draws_within_k_sigma(lc, k):
    n, seed = 4000, 2
    res = wp.fit_ABC(lc, MODEL, prior=PRIOR, n_simulations=n, threshold=k, seed=seed, **RULE)
    draws = [PRIOR.sample(np.random.default_rng([seed, i])) for i in range(n)]
    d = _worst_point(lc, draws)
    inside = d <= k                                           # inclusive, as documented

    assert res.info["n_accepted"] == int(inside.sum()) > 0
    assert res.info["distance"] == "max_abs_z_distance"
    np.testing.assert_array_equal(res.samples["amp"].to_numpy(),
                                  np.array([p["amp"] for p, ok in zip(draws, inside) if ok]))
    np.testing.assert_allclose(res.samples["distance"].to_numpy(), d[inside], rtol=1e-12)
    assert res.min_distance == pytest.approx(d.min(), rel=1e-12)


def test_quantile_keeps_the_draws_with_the_smallest_worst_point(lc):
    res = wp.fit_ABC(lc, MODEL, prior=PRIOR, n_simulations=2000, quantile=0.1, seed=0, **RULE)
    assert res.info["n_accepted"] == 200
    assert res.samples["distance"].max() <= res.info["epsilon"]
    draws = [PRIOR.sample(np.random.default_rng([0, i])) for i in range(2000)]
    assert res.info["epsilon"] == pytest.approx(np.quantile(_worst_point(lc, draws), 0.1))


def test_nothing_within_k_sigma_says_how_close_it_got(lc):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit_ABC(lc, MODEL, prior=PRIOR, n_simulations=500, threshold=0.5, seed=0, **RULE)
    assert res.n_samples == 0
    assert any("accepted 0 of 500" in str(w.message) for w in caught)
    assert res.info["predictive_metrics_skipped"] == "no accepted draws"
    assert res.info["best_params_source"] == "closest_rejected_draw"
    # min_distance is the closest draw's worst point, in sigma: > 0.5 or it would have been kept
    assert 0.5 < res.min_distance < 10.0


# --------------------------------------------------------------------------- GPU against CPU
def _predict_jax(theta_2d, times):
    import jax
    import jax.numpy as jnp

    return jax.vmap(lambda th: SCALE_JY * th[0] * jnp.exp(-times / th[1]))(theta_2d)


def _gpu(lc, **kw):
    pytest.importorskip("jax")
    from whisper_cbpf.samplers.jax.abc_gpu import ABCGPUSampler

    return ABCGPUSampler().fit(lc, MODEL, prior=PRIOR, predict_jax=_predict_jax, **RULE, **kw)


def _k_tol():
    """Rescoring tolerance on k, in sigma: the GPU distance is float32 unless x64 is on (20 mag
    rounds to 2e-6 mag in float32, 2e-5 sigma at these 0.1 mag errors)."""
    jax = pytest.importorskip("jax")
    return 1e-9 if jax.config.jax_enable_x64 else 1e-4


def test_gpu_accepted_draws_pass_the_cpu_rule(lc):
    k, tol = 5.0, _k_tol()
    res = _gpu(lc, n_simulations=6000, threshold=k, seed=1)
    assert res.info["distance"] == "max_abs_z" and res.info["n_accepted"] > 50
    d = _worst_point(lc, res.samples[NAMES].to_dict("records"))
    assert np.all(d <= k + tol), d.max()
    np.testing.assert_allclose(res.samples["distance"].to_numpy(), d, rtol=tol, atol=tol)
    assert res.min_distance == pytest.approx(d.min(), rel=tol, abs=tol)


def test_gpu_counts_match_cpu_within_two_sigma_poisson(lc):
    """On a toy: 10 fits (5 seeds x 2 thresholds), |n_gpu - n_cpu| within 2 sigma of the
    difference of two Poisson counts in at least 9. Both samplers draw the same prior with the same
    rule; only the bit generators differ. (Measured: 10/10 at this budget, counts 48-335; 9/10 at
    20 000 draws, the tenth at -2.9 sigma.)"""
    n = 6000
    within, report = 0, []
    for k in (3.0, 5.0):
        for seed in range(5):
            cpu = wp.fit_ABC(lc, MODEL, prior=PRIOR, n_simulations=n, threshold=k, seed=seed,
                             **RULE).info["n_accepted"]
            gpu = _gpu(lc, n_simulations=n, threshold=k, seed=seed).info["n_accepted"]
            z = (gpu - cpu) / np.sqrt(max(cpu + gpu, 1))
            within += abs(z) <= 2.0
            report.append((k, seed, cpu, gpu, round(float(z), 2)))
    assert min(r[2] for r in report) > 40, report           # counts large enough to mean something
    assert within >= 9, report


def test_gpu_nothing_within_k_sigma_says_how_close_it_got(lc):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _gpu(lc, n_simulations=500, threshold=0.5, seed=0)
    assert res.n_samples == 0
    assert any("accepted 0 of 500" in str(w.message) for w in caught)
    assert res.info["predictive_metrics_skipped"] == "no accepted draws"
    assert res.info["best_params_source"] == "closest_rejected_draw"
    assert 0.5 < res.min_distance < 10.0
    # the reported best is the closest rejected draw, whose worst point IS min_distance
    tol = _k_tol()
    assert _worst_point(lc, [res.best_params])[0] == pytest.approx(res.min_distance,
                                                                    rel=tol, abs=tol)


# --------------------------------------------------------------------------- ABC-SMC
def test_abc_smc_accepts_strictly_below_epsilon_by_the_worst_point(lc):
    sched = [np.inf, 20.0, 6.0]
    cpu = wp.fit_ABC_SMC(lc, MODEL, prior=PRIOR, n_particles=150, epsilon_schedule=sched, seed=0,
                         **RULE)
    assert cpu.n_samples == 150 and cpu.samples["distance"].max() < 6.0
    d = _worst_point(lc, cpu.samples[NAMES].to_dict("records"))
    np.testing.assert_allclose(cpu.samples["distance"].to_numpy(), d, rtol=1e-12)

    tol = _k_tol()
    from whisper_cbpf.samplers.jax.abc_smc_gpu import ABCSMCGPUSampler

    gpu = ABCSMCGPUSampler().fit(lc, MODEL, prior=PRIOR, predict_jax=_predict_jax, n_particles=150,
                                 epsilon_schedule=sched, seed=0, chunk=64, **RULE)
    assert gpu.n_samples == 150 and gpu.samples["distance"].max() < 6.0
    d = _worst_point(lc, gpu.samples[NAMES].to_dict("records"))
    assert np.all(d < 6.0 + tol)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_abc_smc_auto_floor_warns_off_the_chi2_scale(lc, backend):
    """``min_epsilon="auto"`` floors epsilon at best + 2 (k + 2), derived for chi2. On the worst-
    point scale that is 8 sigma above the best draw here, so the schedule silently stops
    tightening; the samplers say so."""
    kw = dict(prior=PRIOR, n_particles=60, n_rounds=2, min_epsilon="auto", seed=0, **RULE)
    with pytest.warns(UserWarning, match="min_epsilon='auto'.*chi2"):
        if backend == "cpu":
            wp.fit_ABC_SMC(lc, MODEL, **kw)
        else:
            pytest.importorskip("jax")
            from whisper_cbpf.samplers.jax.abc_smc_gpu import ABCSMCGPUSampler

            ABCSMCGPUSampler().fit(lc, MODEL, predict_jax=_predict_jax, chunk=64, **kw)
