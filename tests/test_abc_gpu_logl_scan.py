"""``abc_gpu`` and ``abc_smc_gpu`` pick the best fit over EVERY accepted draw.

Both used ``max_logl_scan=2000``: ``abc_gpu`` scored the first 2000 accepted draws in acceptance
order, ``abc_smc_gpu`` the first 2000 rows of the *resampled* population (duplicates included). The
seeds below put the best draw past that head, and the reported maximum must match the maximum
recomputed independently with ``make_likelihood`` and the numpy model. The comparison allows for
float32: the sampler's fluxes come from the device, the reference's from numpy in float64.

A cheap analytic forward model is used deliberately, as in ``test_abc_gpu_scan.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.likelihood import make_likelihood  # noqa: E402
from whisper_cbpf.priors import LogUniform, Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax.abc_gpu import ABCGPUSampler  # noqa: E402
from whisper_cbpf.samplers.jax.abc_smc_gpu import ABCSMCGPUSampler  # noqa: E402

N_OBS = 24
PRIOR = Prior({"amp": Uniform(0.5, 2.0), "tau": LogUniform(1.0, 20.0)})
NAMES = list(PRIOR.names)
OLD_CAP = 2000
#: ln L agreement between device fluxes and the float64 numpy reference.
_TOL = 1e-8 if jax.config.jax_enable_x64 else 1e-4


def _predict_jax(theta_2d, times):
    return jax.vmap(lambda th: th[0] * jnp.exp(-times / th[1]))(theta_2d)


def _predict(parameters, times, bands=None):
    return parameters["amp"] * np.exp(-np.asarray(times, float) / parameters["tau"])


@pytest.fixture(scope="module")
def lc():
    rng = np.random.default_rng(0)
    t = np.linspace(0.5, 20.0, N_OBS)
    flux = 1.2 * np.exp(-t / 6.0)
    err = np.full_like(flux, 0.05 * flux.max())
    return wp.LightCurve(time=t, flux=flux + err * rng.standard_normal(N_OBS), flux_err=err,
                         band=np.array(["r"] * N_OBS))


@pytest.fixture(scope="module")
def model():
    wp.register_model("_abc_logl_scan_toy", _predict, NAMES, prior=PRIOR, overwrite=True)
    return "_abc_logl_scan_toy"


def _logls(lc, samples):
    lik = make_likelihood(lc, space="flux")
    return np.array([lik.log_likelihood(_predict(r, lc.time))
                     for r in samples[NAMES].to_dict("records")])


def test_abc_gpu_best_fit_is_the_max_over_every_accepted_draw(lc, model):
    res = ABCGPUSampler().fit(lc, model, prior=PRIOR, predict_jax=_predict_jax,
                              n_simulations=12000, quantile=0.5, space="flux", seed=0)
    ll = _logls(lc, res.samples)                                   # acceptance order
    best = int(np.argmax(ll))
    assert best >= OLD_CAP and ll[best] > ll[:OLD_CAP].max() + 100 * _TOL

    assert res.max_log_likelihood == pytest.approx(ll[best], rel=0, abs=_TOL)
    assert res.best_params == {nm: float(res.samples[nm].iloc[best]) for nm in NAMES}
    assert res.info["logl_scan_n"] == res.info["n_accepted"] == len(ll)
    assert res.info["logl_scan_capped"] is False
    assert res.info["best_params_source"] == "accepted_draws"
    assert res.info["predictive_metrics"]["scatter_param"] is None      # not `distance`


def test_abc_gpu_capped_scan_keeps_the_lowest_distance_draws(lc, model):
    cap = 50
    res = ABCGPUSampler().fit(lc, model, prior=PRIOR, predict_jax=_predict_jax,
                              n_simulations=12000, quantile=0.5, space="flux", seed=0,
                              max_logl_scan=cap)
    assert res.info["logl_scan_n"] == cap and res.info["logl_scan_capped"] is True
    closest = np.argsort(res.samples["distance"].to_numpy(), kind="stable")[:cap]
    ll = _logls(lc, res.samples.iloc[closest])
    assert res.max_log_likelihood == pytest.approx(ll.max(), rel=0, abs=_TOL)


#: Fit seeds tried, in order, for a resample whose best row lies past ``OLD_CAP``. Which seeds do
#: depends on the session's precision (the device draws differ): of seeds 0-15, 11 do in float32
#: and 10 in float64, seed 5 only in float32. So the test takes the first that does, rather than
#: pinning one; ten failing in a row would be a ~1e-5 event, not bad luck.
_SMC_SEEDS = (5, 0, 2, 6, 7, 8, 11, 13, 14, 1)


def test_abc_smc_gpu_best_fit_covers_the_whole_population(lc, model):
    """The returned samples are a resample of the population, so the best of them is a lower
    bound on the population's best; the old head scan of the resample fell below it."""
    for seed in _SMC_SEEDS:
        res = ABCSMCGPUSampler().fit(lc, model, prior=PRIOR, predict_jax=_predict_jax,
                                     n_particles=8000, n_rounds=1, space="flux", seed=seed)
        ll = _logls(lc, res.samples)
        if int(np.argmax(ll)) >= OLD_CAP and ll.max() > ll[:OLD_CAP].max() + 100 * _TOL:
            break                               # the old head scan would have missed the best
    else:
        pytest.fail(f"no seed in {_SMC_SEEDS} put the resample's best row past {OLD_CAP}")

    assert res.max_log_likelihood >= ll.max() - _TOL
    assert res.info["logl_scan_n"] == 8000 and res.info["logl_scan_capped"] is False
    best_ll = make_likelihood(lc, space="flux").log_likelihood(_predict(res.best_params, lc.time))
    assert res.max_log_likelihood == pytest.approx(best_ll, rel=0, abs=_TOL)
