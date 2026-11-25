"""Every sampler refuses an empty light curve, whatever density it is given.

THE CONTRACT. A light curve with no points is a mistake upstream, not a degenerate fit -- usually
``select_time_window(...)`` applied before ``set_explosion_date(...)``, which compares
days-since-explosion against raw MJD and drops every row. The likelihood constructors have always
refused one, with a message naming that cause, and every sampler that builds its own density goes
through them first.

WHAT WAS LEFT OPEN, MEASURED AT c090254. The four samplers that take a caller-supplied
``log_prob_fn`` (``nuts_gpu``, ``emcee_jax``, ``pymc_jax_gpu_vectorized``,
``pymc_jax_gpu_parallelized``) never build a likelihood on that path, so they ran a complete fit on
zero data and returned it: AIC 4.0 with BIC ``-inf`` from ``nuts_gpu`` and ``emcee_jax``
(``k * log(0)``: the best possible model in any argmin over BIC) and BIC 0.0 from the two PyMC
samplers (``log(max(n, 1))`` in an inline copy of the formula). Same input, two answers, neither of
them an error. ``bugs.md`` §3 had recorded that "every sampler already refuses one"; that held for
the auto-built density only.

Now all of them refuse, before anything is compiled, with the likelihood layer's message
(``samplers.base.check_not_empty``). One test per registered sampler, plus one per sampler that
accepts its own density.
"""
from __future__ import annotations

import numpy as np
import pytest

import whisper_cbpf as wp

#: Modules each sampler needs; the test skips rather than fails without them. Small budgets so a
#: sampler that does NOT refuse finishes quickly and fails the assertion instead of hanging.
SAMPLERS = {
    "abc": ((), dict(n_simulations=200, quantile=0.1)),
    "abc_smc": ((), dict(n_particles=50, n_rounds=2)),
    "mcmc": (("emcee",), dict(nwalkers=8, nsteps=60, burnin=10)),
    "nested": (("dynesty",), dict(nlive=30, maxiter=100)),
    "dynesty": (("dynesty",), dict(nlive=30, maxiter=100)),
    "snpe": (("sbi", "torch"), dict(num_simulations=100, num_rounds=1, num_samples=100)),
    "npe": (("sbi", "torch"), dict(num_simulations=100, num_rounds=1, num_samples=100)),
    "abc_gpu": (("jax",), dict(n_simulations=200, quantile=0.1)),
    "abc_smc_gpu": (("jax",), dict(n_particles=50, n_rounds=2)),
    "emcee_jax": (("jax", "emcee"), dict(nwalkers=8, nsteps=60, burnin=10)),
    "nuts_gpu": (("jax", "numpyro"), dict(num_warmup=8, num_samples=8, num_chains=2)),
    "pymc_jax_gpu_vectorized": (("jax", "numpyro", "pymc"),
                                dict(num_warmup=8, num_samples=8, num_chains=2)),
    "pymc_jax_gpu_parallelized": (("jax", "numpyro", "pymc"),
                                  dict(num_warmup=8, num_samples=8, num_chains=1)),
    "snpe_gpu": (("jax", "sbi", "torch"),
                 dict(num_simulations=100, num_rounds=1, num_samples=100, device="cpu")),
}
#: The samplers that accept a caller-supplied ``log_prob_fn`` -- the path that skipped the check.
OWN_DENSITY = ("nuts_gpu", "emcee_jax", "pymc_jax_gpu_vectorized", "pymc_jax_gpu_parallelized")


def _predict(parameters, times, bands=None):
    t = np.asarray(times, dtype=float)
    return np.asarray(parameters["amp"]) * np.exp(-t / np.asarray(parameters["tau"]))


def _predict_jax(theta, times, band_idx=None):
    import jax.numpy as jnp
    theta = jnp.atleast_2d(theta)
    return theta[:, :1] * jnp.exp(-jnp.asarray(times)[None, :] / theta[:, 1:2])


@pytest.fixture(scope="module")
def model():
    return wp.register_model(
        "empty_lc_toy", _predict, ["amp", "tau"], overwrite=True, predict_jax=_predict_jax,
        prior=wp.Prior({"amp": wp.Uniform(0.5, 5.0), "tau": wp.LogUniform(1.0, 30.0)}))


@pytest.fixture
def empty():
    return wp.LightCurve(time=np.array([]), band=np.array([], dtype="U8"), flux=np.array([]),
                         flux_err=np.array([]), name="empty")


def _needs(sampler):
    for mod in SAMPLERS[sampler][0]:
        pytest.importorskip(mod)


def test_the_table_covers_every_registered_sampler():
    """A sampler registered later must be added above, or it escapes this contract."""
    assert set(wp.list_samplers()) <= set(SAMPLERS), set(wp.list_samplers()) - set(SAMPLERS)


@pytest.mark.parametrize("sampler", sorted(SAMPLERS))
def test_every_sampler_refuses_an_empty_light_curve(sampler, model, empty):
    _needs(sampler)
    with pytest.raises(ValueError, match="no data points"):
        wp.fit(empty, model.name, sampler=sampler, space="flux", seed=0, **SAMPLERS[sampler][1])


@pytest.mark.parametrize("sampler", OWN_DENSITY)
def test_a_caller_supplied_density_does_not_bypass_the_refusal(sampler, model, empty):
    """At c090254 these ran to completion: BIC -inf (nuts_gpu, emcee_jax) or 0.0 (the PyMC pair)."""
    _needs(sampler)
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="no data points"):
        wp.fit(empty, model.name, sampler=sampler, space="flux", seed=0,
               log_prob_fn=lambda th: jnp.asarray(0.0) * th[0], **SAMPLERS[sampler][1])
