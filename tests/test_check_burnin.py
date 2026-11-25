"""A burn-in that discards the whole chain is refused by name, before any sampling (bugs.md §1).

``get_chain(discard=burnin)`` returns zero rows when ``burnin >= nsteps``, and the AIC/BIC step that
followed it died in ``np.nanargmax`` with ``ValueError: attempt to get argmax of an empty sequence``,
a message naming neither setting, after the whole chain had been paid for. It is easy to hit: a
quick run lowers ``nsteps`` and leaves ``burnin`` at its default of 1000.
"""
import numpy as np
import pytest

import whisper_cbpf as wp

_REFUSAL = r"burnin=100 discards the whole chain of nsteps=100"


def _lc():
    t = np.linspace(1.0, 20.0, 24)
    return wp.LightCurve(time=t, band=["r"] * 24, flux=3e-5 * np.exp(-t / 8.0),
                         flux_err=np.full(24, 2e-6), name="toy")


def _decay_model():
    """Two parameters, with a ``predict_jax``, so every arm could otherwise sample it."""
    def predict(parameters, times, bands=None):
        return float(parameters["amp"]) * np.exp(-np.asarray(times, float) / float(parameters["tau"]))

    def predict_jax(theta, times, band_idx=None):
        import jax.numpy as jnp
        return theta[0] * jnp.exp(-jnp.asarray(times) / theta[1])

    return wp.register_model(
        "burnin_check_decay", predict, ["amp", "tau"], overwrite=True, predict_jax=predict_jax,
        prior=wp.Prior({"amp": wp.Uniform(1e-6, 1e-4), "tau": wp.Uniform(2.0, 30.0)}))


def test_mcmc_refuses_a_burnin_as_long_as_the_chain():
    with pytest.raises(ValueError, match=_REFUSAL):
        wp.fit_MCMC(_lc(), _decay_model(), nwalkers=8, nsteps=100, burnin=100, seed=0)


def test_the_emcee_jax_arms_refuse_a_burnin_as_long_as_the_chain():
    pytest.importorskip("jax")
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax, fit_emcee_numpy

    lc, model = _lc(), _decay_model()
    with pytest.raises(ValueError, match=_REFUSAL):
        fit_emcee_jax(lc, model, nwalkers=8, nsteps=100, burnin=100, space="flux",
                      walker_chunk=None)
    with pytest.raises(ValueError, match=_REFUSAL):
        fit_emcee_numpy(lc, model, lc.time, lc.flux, lc.flux_err, nwalkers=8, nsteps=100,
                        burnin=100)
