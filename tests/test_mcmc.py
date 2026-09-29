"""MCMC sampler (emcee): recovery, reproducibility, data-mode-consistent likelihood, cross-sampler agreement."""
import numpy as np

import whisper_cbpf as wp
from whisper_cbpf.models.flare import flare_flux


def _synthetic(truth, n=40, noise_frac=0.02, seed=0):
    times = np.linspace(0.5, 30, n)
    flux = flare_flux(truth, times, None)
    err = np.full_like(flux, noise_frac * flux.max()) + 1e-9
    noisy = flux + np.random.default_rng(seed).normal(0, err)
    return wp.LightCurve(time=times, band=["r"] * n, flux=noisy, flux_err=err, name="synth")


def test_mcmc_registered():
    assert "mcmc" in wp.list_samplers()


def test_mcmc_recovers_and_reproducible():
    lc = _synthetic({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0})
    r1 = wp.fit_MCMC(lc, "flare", nsteps=2500, burnin=600, thin=2, seed=0)
    assert r1.sampler == "mcmc" and r1.n_samples > 0
    assert abs(r1.summary["amplitude"]["median"] - 5.0) < 1.0      # recovery
    assert np.isfinite(r1.aic) and np.isfinite(r1.max_log_likelihood)
    assert set(r1.best_params) == {"amplitude", "rise_time", "decay_time"}
    assert 0.1 < r1.info["mean_acceptance_fraction"] < 0.9          # healthy sampling
    r2 = wp.fit_MCMC(lc, "flare", nsteps=2500, burnin=600, thin=2, seed=0)
    assert r1.samples.equals(r2.samples)                           # reproducible (fixed seed)
    import json
    json.loads(r1.to_json())


def test_mcmc_uses_data_mode_for_likelihood_space():
    """MCMC reuses the shared likelihood, so magnitude data is fit in magnitude space (like the others)."""
    flc = _synthetic({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0})
    mlc = flc.add_mag()                                            # now has a magnitude column
    mlc.meta["data_mode"] = "magnitude"
    r = wp.fit_MCMC(mlc, "flare", nsteps=800, burnin=200, thin=2, seed=0)
    assert r.info["space"] == "magnitude"


def test_mcmc_agrees_with_abc():
    """Sanity check (miniature): ABC and MCMC reach approximately the same posterior on the same data."""
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=30)
    rm = wp.fit_MCMC(lc, "flare", nsteps=3000, burnin=800, thin=3, seed=0)
    ra = wp.fit_ABC(lc, "flare", n_simulations=120_000, quantile=0.004, n_jobs=4, seed=0)
    for p in ("amplitude", "rise_time", "decay_time"):
        m, a = rm.summary[p]["median"], ra.summary[p]["median"]
        assert abs(m - a) < 0.5 * abs(m) + 0.5                     # within ~50% (ABC is broader/approx)


# ------------------------------------------------ U3 / 2.7: stuck walkers, the largest tau, spawn
def _bump_lc(i):
    """A Step 0 bump mock (tests/data/u3_bump_mocks.json) and a numpy bump model for `mcmc`."""
    import json
    import os

    s = json.load(open(os.path.join(os.path.dirname(__file__), "data",
                                     "u3_bump_mocks.json")))["sims"][str(i)]
    prior = wp.Prior({"A": wp.LogUniform(0.3, 10.0), "t0": wp.Uniform(0.0, 30.0),
                      "w": wp.Uniform(1.0, 8.0)})
    wp.register_model("u3_bump_numpy", _bump_numpy, ["A", "t0", "w"], prior=prior,
                      overwrite=True)
    lc = wp.LightCurve(time=np.asarray(s["t_rel"]), band=np.asarray(s["bands"]),
                       flux=np.asarray(s["y"]), flux_err=np.asarray(s["sig"]))
    return lc, s


def _bump_numpy(parameters, times, bands=None):
    t = np.asarray(times, dtype=float)
    return float(parameters["A"]) * np.exp(-0.5 * ((t - float(parameters["t0"]))
                                                  / float(parameters["w"])) ** 2)


def test_mcmc_does_not_call_a_stuck_walker_converged():
    """Bump sim 13, seed 0: walker 0 never leaves the zero-signal optimum, the pooled t0 is 17 sd
    off, and the autocorrelation-only rule (nsteps >= 50 x the MEAN tau) said converged=True.
    ``init="prior"`` with walkers moving in linear coordinates is the setting that stranded it
    (the defaults are now the prior scan and log coordinates for the LogUniform amplitude)."""
    import warnings

    lc, s = _bump_lc(13)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = wp.fit_MCMC(lc, "u3_bump_numpy", space="flux", seed=0, init="prior",
                        walker_coordinates="linear")
    assert r.info["stuck_walkers"] == [0]
    assert r.info["converged"] is False
    assert any("stuck" in p for p in r.info["convergence_problems"])
    assert any("not converged" in str(w.message) for w in caught)
    assert r.info["max_autocorr_time"] >= r.info["mean_autocorr_time"]
    # the same start, walkers moving in log(A): every walker reaches the peak
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wp.fit_MCMC(lc, "u3_bump_numpy", space="flux", seed=0, init="prior")
    assert r.info["walker_coordinates"] == {"coordinates": "own", "log": ["A"]}
    assert r.info["stuck_walkers"] == []


def test_mcmc_convergence_uses_the_largest_autocorrelation_time():
    from whisper_cbpf.samplers.jax._diagnostics import walker_health

    lp = np.zeros((8, 100))
    ok = walker_health(lp, [20.0, 30.0], nsteps=2000)
    assert ok["converged"] is True and ok["max_autocorr_time"] == 30.0
    # mean tau 35 passes 50 x tau at nsteps=2000; the largest (50) does not
    slow = walker_health(lp, [20.0, 50.0], nsteps=2000)
    assert slow["converged"] is False and "largest" in slow["convergence_problems"][0]
    unknown = walker_health(lp, [np.nan, np.nan], nsteps=2000)
    assert unknown["converged"] is False                    # unknown is not a pass


def test_mcmc_workers_are_spawned_and_give_the_serial_answer():
    """2.7: a forked pool under a live XLA runtime hung for 48 min at 0 % CPU.
    The pool is spawned now; emcee draws in the parent, so the chain is identical to a serial one."""
    import pytest

    jnp = pytest.importorskip("jax.numpy")
    jnp.ones(3).sum().block_until_ready()             # a live XLA runtime in the parent, as reported
    from whisper_cbpf.samplers import mcmc as mcmc_module

    assert mcmc_module._MP_CONTEXT.get_start_method() == "spawn"
    lc = _synthetic({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, n=20)
    serial = wp.fit_MCMC(lc, "flare", nsteps=300, burnin=100, thin=2, seed=1)
    pooled = wp.fit_MCMC(lc, "flare", nsteps=300, burnin=100, thin=2, seed=1, n_jobs=2)
    assert pooled.samples.equals(serial.samples)
