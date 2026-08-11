"""MCMC sampler (emcee): recovery, reproducibility, data-mode-consistent likelihood, cross-sampler agreement."""
import numpy as np

import whisper_labia as wp
from whisper_labia.models.flare import flare_flux


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
