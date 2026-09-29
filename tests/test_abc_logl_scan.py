"""ABC's best fit, and the metrics attached to an ABC fit, on the CPU samplers.

Three things these tests pin down:

* **The best fit is chosen over EVERY accepted draw.** ``abc`` and ``abc_smc`` used to score only
  the first 2000 accepted draws by exact log-likelihood, in acceptance order -- an arbitrary subset.
  On real ZTF fits that left ln L_max up to 3.4 low and BIC up to 6.9 high. The toys below are
  seeded so the best accepted draw sits past index 2000, and the reported maximum must equal the
  maximum recomputed independently with ``make_likelihood`` over all of them.
* **ABC's ``distance`` column is a diagnostic, not a scatter term.** ``scatter_param="auto"`` took
  it for one, so every ABC fit reported coverage 1.00 at every level.
* **An ABC fit that accepted nothing says so by name** instead of warning "need at least one array
  to concatenate" from inside the metrics.

The flare model is analytic and cheap on purpose: this is about the sampler's bookkeeping.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from whisper_cbpf import LightCurve, Prior, Uniform, fit_ABC, fit_ABC_SMC
from whisper_cbpf.likelihood import make_likelihood
from whisper_cbpf.metrics._numpy import _resolve_scatter
from whisper_cbpf.models import get_model
from whisper_cbpf.models.flare import flare_flux

MODEL = get_model("flare")
NAMES = list(MODEL.parameters)
OLD_CAP = 2000          # the fixed head the samplers used to scan


@pytest.fixture(scope="module")
def lc():
    truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    n = 30
    times = np.linspace(0.5, 30, n)
    bands = np.array(["r"] * n)
    flux = flare_flux(truth, times, bands)
    err = np.full_like(flux, 0.02 * flux.max())
    noisy = flux + np.random.default_rng(0).normal(0, err)
    return LightCurve(time=times, band=bands, flux=noisy, flux_err=err, name="toy")


def _logls(lc, rows):
    """Exact Gaussian ln L of each parameter dict, computed independently of the sampler."""
    lik = make_likelihood(lc, space="flux")
    times, bands = np.asarray(lc.time, float), np.asarray(lc.band)
    return np.array([lik.log_likelihood(MODEL.predict(r, times, bands)) for r in rows])


# --------------------------------------------------------------------------- 2.1 best-fit scan
def test_abc_best_fit_is_the_max_over_every_accepted_draw(lc):
    res = fit_ABC(lc, "flare", n_simulations=12000, quantile=0.5, seed=1)
    ll = _logls(lc, res.samples[NAMES].to_dict("records"))       # acceptance order
    best = int(np.argmax(ll))
    # The seed is chosen so the old head scan would miss the best draw; guard that it still does.
    assert best >= OLD_CAP and ll[best] > ll[:OLD_CAP].max()

    assert res.max_log_likelihood == pytest.approx(ll[best], rel=0, abs=1e-9)
    assert res.best_params == {nm: float(res.samples[nm].iloc[best]) for nm in NAMES}
    assert res.info["logl_scan_n"] == res.info["n_accepted"] == len(ll)
    assert res.info["logl_scan_capped"] is False
    assert res.info["best_params_source"] == "accepted_draws"


def test_abc_capped_scan_keeps_the_lowest_distance_draws(lc):
    """A cap that binds scores the closest draws, not the first ones accepted."""
    cap = 50
    res = fit_ABC(lc, "flare", n_simulations=12000, quantile=0.5, seed=1, max_logl_scan=cap)
    assert res.info["logl_scan_n"] == cap and res.info["logl_scan_capped"] is True

    closest = np.argsort(res.samples["distance"].to_numpy(), kind="stable")[:cap]
    ll = _logls(lc, res.samples[NAMES].iloc[closest].to_dict("records"))
    assert res.max_log_likelihood == pytest.approx(ll.max(), rel=0, abs=1e-9)
    assert res.best_params == {nm: float(res.samples[nm].iloc[closest[int(np.argmax(ll))]])
                               for nm in NAMES}


def test_abc_smc_best_fit_is_the_max_over_the_whole_population(lc):
    """One round at epsilon = inf accepts the first ``n_particles`` prior draws, each from its
    documented ``default_rng([seed, round, attempt])`` stream, so the population can be rebuilt
    here and scored independently."""
    n_particles, seed = 3000, 0
    res = fit_ABC_SMC(lc, "flare", n_particles=n_particles, n_rounds=1, seed=seed)
    population = [MODEL.default_prior.sample(np.random.default_rng([seed, 0, a]))
                  for a in range(n_particles)]
    # the rebuild is right: every returned (resampled) draw is a member of it
    assert set(res.samples["amplitude"]) <= {p["amplitude"] for p in population}

    ll = _logls(lc, population)
    best = int(np.argmax(ll))
    assert best >= OLD_CAP and ll[best] > ll[:OLD_CAP].max()

    assert res.max_log_likelihood == pytest.approx(ll[best], rel=0, abs=1e-9)
    assert res.best_params == {nm: float(population[best][nm]) for nm in NAMES}
    assert res.info["logl_scan_n"] == n_particles and res.info["logl_scan_capped"] is False


# --------------------------------------------------------------------------- 2.2 nothing accepted
def test_abc_with_no_accepted_draw_skips_the_metrics_by_name(lc):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_ABC(lc, "flare", n_simulations=500, threshold=0.0, seed=0)
    messages = [str(w.message) for w in caught]

    assert res.n_samples == 0
    assert any("accepted 0 of 500" in m for m in messages)          # the sampler's own warning
    assert not any("predictive metrics (WAIC" in m for m in messages)
    assert not any("need at least one array" in m for m in messages)
    assert res.info["predictive_metrics_skipped"] == "no accepted draws"
    assert "predictive_metrics" not in res.info and "predictive_metrics_error" not in res.info
    assert res.info["best_params_source"] == "closest_rejected_draw"

    # AIC/BIC still come from the exact ln L of that closest (rejected) draw.
    ll = _logls(lc, [res.best_params])[0]
    assert res.max_log_likelihood == pytest.approx(ll, rel=0, abs=1e-9)
    assert res.aic == pytest.approx(-2.0 * ll + 2 * len(NAMES))


def test_abc_smc_with_an_empty_population_skips_the_metrics_by_name(lc):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_ABC_SMC(lc, "flare", n_particles=50, n_rounds=1, epsilon_schedule=[0.0],
                          max_attempts_per_round=400, seed=0)
    assert res.n_samples == 0
    assert not any("predictive metrics (WAIC" in str(w.message) for w in caught)
    assert res.info["predictive_metrics_skipped"] == "no accepted draws"


# --------------------------------------------------------------------------- 2.3 distance column
def test_abc_distance_column_is_not_scored_as_scatter(lc):
    res = fit_ABC(lc, "flare", n_simulations=20000, quantile=0.005, seed=0)
    pm = res.info["predictive_metrics"]
    assert pm["scatter_param"] is None
    empirical = [c["empirical"] for c in pm["coverage"]["overall"]]
    # With `distance` (a chi2 of order 60) read as a sigma in Jy, every level covered 1.00;
    # scored against the reported errors the 50 % interval covers 0.83 here.
    assert not all(e == 1.0 for e in empirical), empirical
    assert pm["coverage"]["overall"][0]["nominal"] == 0.5 and empirical[0] < 0.9


def test_abc_genuine_scatter_parameter_is_still_scored(lc):
    """The recorded ``scatter_param`` is what the metrics use, so a real scatter fit keeps it."""
    prior = Prior({**MODEL.default_prior.distributions, "sigma": Uniform(1e-3, 0.5)})
    res = fit_ABC(lc, "flare", prior=prior, n_simulations=3000, quantile=0.02,
                  scatter_param="sigma", seed=0)
    assert res.info["predictive_metrics"]["scatter_param"] == "sigma"


def test_resolve_scatter_auto_ignores_diagnostic_columns():
    params = list(MODEL.parameters)
    assert _resolve_scatter(params + ["distance"], MODEL, None, "auto") == (None, None)
    assert _resolve_scatter(params + ["sigma", "distance"], MODEL, None, "auto") == ("sigma", 3)
    assert _resolve_scatter(params + ["sigma"], MODEL, None, "auto") == ("sigma", 3)
    # an explicit name is still honoured, whatever it is called
    assert _resolve_scatter(params + ["distance"], MODEL, None, "distance") == ("distance", 3)
