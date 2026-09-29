"""What ``wp.fit`` does with no argument beyond the data and the model (release 0.2.0).

* Upper limits are fitted by default: ``space="auto"`` picks flux space for a light curve with
  limits, the censored likelihood takes their significance from ``lc.meta["upper_limit_sigma"]``
  (5 for the survey presets), else 5. ABC and SNPE refuse limits by name.
* ``fit(..., likelihood_max_opt=True)`` keeps the likelihood peak in
  ``info["likelihood_max_opt"]``; ``result.likelihood_max_opt`` keeps it too, on the rows and in
  the box the fit used.
* ``result.forecast`` delegates to ``whisper_cbpf.forecast``.
* The convergence report skips a ``Fixed`` parameter's constant column.
* ``sampler="auto"`` picks a likelihood sampler (the rule ``compare`` uses); the fitted Model is
  kept on the result, so a model that is not registered is scored and found again; a sampler's
  warning points at the caller's line.
"""
from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.io.photometry import flux_density_to_mag
from whisper_cbpf.likelihood import (
    DEFAULT_UPPER_LIMIT_SIGMA,
    GaussianLikelihoodWithUpperLimits,
    make_likelihood,
    resolve_space,
)
from whisper_cbpf.priors import Fixed, LogUniform, Prior, Uniform


def _decay(params, times, bands=None):
    """Flux density [Jy] of an exponential decline from day 0."""
    t = np.asarray(times, dtype=float)
    return np.where(t > 0.0, params["amp"] * np.exp(-t / params["tau"]), 0.0)


MODEL = wp.register_model("fit_defaults_decay", _decay, ["amp", "tau"],
                          prior=Prior({"amp": LogUniform(1e-6, 1e-3), "tau": Uniform(2.0, 40.0)}),
                          overwrite=True)
TRUTH = {"amp": 1e-4, "tau": 10.0}


def _alert_lc(sigma_meta=None):
    """Magnitudes with 0.05 mag errors, and three late non-detections at 21.5 mag."""
    t_det = np.array([1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 15.0])
    mag = flux_density_to_mag(_decay(TRUTH, t_det))
    t = np.concatenate([t_det, [25.0, 30.0, 40.0]])
    ul = np.concatenate([np.zeros(t_det.size, bool), np.ones(3, bool)])
    lc = wp.LightCurve(time=t, band=["lsstr"] * t.size,
                       magnitude=np.concatenate([mag, [21.5, 21.5, 21.5]]),
                       magnitude_err=np.where(ul, np.nan, 0.05), upper_limit=ul)
    if sigma_meta is not None:
        lc.meta["upper_limit_sigma"] = sigma_meta
    return lc


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


# ------------------------------------------------------------------ upper limits by default
def test_auto_space_is_flux_for_a_light_curve_with_limits():
    lc = _alert_lc()
    assert resolve_space(lc, "auto") == "flux"
    assert resolve_space(lc.where(upper_limit=False), "auto") == "magnitude"
    assert resolve_space(lc, "magnitude") == "magnitude"          # explicit choice still honoured
    lik = make_likelihood(lc)
    assert isinstance(lik, GaussianLikelihoodWithUpperLimits) and lik.space == "flux"
    assert lik.upper_limit_sigma == DEFAULT_UPPER_LIMIT_SIGMA == 5.0
    assert make_likelihood(_alert_lc(sigma_meta=3.0)).upper_limit_sigma == 3.0
    assert make_likelihood(_alert_lc(3.0), upper_limit_sigma=4.0).upper_limit_sigma == 4.0


def test_an_explicit_magnitude_space_with_limits_is_still_refused():
    with pytest.raises(ValueError, match=r"flux-only.*space='auto' \(the default; flux for a light "
                                         r"curve with upper limits\)"):
        make_likelihood(_alert_lc(), space="magnitude")
    with pytest.raises(ValueError, match="upper_limit_sigma must be finite and > 0"):
        make_likelihood(_alert_lc(sigma_meta=0.0))


def test_a_fit_uses_the_limits_with_no_argument():
    lc = _alert_lc(sigma_meta=5.0)
    res = _quiet(wp.fit, lc, MODEL, sampler="mcmc", nwalkers=8, nsteps=400, burnin=100, seed=0)
    assert res.info["space"] == "flux"
    assert res.info["likelihood"] == "GaussianLikelihoodWithUpperLimits"
    assert res.n_data == 10
    # The fit scored the censored density at 5 sigma: re-scoring its best draw agrees.
    lik = make_likelihood(lc)
    best = lik.log_likelihood(MODEL.predict(res.best_params, np.asarray(lc.time)))
    assert best == pytest.approx(res.max_log_likelihood, rel=1e-9, abs=1e-9)


def test_the_limits_constrain_the_fit():
    """A decline too slow for the late limits is ruled out; without them it is allowed."""
    lc = _alert_lc()
    slow = {"amp": 1e-4, "tau": 35.0}
    lik = make_likelihood(lc)
    det = make_likelihood(lc.where(upper_limit=False).add_flux(), space="flux")
    t, t_det = np.asarray(lc.time), np.asarray(lc.where(upper_limit=False).time)
    gap_with = lik.log_likelihood(MODEL.predict(TRUTH, t)) - lik.log_likelihood(
        MODEL.predict(slow, t))
    gap_without = det.log_likelihood(MODEL.predict(TRUTH, t_det)) - det.log_likelihood(
        MODEL.predict(slow, t_det))
    assert gap_with > gap_without + 10.0


def test_a_survey_preset_light_curve_is_fitted_with_its_5_sigma_limits():
    src = [{"midpointMjdTai": 61000.0 + d, "band": "r", "psfFlux": f, "psfFluxErr": 60.0}
           for d, f in ((1.0, 2000.0), (3.0, 1500.0), (6.0, 900.0))]
    fp = [{"midpointMjdTai": 60995.0, "band": "r", "psfFlux": 10.0, "psfFluxErr": 60.0},
          {"midpointMjdTai": 61015.0, "band": "r", "psfFlux": 10.0, "psfFluxErr": 60.0}]
    lc = wp.load_lightcurve(src, survey="lsst", limits=fp, redshift=0.1)
    assert lc.meta["upper_limit_sigma"] == 5.0
    lik = make_likelihood(lc)
    assert lik.space == "flux" and lik.upper_limit_sigma == 5.0
    assert lik.summary()["upper_limits"] == 2


@pytest.mark.parametrize("sampler", ["abc", "abc_smc"])
def test_abc_refuses_limits_by_name_and_fits_the_detections(sampler):
    lc = _alert_lc()
    budget = (dict(n_simulations=300, quantile=0.1) if sampler == "abc"
              else dict(n_particles=40, n_rounds=2))
    with pytest.raises(ValueError, match=rf"{sampler} cannot use upper limits"):
        wp.fit(lc, MODEL, sampler=sampler, **budget)
    res = _quiet(wp.fit, lc.where(upper_limit=False), MODEL, sampler=sampler, **budget)
    assert res.n_data == 7


# ------------------------------------------------------------------- likelihood_max_opt
def _flare_lc(n=30):
    t = np.linspace(0.5, 30.0, n)
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
                                          "decay_time": 15.0}, t)
    return wp.LightCurve(time=t, band=["r"] * n, flux=flux, flux_err=np.full(n, 0.1))


def test_fit_likelihood_max_opt_true_keeps_the_peak_and_false_does_not():
    lc = _flare_lc()
    plain = _quiet(wp.fit, lc, "flare", sampler="abc", n_simulations=1000, quantile=0.05, seed=0)
    optimised = _quiet(wp.fit, lc, "flare", sampler="abc", n_simulations=1000, quantile=0.05,
                       seed=0, likelihood_max_opt=True)
    assert "likelihood_max_opt" not in plain.info
    peak = optimised.info["likelihood_max_opt"]
    assert peak["max_log_likelihood"] >= optimised.max_log_likelihood
    assert set(peak["params"]) == {"amplitude", "rise_time", "decay_time"}
    # The posterior itself is not changed by the optimisation.
    np.testing.assert_array_equal(plain.samples.to_numpy(), optimised.samples.to_numpy())


def test_a_failed_likelihood_max_opt_keeps_the_fit(monkeypatch):
    import importlib
    module = importlib.import_module("whisper_cbpf.likelihood_max_opt")

    def refuse(*args, **kwargs):
        raise TypeError("a box cannot express this prior")
    monkeypatch.setattr(module, "likelihood_max_opt", refuse)
    with pytest.warns(UserWarning, match="optimisation failed.*a box cannot express"):
        res = wp.fit(_flare_lc(), "flare", sampler="abc", n_simulations=300, quantile=0.1,
                     likelihood_max_opt=True)
    assert res.n_samples > 0 and "likelihood_max_opt" not in res.info
    assert res.info["likelihood_max_opt_error"] == "TypeError: a box cannot express this prior"


def test_fit_likelihood_max_opt_with_an_unregistered_model_object():
    base = wp.get_model("flare")
    anon = wp.models.Model(name="flare_not_registered", predict=base.predict,
                           parameters=list(base.parameters), default_prior=base.default_prior)
    res = _quiet(wp.fit, _flare_lc(), anon, sampler="abc", n_simulations=500, quantile=0.1,
                 likelihood_max_opt=True)
    assert res.info["likelihood_max_opt"]["max_log_likelihood"] >= res.max_log_likelihood


def test_result_likelihood_max_opt_uses_the_rows_the_fit_used():
    """The light curve passed to the fit, pre-event rows included, is optimised without the
    'another number of points' error."""
    t = np.concatenate([[-3.0, -1.0], np.linspace(0.5, 30.0, 20)])
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
                                          "decay_time": 15.0}, t)
    lc = wp.LightCurve(time=t + 60000.0, band=["r"] * t.size, flux=flux,
                       flux_err=np.full(t.size, 0.1)).set_explosion_date(60000.0)
    res = _quiet(wp.fit, lc, "flare", sampler="abc", n_simulations=1000, quantile=0.05)
    assert res.n_data == 20
    peak = _quiet(res.likelihood_max_opt, lc)
    assert peak.n_data == 20 and res.info["likelihood_max_opt"] == peak.to_dict()


def test_result_likelihood_max_opt_climbs_in_the_box_the_fit_used():
    """The fit's prior excludes the truth (amplitude 5): the peak stays in that box."""
    prior = Prior({"amplitude": Uniform(2.0, 4.0), "rise_time": Uniform(1.0, 10.0),
                   "decay_time": Uniform(5.0, 30.0)})
    res = _quiet(wp.fit, _flare_lc(), "flare", sampler="abc", prior=prior,
                 n_simulations=1000, quantile=0.05)
    peak = _quiet(res.likelihood_max_opt, _flare_lc())
    assert peak.params["amplitude"] <= 4.0 and "amplitude" in peak.at_edge


# ------------------------------------------------------------------------------- forecast
def test_forecast_delegates_to_the_forecast_module(monkeypatch):
    calls = []
    fake = types.ModuleType("whisper_cbpf.forecast")
    fake.forecast = lambda result, times, bands, **kw: calls.append((result, times, bands, kw)) or 7
    monkeypatch.setitem(sys.modules, "whisper_cbpf.forecast", fake)
    res = _quiet(wp.fit, _flare_lc(), "flare", sampler="abc", n_simulations=300, quantile=0.1)
    assert res.forecast([35.0], "r", n_draws=10) == 7
    assert calls == [(res, [35.0], "r", {"n_draws": 10})]


def test_forecast_without_the_module_names_the_cause(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper_cbpf.forecast", None)
    res = _quiet(wp.fit, _flare_lc(), "flare", sampler="abc", n_simulations=300, quantile=0.1)
    with pytest.raises(ImportError, match="needs the whisper_cbpf.forecast module"):
        res.forecast([35.0], "r")


# ------------------------------------------------------------------------ diagnostics, Fixed
def test_the_convergence_report_skips_a_fixed_column():
    prior = Prior({"amplitude": Uniform(0.0, 10.0), "rise_time": Fixed(3.0),
                   "decay_time": Uniform(5.0, 30.0)})
    res = _quiet(wp.fit, _flare_lc(), "flare", sampler="mcmc", prior=prior, nsteps=400,
                 burnin=100, seed=0)
    assert res.info["fixed"] == {"rise_time": 3.0} and "rise_time" in res.samples
    view = res._without_fixed()
    assert view.parameters == ["amplitude", "decay_time"] and "rise_time" not in view.samples
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)       # no 0/0 from a constant column
        report = res.diagnostics()
    rows = {r.check: r for r in report.rows}
    assert "rise_time" not in rows["split R-hat across walkers (largest)"].reason
    assert res.parameters == ["amplitude", "rise_time", "decay_time"]      # result untouched


# ------------------------------------------------------------------------- save and reload
def test_the_pre_event_record_survives_save_and_load(tmp_path):
    t = np.concatenate([[-1.0], np.linspace(0.5, 30.0, 15)])
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
                                          "decay_time": 15.0}, t)
    lc = wp.LightCurve(time=t + 60000.0, band=["r"] * t.size, flux=flux,
                       flux_err=np.full(t.size, 0.1)).set_explosion_date(60000.0)
    res = _quiet(wp.fit, lc, "flare", sampler="abc", n_simulations=300, quantile=0.1)
    back = wp.load_result(res.save(tmp_path / "r"))
    assert back.info["excluded_pre_event"] == 1
    assert back.info["pre_event"] == res.info["pre_event"]
    assert len(back.fitted_lc(lc)) == back.n_data == 15


# ------------------------------------------------------------ the data and a model name alone
def test_the_default_sampler_fits_an_alert_with_limits():
    """``sampler="auto"``: a likelihood sampler, so a light curve with limits needs no argument
    (the 0.1.1 default, ABC, refuses them)."""
    res = _quiet(wp.fit, _alert_lc(), MODEL, nwalkers=8, nsteps=400, burnin=100, seed=0)
    assert res.sampler == "mcmc" and res.info["space"] == "flux" and res.n_data == 10


def test_the_auto_rule_is_the_same_as_compare():
    import importlib
    S = importlib.import_module("whisper_cbpf.samplers")
    C = importlib.import_module("whisper_cbpf.compare")
    assert S._auto_sampler("flare", lambda: True) == "mcmc"            # no JAX half
    if "emcee_jax" in wp.list_samplers():
        assert S._auto_sampler("flare_jax", lambda: True) == "emcee_jax"
        assert S._auto_sampler("flare_jax", lambda: False) == "mcmc"
    assert C._choose_sampler("auto", wp.get_model("flare")) == S._auto_sampler("flare")


def test_a_model_that_is_not_registered_is_found_after_the_fit():
    """A factory-built model is scored for its metrics during the fit and found again after it
    (it used to lose its band and predictive metrics, with two "Unknown model" warnings)."""
    import pickle

    from whisper_cbpf.samplers.base import fitted_model

    base = wp.get_model("flare")
    mine = wp.Model(name="flare_not_registered", predict=base.predict,
                    parameters=list(base.parameters), default_prior=base.default_prior)
    lc = _flare_lc()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit(lc, mine, sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
    assert not [w for w in caught if "Unknown model" in str(w.message)]
    assert "band_metrics" in res.info and "predictive_metrics" in res.info
    assert fitted_model(res) is mine
    assert res.likelihood_max_opt(lc).n_params == 3
    assert len(res.forecast([31.0], "r", n_draws=20)) == 1
    assert _quiet(wp.waic, res, lc)["n_data"] == 30
    back = pickle.loads(pickle.dumps(res))                 # the object is not pickled
    assert not hasattr(back, "_model_object") and back.summary == res.summary
    with pytest.raises(KeyError, match="Pass model="):
        fitted_model(back)


def test_a_samplers_warning_points_at_the_callers_line():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wp.fit(_flare_lc(), "flare", sampler="mcmc", nwalkers=8, nsteps=200, burnin=50, seed=0)
    ours = [w for w in caught if "not converged" in str(w.message)]
    assert ours and all(w.filename == __file__ for w in ours)


def test_one_posterior_draw_skips_the_predictive_metrics_by_name():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit(_flare_lc(), "flare", sampler="abc", n_simulations=100, quantile=0.01,
                     seed=0)
    assert res.n_samples == 1
    assert "at least two" in res.info["predictive_metrics_skipped"]
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
