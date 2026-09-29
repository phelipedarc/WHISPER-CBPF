"""Pre-event data are never fitted (user decision, 2026-09-26).

Rows at or before the event -- day 0 of a light curve whose explosion (or merger) date is set, or,
for a model that fits its explosion time, the rows before the first detection -- are left out of
the likelihood and of the model evaluation. They may only define a prior: the explosion time's,
from the last non-detection before the first detection to the first detection.

The central test: adding pre-event rows changes the posterior and the predictions by EXACTLY zero
at the same seed, for every sampler family.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.priors import Fixed, Normal, Prior, Uniform
from whisper_cbpf.samplers.base import prepare_lc

TRUTH = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
T_POST = np.array([0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 16.0, 21.0, 26.0, 30.0])
MJD0 = 60000.0


def _flare(t):
    return wp.get_model("flare").predict(TRUTH, np.asarray(t, dtype=float))


def _post_lc():
    """Days since the explosion, every row after it."""
    rng = np.random.default_rng(1)
    flux = _flare(T_POST) + rng.normal(0.0, 0.1, T_POST.size)
    return wp.LightCurve(time=T_POST + MJD0, band=["r"] * T_POST.size, flux=flux,
                         flux_err=np.full(T_POST.size, 0.1),
                         upper_limit=np.zeros(T_POST.size, bool)).set_explosion_date(MJD0)


def _with_pre_event(lc):
    """``lc`` plus five pre-event rows: three non-detections, a detection, and one AT day 0."""
    pre_t = np.array([-9.0, -4.0, -1.0, -0.3, 0.0])
    pre_ul = np.array([True, True, True, False, False])
    pre_f = np.where(pre_ul, 0.3, 0.2)
    pre_e = np.where(pre_ul, np.nan, 0.1)
    t = np.concatenate([pre_t, np.asarray(lc.time)])
    order = np.argsort(t, kind="stable")
    full = wp.LightCurve(
        time=(t + MJD0)[order], band=np.array(["r"] * t.size),
        flux=np.concatenate([pre_f, np.asarray(lc.flux)])[order],
        flux_err=np.concatenate([pre_e, np.asarray(lc.flux_err)])[order],
        upper_limit=np.concatenate([pre_ul, np.asarray(lc.upper_limit)])[order])
    return full.set_explosion_date(MJD0)


def _quiet_fit(lc, model, sampler, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.fit(lc, model, sampler=sampler, **kw)


BUDGETS = {
    "mcmc": dict(nwalkers=8, nsteps=300, burnin=100, seed=3),
    "abc": dict(n_simulations=1500, quantile=0.05, seed=3),
    "abc_smc": dict(n_particles=60, n_rounds=2, seed=3),
    "nested": dict(nlive=40, maxiter=400, seed=3),
}


# --------------------------------------------------------------------- the rule changes nothing
@pytest.mark.parametrize("sampler", sorted(BUDGETS))
def test_adding_pre_event_rows_changes_the_posterior_by_exactly_zero(sampler):
    if sampler == "nested":
        pytest.importorskip("dynesty")
    post = _post_lc()
    full = _with_pre_event(post)
    a = _quiet_fit(post, "flare", sampler, **BUDGETS[sampler])
    b = _quiet_fit(full, "flare", sampler, **BUDGETS[sampler])
    assert a.n_data == b.n_data == len(post)
    assert a.info["excluded_pre_event"] == 0 and b.info["excluded_pre_event"] == 5
    assert b.info["pre_event"]["n_upper_limits"] == 3 and b.info["pre_event"]["n_detections"] == 2
    cols = list(a.parameters)
    np.testing.assert_array_equal(a.samples[cols].to_numpy(), b.samples[cols].to_numpy())
    assert a.best_params == b.best_params
    assert (a.max_log_likelihood, a.aic, a.bic) == (b.max_log_likelihood, b.aic, b.bic)
    # ... and so are the predictions: the same draws through the same model, bit for bit.
    grid = np.linspace(0.5, 40.0, 25)
    model = wp.get_model("flare")
    pa = np.array([model.predict(dict(r), grid) for _, r in a.samples[cols].head(50).iterrows()])
    pb = np.array([model.predict(dict(r), grid) for _, r in b.samples[cols].head(50).iterrows()])
    np.testing.assert_array_equal(pa, pb)
    assert a.info.get("predictive_metrics") == b.info.get("predictive_metrics")


def _jax_post_lc():
    truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}
    flux = wp.get_model("flare_jax").predict(truth, T_POST)
    return wp.LightCurve(time=T_POST + MJD0, band=["r"] * T_POST.size, flux=flux,
                         flux_err=np.full(T_POST.size, 0.1),
                         upper_limit=np.zeros(T_POST.size, bool)).set_explosion_date(MJD0)


@pytest.mark.parametrize("entry", ["wp.fit", "fit_emcee_jax"])
def test_emcee_jax_leaves_pre_event_rows_out_too(entry):
    pytest.importorskip("jax")
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    post = _jax_post_lc()
    full = _with_pre_event(post)
    kw = dict(nwalkers=16, nsteps=200, burnin=50, seed=2, walker_chunk=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if entry == "wp.fit":
            a = wp.fit(post, "flare_jax", sampler="emcee_jax", **kw)
            b = wp.fit(full, "flare_jax", sampler="emcee_jax", **kw)
        else:
            a = fit_emcee_jax(post, "flare_jax", **kw)
            b = fit_emcee_jax(full, "flare_jax", **kw)
    assert b.info["excluded_pre_event"] == 5 and b.n_data == a.n_data == len(post)
    np.testing.assert_array_equal(a.samples.to_numpy(), b.samples.to_numpy())
    assert a.max_log_likelihood == b.max_log_likelihood


@pytest.mark.parametrize("sampler,kw", [
    ("abc_gpu", dict(n_simulations=2000, quantile=0.05, seed=2)),
    ("abc_smc_gpu", dict(n_particles=100, n_rounds=2, seed=2)),
])
def test_the_gpu_abc_samplers_leave_pre_event_rows_out_too(sampler, kw):
    pytest.importorskip("jax")
    post = _jax_post_lc()
    a = _quiet_fit(post, "flare_jax", sampler, **kw)
    b = _quiet_fit(_with_pre_event(post), "flare_jax", sampler, **kw)
    assert b.info["excluded_pre_event"] == 5 and b.n_data == a.n_data == len(post)
    np.testing.assert_array_equal(a.samples.to_numpy(), b.samples.to_numpy())
    assert a.max_log_likelihood == b.max_log_likelihood


@pytest.mark.slow
@pytest.mark.parametrize("sampler,kw", [
    ("nuts_gpu", dict(num_warmup=100, num_samples=100, num_chains=2, seed=2)),
    ("pymc_jax_gpu_vectorized", dict(num_warmup=100, num_samples=100, num_chains=2, seed=2)),
])
def test_the_nuts_samplers_leave_pre_event_rows_out_too(sampler, kw):
    pytest.importorskip("numpyro")
    if sampler.startswith("pymc"):
        pytest.importorskip("pymc")
    post = _jax_post_lc()
    a = _quiet_fit(post, "flare_jax", sampler, **kw)
    b = _quiet_fit(_with_pre_event(post), "flare_jax", sampler, **kw)
    assert b.info["excluded_pre_event"] == 5 and b.n_data == a.n_data == len(post)
    np.testing.assert_array_equal(a.samples.to_numpy(), b.samples.to_numpy())


@pytest.mark.slow
def test_snpe_leaves_pre_event_rows_out_too():
    pytest.importorskip("sbi")
    post = _post_lc()
    kw = dict(num_rounds=1, num_simulations=300, seed=3)
    a = _quiet_fit(post, "flare", "snpe", **kw)
    b = _quiet_fit(_with_pre_event(post), "flare", "snpe", **kw)
    assert b.info["excluded_pre_event"] == 5 and b.n_data == a.n_data == len(post)
    np.testing.assert_array_equal(a.samples.to_numpy(), b.samples.to_numpy())


# ----------------------------------------------------------------- the model never sees them
def _recording_model(seen):
    base = wp.get_model("flare")

    def predict(params, times, bands=None):
        seen.append(float(np.min(times)))
        return base.predict(params, times, bands)

    return wp.register_model("flare_recorded", predict, list(base.parameters),
                             prior=base.default_prior, overwrite=True)


@pytest.mark.parametrize("sampler", ["mcmc", "abc"])
def test_the_model_is_never_evaluated_at_a_pre_event_epoch(sampler):
    """Not just the likelihood: the forward model never receives those epochs, so they cannot
    move a model's time grid (the kilonova grid-edge effect)."""
    seen = []
    res = _quiet_fit(_with_pre_event(_post_lc()), _recording_model(seen), sampler,
                     **BUDGETS[sampler])
    assert seen and min(seen) > 0.0
    assert res.info["excluded_pre_event"] == 5


# ------------------------------------------------------------------------------ one message
def test_one_warning_names_what_was_left_out_at_the_callers_line():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wp.fit(_with_pre_event(_post_lc()), "flare", sampler="abc", **BUDGETS["abc"])
    ours = [w for w in caught if "pre-event data are not fitted" in str(w.message)]
    assert len(ours) == 1
    text = str(ours[0].message)
    assert "5 of 15 rows (3 upper limit(s), 2 detection(s))" in text
    assert "'explosion', MJD 60000.00000" in text and "free=['t_exp']" in text
    assert ours[0].filename == __file__


def test_no_message_and_nothing_left_out_without_a_declared_event():
    """A raw clock (no set_explosion_date / set_time_reference) declares no event."""
    lc = _with_pre_event(_post_lc())
    raw = wp.LightCurve(time=np.asarray(lc.time) + MJD0, band=lc.band,
                        flux=np.where(lc.upper_limit, 0.0, lc.flux),
                        flux_err=np.where(lc.upper_limit, 0.1, lc.flux_err))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit(raw, "flare", sampler="abc", n_simulations=300, quantile=0.1)
    assert not [w for w in caught if "pre-event" in str(w.message)]
    assert res.info["excluded_pre_event"] == 0 and res.info["pre_event"]["rule"] == "none"
    assert prepare_lc(raw, "flare") is raw


def test_a_first_detection_reference_is_day_0_too():
    lc = _post_lc()
    ref = wp.LightCurve(time=np.asarray(lc.time) + MJD0, band=lc.band, flux=lc.flux,
                        flux_err=lc.flux_err).set_time_reference(MJD0 + 1.5, "first detection")
    kept = prepare_lc(ref, "flare")
    assert np.all(np.asarray(kept.time) > 0.0) and len(kept) == len(lc) - 2


def _raw_post_lc():
    lc = _post_lc()
    return wp.LightCurve(time=np.asarray(lc.time) + MJD0, band=lc.band, flux=lc.flux,
                         flux_err=lc.flux_err)


def test_a_peak_reference_leaves_nothing_out_for_a_model_with_its_own_epoch():
    """bazin fits its peak time t0: day 0 of a 'peak' clock is not its event, so the rise stays in
    the fit. A model whose clock starts at day 0 (flare) still loses the rows before it, and an
    explosion date is the event for every model."""
    peak = _raw_post_lc().set_time_reference(MJD0 + 8.0, "peak")
    assert prepare_lc(peak, "bazin") is peak
    res = _quiet_fit(peak, "bazin", "abc", n_simulations=300, quantile=0.1)
    assert res.info["excluded_pre_event"] == 0 and res.n_data == len(peak)
    assert res.info["pre_event"]["rule"] == "none"
    assert "'peak', not the explosion or merger" in res.info["pre_event"]["reason"]
    assert "'t0'" in res.info["pre_event"]["reason"]
    assert len(prepare_lc(peak, "flare")) == int(np.sum(np.asarray(peak.time) > 0.0)) == 5
    explosion = _raw_post_lc().set_time_reference(MJD0 + 1.0, "explosion")
    assert len(prepare_lc(explosion, "bazin")) == len(explosion) - 1


def test_a_merger_time_on_a_raw_clock_is_the_event():
    """meta['merger_mjd'] is an MJD: on a raw MJD clock the event sits at that MJD, not at 0."""
    raw = _raw_post_lc()
    raw.meta["merger_mjd"] = MJD0 + 2.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit(raw, "flare", sampler="abc", n_simulations=300, quantile=0.1)
    assert res.info["pre_event"]["rule"] == "day 0"
    assert res.info["pre_event"]["reference"] == MJD0 + 2.0
    assert res.info["excluded_pre_event"] == 2 and res.n_data == len(raw) - 2
    text = " ".join(str(w.message) for w in caught if "pre-event" in str(w.message))
    assert "the merger (MJD 60002.00000, t = 60002 on this light curve's clock)" in text
    # ... and on a clock shifted to another reference, where it falls on that clock.
    shifted = _raw_post_lc().set_time_reference(MJD0 + 1.0, "first detection")
    shifted.meta["merger_mjd"] = MJD0 + 2.0
    assert np.asarray(prepare_lc(shifted, "flare").time).min() == pytest.approx(2.0)


@pytest.mark.parametrize("entry", ["wp.fit", "fit_emcee_jax"])
def test_a_likelihood_object_built_on_the_pre_event_rows_is_refused_by_name(entry):
    """A built likelihood carries its own copy of the data; on the full curve it failed with a
    shape mismatch inside the compiled density. Built on prepare_lc's rows, it fits."""
    pytest.importorskip("jax")
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    full = _with_pre_event(_jax_post_lc())
    kw = dict(nwalkers=16, nsteps=60, burnin=20, seed=2, walker_chunk=None)
    run = ((lambda lik: wp.fit(full, "flare_jax", sampler="emcee_jax", likelihood=lik, **kw))
           if entry == "wp.fit" else
           (lambda lik: fit_emcee_jax(full, "flare_jax", likelihood=lik, **kw)))
    with pytest.raises(ValueError, match=r"likelihood object passed to this fit was built on all "
                                         r"15 rows.*5 row\(s\).*uses 10.*prepare_lc"):
        run(wp.make_likelihood(full))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run(wp.make_likelihood(prepare_lc(full, "flare_jax")))
    assert res.n_data == 10 and res.info["excluded_pre_event"] == 5


# ------------------------------------------------------------------- not enough data, loudly
def test_nothing_after_the_event_is_not_enough_data():
    lc = _post_lc()
    early = wp.LightCurve(time=np.asarray(lc.time) + MJD0, band=lc.band, flux=lc.flux,
                          flux_err=lc.flux_err).set_explosion_date(MJD0 + 40.0)
    with pytest.raises(ValueError, match="not enough data: all 10 rows"):
        wp.fit(early, "flare", sampler="abc", n_simulations=100)


def test_only_limits_after_the_event_is_not_enough_data():
    t = np.array([-3.0, -1.0, 2.0, 4.0])
    lc = wp.LightCurve(time=t + MJD0, band=["r"] * 4, flux=[1.0, 1.0, 0.5, 0.5],
                       flux_err=[0.1, 0.1, np.nan, np.nan],
                       upper_limit=[False, False, True, True]).set_explosion_date(MJD0)
    with pytest.raises(ValueError, match="not enough data: no detection is left after day 0"):
        wp.fit(lc, "flare", sampler="mcmc", nsteps=50, burnin=10)


# -------------------------------------------------------------- ABC and SNPE cannot use limits
@pytest.mark.parametrize("sampler", ["abc", "abc_smc"])
def test_abc_says_it_cannot_use_limits_after_the_event(sampler):
    lc = _post_lc()
    ul = np.zeros(len(lc), bool)
    ul[-2:] = True
    with_limits = wp.LightCurve(time=np.asarray(lc.time) + MJD0, band=lc.band, flux=lc.flux,
                                flux_err=np.where(ul, np.nan, lc.flux_err),
                                upper_limit=ul).set_explosion_date(MJD0)
    with pytest.raises(ValueError, match=rf"{sampler} cannot use upper limits.*2 upper limit"
                                         rf".*lc.where\(upper_limit=False\).*'mcmc'"):
        wp.fit(with_limits, "flare", sampler=sampler, **BUDGETS[sampler])
    # Pre-event limits are left out first, so they never trouble ABC:
    res = _quiet_fit(_with_pre_event(lc), "flare", sampler, **BUDGETS[sampler])
    assert res.info["excluded_pre_event"] == 5


# ---------------------------------------------------------------- a free explosion time
def _shifted_flare(params, times, bands=None):
    return wp.get_model("flare").predict(params, np.asarray(times, float) - params["t_exp"], bands)


PHYS = {"amplitude": Uniform(0.0, 10.0), "rise_time": Uniform(1.0, 10.0),
        "decay_time": Uniform(5.0, 30.0)}


def _free_model(t_exp_prior=None, name="flare_free_texp"):
    dists = dict(PHYS)
    if t_exp_prior is not None:
        dists["t_exp"] = t_exp_prior
    return wp.register_model(name, _shifted_flare,
                             ["amplitude", "rise_time", "decay_time", "t_exp"],
                             prior=Prior(dists), overwrite=True)


def _alert(t_exp=60003.0, nondet=(59990.0, 59995.0, 60001.0)):
    """Raw MJD clock: non-detections, then detections from 60003.4 on."""
    det_t = t_exp + np.array([0.4, 1.0, 2.5, 4.0, 7.0, 11.0, 16.0, 22.0])
    det_f = wp.get_model("flare").predict(TRUTH, det_t - t_exp)
    t = np.concatenate([np.asarray(nondet, float), det_t])
    ul = np.concatenate([np.ones(len(nondet), bool), np.zeros(det_t.size, bool)])
    return wp.LightCurve(time=t, band=["r"] * t.size,
                         flux=np.concatenate([np.full(len(nondet), 0.3), det_f]),
                         flux_err=np.where(ul, np.nan, 0.1), upper_limit=ul)


def test_free_t_exp_takes_its_prior_from_the_non_detections():
    res = _quiet_fit(_alert(), _free_model(), "abc", n_simulations=3000, quantile=0.05, seed=0)
    rec = res.info["t_exp_prior"]
    assert (rec["low"], rec["high"], rec["source"]) == (60001.0, 60003.4, "data")
    assert res.info["pre_event"]["rule"] == "free t_exp"
    assert res.info["excluded_pre_event"] == 3 and res.n_data == 8
    assert res.samples["t_exp"].between(60001.0, 60003.4).all()
    # The provenance records the prior the fit used, not the model's (which has no t_exp).
    assert res.provenance["model"]["prior"]["parameters"]["t_exp"]["low"] == 60001.0


def test_free_t_exp_without_an_earlier_non_detection_uses_the_fallback_window():
    res = _quiet_fit(_alert(nondet=()), _free_model(), "abc", n_simulations=500, quantile=0.1)
    rec = res.info["t_exp_prior"]
    assert rec["source"] == "data, fallback window"
    assert (rec["low"], rec["high"]) == pytest.approx((60003.4 - 30.0, 60003.4))


def test_a_uniform_model_prior_is_cut_to_the_data_window():
    model = _free_model(Uniform(59950.0, 60010.0))
    kept = wp.LightCurve.explosion_time_prior(_alert())
    res = _quiet_fit(_alert(), model, "abc", n_simulations=500, quantile=0.1)
    rec = res.info["t_exp_prior"]
    assert rec["source"] == "model prior, cut to the data window"
    assert (rec["low"], rec["high"]) == (kept.low, kept.high) == (60001.0, 60003.4)
    # a model prior already inside the window is kept as it is
    inside = _quiet_fit(_alert(), _free_model(Uniform(60002.0, 60003.0), "flare_inside"), "abc",
                        n_simulations=300, quantile=0.1)
    assert inside.info["t_exp_prior"]["source"] == "model prior"
    assert (inside.info["t_exp_prior"]["low"], inside.info["t_exp_prior"]["high"]) == (60002.0,
                                                                                       60003.0)


def test_a_non_uniform_model_prior_and_a_prior_passed_to_fit_are_kept():
    normal = _quiet_fit(_alert(), _free_model(Normal(60002.0, 1.0)), "abc",
                        n_simulations=300, quantile=0.1)
    assert normal.info["t_exp_prior"]["source"] == "model prior"
    assert normal.info["t_exp_prior"]["type"] == "Normal"
    mine = Prior({**PHYS, "t_exp": Uniform(59980.0, 60003.0)})
    given = _quiet_fit(_alert(), _free_model(), "abc", prior=mine, n_simulations=300,
                       quantile=0.1)
    assert given.info["t_exp_prior"]["source"] == "passed to fit"
    assert given.info["t_exp_prior"]["low"] == 59980.0


def test_a_model_prior_that_misses_the_data_window_is_refused():
    with pytest.raises(ValueError, match=r"explosion-time prior Uniform\(59900.0, 59950.0\).*do "
                                         r"not overlap"):
        wp.fit(_alert(), _free_model(Uniform(59900.0, 59950.0), "flare_miss"), sampler="abc",
               n_simulations=100)


def test_with_an_explicit_t_exp_prior_pre_detection_rows_change_nothing():
    """The rule's test for a fitted explosion time: with the prior fixed by the caller, adding
    the non-detections before the first detection changes the posterior by exactly zero."""
    prior = Prior({**PHYS, "t_exp": Uniform(59998.0, 60003.4)})
    a = _quiet_fit(_alert(nondet=()), _free_model(), "mcmc", prior=prior, **BUDGETS["mcmc"])
    b = _quiet_fit(_alert(), _free_model(), "mcmc", prior=prior, **BUDGETS["mcmc"])
    assert b.info["excluded_pre_event"] == 3 and a.n_data == b.n_data == 8
    np.testing.assert_array_equal(a.samples.to_numpy(), b.samples.to_numpy())
    assert a.max_log_likelihood == b.max_log_likelihood


def test_a_fixed_t_exp_is_the_event():
    prior = Prior({**PHYS, "t_exp": Fixed(60003.0)})
    lc = _alert(nondet=(59990.0, 60002.0, 60003.0, 60003.2))     # 60003.2: after the event
    kept = prepare_lc(lc, _free_model(), prior=prior)
    assert np.asarray(kept.time).min() == 60003.2 and len(kept) == len(lc) - 3


def test_the_free_texp_message_states_the_prior():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wp.fit(_alert(), _free_model(), sampler="abc", n_simulations=200, quantile=0.1)
    text = " ".join(str(w.message) for w in caught if "pre-event" in str(w.message))
    assert "the 3 upper limit(s) before the first detection (t = 60003.4)" in text
    assert "t_exp ~ Uniform(60001.0, 60003.4)" in text


def test_a_jax_factory_model_with_a_free_t_exp_gets_the_data_window():
    """The feature-1 factories require a t_exp prior; a Uniform one is cut to the data window."""
    pytest.importorskip("jax")
    from whisper_cbpf.models.jax._factories import kilonova_model
    from whisper_cbpf.samplers.base import _pre_event_plan

    model = kilonova_model(["lsstg", "lsstr"], 0.01, 1.3e26, free=["t_exp"],
                           prior=Prior({"t_exp": Uniform(59990.0, 60010.0)}))
    plan = _pre_event_plan(_alert(), model)
    assert plan.prior_changed and plan.lc.time.min() == 60003.4
    assert repr(plan.prior.distributions["t_exp"]) == "Uniform(60001.0, 60003.4)"
    assert list(plan.prior.names) == list(model.default_prior.names)     # order kept


# ------------------------------------------------------------ LightCurve.explosion_time_prior
def test_explosion_time_prior_reads_every_band():
    lc = wp.LightCurve(time=[1.0, 2.0, 2.5, 3.0, 5.0], band=["g", "r", "g", "r", "g"],
                       magnitude=[21.0, 21.0, 19.5, 19.0, 18.5],
                       magnitude_err=[np.nan, np.nan, 0.1, 0.1, 0.1],
                       upper_limit=[True, True, False, False, False])
    assert repr(lc.explosion_time_prior()) == "Uniform(2.0, 2.5)"
    no_limits = lc.where(upper_limit=False)
    assert repr(no_limits.explosion_time_prior(fallback_days=10)) == "Uniform(-7.5, 2.5)"
    with pytest.raises(ValueError, match="fallback_days must be > 0"):
        lc.explosion_time_prior(fallback_days=0)
    with pytest.raises(ValueError, match="not enough data: this light curve has no detection"):
        lc.where(upper_limit=True).explosion_time_prior()


def test_fitted_lc_reapplies_the_recorded_rule():
    full = _with_pre_event(_post_lc())
    res = _quiet_fit(full, "flare", "abc", n_simulations=300, quantile=0.1)
    kept = res.fitted_lc(full)
    assert len(kept) == res.n_data == 10 and np.all(np.asarray(kept.time) > 0)
    assert res.fitted_lc(kept) is kept


# ----------------------------------------- the same rule in log_density, fit_batch and the metrics
def test_log_density_and_fit_batch_leave_pre_event_rows_out_too():
    pytest.importorskip("jax")
    post, full = _jax_post_lc(), _with_pre_event(_jax_post_lc())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = wp.log_density(post, "flare_jax"), wp.log_density(full, "flare_jax")
    assert a.n_data == b.n_data == len(post) and b.pre_event["n_excluded"] == 5
    theta = np.array([1.0, 0.5, 1.5, 10.0])
    assert float(a.fn(theta)) == float(b.fn(theta))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fits = wp.fit_batch([post, full], "flare_jax", nwalkers=16, nsteps=200, burnin=50,
                            seed=[2, 2], metrics=False)
    assert [f.info["excluded_pre_event"] for f in fits] == [0, 5]
    assert fits[1].n_data == fits[0].n_data == len(post)
    np.testing.assert_array_equal(fits[0].samples.to_numpy(), fits[1].samples.to_numpy())
    ours = [w for w in caught if "pre-event data are not fitted" in str(w.message)]
    assert len(ours) == 1 and "1 of 2 light curves" in str(ours[0].message)


def test_the_metrics_of_a_fit_score_its_own_rows():
    """``wp.waic(result, lc)`` and the predictive check drop the rows the fit left out."""
    full = _with_pre_event(_post_lc())
    res = _quiet_fit(full, "flare", "mcmc", **BUDGETS["mcmc"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = wp.waic(res, full)
        ppc = wp.posterior_predictive_check(res, full, n_draws=20)
    assert w["n_data"] == res.n_data == 10 and w["waic"] == res.waic
    assert ppc["dof"] == res.n_data - 3
