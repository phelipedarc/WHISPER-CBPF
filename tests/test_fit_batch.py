"""``fit_batch``: K independent emcee ensembles in one compiled loop.

The claims, each a test:

1. **Each light curve gets its own posterior**: three alerts whose
   posteriors are three different correlated Gaussians, far apart, in one batch -- each ensemble
   recovers its own mean and covariance, and no draw of one lies near another's.
2. **A fit does not depend on the batch**: light curve i of a batch equals the same light curve
   fitted alone with seed + i, draw for draw.
3. **The results are whisper results**: diagnostics, save / load, the prior-edge check, the
   metrics, and every start form.
4. **Several priors, one program**; a model that needs concrete epochs runs one compile per light
   curve and says so; upper limits in flux space work.
5. **Errors name the cause and the fix.**
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
import whisper_cbpf.models as M  # noqa: E402
from whisper_cbpf.priors import Fixed, Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax.batch import fit_batch  # noqa: E402

NAMES = ["c0", "c1", "c2"]


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _quadratic(box=200.0):
    """flux = c0 + c1 t + c2 t^2: linear in the parameters, so with Gaussian errors and a wide flat
    prior the posterior IS a correlated Gaussian, known exactly (weighted least squares)."""
    def predict(p, t, bands=None):
        t = np.asarray(t, dtype=float)
        return p["c0"] + p["c1"] * t + p["c2"] * t * t

    def predict_jax(theta, t):
        return theta[0] + theta[1] * t + theta[2] * t * t

    return M.Model(name="quadratic_toy", predict=predict, parameters=list(NAMES),
                   default_prior=Prior({n: Uniform(-box, box) for n in NAMES}),
                   predict_jax=predict_jax)


def _alert(coef, n, sigma=0.3, seed=0, span=(0.0, 3.0)):
    t = np.linspace(*span, n)
    rng = np.random.default_rng(seed)
    flux = coef[0] + coef[1] * t + coef[2] * t * t + rng.normal(0.0, sigma, n)
    lc = wp.LightCurve(time=t, band=["r"] * n, flux=flux, flux_err=np.full(n, sigma))
    x = np.stack([np.ones(n), t, t * t], axis=1) / sigma
    cov = np.linalg.inv(x.T @ x)
    mean = cov @ (x.T @ (flux / sigma))
    return lc, mean, cov


def _fit(lcs, model, **kw):
    kw = {"nwalkers": 24, "nsteps": 3000, "burnin": 600, "thin": 4, "metrics": False, **kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)          # short chains: "not converged"
        return fit_batch(lcs, model, **kw)


# --- 1. each alert its own correlated Gaussian -------------------------------------------------

def test_each_light_curve_recovers_its_own_correlated_gaussian_with_no_mixing():
    model = _quadratic()
    alerts = [_alert((40.0, -5.0, 2.0), 24, seed=1), _alert((-60.0, 10.0, -1.0), 20, seed=2),
              _alert((5.0, 30.0, 0.5), 30, seed=3, span=(1.0, 4.0))]
    fits = _fit([a[0] for a in alerts], model, nwalkers=32, nsteps=6000, burnin=1000, thin=5)
    for (lc, mean, cov), fit in zip(alerts, fits):
        x = fit.samples[NAMES].to_numpy()
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        assert abs(corr[0, 1]) > 0.5                          # genuinely correlated
        assert np.all(np.abs(x.mean(0) - mean) < 0.15 * sd), (x.mean(0), mean, sd)
        np.testing.assert_allclose(x.std(0), sd, rtol=0.15)
        np.testing.assert_allclose(np.corrcoef(x.T), corr, atol=0.08)
        # no mixing: every draw is within 8 sd of its OWN posterior (the others are > 50 sd away)
        assert np.all(np.abs(x - mean) < 8.0 * sd)
        assert 0.2 < fit.info["mean_acceptance_fraction"] < 0.7
        assert fit.info["batch"]["n_alerts"] == 3 and fit.info["batch"]["bucket"] == 32
        assert fit.info["batch"]["program"] == "data as arguments"
        assert fit.n_data == len(lc.time) and fit.n_params == 3


# --- 2. a fit does not depend on the batch ------------------------------------------------------

def test_a_light_curve_in_a_batch_equals_the_same_light_curve_alone():
    model = _quadratic()
    lcs = [_alert((1.0, 2.0, 0.5), n, seed=s)[0] for n, s in ((20, 1), (22, 2), (19, 3))]
    batch = _fit(lcs, model, seed=10)                  # all three in bucket 24
    alone = _fit([lcs[1]], model, seed=11)             # light curve 1 used seed 10 + 1
    np.testing.assert_array_equal(batch[1].samples.to_numpy(), alone[0].samples.to_numpy())
    assert batch[1].max_log_likelihood == alone[0].max_log_likelihood
    also = _fit([lcs[2], lcs[0]], model, seed=[12, 10])       # a list: one seed per light curve
    np.testing.assert_array_equal(also[1].samples.to_numpy(), batch[0].samples.to_numpy())
    assert batch[0].info["seed"] == 10 and batch[2].info["seed"] == 12


# --- 3. whisper results ------------------------------------------------------------------------

def test_results_carry_diagnostics_provenance_and_save_and_load(tmp_path):
    toy = _quadratic()
    # the metrics look the model up by name, as for every sampler
    model = M.register_model(toy.name, toy.predict, toy.parameters, prior=toy.default_prior,
                             overwrite=True, predict_jax=toy.predict_jax)
    lc, _, _ = _alert((3.0, -1.0, 0.4), 25, seed=4)
    fit = fit_batch([lc], model, nwalkers=24, nsteps=4000, burnin=1000, thin=5, metrics=True)[0]
    assert fit.sampler == "emcee_batch"
    assert fit.samples_by_chain.shape == (24, 600, 3)
    assert fit.info["converged"] is True, fit.info["convergence_problems"]
    report = fit.diagnostics()
    assert report.kind == "ensemble" and report.passed, report.reasons
    assert "predictive_metrics" in fit.info and fit.rmse is not None
    assert fit.provenance["model"]["prior"]["parameters"]["c0"]["type"] == "Uniform"
    assert fit.provenance["sampler"]["name"] == "emcee_batch"
    # AIC / BIC from the best kept draw's log-likelihood
    k, n = 3, 25
    assert fit.bic == pytest.approx(-2.0 * fit.max_log_likelihood + k * np.log(n))
    back = wp.load_result(fit.save(tmp_path / "fit"))
    assert back.summary == fit.summary and back.max_log_likelihood == fit.max_log_likelihood
    np.testing.assert_array_equal(np.asarray(back.samples_by_chain), fit.samples_by_chain)


def test_every_start_form():
    model = _quadratic()
    lcs = [_alert((1.0, 2.0, 0.5), 20, seed=s)[0] for s in (1, 2)]
    scan = _fit(lcs, model, nsteps=400, burnin=100)
    assert scan[0].info["init"] == "prior_scan"
    detail = scan[0].info["init_detail"]
    # emcee_jax's start: scan, climb the best 32, spread the walkers in the best basin
    assert detail["n_draws"] >= 1000 and detail["n_climbed"] == 32
    assert detail["climb_error"] is None and detail["n_reaching_best"] >= 1
    assert np.isfinite(detail["worst_start_log_prob"])
    prior = _fit(lcs, model, init="prior", nsteps=400, burnin=100)
    assert prior[1].info["init"] == "prior"
    arr = np.random.default_rng(0).normal([1.0, 2.0, 0.5], 0.01, size=(2, 24, 3))
    given = _fit(lcs, model, init=arr, nsteps=400, burnin=100)
    assert given[0].info["init"] == "per_walker"
    point = _fit(lcs, model, init=[{"c0": 1.0, "c1": 2.0, "c2": 0.5}, scan[1]], nsteps=400,
                 burnin=100)
    assert point[0].info["init"] == "ball" and point[1].info["init"] == "result"


def test_a_light_curve_with_no_finite_start_is_named_and_the_others_do_not_hang():
    """The start searches run in lockstep threads: one that fails must not leave the rest
    waiting."""
    base = _quadratic()

    def predict_jax(theta, t):                          # NaN physics wherever c0 <= 0
        return jnp.where(theta[0] > 0, theta[0] + theta[1] * t + theta[2] * t * t, jnp.nan)

    model = M.Model(name="nan_quadratic", predict=base.predict, parameters=list(NAMES),
                    default_prior=base.default_prior, predict_jax=predict_jax)
    lcs = [_alert((1.0, 2.0, 0.5), 20, seed=s)[0] for s in (1, 2, 3)]
    ok = Prior({n: Uniform(0.1, 10.0) for n in NAMES})
    walled = Prior({"c0": Uniform(-10.0, -5.0), "c1": Uniform(0.1, 10.0),
                    "c2": Uniform(0.1, 10.0)})
    with pytest.raises(ValueError, match="light curve 1: the log-density is -inf or NaN at all"):
        _fit(lcs, model, prior=[ok, walled, ok], nsteps=200, burnin=50)


def test_vectorised_prior_draws_equal_the_samplers_draws_to_the_bit():
    from whisper_cbpf.priors import LogUniform, Normal, TruncatedNormal
    from whisper_cbpf.samplers.jax import _diagnostics as dg
    from whisper_cbpf.samplers.jax.batch import _prior_draws

    prior = Prior({"a": Uniform(-3.0, 5.0), "b": LogUniform(1e-3, 10.0), "c": Normal(1.0, 0.2),
                   "d": TruncatedNormal(0.3, 0.1, 0.05, 0.6), "e": Fixed(2.5)})
    names = ["a", "b", "c", "d", "e"]
    np.testing.assert_array_equal(_prior_draws(prior, names, 300, (4, 7)),
                                  dg.prior_draws(prior, names, 300, (4, 7)))


# --- 4. priors, closures, limits ---------------------------------------------------------------

def test_one_prior_per_light_curve_shares_one_program():
    model = _quadratic()
    lcs = [_alert((1.0, 2.0, 0.5), 20, seed=1)[0], _alert((1.0, 2.0, 0.5), 20, seed=2)[0]]
    priors = [Prior({n: Uniform(-10.0, 10.0) for n in NAMES}),
              Prior({"c0": Uniform(1.5, 3.0), "c1": Uniform(-10.0, 10.0), "c2": Uniform(-10, 10)})]
    fits = _fit(lcs, model, prior=priors)
    assert fits[0].info["batch"]["n_alerts"] == 2              # one compiled call for both
    assert fits[1].samples["c0"].min() >= 1.5                  # the second prior's box holds
    assert fits[0].samples["c0"].min() < 1.5
    fixed = _fit(lcs, model, prior=Prior({"c0": Uniform(-10, 10), "c1": Uniform(-10, 10),
                                          "c2": Fixed(0.5)}), nwalkers=12)
    assert fixed[0].n_params == 2 and set(fixed[0].samples["c2"]) == {0.5}
    assert fixed[0].info["fixed"] == {"c2": 0.5}


def test_a_model_that_needs_concrete_epochs_runs_one_program_per_light_curve():
    def predict_jax(theta, times):
        t = jnp.asarray(np.asarray(times, dtype=float))      # refuses a traced array
        return theta[0] + theta[1] * t + theta[2] * t * t

    base = _quadratic()
    model = M.Model(name="concrete_quadratic", predict=base.predict, parameters=list(NAMES),
                    default_prior=base.default_prior, predict_jax=predict_jax)
    lcs = [_alert((1.0, 2.0, 0.5), 20, seed=1)[0],
           _alert((1.0, 2.0, 0.5), 20, seed=2, span=(0.5, 3.5))[0]]
    fits = _fit(lcs, model, nsteps=400, burnin=100)
    assert all(f.info["batch"]["n_alerts"] == 1 for f in fits)       # different epochs
    assert fits[0].info["batch"]["program"].startswith("epochs closed over")
    # identical epochs share the closed-over program, so they still run in one call
    same = _fit([lcs[0], _alert((1.0, 2.0, 0.5), 20, seed=3)[0]], model, nsteps=400, burnin=100)
    assert all(f.info["batch"]["n_alerts"] == 2 for f in same)


def test_a_walker_that_never_moves_is_named_and_the_fit_is_not_converged():
    """A frozen walker makes emcee's autocorrelation estimate 0 / 0: no numpy RuntimeWarning
    leaks out, the walker is named, and the fit is not called converged."""
    model = _quadratic()
    lc, mean, _ = _alert((1.0, 2.0, 0.5), 20, sigma=1e-12, seed=6)
    starts = mean + np.random.default_rng(0).normal(0.0, 1e-2, size=(1, 12, 3))
    starts[0, 0] = mean                   # one walker on the peak, the rest ~1e10 sd away
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_batch([lc], model, nwalkers=12, nsteps=60, burnin=20, thin=1, init=starts,
                        metrics=False)[0]
    assert fit.info["frozen_walkers"][:1] == [0]
    assert fit.info["converged"] is False
    assert any("never moved" in p for p in fit.info["convergence_problems"])
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_too_few_points_for_the_parameters_says_not_enough_data():
    model = _quadratic()
    lc = _alert((1.0, 2.0, 0.5), 3, seed=7)[0]                # 3 points, 3 parameters
    with pytest.warns(UserWarning, match="not enough data"):
        fit_batch([lc], model, nwalkers=12, nsteps=300, burnin=100, metrics=False)


def test_upper_limits_in_flux_space():
    model = _quadratic()
    lc, _, _ = _alert((2.0, 1.0, 0.2), 20, seed=5)
    upper = np.zeros(20, dtype=bool)
    upper[:3] = True
    lc["upper_limit"] = upper
    fits = _fit([lc], model, space="flux", likelihood="upper_limits", nsteps=600, burnin=200)
    assert fits[0].info["likelihood"] == "GaussianLikelihoodWithUpperLimits"
    assert np.isfinite(fits[0].max_log_likelihood) and fits[0].n_data == 20


@pytest.mark.slow
def test_supernova_batch_best_log_likelihood_scored_with_redback():
    """The gate: a batch of supernova alerts (free explosion time and redshift), and each
    best fit's log-likelihood recomputed with redback's own light curve at the same parameters."""
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as ra
    from whisper_cbpf.models.jax import _factories as F
    from whisper_cbpf.priors import TruncatedNormal

    bands = ["lsstg", "lsstr", "lssti"]
    prior = Prior({"t_exp": Uniform(-15.0, 0.0),
                   "redshift": TruncatedNormal(0.1, 0.02, 0.04, 0.2)})
    model = F.supernova_model("arnett", bands, free=["t_exp", "redshift"], prior=prior)
    truth = {"f_nickel": 0.1, "mej": 2.0, "vej": 1e4, "kappa": 0.1, "kappa_gamma": 0.1,
             "temperature_floor": 4000.0, "t_exp": -5.0, "redshift": 0.1}
    lcs = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0.0, 30.0, 24))
        b = np.array(bands * 8)
        mag = -2.5 * np.log10(model.predict(truth, t, b) / 3631.0) + rng.normal(0, 0.05, 24)
        lcs.append(wp.LightCurve(time=t, band=b, magnitude=mag, magnitude_err=np.full(24, 0.05)))
    fits = _fit(lcs, model, nwalkers=32, nsteps=1500, burnin=500, thin=5)
    for lc, fit in zip(lcs, fits):
        p = fit.best_params
        cpu = ra.redback_model("arnett", band_names=bands, redshift=p["redshift"],
                               constraint=None)
        flux = cpu.predict({k: p[k] for k in cpu.parameters}, np.asarray(lc.time) - p["t_exp"],
                           np.asarray(lc.band))
        m = -2.5 * np.log10(flux / 3631.0)
        y, s = np.asarray(lc["magnitude"]), np.asarray(lc["magnitude_err"])
        ll = -0.5 * np.sum(((y - m) / s) ** 2) - 0.5 * np.sum(np.log(2 * np.pi * s * s))
        assert abs(ll - fit.max_log_likelihood) < 0.5, (ll, fit.max_log_likelihood)
        assert fit.max_log_likelihood > -0.5 * 24 * np.log(2 * np.pi * 0.05 ** 2) - 30


# --- 5. errors ---------------------------------------------------------------------------------

def test_errors_name_the_cause_and_the_fix():
    model = _quadratic()
    lc = _alert((1.0, 2.0, 0.5), 20, seed=1)[0]
    with pytest.raises(ValueError, match="only 'emcee_jax'|Only 'emcee_jax'"):
        fit_batch([lc], model, sampler="nuts_gpu")
    with pytest.raises(TypeError, match="LIST of light curves"):
        fit_batch(lc, model)
    with pytest.raises(ValueError, match="no light curves"):
        fit_batch([], model)
    with pytest.raises(ValueError, match="burnin"):
        fit_batch([lc], model, nsteps=100, burnin=100)
    with pytest.raises(ValueError, match="keeps no draw"):
        fit_batch([lc], model, nsteps=100, burnin=50, thin=60)
    with pytest.raises(ValueError, match="twice the 3 sampled"):
        fit_batch([lc], model, nwalkers=4)
    with pytest.raises(ValueError, match="one seed per light curve"):
        fit_batch([lc, lc], model, seed=[1, 2, 3])
    with pytest.raises(ValueError, match="one per light curve"):
        fit_batch([lc, lc], model, prior=[model.default_prior])
    with pytest.raises(ValueError, match="unknown init"):
        fit_batch([lc], model, init="box")
    with pytest.raises(ValueError, match="must fit the same parameters"):
        fit_batch([lc, lc], model, prior=[model.default_prior,
                                          Prior({"c0": Uniform(-1, 1), "c1": Uniform(-1, 1),
                                                 "c2": Fixed(0.0)})])
    with pytest.raises(ValueError, match="has no predict_jax"):
        fit_batch([lc], "flare")
