"""Nested sampler (dynesty): registration, recovery, log-evidence correctness, and the wiring
guards that separate it from ``mcmc``.

Everything here runs on the minimal CPU install -- ``flare`` / ``gaussian_rise`` (numpy,
module-level predicts), Whisper's own ``Prior``, and ``dynesty`` (a core dependency). No JAX, no
redback, no sbi, no torch.

The decisive correctness test is :func:`test_log_evidence_matches_an_analytic_integral`: a constant
model against zero data has a closed-form evidence, and the sampler is asked to reproduce it
through the *public* ``wp.fit(..., sampler="nested")`` path. (The alternative the brief allowed --
ranking two models the same way a large-budget AIC does -- was not needed, because the analytic
integral was practical.)
"""
import json

import numpy as np
import pytest
from scipy.special import erf

import whisper_cbpf as wp
from whisper_cbpf.likelihood import make_likelihood
from whisper_cbpf.models import get_model
from whisper_cbpf.models.flare import flare_flux
from whisper_cbpf.priors import LogUniform, Prior, Uniform
from whisper_cbpf.samplers import get_sampler


def _synthetic(truth, n=40, noise_frac=0.02, seed=0):
    times = np.linspace(0.5, 30, n)
    flux = flare_flux(truth, times, None)
    err = np.full_like(flux, noise_frac * flux.max()) + 1e-9
    noisy = flux + np.random.default_rng(seed).normal(0, err)
    return wp.LightCurve(time=times, band=["r"] * n, flux=noisy, flux_err=err, name="synth")


TRUTH = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}

# --- fixtures for the analytic-evidence test -----------------------------------------------------
# Module level (not a closure) so the model is picklable, per whisper_cbpf.models' rule.
_CONST_N = 8
_CONST_HALFWIDTH = 5.0


def _constant_predict(parameters, times, bands=None):
    """A constant flux -- the simplest model whose evidence integral is closed-form."""
    return np.full(np.shape(times), float(parameters["amplitude"]))


def _constant_lc():
    return wp.LightCurve(time=np.linspace(1.0, float(_CONST_N), _CONST_N),
                         band=["r"] * _CONST_N, flux=np.zeros(_CONST_N), flux_err=np.ones(_CONST_N))


def _constant_analytic_log_evidence():
    r"""Closed form for ``flux_err = 1``, ``flux = 0``, ``amplitude ~ Uniform(-w, w)``.

    ``L(A) = (2 pi)^{-n/2} exp(-n A^2 / 2)`` (whisper's GaussianLikelihood carries the full
    normalisation), so

        ``Z = (2 pi)^{-n/2} sqrt(2 pi / n) erf(w sqrt(n/2)) / (2 w)``.

    At ``n = 8``, ``w = 5`` this is ``ln Z = -9.7749``.
    """
    n, w = _CONST_N, _CONST_HALFWIDTH
    return float(-0.5 * n * np.log(2 * np.pi)
                 + np.log(np.sqrt(2 * np.pi / n) * erf(w * np.sqrt(n / 2.0)) / (2 * w)))


class _NoRescale:
    """A prior distribution with everything ``mcmc`` needs and nothing ``nested`` needs."""

    def __init__(self, low, high):
        self.low, self.high = float(low), float(high)

    def sample(self, rng):
        return float(rng.uniform(self.low, self.high))

    def log_prob(self, x):
        return -np.log(self.high - self.low) if self.low <= x <= self.high else -np.inf

    @property
    def bounds(self):
        return (self.low, self.high)


# --- 1. registry ---------------------------------------------------------------------------------
def test_nested_registered():
    """Both names resolve to the same zero-argument-constructible sampler (the registry contract)."""
    names = wp.list_samplers()
    assert "nested" in names and "dynesty" in names
    assert type(get_sampler("nested")) is type(get_sampler("dynesty"))
    assert get_sampler("nested").name == "nested"
    assert callable(wp.fit_nested)


# --- 2. recovery + reproducibility -------------------------------------------------------------
def test_nested_recovers_and_reproducible():
    lc = _synthetic(TRUTH)
    r1 = wp.fit_nested(lc, "flare", nlive=200, dlogz=0.5, seed=0)
    assert r1.sampler == "nested" and r1.n_samples > 0
    assert abs(r1.summary["amplitude"]["median"] - 5.0) < 1.0            # recovery
    assert set(r1.best_params) == {"amplitude", "rise_time", "decay_time"}
    assert np.isfinite(r1.aic) and np.isfinite(r1.max_log_likelihood)
    r2 = wp.fit_nested(lc, "flare", nlive=200, dlogz=0.5, seed=0)
    assert r1.samples.equals(r2.samples)                                # reproducible (fixed seed)
    json.loads(r1.to_json())


# --- 3. the log-evidence is reported and is nested-only -----------------------------------------
def test_nested_returns_log_evidence():
    lc = _synthetic(TRUTH)
    r = wp.fit_nested(lc, "flare", nlive=200, dlogz=0.5, seed=0)
    assert np.isfinite(r.info["log_evidence"]) and r.info["log_evidence_err"] > 0
    assert r.info["information_nats"] > 0                               # H > 0 for informative data
    assert 0 < r.info["n_effective"] <= r.n_samples                     # duplicates from resampling
    # It reaches JSON, which is what a comparison table consumes.
    d = r.to_dict()
    assert np.isfinite(d["info"]["log_evidence"])
    # No other sampler produces one; nothing downstream may assume the key exists.
    rm = wp.fit_MCMC(lc, "flare", nsteps=200, burnin=50, thin=2, seed=0)
    assert "log_evidence" not in rm.info


# --- 4. THE correctness test: ln Z against a closed-form integral --------------------------------
def test_log_evidence_matches_an_analytic_integral():
    """Constant model, ``n=8`` zeros with unit errors, ``amplitude ~ Uniform(-5, 5)``.

    ``ln Z = -9.7749`` exactly. Run through the public ``wp.fit`` dispatch so the registry, the
    prior transform, the likelihood layer and the result assembly are all in the path. Measured at
    ``nlive=200, seed=0``: ``-9.8265 +/- 0.1403`` (0.37 sigma), so the 4-sigma band below is
    generous rather than tuned.
    """
    wp.register_model("_nested_test_constant", _constant_predict, ["amplitude"], overwrite=True)
    prior = Prior({"amplitude": Uniform(-_CONST_HALFWIDTH, _CONST_HALFWIDTH)})
    r = wp.fit(_constant_lc(), "_nested_test_constant", sampler="nested", prior=prior,
               nlive=200, seed=0)
    exact = _constant_analytic_log_evidence()
    assert exact == pytest.approx(-9.7749, abs=1e-4)                    # the closed form itself
    assert abs(r.info["log_evidence"] - exact) < 4 * r.info["log_evidence_err"]


# --- 5. max_log_likelihood is the pure likelihood, not the posterior ----------------------------
def test_max_log_likelihood_is_the_pure_likelihood_not_the_posterior():
    """``res.logl`` never saw the prior, so no ``log_prob - prior.log_prob`` reconstruction is
    needed. With a LogUniform in the prior the two would differ, which is exactly the bias
    ``mcmc.py`` needs five lines to correct."""
    lc = _synthetic(TRUTH)
    prior = Prior({"amplitude": Uniform(0.0, 10.0), "rise_time": LogUniform(0.5, 10.0),
                   "decay_time": LogUniform(5.0, 30.0)})
    r = wp.fit_nested(lc, "flare", prior=prior, nlive=150, dlogz=0.5, seed=0)
    model = get_model("flare")
    flux = model.predict(r.best_params, np.asarray(lc.time, float), np.asarray(lc.band))
    by_hand = make_likelihood(lc).log_likelihood(flux)
    assert r.max_log_likelihood == pytest.approx(by_hand)
    # and it is NOT the log-posterior (the LogUniform terms are non-zero here)
    assert abs(prior.log_prob(r.best_params)) > 1e-6


# --- 6. the prior enters exactly once (and the Occam caveat, executable) ------------------------
def test_prior_enters_exactly_once():
    """Widening ``amplitude``'s prior 100x must move ``ln Z`` by ``-ln 100`` and leave AIC alone.

    If ``_log_likelihood`` ever grows a ``prior.log_prob`` term the prior is counted twice and the
    first assertion fails at once. Measured: ``dlnZ = -4.095 +/- 0.442`` against ``-4.605``
    (1.2 sigma); ``dAIC = -0.010``.
    """
    lc = _synthetic(TRUTH)
    narrow = Prior({"amplitude": Uniform(0.0, 10.0), "rise_time": Uniform(1.0, 10.0),
                    "decay_time": Uniform(5.0, 30.0)})
    wide = Prior({"amplitude": Uniform(0.0, 1000.0), "rise_time": Uniform(1.0, 10.0),
                  "decay_time": Uniform(5.0, 30.0)})
    rn = wp.fit_nested(lc, "flare", prior=narrow, nlive=400, seed=0)
    rw = wp.fit_nested(lc, "flare", prior=wide, nlive=400, seed=0)
    dz = rw.info["log_evidence"] - rn.info["log_evidence"]
    err = np.hypot(rw.info["log_evidence_err"], rn.info["log_evidence_err"])
    assert abs(dz - (-np.log(100.0))) < 4 * err
    assert abs(rw.aic - rn.aic) < 0.5                    # AIC/BIC do not see the prior volume
    assert abs(rw.bic - rn.bic) < 0.5


# --- 7. equal-weight resampling contract --------------------------------------------------------
def test_equal_weight_samples_match_dynesty_resample_equal():
    import dynesty

    lc = _synthetic(TRUTH)
    r = wp.fit_nested(lc, "flare", nlive=150, dlogz=0.5, seed=0)
    res = r.dynesty_results
    expected = dynesty.utils.resample_equal(res.samples, res.importance_weights(),
                                            rstate=np.random.default_rng(0))
    assert np.allclose(r.samples[r.parameters].to_numpy(), expected)
    assert len(r.samples) == res.samples.shape[0]        # one row per dead point, with duplicates


# --- 8. data mode -> likelihood space -----------------------------------------------------------
def test_nested_uses_data_mode_for_likelihood_space():
    """Reuses the shared likelihood layer, so magnitude data is fit in magnitude space."""
    mlc = _synthetic(TRUTH).add_mag()
    mlc.meta["data_mode"] = "magnitude"
    r = wp.fit_nested(mlc, "flare", nlive=100, dlogz=1.0, seed=0)
    assert r.info["space"] == "magnitude"


# --- 9. scatter parameter routing ---------------------------------------------------------------
def _scatter_lc(n=40, err=0.1, extra=0.3, seed=3):
    m = get_model("gaussian_rise")
    truth = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}
    t = np.linspace(0.1, 30, n)
    obs = m.predict(truth, t, None) + np.random.default_rng(seed).normal(
        0, np.sqrt(err ** 2 + extra ** 2), t.shape)
    return wp.LightCurve(time=t, band=["r"] * n, flux=obs, flux_err=np.full_like(t, err))


def test_scatter_param_is_routed_to_the_likelihood():
    """A likelihood-based sampler CAN fit extra scatter -- unlike distance-based ABC, where the
    posterior rails to the smallest allowed value (see docs/CHOOSING.md)."""
    prior = Prior({"amplitude": Uniform(1, 10), "t0": Uniform(2, 15), "sigma_rise": Uniform(0.5, 8),
                   "tau_decay": Uniform(5, 40), "sigma": LogUniform(0.01, 2.0)})
    r = wp.fit_nested(_scatter_lc(), "gaussian_rise", prior=prior, nlive=250, dlogz=0.5,
                      space="flux", likelihood="gaussian_scatter", seed=0)
    assert "sigma" in r.parameters and "sigma" in r.summary
    assert r.info["scatter_param"] == "sigma"
    assert r.n_params == 5                                              # sigma counted in AIC/BIC
    assert 0.1 < r.summary["sigma"]["median"] < 0.9                     # true extra scatter 0.3


# --- 10. refuse a prior that cannot map the unit cube -------------------------------------------
def test_refuses_a_prior_without_rescale():
    lc = _synthetic(TRUTH)
    prior = Prior({"amplitude": Uniform(0.0, 10.0), "rise_time": _NoRescale(1.0, 10.0),
                   "decay_time": Uniform(5.0, 30.0)})
    with pytest.raises(TypeError, match="rescale") as excinfo:
        wp.fit_nested(lc, "flare", prior=prior, nlive=50, seed=0)
    msg = str(excinfo.value)
    assert "rise_time" in msg and "_NoRescale" in msg                   # names the offender
    assert "mcmc" in msg                                                # and the way out


# --- 11. parallel agrees with serial, and is reproducible at fixed n_jobs ------------------------
def test_parallel_matches_serial_within_the_quoted_error():
    """Contracts: (a) the two evidences agree within the quoted errors, (b) a fixed
    ``(seed, n_jobs)`` reproduces. Equality of serial and parallel samples is deliberately NOT a
    contract -- dynesty seeds per worker, so ``n_jobs`` is part of the RNG stream."""
    lc = _synthetic(TRUTH)
    rs = wp.fit_nested(lc, "flare", nlive=100, dlogz=1.0, seed=0)
    rp = wp.fit_nested(lc, "flare", nlive=100, dlogz=1.0, seed=0, n_jobs=2)
    dz = rp.info["log_evidence"] - rs.info["log_evidence"]
    err = np.hypot(rp.info["log_evidence_err"], rs.info["log_evidence_err"])
    assert abs(dz) < 4 * err
    assert rp.info["n_jobs"] == 2 and rs.info["n_jobs"] == 1
    rp2 = wp.fit_nested(lc, "flare", nlive=100, dlogz=1.0, seed=0, n_jobs=2)
    assert rp.samples.equals(rp2.samples)


# --- 12. dynamic agrees with static -------------------------------------------------------------
def test_dynamic_agrees_with_static_on_log_evidence():
    lc = _synthetic(TRUTH)
    rs = wp.fit_nested(lc, "flare", nlive=150, dlogz=0.5, seed=0)
    rd = wp.fit_nested(lc, "flare", nlive=150, dynamic=True, maxbatch=1, seed=0)
    dz = rd.info["log_evidence"] - rs.info["log_evidence"]
    err = np.hypot(rd.info["log_evidence_err"], rs.info["log_evidence_err"])
    assert abs(dz) < 4 * err
    assert rd.info["dynamic"] is True and rd.info["pfrac"] == 0.8
    assert rs.info["dynamic"] is False and rs.info["pfrac"] is None
    assert rs.info["dlogz"] == 0.5 and rd.info["dlogz"] is None


# --- 13. the result serialises ------------------------------------------------------------------
def test_result_survives_to_json():
    """``dynesty_results`` holds ndarrays, so it must be an attribute, never an ``info`` entry."""
    lc = _synthetic(TRUTH)
    r = wp.fit_nested(lc, "flare", nlive=100, dlogz=1.0, seed=0)
    json.loads(r.to_json())
    d = r.to_dict()
    assert "dynesty_results" not in d and "dynesty_results" not in d["info"]
    assert "log_evidence" in d["info"] and "log_evidence_err" in d["info"]


# --- 14. cross-sampler agreement (the strongest single wiring check) ----------------------------
def test_nested_agrees_with_mcmc():
    """Two exact-likelihood samplers over the same prior must land in the same place. Tighter than
    ``test_mcmc.py``'s 50% ABC tolerance for exactly that reason: a wrong prior transform or a
    mis-wired likelihood shows up here."""
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=30)
    rn = wp.fit_nested(lc, "flare", nlive=300, seed=0)
    rm = wp.fit_MCMC(lc, "flare", nsteps=3000, burnin=800, thin=3, seed=0)
    for p in ("amplitude", "rise_time", "decay_time"):
        a, b = rn.summary[p]["median"], rm.summary[p]["median"]
        assert abs(a - b) < 0.2 * abs(b) + 0.2


# --- 15. a model with a constraint wall: the prior inside it is renormalised --------------------
class _WalledLine:
    """``flux = x + y t``, behind the wall ``x + y < 1`` (half of the Uniform(0, 1)^2 prior).

    Shaped like the JAX factories' ``predict_jax`` (``.ctx`` with ``constraint_model``, ``params``
    and ``physical``), which is how the nested sampler finds a model's wall. Module level, so it
    pickles."""

    constraint_model = "toy"
    params = ["x", "y"]

    def __init__(self):
        self.ctx = self

    def physical(self, free):
        return bool(free[0] + free[1] < 1.0)

    def __call__(self, theta, times, band_idx=None):
        return theta[0] + theta[1] * np.asarray(times)


def _walled_predict(parameters, times, bands=None):
    return float(parameters["x"]) + float(parameters["y"]) * np.asarray(times, dtype=float)


def test_a_constrained_prior_is_normalised_inside_its_wall():
    """Two points pin (x, y) near (0.2, 0.2), well inside the wall: the likelihood integrates to 1
    over the plane, so ``ln Z = -ln f = ln 2`` with the prior renormalised to the allowed half, and
    ``ln Z = 0`` without (what every walled model's ln Z was, low by ``-ln f``)."""
    model = wp.Model(name="walled_line", predict=_walled_predict, parameters=["x", "y"],
                     default_prior=Prior({"x": Uniform(0.0, 1.0), "y": Uniform(0.0, 1.0)}),
                     predict_jax=_WalledLine())
    lc = wp.LightCurve(time=[0.0, 1.0], band=["r", "r"], flux=[0.2, 0.4], flux_err=[0.02, 0.02])
    r = wp.fit(lc, model, sampler="nested", nlive=200, seed=0)
    wall = r.info["constraint_prior"]
    assert abs(wall["allowed_fraction"] - 0.5) < 0.02
    assert abs(wall["log_evidence_before_correction"]) < 4 * r.info["log_evidence_err"]
    assert abs(r.info["log_evidence"] - np.log(2.0)) < 4 * r.info["log_evidence_err"]
    assert wall["log_evidence_correction"] == pytest.approx(-np.log(wall["allowed_fraction"]))
    free = wp.fit(lc, "flare", sampler="nested", nlive=50, dlogz=5.0, seed=0,
                  prior=Prior({"amplitude": Uniform(0.0, 1.0), "rise_time": Uniform(1.0, 2.0),
                               "decay_time": Uniform(1.0, 2.0)}))
    assert free.info["constraint_prior"] is None            # no wall, nothing to renormalise


def test_the_physical_models_walls_are_found():
    pytest.importorskip("jax")
    import jax

    from whisper_cbpf.samplers import nested as N

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from whisper_cbpf.models.jax import supernova_model
        walled = supernova_model("arnett", ["lsstg"], redshift=0.1, dl_cm=1.4e27)
        open_ = supernova_model("arnett", ["lsstg"], redshift=0.1, dl_cm=1.4e27, constraint=None)
        wall = N._wall_check(walled)
        assert wall is not None and N._wall_check(open_) is None
        assert N._wall_check(get_model("flare")) is None
        names = list(walled.parameters)
        f, n_ok, n = N._allowed_fraction(wall, walled.default_prior, names, {}, 0, n=4000)
        assert 0.33 < f < 0.45 and n == 4000             # 0.392 on 20 000 draws (review probe)
        bad = next(r for r in N._dg.prior_draws(walled.default_prior, names, 200, 1)
                   if not wall(dict(zip(names, r))))
        assert N._log_likelihood(bad, names, walled.predict, np.array([5.0]), np.array(["lsstg"]),
                                 None, wall=wall) == -np.inf
    finally:
        jax.config.update("jax_enable_x64", old)
