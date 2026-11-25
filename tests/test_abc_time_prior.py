"""``ABC_time_prior=True``: the explosion-time window narrowed by a rejection ABC run.

The rule (:func:`whisper_cbpf.abc_time_prior`): every parameter is drawn from the model's prior and
``t_exp`` from the default window; whisper's ``abc_gpu`` simulates each draw without noise with the
model's JAX version on every detection; a draw is accepted when the RMS of ``data - model`` in flux
is at most 3 times the mean flux error of the detections; the new prior is a flat ``Uniform`` over
the accepted range plus ``max(1 d, 10 %)`` on each side, cut to the default window. The claims,
each a test:

1. **The ABC call**: the threshold is 3 x the mean flux error of the detections (upper limits left
   out), the distance ``rmse`` in flux, no simulated noise, ``t_exp`` drawn from the default window.
2. **The window** is the accepted range plus the margin, cut to the default window (never wider,
   never after the first detection); the margin is ``max(1 d, 10 % of the range)``.
3. **Too few accepted draws**: a plain-English warning naming the counts and the threshold, and
   the default window. Draws the model returns NaN for are never the closest draw (``min_rms``).
4. **Refused**: a ``t_exp`` prior passed with it (``fit`` and ``compare``), a model with no JAX
   version or no ``t_exp`` in ``fit``, a non-bool flag, stray budget keys, a budget without the
   flag, a budget count that is not a whole number >= 1; ``fit`` checks its sampler and
   ``init=`` before the ABC step runs.
5. **A toy JAX model with a known ``t_exp``**: the window holds the truth and is narrower than the
   default; the same seed gives the same window.
6. **``fit``**: the record in ``info["abc_time_prior"]``, the fit's ``t_exp`` prior is the window
   (``mcmc``, ``nested``, ``emcee_jax`` downstream), the seed reaches the ABC step.
7. **``compare``**: each JAX model is fitted in its ABC window and keeps the record; a CPU-only model
   keeps the default window with one warning; with ``parallel=True`` the jobs carry the window.

CPU, small budgets: about two minutes in the CPU container.
"""
import importlib
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf import parallel  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers import abc_window  # noqa: E402
from whisper_cbpf.samplers.jax.abc_gpu import ABCGPUSampler  # noqa: E402

C = importlib.import_module("whisper_cbpf.compare")

T_TRUE = 59996.0
RECORD_KEYS = {"window", "default_window", "accepted_range", "margin_days", "n_simulations",
               "n_accepted", "min_accepted", "threshold", "n_sigma", "mean_flux_err", "min_rms",
               "n_detections", "runtime_s", "fell_back", "seed", "device", "precision", "sampler",
               "distance", "warnings"}


def _rise(p, t, bands=None):
    dt = np.clip(np.asarray(t, float) - p["t_exp"], 0.0, None)
    return p["amp"] * 1e-4 * (1 - np.exp(-dt / 3.0)) * np.exp(-dt / 20.0)


def _rise_jax(theta, t):
    dt = jnp.clip(jnp.asarray(t) - theta[1], 0.0, None)
    return theta[0] * 1e-4 * (1 - jnp.exp(-dt / 3.0)) * jnp.exp(-dt / 20.0)


PRIOR = Prior({"amp": Uniform(0.5, 2.0), "t_exp": Uniform(59900.0, 60100.0)})
MODEL = wp.register_model("abc_window_rise", _rise, ["amp", "t_exp"], prior=PRIOR,
                          predict_jax=_rise_jax, overwrite=True)
CPU_MODEL = wp.register_model("abc_window_rise_cpu", _rise, ["amp", "t_exp"], prior=PRIOR,
                              overwrite=True)


def _lc(err=2e-6, limits=(), seed=1, noise=True, scale=1.0, offset=0.0):
    t = 60000.0 + np.arange(0.0, 40.0, 2.0)
    flux = scale * _rise({"amp": 1.0, "t_exp": T_TRUE}, t) + offset
    err = np.broadcast_to(np.asarray(err, float), t.shape)
    if noise:
        flux = flux + np.random.default_rng(seed).normal(0.0, err, t.size)
    lim = np.asarray(limits, float)
    tt = np.concatenate([lim, t])
    ul = np.concatenate([np.ones(lim.size, bool), np.zeros(t.size, bool)])
    return wp.LightCurve(time=tt, band=["r"] * tt.size,
                         flux=np.concatenate([np.full(lim.size, 1e-6), flux]),
                         flux_err=np.concatenate([np.full(lim.size, np.nan),
                                                  err]),
                         upper_limit=ul)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


class _FakeABC:
    """Stands in for ``ABCGPUSampler.fit``: records the call, returns chosen accepted ``t_exp``."""

    def __init__(self, accepted, min_distance=1e-6):
        self.accepted, self.min_distance, self.calls = list(accepted), min_distance, []

    def __call__(self, lc, model, prior=None, **kw):
        self.calls.append(dict(kw, lc=lc, model=model, prior=prior))
        samples = pd.DataFrame({"amp": np.ones(len(self.accepted)), "t_exp": self.accepted})
        return SimpleNamespace(samples=samples, min_distance=self.min_distance,
                               info={"device": "cpu:0", "precision": "float64"})


@pytest.fixture
def x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ------------------------------------------------------------------------------ 1. the ABC call
def test_the_abc_call_uses_three_times_the_mean_error_of_the_detections(monkeypatch):
    fake = _FakeABC(np.linspace(59985.0, 59995.0, 60))
    monkeypatch.setattr(ABCGPUSampler, "fit", fake)
    lc = _lc(err=np.linspace(1e-6, 3e-6, 20), limits=(59990.0,), noise=False)   # limit: NaN
    window, rec = wp.abc_time_prior(lc, MODEL, n_simulations=1234, seed=5)
    (call,) = fake.calls
    assert call["threshold"] == pytest.approx(3.0 * 2e-6) == rec["threshold"]
    assert rec["mean_flux_err"] == pytest.approx(2e-6) and rec["n_sigma"] == 3.0
    assert (call["distance"], call["space"], call["simulate_noise"]) == ("rmse", "flux", False)
    assert (call["n_simulations"], call["seed"]) == (1234, 5)
    assert len(call["lc"]) == 20 and not np.any(call["lc"].upper_limit)    # detections only
    t_exp = call["prior"].distributions["t_exp"]
    assert t_exp.bounds == wp.default_t_exp_prior(lc).bounds == (59960.0, 60000.0)
    assert call["prior"].distributions["amp"] is PRIOR.distributions["amp"]


# --------------------------------------------------------------------------------- 2. the window
def test_the_window_is_the_accepted_range_plus_the_margin(monkeypatch):
    monkeypatch.setattr(ABCGPUSampler, "fit", _FakeABC(np.linspace(59980.0, 59990.0, 60)))
    window, rec = wp.abc_time_prior(_lc(noise=False), MODEL)
    assert rec["accepted_range"] == [59980.0, 59990.0] and rec["margin_days"] == 1.0
    assert window.bounds == (59979.0, 59991.0) and rec["window"] == [59979.0, 59991.0]
    assert window.name == "t_exp" and not rec["fell_back"]


def test_a_wide_range_takes_ten_percent_and_is_cut_to_the_default_window(monkeypatch):
    monkeypatch.setattr(ABCGPUSampler, "fit", _FakeABC(np.linspace(59972.0, 59999.5, 60)))
    window, rec = wp.abc_time_prior(_lc(noise=False), MODEL)
    assert rec["margin_days"] == pytest.approx(2.75)
    assert window.bounds == (59970.0, 60000.0)          # never wider, never after the first point
    assert rec["default_window"] == [59970.0, 60000.0]


def test_the_margin_rule():
    assert abc_window.window_margin(0.0) == 1.0
    assert abc_window.window_margin(9.0) == 1.0
    assert abc_window.window_margin(30.0) == pytest.approx(3.0)


# ------------------------------------------------------------------------- 3. too few accepted
def test_nothing_accepted_warns_and_keeps_the_default_window():
    lc = _lc(err=1e-9, noise=False, scale=1.5, offset=2e-6)   # no draw fits within 3 sigma
    with pytest.warns(UserWarning, match=r"ABC_time_prior: only 0 of 2,000 draws fit within 3 x "
                                         r"the mean error \(RMS needs <= 0\.003 uJy") as caught:
        window, rec = wp.abc_time_prior(lc, MODEL, n_simulations=2000)
    assert [str(w.message) for w in caught if "ABC_time_prior" in str(w.message)][0].endswith(
        "using the default t_exp window [59970, 60000].")
    assert window.bounds == (59970.0, 60000.0) and rec["fell_back"] is True
    assert rec["n_accepted"] == 0 and rec["accepted_range"] is None and rec["margin_days"] is None
    assert any("accepted 0 of 2000" in w for w in rec["warnings"])     # abc_gpu's, kept here


def test_fewer_than_min_accepted_falls_back(monkeypatch):
    monkeypatch.setattr(ABCGPUSampler, "fit", _FakeABC(np.linspace(59980.0, 59990.0, 12)))
    with pytest.warns(UserWarning, match="only 12 of 1,000,000 draws"):
        window, rec = wp.abc_time_prior(_lc(noise=False), MODEL)
    assert rec["fell_back"] and window.bounds == (59970.0, 60000.0) and rec["min_accepted"] == 50
    monkeypatch.setattr(ABCGPUSampler, "fit", _FakeABC(np.linspace(59980.0, 59990.0, 12)))
    window, rec = wp.abc_time_prior(_lc(noise=False), MODEL, min_accepted=10)
    assert not rec["fell_back"] and window.bounds == (59979.0, 59991.0)


# ------------------------------------------------------------------------------- 4. refused
def test_a_t_exp_prior_passed_with_it_is_refused():
    own = Prior({"amp": Uniform(0.5, 2.0), "t_exp": Uniform(59980.0, 60000.0)})
    with pytest.raises(ValueError, match="the prior passed also names t_exp"):
        wp.fit(_lc(), MODEL, sampler="abc", ABC_time_prior=True, prior=own)
    with pytest.raises(ValueError, match="the prior passed also names t_exp"):
        wp.abc_time_prior(_lc(), MODEL, own)
    with pytest.raises(ValueError, match="the prior passed also names t_exp"):
        wp.compare(_lc(), [MODEL], sampler="abc", ABC_time_prior=True, prior=own)


def test_a_model_with_no_jax_version_or_no_t_exp_is_refused_by_fit():
    with pytest.raises(ValueError, match="has no JAX version"):
        wp.fit(_lc(), CPU_MODEL, sampler="abc", ABC_time_prior=True)
    with pytest.raises(ValueError, match="has no t_exp parameter"):
        wp.fit(_lc(), "flare", sampler="abc", ABC_time_prior=True)


def test_bad_flags_and_budgets_are_refused():
    with pytest.raises(ValueError, match="ABC_time_prior must be True or False"):
        wp.fit(_lc(), MODEL, sampler="abc", ABC_time_prior="yes")
    with pytest.raises(ValueError, match="pass ABC_time_prior=True with it"):
        wp.fit(_lc(), MODEL, sampler="abc", ABC_time_prior_kwargs={"n_simulations": 10})
    with pytest.raises(TypeError, match=r"ABC_time_prior_kwargs takes .*got \['nsim'\]"):
        wp.fit(_lc(), MODEL, sampler="abc", ABC_time_prior=True,
               ABC_time_prior_kwargs={"nsim": 10})
    with pytest.raises(ValueError, match="pass ABC_time_prior=True with it"):
        wp.compare(_lc(), [MODEL], ABC_time_prior_kwargs={"n_simulations": 10})


@pytest.mark.parametrize("budget", [{"n_simulations": 0}, {"n_simulations": -5},
                                    {"n_simulations": 1500.5}, {"n_simulations": "many"},
                                    {"n_simulations": True}, {"min_accepted": 0},
                                    {"min_accepted": -1}])
def test_a_budget_count_below_one_is_refused_before_the_abc_run(monkeypatch, budget):
    fake = _FakeABC([])
    monkeypatch.setattr(ABCGPUSampler, "fit", fake)
    (name, value), = budget.items()
    with pytest.raises(ValueError, match=rf"ABC_time_prior: {name} must be a whole number of at "
                                         rf"least 1; got {value!r}\."):
        wp.abc_time_prior(_lc(), MODEL, **budget)
    with pytest.raises(ValueError, match=rf"{name} must be a whole number"):
        wp.fit(_lc(), MODEL, sampler="abc", ABC_time_prior=True, ABC_time_prior_kwargs=budget)
    assert fake.calls == []


def test_a_float_budget_that_is_whole_is_taken(monkeypatch):
    fake = _FakeABC(np.linspace(59980.0, 59990.0, 60))
    monkeypatch.setattr(ABCGPUSampler, "fit", fake)
    _, rec = wp.abc_time_prior(_lc(), MODEL, n_simulations=2e3, min_accepted=50.0)
    assert fake.calls[0]["n_simulations"] == 2000 and rec["n_simulations"] == 2000
    assert rec["min_accepted"] == 50


def test_fit_checks_the_sampler_and_init_before_the_abc_step(monkeypatch):
    fake = _FakeABC(np.linspace(59980.0, 59990.0, 60))
    monkeypatch.setattr(ABCGPUSampler, "fit", fake)
    with pytest.raises(KeyError):
        wp.fit(_lc(), MODEL, sampler="no_such_sampler", ABC_time_prior=True)
    with pytest.raises(ValueError, match="has no starting point to set"):
        wp.fit(_lc(), MODEL, sampler="nested", init="prior", ABC_time_prior=True)
    assert fake.calls == []                                   # no ABC run was wasted


def _rise_nan_jax(theta, t):
    """``_rise_jax``, but NaN for ``amp > 1.8``: a draw the model cannot evaluate."""
    return jnp.where(theta[0] > 1.8, jnp.nan, _rise_jax(theta, t))


NAN_MODEL = wp.register_model("abc_window_rise_nan", _rise, ["amp", "t_exp"], prior=PRIOR,
                              predict_jax=_rise_nan_jax, overwrite=True)
ALL_NAN_MODEL = wp.register_model("abc_window_rise_all_nan", _rise, ["amp", "t_exp"],
                                  prior=PRIOR, predict_jax=lambda theta, t: _rise_jax(theta, t)
                                  * jnp.nan, overwrite=True)


def test_draws_the_model_returns_nan_for_are_never_the_closest(x64):
    window, rec = _quiet(wp.abc_time_prior, _lc(), NAN_MODEL, n_simulations=5000)
    res = _quiet(ABCGPUSampler().fit, _lc(), NAN_MODEL, n_simulations=5000, threshold=1e-5,
                 distance="rmse", simulate_noise=False, space="flux")
    assert np.isfinite(rec["min_rms"]) and np.isfinite(res.min_distance)
    assert res.min_distance == pytest.approx(float(res.samples["distance"].min()))
    assert not rec["fell_back"] and window.low < T_TRUE < window.high
    bad = _lc(err=1e-9, noise=False, scale=1.5, offset=2e-6)     # nothing accepted
    with pytest.warns(UserWarning, match=r"the closest draw has \d[\d.e+-]* uJy;"):
        _, rec = wp.abc_time_prior(bad, NAN_MODEL, n_simulations=500)
    assert np.isfinite(rec["min_rms"])
    with pytest.warns(UserWarning, match=r"uJy, no draw gave a finite RMS; at least 50"):
        _, rec = wp.abc_time_prior(bad, ALL_NAN_MODEL, n_simulations=500)
    assert np.isnan(rec["min_rms"]) and rec["fell_back"]


# -------------------------------------------------------------------- 5. a toy with known t_exp
def test_the_window_holds_the_true_t_exp_and_is_narrower_than_the_default(x64):
    window, rec = wp.abc_time_prior(_lc(), MODEL, n_simulations=20000)
    assert not rec["fell_back"] and rec["n_accepted"] >= 50
    assert window.low < T_TRUE < window.high
    assert window.high - window.low < 30.0
    assert rec["min_rms"] <= rec["threshold"]
    assert set(rec) == RECORD_KEYS


def test_the_same_seed_gives_the_same_window(x64):
    a = wp.abc_time_prior(_lc(), MODEL, n_simulations=5000, seed=3)
    b = wp.abc_time_prior(_lc(), MODEL, n_simulations=5000, seed=3)
    c = wp.abc_time_prior(_lc(), MODEL, n_simulations=5000, seed=4)
    assert a[0].bounds == b[0].bounds and a[1]["n_accepted"] == b[1]["n_accepted"]
    assert c[0].bounds != a[0].bounds


# ------------------------------------------------------------------------------------- 6. fit
@pytest.mark.parametrize("sampler, kw", [
    ("mcmc", dict(nsteps=400, burnin=100, check_convergence=False)),
    ("nested", dict(nlive=50)),
    ("emcee_jax", dict(nsteps=300, burnin=100, nwalkers=16, check_convergence=False)),
])
def test_fit_uses_the_window_and_records_it(x64, sampler, kw):
    budget = {"n_simulations": 5000}
    res = _quiet(wp.fit, _lc(), MODEL, sampler=sampler, seed=2, ABC_time_prior=True,
                 ABC_time_prior_kwargs=budget, **kw)
    rec = res.info["abc_time_prior"]
    assert set(rec) == RECORD_KEYS and rec["seed"] == 2 and rec["n_simulations"] == 5000
    tp = res.info["t_exp_prior"]
    assert tp["source"] == "ABC_time_prior" and [tp["low"], tp["high"]] == rec["window"]
    t = np.asarray(res.samples["t_exp"], float)
    assert np.all((t >= rec["window"][0]) & (t <= rec["window"][1]))
    window, _ = wp.abc_time_prior(_lc(), MODEL, n_simulations=5000, seed=2)
    assert window.bounds == tuple(rec["window"])              # the helper's window, seed 2


def test_fit_keeps_a_prior_for_the_other_parameters(x64, monkeypatch):
    monkeypatch.setattr(ABCGPUSampler, "fit", _FakeABC(np.linspace(59980.0, 59990.0, 60)))
    own = Prior({"amp": Uniform(0.8, 1.2)})
    res = _quiet(wp.fit, _lc(), MODEL, sampler="abc", ABC_time_prior=True, prior=own,
                 n_simulations=500, quantile=0.2)
    amp = np.asarray(res.samples["amp"], float)
    assert np.all((amp >= 0.8) & (amp <= 1.2))
    assert res.info["t_exp_prior"]["low"] == 59979.0


def test_without_the_flag_nothing_changes():
    res = _quiet(wp.fit, _lc(), MODEL, sampler="abc", n_simulations=500, quantile=0.2)
    assert "abc_time_prior" not in res.info
    assert res.info["t_exp_prior"]["source"] != "ABC_time_prior"


# --------------------------------------------------------------------------------- 7. compare
def test_compare_fits_each_jax_model_in_its_window_and_warns_once_for_a_cpu_model(x64):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cmp = wp.compare(_lc(), [MODEL, CPU_MODEL], sampler="abc", n_simulations=2000,
                         quantile=0.1, evidence_check=False, likelihood_max_opt=False,
                         ABC_time_prior=True, ABC_time_prior_kwargs={"n_simulations": 5000})
    said = [str(w.message) for w in caught if "ABC_time_prior" in str(w.message)]
    assert said == ["ABC_time_prior: 'abc_window_rise_cpu' has no JAX version, so the ABC step "
                    "cannot run for it; the default t_exp window is used instead."]
    jax_res, cpu_res = cmp.results["abc_window_rise"], cmp.results["abc_window_rise_cpu"]
    rec = jax_res.info["abc_time_prior"]
    assert jax_res.info["t_exp_prior"]["source"] == "ABC_time_prior"
    assert [jax_res.info["t_exp_prior"]["low"], jax_res.info["t_exp_prior"]["high"]] == \
        rec["window"]
    assert "abc_time_prior" not in cpu_res.info
    assert (cpu_res.info["t_exp_prior"]["low"], cpu_res.info["t_exp_prior"]["high"]) == \
        (59970.0, 60000.0)
    assert cmp.settings["ABC_time_prior"] is True
    assert cmp.settings["ABC_time_prior_kwargs"] == {"n_simulations": 5000}


def test_compare_in_parallel_sends_the_window_with_each_job(x64, monkeypatch, tmp_path):
    seen = []

    def fake_run_jobs(jobs, *, gpus, cpu_cores, cache_dir):
        seen.extend(jobs)
        raise RuntimeError("stop after the jobs are built")

    monkeypatch.setattr(ABCGPUSampler, "fit", _FakeABC(np.linspace(59980.0, 59990.0, 60)))
    monkeypatch.setattr(parallel, "run_jobs", fake_run_jobs)
    with pytest.raises(RuntimeError, match="stop after the jobs are built"):
        _quiet(wp.compare, _lc(), ["abc_window_rise", "abc_window_rise_cpu"], sampler="abc",
               parallel=True, gpus=None, ABC_time_prior=True, n_simulations=500)
    jobs = {j.model.name: j for j in seen}
    assert jobs["abc_window_rise"].kwargs["prior"].distributions["t_exp"].bounds == \
        (59979.0, 59991.0)
    assert "prior" not in jobs["abc_window_rise_cpu"].kwargs          # the default window
