"""Known answers and regressions for the NUTS samplers' posteriors, starts and diagnostics.

The user's report: NUTS "does not return the true posterior on any synthetic mock problem". The
mock-problem matrix (2 672 fits before the fix)
reproduced it in float64 with a relative clock and default priors, CPU and GPU alike: on a
Gaussian bump 22 / 100 ``nuts_gpu`` runs had chains stranded in a local optimum (the bump shrunk
to its width floor and hidden between epochs), pooled t0 a median 18.6 sd off, while whisper said
``rhat={}`` and ``converged=False`` on every run, good or bad (arviz 1.x). Nothing here needs a GPU:
the same chains strand at the same log-likelihood on both devices.

* T1-T3 pin the posterior against closed forms and grids, T4-T6 the contract guards, T7 the
  float32 hazard (in a subprocess, since x64 is process-global), and the mechanism-2 test a time
  prior 50x wider than the data.
* The regression cases are the matrix's own mocks (``tests/data/u3_bump_mocks.json``: data, fit
  seed and a 160^3 grid posterior per simulation; ``u3_kn1_mocks.json`` for the slow kilonova),
  fitted with the matrix's seeds. The failing ones run with the OLD start -- NumPyro's
  ``init_to_uniform``, now ``init_strategy="uniform"`` (PyMC's ``"jitter"``, emcee's ``"box"``) --
  to show the failure is flagged, and with the default ``"prior_scan"`` start to show it is gone.

T2 and T3 pass on the unfixed code as well: a read-only check found the simple
known-answer cases right, and these pin that they stay right. Every other test here fails there
(``info`` had no ``convergence_problems``, ``rhat`` was ``{}``, the default start stranded these
chains, a log-posterior was accepted silently, point starts and float32 clocks did not warn, pymc
left x64 switched on, the flare density rounded MJD epochs in an x64 session).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import jax.numpy as jnp  # noqa: E402

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.samplers.jax import _diagnostics as dg  # noqa: E402
from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax  # noqa: E402
from whisper_cbpf.samplers.jax.nuts_gpu import NUTSGPUSampler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MOCKS = json.load(open(os.path.join(HERE, "data", "u3_bump_mocks.json")))


def _quiet_fit(*args, **kwargs):
    """``wp.fit`` with warnings collected, not printed. Returns ``(result, [messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = wp.fit(*args, **kwargs)
    return r, [str(w.message) for w in caught]


# ------------------------------------------------------------------------------ the bump mocks
def _bump_predict(parameters, times, bands=None):
    t = np.asarray(times, dtype=float)
    return float(parameters["A"]) * np.exp(-0.5 * ((t - float(parameters["t0"]))
                                                  / float(parameters["w"])) ** 2)


def _bump_predict_jax(theta, times, band_idx=None):
    return theta[0] * jnp.exp(-0.5 * ((jnp.asarray(times) - theta[1]) / theta[2]) ** 2)


def _bump_prior(shift=0.0):
    return wp.Prior({"A": wp.LogUniform(0.3, 10.0), "t0": wp.Uniform(shift, 30.0 + shift),
                     "w": wp.Uniform(1.0, 8.0)})


@pytest.fixture(scope="module")
def bump():
    """The Step 0 bump: the default prior of the matrix, registered through the public API."""
    return wp.register_model("u3_test_bump", _bump_predict, ["A", "t0", "w"], prior=_bump_prior(),
                             predict_jax=_bump_predict_jax, overwrite=True)


@pytest.fixture
def float64():
    """The Step 0 regressions ran in float64; the same seed strands the same chain only there.

    x64 is process-global, and other tests (``tests/t2_autodiff``) leave it on, so each of these
    tests states it and puts back what it found -- the same switch ``pymc_gpu`` now makes.
    """
    before = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


def _sim(i, shift=0.0):
    s = MOCKS["sims"][str(i)]
    lc = wp.LightCurve(time=np.asarray(s["t_rel"]) + shift, band=np.asarray(s["bands"]),
                       flux=np.asarray(s["y"]), flux_err=np.asarray(s["sig"]), name=f"u3_{i}")
    return lc, s


def _assert_matches_grid(r, s, tol_sd=0.25):
    """Pooled mean within ``tol_sd`` grid-sd of the 160^3 grid posterior, sd within 25 %."""
    for nm in ("A", "t0", "w"):
        ref = s["ref"][nm]
        x = r.samples[nm].to_numpy()
        assert abs(x.mean() - ref["mean"]) < tol_sd * ref["sd"], (nm, x.mean(), ref)
        assert abs(x.std() / ref["sd"] - 1.0) < 0.25, (nm, x.std(), ref)


# ----------------------------------------------------------------------- T1: linear, closed form
def test_t1_linear_gaussian_matches_the_closed_form_with_the_prior_in_reversed_order():
    """A straight line in flux with Gaussian errors: the posterior is exactly Gaussian.

    The prior dict lists ``b`` before ``a`` while ``model.parameters`` is ``["a", "b"]``, which pins
    the parameter order through the prior, the scan, the starts and the samples. The finite R-hat
    for every parameter is what failed at HEAD (``rhat == {}`` on arviz 1.x).
    """
    t = np.linspace(0.0, 10.0, 30)
    sig = 0.2
    y = 1.0 + 0.5 * t + np.random.default_rng(1).normal(0.0, sig, t.size)
    m = wp.register_model(
        "u3_line", lambda p, tt, b=None: float(p["a"]) + float(p["b"]) * np.asarray(tt, float),
        ["a", "b"], prior=wp.Prior({"b": wp.Uniform(-2.0, 2.0), "a": wp.Uniform(-5.0, 5.0)}),
        predict_jax=lambda th, tt, bi=None: th[0] + th[1] * jnp.asarray(tt), overwrite=True)
    lc = wp.LightCurve(time=t, band=["r"] * t.size, flux=y, flux_err=np.full(t.size, sig))
    r, _ = _quiet_fit(lc, m.name, sampler="nuts_gpu", space="flux", num_warmup=500,
                      num_samples=1500, seed=3)

    X = np.stack([np.ones_like(t), t], axis=1)
    cov = np.linalg.inv(X.T @ X / sig ** 2)
    mean = cov @ X.T @ y / sig ** 2
    sd = np.sqrt(np.diag(cov))
    got = r.samples[["a", "b"]].to_numpy()
    assert np.all(np.abs(got.mean(0) - mean) < 0.15 * sd), (got.mean(0), mean, sd)
    assert np.all(np.abs(got.std(0) / sd - 1.0) < 0.06), (got.std(0), sd)
    rho = cov[0, 1] / (sd[0] * sd[1])
    assert abs(np.corrcoef(got.T)[0, 1] - rho) < 0.03
    assert set(r.info["rhat"]) == {"a", "b"}
    assert all(np.isfinite(v) for v in r.info["rhat"].values())
    assert isinstance(r.info["max_rhat"], float) and r.info["max_rhat"] < 1.01
    assert r.info["rhat_method"].startswith("arviz") or "numpyro" in r.info["rhat_method"]


# -------------------------------------------------------- T2 / T4: LogUniform and the prior count
def _flat_model():
    return wp.register_model(
        "u3_flat", lambda p, tt, b=None: np.zeros(len(np.asarray(tt))), ["x"],
        prior=wp.Prior({"x": wp.LogUniform(1.0, 1.0e4)}),
        predict_jax=lambda th, tt, bi=None: jnp.zeros_like(jnp.asarray(tt)) * th[0],
        overwrite=True)


def _flat_lc():
    return wp.LightCurve(time=np.linspace(0, 1, 5), band=["r"] * 5, flux=np.zeros(5),
                         flux_err=np.ones(5))


def test_t2_loguniform_prior_under_a_flat_likelihood_is_returned_as_the_prior():
    """With nothing to learn the posterior IS the LogUniform(1, 1e4) prior: log10 x ~ U(0, 4).

    The mutant that counts the prior twice (the density adds -log x) lands > 1 dex off -- which is
    what passing emcee_jax's log-posterior used to do silently, and what T4 now refuses.
    """
    m, lc = _flat_model(), _flat_lc()
    r, _ = _quiet_fit(lc, m.name, sampler="nuts_gpu", space="flux",
                      log_prob_fn=lambda th: 0.0 * th[0], num_warmup=300, num_samples=2000, seed=0)
    q = np.quantile(np.log10(r.samples["x"]), [0.16, 0.5, 0.84])
    assert np.all(np.abs(q - np.array([0.64, 2.0, 3.36])) < 0.15), q

    twice, _ = _quiet_fit(lc, m.name, sampler="nuts_gpu", space="flux",
                          log_prob_fn=lambda th: -jnp.log(th[0]), num_warmup=300,
                          num_samples=2000, seed=0)
    assert abs(np.median(np.log10(twice.samples["x"])) - 2.0) > 1.0


def test_t4_a_log_posterior_is_refused_for_a_non_uniform_prior_and_warned_for_uniform():
    m, lc = _flat_model(), _flat_lc()
    posterior = make_log_prob_jax(lc, m, m.default_prior, space="flux", include_prior=True)
    with pytest.raises(ValueError, match="log-POSTERIOR"):
        NUTSGPUSampler().fit(lc, m, log_prob_fn=posterior, space="flux", num_warmup=5,
                             num_samples=5, num_chains=1)
    uni = wp.Prior({"x": wp.Uniform(1.0, 10.0)})
    posterior = make_log_prob_jax(lc, m, uni, space="flux", include_prior=True)
    with pytest.warns(UserWarning, match="log-POSTERIOR"):
        NUTSGPUSampler().fit(lc, m, prior=uni, log_prob_fn=posterior, space="flux",
                             num_warmup=20, num_samples=20, num_chains=1)


# --------------------------------------------------------------- T3: LogUniform against a grid
def test_t3_loguniform_parameter_matches_a_1500x1500_grid_posterior():
    """``A exp(-t / tau)`` with ``tau ~ LogUniform(1, 100)``: NUTS against the exact grid."""
    t = np.linspace(0.5, 20.0, 25)
    sig = 0.1
    y = 2.0 * np.exp(-t / 6.0) + np.random.default_rng(4).normal(0.0, sig, t.size)
    m = wp.register_model(
        "u3_decay",
        lambda p, tt, b=None: float(p["A"]) * np.exp(-np.asarray(tt, float) / float(p["tau"])),
        ["A", "tau"], prior=wp.Prior({"A": wp.Uniform(0.5, 5.0), "tau": wp.LogUniform(1.0, 100.0)}),
        predict_jax=lambda th, tt, bi=None: th[0] * jnp.exp(-jnp.asarray(tt) / th[1]),
        overwrite=True)
    lc = wp.LightCurve(time=t, band=["r"] * t.size, flux=y, flux_err=np.full(t.size, sig))
    r, _ = _quiet_fit(lc, m.name, sampler="nuts_gpu", space="flux", num_warmup=500,
                      num_samples=1500, seed=5)

    ga = np.linspace(0.5, 5.0, 1500)
    gu = np.linspace(0.0, 2.0, 1500)                       # log10 tau: the prior is flat here
    ll = np.empty((1500, 1500))
    for i in range(0, 1500, 100):                          # chunked: 2.25e6 x 25 residuals
        mdl = ga[i:i + 100, None, None] * np.exp(-t[None, None, :] / 10.0 ** gu[None, :, None])
        ll[i:i + 100] = -0.5 * np.sum(((y - mdl) / sig) ** 2, axis=-1)
    p = np.exp(ll - ll.max())
    p /= p.sum()
    for vals, marg, got in ((ga, p.sum(1), r.samples["A"].to_numpy()),
                            (gu, p.sum(0), np.log10(r.samples["tau"].to_numpy()))):
        mean = np.sum(marg * vals)
        sd = np.sqrt(np.sum(marg * (vals - mean) ** 2))
        assert abs(got.mean() - mean) < 0.15 * sd, (got.mean(), mean, sd)
        assert abs(got.std() / sd - 1.0) < 0.1, (got.std(), sd)


# ----------------------------------------------------------- T5: a shifted chain, any arviz
def _shifted_chains():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 500, 2))
    ll = -0.5 * (x ** 2).sum(-1)
    x[2] += 5.0                                            # chain 2 elsewhere ...
    ll[2] -= 50.0                                          # ... and 50 nats lower
    return x, ll


def _health(x, ll):
    return dg.chain_health(x, ["p", "q"], ll, n_divergences=0, step_size_by_chain=[0.5] * 4)


@pytest.mark.parametrize("hide_arviz", [False, True])
def test_t5_a_shifted_chain_is_flagged_by_name_on_any_arviz(monkeypatch, hide_arviz):
    if hide_arviz:
        monkeypatch.setitem(sys.modules, "arviz", None)     # import arviz -> ImportError
    x, ll = _shifted_chains()
    h = _health(x, ll)
    assert h["converged"] is False and h["stranded_chains"] == [2]
    assert h["max_rhat"] > 1.1 and h["rhat_log_likelihood"] > 1.1
    assert any("chain(s) 2 stranded" in p for p in h["convergence_problems"])
    if hide_arviz:
        assert "numpyro" in h["rhat_method"] and "arviz" in h["rhat_error"]
    else:
        assert h["rhat_method"].startswith("arviz") and h["rhat_error"] is None
    y = np.random.default_rng(1).normal(size=(4, 500, 2))
    healthy = _health(y, -0.5 * (y ** 2).sum(-1))
    assert healthy["converged"] is True, healthy["convergence_problems"]


def test_t5_all_chains_below_an_independent_optimum_is_flagged():
    """Four chains that agree with each other, all 40 nats below a better point: R-hat ~ 1."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 500, 2))
    ll = -0.5 * (x ** 2).sum(-1)
    h = dg.chain_health(x, ["p", "q"], ll, n_divergences=0, reference_ll=40.0)
    assert h["max_rhat"] < 1.01 and h["converged"] is False
    assert any("all chains missed" in p for p in h["convergence_problems"])


def test_frozen_chain_needs_a_small_step_and_no_movement():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 500, 2))
    ll = -0.5 * (x ** 2).sum(-1)
    moving = dg.chain_health(x, ["p", "q"], ll, n_divergences=0,
                             step_size_by_chain=[0.5, 0.5, 2e-4, 0.5])
    assert moving["frozen_chains"] == []                   # tiny step, but it still mixes
    x[1] = x[1] * 1e-4 + 0.3
    frozen = dg.chain_health(x, ["p", "q"], ll, n_divergences=0,
                             step_size_by_chain=[0.5, 1e-5, 0.5, 0.5])
    assert frozen["frozen_chains"] == [1] and frozen["converged"] is False


# --------------------------------------------------------------------------- T6: point starts
def test_t6_point_starts_warn_with_more_than_one_chain(bump):
    lc, s = _sim(0)
    point = {"log10_A": 0.9, "t0": 15.4, "w": 2.9}          # keyed by sample SITE
    for init in ("median", numpyro.infer.init_to_value(values=point)):
        with pytest.warns(UserWarning, match="same point"):
            NUTSGPUSampler().fit(lc, bump, space="flux", init_strategy=init, num_chains=2,
                                 num_warmup=20, num_samples=20)


def test_per_chain_starts_are_used_and_checked(bump):
    lc, s = _sim(0)
    starts = np.array([[8.0, 15.4, 2.9], [8.3, 15.5, 2.8]])
    r, _ = _quiet_fit(lc, bump.name, sampler="nuts_gpu", space="flux", init_strategy=starts,
                      num_chains=2, num_warmup=100, num_samples=100)
    assert r.info["init_strategy"] == "per_chain"
    with pytest.raises(ValueError, match="strictly inside"):
        NUTSGPUSampler().fit(lc, bump, space="flux", num_chains=2, num_warmup=5, num_samples=5,
                             init_strategy=np.array([[8.0, 15.4, 2.9], [8.0, 45.0, 2.9]]))


@pytest.mark.parametrize("sampler", ["nuts_gpu", "pymc_jax_gpu_vectorized"])
def test_prior_and_per_chain_starts_are_never_where_the_density_is_minus_inf(bump, sampler):
    """``init_strategy="prior"`` took the scan's first ``num_chains`` draws whatever their score.
    Behind a wall -- here w >= 2.5, -inf over 79 % of the box, the shape the JAX supernova and TDE
    constraint walls give -- that started 2 of 4 chains at -inf, and at the default budget one PyMC
    chain never left (median log-likelihood -inf, step size 2e-308, its draws pooled into the
    posterior). It now takes the first draws where the density is finite, and a caller's start at
    -inf is refused, as emcee_jax refuses one."""
    if sampler.startswith("pymc"):
        pytest.importorskip("pymc")
    lc, s = _sim(0)
    ll = make_log_prob_jax(lc, bump, bump.default_prior, space="flux")
    wall = lambda th: jnp.where(th[2] < 2.5, ll(th), -jnp.inf)      # noqa: E731
    first = dg.prior_draws(bump.default_prior, ["A", "t0", "w"], 4, 0)
    assert (first[:, 2] >= 2.5).any()                   # the old rule started a chain at -inf
    r, _ = _quiet_fit(lc, bump.name, sampler=sampler, space="flux", log_prob_fn=wall, seed=0,
                      init_strategy="prior", num_chains=4, num_warmup=50, num_samples=50)
    start = r.info["prior_scan"]["start_log_likelihood"]
    assert r.info["init_strategy"] == "prior" and np.all(np.isfinite(start)), start
    with pytest.raises(ValueError, match="log-density is -inf"):
        _quiet_fit(lc, bump.name, sampler=sampler, space="flux", log_prob_fn=wall,
                   init_strategy=np.array([[8.0, 15.4, 2.0], [8.0, 15.4, 3.0]]), num_chains=2,
                   num_warmup=5, num_samples=5)


# ------------------------------------------------ the Step 0 regressions (float64, relative clock)
@pytest.mark.parametrize("i", [8, 13])
def test_ta1_stranded_chain_is_flagged_and_the_default_start_avoids_it(bump, float64, i):
    """T-A1: bump sims 8 and 13, whose old start stranded 2 and 1 of 4 chains at the zero-flux
    model's likelihood (sim 13: chain 2 at -1121.6 against 40.8)."""
    lc, s = _sim(i)
    old, msgs = _quiet_fit(lc, bump.name, sampler="nuts_gpu", space="flux", seed=s["fit_seed"],
                           init_strategy="uniform")
    assert old.info["converged"] is False and old.info["stranded_chains"]
    assert any("stranded" in p for p in old.info["convergence_problems"])
    assert any("not converged" in m and "stranded" in m for m in msgs)

    new, _ = _quiet_fit(lc, bump.name, sampler="nuts_gpu", space="flux", seed=s["fit_seed"])
    assert new.info["init_strategy"] == "prior_scan" and new.info["stranded_chains"] == []
    _assert_matches_grid(new, s)


def _assert_all_chains_missed_is_flagged_when_it_happens(r):
    """The invariant behind T-A2: whenever every chain's best draw is more than 10 nats below the
    run's independent optimum, the run says so by name and does not read converged."""
    ref = r.info["reference_log_likelihood"]
    missed = r.max_log_likelihood < ref - dg.STRANDED_NATS
    if missed:
        assert r.info["converged"] is False
        assert any("all chains missed" in p for p in r.info["convergence_problems"])
    return missed


def test_ta2_all_chains_in_the_wrong_mode_is_flagged(bump, float64):
    """T-A2: nuts sim 88 -- every chain of the old start missed the mode (37 nats below the
    truth); only the independent optimum sees it when the chains agree with each other.

    Where the old start's chains end up is fixed by the seed on the CPU but not between GPU
    processes (XLA's GPU reductions are not bitwise reproducible), so on a GPU the old run is held
    to the invariant only: if it missed, it must say so. The CPU run must also miss, as measured."""
    lc, s = _sim(88)
    old, _ = _quiet_fit(lc, bump.name, sampler="nuts_gpu", space="flux", seed=s["fit_seed"],
                        init_strategy="uniform")
    missed = _assert_all_chains_missed_is_flagged_when_it_happens(old)
    if jax.default_backend() == "cpu":
        assert missed and old.max_log_likelihood < s["ref_max_log_likelihood"] - 10.0
    new, _ = _quiet_fit(lc, bump.name, sampler="nuts_gpu", space="flux", seed=s["fit_seed"])
    assert new.max_log_likelihood > s["ref_max_log_likelihood"] - 3.0


def test_mechanism2_a_time_prior_much_wider_than_the_data_warns_and_the_start_finds_the_event(
        float64):
    """Plan mechanism 2: t0 ~ U(-1000, 1000) for a 3 d bump seen over [-20, 20] d. The old start
    stranded every chain on the zero-flux plateau (4 / 4 seeds). So did the first "prior_scan":
    its free Adam climb stepped ~0.1 logit units = 50 d whatever the gradient and threw 17 of the
    32 best draws off the bump, so every climbed start sat on the plateau (best draw -4097, every
    start -7537.5). The climb is monotone now, and 97 % of the prior being a plateau warns."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(-20.0, 20.0, 30))
    y = 5.0 * np.exp(-0.5 * (t / 3.0) ** 2) + rng.normal(0.0, 0.1, t.size)
    prior = wp.Prior({"A": wp.LogUniform(0.3, 10.0), "t0": wp.Uniform(-1000.0, 1000.0),
                      "w": wp.Uniform(1.0, 8.0)})
    m = wp.register_model("u3_wide_t0", _bump_predict, ["A", "t0", "w"], prior=prior,
                          predict_jax=_bump_predict_jax, overwrite=True)
    lc = wp.LightCurve(time=t, band=["r"] * t.size, flux=y, flux_err=np.full(t.size, 0.1))

    ll = make_log_prob_jax(lc, m, prior, space="flux")
    draws, scores = dg.prior_scan(ll, prior, ["A", "t0", "w"], 0, jnp.float64)
    best = np.argsort(-scores)[:32]
    _, climbed = dg.climb(ll, draws[best], prior, ["A", "t0", "w"], jnp.float64)
    assert np.all(climbed >= scores[best] - 1e-9)           # never worse than its own draw

    with pytest.warns(UserWarning, match="much wider than the data window"):
        r = NUTSGPUSampler().fit(lc, m, space="flux", num_warmup=500, num_samples=1000, seed=0)
    assert r.info["prior_scan"]["plateau_fraction"] > 0.9
    assert r.info["stranded_chains"] == [], r.info["convergence_problems"]
    assert abs(r.samples["t0"].mean()) < 0.3 and r.samples["t0"].std() < 0.2


def test_t3_a_clean_run_reports_finite_rhat_and_reads_converged(bump, float64):
    """T-3: bump sim 0, clean in every Step 0 cell -- and reported converged=False at HEAD."""
    lc, s = _sim(0)
    r, msgs = _quiet_fit(lc, bump.name, sampler="nuts_gpu", space="flux", seed=s["fit_seed"])
    assert not any("much wider than the data window" in m for m in msgs)
    assert set(r.info["rhat"]) == {"A", "t0", "w"}
    assert all(np.isfinite(v) and v < 1.01 for v in r.info["rhat"].values())
    assert r.info["min_ess"] >= 400 and r.info["rhat_log_likelihood"] < 1.01
    assert r.info["converged"] is True, r.info["convergence_problems"]
    assert r.info["convergence_problems"] == [] and r.info["rhat_error"] is None
    _assert_matches_grid(r, s)
    json.loads(r.to_json())                                # every new info entry serialises


# ------------------------------------------------------------------------ emcee_jax (T-4)
def test_t4_emcee_stuck_walkers_are_flagged_and_the_default_start_avoids_them(bump, float64):
    """T-4: bump sim 24. The old linear-box start left 6 of 32 walkers 26 603 nats below the rest
    (walker R-hat 67) and emcee_jax still said converged=True -- the autocorrelation rule alone."""
    lc, s = _sim(24)
    old, _ = _quiet_fit(lc, bump.name, sampler="emcee_jax", space="flux", seed=s["fit_seed"],
                        init="box")
    assert old.info["converged"] is False and old.info["stuck_walkers"]
    assert any("stuck" in p for p in old.info["convergence_problems"])
    new, _ = _quiet_fit(lc, bump.name, sampler="emcee_jax", space="flux", seed=s["fit_seed"])
    assert new.info["init"] == "prior_scan" and new.info["stuck_walkers"] == []
    assert new.info["converged"] is True, new.info["convergence_problems"]
    _assert_matches_grid(new, s)


# ------------------------------------------------------------------------------ pymc (T-A2, x64)
def test_ta2_pymc_all_chains_wrong_is_flagged_and_the_default_start_avoids_it(bump, float64):
    """pymc bump sim 66: PyMC's own jittered start put all 4 chains in the wrong mode with R-hat
    1.002 (t0 35 grid-sd off). As for T-A2, a GPU run of the old start is held to the invariant
    (flagged whenever it missed), and the deterministic CPU run must also miss."""
    pytest.importorskip("pymc")
    lc, s = _sim(66)
    old, _ = _quiet_fit(lc, bump.name, sampler="pymc_jax_gpu_vectorized", space="flux",
                        seed=s["fit_seed"], init_strategy="jitter")
    missed = _assert_all_chains_missed_is_flagged_when_it_happens(old)
    if jax.default_backend() == "cpu":
        assert missed
    new, _ = _quiet_fit(lc, bump.name, sampler="pymc_jax_gpu_vectorized", space="flux",
                        seed=s["fit_seed"])
    assert new.info["init_strategy"] == "prior_scan"
    _assert_matches_grid(new, s)
    # AIC/BIC from the shared aic_bic, as floats: an inline copy returned np.float64 and read
    # log(max(n, 1)) where every other sampler read log(n) (tests/test_empty_light_curve.py).
    from whisper_cbpf.samplers.base import aic_bic
    assert (new.aic, new.bic) == aic_bic(new.max_log_likelihood, new.n_params, new.n_data)
    assert type(new.aic) is float and type(new.bic) is float


def _run_isolated(code):
    """Run ``code`` in a fresh interpreter with x64 OFF (JAX's default); return its stdout JSON."""
    env = dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, cwd=HERE,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


_SUBPROCESS_HEAD = """
    import json, warnings, sys
    import numpy as np
    import jax, jax.numpy as jnp
    import whisper_cbpf as wp
    sys.path.insert(0, ".")
    from test_nuts_gpu_known_answer import _bump_predict, _bump_predict_jax, _bump_prior, MOCKS
"""


def test_tx64_pymc_leaves_the_sessions_precision_alone():
    """T-x64: importing PyTensor's JAX linker switched jax_enable_x64 on for the whole process, so
    every later nuts_gpu / emcee_jax call silently ran in float64."""
    pytest.importorskip("pymc")
    res = _run_isolated(_SUBPROCESS_HEAD + """
    before = bool(jax.config.jax_enable_x64)
    m = wp.register_model("u3_x64", _bump_predict, ["A", "t0", "w"], prior=_bump_prior(),
                          predict_jax=_bump_predict_jax, overwrite=True)
    s = MOCKS["sims"]["0"]
    lc = wp.LightCurve(time=np.asarray(s["t_rel"]), band=np.asarray(s["bands"]),
                       flux=np.asarray(s["y"]), flux_err=np.asarray(s["sig"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wp.fit(lc, m.name, sampler="pymc_jax_gpu_vectorized", space="flux",
                   num_warmup=50, num_samples=50, num_chains=2)
    print(json.dumps({"before": before, "after": bool(jax.config.jax_enable_x64),
                      "ran": r.info["x64"], "session": r.info.get("x64_session")}))
    """)
    assert res == {"before": False, "after": False, "ran": True, "session": False}


# ------------------------------------------------------------ T7 / T-1: float32 with an MJD clock
def test_float32_hazard_is_keyed_on_absolute_size_not_on_the_bound_ratio():
    """U(50000, 70000) has a small bound/width ratio and a 0.0078 d float32 quantum."""
    names = ["A", "t0"]
    mjd = wp.Prior({"A": wp.LogUniform(0.3, 10.0), "t0": wp.Uniform(50000.0, 70000.0)})
    with pytest.warns(UserWarning, match="float32"):
        msg = dg.float32_hazard(np.linspace(0, 40, 5), mjd, names, "nuts_gpu", x64=False)
    assert "'t0'" in msg and "set_explosion_date" in msg
    with pytest.warns(UserWarning, match="lc.time"):
        dg.float32_hazard(60000.0 + np.linspace(0, 40, 5), _bump_prior(), ["A", "t0", "w"],
                          "nuts_gpu", x64=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # none of these may warn
        assert dg.float32_hazard(60000.0 + np.arange(3), mjd, names, "nuts_gpu", x64=True) is None
        assert dg.float32_hazard(np.arange(3.0), _bump_prior(), ["A", "t0", "w"], "x",
                                 x64=False) is None
        big_scale = wp.Prior({"T": wp.LogUniform(100.0, 6000.0)})    # a scale parameter: fine
        assert dg.float32_hazard(np.arange(3.0), big_scale, ["T"], "x", x64=False) is None


def test_flare_density_follows_the_session_precision(float64):
    """``flare.make_log_prob_jax`` hard-coded float32 for times, values and bounds, so an x64
    session still rounded MJD epochs to 0.0039 d: here that moves log L by tens of nats."""
    from whisper_cbpf.models.jax import flare
    from whisper_cbpf.models.jax._flare_spec import PARAMETERS, flare_flux_numpy

    t = 60000.0 + np.array([0.1234567, 0.5432101, 1.3579246, 2.4680135, 3.1415926])
    theta = np.array([0.0, np.log(0.5), np.log(2.0), 60000.8])
    model = flare_flux_numpy(dict(zip(PARAMETERS, theta)), t)
    y, sig = model + 0.01, np.full(t.size, 1e-3)
    ref = -0.5 * np.sum(((y - model) / sig) ** 2 + np.log(2 * np.pi * sig ** 2))
    f = flare.make_log_prob_jax(t, y, sig, [(-5, 5), (-5, 5), (-5, 5), (59990.0, 60010.0)])
    assert float(f(jnp.asarray(theta))) == pytest.approx(ref, rel=1e-9)


def test_t7_float32_mjd_clock_warns_and_a_frozen_run_reads_not_converged():
    """T-1 (plan T7): bump sim 73 on an MJD clock in float32. t0 can only move in 0.0039 d steps,
    and under the old start a chain froze (split R-hat 72, 0 divergences) while whisper said
    nothing. Now the fit warns before sampling, and the frozen run names the problem. ~70 s on one
    CPU core (a frozen chain runs to maximum tree depth)."""
    res = _run_isolated(_SUBPROCESS_HEAD + """
    s = MOCKS["sims"]["73"]
    shift = s["t_first"]
    m = wp.register_model("u3_mjd", _bump_predict, ["A", "t0", "w"], prior=_bump_prior(shift),
                          predict_jax=_bump_predict_jax, overwrite=True)
    lc = wp.LightCurve(time=np.asarray(s["t_rel"]) + shift, band=np.asarray(s["bands"]),
                       flux=np.asarray(s["y"]), flux_err=np.asarray(s["sig"]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = wp.fit(lc, m.name, sampler="nuts_gpu", space="flux", prior=_bump_prior(shift),
                   seed=s["fit_seed"], init_strategy="uniform")
    print(json.dumps({"x64": bool(jax.config.jax_enable_x64),
                      "hazard_warned": any("float32" in str(x.message) for x in w),
                      "hazard": r.info.get("float32_hazard"), "converged": r.info["converged"],
                      "problems": r.info.get("convergence_problems"),
                      "frozen": r.info.get("frozen_chains")}))
    """)
    assert res["x64"] is False and res["hazard_warned"] and "lc.time" in (res["hazard"] or "")
    assert res["converged"] is False and res["problems"]
    assert res["frozen"] or any("R-hat" in p for p in res["problems"])


# ------------------------------------------------------ T-B: a kilonova's prior-box corner (slow)
@pytest.mark.slow
@pytest.mark.parametrize("i", [0, 9])
def test_tb_kilonova_chains_no_longer_park_on_a_prior_box_corner(float64, i):
    """T-B (NEW-B): one-component kilonova sims 0 and 9, where the old start parked chains on the
    corner mej = 0.05, vej = 0.10 at log L -5 178 and -13 656 (7 of 13 such runs at HEAD). 2-8 min
    on one CPU core at the matrix's 500/500 budget."""
    kn = pytest.importorskip("whisper_cbpf.models.jax.kilonova")
    mocks = json.load(open(os.path.join(HERE, "data", "u3_kn1_mocks.json")))
    f, s = mocks["factory"], mocks["sims"][str(i)]
    m = wp.register_kilonova(f["bands"], redshift=f["redshift"], dl_cm=f["dl_cm"],
                             n_wave=f["n_wave"], name="u3_test_kn1",
                             filter_set=kn.make_filter_set(f["bands"], n_wave=f["n_wave"]))
    lc = wp.LightCurve(time=np.asarray(s["t_rel"]), band=np.asarray(s["bands"]),
                       magnitude=np.asarray(s["y"]), magnitude_err=np.asarray(s["sig"]))
    r, _ = _quiet_fit(lc, m.name, sampler="nuts_gpu", space="magnitude", seed=s["fit_seed"],
                      **mocks["budget"])
    ll_truth = float(make_log_prob_jax(lc, m, m.default_prior, space="magnitude")(
        jnp.asarray([s["truth"][k] for k in m.parameters])))
    assert r.info["stranded_chains"] == [], r.info["convergence_problems"]
    assert r.max_log_likelihood > ll_truth - 10.0
