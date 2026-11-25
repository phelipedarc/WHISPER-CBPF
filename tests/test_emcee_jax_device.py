"""``emcee_jax`` runs entirely on the device: K ensembles in one compiled loop.

The claims, each a test:

1. **A known Gaussian is recovered.** On a model linear in its parameters, whose posterior is an
   exactly known correlated Gaussian, the default fit (4 ensembles of 32 walkers, pilot and
   restart) recovers the mean and the covariance, and the R-hat between the ensembles is below
   1.01 on every parameter.
2. **A fixed seed gives identical draws**; another seed gives other draws.
3. **The ensembles are independent.** Each has its own seed (two fits' seeds share none), its own
   start and its own moves: without a pilot, ensemble e of a fit equals the same light curve fitted
   alone by ``fit_batch`` with ensemble e's seed, draw for draw (the same engine).
4. **One compiled loop, no host round trip.** The whole chain, pilot and restart included, is ONE
   call of one compiled program: nothing compiles while it runs, its jaxpr has a single loop at the
   top level and no host callback.
5. **The result keeps the SamplerResult schema** with the ensembles laid out: draws,
   ``samples_by_chain`` (ensemble-major walkers), the per-ensemble info, ``diagnostics()``, save
   and load.
6. **The pilot drops a stranded ensemble.** An ensemble started in a mode 20 nats below the other
   three is restarted from the best pilot ensemble's states (``info["pilot"]`` names it); without
   the pilot the ensembles disagree, and the fit says so in ``convergence_problems``.
7. **The helpers and the errors**: :func:`ensemble_seeds`, :func:`pilot_plan`, ``walker_chunk``,
   ``init`` per ensemble, and errors that name the cause.
8. **The ensemble ESS is not inflated by the walker count.** ``info["ensemble_ess_bulk"]`` and
   ``["ensemble_ess_tail"]`` count every walker as a chain: on walkers that are AR(1) chains with a
   known autocorrelation they match the ESS those chains have (about ``draws x (1 - rho) / (1 +
   rho)``; the tail ESS far below the number of draws), while the R-hat between ensembles is
   unchanged.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
import whisper_cbpf.models as M  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax import batch as B  # noqa: E402
from whisper_cbpf.samplers.jax.emcee_jax import (  # noqa: E402
    ensemble_seeds,
    fit_emcee_jax,
    pilot_plan,
)

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

    return M.Model(name="quadratic_device_toy", predict=predict, parameters=list(NAMES),
                   default_prior=Prior({n: Uniform(-box, box) for n in NAMES}),
                   predict_jax=predict_jax)


def _alert(coef=(40.0, -5.0, 2.0), n=24, sigma=0.3, seed=1):
    t = np.linspace(0.0, 3.0, n)
    rng = np.random.default_rng(seed)
    flux = coef[0] + coef[1] * t + coef[2] * t * t + rng.normal(0.0, sigma, n)
    lc = wp.LightCurve(time=t, band=["r"] * n, flux=flux, flux_err=np.full(n, sigma))
    x = np.stack([np.ones(n), t, t * t], axis=1) / sigma
    cov = np.linalg.inv(x.T @ x)
    return lc, cov @ (x.T @ (flux / sigma)), cov


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _fit(lc, model, **kw):
    kw = {"nsteps": 600, "burnin": 200, "thin": 2, "seed": 0, "space": "flux", **kw}
    return _quiet(fit_emcee_jax, lc, model, **kw)


# --- 1. a known Gaussian --------------------------------------------------------------------------

def test_the_default_fit_recovers_a_known_correlated_gaussian_with_rhat_below_1_01():
    model = _quadratic()
    lc, mean, cov = _alert()
    r = _fit(lc, model, nsteps=4000, burnin=1000, thin=5)
    assert r.info["n_ensembles"] == 4 and r.info["walkers_per_ensemble"] == 32
    assert r.info["pilot"]["steps"] == 800                     # nsteps // 5, the default
    x = r.samples[NAMES].to_numpy()
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    assert abs(corr[0, 1]) > 0.5                               # genuinely correlated
    assert np.all(np.abs(x.mean(0) - mean) < 0.1 * sd), (x.mean(0), mean, sd)
    np.testing.assert_allclose(x.var(0), sd ** 2, rtol=0.1)
    np.testing.assert_allclose(np.corrcoef(x.T), corr, atol=0.05)
    assert max(r.info["ensemble_rhat"].values()) < 1.01, r.info["ensemble_rhat"]
    assert r.info["converged"] is True, r.info["convergence_problems"]
    assert 0.2 < r.info["mean_acceptance_fraction"] < 0.7


# --- 2. a fixed seed ------------------------------------------------------------------------------

def test_a_fixed_seed_gives_identical_draws_and_another_seed_other_draws():
    model = _quadratic()
    lc = _alert()[0]
    a, b, c = (_fit(lc, model, seed=s) for s in (3, 3, 4))
    np.testing.assert_array_equal(a.samples.to_numpy(), b.samples.to_numpy())
    np.testing.assert_array_equal(a.samples_by_chain, b.samples_by_chain)
    assert a.max_log_likelihood == b.max_log_likelihood
    assert a.info["ensemble_seeds"] == b.info["ensemble_seeds"] != c.info["ensemble_seeds"]
    assert not np.array_equal(a.samples.to_numpy(), c.samples.to_numpy())


# --- 3. independent ensembles ---------------------------------------------------------------------

#: XLA flags that make GPU kernels run-to-run deterministic (as in test_fit_batch claim 2): the
#: four-ensemble program and the one-curve program may otherwise round differently.
_DETERMINISTIC_XLA = ("--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0")
_IN_DETERMINISTIC_CHILD = "WHISPER_TEST_DETERMINISTIC_CHILD"


def test_each_ensemble_equals_the_light_curve_fitted_alone_with_its_own_seed(request):
    if jax.default_backend() == "gpu" and not os.environ.get(_IN_DETERMINISTIC_CHILD):
        flags = os.environ.get("XLA_FLAGS", "").split()
        flags += [f for f in _DETERMINISTIC_XLA if f not in flags]
        env = {**os.environ, "XLA_FLAGS": " ".join(flags), "JAX_ENABLE_X64": "1",
               _IN_DETERMINISTIC_CHILD: "1"}
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                              f"{request.node.path}::{request.node.name}"],
                             cwd=request.config.rootpath, env=env, capture_output=True, text=True,
                             timeout=1800)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
        assert "1 passed" in out.stdout, out.stdout[-3000:]
        return
    model = _quadratic()
    lc = _alert()[0]
    w = 16
    r = _fit(lc, model, nwalkers=w, pilot=False, seed=7)
    seeds = r.info["ensemble_seeds"]
    assert len(set(seeds)) == 4 and r.info["pilot"] is None
    by = r.samples_by_chain                                    # (4 * w, draws, 3)
    for e in (0, 2):
        alone = _quiet(B.fit_batch, [lc], model, nwalkers=w, nsteps=600, burnin=200, thin=2,
                       seed=seeds[e], space="flux", metrics=False)[0]
        np.testing.assert_array_equal(by[e * w:(e + 1) * w], alone.samples_by_chain)
    # the starts differ between ensembles: each searched the prior with its own seed
    starts = [r.info["init_detail"]["per_ensemble"][e]["worst_start_log_prob"] for e in range(4)]
    assert len(set(starts)) == 4


# --- 4. one compiled loop -------------------------------------------------------------------------

def test_the_whole_chain_is_one_call_of_one_compiled_loop_with_no_host_callback(monkeypatch):
    import jax.monitoring as monitoring

    state = {"running": False, "compiles_while_running": 0, "calls": 0, "jaxpr": None}
    real = B._make_loop

    class _Timed:
        def __init__(self, compiled):
            self.compiled = compiled

        def __call__(self, *a):
            state["calls"] += 1
            state["running"] = True
            try:
                return jax.block_until_ready(self.compiled(*a))
            finally:
                state["running"] = False

    class _Lowered:
        def __init__(self, lowered):
            self.lowered = lowered

        def compile(self):
            return _Timed(self.lowered.compile())

    class _Run:
        def __init__(self, run):
            self.run = run

        def lower(self, *a):
            state["jaxpr"] = jax.make_jaxpr(self.run)(*a)
            return _Lowered(self.run.lower(*a))

    def spy(*a, **k):
        run, score = real(*a, **k)
        return _Run(run), score

    def listener(event, duration, **kw):
        if event == "/jax/core/compile/backend_compile_duration" and state["running"]:
            state["compiles_while_running"] += 1

    monkeypatch.setattr(B, "_make_loop", spy)
    monitoring.register_event_duration_secs_listener(listener)
    try:
        r = _fit(_alert()[0], _quadratic(), nwalkers=8, nsteps=300, burnin=100)
    finally:
        monitoring.unregister_event_duration_listener(listener)
    assert r.info["pilot"]["steps"] == 60
    assert state["calls"] == 1, "the chain was not one call of the compiled loop"
    assert state["compiles_while_running"] == 0
    body = state["jaxpr"].jaxpr
    while len(body.eqns) == 1 and "jaxpr" in body.eqns[0].params:     # the jit around the loop
        body = body.eqns[0].params["jaxpr"].jaxpr
    top = [e.primitive.name for e in body.eqns]
    # a fori_loop with static bounds is a scan; either way exactly one loop
    assert top.count("while") + top.count("scan") == 1, top

    found = set()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            found.add(eqn.primitive.name)
            for p in eqn.params.values():
                for sub in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(sub, "jaxpr"):
                        walk(sub.jaxpr)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(state["jaxpr"].jaxpr)
    assert not {nm for nm in found if "callback" in nm}, sorted(found)


# --- 5. the result --------------------------------------------------------------------------------

def test_the_result_keeps_the_schema_with_the_ensembles_laid_out(tmp_path):
    model = _quadratic()
    lc = _alert()[0]
    r = _quiet(wp.fit, lc, model, sampler="emcee_jax", space="flux", nwalkers=8, nsteps=500,
               burnin=100, thin=4, n_ensembles=3, pilot_steps=50)
    assert r.sampler == "emcee_jax" and r.parameters == NAMES and r.n_params == 3
    assert r.samples_by_chain.shape == (24, 100, 3) and r.n_samples == 2400
    first = r.samples_by_chain[:, 0]                           # step-major flat order, as emcee's
    np.testing.assert_array_equal(r.samples[NAMES].to_numpy()[:24], first)
    info = r.info
    assert info["nwalkers"] == 24 and info["walkers_per_ensemble"] == 8
    assert info["n_ensembles"] == 3 and len(info["ensemble_seeds"]) == 3
    assert set(info["ensemble_rhat"]) == set(NAMES) and set(info["ensemble_ess_bulk"]) == set(NAMES)
    assert len(info["acceptance_fraction"]) == 24 and len(info["autocorr_time_by_ensemble"]) == 3
    assert info["pilot"]["steps"] == 50 and info["pilot"]["best_ensemble"] in (0, 1, 2)
    assert len(info["pilot"]["max_log_likelihood_each"]) == 3
    assert info["walker_chunk"] == 4 and info["program"] == "data as arguments"
    assert info["init"] == "prior_scan" and len(info["init_detail"]["per_ensemble"]) == 3
    for key in ("compile_time_s", "init_time_s", "postprocess_s"):
        assert np.isfinite(info[key]) and info[key] >= 0.0
    assert np.isfinite(r.aic) and np.isfinite(r.max_log_likelihood)
    report = r.diagnostics()
    assert any(row.check.startswith("split R-hat") for row in report.rows)
    path = r.save(tmp_path / "fit.npz")
    back = wp.load_result(path)
    np.testing.assert_array_equal(back.samples_by_chain, r.samples_by_chain)
    assert back.info["ensemble_seeds"] == info["ensemble_seeds"]


# --- 6. the pilot ---------------------------------------------------------------------------------

def _two_modes():
    """Two unit Gaussians in 2-D at x = -15 and x = +15, the left one 20 nats lower, with a
    barrier of about 90 nats between them: the stretch move does not cross it."""
    def log_post(theta):
        a = -0.5 * jnp.sum((theta - jnp.array([15.0, 0.0])) ** 2)
        b = -0.5 * jnp.sum((theta - jnp.array([-15.0, 0.0])) ** 2) - 20.0
        return jnp.logaddexp(a, b)

    prior = Prior({"x": Uniform(-40.0, 40.0), "y": Uniform(-40.0, 40.0)})
    model = M.Model(name="two_modes_device_toy", predict=lambda p, t, b=None: np.zeros(len(t)),
                    parameters=["x", "y"], default_prior=prior)
    lc = wp.LightCurve(time=np.arange(3.0), band=["r"] * 3, flux=np.ones(3),
                       flux_err=np.ones(3))
    rng = np.random.default_rng(0)
    starts = np.stack([rng.normal([-15.0 if e == 0 else 15.0, 0.0], 0.5, (8, 2)) for e in range(4)])
    return log_post, model, lc, starts


def test_the_pilot_restarts_a_stranded_ensemble_from_the_best_one():
    log_post, model, lc, starts = _two_modes()
    kw = dict(nwalkers=8, nsteps=1500, burnin=500, thin=5, init=starts)
    r = _fit(lc, model, log_prob_fn=log_post, pilot_steps=300, **kw)
    p = r.info["pilot"]
    assert p["best_ensemble"] != 0
    assert p["gap_each"][0] > 15.0 and max(p["gap_each"][1:]) < 5.0
    assert p["n_ensembles_dropped"] == 1
    # every ensemble, the best one included, restarts from the best one's states: not independent
    assert p["restarted_ensembles"] == [0, 1, 2, 3] and p["independent"] is False
    assert set(p["rhat"]) == {"x", "y"} and p["rhat"]["x"] > 1.2   # the pilot saw the two modes
    assert np.all(r.samples_by_chain[..., 0] > 0.0)             # every ensemble in the top mode
    assert max(r.info["ensemble_rhat"].values()) < 1.05
    assert r.info["max_log_likelihood_is"] == "sampled density"
    off = _fit(lc, model, log_prob_fn=log_post, pilot=False, **kw)
    assert np.all(off.samples_by_chain[:8, :, 0] < 0.0)         # ensemble 0 stays stranded
    assert off.info["converged"] is False
    assert any("ensembles disagree" in p for p in off.info["convergence_problems"])


# --- 7. helpers and errors ------------------------------------------------------------------------

def test_ensemble_seeds_are_distinct_stable_and_disjoint_between_seeds():
    a = ensemble_seeds(0, 8)
    assert len(set(a)) == 8 and all(0 <= s < 2 ** 31 for s in a)
    assert ensemble_seeds(0, 3) == a[:3] and ensemble_seeds(0, 8) == a
    assert not set(a) & set(ensemble_seeds(1, 8))


def test_pilot_plan_defaults_and_refusals():
    assert pilot_plan("auto", "auto", 4, 5000) == {"steps": 1000, "burnin": 500, "keep": 64,
                                                    "thin": 7}
    assert pilot_plan("auto", "auto", 4, 20) == {"steps": 4, "burnin": 2, "keep": 2, "thin": 1}
    assert pilot_plan("auto", "auto", 1, 5000) is None
    assert pilot_plan("auto", "auto", 4, 9) is None             # an auto pilot under 2 steps
    assert pilot_plan(True, 100, 1, 5000)["steps"] == 100
    assert pilot_plan(False, 100, 4, 5000) is None
    with pytest.raises(ValueError, match="pilot must be"):
        pilot_plan("yes", "auto", 4, 5000)
    with pytest.raises(ValueError, match="at least 2 steps"):
        pilot_plan(True, 1, 4, 5000)


def test_walker_chunk_blocks_and_per_ensemble_starts():
    model = _quadratic()
    lc = _alert()[0]
    r = _fit(lc, model, nwalkers=12, walker_chunk=4, n_ensembles=2, nsteps=300, burnin=100)
    assert r.info["walker_chunk"] == 4 and np.isfinite(r.max_log_likelihood)
    assert _fit(lc, model, nwalkers=12, walker_chunk=None, n_ensembles=2, nsteps=300,
                burnin=100).info["walker_chunk"] is None
    starts = np.stack([np.random.default_rng(e).normal([40.0, -5.0, 2.0], 0.5, (8, 3))
                       for e in range(2)])
    r = _fit(lc, model, nwalkers=8, n_ensembles=2, pilot=False, init=starts, nsteps=300,
             burnin=100)
    assert r.info["init"] == "per_walker"
    point = _fit(lc, model, nwalkers=8, n_ensembles=2, init={"c0": 40.0, "c1": -5.0, "c2": 2.0},
                 nsteps=300, burnin=100)
    assert point.info["init"] == "ball"


def test_errors_name_the_cause():
    model = _quadratic()
    lc = _alert()[0]
    with pytest.raises(ValueError, match="n_ensembles must be at least 1"):
        _fit(lc, model, n_ensembles=0)
    with pytest.raises(ValueError, match="twice the 3 sampled"):
        _fit(lc, model, nwalkers=4)
    with pytest.raises(ValueError, match="keeps no draw"):
        _fit(lc, model, nsteps=300, burnin=295, thin=10)
    with pytest.raises(ValueError, match="pilot must be"):
        _fit(lc, model, pilot="always")
    with pytest.raises(ValueError, match=r"\(n_ensembles, nwalkers, ndim\)"):
        _fit(lc, model, nwalkers=8, init=np.zeros((3, 8, 3)))
    with pytest.raises(ValueError, match="unknown init"):
        _fit(lc, model, init="uniform")
    with pytest.raises(ValueError, match="walker_chunk"):
        _fit(lc, model, walker_chunk=0)


# --- 8. the ensemble ESS ---------------------------------------------------------------------------

def test_the_ensemble_ess_counts_walkers_as_chains_not_interleaved_draws():
    pytest.importorskip("arviz")
    from whisper_cbpf.samplers.jax import _diagnostics as dg
    from whisper_cbpf.samplers.jax.emcee_jax import _ensemble_rhat

    rng = np.random.default_rng(5)
    rho, n_keep, k, w = 0.95, 2000, 4, 16
    x = np.empty((n_keep, k * w, 2))
    x[0] = rng.standard_normal((k * w, 2))
    for t in range(1, n_keep):                       # every walker its own AR(1) chain
        x[t] = rho * x[t - 1] + np.sqrt(1.0 - rho ** 2) * rng.standard_normal((k * w, 2))
    expected = x.shape[0] * x.shape[1] * (1.0 - rho) / (1.0 + rho)          # about 3300
    out = _ensemble_rhat(x, k, ["a", "b"])
    for nm in ("a", "b"):
        assert 0.6 * expected < out["ess_bulk"][nm] < 1.6 * expected, (out["ess_bulk"], expected)
        assert out["ess_tail"][nm] < 0.25 * x.shape[0] * x.shape[1]        # far below the draws
    # the R-hat is the step-order one, as before; the R-hat-only call leaves the ESS out
    per = np.stack([x[:, e * w:(e + 1) * w].reshape(-1, 2) for e in range(k)])
    assert out["rhat"] == dg.rank_diagnostics(per, ["a", "b"])["rhat"]
    quick = _ensemble_rhat(x, k, ["a", "b"], ess=False)
    assert quick["rhat"] == out["rhat"] and np.isnan(quick["ess_bulk"]["a"])
