"""``emcee_jax`` with ``until_converged=True``: compiled chunks until the checks pass.

The claims, each a test:

1. **A converged toy stops early.** On a model whose posterior is a known Gaussian, the run stops
   at the first check that passes, well before ``max_steps``: the stop reason is "converged", the
   last chunk passed and none before it did, the fit's own checks agree, and the kept chain is
   exactly the steps run.
2. **``max_steps`` is respected.** On a target whose ensembles sit in two modes (so R-hat between
   them never passes), the run stops at ``max_steps`` exactly, with every chunk's failure recorded.
3. **``max_time_s`` is respected.** With a time cap shorter than one chunk, only the first chunk
   runs (it always does), and the stop reason says why.
   The checks run cheap first: the R-hat and ESS checks, which rank every draw, run only once the
   walker and chain-length checks pass, and again only after the kept chain grew by
   ``FULL_CHECK_GROWTH``.
4. **Each chunk is a call of ONE compiled program.** The chunk program is lowered and compiled
   once; nothing compiles while the chunks run; its jaxpr has one loop and no host callback.
5. **Off is unchanged.** ``until_converged=False`` runs ``nsteps`` in one call and says so in
   ``info["until_converged"]``; a chunked run stopped after its first chunk draws the same chain.
6. **:func:`until_plan`** resolves the defaults and refuses a ``check_every`` that is not a
   multiple of ``thin``, a ``max_steps`` below ``nsteps`` and a non-positive ``max_time_s``. The
   defaults are the measured budget: ``max_time_s`` 600 s and ``max_steps`` 40 x ``nsteps``
   (200 000 at the default 5000), and ``until_converged`` stays off unless asked for.
7. **Through ``wp.fit``** a run until converged gives no warning; a short run gives the plain
   "Not converged:" warning, which tells how to run longer.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
import whisper_cbpf.models as M  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax import batch as B  # noqa: E402
from whisper_cbpf.samplers.jax import emcee_jax as EJ  # noqa: E402
from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax, until_plan  # noqa: E402

NAMES = ["c0", "c1", "c2"]


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _quadratic():
    """flux = c0 + c1 t + c2 t^2 with Gaussian errors and a wide flat prior: a known Gaussian."""
    def predict(p, t, bands=None):
        t = np.asarray(t, dtype=float)
        return p["c0"] + p["c1"] * t + p["c2"] * t * t

    def predict_jax(theta, t):
        return theta[0] + theta[1] * t + theta[2] * t * t

    return M.Model(name="quadratic_until_toy", predict=predict, parameters=list(NAMES),
                   default_prior=Prior({n: Uniform(-200.0, 200.0) for n in NAMES}),
                   predict_jax=predict_jax)


def _alert(n=24, sigma=0.3):
    t = np.linspace(0.0, 3.0, n)
    rng = np.random.default_rng(1)
    flux = 40.0 - 5.0 * t + 2.0 * t * t + rng.normal(0.0, sigma, n)
    return wp.LightCurve(time=t, band=["r"] * n, flux=flux, flux_err=np.full(n, sigma))


def _two_modes():
    """Two unit Gaussians 30 apart with a ~90-nat barrier; ensemble 0 starts in the left one."""
    def log_post(theta):
        a = -0.5 * jnp.sum((theta - jnp.array([15.0, 0.0])) ** 2)
        b = -0.5 * jnp.sum((theta - jnp.array([-15.0, 0.0])) ** 2)
        return jnp.logaddexp(a, b)

    prior = Prior({"x": Uniform(-40.0, 40.0), "y": Uniform(-40.0, 40.0)})
    model = M.Model(name="two_modes_until_toy", predict=lambda p, t, b=None: np.zeros(len(t)),
                    parameters=["x", "y"], default_prior=prior)
    lc = wp.LightCurve(time=np.arange(3.0), band=["r"] * 3, flux=np.ones(3), flux_err=np.ones(3))
    rng = np.random.default_rng(0)
    starts = np.stack([rng.normal([-15.0 if e == 0 else 15.0, 0.0], 0.5, (8, 2))
                       for e in range(4)])
    return log_post, model, lc, starts


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _stuck_in_two_modes(**kw):
    log_post, model, lc, starts = _two_modes()
    return _quiet(fit_emcee_jax, lc, model, log_prob_fn=log_post, nwalkers=8, nsteps=400,
                  burnin=100, thin=5, init=starts, pilot=False, until_converged=True,
                  check_every=100, seed=0, **kw)


# --- 1. early stop ----------------------------------------------------------------------------

def test_a_converged_toy_stops_at_the_first_check_that_passes():
    r = _quiet(fit_emcee_jax, _alert(), _quadratic(), nwalkers=16, nsteps=300, burnin=100,
               thin=2, space="flux", seed=0, until_converged=True, max_steps=40000,
               max_time_s=None)
    uc = r.info["until_converged"]
    assert uc["enabled"] is True and uc["stop_reason"] == "converged"
    assert uc["steps_run"] < 40000
    assert uc["steps_run"] == 300 + uc["check_every"] * (uc["n_chunks"] - 1)
    assert uc["check_every"] == 100                          # half of nsteps - burnin
    assert uc["n_chunks"] == uc["compiled_calls"] == len(uc["chunks"]) >= 2
    assert uc["chunks"][-1]["passed"] and not any(c["passed"] for c in uc["chunks"][:-1])
    assert [c["main_steps"] for c in uc["chunks"]] == [
        300 + uc["check_every"] * i for i in range(uc["n_chunks"])]
    assert r.info["nsteps"] == uc["steps_run"] and uc["nsteps_requested"] == 300
    assert r.samples_by_chain.shape == (64, (uc["steps_run"] - 100) // 2, 3)
    assert r.info["converged"] is True, r.info["convergence_problems"]
    assert r.diagnostics().passed
    assert uc["chunks"][-1]["n_over_tau"] >= 50
    assert max(r.info["ensemble_rhat"].values()) < 1.01
    # cheap checks first: R-hat and ESS only once the chain-length check passed, and again only
    # after the kept chain grew by FULL_CHECK_GROWTH
    for c in uc["chunks"]:
        if c["stage"] == 1:
            assert c["ensemble_rhat_max"] is None and c["ess_bulk_min"] is None
        else:
            assert c["n_over_tau"] >= 50 and not c["stuck_walkers"]
    full = [c["kept_draws"] for c in uc["chunks"] if c["stage"] > 1]
    assert full and uc["chunks"][-1]["stage"] == 3
    assert all(b >= (1 + EJ.FULL_CHECK_GROWTH) * a for a, b in zip(full, full[1:]))


# --- 2. max_steps -----------------------------------------------------------------------------

def test_max_steps_is_respected_when_the_checks_never_pass():
    r = _stuck_in_two_modes(max_steps=800, max_time_s=None)
    uc = r.info["until_converged"]
    assert uc["stop_reason"] == "max_steps"
    assert uc["steps_run"] == 800 and uc["n_chunks"] == 5
    assert r.samples_by_chain.shape[1] == (800 - 100) // 5
    assert all(c["failed_checks"] and not c["passed"] for c in uc["chunks"])
    assert max(r.info["ensemble_rhat"].values()) > 1.5
    assert r.info["converged"] is False
    assert any("ensembles disagree" in p for p in r.info["convergence_problems"])


# --- 3. max_time_s ----------------------------------------------------------------------------

def test_max_time_s_stops_after_the_first_chunk():
    r = _stuck_in_two_modes(max_steps=None, max_time_s=1e-3)
    uc = r.info["until_converged"]
    assert uc["stop_reason"] == "max_time_s" and uc["n_chunks"] == 1
    assert uc["steps_run"] == 400 and uc["max_time_s"] == 1e-3 and uc["max_steps"] is None
    for key in ("call_s", "check_s", "elapsed_s"):
        assert uc["chunks"][0][key] >= 0.0


# --- 4. one compiled program ------------------------------------------------------------------

def test_every_chunk_is_a_call_of_one_compiled_program(monkeypatch):
    import jax.monitoring as monitoring

    state = {"running": False, "compiles_while_running": 0, "calls": 0, "lowered": 0,
             "jaxpr": None}
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

    class _Advance:
        def __init__(self, fn):
            self.fn = fn

        def lower(self, *a):
            state["lowered"] += 1
            state["jaxpr"] = jax.make_jaxpr(self.fn)(*a)
            return _Lowered(self.fn.lower(*a))

    def spy(*a, **k):
        made = real(*a, **k)
        assert k.get("segments") is True
        return (_Advance(made[0]),) + tuple(made[1:])

    def listener(event, duration, **kw):
        if event == "/jax/core/compile/backend_compile_duration" and state["running"]:
            state["compiles_while_running"] += 1

    monkeypatch.setattr(B, "_make_loop", spy)
    monitoring.register_event_duration_secs_listener(listener)
    try:
        r = _stuck_in_two_modes(max_steps=700, max_time_s=None)
    finally:
        monitoring.unregister_event_duration_listener(listener)
    uc = r.info["until_converged"]
    assert uc["n_chunks"] == 4
    assert state["lowered"] == 1, "the chunk program was lowered more than once"
    assert state["calls"] == 4 == uc["compiled_calls"]
    assert state["compiles_while_running"] == 0
    body = state["jaxpr"].jaxpr
    while len(body.eqns) == 1 and "jaxpr" in body.eqns[0].params:
        body = body.eqns[0].params["jaxpr"].jaxpr
    top = [e.primitive.name for e in body.eqns]
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


# --- 5. off is unchanged ----------------------------------------------------------------------

def test_off_runs_nsteps_in_one_call_and_a_first_chunk_draws_the_same_chain():
    kw = dict(nwalkers=8, nsteps=400, burnin=100, thin=5, space="flux", seed=3, pilot_steps=60)
    off = _quiet(fit_emcee_jax, _alert(), _quadratic(), **kw)
    assert off.info["until_converged"] == {"enabled": False, "steps_run": 400,
                                           "stop_reason": "nsteps"}
    on = _quiet(fit_emcee_jax, _alert(), _quadratic(), until_converged=True, max_steps=400, **kw)
    assert on.info["until_converged"]["n_chunks"] == 1
    assert on.info["until_converged"]["stop_reason"] in ("converged", "max_steps")
    assert on.samples_by_chain.shape == off.samples_by_chain.shape
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(on.samples_by_chain, off.samples_by_chain)
        assert on.info["pilot"]["best_ensemble"] == off.info["pilot"]["best_ensemble"]
    else:   # another program: GPU kernels may round differently, so compare the posterior
        np.testing.assert_allclose(on.samples[NAMES].mean().to_numpy(),
                                   off.samples[NAMES].mean().to_numpy(), atol=0.5)
    assert on.info["acceptance_fraction"] and on.info["mean_acceptance_fraction"] > 0.0


# --- 6. until_plan ----------------------------------------------------------------------------

def test_until_plan_defaults_and_refusals():
    plan = until_plan(5000, 1000, 10)
    assert plan == {"first_keep": 400, "every_keep": 200, "check_every": 2000,
                    "max_steps": EJ.UNTIL_MAX_STEPS_FACTOR * 5000,
                    "max_time_s": EJ.UNTIL_MAX_TIME_S}
    assert until_plan(300, 100, 7)["check_every"] == 98       # a multiple of thin
    assert until_plan(110, 100, 10)["check_every"] == 10      # at least thin
    assert until_plan(5000, 1000, 10, 500, None, None)["max_time_s"] is None
    with pytest.raises(ValueError, match="multiple of thin=10"):
        until_plan(5000, 1000, 10, check_every=25)
    with pytest.raises(ValueError, match="below nsteps=5000"):
        until_plan(5000, 1000, 10, max_steps=4000)
    with pytest.raises(ValueError, match="positive or None"):
        until_plan(5000, 1000, 10, max_time_s=0)
    with pytest.raises(ValueError, match="'auto'"):
        until_plan(5000, 1000, 10, max_steps="lots")


def test_the_default_budget_is_the_measured_one():
    import inspect
    assert EJ.UNTIL_MAX_TIME_S == 600.0 and EJ.UNTIL_MAX_STEPS_FACTOR == 40
    plan = until_plan(5000, 1000, 10)
    assert plan["max_steps"] == 200_000 and plan["max_time_s"] == 600.0
    assert inspect.signature(fit_emcee_jax).parameters["until_converged"].default is False


# --- 7. through wp.fit ------------------------------------------------------------------------

def test_fit_until_converged_is_silent_and_a_short_run_says_how_to_run_longer():
    lc, model = _alert(), _quadratic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = wp.fit(lc, model, sampler="emcee_jax", nwalkers=16, nsteps=300, burnin=100, thin=2,
                   space="flux", until_converged=True, max_time_s=None)
    assert not [w for w in caught if "Not converged" in str(w.message)]
    assert r.info["convergence_verdict"]["passed"] is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        short = wp.fit(lc, model, sampler="emcee_jax", nwalkers=16, nsteps=120, burnin=60,
                       thin=2, space="flux")
    ours = [str(w.message) for w in caught if "onverged" in str(w.message)]
    assert len(ours) == 1 and ours[0].startswith("Not converged:\n")
    assert "Run longer (until_converged=True, or nsteps=" in ours[0]
