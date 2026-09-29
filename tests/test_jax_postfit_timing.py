"""the GPU samplers' time goes where they say it goes.

#10. ``nuts_gpu``, ``pymc_jax_gpu_*`` and ``emcee_jax`` score every kept draw after sampling, to find
the best fit for AIC/BIC and the per-chain log-likelihood. That re-scan was
``lax.map(log_prob_fn, draws)``, one draw per step, outside ``runtime_s``: on the TDE at
``n_time=5000`` 50.9 ms a draw, ~672 s for 13,200 draws against 156 s of sampling. It now runs in
fixed-width vmapped blocks (``_diagnostics.block_scorer``) and ``info["postprocess_s"]`` reports it.

#13. emcee evaluates the whole ensemble once, then each half on every step. ``emcee_jax`` compiled
only the whole-ensemble shape before its timer started, so the half-ensemble compile (0.7-0.9 s on
the supernovae and the kilonova, 8.9 s on the TDE) landed inside ``runtime_s``.

CPU JAX is enough: the properties are counts (traces, compiles) and equalities, not speeds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("emcee")

import jax.numpy as jnp  # noqa: E402

import whisper_cbpf as wp  # noqa: E402


def _predict(parameters, times, bands=None):
    t = np.asarray(times, dtype=float)
    return np.asarray(parameters["amp"]) * np.exp(-t / np.asarray(parameters["tau"]))


def _predict_jax(theta, times, band_idx=None):
    theta = jnp.atleast_2d(theta)
    return theta[:, :1] * jnp.exp(-jnp.asarray(times)[None, :] / theta[:, 1:2])


@pytest.fixture(scope="module")
def model():
    return wp.register_model(
        "p5_decay", _predict, ["amp", "tau"], overwrite=True, predict_jax=_predict_jax,
        prior=wp.Prior({"amp": wp.Uniform(0.5, 5.0), "tau": wp.LogUniform(1.0, 30.0)}))


@pytest.fixture(scope="module")
def lc():
    t = np.linspace(0.5, 20.0, 12)
    rng = np.random.default_rng(3)
    y = 2.0 * np.exp(-t / 6.0) + rng.normal(0.0, 0.05, t.size)
    return wp.LightCurve(time=t, band=np.array(["ztfg"] * t.size), flux=y,
                         flux_err=np.full(t.size, 0.05), name="decay")


def _counting(fn):
    """``fn`` plus a counter of how many times JAX TRACED it (Python runs only at trace time)."""
    calls = {"n": 0}

    def wrapped(theta):
        calls["n"] += 1
        return fn(theta)

    return wrapped, calls


def _density():
    t = jnp.linspace(0.5, 20.0, 12)
    y = 2.0 * jnp.exp(-t / 6.0)
    return lambda th: -0.5 * jnp.sum(((y - th[0] * jnp.exp(-t / th[1])) / 0.05) ** 2)


# -------------------------------------------------------------------------- #10: block_scorer
@pytest.mark.parametrize("n", [1, 5, 16, 37, 1000])
def test_block_scorer_equals_the_density_draw_by_draw(n):
    """Every row scored exactly once, in order, whatever ``n`` is against the block width --
    including the padded last block and a draw count below one block."""
    from whisper_cbpf.samplers.jax import _diagnostics as dg

    f = _density()
    theta = np.column_stack([np.linspace(0.6, 4.9, n), np.linspace(1.1, 29.0, n)])
    got = dg.block_scorer(f, jnp.float32 if not jax.config.jax_enable_x64 else jnp.float64,
                          width=16)(theta)
    want = np.array([float(f(jnp.asarray(r))) for r in theta])
    assert got.shape == (n,)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_block_scorer_compiles_one_program_for_any_number_of_draws():
    """One trace of the density, however many draws and however many calls: the prior scan, the
    start trials and the post-fit re-scan share it. ``lax.map(..., batch_size=)`` would trace a
    second (remainder) vmap and a new program per draw count."""
    from whisper_cbpf.samplers.jax import _diagnostics as dg

    f, calls = _counting(_density())
    score = dg.block_scorer(f, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32, width=16)
    for n in (1000, 37, 5, 16):
        score(np.column_stack([np.full(n, 2.0), np.full(n, 6.0)]))
    assert calls["n"] == 1, f"the density was traced {calls['n']} times; one width, one trace"


# ------------------------------------------------------------- the default start's curvature
def test_start_curvature_traces_nothing_new():
    """The conditional sd that spreads the default start is a second difference through the scan's
    compiled scorer, so once the scan has run it traces (hence compiles) nothing. It was
    ``jax.hessian`` through the model: 57 of the TDE's 99 s start on one A6000, 76 of 96 s on the
    CPU (``n_time=500``), in an ``emcee_jax`` fit whose sampling took seconds."""
    from whisper_cbpf.samplers.jax import _diagnostics as dg

    f, calls = _counting(_density())
    score = dg.block_scorer(f, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32, width=16)
    score(np.array([[2.0, 6.0]]))
    prior = wp.Prior({"amp": wp.Uniform(0.5, 5.0), "tau": wp.LogUniform(1.0, 30.0)})
    sd = dg._conditional_sd(score, np.array([[2.0, 6.0], [1.5, 8.0]]), prior, ["amp", "tau"])
    assert calls["n"] == 1, f"the curvature traced the density {calls['n'] - 1} more time(s)"
    assert sd.shape == (2, 2) and np.all(sd > 0.0) and np.all(sd <= dg.MAX_SPREAD_U)


def test_start_curvature_is_the_conditional_sd_of_a_gaussian_in_the_samplers_coordinates():
    """On a log-density quadratic in the logit of each prior's own coordinate (log10 for a
    LogUniform) with precision ``P``, the sd is ``1 / sqrt(P_jj)`` -- the CONDITIONAL width, not
    the marginal ``sqrt((P^-1)_jj)`` -- at the mode and away from it, and ``MAX_SPREAD_U`` along a
    coordinate the density ignores. Here the second difference is exact from its first round."""
    from whisper_cbpf.samplers.jax import _diagnostics as dg

    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        names = ["a", "b", "c"]
        prior = wp.Prior({"a": wp.Uniform(-3.0, 5.0), "b": wp.LogUniform(0.1, 100.0),
                          "c": wp.Uniform(0.0, 1.0)})
        is_log, lo, hi = dg._site_box(prior, names)
        prec = np.array([[40.0, 12.0], [12.0, 9.0]])
        mode = np.array([0.3, -0.8])

        def density(x):
            s = jnp.where(jnp.asarray(is_log), jnp.log10(jnp.abs(x)), x)
            p = (s - lo) / (hi - lo)
            d = (jnp.log(p) - jnp.log1p(-p))[:2] - mode
            return -0.5 * d @ jnp.asarray(prec) @ d          # "c" is ignored

        u = np.array([[0.3, -0.8, 0.5], [0.5, -1.0, -2.0], [-0.4, 0.1, 1.0]])
        sd = dg._conditional_sd(dg.block_scorer(density, jnp.float64),
                                dg._from_logit(u, prior, names), prior, names)
    finally:
        jax.config.update("jax_enable_x64", was)
    want = np.tile([1.0 / np.sqrt(40.0), 1.0 / np.sqrt(9.0), dg.MAX_SPREAD_U], (3, 1))
    np.testing.assert_allclose(sd, want, rtol=1e-6)


def test_nuts_gpu_rescans_with_the_prior_scans_blocks_and_times_it(model, lc, monkeypatch):
    """The re-scan traces nothing new after sampling -- it reuses the block scorer the prior scan
    compiled -- agrees with the density at the best draw, and reports ``postprocess_s``. The old
    ``lax.map(log_prob_fn, draws)`` traced the density once more, after ``MCMC.run``."""
    from numpyro.infer import MCMC

    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    base = make_log_prob_jax(lc, model, model.default_prior, space="flux")
    f, calls = _counting(base)
    after_run = {}
    real_run = MCMC.run

    def run(self, *a, **k):
        out = real_run(self, *a, **k)
        after_run["traces"] = calls["n"]
        return out

    monkeypatch.setattr(MCMC, "run", run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wp.fit(lc, model.name, sampler="nuts_gpu", log_prob_fn=f, space="flux",
                   num_warmup=50, num_samples=100, num_chains=2, seed=0)
    assert calls["n"] == after_run["traces"], "the post-fit re-scan compiled its own program"
    post = r.info["postprocess_s"]
    assert np.isfinite(post) and post >= 0.0
    best = np.array([r.best_params[nm] for nm in r.parameters])
    assert r.max_log_likelihood == pytest.approx(float(base(jnp.asarray(best))), rel=1e-6)


def test_emcee_jax_reports_postprocess_time(model, lc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wp.fit(lc, model.name, sampler="emcee_jax", space="flux", nwalkers=8, nsteps=200,
                   burnin=50, seed=0)
    assert np.isfinite(r.info["postprocess_s"]) and r.info["postprocess_s"] >= 0.0


# ------------------------------------------------------------------ #13: no compile in runtime_s
@pytest.mark.parametrize("nwalkers,walker_chunk", [(60, "half"), (11, "half"), (32, 16),
                                                  (11, None), (12, None), (40, 8)])
def test_emcee_jax_compiles_nothing_while_it_samples(model, lc, monkeypatch, nwalkers,
                                                     walker_chunk):
    """No XLA compile between the start of ``run_mcmc`` and its end, so ``runtime_s`` is sampling.

    Every ensemble shape emcee uses -- the whole ensemble, and the two halves the stretch move
    updates (unequal for an odd count) -- must be compiled in the warm-up, with the chunked walker
    mapper and without it.
    """
    import emcee
    import jax.monitoring as monitoring

    state = {"sampling": False, "compiles_while_sampling": 0}

    def listener(event, duration, **kw):
        if event == "/jax/core/compile/backend_compile_duration" and state["sampling"]:
            state["compiles_while_sampling"] += 1

    real = emcee.EnsembleSampler.run_mcmc

    def timed_run(self, *a, **k):
        state["sampling"] = True
        try:
            return real(self, *a, **k)
        finally:
            state["sampling"] = False

    monkeypatch.setattr(emcee.EnsembleSampler, "run_mcmc", timed_run)
    monitoring.register_event_duration_secs_listener(listener)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wp.fit(lc, model.name, sampler="emcee_jax", space="flux", nwalkers=nwalkers,
                   nsteps=30, burnin=10, thin=1, seed=0, walker_chunk=walker_chunk)
    finally:
        monitoring.unregister_event_duration_listener(listener)
    assert state["compiles_while_sampling"] == 0, (
        f"{state['compiles_while_sampling']} XLA compile(s) inside run_mcmc, i.e. inside runtime_s")


# ------------------------------------------------------------------ #15: the walker-chunk default
@pytest.mark.parametrize("nwalkers,width", [(60, 30), (32, 16), (11, 6)])
def test_emcee_jax_default_walker_chunk_is_one_block_per_emcee_call(model, lc, nwalkers, width):
    """emcee scores half the ensemble per call; the default block is that half, rounded up. A fixed
    16 split a 60-walker ensemble's 30-walker half into two sequential blocks: 50.6 ms against
    22.5 ms per call on the TDE at n_time=5000."""
    from whisper_cbpf.samplers.jax.emcee_jax import DEFAULT_WALKER_CHUNK

    assert DEFAULT_WALKER_CHUNK == "half"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = wp.fit(lc, model.name, sampler="emcee_jax", space="flux", nwalkers=nwalkers,
                   nsteps=20, burnin=5, thin=1, seed=0)
    assert r.info["walker_chunk"] == width


def test_emcee_jax_walker_chunk_rejects_an_unknown_name(model, lc):
    with pytest.raises(ValueError, match="walker_chunk"):
        wp.fit(lc, model.name, sampler="emcee_jax", space="flux", nwalkers=8, nsteps=20,
               burnin=5, walker_chunk="halves")
