"""``init=``: where every MCMC-type sampler starts, and handing one fit to the next.

``mcmc``, ``emcee_jax``, ``nuts_gpu`` and ``pymc_jax_gpu_*`` share one ``init=``: ``"prior_scan"``
(the default), ``"prior"``, a point (a ball), one start per chain or walker, or a previous result.
The result form is the ABC -> MCMC handoff and the staging of an alert's fits (3 -> 6 detections
-> +10 d): with at least as many usable draws as walkers the walkers start on a random distinct
subset of them; with fewer, in a ball around the result's likelihood maximum (or best) point, sized
by its own spread. Every start is checked against the box, the constraint wall and a finite
density, and ``info["init"]`` records the kind that ran.

Why: seeding from a previous fit used to need a monkeypatch of ``emcee_jax._init_walkers``; on
real ZTF and LSST supernovae, seeding a second pass from the first cut took the chains missing the
mode 12 -> 2 and those with stranded walkers 54 -> 2 of 90, while ABC accepted 0-43 draws against
60 walkers, hence the ball. CPU ``mcmc`` from independent prior draws left 12-15 of 60 walkers
stuck on redback ``arnett`` / SN2025pgp, hence its new default.
"""
from __future__ import annotations

import json
import os
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.samplers.base import SamplerResult
from whisper_cbpf.samplers.jax import _diagnostics as dg

HERE = os.path.dirname(os.path.abspath(__file__))
MOCKS = json.load(open(os.path.join(HERE, "data", "u3_bump_mocks.json")))

PRIOR = wp.Prior({"T": wp.LogUniform(100.0, 6000.0), "x": wp.Uniform(0.0, 10.0)})
NAMES = ["T", "x"]


def _batch(fn):
    return lambda th: np.array([fn(t) for t in np.asarray(th)], dtype=float)


FLAT = _batch(lambda t: 0.0)


def _result(rows, best=None, info=None, sampler="abc"):
    rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    return SamplerResult(sampler=sampler, model="m", parameters=NAMES,
                         samples=pd.DataFrame(rows, columns=NAMES), summary={},
                         best_params=dict(best or {}), n_data=1, n_params=2, runtime_s=0.0,
                         info=dict(info or {}))


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ------------------------------------------------------------------ seeding from a result (unit)
def test_a_result_with_enough_usable_draws_starts_on_a_random_distinct_subset():
    """An emcee chain repeats a row on every rejected step; the starts are distinct draws."""
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.uniform(200, 5000, 60), rng.uniform(1, 9, 60)])
    rows = np.repeat(rows, 3, axis=0)                           # 180 rows, 60 distinct
    res = _result(rows)
    p0, lp, label, detail = dg.explicit_starts(res, PRIOR, NAMES, 32, 0, FLAT, "t")
    assert label == "result" and detail["mode"] == "draws" and detail["source"] == "abc"
    assert detail["n_draws"] == 180 and detail["n_distinct_in_box"] == 60
    assert len({tuple(r) for r in p0}) == 32 and np.all(np.isfinite(lp))
    assert {tuple(r) for r in p0} <= {tuple(r) for r in rows}
    other, *_ = dg.explicit_starts(res, PRIOR, NAMES, 32, 1, FLAT, "t")
    assert {tuple(r) for r in other} != {tuple(r) for r in p0}       # the subset is random


def test_draws_outside_the_box_or_behind_a_wall_are_never_used():
    """Only draws inside the box and at a finite density here count as usable."""
    rng = np.random.default_rng(1)
    inside = np.column_stack([rng.uniform(200, 5000, 40), rng.uniform(1, 9, 40)])
    outside = np.column_stack([rng.uniform(200, 5000, 10), np.full(10, 12.0)])
    wall = _batch(lambda t: 0.0 if t[1] >= 5.0 else -np.inf)        # -inf for x < 5
    res = _result(np.vstack([inside, outside]), best={"T": 800.0, "x": 7.0})
    n_ok = int((inside[:, 1] >= 5.0).sum())
    p0, _, _, detail = dg.explicit_starts(res, PRIOR, NAMES, n_ok, 0, wall, "t")
    assert detail["mode"] == "draws" and np.all(p0[:, 1] >= 5.0) and np.all(p0[:, 1] < 10.0)
    # one walker more than there are usable draws: the ball, still behind the wall's open side
    p1, _, _, detail = dg.explicit_starts(res, PRIOR, NAMES, n_ok + 1, 0, wall, "t")
    assert detail["mode"] == "ball" and np.all(p1[:, 1] >= 5.0)


def test_too_few_draws_fall_back_to_a_ball_sized_by_the_results_own_spread():
    """An ABC fit that accepted 20 draws, for 60 walkers: a ball around its best point whose width
    is the sd of those draws in each prior's own coordinate (log for the LogUniform)."""
    rng = np.random.default_rng(2)
    rows = np.column_stack([np.exp(rng.normal(np.log(800.0), 0.2, 20)), rng.normal(6.0, 0.5, 20)])
    res = _result(rows, best={"T": 800.0, "x": 6.0})
    p0, lp, label, detail = dg.explicit_starts(res, PRIOR, NAMES, 60, 0, FLAT, "t")
    assert label == "result" and detail["mode"] == "ball" and detail["centre"] == "best_params"
    sd = detail["ball_sd_own"]
    assert sd["T"] == pytest.approx(np.std(np.log(rows[:, 0])))
    assert sd["x"] == pytest.approx(np.std(rows[:, 1]))
    assert abs(np.std(np.log(p0[:, 0])) / sd["T"] - 1.0) < 0.3
    assert abs(np.median(p0[:, 1]) - 6.0) < 0.3 and np.all(np.isfinite(lp))


def test_a_ball_start_that_costs_too_much_is_pulled_back_toward_the_centre():
    """A broad ABC spread around a sharp peak: a start more than max(10, 2k) nats below the centre
    is pulled halfway back, up to 6 times, as the prior scan's spread is -- far out it would sit on
    a flat plateau for the whole run -- while the starts stay distinct."""
    sharp = _batch(lambda t: -0.5 * ((t[1] - 6.0) / 0.1) ** 2)
    rows = np.column_stack([800.0 * np.exp(np.linspace(-1, 1, 10)), np.linspace(1, 9, 10)])
    res = _result(rows, best={"T": 800.0, "x": 6.0})
    p0, lp, _, detail = dg.explicit_starts(res, PRIOR, NAMES, 32, 0, sharp, "t")
    assert detail["mode"] == "ball" and detail["ball_sd_own"]["x"] > 2.0
    assert np.all(lp >= -10.0 - 1e-9), lp.min()
    assert len({tuple(r) for r in p0}) == 32 and p0[:, 1].std() > 0.05


@pytest.mark.parametrize("carrier", ["dict", "object"])
def test_the_ball_centres_on_the_likelihood_maximum_when_the_result_carries_one(carrier):
    optimised = {"T": 1500.0, "x": 3.0}
    peak = ({"params": optimised} if carrier == "dict"
            else types.SimpleNamespace(params=optimised))    # a LikelihoodMaxOptResult has .params
    res = _result(np.array([[800.0, 6.0], [820.0, 6.1]]), best={"T": 800.0, "x": 6.0},
                  info={"likelihood_max_opt": peak})
    p0, _, _, detail = dg.explicit_starts(res, PRIOR, NAMES, 16, 0, FLAT, "t")
    assert detail["mode"] == "ball" and detail["centre"] == "likelihood_max_opt"
    assert abs(np.median(p0[:, 1]) - 3.0) < 0.2


def test_an_abc_fit_that_accepted_nothing_starts_around_its_closest_draw():
    """No accepted draw: ABC's best_params is its closest rejected draw, and the
    ball takes DEFAULT_BALL_SCALE of the prior width, since there is no spread to measure."""
    res = _result(np.empty((0, 2)), best={"T": 800.0, "x": 6.0})
    p0, _, _, detail = dg.explicit_starts(res, PRIOR, NAMES, 16, 0, FLAT, "t")
    assert detail["mode"] == "ball" and "fewer than 2 draws" in detail["spread"]
    assert detail["ball_sd_own"]["x"] == pytest.approx(dg.DEFAULT_BALL_SCALE * 10.0)
    assert np.all(np.abs(p0[:, 1] - 6.0) < 0.1)


def test_a_result_of_another_model_is_refused_by_name():
    other = SamplerResult(sampler="mcmc", model="m2", parameters=["a"],
                          samples=pd.DataFrame({"a": [1.0, 2.0]}), summary={},
                          best_params={"a": 1.0}, n_data=1, n_params=1, runtime_s=0.0)
    with pytest.raises(ValueError, match=r"has no \['T', 'x'\]"):
        dg.explicit_starts(other, PRIOR, NAMES, 8, 0, FLAT, "t")


def test_a_point_and_scale_is_not_mistaken_for_a_two_parameter_point():
    """``(800.0, 3.0)`` is a point in a 2-parameter model; ``([800.0, 3.0], 0.05)`` a point and a
    ball scale."""
    p0, _, label, detail = dg.explicit_starts((800.0, 3.0), PRIOR, NAMES, 16, 0, FLAT, "t")
    assert label == "ball" and detail["scale"] == dg.DEFAULT_BALL_SCALE
    wide, _, label, detail = dg.explicit_starts(([800.0, 3.0], 0.05), PRIOR, NAMES, 16, 0, FLAT,
                                                "t")
    assert label == "ball" and detail["scale"] == 0.05
    assert wide[:, 1].std() > 10 * p0[:, 1].std()
    with pytest.raises(ValueError, match="positive scale"):
        dg.explicit_starts(([800.0, 3.0], -1.0), PRIOR, NAMES, 16, 0, FLAT, "t")
    with pytest.raises(ValueError, match=r"no value for \['x'\]"):
        dg.explicit_starts({"T": 800.0}, PRIOR, NAMES, 16, 0, FLAT, "t")


# ------------------------------------------------------------------------- the bump, end to end
def _bump_numpy(parameters, times, bands=None):
    t = np.asarray(times, dtype=float)
    return float(parameters["A"]) * np.exp(-0.5 * ((t - float(parameters["t0"]))
                                                  / float(parameters["w"])) ** 2)


def _bump_jax(theta, times, band_idx=None):
    import jax.numpy as jnp
    return theta[0] * jnp.exp(-0.5 * ((jnp.asarray(times) - theta[1]) / theta[2]) ** 2)


BUMP_PRIOR = wp.Prior({"A": wp.LogUniform(0.3, 10.0), "t0": wp.Uniform(0.0, 30.0),
                       "w": wp.Uniform(1.0, 8.0)})


@pytest.fixture(scope="module")
def bump():
    kw = {}
    try:
        import jax  # noqa: F401
        kw["predict_jax"] = _bump_jax
    except ImportError:
        pass
    return wp.register_model("starts_bump", _bump_numpy, ["A", "t0", "w"], prior=BUMP_PRIOR,
                             overwrite=True, **kw)


@pytest.fixture
def float64():
    jax = pytest.importorskip("jax")
    before = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


def _sim(i, upto=None):
    s = MOCKS["sims"][str(i)]
    t = np.asarray(s["t_rel"])
    keep = np.ones(t.size, bool) if upto is None else t <= upto
    lc = wp.LightCurve(time=t[keep], band=np.asarray(s["bands"])[keep],
                       flux=np.asarray(s["y"])[keep], flux_err=np.asarray(s["sig"])[keep])
    return lc, s


def _matches_grid(r, s, tol_sd=0.3):
    for nm in ("A", "t0", "w"):
        ref, x = s["ref"][nm], r.samples[nm].to_numpy()
        assert abs(x.mean() - ref["mean"]) < tol_sd * ref["sd"], (nm, x.mean(), ref)
        assert abs(x.std() / ref["sd"] - 1.0) < 0.3, (nm, x.std(), ref)


def test_cpu_mcmc_default_start_avoids_the_stuck_walker_its_prior_start_left(bump):
    """Bump sim 13, seed 0: the old start (``init="prior"``) leaves walker 0 in the zero-signal
    optimum (``tests/test_mcmc.py``); the default prior scan, climbed without a gradient, does not."""
    lc, s = _sim(13)
    r = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", seed=0)
    assert r.info["init"] == "prior_scan" and r.info["init_detail"]["n_climbed"] == 32
    assert r.info["stuck_walkers"] == [], r.info["convergence_problems"]
    assert r.info["init_time_s"] > 0.0 and r.runtime_s >= r.info["init_time_s"]
    _matches_grid(r, s)


def test_walkers_seeded_in_a_tiny_ball_leave_it(bump):
    """A start is only a start: from a 1e-5 ball the walkers spread to the posterior (the
    gate: the walkers must leave the seed region)."""
    lc, s = _sim(0)
    point = {nm: s["truth"][nm] for nm in ("A", "t0", "w")}
    r = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", seed=1, init=(point, 1e-5))
    assert r.info["init"] == "ball" and r.info["init_detail"]["scale"] == 1e-5
    assert r.samples["t0"].std() > 10 * 1e-5 * 30.0                 # 10 x the ball's sd in t0
    _matches_grid(r, s)


def test_mcmc_rejects_init_together_with_initial_guess(bump):
    lc, _ = _sim(0)
    with pytest.raises(ValueError, match="not both"):
        wp.fit_MCMC(lc, bump.name, space="flux", init="prior", initial_guess=[8.0, 15.0, 3.0])
    r = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", nsteps=300, burnin=100,
               initial_guess={"A": 8.0, "t0": 15.4, "w": 2.9})
    assert r.info["init"] == "initial_guess"


def test_an_alert_is_staged_3_then_6_detections_then_10_more_days(bump):
    """Each cut starts from the previous cut's posterior, and the last agrees with a fresh fit of
    the same data from the default start."""
    t = np.asarray(MOCKS["sims"]["0"]["t_rel"])
    cuts = (10.1, 12.25, 22.25)                      # detections from 8.3 d: 3, then 6, then +10 d
    prev = None
    for upto in cuts:
        lc, s = _sim(0, upto)
        kw = {} if prev is None else {"init": prev}
        r = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", seed=2, nsteps=3000, burnin=1000, **kw)
        if prev is not None:
            assert r.info["init"] == "result" and r.info["init_detail"]["mode"] == "draws"
        prev = r
    assert int((t <= cuts[-1]).sum()) == prev.n_data
    fresh = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", seed=3, nsteps=3000, burnin=1000)
    for nm in ("A", "t0", "w"):
        a, b = prev.samples[nm], fresh.samples[nm]
        assert abs(a.mean() - b.mean()) < 0.3 * b.std(), nm
        assert abs(a.std() / b.std() - 1.0) < 0.3, nm
    assert prev.info["stuck_walkers"] == []


def test_abc_hands_over_to_mcmc_by_a_ball_when_it_accepted_fewer_draws_than_walkers(bump):
    lc, s = _sim(0)
    abc = _quiet(wp.fit_ABC, lc, bump.name, n_simulations=2000, quantile=0.004, seed=0,
                 space="flux")
    assert 2 <= abc.n_samples < 32
    r = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", nwalkers=32, seed=0, init=abc)
    assert r.info["init"] == "result" and r.info["init_detail"]["mode"] == "ball"
    assert r.info["init_detail"]["source"] == "abc"
    assert r.info["stuck_walkers"] == []
    _matches_grid(r, s)


def test_init_is_refused_by_a_sampler_that_has_no_start(bump):
    lc, _ = _sim(0)
    with pytest.raises(ValueError, match="init= is for .*'mcmc'"):
        wp.fit(lc, bump.name, sampler="abc", init="prior")


# ------------------------------------------------------------------ the JAX samplers, end to end
def test_nuts_starts_its_chains_on_distinct_draws_of_a_previous_fit(bump, float64):
    pytest.importorskip("numpyro")
    lc, s = _sim(0)
    prev = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", seed=0)
    r = _quiet(wp.fit, lc, bump.name, sampler="nuts_gpu", space="flux", init=prev,
               num_warmup=300, num_samples=1000, seed=0)
    assert r.info["init"] == "result" and r.info["init_detail"]["mode"] == "draws"
    assert r.info["init_strategy"] == "result"
    assert np.all(np.isfinite(r.info["prior_scan"]["start_log_likelihood"]))
    assert r.info["converged"] is True, r.info["convergence_problems"]
    _matches_grid(r, s)
    with pytest.raises(ValueError, match="not both"):
        wp.fit(lc, bump.name, sampler="nuts_gpu", space="flux", init=prev,
               init_strategy="uniform")


def test_emcee_jax_takes_an_abc_result_and_a_point(bump, float64):
    pytest.importorskip("jax")
    lc, s = _sim(0)
    abc = _quiet(wp.fit_ABC, lc, bump.name, n_simulations=20000, quantile=0.005, seed=1,
                 space="flux")
    assert abc.n_samples >= 32
    r = _quiet(wp.fit, lc, bump.name, sampler="emcee_jax", space="flux", init=abc, nsteps=3000,
               burnin=1000, seed=0)
    assert r.info["init"] == "result" and r.info["init_detail"]["mode"] == "draws"
    assert r.info["stuck_walkers"] == []
    _matches_grid(r, s)
    pt = _quiet(wp.fit, lc, bump.name, sampler="emcee_jax", space="flux",
                init={"A": 8.2, "t0": 15.4, "w": 2.9}, nsteps=300, burnin=100)
    assert pt.info["init"] == "ball"


def test_pymc_takes_a_point_and_a_previous_result(bump, float64):
    pytest.importorskip("pymc")
    lc, s = _sim(0)
    prev = _quiet(wp.fit_MCMC, lc, bump.name, space="flux", seed=0)
    r = _quiet(wp.fit, lc, bump.name, sampler="pymc_jax_gpu_vectorized", space="flux",
               init=prev, num_warmup=300, num_samples=500, seed=0)
    assert r.info["init"] == "result" and r.info["init_detail"]["mode"] == "draws"
    pt = _quiet(wp.fit, lc, bump.name, sampler="pymc_jax_gpu_vectorized", space="flux",
                init={"A": 8.2, "t0": 15.4, "w": 2.9}, num_warmup=100, num_samples=100)
    assert pt.info["init"] == "ball"
