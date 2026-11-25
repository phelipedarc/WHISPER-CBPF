"""Where the ensemble samplers move their walkers, and how their starts are spread.

1. ``walker_coordinates="own"`` (the default of ``emcee_jax``, ``mcmc`` and ``fit_batch``) moves a
   LogUniform parameter's walkers in its natural log, with the log-Jacobian added, so the
   posterior is the same. A flat LogUniform direction is then flat for the walkers too, and the
   stuck-walker check no longer reads its ``-ln x`` prior density as "stuck".
2. The default start (``init="prior_scan"``) spreads its walkers even when the optimum is pressed
   against a constraint wall, and a start that does not span every parameter is refused with the
   way out, before emcee's own "Initial state has a large condition number".
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("emcee")

import whisper_cbpf as wp  # noqa: E402
import whisper_cbpf.models as M  # noqa: E402
from whisper_cbpf.priors import LogUniform, Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax import _diagnostics as dg  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _nuisance_model():
    """flux = a exp(-t / 10); b and c are LogUniform(1e-4, 1e4) and do not enter the flux, so the
    posterior of each is its prior, flat in log10 from -4 to 4."""
    def predict(p, t, bands=None):
        return p["a"] * np.exp(-np.asarray(t, dtype=float) / 10.0)

    def predict_jax(theta, t):
        return theta[0] * jnp.exp(-t / 10.0)

    prior = Prior({"a": Uniform(0.1, 10.0), "b": LogUniform(1e-4, 1e4),
                   "c": LogUniform(1e-4, 1e4)})
    return M.Model(name="nuisance_toy", predict=predict, parameters=["a", "b", "c"],
                   default_prior=prior, predict_jax=predict_jax)


def _lc(n=30, seed=0):
    t = np.linspace(0.5, 30.0, n)
    rng = np.random.default_rng(seed)
    flux = 3.0 * np.exp(-t / 10.0)
    return wp.LightCurve(time=t, band=["r"] * n, flux=flux + rng.normal(0.0, 0.05, n),
                         flux_err=np.full(n, 0.05))


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **kw)


# --- 1. the walker coordinates -----------------------------------------------------------------
def test_walker_density_is_the_posterior_plus_the_log_jacobian():
    prior = Prior({"a": Uniform(0.0, 1.0), "b": LogUniform(1e-3, 1e3)})
    is_log = dg.walker_log_columns(prior, ["a", "b"], "own", "t")
    assert is_log.tolist() == [False, True]
    x = np.array([[0.3, 20.0], [0.7, 2e-3]])
    y = dg.to_walker(x, is_log)
    np.testing.assert_allclose(y[:, 1], np.log(x[:, 1]))
    np.testing.assert_allclose(dg.from_walker(y, is_log), x)

    def density(th):
        return -0.5 * th[0] ** 2 - jnp.log(th[1])

    walker = dg.jax_walker_density(density, is_log)
    for xi, yi in zip(x, y):
        assert float(walker(jnp.asarray(yi))) == pytest.approx(
            float(density(jnp.asarray(xi))) + yi[1], rel=1e-12)
    assert not dg.walker_log_columns(prior, ["a", "b"], "linear", "t").any()
    with pytest.raises(ValueError, match="walker_coordinates must be one of"):
        dg.walker_log_columns(prior, ["a", "b"], "log", "emcee_jax")


def test_a_flat_log_uniform_direction_is_not_called_stuck():
    """The walkers spread over b and c's whole LogUniform range, as their posterior (the prior)
    says. In linear coordinates the -ln b prior density put walkers "more than 10 nats below the
    best walker" and 8-26 of 32 were flagged stuck on this model."""
    res = _quiet(wp.fit, _lc(), _nuisance_model(), sampler="emcee_jax", space="flux",
                 nwalkers=32, nsteps=3000, burnin=1000, seed=0)
    assert res.info["walker_coordinates"] == {"coordinates": "own", "log": ["b", "c"]}
    assert res.info["stuck_walkers"] == []
    for k in ("b", "c"):
        lg = np.log10(res.samples[k].to_numpy())
        assert abs(np.mean(lg)) < 0.6 and 1.8 < np.std(lg) < 2.8, (k, np.mean(lg), np.std(lg))
    assert abs(res.summary["a"]["median"] - 3.0) < 0.2


def test_fit_batch_moves_the_same_coordinates():
    from whisper_cbpf.samplers.jax.batch import fit_batch

    res = _quiet(fit_batch, [_lc(seed=1)], _nuisance_model(), nwalkers=32, nsteps=3000,
                 burnin=1000, seed=0, metrics=False)[0]
    assert res.info["walker_coordinates"] == {"coordinates": "own", "log": ["b", "c"]}
    assert res.info["stuck_walkers"] == []
    lg = np.log10(res.samples["b"].to_numpy())
    assert abs(np.mean(lg)) < 0.6 and 1.8 < np.std(lg) < 2.8


# --- 2. the starts -----------------------------------------------------------------------------
def test_a_start_that_does_not_span_every_parameter_is_refused():
    rng = np.random.default_rng(0)
    ok = rng.normal(size=(16, 3))
    assert dg.refuse_collapsed(ok, "emcee_jax", "init='prior_scan'") is ok
    one_point = np.tile([1.0, 2.0, 3.0], (16, 1))
    collinear = np.outer(rng.normal(size=16), [1.0, 2.0, 3.0]) + [1.0, 2.0, 3.0]
    for bad in (one_point, collinear):
        with pytest.raises(ValueError, match=r"do not span all 3 parameters.*init='prior'"):
            dg.refuse_collapsed(bad, "emcee_jax", "init='prior_scan'")


def _walled():
    """A Gaussian peak at (0.3, 0.3, 0.3, 0.3), sd 0.05, behind the wall sum(x) < 1: the optimum
    is pressed against the wall, as a magnetar's is against its rotational-energy bound."""
    names = ["x0", "x1", "x2", "x3"]
    prior = Prior({n: Uniform(-1.0, 1.0) for n in names})

    def score(th):
        th = np.atleast_2d(np.asarray(th, dtype=float))
        out = -0.5 * np.sum(((th - 0.3) / 0.05) ** 2, axis=1)
        return np.where(th.sum(axis=1) < 1.0, out, -np.inf)

    return names, prior, score


def test_a_coordinate_walled_on_one_side_gets_a_short_step():
    """The finite-difference curvature cannot be read across a wall; the step is shrunk there
    instead of set to the full MAX_SPREAD_U, which threw every start through the wall."""
    names, prior, score = _walled()
    centre = np.full((1, 4), 0.25)                         # on the wall: sum = 1 - 0 (just inside)
    centre[0, 0] -= 1e-9
    sd = dg._conditional_sd(score, centre, prior, names)
    assert np.all(sd < 0.1 * dg.MAX_SPREAD_U), sd


def test_the_default_start_spreads_walkers_around_a_walled_optimum():
    """One climbed point (``n_climb=1``, as when a single climb reaches the best basin): every
    walker starts from it, and each must find its own spot on the allowed side of the wall."""
    names, prior, score = _walled()
    sc = _quiet(dg.scan_starts, None, prior, names, 32, 0, np.float64, n_climb=1, score=score)
    x = sc["starts"]
    assert np.all(np.isfinite(sc["start_scores"])) and np.all(x.sum(axis=1) < 1.0)
    assert len(np.unique(x, axis=0)) == 32
    dg.refuse_collapsed(x, "emcee_jax", "init='prior_scan'")
