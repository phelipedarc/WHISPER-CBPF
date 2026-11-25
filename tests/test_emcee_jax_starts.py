"""``init=`` for ``emcee_jax`` / ``fit_emcee_numpy``: every form, in each prior's own coordinate.

The walkers used to start uniformly in LINEAR coordinates over the prior box, with no way to give a
start. That is not the prior
for a LogUniform parameter, and walkers drawn where the model is dark, or in a separate optimum,
stayed there: stuck walkers in 78 of 100 bump fits, 54 of 90 SN2026jkr fits. The end-to-end
regression is ``tests/test_nuts_gpu_known_answer.py::test_t4_...``; these pin the forms themselves
on the resolver, without running emcee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
pytest.importorskip("emcee")

import jax.numpy as jnp  # noqa: E402

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.samplers.base import SamplerResult  # noqa: E402
from whisper_cbpf.samplers.jax import emcee_jax as ej  # noqa: E402

PRIOR = wp.Prior({"T": wp.LogUniform(100.0, 6000.0), "x": wp.Uniform(0.0, 10.0)})
NAMES = ["T", "x"]


def _batch(fn):
    return lambda th: np.array([fn(t) for t in np.asarray(th)], dtype=float)


FLAT = _batch(lambda t: 0.0)


def test_prior_start_is_log_uniform_in_a_log_uniform_parameter():
    """Half of LogUniform(100, 6000)'s mass is below its geometric midpoint, 774.6. The old "box"
    start put half below the arithmetic one, 3050, i.e. only ~10 % below 774.6."""
    p0, label, _ = ej._walker_starts("prior", PRIOR, NAMES, 400, 0, FLAT, "t")
    assert label == "prior" and p0.shape == (400, 2)
    assert 0.4 < np.mean(p0[:, 0] < np.sqrt(100.0 * 6000.0)) < 0.6
    box, label, _ = ej._walker_starts("box", PRIOR, NAMES, 400, 0, FLAT, "t")
    assert label == "box" and np.mean(box[:, 0] < np.sqrt(100.0 * 6000.0)) < 0.2


def test_starts_at_a_non_finite_density_are_redrawn():
    """Any walker whose density is -inf (here: x < 5) is redrawn, not started there."""
    dens = _batch(lambda t: 0.0 if t[1] >= 5.0 else -np.inf)
    p0, _, _ = ej._walker_starts("prior", PRIOR, NAMES, 64, 1, dens, "t")
    assert np.all(p0[:, 1] >= 5.0)


def _double_well(t):
    """A deeper well at x = 8 (+30 nats) and a shallower one at x = 2, each 0.3 wide."""
    return float(np.logaddexp(-0.5 * ((t[1] - 2.0) / 0.3) ** 2,
                              30.0 - 0.5 * ((t[1] - 8.0) / 0.3) ** 2))


def test_default_numpy_start_climbs_into_the_best_basin_without_a_gradient():
    """Without a JAX density (the numpy arm, CPU ``mcmc``) the default start climbs too, by the
    derivative-free ``climb_numpy``: every walker starts in the deeper well, at distinct points."""
    dens = _batch(_double_well)
    p0, label, detail = ej._walker_starts(None, PRIOR, NAMES, 16, 0, dens, "t")
    assert label == "prior_scan" and detail["n_draws"] == 1000 and detail["n_climbed"] == 32
    assert detail["climb_error"] is None
    assert np.all(np.abs(p0[:, 1] - 8.0) < 1.0), p0[:, 1]
    assert len({tuple(r) for r in p0}) == 16


def test_default_jax_start_climbs_into_the_best_basin():
    """With a JAX density the default climbs the best draws and keeps those reaching the best
    basin (``_diagnostics.scan_starts``): a double well in x whose deeper well is at 8."""
    def scalar(th):
        return jnp.logaddexp(-0.5 * ((th[1] - 2.0) / 0.3) ** 2,
                             30.0 - 0.5 * ((th[1] - 8.0) / 0.3) ** 2)
    dens = _batch(lambda t: float(scalar(jnp.asarray(t))))
    p0, label, detail = ej._walker_starts(None, PRIOR, NAMES, 8, 0, dens, "t",
                                          scalar_density=scalar, dtype=jnp.float32)
    assert label == "prior_scan" and detail["n_climbed"] == 32
    assert np.all(np.abs(p0[:, 1] - 8.0) < 1.0), p0[:, 1]
    assert len({tuple(r) for r in p0}) == 8                # distinct starts, not one point


def test_a_point_or_a_point_and_scale_gives_a_ball_in_the_priors_own_coordinate():
    p0, label, _ = ej._walker_starts({"T": 800.0, "x": 3.0}, PRIOR, NAMES, 32, 0, FLAT, "t")
    assert label == "ball"
    assert np.allclose(np.median(np.log(p0[:, 0])), np.log(800.0), atol=0.05)
    wide, _, _ = ej._walker_starts(([800.0, 3.0], 0.05), PRIOR, NAMES, 32, 0, FLAT, "t")
    assert wide[:, 1].std() > 5 * p0[:, 1].std()
    assert np.all((wide[:, 1] > 0.0) & (wide[:, 1] < 10.0))            # clipped into the box


def test_per_walker_starts_are_taken_as_given_and_checked():
    given = np.column_stack([np.geomspace(200, 5000, 8), np.linspace(1, 9, 8)])
    p0, label, _ = ej._walker_starts(given, PRIOR, NAMES, 8, 0, FLAT, "t")
    assert label == "per_walker" and np.array_equal(p0, given)
    with pytest.raises(ValueError, match="strictly inside"):
        ej._walker_starts(np.vstack([given[:7], [[50.0, 5.0]]]), PRIOR, NAMES, 8, 0, FLAT, "t")
    with pytest.raises(ValueError, match="shape"):
        ej._walker_starts(given[:6], PRIOR, NAMES, 8, 0, FLAT, "t")
    with pytest.raises(ValueError, match="-inf or NaN"):
        ej._walker_starts(given, PRIOR, NAMES, 8, 0, _batch(lambda t: -np.inf), "t")


def test_a_previous_result_seeds_distinct_walkers():
    rows = pd.DataFrame({"T": np.geomspace(200, 5000, 50), "x": np.linspace(1, 9, 50)})
    prev = SamplerResult(sampler="mcmc", model="m", parameters=NAMES, samples=rows, summary={},
                         best_params={}, n_data=1, n_params=2, runtime_s=0.0)
    p0, label, detail = ej._walker_starts(prev, PRIOR, NAMES, 16, 0, FLAT, "t")
    assert label == "result" and detail["mode"] == "draws" and len({tuple(r) for r in p0}) == 16
    assert {tuple(r) for r in p0} <= {tuple(r) for r in rows.to_numpy()}


def test_an_unknown_name_is_refused_by_listing_the_forms():
    with pytest.raises(ValueError, match="prior_scan"):
        ej._walker_starts("uniform", PRIOR, NAMES, 8, 0, FLAT, "t")


def test_a_density_jax_cannot_differentiate_falls_back_to_the_raw_scan():
    """emcee needs no gradient, so its density may call back into numpy, which JAX cannot
    differentiate. The climb then gives way to the best raw draws, and the record says why."""
    import jax

    def scalar(th):
        x = jax.pure_callback(lambda v: np.asarray(v, dtype=v.dtype),
                              jax.ShapeDtypeStruct((), th.dtype), th[1],
                              vmap_method="sequential")
        return -0.5 * ((x - 7.0) / 0.5) ** 2

    dens = _batch(lambda t: float(scalar(jnp.asarray(t))))
    p0, label, detail = ej._walker_starts(None, PRIOR, NAMES, 8, 0, dens, "t",
                                          scalar_density=scalar, dtype=jnp.float32)
    assert label == "prior_scan" and detail["climb_error"]
    # the spread stops at 10 nats below the start's centre: |x - 7| <= 0.5 * sqrt(20) = 2.24
    assert np.all(np.abs(p0[:, 1] - 7.0) < 2.3)
