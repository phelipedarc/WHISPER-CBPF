import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf import chi2_distance
from whisper_cbpf.distance import max_abs_z_distance


def test_chi2_zero_when_equal():
    obs = np.array([1.0, 2.0, 3.0])
    err = np.array([0.1, 0.1, 0.1])
    assert chi2_distance(obs, err, obs) == 0.0


def test_chi2_value():
    obs = np.array([1.0, 2.0])
    sim = np.array([1.1, 1.8])
    err = np.array([0.1, 0.2])
    expected = ((1.0 - 1.1) / 0.1) ** 2 + ((2.0 - 1.8) / 0.2) ** 2
    assert np.isclose(chi2_distance(obs, err, sim), expected)


# --------------------------------------------------------------------------- max_abs_z
def test_max_abs_z_is_the_worst_point_in_its_own_sigma():
    obs = np.array([20.0, 20.5, 21.0, 22.0])
    err = np.array([0.1, 0.2, 0.5, 0.05])
    sim = np.array([20.3, 20.5, 19.0, 22.01])
    z = np.abs(obs - sim) / err                         # 3, 0, 4, 0.2
    assert max_abs_z_distance(obs, err, sim) == pytest.approx(z.max())
    assert max_abs_z_distance(obs, err, obs) == 0.0
    # the sign of the residual does not matter, and a point's own error sets its scale: 4 sigma
    # on the 0.5 mag point beats 3 sigma on the 0.1 mag one although its residual is 2 mag vs 0.3
    assert max_abs_z_distance(obs, err, sim) == pytest.approx(4.0)


def test_max_abs_z_rejects_a_draw_on_one_bad_point_that_chi2_forgives():
    """The rule differs from chi2 exactly where the demos needed it: 29 perfect points and one
    6-sigma miss has chi2 = 36 (unremarkable for 30 points) but max |z| = 6 (outside 5 sigma)."""
    obs = np.zeros(30)
    err = np.ones(30)
    sim = np.zeros(30)
    sim[17] = 6.0
    assert chi2_distance(obs, err, sim) == pytest.approx(36.0)
    assert max_abs_z_distance(obs, err, sim) == pytest.approx(6.0)


def test_max_abs_z_non_finite_simulation_is_infinitely_far():
    obs, err = np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1])
    assert max_abs_z_distance(obs, err, np.array([1.0, np.nan, 3.0])) == np.inf
    assert max_abs_z_distance(obs, err, np.array([1.0, np.inf, 3.0])) == np.inf
    assert max_abs_z_distance([], [], []) == np.inf
    # inf, not NaN: it never satisfies a threshold, and a low acceptance quantile stays finite
    # (a single NaN would make np.quantile NaN at every level, and then nothing is accepted)
    ds = [max_abs_z_distance(obs, err, s) for s in (obs, obs + 0.1, np.full(3, np.nan))]
    assert ds[2] == np.inf and not ds[2] <= 1e300
    assert np.quantile(ds, 0.25) == pytest.approx(0.5)
    assert np.isnan(np.quantile([0.0, 1.0, np.nan], 0.25))


def test_max_abs_z_is_registered_on_both_backends():
    names = wp.list_distances()
    assert "max_abs_z" in names and "max_abs_z_jax" in names
    assert wp.get_distance("max_abs_z") is max_abs_z_distance
    assert wp.get_distance("MAX_ABS_Z") is max_abs_z_distance


def test_max_abs_z_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from whisper_cbpf.distance._jax import jnp_distance

    rng = np.random.default_rng(3)
    obs = rng.normal(20.0, 1.0, (200, 25))
    err = rng.uniform(0.02, 0.3, (200, 25))
    sim = obs + err * rng.normal(0.0, 3.0, (200, 25))
    sim[7, 4] = np.nan
    f = jax.vmap(jnp_distance("max_abs_z"))
    got = np.asarray(f(jnp.asarray(obs), jnp.asarray(err), jnp.asarray(sim)), float)
    want = np.array([max_abs_z_distance(o, e, s) for o, e, s in zip(obs, err, sim)])
    assert got[7] == np.inf and want[7] == np.inf
    # float32 rounds a 20 mag value to 2e-6 mag, which is up to ~1e-4 sigma at a 0.02 mag error
    rtol = 1e-12 if jax.config.jax_enable_x64 else 1e-4
    np.testing.assert_allclose(got, want, rtol=rtol)
    # the *_jax registry name resolves to the same function
    g = wp.get_distance("max_abs_z_jax")
    assert float(g(jnp.asarray(obs[0]), jnp.asarray(err[0]), jnp.asarray(sim[0]))) == \
        pytest.approx(want[0], rel=rtol)
