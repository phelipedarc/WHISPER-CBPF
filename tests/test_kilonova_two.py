"""Two-component kilonova: the flux-sum identity, and the traps that identity does not catch."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.jax import kilonova as kn      # noqa: E402
from whisper_cbpf.models.jax import kilonova_two as kn2  # noqa: E402

DAY = 86400.0
P = dict(mej_1=0.01, vej_1=0.25, temperature_floor_1=4000.0, kappa_1=2.0,
         mej_2=0.03, vej_2=0.15, temperature_floor_2=2500.0, kappa_2=10.0)


@pytest.fixture(scope="module")
def setup():
    t = np.geomspace(0.25, 21.0, 40)
    z, dl = 0.01, 1.34e26
    t_src = jnp.asarray(kn.source_time_s(t, z, 0.0))
    lam = np.geomspace(1000.0, 30000.0, 800)
    trans = np.zeros((2, lam.size))
    trans[0][(lam > 4000) & (lam < 5500)] = 1.0
    trans[1][(lam > 5500) & (lam < 7000)] = 1.0
    W, N = kn.ab_weights(lam, trans)
    return t_src, jnp.asarray(np.arange(40) % 2), W, N, jnp.asarray(lam), z, dl


def test_summed_magnitude_equals_magnitude_of_summed_flux(setup):
    """The load-bearing identity: fluxes add BEFORE the band integral and before the log."""
    t_src, bidx, W, N, lam, z, dl = setup
    got = np.asarray(kn2.two_component_magnitude(t_src, bidx, W, N, lam, z, dl, **P))
    m1 = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, z, dl,
                                    P["mej_1"], P["vej_1"], P["kappa_1"], P["temperature_floor_1"]))
    m2 = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, z, dl,
                                    P["mej_2"], P["vej_2"], P["kappa_2"], P["temperature_floor_2"]))
    ref = -2.5 * np.log10(10 ** (-0.4 * m1) + 10 ** (-0.4 * m2))
    assert np.max(np.abs(got - ref)) < 1e-4, np.max(np.abs(got - ref))


def test_summing_magnitudes_would_be_wrong(setup):
    """Control: the identity above is not vacuous -- averaging magnitudes gives a different answer."""
    t_src, bidx, W, N, lam, z, dl = setup
    got = np.asarray(kn2.two_component_magnitude(t_src, bidx, W, N, lam, z, dl, **P))
    m1 = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, z, dl,
                                    P["mej_1"], P["vej_1"], P["kappa_1"], P["temperature_floor_1"]))
    m2 = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, z, dl,
                                    P["mej_2"], P["vej_2"], P["kappa_2"], P["temperature_floor_2"]))
    assert np.max(np.abs(got - 0.5 * (m1 + m2))) > 0.1


def test_two_identical_components_are_exactly_one_brighter_by_0p7526(setup):
    """Doubling the flux is -2.5*log10(2) = -0.7526 mag. Catches a dropped or double-counted term."""
    t_src, bidx, W, N, lam, z, dl = setup
    q = dict(mej_1=0.02, vej_1=0.2, temperature_floor_1=3000.0, kappa_1=5.0,
             mej_2=0.02, vej_2=0.2, temperature_floor_2=3000.0, kappa_2=5.0)
    both = np.asarray(kn2.two_component_magnitude(t_src, bidx, W, N, lam, z, dl, **q))
    one = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, z, dl, 0.02, 0.2, 5.0, 3000.0))
    np.testing.assert_allclose(one - both, 2.5 * np.log10(2.0), atol=1e-4)


def test_flux_density_sums_and_is_positive(setup):
    t_src, _, _, _, _, z, dl = setup
    fd = np.asarray(kn2.two_component_flux_density(t_src, 5e14, z, dl, **P), dtype=np.float64)
    f1 = np.asarray(kn.flux_density_mjy(t_src, 5e14, z, dl, P["mej_1"], P["vej_1"],
                                        P["kappa_1"], P["temperature_floor_1"]), dtype=np.float64)
    f2 = np.asarray(kn.flux_density_mjy(t_src, 5e14, z, dl, P["mej_2"], P["vej_2"],
                                        P["kappa_2"], P["temperature_floor_2"]), dtype=np.float64)
    assert np.all(fd > 0)
    np.testing.assert_allclose(fd, f1 + f2, rtol=1e-5)


def test_gradients_finite_in_all_eight_parameters(setup):
    t_src, bidx, W, N, lam, z, dl = setup
    keys = kn2.PARAMETERS
    g = jax.grad(lambda v: jnp.sum(kn2.two_component_magnitude(
        t_src, bidx, W, N, lam, z, dl, **dict(zip(keys, v)))))(
        jnp.asarray([P[k] for k in keys]))
    g = np.asarray(g)
    assert np.all(np.isfinite(g)), dict(zip(keys, g))
    assert np.any(np.abs(g) > 0)


def test_horizon_uses_the_SHORT_component(setup):
    """The short-t_diff component enters redback's bad-quadrature regime first, so it sets the
    horizon -- using the max, or a flux weighting, would certify a curve that is already wrong."""
    td1 = kn2.t_diff_days(P["mej_1"], P["vej_1"], P["kappa_1"])
    td2 = kn2.t_diff_days(P["mej_2"], P["vej_2"], P["kappa_2"])
    h = kn2.validity_horizon_days(P["mej_1"], P["vej_1"], P["kappa_1"],
                                  P["mej_2"], P["vej_2"], P["kappa_2"])
    assert h == pytest.approx(2.66 * min(td1, td2))
    assert h < 2.66 * max(td1, td2)


def test_default_prior_is_redbacks_verbatim():
    b = {k: v.bounds for k, v in kn2.default_prior().distributions.items()}
    for c in ("1", "2"):        # redback gives BOTH components identical bounds
        assert b[f"mej_{c}"] == pytest.approx((1e-2, 0.03))
        assert b[f"vej_{c}"] == pytest.approx((0.1, 0.5))
        assert b[f"kappa_{c}"] == pytest.approx((1.0, 30.0))
        assert b[f"temperature_floor_{c}"] == pytest.approx((100.0, 6000.0))
