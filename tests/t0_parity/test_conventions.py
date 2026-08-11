"""T0.4 convention probes: cheap, exact identities every photometric model must satisfy.

These do not need redback or goldens-as-reference (the filter set golden is reused as a
convenient set of REAL bandpasses). Each probe pins one convention:

  * a flat AB source (F_nu = 3631 Jy at every frequency) is 0.000 AB in EVERY band --
    the definition of the AB system, and the only test that catches a wrong zero point,
    a wrong photon-counting weight, or a mis-normalised band integral all at once;
  * doubling d_L dims by exactly +5 log10(2) mag (and scales flux by exactly 1/4);
  * at z = 0 the TDE (1+z) flux factor must be a bitwise no-op;
  * magnitude at d_L = 10 pc IS the absolute magnitude: m(d_L) - m(10 pc) = DM(d_L);
  * shifting the kilonova explosion epoch by delta translates the observer-frame light
    curve by exactly delta -- same numbers, shifted axis, bit for bit.

The remaining T0.4 probe -- the (1+z) DIRECTION of the TDE dilation factor, flux ratio
dilation=True / dilation=False == (1+z) exactly at every epoch -- already exists as
``tests/test_tde_vs_redback.py::test_dilation_is_exactly_one_plus_z`` and is deliberately
NOT duplicated here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)      # before any array: the TDE engine needs it
import jax.numpy as jnp  # noqa: E402

from whisper_cbpf.models.jax import kilonova as kn  # noqa: E402
from whisper_cbpf.models.jax import tde as T  # noqa: E402

DAY = 86400.0
GOLDENS = Path(__file__).resolve().parent.parent / "goldens"
TEN_PC_CM = 3.0856775814913673e19

KN_P = (0.01, 0.2, 1.0, 4000.0)                    # canonical kilonova draw
TDE_P = (1.0, 1.0, 0.05, 0.1, 1.0)                 # canonical TDE draw


@pytest.fixture(scope="module")
def filters():
    p = GOLDENS / "filterset_t0.npz"
    if not p.exists():
        pytest.fail(f"golden filter set missing: {p} -- run make_goldens.py")
    fs = np.load(p, allow_pickle=False)
    W, N = kn.ab_weights(fs["lam"], fs["trans"])
    return jnp.asarray(fs["lam"]), W, N


# ---------------------------------------------------------------- flat AB source
def _flat_ab_spectrum(temperature, r_photosphere, dl_cm, nu_source, redshift,
                      pref_const=None, dilation=True):
    """A source with F_nu = 3631 Jy everywhere, in the AB-zero-point units the band
    integral runs in (kilonova identity 8: f_ab = f_nu / AB_ZEROPOINT = 1.0 exactly)."""
    shape = jnp.broadcast_shapes(jnp.shape(temperature), jnp.shape(nu_source))
    return jnp.ones(shape, dtype=jnp.result_type(temperature))


def test_flat_ab_source_is_zero_mag_in_every_band_kilonova(filters, monkeypatch):
    """Kilonova path: patch the Planck spectrum to the flat AB source; the band integral,
    photon-counting weights, zero point and log must then return 0.000 AB exactly (up to
    one rounding of the num/norm quotient) in every band at every epoch."""
    lam, W, N = filters
    monkeypatch.setattr(kn, "_flux_nu", _flat_ab_spectrum)
    n_band = W.shape[0]
    t_src = jnp.asarray(np.geomspace(1.0, 10.0, n_band) * DAY)
    bidx = jnp.arange(n_band)
    mag = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, 0.0, 1e26, *KN_P))
    assert np.all(np.abs(mag) < 1e-9), mag


def test_flat_ab_source_is_zero_mag_in_every_band_tde(filters, monkeypatch):
    """Same probe through the TDE path, which has its own num/den lines (tde.ab_magnitude
    does not share kilonova.ab_magnitude's body, only its weights)."""
    lam, W, N = filters
    monkeypatch.setattr(T, "_flux_nu_tde", _flat_ab_spectrum)
    n_band = W.shape[0]
    temp = jnp.full(n_band, 3.0e4)
    rad = jnp.full(n_band, 1.0e14)
    bidx = jnp.arange(n_band)
    mag = np.asarray(T.ab_magnitude(temp, rad, bidx, W, N, lam, 0.0, 1e26))
    assert np.all(np.abs(mag) < 1e-9), mag


# ---------------------------------------------------------------- distance scaling
def test_doubling_dl_is_exactly_five_log10_two_kilonova(filters):
    lam, W, N = filters
    t_src = jnp.asarray(np.geomspace(0.8, 12.0, 12) * DAY)
    bidx = jnp.asarray(np.arange(12) % W.shape[0])
    dl = 1.34e26
    m1 = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, 0.01, dl, *KN_P))
    m2 = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, 0.01, 2.0 * dl, *KN_P))
    ok = m1 < 39.0                                   # off the mag floor
    assert ok.all(), "pick brighter epochs"
    assert np.allclose((m2 - m1)[ok], 5.0 * np.log10(2.0), rtol=0, atol=1e-10), m2 - m1
    # and in flux: exactly 1/4, bit for bit (dl enters squared, scaling by 2 is exact)
    f1 = np.asarray(kn.flux_density_mjy(t_src, 5e14, 0.01, dl, *KN_P))
    f2 = np.asarray(kn.flux_density_mjy(t_src, 5e14, 0.01, 2.0 * dl, *KN_P))
    assert np.allclose(f2 * 4.0, f1, rtol=1e-14, atol=0.0)


def test_doubling_dl_is_exactly_five_log10_two_tde(filters):
    lam, W, N = filters
    t = jnp.asarray(np.geomspace(5.0, 300.0, 12))
    bidx = jnp.asarray(np.arange(12) % W.shape[0])
    dl = 7e26
    m1 = np.asarray(T.cooling_envelope_ab_magnitude(t, bidx, W, N, lam, 0.05, dl,
                                                    *TDE_P, n_time=500))
    m2 = np.asarray(T.cooling_envelope_ab_magnitude(t, bidx, W, N, lam, 0.05, 2.0 * dl,
                                                    *TDE_P, n_time=500))
    ok = m1 < 39.0
    assert ok.sum() >= 10, "pick brighter epochs"
    assert np.allclose((m2 - m1)[ok], 5.0 * np.log10(2.0), rtol=0, atol=1e-10), m2 - m1
    f1 = np.asarray(T.cooling_envelope_flux_density(t, 6e14, 0.05, dl, *TDE_P, n_time=500))
    f2 = np.asarray(T.cooling_envelope_flux_density(t, 6e14, 0.05, 2.0 * dl, *TDE_P,
                                                    n_time=500))
    assert np.allclose(f2 * 4.0, f1, rtol=1e-14, atol=0.0)


# ---------------------------------------------------------------- z = 0 dilation no-op
def test_tde_dilation_is_a_bitwise_noop_at_z_zero():
    """At z = 0 the (1+z) factor is a multiply by 1.0: dilation=True and dilation=False
    must return the SAME bits. (The (1+z) direction itself -- ratio == 1+z exactly at
    z > 0 -- is test_tde_vs_redback.py::test_dilation_is_exactly_one_plus_z.)"""
    t = jnp.asarray(np.geomspace(1.0, 300.0, 16))
    a = np.asarray(T.cooling_envelope_flux_density(t, 6e14, 0.0, 1e27, *TDE_P,
                                                   n_time=500, dilation=True))
    b = np.asarray(T.cooling_envelope_flux_density(t, 6e14, 0.0, 1e27, *TDE_P,
                                                   n_time=500, dilation=False))
    assert np.array_equal(a, b)
    assert np.any(a > 0)                             # not vacuously equal on zeros


# ---------------------------------------------------------------- absolute magnitude
def test_magnitude_at_ten_pc_is_the_absolute_magnitude_kilonova(filters):
    """M = m(10 pc) at z ~ 0: m(d_L) - m(10 pc) must equal the distance modulus
    5 log10(d_L / 10 pc), which pins the distance NORMALISATION, not just its slope."""
    lam, W, N = filters
    t_src = jnp.asarray(np.geomspace(0.8, 12.0, 8) * DAY)
    bidx = jnp.asarray(np.arange(8) % W.shape[0])
    dl = 1.0e26
    m_abs = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, 0.0, TEN_PC_CM, *KN_P))
    m_dl = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam, 0.0, dl, *KN_P))
    dm = 5.0 * np.log10(dl / TEN_PC_CM)
    ok = m_dl < 39.0
    assert ok.all()
    assert np.allclose((m_dl - m_abs)[ok], dm, rtol=0, atol=1e-9), (m_dl - m_abs) - dm


def test_magnitude_at_ten_pc_is_the_absolute_magnitude_tde(filters):
    lam, W, N = filters
    t = jnp.asarray(np.geomspace(5.0, 300.0, 8))
    bidx = jnp.asarray(np.arange(8) % W.shape[0])
    dl = 1.0e27
    m_abs = np.asarray(T.cooling_envelope_ab_magnitude(t, bidx, W, N, lam, 0.0,
                                                       TEN_PC_CM, *TDE_P, n_time=500))
    m_dl = np.asarray(T.cooling_envelope_ab_magnitude(t, bidx, W, N, lam, 0.0, dl,
                                                      *TDE_P, n_time=500))
    dm = 5.0 * np.log10(dl / TEN_PC_CM)
    ok = m_dl < 39.0
    assert ok.sum() >= 6
    assert np.allclose((m_dl - m_abs)[ok], dm, rtol=0, atol=1e-9), (m_dl - m_abs) - dm


# ---------------------------------------------------------------- t_exp translation
def test_kilonova_t_exp_shift_translates_the_light_curve_exactly(filters):
    """flux(t_obs + delta | t_exp = delta) == flux(t_obs | t_exp = 0), bit for bit:
    source_time_s is the affine map (t_obs - t_exp) * 86400 / (1 + z), and delta and the
    epochs are chosen exactly representable so the subtraction is exact."""
    lam, W, N = filters
    delta = 4.0
    t_obs = np.array([0.75, 1.5, 2.25, 4.0, 6.5, 9.75, 14.0, 20.5])   # exact binary
    z, dl = 0.01, 1.34e26
    bidx = jnp.asarray(np.arange(t_obs.size) % W.shape[0])

    f_ref = np.asarray(kn.flux_density_mjy(kn.source_time_s(t_obs, z, 0.0),
                                           5e14, z, dl, *KN_P))
    f_shift = np.asarray(kn.flux_density_mjy(kn.source_time_s(t_obs + delta, z, delta),
                                             5e14, z, dl, *KN_P))
    assert np.array_equal(f_ref, f_shift)
    assert np.all(f_ref > 0)

    m_ref = np.asarray(kn.ab_magnitude(kn.source_time_s(t_obs, z, 0.0),
                                       bidx, W, N, lam, z, dl, *KN_P))
    m_shift = np.asarray(kn.ab_magnitude(kn.source_time_s(t_obs + delta, z, delta),
                                         bidx, W, N, lam, z, dl, *KN_P))
    assert np.array_equal(m_ref, m_shift)

    # and epochs BEFORE the shifted explosion are exactly zero flux / exactly the floor
    early = np.asarray(kn.flux_density_mjy(kn.source_time_s(t_obs, z, 100.0),
                                           5e14, z, dl, *KN_P))
    assert np.all(early == 0.0)
