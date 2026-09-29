"""The supernova port's central claims, as tests.

There are five, and they are not the same claim:

1. **It reproduces redback's engines exactly.** All nine engines and all twelve bolometric
   models are closed-form arithmetic, so "close enough" is not the bar -- they should agree to
   float64 round-off. Verified against redback where importable, and against independent NumPy
   transcriptions (``_ref_*`` below) always. The transcriptions are deliberately unclever, so a
   disagreement points at the port rather than at them.

2. **It reproduces redback's SEDs**, including the two that are new here (CutoffBlackbody, Line)
   and the one that is only reachable in the radio (Synchrotron). These are the parts with no
   sibling in :mod:`whisper_cbpf.models.jax.tde`, so they carry the most new code.

3. **It is differentiable everywhere the prior reaches**, including at ``theta_pb = 0``, which
   redback's own ``Uniform(0, 3.14/2)`` includes and where ``sin(theta)**-2`` is infinite.

4. **The three documented deviations are the ONLY deviations**, and each is bounded by a number
   rather than by a promise: CHANGE 4 (the CSM interpolation), CHANGE 3 (the exponential
   powerlaw's first dense interval) and CHANGE 7b (the fallback models' missing diffusion).
   Each has a test that pins the size, so a future edit that widens one fails here.

5. **float32 fails loudly.** The engines are 1e43-1e46 erg/s in cgs, past float32's 3.4e38, and
   the failure mode without a guard is a whole light curve at ``mag_floor`` with finite
   gradients -- the exact "plausible wrong answer" the guard exists to prevent (CHANGE 8).

FLOAT64. Required, and the fixture enables it for this module and restores the previous setting
afterwards, so the float32 kilonova tests in the same session are unaffected.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.jax import supernova as S  # noqa: E402

DAY = 86400.0
MSUN = 1.988409870698051e33
CLIGHT = 2.99792458e10
SIGMA_SB = 5.6703744191844314e-5

#: Observer-frame epochs and the source-frame grid every test shares. Starts at 0.5 d so the
#: earliest epoch sits inside the first dense interval (0.29 d) of redback <= 1.15's linear grid,
#: which is where CHANGE 3 lives; ends at 200 d so the nickel tail is covered.
Z = 0.05
T_OBS = np.geomspace(0.5, 200.0, 60)
T_SRC = T_OBS / (1.0 + Z)

BASE = dict(kappa=0.1, kappa_gamma=0.03, mej=2.0, vej=1e4, temperature_floor=4000.0)

#: One value per parameter of every model, inside redback's shipped priors.
DEMO = dict(f_nickel=0.1, mej=2.0, vej=1e4, kappa=0.1, kappa_gamma=0.03,
            temperature_floor=4000.0, log10_mass=-1.0, log10_radius=13.0, log10_energy=47.0,
            nn=10.0, delta=1.1, p0=2.0, bp=1.0, mass_ns=1.4, theta_pb=1.0, csm_mass=1.0,
            v_min=1e4, beta=0.45, shell_radius=1.0, shell_width_ratio=0.1, lbol_0=1e43,
            alpha_1=2.0, alpha_2=1.0, tpeak_d=10.0, logl1=54.0, tr=1.0, l0=1e45, tsd=20.0,
            pp=3.0)


@pytest.fixture(scope="module", autouse=True)
def _x64():
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


@pytest.fixture(scope="module")
def grid(_x64):
    return S.build_sn_grid(T_SRC)


@pytest.fixture(scope="module")
def bands():
    """Two flat top-hat filters. Real bandpasses need sncosmo; the SED maths does not."""
    lam = np.geomspace(1000.0, 30000.0, 2000)
    trans = np.zeros((2, lam.size))
    trans[0][(lam > 4000) & (lam < 5500)] = 1.0
    trans[1][(lam > 5500) & (lam < 7000)] = 1.0
    w, n = S.ab_weights(lam, trans)
    return jnp.asarray(lam), w, n, jnp.asarray(np.arange(T_SRC.size) % 2)


def _maxrel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.max(np.abs(b))
    return np.max(np.abs(a - b)) / (scale if scale > 0 else 1.0)


# =======================================================================================
# independent NumPy references -- literal transcriptions of redback, no JAX involved
# =======================================================================================

def _ref_diffusion(time, dense_times, luminosity, kappa, kappa_gamma, mej, vej):
    """redback ``interaction_processes.Diffusion.convert_input_luminosity``, verbatim."""
    from scipy.interpolate import interp1d
    dc = 2.0 * MSUN / (13.7 * CLIGHT * 1e5)
    tc = 3.0 * MSUN / (4 * np.pi * 1e10)
    tau = np.sqrt(dc * kappa * mej / vej) / DAY
    trap = (tc * kappa_gamma * mej / vej ** 2) / DAY ** 2
    tb = max(0.0, np.min(dense_times))
    li = interp1d(dense_times, luminosity, copy=False, assume_sorted=True)
    ut = np.unique(time[(time >= tb) & (time <= dense_times[-1])])
    lu = len(ut)
    lsp = np.logspace(np.log10(tau / dense_times[-1]) - 3, 0, 50)
    xm = np.unique(np.concatenate((lsp, 1 - lsp)))
    it = np.clip(tb + (ut.reshape(lu, 1) - tb) * xm, tb, dense_times[-1])
    it2 = it[:, -1] ** 2
    with np.errstate(all="ignore"):
        ia = li(it) * it * np.exp((it ** 2 - it2.reshape(lu, 1)) / tau ** 2)
    ia[np.isnan(ia)] = 0.0
    ul = np.trapezoid(ia, it, axis=1) * -2.0 * np.expm1(-trap / it2) / tau ** 2
    return ul[np.searchsorted(ut, time)]


def _ref_dense():
    """redback 1.20's dense grid (``supernova_models.py:961``). <= 1.15 had
    ``np.linspace(0, time[-1]+100, 1000)``; the CHANGE 3 tests below build that one themselves."""
    return np.geomspace(1e-5, T_SRC[-1] + 100.0, 1000)


def _ref_nico(t, f_ni, mej):
    return f_ni * mej * (6.45e43 * np.exp(-t / 8.8) + 1.45e43 * np.exp(-t / 111.3))


def _ref_magnetar(t_s, p0, bp, mns, th, convention="1.15"):
    erot = 2.6e52 * (mns / 1.4) ** 1.5 * p0 ** -2
    tp = 1.3e5 * bp ** -2 * p0 ** 2 * (mns / 1.4) ** 1.5 * np.sin(th) ** -2
    if convention == "1.12":
        return erot / tp / (1. + t_s / tp) ** 2
    return 2. * erot / tp / (1. + 2. * t_s / tp) ** 2


def _ref_shock_cooling(t, mass, radius, energy, nn=10.0, delta=1.1):
    kk = (nn - 3) * (3 - delta) / (4 * np.pi * (nn - delta))
    kap = 0.2
    m = mass * MSUN
    vt = (((nn - 5) * (5 - delta) / ((nn - 3) * (3 - delta))) * (2 * energy / m)) ** 0.5
    td = ((3 * kap * kk * m) / ((nn - 1) * vt * CLIGHT)) ** 0.5
    pre = np.pi * (nn - 1) / (3 * (nn - 5)) * CLIGHT * radius * vt ** 2 / kap
    lb = np.zeros(len(t))
    lb[t < td] = (pre * np.power(td / t, 4 / (nn - 2)))[t < td]
    lb[t >= td] = (pre * np.exp(-0.5 * (t * t / td / td - 1)))[t >= td]
    tph = np.sqrt(3 * kap * kk * m / (2 * (nn - 1) * vt * vt))
    rp = (np.power(tph / t, 2 / (nn - 1)) * vt * t
          + np.power((delta - 1) / (nn - 1) * ((t / td) ** 2 - 1) + 1, -1 / (delta + 1)) * vt * t)
    return lb, rp, np.power(lb / (4 * np.pi * rp ** 2) / SIGMA_SB, 0.25)


def _ref_csm_sbo(t, csm_m, v_min, beta, kappa, sr, swr):
    m = csm_m * MSUN
    v0 = v_min * 1e5
    e0 = 0.5 * m * v0 ** 2
    vel = v0 / beta
    r = sr * 1e14
    w = swr * r
    tdyn, tsh, tt = r / vel, w / vel, t * DAY
    tda = (3 * kappa * m / (4 * np.pi * CLIGHT * vel)) ** 0.5
    t1 = ((tdyn + tsh + tt) ** 3 - (tdyn + beta * tt) ** 3) ** (2 / 3)
    t2 = ((tdyn + tsh) ** 3 - tdyn ** 3) ** (1 / 3)
    t3 = (1 + (1 - beta) * tt / tsh) ** (-3 * (tdyn / tda) ** 2
                                         * ((1 - beta - beta * tsh / tdyn) ** 2) / (1 - beta) ** 3)
    t4 = np.exp(-tt * ((1 - beta ** 3) * tt + (2 - 4 * beta * (beta + 1)) * tsh
                       + 6 * (1 - beta ** 2) * tdyn) / (2 * (1 - beta) ** 2 * tda ** 2))
    lbol = e0 * t1 / (tda ** 2 * (tsh + (1 - beta) * tt) ** 2) * t2 * t3 * t4
    vol = 4. / 3. * np.pi * vel ** 3 * ((tdyn + tsh + tt) ** 3 - (tdyn + beta * tt) ** 3)
    rph = vel * (tdyn + tsh + tt) - 2 * vol / (3 * kappa * m)
    return lbol, rph, (lbol / (4 * np.pi * rph ** 2 * SIGMA_SB)) ** 0.25


# =======================================================================================
# CLAIM 1: the engines and the bolometric models reproduce redback
# =======================================================================================

@pytest.mark.parametrize("f_ni,mej,kappa,kg,vej", [(0.1, 2.0, 0.1, 0.03, 1e4),
                                                   (0.5, 10.0, 0.2, 0.01, 5e3),
                                                   (1e-3, 1e-4, 2.0, 1e4, 1e5)])
def test_arnett_matches_reference(grid, f_ni, mej, kappa, kg, vej):
    dense = _ref_dense()
    ref = _ref_diffusion(T_SRC, dense, _ref_nico(dense, f_ni, mej), kappa, kg, mej, vej)
    ours = S.arnett_bolometric(grid, f_ni, mej, kappa, kg, vej)
    assert _maxrel(ours, ref) < 1e-13


def test_shock_cooling_matches_reference():
    lb, rp, tt = _ref_shock_cooling(T_SRC * DAY, 0.1, 1e13, 1e47, 10.0, 1.1)
    a, b, c, td = S.shock_cooling(jnp.asarray(T_SRC * DAY), 0.1, 1e13, 1e47, 10.0, 1.1)
    assert _maxrel(a, lb) < 1e-13
    assert _maxrel(b, rp) < 1e-13
    assert _maxrel(c, tt) < 1e-13


def test_shock_cooling_nn_delta_are_fitted_not_static(grid):
    """redback's prior gives nn Uniform(8, 12) and delta Uniform(1, 1.5) -- they must trace."""
    g = jax.grad(lambda v: jnp.sum(S.shock_cooling_and_arnett_bolometric(
        grid, -1.0, 13.0, 47.0, 0.1, 2.0, 1e4, 0.1, 0.03, v[0], v[1])))(
        jnp.array([10.0, 1.1]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.all(np.asarray(g) != 0.0), "nn/delta have no effect -- are they static?"


@pytest.mark.parametrize("convention", ["1.12", "1.15"])
def test_magnetar_matches_reference(grid, convention):
    dense = _ref_dense()
    lum = _ref_magnetar(dense * DAY, 2.0, 1.0, 1.4, 1.0, convention)
    ref = _ref_diffusion(T_SRC, dense, lum, 0.1, 0.03, 2.0, 1e4)
    ours = S.basic_magnetar_powered_bolometric(grid, 2.0, 1.0, 1.4, 1.0, 0.1, 0.03, 2.0, 1e4,
                                               magnetar_convention=convention)
    assert _maxrel(ours, ref) < 1e-13


def test_magnetar_conventions_differ_by_a_lot(grid):
    """CHANGE 5: not a rounding detail, and not a constant offset -- the curves cross."""
    a = np.asarray(S.basic_magnetar_powered_bolometric(
        grid, 2.0, 1.0, 1.4, 1.0, 0.1, 0.03, 2.0, 1e4, magnetar_convention="1.12"))
    b = np.asarray(S.basic_magnetar_powered_bolometric(
        grid, 2.0, 1.0, 1.4, 1.0, 0.1, 0.03, 2.0, 1e4, magnetar_convention="1.15"))
    ratio = b / a
    assert ratio.min() < 0.6 and ratio.max() > 1.8
    assert (ratio.min() < 1.0) and (ratio.max() > 1.0), "should cross, not just rescale"


def test_slsn_is_basic_magnetar_powered():
    assert S.slsn_bolometric is S.basic_magnetar_powered_bolometric


def test_type_1a_and_1c_are_arnett():
    assert S.type_1a_bolometric is S.arnett_bolometric
    assert S.type_1c_bolometric is S.arnett_bolometric


def test_magnetar_nickel_sums_before_diffusing(grid):
    """Both engines heat the same ejecta, so ONE kernel applies to their sum."""
    dense = _ref_dense()
    lum = (_ref_nico(dense, 0.1, 5.0) + _ref_magnetar(dense * DAY, 2.0, 1.0, 1.4, 1.0))
    ref = _ref_diffusion(T_SRC, dense, lum, 0.1, 0.03, 5.0, 1e4)
    ours = S.magnetar_nickel_bolometric(grid, 0.1, 5.0, 2.0, 1.0, 1.4, 1.0, 0.1, 0.03, 1e4)
    assert _maxrel(ours, ref) < 1e-13
    # and it is NOT the same as diffusing them separately and adding
    sep = (S.arnett_bolometric(grid, 0.1, 5.0, 0.1, 0.03, 1e4)
           + _ref_diffusion(T_SRC, dense, _ref_magnetar(dense * DAY, 2.0, 1.0, 1.4, 1.0),
                            0.1, 0.03, 5.0, 1e4))
    assert _maxrel(ours, sep) < 1e-13   # diffusion is LINEAR, so here they coincide


def test_csm_shock_breakout_matches_reference():
    lb, rp, tt = _ref_csm_sbo(T_SRC, 1.0, 1e4, 0.45, 0.2, 1.0, 0.1)
    a, b, c = S.csm_shock_breakout(jnp.asarray(T_SRC), 1.0, 1e4, 0.45, 0.2, 1.0, 0.1)
    assert _maxrel(a, lb) < 1e-13
    assert _maxrel(b, rp) < 1e-13
    assert _maxrel(c, tt) < 1e-13


def test_csm_arnett_uses_v_min_as_the_diffusion_velocity():
    """redback passes vej=v_min to the Arnett term; there is no separate vej parameter.

    On the closed-form grid (``csm_interp=False``), so the breakout term is exactly
    ``csm_shock_breakout`` and only the Arnett velocity is under test.
    """
    grid = S.build_sn_grid(T_SRC, csm_interp=False)
    ours = S.csm_shock_and_arnett_bolometric(grid, 2.0, 0.1, 1.0, 1e4, 0.45, 1.0, 0.1, 0.2, 0.03)
    sbo, _, _ = S.csm_shock_breakout(jnp.asarray(T_SRC), 1.0, 1e4, 0.45, 0.2, 1.0, 0.1)
    ref = np.asarray(sbo) + np.asarray(S.arnett_bolometric(grid, 0.1, 2.0, 0.2, 0.03, 1e4))
    assert _maxrel(ours, ref) < 1e-14
    assert "vej" not in S.PARAMETERS["csm_shock_and_arnett"]


@pytest.mark.parametrize("logl1,tr", [(54.0, 1.0), (52.0, 20.0)])
def test_fallback_matches_reference(grid, logl1, tr):
    dense = _ref_dense()
    raw = np.where(dense * DAY < tr * DAY, 10 ** logl1 * (tr * DAY) ** (-5. / 3.),
                   10 ** logl1 * np.maximum(dense, 1e-30) ** (-5. / 3.) / DAY ** (5. / 3.))
    ref = _ref_diffusion(T_SRC, dense, raw, 0.1, 0.03, 2.0, 1e4)
    ours = S.sn_fallback_bolometric(grid, logl1, tr, 0.1, 0.03, 2.0, 1e4)
    assert _maxrel(ours, ref) < 1e-13


def test_general_magnetar_reduces_to_dipole(grid):
    """nn = 3 gives (1+t/tau)^-2, the dipole; that is the whole point of the braking index."""
    t = jnp.asarray(T_SRC * DAY)
    assert _maxrel(S.magnetar_only(t, 1e45, 20 * DAY, 3.0),
                   1e45 * (1.0 + np.asarray(T_SRC) * DAY / (20 * DAY)) ** -2) < 1e-14


# =======================================================================================
# CLAIM 2: the SED layer reproduces redback
# =======================================================================================

def test_cutoff_shape_is_the_masked_ratio():
    """The CutoffBlackbody's two branches collapse to one continuous factor -- exactly."""
    lam = np.geomspace(1000.0, 30000.0, 500)
    shape = np.asarray(S.cutoff_shape(lam, redshift=0.0, cutoff_wavelength_ang=3000.0))
    below, above = lam < 3000.0, lam >= 3000.0
    assert np.allclose(shape[below], lam[below] / 3000.0, rtol=1e-15, atol=0)
    assert np.all(shape[above] == 1.0)       # exactly 1, so the branch is a true no-op


def test_cutoff_norm_renormalises_to_lbol(grid):
    """The norm exists to make int(SED) == L_bol; check it scales linearly in L and is O(1)."""
    lbol = S.arnett_bolometric(grid, 0.1, 2.0, 0.1, 0.03, 1e4)
    temp, rad = S.photosphere(grid, lbol, 1e4, 4000.0)
    n1 = np.asarray(S.cutoff_norm(lbol, temp, rad, 3000.0))
    n2 = np.asarray(S.cutoff_norm(2.0 * lbol, temp, rad, 3000.0))
    assert np.all(np.isfinite(n1)) and np.all(n1 > 0)
    assert _maxrel(n2, 2.0 * n1) < 1e-14
    assert 1e-3 < np.median(n1) < 1e3, "norm should be O(1); check FLUX_CONST_OVER_ANG"


def test_cutoff_norm_precision_does_not_depend_on_import_order():
    """REGRESSION: the cutoff's series constants were a jnp array built AT IMPORT, so a module
    imported before float64 was enabled -- as this file's own collection does, and as any script
    that imports first and configures second does -- froze them in float32. ``slsn`` and
    ``type_1a`` then carried float32's ~7.6e-8 relative error into float64 fits, and the magnetar
    parity in ``test_redback_adapter.py`` failed for ``slsn`` whenever this file was collected
    with it. Measured in fresh interpreters, one per import order."""
    import os
    import subprocess
    import sys

    code = ("import sys, jax\n"
            "if sys.argv[1] == 'after': jax.config.update('jax_enable_x64', True)\n"
            "from whisper_cbpf.models.jax import supernova as S\n"
            "jax.config.update('jax_enable_x64', True)\n"
            "import jax.numpy as jnp\n"
            "t = jnp.asarray([3000.0, 8000.0, 2e4])\n"
            "print(repr(S.cutoff_norm(1e43 * jnp.ones(3), t, 1e15 * jnp.ones(3), 3000.0).tolist()))")
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    got = {order: np.array(eval(subprocess.run(
        [sys.executable, "-c", code, order], capture_output=True, text=True, check=True,
        env=env).stdout)) for order in ("before", "after")}
    assert _maxrel(got["before"], got["after"]) < 1e-14, got


def test_synchrotron_is_negligible_in_the_optical():
    """CHANGE note on type_1c: the 10**22.5 branch step is at 1 GHz, not in any band."""
    nu = np.array([1e9 * 0.999, 1e9 * 1.001, 3e14, 5e14, 8e14])
    syn = np.asarray(S.synchrotron_f_nu(jnp.asarray(nu), 3.0, dl_cm=1e27))
    assert syn[1] / syn[0] > 1e20, "the documented discontinuity should still be there"
    # a plain blackbody at SN-like (T, R) dwarfs it at every optical frequency
    from whisper_cbpf.models.jax.tde import flux_density_mjy
    bb = np.asarray(flux_density_mjy(jnp.full(3, 8000.0), jnp.full(3, 1e15),
                                     jnp.asarray(nu[2:]), 0.0, 1e27, dilation=False)) * 1e-26
    assert np.all(syn[2:] / bb < 1e-8)


def test_line_is_both_subtractive_and_additive(grid, bands):
    """amplitude 0 must be a no-op; a nonzero amplitude must move flux, not only remove it."""
    lam, w, n, bidx = bands
    lbol = S.arnett_bolometric(grid, 0.1, 2.0, 0.1, 0.03, 1e4)
    kw = dict(band_idx=bidx, weights=w, norms=n, lam_obs_ang=lam, redshift=Z, dl_cm=7.09e26)
    m0 = S.sn_ab_magnitude(grid, lbol, 1e4, 4000.0, sed_kind="cutoff", **kw)
    m_off = S.sn_ab_magnitude(grid, lbol, 1e4, 4000.0, sed_kind="cutoff_line",
                              line_amplitude=0.0, **kw)
    m_on = S.sn_ab_magnitude(grid, lbol, 1e4, 4000.0, sed_kind="cutoff_line",
                             line_amplitude=0.3, **kw)
    assert _maxrel(m_off, m0) < 1e-14
    assert np.max(np.abs(np.asarray(m_on) - np.asarray(m0))) > 1e-3


def test_flux_and_magnitude_paths_agree(grid, bands):
    """Two independent code paths through each SED; a narrow band makes them comparable."""
    lam = np.geomspace(4990.0, 5010.0, 400)          # ~monochromatic top hat
    trans = np.ones((1, lam.size))
    w, n = S.ab_weights(lam, trans)
    bidx = jnp.zeros(T_SRC.size, dtype=int)
    nu_obs = jnp.full(T_SRC.size, 2.99792458e18 / 5000.0)
    lbol = S.arnett_bolometric(grid, 0.1, 2.0, 0.1, 0.03, 1e4)
    for kind in ("blackbody", "cutoff", "cutoff_line", "blackbody_synchrotron"):
        mag = np.asarray(S.sn_ab_magnitude(grid, lbol, 1e4, 4000.0, bidx, w, n,
                                           jnp.asarray(lam), Z, 7.09e26, sed_kind=kind))
        fd = np.asarray(S.sn_flux_density(grid, lbol, 1e4, 4000.0, nu_obs, Z, 7.09e26,
                                          sed_kind=kind))
        mag_from_fd = -2.5 * np.log10(fd * 1e-3 / 3631.0)
        assert np.max(np.abs(mag - mag_from_fd)) < 5e-3, kind


# =======================================================================================
# CLAIM 3: gradients
# =======================================================================================

@pytest.mark.parametrize("model", sorted(S.MODELS))
def test_gradients_finite_for_every_model(grid, bands, model):
    lam, w, n, bidx = bands
    free = list(S.PARAMETERS[model])
    v0 = jnp.array([float(DEMO[k]) for k in free])
    for fn in (
        lambda v: jnp.sum(S.ab_magnitude_of(model, grid, dict(zip(free, v)), bidx, w, n,
                                            lam, Z, 7.09e26)),
        lambda v: jnp.sum(S.flux_density(model, grid, dict(zip(free, v)),
                                         jnp.full(T_SRC.size, 5e14), Z, 7.09e26)),
    ):
        g = np.asarray(jax.grad(fn)(v0))
        bad = [free[i] for i in range(len(free)) if not np.isfinite(g[i])]
        assert not bad, f"{model}: non-finite gradient w.r.t. {bad}"


def test_gradient_finite_at_theta_pb_zero(grid):
    """CHANGE 3. redback's prior is Uniform(0, pi/2), so this edge IS drawn from."""
    for th in (0.0, np.pi):
        g = jax.grad(lambda t: jnp.sum(S.basic_magnetar_powered_bolometric(
            grid, 2.0, 1.0, 1.4, t, 0.1, 0.03, 2.0, 1e4)))(th)
        assert np.isfinite(float(g))


def test_gradient_finite_at_time_zero():
    """The dense grid starts at exactly 0; every engine must survive being evaluated there."""
    t0 = jnp.asarray(np.array([0.0, 1e-8, 1.0]))
    for fn, args in [(S.exponential_powerlaw_engine, (1e43, 2.0, 1.0, 10.0)),
                     (S.fallback_lbol, (54.0, 1.0)),
                     (S.nickelcobalt_engine, (0.1, 2.0))]:
        v = np.asarray(fn(t0, *args))
        assert np.all(np.isfinite(v)), fn.__name__
        g = jax.grad(lambda a: jnp.sum(fn(t0, a, *args[1:])))(float(args[0]))
        assert np.isfinite(float(g)), fn.__name__


# =======================================================================================
# CLAIM 4: the documented deviations are bounded
# =======================================================================================

@pytest.mark.parametrize("preset,nodes", [("1.20", np.geomspace(1e-2, 200.0, 300)),
                                          ("1.15", np.linspace(1e-2, 200.0, 300))])
def test_csm_breakout_is_redbacks_interpolation_by_default(preset, nodes):
    """CHANGE 4: the default reproduces redback's interp1d off its 300 fixed nodes -- geometric in
    1.20, linear in <= 1.15 -- and ``csm_interp=False`` the closed form. The two differ by a
    measured amount, so the choice is not a detail."""
    from scipy.interpolate import interp1d

    t = np.concatenate([T_SRC, [250.0]])            # one epoch past the last node, 200 d
    args = (2.0, 0.1, 1.0, 1e4, 0.45, 1.0, 0.1, 0.2, 0.03)
    grid = S.build_sn_grid(t, **S.REDBACK_GRID_PRESETS[preset])
    arnett = np.asarray(S.arnett_bolometric(grid, 0.1, 2.0, 0.2, 0.03, 1e4))
    exact, _, _ = _ref_csm_sbo(t, 1.0, 1e4, 0.45, 0.2, 1.0, 0.1)
    on_nodes, _, _ = _ref_csm_sbo(nodes, 1.0, 1e4, 0.45, 0.2, 1.0, 0.1)
    interp = interp1d(nodes, on_nodes, fill_value="extrapolate")(t)

    ours = np.asarray(S.csm_shock_and_arnett_bolometric(grid, *args))
    inside = t <= 200.0
    assert _maxrel(ours[inside], (interp + arnett)[inside]) < 1e-13
    # past the last node redback extrapolates linearly (negative from ~201 d); the
    # port keeps the closed form there
    assert ours[~inside] == pytest.approx((exact + arnett)[~inside], rel=1e-13)

    closed = S.build_sn_grid(t, csm_interp=False, **S.REDBACK_GRID_PRESETS[preset])
    assert _maxrel(S.csm_shock_and_arnett_bolometric(closed, *args), exact + arnett) < 1e-14
    dmag = np.abs(2.5 * np.log10(np.maximum(interp, 1e-300) / np.maximum(exact, 1e-300)))[inside]
    assert dmag.max() > 1e-3, "if this shrinks, redback changed its grid -- re-measure CHANGE 4"


def test_exponential_powerlaw_deviation_converges_where_the_integral_exists():
    """CHANGE 3, integrable half: alpha_2 - alpha_1 = -1 < 2, so it IS a grid statement."""
    assert S.exponential_powerlaw_integrable(2.0, 1.0)
    prev = None
    for n_dense in (1000, 2000, 5000):
        g = S.build_sn_grid(T_SRC, dense_resolution=n_dense, spacing="linear")   # CHANGE 3's grid
        dense = np.linspace(0.0, T_SRC[-1] + 100.0, n_dense)
        with np.errstate(all="ignore"):
            raw = (1e43 * (1 - np.exp(-dense / 10.0)) ** 2.0 * (dense / 10.0) ** -1.0)
        ref = _ref_diffusion(T_SRC, dense, raw, 0.1, 0.03, 2.0, 1e4)   # NaN at index 0, as redback
        ours = S.exponential_powerlaw_bolometric(g, 1e43, 2.0, 1.0, 10.0, 0.1, 0.03, 2.0, 1e4)
        r = _maxrel(ours, ref)
        if prev is not None:
            assert r < prev / 3.0, f"not converging: {prev:.2e} -> {r:.2e} at n={n_dense}"
        prev = r
    assert prev < 1e-6


def test_exponential_powerlaw_divergent_half_does_not_converge():
    """CHANGE 3, the other half: alpha_2 - alpha_1 > 2, so there is no limit to converge to.

    This is a test that the DOCUMENTED failure is still the failure. If a future edit makes the
    divergent region converge, the integral has been changed, not the quadrature.
    """
    assert not S.exponential_powerlaw_integrable(1.0, 5.0)
    vals = []
    for n_dense in (1000, 10000, 100000):
        g = S.build_sn_grid(T_SRC, dense_resolution=n_dense, spacing="linear")   # CHANGE 3's grid
        lb = np.asarray(S.exponential_powerlaw_bolometric(
            g, 1e43, 1.0, 5.0, 0.1, 0.1, 0.03, 2.0, 1e4))
        assert np.all(np.isfinite(lb))
        vals.append(lb.max())
    # a 100x refinement must NOT settle: consecutive answers differ by more than 10%
    assert abs(vals[1] - vals[0]) / vals[0] > 0.1
    assert abs(vals[2] - vals[1]) / vals[1] > 0.1


def test_exponential_powerlaw_has_no_first_interval_deviation_on_the_geometric_grid(grid):
    """CHANGE 3 lives on the linear grid only. redback 1.20's grid starts at 1e-5 d, where the
    formula is finite, so both codes evaluate the same engine at the same nodes and agree."""
    dense = _ref_dense()
    raw = 1e43 * (1 - np.exp(-dense / 10.0)) ** 2.0 * (dense / 10.0) ** -1.0
    assert np.all(np.isfinite(raw))
    ref = _ref_diffusion(T_SRC, dense, raw, 0.1, 0.03, 2.0, 1e4)
    ours = S.exponential_powerlaw_bolometric(grid, 1e43, 2.0, 1.0, 10.0, 0.1, 0.03, 2.0, 1e4)
    assert _maxrel(ours, ref) < 1e-13


def test_exponential_powerlaw_gradients_at_the_divergent_corner(grid, bands):
    """Regression: a 1e-30 floor made 0*inf = NaN in the dead branch on 30/200 prior draws."""
    lam, w, n, bidx = bands
    free = list(S.PARAMETERS["sn_exponential_powerlaw"])
    for a1, a2, tp in [(9.0, 9.5, 1e-3), (0.5, 9.9, 0.01), (9.9, 0.1, 200.0)]:
        p = dict(DEMO, alpha_1=a1, alpha_2=a2, tpeak_d=tp, lbol_0=1e43)
        v0 = jnp.array([float(p[k]) for k in free])
        g = np.asarray(jax.grad(lambda v: jnp.sum(S.ab_magnitude_of(
            "sn_exponential_powerlaw", grid, dict(zip(free, v)), bidx, w, n, lam, Z,
            7.09e26)))(v0))
        bad = [free[i] for i in range(len(free)) if not np.isfinite(g[i])]
        assert not bad, f"alpha_1={a1} alpha_2={a2} tpeak={tp}: non-finite d/d{bad}"


def test_fallback_interaction_switch(grid):
    """CHANGE 7b: `interaction=False` is redback's model; the two are far apart."""
    on = np.asarray(S.sn_fallback_bolometric(grid, 54.0, 1.0, 0.1, 0.03, 2.0, 1e4))
    off = np.asarray(S.sn_fallback_bolometric(grid, 54.0, 1.0, 0.1, 0.03, 2.0, 1e4,
                                              interaction=False))
    assert _maxrel(off, np.asarray(S.fallback_lbol(jnp.asarray(T_SRC), 54.0, 1.0))) < 1e-15
    assert np.max(np.abs(2.5 * np.log10(on / off))) > 1.0

    # with interaction off, kappa / kappa_gamma / mej do nothing -- as in redback
    other = np.asarray(S.sn_fallback_bolometric(grid, 54.0, 1.0, 2.0, 1e4, 50.0, 1e4,
                                                interaction=False))
    assert np.array_equal(off, other)
    # with it on, they do
    assert not np.allclose(on, np.asarray(S.sn_fallback_bolometric(
        grid, 54.0, 1.0, 2.0, 1e4, 50.0, 1e4)))


# =======================================================================================
# CLAIM 5: float32 fails loudly
# =======================================================================================

def test_float32_raises_rather_than_returning_mag_floor():
    """CHANGE 8. Without the guard this returns a whole light curve at 40.0 mag."""
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="requires float64"):
            S.build_sn_grid(T_SRC)
    finally:
        jax.config.update("jax_enable_x64", was)


# =======================================================================================
# registry, priors and the whisper adapter
# =======================================================================================

def test_every_model_has_a_prior_for_every_free_parameter():
    for model in S.MODELS:
        prior, pinned, _ = S.fallback_prior(model)
        missing = [p for p in S.PARAMETERS[model]
                   if p not in prior.distributions and p not in pinned]
        assert not missing, f"{model}: {missing} have no fallback prior"


def test_engine_params_are_a_subset_of_params():
    """The table drives the adapter; a typo here would surface as a KeyError mid-fit."""
    for model, spec in S.MODELS.items():
        extra = [p for p in spec["engine_params"]
                 if p not in spec["params"] and p not in ("nn", "delta")]
        assert not extra, f"{model}: engine wants {extra}, which is not a parameter"
        assert spec["vej_name"] in spec["params"], model
        assert "temperature_floor" in spec["params"], model


@pytest.mark.parametrize("model", sorted(S.MODELS))
def test_redback_prior_matches_the_transcription(model):
    """`fallback_prior` is a transcription and transcriptions drift. Check it against the file."""
    pytest.importorskip("redback")
    pytest.importorskip("whisper_cbpf")
    try:
        rb, rb_pinned, _ = S.redback_prior(model)
    except (ImportError, KeyError, NotImplementedError) as exc:
        pytest.skip(f"redback prior unavailable for {model}: {exc}")
    fb, fb_pinned, _ = S.fallback_prior(model)
    assert set(rb.distributions) == set(fb.distributions), model
    assert rb_pinned == fb_pinned, model
    for k, d in rb.distributions.items():
        assert type(d).__name__ == type(fb.distributions[k]).__name__, f"{model}.{k}"
        assert d.bounds == pytest.approx(fb.distributions[k].bounds, rel=1e-12), f"{model}.{k}"


def test_adapter_builds_and_predicts(bands):
    """The whisper contract end to end, with a synthetic filter set (no sncosmo needed)."""
    pytest.importorskip("whisper_cbpf")
    from whisper_cbpf.models.jax import supernova_model

    lam, _, _, _ = bands
    trans = np.zeros((2, np.asarray(lam).size))
    la = np.asarray(lam)
    trans[0][(la > 4000) & (la < 5500)] = 1.0
    trans[1][(la > 5500) & (la < 7000)] = 1.0
    fs = {"lam": la, "trans": trans, "names": np.array(["a", "b"])}
    b = np.array(["a" if i % 2 == 0 else "b" for i in range(T_OBS.size)])

    for model in sorted(S.MODELS):
        # constraint=None: DEMO's general_magnetar_slsn breaks redback's rotational-energy
        # constraint, and predict gives zero flux there (the wall, tests/test_constraints.py).
        # This test is the physics contract, at any draw.
        m = supernova_model(model, ["a", "b"], Z, 7.09e26, filter_set=fs, times=T_OBS,
                            constraint=None)
        flux = m.predict({k: DEMO[k] for k in m.parameters}, T_OBS, b)
        assert flux.shape == T_OBS.shape
        assert np.all(np.isfinite(flux)) and np.all(flux > 0), model
        assert set(m.parameters) <= set(S.PARAMETERS[model]) | {"line_time", "line_duration"}
        assert m.default_prior is not None and set(m.default_prior.names) == set(m.parameters)


def _tp_s(p0, bp, mass_ns=1.4, theta_pb=np.pi / 2):
    return 1.3e5 * bp ** -2 * p0 ** 2 * (mass_ns / 1.4) ** 1.5 * np.sin(theta_pb) ** -2


@pytest.mark.parametrize("tp_s,delivered", [(1.43, 0.45), (0.105, 0.057)])
def test_spin_down_energy_redbacks_own_grid_cannot_place(tp_s, delivered):
    """CHANGE 1. With redback 1.20's grid the port matches redback -- and redback does not
    conserve E_rot for a fast spin-down: nothing integrates [0, 1e-5 d] = [0, 0.864 s]. The closed
    form reproduces the measured 45% (t_p = 1.43 s) and 5.7% (0.105 s) delivered."""
    bp = np.sqrt(1.3e5 * 4.0 / tp_s)                    # p0 = 2 ms
    assert _tp_s(2.0, bp) == pytest.approx(tp_s)
    dense = S.dense_grid(T_SRC)
    with pytest.warns(RuntimeWarning, match="spin-down unresolved"):
        missed = S.warn_if_spin_down_unresolved(dense, 2.0, bp, 1.4, np.pi / 2)
    assert 1.0 - missed == pytest.approx(delivered, abs=0.005)


def test_spin_down_warning_is_silent_when_resolved_and_loud_on_the_linear_grid():
    import warnings

    slow = np.sqrt(1.3e5 * 4.0 / (10.0 * DAY))           # t_p = 10 d
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert S.warn_if_spin_down_unresolved(S.dense_grid(T_SRC), 2.0, slow, 1.4, 1.0) < 1e-5
    # the linear grid's first cell is (t_last + 100)/999 = 0.29 d: t_p = 1 h is unresolved there
    hour = np.sqrt(1.3e5 * 4.0 / 3600.0)
    with pytest.warns(RuntimeWarning, match="spin-down unresolved"):
        S.warn_if_spin_down_unresolved(S.dense_grid(T_SRC, spacing="linear"), 2.0, hour, 1.4,
                                       np.pi / 2)


def test_adapter_warns_once_when_the_spin_down_is_unresolved(bands):
    """``predict`` checks the concrete parameters host-side, and says it once per model."""
    import warnings

    from whisper_cbpf.models.jax import supernova_model

    la = np.asarray(bands[0])
    trans = np.zeros((1, la.size))
    trans[0][(la > 4000) & (la < 5500)] = 1.0
    fs = {"lam": la, "trans": trans, "names": np.array(["a"])}
    b = np.array(["a"] * T_OBS.size)
    m = supernova_model("basic_magnetar_powered", ["a"], Z, 7.09e26, filter_set=fs, times=T_OBS)
    p = {k: DEMO[k] for k in m.parameters}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # DEMO's t_p = 0.61 d: resolved
        m.predict(p, T_OBS, b)
    fast = dict(p, bp=np.sqrt(1.3e5 * 4.0 / 0.105) / np.sin(p["theta_pb"]))
    with pytest.warns(RuntimeWarning, match="spin-down unresolved"):
        m.predict(fast, T_OBS, b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.predict(fast, T_OBS, b)                       # once per model, not per call


def test_adapter_pin_removes_a_parameter_without_changing_the_curve(bands):
    pytest.importorskip("whisper_cbpf")
    from whisper_cbpf.models.jax import supernova_model

    la = np.asarray(bands[0])
    trans = np.zeros((1, la.size))
    trans[0][(la > 4000) & (la < 5500)] = 1.0
    fs = {"lam": la, "trans": trans, "names": np.array(["a"])}
    b = np.array(["a"] * T_OBS.size)

    free = supernova_model("arnett", ["a"], Z, 7.09e26, filter_set=fs, times=T_OBS)
    pinned = supernova_model("arnett", ["a"], Z, 7.09e26, filter_set=fs, times=T_OBS,
                             pin={"kappa": 0.1})
    assert "kappa" not in pinned.parameters
    assert set(free.parameters) - set(pinned.parameters) == {"kappa"}
    f1 = free.predict({k: DEMO[k] for k in free.parameters}, T_OBS, b)
    f2 = pinned.predict({k: DEMO[k] for k in pinned.parameters}, T_OBS, b)
    assert np.array_equal(f1, f2)
