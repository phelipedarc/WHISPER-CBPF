# Adapted from redback-jax (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback-jax, commit 3431e2e, GPL-3.0, and from redback (Sarin et al.), https://github.com/nikhil-sarin/redback, GPL-3.0; corrected and modified for whisper_cbpf.
"""Metzger (2017) multi-shell kilonova and the magnetar-boosted kilonova, in JAX.

Two models that redback solves with an explicit-Euler loop over 200 velocity shells:

* ``metzger_kilonova_model`` -- :func:`metzger_kilonova_model`, :func:`register_metzger_kilonova`.
  Ejecta of mass ``mej`` with a power-law velocity profile ``m(>v) ~ v^-beta`` between ``vej`` and
  ``vmax``; every shell is heated by r-process decay (Barnes & Kasen 2016 thermalisation) and by
  the free-neutron decay of the fastest layers, loses energy adiabatically and radiates on its own
  diffusion time; the light curve is the sum over shells and the photosphere is the shell whose
  optical depth is closest to one.
* ``metzger_magnetar_driven_kilonova_model`` -- :func:`magnetar_boosted_kilonova_model`,
  :func:`register_magnetar_boosted_kilonova`. The same ejecta with a central magnetar that heats
  the innermost shell, accelerates the ejecta by pdV work (the kinetic energy and every shell
  velocity evolve), and with a pair-cascade suppression of the luminosity.

Both are line-by-line ports of redback 1.20's ``_metzger_kilonova_model`` and
``_general_metzger_magnetar_driven_kilonova_model`` (``magnetar_heating='first_layer'``): the same
Euler update, the same opacity, neutron-fraction and photosphere rules, the same electron-fraction
interpolation (Tanaka et al. 2019). The structure of a ``lax.scan`` over time nodes with the shell
axis vectorised follows redback-jax; the energy bookkeeping is redback's.

Differences from redback
------------------------
* **Fixed internal time grid.** redback builds its Euler grid from the epochs asked for
  (``get_optimal_time_array``), so its answer at an epoch moves with the other epochs, and the
  scan length depends on them. Here the grid is a constant (:data:`METZGER_GRID`,
  :data:`MAGNETAR_GRID`; geometric, :data:`N_METZGER` and :data:`N_MAGNETAR` nodes), the ODE is
  solved on it once per parameter set, and the photosphere is interpolated to the epochs. A
  light curve therefore does not depend on which epochs accompany it, nor on where the array
  starts, and ``t_exp`` and the redshift can be free parameters.
* **Discretisation.** Euler is first order in the node spacing. The node counts above are set by
  a convergence test (tabulated at the grid constants): against a 16000-node reference the fixed
  grids are several times better than redback's own (500 and 300 nodes). Run on redback's own
  grid, :func:`metzger_engine` and :func:`magnetar_boosted_engine` reproduce redback to rounding
  (1e-14 dex), so what remains between this module and redback at the default grids is redback's
  own Euler error.
* **Float32.** Shell energies are carried so nothing leaves float32's range: the magnetar
  model's energies are in units of 1e40 erg (its kinetic energy is 1e51 erg and its spin-down
  luminosity can reach 1e55 erg/s), and every constant that would be squared in a backward pass
  is folded to O(1). The Metzger model needs no scaling. Neither needs float64.

Spectral energy distribution
----------------------------
redback's convention for both models, kept: a plain blackbody at the photosphere temperature
``T = (L / 4 pi sigma R^2)^(1/4)`` and radius ``R = v_ph t`` of the shell that sets the
photosphere. **No temperature floor and no cutoff** (the one-component kilonova's floor and the
supernova family's cutoff blackbody do not apply). Source-frame time ``t = (t_obs - t_exp) /
(1 + z)`` and ``(1 + z)`` dilation as in :mod:`whisper_cbpf.models.jax.kilonova`; the band flux is
the same synthetic-photometry integral (:func:`whisper_cbpf.synphot.filter_set_for`). As redback
does, ``T`` and ``R`` are interpolated linearly in time to the epochs and the blackbody is taken
there.

redback's own idiosyncrasies are reproduced, not repaired: the initial shell energy
``0.5 m v^2`` with ``m`` in solar masses; the magnetar model's kinetic-energy update from the
innermost shell only; the shell-velocity cap at ``3e10`` cm/s (slightly above ``c``); and the
pair-cascade factor, which redback evaluates with the velocity of the *last* time step for every
epoch (so the magnetar grid ends where redback's ends, 1e7 s).
"""
from __future__ import annotations

import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import kilonova as kn

__all__ = ["metzger_kilonova_model", "register_metzger_kilonova",
           "magnetar_boosted_kilonova_model", "register_magnetar_boosted_kilonova",
           "metzger_engine", "magnetar_boosted_engine", "metzger_bolometric",
           "magnetar_boosted_bolometric", "default_prior_metzger", "default_prior_magnetar_boosted",
           "METZGER_GRID", "MAGNETAR_GRID"]

# --- constants (astropy cgs, as redback.constants) ----------------------------------------------
_MSUN = kn.SOLAR_MASS
_C = kn.SPEED_OF_LIGHT
_SIGMA = kn.SIGMA_SB
_DAY = kn.DAY_TO_S

# Shell mass and speed are carried in units where v is a fraction of c, so every divisor that a
# float32 backward pass squares stays O(1)-O(1e7):
#   td   = 3 kappa m Msun / (4 pi v c t beta)        = _K_DIFF * kappa m / (u t beta),  u = v / c
#   tau  = m Msun kappa / (4 pi (t v)^2)             = _K_TAU  * kappa m / (u t)^2
_K_DIFF = 3.0 * _MSUN / (4.0 * np.pi * _C ** 2)
_K_TAU = _MSUN / (4.0 * np.pi * _C ** 2)
_LOG10_MSUN = float(np.log10(_MSUN))
_LOG10_4PI_SIGMA = float(np.log10(4.0 * np.pi * _SIGMA))
_LOG10_C = float(np.log10(_C))

N_SHELLS = 200                  # redback's mass_len
_T0, _SIG = 1.3, 0.11           # heating-rate switch time and width [s]
_TAU_NEUTRON = 900.0            # free-neutron lifetime [s]
_EDOT_NEUTRON = 3.2e14          # erg/s/g per unit neutron mass fraction
_KAPPA_NEUTRON = 0.4            # neutron opacity [cm^2/g]
_NEUTRON_MASS = 1e-4            # solar masses of the neutron precursor layer
_VCAP = 3e10 / _C               # redback caps shell speeds above 3e10 cm/s at c
_ESCALE = 1e40                  # erg: energy unit of the magnetar model (float32 safety)

# Electron fraction from the grey opacity (Tanaka et al. 2019), redback.utils
# .electron_fraction_from_kappa: piecewise linear, extended linearly beyond the ends.
_YE_KAPPA = np.array([0.5, 0.96, 3.30, 5.36, 5.60, 22.3, 32.2, 35.0])
_YE_VALUE = np.array([0.5, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.10])

# --- the fixed time grids ------------------------------------------------------------------------
# Source-frame seconds, geometric. The end points are redback's (the Metzger model's grid starts at
# 1e-2 s, the magnetar model's at 1e-4 s); both end at 1e7 s = 116 d, where redback's magnetar grid
# ends, which also fixes the pair-cascade velocity. Euler's error is proportional to the node
# spacing, so the node counts come from a convergence test: |dlog10 L| against a 16000-node
# reference, 60 prior draws, L at 12 epochs from 0.2 to 40 d (median / 99th percentile / max):
#
#   nodes    Metzger kilonova            magnetar-driven kilonova
#    500     0.0134 / 0.0356 / 0.0406    0.0157 / 0.0921 / 0.110      (redback: 500 and 300)
#   1000     0.0066 / 0.0175 / 0.0200    0.0078 / 0.0449 / 0.0542
#   2000     0.0031 / 0.0082 / 0.0094    0.0037 / 0.0211 / 0.0255
#   4000     0.0013 / 0.0035 / 0.0041    0.0016 / 0.0091 / 0.0110
#   8000     0.00045 / 0.0012 / 0.0014   0.00053 / 0.0031 / 0.0037
#
# The error halves per doubling. 2000 nodes (Metzger) puts the maximum at 0.009 dex (0.02 mag),
# 4000 (magnetar, whose prior reaches far harder corners: spin-down times from 1e-3 s to 1e21 s)
# at 0.011 dex; redback's own grids carry 0.05 dex (500 nodes) and 0.13 dex (300 nodes).
N_METZGER = 2000
N_MAGNETAR = 4000
METZGER_GRID = np.geomspace(1e-2, 1e7, N_METZGER)
MAGNETAR_GRID = np.geomspace(1e-4, 1e7, N_MAGNETAR)


def _electron_fraction(kappa):
    """Electron fraction from the grey opacity: linear interpolation, linear extension."""
    k = jnp.asarray(kappa)
    kx, ky = jnp.asarray(_YE_KAPPA, k.dtype), jnp.asarray(_YE_VALUE, k.dtype)
    inner = jnp.interp(k, kx, ky)
    lo = (ky[1] - ky[0]) / (kx[1] - kx[0])
    hi = (ky[-1] - ky[-2]) / (kx[-1] - kx[-2])
    return inner + lo * jnp.minimum(k - kx[0], 0.0) + hi * jnp.maximum(k - kx[-1], 0.0)


def _log1p_over_x(x):
    """``log1p(x) / x`` with the series below 1e-2, where the quotient and its derivative
    lose digits (x ~ 1e-7 at the first node); the series error there is below 1e-13."""
    small = jnp.abs(x) < 1e-2
    xs = jnp.where(small, jnp.asarray(1e-2, x.dtype), x)
    series = 1.0 - x * (0.5 - x * (1.0 / 3.0 - x * (0.25 - x * (0.2 - x / 6.0))))
    return jnp.where(small, series, jnp.log1p(xs) / xs)


def _heating(grid, mej, vej):
    """r-process heating per gram, ``edotr`` of redback, at the grid nodes [erg/s/g].

    The time structure is a constant (``late`` selects redback's two branches around
    ``t0 = 1.3 s``); only the Barnes & Kasen thermalisation efficiency depends on the parameters.
    """
    t = np.asarray(grid, dtype=np.float64)
    tdays = t / _DAY
    late = t > _T0
    early_rate = 4.0e18 * (0.5 - np.arctan((t - _T0) / _SIG) / np.pi) ** 1.3
    base = np.where(late, 2.1e10 * tdays ** -1.3, early_rate)
    av, bv, dv = kn.thermalisation_coeffs(mej, vej)
    dtype = jnp.result_type(av)
    td = jnp.asarray(tdays, dtype)
    e_th = 0.36 * (jnp.exp(-av * td) + _log1p_over_x(2.0 * bv * td ** dv))
    return jnp.asarray(base, dtype) * e_th


def _shells(mej, vej, beta, vmax, normalise):
    """Velocity [c], mass [Msun] of the 200 shells; the last shell is dropped by the caller."""
    vel = jnp.linspace(vej, vmax, N_SHELLS)
    w = (vel / vej) ** (-beta)
    if normalise:
        w = w / jnp.sum(w)
    return vel, mej * w


def _neutron_layer(m, kappa):
    """Initial neutron mass fraction ``Xn0`` and r-process fraction ``Xr`` of every shell."""
    ye = _electron_fraction(kappa)
    xn0 = (2.0 / np.pi) * (1.0 - ye) * jnp.arctan(_NEUTRON_MASS / m)
    return xn0, 1.0 - xn0


def _finish(logl, r_c):
    """``(log10 L, T [K], R [cm])`` on all nodes. ``r_c`` is R / c in seconds (``u_ph t``).

    redback leaves its last node at zero (the loop stops one short); here the last node repeats
    the one before it.
    """
    log_r = jnp.log10(r_c) + _LOG10_C
    log_t = 0.25 * (logl - _LOG10_4PI_SIGMA - 2.0 * log_r)
    temp, radius = 10.0 ** log_t, 10.0 ** log_r
    pad = lambda x: jnp.concatenate([x, x[-1:]])            # noqa: E731
    return pad(logl), pad(temp), pad(radius)


def metzger_engine(grid_s, mej, vej, beta, kappa, *, vmax=0.7, neutron_precursor=True):
    """Bolometric luminosity, temperature and photosphere radius of the Metzger (2017) kilonova.

    redback 1.20's ``_metzger_kilonova_model``, integrated with the same explicit-Euler update on
    ``grid_s`` (ascending, concrete). Pass redback's grid to reproduce redback to rounding; the
    model functions pass the fixed :data:`METZGER_GRID`.

    Parameters
    ----------
    grid_s : array
        Source-frame time nodes, seconds since the merger, ascending.
    mej : float
        Ejecta mass, solar masses.
    vej : float
        Minimum (inner-shell) velocity, units of c.
    beta : float
        Velocity power-law slope, ``m(>v) ~ v^-beta``.
    kappa : float
        Grey opacity of the r-process material, cm^2/g.
    vmax : float
        Maximum shell velocity, units of c (redback: 0.7).
    neutron_precursor : bool
        Heat and opacity of the free-neutron layer (redback: on).

    Returns
    -------
    log10_L, T, R : arrays, one value per node
        ``log10`` of the luminosity [erg/s], the photosphere temperature [K] and radius [cm]. The
        last node repeats the one before it.

    Notes
    -----
    The luminosity is ``sum_i lum_i dm_i`` over the shell energies per gram (``erg/g``), times the
    solar mass; the first-order scheme is stable for node spacings up to about 10% in time.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax import kilonova_metzger as km
    >>> logl, temp, rad = km.metzger_engine(km.METZGER_GRID, 0.03, 0.2, 4.0, 10.0)
    >>> day = np.searchsorted(km.METZGER_GRID, 86400.0)
    >>> bool(40.0 < float(logl[day]) < 43.0)
    True
    """
    grid = np.asarray(grid_s, dtype=np.float64)
    heat_r = _heating(grid, mej, vej)
    vel, m = _shells(mej, vej, beta, vmax, normalise=False)
    dtype = jnp.result_type(heat_r)
    vel, m = vel.astype(dtype), m.astype(dtype)
    vi, mi = vel[:-1], m[:-1]
    dm = jnp.abs(jnp.diff(m))
    xn0, xr = _neutron_layer(mi, kappa)
    e0 = 0.5 * mi * (vi * _C) ** 2
    t_nodes = jnp.asarray(grid[:-1], dtype)
    dt_nodes = jnp.asarray(np.diff(grid), dtype)

    @jax.checkpoint
    def step(energy, xs):
        t, dt, er = xs
        if neutron_precursor:
            xn = xn0 * jnp.exp(-t / _TAU_NEUTRON)
            kap = _KAPPA_NEUTRON * (xn0 - xn) + kappa * xr
            heat = er + _EDOT_NEUTRON * xn
        else:
            kap, heat = kappa, er
        td = _K_DIFF * kap * mi / (vi * t * beta)
        lum = energy / (td + t * vi)
        tau = _K_TAU * kap * mi / (t * vi) ** 2
        photo = jnp.argmin(jnp.abs(tau - 1.0))
        new = energy + (heat - energy / t - lum) * dt
        return new, (jnp.sum(lum * dm), vi[photo] * t)

    _, (lsum, r_c) = lax.scan(step, e0, (t_nodes, dt_nodes, heat_r[:-1]))
    logl = jnp.log10(jnp.maximum(lsum, jnp.finfo(lsum.dtype).tiny)) + _LOG10_MSUN
    return _finish(logl, r_c)


def magnetar_boosted_engine(grid_s, mej, vej, beta, kappa, p0, bp, mass_ns, theta_pb,
                            thermalisation_efficiency, *, vmax=0.7, neutron_precursor=True,
                            pair_cascade=True, pair_cascade_fraction=0.01, ejecta_albedo=0.5):
    """Luminosity, temperature and photosphere radius of the magnetar-driven Metzger kilonova.

    redback 1.20's ``_general_metzger_magnetar_driven_kilonova_model`` as called by
    ``metzger_magnetar_driven_kilonova_model`` (``magnetar_heating='first_layer'``, thermalisation
    efficiency given, no gamma-ray opacity), on ``grid_s``. Pass redback's grid
    (``get_optimal_time_array(1e-4, 1e7, 300)``) to reproduce redback to rounding.

    Parameters
    ----------
    grid_s : array
        Source-frame time nodes, seconds, ascending, concrete.
    mej, vej, beta, kappa, vmax, neutron_precursor
        As :func:`metzger_engine`. The mass array is normalised to sum to ``mej`` (redback's
        magnetar model does; its Metzger model does not).
    p0 : float
        Initial spin period, milliseconds.
    bp : float
        Polar magnetic field, units of 1e14 G.
    mass_ns : float
        Neutron star mass, solar masses.
    theta_pb : float
        Angle between the spin and magnetic axes, radians (the dipole spin-down time scales as
        ``sin(theta)^-2``).
    thermalisation_efficiency : float
        Fraction of the spin-down luminosity absorbed by the innermost shell.
    pair_cascade : bool
        Suppress the luminosity by ``1 / (1 + t_life / t)`` (redback: on).
    pair_cascade_fraction, ejecta_albedo : float
        Pair-cascade parameters (redback: 0.01 and 0.5).

    Returns
    -------
    log10_L, T, R : arrays, one value per node
        As :func:`metzger_engine`.

    Notes
    -----
    Heating: the dipole spin-down ``2 E_rot/t_p (1 + 2t/t_p)^-2`` with ``E_rot = 2.6e52
    (M/1.4)^1.5 P0^-2`` erg and ``t_p = 1.3e5 B^-2 P0^2 (M/1.4)^1.5 / sin^2(theta)`` s, redback
    1.20's ``basic_magnetar``. The kinetic energy grows by ``E_0/t dt`` (the innermost shell's
    pdV work) and the shell velocities follow it: ``v = v0 (m/mej)^(-1/beta)``, capped at ``c``.
    Energies are in units of 1e40 erg internally. The cascade factor uses the last step's ``v0``
    for every epoch, as redback does.

    Examples
    --------
    >>> from whisper_cbpf.models.jax import kilonova_metzger as km
    >>> logl, temp, rad = km.magnetar_boosted_engine(km.MAGNETAR_GRID, 0.03, 0.2, 4.0, 10.0,
    ...                                              1.0, 1.0, 1.4, 1.0, 0.5)
    >>> bool(logl.shape == temp.shape == rad.shape == km.MAGNETAR_GRID.shape)
    True
    """
    grid = np.asarray(grid_s, dtype=np.float64)
    heat_r = _heating(grid, mej, vej)
    vel, m = _shells(mej, vej, beta, vmax, normalise=True)
    dtype = jnp.result_type(heat_r)
    vel, m = vel.astype(dtype), m.astype(dtype)
    vi, mi = vel[:-1], m[:-1]
    ratio = mi / mej                                         # (m / mej) of the Metzger shells
    dm = jnp.abs(jnp.diff(m))
    xn0, xr = _neutron_layer(mi, kappa)
    t_all = jnp.asarray(grid, dtype)

    erot = (2.6e52 / _ESCALE) * (mass_ns / 1.4) ** 1.5 * p0 ** -2
    sin_t = jnp.maximum(jnp.sin(theta_pb), 1e-12)
    tp = 1.3e5 * bp ** -2 * p0 ** 2 * (mass_ns / 1.4) ** 1.5 / sin_t ** 2
    lsd = 2.0 * erot / tp / (1.0 + 2.0 * t_all / tp) ** 2     # erg/s in units of 1e40
    heat_scale = dm * (_MSUN / _ESCALE)
    ke0 = (0.5 * _MSUN * _C ** 2 / _ESCALE) * mej * vej ** 2   # kinetic energy, 1e40 erg
    e0 = (0.5 * _C ** 2 / _ESCALE) * mi * vi ** 2             # constants first: 1e40 is not a float32

    @jax.checkpoint
    def step(carry, xs):
        energy, ke = carry
        t, dt, er, ls = xs
        ke = ke + (energy[0] / t) * dt
        u0 = vej * jnp.sqrt(ke / ke0)                        # v0 / c
        u = u0 * ratio ** (-1.0 / beta)
        u = jnp.where(u > _VCAP, 1.0, u)
        if neutron_precursor:
            xn = xn0 * jnp.exp(-t / _TAU_NEUTRON)
            kap = _KAPPA_NEUTRON * (xn0 - xn) + kappa * xr
            heat = er + _EDOT_NEUTRON * xn
        else:
            kap, heat = kappa, er
        td = _K_DIFF * kap * mi / (u * t * beta)
        lum = energy / (td + t * u)
        tau = _K_TAU * kap * mi / (t * u) ** 2
        photo = jnp.argmin(jnp.abs(tau - 1.0))
        inject = (heat * heat_scale).at[0].add(thermalisation_efficiency * ls)
        new = energy + (inject - energy / t - lum) * dt
        return (new, ke), (jnp.sum(lum), u[photo] * t, u0)

    t_nodes = t_all[:-1]
    dt_nodes = jnp.asarray(np.diff(grid), dtype)
    (_, _), (lsum, r_c, u0s) = lax.scan(step, (e0, ke0), (t_nodes, dt_nodes, heat_r[:-1],
                                                          lsd[:-1]))
    if pair_cascade:
        t_life = ((0.6 / (1.0 - ejecta_albedo)) * (pair_cascade_fraction / 0.1) ** 0.5
                  * jnp.sqrt(jnp.maximum(lsd[:-1] * (_ESCALE / 1e45), jnp.finfo(dtype).tiny))
                  * (u0s[-1] / 0.3) ** 0.5
                  * (t_nodes / _DAY) ** -0.5)
        lsum = lsum / (1.0 + t_life)
    logl = jnp.log10(jnp.maximum(lsum, jnp.finfo(lsum.dtype).tiny)) + np.log10(_ESCALE)
    return _finish(logl, r_c)


# --- from the grid to the epochs -----------------------------------------------------------------

def _on_epochs(grid, logl, temp, radius, t_src_s):
    """The engine's output at source-frame times ``t_src_s`` (seconds since the merger).

    ``T`` and ``R`` are interpolated linearly in time, as redback does; ``log10 L`` linearly in
    ``log t``. A time at or before the first node is pre-merger: ``R = 0``, no flux. A time past
    the last node holds the last node's values.
    """
    grid_j = jnp.asarray(grid, temp.dtype)
    t = jnp.asarray(t_src_s)
    live = t > grid_j[0]
    tt = jnp.where(live, t, grid_j[0])
    logl_e = jnp.interp(jnp.log(tt), jnp.log(grid_j), logl)
    return (logl_e, jnp.where(live, jnp.interp(tt, grid_j, temp), temp[0]),
            jnp.where(live, jnp.interp(tt, grid_j, radius), 0.0))


def metzger_bolometric(t_src_s, mej, vej, beta, kappa, *, vmax=0.7, neutron_precursor=True):
    """``(log10 L [erg/s], T [K], R [cm])`` of the Metzger kilonova at source-frame ``t_src_s``.

    The ODE is solved on the fixed :data:`METZGER_GRID`, whatever the epochs, and interpolated.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax import kilonova_metzger as km
    >>> logl, temp, rad = km.metzger_bolometric(np.array([1.0, 3.0]) * 86400.0, 0.03, 0.2, 4.0, 10.0)
    >>> bool(np.all(np.diff(np.asarray(logl)) < 0.0) and np.all(np.asarray(temp) > 0.0))
    True
    """
    out = metzger_engine(METZGER_GRID, mej, vej, beta, kappa, vmax=vmax,
                         neutron_precursor=neutron_precursor)
    return _on_epochs(METZGER_GRID, *out, t_src_s)


def magnetar_boosted_bolometric(t_src_s, mej, vej, beta, kappa, p0, bp, mass_ns, theta_pb,
                                thermalisation_efficiency, **options):
    """:func:`metzger_bolometric` for the magnetar-driven model, on :data:`MAGNETAR_GRID`.

    ``options`` are :func:`magnetar_boosted_engine`'s keywords.
    """
    out = magnetar_boosted_engine(MAGNETAR_GRID, mej, vej, beta, kappa, p0, bp, mass_ns, theta_pb,
                                  thermalisation_efficiency, **options)
    return _on_epochs(MAGNETAR_GRID, *out, t_src_s)


def _ab_magnitude(temp, radius, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, *,
                  mag_floor, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                  r_v_mw=3.1, r_v_host=3.1, law="f99"):
    """AB magnitude of a blackbody ``(T, R)`` per observation: :func:`kilonova.ab_magnitude`'s
    band integral with the photosphere given instead of derived."""
    nu_src = (_C / (lam_obs_ang * 1e-8))[None, :] * (1.0 + redshift)
    f_ab = kn._flux_nu(temp[:, None], radius[:, None], dl_cm, nu_src, redshift, kn.PLANCK_T3_AB)
    f_ab = jax.lax.optimization_barrier(f_ab)   # keep the Planck array out of the band reduction
    tau = kn._ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host, xi_mw, xi_host,
                               r_v_mw, r_v_host, law)
    w = weights if tau is None else weights * tau[None, :]
    num = jnp.sum(f_ab * w[band_idx], axis=1)
    den = jax.lax.optimization_barrier(kn.INV_AB_ZEROPOINT * norms[band_idx])
    ratio_floor = jnp.asarray(10.0, dtype=num.dtype) ** (-0.4 * mag_floor)
    return -2.5 * jnp.log10(jnp.maximum(num / den, ratio_floor))


# --- priors --------------------------------------------------------------------------------------
# redback's prior files (metzger_kilonova_model.prior, metzger_magnetar_driven_kilonova_model.prior,
# 1.20), transcribed for when redback is not importable. Redshift is a factory argument here and
# is dropped; the magnetar file calls the opacity ``kappa_r``, this model ``kappa`` (a
# ``param_aliases`` entry maps it back).
_FALLBACK = {
    "metzger_kilonova_model": {
        "mej": ("Uniform", 1e-2, 0.05), "vej": ("Uniform", 0.1, 0.4),
        "beta": ("Uniform", 3.0, 5.0), "kappa": ("Uniform", 1.0, 30.0)},
    "metzger_magnetar_driven_kilonova_model": {
        "mej": ("Uniform", 1e-3, 0.1), "vej": ("Uniform", 0.1, 0.4),
        "beta": ("Uniform", 3.0, 5.0), "kappa": ("Uniform", 1.0, 30.0),
        "p0": ("LogUniform", 0.7, 1e4), "bp": ("LogUniform", 1e-4, 1e4),
        "mass_ns": ("Uniform", 1.1, 2.2), "theta_pb": ("Uniform", 0.0, 1.57),
        "thermalisation_efficiency": ("Uniform", 0.1, 1.0)},
}
_MAGNETAR_ALIASES = {"kappa": "kappa_r"}


def _redback_prior(model_name, aliases):
    """redback's own prior for ``model_name`` in this module's parameter names, else the
    transcription above (no redback installed). Never narrowed or edited."""
    from ...priors import LogUniform, Prior, Uniform

    try:
        from .tde import redback_prior
        read, pinned = redback_prior(model_name)
        if pinned:
            raise NotImplementedError(f"redback pins {sorted(pinned)} in {model_name}.prior")
        back = {a: k for k, a in aliases.items()}
        return Prior({back.get(k, k): d for k, d in read.distributions.items()})
    except ImportError:
        cls = {"Uniform": Uniform, "LogUniform": LogUniform}
        return Prior({k: cls[c](lo, hi) for k, (c, lo, hi) in _FALLBACK[model_name].items()})


def default_prior_metzger():
    """redback's ``metzger_kilonova_model`` prior: ``mej`` U(0.01, 0.05), ``vej`` U(0.1, 0.4),
    ``beta`` U(3, 5), ``kappa`` U(1, 30). Read from redback when it is installed, else the same
    numbers transcribed."""
    return _redback_prior("metzger_kilonova_model", {})


def default_prior_magnetar_boosted():
    """redback's ``metzger_magnetar_driven_kilonova_model`` prior (redback's ``kappa_r`` is
    ``kappa`` here): ``mej`` U(1e-3, 0.1), ``vej`` U(0.1, 0.4), ``beta`` U(3, 5), ``kappa``
    U(1, 30), ``p0`` LogU(0.7, 1e4) ms, ``bp`` LogU(1e-4, 1e4) x 1e14 G, ``mass_ns`` U(1.1, 2.2),
    ``theta_pb`` U(0, 1.57), ``thermalisation_efficiency`` U(0.1, 1). Read from redback when it is
    installed, else the same numbers transcribed."""
    return _redback_prior("metzger_magnetar_driven_kilonova_model", _MAGNETAR_ALIASES)


# --- the whisper Model ---------------------------------------------------------------------------
from ._factories import (FREE_CONTEXT, _BandIndex, _PhotometricPredict, _PredictJax,   # noqa: E402
                         _check_context, _check_free, _context, _days_since, _filter_set_dict,
                         _floor_note, _fold_extinction, _free_distributions, _free_note,
                         _prior_in_own_names, _register)

_PHYSICAL = {
    "metzger": ["mej", "vej", "beta", "kappa"],
    "magnetar": ["mej", "vej", "beta", "kappa", "p0", "bp", "mass_ns", "theta_pb",
                 "thermalisation_efficiency"],
}


class _MetzgerPredict(_PhotometricPredict):
    """``predict`` / ``predict_jax`` of the two models. See ``_PhotometricPredict``.

    ``kind`` is ``"metzger"`` or ``"magnetar"``; ``options`` are the engine's pinned keywords.
    The ODE grid is a module constant, so unlike the one-component kilonova nothing about the
    epochs reaches the compiled function but their values: a new light curve of the same length
    reuses it, and ``t_exp`` and the redshift trace like any parameter.
    """

    def __init__(self, name, params, band_index, *, kind, options, weights, norms, lam, redshift,
                 dl_cm, t_exp_days, mag_floor, free=(), host_ext=None):
        super().__init__(name, params, band_index)
        self.kind, self.options = kind, dict(options)
        self.free = tuple(free)
        self.weights, self.norms, self.lam = np.asarray(weights), np.asarray(norms), np.asarray(lam)
        self.redshift = None if redshift is None else float(redshift)
        self.dl_cm = None if dl_cm is None else float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.mag_floor = float(mag_floor)
        self.host_ext = dict(host_ext or {})

    def _core(self, times, bidx, *free):
        """Observer-frame days since ``t_exp_days`` -> AB magnitude per observation. Traced."""
        c = _context(self, free)
        v = c.values
        t_src = kn.source_time_s(times - c.shift, c.redshift)
        args = [v[k] for k in _PHYSICAL[self.kind]]
        if self.kind == "metzger":
            grid, out = METZGER_GRID, metzger_engine(METZGER_GRID, *args, **self.options)
        else:
            grid, out = MAGNETAR_GRID, magnetar_boosted_engine(MAGNETAR_GRID, *args, **self.options)
        _, temp, radius = _on_epochs(grid, *out, t_src)
        return _ab_magnitude(temp, radius, bidx, jnp.asarray(self.weights),
                             jnp.asarray(self.norms), jnp.asarray(self.lam), c.redshift, c.dl_cm,
                             mag_floor=self.mag_floor, **self.host_ext)

    def _magnitude(self, times, bidx, free):
        rel = _days_since(times, self.t_exp_days)       # host float64, before any cast
        self._warn_past_grid(rel)
        return self._jitted(None, lambda: self._core)(jnp.asarray(rel), jnp.asarray(bidx), *free)

    def _warn_past_grid(self, rel_days):
        """Epochs past the last node hold its photosphere: say so (concrete epochs, fixed
        redshift and ``t_exp`` only)."""
        if self.free:
            return
        try:
            t_max = float(np.max(np.asarray(rel_days, dtype=np.float64)))
        except (TypeError, ValueError):                 # a tracer, or no epochs
            return
        grid = METZGER_GRID if self.kind == "metzger" else MAGNETAR_GRID
        if t_max * _DAY / (1.0 + self.redshift) > grid[-1]:
            warnings.warn(
                f"model {self.name!r}: epochs past t_src = {grid[-1]:g} s ({grid[-1] / _DAY:.0f} d); "
                f"the model holds its last node's photosphere there.", RuntimeWarning, stacklevel=3)


def _build(kind, what, band_names, redshift, dl_cm, *, name, n_wave, ebv_mw, ebv_host, r_v_mw,
           r_v_host, law, mag_floor, t_exp_days, filter_set, band_aliases, default_system, free,
           prior, redshift_prior, options, description):
    """The shared body of the two factories."""
    from ...models import Model
    from ...priors import Prior

    free = _check_free(free, what)
    _check_context(free, redshift, dl_cm, what)
    band_names = list(band_names)
    fs = _filter_set_dict(band_names, filter_set, n_wave, default_system)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    weights, host_ext = _fold_extinction(kn, weights, fs["lam"], redshift, free, ebv_mw, ebv_host,
                                         r_v_mw, r_v_host, law)
    phys = list(_PHYSICAL[kind])
    params = phys + list(free)
    aliases = dict(_MAGNETAR_ALIASES) if kind == "magnetar" else {}

    ctx = _MetzgerPredict(
        name, params, _BandIndex(band_names, f"{what}(band_names=[...])", band_aliases),
        kind=kind, options=options, weights=weights, norms=norms, lam=lam, redshift=redshift,
        dl_cm=dl_cm, t_exp_days=t_exp_days, mag_floor=mag_floor, free=free, host_ext=host_ext)

    base = default_prior_metzger() if kind == "metzger" else default_prior_magnetar_boosted()
    dists = dict(base.distributions)
    if prior is not None:
        own = _prior_in_own_names(prior, aliases, what).distributions
        unknown = [k for k in own if k not in params]
        if unknown:
            raise ValueError(f"{what}: prior= names {unknown}, which are not parameters of this "
                             f"model ({params}).")
        dists.update({k: d for k, d in own.items() if k in phys})
    dists.update(_free_distributions(free, prior, redshift_prior, what))
    return Model(name=name, predict=ctx.predict, parameters=params,
                 default_prior=Prior({k: dists[k] for k in params}),
                 predict_jax=_PredictJax(ctx), param_aliases=aliases,
                 description=f"{description} {_free_note(free)}{_floor_note(mag_floor)}")


_COMMON_DOC = """
    Parameters
    ----------
    band_names : sequence of str
        Filter names or labels (``lsstg``, ``ztfr``, ``sdssi``, an SVO ID, or a bare letter read in
        ``default_system``), in the order the light curve's ``bands`` column maps to.
    redshift, dl_cm : float
        Fixed for the dataset; ``dl_cm`` is not derived from ``redshift`` here (the cosmology is the
        caller's). Leave both ``None`` when the redshift is fitted (``free=["redshift"]``).
    name : str
        The ``Model`` name (and the registry name of the ``register_`` twin).
    free : list of str or None
        ``"t_exp"`` (merger time on the light curve's clock) and/or ``"redshift"`` (the distance
        follows it, Planck18), fitted as ordinary parameters appended in that order.
        ``t_exp`` needs ``prior=Prior({"t_exp": ...})``. The ODE grid does not depend on either.
    prior : whisper_cbpf.priors.Prior or None
        Distributions replacing the defaults, per parameter (``t_exp``, ``redshift`` included).
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none.
    mag_floor : float
        Magnitudes are capped at this value (a pre-merger epoch is exactly at it).
    t_exp_days : float
        The merger time on the light curve's clock when it is fixed (default 0).
    ebv_mw, ebv_host, r_v_mw, r_v_host, law :
        Optional Milky Way / host extinction, as :func:`whisper_cbpf.models.jax.kilonova_model`.
    filter_set, n_wave, band_aliases, default_system :
        The band integral and the band-name lookup, as :func:`whisper_cbpf.models.jax.kilonova_model`.
    vmax : float
        Maximum shell velocity, units of c (redback: 0.7).
    neutron_precursor : bool
        Free-neutron heating and opacity of the fastest layers (redback: on).
"""


def metzger_kilonova_model(band_names, redshift=None, dl_cm=None, *, name="metzger_kilonova_jax",
                           n_wave=None, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1,
                           law="f99", mag_floor=40.0, t_exp_days=0.0, filter_set=None,
                           band_aliases=None, default_system=None, free=None, prior=None,
                           redshift_prior=None, vmax=0.7, neutron_precursor=True):
    """Bind dataset context to the Metzger (2017) multi-shell kilonova; return a ``Model``.

    redback's ``metzger_kilonova_model``: 200 velocity shells between ``vej`` and ``vmax`` with
    ``m(>v) ~ v^-beta``, r-process heating through Barnes & Kasen (2016) thermalisation, a
    free-neutron precursor in the fastest layers, per-shell diffusion, photosphere at unit optical
    depth. Parameters (all in redback's names and units): ``mej`` [Msun], ``vej`` [c], ``beta``,
    ``kappa`` [cm^2/g]; default prior redback's ``metzger_kilonova_model.prior``
    (``mej`` U(0.01, 0.05), ``vej`` U(0.1, 0.4), ``beta`` U(3, 5), ``kappa`` U(1, 30)).

    It is not the one-component kilonova (:func:`whisper_cbpf.models.jax.kilonova_model`): a
    different model with different parameters, not comparable parameter for parameter.

    The SED is redback's: a blackbody at the photosphere temperature and radius, **no temperature
    floor**, ``(1 + z)`` dilation. The ODE is solved once per parameter set on a fixed grid of
    :data:`N_METZGER` nodes from 1e-2 s to 1e7 s (116 d), independent of the epochs; ``T`` and
    ``R`` are interpolated to them as redback does. Epochs past the grid hold the last node.
    Discretisation error against a converged solution: 0.003 dex median, 0.009 dex maximum in
    ``L`` (redback's 500-node grid: 0.014 / 0.05).

    float32 is safe (within 5e-6 dex of float64 in ``L``); float64 is not required.
    {common}
    References
    ----------
    Metzger, B. D. 2017, Living Reviews in Relativity, 20, 3 (Eq. 25, the thermalisation
    efficiency fits of Barnes et al. 2016); Barnes, J. & Kasen, D. 2016, ApJ, 829, 110
    (thermalisation); Metzger, B. D., et al. 2010, MNRAS, 406, 2650 (neutron-rich ejecta and
    the neutron precursor); Tanaka, M., et al. 2019, the electron fraction against opacity used
    by redback; Sarin, N., et al., redback (GPL-3.0), ``metzger_kilonova_model``.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax import kilonova_metzger as km
    >>> m = km.metzger_kilonova_model(["lsstg", "lsstr"], redshift=0.0098, dl_cm=1.3e26)
    >>> m.parameters
    ['mej', 'vej', 'beta', 'kappa']
    >>> p = dict(mej=0.03, vej=0.2, beta=4.0, kappa=10.0)
    >>> flux = m.predict(p, np.array([1.0, 3.0]), np.array(["lsstg", "lsstr"]))
    >>> bool(np.all(np.isfinite(flux)) and np.all(flux > 0.0))
    True
    """
    return _build("metzger", "metzger_kilonova_model", band_names, redshift, dl_cm, name=name,
                  n_wave=n_wave, ebv_mw=ebv_mw, ebv_host=ebv_host, r_v_mw=r_v_mw,
                  r_v_host=r_v_host, law=law, mag_floor=mag_floor, t_exp_days=t_exp_days,
                  filter_set=filter_set, band_aliases=band_aliases, default_system=default_system,
                  free=free, prior=prior, redshift_prior=redshift_prior,
                  options=dict(vmax=vmax, neutron_precursor=neutron_precursor),
                  description=("Metzger (2017) multi-shell kilonova (JAX/GPU): redback's "
                               "metzger_kilonova_model, explicit Euler on a fixed "
                               f"{N_METZGER}-node grid, blackbody photosphere without a "
                               "temperature floor."))


def magnetar_boosted_kilonova_model(band_names, redshift=None, dl_cm=None, *,
                                    name="metzger_magnetar_driven_kilonova_jax", n_wave=None,
                                    ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1,
                                    law="f99", mag_floor=40.0, t_exp_days=0.0, filter_set=None,
                                    band_aliases=None, default_system=None, free=None, prior=None,
                                    redshift_prior=None, vmax=0.7, neutron_precursor=True,
                                    pair_cascade=True, pair_cascade_fraction=0.01,
                                    ejecta_albedo=0.5):
    """Bind dataset context to the magnetar-boosted Metzger kilonova; return a ``Model``.

    redback's ``metzger_magnetar_driven_kilonova_model`` (``magnetar_heating='first_layer'``): the
    ejecta of :func:`metzger_kilonova_model` (mass array normalised to ``mej``) with a central
    magnetar whose dipole spin-down ``2 E_rot/t_p (1 + 2t/t_p)^-2`` heats the innermost shell; the
    kinetic energy and every shell velocity evolve with the pdV work, and the luminosity is
    suppressed by redback's pair-cascade factor ``1 / (1 + t_life/t)``.

    Parameters (redback's, with ``kappa`` for redback's ``kappa_r``): ``mej`` [Msun], ``vej`` [c],
    ``beta``, ``kappa`` [cm^2/g], ``p0`` [ms], ``bp`` [1e14 G], ``mass_ns`` [Msun], ``theta_pb``
    [rad], ``thermalisation_efficiency``. Default prior redback's
    ``metzger_magnetar_driven_kilonova_model.prior``: ``mej`` U(1e-3, 0.1), ``vej`` U(0.1, 0.4),
    ``beta`` U(3, 5), ``kappa`` U(1, 30), ``p0`` LogU(0.7, 1e4), ``bp`` LogU(1e-4, 1e4),
    ``mass_ns`` U(1.1, 2.2), ``theta_pb`` U(0, 1.57), ``thermalisation_efficiency`` U(0.1, 1).
    The prior spans eight decades in ``bp`` and four in ``p0``; most of it is a weak magnetar.

    SED, time grid and epoch handling are :func:`metzger_kilonova_model`'s (a blackbody from
    ``(T, R_ph)``, no temperature floor, ``(1 + z)`` dilation), on a fixed grid of
    :data:`N_MAGNETAR` nodes from 1e-4 s to 1e7 s. Discretisation error against a converged
    solution: 0.0016 dex median, 0.011 dex maximum in ``L`` (redback's 300-node grid: 0.03 / 0.13).
    float32 is safe (energies are carried in units of 1e40 erg).

    redback's model is internally inconsistent in two ways that are reproduced, not repaired:
    its null-magnetar limit is *not* :func:`metzger_kilonova_model` (the magnetar model evolves
    total shell energies, the Metzger model per-gram energies), and the pair-cascade factor uses
    the ejecta velocity of the last time step for every epoch.
    {common}
    pair_cascade : bool
        Apply the pair-cascade suppression (redback: on).
    pair_cascade_fraction, ejecta_albedo : float
        Its parameters (redback: 0.01 and 0.5).

    References
    ----------
    Metzger, B. D. 2017, Living Reviews in Relativity, 20, 3; Metzger, B. D. & Piro, A. L. 2014,
    MNRAS, 439, 3916 (magnetar-boosted kilonovae); Barnes, J. & Kasen, D. 2016, ApJ, 829, 110
    (thermalisation); Metzger, B. D., et al. 2010, MNRAS, 406, 2650 (neutron precursor);
    Sarin, N., et al., redback (GPL-3.0), ``metzger_magnetar_driven_kilonova_model``.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax import kilonova_metzger as km
    >>> m = km.magnetar_boosted_kilonova_model(["lsstg", "lsstr"], redshift=0.0098, dl_cm=1.3e26)
    >>> m.parameters[-3:]
    ['mass_ns', 'theta_pb', 'thermalisation_efficiency']
    >>> p = dict(mej=0.03, vej=0.2, beta=4.0, kappa=10.0, p0=2.0, bp=1.0, mass_ns=1.4,
    ...          theta_pb=1.0, thermalisation_efficiency=0.5)
    >>> flux = m.predict(p, np.array([1.0, 3.0]), np.array(["lsstg", "lsstr"]))
    >>> bool(np.all(np.isfinite(flux)) and np.all(flux > 0.0))
    True
    """
    return _build("magnetar", "magnetar_boosted_kilonova_model", band_names, redshift, dl_cm,
                  name=name, n_wave=n_wave, ebv_mw=ebv_mw, ebv_host=ebv_host, r_v_mw=r_v_mw,
                  r_v_host=r_v_host, law=law, mag_floor=mag_floor, t_exp_days=t_exp_days,
                  filter_set=filter_set, band_aliases=band_aliases, default_system=default_system,
                  free=free, prior=prior, redshift_prior=redshift_prior,
                  options=dict(vmax=vmax, neutron_precursor=neutron_precursor,
                               pair_cascade=pair_cascade,
                               pair_cascade_fraction=pair_cascade_fraction,
                               ejecta_albedo=ejecta_albedo),
                  description=("Magnetar-boosted Metzger kilonova (JAX/GPU): redback's "
                               "metzger_magnetar_driven_kilonova_model (first-layer heating, "
                               "kinetic-energy evolution, pair cascade), explicit Euler on a fixed "
                               f"{N_MAGNETAR}-node grid, blackbody photosphere without a "
                               "temperature floor."))


metzger_kilonova_model.__doc__ = metzger_kilonova_model.__doc__.replace("{common}", _COMMON_DOC)
magnetar_boosted_kilonova_model.__doc__ = magnetar_boosted_kilonova_model.__doc__.replace(
    "{common}", _COMMON_DOC)


def register_metzger_kilonova(band_names, redshift=None, dl_cm=None, *,
                              name="metzger_kilonova_jax", **kwargs):
    """Build a :func:`metzger_kilonova_model` for this dataset and register it under ``name``.

    Parameters are :func:`metzger_kilonova_model`'s; an existing registration of ``name`` is
    replaced.

    Examples
    --------
    >>> from whisper_cbpf.models.jax import kilonova_metzger as km
    >>> m = km.register_metzger_kilonova(["lsstg", "lsstr"], redshift=0.0098, dl_cm=1.3e26)
    >>> import whisper_cbpf as wp
    >>> m.name in wp.list_models()
    True
    """
    return _register(metzger_kilonova_model(band_names, redshift, dl_cm, name=name, **kwargs))


def register_magnetar_boosted_kilonova(band_names, redshift=None, dl_cm=None, *,
                                       name="metzger_magnetar_driven_kilonova_jax", **kwargs):
    """Build a :func:`magnetar_boosted_kilonova_model` for this dataset and register it.

    Parameters are :func:`magnetar_boosted_kilonova_model`'s; an existing registration of
    ``name`` is replaced.

    Examples
    --------
    >>> from whisper_cbpf.models.jax import kilonova_metzger as km
    >>> m = km.register_magnetar_boosted_kilonova(["lsstg", "lsstr"], redshift=0.0098, dl_cm=1.3e26)
    >>> import whisper_cbpf as wp
    >>> m.name in wp.list_models()
    True
    """
    return _register(magnetar_boosted_kilonova_model(band_names, redshift, dl_cm, name=name,
                                                     **kwargs))
