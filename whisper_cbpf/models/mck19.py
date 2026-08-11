"""``mck19`` -- electromagnetic flare from a binary-black-hole merger in an AGN disk.

Physical model of the optical counterpart of a BBH merger embedded in an AGN accretion disk: the
gravitational-wave recoil ("kick") drives the remnant black hole through the disk gas, producing a
shocked, bound-gas **hotspot** that radiates as a blackbody. The light curve is a delayed flare on top
of the AGN disk baseline -- a ``sin^2`` rise to peak after the ram-pressure delay ``t_ram``, then an
exponential decay back to the disk level.

Reference: McKernan et al. 2019, ApJL 884, L50 (https://iopscience.iop.org/article/10.3847/2041-8213/ab4886).
This implementation follows the code of **Darc 2025** (https://arxiv.org/abs/2506.02224).

This is a **band-dependent** model: it returns flux density (Jy) at each observed ``(time, band)``,
computing the hotspot + disk blackbody at each band's effective wavelength (resolved through Whisper's
band system). It is self-contained (astropy constants + cosmology only). The original used `speclite`
LSST filter integration; here the AB magnitude is evaluated monochromatically at each band's effective
wavelength -- a documented approximation that keeps the model dependency-free.

Parameters
----------
v_kick : float
    GW recoil kick velocity of the remnant [km/s].
M_smbh : float
    Supermassive black hole (disk host) mass [solar masses].
M_bh : float
    Remnant black hole mass [solar masses].
r_bh : float
    Orbital radius of the BBH in the disk [gravitational radii ``R_g``].
redshift : float
    Source redshift (sets the luminosity distance via Planck18 and the time dilation / k-correction).

Notes & limitations
-------------------
* The flux is the **total** disk + hotspot (not baseline-subtracted). Where the hotspot is fainter than
  the disk in a band, that band shows no flare (it stays at the disk level).
* ``t = 0`` is the merger; the flare peaks at the (observer-frame) delay ``t_ram``.
* Disk structure constants (Sirko & Goodman / Shakura-Sunyaev) follow Darc 2025; ``mdot = 0.05`` Edd
  and ``alpha = 0.1`` are fixed (not free parameters -- they do not enter the light-curve *shape*).
"""
from __future__ import annotations

import warnings

import numpy as np
import astropy.units as u
from astropy.constants import G as _G, M_sun as _Msun, c as _c, h as _h, k_B as _kB
from astropy.cosmology import Planck18

from ..priors import LogUniform, Prior, Uniform

# --- physical constants in CGS (floats; the hot path avoids astropy Quantity for speed) ---
_G_CGS = _G.cgs.value          # cm^3 g^-1 s^-2
_C_CGS = _c.cgs.value          # cm/s
_H_CGS = _h.cgs.value          # erg s
_KB_CGS = _kB.cgs.value        # erg/K
_MSUN_G = _Msun.cgs.value      # g
_JY_CGS = 1e-23                # 1 Jy = 1e-23 erg s^-1 cm^-2 Hz^-1
_AB_ZP_JY = 3631.0             # AB zero point
_MDOT_EDD = 0.05               # disk accretion rate [Eddington] (fixed)
_EXP_CLIP = 700.0              # keep exp() finite (just under float64's exp overflow ~709)
_DEFAULT_LAMBDA_AA = 6000.0    # fallback effective wavelength for unresolved bands

# Redshift -> luminosity distance (cm), precomputed + interpolated so predict() stays fast.
_Z_GRID = np.linspace(1e-3, 3.0, 800)
_DL_GRID_CM = Planck18.luminosity_distance(_Z_GRID).to_value(u.cm)


PARAMETERS = ["v_kick", "M_smbh", "M_bh", "r_bh", "redshift"]
DESCRIPTION = "McKernan 2019 BBH-in-AGN-disk kicked-hotspot flare (Darc 2025 implementation)."

#: Default prior spanning the physically motivated grid of Darc 2025.
PRIOR = Prior({
    "v_kick": Uniform(100.0, 800.0),       # km/s
    "M_smbh": LogUniform(1e6, 1e9),        # solar masses
    "M_bh": Uniform(20.0, 160.0),          # solar masses
    "r_bh": LogUniform(500.0, 10000.0),    # gravitational radii
    "redshift": Uniform(0.01, 1.0),
})


def _lum_distance_cm(z):
    return float(np.interp(z, _Z_GRID, _DL_GRID_CM))


def _r_g_cm(m_smbh):
    """Gravitational radius R_g = G M / c^2 [cm]."""
    return _G_CGS * (m_smbh * _MSUN_G) / _C_CGS**2


def _temperature_flare(v_kick):
    """Hotspot temperature [K]: 1e5 K * (v_kick / 100 km/s)."""
    return 1e5 * (v_kick / 100.0)


def _temperature_disk(m_smbh, r_bh):
    """Unperturbed AGN-disk temperature at r_bh [K]."""
    return 6e5 * (m_smbh / 1e8) ** 0.25 * (_MDOT_EDD ** 0.25) * r_bh ** (-0.75)


def _hill_radius_rg(m_smbh, m_bh, r_bh):
    """Hill radius [R_g]: r_bh * (q/3)^(1/3) with q = M_bh / M_smbh."""
    q = m_bh / m_smbh
    return r_bh * (q / 3.0) ** (1.0 / 3.0)


def _bounding_gas_radius_rg(m_smbh, m_bh, r_bh, v_kick):
    """Radius of bound gas (emitting hotspot) [R_g] (ZTF/Graham 2020 assumptions)."""
    q = m_bh / m_smbh
    factor = 0.34 * (q / 1e-6) ** (2.0 / 3.0) * (r_bh / 1e3) ** (-1.0) * (v_kick / 200.0) ** (-2.0)
    return _hill_radius_rg(m_smbh, m_bh, r_bh) * factor


def _t_ram_days(m_bh, v_kick, z):
    """Ram-pressure delay / flare peak time (observer frame) [days]."""
    return 20.0 * (m_bh / 100.0) * (v_kick / 200.0) ** (-3.0) * (1.0 + z)


def _duration_days(m_smbh, m_bh, r_bh, v_kick):
    """Flare decay timescale (rest frame) [days]: hotspot crossing time R_hill / v_kick."""
    rh_cm = _hill_radius_rg(m_smbh, m_bh, r_bh) * _r_g_cm(m_smbh)
    rh_km = rh_cm / 1e5
    return (rh_km / v_kick) / 86400.0


def _ab_mag_blackbody(lambda_obs_aa, temperature, r_emit_cm, dl_cm, z):
    """AB magnitude of a blackbody sphere (radius ``r_emit_cm``) at the observed wavelength(s)."""
    lam_obs_cm = np.asarray(lambda_obs_aa, dtype=float) * 1e-8
    lam_rest_cm = lam_obs_cm / (1.0 + z)
    x = np.clip(_H_CGS * _C_CGS / (lam_rest_cm * _KB_CGS * temperature), None, _EXP_CLIP)
    b_lambda = (2.0 * _H_CGS * _C_CGS**2 / lam_rest_cm**5) / np.expm1(x)   # specific intensity B_lambda
    # Observed flux of a uniform sphere radius r at distance D: F = pi*B*(r/D)^2 (the pi converts
    # specific intensity B to the emergent surface flux pi*B). Darc 2025 / the reference omit this pi,
    # making every predicted flux a factor of pi (~1.24 mag) too faint; restore it for physical fluxes.
    f_lambda_obs = np.pi * b_lambda * (r_emit_cm / dl_cm) ** 2 / (1.0 + z)
    f_nu = f_lambda_obs * lam_obs_cm**2 / _C_CGS                          # erg s^-1 cm^-2 Hz^-1
    f_nu_jy = np.maximum(f_nu / _JY_CGS, 1e-300)
    return -2.5 * np.log10(f_nu_jy / _AB_ZP_JY)


def _flare_shape(t, t_peak, tau_decay):
    """Temporal profile: 0 (t<=0), sin^2 rise to 1 at ``t_peak``, then exp decay (timescale tau)."""
    t = np.asarray(t, dtype=float)
    s = np.zeros_like(t)
    if t_peak > 0:
        rise = (t > 0) & (t <= t_peak)
        s[rise] = np.sin(0.5 * np.pi * t[rise] / t_peak) ** 2
    dec = t > t_peak
    if tau_decay > 0:
        s[dec] = np.exp(-np.clip((t[dec] - t_peak) / tau_decay, None, _EXP_CLIP))
    return s


def _effective_wavelengths(bands):
    """Per-point observed effective wavelength [Angstrom] from Whisper's band system (NaN -> default)."""
    from ..io.bands import resolve_bands
    lam, _, _ = resolve_bands(np.asarray(bands), svo_fallback=False, warn=False)
    return np.where(np.isfinite(lam), lam, _DEFAULT_LAMBDA_AA)


def mck19_flux(parameters, times, bands=None):
    """Predicted flux density [Jy] at each ``(time, band)`` for the McKernan-2019 AGN-disk flare.

    ``times`` are days since the merger; ``bands`` selects the effective wavelength per point. See the
    module docstring for the parameters and physics.
    """
    v_kick = float(parameters["v_kick"])
    m_smbh = float(parameters["M_smbh"])
    m_bh = float(parameters["M_bh"])
    r_bh = float(parameters["r_bh"])
    z = float(parameters["redshift"])

    t = np.asarray(times, dtype=float)
    if bands is None:
        warnings.warn("mck19 is band-dependent; no bands given -> assuming r-band wavelength.", stacklevel=2)
        lam_obs = np.full(t.shape, _DEFAULT_LAMBDA_AA)
    else:
        lam_obs = _effective_wavelengths(bands)

    r_g = _r_g_cm(m_smbh)
    dl_cm = _lum_distance_cm(z)
    r_emit_flare = _bounding_gas_radius_rg(m_smbh, m_bh, r_bh, v_kick) * r_g
    r_emit_disk = r_bh * r_g
    t_peak = _t_ram_days(m_bh, v_kick, z)
    tau_decay = _duration_days(m_smbh, m_bh, r_bh, v_kick) * (1.0 + z)

    peak_mag = _ab_mag_blackbody(lam_obs, _temperature_flare(v_kick), r_emit_flare, dl_cm, z)
    disk_mag = _ab_mag_blackbody(lam_obs, _temperature_disk(m_smbh, r_bh), r_emit_disk, dl_cm, z)

    # Only an EM brightening on top of the disk (no "negative flare" where the hotspot is fainter).
    brighten = np.maximum(disk_mag - peak_mag, 0.0)
    mag_lc = disk_mag - brighten * _flare_shape(t, t_peak, tau_decay)
    return _AB_ZP_JY * np.power(10.0, -0.4 * mag_lc)
