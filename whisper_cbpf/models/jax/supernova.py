"""JAX implementation of redback's self-contained supernova models.

Depends on jax and numpy only. The diffusion kernel, the photosphere and the AB-photometry layer
come from :mod:`whisper_cbpf.models.jax.tde` and :mod:`whisper_cbpf.models.jax.kilonova`.

Twelve model names over nine engines. Engines: ``_nickelcobalt_engine``, ``basic_magnetar``,
``magnetar_only``, ``exponential_powerlaw``, ``fallback_lbol``, ``_shock_cooling`` (Piro+ 2021) and
``_csm_shock_breakout`` (Margalit 2022). Bolometric models: ``arnett``,
``shock_cooling_and_arnett``, ``basic_magnetar_powered`` (== ``slsn``), ``magnetar_nickel``,
``csm_shock_and_arnett``, ``exponential_powerlaw``, ``sn_fallback``, ``sn_nickel_fallback``,
``general_magnetar_slsn``, ``type_1a`` (== ``arnett``) and ``type_1c`` (== ``arnett``). SEDs:
blackbody, cutoff blackbody, line, synchrotron.

Everything here is vectorised arithmetic -- no ODE, no sequential loop, no root-find -- so the
family is cheap, batches perfectly, and differentiates end to end.

**float64 is required**, for a different reason than the TDE's: these are bolometric luminosities
in cgs, and float32 stops at 3.403e38. :func:`build_sn_grid` raises rather than returning a light
curve of exactly ``mag_floor``::

    import jax; jax.config.update("jax_enable_x64", True)   # before the first array exists

``magnetar_convention`` and ``interaction`` select between redback releases that disagree by a
magnitude or more; ``dilation`` selects the ``(1+z)`` flux factor; ``build_sn_grid``'s ``spacing``
selects redback 1.20's geometric dense grids (default) or the linear ones of redback <= 1.15
(:data:`REDBACK_GRID_PRESETS`). Those, the CSM breakout's interpolation, and every other deviation
are in ``docs/PORTING_NOTES.md``.
"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .kilonova import (
    AB_ZEROPOINT,
    BOLTZMANN,
    DAY_TO_S,
    DLSCALE,
    INV_AB_ZEROPOINT,
    LSCALE,
    MJY,
    PLANCK,
    PLANCK_T3_AB,
    SIGMA_SB,
    SOLAR_MASS,
    SPEED_OF_LIGHT,
    _ext_transmission,
    ab_weights,           # noqa: F401  (re-exported: callers build weights with it)
    extinction_shape,     # noqa: F401
    make_filter_set,      # noqa: F401
)
from .tde import (
    _flux_nu_tde,
    ab_magnitude,
    build_interaction_grid,
    diffusion,
    flux_density_mjy,
    temperature_floor_photosphere,
)

__all__ = [
    "build_sn_grid", "dense_grid", "DENSE_RESOLUTION", "TIME_PAD", "GRID_START", "CSM_NODES",
    "REDBACK_GRID_PRESETS", "warn_if_spin_down_unresolved", "SPIN_DOWN_UNRESOLVED_LIMIT",
    # engines
    "nickelcobalt_engine", "basic_magnetar", "magnetar_only", "exponential_powerlaw_engine",
    "fallback_lbol", "shock_cooling", "csm_shock_breakout",
    "exponential_powerlaw_integrable",
    # bolometric
    "arnett_bolometric", "shock_cooling_and_arnett_bolometric",
    "basic_magnetar_powered_bolometric", "slsn_bolometric", "magnetar_nickel_bolometric",
    "csm_shock_and_arnett_bolometric", "exponential_powerlaw_bolometric",
    "sn_fallback_bolometric", "sn_nickel_fallback_bolometric",
    "general_magnetar_slsn_bolometric", "type_1a_bolometric", "type_1c_bolometric",
    # SED layer
    "cutoff_norm", "cutoff_shape", "line_band_term", "synchrotron_f_nu",
    # photometry
    "photosphere", "sn_flux_density", "sn_ab_magnitude",
    # observed-epoch path on a fixed-resolution grid
    "FixedGrid", "build_fixed_grid", "bolometric_at", "ab_magnitude_at",
    "FIXED_GRID_EPOCHS_PER_DECADE", "FIXED_GRID_FIRST", "FIXED_GRID_MARGIN_EPOCHS",
    "DEFAULT_MAX_PHASE_DAYS",
    # registry / priors
    "MODELS", "PARAMETERS", "model_names", "bolometric", "flux_density", "ab_magnitude_of",
    "redback_prior", "fallback_prior", "default_prior",
]

#: redback's ``dense_resolution`` kwarg default, and the ``+100`` padding on the last epoch.
DENSE_RESOLUTION = 1000
TIME_PAD = 100.0
#: First node of redback 1.20's geometric ``ip.Diffusion`` grid, days (CHANGE 1).
GRID_START = 1e-5
#: redback's CSM shock-breakout nodes, ``(first [d], last [d], count)``, fixed and independent of
#: the data (``shock_powered_models.py:544``; CHANGE 4).
CSM_NODES = (1e-2, 200.0, 300)

#: The dense grids of each redback release (CHANGE 1, CHANGE 4). Pass as
#: ``build_sn_grid(t, **REDBACK_GRID_PRESETS["1.15"])``. 1.20 (from 1.18) spaces both grids
#: geometrically; 1.12 and 1.15 space both linearly. The default is 1.20's.
REDBACK_GRID_PRESETS = {
    "1.12": dict(spacing="linear"),
    "1.15": dict(spacing="linear"),
    "1.20": dict(spacing="geometric"),
}

#: :func:`warn_if_spin_down_unresolved` warns once the spin-down releases more than this
#: fraction of ``E_rot`` before the dense grid's first resolved node. See CHANGE 1.
SPIN_DOWN_UNRESOLVED_LIMIT = 0.1

#: Floor for a quantity about to be raised to a fractional power. Mirrors ``tde._L_FLOOR``:
#: ``0 ** 0.25`` is 0 in the forward pass but its derivative is infinite, so an unfloored
#: version returns NaN gradients from points that contribute nothing. No forward value above
#: the floor moves.
_FLOOR = 1e-30

#: redback ``sed.CutoffBlackbody`` class constants (sed.py:302-303), rebuilt from the same
#: astropy-cgs literals :mod:`whisper_cbpf.models.jax.kilonova` carries rather than hardcoded, so a
#: constant can never drift between the two modules.
X_CONST = PLANCK * SPEED_OF_LIGHT / BOLTZMANN        # 1.4387768775 cm K
ANGSTROM_CGS = 1e-8
#: ``FLUX_CONST / angstrom_cgs`` == 8 pi^2 h c^2, folded so the 1e-8 is never a divide.
FLUX_CONST_OVER_ANG = 8.0 * np.pi ** 2 * PLANCK * SPEED_OF_LIGHT ** 2
#: The n = 1..10 term of redback's geometric expansion of 1/(e^u - 1), pre-multiplied by X_CONST.
#: NumPy float64, not a jnp array: one built here would take the dtype in force at IMPORT, and a
#: module imported before x64 is enabled froze it in float32 (7.6e-8 relative in the cutoff SED).
_NXCS = X_CONST * np.arange(1, 11)

#: Scale for the additive Line term, folded so that nothing intermediate can overflow float32
#: (CHANGE 8). The quantity wanted is
#:     f_nu_line / AB_ZEROPOINT = A(t) L / (4 pi dl^2) / (sigma_lam sqrt(2 pi))
#:                                * profile(lam) * lam_cm^2 / c / AB_ZEROPOINT
#: Written directly, ``4 pi dl^2`` is 1.3e55 and ``L`` is 1e43 -- both past float32's 3.4e38.
#: Carrying L in LSCALE units and dl in DLSCALE units, and folding lam_cm^2 = lam_A^2 * 1e-16,
#: leaves LINE_AB_CONST = 7.3e-21 and every variable O(1e8) or smaller.
LINE_AB_CONST = (LSCALE * ANGSTROM_CGS ** 2
                 / (4.0 * np.pi * SPEED_OF_LIGHT * DLSCALE ** 2 * AB_ZEROPOINT))


def _require_x64(what):
    """CHANGE 8. Fail at trace time with the fix, rather than returning 40.0 mag later."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} requires float64. These engines are bolometric luminosities in cgs -- "
            f"1e43 to 1e46 erg/s -- and float32 stops at 3.4e38. Measured: EVERY engine in "
            f"the family returns inf, including at the faintest corner of redback's prior "
            f"(f_nickel=1e-3, mej=1e-4, L = 7.9e36), because ni56_lum = 6.45e43 is formed "
            f"before the nickel mass multiplies it. The guards then turn inf into 0 and every "
            f"band comes back at mag_floor. Enable it BEFORE the first jax array is created:\n"
            f"    import jax; jax.config.update('jax_enable_x64', True)\n"
            f"or set JAX_ENABLE_X64=1 in the environment.")


def _grid_nodes(first, last, n, spacing):
    """``np.geomspace`` (redback 1.20) or ``np.linspace`` (redback <= 1.15) from first to last."""
    if spacing == "geometric":
        return np.geomspace(first, last, n)
    if spacing == "linear":
        return np.linspace(first, last, n)
    raise ValueError(f"spacing must be 'geometric' or 'linear', got {spacing!r}")


def dense_grid(time_days, dense_resolution=DENSE_RESOLUTION, t_pad=TIME_PAD, spacing="geometric"):
    """The NumPy ``ip.Diffusion`` dense grid in days: ``geomspace(1e-5, time[-1] + t_pad, n)``
    (redback 1.20) or ``linspace(0, time[-1] + t_pad, n)`` (``spacing="linear"``, <= 1.15)."""
    first = GRID_START if spacing == "geometric" else 0.0
    return _grid_nodes(first, np.asarray(time_days, dtype=np.float64)[-1] + t_pad,
                       dense_resolution, spacing)


def build_sn_grid(time_days, dense_resolution=DENSE_RESOLUTION, t_pad=TIME_PAD,
                  spacing="geometric", csm_interp=True):
    """Setup-time (NumPy) companion to every model here. Build once per dataset, reuse.

    Reproduces redback 1.20's

        dense_times = np.geomspace(1e-5, time[-1] + 100, dense_resolution)

    plus the ``uniq_times`` / ``searchsorted`` bookkeeping ``ip.Diffusion`` needs (CHANGE 1), and
    the CSM breakout's fixed ``np.geomspace(1e-2, 200, 300)`` nodes (CHANGE 4).
    ``spacing="linear"`` gives redback <= 1.15's ``np.linspace(0, time[-1] + 100, ...)`` and
    ``np.linspace(1e-2, 200, 300)`` instead -- or use :data:`REDBACK_GRID_PRESETS`.
    The shapes here set the compiled kernel size, so a change of ``dense_resolution`` or of
    the number of observations triggers one recompile and nothing else.

    Parameters
    ----------
    time_days : SOURCE-frame observation epochs in days, ASCENDING. redback uses ``time[-1]``
        to size the grid, so the order matters even though nothing else here depends on it;
        this function does not sort for you, precisely so that a caller passing unsorted
        times gets redback's answer rather than a quietly different one.
    spacing : ``"geometric"`` (default, redback 1.20) or ``"linear"`` (redback <= 1.15).
    csm_interp : ``True`` (default) interpolates the CSM breakout off redback's 300 nodes, as
        redback does. ``False`` evaluates its closed form at the epochs, which is exact and was
        this module's only behaviour up to whisper 0.1.0 (CHANGE 4).

    Returns
    -------
    dict with ``time``, ``dense_times``, ``uniq_times``, ``gather_idx``, ``tb``, and
    ``csm_times`` when ``csm_interp``.
    """
    _require_x64("whisper_cbpf.models.jax.supernova")
    time_days = np.asarray(time_days, dtype=np.float64)
    if time_days.ndim != 1 or time_days.size == 0:
        raise ValueError(f"time_days must be a non-empty 1-D array, got shape {time_days.shape}")
    dense_times = dense_grid(time_days, dense_resolution, t_pad, spacing)
    uniq_times, gather_idx, tb = build_interaction_grid(time_days, dense_times)
    grid = dict(time=jnp.asarray(time_days), dense_times=jnp.asarray(dense_times),
                uniq_times=uniq_times, gather_idx=gather_idx, tb=tb)
    if csm_interp:
        grid["csm_times"] = jnp.asarray(_grid_nodes(*CSM_NODES, spacing))
    return grid


def warn_if_spin_down_unresolved(dense_times, p0, bp, mass_ns, theta_pb, convention="1.15",
                                 limit=SPIN_DOWN_UNRESOLVED_LIMIT):
    """Warn when the magnetar spin-down is too fast for the dense grid. Host-side NumPy.

    Returns the fraction of ``E_rot`` released before the grid's first resolved node --
    ``dense_times[0]``, or ``dense_times[1]`` when the grid starts at 0 -- and warns when it
    exceeds ``limit``. ``basic_magnetar`` releases ``E(<t) = E_rot x/(1+x)`` with ``x = 2t/t_p``
    (``"1.15"``) or ``t/t_p`` (``"1.12"``). No quadrature on the grid places that energy right:
    redback 1.20's geometric grid never integrates ``[0, 1e-5 d]``, so it is lost (too FAINT: 45%
    of ``E_rot`` delivered at ``t_p = 1.43 s``, 5.7% at 0.105 s), and the linear grid of redback
    <= 1.15 draws one straight line from ``L(0)`` across its first cell, delivering ``~dt/t_p``
    times ``E_rot`` (too BRIGHT). Matching redback does not make these curves right, so it is
    said rather than refined away (CHANGE 1). Under ``jit`` the parameters are traced and there
    is nothing to test.

    **Where it runs: only in the supernova factory's host ``predict``** (``supernova_model`` /
    ``register_supernova``), once per model -- the entry point the CPU samplers call. Nothing else
    calls it: ``predict_jax``, which every GPU sampler runs, is traced; and the CPU redback adapter
    (``register_redback("basic_magnetar_powered")``, ``"slsn"``, ...), which has the same limit
    because it IS redback's grid, does not import this module. To check a GPU or a redback fit,
    call it on the fitted parameters: ``warn_if_spin_down_unresolved(dense_grid(t_src_days),
    p0, bp, mass_ns, theta_pb)``, with the source-frame days ``(t - t_exp) / (1 + z)``.
    """
    dense_times = np.asarray(dense_times, dtype=float)
    t_first = dense_times[0] if dense_times[0] > 0 else dense_times[1]
    tp = (1.3e5 * bp ** -2 * p0 ** 2 * (mass_ns / 1.4) ** 1.5
          * max(abs(np.sin(theta_pb)), 1e-8) ** -2)             # basic_magnetar's, CHANGE 3
    x = (2.0 if convention == "1.15" else 1.0) * t_first * DAY_TO_S / tp
    frac = float(x / (1.0 + x))
    if frac > limit:
        warnings.warn(
            f"magnetar spin-down unresolved: {100 * frac:.1f}% of E_rot is released before the "
            f"dense grid's first resolved node ({t_first * DAY_TO_S:.3g} s; p0={p0:g}, bp={bp:g}, "
            f"mass_ns={mass_ns:g}, theta_pb={theta_pb:g}). redback's grid mishandles that energy "
            f"the same way -- lost on 1.20's geometric grid (too faint), smeared over the first "
            f"cell on the linear one (too bright) -- so this light curve does not conserve E_rot.",
            RuntimeWarning, stacklevel=2)
    return frac


def _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej):
    """``ip.Diffusion`` on this module's grid. One line, but it is called nine times.

    The x64 check lives here as well as in :func:`build_sn_grid` so that calling a bolometric
    function directly with a hand-built grid still fails loudly (CHANGE 8). It is a Python
    branch on a config flag, so it resolves at TRACE time and costs nothing at run time.
    """
    _require_x64("whisper_cbpf.models.jax.supernova")
    # the observed-epoch path marks its geometric engine grid; key presence is static under jit
    return diffusion(grid['uniq_times'], grid['gather_idx'], grid['tb'],
                     grid['dense_times'], dense_lbols, kappa, kappa_gamma, mej, vej,
                     geometric='dense_is_geometric' in grid)


# --- 1. ARNETT -----------------------------------------------------------------------------
#
# PHYSICS. 56Ni decays to 56Co (8.8 d) which decays to 56Fe (111.3 d), each with a fixed
# specific luminosity. That heating is convolved with the Arnett diffusion kernel, which
# delays and broadens it by the photon escape time through the expanding ejecta. The observed
# peak is set by the competition between the decay timescale and the diffusion timescale.
#
# ORIGINAL (supernova_models.py:357 and :375):
#     def _nickelcobalt_engine(time, f_nickel, mej, **kwargs):
#         ni56_lum = 6.45e43; co56_lum = 1.45e43
#         ni56_life = 8.8; co56_life = 111.3
#         nickel_mass = f_nickel * mej
#         lbol = nickel_mass * (ni56_lum*np.exp(-time/ni56_life)
#                               + co56_lum*np.exp(-time/co56_life))
#         return lbol
#
#     def arnett_bolometric(time, f_nickel, mej, **kwargs):      # redback 1.20, :946
#         _interaction_process = kwargs.get("interaction_process", ip.Diffusion)
#         lbol = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
#         if _interaction_process is not None:
#             dense_resolution = kwargs.get("dense_resolution", 1000)
#             dense_times = np.geomspace(1e-5, time[-1]+100, dense_resolution)
#             dense_lbols = _nickelcobalt_engine(time=dense_times, f_nickel=f_nickel, mej=mej)
#             interaction_class = _interaction_process(time=time, dense_times=dense_times,
#                                                      luminosity=dense_lbols, mej=mej, **kwargs)
#             lbol = interaction_class.new_luminosity
#         return lbol
#
# (redback <= 1.15 built ``np.linspace(0, time[-1]+100, dense_resolution)`` here; see
# REDBACK_GRID_PRESETS.)
#
# CHANGES: CHANGE 1 (the grid) only. Nothing else moves -- two exponentials of a NEGATIVE
# argument have nothing to overflow and nothing to cancel, and they are finite at t = 0, where
# the linear grid of redback <= 1.15 starts. `if _interaction_process is not None` is Python
# CONFIGURATION branching resolved at trace time, so the no-interaction variant is simply
# `nickelcobalt_engine` called directly.

def nickelcobalt_engine(time_days, f_nickel, mej):
    """Ni56 -> Co56 -> Fe56 radioactive heating, erg/s. ``time`` days, ``mej`` Msun."""
    ni56_lum, co56_lum = 6.45e43, 1.45e43
    ni56_life, co56_life = 8.8, 111.3
    nickel_mass = f_nickel * mej
    return nickel_mass * (ni56_lum * jnp.exp(-time_days / ni56_life)
                          + co56_lum * jnp.exp(-time_days / co56_life))


@jax.jit
def arnett_bolometric(grid, f_nickel, mej, kappa, kappa_gamma, vej):
    """Arnett bolometric luminosity, erg/s. ``vej`` km/s, ``kappa``/``kappa_gamma`` cm^2/g."""
    dense_lbols = nickelcobalt_engine(grid['dense_times'], f_nickel, mej)
    return _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej)


#: ``type_1a`` and ``type_1c`` call ``arnett_bolometric`` with no modification at all
#: (supernova_models.py:1504 / :1583, both branches). Their entire distinction from ``arnett``
#: is the SED -- see :func:`sn_ab_magnitude`. The aliases below are exact, not approximate.
type_1a_bolometric = arnett_bolometric
type_1c_bolometric = arnett_bolometric


# --- 2. SHOCK COOLING + ARNETT     (Piro+2021) ---------------------------------------------
#
# PHYSICS. A low-mass extended envelope around the progenitor is shock-heated at explosion and
# radiates as it cools, producing an early blue excess on top of the nickel-powered main peak.
# Before the envelope diffusion time td the luminosity follows a shallow power law; after td
# it cuts off exponentially as the envelope goes optically thin. The two components simply add
# -- they are independent energy sources, and the shock term does NOT pass through diffusion.
#
# ORIGINAL (shock_powered_models.py:613):
#     nn = kwargs.get('nn', 10); delta = kwargs.get('delta', 1.1)
#     kk_pow = (nn - 3)*(3 - delta)/(4*np.pi*(nn - delta))
#     kappa = 0.2
#     mass = mass * cc.solar_mass
#     vt = (((nn-5)*(5-delta)/((nn-3)*(3-delta))) * (2*energy/mass))**0.5
#     td = ((3*kappa*kk_pow*mass)/((nn-1)*vt*cc.speed_of_light))**0.5
#     prefactor = np.pi*(nn-1)/(3*(nn-5))*cc.speed_of_light*radius*vt**2/kappa
#     lbol_pre_td  = prefactor*np.power(td/time, 4/(nn-2))
#     lbol_post_td = prefactor*np.exp(-0.5*(time*time/td/td - 1))
#     lbol = np.zeros(len(time))
#     lbol[time < td]  = lbol_pre_td[time < td]
#     lbol[time >= td] = lbol_post_td[time >= td]
#     tph = np.sqrt(3*kappa*kk_pow*mass/(2*(nn-1)*vt*vt))
#     r_photosphere_pre_td  = np.power(tph/time, 2/(nn-1))*vt*time
#     r_photosphere_post_td = (np.power((delta-1)/(nn-1)*((time/td)**2 - 1) + 1,
#                                       -1/(delta+1))*vt*time)
#     r_photosphere = r_photosphere_pre_td + r_photosphere_post_td
#     sigmaT4 = lbol/(4*np.pi*r_photosphere**2)
#     temperature = np.power(sigmaT4/cc.sigma_sb, 0.25)
#
# CHANGES:
#  1. CHANGE 2 (masked assignment -> where). The condition is unchanged, so the same branch is
#     taken at each point. Both branches must be finite because both are evaluated: lbol_pre_td
#     is a power of td/time (fine for time > 0) and lbol_post_td is exp of a NEGATIVE argument
#     for time > td (it underflows to 0, never overflows). Only `time` is floored, against the
#     exact-zero epoch (CHANGE 3).
#  2. r_photosphere ADDS the pre- and post-td expressions rather than selecting between them.
#     That is redback's own smoothing and it is kept verbatim. Its post-td base can go negative
#     for (nn, delta) outside the shipped prior -- giving a fractional power of a negative
#     number, i.e. NaN -- so the BASE is floored (CHANGE 3). Inside the prior box
#     (nn 8-12, delta 1-1.5) the base is bounded below by 1 - (delta-1)/(nn-1) >= 0.93 and the
#     floor is unreachable.
#  3. nn AND delta ARE TRACED, NOT STATIC. ``shock_cooling_and_arnett.prior`` gives them
#     ``Uniform(8, 12)`` and ``Uniform(1, 1.5)`` -- redback FITS them. Nothing here needs them
#     at trace time (no shape depends on either), so making them static would have silently
#     removed two of the model's eleven free parameters.

def shock_cooling(time_s, mass, radius, energy, nn=10.0, delta=1.1):
    """Piro+2021 shock cooling. ``time_s`` SECONDS, ``mass`` Msun, ``radius``/``energy`` cgs.

    Returns ``(L_bol [erg/s], R_photosphere [cm], T [K], td [s])``.
    """
    _require_x64("shock_cooling")          # prefactor ~ c R vt^2 / kappa reaches 1e46 (CHANGE 8)
    kk_pow = (nn - 3) * (3 - delta) / (4 * jnp.pi * (nn - delta))
    kappa = 0.2
    mass_cgs = mass * SOLAR_MASS

    vt = (((nn - 5) * (5 - delta) / ((nn - 3) * (3 - delta)))
          * (2 * energy / mass_cgs)) ** 0.5
    td = ((3 * kappa * kk_pow * mass_cgs)
          / ((nn - 1) * vt * SPEED_OF_LIGHT)) ** 0.5

    t_safe = jnp.maximum(time_s, _FLOOR)
    prefactor = (jnp.pi * (nn - 1) / (3 * (nn - 5))
                 * SPEED_OF_LIGHT * radius * vt ** 2 / kappa)
    lbol_pre = prefactor * (td / t_safe) ** (4 / (nn - 2))
    lbol_post = prefactor * jnp.exp(-0.5 * (t_safe * t_safe / td / td - 1))
    lbol = jnp.where(time_s < td, lbol_pre, lbol_post)                      # CHANGE 2

    tph = jnp.sqrt(3 * kappa * kk_pow * mass_cgs / (2 * (nn - 1) * vt * vt))
    r_pre = (tph / t_safe) ** (2 / (nn - 1)) * vt * t_safe
    base = (delta - 1) / (nn - 1) * ((t_safe / td) ** 2 - 1) + 1
    r_post = jnp.maximum(base, _FLOOR) ** (-1 / (delta + 1)) * vt * t_safe   # CHANGE 3
    r_photosphere = r_pre + r_post

    r_safe = jnp.maximum(r_photosphere, _FLOOR)
    temperature = (lbol / (4 * jnp.pi * r_safe ** 2) / SIGMA_SB) ** 0.25
    return lbol, r_photosphere, temperature, td


# ORIGINAL (supernova_models.py:456):
#     lbol_1 = shock_cooling_bolometric(time=time*day_to_s, log10_mass=..., log10_radius=...,
#                                       log10_energy=..., **kwargs)
#     lbol_2 = arnett_bolometric(time=time, f_nickel=..., mej=..., vej=..., ...)
#     return lbol_1 + lbol_2
#
# CHANGE: none beyond the two above. Note that redback passes ``time*day_to_s`` to the shock
# term (SECONDS) and ``time`` to the Arnett term (DAYS) -- an easy unit trap, preserved exactly.
# Note also that the shock term is evaluated only at the OBSERVATION times: it never enters the
# dense grid, because it is not diffused.

@jax.jit
def shock_cooling_and_arnett_bolometric(grid, log10_mass, log10_radius, log10_energy,
                                        f_nickel, mej, vej, kappa, kappa_gamma,
                                        nn=10.0, delta=1.1):
    """Piro+2021 shock cooling plus Arnett. Bolometric luminosity, erg/s."""
    lbol_1, _, _, _ = shock_cooling(grid['time'] * DAY_TO_S,
                                    10.0 ** log10_mass, 10.0 ** log10_radius,
                                    10.0 ** log10_energy, nn, delta)
    lbol_2 = arnett_bolometric(grid, f_nickel, mej, kappa, kappa_gamma, vej)
    return lbol_1 + lbol_2


# --- 3. MAGNETAR-POWERED     and     4. SLSN -----------------------------------------------
#
# PHYSICS. A rapidly spinning, highly magnetised neutron star injects its rotational energy
# E_rot into the ejecta by dipole spin-down on a timescale t_p. The luminosity is flat for
# t << t_p and falls as t^-2 afterwards.
#
# ORIGINAL (magnetar_models.py:172), and note that this is the ONE function in this family
# whose body differs between the two installed redbacks -- see CHANGE 5:
#     erot = 2.6e52 * (mass_ns/1.4)**(3./2.) * p0**(-2)
#     tp   = 1.3e5 * bp**(-2) * p0**2 * (mass_ns/1.4)**(3./2.) * (np.sin(theta_pb))**(-2)
#     luminosity =      erot/tp/(1. +      time/tp)**2        # 1.12.0
#     luminosity = 2. * erot/tp/(1. + 2. * time/tp)**2        # 1.15.1, and MOSFiT
#
# CHANGE 3: sin(theta_pb)**-2 diverges at theta_pb = 0 and pi, where redback returns inf. The
# sine is floored so the gradient stays finite at the prior edge -- redback's own
# ``Uniform(0, 3.14/2)`` INCLUDES theta_pb = 0, so this edge is drawn from, not hypothetical.
# At theta_pb = 0 the model says the field is aligned with the spin axis and there is no dipole
# radiation at all: t_p -> inf, L -> 0. The floor makes that limit reachable instead of NaN.

def basic_magnetar(time_s, p0, bp, mass_ns, theta_pb, convention="1.15"):
    """Dipole spin-down luminosity, erg/s. ``time_s`` SECONDS, ``p0`` ms, ``bp`` 1e14 G.

    ``convention`` selects which redback (CHANGE 5): ``"1.15"`` (default, and MOSFiT's) or
    ``"1.12"``. It is a Python string and must be static under jit.
    """
    if convention not in ("1.12", "1.15"):
        raise ValueError(f"convention must be '1.12' or '1.15', got {convention!r}")
    erot = 2.6e52 * (mass_ns / 1.4) ** 1.5 * p0 ** -2
    sin_safe = jnp.maximum(jnp.abs(jnp.sin(theta_pb)), 1e-8)                # CHANGE 3
    tp = 1.3e5 * bp ** -2 * p0 ** 2 * (mass_ns / 1.4) ** 1.5 * sin_safe ** -2
    if convention == "1.12":
        return erot / tp / (1. + time_s / tp) ** 2
    return 2. * erot / tp / (1. + 2. * time_s / tp) ** 2


# ORIGINAL (supernova_models.py:1444, redback 1.20):
#     lbol = basic_magnetar(time=time*day_to_s, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb=theta_pb)
#     dense_times = np.geomspace(1e-5, time[-1]+100, dense_resolution)   # <= 1.15: linspace(0, ...)
#     dense_lbols = basic_magnetar(time=dense_times*day_to_s, ...)
#     interaction_class = _interaction_process(time=time, dense_times=dense_times,
#                                              luminosity=dense_lbols, **kwargs)
#     lbol = interaction_class.new_luminosity
#
# t_p spans decades inside the prior, and when it is shorter than the grid's first node neither
# grid conserves E_rot: :func:`warn_if_spin_down_unresolved` says so. The linear grid was worse
# by far -- the port delivered up to 2.3e5 E_rot and was up to 21 mag too bright against 1.20.
#
# ON slsn_bolometric. In redback it reads, in full (supernova_models.py:821):
#     def slsn_bolometric(time, p0, bp, mass_ns, theta_pb, **kwargs):
#         return basic_magnetar_powered_bolometric(time=time, p0=p0, bp=bp,
#                                    mass_ns=mass_ns, theta_pb=theta_pb, **kwargs)
# It is the SAME function. Its docstring claims "a constraint on rotational_energy/
# kinetic_energy and nebula phase", and that is not a lie -- but the constraint lives in
# ``slsn.prior`` as two bilby ``Constraint`` entries, not in the model. The only difference
# in code is downstream: ``slsn`` defaults to ``sed.CutoffBlackbody`` with
# ``cutoff_wavelength = 3000`` A while ``basic_magnetar_powered`` defaults to ``sed.Blackbody``.
# So SLSN = magnetar-powered bolometric + a UV-suppressed SED, which matters for the observed
# colours and not at all for L_bol. There is nothing separate to port at the bolometric level;
# the SED difference IS ported (:func:`cutoff_shape`), and the constraints are surfaced by
# :func:`redback_prior`.

@partial(jax.jit, static_argnames=("magnetar_convention",))
def basic_magnetar_powered_bolometric(grid, p0, bp, mass_ns, theta_pb,
                                      kappa, kappa_gamma, mej, vej,
                                      magnetar_convention="1.15"):
    """Magnetar-powered bolometric luminosity, erg/s. ALSO ``slsn_bolometric``."""
    dense_lbols = basic_magnetar(grid['dense_times'] * DAY_TO_S,
                                 p0, bp, mass_ns, theta_pb, magnetar_convention)
    return _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej)


slsn_bolometric = basic_magnetar_powered_bolometric   # identical in redback; see above


# ORIGINAL (supernova_models.py:947, the flux_density branch):
#     lbol_mag    = basic_magnetar(time=time*day_to_s, p0=p0, bp=bp, mass_ns=mass_ns,
#                                  theta_pb=theta_pb)
#     lbol_arnett = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
#     ...
#     dense_lbols  = _nickelcobalt_engine(time=dense_times, f_nickel=f_nickel, mej=mej)
#     dense_lbols += basic_magnetar(time=dense_times*day_to_s, ...)
#     lbol = interaction_class(time=time, dense_times=dense_times,
#                              luminosity=dense_lbols, mej=mej, **kwargs).new_luminosity
#
# CHANGE 7: the magnitude branch of redback's ``magnetar_nickel`` never reaches that block --
# it feeds the raw summed engine to the photosphere with no diffusion. This module always
# diffuses. Otherwise unchanged: the two engines are summed on the dense grid BEFORE diffusion,
# which is what redback's flux_density branch does and is physically right -- both sources
# deposit heat in the same ejecta, so one diffusion kernel applies to their sum.

@partial(jax.jit, static_argnames=("magnetar_convention",))
def magnetar_nickel_bolometric(grid, f_nickel, mej, p0, bp, mass_ns, theta_pb,
                               kappa, kappa_gamma, vej, magnetar_convention="1.15"):
    """Magnetar spin-down plus nickel decay, summed then diffused. erg/s."""
    dense_lbols = (basic_magnetar(grid['dense_times'] * DAY_TO_S,
                                  p0, bp, mass_ns, theta_pb, magnetar_convention)
                   + nickelcobalt_engine(grid['dense_times'], f_nickel, mej))
    return _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej)


# --- 5. CSM SHOCK BREAKOUT + ARNETT     (Margalit 2022) ------------------------------------
#
# PHYSICS. The ejecta slam into a dense CSM shell with a hard outer boundary. The shock
# converts kinetic energy to radiation, which then diffuses out of the shocked shell. term1
# and term2 track the swept volume, term3 and term4 the adiabatic and diffusive losses.
#
# ORIGINAL (shock_powered_models.py:471):
#     v0 = v_min*1e5; e0 = 0.5*csm_mass*v0**2; velocity = v0/beta
#     shell_radius *= 1e14; shell_width = shell_width_ratio*shell_radius
#     tdyn = shell_radius/velocity; tshell = shell_width/velocity
#     time = time*cc.day_to_s
#     tda = (3*kappa*csm_mass/(4*np.pi*cc.speed_of_light*velocity))**0.5
#     term1 = ((tdyn + tshell + time)**3 - (tdyn + beta*time)**3)**(2/3)
#     term2 = ((tdyn + tshell)**3 - tdyn**3)**(1/3)
#     term3 = (1 + (1-beta)*time/tshell)**(-3*(tdyn/tda)**2
#              * ((1 - beta - beta*tshell/tdyn)**2)/(1-beta)**3)
#     term4 = np.exp(-time*((1-beta**3)*time + (2 - 4*beta*(beta+1))*tshell
#              + 6*(1-beta**2)*tdyn)/(2*(1-beta)**2*tda**2))
#     lbol = e0*term1/(tda**2*(tshell + (1-beta)*time)**2)*term2*term3*term4
#     volume = 4./3.*np.pi*velocity**3*((tdyn+tshell+time)**3 - (tdyn+beta*time)**3)
#     radius = velocity*(tdyn + tshell + time)
#     rphotosphere = radius - 2*volume/(3*kappa*csm_mass)
#     teff = (lbol/(4*np.pi*rphotosphere**2*cc.sigma_sb))**0.25
#
# CHANGES:
#  1. CHANGE 4 -- none to this function: it is exactly what redback evaluates, at whatever
#     times it is given. What redback does with it is in ``csm_shock_and_arnett_bolometric``.
#  2. term3 is a power whose EXPONENT depends on the parameters and can be large and negative;
#     the base is > 1 for beta < 1, so the result underflows to 0 rather than overflowing.
#     Safe as written; the base is nonetheless floored against beta -> 1, where
#     (1-beta)*time/tshell degenerates (CHANGE 3).
#  3. term4's exponent is negative for beta < 1, so exp never overflows. Left exactly as
#     redback wrote it.
#  4. rphotosphere can go NEGATIVE (the photosphere recedes inside the shell) whenever the
#     shell is optically thick enough that 2V/(3 kappa M) exceeds the outer radius. redback
#     allows this and squares it for teff, so its temperature is still defined; that is
#     preserved, and only the SQUARE is floored so a sign change cannot produce inf. Whether
#     a negative photosphere radius is meaningful is a question for Margalit 2022, not
#     something a port may quietly fix. Note that ``csm_shock_and_arnett`` does not use this
#     radius at all: it puts a single ``TemperatureFloor`` photosphere on the SUMMED
#     luminosity, so only ``lbol`` leaves this function in that model.

@jax.jit
def csm_shock_breakout(time_days, csm_mass, v_min, beta, kappa,
                       shell_radius, shell_width_ratio):
    """Margalit 2022 CSM shell breakout.

    ``csm_mass`` Msun, ``v_min`` km/s, ``shell_radius`` in 1e14 cm, ``kappa`` cm^2/g.
    Returns ``(L_bol [erg/s], R_photosphere [cm], T [K])``.
    """
    _require_x64("csm_shock_breakout")     # e0 = 0.5 M v0^2 reaches 1e51 (CHANGE 8)
    csm_mass_cgs = csm_mass * SOLAR_MASS
    v0 = v_min * 1e5
    e0 = 0.5 * csm_mass_cgs * v0 ** 2
    velocity = v0 / beta
    r_shell = shell_radius * 1e14
    shell_width = shell_width_ratio * r_shell
    tdyn = r_shell / velocity
    tshell = shell_width / velocity
    t = time_days * DAY_TO_S

    tda = (3 * kappa * csm_mass_cgs / (4 * jnp.pi * SPEED_OF_LIGHT * velocity)) ** 0.5

    term1 = ((tdyn + tshell + t) ** 3 - (tdyn + beta * t) ** 3) ** (2 / 3)
    term2 = ((tdyn + tshell) ** 3 - tdyn ** 3) ** (1 / 3)
    base3 = jnp.maximum(1 + (1 - beta) * t / tshell, _FLOOR)                # CHANGE 3
    expo3 = (-3 * (tdyn / tda) ** 2
             * ((1 - beta - beta * tshell / tdyn) ** 2) / (1 - beta) ** 3)
    term3 = base3 ** expo3
    term4 = jnp.exp(-t * ((1 - beta ** 3) * t
                          + (2 - 4 * beta * (beta + 1)) * tshell
                          + 6 * (1 - beta ** 2) * tdyn)
                    / (2 * (1 - beta) ** 2 * tda ** 2))

    lbol = (e0 * term1 / (tda ** 2 * (tshell + (1 - beta) * t) ** 2)
            * term2 * term3 * term4)

    volume = (4. / 3. * jnp.pi * velocity ** 3
              * ((tdyn + tshell + t) ** 3 - (tdyn + beta * t) ** 3))
    radius = velocity * (tdyn + tshell + t)
    rphot = radius - 2 * volume / (3 * kappa * csm_mass_cgs)
    r2 = jnp.maximum(rphot ** 2, _FLOOR)
    teff = (jnp.maximum(lbol, 0.0) / (4 * jnp.pi * r2 * SIGMA_SB)) ** 0.25
    return lbol, rphot, teff


# ORIGINAL (supernova_models.py:2646, redback 1.20):
#     nickel_lbol = arnett_bolometric(time=time, f_nickel=f_nickel, mej=mej,
#                       interaction_process=ip.Diffusion, kappa=kappa, vej=v_min, **kwargs)
#     sbo_output  = csm_shock_breakout_bolometric(time=time, v_min=v_min, beta=beta,
#                       kappa=kappa, csm_mass=csm_mass, shell_radius=shell_radius,
#                       shell_width_ratio=shell_width_ratio, **kwargs)
#     lbol = nickel_lbol + sbo_output
#
# and csm_shock_breakout_bolometric (shock_powered_models.py:529) evaluates the closed form on
# fixed nodes and interpolates:
#     time_temp = get_optimal_time_array(1e-2, 200, 300)   # == np.geomspace(1e-2, 200, 300)
#     func = interp1d(time_temp, outputs.lbol, fill_value='extrapolate')  # <= 1.15: linspace
#
# CHANGE 4: the interpolation is redback's and is reproduced (``grid['csm_times']``), because it
# is not small: against the closed form it moves the curve by up to 0.12 mag at t >= 1 d and
# 0.42 mag at t < 1 d. ``build_sn_grid(..., csm_interp=False)`` evaluates the closed form at the
# epochs instead, which is exact. OUTSIDE the nodes, [0.01 d, 200 d], the closed form is used
# either way: redback extrapolates linearly there, and past 200 d that goes NEGATIVE (from
# 201.09 d; reported upstream) where the light curve decays
# exponentially.
#
# Note also that redback passes ``vej=v_min`` to the Arnett term, so the ejecta velocity used
# for DIFFUSION is the CSM minimum velocity and there is no separate ``vej`` parameter in this
# model. Preserved -- and it is why ``csm_shock_and_arnett.prior`` names its velocity ``v_min``
# while labelling it $v_{ej}$.

@jax.jit
def csm_shock_and_arnett_bolometric(grid, mej, f_nickel, csm_mass, v_min, beta,
                                    shell_radius, shell_width_ratio, kappa, kappa_gamma):
    """CSM shell breakout plus Arnett. Bolometric luminosity, erg/s."""
    nickel_lbol = arnett_bolometric(grid, f_nickel, mej, kappa, kappa_gamma, v_min)
    sbo, _, _ = csm_shock_breakout(grid['time'], csm_mass, v_min, beta, kappa,
                                   shell_radius, shell_width_ratio)
    if "csm_times" in grid:                   # static: the grid's structure, not its values
        nodes = grid["csm_times"]
        on_nodes, _, _ = csm_shock_breakout(nodes, csm_mass, v_min, beta, kappa,
                                            shell_radius, shell_width_ratio)
        inside = (grid["time"] >= nodes[0]) & (grid["time"] <= nodes[-1])
        sbo = jnp.where(inside, jnp.interp(grid["time"], nodes, on_nodes), sbo)
    return nickel_lbol + sbo


# --- 6. EXPONENTIAL POWERLAW ---------------------------------------------------------------
#
# PHYSICS. A purely phenomenological light curve: a rising (1 - e^{-t/tpeak}) term times a
# falling power law. No physical engine -- it is a flexible shape for when you want a fit
# without committing to a power source. It is still passed through the Arnett diffusion
# kernel, which is arguably odd for a phenomenological engine but is what redback does, and
# it is what makes mej, vej, kappa and kappa_gamma parameters of the model.
#
# ORIGINAL (phenomenological_models.py:368, and supernova_models.py:149 for the wrapper):
#     def exponential_powerlaw(time, a_1, alpha_1, alpha_2, tpeak, **kwargs):
#         total = a_1*(1 - np.exp(-time/tpeak))**alpha_1 * (time/tpeak)**(-alpha_2)
#         return total
#
# CHANGE 3, and on the LINEAR grid of redback <= 1.15 it has a MEASURED consequence. (On redback
# 1.20's geometric grid, the default, there is none: its first node is 1e-5 d, where the formula
# is finite, both codes evaluate it there, and they agree exactly -- and a divergent integral is
# then cut off at 1e-5 d by both.) The linear dense grid starts at exactly t = 0 (CHANGE 1),
# where this engine is ``0**alpha_1 * inf**alpha_2`` = NaN -- verified: 1 non-finite entry at
# index 0 of ``linspace(0, 300, 1000)``. That NaN then propagates through ``interp1d`` to every
# quadrature node in the FIRST dense interval, and ``ip.Diffusion``'s
# ``int_args[np.isnan(int_args)] = 0.`` zeroes the integrand there. So redback's effective engine is "0 on [0, dt), the formula
# after", where dt = (t_last + 100)/999 ~ 0.3 d.
#
# This module cannot use a NaN for that: a NaN in the forward pass poisons the gradient through
# the ``where`` that removes it. The engine is instead defined to be exactly 0 at t <= 0 --
# which is the correct limit for alpha_1 > alpha_2 and a floored one otherwise -- so the
# integrand ramps linearly from 0 across [0, dt) where redback holds it at 0. The difference is
# confined to one dense interval out of 1000, weighted by t' <= dt inside the integrand.
#
# WHERE THE ENGINE IS INTEGRABLE, this is a grid statement and it converges away at second order.
# MEASURED at alpha_1 = 2, alpha_2 = 1, tpeak = 10 d, 60 epochs from 0.5 to 200 d
# (``_scratch/sn_03_exppl_firstinterval.py``):
#
#     n_dense   dt [d]   peak-normalised   max |dmag|   max |dmag| at t > 2 d
#       1000    0.291       2.26e-05         0.253            0.0023
#       2000    0.145       3.11e-06         0.035            0.00037
#       5000    0.058       1.59e-07         0.0020           0.000019
#      20000    0.015       3.14e-09         0.0000           0.0000
#
# Read the two magnitude columns together: peak-normalised the difference is negligible, but
# LOCALLY it reaches 0.25 mag, confined to the 11 epochs below 1.3 d where the luminosity is
# 0.16% of peak -- exactly where redback's LINEAR dense grid puts fewer than two nodes inside the
# whole integration range. Raise ``build_sn_grid(..., dense_resolution=)`` (redback has the same
# kwarg) rather than choosing between two under-resolved answers.
#
# WHERE IT IS NOT INTEGRABLE, NEITHER ANSWER MEANS ANYTHING, and that is most of redback's own
# prior. As t -> 0 the engine goes as (t/tpeak)^(alpha_1 - alpha_2), so the diffusion integrand
# L(t') t' goes as t'^(alpha_1 - alpha_2 + 1) and
#
#     int_0 L(t') t' dt'   converges  <=>  alpha_2 - alpha_1 < 2.
#
# ``sn_exponential_powerlaw.prior`` draws BOTH exponents from ``Uniform(0, 10)`` independently,
# so that condition fails on ~32% of draws, and on those draws the integral simply does not
# exist. MEASURED on the linear grid over 400 prior draws (``_scratch/sn_07_exppl_prior.py``),
# peak-normalised difference from redback:
#
#     n_dense    median      p90       p99       max     > 1% of peak
#       1000    3.5e-01   5.7e-01   6.2e-01   6.5e-01       74.5%
#       5000    1.7e-01   5.6e-01   6.8e-01   7.4e-01       63.0%
#      20000    5.7e-02   5.5e-01   6.6e-01   8.3e-01       56.2%
#     100000    1.2e-02   5.3e-01   6.6e-01   8.1e-01       50.5%
#
# THE MEDIAN CONVERGES AND THE TAIL DOES NOT. A hundredfold refinement moves the p90 by 4%.
# Those draws are the divergent ones: median alpha_2 - alpha_1 = +1.67 and median tpeak = 0.12 d
# among the disagreeing draws, against -3.06 and 6.25 d among the agreeing ones; 68% of the
# disagreeing draws have alpha_2 > alpha_1, against 9% of the agreeing ones. redback truncates
# the divergence at its first grid node, this module ramps to it, and both answers are functions
# of the grid rather than of the parameters. That is not a porting difference to be tightened --
# it is the model being ill-posed on a third of its own prior, and the place to fix it is the
# prior. :func:`exponential_powerlaw_integrable` is the condition, closed-form in the two
# exponents.

def exponential_powerlaw_engine(time_days, lbol_0, alpha_1, alpha_2, tpeak_d):
    """Phenomenological rise-and-decay, erg/s. ``time`` and ``tpeak`` share units (days).

    THE SUBSTITUTE IN THE DEAD BRANCH IS ``tpeak_d``, NOT A SMALL FLOOR, and that is not a
    style choice. With ``t = _FLOOR = 1e-30`` the two factors go opposite ways:
    ``(1 - exp(-t/tp))**alpha_1`` underflows to 0 while ``(t/tp)**(-alpha_2)`` overflows to
    inf, and 0 * inf = NaN in the branch the ``where`` discards -- whose VJP then poisons the
    gradient anyway. MEASURED before this fix: 30 of 200 draws from redback's own
    ``sn_exponential_powerlaw`` prior returned NaN gradients, all of them at alpha_1 and
    alpha_2 large enough for both limits to bite. Substituting ``tpeak_d`` makes the dead
    branch ``(1 - e^-1)**alpha_1``, an O(1) number for any exponent in the prior.
    """
    t_live = time_days > 0.0
    t_safe = jnp.where(t_live, time_days, tpeak_d)                          # CHANGE 3
    val = (lbol_0 * (1 - jnp.exp(-t_safe / tpeak_d)) ** alpha_1
           * (t_safe / tpeak_d) ** (-alpha_2))
    return jnp.where(t_live, val, 0.0)


def exponential_powerlaw_integrable(alpha_1, alpha_2):
    """True where the diffusion integral of this engine EXISTS. A prior condition, not a guard.

    The engine goes as ``(t/tpeak)**(alpha_1 - alpha_2)`` at the origin, so the diffusion
    integrand ``L(t') t'`` goes as ``t'**(alpha_1 - alpha_2 + 1)`` and the integral converges
    exactly when ``alpha_2 - alpha_1 < 2``. Outside that, no quadrature has a limit to converge
    to: both codes truncate the divergence at the first dense node (1e-5 d on redback 1.20's
    grid; on the linear grid of <= 1.15 this module ramps to it instead), and the answer moves
    with the grid rather than with the parameters.

    redback's ``sn_exponential_powerlaw.prior`` draws both exponents from ``Uniform(0, 10)``
    independently, so this fails on ~32% of it -- measured 74.5% of draws disagreeing with
    redback by more than 1% of peak at ``n_dense = 1000``, and still 50.5% at 100000. Use it to
    build a prior (or to reject a proposal), the same way
    :func:`whisper_cbpf.models.jax.tde.envelope_exists` is used. It changes nothing about the model.
    """
    return (alpha_2 - alpha_1) < 2.0


@jax.jit
def exponential_powerlaw_bolometric(grid, lbol_0, alpha_1, alpha_2, tpeak_d,
                                    kappa, kappa_gamma, mej, vej):
    """Exponential-powerlaw engine through Arnett diffusion. erg/s.

    NOTE the domain condition: this integral only exists for ``alpha_2 - alpha_1 < 2``. See
    :func:`exponential_powerlaw_integrable`, and CHANGE 3 for what happens outside it.
    """
    dense_lbols = exponential_powerlaw_engine(grid['dense_times'], lbol_0,
                                              alpha_1, alpha_2, tpeak_d)
    return _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej)


# --- 7. SN FALLBACK     and     8. SN NICKEL FALLBACK --------------------------------------
#
# PHYSICS. Accretion of fallback material onto the compact remnant. The mass fallback rate
# follows the canonical t^-5/3 of a disrupted envelope and the luminosity tracks it, held flat
# before a transition time tr (when accretion has not yet become fallback-limited).
# ``sn_nickel_fallback`` adds the usual nickel/cobalt decay -- two heat sources in the same
# ejecta, so they are summed BEFORE diffusion.
#
# ORIGINAL (phenomenological_models.py:315):
#     def fallback_lbol(time, logl1, tr, **kwargs):
#         l1 = 10**logl1
#         time = time * 86400
#         tr = tr * 86400
#         lbol = l1 * time**(-5./3.)
#         lbol[time < tr] = l1 * tr**(-5./3.)
#         return lbol
#
# CHANGE 2 (masked assignment -> where): same condition, same branch at each point. Both
# branches are finite for time > 0, and at time = 0 the plateau branch is the one selected
# for any tr > 0, so the floor on the decay branch is unreachable inside the prior
# (``tr`` is ``LogUniform(1e-4, 100)``, hence strictly positive). Verified: redback returns
# no non-finite value at t = 0 on its own dense grid.
#
# NOTE the docstring says "time in seconds" while the body multiplies by 86400. The input is
# DAYS. Preserved as written, because ``sn_fallback`` calls it with days.

def fallback_lbol(time_days, logl1, tr):
    """t^-5/3 fallback with a flat plateau before ``tr``. Both times in DAYS, erg/s."""
    l1 = 10.0 ** logl1
    t_s = jnp.maximum(time_days, _FLOOR) * DAY_TO_S
    tr_s = tr * DAY_TO_S
    return jnp.where(t_s < tr_s, l1 * tr_s ** (-5. / 3.), l1 * t_s ** (-5. / 3.))


@partial(jax.jit, static_argnames=("interaction",))
def sn_fallback_bolometric(grid, logl1, tr, kappa, kappa_gamma, mej, vej,
                           interaction=True):
    """Fallback accretion, through Arnett diffusion. erg/s.

    ``interaction=False`` reproduces redback, which never applies the interaction process it
    declares -- see CHANGE 7b, and note that it makes ``kappa``, ``kappa_gamma`` and ``mej``
    do nothing at all.
    """
    if not interaction:                       # static: configuration, not data
        return fallback_lbol(grid['time'], logl1, tr)
    dense_lbols = fallback_lbol(grid['dense_times'], logl1, tr)
    return _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej)


@partial(jax.jit, static_argnames=("interaction",))
def sn_nickel_fallback_bolometric(grid, mej, f_nickel, logl1, tr, kappa, kappa_gamma, vej,
                                  interaction=True):
    """Fallback accretion plus nickel decay, summed then diffused. erg/s.

    ``interaction=False`` reproduces redback -- see CHANGE 7b. ``mej`` survives there, because
    it also sets the nickel mass; ``kappa`` and ``kappa_gamma`` do not.
    """
    if not interaction:                       # static: configuration, not data
        return (fallback_lbol(grid['time'], logl1, tr)
                + nickelcobalt_engine(grid['time'], f_nickel, mej))
    dense_lbols = (fallback_lbol(grid['dense_times'], logl1, tr)
                   + nickelcobalt_engine(grid['dense_times'], f_nickel, mej))
    return _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej)


# --- 9. GENERAL MAGNETAR SLSN --------------------------------------------------------------
#
# PHYSICS. Generalised spin-down: instead of assuming a pure dipole (braking index n = 3), the
# braking index is a free parameter, so the late-time decay goes as t^{(1+n)/(1-n)} rather than
# the dipole t^-2. n = 3 recovers the dipole case. l0 and tsd are fitted directly rather than
# derived from p0 and bp, which makes this the agnostic version of models 3 and 4.
#
# ORIGINAL (magnetar_models.py:134). Identical in both installed redbacks:
#     def magnetar_only(time, l0, tau, nn, **kwargs):
#         lum = l0 * (1. + time/tau)**((1. + nn)/(1. - nn))
#         return lum
#
# CHANGE: none. The base is >= 1 for positive time and tau and the exponent is negative for
# nn > 1, so there is nothing to overflow and t = 0 is finite (L = l0). nn = 1 divides by zero
# in the exponent -- redback has the identical singularity, and ``general_magnetar_slsn.prior``
# excludes it with ``Uniform(1.1, 7)``.
#
# WORTH FLAGGING, though it changes no code here: ``general_magnetar_slsn.prior`` gives
# ``tsd = LogUniform(1e2, 1e6)`` with the latex label $\\tau_{sd}~(s)$, but the model's own
# docstring says "spin down damping timescale in source frame DAYS" and the body multiplies by
# ``day_to_s``. Taking the prior at face value therefore samples spin-down times of 274 to
# 2.7 million YEARS. The units here follow the CODE (days), which is what determines the light
# curve; if you adopt redback's prior, know that you are sampling days.

def magnetar_only(time_s, l0, tau_s, nn):
    """Generalised magnetar spin-down, erg/s. ``time_s`` and ``tau_s`` in SECONDS."""
    return l0 * (1. + time_s / tau_s) ** ((1. + nn) / (1. - nn))


@jax.jit
def general_magnetar_slsn_bolometric(grid, l0, tsd, nn, kappa, kappa_gamma, mej, vej):
    """Generalised magnetar SLSN through Arnett diffusion. ``tsd`` in DAYS. erg/s."""
    dense_lbols = magnetar_only(grid['dense_times'] * DAY_TO_S, l0, tsd * DAY_TO_S, nn)
    return _diffuse(grid, dense_lbols, kappa, kappa_gamma, mej, vej)


# --- THE SED LAYER -------------------------------------------------------------------------
#
# Nine of the eleven models use ``sed.Blackbody``, which is already ported: the photosphere is
# ``photosphere.TemperatureFloor`` (:func:`tde.temperature_floor_photosphere`) and the band
# integral is :func:`tde.ab_magnitude` / :func:`tde.flux_density_mjy`. The two that are not are
# ``slsn`` (CutoffBlackbody) and ``type_1a`` (CutoffBlackbody then Line), plus ``type_1c``,
# which ADDS a Synchrotron component to a plain blackbody. All three are below.
#
#
# CutoffBlackbody   (slsn, type_1a)
#
# ORIGINAL _set_sed (sed.py:356, identical in 1.12.0 and 1.15.1):
#     self.sed[self.mask] = self.FLUX_CONST*(self.r_photosphere[self.mask]**2 /
#              self.cutoff_wavelength / self.wavelength[self.mask]**4) \
#              / np.expm1(self.X_CONST/self.wavelength[self.mask]/self.temperature[self.mask])
#     self.sed[~self.mask] = self.FLUX_CONST*(self.r_photosphere[~self.mask]**2 /
#              self.wavelength[~self.mask]**5) \
#              / np.expm1(self.X_CONST/self.wavelength[~self.mask]/self.temperature[~self.mask])
#     self.sed *= self.norms[np.searchsorted(self.unique_times, self.time)]
#
# ORIGINAL _set_norm (sed.py:380): renormalises so the integrated SED matches L_bol, splitting
# the Planck integral at the cutoff and summing the first ten terms of the geometric expansion
# of 1/(e^u - 1) in closed form.
#
# THE ONE OBSERVATION THAT MAKES THIS CHEAP. Comparing the two branches with redback's plain
# ``Blackbody`` (sed = FLUX_CONST R^2 / lam^5 / expm1(...)) gives, EXACTLY,
#
#     sed_cutoff(lam, t) = sed_blackbody(lam, t) * norms(t) * min(lam/lam_cut, 1)
#
# -- one per-time scalar and one per-wavelength shape. Both factor out of the band integral,
# so the CutoffBlackbody needs NO new spectrum evaluation: the existing, hardened blackbody
# path is reused and the two factors are applied to its result. The shape can even be folded
# into the AB weights at setup, exactly as a fixed extinction is.
#
# CHANGES:
#  1. CHANGE 2, in the form above: the masked assignment becomes a continuous ``minimum``.
#  2. redback computes ``norms`` on ``np.unique(time)`` and re-gathers with ``searchsorted``.
#     That is a pure EFFICIENCY optimisation -- ``norms`` depends only on (L, R, T) at each
#     time -- so evaluating it per element gives identical values and removes a data-dependent
#     shape.
#  3. ``scipy``-free and ``gammainc``-free by construction. A generalisation of this SED with a
#     fitted absorption index (MOSFiT's ``cutoffblackbody``, which redback did NOT adopt)
#     requires incomplete gamma functions, and ``jax.scipy.special.gammainc`` has no gradient
#     with respect to its first argument -- so that version could not fit the index it exists
#     to introduce. redback's alpha = 1 closed form has no such problem and is differentiable
#     in every argument, temperature included.

def cutoff_norm(luminosity, temperature, r_photosphere, cutoff_wavelength_ang):
    """redback ``CutoffBlackbody._set_norm``. Dimensionless renormalisation, per epoch.

    Forces the wavelength-integrated cut-off SED to equal ``luminosity``. ``luminosity`` erg/s,
    ``temperature`` K, ``r_photosphere`` cm, ``cutoff_wavelength_ang`` Angstrom.
    """
    lam_c = cutoff_wavelength_ang * ANGSTROM_CGS
    tp = jnp.maximum(temperature, _FLOOR)[..., None]
    tp2, tp3 = tp ** 2, tp ** 3
    nxcs = _NXCS                                    # (10,)

    # exp of a NEGATIVE argument: underflows to 0, never overflows.
    c1 = jnp.exp(-nxcs / (lam_c * tp))
    term_1 = (c1 * (nxcs ** 2 + 2 * (nxcs * lam_c * tp + lam_c ** 2 * tp2))
              / (nxcs ** 3 * lam_c ** 3))
    term_2 = ((6 * tp3 - c1 * (nxcs ** 3 + 3 * nxcs ** 2 * lam_c * tp
                               + 6 * (nxcs * lam_c ** 2 * tp2 + lam_c ** 3 * tp3))
               / lam_c ** 3) / nxcs ** 4)
    f_blue_red = jnp.sum(term_1 + term_2, axis=-1)

    denom = FLUX_CONST_OVER_ANG * r_photosphere ** 2 * jnp.maximum(temperature, _FLOOR)
    return luminosity / jnp.maximum(denom, _FLOOR) / f_blue_red


def cutoff_shape(lam_obs_ang, redshift=0.0, cutoff_wavelength_ang=3000.0):
    """``min(lam_source/lam_cut, 1)`` on the OBSERVER wavelength grid. Setup-time-able.

    redback's SED is built on the k-corrected SOURCE frequency, so the cutoff is compared
    against ``lam_obs/(1+z)`` -- the same frame convention as host-galaxy extinction, and the
    opposite of Milky-Way extinction. Fold it into the AB weights once per dataset::

        weights = weights * cutoff_shape(lam, z)[None, :]
    """
    lam_src = jnp.asarray(lam_obs_ang) / (1.0 + redshift)
    return jnp.minimum(lam_src / cutoff_wavelength_ang, 1.0)


#
# Line   (type_1a)
#
# ORIGINAL _set_sed (redback 1.15.1 sed.py; see CHANGE below for 1.12.0):
#     amplitude_time = self.line_amplitude*np.exp(-0.5*((time_vals - self.line_time)
#                                                       /self.line_duration)**2)
#     flux_modified  = flux_base_value*(1 - amplitude_time)
#     amplitude_scaled = amplitude_time*lum_vals/(4*np.pi*self.luminosity_distance**2)
#     amplitude_scaled /= (self.line_width*np.sqrt(2*np.pi))
#     line_profile = np.exp(-0.5*((self.wavelength - self.line_wavelength)/self.line_width)**2)
#     wavelength_cm = self.wavelength*angstrom_cgs
#     flux_modified += amplitude_scaled*line_profile*(wavelength_cm**2)/speed_of_light
#
# The line is BOTH subtractive (the ``1 - amplitude_time`` attenuation applies at ALL
# wavelengths) and additive (the Gaussian emission profile), so it redistributes flux rather
# than simply removing it -- which is why it is called an absorption line and behaves like a
# P-Cygni-ish feature in the integrated band.
#
# CHANGE 5-style version note, and this one is not cosmetic. redback 1.12.0's ``Line._set_sed``
# operates on ``self.SED.sed`` -- the RAW SED array, in per-Angstrom-ish units -- and then
# labels the result ``erg/s/cm^2/Hz`` without ever dividing by ``4 pi d_L^2`` or applying the
# lam/nu conversion that ``_SED.flux_density`` applies. Its ``type_1a`` flux is therefore wrong
# by a factor ``4 pi d_L^2 nu/lam`` (~1e60), which is not a subtlety anyone can fit through.
# 1.15.1 rewrote it into the form above, which is dimensionally correct. THIS MODULE PORTS
# 1.15.1's. There is no ``line_convention`` switch, because 1.12.0's branch is not a
# convention: it is an unfixed unit error, and reproducing it would mean shipping it.
#
# CHANGE: none to 1.15.1's arithmetic. The band integral is regrouped -- see
# :func:`line_band_term` -- because the wavelength sum has no time dependence at all.

def line_band_term(lam_obs_ang, weights, redshift=0.0, line_wavelength=7.5e3,
                   line_width=500.0):
    """Per-band wavelength integral of the additive line profile, ``(n_band,)``.

    ``sum_lam profile(lam_src) * lam_src_A^2 * w[band, lam]``, with the per-time part factored
    out. It has NO time or luminosity dependence, so it is one small matvec per call and stays
    differentiable in ``line_wavelength`` and ``line_width`` -- which matters, because
    ``type_1a.prior`` pins them and a caller may reasonably want to free them.

    The profile is evaluated at the SOURCE wavelength, matching redback (its ``Line`` sees the
    k-corrected frequency).
    """
    lam_src = jnp.asarray(lam_obs_ang) / (1.0 + redshift)
    profile = jnp.exp(-0.5 * ((lam_src - line_wavelength) / line_width) ** 2)
    return jnp.sum(jnp.asarray(weights) * (profile * lam_src ** 2)[None, :], axis=1)


#
# Synchrotron   (type_1c)
#
# ORIGINAL (sed.py:484):
#     f_max = f0*source_radius**2*nu_max**2.5                      # for SSA
#     mask  = frequency < nu_max
#     sed[mask]  = f0*source_radius**2*(frequency[mask]/nu_max)**2.5 \
#                  * angstrom_cgs/speed_of_light*frequency[mask]**2
#     sed[~mask] = f_max*(frequency[~mask]/nu_max)**(-(pp - 1.)/2.) \
#                  * angstrom_cgs/speed_of_light*frequency[~mask]**2
#
# THE DISCONTINUITY IS REAL, AND IT IS UNREACHABLE FROM OPTICAL PHOTOMETRY. At nu = nu_max the
# two branches differ by a factor nu_max**2.5 -- 10**22.5 at the default nu_max = 1e9 -- because
# ``f_max`` carries a ``nu_max**2.5`` the self-absorbed branch has already divided out. That is
# a defect in redback, not in this port, and it is reproduced rather than guessed at. What
# decides whether it MATTERS is where the observations sit, and that is measurable. At
# dl = 1e27 cm, T = 8000 K, R = 1e15 cm, pp = 3, f0 = 1e-26, nu_max = 1e9:
#
#     nu [Hz]    synchrotron [mJy]   blackbody [mJy]
#     3.0e14        8.39e-13            2.48e-02
#     5.0e14        5.03e-13            3.04e-02
#     8.0e14        3.15e-13            1.97e-02
#     1.001e9       2.51e-07            7.74e-13      <- above the break
#     0.999e9       7.94e-30            7.71e-13      <- below it: the 10**22.5 step
#
# Every optical and near-infrared band is on the nu > nu_max branch, where the synchrotron term
# is ~11 orders of magnitude below the thermal one. So ``type_1c`` in the optical IS
# ``arnett``, to a part in 1e11, and the discontinuity lives at 1 GHz where no photometric
# band is. Fit it in the optical and ``pp`` is unconstrained by construction; fit it in the
# radio and you are fitting the step. Both statements are about redback's model, and both are
# stated here rather than resolved by a port.
#
# CHANGE 2 only (mask -> where). Both branches are positive powers of positive quantities.

def synchrotron_f_nu(nu_src_hz, pp, nu_max=1e9, source_radius=1e13, f0=1e-26,
                     dl_cm=1.0):
    """redback ``sed.Synchrotron`` as a flux density in erg/s/cm^2/Hz.

    ``nu_src_hz`` is the SOURCE-frame frequency. The ``_SED.flux_density`` conversion
    ``sed/(4 pi dl^2) * lam_A/nu`` is folded in here, so the return value is directly
    comparable with :func:`tde.flux_density_mjy` (before its ``/MJY``).
    """
    nu = jnp.asarray(nu_src_hz)
    lo = f0 * source_radius ** 2 * (nu / nu_max) ** 2.5
    f_max = f0 * source_radius ** 2 * nu_max ** 2.5
    hi = f_max * (nu / nu_max) ** (-(pp - 1.) / 2.)
    sed = jnp.where(nu < nu_max, lo, hi) * ANGSTROM_CGS / SPEED_OF_LIGHT * nu ** 2
    # _SED.flux_density: /(4 pi dl^2), then * lam[Angstrom] / nu.  lam_A = c*1e8/nu, so
    # lam_A/nu = c*1e8/nu^2 -- folded, so nu^2 above and 1/nu^2 here cancel exactly and
    # nothing of order 1e29 is ever formed.
    return sed * (SPEED_OF_LIGHT * 1e8 / nu ** 2) / (4 * jnp.pi * dl_cm ** 2)


# --- photometry ----------------------------------------------------------------------------

def photosphere(grid, lbol, vej, temperature_floor):
    """``photosphere.TemperatureFloor`` on this module's time grid. Returns ``(T [K], R [cm])``."""
    return temperature_floor_photosphere(grid['time'], lbol, vej, temperature_floor)


#: mJy per (erg/s/cm^2/Hz) for the Line term, with LINE_AB_CONST's AB_ZEROPOINT divided back
#: out. Folded as one literal so the flux and magnitude paths cannot drift apart.
_LINE_MJY_CONST = LINE_AB_CONST * AB_ZEROPOINT / MJY


def sn_flux_density(grid, lbol, vej, temperature_floor, nu_obs_hz, redshift, dl_cm, *,
                    sed_kind="blackbody", cutoff_wavelength=3000.0,
                    line_wavelength=7.5e3, line_width=500.0, line_time=50.0,
                    line_duration=25.0, line_amplitude=0.3,
                    pp=3.0, nu_max=1e9, source_radius=1e13, f0=1e-26,
                    dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                    r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """Flux density in mJy from a bolometric light curve. ``nu_obs_hz`` OBSERVER frame.

    ``grid['time']`` must already be the SOURCE-frame epochs (redback's
    ``calc_kcorrected_properties`` does ``t_src = t_obs/(1+z)``); the frequency k-correction
    ``nu_src = nu_obs (1+z)`` is applied here. ``sed_kind`` is one of ``"blackbody"``,
    ``"cutoff"``, ``"cutoff_line"``, ``"blackbody_synchrotron"`` and is static. The extinction
    arguments behave exactly as in :func:`tde.flux_density_mjy`.

    Extinction is applied ONCE, to the assembled spectrum, rather than to the thermal term
    alone: dust attenuates whatever reaches it, line and synchrotron included. redback applies
    none here at all (its extinction lives in a separate wrapper model), so this only matters
    if you ask for it.
    """
    temp, rad = photosphere(grid, lbol, vej, temperature_floor)
    out = flux_density_mjy(temp, rad, nu_obs_hz, redshift, dl_cm, dilation=dilation)
    lam_obs = 2.99792458e18 / jnp.asarray(nu_obs_hz)          # c in Angstrom/s, one literal

    if sed_kind == "blackbody_synchrotron":
        nu_src = jnp.asarray(nu_obs_hz) * (1.0 + redshift)
        syn = synchrotron_f_nu(nu_src, pp, nu_max, source_radius, f0, dl_cm) / MJY
        out = out + (syn * (1.0 + redshift) if dilation else syn)
    elif sed_kind != "blackbody":
        out = (out * cutoff_norm(lbol, temp, rad, cutoff_wavelength)
               * cutoff_shape(lam_obs, redshift, cutoff_wavelength))
        if sed_kind == "cutoff_line":
            # attenuate everywhere, then ADD the Gaussian profile -- which carries no cutoff
            # factor, because redback's Line adds it to the CutoffBlackbody's flux rather than
            # to its spectrum.
            lam_src = lam_obs / (1.0 + redshift)
            amp_t = line_amplitude * jnp.exp(
                -0.5 * ((grid['time'] - line_time) / line_duration) ** 2)
            profile = jnp.exp(-0.5 * ((lam_src - line_wavelength) / line_width) ** 2)
            add = (_LINE_MJY_CONST * amp_t * (lbol / LSCALE)
                   / (line_width * jnp.sqrt(2 * jnp.pi))
                   / (dl_cm / DLSCALE) ** 2 * profile * lam_src ** 2)
            out = out * (1.0 - amp_t) + (add * (1.0 + redshift) if dilation else add)
        elif sed_kind != "cutoff":
            raise ValueError(f"unknown sed_kind {sed_kind!r}")

    tau = _ext_transmission(lam_obs, redshift, ebv_mw, ebv_host,
                            xi_mw, xi_host, r_v_mw, r_v_host, law)
    return out if tau is None else out * tau


def sn_ab_magnitude(grid, lbol, vej, temperature_floor, band_idx, weights, norms,
                    lam_obs_ang, redshift, dl_cm, *, sed_kind="blackbody",
                    cutoff_wavelength=3000.0, line_wavelength=7.5e3, line_width=500.0,
                    line_time=50.0, line_duration=25.0, line_amplitude=0.3,
                    pp=3.0, nu_max=1e9, source_radius=1e13, f0=1e-26,
                    mag_floor=40.0, dilation=True,
                    ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                    r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """AB magnitude per observation from a bolometric light curve.

    ``weights, norms`` come from :func:`kilonova.ab_weights`; ``lam_obs_ang`` is its grid.
    ``grid['time']`` is the SOURCE-frame epoch of each observation, as in
    :func:`sn_flux_density`. See :mod:`whisper_cbpf.models.jax.tde`'s ``ab_magnitude`` for the
    ``mag_floor`` and extinction semantics, which are unchanged.

    The three non-blackbody SEDs enter as three factors on ONE band integral -- a per-epoch
    renormalisation, a per-wavelength shape, and an additive per-band term -- rather than as
    three separate spectrum evaluations. See the SED-layer block above for why that is exact.
    """
    temp, rad = photosphere(grid, lbol, vej, temperature_floor)
    ext = dict(ebv_mw=ebv_mw, ebv_host=ebv_host, xi_mw=xi_mw, xi_host=xi_host,
               r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)

    if sed_kind == "blackbody":
        return ab_magnitude(temp, rad, band_idx, weights, norms, lam_obs_ang,
                            redshift, dl_cm, mag_floor=mag_floor, dilation=dilation, **ext)

    # --- everything below rebuilds tde.ab_magnitude's band integral so the SED factors can
    # --- be applied to `num` before the log. Kept textually parallel to it on purpose.
    nu_obs = SPEED_OF_LIGHT / (jnp.asarray(lam_obs_ang) * ANGSTROM_CGS)
    nu_src = nu_obs[None, :] * (1.0 + redshift)
    f_ab = _flux_nu_tde(temp[:, None], rad[:, None], dl_cm, nu_src, redshift,
                        PLANCK_T3_AB, dilation=dilation)
    tau = _ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host,
                            xi_mw, xi_host, r_v_mw, r_v_host, law)
    w = weights if tau is None else weights * tau[None, :]

    if sed_kind == "blackbody_synchrotron":
        syn = synchrotron_f_nu(nu_src, pp, nu_max, source_radius, f0, dl_cm) * INV_AB_ZEROPOINT
        f_ab = f_ab + (syn * (1.0 + redshift) if dilation else syn)
        num = jnp.sum(f_ab * w[band_idx], axis=1)
    else:
        w_cut = w * cutoff_shape(lam_obs_ang, redshift, cutoff_wavelength)[None, :]
        num = (jnp.sum(f_ab * w_cut[band_idx], axis=1)
               * cutoff_norm(lbol, temp, rad, cutoff_wavelength))
        if sed_kind == "cutoff_line":
            amp_t = line_amplitude * jnp.exp(
                -0.5 * ((grid['time'] - line_time) / line_duration) ** 2)
            # `w`, NOT `w_cut`: redback's Line adds its profile to the CutoffBlackbody's
            # FLUX, so the additive term never sees the cutoff suppression.
            g = line_band_term(lam_obs_ang, w, redshift, line_wavelength, line_width)
            add = (LINE_AB_CONST * amp_t * (lbol / LSCALE)
                   / (line_width * jnp.sqrt(2 * jnp.pi))
                   / (dl_cm / DLSCALE) ** 2 * g[band_idx])
            num = num * (1.0 - amp_t) + (add * (1.0 + redshift) if dilation else add)
        elif sed_kind != "cutoff":
            raise ValueError(f"unknown sed_kind {sed_kind!r}")

    den = jax.lax.optimization_barrier(INV_AB_ZEROPOINT * norms[band_idx])
    ratio_floor = jnp.asarray(10.0, dtype=num.dtype) ** (-0.4 * mag_floor)
    return -2.5 * jnp.log10(jnp.maximum(num / den, ratio_floor))


# --- the family, as a table ----------------------------------------------------------------
#
# Eleven models that share nine engines, one interaction process, one photosphere and four
# SEDs. Spelling that out as a table rather than as eleven near-identical entry points is what
# keeps the whisper adapter (:func:`whisper_cbpf.models.jax.supernova_model`) to one function
# instead of eleven, and it is the only place a new model has to be registered.
#
# ``engine``     : the bolometric function, called as ``engine(grid, **{p: v for p in engine_params})``
# ``params``     : the model's FULL free-parameter list, in redback's own order. Everything not
#                  consumed by the engine is consumed by the photosphere or the SED.
# ``sed``        : which branch of :func:`sn_ab_magnitude` / :func:`sn_flux_density` applies.
# ``vej_name``   : which parameter the TemperatureFloor photosphere uses as the ejecta
#                  velocity. It is ``v_min`` for the CSM model -- see its CHANGE note.
# ``prior``      : the name of redback's prior file, which is NOT always the model name.

_ENGINE_ARGS = {
    "arnett": ("f_nickel", "mej", "kappa", "kappa_gamma", "vej"),
    "shock_cooling_and_arnett": ("log10_mass", "log10_radius", "log10_energy", "f_nickel",
                                 "mej", "vej", "kappa", "kappa_gamma", "nn", "delta"),
    "basic_magnetar_powered": ("p0", "bp", "mass_ns", "theta_pb", "kappa", "kappa_gamma",
                               "mej", "vej"),
    "slsn": ("p0", "bp", "mass_ns", "theta_pb", "kappa", "kappa_gamma", "mej", "vej"),
    "magnetar_nickel": ("f_nickel", "mej", "p0", "bp", "mass_ns", "theta_pb", "kappa",
                        "kappa_gamma", "vej"),
    "csm_shock_and_arnett": ("mej", "f_nickel", "csm_mass", "v_min", "beta", "shell_radius",
                             "shell_width_ratio", "kappa", "kappa_gamma"),
    "sn_exponential_powerlaw": ("lbol_0", "alpha_1", "alpha_2", "tpeak_d", "kappa",
                                "kappa_gamma", "mej", "vej"),
    "sn_fallback": ("logl1", "tr", "kappa", "kappa_gamma", "mej", "vej"),
    "sn_nickel_fallback": ("mej", "f_nickel", "logl1", "tr", "kappa", "kappa_gamma", "vej"),
    "general_magnetar_slsn": ("l0", "tsd", "nn", "kappa", "kappa_gamma", "mej", "vej"),
    "type_1a": ("f_nickel", "mej", "kappa", "kappa_gamma", "vej"),
    "type_1c": ("f_nickel", "mej", "kappa", "kappa_gamma", "vej"),
}

_PHOTOMETRY_ONLY = {
    "slsn": ("cutoff_wavelength",),
    "type_1a": ("line_wavelength", "line_width", "line_time", "line_duration",
                "line_amplitude", "cutoff_wavelength"),
    "type_1c": ("pp", "nu_max", "source_radius", "f0"),
}

MODELS = {
    "arnett": dict(
        engine=arnett_bolometric, sed="blackbody", vej_name="vej", prior="arnett",
        params=("f_nickel", "mej", "vej", "kappa", "kappa_gamma", "temperature_floor")),
    "shock_cooling_and_arnett": dict(
        engine=shock_cooling_and_arnett_bolometric, sed="blackbody", vej_name="vej",
        prior="shock_cooling_and_arnett",
        params=("log10_mass", "log10_radius", "log10_energy", "nn", "delta", "f_nickel",
                "mej", "vej", "kappa", "kappa_gamma", "temperature_floor")),
    "basic_magnetar_powered": dict(
        engine=basic_magnetar_powered_bolometric, sed="blackbody", vej_name="vej",
        prior="basic_magnetar_powered",
        params=("p0", "bp", "mass_ns", "theta_pb", "mej", "vej", "kappa", "kappa_gamma",
                "temperature_floor")),
    "slsn": dict(
        engine=slsn_bolometric, sed="cutoff", vej_name="vej", prior="slsn",
        params=("p0", "bp", "mass_ns", "theta_pb", "mej", "vej", "kappa", "kappa_gamma",
                "temperature_floor")),
    "magnetar_nickel": dict(
        engine=magnetar_nickel_bolometric, sed="blackbody", vej_name="vej",
        prior="magnetar_nickel",
        params=("f_nickel", "p0", "bp", "mass_ns", "theta_pb", "mej", "vej", "kappa",
                "kappa_gamma", "temperature_floor")),
    "csm_shock_and_arnett": dict(
        engine=csm_shock_and_arnett_bolometric, sed="blackbody", vej_name="v_min",
        prior="csm_shock_and_arnett",
        params=("mej", "f_nickel", "csm_mass", "v_min", "beta", "kappa", "shell_radius",
                "shell_width_ratio", "kappa_gamma", "temperature_floor")),
    "sn_exponential_powerlaw": dict(
        engine=exponential_powerlaw_bolometric, sed="blackbody", vej_name="vej",
        prior="sn_exponential_powerlaw",
        params=("lbol_0", "alpha_1", "alpha_2", "tpeak_d", "mej", "vej", "kappa",
                "kappa_gamma", "temperature_floor")),
    "sn_fallback": dict(
        engine=sn_fallback_bolometric, sed="blackbody", vej_name="vej", prior="sn_fallback",
        params=("logl1", "tr", "mej", "vej", "kappa", "kappa_gamma", "temperature_floor")),
    "sn_nickel_fallback": dict(
        engine=sn_nickel_fallback_bolometric, sed="blackbody", vej_name="vej",
        prior="sn_nickel_fallback",
        params=("logl1", "tr", "mej", "f_nickel", "vej", "kappa", "kappa_gamma",
                "temperature_floor")),
    "general_magnetar_slsn": dict(
        engine=general_magnetar_slsn_bolometric, sed="blackbody", vej_name="vej",
        prior="general_magnetar_slsn",
        params=("l0", "tsd", "nn", "mej", "vej", "kappa", "kappa_gamma", "temperature_floor")),
    "type_1a": dict(
        engine=type_1a_bolometric, sed="cutoff_line", vej_name="vej", prior="type_1a",
        params=("f_nickel", "mej", "vej", "kappa", "kappa_gamma", "temperature_floor")),
    "type_1c": dict(
        engine=type_1c_bolometric, sed="blackbody_synchrotron", vej_name="vej",
        prior="type_1c",
        params=("f_nickel", "mej", "vej", "kappa", "kappa_gamma", "temperature_floor", "pp")),
}
for _name, _spec in MODELS.items():
    _spec["engine_params"] = _ENGINE_ARGS[_name]
    _spec["sed_params"] = _PHOTOMETRY_ONLY.get(_name, ())

#: ``model -> full free-parameter tuple``, redback's own order.
PARAMETERS = {k: v["params"] for k, v in MODELS.items()}


def model_names():
    """The eleven (twelve, counting ``slsn`` and ``basic_magnetar_powered`` separately) names."""
    return sorted(MODELS)


def _spec(model):
    try:
        return MODELS[model]
    except KeyError:
        raise KeyError(f"unknown supernova model {model!r}. Known: {model_names()}") from None


_MAGNETAR_ENGINES = (basic_magnetar_powered_bolometric, magnetar_nickel_bolometric)
_FALLBACK_ENGINES = (sn_fallback_bolometric, sn_nickel_fallback_bolometric)


def _engine_kwargs(spec, params, magnetar_convention, interaction):
    kw = {}
    for name in spec["engine_params"]:
        if name in ("nn", "delta") and name not in params:
            continue                     # shock_cooling's configuration defaults
        kw[name] = params[name]
    if spec["engine"] in _MAGNETAR_ENGINES:
        kw["magnetar_convention"] = magnetar_convention      # CHANGE 5
    if spec["engine"] in _FALLBACK_ENGINES:
        kw["interaction"] = interaction                      # CHANGE 7b
    return kw


def bolometric(model, grid, params, *, magnetar_convention="1.15", interaction=True):
    """Bolometric luminosity in erg/s for any model in the family. ``model`` is static.

    ``magnetar_convention`` (CHANGE 5) and ``interaction`` (CHANGE 7b) are the two places
    where reproducing redback and believing redback part company. Both are accepted by every
    model and ignored by the ones they do not apply to, so a caller can set them once.
    """
    spec = _spec(model)
    return spec["engine"](grid, **_engine_kwargs(spec, params, magnetar_convention,
                                                 interaction))


def flux_density(model, grid, params, nu_obs_hz, redshift, dl_cm, *,
                 magnetar_convention="1.15", interaction=True, **kw):
    """Flux density in mJy for any model in the family. ``model`` is static."""
    spec = _spec(model)
    lbol = bolometric(model, grid, params, magnetar_convention=magnetar_convention,
                      interaction=interaction)
    sed_kw = {p: params[p] for p in spec["sed_params"] if p in params}
    return sn_flux_density(grid, lbol, params[spec["vej_name"]], params["temperature_floor"],
                           nu_obs_hz, redshift, dl_cm, sed_kind=spec["sed"], **sed_kw, **kw)


def ab_magnitude_of(model, grid, params, band_idx, weights, norms, lam_obs_ang,
                    redshift, dl_cm, *, magnetar_convention="1.15", interaction=True, **kw):
    """AB magnitude per observation for any model in the family. ``model`` is static."""
    spec = _spec(model)
    lbol = bolometric(model, grid, params, magnetar_convention=magnetar_convention,
                      interaction=interaction)
    sed_kw = {p: params[p] for p in spec["sed_params"] if p in params}
    return sn_ab_magnitude(grid, lbol, params[spec["vej_name"]], params["temperature_floor"],
                           band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                           sed_kind=spec["sed"], **sed_kw, **kw)


# --- the observed-epoch path: a fixed-resolution grid, the SED at the observations only -----
#
# WHY. :func:`build_sn_grid` sizes the diffusion grid from the observation epochs, so the epochs
# are a constant of the compilation: an explosion time or a redshift that moves them cannot be
# traced (the factory raised "predict_jax needs CONCRETE times"). Here the diffusion integral runs
# on a FIXED set of source-frame epochs, spaced evenly in log t (30 to a decade) from
# ``first_epoch`` to ``max_phase_days``, and ONLY the diffused bolometric luminosity is interpolated to each
# observation's source-frame epoch ``tau = (t - t_exp) / (1 + z)``, which may be traced. The
# photosphere and the SED are then evaluated at the observations and nowhere else, exactly as
# :func:`ab_magnitude_of` does on redback's grid. Two measured alternatives it
# replaces: interpolating MAGNITUDES off a grid of SEDs spent 85 % of an evaluation on SEDs at
# epochs nobody observed (S1) and was off by 0.05 mag at a shock-cooling peak (S3).
#
# REDBACK'S DISCRETISATION IS KEPT, BECAUSE IT IS NOT SMALL. redback's ``Diffusion`` spreads its
# quadrature nodes over ``[1e-5, time[-1] + 100]`` d and samples the engine on 1000 points of
# that span, so its answer at one epoch moves with the LAST epoch of the light curve. Measured
# against redback on 200 prior draws per family: spreading the nodes over the fixed
# grid's own span instead was off by up to 2.0 mag (magnetar; 5e-3 to 9e-3 mag for the others),
# and a 300-point engine grid by up to 8e-3 mag. So the engine grid is
# redback's, ``geomspace(1e-5, max(tau) + 100, 1000)``, built INSIDE the trace from the traced
# epochs -- the engines are closed-form, so this costs one engine evaluation per point, not a
# data-dependent shape -- and only the epochs the integral is evaluated at are fixed.
#
# WHAT IS INTERPOLATED, AND WHAT IS NOT. Terms redback never diffuses are evaluated in closed
# form at ``tau`` itself: the Piro shock-cooling term of ``shock_cooling_and_arnett`` (its fast
# early peak is what a grid misses, S16), the CSM breakout of ``csm_shock_and_arnett`` (on
# redback's own fixed nodes, as :func:`build_sn_grid` does), and the fallback engines with
# ``interaction=False``. Only the diffused part -- a convolution, and therefore smooth -- goes
# through the epochs, as ``ln L`` against ``ln t`` with a cubic (Catmull-Rom) through four
# neighbouring epochs.
#
# PRE-EXPLOSION. An epoch with ``tau <= 0`` returns ``mag_floor`` (zero flux). redback has no such
# branch (its ``Diffusion`` gathers the first epoch's luminosity for any earlier time), and a
# fitted explosion time needs one: an upper limit before the explosion must see no flux. Epochs
# between 0 and the first grid epoch hold that epoch's diffused luminosity (0.01 d by default,
# where a diffused light curve is ~(0.01 d / t_diff)^2, i.e. 1e-6 to 1e-4, of its peak).

#: Defaults of the fixed epochs (:func:`build_fixed_grid`), chosen against redback on
#: 200 prior draws per family (arnett, magnetar, shock cooling, CSM, type
#: Ia, type Ic; ZTF and LSST epochs): the worst case, the magnetar, is 2.6e-3 mag from redback at
#: 24 epochs per decade, 8.8e-4 at 30 and 2.8e-4 at 36 (p95 <= 7e-5 at 30).
FIXED_GRID_EPOCHS_PER_DECADE = 30
FIXED_GRID_FIRST = 1e-2
#: Epochs the grid runs past ``max_phase_days``, so the cubic's upper neighbours of the last
#: observation are real epochs, not the padded end (which cost up to 1.9e-3 mag in the gate).
FIXED_GRID_MARGIN_EPOCHS = 3
#: Source-frame days since explosion the fixed epochs cover when nothing sizes them
#: (``supernova_model(max_phase_days=)``, or ``times=`` with the priors' bounds).
DEFAULT_MAX_PHASE_DAYS = 200.0


class FixedGrid:
    """The fixed epochs of the observed-epoch path, and how to build the rest in the trace.

    Built by :func:`build_fixed_grid`; read by :func:`bolometric_at` and :func:`ab_magnitude_at`.
    ``arrays`` holds what :func:`diffusion` needs about the epochs (``time``, ``uniq_times``,
    ``gather_idx``, ``tb``, and ``csm_times``), all independent of the observations; the engine's
    dense grid is rebuilt per call from ``dense_resolution``, ``spacing`` and ``t_pad`` (see the
    block above). Plain data, so it pickles.
    """

    def __init__(self, arrays, *, max_phase_days, first_epoch, dense_resolution, spacing, t_pad):
        self.arrays = arrays
        epochs = np.asarray(arrays["time"])
        self.max_phase_days = float(max_phase_days)
        self.first_epoch = float(epochs[0])
        self.n_epochs = int(epochs.size)
        self.dense_resolution = int(dense_resolution)
        self.spacing = str(spacing)
        self.t_pad = float(t_pad)
        self.ln_t0 = float(np.log(epochs[0]))
        self.ln_step = float(np.log(epochs[-1] / epochs[0]) / (self.n_epochs - 1))

    def __repr__(self):
        return (f"FixedGrid({self.n_epochs} epochs, {self.first_epoch:g}-{self.max_phase_days:g} d "
                f"source frame, engine grid {self.dense_resolution} points, {self.spacing})")

    def dense_times(self, t_last):
        """redback's engine grid for observations ending at source-frame day ``t_last``. Traced.

        ``geomspace(1e-5, t_last + t_pad, n)`` (``linspace(0, ...)`` for ``spacing="linear"``),
        as ``ip.Diffusion`` builds it from ``time[-1]``. The end is kept at least two and a half
        epochs past ``t_last``, so the cubic's upper neighbours are always inside it.
        """
        top = jnp.maximum(t_last + self.t_pad, t_last * np.exp(2.5 * self.ln_step))
        if self.spacing == "linear":
            return jnp.linspace(0.0, 1.0, self.dense_resolution) * top
        u = jnp.linspace(0.0, 1.0, self.dense_resolution)
        dense = jnp.exp(np.log(GRID_START) + u * (jnp.log(top) - np.log(GRID_START)))
        return dense.at[0].set(GRID_START).at[-1].set(top)


def build_fixed_grid(max_phase_days=DEFAULT_MAX_PHASE_DAYS,
                     epochs_per_decade=FIXED_GRID_EPOCHS_PER_DECADE,
                     dense_resolution=DENSE_RESOLUTION, first_epoch=FIXED_GRID_FIRST,
                     t_pad=TIME_PAD, spacing="geometric", csm_interp=True):
    """The fixed epochs of the observed-epoch path. Setup-time NumPy; build once.

    Source-frame epochs from ``first_epoch`` to at least ``FIXED_GRID_MARGIN_EPOCHS`` epochs past
    ``max_phase_days`` (days since explosion), ``epochs_per_decade`` to a decade of time, at which
    the diffusion integral is evaluated. Nothing here depends on the observations, so one compiled
    model serves any explosion time, redshift or light curve whose epochs stay inside
    ``max_phase_days``.

    Parameters
    ----------
    max_phase_days : float
        The latest source-frame day since explosion a call may ask for. :func:`ab_magnitude_at`
        holds the last epoch's luminosity beyond the grid, so size it to the data: the model
        factory checks every concrete call against it.
    epochs_per_decade : float
        Resolution (default 30): 134 epochs to 200 d. The cost of an evaluation is about
        proportional to the number of epochs.
    dense_resolution : int
        Points of the engine grid built per call (default redback's 1000; see the block above
        for why fewer is not redback's answer).
    first_epoch : float
        First epoch, days (default 0.01).
    t_pad, spacing, csm_interp :
        As :func:`build_sn_grid`.

    Returns
    -------
    FixedGrid

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax import supernova as sn
    >>> g = sn.build_fixed_grid(60.0)
    >>> g.n_epochs, round(float(g.arrays["time"][-1]), 1)
    (118, 79.4)
    """
    if not (max_phase_days > first_epoch > 0):
        raise ValueError(f"need max_phase_days > first_epoch > 0, got max_phase_days="
                         f"{max_phase_days!r}, first_epoch={first_epoch!r}")
    if not epochs_per_decade > 0:
        raise ValueError(f"epochs_per_decade must be positive, got {epochs_per_decade!r}")
    step = 1.0 / float(epochs_per_decade)                           # decades per epoch
    n_in = int(np.ceil(np.log10(max_phase_days / first_epoch) / step))
    n_epochs = max(n_in + 1 + FIXED_GRID_MARGIN_EPOCHS, 4)          # a cubic needs four
    epochs = first_epoch * 10.0 ** (step * np.arange(n_epochs))
    arrays = build_sn_grid(epochs, int(dense_resolution), t_pad, spacing, csm_interp)
    del arrays["dense_times"]                  # rebuilt per call from the observations' last epoch
    return FixedGrid(arrays, max_phase_days=max_phase_days, first_epoch=first_epoch,
                     dense_resolution=dense_resolution, spacing=spacing, t_pad=t_pad)


def _loglog_cubic(ln_t, fixed, ln_l):
    """Catmull-Rom cubic of ``ln_l`` (on the log-even epochs of ``fixed``) at ``ln_t``. Traced.

    The epochs are evenly spaced in ``ln t``, so the segment is found by arithmetic rather than a
    search. The two ends are padded by linear extrapolation, so the first and last segments are
    cubics too; outside the grid the value is held at the edge (the caller keeps ``tau`` inside).
    """
    n = ln_l.shape[0]
    pad = jnp.concatenate([2.0 * ln_l[:1] - ln_l[1:2], ln_l, 2.0 * ln_l[-1:] - ln_l[-2:-1]])
    s = jnp.clip((ln_t - fixed.ln_t0) / fixed.ln_step, 0.0, n - 1.0)
    i = jnp.clip(jnp.floor(s), 0, n - 2).astype(jnp.int32)
    u = s - i
    p0, p1, p2, p3 = pad[i], pad[i + 1], pad[i + 2], pad[i + 3]
    return p1 + 0.5 * u * ((p2 - p0) + u * ((2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
                                            + u * (3.0 * (p1 - p2) + p3 - p0)))


def _split_bolometric(model, grid, tau, params, magnetar_convention, interaction):
    """``(diffused on grid['time'] or None, direct at tau)``: redback's bolometric, split.

    The diffused part is what goes through :func:`diffusion` and is interpolated; the direct part
    is the terms redback adds WITHOUT diffusing, evaluated at ``tau`` exactly. Their sum is
    :func:`bolometric`, term for term.
    """
    p = params
    if model == "shock_cooling_and_arnett":
        direct = shock_cooling(tau * DAY_TO_S, 10.0 ** p["log10_mass"], 10.0 ** p["log10_radius"],
                               10.0 ** p["log10_energy"], p.get("nn", 10.0),
                               p.get("delta", 1.1))[0]
        return arnett_bolometric(grid, p["f_nickel"], p["mej"], p["kappa"], p["kappa_gamma"],
                                 p["vej"]), direct
    if model == "csm_shock_and_arnett":
        args = (p["csm_mass"], p["v_min"], p["beta"], p["kappa"], p["shell_radius"],
                p["shell_width_ratio"])
        direct, _, _ = csm_shock_breakout(tau, *args)
        if "csm_times" in grid:                   # redback's interpolation, as the data grid does
            nodes = grid["csm_times"]
            on_nodes, _, _ = csm_shock_breakout(nodes, *args)
            inside = (tau >= nodes[0]) & (tau <= nodes[-1])
            direct = jnp.where(inside, jnp.interp(tau, nodes, on_nodes), direct)
        return arnett_bolometric(grid, p["f_nickel"], p["mej"], p["kappa"], p["kappa_gamma"],
                                 p["v_min"]), direct
    if model in ("sn_fallback", "sn_nickel_fallback") and not interaction:
        direct = fallback_lbol(tau, p["logl1"], p["tr"])
        if model == "sn_nickel_fallback":
            direct = direct + nickelcobalt_engine(tau, p["f_nickel"], p["mej"])
        return None, direct
    return bolometric(model, grid, p, magnetar_convention=magnetar_convention,
                      interaction=interaction), 0.0


def bolometric_at(model, fixed, tau_days, params, *, t_last=None, magnetar_convention="1.15",
                  interaction=True):
    """Bolometric luminosity [erg/s] at source-frame days ``tau_days``, on fixed epochs. Traced.

    ``fixed`` is :func:`build_fixed_grid`'s. ``tau_days`` may be traced (a fitted explosion time
    or redshift moves it); keep it positive -- :func:`ab_magnitude_at` masks the rest. ``t_last``
    (default ``max(tau_days)``) is the last observed epoch, from which redback's engine grid is
    built (see the block above).
    """
    tau = jnp.asarray(tau_days)
    t_last = jnp.max(tau) if t_last is None else t_last
    grid = dict(fixed.arrays, dense_times=fixed.dense_times(jnp.maximum(t_last,
                                                                         fixed.first_epoch)))
    if fixed.spacing == "geometric":
        grid["dense_is_geometric"] = 1.0       # a flag: the diffusion's lookup by arithmetic
    diffused, direct = _split_bolometric(model, grid, tau, params, magnetar_convention,
                                         interaction)
    if diffused is None:
        return direct
    ln_l = jnp.log(jnp.maximum(diffused, 1e-30))
    ln_t = jnp.log(jnp.maximum(tau, fixed.first_epoch))
    return jnp.exp(_loglog_cubic(ln_t, fixed, ln_l)) + direct


def ab_magnitude_at(model, fixed, tau_days, params, band_idx, weights, norms, lam_obs_ang,
                    redshift, dl_cm, *, magnetar_convention="1.15", interaction=True,
                    mag_floor=40.0, **kw):
    """AB magnitude per observation at source-frame days ``tau_days``, on fixed epochs. Traced.

    The observed-epoch counterpart of :func:`ab_magnitude_of`: the diffusion integral runs on the
    fixed epochs of ``fixed`` (:func:`build_fixed_grid`), the diffused luminosity is interpolated
    to ``tau_days``, and the photosphere and the SED are evaluated at ``tau_days`` only.
    ``tau_days``, ``redshift`` and ``dl_cm`` may all be traced, so an explosion time and a
    redshift can be fitted under ``jit`` and ``vmap`` with one compilation. An epoch with
    ``tau_days <= 0`` (before the explosion) returns ``mag_floor``.

    As redback's, the answer at one epoch depends on the LAST epoch of the call (redback sizes its
    engine grid from ``time[-1] + 100`` d), by up to ~1e-3 mag.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import numpy as np, jax.numpy as jnp
    >>> from whisper_cbpf.models.jax import supernova as sn
    >>> from whisper_cbpf.synphot import filter_set_for
    >>> fs = filter_set_for(["lsstg", "lsstr"]).to_legacy()
    >>> w, nrm = sn.ab_weights(fs["lam"], fs["trans"])
    >>> g = sn.build_fixed_grid(60.0)
    >>> p = dict(f_nickel=0.1, mej=2.0, vej=1e4, kappa=0.1, kappa_gamma=0.03,
    ...          temperature_floor=4000.0)
    >>> tau = jnp.array([-1.0, 5.0, 20.0])
    >>> m = sn.ab_magnitude_at("arnett", g, tau, p, jnp.array([0, 0, 1]), w, nrm, fs["lam"],
    ...                        0.05, 7.1e26)
    >>> [round(float(x), 2) for x in m][0]       # before the explosion: mag_floor
    40.0
    """
    spec = _spec(model)
    tau = jnp.asarray(tau_days)
    alive = tau > 0.0
    t_eval = jnp.where(alive, tau, fixed.first_epoch)        # finite, and gradient-safe, when dead
    lbol = bolometric_at(model, fixed, t_eval, params, t_last=jnp.max(tau),
                         magnetar_convention=magnetar_convention, interaction=interaction)
    sed_kw = {p: params[p] for p in spec["sed_params"] if p in params}
    mag = sn_ab_magnitude({"time": t_eval}, lbol, params[spec["vej_name"]],
                          params["temperature_floor"], band_idx, weights, norms, lam_obs_ang,
                          redshift, dl_cm, sed_kind=spec["sed"], mag_floor=mag_floor,
                          **sed_kw, **kw)
    return jnp.where(alive, mag, jnp.asarray(mag_floor, mag.dtype))


# --- priors --------------------------------------------------------------------------------

def redback_prior(model, *, drop=("redshift",)):
    """redback's OWN prior for ``model``, READ FROM redback rather than transcribed.

    Returns ``(Prior, pinned, constraints)``:

    ``Prior``        the distributions whisper can represent.
    ``pinned``       parameters redback holds FIXED. bilby parses a bare float in a prior file
                     into a ``DeltaFunction``, which is how ``type_1a.prior``'s
                     ``line_wavelength = 6.5e3``, ``line_width = 500`` and
                     ``line_amplitude = 0.3`` arrive. whisper's prior layer has no delta, so a
                     pinned parameter is bound at the factory and dropped from ``parameters``
                     -- exactly how ``kilonova_model`` handles a fixed ``temperature_floor``.
    ``constraints``  bilby ``Constraint`` entries, as ``{name: (min, max)}``. They are NOT
                     distributions -- bilby uses them to reject samples through a
                     ``conversion_function`` that is not part of the model -- so they are
                     RETURNED rather than silently dropped or silently enforced. Reproducing
                     one means adding a rejection step at the call site.

                     Only ``slsn`` ever has any, AND ONLY IN redback 1.12.0, which declares
                     ``e_rot_constraint`` in (10, 1e10) and ``t_nebula_min`` in (0.1, 500).
                     1.15.1 deleted both lines, so on that install this comes back empty and
                     the two "constraints" ``slsn``'s docstring advertises over
                     ``basic_magnetar_powered`` do not exist anywhere: not in the model (which
                     is literally the same function) and no longer in the prior. Checked, not
                     assumed -- ``get_priors("slsn").constraint_keys`` is ``[]`` on 1.15.1.

    READ rather than COPIED, deliberately: a transcription drifts, and these files differ
    between the two installed redbacks. Reading the file cannot be wrong about which redback
    is installed.

    Raises ImportError if redback is not installed; use :func:`fallback_prior` then.
    """
    from ..redback_adapter import _import_redback
    _import_redback()
    from redback.priors import get_priors

    from ...priors import LogUniform, Prior, Uniform

    rb = get_priors(model=_spec(model)["prior"] if model in MODELS else model)
    dists, pinned, constraints = {}, {}, {}
    for k, v in rb.items():
        if k in drop:
            continue
        kind = type(v).__name__
        if kind == "Uniform":
            dists[k] = Uniform(float(v.minimum), float(v.maximum))
        elif kind == "LogUniform":
            dists[k] = LogUniform(float(v.minimum), float(v.maximum))
        elif kind == "DeltaFunction":
            pinned[k] = float(v.peak)
        elif kind == "Constraint":
            constraints[k] = (float(v.minimum), float(v.maximum))
        else:
            raise NotImplementedError(
                f"redback's {model} prior has a {kind} on {k!r}, which whisper's prior layer "
                f"cannot represent. Pin it at the factory or add the distribution to "
                f"whisper_cbpf.priors.")
    return Prior(dists), pinned, constraints


#: Transcribed from the shipped prior files, for when redback is not importable. The files are
#: the source of truth -- see :func:`redback_prior`, and prefer it. Kept deliberately terse:
#: ``(name, kind, low, high)`` per parameter, then the pinned dict, then the constraints.
_COMMON_DIFFUSION = (("mej", "LogUniform", 1e-4, 100.0),
                     ("vej", "LogUniform", 1e3, 1e5),
                     ("kappa", "Uniform", 0.05, 2.0),
                     ("kappa_gamma", "LogUniform", 1e-4, 1e4),
                     ("temperature_floor", "LogUniform", 1e3, 1e5))
_MAGNETAR = (("p0", "LogUniform", 0.7, 1e4),
             ("bp", "LogUniform", 1e-4, 1e4),
             ("mass_ns", "Uniform", 1.1, 2.2),
             ("theta_pb", "Uniform", 0.0, 3.14 / 2))
_F_NICKEL = (("f_nickel", "LogUniform", 1e-3, 1.0),)

_FALLBACK = {
    "arnett": (_F_NICKEL + _COMMON_DIFFUSION, {}, {}),
    "shock_cooling_and_arnett": (
        (("log10_mass", "Uniform", -2.0, 3.0), ("log10_radius", "Uniform", 10.0, 14.0),
         ("log10_energy", "Uniform", 40.0, 52.0), ("nn", "Uniform", 8.0, 12.0),
         ("delta", "Uniform", 1.0, 1.5)) + _F_NICKEL + _COMMON_DIFFUSION, {}, {}),
    "basic_magnetar_powered": (_MAGNETAR + _COMMON_DIFFUSION, {}, {}),
    # 1.15.1 deleted slsn.prior's two Constraint lines; 1.12.0 has
    # e_rot_constraint in (10, 1e10) and t_nebula_min in (0.1, 500). Transcribed as 1.15.1,
    # like the rest of this table -- see :func:`redback_prior`.
    "slsn": ((("p0", "Uniform", 1.0, 10.0), ("bp", "LogUniform", 0.1, 10.0),
              ("mass_ns", "Uniform", 1.1, 2.2), ("theta_pb", "Uniform", 0.0, 3.14 / 2),
              ("mej", "LogUniform", 0.1, 100.0)) + _COMMON_DIFFUSION[1:], {}, {}),
    "magnetar_nickel": (_F_NICKEL + _MAGNETAR + _COMMON_DIFFUSION, {}, {}),
    "csm_shock_and_arnett": (
        (("mej", "LogUniform", 1e-4, 30.0),) + _F_NICKEL
        + (("csm_mass", "LogUniform", 1e-3, 5.0), ("v_min", "LogUniform", 1e3, 3e4),
           ("beta", "Uniform", 0.4, 0.5), ("kappa", "Uniform", 0.01, 1.0),
           ("shell_radius", "Uniform", 1e-2, 10.0),
           ("shell_width_ratio", "Uniform", 0.1, 0.5),
           ("kappa_gamma", "LogUniform", 1e-4, 1e4),
           ("temperature_floor", "LogUniform", 1e3, 1e5)), {}, {}),
    "sn_exponential_powerlaw": (
        (("lbol_0", "LogUniform", 1e36, 1e48), ("alpha_1", "Uniform", 0.0, 10.0),
         ("alpha_2", "Uniform", 0.0, 10.0), ("tpeak_d", "LogUniform", 1e-3, 200.0))
        + _COMMON_DIFFUSION, {}, {}),
    "sn_fallback": ((("logl1", "Uniform", 51.0, 57.0), ("tr", "LogUniform", 1e-4, 100.0))
                    + _COMMON_DIFFUSION, {}, {}),
    "sn_nickel_fallback": (
        (("logl1", "Uniform", 51.0, 57.0), ("tr", "LogUniform", 1e-4, 100.0),
         ("mej", "LogUniform", 1e-4, 100.0)) + _F_NICKEL + _COMMON_DIFFUSION[1:], {}, {}),
    "general_magnetar_slsn": (
        (("l0", "LogUniform", 1e-40, 1e48), ("tsd", "LogUniform", 1e2, 1e6),
         ("nn", "Uniform", 1.1, 7.0)) + _COMMON_DIFFUSION, {}, {}),
    "type_1a": (_F_NICKEL + _COMMON_DIFFUSION,
                {"line_wavelength": 6.5e3, "line_width": 500.0, "line_amplitude": 0.3}, {}),
    "type_1c": (_F_NICKEL + _COMMON_DIFFUSION + (("pp", "Uniform", 1.1, 4.0),), {}, {}),
}


def fallback_prior(model):
    """:func:`redback_prior` without redback. Same three-tuple return shape; see ``_FALLBACK``."""
    from ...priors import LogUniform, Prior, Uniform

    dists, pinned, constraints = _FALLBACK[model]
    cls = {"Uniform": Uniform, "LogUniform": LogUniform}
    return (Prior({n: cls[k](lo, hi) for n, k, lo, hi in dists}),
            dict(pinned), dict(constraints))


def default_prior(model):
    """``(Prior, pinned, constraints)``: redback's, read from redback where possible.

    NOT modified, narrowed or "improved". If a fit should use a different range, set it at the
    call site, where the choice is visible in the analysis rather than buried in a library
    default. Two of redback's own ranges are worth reading before you adopt them: ``vej`` is
    ``LogUniform(1e3, 1e5)`` km/s, whose upper end is 0.33c, and ``general_magnetar_slsn``'s
    ``tsd`` is labelled seconds but consumed as days (see :func:`magnetar_only`).
    """
    try:
        return redback_prior(model)
    except ImportError:
        return fallback_prior(model)


# --- self-test -----------------------------------------------------------------------------
