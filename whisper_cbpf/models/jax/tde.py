"""JAX implementation of redback's self-contained TDE models.

Depends on jax and numpy only. The blackbody, the AB band integral and the filter export come from
:mod:`whisper_cbpf.models.jax.kilonova`; nothing is copied.

Ported: ``_analytic_fallback``, ``_cooling_envelope`` (the Sarin & Metzger envelope ODE),
``_stream_stream_collision``, ``TemperatureFloor``, ``TDEPhotosphere``, ``CocoonPhotosphere``,
``Diffusion``, ``AsphericalDiffusion``, ``Viscous``, ``CSMDiffusion``, a photometric layer (flux
density and AB band magnitude), and the Gaussian-rise stitched light curve.

**float64 is required.** The Euler integration accumulates increments ~1e-6 of the state, which
float32 cannot resolve, so the engine raises rather than returning a plausible wrong answer::

    import jax; jax.config.update("jax_enable_x64", True)   # before the first array exists

The defaults that differ from one or other redback release -- ``n_time`` (the installed redback's
grid, see :func:`default_n_time`), ``dilation=True``, and the envelope's own termination
criterion -- and every other deviation, with the measurements behind them, are in
``docs/PORTING_NOTES.md``.

The expressions below keep redback's exact spelling on purpose: a forward-Euler recursion
amplifies round-off, so respelling ``Rv ** 2`` as ``Rv * Rv`` is not a no-op. Do not simplify them.
"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .kilonova import (
    DAY_TO_S,
    INV_AB_ZEROPOINT,
    MJY,
    PLANCK_ARG,
    PLANCK_T3,
    PLANCK_T3_AB,
    RATIO_CONST,
    RSCALE,
    DLSCALE,
    NUSCALE,
    SIGMA_SB,
    SOLAR_MASS,
    SPEED_OF_LIGHT,
    _ext_transmission,
    _inverse_expm1,
    ab_weights,           # noqa: F401  (re-exported: the caller needs it to build weights)
    extinction_shape,     # noqa: F401
    make_filter_set,      # noqa: F401
)

__all__ = [
    "analytic_fallback", "calc_tfb", "gaussian_rise", "exponential_powerlaw",
    "cooling_envelope", "cooling_envelope_jit", "ryu_f_debris",
    "stream_stream_collision",
    "temperature_floor_photosphere", "tde_photosphere", "cocoon_photosphere",
    "build_interaction_grid", "diffusion", "aspherical_diffusion", "viscous",
    "csm_diffusion",
    "flux_density_mjy", "ab_magnitude",
    "cooling_envelope_flux_density", "cooling_envelope_ab_magnitude",
    "gaussianrise_cooling_envelope_ab_magnitude",
    "gaussianrise_cooling_envelope_flux_density",
    "PARAMETERS", "PARAMETERS_GAUSSIANRISE", "default_prior", "default_prior_gaussianrise",
    "redback_prior", "fallback_prior",
    "envelope_exists", "rise_peaks_near_fallback",
    "N_TIME", "REDBACK_PRESETS", "REDBACK_ENGINE_PRESETS", "default_n_time",
]

# --- constants not already in kilonova.py (redback/constants.py, astropy cgs) ----------
GRAVITATIONAL_CONSTANT = 6.6743e-8       # cm^3 g^-1 s^-2
SOLAR_RADIUS = 6.957e10                  # cm
KM_CGS = 1e5                             # cm
STEF_CONSTANT = 4.0 * np.pi * SIGMA_SB   # redback/photosphere.py
RADIUS_CONSTANT = KM_CGS * DAY_TO_S      # redback/photosphere.py

#: redback 1.12.0's grid size for `_cooling_envelope`. See CHANGE 6 -- this is part of the
#: model, not a tolerance. No longer the default: that is :func:`default_n_time`.
N_TIME = 5000

#: Exact settings to reproduce each installed redback's ENGINE. See CHANGE 6. The termination
#: RULE is not a preset: it is the model's own physical criterion in both cases (CHANGE 3).
#: 1.20 integrates on 1.15's 500 points (``tde_models.py:105``).
REDBACK_ENGINE_PRESETS = {
    "1.12": dict(n_time=5000),
    "1.15": dict(n_time=500),
    "1.20": dict(n_time=500),
}

#: ... and its PHOTOMETRY, which additionally differs by a ``(1+z)`` flux factor that
#: 1.15.1 applies and 1.12.0 does not (CHANGE 7). Use as
#: ``cooling_envelope_flux_density(..., **REDBACK_PRESETS["1.15"])``; for the bare engine
#: use ``REDBACK_ENGINE_PRESETS``, which omits ``dilation``.
REDBACK_PRESETS = {
    "1.12": dict(**REDBACK_ENGINE_PRESETS["1.12"], dilation=False),
    "1.15": dict(**REDBACK_ENGINE_PRESETS["1.15"], dilation=True),
    "1.20": dict(**REDBACK_ENGINE_PRESETS["1.20"], dilation=True),
}


def default_n_time():
    """``n_time`` of the INSTALLED redback's grid: the default of every function here.

    500 for redback 1.15 and 1.20, and when redback is not installed (the latest release); 5000
    for 1.12. Read from redback's source without importing it
    (:func:`whisper_cbpf.models.redback_adapter.installed_redback_preset`), so the port and a
    redback fit in the same environment integrate the same grid by default.

    500 is 10x coarser than 5000 and it is not free: the photosphere temperature is 1.7% from
    converged, and the termination index moves by grid steps under a 1e-6 change of a parameter
    (CHANGE 6, ``docs/PORTING_NOTES.md`` section 6.1). Pass ``n_time=5000`` for the finer grid.
    """
    from ..redback_adapter import LATEST_REDBACK_PRESET, installed_redback_preset

    return REDBACK_ENGINE_PRESETS[installed_redback_preset() or LATEST_REDBACK_PRESET]["n_time"]

# --- guard sentinels (CHANGE 4) ---------------------------------------------------------
# These are substituted for Ee and Me in the GUARDED half of the step only -- the half that
# produces every returned VALUE. They never touch the RAW half, which is what decides where
# redback stops. Keeping those two jobs in one trajectory does not work, and the failure is
# not subtle in either direction:
#
#   SENTINELS IN THE STOPPING TRAJECTORY: Rv jumps to +1.1e23 cm, so tacc ~ Rv^2 becomes
#   astronomically large, MdotBH = Me/tacc collapses to ~0, and the envelope GROWS instead of
#   draining. redback's Me goes negative within 1-3 steps of Ee crossing zero; this never
#   does, so `Me < 0` never fires, and through redback's shared try/except that sends BOTH
#   constraints to len(time_temp). Measured: 6.55% of the model's own prior box stopped
#   terminating, reporting photosphere radii to -5e24 cm and temperatures of 0.06 K INSIDE
#   the region it called valid, with gradients that were finite, smooth, confident, and wrong
#   by 3-5 orders of magnitude and in sign.
#
#   NO SENTINELS AT ALL (guarding only division by an exact zero, preserving Ee's sign):
#   the stopping index comes back right, and 27 of 40 prior draws come back with NaN
#   gradients -- the defect this whole section exists to prevent.
#
# The resolution is neither: STOP READING THE DIVERGED QUANTITY. CHANGE 3 replaces redback's
# stopping test with the model's own criterion, which is exact while the envelope lives, so
# the guards below can no longer move where the curve ends. Both sentinels are positive and
# O(1) in their own units, so nothing downstream of them can overflow. Verified: with them
# removed, every returned array is BIT-IDENTICAL inside `valid` over 300 draws at each grid.
_EE_DEAD = 1.0                # 1e40 erg
_ME_DEAD = SOLAR_MASS         # g
R_MIN = 1.0                   # cm^2, for the one case Rph == 0 exactly (log(Lamb) == -1)
#: Floor for luminosities that are raised to a fractional power. redback's TDE
#: luminosities contain EXACT zeros (zero-padding outside the simulated range, and
#: NaN -> 0 substitutions). 0**0.25 is 0 in the forward pass but its derivative is
#: infinite, so an unfloored version returns NaN gradients from points that contribute
#: nothing at all. The floor changes no forward value above it.
_L_FLOOR = 1e-30


def _require_x64(what):
    """CHANGE 8. Fail at trace time with the fix, rather than returning inf later."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} requires float64. The envelope ODE accumulates increments that are "
            f"~1e-6 of the state, which float32 (eps 1.2e-7) cannot resolve: measured, "
            f"`constraint` collapses to 1 and the luminosity is inf. Enable it BEFORE the "
            f"first jax array is created:\n"
            f"    import jax; jax.config.update('jax_enable_x64', True)\n"
            f"or set JAX_ENABLE_X64=1 in the environment.")


# --- small closed-form pieces --------------------------------------------------------------
#
# ORIGINAL (redback/utils.py:141):
#     tfb = 58. * (3600. * 24.) * (mbh_6 ** (0.5)) * (stellar_mass ** (0.2)) \
#           * ((binding_energy_const / 0.8) ** (-1.5))

def calc_tfb(binding_energy_const, mbh_6, stellar_mass):
    """Fall-back time of the most tightly bound debris, in SECONDS. Pure arithmetic."""
    return (58. * (3600. * 24.) * mbh_6 ** 0.5 * stellar_mass ** 0.2
            * (binding_energy_const / 0.8) ** -1.5)


# ORIGINAL (phenomenological_models.py:733 / :744). Unchanged.
def gaussian_rise(time, a_1, peak_time, sigma_t):
    """a_1 * exp(-(t - t_peak)^2 / 2 sigma^2). `time` and the two scales share units."""
    return a_1 * jnp.exp(-(time - peak_time) ** 2. / (2 * sigma_t ** 2))


def exponential_powerlaw(time, a_1, alpha_1, alpha_2, tpeak):
    """a_1 (1 - exp(-t/tpeak))^alpha_1 (t/tpeak)^-alpha_2."""
    return a_1 * (1 - jnp.exp(-time / tpeak)) ** alpha_1 * (time / tpeak) ** -alpha_2


#
# _analytic_fallback
#
# ORIGINAL (redback tde_models.py:15):
#     mask = time - t_0 > 0.
#     lbol = np.zeros(len(time))
#     lbol[mask]  = l0 / (time[mask] * 86400)**(5./3.)
#     lbol[~mask] = l0 / (t_0 * 86400)**(5./3.)
#
# CHANGE 2 (representation only): boolean-mask assignment into a preallocated array is an
# in-place write, which JAX arrays do not support. Both branches are finite for time > 0
# and t_0 > 0, so a plain where is exact -- but `time` may legitimately be <= 0 on a grid
# that starts at the trigger, and t^(-5/3) is then NaN with an infinite derivative, so the
# unselected branch is sanitised first (CHANGE 4's discipline, applied to a trivial case).

def analytic_fallback(time, l0, t_0):
    """Bolometric luminosity: t^-5/3 fall-back with a flat plateau before ``t_0``.

    time, t_0 : days.  l0 : bolometric luminosity at 1 second, cgs.
    """
    t_safe = jnp.where(time > 0.0, time, 1.0)
    decay = l0 / (t_safe * DAY_TO_S) ** (5. / 3.)
    plateau = l0 / (t_0 * DAY_TO_S) ** (5. / 3.)
    return jnp.where(time - t_0 > 0., decay, plateau)


# ORIGINAL: redback's `calculate_f_debris=True` branch (newer than 1.12.0).
def ryu_f_debris(mbh_6, stellar_mass, beta):
    """Ryu et al. partial-disruption debris fraction, clipped to [0, 1].

    Not present in redback 1.12.0, where the disrupted mass is always the full stellar
    mass. ``f_debris = 1.0`` in :func:`cooling_envelope` reproduces 1.12.0 exactly.
    """
    f_mbh = 0.80 + 0.26 * mbh_6 ** 0.5
    ex = jnp.exp((stellar_mass - 0.669) / 0.137)
    g_mstar = (1.47 + ex) / (1 + 2.34 * ex)
    return jnp.clip((beta * f_mbh * g_mstar) ** 3, 0.0, 1.0)


# --- _cooling_envelope  (Sarin & Metzger 2024) ---------------------------------------------
#
# ORIGINAL loop body (redback tde_models.py:123-149, `for ii in range(1, len(time_temp))`):
#
#     Me[ii]       = Me[ii-1] - (MdotBH[ii-1] - Mdotfb[ii-1])*(t[ii]-t[ii-1])
#     Ee40[ii]     = Ee40[ii-1] + (Ledd40 - Edotbh40[ii-1])*(t[ii]-t[ii-1])
#     Rv[ii]       = ((2.0*G*mbh_6*1e6*Me[ii])/(5.0*Ee40[ii]))*(2.0e-7)
#     Lamb[ii]     = 0.38*Me[ii]/(10.0*np.pi*Rv[ii]**2)
#     Rph[ii]      = Rv[ii]*(1.0 + np.log(Lamb[ii]))
#     Racc[ii]     = zeta*Rv[0]*(t[ii]/tfb)**(2./3.)
#     Edotfb40[ii] = (G*mbh_6*1e6*Mdotfb[ii]/Racc[ii])*(2.0e-7)
#     Lrad[ii]     = Ledd40 + Edotfb40[ii]
#     Teff[ii]     = 1.0e10*((Ledd40 + Edotfb40[ii])/(4.0*np.pi*sigma_sb*Rph[ii]**2))**0.25
#     tacc[ii]     = 2.2e-17*(10./(3.*alpha))*(Rv[ii]**2)/(G*mbh_6*1e6*Rcirc)**0.5*(hoverR)**(-2.0)
#     MdotBH[ii]   = Me[ii]/tacc[ii]
#     LX40[ii]     = 0.01*(MdotBH[ii]/1.0e20)*(c**2/1.0e20)
#     Edotbh40[ii] = eta*c**2*(Me[ii]/tacc[ii])*(1.0e-40)
#
# See CHANGE 1 (scan), 4 (guards) and 5 (spelling). Nothing else moves.

def _tacc(rv2, mbh_6, alpha, hoverR, Rcirc):
    """redback's accretion timescale, spelled exactly as redback spells it (CHANGE 5)."""
    return (2.2e-17 * (10. / (3.0 * alpha)) * rv2
            / (GRAVITATIONAL_CONSTANT * mbh_6 * 1.0e6 * Rcirc) ** 0.5 * hoverR ** -2.0)


def _cooling_envelope_step(carry, xs, mbh_6, Ledd40, zeta, Rv0, tfb,
                           alpha, hoverR, Rcirc, eta):
    """One forward-Euler step, guarded so every value is finite. See CHANGE 4.

    The guards fire only where the envelope is already dead (``Ee <= 0`` or ``Me <= 0``),
    which CHANGE 3's stopping rule has already excluded from ``valid``. Verified over 300
    prior draws at each grid, against an unguarded copy: ``constraint`` identical 300/300,
    and all thirteen returned arrays BIT-IDENTICAL inside ``valid`` -- so the guards buy
    finiteness outside the light curve at no cost inside it.
    """
    Me_p, Ee_p, MdotBH_p, Edotbh_p = carry
    dt, t_ii, mdotfb_ii, mdotfb_p = xs

    Me = Me_p - (MdotBH_p - mdotfb_p) * dt
    Ee = Ee_p + (Ledd40 - Edotbh_p) * dt

    # The envelope exists while it has energy and mass. Past that point redback's Rv diverges
    # and everything downstream of it is chaotic, so the inputs to the two divisions are
    # sanitised here and the index is excluded from `valid` anyway. While the envelope lives
    # both `where`s select the untouched value, so the trajectory is bit-identical to
    # redback's -- which is the only place it needs to be.
    alive = (Ee > 0.0) & (Me > 0.0)
    Ee_safe = jnp.where(alive, Ee, _EE_DEAD)
    Me_safe = jnp.where(alive, Me, _ME_DEAD)

    Rv = ((2.0 * GRAVITATIONAL_CONSTANT * mbh_6 * 1.0e6 * Me_safe) / (5.0 * Ee_safe)) * 2.0e-7
    Lamb = 0.38 * Me_safe / (10.0 * jnp.pi * Rv ** 2.0)   # > 0: Me_safe > 0 and Rv != 0
    Rph = Rv * (1.0 + jnp.log(Lamb))

    Racc = zeta * Rv0 * (t_ii / tfb) ** (2. / 3.)
    Edotfb = (GRAVITATIONAL_CONSTANT * mbh_6 * 1.0e6 * mdotfb_ii / Racc) * 2.0e-7
    Lrad = Ledd40 + Edotfb
    # Rph vanishes where log(Lamb) == -1 exactly; the numerator is Ledd40 + Edotfb > 0, so
    # only the denominator can misbehave, and only there.
    rph2 = jnp.where(Rph == 0.0, R_MIN, Rph ** 2.0)
    Teff = 1.0e10 * ((Ledd40 + Edotfb) / (4.0 * jnp.pi * SIGMA_SB * rph2)) ** 0.25

    tacc = _tacc(Rv ** 2.0, mbh_6, alpha, hoverR, Rcirc)
    MdotBH = Me_safe / tacc
    LX40 = 0.01 * (MdotBH / 1.0e20) * (SPEED_OF_LIGHT ** 2.0 / 1.0e20)
    Edotbh = eta * SPEED_OF_LIGHT ** 2.0 * (Me_safe / tacc) * 1.0e-40

    return ((Me, Ee, MdotBH, Edotbh),
            (Me, Ee, Rv, Lamb, Rph, Racc, Edotfb, Lrad, Teff, tacc, MdotBH, LX40, alive))


def cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, *,
                     f_debris=1.0, t_0_init=1.0, binding_energy_const=0.8,
                     zeta=2.0, hoverR=0.3, n_time=None):
    """Sarin & Metzger (2024) cooling-envelope TDE engine.

    Parameters
    ----------
    mbh_6 : SMBH mass in 1e6 solar masses.
    stellar_mass : disrupted star mass in solar masses.
    eta : SMBH feedback efficiency.
    alpha : disk viscosity.
    beta : TDE penetration factor (sets the circularisation radius, Rcirc = 2 Rt / beta).
    f_debris : fraction of the star actually disrupted. **1.0 reproduces redback 1.12.0
        exactly**, which has no such parameter; the argument exists so a newer redback's
        partial-disruption branch can be reproduced too (see :func:`ryu_f_debris`).
    t_0_init, binding_energy_const, zeta, hoverR : redback's kwargs, same defaults.
    n_time : grid size. **This is part of the model, not a tolerance** -- see CHANGE 6.
        1.12.0 uses 5000, 1.15.1 and 1.20 use 500, and the difference is 1.7% in the
        photosphere temperature. ``None`` (default) is the installed redback's,
        :func:`default_n_time`.

    Returns
    -------
    dict with, all of shape ``(n_time,)`` unless noted:
        ``time_temp``               source-frame seconds, from tfb to 5000 tfb
        ``time_since_fb``           ``time_temp - time_temp[0]``
        ``valid``                   bool mask replacing redback's ``[:constraint]``
        ``meaningful``              ``valid`` with the ill-conditioned tail trimmed (CHANGE 3)
        ``bolometric_luminosity``   erg/s
        ``photosphere_temperature`` K
        ``photosphere_radius``      cm
        ``envelope_radius``         cm      ``envelope_mass`` g
        ``lum_xray``                1e40 erg/s (proxy; redback leaves index 0 uninitialised
                                    and it is set to 0.0 here)
        ``accretion_radius`` cm, ``SMBH_accretion_rate`` g/s
        scalars: ``tfb`` s, ``rtidal`` cm, ``rcirc`` cm, ``constraint``, ``termination_time`` s

    Notes
    -----
    Outside jit, ``arr[np.asarray(out["valid"])]`` reproduces redback's slice.
    ``constraint`` may be 0 (redback raises IndexError in that case; here the mask is
    simply all-False).
    """
    _require_x64("whisper_cbpf.models.jax.tde.cooling_envelope")
    n_time = default_n_time() if n_time is None else n_time

    Mstar = stellar_mass * SOLAR_MASS
    Mdisrupt = f_debris * Mstar
    Rstar = stellar_mass ** 0.8 * SOLAR_RADIUS
    Rt = Rstar * (mbh_6 * 1.0e6 / stellar_mass) ** (1. / 3.)
    Rcirc = 2.0 * Rt / beta
    tfb = calc_tfb(binding_energy_const, mbh_6, stellar_mass * f_debris)
    Ledd40 = 1.4e4 * mbh_6

    time_temp = jnp.logspace(jnp.log10(1.0 * tfb), jnp.log10(5000 * tfb), n_time)
    Mdotfb = (0.8 * Mdisrupt / (3.0 * tfb)) * (time_temp / tfb) ** (-5. / 3.)

    # --- grid point 0, verbatim from redback (lines 97-120) ---
    Me0 = f_debris * (0.1 * Mstar + (0.4 * Mstar) * (1.0 - t_0_init ** (-2. / 3.)))
    Rv0 = (2. * Rt ** 2.0 / (5.0 * binding_energy_const * Rstar)) * (Me0 / (f_debris * Mstar))
    Ee0 = ((2.0 * GRAVITATIONAL_CONSTANT * mbh_6 * 1.0e6 * Me0) / (5.0 * Rv0)) * 2.0e-7
    Lamb0 = 0.38 * Me0 / (10.0 * jnp.pi * Rv0 ** 2.0)
    Rph0 = Rv0 * (1.0 + jnp.log(Lamb0))
    Racc0 = zeta * Rv0
    Edotfb0 = (GRAVITATIONAL_CONSTANT * mbh_6 * 1.0e6 * Mdotfb[0] / Racc0) * 2.0e-7
    Lrad0 = Ledd40 + Edotfb0
    tacc0 = (2.2e-17 * (10. / (3. * alpha)) * Rv0 ** 2.0
             / (GRAVITATIONAL_CONSTANT * mbh_6 * 1.0e6 * Rcirc) ** 0.5 * hoverR ** -2.0)
    MdotBH0 = Me0 / tacc0
    Edotbh0 = eta * SPEED_OF_LIGHT ** 2.0 * (Me0 / tacc0) * 1.0e-40
    Teff0 = 1.0e10 * ((Ledd40 + Edotfb0) / (4.0 * jnp.pi * SIGMA_SB * Rph0 ** 2.0)) ** 0.25
    LX0 = 0.0                       # redback allocates with np.empty and never writes [0]

    # --- CHANGE 1: the loop is a scan ---
    dt = jnp.diff(time_temp)
    xs = (dt, time_temp[1:], Mdotfb[1:], Mdotfb[:-1])
    step = partial(_cooling_envelope_step, mbh_6=mbh_6, Ledd40=Ledd40, zeta=zeta,
                   Rv0=Rv0, tfb=tfb, alpha=alpha, hoverR=hoverR, Rcirc=Rcirc, eta=eta)
    # unroll=16 is worth 5.2x. It is NOT bit-identical -- an earlier version of this comment
    # claimed it was, and that was measured false: changing only the unroll token moves
    # `constraint` on 5.0% of prior draws at n_time=500 (max 3 steps) and 10.0% at 5000 (max
    # 12). The differences are <= 4e-14 relative over the first half of the curve and only
    # become O(1) past 98%, i.e. they live entirely in the ill-conditioned tail, and the
    # photometric effect over 5-95% of the curve is a median of 0 with a p99 of 5.9e-6 mag
    # at n_time=500 and 3.6e-15 at 5000. (An earlier version of this comment said 8.7%/18.7%
    # and a p99 of 5.0e-3 mag -- measured from a smaller sample, and pessimistic by three
    # orders of magnitude on the photometry. Numbers here are the T2 sweep's.)
    # That is the same scale as the model's own irreproducibility (redback disagrees with
    # itself by up to 12-14 steps under a one-ulp parameter nudge), so it is kept -- but it
    # does mean an XLA scheduling constant is part of the answer, and it is recorded here
    # rather than asserted away.
    # The scan is pure per-iteration launch overhead -- time per step is 13.7-17.4 us and is
    # flat in both n_time and batch size, so it is latency, not arithmetic. Measured at
    # n_time=5000: 167 -> 58 ms at batch 1, 119 -> 23.0 ms at batch 64, 176 -> 33.0 ms at
    # batch 1024. It costs 0.5 -> 7.7 s of compile, repaid after ~80 evaluations.
    # unroll=64 was measured and is WORSE (49.9 ms at batch 1024); 16 is the optimum.
    _, ys = jax.lax.scan(step, (Me0, Ee0, MdotBH0, Edotbh0), xs, unroll=16)

    def _join(head, tail):
        return jnp.concatenate([jnp.atleast_1d(jnp.asarray(head, tail.dtype)), tail])

    (Me, Ee, Rv, Lamb, Rph, Racc, Edotfb, Lrad, Teff, tacc, MdotBH, LX40) = (
        _join(h, t) for h, t in zip(
            (Me0, Ee0, Rv0, Lamb0, Rph0, Racc0, Edotfb0, Lrad0, Teff0, tacc0, MdotBH0, LX0),
            ys[:-1]))
    alive = jnp.concatenate([jnp.array([True]), ys[-1]])

    # --- CHANGE 2 + 3: the light curve stops where the ENVELOPE stops, on one trajectory.
    #
    # redback stops at  min( first(Rv < Rcirc/2), first(Me < 0) )  -- inside a shared
    # try/except, so if EITHER search comes up empty BOTH indices become len(time_temp) and
    # the whole 5000-point curve is returned. Reproducing that exactly is not possible and
    # not desirable, and both halves of that sentence are measurements, not opinions:
    #
    #   NOT POSSIBLE. `Me < 0` is reached one to three steps AFTER Ee crosses zero, i.e.
    #   after Rv has already diverged, so the index depends on chaotic post-divergence
    #   arithmetic. Two implementations of the identical recursion, differing by one ulp in
    #   beta, disagree about `constraint` in 10% of prior draws (max 4 steps). A version of
    #   this file that carried redback's raw trajectory alongside the guarded one -- purely
    #   to reproduce that index -- still disagreed by up to 1793 steps, and could not be made
    #   to agree, because there is no stable answer to agree with.
    #
    #   NOT DESIRABLE. The physical statement is simply: the envelope is over when it runs
    #   out of energy, runs out of mass, or has shrunk to the circularisation radius. Each of
    #   the three terms below is EXACT while the envelope lives, which is the only place any
    #   of them needs to be. The shared try/except is an artefact of exception handling, not
    #   a physical claim -- and it is what turned "one of the two searches found nothing"
    #   into "return 5000 points of post-mortem".
    #
    # DEVIATION, stated plainly: where `Rv < Rcirc/2` fires while the envelope is still alive
    # AND `Me` never goes negative, redback returns the whole curve and this returns the
    # circularisation time. That is redback's accidental branch and this is the model's own
    # criterion. Everywhere else the two agree to within the +-4 steps that redback agrees
    # with itself.
    n = n_time
    stop = (~alive) | (Rv < Rcirc / 2.)
    constraint = jnp.where(jnp.any(stop), jnp.argmax(stop), n)   # argmax of a bool IS min(where)

    valid = jnp.arange(n) < constraint
    # `meaningful` is `valid` narrowed by the model's OWN domain rather than by a percentage
    # of it: the envelope must still be optically thick enough to put its photosphere outside
    # itself, Lamb > 1/e, i.e. Rph > 0. Below that redback's Rph goes negative and is then fed
    # to a blackbody as R^2, which reports a positive flux from a negative radius -- 1.77% of
    # prior draws. An earlier version trimmed a measured 15% of the curve instead; a fraction
    # of an index is not a physical statement, and it trimmed good points on short curves
    # while missing bad ones on long.
    meaningful = valid & (Rph > 0.0) & (Ee > 0.0) & (Me > 0.0)
    termination_time = jnp.where(constraint == n,
                                 time_temp[-1] - tfb,
                                 time_temp[jnp.minimum(constraint, n - 1)] - tfb)

    return dict(time_temp=time_temp, time_since_fb=time_temp - time_temp[0],
                valid=valid, meaningful=meaningful,
                bolometric_luminosity=Lrad * 1e40,
                photosphere_temperature=Teff, photosphere_radius=Rph,
                envelope_radius=Rv, envelope_mass=Me, envelope_energy=Ee,
                optical_depth=Lamb, lum_xray=LX40, accretion_radius=Racc,
                SMBH_accretion_rate=MdotBH, accretion_timescale=tacc,
                tfb=tfb, rtidal=Rt, rcirc=Rcirc,
                constraint=constraint, termination_time=termination_time)


cooling_envelope_jit = jax.jit(cooling_envelope, static_argnames=("n_time",))


# --- _stream_stream_collision --------------------------------------------------------------
#
# ORIGINAL (redback tde_models.py:1305, abridged):
#     kappa = 0.34; rstar = 0.93*mstar**(8/9); mstar_max = 15.0
#     Xi = (1.27 - 0.3*mbh_6**0.242) * ((0.620 + exp((min(mstar_max,mstar)-0.674)/0.212))
#          / (1.0 + 0.553*exp((min(mstar,mstar_max)-0.674)/0.212)))
#     r_tidal = (mbh_6*1e6/mstar)**(1/3)*rstar*solar_radius
#     epsilon = G*(mbh_6*1e6*Msun)*(rstar*Rsun)/r_tidal**2
#     a0 = G*(mbh_6*1e6*Msun)/(Xi*epsilon)
#     t_dyn = pi/sqrt(2)*a0**1.5/sqrt(G*(mbh_6*1e6*Msun));  t_peak = 1.5*t_dyn
#     mdotmax = mstar*Msun/t_dyn/3.0
#     factor_denom = del_omega*sigma_sb*c1**2*a0**2
#     if inc_tcool == 1:
#         semi = a0/2.0; area = pi*(c1*semi)**2
#         tau = kappa*(f*mstar*Msun/2.0)/area/2.0
#         tcool = tau*(h_r)*c1*semi/c;  factor = 2.0/(1.0 + tcool/t_dyn)
#         factor_denom *= (1.0 + 2.0*h_r)/4.0
#     t_output = np.linspace(t_peak, 1500*day_to_s, 1000)
#     Lobs = mdotmax*(Xi*epsilon)/c1*(t_output/t_peak)**(-5/3)*factor
#     Tobs = (Lobs/factor_denom)**(1/4)
#
# CHANGES: (i) Python's builtin min() cannot take a traced value -> jnp.minimum, same
# value; (ii) `inc_tcool` branches on CONFIGURATION, not data, so it is static and Python
# resolves it at trace time -- lax.cond would compile the untaken branch for nothing;
# (iii) n_time is static because it sets a shape. No guard is needed anywhere: every
# quantity is a positive power of a positive parameter.

@partial(jax.jit, static_argnames=("inc_tcool", "n_time"))
def stream_stream_collision(mbh_6, mstar, c1, f, h_r, del_omega,
                            inc_tcool=0, n_time=1000):
    """Stream-stream collision TDE. Returns ``(t_s, L_bol, T_phot, t_dyn, t_peak, r_tidal)``.

    Times in seconds, ``L_bol`` erg/s, ``T_phot`` K, radii cm.
    """
    kappa = 0.34
    rstar = 0.93 * mstar ** (8.0 / 9.0)
    mstar_max = 15.0

    m_clip = jnp.minimum(mstar, mstar_max)
    ex = jnp.exp((m_clip - 0.674) / 0.212)
    Xi = (1.27 - 0.3 * mbh_6 ** 0.242) * ((0.620 + ex) / (1.0 + 0.553 * ex))

    mbh_cgs = mbh_6 * 1e6 * SOLAR_MASS
    r_tidal = (mbh_6 * 1e6 / mstar) ** (1.0 / 3.0) * rstar * SOLAR_RADIUS
    epsilon = GRAVITATIONAL_CONSTANT * mbh_cgs * (rstar * SOLAR_RADIUS) / r_tidal ** 2
    a0 = GRAVITATIONAL_CONSTANT * mbh_cgs / (Xi * epsilon)

    t_dyn = (jnp.pi / jnp.sqrt(2.0) * a0 ** 1.5
             / jnp.sqrt(GRAVITATIONAL_CONSTANT * mbh_cgs))
    t_peak = 1.5 * t_dyn
    mdotmax = mstar * SOLAR_MASS / t_dyn / 3.0
    factor_denom = del_omega * SIGMA_SB * c1 ** 2 * a0 ** 2
    factor = 1.0

    if inc_tcool == 1:                              # static: configuration, not data
        semi = a0 / 2.0
        area = jnp.pi * (c1 * semi) ** 2
        tau = kappa * (f * mstar * SOLAR_MASS / 2.0) / area / 2.0
        tcool = tau * h_r * c1 * semi / SPEED_OF_LIGHT
        factor = 2.0 / (1.0 + tcool / t_dyn)
        factor_denom = factor_denom * (1.0 + 2.0 * h_r) / 4.0

    t_output = jnp.linspace(t_peak, 1500 * DAY_TO_S, n_time)
    Lmax = mdotmax * (Xi * epsilon) / c1
    Lobs = Lmax * (t_output / t_peak) ** (-5.0 / 3.0) * factor
    Tobs = (Lobs / factor_denom) ** 0.25
    return t_output, Lobs, Tobs, t_dyn, t_peak, r_tidal


# --- photosphere.py ------------------------------------------------------------------------
#
# These are classes in redback whose __init__ runs the whole calculation and stores the
# result on self. Here they are plain functions: there is no state to keep, and a class
# instance is not a pytree unless it is registered as one. The property chain
# (rt -> rp -> a_p -> ... -> r_photosphere) is just intermediate variables.
#
# CHANGE common to all: the luminosity is floored to _L_FLOOR wherever it is raised to a
# fractional power (see the constant's comment). No forward value above the floor moves.


def temperature_floor_photosphere(time, luminosity, vej, temperature_floor):
    """redback ``photosphere.TemperatureFloor``. time days, luminosity erg/s, vej km/s.

    ORIGINAL (photosphere.py:86-104):
        radius_squared     = (RADIUS_CONSTANT * v_ejecta * time) ** 2
        rec_radius_squared = luminosity / (STEF_CONSTANT * temperature_floor ** 4)
        mask               = radius_squared <= rec_radius_squared
        r_photosphere            = rec_radius_squared ** 0.5
        r_photosphere[mask]      = radius_squared[mask] ** 0.5
        photosphere_temperature[mask]  = (luminosity[mask] /
                                          (STEF_CONSTANT * radius_squared[mask])) ** 0.25
        photosphere_temperature[~mask] = temperature_floor

    CHANGE 2/4: masked in-place assignment -> where. Both branches are evaluated before
    selection, so both must be finite: radius_squared is zero at time = 0 and 0**0.25 has
    an infinite derivative, so it is floored. The SELECTION is unchanged, so which branch
    each point takes is identical to redback's.

    Returns ``(temperature K, r_photosphere cm)``.
    """
    radius_squared = (RADIUS_CONSTANT * vej * time) ** 2
    rec_radius_squared = luminosity / (STEF_CONSTANT * temperature_floor ** 4)
    mask = radius_squared <= rec_radius_squared

    r2_safe = jnp.maximum(radius_squared, _L_FLOOR)
    rec_safe = jnp.maximum(rec_radius_squared, _L_FLOOR)
    r_photosphere = jnp.where(mask, r2_safe ** 0.5, rec_safe ** 0.5)

    t_free = (jnp.maximum(luminosity, _L_FLOOR) / (STEF_CONSTANT * r2_safe)) ** 0.25
    temperature = jnp.where(mask, t_free, temperature_floor)
    return temperature, r_photosphere


def tde_photosphere(time, luminosity, mass_bh, mass_star, star_radius,
                    tpeak, beta, rph_0, lphoto):
    """redback ``photosphere.TDEPhotosphere``. Photosphere that expands as a power of Mdot.

    ORIGINAL (photosphere.py:146-203, property chain):
        kappa_t = 0.2*(1 + 0.74)
        rt = (mass_bh/mass_star)**(1/3) * star_radius*solar_radius;   rp = rt/beta
        a_p = (G*mass_bh_si*(tpeak*day_to_s/pi)**2)**(1/3)
        a_t = (G*mass_bh_si*(time *day_to_s/pi)**2)**(1/3)
        r_isco = 6*G*mass_bh_si/c**2;   r_photo_max = rp + 2*a_t
        eddington_luminosity = 4*pi*G*mass_bh_si*c/kappa_t
        rphot = rph_0*a_p*(luminosity/eddington_luminosity)**lphoto
        r_photosphere = (rphot*r_photo_max)/(rphot + r_photo_max) + r_photo_min

    There is no branching at all here, so this is a direct transcription; the soft-min /
    soft-max construction is already smooth and differentiable as written. Only the
    luminosity floor is added, because redback's TDE luminosity arrays contain exact
    zeros and ``0**lphoto`` has an infinite derivative.

    time days, masses in Msun, star_radius in Rsun. Returns ``(T K, R_phot cm, rp cm)``.
    """
    kappa_t = 0.2 * (1 + 0.74)
    star_radius_cgs = star_radius * SOLAR_RADIUS
    mass_bh_cgs = mass_bh * SOLAR_MASS

    rt = (mass_bh / mass_star) ** (1. / 3.) * star_radius_cgs
    rp = rt / beta
    a_p = (GRAVITATIONAL_CONSTANT * mass_bh_cgs
           * (tpeak * DAY_TO_S / jnp.pi) ** 2) ** (1. / 3.)
    a_t = (GRAVITATIONAL_CONSTANT * mass_bh_cgs
           * (time * DAY_TO_S / jnp.pi) ** 2) ** (1. / 3.)

    r_isco = 6 * GRAVITATIONAL_CONSTANT * mass_bh_cgs / SPEED_OF_LIGHT ** 2
    r_photo_max = rp + 2 * a_t
    l_edd = (4 * jnp.pi * GRAVITATIONAL_CONSTANT * mass_bh_cgs
             * SPEED_OF_LIGHT / kappa_t)

    l_safe = jnp.maximum(luminosity, _L_FLOOR)
    rphot = rph_0 * a_p * (l_safe / l_edd) ** lphoto
    r_photosphere = (rphot * r_photo_max) / (rphot + r_photo_max) + r_isco
    temperature = (l_safe / (r_photosphere ** 2 * STEF_CONSTANT)) ** 0.25
    return temperature, r_photosphere, rp


def cocoon_photosphere(time, luminosity, t_thin, vej, nn):
    """redback ``photosphere.CocoonPhotosphere``. time days, luminosity erg/s, vej km/s.

    ORIGINAL (photosphere.py:42-49):
        set_vphoto    = vej * (time / t_thin) ** (-2./(nn + 3))
        r_photosphere = RADIUS_CONSTANT * set_vphoto * time
        photosphere_temperature = (luminosity / (STEF_CONSTANT * r_photosphere**2))**0.25

    CHANGE 4 only: `time` is floored where it enters a NEGATIVE power (redback returns inf
    at time = 0 and then discards it), and the radius and luminosity are floored before the
    quarter power. Not used by any TDE model in redback; ported because it is part of the
    same file and costs three lines.

    Returns ``(temperature K, r_photosphere cm)``.
    """
    t_safe = jnp.maximum(time, _L_FLOOR)
    v_photo = vej * (t_safe / t_thin) ** (-2. / (nn + 3))
    r_photosphere = RADIUS_CONSTANT * v_photo * time
    r_safe = jnp.maximum(jnp.abs(r_photosphere), _L_FLOOR)
    temperature = (jnp.maximum(luminosity, _L_FLOOR)
                   / (STEF_CONSTANT * r_safe ** 2)) ** 0.25
    return temperature, r_photosphere


# --- interaction_processes.py --------------------------------------------------------------
#
# Each of these convolves a dense luminosity array with a diffusion / viscous kernel.
# redback builds, for every unique observation time te, a set of quadrature nodes
# int_times = tb + (te - tb)*xm and trapezoids over them.
#
# WHAT NEEDS NO WORK
#   Diffusion and Viscous write their exponent as exp((t'^2 - te^2)/tau^2) and
#   exp((t' - te)/tvisc) with t' <= te, so it is never positive: no overflow, no rewrite.
#   (redback got this right here; contrast CSMDiffusion below, which did not.)
#
# THE ONE STRUCTURAL CHANGE: uniq_times moves to SETUP TIME.
#   np.unique(time[(time >= tb) & (time <= dense_times[-1])]) has a shape that depends on
#   the data, which jit forbids. The observation times are fixed for a dataset, so
#   uniq_times and the searchsorted gather indices are static: build them once with
#   build_interaction_grid() and reuse. Same pattern as the kilonova's filter set.
#
# WHY xm IS FINE EVEN THOUGH IT DEPENDS ON FITTED PARAMETERS
#   lsp = logspace(log10(tau/dense_times[-1]) - 3, 0, num) has parameter-dependent VALUES
#   but a fixed LENGTH (num = timesteps//2), so concat(lsp, 1-lsp) is always 2*num long.
#   np.unique on it is a sort plus a dedupe that fires only if some lsp value exactly
#   equals some 1-lsp value -- a measure-zero coincidence. jnp.sort reproduces it, shapes
#   stay static, and the nodes still move with the parameters as redback intends.
#
# NaN HANDLING
#   redback's `int_args[np.isnan(int_args)] = 0.` is defensive against NaN in the input
#   luminosity. `jnp.where` reproduces the forward value, but a NaN in the luminosity will
#   still poison the gradient upstream of the where -- sanitise the luminosity at its
#   source instead of relying on this.


def build_interaction_grid(time, dense_times):
    """Setup-time (NumPy) companion to :func:`diffusion` / :func:`viscous`.

    Reproduces redback's
        tb         = max(0.0, min(dense_times))
        uniq_times = np.unique(time[(time >= tb) & (time <= dense_times[-1])])
        new_lums   = uniq_lums[np.searchsorted(uniq_times, time)]

    Returns ``(uniq_times, gather_idx, tb)``. Both arrays have shapes fixed by the
    dataset, so this runs once, outside jit.

    DEVIATION: ``gather_idx`` is clipped into range. redback does not clip, so an
    observation past ``uniq_times[-1]`` raises IndexError; here it repeats the last
    entry. Times outside ``[tb, dense_times[-1]]`` are outside the model's domain either
    way -- the clip turns a crash into a saturated value, and a warning says so.
    """
    time = np.asarray(time, dtype=np.float64)
    dense_times = np.asarray(dense_times, dtype=np.float64)
    tb = max(0.0, dense_times.min())
    sel = (time >= tb) & (time <= dense_times[-1])
    if not sel.all():
        warnings.warn(
            f"{(~sel).sum()} of {time.size} observation times fall outside the dense grid "
            f"[{tb:.6g}, {dense_times[-1]:.6g}] and will be clamped to its edge. redback "
            f"would raise IndexError here.", RuntimeWarning, stacklevel=2)
    uniq_times = np.unique(time[sel])
    if uniq_times.size == 0:
        raise ValueError("no observation time lies inside the dense grid; nothing to convolve")
    gather_idx = np.clip(np.searchsorted(uniq_times, time), 0, uniq_times.size - 1)
    return jnp.asarray(uniq_times), jnp.asarray(gather_idx), float(tb)


def _quadrature_nodes(scale, t_max, tb, uniq_times, timesteps):
    """redback's xm / int_times construction. ``timesteps`` is static (it sets a shape)."""
    num = int(round(timesteps / 2.0))
    lsp = jnp.logspace(jnp.log10(scale / t_max) - 3.0, 0.0, num)   # minimum_log_spacing = -3
    xm = jnp.sort(jnp.concatenate([lsp, 1.0 - lsp]))               # np.unique -> sort, see above
    return jnp.clip(tb + (uniq_times[:, None] - tb) * xm[None, :], tb, t_max)


def _interp_geometric(x, xp, fp):
    """``jnp.interp(x, xp, fp)`` for a GEOMETRIC ``xp``, the segment found by arithmetic.

    ``jnp.interp`` searches for each point's segment; on a geometric grid the segment is
    ``floor(log(x / xp[0]) / log step)``. The interpolation itself uses the real nodes, so the
    value is the same piecewise-linear interpolant (a point on a node boundary may take the
    neighbouring segment, which gives the same value). Measured on the supernova observed-epoch
    path: identical output, the diffusion 2.2x faster on a GPU (14.2 -> 6.4 us per evaluation).
    """
    n = xp.shape[0]
    xc = jnp.clip(x, xp[0], xp[-1])
    ln0 = jnp.log(xp[0])
    s = (jnp.log(xc) - ln0) / ((jnp.log(xp[-1]) - ln0) / (n - 1))
    i = jnp.clip(jnp.floor(s).astype(jnp.int32), 0, n - 2)
    x0, x1 = xp[i], xp[i + 1]
    return fp[i] + (xc - x0) / (x1 - x0) * (fp[i + 1] - fp[i])


@partial(jax.jit, static_argnames=("timesteps", "geometric"))
def diffusion(uniq_times, gather_idx, tb, dense_times, luminosity,
              kappa, kappa_gamma, mej, vej, timesteps=100, geometric=False):
    """redback ``interaction_processes.Diffusion`` (Arnett 1982).

    ORIGINAL (interaction_processes.py:33-65):
        tau_diff   = sqrt(diffusion_constant*kappa*mej/vej)/day_to_s
        trap_coeff = (trapping_constant*kappa_gamma*mej/vej**2)/day_to_s**2
        int_te2s   = int_times[:, -1]**2
        int_args   = int_lums*int_times*exp((int_times**2 - int_te2s[:,None])/tau_diff**2)
        uniq_lums  = trapz(int_args, int_times, axis=1)
        uniq_lums *= -2.0*expm1(-trap_coeff/int_te2s)/tau_diff**2

    CHANGES: interp1d -> jnp.interp (both piecewise linear, identical); np.unique -> sort;
    uniq_times/gather_idx precomputed. The arithmetic is untouched. ``geometric=True`` (static)
    declares ``dense_times`` geometric and finds each node's segment by arithmetic
    (:func:`_interp_geometric`): same values, fewer operations; the supernova observed-epoch path
    uses it.

    Times in days, mej in Msun, vej in km/s.
    """
    diffusion_constant = 2.0 * SOLAR_MASS / (13.7 * SPEED_OF_LIGHT * KM_CGS)
    trapping_constant = 3.0 * SOLAR_MASS / (4 * jnp.pi * KM_CGS ** 2)

    tau_diff = jnp.sqrt(diffusion_constant * kappa * mej / vej) / DAY_TO_S
    trap_coeff = (trapping_constant * kappa_gamma * mej / vej ** 2) / DAY_TO_S ** 2

    t_max = dense_times[-1]
    int_times = _quadrature_nodes(tau_diff, t_max, tb, uniq_times, timesteps)

    int_te2s = int_times[:, -1] ** 2
    int_lums = (_interp_geometric(int_times, dense_times, luminosity) if geometric
                else jnp.interp(int_times, dense_times, luminosity))
    # exponent <= 0 by construction (int_times <= int_times[:, -1]): nothing can overflow
    int_args = int_lums * int_times * jnp.exp(
        (int_times ** 2 - int_te2s[:, None]) / tau_diff ** 2)
    int_args = jnp.where(jnp.isnan(int_args), 0.0, int_args)

    uniq_lums = jnp.trapezoid(int_args, int_times, axis=1)
    uniq_lums = uniq_lums * -2.0 * jnp.expm1(-trap_coeff / int_te2s) / tau_diff ** 2
    return uniq_lums[gather_idx]


@partial(jax.jit, static_argnames=("timesteps",))
def aspherical_diffusion(uniq_times, gather_idx, tb, dense_times, luminosity,
                         kappa, kappa_gamma, mej, vej,
                         area_projection, area_reference, timesteps=100):
    """redback ``AsphericalDiffusion`` -- Diffusion plus the Darbha+2020 projected area.

    ORIGINAL extra line (interaction_processes.py:126):
        uniq_lums *= (1 + 1.4*(2 + uniq_times/tau_diff/0.59)/(1 + exp(uniq_times/tau_diff/0.59))
                      * (area_projection/area_reference - 1))

    CHANGE (exact identity): exp(x) with x = uniq_times/tau_diff/0.59 overflows float32
    past x ~ 88 and float64 past ~709. The whole term tends to 0 as x grows, so redback's
    +inf in the DENOMINATOR happens to give the right answer -- but only because the inf
    lands there. Rewriting

        1.4*(2 + x)/(1 + exp(x))  ==  1.4*(2 + x)*exp(-x)/(exp(-x) + 1)

    is the same function with no positive exponent, so nothing can overflow in either
    precision. exp(-x) underflows smoothly to 0, which is the correct limit.
    """
    diffusion_constant = 2.0 * SOLAR_MASS / (13.7 * SPEED_OF_LIGHT * KM_CGS)
    tau_diff = jnp.sqrt(diffusion_constant * kappa * mej / vej) / DAY_TO_S

    base = diffusion(uniq_times, jnp.arange(uniq_times.size), tb, dense_times,
                     luminosity, kappa, kappa_gamma, mej, vej, timesteps)

    x = uniq_times / tau_diff / 0.59
    emx = jnp.exp(-jnp.maximum(x, 0.0))
    shape = 1.4 * (2 + x) * emx / (emx + 1.0)
    uniq_lums = base * (1 + shape * (area_projection / area_reference - 1))
    return uniq_lums[gather_idx]


@partial(jax.jit, static_argnames=("timesteps",))
def viscous(uniq_times, gather_idx, tb, dense_times, luminosity, t_viscous,
            timesteps=1000):
    """redback ``interaction_processes.Viscous`` -- used by ``tde_fallback``.

    ORIGINAL (interaction_processes.py:214-238):
        int_tes   = int_times[:, -1]
        int_args  = int_lums*exp((int_times - int_tes[:,None])/tvisc)
        uniq_lums = trapz(int_args, int_times, axis=1)/tvisc

    Same three changes as :func:`diffusion`; the exponent is again <= 0 by construction.
    Times in days.
    """
    t_max = dense_times[-1]
    int_times = _quadrature_nodes(t_viscous, t_max, tb, uniq_times, timesteps)

    int_tes = int_times[:, -1]
    int_lums = jnp.interp(int_times, dense_times, luminosity)
    int_args = int_lums * jnp.exp((int_times - int_tes[:, None]) / t_viscous)
    int_args = jnp.where(jnp.isnan(int_args), 0.0, int_args)

    uniq_lums = jnp.trapezoid(int_args, int_times, axis=1) / t_viscous
    return uniq_lums[gather_idx]


@partial(jax.jit, static_argnames=("timesteps",))
def csm_diffusion(uniq_times, gather_idx, tb, dense_times, luminosity,
                  kappa, r_photosphere, mass_csm_threshold, timesteps=3000):
    """redback ``interaction_processes.CSMDiffusion`` (Chatzopoulos+2013).

    ORIGINAL (interaction_processes.py:159-194):
        beta = 4*pi**3/9
        t0   = kappa*mass_csm_threshold/(beta*c*r_photosphere)/day_to_s
        int_times = tb + (uniq_times[:,None] - tb)*xm        # NOTE: no clip here
        int_args  = int_lums*np.exp(int_times/t0)
        uniq_lums = trapz(int_args, int_times, axis=1)*exp(-int_tes/t0)/t0

    CHANGE (exact identity, and it is NOT cosmetic). redback's exponent here is POSITIVE:
    exp(int_times/t0) overflows float64 once int_times/t0 > 709, and t0 is a fitted
    quantity, so the overflow is reachable from an ordinary prior. It then meets
    exp(-int_tes/t0) = 0 and gives inf*0 = NaN. Folding the two exponentials into one,

        exp(t'/t0) * exp(-te/t0)  ==  exp((t' - te)/t0),   t' <= te,

    makes the exponent non-positive everywhere and removes the overflow entirely. This is
    the same rearrangement redback already applies in Diffusion and Viscous; only CSM was
    left in the unstable form.

    NOT used by any TDE model (it belongs to the CSM supernovae) -- ported for
    completeness because it shares this file. Note that a *different*, cumulative
    formulation of CSM diffusion exists in newer redback and in MOSFIT; this is the
    quadrature form that redback 1.12.0 actually ships.
    """
    beta = 4. * jnp.pi ** 3. / 9.
    t0 = kappa * mass_csm_threshold / (beta * SPEED_OF_LIGHT * r_photosphere) / DAY_TO_S

    t_max = dense_times[-1]
    int_times = _quadrature_nodes(t0, t_max, tb, uniq_times, timesteps)
    int_tes = int_times[:, -1]

    int_lums = jnp.interp(int_times, dense_times, luminosity)
    int_args = int_lums * jnp.exp((int_times - int_tes[:, None]) / t0)   # <= 0, always
    int_args = jnp.where(jnp.isnan(int_args), 0.0, int_args)

    uniq_lums = jnp.trapezoid(int_args, int_times, axis=1) / t0
    return uniq_lums[gather_idx]


# --- photometry ----------------------------------------------------------------------------

def _flux_nu_tde(temperature, r_photosphere, dl_cm, nu_source, redshift,
                 pref_const=PLANCK_T3, dilation=True):
    """Blackbody F_nu in erg/s/Hz/cm^2, observer frame. redback/sed.py:198.

    Identical to ``kilonova._flux_nu`` except for the redshift factor -- see CHANGE 7.
    redback's TDE path applies the k-correction to the FREQUENCY and the TIME only, and
    its ``blackbody_to_flux_density`` carries no ``(1 + z)``; its kilonova path does. Pass
    ``dilation=True`` for the kilonova convention.

    Every large quantity is pre-scaled into a precomputed literal, and the frequency scale
    is pinned with ``optimization_barrier``, for the reasons documented as identities 4
    and 8 in :mod:`whisper_cbpf.models.jax.kilonova`. ``pref_const`` selects the output unit:
    ``PLANCK_T3`` gives cgs, ``PLANCK_T3_AB`` the same spectrum in AB-zero-point units.
    """
    ratio = RATIO_CONST * ((r_photosphere / RSCALE) / (dl_cm / DLSCALE)) ** 2
    nu14 = jax.lax.optimization_barrier(nu_source / NUSCALE)
    arg = PLANCK_ARG * nu14 / temperature
    pref = pref_const * arg ** 3 * temperature ** 3
    out = pref * ratio * _inverse_expm1(arg)
    return out * (1.0 + redshift) if dilation else out


def flux_density_mjy(temperature, r_photosphere, nu_obs_hz, redshift, dl_cm, *,
                     dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                     r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """Flux density in mJy from a photosphere ``(T, R)`` at observer frequencies.

    ``temperature`` and ``r_photosphere`` are evaluated at the SOURCE-frame epochs matching
    ``nu_obs_hz``; the k-correction ``nu_src = nu_obs (1 + z)`` is applied here. Extinction
    arguments behave exactly as in :func:`ab_magnitude`.
    """
    nu_src = nu_obs_hz * (1.0 + redshift)
    fd = _flux_nu_tde(temperature, r_photosphere, dl_cm, nu_src, redshift,
                      dilation=dilation) / MJY
    lam_obs = 2.99792458e18 / nu_obs_hz                 # c in Angstrom/s, one literal
    tau = _ext_transmission(lam_obs, redshift, ebv_mw, ebv_host,
                            xi_mw, xi_host, r_v_mw, r_v_host, law)
    return fd if tau is None else fd * tau


def ab_magnitude(temperature, r_photosphere, band_idx, weights, norms, lam_obs_ang,
                 redshift, dl_cm, *, mag_floor=40.0, dilation=True,
                 ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                 r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """AB magnitude per observation, from a photosphere ``(T, R)`` and a filter set.

    ``weights, norms`` come from ``kilonova.ab_weights``; ``lam_obs_ang`` is its grid.
    ``temperature`` and ``r_photosphere`` are ``(n_obs,)``, already interpolated onto the
    source-frame epochs of the observations.

    This integrates the Planck spectrum against the real bandpass, which is what
    :mod:`whisper_cbpf.models.jax.kilonova` does. redback instead evaluates the spectrum on a
    100-point wavelength grid, builds an sncosmo ``TimeSeriesSource`` and splines it --
    see "DELIBERATE DEVIATION" in the kilonova docstring; the same applies here.

    ``mag_floor`` caps the result at ``min(mag, mag_floor)`` by clamping the flux RATIO,
    never the magnitude: ``log10(0)`` is ``-inf`` with an infinite derivative, so clamping
    after the log would return a finite value with a NaN gradient.
    """
    nu_obs = SPEED_OF_LIGHT / (lam_obs_ang * 1e-8)
    nu_src = nu_obs[None, :] * (1.0 + redshift)
    f_ab = _flux_nu_tde(temperature[:, None], r_photosphere[:, None], dl_cm, nu_src,
                        redshift, PLANCK_T3_AB, dilation=dilation)
    tau = _ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host,
                            xi_mw, xi_host, r_v_mw, r_v_host, law)
    w = weights if tau is None else weights * tau[None, :]
    num = jnp.sum(f_ab * w[band_idx], axis=1)
    den = jax.lax.optimization_barrier(INV_AB_ZEROPOINT * norms[band_idx])
    ratio_floor = jnp.asarray(10.0, dtype=num.dtype) ** (-0.4 * mag_floor)
    return -2.5 * jnp.log10(jnp.maximum(num / den, ratio_floor))


def _interp_photosphere(out, t_src_s):
    """(T, R) at source-frame seconds SINCE FALLBACK, from a cooling-envelope solution.

    redback does ``interp1d(output.time_since_fb, output.photosphere_temperature)`` on the
    already-sliced arrays. Here the arrays are full length, so the dead tail is pinned to
    the last valid sample before interpolating: the index array is clamped at
    ``constraint - 1``, which makes the time axis non-decreasing (a requirement of
    ``jnp.interp``) and makes any query past termination return the last valid value.

    DEVIATION: redback's interp1d has no fill_value here, so a query past the last valid
    epoch raises ValueError; ``gaussianrise_cooling_envelope`` uses
    ``fill_value='extrapolate'`` and linearly extrapolates a terminated model instead.
    Saturating is the honest option of the three, and it keeps the gradient finite.

    OUTSIDE THE MODEL'S OWN TIME SPAN THERE IS NO MODEL, so the flux is exactly zero rather
    than the saturated edge value. Guarding only ``constraint == 0`` is not enough, and the
    gap was measured: at ``constraint == 1`` the clamped index collapses the time axis to a
    single point and ``jnp.interp`` returns ``(T[0], R[0])`` at EVERY epoch -- a flat,
    finite, differentiable 400-day light curve at magnitude 20.5, with a perfectly ordinary
    likelihood, for a parameter set whose envelope died 0.02 days after fallback. That is
    **9.85% of redback's shipped prior at n_time=500** (0.45% at 5000), and neither redback
    reproduces it: 1.12.0 raises, 1.15.1 floors the slice at 4 points.
    A short-lived envelope is now short-lived: zero flux before its first epoch and after
    its last, which a sampler rejects instead of fitting.

    AND IT HAS A GRADIENT CONSEQUENCE, not only a forward one. Where `constraint < 2` the
    flux is exactly zero at every epoch, so the gradient is exactly zero in ALL FIVE
    parameters -- a flat region a gradient sampler can neither escape nor detect the edge of.
    Measured over redback's own prior: **6.45% of draws at n_time=500**, and empty at
    n_time=5000 in 512 draws. It reached the factory too -- 3.1% of
    `tde_model(rise="gaussian", n_time=500)`'s prior -- until the rise took its normaliser from
    the envelope's first sample (:func:`_rise_normaliser`); there only the envelope after the
    stitch is flat now. :func:`envelope_exists` bounds only the closed-form 0.431% of it; the
    rest needs the integration to detect.

    THE COST, STATED: that is a genuine discontinuity in time AND in the parameters, and the
    gradient does not know about it. Measured at (1, 1, 0.05, 0.1, 1.0), z = 0.05, envelope
    ending at 3709.86 d, the AB magnitude runs 19.76 / 20.03 / 20.07 at t/t_end =
    0.999/0.9999/1.0 and then 40.0 -- a 20-mag step -- while ``jax.grad`` returns a perfectly
    finite gradient across it, because it sees only the smooth side. The cliff MOVES with the
    parameters, so a likelihood that straddles it is discontinuous in a way HMC cannot see.
    Every alternative is worse (saturating invents a 400-day plateau on 9.85% of the prior;
    extrapolating invents a light curve outright; raising is impossible under jit), and a
    zero flux against a real detection gives a huge chi^2, which is the rejection one wants.
    But keep observations inside the envelope's own span, and treat a fit that pushes the
    termination time through the data as unconverged rather than as a solution.

    ``R_phot < 0`` is clamped to zero for the same reason. It happens on 3.6% of draws at
    ``n_time=5000``, at ISOLATED INTERIOR indices rather than in a tail, where the envelope
    goes optically thin (``Lamb < 1/e``, so ``Rph = Rv(1 + ln Lamb) < 0``). redback squares
    that into a blackbody and reports a positive flux from a negative radius; interpolating
    through it moves the AB magnitude by up to 3.89 mag locally. Clamping is continuous --
    ``Rph -> 0`` smoothly as ``Lamb -> 1/e`` -- and says the physical thing: no photosphere,
    no thermal emission. It is applied BEFORE the interpolation, so the spike never enters
    the interpolant.
    """
    n = out["time_since_fb"].shape[0]
    kmax = jnp.maximum(out["constraint"] - 1, 0)
    idx = jnp.minimum(jnp.arange(n), kmax)
    t_grid = out["time_since_fb"][idx]
    temp = jnp.interp(t_src_s, t_grid, out["photosphere_temperature"][idx])
    rad = jnp.interp(t_src_s, t_grid, jnp.maximum(out["photosphere_radius"][idx], 0.0))
    inside = ((t_src_s >= t_grid[0]) & (t_src_s <= t_grid[-1])
              & (out["constraint"] >= 2)).astype(rad.dtype)
    return temp, rad * inside


def cooling_envelope_flux_density(t_obs_days, nu_obs_hz, redshift, dl_cm,
                                  mbh_6, stellar_mass, eta, alpha, beta, *,
                                  n_time=None, dilation=True, **kw):
    """redback ``cooling_envelope(..., output_format='flux_density')``, in mJy.

    ``t_obs_days`` is observer-frame days SINCE FALLBACK (redback interpolates against
    ``time_since_fb`` after converting with ``t_src = t_obs/(1+z)``).
    """
    out = cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, n_time=n_time, **kw)
    t_src = jnp.asarray(t_obs_days) * DAY_TO_S / (1.0 + redshift)
    temp, rad = _interp_photosphere(out, t_src)
    return flux_density_mjy(temp, rad, nu_obs_hz, redshift, dl_cm, dilation=dilation)


def cooling_envelope_ab_magnitude(t_obs_days, band_idx, weights, norms, lam_obs_ang,
                                  redshift, dl_cm, mbh_6, stellar_mass, eta, alpha, beta,
                                  *, n_time=None, mag_floor=40.0, dilation=True,
                                  ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                                  r_v_mw=3.1, r_v_host=3.1, law='f99', **kw):
    """redback ``cooling_envelope(..., output_format='magnitude')``, AB magnitudes."""
    out = cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, n_time=n_time, **kw)
    t_src = jnp.asarray(t_obs_days) * DAY_TO_S / (1.0 + redshift)
    temp, rad = _interp_photosphere(out, t_src)
    return ab_magnitude(temp, rad, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                        mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw,
                        ebv_host=ebv_host, xi_mw=xi_mw, xi_host=xi_host,
                        r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)


#
# gaussianrise_cooling_envelope
#
# ORIGINAL (tde_models.py:298-414): the cooling envelope is only valid AFTER circularisation,
# so the rise is a Gaussian normalised to meet it at the stitching point xi*tfb:
#
#     f1    = pm.gaussian_rise(stitching_point, a_1=1., peak_time, sigma_t)
#     norm  = model_flux_at_first_valid_epoch / f1                 # per band / frequency
#     pre   = pm.gaussian_rise(tt_pre_fb, a_1=norm, peak_time, sigma_t)
#     total = interp1d(concat(tt_pre_fb, tt_post_fb), concat(pre, post), 'extrapolate')
#
# CHANGES. redback samples the rise onto a grid, concatenates it with the envelope, and
# interpolates the join. All three steps are avoidable, and each of them costs something:
#
#  1. THE RISE IS EVALUATED ANALYTICALLY, NOT SAMPLED AND INTERPOLATED. The rise is a pure
#     Gaussian, so in MAGNITUDE it is exactly a parabola in t, and the normalisation is one
#     additive constant. Sampling it on `n_pre` LINEAR nodes and interpolating linearly in
#     FLUX -- redback's scheme -- has a measured error of 0.13 mag at sigma_t = 5 d and
#     0.0128 mag at sigma_t = 2 d on redback's own 100-node grid, and it grows in proportion
#     to tfb, which reaches 411.1 d at the mbh_6/m* corner. (An earlier version of this note
#     quoted 0.13 and 3.67 mag and said redback interpolates the rise in FLUX. Both were
#     wrong: redback's 100-node BAND branch interpolates in MAGNITUDE -- only its 200-node
#     flux_density branch is in flux -- and the measured error at (1,1), z=0.05, peak_time=20
#     is 0.0128 mag at sigma_t=2, 0.0021 at 5 and 5e-4 at 10, each matching
#     (dt^2/8)(2.5/ln10)/sigma^2 exactly. sigma_t = 2 and 5 are outside this module's own
#     LogUniform(10, 60) anyway.) Evaluating the closed form has no such error, no `n_pre`,
#     and no grid -- and it is what makes the overflow in CHANGE 2 removable.
#
#  2. THE NORMALISING RATIO IS NEVER FORMED. redback computes
#         norm = F_env(stitch) / exp(-(stitch - t_peak)^2 / (2 sigma^2))
#     whose denominator underflows to 0 whenever the stitch point sits far in the Gaussian's
#     tail -- which is common, because tfb = 58 mbh_6^0.5 m*^0.2 days can be hundreds of days
#     while peak_time may be 0.1 d. Measured on this module's own default prior: 0.23% of
#     draws at z = 0.5 returned NaN magnitudes AND NaN gradients (d/d(peak_time),
#     d/d(sigma_t), d/d(mbh_6), d/d(stellar_mass) all NaN), rising to 2.87% with 1/f1 > 1e100.
#     Taking -2.5 log10 of the ratio analytically leaves
#         mag(t) = mag_env(stitch) + (2.5/ln10) * [(t-tp)^2 - (stitch-tp)^2] / (2 sigma^2)
#     which is a difference of two O(1) squares and cannot overflow in either direction.
#
#  3. THERE IS NO CONCATENATED TIME AXIS, so it cannot be non-monotonic. redback's is:
#     `tt_pre` runs to xi*tfb_obs while `tt_post` starts at tfb_obs, so for xi > 1 the joined
#     axis DESCENDS by (xi-1)*tfb_obs. scipy's interp1d sorts, so redback survives it;
#     `jnp.interp` does not, and the previous version of this file returned ~21 mag of
#     garbage at xi = 2 (measured: -6.79 vs +14.18 mag). `xi` is an exposed argument.
#
# What remains of redback's scheme is exactly its physics: a Gaussian rise normalised to meet
# the envelope at xi*tfb, in the observer frame, switching to the envelope after it.
#
# ALSO NOTE redback's own two branches disagree with each other: its flux_density branch uses
# tt_post_fb = xi*(time_temp*(1+z)) and 200 pre-fb nodes, its band branch uses
# tt_post_fb = time_temp*(1+z) and 100. The xi factor on the flux_density branch scales the
# post-fallback TIMES, which cannot be intended (xi is the stitch point, not a dilation) and
# is absent from the band branch. The band branch's convention is used here for both.

_MAG_PER_EFOLD = 2.5 / np.log(10.0)          # 1.0857362048, folded so nothing forms exp()


def _gaussian_rise_magnitude(t_s, stitch_s, mag_at_stitch, peak_time_days, sigma_t_days):
    """AB magnitude of the rise, normalised to ``mag_at_stitch`` at ``stitch_s``. Exact.

    F(t) = A exp(-(t-tp)^2 / 2 sigma^2)  =>  mag(t) - mag(stitch) is exactly
    ``(2.5/ln10) * [(t-tp)^2 - (stitch-tp)^2] / (2 sigma^2)``, with A cancelling. See
    CHANGE 2 in the block above for why the cancellation must be done on paper.
    """
    tp = peak_time_days * DAY_TO_S
    sig = sigma_t_days * DAY_TO_S
    return mag_at_stitch + _MAG_PER_EFOLD * ((t_s - tp) ** 2 - (stitch_s - tp) ** 2) / (2.0 * sig ** 2)


def _envelope_TR_observer(out, t_obs_s, redshift):
    """(T, R_phot) at OBSERVER-frame seconds on the envelope's own grid origin.

    Interpolating the photosphere and then integrating the band -- rather than integrating
    the band on all ``n_time`` grid epochs and interpolating the result -- is redback's own
    convention in ``cooling_envelope`` (``temp_func``/``rad_func``), and it makes the
    spectrum ``(n_obs, n_wave)`` instead of ``(n_time, n_wave)``: 6.4 MB per light curve
    instead of 80.2 MB, which is what previously capped the batch at 128 and made the
    spectrum 52% of the photometry's runtime.
    """
    n = out["time_temp"].shape[0]
    idx = jnp.minimum(jnp.arange(n), jnp.maximum(out["constraint"] - 1, 0))
    tt_post = out["time_temp"][idx] * (1. + redshift)
    temp = jnp.interp(t_obs_s, tt_post, out["photosphere_temperature"][idx])
    rad = jnp.interp(t_obs_s, tt_post, jnp.maximum(out["photosphere_radius"][idx], 0.0))
    # zero flux outside the envelope's own span, and clamped Rph -- see _interp_photosphere
    inside = ((t_obs_s <= tt_post[-1]) & (out["constraint"] >= 2)).astype(rad.dtype)
    return temp, rad * inside


def _rise_normaliser(out, stitch_s, redshift, xi):
    """``(T, R_phot, exists)``: the envelope's photosphere at the stitch, the rise's normaliser.

    :func:`_envelope_TR_observer` at ``stitch_s``, except where the envelope's integration ends
    after its FIRST sample (``constraint == 1``) and the stitch is at or before fallback
    (``xi <= 1``). There the normaliser is that first sample: the closed-form initial envelope,
    which is redback's normaliser in every case (``photosphere_temperature[0]``) and involves
    no integration. Asking for two live samples, as the envelope itself must (see
    :func:`_interp_photosphere`), took the RISE away on those draws and left ``mag_floor`` at
    every epoch -- and on redback 1.20's 500-point grid the first forward-Euler step overshoots
    ``Ee`` through zero on 4.8% of ``gaussianrise_cooling_envelope.prior`` (none at 5000,
    where the same draws live ~3100 steps). Measured against redback on prior draws: up to 47.7 mag
    from redback 1.20 on epochs that all lie on the rise. ``exists`` is False where there is
    nothing to normalise to: the envelope was never born, or the stitch lies past its end.
    """
    temp, rad = _envelope_TR_observer(out, stitch_s, redshift)
    first = (out["constraint"] == 1) & (jnp.asarray(xi) <= 1.0)
    return (jnp.where(first, out["photosphere_temperature"][0], temp),
            jnp.where(first, jnp.maximum(out["photosphere_radius"][0], 0.0), rad),
            (out["constraint"] >= 2) | first)


def gaussianrise_cooling_envelope_flux_density(
        t_obs_days, nu_obs_hz, redshift, dl_cm, peak_time, sigma_t,
        mbh_6, stellar_mass, eta, alpha, beta, *, xi=1.0, n_pre=200,
        n_time=None, dilation=True,
        ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
        r_v_mw=3.1, r_v_host=3.1, law='f99', **kw):
    """Gaussian rise stitched onto the cooling envelope, flux density in mJy.

    ``t_obs_days`` is observer-frame days from the ORIGIN of the light curve (0 = the start
    of the Gaussian rise), not from fallback -- this is redback's convention for this model.
    ``nu_obs_hz`` must be a scalar or broadcast against ``t_obs_days``.

    ``n_pre`` is accepted and ignored: the rise is evaluated in closed form (CHANGE 1 in the
    block above), so there is no grid to size.
    """
    out = cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, n_time=n_time, **kw)
    t_s = jnp.asarray(t_obs_days) * DAY_TO_S
    stitch_s = xi * out["tfb"] * (1. + redshift)
    ext = dict(ebv_mw=ebv_mw, ebv_host=ebv_host, xi_mw=xi_mw, xi_host=xi_host,
               r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)

    temp, rad = _envelope_TR_observer(out, t_s, redshift)
    f_env = flux_density_mjy(temp, rad, nu_obs_hz, redshift, dl_cm, dilation=dilation, **ext)

    temp_s, rad_s, _ = _rise_normaliser(out, stitch_s, redshift, xi)
    f_stitch = flux_density_mjy(jnp.broadcast_to(temp_s, jnp.shape(f_env)),
                                jnp.broadcast_to(rad_s, jnp.shape(f_env)),
                                nu_obs_hz, redshift, dl_cm, dilation=dilation, **ext)
    # The rise, in the log form so the normalising ratio is never formed -- and with the
    # exponent CLAMPED, which the magnitude branch does not need and this one does.
    #
    # Magnitude can represent an absurdly bright rise (-800 mag is a perfectly good float);
    # flux cannot. The exponent (stitch-tp)^2/(2 sigma^2) reaches 931 at the prior corner
    # (peak_time 0.1 d against tfb = 411 d), and exp(931) overflows float64 at 709.78.
    # Measured before this clamp: 0.28% of draws from the module's own default prior returned
    # +inf at z = 0.5, with NaN/-inf gradients in d/d(peak_time) and d/d(sigma_t) -- i.e. the
    # flux branch reintroduced, at 0.31%, exactly the failure the magnitude branch removed at
    # 0.27%. Saturating at exp(700) ~ 1e304 keeps it finite and monotone; no observation can
    # live there, and the magnitude branch reports the same source as a number.
    tp, sig = peak_time * DAY_TO_S, sigma_t * DAY_TO_S
    expo = ((stitch_s - tp) ** 2 - (t_s - tp) ** 2) / (2.0 * sig ** 2)
    # f_stitch is exactly 0 when there is no normaliser (the envelope was never born, or the
    # stitch sits past its last epoch -- see _rise_normaliser); 0 * exp(huge) is 0 * inf = NaN,
    # so select rather than multiply.
    f_rise = jnp.where(f_stitch > 0.0,
                       f_stitch * jnp.exp(jnp.minimum(expo, 700.0)), 0.0)
    return jnp.where(t_s < stitch_s, f_rise, f_env)


def gaussianrise_cooling_envelope_ab_magnitude(
        t_obs_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
        peak_time, sigma_t, mbh_6, stellar_mass, eta, alpha, beta, *,
        xi=1.0, n_pre=100, n_time=None, mag_floor=40.0, dilation=True,
        ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
        r_v_mw=3.1, r_v_host=3.1, law='f99', **kw):
    """Gaussian rise stitched onto the cooling envelope, AB magnitudes per observation.

    The rise is evaluated in closed form in MAGNITUDE, where a Gaussian is exactly a
    parabola and the normalisation is one additive constant -- so there is no grid, no
    interpolation error, no concatenated axis, and the normalising ratio that underflows in
    redback is never formed. See the three numbered changes in the block above.

    ``mag_floor`` still caps the faint end. There is deliberately no cap on the BRIGHT end:
    when ``peak_time`` is far below ``tfb`` the model genuinely predicts an absurdly bright
    rise -- measured brightest over 120k prior draws x 24 epochs: -804 mag at z = 0, -887 at
    z = 0.05, -1827 at z = 0.5 -- and that is the model being used outside its domain, not a
    numerical artefact. Capping it would hide the one signal that says so. (The flux
    entry point cannot represent those numbers and saturates instead; see its own comment.)

    ``n_pre`` is accepted and ignored (there is no grid to size).
    """
    out = cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, n_time=n_time, **kw)
    t_s = jnp.asarray(t_obs_days) * DAY_TO_S
    stitch_s = xi * out["tfb"] * (1. + redshift)
    ext = dict(mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
               xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)

    temp, rad = _envelope_TR_observer(out, t_s, redshift)
    mag_env = ab_magnitude(temp, rad, band_idx, weights, norms, lam_obs_ang,
                           redshift, dl_cm, **ext)

    # the envelope's magnitude AT the stitch point, in each observation's own band
    temp_s, rad_s, normalised = _rise_normaliser(out, stitch_s, redshift, xi)
    shape = jnp.shape(mag_env)
    mag_stitch = ab_magnitude(jnp.broadcast_to(temp_s, shape), jnp.broadcast_to(rad_s, shape),
                              band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, **ext)

    mag_rise = _gaussian_rise_magnitude(t_s, stitch_s, mag_stitch, peak_time, sigma_t)
    # An envelope that never lived has no rise to normalise against: `mag_stitch` is already
    # `mag_floor` there, but the parabola then climbs BRIGHTER than the floor wherever
    # peak_time < stitch (measured: down to 39.51, on 34.5% of (draw, epoch) pairs). The
    # whole light curve must be the floor, not just its stitch point. "Never lived" is
    # _rise_normaliser's: one born sample is enough for a stitch at or before fallback.
    dead = ~normalised
    out_mag = jnp.where(t_s < stitch_s, jnp.minimum(mag_rise, mag_floor), mag_env)
    return jnp.where(dead, jnp.asarray(mag_floor, out_mag.dtype), out_mag)


#
# physical validity -- conditions a prior can enforce, so the model is never ASKED for a
# light curve it cannot produce
#
#
# Some of this model's degenerate cases are not numerical at all: they are parameter
# combinations that describe no transient, and the honest place to exclude them is the prior,
# not a runtime special case downstream. Two of them are closed-form in the parameters and
# are given here as predicates. They change nothing about the model; they are for building a
# prior, for rejecting a proposal, or for reporting why a draw produced nothing.


def envelope_exists(mbh_6, stellar_mass, beta, binding_energy_const=0.8):
    """True where the envelope is born OUTSIDE the circularisation radius, i.e. exists at all.

    The initial envelope radius and the circularisation radius are both fixed by the
    parameters, with no integration involved:

        Me0     = 0.1 Mstar                          (the t_0_init = 1 term vanishes)
        Rv0     = (2 Rt^2 / (5 bec Rstar)) (Me0/Mstar) = 0.04 Rt^2 / (bec Rstar)
        Rcirc/2 = Rt / beta

    so ``Rv0 < Rcirc/2`` reduces exactly to

        beta * (Rt/Rstar) < 25 * binding_energy_const        (= 20 at the default 0.8)

    with ``Rt/Rstar = (1e6 mbh_6 / Mstar)^(1/3)``. Below that the envelope starts already
    inside the radius at which the model declares it finished, so ``constraint == 0`` and
    there is no light curve. Verified against the integrator: **400/400 draws agree**, and it
    covers **0.431%** of redback's shipped prior (2e6 Monte Carlo draws).

    NOTE what this does NOT cover. A further ~14% of draws at ``n_time = 500`` give
    ``0 < constraint < 6`` -- an envelope that exists but dies within a few steps. That is not
    closed-form; it depends on the integration, so it can only be checked by running the
    model and reading ``constraint`` (or ``termination_time``) back.
    """
    return beta * (mbh_6 * 1.0e6 / stellar_mass) ** (1. / 3.) >= 25.0 * binding_energy_const


def rise_peaks_near_fallback(peak_time, sigma_t, mbh_6, stellar_mass, *, n_sigma=3.0,
                             xi=1.0, binding_energy_const=0.8):
    """True where the Gaussian rise peaks within ``n_sigma`` of the stitch point.

    The rise is normalised to meet the envelope at ``xi * tfb``, so if the stitch sits far out
    in the Gaussian's tail the rise is brighter than the envelope by ``exp(n_sigma^2/2)``.
    That is not a numerical artefact -- it is the model being asked for a transient that peaks
    hundreds of days before its own fallback time. Measured:

        mbh_6 = 1,  M* = 1   (tfb =  58.0 d), peak = 20, sigma = 15  ->  2.5 sigma, x 2.5e1
        mbh_6 = 20, M* = 10  (tfb = 411.1 d), peak = 20, sigma = 15  -> 26.1 sigma, x 4.1e147
        mbh_6 = 20, M* = 10                 , peak = 0.1, sigma = 10 -> 41.1 sigma, x 1.0e304

    The module's own ``default_prior_gaussianrise()`` -- redback's -- allows all three,
    because ``peak_time`` and ``sigma_t`` are drawn independently of ``mbh_6`` and
    ``stellar_mass``, which together set ``tfb``. A prior that conditions the rise on the
    fallback time removes the entire regime, and with it the reason the flux entry point has
    to clamp its exponent at all.

    ONE-SIDED on purpose: a peak AFTER the stitch is fine (the rise is then monotonically
    increasing up to the stitch, which is what a rise should be). Only a stitch far PAST the
    peak is pathological.
    """
    tfb_days = calc_tfb(binding_energy_const, mbh_6, stellar_mass) / DAY_TO_S
    return (xi * tfb_days - peak_time) <= n_sigma * sigma_t


# --- parameters and priors -----------------------------------------------------------------
#: redback's positional order for `cooling_envelope`, minus time/redshift.
PARAMETERS = ["mbh_6", "stellar_mass", "eta", "alpha", "beta"]
#: ... and for `gaussianrise_cooling_envelope`.
PARAMETERS_GAUSSIANRISE = ["peak_time", "sigma_t"] + PARAMETERS


def redback_prior(model_name, *, drop=("redshift",)):
    """redback's OWN prior for ``model_name``, read from redback, not transcribed.

    Returns ``(Prior, pinned)``: the distributions whisper can represent, and a dict of
    parameters redback holds FIXED.

    READ FROM redback RATHER THAN COPIED, deliberately. A transcription drifts, and this one
    did: these files differ between the two installed redbacks, and the version transcribed
    here first had ``peak_time`` LogUniform(0.1, 60) and ``mbh_6`` LogUniform(0.01, 20) where
    1.15.1 has (1, 60) and (0.1, 20). Reading the file cannot be wrong about which redback is
    installed.

    ``redshift`` is dropped by default: it is a factory argument here, because ``dl_cm`` would
    otherwise have to move with it through a cosmology on every likelihood call.

    WHAT ``pinned`` IS. A scalar assignment in a redback prior file, which bilby returns as a
    ``DeltaFunction``. whisper's prior layer has no delta, so a fixed parameter is expressed the
    way :func:`kilonova_model` expresses a fixed ``temperature_floor``: bound at the factory
    and absent from ``parameters``. Up to redback 1.15, ``cooling_envelope.prior`` declared five
    distributions and then appended ``beta = 0.9``, ``eta = 0.1``, ``alpha = 0.1``,
    ``mbh_6 = 1``; bilby parses the file into a dict, so those four won, and redback's shipped
    cooling-envelope fit had **one** free parameter, ``stellar_mass`` (plus redshift). redback
    1.18 (commit a4ce717a, so also 1.20) replaced them with free priors -- ``mbh_6``
    LogUniform(0.1, 10), ``stellar_mass`` LogUniform(0.5, 10) -- and a comment that the
    ``eta``/``beta`` bounds need ``constraint=True``; whisper applies them in the JAX samplers
    through ``tde_model(constraint=...)``. ``pinned`` is then empty and all five are free.

    ``gaussianrise_cooling_envelope.prior`` has no delta functions in 1.15.1 or 1.20, so all
    seven of its parameters are free.

    Raises ImportError if redback is not installed; use :func:`fallback_prior` then.
    """
    from ..redback_adapter import _import_redback
    _import_redback()
    from redback.priors import get_priors

    from ...priors import LogUniform, Prior, Uniform

    rb = get_priors(model=model_name)
    dists, pinned = {}, {}
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
        else:
            raise NotImplementedError(
                f"redback's {model_name} prior has a {kind} on {k!r}, which whisper's prior "
                f"layer cannot represent. Pin it at the factory or add the distribution to "
                f"whisper_cbpf.priors.")
    return Prior(dists), pinned


#: Transcribed from the LATEST redback's prior files (1.20; ``LATEST_REDBACK_PRESET``, which
#: ``default_n_time`` follows too), for when redback is not importable. The shipped files are
#: the source of truth -- see :func:`redback_prior`, and prefer it. Up to whisper 0.1.0 this was
#: 1.15.1's, whose ``cooling_envelope`` had ``stellar_mass`` LogUniform(0.1, 10) free and pinned
#: ``mbh_6=1, eta=0.1, alpha=0.1, beta=0.9``; ``tde_model(rise="none", pin={...})`` rebuilds it.
_FALLBACK = {
    "cooling_envelope": (
        {"mbh_6": ("LogUniform", 0.1, 10.0), "stellar_mass": ("LogUniform", 0.5, 10.0),
         "eta": ("LogUniform", 1e-4, 0.1), "alpha": ("LogUniform", 0.1, 1.0),
         "beta": ("Uniform", 1.0, 5.0)},
        {},
    ),
    "gaussianrise_cooling_envelope": (
        {"peak_time": ("LogUniform", 1.0, 60.0), "sigma_t": ("LogUniform", 10.0, 60.0),
         "mbh_6": ("LogUniform", 0.1, 20.0), "stellar_mass": ("LogUniform", 0.1, 10.0),
         "eta": ("LogUniform", 1e-4, 0.1), "alpha": ("LogUniform", 0.1, 1.0),
         "beta": ("Uniform", 1.0, 5.0)},
        {},
    ),
}


def fallback_prior(model_name):
    """:func:`redback_prior` without redback: the latest release's (1.20) files, transcribed.
    Same return shape; see _FALLBACK."""
    from ...priors import LogUniform, Prior, Uniform

    dists, pinned = _FALLBACK[model_name]
    cls = {"Uniform": Uniform, "LogUniform": LogUniform}
    return Prior({k: cls[c](lo, hi) for k, (c, lo, hi) in dists.items()}), dict(pinned)


def default_prior(model_name="cooling_envelope"):
    """``(Prior, pinned)`` for a TDE model: redback's, read from redback where possible.

    NOT modified, narrowed or "improved" -- including where redback's own file leaves only one
    parameter free (1.15's ``cooling_envelope``; see :func:`redback_prior`). If a fit should vary
    more than that, widen the prior at the call site, where the choice is visible in the
    analysis rather than buried in a library default.
    """
    try:
        return redback_prior(model_name)
    except ImportError:
        return fallback_prior(model_name)


def default_prior_gaussianrise():
    """``(Prior, pinned)`` for ``gaussianrise_cooling_envelope``. All seven free (1.15.1, 1.20)."""
    return default_prior("gaussianrise_cooling_envelope")


# --- self-test -----------------------------------------------------------------------------
