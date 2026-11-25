"""MOSFiT's ``tde`` model: fall-back, viscous delay, MOSFiT photosphere and constraints, in JAX.

Adapted from MOSFiT (James Guillochon et al.), https://github.com/guillochon/MOSFiT, commit 0bc772e
(version 2.1.1), MIT licence, Copyright (c) 2016-2026 James Guillochon (its copyright and permission
notice follow this docstring and are in ``THIRD_PARTY_NOTICES.md``); modified for whisper_cbpf. The
fall-back engine and the viscous kernel are those of :mod:`whisper_cbpf.models.jax.tde_fallback`,
imported and not copied: its engine is MOSFiT's ``Fallback`` (adapted from MOSFiT's
``mosfit/modules/engines/fallback.py``, MIT; redback's ``_tde_mosfit_engine`` is the same code) and
its viscous kernel redback's (adapted from redback-jax and redback, GPL-3.0); the Guillochon &
Ramirez-Ruiz tables are the ones MOSFiT ships (byte-identical to the ones redback ships).

The model (Mockler, Guillochon & Ramirez-Ruiz 2019; MOSFiT, Guillochon et al. 2018). MOSFiT's
``tde.json`` chains ``Fallback`` (the returning mass ``dM/dt`` of a star of mass ``starmass`` on a
black hole of ``bhmass`` solar masses, impact parameter ``b``, from simulated polytropes of index
4/3 and 5/3; luminosity ``efficiency dM/dt c^2`` capped at ``Leddlim`` times the Eddington
luminosity by ``L Llim / (L + Llim)``), ``Viscous`` (exponential delay ``Tviscous``),
``TdePhotosphere`` (``R = Rph0 a_p (L / Llim)^lphoto + r_isco``, ``a_p`` the semi-major axis of the
debris that returns at the peak of ``dM/dt``; a black body of that radius and ``T = (L / 4 pi sigma
R^2)^(1/4)``), a pure ``Blackbody`` SED and ``TDEConstraints`` (a penalty added to the
log-likelihood). This module ports that chain.

**Not the same model as** :mod:`whisper_cbpf.models.jax.tde_fallback` **although the luminosity is
the same.** redback's ``tde_fallback`` normalises the photosphere by ``L_Edd`` (not by
``Leddlim L_Edd``) and bounds it with a soft maximum ``r_p + 2 a_t`` inside the forward model;
MOSFiT does neither, and puts ``r_isco`` outside. At the canonical point (1e6 solar masses, 1 solar
mass, ``b = 1``, ``efficiency = 0.1``) the redback photosphere is 4 times smaller and 2 times
hotter at the peak (measured in ``tests/test_tde_mosfit.py``). The light curve in magnitude differs
accordingly: this is the MOSFiT model, not redback's port of it.

**Time.** Source-frame days since the first fall-back, which is MOSFiT's ``texplosion`` (the model's
time zero). In :func:`tde_mosfit_model` the epochs are observer-frame, ``t_exp`` (or ``t_exp_days``)
is that zero, and an epoch at or before it is at ``mag_floor``.

**What is shared with** ``tde_fallback`` **and its measured size.** The fall-back rate, ``L(t)``,
``R_*``, ``t_peak`` and ``beta`` come from ``tde_fallback.fallback_engine`` and ``viscous_log10``.
The engine is MOSFiT's ``Fallback`` node for node (its luminosity before the viscous delay equals
MOSFiT's to a relative 1.64e-13 at worst); what differs is the viscous delay, redback's node quadrature on
the dense grid ``geomspace(1e-5, t_last + 100, 1000)`` instead of MOSFiT's exact piecewise-linear
solution on its ``DenseTimes`` grid. Against MOSFiT's own ``Fallback`` + ``Viscous`` (88 parameter
sets, 64 epochs each from 0.03 to 30 times the peak time, ``tests/goldens/tde_mosfit_oracle.npz``)
the luminosity differs by a median of 4.9e-5 dex, 99th percentile 2.2e-4 dex and at most 1.7e-3
dex (the largest at the earliest epochs, where the 1000-node grid is coarse for the rise); ``t_peak``
and ``beta`` agree to 3e-15 (relative). A MOSFiT run on its own default dense grid (100 log steps)
differs from its converged answer by a similar median and a larger tail (median 5e-5 dex, 99th
percentile 2.5e-3, maximum 3.9e-3). The photosphere, ``T`` and ``R`` then follow to a similar level
(``tests/test_tde_mosfit.py``).

**What is not ported.**

1. *Host extinction* (``losextinction``: ``nhhost``, ``rvhost``, ``ebv`` with the Morrison &
   McCammon / O'Donnell laws): not ported. The ``ebv_mw`` and ``ebv_host`` arguments are whisper's
   own extinction laws (the same as the other models), not MOSFiT's.
2. *The early-time extrapolation* of ``Fallback`` (a power law fitted to the first 10 to 15 % of the
   rising branch, used only when the first fall-back is later than the first dense epoch): not
   ported. Measured at the canonical point with the first fall-back 10 days after the first epoch,
   it changes the luminosity by at most 0.018 dex, only within 6 days of the first fall-back.
3. *MOSFiT's likelihood* (the ``variance`` parameter, the diagonal objective) and the soft
   ``tde_constraints`` score as a term of the sampler: the score is exposed
   (:func:`tde_mosfit_constraint_score`) and a hard wall is offered
   (:func:`tde_mosfit_constraint`), but nothing adds the score to a log-likelihood here.
4. *MOSFiT's Kroupa prior on the stellar mass*: whisper's prior layer has no such family. The
   default prior is log-uniform on the box of ``parameters.json`` (0.01 to 30); :func:`kroupa_icdf`
   and :func:`kroupa_log_pdf` are MOSFiT's Kroupa distribution for whoever wants to sample it.

**Precision.** float64 is required: ``L`` is 1e43 to 1e46 erg/s in cgs and the photometric functions
raise rather than return a light curve at ``mag_floor``::

    import jax; jax.config.update("jax_enable_x64", True)   # before the first array exists

References
----------
Mockler, B., Guillochon, J., Ramirez-Ruiz, E. 2019, ApJ 872, 151
(https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M)

Guillochon, J., Nicholl, M., Villar, V. A. et al. 2018, ApJS 236, 6 (MOSFiT)
(https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G)

Guillochon, J., Ramirez-Ruiz, E. 2013, ApJ 767, 25
(https://ui.adsabs.harvard.edu/abs/2013ApJ...767...25G)

Kroupa, P. 2001, MNRAS 322, 231 (the stellar-mass distribution of :func:`kroupa_icdf`)

Tout, C. A., Pols, O. R., Eggleton, P. P., Han, Z. 1996, MNRAS 281, 257 (the stellar radius)
"""
# MOSFiT's copyright and permission notice, which applies to the parts of this file adapted from
# MOSFiT:
#
# MIT License
#
# Copyright (c) 2016-2026 James Guillochon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ._adopted_tde import (GRID_START, _at_epochs, _guard_epochs, _to_rows, assemble,
                           geometric_dense_times, require_x64)
from .kilonova import DAY_TO_S, SOLAR_MASS, SPEED_OF_LIGHT
from .tde import (GRAVITATIONAL_CONSTANT, SOLAR_RADIUS, STEF_CONSTANT, ab_magnitude,
                  flux_density_mjy)
from .tde_fallback import _require_segments, fallback_engine, viscous_log10

__all__ = [
    "PARAMETERS", "default_prior", "mosfit_prior", "CONSTRAINT_BOUNDS", "WALL_EPOCHS",
    "tde_mosfit_curve", "tde_mosfit_photosphere", "tde_mosfit_constraint_terms",
    "tde_mosfit_constraint_score", "tde_mosfit_constraint", "tde_mosfit_flux_density_mjy",
    "tde_mosfit_ab_magnitude", "kroupa_icdf", "kroupa_log_pdf", "register_wall",
    "tde_mosfit_model", "register_tde_mosfit",
]

#: MOSFiT's parameters of the TDE model, in the order and units of ``tde.json`` /
#: ``parameters.json``: ``bhmass`` and ``starmass`` in solar masses, ``b`` the scaled impact
#: parameter (0 minimal, 1 full disruption, 2 the deepest simulated), ``efficiency`` the radiative
#: efficiency, ``Tviscous`` in days, ``Rph0`` the photosphere normalisation in units of ``a_p``,
#: ``lphoto`` its exponent, ``Leddlim`` the multiple of the Eddington luminosity that caps ``L``
#: (fixed at 1 in ``tde.json``). ``texplosion`` is the model's time zero (``t_exp``).
PARAMETERS = ("bhmass", "starmass", "b", "efficiency", "Tviscous", "Rph0", "lphoto", "Leddlim")

_KAPPA_T = 0.2 * (1 + 0.74)                      # tde_photosphere.py, fallback.py
_L_FLOOR = 1e-30                                 # erg/s: stands for MOSFiT's exact zero


# --- the photosphere and the constraints ---------------------------------------------------------

def tde_mosfit_photosphere(luminosity, bhmass, starmass, rstar, tpeak, beta, Rph0, lphoto,
                           Leddlim):
    """MOSFiT's ``TdePhotosphere``: returns ``(T [K], R [cm], rp [cm])`` at the luminosities.

    ORIGINAL (photospheres/tde_photosphere.py:30-62, ``resttexplosion = 0``):
        rt = (Mh / Mstar)**(1. / 3.) * Rstar;  rp = rt / beta
        r_isco = 6 * G * Mh * M_sun / (c * c)
        a_p = (G * Mh * M_sun * ((tpeak - resttexplosion) * DAY / pi)**2)**(1. / 3.)
        Llim = Leddlim * Ledd
        rphot = Rph0 * a_p * (luminosities / Llim)**l
        rphot = rphot + rphotmin                        # "adding rphotmin on to rphot for soft min"
        Tphot = (luminosities / (rphot**2 * STEF_CONST))**0.25

    There is no soft maximum and no other normalisation: the radius is the power law plus
    ``r_isco``. (The comment in the original says "soft min"; the code is a plain sum.) The
    luminosity is floored at 1e-30 erg/s inside the power (rule b: MOSFiT's exact zero has an
    infinite derivative in ``0**lphoto``; at the floor the term is below 1e-70 of ``r_isco``).
    ``luminosity`` in erg/s, ``bhmass`` and ``starmass`` in solar masses, ``rstar`` in solar radii,
    ``tpeak`` in days since the first fall-back.
    """
    kappa_t = _KAPPA_T
    Mh = bhmass * SOLAR_MASS
    Rstar = rstar * SOLAR_RADIUS
    rt = (bhmass / starmass) ** (1. / 3.) * Rstar
    rp = rt / beta
    r_isco = 6 * GRAVITATIONAL_CONSTANT * Mh / (SPEED_OF_LIGHT * SPEED_OF_LIGHT)
    rphotmin = r_isco
    Ledd = 4.0 * math.pi * GRAVITATIONAL_CONSTANT * Mh * SPEED_OF_LIGHT / kappa_t
    Llim = Leddlim * Ledd
    a_p = (GRAVITATIONAL_CONSTANT * Mh * ((tpeak - 0.0) * DAY_TO_S / math.pi) ** 2) ** (1. / 3.)
    lum = jnp.maximum(luminosity, _L_FLOOR)
    rphot = Rph0 * a_p * (lum / Llim) ** lphoto
    rphot = rphot + rphotmin
    Tphot = (lum / (rphot ** 2 * STEF_CONSTANT)) ** 0.25
    return Tphot, rphot, rp


def tde_mosfit_constraint_terms(tau_days, radius_phot, bhmass, rp, rphotmaxwind=True,
                                vphotmaxwind=0.5):
    """The two terms of MOSFiT's ``TDEConstraints`` score: ``(Rs / rp, max(R / R_max))``.

    ORIGINAL (constraints/tde_constraints.py:29-71, ``resttexplosion = 0``):
        Rs = 2 * G * bhmass * M_sun / (c * c)
        score_modifier -= 10.0 * (Rs / rp) ** 3
        a_t = (G * bhmass * M_sun * ((times - resttexplosion) * DAY / pi)**2)**(1. / 3.)
        a_t[times < resttexplosion] = 0.0
        if rphotmaxwind:
            rphotmax = 2*rp + vphotmaxwind * c * (times - resttexplosion) * DAY + 2 * a_t
            rphotmax[rphotmax < 2*rp] = 2*rp
        else:
            rphotmax = rp + 2 * a_t
        score_modifier -= 10.0 * (np.max(radius_phot / rphotmax)) ** 2

    ``tau_days`` are the epochs in source-frame days since the first fall-back (MOSFiT's
    ``rest_times - resttexplosion``), ``radius_phot`` the photosphere there, ``rp`` the pericentre
    in cm. ``rphotmaxwind`` is a Python flag (it selects the branch at trace time). The cube root
    of the squared time is guarded at ``tau <= 0`` (rule b: ``a_t`` is exactly 0 there, as in the
    original, and its derivative is finite).
    """
    tau = jnp.asarray(tau_days)
    Mh = bhmass * SOLAR_MASS
    Rs = 2 * GRAVITATIONAL_CONSTANT * Mh / (SPEED_OF_LIGHT * SPEED_OF_LIGHT)
    pos = tau > 0.0
    tau_safe = jnp.where(pos, tau, 1.0)
    a_t = (GRAVITATIONAL_CONSTANT * Mh * (tau_safe * DAY_TO_S / math.pi) ** 2) ** (1. / 3.)
    a_t = jnp.where(pos, a_t, 0.0)
    if rphotmaxwind:
        rphotmax = 2 * rp + vphotmaxwind * SPEED_OF_LIGHT * tau * DAY_TO_S + 2 * a_t
        rphotmax = jnp.where(rphotmax < 2 * rp, 2 * rp, rphotmax)
    else:
        rphotmax = rp + 2 * a_t
    return Rs / rp, jnp.max(radius_phot / rphotmax)


def tde_mosfit_curve(tau_days, bhmass, starmass, b, efficiency, Tviscous, Rph0, lphoto,
                     Leddlim=1.0, *, dense_resolution=1000, geometric=True):
    """MOSFiT's ``Fallback`` -> ``Viscous`` -> ``TdePhotosphere`` at ``tau_days``. Traced; float64.

    ``tau_days`` are source-frame days since the first fall-back (the model's time zero); the
    parameters are :data:`PARAMETERS`. Returns ``(L, T, R, rstar, t_peak, beta, rp)``: the
    luminosity [erg/s] after the viscous delay, the photosphere temperature [K] and radius [cm] at
    each epoch, the stellar radius [Rsun], the time of the peak of ``dM/dt`` [d since the first
    fall-back], the physical penetration factor and the pericentre [cm]. An epoch at or before 0 is
    at the luminosity floor 1e-30 erg/s (MOSFiT: exactly 0). ``b`` is held in [0, 2]: where MOSFiT
    fails (see ``upstream_issues/10``) the table edge is used, consistently for the rate and for
    ``beta``. ``geometric`` (default True) goes to
    :func:`~whisper_cbpf.models.jax.tde_fallback.viscous_log10` with the geometric dense grid this
    function builds: True finds the viscous kernel's segments by arithmetic (rule (a), the same
    interpolant; ``dense_resolution`` at least 2), False by a search (``jnp.interp``), the code
    before the option existed. :func:`tde_mosfit_constraint_score` and
    :func:`tde_mosfit_flux_density_mjy` pass their own ``geometric`` (default True); the constraint
    wall (:func:`tde_mosfit_constraint`) calls it with ``geometric=False``: it keeps the search.
    """
    require_x64("tde_mosfit")
    tau = jnp.asarray(tau_days)
    alive = tau > 0.0
    t_eval = jnp.where(alive, jnp.maximum(tau, GRID_START), 1.0)     # finite when dead
    dense = geometric_dense_times(jnp.max(jnp.where(alive, tau, 0.0)), dense_resolution)
    log_l, rstar, t_peak, beta = fallback_engine(
        dense, bhmass / 1.0e6, starmass, jnp.clip(b, 0.0, 2.0), efficiency, Leddlim)
    lum = jnp.power(10.0, viscous_log10(t_eval, dense, log_l, Tviscous, geometric=geometric))
    lum = jnp.where(alive, lum, _L_FLOOR)
    temp, rad, rp = tde_mosfit_photosphere(lum, bhmass, starmass, rstar, t_peak, beta, Rph0,
                                           lphoto, Leddlim)
    return lum, temp, rad, rstar, t_peak, beta, rp


def tde_mosfit_constraint_score(tau_days, bhmass, starmass, b, efficiency, Tviscous, Rph0, lphoto,
                                Leddlim=1.0, *, rphotmaxwind=True, vphotmaxwind=0.5,
                                dense_resolution=1000, geometric=True):
    """MOSFiT's ``TDEConstraints`` ``score_modifier`` for a parameter set at the epochs ``tau_days``.

    ``-10 (Rs / rp)^3 - 10 max_t (R_phot / R_max)^2``: zero when the black hole is small and the
    photosphere stays inside the maximum, ``-10`` when either term reaches 1, and growing as the
    cube and the square beyond. MOSFiT adds it to the log-likelihood; it is a soft penalty, not a
    rejection. ``rphotmaxwind`` (a Python flag) and ``vphotmaxwind`` are ``tde.json``'s fixed
    ``1`` and ``0.5``; ``rphotmaxwind=False`` bounds the photosphere by the apocentre of the debris
    instead. The maximum runs over ``tau_days``, as MOSFiT's runs over the observation epochs.
    Traced; float64. ``geometric`` (default True) goes to :func:`tde_mosfit_curve`: the viscous
    kernel finds its segments by arithmetic; False finds them by a search, the function before
    the option existed, as the wall (:func:`tde_mosfit_constraint`) always does.
    """
    _, _, rad, _, _, _, rp = tde_mosfit_curve(
        tau_days, bhmass, starmass, b, efficiency, Tviscous, Rph0, lphoto, Leddlim,
        dense_resolution=dense_resolution, geometric=geometric)
    rs_over_rp, ratio = tde_mosfit_constraint_terms(tau_days, rad, bhmass, rp, rphotmaxwind,
                                                    vphotmaxwind)
    return -10.0 * rs_over_rp ** 3 - 10.0 * ratio ** 2


#: The hard wall on the two terms of the score, as ``Constraint(lo, hi)`` bounds in the form of
#: :data:`whisper_cbpf.models.constraints.MODELS`: the star is not swallowed (``Rs / rp < 1``,
#: the first term of the score reaching its -10) and the photosphere stays inside its maximum
#: (``R / R_max < 1``, the second term reaching -10). MOSFiT only penalises; the wall is whisper's.
CONSTRAINT_BOUNDS = {"rs_over_rp": (0, 1), "rphot_over_rmax": (0, 1)}
#: The wall's epochs, in units of the time of the peak of ``dM/dt``: 64 log-spaced from 0.03 to 30.
WALL_EPOCHS = np.geomspace(0.03, 30.0, 64)


def tde_mosfit_constraint(p, xp, e_burn):
    """The wall's conversion function, in the signature of :data:`whisper_cbpf.models.constraints.MODELS`.

    ``p`` holds the parameters of :data:`PARAMETERS` (floats or traced), ``xp`` and ``e_burn`` are
    unused (the evaluation is JAX, float64 required). Returns ``{"rs_over_rp": Rs / rp,
    "rphot_over_rmax": max over WALL_EPOCHS x t_peak of R_phot / R_max}`` with ``rphotmaxwind=1``,
    ``vphotmaxwind=0.5`` (``tde.json``). ``nan`` becomes ``inf``, which fails, as redback's
    conversion functions do. The viscous kernel finds its segments by a search
    (``geometric=False``) whatever the model's ``geometric``: the wall is the same function for
    every model, so ``unique_epochs=False, geometric=False`` with a wall is the code before these
    options existed.
    """
    args = tuple(jnp.asarray(p[k], dtype=jnp.float64) for k in PARAMETERS)
    _, _, _, _, t_peak, _, _ = tde_mosfit_curve(jnp.ones(1), *args, geometric=False)
    tau = t_peak * jnp.asarray(WALL_EPOCHS)
    _, _, rad, _, _, _, rp = tde_mosfit_curve(tau, *args, geometric=False)
    rs_over_rp, ratio = tde_mosfit_constraint_terms(tau, rad, args[0], rp)
    return {"rs_over_rp": jnp.nan_to_num(rs_over_rp, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf),
            "rphot_over_rmax": jnp.nan_to_num(ratio, nan=jnp.inf, posinf=jnp.inf,
                                              neginf=jnp.inf)}


def register_wall():
    """Make ``constraint=`` work for ``kind="tde_mosfit"``: add the wall to the constraints table.

    ``whisper_cbpf.models.constraints.MODELS`` is keyed by model name; this adds
    ``"tde_mosfit"`` if it is not there already (a no-op once the table lists it). Called by
    :func:`tde_mosfit_model` when a wall is requested.
    """
    from .. import constraints as C

    C.MODELS.setdefault("tde_mosfit", (tde_mosfit_constraint, CONSTRAINT_BOUNDS))


# --- photometry -----------------------------------------------------------------------------------

def tde_mosfit_flux_density_mjy(tau_days, nu_obs_hz, redshift, dl_cm, bhmass, starmass, b,
                                efficiency, Tviscous, Rph0, lphoto, Leddlim=1.0, *,
                                dense_resolution=1000, dilation=True, geometric=True):
    """Monochromatic flux density [mJy] of MOSFiT's pure black body at ``tau_days``. Traced; float64.

    ``F_nu = (1 + z) pi B_nu(T, nu (1 + z)) R^2 / d_L^2`` (MOSFiT's ``Blackbody`` module through the
    band path of ``Photometry``; ``dilation=False`` drops the leading ``1 + z``). ``tau_days`` are
    source-frame days since the first fall-back, ``nu_obs_hz`` observer-frame frequencies, one per
    epoch. Epochs at or before 0 give the floor luminosity, hence ~0 flux. ``geometric`` (default
    True) goes to :func:`tde_mosfit_curve`: the viscous kernel finds its segments by arithmetic;
    False finds them by a search, the function before the option existed.
    """
    require_x64("tde_mosfit")
    _, temp, rad, _, _, _, _ = tde_mosfit_curve(
        tau_days, bhmass, starmass, b, efficiency, Tviscous, Rph0, lphoto, Leddlim,
        dense_resolution=dense_resolution, geometric=geometric)
    return flux_density_mjy(temp, rad, nu_obs_hz, redshift, dl_cm, dilation=dilation)


def tde_mosfit_ab_magnitude(tau_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                            bhmass, starmass, b, efficiency, Tviscous, Rph0, lphoto,
                            Leddlim=1.0, *, epoch_idx=None, dense_resolution=1000,
                            geometric=True, mag_floor=40.0, dilation=True, ebv_mw=None,
                            ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1,
                            law="f99"):
    """AB magnitude of MOSFiT's ``tde`` per observation. Traced; **float64 only**.

    ``tau_days`` are SOURCE-frame days since the first fall-back (``(t - t_exp) / (1 + z)``); an
    epoch at or before 0 is at ``mag_floor``. ``weights, norms`` come from
    :func:`whisper_cbpf.models.jax.kilonova.ab_weights` and ``lam_obs_ang`` is its grid, as for
    :func:`whisper_cbpf.models.jax.tde.ab_magnitude`, whose ``mag_floor``, ``dilation`` and
    extinction arguments these are (whisper's laws, not MOSFiT's ``losextinction``).

    ``epoch_idx`` is ``None`` (every observation evaluated) or ``(first, inv)``, int32, from
    ``predict_jax.epoch_index(times[, size])``. With it, :func:`tde_mosfit_curve` (the engine, the
    viscous integral and the photosphere) runs at ``tau[first]`` and ``T``, ``R`` are copied back
    with ``inv`` before the band integral: the same numbers, to round-off (porting rule (a);
    MOSFiT's ``Viscous`` does the same). A map that does not describe ``tau`` makes every
    magnitude NaN. The derivative with respect to ``tau_days`` itself is then per distinct epoch,
    carried by its first observation.

    ``geometric`` (default True) finds the viscous kernel's segments by arithmetic on the
    geometric dense grid of :func:`tde_mosfit_curve` (rule (a), the same interpolant; see
    :func:`~whisper_cbpf.models.jax.tde_fallback.viscous_log10`); False finds them by a search
    (``jnp.interp``). ``epoch_idx=None, geometric=False`` is the code before these options existed.
    """
    require_x64("tde_mosfit")
    tau = jnp.asarray(tau_days)
    alive = tau > 0.0
    _, temp, rad, _, _, _, _ = tde_mosfit_curve(
        _at_epochs(tau, epoch_idx), bhmass, starmass, b, efficiency, Tviscous, Rph0, lphoto,
        Leddlim, dense_resolution=dense_resolution, geometric=geometric)
    # the band integral stays per observation: T and R copied to the rows of their epoch
    mag = ab_magnitude(_to_rows(temp, epoch_idx), _to_rows(rad, epoch_idx), band_idx, weights,
                       norms, lam_obs_ang, redshift, dl_cm,
                       mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
                       xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)
    mag = jnp.where(alive, mag, jnp.asarray(mag_floor, mag.dtype))
    return _guard_epochs(mag, tau, epoch_idx)


# --- priors ---------------------------------------------------------------------------------------

#: Transcribed from MOSFiT's ``models/tde/parameters.json`` (the box of each free parameter) and
#: ``tde.json`` (which fixes ``Leddlim = 1``). ``starmass`` is a Kroupa-distributed parameter there;
#: the closest family of the whisper prior layer is log-uniform on the same box.
_MOSFIT_PRIOR = (
    {"bhmass": ("LogUniform", 5e4, 5e8), "starmass": ("LogUniform", 0.01, 30.0),
     "b": ("Uniform", 0.0, 2.0), "efficiency": ("LogUniform", 1e-4, 0.4),
     "Tviscous": ("LogUniform", 1e-3, 1e2), "Rph0": ("LogUniform", 1e-4, 1e4),
     "lphoto": ("Uniform", 0.0, 2.0)},
    {"Leddlim": 1.0},
)


def mosfit_prior():
    """``(Prior, pinned)`` of MOSFiT's ``tde`` model, transcribed from ``parameters.json``.

    ``bhmass`` LogUniform(5e4, 5e8) Msun, ``starmass`` LogUniform(0.01, 30) Msun (MOSFiT: Kroupa),
    ``b`` Uniform(0, 2), ``efficiency`` LogUniform(1e-4, 0.4), ``Tviscous`` LogUniform(1e-3, 100) d,
    ``Rph0`` LogUniform(1e-4, 1e4), ``lphoto`` Uniform(0, 2); pinned ``Leddlim = 1``.
    ``texplosion`` (Uniform(-500, 0) in MOSFiT) and ``redshift`` are not here: they are the factory's
    ``t_exp_days`` / ``free=`` and the dataset's.
    """
    from ...priors import LogUniform, Prior, Uniform

    dists, pinned = _MOSFIT_PRIOR
    cls = {"Uniform": Uniform, "LogUniform": LogUniform}
    return Prior({k: cls[c](lo, hi) for k, (c, lo, hi) in dists.items()}), dict(pinned)


def default_prior():
    """:func:`mosfit_prior`, the ``(Prior, pinned)`` the factory reads (the name the adapter looks for)."""
    return mosfit_prior()


# --- MOSFiT's Kroupa distribution of the stellar mass --------------------------------------------

_K1, _K2 = 0.08, 0.5            # break masses of the Kroupa (2001) broken power law
_KA0 = _K1 ** 0.3               # density m^-0.3 below 0.08 (unnormalised, continuous)
_KA1 = _K1 ** 1.3               # density m^-1.3 from 0.08 to 0.5
_KA2 = _K1 ** 1.3 * 0.5         # density m^-2.3 above 0.5


def _kroupa_cum(m):
    """Unnormalised cumulative of the density, from 0, at masses ``m`` (NumPy)."""
    m = np.asarray(m, dtype=float)
    c1 = _KA0 * _K1 ** 0.7 / 0.7
    c2 = c1 + _KA1 * (_K2 ** -0.3 - _K1 ** -0.3) / -0.3
    below = _KA0 * m ** 0.7 / 0.7
    mid = c1 + _KA1 * (m ** -0.3 - _K1 ** -0.3) / -0.3
    above = c2 + _KA2 * (m ** -1.3 - _K2 ** -1.3) / -1.3
    return np.where(m < _K1, below, np.where(m < _K2, mid, above))


def kroupa_icdf(u, min_mass=0.01, max_mass=30.0):
    """Stellar mass [Msun] at cumulative probability ``u`` of MOSFiT's Kroupa prior (NumPy).

    MOSFiT's ``Kroupa.prior_icdf`` returns the mass as a fraction of ``[min_mass, max_mass]``;
    this returns the mass itself (``min_mass + fraction (max_mass - min_mass)``), by inverting the
    analytic cumulative of the three power laws (slopes -0.3, -1.3, -2.3, breaks 0.08 and 0.5 Msun)
    segment by segment: the same numbers as MOSFiT's three branches (rule a), valid for
    ``min_mass >= 0.01``. A host-side helper for building draws, not traced.
    """
    u = np.asarray(u, dtype=float)
    c_lo, c_hi = float(_kroupa_cum(min_mass)), float(_kroupa_cum(max_mass))
    c = c_lo + u * (c_hi - c_lo)
    c1 = float(_kroupa_cum(_K1))
    c2 = float(_kroupa_cum(_K2))
    m0 = (np.clip(c, 0.0, None) * 0.7 / _KA0) ** (1.0 / 0.7)
    m1 = ((c - c1) * -0.3 / _KA1 + _K1 ** -0.3) ** (1.0 / -0.3)
    m2 = ((c - c2) * -1.3 / _KA2 + _K2 ** -1.3) ** (1.0 / -1.3)
    return np.where(c < c1, m0, np.where(c < c2, m1, m2))


def kroupa_log_pdf(mass):
    """MOSFiT's ``Kroupa.lnprior_pdf`` at ``mass`` [Msun]: ``ln`` of the unnormalised density.

    ORIGINAL (parameters/kroupa.py, ``lnprior_pdf``):
        if value < 0.08: np.log((value / 0.08)**(-0.3))
        elif value < 0.5: np.log((value / 0.08)**(-1.3))
        else: np.log((0.5 / 0.08)**(-1.3) * (value / 0.5)**(-2.3))
    """
    value = np.asarray(mass, dtype=float)
    return np.where(value < 0.08, np.log((value / 0.08) ** (-0.3)),
                    np.where(value < 0.5, np.log((value / 0.08) ** (-1.3)),
                             np.log((0.5 / 0.08) ** (-1.3) * (value / 0.5) ** (-2.3))))


# --- the whisper model ----------------------------------------------------------------------------

def tde_mosfit_model(band_names, redshift=None, dl_cm=None, *, name="tde_mosfit_jax",
                     n_wave=None, dense_resolution=1000, pin=None, prior=None, mag_floor=40.0,
                     dilation=True, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1,
                     law="f99", filter_set=None, band_aliases=None, default_system=None,
                     t_exp_days=0.0, constraint=None, free=None, redshift_prior=None,
                     unique_epochs=True, geometric=True):
    """Bind dataset context to MOSFiT's ``tde`` model and return a whisper ``Model``.

    Photometric, so a factory rather than a registry entry: ``predict(parameters, times, bands)``
    carries no filter set, redshift or distance, and a guessed distance would rescale every fitted
    black-hole mass. See :func:`whisper_cbpf.models.jax.tde_model`, whose arguments these follow.

    Parameters
    ----------
    band_names : sequence of str
        Filter names or labels, in the order the light curve's ``bands`` column maps to (bare
        letters are LSST unless ``default_system=``).
    redshift, dl_cm : float
        Fixed for the dataset; leave both ``None`` when the redshift is fitted
        (``free=["redshift"]``).
    name : str
        The model's name (default ``"tde_mosfit_jax"``).
    dense_resolution : int
        Points of the engine grid (1000); at least 2 with ``geometric=True`` (``ValueError`` here
        otherwise).
    pin : dict or None
        Override which parameters are held fixed: ``{"b": 1.0}`` pins one, ``{"Leddlim": None}``
        frees the pinned ``Leddlim`` (and needs ``prior=`` for it).
    prior : whisper_cbpf.priors.Prior or None
        Distributions overriding MOSFiT's, per parameter. The default is MOSFiT's own box
        (:func:`mosfit_prior`; the stellar mass log-uniform, not Kroupa).
    mag_floor, dilation, ebv_mw, ebv_host, r_v_mw, r_v_host, law, filter_set, n_wave, band_aliases, default_system :
        As :func:`whisper_cbpf.models.jax.tde_model` (Gauss-16 band integral by default, the
        ``(1 + z)`` flux factor of MOSFiT's band path, optional whisper extinction; MOSFiT's own
        host extinction is not ported).
    t_exp_days : float
        Time of the first fall-back on the light curve's own clock (MOSFiT's ``texplosion``),
        subtracted on the host in float64. Default 0: ``times`` are days since it.
    constraint : {"corrected", "redback", None}
        ``None`` (default): no wall. ``"corrected"`` and ``"redback"`` (the same here: the names
        are the constraints module's) arm the wall of :func:`tde_mosfit_constraint` (the star is not
        swallowed, the photosphere stays inside its maximum) as a hard wall that rejects 45 % of
        MOSFiT's box prior; MOSFiT itself only subtracts :func:`tde_mosfit_constraint_score` from
        the log-likelihood.
    free : list of str or None
        ``"t_exp"`` and/or ``"redshift"`` fitted as parameters, appended in that order; see
        :func:`whisper_cbpf.models.jax.tde_model`. ``t_exp`` needs ``prior=Prior({"t_exp": ...})``.
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none.
    unique_epochs : bool
        Evaluate the engine, the viscous delay and the photosphere once per distinct epoch and copy
        them to the observations of that epoch (default True; exact to round-off, see
        PORTING_NOTES), as MOSFiT's ``Viscous`` does. False evaluates every observation, with
        ``geometric=False`` the code before these options existed: the same numbers to round-off,
        slower on multi-band light curves.
    geometric : bool
        Default True: the viscous kernel finds its segments on the geometric dense grid by
        arithmetic instead of a search (porting rule (a), the same interpolant; see
        :func:`whisper_cbpf.models.jax.tde_fallback.viscous_log10`). False keeps the search. The
        constraint wall always keeps the search. ``unique_epochs=False, geometric=False`` is the
        code before these options existed. ``geometric`` enters the model's state (and so its
        ``model_record`` identity) only when True: a ``geometric=False`` model keeps the identity
        it had before the option existed, and the default model is identified as geometric.

    Returns
    -------
    Model
        ``parameters`` ``["bhmass", "starmass", "b", "efficiency", "Tviscous", "Rph0", "lphoto"]``
        (plus the freed context values); ``Leddlim`` is pinned at 1 as in ``tde.json``.

    References
    ----------
    Mockler, Guillochon & Ramirez-Ruiz 2019, ApJ 872, 151; Guillochon et al. 2018, ApJS 236, 6
    (MOSFiT); Guillochon & Ramirez-Ruiz 2013, ApJ 767, 25; Tout et al. 1996, MNRAS 281, 257 (the
    stellar radius). Full list in the module docstring.

    Notes
    -----
    **float64 is required** (``jax.config.update("jax_enable_x64", True)`` before the first array
    exists); the guard raises at the first ``predict``, not here. This is MOSFiT's photosphere, not
    redback's (:func:`whisper_cbpf.models.jax.tde_fallback_model`): at the same parameters the
    radius and temperature differ by factors of 4 and 2 at peak. A light curve depends on the other
    epochs only through the end of the engine grid, as in redback (about 1e-3 mag).

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax.tde_mosfit import tde_mosfit_model
    >>> m = tde_mosfit_model(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.parameters
    ['bhmass', 'starmass', 'b', 'efficiency', 'Tviscous', 'Rph0', 'lphoto']
    """
    if geometric:
        _require_segments(dense_resolution, "tde_mosfit_model")
    if constraint is not None:
        register_wall()
    return assemble(
        "tde_mosfit", "tde_mosfit", band_names, redshift, dl_cm, name=name, n_wave=n_wave,
        pin=pin, prior=prior, mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw,
        ebv_host=ebv_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law, filter_set=filter_set,
        band_aliases=band_aliases, default_system=default_system, t_exp_days=t_exp_days, free=free,
        redshift_prior=redshift_prior, constraint=constraint,
        # `geometric` enters the state only when True (the default): a geometric=False model keeps
        # the state (and so the `model_record` identity) it had before the option existed, and
        # `DirectTDEPredict` reads the absent key as geometric=False, whatever the default here
        engine_kw=dict(dense_resolution=dense_resolution,
                       **({"geometric": True} if geometric else {})),
        summary="MOSFiT fall-back (Guillochon tables), viscous delay, MOSFiT photosphere",
        caller="tde_mosfit_model", unique_epochs=unique_epochs)


def register_tde_mosfit(band_names, redshift=None, dl_cm=None, *, name="tde_mosfit_jax",
                        **kwargs):
    """Build a MOSFiT ``tde`` ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`tde_mosfit_model`.
    name : str, default "tde_mosfit_jax"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model

    References
    ----------
    As :func:`tde_mosfit_model`.

    Notes
    -----
    The model needs float64, as :func:`tde_mosfit_model` says.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.jax.tde_mosfit import register_tde_mosfit
    >>> m = register_tde_mosfit(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.name in wp.list_models()
    True
    """
    from ._factories import _register

    return _register(tde_mosfit_model(band_names, redshift, dl_cm, name=name, **kwargs))
