"""redback's physical-validity constraints, for the CPU adapter and the JAX samplers.

redback keeps these outside its models and outside its ``.prior`` files: since 1.20,
``redback.priors._constraint_settings`` maps a model name to a bilby *conversion function* and the
``Constraint(lo, hi)`` bounds its outputs must fall in, and ``get_priors(model, constraint=True)``
attaches them. A draw that breaks one is not a light curve at all -- an Arnett supernova whose
ejecta carry more kinetic energy than burning its nickel released, a magnetar that spins down
more energy than it has, a cooling envelope with ``eta`` below ``eta_min``. Up to whisper 0.1.0
they were dropped, so every sampler explored those corners: 6-13% of GPU TDE draws broke the
``eta`` bound, and Arnett fits won through ``f_nickel -> 1`` and ``M_Ni`` up to 100 Msun.

whisper applies them as a **hard wall**: a draw that breaks one predicts zero flux in the CPU
adapter (:mod:`whisper_cbpf.models.redback_adapter`) and in the JAX supernova and TDE models' host
``predict`` (what the CPU samplers call), zero flux in the JAX samplers' batched forward map and
``-inf`` in their log-density (:mod:`whisper_cbpf.samplers.jax._adapters`). The
decision is bilby's, ``lo < value < hi`` for every bound, strict on both sides. Nested sampling
also renormalises the prior to the region the wall allows, as bilby does, so a walled model's ln Z
is not lowered by the prior volume behind its wall (:mod:`whisper_cbpf.samplers.nested`).

Two modes (``constraint=`` on the builders), plus ``None`` for whisper <= 0.1.0's no-wall:

``"corrected"`` (default)
    redback's constraints with ONE value corrected. ``nuclear_burning_constraints`` bounds the
    kinetic energy by ``-(14 * 2.4249 - 53.9037) MeV / m_p`` per gram of nickel, 1.91e19 erg/g:
    a sign slip on ``Delta(Ni56)`` and one nucleon mass where 56 atomic mass units belong. Burning
    14 He-4 to Ni-56 releases ``Q = 14 Delta(He4) - Delta(Ni56) = 87.85 MeV`` per 56 u,
    **1.51e18 erg/g** (:data:`E_BURN_PER_G`), 12.6x less -- the largest release of any fuel burnt
    to Ni-56, so the corrected bound is still the lenient one (answered 2026-09-25).
``"redback"``
    redback 1.20's decisions exactly, lenient bound included (:data:`E_BURN_PER_G_REDBACK`).

The formulas are **transcribed** here, because a traced JAX sampler cannot call redback's numpy
conversion functions; ``xp`` selects numpy or ``jax.numpy`` and the expressions are redback's,
textually, in its own order. ``tests/test_constraints.py`` holds the transcription to redback's
own ``get_priors(model, constraint=True).evaluate_constraints`` on 20 000 prior draws per model:
identical accept/reject in ``"redback"`` mode, numpy and JAX. For a redback model not transcribed
here the CPU adapter evaluates redback's own conversion function
(:func:`redback_constraint_setting`), in either mode -- no other redback constraint has a known
defect.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CONSTRAINT_MODES", "E_BURN_PER_G", "E_BURN_PER_G_REDBACK", "MODELS", "check_mode",
           "constraint_values", "constraint_ok", "redback_constraint_setting"]

#: ``constraint=`` values: ``"corrected"`` (default), ``"redback"``, or ``None`` (no wall).
CONSTRAINT_MODES = ("corrected", "redback", None)

# redback.constants (1.20), verbatim, so the "redback" mode is redback's arithmetic.
SOLAR_MASS = 1.988409870698051e33          # g
KM_CGS = 1e5                               # cm
PROTON_MASS = 1.67262192595e-24            # g
MEV_CGS = 1.602176634e-06                  # erg
DAY_TO_S = 86400
#: The atomic mass unit [g], CODATA 2022 (``astropy.constants.u``), the release redback's proton
#: mass is taken from.
AMU_CGS = 1.66053906892e-24

#: Energy released per gram of Ni-56 made by burning helium, ``(14 Delta(He4) - Delta(Ni56)) /
#: 56 u`` with redback's own mass excesses (2.4249 and -53.9037 MeV): 87.85 MeV per 56 u,
#: 1.514e18 erg/g. The default (``"corrected"``) nuclear-burning bound.
E_BURN_PER_G = (56.0 / 4.0 * 2.4249 + 53.9037) * MEV_CGS / (56.0 * AMU_CGS)

#: redback 1.20's ``excess_constant`` in ``nuclear_burning_constraints``, verbatim: 1.911e19
#: erg/g, 12.6x :data:`E_BURN_PER_G`. Used by ``constraint="redback"``.
E_BURN_PER_G_REDBACK = -(56.0 / 4.0 * 2.4249 - 53.9037) / PROTON_MASS * MEV_CGS


def _nan_to_inf(x, xp):
    """redback's ``np.nan_to_num(x, nan=inf, posinf=inf, neginf=inf)``: fail, never pass."""
    return xp.nan_to_num(x, nan=xp.inf, posinf=xp.inf, neginf=xp.inf)


# --- redback's conversion functions (redback/constraints.py, 1.20), transcribed -------------------

def _nuclear_burning(p, xp, e_burn):
    """``nuclear_burning_constraints``: nuclear-burning energy >= ejecta kinetic energy."""
    mej = p["mej"] * SOLAR_MASS
    vej = p["vej"] * KM_CGS
    kinetic_energy = 0.5 * mej * (vej / 2.0) ** 2
    emax = e_burn * mej * p["f_nickel"]
    return {"emax_constraint": kinetic_energy / emax}


def _basic_magnetar(p, xp, e_burn):
    """``basic_magnetar_powered_sn_constraints``: rotational energy >= ejecta kinetic energy."""
    mej = p["mej"] * SOLAR_MASS
    vej = p["vej"] * KM_CGS
    kinetic_energy = 0.5 * mej * vej ** 2
    rotational_energy = 2.6e52 * (p["mass_ns"] / 1.4) ** (3. / 2.) * p["p0"] ** (-2)
    return {"erot_constraint": kinetic_energy / rotational_energy}


def _slsn(p, xp, e_burn):
    """``slsn_constraint``: rotational energy >= kinetic + 1e51 erg, nebular phase at 100-500 d."""
    mej = p["mej"] * SOLAR_MASS
    vej = p["vej"] * KM_CGS
    kinetic_energy = 0.5 * mej * vej ** 2
    rotational_energy = 2.6e52 * (p["mass_ns"] / 1.4) ** (3. / 2.) * p["p0"] ** (-2)
    tnebula = xp.sqrt(3 * p["kappa"] * mej / (4 * np.pi * vej ** 2)) / 86400
    total_energy = kinetic_energy + 1e51
    return {"e_rot_constraint": total_energy / rotational_energy,
            "t_nebula_min": tnebula - 100}


def _general_magnetar(p, xp, e_burn):
    """``general_magnetar_powered_sn_constraints``: ``2 l0 tsd`` >= ejecta kinetic energy."""
    mej = p["mej"] * SOLAR_MASS
    vej = p["vej"] * KM_CGS
    kinetic_energy = 0.5 * mej * vej ** 2
    rotational_energy = 2 * p["l0"] * p["tsd"]
    return {"erot_constraint": kinetic_energy / rotational_energy}


def _cooling_envelope(p, xp, e_burn):
    """``cooling_envelope_constraints``: ``eta >= eta_min``, ``beta <= beta_max``, and, for a
    Gaussian rise (``sigma_t`` present), the stitch at ``xi tfb (1+z)`` within 35 sigma of the
    peak."""
    ms, mbh6 = p["stellar_mass"], p["mbh_6"]
    etamin = 0.01 * (ms ** (-7. / 15.)) * (mbh6 ** (2. / 3.))
    betamax = 12. * (ms ** (7. / 15.)) * (mbh6 ** (-2. / 3.))
    out = {"eta_min_ratio": _nan_to_inf(etamin / p["eta"], xp),
           "beta_max_ratio": _nan_to_inf(p["beta"] / betamax, xp)}
    if "sigma_t" in p:
        bec = p.get("binding_energy_const", 0.8)
        # redback.utils.calc_tfb, seconds
        tfb = 58. * (3600. * 24.) * (mbh6 ** (0.5)) * (ms ** (0.2)) * ((bec / 0.8) ** (-1.5))
        transition_time = p.get("xi", 1.) * tfb * (1. + p.get("redshift", 0.))
        tail = xp.abs(transition_time - p["peak_time"] * DAY_TO_S) / (p["sigma_t"] * DAY_TO_S)
        out["gaussian_stitching_tail"] = _nan_to_inf(tail, xp)
    return out


#: redback model name -> (conversion function, ``{output: (lo, hi)}``): redback 1.20's
#: ``_constraint_settings`` entries for the models whisper ports to JAX. Every other redback model
#: that declares constraints is evaluated through redback itself (CPU adapter only).
MODELS = {
    "arnett": (_nuclear_burning, {"emax_constraint": (0, 1)}),
    "basic_magnetar_powered": (_basic_magnetar, {"erot_constraint": (0, 1)}),
    "slsn": (_slsn, {"e_rot_constraint": (0, 1), "t_nebula_min": (0, 400)}),
    "general_magnetar_slsn": (_general_magnetar, {"erot_constraint": (0, 1)}),
    "cooling_envelope": (_cooling_envelope, {"eta_min_ratio": (0, 1), "beta_max_ratio": (0, 1)}),
    "gaussianrise_cooling_envelope": (_cooling_envelope,
                                      {"eta_min_ratio": (0, 1), "beta_max_ratio": (0, 1),
                                       "gaussian_stitching_tail": (0, 35)}),
}


def check_mode(constraint):
    """Validate a ``constraint=`` value and return it."""
    if constraint not in CONSTRAINT_MODES:
        raise ValueError(f"constraint must be one of {CONSTRAINT_MODES}, got {constraint!r}")
    return constraint


def constraint_values(model, p, *, mode="corrected", xp=np):
    """redback's conversion-function outputs for ``model`` at parameters ``p`` (a dict of scalars
    or arrays; ``redshift``, ``xi``, ``binding_energy_const`` read where redback reads them)."""
    fn, _ = MODELS[model]
    return fn(p, xp, E_BURN_PER_G if check_mode(mode) == "corrected" else E_BURN_PER_G_REDBACK)


def constraint_ok(model, p, *, mode="corrected", xp=np):
    """True where ``p`` passes every constraint of ``model``: bilby's ``lo < value < hi``.

    Always true for ``mode=None`` or a model not in :data:`MODELS`. Traceable with ``xp=jax.numpy``
    (one parameter set, or arrays of them). The supernova and TDE ports run in float64, which is
    what the transcription is checked in; several of these cgs energies (1e55 erg) overflow
    float32.
    """
    if check_mode(mode) is None or model not in MODELS:
        return True
    _, bounds = MODELS[model]
    vals = constraint_values(model, p, mode=mode, xp=xp)
    ok = True
    for key, (lo, hi) in bounds.items():
        ok = ok & (vals[key] > lo) & (vals[key] < hi)
    return ok


def redback_constraint_setting(model):
    """``(conversion_function, {output: (lo, hi)})`` the INSTALLED redback declares for ``model``,
    or ``None`` (none declared, redback < 1.20, or redback absent)."""
    try:
        from .redback_adapter import _import_redback
        _import_redback()
        import redback.priors as rp
    except Exception:                       # no redback: nothing it declares
        return None
    return getattr(rp, "_constraint_settings", {}).get(model)
