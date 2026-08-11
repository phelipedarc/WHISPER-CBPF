"""Astropy-unit validation and conversion for the three light-curve data modes.

Whisper stores **one canonical internal unit per data mode** so the rest of the package never has to
think about units again:

================  ===========================  ==============================================
``data_mode``     canonical internal unit      accepted inputs
================  ===========================  ==============================================
``flux_density``  janskys (:data:`CANON_FD`)   F_nu (Jy, mJy, uJy, ...) *or* F_lambda
                                               (erg s^-1 cm^-2 Angstrom^-1) -- the latter is
                                               converted with ``u.spectral_density(lambda_eff)``
                                               and therefore REQUIRES a per-point effective
                                               wavelength.
``magnitude``     dimensionless AB mag         a dimensionless / mag column only.
``flux``          erg s^-1 cm^-2 (:data:`CANON_F`)  any band-integrated energy flux
                                               (power / area).
================  ===========================  ==============================================

Everything funnels through :func:`to_canonical`, which takes raw values + a unit (string or
``astropy.units.Unit``) + the target ``data_mode`` and returns plain ``float`` ``ndarray`` in the
canonical unit. A column that arrives with *no* unit metadata gets a documented per-mode default
(:data:`DEFAULT_UNITS`) and a warning -- we never silently guess a unit from a different mode.
"""
from __future__ import annotations

import warnings

import numpy as np
import astropy.units as u

# --- canonical internal units (one per data mode) ---
CANON_FD = u.Jy                          # flux_density  -> janskys
CANON_F = u.erg / u.s / u.cm**2          # flux (band-integrated) -> energy flux
# magnitude has no astropy unit object we store; it is dimensionless AB mag.

#: Documented default unit applied (with a warning) when a column carries no unit metadata.
DEFAULT_UNITS = {
    "flux_density": "Jy",                # historical Whisper behaviour: bare flux column == Jy
    "flux": "erg / (s cm2)",
    "magnitude": "",                     # dimensionless
}

VALID_DATA_MODES = ("flux_density", "magnitude", "flux")

# Physical types used to classify an incoming flux_density unit.
_FNU_PTYPE = (u.Jy).physical_type                    # 'spectral flux density'
_FLAM_PTYPE = (u.erg / u.s / u.cm**2 / u.AA).physical_type  # 'spectral flux density wav'


def as_unit(unit):
    """Coerce ``unit`` (``str`` | ``astropy`` unit | ``None``) to a :class:`~astropy.units.UnitBase`.

    ``None`` and ``""`` mean *dimensionless*. Raises ``ValueError`` on an unparseable string.
    """
    if unit is None or unit == "":
        return u.dimensionless_unscaled
    if isinstance(unit, str):
        try:
            return u.Unit(unit)
        except Exception as exc:  # pragma: no cover - exercised via to_canonical
            raise ValueError(f"Could not parse unit {unit!r}: {exc}") from None
    return u.Unit(unit)


def _is_fnu(unit):
    return unit.physical_type == _FNU_PTYPE


def _is_flambda(unit):
    return unit.physical_type == _FLAM_PTYPE


def to_flux_density_jy(values, unit, lambda_eff=None):
    """Convert ``values * unit`` to flux density in **Jy** (the canonical flux_density unit).

    Accepts F_nu (Jy/mJy/uJy/...) directly. F_lambda (erg s^-1 cm^-2 Angstrom^-1) is converted via
    ``u.spectral_density(lambda_eff)`` and so REQUIRES ``lambda_eff`` (scalar or per-point, in
    Angstrom); a clear ``ValueError`` is raised if it is missing.
    """
    unit = as_unit(unit)
    values = np.asarray(values, dtype=float)
    if _is_fnu(unit):
        return (values * unit).to_value(CANON_FD)
    if _is_flambda(unit):
        if lambda_eff is None:
            raise ValueError(
                "Converting F_lambda flux density "
                f"({unit}) to Jy requires a per-point effective wavelength (lambda_eff), "
                "but none is available. Resolve the band (FILTER_LOOKUP/SVO) or pass "
                "lambda_eff explicitly.")
        lam = np.asarray(lambda_eff, dtype=float)
        # A point with a real flux but no resolved wavelength cannot be converted -- per the spec
        # this must error clearly, not silently become NaN (and then get dropped by a quality cut).
        missing = ~np.isfinite(lam) & np.isfinite(values)
        if np.any(missing):
            n = int(np.count_nonzero(missing))
            raise ValueError(
                f"Converting F_lambda flux density ({unit}) to Jy needs a per-point effective "
                f"wavelength, but {n} point(s) have an unresolved band (NaN lambda_eff). Resolve "
                "the band (FILTER_LOOKUP/SVO), register it with "
                "whisper_labia.io.svo.register_manual_band(...), or pass lambda_eff explicitly.")
        return (values * unit).to_value(CANON_FD, equivalencies=u.spectral_density(lam * u.AA))
    raise ValueError(
        f"Unit {unit} is not a recognised flux-density unit. Expected F_nu (e.g. Jy, mJy, uJy) "
        "or F_lambda (e.g. erg / (s cm2 Angstrom)).")


def to_flux_cgs(values, unit):
    """Convert band-integrated ``values * unit`` to canonical energy flux (erg s^-1 cm^-2)."""
    unit = as_unit(unit)
    values = np.asarray(values, dtype=float)
    target_ptype = CANON_F.physical_type
    if unit.physical_type != target_ptype:
        raise ValueError(
            f"data_mode='flux' expects a band-integrated energy flux (physical type "
            f"{target_ptype!r}, e.g. erg / (s cm2)); got unit {unit} with physical type "
            f"{unit.physical_type!r}.")
    return (values * unit).to_value(CANON_F)


def check_magnitude_unit(unit):
    """Validate that a magnitude column is dimensionless / an AB magnitude.

    Magnitudes are logarithmic and must not carry a flux unit. Raises a clear ``ValueError`` when
    given e.g. ``Jy`` or ``erg/(s cm2)``.
    """
    unit = as_unit(unit)
    if unit == u.dimensionless_unscaled:
        return
    # astropy models magnitudes as a logarithmic unit whose physical_type is 'unknown'/dimensionless.
    try:
        if unit.physical_type in ("dimensionless", "unknown") and unit.is_equivalent(u.mag):
            return
    except Exception:
        pass
    if unit.is_equivalent(u.mag):
        return
    raise ValueError(
        f"data_mode='magnitude' requires a dimensionless AB magnitude column, but the column "
        f"carries unit {unit} (physical type {unit.physical_type!r}). Magnitudes cannot have a "
        "flux unit -- pass the data as data_mode='flux_density' instead, or drop the unit.")


def to_canonical(values, unit, data_mode, *, lambda_eff=None, warn_default=True):
    """Validate + convert ``values`` (with ``unit``) into the canonical unit for ``data_mode``.

    ``unit=None`` means "no unit metadata": a documented per-mode default (:data:`DEFAULT_UNITS`) is
    applied with a warning. Returns a plain ``float`` ``ndarray``. See the module docstring for the
    canonical-unit table.
    """
    if data_mode not in VALID_DATA_MODES:
        raise ValueError(f"Unknown data_mode {data_mode!r}; expected one of {VALID_DATA_MODES}.")

    if unit is None:
        default = DEFAULT_UNITS[data_mode]
        if warn_default:
            shown = default or "dimensionless"
            warnings.warn(
                f"Column for data_mode={data_mode!r} has no unit metadata; assuming the documented "
                f"default {shown!r}. Pass an explicit unit to silence this.",
                stacklevel=2)
        unit = default

    if data_mode == "magnitude":
        check_magnitude_unit(unit)
        return np.asarray(values, dtype=float)
    if data_mode == "flux_density":
        return to_flux_density_jy(values, unit, lambda_eff=lambda_eff)
    if data_mode == "flux":
        return to_flux_cgs(values, unit)
    raise AssertionError("unreachable")  # pragma: no cover
