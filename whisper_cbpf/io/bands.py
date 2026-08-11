"""Band-name normalization and grouping.

Two composable layers:

1. :func:`normalize_band` / :func:`normalize_bands` -- light formatting of raw survey codes via
   :data:`DEFAULT_BAND_ALIASES` (e.g. 'zg' / 'ZTF_g' -> 'ztfg'; aliases are matched case-insensitively).
   Unknown labels pass through unchanged.

2. :func:`group_bands` -- collapse many heterogeneous filters into a small effective-band ladder
   via :data:`FILTER_LOOKUP` (e.g. 'B' -> 'g-band', 'V' -> 'r-band', 'Ks' -> 'K-band',
   'F606W' -> 'r-band'). Band labels are case-sensitive in astronomy ('r' is SDSS r, 'R' is
   Cousins R), so the lookup lists case variants explicitly.

They compose: e.g. 'zg' --normalize--> 'ztfg' --group--> 'g-band'.
"""
from __future__ import annotations

import warnings

import numpy as np

# Light survey-code normalization (extend per-call with the ``aliases`` argument).
DEFAULT_BAND_ALIASES = {
    "zg": "ztfg", "zr": "ztfr", "zi": "ztfi",
    "ztf_g": "ztfg", "ztf_r": "ztfr", "ztf_i": "ztfi", "ztf_z": "ztfz",
}

# Broadband grouping supplied by the user: collapse heterogeneous filters into an effective-band
# ladder. Comments give the rough wavelength-order index of each group.
FILTER_LOOKUP = {
    # U-group (+2)
    'u': 'U-band', 'U': 'U-band', "u'": 'U-band',
    'uvw2': 'U-band', 'uvm2': 'U-band', 'uvw1': 'U-band', 'uvot_u': 'U-band',
    'F336W': 'U-band', 'f336w': 'U-band',
    # g-group (+1)
    'g': 'g-band', 'G': 'g-band',
    'lsstg': 'g-band', 'g-ztf': 'g-band', 'ztfg': 'g-band', 'g-p1': 'g-band',
    'ZTF_g': 'g-band', 'ztf_g': 'g-band',
    'B': 'g-band', 'b': 'g-band',
    'F475W': 'g-band', 'f475w': 'g-band',
    # r-group (0)
    'r': 'r-band', 'R': 'r-band', "r'": 'r-band',
    'lsstr': 'r-band', 'r-ztf': 'r-band', 'ztfr': 'r-band', 'r-p1': 'r-band',
    'ZTF_r': 'r-band', 'ztf_r': 'r-band',
    'V': 'r-band', 'v': 'r-band', 'Vc': 'r-band',
    'F606W': 'r-band', 'F070W': 'r-band',
    'F625W': 'r-band', 'f625w': 'r-band',
    # i-group (-1)
    'i': 'i-band', 'I': 'i-band', 'Ic': 'i-band', "i'": 'i-band',
    'lssti': 'i-band', 'i-ztf': 'i-band', 'ztfi': 'i-band', 'i-p1': 'i-band',
    'ZTF_i': 'i-band', 'ztf_i': 'i-band',
    'F814W': 'i-band',
    'F775W': 'i-band', 'f775w': 'i-band',
    # z-group (-2)
    'z': 'z-band', 'Z': 'z-band', "z'": 'z-band',
    'ZTF_z': 'z-band', 'ztf_z': 'z-band', 'ztfz': 'z-band',
    'lsstz': 'z-band', 'z-ztf': 'z-band', 'ztfz': 'z-band', 'z-p1': 'z-band',
    'y': 'z-band', 'Y': 'z-band', 'lssty': 'z-band', 'y-p1': 'z-band',
    'F090W': 'z-band',
    'F850W': 'z-band', 'f850w': 'z-band',
    # J-group (-3)
    'J': 'J-band', 'j': 'J-band', '2massj': 'J-band',
    'F110W': 'J-band', 'f110w': 'J-band',
    'F115W': 'J-band', 'f115w': 'J-band',
    'F125W': 'J-band', 'f125w': 'J-band',
    'J1': 'J-band', 'j1': 'J-band',
    # H-group (-4)  -- NOTE: original '2massh' value was 'H-Band' (capital B); fixed to 'H-band'.
    'H': 'H-band', 'h': 'H-band', '2massh': 'H-band',
    'F150W': 'H-band', 'f150w': 'H-band',
    'F160W': 'H-band', 'f160w': 'H-band',
    # K-group (-5)
    'K': 'K-band', 'Ks': 'K-band', 'k': 'K-band', '2massks': 'K-band',
    'F200W': 'K-band', 'f200W': 'K-band',
    'F277W': 'K-band', 'f277W': 'K-band',
    # JWST sim bands (-6)
    'F356W': 'F356W-band', 'f356w': 'F356W-band',
    'F444W': 'F444W-band', 'f444w': 'F444W-band',
}


# ---------------------------------------------------------------------------
# Effective-wavelength + zero-point table for each FILTER_LOOKUP group.
#
# Per the spec, the OPTICAL effective bands are anchored to the LSST ugrizy filters (zero point in
# the AB system == 3631 Jy by definition; the effective wavelengths come from the LSST throughputs).
# The NIR groups (J/H/K) and the JWST sim bands have no LSST equivalent, so they carry documented
# 2MASS / JWST effective wavelengths -- still on the AB zero point Whisper uses internally. ``source``
# records the provenance so callers can tell an LSST anchor from a documented NIR value.
# ---------------------------------------------------------------------------
_AB_ZP_JY = 3631.0  # AB zero point (== whisper_labia.io.photometry.AB_ZEROPOINT_JY)

LSST_BAND_INFO = {
    # optical -- LSST ugrizy anchors (lambda_eff in Angstrom)
    "U-band": {"lambda_eff": 3671.0, "zero_point": _AB_ZP_JY, "filter_id": "LSST/LSST.u", "source": "lsst"},
    "g-band": {"lambda_eff": 4866.0, "zero_point": _AB_ZP_JY, "filter_id": "LSST/LSST.g", "source": "lsst"},
    "r-band": {"lambda_eff": 6215.0, "zero_point": _AB_ZP_JY, "filter_id": "LSST/LSST.r", "source": "lsst"},
    "i-band": {"lambda_eff": 7545.0, "zero_point": _AB_ZP_JY, "filter_id": "LSST/LSST.i", "source": "lsst"},
    # the FILTER_LOOKUP 'z-band' group folds in LSST y as well; anchored on LSST z.
    "z-band": {"lambda_eff": 8679.0, "zero_point": _AB_ZP_JY, "filter_id": "LSST/LSST.z", "source": "lsst"},
    # NIR -- documented 2MASS effective wavelengths (AB zero point as used internally)
    "J-band": {"lambda_eff": 12350.0, "zero_point": _AB_ZP_JY, "filter_id": "2MASS/2MASS.J", "source": "documented"},
    "H-band": {"lambda_eff": 16620.0, "zero_point": _AB_ZP_JY, "filter_id": "2MASS/2MASS.H", "source": "documented"},
    "K-band": {"lambda_eff": 21590.0, "zero_point": _AB_ZP_JY, "filter_id": "2MASS/2MASS.Ks", "source": "documented"},
    # JWST NIRCam sim bands -- documented effective wavelengths
    "F356W-band": {"lambda_eff": 35690.0, "zero_point": _AB_ZP_JY, "filter_id": "JWST/NIRCam.F356W", "source": "documented"},
    "F444W-band": {"lambda_eff": 44040.0, "zero_point": _AB_ZP_JY, "filter_id": "JWST/NIRCam.F444W", "source": "documented"},
}


def resolve_band(band, *, lookup=None, svo_fallback=True, lambda_eff_hint=None, warn=True):
    """Resolve one band to ``{group, lambda_eff (AA), zero_point (Jy), filter_id, source}``.

    Resolution order:

    1. group ``band`` via ``lookup`` (default :data:`FILTER_LOOKUP`); if the resulting group (or the
       band itself) is in :data:`LSST_BAND_INFO`, return that anchored wavelength + zero point;
    2. otherwise the band has no local match -- emit a clear warning naming it and, if
       ``svo_fallback`` is on, query the SVO Filter Profile Service
       (:func:`whisper_labia.io.svo.resolve_band_svo`);
    3. if SVO is unavailable / unknown, warn and return ``lambda_eff=zero_point=None`` (the load
       still succeeds; the user can supply values via ``svo.register_manual_band``).

    ``source`` is one of ``'lsst'`` / ``'documented'`` / ``'svo'`` / ``'manual'`` / ``'unresolved'``.
    """
    from . import svo as _svo  # local import keeps astroquery fully optional

    lookup = FILTER_LOOKUP if lookup is None else lookup
    raw = str(band).strip()
    # Compose alias -> group: a label not in the lookup is normalised first (e.g. 'ZTF_g' -> 'ztfg'),
    # so survey codes resolve without duplicating every case variant in the lookup table.
    key = raw if raw in lookup else normalize_band(raw)
    group = lookup.get(key, key)

    info = LSST_BAND_INFO.get(group) or LSST_BAND_INFO.get(key) or LSST_BAND_INFO.get(raw)
    if info is not None:
        return {"group": group, "lambda_eff": info["lambda_eff"],
                "zero_point": info["zero_point"], "filter_id": info["filter_id"],
                "source": info["source"]}

    # A manual override takes precedence and is not a "miss" worth warning about.
    if raw in _svo._MANUAL_BANDS:
        res = _svo.resolve_band_svo(raw, lambda_eff_hint=lambda_eff_hint)
        return {"group": group, "lambda_eff": res["lambda_eff"],
                "zero_point": res["zero_point"], "filter_id": res["filter_id"],
                "source": res["source"]}

    # No local match.
    if key not in lookup:
        if warn:
            warnings.warn(
                f"Band {raw!r} is not in FILTER_LOOKUP; attempting SVO fallback."
                if svo_fallback else
                f"Band {raw!r} is not in FILTER_LOOKUP and SVO fallback is disabled.",
                stacklevel=2)
    if svo_fallback:
        try:
            res = _svo.resolve_band_svo(raw, lambda_eff_hint=lambda_eff_hint)
            return {"group": group, "lambda_eff": res["lambda_eff"],
                    "zero_point": res["zero_point"], "filter_id": res["filter_id"],
                    "source": res["source"]}
        except _svo.SvoUnavailable as exc:
            if warn:
                warnings.warn(
                    f"Could not resolve band {raw!r} via SVO ({exc}). Effective wavelength / zero "
                    f"point are unknown; supply them with "
                    f"whisper_labia.io.svo.register_manual_band({raw!r}, lambda_eff, zero_point).",
                    stacklevel=2)
    return {"group": group, "lambda_eff": None, "zero_point": None,
            "filter_id": None, "source": "unresolved"}


def resolve_bands(bands, *, lookup=None, svo_fallback=True, warn=True):
    """Vectorized :func:`resolve_band`.

    Returns ``(lambda_eff, zero_point, info)`` where the first two are ``float`` ``ndarray`` (NaN
    where unresolved) aligned with ``bands``, and ``info`` maps each distinct raw band to its full
    resolution dict. Each distinct band is resolved once (so SVO/network is hit at most once per
    band, then cached).
    """
    info = {}
    for b in bands:
        key = str(b).strip()
        if key not in info:
            info[key] = resolve_band(key, lookup=lookup, svo_fallback=svo_fallback, warn=warn)
    lam = np.array([info[str(b).strip()]["lambda_eff"]
                    if info[str(b).strip()]["lambda_eff"] is not None else np.nan
                    for b in bands], dtype=float)
    zp = np.array([info[str(b).strip()]["zero_point"]
                   if info[str(b).strip()]["zero_point"] is not None else np.nan
                   for b in bands], dtype=float)
    return lam, zp, info


def normalize_band(band, aliases=None, warn_unknown=False, known=None):
    """Normalize a single band label (trim whitespace, apply the alias map).

    The alias map is tried case-sensitively first, then case-insensitively — so survey codes reach their
    canonical band regardless of case (``ZTF_g`` / ``ztf_g`` / ``ZTF_G`` → ``ztfg``). The alias keys are
    survey prefixes that never collide with case-sensitive single-letter bands, so this is safe.
    """
    table = DEFAULT_BAND_ALIASES if aliases is None else {**DEFAULT_BAND_ALIASES, **aliases}
    raw = str(band).strip()
    if raw in table:
        return table[raw]
    lower = {k.lower(): v for k, v in table.items()}
    if raw.lower() in lower:
        return lower[raw.lower()]
    if warn_unknown and known is not None and raw not in known:
        warnings.warn(f"Unrecognized band {raw!r} (left unchanged).", stacklevel=2)
    return raw


def normalize_bands(bands, aliases=None, warn_unknown=False, known=None):
    """Vectorized :func:`normalize_band` -> ``np.ndarray`` of normalized labels."""
    return np.array([
        normalize_band(b, aliases=aliases, warn_unknown=warn_unknown, known=known)
        for b in bands
    ])


def group_bands(bands, lookup=None, default=None, warn_unknown=False):
    """Map each band to its effective broadband via ``lookup`` (default :data:`FILTER_LOOKUP`).

    Labels absent from the lookup are left unchanged (or set to ``default``); set
    ``warn_unknown=True`` to warn about them.
    """
    lookup = FILTER_LOOKUP if lookup is None else lookup
    out, unknown = [], []
    for b in bands:
        key = str(b).strip()
        if key in lookup:
            out.append(lookup[key])
        else:
            unknown.append(key)
            out.append(key if default is None else default)
    if warn_unknown and unknown:
        warnings.warn(
            f"{len(set(unknown))} band(s) not in lookup, left ungrouped: {sorted(set(unknown))}",
            stacklevel=2)
    return np.array(out)


def unmapped_bands(bands, lookup=None):
    """Return the sorted distinct labels in ``bands`` not covered by ``lookup``."""
    lookup = FILTER_LOOKUP if lookup is None else lookup
    return sorted({str(b).strip() for b in bands if str(b).strip() not in lookup})
