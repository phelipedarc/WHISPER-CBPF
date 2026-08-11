"""Flexible CSV -> :class:`LightCurve` loader."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .bands import FILTER_LOOKUP, group_bands, normalize_bands, resolve_bands
from .schema import LightCurve
from .units import to_canonical

# Canonical field -> accepted header synonyms (matched case-insensitively).
CANONICAL_SYNONYMS = {
    "time": ["time", "mjd", "jd", "hjd", "t", "date"],
    "magnitude": ["magnitude", "mag", "apparent_mag", "app_mag", "appmag"],
    "magnitude_err": ["e_magnitude", "magnitude_err", "magnitude_error", "magerr",
                      "mag_err", "e_mag", "emag", "apparent_magerr", "dmag", "sigma_mag"],
    "flux": ["flux", "flux_density", "fluxdensity", "forcediffimflux", "fnu"],
    "flux_err": ["flux_err", "fluxerr", "flux_error", "e_flux", "flux_density_err",
                 "forcediffimfluxunc", "sigma_flux"],
    "band": ["band", "filter", "filtercode", "filtername", "passband", "bandpass"],
    "system": ["system", "magsystem", "magsys", "photsystem"],
    "name": ["event", "name", "object", "objectid", "oid", "iau", "transient", "sn"],
    "upper_limit": ["upper_limit", "upperlimit", "islimit", "is_limit", "nondetection", "ul"],
    "redshift": ["redshift", "zhel", "zhelio", "z_helio", "zspec", "z_spec", "z_cmb", "zcmb"],
}

_TRUE = {"1", "true", "t", "yes", "y"}


def _read_table(path, delimiter=None):
    if delimiter is not None:
        return pd.read_csv(path, delimiter=delimiter)
    try:  # auto-sniff comma/semicolon/whitespace
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)


def _norm(header):
    return str(header).strip().lower()


def _to_bool(series):
    return np.array([str(v).strip().lower() in _TRUE for v in series])


def _resolve_redshift(redshift_arg, df, cols):
    """Resolve redshift: explicit argument > first finite 'redshift' column value > unknown.

    A finite negative value is returned as-is and rejected later by ``LightCurve`` validation (fatal).
    A present-but-all-NaN/blank column has no usable value, so it degrades to *unknown* (warn + default
    prior) rather than being fatal -- an empty column is missing data, not an invalid redshift.
    """
    if redshift_arg is not None:
        return float(redshift_arg)
    if "redshift" in cols:
        vals = pd.to_numeric(df[cols["redshift"]], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            return float(finite[0])
    warnings.warn(
        "No usable redshift (no redshift= argument and no finite 'redshift' column value). The light "
        "curve is marked redshift_known=False and carries a default redshift_prior; a redshift prior "
        "will be SAMPLED, not assumed, when fitting models. Pass redshift=... to set it explicitly.",
        stacklevel=3)
    return None


def _resolve_columns(columns, column_map=None):
    by_norm = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    resolved = dict(column_map or {})
    for canon, syns in CANONICAL_SYNONYMS.items():
        if canon in resolved:
            continue
        for s in syns:
            if s in by_norm:
                resolved[canon] = by_norm[s]
                break
    return resolved


def load_lightcurve(path, *, name=None, redshift=None, luminosity_distance=None, data_mode=None,
                    flux_unit=None, magnitude_unit=None, column_map=None, band_aliases=None,
                    band_lookup=None, normalize=True, default_band=None, quality_cuts=True,
                    drop_nonfinite=True, flag_filters=None, time_min=None, time_max=None,
                    bands=None, min_snr=None, explosion_date=None, delimiter=None,
                    resolve_band_info=True, svo_fallback=True) -> LightCurve:
    """Load a light-curve CSV into a canonical :class:`LightCurve`.

    Auto-detects columns (case-insensitive synonyms; override with ``column_map``), normalizes band
    names, drops bad rows (non-finite values, non-positive errors -- upper limits are kept), and
    optionally filters by band/time/SNR. Supports magnitude- or flux-based inputs.

    **Redshift** is resolved in priority order ``redshift=`` argument > a ``redshift`` column > unknown.
    When unknown the load does *not* fail: the :class:`LightCurve` records ``redshift_known=False`` and
    a default ``redshift_prior``, and a warning is emitted that a redshift prior must be supplied for
    the models. ``z`` is validated (``z >= 0``; ``z == 0`` needs ``luminosity_distance``; negative/NaN
    is fatal).

    **Units** use ``astropy.units``. ``flux_unit`` may be F_nu (Jy/mJy/uJy) or F_lambda
    (erg/s/cm^2/Angstrom; converted to Jy via the per-band effective wavelength); ``magnitude_unit``
    must be a dimensionless AB magnitude. A flux column with no ``flux_unit`` warns and assumes the
    documented default (Jy for ``flux_density``). ``data_mode`` is one of
    ``flux_density`` / ``magnitude`` / ``flux`` (inferred from the columns when omitted).

    **Bands** are resolved (``resolve_band_info=True``) to per-point effective wavelength + zero
    point via FILTER_LOOKUP, falling back to the SVO Filter Profile Service for unknown bands
    (``svo_fallback``; degrades gracefully + warns when SVO/astroquery is unavailable).

    Other options: ``column_map={'time': 'MJD', ...}`` to force a mapping; ``default_band='ztfg'``;
    ``band_lookup=True`` for broadband grouping; ``flag_filters={'catflags': 0}``; ``time_min/time_max``
    (MJD), ``bands=[...]`` and ``min_snr``; ``explosion_date=<MJD>`` for days-since-explosion.
    """
    path = Path(path)
    df = _read_table(path, delimiter)
    cols = _resolve_columns(df.columns, column_map)

    if "time" not in cols:
        raise ValueError(
            f"No time column found (looked for {CANONICAL_SYNONYMS['time']}). "
            f"Available columns: {list(df.columns)}. Pass column_map={{'time': '<column>'}}.")
    time = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy(dtype=float)

    if "band" in cols:
        band = np.array([str(b) for b in df[cols["band"]].to_numpy()])
    elif default_band is not None:
        band = np.array([str(default_band)] * len(df))
    else:
        raise ValueError(
            f"No band/filter column found (looked for {CANONICAL_SYNONYMS['band']}). "
            f"Available columns: {list(df.columns)}. "
            f"Pass default_band='...' or column_map={{'band': '<column>'}}.")

    def num(field):
        return (pd.to_numeric(df[cols[field]], errors="coerce").to_numpy(dtype=float)
                if field in cols else None)

    magnitude, magnitude_err = num("magnitude"), num("magnitude_err")
    flux, flux_err = num("flux"), num("flux_err")
    if magnitude is None and flux is None:
        raise ValueError(
            f"No magnitude or flux column found. Available columns: {list(df.columns)}.")

    upper_limit = _to_bool(df[cols["upper_limit"]]) if "upper_limit" in cols else None

    system = None
    if "system" in cols:
        system = np.array([
            "unknown" if str(s).strip().lower() in ("nan", "none", "") else str(s).strip()
            for s in df[cols["system"]].to_numpy()
        ])

    if name is None and "name" in cols and len(df):
        name = str(df[cols["name"]].iloc[0])

    if normalize:
        band = normalize_bands(band, aliases=band_aliases)
    if band_lookup is not None and band_lookup is not False:
        lookup = FILTER_LOOKUP if band_lookup is True else band_lookup
        band = group_bands(band, lookup=lookup)

    # --- redshift: explicit argument > 'redshift' column > unknown (warn, do not fail) ---
    redshift = _resolve_redshift(redshift, df, cols)

    # --- data_mode: explicit > inferred from columns present ---
    if data_mode is None:
        data_mode = "magnitude" if magnitude is not None else "flux_density"

    # --- per-band effective wavelength + zero point (FILTER_LOOKUP -> SVO fallback) ---
    lambda_eff = zero_point = None
    if resolve_band_info:
        lambda_eff, zero_point, _ = resolve_bands(band, svo_fallback=svo_fallback)

    # --- astropy unit handling: convert flux/magnitude columns to canonical units ---
    if magnitude is not None:
        magnitude = to_canonical(magnitude, magnitude_unit, "magnitude", warn_default=True)
    if flux is not None:
        flux_mode = data_mode if data_mode == "flux" else "flux_density"
        flux = to_canonical(flux, flux_unit, flux_mode, lambda_eff=lambda_eff, warn_default=True)
        if flux_err is not None:
            flux_err = to_canonical(flux_err, flux_unit, flux_mode,
                                    lambda_eff=lambda_eff, warn_default=False)

    # --- row mask: quality cuts (upper limits are exempt from the error cut) ---
    ul_mask = upper_limit if upper_limit is not None else np.zeros(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    primary = magnitude if magnitude is not None else flux
    primary_err = magnitude_err if magnitude is not None else flux_err
    if drop_nonfinite:
        mask &= np.isfinite(time)
        mask &= np.isfinite(primary)
    if quality_cuts and primary_err is not None:
        mask &= (np.isfinite(primary_err) & (primary_err > 0)) | ul_mask
    if flag_filters:
        for fcol, cond in flag_filters.items():
            if fcol not in df.columns:
                raise ValueError(f"flag_filters column {fcol!r} not in {list(df.columns)}")
            values = df[fcol].to_numpy()
            mask &= (np.array([bool(cond(v)) for v in values]) if callable(cond)
                     else (values == cond))

    def m(v):
        return None if v is None else v[mask]

    lc = LightCurve(
        time=time[mask], band=band[mask],
        magnitude=m(magnitude), magnitude_err=m(magnitude_err),
        flux=m(flux), flux_err=m(flux_err), upper_limit=m(upper_limit), system=m(system),
        lambda_eff=m(lambda_eff), zero_point=m(zero_point),
        name=name, redshift=redshift, luminosity_distance=luminosity_distance, data_mode=data_mode,
        meta={"source_file": str(path), "n_rows_raw": int(len(df)),
              "n_rows_kept": int(mask.sum())},
    )

    if bands is not None:
        lc = lc.select_bands(bands)
    if time_min is not None or time_max is not None:
        lc = lc.select_time_window(time_min, time_max)
    if min_snr is not None:
        lc = lc.select_snr(min_snr)
    if explosion_date is not None:
        lc = lc.set_explosion_date(explosion_date)
    return lc
