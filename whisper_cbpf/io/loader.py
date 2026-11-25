"""Flexible table -> :class:`LightCurve` loader, with ZTF and LSST alert presets (``survey=``)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import surveys as _surveys
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

#: Time columns that hold Julian dates when their values are JD-sized; they are read as MJD.
_JD_COLUMNS = ("jd", "hjd")
#: ``MJD = JD - JD_MJD_OFFSET``.
JD_MJD_OFFSET = 2400000.5


def _read_table(path, delimiter=None):
    if isinstance(path, pd.DataFrame):
        return path.copy()
    if isinstance(path, (list, tuple)):
        return pd.DataFrame(list(path))
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


def _check_survey_arguments(survey, given):
    """Refuse loader arguments a survey preset sets itself (``normalize`` must stay True)."""
    clash = [k for k, v in given.items()
             if ((v is not True) if k == "normalize" else (v is not None and v is not False))]
    if clash:
        raise ValueError(
            f"{', '.join(clash)} cannot be combined with survey={survey!r}: the preset fixes the "
            f"fields, units and survey band labels itself (whisper_cbpf.io.surveys). Leave "
            f"{'it' if len(clash) == 1 else 'them'} out, or load with survey=None and map the "
            f"columns yourself.")


def _select_bands(lc, bands, *, survey, normalize, aliases, lookup):
    """``lc.select_bands`` after spelling ``bands`` the way the loader spelled the band column."""
    asked = [str(b) for b in ([bands] if isinstance(bands, str) else bands)]
    if survey is not None:
        wanted = [_surveys.survey_band(b, survey) for b in asked]
    else:
        wanted = [str(b) for b in normalize_bands(asked, aliases=aliases)] if normalize else asked
        if lookup is not None:
            wanted = [str(b) for b in group_bands(wanted, lookup=lookup)]
    out = lc.select_bands(wanted)
    if len(out) == 0 and len(lc) > 0:
        raise ValueError(
            f"bands={asked!r} (read as {wanted!r}) match none of this light curve's bands "
            f"{lc.bands!r}. Name bands as the band column spells them after normalisation "
            f"(e.g. 'zg' is 'ztfg'), or leave bands= out.")
    return out


def load_lightcurve(path, *, survey=None, limits=None, name=None, redshift=None,
                    luminosity_distance=None, data_mode=None, flux_unit=None, magnitude_unit=None,
                    column_map=None, band_aliases=None, band_lookup=None, normalize=True,
                    default_band=None, quality_cuts=True, drop_nonfinite=True, flag_filters=None,
                    time_min=None, time_max=None, bands=None, min_snr=None, explosion_date=None,
                    delimiter=None, resolve_band_info=True, svo_fallback=True) -> LightCurve:
    """Load a light curve (a file, a table or alert records) into a :class:`LightCurve`.

    Without ``survey=`` the table's columns are found by name (case-insensitive synonyms, see
    ``CANONICAL_SYNONYMS``; override with ``column_map``), band labels are normalised (``zg`` ->
    ``ztfg``), and bad rows are dropped (non-finite values, non-positive errors; upper limits are
    kept). Magnitude and flux inputs are both accepted.

    With ``survey="ztf"`` or ``"lsst"`` the records are read in the broker's own fields (ALeRCE, the
    ZTF alert packet, Fink, the Rubin alert packet) and turned into difference-imaging AB magnitudes:
    ZTF ``magpsf`` / ``sigmapsf`` where ``isdiffpos`` is positive, with ``diffmaglim`` non-detections
    as upper limits; LSST diaSource ``psfFlux`` / ``psfFluxErr`` in nJy as ``m = 31.4 - 2.5
    log10(psfFlux)``, with forced-photometry epochs as upper limits at ``31.4 - 2.5 log10(5
    psfFluxErr)``. ``magpsf_corr`` and ``scienceFlux`` are never used. Bands are always labelled
    ``ztfg ztfr ztfi`` / ``lsstu lsstg lsstr lssti lsstz lssty``, never bare letters (a bare ``g`` is
    SDSS in redback). Rows are sorted by time; negative fluxes, flagged or withdrawn rows, repeats of
    one exposure and limits at a detection's epoch are dropped and counted in
    ``lc.meta["n_dropped"]``. The full rules are in :mod:`whisper_cbpf.io.surveys`.

    Selections run in this order, all on the input's MJD clock: ``bands``, ``time_min`` /
    ``time_max``, ``min_snr``; ``explosion_date`` shifts the clock last, so a window is never compared
    against days since explosion.

    Parameters
    ----------
    path : str, pathlib.Path, file-like, pandas.DataFrame or list of dict
        The data: a CSV (or, with ``survey=``, a ``.json`` file), a table, or a list of records. With
        ``survey=`` also one alert packet or ALeRCE's ``{"detections": ..., "non_detections": ...}``.
    survey : {"ztf", "lsst"}, optional
        Read ``path`` as that survey's alert photometry (see above).
    limits : same types as ``path``, optional
        With ``survey=``: rows that become upper limits, ZTF non-detections (``diffmaglim``) or LSST
        forced photometry (``psfFluxErr``, e.g. Fink's ``/fp``). Rubin alert packets carry theirs.
    name : str, optional
        Object name; defaults to a name / ``objectId`` / ``diaObjectId`` field.
    redshift : float, optional
        Priority: this argument, then a ``redshift`` column, then unknown. Unknown does not fail: the
        light curve records ``redshift_known=False`` and a default ``redshift_prior``, with a
        warning. ``z >= 0``; ``z == 0`` needs ``luminosity_distance``.
    luminosity_distance : float, optional
        Mpc; required when ``redshift == 0``.
    data_mode : {"magnitude", "flux_density", "flux"}, optional
        Inferred from the columns when omitted. A survey preset gives ``"magnitude"``.
    flux_unit, magnitude_unit : str or astropy unit, optional
        Units of the flux column (F_nu such as Jy, or F_lambda in erg/s/cm^2/Angstrom, converted
        with each band's effective wavelength) and of the magnitudes (dimensionless AB). ``None``
        warns and assumes Jy / AB.
    column_map : dict, optional
        ``{canonical: column}`` overrides, e.g. ``{"time": "MJD"}``.
    band_aliases : dict, optional
        Extra ``{raw: label}`` entries over ``DEFAULT_BAND_ALIASES``.
    band_lookup : bool or dict, optional
        ``True`` groups bands into effective bands with ``FILTER_LOOKUP`` (``B`` -> ``g-band``).
    normalize : bool, default True
        Apply the band-alias map to the band column.
    default_band : str, optional
        Band for a table without a band column.
    quality_cuts : bool, default True
        Drop detections whose error is not finite and positive.
    drop_nonfinite : bool, default True
        Drop rows with a non-finite time or measurement.
    flag_filters : dict, optional
        ``{column: value or callable}``; keep the rows that match, e.g. ``{"catflags": 0}``.
    time_min, time_max : float, optional
        Keep ``time_min <= time <= time_max``, in the input's MJD.
    bands : str or list of str, optional
        Keep these bands, spelled as in the data or as the loader writes them (``"zg"`` and
        ``"ztfg"`` both work; with ``survey=``, ``"g"`` means the survey's g). Raises when no point
        is left.
    min_snr : float, optional
        Drop detections below this signal-to-noise. Upper limits have no SNR and are kept.
    explosion_date : float, optional
        MJD of day 0: ``time`` becomes days since it (:meth:`LightCurve.set_explosion_date`).
    delimiter : str, optional
        CSV delimiter; sniffed when omitted.
    resolve_band_info : bool, default True
        Fill per-point ``lambda_eff`` and ``zero_point`` from the band labels.
    svo_fallback : bool, default True
        Ask the SVO Filter Profile Service for bands missing from ``FILTER_LOOKUP``.

    Returns
    -------
    LightCurve
        With ``survey=``, ``meta`` also holds ``survey``, ``photometry``, ``time_system``,
        ``upper_limit_sigma`` (5.0, the limits' significance, which the censored likelihood reads
        from here), ``n_dropped`` and ``first_detection_mjd`` (MJD of the first detection left
        after the cuts). A time column named ``jd`` or ``hjd`` holding Julian dates is converted
        to MJD (``JD - 2400000.5``) and ``meta["time_converted"]`` says so.

    Raises
    ------
    ValueError
        A missing time, band or measurement field; ``bands`` that match nothing; an unknown survey,
        band, or an argument the preset fixes itself (``column_map``, ``band_lookup``, ...);
        ``magpsf_corr`` or ``scienceFlux`` without the difference-image field; and, with
        ``survey=``, no detection left after the cuts (not enough data to fit).

    See Also
    --------
    whisper_cbpf.io.surveys : the survey presets' field names and rules.
    LightCurve.set_time_reference : count days from the first detection instead of an explosion.

    Notes
    -----
    A light curve with upper limits is fitted with the censored likelihood, in flux space, by
    default (``space="auto"``); fit the detections alone with ``lc.where(upper_limit=False)``.

    Examples
    --------
    A table in whisper's own columns; ``zg`` is normalised to ``ztfg``:

    >>> import whisper_cbpf as wp
    >>> rows = [{"time": 60000.0, "band": "zg", "magnitude": 19.2, "magnitude_err": 0.08},
    ...         {"time": 60001.0, "band": "zr", "magnitude": 19.0, "magnitude_err": 0.06}]
    >>> wp.load_lightcurve(rows, redshift=0.05, magnitude_unit="mag")
    LightCurve(name=None, n_points=2, bands=['ztfg', 'ztfr'], mode='magnitude', z=0.05)

    ZTF alerts as ALeRCE returns them: the non-detection becomes an upper limit.

    >>> alerts = {"detections": [
    ...     {"mjd": 60001.0, "fid": 1, "magpsf": 19.2, "sigmapsf": 0.08, "isdiffpos": 1},
    ...     {"mjd": 60002.0, "fid": 2, "magpsf": 19.0, "sigmapsf": 0.06, "isdiffpos": 1}],
    ...     "non_detections": [{"mjd": 59999.0, "fid": 1, "diffmaglim": 20.6}]}
    >>> lc = wp.load_lightcurve(alerts, survey="ztf", redshift=0.05)
    >>> lc.bands, lc.upper_limit.tolist(), lc.meta["first_detection_mjd"]
    (['ztfg', 'ztfr'], [True, False, False], 60001.0)

    LSST diaSources (nJy) with Fink's forced photometry as limits, windowed in MJD and then counted
    from the first detection:

    >>> src = [{"midpointMjdTai": 61000.0, "band": "g", "psfFlux": 1000.0, "psfFluxErr": 100.0},
    ...        {"midpointMjdTai": 61003.0, "band": "r", "psfFlux": 1500.0, "psfFluxErr": 120.0}]
    >>> fp = [{"midpointMjdTai": 60998.0, "band": "r", "psfFlux": 20.0, "psfFluxErr": 60.0}]
    >>> lc = wp.load_lightcurve(src, survey="lsst", limits=fp, redshift=0.3, time_max=61002.0)
    >>> [round(float(m), 3) for m in lc.magnitude]
    [25.207, 23.9]
    >>> lc.set_time_reference(lc.meta["first_detection_mjd"], "first detection").time.tolist()
    [-2.0, 0.0]
    """
    if survey is not None:
        survey = _surveys.check_survey(survey)
        _check_survey_arguments(survey, {
            "column_map": column_map, "band_aliases": band_aliases, "band_lookup": band_lookup,
            "default_band": default_band, "flux_unit": flux_unit, "magnitude_unit": magnitude_unit,
            "flag_filters": flag_filters, "normalize": normalize})
        if data_mode not in (None, "magnitude"):
            raise ValueError(f"data_mode={data_mode!r} cannot be combined with survey={survey!r}: "
                             f"the presets write AB magnitudes. Leave data_mode out.")
        df, survey_info = _surveys.survey_table(path, survey, limits=limits, delimiter=delimiter)
        cols = {c: c for c in _surveys.COLUMNS}
    else:
        if limits is not None:
            raise ValueError(
                "limits= is read by the survey presets only: pass survey='ztf' or survey='lsst', "
                "or put the limits in the table as rows with upper_limit=True.")
        df = _read_table(path, delimiter)
        cols = _resolve_columns(df.columns, column_map)
        survey_info = None
    source = (str(path) if isinstance(path, (str, Path)) else f"<{type(path).__name__}>")

    if "time" not in cols:
        raise ValueError(
            f"No time column found (looked for {CANONICAL_SYNONYMS['time']}). "
            f"Available columns: {list(df.columns)}. Pass column_map={{'time': '<column>'}}.")
    time = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy(dtype=float)
    time_converted = None
    is_jd = (survey_info is None and _norm(cols["time"]) in _JD_COLUMNS
             and np.isfinite(time).any() and np.nanmedian(time) > JD_MJD_OFFSET)
    if is_jd:
        # whisper's clock is MJD (explosion_date, time_min/time_max, the survey presets).
        time = time - JD_MJD_OFFSET
        time_converted = f"column {cols['time']!r} read as JD: time = JD - {JD_MJD_OFFSET}"

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
    if name is None and survey_info is not None:
        name = survey_info["name"]

    if normalize:
        band = normalize_bands(band, aliases=band_aliases)
    lookup = None
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
    # (a survey preset has written AB magnitudes itself, so there is no unit to guess)
    if magnitude is not None:
        magnitude = to_canonical(magnitude, magnitude_unit, "magnitude",
                                 warn_default=survey is None)
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
        meta={"source_file": source,
              "n_rows_raw": int(len(df) if survey_info is None else survey_info["n_rows_raw"]),
              "n_rows_kept": int(mask.sum())},
    )
    if time_converted:
        lc.meta["time_converted"] = time_converted

    if bands is not None:
        lc = _select_bands(lc, bands, survey=survey, normalize=normalize, aliases=band_aliases,
                           lookup=lookup)
    if time_min is not None or time_max is not None:
        lc = lc.select_time_window(time_min, time_max)
    if survey_info is None:
        if min_snr is not None:
            lc = lc.select_snr(min_snr)
    else:
        lc = _finish_survey(lc, survey_info, min_snr)
    if explosion_date is not None:
        lc = lc.set_explosion_date(explosion_date)
    return lc


def _finish_survey(lc, info, min_snr):
    """The survey preset's SNR cut (detections only), its meta, and the not-enough-data check."""
    dropped = dict(info["n_dropped"], low_snr=0)
    ul = np.asarray(lc.upper_limit, dtype=bool)
    if min_snr is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            keep = ul | (lc.snr >= min_snr)
        dropped["low_snr"] = int(np.sum(~keep))
        lc, ul = lc[keep], ul[keep]
    if not np.any(~ul):
        raise ValueError(
            f"not enough data: no detection is left in this {info['survey'].upper()} light curve "
            f"({int(ul.sum())} upper limit(s); {info['n_rows_raw']} rows read, dropped: "
            f"{ {k: v for k, v in dropped.items() if v} }). Widen time_min/time_max or bands=, "
            f"lower min_snr, or wait for more detections.")
    lc.meta.update(survey=info["survey"], photometry=info["photometry"],
                   time_system=info["time_system"], upper_limit_sigma=info["upper_limit_sigma"],
                   n_dropped=dropped,
                   first_detection_mjd=float(np.min(np.asarray(lc.time)[~ul])))
    return lc
