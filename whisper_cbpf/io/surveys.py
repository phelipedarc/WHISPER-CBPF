"""Survey ingest presets: ZTF and LSST alert photometry as one canonical table.

``load_lightcurve(x, survey="ztf" | "lsst")`` calls :func:`survey_table`, which reads the brokers'
own field names and applies the survey's photometry rule; the loader then treats the result like any
other table (quality cuts, band resolution, selections).

The photometry is difference-imaging PSF photometry, the measurement an alert is made of:

* **ZTF**: ``magpsf`` / ``sigmapsf``, the PSF magnitude of the flux on the difference image, where
  ``isdiffpos`` says the difference is positive (``'t'``, ``'1'``, ``1``). On a negative difference
  (``'f'``, ``'0'``, ``-1``) ``magpsf`` is the magnitude of ``|flux|``, not a magnitude of the
  transient, so the row is dropped and counted. ``magpsf_corr`` (the reference image's flux added
  back, a total magnitude for variable stars) is never read. A non-detection (no ``magpsf``) becomes a
  5-sigma upper limit at its ``diffmaglim``.
* **LSST**: diaSource ``psfFlux`` / ``psfFluxErr`` in nJy, as ``m = 31.4 - 2.5 log10(psfFlux)`` and
  ``sigma_m = (2.5 / ln 10) psfFluxErr / psfFlux``. A negative flux (``isNegative``, or
  ``psfFlux <= 0``) has no AB magnitude and a failed fit (``psfFlux_flag``) no value, so both are
  dropped and counted. ``scienceFlux`` (the direct image, host included) is never read. A forced-
  photometry epoch without a diaSource becomes a 5-sigma upper limit, ``31.4 - 2.5 log10(5
  psfFluxErr)``; a withdrawn row (``timeWithdrawnMjdTai`` set) is dropped.

An upper limit at the epoch of a detection (the same visit, or within :data:`SAME_EPOCH_DAYS`) is that
exposure's detection, not a non-detection, and is dropped. Repeats of one exposure (the same time,
band and kind, as when alerts of one object are concatenated) are kept once.

Inputs, for ``x`` and ``limits=``: a path (``.csv`` or ``.json``), a DataFrame, one record or alert,
or a list of them. Field names follow the ZTF alert packet and ALeRCE (``jd`` or ``mjd``, ``fid``,
``magpsf``, ``sigmapsf``, ``isdiffpos``, ``diffmaglim``) and the Rubin alert schema
(``midpointMjdTai``, ``band``, ``psfFlux``, ``psfFluxErr``, ``isNegative``, ``visit``), matched
case-insensitively and with Fink's column prefixes (``i:``, ``d:``, ``r:``) removed; Fink's ZTF
``d:tag == "badquality"`` rows are dropped. Whole packets are unpacked: a ZTF alert (``candidate`` +
``prv_candidates``), a Rubin alert (``diaSource`` + ``prvDiaSources`` + ``prvDiaForcedSources``) and
ALeRCE's ``{"detections": [...], "non_detections": [...]}``. A table in whisper's own columns
(``time``, ``band``, ``magnitude``, ``magnitude_err``, ``upper_limit``) is read as it is, with its
bands mapped onto the survey's labels (:data:`~whisper_cbpf.io.bands.SURVEY_BANDS`).

Examples
--------
>>> from whisper_cbpf.io import surveys
>>> surveys.survey_band("zg", "ztf"), surveys.survey_band("y", "lsst")
('ztfg', 'lssty')
>>> src = [{"midpointMjdTai": 61000.0, "band": "g", "psfFlux": 1000.0, "psfFluxErr": 100.0}]
>>> table, info = surveys.survey_table(src, "lsst")
>>> table[["time", "band", "magnitude", "upper_limit"]].to_dict("records")
[{'time': 61000.0, 'band': 'lsstg', 'magnitude': 23.9, 'upper_limit': False}]
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .bands import SURVEY_BANDS, normalize_band
from .photometry import POGSON

#: AB magnitude of a flux density of 1 nJy, the zero point of Rubin's fluxes.
LSST_ZP_NJY = 31.4
#: Significance of the upper limits the presets write: ZTF's ``diffmaglim`` and the LSST forced
#: depth are both 5 sigma. Recorded as ``lc.meta["upper_limit_sigma"]``.
LIMIT_SIGMA = 5.0
#: Two rows closer in time than this (days, 8.6 s) are one exposure: consecutive ZTF or LSST
#: exposures are at least ~40 s apart.
SAME_EPOCH_DAYS = 1e-4
#: Julian date minus modified Julian date.
MJD_OFFSET = 2400000.5
#: ZTF filter ids (``fid``) as whisper band labels.
ZTF_FID = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

#: What each preset reads, as recorded in ``lc.meta``.
SURVEYS = {
    "ztf": {"name": "ZTF",
            "photometry": "ZTF difference-image PSF magnitudes (magpsf/sigmapsf), positive "
                          "subtractions (isdiffpos) only; magpsf_corr is not used",
            "limits": "5-sigma diffmaglim of the non-detections",
            "time_system": "MJD (UTC)"},
    "lsst": {"name": "LSST",
             "photometry": "LSST diaSource difference-image PSF flux as AB magnitude, "
                           "m = 31.4 - 2.5 log10(psfFlux / nJy); scienceFlux is not used",
             "limits": "5-sigma depth of the forced photometry, 31.4 - 2.5 log10(5 psfFluxErr)",
             "time_system": "MJD (TAI, midpointMjdTai)"},
}

#: The columns :func:`survey_table` returns, in whisper's own names.
COLUMNS = ("time", "band", "magnitude", "magnitude_err", "upper_limit")
#: Why a row was dropped: the keys of ``lc.meta["n_dropped"]`` (the loader adds ``"low_snr"``).
DROP_REASONS = ("negative_flux", "bad_quality", "flagged", "withdrawn", "no_measurement",
                "limit_at_detection", "duplicate")

_PREFIXES = ("i:", "d:", "r:", "v:", "b:")          # Fink's column families
_BAND_FIELDS = ("band", "filtername", "filter", "filtercode", "passband")
_NAME_FIELDS = ("objectid", "diaobjectid", "oid")
_TRUE = {"1", "1.0", "t", "true", "yes", "y"}
_FALSE_SIGN = {"f", "false", "0", "0.0", "-1", "-1.0"}


def check_survey(survey):
    """The preset key for ``survey`` (``"ztf"`` or ``"lsst"``, any case). Raises ``ValueError``.

    Examples
    --------
    >>> from whisper_cbpf.io.surveys import check_survey
    >>> check_survey("LSST")
    'lsst'
    """
    key = str(survey).strip().lower()
    if key not in SURVEYS:
        raise ValueError(
            f"survey={survey!r} is not a preset; use one of {tuple(SURVEYS)}, or survey=None for "
            f"a table in whisper's own columns (time, band, magnitude or flux, errors).")
    return key


def survey_band(label, survey):
    """The survey band label that ``label`` stands for in ``survey``.

    A survey label passes through (``'lsstg'``); a bare lower-case letter takes the survey's prefix
    (``'g'`` -> ``'lsstg'`` or ``'ztfg'``); a survey code is normalised (``'zg'``, ``'ZTF_g'`` ->
    ``'ztfg'``).

    Parameters
    ----------
    label : str
        The band as the data or the caller spells it.
    survey : {"ztf", "lsst"}
        The survey preset.

    Returns
    -------
    str
        One of ``SURVEY_BANDS[survey]``.

    Raises
    ------
    ValueError
        When ``label`` is not one of the survey's bands (``'y'`` in ZTF, a Cousins ``'R'``).

    Examples
    --------
    >>> from whisper_cbpf.io.surveys import survey_band
    >>> survey_band("r", "lsst"), survey_band("ZTF_i", "ztf")
    ('lsstr', 'ztfi')
    """
    key = check_survey(survey)
    bands = SURVEY_BANDS[key]
    raw = str(label).strip()
    for cand in (raw.lower(), normalize_band(raw), key + raw):
        if cand in bands:
            return cand
    raise ValueError(
        f"band {raw!r} is not one of the {SURVEYS[key]['name']} bands: {', '.join(bands)} (a bare "
        f"lower-case letter or a survey code such as 'zg' maps onto them). Select only these, or "
        f"load another survey's photometry without survey=.")


def survey_table(x, survey, *, limits=None, delimiter=None):
    """Alert photometry of ``survey`` as a table in whisper's columns, plus what was done to it.

    The engine behind ``load_lightcurve(x, survey=...)``; the module docstring gives the rules.

    Parameters
    ----------
    x : str, pathlib.Path, pandas.DataFrame, dict or list of dict
        The detections: a ``.csv`` or ``.json`` file, a table, one record or alert packet, or a list.
        Rubin alert packets also carry their forced photometry.
    survey : {"ztf", "lsst"}
        The preset.
    limits : same types as ``x``, optional
        Rows that become upper limits: ZTF non-detections (``diffmaglim``), or LSST forced
        photometry (``psfFluxErr``), e.g. Fink's ``/fp`` output.
    delimiter : str, optional
        CSV delimiter; sniffed when omitted.

    Returns
    -------
    table : pandas.DataFrame
        Columns :data:`COLUMNS`, sorted by time. Upper limits carry the limiting magnitude and a NaN
        error.
    info : dict
        ``survey``, ``name`` (the object id, when the records carry one), ``n_rows_raw``,
        ``n_dropped`` (a count per :data:`DROP_REASONS`), ``photometry``, ``limits``,
        ``time_system`` and ``upper_limit_sigma``.

    Raises
    ------
    ValueError
        On an unknown survey or band, on ``magpsf_corr`` or ``scienceFlux`` without the
        difference-image field, and on records with no time, band or photometry field.

    Examples
    --------
    >>> from whisper_cbpf.io.surveys import survey_table
    >>> alerts = {"detections": [{"mjd": 60001.0, "fid": 2, "magpsf": 19.0, "sigmapsf": 0.1,
    ...                           "isdiffpos": 1}],
    ...           "non_detections": [{"mjd": 59999.0, "fid": 2, "diffmaglim": 20.5}]}
    >>> table, info = survey_table(alerts, "ztf")
    >>> table.to_dict("records")  # doctest: +NORMALIZE_WHITESPACE
    [{'time': 59999.0, 'band': 'ztfr', 'magnitude': 20.5, 'magnitude_err': nan, 'upper_limit': True},
     {'time': 60001.0, 'band': 'ztfr', 'magnitude': 19.0, 'magnitude_err': 0.1, 'upper_limit': False}]
    >>> info["upper_limit_sigma"], info["n_dropped"]["limit_at_detection"]
    (5.0, 0)
    """
    key = check_survey(survey)
    main, forced, name = _collect(x, delimiter)
    extra = []
    if limits is not None:
        lmain, lforced, _ = _collect(limits, delimiter)
        extra = [f for f in (lmain, lforced) if f is not None]
    n_raw = sum(len(f) for f in (main, forced, *extra) if f is not None)
    if main is None or not len(main):
        raise ValueError(f"no {SURVEYS[key]['name']} records to read in {_describe(x)}.")

    if key == "ztf":
        parts = [_ztf_rows(f) for f in (main, *extra) if len(f)]
    else:
        parts = [_lsst_sources(main)] + [_lsst_forced(f) for f in (forced, *extra)
                                         if f is not None and len(f)]
    rows = pd.concat(parts, ignore_index=True)

    ul = rows["upper_limit"].to_numpy(dtype=bool)
    det_t = rows["time"].to_numpy(dtype=float)[~ul]
    det_v = rows["visit"].to_numpy(dtype=float)[~ul]
    t, v = rows["time"].to_numpy(dtype=float), rows["visit"].to_numpy(dtype=float)
    same = _near(t, det_t) | (np.isfinite(v) & np.isin(v, det_v[np.isfinite(det_v)]))
    rows.loc[ul & same & (rows["drop"] == ""), "drop"] = "limit_at_detection"

    counts = dict.fromkeys(DROP_REASONS, 0)
    for reason, n in rows["drop"].value_counts().items():
        if reason:
            counts[reason] = int(n)
    kept = rows[rows["drop"] == ""]
    dup = kept.duplicated(subset=["time", "band", "upper_limit"])
    counts["duplicate"] = int(dup.sum())
    kept = kept[~dup].sort_values("time", kind="stable")

    if counts["negative_flux"]:
        warnings.warn(
            f"{counts['negative_flux']} {SURVEYS[key]['name']} detection(s) with a negative "
            f"difference flux have no AB magnitude and were dropped (lc.meta['n_dropped']).",
            UserWarning, stacklevel=3)
    if name is None:
        name = _first_name(main)
    info = {"survey": key, "name": None if name is None else str(name), "n_rows_raw": int(n_raw),
            "n_dropped": counts, "photometry": SURVEYS[key]["photometry"],
            "limits": SURVEYS[key]["limits"], "time_system": SURVEYS[key]["time_system"],
            "upper_limit_sigma": LIMIT_SIGMA}
    return kept[list(COLUMNS)].reset_index(drop=True), info


# --- reading -----------------------------------------------------------------------------------

def _describe(x):
    return str(x) if isinstance(x, (str, Path)) else f"the {type(x).__name__} passed"


def _collect(x, delimiter):
    """``x`` as (main rows, forced-photometry rows or None, object id or None), unpacking packets."""
    if isinstance(x, (str, Path)):
        path = Path(x)
        suffix = path.suffix.lower()
        if suffix == ".avro":
            raise ValueError(
                f"{path} is an Avro alert packet; read it with fastavro "
                f"(list(fastavro.reader(open(path, 'rb')))) and pass the records.")
        if suffix != ".json":
            if delimiter is not None:
                return pd.read_csv(path, delimiter=delimiter), None, None
            return pd.read_csv(path, sep=None, engine="python"), None, None
        x = json.loads(path.read_text())
    if hasattr(x, "read"):                                                # an open CSV
        return pd.read_csv(x, sep=delimiter, engine="python"), None, None
    if isinstance(x, pd.DataFrame):
        return x.copy(), None, None
    if isinstance(x, dict):
        x = [x]
    rows, forced, name = [], [], None
    for item in x:
        if not isinstance(item, dict):
            raise TypeError(f"alert records must be dicts, got {type(item).__name__}.")
        if "detections" in item or "non_detections" in item:            # ALeRCE light curve
            rows += list(item.get("detections") or []) + list(item.get("non_detections") or [])
        elif "candidate" in item:                                         # ZTF alert packet
            rows += [item["candidate"]] + list(item.get("prv_candidates") or [])
            name = name if name is not None else item.get("objectId")
        elif "diaSource" in item:                                         # Rubin alert packet
            rows += [item["diaSource"]] + list(item.get("prvDiaSources") or [])
            forced += list(item.get("prvDiaForcedSources") or [])
            obj = item.get("diaObject") or {}
            if name is None:
                name = obj.get("diaObjectId", item["diaSource"].get("diaObjectId"))
        else:
            rows.append(item)
    return (pd.DataFrame(rows) if rows else None), (pd.DataFrame(forced) if forced else None), name


def _index(df):
    """``{field: column}``: lower case, Fink's family prefix (``i:``, ``r:``, ...) removed."""
    out = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key[:2] in _PREFIXES:
            key = key[2:]
        out.setdefault(key, c)
    return out


def _num(df, idx, *names):
    for n in names:
        if n in idx:
            return pd.to_numeric(df[idx[n]], errors="coerce").to_numpy(dtype=float)
    return None


def _truthy(v):
    if v is None or isinstance(v, float) and np.isnan(v):
        return False
    return str(v).strip().lower() in _TRUE


def _flag(df, idx, *names):
    for n in names:
        if n in idx:
            return np.array([_truthy(v) for v in df[idx[n]]], dtype=bool)
    return np.zeros(len(df), dtype=bool)


def _time(df, idx, survey):
    t = _num(df, idx, "midpointmjdtai") if survey == "lsst" else None
    if t is None:
        t = _num(df, idx, "mjd")
    if t is None and "jd" in idx:
        t = _num(df, idx, "jd") - MJD_OFFSET
    if t is None:
        t = _num(df, idx, "time")
    if t is None:
        need = ("midpointMjdTai, mjd, jd" if survey == "lsst" else "mjd, jd") + " or time"
        raise ValueError(f"no time field in the {SURVEYS[survey]['name']} records (looked for "
                         f"{need}); columns: {list(df.columns)}.")
    return t


def _bands(df, idx, survey):
    if survey == "ztf" and "fid" in idx:
        fid = _num(df, idx, "fid")
        bad = sorted({f"{v:g}" for v in fid if not (np.isfinite(v) and int(v) in ZTF_FID)})
        if bad:
            raise ValueError(f"ZTF filter id(s) fid={', '.join(bad)} are not g (1), r (2) or "
                             f"i (3); drop those rows before loading.")
        return np.array([ZTF_FID[int(v)] for v in fid], dtype=object)
    for n in _BAND_FIELDS:
        if n in idx:
            raw = [str(b) for b in df[idx[n]]]
            memo = {b: survey_band(b, survey) for b in dict.fromkeys(raw)}
            return np.array([memo[b] for b in raw], dtype=object)
    need = "fid, band or filter" if survey == "ztf" else "band or filter"
    raise ValueError(f"no band field in the {SURVEYS[survey]['name']} records (looked for "
                     f"{need}); columns: {list(df.columns)}.")


def _own_columns(df, idx):
    """Magnitude, error and upper-limit flag from whisper's own column names (None: no magnitude)."""
    from .loader import CANONICAL_SYNONYMS

    mag = _num(df, idx, *CANONICAL_SYNONYMS["magnitude"])
    if mag is None:
        return None
    err = _num(df, idx, *CANONICAL_SYNONYMS["magnitude_err"])
    err = np.full(len(df), np.nan) if err is None else err
    return mag, err, _flag(df, idx, *CANONICAL_SYNONYMS["upper_limit"])


def _no_photometry(df, survey, looked_for):
    return ValueError(
        f"no {SURVEYS[survey]['name']} photometry field (looked for {looked_for}, or whisper's "
        f"own 'magnitude' / 'magnitude_err' / 'upper_limit'); columns: {list(df.columns)}.")


def _withdrawn(df, idx):
    w = _num(df, idx, "timewithdrawnmjdtai")
    return np.zeros(len(df), dtype=bool) if w is None else np.isfinite(w)


def _rows(time, band, mag, err, ul, drop, visit=None):
    return pd.DataFrame({"time": time, "band": band, "magnitude": mag,
                         "magnitude_err": np.where(ul, np.nan, err), "upper_limit": ul,
                         "drop": drop,
                         "visit": np.full(len(time), np.nan) if visit is None else visit})


def _first_name(df):
    idx = _index(df)
    for n in _NAME_FIELDS:
        if n in idx:
            vals = df[idx[n]].dropna()
            if len(vals):
                return vals.iloc[0]
    return None


# --- ZTF ---------------------------------------------------------------------------------------

def _isdiffpos(df, idx):
    """+1 / -1 per row from ``isdiffpos`` (NaN where empty); None when the field is absent."""
    if "isdiffpos" not in idx:
        return None
    out = np.full(len(df), np.nan)
    for i, v in enumerate(df[idx["isdiffpos"]]):
        s = str(v).strip().lower()
        out[i] = 1.0 if s in _TRUE else (-1.0 if s in _FALSE_SIGN else np.nan)
    return out


def _ztf_rows(df):
    idx = _index(df)
    n = len(df)
    time, band = _time(df, idx, "ztf"), _bands(df, idx, "ztf")
    mag, err = _num(df, idx, "magpsf"), _num(df, idx, "sigmapsf")
    if mag is None:
        if "magpsf_corr" in idx:
            raise ValueError(
                "these ZTF records carry magpsf_corr but not magpsf. magpsf_corr adds the "
                "reference image's flux back (a total magnitude, for variable stars), so it is not "
                "the transient's flux; whisper fits the difference-image magnitude magpsf / "
                "sigmapsf. Export those fields from the broker (ALeRCE and Fink both provide them).")
        own = _own_columns(df, idx)
        if own is None and "diffmaglim" not in idx:
            raise _no_photometry(df, "ztf", "magpsf/sigmapsf or diffmaglim")
        nan = np.full(n, np.nan)
        mag, err, ul = own if own is not None else (nan, nan, np.zeros(n, dtype=bool))
        sign = np.ones(n)
    else:
        err = np.full(n, np.nan) if err is None else err
        ul = np.zeros(n, dtype=bool)
        sign = _isdiffpos(df, idx)
        det = np.isfinite(mag)
        if sign is None or np.any(det & np.isnan(sign)):
            warnings.warn(
                "ZTF detections without isdiffpos: a negative subtraction cannot be told apart, "
                "so those rows are taken as positive. Include isdiffpos to drop them.",
                UserWarning, stacklevel=4)
            sign = np.ones(n) if sign is None else np.where(np.isnan(sign), 1.0, sign)
    diffmaglim = _num(df, idx, "diffmaglim")
    limit = np.where(ul, mag, np.nan if diffmaglim is None else diffmaglim)
    det = np.isfinite(mag) & ~ul
    lim = ~det & np.isfinite(limit)
    tag = (df[idx["tag"]].astype(str).str.strip().str.lower().to_numpy() if "tag" in idx
           else np.full(n, ""))

    drop = np.full(n, "", dtype=object)
    drop[~det & ~lim] = "no_measurement"
    drop[det & ~(np.isfinite(err) & (err > 0))] = "no_measurement"
    drop[det & (tag == "badquality")] = "bad_quality"
    drop[det & (sign < 0)] = "negative_flux"
    return _rows(time, band, np.where(det, mag, limit), err, ~det, drop)


# --- LSST --------------------------------------------------------------------------------------

def _lsst_sources(df):
    idx = _index(df)
    n = len(df)
    if "diaforcedsourceid" in idx and "diasourceid" not in idx:
        raise ValueError(
            "these LSST records are forced photometry (diaForcedSourceId), not detections: pass "
            "the diaSources as the first argument and the forced photometry as limits=.")
    time, band = _time(df, idx, "lsst"), _bands(df, idx, "lsst")
    visit = _num(df, idx, "visit")
    flux, ferr = _num(df, idx, "psfflux"), _num(df, idx, "psffluxerr")
    drop = np.full(n, "", dtype=object)
    if flux is None:
        if "scienceflux" in idx:
            raise ValueError(
                "these LSST records carry scienceFlux but not psfFlux. scienceFlux is measured on "
                "the direct image, host galaxy included, so it is not the transient's flux; "
                "whisper fits the difference-image flux psfFlux / psfFluxErr (nJy). Request those "
                "diaSource fields from the broker (Fink: r:psfFlux, r:psfFluxErr).")
        own = _own_columns(df, idx)
        if own is None:
            raise _no_photometry(df, "lsst", "psfFlux/psfFluxErr")
        mag, err, ul = own
        ok = np.isfinite(mag) & (ul | (np.isfinite(err) & (err > 0)))
        drop[~ok] = "no_measurement"
        return _rows(time, band, mag, err, ul, drop, visit)
    if ferr is None:
        raise ValueError(f"these LSST records carry psfFlux but not psfFluxErr; columns: "
                         f"{list(df.columns)}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = LSST_ZP_NJY - 2.5 * np.log10(flux)
        err = POGSON * ferr / flux
    drop[_flag(df, idx, "psfflux_flag")] = "flagged"
    drop[_flag(df, idx, "isnegative") | (np.isfinite(flux) & ~(flux > 0))] = "negative_flux"
    drop[~np.isfinite(flux) | ~(np.isfinite(ferr) & (ferr > 0))] = "no_measurement"
    drop[_withdrawn(df, idx)] = "withdrawn"
    return _rows(time, band, mag, err, np.zeros(n, dtype=bool), drop, visit)


def _lsst_forced(df):
    idx = _index(df)
    n = len(df)
    time, band = _time(df, idx, "lsst"), _bands(df, idx, "lsst")
    ferr = _num(df, idx, "psffluxerr")
    if ferr is None:
        raise ValueError(
            f"LSST limits= must be forced photometry (diaForcedSource rows with psfFluxErr, e.g. "
            f"Fink's /fp output); columns: {list(df.columns)}.")
    ok = np.isfinite(ferr) & (ferr > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        limit = LSST_ZP_NJY - 2.5 * np.log10(LIMIT_SIGMA * ferr)
    drop = np.full(n, "", dtype=object)
    drop[~ok] = "no_measurement"
    drop[_withdrawn(df, idx)] = "withdrawn"
    return _rows(time, band, limit, np.full(n, np.nan), np.ones(n, dtype=bool), drop,
                 _num(df, idx, "visit"))


def _near(t, ref, tol=SAME_EPOCH_DAYS):
    """Whether each ``t`` lies within ``tol`` of any ``ref``."""
    ref = np.sort(ref[np.isfinite(ref)])
    if ref.size == 0:
        return np.zeros(np.shape(t), dtype=bool)
    i = np.clip(np.searchsorted(ref, t), 1, ref.size) - 1
    j = np.clip(i + 1, 0, ref.size - 1)
    return (np.abs(t - ref[i]) < tol) | (np.abs(t - ref[j]) < tol)
