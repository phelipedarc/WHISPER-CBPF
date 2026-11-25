"""SVO Filter Profile Service fallback for bands missing from :data:`FILTER_LOOKUP`.

When a band cannot be resolved locally we ask the `SVO Filter Profile Service
<http://svo2.cab.inta-csic.es/theory/fps/>`_ for its effective wavelength and zero point, and for
its transmission curve, which :func:`whisper_cbpf.synphot.gauss_rule` integrates when a model is
bound to an SVO ID (``"LSST/LSST.g"``).

**Two clients, pyphot first.** With `pyphot <https://github.com/mfouesneau/pyphot>`_ installed
(``pip install pyphot``) both queries go through it: its SVO client and VOTable parser for the
metadata, :func:`pyphot.svo.get_pyphot_filter` for the curve. ``astroquery.svo_fps`` is the
fallback, used when pyphot is absent or fails. Either returns SVO's own numbers: ``WavelengthEff``
is SVO's Vega-weighted effective wavelength and ``ZeroPoint`` SVO's zero point in its published
system (Vega for most filters), recorded as metadata; whisper's magnitudes are AB throughout.

**Design rules (so this never crashes a load and never hits the network in CI):**

* All network access goes through three thin private wrappers
  (:func:`_svo_fetch_metadata`, :func:`_svo_fetch_index`, :func:`_svo_fetch_transmission`). Tests
  monkeypatch *those* -- nothing else touches the network, and ``pyphot`` and ``astroquery`` are
  imported lazily so the package installs and imports without either. ``_svo_fetch_index`` (the
  wavelength search) needs astroquery: pyphot has no index query.
* Results are cached **by filter ID** both in memory and on disk
  (``$WHISPER_SVO_CACHE`` or ``~/.cache/whisper_cbpf/svo_cache.json``; curves beside it in
  ``svo_curves/``), so re-runs are offline-safe and a repeated lookup never re-queries the service.
* Every failure path degrades gracefully: a network error / missing filter raises
  :class:`SvoUnavailable`, which callers turn into a warning plus a manual-override path
  (:func:`register_manual_band`) rather than an exception.

Filter IDs follow SVO's ``Facility/Instrument.Filter`` convention, e.g. ``'PAN-STARRS/PS1.r'`` or
``'2MASS/2MASS.J'``.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import astropy.units as u


class SvoUnavailable(RuntimeError):
    """Raised when SVO cannot resolve a band (astroquery missing, network down, unknown filter).

    :func:`whisper_cbpf.resolve_band` catches it and returns the band unresolved, with a warning
    naming :func:`register_manual_band`; catch it yourself around a direct SVO query.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> issubclass(wp.SvoUnavailable, RuntimeError)
    True
    """


#
# Documented default SVO filter IDs for effective bands we may meet outside the
# local LSST table. Used to disambiguate before falling back to a wavelength search.
#
DEFAULT_SVO_IDS = {
    "U-band": "SLOAN/SDSS.u", "u-band": "SLOAN/SDSS.u",
    "g-band": "PAN-STARRS/PS1.g", "r-band": "PAN-STARRS/PS1.r",
    "i-band": "PAN-STARRS/PS1.i", "z-band": "PAN-STARRS/PS1.z",
    "J-band": "2MASS/2MASS.J", "H-band": "2MASS/2MASS.H", "K-band": "2MASS/2MASS.Ks",
    "F356W-band": "JWST/NIRCam.F356W", "F444W-band": "JWST/NIRCam.F444W",
}

# Manual user overrides: band label -> {"lambda_eff": AA, "zero_point": Jy}.
_MANUAL_BANDS: dict = {}

# In-memory metadata cache: filter_id -> {"WavelengthEff": AA, "ZeroPoint": Jy, "filter_id": ...}.
_META_CACHE: dict = {}
# In-memory curve cache: filter_id -> (wavelength_AA, throughput); on disk in svo_curves/.
_CURVE_CACHE: dict = {}
_DISK_LOADED = False


#
# Disk cache (offline-safe across runs)
#
def _cache_path() -> Path:
    env = os.environ.get("WHISPER_SVO_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "whisper_cbpf" / "svo_cache.json"


def _valid_meta(entry):
    """A cache entry is trustworthy only if it is a dict with finite WavelengthEff + ZeroPoint."""
    return (isinstance(entry, dict)
            and isinstance(entry.get("WavelengthEff"), (int, float))
            and isinstance(entry.get("ZeroPoint"), (int, float))
            and bool(np.isfinite(entry["WavelengthEff"]))
            and bool(np.isfinite(entry["ZeroPoint"])))


def _load_disk_cache():
    global _DISK_LOADED
    if _DISK_LOADED:
        return
    _DISK_LOADED = True
    path = _cache_path()
    try:
        if path.exists():
            with open(path) as fh:
                data = json.load(fh)
            # Only merge well-formed dict entries -- a corrupt/partial cache must never break a load.
            if isinstance(data, dict):
                _META_CACHE.update({k: v for k, v in data.items() if _valid_meta(v)})
    except Exception as exc:  # unreadable / non-JSON cache must never break a load
        warnings.warn(f"Could not read SVO cache {path}: {exc}", stacklevel=2)


def _save_disk_cache():
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(_META_CACHE, fh, indent=0, sort_keys=True)
    except Exception as exc:  # pragma: no cover - best-effort persistence
        warnings.warn(f"Could not write SVO cache {path}: {exc}", stacklevel=2)


def clear_cache(disk=False):
    """Drop the in-memory caches (and the on-disk files if ``disk=True``). Mainly for tests."""
    _META_CACHE.clear()
    _CURVE_CACHE.clear()
    global _DISK_LOADED
    _DISK_LOADED = False
    if disk:
        try:
            _cache_path().unlink()
        except FileNotFoundError:
            pass
        curves = _cache_path().parent / "svo_curves"
        if curves.is_dir():
            for f in curves.glob("*.npz"):
                f.unlink()


#
# Thin network boundary -- THE ONLY functions that import pyphot / astroquery or hit SVO.
# Tests monkeypatch these.
#
def _pyphot():
    """``pyphot.svo`` if pyphot is installed, else ``None``."""
    try:
        from pyphot import svo as pyphot_svo
    except Exception:
        return None
    return pyphot_svo


def transmission_backend():
    """Which client fetches SVO curves here: ``"pyphot <version>"`` or ``"astroquery"``."""
    if _pyphot() is None:
        return "astroquery"
    import pyphot

    return f"pyphot {getattr(pyphot, '__VERSION__', '?')}"


def _svo() :
    """Lazily import ``astroquery.svo_fps.SvoFps`` (raises :class:`SvoUnavailable` if absent)."""
    try:
        from astroquery.svo_fps import SvoFps
    except Exception as exc:
        raise SvoUnavailable(
            "astroquery is not installed; cannot query the SVO Filter Profile Service. "
            "Install it (`pip install astroquery`) or supply the band manually via "
            "register_manual_band(band, lambda_eff, zero_point).") from exc
    return SvoFps


def _svo_fetch_metadata(filter_id):
    """Return ``{'WavelengthEff': <AA>, 'ZeroPoint': <Jy>, 'filter_id': ...}`` for one filter ID.

    There is no single-filter metadata call in astroquery (``get_filter_metadata`` never existed;
    ``fps.php?ID=`` returns the transmission curve, and its metadata lives in VOTable PARAMs that
    ``to_table()`` drops). The columns we need come from the per-facility filter list, which we then
    match on the exact ``filterID``. ``instrument`` is deliberately NOT passed: for several
    facilities the instrument segment of the ID is not an SVO instrument, and supplying it returns
    nothing at all (``SLOAN/SDSS``, ``PAN-STARRS/PS1`` and ``2MASS/2MASS`` all fail that way). The
    per-facility list is a few dozen rows and the result is cached by filter ID, so it is fetched once.

    With pyphot installed, the single-filter ``fps.php?ID=`` response is parsed with pyphot's
    VOTable reader instead, which keeps those PARAMs: one small request, the same SVO numbers.
    astroquery is the fallback if pyphot is absent or its query fails.
    """
    if "/" not in str(filter_id):
        raise SvoUnavailable(
            f"{filter_id!r} is not an SVO filter ID (expected 'Facility/Instrument.Filter').")
    pyphot_svo, pyphot_error = _pyphot(), None
    if pyphot_svo is not None:
        try:
            from io import BytesIO

            import requests
            from pyphot.io.votable import from_votable

            response = requests.get(pyphot_svo.QUERY_URL, params={"ID": str(filter_id)},
                                    timeout=60)
            response.raise_for_status()
            params = from_votable(BytesIO(response.content))[1].header
            return {
                "filter_id": filter_id,
                "WavelengthEff": float(params["WavelengthEff"]["value"]),   # Angstrom
                "ZeroPoint": float(params["ZeroPoint"]["value"]),           # Jy
            }
        except Exception as exc:        # unknown ID (no PARAMs), network, parser: try astroquery
            pyphot_error = exc
    try:
        SvoFps = _svo()
    except SvoUnavailable as exc:
        if pyphot_error is None:
            raise
        raise SvoUnavailable(f"SVO metadata query failed for {filter_id!r} through pyphot "
                             f"({pyphot_error}) and astroquery is not installed.") from exc
    facility = str(filter_id).split("/")[0]
    try:
        table = SvoFps.get_filter_list(facility=facility)
    except Exception as exc:
        raise SvoUnavailable(f"SVO metadata query failed for {filter_id!r}: {exc}") from exc
    rows = [r for r in table if str(r["filterID"]) == str(filter_id)] if table is not None else []
    if not rows:
        raise SvoUnavailable(f"SVO returned no metadata for filter {filter_id!r}.")
    row = rows[0]
    return {
        "filter_id": filter_id,
        "WavelengthEff": float(row["WavelengthEff"]),   # Angstrom
        "ZeroPoint": float(row["ZeroPoint"]),           # Jy
    }


def _svo_fetch_index(wl_min_aa, wl_max_aa):
    """Return a list of ``{'filterID', 'WavelengthEff', 'ZeroPoint'}`` dicts in a wavelength window."""
    SvoFps = _svo()
    try:
        table = SvoFps.get_filter_index(wl_min_aa * u.angstrom, wl_max_aa * u.angstrom)
    except Exception as exc:
        raise SvoUnavailable(
            f"SVO index query failed for {wl_min_aa}-{wl_max_aa} AA: {exc}") from exc
    out = []
    for row in table:
        out.append({
            "filterID": str(row["filterID"]),
            "WavelengthEff": float(row["WavelengthEff"]),
            "ZeroPoint": float(row["ZeroPoint"]) if "ZeroPoint" in table.colnames else np.nan,
        })
    return out


def _svo_fetch_transmission(filter_id):
    """Return the transmission curve as ``(wavelength_AA, throughput)`` arrays.

    pyphot's ``get_pyphot_filter`` when installed (it sorts the curve and clips negative
    throughput to zero), else astroquery's ``get_transmission_data``.
    """
    pyphot_svo, pyphot_error = _pyphot(), None
    if pyphot_svo is not None:
        try:
            filt = pyphot_svo.get_pyphot_filter(str(filter_id))
            return (np.asarray(filt.wavelength.to("AA").value, dtype=float),
                    np.asarray(filt.transmit, dtype=float))
        except Exception as exc:
            pyphot_error = exc
    try:
        SvoFps = _svo()
    except SvoUnavailable as exc:
        if pyphot_error is None:
            raise
        raise SvoUnavailable(f"SVO transmission query failed for {filter_id!r} through pyphot "
                             f"({pyphot_error}) and astroquery is not installed.") from exc
    try:
        table = SvoFps.get_transmission_data(filter_id)
    except Exception as exc:
        raise SvoUnavailable(f"SVO transmission query failed for {filter_id!r}: {exc}") from exc
    return (np.asarray(table["Wavelength"], dtype=float),
            np.asarray(table["Transmission"], dtype=float))


#
# Public, cached, mock-friendly API
#
def register_manual_band(band, lambda_eff, zero_point):
    """Manually supply ``lambda_eff`` (Angstrom) + ``zero_point`` (Jy) for ``band``.

    Use this when SVO is unavailable or the automatic filter mapping is wrong. Takes precedence over
    every SVO lookup.

    .. note::
       This is a **process-global** registration: it persists for the rest of the interpreter session
       and affects *all* subsequent band resolutions. Undo it with :func:`unregister_manual_band` (one
       band) or :func:`clear_manual_bands` (all).

    Parameters
    ----------
    band : str
        The band label.
    lambda_eff : float
        Effective wavelength [Angstrom].
    zero_point : float
        Zero point [Jy] (3631 for AB).

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.register_manual_band("my_halpha", 6563.0, 3631.0)
    >>> wp.resolve_band("my_halpha")["lambda_eff"], wp.resolve_band("my_halpha")["source"]
    (6563.0, 'manual')
    >>> wp.unregister_manual_band("my_halpha")
    """
    _MANUAL_BANDS[str(band)] = {
        "lambda_eff": float(lambda_eff), "zero_point": float(zero_point)}


def unregister_manual_band(band):
    """Remove a single manual band override (no-op if it was not registered).

    Parameters
    ----------
    band : str
        The label passed to :func:`register_manual_band`.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.register_manual_band("my_nb", 6563.0, 3631.0)
    >>> wp.unregister_manual_band("my_nb")
    >>> wp.resolve_band("my_nb", svo_fallback=False, warn=False)["source"]
    'unresolved'
    """
    _MANUAL_BANDS.pop(str(band), None)


def clear_manual_bands():
    """Remove all manual band overrides registered via :func:`register_manual_band`.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.register_manual_band("my_nb", 6563.0, 3631.0)
    >>> wp.clear_manual_bands()
    >>> wp.resolve_band("my_nb", svo_fallback=False, warn=False)["lambda_eff"] is None
    True
    """
    _MANUAL_BANDS.clear()


def get_filter_metadata(filter_id, *, use_cache=True):
    """Effective wavelength + zero point for an SVO ``filter_id`` (cached by ID; offline-safe)."""
    _load_disk_cache()
    if use_cache and _valid_meta(_META_CACHE.get(filter_id)):
        return dict(_META_CACHE[filter_id])
    meta = _svo_fetch_metadata(filter_id)
    if not _valid_meta(meta):   # never cache (or trust) non-finite wavelength/zero point
        raise SvoUnavailable(
            f"SVO returned unusable metadata for {filter_id!r}: {meta}")
    _META_CACHE[filter_id] = meta
    _save_disk_cache()
    return dict(meta)


def _curve_path(filter_id):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(filter_id))
    return _cache_path().parent / "svo_curves" / f"{safe}.npz"


def get_transmission_data(filter_id, *, use_cache=True):
    """Transmission curve ``(wavelength_AA, throughput)`` for ``filter_id``. Cached by ID.

    What :func:`whisper_cbpf.synphot.gauss_rule` integrates for an SVO ID. The curve is kept in
    memory and in ``svo_curves/`` beside the metadata cache, so a model bound to an SVO filter
    rebuilds offline. ``use_cache=False`` forces a fresh fetch (e.g. after SVO revises a filter).
    """
    key = str(filter_id)
    if use_cache and key in _CURVE_CACHE:
        return tuple(a.copy() for a in _CURVE_CACHE[key])
    path = _curve_path(key)
    if use_cache and path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                curve = (np.asarray(data["wave"], dtype=float), np.asarray(data["trans"], dtype=float))
            _CURVE_CACHE[key] = curve
            return tuple(a.copy() for a in curve)
        except Exception as exc:        # a corrupt cache file must never break a fit
            warnings.warn(f"Could not read cached SVO curve {path}: {exc}", stacklevel=2)
    wave, trans = _svo_fetch_transmission(key)
    wave, trans = np.asarray(wave, dtype=float), np.asarray(trans, dtype=float)
    if wave.ndim != 1 or wave.shape != trans.shape or wave.size < 2 or \
            not np.all(np.isfinite(wave)) or not np.all(np.isfinite(trans)):
        raise SvoUnavailable(f"SVO returned an unusable transmission curve for {key!r}")
    _CURVE_CACHE[key] = (wave, trans)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, wave=wave, trans=trans)
    except Exception as exc:  # pragma: no cover - best-effort persistence
        warnings.warn(f"Could not write SVO curve cache {path}: {exc}", stacklevel=2)
    return wave.copy(), trans.copy()


def find_filter_id(band, *, lambda_eff_hint=None, tol_frac=0.05):
    """Resolve a band label to a single SVO filter ID.

    Priority: documented default (:data:`DEFAULT_SVO_IDS`) > wavelength search around
    ``lambda_eff_hint``. The wavelength search looks within ``±tol_frac`` (default 5%) of the hint;
    widen it for more candidates (and more ambiguity), narrow it for fewer misses. When a search yields
    several candidates the choice is ambiguous: we warn, list the candidates, and return the
    closest-in-wavelength one rather than failing silently. Raises :class:`SvoUnavailable` if nothing
    matches.
    """
    key = str(band).strip()
    if "/" in key and "." in key:        # already a 'Facility/Instrument.Filter' SVO ID
        return key
    if key in DEFAULT_SVO_IDS:
        return DEFAULT_SVO_IDS[key]
    if lambda_eff_hint is None:
        raise SvoUnavailable(
            f"No documented SVO filter ID for band {key!r} and no wavelength hint to search with. "
            "Pass register_manual_band(...) or a lambda_eff hint.")
    lo = lambda_eff_hint * (1 - tol_frac)
    hi = lambda_eff_hint * (1 + tol_frac)
    candidates = _svo_fetch_index(lo, hi)
    if not candidates:
        raise SvoUnavailable(
            f"SVO returned no filters near {lambda_eff_hint} AA for band {key!r}.")
    candidates.sort(key=lambda c: abs(c["WavelengthEff"] - lambda_eff_hint))
    chosen = candidates[0]["filterID"]
    if len(candidates) > 1:
        warnings.warn(
            f"Band {key!r} ambiguously matches {len(candidates)} SVO filters near "
            f"{lambda_eff_hint} AA: {[c['filterID'] for c in candidates[:6]]}. "
            f"Using closest match {chosen!r}; override with register_manual_band(...).",
            stacklevel=2)
    return chosen


def resolve_band_svo(band, *, lambda_eff_hint=None):
    """Resolve ``band`` to ``{'lambda_eff', 'zero_point', 'filter_id', 'source'}`` via SVO.

    Honours :func:`register_manual_band` overrides first. Raises :class:`SvoUnavailable` on any
    failure so the caller can warn and offer the manual-override path.
    """
    key = str(band).strip()
    if key in _MANUAL_BANDS:
        m = _MANUAL_BANDS[key]
        return {"lambda_eff": m["lambda_eff"], "zero_point": m["zero_point"],
                "filter_id": None, "source": "manual"}
    filter_id = find_filter_id(key, lambda_eff_hint=lambda_eff_hint)
    meta = get_filter_metadata(filter_id)
    return {"lambda_eff": meta["WavelengthEff"], "zero_point": meta["ZeroPoint"],
            "filter_id": filter_id, "source": "svo"}
