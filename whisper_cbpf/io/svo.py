"""SVO Filter Profile Service fallback for bands missing from :data:`FILTER_LOOKUP`.

When a band cannot be resolved locally we ask the `SVO Filter Profile Service
<http://svo2.cab.inta-csic.es/theory/fps/>`_ (via ``astroquery.svo_fps.SvoFps``) for its effective
wavelength and zero point, and -- later, for Phase-2 spectral integration -- its transmission curve.

**Design rules (so this never crashes a load and never hits the network in CI):**

* All astroquery/network access goes through three thin private wrappers
  (:func:`_svo_fetch_metadata`, :func:`_svo_fetch_index`, :func:`_svo_fetch_transmission`). Tests
  monkeypatch *those* -- nothing else touches the network, and ``astroquery`` is imported lazily so
  the package installs and imports without it.
* Results are cached **by filter ID** both in memory and on disk
  (``$WHISPER_SVO_CACHE`` or ``~/.cache/whisper_labia/svo_cache.json``), so re-runs are offline-safe
  and a repeated lookup never re-queries the service.
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
    """Raised when SVO cannot resolve a band (astroquery missing, network down, unknown filter)."""


# ---------------------------------------------------------------------------
# Documented default SVO filter IDs for effective bands we may meet outside the
# local LSST table. Used to disambiguate before falling back to a wavelength search.
# ---------------------------------------------------------------------------
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
_DISK_LOADED = False


# ---------------------------------------------------------------------------
# Disk cache (offline-safe across runs)
# ---------------------------------------------------------------------------
def _cache_path() -> Path:
    env = os.environ.get("WHISPER_SVO_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "whisper_labia" / "svo_cache.json"


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
    """Drop the in-memory cache (and the on-disk file if ``disk=True``). Mainly for tests."""
    _META_CACHE.clear()
    global _DISK_LOADED
    _DISK_LOADED = False
    if disk:
        try:
            _cache_path().unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Thin network boundary -- THE ONLY functions that import astroquery / hit SVO.
# Tests monkeypatch these.
# ---------------------------------------------------------------------------
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
    """Return ``{'WavelengthEff': <AA>, 'ZeroPoint': <Jy>, 'filter_id': ...}`` for one filter ID."""
    SvoFps = _svo()
    try:
        table = SvoFps.get_filter_metadata(filter_id)
    except Exception as exc:
        raise SvoUnavailable(f"SVO metadata query failed for {filter_id!r}: {exc}") from exc
    if table is None or len(table) == 0:
        raise SvoUnavailable(f"SVO returned no metadata for filter {filter_id!r}.")
    row = table[0]
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
    """Return the transmission curve as ``(wavelength_AA, throughput)`` arrays (Phase-2 use)."""
    SvoFps = _svo()
    try:
        table = SvoFps.get_transmission_data(filter_id)
    except Exception as exc:
        raise SvoUnavailable(f"SVO transmission query failed for {filter_id!r}: {exc}") from exc
    return (np.asarray(table["Wavelength"], dtype=float),
            np.asarray(table["Transmission"], dtype=float))


# ---------------------------------------------------------------------------
# Public, cached, mock-friendly API
# ---------------------------------------------------------------------------
def register_manual_band(band, lambda_eff, zero_point):
    """Manually supply ``lambda_eff`` (Angstrom) + ``zero_point`` (Jy) for ``band``.

    Use this when SVO is unavailable or the automatic filter mapping is wrong. Takes precedence over
    every SVO lookup.

    .. note::
       This is a **process-global** registration: it persists for the rest of the interpreter session
       and affects *all* subsequent band resolutions. Undo it with :func:`unregister_manual_band` (one
       band) or :func:`clear_manual_bands` (all).
    """
    _MANUAL_BANDS[str(band)] = {
        "lambda_eff": float(lambda_eff), "zero_point": float(zero_point)}


def unregister_manual_band(band):
    """Remove a single manual band override (no-op if it was not registered)."""
    _MANUAL_BANDS.pop(str(band), None)


def clear_manual_bands():
    """Remove all manual band overrides registered via :func:`register_manual_band`."""
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


def get_transmission_data(filter_id):
    """Transmission curve ``(wavelength_AA, throughput)`` for ``filter_id`` (Phase-2 spectral integ.)."""
    return _svo_fetch_transmission(filter_id)


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
