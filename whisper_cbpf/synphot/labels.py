"""From a band LABEL in your data to a FILTER: :func:`resolve_filter`, the one map every model uses.

A label is what a light curve's ``band`` column holds (``"ztfg"``, ``"g"``, ``"B"``,
``"LSST/LSST.g"``). A filter is a transmission curve with a name: an sncosmo bandpass name, one of
the package's shipped FilterSets, or an SVO Filter Profile Service ID. Every band-integrating model
-- the redback CPU adapter, the two-component kilonova, ``mck19`` and the JAX factories -- asks
this module, so a label means the same filter on the CPU and on the GPU.

**Bare letters mean LSST.** ``u g r i z y`` name no survey. whisper is built for LSST alerts, so a
bare letter is LSST's filter everywhere, with one warning per session the first time it happens.
Change it for the session with :func:`set_default_band_system` (``"sdss"``, ``"ztf"``, ``"ps1"``,
``"des"``), or per call with ``default_system=`` or ``band_aliases=``. This changed in whisper
0.1.1: the redback adapter used to hand a bare letter to redback, whose table defines it as SDSS
(``y`` as PS1). AT2017GFO's ``g r i`` are SDSS photometry, so its fits pass
``default_system="sdss"``.

**A grouped label names no filter.** ``"g-band"`` (``load_lightcurve(band_lookup=...)``) merges
every g filter into one effective band; there is no curve to integrate, so it raises. Load
without ``band_lookup``, or map it with ``band_aliases={"g-band": "ztfg"}``.
"""
from __future__ import annotations

import os
import warnings

__all__ = ["resolve_filter", "resolve_filters", "set_default_band_system", "default_band_system",
           "BAND_SYSTEMS", "BARE_LETTERS"]

#: Bare letters and the systems they may be read in: system -> letter -> filter name.
BARE_LETTERS = ("u", "g", "r", "i", "z", "y")
BAND_SYSTEMS = {
    "lsst": {b: f"lsst{b}" for b in BARE_LETTERS},
    "sdss": {b: f"sdss{b}" for b in "ugriz"},
    "ztf": {b: f"ztf{b}" for b in "gri"},
    "ps1": {b: f"ps1::{b}" for b in "grizy"},
    "des": {b: f"des{b}" for b in "grizy"},
}

#: The session default is mirrored in this environment variable, so worker processes a sampler
#: starts (``spawn``: a fresh interpreter that inherits the environment, not the module state) read
#: bare letters the way the parent does. Setting it before Python starts sets the default too.
BAND_SYSTEM_ENV = "WHISPER_BAND_SYSTEM"
_DEFAULT_SYSTEM = os.environ.get(BAND_SYSTEM_ENV, "lsst").strip().lower()
if _DEFAULT_SYSTEM not in BAND_SYSTEMS:          # a typo in the environment must not break import
    _DEFAULT_SYSTEM = "lsst"
_WARNED = False
_SNCOSMO_KNOWS: dict = {}
_REDBACK_TABLE = None


def set_default_band_system(system):
    """Read bare ``u g r i z y`` as ``system`` for the rest of the session. Returns the old one.

    ``system`` is one of :data:`BAND_SYSTEMS` (``"lsst"``, the default, ``"sdss"``, ``"ztf"``,
    ``"ps1"``, ``"des"``). A per-call ``default_system=`` still wins over this. Also written to
    ``$WHISPER_BAND_SYSTEM`` (:data:`BAND_SYSTEM_ENV`), so worker processes started after this call
    use the same system.

    Parameters
    ----------
    system : str
        One of :data:`BAND_SYSTEMS`.

    Returns
    -------
    str
        The system in use before this call.

    Raises
    ------
    ValueError
        An unknown system.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> old = wp.set_default_band_system("sdss")
    >>> wp.resolve_filter("g")
    'sdssg'
    >>> wp.set_default_band_system(old)
    'sdss'
    """
    global _DEFAULT_SYSTEM
    system = _check_system(system)
    old, _DEFAULT_SYSTEM = _DEFAULT_SYSTEM, system
    os.environ[BAND_SYSTEM_ENV] = system
    return old


def default_band_system():
    """The system bare letters are read in (see :func:`set_default_band_system`).

    Returns
    -------
    str
        ``"lsst"`` unless :func:`set_default_band_system` or ``$WHISPER_BAND_SYSTEM`` chose
        another.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.default_band_system() in ("lsst", "sdss", "ztf", "ps1", "des")
    True
    """
    return _DEFAULT_SYSTEM


def _check_system(system):
    key = str(system).strip().lower()
    if key not in BAND_SYSTEMS:
        raise ValueError(f"unknown band system {system!r}; pick one of {sorted(BAND_SYSTEMS)}")
    return key


def bare_letter_filter(label, default_system=None):
    """The filter a bare letter names, or ``None`` if ``label`` is not a bare letter.

    Warns once per session when the system comes from the session default rather than the call.
    """
    global _WARNED
    if label not in BARE_LETTERS:
        return None
    system = _DEFAULT_SYSTEM if default_system is None else _check_system(default_system)
    table = BAND_SYSTEMS[system]
    if label not in table:
        raise ValueError(f"band system {system!r} has no {label!r} filter (it has "
                         f"{sorted(table)}); pass band_aliases={{'{label}': '<filter>'}}")
    if default_system is None and not _WARNED:
        _WARNED = True
        warnings.warn(
            f"bare band letter {label!r} read as {system.upper()} ({table[label]!r}): whisper reads "
            f"u g r i z y in the {system.upper()} system everywhere. For another survey call "
            f"whisper_cbpf.set_default_band_system('sdss'), or pass default_system= / "
            f"band_aliases= to the model. (Shown once per session.)", UserWarning, stacklevel=3)
    return table[label]


def _sncosmo_knows(name):
    """Whether sncosmo has a bandpass called ``name`` (False when sncosmo is absent). Memoised."""
    hit = _SNCOSMO_KNOWS.get(name)
    if hit is None:
        try:
            import sncosmo

            sncosmo.get_bandpass(name)
            hit = True
        except Exception:                       # not installed, unknown name, download failed
            hit = False
        _SNCOSMO_KNOWS[name] = hit
    return hit


def redback_filter_table():
    """redback's ``tables/filters.csv`` as ``{label: sncosmo name}``, read from disk, no import.

    Importing redback costs seconds and installs a global ``warnings.simplefilter("ignore")``;
    a label lookup may do neither. Empty when redback is not installed.
    """
    global _REDBACK_TABLE
    if _REDBACK_TABLE is None:
        _REDBACK_TABLE = {}
        try:
            import csv

            from ..models.redback_adapter import redback_package_dir

            pkg = redback_package_dir()
            if pkg is not None:
                with open(pkg / "tables" / "filters.csv", newline="") as fh:
                    _REDBACK_TABLE = {row["bands"]: row["sncosmo_name"]
                                      for row in csv.DictReader(fh)}
        except (ImportError, ValueError, OSError, KeyError):
            _REDBACK_TABLE = {}
    return _REDBACK_TABLE


def resolve_filter(label, aliases=None, default_system=None, known=()):
    """The filter name ``label`` stands for. Raises ``ValueError`` if it names no filter.

    In order: ``aliases`` (yours, applied first); a name in ``known`` (the filters of a FilterSet
    you passed, so a curve of your own can carry any name); a bare letter (the default system, see
    the module docstring); a survey spelling (``zg``, ``ZTF_g`` -> ``ztfg``); a shipped or sncosmo
    filter name; a friendly label from redback's filter table (``B`` -> ``bessellb``,
    ``J`` -> ``2massj``); an SVO ID (``Facility/Instrument.Filter``). A grouped label
    (``g-band``) raises, and so does a redback label whose curve sncosmo does not have.

    Parameters
    ----------
    label : str
        A band label as the data spell it.
    aliases : dict, optional
        ``{your label: filter name}``, applied first.
    default_system : str, optional
        The system a bare letter is read in for this call (default
        :func:`default_band_system`, ``"lsst"``).
    known : sequence of str, optional
        Filter names of a FilterSet you pass, accepted as they are.

    Returns
    -------
    str
        The filter name (``"lsstg"``, ``"ztfr"``, ``"bessellb"``, ...).

    Raises
    ------
    ValueError
        ``label`` names no filter, or a grouped band.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.resolve_filter("zg"), wp.resolve_filter("g", default_system="sdss")
    ('ztfg', 'sdssg')
    >>> wp.resolve_filter("my_r", aliases={"my_r": "lsstr"})
    'lsstr'
    """
    raw = str(label).strip()
    if aliases and raw in aliases:
        raw = str(aliases[raw]).strip()
    if raw in known:
        return raw
    return _resolve(raw, default_system)


def _resolve(raw, default_system):
    from ..io.bands import FILTER_LOOKUP, normalize_band
    from .filterset import shipped_filter_names

    bare = bare_letter_filter(raw, default_system)
    if bare is not None:
        return bare
    name = normalize_band(raw)
    if name in set(FILTER_LOOKUP.values()):
        raise ValueError(
            f"band {raw!r} is a grouped effective-band label: it merges every filter of that "
            f"colour, so there is no single transmission curve to integrate. Load the light curve "
            f"without band_lookup= (keeps the survey labels), or map it to a filter with "
            f"band_aliases={{'{raw}': 'ztfg'}} (or lsstg, sdssg, ...).")
    if name in shipped_filter_names() or _sncosmo_knows(name):
        return name
    rb = redback_filter_table()
    for cand in (name, raw):
        if cand in rb:
            target = rb[cand]
            if target in shipped_filter_names() or _sncosmo_knows(target):
                return target
            raise ValueError(
                f"band {raw!r}: redback's filter table maps it to {target!r}, but sncosmo has no "
                f"transmission curve of that name, so it cannot be band-integrated. Pass its SVO ID "
                f"(e.g. band_aliases={{'{raw}': 'Facility/Instrument.Filter'}}) or use "
                f"photometry='monochromatic' on the redback adapter.")
    if "/" in name and "." in name.split("/")[-1]:
        return name                             # an SVO ID; the curve is fetched when built
    raise ValueError(
        f"band {raw!r} names no filter whisper knows: not a bare letter, a shipped filter "
        f"(LSST ugrizy, ZTF gri, SDSS ugriz), an sncosmo bandpass, a label in redback's filter "
        f"table or an SVO ID. Map it with band_aliases={{'{raw}': '<filter name>'}}.")


def resolve_filters(labels, aliases=None, default_system=None, known=()):
    """:func:`resolve_filter` over an array, each distinct label once. Returns a list of names."""
    labels = [str(b).strip() for b in labels]
    memo = {b: resolve_filter(b, aliases, default_system, known) for b in dict.fromkeys(labels)}
    return [memo[b] for b in labels]
