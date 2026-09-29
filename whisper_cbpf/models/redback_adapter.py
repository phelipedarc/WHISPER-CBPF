"""Bind **any** redback model to WHISPER's ``Model`` contract, on the CPU.

redback ships the physics for 347 models behind a single calling convention, so one adapter that
looks a name up in ``redback.model_library.all_models_dict`` covers all of them::

    from whisper_cbpf import register_redback

    register_redback("arnett", redshift=0.0098)             # -> model "arnett_redback"
    register_redback("cooling_envelope", redshift=0.05)     # -> model "cooling_envelope_redback"

The only per-model facts are the parameter names and the prior, and redback ships both:
:func:`redback_parameters` reads the first from the function signature plus the prior file, and
:func:`redback_prior` reads the second from redback's own ``.prior`` file. Nothing is transcribed
here, so nothing can drift. A per-model file is reserved for the JAX *ports*
(``models/jax/tde.py``, ``models/jax/supernova.py``), which have per-model code to hold.

The module is not called ``redback.py`` because it imports the third-party ``redback``: two things
of that name in scope, one shadowing the other on any relative import, helps nobody.

``predict`` maps WHISPER's ``(parameters, times, bands) -> flux [Jy]`` onto redback's
``fn(time, output_format="flux_density", frequency=..., **params) -> mJy``, integrated over each
observation's filter (``photometry="band"``, the default since whisper 0.1.1): every observation
is expanded into its band's 16 Gauss nodes and redback is called once on all of them
(:mod:`whisper_cbpf.synphot`), the same band integral the JAX models compute. ``photometry=
"monochromatic"`` keeps the pre-0.1.1 behaviour, redback's SED at one reference frequency per
band, for physics-parity tests. The unit factor, the one-call-per-light-curve rule, the
out-of-domain handling (warned, and raised when every draw shares the limit), the band
resolution, the double-dilation shim, the refusal of bolometric engines and redback's
``Constraint`` priors as a wall each have a test, and are explained in ``docs/PORTING_NOTES.md``
§9 together with the conventions this adapter inherits from redback rather than choosing.

redback is imported lazily, inside the functions, so ``import whisper_cbpf`` works without the
``[models]`` extra and only calling ``predict`` needs it.
"""
from __future__ import annotations

import functools

import numpy as np

from ..priors import LogUniform, Prior, Uniform

__all__ = ["redback_model", "register_redback", "redback_flux_jy", "redback_parameters",
           "redback_prior", "redback_pinned", "redback_luminosity_distance_cm",
           "redback_applies_dilation", "redback_double_dilation", "installed_redback_preset",
           "LATEST_REDBACK_PRESET", "MJY_TO_JY", "PHOTOMETRY_MODES"]

#: redback's ``output_format="flux_density"`` returns **mJy**; WHISPER's contract is **Jy**.
MJY_TO_JY = 1e-3

#: The newest redback release the JAX ports' presets reproduce. Their defaults follow it when no
#: redback is installed to ask.
LATEST_REDBACK_PRESET = "1.20"

#: Epochs from 0 up to this are clipped up to it (epochs before 0 have no light: zero flux). ``t <= 0``
#: fails *silently* in redback rather than loudly:
#: ``redback.interaction_processes.Diffusion`` keeps only
#: ``time[(time >= 0) & (time <= dense[-1])]``
#: and then indexes with ``searchsorted``, so a negative epoch is handed the first valid epoch's
#: value with no warning, and at exactly ``t = 0`` the photosphere radius is 0 and the temperature
#: diverges. Neither is a plausible answer. Overridable per model via ``min_time_day=``.
MIN_TIME_DAY = 1e-3

#: ``photometry=`` values. ``"band"`` (default) integrates redback's SED over the filter with the
#: shared :mod:`whisper_cbpf.synphot` FilterSet; ``"monochromatic"`` evaluates it at redback's
#: reference frequency for the band (whisper <= 0.1.0), for physics-parity tests and for labels
#: redback knows but no filter curve exists for.
PHOTOMETRY_MODES = ("band", "monochromatic")

#: Prior draws the domain probe evaluates before it calls an epoch limit the MODEL's rather than a
#: draw's: redback must refuse the same epochs, with the same message, at every one.
DOMAIN_PROBE_DRAWS = 8

#: The build-time frequency probe: redback's ``flux_density`` at these two frequencies
#: [Hz] (about LSST i and g) and epochs [d] must differ, or the output is not a flux density.
_PROBE_FREQ_HZ = (4.0e14, 6.5e14)
_PROBE_DAYS = (1.0, 3.0, 5.0)

#: Names a caller reaches for to fix or fit the time origin (``t0``, ``t_exp``,
#: ``explosion_time``, ...). redback's photometric models have none -- the epochs themselves are
#: days since the model's t = 0 -- so ``pin=``/``prior=`` naming one that the model lacks gets the
#: convention in its error. A model that does take one (``t0`` in some prompt and
#: afterglow shapes) never reaches the error.
_TIME_ORIGIN_NAME = r"t_?0|t_?exp\w*|t_?(start|ref|merger|peak)|mjd\w*|\w*(explosion|merger)\w*"

#: bilby prior class name -> WHISPER distribution. Dispatched on the class *name*, not on
#: ``isinstance``:
#: bilby's ``LogUniform`` is a ``PowerLaw`` subclass and ``Sine``/``Cosine`` share base classes with
#: ``Uniform``, so an ``isinstance`` ladder would silently accept a distribution it cannot
#: represent.
_BILBY_TO_WHISPER = {"Uniform": Uniform, "LogUniform": LogUniform}

_MODEL_CACHE: dict = {}          # redback model name -> redback's own function
_FREQ_TABLE = None               # {redback band name -> reference frequency [Hz]}, from filters.csv
_BAND_CACHE: dict = {}           # (label after aliases, band system) -> redback band name
_PRIOR_CACHE: dict = {}          # redback model name -> (dists, pinned, unsupported)
_WARNED_SPANS: set = set()       # (model, accepted span, n epochs) already reported, per process


# --- redback resolution (all lazy) ---------------------------------------------------------------

def _import_redback():
    """Import redback without letting it reconfigure this process's matplotlib or warnings.

    Importing redback sets ``matplotlib.rcParams["text.usetex"] = True`` globally. On a machine
    with no LaTeX installation every subsequent plot then raises
    ``RuntimeError: Failed to process string with tex because latex could not be found`` -- including
    plots that have nothing to do with redback. Binding a model must not change how the rest of the
    session draws, so the setting is restored afterwards. Set it yourself if you want it.

    It also runs ``warnings.simplefilter("ignore")`` (redback 1.20 ``result.py:17``), which silenced
    every warning whisper raises from then on -- the out-of-domain notice below and the SNPE
    fallback notice included. So ``warnings.filters`` is snapshotted and restored
    too: the caller's filters come back as they were, in their order (``simplefilter`` MOVES an
    existing blanket filter to the front, over the caller's exceptions to it), and redback's
    blanket goes. The 15 filters the libraries redback imports register for their own warnings
    (bilby, numpy, urllib3, ...; measured on redback 1.20) are kept, in front, as those libraries
    would have them had the caller imported them.
    """
    import warnings

    import matplotlib

    keys = ("text.usetex", "font.family", "text.latex.preamble")
    before = {k: matplotlib.rcParams[k] for k in keys if k in matplotlib.rcParams}
    filters = list(warnings.filters)
    try:
        import redback
        return redback
    finally:
        for k, v in before.items():
            matplotlib.rcParams[k] = v
        blanket = ("ignore", None, Warning, None, 0)
        restored = [f for f in warnings.filters if f not in filters and f != blanket] + filters
        if restored != warnings.filters:
            warnings.filters[:] = restored
            mutated = getattr(warnings, "_filters_mutated", None)   # stdlib's own cache reset
            if mutated is not None:
                mutated()


def _model_fn(model):
    """redback's own function for ``model``, memoised. Clear error if the extra is missing."""
    fn = _MODEL_CACHE.get(model)
    if fn is None:
        try:
            import logging

            _import_redback()
            from redback.model_library import all_models_dict
            # redback/bilby are chatty; quiet them for the per-evaluation fitting loop.
            for name in ("redback", "bilby"):
                logging.getLogger(name).setLevel(logging.WARNING)
        except Exception as exc:            # ImportError, or any redback init failure
            raise ImportError(
                f"the redback-backed model {model!r} requires the optional 'redback' package. "
                f"Install it with:  pip install 'whisper-cbpf[models]'  (or: pip install redback)."
            ) from exc
        try:
            fn = all_models_dict[model]
        except KeyError:
            raise KeyError(
                f"{model!r} is not a redback model. Pick a key of "
                f"redback.model_library.all_models_dict ({len(all_models_dict)} available)."
            ) from None
        _MODEL_CACHE[model] = fn
    return fn


def _freq_table():
    """redback's ``band -> reference frequency [Hz]`` map, read **once** per process.

    The same ``tables/filters.csv`` that ``redback.utils.bands_to_frequency`` reads -- 264 rows,
    every band redback knows (bare optical letters, SDSS/PS1/LSST/DES/ZTF/ATLAS, Bessell, 2MASS,
    Swift UVOT, HST). Empty when redback is absent, so :func:`_redback_band` degrades to
    pass-through and the ImportError that reaches the caller names redback rather than pandas.
    """
    global _FREQ_TABLE
    if _FREQ_TABLE is None:
        try:
            import pandas as pd
            _import_redback()
            d = pd.read_csv(redback_package_dir() / "tables" / "filters.csv")
            _FREQ_TABLE = {str(b): float(f) for b, f in zip(d["bands"], d["wavelength [Hz]"])}
        except Exception:                   # redback absent -> pass-through, see docstring
            _FREQ_TABLE = {}
    return _FREQ_TABLE


def _redback_band(band, default_system=None, aliases=None):
    """Map a WHISPER band label to a band name redback's filters table knows (monochromatic mode).

    ``aliases`` first; then a **bare letter is read in the default system** -- LSST unless
    :func:`whisper_cbpf.set_default_band_system` or ``default_system=`` says otherwise -- exactly
    as in band mode (:func:`whisper_cbpf.synphot.resolve_filter`). Up to whisper 0.1.0 a bare
    letter passed through, and redback's table reads it as SDSS (``y`` as PS1). Anything else
    already in the table passes straight through; survey spellings (``zg``) are normalised; a
    grouped label (``g-band``) is looked up as its letter in the default system, the one place a
    grouped label still resolves (band mode refuses it: it names no filter curve).
    """
    from ..synphot.labels import bare_letter_filter, default_band_system

    raw = str(band).strip()
    raw = str(dict(aliases or {}).get(raw, raw)).strip()
    key = (raw, default_system or default_band_system())    # the session default can change
    hit = _BAND_CACHE.get(key)
    if hit is not None:
        return hit
    name = bare_letter_filter(raw, default_system) or raw
    tbl = _freq_table()
    if not tbl:                             # redback absent: trust the caller, fail in redback
        return name
    from ..io.bands import normalize_band, resolve_band
    for cand in (name, name.lower(), name.capitalize(), normalize_band(name)):
        if cand in tbl:
            _BAND_CACHE[key] = cand
            return cand
    group = (resolve_band(name, svo_fallback=False, warn=False).get("group") or "")
    letter = group.replace("-band", "").strip().lower()
    cand = bare_letter_filter(letter, default_system) if letter else None
    if cand in tbl:
        _BAND_CACHE[key] = cand
        return cand
    raise ValueError(
        f"band {band!r} is not in redback's filters table and has no optical group that is. Pass a "
        f"name redback knows (sdss*/ps1*/lsst*/ztf*, bessell*, 2mass*, uvot::*, an HST filter) "
        f"or map it with band_aliases=.")


def _frequencies_hz(bands, default_system=None, aliases=None):
    """Per-observation OBSERVER-frame reference frequency [Hz] (monochromatic mode).

    Observer frame is what redback wants: its models hand the frequency to
    ``calc_kcorrected_properties``, which applies the ``(1 + redshift)`` itself.
    """
    tbl = _freq_table()
    bands = np.asarray(bands)
    freq = np.empty(bands.shape, dtype=float)
    for b in np.unique(bands):
        freq[bands == b] = tbl[_redback_band(b, default_system, aliases)]
    return freq


# --- parameters and priors, read from redback ----------------------------------------------------

def _bilby_priors(model):
    """``redback.priors.get_priors(model)``, or ``{}`` when redback ships no prior file for it."""
    _import_redback()
    from redback.priors import get_priors
    try:
        return dict(get_priors(model=model))
    except Exception:                       # no prior file, or a prior that needs data
        return {}


def _translated_prior(model):
    """``(dists, pinned, unsupported)`` for ``model``, from redback's own prior file. Memoised.

    ``dists`` are the free parameters WHISPER can represent, ``pinned`` the ``DeltaFunction``
    entries (name -> value), ``unsupported`` the rest (name -> bilby class name). ``Constraint``
    entries are not parameters, so they are not here; the model applies redback's constraints as
    a wall instead (``constraint=`` on :func:`redback_model`, see
    :mod:`whisper_cbpf.models.constraints`).
    """
    hit = _PRIOR_CACHE.get(model)
    if hit is None:
        dists, pinned, unsupported = {}, {}, {}
        for key, dist in _bilby_priors(model).items():
            kind = type(dist).__name__
            if kind == "Constraint":
                continue
            if kind == "DeltaFunction":
                pinned[key] = float(dist.peak)
            elif kind in _BILBY_TO_WHISPER:
                dists[key] = _BILBY_TO_WHISPER[kind](float(dist.minimum), float(dist.maximum))
            else:
                unsupported[key] = kind
        hit = _PRIOR_CACHE[model] = (dists, pinned, unsupported)
    return hit


def redback_parameters(model):
    """Every parameter redback's ``model`` takes, in redback's own order.

    The signature alone is not enough: ``arnett``'s is ``(time, redshift, f_nickel, mej, **kwargs)``
    and it reads ``vej``, ``kappa``, ``kappa_gamma`` and ``temperature_floor`` out of ``**kwargs``.
    redback's ``.prior`` file is where those are declared, so the answer is the signature's named
    parameters followed by any prior key not already there.
    """
    import inspect

    fn = _model_fn(model)
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    params = [p.name for p in inspect.signature(fn).parameters.values()
              if p.kind in kinds and p.name != "time"]
    dists, pinned, unsupported = _translated_prior(model)
    for key in list(dists) + list(pinned) + list(unsupported):
        if key not in params:
            params.append(key)
    return params


def redback_prior(model):
    """redback's own prior for ``model`` as a WHISPER :class:`~whisper_cbpf.priors.Prior`.

    Only the parameters redback leaves free *and* WHISPER can represent; see the module docstring
    for what happens to ``DeltaFunction`` (-> :func:`redback_pinned`), ``Constraint`` (not a
    parameter: a wall, ``constraint=`` on :func:`redback_model`) and everything else
    (:func:`redback_model` raises).
    """
    _model_fn(model)                            # validate the name, and name redback if it is absent
    return Prior(dict(_translated_prior(model)[0]))


def redback_pinned(model):
    """The parameters redback's prior file fixes with a ``DeltaFunction``, as ``{name: value}``."""
    _model_fn(model)
    return dict(_translated_prior(model)[1])


def redback_luminosity_distance_cm(redshift, model="arnett"):
    """The luminosity distance [cm] redback will use for ``model`` at this redshift.

    redback models take no ``dl_cm``; they derive it from their module's own ``cosmo`` (Planck18).
    A JAX twin takes one explicitly, so pass this or the two differ by a distance ratio squared
    before the physics is reached. Read from the *same* module attribute the model itself reads, so
    it cannot disagree with what the model does even if a future redback changes its cosmology.
    """
    import inspect

    fn = _model_fn(model)
    cosmo = getattr(inspect.getmodule(fn), "cosmo", None)
    if cosmo is None:                       # a model outside the transient_models packages
        from astropy.cosmology import Planck18 as cosmo
    return float(cosmo.luminosity_distance(redshift).cgs.value)


def redback_applies_dilation(model="arnett"):
    """Whether the installed redback multiplies ``model``'s ``flux_density`` by ``(1 + redshift)``.

    ``True`` for redback 1.15.1 (matching the JAX ports' default ``dilation=True``), ``False`` for
    1.12.0, which is the version in this project's CPU container. Determined by reading the model's
    own source, because ``redback.__version__`` reports ``"unknown"`` in the GPU container and a
    version number says nothing about where the change landed anyway.
    """
    import inspect

    # The flux_density branch is everything before the `else:` that opens the spline branch.
    head = inspect.getsource(_model_fn(model)).split("else:")[0]
    return "*(1+redshift)" in "".join(head.split())


def redback_package_dir():
    """redback's package directory (``.../redback/redback``), found WITHOUT importing redback.

    ``None`` when redback is not installed. A directory named ``redback`` without an
    ``__init__.py`` earlier on ``sys.path`` -- a redback CLONE's repository root, when Python runs
    from the folder holding it -- makes ``find_spec("redback")`` a namespace package: redback's
    submodules still load (its installed finder serves them), but ``redback.__file__`` is ``None``
    and ``<location>/tables`` does not exist. So a location counts only if it holds
    ``transient_models/``; otherwise the directory of ``redback.utils`` is used, which the installed
    finder resolves (and, a namespace package having no code, without a heavy import).
    """
    import importlib.util
    from pathlib import Path

    try:
        spec = importlib.util.find_spec("redback")
    except (ImportError, ValueError):       # ValueError: sys.modules["redback"] is None
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for loc in spec.submodule_search_locations:
        if (Path(loc) / "transient_models").is_dir():
            return Path(loc)
    try:
        sub = importlib.util.find_spec("redback.utils")
    except (ImportError, ValueError):
        return None
    return Path(sub.origin).parent if sub is not None and sub.origin else None


@functools.lru_cache(maxsize=None)
def redback_double_dilation(model):
    """Whether redback's ``model`` dilates time twice in its ``flux_density`` branch.

    ``gaussianrise_cooling_envelope``, ``bpl_cooling_envelope`` and ``stream_stream_tde`` (redback
    1.20, ``tde_models.py:496/510``, ``:617/631``, ``:1612/1617``) convert the epochs to the SOURCE
    frame with ``calc_kcorrected_properties`` and then look them up on a light curve they built in
    the OBSERVER frame (``time_temp * (1 + redshift)``). The curve comes out stretched by ``1 + z``
    too little: 0.49 mag off at z = 0.062, 2.70 mag at z = 0.35.

    Detected from the source of that branch -- both halves of the pattern must be there -- so it
    switches itself off in a redback that fixes either half. In redback 1.20 exactly these three of
    its 347 models match. :func:`redback_flux_jy` then calls redback at ``t (1 + z)``, which gives
    the correctly dilated light curve: the JAX port (always right) agrees to 0.3 mmag median over
    prior draws, its residual tail being the port's own. whisper corrects, and says so in the model's description.
    """
    import inspect
    import re

    src = inspect.getsource(_model_fn(model))
    m = re.search(r"if kwargs\['output_format'\] == 'flux_density':(.*?)\n    else:", src, re.S)
    flat = "".join((m.group(1) if m else "").split())
    to_source = "calc_kcorrected_properties(frequency=frequency,redshift=redshift,time=time)" in flat
    observer_grid = re.search(r"(time_temp|full_time|time_since_fb)\*\(1\.?\+redshift\)", flat)
    return bool(to_source and observer_grid)


@functools.lru_cache(maxsize=1)
def installed_redback_preset():
    """The preset key that reproduces the INSTALLED redback: ``"1.20"``, ``"1.15"``, ``"1.12"``,
    or ``None`` when redback is not installed.

    The JAX ports keep one preset table per model family (``tde.REDBACK_ENGINE_PRESETS``,
    ``supernova.REDBACK_GRID_PRESETS``), keyed the same way. The TDE's default ``n_time``
    follows this (:func:`whisper_cbpf.models.jax.tde.default_n_time`), with
    :data:`LATEST_REDBACK_PRESET` standing in for ``None``; the supernova grids default to the
    latest release whatever is installed. The parity tests use it to pick their preset.

    Keyed on the SOURCE, not on ``redback.__version__`` (which reads ``"unknown"`` in some
    installs), and read from disk WITHOUT importing redback: an import costs seconds and installs a
    process-wide ``warnings.simplefilter("ignore")``, neither of which a default may cause. Two
    markers, each the commit that changed the numbers:

    - no ``f_debris`` in ``tde_models.py``: 1.12. The same commit took the TDE grid from 5000
      points to 500.
    - ``geomspace(1e-5`` in ``supernova_models.py``: 1.20 (from 1.18). The ``ip.Diffusion`` dense
      grid became geometric from 1e-5 d instead of linear from 0.
    - otherwise 1.15. redback 1.16-1.17 also land here although their CSM breakout and kilonova
      grids already moved; no preset reproduces those two releases.
    """
    pkg = redback_package_dir()
    if pkg is None:
        return None
    src = pkg / "transient_models"
    try:
        tde = (src / "tde_models.py").read_text()
        sn = (src / "supernova_models.py").read_text()
    except OSError:
        return None
    if "f_debris" not in tde:
        return "1.12"
    return "1.20" if "geomspace(1e-5" in sn else "1.15"


# --- the forward map -----------------------------------------------------------------------------

def _accepted_span(call, order):
    """``(lo, hi)`` into ``order`` -- the slice of epochs redback will answer for.

    Called only after redback has *refused* the full array. redback's models interpolate their
    solution over a monotone time grid, so the epochs they accept are a contiguous run in time; this
    finds it by bisection over redback's own accept/reject, which is why it needs to know nothing
    about any particular model. ``O(log n)`` extra solves, and only on the draws that raise.

    The cheap case first: if the earliest epoch is accepted on its own, the domain is bounded above
    only (the measured ``cooling_envelope`` failure -- the envelope terminates inside the window)
    and one bisection suffices.
    """
    n = order.size
    lo = 0
    if call(order[:1]) is None:                     # even the earliest epoch is out of domain
        a, b = 1, n                                 # invariant: every index < a is rejected alone
        while a < b:
            m = (a + b) // 2
            if call(order[m:m + 1]) is None:
                a = m + 1
            else:
                b = m
        lo = a
        if lo >= n:                                 # redback answers for no epoch at all
            return 0, 0
    a, b = lo + 1, n                                # invariant: order[lo:a] accepted, order[lo:b] ?
    while a < b:
        m = (a + b + 1) // 2
        if call(order[lo:m]) is None:
            b = m - 1
        else:
            a = m
    return lo, a


def _check_photometry(photometry):
    if photometry not in PHOTOMETRY_MODES:
        raise ValueError(f"photometry must be one of {PHOTOMETRY_MODES}, got {photometry!r}")
    return photometry


def redback_flux_jy(model, parameters, times, bands=None, *, min_time_day=MIN_TIME_DAY,
                    redback_kwargs=None, photometry="band", filter_set=None,
                    default_system=None, band_aliases=None):
    """Flux density [Jy] at each ``(time, band)`` from redback's ``model``.

    ``parameters`` keys must be redback's own parameter names (:func:`redback_parameters`);
    ``times`` are observer-frame days in whatever epoch convention the redback model uses (days
    since explosion for ``arnett``, days since fallback for ``cooling_envelope``).

    ``photometry="band"`` (default): redback's SED integrated over each observation's filter,
    with ``filter_set`` (a :class:`whisper_cbpf.synphot.FilterSet`; default Gauss-16 for the
    filters the labels resolve to). ``"monochromatic"``: redback's SED at the band's reference
    frequency from redback's ``filters.csv``. Labels resolve through
    :func:`whisper_cbpf.synphot.resolve_filter` in both modes: bare ``u g r i z y`` are LSST unless
    ``default_system=`` (or :func:`whisper_cbpf.set_default_band_system`) says otherwise, and
    ``band_aliases`` maps your labels first.

    Epochs redback refuses -- ANY exception from redback's function, not only the ``ValueError`` /
    ``IndexError`` of an interpolation past its grid -- come back as exactly ``0.0`` and warn once
    per model and span; so does any observation whose flux (or, in band mode, any of whose nodes)
    is NaN/inf. That is what a sampler must reject, and what the JAX ports return for the same
    parameters. This is the physics at one parameter set: no constraint wall, no domain probe
    (both are :func:`redback_model`'s). For the double-dilation TDEs
    (:func:`redback_double_dilation`) redback is called at ``t (1 + z)``.
    """
    out, dropped = _redback_evaluate(model, parameters, times, bands, min_time_day=min_time_day,
                                     redback_kwargs=redback_kwargs, photometry=photometry,
                                     filter_set=filter_set, default_system=default_system,
                                     band_aliases=band_aliases)
    if dropped is not None:
        _warn_dropped(model, times, min_time_day, dropped)
    return out


def _redback_evaluate(model, parameters, times, bands, *, min_time_day, redback_kwargs,
                      photometry, filter_set, default_system, band_aliases):
    """:func:`redback_flux_jy` without the warning: ``(flux, dropped)``, see :func:`_evaluate`."""
    # Validate the ARGUMENTS before resolving the optional dependency: a caller who forgot `bands`
    # should be told that, not told to install redback. It also keeps this check testable on the
    # minimal install the package promises to support.
    if bands is None:
        raise ValueError(f"the redback model {model!r} is band-dependent; `bands` is required.")
    _check_photometry(photometry)
    fn = _model_fn(model)
    kw = dict(redback_kwargs or {})
    kw.update({k: float(v) for k, v in parameters.items()})
    # redback evaluates these at t/(1+z) on an observer-frame curve; t(1+z) undoes it.
    stretch = 1.0 + kw.get("redshift", 0.0) if redback_double_dilation(model) else 1.0
    return _evaluate(fn, kw, times, bands, min_time_day=min_time_day, photometry=photometry,
                     filter_set=filter_set, default_system=default_system,
                     band_aliases=band_aliases, time_stretch=stretch)


def _flux_jy(fn, kw, times, bands, *, min_time_day, photometry, filter_set=None,
             default_system=None, band_aliases=None, time_stretch=1.0, label=None):
    """Flux [Jy] from any function with redback's ``flux_density`` convention (:func:`_evaluate`),
    warning once per span under ``label`` when epochs are refused. The two-component kilonova
    (``models/two_component_kilonova.py``) passes the sum of two one-component calls."""
    out, dropped = _evaluate(fn, kw, times, bands, min_time_day=min_time_day,
                             photometry=photometry, filter_set=filter_set,
                             default_system=default_system, band_aliases=band_aliases,
                             time_stretch=time_stretch)
    if dropped is not None and label is not None:
        _warn_dropped(label, times, min_time_day, dropped)
    return out


class _RedbackRefused(Exception):
    """redback's own function raised: those epochs, at those parameters, are out of its domain."""


def _evaluate(fn, kw, times, bands, **options):
    """``(flux [Jy], dropped)`` of :func:`_evaluate_clipped`, with no light before day 0.

    An epoch before the model's t = 0 (the explosion or merger) has zero flux, as in the JAX
    ports: redback cannot evaluate it, and the flux at ``min_time_day`` it was clipped to is light
    the transient did not yet emit. It is still passed (clipped) to redback, so the dense grid
    redback builds from the epochs is the same as the JAX side's.
    """
    out, dropped = _evaluate_clipped(fn, kw, times, bands, **options)
    before = np.asarray(times, dtype=float) < 0.0
    if before.any():
        out = np.where(before, 0.0, out)
    return out, dropped


def _evaluate_clipped(fn, kw, times, bands, *, min_time_day, photometry, filter_set=None,
                      default_system=None, band_aliases=None, time_stretch=1.0):
    """``(flux [Jy], dropped)`` for ``fn(time, output_format="flux_density", frequency=nu, **kw)
    -> mJy``, every epoch clipped up to ``min_time_day``.

    ``dropped`` is ``None`` when redback gave a finite flux at every epoch; otherwise ``(kept,
    reason, accepted)``: the indices with a finite flux (possibly none), why the rest have none,
    and -- when redback RAISED on the whole light curve -- the indices of the span it accepted
    (``None`` when it only returned NaN/inf, as redback 1.20's ``cooling_envelope`` does past the
    envelope's end, or when it accepted every epoch once they were in time order). Any exception raised INSIDE redback's function refuses the epochs of that call
    (only ``ValueError``/``IndexError`` were caught, so any other redback failure on one
    draw aborted the run); an exception in whisper's own band integral still propagates.
    """
    t = np.clip(np.asarray(times, dtype=float), min_time_day, None)
    bands = np.asarray(bands)
    if bands.shape != t.shape:
        raise ValueError(f"`bands` and `times` must have the same shape, got {bands.shape} and "
                         f"{t.shape}")
    t_model = t * time_stretch if time_stretch != 1.0 else t

    def mjy(tt, nu):
        try:
            return np.asarray(fn(tt, output_format="flux_density", frequency=nu, **kw),
                              dtype=float)
        except Exception as exc:            # noqa: BLE001 - any redback failure: see docstring
            raise _RedbackRefused(f"{type(exc).__name__}: {exc}") from exc

    if photometry == "band":
        from ..synphot import band_flux_jy, filter_set_for, resolve_filters

        known = filter_set.names if filter_set is not None else ()
        names = np.asarray(resolve_filters(bands, band_aliases, default_system, known))
        fs = filter_set if filter_set is not None else filter_set_for(np.unique(names))

        def flux(idx):
            return band_flux_jy(lambda tt, nu: MJY_TO_JY * mjy(tt, nu), t_model[idx],
                                names[idx], fs)
    else:
        freq = _frequencies_hz(bands, default_system, band_aliases)

        def flux(idx):
            return MJY_TO_JY * mjy(t_model[idx], freq[idx])

    reasons = []

    def call(idx):
        """Jy at ``t[idx]``, or ``None`` if redback refuses those epochs."""
        try:
            return flux(idx)
        except _RedbackRefused as exc:
            reasons.append(str(exc))
            return None

    out = np.zeros(t.shape, dtype=float)
    # ONE call for the whole light curve (point 2 of PORTING_NOTES §9): the dense grids redback
    # builds depend on the time array, so a per-band loop would give each band a different one.
    # In band mode that one call carries every observation's nodes, the last observation last.
    idx = np.arange(t.size)
    jy = call(idx)
    accepted = reason = None
    if jy is None:
        order = np.argsort(t, kind="stable")
        lo, hi = _accepted_span(call, order)
        idx, reason = order[lo:hi], reasons[0]
        accepted = np.sort(idx)
        if hi <= lo:
            return out, (accepted, reason, accepted)    # no epoch is in the model's domain
        jy = call(idx)
        if jy is None:                              # should not happen; stay finite if it does
            return out, (idx[:0], reason, idx[:0])
        if idx.size == t.size:
            # Refused in observation order, answered for every epoch in time order (redback's
            # supernova engines size their dense grid by the LAST epoch handed to them, time[-1] +
            # 100 d, so a band-ordered light curve can end before an earlier row's epoch): no epoch
            # is out of the domain, so nothing is dropped and nothing is probed.
            accepted = reason = None
    finite = np.isfinite(jy)
    out[idx] = np.where(finite, jy, 0.0)
    if accepted is None and finite.all():
        return out, None
    return out, (np.sort(idx[finite]), reason or "it returned NaN/inf there", accepted)


def _describe_drop(times, min_time_day, kept):
    """``(lost, kept)`` phrases naming, in the caller's own days, the epochs with no flux and the
    span of the ones with."""
    t = np.clip(np.asarray(times, dtype=float), min_time_day, None)
    out = np.setdiff1d(np.arange(t.size), kept)
    lost = f"{out.size} of {t.size} epochs ({t[out].min():g}-{t[out].max():g} d)"
    return lost, (f"it answers only for {t[kept].min():g}-{t[kept].max():g} d" if kept.size
                  else "it answers for none of them")


def _warn_dropped(model, times, min_time_day, dropped):
    """Warn, once per model and span, that redback gave no flux at some epochs."""
    import warnings

    kept, reason, _ = dropped
    lost, span = _describe_drop(times, min_time_day, kept)
    key = (str(model), lost, span)
    if key in _WARNED_SPANS:
        return
    _WARNED_SPANS.add(key)
    warnings.warn(
        f"redback's {model!r} gives no flux at {lost} for these parameters -- {span}. They are "
        f"returned as zero flux, which a sampler scores as a rejected draw (redback: {reason}). "
        f"Reported once per model and span.", UserWarning, stacklevel=3)


@functools.lru_cache(maxsize=None)
def _constraint_predicate(model, constraint):
    """``ok(parameters) -> bool`` for redback's ``model`` under ``constraint=``, or ``None`` when
    there is nothing to apply. The transcription in :mod:`whisper_cbpf.models.constraints`
    where there is one -- the formulas the JAX ports use, so CPU and GPU decide alike -- else
    redback's own conversion function and bounds (bilby's strict ``lo < value < hi``)."""
    from . import constraints as C

    if C.check_mode(constraint) is None:
        return None
    if model in C.MODELS:
        return functools.partial(_transcribed_ok, model, constraint)
    setting = C.redback_constraint_setting(model)
    return None if setting is None else functools.partial(_redback_ok, *setting)


def _transcribed_ok(model, constraint, p):
    from . import constraints as C
    return bool(C.constraint_ok(model, p, mode=constraint))


def _redback_ok(conversion, bounds, p):
    out = conversion(dict(p))
    return all(bool((out[k] > lo) & (out[k] < hi)) for k, (lo, hi) in bounds.items() if k in out)


class _RedbackPredict:
    """A bound redback model as WHISPER's ``predict``, picklable.

    A module-level class rather than a closure or a ``functools.partial``, for the reason
    :class:`whisper_cbpf.models.jax._factories._PhotometricPredict` documents: a closure cannot be
    pickled, so a multiprocess sampler (``abc``/``abc_smc`` at ``n_jobs > 1``) fails on it while the
    same fit at ``n_jobs=1`` succeeds. Every attribute here is plain data -- a string, a list of
    strings, dicts of floats, a :class:`~whisper_cbpf.synphot.FilterSet` of numpy arrays, a
    :class:`~whisper_cbpf.priors.Prior`, a set of bytes -- so an instance pickles by value with no
    ``__getstate__``; the unpicklable-in-principle caches (redback's function object, the band
    table) live at module level and each worker process fills its own.

    On top of :func:`redback_flux_jy` it holds the two things that need the model, not one
    parameter set: the constraint wall (a draw redback's constraints reject predicts zero flux
    without calling redback) and the domain probe (the first time redback refuses epochs of a
    light curve, :data:`DOMAIN_PROBE_DRAWS` prior draws decide whether the limit is the model's,
    which raises, or the draw's, which warns once per span).
    """

    def __init__(self, model, parameters, pinned, min_time_day=MIN_TIME_DAY, redback_kwargs=None,
                 photometry="band", filter_set=None, default_system=None, band_aliases=None,
                 prior=None, constraint="corrected"):
        self.model = str(model)
        self.parameters = list(parameters)                  # the FREE ones, in order
        self.pinned = {str(k): float(v) for k, v in dict(pinned or {}).items()}
        self.min_time_day = float(min_time_day)
        self.redback_kwargs = dict(redback_kwargs or {})
        self.photometry = _check_photometry(photometry)
        self.filter_set = filter_set
        self.default_system = default_system
        self.band_aliases = dict(band_aliases or {})
        self.prior = prior                                  # the probe's draws
        self.constraint = constraint
        self.probed = set()                                 # light curves probed (time bytes)

    def physical(self, pars):
        """Whether ``pars`` (free and pinned) pass the model's constraints."""
        ok = _constraint_predicate(self.model, self.constraint)
        return True if ok is None else ok({**self.redback_kwargs, **pars})

    def _evaluate(self, pars, times, bands):
        return _redback_evaluate(self.model, pars, times, bands, min_time_day=self.min_time_day,
                                 redback_kwargs=self.redback_kwargs, photometry=self.photometry,
                                 filter_set=self.filter_set, default_system=self.default_system,
                                 band_aliases=self.band_aliases)

    def __call__(self, parameters, times, bands=None):
        pars = {k: parameters[k] for k in self.parameters}
        pars.update(self.pinned)                            # pinned wins over anything passed in
        if bands is not None and not self.physical(pars):
            return np.zeros(np.shape(times), dtype=float)   # the wall: redback is not asked
        out, dropped = self._evaluate(pars, times, bands)
        if dropped is not None:
            key = np.asarray(times, dtype=float).tobytes()
            if dropped[2] is not None and key not in self.probed:
                self.probe_domain(times, bands, seen=dropped)   # raises on a limit of the model
                self.probed.add(key)
            _warn_dropped(self.model, times, self.min_time_day, dropped)
        return out

    def _probe_draws(self):
        """Up to :data:`DOMAIN_PROBE_DRAWS` seeded prior draws that pass the constraints."""
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(50 * DOMAIN_PROBE_DRAWS):
            pars = {**self.prior.sample(rng), **self.pinned}
            if self.physical(pars):
                draws.append(pars)
                if len(draws) == DOMAIN_PROBE_DRAWS:
                    break
        return draws

    def probe_domain(self, times, bands, seen=None):
        """Raise if redback RAISES on the same epochs of ``times``, with the same message, at every
        parameter set tried -- ``seen`` (the draw that just lost them, if any) and
        :data:`DOMAIN_PROBE_DRAWS` prior draws: a limit of the model, which no draw a sampler could
        propose escapes.

        A limit that moves with the parameters gives different spans or messages (``interp1d``
        names the end of the grid it was built on) and passes. Epochs redback answers with NaN/inf
        rather than an exception carry no such message and never raise here; they only warn.
        """
        sigs = [] if seen is None else [seen]
        draws = self._probe_draws() if self.prior is not None else []
        for pars in draws:
            _, dropped = self._evaluate(pars, times, bands)
            if dropped is None or dropped[2] is None:
                return                                      # answered everywhere: no limit
            sigs.append(dropped)
            if (dropped[2].tobytes(), dropped[1]) != (sigs[0][2].tobytes(), sigs[0][1]):
                return                                      # the limit moves with the draw
        if not draws:
            return
        _, reason, accepted = sigs[0]
        lost, span = _describe_drop(times, self.min_time_day, accepted)
        tried = (f"{len(sigs)} parameter sets tried (this one and {len(draws)} prior draws)"
                 if seen is not None else f"{len(draws)} prior draws")
        raise ValueError(
            f"redback's {self.model!r} refuses {lost} at every one of {tried} -- {span} -- so "
            f"this is a limit of the model, not a draw a sampler could reject (redback: "
            f"{reason}). Drop those epochs, check the time origin (days since the model's t = 0: "
            f"explosion, merger or fallback), or use a model whose domain covers them.")

    def __repr__(self):
        return (f"_RedbackPredict({self.model!r}, free={self.parameters}, "
                f"pinned={self.pinned}, photometry={self.photometry!r}, "
                f"constraint={self.constraint!r})")


# --- building and registering --------------------------------------------------------------------

def _refuse_bolometric(model):
    """Raise if redback's ``model`` is a bolometric engine (static half).

    redback's ``*_bolometric`` functions take ``**kwargs``, so ``output_format="flux_density"``
    and ``frequency=`` are swallowed without an error and the return value is L_bol in erg/s:
    ``shock_cooling_and_arnett_bolometric`` predicted a median 1.2e37 "Jy" where its photometric
    wrapper predicts 1.8e-8 Jy. They share the suffix and, unlike the photometric models, take no
    ``redshift``; both must hold, so a flux model that merely ends in the suffix is not refused.
    """
    import inspect

    fn = _model_fn(model)
    if not (model.endswith("_bolometric") and "redshift" not in inspect.signature(fn).parameters):
        return
    stem = model[:-len("_bolometric")]
    try:
        wrapper = "output_format" in inspect.getsource(_model_fn(stem))
    except Exception:                       # noqa: BLE001 - no such model, or no source
        wrapper = False
    hint = (f"Use its photometric wrapper {stem!r} (register_redback({stem!r}, ...)), the same "
            f"engine behind a photosphere and an SED." if wrapper else
            "redback ships no photometric model of the same name; pick one that implements "
            "output_format='flux_density'.")
    raise ValueError(
        f"redback's {model!r} is a bolometric engine, not a flux density: it returns a luminosity "
        f"in erg/s, takes no redshift, and swallows output_format and frequency in **kwargs, so "
        f"fitted as Jy it is wrong by ~45 orders of magnitude. {hint} Fitting a bolometric light "
        f"curve needs its own data mode, which whisper does not have.")


def _refuse_frequency_independent(model, fn, draws, min_time_day):
    """Raise if redback's ``flux_density`` for ``model`` is the same at two frequencies:
    the probe that catches what the name does not: ``basic_magnetar``, a luminosity engine without
    the suffix, or ``bazin_sne``, a unitless curve).

    Evaluated at each draw in turn until one gives finite, not-all-zero output at both
    frequencies; a draw redback refuses, or one that is zero everywhere, decides nothing, and if no
    draw decides, nothing is refused.
    """
    t = np.clip(np.asarray(_PROBE_DAYS, dtype=float), min_time_day, None)
    for kw in draws:
        try:
            a, b = (np.asarray(fn(t, output_format="flux_density", frequency=np.full(t.size, f),
                                  **kw), dtype=float) for f in _PROBE_FREQ_HZ)
        except Exception:                   # noqa: BLE001 - out of domain here: try another draw
            continue
        if (a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
                or not np.any(a)):
            continue
        if np.array_equal(a, b):
            raise ValueError(
                f"redback's {model!r} returns the same value at {_PROBE_FREQ_HZ[0]:.2g} and "
                f"{_PROBE_FREQ_HZ[1]:.2g} Hz ({a[0]:.3g} at {t[0]:g} d), so it is not a flux "
                f"density: a luminosity engine or a unitless phenomenological curve, which fitted "
                f"as Jy would give every band the same light curve. Use a redback model whose "
                f"flux_density depends on frequency, or whisper's own bazin / gaussian_rise for a "
                f"one-band shape.")
        return


def _constraint_note(model, constraint):
    """The description's account of the constraint wall."""
    from . import constraints as C

    if model in C.MODELS:
        bounds = C.MODELS[model][1]
    else:
        setting = C.redback_constraint_setting(model)
        bounds = {} if setting is None else setting[1]
    if not bounds:
        return ""
    names = ", ".join(f"{lo:g} < {k} < {hi:g}" for k, (lo, hi) in bounds.items())
    if constraint is None:
        return f"; redback's constraints ({names}) NOT applied (constraint=None)"
    note = f"; redback's constraints {names} are a hard wall (zero flux outside)"
    if constraint == "corrected" and "emax_constraint" in bounds:
        note += (f", with the nuclear-burning energy corrected to {C.E_BURN_PER_G / 1e18:.2f}e18 "
                 f"erg/g (redback: {C.E_BURN_PER_G_REDBACK / 1e19:.2f}e19; constraint='redback' "
                 f"keeps it)")
    return note


def _time_origin_hint(key):
    """The time convention, for an error about ``key`` when it looks like a time origin."""
    import re

    if not re.fullmatch(_TIME_ORIGIN_NAME, str(key), re.I):
        return ""
    return (" redback's models take no time-origin parameter: `times` are observer-frame days "
            "since the model's own t = 0 (the explosion for a supernova, the merger for a "
            "kilonova, the fallback for cooling_envelope), not MJD. Put day 0 on the light curve "
            "instead: lc.set_explosion_date(mjd).")


def _build(model, band_names=None, *, redshift=None, name=None, prior=None, pin=None,
           min_time_day=MIN_TIME_DAY, redback_kwargs=None, description=None, photometry="band",
           filter_set=None, default_system=None, band_aliases=None, constraint="corrected",
           times=None):
    """The shared body of :func:`redback_model` and :func:`register_redback`."""
    from . import constraints as C

    _check_photometry(photometry)
    C.check_mode(constraint)
    if times is not None and not band_names:
        raise ValueError("times= runs the domain probe at registration, which evaluates redback "
                         "in a band: pass band_names= too.")
    _refuse_bolometric(model)
    all_params = redback_parameters(model)
    dists, pinned, unsupported = _translated_prior(model)
    dists, pinned = dict(dists), dict(pinned)

    pin = dict(pin or {})
    if redshift is not None:
        if "redshift" in pin:
            raise ValueError("pass the redshift once: `redshift=` or `pin={'redshift': ...}`")
        pin["redshift"] = float(redshift)
    for key, value in pin.items():
        if key not in all_params:
            how = ("redshift= pins" if key == "redshift" and redshift is not None
                   else "pin names")
            hint = (" If the model reads it from **kwargs, pass redback_kwargs={'redshift': z}."
                    if key == "redshift" else _time_origin_hint(key))
            raise ValueError(f"{how} {key!r}, which is not a parameter of redback's {model!r}. "
                             f"Its parameters are {all_params}.{hint}")
        pinned[key] = float(value)
    overrides = prior.distributions if isinstance(prior, Prior) else dict(prior or {})
    for key, dist in overrides.items():
        if key not in all_params:
            raise ValueError(f"prior names {key!r}, which is not a parameter of redback's "
                             f"{model!r}. Its parameters are {all_params}."
                             f"{_time_origin_hint(key)}")
        dists[key] = dist
        pinned.pop(key, None)               # an explicit prior unpins a DeltaFunction parameter

    free = [p for p in all_params if p not in pinned]
    missing = [p for p in free if p not in dists]
    if missing:
        why = {p: unsupported.get(p, "no entry in redback's prior file") for p in missing}
        raise ValueError(
            f"redback's prior for {model!r} gives no WHISPER-representable prior for {missing}: "
            f"{why}. Supply one with prior={{'{missing[0]}': Uniform(lo, hi)}} or fix the value "
            f"with pin={{'{missing[0]}': value}}.")
    model_prior = Prior({k: dists[k] for k in free})
    rng = np.random.default_rng(0)
    _refuse_frequency_independent(
        model, _model_fn(model),
        [{**dict(redback_kwargs or {}), **model_prior.sample(rng), **pinned} for _ in range(4)],
        float(min_time_day))
    if band_names is not None:
        # Fail here rather than inside a likelihood: resolve every band once, up front (in band
        # mode that also builds, or loads, its filter curve), which also warms the memoised tables.
        if photometry == "band":
            from ..synphot import filter_set_for, resolve_filters

            known = filter_set.names if filter_set is not None else ()
            names = resolve_filters(band_names, band_aliases, default_system, known)
            missing_fs = [n for n in names if filter_set is not None
                          and n not in filter_set.names]
            if missing_fs:
                raise ValueError(f"filter_set has no {missing_fs}; it holds {filter_set.names}")
            if filter_set is None:
                filter_set_for(names)
        else:
            for band in band_names:
                _redback_band(band, default_system, band_aliases)

    predict = _RedbackPredict(model, free, pinned, min_time_day, redback_kwargs, photometry,
                              filter_set, default_system, band_aliases, prior=model_prior,
                              constraint=constraint)
    if times is not None:
        t = np.asarray(times, dtype=float)
        predict.probe_domain(t, np.array([list(band_names)[0]] * t.size))
        predict.probed.add(t.tobytes())

    note = "" if not pinned else "; pinned " + ", ".join(f"{k}={v:g}" for k, v in pinned.items())
    if redback_double_dilation(model):
        note += "; redback called at t(1+z) to undo its double time dilation"
    note += _constraint_note(model, constraint)
    how = ("integrated over each band's filter (whisper_cbpf.synphot)" if photometry == "band"
           else "at each band's reference frequency (monochromatic)")
    return dict(
        name=str(name) if name else f"{model}_redback",
        predict=predict,
        parameters=free,
        prior=model_prior,
        description=description if description is not None else (
            f"redback {model!r} (CPU); redback flux_density [mJy] {how} -> flux density "
            f"(Jy){note}."),
    )


def redback_model(model, band_names=None, *, redshift=None, name=None, prior=None, pin=None,
                  min_time_day=MIN_TIME_DAY, redback_kwargs=None, description=None,
                  photometry="band", filter_set=None, default_system=None, band_aliases=None,
                  constraint="corrected", times=None):
    """Build (but do not register) a WHISPER :class:`~whisper_cbpf.models.Model` for a redback name.

    Parameters
    ----------
    model : str
        Any key of ``redback.model_library.all_models_dict`` whose ``flux_density`` is one. A
        ``*_bolometric`` engine (erg/s, no ``redshift``) raises, naming its photometric wrapper,
        and so does any model whose output is the same at two frequencies.
    band_names : sequence of str, optional
        Bands to resolve up front. Purely a fail-fast: a typo'd band otherwise raises inside the
        first likelihood evaluation instead of here. Bands are resolved lazily either way, so this
        does not restrict what ``predict`` accepts.
    redshift : float, optional
        Sugar for ``pin={"redshift": z}``, with the same check: it raises if the model has no
        ``redshift`` parameter. Nearly every redback photometric model takes ``redshift``
        and nearly every real dataset knows it, and pinning it is what makes the model comparable
        with a JAX factory, which takes the redshift as dataset context and never fits it.
    name : str, optional
        Registry name. Defaults to ``f"{model}_redback"``.
    prior : Prior or dict, optional
        Overrides redback's prior per parameter. A parameter named here is **free** with the given
        distribution, even if redback's file pinned it with a ``DeltaFunction``.
    pin : dict, optional
        Fix parameters at a value; they leave ``Model.parameters`` and the prior. Merged on top of
        the ``DeltaFunction`` entries redback's own prior file already pins. There is no
        explosion time to pin: see *Time* in the Notes.
    min_time_day : float, optional
        Epochs are clipped up to this (see :data:`MIN_TIME_DAY`).
    redback_kwargs : dict, optional
        Extra fixed keyword arguments forwarded to redback on every call, for the models that need
        one (``cosmology=``, a ``base_model=``, ...).
    description : str, optional
        Overrides the generated description.
    photometry : {"band", "monochromatic"}
        ``"band"`` (default) integrates redback's SED over each observation's filter, the band
        integral the JAX models compute (docs/PHOTOMETRY.md). ``"monochromatic"`` is whisper <=
        0.1.0's reference-frequency evaluation: 4-40 mmag median and up to 0.65 mag off the band
        integral on redback's own priors, kept for physics-parity tests.
    filter_set : whisper_cbpf.synphot.FilterSet, optional
        The band integral to use (band mode). Default: Gauss-16 for the filters the labels
        resolve to. Pass the one a JAX model was built with to make the two compute the same
        integral.
    default_system : str, optional
        The system bare ``u g r i z y`` are read in, for this model only (default: the session's,
        LSST unless :func:`whisper_cbpf.set_default_band_system` changed it).
    band_aliases : dict, optional
        Your labels -> filter names, applied first, e.g. ``{"g": "sdssg"}``.
    constraint : {"corrected", "redback", None}
        redback's ``Constraint`` priors (``redback.priors._constraint_settings``: Arnett nuclear
        burning >= kinetic energy, magnetar rotational >= kinetic energy, cooling-envelope
        ``eta >= eta_min`` and ``beta <= beta_max``, ...) as a hard wall: a draw that breaks one
        predicts zero flux and redback is not called. ``"corrected"`` (default) bounds the Arnett
        kinetic energy by 1.51e18 erg/g of nickel, the energy burning helium to Ni-56 releases;
        ``"redback"`` keeps redback's 1.91e19 (12.6x too lenient). ``None`` applies
        none, as whisper <= 0.1.0 did. See :mod:`whisper_cbpf.models.constraints`.
    times : array, optional
        The epochs the model will be fitted to, on its own clock. With them the domain probe runs
        here: if redback refuses the same epochs at every one of :data:`DOMAIN_PROBE_DRAWS` prior
        draws -- a limit of the model, like ``two_component_kilonova_model``'s 6-day grid -- this
        raises, naming the span. Without them the same probe runs at the first
        ``predict`` that loses epochs. Needs ``band_names``.

    Returns
    -------
    Model
        Not registered; pass it as an object, or use :func:`register_redback`.

    Notes
    -----
    **Time.** ``predict``'s ``times`` are observer-frame **days since the model's own t = 0** --
    the explosion for ``arnett`` and the other supernovae, the merger for a kilonova, the fallback
    for ``cooling_envelope`` -- exactly as :func:`redback_flux_jy` hands them to redback. redback's
    photometric models take no explosion-time parameter (the epochs are the time origin), so
    ``pin={"t0": mjd}`` raises, naming this. Put day 0 on the light curve instead:
    ``lc.set_explosion_date(mjd)`` (:class:`~whisper_cbpf.LightCurve`). Epochs before 0 have zero
    flux (no light before the explosion, as in the JAX ports), and epochs from 0 up to
    ``min_time_day`` are clipped to it. A fit never sees either: pre-event rows are not fitted.

    Epochs redback refuses at some parameters and not others (a TDE envelope that dies inside the
    window) are zero flux for that draw -- what a sampler must reject -- and warn once per model
    and span.

    Examples
    --------
    Needs the ``[models]`` extra (redback):

    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> m = wp.redback_model("arnett", ["ztfg", "ztfr"], redshift=0.05)
    >>> m.name, m.parameters
    ('arnett_redback', ['f_nickel', 'mej', 'vej', 'kappa', 'kappa_gamma', 'temperature_floor'])
    >>> p = {"f_nickel": 0.1, "mej": 2.0, "vej": 1e4, "kappa": 0.1, "kappa_gamma": 1.0,
    ...      "temperature_floor": 5000.0}
    >>> flux = m.predict(p, np.array([-1.0, 5.0, 20.0]), np.array(["ztfg", "ztfr", "ztfg"]))
    >>> float(flux[0]), bool(np.all(flux[1:] > 0))          # dark before the explosion
    (0.0, True)
    """
    from . import Model

    built = _build(model, band_names, redshift=redshift, name=name, prior=prior, pin=pin,
                   min_time_day=min_time_day, redback_kwargs=redback_kwargs,
                   description=description, photometry=photometry, filter_set=filter_set,
                   default_system=default_system, band_aliases=band_aliases,
                   constraint=constraint, times=times)
    return Model(name=built["name"], predict=built["predict"], parameters=built["parameters"],
                 default_prior=built["prior"], description=built["description"])


def register_redback(model, band_names=None, *, redshift=None, name=None, prior=None, pin=None,
                     min_time_day=MIN_TIME_DAY, redback_kwargs=None, description=None,
                     overwrite=False, photometry="band", filter_set=None, default_system=None,
                     band_aliases=None, constraint="corrected", times=None):
    """:func:`redback_model`, registered under its name so ``fit_ABC(lc, name)`` resolves it.

    This is why ``whisper_cbpf/models/__init__.py`` needs no per-model edit to add a redback model.
    The arguments are :func:`redback_model`'s, and so is the time convention: the light curve's
    ``time`` must be days since the model's own t = 0 (explosion, merger or fallback), not MJD --
    there is no ``t0`` to ``pin``; shift the curve with ``lc.set_explosion_date(mjd)`` (*Time* in
    :func:`redback_model`'s Notes).

    Parameters
    ----------
    model, band_names, redshift, name, prior, pin, min_time_day, redback_kwargs, description,
    photometry, filter_set, default_system, band_aliases, constraint, times
        As :func:`redback_model`.
    overwrite : bool, default False
        Replace a model already registered under that name.

    Returns
    -------
    Model

    Examples
    --------
    Needs the ``[models]`` extra (redback):

    >>> import whisper_cbpf as wp
    >>> m = wp.register_redback("arnett", ["ztfg"], redshift=0.05, name="arnett_example",
    ...                         overwrite=True)
    >>> wp.get_model("arnett_example") is m
    True
    """
    from . import register_model

    built = _build(model, band_names, redshift=redshift, name=name, prior=prior, pin=pin,
                   min_time_day=min_time_day, redback_kwargs=redback_kwargs,
                   description=description, photometry=photometry, filter_set=filter_set,
                   default_system=default_system, band_aliases=band_aliases,
                   constraint=constraint, times=times)
    return register_model(built["name"], built["predict"], built["parameters"],
                          prior=built["prior"], description=built["description"],
                          overwrite=overwrite)
