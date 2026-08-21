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
``fn(time, output_format="flux_density", frequency=..., **params) -> mJy``. The unit factor, the
one-call-per-light-curve rule, the out-of-domain handling, the band pass-through and the memoised
band table each have a test, and are explained in ``docs/PORTING_NOTES.md`` together with the
conventions this adapter inherits from redback rather than choosing.

redback is imported lazily, inside the functions, so ``import whisper_cbpf`` works without the
``[models]`` extra and only calling ``predict`` needs it.
"""
from __future__ import annotations

import numpy as np

from ..priors import LogUniform, Prior, Uniform

__all__ = ["redback_model", "register_redback", "redback_flux_jy", "redback_parameters",
           "redback_prior", "redback_pinned", "redback_luminosity_distance_cm",
           "redback_applies_dilation", "MJY_TO_JY"]

#: redback's ``output_format="flux_density"`` returns **mJy**; WHISPER's contract is **Jy**.
MJY_TO_JY = 1e-3

#: Epochs at or below this are clipped up to it. ``t <= 0`` fails *silently* rather than loudly:
#: ``redback.interaction_processes.Diffusion`` keeps only
#: ``time[(time >= 0) & (time <= dense[-1])]``
#: and then indexes with ``searchsorted``, so a negative epoch is handed the first valid epoch's
#: value with no warning, and at exactly ``t = 0`` the photosphere radius is 0 and the temperature
#: diverges. Neither is a plausible answer. Overridable per model via ``min_time_day=``.
MIN_TIME_DAY = 1e-3

#: bilby prior class name -> WHISPER distribution. Dispatched on the class *name*, not on
#: ``isinstance``:
#: bilby's ``LogUniform`` is a ``PowerLaw`` subclass and ``Sine``/``Cosine`` share base classes with
#: ``Uniform``, so an ``isinstance`` ladder would silently accept a distribution it cannot
#: represent.
_BILBY_TO_WHISPER = {"Uniform": Uniform, "LogUniform": LogUniform}

_MODEL_CACHE: dict = {}          # redback model name -> redback's own function
_FREQ_TABLE = None               # {redback band name -> reference frequency [Hz]}, from filters.csv
_BAND_CACHE: dict = {}           # WHISPER band label -> redback band name
_PRIOR_CACHE: dict = {}          # redback model name -> (dists, pinned, unsupported)


# --- redback resolution (all lazy) ---------------------------------------------------------------

def _import_redback():
    """Import redback without letting it reconfigure this process's matplotlib.

    Importing redback sets ``matplotlib.rcParams["text.usetex"] = True`` globally. On a machine
    with no LaTeX installation every subsequent plot then raises
    ``RuntimeError: Failed to process string with tex because latex could not be found`` -- including
    plots that have nothing to do with redback. Binding a model must not change how the rest of the
    session draws, so the setting is restored afterwards. Set it yourself if you want it.
    """
    import matplotlib

    keys = ("text.usetex", "font.family", "text.latex.preamble")
    before = {k: matplotlib.rcParams[k] for k in keys if k in matplotlib.rcParams}
    try:
        import redback
        return redback
    finally:
        for k, v in before.items():
            matplotlib.rcParams[k] = v


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
            import os

            import pandas as pd
            redback = _import_redback()
            tbl = os.path.join(os.path.dirname(redback.__file__), "tables", "filters.csv")
            d = pd.read_csv(tbl)
            _FREQ_TABLE = {str(b): float(f) for b, f in zip(d["bands"], d["wavelength [Hz]"])}
        except Exception:                   # redback absent -> pass-through, see docstring
            _FREQ_TABLE = {}
    return _FREQ_TABLE


def _redback_band(band):
    """Map a WHISPER band label to a band name redback's filters table knows.

    Anything already in the table passes straight through, **including the bare optical letters**:
    redback's table defines ``g``/``r``/``i`` with SDSS wavelengths, so pass-through *is* redback's
    answer (point 4 of the module docstring). Unrecognised labels fall back to WHISPER's optical
    band resolver, whose group (``g-band`` ...) is looked up as a bare letter.
    """
    raw = str(band).strip()
    hit = _BAND_CACHE.get(raw)
    if hit is not None:
        return hit
    tbl = _freq_table()
    if not tbl:                             # redback absent: trust the caller, fail in redback
        return raw
    for cand in (raw, raw.lower(), raw.capitalize()):
        if cand in tbl:
            _BAND_CACHE[raw] = cand
            return cand
    from ..io.bands import resolve_band      # last resort: optical group -> bare letter
    group = (resolve_band(band, svo_fallback=False, warn=False).get("group") or "")
    key = group.replace("-band", "").strip().lower()
    if key in tbl:
        _BAND_CACHE[raw] = key
        return key
    raise ValueError(
        f"band {band!r} is not in redback's filters table and has no optical group that is. Pass a "
        f"name redback knows (bare optical letters, sdss*/ps1*/lsst*/ztf*, bessell*, 2mass*, "
        f"uvot::*, or an HST filter).")


def _frequencies_hz(bands):
    """Per-observation OBSERVER-frame frequency [Hz] for an array of WHISPER band labels.

    Observer frame is what redback wants: its models hand the frequency to
    ``calc_kcorrected_properties``, which applies the ``(1 + redshift)`` itself.
    """
    tbl = _freq_table()
    bands = np.asarray(bands)
    freq = np.empty(bands.shape, dtype=float)
    for b in np.unique(bands):
        freq[bands == b] = tbl[_redback_band(b)]
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
    entries are dropped: they are not parameters.
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
    for what happens to ``DeltaFunction`` (-> :func:`redback_pinned`), ``Constraint`` (dropped) and
    everything else (:func:`redback_model` raises).
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


def redback_flux_jy(model, parameters, times, bands=None, *, min_time_day=MIN_TIME_DAY,
                    redback_kwargs=None):
    """Flux density [Jy] at each ``(time, band)`` from redback's ``model``.

    ``parameters`` keys must be redback's own parameter names (:func:`redback_parameters`);
    ``times`` are observer-frame days in whatever epoch convention the redback model uses (days
    since explosion for ``arnett``, days since fallback for ``cooling_envelope``); ``bands``
    selects each point's reference frequency.

    Out-of-domain epochs come back as exactly ``0.0``, and so do NaN/inf, which is what a sampler
    must reject and what the JAX ports return for the same parameters.
    """
    # Validate the ARGUMENTS before resolving the optional dependency: a caller who forgot `bands`
    # should be told that, not told to install redback. It also keeps this check testable on the
    # minimal install the package promises to support.
    if bands is None:
        raise ValueError(f"the redback model {model!r} is band-dependent; `bands` is required.")
    fn = _model_fn(model)
    t = np.clip(np.asarray(times, dtype=float), min_time_day, None)
    freq = _frequencies_hz(bands)
    if freq.shape != t.shape:
        raise ValueError(f"`bands` and `times` must have the same shape, got {freq.shape} and "
                         f"{t.shape}")
    kw = dict(redback_kwargs or {})
    kw.update({k: float(v) for k, v in parameters.items()})

    def call(idx):
        """redback's mJy at ``t[idx]``, or ``None`` if redback refuses those epochs."""
        try:
            return np.asarray(fn(t[idx], output_format="flux_density", frequency=freq[idx], **kw),
                              dtype=float)
        except (ValueError, IndexError):
            return None

    out = np.zeros(t.shape, dtype=float)
    # ONE call for the whole light curve (point 2 of the module docstring): the dense grids redback
    # builds depend on the time array, so a per-band loop would give each band a different one.
    idx = np.arange(t.size)
    mjy = call(idx)
    if mjy is None:
        order = np.argsort(t, kind="stable")
        lo, hi = _accepted_span(call, order)
        if hi <= lo:
            return out                              # no epoch is in the model's domain
        idx = order[lo:hi]
        mjy = call(idx)
        if mjy is None:                             # should not happen; stay finite if it does
            return out
    out[idx] = MJY_TO_JY * np.nan_to_num(mjy, nan=0.0, posinf=0.0, neginf=0.0)
    return out


class _RedbackPredict:
    """A bound redback model as WHISPER's ``predict``, picklable.

    A module-level class rather than a closure or a ``functools.partial``, for the reason
    :class:`whisper_cbpf.models.jax._factories._PhotometricPredict` documents: a closure cannot be
    pickled, so a multiprocess sampler (``abc``/``abc_smc`` at ``n_jobs > 1``) fails on it while the
    same fit at ``n_jobs=1`` succeeds. Every attribute here is plain data -- a string, a list of
    strings, dicts of floats -- so an instance pickles by value with no ``__getstate__``; the
    unpicklable-in-principle caches (redback's function object, the band table) live at module level
    and each worker process fills its own.
    """

    def __init__(self, model, parameters, pinned, min_time_day=MIN_TIME_DAY, redback_kwargs=None):
        self.model = str(model)
        self.parameters = list(parameters)                  # the FREE ones, in order
        self.pinned = {str(k): float(v) for k, v in dict(pinned or {}).items()}
        self.min_time_day = float(min_time_day)
        self.redback_kwargs = dict(redback_kwargs or {})

    def __call__(self, parameters, times, bands=None):
        pars = {k: parameters[k] for k in self.parameters}
        pars.update(self.pinned)                            # pinned wins over anything passed in
        return redback_flux_jy(self.model, pars, times, bands,
                              min_time_day=self.min_time_day,
                              redback_kwargs=self.redback_kwargs)

    def __repr__(self):
        return (f"_RedbackPredict({self.model!r}, free={self.parameters}, "
                f"pinned={self.pinned})")


# --- building and registering --------------------------------------------------------------------

def _build(model, band_names=None, *, redshift=None, name=None, prior=None, pin=None,
           min_time_day=MIN_TIME_DAY, redback_kwargs=None, description=None):
    """The shared body of :func:`redback_model` and :func:`register_redback`."""
    all_params = redback_parameters(model)
    dists, pinned, unsupported = _translated_prior(model)
    dists, pinned = dict(dists), dict(pinned)

    if redshift is not None:
        if pin is not None and "redshift" in pin:
            raise ValueError("pass the redshift once: `redshift=` or `pin={'redshift': ...}`")
        pinned["redshift"] = float(redshift)
    for key, value in dict(pin or {}).items():
        if key not in all_params:
            raise ValueError(f"pin names {key!r}, which is not a parameter of redback's {model!r}. "
                             f"Its parameters are {all_params}.")
        pinned[key] = float(value)
    overrides = prior.distributions if isinstance(prior, Prior) else dict(prior or {})
    for key, dist in overrides.items():
        if key not in all_params:
            raise ValueError(f"prior names {key!r}, which is not a parameter of redback's "
                             f"{model!r}. Its parameters are {all_params}.")
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
    if band_names is not None:
        # Fail here rather than inside a likelihood: resolve every band once, up front, which also
        # warms the memoised table.
        for band in band_names:
            _redback_band(band)

    note = "" if not pinned else "; pinned " + ", ".join(f"{k}={v:g}" for k, v in pinned.items())
    return dict(
        name=str(name) if name else f"{model}_redback",
        predict=_RedbackPredict(model, free, pinned, min_time_day, redback_kwargs),
        parameters=free,
        prior=Prior({k: dists[k] for k in free}),
        description=description if description is not None else (
            f"redback {model!r} (CPU); redback flux_density [mJy] at each band's reference "
            f"frequency -> flux density (Jy){note}."),
    )


def redback_model(model, band_names=None, *, redshift=None, name=None, prior=None, pin=None,
                  min_time_day=MIN_TIME_DAY, redback_kwargs=None, description=None):
    """Build (but do not register) a WHISPER :class:`~whisper_cbpf.models.Model` for a redback name.

    Parameters
    ----------
    model : str
        Any key of ``redback.model_library.all_models_dict``.
    band_names : sequence of str, optional
        Bands to resolve up front. Purely a fail-fast: a typo'd band otherwise raises inside the
        first likelihood evaluation instead of here. Bands are resolved lazily either way, so this
        does not restrict what ``predict`` accepts.
    redshift : float, optional
        Sugar for ``pin={"redshift": z}``. Nearly every redback photometric model takes ``redshift``
        and nearly every real dataset knows it, and pinning it is what makes the model comparable
        with a JAX factory, which takes the redshift as dataset context and never fits it.
    name : str, optional
        Registry name. Defaults to ``f"{model}_redback"``.
    prior : Prior or dict, optional
        Overrides redback's prior per parameter. A parameter named here is **free** with the given
        distribution, even if redback's file pinned it with a ``DeltaFunction``.
    pin : dict, optional
        Fix parameters at a value; they leave ``Model.parameters`` and the prior. Merged on top of
        the ``DeltaFunction`` entries redback's own prior file already pins.
    min_time_day : float, optional
        Epochs are clipped up to this (see :data:`MIN_TIME_DAY`).
    redback_kwargs : dict, optional
        Extra fixed keyword arguments forwarded to redback on every call, for the models that need
        one (``cosmology=``, a ``base_model=``, ...).
    description : str, optional
        Overrides the generated description.
    """
    from . import Model

    built = _build(model, band_names, redshift=redshift, name=name, prior=prior, pin=pin,
                   min_time_day=min_time_day, redback_kwargs=redback_kwargs,
                   description=description)
    return Model(name=built["name"], predict=built["predict"], parameters=built["parameters"],
                 default_prior=built["prior"], description=built["description"])


def register_redback(model, band_names=None, *, redshift=None, name=None, prior=None, pin=None,
                     min_time_day=MIN_TIME_DAY, redback_kwargs=None, description=None,
                     overwrite=False):
    """:func:`redback_model`, registered under its name so ``fit_ABC(lc, name)`` resolves it.

    This is why ``whisper_cbpf/models/__init__.py`` needs no per-model edit to add a redback model.
    """
    from . import register_model

    built = _build(model, band_names, redshift=redshift, name=name, prior=prior, pin=pin,
                   min_time_day=min_time_day, redback_kwargs=redback_kwargs,
                   description=description)
    return register_model(built["name"], built["predict"], built["parameters"],
                          prior=built["prior"], description=built["description"],
                          overwrite=overwrite)
