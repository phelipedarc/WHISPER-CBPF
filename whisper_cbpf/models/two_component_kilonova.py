"""``two_component_kilonova`` -- two-component (blue + red) kilonova, backed by **redback**.

This is WHISPER's first model backed by the optional **redback** package (the ``[models]`` extra). It
is redback's ``two_component_kilonova_model`` -- a Metzger-type kilonova with two lanthanide-poor /
lanthanide-rich ejecta components (each with its own ejecta mass, velocity, opacity, and temperature
floor) -- computed as **the sum of two ``one_component_kilonova_model`` calls** in flux density,
integrated over each observation's filter with :mod:`whisper_cbpf.synphot` (the band integral every
whisper model uses) and returned in WHISPER's canonical **flux density (Jy)**.

Why not call ``two_component_kilonova_model`` itself (whisper <= 0.1.0 did, through its
``magnitude`` branch): redback solves it on a grid that stops at 6 days in the source frame
(``kilonova_models.py:1241``), so every epoch after 6 d raises inside redback. Its
two components are independent blackbodies summed in flux density, which is exactly what two
one-component calls give -- to 0.006 mmag inside 6 d, finite after it, and 13-30x faster than the
old per-band ``magnitude`` calls. Each component uses redback's default ``dense_resolution=500``
(the default, for parity with redback), so it shares redback's late-time error past ~2.66 t_diff.

redback is imported lazily, so WHISPER (and ``list_models()``) work without it; only calling
``predict`` requires it. Install with ``pip install 'whisper-cbpf[models]'`` (or ``pip install
redback`` -- nothing here needs the extra itself, only an importable redback).

Reference: redback (Sarin et al.); the two-component kilonova follows Metzger 2017 / Kasen et al. 2017.

Parameters (redback names; the default prior follows the Darc kilonova-simulation setup)
----------------------------------------------------------------------------------------
mej_1, mej_2 : ejecta mass of each component [M_sun].
vej_1, vej_2 : ejecta velocity of each component [c].
kappa_1, kappa_2 : grey opacity of each component [cm^2/g] (component 1 = "blue"/low-κ,
    component 2 = "red"/high-κ).
temperature_floor_1, temperature_floor_2 : photospheric temperature floor of each component [K].
redshift : source redshift (sets the luminosity distance via redback's cosmology).

Notes
-----
* Band-dependent: returns flux density per ``(time, band)``. Labels resolve through
  :func:`whisper_cbpf.synphot.resolve_filter`: bare ``g r i z y u`` are LSST (as they always were
  here), ``B``/``J``/``uvot::uvw1`` go through redback's filter table to sncosmo's curves, and a
  grouped label (``i-band``) raises -- it names no filter to integrate over.
* One redback call per component for the whole light curve (6-7 ms for 200 observations).
  ``predict`` is module-level (picklable) so parallel ABC works.
"""
from __future__ import annotations

import numpy as np

from ..priors import LogUniform, Prior, Uniform

PARAMETERS = [
    "mej_1", "vej_1", "kappa_1", "temperature_floor_1",
    "mej_2", "vej_2", "kappa_2", "temperature_floor_2",
    "redshift",
]
DESCRIPTION = ("Redback two-component (blue+red) kilonova, as the sum of two one-component redback "
               "calls; band-integrated flux density (Jy).")
#: The redback model whose physics this is. It is not called: see the module docstring.
REDBACK_MODEL = "two_component_kilonova_model"
#: What is called, once per component.
REDBACK_COMPONENT_MODEL = "one_component_kilonova_model"

#: Default prior (the Darc kilonova-simulation setup: wide ejecta, low-κ "blue" + high-κ "red"
#: components; temperature floors + redshift at redback's defaults).
PRIOR = Prior({
    "mej_1": Uniform(1e-4, 0.1),
    "vej_1": Uniform(0.01, 0.7),
    "kappa_1": Uniform(0.1, 0.5),
    "temperature_floor_1": LogUniform(100.0, 6000.0),
    "mej_2": Uniform(1e-4, 0.1),
    "vej_2": Uniform(0.01, 0.7),
    "kappa_2": Uniform(1.0, 30.0),
    "temperature_floor_2": LogUniform(100.0, 6000.0),
    "redshift": Uniform(0.001, 0.1),
})

_MIN_TIME_DAY = 1e-3          # redback kilonova flux is undefined at t<=0

_redback_fn = None


def _get_redback_model():
    """Lazily fetch redback's one-component model (clear error if the optional extra is missing)."""
    global _redback_fn
    if _redback_fn is None:
        try:
            import logging
            from .redback_adapter import _import_redback
            _import_redback()
            from redback.model_library import all_models_dict
            # redback/bilby are chatty; quiet them for the per-evaluation fitting loop.
            for name in ("redback", "bilby"):
                logging.getLogger(name).setLevel(logging.WARNING)
        except Exception as exc:  # ImportError or any redback init failure
            raise ImportError(
                "The 'two_component_kilonova' model requires the optional 'redback' package. "
                "Install it with:  pip install 'whisper-cbpf[models]'  (or: pip install redback)."
            ) from exc
        _redback_fn = all_models_dict[REDBACK_COMPONENT_MODEL]
    return _redback_fn


def _two_components_mjy(time, *, output_format, frequency, redshift, mej_1, vej_1, kappa_1,
                        temperature_floor_1, mej_2, vej_2, kappa_2, temperature_floor_2):
    """redback's two-component flux density [mJy]: two one-component calls, summed.

    Same calling convention as a redback model, so the redback adapter's band integral and
    out-of-domain handling apply unchanged. Each call builds its own grid from the same epochs,
    at redback's default ``dense_resolution``, exactly as ``two_component_kilonova_model`` does
    for its components (but on the one-component grid, which does not stop at 6 d).
    """
    one = _get_redback_model()
    comps = ((mej_1, vej_1, kappa_1, temperature_floor_1), (mej_2, vej_2, kappa_2,
                                                           temperature_floor_2))
    return sum(np.asarray(one(time, redshift, mej, vej, kappa, temperature_floor=tf,
                              output_format=output_format, frequency=frequency), dtype=float)
               for mej, vej, kappa, tf in comps)


def two_component_kilonova_flux(parameters, times, bands=None):
    """Flux density [Jy] at each ``(time, band)`` from redback's two-component kilonova.

    ``times`` are observer-frame days since merger; each observation is integrated over the filter
    its band label resolves to (bare letters are LSST). Epochs past redback's one-component grid
    (7e6 s = 81 d in the source frame) are 0.0, as the redback adapter returns them, with a
    warning once per span.
    """
    # Validate the ARGUMENTS before resolving the optional dependency: a caller who forgot `bands`
    # should be told that, not told to install redback. It also keeps this check testable on the
    # minimal install the package promises to support.
    if bands is None:
        raise ValueError("two_component_kilonova is band-dependent; `bands` is required.")
    from .redback_adapter import _flux_jy

    # Resolve redback HERE, outside `_flux_jy`: inside it every exception from the model function
    # is a refused epoch, so a missing redback came back as zero flux and a warning.
    _get_redback_model()
    kw = {k: float(parameters[k]) for k in PARAMETERS}
    return _flux_jy(_two_components_mjy, kw, times, bands, min_time_day=_MIN_TIME_DAY,
                    photometry="band", label="two_component_kilonova")
