"""``two_component_kilonova`` -- two-component (blue + red) kilonova, backed by **redback**.

This is WHISPER's first model backed by the optional **redback** package (the ``[models]`` extra). It
wraps redback's ``two_component_kilonova_model`` -- a Metzger-type kilonova with two lanthanide-poor /
lanthanide-rich ejecta components (each with its own ejecta mass, velocity, opacity, and temperature
floor). redback computes band-integrated apparent AB magnitudes (redshift-aware, via its Planck18
cosmology); WHISPER converts them to its canonical **flux density (Jy)** so the shared likelihood /
samplers treat this model exactly like any other.

redback is imported lazily, so WHISPER (and ``list_models()``) work without it; only calling
``predict`` requires it. Install with ``pip install 'whisper-labia[models]'`` (redback needs a C
compiler for sncosmo).

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
* Band-dependent: returns flux density per ``(time, band)``; WHISPER bands are mapped to redback LSST
  filters (``g -> lsstg`` ...). Unmappable bands raise a clear error.
* Expensive simulator (~30 ms per band-call): **SNPE** (amortized) is the natural sampler; ABC/MCMC
  want modest budgets. ``predict`` is module-level (picklable) so parallel ABC works.
"""
from __future__ import annotations

import numpy as np

from ..priors import LogUniform, Prior, Uniform

PARAMETERS = [
    "mej_1", "vej_1", "kappa_1", "temperature_floor_1",
    "mej_2", "vej_2", "kappa_2", "temperature_floor_2",
    "redshift",
]
DESCRIPTION = "Redback two-component (blue+red) kilonova; band-integrated AB mag -> flux density (Jy)."
REDBACK_MODEL = "two_component_kilonova_model"

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

_AB_ZP_JY = 3631.0
_MIN_TIME_DAY = 1e-3          # redback kilonova flux is undefined at t<=0
_FAINT_MAG = 99.0             # stand-in for NaN/inf (non-emitting) -> ~0 flux, finite
# WHISPER effective band -> redback LSST filter (bare optical letters; keeps grizy on one system)
_LSST = {"u": "lsstu", "g": "lsstg", "r": "lsstr", "i": "lssti", "z": "lsstz", "y": "lssty"}

_redback_fn = None
_sncosmo_map = None              # {friendly band name -> sncosmo bandpass name}, from redback's table


def _redback_sncosmo_map():
    """Lazily load redback's ``friendly band -> sncosmo bandpass`` map (its ``tables/filters.csv``).

    redback computes magnitudes through **sncosmo**, which needs the registered bandpass name
    (``bessellb``, ``2massj``, ``uvot::uvw1`` …), not the friendly label (``B``, ``J`` …). This map lets
    real UV/optical/NIR photometry (``H``, ``J``, ``Ks``, ``U``, ``uvot::uvw1`` …) resolve to the right
    bandpass with no hand-tuning."""
    global _sncosmo_map
    if _sncosmo_map is None:
        try:
            import os

            import pandas as pd
            import redback
            tbl = os.path.join(os.path.dirname(redback.__file__), "tables", "filters.csv")
            d = pd.read_csv(tbl)
            _sncosmo_map = dict(zip(d["bands"].astype(str), d["sncosmo_name"].astype(str)))
        except Exception:            # redback absent -> non-LSST bands fall through to the resolver
            _sncosmo_map = {}
    return _sncosmo_map


def _get_redback_model():
    """Lazily fetch redback's model function (clear error if the optional extra is missing)."""
    global _redback_fn
    if _redback_fn is None:
        try:
            import logging
            from redback.model_library import all_models_dict
            # redback/bilby are chatty; quiet them for the per-evaluation fitting loop.
            for name in ("redback", "bilby"):
                logging.getLogger(name).setLevel(logging.WARNING)
        except Exception as exc:  # ImportError or any redback init failure
            raise ImportError(
                "The 'two_component_kilonova' model requires the optional 'redback' package. "
                "Install it with:  pip install 'whisper-labia[models]'  (redback needs a C compiler)."
            ) from exc
        _redback_fn = all_models_dict[REDBACK_MODEL]
    return _redback_fn


def _redback_band(band):
    """Map a WHISPER band label to a redback filter name.

    Bare optical letters (``g r i z y u``) map to the LSST system for consistency with the g/r/i
    analysis; any name redback already knows — NIR (``J H K Ks 2massj`` …), Swift UVOT
    (``uvot::uvw1`` …), HST, Bessell, SDSS/PS1 — passes straight through; otherwise the optical
    band-resolver provides a last-resort LSST fallback.
    """
    raw = str(band).strip()
    s = raw.lower()
    # bare LOWERCASE grizy -> LSST (SDSS/LSST-like optical, one consistent system). Case matters:
    # UPPERCASE Johnson-Cousins letters (U B V R I) route to their Bessell bandpasses below.
    if raw in _LSST:
        return _LSST[raw]
    if s in _LSST.values():
        return s
    snc = _redback_sncosmo_map()                # NIR / UVOT / HST / Bessell / SDSS -> sncosmo bandpass
    for cand in (raw, raw.capitalize()):
        if cand in snc:
            return snc[cand]
    if raw in set(snc.values()):                # already a valid sncosmo bandpass name
        return raw
    from ..io.bands import resolve_band          # last resort: resolve an optical group -> LSST
    group = (resolve_band(band, svo_fallback=False, warn=False).get("group") or "")
    key = group.replace("-band", "").strip().lower()
    if key in _LSST:
        return _LSST[key]
    raise ValueError(
        f"two_component_kilonova: band {band!r} is not a recognised redback filter and has no "
        f"optical LSST mapping (bare optical bands {sorted(_LSST)} map to LSST; NIR/UV/HST names "
        "must match redback's filters table)."
    )


def two_component_kilonova_flux(parameters, times, bands=None):
    """Flux density [Jy] at each ``(time, band)`` from redback's two-component kilonova.

    ``times`` are rest-... observer-frame days since merger; ``bands`` selects the redback LSST filter.
    """
    fn = _get_redback_model()
    if bands is None:
        raise ValueError("two_component_kilonova is band-dependent; `bands` is required.")
    t = np.clip(np.asarray(times, dtype=float), _MIN_TIME_DAY, None)
    bands = np.asarray(bands)
    kw = {k: float(parameters[k]) for k in PARAMETERS}

    flux_jy = np.empty(t.shape, dtype=float)
    for b in np.unique(bands):
        sel = bands == b
        mag = np.asarray(
            fn(t[sel], output_format="magnitude", bands=[_redback_band(b)], **kw),
            dtype=float,
        )
        mag = np.nan_to_num(mag, nan=_FAINT_MAG, posinf=_FAINT_MAG, neginf=_FAINT_MAG)
        flux_jy[sel] = _AB_ZP_JY * np.power(10.0, -0.4 * mag)
    return flux_jy
