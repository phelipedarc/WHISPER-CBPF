"""JAX transient models: the flare, the kilonovae, the TDE and the supernova family.

Implementations live in this subpackage; the binding factories are in :mod:`._factories`
(whisper-GPU @ 10796a0, ``models/__init__.py``). Every name resolves lazily (PEP 562) because these
modules import jax at module scope and ``import whisper_cbpf`` must not require it.

**The photometric models do not auto-register, and that is correct.** Turning parameters into a band
flux needs a filter set, a redshift and a luminosity distance. None of those are carried by
``predict(parameters, times, bands)`` and none has an honest default -- a guessed distance silently
rescales every fitted ejecta mass or black-hole mass. Bind the dataset context yourself::

    import whisper_cbpf as wp
    wp.register_kilonova(["sdssg", "sdssr", "sdssi"], redshift=0.00984, dl_cm=1.256e26)
    wp.register_tde(["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)

The TDE additionally requires float64 -- ``jax.config.update("jax_enable_x64", True)`` **before the
first array exists**. Its envelope ODE accumulates Euler increments below float32's epsilon, so in
f32 the curve terminates after one step and the luminosity is ``inf``. The flag is global and
session-wide, and it slows the kilonova 11-23x, so it is never set for you.

The two kilonova implementations here and at :mod:`whisper_cbpf.models.two_component_kilonova` are
two models, not one model twice: they are validated against different references and use disjoint
priors. See ``DUPLICATES.md``.
"""
from __future__ import annotations

__all__ = ["flare_model", "kilonova_model", "register_kilonova",
           "tde_model", "register_tde",
           "supernova_model", "supernova_models", "register_supernova"]


def __getattr__(name):
    """Resolve a factory on first access, naming the extra if JAX is absent."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module("._factories", __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the [gpu] extra: pip install whisper-cbpf[gpu]"
        ) from exc
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
