"""JAX transient models: the flare, the kilonovae (one to three components, and the Metzger
multi-shell pair), the TDEs (cooling envelope, fall-back, analytical), the supernova family (sixteen
models) and four phenomenological light-curve shapes.

Implementations live in this subpackage; the binding factories are in :mod:`._factories`,
adapted from whisper-GPU @ 10796a0 (discontinued; superseded by whisper_cbpf), ``models/__init__.py``.
Every name resolves lazily (PEP 562) because these modules import jax at module scope and
``import whisper_cbpf`` must not require it.

**The photometric models do not auto-register, and that is correct.** Turning parameters into a band
flux needs a filter set, a redshift and a luminosity distance. None of those are carried by
``predict(parameters, times, bands)`` and none has an honest default -- a guessed distance silently
rescales every fitted ejecta mass or black-hole mass. Bind the dataset context yourself::

    import whisper_cbpf as wp
    wp.register_kilonova(["sdssg", "sdssr", "sdssi"], redshift=0.00984, dl_cm=1.256e26)
    wp.register_tde(["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)

Or fit the explosion time and the redshift as ordinary parameters, the distance following the
redshift through Planck18 (:mod:`whisper_cbpf.models.cosmology`)::

    m = supernova_model("arnett", ["lsstg", "lsstr"], free=["t_exp", "redshift"],
                        prior=Prior({"t_exp": Uniform(t_first - 20, t_first)}),
                        redshift_prior=lc.redshift_prior)

The TDE (and ``tde_fallback``, ``tde_mosfit``, ``tde_analytical``, ``tde_stream_stream``) **and every supernova** additionally require
float64 -- ``jax.config.update("jax_enable_x64", True)`` **before the first array exists**. The TDE's envelope
ODE accumulates Euler increments below float32's epsilon, so in f32 the curve terminates after one
step and the luminosity is ``inf``; the supernova engines instead carry cgs luminosities of 1e43 to
1e46 erg/s, over float32's 3.4e38 ceiling, so every one of them returns ``inf``. Both raise rather
than returning a plausible wrong answer, and **both raise at the first ``predict``, not at
registration** -- so a registry containing ``tde_gaussianrise_jax`` or ``arnett_jax`` is not
evidence that this session can run either. (This said ``register_supernova`` raises at registration
until it was measured: in float32 all 12/12 supernovae register successfully and enter the registry.
``build_sn_grid`` is only reached at factory time when ``times=`` is passed, and the default is
``times=None``.) The flag is global and session-wide, and what it costs is a function of batch size
-- ~4.3x for ``value_and_grad`` at batch 1, 52-57x from batch 100 up -- so it is never set for you.

The two kilonova implementations here and at :mod:`whisper_cbpf.models.two_component_kilonova` are
two models, not one model twice: they are validated against different references and use disjoint
priors.
"""
from __future__ import annotations

from ...backends import gpu_extra_error

__all__ = ["flare_model", "kilonova_model", "register_kilonova",
           "kilonova_two_model", "register_kilonova_two",
           "kilonova_three_model", "register_kilonova_three",
           "tde_model", "register_tde",
           "supernova_model", "supernova_models", "register_supernova",
           "phenomenological_models", "phenomenological_model", "register_phenomenological",
           "tde_fallback_model", "register_tde_fallback",
           "tde_mosfit_model", "register_tde_mosfit",
           "tde_analytical_model", "register_tde_analytical",
           "tde_stream_stream_model", "register_tde_stream_stream",
           "metzger_kilonova_model", "register_metzger_kilonova",
           "magnetar_boosted_kilonova_model", "register_magnetar_boosted_kilonova"]

#: Public names that live in their own module rather than in :mod:`._factories`.
_OWN_MODULE = {
    "phenomenological_models": "phenomenological",
    "phenomenological_model": "phenomenological",
    "register_phenomenological": "phenomenological",
    "tde_fallback_model": "tde_fallback",
    "register_tde_fallback": "tde_fallback",
    "tde_mosfit_model": "tde_mosfit",
    "register_tde_mosfit": "tde_mosfit",
    "tde_analytical_model": "tde_analytical",
    "register_tde_analytical": "tde_analytical",
    "tde_stream_stream_model": "tde_stream_stream",
    "register_tde_stream_stream": "tde_stream_stream",
    "metzger_kilonova_model": "kilonova_metzger",
    "register_metzger_kilonova": "kilonova_metzger",
    "magnetar_boosted_kilonova_model": "kilonova_metzger",
    "register_magnetar_boosted_kilonova": "kilonova_metzger",
}

_SUBMODULES = ("flare", "_flare_spec", "kilonova", "kilonova_two", "kilonova_metzger", "supernova",
               "tde", "tde_fallback", "tde_mosfit", "tde_analytical", "tde_stream_stream",
               "phenomenological",
               "_phenomenological_spec", "csm_interaction", "magnetar_ode", "shock_models")


def _guarded(fn, name):
    """Wrap a factory so a missing ``[gpu]`` extra is reported by name, not as a bare ImportError.

    ``_factories`` itself imports cleanly without JAX -- it defers ``from . import tde`` and friends
    to inside each factory -- so the failure would otherwise surface as a raw
    ``ModuleNotFoundError: No module named 'jax'`` from deep inside a model module, with nothing to
    tell the user which install fixes it.

    ``functools.wraps`` copies ``__name__``/``__doc__`` and sets ``__wrapped__``, so
    ``help()`` and ``inspect.signature()`` still see the real factory.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            import jax  # noqa: F401
        except ImportError as exc:
            raise gpu_extra_error(name) from exc
        return fn(*args, **kwargs)

    return wrapper


def _lazy_public(name, module):
    """A stand-in for ``name`` of ``module``, whose module imports jax at import time.

    ``import whisper_cbpf`` must not import jax, so these functions cannot be bound by importing
    their module. The stand-in carries the real docstring and signature, read from the module's
    source without executing it, and imports the module (naming the ``[gpu]`` extra if jax is
    missing) only when called.
    """
    import ast
    import importlib
    import importlib.util
    import inspect

    node = None
    spec = importlib.util.find_spec(f".{module}", __name__)
    if spec is not None and spec.origin:
        with open(spec.origin, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)

    def wrapper(*args, **kwargs):
        try:
            real = getattr(importlib.import_module(f".{module}", __name__), name)
        except ImportError as exc:
            raise gpu_extra_error(name) from exc
        return real(*args, **kwargs)

    wrapper.__name__ = wrapper.__qualname__ = name
    wrapper.__module__ = f"{__name__}.{module}"
    if node is not None:
        wrapper.__doc__ = ast.get_docstring(node)
        try:
            namespace = {}
            exec(f"def {name}({ast.unparse(node.args)}): pass", namespace)  # literal defaults only
            wrapper.__signature__ = inspect.signature(namespace[name])
        except Exception:  # a default that is not a literal: keep the docstring, drop the signature
            pass
    return wrapper


def __getattr__(name):
    """Resolve a factory on first access, naming the extra if JAX is absent."""
    import importlib

    if name not in __all__:
        # Submodule access: `whisper_cbpf.models.jax.tde`. Lazy for the same reason.
        if name in _SUBMODULES:
            try:
                return importlib.import_module(f".{name}", __name__)
            except ImportError as exc:
                raise gpu_extra_error(name) from exc
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in _OWN_MODULE:
        value = _lazy_public(name, _OWN_MODULE[name])
        globals()[name] = value
        return value
    try:
        mod = importlib.import_module("._factories", __name__)
    except ImportError as exc:
        raise gpu_extra_error(name) from exc
    value = _guarded(getattr(mod, name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
