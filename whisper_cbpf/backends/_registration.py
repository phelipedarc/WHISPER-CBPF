"""whisper-GPU -- JAX/GPU extension for WHISPER.

Importing this package registers its GPU samplers and JAX transient models into WHISPER's own
registries, so everything is reachable through the normal whisper API::

    import whisper_labia as wp
    import whisper_gpu                       # registers on import

    wp.list_samplers()                       # -> [..., 'nuts_gpu', ...]
    res = wp.fit(lc, "kilonova_one_jax", sampler="nuts_gpu", log_prob_fn=...)

Design notes
------------
*Registration, not duplication.* whisper's ``register_sampler`` / ``register_model`` are plain dict
writes, so an external package can extend them at import time with no change on the whisper side.
There is deliberately no parallel registry here.

*Import must not require a GPU.* Registering a class does not instantiate it, so ``import
whisper_gpu`` succeeds on a CPU-only box and ``wp.list_samplers()`` still tells you what exists.
The GPU requirement surfaces at ``fit()`` time via :func:`whisper_gpu._env.require_jax`, with a
message that says which command to run. Importing jax eagerly here would make the package
unimportable exactly where a user most needs to read its error message.

*Registration is idempotent.* ``register_sampler`` raises on a duplicate name unless
``overwrite=True``; a re-import or ``importlib.reload`` would otherwise explode.
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- whisper compatibility guard -------------------------------------------------------------
# The GPU samplers call attach_band_metrics / attach_predictive_metrics so their SamplerResult
# carries the same metric blocks every whisper sampler produces. Both were added to
# whisper_labia/samplers/base.py recently; against an older checkout (or a stale
# `pip install whisper-labia`) the failure is an ImportError raised deep inside a sampler, long
# after the user has paid for a fit. Fail here instead, with the fix in the message.
_REQUIRED_WHISPER_SYMBOLS = {
    "whisper_labia.samplers.base": ("BaseSampler", "SamplerResult", "summarize_posterior",
                                     "attach_band_metrics", "attach_predictive_metrics"),
    "whisper_labia.models": ("Model", "register_model", "get_model", "list_models"),
    "whisper_labia.priors": ("Prior", "Uniform", "LogUniform"),
    "whisper_labia.io.schema": ("LightCurve",),
    "whisper_labia.samplers": ("register_sampler", "get_sampler", "list_samplers"),
}


def _check_whisper():
    import importlib
    try:
        import whisper_labia  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "whisper-GPU extends WHISPER and cannot be imported without it.\n"
            "  Expected layout:  <whisper repo>/whisper_gpu/   (clone this repo as 'whisper_gpu'\n"
            "  inside the whisper checkout, so both are importable from the repo root)."
        ) from exc

    missing = []
    for mod_name, symbols in _REQUIRED_WHISPER_SYMBOLS.items():
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            missing.append(f"{mod_name} (whole module)")
            continue
        missing += [f"{mod_name}.{s}" for s in symbols if not hasattr(mod, s)]
    if missing:
        raise ImportError(
            "whisper-GPU needs a newer WHISPER than the one on this path. Missing:\n"
            + "".join(f"  - {m}\n" for m in missing)
            + f"  found whisper_labia at: {whisper_labia.__file__}\n"
            "  Update the whisper checkout (or `pip uninstall whisper-labia` if a stale installed\n"
            "  copy is shadowing your working tree -- check the path above)."
        )


_check_whisper()

# --- registration ------------------------------------------------------------------------------
from whisper_labia.models import list_models, register_model  # noqa: E402
from whisper_labia.samplers import list_samplers, register_sampler  # noqa: E402

from . import _env  # noqa: E402,F401
from . import distances as _distances  # noqa: E402
from .models import (  # noqa: E402,F401
    flare_model,
    kilonova_model,
    register_kilonova,
    register_supernova,
    register_tde,
    supernova_model,
    supernova_models,
    tde_model,
)
from .samplers.abc_gpu import ABCGPUSampler  # noqa: E402
from .samplers.abc_smc_gpu import ABCSMCGPUSampler  # noqa: E402
from .samplers.emcee_jax import EmceeJAXSampler  # noqa: E402
from .samplers.nuts_gpu import NUTSGPUSampler  # noqa: E402
from .samplers.pymc_gpu import (  # noqa: E402
    PyMCJAXParallelizedSampler,
    PyMCJAXVectorizedSampler,
)

#: name -> whisper sampler class. Every class must construct with ZERO arguments, because
#: whisper's ``get_sampler(name)`` does ``_SAMPLERS[name]()``; all GPU configuration therefore
#: travels through ``fit(**kwargs)``, never ``__init__``.
#:
#: ``snpe_gpu`` is DELIBERATELY ABSENT, for the same reason the kilonova model is absent from
#: ``GPU_MODELS`` below: it cannot honour the contract this dict advertises. ``fit_snpe_gpu`` takes
#: a ``model`` argument but always passes ``flare_jax.predict_torch``, so registering it would let
#: ``wp.fit(lc, "kilonova_one_jax", sampler="snpe_gpu")`` return FLARE simulations labelled with a
#: kilonova's parameter names — a wrong answer with no error. It now raises ``NotImplementedError``
#: on a non-flare model, and stays reachable as a direct call::
#:
#:     from whisper_gpu.samplers.snpe_gpu import fit_snpe_gpu
#:
#: Generalising its ``predict_torch`` hook is what would make it registrable.
#: The two PyMC entries import PyMC lazily inside ``fit``, so registering them costs nothing and
#: ``import whisper_gpu`` still works without PyMC installed -- they raise a message naming the
#: install command only if someone actually calls them.
GPU_SAMPLERS = {
    "nuts_gpu": NUTSGPUSampler,
    "abc_gpu": ABCGPUSampler,
    "abc_smc_gpu": ABCSMCGPUSampler,
    "emcee_jax": EmceeJAXSampler,
    "pymc_jax_gpu_vectorized": PyMCJAXVectorizedSampler,
    "pymc_jax_gpu_parallelized": PyMCJAXParallelizedSampler,
}

#: name -> zero-argument factory returning a ``whisper_labia.models.Model``.
#:
#: Only SELF-CONTAINED models appear here. The kilonova, the TDE and the twelve supernovae are
#: deliberately absent: they are photometric, so turning parameters into a band flux needs a
#: filter set, a redshift and a luminosity distance, none of which whisper's
#: ``predict(parameters, times, bands)`` signature carries and none of which have an honest
#: default -- a guessed distance silently rescales every fitted ejecta mass (or black-hole mass).
#: Bind the dataset context yourself and register the result::
#:
#:     import whisper_gpu
#:     whisper_gpu.register_kilonova(["sdssg", "sdssr", "sdssi"], redshift=0.00984, dl_cm=1.256e26)
#:     whisper_gpu.register_tde(["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)
#:     whisper_gpu.register_supernova("arnett", ["ztfg", "ztfr"], redshift=0.05, dl_cm=7.09e26)
#:
#: ``whisper_gpu.supernova_models()`` lists the twelve names the last one accepts. Two of its
#: settings change the MODEL rather than its accuracy -- ``magnetar_convention`` (which
#: redback's ``basic_magnetar``, worth up to 0.75 mag) and ``interaction`` (whether the fallback
#: models apply the diffusion they declare, worth up to 15 mag) -- and both are reported in the
#: registered model's description rather than chosen silently.
#:
#: The TDE additionally requires float64 (``jax.config.update("jax_enable_x64", True)`` before the
#: first array is created) -- its envelope ODE accumulates increments float32 cannot resolve. It
#: raises rather than returning a plausible wrong answer, which is another thing a zero-argument
#: registry factory could not communicate. The supernovae do not: they have no accumulator.
GPU_MODELS = {
    "flare_jax": flare_model,
}


def register(overwrite=True):
    """Register every GPU sampler and JAX model into whisper's registries.

    Called automatically on import. Idempotent by default so a reload is harmless.
    """
    for name, cls in GPU_SAMPLERS.items():
        if name not in list_samplers() or overwrite:
            register_sampler(name, cls, overwrite=True)
    # whisper ships only `chi2`; add MSE/MAE/weighted variants to its registry so the CPU ABC
    # gains them too and CPU/GPU runs can be compared under an identical distance.
    _distances.register_into_whisper(overwrite=overwrite)
    known = set(list_models())
    for name, factory in GPU_MODELS.items():
        if name in known and not overwrite:
            continue
        m = factory()
        register_model(name, m.predict, m.parameters, prior=m.default_prior,
                       description=m.description, overwrite=True)
    return sorted(GPU_SAMPLERS), sorted(GPU_MODELS)


registered_samplers, registered_models = register()

__all__ = ["register", "GPU_SAMPLERS", "GPU_MODELS",
           "registered_samplers", "registered_models", "__version__"]
