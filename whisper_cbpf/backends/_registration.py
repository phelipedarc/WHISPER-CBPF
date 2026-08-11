"""Registers the JAX/GPU samplers and models into the package's own registries.

Adapted from whisper-GPU @ 10796a0, ``__init__.py``. Two things changed, both forced by the merge:

1. **The whisper compatibility guard is gone.** It verified that an external ``whisper_labia``
   checkout was importable and new enough, because whisper-GPU was a separate repo cloned into a
   whisper checkout. There is no external checkout any more -- the CPU core is this same package,
   so a version skew between the two halves is not expressible.

2. **Registration is lazy.** The legacy module imported the six sampler classes eagerly, and each
   of those modules imports jax at module scope, so ``import whisper_gpu`` required jax. This
   package promises ``import whisper_cbpf`` works with **no jax, no CUDA, no torch, no redback**,
   so the registry holds a zero-argument :class:`_LazySampler` instead of the class object.
   ``get_sampler(name)`` still does ``_SAMPLERS[name]()`` and still gets a sampler instance --
   the contract is unchanged; only the stored value's type differs. Anything introspecting
   ``_SAMPLERS["nuts_gpu"]`` as a class will see a callable instead. Recorded in ``FINDINGS.md``.

*Registration, not duplication.* ``register_sampler`` / ``register_model`` are plain dict writes,
so this module extends the registries at import with no change on the CPU side. There is
deliberately no parallel registry.

*Discoverable before installable.* Registering costs nothing and instantiates nothing, so
``list_samplers()`` names every GPU sampler on a machine with no JAX at all. The requirement
surfaces when you actually ask for one, with a message naming the extra.

*Idempotent.* ``register_sampler`` raises on a duplicate name unless ``overwrite=True``, so a
re-import or ``importlib.reload`` would otherwise explode.
"""
from __future__ import annotations

from ..models import list_models, register_model
from ..samplers import list_samplers, register_sampler


class _LazySampler:
    """Zero-argument factory that imports its sampler on first call and returns an instance.

    Stored in the sampler registry in place of the class, so that registering a GPU sampler does
    not import jax. Module-level and holding only strings, so it stays picklable for the parallel
    paths that pickle sampler configuration.
    """

    __slots__ = ("module", "cls_name")

    def __init__(self, module, cls_name):
        self.module = module
        self.cls_name = cls_name

    def __call__(self):
        import importlib

        try:
            mod = importlib.import_module(f"..samplers.jax.{self.module}", __package__)
        except ImportError as exc:
            raise ImportError(
                f"{self.cls_name} requires the [gpu] extra: pip install whisper-cbpf[gpu]"
            ) from exc
        return getattr(mod, self.cls_name)()

    def __repr__(self):
        return f"<lazy {self.cls_name} from whisper_cbpf.samplers.jax.{self.module}>"


def _flare_jax_predict(parameters, times, bands=None):
    """Deferred predict for ``flare_jax``: resolves the JAX flare on first call.

    Module-level so it stays picklable for parallel ABC, exactly as the model contract requires.
    Delegates to ``models.jax.flare.predict_numpy`` -- the same callable the legacy
    ``flare_model()`` bound.
    """
    from ..models.jax.flare import predict_numpy

    return predict_numpy(parameters, times, bands)


#: name -> zero-argument factory returning a sampler instance. Every entry must construct with ZERO
#: arguments, because ``get_sampler(name)`` does ``_SAMPLERS[name]()``; all GPU configuration
#: therefore travels through ``fit(**kwargs)``, never ``__init__``.
#:
#: ``snpe_gpu`` is DELIBERATELY ABSENT, for the same reason the kilonova model is absent from
#: ``GPU_MODELS`` below: it cannot honour the contract this dict advertises. ``fit_snpe_gpu`` takes
#: a ``model`` argument but always passes ``flare_jax.predict_torch``, so registering it would let
#: ``wp.fit(lc, "kilonova_one_jax", sampler="snpe_gpu")`` return FLARE simulations labelled with a
#: kilonova's parameter names — a wrong answer with no error. It raises ``NotImplementedError``
#: on a non-flare model, and stays reachable as a direct call::
#:
#:     from whisper_cbpf.samplers.jax.snpe_gpu import fit_snpe_gpu
#:
#: Generalising its ``predict_torch`` hook is what would make it registrable.
#: The two PyMC entries import PyMC lazily inside ``fit``, so registering them costs nothing and
#: ``import whisper_cbpf`` still works without PyMC installed -- they raise a message naming the
#: install command only if someone actually calls them.
GPU_SAMPLERS = {
    "nuts_gpu": _LazySampler("nuts_gpu", "NUTSGPUSampler"),
    "abc_gpu": _LazySampler("abc_gpu", "ABCGPUSampler"),
    "abc_smc_gpu": _LazySampler("abc_smc_gpu", "ABCSMCGPUSampler"),
    "emcee_jax": _LazySampler("emcee_jax", "EmceeJAXSampler"),
    "pymc_jax_gpu_vectorized": _LazySampler("pymc_gpu", "PyMCJAXVectorizedSampler"),
    "pymc_jax_gpu_parallelized": _LazySampler("pymc_gpu", "PyMCJAXParallelizedSampler"),
}

#: name -> (predict, parameters-provider) for SELF-CONTAINED JAX models only.
#:
#: The kilonovae, the TDE and the twelve supernovae are deliberately absent: they are photometric,
#: so turning parameters into a band flux needs a filter set, a redshift and a luminosity distance,
#: none of which ``predict(parameters, times, bands)`` carries and none of which have an honest
#: default -- a guessed distance silently rescales every fitted ejecta mass (or black-hole mass).
#: Bind the dataset context yourself and register the result::
#:
#:     import whisper_cbpf as wp
#:     wp.register_kilonova(["sdssg", "sdssr", "sdssi"], redshift=0.00984, dl_cm=1.256e26)
#:     wp.register_tde(["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)
#:
#: The TDE additionally requires float64 (``jax.config.update("jax_enable_x64", True)`` before the
#: first array is created) -- its envelope ODE accumulates increments float32 cannot resolve. It
#: raises rather than returning a plausible wrong answer, which is another thing a zero-argument
#: registry factory could not communicate.
GPU_MODELS = ("flare_jax",)


def register(overwrite=True):
    """Register every GPU sampler and self-contained JAX model. Called on import. Idempotent."""
    for name, factory in GPU_SAMPLERS.items():
        if name not in list_samplers() or overwrite:
            register_sampler(name, factory, overwrite=True)

    # flare_jax's metadata lives in _flare_spec, which is pure numpy -- so the model registers with
    # its real parameters and prior without importing jax. Only `predict` is deferred.
    if "flare_jax" not in list_models() or overwrite:
        from ..models.jax._flare_spec import PARAMETERS, default_prior

        register_model(
            "flare_jax", _flare_jax_predict, list(PARAMETERS), prior=default_prior(),
            description="JAX Gaussian-rise/exponential-decay flare, log-space parameters "
                        "(JAX/GPU; registered by whisper_cbpf as 'flare_jax').",
            overwrite=True,
        )
    return sorted(GPU_SAMPLERS), sorted(GPU_MODELS)


registered_samplers, registered_models = register()

__all__ = ["register", "GPU_SAMPLERS", "GPU_MODELS",
           "registered_samplers", "registered_models"]
