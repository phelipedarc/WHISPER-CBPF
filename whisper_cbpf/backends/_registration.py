"""Registers the JAX/GPU samplers and models into the package's own registries.

Adapted from whisper-GPU @ 10796a0 (discontinued; superseded by whisper_cbpf), ``__init__.py``.
Two things changed, both forced by the merge:

1. **The whisper compatibility guard is gone.** It verified that an external checkout of the CPU
   package was importable and new enough, because the GPU half was a separate repo cloned into
   that checkout. There is no external checkout any more -- the CPU core is this same package,
   so a version skew between the two halves is not expressible.

2. **Registration is lazy.** The legacy module imported the six sampler classes eagerly, and each
   of those modules imports jax at module scope, so importing the legacy package required jax. This
   package promises ``import whisper_cbpf`` works with **no jax, no CUDA, no torch, no redback**,
   so the registry holds a zero-argument :class:`_LazySampler` instead of the class object.
   ``get_sampler(name)`` still does ``_SAMPLERS[name]()`` and still gets a sampler instance --
   the contract is unchanged; only the stored value's type differs. Anything introspecting
   ``_SAMPLERS["nuts_gpu"]`` as a class will see a callable instead.

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

from ._env import gpu_extra_error

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
            raise gpu_extra_error(self.cls_name) from exc
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


def _flare_jax_predict_jax(theta, times, band_idx=None):
    """``Model.predict_jax`` for ``flare_jax``: flat theta, jax-traceable, band-independent.

    The slot was simply never filled. ``models.jax.flare`` has had a ``predict_jax`` all along, but
    with the module's own signature -- a **dict** of named parameters -- while ``Model.predict_jax``
    is contractually ``(theta_flat, times, band_idx=None)``. This is that adapter, and it is what
    lets the generic GPU forward map (``samplers.jax._adapters.make_batched_predict_jax``) build a
    batched simulator for the flare exactly as it does for the kilonovae, instead of ``snpe_gpu``
    carrying a flare-shaped special case.

    ``band_idx`` is accepted and ignored: the flare is self-contained (one flux for all filters), so
    it carries no ``.band_index`` attribute and the adapter never passes one. Module-level so it
    stays picklable, like ``_flare_jax_predict`` above.
    """
    from ..models.jax._flare_spec import PARAMETERS
    from ..models.jax.flare import predict_jax

    return predict_jax({nm: theta[i] for i, nm in enumerate(PARAMETERS)}, times)


#: name -> zero-argument factory returning a sampler instance. Every entry must construct with ZERO
#: arguments, because ``get_sampler(name)`` does ``_SAMPLERS[name]()``; all GPU configuration
#: therefore travels through ``fit(**kwargs)``, never ``__init__``.
#:
#: ``snpe_gpu`` is registered like the rest: its simulator is generic, so it works for any model
#: with a ``predict_jax``. It used to be left out, and the reason was real: ``fit_snpe_gpu`` took a
#: ``model`` argument but always passed ``flare_jax.predict_torch``, so registering it would have let
#: ``wp.fit(lc, "kilonova_one_jax", sampler="snpe_gpu")`` return FLARE simulations labelled with a
#: kilonova's parameter names -- a wrong answer with no error. The root cause was that SNPE's
#: ``predict_torch(theta, times)`` hook had **no bands argument**, so no photometric model could be
#: expressed through it at all. It now takes ``(theta, times, bands)`` and the simulator is built
#: from ``model.predict_jax`` via ``samplers.jax._adapters.make_batched_predict_jax``, so the
#: sampler honours the contract this dict advertises: it fits whatever model it is given, and
#: raises naming ``sampler='snpe'`` for a model with no ``predict_jax`` to build from.
#: The two PyMC entries import PyMC lazily inside ``fit``, so registering them costs nothing and
#: ``import whisper_cbpf`` still works without PyMC installed -- they raise a message naming the
#: install command only if someone actually calls them.
GPU_SAMPLERS = {
    "nuts_gpu": _LazySampler("nuts_gpu", "NUTSGPUSampler"),
    "abc_gpu": _LazySampler("abc_gpu", "ABCGPUSampler"),
    "abc_smc_gpu": _LazySampler("abc_smc_gpu", "ABCSMCGPUSampler"),
    "emcee_jax": _LazySampler("emcee_jax", "EmceeJAXSampler"),
    "snpe_gpu": _LazySampler("snpe_gpu", "SNPEGPUSampler"),
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
#: The TDE and the twelve supernovae additionally require float64
#: (``jax.config.update("jax_enable_x64", True)`` before the first array is created) -- the TDE's
#: envelope ODE accumulates increments float32 cannot resolve, and the supernova engines' cgs
#: luminosities (1e43-1e46 erg/s) overflow float32 outright. Both raise rather than returning a
#: plausible wrong answer, which is another thing a zero-argument registry factory could not
#: communicate.
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
            predict_jax=_flare_jax_predict_jax,
            overwrite=True,
        )
    return sorted(GPU_SAMPLERS), sorted(GPU_MODELS)


registered_samplers, registered_models = register()

__all__ = ["register", "GPU_SAMPLERS", "GPU_MODELS",
           "registered_samplers", "registered_models"]
