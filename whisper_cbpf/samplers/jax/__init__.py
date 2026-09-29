"""JAX/GPU samplers.

Every sampler here satisfies the ``BaseSampler`` contract and returns a ``SamplerResult``, so its
results are interchangeable with the CPU samplers'. Two constraints follow from the registry and
are load-bearing:

* ``get_sampler(name)`` does ``_SAMPLERS[name]()``, so a sampler class must construct with **zero
  arguments**. All configuration travels through ``fit(**kwargs)``.
* jax is imported at module scope in these files, so every name below resolves lazily (PEP 562) --
  importing :mod:`whisper_cbpf` must not require JAX, CUDA or torch to be installed.

``snpe_gpu`` is registered like the rest, as of the generalisation of its simulator; see
:mod:`whisper_cbpf.backends._registration` for how the lazy binding works.

Adapted from whisper-GPU @ 10796a0 (discontinued; superseded by whisper_cbpf),
``samplers/__init__.py``, which lazily exposed the NUTS sampler only. The same pattern now covers
all six, because the merged package promises a CPU-only import.
"""
from __future__ import annotations

from ...backends import gpu_extra_error

__all__ = ["NUTSGPUSampler", "fit_NUTSGPU",
           "ABCGPUSampler", "fit_ABC_GPU",
           "ABCSMCGPUSampler", "fit_ABCSMCGPU",
           "EmceeJAXSampler", "fit_emcee_jax", "fit_emcee_numpy",
           "PyMCJAXVectorizedSampler", "fit_PyMCJAXVectorized",
           "PyMCJAXParallelizedSampler", "fit_PyMCJAXParallelized",
           "SNPEGPUSampler", "fit_snpe_gpu", "make_predict_torch",
           "ContextEmbedding", "encode_bands"]

_MODULE_OF = {
    "NUTSGPUSampler": "nuts_gpu", "fit_NUTSGPU": "nuts_gpu",
    "ABCGPUSampler": "abc_gpu", "fit_ABC_GPU": "abc_gpu",
    "ABCSMCGPUSampler": "abc_smc_gpu", "fit_ABCSMCGPU": "abc_smc_gpu",
    "EmceeJAXSampler": "emcee_jax", "fit_emcee_jax": "emcee_jax",
    "fit_emcee_numpy": "emcee_jax",
    "PyMCJAXVectorizedSampler": "pymc_gpu", "fit_PyMCJAXVectorized": "pymc_gpu",
    "PyMCJAXParallelizedSampler": "pymc_gpu", "fit_PyMCJAXParallelized": "pymc_gpu",
    "SNPEGPUSampler": "snpe_gpu", "fit_snpe_gpu": "snpe_gpu",
    "make_predict_torch": "snpe_gpu",
    "ContextEmbedding": "snpe_gpu", "encode_bands": "snpe_gpu",
}


def __getattr__(name):
    """PEP 562 lazy attribute access.

    ``from whisper_cbpf.samplers.jax import NUTSGPUSampler`` works, but merely importing the
    subpackage does not drag in jax/numpyro/torch. Without the extra installed the failure names
    the extra rather than surfacing a bare ImportError from deep inside a sampler.
    """
    import importlib

    mod_name = _MODULE_OF.get(name)
    if mod_name is None:
        # Submodule access: `whisper_cbpf.samplers.jax.abc_gpu`. Import it lazily too, so the
        # module objects stay reachable without eagerly importing every sampler (and jax with them).
        if name in set(_MODULE_OF.values()):
            try:
                return importlib.import_module(f".{name}", __name__)
            except ImportError as exc:
                raise gpu_extra_error(name) from exc
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        mod = importlib.import_module(f".{mod_name}", __name__)
    except ImportError as exc:
        raise gpu_extra_error(name) from exc
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
