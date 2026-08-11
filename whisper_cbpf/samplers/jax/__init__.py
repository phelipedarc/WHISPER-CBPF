"""JAX/GPU samplers.

Every sampler here satisfies the ``BaseSampler`` contract and returns a ``SamplerResult``, so its
results are interchangeable with the CPU samplers'. Two constraints follow from the registry and
are load-bearing:

* ``get_sampler(name)`` does ``_SAMPLERS[name]()``, so a sampler class must construct with **zero
  arguments**. All configuration travels through ``fit(**kwargs)``.
* jax is imported at module scope in these files, so every name below resolves lazily (PEP 562) --
  importing :mod:`whisper_cbpf` must not require JAX, CUDA or torch to be installed.

``snpe_gpu`` is reachable here by direct import but is deliberately **not** registered; see
:mod:`whisper_cbpf.backends._registration` for why.

Adapted from whisper-GPU @ 10796a0, ``samplers/__init__.py``, which lazily exposed the NUTS sampler
only. The same pattern now covers all six, because the merged package promises a CPU-only import.
"""
from __future__ import annotations

__all__ = ["NUTSGPUSampler", "fit_NUTSGPU",
           "ABCGPUSampler", "fit_ABC_GPU",
           "ABCSMCGPUSampler", "fit_ABCSMCGPU",
           "EmceeJAXSampler", "fit_emcee_jax", "fit_emcee_numpy",
           "PyMCJAXVectorizedSampler", "fit_PyMCJAXVectorized",
           "PyMCJAXParallelizedSampler", "fit_PyMCJAXParallelized",
           "ContextEmbedding", "encode_bands", "fit_snpe_gpu"]

_MODULE_OF = {
    "NUTSGPUSampler": "nuts_gpu", "fit_NUTSGPU": "nuts_gpu",
    "ABCGPUSampler": "abc_gpu", "fit_ABC_GPU": "abc_gpu",
    "ABCSMCGPUSampler": "abc_smc_gpu", "fit_ABCSMCGPU": "abc_smc_gpu",
    "EmceeJAXSampler": "emcee_jax", "fit_emcee_jax": "emcee_jax",
    "fit_emcee_numpy": "emcee_jax",
    "PyMCJAXVectorizedSampler": "pymc_gpu", "fit_PyMCJAXVectorized": "pymc_gpu",
    "PyMCJAXParallelizedSampler": "pymc_gpu", "fit_PyMCJAXParallelized": "pymc_gpu",
    "ContextEmbedding": "snpe_gpu", "encode_bands": "snpe_gpu", "fit_snpe_gpu": "snpe_gpu",
}


def __getattr__(name):
    """PEP 562 lazy attribute access.

    ``from whisper_cbpf.samplers.jax import NUTSGPUSampler`` works, but merely importing the
    subpackage does not drag in jax/numpyro/torch. Without the extra installed the failure names
    the extra rather than surfacing a bare ImportError from deep inside a sampler.
    """
    mod_name = _MODULE_OF.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module(f".{mod_name}", __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the [gpu] extra: pip install whisper-cbpf[gpu]"
        ) from exc
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
