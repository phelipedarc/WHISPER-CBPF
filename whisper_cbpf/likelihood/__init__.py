"""Likelihoods for transient inference, in flux **or** apparent-magnitude space.

Whisper models predict **flux**; a likelihood compares that prediction to the data in a chosen
``space``:

* ``space='flux'`` — residuals/errors in flux (Jy). Upper limits (non-detections) can be used here.
* ``space='magnitude'`` — the model flux is converted to AB magnitude and compared to the observed
  magnitudes/errors.
* ``space='auto'`` (default) — magnitude data -> magnitude space, flux data -> flux space, and a
  light curve with upper limits -> flux space, where ``likelihood='auto'`` fits the limits with the
  censored likelihood (their significance from ``lc.meta["upper_limit_sigma"]``, else 5 sigma).
  This is the "correct" default; users override only for edge cases (e.g. outliers).

Likelihoods are picklable so they can cross process boundaries for parallel inference. Each exposes
``log_likelihood(model_flux) -> float``.

Two backends, in the layout ``distance`` and ``metrics`` already use:

* ``_numpy.py`` — the likelihood classes and the ``make_likelihood`` registry. Imported eagerly.
* ``_jax.py`` — :func:`log_likelihood_jax`, which turns an already-built likelihood object into the
  traceable JAX function of the model flux that a gradient sampler needs. Resolved lazily, so a
  CPU-only install never imports JAX.

``_numpy.py`` is the default backend and ``_jax.py`` the lazy JAX half; the split changes nothing about
the classes, so ``from whisper_cbpf.likelihood import GaussianLikelihood`` still resolves.
"""
from __future__ import annotations

from ._numpy import (  # noqa: F401
    DEFAULT_UPPER_LIMIT_SIGMA,
    GaussianLikelihood,
    GaussianLikelihoodWithScatter,
    GaussianLikelihoodWithUpperLimits,
    MixtureGaussianLikelihood,
    flux_to_space,
    list_likelihoods,
    make_likelihood,
    register_likelihood,
    resolve_space,
)
# Not public API, re-exported because they were importable from the flat ``likelihood.py`` this
# package replaced and are referenced by name elsewhere (``samplers/snpe.py``'s float32 floor,
# ``samplers/jax/abc_gpu.py``'s copy of it). A split that quietly moved them would break those
# call sites for a reason unrelated to what they do.
from ._numpy import _LN2PI, _MIN_FLUX_JY, _MIN_PROB  # noqa: F401

__all__ = ["GaussianLikelihood", "GaussianLikelihoodWithUpperLimits",
           "MixtureGaussianLikelihood", "GaussianLikelihoodWithScatter",
           "register_likelihood", "list_likelihoods", "make_likelihood",
           "resolve_space", "flux_to_space", "log_likelihood_jax"]

_LAZY = {"log_likelihood_jax": "._jax"}


def __getattr__(name):
    """Resolve the ``_jax``-backend names on first access (PEP 562)."""
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(_LAZY[name], __name__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
