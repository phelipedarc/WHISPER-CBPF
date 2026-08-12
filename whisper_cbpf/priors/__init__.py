"""Prior distributions for model parameters.

Small, picklable distribution classes (so priors can cross process boundaries for parallel ABC) with
the hooks future samplers need: ``sample`` (ABC/MCMC init), ``log_prob`` (MCMC), ``rescale``
(unit-cube -> value, for nested sampling).

Two backends over the same distributions, in the layout ``distance`` and ``metrics`` already use:

* ``_numpy.py`` — the classes themselves (:class:`Uniform`, :class:`LogUniform`, :class:`Prior`).
  No JAX, imported eagerly; this is what ``Prior.log_prob`` and every CPU sampler call.
* ``_jax.py`` — :func:`log_prob_jax`, the *same* density as ``Prior.log_prob`` written as a
  traceable JAX function of a flat theta vector. Resolved lazily on first access, so a CPU-only
  install never imports JAX.

The module used to be a single ``priors.py``; the split adds a backend and changes nothing about
the classes, so ``from whisper_cbpf.priors import Prior`` still resolves.
"""
from __future__ import annotations

from ._numpy import LogUniform, Prior, Uniform  # noqa: F401

__all__ = ["Uniform", "LogUniform", "Prior", "log_prob_jax"]

_LAZY = {"log_prob_jax": "._jax"}


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
