"""Prior distributions for model parameters.

Small, picklable distribution classes (so priors can cross process boundaries for parallel ABC) with
the hooks future samplers need: ``sample`` (ABC/MCMC init), ``log_prob`` (MCMC), ``rescale``
(unit-cube -> value, for nested sampling).

Two backends over the same distributions, in the layout ``distance`` and ``metrics`` already use:

* ``_numpy.py`` — the classes themselves: the boxes :class:`Uniform` and :class:`LogUniform`,
  :class:`Normal`, :class:`TruncatedNormal`, :class:`Fixed` (a pinned value) and :class:`Prior`.
  No JAX, imported eagerly; this is what ``Prior.log_prob`` and every CPU sampler call.
* ``_jax.py`` — :func:`log_prob_jax`, the *same* density as ``Prior.log_prob`` written as a
  traceable JAX function of a flat theta vector, and :func:`ppf_jax`, a distribution's inverse CDF
  (the NUTS samplers sample a TruncatedNormal through it). Resolved lazily on first access, so a
  CPU-only install never imports JAX.

``_numpy.py`` is the default backend and ``_jax.py`` the lazy JAX half; the split changes nothing about
the classes, so ``from whisper_cbpf.priors import Prior`` still resolves.
"""
from __future__ import annotations

from ._numpy import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform  # noqa: F401

__all__ = ["Uniform", "LogUniform", "Normal", "TruncatedNormal", "Fixed", "Prior", "log_prob_jax",
           "ppf_jax"]

_LAZY = {"log_prob_jax": "._jax", "ppf_jax": "._jax"}


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
