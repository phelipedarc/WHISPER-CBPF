"""Posterior and predictive metrics.

Two disjoint sets share this subpackage, and they are not a numpy/JAX pair:

* ``_numpy.py`` — :func:`waic`, :func:`per_band_metrics`, :func:`predictive_metrics`.
* ``_jax.py`` — :func:`ess_by_parameter`, :func:`ess_summary` (ArviZ effective sample size, and
  ESS per second). Despite the file name it imports numpy and, lazily, arviz — no JAX. The name
  marks it as the GPU-side slot, and it is resolved lazily so that adding a genuinely JAX-backed
  metric later does not move anything.

Metrics are called directly rather than looked up by name, so there is no registry here.
"""
from __future__ import annotations

from ._numpy import per_band_metrics, predictive_metrics, waic  # noqa: F401

__all__ = ["waic", "per_band_metrics", "predictive_metrics",
           "ess_by_parameter", "ess_summary"]

_LAZY = {"ess_by_parameter": "._jax", "ess_summary": "._jax"}


def __getattr__(name):
    """Resolve the ``_jax``-slot metrics on first access (PEP 562)."""
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(_LAZY[name], __name__)
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
