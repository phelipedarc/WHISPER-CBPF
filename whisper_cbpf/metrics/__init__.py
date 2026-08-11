"""Posterior and predictive metrics.

Two disjoint sets share this subpackage. They are **not** a numpy/JAX pair and nothing about them
has been reconciled:

* ``_numpy.py`` — WHISPER_AI @ 8ba3843, ``whisper_labia/metrics.py``:
  ``waic``, ``per_band_metrics``, ``predictive_metrics``.
* ``_jax.py`` — whisper-GPU @ 10796a0, ``metrics.py``: ``ess_by_parameter``, ``ess_summary``
  (ArviZ ESS and ESS/sec).

The ``_jax`` slot is the brief's naming for "the GPU-side half", not a dependency claim: that module
imports numpy and lazily arviz, and no JAX at all today. It is still resolved lazily here so the
slot's contract stays stable as JAX-backed metrics are added.

Neither source package had a metrics *registry* — metrics are called directly, not looked up by
name — so none is invented here. Adding one would be new behaviour, which this migration does not do.
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
