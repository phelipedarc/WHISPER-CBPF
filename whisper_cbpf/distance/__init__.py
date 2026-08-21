"""Distance metrics for ABC, and pseudo-likelihoods for model selection.

A distance has the signature ``f(obs_flux, obs_flux_err, sim_flux, bands) -> float`` and is
treated as a black box by the sampler, so any custom metric plugs in.

**numpy is the default backend.** ``from whisper_cbpf.distance import mse_distance`` gives the
numpy one, and nothing in this subpackage imports JAX at module level. The jnp versions are
reached by the ``_jax`` suffix (``get_distance("mse_jax")``), which imports :mod:`._jax` lazily —
so a CPU-only install can import the package, list every distance, and only meets JAX if it asks
for one.

Adding a distance means one function in ``_numpy.py``, optionally one in ``_jax.py``, and one
entry in ``_DISTANCES``.
"""
from __future__ import annotations

from ..backends import gpu_extra_error

from ._numpy import (NUMPY_DISTANCES, chi2_distance, mae_distance, mse_distance,  # noqa: F401
                     rmse_distance, wmae_distance, wmse_distance)

__all__ = ["chi2_distance", "mse_distance", "rmse_distance", "mae_distance",
           "wmse_distance", "wmae_distance", "NUMPY_DISTANCES",
           "register_distance", "get_distance", "list_distances"]

# Registry, so distances are usable by name like models, samplers and likelihoods. Seeded with
# every numpy metric: the CPU ABC keeps all six, which is what makes a CPU/GPU comparison at a
# fixed distance possible. Do not narrow this to chi2.
_DISTANCES = dict(NUMPY_DISTANCES)
_DISTANCES["chi_square"] = chi2_distance

#: jnp-backed entries, exposed by name so they are discoverable **before** JAX is installed.
#: Suffix convention matches the samplers (`nuts_gpu`, `abc_gpu`): `mse` -> `mse_jax`.
_JAX_DISTANCE_NAMES = ("chi2_jax", "chi_square_jax", "mse_jax", "rmse_jax",
                       "mae_jax", "wmse_jax", "wmae_jax")


def register_distance(name, fn, *, overwrite=False):
    """Register a distance ``f(obs_flux, obs_flux_err, sim_flux, bands) -> float`` under ``name``."""
    key = str(name).lower()
    if key in _DISTANCES and not overwrite:
        raise ValueError(f"Distance {name!r} already registered (pass overwrite=True).")
    _DISTANCES[key] = fn


def get_distance(distance):
    """Resolve ``distance``: a callable passes through; a registered name is looked up.

    A ``*_jax`` name resolves through :mod:`._jax`, imported on first use. Without the ``[gpu]``
    extra installed that raises a message naming the extra, never a bare ``ImportError``.
    """
    if callable(distance):
        return distance
    key = str(distance).lower()
    if key in _DISTANCES:
        return _DISTANCES[key]
    if key in _JAX_DISTANCE_NAMES:
        # `_jax` itself imports cleanly -- it defers `import jax.numpy` to inside jnp_distance --
        # so the guard has to wrap the CALL, not just the module import.
        from ._jax import jnp_distance

        try:
            return jnp_distance(key[: -len("_jax")])
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise gpu_extra_error(key) from exc
    raise KeyError(f"Unknown distance {distance!r}. Available: {list_distances()}")


def list_distances():
    """Sorted list of registered distance names.

    Includes the ``*_jax`` entries even when JAX is not installed -- they are *available, requiring
    the ``[gpu]`` extra*, and hiding them would make the GPU path undiscoverable.
    """
    return sorted(set(_DISTANCES) | set(_JAX_DISTANCE_NAMES))
