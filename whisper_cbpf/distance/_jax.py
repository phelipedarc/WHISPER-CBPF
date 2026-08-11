"""ABC distance metrics — JAX backend.

From whisper-GPU @ 10796a0, ``distances.py``. Function body unchanged.

Never imported at package import time: ``whisper_cbpf.distance.__init__`` imports this module
lazily, on the first request for a ``*_jax`` entry, so a CPU-only install never sees JAX.
"""
from __future__ import annotations

__all__ = ["jnp_distance"]


def jnp_distance(name):
    """Return ``f(obs, err, sim) -> scalar`` in jnp, for use inside a vmapped simulate step.

    Deliberately mirrors the numpy versions term for term, including what they do NOT do: no
    masking of non-finite values, no clipping of the error. whisper's ABC lets a NaN propagate into
    the distance (and thence into the acceptance quantile) rather than hiding it, and the GPU port
    matches that so the two produce the same diagnosis on bad input.
    """
    import jax.numpy as jnp

    def _chi2(o, e, s):
        r = (o - s) / e
        return jnp.sum(r * r)

    def _mse(o, e, s):
        r = o - s
        return jnp.mean(r * r)

    def _rmse(o, e, s):
        r = o - s
        return jnp.sqrt(jnp.mean(r * r))

    def _mae(o, e, s):
        return jnp.mean(jnp.abs(o - s))

    def _wmse(o, e, s):
        r = (o - s) / e
        return jnp.mean(r * r)

    def _wmae(o, e, s):
        return jnp.mean(jnp.abs((o - s) / e))

    table = {"chi2": _chi2, "chi_square": _chi2, "mse": _mse, "rmse": _rmse,
             "mae": _mae, "wmse": _wmse, "wmae": _wmae}
    key = str(name).lower()
    if key not in table:
        raise KeyError(f"abc_gpu has no jnp distance {name!r}. Available: {sorted(table)}")
    return table[key]
