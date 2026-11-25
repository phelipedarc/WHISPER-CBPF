"""``band_flux_jy``: a band flux per observation from any model that gives F_nu at (time, frequency).

The CPU half of the photometry. Every observation is expanded into its band's quadrature nodes,
the model is called ONCE on the flattened ``(time, frequency)`` pairs, and the weighted sum is
folded back per observation with ``np.add.reduceat``:

    F_band(t_i) = sum_k w_bk F_nu(t_i, c/lam_bk) / sum_k w_bk

Two layout rules make that one call reproduce what the model returns for the observations
themselves, and both are load-bearing for redback:

* **each observation's nodes are contiguous**, in observation order;
* **so the last observation stays last.** redback sizes several dense grids by ``time[-1]``
  (``ip.Diffusion``), not by ``max(time)``, so moving a different epoch to the end of
  the flattened array changes the physics -- by up to 0.8 mmag when rows were reordered.

The band-dependent part of the expansion is memoised on the observations' bands and the
FilterSet's hash; the times are repeated each call (one ``np.repeat``).
"""
from __future__ import annotations

import numpy as np

__all__ = ["band_flux_jy", "C_AA_PER_S"]

C_AA_PER_S = 2.99792458e18              # speed of light [Angstrom / s]

_EXPANSION_CACHE: dict = {}
_EXPANSION_CACHE_MAX = 64


def _expansion(names, filter_set):
    """``(nu_flat, w_flat, counts, starts, norms)`` for these per-observation filter names."""
    names = np.asarray(names).astype(str)
    key = (filter_set.hash, names.shape, names.tobytes())
    hit = _EXPANSION_CACHE.get(key)
    if hit is None:
        idx = np.array([filter_set.index(n) for n in names], dtype=int) if names.size else \
            np.zeros(0, dtype=int)
        counts = np.array([filter_set.nodes[k].size for k in idx], dtype=int)
        lam = np.concatenate([filter_set.nodes[k] for k in idx]) if idx.size else np.zeros(0)
        w = np.concatenate([filter_set.weights[k] for k in idx]) if idx.size else np.zeros(0)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if idx.size else np.zeros(0, int)
        hit = (C_AA_PER_S / lam, w, counts, starts, filter_set.norms[idx])
        if len(_EXPANSION_CACHE) >= _EXPANSION_CACHE_MAX:
            _EXPANSION_CACHE.pop(next(iter(_EXPANSION_CACHE)))
        _EXPANSION_CACHE[key] = hit
    return hit


def band_flux_jy(flux_jy, times, names, filter_set):
    """Band flux [Jy] per observation. ``flux_jy(t_flat, nu_flat_hz)`` is the model, in Jy.

    ``times`` and ``names`` (filter names in ``filter_set``, one per observation) have one entry
    per observation. ``flux_jy`` receives observer-frame times repeated once per node and the
    matching observer-frame frequencies, as two equal-length arrays, and must return F_nu at each
    pair. An observation any of whose nodes comes back NaN or infinite returns exactly 0.0, which
    is what a sampler must reject and what the redback adapter has always returned for them.
    """
    times = np.asarray(times, dtype=float)
    nu, w, counts, starts, norms = _expansion(names, filter_set)
    if times.shape != counts.shape:
        raise ValueError(f"times {times.shape} and bands {counts.shape} must have the same shape")
    if times.size == 0:
        return np.zeros(0)
    f = np.asarray(flux_jy(np.repeat(times, counts), nu), dtype=float)
    bad = ~np.isfinite(f)
    out = np.add.reduceat(w * np.where(bad, 0.0, f), starts) / norms
    if bad.any():
        out[np.add.reduceat(bad.astype(int), starts) > 0] = 0.0
    return out
