"""The grid rule: a band integral on one shared wavelength grid. Moved here from ``kilonova.py``.

This is the photometry the JAX models used exclusively up to whisper 0.1.0: every bandpass is
exported once onto a geometric ``lam_min``-``lam_max`` grid of ``n_wave`` points, with EXACT cell
integrals of the piecewise-linear transmission (:func:`make_filter_set`), and the band flux is one
matrix product with the photon-counting AB weights (:func:`ab_weights`).

The three functions are moved **verbatim** from :mod:`whisper_cbpf.models.jax.kilonova`, which
re-exports them, so every stored filter set and every t0 golden reproduces bit for bit. Two things
changed around them, neither numerical: ``jax.numpy`` is imported inside :func:`ab_weights` (so
this module, like the rest of :mod:`whisper_cbpf.synphot`, imports without jax), and sncosmo stays
imported inside :func:`make_filter_set`.

The rule is no longer the factories' default: a band-adapted Gauss rule
(:func:`whisper_cbpf.synphot.gauss_rule`) reaches the same accuracy with 16 nodes per band instead
of 2000 shared ones. Passing ``n_wave=`` to a JAX factory still selects this rule. See
``docs/PHOTOMETRY.md``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_filter_set", "ab_weights", "AB_ZEROPOINT"]

AB_ZEROPOINT = 3631.0e-23              # erg s^-1 cm^-2 Hz^-1, as in kilonova.py


def _cell_widths(lam):
    """Trapezoid cell widths for a (possibly non-uniform) 1-D grid.

    np.gradient(lam) is the CENTRAL difference (lam[i+1]-lam[i-1])/2, which is
    exactly the trapezoid weight for INTERIOR nodes -- but at the two ends it
    returns the FULL first/last spacing instead of half of it, i.e. 2x too much.
    That is harmless while T == 0 at both grid edges (true for every sncosmo
    bandpass inside the default 1000-30000 A window: measured max |T| at the
    edges over 33 real filters is exactly 0.0), but it silently biases any band
    the caller clips by narrowing lam_min/lam_max -- measured up to 3.7 mmag for
    uvot::uvw2 on a 1700 A lower edge, 0.85 mmag for 2massks on a 22000 A upper
    edge. Cheap to get right, so get it right.
    """
    lam = np.asarray(lam, dtype=np.float64)
    d = np.empty_like(lam)
    d[1:-1] = 0.5 * (lam[2:] - lam[:-2])
    d[0] = 0.5 * (lam[1] - lam[0])
    d[-1] = 0.5 * (lam[-1] - lam[-2])
    return d


def make_filter_set(band_names, lam_min=1000.0, lam_max=30000.0, n_wave=5000):
    """Export sncosmo bandpasses to plain arrays. Run ONCE, offline, then save.

    `trans` is the CELL-AVERAGED transmission, not a point sample: it is defined
    so that ab_weights' w = trans/lam*dlam equals the EXACT integral of
    T(lam)/lam over that cell, T being the piecewise-linear interpolant of the
    tabulated bandpass (which is precisely what sncosmo integrates). Point
    sampling a filter tabulated at 1 A (LSST, ZTF) onto a ~20 A log grid aliases
    its edges: the error is not smooth in n_wave, it is quasi-random, and it
    reaches 2.1 mmag at n_wave=1000 (uvot::uvw2, 3000 K source). The closed-form
    cell integral removes it -- 0.13 mmag at n_wave=1000, 0.005 mmag at 5000.

    NOTE the semantic change: `trans[k, i]` is no longer T(lam[i]). Plot it against
    a point sample of the bandpass and the two agree to the cell width, but they are
    not the same array, and only this one integrates exactly.

    >>> fs = make_filter_set(['lsstg', 'lsstr', 'lssti'])
    >>> np.savez('filters.npz', **fs)
    """
    import sncosmo
    lam = np.geomspace(lam_min, lam_max, n_wave)
    dlam = _cell_widths(lam)
    # cell edges: geometric midpoints, half-cells at the two ends.
    edges = np.empty(n_wave + 1)
    edges[1:-1] = np.sqrt(lam[:-1] * lam[1:])
    edges[0] = lam[0] ** 2 / edges[1]
    edges[-1] = lam[-1] ** 2 / edges[-2]

    trans = np.zeros((len(band_names), n_wave))
    for k, name in enumerate(band_names):
        bp = sncosmo.get_bandpass(name)
        wb = np.asarray(bp.wave, dtype=np.float64)
        tb = np.asarray(bp.trans, dtype=np.float64)
        if wb[0] < lam_min or wb[-1] > lam_max:
            raise ValueError(
                f"bandpass {name!r} spans {wb[0]:.1f}-{wb[-1]:.1f} A but the grid is "
                f"{lam_min:.1f}-{lam_max:.1f} A. Widen lam_min/lam_max: silently "
                f"truncating a filter moves its AB zeropoint (np.interp would have "
                f"zero-filled the missing wings without a word).")
        # T is piecewise linear on wb, so the cell integral is closed-form:
        # int (icpt + slope*l)/l dl = icpt*ln l + slope*l.
        slope = np.diff(tb) / np.diff(wb)
        icpt = tb[:-1] - slope * wb[:-1]
        cum = np.concatenate([[0.0], np.cumsum(
            icpt * np.log(wb[1:] / wb[:-1]) + slope * (wb[1:] - wb[:-1]))])
        x = np.clip(edges, wb[0], wb[-1])
        j = np.clip(np.searchsorted(wb, x, side='right') - 1, 0, wb.size - 2)
        cell = np.diff(cum[j] + icpt[j] * np.log(x / wb[j]) + slope[j] * (x - wb[j]))
        trans[k] = cell * lam / dlam        # so that trans/lam*dlam == cell, exactly
    return {'lam': lam, 'trans': trans, 'names': np.array(band_names)}


def ab_weights(lam, trans):
    """Photon-counting AB weights: w = T(lam)/lam * dlam, norm = 3631Jy*sum(w).

    sncosmo bandpasses are photon-counting, so counts ~ int f_lam T lam dlam
    = c int f_nu T dlam/lam. Precompute once; the model then needs one matmul.
    """
    import jax.numpy as jnp

    lam = np.asarray(lam, dtype=np.float64)
    trans = np.atleast_2d(np.asarray(trans, dtype=np.float64))
    # _cell_widths, not np.gradient: they differ only in the two END cells, where
    # np.gradient is 2x too wide (see _cell_widths). Identical for any bandpass that
    # is zero at the grid edges, which is why the change is invisible in regression.
    w = trans / lam[None, :] * _cell_widths(lam)[None, :]
    return jnp.asarray(w), jnp.asarray(AB_ZEROPOINT * w.sum(axis=1))
