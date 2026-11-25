"""``FilterSet``: the one object that carries a band integral to both backends.

A FilterSet is, per band, a list of wavelength nodes and positive weights ``w_k`` such that

    F_band = sum_k w_k F_nu(lam_k) / sum_k w_k            m_b = -2.5 log10(F_band / 3631 Jy)

approximates the photon-counting band integral of ``F_nu``. It is plain NumPy (tuples of float64
arrays, a string, a dict), so it pickles by value into a sampler's worker processes, and it is
built once: the CPU path (:func:`whisper_cbpf.synphot.band_flux_jy`) reads its nodes directly and
the JAX kernels read :meth:`FilterSet.to_legacy`, the ``{'lam', 'trans'}`` dict their
``filter_set=`` argument has always taken. So a CPU and a GPU model built on the same FilterSet
compute the same band integral, and :attr:`FilterSet.hash` proves it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["FilterSet", "filter_set_for", "shipped_filter_names", "LIBRARY_DIR"]

#: Where the shipped FilterSets live (package data).
LIBRARY_DIR = Path(__file__).resolve().parent / "data"

#: The shipped FilterSets: file stem -> the sncosmo names it holds (Gauss-16, sncosmo's curves).
LIBRARY = {
    "lsst": ("lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"),
    "ztf": ("ztfg", "ztfr", "ztfi"),
    "sdss": ("sdssu", "sdssg", "sdssr", "sdssi", "sdssz"),
}


class FilterSet:
    """Per-band quadrature nodes [Angstrom, observer frame] and weights for the band integral.

    Build one with :func:`whisper_cbpf.synphot.gauss_rule` or :func:`filter_set_for`; wrap a
    grid-rule dict with :meth:`from_legacy`. ``provenance`` maps each name to where its curve came
    from (source, sha256 of the curve, the self-check error).
    """

    def __init__(self, names, nodes, weights, *, rule, provenance=None):
        self.names = tuple(str(n) for n in names)
        self.nodes = tuple(np.ascontiguousarray(x, dtype=np.float64) for x in nodes)
        self.weights = tuple(np.ascontiguousarray(w, dtype=np.float64) for w in weights)
        self.rule = str(rule)
        self.provenance = {str(k): dict(v) for k, v in dict(provenance or {}).items()}
        if not (len(self.names) == len(self.nodes) == len(self.weights)):
            raise ValueError("names, nodes and weights must have one entry per band")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate band names in {self.names}")
        for name, x, w in zip(self.names, self.nodes, self.weights):
            if x.ndim != 1 or x.shape != w.shape or x.size == 0:
                raise ValueError(f"band {name!r}: nodes and weights must be equal-length 1-D")
            if np.any(w < 0) or not np.all(np.isfinite(w)) or not np.all(x > 0):
                raise ValueError(f"band {name!r}: weights must be finite and >= 0, nodes > 0")

    # --- identity --------------------------------------------------------------------------------
    @property
    def hash(self):
        """sha256 of the names, nodes and weights: equal hashes, identical band integrals.

        Computed once: a FilterSet is not meant to be modified after it is built.
        """
        if getattr(self, "_hash", None) is None:
            h = hashlib.sha256()
            for name, x, w in zip(self.names, self.nodes, self.weights):
                h.update(name.encode() + b"\0" + x.tobytes() + w.tobytes())
            self._hash = h.hexdigest()
        return self._hash

    @property
    def norms(self):
        """``sum_k w_k`` per band, i.e. ``int T dlam/lam`` [dimensionless]."""
        return np.array([w.sum() for w in self.weights])

    def index(self, name):
        """Position of ``name`` in :attr:`names`."""
        try:
            return self.names.index(str(name))
        except ValueError:
            raise KeyError(f"{name!r} is not in this FilterSet {self.names}") from None

    def select(self, names):
        """A FilterSet holding only ``names``, in that order."""
        idx = [self.index(n) for n in names]
        return FilterSet([self.names[i] for i in idx], [self.nodes[i] for i in idx],
                         [self.weights[i] for i in idx], rule=self.rule,
                         provenance={self.names[i]: self.provenance.get(self.names[i], {})
                                     for i in idx})

    @classmethod
    def concat(cls, sets):
        """One FilterSet from several (names must not repeat). The rule reads ``"a+b"`` if mixed."""
        sets = list(sets)
        rules = sorted({s.rule for s in sets})
        prov = {}
        for s in sets:
            prov.update(s.provenance)
        return cls([n for s in sets for n in s.names], [x for s in sets for x in s.nodes],
                   [w for s in sets for w in s.weights], rule="+".join(rules), provenance=prov)

    def __repr__(self):
        return (f"FilterSet({list(self.names)}, rule={self.rule!r}, "
                f"nodes={[x.size for x in self.nodes]})")

    def __eq__(self, other):
        return isinstance(other, FilterSet) and self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)

    # --- persistence -----------------------------------------------------------------------------
    def save(self, path):
        """Write an ``.npz``: the arrays, plus the rule and provenance as JSON. No pickle."""
        arrays = {}
        for i, (x, w) in enumerate(zip(self.nodes, self.weights)):
            arrays[f"nodes_{i}"] = x
            arrays[f"weights_{i}"] = w
        meta = {"names": list(self.names), "rule": self.rule, "provenance": self.provenance,
                "hash": self.hash}
        np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)

    @classmethod
    def load(cls, path):
        """Read what :meth:`save` wrote, and check the stored hash."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            n = len(meta["names"])
            fs = cls(meta["names"], [data[f"nodes_{i}"] for i in range(n)],
                     [data[f"weights_{i}"] for i in range(n)], rule=meta["rule"],
                     provenance=meta["provenance"])
        if fs.hash != meta["hash"]:
            raise ValueError(f"{path}: the arrays do not match the stored hash")
        return fs

    # --- the JAX factories' format -----------------------------------------------------------------
    def to_legacy(self):
        """The ``{'lam', 'trans', 'names'}`` dict the JAX factories' ``filter_set=`` takes.

        All nodes on one sorted grid, and ``trans`` defined so that
        :func:`whisper_cbpf.synphot.grid_rule.ab_weights` gives back each band's own weights at its
        own nodes and zero elsewhere (round trip ~2e-16 relative). No JAX kernel changes: it
        integrates over ``16 x n_bands`` points instead of 2000.
        """
        from .grid_rule import _cell_widths

        lam = np.unique(np.concatenate(self.nodes))
        dlam = _cell_widths(lam)
        trans = np.zeros((len(self.names), lam.size))
        for k, (x, w) in enumerate(zip(self.nodes, self.weights)):
            i = np.searchsorted(lam, x)
            trans[k, i] = w * lam[i] / dlam[i]
        return {"lam": lam, "trans": trans, "names": np.array(self.names)}

    @classmethod
    def from_legacy(cls, fs):
        """Wrap a grid-rule dict (``make_filter_set`` output, or :meth:`to_legacy`'s) as a FilterSet.

        The nodes of a band are the grid points where its weight is positive, so the CPU and the
        JAX path integrate a grid-rule filter set identically too.
        """
        from .grid_rule import _cell_widths

        lam = np.asarray(fs["lam"], dtype=np.float64)
        trans = np.atleast_2d(np.asarray(fs["trans"], dtype=np.float64))
        w = trans / lam[None, :] * _cell_widths(lam)[None, :]
        names = [str(n) for n in fs.get("names", [f"band{k}" for k in range(trans.shape[0])])]
        keep = [row > 0 for row in w]
        return cls(names, [lam[k] for k in keep], [row[k] for row, k in zip(w, keep)],
                   rule="grid")


def shipped_filter_names():
    """Every filter name the package ships a Gauss-16 FilterSet for."""
    return tuple(n for names in LIBRARY.values() for n in names)


_LIBRARY_CACHE: dict = {}
_BUILT_CACHE: dict = {}          # (name, n_nodes) -> single-band FilterSet
_SET_CACHE: dict = {}            # (names, n_nodes) -> FilterSet


def _library():
    """The shipped FilterSets, loaded once: name -> single-band FilterSet."""
    if not _LIBRARY_CACHE:
        for stem in LIBRARY:
            fs = FilterSet.load(LIBRARY_DIR / f"{stem}.npz")
            for name in fs.names:
                _LIBRARY_CACHE[name] = fs.select([name])
    return _LIBRARY_CACHE


def filter_set_for(names, n_nodes=None):
    """The default FilterSet for ``names``: Gauss-16 per band, shipped where possible. Memoised.

    Names in :func:`shipped_filter_names` come from the package's own files (no sncosmo needed);
    anything else is built from its curve by :func:`whisper_cbpf.synphot.gauss_rule`. The shipped
    sets ARE Gauss-16 on sncosmo's curves (``tests/test_synphot_rules.py`` rebuilds and compares
    them), so the two routes agree. ``n_nodes`` other than 16 always builds. A repeated name is
    kept once, in first-seen order.
    """
    from .gauss_rule import DEFAULT_N_NODES, gauss_rule

    n_nodes = DEFAULT_N_NODES if n_nodes is None else int(n_nodes)
    unique = tuple(dict.fromkeys(str(n) for n in names))    # unique, order kept
    hit = _SET_CACHE.get((unique, n_nodes))
    if hit is not None:
        return hit
    parts = []
    for name in unique:
        key = (name, n_nodes)
        if key not in _BUILT_CACHE:
            lib = _library() if n_nodes == DEFAULT_N_NODES else {}
            _BUILT_CACHE[key] = lib[name] if name in lib else gauss_rule([name], n_nodes)
        parts.append(_BUILT_CACHE[key])
    hit = _SET_CACHE[(unique, n_nodes)] = FilterSet.concat(parts)
    return hit


def build_library(out_dir=None):
    """Rebuild the shipped FilterSets from sncosmo's curves. Needs sncosmo; run by maintainers.

    Writes ``lsst.npz``, ``ztf.npz`` and ``sdss.npz`` (Gauss-16) into ``out_dir`` (default: the
    package's own ``data/``). Each band's provenance carries the sncosmo version and the sha256 of
    the curve it was built from; ``tests/test_synphot_rules.py`` checks both against the installed
    sncosmo.
    """
    from .gauss_rule import gauss_rule

    out = Path(out_dir) if out_dir is not None else LIBRARY_DIR
    out.mkdir(parents=True, exist_ok=True)
    for stem, names in LIBRARY.items():
        gauss_rule(names).save(out / f"{stem}.npz")
