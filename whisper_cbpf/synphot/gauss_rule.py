"""The Gauss rule: a band-adapted Gaussian quadrature, 16 nodes per band by default.

A band magnitude is

    m_b = -2.5 log10[ int F_nu(lam) T_b(lam) dlam/lam / (3631 Jy int T_b(lam) dlam/lam) ]

for a photon-counting bandpass ``T_b``. ``dmu_b = T_b(lam) dlam / lam`` is a POSITIVE measure on
the band's support, so it has a Gaussian quadrature of its own: N nodes and N weights, exact for
every polynomial of degree <= 2N-1 against that measure, with every weight positive and every node
inside the band. A model SED is smooth across a band, so 16 nodes reach what the grid rule needs
2000 shared points for (``docs/PHOTOMETRY.md`` has the measurements).

How it is built, per band:

1. **The fine measure.** The tabulated transmission is taken piecewise linear between its own
   nodes (what sncosmo integrates, and what :func:`whisper_cbpf.synphot.grid_rule.make_filter_set`
   integrates exactly), clipped at zero, and every tabulation segment with ``T > 0`` at either end
   gets an 8-point Gauss-Legendre rule. On a segment ``T/lam`` is ``(a + b lam)/lam``, analytic and
   nearly flat, so this discrete measure reproduces ``int p(lam) T dlam/lam`` for any polynomial
   ``p`` of degree <= 31 to float64 round-off. It is also the **fine integral** every accuracy
   claim here is made against.
2. **The rule.** Lanczos on ``diag(lam)`` in that measure's inner product (full
   re-orthogonalisation, on ``lam`` scaled to [-1, 1] over the support) gives the N x N Jacobi
   matrix; its eigenvalues are the nodes and ``mu_0 v_0^2`` the weights (Golub & Welsch 1969).
   ``sum(w) = mu_0 = int T dlam/lam`` to round-off.
3. **The self-check.** The rule is compared with the fine integral on blackbodies from 1500 to
   50 000 K. A curve with structure 16 nodes cannot follow (a notch, a red leak, a two-peaked SVO
   profile) shows up here, and :func:`gauss_rule` warns naming the band, the error and the fix
   (twice the nodes). The number is kept in the FilterSet's provenance either way. Measured: 16
   nodes pass it to 1e-14 mag on every shipped filter and even on a 1000-30000 A top hat; 8
   nodes fail on the top hat (0.05 mmag).

Curves come from sncosmo, or from the SVO Filter Profile Service for an SVO ID
(``"LSST/LSST.g"``), fetched through :mod:`whisper_cbpf.io.svo` (pyphot when installed). The
package also ships Gauss-16 FilterSets for LSST ugrizy, ZTF gri and SDSS ugriz built from
sncosmo's curves (:func:`whisper_cbpf.synphot.filter_set_for`), so the common bands need neither.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np

__all__ = ["gauss_rule", "curve", "fine_measure", "selfcheck_mag", "SELFCHECK_TEMPERATURES",
           "SELFCHECK_TOL_MAG", "DEFAULT_N_NODES"]

#: Nodes per band. 16 is exact to degree 31 and was measured to <= 0.02 mmag on blackbodies,
#: cut-off and line spectra; 8 is enough for smooth SEDs at half the model calls.
DEFAULT_N_NODES = 16

#: Blackbody temperatures [K] of the build-time self-check.
SELFCHECK_TEMPERATURES = np.geomspace(1500.0, 5.0e4, 13)

#: Worst self-check error [mag] before :func:`gauss_rule` warns (0.02 mmag).
SELFCHECK_TOL_MAG = 2.0e-5

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)   # per tabulation segment
_HC_OVER_K_AA = 1.4387768775039337e8                         # h c / k_B [Angstrom K]


def curve(name):
    """``(wave [Angstrom], transmission, provenance)`` for one filter name. NumPy, host only.

    ``name`` is an sncosmo bandpass name (``"lsstg"``, ``"ztfr"``, ``"bessellb"``) or an SVO
    Filter Profile Service ID (``"LSST/LSST.g"``, anything with a ``/``). The provenance records
    where the curve came from and the sha256 of its float64 bytes, so a saved FilterSet can prove
    which curve it was built from.
    """
    name = str(name)
    if "/" in name:
        from ..io import svo

        wave, trans = svo.get_transmission_data(name)
        source = f"SVO Filter Profile Service ({svo.transmission_backend()})"
    else:
        try:
            import sncosmo
        except ImportError as exc:
            raise ImportError(
                f"filter {name!r} is not in whisper's shipped FilterSets (LSST ugrizy, ZTF gri, "
                f"SDSS ugriz), and building it needs sncosmo (pip install sncosmo, or the "
                f"[models] extra), or pass an SVO ID such as 'LSST/LSST.g'.") from exc
        bp = sncosmo.get_bandpass(name)
        wave, trans = bp.wave, bp.trans
        source = f"sncosmo {sncosmo.__version__}"
    wave = np.ascontiguousarray(wave, dtype=np.float64)
    trans = np.ascontiguousarray(trans, dtype=np.float64)
    digest = hashlib.sha256(wave.tobytes() + trans.tobytes()).hexdigest()
    info = {"source": source, "sha256": digest, "n_tabulated": int(wave.size)}
    if "/" not in name:
        info["release"] = _sncosmo_release(name)
    return wave, trans, info


def _sncosmo_release(name):
    """sncosmo's own note on which throughput release a bandpass is (empty if it has none)."""
    try:
        import sncosmo

        for meta in sncosmo.bandpasses._BANDPASSES.get_loaders_metadata():
            if meta.get("name") == name:
                ref = meta.get("reference")
                return "; ".join(str(x) for x in (
                    meta.get("description"), meta.get("dataurl"),
                    ref[1] if isinstance(ref, (tuple, list)) and len(ref) > 1 else ref,
                    meta.get("retrieved") and f"retrieved {meta['retrieved']}") if x)
    except Exception:                           # private registry API; provenance is best effort
        pass
    return ""


def fine_measure(wave, trans):
    """The fine discrete measure of ``T dlam/lam``: ``(lam, m)``, 8 Gauss-Legendre nodes per segment.

    ``T`` is the piecewise-linear interpolant of the tabulation, clipped at zero (a few published
    curves carry tiny negative noise, which a positive measure cannot). ``sum(m)`` is
    ``int T dlam/lam`` and ``sum(m * F(lam))`` the fine band integral of any smooth ``F``.
    """
    wave = np.asarray(wave, dtype=np.float64)
    trans = np.clip(np.asarray(trans, dtype=np.float64), 0.0, None)
    if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0):
        raise ValueError("a filter curve needs >= 2 strictly increasing wavelengths")
    a, b, ta, tb = wave[:-1], wave[1:], trans[:-1], trans[1:]
    keep = (ta > 0) | (tb > 0)
    if not keep.any():
        raise ValueError("the filter curve has no positive transmission")
    a, b, ta, tb = a[keep], b[keep], ta[keep], tb[keep]
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    lam = mid[:, None] + half[:, None] * _GL_NODES[None, :]
    t = ta[:, None] + (tb - ta)[:, None] * (lam - a[:, None]) / (b - a)[:, None]
    m = half[:, None] * _GL_WEIGHTS[None, :] * t / lam
    return lam.ravel(), m.ravel()


def _gauss_from_measure(lam, m, n_nodes):
    """Golub-Welsch: the ``n_nodes``-point Gaussian rule of the discrete measure ``(lam, m)``."""
    lo, hi = float(lam.min()), float(lam.max())
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    s = (lam - c) / h                           # [-1, 1]: keeps the Krylov basis well scaled
    mu0 = float(m.sum())
    q = np.sqrt(m / mu0)
    basis = np.zeros((n_nodes, lam.size))
    alpha = np.zeros(n_nodes)
    beta = np.zeros(n_nodes - 1)
    basis[0] = q
    for k in range(n_nodes):
        v = s * basis[k]
        alpha[k] = basis[k] @ v
        v -= alpha[k] * basis[k]
        if k:
            v -= beta[k - 1] * basis[k - 1]
        for _ in range(2):                      # full re-orthogonalisation, twice is enough
            v -= basis[:k + 1].T @ (basis[:k + 1] @ v)
        if k < n_nodes - 1:
            beta[k] = np.linalg.norm(v)
            basis[k + 1] = v / beta[k]
    jacobi = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    evals, evecs = np.linalg.eigh(jacobi)
    return c + h * evals, mu0 * evecs[0] ** 2


def _blackbody_fnu(lam_aa, temperature):
    """B_nu shape at ``lam_aa`` [Angstrom]: nu^3 / (exp(h nu / k T) - 1), up to a constant."""
    x = _HC_OVER_K_AA / (np.asarray(lam_aa)[..., None] * np.atleast_1d(temperature)[None, :])
    with np.errstate(over="ignore"):
        return (1.0 / lam_aa[..., None] ** 3) / np.expm1(np.minimum(x, 700.0))


def selfcheck_mag(nodes, weights, lam, m, temperatures=SELFCHECK_TEMPERATURES):
    """Worst |m_rule - m_fine| [mag] over blackbodies at ``temperatures``."""
    rule = weights @ _blackbody_fnu(nodes, temperatures) / weights.sum()
    fine = m @ _blackbody_fnu(lam, temperatures) / m.sum()
    return float(np.max(np.abs(2.5 * np.log10(rule / fine))))


def gauss_rule(names, n_nodes=DEFAULT_N_NODES, *, curves=None):
    """A :class:`~whisper_cbpf.synphot.FilterSet` holding the ``n_nodes``-point Gauss rule per band.

    ``names`` are filter names (sncosmo names or SVO IDs, see :func:`curve`). ``curves`` optionally
    maps a name to ``(wave [Angstrom], transmission)`` to use instead of fetching it -- a curve
    of your own, or one from pyphot. Warns when the self-check (module docstring, step 3) exceeds
    :data:`SELFCHECK_TOL_MAG`.
    """
    from .filterset import FilterSet

    n_nodes = int(n_nodes)
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    names = [str(n) for n in names]
    nodes, weights, prov = [], [], {}
    for name in names:
        if curves is not None and name in curves:
            wave, trans = (np.ascontiguousarray(a, dtype=np.float64) for a in curves[name])
            info = {"source": "caller-supplied curve", "n_tabulated": int(wave.size),
                    "sha256": hashlib.sha256(wave.tobytes() + trans.tobytes()).hexdigest()}
        else:
            wave, trans, info = curve(name)
        lam, m = fine_measure(wave, trans)
        x, w = _gauss_from_measure(lam, m, n_nodes)
        err = selfcheck_mag(x, w, lam, m)
        if err > SELFCHECK_TOL_MAG:
            warnings.warn(
                f"gauss_rule: {n_nodes} nodes reproduce the fine band integral of {name!r} only to "
                f"{err * 1e3:.3f} mmag on 1500-50000 K blackbodies (tolerance "
                f"{SELFCHECK_TOL_MAG * 1e3:.3f} mmag). The curve has structure the rule cannot "
                f"follow; rebuild with n_nodes={2 * n_nodes}.", RuntimeWarning, stacklevel=2)
        nodes.append(x)
        weights.append(w)
        prov[name] = dict(info, selfcheck_max_mag=err, int_T_dlam_over_lam=float(m.sum()),
                          support=[float(lam.min()), float(lam.max())])
    return FilterSet(names, nodes, weights, rule=f"gauss{n_nodes}", provenance=prov)
