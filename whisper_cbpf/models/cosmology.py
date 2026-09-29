"""Luminosity distance as a function of redshift that a JAX model can trace.

A model whose redshift is a free parameter needs the distance to move with it on every likelihood
call. astropy cannot run inside ``jax.jit``, so the Planck18 distance is tabulated once (with
astropy, on first use) and interpolated in log-log space: ``ln d_L`` against ``ln z`` on 2000
geometric nodes over ``[Z_MIN, Z_MAX]``. That is linear interpolation of a nearly straight line
(``d_L ~ c z / H0`` at low redshift), so the error is set by the curvature alone: at most
8.3e-7 mag against astropy's own Planck18 over the whole range (measured on 200,000 random
redshifts; ``tests/test_free_texp_redshift.py`` holds it to 1e-5 mag). redback's models read the
same astropy Planck18, so a model built on this table and a redback model at the same redshift
differ by that much and no more.

The same function serves NumPy and JAX (``xp=``), so a CPU check and a GPU fit use one table.
"""
from __future__ import annotations

import numpy as np

__all__ = ["luminosity_distance_cm", "Z_MIN", "Z_MAX", "COSMOLOGY_NAME"]

#: Redshift range of the table. Below ``Z_MIN`` and above ``Z_MAX`` the distance is held at the
#: edge value (a traced call cannot raise); :func:`luminosity_distance_cm` raises there on concrete
#: input instead.
Z_MIN, Z_MAX = 1e-4, 10.0
#: The cosmology the table is built from (astropy's), recorded in model descriptions.
COSMOLOGY_NAME = "Planck18"
_N_NODES = 2000
_TABLE = None


def _table():
    """``(ln z, ln d_L[cm])`` on the nodes, built with astropy once per process."""
    global _TABLE
    if _TABLE is None:
        import astropy.units as u
        from astropy.cosmology import Planck18

        z = np.geomspace(Z_MIN, Z_MAX, _N_NODES)
        _TABLE = (np.log(z), np.log(Planck18.luminosity_distance(z).to_value(u.cm)))
    return _TABLE


def luminosity_distance_cm(z, xp=np):
    """Planck18 luminosity distance in cm, from a table that JAX can trace.

    Parameters
    ----------
    z : float or array
        Redshift, ``Z_MIN <= z <= Z_MAX`` (1e-4 to 10). A JAX tracer is accepted with
        ``xp=jax.numpy``; the result is then differentiable in ``z``.
    xp : module
        ``numpy`` (default) or ``jax.numpy``.

    Returns
    -------
    float or array
        ``d_L`` in cm, within 1e-6 mag (as ``5 log10 d_L``) of ``astropy.cosmology.Planck18``.

    Raises
    ------
    ValueError
        With ``xp=numpy``, when a redshift lies outside ``[Z_MIN, Z_MAX]``. A traced call holds
        the edge value instead, since it cannot raise.

    Examples
    --------
    >>> from whisper_cbpf.models.cosmology import luminosity_distance_cm
    >>> round(luminosity_distance_cm(0.1) / 3.0857e24)        # Mpc
    476
    >>> import jax, jax.numpy as jnp
    >>> f = jax.jit(lambda z: luminosity_distance_cm(z, xp=jnp))
    >>> bool(abs(float(f(0.1)) / luminosity_distance_cm(0.1) - 1) < 1e-6)
    True
    """
    ln_z, ln_dl = _table()
    if xp is np:
        zz = np.asarray(z, dtype=float)
        if np.any(~np.isfinite(zz)) or np.any(zz < Z_MIN) or np.any(zz > Z_MAX):
            raise ValueError(
                f"redshift {zz} is outside the luminosity-distance table [{Z_MIN:g}, {Z_MAX:g}]. "
                f"Keep the redshift prior inside it (a redshift of 0 has no luminosity distance: "
                f"pass the distance itself, dl_cm=, with the redshift fixed).")
        out = np.exp(np.interp(np.log(zz), ln_z, ln_dl))
        return float(out) if out.ndim == 0 else out
    zc = xp.clip(z, Z_MIN, Z_MAX)
    return xp.exp(xp.interp(xp.log(zc), xp.asarray(ln_z), xp.asarray(ln_dl)))
