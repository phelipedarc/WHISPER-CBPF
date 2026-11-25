"""Shared context for the T3 tier: the model exactly as a sampler sees it.

Everything the three T3 test modules (prior predictive, continuity, likelihood parity)
must AGREE on lives here, so a discrepancy between them is a finding about the model and
never about two test files quietly using different filters, distances or epoch grids.

FLOAT64 FIRST. The TDE engine refuses float32 (CHANGE 8 in whisper_cbpf/models/tde.py), and
the config flag only affects arrays created AFTER it is set -- so this module flips it at
import time, before any jnp array exists. Import this before anything that touches jax.

Run from the WHISPER-CBPF repo root, with the ``[gpu,models,dev]`` extras installed and the
bootstrap sourced so ``whisper_cbpf`` and jax are both importable::

    export JAX_PLATFORMS=cpu
    source "$(whisper-cbpf-env)"
    python -u tests/t3_inference/test_prior_predictive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `whisper_cbpf` importable when a test file is executed as a script: python puts the
# SCRIPT's directory on sys.path, not the cwd, so the repo root (two levels up from
# tests/t3_inference/) has to be added by hand. Harmless under pytest.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax

jax.config.update("jax_enable_x64", True)          # BEFORE any array -- see module docstring

import jax.numpy as jnp
import numpy as np

DAY = 86400.0

# --- the shared dataset context --------------------------------------------------------
#: One redshift for every T3 dataset. z = 0.05 is the value the TDE module's own
#: measurements are quoted at (CHANGE 7: the (1+z) factor is worth 0.053 mag here).
Z = 0.05

#: Planck18 luminosity distance at Z, cgs -- computed from astropy when available because
#: redback's own `cooling_envelope` photometry uses Planck18 internally, and likelihood
#: parity (T3.3) is only meaningful if both sides use the SAME distance. The literal
#: fallback is the same number, frozen, for environments without astropy.
try:
    from astropy.cosmology import Planck18

    DL_CM = float(Planck18.luminosity_distance(Z).cgs.value)
except Exception:                                   # pragma: no cover
    DL_CM = 7.033556e26

#: The two ZTF bands every T3 dataset observes in.
BANDS = ["ztfg", "ztfr"]

#: Filter grid size. 1000 is the measured 0.127-mmag point of the exact-cell-integral
#: scheme (kilonova.py, "optional extras"), far below the 0.05-mag noise these tests use.
N_WAVE = 1000

_FILTERS = None


def filters():
    """(lam, weights, norms) for BANDS, built once per process (sncosmo is slow)."""
    global _FILTERS
    if _FILTERS is None:
        from whisper_cbpf.models.jax import kilonova as kn

        fs = kn.make_filter_set(BANDS, n_wave=N_WAVE)
        W, N = kn.ab_weights(fs["lam"], fs["trans"])
        _FILTERS = (jnp.asarray(fs["lam"]), W, N, fs)
    return _FILTERS


# --- vectorised prior draws ------------------------------------------------------------
def sample_prior_matrix(prior, names, n, rng):
    """(n, len(names)) draws from a whisper Prior, vectorised.

    ``Prior.sample`` is one python-loop draw per call; 20,000 of those is pointless
    overhead. The distributions here are boxes, so the vectorisation is exact: Uniform
    draws uniformly on [lo, hi], LogUniform uniformly in log. Anything else refuses --
    silently substituting a shape is how priors go wrong (see nuts_gpu._numpyro_priors).
    """
    cols = []
    for nm in names:
        d = prior.distributions[nm]
        lo, hi = (float(x) for x in d.bounds)
        kind = type(d).__name__
        if kind == "Uniform":
            cols.append(rng.uniform(lo, hi, n))
        elif kind == "LogUniform":
            cols.append(np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
        else:
            raise NotImplementedError(
                f"sample_prior_matrix cannot vectorise a {kind} prior on {nm!r}")
    return np.stack(cols, axis=1)


# --- chunked vmap ----------------------------------------------------------------------
def chunked_vmap(fn, theta, chunk=256, desc=None):
    """vmap(fn) over rows of theta, in fixed-size chunks, returning stacked numpy.

    Fixed-size: the last partial chunk is PADDED to `chunk` rows (repeating row 0) and the
    padding dropped after, so XLA compiles exactly one batch shape instead of two.
    `fn` may return a single array or a tuple of arrays.
    """
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.shape[0]
    f = jax.jit(jax.vmap(fn))
    outs = None
    for i0 in range(0, n, chunk):
        block = theta[i0:i0 + chunk]
        pad = chunk - block.shape[0]
        if pad:
            block = np.concatenate([block, np.repeat(theta[:1], pad, axis=0)], axis=0)
        res = f(jnp.asarray(block))
        res = res if isinstance(res, tuple) else (res,)
        res = [np.asarray(r)[:chunk - pad if pad else chunk] for r in res]
        if outs is None:
            outs = [[] for _ in res]
        for acc, r in zip(outs, res):
            acc.append(r)
        if desc and (i0 // chunk) % 20 == 0:
            print(f"    {desc}: {min(i0 + chunk, n)}/{n}", flush=True)
    stacked = tuple(np.concatenate(a, axis=0) for a in outs)
    return stacked if len(stacked) > 1 else stacked[0]


# --- epoch grids -----------------------------------------------------------------------
def epochs_two_bands(t_lo, t_hi, n_epochs, spacing="geom"):
    """(t_obs_days, band_idx) with every epoch observed in both BANDS, interleaved."""
    t = (np.geomspace if spacing == "geom" else np.linspace)(t_lo, t_hi, n_epochs)
    t_obs = np.repeat(t, 2)
    band_idx = np.tile(np.array([0, 1]), n_epochs)
    return t_obs, band_idx


def gaussian_loglike(model_mags, obs_mags, sigma):
    """The Gaussian log-likelihood every T3 test means when it says logL (constant dropped)."""
    r = (model_mags - obs_mags) / sigma
    return -0.5 * jnp.sum(r * r)
