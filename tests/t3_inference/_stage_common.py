"""Shared plumbing for the STAGED (not-run-here) T3 sampler jobs: SBC and injection-recovery.

Everything here is deliberately identical to what the T3.1-T3.3 tests used --
same bands, same redshift, same distance, same epoch grid, same sigma -- so a
calibration failure found by the staged jobs points at the SAMPLER x MODEL pair,
never at a second copy of the dataset conventions.

These jobs need a GPU. They must run with ``CUDA_VISIBLE_DEVICES`` set and
``source "$(whisper-cbpf-env)"`` done first; see the __main__ docstrings of stage_sbc.py /
stage_injection_recovery.py for the exact command.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax

jax.config.update("jax_enable_x64", True)          # BEFORE any array (TDE requires x64)

import jax.numpy as jnp
import numpy as np

DAY = 86400.0
Z = 0.05
BANDS = ["ztfg", "ztfr"]
N_WAVE = 1000
SIGMA_MAG = 0.05
MAG_FLOOR = 40.0
N_EPOCHS = 24                                       # x2 bands = 48 observations, 1-400 d
AB_ZEROPOINT_JY = 3631.0

try:
    from astropy.cosmology import Planck18

    DL_CM = float(Planck18.luminosity_distance(Z).cgs.value)
except Exception:                                   # pragma: no cover
    DL_CM = 7.033556e26


def require_gpu(allow_cpu=False):
    backend = jax.default_backend()
    if backend != "gpu" and not allow_cpu:
        raise SystemExit(
            f"jax backend is {backend!r}, not 'gpu'. These jobs are sized for an A6000; "
            f"run with CUDA_VISIBLE_DEVICES set and the bootstrap `whisper-cbpf-env` prints "
            f"sourced (and WITHOUT JAX_PLATFORMS=cpu), or pass --allow-cpu to accept "
            f"a ~100x slowdown.")
    return backend


def build_context(n_time=500):
    """(model, mags_fn, t_obs_days, band_idx, prior_matrix_sampler) for the TDE job.

    ``mags_fn(theta_vec) -> (48,) AB magnitudes`` is the exact band-magnitude path the
    T3 tests verified, jit-compiled once; theta order is PARAMETERS_GAUSSIANRISE.
    """
    from whisper_cbpf.models.jax import kilonova as kn
    from whisper_cbpf.models.jax import tde as T
    from whisper_cbpf.models.jax import tde_model

    fs = kn.make_filter_set(BANDS, n_wave=N_WAVE)
    lam = jnp.asarray(fs["lam"])
    W, N = kn.ab_weights(fs["lam"], fs["trans"])

    t_obs = np.repeat(np.geomspace(1.0, 400.0, N_EPOCHS), 2)
    band_idx = np.tile(np.array([0, 1]), N_EPOCHS)
    bands = np.array(BANDS)[band_idx]
    t_j, b_j = jnp.asarray(t_obs), jnp.asarray(band_idx)

    model = tde_model(BANDS, Z, DL_CM, rise="gaussian", n_time=n_time,
                      filter_set=fs, mag_floor=MAG_FLOOR)
    names = list(model.parameters)
    assert names == T.PARAMETERS_GAUSSIANRISE, names

    @jax.jit
    def mags_fn(theta):
        return T.gaussianrise_cooling_envelope_ab_magnitude(
            t_j, b_j, W, N, lam, Z, DL_CM,
            theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6],
            mag_floor=MAG_FLOOR, n_time=n_time, dilation=True)

    def sample_theta(rng):
        d = model.default_prior.sample(rng)
        return np.array([d[nm] for nm in names])

    return model, mags_fn, t_obs, bands, names, sample_theta


def simulate_lightcurve(mags_fn, theta, t_obs, bands, rng):
    """A LightCurve at ``theta`` with sigma = SIGMA_MAG magnitude noise (T3 convention)."""
    from whisper_cbpf.io.schema import LightCurve

    mag_true = np.asarray(mags_fn(jnp.asarray(theta)))
    mag_obs = mag_true + rng.normal(0.0, SIGMA_MAG, mag_true.size)
    flux = AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag_obs)
    flux_err = flux * (np.log(10.0) / 2.5) * SIGMA_MAG
    lc = LightCurve(time=t_obs, band=bands, magnitude=mag_obs,
                    magnitude_err=np.full(mag_obs.size, SIGMA_MAG),
                    flux=flux, flux_err=flux_err, redshift=Z,
                    name="t3_synthetic_tde")
    return lc, mag_obs


def make_log_prob(mags_fn, mag_obs):
    """Gaussian log-likelihood in MAGNITUDE space, sigma = SIGMA_MAG -- the T3 likelihood.

    Jitted against a device copy of the data; a new dataset of the same shape reuses the
    same executable (the data is an ARGUMENT of the jitted core, not a closure constant),
    which is what keeps a 300-replicate loop from paying 300 XLA compiles for this part.
    NumPyro's NUTS kernel is still re-traced per MCMC object; that cost is in the
    per-replicate estimate.
    """
    obs = jnp.asarray(mag_obs)

    @jax.jit
    def _core(theta, data):
        r = (mags_fn(theta) - data) / SIGMA_MAG
        return -0.5 * jnp.sum(r * r)

    return lambda theta: _core(theta, obs)


def fit_one(lc, model, log_prob_fn, *, num_warmup, num_samples, num_chains, seed,
            target_accept_prob=0.8):
    """One NUTS fit through whisper's own registered GPU sampler."""
    from whisper_cbpf.samplers.jax.nuts_gpu import NUTSGPUSampler

    t0 = time.perf_counter()
    res = NUTSGPUSampler().fit(
        lc, model, prior=model.default_prior, log_prob_fn=log_prob_fn,
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        seed=seed, target_accept_prob=target_accept_prob, space="flux")
    res.info["wall_s"] = time.perf_counter() - t0
    return res
