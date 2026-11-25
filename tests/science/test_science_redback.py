"""(d) The supernova models give redback's light curve, on the CPU and on the GPU.

A small redback consistency check. For each supernova family ``compare`` binds,
seeded draws of redback's own prior are evaluated at an LSST and a ZTF cadence three ways:

- **reference**: redback's own SED (``flux_density`` at many observer-frame frequencies, one call)
  integrated over the sncosmo bandpass on a 1 A grid, photon-counting AB. This is what a
  broadband measurement of redback's model reads, computed without whisper's photometry code.
- **CPU**: whisper's redback adapter (``redback_model``), which integrates redback's SED over the
  filter with whisper's own quadrature.
- **JAX**: whisper's JAX port (``supernova_model``) on the default JAX device (the GPU when there
  is one).

Pass: every point brighter than 30 mag has |CPU - reference| <= 2e-5 mag and |CPU - JAX| <= 1e-10
mag. Redback's constraint priors are switched off here, so the physics is compared at every draw.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("jax")
redback = pytest.importorskip("redback")
sncosmo = pytest.importorskip("sncosmo")

import _sim  # noqa: E402
import _studies  # noqa: E402

#: whisper family name -> redback model name.
SN_FAMILIES = {"arnett": "arnett", "basic_magnetar_powered": "basic_magnetar_powered",
               "shock_cooling_and_arnett": "shock_cooling_and_arnett",
               "csm_shock_and_arnett": "csm_shock_and_arnett"}
#: survey -> (bands, redshift, cadence gap in days).
SURVEYS = {"lsst": (("lsstg", "lsstr", "lssti", "lsstz"), 0.1, (2.0, 4.0)),
           "ztf": (("ztfg", "ztfr"), 0.05, (1.0, 3.0))}
N_DRAWS = 8
TOL_REFERENCE = 2e-5
TOL_CPU_JAX = 1e-10
FAINT = 30.0
C_AA_PER_S = 2.99792458e18


def _mag(flux_jy):
    f = np.asarray(flux_jy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, -2.5 * np.log10(f / 3631.0), np.nan)


def _nodes(band, step_aa=1.0):
    bp = sncosmo.get_bandpass(band)
    lam = np.arange(bp.wave[0], bp.wave[-1] + 0.5 * step_aa, step_aa)
    trans = np.interp(lam, bp.wave, bp.trans, left=0.0, right=0.0)
    cell = np.full(lam.size, step_aa)
    cell[0] = cell[-1] = 0.5 * step_aa
    w = trans / lam * cell
    keep = w > 0
    return lam[keep], w[keep]


def _reference_mags(fn, t, bands, kw, nodes):
    """redback's SED integrated over each observation's bandpass, in one flattened call."""
    t_flat, nu_flat, owner, weights = [], [], [], []
    for i, (ti, b) in enumerate(zip(t, bands)):
        lam, w = nodes[b]
        t_flat.append(np.full(lam.size, ti))
        nu_flat.append(C_AA_PER_S / lam)
        owner.append(np.full(lam.size, i))
        weights.append(w)
    owner, weights = np.concatenate(owner), np.concatenate(weights)
    mjy = np.asarray(fn(np.concatenate(t_flat), output_format="flux_density",
                        frequency=np.concatenate(nu_flat), **kw), dtype=float)
    num = np.bincount(owner, weights=weights * mjy * 1e-3, minlength=len(t))
    den = np.bincount(owner, weights=weights, minlength=len(t))
    return _mag(num / den)


@pytest.mark.parametrize("survey", sorted(SURVEYS))
@pytest.mark.parametrize("family", sorted(SN_FAMILIES))
def test_supernova_band_magnitudes_match_redback_on_cpu_and_jax(family, survey):
    import jax

    import whisper_cbpf as wp
    from whisper_cbpf.models import redback_adapter as ra

    bands_all, z, gap = SURVEYS[survey]
    rng = np.random.default_rng([7, len(family), len(survey)])
    t, bands, _ = _sim.lsst_cadence(rng, 0.5, 80.0, bands=bands_all, gap=gap)
    rb = SN_FAMILIES[family]
    fn = redback.model_library.all_models_dict[rb]
    _dists, pinned, _ = ra._translated_prior(rb)
    uniq = sorted(set(bands.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = ra.redback_model(rb, band_names=uniq, redshift=z, constraint=None)
        gpu = wp.supernova_model(family, uniq, z, ra.redback_luminosity_distance_cm(z, model=rb),
                                 constraint=None)
    nodes = {b: _nodes(b) for b in uniq}
    d_ref, d_jax = [], []
    for _ in range(N_DRAWS):
        draw = cpu.default_prior.sample(rng)
        kw = {**pinned, **{k: float(v) for k, v in draw.items()}, "redshift": z}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = _reference_mags(fn, t, bands, kw, nodes)
            m_cpu = _mag(cpu.predict(draw, t, bands))
            m_jax = _mag(gpu.predict({k: v for k, v in draw.items() if k in gpu.parameters},
                                     t, bands))
        ok = np.isfinite(ref) & (ref < FAINT)
        d_ref.append(np.abs(m_cpu - ref)[ok])
        d_jax.append(np.abs(m_cpu - m_jax)[ok])
    d_ref, d_jax = np.concatenate(d_ref), np.concatenate(d_jax)
    assert d_ref.size > 20, "too few points brighter than 30 mag to compare"
    device = jax.devices()[0].platform
    rec = {"family": family, "survey": survey, "redshift": z, "n_draws": N_DRAWS,
           "n_points": int(d_ref.size), "device": device,
           "cpu_ref_median": float(np.median(d_ref)), "cpu_ref_max": float(d_ref.max()),
           "cpu_jax_median": float(np.nanmedian(d_jax)), "cpu_jax_max": float(np.nanmax(d_jax))}
    _studies.record("redback_consistency", rec)
    assert np.all(np.isfinite(d_jax)), "the JAX model is dark or NaN where redback is bright"
    assert rec["cpu_ref_max"] <= TOL_REFERENCE, rec
    assert rec["cpu_jax_max"] <= TOL_CPU_JAX, rec
