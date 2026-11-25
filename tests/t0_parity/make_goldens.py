"""T0 golden-file generator. Run ONCE, inside the reference container, then commit the npz.

From the repo root, with the ``[gpu,models,dev]`` extras installed::

    export JAX_PLATFORMS=cpu
    source "$(whisper-cbpf-env)"
    python -u tests/t0_parity/make_goldens.py

Everything asserted by ``test_goldens.py`` is computed here FROM REDBACK ONLY -- numpy in,
numpy out, no JAX anywhere in the reference path. The one exception is ``filterset_t0.npz``,
which is NOT a reference output: it is the port's own offline sncosmo filter export
(``kilonova.make_filter_set``), stored so the tests do not need sncosmo at run time.

PROVENANCE. ``redback.__version__`` reports "unknown" in this container, so the version is
recorded from dist-info (``importlib.metadata.version``) and, more bindingly, as the SHA256
of the two transient-model source files actually imported. If either hash changes, the
goldens no longer describe the installed reference and must be regenerated.

PARAMETER GRIDS
    TDE  : N=256 Latin hypercube over redback's gaussianrise_cooling_envelope prior box
           (mbh_6 LogU(0.1,20), stellar_mass LogU(0.1,10), eta LogU(1e-4,0.1),
           alpha LogU(0.1,1), beta U(1,5)) + all 32 box corners + the canonical
           (1, 1, 0.05, 0.1, 1). Engine outputs on redback 1.15.1's own 500-point grid.
    KN   : N=128 Latin hypercube over (mej U(0.01,0.05), vej U(0.1,0.5), kappa U(1,30),
           temperature_floor LogU(100,6000)) + 16 corners + the canonical
           (0.01, 0.2, 1.0, 4000). Evaluated on redback's OWN dense grid
           geomspace(1e-3, 7e6, 300): `_one_component_kilonova_model` runs
           cumulative_trapezoid over exactly the array it is handed, so handing it sparse
           epochs would silently wreck its quadrature (see test_kilonova_vs_redback.py).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time as _time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc

DAY = 86400.0
SEED_TDE = 20260808
SEED_KN = 20260809
Z_TDE = 0.05
NU_PAIR = (6.0e14, 1.2e15)          # alternating per epoch: optical + near-UV
BANDS = ["sdssu", "sdssg", "sdssr", "sdssi"]
N_EPOCH_FD = 16
N_EPOCH_MAG = 12
N_MAG_DRAWS = 64                    # LHS draws carried into the (slow) magnitude golden

OUT_DIR = Path(__file__).resolve().parent.parent / "goldens"


def provenance(extra):
    import redback
    from redback.transient_models import kilonova_models, tde_models

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    info = {
        "redback_dist_version": importlib.metadata.version("redback"),
        "redback_reported_version": getattr(redback, "__version__", "unknown"),
        "redback_tde_models_sha256": sha(tde_models.__file__),
        "redback_kilonova_models_sha256": sha(kilonova_models.__file__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "dtype": "float64",
        "generated_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "generator": "tests/t0_parity/make_goldens.py",
    }
    info.update(extra)
    return json.dumps(info, indent=1)


# ---------------------------------------------------------------- parameter grids
def _scaled(unit, log_lo_hi_flags):
    """unit in [0,1]^d -> physical box. flags: (lo, hi, is_log) per column."""
    cols = []
    for j, (lo, hi, is_log) in enumerate(log_lo_hi_flags):
        u = unit[:, j]
        if is_log:
            cols.append(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
        else:
            cols.append(lo + u * (hi - lo))
    return np.column_stack(cols)


def _corners(log_lo_hi_flags):
    grids = np.meshgrid(*[[lo, hi] for lo, hi, _ in log_lo_hi_flags], indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


TDE_BOX = [(0.1, 20.0, True), (0.1, 10.0, True), (1e-4, 0.1, True),
           (0.1, 1.0, True), (1.0, 5.0, False)]
KN_BOX = [(0.01, 0.05, False), (0.1, 0.5, False), (1.0, 30.0, False),
          (100.0, 6000.0, True)]


def tde_params():
    lhs = qmc.LatinHypercube(d=5, seed=SEED_TDE).random(256)
    p = np.vstack([_scaled(lhs, TDE_BOX), _corners(TDE_BOX),
                   np.array([[1.0, 1.0, 0.05, 0.1, 1.0]])])
    kind = np.array(["lhs"] * 256 + ["corner"] * 32 + ["canonical"])
    return p, kind


def kn_params():
    lhs = qmc.LatinHypercube(d=4, seed=SEED_KN).random(128)
    p = np.vstack([_scaled(lhs, KN_BOX), _corners(KN_BOX),
                   np.array([[0.01, 0.2, 1.0, 4000.0]])])
    kind = np.array(["lhs"] * 128 + ["corner"] * 16 + ["canonical"])
    return p, kind


# ---------------------------------------------------------------- TDE goldens
def make_tde():
    from redback.transient_models.tde_models import _cooling_envelope, cooling_envelope

    import astropy.cosmology as cosmo
    dl = float(cosmo.Planck18.luminosity_distance(Z_TDE).cgs.value)

    p, kind = tde_params()
    n = p.shape[0]
    NT = 500                                        # redback 1.15.1's own grid
    pad = lambda a: np.concatenate([a, np.full(NT - a.size, np.nan)])  # noqa: E731

    k_rb = np.zeros(n, dtype=np.int64)
    tfb = np.zeros(n)
    L = np.full((n, NT), np.nan)
    T = np.full((n, NT), np.nan)
    Rph = np.full((n, NT), np.nan)
    t_fb = np.full((n, NT), np.nan)                 # time_since_fb, source-frame s

    t_obs = np.zeros((n, N_EPOCH_FD))
    freq = np.tile(np.array(NU_PAIR), N_EPOCH_FD // 2)
    fd = np.full((n, N_EPOCH_FD), np.nan)

    for i in range(n):
        out = _cooling_envelope(*p[i])
        k = len(out.time_temp)
        k_rb[i] = k
        tfb[i] = float(out.time_temp[0]) if k else np.nan
        L[i] = pad(np.asarray(out.bolometric_luminosity, dtype=np.float64))
        T[i] = pad(np.asarray(out.photosphere_temperature, dtype=np.float64))
        Rph[i] = pad(np.asarray(out.photosphere_radius, dtype=np.float64))
        t_fb[i] = pad(np.asarray(out.time_since_fb, dtype=np.float64))

        # flux_density at 16 observer-frame epochs inside redback's own span
        tmax = out.time_since_fb[-1] * (1.0 + Z_TDE) / DAY
        lo = max(tmax * 1e-3, min(0.5, 0.3 * tmax))
        t_obs[i] = np.geomspace(lo, 0.9 * tmax, N_EPOCH_FD)
        fd[i] = np.asarray(cooling_envelope(
            t_obs[i], Z_TDE, *p[i], output_format="flux_density", frequency=freq),
            dtype=np.float64)
        if (i + 1) % 50 == 0:
            print(f"  tde engine+fd {i + 1}/{n}")

    prov = provenance({
        "model": "redback.transient_models.tde_models._cooling_envelope / cooling_envelope",
        "n_time": NT, "redshift": Z_TDE, "dl_cm": dl, "cosmology": "Planck18",
        "seed": SEED_TDE, "prior_box": str(TDE_BOX),
        "notes": "arrays padded to n_time with NaN past redback's own [:constraint] slice; "
                 "k_rb is len(time_temp) as returned by redback 1.15.1 (its constraint "
                 "after the max(min(c1,c2),4) floor and the c1==0 -> 5000 sentinel).",
    })
    np.savez_compressed(
        OUT_DIR / "tde_cooling_envelope.npz",
        provenance=prov, params=p, kind=kind, k_rb=k_rb, tfb=tfb,
        L=L, T=T, Rph=Rph, time_since_fb=t_fb,
        redshift=Z_TDE, dl_cm=dl, t_obs_days=t_obs, frequency_hz=freq, flux_density_mjy=fd)
    print(f"wrote tde_cooling_envelope.npz  (k_rb: min {k_rb.min()}, max {k_rb.max()}, "
          f"full-curve draws {(k_rb == NT).sum()})")
    return p, kind, k_rb


def make_tde_magnitudes(p, kind, k_rb):
    from redback.transient_models.tde_models import _cooling_envelope, cooling_envelope

    import astropy.cosmology as cosmo
    dl = float(cosmo.Planck18.luminosity_distance(Z_TDE).cgs.value)

    # subset: the canonical draw + the first N_MAG_DRAWS healthy LHS draws. "Healthy" =
    # terminated on its own (not the full-curve sentinel) with enough curve to interpolate.
    healthy = np.where((k_rb >= 10) & (k_rb < 500))[0]
    idx = [int(np.where(kind == "canonical")[0][0])]
    idx += [int(j) for j in healthy[healthy < 256][:N_MAG_DRAWS]]
    idx = np.array(idx)

    bands = np.array(BANDS * (N_EPOCH_MAG // len(BANDS)))
    band_idx = np.tile(np.arange(len(BANDS)), N_EPOCH_MAG // len(BANDS))
    t_obs = np.zeros((idx.size, N_EPOCH_MAG))
    mag = np.full((idx.size, N_EPOCH_MAG), np.nan)

    for row, i in enumerate(idx):
        out = _cooling_envelope(*p[i])
        tmax = out.time_since_fb[-1] * (1.0 + Z_TDE) / DAY
        lo = max(tmax * 5e-3, min(0.5, 0.3 * tmax))
        t_obs[row] = np.geomspace(lo, 0.85 * tmax, N_EPOCH_MAG)
        mag[row] = np.asarray(cooling_envelope(
            t_obs[row], Z_TDE, *p[i], output_format="magnitude", bands=bands),
            dtype=np.float64)
        if (row + 1) % 10 == 0:
            print(f"  tde magnitude {row + 1}/{idx.size}")

    prov = provenance({
        "model": "redback cooling_envelope(output_format='magnitude')",
        "n_time": 500, "redshift": Z_TDE, "dl_cm": dl, "cosmology": "Planck18",
        "bands": BANDS,
        "notes": "redback evaluates the SED on a 100-point wavelength grid "
                 "(geomspace(100, 60000, 100)), builds an sncosmo TimeSeriesSource and "
                 "splines it; the port integrates the Planck spectrum against the real "
                 "bandpass. The comparison is therefore a DOCUMENTED-DEVIATION bound, "
                 "not a parity gate.",
    })
    np.savez_compressed(
        OUT_DIR / "tde_cooling_envelope_magnitude.npz",
        provenance=prov, params=p[idx], param_rows=idx, k_rb=k_rb[idx],
        redshift=Z_TDE, dl_cm=dl, bands=bands, band_idx=band_idx,
        t_obs_days=t_obs, magnitude=mag)
    print("wrote tde_cooling_envelope_magnitude.npz")


# ---------------------------------------------------------------- kilonova goldens
def make_kilonova():
    from redback.transient_models.kilonova_models import _one_component_kilonova_model

    p, kind = kn_params()
    n = p.shape[0]
    grid = np.geomspace(1e-3, 7e6, 300)             # redback's own default dense grid
    L = np.zeros((n, grid.size))
    T = np.zeros((n, grid.size))
    R = np.zeros((n, grid.size))
    for i in range(n):
        li, ti, ri = _one_component_kilonova_model(
            grid, p[i, 0], p[i, 1], p[i, 2], temperature_floor=p[i, 3])
        L[i], T[i], R[i] = li, ti, ri

    prov = provenance({
        "model": "redback.transient_models.kilonova_models._one_component_kilonova_model",
        "grid": "np.geomspace(1e-3, 7e6, 300) source-frame seconds (redback's default; "
                "its cumulative_trapezoid quadrature runs over exactly this array)",
        "seed": SEED_KN, "prior_box": str(KN_BOX),
        "notes": "compare only at grid NODES (index >= 2: redback copies L[0] = L[1]). "
                 "Parity region: t < 1.4 * t_diff. Documented deviation: past "
                 "~2.66 * t_diff redback's trapezoid is under-resolved and TOO BRIGHT "
                 "(see whisper_cbpf/models/kilonova.py identity 6 and "
                 "tests/test_kilonova_vs_redback.py).",
    })
    np.savez_compressed(
        OUT_DIR / "kilonova_one_component.npz",
        provenance=prov, params=p, kind=kind, grid_s=grid, L=L, T=T, Rph=R)
    print(f"wrote kilonova_one_component.npz  ({n} draws x {grid.size} nodes)")


# ---------------------------------------------------------------- filter set (port input)
def make_filterset():
    """NOT a reference output: the port's own offline sncosmo export, frozen for the tests."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from whisper_cbpf.models.jax.kilonova import make_filter_set

    fs = make_filter_set(BANDS, n_wave=5000)
    np.savez_compressed(
        OUT_DIR / "filterset_t0.npz",
        provenance=provenance({
            "model": "whisper_cbpf.models.jax.kilonova.make_filter_set (port input, NOT a "
                     "redback reference output)",
            "bands": BANDS, "n_wave": 5000}),
        **fs)
    print("wrote filterset_t0.npz")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("generating kilonova goldens ...")
    make_kilonova()
    print("generating TDE goldens ...")
    p, kind, k_rb = make_tde()
    make_tde_magnitudes(p, kind, k_rb)
    make_filterset()
    print("done:", sorted(f.name for f in OUT_DIR.glob("*.npz")))
