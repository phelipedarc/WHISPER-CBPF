"""Float32 forward-pass probe for the kilonova port. NOT a test module.

Run by test_goldens.py::test_precision_matrix_kilonova_float32_clean_on_16_draws in a
subprocess with JAX_ENABLE_X64=0, because x64 is process-wide and the rest of the T0
suite requires it ON. Usage: python _f32_probe.py GOLDENS_DIR OUT_NPZ
"""
import sys
from pathlib import Path

import numpy as np

import jax

assert not jax.config.jax_enable_x64, "probe must run with x64 OFF"
import jax.numpy as jnp  # noqa: E402

from whisper_cbpf.models.jax import kilonova as kn  # noqa: E402

goldens, out = Path(sys.argv[1]), Path(sys.argv[2])
gk = np.load(goldens / "kilonova_one_component.npz", allow_pickle=False)
fs = np.load(goldens / "filterset_t0.npz", allow_pickle=False)

P = gk["params"][:16]
z, dl = 0.01, 1.34e26
t_obs = np.geomspace(0.5, 8.0, 12)
band_idx = np.tile(np.arange(fs["trans"].shape[0]), 3)

W, N = kn.ab_weights(fs["lam"], fs["trans"])
lam_j = jnp.asarray(fs["lam"])
bidx = jnp.asarray(band_idx)

mag = np.zeros((P.shape[0], t_obs.size), dtype=np.float32)
L = np.zeros_like(mag)
temp = np.zeros_like(mag)
for i in range(P.shape[0]):
    t_src = kn.source_time_s(t_obs, z)
    assert t_src.dtype == jnp.float32, t_src.dtype
    li, ti, _ = kn.bolometric(t_src, float(P[i, 0]), float(P[i, 1]),
                              float(P[i, 2]), float(P[i, 3]))
    L[i], temp[i] = np.asarray(li), np.asarray(ti)
    mag[i] = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam_j, z, dl,
                                        float(P[i, 0]), float(P[i, 1]),
                                        float(P[i, 2]), float(P[i, 3])))

np.savez(out, mag=mag, L=L, temp=temp, t_obs_days=t_obs, redshift=z, dl_cm=dl,
         band_idx=band_idx, dtype="float32")
print("f32 probe ok:", mag.shape, "finite:", bool(np.all(np.isfinite(mag))))
