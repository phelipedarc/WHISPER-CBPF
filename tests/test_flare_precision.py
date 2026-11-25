"""``flare_jax``'s host-side hooks follow the session's precision.

``predict_numpy`` (the ``Model.predict`` the CPU samplers call) and ``predict_torch`` (the SNPE
simulator hook) cast the epochs -- and ``predict_torch`` the parameters -- to float32 whatever the
session asked for. On an MJD clock float32 resolves only 0.0039 d, so an x64 session still got a
single-precision flare there: 2.3e-2 relative against the float64 NumPy twin on a 0.3-d rise at
MJD 59000 (``_flare_spec.flare_flux_numpy``), in both hooks. ``make_log_prob_jax`` had the same
cast and lost it earlier.

Run in a fresh interpreter with ``JAX_ENABLE_X64=1``: x64 is a process-wide flag other test files
set and unset by import order, and the float32 session is covered by every other flare test.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("jax")

_CODE = """
import sys, numpy as np
from whisper_cbpf.models.jax import flare
from whisper_cbpf.models.jax._flare_spec import flare_flux_numpy
p = dict(log_amp=0.5, log_sigma=np.log(0.3), log_tau=1.2, t0=59000.37)
t = 59000.37 + np.array([-0.9, -0.41, -0.07, 0.0, 0.013, 0.6, 3.3])
ref = flare_flux_numpy(p, t)
got = flare.predict_numpy(p, t)
out = {"numpy_dtype": str(got.dtype), "numpy_rel": float(np.max(np.abs(got - ref) / ref))}
if sys.argv[1] == "torch":
    import torch
    theta = torch.tensor([[p[k] for k in flare.PARAMETERS]] * 2, dtype=torch.float64)
    got_t = flare.predict_torch(theta, t, device="cpu").numpy()
    out["torch_dtype"] = str(got_t.dtype)
    out["torch_rel"] = float(np.max(np.abs(got_t - ref[None, :]) / ref[None, :]))
print(repr(out))
"""


def _run(mode):
    env = {**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"}
    return eval(subprocess.run([sys.executable, "-c", _CODE, mode], capture_output=True,
                               text=True, check=True, env=env).stdout)


def test_predict_numpy_is_float64_in_an_x64_session():
    got = _run("numpy")
    assert got["numpy_dtype"] == "float64", got
    assert got["numpy_rel"] < 1e-12, got


def test_predict_torch_is_float64_in_an_x64_session():
    pytest.importorskip("torch")
    got = _run("torch")
    assert got["torch_dtype"] == "float64", got
    assert got["torch_rel"] < 1e-12, got
