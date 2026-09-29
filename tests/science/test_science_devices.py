"""(c) The same fit on the CPU and on the GPU: same answer, and each device repeats itself.

One simulated LSST supernova alert is fitted with ``emcee_jax`` (the sampler ``sampler="auto"``
picks on a GPU) at its default settings, twice per device, each device in its own fresh process
(as a user runs it):

- **agreement**: every posterior median agrees between the CPU and the GPU within Monte Carlo
  error, ``|median_cpu - median_gpu| <= 4 sqrt(se_cpu^2 + se_gpu^2)`` with
  ``se = 1.2533 sd / sqrt(ESS)`` (the standard error of a median), ESS from the walkers'
  integrated autocorrelation time;
- **determinism**: on each device two runs at the same seed give identical draws. On the GPU this
  is checked twice: as a user runs it, and with XLA's deterministic GPU operations
  (``XLA_FLAGS=--xla_gpu_deterministic_ops=true``).

Needs a GPU; slow (about 6 fits; the CPU pair takes about 4 minutes).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import _studies  # noqa: E402

HERE = Path(__file__).parent
DETERMINISTIC = "--xla_gpu_deterministic_ops=true"

CODE = textwrap.dedent("""
    import json, sys, time, warnings
    sys.path.insert(0, {here!r})
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import whisper_cbpf as wp
    import _sim, _studies
    rng = np.random.default_rng(404)
    alert = _sim.simulate_alert("arnett", rng, redshift=0.08, min_detections=15, min_bands=3)
    lc = _studies.alert_lc(alert, explosion_known=True)
    model = _sim.build_model("arnett", lc.band, alert["redshift"])
    arrays, meta = {{}}, {{"device": jax.devices()[0].platform, "runs": []}}
    for rep in range(2):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = wp.fit(lc, model, sampler="emcee_jax", seed=7)
        meta["runs"].append({{"wall_s": time.perf_counter() - t0,
                             "max_log_likelihood": float(res.max_log_likelihood)}})
        arrays[f"samples{{rep}}"] = res.samples[model.parameters].to_numpy()
        arrays[f"chain{{rep}}"] = np.asarray(res.samples_by_chain)
    meta["parameters"] = list(model.parameters)
    np.savez(sys.argv[1], **arrays)
    print(json.dumps(meta))
""")


def _run(device, out, xla_flags=None):
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "XLA_FLAGS")}
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    if xla_flags:
        env["XLA_FLAGS"] = xla_flags
    proc = subprocess.run([sys.executable, "-c", CODE.format(here=str(HERE)), str(out)],
                          env=env, capture_output=True, text=True, timeout=7200)
    assert proc.returncode == 0, proc.stderr[-3000:]
    meta = json.loads(proc.stdout.strip().splitlines()[-1])
    assert meta["device"] == device, meta
    arrays = dict(np.load(out))
    meta["identical"] = bool(np.array_equal(arrays["samples0"], arrays["samples1"]))
    meta["wall_s"] = [r["wall_s"] for r in meta["runs"]]
    return meta, arrays


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not _studies.gpu_for_subprocess():
        pytest.skip("needs a GPU visible to a fresh process")
    d = tmp_path_factory.mktemp("devices")
    out = {"cpu": _run("cpu", d / "cpu.npz"), "gpu": _run("gpu", d / "gpu.npz"),
           "gpu_deterministic": _run("gpu", d / "gpu_det.npz", DETERMINISTIC)}
    _studies.record("devices", {k: {"identical": m["identical"], "wall_s": m["wall_s"],
                                    "max_log_likelihood": m["runs"][0]["max_log_likelihood"]}
                                for k, (m, _a) in out.items()})
    return out


def _ess(chain):
    """Effective sample size of each parameter from a (walkers, draws, params) chain."""
    import emcee

    x = np.swapaxes(np.asarray(chain, dtype=float), 0, 1)          # (draws, walkers, params)
    tau = emcee.autocorr.integrated_time(x, tol=0, quiet=True)
    return x.shape[0] * x.shape[1] / np.maximum(tau, 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("gpu_run", ["gpu", "gpu_deterministic"])
def test_cpu_and_gpu_posteriors_agree_within_monte_carlo_error(runs, gpu_run):
    (mc, ac), (mg, ag) = runs["cpu"], runs[gpu_run]
    names = mc["parameters"]
    med_c, med_g = np.median(ac["samples0"], 0), np.median(ag["samples0"], 0)
    se_c = 1.2533 * ac["samples0"].std(0) / np.sqrt(_ess(ac["chain0"]))
    se_g = 1.2533 * ag["samples0"].std(0) / np.sqrt(_ess(ag["chain0"]))
    z = np.abs(med_c - med_g) / np.sqrt(se_c ** 2 + se_g ** 2)
    _studies.record("devices_agreement", {"gpu_run": gpu_run,
                                          "median_z": dict(zip(names, z.tolist()))})
    assert np.all(z <= 4.0), dict(zip(names, z.round(2).tolist()))


@pytest.mark.slow
def test_the_cpu_repeats_itself_at_a_fixed_seed(runs):
    assert runs["cpu"][0]["identical"]


@pytest.mark.slow
def test_the_gpu_repeats_itself_at_a_fixed_seed(runs):
    assert runs["gpu"][0]["identical"], (
        "two GPU fits at the same seed differ: the start's gradient climb is not bitwise "
        f"reproducible on the GPU. XLA_FLAGS={DETERMINISTIC} makes it so (see the next test).")


@pytest.mark.slow
def test_the_gpu_repeats_itself_with_deterministic_xla_ops(runs):
    assert runs["gpu_deterministic"][0]["identical"]
