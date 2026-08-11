"""GPU / JAX environment resolution.

The shell bootstrap (``env.sh``) sets three things that must be in place **before python starts**,
because JAX reads them at import: which device is visible, how much of it to preallocate, and where
the CUDA shared libraries live. This module is the Python-side counterpart -- it reports what
actually happened and explains, in words a user can act on, what to fix when it did not.

Nothing here imports jax at module scope. Importing :mod:`whisper_cbpf` must work on a machine with
no GPU (so ``wp.list_samplers()`` still shows what is available); the failure belongs at ``fit()``
time, with a message that says what to do.
"""
from __future__ import annotations

import os
import shutil
import subprocess

#: Environment variables that must be set before the first ``import jax``.
REQUIRED_ENV = ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")

_HERE = os.path.dirname(os.path.abspath(__file__))


def env_script():
    """Absolute path to this package's ``env.sh``, whatever the package was cloned as."""
    return os.path.join(_HERE, "env.sh")


def env_report():
    """A dict describing the current GPU environment. Never raises, never imports jax."""
    rep = {k: os.environ.get(k) for k in REQUIRED_ENV}
    rep["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")
    rep["env_script"] = env_script()
    try:                                                  # nvidia-smi is optional
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        rep["nvidia_smi"] = out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        rep["nvidia_smi"] = None
    return rep


def require_jax(feature="this sampler"):
    """Import and return ``(jax, jnp)``, or raise with an actionable message.

    Call this at the top of a ``fit()``, never at module import.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires JAX, which is not installed.\n"
            f"  pip install 'jax[cuda12]' numpyro\n"
            f"then source the environment bootstrap so JAX can find the GPU:\n"
            f"  source {env_script()}"
        ) from exc
    return jax, jnp


def check_gpu(strict=False):
    """Return ``(ok, message)`` describing whether JAX can actually see a GPU.

    ``strict=True`` raises instead of returning ``ok=False``. The common failure here is not a
    missing GPU but a missing ``LD_LIBRARY_PATH``: JAX's pip-installed CUDA libraries and torch's
    conda-installed ones live in different trees, and if neither is on the path JAX silently falls
    back to CPU rather than erroring -- so a run that looks fine is 50x slower with no warning.
    """
    jax, _ = require_jax("check_gpu")
    devices = jax.devices()
    kinds = {d.platform for d in devices}
    if "gpu" in kinds or "cuda" in kinds:
        return True, f"JAX sees {len(devices)} GPU device(s): {devices}"

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    msg = (f"JAX is running on CPU ({devices}). This is usually an environment problem, not a "
           f"missing GPU.\n")
    if missing:
        msg += f"  - unset before import: {', '.join(missing)}\n"
    if not os.environ.get("LD_LIBRARY_PATH"):
        msg += ("  - LD_LIBRARY_PATH is empty; JAX cannot find libcudart/libcudnn and falls back "
                "to CPU SILENTLY\n")
    if shutil.which("nvidia-smi") is None:
        msg += "  - nvidia-smi is not on PATH; this may be a container without GPU access\n"
    msg += f"  fix: source {env_script()}  (before starting python)"
    if strict:
        raise RuntimeError(msg)
    return False, msg


def x64_enabled():
    """Whether JAX is in float64 mode. Must be decided BEFORE any jax array is created.

    Production NUTS runs in float64: float32 is fine at survey SNR but fails to mix on a tight
    high-SNR posterior (r-hat up to 19.6 measured on a 5-parameter kilonova fit), while costing
    only ~1.7x at batch 1. Batched simulation is the opposite case -- float64 is 51x slower at
    batch 1e4 on an A6000, whose FP64 rate is 1/32 of FP32.
    """
    jax, _ = require_jax("x64_enabled")
    return bool(jax.config.jax_enable_x64)


# ---------------------------------------------------------------------------------------------
# PRODUCTION RESOURCE BUDGET
#
# These are the limits real runs operate under, kept here so every script reads the same numbers
# instead of each hard-coding its own. A benchmark that quietly takes the whole machine measures a
# configuration nobody can reproduce in production.
#
#: GPUs a production run may use. Multi-GPU NUTS wants num_chains to be a MULTIPLE of this -- a
#: remainder leaves devices idle in the final pmap pass and costs the wall-clock of a full extra
#: pass (measured: 4 chains over 3 GPUs ran SLOWER than 4 chains on 1).
MAX_GPUS = 2

#: Fraction of the host's cores a production run may use.
CPU_FRACTION = 0.60


def n_jobs(fraction=CPU_FRACTION):
    """Worker count at the production CPU budget (60% of cores by default)."""
    return max(1, int(round(fraction * (os.cpu_count() or 1))))


def gpu_list(n=MAX_GPUS, prefer_idle=True):
    """The device ids to expose, as a CUDA_VISIBLE_DEVICES string.

    Prefers GPUs that are actually free: this box is shared, and grabbing a device that already
    holds someone else's 20 GB gives an OOM or a run whose timings are meaningless. Falls back to
    the first ``n`` ids if nvidia-smi is unavailable.
    """
    if prefer_idle:
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            free = []
            for ln in out.strip().splitlines():
                idx, mem, util = (int(x) for x in ln.split(","))
                # BOTH tests are needed. Memory alone is not enough: a compute-bound job can hold a
                # device at 100% utilisation while allocating almost nothing, and picking it gives a
                # run that contends for SMs and reports meaningless timings. Utilisation alone is
                # not enough either, since an idle process can sit on 20 GB and OOM the newcomer.
                if mem < 512 and util < 10:
                    free.append(idx)
            if len(free) >= n:
                return ",".join(str(i) for i in free[:n])
            if free:
                import warnings
                warnings.warn(
                    f"only {len(free)} of {n} requested GPUs are idle ({free}); using those. "
                    f"Timings from a contended device are not comparable.", stacklevel=2)
                return ",".join(str(i) for i in free)
        except Exception:
            pass
    return ",".join(str(i) for i in range(n))
