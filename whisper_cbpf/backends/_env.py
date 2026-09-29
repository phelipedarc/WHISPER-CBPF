"""GPU / JAX environment resolution.

The shell bootstrap (``whisper_cbpf/backends/env.sh``) sets three things that must be in place
**before python starts**, because JAX reads them at import: which device is visible, how much of it
to preallocate, and where the CUDA shared libraries live. This module is the Python-side counterpart
-- it reports what actually happened and explains, in words a user can act on, what to fix when it
did not.

Nothing here imports jax at module scope. Importing :mod:`whisper_cbpf` must work on a machine with
no GPU (so ``wp.list_samplers()`` still shows what is available); the failure belongs at ``fit()``
time, with a message that says what to do.
"""
from __future__ import annotations

import os
import shutil
import subprocess

#: One wording for "this needs the optional GPU stack", so the six lazy layers that can raise it
#: cannot drift apart.
GPU_EXTRA_HINT = "{name} requires the [gpu] extra: pip install whisper-cbpf[gpu]"


def gpu_extra_error(name, exc=None):
    """An ``ImportError`` naming ``name`` and the extra that provides it."""
    return ImportError(GPU_EXTRA_HINT.format(name=name))


#: Environment variables that must be set before the first ``import jax``.
REQUIRED_ENV = ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Checkout root, from ``<root>/whisper_cbpf/backends/_env.py``. Derived rather than hardcoded so
#: the path follows the checkout wherever it is cloned.
_ROOT = os.path.dirname(os.path.dirname(_HERE))


def env_script():
    """Absolute path to the shell bootstrap, ``whisper_cbpf/backends/env.sh``.

    It lives inside the package and is listed in ``package-data``, so **it exists in a wheel install
    as well as a checkout**. Source it before starting Python (``source "$(whisper-cbpf-env)"``):
    without it JAX can fall back to the CPU, about 50x slower, with no error at all.

    Returns
    -------
    str

    Examples
    --------
    >>> import os
    >>> import whisper_cbpf as wp
    >>> os.path.basename(wp.env_script()), os.path.exists(wp.env_script())
    ('env.sh', True)
    """
    return os.path.join(_HERE, "env.sh")


def _print_env_script():
    """Console entry point ``whisper-cbpf-env``: print the bootstrap's path so it can be sourced.

    The one command that works from a wheel and from a checkout alike::

        source "$(whisper-cbpf-env)"

    A path outside the package could not work, because only ``whisper_cbpf*`` is installed.
    """
    print(env_script())
    return 0


def env_script_hint():
    """The ``source ...`` line for an error message, or an honest pointer when it is not shipped."""
    path = env_script()
    if os.path.exists(path):
        return f"source {path}"
    return ("source the GPU environment bootstrap -- run `whisper-cbpf-env` to print its path, "
            "then `source \"$(whisper-cbpf-env)\"`. See docs/GPU_SETUP.md.")


def env_report():
    """A dict describing the current GPU environment. Never raises, never imports jax.

    Returns
    -------
    dict
        The environment variables the bootstrap sets, ``LD_LIBRARY_PATH``, the bootstrap's path
        and whether it exists, and ``nvidia-smi``'s device table (``None`` without it).

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> rep = wp.env_report()
    >>> rep["env_script_exists"], "nvidia_smi" in rep
    (True, True)
    """
    rep = {k: os.environ.get(k) for k in REQUIRED_ENV}
    rep["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")
    rep["env_script"] = env_script()
    rep["env_script_exists"] = os.path.exists(rep["env_script"])
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

    Parameters
    ----------
    feature : str
        What needs JAX, named in the error.

    Returns
    -------
    tuple
        ``(jax, jax.numpy)``.

    Raises
    ------
    ImportError
        JAX is not installed; the message names the ``[gpu]`` extra and the bootstrap.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> jax, jnp = wp.require_jax("my analysis")        # needs the [gpu] extra
    >>> float(jnp.sum(jnp.ones(3)))
    3.0
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires JAX, which is not installed.\n"
            f"  pip install 'whisper-cbpf[gpu]'\n"
            f"then source the environment bootstrap so JAX can find the GPU:\n"
            f"  {env_script_hint()}"
        ) from exc
    return jax, jnp


def check_gpu(strict=False):
    """Return ``(ok, message)`` describing whether JAX can actually see a GPU.

    ``strict=True`` raises instead of returning ``ok=False``. The common failure here is not a
    missing GPU but a missing ``LD_LIBRARY_PATH``: JAX's pip-installed CUDA libraries and torch's
    conda-installed ones live in different trees, and if neither is on the path JAX silently falls
    back to CPU rather than erroring -- so a run that looks fine is 50x slower with no warning.

    Parameters
    ----------
    strict : bool, default False
        Raise ``RuntimeError`` instead of returning ``ok=False``.

    Returns
    -------
    tuple
        ``(ok, message)``; the message names what to fix when ``ok`` is False.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> ok, message = wp.check_gpu()                  # needs the [gpu] extra
    >>> isinstance(ok, bool) and isinstance(message, str)
    True
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
    msg += f"  fix: {env_script_hint()}  (before starting python)"
    if strict:
        raise RuntimeError(msg)
    return False, msg


def x64_enabled():
    """Whether JAX is in float64 mode. Must be decided BEFORE any jax array is created.

    Production NUTS runs in float64: float32 is fine at survey SNR but fails to mix on a tight
    high-SNR posterior (r-hat up to 19.6 measured on a 5-parameter kilonova fit), while costing
    only ~1.7x at batch 1. Batched simulation is the opposite case -- float64 is 51x slower at
    batch 1e4 on an A6000, whose FP64 rate is 1/32 of FP32.

    Returns
    -------
    bool

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> import jax                                    # needs the [gpu] extra
    >>> was = jax.config.jax_enable_x64
    >>> jax.config.update("jax_enable_x64", True)
    >>> wp.x64_enabled()
    True
    >>> jax.config.update("jax_enable_x64", was)
    """
    jax, _ = require_jax("x64_enabled")
    return bool(jax.config.jax_enable_x64)


#
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
    """Worker count at the production CPU budget (60% of cores by default).

    Parameters
    ----------
    fraction : float, default 0.6
        Share of ``os.cpu_count()``.

    Returns
    -------
    int
        At least 1.

    Examples
    --------
    >>> import os
    >>> import whisper_cbpf as wp
    >>> 1 <= wp.n_jobs() <= (os.cpu_count() or 1)
    True
    """
    return max(1, int(round(fraction * (os.cpu_count() or 1))))


def gpu_list(n=MAX_GPUS, prefer_idle=True):
    """The device ids to expose, as a CUDA_VISIBLE_DEVICES string.

    Prefers GPUs that are actually free: on a shared machine, grabbing a device that already holds
    someone else's 20 GB gives an OOM or a run whose timings are meaningless. Falls back to the
    first ``n`` ids if nvidia-smi is unavailable.

    Parameters
    ----------
    n : int, default 2
        Devices wanted.
    prefer_idle : bool, default True
        Pick devices with under 512 MB in use and under 10 % utilisation.

    Returns
    -------
    str
        Comma-separated device ids, e.g. ``"0,1"``.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.gpu_list(2, prefer_idle=False)
    '0,1'
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
