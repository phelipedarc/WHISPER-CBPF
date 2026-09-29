"""Backend detection and the GPU environment policy.

Re-export layer over :mod:`._env`. Imports nothing heavier than the standard library, so
``import whisper_cbpf`` never touches JAX, CUDA or torch.

The environment bootstrap is a **shell script**, ``whisper_cbpf/backends/env.sh``, and has to be:
it sets the CUDA variables *before Python starts*, because JAX reads device visibility and memory
fraction at import time; if ``LD_LIBRARY_PATH`` is wrong JAX falls back to CPU silently, at roughly
50x. Source it with ``source "$(whisper-cbpf-env)"``. See :func:`env_script` for its path and
``docs/GPU_SETUP.md`` for the trap.
"""
from __future__ import annotations

from ._env import (  # noqa: F401
    CPU_FRACTION,
    MAX_GPUS,
    check_gpu,
    env_report,
    env_script,
    env_script_hint,
    gpu_extra_error,
    gpu_list,
    n_jobs,
    require_jax,
    x64_enabled,
)

__all__ = ["check_gpu", "require_jax", "x64_enabled", "gpu_list", "n_jobs",
           "env_script", "env_script_hint", "env_report", "MAX_GPUS", "CPU_FRACTION"]
