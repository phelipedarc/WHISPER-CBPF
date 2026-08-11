#!/usr/bin/env bash
# whisper-GPU environment bootstrap. Source it BEFORE starting python:
#
#     source whisper_gpu/env.sh
#     python3 -c "import whisper_gpu"
#
# JAX reads device visibility and memory settings at import, so they cannot be set from inside
# Python once jax is loaded.
#
# Every path below is DERIVED, not hardcoded. The original version pinned the repo root to one
# machine's absolute path inside an executable `export PYTHONPATH`, which silently made the tree
# unimportable after a clone -- and the failure looked like a missing package, not a bad path.
# Everything is overridable from the caller's environment.

# --- repo root: the directory CONTAINING this package, so `import whisper_gpu` resolves ---------
_WG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_WG_DIR}/.." && pwd)"
# PYTHONPATH is searched before site-packages, so the live whisper checkout wins over any stale
# `pip install whisper-labia` that may be shadowing it.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# --- python environment --------------------------------------------------------------------------
# Override with WHISPER_GPU_VENV=/path/to/venv before sourcing.
WHISPER_GPU_VENV="${WHISPER_GPU_VENV:-/opt/jax_gpu_venv}"
if [ -x "${WHISPER_GPU_VENV}/bin/python3" ]; then
  export PATH="${WHISPER_GPU_VENV}/bin:${PATH}"
fi

# --- CUDA libraries --------------------------------------------------------------------------------
# JAX ships its CUDA libraries as pip packages under <site-packages>/nvidia/*/lib; torch ships its
# own set, often under a DIFFERENT interpreter. If neither is on LD_LIBRARY_PATH, JAX does not
# error -- it falls back to CPU SILENTLY, so the run merely looks slow. Collect both, venv first
# (jaxlib generally needs a newer cuDNN than torch pins; cuDNN 9.x is minor-version compatible).
_wg_nvidia_libs() {
  [ -d "$1" ] || return 0
  find "$1" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -
}
_WG_VENV_LIBS="$(_wg_nvidia_libs "${WHISPER_GPU_VENV}/lib/python3.11/site-packages/nvidia")"
_WG_SYS_LIBS="$(_wg_nvidia_libs "${WHISPER_GPU_CONDA:-/opt/conda}/lib/python3.11/site-packages/nvidia")"
export LD_LIBRARY_PATH="${_WG_VENV_LIBS}:${_WG_SYS_LIBS}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

# --- device selection ------------------------------------------------------------------------------
# PCI_BUS_ID makes CUDA_VISIBLE_DEVICES indices match nvidia-smi's; without it the enumeration order
# is driver-defined and "device 0" may not be the card you looked at.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Fraction of the visible GPU XLA preallocates. Keep well under 1.0 on a shared machine: XLA grabs
# its pool at first use and never returns it, so two jobs at 0.5 will not co-exist.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"

echo "whisper-GPU env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
     "mem_fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION} python=$(command -v python3)"
