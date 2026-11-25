#!/usr/bin/env bash
# whisper_cbpf GPU environment bootstrap. Source it BEFORE starting python:
#
#     source "$(whisper-cbpf-env)"        # from a wheel OR a checkout
#     source whisper_cbpf/backends/env.sh  # from a checkout, directly
#     python3 -c "import whisper_cbpf"
#
# JAX reads device visibility and memory settings at import, so they cannot be set from inside
# Python once jax is loaded.
#
# Every path below is DERIVED, not hardcoded. The original version pinned the repo root to one
# machine's absolute path inside an executable `export PYTHONPATH`, which silently made the tree
# unimportable after a clone -- and the failure looked like a missing package, not a bad path.
# Everything is overridable from the caller's environment.

# --- repo root: only meaningful in a SOURCE CHECKOUT ---------------------------------------------
# This file lives at whisper_cbpf/backends/env.sh, so the checkout root is ../../ -- but in a wheel
# install that is site-packages, where prepending PYTHONPATH is pointless and would shadow nothing.
# A real checkout is the one with a pyproject.toml at that level; test for it rather than assume.
# (It used to sit in scripts/ and prepend unconditionally. It moved so that `pip install` ships it:
# packages.find takes only whisper_cbpf*, so a wheel user had no scripts/ directory at all, and this
# script is the ONE thing standing between them and a silent 50x CPU fallback.)
_WG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_WG_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/pyproject.toml" ]; then
  # PYTHONPATH is searched before site-packages, so the live checkout wins over any stale install.
  # No trailing ':' when it was empty: python reads an empty entry as the current directory.
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# --- python environment --------------------------------------------------------------------------
# To use a virtualenv that is not already first on PATH, name it before sourcing:
#     export WHISPER_CBPF_VENV=/path/to/venv
# and its interpreter goes first on PATH. Unset, PATH is left alone and the python3 already on it
# is the one used. The discontinued name WHISPER_GPU_VENV is still honoured.
WHISPER_CBPF_VENV="${WHISPER_CBPF_VENV:-${WHISPER_GPU_VENV:-}}"
if [ -n "${WHISPER_CBPF_VENV}" ] && [ -x "${WHISPER_CBPF_VENV}/bin/python3" ]; then
  export PATH="${WHISPER_CBPF_VENV}/bin:${PATH}"
fi

# --- CUDA libraries --------------------------------------------------------------------------------
# JAX ships its CUDA libraries as pip packages under <site-packages>/nvidia/*/lib; torch ships its
# own set, often under a DIFFERENT interpreter. If neither is on LD_LIBRARY_PATH, JAX does not
# error -- it falls back to CPU SILENTLY, so the run merely looks slow. Collect both, venv first
# (jaxlib generally needs a newer cuDNN than torch pins; cuDNN 9.x is minor-version compatible):
# the site-packages of the python3 now on PATH, then of its base interpreter when that python3 is
# a venv's, asked of python itself rather than guessed from a version number. Only non-empty
# parts are joined, because the dynamic loader reads an empty entry as the current directory.
_wg_nvidia_libs() {
  [ -d "$1" ] || return 0
  find "$1" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -
}
_WG_LD=""
_wg_append() {
  if [ -n "$1" ]; then _WG_LD="${_WG_LD:+${_WG_LD}:}$1"; fi
}
while IFS= read -r _wg_site; do
  _wg_append "$(_wg_nvidia_libs "${_wg_site}/nvidia")"
done < <(python3 -c '
import sys, sysconfig
paths = [sysconfig.get_path(kind, vars={"base": b, "platbase": b})
         for b in (sys.prefix, sys.base_prefix) for kind in ("purelib", "platlib")]
print("\n".join(dict.fromkeys(paths)))' 2>/dev/null)
_wg_append /usr/local/nvidia/lib
_wg_append /usr/local/nvidia/lib64
_wg_append "${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${_WG_LD}"

# --- device selection ------------------------------------------------------------------------------
# PCI_BUS_ID makes CUDA_VISIBLE_DEVICES indices match nvidia-smi's; without it the enumeration order
# is driver-defined and "device 0" may not be the card you looked at.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Fraction of the visible GPU XLA preallocates. Keep well under 1.0 on a shared machine: XLA grabs
# its pool at first use and never returns it, so two jobs at 0.5 will not co-exist.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"

echo "whisper_cbpf GPU env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
     "mem_fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION} python=$(command -v python3)"
