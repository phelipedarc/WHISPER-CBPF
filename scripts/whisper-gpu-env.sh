#!/usr/bin/env bash
# REDIRECT, not a copy.
#
# The bootstrap now lives at whisper_cbpf/backends/env.sh, inside the package, so that
# `pip install whisper-cbpf[gpu]` actually ships it: pyproject's packages.find takes only
# `whisper_cbpf*`, so a wheel user previously got no scripts/ directory at all -- and this script is
# the one thing standing between them and JAX silently falling back to the CPU at ~50x with no
# error. See docs/GPU_SETUP.md and OPEN_ITEMS.md.
#
# This file stays because `source scripts/whisper-gpu-env.sh` is in the pushed README, in
# docs/GPU_SETUP.md and in every run recipe written before 2026-08-14. It sources the real thing
# rather than duplicating it -- DUPLICATES.md is explicit that two copies is the worse outcome, and
# a redirect is not a copy: there is still exactly one set of exports to keep correct.
#
# Prefer, from anywhere:
#
#     source "$(whisper-cbpf-env)"
#
_WG_SHIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_WG_TARGET="${_WG_SHIM_DIR}/../whisper_cbpf/backends/env.sh"

if [ ! -f "${_WG_TARGET}" ]; then
  echo "whisper_cbpf: cannot find ${_WG_TARGET}." >&2
  echo "  This shim expects to sit in scripts/ next to the package. From an installed wheel use:" >&2
  echo "      source \"\$(whisper-cbpf-env)\"" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=../whisper_cbpf/backends/env.sh
. "${_WG_TARGET}"
