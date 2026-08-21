"""Root pytest configuration.

``tests/t3_inference/`` imports ``_t3_common``, which calls ``import jax`` at module scope: it has
to set ``jax_enable_x64`` before the first JAX array exists, so ``importorskip`` would defeat the
point. Without JAX installed that raises at *collection* and aborts the whole run, so the tier is
excluded there. With JAX installed nothing here applies.
"""
from __future__ import annotations

from importlib.util import find_spec

#: Test paths whose *import* requires JAX, rather than skipping on it at runtime.
_JAX_ONLY = ("tests/t3_inference",)

collect_ignore_glob: list[str] = []

if find_spec("jax") is None:
    collect_ignore_glob += [f"{p}/*" for p in _JAX_ONLY]
