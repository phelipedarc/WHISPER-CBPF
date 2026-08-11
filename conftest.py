"""Root pytest configuration for the merged suite.

Created during the merge; not copied from either source repo.

**Why this file exists.** Before the merge the two suites never ran together: the CPU tests ran in
``phe_sbi`` and the GPU tests in ``phe_sbi_2``, so nothing ever collected a JAX test on a machine
without JAX. Most GPU tests guard themselves with ``pytest.importorskip("jax")`` and skip cleanly.
Three cannot: ``tests/t3_inference/`` imports ``_t3_common``, which does a bare ``import jax``
because it must set ``jax_enable_x64`` **before the first JAX array exists** -- an importorskip
there would defeat the point of the module. On a CPU-only box those three raise at *collection*,
which aborts the whole run.

Rather than edit the tests (the migration does not restructure them), collection of the JAX-only
tier is skipped when JAX is absent. With JAX installed nothing here applies and the full suite
collects exactly as it does in ``phe_sbi_2``.
"""
from __future__ import annotations

from importlib.util import find_spec

#: Test paths whose *import* requires JAX, rather than skipping on it at runtime.
_JAX_ONLY = ("tests/t3_inference",)

#: Not collected: it loads ``sanity_check/benchmark_kilonova_modes.py`` by path, and the
#: sanity_check/ tree is excluded from this package by the functionality-only scope rule
#: (the package ships code, not the analyses we ran). The file is kept, not deleted, so the
#: dependency is visible and the decision is reversible. 3 tests.
_NEEDS_EXCLUDED_TREES = ("tests/test_benchmark_kilonova.py",)

collect_ignore_glob: list[str] = []

collect_ignore_glob += list(_NEEDS_EXCLUDED_TREES)

if find_spec("jax") is None:
    collect_ignore_glob += [f"{p}/*" for p in _JAX_ONLY]
