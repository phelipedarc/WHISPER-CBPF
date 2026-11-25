"""Science-validation suite: shared setup.

Every test here carries the ``science`` marker. The fast ones run with the normal suite
(``pytest tests -m "not slow"``); the studies that need a GPU and tens of minutes are also
``slow``. See ``docs/VALIDATION.md`` for what each one measures and the numbers it gave.

The supernova, TDE and kilonova models are float64 engines, so every test here runs with float64
switched on, and the session's setting is restored afterwards.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))       # the helpers: _sim, _studies

try:
    import jax
except ImportError:                     # pragma: no cover - the JAX tests skip themselves
    jax = None


@pytest.fixture(autouse=True)
def _float64():
    if jax is None:
        yield
        return
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if HERE in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.science)


@pytest.fixture
def needs_gpu():
    """Skip unless JAX in this process runs on a GPU."""
    import _studies

    if not _studies.gpu_visible():
        pytest.skip("needs JAX on a GPU (run with CUDA_VISIBLE_DEVICES set and without "
                    "JAX_PLATFORMS=cpu; see docs/VALIDATION.md)")
