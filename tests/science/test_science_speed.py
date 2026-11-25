"""(f) Speed budgets: seconds per alert, recorded, and a test fails on a 2x regression.

Each budget is a wall time measured on the validation machine (one NVIDIA RTX A6000; a 96-core
host shared with other work) with the default settings; ``docs/VALIDATION.md`` lists the
measurements. A test fails when a new measurement exceeds ``REGRESSION`` times its budget, so a
change that doubles the cost of an alert is caught. On other hardware, set
``WHISPER_SPEED_SCALE`` (e.g. 2 for a machine half as fast).

- **compare, one GPU**: ``wp.compare`` of the five supernova and TDE families on the bundled LSST
  alert (74 rows, explosion time unknown, known redshift), ``sampler="auto"`` (``emcee_jax``),
  ``evidence_check=False``. The first call pays the compiles; the second (same process) is what
  each further alert costs.
- **compare, CPU**: the same comparison with ``JAX_PLATFORMS=cpu`` (``sampler="auto"`` picks CPU
  ``mcmc``).
- **fit_batch, one GPU**: 64 simulated supernova alerts in one call, default walkers and steps.

All ``slow``; the GPU ones need a GPU, the CPU one a CPU-only session.
"""
from __future__ import annotations

import os

import pytest

import _studies  # noqa: E402

REGRESSION = 2.0
#: Measured wall seconds, 2026-09-28 (see docs/VALIDATION.md, "Speed").
BUDGETS = {
    "compare5_gpu_first_s": 400.0,
    "compare5_gpu_next_s": 392.0,
    "compare5_cpu_s": 1940.0,
    "fit_batch_gpu_per_alert_s": 3.0,
}


def _limit(name):
    budget = BUDGETS[name]
    if budget is None:
        pytest.skip(f"no budget recorded for {name} yet")
    return REGRESSION * budget * float(os.environ.get("WHISPER_SPEED_SCALE") or 1.0)


@pytest.mark.slow
def test_five_family_compare_per_alert_on_one_gpu(needs_gpu):
    rec = _studies.speed_compare("compare5_gpu", repeats=2)
    first, nxt = (r["wall_s"] for r in rec["runs"])
    assert first <= _limit("compare5_gpu_first_s"), rec
    assert nxt <= _limit("compare5_gpu_next_s"), rec


@pytest.mark.slow
def test_five_family_compare_per_alert_on_the_cpu():
    pytest.importorskip("jax")                  # the supernova and TDE families are JAX models
    if _studies.gpu_visible():
        pytest.skip("the CPU budget is measured in a CPU-only session (JAX_PLATFORMS=cpu)")
    rec = _studies.speed_compare("compare5_cpu", repeats=1)
    assert rec["runs"][0]["wall_s"] <= _limit("compare5_cpu_s"), rec


@pytest.mark.slow
def test_fit_batch_throughput_on_one_gpu(needs_gpu):
    rec = _studies.speed_fit_batch("fit_batch_gpu", k=64)
    assert rec["per_alert_s"] <= _limit("fit_batch_gpu_per_alert_s"), rec
