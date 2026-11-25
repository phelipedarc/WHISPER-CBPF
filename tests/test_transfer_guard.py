"""check_device_transfers: a card-to-card copy that does not arrive intact is caught, not used.

Five claims, each a test:

1. **Intact copies pass.** On four host (CPU) devices every one of the 12 ordered pairs is copied and
   compared bit for bit, none is bad, and the array is float64 in 64-bit mode and 4 MiB.
2. **A bad copy raises, naming every bad pair.** With the copy replaced by one that returns zeros on
   one pair and stale data on another, ``RuntimeError`` names exactly those two pairs (and no good
   one), how many values differed, and points to one process per card (``run_jobs``).
3. **The result is kept per process and per device set.** A second call copies nothing and says
   ``cached``; a failed set raises again from the cache; ``refresh=True`` copies again; another
   device set is checked on its own.
4. **No GPU, nothing to check.** The default device list on a CPU-only JAX is empty and the report
   has no pairs.
5. **On a real multi-GPU host the known-bad pairs are reported, and only those.** Skipped unless
   ``WHISPER_TEST_BAD_PAIRS`` lists the pairs, as ``CUDA_VISIBLE_DEVICES`` entries, e.g.
   ``CUDA_VISIBLE_DEVICES=0,1,2 WHISPER_TEST_BAD_PAIRS=0:1,2:3 python -m pytest
   tests/test_transfer_guard.py``, and at least two GPUs are visible. Pairs not listed must pass.

Claims 1-4 run in a fresh interpreter with ``--xla_force_host_platform_device_count=4``, since the
device count is fixed when JAX starts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

import whisper_cbpf as wp

pytest.importorskip("jax")

_SCRIPT = textwrap.dedent("""
    import json, os
    import jax
    import jax.numpy as jnp
    import numpy as np
    from whisper_cbpf import parallel
    import whisper_cbpf as wp

    out = {"default": wp.check_device_transfers()}
    cpu = jax.devices("cpu")
    calls = []
    real_copy = parallel._copy

    def counting(x, device):
        calls.append(1)
        return real_copy(x, device)

    parallel._copy = counting
    first = wp.check_device_transfers(cpu)
    n_first = len(calls)
    second = wp.check_device_transfers(cpu)
    n_second = len(calls) - n_first
    third = wp.check_device_transfers(cpu, refresh=True)
    n_third = len(calls) - n_first - n_second
    sub = wp.check_device_transfers(cpu[:2])
    out.update(first=first, second=second, third=third, sub=sub,
               copies=[n_first, n_second, n_third])

    # a faulty copy: zeros on cpu:0 -> cpu:1, the previous array (stale) on cpu:2 -> cpu:3
    last = {}

    def faulty(x, device):
        calls.append(1)
        src = next(iter(x.devices())).id if isinstance(x, jax.Array) else None
        if src == 0 and device.id == 1:
            return jax.device_put(jnp.zeros(x.shape, x.dtype), device)
        if src == 2 and device.id == 3 and "prev" in last:
            return jax.device_put(last["prev"], device)
        if isinstance(x, np.ndarray):
            last["prev"] = last.get("cur", x)
            last["cur"] = x
        return real_copy(x, device)

    parallel._copy = faulty
    errors = []
    before = len(calls)
    for refresh in (True, False):
        try:
            wp.check_device_transfers(cpu, refresh=refresh)
            errors.append(None)
        except RuntimeError as exc:
            errors.append(str(exc))
    out.update(errors=errors, faulty_copies=len(calls) - before)
    print("RESULT" + json.dumps(out))
""")


@pytest.fixture(scope="module")
def cpu_run():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=4")
    run = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, env=env,
                         timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    line = [ln for ln in run.stdout.splitlines() if ln.startswith("RESULT")][-1]
    return json.loads(line[len("RESULT"):])


def test_intact_copies_pass(cpu_run):
    first = cpu_run["first"]
    assert first["devices"] == ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    assert len(first["pairs"]) == 12 and len({tuple(p) for p in first["pairs"]}) == 12
    assert all(a != b for a, b in first["pairs"])
    assert first["bad_pairs"] == [] and first["cached"] is False
    assert first["dtype"] == "float64" and first["bytes"] == 4 * 1024 * 1024
    assert "message" not in first
    # two copies per pair: host -> source, source -> destination
    assert cpu_run["copies"][0] == 24


def test_a_bad_copy_raises_naming_every_bad_pair(cpu_run):
    msg = cpu_run["errors"][0]
    assert msg is not None
    assert "2 of 12 ordered pairs" in msg
    assert "cpu:0 -> cpu:1 (source->destination)" in msg
    assert "cpu:2 -> cpu:3 (source->destination)" in msg
    bad_lines = [ln for ln in msg.splitlines() if " -> " in ln]
    assert len(bad_lines) == 2, msg
    assert msg.count("values differ") == 2
    assert "whisper_cbpf.run_jobs" in msg and "own process" in msg


def test_results_are_cached_per_process_and_device_set(cpu_run):
    n_first, n_second, n_third = cpu_run["copies"]
    assert cpu_run["second"]["cached"] is True and n_second == 0
    assert {k: v for k, v in cpu_run["second"].items() if k != "cached"} == \
        {k: v for k, v in cpu_run["first"].items() if k != "cached"}
    assert cpu_run["third"]["cached"] is False and n_third == 24
    assert cpu_run["sub"]["cached"] is False and len(cpu_run["sub"]["pairs"]) == 2
    # the failed set raises again from the cache, copying nothing more than the refreshed run did
    assert cpu_run["errors"][1] == cpu_run["errors"][0]
    assert cpu_run["faulty_copies"] == 24


def test_no_gpu_checks_nothing(cpu_run):
    default = cpu_run["default"]
    assert default["devices"] == [] and default["pairs"] == [] and default["bad_pairs"] == []


def _gpus():
    try:
        import jax

        return jax.devices("gpu")
    except Exception:
        return []


@pytest.mark.skipif(not os.environ.get("WHISPER_TEST_BAD_PAIRS"),
                    reason="set WHISPER_TEST_BAD_PAIRS (and CUDA_VISIBLE_DEVICES) to run on GPUs")
def test_known_bad_gpu_pairs_are_reported():
    gpus = _gpus()
    if len(gpus) < 2:
        pytest.skip("needs at least two visible GPUs")
    known = {frozenset(p.split(":")) for p in os.environ["WHISPER_TEST_BAD_PAIRS"].split(",")}
    entries = [e.strip() for e in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    assert len(entries) >= len(gpus), "set CUDA_VISIBLE_DEVICES so the pairs can be named"

    def entry(label):
        return re.findall(r"CUDA_VISIBLE_DEVICES entry ([^)]+)\)", label)

    try:
        report = wp.check_device_transfers(gpus, refresh=True)
        bad = set()
        labels = report["devices"]
    except RuntimeError as exc:
        bad = {frozenset(entry(ln)) for ln in str(exc).splitlines() if " -> " in ln}
        labels = [wp.parallel._device_label(d) for d in gpus]
    print("bad pairs (CUDA_VISIBLE_DEVICES entries):", sorted(map(sorted, bad)))
    visible = {entry(lab)[0] for lab in labels}
    expected = {p for p in known if p <= visible}
    assert bad == expected, (f"bad pairs {sorted(map(sorted, bad))}, "
                             f"expected {sorted(map(sorted, expected))}")


def test_docstring_examples_run():
    """Claims 1 and 3, as the docstring shows them on one CPU device (no pairs, then cached)."""
    import doctest

    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for test in doctest.DocTestFinder().find(wp.check_device_transfers, "check_device_transfers",
                                             globs={}):
        runner.run(test)
    assert runner.tries > 0 and runner.failures == 0
