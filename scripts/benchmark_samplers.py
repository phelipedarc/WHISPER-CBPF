#!/usr/bin/env python3
"""Honest end-to-end sampler benchmark for ``whisper_cbpf``, on AT2017GFO.

Runs one sampler configuration per **subprocess**, measures it with instruments that can be named,
and writes a JSON sidecar next to a markdown report so every number in the report can be re-checked
against the record that produced it.

Why one subprocess per cell
---------------------------
* ``CUDA_VISIBLE_DEVICES`` / ``JAX_PLATFORMS`` / ``XLA_PYTHON_CLIENT_MEM_FRACTION`` are read by JAX
  **at import**, so they can only be set before python starts.
* The XLA executable cache is per process; a fresh process makes ``t_cold - t_warm`` a real
  per-cell compile measurement instead of a leftover from the previous cell.
* ``jax.Device.memory_stats()['peak_bytes_in_use']`` is a since-process-start high-water mark, so
  it attributes to exactly one cell.
* A JAX-warmed process contaminates the numpy arms — the hazard
  ``whisper_cbpf/samplers/jax/emcee_jax.py`` documents for ``fit_emcee_numpy``.

Cells run **strictly sequentially**. There is no ``--jobs`` for cells, by design: two concurrent
cells make every timing in the run meaningless. The only concurrency is the resource-monitor thread.

The measurement axis, and what a row licenses
---------------------------------------------
Primary: **warm wall-clock seconds**, alongside ``ESS_worst`` and ``ESS_worst/s`` computed from the
warm time only. ESS/s is never a ranking key on its own -- ABC/ABC-SMC/SNPE target approximations
of the posterior and produce near-independent draws, so their ESS/s is enormous for draws from the
wrong distribution. The full statement of what a row does and does not license is in
:data:`AXIS_MD` and :data:`NOT_LICENSED_MD`, which are the *single source* for both this contract
and the generated report's Method section -- edit them, not a second copy here.

Instruments (all sampled/derived quantities are labelled as such in the sidecar)
-------------------------------------------------------------------------------
=========================  =====================================================================
wall time                  ``time.perf_counter()`` around ``wp.fit`` only
cold start                 ``t_cold - t_warm`` across repeats in one process. On a JAX cell this
                           is dominated by XLA compile, but it is an **upper bound on compile and
                           not a compile measurement**: it also absorbs allocator first-touch,
                           cuDNN autotune and any first-call library import inside the metrics
                           (measured: 0.83 s on a pure-numpy `bazin`/`abc` cell, where there is no
                           compiler at all). Cross-checked against ``info['compile_time_s']``
                           where the sampler reports its own; a disagreement over 25% is flagged.
model warm-up              one blocked ``model.predict`` at the prior median, before any timer
                           (the JAX models' ``predict`` is itself ``jax.jit``ed, so even a "CPU"
                           sampler pays an XLA compile on its first call)
peak GPU memory            ``jax.devices()[0].memory_stats()['peak_bytes_in_use']`` (exact) and
                           ``torch.cuda.max_memory_allocated()`` for SNPE. **Never nvidia-smi**:
                           under this repo's own env script XLA preallocates
                           ``XLA_PYTHON_CLIENT_MEM_FRACTION`` of the card, so nvidia-smi reports
                           the pool, not the run.
nvidia-smi                 SAMPLED at ``--monitor-interval`` (default 1.0 s) for **contention**
                           only: peak memory on the card, mean utilisation, throttle reasons,
                           foreign compute PIDs. A sampled estimate, not an integral; a run
                           shorter than a few intervals gets a handful of points and says so
                           (``smi_n_samples``).
CPU core-seconds           ``resource.getrusage(RUSAGE_SELF) + getrusage(RUSAGE_CHILDREN)`` delta
                           across the fit. Exact, not sampled: emcee's and ABC's ``fork``ed pools
                           are joined inside ``fit``, so RUSAGE_CHILDREN captures them in full,
                           where a 0.5 s sampler would miss short-lived children entirely.
                           Caveat: RUSAGE_CHILDREN counts *any* child this process reaped.
ESS                        ``whisper_cbpf.metrics.ess_summary`` (arviz bulk ESS, worst parameter)
accuracy                   ``sbi.utils.metrics.c2st`` + normalised 1-Wasserstein against a cached
                           reference posterior, when one exists (``--build-references``)
=========================  =====================================================================

Usage
-----
    # CPU box (phe_sbi, no JAX): the minimal install the package advertises
    /opt/conda/bin/python scripts/benchmark_samplers.py --device cpu --quick

    # GPU box (phe_sbi_2), CPU-JAX arm: same interpreter, same code, JAX pinned to CPU
    source scripts/whisper-gpu-env.sh
    /opt/jax_gpu_venv/bin/python3 scripts/benchmark_samplers.py --device cpu --quick

    # GPU box, GPU arm. Aborts rather than downgrading if no card is idle in BOTH mem and util.
    /opt/jax_gpu_venv/bin/python3 scripts/benchmark_samplers.py --device gpu --gpu-ids 5 --quick

    # re-render a finished run's markdown without re-running anything
    python scripts/benchmark_samplers.py --report-only _scratch/bench/<run_id>

    # the driver's own consistency checks (no fits, no GPU, safe in CI)
    python scripts/benchmark_samplers.py --self-test

Output lands in gitignored ``_scratch/bench/<run_id>/``. ``--publish`` (default off) copies the
report to ``docs/BENCHMARKS.md``; the human commits.

Scope note
----------
The design this implements also specified ``scripts/bench_crossover.py``,
``scripts/bench_report.py`` and ``tests/test_benchmark_driver.py``. This session owns only
``scripts/benchmark_samplers.py`` and ``docs/BENCHMARKS.md``, so the report renderer
(``--report-only``), the crossover sweep (``--crossover``) and the driver checks (``--self-test``)
are folded into this file instead of living in three. They are separable entry points, not a merge
of concerns.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported, not duplicated: the impossibility table, the data loader and the model registrations
# stay single-sourced in the acceptance script. See scripts/compare_at2017gfo.py.
from compare_at2017gfo import (  # noqa: E402
    BAND_ALIASES,
    CSV,
    DL_CM,
    FILTER_SET,
    FIT_BANDS,
    REDSHIFT,
    T_EXPLOSION_MJD,
    _budget,
    _load_lc,
    _register_models,
    impossible_reason,
)

SCHEMA_VERSION = 1
OUT_ROOT = REPO / "_scratch" / "bench"

#: worst-parameter bulk-ESS the two-stage scaling aims for (~5% MC error on a 90% CI bound).
QUALITY_TARGET_ESS = 400
#: c2st at or below this counts as "indistinguishable from the reference".
C2ST_PASS = 0.60
#: clip on the stage-2 budget multiplier, so one noisy pilot cannot ask for a 400x run.
BUDGET_SCALE_CLIP = (0.25, 16.0)
MAX_CELL_SECONDS = 1800
MONITOR_INTERVAL_S = 1.0
#: idle thresholds; identical to whisper_cbpf.backends._env.gpu_list, deliberately.
GPU_IDLE_MEM_MIB = 512
GPU_IDLE_UTIL_PCT = 10
#: How much device memory beyond our own preallocated XLA pool counts as somebody else's.
#: MEASURED, not guessed: on this box our worker's own non-pool footprint (CUDA context plus
#: cuBLAS/cuDNN workspaces, which live outside the pool) ranged over 279-866 MiB across four cells
#: on the same card and model, while a genuine neighbour showed up as +19,991 MiB (a second
#: process taking its own 0.4-fraction pool). 512 MiB therefore misfires on our own context and
#: 2048 MiB separates the two cases cleanly. The cost is a blind spot: this test cannot see a
#: neighbour smaller than ~2 GiB, and the report says so.
GPU_FOREIGN_SLACK_MIB = 2048

#: The measurement axis, in the report's own words. Single-sourced: the module docstring points
#: here rather than keeping a second copy that can drift.
AXIS_MD = """\
The primary axis is **warm wall-clock seconds**, reported alongside `ESS_worst` (worst-parameter
bulk ESS, `whisper_cbpf.metrics.ess_summary`) and `ESS_worst/s` computed from the **warm** time
only. Compile is never amortised into a per-sample figure, and no per-sample figure is computed
from a cold time. With `--two-stage` each cell's budget knob is rescaled from a pilot run so the
cells are compared at a matched posterior-quality target (`ESS_worst >= {target}`).

**ESS/s is not a ranking key on its own.** ABC, ABC-SMC and SNPE target *approximations* of the
posterior; their draws are near-independent, so `ESS ~ n_draws` and their ESS/s is enormous — for
draws from the wrong distribution. A row whose accuracy verdict is not `PASS`, or whose ESS is
`iid_assumed` (no chain axis), has its ESS/s struck through and is excluded from any speed ranking.
This is the single most important honesty rule in the design.

**Axes deliberately rejected.** *Fixed forward-model evaluations* cannot be controlled: NUTS's
evaluation count is adaptive, and in this repository it is not even knowable after the fact
(`nuts_gpu.py` never collects `num_steps`). A NUTS evaluation is also a *gradient* evaluation
(~2-4x a forward one) while `abc_gpu`'s are batched 16 wide — so equal counts would be
conspicuously unequal work, which is the effect under study. Evaluations are reported as a
covariate instead. *Fixed wall time, compare quality* needs a pilot run anyway, and then compares
at a budget no sampler was tuned for; the two-stage design spends that same pilot on
quality-matching, which answers the more useful question.

**What a passing row licenses:** "On AT2017GFO as selected here, for *this* model, on *this* box
and *this* card, at *this* budget, sampler X produced N effective samples of its worst-constrained
parameter in T warm seconds."
""".format(target=QUALITY_TARGET_ESS)

NOT_LICENSED_MD = """\
1. **Cross-sampler speed comparison when the accuracy gate failed or was not run.** A struck-through
   ESS/s is not a slower number, it is *not a number*.
2. **Extrapolation to other datasets or models.** ESS/s depends on `n_data`, the parameter count,
   the forward-model cost and the posterior geometry. The `bazin` control cell exists precisely to
   expose how much of each sampler's cost is fixed overhead rather than model cost.
3. **"The GPU is N x faster" as a hardware claim.** The CPU and GPU arms run *different code paths
   for the same physics* — `predict` one curve at a time against `vmap(predict_jax)` at
   `chunk=16`. Every such ratio is whisper's CPU arm against whisper's GPU arm, never CPU silicon
   against GPU silicon.
4. **Anything about model quality.** AIC/BIC/WAIC compare **models**; they are carried so a reader
   can see whether a fast run was also a good one, never to rank samplers.
5. **Cross-space comparison.** A flux-space fit and a magnitude-space fit have different
   likelihoods over different data representations (`scripts/compare_at2017gfo.py:24-26`). The
   matrix fixes magnitude space and carries exactly one flux cell, and that cell is a control on
   *timing*, not an invitation to compare information criteria across spaces.
6. **Reproduction on a quiet machine.** Every number here was measured on a shared, loaded box.
   Rows that could not be trusted say so; rows that could still carry that box's noise floor.
"""
#: BLAS threading pinned in every worker so n_jobs is the only parallelism axis.
THREAD_PINS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
#: attach_predictive_metrics(..., n_draws=200) in whisper_cbpf/samplers/base.py
PREDICTIVE_DRAWS = 200
#: the deterministic max-log-likelihood scan cap in abc.py / abc_gpu.py / snpe.py
MAX_LOGL_SCAN = 2000

DEVICES = ("cpu-min", "cpu-jax", "gpu")

#: which sampler kwarg the two-stage rescaling turns, and what it must stay a multiple of.
BUDGET_KNOB = {
    "abc": "n_simulations",
    "abc_smc": "n_particles",
    "abc_gpu": "n_simulations",
    "abc_smc_gpu": "n_particles",
    "mcmc": "nsteps",
    "nested": "nlive",          # more live points -> more evidence resolution AND more draws
    "emcee_jax": "nsteps",
    "snpe": "num_simulations",
    "nuts_gpu": "num_samples",
    "pymc_jax_gpu_vectorized": "num_samples",
    "pymc_jax_gpu_parallelized": "num_samples",
}

#: samplers whose draws have no chain axis -> ESS is computed as if iid, which for ABC/SMC/SNPE is
#: an assumption, not a measurement. Every such row is flagged in the report.
IID_ASSUMED = ("abc", "abc_smc", "snpe", "npe", "abc_smc_gpu", "abc_gpu", "nested")

#: model_key -> (registration token for _register_models, registered name predicate, kind)
MODEL_KEYS = {
    "bazin": ("toys", "bazin", "toy"),
    "gaussian_rise": ("toys", "gaussian_rise", "toy"),
    "mck19": ("toys", "mck19", "toy"),
    "kn2_redback": ("kn2_redback", "two_component_kilonova", "redback"),
    "kn1": ("kn1", None, "kilonova"),
    "kn2": ("kn2", None, "kilonova"),
}

MODEL_LABEL = {
    "bazin": "bazin (analytic control)",
    "gaussian_rise": "gaussian_rise (analytic)",
    "mck19": "mck19 (analytic)",
    "kn2_redback": "kilonova-2c (redback, numpy)",
    "kn1": "kilonova-1c (JAX)",
    "kn2": "kilonova-2c (JAX)",
}

#: Cells deliberately left out of the matrix, with the reason printed in the report. Never a
#: silent skip -- the same discipline compare_at2017gfo.py applies to impossible cells.
STATIC_EXCLUSIONS = [
    ("tde, tde_ce, all twelve supernovae x any sampler",
     "float64 is mandatory for these families; impossible_reason() already refuses abc_gpu/"
     "abc_smc_gpu on them (~57x batched float64 cost, docs/GPU_SETUP.md), and a NUTS cell here "
     "would run for hours to restate what the impossibility table already says."),
    ("flare_jax x any JAX sampler",
     "structurally unreachable: models/jax/flare.py's predict_jax takes a parameter dict, not the "
     "flat theta Model.predict_jax specifies, so flare_jax registers without the slot "
     "(docs/CHOOSING.md, OPEN_ITEMS.md)."),
    ("snpe_gpu",
     "flare-only and deliberately unregistered (DEPLOYMENT_AUDIT.md section 5 item 10)."),
    ("mck19, gaussian_rise",
     "dominated by bazin as an analytic control; nothing new would be measured."),
    ("flux space, everywhere except one control cell",
     "AIC/BIC/WAIC are not comparable across spaces (scripts/compare_at2017gfo.py:24-26). One "
     "kn1/mcmc/cpu-jax/flux cell tests the assertion that TIMING is space-independent."),
    ("abc_smc_gpu",
     "runnable, but it does not set result.samples_by_chain (DEPLOYMENT_AUDIT.md section 5 item "
     "11), so ess_summary has nothing to consume and its ESS/s would be iid_assumed guesswork on "
     "a sampler that is not iid. Include it once that is fixed."),
]

#: Instrumentation gaps that are REPORTED, never patched around. A benchmark that edits the thing
#: it measures is not a benchmark.
KNOWN_GAPS = [
    ("NUTS forward-model / gradient evaluations",
     "whisper_cbpf/samplers/jax/nuts_gpu.py:170,179 pass extra_fields=('diverging',) only, so "
     "num_steps is never collected and leapfrog counts are unrecoverable from result.numpyro_mcmc. "
     "Reported as `unknown`. WHAT WOULD SETTLE IT: add 'num_steps' to both extra_fields tuples and "
     "report sum(num_steps) as GRADIENT evaluations (~2-4x the cost of a forward evaluation). "
     "Filed, not patched."),
    ("abc_smc_gpu ESS",
     "no result.samples_by_chain (DEPLOYMENT_AUDIT.md section 5 item 11); cell excluded rather "
     "than reported with a fabricated chain axis."),
    ("forward-model evaluation counts are exact only for `abc` and `abc_gpu`",
     "measured, not assumed: `--validate-evals` wraps a numpy model's `predict` in a counter and "
     "compares. `abc` matched exactly at every budget tried. `mcmc`/`emcee_jax` are an UPPER bound "
     "(a proposal outside the prior support returns -inf without calling `predict`, "
     "samplers/mcmc.py:46-48); `abc_smc` is an UPPER bound (`total_simulations` counts attempts, "
     "and post-round-0 attempts outside the support skip `predict`, samplers/abc_smc.py:77-78); "
     "`snpe` is a LOWER bound by a fixed, budget-independent +3. The shim cannot reach `n_jobs>1` "
     "(the counter lives in the forked child) or the JAX samplers (they call `predict_jax`)."),
    ("nvidia-smi as a memory instrument",
     "useless under scripts/whisper-gpu-env.sh: XLA preallocates "
     "XLA_PYTHON_CLIENT_MEM_FRACTION of the card, so memory.used reports the pool, not the run. "
     "Peak memory comes from jax memory_stats()/torch; nvidia-smi is a CONTENTION signal only. "
     "jaxlib 0.10.2 also returns peak_bytes_reserved = 0, so our own footprint on the card is "
     "taken from bytes_limit (the pool) instead."),
    ("the GPU contention detector cannot see a neighbour smaller than ~2 GiB",
     "measured: our own non-pool footprint (CUDA context plus cuBLAS/cuDNN workspaces) ranged "
     "over 279-866 MiB across four cells on one card and one model, so a 512 MiB slack flags our "
     "own context as a neighbour. A real neighbour on the same card appeared as +19,991 MiB. The "
     "slack is therefore 2048 MiB, and anything smaller than that is invisible to this test. "
     "WHAT WOULD SETTLE IT: per-process attribution via nvidia-smi --query-compute-apps, which "
     "needs a PID namespace the container does not have."),
    ("foreign-PID attribution inside a container",
     "nvidia-smi --query-compute-apps reports PIDs from a namespace the container cannot map, so "
     "'is this PID ours?' is unanswerable here. Each cell records "
     "contention.foreign_pid_detection to say whether the test was usable; the memory-excess and "
     "utilisation tests do not depend on PIDs."),
]


# =============================================================================================
# Cell
# =============================================================================================

@dataclass(frozen=True)
class Cell:
    cell_id: str
    group: str
    model_key: str
    sampler: str
    device: str
    space: str = "magnitude"
    x64: bool = False
    n_devices: int = 1
    n_jobs: int = 1
    repeats: int = 3
    seed: int = 0
    role: str = "cell"
    budget: dict = field(default_factory=dict)
    note: str = ""

    @property
    def model_kind(self):
        return MODEL_KEYS[self.model_key][2]

    @property
    def has_jax(self):
        return self.device in ("cpu-jax", "gpu")


def _cell_id(group, model_key, sampler, device, space):
    base = f"{group}/{model_key}/{sampler}/{device}"
    return base if space == "magnitude" else f"{base}/{space}"


def _mk(group, model_key, sampler, device, budgets, **kw):
    space = kw.pop("space", "magnitude")
    b = dict(budgets.get(sampler, {}))
    b.update(kw.pop("budget_override", {}))
    n_jobs = kw.pop("n_jobs", None)
    if n_jobs is None:
        # A fork()ed pool inside a JAX-warmed process risks deadlock (emcee_jax.py:235-248
        # documents exactly this). Serial by default on the JAX arms; --allow-fork-with-jax opts in.
        n_jobs = int(b.get("n_jobs", 1)) if device == "cpu-min" else 1
    b.pop("n_jobs", None)
    return Cell(cell_id=_cell_id(group, model_key, sampler, device, space), group=group,
                model_key=model_key, sampler=sampler, device=device, space=space,
                budget=b, n_jobs=int(n_jobs), **kw)


def refusal_reason(cell, allow_fork_with_jax=False):
    """Why this cell must not be run, or ``None``. The single gate ``build_matrix`` applies.

    Delegates the structural question to ``compare_at2017gfo.impossible_reason`` so the
    impossibility table stays in one place.
    """
    why = impossible_reason(cell.model_kind, cell.sampler, cell.space, cell.has_jax)
    if why:
        return f"impossible_reason(): {why}"
    if cell.device != "cpu-min" and cell.n_jobs > 1 and not allow_fork_with_jax:
        return ("n_jobs>1 on a JAX arm needs --allow-fork-with-jax (forking a process with JAX's "
                "threads live risks deadlock; samplers/jax/emcee_jax.py:235-248)")
    return None


def build_matrix(budgets, groups, device_set, repeats=3, allow_fork_with_jax=False):
    """The defensible matrix. Every cell is checked by ``refusal_reason`` before it exists.

    Returns ``(cells, refused)`` where ``refused`` records anything the gate rejected, so the
    report can print it rather than silently dropping it.
    """
    cells, refused = [], []

    def add(c):
        why = refusal_reason(c, allow_fork_with_jax)
        if why:
            refused.append({"cell": c.cell_id, "reason": why})
            return
        cells.append(replace(c, repeats=repeats))

    # ---- A: gradient-free numpy models on the minimal install we advertise -------------------
    if "A" in groups and "cpu-min" in device_set:
        for mk in ("kn2_redback", "bazin"):
            for s in ("abc", "abc_smc", "mcmc", "nested", "snpe"):
                add(_mk("A", mk, s, "cpu-min", budgets))

    # ---- B: what the GPU buys for the same JAX model -----------------------------------------
    if "B" in groups:
        for dev in ("cpu-jax", "gpu"):
            if dev not in device_set:
                continue
            for s in ("abc", "mcmc", "abc_gpu", "emcee_jax", "nuts_gpu"):
                add(_mk("B", "kn1", s, dev, budgets))
            for s in ("mcmc", "abc_gpu", "nuts_gpu"):
                add(_mk("B", "kn2", s, dev, budgets))
        # the single flux control: tests the assertion that timing is space-independent.
        if "cpu-jax" in device_set:
            add(_mk("B", "kn1", "mcmc", "cpu-jax", budgets, space="flux",
                    note="flux control for the space-independence-of-timing assertion"))

    # ---- C: three NUTS frontends, one kernel; plus the docs' 3-arm chain-method table ---------
    if "C" in groups and "gpu" in device_set:
        for s in ("nuts_gpu", "pymc_jax_gpu_vectorized", "pymc_jax_gpu_parallelized"):
            add(_mk("C", "kn2", s, "gpu", budgets))
        # Re-test of docs/GPU_SETUP.md:130. Arm A needs 4 idle cards, arm B needs 2; preflight
        # aborts them rather than letting NumPyro downgrade 'parallel' to 'sequential' silently.
        base = dict(budgets.get("nuts_gpu", {}))
        for tag, sampler, nch, ndev in (("armA-parallel-4", "pymc_jax_gpu_parallelized", 4, 4),
                                        ("armB-parallel-2", "pymc_jax_gpu_parallelized", 2, 2),
                                        ("armC-vectorized-4", "pymc_jax_gpu_vectorized", 4, 1)):
            c = Cell(cell_id=f"C/kn2/{tag}/gpu", group="C", model_key="kn2", sampler=sampler,
                     device="gpu", n_devices=ndev,
                     budget={**base, "num_chains": nch},
                     note=f"re-test of docs/GPU_SETUP.md:130 {tag}")
            add(c)

    # ---- E: does the sbi/torch arm benefit from a GPU? ----------------------------------------
    if "E" in groups:
        for dev in ("cpu-min", "gpu"):
            if dev in device_set:
                add(_mk("E", "kn2_redback", "snpe", dev, budgets))

    return cells, refused


# =============================================================================================
# Environment capture
# =============================================================================================

def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


SMI_GPU_QUERY = ("index,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,"
                 "clocks_throttle_reasons.active")


def smi_gpus():
    """One nvidia-smi sample of every visible card, or ``[]`` when nvidia-smi is unusable."""
    out = _run(["nvidia-smi", f"--query-gpu={SMI_GPU_QUERY}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    rows = []
    for ln in out.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 7:
            continue
        try:
            rows.append({"index": int(f[0]), "mem_used_mib": int(f[1]),
                         "mem_total_mib": int(f[2]), "util_pct": int(f[3]),
                         "temp_c": int(f[4]), "sm_clock_mhz": int(f[5]), "throttle": f[6]})
        except ValueError:
            continue
    return rows


def smi_compute_apps():
    out = _run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    apps = []
    for ln in out.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 3:
            try:
                apps.append({"pid": int(f[0]), "gpu_uuid": f[1], "used_mib": int(f[2])})
            except ValueError:
                pass
    return apps


#: NVML clocksThrottleReasons bits that mean "this run was slowed down". Deliberately EXCLUDES
#: GpuIdle (0x1), ApplicationsClocksSetting (0x2) and DisplayClockSetting (0x100): an idle card
#: reports 0x1 and a clock-capped card reports 0x2, and neither is contention. Measured on this
#: box: an untouched, genuinely idle A6000 reports 0x0000000000000001, so a naive `!= 0` test
#: marks every idle card as throttled.
THROTTLE_BITS = {
    0x004: "SwPowerCap", 0x008: "HwSlowdown", 0x020: "SwThermalSlowdown",
    0x040: "HwThermalSlowdown", 0x080: "HwPowerBrakeSlowdown",
}
THROTTLE_MASK = 0x004 | 0x008 | 0x020 | 0x040 | 0x080


def _throttle_active(text):
    """``(throttled, readable)`` from nvidia-smi's ``clocks_throttle_reasons.active``.

    The field is a hex bitmask on this driver (masked against :data:`THROTTLE_MASK`), reads as
    text on others (``Not Active``), and can be ``[System is not in ready state]`` when the driver
    will not answer -- which is *unknown*, not *fine*, so it sets ``readable=False``.
    """
    t = (text or "").strip()
    if t.startswith("0x"):
        try:
            return bool(int(t, 16) & THROTTLE_MASK), True
        except ValueError:
            return False, False
    low = t.lower()
    if "not active" in low:
        return False, True
    if "active" in low:
        return True, True
    return False, False


def _throttle_names(text):
    t = (text or "").strip()
    if not t.startswith("0x"):
        return []
    try:
        bits = int(t, 16)
    except ValueError:
        return []
    return [nm for bit, nm in THROTTLE_BITS.items() if bits & bit]


def _pkg_versions():
    import importlib
    import importlib.util
    out = {}
    for m in ("numpy", "scipy", "pandas", "emcee", "arviz", "dynesty", "sbi", "torch", "jax",
              "jaxlib", "numpyro", "pymc", "redback", "sncosmo", "psutil", "sklearn"):
        if importlib.util.find_spec(m) is None:
            out[m] = None
            continue
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except Exception as exc:
            out[m] = f"import-failed: {type(exc).__name__}"
    return out


def _git_sha():
    """Short HEAD sha. ``git`` is often absent from these containers, so fall back to reading
    ``.git`` directly rather than stamping the run 'unknown'."""
    sha = _run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"])
    if sha:
        return sha
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = REPO / ".git" / head[5:]
            if ref.exists():
                return ref.read_text().strip()[:7]
            packed = (REPO / ".git" / "packed-refs").read_text().splitlines()
            for ln in packed:
                if ln.endswith(" " + head[5:]):
                    return ln.split()[0][:7]
            return "unknown"
        return head[:7]
    except Exception:
        return "unknown"


def env_snapshot():
    import importlib.util
    git_sha = _git_sha()
    dirty = _run(["git", "-C", str(REPO), "status", "--porcelain"])
    dirty = bool(dirty) if dirty is not None else None
    return {
        "hostname": socket.gethostname(),
        "container_hint": os.environ.get("HOSTNAME", ""),
        "git_sha": git_sha, "git_dirty": dirty,
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "loadavg_at_start": list(os.getloadavg()),
        "packages": _pkg_versions(),
        "has_jax": importlib.util.find_spec("jax") is not None,
        "nvidia_smi_usable": bool(smi_gpus()),
        "gpus": smi_gpus(),
        "compute_apps": smi_compute_apps(),
        "env_pins_orchestrator": {k: os.environ.get(k) for k in
                                  ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "JAX_PLATFORMS",
                                   "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS",
                                   "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# =============================================================================================
# Pre-flight and worker environment
# =============================================================================================

def preflight_gpu(n=1, requested=None):
    """``(ids, snapshot, reason)``. ``ids == []`` means: abort the GPU arm, do not downgrade.

    Both tests are required, exactly as ``whisper_cbpf.backends._env.gpu_list`` applies them: a
    device at 5 MiB and 100% utilisation is BUSY, and a device at 0% holding 20 GB will OOM us.
    """
    snap = smi_gpus()
    if not snap:
        return [], snap, "nvidia-smi is not usable in this container (no NVML): no GPU arm here"
    idle = [g["index"] for g in snap
            if g["mem_used_mib"] < GPU_IDLE_MEM_MIB and g["util_pct"] < GPU_IDLE_UTIL_PCT]
    if requested is not None:
        want = list(requested)
        busy = [i for i in want if i not in idle]
        if busy:
            return want, snap, (f"requested GPU(s) {busy} are NOT idle in both memory and "
                                f"utilisation (idle right now: {idle or 'none'})")
        return want, snap, ""
    if len(idle) < n:
        return [], snap, (f"only {len(idle)} of {n} requested GPUs are idle in BOTH memory "
                          f"(<{GPU_IDLE_MEM_MIB} MiB) and utilisation (<{GPU_IDLE_UTIL_PCT}%): "
                          f"idle={idle}. Refusing to downgrade onto a contended card.")
    return idle[:n], snap, ""


def worker_env(cell, gpu_ids, mem_fraction=None):
    env = dict(os.environ)
    env.update(THREAD_PINS)
    env["PYTHONUNBUFFERED"] = "1"
    if cell.device == "cpu-min":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop("JAX_PLATFORMS", None)
    elif cell.device == "cpu-jax":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"          # read by JAX at import; must precede python
    else:
        env.pop("JAX_PLATFORMS", None)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            mem_fraction if mem_fraction is not None
            else env.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4"))
    return env


# =============================================================================================
# Resource monitor (orchestrator side)
# =============================================================================================

class ResourceMonitor:
    """Polls nvidia-smi and the host load while one cell runs. SAMPLED, not integrated.

    ``interval_s`` is the sampling period; a cell shorter than a few periods yields a handful of
    points and the verdict records ``smi_n_samples`` so nobody reads a mean off two samples.
    """

    GPU_HEADER = ["ts", "cell_id", "gpu_index", "mem_used_mib", "util_pct", "temp_c",
                  "sm_clock_mhz", "throttle", "n_compute_apps_on_uuid"]
    CPU_HEADER = ["ts", "cell_id", "loadavg1", "loadavg5"]

    def __init__(self, cell_id, gpu_ids, out_dir, interval_s=MONITOR_INTERVAL_S):
        self.cell_id = cell_id
        self.gpu_ids = list(gpu_ids or [])
        self.out_dir = Path(out_dir)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
        self.load_samples = []
        self.own_pid_seen = False

    def _write(self, path, header, rows):
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header)
            w.writerows(rows)

    def _loop(self):
        gpu_rows, cpu_rows = [], []
        while not self._stop.is_set():
            ts = time.time()
            try:
                la = os.getloadavg()
                self.load_samples.append(la[0])
                cpu_rows.append([f"{ts:.3f}", self.cell_id, f"{la[0]:.2f}", f"{la[1]:.2f}"])
            except OSError:
                pass
            if self.gpu_ids:
                apps = smi_compute_apps()
                for g in smi_gpus():
                    if g["index"] not in self.gpu_ids:
                        continue
                    self.samples.append(g)
                    gpu_rows.append([f"{ts:.3f}", self.cell_id, g["index"], g["mem_used_mib"],
                                     g["util_pct"], g["temp_c"], g["sm_clock_mhz"],
                                     g["throttle"], len(apps)])
            self._stop.wait(self.interval_s)
        if gpu_rows:
            self._write(self.out_dir / "gpu_trace.csv", self.GPU_HEADER, gpu_rows)
        if cpu_rows:
            self._write(self.out_dir / "cpu_trace.csv", self.CPU_HEADER, cpu_rows)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3 + 5)

    def verdict(self, own_footprint_bytes=None, baseline_mib=None, n_cores_used=1):
        v = {"monitor_interval_s": self.interval_s,
             "smi_n_samples": len(self.samples),
             "load_n_samples": len(self.load_samples),
             "sampled_estimate": True,
             "covers": "the whole worker subprocess (import, warm-up, every repeat, metrics), "
                       "not only the timed fit -- so mean utilisation is diluted by setup"}
        if self.samples:
            mem = [s["mem_used_mib"] for s in self.samples]
            util = [s["util_pct"] for s in self.samples]
            thr = [_throttle_active(s["throttle"]) for s in self.samples]
            v["smi_peak_mem_mib"] = int(max(mem))
            v["smi_mean_util_pct"] = float(np.mean(util))
            v["smi_max_util_pct"] = int(max(util))
            v["throttled"] = bool(any(a for a, ok in thr))
            v["throttle_readable"] = bool(all(ok for _, ok in thr))
            v["throttle_reasons"] = sorted({nm for s in self.samples
                                            for nm in _throttle_names(s["throttle"])})
            excess = None
            if own_footprint_bytes:
                # Our own footprint as nvidia-smi sees it is the PREALLOCATED POOL, not the peak
                # we actually used: with XLA_PYTHON_CLIENT_MEM_FRACTION set (this repo's env
                # script sets 0.4), XLA grabs that fraction of the card at first use. Measured:
                # our worker showed 19,706 MiB in nvidia-smi against a 19,422 MiB bytes_limit and
                # a 32 MiB peak_bytes_in_use. Subtracting peak_bytes_reserved instead (0 on
                # jaxlib 0.10.2) would call every one of our own runs "contended".
                own_mib = own_footprint_bytes / (1024 ** 2)
                base = baseline_mib or 0
                excess = float(max(mem) - own_mib - base)
                v["smi_own_footprint_mib"] = own_mib
                v["smi_mem_excess_mib"] = excess
            # CONTENDED and POWER-CAPPED are different failures and must not be one flag.
            #
            # A card SHARED with another job gives an irreproducible timing: it measures the
            # neighbour. That invalidates the number.
            #
            # A card at its own POWER LIMIT gives a reproducible timing that is exactly what a user
            # gets. An A6000 under sustained compute sits at SwPowerCap essentially always -- in the
            # production matrix, 12 of 15 GPU cells reported it while foreign memory never exceeded
            # 727 MiB against a 2048 MiB threshold. Folding the two together marked every sustained
            # GPU measurement untrusted, i.e. the harness could never publish a GPU number at all.
            #
            # So: throttling ANNOTATES a row, foreign occupancy INVALIDATES it.
            v["contended_rule"] = (f"nvidia-smi peak exceeds our own pool + the pre-run baseline by "
                                   f"more than {GPU_FOREIGN_SLACK_MIB} MiB OR the card already held "
                                   f"more than {GPU_IDLE_MEM_MIB} MiB before we started. Power "
                                   f"throttling is reported separately in 'throttled' and does NOT "
                                   f"clear trust_timing.")
            v["contended"] = bool((excess is not None and excess > GPU_FOREIGN_SLACK_MIB)
                                  or (baseline_mib is not None and baseline_mib > GPU_IDLE_MEM_MIB))
        else:
            v["contended"] = False
            v["throttled"] = False
        v["foreign_pid_detection"] = "unreliable_in_container (PID namespace not mappable)"
        if self.load_samples:
            n_cpu = float(os.cpu_count() or 1)
            v["loadavg1_max"] = float(max(self.load_samples))
            v["loadavg1_mean"] = float(np.mean(self.load_samples))
            v["cpu_count"] = int(n_cpu)
            v["cpu_headroom_cores"] = float(n_cpu - v["loadavg1_mean"])
            # OVERSUBSCRIPTION, not "the box is busy". The design's rule was
            # `loadavg1 - our_cores > 8`, which on a 96-core host flags a serial cell whenever
            # anything else is running at all -- it marked 3 of 4 smoke cells untrusted at
            # loadavg 15/96, where 80 cores were idle. The defensible test is whether the runnable
            # queue plus our own demand exceeds the machine: only then were we denied cores we
            # asked for. loadavg1_mean is recorded in every row so a reader can apply a stricter
            # rule; note that even with headroom a shared box perturbs timings through cache and
            # memory bandwidth, which no loadavg test can see.
            v["cpu_contended_rule"] = "loadavg1_mean + n_cores_used > cpu_count"
            v["cpu_contended"] = bool(v["loadavg1_mean"] + float(n_cores_used) > n_cpu)
        return v


# =============================================================================================
# Orchestrator: run one cell in its own process
# =============================================================================================

def run_cell_subprocess(cell, out_dir, gpu_ids, timeout_s, monitor_interval, baseline_mib=None,
                        quiet=False):
    spec_path = out_dir / "specs" / f"{cell.cell_id.replace('/', '__')}.json"
    res_path = out_dir / "cells" / f"{cell.cell_id.replace('/', '__')}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    # A stale record from an earlier attempt would be read back as this attempt's result if the
    # worker died before writing. Delete first, so "no file" unambiguously means "the worker did
    # not finish".
    res_path.unlink(missing_ok=True)
    spec = {"cell": asdict(cell), "result_path": str(res_path)}
    spec_path.write_text(json.dumps(spec, indent=2))

    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", str(spec_path)]
    env = worker_env(cell, gpu_ids)
    t0 = time.perf_counter()
    with ResourceMonitor(cell.cell_id, gpu_ids if cell.device == "gpu" else [], out_dir,
                         monitor_interval) as mon:
        try:
            proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True,
                                  timeout=timeout_s)
            rc, stdout, stderr, timed_out = proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as exc:
            rc, timed_out = -9, True
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    subprocess_wall = time.perf_counter() - t0

    if res_path.exists():
        rec = json.loads(res_path.read_text())
    else:
        rec = {**asdict(cell), "status": "timeout" if timed_out else "worker_failed",
               "error": (stderr or stdout or "")[-1500:]}
    rec["subprocess_wall_s"] = subprocess_wall
    rec["worker_returncode"] = rc
    rec["gpu_ids"] = list(gpu_ids) if cell.device == "gpu" else []
    if timed_out:
        rec["status"] = "timeout"
    rec["stderr_tail"] = (stderr or "")[-800:]

    r = rec.get("resources") if isinstance(rec.get("resources"), dict) else {}
    own_footprint = max(int(r.get("gpu_peak_bytes_reserved") or 0),
                        int(r.get("gpu_bytes_limit") or 0)) or None
    rec["contention"] = mon.verdict(own_footprint_bytes=own_footprint,
                                    baseline_mib=baseline_mib,
                                    n_cores_used=max(1, cell.n_jobs))
    rec["trust_timing"] = bool(rec.get("status") == "works"
                               and not rec["contention"].get("contended")
                               and not rec["contention"].get("cpu_contended"))
    res_path.write_text(json.dumps(rec, indent=2, default=str))
    if not quiet:
        _print_cell_line(rec)
    return rec


def _print_cell_line(rec):
    st = rec.get("status", "?")
    if st != "works":
        print(f"    -> {st}: {str(rec.get('error', ''))[:140]}")
        return
    t = rec.get("timing", {})
    p = rec.get("posterior", {})
    trust = "ok" if rec.get("trust_timing") else "UNTRUSTED"
    print(f"    -> t_warm={t.get('t_warm_s', float('nan')):.2f}s  "
          f"t_cold={t.get('t_cold_s', float('nan')):.2f}s  "
          f"cold_start={t.get('cold_start_s', float('nan')):.2f}s  "
          f"ESS_worst={p.get('ess_worst', float('nan')):.0f} ({p.get('ess_kind')})  [{trust}]")


def scale_budget(cell, pilot_rec):
    """Stage-2 budget: scale the sampler's knob so ``ESS_worst`` should reach the target."""
    post = (pilot_rec or {}).get("posterior") or {}
    ess = post.get("ess_worst")
    knob = BUDGET_KNOB.get(cell.sampler)
    if not knob or knob not in cell.budget or not ess or not np.isfinite(ess) or ess <= 0:
        return cell, 1.0, "no pilot ESS or no knob: budget unchanged"
    scale = float(np.clip(QUALITY_TARGET_ESS / float(ess), *BUDGET_SCALE_CLIP))
    b = dict(cell.budget)
    b[knob] = int(max(1, round(b[knob] * scale)))
    return replace(cell, budget=b), scale, f"{knob} x{scale:.2f} from pilot ESS_worst={ess:.0f}"


# =============================================================================================
# Worker
# =============================================================================================

class DeviceAssertError(RuntimeError):
    pass


def _device_evidence(cell):
    import importlib.util
    ev = {"device_requested": cell["device"],
          "jax_importable": importlib.util.find_spec("jax") is not None,
          "env_pins": {k: os.environ.get(k) for k in
                       ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "JAX_PLATFORMS",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS", "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
          "ld_library_path_has_nvidia": "nvidia/" in (os.environ.get("LD_LIBRARY_PATH") or "")}
    if ev["jax_importable"]:
        import jax
        jax.config.update("jax_enable_x64", bool(cell["x64"]))   # BEFORE the first array exists
        import whisper_cbpf as wp
        ok, msg = wp.check_gpu()
        ev.update(jax_version=jax.__version__,
                  jax_devices=[str(d) for d in jax.devices()],
                  jax_default_backend=str(jax.default_backend()),
                  check_gpu_ok=bool(ok), check_gpu_msg=str(msg),
                  x64=bool(jax.config.jax_enable_x64))
    try:
        import torch
        ev["torch_cuda"] = bool(torch.cuda.is_available())
        ev["torch_device_name"] = (torch.cuda.get_device_name(0) if ev["torch_cuda"] else None)
    except Exception:
        ev["torch_cuda"] = None
    return ev


def _assert_device(cell, ev):
    """Hard-fail rather than silently benchmarking a CPU fallback."""
    dev = cell["device"]
    if dev == "cpu-min":
        if ev["jax_importable"]:
            raise DeviceAssertError(
                "device='cpu-min' means the minimal install the package advertises, but JAX is "
                "importable in this interpreter. Run this arm in the no-JAX container, or use "
                "device='cpu-jax'.")
        return
    if not ev["jax_importable"]:
        raise DeviceAssertError(f"device={dev!r} needs JAX; it is not importable here.")
    if dev == "cpu-jax" and ev.get("jax_default_backend") != "cpu":
        raise DeviceAssertError(
            f"device='cpu-jax' requires jax.default_backend()=='cpu', got "
            f"{ev.get('jax_default_backend')!r} (devices={ev.get('jax_devices')}). JAX reads "
            f"JAX_PLATFORMS/CUDA_VISIBLE_DEVICES at import; the worker env did not take.")
    if dev == "gpu":
        if not ev.get("check_gpu_ok"):
            raise DeviceAssertError("wp.check_gpu() is False -- refusing to report a CPU fallback "
                                    f"as a GPU measurement.\n{ev.get('check_gpu_msg')}")
        if ev.get("jax_default_backend") != "gpu":
            raise DeviceAssertError(
                f"jax.default_backend()=={ev.get('jax_default_backend')!r}, expected 'gpu'.")
        if cell["sampler"] in ("snpe", "npe") and not ev.get("torch_cuda"):
            raise DeviceAssertError(
                "a GPU SNPE cell requires torch.cuda.is_available(); SNPE only WARNS and falls "
                "back to CPU (whisper_cbpf/samplers/snpe.py:376), which would be reported as a "
                "GPU number.")


def _register_for_cell(model_key, has_jax):
    token, want_name, kind = MODEL_KEYS[model_key]
    out = _register_models(has_jax, {token})
    if not out:
        raise RuntimeError(f"model_key {model_key!r} could not be registered "
                           f"(token={token!r}, has_jax={has_jax})")
    if want_name is not None:
        for label, name, k in out:
            if name == want_name:
                return name, k
        raise RuntimeError(f"{want_name!r} not among registered: {[n for _, n, _ in out]}")
    return out[0][1], out[0][2]


def _prior_median_params(prior):
    return {nm: float(np.median([prior.distributions[nm].sample(np.random.default_rng(7 + i))
                                 for i in range(9)])) for nm in prior.names}


def _warmup_model(model_name, lc, prior):
    """One blocked ``predict`` before any timer.

    The JAX models' numpy-contract ``predict`` is a ``jax.jit``ed call
    (whisper_cbpf/models/jax/_factories.py:156), so even a 'CPU' sampler pays an XLA compile on its
    first evaluation. Charging that to the sampler would be wrong; it is reported on its own.
    """
    import whisper_cbpf as wp
    model = wp.get_model(model_name)
    params = _prior_median_params(prior)
    t0 = time.perf_counter()
    out = np.asarray(model.predict(params, np.asarray(lc.time, dtype=float),
                                   np.asarray(lc.band)), dtype=float)
    float(out.sum())                                     # force any lazy device computation
    return time.perf_counter() - t0


def _rusage_now():
    import resource
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"utime": s.ru_utime + c.ru_utime, "stime": s.ru_stime + c.ru_stime,
            "maxrss_kb": max(s.ru_maxrss, c.ru_maxrss)}


def _gpu_mem_stats(cell):
    out = {"gpu_peak_source": None, "gpu_peak_bytes_in_use": None,
           "gpu_peak_bytes_reserved": None, "gpu_bytes_limit": None, "torch_peak_bytes": None}
    if cell["device"] == "gpu":
        try:
            import jax
            st = jax.devices()[0].memory_stats() or {}
            out.update(gpu_peak_source="jax.Device.memory_stats",
                       gpu_peak_bytes_in_use=st.get("peak_bytes_in_use"),
                       gpu_peak_bytes_reserved=st.get("peak_bytes_reserved"),
                       gpu_bytes_limit=st.get("bytes_limit"))
        except Exception as exc:
            out["gpu_peak_source"] = f"unavailable: {type(exc).__name__}"
        try:
            import torch
            if torch.cuda.is_available():
                out["torch_peak_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
    return out


def _samples_by_chain(result, names):
    sbc = getattr(result, "samples_by_chain", None)
    if sbc is not None:
        arr = np.asarray(sbc, dtype=float)
        return arr, ("iid_assumed" if result.sampler in IID_ASSUMED else "chain")
    if result.sampler == "mcmc" and getattr(result, "emcee_sampler", None) is not None:
        info = result.info
        chain = result.emcee_sampler.get_chain(discard=int(info.get("burnin", 0)),
                                               thin=int(info.get("thin", 1)))   # (steps, walk, k)
        return np.swapaxes(np.asarray(chain, dtype=float), 0, 1), "chain"
    return result.samples[names].to_numpy(dtype=float)[None, :, :], "iid_assumed"


def _forward_evals(result, cell):
    """The forward-model ledger. ``fit_evals`` is the posterior computation; ``overhead_evals`` is
    what the metrics cost. On a small ABC cell the metrics can cost more than the posterior --
    which is exactly why they are two columns."""
    s, info = result.sampler, result.info
    n_acc = int(info.get("n_accepted") or result.n_samples)
    ovh = min(result.n_samples, PREDICTIVE_DRAWS) + 1
    ovh_note = f"predictive_metrics min(n_samples,{PREDICTIVE_DRAWS}) + band_metrics 1"
    fit_evals, exact, reason, launches = None, False, None, None

    if s == "abc":
        fit_evals, exact = int(info["n_simulations"]), True
        ovh += min(MAX_LOGL_SCAN, n_acc)
        ovh_note += f" + max-logl scan min({MAX_LOGL_SCAN}, n_accepted={n_acc})"
    elif s == "abc_smc":
        fit_evals, exact = int(info["total_simulations"]), False
        reason = ("UPPER BOUND. info['total_simulations'] counts ATTEMPTS; from round 1 on, an "
                  "attempt whose perturbed theta falls outside the prior support is rejected "
                  "before predict is called (samplers/abc_smc.py:77-78). Measured by "
                  "--validate-evals on bazin (n_particles=12, n_rounds=2, n_jobs=1): formula 425, "
                  "observed 275 predict calls.")
        ovh += min(MAX_LOGL_SCAN, n_acc)
        ovh_note += f" + max-logl scan min({MAX_LOGL_SCAN}, n_accepted={n_acc})"
    elif s == "abc_gpu":
        fit_evals, exact = int(info["n_simulations"]), True
        launches = int(np.ceil(fit_evals / max(int(info.get("chunk", 16)), 1)))
        ovh += min(MAX_LOGL_SCAN, n_acc)
        ovh_note += f" + max-logl scan min({MAX_LOGL_SCAN}, n_accepted={n_acc}) (batched)"
    elif s == "abc_smc_gpu":
        tot = info.get("total_simulations")
        if tot is None:
            tot = sum(int(r.get("n_simulations", 0)) for r in info.get("rounds", []))
        fit_evals, exact = int(tot), True
    elif s in ("mcmc", "emcee_jax"):
        nw, ns = int(info["nwalkers"]), int(info["nsteps"])
        fit_evals, exact = nw * ns + nw, False
        reason = ("UPPER BOUND on FORWARD-MODEL calls. emcee evaluates the log-posterior "
                  "nwalkers*(nsteps+1) times, but _log_prob returns -inf WITHOUT calling predict "
                  "for a proposal outside the prior support (samplers/mcmc.py:46-48), so the "
                  "forward-model count is lower by the number of out-of-support proposals and "
                  "depends on the posterior's position in the prior box. Measured by "
                  "--validate-evals on bazin (nwalkers=8, nsteps=25): formula 369 total, observed "
                  "280 predict calls.")
        try:                                             # cross-check against emcee's own record
            got = int(result.emcee_sampler.get_log_prob().size)
            if got != nw * ns:
                reason += f" [emcee reports {got} log-prob evals, formula gives {nw * ns}]"
        except Exception:
            pass
    elif s in ("snpe", "npe"):
        fit_evals, exact = int(info["total_simulations"]), False
        reason = ("LOWER BOUND by a fixed +3. info['total_simulations'] is num_rounds x "
                  "num_simulations, but --validate-evals measured a constant 3 extra predict "
                  "calls at num_simulations in {40, 60, 100} on bazin -- budget-independent "
                  "setup calls whose source was not attributed.")
        ovh += min(MAX_LOGL_SCAN, result.n_samples)
        ovh_note += f" + max-logl scan min({MAX_LOGL_SCAN}, n_samples)"
    elif s in ("nuts_gpu", "pymc_jax_gpu_vectorized", "pymc_jax_gpu_parallelized"):
        reason = ("leapfrog counts not collected: whisper_cbpf/samplers/jax/nuts_gpu.py:170,179 "
                  "pass extra_fields=('diverging',) only, so num_steps never reaches "
                  "result.numpyro_mcmc. Reported as unknown; adding 'num_steps' there would "
                  "settle it (as GRADIENT evaluations, ~2-4x a forward evaluation).")
    return {"fit_evals": fit_evals, "fit_evals_exact": bool(exact),
            "fit_evals_reason": reason, "fit_evals_validated": False,
            "overhead_evals": int(ovh), "overhead_evals_note": ovh_note,
            "kernel_launches": launches}


def _reference_key(cell, lc):
    import hashlib
    h = hashlib.sha256()
    h.update(CSV.read_bytes())
    h.update(repr([FIT_BANDS, BAND_ALIASES, FILTER_SET, REDSHIFT, DL_CM, T_EXPLOSION_MJD,
                   cell["model_key"], cell["space"], int(lc.n_points)]).encode())
    return h.hexdigest()[:12]


def _reference_path(cell, lc):
    return (OUT_ROOT / "_references" /
            f"{cell['model_key']}__{cell['space']}__{_reference_key(cell, lc)}.npz")


def _accuracy(draws, names, ref_path, seed=0):
    """c2st + normalised 1-Wasserstein against a cached reference posterior."""
    out = {"verdict": "not measured", "reference": None, "c2st": None, "c2st_floor": None,
           "w1_max": None, "reason": None}
    if ref_path is None or not Path(ref_path).exists():
        out["reason"] = ("no reference posterior cached for this (model, space, data); run "
                         "--build-references first")
        return out
    z = np.load(ref_path, allow_pickle=True)
    if not bool(np.asarray(z["qualifies"]).ravel()[0]):
        out["reason"] = (f"a reference exists at {Path(ref_path).name} but it FAILED its own "
                         f"acceptance criteria ({str(z['criteria'])}); comparing against an "
                         f"unconverged reference would manufacture a verdict")
        return out
    ref, ref_names = np.asarray(z["draws"], dtype=float), [str(x) for x in z["params"]]
    idx = [ref_names.index(nm) for nm in names if nm in ref_names]
    if len(idx) != len(names):
        out["reason"] = f"reference parameters {ref_names} do not cover {names}"
        return out
    ref = ref[:, idx]
    rng = np.random.default_rng(seed)
    n = int(min(2000, len(draws), len(ref)))
    a = draws[rng.choice(len(draws), n, replace=False)]
    b = ref[rng.choice(len(ref), n, replace=False)]
    try:
        from sbi.utils.metrics import c2st as _c2st
        import torch
        out["c2st"] = float(np.asarray(_c2st(torch.as_tensor(a, dtype=torch.float32),
                                             torch.as_tensor(b, dtype=torch.float32))).ravel()[0])
        half = len(ref) // 2
        r1 = ref[rng.choice(half, min(n, half), replace=False)]
        r2 = ref[half + rng.choice(len(ref) - half, min(n, len(ref) - half), replace=False)]
        out["c2st_floor"] = float(np.asarray(_c2st(torch.as_tensor(r1, dtype=torch.float32),
                                                   torch.as_tensor(r2, dtype=torch.float32))
                                             ).ravel()[0])
    except Exception as exc:
        out["reason"] = f"c2st unavailable: {type(exc).__name__}: {exc}"
    try:
        from scipy.stats import wasserstein_distance
        out["w1_max"] = float(max(wasserstein_distance(a[:, j], b[:, j]) / (np.std(b[:, j]) or 1.0)
                                  for j in range(a.shape[1])))
    except Exception as exc:
        out["reason"] = (out["reason"] or "") + f" | w1 unavailable: {type(exc).__name__}"
    if out["c2st"] is not None:
        out["verdict"] = "PASS" if out["c2st"] <= C2ST_PASS else "FAIL"
        out["reference"] = str(ref_path)
    return out


def _one_fit(lc, cell, seed):
    import whisper_cbpf as wp
    kwargs = dict(cell["budget"])
    if cell["sampler"] in ("abc", "abc_smc", "mcmc"):
        # ALWAYS explicit, never defaulted. abc's default is min(cpu_count, 8)
        # (whisper_cbpf/samplers/abc.py:146), which on a JAX-registered model raises
        # "Can't pickle local object 'kilonova_model.<locals>.predict'" -- the factories return a
        # closure, and abc's own docstring says to use n_jobs=1 for closures. Leaving it defaulted
        # would also make the parallelism an uncontrolled variable that changes with the host.
        kwargs["n_jobs"] = int(cell["n_jobs"])
    if cell["sampler"] in ("snpe", "npe"):
        kwargs["device"] = "cuda" if cell["device"] == "gpu" else "cpu"
    t0 = time.perf_counter()
    res = wp.fit(lc, cell["model_name"], sampler=cell["sampler"], space=cell["space"],
                 seed=int(seed), **kwargs)
    return res, time.perf_counter() - t0


def worker_main(spec_path):
    import warnings
    spec = json.loads(Path(spec_path).read_text())
    cell = spec["cell"]
    res_path = Path(spec["result_path"])
    rec = {**cell, "status": "worker_failed", "schema_version": SCHEMA_VERSION,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        ev = _device_evidence(cell)
        rec["device_evidence"] = ev
        _assert_device(cell, ev)

        import whisper_cbpf as wp
        from whisper_cbpf.likelihood import resolve_space
        from whisper_cbpf.metrics import ess_summary

        lc = _load_lc()
        model_name, kind = _register_for_cell(cell["model_key"], ev["jax_importable"])
        cell["model_name"] = model_name
        rec["model_name"], rec["model_kind"] = model_name, kind
        rec["model_label"] = MODEL_LABEL.get(cell["model_key"], cell["model_key"])
        rec["dataset"] = {"csv": str(CSV.relative_to(REPO)), "n_points": int(lc.n_points),
                          "bands": {b: int(np.sum(np.asarray(lc.band) == b)) for b in FIT_BANDS},
                          "data_mode": str(lc.data_mode),
                          "resolved_space": str(resolve_space(lc, cell["space"])),
                          "window_days": [0.4, 12.0], "redshift": REDSHIFT, "dl_cm": DL_CM}

        prior = wp.get_model(model_name).default_prior
        rec["model_warmup_s"] = _warmup_model(model_name, lc, prior)

        walls, results, warn_counts = [], [], []
        for i in range(int(cell["repeats"])):
            ru0 = _rusage_now()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res, wall = _one_fit(lc, cell, cell["seed"])
            ru1 = _rusage_now()
            walls.append(wall)
            warn_counts.append(len([c for c in caught if issubclass(c.category, UserWarning)]))
            if i == 0 or i == int(cell["repeats"]) - 1:
                results.append((res, ru0, ru1))
        res, ru0, ru1 = results[-1]

        t_cold = float(walls[0])
        warm = walls[1:] or walls
        t_warm = float(np.median(warm))
        cold_start = float(t_cold - t_warm)
        reported = res.info.get("compile_time_s")
        breakdown = {k: res.info.get(k) for k in ("warmup_time_s", "sampling_time_s")
                     if res.info.get(k) is not None}
        disagree = None
        if reported is not None and cold_start > 0:
            disagree = bool(abs(reported - cold_start) / max(cold_start, 1e-9) > 0.25)

        names = list(res.parameters)
        sbc, ess_kind = _samples_by_chain(res, names)
        ess = ess_summary(sbc, names, t_warm)
        acc = _accuracy(sbc.reshape(-1, sbc.shape[-1]), names,
                        _reference_path(cell, lc) if cell["role"] == "cell" else None,
                        seed=int(cell["seed"]))
        if cell["role"] == "reference":
            # Cache the draws so the accuracy gate has something to compare against, and stamp
            # them with whether they met their own acceptance criteria -- an unconverged
            # reference must not be silently reused as if it were one.
            qual = _reference_qualifies(
                {"posterior": {"max_rhat": res.info.get("max_rhat"),
                               "n_divergences": res.info.get("n_divergences"),
                               "ess_worst": ess["ess_worst"],
                               "converged": res.info.get("converged")}}, cell["sampler"])
            path = _reference_path(cell, lc)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, draws=sbc.reshape(-1, sbc.shape[-1]),
                                params=np.asarray(names), qualifies=np.asarray([bool(qual)]),
                                criteria=np.asarray(str(cell.get("note", ""))))
            rec["reference_path"] = str(path)
            rec["reference_qualifies"] = bool(qual)

        pm = res.info.get("predictive_metrics") or {}
        w = pm.get("waic") if isinstance(pm.get("waic"), dict) else None
        core_s = (ru1["utime"] + ru1["stime"]) - (ru0["utime"] + ru0["stime"])

        rec.update(
            status="works",
            budget_effective=dict(cell["budget"]),
            timing={"t_cold_s": t_cold, "t_warm_s": t_warm, "t_all_s": [float(x) for x in walls],
                    "cold_start_s": cold_start,
                    "cold_start_s_is": "t_cold - t_warm in ONE process. An UPPER BOUND on XLA "
                                       "compile, not a compile measurement: it also absorbs "
                                       "allocator first-touch, autotune and first-call library "
                                       "imports inside the metrics (nonzero even on pure-numpy "
                                       "cells, where nothing compiles).",
                    "compile_s_reported": (float(reported) if reported is not None else None),
                    "compile_disagreement": disagree,
                    "model_warmup_s": rec["model_warmup_s"],
                    "sampler_breakdown": breakdown,
                    "sampler_runtime_s": float(res.runtime_s)},
            evals=_forward_evals(res, cell),
            posterior={"n_draws": int(ess["n_draws"]), "ess_kind": ess_kind,
                       "ess_worst": ess["ess_worst"], "ess_worst_param": ess["ess_worst_param"],
                       "ess_by_param": ess["ess_by_param"],
                       "ess_worst_per_sec_warm": ess["ess_worst_per_sec"],
                       "ess_worst_per_sec_cold": (ess["ess_worst"] / t_cold
                                                  if t_cold > 0 else None),
                       "max_rhat": res.info.get("max_rhat"),
                       "n_divergences": res.info.get("n_divergences"),
                       "mean_acceptance_fraction": res.info.get("mean_acceptance_fraction"),
                       "converged": res.info.get("converged")},
            accuracy=acc,
            model_selection={"aic": float(res.aic), "bic": float(res.bic),
                             "waic": (w or {}).get("waic"),
                             "p_waic_reliable": (w or {}).get("p_waic_reliable"),
                             "predictive_metrics_error": res.info.get("predictive_metrics_error")},
            resources={**_gpu_mem_stats(cell),
                       "cpu_core_seconds": float(core_s),
                       "cpu_core_seconds_source": "getrusage(SELF)+getrusage(CHILDREN) delta "
                                                  "across the fit (exact; counts any child reaped)",
                       "parallel_efficiency": (float(core_s / (t_warm * max(cell["n_jobs"], 1)))
                                               if t_warm > 0 else None),
                       "rss_peak_mib": float(ru1["maxrss_kb"]) / 1024.0},
            sampler_info={k: v for k, v in res.info.items()
                          if k not in ("band_metrics", "predictive_metrics", "rounds", "rhat")},
            warnings=int(max(warn_counts)) if warn_counts else 0,
        )
    except DeviceAssertError as exc:
        rec.update(status="device_assert_failed", error=str(exc))
    except Exception as exc:                        # noqa: BLE001 - recorded, never hidden
        import traceback
        rec.update(status="broken", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-3000:])
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(rec, indent=2, default=str))
    return 0 if rec["status"] == "works" else 1


# =============================================================================================
# Forward-evaluation ledger validation
# =============================================================================================

def validate_evals(out_dir, n_sim=60):
    """Check the ``_forward_evals`` formulas against a counting shim. Runs in-process, numpy only.

    Registers ``bazin__counted``, whose ``predict`` increments a counter, and compares the observed
    call count with ``fit_evals + overhead_evals``.

    **Where this cannot reach**, and why those rows stay formula-only:
    * ``n_jobs > 1`` -- the counter lives in the forked child and dies with it.
    * the JAX samplers -- they call ``predict_jax``/``log_prob_jax``, not ``predict``.
    """
    import whisper_cbpf as wp
    lc = _load_lc()
    _register_models(False, {"toys"})
    base = wp.get_model("bazin")
    counter = {"n": 0}

    def counted(parameters, times, bands=None):
        counter["n"] += 1
        return base.predict(parameters, times, bands)

    wp.register_model("bazin__counted", counted, base.parameters, prior=base.default_prior,
                      description="counting shim for the benchmark's eval ledger", overwrite=True)

    plan = [("abc", dict(n_simulations=n_sim, quantile=0.2, n_jobs=1)),
            ("abc_smc", dict(n_particles=12, n_rounds=2, n_jobs=1)),
            ("mcmc", dict(nsteps=25, burnin=5, nwalkers=8, thin=1, n_jobs=1)),
            ("snpe", dict(num_rounds=1, num_simulations=n_sim, num_samples=100))]
    rows = []
    for sampler, budget in plan:
        counter["n"] = 0
        try:
            res = wp.fit(lc, "bazin__counted", sampler=sampler, space="magnitude", seed=0, **budget)
        except Exception as exc:
            rows.append({"sampler": sampler, "status": f"{type(exc).__name__}: {exc}"})
            continue
        led = _forward_evals(res, {"sampler": sampler})
        pred = (led["fit_evals"] or 0) + led["overhead_evals"]
        rows.append({"sampler": sampler, "status": "ok", "budget": budget,
                     "observed_predict_calls": counter["n"], "fit_evals": led["fit_evals"],
                     "declared_exact": led["fit_evals_exact"],
                     "overhead_evals": led["overhead_evals"], "predicted_total": pred,
                     "delta": counter["n"] - pred, "matches": counter["n"] == pred,
                     "fit_evals_reason": led["fit_evals_reason"],
                     "overhead_note": led["overhead_evals_note"]})
    (Path(out_dir) / "validate_evals.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n{'sampler':<10} {'observed':>10} {'formula':>10} {'delta':>8}  {'declared':<12} "
          f"verdict")
    bad = []
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['sampler']:<10} {'-':>10} {'-':>10} {'-':>8}  {'-':<12} {r['status'][:60]}")
            bad.append(r["sampler"])
            continue
        decl = "exact" if r["declared_exact"] else "bound"
        if r["declared_exact"]:
            ok = r["matches"]
            bad += [] if ok else [r["sampler"]]
            verdict = "CONFIRMED exact" if ok else "BROKEN: declared exact but does not match"
        else:
            verdict = "bound confirmed (delta is the documented gap)"
        print(f"{r['sampler']:<10} {r['observed_predict_calls']:>10} {r['predicted_total']:>10} "
              f"{r['delta']:>+8}  {decl:<12} {verdict}")
    print(f"\nwritten to {Path(out_dir) / 'validate_evals.json'}")
    print("A 'bound' row is not a failure: the formula is documented as an upper/lower bound in "
          "_forward_evals, and this run is where that documentation's numbers come from.")
    return rows


# =============================================================================================
# References
# =============================================================================================

REFERENCE_SPECS = {
    # (model_key) -> (sampler, budget, acceptance criteria description)
    "kn1": ("nuts_gpu", dict(num_warmup=4000, num_samples=4000, num_chains=4,
                             target_accept_prob=0.9),
            "max_rhat<1.01, n_divergences==0, ESS_worst>2000"),
    # kn2 uses NESTED, not NUTS, and that is not a preference -- it is structural. The
    # 2-component AT2017GFO posterior is MULTIMODAL (OPEN_ITEMS: "four chains in three distinct
    # modes"), and NUTS is a local sampler that cannot hop between separated modes AT ANY WARMUP
    # BUDGET. Measured here: nuts_gpu at 4000+4000 x 4 chains gave ESS_worst = 7.1 with r-hat 1.53
    # and ZERO divergences -- the signature of chains that are individually healthy and sitting in
    # different places. It was correctly refused as a reference. OPEN_ITEMS' own remedy list says
    # "use a multimodal sampler (abc_smc_gpu, or dynesty on CPU)"; dynesty exists here now.
    "kn2": ("nested", dict(nlive=1000, dlogz=0.1),
            "converged and n_effective>2000 (nested integrates across modes; NUTS cannot)"),
    "kn2_redback": ("mcmc", dict(nwalkers=64, nsteps=40000, burnin=10000, thin=20),
                    "nsteps>50*tau (the guard in samplers/mcmc.py:185)"),
    "bazin": ("mcmc", dict(nwalkers=32, nsteps=20000, burnin=5000, thin=10),
              "nsteps>50*tau"),
}


def build_references(out_dir, device, gpu_ids, model_keys, timeout_s, monitor_interval):
    """Build the reference posteriors the accuracy gate compares against.

    emcee is an exact-target sampler, so a long emcee chain is a legitimate reference for the
    numpy arm, where no gradient reference can exist.
    """
    (OUT_ROOT / "_references").mkdir(parents=True, exist_ok=True)
    made = []
    for mk in model_keys:
        if mk not in REFERENCE_SPECS:
            print(f"  [skip] no reference spec for {mk!r}")
            continue
        sampler, budget, criteria = REFERENCE_SPECS[mk]
        dev = "cpu-min" if MODEL_KEYS[mk][2] in ("toy", "redback") and device == "cpu" else (
            "gpu" if device == "gpu" else "cpu-jax")
        cell = Cell(cell_id=f"ref/{mk}/{sampler}/{dev}", group="ref", model_key=mk,
                    sampler=sampler, device=dev, x64=(sampler == "nuts_gpu"),
                    budget=budget, repeats=1, role="reference",
                    note=f"reference posterior; acceptance: {criteria}")
        why = impossible_reason(cell.model_kind, cell.sampler, cell.space, cell.has_jax)
        if why:
            print(f"  [refused] {cell.cell_id}: {why}")
            continue
        print(f"  building reference {cell.cell_id} ({criteria}) ...", flush=True)
        rec = run_cell_subprocess(cell, out_dir, gpu_ids, timeout_s, monitor_interval)
        if rec.get("status") != "works":
            print(f"  [FAILED] {cell.cell_id}: {rec.get('error', '')[:200]}")
            continue
        qual = bool(rec.get("reference_qualifies"))
        made.append({"model_key": mk, "sampler": sampler, "criteria": criteria,
                     "cell": rec["cell_id"], "diagnostics": rec.get("posterior"),
                     "path": rec.get("reference_path"), "qualifies": qual})
        print(f"  -> {'QUALIFIES' if qual else 'DOES NOT QUALIFY'} ({criteria}); "
              f"cached at {rec.get('reference_path')}")
    (out_dir / "references.json").write_text(json.dumps(made, indent=2, default=str))
    print(f"\nreference build recorded in {out_dir / 'references.json'}")
    print("A reference that fails its criteria is still cached, stamped qualifies=False, and the "
          "accuracy gate refuses it: comparing against an unconverged reference would manufacture "
          "a verdict rather than measure one.")
    return made


def _reference_qualifies(rec, sampler):
    p = rec.get("posterior") or {}
    if sampler == "nuts_gpu":
        rhat, div, ess = p.get("max_rhat"), p.get("n_divergences"), p.get("ess_worst")
        return bool(rhat is not None and rhat < 1.01 and div == 0
                    and ess is not None and ess > 2000)
    if sampler in ("nested", "dynesty"):
        # `converged` alone is too weak a bar for a REFERENCE: nested sets it to "did not hit the
        # iteration cap", which is not the same as "sampled the posterior well". Use the sampler's
        # OWN effective sample count, from the importance weights -- the harness's ess_worst treats
        # nested as iid (it has no chain axis) and would simply report the draw count, overstating
        # it. Threshold matched to the NUTS bar so the two references are comparable.
        si = rec.get("sampler_info") or {}
        neff = si.get("n_effective")
        return bool(si.get("converged") and neff is not None and neff > 2000)
    return bool(p.get("converged"))


# =============================================================================================
# Crossover sweep (Group D)
# =============================================================================================

N_SWEEP = (1_000, 3_000, 10_000, 30_000, 50_000, 100_000, 300_000)
CLAIMED_CROSSOVER = 50_000          # docs/CHOOSING.md:19, asserted with no measurement in-repo


def fit_linear(n, wall):
    n, wall = np.asarray(n, float), np.asarray(wall, float)
    if len(n) < 2 or np.ptp(n) == 0:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan")}
    b, a = np.polyfit(n, wall, 1)
    pred = a + b * n
    ss = float(np.sum((wall - pred) ** 2))
    tot = float(np.sum((wall - wall.mean()) ** 2))
    return {"a": float(a), "b": float(b), "r2": (1 - ss / tot) if tot > 0 else float("nan")}


def crossover_point(fit_cpu, fit_gpu):
    denom = fit_cpu["b"] - fit_gpu["b"]
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return float((fit_gpu["a"] - fit_cpu["a"]) / denom)


def bootstrap_crossover(rows_cpu, rows_gpu, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ns = sorted({r["n"] for r in rows_cpu} & {r["n"] for r in rows_gpu})
    if len(ns) < 2:
        return {"n_star": float("nan"), "ci_lo": None, "ci_hi": None, "n_boot": 0}
    by = lambda rows: {n: [r["wall"] for r in rows if r["n"] == n] for n in ns}  # noqa: E731
    cpu_by, gpu_by = by(rows_cpu), by(rows_gpu)
    draws = []
    for _ in range(int(n_boot)):
        c = fit_linear(ns, [rng.choice(cpu_by[n]) for n in ns])
        g = fit_linear(ns, [rng.choice(gpu_by[n]) for n in ns])
        draws.append(crossover_point(c, g))
    d = np.asarray([x for x in draws if np.isfinite(x)], dtype=float)
    point = crossover_point(fit_linear(ns, [np.median(cpu_by[n]) for n in ns]),
                            fit_linear(ns, [np.median(gpu_by[n]) for n in ns]))
    return {"n_star": float(point), "n_boot": int(len(d)),
            "ci_lo": (float(np.percentile(d, 2.5)) if len(d) else None),
            "ci_hi": (float(np.percentile(d, 97.5)) if len(d) else None)}


def verdict_vs_docs(cx, claimed=CLAIMED_CROSSOVER, baseline=""):
    lo, hi = cx.get("ci_lo"), cx.get("ci_hi")
    if lo is None or hi is None or not np.isfinite(cx.get("n_star", float("nan"))):
        return f"not measured ({baseline})"
    inside = lo <= claimed <= hi
    return (f"N* = {cx['n_star']:.0f} simulations, 95% CI [{lo:.0f}, {hi:.0f}] against {baseline}. "
            f"docs/CHOOSING.md:19's unqualified '{claimed:,}' "
            f"{'FALLS INSIDE' if inside else 'falls OUTSIDE'} this interval.")


def run_crossover(out_dir, gpu_ids, sweep, repeats, timeout_s, monitor_interval, model_key="kn1",
                  cpu_jobs=(1,), quiet=False):
    """Test the crossover claim rather than restating it.

    CONFOUND, printed with every result: the CPU arm calls ``predict`` (jitted JAX, one curve per
    call) and the GPU arm calls ``vmap(predict_jax)`` at chunk 16. N* measures whisper's CPU ABC
    against whisper's GPU ABC, not CPU silicon against GPU silicon.
    """
    rows = []
    for n in sweep:
        for jobs in cpu_jobs:
            c = Cell(cell_id=f"D/{model_key}/abc/cpu-jax/n{n}/j{jobs}", group="D",
                     model_key=model_key, sampler="abc", device="cpu-jax", n_jobs=jobs,
                     budget=dict(n_simulations=int(n), quantile=0.01), repeats=repeats)
            rec = run_cell_subprocess(c, out_dir, [], timeout_s, monitor_interval, quiet=quiet)
            if rec.get("status") == "works":
                rows.append({"arm": f"cpu-jax abc n_jobs={jobs}", "n": int(n),
                             "wall": rec["timing"]["t_warm_s"],
                             "wall_cold": rec["timing"]["t_cold_s"]})
        if gpu_ids:
            c = Cell(cell_id=f"D/{model_key}/abc_gpu/gpu/n{n}", group="D", model_key=model_key,
                     sampler="abc_gpu", device="gpu",
                     budget=dict(n_simulations=int(n), quantile=0.01), repeats=repeats)
            rec = run_cell_subprocess(c, out_dir, gpu_ids, timeout_s, monitor_interval, quiet=quiet)
            if rec.get("status") == "works":
                rows.append({"arm": "gpu abc_gpu", "n": int(n),
                             "wall": rec["timing"]["t_warm_s"],
                             "wall_cold": rec["timing"]["t_cold_s"]})

    gpu_rows = [r for r in rows if r["arm"].startswith("gpu")]
    result = {"sweep": list(sweep), "repeats": repeats, "model_key": model_key, "rows": rows,
              "claimed_crossover": CLAIMED_CROSSOVER, "crossovers": [],
              "confound": ("the CPU arm calls predict (jitted JAX, one curve per call); the GPU "
                           "arm calls vmap(predict_jax) at chunk 16. N* compares whisper's CPU "
                           "ABC with whisper's GPU ABC, not CPU silicon with GPU silicon.")}
    for jobs in cpu_jobs:
        cpu_rows = [r for r in rows if r["arm"] == f"cpu-jax abc n_jobs={jobs}"]
        if not cpu_rows or not gpu_rows:
            result["crossovers"].append({"baseline": f"cpu-jax abc n_jobs={jobs}",
                                         "verdict": "not measured (arm produced no rows)"})
            continue
        warm = bootstrap_crossover(cpu_rows, gpu_rows)
        cold = bootstrap_crossover([{**r, "wall": r["wall_cold"]} for r in cpu_rows],
                                   [{**r, "wall": r["wall_cold"]} for r in gpu_rows])
        result["crossovers"].append({
            "baseline": f"cpu-jax abc n_jobs={jobs}",
            "n_star_warm": warm, "n_star_cold": cold,
            "verdict_warm": verdict_vs_docs(warm, baseline=f"cpu-jax abc n_jobs={jobs}, warm"),
            "verdict_cold": verdict_vs_docs(cold, baseline=f"cpu-jax abc n_jobs={jobs}, cold")})
    (out_dir / "crossover.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# =============================================================================================
# Reporting (no fits here -- --report-only re-renders a finished run)
# =============================================================================================

def _f(x, fmt="{:.2f}", dash="-"):
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return dash if not np.isfinite(v) else fmt.format(v)


def _budget_str(b):
    return " ".join(f"{k}={v}" for k, v in sorted(b.items())) if b else "-"


def _device_str(rec):
    """The device *and the card*, so a GPU row is self-labelling rather than relying on the arm."""
    ids = rec.get("gpu_ids") or []
    return f"{rec['device']} (card {','.join(str(i) for i in ids)})" if ids else rec["device"]


def render_group_table(cells, group):
    rows = [c for c in cells if c.get("group") == group]
    if not rows:
        return ""
    cols = ["cell", "model", "sampler", "device", "budget", "t_warm s", "t_cold s",
            "cold-start s", "model warm s", "fit evals", "ovh evals", "ESS_worst",
            "ESS_worst/s (warm)", "ess kind", "c2st", "w1_max", "acc", "rhat", "div", "AIC",
            "BIC", "WAIC", "GPU peak MiB", "core-s", "par-eff", "trust"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        if r.get("status") != "works":
            head4 = [f"`{r['cell_id']}`", MODEL_LABEL.get(r["model_key"], r["model_key"]),
                     r["sampler"], _device_str(r), _budget_str(r.get("budget"))]
            out.append("| " + " | ".join(head4 + ["-"] * (len(cols) - 6)
                                         + [f"**{r.get('status')}**"]) + " |")
            continue
        t, p, e = r["timing"], r["posterior"], r["evals"]
        a, ms, res_ = r["accuracy"], r["model_selection"], r["resources"]
        trusted = r.get("trust_timing", False)
        gpu_mib = (res_.get("gpu_peak_bytes_in_use") / (1024 ** 2)
                   if res_.get("gpu_peak_bytes_in_use") else None)
        waic = ms.get("waic")
        waic_s = "-" if waic is None else (_f(waic, "{:.1f}")
                                           + ("" if ms.get("p_waic_reliable") else " !"))
        essps = _f(p.get("ess_worst_per_sec_warm"), "{:.1f}")
        if p.get("ess_kind") == "iid_assumed" or a.get("verdict") != "PASS":
            essps = f"~~{essps}~~" if essps != "-" else "-"
        cells_txt = [
            f"`{r['cell_id']}`", MODEL_LABEL.get(r["model_key"], r["model_key"]), r["sampler"],
            _device_str(r), _budget_str(r.get("budget")),
            ("-" if not trusted else _f(t.get("t_warm_s"))),
            ("-" if not trusted else _f(t.get("t_cold_s"))),
            ("-" if not trusted else _f(t.get("cold_start_s"))),
            _f(t.get("model_warmup_s"), "{:.3f}"),
            ("unknown" if e.get("fit_evals") is None else f"{e['fit_evals']:,}"),
            f"{e.get('overhead_evals', 0):,}",
            _f(p.get("ess_worst"), "{:.0f}"), ("-" if not trusted else essps),
            str(p.get("ess_kind")),
            _f(a.get("c2st"), "{:.3f}"), _f(a.get("w1_max"), "{:.3f}"), str(a.get("verdict")),
            _f(p.get("max_rhat"), "{:.4f}"),
            ("-" if p.get("n_divergences") in (None, -1) else str(p.get("n_divergences"))),
            _f(ms.get("aic"), "{:.1f}"), _f(ms.get("bic"), "{:.1f}"), waic_s,
            _f(gpu_mib, "{:.0f}"), _f(res_.get("cpu_core_seconds"), "{:.1f}"),
            _f(res_.get("parallel_efficiency"), "{:.2f}"),
            ("ok" if trusted else ("contended" if r.get("contention", {}).get("contended")
                                   else ("cpu-contended" if r.get("contention", {}).get(
                                       "cpu_contended") else "untrusted"))),
        ]
        assert len(cells_txt) == len(cols), (len(cells_txt), len(cols))
        out.append("| " + " | ".join(cells_txt) + " |")
    return "\n".join(out)


def render_gpu_trace_table(cells, interval_s):
    """The SAMPLED nvidia-smi summary, kept out of the main table so it cannot be mistaken for the
    exact per-process peak that sits in the `GPU peak MiB` column."""
    rows = [c for c in cells if (c.get("contention") or {}).get("smi_n_samples")]
    if not rows:
        return ""
    out = ["| cell | card | smi samples | smi peak MiB | our pool MiB | excess MiB | "
           "smi mean util % | smi max util % | throttle | verdict |",
           "|" + "---|" * 10]
    for r in rows:
        c = r["contention"]
        thr = (", ".join(c.get("throttle_reasons") or [])
               or ("none" if c.get("throttle_readable") else "unreadable"))
        out.append(
            f"| `{r['cell_id']}` | {','.join(str(i) for i in (r.get('gpu_ids') or [])) or '-'} | "
            f"{c['smi_n_samples']} | {c.get('smi_peak_mem_mib', '-')} | "
            f"{_f(c.get('smi_own_footprint_mib'), '{:.0f}')} | "
            f"{_f(c.get('smi_mem_excess_mib'), '{:.0f}')} | "
            f"{_f(c.get('smi_mean_util_pct'), '{:.1f}')} | {c.get('smi_max_util_pct', '-')} | "
            f"{thr} | {'CONTENDED' if c.get('contended') else 'clean'} |")
    return "\n".join(out)


def render_evals_validation(rows):
    out = ["| sampler | budget | observed `predict` calls | formula (fit + overhead) | delta | "
           "declared | verdict |", "|" + "---|" * 7]
    for r in rows:
        if r.get("status") != "ok":
            out.append(f"| {r['sampler']} | - | - | - | - | - | {r.get('status')} |")
            continue
        decl = "exact" if r["declared_exact"] else "bound"
        verdict = (("CONFIRMED exact" if r["matches"] else "**BROKEN — declared exact, does not "
                    "match**") if r["declared_exact"] else "bound confirmed; delta is the gap")
        out.append(f"| {r['sampler']} | {_budget_str(r.get('budget'))} | "
                   f"{r['observed_predict_calls']} | {r['fit_evals']} + {r['overhead_evals']} = "
                   f"{r['predicted_total']} | {r['delta']:+d} | {decl} | {verdict} |")
    return "\n".join(out)


GROUP_TITLES = {
    "A": ("Group A — gradient-free numpy models on the minimal install",
          "Which sampler for a model with no gradients, on the install the package advertises "
          "(`cpu-min` = no JAX)? `bazin` is the cheap analytic control: it separates fixed sampler "
          "overhead from model cost."),
    "B": ("Group B — what the GPU buys for the same JAX model",
          "`cpu-jax` is the *controlled* CPU arm: same interpreter, same code, same numerics as "
          "the GPU arm, only the device changes. That is what makes this a device comparison "
          "rather than a container comparison."),
    "C": ("Group C — three NUTS frontends, one kernel",
          "Plus a re-test of the 3-arm chain-method table in `docs/GPU_SETUP.md:130`. The "
          "`parallel` arms need `n_devices` idle cards; pre-flight aborts them rather than letting "
          "NumPyro downgrade `parallel` to `sequential` silently."),
    "E": ("Group E — does the sbi/torch arm benefit from a GPU?",
          "Nothing in the repository currently answers this."),
    "D": ("Group D — the CPU/GPU crossover", "See the crossover section."),
    "ref": ("Reference posteriors", "Targets for the accuracy gate."),
}


def render_arm(run, A, level="##"):
    """The provenance block for one invocation: box, interpreter, packages, cards."""
    env = run["env"]
    label = (run.get("arm_label") or " + ".join(sorted({c.get("device") for c in run["cells"]
                                                        if c.get("device")})) or "no cells")
    A(f"{level} Arm `{run['run_id']}` — {label}")
    A("")
    A(f"- date (UTC): {env['created_utc']}")
    A(f"- box: `{env['hostname']}` (container `{env.get('container_hint') or 'unknown'}`), "
      f"{env['cpu_count']} cores, 1-min loadavg at start {env['loadavg_at_start'][0]:.2f}")
    dirty = {True: " (working tree DIRTY)", False: "",
             None: " (cleanliness unknown: no `git` in this container)"}[env.get("git_dirty")]
    A(f"- git: `{env['git_sha']}`{dirty}")
    A(f"- python {env['python']} — `{env['executable']}`")
    A(f"- preset `{run['preset']}`, repeats {run['repeats']}, "
      f"two-stage {'on' if run.get('two_stage') else 'off'}")
    A(f"- cards: {run.get('gpu_ids') or 'none (no GPU arm in this invocation)'}")
    if run.get("gpu_preflight_reason"):
        A(f"- **GPU pre-flight**: {run['gpu_preflight_reason']}")
    A("")
    A("| package | version |")
    A("|---|---|")
    for k, v in sorted(env["packages"].items()):
        A(f"| {k} | {v if v else '*not installed*'} |")
    A("")
    if env.get("gpus"):
        A("`nvidia-smi` at run start (one sample):")
        A("")
        A("| gpu | mem used MiB | mem total MiB | util % | temp C | sm clock MHz |")
        A("|---|---|---|---|---|---|")
        for g in env["gpus"]:
            A(f"| {g['index']} | {g['mem_used_mib']} | {g['mem_total_mib']} | {g['util_pct']} | "
              f"{g['temp_c']} | {g['sm_clock_mhz']} |")
    else:
        A("`nvidia-smi` is not usable from this container (NVML failed), so this invocation has "
          "no GPU arm and no GPU trace.")
    A("")
    loads = [c["contention"]["loadavg1_mean"] for c in run["cells"]
             if isinstance(c.get("contention"), dict) and "loadavg1_mean" in c["contention"]]
    if loads:
        A(f"1-minute load average sampled every {run.get('monitor_interval_s', 1.0)} s while this "
          f"arm's cells ran: {min(loads):.1f} to {max(loads):.1f} on {env['cpu_count']} cores.")
        A("")


def merge_runs(runs, run_id):
    """One report from several invocations — the design runs one invocation per container."""
    cells = []
    for r in runs:
        for c in r["cells"]:
            cells.append({**c, "source_run": r["run_id"]})
    # Keep every refusal, including "--cells did not select it" -- a reader must be able to see
    # that a cell was NOT MEASURED rather than infer it from an absent row. Drop only refusals for
    # cells that some other arm did run, and de-duplicate identical (cell, reason) pairs.
    ran = {c["cell_id"] for c in cells}
    seen, refused = set(), []
    for r in runs:
        for x in r.get("refused", []):
            key = (x["cell"], x["reason"])
            if key not in seen and x["cell"] not in ran:
                seen.add(key)
                refused.append(x)
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "arms": runs,
            "env": runs[0]["env"], "preset": runs[0]["preset"], "repeats": runs[0]["repeats"],
            "gpu_ids": [g for r in runs for g in (r.get("gpu_ids") or [])],
            "monitor_interval_s": runs[0].get("monitor_interval_s", MONITOR_INTERVAL_S),
            "cells": cells, "refused": refused,
            "crossover": next((r["crossover"] for r in runs if r.get("crossover")), None)}


def render_markdown(run):
    cells = run["cells"]
    L = []
    A = L.append
    A("# whisper_cbpf sampler benchmark")
    A("")
    A("Generated by `scripts/benchmark_samplers.py`. **Every number below was produced by that "
      "script in the run recorded alongside it**; nothing here is estimated, carried over, or "
      "rounded from a previous session. A cell that did not run says so.")
    A("")
    A(f"- report id: `{run['run_id']}`")
    A("")
    presets = sorted({a.get("preset") for a in (run.get("arms") or [run])})
    if presets == ["smoke"]:
        A("> **THIS IS A SMOKE RUN, NOT THE PRODUCTION MATRIX.** Every budget below is "
          "deliberately tiny (tens of draws, hundreds of simulations), chosen to exercise it in "
          "minutes, not to converge anything. The posteriors are **not** converged; `ESS_worst` is "
          "in the tens; and the AIC / BIC / WAIC columns are consequently meaningless as model "
          "comparisons — every WAIC here carries the `!` its own `p_waic` earns. What these rows "
          "*do* establish is that the harness runs end to end on both boxes and that its "
          "instruments report. Ranking samplers is the production run's job "
          "(`--preset production --two-stage`), and it has not been done.")
        A("")
    A("## Environment")
    A("")
    arms = run.get("arms") or [run]
    if len(arms) > 1:
        A(f"This report merges {len(arms)} invocations — one per container, which is how the "
          f"design runs the matrix: the arms differ in interpreter and installed packages, and "
          f"that difference is the experiment. Each cell's `source_run` in the sidecar says which "
          f"arm produced it.")
        A("")
    for arm in arms:
        render_arm(arm, A, level="###" if len(arms) > 1 else "###")
    A("**These boxes are shared.** A cell is marked `cpu-contended` only when "
      "`loadavg + our cores > cpu_count`, i.e. when we were actually denied cores we asked for; "
      "GPU contention is judged from `nvidia-smi` samples against our own preallocated pool. Even "
      "with headroom a shared box perturbs timings through cache and memory bandwidth, which no "
      "load-average test can see — treat single-digit-percent differences between rows as noise.")
    A("")
    ds = next((c.get("dataset") for c in cells if c.get("dataset")), None)
    if ds:
        A("## Dataset")
        A("")
        A(f"`{ds['csv']}` — {ds['n_points']} points, bands "
          + ", ".join(f"{b}:{n}" for b, n in ds["bands"].items())
          + f", `data_mode={ds['data_mode']}`, window {ds['window_days']} days since explosion, "
            f"z={ds['redshift']}, d_L={ds['dl_cm']:.3g} cm. "
            f"`resolve_space` -> `{ds['resolved_space']}`.")
        A("")
    A("## Method, and what the axis licenses")
    A("")
    A(AXIS_MD.strip())
    A("")
    A("### Instruments")
    A("")
    A("| quantity | instrument | exact or sampled |")
    A("|---|---|---|")
    A("| wall time | `time.perf_counter()` around `wp.fit` only | exact |")
    A("| cold start | `t_cold - t_warm` across repeats in one process | derived. **Upper bound "
      "on XLA compile, not a compile measurement**: it also absorbs allocator first-touch, "
      "autotune and first-call library imports inside the metrics, and is nonzero on pure-numpy "
      "cells where nothing compiles |")
    A("| model warm-up | one blocked `model.predict` at the prior median before any timer | "
      "exact |")
    A("| peak GPU memory | `jax.Device.memory_stats()['peak_bytes_in_use']`, "
      "`torch.cuda.max_memory_allocated()` | exact |")
    A(f"| GPU contention | `nvidia-smi` polled every {run.get('monitor_interval_s', 1.0)} s | "
      "**SAMPLED estimate, not an integral** |")
    A("| CPU core-seconds | `getrusage(SELF)+getrusage(CHILDREN)` delta across the fit | exact |")
    A("| ESS | `whisper_cbpf.metrics.ess_summary` (arviz bulk, worst parameter) | exact |")
    A("| accuracy | `sbi.utils.metrics.c2st` + normalised 1-Wasserstein vs a cached reference | "
      "exact when a reference exists, otherwise *not measured* |")
    A("")
    A("Struck-through `ESS_worst/s` means the row is **excluded from any speed ranking**: either "
      "its ESS is `iid_assumed` (no chain axis — ABC/ABC-SMC/SNPE), or its accuracy gate did not "
      "return PASS. `WAIC !` marks a WAIC whose own `p_waic` says it is unreliable — the same "
      "convention as `scripts/compare_at2017gfo.py`. A row with `trust = contended` shows `-` for "
      "every timing rather than a number nobody can reproduce.")
    A("")
    for g in ("A", "B", "C", "E", "ref"):
        tbl = render_group_table(cells, g)
        if not tbl:
            continue
        title, blurb = GROUP_TITLES[g]
        A(f"## {title}")
        A("")
        A(blurb)
        A("")
        A(tbl)
        A("")
    gtrace = render_gpu_trace_table(cells, run.get("monitor_interval_s", MONITOR_INTERVAL_S))
    if gtrace:
        A("## GPU trace — the sampled half")
        A("")
        A(f"`nvidia-smi` polled every {run.get('monitor_interval_s', MONITOR_INTERVAL_S)} s for "
          f"the lifetime of each GPU worker (import, warm-up, every repeat and the metrics — not "
          f"only the timed fit, so mean utilisation is diluted by setup). **These are sampled "
          f"estimates, not integrals**, and a cell shorter than a few periods gets a handful of "
          f"points; the sample count is in the table so nobody reads a mean off two of them.")
        A("")
        A("`smi peak MiB` is *not* the per-process peak — under this repo's `whisper-gpu-env.sh` "
          "XLA preallocates `XLA_PYTHON_CLIENT_MEM_FRACTION` of the card, so `nvidia-smi` reports "
          "the pool. The exact per-process peak is the `GPU peak MiB` column of the group tables, "
          "from `jax.Device.memory_stats()['peak_bytes_in_use']`. `excess MiB` is what was on the "
          f"card beyond our own pool and the pre-run baseline; over {GPU_FOREIGN_SLACK_MIB} MiB "
          f"is called contention. That threshold is measured, not assumed — see *Known "
          f"measurement gaps*.")
        A("")
        A(gtrace)
        A("")
    ev = run.get("evals_validation")
    if ev:
        A("## Forward-evaluation ledger, checked against a counting shim")
        A("")
        A("`--validate-evals` registers `bazin__counted`, whose `predict` increments a counter, "
          "and compares the observed call count with the ledger's `fit_evals + overhead_evals`. "
          "It cannot reach `n_jobs > 1` (the counter dies with the forked child) or the JAX "
          "samplers (they call `predict_jax`), so those rows stay formula-only.")
        A("")
        A(render_evals_validation(ev))
        A("")
    A("## Crossover (Group D)")
    A("")
    cx = run.get("crossover")
    if not cx:
        A("**Not measured in this run.** `docs/CHOOSING.md:19` asserts \"the GPU crosses over at "
          "roughly 50,000 simulations\" and no measurement supporting it exists anywhere in this "
          "repository. Run `--crossover` to test it; the harness fits `wall(N) = a + b*N` per arm "
          "on the warm runs, solves for `N*`, bootstraps a 95% CI over the repeats, and reports "
          "`N*_warm` (compile excluded) and `N*_cold` (compile included) against each CPU "
          "baseline separately — `n_jobs=1` and `n_jobs=8` will not give the same answer, and the "
          "doc's number is currently unqualified.")
    else:
        A(f"Model `{cx['model_key']}`, sweep {cx['sweep']}, {cx['repeats']} repeats.")
        A("")
        A(f"**Confound:** {cx['confound']}")
        A("")
        for c in cx["crossovers"]:
            A(f"- **{c['baseline']}**: {c.get('verdict_warm', c.get('verdict'))}")
            if c.get("verdict_cold"):
                A(f"  - cold (compile included): {c['verdict_cold']}")
        A("")
        A("| arm | N | warm wall s | cold wall s |")
        A("|---|---|---|---|")
        for r in cx["rows"]:
            A(f"| {r['arm']} | {r['n']:,} | {r['wall']:.2f} | {r['wall_cold']:.2f} |")
    A("")
    A("## Cells that were not run, and why")
    A("")
    A("Never a silent skip — the same discipline `scripts/compare_at2017gfo.py` applies to "
      "impossible cells.")
    A("")
    A("| cell | reason |")
    A("|---|---|")
    for e in run.get("refused", []):
        A(f"| `{e['cell']}` | {e['reason']} |")
    for what, why in STATIC_EXCLUSIONS:
        A(f"| {what} | {why} |")
    A("")
    A("## What this benchmark does not license")
    A("")
    A(NOT_LICENSED_MD.strip())
    A("")
    A("## Known measurement gaps")
    A("")
    for what, why in KNOWN_GAPS:
        A(f"- **{what}** — {why}")
    A("")
    A("## How this was produced")
    A("")
    A("Exactly these invocations, in this order, one per container:")
    A("")
    A("```")
    for arm in arms:
        A(f"# arm {arm['run_id']}  ->  {arm.get('run_dir', '(run dir not recorded)')}")
        argv = arm.get("argv")
        if not argv:
            A("# (argv not recorded for this run)")
            continue
        # The interpreter is load-bearing: it is what decides cpu-min from cpu-jax.
        A(" ".join([arm["env"]["executable"]] + list(argv)))
    A("```")
    A("")
    A(f"Sidecar: `benchmark.json` per arm (schema_version {SCHEMA_VERSION}), per-cell records in "
      f"`<run_dir>/cells/`, traces in `gpu_trace.csv` / `cpu_trace.csv`. Re-render a single arm "
      f"with `--report-only <run_dir>`; re-render this merged report with "
      f"`--merge <dir1>,<dir2>,... --report-out <path>`. The driver's own consistency checks run "
      f"with `--self-test` (no fits, no GPU); the forward-evaluation formulas are checked against "
      f"a counting shim with `--validate-evals`.")
    return "\n".join(L) + "\n"


def load_run(run_dir):
    run_dir = Path(run_dir)
    p = run_dir / "benchmark.json"
    if p.exists():
        return json.loads(p.read_text())
    cells = [json.loads(f.read_text()) for f in sorted((run_dir / "cells").glob("*.json"))]
    return {"schema_version": SCHEMA_VERSION, "run_id": run_dir.name,
            "env": json.loads((run_dir / "env.json").read_text()), "cells": cells,
            "refused": [], "preset": "unknown", "repeats": None}


# =============================================================================================
# Self-test (would have been tests/test_benchmark_driver.py; this session owns only two files)
# =============================================================================================

def self_test():
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    budgets = _budget(quick=True)
    cells, refused = build_matrix(budgets, {"A", "B", "C", "E"}, set(DEVICES))
    check("build_matrix yields cells", len(cells) > 0, f"{len(cells)} cells")
    bad = [c for c in cells
           if impossible_reason(c.model_kind, c.sampler, c.space, c.has_jax) is not None]
    check("no cell survives impossible_reason", not bad, str([c.cell_id for c in bad]))
    check("every cell_id is unique", len({c.cell_id for c in cells}) == len(cells))
    check("group A is cpu-min only", all(c.device == "cpu-min"
                                         for c in cells if c.group == "A"))
    check("exactly one flux control cell", sum(c.space == "flux" for c in cells) == 1)
    check("every refusal carries a reason", all(r.get("reason") for r in refused),
          f"{len(refused)} refused")
    illegal = Cell(cell_id="t", group="A", model_key="kn2_redback", sampler="nuts_gpu",
                   device="gpu")
    forky = Cell(cell_id="t", group="B", model_key="kn1", sampler="mcmc", device="cpu-jax",
                 n_jobs=8)
    check("refusal_reason delegates to impossible_reason",
          "impossible_reason()" in (refusal_reason(illegal) or ""), refusal_reason(illegal))
    check("refusal_reason blocks fork-with-JAX unless opted in",
          refusal_reason(forky) is not None and refusal_reason(forky, True) is None)

    c = next(x for x in cells if x.device == "cpu-jax")
    e = worker_env(c, [])
    check("cpu-jax worker env pins JAX to CPU",
          e["JAX_PLATFORMS"] == "cpu" and e["CUDA_VISIBLE_DEVICES"] == "")
    check("cpu-jax worker env pins BLAS threads", e["OMP_NUM_THREADS"] == "1")
    g = Cell(cell_id="x", group="B", model_key="kn1", sampler="nuts_gpu", device="gpu")
    eg = worker_env(g, [5])
    check("gpu worker env pins the card and drops JAX_PLATFORMS",
          eg["CUDA_VISIBLE_DEVICES"] == "5" and "JAX_PLATFORMS" not in eg
          and eg["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID")
    m = Cell(cell_id="x", group="A", model_key="bazin", sampler="abc", device="cpu-min")
    check("cpu-min worker env blanks CUDA_VISIBLE_DEVICES",
          worker_env(m, [])["CUDA_VISIBLE_DEVICES"] == "")

    ok = True
    for spec, ev in (
        ({"device": "gpu", "sampler": "nuts_gpu"},
         {"jax_importable": True, "check_gpu_ok": False, "check_gpu_msg": "cpu",
          "jax_default_backend": "cpu"}),
        ({"device": "cpu-jax", "sampler": "mcmc"},
         {"jax_importable": True, "check_gpu_ok": True, "jax_default_backend": "gpu"}),
        ({"device": "cpu-min", "sampler": "abc"}, {"jax_importable": True}),
        ({"device": "gpu", "sampler": "snpe"},
         {"jax_importable": True, "check_gpu_ok": True, "jax_default_backend": "gpu",
          "torch_cuda": False}),
    ):
        try:
            _assert_device(spec, ev)
            ok = False
        except DeviceAssertError:
            pass
    check("_assert_device rejects every silent-fallback shape", ok)
    try:
        _assert_device({"device": "gpu", "sampler": "nuts_gpu"},
                       {"jax_importable": True, "check_gpu_ok": True,
                        "jax_default_backend": "gpu", "torch_cuda": True})
        good = True
    except DeviceAssertError:
        good = False
    check("_assert_device accepts a genuine GPU cell", good)

    thr = [_throttle_active(x) for x in ("0x0000000000000000", "0x0000000000000004",
                                         "Not Active", "[System is not in ready state]",
                                         "0x0000000000000001")]
    check("_throttle_active reads the hex bitmask and the text form",
          thr[0] == (False, True) and thr[1] == (True, True) and thr[2] == (False, True)
          and thr[3] == (False, False))
    check("_throttle_active does not call an IDLE card (0x1 GpuIdle) throttled",
          thr[4] == (False, True))
    check("_throttle_names decodes the reason", _throttle_names("0x24") == ["SwPowerCap",
                                                                           "SwThermalSlowdown"])

    fc = fit_linear([1, 2, 3], [10.0, 20.0, 30.0])
    fg = fit_linear([1, 2, 3], [15.0, 20.0, 25.0])
    check("crossover_point solves the intersection",
          abs(crossover_point(fc, fg) - 2.0) < 1e-9, f"got {crossover_point(fc, fg)}")

    fake = {"schema_version": SCHEMA_VERSION, "run_id": "t", "preset": "smoke", "repeats": 2,
            "gpu_ids": [], "monitor_interval_s": 1.0, "refused": [{"cell": "x", "reason": "y"}],
            "env": {"created_utc": "now", "hostname": "h", "container_hint": "c", "cpu_count": 8,
                    "loadavg_at_start": [1.0, 1, 1], "git_sha": "abc", "git_dirty": False,
                    "python": "3.11", "executable": "/p", "packages": {"numpy": "1.0"},
                    "gpus": []},
            "cells": [{"cell_id": "A/bazin/abc/cpu-min", "group": "A", "model_key": "bazin",
                       "sampler": "abc", "device": "cpu-min", "status": "works",
                       "budget": {"n_simulations": 10}, "trust_timing": False,
                       "contention": {"contended": True},
                       "timing": {"t_warm_s": 1.0, "t_cold_s": 2.0, "cold_start_s": 1.0,
                                  "model_warmup_s": 0.01},
                       "evals": {"fit_evals": 10, "overhead_evals": 11},
                       "posterior": {"ess_worst": 5.0, "ess_kind": "iid_assumed",
                                     "ess_worst_per_sec_warm": 5.0},
                       "accuracy": {"verdict": "not measured"},
                       "model_selection": {"aic": 1.0, "bic": 2.0, "waic": None},
                       "resources": {"cpu_core_seconds": 1.0}}]}
    fake["cells"].append({**fake["cells"][0], "cell_id": "A/bazin/mcmc/cpu-min",
                          "sampler": "mcmc", "status": "broken", "error": "boom"})
    merged = merge_runs([fake, {**fake, "run_id": "t2",
                                "cells": [{**fake["cells"][0], "device": "gpu"}]}], "t+t2")
    check("merge_runs concatenates cells and stamps their source run",
          len(merged["cells"]) == 3 and all("source_run" in c for c in merged["cells"]))
    mmd = render_markdown(merged)
    check("a merged report renders one Arm block per invocation", mmd.count("### Arm `") == 2)

    md = render_markdown(fake)
    widths = {ln.count("|") for ln in render_group_table(fake["cells"], "A").splitlines()}
    check("every group-table row has the header's column count", len(widths) == 1, str(widths))
    check("render_markdown produces a report", "whisper_cbpf sampler benchmark" in md)
    check("a broken cell is rendered, not dropped", "**broken**" in md)
    check("a contended row renders '-' for timings, never a number", "| 1.00 |" not in md)
    check("render_markdown is deterministic", render_markdown(fake) == md)
    check("the crossover section says 'not measured' when absent", "Not measured" in md)

    print(f"\nself-test: {len(fails)} failure(s)" + (f": {fails}" if fails else ""))
    return 1 if fails else 0


# =============================================================================================
# main
# =============================================================================================

SMOKE_BUDGET = {
    "abc": dict(n_simulations=120, quantile=0.2, n_jobs=8),
    "abc_smc": dict(n_particles=24, n_rounds=2, n_jobs=8),
    "mcmc": dict(nsteps=60, burnin=20, nwalkers=24, thin=1),
    "nested": dict(nlive=60, dlogz=1.0),
    "snpe": dict(num_rounds=1, num_simulations=120, num_samples=200),
    "abc_gpu": dict(n_simulations=256, quantile=0.1),
    "abc_smc_gpu": dict(n_particles=24, n_rounds=2),
    "nuts_gpu": dict(num_warmup=40, num_samples=40, num_chains=2),
    "emcee_jax": dict(nwalkers=24, nsteps=80, burnin=20, thin=1),
    "pymc_jax_gpu_vectorized": dict(num_warmup=40, num_samples=40, num_chains=2),
    "pymc_jax_gpu_parallelized": dict(num_warmup=40, num_samples=40, num_chains=2),
}


def budgets_for(preset):
    if preset == "smoke":
        return {k: dict(v) for k, v in SMOKE_BUDGET.items()}
    return _budget(quick=(preset == "pilot"))


def resolve_device_set(device_arg, explicit):
    if explicit:
        return set(explicit)
    if device_arg == "gpu":
        return {"gpu"}
    import importlib.util
    return {"cpu-jax" if importlib.util.find_spec("jax") is not None else "cpu-min"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="cpu resolves to cpu-min or cpu-jax by whether JAX is importable here")
    ap.add_argument("--device-set", action="append", choices=list(DEVICES), default=None,
                    help="force the device arm(s); repeatable")
    ap.add_argument("--preset", choices=["smoke", "pilot", "production"], default="pilot")
    ap.add_argument("--quick", action="store_true",
                    help="alias for --preset pilot (the same budgets compare_at2017gfo --quick "
                         "uses); posteriors are NOT converged")
    ap.add_argument("--groups", default="A,B,C,E")
    ap.add_argument("--cells", default=None,
                    help="comma-separated fnmatch patterns on cell_id, "
                         "e.g. 'A/bazin/*,B/kn1/mcmc/*'")
    ap.add_argument("--repeats", type=int, default=3,
                    help="same-seed timing replicates in ONE process; run1 is cold, "
                         "median(run2..) is warm, and the difference is the compile estimate")
    ap.add_argument("--two-stage", action="store_true",
                    help="pilot every cell, then rescale its budget knob toward "
                         f"ESS_worst >= {QUALITY_TARGET_ESS} and re-run")
    ap.add_argument("--gpu-ids", default="auto",
                    help="'auto' (via the nvidia-smi idle test) or explicit, e.g. 5 or 4,5")
    ap.add_argument("--allow-contended", action="store_true",
                    help="run the GPU arm on a card that failed the idle test. Every such cell is "
                         "recorded with trust_timing=false and its timings render as '-'.")
    ap.add_argument("--allow-fork-with-jax", action="store_true",
                    help="permit n_jobs>1 on the cpu-jax/gpu arms (forking with JAX threads live "
                         "risks deadlock; see samplers/jax/emcee_jax.py)")
    ap.add_argument("--max-cell-seconds", type=int, default=MAX_CELL_SECONDS)
    ap.add_argument("--retries", type=int, default=0,
                    help="re-run a cell that came back contended (its timings are a measurement "
                         "of the neighbours, not of the sampler). Default 0: report it as '-'.")
    ap.add_argument("--monitor-interval", type=float, default=MONITOR_INTERVAL_S)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--publish", action="store_true", help="copy the report to docs/BENCHMARKS.md")
    ap.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    ap.add_argument("--build-references", action="store_true")
    ap.add_argument("--validate-evals", action="store_true",
                    help="check the forward-evaluation formulas against a counting shim, then exit")
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--crossover-sweep", default=",".join(str(n) for n in N_SWEEP))
    ap.add_argument("--report-only", default=None, help="re-render a finished run directory")
    ap.add_argument("--merge", default=None,
                    help="comma-separated run directories to render as ONE report (the design "
                         "runs one invocation per container, so a full matrix is several runs)")
    ap.add_argument("--report-out", default=None, help="where --merge writes its markdown")
    ap.add_argument("--evals-json", default=None,
                    help="a validate_evals.json to fold into the merged report")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args._worker:
        return worker_main(Path(args._worker))
    if args.self_test:
        return self_test()
    if args.report_only:
        run = load_run(args.report_only)
        out = Path(args.report_only) / "report.md"
        out.write_text(render_markdown(run))
        print(f"report written to {out}")
        return 0
    if args.merge:
        dirs = [Path(d.strip()) for d in args.merge.split(",") if d.strip()]
        runs = [load_run(d) for d in dirs]
        merged = merge_runs(runs, args.run_id or "+".join(r["run_id"] for r in runs))
        if args.evals_json:
            merged["evals_validation"] = json.loads(Path(args.evals_json).read_text())
        out = Path(args.report_out) if args.report_out else (dirs[0].parent / "merged_report.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(merged))
        (out.with_suffix(".json")).write_text(json.dumps(merged, indent=2, default=str))
        print(f"merged report : {out}\nmerged sidecar: {out.with_suffix('.json')}")
        if args.publish:
            dst = REPO / "docs" / "BENCHMARKS.md"
            shutil.copyfile(out, dst)
            print(f"published: {dst}  (the human commits)")
        return 0

    if args.quick:
        args.preset = "pilot"
    env = env_snapshot()
    run_id = args.run_id or (time.strftime("%Y%m%d-%H%M%S") + f"-{env['git_sha']}-"
                             + env["hostname"].split(".")[0])
    out_dir = Path(args.out) if args.out else (OUT_ROOT / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, default=str))

    device_set = resolve_device_set(args.device, args.device_set)
    gpu_ids, preflight_reason = [], ""
    if "gpu" in device_set:
        requested = (None if args.gpu_ids == "auto"
                     else [int(x) for x in args.gpu_ids.split(",") if x.strip()])
        gpu_ids, snap, preflight_reason = preflight_gpu(n=(len(requested) if requested else 1),
                                                        requested=requested)
        if preflight_reason:
            print(f"\nGPU PRE-FLIGHT: {preflight_reason}")
        if gpu_ids and preflight_reason and not args.allow_contended:
            print("refusing to run the GPU arm on a contended card. Pass --allow-contended to "
                  "record it anyway (timings will be marked untrusted and rendered as '-'), or "
                  "wait for an idle card.")
            gpu_ids = []
        if not gpu_ids:
            device_set = device_set - {"gpu"}
            if not device_set:
                run = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "env": env,
                       "preset": args.preset, "repeats": args.repeats, "cells": [],
                       "refused": [{"cell": "ALL gpu cells", "reason": preflight_reason}],
                       "gpu_ids": [], "gpu_preflight_reason": preflight_reason,
                       "monitor_interval_s": args.monitor_interval, "two_stage": args.two_stage}
                (out_dir / "benchmark.json").write_text(json.dumps(run, indent=2, default=str))
                (out_dir / "report.md").write_text(render_markdown(run))
                print(f"\nno GPU arm ran. Record written to {out_dir}")
                return 2

    budgets = budgets_for(args.preset)
    if args.validate_evals:
        rows = validate_evals(out_dir)
        return 0 if all(r.get("matches") for r in rows
                        if r["status"] == "ok" and r.get("declared_exact")) else 1

    if args.build_references:
        keys = sorted({c.model_key for c in
                       build_matrix(budgets, set(args.groups.split(",")), device_set)[0]})
        return 0 if build_references(out_dir, args.device, gpu_ids, keys,
                                     args.max_cell_seconds, args.monitor_interval) else 1

    if args.crossover:
        sweep = [int(x) for x in args.crossover_sweep.split(",") if x.strip()]
        cx = run_crossover(out_dir, gpu_ids, sweep, args.repeats, args.max_cell_seconds,
                           args.monitor_interval)
        print(json.dumps(cx["crossovers"], indent=2, default=str))
        return 0

    cells, refused = build_matrix(budgets, set(args.groups.split(",")), device_set,
                                  repeats=args.repeats,
                                  allow_fork_with_jax=args.allow_fork_with_jax)
    if args.cells:
        pats = [p.strip() for p in args.cells.split(",") if p.strip()]
        keep = [c for c in cells if any(fnmatch.fnmatch(c.cell_id, p) or p in c.cell_id
                                        for p in pats)]
        refused += [{"cell": c.cell_id, "reason": f"not selected by --cells {args.cells!r}"}
                    for c in cells if c not in keep]
        cells = keep

    print(f"\nrun_id  : {run_id}")
    print(f"out     : {out_dir}")
    print(f"devices : {sorted(device_set)}   gpu_ids={gpu_ids or 'none'}")
    print(f"preset  : {args.preset}   repeats={args.repeats}   "
          f"two-stage={'on' if args.two_stage else 'off'}")
    print(f"matrix  : {len(cells)} cell(s), {len(refused)} not run")
    for c in cells:
        print(f"  {c.cell_id:<44} {_budget_str(c.budget)}")
    if args.dry_run:
        print("\n--dry-run: nothing executed. Projected wall time is UNKNOWN until a pilot has "
              "run; re-run --dry-run after a --preset pilot pass and this line will carry "
              "sum(t_warm x budget_scale x repeats) instead.")
        return 0

    baseline_mib = None
    if gpu_ids:
        baseline_mib = max((g["mem_used_mib"] for g in smi_gpus() if g["index"] in gpu_ids),
                           default=0)
        print(f"\nbaseline nvidia-smi memory on card(s) {gpu_ids}: {baseline_mib} MiB "
              f"(subtracted before judging mid-run contention)")

    records = []
    for i, cell in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell.cell_id}  {_budget_str(cell.budget)}", flush=True)
        rec = run_cell_subprocess(cell, out_dir, gpu_ids, args.max_cell_seconds,
                                  args.monitor_interval, baseline_mib)
        # A contended cell is a measurement of the neighbours, not of the sampler. Retry it;
        # if it stays contended it keeps trust_timing=false and renders as '-'.
        for attempt in range(1, int(args.retries) + 1):
            if rec.get("status") != "works" or rec.get("trust_timing"):
                break
            print(f"    contended -> retry {attempt}/{args.retries}", flush=True)
            rec = run_cell_subprocess(cell, out_dir, gpu_ids, args.max_cell_seconds,
                                      args.monitor_interval, baseline_mib)
            rec["retries_used"] = attempt
        if args.two_stage and rec.get("status") == "works":
            scaled, factor, why = scale_budget(cell, rec)
            if abs(factor - 1.0) > 1e-9:
                print(f"    stage 2: {why}", flush=True)
                rec2 = run_cell_subprocess(scaled, out_dir, gpu_ids, args.max_cell_seconds,
                                           args.monitor_interval, baseline_mib)
                rec2["budget_scale_from_pilot"] = factor
                rec2["pilot"] = {"budget": cell.budget,
                                 "ess_worst": (rec.get("posterior") or {}).get("ess_worst"),
                                 "t_warm_s": (rec.get("timing") or {}).get("t_warm_s")}
                rec = rec2
        records.append(rec)

    run = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "created_utc": env["created_utc"],
           "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "preset": args.preset,
           "devices": sorted(device_set), "arm_label": " + ".join(sorted(device_set)),
           "argv": list(sys.argv), "run_dir": str(out_dir),
           "repeats": args.repeats, "two_stage": args.two_stage, "env": env,
           "gpu_ids": gpu_ids, "gpu_preflight_reason": preflight_reason,
           "monitor_interval_s": args.monitor_interval,
           "axis": {"primary": "warm wall seconds; ESS_worst and ESS_worst/s (warm) alongside",
                    "quality_target_ess": QUALITY_TARGET_ESS, "c2st_pass": C2ST_PASS},
           "cells": records, "refused": refused}
    (out_dir / "benchmark.json").write_text(json.dumps(run, indent=2, default=str))
    report = out_dir / "report.md"
    report.write_text(render_markdown(run))

    n_ok = sum(r.get("status") == "works" for r in records)
    n_trust = sum(bool(r.get("trust_timing")) for r in records)
    print(f"\n{n_ok}/{len(records)} cells produced a result; {n_trust} have trustworthy timings.")
    for r in records:
        if r.get("status") != "works":
            print(f"  {r['cell_id']}: {r.get('status')} -- {str(r.get('error', ''))[:160]}")
    print(f"\nreport : {report}")
    print(f"sidecar: {out_dir / 'benchmark.json'}")
    if args.publish:
        dst = REPO / "docs" / "BENCHMARKS.md"
        shutil.copyfile(report, dst)
        print(f"published: {dst}  (the human commits)")
    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
