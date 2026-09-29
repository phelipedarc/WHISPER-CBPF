"""Run many fits at once: one process per job, spread over the GPUs and CPU cores you allow.

:func:`run_jobs` takes a list of fits -- each a light curve, a model and a sampler, a :class:`Job` --
and runs every one in its own fresh Python process:

* **GPU jobs** (the samplers that run on JAX: ``nuts_gpu``, ``emcee_jax``, ``abc_gpu``, ...) get one
  card each, one job per card at a time. Only cards this process may see (``CUDA_VISIBLE_DEVICES``)
  and that are idle are used, and every card is checked again before each job starts on it, so no job
  is started on a card someone else is using.
* **CPU jobs** (``abc``, ``mcmc``, ``nested``, ...) run beside them on their own cores.
* Every job process is pinned to its own share of the CPU cores before Python imports anything, and
  starts with ``NCCL_P2P_DISABLE=1``, ``JAX_PLATFORMS`` naming its device and ``CUDA_VISIBLE_DEVICES``
  naming its one card (or none).

Why a fresh process per job: a process that has started JAX cannot safely ``fork`` (the fork keeps
XLA's thread locks held and the child waits forever), device visibility is fixed when JAX starts, and
one job's crash or out-of-memory kill must not take the batch down. A failed job is retried; with
``cache_dir`` finished fits are kept, so an interrupted batch resumes where it stopped
(:func:`whisper_cbpf.fit_cached`).

Nothing here imports JAX, so the parent never holds a GPU.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from .backends._env import CPU_FRACTION

#: A GPU is idle below both thresholds (the rule :func:`whisper_cbpf.gpu_list` uses). Memory alone
#: misses a compute-bound job that allocates little; utilisation alone misses an idle process that
#: holds most of the card.
IDLE_MEMORY_MIB = 512
IDLE_UTILIZATION_PCT = 10

#: Before each job starts on a card the card is checked again. One still busy after this many seconds
#: (someone else started using it) takes no more jobs from this run.
IDLE_WAIT_S = 60.0
_POLL_S = 2.0

#: A job process that outlives ``timeout=`` is asked to stop (SIGTERM) and killed if it is still
#: running this many seconds later.
STOP_GRACE_S = 10.0

#: Set in every job process, so that a script which calls :func:`run_jobs` without an
#: ``if __name__ == "__main__":`` guard fails with a message instead of starting jobs recursively.
_CHILD_FLAG = "WHISPER_RUN_JOBS_CHILD"

#: The directory that holds the ``whisper_cbpf`` package, put first on a job process's path.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The job process's first lines: pin the cores before any import can start a thread pool (numpy's
#: BLAS, XLA's), then hand over to :func:`_job_main`.
_BOOT = (
    "import os, sys\n"
    "cores = os.environ.get('WHISPER_JOB_CORES')\n"
    "if cores and hasattr(os, 'sched_setaffinity'):\n"
    "    os.sched_setaffinity(0, [int(c) for c in cores.split(',')])\n"
    "sys.path.insert(0, sys.argv[2])\n"
    "from whisper_cbpf.parallel import _job_main\n"
    "sys.exit(_job_main(sys.argv[1]))\n"
)


@dataclass(frozen=True)
class Job:
    """One fit for :func:`run_jobs`: a light curve, a model and a sampler.

    Parameters
    ----------
    lc : LightCurve
        The data.
    model : str or Model
        A registered model name or a :class:`~whisper_cbpf.models.Model`. A name is resolved in the
        calling process, so a model you registered there works too.
    sampler : str
        A registered sampler name (:func:`whisper_cbpf.list_samplers`).
    kwargs : dict, optional
        Keyword arguments for :func:`whisper_cbpf.fit` (``prior=``, ``seed=``, ``nsteps=``, ...).
    name : str, optional
        A unique label for the job in the log and the summary. Default
        ``"<lc.name>__<model>__<sampler>"``.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=np.linspace(0, 30, 20), band=["r"] * 20, flux=np.ones(20),
    ...                    flux_err=np.full(20, 0.1), name="sn1")
    >>> job = wp.Job(lc, "bazin", "abc", kwargs={"n_simulations": 2000})
    >>> job.sampler
    'abc'
    """

    lc: object
    model: object
    sampler: str
    kwargs: dict = field(default_factory=dict)
    name: str = None


@dataclass(frozen=True)
class _Card:
    index: int
    uuid: str
    memory_mib: float
    utilization_pct: float

    @property
    def idle(self):
        return self.memory_mib < IDLE_MEMORY_MIB and self.utilization_pct < IDLE_UTILIZATION_PCT

    def describe(self):
        return f"GPU {self.index} ({self.memory_mib:.0f} MiB used, {self.utilization_pct:.0f}% busy)"


def _gpu_table():
    """Every GPU ``nvidia-smi`` lists, as :class:`_Card`; ``None`` when ``nvidia-smi`` cannot run.

    A value ``nvidia-smi`` reports as ``[N/A]`` reads as busy (infinite), so a card whose load cannot
    be read is never used.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None

    def number(s):
        try:
            return float(s)
        except ValueError:
            return float("inf")

    cards = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit():
            cards.append(_Card(int(parts[0]), parts[1], number(parts[2]), number(parts[3])))
    return cards


def _visible(cards):
    """The cards ``CUDA_VISIBLE_DEVICES`` lets this process use (all of them when it is unset).

    Tokens are ``nvidia-smi`` indices (``CUDA_DEVICE_ORDER=PCI_BUS_ID``, as the GPU environment script
    sets) or GPU UUIDs. As in CUDA, the list ends at the first token that names no card, so ``""``
    and ``"-1"`` hide every card.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return list(cards)
    by_index = {str(c.index): c for c in cards}
    out = []
    for tok in (t.strip() for t in cvd.split(",")):
        card = by_index.get(tok) or next((c for c in cards if tok.startswith("GPU-") and c.uuid.startswith(tok)),
                                         None)
        if card is None:
            break
        if card not in out:
            out.append(card)
    return out


def _choose_gpus(gpus):
    """``(idle cards to use, notes for the log)`` for ``gpus`` = ``"auto"``, a list of indices or None."""
    if isinstance(gpus, int) and not isinstance(gpus, bool):
        gpus = [gpus]
    if gpus is None or (not isinstance(gpus, str) and len(gpus) == 0):
        return [], []
    if isinstance(gpus, str) and gpus != "auto":
        gpus = [t for t in gpus.split(",") if t.strip()]
    explicit = not isinstance(gpus, str)
    wanted = []
    if explicit:
        for g in gpus:
            try:
                g = int(g)
            except (TypeError, ValueError):
                raise ValueError(f"gpus={gpus!r}: {g!r} is not a GPU index. Pass 'auto', a list of "
                                 f"nvidia-smi indices such as [0, 3], or None for the CPU.") from None
            if g not in wanted:
                wanted.append(g)
    table = _gpu_table()
    if table is None:
        if explicit:
            raise RuntimeError(
                f"nvidia-smi could not be run, so run_jobs cannot check that GPU(s) {wanted} are idle, "
                f"and it never starts a job on a card it cannot check. Make nvidia-smi available, or "
                f"pass gpus=None to run every job on the CPU.")
        return [], ["nvidia-smi could not be run: no GPU is used"]
    visible = _visible(table)
    if explicit:
        known = {c.index: c for c in table}
        missing = [g for g in wanted if g not in known]
        if missing:
            raise ValueError(f"GPU(s) {missing} do not exist; nvidia-smi lists {sorted(known)}.")
        hidden = [g for g in wanted if known[g] not in visible]
        if hidden:
            raise ValueError(
                f"GPU(s) {hidden} are outside CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}, so this process may not use them. Pick "
                f"from {[c.index for c in visible]}, or change CUDA_VISIBLE_DEVICES before starting Python.")
        candidates = [known[g] for g in wanted]
    else:
        candidates = visible
    idle = [c for c in candidates if c.idle]
    busy = [c for c in candidates if not c.idle]
    if explicit and not idle:
        raise RuntimeError(
            f"none of GPU(s) {wanted} is idle: {'; '.join(c.describe() for c in busy)}. run_jobs never "
            f"shares a busy card. Wait for one to free up, name other cards, or pass gpus=None to run on "
            f"the CPU.")
    notes = [f"skipped busy {c.describe()}" for c in busy]
    return idle, notes


def _card_is_idle(index):
    """Whether card ``index`` is idle right now; ``False`` when it cannot be checked."""
    table = _gpu_table()
    card = next((c for c in table or [] if c.index == index), None)
    return card is not None and card.idle


def _wait_until_idle(index, stop):
    """Poll card ``index`` until it is idle (True) or :data:`IDLE_WAIT_S` pass (False)."""
    deadline = time.monotonic() + IDLE_WAIT_S
    while not stop.is_set():
        if _card_is_idle(index):
            return True
        if time.monotonic() >= deadline:
            return False
        stop.wait(_POLL_S)
    return False


def _parse_cpu_list(spec):
    """``"0-3,8"`` -> ``[0, 1, 2, 3, 8]`` (the kernel's cpu-list syntax, as ``taskset -c`` takes)."""
    cores = set()
    for part in (p.strip() for p in str(spec).split(",")):
        lo, sep, hi = part.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()) or (sep and int(hi) < int(lo)):
            raise ValueError(f"cpu_cores={spec!r}: {part!r} is not N or N-M (for example '0-7,16').")
        cores.update(range(int(lo), int(hi if sep else lo) + 1))
    return sorted(cores)


def _core_budget(cpu_cores):
    """The CPU cores this run may use, sorted: see ``cpu_cores`` in :func:`run_jobs`."""
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
    else:
        allowed = list(range(os.cpu_count() or 1))
    if cpu_cores is None:
        return allowed[:max(1, int(round(CPU_FRACTION * len(allowed))))]
    if isinstance(cpu_cores, int) and not isinstance(cpu_cores, bool):
        if not 1 <= cpu_cores <= len(allowed):
            raise ValueError(f"cpu_cores={cpu_cores} but this process may use {len(allowed)} cores; pass "
                             f"a number from 1 to {len(allowed)}.")
        return allowed[:cpu_cores]
    cores = _parse_cpu_list(cpu_cores) if isinstance(cpu_cores, str) else sorted({int(c) for c in cpu_cores})
    if not cores:
        raise ValueError("cpu_cores is empty; pass None for the default share, a number, or core ids.")
    outside = [c for c in cores if c not in allowed]
    if outside:
        raise ValueError(f"cores {outside} are outside this process's CPU affinity ({_cores_text(allowed)}); "
                         f"run_jobs can only hand out cores it may use itself.")
    return cores


def _cores_text(cores):
    """``[0, 1, 2, 5]`` -> ``"0-2,5"``."""
    out, start = [], None
    for i, c in enumerate(cores):
        if start is None:
            start = c
        if i + 1 == len(cores) or cores[i + 1] != c + 1:
            out.append(str(start) if start == c else f"{start}-{c}")
            start = None
    return ",".join(out)


def _split(cores, n):
    """``cores`` in ``n`` contiguous blocks whose sizes differ by at most one."""
    size, extra = divmod(len(cores), n)
    out, i = [], 0
    for k in range(n):
        j = i + size + (1 if k < extra else 0)
        out.append(cores[i:j])
        i = j
    return out


def _needs_gpu(sampler):
    """Whether ``sampler`` runs on JAX (a GPU job) rather than on the CPU."""
    from .backends._registration import GPU_SAMPLERS
    return sampler in GPU_SAMPLERS


def _as_job(job, i):
    if isinstance(job, Job):
        return job
    if isinstance(job, Mapping):
        return Job(**job)
    if isinstance(job, (tuple, list)) and len(job) in (3, 4):
        return Job(*job)
    raise TypeError(f"job {i} is a {type(job).__name__}; pass a whisper_cbpf.Job, a tuple "
                    f"(lc, model, sampler[, kwargs]) or a dict with those keys.")


def _preparation_data():
    """What a job process needs to find the objects this one pickles: its path, cwd and ``__main__``.

    The same data ``multiprocessing``'s ``spawn`` hands its workers (applied there by
    ``multiprocessing.spawn.prepare``), built here without touching the global start method.
    """
    main = sys.modules.get("__main__")
    data = dict(sys_path=[os.getcwd() if p == "" else p for p in sys.path], sys_argv=list(sys.argv),
                dir=os.getcwd())
    spec_name = getattr(getattr(main, "__spec__", None), "name", None)
    main_path = getattr(main, "__file__", None)
    if spec_name is not None:
        data["init_main_from_name"] = spec_name
    elif main_path is not None and os.path.isfile(main_path):     # not "<stdin>" nor a notebook
        data["init_main_from_path"] = os.path.abspath(main_path)
    return data


def _write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def _job_main(job_dir):
    """Entry point of a job process: rebuild the job, fit, write ``result.npz`` and ``status.json``."""
    status = dict(token=os.environ.get("WHISPER_JOB_TOKEN", ""), ok=False, stage="rebuild", error=None,
                  fit_s=None, resumed=False)
    print(f"whisper_cbpf job process {os.getpid()}: CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}, JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r}, "
          f"cores {os.environ.get('WHISPER_JOB_CORES')}", flush=True)
    try:
        import multiprocessing.spawn as mp_spawn

        with open(os.path.join(job_dir, "job.pkl"), "rb") as fh:
            mp_spawn.prepare(pickle.load(fh))            # sys.path, cwd and the caller's __main__
            job = pickle.load(fh)
    except BaseException as exc:                          # noqa: BLE001 - reported, then the process ends
        traceback.print_exc()
        status["error"] = (
            f"the job could not be rebuilt in its own process ({type(exc).__name__}: {exc}). Everything a "
            f"job carries must be importable by name in a new Python process: define model functions and "
            f"samplers in a module (not in a notebook cell, nor as a lambda or a closure), and keep the "
            f"run_jobs(...) call of a script under `if __name__ == '__main__':`.")
        _write_json(os.path.join(job_dir, "status.json"), status)
        return 3
    status["stage"] = "fit"
    try:
        from .samplers import fit, register_sampler

        register_sampler(job["sampler"], job["sampler_factory"], overwrite=True)
        t0 = time.perf_counter()
        if job["cache_dir"] is not None:
            from . import fit_cached
            result = fit_cached(job["lc"], job["model"], job["sampler"], job["cache_dir"], **job["kwargs"])
        else:
            result = fit(job["lc"], job["model"], job["sampler"], **job["kwargs"])
        status["fit_s"] = time.perf_counter() - t0
        status["resumed"] = getattr(result, "loaded_from", None) is not None    # fit_cached's saved fit
        status["stage"] = "save"
        # SamplerResult.save, not pickle: a result's info can hold objects that do not pickle (a
        # closure of the sampler's), and the saved form carries the provenance load_result checks.
        result.save(os.path.join(job_dir, "result.npz"), overwrite=True)
    except Exception as exc:                              # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        status["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(os.path.join(job_dir, "status.json"), status)
        return 1
    status["ok"] = True
    _write_json(os.path.join(job_dir, "status.json"), status)
    return 0


def _stop_process(proc):
    """SIGTERM ``proc``, SIGKILL it after :data:`STOP_GRACE_S`, and return its exit code."""
    proc.terminate()
    try:
        return proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


def _last_line(path):
    try:
        with open(path, errors="replace") as fh:
            lines = [ln.strip() for ln in fh.read()[-4000:].splitlines() if ln.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


@dataclass(frozen=True, eq=False)
class JobsReport:
    """What :func:`run_jobs` did: one row per job, in the order the jobs were given.

    Attributes
    ----------
    rows : list of dict
        Per job: ``name``, ``sampler``, ``model``, ``device`` (``"gpu 3"`` or ``"cpu"``), ``cores``,
        ``status`` (``"ok"``, ``"failed"`` or ``"not run"``), ``attempts``, ``wall_s`` (all attempts,
        process start included), ``resumed`` (the fit was loaded from ``cache_dir``, not run),
        ``error`` (the last one), ``log`` (the job's own log file) and ``result`` (the saved
        :class:`~whisper_cbpf.samplers.base.SamplerResult`, ``result.npz``, for ``"ok"``).
    run_dir : str
        Where the logs, results and ``run_jobs_summary.json`` are.
    log : str
        The progress log.
    """

    rows: list
    run_dir: str
    log: str

    @property
    def passed(self):
        """``True`` when every job finished."""
        return all(r["status"] == "ok" for r in self.rows)

    @property
    def failed(self):
        """Names of the jobs that did not finish (failed or not run)."""
        return [r["name"] for r in self.rows if r["status"] != "ok"]

    def table(self):
        """The rows as a pandas DataFrame indexed by job name."""
        import pandas as pd

        cols = ["sampler", "model", "device", "cores", "status", "attempts", "wall_s", "resumed", "error"]
        return pd.DataFrame([{k: r[k] for k in ["name"] + cols} for r in self.rows],
                            columns=["name"] + cols).set_index("name")

    def result(self, name):
        """Load the :class:`~whisper_cbpf.samplers.base.SamplerResult` of finished job ``name``
        (:func:`whisper_cbpf.load_result`)."""
        from .results import load_result

        row = next((r for r in self.rows if r["name"] == name), None)
        if row is None:
            raise KeyError(f"no job named {name!r}; the jobs are {[r['name'] for r in self.rows]}.")
        if row["status"] != "ok":
            raise ValueError(f"job {name!r} did not finish ({row['status']}: {row['error']}); its log is "
                             f"{row['log']}.")
        return load_result(row["result"])

    @property
    def results(self):
        """``{name: SamplerResult}`` for every finished job, loaded from disk on each access."""
        return {r["name"]: self.result(r["name"]) for r in self.rows if r["status"] == "ok"}

    def __repr__(self):
        n_ok = sum(r["status"] == "ok" for r in self.rows)
        with_pd = self.table()
        with_pd["error"] = with_pd["error"].fillna("").str.slice(0, 60)
        text = with_pd.to_string(float_format=lambda v: f"{v:.1f}") if self.rows else "(no jobs)"
        return (f"JobsReport: {n_ok} of {len(self.rows)} jobs finished; files in {self.run_dir}\n{text}\n"
                f"Next: .results loads the fits; a failed job's log is in .rows[i]['log'].")


class _Progress:
    """The progress log: one timestamped line per event, printed and appended to a file."""

    def __init__(self, path):
        self.path, self._lock = path, threading.Lock()

    def __call__(self, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        with self._lock:
            print(line, flush=True)
            with open(self.path, "a") as fh:
                fh.write(line + "\n")


def run_jobs(jobs, *, gpus="auto", cpu_cores=None, cache_dir=None, retries=1, timeout=None):
    """Run many fits, one fresh process per job, over the GPUs and CPU cores this run may use.

    Each job is a light curve, a model and a sampler (a :class:`Job`). Jobs with a JAX sampler
    (``nuts_gpu``, ``emcee_jax``, ``abc_gpu``, ``abc_smc_gpu``, ``snpe_gpu``, ``pymc_jax_gpu_*``) run
    one per GPU at a time; the others run on CPU cores beside them. Every job process is pinned to its
    own block of cores and started with ``NCCL_P2P_DISABLE=1``; a GPU job sees only its card
    (``CUDA_VISIBLE_DEVICES`` set to the card's UUID, ``JAX_PLATFORMS=cuda``, so a CUDA failure raises
    instead of running on the CPU unnoticed), a CPU job none (``JAX_PLATFORMS=cpu``). A failed job is
    run again, up to ``retries`` more times; so is one that runs longer than ``timeout``. A one-line
    progress log records every start, finish and failure; the summary is returned and written to
    ``run_jobs_summary.json``.

    Parameters
    ----------
    jobs : list of Job
        The fits. A tuple ``(lc, model, sampler[, kwargs])`` or a dict with :class:`Job`'s fields
        also works. Names must be unique (see :class:`Job`).
    gpus : "auto", list of int, or None, optional
        ``"auto"`` (default): every card this process may see (``CUDA_VISIBLE_DEVICES``, when set)
        that is idle when the run starts; busy cards are skipped and named in the log. With no such
        card the GPU jobs run on the CPU, with a warning. A list of ``nvidia-smi`` indices: only those,
        which must be visible; the busy ones among them are skipped, and none idle is an error.
        ``None``: no GPU; every job runs on the CPU. A card is idle below
        :data:`IDLE_MEMORY_MIB` of memory and :data:`IDLE_UTILIZATION_PCT` utilisation, and is
        checked again before every job starts on it: one that stays busy for :data:`IDLE_WAIT_S`
        takes no more jobs.
    cpu_cores : None, int, str or list of int, optional
        The cores the jobs may use, shared out in contiguous blocks: one block per GPU in use, and one
        per CPU job running at a time (as many as the remaining cores allow). ``None`` (default): the
        first 60% of the cores this process may use (``whisper_cbpf.backends.CPU_FRACTION``); an int:
        that many of them; a string such as ``"0-15,32"`` or a list: exactly those.
    cache_dir : str or path, optional
        Keep finished fits here and resume: each job calls :func:`whisper_cbpf.fit_cached`, which
        returns the saved result of a fit with the same configuration instead of fitting again, so
        running an interrupted batch again fits only what had not finished. The logs and the summary
        are written here too. Default ``None``: nothing is cached and the files go to a new temporary
        directory, named in the log.
    retries : int, optional
        How many more times a failed job is run (default 1: two attempts at most). A retry may land on
        another card.
    timeout : float, optional
        Seconds one attempt of one job may run, process start-up and JAX compile included. A job
        still running then is stopped (SIGTERM, then SIGKILL after :data:`STOP_GRACE_S`), its error
        reads ``"timed out after ... s"``, and it counts as a failed attempt: it is retried while
        ``retries`` allow, which frees a slot held by a hung fit (a deadlocked worker pool, a stalled
        rejection loop). Default ``None``: no limit.

    Returns
    -------
    JobsReport
        One row per job (``status``, ``device``, ``cores``, ``attempts``, ``wall_s``, ``error``,
        ``log``), ``passed``, ``failed``, ``table()`` and ``results``.

    Raises
    ------
    ValueError, TypeError, KeyError
        Before any job starts: a malformed or unpicklable job, an unknown model or sampler, duplicate
        names, GPUs that do not exist or are not visible, or cores this process may not use.
    RuntimeError
        ``gpus`` names cards of which none is idle or that cannot be checked (no ``nvidia-smi``);
        ``cache_dir`` without :func:`whisper_cbpf.fit_cached` in this installation; or ``run_jobs``
        called from inside a job process (a script without an ``if __name__ == "__main__":`` guard).

    Notes
    -----
    Each job process starts a new Python interpreter, the equivalent of ``multiprocessing``'s
    ``spawn``: nothing is inherited but the environment, the import path and your script's
    ``__main__`` module, which is imported again (as ``multiprocessing`` does). So everything a job
    carries is pickled: a model function or sampler must be defined in a module, not in a notebook
    cell or as a lambda, and a script must call ``run_jobs`` under ``if __name__ == "__main__":``.
    Expect a few seconds per job to start Python and import the package (and a JAX compile per
    process). On a keyboard interrupt the running job processes are stopped, the summary written, and
    the interrupt raised again.

    Examples
    --------
    Three cheap fits on two CPU cores, no GPU:

    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.0, 40.0, 25)
    >>> truth = {"amplitude": 5.0, "t0": 8.0, "tau_rise": 3.0, "tau_fall": 15.0}
    >>> flux = wp.get_model("bazin").predict(truth, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 25, flux=flux, flux_err=np.full(25, 0.2), name="sn1")
    >>> jobs = [wp.Job(lc, "bazin", "abc", kwargs={"n_simulations": 2000, "seed": s}, name=f"seed{s}")
    ...         for s in range(3)]
    >>> report = wp.run_jobs(jobs, gpus=None, cpu_cores=2)          # doctest: +ELLIPSIS
    20... run_jobs: 3 jobs; GPUs: none; cores ... in 2 slot(s); files in ...
    ... done: 3 ok, 0 failed, 0 not run in ...
    >>> report.passed, sorted(report.results)
    (True, ['seed0', 'seed1', 'seed2'])

    On a GPU machine the same call with ``emcee_jax`` jobs and ``gpus="auto"`` spreads them over the
    idle cards; with ``cache_dir="runs/"`` a second call after an interruption fits only the rest.
    With ``timeout=600`` a fit that hangs frees its card after ten minutes instead of holding it
    until Ctrl-C.
    """
    if os.environ.get(_CHILD_FLAG):
        raise RuntimeError(
            "run_jobs() was called inside one of its own job processes: the job process imports your "
            "script again to find what the job needs, and the script started a new batch. Put the "
            "run_jobs(...) call under `if __name__ == '__main__':`.")
    if int(retries) != retries or retries < 0:
        raise ValueError(f"retries={retries!r} must be a whole number >= 0 (extra attempts per job).")
    if timeout is not None:
        try:
            ok = float(timeout) > 0 and float(timeout) != float("inf")
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"timeout={timeout!r} must be a positive number of seconds per attempt of "
                             f"one job, or None for no limit.")
        timeout = float(timeout)
    from .models import get_model
    from .samplers import _SAMPLERS, list_samplers

    specs = [_as_job(j, i) for i, j in enumerate(jobs)]
    if cache_dir is not None:
        try:
            from . import fit_cached  # noqa: F401  (the job processes call it)
        except ImportError:
            raise RuntimeError(
                "cache_dir= keeps and resumes fits through whisper_cbpf.fit_cached, which this "
                "installation of whisper_cbpf does not provide. Pass cache_dir=None (no resume), or "
                "use a whisper_cbpf that has it.") from None
    names = []
    for i, s in enumerate(specs):
        mname = s.model if isinstance(s.model, str) else getattr(s.model, "name", "model")
        names.append(s.name or f"{getattr(s.lc, 'name', None) or f'lc{i}'}__{mname}__{s.sampler}")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"job names must be unique, and {dupes} repeat: pass name= to each Job.")
    safe = [re.sub(r"[^A-Za-z0-9._+-]+", "_", n) for n in names]
    if len(set(safe)) != len(safe):
        raise ValueError(f"job names {names} collide once made file-safe; pass names that differ in "
                         f"letters or digits.")

    # Pickle every job now: one that cannot reach a new process fails here, before any job starts.
    cache_abs = None if cache_dir is None else os.path.abspath(os.fspath(cache_dir))
    entries = []
    for i, (s, name) in enumerate(zip(specs, names)):
        if s.sampler not in _SAMPLERS:
            raise KeyError(f"job {name!r}: unknown sampler {s.sampler!r}. Available: {list_samplers()}")
        model = get_model(s.model)
        payload = dict(lc=s.lc, model=model, sampler=s.sampler, sampler_factory=_SAMPLERS[s.sampler],
                       kwargs=dict(s.kwargs or {}), cache_dir=cache_abs)
        try:
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:                          # noqa: BLE001 - re-raised with the fix
            raise TypeError(
                f"job {name!r} cannot be sent to its own process ({type(exc).__name__}: {exc}). A job's "
                f"light curve, model, prior and keyword arguments must be picklable: define a model's "
                f"predict function in a module, not as a lambda or a closure.") from exc
        entries.append(dict(index=i, name=name, blob=blob, gpu=_needs_gpu(s.sampler),
                            row=dict(name=name, sampler=s.sampler, model=model.name, device=None,
                                     cores=None, status="not run", attempts=0, wall_s=0.0, resumed=False,
                                     error=None, log=None, result=None)))

    # Devices and cores: one slot per GPU in use, and one per CPU job running at a time.
    n_gpu_jobs = sum(e["gpu"] for e in entries)
    cards, notes = _choose_gpus(gpus) if n_gpu_jobs else ([], [])
    cards = cards[:n_gpu_jobs]
    cores = _core_budget(cpu_cores)
    if len(cards) > len(cores):
        raise ValueError(f"{len(cores)} core(s) for {len(cards)} GPU(s): every job needs at least one core. "
                         f"Pass more cpu_cores, or fewer gpus.")
    gpu_q = deque(e for e in entries if e["gpu"] and cards)
    cpu_q = deque(e for e in entries if not (e["gpu"] and cards))
    n_cpu_slots = min(len(cpu_q), len(cores) - len(cards))
    if cpu_q and n_cpu_slots < 1:
        raise ValueError(f"all {len(cores)} core(s) go to the {len(cards)} GPU(s), leaving none for the "
                         f"{len(cpu_q)} CPU job(s). Pass more cpu_cores.")
    blocks = _split(cores, len(cards) + n_cpu_slots) if entries else []
    slots = ([(gpu_q, card, blocks[k]) for k, card in enumerate(cards)]
             + [(cpu_q, None, blocks[len(cards) + k]) for k in range(n_cpu_slots)])

    # Everything is checked: write each job for its process (the caller's path and __main__ first).
    run_dir = cache_abs if cache_abs is not None else tempfile.mkdtemp(prefix="whisper_jobs_")
    prep = pickle.dumps(_preparation_data(), protocol=pickle.HIGHEST_PROTOCOL)
    for e, sname in zip(entries, safe):
        e["dir"] = os.path.join(run_dir, "jobs", sname)
        os.makedirs(e["dir"], exist_ok=True)
        with open(os.path.join(e["dir"], "job.pkl"), "wb") as fh:
            fh.write(prep + e.pop("blob"))
        e["row"]["log"] = os.path.join(e["dir"], "log.txt")

    log = _Progress(os.path.join(run_dir, "run_jobs.log"))
    log(f"run_jobs: {len(entries)} jobs; GPUs: {', '.join(f'GPU {c.index}' for c in cards) or 'none'}; "
        f"cores {_cores_text(cores)} in {len(slots)} slot(s); files in {run_dir}")
    for note in notes:
        log(note)
    if n_gpu_jobs and not cards and gpus == "auto":
        msg = (f"no idle GPU is available, so the {n_gpu_jobs} GPU-sampler job(s) run on the CPU "
               f"(JAX_PLATFORMS=cpu), typically many times slower. Pass gpus=None to choose the CPU "
               f"explicitly, or free a card.")
        import warnings
        warnings.warn(msg, stacklevel=2)
        log(msg)

    lock, stop, running = threading.Lock(), threading.Event(), set()
    total, t_start = len(entries), time.perf_counter()

    def attempt(e, card, block):
        """Run one attempt of job ``e`` in a new process; True when it finished."""
        row = e["row"]
        row["attempts"] += 1
        device = "cpu" if card is None else f"gpu {card.index}"
        row.update(device=device, cores=_cores_text(block))
        token = uuid.uuid4().hex
        env = dict(os.environ)
        env.update({_CHILD_FLAG: "1", "WHISPER_JOB_TOKEN": token, "NCCL_P2P_DISABLE": "1",
                    "WHISPER_JOB_CORES": ",".join(str(c) for c in block), "PYTHONUNBUFFERED": "1"})
        if card is None:
            env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
        else:
            env.update(CUDA_VISIBLE_DEVICES=card.uuid, JAX_PLATFORMS="cuda")
        log(f"start  [{e['index'] + 1}/{total}] {e['name']} on {device}, cores {row['cores']}, "
            f"attempt {row['attempts']} of {retries + 1}")
        t0 = time.perf_counter()
        with open(row["log"], "a") as fh:
            fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {row['attempts']}: {device}, "
                     f"cores {row['cores']} ===\n")
            fh.flush()
            proc = subprocess.Popen([sys.executable, "-c", _BOOT, e["dir"], _PKG_ROOT],
                                    stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, env=env)
            with lock:
                running.add(proc)
            timed_out = False
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = _stop_process(proc)
                fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} stopped: still running after the "
                         f"timeout of {timeout:g} s ===\n")
            with lock:
                running.discard(proc)
        row["wall_s"] += time.perf_counter() - t0
        if timed_out:
            row["error"] = (f"timed out after {timeout:g} s (timeout=): the job process was stopped. "
                            f"Raise timeout= if the fit needs longer; if it should not, look for a hang "
                            f"in its log.")
            return False
        try:
            with open(os.path.join(e["dir"], "status.json")) as fh:
                status = json.load(fh)
        except (OSError, ValueError):
            status = {}
        if status.get("token") != token:
            status = {}                                   # left by an earlier attempt or run
        if rc == 0 and status.get("ok"):
            row.update(status="ok", error=None, result=os.path.join(e["dir"], "result.npz"),
                       resumed=bool(status.get("resumed")))
            log(f"ok     [{e['index'] + 1}/{total}] {e['name']} in {row['wall_s']:.1f} s"
                + (" (the saved fit, from cache_dir)" if row["resumed"] else ""))
            return True
        row["error"] = status.get("error") or (f"the job process exited with code {rc}: "
                                               f"{_last_line(row['log']) or 'no output'}")
        return False

    def worker(q, card, block):
        while not stop.is_set():
            with lock:
                if not q:
                    return
                e = q.popleft()
            if card is not None and not _wait_until_idle(card.index, stop):
                with lock:
                    q.appendleft(e)                       # for another card
                if not stop.is_set():
                    log(f"GPU {card.index} is in use by another process: no more jobs start on it")
                return
            if attempt(e, card, block):
                continue
            row, label = e["row"], f"[{e['index'] + 1}/{total}] {e['name']}"
            if stop.is_set():
                row.update(status="not run", error="interrupted")
            elif row["attempts"] <= retries:
                log(f"failed {label}, attempt {row['attempts']}: {row['error']} -- trying again")
                with lock:
                    q.appendleft(e)
            else:
                row["status"] = "failed"
                log(f"failed {label} after {row['attempts']} attempt(s): {row['error']} (log: {row['log']})")

    def finish(interrupted):
        for e in entries:
            if e["row"]["status"] == "not run" and e["row"]["error"] is None:
                e["row"]["error"] = ("interrupted before it started" if interrupted else
                                     "no idle GPU was left to run it (every card was in use by another "
                                     "process)")
        rows = [dict(e["row"], wall_s=round(e["row"]["wall_s"], 3)) for e in entries]
        summary = os.path.join(run_dir, "run_jobs_summary.json")
        _write_json(summary, dict(rows=rows, gpus=[c.describe() for c in cards], cores=_cores_text(cores),
                                  retries=int(retries), timeout_s=timeout, interrupted=interrupted,
                                  wall_s=round(time.perf_counter() - t_start, 3)))
        counts = {k: sum(r["status"] == k for r in rows) for k in ("ok", "failed", "not run")}
        log(f"done: {counts['ok']} ok, {counts['failed']} failed, {counts['not run']} not run in "
            f"{time.perf_counter() - t_start:.1f} s{' (interrupted)' if interrupted else ''}; "
            f"summary {summary}")
        return JobsReport(rows=rows, run_dir=run_dir, log=log.path)

    # Each worker sets its own event when it ends. The main thread waits on those, not on
    # Thread.join: a KeyboardInterrupt that lands inside join() can leave that thread marked as
    # stopped while it still runs, and the summary would then be written before it finished.
    finished = [threading.Event() for _ in slots]

    def run_slot(k, q, card, block):
        try:
            worker(q, card, block)
        finally:
            finished[k].set()

    for k, slot in enumerate(slots):
        threading.Thread(target=run_slot, args=(k, *slot), daemon=True).start()
    try:
        for ev in finished:
            while not ev.wait(0.5):
                pass
    except KeyboardInterrupt:
        stop.set()
        with lock:
            procs = list(running)
        for p in procs:
            p.terminate()
        deadline = time.monotonic() + 30.0
        for ev in finished:
            ev.wait(max(0.0, deadline - time.monotonic()))
        with lock:
            for p in running:
                p.kill()
        finish(interrupted=True)
        raise
    return finish(interrupted=False)
