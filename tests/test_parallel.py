"""run_jobs: many fits, one process per job, over the GPUs and cores allowed.

Most tests run a probe sampler (defined below, importable by a job process through this module) that
records where it ran: the environment it was given, the cores it was pinned to, and when it ran. The
multi-card test gives the scheduler a fake ``nvidia-smi`` table, so it can check card assignment on a
machine whose real cards belong to other people; the probe never touches CUDA, so no card is used.

One test runs real jobs on a real card and is skipped unless ``WHISPER_TEST_GPU`` names an
``nvidia-smi`` index, e.g. ``WHISPER_TEST_GPU=3 CUDA_VISIBLE_DEVICES=3 python -m pytest
tests/test_parallel.py``.
"""
from __future__ import annotations

import _thread
import json
import os
import re
import threading
import time

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf import parallel
from whisper_cbpf.parallel import Job, run_jobs
from whisper_cbpf.samplers import _SAMPLERS
from whisper_cbpf.samplers.base import BaseSampler, SamplerResult

T = np.linspace(0.0, 40.0, 25)
TRUTH = {"amplitude": 5.0, "t0": 8.0, "tau_rise": 3.0, "tau_fall": 15.0}
LC = wp.LightCurve(time=T, band=["r"] * 25, flux=wp.get_model("bazin").predict(TRUTH, T, None),
                   flux_err=np.full(25, 0.2), name="sn1")


class ProbeSampler(BaseSampler):
    """Records where it ran; can sleep, fail once (``fail_first=<marker path>``) or always."""

    name = "probe"

    def fit(self, lc, model, prior=None, *, sleep=0.0, fail_first=None, fail_always=False, use_jax=False,
            count_dir=None):
        rec = dict(pid=os.getpid(), cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   jax_platforms=os.environ.get("JAX_PLATFORMS"), nccl=os.environ.get("NCCL_P2P_DISABLE"),
                   affinity=sorted(os.sched_getaffinity(0)), start=time.time())
        if count_dir is not None:                        # one file per fit that really ran
            open(os.path.join(count_dir, f"{os.getpid()}_{time.time_ns()}"), "w").close()
        if fail_first is not None and not os.path.exists(fail_first):
            open(fail_first, "w").close()
            raise RuntimeError("forced failure on the first attempt")
        if fail_always:
            raise RuntimeError("this job always fails")
        if use_jax:
            import jax
            rec["jax_devices"] = [d.platform for d in jax.devices()]
        time.sleep(sleep)
        rec["end"] = time.time()
        return SamplerResult(sampler="probe", model=getattr(model, "name", str(model)), parameters=["x"],
                             samples=pd.DataFrame({"x": [0.0, 1.0]}), summary={}, best_params={"x": 0.5},
                             n_data=len(lc.time), n_params=1, runtime_s=rec["end"] - rec["start"],
                             info={"probe": rec})


@pytest.fixture
def probe():
    wp.register_sampler("probe", ProbeSampler, overwrite=True)
    yield
    _SAMPLERS.pop("probe", None)


def _fake_table(busy=(), cards=(0, 1, 2, 3)):
    return lambda: [parallel._Card(i, f"GPU-fake-{i}", 9000.0 if i in busy else 5.0, 0.0) for i in cards]


def _cores(text):
    return parallel._parse_cpu_list(text)


# --- choosing cards and cores (no process started) -----------------------------------------------------

def test_visible_cards_follow_cuda_visible_devices(monkeypatch):
    cards = _fake_table()()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert [c.index for c in parallel._visible(cards)] == [0, 1, 2, 3]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert [c.index for c in parallel._visible(cards)] == [3, 1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fake-2")
    assert [c.index for c in parallel._visible(cards)] == [2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,7,1")          # CUDA stops at the first bad token
    assert [c.index for c in parallel._visible(cards)] == [0]
    for hidden in ("", "-1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", hidden)
        assert parallel._visible(cards) == []


def test_auto_takes_the_idle_visible_cards_and_names_the_busy_ones(monkeypatch):
    monkeypatch.setattr(parallel, "_gpu_table", _fake_table(busy={1}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3")
    cards, notes = parallel._choose_gpus("auto")
    assert [c.index for c in cards] == [0, 3] and cards[0].uuid == "GPU-fake-0"
    assert notes == ["skipped busy GPU 1 (9000 MiB used, 0% busy)"]
    assert parallel._choose_gpus(None) == ([], []) and parallel._choose_gpus([]) == ([], [])


def test_explicit_gpus_are_checked(monkeypatch):
    monkeypatch.setattr(parallel, "_gpu_table", _fake_table(busy={1, 3}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3")
    assert [c.index for c in parallel._choose_gpus([0, 1])[0]] == [0]
    assert [c.index for c in parallel._choose_gpus("0,1")[0]] == [0]
    assert [c.index for c in parallel._choose_gpus(0)[0]] == [0]
    with pytest.raises(ValueError, match=r"GPU\(s\) \[2\] are outside CUDA_VISIBLE_DEVICES='0,1,3'"):
        parallel._choose_gpus([2])
    with pytest.raises(ValueError, match=r"GPU\(s\) \[9\] do not exist"):
        parallel._choose_gpus([9])
    with pytest.raises(RuntimeError, match=r"none of GPU\(s\) \[1, 3\] is idle.*never shares a busy card"):
        parallel._choose_gpus([1, 3])
    with pytest.raises(ValueError, match="is not a GPU index"):
        parallel._choose_gpus(["first"])


def test_without_nvidia_smi_auto_uses_no_card_and_explicit_refuses(monkeypatch):
    monkeypatch.setattr(parallel, "_gpu_table", lambda: None)
    assert parallel._choose_gpus("auto") == ([], ["nvidia-smi could not be run: no GPU is used"])
    with pytest.raises(RuntimeError, match="cannot check that GPU"):
        parallel._choose_gpus([0])


def test_core_budget_and_split():
    allowed = sorted(os.sched_getaffinity(0))
    assert parallel._core_budget(None) == allowed[:max(1, round(0.6 * len(allowed)))]
    assert parallel._core_budget(1) == allowed[:1]
    assert parallel._core_budget(str(allowed[0])) == allowed[:1]
    assert parallel._parse_cpu_list("0-3,8,2") == [0, 1, 2, 3, 8]
    assert parallel._cores_text([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert parallel._split(list(range(7)), 3) == [[0, 1, 2], [3, 4], [5, 6]]
    with pytest.raises(ValueError, match="outside this process's CPU affinity"):
        parallel._core_budget([max(allowed) + 1000])
    with pytest.raises(ValueError, match="is not N or N-M"):
        parallel._parse_cpu_list("3-1")
    with pytest.raises(ValueError, match="this process may use"):
        parallel._core_budget(len(allowed) + 1)


# --- errors before any job starts ------------------------------------------------------------------------

def test_malformed_jobs_are_refused_before_anything_runs(probe):
    with pytest.raises(TypeError, match="pass a whisper_cbpf.Job"):
        run_jobs([object()], gpus=None)
    with pytest.raises(ValueError, match=r"\['sn1__bazin__probe'\] repeat: pass name="):
        run_jobs([(LC, "bazin", "probe"), (LC, "bazin", "probe")], gpus=None)
    with pytest.raises(KeyError, match="unknown sampler 'nope'"):
        run_jobs([Job(LC, "bazin", "nope")], gpus=None)
    with pytest.raises(KeyError, match="Unknown model"):
        run_jobs([Job(LC, "no_such_model", "probe")], gpus=None)
    closure = wp.Model("closure", lambda p, t, b: np.ones(len(t)), ["x"])
    with pytest.raises(TypeError, match="cannot be sent to its own process.*not as a lambda"):
        run_jobs([Job(LC, closure, "probe")], gpus=None)
    with pytest.raises(ValueError, match="retries"):
        run_jobs([Job(LC, "bazin", "probe")], gpus=None, retries=-1)


def test_run_jobs_inside_a_job_process_refuses(monkeypatch):
    monkeypatch.setenv(parallel._CHILD_FLAG, "1")
    with pytest.raises(RuntimeError, match="if __name__ == '__main__'"):
        run_jobs([Job(LC, "bazin", "abc")], gpus=None)


def test_cache_dir_needs_fit_cached(monkeypatch, tmp_path):
    monkeypatch.delattr(wp, "fit_cached", raising=False)
    with pytest.raises(RuntimeError, match="whisper_cbpf.fit_cached"):
        run_jobs([Job(LC, "bazin", "abc")], gpus=None, cache_dir=tmp_path)


def test_no_jobs_is_an_empty_report():
    rep = run_jobs([], gpus=None)
    assert rep.passed and rep.rows == [] and "0 of 0 jobs" in repr(rep)


# --- real processes -----------------------------------------------------------------------------------------

def test_six_jobs_over_two_cards(probe, monkeypatch, tmp_path):
    """The gate on fake cards: 6 jobs, 2 idle cards, a busy card and a hidden
    card never used, one job per card at a time, a forced failure retried, pinned cores."""
    monkeypatch.setattr(parallel, "_gpu_table", _fake_table(busy={1}))
    monkeypatch.setattr(parallel, "_needs_gpu", lambda sampler: sampler == "probe")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3")                        # card 2 is not ours
    marker = tmp_path / "failed_once"
    jobs = [Job(LC, "bazin", "probe", kwargs={"sleep": 0.5}, name=f"job{i}") for i in range(5)]
    jobs.append(Job(LC, "bazin", "probe", kwargs={"sleep": 0.5, "fail_first": str(marker)}, name="flaky"))
    cores = sorted(os.sched_getaffinity(0))[:4]
    rep = run_jobs(jobs, cpu_cores=cores)

    assert rep.passed, rep
    assert {r["device"] for r in rep.rows} == {"gpu 0", "gpu 3"}
    flaky = next(r for r in rep.rows if r["name"] == "flaky")
    assert flaky["attempts"] == 2 and marker.exists()
    assert all(r["attempts"] == 1 for r in rep.rows if r["name"] != "flaky")
    probes = {name: res.info["probe"] for name, res in rep.results.items()}
    blocks = {}
    for row in rep.rows:
        p = probes[row["name"]]
        card = int(row["device"].split()[1])
        assert p["cuda"] == f"GPU-fake-{card}" and p["jax_platforms"] == "cuda" and p["nccl"] == "1"
        assert parallel._cores_text(p["affinity"]) == row["cores"]
        blocks.setdefault(card, set()).add(row["cores"])
    assert all(len(v) == 1 for v in blocks.values())                            # one core block per card
    b0, b3 = (set(_cores(next(iter(blocks[c])))) for c in (0, 3))
    assert not (b0 & b3) and b0 | b3 == set(cores)
    for card in (0, 3):                                                        # one job per card at a time
        spans = sorted((probes[r["name"]]["start"], probes[r["name"]]["end"]) for r in rep.rows
                       if r["device"] == f"gpu {card}")
        assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))
    text = open(rep.log).read()
    assert "skipped busy GPU 1" in text and "forced failure on the first attempt -- trying again" in text
    assert "GPU 2" not in text
    summary = json.load(open(os.path.join(rep.run_dir, "run_jobs_summary.json")))
    assert [r["status"] for r in summary["rows"]] == ["ok"] * 6


def test_real_cpu_jobs_match_an_in_process_fit_and_failures_are_reported(probe):
    jobs = [Job(LC, "bazin", "abc", kwargs={"n_simulations": 2000, "seed": 1}, name="abc1"),
            Job(LC, "bazin", "probe", kwargs={"fail_always": True}, name="broken")]
    rep = run_jobs(jobs, gpus=None, cpu_cores=2, retries=1)
    assert not rep.passed and rep.failed == ["broken"]
    ok, bad = rep.rows
    assert ok["status"] == "ok" and ok["device"] == "cpu" and ok["attempts"] == 1
    direct = wp.fit(LC, "bazin", "abc", n_simulations=2000, seed=1)
    got = rep.result("abc1")
    assert got.samples.equals(direct.samples) and got.best_params == direct.best_params
    assert bad["status"] == "failed" and bad["attempts"] == 2
    assert bad["error"] == "RuntimeError: this job always fails"
    assert "Traceback" in open(bad["log"]).read()
    with pytest.raises(ValueError, match="did not finish"):
        rep.result("broken")
    assert "broken" in rep.table().index and "1 of 2 jobs finished" in repr(rep)


def test_a_card_that_becomes_busy_takes_no_more_jobs(probe, monkeypatch):
    calls = {"n": 0}

    def table():                                 # idle for the choice and the first launch, then busy
        calls["n"] += 1
        return _fake_table(busy={0} if calls["n"] > 2 else set(), cards=(0,))()
    monkeypatch.setattr(parallel, "_gpu_table", table)
    monkeypatch.setattr(parallel, "_needs_gpu", lambda sampler: sampler == "probe")
    monkeypatch.setattr(parallel, "IDLE_WAIT_S", 0.5)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rep = run_jobs([Job(LC, "bazin", "probe", name=f"j{i}") for i in range(2)], cpu_cores=1)
    assert [r["status"] for r in rep.rows] == ["ok", "not run"]
    assert "no idle GPU was left" in rep.rows[1]["error"]
    assert "GPU 0 is in use by another process" in open(rep.log).read()


def test_keyboard_interrupt_stops_the_jobs_and_writes_the_summary(probe, capsys):
    jobs = [Job(LC, "bazin", "probe", kwargs={"sleep": 120}, name=f"slow{i}") for i in range(3)]
    timer = threading.Timer(8.0, _thread.interrupt_main)
    timer.start()
    t0 = time.time()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_jobs(jobs, gpus=None, cpu_cores=2)
    finally:
        timer.cancel()
    assert time.time() - t0 < 60
    run_dir = re.search(r"files in (\S+)", capsys.readouterr().out).group(1)
    summary = json.load(open(os.path.join(run_dir, "run_jobs_summary.json")))
    assert summary["interrupted"]
    assert [r["status"] for r in summary["rows"]] == ["not run"] * 3
    assert [r["error"] for r in summary["rows"]] == ["interrupted", "interrupted",
                                                      "interrupted before it started"]


def test_cache_dir_resumes_what_had_not_finished(probe, tmp_path):
    """Item 10's gate: an interrupted batch run again fits only the jobs that had not finished."""
    if not callable(getattr(wp, "fit_cached", None)):
        pytest.skip("whisper_cbpf.fit_cached is not available")
    fits = tmp_path / "fits"
    fits.mkdir()
    jobs = [Job(LC, "bazin", "probe", kwargs={"count_dir": str(fits), "sleep": 0.01 * i}, name=f"j{i}")
            for i in range(3)]
    first = run_jobs(jobs[:2], gpus=None, cpu_cores=2, cache_dir=tmp_path / "cache")   # "stopped" after 2
    assert first.passed and len(os.listdir(fits)) == 2
    again = run_jobs(jobs, gpus=None, cpu_cores=2, cache_dir=tmp_path / "cache")
    assert again.passed and len(os.listdir(fits)) == 3                          # only j2 was fitted
    assert [r["resumed"] for r in again.rows] == [True, True, False]
    assert again.result("j0").info["probe"] == first.result("j0").info["probe"]


def test_docstring_examples_run():
    import doctest

    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for obj, name in ((parallel.run_jobs, "run_jobs"), (parallel.Job, "Job")):
        for test in doctest.DocTestFinder().find(obj, name, globs={}):
            runner.run(test)
    assert runner.tries > 0 and runner.failures == 0


# --- a real card ------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("WHISPER_TEST_GPU"),
                    reason="set WHISPER_TEST_GPU=<nvidia-smi index> to run jobs on that card")
def test_jobs_on_a_real_card_see_only_that_card(probe, monkeypatch):
    pytest.importorskip("jax")
    index = int(os.environ["WHISPER_TEST_GPU"])
    card = next((c for c in parallel._gpu_table() or [] if c.index == index), None)
    if card is None or not card.idle:
        pytest.skip(f"GPU {index} is not idle")
    monkeypatch.setattr(parallel, "_needs_gpu", lambda s: s in ("probe", "emcee_jax"))
    from whisper_cbpf.models.flare import flare_flux
    t = np.linspace(0.5, 30.0, 40)
    flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    toy = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1), name="toy")
    jobs = [Job(LC, "bazin", "probe", kwargs={"use_jax": True}, name="probe_gpu"),
            Job(toy, "flare_jax", "emcee_jax", name="emcee_gpu",
                kwargs=dict(nwalkers=16, nsteps=300, burnin=100, seed=0, init="prior"))]
    rep = run_jobs(jobs, gpus=[index], cpu_cores=4)
    assert rep.passed, rep
    assert [r["device"] for r in rep.rows] == [f"gpu {index}"] * 2
    rec = rep.result("probe_gpu").info["probe"]
    assert rec["cuda"] == card.uuid and rec["jax_platforms"] == "cuda" and rec["nccl"] == "1"
    assert rec["jax_devices"] == ["gpu"]
    emcee = rep.result("emcee_gpu")
    assert emcee.n_samples > 0 and emcee.provenance["jax_devices"]["backend"] == "gpu"
