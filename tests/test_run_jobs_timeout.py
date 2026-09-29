"""run_jobs(timeout=): a job that runs too long is stopped, reported and retried; others finish.

The sleeper sampler is defined here so a job process can import it through this module (as
``tests/test_parallel.py`` does with its probe); it never touches a GPU.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.parallel import Job, run_jobs
from whisper_cbpf.samplers import _SAMPLERS
from whisper_cbpf.samplers.base import BaseSampler, SamplerResult

T = np.linspace(0.0, 40.0, 25)
LC = wp.LightCurve(time=T, band=["r"] * 25, flux=np.ones(25), flux_err=np.full(25, 0.2), name="sn1")


class SleeperSampler(BaseSampler):
    """Sleeps ``sleep`` seconds, then returns a one-parameter result."""

    name = "sleeper"

    def fit(self, lc, model, prior=None, *, sleep=0.0):
        time.sleep(sleep)
        return SamplerResult(sampler="sleeper", model=getattr(model, "name", str(model)),
                             parameters=["x"], samples=pd.DataFrame({"x": [0.0, 1.0]}), summary={},
                             best_params={"x": 0.5}, n_data=len(lc.time), n_params=1,
                             runtime_s=float(sleep), info={"slept": float(sleep)})


@pytest.fixture
def sleeper():
    wp.register_sampler("sleeper", SleeperSampler, overwrite=True)
    yield
    _SAMPLERS.pop("sleeper", None)


def test_a_job_past_its_timeout_is_stopped_retried_and_reported(sleeper):
    jobs = [Job(LC, "bazin", "sleeper", kwargs={"sleep": 300}, name="hung"),
            Job(LC, "bazin", "sleeper", kwargs={"sleep": 0.1}, name="quick")]
    t0 = time.time()
    # 20 s leaves the quick job room to start a process and import on one busy core (6 s measured
    # on a loaded 96-core host; 8 s timed it out there).
    rep = run_jobs(jobs, gpus=None, cpu_cores=2, retries=1, timeout=20)
    elapsed = time.time() - t0
    assert elapsed < 150, elapsed                          # two 20 s attempts, not 300 s
    hung, quick = rep.rows
    assert quick["status"] == "ok" and rep.result("quick").info["slept"] == 0.1
    assert hung["status"] == "failed" and hung["attempts"] == 2
    assert hung["error"].startswith("timed out after 20 s") and "Raise timeout=" in hung["error"]
    assert 40 <= hung["wall_s"] < 140
    assert "still running after the timeout of 20 s" in open(hung["log"]).read()
    summary = json.load(open(os.path.join(rep.run_dir, "run_jobs_summary.json")))
    assert summary["timeout_s"] == 20.0 and rep.failed == ["hung"]


@pytest.mark.parametrize("bad", [0, -1, float("inf"), "soon", float("nan")])
def test_a_timeout_that_is_not_a_positive_number_is_refused(bad):
    with pytest.raises(ValueError, match="timeout=.*positive number of seconds"):
        run_jobs([], gpus=None, timeout=bad)
