"""``compare(..., parallel=True)``: every model's fit in its own process, through ``run_jobs``.

The claims, each a test:

1. **Parallel equals serial.** With the same seed, two cheap models fitted by nested sampling in
   two processes on two CPU cores give the same comparison as the serial run: the same table,
   draws, ln Z, likelihood maxima, convergence reports and summary. Where each fit ran is
   recorded in ``settings["jobs"]``.
2. **Job assignment** (``run_jobs`` mocked, nothing runs on a card): one job per model, each
   carrying its bound model, its sampler and only the settings that sampler takes; a CPU model
   (TiDE) and a nested fit are CPU jobs, an ``emcee_jax`` fit of a JAX model is a GPU job; every
   job pickles; ``gpus`` and ``cpu_cores`` reach ``run_jobs`` unchanged.
3. **A failed job** leaves its model out with the job's error and log; the others are ranked.
4. **The evidence check** runs through the same jobs, and only for a model without a ln Z.
5. **On real cards** (skipped unless ``WHISPER_TEST_GPUS`` names two ``nvidia-smi`` indices,
   e.g. ``WHISPER_TEST_GPUS=0,1``): two JAX models with ``emcee_jax``, one card each.
"""
from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf import parallel
from whisper_cbpf.compare import compare

import importlib

C = importlib.import_module("whisper_cbpf.compare")     # wp.compare is the function

TRUTH = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}


def _toy(n=20, seed=1):
    t = np.linspace(0.5, 30.0, n)
    flux = wp.get_model("flare").predict(TRUTH, t, None)
    noisy = flux + np.random.default_rng(seed).normal(0.0, 0.1, n)
    return wp.LightCurve(time=t, band=["r"] * n, flux=noisy, flux_err=np.full(n, 0.1), name="toy")


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _same_peak(a, b):
    da, db = a.to_dict(), b.to_dict()
    da.pop("runtime_s", None)
    db.pop("runtime_s", None)
    return repr(da) == repr(db)


# --- 1. parallel equals serial -------------------------------------------------------------------
def test_parallel_equals_serial_on_the_cpu():
    lc = _toy()
    serial = _quiet(compare, lc, ["flare", "bazin"], nlive=100, seed=4)
    par = _quiet(compare, lc, ["flare", "bazin"], nlive=100, seed=4, parallel=True, gpus=None,
                 cpu_cores=2)
    assert par.table.equals(serial.table)
    assert par.winner == serial.winner == "flare" and par.criterion == serial.criterion == "ln Z"
    for lab, res in serial.results.items():
        back = par.results[lab]
        assert np.array_equal(np.asarray(back.samples), np.asarray(res.samples))
        assert back.info["log_evidence"] == res.info["log_evidence"]
        assert back.summary == res.summary
        assert _same_peak(par.peaks[lab], serial.peaks[lab])
        assert par.diagnostics[lab].to_dict() == serial.diagnostics[lab].to_dict()
        assert back.info["likelihood_max_opt"]["max_log_likelihood"] == \
            res.info["likelihood_max_opt"]["max_log_likelihood"]
        assert par.models[lab] is serial.models[lab]                 # the registered model
    assert par.summary() == serial.summary()
    assert par.evidence_check == serial.evidence_check
    jobs = par.settings["jobs"]
    assert [j["model"] for j in jobs] == ["flare", "bazin"]
    assert {j["device"] for j in jobs} == {"cpu"} and {j["status"] for j in jobs} == {"ok"}
    assert len({j["cores"] for j in jobs}) == 2                       # a core block each
    assert par.settings["parallel"] is True and "parallel" not in serial.settings


# --- 2. job assignment (run_jobs mocked) ---------------------------------------------------------
class _FakeRunJobs:
    """Stands in for ``run_jobs``: records the call, fits the CPU jobs here (or fails them, when
    told to), and fails every GPU job without touching a card."""

    def __init__(self, tmp_path, fail=()):
        self.calls, self.tmp, self.fail = [], tmp_path, set(fail)

    def __call__(self, jobs, *, gpus, cpu_cores, cache_dir):
        self.calls.append({"jobs": list(jobs), "gpus": gpus, "cpu_cores": cpu_cores,
                           "cache_dir": cache_dir})
        rows = []
        for i, job in enumerate(jobs):
            pickle.loads(pickle.dumps((job.lc, job.model, job.kwargs)))   # it reaches a process
            gpu = parallel._needs_gpu(job.sampler)
            row = {"name": job.name, "sampler": job.sampler, "model": job.model.name,
                   "device": "gpu 0" if gpu else "cpu", "cores": str(i), "status": "failed",
                   "attempts": 1, "wall_s": 0.5, "resumed": False, "error": None,
                   "log": str(self.tmp / f"{job.name}.log"), "result": None}
            if gpu or job.name in self.fail:
                row["error"] = "RuntimeError: no card in this test"
            else:
                res = _quiet(wp.fit, job.lc, job.model, job.sampler, **job.kwargs)
                path = self.tmp / job.name
                res.save(path, overwrite=True)
                row.update(status="ok", result=str(path))
            rows.append(row)
        return parallel.JobsReport(rows=rows, run_dir=str(self.tmp), log="")


@pytest.fixture
def x64():
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _alert():
    t = [60000.0, 60002.0, 60003.0, 60004.5, 60006.0, 60009.0, 60012.0, 60016.0]
    ul = [True] + [False] * 7
    return wp.LightCurve(time=t, band=["lsstg", "lsstr", "lsstg", "lsstr", "lssti", "lsstg",
                                       "lsstr", "lssti"],
                         magnitude=[22.5, 21.0, 20.6, 20.4, 20.5, 20.7, 20.9, 21.2],
                         magnitude_err=[np.nan] + [0.1] * 7, upper_limit=ul, redshift=0.05,
                         name="alert")


def test_tide_and_nested_go_to_the_cpu_and_emcee_jax_to_a_card(x64, monkeypatch, tmp_path):
    fake = _FakeRunJobs(tmp_path, fail={"fit00_tide"})
    monkeypatch.setattr(parallel, "run_jobs", fake)
    with pytest.raises(RuntimeError, match="no card in this test"):    # every fit failed here
        _quiet(compare, _alert(), ["tide", "tde"], sampler={"tde": "emcee_jax"},
               evidence_check=False, parallel=True, gpus=[0, 3], cpu_cores="0-3", nlive=50)
    (call,) = fake.calls
    assert call["gpus"] == [0, 3] and call["cpu_cores"] == "0-3" and call["cache_dir"] is None
    jobs = {j.name: j for j in call["jobs"]}
    assert list(jobs) == ["fit00_tide", "fit01_tde"]
    tide, tde = jobs["fit00_tide"], jobs["fit01_tde"]
    assert (tide.sampler, tde.sampler) == ("nested", "emcee_jax")
    assert tide.model.predict_jax is None and tde.model.predict_jax is not None
    assert not parallel._needs_gpu(tide.sampler) and parallel._needs_gpu(tde.sampler)
    assert tide.kwargs == {"nlive": 50, "seed": 0}                   # nested takes nlive
    assert "nlive" not in tde.kwargs and tde.kwargs["seed"] == 0      # emcee_jax does not
    assert tide.lc is tde.lc


def test_every_jax_model_with_auto_is_a_cpu_job(x64, monkeypatch, tmp_path):
    fake = _FakeRunJobs(tmp_path, fail={"fit00_tde", "fit01_tde_analytical"})
    monkeypatch.setattr(parallel, "run_jobs", fake)
    with pytest.raises(RuntimeError):
        _quiet(compare, _alert(), ["tde", "tde_analytical"], evidence_check=False,
               parallel=True, nlive=50)
    jobs = fake.calls[0]["jobs"]
    assert [j.sampler for j in jobs] == ["nested", "nested"]
    assert not any(parallel._needs_gpu(j.sampler) for j in jobs)
    assert fake.calls[0]["gpus"] == "auto" and fake.calls[0]["cpu_cores"] is None


# --- 3. a failed job -----------------------------------------------------------------------------
def test_a_failed_job_leaves_its_model_out_with_its_log(monkeypatch, tmp_path):
    fake = _FakeRunJobs(tmp_path, fail={"fit01_bazin"})
    monkeypatch.setattr(parallel, "run_jobs", fake)
    with pytest.warns(UserWarning, match="'bazin' failed"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="(?!compare:)")
            cmp = compare(_toy(), ["flare", "bazin"], parallel=True, nlive=100)
    row = cmp.table.set_index("model").loc["bazin"]
    assert row["status"] == "left out" and "no card in this test" in row["left_out_reason"]
    assert "fit01_bazin.log" in row["left_out_reason"]
    assert cmp.winner == "flare" and cmp.timing["flare"]["fit_s"] == 0.5
    assert [j["status"] for j in cmp.settings["jobs"]] == ["ok", "failed"]


# --- 4. the evidence check through the same jobs -------------------------------------------------
def test_the_evidence_check_runs_as_jobs_only_for_a_model_without_ln_z(monkeypatch, tmp_path):
    fake = _FakeRunJobs(tmp_path)
    monkeypatch.setattr(parallel, "run_jobs", fake)
    cmp = _quiet(compare, _toy(), ["flare", "bazin"], sampler={"bazin": "mcmc"},
                 evidence_check=True, parallel=True, nlive=100, nsteps=1500, burnin=500)
    fit_call, check_call = fake.calls
    assert [j.sampler for j in fit_call["jobs"]] == ["nested", "mcmc"]
    assert [(j.name, j.sampler) for j in check_call["jobs"]] == [("evidence00_bazin", "nested")]
    assert set(cmp.evidence) == {"bazin"} and cmp.evidence_check["ran"]
    assert [j["stage"] for j in cmp.settings["jobs"]] == ["fit", "fit", "evidence"]


# --- 5. on real cards ----------------------------------------------------------------------------
GPUS = [int(g) for g in os.environ.get("WHISPER_TEST_GPUS", "").split(",") if g.strip()]


@pytest.mark.slow
@pytest.mark.skipif(len(GPUS) < 2, reason="set WHISPER_TEST_GPUS to two idle nvidia-smi indices")
def test_two_jax_models_on_two_cards(x64):
    cmp = _quiet(compare, _alert(), ["tde", "tde_analytical"], sampler="emcee_jax",
                 evidence_check=False, parallel=True, gpus=GPUS[:2])
    devices = [j["device"] for j in cmp.settings["jobs"]]
    assert sorted(devices) == sorted(f"gpu {g}" for g in GPUS[:2])
    assert all(np.isfinite(cmp.table["max_log_likelihood"]))
