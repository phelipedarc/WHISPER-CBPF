"""Save / load with provenance, resumable runs, SamplerResult delegation.

The gates: a round trip gives identical summaries and hashes; a batch killed part-way finishes only
what is missing when it is run again.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf import results as R
from whisper_cbpf.models.flare import flare_flux
from whisper_cbpf.priors import Prior, Uniform
from whisper_cbpf.samplers.base import SamplerResult

TRUTH = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}


def _lc(scale=1.0, name="toy"):
    t = np.linspace(0.5, 30.0, 40)
    flux = scale * flare_flux(TRUTH, t, None)
    return wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1), name=name)


def _quiet_fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.fit(*args, **kwargs)


@pytest.fixture(scope="module")
def lc():
    return _lc()


@pytest.fixture(scope="module")
def mcmc(lc):
    return _quiet_fit(lc, "flare", sampler="mcmc", nsteps=600, burnin=200, seed=0)


def _same_json(a, b):
    return json.dumps(R._encode(a, "", []), sort_keys=True) == json.dumps(R._encode(b, "", []),
                                                                          sort_keys=True)


def _manifest(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            return json.loads(z["__manifest__"].tobytes())
    return json.loads((path / R.MANIFEST).read_text())


def _without_time(m):
    """A manifest without the save time and the list of live objects (a loaded result has none)."""
    return {k: v for k, v in m.items() if k not in ("saved_utc", "not_saved")}


# ------------------------------------------------------------------------------------ provenance
def test_the_fit_records_its_provenance(lc, mcmc):
    p = mcmc.provenance
    assert p["recorded_at"] == "fit" and "error" not in p
    assert p["whisper"]["version"] == wp.__version__
    assert p["packages"]["numpy"] == np.__version__ and p["packages"]["pandas"] == pd.__version__
    assert p["data"]["hash"] == R.data_hash(lc) and p["data"]["n_points"] == 40
    m = p["model"]
    assert m["name"] == "flare" and m["parameters"] == ["amplitude", "rise_time", "decay_time"]
    assert m["description"] == wp.get_model("flare").description
    assert m["prior"]["repr"] == repr(wp.get_model("flare").default_prior)
    assert m["prior"]["parameters"]["amplitude"] == {"type": "Uniform", "low": 0.0, "high": 10.0}
    assert m["prior_source"] == "the model's default"
    s = p["sampler"]
    assert s["name"] == "mcmc" and s["seed"] == 0
    assert s["kwargs"] == {"nsteps": 600, "burnin": 200, "seed": 0}
    assert "thin" in s["defaults"]                       # the settings that ran, not only those passed
    assert p["environment"]["band_system"] == wp.default_band_system()
    assert p["wall_s"] >= mcmc.runtime_s > 0


def test_a_prior_passed_to_fit_is_the_one_recorded(lc):
    prior = Prior({"amplitude": Uniform(1.0, 9.0), "rise_time": Uniform(1.0, 10.0),
                   "decay_time": Uniform(5.0, 30.0)})
    res = _quiet_fit(lc, "flare", sampler="abc", prior=prior, n_simulations=500, quantile=0.1,
                     seed=0)
    assert res.provenance["model"]["prior"]["parameters"]["amplitude"]["low"] == 1.0
    assert res.provenance["model"]["prior_source"] == "passed to fit"
    assert "prior" not in res.provenance["sampler"]["kwargs"]


def test_a_failure_to_record_never_fails_the_fit(lc, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no record today")
    monkeypatch.setattr(R, "record_call", boom)
    res = _quiet_fit(lc, "flare", sampler="abc", n_simulations=300, quantile=0.1, seed=0)
    assert res.n_samples > 0
    assert res.provenance["error"] == "RuntimeError: no record today"


def test_data_hash_sees_values_metadata_and_nothing_else():
    a, b = _lc(), _lc()
    assert R.data_hash(a) == R.data_hash(b)
    b["flux"][3] += 1e-12
    assert R.data_hash(a) != R.data_hash(b)
    c = a.copy()
    c.meta["redshift"] = 0.1
    assert R.data_hash(a) != R.data_hash(c)


# ------------------------------------------------------------------------------------ round trip
@pytest.mark.parametrize("name", ["saved_dir", "saved.npz"])
def test_round_trip_is_exact(tmp_path, mcmc, name):
    path = mcmc.save(tmp_path / name)
    back = wp.load_result(path)
    assert back.summary == mcmc.summary
    assert back.samples.equals(mcmc.samples)
    assert back.best_params == mcmc.best_params
    for f in ("sampler", "model", "parameters", "n_data", "n_params", "runtime_s", "aic", "bic",
              "max_log_likelihood"):
        assert getattr(back, f) == getattr(mcmc, f), f
    assert np.isnan(back.min_distance) and np.isnan(mcmc.min_distance)
    assert _same_json(back.info, mcmc.info)
    assert back.provenance == json.loads(json.dumps(mcmc.provenance))
    i = mcmc.info                                          # emcee stores step by step: rebuilt exactly
    ref = np.swapaxes(mcmc.emcee_sampler.get_chain(discard=i["burnin"], thin=i["thin"]), 0, 1)
    assert np.array_equal(back.samples_by_chain, ref)
    assert back.loaded_from == str(path)
    m = _manifest(path)
    assert m["hashes"]["samples"] == R.samples_hash(mcmc.samples) == R.samples_hash(back.samples)
    assert m["hashes"]["data"] == R.data_hash(_lc())
    assert m["result"]["seed"] == 0 and m["result"]["chains_shape"] == list(ref.shape)
    assert m["not_saved"] == ["emcee_sampler"] and m["lossy_info_keys"] == []
    assert set(m["timing"]) >= {"runtime_s", "compile_s", "run_s", "wall_s", "note"}


def test_saving_a_loaded_result_reproduces_the_manifest(tmp_path, mcmc):
    first = mcmc.save(tmp_path / "a")
    second = wp.load_result(first).save(tmp_path / "b.npz")
    third = wp.load_result(second).save(tmp_path / "c")
    m1, m2, m3 = (_without_time(_manifest(p)) for p in (first, second, third))
    assert m1 == m2 == m3
    assert (tmp_path / "a" / R.RESULT).read_bytes() == (tmp_path / "c" / R.RESULT).read_bytes()


def test_existing_saves_are_kept_unless_overwrite(tmp_path, mcmc):
    path = mcmc.save(tmp_path / "r")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        mcmc.save(path)
    mcmc.save(path, overwrite=True)
    single = mcmc.save(tmp_path / "r.npz")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        mcmc.save(single)
    (tmp_path / "file").write_text("x")
    with pytest.raises(FileExistsError, match="is a file"):
        mcmc.save(tmp_path / "file")


def test_a_modified_save_is_refused(tmp_path, mcmc):
    path = mcmc.save(tmp_path / "r")
    body = (path / R.RESULT).read_text()
    (path / R.RESULT).write_text(body.replace('"aic"', '"aic" ', 1))
    with pytest.raises(ValueError, match="hash of the result file differs.*modified or truncated"):
        wp.load_result(path)
    path = mcmc.save(tmp_path / "s")
    with np.load(path / R.ARRAYS) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["samples/0"] = arrays["samples/0"] + 1e-9
    np.savez(path / R.ARRAYS, **arrays)
    with pytest.raises(ValueError, match="hash of the posterior draws differs"):
        wp.load_result(path)


def test_load_errors_name_the_cause_and_the_fix(tmp_path, mcmc):
    with pytest.raises(FileNotFoundError, match="no saved result"):
        wp.load_result(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=r"not a saved whisper result.*result\.save"):
        wp.load_result(tmp_path / "empty")
    np.savez(tmp_path / "plain.npz", x=np.arange(3))
    with pytest.raises(ValueError, match="not a saved whisper result"):
        wp.load_result(tmp_path / "plain.npz")
    path = mcmc.save(tmp_path / "future")
    m = _manifest(path)
    m["format_version"] = R.FORMAT_VERSION + 1
    (path / R.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Upgrade whisper_cbpf"):
        wp.load_result(path)


def _hand_built(info, index=None):
    samples = pd.DataFrame({"a": np.linspace(0, 1, 6), "b": np.arange(6, dtype=np.float32)},
                           index=index)
    return SamplerResult(sampler="custom", model="toy", parameters=["a", "b"], samples=samples,
                         summary={"a": {"median": 0.5}}, best_params={"a": 0.5, "b": 2.0},
                         n_data=10, n_params=2, runtime_s=1.5, info=info)


def test_info_arrays_tuples_and_keys_survive(tmp_path):
    class Opaque:
        pass
    info = {"f32": np.arange(3, dtype=np.float32) / 3, "i64": np.array([[1, 2], [3, 4]]),
            "flags": np.array([True, False]), "pair": (1, "x"), "int_keys": {1: "a", 2: [1.5]},
            "scalars": [np.float32(0.1), np.int64(7), np.bool_(True), float("nan")],
            "opaque": Opaque()}
    res = _hand_built(info, index=[5, 7, 9, 11, 13, 15])
    path = res.save(tmp_path / "h")
    back = wp.load_result(path)
    bi = back.info
    assert bi["f32"].dtype == np.float32 and np.array_equal(bi["f32"], info["f32"])
    assert bi["i64"].dtype == np.int64 and bi["i64"].shape == (2, 2)
    assert bi["flags"].dtype == bool and bi["pair"] == (1, "x") and bi["int_keys"] == {1: "a", 2: [1.5]}
    assert bi["scalars"][0] == float(np.float32(0.1)) and bi["scalars"][1] == 7
    assert np.isnan(bi["scalars"][3])
    assert bi["opaque"]["type"].endswith("Opaque")
    assert back.samples.equals(res.samples) and list(back.samples.index) == [5, 7, 9, 11, 13, 15]
    assert back.samples["b"].dtype == np.float32
    m = _manifest(path)
    assert m["lossy_info_keys"] == ["info.opaque"]
    assert m["provenance"]["recorded_at"] == "save"            # a result built by hand
    assert "not recorded" in m["provenance"]["note"]
    again = back.save(tmp_path / "h2")                          # a placeholder saves as itself
    assert _manifest(again)["hashes"] == m["hashes"]


def test_samples_by_chain_attribute_is_saved(tmp_path):
    res = _hand_built({})
    res.samples_by_chain = np.arange(12.0).reshape(2, 3, 2)
    back = wp.load_result(res.save(tmp_path / "c.npz"))
    assert np.array_equal(back.samples_by_chain, res.samples_by_chain)


# -------------------------------------------------------------------------- backwards compatible
def test_old_construction_and_to_dict_are_unchanged():
    args = ("abc", "flare", ["a"], pd.DataFrame({"a": [1.0]}), {}, {"a": 1.0}, 5, 1, 0.1,
            {"k": 1}, 0.5, -1.0, 4.0, 5.0)
    res = SamplerResult(*args)
    assert res.provenance == {} and res.min_distance == 0.5 and res.bic == 5.0
    assert set(res.to_dict()) == {"sampler", "model", "parameters", "n_data", "n_params",
                                  "n_samples", "runtime_s", "min_distance", "max_log_likelihood",
                                  "aic", "bic", "best_params", "summary", "info"}
    assert repr(res).startswith("SamplerResult(sampler='abc'")


# ---------------------------------------------------------------------------- likelihood_max_opt
def test_likelihood_max_opt_delegates_to_its_module(monkeypatch, mcmc, lc):
    import importlib
    module = importlib.import_module("whisper_cbpf.likelihood_max_opt")
    seen = {}

    def fake(result, lc_, model=None, **kw):
        seen.update(result=result, lc=lc_, model=model, kw=kw)
        return "peak"
    monkeypatch.setattr(module, "likelihood_max_opt", fake)
    assert mcmc.likelihood_max_opt(lc, n_starts=2, seed=3) == "peak"
    assert seen["result"] is mcmc and seen["lc"] is lc and seen["model"] is None
    assert seen["kw"] == {"n_starts": 2, "seed": 3}


def test_likelihood_max_opt_without_the_module_says_what_is_missing(monkeypatch, mcmc, lc):
    monkeypatch.setitem(sys.modules, "whisper_cbpf.likelihood_max_opt", None)
    with pytest.raises(ImportError, match="whisper_cbpf.likelihood_max_opt module"):
        mcmc.likelihood_max_opt(lc)


# ------------------------------------------------------------------------------------ fit_cached
@pytest.fixture
def counted(monkeypatch):
    """Count the fits that actually run inside fit_cached."""
    import whisper_cbpf.samplers as S
    calls = []
    real = S.fit

    def fit(*args, **kwargs):
        calls.append(kwargs.get("seed"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return real(*args, **kwargs)
    monkeypatch.setattr(S, "fit", fit)
    return calls


ABC = dict(n_simulations=400, quantile=0.1)


def test_fit_cached_runs_once_then_loads(tmp_path, lc, counted):
    first = wp.fit_cached(lc, "flare", "abc", tmp_path, seed=0, **ABC)
    again = wp.fit_cached(lc, "flare", "abc", tmp_path, seed=0, **ABC)
    assert counted == [0]
    assert getattr(first, "loaded_from", None) is None and again.loaded_from is not None
    assert again.summary == first.summary and again.samples.equals(first.samples)
    key = again.provenance["cache"]["config_hash"]
    assert key == R.cache_config(lc, "flare", "abc", seed=0, **ABC)[1]
    assert Path(again.loaded_from).name == f"toy__flare__abc__{key[:16]}"
    assert [p.name for p in tmp_path.iterdir()] == [Path(again.loaded_from).name]


def test_what_changes_the_configuration(tmp_path, lc):
    def key(lc_=lc, model="flare", sampler="abc", **kw):
        return R.cache_config(lc_, model, sampler, **{**ABC, "seed": 0, **kw})[1]
    base = key()
    assert key(progress=True) == base                        # display only
    assert key(seed=1) != base
    assert key(quantile=0.2) != base
    assert key(lc_=_lc(scale=1.01)) != base
    assert (R.cache_config(lc, "flare", "abc", seed=0)[1]
            != R.cache_config(lc, "flare", "abc_smc", seed=0)[1])
    with pytest.raises(TypeError, match="sampler 'abc_smc'.*n_simulations"):
        key(sampler="abc_smc")
    prior = Prior({"amplitude": Uniform(0.0, 9.0), "rise_time": Uniform(1.0, 10.0),
                   "decay_time": Uniform(5.0, 30.0)})
    assert key(prior=prior) != base

    def model(scale):                                        # same name, different closure data
        def predict(p, t, b):
            return scale * flare_flux(p, t, b)
        return wp.Model(name="scaled_flare", predict=predict,
                        parameters=list(TRUTH), default_prior=wp.get_model("flare").default_prior)
    assert key(model=model(1.0)) == key(model=model(1.0))
    assert key(model=model(1.0)) != key(model=model(2.0))


def test_a_model_that_refers_to_itself_still_hashes(lc):
    class Predict:
        def __call__(self, p, t, b):
            return flare_flux(p, t, b)
    predict = Predict()
    model = wp.Model(name="loop", predict=predict, parameters=list(TRUTH),
                     default_prior=wp.get_model("flare").default_prior)
    predict.model = model                                    # a cycle: nesting is bounded
    assert len(R.cache_config(lc, model, "abc", seed=0)[1]) == 64


def test_fit_cached_errors_name_the_fix(tmp_path, lc):
    with pytest.raises(TypeError, match="registry name"):
        wp.fit_cached(lc, "flare", object(), tmp_path)
    with pytest.raises(KeyError, match="Available"):
        wp.fit_cached(lc, "flare", "no_such_sampler", tmp_path)


def test_a_directory_of_another_configuration_is_refused(tmp_path, lc, counted):
    res = wp.fit_cached(lc, "flare", "abc", tmp_path, seed=0, **ABC)
    path = next(tmp_path.iterdir())
    m = json.loads((path / R.MANIFEST).read_text())
    m["provenance"]["cache"]["config_hash"] = "0" * 64
    (path / R.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(ValueError, match="another configuration"):
        wp.fit_cached(lc, "flare", "abc", tmp_path, seed=0, **ABC)
    assert res.n_samples > 0


def test_an_interrupted_batch_finishes_only_the_rest(tmp_path, lc, counted, monkeypatch):
    import whisper_cbpf.samplers as S
    counting = S.fit

    def dies_on_the_fourth(*args, **kwargs):
        if len(counted) == 3:
            raise KeyboardInterrupt
        return counting(*args, **kwargs)
    monkeypatch.setattr(S, "fit", dies_on_the_fourth)
    with pytest.raises(KeyboardInterrupt):
        for seed in range(6):
            wp.fit_cached(lc, "flare", "abc", tmp_path, seed=seed, **ABC)
    assert counted == [0, 1, 2] and len(list(tmp_path.iterdir())) == 3
    monkeypatch.setattr(S, "fit", counting)
    for seed in range(6):
        wp.fit_cached(lc, "flare", "abc", tmp_path, seed=seed, **ABC)
    assert counted == [0, 1, 2, 3, 4, 5]
    assert sorted(p.name.startswith(".") for p in tmp_path.iterdir()) == [False] * 6


BATCH = textwrap.dedent("""
    import sys, warnings
    warnings.simplefilter("ignore")
    import numpy as np, whisper_cbpf as wp
    import whisper_cbpf.samplers as S
    from whisper_cbpf.models.flare import flare_flux
    t = np.linspace(0.5, 30.0, 40)
    f = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 40, flux=f, flux_err=np.full(40, 0.1), name="toy")
    real = S.fit
    def fit(*a, **k):
        print("RUN", k["seed"], flush=True)
        return real(*a, **k)
    S.fit = fit
    for seed in range(6):
        wp.fit_cached(lc, "flare", "abc", sys.argv[1], seed=seed, n_simulations=3000,
                      quantile=0.05)
    print("DONE", flush=True)
""")


@pytest.mark.slow
def test_a_killed_batch_resumes(tmp_path):
    """The gate: kill a 6-job batch part-way; the rerun finishes only the rest."""
    script = tmp_path / "batch.py"
    script.write_text(BATCH)
    cache = tmp_path / "cache"
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    proc = subprocess.Popen([sys.executable, str(script), str(cache)], stdout=subprocess.PIPE,
                            text=True, env=env)

    def finished():                     # a ".partial" directory is a save the kill interrupted
        if not cache.exists():
            return []
        return [p for p in cache.iterdir() if not p.name.startswith(".")
                and (p / R.MANIFEST).exists()]
    deadline = time.time() + 120
    while len(finished()) < 2 and proc.poll() is None and time.time() < deadline:
        time.sleep(0.02)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    done = {p.name for p in finished()}
    assert 2 <= len(done) < 6, done
    before = {p.name: (p / R.MANIFEST).stat().st_mtime_ns for p in finished()}
    out = subprocess.run([sys.executable, str(script), str(cache)], capture_output=True, text=True,
                         env=env, timeout=300, check=True).stdout
    ran = [line for line in out.splitlines() if line.startswith("RUN")]
    assert "DONE" in out and len(ran) == 6 - len(done)
    assert len(finished()) == 6
    assert all((cache / n / R.MANIFEST).stat().st_mtime_ns == t for n, t in before.items())
