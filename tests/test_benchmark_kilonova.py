"""Tests for the kilonova flux-vs-magnitude benchmark (`sanity_check/benchmark_kilonova_modes.py`).

The setup + magnitude-distance tests need no redback; the end-to-end fit+plot is slow and guarded.
"""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_kilonova_modes", os.path.join(HERE, "sanity_check", "benchmark_kilonova_modes.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_benchmark_setup_loads_data_and_prior():
    lc, prior = bench.setup()
    assert lc.n_points == 211
    assert sorted(set(lc.band)) == ["g", "i", "r"]
    assert lc.flux is not None                         # add_flux() ran -> flux-mode ready
    # 6 free ejecta params + 3 pinned (redshift + two temperature floors)
    assert bench.FREE == ["mej_1", "vej_1", "kappa_1", "mej_2", "vej_2", "kappa_2"]
    assert set(prior.names) == set(bench.FREE) | {"temperature_floor_1", "temperature_floor_2", "redshift"}


def test_mag_chi2_distance():
    obs = np.array([1e-4, 2e-4, 5e-4])
    err = np.array([1e-5, 2e-5, 5e-5])
    # identical sim -> zero distance
    assert bench.mag_chi2(obs, err, obs.copy()) == pytest.approx(0.0, abs=1e-9)
    # non-zero, finite, and matches a direct magnitude-space chi-square
    sim = obs * 1.1
    om, sm = -2.5 * np.log10(obs / bench.AB), -2.5 * np.log10(sim / bench.AB)
    me = (2.5 / np.log(10.0)) * err / obs
    assert bench.mag_chi2(obs, err, sim) == pytest.approx(float(np.sum(((om - sm) / me) ** 2)))


@pytest.mark.slow
def test_benchmark_fit_and_plot(tmp_path, monkeypatch):
    """One tiny config end-to-end + the publication report renders (redback + sbi)."""
    pytest.importorskip("redback")
    monkeypatch.setattr(bench, "FIGDIR", str(tmp_path))
    # n_jobs=1: this module is imported via importlib here, so its custom distance can't be re-imported
    # in worker processes (a test-only artifact; the real `__main__` run pickles fine under loky).
    monkeypatch.setitem(bench.CONFIG, "abc",
                        dict(n_simulations=400, quantile=0.05, n_jobs=1, seed=0))
    bench.fit("flux", "abc")
    bench.fit("magnitude", "abc")
    assert os.path.exists(os.path.join(tmp_path, "kilonova_bench_flux_abc.json"))
    bench.plot()
    assert os.path.exists(os.path.join(tmp_path, "kilonova_benchmark_report.png"))
