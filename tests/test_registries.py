"""Discoverability/extensibility: likelihood, distance, and manual-band registries."""
import numpy as np
import pytest

import whisper_labia as wp
from whisper_labia.io import svo


def test_list_likelihoods_and_register():
    names = wp.list_likelihoods()
    assert "gaussian" in names and "mixture" in names
    wp.register_likelihood("my_gauss", wp.GaussianLikelihood, overwrite=True)
    assert "my_gauss" in wp.list_likelihoods()
    lc = wp.LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.5], flux_err=[0.1, 0.05])
    lik = wp.make_likelihood(lc, kind="my_gauss", space="flux")
    assert isinstance(lik, wp.GaussianLikelihood)
    with pytest.raises(ValueError, match="already registered"):
        wp.register_likelihood("gaussian", wp.GaussianLikelihood)


def test_list_distances_register_and_get():
    assert "chi2" in wp.list_distances()
    assert wp.get_distance("chi2") is wp.chi2_distance
    assert wp.get_distance(wp.chi2_distance) is wp.chi2_distance        # callable passthrough

    def absdist(obs, err, sim, bands=None):
        return float(np.sum(np.abs((np.asarray(obs) - np.asarray(sim)) / np.asarray(err))))

    wp.register_distance("abs", absdist, overwrite=True)
    assert "abs" in wp.list_distances() and wp.get_distance("abs") is absdist
    with pytest.raises(KeyError, match="Unknown distance"):
        wp.get_distance("nope")


def test_distance_by_name_in_fit():
    m = wp.get_model("flare") if hasattr(wp, "get_model") else None
    from whisper_labia.models import get_model
    t = np.linspace(0.1, 10, 20)
    f = get_model("flare").predict({"amplitude": 1.0, "rise_time": 1.0, "decay_time": 3.0}, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 20, flux=f, flux_err=np.full_like(f, 0.05))
    prior = wp.Prior({"amplitude": wp.Uniform(0, 5), "rise_time": wp.Uniform(0.1, 5),
                      "decay_time": wp.Uniform(0.5, 10)})
    res = wp.fit_ABC(lc, "flare", prior=prior, n_simulations=500, distance="chi2", n_jobs=1, seed=0)
    assert res.sampler == "abc" and np.isfinite(res.aic)


def test_manual_band_register_and_undo():
    svo.clear_manual_bands()
    wp.register_manual_band("zzz_band", 5000.0, 3631.0)
    r = wp.resolve_band("zzz_band")
    assert r["source"] == "manual" and r["lambda_eff"] == 5000.0
    wp.unregister_manual_band("zzz_band")
    assert "zzz_band" not in svo._MANUAL_BANDS
    wp.register_manual_band("a", 1.0, 1.0)
    wp.register_manual_band("b", 2.0, 2.0)
    wp.clear_manual_bands()
    assert svo._MANUAL_BANDS == {}
