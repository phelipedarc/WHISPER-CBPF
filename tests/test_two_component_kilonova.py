"""Tests for the redback-backed ``two_component_kilonova`` model.

The registry/prior tests run without redback; the physics tests are guarded by
``pytest.importorskip("redback")`` so the suite still passes without the optional ``[models]`` extra.
"""
import numpy as np
import pytest

from whisper_labia.models import get_model, list_models
from whisper_labia.models import two_component_kilonova as tck

PARAMS = ["mej_1", "vej_1", "kappa_1", "temperature_floor_1",
          "mej_2", "vej_2", "kappa_2", "temperature_floor_2", "redshift"]
SAMPLE = {"mej_1": 0.04, "vej_1": 0.25, "kappa_1": 0.3, "temperature_floor_1": 2500.0,
          "mej_2": 0.02, "vej_2": 0.15, "kappa_2": 10.0, "temperature_floor_2": 3000.0,
          "redshift": 0.00984}


# --- registry / prior (no redback needed) ---------------------------------------------------------

def test_kilonova_registered_without_redback():
    """The model registers (and the package imports) even when redback is absent — it's lazy."""
    assert "two_component_kilonova" in list_models()
    m = get_model("two_component_kilonova")
    assert m.parameters == PARAMS
    assert set(m.default_prior.names) == set(PARAMS)


def test_kilonova_band_mapping():
    assert tck._redback_band("g") == "lsstg"
    assert tck._redback_band("lsstr") == "lsstr"      # already a redback name
    assert tck._redback_band("i-band") == "lssti"     # WHISPER effective-band label
    with pytest.raises(ValueError, match="not a recognised redback filter"):
        tck._redback_band("not_a_band_xyz")


def test_kilonova_uv_nir_band_mapping():
    """UV/NIR bands resolve to redback's sncosmo bandpasses (needs redback's filters table)."""
    pytest.importorskip("redback")
    for band, sncosmo in [("H", "2massh"), ("J", "2massj"), ("Ks", "2massks"), ("K", "2massks"),
                          ("B", "bessellb"), ("V", "bessellv"), ("U", "bessellux"),
                          ("uvot::uvw1", "uvot::uvw1")]:
        assert tck._redback_band(band) == sncosmo, band


def test_kilonova_requires_bands():
    with pytest.raises(ValueError, match="band-dependent"):
        tck.two_component_kilonova_flux(SAMPLE, np.linspace(1, 5, 4), bands=None)


# --- physics (needs the redback backend) ----------------------------------------------------------

def test_kilonova_flux_finite_and_positive():
    pytest.importorskip("redback")
    m = get_model("two_component_kilonova")
    t = np.linspace(0.5, 10, 25)
    flux = m.predict(SAMPLE, t, np.array(["r"] * t.size))
    assert flux.shape == t.shape
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)


def test_kilonova_is_band_dependent():
    pytest.importorskip("redback")
    m = get_model("two_component_kilonova")
    t = np.full(4, 3.0)
    fg = m.predict(SAMPLE, t, np.array(["g"] * 4))
    fr = m.predict(SAMPLE, t, np.array(["r"] * 4))
    fi = m.predict(SAMPLE, t, np.array(["i"] * 4))
    assert not np.allclose(fg, fr) and not np.allclose(fr, fi)


def test_kilonova_matches_redback_magnitude():
    """WHISPER's flux is an exact round-trip of redback's own AB magnitude."""
    pytest.importorskip("redback")
    from redback.model_library import all_models_dict
    rb = all_models_dict["two_component_kilonova_model"]
    t = np.linspace(0.5, 10, 20)
    flux = get_model("two_component_kilonova").predict(SAMPLE, t, np.array(["i"] * t.size))
    mag_wp = -2.5 * np.log10(flux / 3631.0)
    mag_rb = np.asarray(rb(t, output_format="magnitude", bands=["lssti"], **SAMPLE), dtype=float)
    assert np.nanmax(np.abs(mag_wp - mag_rb)) < 1e-6


def test_kilonova_mixed_band_predict():
    pytest.importorskip("redback")
    m = get_model("two_component_kilonova")
    out = m.predict(SAMPLE, np.array([3.0, 3.0, 3.0]), np.array(["g", "r", "i"]))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out)) and np.all(out > 0)


@pytest.mark.slow
def test_kilonova_snpe_recovers():
    """End-to-end: SNPE recovers a synthetic two_component_kilonova injection (redback + sbi)."""
    pytest.importorskip("redback")
    pytest.importorskip("sbi")
    import whisper_labia as wp
    from whisper_labia.priors import Prior, Uniform

    m = get_model("two_component_kilonova")
    truth = dict(SAMPLE, temperature_floor_1=2500.0, temperature_floor_2=2500.0)
    t = np.concatenate([np.linspace(0.5, 8, 12)] * 2)
    b = np.array(["g"] * 12 + ["r"] * 12)
    mag = -2.5 * np.log10(m.predict(truth, t, b) / 3631.0)
    lc = wp.LightCurve(time=t, band=b, magnitude=mag, magnitude_err=np.full_like(mag, 0.05),
                       redshift=0.00984, data_mode="magnitude", name="syn_kn").add_flux()
    prior = Prior({
        "mej_1": Uniform(1e-4, 0.1), "vej_1": Uniform(0.01, 0.7), "kappa_1": Uniform(0.1, 0.5),
        "mej_2": Uniform(1e-4, 0.1), "vej_2": Uniform(0.01, 0.7), "kappa_2": Uniform(1.0, 30.0),
        "temperature_floor_1": Uniform(2499.0, 2501.0), "temperature_floor_2": Uniform(2499.0, 2501.0),
        "redshift": Uniform(0.00983, 0.00985),
    })
    res = wp.fit_SNPE(lc, "two_component_kilonova", prior=prior, num_rounds=1,
                      num_simulations=1500, num_samples=2000, space="flux", seed=0)
    assert np.isfinite(res.aic)
    best = {k: res.best_params[k] for k in truth}
    pmag = -2.5 * np.log10(m.predict(best, t, b) / 3631.0)
    assert np.sqrt(np.nanmean((pmag - mag) ** 2)) < 0.5   # fits the injection
