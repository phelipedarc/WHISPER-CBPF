"""Tests for the redback-backed ``two_component_kilonova`` model.

The registry/prior tests run without redback; the physics tests are guarded by
``pytest.importorskip("redback")`` so the suite still passes without the optional ``[models]`` extra.
"""
import numpy as np
import pytest

from whisper_cbpf.models import get_model, list_models
from whisper_cbpf.models import two_component_kilonova as tck

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
    """Labels resolve through the package's one map (``synphot.resolve_filter``), as they do for
    every other model. A grouped label names no filter curve, so band integration refuses it;
    up to whisper 0.1.0 this model quietly read ``i-band`` as LSST i."""
    from whisper_cbpf.synphot import resolve_filter

    assert resolve_filter("g") == "lsstg"             # bare letters: LSST, as they always were here
    assert resolve_filter("lsstr") == "lsstr"
    with pytest.raises(ValueError, match="grouped effective-band label"):
        resolve_filter("i-band")
    with pytest.raises(ValueError, match="names no filter"):
        resolve_filter("not_a_band_xyz")


def test_kilonova_uv_nir_band_mapping():
    """UV/NIR bands resolve to redback's sncosmo bandpasses (needs redback's filters table)."""
    pytest.importorskip("redback")
    from whisper_cbpf.synphot import resolve_filter

    for band, sncosmo in [("H", "2massh"), ("J", "2massj"), ("Ks", "2massks"), ("K", "2massks"),
                          ("B", "bessellb"), ("V", "bessellv"), ("U", "bessellux"),
                          ("uvot::uvw1", "uvot::uvw1")]:
        assert resolve_filter(band) == sncosmo, band


def test_kilonova_requires_bands():
    with pytest.raises(ValueError, match="band-dependent"):
        tck.two_component_kilonova_flux(SAMPLE, np.linspace(1, 5, 4), bands=None)


def test_kilonova_without_redback_raises_instead_of_predicting_zero(monkeypatch):
    """REGRESSION: with redback missing, ``predict`` raises ImportError, as in whisper 0.1.0. Once
    the redback call sat inside the adapter's band integral, which reads any exception from the
    model function as refused epochs, it returned 0 Jy at every epoch with a 'gives no flux'
    warning, and a fit on a core install ran to a meaningless answer."""
    from whisper_cbpf.models import redback_adapter as RA

    def missing():
        raise ImportError("No module named 'redback'")

    monkeypatch.setattr(RA, "_import_redback", missing)
    monkeypatch.setattr(tck, "_redback_fn", None)
    with pytest.raises(ImportError, match="requires the optional 'redback' package"):
        get_model("two_component_kilonova").predict(SAMPLE, np.array([1.0, 2.0, 3.0]),
                                                   np.array(["lsstg", "lsstr", "lssti"]))


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


def _fine_band_mag(fn, t, band, step=1.0):
    """``fn``'s flux density integrated over sncosmo's ``band`` on a 1 A trapezoid, as AB mag."""
    import sncosmo

    bp = sncosmo.get_bandpass(band)
    lam = np.arange(bp.wave[0], bp.wave[-1] + 0.5 * step, step)
    w = np.interp(lam, bp.wave, bp.trans, left=0.0, right=0.0) / lam * step
    w[[0, -1]] *= 0.5
    mjy = np.asarray(fn(np.repeat(t, lam.size), output_format="flux_density",
                        frequency=np.tile(2.99792458e18 / lam, t.size), **SAMPLE))
    return -2.5 * np.log10(1e-3 * mjy.reshape(t.size, lam.size) @ w / w.sum() / 3631.0)


def test_kilonova_matches_redback_magnitude():
    """Inside redback's 6-day domain the model is redback's two-component kilonova in the band.

    Two independent references, both redback's: its own ``magnitude`` branch, and the fine band
    integral of its ``flux_density`` branch. The model is two one-component calls,
    each on the one-component grid rather than the two-component one, so they differ by that
    discretisation: 1.3-2.0 mmag measured here. Up to whisper 0.1.0 this was an exact round trip
    of the magnitude branch -- and 99 mag past 6 d.
    """
    pytest.importorskip("redback")
    pytest.importorskip("sncosmo")
    from redback.model_library import all_models_dict
    rb = all_models_dict["two_component_kilonova_model"]
    t = np.linspace(0.5, 5.9, 20)
    flux = get_model("two_component_kilonova").predict(SAMPLE, t, np.array(["i"] * t.size))
    mag_wp = -2.5 * np.log10(flux / 3631.0)
    mag_rb = np.asarray(rb(t, output_format="magnitude", bands=["lssti"], **SAMPLE), dtype=float)
    assert np.nanmax(np.abs(mag_wp - mag_rb)) < 5e-3
    assert np.max(np.abs(mag_wp - _fine_band_mag(rb, t, "lssti"))) < 5e-3


def test_kilonova_is_the_band_integral_of_two_one_component_calls():
    """What the model IS, to the quadrature error: <= 2e-5 mag against a 1 A band integral of
    ``one_component(mej_1, ...) + one_component(mej_2, ...)``, at any epoch, before or after 6 d."""
    pytest.importorskip("redback")
    pytest.importorskip("sncosmo")
    t = np.geomspace(0.3, 20.0, 25)
    for band in ("lsstg", "lssti", "ztfr"):
        got = -2.5 * np.log10(get_model("two_component_kilonova").predict(
            SAMPLE, t, np.array([band] * t.size)) / 3631.0)
        want = _fine_band_mag(tck._two_components_mjy, t, band)
        assert np.max(np.abs(got - want)) < 2e-5, band


def test_kilonova_is_physical_after_six_days():
    """redback's ``two_component_kilonova_model`` stops at 6 d source frame; whisper
    <= 0.1.0 turned the NaN it returned there into 99 mag (flux 4e-37 Jy), which a fit reads as
    "no kilonova" at every late epoch. The late light curve must be the continuation of the early
    one: finite, fading smoothly, a few mag below peak -- not 70 mag below it."""
    pytest.importorskip("redback")
    t = np.array([2.0, 4.0, 5.9, 6.5, 8.0, 10.0, 12.0])
    mag = -2.5 * np.log10(get_model("two_component_kilonova").predict(
        SAMPLE, t, np.array(["lssti"] * t.size)) / 3631.0)
    assert np.all(np.isfinite(mag))
    assert np.all(np.diff(mag) > 0)                      # fading after the peak
    assert abs(mag[3] - mag[2]) < 0.5                    # no jump across 6 d (5.9 -> 6.5 d)
    assert mag[-1] < 30.0


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
    import whisper_cbpf as wp
    from whisper_cbpf.priors import Prior, Uniform

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
