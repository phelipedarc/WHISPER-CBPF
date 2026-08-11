"""Tests for the ``mck19`` model (BBH-in-AGN-disk kicked-hotspot flare, Darc 2025 implementation)."""
import numpy as np
import pytest

import whisper_labia as wp
from whisper_labia.models import get_model, list_models
from whisper_labia.models import mck19

# A representative point from the Darc 2025 grid (the GW-event redshift z=0.28).
TRUE = {"v_kick": 300.0, "M_smbh": 1e8, "M_bh": 80.0, "r_bh": 700.0, "redshift": 0.28}


def _bands(labels, n):
    return np.array([lab for lab in labels for _ in range(n)])


def test_mck19_registered():
    assert "mck19" in list_models()
    m = get_model("mck19")
    assert m.parameters == ["v_kick", "M_smbh", "M_bh", "r_bh", "redshift"]
    # the default prior covers exactly the model parameters
    assert set(m.default_prior.names) == set(m.parameters)


def test_mck19_flux_finite_and_positive():
    m = get_model("mck19")
    t = np.linspace(0, 120, 200)
    flux = m.predict(TRUE, t, np.array(["r"] * t.size))
    assert flux.shape == t.shape
    assert np.all(np.isfinite(flux))
    assert np.all(flux > 0)


def test_mck19_is_band_dependent():
    """Distinct effective wavelengths -> distinct blackbody flux per band."""
    m = get_model("mck19")
    t = np.full(5, 6.0)  # near the peak
    bands = np.array(["u", "g", "r", "i", "z"])
    flux = m.predict(TRUE, t, bands)
    assert len(set(np.round(flux, 30))) == 5  # all five bands differ


def test_mck19_flare_peaks_near_t_ram():
    """The flare brightens to a maximum at the observer-frame ram-pressure delay."""
    m = get_model("mck19")
    t = np.linspace(0, 60, 600)
    flux = m.predict(TRUE, t, np.array(["g"] * t.size))
    t_peak_model = t[np.argmax(flux)]
    t_ram = mck19._t_ram_days(TRUE["M_bh"], TRUE["v_kick"], TRUE["redshift"])
    assert abs(t_peak_model - t_ram) < 1.0
    # and the peak is a real brightening over the pre-flare baseline
    assert flux.max() > 10 * flux[0]


def test_mck19_higher_redshift_is_fainter():
    """A more distant source (larger z -> larger luminosity distance) is fainter at peak."""
    m = get_model("mck19")
    t = np.full(1, 6.0)
    near = dict(TRUE, redshift=0.1)
    far = dict(TRUE, redshift=0.6)
    assert m.predict(near, t, np.array(["g"]))[0] > m.predict(far, t, np.array(["g"]))[0]


def test_mck19_no_bands_warns_and_runs():
    m = get_model("mck19")
    t = np.linspace(0, 30, 10)
    with pytest.warns(UserWarning, match="band-dependent"):
        flux = m.predict(TRUE, t, None)
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)


@pytest.mark.slow
def test_mck19_mcmc_recovers_and_is_magnitude_space():
    """End-to-end: fit synthetic mck19 magnitude data; the shared likelihood runs in magnitude space
    (the model predicts flux, the data is magnitude) and the posterior recovers the truth."""
    m = get_model("mck19")
    rng = np.random.default_rng(0)
    t = np.concatenate([np.linspace(0, 40, 18)] * 3)
    b = _bands(["g", "r", "i"], 18)
    mag = -2.5 * np.log10(m.predict(TRUE, t, b) / 3631.0)
    mag_obs = mag + rng.normal(0, 0.05, mag.shape)
    lc = wp.LightCurve(time=t, band=b, magnitude=mag_obs, magnitude_err=np.full_like(mag, 0.05),
                       redshift=0.28, data_mode="magnitude", name="synthetic_mck19")

    res = wp.fit_MCMC(lc, "mck19", nsteps=1500, burnin=500, thin=5, seed=0)

    assert res.info["space"] == "magnitude"          # data-mode-consistent likelihood
    assert np.isfinite(res.aic) and np.isfinite(res.bic)
    # timing + normalisation parameters recovered to ~15%
    for key, tol in [("v_kick", 0.20), ("M_bh", 0.25), ("r_bh", 0.20), ("redshift", 0.20)]:
        assert abs(res.summary[key]["median"] - TRUE[key]) / TRUE[key] < tol
    best = {k: res.summary[k]["median"] for k in TRUE}
    pmag = -2.5 * np.log10(m.predict(best, t, b) / 3631.0)
    assert np.sqrt(np.mean((pmag - mag_obs) ** 2)) < 0.1  # fits to the noise floor
