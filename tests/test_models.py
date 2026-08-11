import numpy as np
import pytest

from whisper_labia import Prior, Uniform, get_model, list_models, register_model
from whisper_labia.models.bazin import bazin_flux
from whisper_labia.models.flare import flare_flux
from whisper_labia.models.gaussian_rise import gaussian_rise_flux


def test_flare_registered():
    assert "flare" in list_models()
    m = get_model("flare")
    assert m.parameters == ["amplitude", "rise_time", "decay_time"]
    assert m.default_prior is not None


def test_flare_flux_vectorized():
    t = np.array([0.0, 1.0, 5.0, 10.0])
    f = flare_flux({"amplitude": 2.0, "rise_time": 2.0, "decay_time": 10.0}, t)
    assert f.shape == t.shape
    assert np.isclose(f[0], 0.0)     # (1 - exp(0)) == 0 at t=0
    assert np.all(f[1:] > 0)


def test_register_custom_model():
    def quad(params, times, bands=None):
        return params["a"] * np.asarray(times, float) ** 2

    register_model("quad_test", quad, ["a"], prior=Prior({"a": Uniform(0, 1)}), overwrite=True)
    assert "quad_test" in list_models()
    assert np.allclose(get_model("quad_test")({"a": 2.0}, [1, 2, 3]), [2, 8, 18])


def test_bazin_registered_and_finite():
    assert "bazin" in list_models()
    t = np.linspace(-5, 40, 60)
    f = bazin_flux({"amplitude": 5.0, "t0": 2.0, "tau_rise": 3.0, "tau_fall": 15.0}, t)
    assert f.shape == t.shape and np.all(np.isfinite(f)) and np.all(f >= 0) and f.max() > 0


def test_gaussian_rise_registered_and_peak():
    assert "gaussian_rise" in list_models()
    t = np.array([0.0, 5.0, 10.0])     # t0=5 -> peak in the middle
    f = gaussian_rise_flux({"amplitude": 3.0, "t0": 5.0, "sigma_rise": 2.0, "tau_decay": 10.0}, t)
    assert np.isclose(f[1], 3.0)       # at t0, flux == amplitude
    assert f[0] < f[1] and f[2] < f[1]  # rises to peak, then decays


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("does_not_exist")


def test_flare_zero_before_explosion():
    f = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0},
                   np.array([-2.0, -0.5, 0.0, 1.0]))
    assert f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0 and f[3] > 0   # no pre-explosion emission


def test_bazin_vanishes_far_before_peak():
    # tau_rise < tau_fall must decay toward 0 far before the peak (not plateau at amplitude).
    f = bazin_flux({"amplitude": 5.0, "t0": 0.0, "tau_rise": 1.0, "tau_fall": 20.0},
                   np.array([0.0, -1000.0]))
    assert f[0] > 0 and f[1] < 1e-3
