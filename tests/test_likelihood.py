import pickle

import numpy as np

from whisper_labia import LightCurve
from whisper_labia.io.photometry import mag_to_flux_density
from whisper_labia.likelihood import (
    GaussianLikelihood,
    GaussianLikelihoodWithUpperLimits,
    MixtureGaussianLikelihood,
    make_likelihood,
)


def test_gaussian_flux_matches_manual():
    lc = LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3,
                    flux=[1.0, 0.5, 0.2], flux_err=[0.1, 0.05, 0.02])
    lik = GaussianLikelihood(lc, space="flux")
    model = np.array([0.9, 0.55, 0.2])
    res = (np.array([1.0, 0.5, 0.2]) - model) / np.array([0.1, 0.05, 0.02])
    expected = -0.5 * np.sum(res ** 2 + np.log(2 * np.pi * np.array([0.1, 0.05, 0.02]) ** 2))
    assert lik.space == "flux"
    assert np.isclose(lik.log_likelihood(model), expected)


def test_zero_residual_is_max():
    lc = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 2.0], flux_err=[0.1, 0.1])
    lik = GaussianLikelihood(lc, space="flux")
    assert lik.log_likelihood([1.0, 2.0]) > lik.log_likelihood([1.2, 2.0])


def test_space_auto():
    mlc = LightCurve(time=[1.0, 2.0], band=["r", "r"], magnitude=[20.0, 21.0], magnitude_err=[0.1, 0.1])
    flc = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.5], flux_err=[0.1, 0.05])
    assert GaussianLikelihood(mlc, space="auto").space == "magnitude"
    assert GaussianLikelihood(flc, space="auto").space == "flux"


def test_magnitude_space_uses_mag_residuals():
    mlc = LightCurve(time=[1.0, 2.0], band=["r", "r"], magnitude=[20.0, 21.0], magnitude_err=[0.05, 0.05])
    lik = GaussianLikelihood(mlc, space="magnitude")
    model_flux = mag_to_flux_density(np.array([20.0, 21.0]))     # converts back to mag [20, 21]
    assert lik.log_likelihood(model_flux) > lik.log_likelihood(model_flux * 2)


def test_upper_limits_flux_penalize_bright_model():
    lc = LightCurve(time=[1.0, 2.0, 3.0, 4.0], band=["r"] * 4,
                    flux=[1.0, 0.5, 0.2, 0.3], flux_err=[0.1, 0.05, 0.02, np.nan],
                    upper_limit=[False, False, False, True])
    lik = GaussianLikelihoodWithUpperLimits(lc, space="flux", upper_limit_sigma=3.0)
    det = [1.0, 0.5, 0.2]
    ll_faint = lik.log_likelihood(np.array(det + [0.05]))    # below the limit -> consistent
    ll_bright = lik.log_likelihood(np.array(det + [1.0]))    # above the limit -> penalized
    assert ll_faint > ll_bright


def test_upper_limits_detection_only_equals_gaussian():
    lc = LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[1.0, 0.5, 0.2],
                    flux_err=[0.1, 0.05, 0.02], upper_limit=[False, False, False])
    m = np.array([0.9, 0.55, 0.2])
    assert np.isclose(GaussianLikelihood(lc, space="flux").log_likelihood(m),
                      GaussianLikelihoodWithUpperLimits(lc, space="flux").log_likelihood(m))


def test_make_likelihood_auto_picks_upper_limits():
    lc = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.3], flux_err=[0.1, np.nan],
                    upper_limit=[False, True])
    assert isinstance(make_likelihood(lc), GaussianLikelihoodWithUpperLimits)


def test_mixture_tolerates_outlier():
    lc = LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[1.0, 1.0, 1.0], flux_err=[0.1, 0.1, 0.1])
    model_outlier = np.array([1.0, 1.0, 5.0])     # third point a gross outlier
    gauss = GaussianLikelihood(lc, space="flux").log_likelihood(model_outlier)
    mix = MixtureGaussianLikelihood(lc, space="flux", alpha=0.9, sigma_out_scale=10).log_likelihood(model_outlier)
    assert mix > gauss


def test_likelihood_picklable():
    lc = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.5], flux_err=[0.1, 0.05])
    lik = GaussianLikelihood(lc, space="flux")
    lik2 = pickle.loads(pickle.dumps(lik))
    assert np.isclose(lik2.log_likelihood([1.0, 0.5]), lik.log_likelihood([1.0, 0.5]))
