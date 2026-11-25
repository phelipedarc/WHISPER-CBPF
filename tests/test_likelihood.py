import pickle

import numpy as np
import pytest

from whisper_cbpf import LightCurve
from whisper_cbpf.io.photometry import mag_to_flux_density
from whisper_cbpf.likelihood import (
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


def test_upper_limits_survival_term_is_not_flat_in_the_tail():
    """``0.5*(1 + erf(z/sqrt2))`` saturated at the ``_MIN_PROB`` clip for ``z <~ -8.3``.

    That gave one flat value (-69.0776) with zero gradient over the whole range a sampler has to
    cross to bring an over-bright model back under a limit — measured 26 log units wrong at
    ``z = -8.94``. ``ndtr`` computes the same function through ``erfc`` and has no cancellation.
    """
    lc = LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[np.nan], upper_limit=[True])
    lik = GaussianLikelihoodWithUpperLimits(lc, space="flux", upper_limit_sigma=1.0)
    # sigma_ul = limit / 1 = 1, so model = 1 - z puts the row at exactly z.
    ll = [lik.log_likelihood(np.array([1.0 - z])) for z in (-8.0, -8.5, -9.0, -10.0, -11.0)]
    assert all(a > b for a, b in zip(ll, ll[1:])), f"survival term is flat in the tail: {ll}"
    assert ll[1] == pytest.approx(-39.1974, abs=1e-3)      # z=-8.5: was clipped to -69.0776
    # ... and it still bottoms out at the clip rather than going to -inf.
    assert lik.log_likelihood(np.array([1e6])) == pytest.approx(np.log(1e-30))


def test_upper_limits_refuses_magnitude_space():
    """The magnitude branch was REMOVED, so an explicit ``space="magnitude"`` fails loudly.

    A non-detection is a statement about flux; the magnitude of zero flux is undefined. The old
    branch widened every limit by the Pogson constant instead of by the data, and it had no JAX
    twin — one density, two backends, is the point. ``"auto"`` resolves to flux space for a light
    curve with limits, so the default fit uses them.
    """
    mlc = LightCurve(time=[1.0, 2.0], band=["r", "r"], magnitude=[20.0, 22.0],
                     magnitude_err=[0.1, np.nan], upper_limit=[False, True])
    with pytest.raises(ValueError, match="flux-only"):
        GaussianLikelihoodWithUpperLimits(mlc, space="magnitude")
    assert GaussianLikelihoodWithUpperLimits(mlc, space="auto").space == "flux"
    # ... and the same data IS fittable once it is in flux space.
    lik = GaussianLikelihoodWithUpperLimits(mlc.add_flux(), space="flux")
    assert np.isfinite(lik.log_likelihood(np.array([1e-8, 1e-11])))


def test_make_likelihood_auto_fits_magnitude_upper_limits_in_flux_space():
    """``kind='auto'`` routes magnitude data with non-detections to the censored flux likelihood."""
    mlc = LightCurve(time=[1.0, 2.0], band=["r", "r"], magnitude=[20.0, 22.0],
                     magnitude_err=[0.1, np.nan], upper_limit=[False, True])
    lik = make_likelihood(mlc)
    assert isinstance(lik, GaussianLikelihoodWithUpperLimits) and lik.space == "flux"


def test_gaussian_on_upper_limits_fails_loudly_instead_of_returning_nan():
    """Forcing ``likelihood='gaussian'`` on censored rows used to give a silent NaN fit.

    ``_log_norm`` sums ``log(sigma)`` over every row, so the single NaN error bar of the
    non-detection made ``log_likelihood`` NaN at EVERY parameter value — and every sampler
    completes on that and reports NaN scores as a result.
    """
    lc = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.3], flux_err=[0.1, np.nan],
                    upper_limit=[False, True])
    with pytest.raises(ValueError, match="upper_limit=True"):
        make_likelihood(lc, kind="gaussian")
    # Two layers, and the second one matters on its own: the flag catches the censored rows, but a
    # DETECTION with a missing or zero error bar is just as fatal and carries no flag to catch it.
    bad = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.3], flux_err=[0.1, 0.0])
    with pytest.raises(ValueError, match="not finite and positive"):
        GaussianLikelihood(bad, space="flux")


def test_free_scatter_and_upper_limits_are_refused_as_a_combination():
    """There is no combined scatter + censoring likelihood, so the conflict must not be silent.

    The metrics layer builds ``GaussianLikelihoodWithScatter`` whenever a fit has a scatter column,
    which on a censored light curve used to drop the censoring term without a word — a plain
    scatter fit reported as a fit to data that includes non-detections. The refusal is on the
    ``upper_limit`` FLAG, not on the NaN error bar, so a limit carrying a finite placeholder error
    is caught too.
    """
    from whisper_cbpf.likelihood import GaussianLikelihoodWithScatter

    for err in (np.nan, 0.1):        # NaN limits and finite-placeholder limits alike
        lc = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.3], flux_err=[0.1, err],
                        upper_limit=[False, True])
        with pytest.raises(ValueError, match="mutually exclusive"):
            GaussianLikelihoodWithScatter(lc, space="flux")


def test_upper_limits_tolerates_the_nan_errors_it_is_built_for():
    """The detections-only normaliser is the reason this class survives what the base refuses."""
    lc = LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.3], flux_err=[0.1, np.nan],
                    upper_limit=[False, True])
    lik = GaussianLikelihoodWithUpperLimits(lc, space="flux")
    assert np.isfinite(lik.log_likelihood(np.array([1.0, 0.05])))
    assert np.all(np.isfinite(lik.log_likelihood_pointwise(np.array([1.0, 0.05]))))


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
