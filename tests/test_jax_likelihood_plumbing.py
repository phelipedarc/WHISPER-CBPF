"""``likelihood=`` from a GPU-family sampler down to the JAX likelihood, and back.

Three things are checked here, and each of them used to be reachable only by reading the source:

* the auto-built JAX density can now be something other than a plain Gaussian — ``upper_limits``
  and ``gaussian_scatter`` are ported, ``mixture`` is refused *on purpose* and the message says so;
* a sampled column that is not a model parameter is admitted **only** when the resolved likelihood
  consumes it. The refusal was unconditional, and correctly so: without the scatter arm the density
  is exactly constant in that column and the sampler reports the prior as its marginal;
* the resolved class reaches ``result.info["likelihood"]``, which is what ``waic`` and
  ``predictive_metrics`` read back to re-score a fit under the density it was fitted with.

CPU-JAX is enough. Nothing here touches the global ``jax_enable_x64`` flag — ``_rel()`` reports the
tolerance the current session's precision supports instead, so the file behaves the same whether or
not something earlier in the suite turned x64 on.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.likelihood import (  # noqa: E402
    GaussianLikelihoodWithScatter,
    GaussianLikelihoodWithUpperLimits,
)
from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax  # noqa: E402


def _rel():
    """JAX-vs-numpy tolerance at whatever precision this session is running.

    ``tests/t2_autodiff/`` turns ``jax_enable_x64`` on globally and irreversibly, so a pinned 1e-10
    would assert the collection order rather than the density.
    """
    return 1e-10 if jax.config.jax_enable_x64 else 2e-6


# --------------------------------------------------------------------------------- fixtures
def _toy_model(name="likelihood_plumbing_toy"):
    """A 2-parameter analytic model with both ``predict`` and ``predict_jax``.

    Band-independent, so ``predict_jax`` carries no ``.band_index`` and the whole fit stays fast
    enough to run unmarked.
    """
    def predict(parameters, times, bands=None):
        t = np.asarray(times, dtype=float)
        return (np.asarray(parameters["amp"]) * np.exp(-t / np.asarray(parameters["tau"]))
                + 1e-7)

    def predict_jax(theta, times, band_idx=None):
        return theta[0] * jnp.exp(-jnp.asarray(times) / theta[1]) + 1e-7

    return wp.register_model(
        name, predict, ["amp", "tau"], overwrite=True, predict_jax=predict_jax,
        prior=wp.Prior({"amp": wp.Uniform(1e-6, 1e-4), "tau": wp.Uniform(2.0, 30.0)}))


@pytest.fixture()
def toy_model():
    return _toy_model()


@pytest.fixture()
def ul_lc():
    """A flux light curve carrying real non-detections.

    ``tests/data/at2017gfo.csv`` has no ``upper_limit`` column, so the censored case has to be
    built. NaN in ``flux_err`` on those rows is what whisper's loader writes for a non-detection.
    """
    rng = np.random.default_rng(0)
    n, t = 24, np.linspace(1.0, 20.0, 24)
    truth = 3e-5 * np.exp(-t / 8.0)
    ul = np.zeros(n, dtype=bool)
    ul[rng.choice(n, size=8, replace=False)] = True
    return wp.LightCurve(time=t, band=np.array(["r"] * n),
                         flux=np.where(ul, 6e-6, truth + 2e-6 * rng.standard_normal(n)),
                         flux_err=np.where(ul, np.nan, 2e-6), upper_limit=ul, name="ul")


@pytest.fixture()
def plain_lc():
    rng = np.random.default_rng(0)
    t = np.linspace(1.0, 20.0, 24)
    truth = 3e-5 * np.exp(-t / 8.0)
    return wp.LightCurve(time=t, band=np.array(["r"] * 24),
                         flux=truth + 2e-6 * rng.standard_normal(24),
                         flux_err=np.full(24, 2e-6), name="plain")


# ------------------------------------------------------------------- make_log_prob_jax: selection
def test_auto_density_on_a_censored_light_curve_is_the_upper_limit_density(ul_lc, toy_model):
    """``kind='auto'`` picks the censoring model, and the JAX arm now implements it.

    Before this change the same call raised ``NotImplementedError``: every light curve with a
    non-detection was closed to the whole gradient-sampler family.
    """
    f = make_log_prob_jax(ul_lc, toy_model, space="flux")
    assert f.likelihood == "GaussianLikelihoodWithUpperLimits"
    theta = jnp.asarray([3e-5, 8.0])
    expected = GaussianLikelihoodWithUpperLimits(ul_lc, space="flux").log_likelihood(
        toy_model.predict({"amp": 3e-5, "tau": 8.0}, ul_lc.time, ul_lc.band))
    assert float(f(theta)) == pytest.approx(expected, rel=_rel())


def test_the_censoring_term_is_differentiable_and_actually_censors(ul_lc, toy_model):
    """A model brighter than the limits must score worse, with a finite gradient pushing it down.

    A survival term that is flat (or NaN) in the model is the failure that leaves NUTS wandering:
    the value can look right while the trajectory carries no information from the non-detections.
    """
    f = make_log_prob_jax(ul_lc, toy_model, space="flux")
    faint, bright = jnp.asarray([3e-5, 8.0]), jnp.asarray([9e-5, 25.0])
    assert float(f(faint)) > float(f(bright))
    g = np.asarray(jax.grad(f)(bright))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def test_forcing_gaussian_on_a_censored_light_curve_is_refused_not_nan(ul_lc, toy_model):
    """The NaN error bars of the non-detections used to flow straight into ``_log_norm``."""
    with pytest.raises(ValueError, match="upper_limit=True"):
        make_log_prob_jax(ul_lc, toy_model, space="flux", likelihood="gaussian")


def test_mixture_is_refused_as_a_decision_not_an_omission(plain_lc, toy_model):
    with pytest.raises(NotImplementedError, match="MixtureGaussianLikelihood"):
        make_log_prob_jax(plain_lc, toy_model, space="flux", likelihood="mixture")


def test_likelihood_accepts_a_prebuilt_object_as_well_as_a_kind(plain_lc, toy_model):
    like = GaussianLikelihoodWithScatter(plain_lc, space="flux", scatter_param="s")
    prior = wp.Prior({"amp": wp.Uniform(1e-6, 1e-4), "tau": wp.Uniform(2.0, 30.0),
                      "s": wp.Uniform(1e-8, 1e-4)})
    f = make_log_prob_jax(plain_lc, toy_model, prior, space="flux",
                          names=["amp", "tau", "s"], likelihood=like)
    assert f.likelihood == "GaussianLikelihoodWithScatter" and f.scatter_param == "s"


# --------------------------------------------------------------- make_log_prob_jax: scatter column
@pytest.fixture()
def scatter_setup(plain_lc, toy_model):
    prior = wp.Prior({"amp": wp.Uniform(1e-6, 1e-4), "tau": wp.Uniform(2.0, 30.0),
                      "sigma": wp.Uniform(1e-8, 1e-4)})
    f = make_log_prob_jax(plain_lc, toy_model, prior, space="flux",
                          names=["amp", "tau", "sigma"], likelihood="gaussian_scatter")
    return plain_lc, toy_model, prior, f


@pytest.mark.parametrize("sigma_extra", [2e-6, 1e-5, 4e-5])
def test_the_scatter_column_reaches_the_likelihood(scatter_setup, sigma_extra):
    """logL must MOVE with the scatter column, and match the numpy twin at every value.

    This is the measurement the refusal comment recorded: with a plain Gaussian the density was
    byte-identical at every sigma with a gradient of exactly 0.0, so the sampler converged and
    reported the *prior* as sigma's marginal, labelled as a fit.
    """
    lc, model, _, f = scatter_setup
    flux = model.predict({"amp": 3e-5, "tau": 8.0}, lc.time, lc.band)
    expected = GaussianLikelihoodWithScatter(lc, space="flux").log_likelihood(
        flux, sigma_extra=sigma_extra)
    assert float(f(jnp.asarray([3e-5, 8.0, sigma_extra]))) == pytest.approx(expected, rel=_rel())


def test_the_scatter_column_has_a_non_zero_gradient(scatter_setup):
    lc, model, _, f = scatter_setup
    values = [2e-6, 1e-5, 4e-5]
    ll = [float(f(jnp.asarray([3e-5, 8.0, s]))) for s in values]
    grads = [float(np.asarray(jax.grad(f)(jnp.asarray([3e-5, 8.0, s])))[2]) for s in values]
    assert len(set(ll)) == len(ll), f"density is constant in the scatter column: {ll}"
    assert all(np.isfinite(g) and g != 0.0 for g in grads), grads


def test_a_scatter_column_without_the_scatter_likelihood_is_still_refused(scatter_setup):
    """The pre-existing refusal, unchanged for every column the likelihood does not consume."""
    lc, model, prior, _ = scatter_setup
    with pytest.raises(ValueError, match="sigma"):
        make_log_prob_jax(lc, model, prior, space="flux", names=["amp", "tau", "sigma"],
                          likelihood="gaussian")


def test_the_scatter_likelihood_without_its_column_is_refused(plain_lc, toy_model):
    """Silently evaluating at ``sigma_extra = 0`` is a plain Gaussian reported as a scatter fit."""
    with pytest.raises(ValueError, match="sigma"):
        make_log_prob_jax(plain_lc, toy_model, space="flux", likelihood="gaussian_scatter")


# --------------------------------------------------------------------------- end to end, a sampler
def test_emcee_jax_runs_end_to_end_with_upper_limits(ul_lc, toy_model):
    """A GPU-family sampler, on a light curve with real non-detections, start to finish.

    ``emcee_jax`` is the cheap member of the family that consumes the same auto-built density as
    ``nuts_gpu``/``pymc_gpu``, so this covers the plumbing without paying for NUTS warmup.
    """
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    res = fit_emcee_jax(ul_lc, toy_model, nwalkers=8, nsteps=80, burnin=20, thin=2, seed=0,
                        space="flux", likelihood="upper_limits", walker_chunk=None)
    assert res.info["likelihood"] == "GaussianLikelihoodWithUpperLimits"
    assert res.info["space"] == "flux" and res.info["log_prob_fn"] == "auto"
    assert np.isfinite(res.max_log_likelihood) and np.isfinite(res.aic)
    assert list(res.samples.columns) == ["amp", "tau"] and len(res.samples) > 0
    # The recorded kind is the point of the plumbing: it is what `attach_predictive_metrics` uses
    # to re-score the fit under the density it was fitted with.
    from whisper_cbpf.samplers.base import _LIKELIHOOD_KINDS
    assert _LIKELIHOOD_KINDS[res.info["likelihood"]] == "gaussian_upper_limits"


def test_a_censored_fit_comes_back_with_band_metrics(ul_lc, toy_model):
    """`attach_band_metrics` has to SUCCEED here, not warn and record why it could not.

    `per_band_metrics` used to build a `GaussianLikelihood` purely as a space converter, and that
    class refuses a censored light curve — so on this exact configuration (24 flux points, 8
    non-detections, NaN `flux_err`) every GPU-family fit landed `info['band_metrics_error']`
    instead of `info['band_metrics']`. The fit was never affected; the per-band MSE/MAE was simply
    missing from the result the user saves.
    """
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    res = fit_emcee_jax(ul_lc, toy_model, nwalkers=8, nsteps=80, burnin=20, thin=2, seed=0,
                        space="flux", likelihood="upper_limits", walker_chunk=None)
    assert "band_metrics_error" not in res.info
    bm = res.info["band_metrics"]

    n_ul = int(np.sum(np.asarray(ul_lc.upper_limit, dtype=bool)))
    assert bm["n_upper_limits_excluded"] == n_ul
    assert bm["overall"]["n"] == ul_lc.n_points - n_ul                  # detections only
    assert np.isfinite([bm["overall"][k] for k in ("mse", "rmse", "mae")]).all()
    assert bm["space"] == "flux" and bm["unit"] == "Jy"


def test_emcee_jax_samples_the_scatter_column_when_asked_for_it(plain_lc, toy_model):
    """``likelihood='gaussian_scatter'`` has to GROW the sampled vector by one column.

    Without that the sampler builds the scatter density and never samples the parameter it exists
    to fit, so ``sigma_extra`` stays at its default 0 and the run is a plain Gaussian fit wearing a
    scatter label.
    """
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    prior = wp.Prior({"amp": wp.Uniform(1e-6, 1e-4), "tau": wp.Uniform(2.0, 30.0),
                      "sigma": wp.Uniform(1e-8, 1e-4)})
    res = fit_emcee_jax(plain_lc, toy_model, prior=prior, nwalkers=8, nsteps=80, burnin=20,
                        thin=2, seed=0, space="flux", likelihood="gaussian_scatter",
                        walker_chunk=None)
    assert res.info["likelihood"] == "GaussianLikelihoodWithScatter"
    assert res.info["scatter_param"] == "sigma"
    assert list(res.samples.columns) == ["amp", "tau", "sigma"]
    assert res.samples["sigma"].std() > 0.0
    assert np.isfinite(res.max_log_likelihood)


def test_a_caller_supplied_density_records_no_likelihood(plain_lc, toy_model):
    """``info["likelihood"]`` must stay ``None`` when we did not choose the density.

    Reporting a guess there would make ``attach_predictive_metrics`` re-score someone's hand-built
    density under a likelihood nobody asked for.
    """
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    lows = np.array([1e-6, 2.0])
    highs = np.array([1e-4, 30.0])

    def log_prob_fn(theta):
        flux = toy_model.predict_jax(theta, jnp.asarray(np.asarray(plain_lc.time, dtype=float)))
        res = (jnp.asarray(np.asarray(plain_lc.flux, dtype=float)) - flux) / 2e-6
        inside = jnp.all((theta >= jnp.asarray(lows)) & (theta <= jnp.asarray(highs)))
        return jnp.where(inside, -0.5 * jnp.sum(res * res), -jnp.inf)

    res = fit_emcee_jax(plain_lc, toy_model, log_prob_fn, nwalkers=8, nsteps=40, burnin=10,
                        thin=2, seed=0, space="flux", walker_chunk=None)
    assert res.info["likelihood"] is None and res.info["log_prob_fn"] == "caller"
