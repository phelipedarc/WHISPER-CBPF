"""The JAX halves of ``priors``, ``likelihood`` and the one-component kilonova ``Model``.

Three pieces, added together so a gradient sampler can assemble a log-density without a
model-specific hand-written one:

* ``priors.log_prob_jax``        — the JAX twin of ``Prior.log_prob``
* ``likelihood.log_likelihood_jax`` — the JAX twin of ``GaussianLikelihood.log_likelihood``
* ``Model.predict_jax``          — the JAX twin of ``Model.predict``

What these tests are for. A paired backend fails in one of two ways and neither shows up as an
exception: the two halves quietly compute *different numbers* (checked here against the numpy half
in every case, not against a golden), or the JAX half is correct but re-traces on every call, which
is a 400x slowdown that looks exactly like a slow model (``test_predict_jax_compiles_once``; the
factory carried that defect before ``_core`` was jitted -- 65.3 ms per predict against 0.14 ms).
The gradient checks cover the third: a value that is right while its derivative is NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.likelihood import GaussianLikelihood, log_likelihood_jax
from whisper_cbpf.models import register_model
from whisper_cbpf.priors import LogUniform, Prior, Uniform, log_prob_jax

BANDS = ["sdssg", "sdssr", "sdssi"]
Z, DL_CM = 0.00984, 4.3e25          # AT2017gfo
THETA = np.array([0.03, 0.2, 5.0, 3000.0])          # mej, vej, kappa, temperature_floor


# --------------------------------------------------------------------------- Model slots (no JAX)
def test_model_jax_slots_default_to_none():
    m = wp.get_model("bazin")
    assert m.predict_jax is None and m.log_prob_jax is None


def test_register_model_carries_the_jax_slots_through():
    # register_kilonova goes through register_model, so a slot dropped here is a slot the
    # registered model silently does not have.
    sentinel_p, sentinel_l = object(), object()
    m = register_model("_jax_slot_probe", predict=lambda p, t, b: np.zeros_like(t),
                       parameters=["a"], overwrite=True,
                       predict_jax=sentinel_p, log_prob_jax=sentinel_l)
    assert m.predict_jax is sentinel_p and m.log_prob_jax is sentinel_l
    assert wp.get_model("_jax_slot_probe").predict_jax is sentinel_p


# --------------------------------------------------------------------------- priors._jax
@pytest.fixture()
def jnp():
    pytest.importorskip("jax")
    import jax.numpy as _jnp
    return _jnp


def _rel():
    """Tolerance for JAX-vs-numpy parity, at whatever precision the session is running.

    The numpy half is always float64; the JAX half is float64 only when ``jax_enable_x64`` is set,
    which is a session-wide decision no test may make for the rest of the suite. In float32 these
    comparisons are limited by the JAX side's accumulation (~3e-7 relative over a few hundred
    residuals), not by any disagreement about the density.
    """
    import jax

    return 1e-10 if jax.config.jax_enable_x64 else 1e-5


def test_log_prob_jax_matches_the_numpy_prior(jnp):
    prior = Prior({"a": Uniform(0.0, 2.0), "b": LogUniform(1.0, 100.0)})
    f = log_prob_jax(prior, ["a", "b"])
    for a, b in ((0.5, 2.0), (1.9, 99.0), (0.0, 1.0)):      # including both closed edges
        assert float(f(jnp.asarray([a, b]))) == pytest.approx(prior.log_prob({"a": a, "b": b}))


def test_log_prob_jax_is_minus_inf_outside_with_a_finite_gradient(jnp):
    """The value must be -inf, and the DERIVATIVE must not be NaN.

    ``jnp.where`` evaluates both branches, so ``-log(x)`` on the rejected branch differentiates to
    ``-1/x`` at an x that may be zero or negative; ``0 * inf = nan`` then survives the mask into the
    backward pass. ``_jax`` clamps into the support before the log to prevent exactly that.
    """
    import jax

    prior = Prior({"a": Uniform(0.0, 2.0), "b": LogUniform(1.0, 100.0)})
    f = log_prob_jax(prior, ["a", "b"])
    for bad in ([-1.0, 10.0], [1.0, 0.0], [1.0, -5.0], [1e9, 1e9]):
        v = float(f(jnp.asarray(bad)))
        g = np.asarray(jax.grad(f)(jnp.asarray(bad)))
        assert np.isneginf(v), (bad, v)
        assert np.all(np.isfinite(g)), (bad, g)


def test_log_prob_jax_honours_the_names_order(jnp):
    prior = Prior({"a": Uniform(0.0, 1.0), "b": Uniform(10.0, 20.0)})
    assert np.isneginf(float(log_prob_jax(prior, ["b", "a"])(jnp.asarray([0.5, 15.0]))))
    assert np.isfinite(float(log_prob_jax(prior, ["b", "a"])(jnp.asarray([15.0, 0.5]))))


def test_log_prob_jax_refuses_an_unmappable_distribution():
    class Normal:
        bounds = (-1.0, 1.0)

    with pytest.raises(TypeError, match="Normal"):
        log_prob_jax(Prior({"a": Normal()}))


def test_log_prob_jax_reports_a_missing_name():
    with pytest.raises(KeyError, match="nope"):
        log_prob_jax(Prior({"a": Uniform(0.0, 1.0)}), ["a", "nope"])


# --------------------------------------------------------------------------- likelihood._jax
@pytest.fixture()
def lc():
    from pathlib import Path

    path = Path(__file__).parent / "data" / "at2017gfo.csv"
    return wp.load_lightcurve(str(path), explosion_date=57982.0, bands=["g", "r", "i"],
                              redshift=Z).add_flux()


@pytest.mark.parametrize("space", ["flux", "magnitude"])
def test_log_likelihood_jax_matches_the_object_it_was_built_from(lc, jnp, space):
    like = GaussianLikelihood(lc, space=space)
    rng = np.random.default_rng(0)
    flux = np.abs(np.asarray(like.y if space == "flux" else 1e-5 * np.ones(like.y.size))
                  * (1.0 + 0.1 * rng.standard_normal(like.y.size)))
    assert float(log_likelihood_jax(like)(jnp.asarray(flux))) == pytest.approx(
        like.log_likelihood(flux), rel=_rel())


@pytest.mark.parametrize("space", ["flux", "magnitude"])
def test_log_likelihood_jax_scatter_matches_at_every_sigma(lc, jnp, space):
    """The scatter arm takes a SECOND argument, so parity has to be checked along it too.

    ``sigma_extra = 0`` must additionally reproduce the plain Gaussian exactly — that is the
    class's own documented reduction, and it is the one value at which a wrong normaliser hides.
    """
    like = wp.GaussianLikelihoodWithScatter(lc, space=space)
    rng = np.random.default_rng(1)
    flux = np.abs(np.asarray(lc.flux) * (1.0 + 0.1 * rng.standard_normal(like.y.size)))
    f = log_likelihood_jax(like)
    for sigma_extra in (0.0, 0.02, 0.2):
        assert float(f(jnp.asarray(flux), sigma_extra)) == pytest.approx(
            like.log_likelihood(flux, sigma_extra=sigma_extra), rel=_rel())
    assert float(f(jnp.asarray(flux), 0.0)) == pytest.approx(
        GaussianLikelihood(lc, space=space).log_likelihood(flux), rel=_rel())


def _upper_limit_lc(n=24, n_ul=8, seed=0):
    """A flux light curve whose non-detections carry a limiting flux and a NaN error.

    ``tests/data/at2017gfo.csv`` has no ``upper_limit`` column, so the censored case has to be
    constructed. NaN in ``flux_err`` on those rows is what whisper's loader writes for a
    non-detection, and it is exactly what must not reach a plain Gaussian.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(1.0, 20.0, n)
    truth = 3e-5 * np.exp(-t / 8.0)
    ul = np.zeros(n, dtype=bool)
    ul[rng.choice(n, size=n_ul, replace=False)] = True
    return wp.LightCurve(time=t, band=np.array(["r"] * n),
                         flux=np.where(ul, 6e-6, truth + 2e-6 * rng.standard_normal(n)),
                         flux_err=np.where(ul, np.nan, 2e-6), upper_limit=ul, name="ul")


@pytest.mark.parametrize("scale", [1.0, 0.02, 60.0])
def test_log_likelihood_jax_upper_limits_matches_the_object_it_was_built_from(jnp, scale):
    """Flux space only — the magnitude branch does not exist on either backend.

    Three regimes, because the censoring term is the only piece with a non-quadratic shape:
    a model at the data, one far below the limits (``prob -> 1``, log-term -> 0) and one far above
    (``prob`` into the ``_MIN_PROB`` clip).
    """
    ulc = _upper_limit_lc()
    like = wp.GaussianLikelihoodWithUpperLimits(ulc, space="flux", upper_limit_sigma=3.0)
    flux = np.abs(scale * 3e-5 * np.exp(-np.asarray(ulc.time) / 8.0))
    assert float(log_likelihood_jax(like)(jnp.asarray(flux))) == pytest.approx(
        like.log_likelihood(flux), rel=_rel())


@pytest.mark.parametrize("ul", [[False, False, False], [True, True, True]])
def test_log_likelihood_jax_upper_limits_handles_an_empty_half(jnp, ul):
    """All-detections and all-censored both drop one of the two terms; neither may drop both."""
    ulc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[1.0, 0.5, 0.2],
                        flux_err=[np.nan if u else 0.05 for u in ul], upper_limit=ul)
    like = wp.GaussianLikelihoodWithUpperLimits(ulc, space="flux")
    flux = np.array([0.9, 0.55, 0.2])
    assert float(log_likelihood_jax(like)(jnp.asarray(flux))) == pytest.approx(
        like.log_likelihood(flux), rel=_rel())


def test_log_likelihood_jax_refuses_the_mixture_by_name(lc):
    """The one remaining refusal, and it is a decision rather than an omission — so the message
    has to say so, not read as "not implemented yet"."""
    pytest.importorskip("jax")
    with pytest.raises(NotImplementedError, match="MixtureGaussianLikelihood"):
        log_likelihood_jax(wp.MixtureGaussianLikelihood(lc, space="flux"))


def test_log_likelihood_jax_refuses_a_user_subclass_rather_than_scoring_its_base(lc):
    """Dispatch is on the exact type. ``isinstance`` would score this as a plain Gaussian."""
    pytest.importorskip("jax")

    class MyLikelihood(GaussianLikelihood):
        pass

    with pytest.raises(NotImplementedError, match="MyLikelihood"):
        log_likelihood_jax(MyLikelihood(lc, space="flux"))


def test_log_likelihood_jax_advertises_its_scatter_column(lc):
    """``.scatter_param`` is how the adapter tells a consumed sampling column from a flat one."""
    pytest.importorskip("jax")
    assert log_likelihood_jax(GaussianLikelihood(lc, space="flux")).scatter_param is None
    assert log_likelihood_jax(
        wp.GaussianLikelihoodWithScatter(lc, space="flux", scatter_param="s")).scatter_param == "s"


# --------------------------------------------------------------------------- kilonova predict_jax
@pytest.fixture(scope="module")
def kn_model():
    pytest.importorskip("jax")
    return wp.kilonova_model(BANDS, Z, DL_CM, n_wave=400)


def test_predict_jax_reproduces_predict(kn_model, jnp):
    n = 24
    times = np.geomspace(0.5, 15.0, n)
    bands = np.array([BANDS[i % 3] for i in range(n)])
    params = dict(zip(kn_model.parameters, THETA))
    cpu = np.asarray(kn_model.predict(params, times, bands), dtype=float)
    gpu = np.asarray(kn_model.predict_jax(jnp.asarray(THETA), jnp.asarray(times),
                                          jnp.asarray(kn_model.predict_jax.band_index(bands))))
    assert np.all(np.isfinite(cpu)) and np.all(cpu > 0.0)
    assert gpu == pytest.approx(cpu, rel=_rel())


def test_predict_jax_band_index_rejects_an_unbound_band(kn_model):
    with pytest.raises(KeyError, match="not bound"):
        kn_model.predict_jax.band_index(np.array(["sdssg", "lsstz"]))


def test_predict_jax_requires_bands_like_predict_does(kn_model, jnp):
    with pytest.raises(ValueError, match="photometric"):
        kn_model.predict_jax(jnp.asarray(THETA), jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="photometric"):
        kn_model.predict(dict(zip(kn_model.parameters, THETA)), np.array([1.0, 2.0]))


def test_predict_jax_gradients_are_finite_over_the_prior(kn_model, jnp):
    """Every parameter, over 64 draws from the model's own prior -- not one fiducial point."""
    import jax

    times = jnp.asarray(np.geomspace(0.5, 15.0, 24))
    bidx = jnp.asarray(np.arange(24) % 3)
    names = list(kn_model.parameters)
    rng = np.random.default_rng(0)
    draws = np.stack([[kn_model.default_prior.distributions[k].sample(rng) for k in names]
                      for _ in range(64)])

    def loss(theta):
        return jnp.sum(kn_model.predict_jax(theta, times, bidx))

    grads = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(jnp.asarray(draws)))
    bad = ~np.isfinite(grads)
    assert not bad.any(), f"{int(bad.sum())} non-finite gradients at {draws[bad.any(axis=1)][:3]}"


def test_predict_jax_compiles_once(kn_model, jnp):
    """A fresh theta must not re-trace. The factory's ``_core`` is jitted for exactly this."""
    import contextlib
    import io

    import jax

    times = jnp.asarray(np.geomspace(0.5, 15.0, 24))
    bidx = jnp.asarray(np.arange(24) % 3)
    kn_model.predict_jax(jnp.asarray(THETA), times, bidx).block_until_ready()   # the one compile

    rng = np.random.default_rng(1)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), jax.log_compiles(True):
        for _ in range(5):
            theta = THETA * (1.0 + 0.1 * rng.standard_normal(4))
            kn_model.predict_jax(jnp.asarray(theta), times, bidx).block_until_ready()
    assert buf.getvalue().count("Finished XLA compilation") == 0, buf.getvalue()


def test_registered_kilonova_keeps_its_predict_jax(kn_model):
    m = wp.register_kilonova(BANDS, Z, DL_CM, name="_kn_jax_probe", n_wave=400)
    assert m.predict_jax is not None
    assert wp.get_model("_kn_jax_probe").predict_jax is m.predict_jax


def test_the_three_pieces_assemble_into_a_finite_log_posterior(kn_model, lc, jnp):
    """What the pieces exist for: likelihood o predict_jax + prior, value and gradient finite."""
    import jax

    like = GaussianLikelihood(lc, space="flux")
    ll = log_likelihood_jax(like)
    lp = log_prob_jax(kn_model.default_prior, kn_model.parameters)
    t = jnp.asarray(np.asarray(lc.time, dtype=float))
    b = jnp.asarray(kn_model.predict_jax.band_index(
        np.array(["sdss" + str(x) for x in np.asarray(lc.band)])))

    def log_posterior(theta):
        return ll(kn_model.predict_jax(theta, t, b)) + lp(theta)

    value, grad = jax.value_and_grad(log_posterior)(jnp.asarray(THETA))
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))
