"""Priors beyond boxes: ``Normal``, ``TruncatedNormal`` and ``Fixed``, on the CPU and the GPU paths.

The JAX prior layer took only Uniform and LogUniform (``priors/_jax.py``, ``_adapters._box_bounds``,
``nuts_gpu._numpyro_priors``), so a redshift from a host spectrum or an explosion epoch from a
last non-detection could not be given as what it is. These pin, against scipy:

* the densities, truncation normalisation included, in both backends, in the tails too;
* the draws (``sample``, ``rescale``, and ``ppf_jax`` -- the probability integral transform the NUTS
  samplers sample a TruncatedNormal by) with a KS test against ``scipy.stats.truncnorm``;
* the posterior, on a conjugate case with a closed form: ``y = a + b (t - tbar) + c``, ``a`` Normal,
  ``b`` TruncatedNormal cut where the posterior still has mass, ``c`` Fixed -- for every sampler
  (the ABC and SNPE paths in detail: ``tests/test_priors_in_abc_nested_snpe.py``).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import integrate, stats

import whisper_cbpf as wp
from whisper_cbpf.priors import Fixed, Normal, Prior, TruncatedNormal, Uniform

#: (mu, sigma, low, high): central, one-sided either way, deep in either tail, and very tight.
TN_CASES = [(0.0, 1.0, -1.0, 2.0), (0.3, 0.5, 0.0, np.inf), (1.0, 2.0, -np.inf, 0.5),
            (0.0, 1.0, 5.0, 8.0), (2.0, 0.5, -3.0, -1.0), (10.0, 3.0, 9.99, 10.02)]


def _scipy(mu, sigma, lo, hi):
    return stats.truncnorm((lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma)


# ------------------------------------------------------------------------------ the densities
def test_normal_matches_scipy():
    d = Normal(0.051, 0.002)
    x = np.linspace(0.04, 0.06, 11)
    assert np.allclose([d.log_prob(v) for v in x], stats.norm.logpdf(x, 0.051, 0.002), rtol=1e-12)
    p = np.array([1e-6, 0.1, 0.5, 0.975, 1 - 1e-9])
    assert np.allclose([d.rescale(v) for v in p], stats.norm.ppf(p, 0.051, 0.002), rtol=1e-12)
    assert np.allclose(d.cdf(x), stats.norm.cdf(x, 0.051, 0.002), rtol=1e-12)
    assert d.bounds == (-math.inf, math.inf) and d.std == 0.002


@pytest.mark.parametrize("case", TN_CASES)
def test_truncated_normal_density_is_exact_and_normalised(case):
    mu, sigma, lo, hi = case
    d, ref = TruncatedNormal(*case), _scipy(*case)
    a, b = ref.ppf(1e-6), ref.ppf(1 - 1e-6)
    x = np.linspace(a, b, 23)
    assert np.allclose([d.log_prob(v) for v in x], ref.logpdf(x), rtol=1e-9, atol=1e-9)
    total, _ = integrate.quad(lambda v: math.exp(d.log_prob(v)), max(lo, mu - 40 * sigma),
                              min(hi, mu + 40 * sigma), points=[a, b], limit=200)
    assert total == pytest.approx(1.0, abs=1e-7)
    if np.isfinite(lo):
        assert d.log_prob(lo - 1e-9 * max(1.0, abs(lo))) == -math.inf
    if np.isfinite(hi):
        assert d.log_prob(hi + 1e-9 * max(1.0, abs(hi))) == -math.inf
    p = np.array([1e-9, 0.01, 0.3, 0.5, 0.9, 1 - 1e-9])
    assert np.allclose(d.ppf(p), ref.ppf(p), rtol=1e-8, atol=1e-10 * sigma)
    assert np.allclose(d.cdf(ref.ppf(p)), p, rtol=1e-7, atol=1e-12)
    assert d.std == pytest.approx(ref.std(), rel=1e-9)


@pytest.mark.parametrize("case", TN_CASES)
def test_truncated_normal_draws_pass_a_ks_test_against_scipy(case):
    d, ref = TruncatedNormal(*case), _scipy(*case)
    rng = np.random.default_rng(11)
    draws = np.array([d.sample(rng) for _ in range(20000)])
    assert np.all((draws >= d.low) & (draws <= d.high))
    assert stats.kstest(draws, ref.cdf).pvalue > 1e-3


@pytest.mark.parametrize("x64", [True, False])
@pytest.mark.parametrize("case", TN_CASES)
def test_the_probability_integral_transform_on_the_jax_side_is_exact(case, x64):
    """``ppf_jax`` of U(0, 1) draws is what the NUTS samplers sample: it must BE the distribution,
    in float64 and, to float32 precision, in float32."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from whisper_cbpf.priors import ppf_jax

    before = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", x64)
    try:
        d, ref = TruncatedNormal(*case), _scipy(*case)
        p = np.random.default_rng(12).uniform(size=20000)
        x = np.asarray(ppf_jax(d)(jnp.asarray(p)), dtype=float)
        ftype = np.float64 if x64 else np.float32               # the bounds as the device holds them
        assert np.all((x >= ftype(d.low)) & (x <= ftype(d.high)))
        assert stats.kstest(x, ref.cdf).pvalue > 1e-3
        grid = np.array([0.01, 0.3, 0.5, 0.7, 0.99])
        got = np.asarray(ppf_jax(d)(jnp.asarray(grid)), dtype=float)
        assert np.allclose(got, ref.ppf(grid), rtol=1e-9 if x64 else 1e-4,
                           atol=(1e-12 if x64 else 1e-5) * case[1])
    finally:
        jax.config.update("jax_enable_x64", before)


def test_log_prob_jax_equals_prior_log_prob_for_every_new_family():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from whisper_cbpf.priors import log_prob_jax

    before = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        prior = Prior({"z": Normal(0.05, 0.01), "m": TruncatedNormal(0.3, 0.5, 0.0, np.inf),
                       "t": TruncatedNormal(0.0, 1.0, 5.0, 8.0), "c": Fixed(2.5),
                       "u": Uniform(-1.0, 1.0)})
        f = log_prob_jax(prior)
        for row in ([0.05, 0.1, 5.5, 2.5, 0.0], [0.0, 3.0, 7.9, 2.5, -0.9]):
            want = prior.log_prob(dict(zip(prior.names, row)))
            assert float(f(jnp.asarray(row))) == pytest.approx(want, rel=1e-12)
        outside = jnp.asarray([0.05, -0.1, 4.0, 2.5, 0.0])      # m < 0 and t < 5
        assert float(f(outside)) == -np.inf
        assert np.all(np.isfinite(np.asarray(jax.grad(f)(outside))))
        assert float(f(jnp.asarray([0.05, 0.1, 5.5, 2.4, 0.0]))) == -np.inf     # c off its value
    finally:
        jax.config.update("jax_enable_x64", before)


# ----------------------------------------------------------------------------- Fixed and errors
def test_fixed_is_its_value_and_refuses_a_moved_value_by_name():
    d = Fixed(0.051)
    assert d.sample(np.random.default_rng(0)) == 0.051 and d.rescale(0.3) == 0.051
    assert d.log_prob(0.051) == 0.0 and d.bounds == (0.051, 0.051)
    with pytest.raises(ValueError, match="Every whisper sampler holds it at its value"):
        d.log_prob(0.0510001)
    prior = Prior({"a": Uniform(0, 1), "z": Fixed(0.051)})
    assert prior.fixed == {"z": 0.051}
    assert Prior({"a": Uniform(0, 1)}).fixed == {}


@pytest.mark.parametrize("args,match", [
    ((Normal, 0.0, 0.0), "sigma > 0"), ((Normal, np.nan, 1.0), "finite mu"),
    ((TruncatedNormal, 0.0, 1.0, 1.0, 1.0), "low < high"),
    ((TruncatedNormal, 0.0, 1.0, 40.0, 41.0), "keeps no probability"),
    ((Fixed, np.inf), "finite value"),
])
def test_bad_parameters_are_refused_with_the_reason(args, match):
    cls, *a = args
    with pytest.raises(ValueError, match=match):
        cls(*a)


# ------------------------------------------------------------- the conjugate posterior, per sampler
T = np.linspace(-5.0, 5.0, 20)
SIG = 0.5
A_PRIOR, B_PRIOR = Normal(1.0, 0.1), TruncatedNormal(0.0, 0.05, 0.0, 1.0)
PRIOR = Prior({"a": A_PRIOR, "b": B_PRIOR, "c": Fixed(0.0)})


def _line(p, t, bands=None):
    t = np.asarray(t, dtype=float)
    return float(p["a"]) + float(p["b"]) * (t - T.mean()) + float(p["c"])


def _line_jax(theta, t, band_idx=None):
    import jax.numpy as jnp
    return theta[0] + theta[1] * (jnp.asarray(t) - T.mean()) + theta[2]


@pytest.fixture(scope="module")
def conjugate():
    """``y = a + b (t - tbar) + c`` with ``sum(t - tbar) = 0``: a and b are independent in the
    likelihood, so a's posterior is the Normal-Normal closed form and b's the Normal-Normal
    posterior truncated to b's range (a truncated normal, moments from scipy)."""
    y = 1.3 + 0.02 * (T - T.mean()) + np.random.default_rng(3).normal(0.0, SIG, T.size)
    kw = {}
    try:
        import jax  # noqa: F401
        kw["predict_jax"] = _line_jax
    except ImportError:
        pass
    m = wp.register_model("priors_beyond_boxes_line", _line, ["a", "b", "c"], prior=PRIOR,
                          overwrite=True, **kw)
    lc = wp.LightCurve(time=T, band=["r"] * T.size, flux=y, flux_err=np.full(T.size, SIG))
    w = np.full(T.size, SIG ** -2)
    s_a = (A_PRIOR.sigma ** -2 + w.sum()) ** -0.5
    m_a = s_a ** 2 * (A_PRIOR.mu / A_PRIOR.sigma ** 2 + np.sum(w * y))
    tc = T - T.mean()
    s_b = (B_PRIOR.sigma ** -2 + np.sum(w * tc ** 2)) ** -0.5
    m_b = s_b ** 2 * (B_PRIOR.mu / B_PRIOR.sigma ** 2 + np.sum(w * tc * y))
    b_post = stats.truncnorm((0.0 - m_b) / s_b, (1.0 - m_b) / s_b, loc=m_b, scale=s_b)
    assert b_post.cdf(m_b) > 0.1                     # the cut takes a real share of the posterior
    return m, lc, {"a": (m_a, s_a), "b": (b_post.mean(), b_post.std())}


def _check_posterior(r, want, tol_mean=0.12, tol_sd=0.12):
    for nm, (mean, sd) in want.items():
        x = r.samples[nm].to_numpy()
        assert abs(x.mean() - mean) < tol_mean * sd, (r.sampler, nm, x.mean(), mean, sd)
        assert abs(x.std() / sd - 1.0) < tol_sd, (r.sampler, nm, x.std(), sd)
    assert np.all(r.samples["b"] >= 0.0) and np.all(r.samples["c"] == 0.0)
    assert r.best_params["c"] == 0.0


def _fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.fit(*args, **kwargs)


@pytest.fixture
def float64():
    jax = pytest.importorskip("jax")
    before = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


def test_cpu_mcmc_recovers_the_conjugate_posterior_and_holds_the_fixed_parameter(conjugate):
    m, lc, want = conjugate
    r = _fit(lc, m.name, sampler="mcmc", space="flux", nwalkers=16, nsteps=6000, burnin=1000,
             seed=0)
    _check_posterior(r, want)
    assert r.n_params == 2 and r.info["fixed"] == {"c": 0.0} and r.parameters == ["a", "b", "c"]


def test_nested_sampling_recovers_it_through_rescale(conjugate):
    m, lc, want = conjugate
    r = _fit(lc, m.name, sampler="nested", space="flux", nlive=300, seed=0)
    _check_posterior(r, want, tol_mean=0.2, tol_sd=0.2)


def test_emcee_jax_recovers_it(conjugate, float64):
    m, lc, want = conjugate
    r = _fit(lc, m.name, sampler="emcee_jax", space="flux", nsteps=4000, burnin=1000, seed=0)
    _check_posterior(r, want)
    assert r.n_params == 2 and r.info["fixed"] == {"c": 0.0}


def test_nuts_recovers_it_through_the_probability_integral_transform(conjugate, float64):
    pytest.importorskip("numpyro")
    m, lc, want = conjugate
    r = _fit(lc, m.name, sampler="nuts_gpu", space="flux", num_warmup=500, num_samples=2000,
             seed=0)
    _check_posterior(r, want)
    assert r.info["sample_sites"] == {"a": "a", "b": "cdf_b"}
    assert set(r.info["rhat"]) == {"a", "b"} and r.info["converged"] is True
    assert r.n_params == 2 and r.samples_by_chain.shape[-1] == 3


def test_pymc_recovers_it(conjugate, float64):
    pytest.importorskip("pymc")
    m, lc, want = conjugate
    r = _fit(lc, m.name, sampler="pymc_jax_gpu_vectorized", space="flux", num_warmup=500,
             num_samples=1500, seed=0)
    _check_posterior(r, want)
    assert r.n_params == 2 and r.info["fixed"] == {"c": 0.0}


def test_a_flat_likelihood_returns_the_truncated_prior_itself(float64):
    """The prior alone through each JAX sampler's own mapping: the posterior of a parameter the
    data say nothing about IS its prior, so the draws must pass a KS test against scipy."""
    pytest.importorskip("numpyro")
    import jax.numpy as jnp

    d = TruncatedNormal(0.3, 0.5, 0.0, np.inf)
    m = wp.register_model("priors_beyond_boxes_flat", lambda p, t, b=None: np.zeros(len(t)), ["x"],
                          prior=Prior({"x": d}),
                          predict_jax=lambda th, t, bi=None: jnp.zeros_like(jnp.asarray(t)) * th[0],
                          overwrite=True)
    lc = wp.LightCurve(time=np.linspace(0, 1, 5), band=["r"] * 5, flux=np.zeros(5),
                       flux_err=np.ones(5))
    ref = _scipy(0.3, 0.5, 0.0, np.inf)
    r = _fit(lc, m.name, sampler="nuts_gpu", space="flux", log_prob_fn=lambda th: 0.0 * th[0],
             num_warmup=300, num_samples=3000, seed=1)
    assert stats.kstest(r.samples["x"].to_numpy()[::5], ref.cdf).pvalue > 1e-3
    e = _fit(lc, m.name, sampler="emcee_jax", space="flux", nsteps=6000, burnin=1000, seed=1)
    assert stats.kstest(e.samples["x"].to_numpy()[::4], ref.cdf).pvalue > 1e-3


# -------------------------------------------------------------------------- ABC and SNPE take them
def test_abc_draws_them_and_holds_the_fixed_parameter(conjugate):
    m, lc, _ = conjugate
    r = _fit(lc, m.name, sampler="abc", n_simulations=4000, quantile=0.05, seed=0, space="flux")
    assert np.all(r.samples["c"] == 0.0) and np.all(r.samples["b"] >= 0.0)


def test_abc_smc_perturbs_the_free_parameters_and_holds_the_fixed_one(conjugate):
    m, lc, _ = conjugate
    r = _fit(lc, m.name, sampler="abc_smc", n_particles=40, n_rounds=2, seed=0, space="flux")
    assert np.all(r.samples["c"] == 0.0) and np.all(r.samples["b"] >= 0.0)
    assert r.n_params == 2 and r.info["fixed"] == {"c": 0.0}


def test_the_gpu_abc_and_snpe_take_a_normal_prior(conjugate):
    m, lc, _ = conjugate
    prior = Prior({"a": A_PRIOR, "b": Uniform(0.0, 1.0), "c": Uniform(-0.1, 0.1)})
    pytest.importorskip("jax")
    r = _fit(lc, m.name, sampler="abc_gpu", prior=prior, n_simulations=500, space="flux")
    assert r.n_params == 3 and np.all(np.isfinite(r.samples["a"]))
    pytest.importorskip("sbi")
    r = _fit(lc, m.name, sampler="snpe", prior=prior, num_simulations=200, num_rounds=1,
             max_num_epochs=10, space="flux")
    assert r.n_params == 3 and np.all(np.isfinite(r.samples["a"]))


def test_an_all_fixed_prior_is_refused():
    m = wp.register_model("priors_beyond_boxes_const", lambda p, t, b=None: np.full(len(t), p["c"]),
                          ["c"], prior=Prior({"c": Fixed(1.0)}), overwrite=True)
    lc = wp.LightCurve(time=np.linspace(0, 1, 5), band=["r"] * 5, flux=np.ones(5),
                       flux_err=np.ones(5))
    with pytest.raises(ValueError, match="every parameter is Fixed"):
        _fit(lc, m.name, sampler="mcmc", space="flux")
