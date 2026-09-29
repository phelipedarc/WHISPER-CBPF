"""Normal, TruncatedNormal and Fixed priors in ``nested``, ``abc``, ``abc_smc``, ``abc_gpu``,
``abc_smc_gpu``, ``snpe`` and ``snpe_gpu``.

What each sampler promises:

* a ``Fixed`` parameter is held at its value: never a sampled dimension (not in nested's unit cube,
  never perturbed by ABC-SMC, not in SNPE's torch prior), a constant column in ``samples``, listed
  in ``info["fixed"]``, and NOT counted in AIC/BIC (``n_params``);
* the GPU ABC samplers draw ``Normal`` / ``TruncatedNormal`` on the device through the inverse CDF,
  so accepting every draw returns the prior itself (a KS test against scipy);
* SNPE's torch prior carries the exact density and draws of the whisper distributions.

The model is ``y = a + b (t - tbar) + c`` with ``a ~ Normal``, ``b ~ TruncatedNormal`` (cut at 0)
and ``c`` Fixed at 0, as in ``tests/test_priors_beyond_boxes.py``.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

import whisper_cbpf as wp
from whisper_cbpf.priors import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform

T = np.linspace(-5.0, 5.0, 20)
SIG = 0.5
A_PRIOR, B_PRIOR = Normal(1.0, 0.1), TruncatedNormal(0.0, 0.05, 0.0, 1.0)
PRIOR = Prior({"a": A_PRIOR, "b": B_PRIOR, "c": Fixed(0.0)})
BOX = Prior({"a": Uniform(0.0, 2.0), "b": LogUniform(1e-3, 1.0), "c": Uniform(-0.1, 0.1)})


def _line(p, t, bands=None):
    t = np.asarray(t, dtype=float)
    return float(p["a"]) + float(p["b"]) * (t - T.mean()) + float(p["c"])


def _line_jax(theta, t, band_idx=None):
    import jax.numpy as jnp
    return theta[0] + theta[1] * (jnp.asarray(t) - T.mean()) + theta[2]


@pytest.fixture(scope="module")
def line():
    kw = {}
    try:
        import jax  # noqa: F401
        kw["predict_jax"] = _line_jax
    except ImportError:
        pass
    wp.register_model("priors_every_sampler_line", _line, ["a", "b", "c"], prior=PRIOR,
                      overwrite=True, **kw)
    y = 1.3 + 0.02 * (T - T.mean()) + np.random.default_rng(3).normal(0.0, SIG, T.size)
    return wp.LightCurve(time=T, band=["r"] * T.size, flux=y, flux_err=np.full(T.size, SIG))


def _fit(lc, sampler, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.fit(lc, "priors_every_sampler_line", sampler=sampler, space="flux", seed=0, **kw)


def _check_fixed(r):
    """The Fixed parameter is held and not counted; AIC/BIC use the two free parameters."""
    assert r.parameters == ["a", "b", "c"]
    assert r.info["fixed"] == {"c": 0.0}
    assert r.n_params == 2
    assert bool((r.samples["c"] == 0.0).all()) and r.best_params["c"] == 0.0
    assert bool((r.samples["b"] >= 0.0).all())
    n = r.n_data
    assert r.aic == pytest.approx(-2.0 * r.max_log_likelihood + 2 * 2, abs=1e-9)
    assert r.bic == pytest.approx(-2.0 * r.max_log_likelihood + 2 * math.log(n), abs=1e-9)


# ------------------------------------------------------------------------------------------ CPU
def test_nested_keeps_a_fixed_parameter_out_of_the_unit_cube_and_the_count(line):
    r = _fit(line, "nested", nlive=150)
    _check_fixed(r)
    assert r.dynesty_results.samples.shape[1] == 2           # two dimensions, not three


def test_abc_does_not_count_a_fixed_parameter(line):
    _check_fixed(_fit(line, "abc", n_simulations=3000, quantile=0.05))


def test_abc_smc_never_perturbs_a_fixed_parameter(line):
    """0.1.1 perturbed every parameter and stopped at Fixed.log_prob's ValueError; now the kernel
    moves the free parameters only, so the Fixed value is exact in every particle."""
    r = _fit(line, "abc_smc", n_particles=150, n_rounds=3)
    _check_fixed(r)
    assert len(r.info["rounds"]) == 3


def test_abc_smc_without_a_fixed_parameter_is_unchanged_by_the_split(line):
    """No Fixed parameter: the free set is every parameter, so the result is the one a prior with
    no split would give -- reproducible for a seed and independent of n_jobs as before."""
    prior = Prior({"a": A_PRIOR, "b": B_PRIOR, "c": Uniform(-0.1, 0.1)})
    a = _fit(line, "abc_smc", prior=prior, n_particles=100, n_rounds=2)
    b = _fit(line, "abc_smc", prior=prior, n_particles=100, n_rounds=2)
    assert a.samples.equals(b.samples) and a.n_params == 3 and a.info["fixed"] == {}


def test_every_parameter_fixed_is_refused_by_name(line):
    prior = Prior({"a": Fixed(1.0), "b": Fixed(0.0), "c": Fixed(0.0)})
    for sampler, kw in (("nested", {"nlive": 50}), ("abc", {"n_simulations": 100}),
                        ("abc_smc", {"n_particles": 20, "n_rounds": 1})):
        with pytest.raises(ValueError, match=f"{sampler}: every parameter is Fixed"):
            _fit(line, sampler, prior=prior, **kw)


# ------------------------------------------------------------------------------------------ GPU ABC
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_abc_gpu_draws_normal_and_truncated_normal_exactly(line, precision):
    """threshold=inf accepts every draw, so the samples ARE the prior draws."""
    pytest.importorskip("jax")
    r = _fit(line, "abc_gpu", n_simulations=20000, threshold=np.inf, precision=precision)
    assert len(r.samples) == 20000
    assert stats.kstest(r.samples["a"], stats.norm(1.0, 0.1).cdf).pvalue > 1e-3
    b_ref = stats.truncnorm(0.0, 20.0, loc=0.0, scale=0.05)
    assert stats.kstest(r.samples["b"], b_ref.cdf).pvalue > 1e-3
    _check_fixed(r)


def test_abc_gpu_one_sided_truncation_never_draws_an_infinity(line):
    """jax.random.uniform can return exactly 0, whose Normal quantile is -inf: kept inside (0, 1)."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from whisper_cbpf.samplers.jax.abc_gpu import _PriorDraw

    spec = _PriorDraw(Prior({"a": Normal(0.0, 1.0), "b": TruncatedNormal(0.0, 1.0, -np.inf, 0.5),
                             "c": Fixed(2.0)}), ["a", "b", "c"], "abc_gpu")
    draw = jax.vmap(spec.build(jnp))
    theta = np.asarray(draw(jnp.array([[0.0, 0.0, 0.3], [1.0 - 2 ** -24, 1.0 - 2 ** -24, 0.7]])))
    assert np.all(np.isfinite(theta)) and np.all(theta[:, 1] <= 0.5) and np.all(theta[:, 2] == 2.0)


def test_abc_gpu_box_priors_draw_what_0_1_1_drew(line):
    """The inverse-CDF refactor leaves Uniform / LogUniform draws bit-identical: draw i is
    ``lo + u (hi - lo)`` (in log space for LogUniform) of the uniform folded from (seed, i)."""
    pytest.importorskip("jax")
    import jax

    r = _fit(line, "abc_gpu", prior=BOX, n_simulations=500, threshold=np.inf)
    base = jax.random.PRNGKey(0)
    lo = np.array([0.0, np.log(1e-3), -0.1])
    hi = np.array([2.0, np.log(1.0), 0.1])
    for i in (0, 7, 499):
        k_theta, _ = jax.random.split(jax.random.fold_in(base, i))
        u = np.asarray(jax.random.uniform(k_theta, (3,)))
        raw = lo + u * (hi - lo)
        want = np.array([raw[0], np.exp(raw[1]), raw[2]])
        np.testing.assert_allclose(r.samples[["a", "b", "c"]].to_numpy()[i], want, rtol=1e-5)
    assert r.n_params == 3 and r.info["fixed"] == {}


def test_abc_smc_gpu_takes_them_and_holds_the_fixed_parameter(line):
    pytest.importorskip("jax")
    r = _fit(line, "abc_smc_gpu", n_particles=200, n_rounds=3)
    _check_fixed(r)


def test_gpu_abc_refuses_an_unknown_family_naming_the_supported_ones(line):
    pytest.importorskip("jax")

    class Weird:
        bounds = (0.0, 1.0)

    prior = Prior({"a": Weird(), "b": B_PRIOR, "c": Fixed(0.0)})
    for sampler, kw in (("abc_gpu", {"n_simulations": 100}), ("abc_smc_gpu", {"n_particles": 20})):
        with pytest.raises(TypeError, match="parameter 'a' is Weird.*CPU sampler"):
            _fit(line, sampler, prior=prior, **kw)


# ------------------------------------------------------------------------------------------ SNPE
def test_torch_prior_carries_the_exact_density_and_draws():
    pytest.importorskip("sbi")
    import torch
    from sbi.utils.sbiutils import within_support

    from whisper_cbpf.samplers.snpe import _require_sbi, _to_torch_prior

    sb = _require_sbi()
    prior = Prior({"a": A_PRIOR, "b": B_PRIOR, "u": Uniform(-1.0, 1.0),
                   "d": TruncatedNormal(1.0, 2.0, 0.5, np.inf)})
    tp, _, _ = sb.process_prior(_to_torch_prior(prior, sb))
    torch.manual_seed(0)
    s = tp.sample((20000,)).double().numpy()
    assert stats.kstest(s[:, 0], stats.norm(1.0, 0.1).cdf).pvalue > 1e-3
    assert stats.kstest(s[:, 1], stats.truncnorm(0.0, 20.0, loc=0.0, scale=0.05).cdf).pvalue > 1e-3
    assert stats.kstest(s[:, 3], stats.truncnorm(-0.25, np.inf, loc=1.0, scale=2.0).cdf).pvalue > 1e-3
    assert s[:, 1].min() >= 0.0 and s[:, 3].min() >= 0.5
    x = np.array([[1.05, 0.03, 0.2, 0.9], [0.9, 0.2, -0.5, 4.0]])
    got = tp.log_prob(torch.tensor(x, dtype=torch.float32)).double().numpy()
    want = [prior.log_prob(dict(zip(["a", "b", "u", "d"], row))) for row in x]
    np.testing.assert_allclose(got, want, rtol=1e-5)
    outside = torch.tensor([[1.0, -0.01, 0.0, 1.0], [1.0, 0.01, 0.0, 0.4]])
    assert not bool(within_support(tp, outside).any())
    assert bool(torch.all(~torch.isfinite(tp.log_prob(outside))))
    with pytest.raises(TypeError, match="is Fixed.*holds it at its value"):
        _to_torch_prior(Prior({"a": A_PRIOR, "c": Fixed(0.0)}), sb)


def test_snpe_holds_a_fixed_parameter_outside_the_network(line):
    """Two rounds: the second trains with sbi's atomic loss, which evaluates the torch prior's
    density at every proposal draw (a Normal and a TruncatedNormal here)."""
    pytest.importorskip("sbi")
    r = _fit(line, "snpe", num_simulations=400, num_rounds=2, max_num_epochs=20,
             num_samples=500)
    _check_fixed(r)
    assert r.posterior.sample((5,), show_progress_bars=False).shape[-1] == 2


def test_snpe_takes_a_single_free_parameter_of_any_family(line):
    """One free parameter: the torch prior is that one distribution (sbi's MultipleIndependent
    needs two), for a Normal, a TruncatedNormal and a LogUniform alike."""
    pytest.importorskip("sbi")
    for dist in (A_PRIOR, TruncatedNormal(1.0, 0.5, 0.0, np.inf), LogUniform(0.1, 10.0)):
        prior = Prior({"a": dist, "b": Fixed(0.02), "c": Fixed(0.0)})
        r = _fit(line, "snpe", prior=prior, num_simulations=300, num_rounds=2, max_num_epochs=10,
                 num_samples=300)
        assert r.n_params == 1 and r.info["fixed"] == {"b": 0.02, "c": 0.0}
        assert bool((r.samples["b"] == 0.02).all()) and 0.5 < r.samples["a"].median() < 2.0


def test_snpe_gpu_simulator_appends_the_fixed_values(line):
    pytest.importorskip("sbi")
    pytest.importorskip("jax")
    import torch

    from whisper_cbpf.samplers.jax.snpe_gpu import make_predict_torch

    model = wp.get_model("priors_every_sampler_line")
    sim = make_predict_torch(line, model, names=["b", "a"], fixed={"c": 0.25}, chunk=None)
    out = sim(torch.tensor([[0.1, 1.0], [0.0, 2.0]])).double().cpu().numpy()
    want = [_line({"a": 1.0, "b": 0.1, "c": 0.25}, T), _line({"a": 2.0, "b": 0.0, "c": 0.25}, T)]
    np.testing.assert_allclose(out, want, rtol=1e-6)
    assert sim.names == ["b", "a"] and sim.fixed == {"c": 0.25}
    r = _fit(line, "snpe_gpu", num_simulations=400, num_rounds=1, max_num_epochs=20,
             num_samples=500, device="cpu")
    _check_fixed(r)
