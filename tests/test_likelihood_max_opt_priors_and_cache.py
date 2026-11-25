"""``likelihood_max_opt`` with priors beyond boxes, the fit's own prior and rows, and
compiled-program reuse.

* A ``Normal`` or one-sided ``TruncatedNormal`` parameter has no finite box: it is climbed in
  standardised coordinates, and a finite end of a truncation is still reached exactly.
* A ``Fixed`` parameter (in the prior, or in ``result.info["fixed"]``) is never optimised and not
  counted in AIC/BIC.
* Without ``prior=``, the prior recorded with the fit sets the box; the fit's pre-event rows are
  left out as the fit left them out.
* The JAX backend keeps its compiled programs per (model, prior families, likelihood, bucket):
  a second optimisation, of the same density or of another light curve of that size, does not
  compile again, and gives its own data's peak.
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.likelihood import make_likelihood
from whisper_cbpf.priors import Fixed, Normal, Prior, TruncatedNormal, Uniform
from whisper_cbpf.samplers.base import SamplerResult

# wp.likelihood_max_opt is the function, not the module
lmo_mod = sys.modules["whisper_cbpf.likelihood_max_opt"]

T = np.linspace(0.5, 20.0, 25)
BOX = Prior({"a": Uniform(-10.0, 10.0), "b": Uniform(-1.0, 1.0)})


def _line(p, t, bands=None):
    return p["a"] + p["b"] * np.asarray(t, dtype=float)


def _line_jax(theta, t, band_idx=None):
    import jax.numpy as jnp
    return theta[0] + theta[1] * jnp.asarray(t)


def _register(name="_pp_line", jax=False):
    kw = {"predict_jax": _line_jax} if jax else {}
    return wp.register_model(name, _line, ["a", "b"], prior=BOX, overwrite=True, **kw)


_register()


def _lc(truth=(1.0, 0.2), sigma=0.1, seed=3, t=T):
    y = truth[0] + truth[1] * t + np.random.default_rng(seed).normal(0.0, sigma, t.size)
    return wp.LightCurve(time=t, band=["r"] * t.size, flux=y, flux_err=np.full(t.size, sigma))


def _result(lc, draws, model="_pp_line", info=None):
    """A SamplerResult whose best draw is the argmax of its draws under the fit's likelihood."""
    m = wp.get_model(model)
    lik = make_likelihood(lc, space="flux")
    t = np.asarray(lc.time, float)
    ll = np.array([lik.log_likelihood(m.predict(r, t, None)) for r in draws.to_dict("records")])
    i = int(np.argmax(ll))
    names = list(draws.columns)
    return SamplerResult(
        sampler="test", model=m.name, parameters=names, samples=draws, summary={},
        best_params={nm: float(draws[nm].iloc[i]) for nm in names}, n_data=lc.n_points,
        n_params=len(names), runtime_s=0.0,
        info={"space": "flux", "likelihood": "GaussianLikelihood", **(info or {})},
        max_log_likelihood=float(ll[i]))


def _draws(a, b, n=80, seed=0, spread=(0.05, 0.005)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": a + spread[0] * rng.standard_normal(n),
                         "b": b + spread[1] * rng.standard_normal(n)})


def _wls(lc, a_fixed=None, b_fixed=None):
    """ln L max of the line in closed form, with a or b held at a value if asked."""
    t, y, s = (np.asarray(v, float) for v in (lc.time, lc.flux, lc.flux_err))
    w = 1.0 / s ** 2
    if a_fixed is not None:
        b = float(np.sum(w * t * (y - a_fixed)) / np.sum(w * t * t))
        a = a_fixed
    elif b_fixed is not None:
        a = float(np.sum(w * (y - b_fixed * t)) / np.sum(w))
        b = b_fixed
    else:
        X = np.column_stack([np.ones_like(t), t])
        a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    norm = -0.5 * float(np.sum(np.log(2 * np.pi) + 2 * np.log(s)))
    return float(a), float(b), -0.5 * float(np.sum(w * (y - a - b * t) ** 2)) + norm


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@pytest.fixture
def x64():
    jax = pytest.importorskip("jax")
    before = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


# --------------------------------------------------------------------- unbounded coordinates
def test_normal_priors_are_climbed_to_the_unconstrained_peak():
    lc = _lc()
    a, b, ll = _wls(lc)
    prior = Prior({"a": Normal(0.0, 5.0), "b": Normal(0.0, 1.0)})
    pr = wp.likelihood_max_opt(_result(lc, _draws(1.3, 0.18)), lc, prior=prior)
    assert pr.params["a"] == pytest.approx(a, abs=1e-6)
    assert pr.params["b"] == pytest.approx(b, abs=1e-7)
    assert pr.max_log_likelihood == pytest.approx(ll, abs=1e-9)
    assert pr.at_edge == []


@pytest.mark.parametrize("cut", ["upper", "lower"])
def test_the_finite_end_of_a_one_sided_truncation_is_reached_exactly(cut):
    """The slope's truncation excludes the least-squares slope (~0.2): the peak sits ON the
    finite end, 0.15 from above or 0.3 from below, with a at its conditional optimum."""
    lc = _lc()
    edge = 0.15 if cut == "upper" else 0.3
    tn = (TruncatedNormal(0.0, 1.0, -np.inf, edge) if cut == "upper"
          else TruncatedNormal(0.0, 1.0, edge, np.inf))
    a_edge, _, ll_edge = _wls(lc, b_fixed=edge)
    start = _draws(1.0, edge - 0.02 if cut == "upper" else edge + 0.02)
    pr = wp.likelihood_max_opt(_result(lc, start), lc,
                               prior=Prior({"a": Normal(0.0, 5.0), "b": tn}))
    assert pr.params["b"] == edge                               # exactly on the finite end
    assert pr.params["a"] == pytest.approx(a_edge, abs=1e-6)
    assert pr.max_log_likelihood == pytest.approx(ll_edge, abs=1e-9)
    assert pr.at_edge == ["b"]


def test_standardised_coordinates_round_trip():
    box = lmo_mod._Box(Prior({"a": Normal(2.0, 0.5), "b": TruncatedNormal(0.0, 1.0, 0.3, np.inf),
                                 "c": Uniform(0.0, 4.0), "d": Fixed(7.0)}), ["a", "b", "c", "d"])
    assert box.names == ["a", "b", "c"] and box.fixed == {"d": 7.0}
    x = np.array([[2.5, 1.3, 1.0]])
    np.testing.assert_allclose(box.to_u(x), [[1.0, 1.3, 0.25]])
    np.testing.assert_allclose(box.to_x(box.to_u(x)), x)
    assert box.ulo.tolist() == [-np.inf, 0.3, 0.0] and box.uhi.tolist() == [np.inf, np.inf, 1.0]
    assert box.to_x(np.array([[0.0, -5.0, 0.0]]))[0, 1] == 0.3       # clipped onto the end
    np.testing.assert_allclose(box.full(x), [[2.5, 1.3, 1.0, 7.0]])


# --------------------------------------------------------------------- Fixed parameters
def test_a_fixed_parameter_is_never_optimised_nor_counted():
    lc = _lc()
    _, b, ll = _wls(lc, a_fixed=1.0)
    draws = _draws(1.0, 0.19)
    draws["a"] = 1.0
    pr = wp.likelihood_max_opt(_result(lc, draws), lc,
                               prior=Prior({"a": Fixed(1.0), "b": Uniform(-1.0, 1.0)}))
    assert pr.params == {"a": 1.0, "b": pytest.approx(b, abs=1e-7)}
    assert pr.max_log_likelihood == pytest.approx(ll, abs=1e-9)
    assert pr.n_params == 1
    assert pr.bic == pytest.approx(-2.0 * pr.max_log_likelihood + math.log(lc.n_points), abs=1e-9)


def test_the_fits_own_fixed_record_holds_the_parameter_under_the_default_prior():
    lc = _lc()
    draws = _draws(1.0, 0.19)
    draws["a"] = 1.0
    pr = wp.likelihood_max_opt(_result(lc, draws, info={"fixed": {"a": 1.0}}), lc)
    assert pr.params["a"] == 1.0 and pr.n_params == 1


def test_every_parameter_fixed_is_refused():
    lc = _lc()
    draws = _draws(1.0, 0.2)
    draws["a"], draws["b"] = 1.0, 0.2
    with pytest.raises(ValueError, match="every parameter is Fixed"):
        wp.likelihood_max_opt(_result(lc, draws), lc,
                              prior=Prior({"a": Fixed(1.0), "b": Fixed(0.2)}))


# --------------------------------------------------------------------- the fit's prior and rows
def test_without_prior_the_prior_recorded_with_the_fit_sets_the_box():
    """The fit ran under b <= 0.15; optimising it without prior= climbs in THAT box (the model's
    default, b <= 1, would put the peak at ~0.2, outside the fit's prior)."""
    lc = _lc()
    narrow = Prior({"a": Uniform(-10.0, 10.0), "b": Uniform(-1.0, 0.15)})
    res = _quiet(wp.fit, lc, "_pp_line", sampler="abc", prior=narrow, n_simulations=3000,
                 quantile=0.02, seed=0)
    pr = wp.likelihood_max_opt(res, lc)
    assert pr.params["b"] == 0.15 and pr.at_edge == ["b"]


def test_the_fits_pre_event_rows_are_left_out_as_the_fit_left_them():
    t = np.linspace(-4.0, 20.0, 25)
    lc = _lc(t=t)
    lc = wp.LightCurve(time=np.asarray(lc.time) + 60000.0, band=lc.band, flux=lc.flux,
                       flux_err=lc.flux_err).set_explosion_date(60000.0)
    res = _quiet(wp.fit, lc, "_pp_line", sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
    assert res.info.get("excluded_pre_event", 0) > 0 and res.n_data < lc.n_points
    pr = _quiet(wp.likelihood_max_opt, res, lc)             # the full curve: rows left out here
    assert pr.n_data == res.n_data and pr.max_log_likelihood >= res.max_log_likelihood


# --------------------------------------------------------------------- JAX backend and its cache
def _jax_case():
    _register("_pp_line_jax", jax=True)
    lc = _lc()
    prior = Prior({"a": Normal(0.0, 5.0), "b": TruncatedNormal(0.0, 1.0, -np.inf, 0.15)})
    return lc, prior, _result(lc, _draws(1.0, 0.13), model="_pp_line_jax")


def test_jax_and_cpu_backends_agree_with_unbounded_and_truncated_priors(x64):
    lc, prior, res = _jax_case()
    cpu = wp.likelihood_max_opt(res, lc, prior=prior, backend="cpu")
    gpu = wp.likelihood_max_opt(res, lc, prior=prior, backend="jax")
    assert gpu.params["b"] == cpu.params["b"] == 0.15
    assert gpu.params["a"] == pytest.approx(cpu.params["a"], abs=1e-7)
    assert gpu.max_log_likelihood == pytest.approx(cpu.max_log_likelihood, abs=1e-9)
    fixed = Prior({"a": Fixed(1.0), "b": Uniform(-1.0, 1.0)})
    draws = _draws(1.0, 0.19)
    draws["a"] = 1.0
    res_f = _result(lc, draws, model="_pp_line_jax")
    c, g = (wp.likelihood_max_opt(res_f, lc, prior=fixed, backend=be) for be in ("cpu", "jax"))
    assert g.params["a"] == c.params["a"] == 1.0 and g.n_params == c.n_params == 1
    assert g.params["b"] == pytest.approx(c.params["b"], abs=1e-8)


def test_a_second_optimisation_of_the_same_density_reuses_the_compiled_programs(x64):
    lc, prior, res = _jax_case()
    lmo_mod._PROGRAMS.clear()
    first = wp.likelihood_max_opt(res, lc, prior=prior, backend="jax")
    assert "reused" not in first.method and len(lmo_mod._PROGRAMS) == 1
    programs = next(iter(lmo_mod._PROGRAMS.values()))
    other = _result(lc, _draws(0.9, 0.14, seed=5), model="_pp_line_jax")    # another fit, same density
    second = wp.likelihood_max_opt(other, lc, prior=prior, backend="jax")
    assert "reused" in second.method and next(iter(lmo_mod._PROGRAMS.values())) is programs
    assert second.params == pytest.approx(first.params, abs=1e-8)
    # The data and the prior's numbers are ARGUMENTS of the programs: another light curve of the
    # same size bucket, or another prior of the same families, reuses them, each with its own peak.
    lc2 = _lc(seed=4)
    res2 = _result(lc2, _draws(1.0, 0.13), model="_pp_line_jax")
    other_data = wp.likelihood_max_opt(res2, lc2, prior=prior, backend="jax")
    assert "reused" in other_data.method
    assert other_data.max_log_likelihood == pytest.approx(
        wp.likelihood_max_opt(res2, lc2, prior=prior, backend="cpu").max_log_likelihood, abs=1e-9)
    wide = Prior({"a": Normal(0.0, 5.0), "b": TruncatedNormal(0.0, 1.0, -np.inf, 0.16)})
    other_numbers = wp.likelihood_max_opt(res, lc, prior=wide, backend="jax")
    assert "reused" in other_numbers.method and other_numbers.params["b"] == 0.16
    # what changes the program is a new one: other prior families, another bucket, a new model
    box = Prior({"a": Uniform(-10.0, 10.0), "b": Uniform(-1.0, 0.15)})
    assert "reused" not in wp.likelihood_max_opt(res, lc, prior=box, backend="jax").method
    lc3 = _lc(t=np.linspace(0.5, 20.0, 40))                   # 40 points: bucket 48, not 32
    res3 = _result(lc3, _draws(1.0, 0.13), model="_pp_line_jax")
    assert "reused" not in wp.likelihood_max_opt(res3, lc3, prior=prior, backend="jax").method
    _register("_pp_line_jax", jax=True)          # registered again, the same forward function
    assert "reused" in wp.likelihood_max_opt(res, lc, prior=prior, backend="jax").method

    def other_forward(theta, t, band_idx=None):
        return _line_jax(theta, t, band_idx)

    wp.register_model("_pp_line_jax", _line, ["a", "b"], prior=BOX, overwrite=True,
                      predict_jax=other_forward)     # another forward function: a new program
    assert "reused" not in wp.likelihood_max_opt(res, lc, prior=prior, backend="jax").method
    _register("_pp_line_jax", jax=True)
    assert len(lmo_mod._PROGRAMS) <= lmo_mod.PROGRAM_CACHE_SIZE
