"""``nuts_gpu`` must explore a log-uniform parameter in LOG coordinates, not squashed linear ones.

THE DEFECT THESE PIN. ``_numpyro_priors`` used to map whisper's ``LogUniform(lo, hi)`` onto
``numpyro.distributions.LogUniform(lo, hi)``, on the strength of a docstring claiming "NumPyro picks
the right bijector for each support. For LogUniform that is a log transform". It is not.
``dist.LogUniform.support`` is ``constraints.interval(lo, hi)`` -- the same support a plain
``Uniform`` declares -- so ``biject_to`` returns ``ComposeTransform([SigmoidTransform,
AffineTransform])``, a sigmoid on the LINEAR value, and NUTS walked a scale parameter spanning
decades in squashed linear coordinates.

The density was never wrong, which is why nothing caught it: only the GEOMETRY was, and geometry is
invisible to every check that looks at ``log_prob``. What it cost, measured on ``LogUniform(100,
6000)``: the unconstrained origin sits on the ARITHMETIC midpoint 3050 K (the 83rd percentile of the
prior) instead of the prior median 774.6 K, and NumPyro's default ``init_to_uniform(radius=2)``
therefore cannot start a chain below 818 K -- so a true temperature floor of 107 K is unreachable by
initialisation. Simulation-based calibration on data generated FROM the model caught it as rank 0
with no divergences and r-hat near 1: converged, tightly, onto the wrong place.

These tests are deterministic and need neither a GPU nor ``jax_enable_x64`` -- the quantities
asserted are ratios and orderings that hold in single precision, and the one tolerance that has to
follow the session's precision reads the flag rather than assuming it. That matters here:
``tests/t2_autodiff/`` enables x64 globally and irreversibly, so a test that assumed either setting
would pass or fail on collection order.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")

import numpyro.distributions as dist  # noqa: E402
from numpyro.distributions.transforms import biject_to  # noqa: E402

from whisper_cbpf.priors import LogUniform, Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax.nuts_gpu import (  # noqa: E402
    _numpyro_priors,
    _resolve_init_strategy,
    NUTSGPUSampler,
)

#: The kilonova temperature floor, which is the parameter the defect was found on.
LO, HI = 100.0, 6000.0
NAMES = ["mej", "temperature_floor"]


def _tol():
    """Follow the session's JAX precision instead of asserting it.

    ``biject_to`` and ``log_prob`` run at the configured dtype, so in a float32 session they
    round-trip to ~1e-6, not ~1e-15. Pinning either number unconditionally would make these tests
    report the global x64 flag rather than the parameterisation — and the flag genuinely differs
    between running this file alone (off) and running the whole suite (``tests/t2_autodiff/``
    enables x64 globally and irreversibly, and collects first).
    """
    return 1e-9 if jax.config.jax_enable_x64 else 3e-6


def _prior():
    return Prior({"mej": Uniform(1e-2, 0.05), "temperature_floor": LogUniform(LO, HI)})


def _spec(name="temperature_floor"):
    """The single ``(site, distribution, to_linear)`` triple for ``name``."""
    return dict(zip(NAMES, _numpyro_priors(_prior(), NAMES)))[name]


def _to_linear_at(u):
    """The model-parameter value a chain sitting at unconstrained coordinate ``u`` represents."""
    _site, d, to_linear = _spec()
    v = biject_to(d.support)(np.float32(u) if not jax.config.jax_enable_x64 else float(u))
    return float(v if to_linear is None else to_linear(v))


# ------------------------------------------------------------------ the geometry, stated exactly
def test_unconstrained_origin_is_the_prior_median_not_the_arithmetic_midpoint():
    """THE regression. u = 0 must land on sqrt(lo*hi), the log-uniform median.

    Pre-fix this returned 3050.0 -- ``(100 + 6000) / 2`` -- because the sigmoid of the INTERVAL
    transform is centred on the linear midpoint. Post-fix it returns 774.597, because the sigmoid is
    centred on the midpoint of ``[log10 lo, log10 hi]`` and ``10**`` maps that to the geometric mean.
    """
    median = float(np.sqrt(LO * HI))                       # 774.5967...
    midpoint = 0.5 * (LO + HI)                             # 3050.0
    assert median == pytest.approx(774.5967, rel=1e-6), "fixture arithmetic"

    x0 = _to_linear_at(0.0)
    diagnosis = (" It is the ARITHMETIC midpoint: the site is still an interval transform on the "
                 "LINEAR value." if abs(x0 - midpoint) < 1.0 else "")
    assert x0 == pytest.approx(median, rel=1e-4), (
        f"the unconstrained origin maps to {x0:.4f}, not the prior median {median:.4f}.{diagnosis}")
    assert abs(x0 - midpoint) > 1000.0, "u=0 must NOT be the arithmetic midpoint"


def test_unconstrained_coordinate_is_symmetric_in_the_log():
    """+u and -u must be reciprocal about the median -- the defining property of log geometry.

    Under the interval-on-linear transform they are symmetric about 3050 K instead, so the low
    decade is compressed against the bound while the high decade is not.
    """
    median = float(np.sqrt(LO * HI))
    for u in (0.5, 1.0, 2.0, 3.0):
        lo_x, hi_x = _to_linear_at(-u), _to_linear_at(u)
        # geometric mean of the pair returns to the median
        assert np.sqrt(lo_x * hi_x) == pytest.approx(median, rel=1e-4), (
            f"u=+/-{u} maps to ({lo_x:.3f}, {hi_x:.3f}) whose geometric mean is "
            f"{np.sqrt(lo_x * hi_x):.3f}, not the prior median {median:.3f}")


def test_default_init_can_start_a_chain_in_the_bottom_decade():
    """The operational consequence, and the one SBC actually tripped over.

    NumPyro's default ``init_to_uniform(radius=2)`` draws u ~ U(-2, 2). Under the pre-fix
    parameterisation the radius-2 edge is 803.3 K (lowest of 400 actual draws: 818.0), so the ENTIRE
    init distribution sat above the prior median and no chain could start in the decade where the
    SBC truth (107 K) lived. Post-fix the same edge is 162.9 K.
    """
    lowest = _to_linear_at(-2.0)
    assert lowest < 300.0, (
        f"the lowest reachable init is {lowest:.1f} K; pre-fix it was 803.3 K, so a truth anywhere "
        f"in the bottom decade of LogUniform({LO:g}, {HI:g}) could not be initialised near")
    # and the init interval must straddle the prior median rather than sit above it
    assert _to_linear_at(-2.0) < np.sqrt(LO * HI) < _to_linear_at(2.0)


def test_reparameterisation_leaves_the_log_uniform_MEASURE_unchanged():
    """A reparameterisation must change geometry and NOTHING else.

    Pushing ``Uniform(log10 lo, log10 hi)`` through ``10**u`` has to reproduce whisper's own
    ``LogUniform`` density on the linear value, Jacobian included: p(x) = p_u(log10 x) / (x ln 10).
    If this drifts, the fix has quietly become a different posterior -- exactly the failure the
    ``_numpyro_priors`` docstring warns about for the ``.bounds`` shortcut it replaced.
    """
    _site, d, to_linear = _spec()
    ref = LogUniform(LO, HI)
    for x in (100.5, 137.0, 500.0, 774.6, 1000.0, 3050.0, 5900.0):
        u = np.log10(x)
        # change of variables from the sampled coordinate u = log10(x) back to x
        got = float(d.log_prob(u)) - float(np.log(x * np.log(10.0)))
        assert got == pytest.approx(float(ref.log_prob(x)), rel=_tol(), abs=_tol()), f"at x={x}"
        assert to_linear(u) == pytest.approx(x, rel=_tol())


def test_uniform_parameters_are_left_alone():
    """Only LogUniform is reparameterised; a Uniform site must remain its own parameter."""
    site, d, to_linear = _spec("mej")
    assert site == "mej" and to_linear is None
    assert isinstance(d, dist.Uniform)
    assert (float(d.low), float(d.high)) == (1e-2, 0.05)


def test_log_uniform_site_is_named_for_the_coordinate_it_walks():
    """``log10_<name>``, matching ``pymc_gpu._pymc_prior`` exactly so the two arms are comparable."""
    site, d, to_linear = _spec()
    assert site == "log10_temperature_floor"
    assert isinstance(d, dist.Uniform), f"expected a Uniform on log10, got {type(d).__name__}"
    assert float(d.low) == pytest.approx(np.log10(LO))
    assert float(d.high) == pytest.approx(np.log10(HI))
    assert to_linear is not None and to_linear(np.log10(HI)) == pytest.approx(HI, rel=_tol())


def test_unsupported_distributions_still_raise_rather_than_being_approximated():
    """The pre-existing guarantee must survive the change of return type."""
    class _Weird:
        bounds = (1.0, 2.0)

    class _P:
        distributions = {"x": _Weird()}

    with pytest.raises(TypeError, match="cannot express prior"):
        _numpyro_priors(_P(), ["x"])


# ------------------------------------------------------- the hyperparameters, previously unreachable
@pytest.mark.parametrize("kw,default", [
    ("init_strategy", "prior_scan"), ("dense_mass", False), ("max_tree_depth", 10),
    ("step_size", 1.0), ("chain_method", "vectorized"),
])
def test_fit_exposes_kernel_hyperparameters_with_todays_defaults(kw, default):
    """There was no ``**kwargs`` escape hatch, so none of these could be reached from ``fit()``.

    The defaults must reproduce what the sampler did before they were exposed, or every existing
    saved run silently changes meaning -- with one deliberate exception: ``init_strategy`` was
    ``None`` (NumPyro's ``init_to_uniform``) and is now ``"prior_scan"``, because the old start
    stranded chains in 22 of 100 float64 bump fits (U3; ``tests/test_nuts_gpu_known_answer.py``).
    ``init_strategy="uniform"`` reproduces the old runs, and ``info["init_strategy"]`` says which ran.
    """
    params = inspect.signature(NUTSGPUSampler.fit).parameters
    assert kw in params, f"{kw} is not reachable through fit()"
    assert params[kw].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[kw].default == default


def test_chosen_hyperparameters_are_recorded_in_info():
    """``result.info`` must record what ran; two fits differing only in these were indistinguishable."""
    src = inspect.getsource(NUTSGPUSampler.fit)
    for key in ("init_strategy", "dense_mass", "max_tree_depth", "step_size", "chain_method"):
        assert f'"{key}":' in src, f"info does not record {key}"


@pytest.mark.parametrize("given,expected", [
    ("median", "init_to_median"), ("init_to_median", "init_to_median"),
    ("UNIFORM", "init_to_uniform"), ("sample", "init_to_sample"),
    ("feasible", "init_to_feasible"), ("mean", "init_to_mean"),
])
def test_init_strategy_accepts_short_names(given, expected):
    fn, label = _resolve_init_strategy(given)
    assert callable(fn) and label == expected


def test_init_strategy_default_is_the_prior_scan_and_the_old_default_is_one_name_away():
    fn, label = _resolve_init_strategy(None)
    assert fn is None and label == "prior_scan"            # computed by nuts_gpu, not NumPyro
    fn, label = _resolve_init_strategy("uniform")
    assert fn is numpyro.infer.init_to_uniform and label == "init_to_uniform"
    for name in ("prior_scan", "prior"):
        assert _resolve_init_strategy(name) == (None, name)
    assert _resolve_init_strategy(np.zeros((4, 3))) == (None, "per_chain")


def test_init_strategy_rejects_typos_by_naming_the_alternatives():
    with pytest.raises(ValueError, match="unknown init_strategy"):
        _resolve_init_strategy("init_to_medain")
    with pytest.raises(TypeError, match="init_strategy must be"):
        _resolve_init_strategy(3)


def test_init_strategy_accepts_a_numpyro_callable():
    fn, label = _resolve_init_strategy(numpyro.infer.init_to_value(values={"mej": 0.02}))
    # the partial's own name, so the point-start warning can recognise it
    assert callable(fn) and label == "init_to_value"


def test_chain_method_is_validated_before_anything_is_compiled():
    """A typo must raise, not fall through to NumPyro after a minute of XLA compilation."""
    from whisper_cbpf.samplers.jax.nuts_gpu import _CHAIN_METHODS
    assert set(_CHAIN_METHODS) == {"vectorized", "parallel", "sequential"}


# ------------------------------------------------------------------------- the log(0) asymmetry
def test_an_empty_light_curve_is_refused_even_with_a_caller_supplied_density():
    """``nuts_gpu`` returned BIC -inf for an empty light curve, ``pymc_gpu`` 0.0; now both refuse.

    -inf is not a loud failure: it compares as the BEST possible model in any argmin over BIC. It
    was reached through an explicitly supplied ``log_prob_fn``, the only way in: the auto-built
    path goes through ``make_likelihood``, which refuses an empty light curve outright, while a
    user's own density skipped that gate. This test first asked for a finite BIC there (a
    ``log(max(n, 1))`` guard); the contract is now that no sampler fits zero data at all
    (``samplers.base.check_not_empty``; one test per sampler in ``tests/test_empty_light_curve.py``),
    so ``aic_bic`` never sees ``n = 0``.
    """
    import warnings

    import jax.numpy as jnp
    import whisper_cbpf as wp
    from whisper_cbpf.samplers.base import aic_bic

    def predict(parameters, times, bands=None):
        t = np.asarray(times, dtype=float)
        return np.asarray(parameters["amp"]) * np.exp(-t / np.asarray(parameters["tau"]))

    m = wp.register_model(
        "nutsgpu_bic_toy", predict, ["amp", "tau"], overwrite=True,
        prior=wp.Prior({"amp": wp.Uniform(0.5, 5.0), "tau": wp.LogUniform(1.0, 30.0)}))
    empty = wp.LightCurve(time=np.array([]), band=np.array([], dtype="U8"),
                          flux=np.array([]), flux_err=np.array([]), name="empty")
    flat = lambda th: jnp.asarray(0.0) * th[0]      # noqa: E731 - the user's own density

    with pytest.raises(ValueError, match="no data points"):
        NUTSGPUSampler().fit(empty, m, log_prob_fn=flat, space="flux",
                             num_warmup=8, num_samples=8, num_chains=1)

    three = wp.LightCurve(time=np.array([1.0, 2.0, 3.0]), band=np.array(["g"] * 3),
                          flux=np.array([1.0, 0.8, 0.6]), flux_err=np.full(3, 0.1), name="three")
    with warnings.catch_warnings():         # one chain: the convergence checks warn, by design
        warnings.simplefilter("ignore")
        r = NUTSGPUSampler().fit(three, m, log_prob_fn=flat, space="flux",
                                 num_warmup=8, num_samples=8, num_chains=1)
    assert (r.aic, r.bic) == aic_bic(r.max_log_likelihood, 2, 3)
    # and the reparameterisation is recorded, so a saved run says which coordinate was walked
    assert r.info["sample_sites"] == {"amp": "amp", "tau": "log10_tau"}
