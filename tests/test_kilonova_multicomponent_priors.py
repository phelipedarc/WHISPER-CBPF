"""The multi-component kilonova defaults: Villar+2017 opacities, disjoint by construction.

What makes a kilonova component *blue*, *purple* or *red* is not its label -- it is the opacity
corridor its prior allows. Villar et al. 2017 (ApJL 851, L21) fit GW170817 with three components at
kappa = 0.5, 3 and 10 cm^2 g^-1. These tests pin the corridors that bracket those values, and pin
the property that matters statistically: the corridors do not overlap, so the components cannot be
relabelled and the posterior has no mirror modes.

The regression these guard against is real and was measured. redback's own two-component prior gives
BOTH components kappa U(1, 30). It therefore cannot contain the published AT2017GFO solution --
Villar's kappa_blue = 0.5 is below its floor -- and a fit inside it rails. On 53 g-band points the
best chi-square over 400 random prior draws was 38450 under the symmetric box against 1574 with the
blue corridor restored, a factor of 24, and only the split reached the observed brightness at all.
"""
import pytest

pytest.importorskip("jax")


def _prior_of(model):
    return {n: model.default_prior.distributions[n] for n in model.parameters}


def _register(n_components):
    import whisper_cbpf as wp

    kw = dict(redshift=0.0098, dl_cm=1.23e26, n_wave=64, band_aliases={"g": "sdssg"})
    if n_components == 2:
        return wp.register_kilonova_two(["sdssg"], name="_t_kn2", **kw)
    return wp.register_kilonova_three(["sdssg"], name="_t_kn3", **kw)


@pytest.mark.parametrize("n_components,expected", [
    (2, {"kappa_blue": (0.1, 1.0), "kappa_red": (1.0, 30.0)}),
    (3, {"kappa_blue": (0.1, 1.0), "kappa_purple": (1.0, 5.0), "kappa_red": (5.0, 30.0)}),
])
def test_default_opacity_corridors_are_villar(n_components, expected):
    """The shipped default prior must be Villar+2017's, not redback's symmetric box."""
    p = _prior_of(_register(n_components))
    for name, (lo, hi) in expected.items():
        assert (float(p[name].low), float(p[name].high)) == (lo, hi), (
            f"{name} is {p[name].low}-{p[name].high}, not Villar's {lo}-{hi}")


@pytest.mark.parametrize("n_components,fixed", [
    (2, {"kappa_blue": 0.5, "kappa_red": 10.0}),
    (3, {"kappa_blue": 0.5, "kappa_purple": 3.0, "kappa_red": 10.0}),
])
def test_corridors_bracket_villars_fixed_opacities(n_components, fixed):
    """Villar+2017 held kappa at 0.5 / 3 / 10; a corridor that excludes its own anchor is wrong.

    This is the check that would have caught redback's box: kappa U(1, 30) cannot reach 0.5.
    """
    p = _prior_of(_register(n_components))
    for name, k in fixed.items():
        assert float(p[name].low) <= k <= float(p[name].high), (
            f"{name} corridor {p[name].low}-{p[name].high} excludes Villar's fixed {k}")


@pytest.mark.parametrize("n_components", [2, 3])
def test_opacity_corridors_are_disjoint(n_components):
    """Overlapping corridors make components exchangeable, and an exchangeable posterior is a
    mixture of relabellings -- r-hat then measures which labelling each chain fell into rather
    than convergence, and a per-parameter median is a median of two modes."""
    p = _prior_of(_register(n_components))
    spans = [(n, float(p[n].low), float(p[n].high)) for n in p if n.startswith("kappa")]
    spans.sort(key=lambda s: s[1])
    for (n_a, _, hi_a), (n_b, lo_b, _) in zip(spans, spans[1:]):
        assert hi_a <= lo_b, f"{n_a} and {n_b} overlap, so the components can label-switch"


@pytest.mark.parametrize("n_components,k", [(2, 8), (3, 12)])
def test_prior_names_match_parameters_exactly(n_components, k):
    """Order matters: samplers build theta positionally from ``model.parameters``.

    A prior whose keys are the same set in a different order silently fits a permuted model.
    """
    m = _register(n_components)
    assert len(m.parameters) == k
    assert list(m.default_prior.names) == list(m.parameters)


@pytest.mark.parametrize("n_components", [2, 3])
def test_sigma_is_not_a_model_parameter(n_components):
    """Villar's white-noise term belongs to the likelihood, not the model.

    ``villar_prior`` offers it with ``with_sigma=True``; the model default must NOT include it,
    because a prior column the model cannot consume is refused by the JAX density adapter.
    """
    m = _register(n_components)
    assert "sigma" not in m.parameters
    assert "sigma" not in list(m.default_prior.names)


@pytest.mark.parametrize("n_components", [2, 3])
def test_predicts_finite_positive_flux_at_the_prior_midpoint(n_components):
    import numpy as np

    m = _register(n_components)
    mid = {}
    for n in m.parameters:
        d = m.default_prior.distributions[n]
        lo, hi = float(d.low), float(d.high)
        mid[n] = float(np.sqrt(lo * hi)) if type(d).__name__ == "LogUniform" else 0.5 * (lo + hi)
    t = np.linspace(0.5, 10.0, 12)
    f = np.asarray(m.predict(mid, t, np.array(["g"] * 12)), dtype=float)
    assert np.all(np.isfinite(f)) and np.all(f > 0)


def test_redback_box_is_still_available_for_comparison():
    """The symmetric prior is not deleted -- it is the reference for an apples-to-apples redback
    comparison, and that is the only thing it is for."""
    from whisper_cbpf.models.jax import kilonova_two as kn2

    d = kn2.default_prior().distributions
    assert (float(d["kappa_1"].low), float(d["kappa_1"].high)) == (1.0, 30.0)
    assert (float(d["kappa_2"].low), float(d["kappa_2"].high)) == (1.0, 30.0)
