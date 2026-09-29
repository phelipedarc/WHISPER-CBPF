"""check_parity: two models, the same parameters, the same magnitudes?

The first group uses cheap numpy models with a known, injected difference, so every number the report
gives is known in advance. The second group reproduces a reference redback comparison
(SN2025pgp's 28 ZTF epochs at +3 d, 200 prior draws
at seed 0, both models built with ``constraint=None``) and checks its numbers come back:

* JAX ``arnett`` against the redback adapter: max |dmag| 5.33e-13 mag (0.1.1);
* JAX ``basic_magnetar_powered`` on the pre-0.1.1 linear grid (``spacing="linear"``): the 1.1 defect,
  max 18.3 mag over the points compared (redback brighter than 30 mag), 20.5 mag over
  every point where either model is;
* the redback adapter's pre-0.1.1 ``photometry="monochromatic"`` against JAX ``arnett``: the
  band-integral defect, max 0.546 mag.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.models import Model
from whisper_cbpf.samplers.base import SamplerResult

T = np.linspace(-10.0, 60.0, 30)
B = np.where(np.arange(30) % 2 == 0, "g", "r")
BAZIN = wp.get_model("bazin")
POINT = {"amplitude": 5.0, "t0": 5.0, "tau_rise": 3.0, "tau_fall": 20.0}


def _scaled(name, factor_by_band, base=BAZIN, parameters=None, aliases=None):
    """``base`` times a per-band flux factor, as a new Model."""
    def predict(params, times, bands):
        f = np.array([factor_by_band.get(str(x), 1.0) for x in np.asarray(bands)])
        return f * base.predict(params, times, bands)
    return Model(name, predict, list(parameters or base.parameters), default_prior=base.default_prior,
                 param_aliases=dict(aliases or {}))


# --- numbers known in advance -----------------------------------------------------------------------

def test_identical_models_agree_exactly():
    r = wp.check_parity(BAZIN, "bazin", BAZIN.default_prior, T, B, n=20)
    assert r.passed and r.max_abs == 0.0 and r.n_draws == 20
    assert set(r.per_band) == {"g", "r"} and all(s["passed"] for s in r.per_band.values())
    assert "<= tolerance" in r.reason


def test_a_flux_error_in_one_band_is_measured_there_only():
    off = _scaled("bazin_r_plus_2pc", {"r": 1.02})
    r = wp.check_parity(BAZIN, off, BAZIN.default_prior, T, B, n=20)
    expected = 2.5 * np.log10(1.02)                                    # 0.0215 mag, a -> brighter b
    assert not r.passed
    assert r.per_band["r"]["max_abs"] == pytest.approx(expected, abs=1e-12)
    assert r.per_band["r"]["median_signed"] == pytest.approx(expected, abs=1e-12)
    assert r.per_band["g"]["max_abs"] == 0.0 and r.per_band["g"]["passed"]
    assert r.worst[0]["band"] == "r" and "band r" in r.reason
    # the same models pass a looser bar
    assert wp.check_parity(BAZIN, off, BAZIN.default_prior, T, B, n=20, tolerance=0.03).passed


def test_one_model_dark_where_the_other_is_bright_fails_with_an_infinite_difference():
    def late_dark(params, times, bands):
        f = BAZIN.predict(params, times, bands)
        return np.where(np.asarray(times) > 40.0, 0.0, f)
    dark = Model("bazin_dark_late", late_dark, BAZIN.parameters)
    r = wp.check_parity(BAZIN, dark, POINT, T, B)
    assert not r.passed and np.isinf(r.max_abs)
    assert r.worst[0]["time"] > 40.0 and np.isinf(r.worst[0]["mag_b"])
    assert "bazin_dark_late predicts no flux" in r.reason
    assert np.isfinite(r.median_abs)                   # most points agree: the median is 0, not NaN


def test_nothing_bright_enough_says_not_enough_data():
    def faint(params, times, bands):
        return np.full(np.shape(times), 1e-15)        # 46 mag
    m = Model("faint", faint, BAZIN.parameters)
    r = wp.check_parity(m, m, POINT, T, B)
    assert not r.passed and r.n_points == 0 and np.isnan(r.max_abs)
    assert r.reason.startswith("Not enough data")


def test_a_nonfinite_flux_fails_even_when_the_other_points_agree():
    def one_nan(params, times, bands):
        f = BAZIN.predict(params, times, bands).astype(float)
        f[3] = np.nan
        return f
    m = Model("bazin_nan", one_nan, BAZIN.parameters)
    r = wp.check_parity(BAZIN, m, POINT, T, B)
    assert not r.passed and r.nonfinite == {"a": 0, "b": 1}
    assert "Non-finite flux" in r.reason and "bazin_nan at 1 points" in r.reason


def test_worst_draws_are_ranked_and_located():
    # b is off only where amplitude > 8, by a factor that grows with the amplitude
    def grows(params, times, bands):
        a = params["amplitude"]
        return (1.0 + 0.01 * max(a - 8.0, 0.0)) * BAZIN.predict(params, times, bands)
    m = Model("bazin_grows", grows, BAZIN.parameters)
    r = wp.check_parity(BAZIN, m, BAZIN.default_prior, T, B, n=50, seed=3)
    maxes = [w["max_abs_dmag"] for w in r.worst]
    assert len(r.worst) == 5 and maxes == sorted(maxes, reverse=True)
    amps = [w["params"]["amplitude"] for w in r.worst]
    assert amps == sorted(amps, reverse=True) and amps[0] > 8.0
    top = r.worst[0]
    assert top["max_abs_dmag"] == pytest.approx(2.5 * np.log10(1.0 + 0.01 * (amps[0] - 8.0)), rel=1e-9)
    assert r.dmag.shape == (50, 30)


# --- where the parameters come from ------------------------------------------------------------------

def test_prior_draws_are_the_reference_draws():
    r = wp.check_parity(BAZIN, BAZIN, BAZIN.default_prior, T, B, n=7, seed=11)
    rng = np.random.default_rng(11)
    first = [BAZIN.default_prior.sample(rng) for _ in range(7)]
    assert r.n_draws == 7 and r.draws == "7 prior draws (seed 11)"
    assert [w["draw"] for w in r.worst] == [0, 1, 2, 3, 4]            # all agree: stable order
    assert r.worst[0]["params"] == pytest.approx(first[0])


def test_posterior_dataframe_and_sampler_result():
    rng = np.random.default_rng(0)
    post = pd.DataFrame({k: v * (1 + 0.01 * rng.normal(size=500)) for k, v in POINT.items()})
    r = wp.check_parity(BAZIN, BAZIN, post, T, B, n=40, seed=1)
    assert r.n_draws == 40 and r.draws == "40 posterior draws"
    assert wp.check_parity(BAZIN, BAZIN, post.head(10), T, B, n=40).n_draws == 10    # all rows if fewer
    res = SamplerResult(sampler="x", model="bazin", parameters=list(POINT), samples=post, summary={},
                        best_params=dict(POINT), n_data=30, n_params=4, runtime_s=0.0)
    r = wp.check_parity(BAZIN, BAZIN, res, T, B, n=25)
    assert r.n_draws == 25 and r.draws == "the best fit and 24 posterior draws"
    assert wp.check_parity(BAZIN, BAZIN, [POINT, POINT], T, B).n_draws == 2
    assert wp.check_parity(BAZIN, BAZIN, POINT, T, B, n=99).n_draws == 1


def test_parameters_named_differently_pair_through_param_aliases():
    renamed = {"A": "amplitude", "t_peak": "t0", "rise": "tau_rise", "fall": "tau_fall"}

    def predict(params, times, bands):
        return BAZIN.predict({renamed[k]: v for k, v in params.items()}, times, bands)
    twin = Model("bazin_renamed", predict, list(renamed), param_aliases=renamed)
    r = wp.check_parity(twin, BAZIN, BAZIN.default_prior, T, B, n=10)   # draws in bazin's names
    assert r.passed and r.max_abs == 0.0
    r = wp.check_parity(BAZIN, twin, BAZIN.default_prior, T, B, n=10)   # either order
    assert r.passed
    no_alias = Model("bazin_renamed_bare", predict, list(renamed))
    with pytest.raises(ValueError, match=r"needs \['A', 't_peak', 'rise', 'fall'\].*param_aliases"):
        wp.check_parity(no_alias, BAZIN, POINT, T, B)


# --- inputs ---------------------------------------------------------------------------------------------

def test_epoch_order_does_not_matter():
    off = _scaled("bazin_r_plus_1pc", {"r": 1.01})
    perm = np.random.default_rng(2).permutation(T.size)
    r1 = wp.check_parity(BAZIN, off, BAZIN.default_prior, T, B, n=10)
    r2 = wp.check_parity(BAZIN, off, BAZIN.default_prior, T[perm], B[perm], n=10)
    assert np.array_equal(r1.times, r2.times) and np.array_equal(r1.dmag, r2.dmag, equal_nan=True)


def test_single_band_label_and_band_free_models():
    assert wp.check_parity(BAZIN, BAZIN, POINT, T, "r").per_band["r"]["n"] == 30
    flare = wp.get_model("flare")
    p = flare.default_prior.sample(np.random.default_rng(0))
    r = wp.check_parity(flare, flare, p, np.linspace(0.0, 30.0, 20), None)
    assert r.per_band == {} and r.bands is None and "all" in r.table().index


@pytest.mark.parametrize("kwargs, message", [
    (dict(times=[], bands=[]), "times is empty"),
    (dict(times=T, bands=B[:5]), "one band per epoch"),
    (dict(times=[0.0, np.nan], bands=["g", "g"]), "NaN or infinite"),
    (dict(times=T, bands=B, tolerance=0.0), "positive number of magnitudes"),
    (dict(times=T, bands=B, n=0), "positive integer"),
])
def test_bad_inputs_name_the_fix(kwargs, message):
    with pytest.raises(ValueError, match=message):
        wp.check_parity(BAZIN, BAZIN, POINT, **kwargs)


def test_unusable_parameter_sources_raise():
    with pytest.raises(TypeError, match="parameter dict, a list of dicts"):
        wp.check_parity(BAZIN, BAZIN, 3.0, T, B)
    with pytest.raises(ValueError, match="no rows"):
        wp.check_parity(BAZIN, BAZIN, pd.DataFrame(columns=list(POINT)), T, B)
    empty = SamplerResult(sampler="abc", model="bazin", parameters=list(POINT),
                          samples=pd.DataFrame(columns=list(POINT)), summary={}, best_params={},
                          n_data=30, n_params=4, runtime_s=0.0)
    with pytest.raises(ValueError, match="no accepted draw"):
        wp.check_parity(BAZIN, BAZIN, empty, T, B)


def test_repr_and_table_say_what_to_read():
    r = wp.check_parity(BAZIN, _scaled("off", {"r": 1.05}), POINT, T, B)
    text = repr(r)
    assert "bazin vs off -- FAILED" in text and ".worst" in text and r.reason in text
    tab = r.table()
    assert list(tab.index) == ["g", "r", "all"]
    assert list(tab.columns) == ["n", "median_abs", "p99_abs", "max_abs", "median_signed", "passed"]


def test_docstring_example_runs():
    import doctest

    from whisper_cbpf import validation
    runner = doctest.DocTestRunner()
    for test in doctest.DocTestFinder().find(validation.check_parity, "check_parity", globs={}):
        runner.run(test)
    assert runner.tries > 0 and runner.failures == 0


# --- the reference redback comparison, reproduced -----------------------------------------------------

#: SN2025pgp's ZTF detections, days since first detection.
ZTF_T_REL = [
    0.0, 1.983576299622655, 2.045497599989176, 4.02018509991467, 6.026076299611304, 6.046134199947119,
    8.058634199667722, 8.109351799823344, 9.948946699965745, 10.000925899948925, 11.994756899774075,
    12.041388799894776, 13.992604099679738, 14.041504599619657, 15.894803199917078, 15.974085599649698,
    17.94416659977287, 17.98112259991467, 19.936226799618453, 19.978946699760854, 22.978634199600492,
    23.02064809994772, 24.972233799751848, 25.040080999955535, 26.957592599559575, 27.01650459971279,
    28.916724499780685, 28.958101799711585]
ZTF_BAND = ["ztfg", "ztfg", "ztfr", "ztfg", "ztfr", "ztfg", "ztfr", "ztfg", "ztfg", "ztfr", "ztfr", "ztfg",
            "ztfr", "ztfg", "ztfg", "ztfr", "ztfg", "ztfr", "ztfg", "ztfr", "ztfr", "ztfg", "ztfr", "ztfg",
            "ztfr", "ztfg", "ztfg", "ztfr"]
ZTF_Z = 0.051


@pytest.fixture(scope="module")
def redback_pair():
    """The arnett/magnetar pair builder, on JAX-CPU in float64."""
    pytest.importorskip("redback")
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    from whisper_cbpf.models import redback_adapter as ra
    from whisper_cbpf.models.jax import _factories as F

    t = np.asarray(ZTF_T_REL) + 3.0
    bands = np.asarray(ZTF_BAND)

    def build(family, cpu_kw=None, **jax_kw):
        cpu = ra.redback_model(family, band_names=["ztfg", "ztfr"], redshift=ZTF_Z, constraint=None,
                               **(cpu_kw or {}))
        dl = ra.redback_luminosity_distance_cm(ZTF_Z, model=family)
        gpu = F.supernova_model(family, ["ztfg", "ztfr"], ZTF_Z, dl, constraint=None, **jax_kw)
        return gpu, cpu

    yield build, t, bands
    jax.config.update("jax_enable_x64", was)


def test_redback_pair_arnett_jax_equals_the_redback_adapter(redback_pair):
    build, t, bands = redback_pair
    gpu, cpu = build("arnett")
    r = wp.check_parity(gpu, cpu, cpu.default_prior, t, bands)
    # reference run, arnett on ZTF, CPU against JAX: n 3951, median 0, p99 4.26e-14, max 5.33e-13
    assert r.passed and r.n_points == 3951
    assert r.max_abs < 1e-11 and r.median_abs < 1e-13


def test_redback_pair_magnetar_linear_grid_defect_is_caught(redback_pair):
    build, t, bands = redback_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # the spin-down warning is the point, not tested
        old, cpu = build("basic_magnetar_powered", spacing="linear")
        new, _ = build("basic_magnetar_powered")
        r_old = wp.check_parity(old, cpu, cpu.default_prior, t, bands)
        r_new = wp.check_parity(new, cpu, cpu.default_prior, t, bands)
    # reference run before 0.1.1, magnetar on ZTF, JAX against redback: max 18.34 mag over redback-bright points
    assert not r_old.passed and r_old.max_abs > 10.0
    # the reference compares only where redback is brighter than 30 mag; the same draws, masked so
    rng = np.random.default_rng(0)
    ref_bright = np.array([cpu.predict(cpu.default_prior.sample(rng), t, bands) for _ in range(200)]) \
        > wp.validation.AB_ZEROPOINT_JY * 10 ** (-0.4 * 30.0)
    bright_max = np.nanmax(np.abs(np.where(ref_bright, r_old.dmag, np.nan)))
    assert bright_max == pytest.approx(18.34, abs=0.01)
    assert r_old.worst[0]["mag_a"] < r_old.worst[0]["mag_b"]          # the linear grid is too bright
    # the geometric grid (0.1.1 default): CPU against JAX max 4.69e-13
    assert r_new.passed and r_new.max_abs < 1e-11


def test_redback_pair_monochromatic_photometry_defect_is_caught(redback_pair):
    build, t, bands = redback_pair
    gpu, mono = build("arnett", cpu_kw=dict(photometry="monochromatic"))
    r = wp.check_parity(gpu, mono, mono.default_prior, t, bands)
    # reference run before 0.1.1, arnett on ZTF, CPU against JAX: median 0.0117, max 0.546
    assert not r.passed
    assert r.max_abs == pytest.approx(0.546, abs=0.001) and r.median_abs == pytest.approx(0.0117, abs=5e-4)
