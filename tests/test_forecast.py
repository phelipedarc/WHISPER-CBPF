"""Forecasts in magnitudes and the discrimination metric D (``whisper_cbpf.forecast``).

Every number is recomputed here from the draws with numpy, on toy models whose magnitude is a
known function of the parameters. The JAX batch is checked against the per-draw CPU loop on the same
model, and the gate against redback: the JAX Arnett model and redback's Arnett, at the same
draws, give the same forecast (median per cell within 0.02 mag; in fact to ~1e-12 mag).
"""
import doctest
import importlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import whisper_cbpf as wp
from whisper_cbpf.forecast import discriminate, forecast
from whisper_cbpf.io.photometry import AB_ZEROPOINT_JY
from whisper_cbpf.models import Model
from whisper_cbpf.priors import Prior, Uniform
from whisper_cbpf.samplers.base import SamplerResult

#: The module itself: ``whisper_cbpf.forecast`` as an attribute is the function the package exports.
F = importlib.import_module("whisper_cbpf.forecast")

BANDS = ["a", "b"]
OFFSET = {"a": 0.0, "b": 0.5}


def toy_mag(p, t, bands):
    """AB magnitude m0 + slope * t (+0.5 in band b); no light after t_end."""
    off = np.array([OFFSET[str(b)] for b in np.asarray(bands)])
    m = p["m0"] + p["slope"] * np.asarray(t, float) + off
    return np.where(np.asarray(t, float) <= p["t_end"], m, np.inf)


def toy_predict(p, t, bands=None):
    m = toy_mag(p, t, bands)
    return np.where(np.isfinite(m), AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.where(np.isfinite(m), m, 0)),
                    0.0)


class _BandIndex:
    def __call__(self, bands):
        return np.array([BANDS.index(str(b)) for b in np.asarray(bands)], dtype=int)


class ToyJax:
    """``predict_jax`` of the toy: flat theta (m0, slope, t_end), integer bands."""

    band_index = _BandIndex()
    mag_floor = None

    def __init__(self, wall=None):
        self.constraint_ok = wall

    def __call__(self, theta, times, band_idx=None):
        import jax.numpy as jnp
        off = jnp.asarray([OFFSET[b] for b in BANDS])[band_idx]
        m = theta[0] + theta[1] * jnp.asarray(times) + off
        return jnp.where(jnp.asarray(times) <= theta[2], AB_ZEROPOINT_JY * 10.0 ** (-0.4 * m), 0.0)


PARAMS = ["m0", "slope", "t_end"]
PRIOR = Prior({"m0": Uniform(15, 25), "slope": Uniform(0, 0.2), "t_end": Uniform(0, 100)})
NUMPY_TOY = Model("toy_forecast", toy_predict, PARAMS, PRIOR)


@pytest.fixture(autouse=True, scope="module")
def registered_toy():
    """Results name their model; the toy is registered so a result alone can be forecast."""
    wp.register_model("toy_forecast", toy_predict, PARAMS, PRIOR, overwrite=True)


def jax_toy(wall=None):
    return Model("toy_forecast_jax", toy_predict, PARAMS, PRIOR, predict_jax=ToyJax(wall))


def result_of(samples, model="toy_forecast", n_params=3):
    samples = pd.DataFrame(samples)
    return SamplerResult(sampler="hand", model=model, parameters=list(samples.columns),
                         samples=samples, summary={}, best_params={}, n_data=10,
                         n_params=n_params, runtime_s=0.0)


def draws(n=300, seed=1, t_end=(50.0, 50.0)):
    rng = np.random.default_rng(seed)
    return {"m0": rng.normal(20.0, 0.3, n), "slope": rng.uniform(0.02, 0.08, n),
            "t_end": rng.uniform(*t_end, n)}


# --- the statistics, recomputed ---------------------------------------------------------------------

def test_every_cell_is_recomputable_from_the_draws():
    s = draws()
    times = [10.0, 20.0, 35.0]
    fc = forecast(result_of(s), times, BANDS, model=NUMPY_TOY, survey_depth={"a": 21.2, "b": 21.4})
    assert list(fc.columns) == ["time", "band", "mean_mag", "sd_mag", "q2.5", "q16", "q50", "q84",
                                "q97.5", "frac_too_faint", "frac_dark", "n_draws", "depth"]
    assert list(zip(fc["time"], fc["band"])) == [(t, b) for t in times for b in BANDS]   # time-major
    for _, row in fc.iterrows():
        m = s["m0"] + s["slope"] * row["time"] + OFFSET[row["band"]]
        assert row["mean_mag"] == pytest.approx(m.mean(), abs=1e-10)
        assert row["sd_mag"] == pytest.approx(m.std(ddof=1), abs=1e-10)
        for q in F.DEFAULT_QUANTILES:
            assert row[F._quantile_name(q)] == pytest.approx(np.quantile(m, q), abs=1e-10)
        depth = {"a": 21.2, "b": 21.4}[row["band"]]
        assert row["frac_too_faint"] == pytest.approx(np.mean(m > depth))
        assert row["depth"] == depth and row["frac_dark"] == 0.0 and row["n_draws"] == 300
    assert fc.attrs["backend"] == "numpy" and fc.attrs["n_draws"] == 300
    assert fc.attrs["survey_depth"] == {"a": 21.2, "b": 21.4}


def test_draws_that_predict_no_light_are_counted_not_averaged():
    s = draws(n=400, t_end=(10.0, 30.0))            # the light stops somewhere in 10-30 d
    fc = forecast(result_of(s), [20.0], "a", model=NUMPY_TOY, survey_depth=21.0)
    row = fc.iloc[0]
    lit = s["t_end"] >= 20.0
    m = (s["m0"] + s["slope"] * 20.0)[lit]
    assert row["frac_dark"] == pytest.approx(1 - lit.mean())
    assert row["n_draws"] == lit.sum()
    assert row["mean_mag"] == pytest.approx(m.mean(), abs=1e-10)          # over the lit draws only
    assert row["frac_too_faint"] == pytest.approx(np.mean(~lit | (s["m0"] + s["slope"] * 20 > 21)))
    # no depth: "too faint" is "no light"
    nodepth = forecast(result_of(s), [20.0], "a", model=NUMPY_TOY).iloc[0]
    assert nodepth["frac_too_faint"] == nodepth["frac_dark"] and np.isnan(nodepth["depth"])


def test_fewer_than_two_lit_draws_is_not_enough_data():
    s = draws(n=5)
    s["t_end"] = np.array([100.0, 1.0, 1.0, 1.0, 1.0])     # one draw shines at t = 20
    with pytest.warns(UserWarning, match="not enough data at 1 of 1 cells"):
        row = forecast(result_of(s), [20.0], "a", model=NUMPY_TOY).iloc[0]
    assert np.isnan(row["mean_mag"]) and np.isnan(row["sd_mag"]) and np.isnan(row["q50"])
    assert row["n_draws"] == 1 and row["frac_dark"] == pytest.approx(0.8)


def test_draws_are_a_seeded_subset_without_replacement():
    s = draws(n=1000)
    s["m0"] = np.arange(1000.0) * 1e-3 + 18.0          # each draw identifiable by its m0
    res = result_of(s)
    theta, n_total = F._draws(res, NUMPY_TOY, 400, seed=3)
    assert n_total == 1000 and theta.shape == (400, 3)
    assert len(np.unique(theta[:, 0])) == 400
    assert np.array_equal(theta, F._draws(res, NUMPY_TOY, 400, seed=3)[0])
    assert not np.array_equal(theta, F._draws(res, NUMPY_TOY, 400, seed=4)[0])
    everything, _ = F._draws(res, NUMPY_TOY, 5000, seed=0)            # fewer than asked: all
    assert np.array_equal(everything[:, 0], s["m0"])
    fc = forecast(res, [1.0], "a", model=NUMPY_TOY, n_draws=50, seed=9)
    assert fc.attrs["n_draws"] == 50 and fc.attrs["n_samples"] == 1000 and fc.attrs["seed"] == 9


def test_custom_quantiles_name_their_columns():
    fc = forecast(result_of(draws()), [1.0], "a", model=NUMPY_TOY, quantiles=(0.05, 0.5, 0.95))
    assert {"q5", "q50", "q95"} <= set(fc.columns) and "q16" not in fc.columns


def test_a_twin_reads_the_draws_through_its_aliases():
    s = draws()
    twin = Model("toy_twin", toy_predict, PARAMS, PRIOR,
                 param_aliases={"m0": "M0", "slope": "SLOPE"})
    renamed = pd.DataFrame(s).rename(columns={"m0": "M0", "slope": "SLOPE"})
    a = forecast(result_of(renamed), [5.0], "a", model=twin)
    b = forecast(result_of(s), [5.0], "a", model=NUMPY_TOY)
    pd.testing.assert_frame_equal(a, b)


def test_the_time_label_follows_the_light_curve():
    lc = wp.LightCurve(time=np.array([60000.0, 60001.0]), band=["a", "a"],
                       magnitude=[20.0, 20.1], magnitude_err=[0.1, 0.1], name="x")
    lc = lc.set_time_reference(60000.0, "first detection")
    fc = forecast(result_of(draws()), [2.0, 3.0], "a", model=NUMPY_TOY, lc=lc)
    assert fc.attrs["time_label"] == "days since first detection (MJD 60000.000)"
    assert fc.attrs["lc_name"] == "x"
    assert forecast(result_of(draws()), [2.0], "a", model=NUMPY_TOY).attrs["time_label"] is None


# --- errors that name the cause and the fix ------------------------------------------------------

def test_times_on_another_clock_are_refused_with_the_shift_to_apply():
    lc = wp.LightCurve(time=np.array([60000.0, 60010.0]), band=["a", "a"],
                       magnitude=[20.0, 20.1], magnitude_err=[0.1, 0.1])
    lc = lc.set_time_reference(60000.0, "first detection")
    with pytest.raises(ValueError, match=r"another clock.*subtract it, times - 60000.000"):
        forecast(result_of(draws()), [60011.0], "a", model=NUMPY_TOY, lc=lc)
    with pytest.warns(UserWarning, match="not enough data"):     # a year later: no refusal
        forecast(result_of(draws()), [365.0], "a", model=NUMPY_TOY, lc=lc)


@pytest.mark.parametrize("kwargs, match", [
    (dict(times=[]), "non-empty"),
    (dict(times=[1.0, np.nan]), "finite"),
    (dict(bands=[]), "bands is empty"),
    (dict(survey_depth={"a": 21.0}, bands=["a", "b"]), r"no depth for \['b'\]"),
    (dict(survey_depth="deep"), "survey_depth must be"),
    (dict(quantiles=(0.5, 1.0)), "strictly between 0 and 1"),
    (dict(n_draws=0), "n_draws must be at least 1"),
])
def test_bad_arguments_name_the_fix(kwargs, match):
    args = dict(times=[1.0], bands=["a"])
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=match):
        forecast(result_of(draws()), args.pop("times"), args.pop("bands"), model=NUMPY_TOY, **args)


def test_an_empty_posterior_is_not_enough_data():
    empty = result_of({k: np.array([]) for k in PARAMS})
    with pytest.raises(ValueError, match="not enough data: the fit of 'toy_forecast' has no posterior"):
        forecast(empty, [1.0], "a", model=NUMPY_TOY)


def test_an_unregistered_model_and_missing_columns_say_what_to_pass():
    with pytest.raises(ValueError, match="not registered in this session. Pass model="):
        forecast(result_of(draws(), model="no_such_model_here"), [1.0], "a")
    with pytest.raises(ValueError, match=r"no column for \['t_end'\].*rename"):
        forecast(result_of({k: v for k, v in draws().items() if k != "t_end"}), [1.0], "a",
                 model=NUMPY_TOY)


# --- the JAX batch ----------------------------------------------------------------------------------

@pytest.fixture
def x64():
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", was)


def test_the_jax_batch_equals_the_per_draw_loop_and_compiles_once(x64):
    s = draws(n=300, t_end=(10.0, 30.0))
    res = result_of(s)
    times = np.linspace(0.0, 40.0, 9)                         # all dark after 30 d: NaN cells too
    model = jax_toy()
    with pytest.warns(UserWarning, match="not enough data at 6 of 18 cells"):
        cpu = forecast(res, times, BANDS, model=NUMPY_TOY, survey_depth=21.0)
    with pytest.warns(UserWarning, match="not enough data at 6 of 18 cells"):
        gpu = forecast(res, times, BANDS, model=model, survey_depth=21.0)
    assert gpu.attrs["backend"] == "jax"
    num = [c for c in cpu.columns if c not in ("band",)]
    np.testing.assert_allclose(gpu[num].to_numpy(float), cpu[num].to_numpy(float), rtol=0,
                               atol=1e-10)
    n_cached = len(F._JAX_CACHE)
    with pytest.warns(UserWarning, match="not enough data"):
        again = forecast(res, times, BANDS, model=model, survey_depth=21.0, seed=5)
    assert len(F._JAX_CACHE) == n_cached                     # same epochs and bands: no new compile
    assert again.attrs["backend"] == "jax"


def test_many_draws_run_in_equal_blocks_with_the_same_answer(x64, monkeypatch):
    res = result_of(draws(n=50))
    model = jax_toy()
    whole = forecast(res, [5.0, 15.0], BANDS, model=model)
    monkeypatch.setattr(F, "MAX_JAX_BLOCK", 7)                # 50 draws = 8 blocks, last one padded
    blocks = forecast(res, [5.0, 15.0], BANDS, model=model)
    pd.testing.assert_frame_equal(whole, blocks, check_exact=False, atol=1e-12)


def test_the_constraint_wall_darkens_a_walled_draw_on_the_jax_path(x64):
    s = draws(n=100)
    model = jax_toy(wall=lambda theta: theta[0] < 20.0)       # draws with m0 >= 20 are walled
    row = forecast(result_of(s), [5.0], "a", model=model).iloc[0]
    walled = s["m0"] >= 20.0
    assert row["frac_dark"] == pytest.approx(walled.mean())
    assert row["mean_mag"] == pytest.approx((s["m0"] + 5.0 * s["slope"])[~walled].mean(), abs=1e-10)


# --- discriminate -----------------------------------------------------------------------------------

def two_models():
    rng = np.random.default_rng(7)
    a = {"m0": rng.normal(20.0, 0.2, 400), "slope": np.full(400, 0.05), "t_end": np.full(400, 100.0)}
    b = {"m0": rng.normal(20.0, 0.3, 400), "slope": np.full(400, 0.10), "t_end": np.full(400, 100.0)}
    return a, b


def test_D_is_the_gap_over_the_combined_spread():
    a, b = two_models()
    ra, rb = result_of(a), result_of(b)
    times = [0.0, 10.0, 30.0]
    fa = forecast(ra, times, BANDS, model=NUMPY_TOY)
    fb = forecast(rb, times, BANDS, model=NUMPY_TOY)
    d = discriminate({"A": fa, "B": fb}, times, BANDS, phot_sigma=0.05)
    assert list(d.columns) == ["model_a", "model_b", "time", "band", "mean_a", "mean_b", "sd_a",
                               "sd_b", "sd_phot", "D", "observable"]
    want = np.abs(fa["mean_mag"] - fb["mean_mag"]) / np.sqrt(
        fa["sd_mag"] ** 2 + fb["sd_mag"] ** 2 + 0.05 ** 2)
    np.testing.assert_allclose(d["D"], want, rtol=1e-12)
    assert d["observable"].all()
    best = d.attrs["best"]
    assert best["time"] == 30.0 and best["D"] == pytest.approx(want.max())
    assert best["model_a"] == "A" and best["model_b"] == "B"


def test_the_depth_sets_the_photometric_error_and_what_is_observable():
    a, b = two_models()
    fa = forecast(result_of(a), [0.0, 30.0], "a", model=NUMPY_TOY, survey_depth=22.0)
    fb = forecast(result_of(b), [0.0, 30.0], "a", model=NUMPY_TOY, survey_depth=22.0)
    d = discriminate({"A": fa, "B": fb}, [0.0, 30.0], "a", survey_depth=22.0)
    m = 0.5 * (fa["mean_mag"] + fb["mean_mag"])
    sp = F.SIGMA_AT_LIMIT * 10 ** (0.4 * (m - 22.0))
    np.testing.assert_allclose(d["sd_phot"], sp, rtol=1e-12)
    assert F.SIGMA_AT_LIMIT == pytest.approx(1.0857 / 5, abs=1e-4)
    # at t = 30 model B is at ~23 mag, fainter than the depth: not observable, so not "best"
    assert list(d["observable"]) == [True, False]
    assert d.attrs["best"]["time"] == 0.0


def test_no_observable_cell_gives_a_reason_not_a_number():
    a, b = two_models()
    ra, rb = result_of(a), result_of(b)
    d = discriminate({"A": ra, "B": rb}, [0.0, 30.0], "a", survey_depth=15.0)   # all too faint
    assert d.attrs["best"] is None
    assert "no cell where both models" in d.attrs["reason"]


def test_results_are_forecast_by_their_model_and_empty_fits_are_left_out():
    a, b = two_models()
    empty = result_of({k: np.array([]) for k in PARAMS})
    d = discriminate({"A": result_of(a), "B": result_of(b), "C": empty}, [5.0], "a")
    assert d.attrs["left_out"] == {"C": "no posterior draws"}
    assert set(zip(d["model_a"], d["model_b"])) == {("A", "B")}
    with pytest.raises(ValueError, match="fewer than two models have posterior draws"):
        discriminate({"A": result_of(a), "C": empty}, [5.0], "a")
    with pytest.raises(ValueError, match="at least two models"):
        discriminate({"A": result_of(a)}, [5.0], "a")


def test_a_forecast_for_other_cells_or_another_depth_is_refused():
    a, b = two_models()
    fa = forecast(result_of(a), [5.0], "a", model=NUMPY_TOY)
    fb = forecast(result_of(b), [6.0], "a", model=NUMPY_TOY)
    with pytest.raises(ValueError, match="other cells than times x bands"):
        discriminate({"A": fa, "B": fb}, [5.0], "a")
    fb = forecast(result_of(b), [5.0], "a", model=NUMPY_TOY)
    with pytest.raises(ValueError, match="made with survey_depth=None"):
        discriminate({"A": fa, "B": fb}, [5.0], "a", survey_depth=21.0)
    with pytest.raises(ValueError, match="phot_sigma must be finite"):
        discriminate({"A": fa, "B": fb}, [5.0], "a", phot_sigma=-1.0)


def test_sampler_result_forecast_delegates_here():
    if not hasattr(SamplerResult, "forecast"):
        pytest.skip("SamplerResult.forecast is added by the samplers/base.py owner")
    res = result_of(draws())
    pd.testing.assert_frame_equal(res.forecast([1.0, 2.0], BANDS),
                                  forecast(res, [1.0, 2.0], BANDS))


def test_the_docstring_examples_run():
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in (forecast, discriminate):
            for test in doctest.DocTestFinder().find(obj, obj.__name__, globs={}):
                runner.run(test, out=io.StringIO().write)
    assert runner.tries > 0 and runner.failures == 0


# --- the gate: the JAX Arnett forecast against redback's, at the same draws ---------------------------

#: SN2025pgp's 30-day ZTF cut (tests/test_likelihood_max_opt.py), days since first detection + 3 d;
#: z = 0.051.
PGP_LAST = 28.958101799711585 + 3.0
PGP_Z = 0.051
#: The best draw of the SN2025pgp reference chain (tests/test_likelihood_max_opt.py REF_BEST).
PGP_BEST = {"f_nickel": 0.9813512249856495, "mej": 0.45026143953444986, "vej": 8207.61852161502,
            "kappa": 0.052541421376849515, "kappa_gamma": 0.011243343382461652,
            "temperature_floor": 6850.259466620045}
HORIZONS = np.array([0.25, 0.5, 1.0, 2.0, 3.0])


@pytest.fixture(scope="module")
def arnett_pair():
    pytest.importorskip("redback")
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    from whisper_cbpf.models import redback_adapter as ra
    bands = ["ztfg", "ztfr", "ztfi"]
    cpu = ra.redback_model("arnett", band_names=bands, redshift=PGP_Z)
    gpu = wp.supernova_model("arnett", bands, PGP_Z, ra.redback_luminosity_distance_cm(PGP_Z))
    yield cpu, gpu, bands
    jax.config.update("jax_enable_x64", was)


def _posterior_like(prior, centre, n, seed):
    """A cloud around ``centre``: 3 % of each prior width (log10 for a LogUniform), inside the box."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, dist in prior.distributions.items():
        lo, hi = dist.bounds
        log = type(dist).__name__ == "LogUniform"
        c, a, b = (np.log10([centre[name], lo, hi]) if log else (centre[name], lo, hi))
        x = np.clip(rng.normal(c, 0.03 * (b - a), n), a, b)
        out[name] = 10 ** x if log else x
    return pd.DataFrame(out)


@pytest.mark.parametrize("which", ["posterior", "prior"])
def test_gate_jax_arnett_forecast_equals_redback_at_the_same_draws(arnett_pair, which):
    cpu, gpu, bands = arnett_pair
    prior = cpu.default_prior
    if which == "posterior":
        samples = _posterior_like(prior, PGP_BEST, 400, seed=0)
    else:
        rng = np.random.default_rng(0)
        samples = pd.DataFrame([prior.sample(rng) for _ in range(400)])[cpu.parameters]
    res = result_of(samples, model=cpu.name, n_params=6)
    times = PGP_LAST + HORIZONS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_cpu = forecast(res, times, bands, model=cpu, survey_depth=20.5)
        f_gpu = forecast(res, times, bands, model=gpu, survey_depth=20.5)
    assert (f_cpu.attrs["backend"], f_gpu.attrs["backend"]) == ("numpy", "jax")
    assert len(f_gpu) == 15
    # the gate: median per cell within 0.02 mag
    dq50 = np.abs(f_gpu["q50"] - f_cpu["q50"]).to_numpy()
    assert np.all(dq50 <= 0.02), dq50
    # and, measured: every statistic agrees to round-off
    for col in ("mean_mag", "sd_mag", "q2.5", "q16", "q50", "q84", "q97.5"):
        np.testing.assert_allclose(f_gpu[col], f_cpu[col], rtol=0, atol=1e-8, err_msg=col)
    for col in ("frac_too_faint", "frac_dark", "n_draws"):
        np.testing.assert_array_equal(f_gpu[col], f_cpu[col])
    # per draw, too: the same magnitudes
    theta = samples[cpu.parameters].to_numpy(float)
    t = np.repeat(times, 3)
    b = np.tile(bands, 5)
    m_cpu, d_cpu, _ = F._draw_magnitudes(cpu, theta, t, b)
    m_gpu, d_gpu, _ = F._draw_magnitudes(gpu, theta, t, b)
    assert np.array_equal(d_cpu, d_gpu)
    lit = ~d_cpu
    assert np.max(np.abs(m_gpu[lit] - m_cpu[lit])) < 1e-8
