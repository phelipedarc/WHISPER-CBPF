"""Explosion time and redshift as ordinary free parameters.

The claims, each a test:

1. **The luminosity distance is traceable and right**: the Planck18 table matches astropy to
   1e-5 mag over the whole range, under ``jit``, and refuses a redshift outside it.
2. **Freeing a value changes nothing but where it comes from.** With ``t_exp`` and the redshift
   at the truth, the free model equals the model with them baked in to ~1e-14 mag, for every
   supernova, the TDE and the three kilonovae (measured: 1.4e-14 for five families).
3. **One compilation** under ``jit`` + ``vmap`` over 200 walkers with 200 different redshifts and
   explosion times, all finite.
4. **redback at identical z and t_exp**: the free supernova model against redback's own light
   curve (the CPU adapter, band-integrated on the same filter set) within the S2 budget, 2e-3 mag.
5. **The redshift hint is consumed**: ``LightCurve.redshift_prior`` becomes the default prior.
6. **Errors name the cause and the fix**, and pre-explosion epochs are dark with finite gradients.

FLOAT64: the supernova and TDE ports require it; the fixture turns it on for this module.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.cosmology import Z_MAX, Z_MIN, luminosity_distance_cm  # noqa: E402
from whisper_cbpf.priors import LogUniform, Prior, Uniform  # noqa: E402

BANDS = ["lsstg", "lsstr", "lssti"]
T_EXP, Z = -5.0, 0.08
TIMES = np.sort(np.random.default_rng(0).uniform(0.0, 45.0, 24))
BAND_OF = np.array(BANDS * 8)
SN_MODELS = ("arnett", "shock_cooling_and_arnett", "basic_magnetar_powered", "slsn",
             "magnetar_nickel", "csm_shock_and_arnett", "sn_exponential_powerlaw", "sn_fallback",
             "sn_nickel_fallback", "general_magnetar_slsn", "type_1a", "type_1c")
FREE_PRIOR = Prior({"t_exp": Uniform(-20.0, 0.0)})
Z_HINT = {"type": "Uniform", "low": 0.01, "high": 0.3}


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _mag(flux):
    return -2.5 * np.log10(np.asarray(flux, dtype=float) / 3631.0)


def _factories():
    from whisper_cbpf.models.jax import _factories as F
    return F


# --- 1. the distance -------------------------------------------------------------------------

def test_luminosity_distance_matches_astropy_to_1e5_mag():
    import astropy.units as u
    from astropy.cosmology import Planck18

    rng = np.random.default_rng(3)
    z = np.exp(rng.uniform(np.log(Z_MIN), np.log(Z_MAX), 20000))
    ref = Planck18.luminosity_distance(z).to_value(u.cm)
    err = 5.0 * np.abs(np.log10(luminosity_distance_cm(z) / ref))
    assert err.max() < 1e-5, err.max()
    traced = np.asarray(jax.jit(lambda x: luminosity_distance_cm(x, xp=jnp))(jnp.asarray(z)))
    np.testing.assert_allclose(traced, luminosity_distance_cm(z), rtol=1e-13)
    grad = jax.grad(lambda x: luminosity_distance_cm(x, xp=jnp))(0.1)
    assert np.isfinite(float(grad)) and float(grad) > 0


def test_luminosity_distance_refuses_a_redshift_outside_the_table():
    for bad in (0.0, -0.1, 11.0, np.nan):
        with pytest.raises(ValueError, match="outside the luminosity-distance table"):
            luminosity_distance_cm(bad)


# --- 2. free equals baked-in -----------------------------------------------------------------

@pytest.mark.parametrize("model", SN_MODELS)
def test_supernova_free_equals_baked_in(model):
    F = _factories()
    free = F.supernova_model(model, BANDS, free=["t_exp", "redshift"], prior=FREE_PRIOR,
                             redshift_prior=Z_HINT, constraint=None, max_phase_days=150.0)
    baked = F.supernova_model(model, BANDS, Z, luminosity_distance_cm(Z), t_exp_days=T_EXP,
                              diffusion_grid="fixed", max_phase_days=150.0, constraint=None)
    assert free.parameters == baked.parameters + ["t_exp", "redshift"]
    rng = np.random.default_rng(1)
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # spin-down notes on odd draws
        for _ in range(3):
            p = baked.default_prior.sample(rng)
            m_b = _mag(baked.predict(p, TIMES, BAND_OF))
            m_f = _mag(free.predict({**p, "t_exp": T_EXP, "redshift": Z}, TIMES, BAND_OF))
            worst = max(worst, float(np.max(np.abs(m_b - m_f))))
    assert worst < 1e-12, (model, worst)


@pytest.mark.parametrize("kind", ["tde", "kilonova", "kilonova_two", "kilonova_three"])
def test_tde_and_kilonova_free_equal_baked_in(kind):
    F = _factories()
    build = {"tde": lambda **k: F.tde_model(BANDS, constraint=None, **k),
             "kilonova": lambda **k: F.kilonova_model(BANDS, **k),
             "kilonova_two": lambda **k: F.kilonova_two_model(BANDS, **k),
             "kilonova_three": lambda **k: F.kilonova_three_model(BANDS, **k)}[kind]
    times = TIMES if kind == "tde" else TIMES / 8.0
    free = build(free=["t_exp", "redshift"], prior=FREE_PRIOR, redshift_prior=Z_HINT)
    extra = {} if kind == "tde" else {"time_grid": None}     # the free path's quadrature
    baked = build(redshift=Z, dl_cm=luminosity_distance_cm(Z), t_exp_days=T_EXP, **extra)
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(3):
        p = baked.default_prior.sample(rng)
        f_b = baked.predict(p, times, BAND_OF)
        f_f = free.predict({**p, "t_exp": T_EXP, "redshift": Z}, times, BAND_OF)
        worst = max(worst, float(np.max(np.abs(_mag(f_b) - _mag(f_f)))))
    assert worst < 1e-12, (kind, worst)


def test_kilonova_auto_time_grid_is_redbacks_with_nothing_free():
    F = _factories()
    fixed = F.kilonova_model(BANDS, Z, luminosity_distance_cm(Z))
    assert fixed.predict_jax.ctx.time_grid == "redback"
    free = F.kilonova_model(BANDS, free=["t_exp"], prior=FREE_PRIOR, redshift=Z,
                            dl_cm=luminosity_distance_cm(Z))
    assert free.predict_jax.ctx.time_grid is None


# --- 3. one compilation ----------------------------------------------------------------------

def test_one_compile_for_200_walkers_with_200_redshifts():
    F = _factories()
    m = F.supernova_model("arnett", BANDS, free=["t_exp", "redshift"], prior=FREE_PRIOR,
                          redshift_prior=Z_HINT)
    pj = m.predict_jax
    bidx = pj.band_index(BAND_OF)
    rng = np.random.default_rng(0)
    theta = np.array([[m.default_prior.distributions[k].sample(rng) for k in m.parameters]
                      for _ in range(200)])
    theta[:, -1] = np.linspace(0.011, 0.29, 200)              # 200 different redshifts
    theta[:, -2] = np.linspace(-19.5, -0.1, 200)              # and explosion times
    traces = []

    def one(th):
        traces.append(1)
        return pj(th, TIMES, bidx)

    f = jax.jit(jax.vmap(one))
    out1 = np.asarray(f(jnp.asarray(theta)))
    out2 = np.asarray(f(jnp.asarray(theta[::-1].copy())))
    assert len(traces) == 1 and f._cache_size() == 1
    assert np.isfinite(out1).all() and np.isfinite(out2).all()
    assert out1.shape == (200, TIMES.size)


# --- 4. redback at identical z and t_exp -----------------------------------------------------

@pytest.mark.parametrize("model", ["arnett", "basic_magnetar_powered", "shock_cooling_and_arnett",
                                   "csm_shock_and_arnett", "type_1a", "type_1c"])
def test_free_supernova_matches_redback_at_identical_z_and_texp(model):
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as ra

    F = _factories()
    bands = ["ztfg", "ztfr"]
    cpu = ra.redback_model(model, band_names=bands, redshift=Z, constraint=None)
    free = F.supernova_model(model, bands, free=["t_exp", "redshift"], prior=FREE_PRIOR,
                             redshift_prior=Z_HINT, constraint=None)
    t_abs = T_EXP + np.linspace(2.0, 60.0, 20)
    b = np.array(bands * 10)
    rng = np.random.default_rng(5)
    worst, n = 0.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(8):
            p = cpu.default_prior.sample(rng)
            m_rb = _mag(cpu.predict(p, t_abs - T_EXP, b))
            m_fr = _mag(free.predict({**{k: p[k] for k in free.parameters if k in p},
                                      "t_exp": T_EXP, "redshift": Z}, t_abs, b))
            ok = np.isfinite(m_rb) & (m_rb < 30.0)
            if ok.any():
                worst, n = max(worst, float(np.max(np.abs(m_rb - m_fr)[ok]))), n + int(ok.sum())
    assert n > 0
    # redback's luminosity distance is astropy's; the table's is within 1e-6 mag of it
    assert worst < 2e-3, (model, worst)


# --- 5. the redshift hint --------------------------------------------------------------------

def test_redshift_prior_hint_of_a_light_curve_becomes_the_default_prior():
    import whisper_cbpf as wp

    F = _factories()
    lc = wp.LightCurve(time=TIMES, band=BAND_OF, magnitude=np.full(TIMES.size, 20.0),
                       magnitude_err=np.full(TIMES.size, 0.1))
    assert not lc.redshift_known
    m = F.supernova_model("arnett", BANDS, free=["redshift"], redshift_prior=lc.redshift_prior,
                          constraint=None)
    d = m.default_prior.distributions["redshift"]
    assert (d.low, d.high) == (lc.redshift_prior["low"], lc.redshift_prior["high"])
    # explicit prior wins over the hint, a hint wins over the package default
    m2 = F.tde_model(BANDS, free=["redshift"], redshift_prior={"type": "LogUniform",
                                                               "low": 0.02, "high": 0.5},
                     prior=Prior({"redshift": Uniform(0.05, 0.1)}))
    assert repr(m2.default_prior.distributions["redshift"]) == "Uniform(0.05, 0.1)"
    m3 = F.kilonova_model(BANDS, free=["redshift"],
                          redshift_prior={"type": "LogUniform", "low": 0.02, "high": 0.5})
    assert isinstance(m3.default_prior.distributions["redshift"], LogUniform)


def test_a_normal_redshift_hint_needs_a_truncated_normal_prior():
    from whisper_cbpf import priors as P
    from whisper_cbpf.io.schema import redshift_distribution

    hint = {"type": "Normal", "mu": 0.1, "sigma": 0.01}
    if getattr(P, "TruncatedNormal", None) is None:
        with pytest.raises(NotImplementedError, match=r"Uniform\(0.05, 0.15\)"):
            redshift_distribution(hint)
    else:
        d = redshift_distribution(hint)
        assert d.bounds[0] == pytest.approx(0.05) and d.bounds[1] == pytest.approx(0.15)


# --- 6. errors, pre-explosion epochs, gradients ----------------------------------------------

@pytest.mark.parametrize("call, match", [
    (lambda F: F.supernova_model("arnett", BANDS, free=["t_exp"], redshift=Z, dl_cm=1e27),
     "needs a prior on the light curve's own clock"),
    (lambda F: F.supernova_model("arnett", BANDS, Z, 1e27, free=["redshift"]),
     "Pass one: to fit it, redshift=None"),
    (lambda F: F.supernova_model("arnett", BANDS, Z), "redshift and dl_cm are required"),
    (lambda F: F.tde_model(BANDS, Z, 1e27, free=["mbh_6"]), "Only 't_exp'"),
    (lambda F: F.kilonova_model(BANDS, Z, 1e27, prior=Prior({"redshift": Uniform(0.01, 0.1)})),
     "Add them to free="),
    (lambda F: F.supernova_model("arnett", BANDS, free=["redshift"], diffusion_grid="data"),
     "diffusion_grid='fixed'"),
    (lambda F: F.kilonova_model(BANDS, free=["redshift"], time_grid="redback"),
     "time_grid=None"),
    (lambda F: F.supernova_model("arnett", BANDS, free=["redshift"],
                                 redshift_prior={"type": "Uniform", "low": 0.0, "high": 0.1}),
     "outside"),
    (lambda F: F.supernova_model("arnett", BANDS, Z, 1e27, max_phase_days=60.0),
     "diffusion_grid='fixed'"),
])
def test_errors_name_the_cause_and_the_fix(call, match):
    with pytest.raises(ValueError, match=match):
        call(_factories())


def test_an_epoch_past_the_fixed_epochs_raises_with_the_value_to_use():
    F = _factories()
    m = F.supernova_model("arnett", BANDS, free=["t_exp"], prior=FREE_PRIOR, redshift=Z,
                          dl_cm=luminosity_distance_cm(Z), max_phase_days=30.0, constraint=None)
    p = {**m.default_prior.sample(np.random.default_rng(0)), "t_exp": -5.0}
    with pytest.raises(ValueError, match="max_phase_days="):
        m.predict(p, np.array([10.0, 60.0]), np.array(["lsstg", "lsstr"]))
    with pytest.raises(ValueError, match="earliest t_exp"):
        m.predict_jax(np.array([p[k] for k in m.parameters]), np.array([10.0, 30.0]),
                      np.array([0, 1]))
    assert np.isfinite(m.predict(p, np.array([10.0, 20.0]), np.array(["lsstg", "lsstr"]))).all()


def test_a_density_on_epochs_past_the_fixed_epochs_raises_before_it_is_built():
    """log_density (and so fit_batch, likelihood_max_opt and profile) passes the epochs to the model
    traced, where no phase check can run; past the fixed epochs the luminosity is held at its last
    value.
    The density builders check the concrete epochs on the host first, under the fit's prior."""
    import whisper_cbpf as wp

    F = _factories()
    m = F.supernova_model("arnett", BANDS, free=["t_exp"], prior=FREE_PRIOR, redshift=Z,
                          dl_cm=luminosity_distance_cm(Z), max_phase_days=60.0, constraint=None)
    t = np.array([2.0, 10.0, 30.0, 70.0])
    b = np.array(BANDS + ["lsstg"])
    lc = wp.LightCurve(time=t, band=b, magnitude=[21.0, 20.0, 20.5, 22.0],
                       magnitude_err=np.full(4, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="max_phase_days=.*or narrow the t_exp"):
            wp.log_density(lc, m)
        narrow = Prior({**m.default_prior.distributions, "t_exp": Uniform(-2.0, 0.0)})
        assert wp.log_density(lc[:3], m, prior=narrow).n_data == 3


def test_a_fixed_epoch_model_refuses_an_explosion_prior_with_no_lower_bound():
    """The fixed epochs are sized from the earliest explosion the prior allows; a Normal prior
    has none (it used to size them to an infinite span and predict the magnitude floor)."""
    from whisper_cbpf.priors import Normal

    F = _factories()
    with pytest.raises(ValueError, match="no lower bound.*TruncatedNormal"):
        F.supernova_model("arnett", BANDS, free=["t_exp"], prior=Prior({"t_exp": Normal(-5, 2)}),
                          redshift=Z, dl_cm=luminosity_distance_cm(Z))


def test_pre_explosion_epochs_are_dark_and_the_gradient_is_finite():
    F = _factories()
    m = F.supernova_model("arnett", BANDS, free=["t_exp", "redshift"], prior=FREE_PRIOR,
                          redshift_prior=Z_HINT, constraint=None)
    p = dict(f_nickel=0.1, mej=2.0, vej=5e3, kappa=0.1, kappa_gamma=10.0,
             temperature_floor=5000.0, t_exp=-3.0, redshift=0.05)
    t = np.array([-10.0, -3.5, -2.0, 5.0, 20.0])
    mag = _mag(m.predict(p, t, np.array(["lsstg"] * 5)))
    assert np.allclose(mag[:2], 40.0) and np.all(mag[2:] < 40.0)
    theta = jnp.asarray([p[k] for k in m.parameters])
    bidx = m.predict_jax.band_index(np.array(["lsstg"] * 5))
    obs = np.asarray(m.predict(p, t, np.array(["lsstg"] * 5))) * 1.05

    def loglike(th):
        f = m.predict_jax(th, t, bidx)
        return -0.5 * jnp.sum(((f - obs) / (0.1 * obs)) ** 2)

    g = np.asarray(jax.grad(loglike)(theta))
    assert np.isfinite(g).all() and g[-1] != 0.0 and g[-2] != 0.0


def test_free_models_pickle_for_the_multiprocess_samplers():
    import pickle

    F = _factories()
    m = F.supernova_model("arnett", BANDS, free=["t_exp", "redshift"], prior=FREE_PRIOR,
                          redshift_prior=Z_HINT)
    p = m.default_prior.sample(np.random.default_rng(0))
    before = m.predict(p, TIMES, BAND_OF)                    # builds the fixed epochs and the jit
    after = pickle.loads(pickle.dumps(m.predict))(p, TIMES, BAND_OF)
    np.testing.assert_array_equal(before, after)


def test_the_samplers_density_takes_the_free_parameters():
    import whisper_cbpf as wp
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    F = _factories()
    m = F.supernova_model("arnett", BANDS, free=["t_exp", "redshift"], prior=FREE_PRIOR,
                          redshift_prior=Z_HINT)
    p = dict(f_nickel=0.1, mej=2.0, vej=5e3, kappa=0.1, kappa_gamma=10.0,
             temperature_floor=5000.0, t_exp=-3.0, redshift=0.05)
    flux = m.predict(p, TIMES, BAND_OF)
    lc = wp.LightCurve(time=TIMES, band=BAND_OF, flux=flux, flux_err=0.05 * flux)
    logp = make_log_prob_jax(lc, m)
    at_truth = float(logp(jnp.asarray([p[k] for k in m.parameters])))
    off = dict(p, redshift=0.1)
    assert np.isfinite(at_truth)
    assert at_truth > float(logp(jnp.asarray([off[k] for k in m.parameters])))
