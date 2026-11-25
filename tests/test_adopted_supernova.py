"""The four supernova models adapted from redback-jax and redback, as tests.

``csm_interaction`` (Chevalier-Chatzopoulos CSM interaction), ``shock_cooling`` (Piro 2021, alone),
``shocked_cocoon`` (Piro & Kollmeier 2018) and ``general_magnetar_driven`` (the relativistic
magnetar-driven ejecta ODE) are bound in ``supernova.py`` (``ADOPTED_MODELS``) on engines in
``csm_interaction.py``, ``shock_models.py`` and ``magnetar_ode.py``. The claims, each with a number
(measured values are in the comments, budgets leave a factor of 3 to 10 for another libm):

1. **They reproduce redback**, on 50 to 60 seeded prior draws, LSST g r i z and ZTF g r at z = 0.05,
   through whisper's own band integral on both sides. ``shock_cooling``, ``shocked_cocoon`` and the
   Euler solver of the ODE are redback's to round-off; the CSM engine and both diffusion methods
   are redback's to round-off on redback's own grid, and differ on the default grid by redback's
   own discretisation error (the converged answer is closer to the default grid than redback is).
2. **They reproduce redback-jax on the parts the two share** (the cocoon, the CSM engine, the ODE's
   right-hand side), in a subprocess, because importing redback-jax turns on ``jax_enable_x64``.
3. **Properties of the adapted engines**: the CSM is traceable in ``eta`` and compiled once and does
   not move with the last epoch; the shock-cooling diffusion time is the closed form; the cocoon
   taper has a gradient that agrees with finite differences.
4. **The ODE solvers** agree with a converged solution, the fallback without diffrax is the
   documented one and warns once, and the radiated energy respects the energy budget.
5. **float32 fails loudly**, ``jit`` / ``vmap`` / ``grad`` are finite on prior draws, and the
   whisper ``Model`` (data grid and free ``t_exp`` and redshift) is consistent.

FLOAT64. Required; the fixture below enables it for this module and restores the setting.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import textwrap
import warnings
from importlib.util import find_spec

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.models.jax import csm_interaction as C  # noqa: E402
from whisper_cbpf.models.jax import magnetar_ode as M  # noqa: E402
from whisper_cbpf.models.jax import shock_models as SM  # noqa: E402
from whisper_cbpf.models.jax import supernova as S  # noqa: E402
from whisper_cbpf.models.jax import supernova_model  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402

Z = 0.05
#: LSST g, r, i, z and ZTF g, r.
BANDS = ["lsstg", "lsstr", "lssti", "lsstz", "ztfg", "ztfr"]
N_DRAWS = 60
SEED = 20261001
NEW = ("csm_interaction", "shock_cooling", "shocked_cocoon", "general_magnetar_driven")
#: Source-frame epochs (days) each model is compared at, spread over its light curve.
EPOCHS = {
    "csm_interaction": np.array([3.0, 10.0, 30.0, 60.0, 100.0, 200.0]),
    "shock_cooling": np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
    "shocked_cocoon": np.array([0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0]),
    "general_magnetar_driven": np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]),
}
HAS_DIFFRAX = find_spec("diffrax") is not None


@pytest.fixture(scope="module", autouse=True)
def _x64():
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


@pytest.fixture(scope="module")
def dl():
    """Planck18's luminosity distance at Z [cm], astropy's, which is what redback's models use."""
    from astropy.cosmology import Planck18

    return float(Planck18.luminosity_distance(Z).cgs.value)


def _model(name, dl, **kw):
    if name == "general_magnetar_driven" and "engine_options" not in kw:
        kw["engine_options"] = {"solver": "rk4"}
    return supernova_model(name, BANDS, Z, dl, **kw)


def _draws(name, n=N_DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    prior = S.default_prior(name)[0]
    return [prior.sample(rng) for _ in range(n)]


def _layout(epochs):
    """Every epoch in every band, time-ordered (the last observation is the latest)."""
    return np.repeat(epochs, len(BANDS)), np.tile(np.array(BANDS), epochs.size)


def _redback_mags(model_call, t, b):
    """redback's ``flux_density`` through whisper's band integral: AB magnitudes."""
    from whisper_cbpf.synphot import band_flux_jy, filter_set_for

    fjy = band_flux_jy(lambda tt, nu: np.asarray(model_call(tt, nu)) * 1e-3, t, b,
                       filter_set_for(BANDS))
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(fjy / 3631.0)


def _mags(model, p, t, b):
    return -2.5 * np.log10(np.asarray(model.predict(p, t, b)) / 3631.0)


def _stats(x):
    x = np.abs(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(np.percentile(x, 99)), float(np.max(x))


def _trapezoid(y, x):
    return (getattr(np, "trapezoid", None) or np.trapz)(y, x)


def _lg(x):
    return np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def _photometry_error(model, name, redback_flux, n=N_DRAWS, seed=SEED):
    """``model`` minus redback, in magnitudes, over ``n`` prior draws, where both are visible."""
    t, b = _layout(EPOCHS[name] * (1.0 + Z))
    err = []
    for p in _draws(name, n, seed):
        ref = _redback_mags(lambda tt, nu, p=p: redback_flux(p, tt, nu), t, b)
        mine = _mags(model, p, t, b)
        assert np.isfinite(mine).all(), "a prediction is NaN or infinite"
        ok = np.isfinite(ref) & (ref < 38.0) & (mine < 39.9)
        err.append(np.where(ok, mine - ref, np.nan))
    err = np.array(err)
    assert np.isfinite(err).sum() > 0.3 * err.size, "too few visible points: the test cannot fail"
    return err


def _rbjax(code, payload):
    """Run ``code`` in a fresh interpreter where redback-jax imports; its stdout is JSON."""
    if find_spec("redback_jax") is None:
        pytest.skip("redback-jax is not installed")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
                         input=json.dumps(payload), capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# =======================================================================================
# 0. the registry, the priors, the data file
# =======================================================================================

def test_the_four_models_are_registered_and_do_not_collide():
    names = S.model_names()
    assert set(NEW) <= set(names)
    assert len(names) == len(set(names)) == 16
    # the existing Margalit CSM breakout is a different model under its own name
    assert "csm_shock_and_arnett" in names and "csm_interaction" in names
    for name in NEW:
        assert S.PARAMETERS[name] == S.ALL_MODELS[name]["params"]
        assert name in wp.supernova_models()


def test_default_priors_are_redbacks_prior_files():
    """Each default prior is the installed redback's ``.prior`` file, bound for bound; the
    transcription used without redback is the same."""
    pytest.importorskip("redback")
    pytest.importorskip("bilby")
    for name in NEW:
        rb, rb_pinned, _ = S.redback_prior(name)
        tr, tr_pinned, _ = S.fallback_prior(name)
        assert {k: repr(v) for k, v in rb.distributions.items()} == \
            {k: repr(v) for k, v in tr.distributions.items()}, name
        assert rb_pinned == tr_pinned, name
        assert set(rb.distributions) | set(rb_pinned) >= set(S.PARAMETERS[name]) - set(rb_pinned)
    _, pinned, _ = S.redback_prior("csm_interaction")
    assert pinned == {"nn": 12.0, "delta": 1.0, "efficiency": 0.5}


def test_the_packaged_csm_table_is_redbacks():
    redback = pytest.importorskip("redback")
    from pathlib import Path

    theirs = Path(redback.__file__).parent / "tables" / "csm_table.txt"
    assert C.TABLE_FILE.read_bytes() == theirs.read_bytes()
    assert C.TABLE_FILE.stat().st_size < 30_000
    from redback.utils import get_csm_properties

    for nn, eta in [(12.0, 0.0), (12.0, 0.37), (9.3, 1.8), (6.0, 2.0), (14.0, 0.5), (7.77, 1.234)]:
        ref = get_csm_properties(nn, eta)
        got = [float(x) for x in C.csm_coefficients(nn, eta)]
        np.testing.assert_allclose(got, [ref.AA, ref.Bf, ref.Br], rtol=1e-13)


def test_engine_options_are_checked():
    S.check_engine_options("csm_interaction", {"csm_diffusion_method": "quadrature"})
    S.check_engine_options("general_magnetar_driven", {"solver": "euler", "n_grid": 100})
    with pytest.raises(ValueError, match="not settings of 'shocked_cocoon'"):
        S.check_engine_options("shocked_cocoon", {"solver": "rk4"})
    with pytest.raises(ValueError, match="csm_diffusion_method"):
        S.check_engine_options("csm_interaction", {"method": "quadrature"})
    with pytest.raises(ValueError, match="cumulative"):
        C.csm_interaction_bolometric({"time": jnp.array([1.0, 2.0])}, 10.0, 5.0, 5000.0, 0.0,
                                     1e-12, 0.34, 100.0, csm_diffusion_method="nope")


# =======================================================================================
# 1. agreement with redback
# =======================================================================================

def _csm_on_redbacks_grid(p, T, method):
    """Our engine and diffusion on the dense grid redback builds for these epochs."""
    from redback.utils import get_optimal_time_array

    nodes = get_optimal_time_array(min(0.1, T.min()), T.max() + 100, 1000, user_times=T,
                                   time_units="days")
    lum, r_ph, m_thick = C.csm_engine(jnp.asarray(nodes), p["mej"], p["csm_mass"], p["vej"],
                                      p["eta"], p["rho"], p["kappa"], p["r0"])
    t_diff = C.csm_diffusion_timescale(p["kappa"], r_ph, m_thick)
    return np.asarray(C.csm_diffusion(jnp.asarray(T), jnp.asarray(nodes), lum, t_diff, method))


@pytest.mark.parametrize("method", ["cumulative", "quadrature"])
def test_csm_engine_and_both_diffusion_methods_are_redbacks_on_its_grid(method):
    """redback's engine, table lookup and diffusion (its default ``cumulative`` method and its legacy
    ``quadrature``) on the dense grid redback builds: 60 draws x 50 epochs, measured worst case
    9e-13 dex (``cumulative``) and 2e-14 (``quadrature``)."""
    pytest.importorskip("redback")
    from redback.transient_models import supernova_models as osn

    T = np.geomspace(0.5, 300.0, 50)
    err = []
    for p in _draws("csm_interaction"):
        kw = dict(mej=p["mej"], csm_mass=p["csm_mass"], vej=p["vej"], eta=p["eta"], rho=p["rho"],
                  kappa=p["kappa"], r0=p["r0"])
        ref = osn.csm_interaction_bolometric(T, csm_diffusion_method=method, **kw)
        mine = _csm_on_redbacks_grid(p, T, method)
        ok = ref > 1e30
        err.append(np.where(ok, _lg(mine) - _lg(ref), np.nan))
    assert np.isfinite(err).sum() > 1000
    assert _stats(err)[2] < 1e-9, _stats(err)


def test_csm_default_grid_against_redback_and_against_the_converged_answer():
    """On the default grid (1000 geometric nodes holding the epochs) the bolometric light curve is
    within redback's own grid error of redback and much closer to a converged answer than redback
    is. 60 draws x 50 epochs, points within 2 dex of each draw's peak. Against redback: median 4.0e-3,
    p99 0.14, max 0.17 dex (all of it redback's grid: linear interpolation of the engine between
    sparse nodes). Against an exact recurrence on 60 000 nodes: ours 1.9e-4 / 5.9e-3 / 0.021 dex,
    redback's 3.4e-3 / 0.15 / 0.18."""
    pytest.importorskip("redback")
    from redback.transient_models import supernova_models as osn

    T = np.geomspace(0.5, 300.0, 50)
    nodes = np.unique(np.concatenate([np.geomspace(1e-4, T[-1] + 100.0, 60000), T]))
    vs_rb, vs_ref, rb_vs_ref = [], [], []
    for p in _draws("csm_interaction"):
        kw = dict(mej=p["mej"], csm_mass=p["csm_mass"], vej=p["vej"], eta=p["eta"], rho=p["rho"],
                  kappa=p["kappa"], r0=p["r0"])
        rb = osn.csm_interaction_bolometric(T, **kw)
        mine = np.asarray(C.csm_interaction_bolometric({"time": jnp.asarray(T)}, **kw))
        lum, r_ph, m_thick = C.csm_engine(jnp.asarray(nodes), **kw)
        y = np.asarray(C.exp_convolution(jnp.asarray(nodes), lum,
                                         C.csm_diffusion_timescale(p["kappa"], r_ph, m_thick)))
        ref = np.interp(T, nodes, y)
        ok = (rb > 1e30) & (ref > 1e30) & (_lg(ref) > _lg(ref).max() - 2.0)
        vs_rb.append(np.where(ok, _lg(mine) - _lg(rb), np.nan))
        vs_ref.append(np.where(ok, _lg(mine) - _lg(ref), np.nan))
        rb_vs_ref.append(np.where(ok, _lg(rb) - _lg(ref), np.nan))
    a, b, c = _stats(vs_rb), _stats(vs_ref), _stats(rb_vs_ref)
    print(f"\ncsm default grid vs redback {a}, vs converged {b}; redback vs converged {c}")
    assert a[0] < 0.02 and a[1] < 0.4 and a[2] < 0.5, a
    assert b[0] < 1e-3 and b[1] < 0.02 and b[2] < 0.08, b
    assert b[0] < c[0] and b[1] < c[1] and b[2] < c[2]


def test_csm_photometry_matches_redback(dl):
    """60 draws, LSST griz + ZTF gr at z = 0.05. Measured: median 3.6e-3 mag, p99 0.26, max 0.99,
    which is redback's grid error again (the p99 band is 0.07 mag against the converged answer)."""
    pytest.importorskip("redback")
    from redback.transient_models import supernova_models as osn

    from whisper_cbpf.models.redback_adapter import redback_luminosity_distance_cm

    model = _model("csm_interaction", redback_luminosity_distance_cm(Z, "csm_interaction"))
    err = _photometry_error(model, "csm_interaction", lambda p, tt, nu: osn.csm_interaction(
        tt, Z, p["mej"], p["csm_mass"], p["vej"], p["eta"], p["rho"], p["kappa"], p["r0"],
        temperature_floor=p["temperature_floor"], output_format="flux_density", frequency=nu))
    med, p99, mx = _stats(err)
    print(f"\ncsm_interaction vs redback: median {med:.2e}, p99 {p99:.2e}, max {mx:.2e} mag")
    assert med < 0.02 and p99 < 0.6 and mx < 1.6, (med, p99, mx)


def test_shock_cooling_matches_redback(dl):
    """Piro's engine and its own photosphere: redback's to round-off. Measured over 60 draws:
    photometry 1.4e-14 mag; bolometric (100 draws x 60 epochs, 0.02 to 100 d) 7e-15 dex."""
    pytest.importorskip("redback")
    from redback.transient_models import shock_powered_models as osp

    model = _model("shock_cooling", dl)
    err = _photometry_error(model, "shock_cooling", lambda p, tt, nu: osp.shock_cooling(
        tt, Z, p["log10_mass"], p["log10_radius"], p["log10_energy"], nn=p["nn"], delta=p["delta"],
        output_format="flux_density", frequency=nu))
    assert _stats(err)[2] < 1e-9, _stats(err)

    t = np.geomspace(0.02, 100.0, 60)
    worst = 0.0
    for p in _draws("shock_cooling", 100, SEED + 1):
        ref = osp.shock_cooling_bolometric(t * 86400.0, p["log10_mass"], p["log10_radius"],
                                           p["log10_energy"], nn=p["nn"], delta=p["delta"])
        mine, _, _ = S.shock_cooling_model({"time": jnp.asarray(t)}, p["log10_mass"],
                                           p["log10_radius"], p["log10_energy"], p["nn"],
                                           p["delta"])
        ok = ref > 1e30
        worst = max(worst, float(np.max(np.abs(_lg(mine)[ok] - _lg(ref)[ok]))))
    assert worst < 1e-10, worst


def test_shocked_cocoon_matches_redback(dl):
    """The closed form with the stable taper is redback's to round-off where redback has a value:
    photometry 9e-14 mag over 60 draws; bolometric (100 draws x 60 epochs, 0.01 to 100 d, within
    2 dex of peak) 7e-15 dex."""
    pytest.importorskip("redback")
    from redback.transient_models import shock_powered_models as osp

    model = _model("shocked_cocoon", dl)
    err = _photometry_error(model, "shocked_cocoon", lambda p, tt, nu: osp.shocked_cocoon(
        tt, Z, p["mej"], p["vej"], p["eta"], p["tshock"], p["shocked_fraction"],
        p["cos_theta_cocoon"], p["kappa"], output_format="flux_density", frequency=nu))
    assert _stats(err)[2] < 1e-9, _stats(err)

    t = np.geomspace(0.01, 100.0, 60)
    rng = np.random.default_rng(SEED + 2)
    worst = 0.0
    for _ in range(100):
        p = dict(mej=rng.uniform(1e-4, 0.1), vej=rng.uniform(0.1, 0.7), eta=rng.uniform(1, 5),
                 tshock=rng.uniform(1, 2), shocked_fraction=rng.uniform(0.1, 1),
                 cos_theta_cocoon=rng.uniform(0.2, 0.9), kappa=rng.uniform(1, 30))
        ref = osp.shocked_cocoon_bolometric(t, **p)
        mine = np.asarray(SM.shocked_cocoon_model({"time": jnp.asarray(t)}, **p)[0])
        bright = ref > ref.max() * 1e-2
        worst = max(worst, float(np.max(np.abs(_lg(mine)[bright] - _lg(ref)[bright]))))
    assert worst < 1e-10, worst


def _ode_args(p):
    return (p["mej"], p["E_sn"], p["kappa"], p["l0"], p["tau_sd"], p["nn"], p["kappa_gamma"],
            p["f_nickel"])


def _ode_batch(draws, epochs, **kw):
    """log10 L of every draw at ``epochs`` (vmapped, so one compilation), and ``ok`` of each."""
    a = jnp.asarray([_ode_args(p) for p in draws])
    f = jax.jit(jax.vmap(lambda x: M.magnetar_ode_solve(jnp.asarray(epochs), *[x[j] for j in range(8)],
                                                       **kw)[::2]))
    lum, ok = f(a)
    return _lg(lum), np.asarray(ok)


def test_magnetar_euler_solver_is_redbacks_recursion(dl):
    """``solver='euler'`` is redback's own explicit Euler on redback's own 1998-node grid: the
    bolometric luminosity agrees to 3e-12 dex over 100 draws x 40 epochs, the photometry (redback's
    photosphere interpolation included) to 5e-12 mag over 60 draws."""
    pytest.importorskip("redback")
    from redback.transient_models import supernova_models as osn
    from redback.utils import get_optimal_time_array

    assert np.array_equal(M.redback_time_grid(), get_optimal_time_array(1.0, 1e8, 2000))

    t = np.geomspace(0.3, 400.0, 40)
    draws = _draws("general_magnetar_driven", 100, SEED + 3)
    refs = np.array([osn.general_magnetar_driven_supernova_bolometric(
        t, *_ode_args(p)[:7], f_nickel=p["f_nickel"], output_format="lbol") for p in draws])
    mine, ok = _ode_batch(draws, t, solver="euler")
    visible = np.isfinite(refs) & (refs > 1e30)
    assert visible.sum() > 2500 and ok.all()
    assert np.max(np.abs(mine - _lg(refs))[visible]) < 1e-9

    model = _model("general_magnetar_driven", dl, engine_options={"solver": "euler"})
    err = _photometry_error(model, "general_magnetar_driven",
                            lambda p, tt, nu: osn.general_magnetar_driven_supernova(
                                tt, Z, p["mej"], p["E_sn"], p["kappa"], p["l0"], p["tau_sd"], p["nn"],
                                p["kappa_gamma"], f_nickel=p["f_nickel"],
                                temperature_floor=p["temperature_floor"],
                                output_format="flux_density", frequency=nu))
    assert _stats(err)[2] < 1e-8, _stats(err)


def test_redbacks_euler_error_is_the_euler_error_and_rk4_is_converged():
    """redback's recursion is first order. Against a converged solution (RK4 on 20 000 steps) the
    ``euler`` solver and redback itself have the same error (the same algorithm): median 8e-3,
    p99 0.06, max 0.07 dex over 100 draws; the default ``rk4`` (1000 steps) has 6e-6 / 1e-4 /
    1.6e-3."""
    pytest.importorskip("redback")
    from redback.transient_models import supernova_models as osn

    t = np.geomspace(0.3, 400.0, 40)
    draws = _draws("general_magnetar_driven", 100, SEED + 3)
    ref, _ = _ode_batch(draws, t, solver="rk4", n_grid=20000)
    euler, _ = _ode_batch(draws, t, solver="euler")
    rk4, _ = _ode_batch(draws, t, solver="rk4")
    rb = np.array([_lg(osn.general_magnetar_driven_supernova_bolometric(
        t, *_ode_args(p)[:7], f_nickel=p["f_nickel"], output_format="lbol")) for p in draws])
    bright = (ref > ref.max(axis=1, keepdims=True) - 2) & np.isfinite(rb) & (rb > 30)
    d_euler = np.where(bright, euler - ref, np.nan)
    d_rb = np.where(bright, rb - ref, np.nan)
    np.testing.assert_allclose(d_euler[bright], d_rb[bright], atol=1e-9)
    med, p99, mx = _stats(d_rb)
    assert 1e-3 < med < 3e-2 and mx < 0.2, (med, p99, mx)
    med, p99, mx = _stats(np.where(bright, rk4 - ref, np.nan))
    print(f"\nrk4 (1000 steps) vs converged: {med:.2e} / {p99:.2e} / {mx:.2e} dex")
    assert med < 5e-5 and p99 < 5e-4 and mx < 6e-3, (med, p99, mx)


@pytest.mark.skipif(not HAS_DIFFRAX, reason="diffrax is an optional dependency (the [ode] extra)")
def test_magnetar_diffrax_solver_is_converged():
    """Tsit5 at rtol 1e-5 against Tsit5 at rtol 1e-10, 100 draws x 40 epochs, bright points:
    median 7.8e-6, p99 1.2e-4, max 8e-4 dex (300 draws); and against the RK4 fallback."""
    t = np.geomspace(0.3, 400.0, 40)
    draws = _draws("general_magnetar_driven", 100, SEED + 3)
    ref, ok_ref = _ode_batch(draws, t, solver="diffrax", rtol=1e-10, atol=1e-12, max_steps=100000)
    fast, ok = _ode_batch(draws, t, solver="diffrax")
    rk4, _ = _ode_batch(draws, t, solver="rk4")
    assert ok_ref.all() and ok.all()
    bright = ref > ref.max(axis=1, keepdims=True) - 2
    med, p99, mx = _stats(np.where(bright, fast - ref, np.nan))
    print(f"\ndiffrax vs tight reference: {med:.2e} / {p99:.2e} / {mx:.2e} dex")
    assert med < 3e-5 and p99 < 4e-4 and mx < 4e-3, (med, p99, mx)
    med, p99, mx = _stats(np.where(bright, rk4 - ref, np.nan))
    assert med < 5e-5 and p99 < 5e-4 and mx < 6e-3, (med, p99, mx)


@pytest.mark.skipif(not HAS_DIFFRAX, reason="diffrax is an optional dependency (the [ode] extra)")
def test_magnetar_diffrax_handles_repeated_and_unsorted_epochs_and_reports_failure():
    p = _draws("general_magnetar_driven", 1)[0]
    tt = jnp.array([20.0, 5.0, 5.0, 100.0, 20.0, 0.5])
    lum, _, ok = M.magnetar_ode_solve(tt, *_ode_args(p), solver="diffrax")
    ref, _, _ = M.magnetar_ode_solve(jnp.sort(tt), *_ode_args(p), solver="diffrax")
    assert bool(ok)
    np.testing.assert_allclose(np.asarray(lum)[np.argsort(np.asarray(tt))], np.asarray(ref),
                               rtol=1e-12)
    _, _, ok = M.magnetar_ode_solve(tt, *_ode_args(p), solver="diffrax", max_steps=5)
    assert not bool(ok)                          # out of steps is reported, not raised
    out = M.general_magnetar_driven_model({"time": tt}, *_ode_args(p), temperature_floor=5000.0,
                                          solver="diffrax", max_steps=5)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in out)       # and the draw is finite


@pytest.mark.skipif(not HAS_DIFFRAX, reason="diffrax is an optional dependency (the [ode] extra)")
def test_magnetar_photometry_with_diffrax_matches_redback_within_its_euler_error(dl):
    """The converged solution against redback's first-order one in photometry: median 0.015 mag
    over 60 draws, p99 0.2, max 0.55 -- redback's own error (its luminosity is off by up to
    0.09 dex, and the photosphere is steeply sensitive at the temperature floor)."""
    pytest.importorskip("redback")
    from redback.transient_models import supernova_models as osn

    model = _model("general_magnetar_driven", dl, engine_options={"solver": "diffrax"})
    err = _photometry_error(model, "general_magnetar_driven",
                            lambda p, tt, nu: osn.general_magnetar_driven_supernova(
                                tt, Z, p["mej"], p["E_sn"], p["kappa"], p["l0"], p["tau_sd"], p["nn"],
                                p["kappa_gamma"], f_nickel=p["f_nickel"],
                                temperature_floor=p["temperature_floor"],
                                output_format="flux_density", frequency=nu))
    med, p99, mx = _stats(err)
    assert med < 0.04 and p99 < 0.5 and mx < 1.2, (med, p99, mx)


# =======================================================================================
# 2. agreement with redback-jax on the shared parts
# =======================================================================================

def test_cocoon_csm_and_ode_agree_with_redback_jax():
    """The shocked-cocoon engine (bright points, 60 draws x 60 epochs: median 8.6e-5, max 3.3e-4 dex
    -- rounded constants), the CSM engine at identical coefficients (5e-5 dex; table
    lookup identical to 2e-16) and the ODE (Euler on 2000 geometric nodes and Tsit5, 40 draws:
    5e-4 and 7e-4 dex worst case)."""
    if find_spec("redback_jax") is None:
        pytest.skip("redback-jax is not installed")
    rng = np.random.default_rng(SEED + 4)
    cocoon = [dict(mej=rng.uniform(1e-4, 0.1), vej=rng.uniform(0.1, 0.7), eta=rng.uniform(1, 5),
                   tshock=rng.uniform(1, 2), shocked_fraction=rng.uniform(0.1, 1),
                   cos_theta_cocoon=rng.uniform(0.2, 0.9), kappa=rng.uniform(1, 30))
              for _ in range(60)]
    csm = _draws("csm_interaction", 60, SEED + 5)
    ode = _draws("general_magnetar_driven", 40, SEED + 6)
    mine_cocoon = [np.asarray(SM.shocked_cocoon_model({"time": jnp.asarray(np.geomspace(0.01, 100, 60))},
                                                      **p)[0]) for p in cocoon]
    t_csm = np.geomspace(0.5, 300.0, 40)
    t_ode = np.geomspace(0.3, 400.0, 40)
    mine_csm = [[np.asarray(x) for x in C.csm_engine(jnp.asarray(t_csm), p["mej"], p["csm_mass"], p["vej"],
                                                     p["eta"], p["rho"], p["kappa"], p["r0"])]
                for p in csm]
    coef = [[float(x) for x in C.csm_coefficients(12.0, p["eta"])] for p in csm]
    mine_euler = _ode_batch(ode, t_ode, solver="euler", n_grid=2000)[0]
    mine_dfx = _ode_batch(ode, t_ode, solver="diffrax")[0] if HAS_DIFFRAX else None

    theirs = _rbjax("""
        import json, sys
        import numpy as np, jax, jax.numpy as jnp
        from redback_jax.models import shock_powered_models as sp, supernova_models as jsn
        jax.config.update("jax_enable_x64", True)
        d = json.loads(sys.stdin.read())
        t = np.geomspace(0.01, 100, 60)
        cocoon = [np.asarray(sp.shocked_cocoon_bolometric(jnp.asarray(t), **p)).tolist() for p in d["cocoon"]]
        csm, coef = [], []
        for p in d["csm"]:
            AA, Bf, Br = jsn._get_csm_coefficients(12.0, p["eta"])
            coef.append([AA, Bf, Br])
            out = jsn._csm_engine(jnp.asarray(d["t_csm"]), p["mej"], p["csm_mass"], p["vej"], p["eta"],
                                  p["rho"], p["kappa"], p["r0"], 12.0, AA, Bf, Br, 1.0, 0.5)
            csm.append([np.asarray(x).tolist() for x in out])
        ode = []
        for p in d["ode"]:
            a = [p[k] for k in ("mej", "E_sn", "kappa", "l0", "tau_sd", "nn", "kappa_gamma", "f_nickel")]
            e = np.asarray(jsn._run_magnetar_ode(jnp.asarray(d["t_ode"]), *a, 2000)[0])
            try:
                f = np.asarray(jsn._run_magnetar_ode_diffrax(jnp.asarray(d["t_ode"]), *a, 1e-5, 1e-8)[0])
            except ImportError:
                f = e * np.nan
            ode.append([e.tolist(), f.tolist()])
        print(json.dumps(dict(cocoon=cocoon, csm=csm, coef=coef, ode=ode)))
        """, dict(cocoon=cocoon, csm=[{k: float(v) for k, v in p.items()} for p in csm],
                  t_csm=t_csm.tolist(), ode=[{k: float(v) for k, v in p.items()} for p in ode],
                  t_ode=t_ode.tolist()))

    err = []
    for ref, mine in zip(theirs["cocoon"], mine_cocoon):
        ref = np.asarray(ref)
        bright = ref > ref.max() - 2
        err.append((_lg(mine) - ref)[bright])
    assert np.max(np.abs(np.concatenate(err))) < 1e-3

    np.testing.assert_allclose(np.asarray(coef), np.asarray(theirs["coef"]), rtol=1e-12)
    err = []
    for ref, mine in zip(theirs["csm"], mine_csm):
        lum_ref, lum = np.asarray(ref[0]), mine[0]
        assert np.array_equal(lum_ref > 1e30, lum > 1e30)         # the shock cut-offs fall together
        ok = lum_ref > 1e30
        err.append((_lg(lum) - _lg(lum_ref))[ok])
        np.testing.assert_allclose(mine[1], ref[1], rtol=1e-2)    # r_ph, rounded AU and Msun
    assert np.max(np.abs(np.concatenate(err))) < 5e-4

    e_euler = [(mine - np.asarray(ref[0]))[np.asarray(ref[0]) > np.asarray(ref[0]).max() - 2]
               for mine, ref in zip(mine_euler, theirs["ode"])]
    assert np.max(np.abs(np.concatenate(e_euler))) < 3e-3
    if mine_dfx is not None and np.all(np.isfinite([r[1] for r in theirs["ode"]])):
        e_dfx = [(mine - np.asarray(ref[1]))[np.asarray(ref[1]) > np.asarray(ref[1]).max() - 2]
                 for mine, ref in zip(mine_dfx, theirs["ode"])]
        assert np.max(np.abs(np.concatenate(e_dfx))) < 4e-3


# =======================================================================================
# 3. properties of the adapted engines
# =======================================================================================

def _csm_args(p):
    return [p[k] for k in ("mej", "csm_mass", "vej", "eta", "rho", "kappa", "r0")]


def test_csm_is_traceable_and_differentiable_in_eta_and_nn():
    """The table lookup is jnp: ``jit``, ``vmap`` and ``grad`` over ``eta`` and ``nn`` work, and the
    gradients agree with central differences (relative 1e-3 or better over 20 draws and 7
    parameters; measured 1.7e-6, 7e-5 at the worst of another 20)."""
    t = jnp.asarray(np.geomspace(1.0, 200.0, 12))
    names = ("mej", "csm_mass", "vej", "eta", "rho", "kappa", "r0")

    def loglum(th, nn=12.0):
        lum = C.csm_interaction_bolometric({"time": t}, *[th[i] for i in range(7)], nn=nn)
        return jnp.sum(jnp.log10(jnp.maximum(lum, 1e-30)))

    grad, value = jax.jit(jax.grad(loglum)), jax.jit(loglum)
    worst = 0.0
    for p in _draws("csm_interaction", 20, SEED + 7):
        th = jnp.asarray([p[n] for n in names])
        g = np.asarray(grad(th))
        assert np.all(np.isfinite(g))
        for i in range(7):
            h = 1e-5
            e = np.zeros(7)
            e[i] = 1.0
            fd = (float(value(th * jnp.exp(h * e))) - float(value(th * jnp.exp(-h * e)))) / (2 * h)
            worst = max(worst, abs(g[i] * float(th[i]) - fd) / max(abs(fd), 1e-3))
    assert worst < 1e-3, worst
    p = _draws("csm_interaction", 1)[0]
    th = jnp.asarray([p[n] for n in names])
    dnn = float(jax.grad(lambda nn: loglum(th, nn))(12.0))
    fd = (float(loglum(th, 12.0 + 1e-5)) - float(loglum(th, 12.0 - 1e-5))) / 2e-5
    assert dnn == pytest.approx(fd, rel=1e-5)
    etas = jnp.linspace(0.1, 1.9, 7)
    batch = jax.vmap(lambda e: loglum(th.at[3].set(e)))(etas)
    assert batch.shape == (7,) and bool(jnp.all(jnp.isfinite(batch)))


def test_csm_compiles_once_for_any_parameter_values():
    """Five different draws, including a different ``eta`` and ``nn``, add nothing to the cache of
    the jitted kernel."""
    t = {"time": jnp.asarray(np.geomspace(1.0, 100.0, 9))}
    draws = _draws("csm_interaction", 6, SEED + 8)
    C.csm_interaction_bolometric(t, *_csm_args(draws[0]))
    before = C._jitted_bolometric._cache_size()
    for p, nn in zip(draws[1:], (8.0, 10.0, 12.0, 13.5, 7.0)):
        C.csm_interaction_bolometric(t, *_csm_args(p), nn=nn, delta=0.5, efficiency=0.3)
    assert C._jitted_bolometric._cache_size() == before


def test_csm_does_not_move_with_the_last_epoch():
    """The answer at fixed epochs when the light curve is extended: the first epochs, with later ones
    appended to 300, 1000 and 3000 d, move by a median 4e-7 dex, p99 0.014, max 0.055 (the engine
    grid's own resolution, 1000 nodes)."""
    first = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    err = []
    for p in _draws("csm_interaction", 60, SEED + 9):
        kw = dict(zip(("mej", "csm_mass", "vej", "eta", "rho", "kappa", "r0"), _csm_args(p)))
        base = np.asarray(C.csm_interaction_bolometric({"time": jnp.asarray(first)}, **kw))
        for last in (300.0, 1000.0, 3000.0):
            t = np.concatenate([first, [x for x in (300.0, 1000.0, 3000.0) if x <= last]])
            lum = np.asarray(C.csm_interaction_bolometric({"time": jnp.asarray(t)}, **kw))[:5]
            ok = (base > 1e30) & (lum > 1e30)
            err.append(np.where(ok, _lg(lum) - _lg(base), np.nan))
    med, p99, mx = _stats(err)
    print(f"\ncsm vs last epoch: {med:.1e} / {p99:.1e} / {mx:.1e} dex")
    assert med < 1e-5 and p99 < 0.03 and mx < 0.1, (med, p99, mx)


def test_csm_cumulative_recurrence_is_exact_for_known_sources():
    """A constant source: ``1 - exp(-t / t_d)``; a source linear in time: ``t - t_d (1 - exp(-t/t_d))``,
    both to round-off, for a diffusion time from 0.05 to 100 d on a grid starting at 0."""
    t = jnp.concatenate([jnp.zeros(1), jnp.geomspace(1e-4, 500.0, 4000)])
    for t_d in (0.05, 1.0, 20.0, 100.0):
        y = np.asarray(C.exp_convolution(t, jnp.ones_like(t), t_d))
        np.testing.assert_allclose(y, 1.0 - np.exp(-np.asarray(t) / t_d), atol=1e-12)
        y = np.asarray(C.exp_convolution(t, t, t_d))
        np.testing.assert_allclose(y, np.asarray(t) - t_d * (1.0 - np.exp(-np.asarray(t) / t_d)),
                                   atol=1e-9)
    # equal neighbouring nodes (an epoch on a node) add nothing and break nothing
    tt = jnp.array([0.0, 1.0, 1.0, 2.0])
    y = C.exp_convolution(tt, jnp.ones(4), 1.0)
    assert bool(jnp.all(jnp.isfinite(y))) and float(y[1]) == pytest.approx(float(y[2]), abs=1e-15)


def test_shock_cooling_diffusion_time_is_in_the_right_units():
    """``td`` and the light curve are the closed form of Piro's solution, evaluated independently in
    NumPy (``td`` in seconds, converted to days only for the epochs)."""
    from astropy import constants as cc

    p = dict(log10_mass=-1.0, log10_radius=13.0, log10_energy=47.0, nn=10.0, delta=1.1)
    n, d, kappa = p["nn"], p["delta"], 0.2
    mass = 10.0 ** p["log10_mass"] * cc.M_sun.cgs.value
    c = cc.c.cgs.value
    kk = (n - 3) * (3 - d) / (4 * np.pi * (n - d))
    vt = (((n - 5) * (5 - d) / ((n - 3) * (3 - d))) * 2 * 10.0 ** p["log10_energy"] / mass) ** 0.5
    td = ((3 * kappa * kk * mass) / ((n - 1) * vt * c)) ** 0.5            # seconds
    prefactor = np.pi * (n - 1) / (3 * (n - 5)) * c * 10.0 ** p["log10_radius"] * vt ** 2 / kappa

    _, _, _, td_engine = S.shock_cooling(jnp.array([1.0]), 10.0 ** p["log10_mass"], 1e13, 1e47, n, d)
    assert float(td_engine) == pytest.approx(td, rel=1e-12)
    assert 1.0 < td / 86400.0 < 100.0                                    # 13.5 d: a plausible envelope
    t = np.array([0.3 * td, 0.9 * td, 1.0 * td, 2.0 * td, 4.0 * td]) / 86400.0
    lum = np.asarray(S.shock_cooling_model({"time": jnp.asarray(t)}, *p.values())[0])
    expected = np.where(t * 86400.0 < td, prefactor * (td / (t * 86400.0)) ** (4 / (n - 2)),
                        prefactor * np.exp(-0.5 * ((t * 86400.0) ** 2 / td ** 2 - 1)))
    np.testing.assert_allclose(lum, expected, rtol=1e-12)
    # the pre-peak slope of Piro's solution, -4 / (n - 2), exactly
    slope = (np.log10(lum[1]) - np.log10(lum[0])) / (np.log10(t[1]) - np.log10(t[0]))
    assert slope == pytest.approx(-4.0 / (n - 2), rel=1e-10)


def test_cocoon_taper_is_stable_and_its_gradient_is_right():
    """``(1 + tanh(x)) / 2`` cancels: it is noise below 1e-8 of the peak and zero beyond 18 d, and
    the gradient through it disagrees with finite differences on most draws. The sigmoid form is
    exact there, and the gradient agrees with central differences on every parameter over
    0.01 to 100 d (worst relative error over 40 draws, 7 parameters, below 1e-6)."""
    names = ("mej", "vej", "eta", "tshock", "shocked_fraction", "cos_theta_cocoon", "kappa")
    rng = np.random.default_rng(SEED + 10)
    t = jnp.asarray(np.geomspace(0.01, 100.0, 40))

    def log_lum(th, shipped=False):
        lum, t_thin, _ = SM.shocked_cocoon_engine(t, *[th[i] for i in range(7)])
        if shipped:                       # the ``tanh`` form redback evaluates (the control)
            lum = lum / jax.nn.sigmoid(2.0 * (t_thin - t)) * (1.0 + jnp.tanh(t_thin - t)) / 2.0
        return jnp.sum(jnp.log10(jnp.maximum(lum, 1e-300)))

    fns = {sh: (jax.jit(lambda x, sh=sh: log_lum(x, sh)),
                jax.jit(jax.grad(lambda x, sh=sh: log_lum(x, sh)))) for sh in (False, True)}
    worst, bad_shipped = 0.0, 0
    for _ in range(40):
        th = jnp.asarray([rng.uniform(1e-4, 0.1), rng.uniform(0.1, 0.7), rng.uniform(1, 5),
                          rng.uniform(1, 2), rng.uniform(0.1, 1), rng.uniform(0.2, 0.9),
                          rng.uniform(1, 30)])
        for shipped, (f, grad) in fns.items():
            g = np.asarray(grad(th))
            for i in range(7):
                h, e = 1e-5, np.zeros(7)
                e[i] = 1.0
                fd = (float(f(th * jnp.exp(h * e))) - float(f(th * jnp.exp(-h * e)))) / (2 * h)
                rel = abs(g[i] * float(th[i]) - fd) / max(abs(fd), 1e-3)
                if shipped:
                    bad_shipped += rel > 0.1
                else:
                    worst = max(worst, rel)
    assert worst < 1e-6, worst
    assert bad_shipped > 20, "the control no longer reproduces the tanh form's gradient defect"

    # the deep taper, 15 d past the transparency time: positive, and the exact sigmoid
    args = (0.01, 0.3, 3.0, 1.5, 0.5, 0.7, 10.0)
    _, t_thin, _ = SM.shocked_cocoon_engine(jnp.array(1.0), *args)
    t_thin = float(t_thin)
    at_peak = float(SM.shocked_cocoon_engine(jnp.array(t_thin), *args)[0])
    deep = float(SM.shocked_cocoon_engine(jnp.array(t_thin + 15.0), *args)[0])
    sigmoid = np.exp(-30.0) / (1.0 + np.exp(-30.0))                  # sigma(2 (t_thin - t))
    expected = at_peak * ((t_thin + 15.0) / t_thin) ** (-4.0 / 5.0) * sigmoid / 0.5   # eta = 3
    assert deep > 0.0 and deep == pytest.approx(expected, rel=1e-10)


# =======================================================================================
# 4. the ODE
# =======================================================================================

def test_without_diffrax_the_fallback_is_rk4_with_one_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffrax", None)             # import diffrax -> ImportError
    monkeypatch.setattr(M, "_warned", [])
    with pytest.warns(UserWarning, match=r"RK4.*1000.*log-spaced.*\[ode\]") as rec:
        assert M.resolve_solver("auto") == "rk4"
    assert len(rec) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")                             # the second call is silent
        assert M.resolve_solver("auto") == "rk4"
    with pytest.raises(ImportError, match=r"whisper_cbpf\[ode\]"):
        M.resolve_solver("diffrax")
    assert M.resolve_solver("rk4") == "rk4" and M.resolve_solver("euler") == "euler"
    with pytest.raises(ValueError, match="solver"):
        M.resolve_solver("heun")
    # the fallback is a working solver: a light curve comes out of the auto path
    p = _draws("general_magnetar_driven", 1)[0]
    lum, vej, ok = M.magnetar_ode_solve(jnp.array([5.0, 50.0]), *_ode_args(p))
    assert bool(ok) and bool(jnp.all(jnp.isfinite(lum))) and bool(jnp.all(vej > 0))


def test_auto_picks_diffrax_when_it_is_installed():
    if not HAS_DIFFRAX:
        pytest.skip("diffrax is not installed")
    assert M.resolve_solver("auto") == "diffrax"


def test_magnetar_energy_budget_and_the_spin_down_integral():
    """The radiated energy cannot exceed the supernova energy plus the spin-down energy
    ``l0 tau (n - 1) / 2`` (the closed-form integral of the injected luminosity) plus the nickel
    energy: over 30 prior draws the ratio is at most 0.54 (median 0.02)."""
    tau, l0 = 1e5, 1e45
    for n in (2.0, 3.0, 5.0):
        s = np.geomspace(1e-3, 1e12, 400000)
        injected = _trapezoid(l0 * (1.0 + s / tau) ** ((1 + n) / (1 - n)), s)
        assert injected == pytest.approx(l0 * tau * (n - 1) / 2, rel=1e-3)

    draws = _draws("general_magnetar_driven", 30, SEED + 11)
    t_d = np.geomspace(1.0, 1e8, 600) / 86400.0
    lum, ok = _ode_batch(draws, t_d, solver="rk4")
    assert ok.all()
    ratios = []
    for p, row in zip(draws, lum):
        radiated = _trapezoid(10.0 ** row, t_d * 86400.0)
        e_ni = p["f_nickel"] * p["mej"] * (6.45e43 * 8.8 * 86400.0 + 1.45e43 * 111.3 * 86400.0)
        ratios.append(radiated / (p["E_sn"] + p["l0"] * p["tau_sd"] * (p["nn"] - 1) / 2 + e_ni))
    assert max(ratios) < 1.0 and np.median(ratios) < 0.2, (max(ratios), np.median(ratios))


def test_magnetar_nickel_limit_is_arnetts_rule():
    """Without a magnetar the luminosity peaks where it equals the nickel heating rate (Arnett's
    rule): L_peak / L_Ni(t_peak) within 6% for four ejecta (measured 1.03, 1.03, 0.995, 1.06)."""
    t = np.geomspace(1.0, 300.0, 120)
    for mej, f_ni, kappa in [(2.0, 0.05, 0.1), (5.0, 0.1, 0.1), (10.0, 0.2, 0.1), (1.0, 0.03, 0.2)]:
        lum, _, _ = M.magnetar_ode_solve(jnp.asarray(t), mej, 1e51, kappa, 1e30, 1e5, 3.0, 0.03, f_ni,
                                         solver="rk4", n_grid=4000)
        lum = np.asarray(lum)
        peak = t[np.argmax(lum)]
        l_ni = f_ni * mej * (6.45e43 * np.exp(-peak / 8.8) + 1.45e43 * np.exp(-peak / 111.3))
        assert lum.max() / l_ni == pytest.approx(1.0, abs=0.07), (mej, lum.max() / l_ni)


@pytest.mark.slow
def test_rk4_fallback_accuracy_over_300_prior_draws():
    """The documented fallback, 300 draws x 40 epochs against 20 000 steps (bright points): at 1000
    steps median 6.4e-6, p99 1e-4, max 2.6e-3 dex; none non-finite. Few steps are unstable (one draw
    in 300 is off by 18 dex at 250)."""
    t = np.geomspace(0.3, 400.0, 40)
    draws = _draws("general_magnetar_driven", 300, 1234)
    ref, _ = _ode_batch(draws, t, solver="rk4", n_grid=20000)
    out, ok = _ode_batch(draws, t, solver="rk4")
    assert ok.all() and np.all(np.isfinite(out))
    bright = ref > ref.max(axis=1, keepdims=True) - 2
    med, p99, mx = _stats(np.where(bright, out - ref, np.nan))
    assert med < 3e-5 and p99 < 5e-4 and mx < 6e-3, (med, p99, mx)
    euler, _ = _ode_batch(draws, t, solver="euler", n_grid=20000)   # first order needs 20x the steps
    assert _stats(np.where(bright, euler - ref, np.nan))[0] > 100 * med


# =======================================================================================
# 5. jit / vmap / grad, float32, and the whisper Model
# =======================================================================================

def _finite_gradients(fn, draws, names):
    fv = jax.jit(jax.vmap(jax.grad(lambda th: fn(th))))
    th = jnp.asarray([[p[n] for n in names] for p in draws])
    g = np.asarray(fv(th))
    return np.isfinite(g).all(axis=1), g


def test_all_four_are_finite_under_jit_vmap_and_grad_on_prior_draws():
    t = jnp.asarray(np.geomspace(0.05, 150.0, 20))

    def total(lum):
        return jnp.sum(jnp.log10(jnp.maximum(lum, 1e-30)))

    cases = {
        "csm_interaction": (("mej", "csm_mass", "vej", "eta", "rho", "kappa", "r0"),
                            lambda th: total(C.csm_interaction_bolometric({"time": t}, *th))),
        "shock_cooling": (("log10_mass", "log10_radius", "log10_energy", "nn", "delta"),
                          lambda th: total(S.shock_cooling_model({"time": t}, *th)[0])),
        "shocked_cocoon": (("mej", "vej", "eta", "tshock", "shocked_fraction", "cos_theta_cocoon",
                            "kappa"),
                           lambda th: total(SM.shocked_cocoon_model({"time": t}, *th)[0])),
    }
    for name, (names, fn) in cases.items():
        ok, g = _finite_gradients(lambda th, fn=fn: fn([th[i] for i in range(len(names))]),
                                  _draws(name, 40, SEED + 12), names)
        assert ok.all(), (name, int((~ok).sum()))
    # the ODE: gradients with respect to the log of every parameter, RK4 and Euler
    names = ("mej", "E_sn", "kappa", "l0", "tau_sd", "nn", "kappa_gamma", "f_nickel")
    draws = _draws("general_magnetar_driven", 20, SEED + 12)
    for solver in ("rk4", "euler"):
        def fn(lth, solver=solver):
            th = jnp.exp(lth)
            return total(M.magnetar_ode_solve(t, *[th[i] for i in range(8)], solver=solver)[0])

        lth = jnp.log(jnp.asarray([[p[n] for n in names] for p in draws]))
        g = np.asarray(jax.jit(jax.vmap(jax.grad(fn)))(lth))
        assert np.isfinite(g).all(), solver
    if HAS_DIFFRAX:
        def fn_d(lth):
            th = jnp.exp(lth)
            return total(M.magnetar_ode_solve(t, *[th[i] for i in range(8)], solver="diffrax")[0])

        assert np.isfinite(np.asarray(jax.jit(jax.vmap(jax.grad(fn_d)))(lth))).all()


@pytest.mark.parametrize("name", NEW)
def test_predictions_are_finite_across_the_prior_even_when_the_light_has_gone(name, dl):
    """A light curve that has faded (Piro's luminosity underflows to exactly zero past ~13 diffusion
    times, and so does its temperature) comes back at the magnitude floor, never NaN: 60 prior
    draws, epochs from the model's range out to 600 d, ``predict`` and ``predict_jax``."""
    model = _model(name, dl)
    t, b = _layout(np.concatenate([EPOCHS[name], [100.0, 600.0]]) * (1.0 + Z))
    idx = model.predict_jax.band_index(b)
    faded = 0
    for p in _draws(name, 60, SEED + 14):
        flux = np.asarray(model.predict(p, t, b))
        theta = np.array([p[k] for k in model.parameters])
        jflux = np.asarray(model.predict_jax(theta, t, idx))
        assert np.isfinite(flux).all() and np.isfinite(jflux).all(), (name, p)
        faded += int(np.any(-2.5 * np.log10(flux / 3631.0) > 39.9))
    assert faded > 0 or name == "general_magnetar_driven"       # the control: some draws did fade


def test_ode_gradient_agrees_with_central_differences():
    names = ("mej", "E_sn", "kappa", "l0", "tau_sd", "nn", "kappa_gamma", "f_nickel")
    t = jnp.asarray(np.geomspace(2.0, 150.0, 10))
    p = dict(mej=5.0, E_sn=1e51, kappa=0.1, l0=1e44, tau_sd=1e5, nn=3.0, kappa_gamma=0.1, f_nickel=0.05)
    lth = jnp.log(jnp.asarray([p[n] for n in names]))

    def f(lth_):
        th = jnp.exp(lth_)
        return jnp.sum(jnp.log10(M.magnetar_ode_solve(t, *[th[i] for i in range(8)], solver="rk4")[0]))

    g = np.asarray(jax.jit(jax.grad(f))(lth))
    fj = jax.jit(f)
    for i in range(8):
        e = np.zeros(8)
        e[i] = 1.0
        fd = (float(fj(lth + 1e-5 * e)) - float(fj(lth - 1e-5 * e))) / 2e-5
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-6), (names[i], g[i], fd)


def test_float32_fails_loudly_and_names_float64():
    t = jnp.array([1.0, 2.0, 5.0])
    p = _draws("csm_interaction", 1)[0]
    q = _draws("general_magnetar_driven", 1)[0]
    with jax.enable_x64(False):
        for call in (
            lambda: C.csm_interaction_bolometric({"time": t}, *_csm_args(p)),
            lambda: SM.shocked_cocoon_model({"time": t}, 0.01, 0.3, 3.0, 1.5, 0.5, 0.7, 10.0),
            lambda: S.shock_cooling_model({"time": t}, -1.0, 13.0, 47.0, 10.0, 1.1),
            lambda: M.magnetar_ode_solve(t, *_ode_args(q), solver="rk4"),
            lambda: M.general_magnetar_driven_model({"time": t}, *_ode_args(q), temperature_floor=5e3),
        ):
            with pytest.raises(RuntimeError, match="float64"):
                call()


@pytest.mark.parametrize("name", NEW)
def test_whisper_model_is_consistent(name, dl):
    model = _model(name, dl)
    assert model.parameters == list(S.PARAMETERS[name])
    assert set(model.default_prior.distributions) == set(model.parameters)
    p = _draws(name, 1)[0]
    t, b = _layout(EPOCHS[name] * (1.0 + Z))
    flux = np.asarray(model.predict(p, t, b))
    assert flux.shape == t.shape and np.all(flux >= 0) and np.any(flux > 0)
    theta = np.array([p[k] for k in model.parameters])
    jflux = np.asarray(model.predict_jax(theta, t, model.predict_jax.band_index(b)))
    np.testing.assert_allclose(jflux, flux, rtol=1e-10)
    clone = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(np.asarray(clone.predict(p, t, b)), flux, rtol=1e-12)


@pytest.mark.parametrize("name", NEW)
def test_free_explosion_time_and_redshift_reproduce_the_data_grid(name, dl):
    """These four are evaluated at the epochs themselves, so the traced explosion time and redshift
    give the data-grid answer to round-off, and an epoch before the explosion has no light."""
    eo = {"engine_options": {"solver": "rk4"}} if name == "general_magnetar_driven" else {}
    data = _model(name, dl)
    free = supernova_model(name, BANDS, free=["t_exp", "redshift"],
                           prior=Prior({"t_exp": Uniform(-3.0, 0.0)}),
                           redshift_prior={"type": "Uniform", "low": 0.01, "high": 0.2}, **eo)
    t, b = _layout(EPOCHS[name] * (1.0 + Z))
    for p in _draws(name, 20, SEED + 13):                 # the first draw that is visible
        ref = _mags(data, p, t, b)
        if (ref < 39.9).sum() > 5:
            break
    got = _mags(free, dict(p, t_exp=0.0, redshift=Z), t, b)
    ok = ref < 39.9
    assert ok.sum() > 5
    np.testing.assert_allclose(got[ok], ref[ok], atol=1e-6)
    before = _mags(free, dict(p, t_exp=EPOCHS[name][0] * (1 + Z) + 1.0, redshift=Z), t[:1], b[:1])
    assert float(before[0]) == pytest.approx(40.0)
    # and an epoch at 900 d is not refused by a grid that ends at 200 d
    far = free.predict(dict(p, t_exp=-1.0, redshift=Z), np.array([900.0]), np.array(["lsstr"]))
    assert np.isfinite(np.asarray(far)).all()


def test_pin_and_engine_options_reach_the_engines(dl):
    """``nn`` of the CSM is pinned at redback's 12 and can be pinned elsewhere or freed; the diffusion
    method and the ODE solver are options."""
    m12 = _model("csm_interaction", dl)
    m10 = _model("csm_interaction", dl, pin={"nn": 10.0})
    mfree = _model("csm_interaction", dl, pin={"nn": None}, prior=Prior({"nn": Uniform(8.0, 13.0)}))
    mquad = _model("csm_interaction", dl, engine_options={"csm_diffusion_method": "quadrature"})
    assert "nn" not in m12.parameters and "nn" in mfree.parameters
    p = _draws("csm_interaction", 1)[0]
    t, b = _layout(EPOCHS["csm_interaction"] * (1 + Z))
    a12, a10 = _mags(m12, p, t, b), _mags(m10, p, t, b)
    assert np.max(np.abs(a12 - a10)) > 0.01
    np.testing.assert_allclose(_mags(mfree, dict(p, nn=12.0), t, b), a12, atol=1e-10)
    quad = _mags(mquad, p, t, b)
    ok = (a12 < 39.9) & (quad < 39.9)
    assert np.max(np.abs(quad - a12)[ok]) < 0.5            # the same integral, the legacy quadrature
    with pytest.raises(ValueError, match="not settings"):
        _model("shocked_cocoon", dl, engine_options={"solver": "rk4"})


def test_docstring_examples_run():
    """The examples in the module and function docstrings, in a fresh interpreter with x64 on (they
    enable it themselves; running them here would leave the flag set for the session)."""
    code = """
        import doctest, jax
        jax.config.update("jax_enable_x64", True)
        from whisper_cbpf.models.jax import csm_interaction, magnetar_ode, shock_models, supernova
        failed = attempted = 0
        for mod in (csm_interaction, magnetar_ode, shock_models):
            r = doctest.testmod(mod, optionflags=doctest.ELLIPSIS, verbose=False)
            failed += r.failed; attempted += r.attempted
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
        for test in doctest.DocTestFinder().find(supernova.shock_cooling_model, "shock_cooling_model", globs={}):
            runner.run(test)
        failed += runner.summarize(verbose=False).failed
        attempted += runner.tries
        print(failed, attempted)
        """
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    failed, attempted = map(int, out.stdout.strip().splitlines()[-1].split())
    assert attempted >= 12 and failed == 0, out.stdout[-2000:]
