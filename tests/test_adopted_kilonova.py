"""The Metzger (2017) multi-shell kilonova and the magnetar-boosted kilonova, as tests.

``whisper_cbpf.models.jax.kilonova_metzger`` ports redback 1.20's ``_metzger_kilonova_model`` and
``_general_metzger_magnetar_driven_kilonova_model`` (adapted from redback-jax and redback). The
claims, and what each test measures:

1. **The port is redback, to rounding, on redback's grid.** The same Euler scheme on the grid
   redback builds gives redback's luminosity, temperature and photosphere radius to 7e-15, and the
   band magnitudes (redback's own temperature/radius interpolation, blackbody, ``(1 + z)`` and
   whisper's band integral) to 6e-14 mag.
2. **On its own fixed grid it is redback with redback's discretisation error removed.** Euler is
   first order, and redback's grids (500 and 300 nodes) carry that error; the fixed grids are
   finer. 50 or more seeded prior draws: the differences against redback are redback's own error
   (budgets below, measured), and the differences against a 8x finer grid are an order of
   magnitude smaller.
3. **Nothing depends on the epoch array**: not where it starts, not what else is in it.
4. ``jit`` / ``vmap`` / ``grad`` are finite; float32 agrees with float64; the radiated energy
   never exceeds the injected energy.
5. The whisper ``Model``: parameters, priors, free ``t_exp`` and redshift, pickling, doctests.

float64 is switched on per test (and restored), never for the session.
"""
from __future__ import annotations

import doctest
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.io.photometry import AB_ZEROPOINT_JY  # noqa: E402
from whisper_cbpf.models.jax import kilonova as kn  # noqa: E402
from whisper_cbpf.models.jax import kilonova_metzger as km  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400.0
BANDS = ["lsstg", "lsstr", "lssti", "lsstz", "ztfg", "ztfr"]
EPOCHS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
REDSHIFTS = (0.0098, 0.05)
N_DRAWS = 50
KINDS = ("metzger", "magnetar")
NAMES = {"metzger": ("mej", "vej", "beta", "kappa"),
         "magnetar": ("mej", "vej", "beta", "kappa", "p0", "bp", "mass_ns", "theta_pb",
                      "thermalisation_efficiency")}


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _prior(kind):
    return km.default_prior_metzger() if kind == "metzger" else km.default_prior_magnetar_boosted()


def _draws(kind, n=N_DRAWS, seed=2026):
    rng = np.random.default_rng(seed + (kind == "magnetar"))
    prior = _prior(kind)
    return np.array([[prior.sample(rng)[k] for k in NAMES[kind]] for _ in range(n)])


def _dl(z):
    from whisper_cbpf.models.cosmology import luminosity_distance_cm

    return float(luminosity_distance_cm(z))


def _model(kind, z, **kw):
    make = km.metzger_kilonova_model if kind == "metzger" else km.magnetar_boosted_kilonova_model
    return make(BANDS, z, _dl(z), **kw)


def _stats(x):
    x = np.abs(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(x.max())


# --- redback, the arbiter ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def rb():
    pytest.importorskip("redback")
    from whisper_cbpf.models.redback_adapter import _import_redback

    _import_redback()
    from redback.transient_models import kilonova_models as kmod
    from redback.transient_models import magnetar_driven_ejecta_models as om
    from redback.transient_models.magnetar_models import basic_magnetar
    from redback.utils import get_optimal_time_array

    def grid(kind, t_src_s=None):
        """The grid redback's model function builds."""
        if kind == "metzger":
            return get_optimal_time_array(1e-2, 7e6, 500, user_times=t_src_s)
        return get_optimal_time_array(1e-4, 1e7, 300)

    def bolometric(kind, p, tsec):
        """redback's (L, T, R) on ``tsec``."""
        if kind == "metzger":
            return kmod._metzger_kilonova_model(tsec, *p)
        lsd = basic_magnetar(time=tsec, p0=p[4], bp=p[5], mass_ns=p[6], theta_pb=p[7])
        out = om._general_metzger_magnetar_driven_kilonova_model(
            tsec, *p[:4], lsd, False, thermalisation_efficiency=p[8])
        return out.bolometric_luminosity, out.temperature, out.r_photosphere

    def mjy(kind, z, p, t, nu):
        fn = (kmod.metzger_kilonova_model if kind == "metzger"
              else om.metzger_magnetar_driven_kilonova_model)
        return np.asarray(fn(t, z, *p, output_format="flux_density", frequency=nu))

    return SimpleNamespace(grid=grid, bolometric=bolometric, mjy=mjy)


def _band_mags(flux_nu, t, b, fs):
    """AB magnitudes from ``flux_nu(t_obs_days, nu_obs_hz) -> Jy``, whisper's band integral."""
    from whisper_cbpf.synphot import band_flux_jy

    with np.errstate(all="ignore"):
        return -2.5 * np.log10(band_flux_jy(flux_nu, t, b, fs) / AB_ZEROPOINT_JY)


def _filters():
    from whisper_cbpf.synphot import filter_set_for

    return filter_set_for(BANDS)


def _layout():
    return np.repeat(EPOCHS, len(BANDS)), np.tile(np.array(BANDS), len(EPOCHS))


def _ours_on_redbacks_grid(rb, kind, p, z):
    """Band flux [Jy] from the port's engine on the grid redback builds for these nodes.

    The distance is astropy's Planck18, as redback's: the model's tabulated distance differs from
    it by up to 8e-7 mag, which would hide the 1e-13 agreement this is for."""
    from astropy.cosmology import Planck18

    dl = float(Planck18.luminosity_distance(z).cgs.value)

    def flux(tt, nu):
        t_src = tt * DAY / (1.0 + z)
        grid = rb.grid(kind, t_src)
        engine = km.metzger_engine if kind == "metzger" else km.magnetar_boosted_engine
        _, temp, rad = km._on_epochs(grid, *engine(grid, *p), t_src)
        return np.asarray(kn._flux_nu(temp, rad, dl, nu * (1.0 + z), z)) / 1e-23     # cgs -> Jy

    return flux


# --- 1. the port is redback on redback's grid --------------------------------------------------

#: Measured over 12 draws per model: luminosity 7.1e-15 dex, temperature 5.5e-15 and radius
#: 6.7e-15 relative -- the last node, which redback leaves at zero, excluded.
@pytest.mark.parametrize("kind", KINDS)
def test_engine_equals_redback_on_redbacks_grid(x64, rb, kind):
    grid = rb.grid(kind, EPOCHS * DAY)
    engine = km.metzger_engine if kind == "metzger" else km.magnetar_boosted_engine
    for p in _draws(kind, n=12, seed=7):
        lum, temp, rad = rb.bolometric(kind, p, grid)
        logl, t_ours, r_ours = (np.asarray(x) for x in engine(grid, *p))
        assert np.abs(logl[:-1] - np.log10(lum[:-1])).max() < 1e-11
        assert np.abs(t_ours[:-1] / temp[:-1] - 1.0).max() < 1e-11
        assert np.abs(r_ours[:-1] / rad[:-1] - 1.0).max() < 1e-11


#: Measured over 30 draws at both redshifts: 5.7e-14 mag (Metzger), 6.4e-14 (magnetar).
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("z", REDSHIFTS)
def test_photometry_equals_redback_on_redbacks_grid(x64, rb, kind, z):
    t, b = _layout()
    fs = _filters()
    worst, compared = 0.0, 0
    for p in _draws(kind, n=10, seed=8):
        theirs = _band_mags(lambda tt, nu: 1e-3 * rb.mjy(kind, z, p, tt, nu), t, b, fs)
        ours = _band_mags(_ours_on_redbacks_grid(rb, kind, p, z), t, b, fs)
        ok = theirs < 35
        if ok.any():
            worst, compared = max(worst, np.abs(ours - theirs)[ok].max()), compared + int(ok.sum())
    assert compared > 100 and worst < 1e-9, (compared, worst)


# --- 2. the fixed grid ---------------------------------------------------------------------------

#: Measured over 60 draws (window 0.3 to 40 d at redback's nodes): redback's Euler error against
#: ours, median / max in dex: Metzger 0.0092 / 0.080 (redback: 500 nodes), magnetar 0.025 / 0.170
#: (redback: 300 nodes, ~75 of them between 0.2 and 116 d).
@pytest.mark.parametrize("kind,budget_median,budget_max", [("metzger", 0.02, 0.12),
                                                            ("magnetar", 0.05, 0.25)])
def test_bolometric_matches_redback_within_redbacks_discretisation(x64, rb, kind, budget_median,
                                                                   budget_max):
    tsec = rb.grid(kind, EPOCHS * DAY)
    win = (tsec > 0.3 * DAY) & (tsec < 40 * DAY)
    bolo = km.metzger_bolometric if kind == "metzger" else km.magnetar_boosted_bolometric
    ours = jax.jit(jax.vmap(lambda p: bolo(tsec, *p)[0]))(jnp.asarray(_draws(kind)))
    err = []
    for p, row in zip(_draws(kind), np.asarray(ours)):
        lum = rb.bolometric(kind, p, tsec)[0]
        err.append((row - np.log10(lum))[win])
    median, worst = _stats(np.concatenate(err))
    assert median < budget_median and worst < budget_max, (median, worst)


#: Measured over 60 draws, 8 epochs x 6 bands, points with redback magnitude < 35; our fixed grid
#: against redback (median / max mag): Metzger z=0.0098 0.081 / 0.213, z=0.05 0.082 / 0.211;
#: magnetar 0.093 / 2.37 and 0.083 / 1.99. These are redback's own errors: redback interpolates a
#: photosphere radius that jumps between shells linearly across 8% time steps, which is where the
#: magnetar's worst cases come from (its 300 nodes carry only ~75 between 0.2 and 116 d). On
#: redback's grid the difference is 6e-14 mag (the test above).
@pytest.mark.parametrize("kind,budget_median,budget_max", [("metzger", 0.12, 0.35),
                                                            ("magnetar", 0.15, 4.0)])
@pytest.mark.parametrize("z", REDSHIFTS)
def test_magnitudes_match_redback_within_redbacks_discretisation(x64, rb, kind, budget_median,
                                                                 budget_max, z):
    t, b = _layout()
    fs = _filters()
    model = _model(kind, z)
    err = []
    for p in _draws(kind):
        theirs = _band_mags(lambda tt, nu: 1e-3 * rb.mjy(kind, z, p, tt, nu), t, b, fs)
        ours = -2.5 * np.log10(model.predict(dict(zip(model.parameters, p)), t, b)
                               / AB_ZEROPOINT_JY)
        ok = (theirs < 35) & (ours < 35)
        err.append((ours - theirs)[ok])
    median, worst = _stats(np.concatenate(err))
    assert median < budget_median and worst < budget_max, (median, worst)


#: Euler's error is first order, so a finer grid must land closer to the converged answer: on 20
#: draws against a 16000-node reference, 2000 and 4000 (Metzger) and 4000 and 8000 (magnetar)
#: nodes. Measured (dex, maximum over draws and epochs): Metzger 0.0094 and 0.0041, magnetar
#: 0.011 and 0.0037, i.e. the error halves per doubling.
@pytest.mark.slow
@pytest.mark.parametrize("kind,counts", [("metzger", (2000, 4000)), ("magnetar", (4000, 8000))])
def test_fixed_grid_is_converged_and_first_order(x64, kind, counts):
    lo = 1e-2 if kind == "metzger" else 1e-4
    engine = km.metzger_engine if kind == "metzger" else km.magnetar_boosted_engine
    epochs = np.geomspace(0.2, 40.0, 12) * DAY
    p = jnp.asarray(_draws(kind, n=20, seed=3))

    def on(n):
        g = np.geomspace(lo, 1e7, n)
        return np.asarray(jax.jit(jax.vmap(lambda q: km._on_epochs(g, *engine(g, *q), epochs)[0]))(p))

    ref = on(16000)
    e1, e2 = (np.abs(on(n) - ref) for n in counts)
    assert e1.max() < 0.02 and e2.max() < 0.8 * e1.max() and np.median(e2) < 0.7 * np.median(e1)
    assert counts[0] == (km.N_METZGER if kind == "metzger" else km.N_MAGNETAR)


def test_the_fixed_grids_are_the_documented_ones():
    assert km.METZGER_GRID.size == km.N_METZGER and km.MAGNETAR_GRID.size == km.N_MAGNETAR
    assert (km.METZGER_GRID[0], km.METZGER_GRID[-1]) == (1e-2, 1e7)
    assert (km.MAGNETAR_GRID[0], km.MAGNETAR_GRID[-1]) == (1e-4, 1e7)
    for grid in (km.METZGER_GRID, km.MAGNETAR_GRID):
        step = np.diff(np.log(grid))
        assert np.allclose(step, step[0]) and step[0] < 0.011     # < 1.1% per node: Euler-stable


# --- 3. independence of the epoch array ---------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_luminosity_does_not_depend_on_the_epoch_array(x64, kind):
    bolo = km.metzger_bolometric if kind == "metzger" else km.magnetar_boosted_bolometric
    p = _draws(kind, n=3, seed=11)
    core = np.array([1.0, 2.5, 6.0, 15.0]) * DAY
    for q in p:
        want = [np.asarray(x) for x in bolo(core, *q)]
        for other in (np.concatenate([[1e-3, 0.05 * DAY, 0.3 * DAY], core]),
                      np.concatenate([core, [40.0 * DAY, 90.0 * DAY]]),
                      np.concatenate([core[::-1], [0.7 * DAY]])):
            got = [np.asarray(x) for x in bolo(other, *q)]
            sel = np.array([np.flatnonzero(other == c)[0] for c in core])
            for g, w in zip(got, want):
                assert np.array_equal(g[sel], w)


@pytest.mark.parametrize("kind", KINDS)
def test_magnitude_at_an_epoch_does_not_depend_on_the_others(x64, kind):
    model = _model(kind, 0.0098)
    p = dict(zip(model.parameters, _draws(kind, n=1, seed=5)[0]))
    t, b = np.array([1.0, 4.0]), np.array(["lsstr", "ztfg"])
    base = model.predict(p, t, b)
    for extra_t in (np.array([0.02, 0.1]), np.array([30.0, 60.0])):
        more = model.predict(p, np.concatenate([extra_t, t]),
                             np.concatenate([np.array(["lssti", "lsstz"]), b]))
        assert np.array_equal(more[2:], base)


# --- 4. jit, vmap, grad; float32; energy ---------------------------------------------------------

def _loss(model, t, bidx):
    return lambda theta: jnp.sum(model.predict_jax(theta, t, bidx) * 1e6)


@pytest.mark.parametrize("kind", KINDS)
def test_jit_vmap_and_grad_are_finite(x64, kind):
    model = _model(kind, 0.0098)
    t = np.array([0.7, 1.5, 3.0, 6.0, 12.0])
    bidx = model.predict_jax.band_index(np.array(["lsstg", "lsstr", "lssti", "ztfg", "ztfr"]))
    theta = jnp.asarray(_draws(kind, n=16, seed=21))
    flux = jax.jit(jax.vmap(lambda q: model.predict_jax(q, t, bidx)))(theta)
    grad = jax.jit(jax.vmap(jax.grad(_loss(model, t, bidx))))(theta)
    assert flux.shape == (16, 5) and bool(jnp.all(jnp.isfinite(flux))) and bool(jnp.all(flux > 0))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_gradient_matches_finite_differences(x64):
    """In a smooth region (the photosphere sits in one shell): mej, kappa and beta of the Metzger
    model, and the magnetar's spin period."""
    for kind, theta, probes in (
            ("metzger", [0.03, 0.2, 4.0, 10.0], (0, 3, 2)),
            ("magnetar", [0.03, 0.2, 4.0, 10.0, 2.0, 1.0, 1.4, 1.0, 0.5], (0, 4))):
        model = _model(kind, 0.0098)
        t = np.array([2.0, 5.0])
        bidx = model.predict_jax.band_index(np.array(["lsstr", "lsstr"]))
        f = _loss(model, t, bidx)
        th = jnp.asarray(theta)
        g = np.asarray(jax.grad(f)(th))
        for i in probes:
            h = 1e-5 * theta[i]
            up, dn = th.at[i].add(h), th.at[i].add(-h)
            fd = float((f(up) - f(dn)) / (2 * h))
            assert abs(g[i] - fd) < 2e-3 * abs(fd) + 1e-12, (kind, i, g[i], fd)


#: Measured over 30 prior draws, 8 epochs x 6 bands, float32 (x64 off) against float64:
#: Metzger 4.8e-5 mag maximum (median 8e-6), magnetar 1.9e-4 (median 1.2e-5); engine luminosity
#: from t = 0.01 s on: 4.3e-6 dex (Metzger), 4.4e-6 (magnetar).
@pytest.mark.parametrize("kind", KINDS)
def test_float32_agrees_with_float64(kind):
    t, b = _layout()
    draws = _draws(kind, n=30, seed=17)
    engine = km.metzger_engine if kind == "metzger" else km.magnetar_boosted_engine
    grid = km.METZGER_GRID if kind == "metzger" else km.MAGNETAR_GRID

    def run(enabled):
        with jax.enable_x64(enabled):
            model = _model(kind, 0.0098)
            names = model.parameters
            mags = np.array([-2.5 * np.log10(model.predict(dict(zip(names, p)), t, b)
                                             / AB_ZEROPOINT_JY) for p in draws])
            logl = np.asarray(jax.jit(jax.vmap(lambda q: engine(grid, *q)[0]))(jnp.asarray(draws)))
            return mags, logl, logl.dtype

    low, logl_low, dtype_low = run(False)
    high, logl_high, dtype_high = run(True)
    assert dtype_low == np.float32 and dtype_high == np.float64
    ok = high < 35
    assert np.isfinite(low).all() and np.abs(low - high)[ok].max() < 1e-3
    inner = (grid[:-1] >= 1e-2)       # earlier, the magnetar's luminosity is below float32's range
    assert np.abs(logl_low[:, :-1][:, inner] - logl_high[:, :-1][:, inner]).max() < 5e-5


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_are_finite_in_float32(kind):
    with jax.enable_x64(False):
        model = _model(kind, 0.0098)
        t = np.array([1.0, 3.0, 8.0])
        bidx = model.predict_jax.band_index(np.array(["lsstg", "lsstr", "ztfr"]))
        theta = jnp.asarray(_draws(kind, n=12, seed=19), dtype=jnp.float32)
        grad = jax.jit(jax.vmap(jax.grad(_loss(model, t, bidx))))(theta)
        assert grad.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(grad)))


def _injected_energy(kind, p):
    """Energy put into the ejecta by the heating terms, erg: the sum of the shells' initial
    energy and of everything the Euler scheme adds, in numpy float64."""
    grid = km.METZGER_GRID if kind == "metzger" else km.MAGNETAR_GRID
    dt = np.diff(grid)
    mej, vej, beta, kappa = p[:4]
    er = np.asarray(km._heating(grid, mej, vej))[:-1]
    vel, m = (np.asarray(x) for x in km._shells(mej, vej, beta, 0.7, normalise=kind == "magnetar"))
    xn0 = np.asarray(km._neutron_layer(m[:-1], kappa)[0])
    dm = np.abs(np.diff(m)) * km._MSUN
    # the neutron term as the scheme adds it: the sum over nodes of edot_n(t_n) dt_n
    per_gram = (er * dt).sum() + 3.2e14 * xn0 * (np.exp(-grid[:-1] / 900.0) * dt).sum()
    e0 = 0.5 * m[:-1] * (vel[:-1] * km._C) ** 2
    total = ((e0 + per_gram) * dm).sum()
    if kind == "magnetar":
        p0, bp, mass_ns, theta, eff = p[4:]
        erot = 2.6e52 * (mass_ns / 1.4) ** 1.5 * p0 ** -2
        tp = 1.3e5 * bp ** -2 * p0 ** 2 * (mass_ns / 1.4) ** 1.5 / np.sin(theta) ** 2
        lsd = 2.0 * erot / tp / (1.0 + 2.0 * grid[:-1] / tp) ** 2
        total += eff * (lsd * dt).sum()
    return total


@pytest.mark.parametrize("kind", KINDS)
def test_radiated_energy_never_exceeds_the_injected_energy(x64, kind):
    """Left Riemann sum of L over the nodes is exactly what the scheme radiates; with the shell
    energies non-negative it cannot exceed what was injected. The trapezoid, which is what a user
    integrating the light curve would take, to 5 %. Adiabatic losses dominate, so the fraction is
    well below one (measured, left sum over injected: Metzger 2.7e-4 median, 5.7e-4 maximum --
    redback's initial shell energy ``0.5 m v^2`` is most of what is injected; magnetar 1.8e-3
    median, 0.49 maximum)."""
    engine = km.metzger_engine if kind == "metzger" else km.magnetar_boosted_engine
    grid = km.METZGER_GRID if kind == "metzger" else km.MAGNETAR_GRID
    dt = np.diff(grid)
    fractions = []
    for p in _draws(kind, n=20, seed=23):
        lum = 10.0 ** np.asarray(engine(grid, *p)[0])[:-1]
        injected = _injected_energy(kind, p)
        left, trapezoid = (lum * dt).sum(), 0.5 * ((lum[1:] + lum[:-1]) * dt[:-1]).sum()
        assert left <= injected * (1.0 + 1e-9), (p, left, injected)
        assert trapezoid <= injected * 1.05
        fractions.append(left / injected)
    assert 1e-5 < np.median(fractions) < 0.5


# --- 5. the whisper Model ----------------------------------------------------------------------

def test_parameters_priors_and_names():
    m = _model("metzger", 0.0098)
    assert m.name == "metzger_kilonova_jax"
    assert m.parameters == ["mej", "vej", "beta", "kappa"] and m.param_aliases == {}
    d = m.default_prior.distributions
    assert [d[k].bounds for k in m.parameters] == [(0.01, 0.05), (0.1, 0.4), (3.0, 5.0), (1.0, 30.0)]
    g = _model("magnetar", 0.0098)
    assert g.name == "metzger_magnetar_driven_kilonova_jax" and g.param_aliases == {"kappa": "kappa_r"}
    assert g.parameters == list(NAMES["magnetar"])
    d = g.default_prior.distributions
    assert [d[k].bounds for k in g.parameters] == [
        (1e-3, 0.1), (0.1, 0.4), (3.0, 5.0), (1.0, 30.0), (0.7, 1e4), (1e-4, 1e4), (1.1, 2.2),
        (0.0, 1.57), (0.1, 1.0)]
    assert type(d["p0"]).__name__ == type(d["bp"]).__name__ == "LogUniform"


@pytest.mark.parametrize("kind,file", [("metzger", "metzger_kilonova_model"),
                                        ("magnetar", "metzger_magnetar_driven_kilonova_model")])
def test_default_prior_is_redbacks_file_and_the_fallback_is_its_transcription(kind, file):
    pytest.importorskip("redback")
    from whisper_cbpf.priors import LogUniform

    read = _prior(kind).distributions                      # read from the installed redback
    cls = {"Uniform": Uniform, "LogUniform": LogUniform}
    fallback = {k: cls[c](lo, hi) for k, (c, lo, hi) in km._FALLBACK[file].items()}
    assert list(read) == list(fallback) == list(NAMES[kind])
    for k in read:
        assert type(read[k]) is type(fallback[k]) and read[k].bounds == fallback[k].bounds, k


def test_a_prior_override_may_use_either_spelling_of_kappa():
    for spelling in ("kappa", "kappa_r"):
        g = km.magnetar_boosted_kilonova_model(["lsstg"], 0.01, _dl(0.01),
                                               prior=Prior({spelling: Uniform(2.0, 3.0)}))
        assert g.default_prior.distributions["kappa"].bounds == (2.0, 3.0)
    with pytest.raises(ValueError, match="not parameters"):
        km.metzger_kilonova_model(["lsstg"], 0.01, _dl(0.01), prior=Prior({"nope": Uniform(0, 1)}))


@pytest.mark.parametrize("kind", KINDS)
def test_predict_equals_predict_jax_and_pickles(kind):
    model = _model(kind, 0.0098)
    p = dict(zip(model.parameters, _draws(kind, n=1, seed=29)[0]))
    t, b = np.array([1.0, 2.0, 5.0]), np.array(["lsstg", "lsstr", "ztfr"])
    flux = model.predict(p, t, b)
    theta = jnp.asarray([p[k] for k in model.parameters])
    jaxed = model.predict_jax(theta, t, model.predict_jax.band_index(b))
    assert np.allclose(flux, np.asarray(jaxed), rtol=1e-5)
    again = pickle.loads(pickle.dumps(model))
    assert np.array_equal(again.predict(p, t, b), flux)


@pytest.mark.parametrize("kind", KINDS)
def test_free_explosion_time_and_redshift(kind):
    prior = Prior({"t_exp": Uniform(-1.0, 0.0), "redshift": Uniform(0.005, 0.05)})
    model = (km.metzger_kilonova_model if kind == "metzger" else km.magnetar_boosted_kilonova_model)(
        ["lsstg", "lsstr"], free=["t_exp", "redshift"], prior=prior)
    assert model.parameters[-2:] == ["t_exp", "redshift"]
    assert model.default_prior.distributions["t_exp"].bounds == (-1.0, 0.0)
    bright = {"metzger": [0.03, 0.2, 4.0, 10.0],
              "magnetar": [0.03, 0.2, 4.0, 10.0, 2.0, 1.0, 1.4, 1.0, 0.5]}[kind]
    p = dict(zip(model.parameters, bright + [-0.3, 0.01]))
    t, b = np.array([-0.5, 1.0, 2.0]), np.array(["lsstg", "lsstr", "lsstg"])
    flux = model.predict(p, t, b)
    assert flux[0] == pytest.approx(AB_ZEROPOINT_JY * 10.0 ** (-0.4 * 40.0), rel=1e-4)   # mag_floor
    assert flux[1] > flux[0] and flux[2] > flux[0]
    # a later merger time is the same light curve shifted
    p2 = dict(p, t_exp=-0.1)
    assert np.allclose(model.predict(p2, t + 0.2, b), flux, rtol=1e-4)
    bidx = model.predict_jax.band_index(b)
    g = jax.grad(lambda th: jnp.sum(model.predict_jax(th, t[1:], bidx[1:]) * 1e6))(
        jnp.asarray([p[k] for k in model.parameters]))
    assert bool(jnp.all(jnp.isfinite(g))) and float(g[-2]) != 0.0 and float(g[-1]) != 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_the_distance_scales_the_flux(kind):
    """No temperature floor and no cutoff: a blackbody, so the flux goes as the inverse square of
    the distance exactly (the redshift dilation held fixed)."""
    make = km.metzger_kilonova_model if kind == "metzger" else km.magnetar_boosted_kilonova_model
    near, far = make(["lsstr"], 0.01, 1e26), make(["lsstr"], 0.01, 2e26)
    bright = {"metzger": [0.03, 0.2, 4.0, 10.0],
              "magnetar": [0.03, 0.2, 4.0, 10.0, 2.0, 1.0, 1.4, 1.0, 0.5]}[kind]
    p = dict(zip(near.parameters, bright))
    t, b = np.array([2.0, 6.0]), np.array(["lsstr", "lsstr"])
    assert np.allclose(near.predict(p, t, b) / far.predict(p, t, b), 4.0, rtol=1e-5)


def test_physically_reasonable_light_curves(x64):
    """A GW170817-like draw: the Metzger kilonova peaks in the first days at 1e41 to 1e42 erg/s,
    and a strong magnetar outshines it at 1 d."""
    day = np.array([0.5, 1.0, 3.0, 10.0]) * DAY
    logl = np.asarray(km.metzger_bolometric(day, 0.03, 0.2, 4.0, 10.0)[0])
    assert 40.5 < logl[1] < 42.5 and logl[3] < logl[1]
    boosted = np.asarray(km.magnetar_boosted_bolometric(
        day, 0.03, 0.2, 4.0, 10.0, 1.0, 1.0, 1.4, 1.2, 0.5)[0])
    assert boosted[1] > logl[1] + 1.0


def test_epochs_past_the_grid_warn_once_and_hold_the_last_node():
    model = _model("metzger", 0.0098)
    p = dict(zip(model.parameters, _draws("metzger", n=1, seed=37)[0]))
    with pytest.warns(RuntimeWarning, match="last node"):
        model.predict(p, np.array([1.0, 200.0]), np.array(["lsstr", "lsstr"]))


def test_register_functions_register_a_usable_model():
    from whisper_cbpf.models import _REGISTRY

    names = ("adopted_metzger_test", "adopted_magnetar_test")
    try:
        a = km.register_metzger_kilonova(["lsstg", "lsstr"], 0.0098, _dl(0.0098), name=names[0])
        b = km.register_magnetar_boosted_kilonova(["lsstg", "lsstr"], 0.0098, _dl(0.0098), name=names[1])
        for name, m in zip(names, (a, b)):
            got = wp.get_model(name)
            assert got.parameters == m.parameters and got.predict_jax is not None
    finally:
        for name in names:
            _REGISTRY.pop(name, None)


def test_docstring_examples_run():
    from whisper_cbpf.models import _REGISTRY

    before = set(_REGISTRY)
    try:
        result = doctest.testmod(km, optionflags=doctest.ELLIPSIS, verbose=False)
        assert result.failed == 0 and result.attempted >= 8, result
    finally:
        for name in set(_REGISTRY) - before:
            _REGISTRY.pop(name, None)


def test_the_module_sets_no_global_precision():
    code = ("import jax; import whisper_cbpf.models.jax.kilonova_metzger as K; "
            "assert not jax.config.jax_enable_x64; K.metzger_bolometric([1e5], 0.03, 0.2, 4.0, 10.0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"}, timeout=300)
    assert done.returncode == 0, done.stderr[-1500:]
