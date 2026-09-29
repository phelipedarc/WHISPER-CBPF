"""Supernova colours at the observed epochs only, on a fixed-resolution grid.

``supernova.ab_magnitude_at`` evaluates the diffusion integral on fixed source-frame epochs
(30 to a decade), interpolates only the diffused bolometric luminosity to the observations, and
evaluates the photosphere and the SED at the observations only. The claims:

1. **At the fixed epochs themselves it IS redback's scheme**: the interpolant passes through the
   epochs, so there it equals ``bolometric`` on the same epochs to round-off.
2. **Terms redback does not diffuse are not interpolated**: shock cooling, the CSM breakout and
   the undiffused fallback engines are evaluated at the observation itself.
3. **The gate against redback**, on prior draws: p95 at most 1e-3 mag and max at most 2e-3 mag
   worse than the current (data-grid) model, which equals redback to <= 8e-6 mag. Over
   200 draws per family against redback's own band integral, the
   worst is the magnetar at 8.8e-4 mag.
4. **Fixed resolution**: the epochs sit at the same log positions whatever the span, so a longer
   ``max_phase_days`` costs epochs, not accuracy; the times may be traced.
5. **Before the explosion there is no light.**

FLOAT64 for the whole module, as the supernova port requires.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.cosmology import luminosity_distance_cm  # noqa: E402

BANDS = ["ztfg", "ztfr"]
Z = 0.05
T_OBS = np.array([3.0, 3.9, 5.1, 7.2, 9.0, 12.4, 15.0, 19.8, 24.1, 30.3, 36.0, 44.7])
B_OBS = np.array(BANDS * 6)
GATE_FAMILIES = ("arnett", "basic_magnetar_powered", "shock_cooling_and_arnett",
                 "csm_shock_and_arnett", "type_1a", "type_1c")


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _mag(flux):
    return -2.5 * np.log10(np.asarray(flux, dtype=float) / 3631.0)


def _pair(model):
    """The current (data-grid) model and the observed-epoch one, same context."""
    from whisper_cbpf.models.jax import supernova_model

    dl = luminosity_distance_cm(Z)
    data = supernova_model(model, BANDS, Z, dl, constraint=None)
    fixed = supernova_model(model, BANDS, Z, dl, constraint=None, diffusion_grid="fixed")
    return data, fixed


def test_fixed_grid_resolution_and_margin():
    from whisper_cbpf.models.jax import supernova as sn

    g = sn.build_fixed_grid(200.0)
    t = np.asarray(g.arrays["time"])
    assert g.n_epochs == t.size
    assert np.allclose(np.diff(np.log10(t)), 1.0 / sn.FIXED_GRID_EPOCHS_PER_DECADE)
    assert t[0] == pytest.approx(sn.FIXED_GRID_FIRST)
    # the last observation keeps real epochs above it for the cubic
    assert sn.FIXED_GRID_MARGIN_EPOCHS <= np.sum(t > 200.0) <= sn.FIXED_GRID_MARGIN_EPOCHS + 1
    assert g.n_epochs == 134                                   # 0.01 to 200 d, plus the margin
    with pytest.raises(ValueError, match="max_phase_days > first_epoch"):
        sn.build_fixed_grid(0.001)


@pytest.mark.parametrize("model", ["arnett", "basic_magnetar_powered", "csm_shock_and_arnett",
                                   "shock_cooling_and_arnett"])
def test_at_the_fixed_epochs_it_is_redbacks_scheme(model):
    """Where tau is an epoch of the grid, interpolation is exact: bolometric_at == bolometric."""
    from whisper_cbpf.models.jax import supernova as sn

    g = sn.build_fixed_grid(60.0)
    tau = np.asarray(g.arrays["time"])[40:120:7]
    t_last = float(tau[-1])
    ref_grid = sn.build_sn_grid(np.asarray(g.arrays["time"]))
    ref_grid["dense_times"] = g.dense_times(t_last)
    p = dict(f_nickel=0.1, mej=2.0, vej=8e3, kappa=0.1, kappa_gamma=0.5, temperature_floor=5000.0,
             p0=3.0, bp=1.0, mass_ns=1.4, theta_pb=1.0, csm_mass=1.0, v_min=8e3, beta=0.45,
             shell_radius=1.0, shell_width_ratio=0.2, log10_mass=-1.0, log10_radius=13.0,
             log10_energy=50.0, nn=10.0, delta=1.1)
    idx = np.searchsorted(np.asarray(g.arrays["time"]), tau)
    want = np.asarray(sn.bolometric(model, ref_grid, p))[idx]
    got = np.asarray(sn.bolometric_at(model, g, jnp.asarray(tau), p, t_last=t_last))
    np.testing.assert_allclose(got, want, rtol=1e-11)


def test_shock_cooling_is_evaluated_at_the_observation_not_interpolated():
    """Between two epochs, the shock term is exact and only the Arnett part is interpolated."""
    from whisper_cbpf.models.jax import supernova as sn

    g = sn.build_fixed_grid(60.0)
    p = dict(f_nickel=1e-3, mej=0.5, vej=8e3, kappa=0.1, kappa_gamma=0.5, log10_mass=-1.0,
             log10_radius=13.5, log10_energy=51.0, nn=10.0, delta=1.1)
    tau = jnp.asarray([0.3, 0.77, 1.9])                        # between grid epochs
    shock = np.asarray(sn.shock_cooling(tau * sn.DAY_TO_S, 0.1, 10 ** 13.5, 1e51)[0])
    arnett = np.asarray(sn.bolometric_at("arnett", g, tau, p, t_last=1.9))
    total = np.asarray(sn.bolometric_at("shock_cooling_and_arnett", g, tau, p, t_last=1.9))
    np.testing.assert_allclose(total, shock + arnett, rtol=1e-12)


@pytest.mark.parametrize("model", GATE_FAMILIES)
def test_s2_gate_against_the_current_model_on_prior_draws(model):
    """p95 <= 1e-3 and max <= 2e-3 mag from the current model (itself <= 8e-6 from redback)."""
    data, fixed = _pair(model)
    rng = np.random.default_rng(11)
    d = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(25):
            p = data.default_prior.sample(rng)
            m_d = _mag(data.predict(p, T_OBS, B_OBS))
            m_f = _mag(fixed.predict(p, T_OBS, B_OBS))
            ok = m_d < 30.0
            d.append(np.abs(m_f - m_d)[ok])
    d = np.concatenate(d)
    assert d.size > 20
    assert np.quantile(d, 0.95) <= 1e-3 and d.max() <= 2e-3, (model, np.quantile(d, 0.95),
                                                                d.max())


def test_against_redback_directly():
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as ra

    cpu = ra.redback_model("arnett", band_names=BANDS, redshift=Z, constraint=None)
    _, fixed = _pair("arnett")
    rng = np.random.default_rng(4)
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(10):
            p = cpu.default_prior.sample(rng)
            m_rb = _mag(cpu.predict(p, T_OBS, B_OBS))
            m_f = _mag(fixed.predict({k: p[k] for k in fixed.parameters}, T_OBS, B_OBS))
            ok = np.isfinite(m_rb) & (m_rb < 30.0)
            if ok.any():
                worst = max(worst, float(np.max(np.abs(m_rb - m_f)[ok])))
    assert worst < 2e-3, worst


def test_a_longer_span_costs_epochs_not_accuracy():
    from whisper_cbpf.models.jax import supernova_model

    dl = luminosity_distance_cm(Z)
    short = supernova_model("arnett", BANDS, Z, dl, constraint=None, diffusion_grid="fixed",
                            max_phase_days=60.0)
    long_ = supernova_model("arnett", BANDS, Z, dl, constraint=None, diffusion_grid="fixed",
                            max_phase_days=1000.0)
    assert long_.predict_jax.ctx.fixed_grid().n_epochs > short.predict_jax.ctx.fixed_grid().n_epochs
    p = short.default_prior.sample(np.random.default_rng(0))
    np.testing.assert_allclose(_mag(short.predict(p, T_OBS, B_OBS)),
                               _mag(long_.predict(p, T_OBS, B_OBS)), atol=1e-11)


def test_times_may_be_traced():
    """The data grid refused traced times; the fixed epochs take them, under vmap over times."""
    _, fixed = _pair("arnett")
    data, _ = _pair("arnett")
    pj = fixed.predict_jax
    p = fixed.default_prior.sample(np.random.default_rng(2))
    theta = jnp.asarray([p[k] for k in fixed.parameters])
    bidx = pj.band_index(B_OBS)
    shifts = jnp.asarray([0.0, 0.5, 1.0])
    out = jax.vmap(lambda s: pj(theta, jnp.asarray(T_OBS) + s, bidx))(shifts)
    assert out.shape == (3, T_OBS.size) and np.isfinite(np.asarray(out)).all()
    np.testing.assert_allclose(np.asarray(out[0]), fixed.predict(p, T_OBS, B_OBS), rtol=1e-12)
    with pytest.raises(TypeError, match="CONCRETE"):
        jax.vmap(lambda s: data.predict_jax(theta, jnp.asarray(T_OBS) + s, bidx))(shifts)


def test_no_light_before_the_explosion():
    from whisper_cbpf.models.jax import supernova_model

    m = supernova_model("csm_shock_and_arnett", BANDS, Z, luminosity_distance_cm(Z),
                        t_exp_days=10.0, constraint=None, diffusion_grid="fixed")
    p = m.default_prior.sample(np.random.default_rng(3))
    mag = _mag(m.predict(p, np.array([2.0, 9.99, 10.0, 25.0]), np.array(["ztfg"] * 4)))
    assert np.allclose(mag[:3], 40.0)
