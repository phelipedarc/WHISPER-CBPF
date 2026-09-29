"""Loud failures instead of silent wrong numbers: one test per 0.1.1 case.

Each case below used to return a plausible, wrong number without a word. Each now raises, warns,
or is recorded where a reader of the result finds it. The fixes and their measurements are in
``CHANGELOG.md`` (0.1.1); the detailed regression tests live beside each module. This file is the
one place that lists them all, one short test each, so a change that makes any of them silent again
fails here by name.

=====================================  =================================  =========================
case                                   before (0.1.0)                     now
=====================================  =================================  =========================
band selection matching nothing        an empty light curve               ValueError naming bands
bolometric engine as a band model      erg/s fitted as Jy (1.2e37 "Jy")   ValueError naming wrapper
epochs outside a model's time domain   0 Jy at every draw                 ValueError naming span
unknown engine keyword (register_tde)  TypeError at the first predict     ValueError at registration
magnetar spin-down the grid misses     up to 21 mag too bright            RuntimeWarning, once
TDE engine grid                        5000 steps, redback uses 500       redback's n_time, stated
points simulated at mag_floor          a wall of identical points         counted, warned > 10 %
redback's warnings.simplefilter        every whisper warning hidden       whisper warnings survive
band with no resolvable filter         evaluated at 6000 A                ValueError naming band
grouped label in band-integrating mode modelled as SDSS g                 ValueError naming label
pre-event rows                         fitted (moved the model's grid)    left out, counted, warned
=====================================  =================================  =========================
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import whisper_cbpf as wp


@pytest.fixture
def x64():
    jax = pytest.importorskip("jax")
    before = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


def test_a_band_selection_that_matches_nothing_raises(tmp_path):
    """``bands=`` that matched no row returned an empty light curve."""
    p = tmp_path / "zg.csv"
    p.write_text("time,band,magnitude,magnitude_err\n1.0,zg,19.0,0.1\n2.0,zr,19.1,0.1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert wp.load_lightcurve(p, bands=["zg"]).bands == ["ztfg"]    # spelled as the column
        with pytest.raises(ValueError, match=r"match none of this light curve's bands"):
            wp.load_lightcurve(p, bands=["lsstg"])


def test_a_bolometric_engine_is_refused_as_a_band_model():
    """1.5: a ``*_bolometric`` engine swallowed ``output_format`` and its erg/s were fitted as Jy."""
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as RA

    with pytest.raises(ValueError, match=r"not a flux density.*'shock_cooling_and_arnett'"):
        RA.redback_model("shock_cooling_and_arnett_bolometric", ["ztfg", "ztfr"], redshift=0.05)


def test_epochs_outside_a_models_time_domain_raise():
    """1.4: redback's two-component kilonova returned 0 Jy past its 6-day grid, at every draw."""
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as RA

    t = np.linspace(0.5, 12.5, 25)
    with pytest.raises(ValueError, match=r"every one of .* prior draws"):
        RA.redback_model("two_component_kilonova_model", ["ztfg", "ztfr", "ztfi"],
                         redshift=0.0098, times=t)


def test_an_unknown_engine_keyword_is_refused_at_registration(x64):
    """1.8: ``register_tde`` took any keyword and failed at the first predict, naming neither."""
    from whisper_cbpf.models.jax import _factories as F

    with pytest.raises(ValueError, match=r"'hoverr'.*hoverR"):
        F.tde_model(["lsstg", "lsstr"], 0.05, 6.9e26, hoverr=0.3)


def test_a_magnetar_spin_down_the_grid_cannot_resolve_warns_once(x64):
    """1.1: redback's own grid delivers 45 % of E_rot at t_p = 1.43 s; the factory says so."""
    t = np.geomspace(0.5, 200.0, 30)
    m = wp.supernova_model("basic_magnetar_powered", ["lsstg", "lsstr"], 0.05, 7.09e26, times=t,
                           constraint=None)
    b = np.array(["lsstg", "lsstr"] * 15)
    demo = dict(p0=2.0, bp=1.0, mass_ns=1.4, theta_pb=1.0, mej=2.0, vej=1e4, kappa=0.1,
                kappa_gamma=0.03, temperature_floor=4000.0)
    p = {k: demo[k] for k in m.parameters}
    fast = dict(p, bp=np.sqrt(1.3e5 * 4.0 / 0.105) / np.sin(p["theta_pb"]))    # t_p = 0.105 s
    with pytest.warns(RuntimeWarning, match="spin-down unresolved"):
        m.predict(fast, t, b)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.predict(fast, t, b)                                     # once per model, not per call


def test_the_tde_engine_grid_follows_the_installed_redback_and_says_so(x64):
    """1.2: the JAX TDE integrated 5000 steps where redback 1.20 integrates 500 (22.9 mag)."""
    from whisper_cbpf.models import redback_adapter as RA
    from whisper_cbpf.models.jax import tde

    preset = RA.installed_redback_preset() or RA.LATEST_REDBACK_PRESET
    n_time = tde.REDBACK_ENGINE_PRESETS[preset]["n_time"]
    assert tde.default_n_time() == n_time == (5000 if preset == "1.12" else 500)
    m = wp.tde_model(["lsstg", "lsstr"], 0.05, 6.9e26)
    assert f"n_time={n_time}" in m.description


def test_points_simulated_at_mag_floor_are_counted_and_warned():
    """1.7: the JAX models cap magnitudes at mag_floor; a sampler trained on a wall of floored
    points said nothing. The batched forward map counts them and warns past 10 %."""
    jnp = pytest.importorskip("jax.numpy")
    from whisper_cbpf.io.photometry import AB_ZEROPOINT_JY
    from whisper_cbpf.samplers.jax._adapters import (MAG_FLOOR_WARN_FRACTION,
                                                     make_batched_predict_jax)

    floor = AB_ZEROPOINT_JY * 10 ** (-0.4 * 40.0)

    def predict_jax(theta, times, band_idx=None):
        return jnp.maximum(theta[0] * jnp.exp(-jnp.asarray(times) / theta[1]), floor)

    predict_jax.mag_floor = 40.0
    m = wp.register_model("_no_silent_floor", lambda p, t, b=None: np.zeros(len(t)),
                          ["amp", "tau"], prior=wp.Prior({"amp": wp.Uniform(1e-3, 1e-2),
                                                          "tau": wp.LogUniform(0.05, 20.0)}),
                          predict_jax=predict_jax, overwrite=True)
    t = np.linspace(0.5, 30.0, 20)
    lc = wp.LightCurve(time=t, band=["ztfg"] * 20, flux=np.ones(20), flux_err=np.ones(20))
    batched = make_batched_predict_jax(lc, m, chunk=None)
    with pytest.warns(UserWarning, match="mag_floor"):
        batched(jnp.array([[5e-3, 0.05], [5e-3, 0.06], [5e-3, 20.0]]))
    stats = batched.floor_stats
    assert stats["n_points"] == 60 and stats["fraction"] > MAG_FLOOR_WARN_FRACTION
    assert stats["mag_floor"] == 40.0


def test_redback_cannot_silence_whisper_warnings():
    """redback 1.20 installs a process-wide ``simplefilter("ignore")`` on import, which hid
    every whisper warning once a redback model was bound (0 of 46 run logs showed SNPE's
    fallback notice, though 39 runs fell back)."""
    pytest.importorskip("redback")
    code = ("import whisper_cbpf as wp\n"
            "wp.register_redback('arnett', redshift=0.05)\n"
            "wp.resolve_band('zzz_not_a_band', svo_fallback=False)\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=600, env=env)
    assert run.returncode == 0, run.stderr[-1500:]
    assert "Band 'zzz_not_a_band' is not in FILTER_LOOKUP" in run.stderr, run.stderr[-1500:]


def test_a_band_with_no_resolvable_filter_raises_instead_of_6000_angstrom():
    """mck19 evaluated any band it could not resolve at 6000 A, without a word."""
    m = wp.get_model("mck19")
    truth = {"v_kick": 300.0, "M_smbh": 1e8, "M_bh": 80.0, "r_bh": 700.0, "redshift": 0.28}
    with pytest.raises(ValueError, match="not_a_band"):
        m.predict(truth, np.full(2, 6.0), np.array(["not_a_band", "not_a_band"]))


def test_a_grouped_label_raises_in_a_band_integrating_model():
    """``g-band`` (a grouped label) reached redback as SDSS g, even for ZTF g data."""
    from whisper_cbpf.synphot import resolve_filter

    with pytest.raises(ValueError, match="grouped effective-band label"):
        resolve_filter("g-band")
    m = wp.get_model("mck19")
    truth = {"v_kick": 300.0, "M_smbh": 1e8, "M_bh": 80.0, "r_bh": 700.0, "redshift": 0.28}
    with pytest.raises(ValueError, match="r-band"):
        m.predict(truth, np.full(2, 6.0), np.array(["r-band", "r-band"]))


def test_pre_event_rows_are_left_out_counted_and_reported():
    """Pre-event data rule (2026-09-26): rows at or before day 0 are not fitted. They used to enter
    the likelihood and move model time grids (the kilonova grid-edge effect)."""
    base = pytest.importorskip("whisper_cbpf.samplers.base")
    if not hasattr(base, "prepare_lc"):
        pytest.skip("the pre-event rule is not in this installation")
    t = np.array([-3.0, -1.0, 0.0, 0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 16.0, 21.0, 26.0])
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
                                          "decay_time": 15.0}, np.clip(t, 0.0, None))
    lc = wp.LightCurve(time=t + 60000.0, band=["r"] * t.size, flux=flux,
                       flux_err=np.full(t.size, 0.1)).set_explosion_date(60000.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = wp.fit(lc, "flare", sampler="abc", n_simulations=500, quantile=0.1, seed=0)
    assert res.info["excluded_pre_event"] == 3 and res.n_data == t.size - 3
    assert [w for w in caught if "pre-event data are not fitted" in str(w.message)]
