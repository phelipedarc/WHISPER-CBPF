"""The JAX models' time origin and their inputs.

1.3 **float32 on a raw-MJD clock.** The kilonova factories take ``t_exp_days=`` on the light
    curve's own clock and subtracted it inside the trace, in the model's dtype: near MJD 58000
    float32 resolves 2^-8 = 0.0039 d, which cost 7.2 mmag and a log-likelihood error of 8.7 on
    AT2017GFO, and the samplers' adapters had already cast the epochs to float32
    before the model saw them. Now the subtraction happens on the host in float64, and the adapters
    hand the model the float64 epochs.
1.7 **The ``mag_floor`` clamp is recorded.** A simulation-based sampler trains on prior draws, and
    half of a magnetar's simulated points can sit at exactly ``mag_floor`` (redback has no floor).
    The batched forward map counts them, keeps the count, and warns above
    ``MAG_FLOOR_WARN_FRACTION``; predictions do not change.
1.8 **``register_tde`` checks its engine kwargs, and takes ``t_exp_days``.** Any keyword used to
    register and then fail at the first predict (``t_exp_days=`` among them).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

T_EXP_MJD = 57982.52852                  # GW170817's merger, AT2017GFO's t0
DAYS = np.geomspace(0.45, 10.5, 30)
BANDS = ["ztfg", "ztfr", "ztfi"]
Z, DL = 0.0098, 1.3499e26

_FLOAT32_PROBE = r"""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
assert not jax.config.jax_enable_x64
import whisper_cbpf as wp
from whisper_cbpf.models.jax import _factories as F
from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

t_exp, days = float(sys.argv[1]), np.asarray(json.loads(sys.argv[2]))
bands = np.array(["ztfg", "ztfr", "ztfi"] * 10)
out = {}
for kind, build in (("one", F.kilonova_model), ("two", F.kilonova_two_model)):
    mjd = build(["ztfg", "ztfr", "ztfi"], 0.0098, 1.3499e26, t_exp_days=t_exp)
    rel = build(["ztfg", "ztfr", "ztfi"], 0.0098, 1.3499e26)
    rng = np.random.default_rng(1)
    worst_mag = worst_ll = 0.0
    for _ in range(20):
        p = mjd.default_prior.sample(rng)
        f_mjd = mjd.predict(p, t_exp + days, bands)
        f_rel = rel.predict(p, days, bands)
        worst_mag = max(worst_mag, float(np.max(np.abs(2.5 * np.log10(f_mjd / f_rel)))))
        # through the samplers' auto-built density: same light curve on the two clocks
        lc_mjd = wp.LightCurve(time=t_exp + days, band=bands, flux=f_rel * 1.01,
                               flux_err=0.02 * f_rel)
        lc_rel = wp.LightCurve(time=days, band=bands, flux=f_rel * 1.01, flux_err=0.02 * f_rel)
        th = np.array([p[k] for k in mjd.parameters])
        ll_mjd = float(make_log_prob_jax(lc_mjd, mjd)(th))
        ll_rel = float(make_log_prob_jax(lc_rel, rel)(th))
        worst_ll = max(worst_ll, abs(ll_mjd - ll_rel))
    out[kind] = dict(max_dmag=worst_mag, max_dlogl=worst_ll)
print(json.dumps(out))
"""


def test_float32_raw_mjd_clock_matches_days_since_t0():
    """The bug: 7.2 mmag and dlogL 8.7 between the two clocks in float32."""
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    run = subprocess.run([sys.executable, "-c", _FLOAT32_PROBE, repr(T_EXP_MJD),
                          json.dumps(DAYS.tolist())], capture_output=True, text=True, env=env,
                         timeout=900)
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads(run.stdout.strip().splitlines()[-1])
    for kind, r in got.items():
        # float32 round-off on days since t0 is ~1e-5 mag; the MJD clock was 7e-3.
        assert r["max_dmag"] < 1e-4, (kind, r)
        assert r["max_dlogl"] < 0.05, (kind, r)


@pytest.fixture()
def _x64():
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


def test_float64_clocks_agree_exactly_enough(_x64):
    from whisper_cbpf.models.jax import _factories as F

    bands = np.array(BANDS * 10)
    mjd = F.kilonova_model(BANDS, Z, DL, t_exp_days=T_EXP_MJD)
    rel = F.kilonova_model(BANDS, Z, DL)
    p = mjd.default_prior.sample(np.random.default_rng(0))
    a, b = mjd.predict(p, T_EXP_MJD + DAYS, bands), rel.predict(p, DAYS, bands)
    assert np.max(np.abs(a / b - 1.0)) < 1e-10


# --- 1.8: register_tde ----------------------------------------------------------------------------

def test_tde_rejects_an_unknown_engine_kwarg_at_registration(_x64):
    """The bug: this registered, then died at the first predict with a TypeError from inside the
    jitted engine that named neither the factory nor the keyword."""
    from whisper_cbpf.models.jax import _factories as F

    with pytest.raises(ValueError, match=r"'hoverr'.*hoverR"):
        F.tde_model(BANDS, 0.05, 6.9e26, hoverr=0.3)
    F.tde_model(BANDS, 0.05, 6.9e26, hoverR=0.3, binding_energy_const=0.8)   # real ones pass


def test_tde_takes_t_exp_days_on_the_light_curves_clock(_x64):
    """``t_exp_days`` is subtracted on the host in float64: a raw-MJD light curve then gives what
    days since t0 give (verified to 2e-16 the same way)."""
    from whisper_cbpf.models.jax import _factories as F

    t0 = 59019.25
    days = np.geomspace(1.0, 120.0, 12)
    bands = np.array(BANDS * 4)
    mjd = F.tde_model(BANDS, 0.062, 8.6e26, t_exp_days=t0)
    rel = F.tde_model(BANDS, 0.062, 8.6e26)
    p = dict(peak_time=30.0, sigma_t=15.0, mbh_6=1.0, stellar_mass=1.0, eta=0.05, alpha=0.1,
             beta=1.0)
    a, b = mjd.predict(p, t0 + days, bands), rel.predict(p, days, bands)
    assert np.all(b > 0)
    assert np.max(np.abs(a / b - 1.0)) < 1e-12
    th = np.array([p[k] for k in mjd.parameters])
    bi = mjd.predict_jax.band_index(bands)
    assert np.allclose(np.asarray(mjd.predict_jax(th, t0 + days, bi)), b, rtol=1e-12, atol=0)


# --- 1.7: the mag_floor clamp is recorded ---------------------------------------------------------

def test_batched_map_records_the_floored_fraction_and_warns(_x64):
    """A kilonova 40 Mpc away is 17-24 mag over these epochs; ``mag_floor=21`` clamps most points.
    Predictions are unchanged: the floor is the model's, the record is new."""
    import whisper_cbpf as wp
    from whisper_cbpf.models.jax import _factories as F
    from whisper_cbpf.samplers.jax import _adapters as A

    bands = np.array(BANDS * 10)
    lc = wp.LightCurve(time=DAYS, band=bands, flux=np.full(30, 1e-5), flux_err=np.full(30, 1e-6))
    m = F.kilonova_model(BANDS, Z, DL, mag_floor=21.0)
    assert m.predict_jax.mag_floor == 21.0 and "mag_floor=21" in m.description
    f = A.make_batched_predict_jax(lc, m)
    rng = np.random.default_rng(0)
    theta = np.array([[m.default_prior.sample(rng)[k] for k in m.parameters] for _ in range(8)])
    with pytest.warns(UserWarning, match=r"at mag_floor=21"):
        flux = np.asarray(f(theta))
    floor = 3631.0 * 10 ** (-0.4 * 21.0)
    at_floor = np.isclose(flux, floor, rtol=1e-4, atol=0.0)
    assert f.floor_stats["n_points"] == flux.size
    assert f.floor_stats["n_floored"] == int(at_floor.sum())
    assert f.floor_stats["fraction"] > A.MAG_FLOOR_WARN_FRACTION
    # same predictions as the model itself: recording changes nothing
    bi = m.predict_jax.band_index(bands)
    one = np.asarray(m.predict_jax(theta[0], DAYS, bi))
    assert np.allclose(flux[0], one, rtol=1e-12, atol=0)


def test_no_warning_when_the_floor_is_not_reached(_x64):
    import warnings

    import whisper_cbpf as wp
    from whisper_cbpf.models.jax import _factories as F
    from whisper_cbpf.samplers.jax import _adapters as A

    bands = np.array(BANDS * 10)
    lc = wp.LightCurve(time=DAYS, band=bands, flux=np.full(30, 1e-5), flux_err=np.full(30, 1e-6))
    m = F.kilonova_model(BANDS, Z, DL)                      # mag_floor=40: never reached here
    f = A.make_batched_predict_jax(lc, m)
    theta = np.array([[0.03, 0.2, 5.0, 3000.0]])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f(theta)
    assert not [w for w in rec if "mag_floor" in str(w.message)]
    assert f.floor_stats["n_floored"] == 0
