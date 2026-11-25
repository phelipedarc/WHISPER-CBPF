"""The free-``t_exp`` conventions of the five JAX TDE models, and a detection at the explosion.

The claims, each a test:

1. **``t_exp`` is a pure time shift** on the light curve's clock: predicting at ``(t, t_exp=T)``
   equals predicting at ``(t - T, t_exp=0)`` to round-off, for every model. (Source-frame time is
   ``(t - t_exp) / (1 + z)`` for fall-back, MOSFiT, stream-stream and analytical, as each module's
   "Time" paragraph says; the shift test covers ``t_exp`` itself.)
2. **Before ``t_exp``** the fall-back, MOSFiT, stream-stream and analytical models are dark
   (``mag_floor`` = 40 AB, 3.631e-13 Jy); the cooling-envelope Gaussian rise is not masked before
   ``t_exp`` (a Gaussian centred ``peak_time`` after it, defined for any epoch).
3. **A fit cuts the model's Uniform ``t_exp`` prior to the default window**
   (:func:`whisper_cbpf.default_t_exp_prior`): its upper edge is the first detection, its lower
   edge 30 days before the first detection when no non-detection precedes it.
4. **A detection at the true explosion date** (``t_exp`` = the first detection): the truth is the
   default prior's upper edge and inside its support, the first epoch is fitted (not left out as
   pre-event), and the log density is finite there, higher than one day earlier, and ``-inf`` just
   after (the prior ends at the first detection).

CPU, no sampling: about a minute in the CPU container (one compile per model for claim 4).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.models.jax import tde_analytical as TA  # noqa: E402
from whisper_cbpf.models.jax import tde_fallback as TF  # noqa: E402
from whisper_cbpf.models.jax import tde_mosfit as TM  # noqa: E402
from whisper_cbpf.models.jax import tde_stream_stream as TS  # noqa: E402
from whisper_cbpf.models.jax._factories import tde_model  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.base import _pre_event_plan  # noqa: E402

BANDS = ["lsstg", "lsstr"]
Z, DL = 0.05, 7.093375016731583e26
TRUTH = {
    "tde_fallback": dict(mbh6=1.5, mstar=1.0, tvisc=2.0, bb=1.0, eta=0.1, rph0=6.31, lphoto=1.5),
    "tde_mosfit": dict(bhmass=1.5e6, starmass=1.0, b=1.0, efficiency=0.1, Tviscous=2.0, Rph0=6.31,
                       lphoto=1.5),
    "tde_cooling_envelope": dict(peak_time=20.0, sigma_t=20.0, mbh_6=1.5, stellar_mass=1.0,
                                 eta=0.05, alpha=0.2, beta=1.5),
    "tde_stream_stream": dict(mbh_6=1.5, mstar=1.0, sigma_t=20.0, peak_time=30.0),
    "tde_analytical": dict(l0=1.8e54, t_0_turn=30.0, mej=1.0, vej=1e4, kappa=0.2,
                           kappa_gamma=100.0, temperature_floor=5000.0),
}
DARK_BEFORE_T_EXP = {"tde_fallback", "tde_mosfit", "tde_stream_stream", "tde_analytical"}
FLOOR_JY = 3631.0 * 10 ** (-0.4 * 40.0)


def _build(name, window=(-60.0, 0.0)):
    kw = dict(free=["t_exp"], prior=Prior({"t_exp": Uniform(*window)}))
    if name == "tde_fallback":
        return TF.tde_fallback_model(BANDS, Z, DL, **kw)
    if name == "tde_mosfit":
        return TM.tde_mosfit_model(BANDS, Z, DL, **kw)
    if name == "tde_cooling_envelope":
        return tde_model(BANDS, Z, DL, name="tde_cooling_envelope", **kw)
    if name == "tde_stream_stream":
        return TS.tde_stream_stream_model(BANDS, Z, DL, **kw)
    return TA.tde_analytical_model(BANDS, Z, DL, constraint=None, **kw)


def _peak(m, name):
    t = np.linspace(1.0, 200.0, 400)
    return float(np.max(np.asarray(m.predict(dict(TRUTH[name], t_exp=0.0), t,
                                             np.array(["lsstr"] * t.size)), float)))


@pytest.mark.parametrize("name", list(TRUTH))
def test_t_exp_is_a_pure_time_shift(name):
    m = _build(name)
    tau = np.geomspace(0.5, 300.0, 25)
    b = np.array(["lsstr"] * tau.size)
    T = 60000.0
    f0 = np.asarray(m.predict(dict(TRUTH[name], t_exp=0.0), tau, b), float)
    f1 = np.asarray(m.predict(dict(TRUTH[name], t_exp=T), T + tau, b), float)
    np.testing.assert_allclose(f1, f0, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("name", list(TRUTH))
def test_before_t_exp(name):
    m = _build(name)
    t = np.array([-30.0, -5.0, -0.5])
    f = np.asarray(m.predict(dict(TRUTH[name], t_exp=0.0), t, np.array(["lsstr"] * t.size)),
                   float)
    if name in DARK_BEFORE_T_EXP:
        np.testing.assert_allclose(f, FLOOR_JY, rtol=1e-6)
    else:                       # the cooling envelope's Gaussian rise is defined before its origin
        assert np.all(f > 1e-3 * _peak(m, name))


def test_fit_cuts_the_model_t_exp_prior_to_the_default_window():
    m = _build("tde_stream_stream", window=(59940.0, 60010.0))
    t = np.repeat(60000.0 + np.geomspace(1.0, 200.0, 12) - 1.0, 2)
    b = np.tile(np.array(BANDS), 12)
    f = np.asarray(m.predict(dict(TRUTH["tde_stream_stream"], t_exp=59990.0), t, b), float)
    lc = wp.LightCurve(time=t, band=b, flux=f, flux_err=0.05 * f + 1e-6, redshift=Z)
    plan = _pre_event_plan(lc, m, None)
    rec = plan.t_exp_record
    assert rec["high"] == pytest.approx(60000.0)            # the first detection
    assert rec["low"] == pytest.approx(60000.0 - 30.0)      # no limit: first detection - 30 d
    assert rec["source"] == "model prior, cut to the default window"


def _at_explosion(name, m, t_exp=60000.0):
    """35 epochs x 2 bands, the first epoch AT the true explosion; noise-free, 5 % + 1 % of peak
    errors (the matrix's error model), every row a detection."""
    tau = np.concatenate([[0.0], np.geomspace(1.0, 250.0, 34)])
    t = np.repeat(t_exp + tau, 2)
    b = np.tile(np.array(BANDS), tau.size)
    f = np.asarray(m.predict(dict(TRUTH[name], t_exp=t_exp), t, b), float)
    err = 0.05 * np.abs(f) + 0.01 * _peak(m, name)
    return wp.LightCurve(time=t, band=b, flux=f, flux_err=err, redshift=Z, name=f"{name}_o0")


@pytest.mark.parametrize("name", list(TRUTH))
def test_a_detection_at_the_true_explosion_date(name):
    truth = 60000.0
    m = _build(name, window=(59900.0, 60100.0))
    lc = _at_explosion(name, m, truth)
    plan = _pre_event_plan(lc, m, None)
    rec = plan.t_exp_record
    assert (rec["low"], rec["high"]) == (truth - 30.0, truth)       # truth = the upper edge
    assert rec["anchor"] == truth and rec["last_non_detection"] is None
    assert plan.rule["n_excluded"] == 0 and len(plan.lc) == len(lc)  # the first epoch is fitted
    t_dist = plan.prior.distributions["t_exp"]
    assert np.isfinite(t_dist.log_prob(truth))                      # the support is closed

    ld = wp.log_density(lc, m)
    assert ld.n_data == len(lc)

    def at(t_exp):
        theta = np.array([dict(TRUTH[name], t_exp=t_exp)[p] for p in ld.names], float)
        return float(ld.fn(theta))

    assert np.isfinite(at(truth))
    assert at(truth) > at(truth - 1.0)
    assert at(truth + 0.01) == -np.inf                              # after the first detection
