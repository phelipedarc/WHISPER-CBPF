"""A zero-flux first detection is flagged before the fit starts (owner decision, texp3).

A model that switches on as a step at its explosion time (``tde_stream_stream`` jumps from 0 to a
third of its peak in a thousandth of a day) cannot fit a first "detection" of zero flux anywhere
after ``t_exp``: the only good region is the zero-width top edge of the window, and every sampler
settles weeks early while reporting convergence. The fix is in the data (mark the point as an upper
limit), so the package says so, in one plain warning, and never changes the data itself.

The claims, each a test:

1. A fit whose earliest detection has S/N < 3 warns once, naming the time, band and S/N.
2. Good data, or the same point flagged as an upper limit, give no such warning.
3. A model with no event (no ``t_exp``, no explosion date) is not warned: it rises from zero.
4. The standard warnings filter on :class:`ZeroFluxWarning` silences it.
5. :func:`whisper_cbpf.compare` warns once for the comparison, not once per model.
6. S/N comes from the magnitude error for magnitude-only data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.samplers import ZERO_FLUX_SNR, ZeroFluxWarning, zero_flux_first_detections

MJD0 = 60000.0
TRUTH = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
T = np.array([0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 16.0, 21.0, 26.0, 30.0])
MCMC = dict(sampler="mcmc", nwalkers=8, nsteps=40, burnin=10, check_convergence=False)


def _lc(first_flux=None, first_ul=False, explosion=True):
    rng = np.random.default_rng(3)
    flux = wp.get_model("flare").predict(TRUTH, T) + rng.normal(0.0, 0.1, T.size)
    if first_flux is not None:
        flux[0] = first_flux
    ul = np.zeros(T.size, bool)
    ul[0] = first_ul
    lc = wp.LightCurve(time=T + MJD0, band=["g"] + ["r"] * (T.size - 1), flux=flux,
                       flux_err=np.full(T.size, 0.1), upper_limit=ul)
    return lc.set_explosion_date(MJD0 - 1.0) if explosion else lc


def _zero_flux(caught):
    return [w for w in caught if issubclass(w.category, ZeroFluxWarning)]


def _fit(lc, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wp.fit(lc, "flare", **{**MCMC, **kw})
    return _zero_flux(caught)


def test_a_zero_flux_first_detection_warns_once_with_time_band_and_snr():
    got = _fit(_lc(first_flux=0.04))
    assert len(got) == 1
    msg = str(got[0].message)
    assert "MJD 60000.50 (band g)" in msg and "S/N 0.4" in msg
    assert "consistent with zero flux" in msg and "upper limit" in msg
    assert "tde_stream_stream" in msg


def test_good_data_do_not_warn():
    assert _fit(_lc()) == []


def test_the_point_flagged_as_an_upper_limit_does_not_warn():
    assert _fit(_lc(first_flux=0.04, first_ul=True)) == []


def test_a_model_without_an_event_is_not_warned():
    assert _fit(_lc(first_flux=0.04, explosion=False)) == []


def test_the_warnings_filter_silences_it():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ZeroFluxWarning)
        wp.fit(_lc(first_flux=0.04), "flare", **MCMC)
    assert _zero_flux(caught) == []


def test_the_data_are_not_changed():
    lc = _lc(first_flux=0.04)
    before = np.asarray(lc.flux).copy(), np.asarray(lc.upper_limit).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit(lc, "flare", **MCMC)
    assert np.array_equal(np.asarray(lc.flux), before[0])
    assert np.array_equal(np.asarray(lc.upper_limit), before[1])
    assert res.n_data == T.size


def test_compare_warns_once_for_the_comparison():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wp.compare(_lc(first_flux=0.04), ["flare", "bazin"],
                   sampler="mcmc", nwalkers=8, nsteps=40, burnin=10, evidence_check=False,
                   likelihood_max_opt=False)
    assert len(_zero_flux(caught)) == 1


def test_leading_low_detections_are_all_listed_and_the_walk_stops_at_a_good_one():
    lc = wp.LightCurve(time=[1.0, 2.0, 3.0, 4.0], band=["g", "r", "g", "r"],
                       flux=[0.1, -0.2, 5.0, 0.1], flux_err=[0.1, 0.1, 0.1, 0.1])
    got = zero_flux_first_detections(lc)
    assert [(d["time"], d["band"]) for d in got] == [(1.0, "g"), (2.0, "r")]
    assert got[0]["snr"] == pytest.approx(1.0) and got[1]["snr"] == pytest.approx(2.0)
    assert got[0]["mjd"] is None


def test_the_threshold_is_three_and_can_be_changed():
    lc = wp.LightCurve(time=[1.0, 2.0], band=["g", "g"], flux=[0.35, 5.0], flux_err=[0.1, 0.1])
    assert ZERO_FLUX_SNR == 3.0
    assert zero_flux_first_detections(lc) == []                    # S/N 3.5
    assert len(zero_flux_first_detections(lc, threshold=4.0)) == 1


def test_magnitude_only_data_use_the_magnitude_error():
    lc = wp.LightCurve(time=[1.0, 2.0], band=["g", "g"], magnitude=[21.0, 19.0],
                       magnitude_err=[0.8, 0.05])
    got = zero_flux_first_detections(lc)
    assert len(got) == 1
    assert got[0]["snr"] == pytest.approx(2.5 / np.log(10) / 0.8)


def test_mjd_is_reported_on_a_shifted_clock():
    lc = _lc(first_flux=0.04)
    got = zero_flux_first_detections(lc)
    assert got[0]["mjd"] == pytest.approx(MJD0 + 0.5)
    assert got[0]["time"] == pytest.approx(1.5)                    # day 1.5 after the explosion


def test_empty_and_no_detection_light_curves_return_nothing():
    lc = wp.LightCurve(time=[1.0], band=["g"], flux=[0.0], flux_err=[0.1], upper_limit=[True])
    assert zero_flux_first_detections(lc) == []
    assert zero_flux_first_detections(lc[np.zeros(1, bool)]) == []


# --- once per fit on the JAX paths (texp3 verify) -----------------------------------------------
# emcee_jax ran the pre-event rule three times per fit (the fit wrapper, fit_emcee_jax and
# log_density), likelihood_max_opt a fourth, and fit_batch once per light curve through
# log_density: under simplefilter("always") (pytest, notebooks) the warning repeated.

EMCEE_JAX = dict(nwalkers=8, nsteps=60, burnin=20, n_ensembles=1)


@pytest.fixture
def _x64():
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _count(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return _zero_flux(caught)


def test_emcee_jax_warns_once_per_fit(_x64):
    from whisper_cbpf.samplers.jax.emcee_jax import fit_emcee_jax

    lc = _lc(first_flux=0.04)
    assert len(_count(lambda: wp.fit(lc, "flare_jax", sampler="emcee_jax",
                                     check_convergence=False, **EMCEE_JAX))) == 1
    assert len(_count(lambda: wp.fit(lc, "flare_jax", sampler="emcee_jax",
                                     likelihood_max_opt=True, check_convergence=False,
                                     **EMCEE_JAX))) == 1
    assert len(_count(lambda: fit_emcee_jax(lc, "flare_jax", **EMCEE_JAX))) == 1
    assert len(_count(lambda: wp.log_density(lc, "flare_jax"))) == 1     # a call of its own
    assert _count(lambda: fit_emcee_jax(_lc(), "flare_jax", **EMCEE_JAX)) == []


def test_fit_batch_warns_once_for_the_batch_and_names_the_light_curves(_x64):
    from whisper_cbpf.samplers.jax.batch import fit_batch

    lcs = [_lc(first_flux=0.04), _lc(), _lc(first_flux=-0.02)]
    got = _count(lambda: fit_batch(lcs, "flare_jax", nwalkers=8, nsteps=60, burnin=20))
    assert len(got) == 1
    msg = str(got[0].message)
    assert "2 of 3 light curves (indices [0, 2])" in msg and "S/N below 3" in msg
    assert _count(lambda: fit_batch([_lc(), _lc()], "flare_jax", nwalkers=8, nsteps=60,
                                    burnin=20)) == []


def test_the_zero_flux_flag_is_restored_after_the_block():
    from whisper_cbpf.samplers.base import (ZERO_FLUX_WARNED, _zero_flux_batch_message,
                                            _zero_flux_given)

    assert ZERO_FLUX_WARNED.get() is False
    with pytest.raises(RuntimeError):
        with _zero_flux_given():
            assert ZERO_FLUX_WARNED.get() is True
            raise RuntimeError("inside")
    assert ZERO_FLUX_WARNED.get() is False
    assert _zero_flux_batch_message([], 4) is None
    many = _zero_flux_batch_message(list(range(25)), 30)
    assert "25 of 30" in many and ", ...)" in many
