"""The default explosion-time prior of a fit with a free ``t_exp``.

The rule (:func:`whisper_cbpf.default_t_exp_prior`): upper bound the first detection; lower bound
the anchor minus a window (default :data:`whisper_cbpf.io.schema.T_EXP_WINDOW_DAYS`, 30 d,
observer frame), the anchor being the last upper limit before the first detection when there is
one, otherwise the first detection. The claims, each a test:

1. **With and without upper limits**: the anchor is the last limit before the first detection, in
   any band; limits after the first detection do not count; with none it is the first detection.
2. **The window is a parameter**: ``window=`` of the function, and the module constant for every
   fit (read at call time); a non-positive window is refused; no detection is "not enough data".
3. **In a fit**: a model with no ``t_exp`` prior takes the window; a model's Uniform prior is cut to
   it (never widened); a non-Uniform model prior is kept; a prior passed to the fit wins as given.
4. **Models without ``t_exp``** are not touched: no record, the prior unchanged.
5. **``compare``** binds a family's explosion time to this window, and a ``t_exp`` prior given to
   it is used as given.

No sampling: a few seconds on a CPU.
"""
import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.io import schema
from whisper_cbpf.priors import Normal, Prior, Uniform
from whisper_cbpf.samplers.base import _pre_event_plan

PHYS = {"amplitude": Uniform(0.0, 10.0), "rise_time": Uniform(1.0, 10.0),
        "decay_time": Uniform(5.0, 30.0)}


def _shifted_flare(params, times, bands=None):
    return wp.get_model("flare").predict(params, np.asarray(times, float) - params["t_exp"], bands)


def _free_model(t_exp_prior=None, name="flare_default_texp"):
    dists = dict(PHYS)
    if t_exp_prior is not None:
        dists["t_exp"] = t_exp_prior
    return wp.register_model(name, _shifted_flare,
                             ["amplitude", "rise_time", "decay_time", "t_exp"],
                             prior=Prior(dists), overwrite=True)


def _lc(limits=(59990.0, 59995.0, 60001.0), later_limits=(), bands=None):
    """Raw MJD clock: limits, detections from 60003.4 on, then optional later limits."""
    det = 60003.4 + np.array([0.0, 0.6, 2.1, 3.6, 6.6, 10.6])
    t = np.concatenate([np.asarray(limits, float), det, np.asarray(later_limits, float)])
    ul = np.concatenate([np.ones(len(limits), bool), np.zeros(det.size, bool),
                         np.ones(len(later_limits), bool)])
    flux = np.where(ul, 0.3, 2.0)
    return wp.LightCurve(time=t, band=bands or ["r"] * t.size, flux=flux,
                         flux_err=np.where(ul, np.nan, 0.1), upper_limit=ul)


# ------------------------------------------------------------------- 1. with / without limits
def test_with_an_upper_limit_the_window_starts_30_days_before_it():
    p = wp.default_t_exp_prior(_lc())
    assert isinstance(p, Uniform) and p.name == "t_exp"
    assert p.bounds == (60001.0 - 30.0, 60003.4)


def test_the_anchor_is_the_last_limit_before_the_first_detection_in_any_band():
    lc = _lc(limits=(59990.0, 60002.0), later_limits=(60020.0,),
             bands=["g", "r"] + ["g"] * 6 + ["r"])
    assert wp.default_t_exp_prior(lc).bounds == (60002.0 - 30.0, 60003.4)


def test_without_an_upper_limit_the_anchor_is_the_first_detection():
    p = wp.default_t_exp_prior(_lc(limits=()))
    assert p.bounds == pytest.approx((60003.4 - 30.0, 60003.4))
    no_column = wp.LightCurve(time=[5.0, 6.0], band=["r", "r"], flux=[1.0, 2.0],
                              flux_err=[0.1, 0.1])
    assert wp.default_t_exp_prior(no_column).bounds == (-25.0, 5.0)


# --------------------------------------------------------------------------- 2. the window
def test_the_window_is_a_parameter():
    assert wp.default_t_exp_prior(_lc(), window=10.0).bounds == (59991.0, 60003.4)
    assert wp.default_t_exp_prior(_lc(limits=()), window=5).bounds == \
        pytest.approx((59998.4, 60003.4))
    for bad in (0.0, -3.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="window must be > 0 days and finite"):
            wp.default_t_exp_prior(_lc(), window=bad)
    with pytest.raises(ValueError, match="not enough data"):
        wp.default_t_exp_prior(_lc().where(upper_limit=True))


def test_an_infinite_module_constant_is_refused_by_name(monkeypatch):
    # Uniform(-inf, first detection) has density -inf everywhere: a fit could not start.
    monkeypatch.setattr(schema, "T_EXP_WINDOW_DAYS", float("inf"))
    with pytest.raises(ValueError, match="T_EXP_WINDOW_DAYS = inf"):
        wp.default_t_exp_prior(_lc())
    with pytest.raises(ValueError, match="T_EXP_WINDOW_DAYS = inf"):
        _pre_event_plan(_lc(), _free_model())


def test_the_module_constant_sets_the_default_of_every_later_fit(monkeypatch):
    monkeypatch.setattr(schema, "T_EXP_WINDOW_DAYS", 50.0)
    assert wp.default_t_exp_prior(_lc()).bounds == (59951.0, 60003.4)
    rec = _pre_event_plan(_lc(), _free_model()).t_exp_record
    assert (rec["low"], rec["high"], rec["window_days"]) == (59951.0, 60003.4, 50.0)


def test_it_is_wider_than_the_data_window():
    lc = _lc()
    narrow, default = lc.explosion_time_prior(), wp.default_t_exp_prior(lc)
    assert narrow.bounds == (60001.0, 60003.4)                         # unchanged
    assert default.low == narrow.low - 30.0 and default.high == narrow.high


# ------------------------------------------------------------------------------ 3. in a fit
def test_a_model_without_a_t_exp_prior_takes_the_window():
    plan = _pre_event_plan(_lc(), _free_model())
    rec = plan.t_exp_record
    assert plan.prior_changed and rec["source"] == "default window"
    assert (rec["low"], rec["high"], rec["anchor"]) == (59971.0, 60003.4, 60001.0)
    assert plan.prior.distributions["t_exp"].bounds == (59971.0, 60003.4)
    assert plan.rule["n_excluded"] == 3 and plan.lc.time.min() == 60003.4


def test_a_uniform_model_prior_is_cut_never_widened():
    wide = _pre_event_plan(_lc(), _free_model(Uniform(59900.0, 60100.0), "flare_wide"))
    assert wide.t_exp_record["source"] == "model prior, cut to the default window"
    assert wide.prior.distributions["t_exp"].bounds == (59971.0, 60003.4)
    narrow = _pre_event_plan(_lc(), _free_model(Uniform(59980.0, 60002.0), "flare_narrow"))
    assert narrow.t_exp_record["source"] == "model prior" and not narrow.prior_changed
    half = _pre_event_plan(_lc(), _free_model(Uniform(59960.0, 60002.0), "flare_half"))
    assert half.prior.distributions["t_exp"].bounds == (59971.0, 60002.0)
    with pytest.raises(ValueError, match="default window .* do not overlap"):
        _pre_event_plan(_lc(), _free_model(Uniform(59900.0, 59950.0), "flare_miss_default"))


def test_a_normal_model_prior_is_kept():
    plan = _pre_event_plan(_lc(), _free_model(Normal(60002.0, 1.0), "flare_normal"))
    assert plan.t_exp_record["source"] == "model prior" and not plan.prior_changed


def test_an_explicit_prior_passed_to_the_fit_wins():
    mine = Prior({**PHYS, "t_exp": Uniform(59900.0, 60003.0)})
    plan = _pre_event_plan(_lc(), _free_model(Uniform(59980.0, 60002.0), "flare_given"), mine)
    rec = plan.t_exp_record
    assert rec["source"] == "passed to fit" and (rec["low"], rec["high"]) == (59900.0, 60003.0)
    assert plan.prior is mine and not plan.prior_changed


def test_a_fit_samples_inside_the_window():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wp.fit(_lc(), _free_model(), sampler="abc", n_simulations=400, quantile=0.1,
                     seed=1)
    assert res.info["t_exp_prior"]["source"] == "default window"
    assert res.samples["t_exp"].between(59971.0, 60003.4).all()


# --------------------------------------------------------------------- 4. models without t_exp
def test_a_model_without_t_exp_is_not_touched():
    flare = wp.get_model("flare")
    plan = _pre_event_plan(_lc(), flare)
    assert plan.t_exp_record is None and plan.prior is None and not plan.prior_changed
    assert plan.rule["rule"] == "none" and len(plan.lc) == len(_lc())


# ------------------------------------------------------------------------------- 5. compare
def test_compare_binds_a_family_to_the_window_and_keeps_a_given_prior():
    pytest.importorskip("jax")
    import importlib

    C = importlib.import_module("whisper_cbpf.compare")

    t = [60000.0, 60002.0, 60003.0, 60004.5, 60006.0]
    lc = wp.LightCurve(time=t, band=["lsstg", "lsstr", "lsstg", "lsstr", "lssti"],
                       magnitude=[22.5, 21.0, 20.6, 20.4, 20.5],
                       magnitude_err=[np.nan, 0.1, 0.1, 0.1, 0.1],
                       upper_limit=[True, False, False, False, False], redshift=0.05)
    m = C._bind_family("magnetar", lc, None)
    assert m.default_prior.distributions["t_exp"].bounds == (59970.0, 60002.0)
    given = C._resolve_models(lc, ["magnetar"],
                              {"magnetar": Prior({"t_exp": Uniform(59900.0, 60002.0)})})
    assert given["magnetar"]["prior"].distributions["t_exp"].bounds == (59900.0, 60002.0)
