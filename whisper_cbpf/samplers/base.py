"""Sampler base class and the unified result container.

Also the one entry step every registered sampler's ``fit`` runs before it starts (the pre-event data
rule, :func:`prepare_lc`): rows at or before the event are never fitted.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def summarize_posterior(samples, parameters):
    """Per-parameter median / 16th-84th percentiles / mean / std from accepted samples."""
    summary = {}
    if len(samples) == 0:
        return summary
    for p in parameters:
        if p in samples:
            v = samples[p].to_numpy(dtype=float)
            summary[p] = {
                "median": float(np.median(v)),
                "ci16": float(np.percentile(v, 16)),
                "ci84": float(np.percentile(v, 84)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            }
    return summary


def aic_bic(max_log_likelihood, n_params, n_data):
    """``(AIC, BIC)`` from the maximum log-likelihood, both on the deviance scale.

    No guard for ``n_data = 0``: no sampler reaches this with an empty light curve
    (:func:`check_not_empty`).
    """
    k, n = int(n_params), int(n_data)
    return (float(-2.0 * max_log_likelihood + 2 * k),
            float(-2.0 * max_log_likelihood + k * np.log(n)))


def check_not_empty(lc):
    """Refuse an empty light curve, with the message the likelihood constructors give.

    Every density whisper builds goes through a likelihood, and ``GaussianLikelihood`` already
    refuses one. A caller-supplied ``log_prob_fn`` (``nuts_gpu``, ``pymc_jax_gpu_*``,
    ``emcee_jax``) builds none, so those fits ran on zero data and returned AIC ``2k`` with BIC
    ``-inf`` (``k * log(0)``, the best model in any argmin) or ``0.0`` (the PyMC pair's inline
    ``log(max(n, 1))``): one input, two answers, no error (``tests/test_empty_light_curve.py``).
    """
    if len(lc.time) == 0:
        raise ValueError(
            "the light curve has no data points, so there is nothing to fit. A common cause is "
            "calling select_time_window(...) BEFORE set_explosion_date(...) -- the window is "
            "then compared against raw MJD. Check lc.n_points after each selection step.")


def check_burnin(burnin, nsteps):
    """Refuse a burn-in that would discard the whole chain.

    ``get_chain(discard=burnin)`` returns zero rows when ``burnin >= nsteps``, and the AIC/BIC
    step that follows it calls ``np.nanargmax`` on that empty array. Raising here names the two
    numbers, and does it before the sampling rather than after.
    """
    if int(burnin) >= int(nsteps):
        raise ValueError(
            f"burnin={int(burnin)} discards the whole chain of nsteps={int(nsteps)}. "
            f"The burn-in must be shorter than the chain; raise nsteps or lower burnin.")


# ================================================================================ pre-event data
# The rule (user decision, 2026-09-26): pre-explosion / pre-merger data are never fitted. They are
# left out of the likelihood AND of the model evaluation, so they cannot move a model's time grid,
# and they may only define a prior (the explosion time's, when it is fitted).

#: Samplers that compare simulated and observed values point by point (ABC's distance, SNPE's
#: simulator). A non-detection is a bound, not a value, so they cannot use one; the likelihood
#: samplers fit it with the censored likelihood.
NO_UPPER_LIMIT_SAMPLERS = ("abc", "abc_smc", "abc_gpu", "abc_smc_gpu", "snpe", "npe", "snpe_gpu")

#: The likelihood samplers a light curve with upper limits can go to instead.
_LIMIT_SAMPLERS = "'mcmc', 'nested', 'emcee_jax' or 'nuts_gpu'"

#: ``lc.meta`` keys that declare the event itself (an MJD), with its name.
#: :meth:`~whisper_cbpf.LightCurve.set_explosion_date` writes the first; a merger time may be
#: recorded under the second.
_EVENT_KEYS = (("explosion_mjd", "explosion"), ("merger_mjd", "merger"))

#: :meth:`~whisper_cbpf.LightCurve.set_time_reference` labels that name the event itself.
_EVENT_LABELS = ("explosion", "merger")

#: Parameters by which a model fits its own epoch (the peak time of ``bazin``, ``gaussian_rise``
#: and ``flare_jax``; redback's phenomenological models call it ``tpeak``). Such a model has no
#: event at day 0, so a time reference that is not the explosion or merger (a peak, a first
#: detection) leaves nothing out for it.
_EPOCH_PARAMS = ("t0", "tpeak", "t_peak")


def _declared_event(lc):
    """``(label, mjd, t, is_event)`` of the light curve's declared reference, or ``None``.

    ``mjd`` is the reference's MJD, ``t`` where it falls on the curve's own clock (0 after
    :meth:`~whisper_cbpf.LightCurve.set_explosion_date`; the MJD itself on a raw MJD clock), and
    ``is_event`` whether it is the explosion or merger rather than another reference (a first
    detection, a peak).
    """
    meta = getattr(lc, "meta", None) or {}
    ref_mjd = meta.get("time_reference_mjd")
    offset = 0.0 if ref_mjd is None else float(ref_mjd)       # raw MJD clock without a reference
    for key, name in _EVENT_KEYS:
        if meta.get(key) is not None:
            mjd = float(meta[key])
            return name, mjd, mjd - offset, True
    if ref_mjd is not None:
        label = str(meta.get("time_reference") or "time reference")
        return label, float(ref_mjd), 0.0, label.strip().lower() in _EVENT_LABELS
    return None


def _upper_limits(lc):
    ul = getattr(lc, "upper_limit", None)
    return (np.zeros(len(lc), dtype=bool) if ul is None else np.asarray(ul, dtype=bool))


def _resolve_model(model):
    """The :class:`~whisper_cbpf.models.Model` behind ``model``, or ``None`` if it has none (the
    sampler then raises its own error for it)."""
    from ..models import Model, get_model
    if isinstance(model, Model):
        return model
    try:
        return get_model(model)
    except (KeyError, TypeError):
        return None


def _keep_model(result, model):
    """Keep the fitted :class:`~whisper_cbpf.models.Model` on ``result`` (not saved, not pickled)."""
    from ..models import Model
    if isinstance(model, Model) and getattr(model, "name", None) == getattr(result, "model", None):
        result._model_object = model


def fitted_model(result, model=None):
    """The :class:`~whisper_cbpf.models.Model` behind a fit.

    ``model`` when it is given (a name or an object); else the Model object the fit ran with, which
    every sampler keeps on its result (so a model built by :func:`whisper_cbpf.supernova_model`,
    :func:`whisper_cbpf.tde_model` or :func:`whisper_cbpf.compare` is found although it is not
    registered by name); else the registered model ``result.model`` names. A result loaded from
    disk has only the name.

    Parameters
    ----------
    result : SamplerResult
        The fit.
    model : str or Model, optional
        Overrides the fit's own model.

    Returns
    -------
    Model

    Raises
    ------
    KeyError
        The fit's model is not registered in this session and the result does not carry it (a
        loaded result of a factory-built model): pass ``model=``.

    Examples
    --------
    >>> import numpy as np, whisper_cbpf as wp
    >>> from whisper_cbpf.samplers.base import fitted_model
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> flux = wp.get_model("flare").predict(
    ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
    >>> mine = wp.Model(name="my_flare", predict=wp.get_model("flare").predict,
    ...                 parameters=["amplitude", "rise_time", "decay_time"],
    ...                 default_prior=wp.get_model("flare").default_prior)
    >>> res = wp.fit(lc, mine, sampler="abc", n_simulations=500, quantile=0.1)
    >>> fitted_model(res) is mine                  # not registered, still found
    True
    """
    from ..models import get_model
    if model is not None:
        return get_model(model)
    obj = getattr(result, "_model_object", None)
    if obj is not None:
        return obj
    name = getattr(result, "model", None)
    try:
        return get_model(name)
    except KeyError:
        raise KeyError(
            f"the fit's model {name!r} is not registered in this session, and this result does "
            f"not carry the Model object (a result loaded from disk has only its name). Pass "
            f"model= (the Model the fit used, e.g. comparison.models[{name!r}] or the "
            f"supernova_model(...) call that built it).") from None


def _has_t_exp(model):
    return model is not None and "t_exp" in [str(p) for p in model.parameters]


def _t_exp_prior(lc, model, base, given):
    """``(distribution, record)``: the explosion-time prior a fit of ``model`` uses.

    ``given`` is the prior passed to the fit (it wins when it names ``t_exp``), ``base`` the one the
    fit starts from (``given`` or the model's default). The data window runs from the last
    non-detection before the first detection to the first detection
    (:meth:`~whisper_cbpf.LightCurve.explosion_time_prior`).
    """
    from ..io.schema import EXPLOSION_FALLBACK_DAYS, explosion_window
    from ..priors import Uniform

    w = explosion_window(lc.time, getattr(lc, "upper_limit", None))
    t_first, last = w["first_detection"], w["last_non_detection"]
    own = base.distributions.get("t_exp") if base is not None else None
    if given is not None and "t_exp" in given.distributions:
        new, source = given.distributions["t_exp"], "passed to fit"
    elif own is None:
        if last is not None:
            new, source = Uniform(last, t_first), "data"
        else:
            new, source = Uniform(t_first - EXPLOSION_FALLBACK_DAYS, t_first), "data, fallback window"
    elif isinstance(own, Uniform):
        low = own.low if last is None else max(own.low, last)
        high = min(own.high, t_first)
        if not low < high:
            lo = "-inf" if last is None else f"{last:.10g}"
            raise ValueError(
                f"model {model.name!r}: its explosion-time prior {own!r} and the window the data "
                f"allow ({lo}, {t_first:.10g}] -- after the last non-detection, up to the first "
                f"detection, on this light curve's clock -- do not overlap. Pass prior= with a "
                f"t_exp prior that covers the window, or check the light curve's clock (the "
                f"model's t_exp is on it).")
        changed = (low, high) != (own.low, own.high)
        new = Uniform(low, high) if changed else own
        source = "model prior, cut to the data window" if changed else "model prior"
    else:
        new, source = own, "model prior"
    bounds = getattr(new, "bounds", (None, None))
    record = {"type": type(new).__name__, "low": _num(bounds[0]), "high": _num(bounds[1]),
              "repr": repr(new), "source": source, "first_detection": t_first,
              "last_non_detection": last, "model_prior": None if own is None else repr(own)}
    return new, record


def _num(x):
    return None if x is None else float(x)


def _event_rule(lc, model, t_exp_dist):
    """Which rows of ``lc`` are pre-event: ``{"rule", "reference", "inclusive", "label", ...}``.

    ``"free t_exp"``: the model fits its explosion time, so the rows before the first detection
    (non-detections) are left out -- they define the explosion-time prior instead. ``"fixed t_exp"``:
    a ``Fixed`` explosion time; rows at or before it are left out. ``"day 0"``: the light curve
    declares its event -- the explosion or merger (``reference`` is where it falls on the clock),
    or day 0 of another reference, which is the event of a model whose clock starts there -- and
    rows at or before it are left out. ``"none"``: no event on this clock (or a non-event
    reference and a model that fits its own epoch, ``reason`` says so); nothing left out.
    """
    from ..priors import Fixed

    if _has_t_exp(model):
        if isinstance(t_exp_dist, Fixed):
            return {"rule": "fixed t_exp", "reference": float(t_exp_dist.value),
                    "inclusive": True, "label": "the fixed explosion time"}
        from ..io.schema import explosion_window
        t_first = explosion_window(lc.time, getattr(lc, "upper_limit", None))["first_detection"]
        return {"rule": "free t_exp", "reference": t_first, "inclusive": False,
                "label": "first detection"}
    event = _declared_event(lc)
    if event is not None:
        label, mjd, ref, is_event = event
        params = [] if model is None else [str(p) for p in model.parameters]
        epoch = next((p for p in _EPOCH_PARAMS if p in params), None)
        if is_event or epoch is None:
            return {"rule": "day 0", "reference": ref, "inclusive": True, "label": label,
                    "reference_mjd": mjd}
        return {"rule": "none", "reference": None, "inclusive": None, "label": label,
                "reason": f"day 0 of this light curve is its {label!r}, not the explosion or "
                          f"merger, and model {model.name!r} fits its own epoch ({epoch!r})"}
    return {"rule": "none", "reference": None, "inclusive": None, "label": None}


def _pre_event_mask(time, rule):
    """``True`` for the rows a fit keeps under ``rule`` (an ``info["pre_event"]`` record)."""
    t = np.asarray(time, dtype=float)
    ref = (rule or {}).get("reference")
    if (rule or {}).get("rule") in (None, "none") or ref is None:
        return np.ones(t.size, dtype=bool)
    return t > float(ref) if rule.get("inclusive") else t >= float(ref)


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _user_stacklevel():
    """The ``stacklevel`` (for a ``warnings.warn`` called in the caller of this function) of the
    first frame outside this package: the user's own line, however deep the package's calls go
    (``wp.fit`` -> the provenance wrapper -> a sampler -> a helper)."""
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR):
        frame, level = frame.f_back, level + 1
    return level


def _warn_user(message, category=UserWarning):
    """Warn at the first frame outside this package: the user's own line."""
    warnings.warn(message, category, stacklevel=_user_stacklevel())


class _PreEventPlan:
    """What a fit of ``model`` to ``lc`` leaves out before it starts, and its explosion-time prior."""

    def __init__(self, lc, prior, prior_changed, rule, t_exp_record, message):
        self.lc, self.prior, self.prior_changed = lc, prior, prior_changed
        self.rule, self.t_exp_record, self.message = rule, t_exp_record, message

    def warn(self):
        if self.message:
            _warn_user(self.message)

    def stamp(self, result):
        """Record the rule in ``result.info``: ``excluded_pre_event``, ``pre_event``, ``t_exp_prior``."""
        if not isinstance(result.info, dict):
            return
        result.info["excluded_pre_event"] = int(self.rule.get("n_excluded", 0))
        result.info["pre_event"] = dict(self.rule)
        if self.t_exp_record is not None:
            result.info["t_exp_prior"] = dict(self.t_exp_record)


def _pre_event_plan(lc, model=None, prior=None, *, sampler=None, own_density=False,
                    likelihood=None):
    """The pre-event rule for one fit: the rows kept, the prior used, the record and the message.

    Raises for "not enough data" (nothing, or no detection, after the event), for a sampler that
    cannot use the upper limits left after the event (:data:`NO_UPPER_LIMIT_SAMPLERS`), and for a
    likelihood object built on rows the fit leaves out.
    """
    if lc is None or not hasattr(lc, "time") or len(lc) == 0:
        # An empty light curve keeps the samplers' own message (`check_not_empty`).
        return _PreEventPlan(lc, prior, False, {"rule": "none", "reference": None,
                                                "inclusive": None, "label": None,
                                                "n_excluded": 0}, None, None)
    model = _resolve_model(model)
    base = prior if prior is not None else getattr(model, "default_prior", None)
    used, changed, t_rec = prior, False, None
    if _has_t_exp(model):
        t_dist, t_rec = _t_exp_prior(lc, model, base, prior)
        if base is not None and base.distributions.get("t_exp") is not t_dist:
            used = type(base)({**base.distributions, "t_exp": t_dist})
            changed = True
    else:
        t_dist = None
    rule = _event_rule(lc, model, t_dist)
    ul = _upper_limits(lc)
    keep = _pre_event_mask(lc.time, rule)
    n_excl = int(np.sum(~keep))
    rule.update(n_excluded=n_excl, n_detections=int(np.sum(~keep & ~ul)),
                n_upper_limits=int(np.sum(~keep & ul)))
    lc_fit = lc[keep] if n_excl else lc
    ul_fit = ul[keep]
    _check_after_event(lc, rule, ul_fit)
    if sampler in NO_UPPER_LIMIT_SAMPLERS and np.any(ul_fit):
        raise ValueError(
            f"{sampler} cannot use upper limits: it compares simulated and observed values point "
            f"by point, and a non-detection is a bound, not a value. This light curve has "
            f"{int(np.sum(ul_fit))} upper limit(s) after the event. Fit "
            f"lc.where(upper_limit=False) with {sampler} (the limits then play no part in its "
            f"result), or use a likelihood sampler ({_LIMIT_SAMPLERS}), which fits them with the "
            f"censored likelihood by default; a {sampler} fit of the detections can start it "
            f"(init=).")
    n_built = np.size(getattr(likelihood, "y", None)) if not isinstance(likelihood, str) else 0
    if n_excl and likelihood is not None and n_built == len(lc):
        # A built likelihood holds its own copy of the data: on the full curve it would score the
        # pre-event rows (or fail on a shape mismatch deep inside a compiled density).
        raise ValueError(
            f"the likelihood object passed to this fit was built on all {len(lc)} rows, but "
            f"pre-event data are not fitted: {n_excl} row(s) at or before {_where(rule)} are left "
            f"out, so the fit uses {len(lc_fit)}. Build it on the rows the fit uses "
            f"(make_likelihood(prepare_lc(lc, model), ...), prepare_lc from "
            f"whisper_cbpf.samplers.base), or pass the likelihood's name, e.g. "
            f"likelihood='upper_limits'.")
    message = _pre_event_message(lc, rule, t_rec, changed, own_density)
    return _PreEventPlan(lc_fit, used, changed, rule, t_rec, message)


def _check_after_event(lc, rule, ul_fit):
    """"Not enough data" when nothing, or no detection, is left after the event."""
    where = _where(rule)
    if rule.get("rule") == "none" and not np.any(~ul_fit):
        raise ValueError(
            f"not enough data: this light curve has no detection ({int(ul_fit.size)} upper "
            f"limit(s) only). A fit needs at least one detection; wait for one.")
    if ul_fit.size == 0:
        raise ValueError(
            f"not enough data: all {len(lc)} rows of this light curve are at or before {where}, "
            f"and pre-event data are never fitted. Check the time reference (it should be the "
            f"explosion or merger), or wait for data after the event.")
    if not np.any(~ul_fit):
        raise ValueError(
            f"not enough data: no detection is left after {where} ({int(ul_fit.size)} upper "
            f"limit(s) only). A fit needs at least one detection after the event; wait for one.")


def _where(rule):
    kind = rule.get("rule")
    if kind == "day 0":
        if rule["reference"] == 0.0:
            return (f"day 0 of this light curve ({rule['label']!r}, MJD "
                    f"{rule['reference_mjd']:.5f})")
        return (f"the {rule['label']} (MJD {rule['reference_mjd']:.5f}, t = "
                f"{rule['reference']:.10g} on this light curve's clock)")
    if kind == "fixed t_exp":
        return f"the fixed explosion time (t_exp = {rule['reference']:.10g})"
    if kind == "free t_exp":
        return f"the first detection (t = {rule['reference']:.10g})"
    return "the event"


def _pre_event_message(lc, rule, t_rec, changed, own_density):
    """The one message a fit gives about the rule, or ``None`` when it changed nothing."""
    parts = []
    n, n_det, n_ul = rule["n_excluded"], rule["n_detections"], rule["n_upper_limits"]
    if n:
        if rule["rule"] == "free t_exp":
            sets = changed and t_rec is not None and t_rec["source"] != "data, fallback window"
            parts.append(f"pre-event data are not fitted: the {n} upper limit(s) before "
                         f"{_where(rule)} are left out of the fit"
                         + (" and set the explosion-time prior instead" if sets else ""))
        else:
            parts.append(f"pre-event data are not fitted: {n} of {len(lc)} rows ({n_ul} upper "
                         f"limit(s), {n_det} detection(s)) are at or before {_where(rule)} and "
                         f"are left out; the fit uses the other {len(lc) - n}")
            if n_det:
                parts.append("a detection at or before the event means the reference is not the "
                             "explosion or merger: set it to the event (lc.set_explosion_date), "
                             "or fit the explosion time (a model built with free=['t_exp'])")
    if changed and t_rec is not None:
        if t_rec["source"] == "data":
            why = "from the last non-detection before the first detection to the first detection"
        elif t_rec["source"] == "data, fallback window":
            why = ("no non-detection precedes the first detection, so a fallback window of "
                   "whisper_cbpf.io.schema.EXPLOSION_FALLBACK_DAYS days before it")
        else:
            why = f"the model's {t_rec['model_prior']} cut to the window the data allow"
        parts.append(f"explosion-time prior: t_exp ~ {t_rec['repr']} ({why}); pass prior= to "
                     f"choose another")
    if not parts:
        return None
    if own_density and n:
        parts.append("the log_prob_fn passed to this fit was built by the caller: build it on "
                     "whisper_cbpf.samplers.base.prepare_lc(lc, model) so it leaves out the same "
                     "rows")
    return "; ".join(parts) + ". See result.info['pre_event']."


def prepare_lc(lc, model=None, *, prior=None):
    """The rows of ``lc`` a fit of ``model`` uses: pre-event data left out.

    Every registered sampler's ``fit`` applies this rule before it starts, so its likelihood and
    its model evaluations never see a row at or before the event (a pre-event epoch cannot move a
    model's time grid either). Call it to build a density of your own on the same rows, or to see
    which rows a fit will use.

    - **Explosion or merger declared** (:meth:`~whisper_cbpf.LightCurve.set_explosion_date`, a
      ``set_time_reference`` labelled ``"explosion"`` or ``"merger"``, or ``meta["merger_mjd"]``,
      an MJD): rows at or before it are left out (``time <= 0`` after ``set_explosion_date``).
    - **Another time reference** (``set_time_reference(mjd, "first detection")``, ``"peak"``,
      ...): a model's clock starts at its day 0, so rows with ``time <= 0`` are left out -- except
      for a model that fits its own epoch (a ``t0``/``tpeak`` parameter: ``bazin``,
      ``gaussian_rise``, ``flare_jax``), whose clock has no event there; nothing is left out for it.
    - **Free explosion time** (a model with a ``t_exp`` parameter, e.g. ``free=["t_exp"]``): the
      rows before the first detection -- non-detections -- are left out, and set the explosion-time
      prior instead (:meth:`~whisper_cbpf.LightCurve.explosion_time_prior`). A ``Fixed`` t_exp
      leaves out the rows at or before its value.
    - **No declared event** (a raw MJD clock, a model without ``t_exp``): nothing is left out.

    Parameters
    ----------
    lc : LightCurve
        The data.
    model : str or Model, optional
        The model to be fitted; it decides whether the explosion time is free.
    prior : Prior, optional
        The prior to be passed to the fit (a ``Fixed`` t_exp in it sets the event).

    Returns
    -------
    LightCurve
        ``lc`` itself when nothing is left out, else the rows kept.

    Raises
    ------
    ValueError
        "not enough data": no row, or no detection, is left after the event.

    Examples
    --------
    >>> import numpy as np, whisper_cbpf as wp
    >>> from whisper_cbpf.samplers.base import prepare_lc
    >>> lc = wp.LightCurve(time=[59998.0, 59999.5, 60001.0, 60004.0], band=["r"] * 4,
    ...                    flux=[0.0, 0.1, 2.0, 3.0], flux_err=[0.1] * 4)
    >>> prepare_lc(lc.set_explosion_date(60000.0), "flare").time.tolist()   # day 0 = explosion
    [1.0, 4.0]
    >>> len(prepare_lc(lc, "flare"))                    # raw MJD clock: nothing is left out
    4
    >>> peak = lc.set_time_reference(60001.0, "peak")   # bazin fits its own epoch, t0
    >>> len(prepare_lc(peak, "bazin")), len(prepare_lc(peak, "flare"))
    (4, 1)
    """
    return _pre_event_plan(lc, model, prior).lc


def _recorded_prior(result):
    """The prior a fit used when it was not the model's default (one passed to the fit, or
    completed by the pre-event rule), rebuilt from its provenance record; ``None`` otherwise, or
    when it cannot be rebuilt."""
    prov = getattr(result, "provenance", None)
    model = (prov or {}).get("model") if isinstance(prov, dict) else None
    if not isinstance(model, dict) or model.get("prior_source") != "passed to fit":
        return None
    return _prior_from_record(model.get("prior"))


def _prior_from_record(record):
    """A :class:`~whisper_cbpf.priors.Prior` rebuilt from a provenance prior record
    (:func:`whisper_cbpf.results.prior_record`); ``None`` when it cannot be rebuilt."""
    from ..priors import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform

    rec = (record or {}).get("parameters") if isinstance(record, dict) else None
    if not rec:
        return None
    build = {"Uniform": lambda r: Uniform(r["low"], r["high"]),
             "LogUniform": lambda r: LogUniform(r["low"], r["high"]),
             "Normal": lambda r: Normal(r["mu"], r["sigma"]),
             "TruncatedNormal": lambda r: TruncatedNormal(r["mu"], r["sigma"], r["low"], r["high"]),
             "Fixed": lambda r: Fixed(r["value"])}
    try:
        return Prior({name: build[r["type"]](r) for name, r in rec.items()})
    except (KeyError, TypeError, ValueError):
        return None


#: ``type(likelihood).__name__`` (what samplers record in ``info["likelihood"]``) -> registry key.
#: Anything unrecognised falls back to ``"auto"``.
_LIKELIHOOD_KINDS = {
    "GaussianLikelihood": "gaussian",
    "GaussianLikelihoodWithUpperLimits": "gaussian_upper_limits",
    "GaussianLikelihoodWithScatter": "gaussian_scatter",
    "MixtureGaussianLikelihood": "mixture",
}


def _warn_without_raising(message):
    """Emit a UserWarning that cannot abort the caller, even under warnings-as-errors.

    Both ``attach_*`` helpers warn from inside an ``except`` block. Under ``-W error`` the warning
    is itself an exception, so it would escape that handler and kill a fit that had already
    succeeded. A diagnostic must not destroy the thing it is diagnosing, so the filter is forced
    locally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, stacklevel=_user_stacklevel())


def attach_band_metrics(info, lc, model, best_params, space):
    """Populate ``info['band_metrics']`` with per-band MSE/MAE at the best fit (best-effort).

    Shared by every sampler, so per-band goodness-of-fit lands in ``SamplerResult.to_json``. The
    broad guard is deliberate: a metric failure must not break an otherwise-successful fit. The
    failure is reported rather than swallowed, in ``info['band_metrics_error']`` and in a warning.
    """
    try:
        from ..metrics import per_band_metrics
        info["band_metrics"] = per_band_metrics(lc, model, best_params, space=space)
    except Exception as exc:                                  # noqa: BLE001 - deliberate, see above
        info["band_metrics_error"] = f"{type(exc).__name__}: {exc}"
        _warn_without_raising(f"per-band metrics could not be computed and are absent from this "
                              f"result ({type(exc).__name__}: {exc}). The fit itself is unaffected.")


def attach_predictive_metrics(result, lc, space, n_draws=200, *, model=None):
    """Populate ``result.info['predictive_metrics']`` with the posterior predictive metric block.

    RMSE (posterior-mean), LPD, ELPD (PSIS-LOO), WAIC, AIC/BIC and the coverage-calibration curve —
    see :func:`whisper_cbpf.metrics.predictive_metrics`. Called after the result is built (it reuses
    the fit's ``aic`` / ``bic``); ``n_draws`` bounds the extra forward-model evaluations. ``model``
    is the fitted :class:`~whisper_cbpf.models.Model` (by default the one ``result.model`` names),
    so a model built by a factory and never registered by name is scored too.

    Best-effort: a failure (slow or failing model, no likelihood) leaves the fit untouched but not
    the user uninformed. WAIC is one of the three numbers this package exists to put side by side,
    so "absent" and "absent because your bands did not resolve" have to be distinguishable: the
    reason lands in ``info['predictive_metrics_error']`` and in a warning.

    An empty posterior (an ABC fit that accepted no draw) is not a failure but a state: there is
    nothing to score, and ``info['predictive_metrics_skipped']`` says so by name.

    The fitted ``model`` is also kept on the result, so :meth:`SamplerResult.likelihood_max_opt`,
    :meth:`SamplerResult.forecast`, :func:`whisper_cbpf.waic` and the plots find it without
    ``model=`` even when it is not registered by name (see :func:`fitted_model`).
    """
    _keep_model(result, model)
    if result.n_samples == 0:
        # Without this the metrics failed on an empty stack and warned "need at least one array
        # to concatenate", which reads like a crash rather than "ABC accepted nothing".
        result.info["predictive_metrics_skipped"] = "no accepted draws"
        return
    if result.n_samples == 1:
        # One draw has no posterior variance: WAIC and LOO are undefined (numpy warned about
        # "degrees of freedom <= 0" and returned NaN).
        result.info["predictive_metrics_skipped"] = ("one posterior draw: the predictive metrics "
                                                     "need at least two")
        return
    # Score under the density the fit was run with. `predictive_metrics` defaults to kind="auto",
    # a plain Gaussian, which would rescore a `likelihood="mixture"` fit under a different density
    # and then blame the model for the resulting p_waic: on 40 points with one 30-sigma outlier
    # that reads as p_waic = 324.11 and "unreliable" against 2.35 and reliable under the mixture.
    kind = _LIKELIHOOD_KINDS.get(str(result.info.get("likelihood", "")), "auto")
    # Likewise the scatter column: the one the fit recorded (None when it fitted none), not the
    # "auto" guess from the posterior's columns -- that guess took ABC's `distance` for a sigma.
    # A sampler that records nothing keeps "auto".
    scatter = result.info.get("scatter_param", "auto")
    try:
        from ..metrics import predictive_metrics
        result.info["predictive_metrics"] = predictive_metrics(
            result, lc, model, space=space, likelihood=kind, scatter_param=scatter,
            n_draws=n_draws)
    except Exception as exc:                                  # noqa: BLE001 - deliberate, see above
        result.info["predictive_metrics_error"] = f"{type(exc).__name__}: {exc}"
        _warn_without_raising(f"predictive metrics (WAIC / LPD / PSIS-LOO / coverage) could not be "
                              f"computed and are absent from this result ({type(exc).__name__}: "
                              f"{exc}). AIC and BIC are unaffected.")


@dataclass
class SamplerResult:
    """Unified result for every Whisper sampler.

    ``samples`` holds the accepted/posterior draws. Model-selection metrics (``aic``, ``bic``,
    ``max_log_likelihood``) come from the best fit. With a chi-square distance these use
    ``chi2 = -2 ln L`` **up to an additive constant** (the Gaussian normalization is dropped, so
    absolute values are offset, but model comparison on the same data is unaffected); a proper
    ``whisper_cbpf.likelihood`` gives exact values. ``info`` carries sampler-specific diagnostics.

    ``aic`` and ``bic`` are plain fields; the posterior-predictive numbers live in the
    ``info['predictive_metrics']`` block and are surfaced beside them as the read-only properties
    :attr:`waic`, :attr:`waic_reliable`, :attr:`elpd_loo` and :attr:`rmse`, each ``None`` rather
    than raising when that block is missing or failed.

    ``provenance`` is how the fit was made -- whisper and package versions, git state, the data's
    hash, the model with its prior, the sampler with its settings and seed, devices, wall time --
    recorded by every registered sampler's ``fit`` (:class:`BaseSampler`) and written by
    :meth:`save`. Empty for a result built by hand.

    Attributes
    ----------
    sampler, model : str
        The sampler's and the model's names.
    parameters : list of str
        The posterior's columns, in the model's order (a ``Fixed`` one as a constant column).
    samples : pandas.DataFrame
        The posterior draws (ABC: the accepted draws, with their ``distance``).
    summary : dict
        Per parameter: median, 16th and 84th percentiles, mean, std.
    best_params : dict
        The draw with the highest likelihood (ABC: the accepted draw with the highest likelihood).
    n_data, n_params : int
        Points fitted (pre-event rows excluded) and free parameters.
    max_log_likelihood, aic, bic : float
        At ``best_params``; :meth:`likelihood_max_opt` finds the likelihood peak behind them.
    runtime_s : float
    info : dict
        Sampler diagnostics and records (``space``, ``likelihood``, ``converged``,
        ``pre_event``, ``likelihood_max_opt``, the predictive metrics, ...).
    provenance : dict

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=wp.get_model("flare").predict(truth, t),
    ...                    flux_err=np.full(30, 0.2))
    >>> res = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0)
    >>> res.parameters, res.n_data, res.n_params
    (['amplitude', 'rise_time', 'decay_time'], 30, 3)
    >>> bool(np.isclose(res.bic, res.n_params * np.log(res.n_data) - 2 * res.max_log_likelihood))
    True
    """

    sampler: str
    model: str
    parameters: list
    samples: pd.DataFrame
    summary: dict
    best_params: dict
    n_data: int
    n_params: int
    runtime_s: float
    info: dict = field(default_factory=dict)
    min_distance: float = float("nan")
    max_log_likelihood: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    provenance: dict = field(default_factory=dict)

    @property
    def n_samples(self):
        return int(len(self.samples))

    def __getstate__(self):
        # The fitted Model object (see fitted_model) can hold compiled functions that do not
        # pickle; a copy or an unpickled result finds its model by name instead.
        state = dict(self.__dict__)
        state.pop("_model_object", None)
        return state

    def _metric(self, *path):
        """Read one number out of ``info['predictive_metrics']``; ``None`` if it is not there.

        The block is best-effort (see :func:`attach_predictive_metrics`): when it fails the key is
        absent and the reason is parked in ``info['predictive_metrics_error']``. Walking the path
        returns ``None`` at every level rather than raising ``KeyError`` on exactly the fits where
        something has already gone wrong -- including ``elpd_loo is None`` when arviz is absent.
        """
        node = self.info.get("predictive_metrics") if isinstance(self.info, dict) else None
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    @property
    def waic(self):
        """WAIC on the deviance scale (lower is better), or ``None`` if the metric block is absent.

        Equal by construction to ``whisper_cbpf.waic(result, result.fitted_lc(lc))["waic"]`` -- the
        manual call adopts the likelihood, space, scatter column and draw count recorded here, and
        :meth:`fitted_lc` drops the pre-event rows the fit left out. Check :attr:`waic_reliable`
        before comparing it against another model's.
        """
        return self._metric("waic", "waic")

    @property
    def waic_reliable(self):
        """``p_waic <= n_data/2`` (Gelman/Hwang/Vehtari 2014 §4), or ``None`` if WAIC is absent.

        ``False`` means the posterior's pointwise log-likelihood varies by orders of magnitude --
        usually a stranded chain -- and this WAIC must not be compared or read beside AIC/BIC.
        """
        return self._metric("waic", "p_waic_reliable")

    @property
    def elpd_loo(self):
        """PSIS-LOO expected log predictive density (higher is better).

        ``None`` when the metric block is absent **or** ``arviz`` is not installed; in the latter
        case ``info['predictive_metrics']['elpd_loo']`` is ``None`` and, if arviz is installed but
        errored, ``info['predictive_metrics']['elpd_loo_error']`` says why.
        """
        return self._metric("elpd_loo", "elpd_loo")

    @property
    def rmse(self):
        """Overall RMSE of ``observed - posterior-mean prediction``, in the fit's comparison space.

        Jy for a flux fit, mag for a magnitude fit (``info['predictive_metrics']['unit']``); per-band
        values are under ``['rmse']['bands']``. ``None`` if the metric block is absent.
        """
        return self._metric("rmse", "overall")

    def to_dict(self):
        # waic/elpd_loo/rmse are views on info["predictive_metrics"], which this dict already
        # carries in full. Promoting them would duplicate the same numbers under new top-level
        # keys and change the JSON shape that existing readers parse.
        return {
            "sampler": self.sampler,
            "model": self.model,
            "parameters": list(self.parameters),
            "n_data": int(self.n_data),
            "n_params": int(self.n_params),
            "n_samples": self.n_samples,
            "runtime_s": float(self.runtime_s),
            "min_distance": float(self.min_distance),
            "max_log_likelihood": float(self.max_log_likelihood),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "best_params": {k: float(v) for k, v in self.best_params.items()},
            "summary": self.summary,
            "info": self.info,
        }

    def to_json(self, path=None, indent=2):
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def save(self, path, *, overwrite=False):
        """Save this result, with its provenance, so :func:`whisper_cbpf.load_result` restores it.

        Parameters
        ----------
        path : str or Path
            A directory (``manifest.json``, ``result.json``, ``arrays.npz``), or a path ending in
            ``.npz`` for one file holding the same content.
        overwrite : bool, default False
            Replace a result already saved at ``path``.

        Returns
        -------
        Path
            Where the result was written.

        Notes
        -----
        Saved: the draws (and, for chain samplers, the draws by chain), the summary, the best fit,
        AIC/BIC and the other metrics, ``info``, and a manifest with the provenance (whisper version
        and git state, package versions, the model's name, description, parameters and prior, the
        sampler and its settings, the seed, the devices, the time split into compile and run where
        the sampler records it, the data's hash) and the hashes :func:`~whisper_cbpf.load_result`
        checks. Live sampler objects (``emcee_sampler``, ``numpyro_mcmc``, ...) are not saved; the
        manifest lists them under ``"not_saved"``.

        Raises
        ------
        FileExistsError
            ``path`` already holds a result and ``overwrite`` is False.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> from whisper_cbpf.models.flare import flare_flux
        >>> t = np.linspace(0.5, 30.0, 40)
        >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
        >>> res = wp.fit(lc, "flare", sampler="mcmc", nsteps=600, burnin=200, seed=0)
        >>> res.save("flare_mcmc.npz", overwrite=True).name
        'flare_mcmc.npz'
        >>> wp.load_result("flare_mcmc.npz").summary == res.summary
        True
        """
        from ..results import save_result
        return save_result(self, path, overwrite=overwrite)

    def diagnostics(self, *, prior=None, likelihood_max_opt=None):
        """One convergence report for this fit: every applicable check, pass or fail, and why.

        The checks depend on the sampler. Every fit: posterior draws exist, more data points than
        free parameters, no pile-up of draws against a prior bound. NUTS (``nuts_gpu``,
        ``pymc_jax_gpu_*``): divergences, rank-normalised split R-hat on every parameter and on the
        log-likelihood (< 1.01), bulk and tail ESS (>= 100 per chain), stranded and frozen chains,
        and the gap between the best draw and the prior scan's independent optimum (<= 10 nats).
        emcee (``mcmc``, ``emcee_jax``): stuck walkers, chain length over the LARGEST
        autocorrelation time (>= 50), R-hat across walkers (< 1.05; walkers are not independent
        chains) and bulk and tail ESS (>= 400). ABC: accepted draws (>= 100). ABC-SMC: distinct
        particles (>= 100). Nested sampling: stopped on ``dlogz``, effective sample size (>= 100).
        SNPE: final draw by rejection, without the MCMC fallback.

        Parameters
        ----------
        prior : Prior, optional
            The prior for the edge check. By default the prior recorded with the fit.
        likelihood_max_opt : LikelihoodMaxOptResult or float, optional
            The optimised likelihood maximum (:meth:`likelihood_max_opt`), or its ln L. Adds a
            check that the sampler's best draw is within 5 nats of it: beyond that the sampler
            never reached the peak.

        Returns
        -------
        DiagnosticsReport
            ``.passed``, ``.rows`` (check, value, threshold, passed, reason), ``.reasons`` (why each
            failed check failed); its ``repr`` is a table.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> from whisper_cbpf.models.flare import flare_flux
        >>> t = np.linspace(0.5, 30.0, 40)
        >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
        >>> report = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0).diagnostics()
        >>> report.passed
        True
        >>> print(report)                                     # doctest: +ELLIPSIS
        Diagnostics of the nested fit of 'flare': PASSED
        ...
        """
        from ..results import diagnose
        return diagnose(self._without_fixed(), prior=prior, likelihood_max_opt=likelihood_max_opt)

    def _without_fixed(self):
        """This result without its ``Fixed`` parameters' constant columns (``info["fixed"]``).

        A pinned parameter was not sampled: its R-hat and ESS are undefined (a constant column),
        so the convergence report skips it rather than computing them from zero variance.
        """
        fixed = [p for p in ((self.info or {}).get("fixed") or {}) if p in self.parameters]
        if not fixed:
            return self
        keep = [p for p in self.parameters if p not in fixed]
        view = dataclasses.replace(self, parameters=keep,
                                   samples=self.samples.drop(columns=fixed, errors="ignore"))
        chains = getattr(self, "samples_by_chain", None)
        if chains is not None:
            chains = np.asarray(chains)
            if chains.ndim == 3 and chains.shape[2] == len(self.parameters):
                chains = chains[..., [self.parameters.index(p) for p in keep]]
            view.samples_by_chain = chains
        return view

    def fitted_lc(self, lc):
        """The rows of ``lc`` this fit used: the pre-event rows it left out removed.

        The rule the fit applied is recorded in ``info["pre_event"]`` (see
        :func:`prepare_lc`); a result without that record (a result built by hand, or saved
        before 0.2.0) returns ``lc`` unchanged. Use it wherever the fit's own data points are
        needed: to plot the fitted rows apart from the excluded ones, or to score the posterior
        on the same data (:meth:`likelihood_max_opt` does this itself).

        Parameters
        ----------
        lc : LightCurve
            The light curve passed to the fit (with or without its pre-event rows).

        Returns
        -------
        LightCurve

        Examples
        --------
        Two non-detections before the explosion, six detections after it:

        >>> import numpy as np, whisper_cbpf as wp
        >>> t = np.array([-2.0, -0.5, 1.0, 3.0, 6.0, 10.0, 15.0, 20.0])
        >>> ul = t < 0
        >>> flux = np.where(ul, 0.3, wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t))
        >>> lc = wp.LightCurve(time=t + 60000.0, band=["r"] * 8, flux=flux,
        ...                    flux_err=np.where(ul, np.nan, 0.1), upper_limit=ul)
        >>> lc = lc.set_explosion_date(60000.0)
        >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=500, quantile=0.1)
        >>> res.info["excluded_pre_event"], res.n_data, len(res.fitted_lc(lc))
        (2, 6, 6)
        """
        info = self.info if isinstance(self.info, dict) else {}
        keep = _pre_event_mask(lc.time, info.get("pre_event"))
        return lc if bool(np.all(keep)) else lc[keep]

    def forecast(self, times, bands, **kwargs):
        """Predicted magnitudes at future epochs, from this posterior; see
        :func:`whisper_cbpf.forecast.forecast`.

        Delegates to ``whisper_cbpf.forecast.forecast(self, times, bands, **kwargs)``.

        Parameters
        ----------
        times : array_like
            Epochs, on the fitted light curve's clock.
        bands : str or array_like
            One band or several. Every combination of a time and a band is one cell.
        **kwargs
            Passed on (``lc``, ``model``, ``n_draws``, ``quantiles``, ``survey_depth``, ``seed``).

        Returns
        -------
        pandas.DataFrame
            One row per (time, band) cell, time-major: ``mean_mag``, ``sd_mag``, the quantiles and
            ``frac_too_faint``.

        Raises
        ------
        ImportError
            This installation has no ``whisper_cbpf.forecast`` module.

        Examples
        --------
        >>> import numpy as np, whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
        ...                                       "decay_time": 15.0}, t)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=2000, quantile=0.05)
        >>> table = res.forecast([35.0, 40.0], "r", n_draws=50)
        >>> len(table)
        2
        """
        try:
            module = importlib.import_module("..forecast", __package__)
        except ModuleNotFoundError as exc:
            if exc.name != "whisper_cbpf.forecast":
                raise
            raise ImportError(
                "result.forecast() needs the whisper_cbpf.forecast module, which this "
                "installation does not have. Install a whisper_cbpf release that includes it "
                "(0.2.0 or later).") from exc
        return module.forecast(self, times, bands, **kwargs)

    def facts(self, lc, **kwargs):
        """This fit's computed facts as a JSON-ready dict; see :func:`whisper_cbpf.result_facts`.

        Delegates to ``whisper_cbpf.facts.result_facts(self, lc, **kwargs)``: per parameter the
        median and 68 % interval, posterior width over prior width with the ``prior_dominated`` and
        ``at_prior_edge`` flags, what the data show per band, and the caveats (converged, not enough
        data, a large gain of the optimised likelihood maximum), each computed by a stated rule.

        Parameters
        ----------
        lc : LightCurve
            The data the fit was run on.
        **kwargs
            Passed on (``model``, ``prior``, ``thresholds``).

        Returns
        -------
        dict

        Examples
        --------
        >>> import numpy as np, whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
        ...                                       "decay_time": 15.0}, t)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=2000, quantile=0.05)
        >>> sorted(res.facts(lc)["parameters"])
        ['amplitude', 'decay_time', 'rise_time']
        """
        from ..facts import result_facts
        return result_facts(self, lc, **kwargs)

    def likelihood_max_opt(self, lc, model=None, **kwargs):
        """The likelihood peak behind this fit's best draw; see
        :func:`whisper_cbpf.likelihood_max_opt.likelihood_max_opt`.

        Delegates to ``whisper_cbpf.likelihood_max_opt.likelihood_max_opt(self, lc, model,
        **kwargs)`` and keeps the peak in ``info["likelihood_max_opt"]``
        (``LikelihoodMaxOptResult.to_dict()``), where the next optimisation starts from it and
        :meth:`save` writes it. The posterior draws, medians and error bars of this result are
        not changed. The rows the fit left out as pre-event data are left out here too
        (:meth:`fitted_lc`), and when the fit did not use the model's default prior (a prior passed
        to it, or an explosion-time prior the data set) the box is the prior its provenance
        records (pass ``prior=`` to choose another).

        Parameters
        ----------
        lc : LightCurve
            The data the fit was run on.
        model : str or Model, optional
            The model; by default the one named by ``self.model``.
        **kwargs
            Passed on (``n_candidates``, ``n_starts``, ``tol``, ``max_rounds``, ``space``,
            ``likelihood``, ``backend``, ``seed``, ``prior``).

        Returns
        -------
        LikelihoodMaxOptResult
            Peak parameters and log-likelihood, the gain over the sampler's best draw, parameters
            on a prior edge, and AIC/BIC at the peak.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> from whisper_cbpf.models.flare import flare_flux
        >>> t = np.linspace(0.5, 30.0, 40)
        >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
        >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
        >>> peak = res.likelihood_max_opt(lc)
        >>> peak.max_log_likelihood >= res.max_log_likelihood
        True
        >>> res.info["likelihood_max_opt"]["max_log_likelihood"] == peak.max_log_likelihood
        True
        """
        try:
            module = importlib.import_module("..likelihood_max_opt", __package__)
        except ModuleNotFoundError as exc:
            if exc.name != "whisper_cbpf.likelihood_max_opt":
                raise
            raise ImportError(
                "result.likelihood_max_opt() needs the whisper_cbpf.likelihood_max_opt module, "
                "which this installation does not have. Install a whisper_cbpf release that "
                "includes it (0.2.0 or later)."
            ) from exc
        if kwargs.get("prior") is None:
            kwargs.pop("prior", None)
            recorded = _recorded_prior(self)
            if recorded is not None:
                kwargs["prior"] = recorded
        peak = module.likelihood_max_opt(self, self.fitted_lc(lc), model, **kwargs)
        if isinstance(self.info, dict) and hasattr(peak, "to_dict"):
            self.info["likelihood_max_opt"] = peak.to_dict()
        return peak

    def __repr__(self):
        return (f"SamplerResult(sampler={self.sampler!r}, model={self.model!r}, "
                f"n_samples={self.n_samples}, AIC={self.aic:.1f}, runtime={self.runtime_s:.2f}s)")


def _plan_for_call(sampler, signature, bound):
    """:func:`_pre_event_plan` for one ``fit`` call, from its bound arguments (``None`` if the
    call carries no light curve)."""
    names = [p for p in signature.parameters if p != "self"]
    if len(names) < 2:
        return None
    lc = bound.arguments.get(names[0])
    if lc is None or not (hasattr(lc, "time") and hasattr(lc, "meta")):
        return None
    extra = next((bound.arguments.get(p.name) or {} for p in signature.parameters.values()
                  if p.kind is inspect.Parameter.VAR_KEYWORD), {})
    own = (bound.arguments.get("log_prob_fn") or extra.get("log_prob_fn")) is not None
    likelihood = bound.arguments.get("likelihood", extra.get("likelihood"))
    return _pre_event_plan(lc, bound.arguments.get(names[1]), bound.arguments.get("prior"),
                           sampler=getattr(sampler, "name", None), own_density=own,
                           likelihood=likelihood)


def _recording(fit):
    """Wrap a sampler's ``fit``: the pre-event rule before it, ``provenance`` after it.

    Before the fit: :func:`prepare_lc`'s rule. The pre-event rows are removed from the light curve
    the fit receives (so neither its likelihood nor its model evaluations see them), the
    explosion-time prior the data define is filled in for a model with a free ``t_exp``, and one
    warning says what was left out; afterwards ``info["excluded_pre_event"]``,
    ``info["pre_event"]`` and ``info["t_exp_prior"]`` record it. Every registered sampler reaches
    its ``fit`` through here, so the rule has one implementation.

    After the fit: the provenance, recorded from the arguments the fit was called with (the prior
    being the one it used; see :func:`whisper_cbpf.results.record_call`). A failure to record
    never fails the fit: the error is kept in ``provenance["error"]`` instead.
    """
    signature = inspect.signature(fit)

    @functools.wraps(fit)
    def fit_and_record(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError:
            bound = None                        # a bad call: the fit raises its own error for it
        plan = None if bound is None else _plan_for_call(self, signature, bound)
        if plan is None:
            result = fit(self, *args, **kwargs)
        else:
            plan.warn()
            call = signature.bind(self, *args, **kwargs)
            call.arguments[[p for p in signature.parameters if p != "self"][0]] = plan.lc
            if plan.prior_changed and "prior" in signature.parameters:
                call.arguments["prior"] = plan.prior
            result = fit(*call.args, **call.kwargs)
        wall = time.perf_counter() - start
        if isinstance(result, SamplerResult):
            if plan is not None:
                plan.stamp(result)
                if plan.prior_changed and "prior" in signature.parameters:
                    bound.arguments["prior"] = plan.prior
            try:
                from ..results import record_call
                result.provenance = record_call(
                    self, bound if bound is not None else signature.bind(self, *args, **kwargs),
                    wall)
            except Exception as exc:                          # noqa: BLE001 - see the docstring
                result.provenance = {"recorded_at": "fit", "wall_s": float(wall),
                                     "error": f"{type(exc).__name__}: {exc}"}
        return result

    fit_and_record._records_provenance = True
    return fit_and_record


class BaseSampler:
    """Contract for samplers: implement ``fit(lc, model, prior=None, **kwargs) -> SamplerResult``.

    Every subclass's ``fit`` records its call in ``result.provenance`` (the data's hash, the model
    and prior, the sampler settings and seed, versions, devices, wall time), so any result can be
    saved with :meth:`SamplerResult.save` without being told how it was made.
    """

    name = "base"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fit = cls.__dict__.get("fit")
        if fit is not None and not getattr(fit, "_records_provenance", False):
            cls.fit = _recording(fit)

    def fit(self, lc, model, prior=None, **kwargs):
        raise NotImplementedError
