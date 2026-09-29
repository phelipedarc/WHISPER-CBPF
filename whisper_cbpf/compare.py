"""Compare models on one light curve: fit each, find each likelihood maximum, rank them, weigh them,
grade the winner.

Every model is fitted to the same data with the same settings, its likelihood
peak is found by :func:`whisper_cbpf.likelihood_max_opt` (so a ranking does not carry sampler
noise), its convergence report is attached, and the models are ranked:

- by **ln Z** (the Bayesian evidence) when every ranked model has a trustworthy one: a nested-sampling
  run that stopped on ``dlogz``;
- by **BIC at the optimised likelihood maximum** otherwise, ``-2 ln L_max + k ln n``.

Weights are ``exp(-dBIC / 2)`` (or ``exp(-d ln Z)``), normalised over the ranked models. The grade is
Jeffreys' scale on the log Bayes factor ``ln B`` (``dBIC / 2`` stands in for it under BIC; Kass &
Raftery 1995, section 4.1.3): below ``ln 10^0.5 = 1.15`` "inconclusive", below ``ln 10 = 2.30``
"substantial", below ``ln 100 = 4.61`` "strong", above "decisive". In BIC terms the cuts are a
difference of 2.3, 4.6 and 9.2.

Only models fitted to the same points, scored in the same space (flux or magnitude), are compared (the
likelihoods of different data are not comparable), and a model with at least as many free parameters
as points is left out ("not enough
data": its BIC is undefined and its peak is not set by the data). Every model left out is listed with
its reason.

BIC only approximates the evidence, and a fitted redshift can hide a misfit that BIC does not charge
for. When the BIC gap between the top two is small, ``evidence_check="auto"`` runs nested sampling on
those two and reports whether ln Z agrees.
"""
from __future__ import annotations

import inspect
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Comparison", "compare"]

#: Jeffreys' (1961) scale on the log Bayes factor ``ln B``: the upper cut of each grade. Above the
#: last cut the grade is "decisive".
GRADE_CUTS = ((math.log(10 ** 0.5), "inconclusive"), (math.log(10.0), "substantial"),
              (math.log(100.0), "strong"))
#: ``evidence_check="auto"`` runs nested sampling on the top two when their ``ln B`` from BIC is
#: below this (``ln 10``: a grade below "strong", a BIC gap below 4.61).
EVIDENCE_CHECK_LN_B = math.log(10.0)
#: An optimised likelihood maximum more than this (ln L) above the sampler's best draw says the
#: sampler never reached the likelihood peak.
LIKELIHOOD_MAX_GAIN_FLAG = 1.0
#: The columns of :attr:`Comparison.table`, in order.
COLUMNS = ["model", "sampler", "n_params", "n_data", "max_log_likelihood", "aic", "bic",
           "log_evidence", "log_evidence_err", "delta", "weight", "grade", "converged",
           "problems", "status", "left_out_reason"]
#: Short family names, mapped onto :func:`whisper_cbpf.supernova_model` names.
FAMILY_ALIASES = {"magnetar": "basic_magnetar_powered",
                  "csm_shock_arnett": "csm_shock_and_arnett",
                  "shock_cooling_arnett": "shock_cooling_and_arnett"}
#: Names bound to :func:`whisper_cbpf.tde_model` (the Gaussian rise + cooling envelope).
TDE_NAMES = ("tde", "tde_gaussianrise")
#: Version of the layout :meth:`Comparison.save` writes.
FORMAT_VERSION = 1


# ================================================================================== the entry point
def compare(lc, models, sampler="auto", *, likelihood_max_opt=True, cache_dir=None,
            evidence_check="auto", n_jobs=None, prior=None, seed=0, **fit_kwargs):
    """Fit several models to one light curve and rank them.

    Every model is fitted to the same data with the same settings, its likelihood peak is found by
    :func:`whisper_cbpf.likelihood_max_opt` (AIC and BIC come from the peak, not from the
    sampler's best draw), and its convergence report is attached. The models are then ranked by
    ln Z when every ranked model has a trustworthy evidence (a nested-sampling run that stopped on
    ``dlogz``), and by BIC at the optimised likelihood maximum otherwise. See the module docstring
    for the weights and the grade.

    Parameters
    ----------
    lc : LightCurve
        The data. Every model sees exactly this light curve.
    models : sequence of str or Model
        The models to compare. Each is a :class:`~whisper_cbpf.Model`, a registered model name
        (:func:`~whisper_cbpf.list_models`), or a supernova or TDE family name that is bound to
        ``lc`` here: any of :func:`~whisper_cbpf.supernova_models`, the short names ``"magnetar"``,
        ``"csm_shock_arnett"``, ``"shock_cooling_arnett"``, and ``"tde"`` /
        ``"tde_gaussianrise"``. A bound family takes the light curve's bands; the redshift is fixed
        at ``lc.redshift`` when it is known and fitted from ``lc.redshift_prior`` otherwise; the
        explosion time is fixed at 0 when the time reference is the explosion, and otherwise fitted
        with the prior :meth:`~whisper_cbpf.LightCurve.explosion_time_prior` gives: from the last
        non-detection before the first detection to the first detection (the pre-event rule), or
        a fallback window when no non-detection precedes it (the comparison then says so).
    sampler : str, default "auto"
        ``"auto"``: ``emcee_jax`` on the GPU for a model with ``predict_jax`` when JAX sees a GPU,
        CPU ``mcmc`` otherwise. Any registered sampler name runs every model with that sampler;
        ``"nested"`` gives every model a ln Z, and the ranking is then by ln Z.
    likelihood_max_opt : bool, default True
        Find each fit's likelihood peak (:func:`whisper_cbpf.likelihood_max_opt`) and take AIC and
        BIC from it. ``False`` ranks on the samplers' best draws, which carries sampler noise.
    cache_dir : str or Path, optional
        Fit through :func:`whisper_cbpf.fit_cached`: a fit (and an evidence-check run) already
        saved there is loaded, not rerun, so an interrupted comparison resumes.
    evidence_check : {"auto", True, False}, default "auto"
        Nested sampling on the top two models under BIC: ``"auto"`` when their ``ln B`` is below
        ``EVIDENCE_CHECK_LN_B`` (``ln 10``, a BIC gap under 4.61), ``True`` always, ``False``
        never. When the two ln Z disagree with BIC the winner's grade becomes "inconclusive" and
        the comparison says why. When the two are the only ranked models, they are ranked by ln Z.
        It is CPU nested sampling (500 live points): about 8 and 12 minutes for
        ``shock_cooling_arnett`` and ``csm_shock_arnett`` on a 34-point LSST alert, serial. Pass
        ``n_jobs=`` to spread it over processes, ``cache_dir=`` to pay it once, or ``False`` to
        skip it (the summary then says the check did not run).
    n_jobs : int, optional
        Worker processes for the CPU samplers that take ``n_jobs`` (``mcmc``, ``nested``,
        ``abc``), including the evidence check. Default: each sampler's own.
    prior : Prior or dict, optional
        A :class:`~whisper_cbpf.Prior` for every model, or ``{model: Prior}``. For a bound family
        it overrides the named parameters of its default prior (e.g. ``{"t_exp": Uniform(lo, hi)}``),
        and an explosion-time prior given this way is used as given, not cut to the data window:
        this is how to widen the window when the non-detections are shallow. For a supernova
        family the ``t_exp`` prior needs a finite lower bound (a Uniform or a TruncatedNormal),
        since its diffusion epochs are sized from the earliest explosion it allows. For a Model
        or a registered name it is the prior the fit gets.
    seed : int, default 0
        Seed of every fit, maximum-likelihood optimisation and evidence run.
    **fit_kwargs
        Passed to every fit (``nwalkers=``, ``nsteps=``, ``burnin=``, ``space=``, ...).

    Returns
    -------
    Comparison
        ``.table`` (one row per model: model, sampler, n_params, n_data, max_log_likelihood,
        aic, bic, log_evidence, log_evidence_err, delta, weight, grade, converged, problems,
        status, left_out_reason; ranked models first, best first), ``.results``, ``.peaks``,
        ``.winner`` (``None`` when nothing could be ranked), ``.criterion`` (``"ln Z"`` or
        ``"BIC"``), ``.summary()``, ``.save(dir)``.

    Raises
    ------
    ValueError
        An unknown model name, a repeated model, a ``prior`` for a model not compared, an
        ``evidence_check`` or ``sampler`` value out of range, or a light curve with no detection.
    TypeError
        A setting ``**fit_kwargs`` that one of the samplers does not take (before any fit runs).
    Exception
        When every fit fails, the first fit's error is raised (there is nothing to compare). A
        single failed fit leaves that model out, with the error as its reason.

    Notes
    -----
    **The data each fit uses** are the fits' own rules, the same for every model: rows at or before
    the event are never fitted (the pre-event rule; a model with a free explosion time leaves out
    the non-detections before the first detection, ``result.info["excluded_pre_event"]``), and
    upper limits after it are fitted in flux space with the censored likelihood at
    ``lc.meta["upper_limit_sigma"]``. Only fits on the same rows, scored in the same space, are
    ranked against each other: the most common number of points first, then the most common set
    of rows and space among those; the table says which models were left out and why.
    ``lc.where(upper_limit=False)`` compares on the detections alone.

    **A fitted redshift** can bias the comparison: a model can move the source to buy a fit it
    cannot make at the true distance, and BIC charges every model the same for that freedom. The
    summary says so whenever a ranked model fits the redshift.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1),
    ...                    name="toy")
    >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
    ...                  evidence_check=False)
    >>> cmp.winner, cmp.criterion
    ('flare', 'BIC')
    >>> list(cmp.table["model"])
    ['flare', 'bazin']
    >>> bool(cmp.table["weight"].sum() > 0.999)
    True
    """
    if evidence_check not in ("auto", True, False):
        raise ValueError(f"evidence_check must be 'auto', True or False; got {evidence_check!r}.")
    if isinstance(models, (str, bytes)) or not hasattr(models, "__iter__"):
        raise TypeError(f"models must be a list of model names or Model objects, e.g. "
                        f"['arnett', 'magnetar']; got {type(models).__name__}.")
    labels_in = list(models)
    if not labels_in:
        raise ValueError("models is empty: name at least one model to fit.")
    priors = _priors_by_label(prior, labels_in)
    entries = _resolve_models(lc, labels_in, priors)
    samplers = {lab: _choose_sampler(sampler, e["model"]) for lab, e in entries.items()}
    _check_fit_kwargs(set(samplers.values()), fit_kwargs)

    records = {}
    for lab, e in entries.items():
        records[lab] = _fit_one(lab, e, samplers[lab], lc, cache_dir, likelihood_max_opt, n_jobs,
                                seed, fit_kwargs)
    failed = [r for r in records.values() if r["error"] is not None]
    if len(failed) == len(records):
        raise failed[0]["exception"]

    ranking = _rank([_rank_row(r) for r in records.values()])
    check = {"requested": evidence_check, "ran": False, "models": [], "agrees": None,
             "reason": _no_check_reason(evidence_check, ranking)}
    if _should_check(evidence_check, ranking):
        check = _evidence_check(records, ranking, lc, cache_dir, n_jobs, seed, fit_kwargs,
                                evidence_check)
        ranking = _rank([_rank_row(r) for r in records.values()])

    return _build(records, ranking, check, lc, likelihood_max_opt, {
        "sampler": sampler, "likelihood_max_opt": bool(likelihood_max_opt),
        "evidence_check": evidence_check,
        "seed": int(seed), "n_jobs": n_jobs, "cache_dir": None if cache_dir is None else
        str(cache_dir), "fit_kwargs": {k: _plain(v) for k, v in sorted(fit_kwargs.items())}})


# ============================================================================= resolving the inputs
def _priors_by_label(prior, labels):
    from .priors import Prior

    names = [_label(m) for m in labels]
    if prior is None:
        return {}
    if isinstance(prior, Prior):
        return {nm: prior for nm in names}
    if isinstance(prior, dict):
        stray = [k for k in prior if k not in names]
        if stray:
            raise ValueError(f"prior= names {stray}, which are not among the models compared "
                             f"({names}). Key the dict by model name.")
        bad = [k for k, v in prior.items() if v is not None and not isinstance(v, Prior)]
        if bad:
            raise TypeError(f"prior= must map a model name to a Prior; the value for {bad[0]!r} "
                            f"is a {type(prior[bad[0]]).__name__}. Wrap its distributions in "
                            f"Prior({{...}}).")
        return {k: v for k, v in prior.items() if v is not None}
    raise TypeError(f"prior= must be a Prior (for every model) or a dict {{model: Prior}}; got "
                    f"{type(prior).__name__}.")


def _label(model):
    from .models import Model

    if isinstance(model, Model):
        return model.name
    if isinstance(model, str):
        return model
    raise TypeError(f"a model is a name or a Model object; got {type(model).__name__}.")


def _resolve_models(lc, labels, priors):
    """``{label: {"model": Model, "prior": Prior or None (what fit() gets), "bound": bool}}``."""
    from .models import Model, list_models
    from .models import _REGISTRY as registry

    out = {}
    for m in labels:
        lab = _label(m)
        if lab in out:
            raise ValueError(f"model {lab!r} is listed twice. Give each model once (two variants "
                             f"of one model need different names).")
        pr = priors.get(lab)
        if isinstance(m, Model):
            out[lab] = {"model": m, "prior": pr, "bound": False}
        elif m in registry:
            out[lab] = {"model": registry[m], "prior": pr, "bound": False}
        else:
            bound = _bind_family(m, lc, pr)
            if bound is None:
                raise ValueError(
                    f"unknown model {m!r}. Name a registered model ({list_models()}), a "
                    f"supernova family ({_family_names()}), 'tde', or pass a Model object "
                    f"(e.g. wp.supernova_model(...))." + _no_jax_note())
            # An explosion-time prior the caller gave is passed to the fit as given: left to the
            # model's default prior, the fit would cut it back to the data window.
            given_t = pr is not None and "t_exp" in pr.distributions and "t_exp" in bound.parameters
            out[lab] = {"model": bound, "prior": bound.default_prior if given_t else None,
                        "bound": True}
            if "t_exp" in bound.parameters and not given_t and _no_pre_detection_limit(lc):
                out[lab]["problems"] = [
                    f"no non-detection precedes the first detection, so the explosion-time prior "
                    f"is a fallback window, {bound.default_prior.distributions['t_exp']!r}: pass "
                    f"prior={{{lab!r}: Prior({{'t_exp': ...}})}} to set it"]
    return out


def _no_pre_detection_limit(lc):
    from .io.schema import explosion_window
    return explosion_window(lc.time, lc.upper_limit)["last_non_detection"] is None


def _no_jax_note():
    """Why a supernova or TDE family name is unknown when JAX is not installed, else ``""``."""
    import importlib.util

    if importlib.util.find_spec("jax") is not None:
        return ""
    return (" JAX is not installed, and the supernova and TDE families ('arnett', 'magnetar', "
            "'tde', ...) are JAX models: install JAX (pip install jax on a CPU, or the [gpu] "
            "extra on an NVIDIA GPU; see INSTALL.md).")


def _family_names():
    try:
        from .models.jax import supernova_models
        return sorted(set(supernova_models()) | set(FAMILY_ALIASES))
    except Exception:                                         # noqa: BLE001 - listing only
        return sorted(FAMILY_ALIASES)


def _bind_family(name, lc, prior):
    """A supernova or TDE family bound to ``lc``'s bands, redshift and explosion-time window."""
    family = FAMILY_ALIASES.get(name, name)
    is_tde = name in TDE_NAMES
    if not is_tde:
        try:
            from .models.jax import supernova_models
            known = set(supernova_models())
        except ImportError:
            known = set()
        if family not in known:
            return None
    from .models.cosmology import luminosity_distance_cm
    from .models.jax import supernova_model, tde_model
    from .priors import Prior

    from .io.schema import explosion_window

    bands = list(dict.fromkeys(str(b) for b in np.asarray(lc.band)))
    over = dict(prior.distributions) if prior is not None else {}
    free, kw = [], {}
    if str(lc.meta.get("time_reference") or "").lower() == "explosion":
        kw["t_exp_days"] = 0.0
    else:
        free.append("t_exp")
        if "t_exp" not in over:                   # the pre-event rule's window (or its fallback)
            over["t_exp"] = lc.explosion_time_prior()
        # The host float64 reference the epochs are shifted by: the first detection, near the
        # data whatever the prior (a Normal prior has no upper bound to take).
        kw["t_exp_days"] = float(explosion_window(lc.time, lc.upper_limit)["first_detection"])
    if lc.redshift_known:
        z = float(lc.redshift)
        kw.update(redshift=z, dl_cm=float(luminosity_distance_cm(z)))
    else:
        free.append("redshift")
        kw["redshift_prior"] = lc.redshift_prior
    kw["prior"] = Prior(over) if over else None
    if free:
        kw["free"] = free
    if is_tde:
        return tde_model(bands, name=name, rise="gaussian", **kw)
    if free:        # fixed diffusion epochs: size them to the latest epoch (else they stop at 200 d)
        kw["times"] = np.asarray(lc.time, dtype=float)
    return supernova_model(family, bands, name=name, **kw)


def _rebind_family(name, lc, result):
    """A family :func:`compare` bound by name, bound again to the saved light curve with the prior
    its fit recorded (so the explosion-time reference is the same); ``None`` if ``name`` is not a
    family, the JAX extra is missing, or the parameters differ from the fit's."""
    from .samplers.base import _prior_from_record

    prov = getattr(result, "provenance", None) or {}
    prior = _prior_from_record(((prov.get("model") or {}) if isinstance(prov, dict)
                                else {}).get("prior"))
    try:
        model = _bind_family(name, lc, prior)
    except Exception:                                         # noqa: BLE001 - keep the name
        return None
    if model is None or list(model.parameters) != [p for p in result.parameters
                                                    if p in model.parameters] \
            or not set(model.parameters) <= set(result.parameters):
        return None
    return model


def _gpu_visible():
    from .samplers import _gpu_visible as visible
    return visible()


def _choose_sampler(sampler, model):
    from .samplers import _auto_sampler, list_samplers

    if sampler == "auto":                     # the same rule as fit(sampler="auto")
        return _auto_sampler(model, _gpu_visible)
    if not isinstance(sampler, str) or sampler not in list_samplers():
        raise ValueError(f"sampler must be 'auto' or one of {list_samplers()}; got {sampler!r}.")
    return sampler


def _fit_signature(name):
    from .samplers import get_sampler
    return inspect.signature(type(get_sampler(name)).fit)


def _check_fit_kwargs(samplers, fit_kwargs):
    """Refuse a setting a sampler does not take, before anything is fitted."""
    for s in sorted(samplers):
        params = _fit_signature(s).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue                            # checked by the sampler itself
        stray = [k for k in fit_kwargs if k not in params]
        if stray:
            raise TypeError(f"sampler {s!r} does not take {stray}. Its settings are listed in "
                            f"help(whisper_cbpf.samplers.get_sampler({s!r}).fit).")


def _takes(sampler, name):
    return name in _fit_signature(sampler).parameters


# ===================================================================================== one model
def _fit_one(label, entry, sampler, lc, cache_dir, do_opt, n_jobs, seed, fit_kwargs):
    rec = {"label": label, "model": entry["model"], "fit_prior": entry["prior"],
           "sampler": sampler, "result": None, "peak": None, "diagnostics": None,
           "evidence": None, "problems": list(entry.get("problems", [])), "error": None,
           "exception": None,
           "timing": {"fit_s": None, "likelihood_max_opt_s": None, "evidence_s": None}}
    kw = dict(fit_kwargs, seed=seed)
    if entry["prior"] is not None:
        kw["prior"] = entry["prior"]
    if n_jobs is not None and _takes(sampler, "n_jobs"):
        kw["n_jobs"] = n_jobs
    t0 = time.perf_counter()
    try:
        rec["result"] = _run(lc, entry["model"], sampler, cache_dir, kw)
    except Exception as exc:                                  # noqa: BLE001 - reported per model
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["exception"] = exc
        warnings.warn(f"compare: the {sampler} fit of {label!r} failed and the model is left out "
                      f"({rec['error']}).", stacklevel=3)
        return rec
    finally:
        rec["timing"]["fit_s"] = time.perf_counter() - t0
    res = rec["result"]
    rec["data_key"] = _data_key(res, lc)
    if res.n_samples == 0:
        return rec
    prior_used = _prior_used(entry, res)
    if do_opt:
        t0 = time.perf_counter()
        rec["peak"], why = _likelihood_max_opt(res, lc, entry["model"], prior_used, seed)
        rec["timing"]["likelihood_max_opt_s"] = time.perf_counter() - t0
        if rec["peak"] is None:
            rec["problems"].append(f"likelihood maximum not found ({why}); AIC and BIC are the "
                                   f"sampler's")
        else:
            res.info["likelihood_max_opt"] = rec["peak"].to_dict()
            if rec["peak"].gain > LIKELIHOOD_MAX_GAIN_FLAG:
                rec["problems"].append(
                    f"the likelihood maximum is {rec['peak'].gain:.2f} ln L above the sampler's "
                    f"best draw: the sampler never reached the peak (its own BIC was off by "
                    f"{2 * rec['peak'].gain:.1f})")
            if rec["peak"].at_edge:
                rec["problems"].append(
                    f"peak on a prior edge: {', '.join(rec['peak'].at_edge)} (widen that prior, "
                    f"or quote a bound)")
    try:
        rec["diagnostics"] = res.diagnostics(prior=prior_used, likelihood_max_opt=rec["peak"])
        rec["problems"] = list(rec["diagnostics"].reasons) + rec["problems"]
    except Exception as exc:                                  # noqa: BLE001 - reported, not fatal
        rec["problems"].append(f"diagnostics could not run ({type(exc).__name__}: {exc})")
    return rec


def _data_key(result, lc):
    """What a fit's ln L is a density of: the rows it used and the space it scored them in. Two
    fits are ranked against each other only when these agree (``None``: unknown, not checked)."""
    try:
        from .results import data_hash
        rows = result.fitted_lc(lc) if hasattr(result, "fitted_lc") else lc
        return f"{(result.info or {}).get('space')}:{data_hash(rows)}"
    except Exception:                                         # noqa: BLE001 - not checked, not wrong
        return None


def _run(lc, model, sampler, cache_dir, kw):
    if cache_dir is not None:
        from .results import fit_cached
        return fit_cached(lc, model, sampler, cache_dir, **kw)
    from .samplers import fit
    return fit(lc, model, sampler=sampler, **kw)


def _prior_used(entry, result):
    """The prior the fit ran under: the one given (or the model's), with the explosion-time prior
    the pre-event rule set (``info["t_exp_prior"]``, when it cut a Uniform to the data window)."""
    from .priors import Prior, Uniform

    prior = entry["prior"] if entry["prior"] is not None else entry["model"].default_prior
    rec = (result.info or {}).get("t_exp_prior")
    if prior is None or not isinstance(rec, dict) or rec.get("type") != "Uniform" \
            or "t_exp" not in prior.distributions:
        return prior
    old = prior.distributions["t_exp"]
    if tuple(getattr(old, "bounds", ())) == (rec["low"], rec["high"]):
        return prior
    return Prior({**prior.distributions, "t_exp": Uniform(rec["low"], rec["high"], name="t_exp")})


def _likelihood_max_opt(result, lc, model, prior, seed):
    from .likelihood_max_opt import likelihood_max_opt

    rows = result.fitted_lc(lc) if hasattr(result, "fitted_lc") else lc   # the fit's own rows
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="not enough data")   # the table says so
            return likelihood_max_opt(result, rows, model, prior=prior, seed=seed), None
    except Exception as exc:                                  # noqa: BLE001 - reported per model
        return None, f"{type(exc).__name__}: {exc}"


# ======================================================================================== ranking
def _rank_row(rec):
    res, peak = rec["result"], rec["peak"]
    row = {"label": rec["label"], "error": rec["error"], "n_samples": 0, "n_params": None,
           "n_data": None, "max_ll": float("nan"), "aic": float("nan"), "bic": float("nan"),
           "lnz": float("nan"), "lnz_err": float("nan"), "lnz_ok": False}
    if res is None:
        return row
    row.update(n_samples=res.n_samples, n_params=int(res.n_params), n_data=int(res.n_data),
               excluded=int((res.info or {}).get("excluded_pre_event") or 0),
               data_key=rec.get("data_key"), space=(res.info or {}).get("space"))
    if peak is not None:
        row.update(max_ll=peak.max_log_likelihood, aic=peak.aic, bic=peak.bic)
    else:
        row.update(max_ll=float(res.max_log_likelihood), aic=float(res.aic), bic=float(res.bic))
    nested = res if res.sampler == "nested" else rec["evidence"]
    if nested is not None:
        lnz = float(nested.info.get("log_evidence", float("nan")))
        row.update(lnz=lnz, lnz_err=float(nested.info.get("log_evidence_err", float("nan"))),
                   lnz_ok=bool(nested.info.get("converged")) and math.isfinite(lnz)
                   and int(nested.n_data) == int(res.n_data))
    return row


def _grade(ln_b):
    """Jeffreys' grade of a log Bayes factor (``None`` when there is none)."""
    if ln_b is None or not math.isfinite(ln_b):
        return None
    for cut, label in GRADE_CUTS:
        if ln_b < cut:
            return label
    return "decisive"


def _rank(rows):
    """Rank plain rows (see :func:`_rank_row`). Pure: no fitting, no I/O."""
    left, usable = {}, []
    for r in rows:
        if r["error"] is not None:
            left[r["label"]] = f"the fit failed: {r['error']}"
        elif not r["n_samples"]:
            left[r["label"]] = ("no posterior draws (an ABC fit that accepted none): nothing to "
                                "rank")
        else:
            usable.append(r)
    counts = {}
    for r in usable:
        counts[r["n_data"]] = counts.get(r["n_data"], 0) + 1
    ref_n = max(counts, key=counts.get) if counts else None   # first most common, in model order
    keys = {}                        # within that n: the rows used and the space they were scored in
    for r in usable:
        if r["n_data"] == ref_n and r.get("data_key") is not None:
            keys[r["data_key"]] = keys.get(r["data_key"], 0) + 1
    ref_key = max(keys, key=keys.get) if keys else None
    ref_space = next((r.get("space") for r in usable if r.get("data_key") == ref_key), None)
    ranked = []
    for r in usable:
        k, n = r["n_params"], r["n_data"]
        if n != ref_n:
            pre = (f" ({r['excluded']} pre-event row(s) left out by its fit)"
                   if r.get("excluded") else "")
            left[r["label"]] = (f"fitted to n = {n} points{pre}, the others to n = {ref_n}: "
                                f"likelihoods of different data cannot be compared")
        elif ref_key is not None and r.get("data_key") not in (None, ref_key):
            if r.get("space") != ref_space:
                left[r["label"]] = (f"fitted in {r.get('space')} space, the others in "
                                    f"{ref_space} space (n = {n} for both): likelihoods in "
                                    f"different spaces cannot be compared. Pass space= to "
                                    f"compare() so every fit uses the same one")
            else:
                left[r["label"]] = (f"fitted to other rows of the light curve than the others "
                                    f"(n = {n} for both; its pre-event rule differs, see its "
                                    f"info['pre_event']): likelihoods of different data cannot "
                                    f"be compared")
        elif k >= n:
            left[r["label"]] = (f"not enough data: k = {k} free parameters >= n = {n} points, so "
                                f"BIC is undefined and the peak is not set by the data. Wait for "
                                f"more points or fit fewer parameters")
        else:
            ranked.append(r)
    use_lnz = bool(ranked) and all(r["lnz_ok"] for r in ranked)
    if use_lnz:
        value = {r["label"]: r["lnz"] for r in ranked}
    else:
        value = {}
        for r in ranked:
            if math.isfinite(r["bic"]):
                value[r["label"]] = r["bic"]
            else:
                left[r["label"]] = ("no finite BIC: the likelihood was not finite at any draw "
                                    "(a model dark at every epoch, or outside its constraints)")
    sign = -1.0 if use_lnz else 1.0
    order = sorted(value, key=lambda lab: sign * value[lab])            # stable: ties keep order
    delta, weight, grade = {}, {}, {}
    if order:
        best = value[order[0]]
        delta = {lab: (best - value[lab]) if use_lnz else (value[lab] - best) for lab in order}
        ln_b = {lab: (delta[lab] if use_lnz else delta[lab] / 2.0) for lab in order}
        top = max(-ln_b[lab] for lab in order)
        w = {lab: math.exp(-ln_b[lab] - top) for lab in order}
        total = sum(w.values())
        weight = {lab: w[lab] / total for lab in order}
        grade = {lab: _grade(ln_b[lab]) for lab in order[1:]}
        grade[order[0]] = _grade(ln_b[order[1]]) if len(order) > 1 else None
    return {"criterion": "ln Z" if use_lnz else "BIC", "order": order, "ref_n": ref_n,
            "delta": delta, "weight": weight, "grade": grade, "left_out": left}


# ================================================================================ evidence check
def _no_check_reason(evidence_check, ranking):
    if evidence_check is False:
        return "not requested (evidence_check=False)"
    if ranking["criterion"] == "ln Z":
        return "not needed: every ranked model has a nested-sampling ln Z"
    if len(ranking["order"]) < 2:
        return "fewer than two ranked models"
    ln_b = ranking["delta"][ranking["order"][1]] / 2.0
    return (f"not needed: ln B = {ln_b:.2f} from BIC between the top two is at least "
            f"{EVIDENCE_CHECK_LN_B:.2f} (a grade of strong or more)")


def _should_check(evidence_check, ranking):
    if evidence_check is False or ranking["criterion"] == "ln Z" or len(ranking["order"]) < 2:
        return False
    if evidence_check is True:
        return True
    return ranking["delta"][ranking["order"][1]] / 2.0 < EVIDENCE_CHECK_LN_B


def _evidence_check(records, ranking, lc, cache_dir, n_jobs, seed, fit_kwargs, requested):
    top = ranking["order"][:2]
    kw = {k: fit_kwargs[k] for k in ("space", "likelihood") if k in fit_kwargs}
    kw["seed"] = seed
    if n_jobs is not None:
        kw["n_jobs"] = n_jobs
    for lab in top:
        rec = records[lab]
        if rec["result"].sampler == "nested":
            continue
        if rec["fit_prior"] is not None:
            kw_m = dict(kw, prior=rec["fit_prior"])
        else:
            kw_m = kw
        t0 = time.perf_counter()
        try:
            rec["evidence"] = _run(lc, rec["model"], "nested", cache_dir, kw_m)
        except Exception as exc:                              # noqa: BLE001 - reported
            rec["problems"].append(f"evidence check failed ({type(exc).__name__}: {exc})")
        rec["timing"]["evidence_s"] = time.perf_counter() - t0
    rows = {lab: _rank_row(records[lab]) for lab in top}
    ok = all(rows[lab]["lnz_ok"] for lab in top)
    ln_b_bic = ranking["delta"][top[1]] / 2.0
    why = ("requested (evidence_check=True)" if requested is True else
           f"ln B = {ln_b_bic:.2f} from BIC between the top two is below "
           f"{EVIDENCE_CHECK_LN_B:.2f}: BIC only approximates the evidence, too coarsely to "
           f"settle a gap this small" + _redshift_note(records, top))
    out = {"requested": requested, "ran": True, "models": list(top), "reason": why,
           "agrees": None, "ln_b": None, "ln_b_err": None}
    if ok:
        a, b = rows[top[0]], rows[top[1]]
        out.update(agrees=bool(a["lnz"] >= b["lnz"]), ln_b=a["lnz"] - b["lnz"],
                   ln_b_err=math.hypot(a["lnz_err"], b["lnz_err"]))
    else:
        bad = [lab for lab in top if not rows[lab]["lnz_ok"]]
        out["reason"] += (f"; inconclusive: no trustworthy ln Z for {bad} (nested sampling did "
                          f"not stop on dlogz, or failed)")
    return out


def _redshift_note(records, labels):
    """``"; a fitted redshift ..."`` when one of ``labels`` fits the redshift, else ``""``."""
    if any(records[lab]["result"] is not None and "redshift" in records[lab]["result"].parameters
           for lab in labels):
        return "; a fitted redshift can also hide a misfit that BIC does not charge for"
    return ""


def _prior_volume_note(check, rows):
    """A caveat when the evidence check's ln Z gap between the top two is not the gap between their
    likelihood peaks: the rest comes from the priors, which ln Z weighs and BIC does not."""
    if not check.get("ran") or check.get("ln_b") is None:
        return ""
    a, b = check["models"]
    d_ll = rows[a]["max_ll"] - rows[b]["max_ll"]
    ln_b = check["ln_b"]
    if not (math.isfinite(d_ll) and abs(ln_b - d_ll) > EVIDENCE_CHECK_LN_B):
        return ""
    return (f"ln Z({a!r}) - ln Z({b!r}) = {ln_b:.2f}, while their likelihood peaks differ by "
            f"{d_ll:.2f} ln L: the rest of the gap comes from the priors (the prior volume each "
            f"model spends away from the data), so it is only as meaningful as the two priors. "
            f"Rank by ln Z only between priors you would defend before seeing the data.")


# ===================================================================================== the result
def _build(records, ranking, check, lc, optimised, settings):
    order = ranking["order"]
    rows_by = {r["label"]: r for r in (_rank_row(rec) for rec in records.values())}
    labels = order + [lab for lab in records if lab not in order]
    grade = dict(ranking["grade"])
    problems = []
    if check["ran"] and check["agrees"] is False and ranking["criterion"] == "BIC":
        a, b = check["models"]
        grade[a] = "inconclusive"
        problems.append(
            f"the evidence check disagrees with BIC: nested sampling prefers {b!r} over {a!r} by "
            f"ln B = {-check['ln_b']:.2f} +/- {check['ln_b_err']:.2f}. Treat the top two as "
            f"undecided" + _redshift_note(records, [a, b]) + ".")
    note = _prior_volume_note(check, rows_by)
    if note:
        problems.append(note)
    table_rows = []
    for lab in labels:
        rec, row = records[lab], rows_by[lab]
        res, diag = rec["result"], rec["diagnostics"]
        ranked = lab in order
        table_rows.append({
            "model": lab, "sampler": rec["sampler"] if res is None else res.sampler,
            "n_params": row["n_params"] if row["n_params"] is not None else
            len(rec["model"].parameters),
            "n_data": row["n_data"], "max_log_likelihood": row["max_ll"], "aic": row["aic"],
            "bic": row["bic"], "log_evidence": row["lnz"], "log_evidence_err": row["lnz_err"],
            "delta": ranking["delta"].get(lab, float("nan")),
            "weight": ranking["weight"].get(lab, float("nan")),
            "grade": grade.get(lab) if ranked else None,
            "converged": None if diag is None else bool(diag.passed),
            "problems": list(rec["problems"]),
            "status": "ranked" if ranked else "left out",
            "left_out_reason": None if ranked else ranking["left_out"].get(lab)})
    table = _table(table_rows)
    winner = order[0] if order else None
    if winner is None:
        problems.append("no model could be ranked: " + "; ".join(
            f"{lab}: {why}" for lab, why in ranking["left_out"].items()))
    elif len(order) == 1:
        problems.append(f"only {winner!r} could be ranked, so there is nothing to weigh it "
                        f"against; the others were left out (see left_out_reason).")
    if winner is not None and records[winner]["diagnostics"] is not None \
            and not records[winner]["diagnostics"].passed:
        peak = ("Its likelihood maximum and BIC stand" if records[winner]["peak"] is not None else
                "Its BIC is the sampler's best draw's")
        problems.append(f"the winner's fit ({winner!r}) did not pass its convergence report "
                        f"(its failed checks are listed below). {peak}; do not quote its error "
                        f"bars from this run.")
    if not optimised:
        problems.append("BIC from the samplers' best draws (likelihood_max_opt=False): the ranking "
                        "carries sampler noise.")
    elif any(records[lab]["peak"] is None for lab in order):
        miss = [lab for lab in order if records[lab]["peak"] is None]
        problems.append(f"no likelihood maximum was found for {miss}, so their BIC is the "
                        f"sampler's and the ranking mixes optimised maxima with samplers' best "
                        f"draws.")
    if any("redshift" in records[lab]["result"].parameters for lab in order):
        problems.append("a ranked model fits the redshift: a model can move the source to buy a "
                        "fit it cannot make at the true distance, and BIC charges every model the "
                        "same for that freedom. Read each redshift posterior against its prior "
                        "before trusting the ranking.")
    return Comparison(
        table=table, results={lab: rec["result"] for lab, rec in records.items()
                              if rec["result"] is not None},
        peaks={lab: rec["peak"] for lab, rec in records.items() if rec["peak"] is not None},
        winner=winner, criterion=ranking["criterion"], lc=lc,
        models={lab: rec["model"] for lab, rec in records.items()},
        diagnostics={lab: rec["diagnostics"] for lab, rec in records.items()
                     if rec["diagnostics"] is not None},
        evidence={lab: rec["evidence"] for lab, rec in records.items()
                  if rec["evidence"] is not None},
        evidence_check=check, n_data=ranking["ref_n"], problems=problems,
        timing={lab: dict(rec["timing"]) for lab, rec in records.items()},
        settings=settings)


def _table(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ("n_params", "n_data"):
        df[col] = df[col].astype("Int64")
    for col in ("max_log_likelihood", "aic", "bic", "log_evidence", "log_evidence_err", "delta",
                "weight"):
        df[col] = df[col].astype(float)
    for col in ("grade", "converged", "left_out_reason"):
        # pandas >= 3 stores a string column with NaN for its gaps: keep None, as documented.
        df[col] = pd.Series([None if v is None or v is pd.NA or (isinstance(v, float) and
                                                                math.isnan(v)) else v
                             for v in df[col]], index=df.index, dtype=object)
    return df


def _plain(v):
    """A JSON-able stand-in for a setting (a Prior or a callable becomes its repr)."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return repr(v)


def _num(x, fmt):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


class Comparison:
    """The ranked result of :func:`compare`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per model (columns ``COLUMNS``): the ranked models first, best first, then the
        models left out, each with ``left_out_reason``. ``max_log_likelihood``, ``aic`` and ``bic``
        are those of the optimised likelihood maximum; ``delta`` is ``BIC - BIC_best`` (or
        ``ln Z_best - ln Z``); ``weight`` sums to 1 over the ranked models; ``grade`` is Jeffreys' on ``ln B`` of the
        winner against that model (for the winner: against the runner-up); ``converged`` is the
        fit's convergence report; ``problems`` lists every failed check and every flag raised by
        the maximum-likelihood optimisation.
    results : dict
        ``{model: SamplerResult}``.
    peaks : dict
        ``{model: LikelihoodMaxOptResult}`` of the models whose likelihood maximum was found.
    winner : str or None
        The best-ranked model; ``None`` when no model could be ranked.
    criterion : {"ln Z", "BIC"}
        What ranked the models.
    lc : LightCurve
        The data every model was fitted to.
    models : dict
        ``{model: Model}``.
    diagnostics : dict
        ``{model: DiagnosticsReport}``.
    evidence : dict
        ``{model: SamplerResult}`` of the evidence check's nested-sampling runs.
    evidence_check : dict
        Whether the check ran, on which models, why, and whether ln Z agrees with BIC.
    n_data : int
        The number of points the ranked models were fitted to.
    problems : list of str
        Caveats about the comparison as a whole.
    timing : dict
        ``{model: {"fit_s", "likelihood_max_opt_s", "evidence_s"}}`` wall seconds (compilation
        included).
    settings : dict
        The call's settings.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> flux = wp.get_model("flare").predict(
    ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
    >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500, evidence_check=False)
    >>> cmp                                                  # doctest: +ELLIPSIS
    Comparison of 2 models on 30 points by BIC: winner 'flare' (...)
    ...
    """

    def __init__(self, *, table, results, peaks, winner, criterion, lc, models, diagnostics,
                 evidence, evidence_check, n_data, problems, timing, settings):
        self.table = table
        self.results = results
        self.peaks = peaks
        self.winner = winner
        self.criterion = criterion
        self.lc = lc
        self.models = models
        self.diagnostics = diagnostics
        self.evidence = evidence
        self.evidence_check = evidence_check
        self.n_data = n_data
        self.problems = problems
        self.timing = timing
        self.settings = settings

    # ------------------------------------------------------------------------------- reading
    def _ranked(self):
        return list(self.table.loc[self.table["status"] == "ranked", "model"])

    def _headline(self):
        n = len(self.table)
        if self.winner is None:
            return f"Comparison of {n} models: not enough data -- no model could be ranked"
        ranked = self._ranked()
        row = self.table.iloc[0]
        if len(ranked) > 1:
            runner = ranked[1]
            d = float(self.table.loc[self.table["model"] == runner, "delta"].iloc[0])
            ln_b = d if self.criterion == "ln Z" else d / 2.0
            how = f"{row['grade']} over {runner!r}, ln B = {ln_b:.2f}"
        else:
            how = "the only model ranked"
        return (f"Comparison of {n} models on {self.n_data} points by {self.criterion}: winner "
                f"{self.winner!r} ({how})")

    def summary(self):
        """The ranking as text: the winner and its grade, the table, models left out, caveats.

        Returns
        -------
        str

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
        ...                  evidence_check=False)
        >>> print(cmp.summary())                             # doctest: +ELLIPSIS
        Comparison of 2 models on 30 points by BIC: winner 'flare' (...)
        <BLANKLINE>
          rank  model  sampler  k  n   max ln L ...
        """
        lines = [self._headline(), ""]
        crit = "d ln Z" if self.criterion == "ln Z" else "dBIC"
        head = ["rank", "model", "sampler", "k", "n", "max ln L", "BIC", "ln Z", crit, "weight",
                "grade", "converged"]
        body = []
        for i, r in enumerate(self.table.itertuples(index=False)):
            ranked = r.status == "ranked"
            lnz = (f"{r.log_evidence:.2f}+/-{r.log_evidence_err:.2f}"
                   if math.isfinite(r.log_evidence) else "-")
            body.append([str(i + 1) if ranked else "-", r.model, r.sampler,
                         _num(None if pd.isna(r.n_params) else int(r.n_params), "d"),
                         _num(None if pd.isna(r.n_data) else int(r.n_data), "d"),
                         _num(r.max_log_likelihood, ".2f"), _num(r.bic, ".2f"), lnz,
                         _num(r.delta, ".2f"), _num(r.weight, ".3f"), r.grade or "-",
                         {True: "yes", False: "no", None: "-"}[r.converged]])
        widths = [max(len(x) for x in col) for col in zip(head, *body)]
        for row in [head] + body:
            lines.append("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        out = self.table[self.table["status"] != "ranked"]
        if len(out):
            lines += ["", "Left out:"]
            lines += [f"  - {r.model}: {r.left_out_reason}" for r in out.itertuples(index=False)]
        ec = self.evidence_check
        if ec.get("ran"):
            verdict = {True: "agrees with BIC", False: "DISAGREES with BIC",
                       None: "inconclusive"}[ec.get("agrees")]
            if ec.get("ln_b") is not None:
                a, b = ec["models"]
                verdict += (f": ln Z({a}) - ln Z({b}) = {ec['ln_b']:.2f} +/- "
                            f"{ec['ln_b_err']:.2f}")
            lines += ["", f"Evidence check on {ec['models']}: {verdict} ({ec['reason']})"]
        flagged = [r for r in self.table.itertuples(index=False) if r.problems]
        if self.problems or flagged:
            lines += ["", "Caveats:"]
            lines += [f"  - {p}" for p in self.problems]
            for r in flagged:
                lines.append(f"  - {r.model}: " + "; ".join(r.problems))
        lines += ["", "Read next: .table (every number), .results[model].diagnostics(), "
                      ".peaks[model], .report(path)"]
        return "\n".join(lines)

    def __repr__(self):
        return (self._headline() + "\n  read next: .summary(), .table, .report(path), "
                ".save(dir)")

    # ------------------------------------------------------------------------ saving, loading
    def save(self, path, *, overwrite=False):
        """Save the comparison, its results and its data so :meth:`load` rebuilds it exactly.

        Writes ``comparison.json`` (the table, peaks, diagnostics, evidence check, caveats,
        timing and settings), ``lightcurve.ecsv`` (the data every model was fitted to) and one
        saved result per model under ``results/`` (and ``evidence/`` for the evidence check), each
        with its provenance (:meth:`SamplerResult.save`).

        Parameters
        ----------
        path : str or Path
            A directory.
        overwrite : bool, default False
            Replace a comparison already saved there.

        Returns
        -------
        Path

        Raises
        ------
        FileExistsError
            A comparison is already saved at ``path`` and ``overwrite`` is False.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> from whisper_cbpf.compare import Comparison
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
        ...                  evidence_check=False)
        >>> import tempfile
        >>> path = cmp.save(tempfile.mkdtemp())
        >>> sorted(p.name for p in path.iterdir())
        ['comparison.json', 'lightcurve.ecsv', 'results']
        >>> Comparison.load(path).table.equals(cmp.table)
        True
        """
        root = Path(path)
        if (root / "comparison.json").exists() and not overwrite:
            raise FileExistsError(f"{root} already holds a saved comparison. Pass overwrite=True "
                                  f"to replace it, or choose another directory.")
        root.mkdir(parents=True, exist_ok=True)
        dirs = {}
        for kind, group in (("results", self.results), ("evidence", self.evidence)):
            for i, (lab, res) in enumerate(group.items()):
                rel = f"{kind}/{i:02d}_{_slug(lab)}"
                res.save(root / rel, overwrite=True)
                dirs.setdefault(kind, {})[lab] = rel
        self.lc.write(root / "lightcurve.ecsv", format="ascii.ecsv", overwrite=True)
        doc = {"format": FORMAT_VERSION, "criterion": self.criterion, "winner": self.winner,
               "n_data": self.n_data, "problems": list(self.problems),
               "evidence_check": _plain(self.evidence_check), "settings": self.settings,
               "timing": self.timing, "columns": COLUMNS,
               "table": _records(self.table),
               "peaks": {lab: pk.to_dict() for lab, pk in self.peaks.items()},
               "diagnostics": {lab: d.to_dict() for lab, d in self.diagnostics.items()},
               "models": {lab: getattr(m, "name", str(m)) for lab, m in self.models.items()},
               "results": dirs.get("results", {}), "evidence": dirs.get("evidence", {})}
        (root / "comparison.json").write_text(json.dumps(doc, indent=1, sort_keys=True,
                                                         allow_nan=True))
        return root

    @classmethod
    def load(cls, path, *, models=None):
        """Load a comparison written by :meth:`save`.

        Parameters
        ----------
        path : str or Path
            The directory :meth:`save` wrote.
        models : dict, optional
            ``{model: Model}`` to re-attach models that are not registered by name (a model you
            built with a factory such as :func:`~whisper_cbpf.supernova_model`), for forecasts and
            reports. A registered model is found by name, and a family that :func:`compare`
            bound by name (``"arnett"``, ``"magnetar"``, ``"tde"``, ...) is bound again from the
            saved light curve and the prior its fit recorded.

        Returns
        -------
        Comparison
            Its table, peaks, results and data equal the saved ones.

        Raises
        ------
        FileNotFoundError
            No comparison is saved at ``path``.
        ValueError
            It was written by a newer whisper.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> from whisper_cbpf.compare import Comparison
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> import tempfile
        >>> path = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
        ...                   evidence_check=False).save(tempfile.mkdtemp())
        >>> Comparison.load(path).winner
        'flare'
        """
        from .io import LightCurve
        from .models import _REGISTRY as registry
        from .likelihood_max_opt import LikelihoodMaxOptResult
        from .results import DiagnosticRow, DiagnosticsReport, load_result

        root = Path(path)
        if not (root / "comparison.json").exists():
            raise FileNotFoundError(f"no saved comparison at {root} (comparison.json is "
                                    f"missing). Pass the directory Comparison.save wrote.")
        doc = json.loads((root / "comparison.json").read_text())
        if int(doc.get("format", 0)) > FORMAT_VERSION:
            raise ValueError(f"{root} was saved in comparison format {doc['format']}, newer than "
                             f"this whisper_cbpf reads ({FORMAT_VERSION}). Upgrade whisper_cbpf.")
        table = _table([{c: rec.get(c) for c in COLUMNS} for rec in doc["table"]])
        results = {lab: load_result(root / rel) for lab, rel in doc["results"].items()}
        evidence = {lab: load_result(root / rel) for lab, rel in doc.get("evidence", {}).items()}
        peaks = {lab: LikelihoodMaxOptResult.from_dict(d) for lab, d in doc["peaks"].items()}
        diagnostics = {lab: DiagnosticsReport(
            sampler=d["sampler"], model=d["model"], kind=d["kind"],
            rows=[DiagnosticRow(r["check"], r["value"], r["threshold"], r["passed"], r["reason"])
                  for r in d["rows"]]) for lab, d in doc["diagnostics"].items()}
        given = dict(models or {})
        lc = LightCurve.read(root / "lightcurve.ecsv", format="ascii.ecsv")
        mods = {}
        for lab, name in doc["models"].items():
            m = given.get(lab, registry.get(name))
            if m is None and lab in results:     # a family compare bound: bind it again
                m = _rebind_family(name, lc, results[lab])
            mods[lab] = m if m is not None else name
            if not isinstance(mods[lab], str):
                for group in (results, evidence):
                    if lab in group:
                        from .samplers.base import _keep_model
                        _keep_model(group[lab], mods[lab])
        out = cls(table=table, results=results, peaks=peaks, winner=doc["winner"],
                  criterion=doc["criterion"], lc=lc, models=mods, diagnostics=diagnostics,
                  evidence=evidence, evidence_check=doc["evidence_check"], n_data=doc["n_data"],
                  problems=list(doc["problems"]), timing=doc["timing"],
                  settings=doc["settings"])
        out.loaded_from = str(root)
        return out

    # ------------------------------------------------------------- delegated to other modules
    def facts(self, **kwargs):
        """The computed facts of this comparison (:func:`whisper_cbpf.facts.comparison_facts`).

        Parameters
        ----------
        **kwargs
            Passed on (``thresholds=``).

        Returns
        -------
        dict
            JSON-serialisable facts: the winner, dBIC, weights and grade, per-parameter summaries
            and flags, the thresholds used and the input hashes.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
        ...                  evidence_check=False)
        >>> isinstance(cmp.facts(), dict)
        True
        """
        return _module("facts", "Comparison.facts()").comparison_facts(self, self.lc, **kwargs)

    def report(self, path, **kwargs):
        """Write the one-file HTML report of this comparison (:func:`whisper_cbpf.report.report`).

        Parameters
        ----------
        path : str or Path
            Where to write it.
        **kwargs
            Passed on (``forecast_times=``, ``decision=``, ``title=``).

        Returns
        -------
        Path
            The HTML file.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
        ...                  evidence_check=False)
        >>> import tempfile
        >>> cmp.report(tempfile.mkdtemp()).name
        'report.html'
        """
        return _module("report", "Comparison.report()").report(self, self.lc, path, **kwargs)

    def forecast(self, times, bands, **kwargs):
        """Forecast magnitudes of every ranked model (:func:`whisper_cbpf.forecast.forecast`).

        Parameters
        ----------
        times, bands : array-like
            The epochs (on the light curve's clock) and bands to forecast.
        **kwargs
            Passed on (``n_draws=``, ``quantiles=``, ``survey_depth=``, ``seed=``).

        Returns
        -------
        pandas.DataFrame
            The forecast table of each ranked model, best first, with a leading ``model`` column.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
        ...                  evidence_check=False)
        >>> fc = cmp.forecast([31.0, 33.0], "r")
        >>> list(fc["model"])
        ['flare', 'flare', 'bazin', 'bazin']
        >>> list(fc.columns[:4])
        ['model', 'time', 'band', 'mean_mag']
        """
        fc = _module("forecast", "Comparison.forecast()")
        frames = []
        for lab in self._ranked():
            df = fc.forecast(self.results[lab], times, bands, lc=self.lc,
                             model=self._model(lab), **kwargs)
            df.insert(0, "model", lab)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def discriminate(self, times, bands, survey_depth=None):
        """When and in which band the top two models differ most
        (:func:`whisper_cbpf.forecast.discriminate`).

        Parameters
        ----------
        times, bands : array-like
            The candidate epochs and bands.
        survey_depth : dict or float, optional
            The limiting magnitude (per band); a cell counts only where both models are brighter.

        Returns
        -------
        pandas.DataFrame
            Per cell: ``model_a``, ``model_b``, ``D`` and ``observable``; ``.attrs["best"]`` is the
            most discriminating cell.

        Raises
        ------
        ValueError
            Fewer than two models were ranked.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> cmp = wp.compare(lc, ["flare", "bazin"], nsteps=1500, burnin=500,
        ...                  evidence_check=False)
        >>> d = cmp.discriminate([31.0, 35.0, 45.0], "r")
        >>> sorted(set(d["model_a"]) | set(d["model_b"]))
        ['bazin', 'flare']
        >>> d.attrs["best"]["time"] in (31.0, 35.0, 45.0)
        True
        """
        top = self._ranked()[:2]
        if len(top) < 2:
            raise ValueError(f"discriminate needs two ranked models; this comparison ranked "
                             f"{top}. The others were left out: see table['left_out_reason'].")
        fc = _module("forecast", "Comparison.discriminate()")
        # Forecast here, with each model object, so a model built by a factory and never
        # registered by name is still evaluated.
        forecasts = {lab: fc.forecast(self.results[lab], times, bands, lc=self.lc,
                                      model=self._model(lab), survey_depth=survey_depth)
                     for lab in top}
        return fc.discriminate(forecasts, times, bands, survey_depth=survey_depth)

    def _model(self, label):
        m = self.models.get(label)
        return m if m is not None and not isinstance(m, str) else None


def _module(name, what):
    import importlib

    try:
        return importlib.import_module(f"whisper_cbpf.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"whisper_cbpf.{name}":
            raise
        raise ImportError(f"{what} needs the whisper_cbpf.{name} module, which this installation "
                          f"does not have. Install a whisper_cbpf release that includes it (0.2.0 "
                          f"or later).") from exc


def _slug(text):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._") or "model"


def _records(df):
    out = []
    for rec in df.to_dict(orient="records"):
        row = {}
        for k, v in rec.items():
            if v is pd.NA:
                v = None
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            row[k] = v
        out.append(row)
    return out
