"""Facts: the numbers a reader quotes from a fit or a comparison, each computed by a stated rule.

A fit answers many small questions that people used to answer by hand from its tables: which model
won and by how much, whether a parameter was measured or is still the prior, how far the redshift
and the explosion time narrowed, how bright the source is and how fast it is rising, what its
latest colour is, what absolute magnitude its distance allows. Done by hand, these went wrong in the
demos (a colour missing on 9 of 23 fact sheets; a distance-modulus range of -8.62/+1.62 mag quoted
as symmetric). This module computes them once, from the fit and the data alone:

* :func:`result_facts` -- one fit (a :class:`~whisper_cbpf.samplers.base.SamplerResult`);
* :func:`comparison_facts` -- a model comparison (``wp.compare``), every model plus the ranking;
* :func:`write_facts` -- the JSON file.

Every entry is a number, a boolean, a short string or ``None`` (never NaN), and the file carries
the thresholds its flags used, the rules in words (``facts["rules"]``) and the SHA-256 of every input
(data, posterior draws, model, prior), so each flag can be recomputed. There is no time stamp: the
same inputs give the same bytes, and ``facts["sha256"]`` pins them. Nothing in it is generated
prose: the HTML report (:mod:`whisper_cbpf.report`) shows this file as tables, and any program the
user adds reads the same file.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np

__all__ = ["result_facts", "comparison_facts", "write_facts", "DEFAULT_THRESHOLDS", "RULES",
           "GRADES", "FORMAT_VERSION"]

#: Version of the layout written here; bumped when a key changes meaning.
FORMAT_VERSION = 1
#: One parsec in cm (IAU 2015 B2), for the distance modulus ``5 log10(d_L / 10 pc)``.
PARSEC_CM = 3.0856775814913673e18
#: The parameter names read as the redshift and the explosion (or merger) time.
REDSHIFT, EXPLOSION_TIME = "redshift", "t_exp"
#: Jeffreys' (1961) grades on the ln Bayes factor, weakest first.
GRADES = ("inconclusive", "substantial", "strong", "decisive")

#: The numbers the flags depend on. Override any of them with ``thresholds={...}``.
DEFAULT_THRESHOLDS = {
    # posterior 68 % width / prior 68 % width above this: the prior carries more information than
    # the data (Gaussian limit: ratio = 1 / sqrt(1 + I_data / I_prior)).
    "prior_dominated_ratio": 1.0 / math.sqrt(2.0),
    # the prior-edge rule of result.diagnostics(): more than edge_fraction of the draws in the outer
    # edge_band of a finite prior range (in the prior's own coordinate).
    "edge_band": 0.01,
    "edge_fraction": 0.05,
    # an optimised likelihood maximum this many nats above the sampler's best draw moves BIC by
    # more than 2: the sampler missed the peak.
    "likelihood_max_gain_large": 1.0,
    # two detections in neighbouring bands this close in time make a colour.
    "colour_max_separation_days": 1.0,
    # ln Bayes factor cuts between the grades: ln sqrt(10), ln 10, ln 100.
    "grade_cuts_ln_b": (math.log(10 ** 0.5), math.log(10.0), math.log(100.0)),
}

#: The rules behind every entry, in words; written into every facts file.
RULES = {
    "percentiles": "median, p16 and p84 are numpy.percentile of the posterior draws at 50, 16, 84.",
    "width_ratio": "central 68 % width (p84 - p16) of the posterior over that of the prior, both in "
                   "the prior's own coordinate (log10 for LogUniform, linear otherwise); the "
                   "prior's percentiles are its inverse CDF (rescale) at 0.16, 0.5 and 0.84.",
    "prior_dominated": "width_ratio > thresholds.prior_dominated_ratio (1/sqrt(2)): in the "
                       "Gaussian limit the data then carry less information than the prior.",
    "at_prior_edge": "more than thresholds.edge_fraction of the draws within thresholds.edge_band "
                     "of the range from a finite prior bound, in the prior's own coordinate (the "
                     "rule of result.diagnostics()).",
    "narrowing": "for the redshift and the explosion time t_exp: width_ratio as above, and "
                 "narrowing_factor = 1 / width_ratio.",
    "ranking": "among the models the comparison ranked: delta = lnZ_winner - lnZ_i (ln Z) or "
               "BIC_i - BIC_winner (BIC); weight = exp(lnZ_i) or exp(-BIC_i / 2), normalised; "
               "ln_b = delta (ln Z) or delta / 2 (BIC, Schwarz); grade = Jeffreys on ln_b with "
               "thresholds.grade_cuts_ln_b (inconclusive, substantial, strong, decisive). The "
               "headline grade is the winner's over the runner-up, and 'inconclusive' when the "
               "comparison's evidence check (nested sampling on the top two under BIC) "
               "disagrees with BIC.",
    "aic_bic": "AIC = -2 lnL + 2k, BIC = -2 lnL + k ln n, with lnL the optimised likelihood "
               "maximum when there is one (else the sampler's best draw); None when n <= k (not "
               "enough data).",
    "detections": "rows that are not upper limits, with a finite magnitude and a finite positive "
                  "error; flux-only data are converted to AB magnitudes first.",
    "brightest": "the detection of smallest magnitude in the band (the earliest on a tie). state: "
                 "'single detection', 'rising' (it is the last detection), 'declining' (the first) "
                 "or 'peaked' (in between).",
    "rates": "rise_rate and decline_rate in mag/day: weighted least-squares slope (weights "
             "1/error^2) of the detections up to / from the brightest, sign chosen so that "
             "brightening (rise) and fading (decline) are positive; error = "
             "1/sqrt(sum w (t - t_w)^2). Needs 2 detections at distinct times on that side.",
    "colours": "for each pair of neighbouring bands (by effective wavelength) with detections: of "
               "all detection pairs within thresholds.colour_max_separation_days, the one whose "
               "later point is latest (ties: the closer pair); colour = m_blue - m_red, error = "
               "sqrt(error_blue^2 + error_red^2), time = the later point.",
    "absolute_magnitude": "brightest detection minus the distance modulus 5 log10(d_L / 10 pc), "
                          "d_L from Planck18. Redshift: the fit's posterior when the redshift is "
                          "fitted, else the light curve's known redshift or luminosity distance, "
                          "else its redshift prior hint. value at z50; brighter_by = DM(z84) - "
                          "DM(z50); fainter_by = DM(z50) - DM(z16) (asymmetric). No K-correction, "
                          "no extinction correction.",
    "converged": "every sampler-specific check of result.diagnostics() that could run passed "
                 "(None when none could run).",
    "stranded_walkers": "the 'stuck walkers' (emcee) or 'stranded chains' (NUTS) check failed.",
    "large_likelihood_max_gain": "optimised likelihood maximum - sampler's best draw > "
                                 "thresholds.likelihood_max_gain_large.",
    "hashes": "SHA-256 of the data (whisper_cbpf.results.data_hash), of each posterior table "
              "(samples_hash), the model identity recorded with the fit, and the prior record; "
              "sha256 is over the canonical JSON of every other key.",
}

#: Rows of the diagnostics report that are not about the sampler's convergence.
_NOT_CONVERGENCE = frozenset({"posterior draws", "data points", "prior-edge pile-up",
                              "gap to the likelihood maximum"})
_STRANDED = frozenset({"stuck walkers", "stranded chains"})


# ========================================================================================= helpers
def _thresholds(thresholds):
    """``DEFAULT_THRESHOLDS`` with the caller's overrides, checked."""
    out = dict(DEFAULT_THRESHOLDS)
    if thresholds is None:
        return out
    if not isinstance(thresholds, dict):
        raise TypeError(f"thresholds must be a dict such as {{'prior_dominated_ratio': 0.9}}; got "
                        f"{type(thresholds).__name__}.")
    unknown = sorted(set(thresholds) - set(DEFAULT_THRESHOLDS))
    if unknown:
        raise ValueError(f"unknown threshold(s) {unknown}. Known: {sorted(DEFAULT_THRESHOLDS)}.")
    out.update(thresholds)
    cuts = tuple(float(c) for c in out["grade_cuts_ln_b"])
    if len(cuts) != 3 or not (0 < cuts[0] < cuts[1] < cuts[2]):
        raise ValueError(f"grade_cuts_ln_b must be three increasing positive numbers (the default "
                         f"is ln sqrt(10), ln 10, ln 100); got {out['grade_cuts_ln_b']!r}.")
    out["grade_cuts_ln_b"] = cuts
    for key in ("prior_dominated_ratio", "edge_band", "edge_fraction", "likelihood_max_gain_large",
                "colour_max_separation_days"):
        value = float(out[key])
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"threshold {key!r} must be a finite number > 0; got {out[key]!r}.")
        out[key] = value
    return out


def _num(x):
    """A JSON number: a Python float, or ``None`` for NaN, inf, or anything that is not a number."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean(obj):
    """JSON-ready copy: numpy scalars to Python, tuples to lists, non-finite floats to ``None``."""
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    return str(obj)


def _canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=False).encode()


def _sha(obj):
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _seal(facts):
    """Clean ``facts`` and add ``sha256`` over everything else."""
    facts = _clean(facts)
    facts.pop("sha256", None)
    facts["sha256"] = _sha(facts)
    return facts


def _g(x, digits=3):
    """``x`` to ``digits`` significant figures, for the plain-language readings."""
    return "n/a" if x is None else f"{float(x):.{digits}g}"


def _percentiles(x):
    q16, q50, q84 = np.percentile(np.asarray(x, dtype=float), [16.0, 50.0, 84.0])
    return {"p16": float(q16), "median": float(q50), "p84": float(q84)}


# ========================================================================================== priors
def _prior_from_record(record):
    """Rebuild a :class:`~whisper_cbpf.priors.Prior` from ``results.prior_record``'s output.

    Distributions of an unknown family are left out (their parameters get no prior facts).
    """
    from .priors import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform

    params = (record or {}).get("parameters") or {}
    dists = {}
    for name, rec in params.items():
        kind = rec.get("type")
        try:
            if kind == "Uniform":
                dists[name] = Uniform(rec["low"], rec["high"])
            elif kind == "LogUniform":
                dists[name] = LogUniform(rec["low"], rec["high"])
            elif kind == "Normal":
                dists[name] = Normal(rec["mu"], rec["sigma"])
            elif kind == "TruncatedNormal":
                dists[name] = TruncatedNormal(rec["mu"], rec["sigma"], rec["low"], rec["high"])
            elif kind == "Fixed":
                dists[name] = Fixed(rec["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return Prior(dists) if dists else None


def _resolve_prior(result, model, prior):
    """``(prior, source)``: the prior passed, else the one recorded with the fit, else the model's."""
    if prior is not None:
        if not hasattr(prior, "distributions"):
            raise TypeError(f"prior= must be a whisper_cbpf.priors.Prior; got "
                            f"{type(prior).__name__}.")
        return prior, "passed to result_facts"
    prov = getattr(result, "provenance", None)
    rec = ((prov or {}).get("model") or {}).get("prior") if isinstance(prov, dict) else None
    rebuilt = _prior_from_record(rec)
    if rebuilt is not None:
        return rebuilt, "recorded with the fit"
    from .samplers.base import fitted_model
    try:
        m = fitted_model(result, model)
    except (KeyError, TypeError):
        if model is not None:
            raise
        return None, "none: the fit recorded no prior and its model is not registered"
    if m.default_prior is not None:
        return m.default_prior, "the model's default"
    return None, "none: the fit recorded no prior and the model has no default"


def _coordinate(dist):
    return "log10" if type(dist).__name__ == "LogUniform" else "linear"


def _to_coord(x, coord):
    x = np.asarray(x, dtype=float)
    if coord == "log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(x)
    return x


def _prior_summary(dist):
    kind = type(dist).__name__
    out = {"type": kind}
    for key in ("low", "high", "mu", "sigma", "value"):
        if hasattr(dist, key):
            out[key] = _num(getattr(dist, key))
    lo, hi = getattr(dist, "bounds", (None, None))
    out["low"], out["high"] = _num(lo), _num(hi)
    if hasattr(dist, "rescale"):
        out["p16"], out["median"], out["p84"] = (float(dist.rescale(u)) for u in (0.16, 0.5, 0.84))
    return out


# ================================================================================ per-fit pieces
def _peak_dict(peak):
    """An optimised likelihood maximum as a dict (LikelihoodMaxOptResult, dict, or a list of
    either: the highest)."""
    if peak is None:
        return None
    if isinstance(peak, (list, tuple)):
        found = [p for p in (_peak_dict(q) for q in peak) if p is not None]
        return max(found, key=lambda p: _num(p.get("max_log_likelihood")) or -math.inf,
                   default=None)
    if hasattr(peak, "to_dict"):
        peak = peak.to_dict()
    if not isinstance(peak, dict) or _num(peak.get("max_log_likelihood")) is None:
        return None
    return peak


def _parameters(result, prior, th, peak):
    """``(parameters, fixed)``: the per-parameter facts, and the parameters the prior pins."""
    from .priors import Fixed

    samples = result.samples
    dists = getattr(prior, "distributions", {}) if prior is not None else {}
    peak_params = (peak or {}).get("params") or {}
    peak_edge = set((peak or {}).get("at_edge") or [])
    info = result.info if isinstance(result.info, dict) else {}
    pinned = dict(info.get("fixed") or {}) if isinstance(info.get("fixed"), dict) else {}
    params, fixed = {}, {}
    for name in result.parameters:
        dist = dists.get(name)
        if isinstance(dist, Fixed) or name in pinned:
            fixed[name] = _num(dist.value if isinstance(dist, Fixed) else pinned[name])
            continue
        if name not in samples.columns or len(samples) == 0:
            continue
        x = samples[name].to_numpy(dtype=float)
        finite = np.isfinite(x)
        rec = {"n_draws": int(finite.sum())}
        if not finite.any():
            rec["reading"] = f"not enough data: no finite draw of {name}"
            params[name] = rec
            continue
        x = x[finite]
        rec.update(_percentiles(x))
        rec["prior"] = _prior_summary(dist) if dist is not None else None
        rec.update({"coordinate": None, "posterior_width": None, "prior_width": None,
                    "width_ratio": None, "prior_dominated": None, "at_prior_edge": None,
                    "edge_side": None, "edge_fraction_lower": None, "edge_fraction_upper": None})
        if dist is not None and hasattr(dist, "rescale"):
            coord = _coordinate(dist)
            post = _to_coord(np.percentile(x, [16.0, 84.0]), coord)
            pri = _to_coord([dist.rescale(0.16), dist.rescale(0.84)], coord)
            post_w, pri_w = float(post[1] - post[0]), float(pri[1] - pri[0])
            ratio = post_w / pri_w if pri_w > 0 else None
            rec.update(coordinate=coord, posterior_width=post_w, prior_width=pri_w,
                       width_ratio=ratio,
                       prior_dominated=None if ratio is None else
                       bool(ratio > th["prior_dominated_ratio"]))
            rec.update(_edge(x, dist, coord, th))
        if name in peak_params:
            rec["peak"] = _num(peak_params[name])
            rec["peak_at_prior_edge"] = name in peak_edge
        rec["reading"] = _reading(name, rec, th)
        params[name] = rec
    return params, fixed


def _edge(x, dist, coord, th):
    """The prior-edge pile-up of one parameter (the rule of ``result.diagnostics()``)."""
    lo, hi = (float(b) for b in dist.bounds)
    if coord == "log10":
        lo, hi = math.log10(lo), math.log10(hi)
    y = _to_coord(x, coord)
    out = {"edge_fraction_lower": None, "edge_fraction_upper": None}
    if not (math.isfinite(lo) or math.isfinite(hi)) or not hi > lo:
        out.update(at_prior_edge=False, edge_side=None)
        return out
    band = th["edge_band"] * (hi - lo) if math.isfinite(hi - lo) else None
    sides = []
    if math.isfinite(lo) and band is not None:
        out["edge_fraction_lower"] = float(np.mean(y <= lo + band))
        if out["edge_fraction_lower"] > th["edge_fraction"]:
            sides.append("lower")
    if math.isfinite(hi) and band is not None:
        out["edge_fraction_upper"] = float(np.mean(y >= hi - band))
        if out["edge_fraction_upper"] > th["edge_fraction"]:
            sides.append("upper")
    out.update(at_prior_edge=bool(sides), edge_side="+".join(sides) or None)
    return out


def _reading(name, rec, th):
    """One plain sentence for a parameter: a measurement, the prior, or a limit."""
    interval = f"median {_g(rec['median'])}, 16th-84th percentile {_g(rec['p16'])} to " \
               f"{_g(rec['p84'])}"
    parts = []
    if rec.get("prior_dominated"):
        parts.append(f"not enough data to measure {name}: its posterior is "
                     f"{rec['width_ratio']:.2f} of the prior's width (above "
                     f"{th['prior_dominated_ratio']:.2f} the prior outweighs the data), so the "
                     f"prior sets it")
    if rec.get("at_prior_edge"):
        side = rec["edge_side"]
        bound = {"lower": rec["prior"]["low"], "upper": rec["prior"]["high"]}.get(side)
        where = f"its {side} prior bound ({_g(bound)})" if bound is not None else "both bounds"
        parts.append(f"{name} piles up against {where}: the bound, not the data, limits it; "
                     f"quote a limit, not a value")
    if parts:
        return "; ".join(parts) + f" ({interval})"
    return (f"{name} = {_g(rec['median'])} (16th-84th percentile {_g(rec['p16'])} to "
            f"{_g(rec['p84'])})")


def _narrowing(params, fixed, lc):
    """How far the redshift and the explosion time narrowed, or what value they were held at."""
    out = {}
    for name in (REDSHIFT, EXPLOSION_TIME):
        if name in params and params[name].get("width_ratio") is not None:
            p = params[name]
            ratio = p["width_ratio"]
            out[name] = {"fitted": True,
                         "prior": {k: p["prior"].get(k) for k in ("p16", "median", "p84")},
                         "posterior": {k: p[k] for k in ("p16", "median", "p84")},
                         "coordinate": p["coordinate"], "width_ratio": ratio,
                         "narrowing_factor": (1.0 / ratio) if ratio and ratio > 0 else None,
                         "prior_dominated": p["prior_dominated"]}
        elif name in params:
            p = params[name]
            out[name] = {"fitted": True, "prior": None,
                         "posterior": {k: p.get(k) for k in ("p16", "median", "p84")},
                         "width_ratio": None, "narrowing_factor": None, "prior_dominated": None,
                         "reason": "no prior recorded for this parameter"}
        elif name in fixed:
            out[name] = {"fitted": False, "value": fixed[name], "source": "held fixed by the prior"}
        elif name == REDSHIFT and lc is not None and lc.meta.get("redshift") is not None:
            out[name] = {"fitted": False, "value": float(lc.meta["redshift"]),
                         "source": "the light curve's redshift"}
        else:
            out[name] = {"fitted": False, "value": None,
                         "source": "not a parameter of this fit"}
    return out


def _fit_block(result, peak):
    n, k = int(result.n_data), int(result.n_params)
    enough = n > k
    sampler_ll = _num(result.max_log_likelihood)
    ll = _num(peak["max_log_likelihood"]) if peak else sampler_ll
    info = result.info if isinstance(result.info, dict) else {}
    out = {"sampler": str(result.sampler), "n_data": n, "n_params": k,
           "n_draws": int(result.n_samples), "enough_data": enough,
           "sampler_max_log_likelihood": sampler_ll,
           "optimised_max_log_likelihood": _num(peak["max_log_likelihood"]) if peak else None,
           "likelihood_max_gain": (None if not peak or sampler_ll is None else
                                   float(peak["max_log_likelihood"]) - sampler_ll),
           "log_likelihood_source": ("optimised likelihood maximum" if peak else
                                     "sampler's best draw"),
           "max_log_likelihood": ll,
           "aic": None, "bic": None,
           "log_evidence": _num(info.get("log_evidence")),
           "log_evidence_err": _num(info.get("log_evidence_err")),
           "runtime_s": _num(result.runtime_s)}
    if enough and ll is not None:
        out["aic"] = -2.0 * ll + 2.0 * k
        out["bic"] = -2.0 * ll + k * math.log(n)
    else:
        out["reading"] = (f"not enough data: {n} data points for {k} free parameters, so AIC and "
                          f"BIC are not given" if not enough else "no finite log-likelihood")
    return out


def _diagnostics(result, prior, peak):
    peak_ll = float(peak["max_log_likelihood"]) if peak else None
    report = result.diagnostics(prior=prior, likelihood_max_opt=peak_ll)
    rows = report.to_dict()
    conv = [r["passed"] for r in rows["rows"] if r["check"] not in _NOT_CONVERGENCE
            and r["passed"] is not None]
    stranded = [r["passed"] for r in rows["rows"] if r["check"] in _STRANDED
                and r["passed"] is not None]
    converged = None if not conv else all(conv)
    stranded_flag = None if not stranded else not all(stranded)
    return rows, converged, stranded_flag


def _result_core(result, lc, model, prior, th, peak):
    """Everything :func:`result_facts` says about the fit itself (no data block, no seal)."""
    from .results import data_hash, prior_record, samples_hash

    peak = _peak_dict(peak) or _peak_dict((result.info or {}).get("likelihood_max_opt")
                                          if isinstance(result.info, dict) else None)
    prior, prior_source = _resolve_prior(result, model, prior)
    params, fixed = _parameters(result, prior, th, peak)
    fit = _fit_block(result, peak)
    diag, converged, stranded = _diagnostics(result, prior, peak)
    dominated = sorted(p for p, r in params.items() if r.get("prior_dominated"))
    edge = sorted(p for p, r in params.items() if r.get("at_prior_edge"))
    prov = result.provenance if isinstance(getattr(result, "provenance", None), dict) else {}
    fit_hash = (prov.get("data") or {}).get("hash")
    lc_hash = data_hash(lc)
    gain = fit["likelihood_max_gain"]
    caveats = {
        "converged": converged,
        "stranded_walkers": stranded,
        "not_enough_data": not fit["enough_data"],
        "no_posterior_draws": result.n_samples == 0,
        "large_likelihood_max_gain": (None if gain is None else
                                      bool(gain > th["likelihood_max_gain_large"])),
        "prior_dominated": bool(dominated),
        "at_prior_edge": bool(edge),
        "data_differs_from_fit": None if fit_hash is None else fit_hash != lc_hash,
        "prior_dominated_parameters": dominated,
        "at_prior_edge_parameters": edge,
        "diagnostics_failed": diag["reasons"],
    }
    readings = []
    if result.n_samples == 0:
        readings.append("not enough data: the fit returned no posterior draws, so no parameter "
                        "is measured")
    if not fit["enough_data"]:
        readings.append(fit["reading"])
    return {
        "model": str(result.model),
        "sampler": str(result.sampler),
        "fit": fit,
        "parameters": params,
        "fixed": fixed,
        "narrowing": _narrowing(params, fixed, lc),
        "absolute_magnitude": None,               # filled by the caller (needs the data block)
        "diagnostics": diag,
        "caveats": caveats,
        "readings": readings,
        "inputs": {
            "data_sha256": lc_hash,
            "fit_data_sha256": fit_hash,
            "samples_sha256": samples_hash(result.samples),
            "model_identity_sha256": (prov.get("model") or {}).get("identity_hash"),
            "prior_source": prior_source,
            "prior_sha256": None if prior is None else _sha(_clean(prior_record(prior))),
        },
    }


# ======================================================================================== the data
def _time_axis(lc):
    meta = lc.meta
    ref = meta.get("time_reference")
    if ref is None and "explosion_mjd" in meta:
        ref = "explosion"
    return {"unit": "days",
            "reference": ref if ref is not None else "MJD",
            "reference_mjd": _num(meta.get("time_reference_mjd", meta.get("explosion_mjd")))}


def _magnitudes(lc):
    """``(mag, err, upper_limit, reason)`` of every row; ``reason`` is set when there are none."""
    n = len(lc)
    ul = (np.asarray(lc["upper_limit"], dtype=bool) if "upper_limit" in lc.colnames
          else np.zeros(n, dtype=bool))
    if "magnitude" in lc.colnames:
        mag = np.asarray(lc["magnitude"], dtype=float)
        err = (np.asarray(lc["magnitude_err"], dtype=float) if "magnitude_err" in lc.colnames
               else np.full(n, np.nan))
        return mag, err, ul, None
    if "flux" in lc.colnames and lc.meta.get("data_mode") != "flux":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore", invalid="ignore"):
                converted = lc.add_mag()
        mag = np.asarray(converted["magnitude"], dtype=float)
        err = (np.asarray(converted["magnitude_err"], dtype=float)
               if "magnitude_err" in converted.colnames else np.full(n, np.nan))
        return mag, err, ul, None
    return None, None, ul, ("band-integrated flux (data_mode='flux') has no AB magnitude, so "
                            "no magnitude facts are given")


def _slope(t, m, e):
    """Weighted least-squares slope of ``m`` on ``t`` (weights ``1/e^2``) and its 1-sigma error."""
    w = 1.0 / np.asarray(e, dtype=float) ** 2
    t, m = np.asarray(t, dtype=float), np.asarray(m, dtype=float)
    tw, mw = np.sum(w * t) / np.sum(w), np.sum(w * m) / np.sum(w)
    s = float(np.sum(w * (t - tw) ** 2))
    if not s > 0:
        return None, None
    return float(np.sum(w * (t - tw) * (m - mw)) / s), 1.0 / math.sqrt(s)


def _rate(t, m, e, sign, side):
    n = len(t)
    if n < 2 or np.ptp(t) <= 0:
        return {"value": None, "error": None, "n_points": int(n), "baseline_days": None,
                "reason": f"not enough data: {n} detection(s) at distinct times {side} the "
                          f"brightest (2 needed)"}
    slope, err = _slope(t, m, e)
    return {"value": sign * slope, "error": err, "n_points": int(n),
            "baseline_days": float(np.ptp(t))}


def _band_order(bands):
    """Bands by effective wavelength (unknown ones last, by name), and the known wavelengths."""
    from .io.bands import resolve_bands

    lam = {}
    if bands:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, _, _ = resolve_bands(list(bands), svo_fallback=False, warn=False)
        lam = {b: float(v) for b, v in zip(bands, values) if np.isfinite(v)}
    order = sorted(bands, key=lambda b: (b not in lam, lam.get(b, 0.0), b))
    return order, lam


def _latest_colour(det, blue, red, max_sep):
    """The latest colour ``blue - red`` from detection pairs within ``max_sep`` days."""
    tb, mb, eb = det[blue]
    tr, mr, er = det[red]
    best = None
    for i in range(len(tb)):
        for j in range(len(tr)):
            sep = abs(tb[i] - tr[j])
            if sep > max_sep:
                continue
            key = (max(tb[i], tr[j]), -sep)
            if best is None or key > best[0]:
                best = (key, i, j, sep)
    if best is None:
        return {"bands": [blue, red], "colour": None, "error": None, "time": None,
                "separation_days": None,
                "reason": f"not enough data: no {blue} and {red} detections within "
                          f"{max_sep:g} d of each other"}
    _, i, j, sep = best
    return {"bands": [blue, red], "colour": float(mb[i] - mr[j]),
            "error": float(math.hypot(eb[i], er[j])), "time": float(max(tb[i], tr[j])),
            "separation_days": float(sep)}


def _data_block(lc, th):
    """What the photometry says directly, before any model: detections, peak, rates, colours."""
    n = len(lc)
    band = np.asarray(lc["band"]).astype(str) if n else np.array([], dtype=str)
    time = np.asarray(lc["time"], dtype=float) if n else np.array([], dtype=float)
    mag, err, ul, reason = _magnitudes(lc)
    names = sorted(set(band.tolist()))
    order, lam = _band_order(names)
    out = {"name": lc.meta.get("name"), "n_points": int(n), "n_upper_limits": int(ul.sum()),
           "bands": order, "time_axis": _time_axis(lc)}
    if reason is not None:
        out["reading"] = reason
        return out
    usable = (~ul) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    out["n_detections"] = int(usable.sum())
    out["n_unusable"] = int(((~ul) & ~usable).sum())
    if not usable.any():
        out["reading"] = "not enough data: no detection with a finite magnitude and error"
        out["per_band"], out["colours"] = {}, []
        return out
    t_first = float(time[usable].min())
    out.update(first_detection_time=t_first, last_detection_time=float(time[usable].max()),
               span_days=float(time[usable].max() - t_first))
    before = ul & (time < t_first) & np.isfinite(mag)
    if before.any():
        i = int(np.flatnonzero(before)[np.argmax(time[before])])
        out["last_nondetection_before_first_detection"] = {
            "time": float(time[i]), "band": band[i], "limiting_magnitude": float(mag[i])}
    else:
        out["last_nondetection_before_first_detection"] = None
    per_band, det = {}, {}
    for b in order:
        sel = usable & (band == b)
        rec = {"effective_wavelength_angstrom": lam.get(b), "n_detections": int(sel.sum()),
               "n_upper_limits": int((ul & (band == b)).sum())}
        if not sel.any():
            rec["reading"] = "no detection in this band"
            per_band[b] = rec
            continue
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(time[idx], kind="stable")]
        t, m, e = time[idx], mag[idx], err[idx]
        det[b] = (t, m, e)
        k = int(np.argmin(m))                        # the earliest of equal magnitudes
        state = ("single detection" if len(t) == 1 else "rising" if k == len(t) - 1
                 else "declining" if k == 0 else "peaked")
        rec.update(
            first_detection_time=float(t[0]), last_detection_time=float(t[-1]),
            brightest={"time": float(t[k]), "magnitude": float(m[k]), "error": float(e[k])},
            last_detection={"time": float(t[-1]), "magnitude": float(m[-1]),
                            "error": float(e[-1])},
            state=state, peak_observed=state == "peaked",
            rise_rate=_rate(t[:k + 1], m[:k + 1], e[:k + 1], -1.0, "up to"),
            decline_rate=_rate(t[k:], m[k:], e[k:], 1.0, "from"))
        per_band[b] = rec
    out["per_band"] = per_band
    ordered = [b for b in order if b in det and b in lam]
    out["colours"] = [_latest_colour(det, blue, red, th["colour_max_separation_days"])
                      for blue, red in zip(ordered, ordered[1:])]
    if not out["colours"]:
        out["colours_reading"] = ("not enough data: detections in fewer than two bands of known "
                                  "wavelength, so there is no colour")
    return out


# ============================================================================ absolute magnitude
def _distance_modulus(z):
    from .models.cosmology import Z_MAX, Z_MIN, luminosity_distance_cm

    if z is None or not (Z_MIN <= float(z) <= Z_MAX):
        return None
    return 5.0 * math.log10(float(luminosity_distance_cm(float(z))) / (10.0 * PARSEC_CM))


def _redshift_source(params, fixed, lc):
    """``(kind, z16, z50, z84, low, high, text)`` for the distance, or ``None``."""
    if REDSHIFT in params and params[REDSHIFT].get("median") is not None:
        p = params[REDSHIFT]
        prior = p.get("prior") or {}
        return ("posterior", p["p16"], p["median"], p["p84"], prior.get("low"), prior.get("high"),
                "the fit's redshift posterior")
    if REDSHIFT in fixed and fixed[REDSHIFT] is not None:
        z = fixed[REDSHIFT]
        return ("fixed", z, z, z, None, None, "the redshift the prior holds fixed")
    if lc.meta.get("luminosity_distance") is not None:
        return ("luminosity_distance", None, None, None, None, None,
                "the light curve's luminosity distance")
    if lc.meta.get("redshift") is not None:
        z = float(lc.meta["redshift"])
        return ("known", z, z, z, None, None, "the light curve's redshift")
    hint = lc.meta.get("redshift_prior")
    if hint is not None:
        from .io.schema import redshift_distribution
        try:
            dist = redshift_distribution(hint)
        except (ValueError, NotImplementedError):
            return None
        lo, hi = dist.bounds
        return ("prior", float(dist.rescale(0.16)), float(dist.rescale(0.5)),
                float(dist.rescale(0.84)), _num(lo), _num(hi), "the light curve's redshift prior")
    return None


def _absolute_magnitude(data, params, fixed, lc):
    per_band = data.get("per_band") or {}
    source = _redshift_source(params, fixed, lc)
    if source is None:
        return {"available": False,
                "reading": "not enough data: no redshift, luminosity distance or redshift prior"}
    kind, z16, z50, z84, low, high, text = source
    if kind == "luminosity_distance":
        dm = 5.0 * math.log10(float(lc.meta["luminosity_distance"]) * 1e6 / 10.0)
        dm16 = dm50 = dm84 = dm
    else:
        dm16, dm50, dm84 = (_distance_modulus(z) for z in (z16, z50, z84))
    if None in (dm16, dm50, dm84):
        return {"available": False,
                "reading": f"the redshift from {text} ({_g(z16)}-{_g(z84)}) is outside the "
                           f"distance table (1e-4 to 10), so no distance modulus is given"}
    dm_low, dm_high = _distance_modulus(low), _distance_modulus(high)
    out = {"available": True, "distance_source": text, "redshift_kind": kind,
           "redshift": None if kind == "luminosity_distance" else
           {"p16": z16, "median": z50, "p84": z84, "prior_low": low, "prior_high": high},
           "distance_modulus": {"value": dm50, "brighter_by": dm84 - dm50,
                                "fainter_by": dm50 - dm16, "at_prior_low": dm_low,
                                "at_prior_high": dm_high},
           "note": "brightest detection minus the distance modulus (Planck18); no K-correction, "
                   "no extinction correction",
           "per_band": {}}
    for b, rec in per_band.items():
        if "brightest" not in rec:
            continue
        m = rec["brightest"]["magnitude"]
        row = {"apparent_magnitude": m, "error": rec["brightest"]["error"],
               "value": m - dm50, "brighter_by": dm84 - dm50, "fainter_by": dm50 - dm16,
               "range_at_prior_bounds": None}
        if dm_low is not None and dm_high is not None:
            row["range_at_prior_bounds"] = [m - dm_high, m - dm_low]
        out["per_band"][b] = row
    return out


# ============================================================================= public functions
def result_facts(result, lc, *, model=None, prior=None, thresholds=None):
    """The facts of one fit: every number a reader quotes from it, each computed by a stated rule.

    Parameters
    ----------
    result : SamplerResult
        A fit (from :func:`whisper_cbpf.fit`, or loaded with :func:`whisper_cbpf.load_result`).
        An optimised likelihood maximum kept in ``result.info["likelihood_max_opt"]`` is used for
        AIC/BIC and the gain.
    lc : LightCurve
        The data the fit was run on. Its hash is compared with the one recorded with the fit
        (``caveats["data_differs_from_fit"]``).
    model : str or Model, optional
        Only read for its default prior, when neither ``prior=`` nor the fit's record gives one.
    prior : Prior, optional
        The prior for the widths and the edge flags. Default: the prior recorded with the fit.
    thresholds : dict, optional
        Overrides of :data:`DEFAULT_THRESHOLDS` (``prior_dominated_ratio``, ``edge_band``,
        ``edge_fraction``, ``likelihood_max_gain_large``, ``colour_max_separation_days``,
        ``grade_cuts_ln_b``).

    Returns
    -------
    dict
        JSON-ready (no NaN): ``fit`` (n, k, draws, ln L of the sampler and of the optimised
        likelihood maximum, the gain, AIC, BIC, ln Z), ``parameters`` (per parameter: median,
        p16, p84, the prior, the posterior/prior width ratio, ``prior_dominated``, ``at_prior_edge``, a ``reading``),
        ``fixed``, ``narrowing`` (redshift and explosion time), ``data`` (per band: detections,
        brightest, state, rise and decline rates; the latest colours), ``absolute_magnitude``
        (asymmetric range from the distance), ``diagnostics``, ``caveats`` (booleans),
        ``readings``, ``thresholds``, ``rules``, ``inputs`` (hashes) and ``sha256``.

    Raises
    ------
    TypeError
        ``result`` is not a fit, or ``lc`` is not a light curve.
    ValueError
        An unknown threshold, or a threshold that is not a positive number.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.facts import result_facts
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = wp.get_model("flare").predict(
    ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["ztfr"] * 40, flux=flux, flux_err=np.full(40, 0.1),
    ...                    redshift=0.05)
    >>> res = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0)
    >>> facts = result_facts(res, lc)
    >>> facts["parameters"]["decay_time"]["prior_dominated"]
    False
    >>> facts["caveats"]["not_enough_data"], facts["data"]["per_band"]["ztfr"]["state"]
    (False, 'peaked')
    >>> round(facts["absolute_magnitude"]["distance_modulus"]["value"], 2)
    36.81
    """
    th = _thresholds(thresholds)
    _check_inputs(result, lc)
    core = _result_core(result, lc, model, prior, th, None)
    data = _data_block(lc, th)
    core["absolute_magnitude"] = _absolute_magnitude(data, core["parameters"], core["fixed"], lc)
    facts = {"kind": "whisper_cbpf.result_facts", "format_version": FORMAT_VERSION,
             **core, "data": data, "thresholds": th, "rules": RULES}
    return _seal(facts)


def _check_inputs(result, lc):
    if not all(hasattr(result, a) for a in ("samples", "parameters", "n_data", "n_params")):
        raise TypeError(f"expected a fit (SamplerResult, e.g. from wp.fit or wp.load_result); got "
                        f"{type(result).__name__}.")
    if lc is None or not (hasattr(lc, "colnames") and hasattr(lc, "meta")):
        raise TypeError("the facts need the light curve the fit was run on: pass lc= (a "
                        f"LightCurve); got {type(lc).__name__}.")


def _grade(ln_b, cuts):
    if ln_b is None:
        return None
    for cut, label in zip(cuts, GRADES):
        if ln_b < cut:
            return label
    return GRADES[-1]


def _cell(row, key):
    """``row[key]`` with pandas' missing values (NaN, NA, None) read as ``None``."""
    value = row.get(key) if hasattr(row, "get") else None
    if value is None:
        return None
    if np.ndim(value) == 0:
        import pandas as pd
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
    return value


def _evidence_check(comparison):
    """The comparison's evidence check (nested sampling on the top two under BIC), if it ran."""
    check = getattr(comparison, "evidence_check", None)
    if not isinstance(check, dict) or not check.get("ran"):
        return {"ran": False,
                "reason": None if not isinstance(check, dict) else check.get("reason")}
    return {"ran": True, "models": [str(m) for m in check.get("models") or []],
            "agrees_with_bic": check.get("agrees"), "ln_b": _num(check.get("ln_b")),
            "ln_b_err": _num(check.get("ln_b_err")), "reason": check.get("reason")}


def _ranking(comparison, th):
    """The ranking facts from the comparison's table: deltas, weights, ln B and grades recomputed.

    The models ranked are the ones the comparison ranked (``status == "ranked"``, or no
    ``left_out_reason`` for a table without ``status``) with a finite criterion value.
    """
    table = comparison.table
    criterion = str(comparison.criterion)
    lnz = criterion.replace(" ", "").lower() in ("lnz", "log_evidence", "evidence")
    rows = table.to_dict(orient="records")
    ranked, left_out = [], []
    for row in rows:
        reason = _cell(row, "left_out_reason")
        status = _cell(row, "status")
        value = _num(_cell(row, "log_evidence" if lnz else "bic"))
        out = (status is not None and status != "ranked") or reason not in (None, "")
        if out or value is None:
            left_out.append({"model": str(row["model"]),
                             "reason": str(reason) if reason not in (None, "") else
                             f"no finite {'ln Z' if lnz else 'BIC'}"})
        else:
            ranked.append((str(row["model"]), value, row))
    ranked.sort(key=lambda r: -r[1] if lnz else r[1])
    models = []
    best = ranked[0][1] if ranked else None
    if ranked:
        logw = [(v - best) if lnz else -(v - best) / 2.0 for _, v, _ in ranked]
        total = sum(math.exp(x) for x in logw)
        bic_best = _num(_cell(ranked[0][2], "bic"))
        for i, ((name, value, row), lw) in enumerate(zip(ranked, logw)):
            delta = (best - value) if lnz else (value - best)
            ln_b = delta if lnz else delta / 2.0
            bic = _num(_cell(row, "bic"))
            models.append({
                "model": name, "rank": i + 1, "sampler": _cell(row, "sampler"),
                "n_params": _cell(row, "n_params"), "n_data": _cell(row, "n_data"),
                "max_log_likelihood": _num(_cell(row, "max_log_likelihood")),
                "aic": _num(_cell(row, "aic")), "bic": bic,
                "log_evidence": _num(_cell(row, "log_evidence")),
                "log_evidence_err": _num(_cell(row, "log_evidence_err")),
                "delta": delta,
                "delta_bic": None if bic is None or bic_best is None else bic - bic_best,
                "weight": math.exp(lw) / total,
                "ln_b_winner_over_this": ln_b if i else None,
                "grade_winner_over_this": _grade(ln_b, th["grade_cuts_ln_b"]) if i else None,
                "diagnostics_passed": _cell(row, "converged"),
                "converged": None})                       # filled from the model's own facts
    winner = models[0]["model"] if models else None
    runner = models[1] if len(models) > 1 else None
    stated = getattr(comparison, "winner", None)
    check = _evidence_check(comparison)
    grade = runner["grade_winner_over_this"] if runner else None
    grade_note = None
    if (runner and check["ran"] and check["agrees_with_bic"] is False and not lnz
            and set(check["models"]) == {winner, runner["model"]}):
        grade = GRADES[0]
        grade_note = ("inconclusive because the evidence check (nested sampling on the top two) "
                      "disagrees with BIC")
    return {
        "criterion": "ln Z" if lnz else "BIC",
        "winner": winner,
        "comparison_winner": None if stated is None else str(stated),
        "runner_up": runner["model"] if runner else None,
        "ln_b_winner_over_runner_up": runner["ln_b_winner_over_this"] if runner else None,
        "grade": grade,
        "grade_note": grade_note,
        "evidence_check": check,
        "reading": _ranking_reading(models, lnz, grade, grade_note),
        "models": models,
        "left_out": left_out,
    }


def _ranking_reading(models, lnz, grade, grade_note):
    if not models:
        return "not enough data: no model could be ranked"
    if len(models) == 1:
        return f"only {models[0]['model']} could be ranked, so there is no comparison"
    w, r = models[0], models[1]
    what = "ln Z" if lnz else "BIC"
    text = (f"{w['model']} is preferred over {r['model']} by delta {what} = {r['delta']:.1f} "
            f"(ln B = {r['ln_b_winner_over_this']:.2f}: {grade}); weight {w['weight']:.3f}")
    return text + (f"; {grade_note}" if grade_note else "")


def comparison_facts(comparison, lc=None, *, thresholds=None):
    """The facts of a model comparison: the ranking, and :func:`result_facts` for every model.

    Parameters
    ----------
    comparison : Comparison
        From ``wp.compare(lc, models)`` (or ``Comparison.load``). Read: ``table``, ``results``,
        ``peaks`` (each model's optimised likelihood maximum), ``criterion`` and ``winner``.
    lc : LightCurve, optional
        The data every model was fit to. Default: ``comparison.lc``.
    thresholds : dict, optional
        Overrides of :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    dict
        JSON-ready: ``ranking`` (criterion, winner, runner-up, ln B and Jeffreys grade of the winner
        over the runner-up, "inconclusive" when the comparison's evidence check disagrees with BIC;
        per model its rank, delta, delta BIC, weight, ln B and grade against the winner, whether
        its diagnostics passed and whether it converged; the evidence check; the models left out
        and why), ``models`` (per model the ``result_facts`` entries ``fit``, ``parameters``,
        ``fixed``, ``narrowing``, ``absolute_magnitude``, ``diagnostics``, ``caveats``,
        ``readings``, ``inputs``), ``data`` (as in :func:`result_facts`, once),
        ``absolute_magnitude`` (from the light curve's own redshift, distance or redshift prior,
        whatever the models fitted; each model's block uses its redshift posterior), ``caveats``
        (models left out, which models raised each flag, the comparison's own problems),
        ``thresholds``, ``rules``, ``inputs`` and ``sha256``.

    Raises
    ------
    TypeError
        ``comparison`` lacks ``table`` / ``results`` / ``criterion``, or no light curve is given.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.facts import comparison_facts
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = wp.get_model("flare").predict(
    ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["ztfr"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> cmp = wp.compare(lc, ["flare", "bazin"], sampler="nested", nlive=100)
    >>> facts = comparison_facts(cmp, lc)
    >>> facts["ranking"]["winner"], facts["ranking"]["criterion"]
    ('flare', 'ln Z')
    >>> sorted(facts["models"])
    ['bazin', 'flare']
    """
    th = _thresholds(thresholds)
    for attr in ("table", "results", "criterion"):
        if not hasattr(comparison, attr):
            raise TypeError(f"expected a Comparison from wp.compare (with .table, .results, "
                            f".criterion); {type(comparison).__name__} has no .{attr}.")
    lc = lc if lc is not None else getattr(comparison, "lc", None)
    if lc is None or not (hasattr(lc, "colnames") and hasattr(lc, "meta")):
        raise TypeError("comparison_facts needs the light curve the models were fit to: pass lc= "
                        "(this comparison does not carry one).")
    peaks = getattr(comparison, "peaks", None) or {}
    ranking = _ranking(comparison, th)
    data = _data_block(lc, th)
    models = {}
    results = {str(k): v for k, v in comparison.results.items() if v is not None}
    ranked = [m["model"] for m in ranking["models"] if m["model"] in results]
    for name in ranked + sorted(set(results) - set(ranked)):      # rank order, then by name
        result = results[name]
        core = _result_core(result, lc, None, None, th, peaks.get(name))
        core["absolute_magnitude"] = _absolute_magnitude(data, core["parameters"], core["fixed"],
                                                         lc)
        models[str(name)] = core
    for row in ranking["models"]:
        row["converged"] = (models.get(row["model"]) or {}).get("caveats", {}).get("converged")
    caveat_models = {
        key: sorted(m for m, c in models.items() if c["caveats"].get(key))
        for key in ("stranded_walkers", "not_enough_data", "no_posterior_draws",
                    "large_likelihood_max_gain", "prior_dominated", "at_prior_edge",
                    "data_differs_from_fit")}
    caveat_models["not_converged"] = sorted(m for m, c in models.items()
                                            if c["caveats"]["converged"] is False)
    caveats = {"models_left_out": ranking["left_out"],
               "any_model_left_out": bool(ranking["left_out"]),
               "winner_differs_from_comparison": (
                   None if ranking["comparison_winner"] is None or ranking["winner"] is None
                   else ranking["winner"] != ranking["comparison_winner"]),
               "winner_converged": (models.get(ranking["winner"], {}).get("caveats", {})
                                    .get("converged") if ranking["winner"] else None),
               "winner_prior_dominated_parameters": (
                   models.get(ranking["winner"], {}).get("caveats", {})
                   .get("prior_dominated_parameters") if ranking["winner"] else None),
               "evidence_check_disagrees": (ranking["evidence_check"].get("agrees_with_bic")
                                            is False),
               "models_with": caveat_models,
               "comparison_problems": [str(p) for p in
                                       (getattr(comparison, "problems", None) or [])]}
    from .results import data_hash
    facts = {"kind": "whisper_cbpf.comparison_facts", "format_version": FORMAT_VERSION,
             "ranking": ranking, "models": models, "data": data,
             "absolute_magnitude": _absolute_magnitude(data, {}, {}, lc),   # the data's own z
             "caveats": caveats,
             "thresholds": th, "rules": RULES,
             "inputs": {"data_sha256": data_hash(lc),
                        "results": {m: {k: c["inputs"][k] for k in
                                        ("samples_sha256", "model_identity_sha256",
                                         "prior_sha256")} for m, c in models.items()}}}
    return _seal(facts)


def write_facts(facts, path):
    """Write a facts dict as JSON (indented, UTF-8, no NaN), byte for byte the same every time.

    Parameters
    ----------
    facts : dict
        From :func:`result_facts` or :func:`comparison_facts`.
    path : str or Path
        A file (``.json``), or a directory, which gets ``facts.json``. Parent directories are made.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    TypeError
        ``facts`` is not a dict.

    Examples
    --------
    >>> import json, tempfile
    >>> from whisper_cbpf.facts import write_facts
    >>> out = write_facts({"kind": "example", "value": 1.5}, tempfile.mkdtemp())
    >>> out.name, json.loads(out.read_text())["value"]
    ('facts.json', 1.5)
    """
    if not isinstance(facts, dict):
        raise TypeError(f"write_facts writes a dict from result_facts or comparison_facts; got "
                        f"{type(facts).__name__}.")
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path / "facts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(facts), indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path
