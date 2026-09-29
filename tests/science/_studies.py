"""The science studies behind the slow tests, written so a test and a rerun share one code path.

Each study returns plain records (dicts). When ``WHISPER_VALIDATION_OUT`` names a directory, every
record is also appended there as one JSON line (``<study>.jsonl``), so a long study can be read
while it runs, resumed after an interruption, and summarised for ``docs/VALIDATION.md``.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

import _sim

#: Interval levels whose coverage the recovery study measures.
LEVELS = (0.68, 0.95)
#: Posterior draws the SBC rank is taken among (ranks run 0..SBC_DRAWS).
SBC_DRAWS = 49


# ------------------------------------------------------------------------------------ plumbing
def out_dir():
    """The directory records are written to (``WHISPER_VALIDATION_OUT``), or None."""
    d = os.environ.get("WHISPER_VALIDATION_OUT")
    if not d:
        return None
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def record(study, rec):
    """Append ``rec`` to ``<out>/<study>.jsonl`` when an output directory is set."""
    d = out_dir()
    if d is not None:
        with open(d / f"{study}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=_plain) + "\n")
    return rec


def done(study, key):
    """Records of ``study`` already written for ``key`` (to resume a long run)."""
    d = out_dir()
    if d is None or not (d / f"{study}.jsonl").exists():
        return []
    out = []
    for line in (d / f"{study}.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("key") == key:
            out.append(r)
    return out


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    return repr(v)


def rng_for(study, family, i):
    """A seeded generator per (study, family, injection): reruns draw the same alerts."""
    return np.random.default_rng([zlib.crc32(study.encode()), zlib.crc32(family.encode()), i])


def gpu_visible():
    """Whether JAX in this process runs on a GPU."""
    try:
        import jax
        return any(d.platform in ("gpu", "cuda") for d in jax.devices())
    except Exception:                                          # noqa: BLE001 - no JAX, no GPU
        return False


_GPU_PROBE = None


def gpu_for_subprocess():
    """Whether a fresh Python process (without ``JAX_PLATFORMS=cpu``) sees a GPU."""
    global _GPU_PROBE
    if _GPU_PROBE is None:
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
        try:
            out = subprocess.run(
                [sys.executable, "-c", "import jax; print(any(d.platform in ('gpu', 'cuda') "
                 "for d in jax.devices()))"],
                env=env, capture_output=True, text=True, timeout=300)
            _GPU_PROBE = out.stdout.strip().endswith("True")
        except Exception:                                      # noqa: BLE001
            _GPU_PROBE = False
    return _GPU_PROBE


def fit_auto(lc, model, *, sampler="auto", **kw):
    """``wp.fit`` with the recommended sampler (``sampler="auto"``), warnings silenced."""
    import whisper_cbpf as wp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.fit(lc, model, sampler=sampler, **kw)


def alert_lc(alert, *, explosion_known):
    """The alert as a user loads it (LSST preset), with the explosion date set when known."""
    import whisper_cbpf as wp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = wp.load_lightcurve(alert["packet"], survey="lsst", redshift=alert["redshift"])
    return lc.set_explosion_date(alert["event_mjd"]) if explosion_known else lc


def log_likelihood_at(result, lc, model, theta):
    """The fit's own log-likelihood at ``theta`` (a dict), on the rows the fit used."""
    import jax.numpy as jnp
    import whisper_cbpf as wp

    rows = result.fitted_lc(lc) if hasattr(result, "fitted_lc") else lc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ld = wp.log_density(rows, model, space=result.info.get("space", "auto"))
        x = jnp.asarray([float(theta[k]) for k in ld.names])
        return float(ld.log_likelihood(x))


# ------------------------------------------------------------------------ (a) known-answer study
def recovery(family, n, *, study="recovery", explosion_known=True, fit_kwargs=None,
             score_truth=True, texp_window="data"):
    """Inject ``n`` prior draws of ``family`` as LSST alerts, fit each, score the intervals.

    Returns one record per injection: truth, 2.5/16/50/84/97.5 % quantiles, whether the 68 % and
    95 % central intervals hold the truth, the SBC rank, the convergence verdict, the sampler's
    best ln L and the ln L at the truth.

    With the explosion date unknown the explosion time is fitted, over ``texp_window``:
    ``"data"`` is the default window (``lc.explosion_time_prior()``); ``"wide30"`` is the 30 days
    before the first detection.
    """
    fit_kwargs = dict(fit_kwargs or {})
    key = f"{family}|known={explosion_known}|{json.dumps(fit_kwargs, sort_keys=True)}"
    if not explosion_known and texp_window != "data":
        key += f"|window={texp_window}"
    recs = {r["i"]: r for r in done(study, key)}
    for i in range(n):
        if i in recs:
            continue
        rng = rng_for(study, family, i)
        t0 = time.perf_counter()
        alert = _sim.simulate_alert(family, rng, object_id=i + 1)
        lc = alert_lc(alert, explosion_known=explosion_known)
        truth = dict(alert["truth"])
        fit_kw = dict(fit_kwargs)
        if explosion_known:
            model = _sim.build_model(family, lc.band, alert["redshift"])
        else:
            window = _texp_prior(lc, texp_window)
            model = _sim.build_model(family, lc.band, alert["redshift"], free=["t_exp"],
                                     prior=window,
                                     t_exp_days=float(window.distributions["t_exp"].bounds[1]))
            if texp_window != "data":           # a prior passed to fit() is used as given
                fit_kw["prior"] = model.default_prior
            truth["t_exp"] = alert["event_mjd"]
        t_sim = time.perf_counter() - t0
        t0 = time.perf_counter()
        try:
            res = fit_auto(lc, model, seed=i, **fit_kw)
        except Exception as exc:                              # noqa: BLE001 - counted, reported
            recs[i] = record(study, {"key": key, "i": i, "family": family,
                                     "redshift": alert["redshift"],
                                     "n_detections": alert["n_detections"],
                                     "failed": f"{type(exc).__name__}: {exc}"[:300],
                                     "fit_s": round(time.perf_counter() - t0, 2)})
            continue
        t_fit = time.perf_counter() - t0
        rec = {"key": key, "i": i, "family": family, "redshift": alert["redshift"],
               "n_detections": alert["n_detections"], "n_data": int(res.n_data),
               "n_rows": len(lc), "excluded_pre_event": res.info.get("excluded_pre_event"),
               "sampler": res.sampler, "space": res.info.get("space"),
               "converged": bool(res.info.get("converged")),
               "problems": list(res.info.get("convergence_problems") or []),
               "fit_s": round(t_fit, 2), "sim_s": round(t_sim, 2),
               "max_log_likelihood": float(res.max_log_likelihood), "params": {}}
        if not explosion_known:
            rec["t_exp_prior"] = list(_texp_prior(lc).distributions["t_exp"].bounds)
            rec["t_exp_fit_prior"] = list(window.distributions["t_exp"].bounds)
        for k in model.parameters:
            s = res.samples[k].to_numpy(dtype=float)
            q = np.quantile(s, [0.025, 0.16, 0.5, 0.84, 0.975])
            rec["params"][k] = {"truth": float(truth[k]), "q": q.tolist(),
                                "in68": _sim.interval_hits(s, truth[k], 0.68),
                                "in95": _sim.interval_hits(s, truth[k], 0.95),
                                "rank": _sim.sbc_rank(s, truth[k], SBC_DRAWS)}
        if score_truth:
            try:
                rec["log_likelihood_truth"] = log_likelihood_at(res, lc, model, truth)
            except Exception as exc:                          # noqa: BLE001 - recorded
                rec["log_likelihood_truth_error"] = f"{type(exc).__name__}: {exc}"[:300]
        recs[i] = record(study, rec)
    return [recs[i] for i in range(n)]


def _texp_prior(lc, window="data"):
    """The explosion-time prior: the default data window, or the 30 days before the first
    detection (``"wide30"``)."""
    from whisper_cbpf.priors import Prior, Uniform

    if window == "data":
        return Prior({"t_exp": lc.explosion_time_prior()})
    first = float(lc.meta["first_detection_mjd"])
    return Prior({"t_exp": Uniform(first - 30.0, first)})


def summarize_recovery(recs, *, only_converged=False):
    """Coverage per level and SBC uniformity, pooled over the parameters of every injection.

    A fit that raised is counted in ``n_failed`` (with its errors) and left out of the rest.
    """
    failed = [r for r in recs if "failed" in r]
    fitted = [r for r in recs if "failed" not in r]
    use = [r for r in fitted if r["converged"] or not only_converged]
    out = {"n_injections": len(recs), "n_failed": len(failed),
           "failures": sorted({r["failed"] for r in failed}),
           "n_scored": len(use), "n_converged": sum(r["converged"] for r in fitted),
           "coverage": {}, "per_param": {}}
    names = list(use[0]["params"]) if use else []
    for lev in LEVELS:
        tag = f"in{int(lev * 100)}"
        hits = [p[tag] for r in use for p in r["params"].values()]
        n = len(hits)
        lo, hi = _sim.binomial_band(lev, n)
        out["coverage"][lev] = {"n": n, "rate": float(np.mean(hits)) if n else float("nan"),
                                "band": [lo, hi]}
    ranks = [p["rank"] for r in use for p in r["params"].values()]
    p, counts = _sim.rank_uniformity_p(ranks, SBC_DRAWS) if ranks else (float("nan"), [])
    out["sbc"] = {"n": len(ranks), "p": p, "counts": counts}
    for k in names:
        ks = [r["params"][k] for r in use]
        out["per_param"][k] = {"in68": float(np.mean([x["in68"] for x in ks])),
                               "in95": float(np.mean([x["in95"] for x in ks])),
                               "rank_p": _sim.rank_uniformity_p([x["rank"] for x in ks],
                                                                SBC_DRAWS)[0]}
    gaps = [r["log_likelihood_truth"] - r["max_log_likelihood"] for r in use
            if "log_likelihood_truth" in r and math.isfinite(r["max_log_likelihood"])]
    out["truth_beats_best_draw"] = {
        "n": len(gaps), "by_more_than_1": int(sum(g > 1.0 for g in gaps)),
        "by_more_than_5": int(sum(g > 5.0 for g in gaps)),
        "max": float(max(gaps)) if gaps else float("nan")}
    out["fit_s_median"] = float(np.median([r["fit_s"] for r in use])) if use else float("nan")
    return out


# ------------------------------------------------------------------ (b) model selection studies
#: One cadence for every class, so the cadence itself says nothing about the class.
SELECTION_CADENCE = dict(t_start=-10.0, t_end=60.0, gap=(1.0, 2.0))
#: The classes of the confusion matrix: a supernova, a TDE and a kilonova.
SELECTION_CANDIDATES = ("arnett", "tde", "kilonova")


def candidate_models(lc, redshift, names):
    """What ``compare`` is given for each candidate, with the explosion time unknown.

    Supernova and TDE family names go in as names (``compare`` binds them: free explosion time
    over the data window, known redshift). A kilonova is built the same way by hand.
    """
    from whisper_cbpf.priors import Prior

    out = []
    for nm in names:
        if nm.startswith("kilonova"):
            pr = lc.explosion_time_prior()
            out.append(_sim.build_model(nm, lc.band, redshift, free=["t_exp"],
                                        prior=Prior({"t_exp": pr}),
                                        t_exp_days=float(pr.bounds[1])))
        else:
            out.append(nm)
    return out


def compare_quiet(lc, models, **kw):
    import whisper_cbpf as wp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.compare(lc, models, **kw)


def comparison_record(cmp):
    """The numbers of a Comparison that the studies keep."""
    rows = []
    for r in cmp.table.itertuples(index=False):
        rows.append({"model": r.model, "status": r.status, "n_params": _int(r.n_params),
                     "n_data": _int(r.n_data), "max_log_likelihood": float(r.max_log_likelihood),
                     "bic": float(r.bic), "delta": float(r.delta), "weight": float(r.weight),
                     "grade": r.grade, "converged": r.converged,
                     "left_out_reason": r.left_out_reason})
    sampler_ll = {m: float(res.max_log_likelihood) for m, res in cmp.results.items()}
    return {"winner": cmp.winner, "criterion": cmp.criterion, "n_data": _int(cmp.n_data),
            "headline": repr(cmp).splitlines()[0], "table": rows,
            "sampler_max_log_likelihood": sampler_ll,
            "gain": {m: float(p.gain) for m, p in cmp.peaks.items()},
            "timing": cmp.timing}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def selection(truth_family, n, *, candidates=SELECTION_CANDIDATES, study="selection",
              min_detections=12, early=None):
    """Simulate ``n`` good-SNR alerts of ``truth_family`` and rank ``candidates`` on each.

    With ``early=k`` each alert is cut at its k-th detection first (an alert just raised).
    """
    key = f"{truth_family}|{','.join(candidates)}|early={early}"
    recs = {r["i"]: r for r in done(study, key)}
    for i in range(n):
        if i in recs:
            continue
        rng = rng_for("selection", truth_family, i)       # the same alerts, early or not
        alert = _sim.simulate_alert(truth_family, rng, min_detections=min_detections,
                                    min_bands=3, cadence=SELECTION_CADENCE, object_id=i + 1)
        packet = alert["packet"] if early is None else _sim.truncate_packet(alert["packet"],
                                                                            early)
        lc = alert_lc(dict(alert, packet=packet), explosion_known=False)
        models = candidate_models(lc, alert["redshift"], candidates)
        t0 = time.perf_counter()
        rec = {"key": key, "i": i, "truth": truth_family, "redshift": alert["redshift"],
               "n_detections": int((~lc.upper_limit).sum()), "n_rows": len(lc)}
        try:
            cmp = compare_quiet(lc, models, evidence_check=False, seed=i)
        except Exception as exc:                              # noqa: BLE001 - every fit failed
            rec.update(winner=None, headline=f"compare raised {type(exc).__name__}: {exc}"[:300],
                       table=[], failed=True, wall_s=round(time.perf_counter() - t0, 1))
            recs[i] = record(study, rec)
            continue
        rec.update(wall_s=round(time.perf_counter() - t0, 1), **comparison_record(cmp))
        recs[i] = record(study, rec)
    return [recs[i] for i in range(n)]


#: The first seed word of the stability study's alert (see :func:`rng_for`): the alert behind the
#: numbers of docs/VALIDATION.md section 5.3.
STABILITY_SEED_WORD = 2993559556


def likelihood_max_stability(truth_family, models, seeds, *, i=0,
                             study="likelihood_max_stability", fit_kwargs=None,
                             explosion_known=True):
    """One alert, one comparison per seed: rank by the sampler's best draw and by the peak.

    The explosion date is known by default, so the comparison is well posed and what varies
    from seed to seed is the samplers' noise alone.
    """
    fit_kwargs = dict(fit_kwargs or {})
    key = (f"{truth_family}|{i}|{','.join(models)}|known={explosion_known}|"
           f"{json.dumps(fit_kwargs, sort_keys=True)}")
    recs = {r["seed"]: r for r in done(study, key)}
    rng = np.random.default_rng([STABILITY_SEED_WORD, zlib.crc32(truth_family.encode()), i])
    alert = _sim.simulate_alert(truth_family, rng, min_detections=12, min_bands=3,
                                object_id=i + 1)
    lc = alert_lc(alert, explosion_known=explosion_known)
    for s in seeds:
        if s in recs:
            continue
        cmp = compare_quiet(lc, candidate_models(lc, alert["redshift"], models),
                            evidence_check=False, seed=s, **fit_kwargs)
        recs[s] = record(study, {"key": key, "seed": s, "truth": truth_family,
                                 **comparison_record(cmp)})
    return [recs[s] for s in seeds]


def ranking_by(rec, which):
    """Order of the ranked models by BIC from the likelihood maximum ("optimised") or the best draw
    ("sampler")."""
    rows = [r for r in rec["table"] if r["status"] == "ranked"]
    if which == "optimised":
        bic = {r["model"]: r["bic"] for r in rows}
    else:
        bic = {r["model"]: -2.0 * rec["sampler_max_log_likelihood"][r["model"]]
               + r["n_params"] * math.log(r["n_data"]) for r in rows}
    return sorted(bic, key=bic.get), bic


def summarize_likelihood_max_stability(recs):
    """Seed-to-seed spread of each model's ln L and of the BIC gaps, by either criterion.

    Only the models ranked in every run are compared; ``not_always_ranked`` lists the others.
    """
    ranked = [{x["model"] for x in r["table"] if x["status"] == "ranked"} for r in recs]
    models = [x["model"] for x in recs[0]["table"] if all(x["model"] in s for s in ranked)]
    out = {"n_seeds": len(recs), "models": {}, "orders": {}, "winners": {}, "gaps": {},
           "not_always_ranked": sorted(set().union(*ranked) - set(models))}
    recs = [dict(r, table=[x for x in r["table"] if x["model"] in models]) for r in recs]
    for m in models:
        ls = [r["sampler_max_log_likelihood"][m] for r in recs]
        lp = [next(x["max_log_likelihood"] for x in r["table"] if x["model"] == m)
              for r in recs]
        out["models"][m] = {"sampler_sd": float(np.std(ls, ddof=1)),
                            "optimised_sd": float(np.std(lp, ddof=1)),
                            "sampler_range": float(np.ptp(ls)),
                            "optimised_range": float(np.ptp(lp)),
                            "min_gain": float(np.min(np.subtract(lp, ls))),
                            "mean_gain": float(np.mean(np.subtract(lp, ls))),
                            "max_gain": float(np.max(np.subtract(lp, ls)))}
    for which in ("sampler", "optimised"):
        orders = [tuple(ranking_by(r, which)[0]) for r in recs]
        out["orders"][which] = len(set(orders))
        out["winners"][which] = sorted({o[0] for o in orders})
        bics = [ranking_by(r, which)[1] for r in recs]
        ref = min(models, key=lambda m: np.mean([b[m] for b in bics]))
        out["gaps"][which] = {m: float(np.std([b[m] - b[ref] for b in bics], ddof=1))
                              for m in models if m != ref}
    return out


# ------------------------------------------------------------------------------- (f) speed
#: The five families of the per-alert budget (supernovae and a TDE, bound by name).
SPEED_FAMILIES = ("arnett", "magnetar", "shock_cooling_arnett", "csm_shock_arnett", "tde")
#: The bundled LSST alert (a supernova at z = 0.1; ``data/make_alerts.py``).
ALERT_FILE = Path(__file__).parent / "data" / "lsst_alert_sn.json"


def bundled_alert():
    """The bundled alert as a user loads it: LSST preset, known redshift, explosion unknown."""
    import whisper_cbpf as wp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.load_lightcurve(ALERT_FILE, survey="lsst", redshift=0.1)


def speed_compare(label, *, families=SPEED_FAMILIES, repeats=2):
    """Wall seconds of a default ``compare`` of ``families`` on the bundled alert, ``repeats``
    times in this process (the first pays the compiles)."""
    import jax

    lc = bundled_alert()
    out = {"label": label, "device": jax.devices()[0].platform, "families": list(families),
           "n_rows": len(lc), "runs": []}
    for rep in range(repeats):
        t0 = time.perf_counter()
        cmp = compare_quiet(lc, list(families), evidence_check=False, seed=rep)
        out["runs"].append({"wall_s": round(time.perf_counter() - t0, 1),
                            "timing": cmp.timing, "winner": cmp.winner,
                            "n_data": _int(cmp.n_data)})
    return record("speed", out)


def speed_fit_batch(label, k=64, *, nsteps=5000):
    """Wall seconds of one ``fit_batch`` call over ``k`` simulated supernova alerts (free
    explosion time, known redshift 0.1), default walkers and steps."""
    import jax
    import whisper_cbpf as wp
    from whisper_cbpf.priors import Prior

    lcs, priors = [], []
    for i in range(k):
        alert = _sim.simulate_alert("arnett", rng_for("speed", "arnett", i), redshift=0.1,
                                    min_detections=12, min_bands=3, object_id=i + 1)
        lc = alert_lc(alert, explosion_known=False)
        lcs.append(lc)
        priors.append(lc.explosion_time_prior())
    model = _sim.build_model("arnett", _sim.WFD_BANDS, 0.1, free=["t_exp"],
                             prior=Prior({"t_exp": priors[0]}), t_exp_days=_sim.MJD_EVENT)
    per_lc = [Prior({**model.default_prior.distributions, "t_exp": p}) for p in priors]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = wp.fit_batch(lcs, model, prior=per_lc, nsteps=nsteps, burnin=nsteps // 5)
    wall = time.perf_counter() - t0
    return record("speed", {"label": label, "device": jax.devices()[0].platform, "k": k,
                            "nsteps": nsteps, "wall_s": round(wall, 1),
                            "per_alert_s": round(wall / k, 2),
                            "batch_info": _plain_info(fits[0].info.get("batch"))})


def _plain_info(d):
    try:
        return json.loads(json.dumps(d, default=_plain))
    except Exception:                                          # noqa: BLE001
        return repr(d)
