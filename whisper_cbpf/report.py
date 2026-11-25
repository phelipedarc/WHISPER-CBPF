"""One self-contained HTML page for a model comparison: the answer, how sure it is, and why.

:func:`report` turns a comparison (``wp.compare``) into one HTML file that opens anywhere, offline
and on a phone: the ranking, the facts (:mod:`whisper_cbpf.facts`) as tables, the figures of the plot
kit, every diagnostic with pass or fail, forecasts when asked for, and the provenance. Styles are
inline and figures are embedded as base64 PNG; the page loads nothing and runs no script.

The page shows results and how to read them, nothing else. Every number on it comes from
:func:`whisper_cbpf.facts.comparison_facts`, so the page and the facts file agree, and the page
carries that file as a download. Its sentences are fixed templates filled from those numbers;
an optional "decision" slot shows a follow-up decision supplied by the caller, labelled as such.

The same inputs give the same bytes (no time stamp, no path, seeded draws, PNGs written without a
software tag), so a report rebuilt from saved results is identical to the first one.
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np

__all__ = ["report", "REPORT_NAME"]

#: File name used when ``out`` is a directory.
REPORT_NAME = "report.html"
#: Resolution of the embedded figures.
FIGURE_DPI = 110
#: Forecast tables are shown for this many models, best first.
FORECAST_MODELS = 3
#: Rows of the "where the models differ most" table.
DISCRIMINATE_ROWS = 5

_CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#5f6368;--line:#dadce0;--card:#f6f8fa;--ok:#137333;
--bad:#b3261e;--accent:#1a4f8b}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#e8eaed;--muted:#9aa0a6;
--line:#3c4043;--card:#1e1f21;--ok:#81c995;--bad:#f28b82;--accent:#8ab4f8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:60rem;margin:0 auto;padding:1rem}
a{color:var(--accent)}
h1{font-size:1.5rem;margin:.5rem 0 .25rem}
h2{font-size:1.2rem;margin:2rem 0 .5rem;border-bottom:1px solid var(--line);padding-bottom:.25rem}
h3{font-size:1rem;margin:1.25rem 0 .4rem}
p{margin:.5rem 0}
.sub,.note,figcaption{color:var(--muted);font-size:.9rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.75rem 1rem;
margin:.75rem 0}
.lead{font-size:1.05rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.5rem 0}
table{border-collapse:collapse;font-size:.88rem;min-width:100%}
th,td{padding:.3rem .5rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;
white-space:nowrap}
th{font-weight:600;background:var(--card)}
td.wrap{white-space:normal;min-width:16rem}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.pass{color:var(--ok);font-weight:600}
.fail{color:var(--bad);font-weight:600}
.none{color:var(--muted)}
figure{margin:1rem 0}
figure img{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--line);
border-radius:4px}
ul{padding-left:1.2rem;margin:.4rem 0}
li{margin:.2rem 0}
code{font-size:.85em;word-break:break-all}
nav{font-size:.9rem;margin:.5rem 0 1rem}
nav a{margin-right:.8rem;white-space:nowrap}
footer{color:var(--muted);font-size:.8rem;margin:2.5rem 0 1rem;border-top:1px solid var(--line);
padding-top:.5rem}
""".strip()

_SECTIONS = (("answer", "Answer"), ("ranking", "Ranking"), ("figures", "Figures"),
             ("parameters", "Parameters"), ("data", "Data"), ("diagnostics", "Diagnostics"),
             ("forecast", "Forecast"), ("provenance", "Provenance"))


# ========================================================================================= format
def _e(x):
    return html.escape(str(x), quote=True)


def _na(x):
    return x is None or (isinstance(x, float) and not np.isfinite(x))


def _f(x, digits=2):
    return "n/a" if _na(x) else f"{float(x):.{digits}f}"


def _g(x, digits=3):
    return "n/a" if _na(x) else f"{float(x):.{digits}g}"


def _weight(w):
    if _na(w):
        return "n/a"
    if w < 0.001:
        return "&lt;0.001"
    if w > 0.999:
        return "&gt;0.999"
    return f"{w:.3f}"


def _flag(value, yes="yes", no="no"):
    if value is None:
        return '<span class="none">not checked</span>'
    return f'<span class="fail">{yes}</span>' if value else f'<span class="pass">{no}</span>'


def _verdict(passed):
    if passed is None:
        return '<span class="none">not checked</span>'
    return '<span class="pass">pass</span>' if passed else '<span class="fail">FAIL</span>'


def _table(headers, rows, numeric=()):
    """An HTML table in a horizontally scrolling box. Cells are HTML already escaped."""
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cls = ' class="num"' if i in numeric else (' class="wrap"' if i == len(row) - 1
                                                       and len(str(cell)) > 60 else "")
            cells.append(f"<td{cls}>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>'
            + "".join(body) + "</tbody></table></div>")


def _sentence(text, keep=()):
    """``text`` as a sentence: a capital first letter (not when it opens with a model name in
    ``keep``, which is shown as spelled) and a final full stop. Not escaped."""
    text = str(text).strip()
    if not text:
        return text
    if not any(text.startswith(str(name)) for name in keep):
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def _model_names(facts):
    """Every model name in the facts (ranked, fitted or left out), longest first."""
    names = set(facts["models"]) | {r["model"] for r in facts["ranking"]["left_out"]}
    return sorted(names, key=lambda n: (-len(n), n))


def _pm(value, err, digits=2):
    if _na(value):
        return "n/a"
    return f"{float(value):.{digits}f} &#177; {_f(err, digits)}"


def _interval(rec, digits=3):
    return f"{_g(rec.get('p16'), digits)} to {_g(rec.get('p84'), digits)}"


# ======================================================================================== figures
def _png(make):
    """Run a plot function and return its figure as base64 PNG (the figure is then closed)."""
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    try:
        axes = make()
        fig = axes if hasattr(axes, "savefig") else \
            np.ravel(np.asarray(axes, dtype=object))[0].figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=FIGURE_DPI, bbox_inches="tight",
                    metadata={"Software": None})
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        for num in set(plt.get_fignums()) - before:
            plt.close(num)


def _free_columns(result, facts_model):
    params = facts_model.get("parameters") or {}
    keep = []
    for name, rec in params.items():
        if name in result.samples.columns:
            x = result.samples[name].to_numpy(dtype=float)
            if np.isfinite(x).all() and np.ptp(x) > 0:
                keep.append(name)
    logs = [p for p in keep if (params[p].get("coordinate") == "log10")]
    return keep, logs


def _figures(comparison, lc, facts, forecasts):
    """``(figures, missing)``: ``[(key, caption, png)]`` and ``[(caption, reason)]``."""
    from . import plotting

    winner = facts["ranking"]["winner"]
    res_w = comparison.results.get(winner) if winner else None
    specs = [("models", "The data and the posterior prediction of every model, with residuals.",
              "plot_models", lambda fn: fn(comparison, lc)),
             ("ranking", "The ranking: each model's weight and its difference to the winner.",
              "plot_model_comparison", lambda fn: fn(comparison))]
    if res_w is not None and res_w.n_samples > 0:
        cols, logs = _free_columns(res_w, facts["models"].get(winner, {}))
        specs.append(("widths", f"Posterior width over prior width, per parameter of {winner} "
                      f"(near 1: the prior, not the data, sets it).", "plot_widths",
                      lambda fn: fn(res_w)))
        if cols:
            specs.append(("corner", f"Posterior of {winner} (log10 axes for log-uniform "
                          f"parameters).", "plot_corner",
                          lambda fn: fn([res_w], labels=[winner], parameters=cols,
                                        log_params=logs)))
    if forecasts and winner in forecasts:
        specs.append(("forecast", f"Forecast of {winner}: predicted magnitude per band, median "
                      f"with 68 and 95 % intervals.", "plot_forecast",
                      lambda fn: fn(forecasts[winner])))
    figures, missing = [], []
    for key, caption, name, call in specs:
        fn = getattr(plotting, name, None)
        if fn is None:
            reason = f"whisper_cbpf.plotting.{name} is not part of this installation"
        else:
            try:
                figures.append((key, caption, _png(lambda: call(fn))))
                continue
            except Exception as exc:                          # noqa: BLE001 - reported below
                first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
                first = re.sub(r" at 0x[0-9a-fA-F]+", "", first)     # same text on every build
                reason = f"{name} raised {type(exc).__name__}: {first}"[:300]
        warnings.warn(f"report: the '{key}' figure is left out because {reason}. The rest of the "
                      f"report is written.", UserWarning, stacklevel=3)
        missing.append((caption, reason))
    return figures, missing


# ====================================================================================== forecasts
def _forecasts(comparison, lc, facts, times):
    """``(forecasts, discrimination, reason)`` for the leading models; ``reason`` if none."""
    from .forecast import discriminate, forecast

    bands = [b for b in facts["data"]["bands"]
             if (facts["data"].get("per_band") or {}).get(b, {}).get("n_detections")]
    ranked = [m["model"] for m in facts["ranking"]["models"]][:FORECAST_MODELS]
    if not bands or not ranked:
        return None, None, "not enough data: no ranked model or no band with a detection"
    models = getattr(comparison, "models", None) or {}
    try:
        fcs = {name: forecast(comparison.results[name], times, bands, lc=lc,
                              model=models.get(name) if not isinstance(models.get(name), str)
                              else None)
               for name in ranked}
        disc = discriminate(fcs, times, bands) if len(fcs) >= 2 else None
    except Exception as exc:                                  # noqa: BLE001 - shown on the page
        reason = re.sub(r" at 0x[0-9a-fA-F]+", "", f"{type(exc).__name__}: {exc}")[:300]
        warnings.warn(f"report: no forecast, because {reason}. The rest of the report is "
                      f"written.", UserWarning, stacklevel=3)
        return None, None, f"no forecast: {reason}"
    return fcs, disc, None


# ======================================================================================= sections
def _answer(facts, decision):
    rk = facts["ranking"]
    cav = facts["caveats"]
    with_ = cav["models_with"]
    winner = rk["winner"]
    names = _model_names(facts)
    out = ['<section id="answer"><h2>Answer</h2><div class="card">',
           f'<p class="lead">{_e(_sentence(rk["reading"], names))}</p>']
    if winner:
        out.append(f'<p class="sub">Criterion: {_e(rk["criterion"])}. Ranked models: '
                   f'{len(rk["models"])}; left out: {len(rk["left_out"])}.</p>')
    items = []
    wc = facts["models"].get(winner, {}).get("caveats", {}) if winner else {}
    if winner and wc.get("converged") is False:
        items.append(_e(_sentence(f"The winner's fit ({winner}) did not pass its convergence "
                                  f"checks: " + "; ".join(wc.get("diagnostics_failed") or []))))
    labels = (("not_converged", "Did not pass the convergence checks"),
              ("stranded_walkers", "Stranded walkers or chains"),
              ("not_enough_data", "Not enough data (at least as many free parameters as data "
               "points)"),
              ("no_posterior_draws", "No posterior draws"),
              ("large_likelihood_max_gain", "The sampler stopped more than "
               f"{facts['thresholds']['likelihood_max_gain_large']:g} nat below the likelihood "
               "peak "
               "(the ranking uses the peak; the posterior may be in another region)"),
              ("data_differs_from_fit", "The light curve given here is not the one the fit "
               "recorded"))
    for key, text in labels:
        if with_.get(key):
            items.append(f"{text}: {_e(', '.join(with_[key]))}.")
    for key, text in (("prior_dominated_parameters", "prior-dominated (quote the prior, not a "
                       "measurement)"),
                      ("at_prior_edge_parameters", "piled against a prior bound (quote a limit)")):
        if wc.get(key):
            items.append(f"Parameters of {_e(winner)} {text}: {_e(', '.join(wc[key]))}.")
    for row in rk["left_out"]:
        items.append(f"Left out of the ranking: {_e(row['model'])} ({_e(row['reason'])}).")
    if cav.get("winner_differs_from_comparison"):
        items.append(f"By its own criterion the comparison ranks {_e(winner)} first, but names "
                     f"{_e(rk['comparison_winner'])} as its winner.")
    items += [_e(_sentence(p, names)) for p in cav.get("comparison_problems") or []]
    if items:
        out.append("<p>Read with care:</p><ul>" + "".join(f"<li>{i}</li>" for i in items)
                   + "</ul>")
    else:
        out.append('<p class="pass">No caveat was raised.</p>')
    out.append("</div>")
    if decision is not None:
        out.append('<div class="card"><h3>Decision (supplied by the caller, not computed by '
                   'whisper)</h3>')
        if isinstance(decision, dict):
            rows = [[_e(k), _e(v if isinstance(v, str) else json.dumps(v, sort_keys=True))]
                    for k, v in decision.items()]
            out.append(_table(["", ""], rows))
        else:
            out.append(f"<p>{_e(decision)}</p>")
        out.append("</div>")
    out.append("</section>")
    return out


def _ranking(facts):
    rk = facts["ranking"]
    lnz = rk["criterion"] == "ln Z"
    rows = []
    for m in rk["models"]:
        lnz_cell = "n/a" if m["log_evidence"] is None else _pm(m["log_evidence"],
                                                                 m["log_evidence_err"])
        grade = "winner" if m["rank"] == 1 else \
            f"{_e(m['grade_winner_over_this'])} (ln B {_f(m['ln_b_winner_over_this'])})"
        rows.append([str(m["rank"]), f"<b>{_e(m['model'])}</b>", _e(m["sampler"]),
                     _e(m["n_params"]), _e(m["n_data"]), _f(m["max_log_likelihood"]),
                     _f(m["bic"], 1), _f(m["delta_bic"], 1), lnz_cell, _weight(m["weight"]),
                     grade, _flag(None if m["converged"] is None else not m["converged"],
                                  yes="no", no="yes"),
                     _verdict(m["diagnostics_passed"])])
    out = ['<section id="ranking"><h2>Ranking</h2>']
    if rows:
        out.append(_table(["Rank", "Model", "Sampler", "k", "n", "ln L (peak)", "BIC", "ΔBIC",
                           "ln Z", "Weight", "Grade (winner over it)", "Converged",
                           "All checks"], rows, numeric=(0, 3, 4, 5, 6, 7, 9)))
    else:
        out.append(f"<p>{_e(_sentence(rk['reading'], _model_names(facts)))}</p>")
    check = rk["evidence_check"]
    if check.get("ran"):
        verdict = {True: "agrees with BIC", False: "disagrees with BIC",
                   None: "is inconclusive"}[check.get("agrees_with_bic")]
        extra = "" if check.get("ln_b") is None else \
            f" (ln Z difference {_f(check['ln_b'])} &#177; {_f(check['ln_b_err'])})"
        out.append(f"<p>Evidence check: nested sampling on {_e(', '.join(check['models']))} "
                   f"{verdict}{extra}: {_e((check.get('reason') or '').rstrip('.'))}.</p>")
    if rk.get("grade_note"):
        out.append(f"<p>The winner's grade is {_e(rk['grade_note'])}.</p>")
    if rk["left_out"]:
        out.append("<h3>Left out of the ranking</h3>")
        out.append(_table(["Model", "Reason"], [[_e(r["model"]), _e(r["reason"])]
                                                 for r in rk["left_out"]]))
    cuts = facts["thresholds"]["grade_cuts_ln_b"]
    basis = ("the models are ranked by their evidence ln Z (higher is better); the weight is "
             "exp(ln Z), normalised" if lnz else
             "the models are ranked by BIC = -2 ln L + k ln n at the optimised likelihood maximum "
             "(lower is better); the weight is exp(-BIC/2), normalised")
    note = (f"How to read it: {basis}, the probability of each model if all were equally likely "
            f"beforehand. ΔBIC is BIC minus the winner's. The grade is Jeffreys' scale on the "
            f"winner's ln Bayes factor over that model (ln Z difference, or ΔBIC/2): below "
            f"{cuts[0]:.2f} inconclusive, below {cuts[1]:.2f} substantial, below {cuts[2]:.2f} "
            f"strong, above decisive.")
    if any(c["narrowing"]["redshift"]["fitted"] for c in facts["models"].values()):
        note += (" The redshift is fitted: a model can move the source to buy a better fit, so "
                 "read the redshift narrowing under Parameters before trusting the ranking.")
    out.append(f'<p class="note">{_e(note)}</p></section>')
    return out


def _parameters(facts):
    rk = facts["ranking"]
    order = [m["model"] for m in rk["models"]]
    order += sorted(m for m in facts["models"] if m not in order)
    out = ['<section id="parameters"><h2>Parameters</h2>',
           '<p class="note">Median and 16th&#8211;84th percentile of each posterior. Width / '
           'prior: the posterior\'s 68 % width over the prior\'s, in the prior\'s own coordinate '
           f'(log10 for log-uniform priors); above '
           f'{facts["thresholds"]["prior_dominated_ratio"]:.2f} the prior carries more '
           'information than the data.</p>']
    narrow = []
    for name in order:
        core = facts["models"].get(name)
        if core is None:
            continue
        out.append(f"<h3>{_e(name)}</h3>")
        for line in core["readings"]:
            out.append(f"<p>{_e(_sentence(line))}</p>")
        if core["caveats"]["not_enough_data"]:
            out.append('<p class="note">No parameter values are shown: with at least as many '
                       'free parameters as data points the posterior mostly restates the prior '
                       '(the numbers are in facts.json, flagged).</p>')
            continue
        rows, notes = [], []
        for p, rec in core["parameters"].items():
            prior = rec.get("prior") or {}
            prior_txt = "n/a" if not prior else (
                f"{_e(prior['type'])} [{_g(prior.get('low'))}, {_g(prior.get('high'))}]"
                if prior["type"] in ("Uniform", "LogUniform") else
                f"{_e(prior['type'])} ({_g(prior.get('mu'))}, {_g(prior.get('sigma'))})"
                + ("" if prior["type"] != "TruncatedNormal" else
                   f" on [{_g(prior.get('low'))}, {_g(prior.get('high'))}]"))
            flags = []
            if rec.get("prior_dominated"):
                flags.append('<span class="fail">prior-dominated</span>')
            if rec.get("at_prior_edge"):
                flags.append(f'<span class="fail">at {_e(rec["edge_side"])} bound</span>')
            if rec.get("prior_dominated") or rec.get("at_prior_edge"):
                notes.append(rec["reading"])
            if not flags:
                flags = ['<span class="none">no finite draw</span>' if rec.get("median") is None
                         else '<span class="none">not checked (no prior)</span>'
                         if rec.get("prior_dominated") is None
                         else '<span class="pass">measured</span>']
            rows.append([f"<b>{_e(p)}</b>", _g(rec.get("median")), _interval(rec),
                         _g(rec.get("peak")), prior_txt, _f(rec.get("width_ratio")),
                         " ".join(flags)])
        if rows:
            out.append(_table(["Parameter", "Median", "16th–84th", "Peak", "Prior",
                               "Width / prior", "Reading"], rows, numeric=(1, 3, 5)))
        if core["fixed"]:
            out.append('<p class="note">Held fixed: ' + ", ".join(
                f"{_e(k)} = {_g(v, 6)}" for k, v in core["fixed"].items()) + ".</p>")
        if notes:
            out.append("<ul>" + "".join(f"<li>{_e(_sentence(n, core['parameters']))}</li>"
                                        for n in notes) + "</ul>")
        for pname, rec in core["narrowing"].items():
            if rec["fitted"] and rec.get("prior"):
                narrow.append([_e(name), _e(pname),
                               f"{_g(rec['prior']['p16'])} to {_g(rec['prior']['p84'])}",
                               f"{_g(rec['posterior']['p16'])} to {_g(rec['posterior']['p84'])}",
                               _f(rec["width_ratio"]), _f(rec["narrowing_factor"], 1),
                               _flag(rec["prior_dominated"])])
    out.append("<h3>Redshift and explosion time</h3>")
    if narrow:
        out.append(_table(["Model", "Parameter", "Prior 16th–84th", "Posterior 16th–84th",
                           "Width / prior", "Narrowed ×", "Prior-dominated"], narrow,
                          numeric=(4, 5)))
    else:
        some = next(iter(facts["models"].values()), None)
        z = (some or {}).get("narrowing", {}).get("redshift", {})
        text = ("Neither the redshift nor the explosion time is a fitted parameter"
                + (f"; the redshift is {_g(z.get('value'), 6)} ({_e(z.get('source'))})"
                   if z.get("value") is not None else "") + ".")
        out.append(f'<p class="note">{text}</p>')
    out.append("</section>")
    return out


def _rate(rec):
    if rec is None:
        return "n/a"
    if rec.get("value") is None:
        return '<span class="none">not enough data</span>'
    return f"{rec['value']:.3f} &#177; {rec['error']:.3f}"


def _data(facts):
    d = facts["data"]
    axis = d["time_axis"]
    ref = ("MJD" if axis["reference"] == "MJD" else
           f"days since {axis['reference']}" + ("" if axis["reference_mjd"] is None else
                                                f" (MJD {axis['reference_mjd']:.3f})"))
    out = ['<section id="data"><h2>What the data show</h2>',
           f'<p>{d["n_points"]} points in {len(d["bands"])} band(s) '
           f'({_e(", ".join(d["bands"]))}): {d.get("n_detections", 0)} detections and '
           f'{d["n_upper_limits"]} upper limits. Times are {_e(ref)}.</p>']
    if d.get("reading"):
        out.append(f"<p>{_e(_sentence(d['reading']))}</p></section>")
        return out
    out.append(f'<p>First detection at {_f(d["first_detection_time"], 3)}, last at '
               f'{_f(d["last_detection_time"], 3)} ({_f(d["span_days"], 2)} days).')
    last = d.get("last_nondetection_before_first_detection")
    if last:
        out.append(f' Last non-detection before it: {_e(last["band"])} fainter than '
                   f'{_f(last["limiting_magnitude"])} mag at {_f(last["time"], 3)}.')
    out.append("</p>")
    rows = []
    for b, rec in d["per_band"].items():
        if "brightest" not in rec:
            rows.append([f"<b>{_e(b)}</b>", _g(rec["effective_wavelength_angstrom"], 5), "0",
                         str(rec["n_upper_limits"]), "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        br, la = rec["brightest"], rec["last_detection"]
        rows.append([f"<b>{_e(b)}</b>", _g(rec["effective_wavelength_angstrom"], 5),
                     str(rec["n_detections"]), str(rec["n_upper_limits"]),
                     f"{_pm(br['magnitude'], br['error'])} at {_f(br['time'], 2)}",
                     _e(rec["state"]), _rate(rec["rise_rate"]), _rate(rec["decline_rate"]),
                     f"{_pm(la['magnitude'], la['error'])} at {_f(la['time'], 2)}"])
    out.append(_table(["Band", "λ eff (Å)", "Detections", "Limits", "Brightest (mag)",
                       "State", "Rise (mag/d)", "Decline (mag/d)", "Last detection (mag)"],
                      rows, numeric=(1, 2, 3)))
    out.append('<p class="note">Rates: weighted least-squares slope of the detections up to '
               '(rise) and from (decline) the brightest one; brightening and fading are '
               'positive. State: rising (brightest is the last detection), declining (the '
               'first), peaked (in between).</p>')
    out.append("<h3>Latest colours</h3>")
    if d.get("colours"):
        rows = []
        for c in d["colours"]:
            label = f"{c['bands'][0]} &#8722; {c['bands'][1]}"
            if c["colour"] is None:
                rows.append([label, f'<span class="none">{_e(c["reason"])}</span>', "", ""])
            else:
                rows.append([label, _pm(c["colour"], c["error"]), _f(c["time"], 2),
                             _f(c["separation_days"], 3)])
        out.append(_table(["Colour", "mag", "At", "Pair separation (d)"], rows,
                          numeric=(2, 3)))
    else:
        out.append(f"<p>{_e(d.get('colours_reading', 'No colour.'))}</p>")
    winner = facts["ranking"]["winner"]
    absm = (facts["models"].get(winner) or {}).get("absolute_magnitude") if winner else None
    if not absm:
        absm, winner = facts.get("absolute_magnitude"), None      # the light curve's own distance
    out.append("<h3>Absolute magnitude</h3>")
    if not absm or not absm.get("available"):
        why = (absm or {}).get("reading", "not enough data: no distance")
        out.append(f"<p>{_e(_sentence(why))}</p>")
    else:
        dm = absm["distance_modulus"]
        z = absm.get("redshift") or {}
        uncertain = absm["redshift_kind"] in ("posterior", "prior")
        src = _e(absm["distance_source"])
        if absm["redshift_kind"] == "posterior":
            src += f" of {_e(winner)}"
        if uncertain:
            src += (f" (z = {_g(z['median'])}, 16th&#8211;84th percentile {_g(z['p16'])} to "
                    f"{_g(z['p84'])})")
            out.append(f"<p>From {src}: distance modulus {_f(dm['value'])} mag, up to "
                       f"{_f(dm['brighter_by'])} larger and {_f(dm['fainter_by'])} smaller "
                       f"within the 68 % range of the redshift.</p>")
        else:
            out.append(f"<p>From {src}: distance modulus {_f(dm['value'])} mag (a known "
                       f"distance, so the only uncertainty is the photometric one).</p>")
        rows = []
        for b, r in absm["per_band"].items():
            bounds = r["range_at_prior_bounds"]
            row = [f"<b>{_e(b)}</b>", _pm(r["apparent_magnitude"], r["error"]), _f(r["value"])]
            if uncertain:
                row += [f"&#8722;{_f(r['brighter_by'])} / +{_f(r['fainter_by'])}",
                        "n/a" if not bounds else f"{_f(bounds[0])} to {_f(bounds[1])}"]
            rows.append(row)
        headers = ["Band", "Brightest (mag)", "Absolute (mag)"]
        if uncertain:
            headers += ["68 % range: brighter / fainter", "At the prior's bounds"]
        out.append(_table(headers, rows, numeric=(2,)))
        note = _e(_sentence(absm["note"]))
        if uncertain and abs(dm["fainter_by"] - dm["brighter_by"]) > 0.005:
            note += (f" The range is not symmetric: the distance modulus is not linear in the "
                     f"redshift, so quote it as &#8722;{_f(dm['brighter_by'])} / "
                     f"+{_f(dm['fainter_by'])} mag, not as one &#177; number.")
        z_fit = ((facts["models"].get(winner) or {}).get("narrowing") or {}).get("redshift") \
            if absm["redshift_kind"] == "posterior" else None
        if z_fit and z_fit.get("prior_dominated"):
            note += (f" The redshift of {_e(winner)} is prior-dominated, so this range restates "
                     f"the redshift prior; it is not a measured distance.")
        out.append(f'<p class="note">{note}</p>')
    out.append("</section>")
    return out


def _diagnostics(facts):
    rk = facts["ranking"]
    order = [m["model"] for m in rk["models"]]
    order += sorted(m for m in facts["models"] if m not in order)
    out = ['<section id="diagnostics"><h2>Diagnostics</h2>']
    for name in order:
        core = facts["models"].get(name)
        if core is None:
            continue
        diag = core["diagnostics"]
        verdict = '<span class="pass">PASSED</span>' if diag["passed"] else \
            '<span class="fail">FAILED</span>'
        out.append(f"<h3>{_e(name)} ({_e(diag['sampler'])}): {verdict}</h3>")
        rows = [[_e(r["check"]), _e(_value(r["value"])), _e(r["threshold"]),
                 _verdict(r["passed"]), _e(r["reason"])] for r in diag["rows"]]
        out.append(_table(["Check", "Value", "Threshold", "Result", "Reason"], rows,
                          numeric=(1,)))
    out.append("</section>")
    return out


def _value(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _forecast_section(forecasts, disc, reason, times):
    out = ['<section id="forecast"><h2>Forecast</h2>']
    if times is None:
        out.append('<p class="note">No forecast was asked for (pass forecast_times= to '
                   'report).</p></section>')
        return out
    if forecasts is None:
        out.append(f"<p>{_e(_sentence(reason))}</p></section>")
        return out
    out.append('<p class="note">Predicted AB magnitude over the posterior draws that predict '
               'light: mean &#177; sd and 16th&#8211;84th percentile; "no light" is the '
               'fraction of draws predicting none.</p>')
    for name, fc in forecasts.items():
        out.append(f"<h3>{_e(name)}</h3>")
        rows = [[_f(r["time"], 2), _e(r["band"]), _pm(r["mean_mag"], r["sd_mag"]),
                 f"{_f(r['q16'])} to {_f(r['q84'])}", _f(r["frac_dark"])]
                for r in fc.to_dict(orient="records")]
        out.append(_table(["Time", "Band", "Mean (mag)", "16th–84th", "No light"], rows,
                          numeric=(0, 4)))
    if disc is not None:
        best = disc.attrs.get("best")
        out.append("<h3>Where the models differ most</h3>")
        if best:
            out.append(f"<p>The most telling observation: {_e(best['band'])} at "
                       f"{_f(best['time'], 2)}, where {_e(best['model_a'])} and "
                       f"{_e(best['model_b'])} differ by D = {_f(best['D'])} (their gap in "
                       f"units of the combined uncertainty; 3 or more tells them apart).</p>")
        else:
            out.append(f"<p>{_e(disc.attrs.get('reason') or 'No observable cell.')}</p>")
        top = disc.sort_values(["observable", "D"], ascending=[False, False],
                               kind="mergesort").head(DISCRIMINATE_ROWS)
        rows = [[_e(r["model_a"]), _e(r["model_b"]), _f(r["time"], 2), _e(r["band"]),
                 _f(r["mean_a"]), _f(r["mean_b"]), _f(r["D"]), _e("yes" if r["observable"]
                                                                   else "no")]
                for r in top.to_dict(orient="records")]
        out.append(_table(["Model A", "Model B", "Time", "Band", "Mean A", "Mean B", "D",
                           "Observable"], rows, numeric=(2, 4, 5, 6)))
    out.append("</section>")
    return out


def _provenance(comparison, facts, missing):
    import whisper_cbpf

    rk = facts["ranking"]
    order = [m["model"] for m in rk["models"]]
    order += sorted(m for m in facts["models"] if m not in order)
    first = next((comparison.results[m] for m in order if comparison.results.get(m) is not None),
                 None)
    prov = (getattr(first, "provenance", None) or {}) if first is not None else {}
    env = prov.get("whisper") or {}
    git = env.get("git") or {}
    out = ['<section id="provenance"><h2>Provenance</h2>']
    lines = [f"whisper_cbpf {_e(env.get('version', whisper_cbpf.__version__))}"
             + (f", commit <code>{_e(str(git.get('commit', ''))[:12])}</code>"
                + (" (with local changes)" if git.get("dirty") else "") if git else "")]
    if prov.get("python"):
        lines.append(f"Python {_e(prov['python'])}")
    pkgs = {k: v for k, v in (prov.get("packages") or {}).items() if v}
    if pkgs:
        lines.append("Packages: " + ", ".join(f"{_e(k)} {_e(v)}" for k, v in sorted(pkgs.items())))
    data = facts["data"]
    lines.append(f"Data: {_e(data.get('name') or 'unnamed')}, {data['n_points']} points, SHA-256 "
                 f"<code>{_e(facts['inputs']['data_sha256'])}</code>")
    out.append("<ul>" + "".join(f"<li>{x}</li>" for x in lines) + "</ul>")
    rows = []
    for name in order:
        res = comparison.results.get(name)
        if res is None:
            continue
        p = getattr(res, "provenance", None) or {}
        sampler = p.get("sampler") or {}
        settings = json.dumps(sampler.get("kwargs") or {}, sort_keys=True, separators=(",", ":"),
                              default=str)
        prior = ((p.get("model") or {}).get("prior") or {}).get("repr", "n/a")
        rows.append([f"<b>{_e(name)}</b>", _e(res.sampler), _e(sampler.get("seed")),
                     f"<code>{_e(settings)}</code>", _f(res.runtime_s, 1), _e(prior)])
    out.append(_table(["Model", "Sampler", "Seed", "Settings", "Run time (s)", "Prior"], rows,
                      numeric=(4,)))
    th = facts["thresholds"]
    out.append("<h3>Thresholds behind the flags</h3>")
    out.append(_table(["Threshold", "Value"], [[_e(k), _e(v if not isinstance(v, float)
                                                            else f"{v:.6g}")]
                                                for k, v in th.items()]))
    blob = json.dumps(facts, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    href = "data:application/json;base64," + base64.b64encode(blob.encode("utf-8")).decode()
    out.append(f'<p>Every number on this page is in the facts file, with the rules behind it: '
               f'<a download="facts.json" href="{href}">facts.json</a> (SHA-256 '
               f'<code>{_e(facts["sha256"])}</code>).</p>')
    if missing:
        out.append("<p>Figures not shown:</p><ul>" + "".join(
            f"<li>{_e(c)} ({_e(r)})</li>" for c, r in missing) + "</ul>")
    out.append("</section>")
    return out


def _page(title, facts, figures, sections):
    import whisper_cbpf

    rk = facts["ranking"]
    d = facts["data"]
    sub = (f"{d.get('n_detections', 0)} detections and {d['n_upper_limits']} upper limits in "
           f"{', '.join(d['bands'])}; {len(rk['models'])} "
           f"{'model' if len(rk['models']) == 1 else 'models'} ranked by {rk['criterion']}")
    nav = "".join(f'<a href="#{k}">{_e(v)}</a>' for k, v in _SECTIONS)
    figs = ['<section id="figures"><h2>Figures</h2>']
    if not figures:
        figs.append('<p class="note">No figure could be drawn (see Provenance).</p>')
    for key, caption, png in figures:
        figs.append(f'<figure id="fig-{_e(key)}"><img alt="{_e(caption)}" '
                    f'src="data:image/png;base64,{png}"><figcaption>{_e(caption)}</figcaption>'
                    f'</figure>')
    figs.append("</section>")
    answer, ranking, params, data, diag, fc, prov = sections
    body = (answer + ranking + figs + params + data + diag + fc + prov)
    return "\n".join([
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="color-scheme" content="light dark">',
        f"<title>{_e(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body><main>",
        f"<header><h1>{_e(title)}</h1><p class=\"sub\">{_e(sub)}</p></header>",
        f"<nav>{nav}</nav>",
        *body,
        f"<footer>Made with whisper_cbpf {_e(whisper_cbpf.__version__)}. Every number is "
        f"computed by a stated rule (facts.json).</footer>",
        "</main></body>",
        "</html>",
        ""])


# ========================================================================================= public
def report(comparison, lc, out, *, forecast_times=None, decision=None, title=None):
    """Write one self-contained HTML report of a model comparison and return its path.

    The page holds the answer and its caveats, the ranking table, the figures of the plot kit, the
    facts (:func:`whisper_cbpf.facts.comparison_facts`) as tables (parameters with their prior
    widths and flags, redshift and explosion-time narrowing, what the photometry shows per band,
    colours, absolute magnitude), every diagnostic with pass or fail, forecasts when asked for, and
    the provenance with the facts file as a download. Styles are inline and figures embedded, so it
    opens offline and on a phone; the same inputs give the same bytes.

    Parameters
    ----------
    comparison : Comparison
        From ``wp.compare(lc, models)``, or ``Comparison.load(dir)``.
    lc : LightCurve or None
        The data the models were fit to; ``None`` uses ``comparison.lc``.
    out : str or Path
        The HTML file (``.html``), or a directory that gets ``report.html``. Parent directories
        are made; an existing file is replaced.
    forecast_times : array_like, optional
        Epochs (on the light curve's clock) at which to forecast the leading models in every band
        with a detection, and to find where they differ most.
    decision : str or dict, optional
        Text for a "decision" card: a follow-up decision supplied by the caller, shown
        labelled as such. whisper never fills it.
    title : str, optional
        Page title. Default: "<object name>: model comparison".

    Returns
    -------
    Path
        The report file.

    Raises
    ------
    TypeError
        ``comparison`` is not a comparison, or there is no light curve.

    Warns
    -----
    UserWarning
        A figure could not be drawn (it is left out and listed under Provenance), or the forecast
        failed (its reason is shown in the Forecast section).

    Examples
    --------
    >>> import tempfile
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.report import report
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = wp.get_model("flare").predict(
    ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["ztfr"] * 40, flux=flux, flux_err=np.full(40, 0.1),
    ...                    name="toy flare")
    >>> cmp = wp.compare(lc, ["flare", "bazin"], sampler="nested", nlive=100)
    >>> path = report(cmp, lc, tempfile.mkdtemp(), forecast_times=[32.0, 36.0])
    >>> path.name
    'report.html'
    >>> text = path.read_text()
    >>> "<script" in text, "http://" in text or "https://" in text
    (False, False)
    """
    from .facts import comparison_facts

    for attr in ("table", "results", "criterion"):
        if not hasattr(comparison, attr):
            raise TypeError(f"report expects a Comparison from wp.compare (with .table, .results, "
                            f".criterion); {type(comparison).__name__} has no .{attr}.")
    lc = lc if lc is not None else getattr(comparison, "lc", None)
    if lc is None or not (hasattr(lc, "colnames") and hasattr(lc, "meta")):
        raise TypeError("report needs the light curve the models were fit to: pass lc= (this "
                        "comparison does not carry one).")
    path = Path(out)
    if path.suffix.lower() not in (".html", ".htm"):
        path = path / REPORT_NAME
    facts = comparison_facts(comparison, lc)
    forecasts = disc = reason = None
    if forecast_times is not None:
        forecasts, disc, reason = _forecasts(comparison, lc, facts, forecast_times)
    figures, missing = _figures(comparison, lc, facts, forecasts)
    sections = (_answer(facts, decision), _ranking(facts), _parameters(facts), _data(facts),
                _diagnostics(facts), _forecast_section(forecasts, disc, reason, forecast_times),
                _provenance(comparison, facts, missing))
    name = lc.meta.get("name")
    page = _page(title or f"{name or 'Transient'}: model comparison", facts, figures, sections)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(page.encode("utf-8"))
    return path
