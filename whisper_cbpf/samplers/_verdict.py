"""The plain-English convergence verdict ``wp.fit`` gives after an MCMC fit.

:meth:`SamplerResult.diagnostics <whisper_cbpf.samplers.base.SamplerResult.diagnostics>` runs every
check; this module turns each failed check into one short sentence that names the metric in words,
the parameter, the value and the threshold, and says what to do. :func:`verdict_message` joins
them under a first line, :func:`verdict_heading`'s: ``"Not converged:"``, or ``"Check the
posterior:"`` when only checks of the posterior itself failed (a pile-up at a prior bound, too few
data points), which no longer run fixes. Numpy only: importing it costs nothing.
"""
from __future__ import annotations

import math
import re

#: The report kinds (``DiagnosticsReport.kind``) of the MCMC-type samplers: emcee ensembles
#: (``mcmc``, ``emcee_jax``) and NUTS (``nuts_gpu``, ``pymc_jax_gpu_*``).
MCMC_KINDS = ("ensemble", "nuts")

#: Checks that judge the posterior against its prior or the data, not the sampler's mixing: a
#: chain that passes every other check has converged even when these fail, and running longer
#: does not change them. When they are the only failures the verdict's first line is
#: ``"Check the posterior:"`` instead of ``"Not converged:"``.
POSTERIOR_CHECKS = ("prior-edge pile-up", "data points")

_ON_PARAM = re.compile(r"on '([^']+)'")
_EDGE = re.compile(r"(\d+%) of '([^']+)' at its (lower|upper) bound \(([^)]*)\)")


def _num(threshold):
    """The number in a threshold string such as ``"< 1.01"`` or ``">= 400 (largest tau)"``."""
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(threshold))
    return float(m.group(0)) if m else float("nan")


def _needs(threshold):
    """A threshold in words: ``">= 400"`` -> ``"at least 400"``, ``"<= 10 nats"`` -> ``"at most 10
    nats"``; ``"< 1.01"`` stays as it is."""
    t = str(threshold).split(" (")[0].strip()
    if t.startswith(">="):
        return "at least " + t[2:].strip()
    if t.startswith("<="):
        return "at most " + t[2:].strip()
    return t


def _rhat(v):
    """R-hat to the digits that show it is above 1.01: 1.18, 1.012."""
    return f"{v:.2f}" if v >= 1.1 else f"{v:.3f}"


def _param(row):
    m = _ON_PARAM.search(row.reason or "")
    return m.group(1) if m else None


def _steps(result):
    """The main run's length in steps and how to ask for more: ``(steps, keyword)``."""
    info = result.info if isinstance(result.info, dict) else {}
    uc = info.get("until_converged")
    if isinstance(uc, dict) and uc.get("enabled"):
        return int(uc.get("steps_run") or info.get("nsteps") or 0), "max_steps"
    if "num_samples" in info:
        return int(info.get("num_samples") or 0), "num_samples"
    return int(info.get("nsteps") or 0), "nsteps"


def _longer(result, target=None):
    """``"Run longer (max_steps=8000)"``: the keyword that lengthens this sampler's run, set to
    ``target`` steps (default twice the run). A run until converged that stopped at its time cap
    is lengthened by ``max_time_s`` (scaled from its own pace), and by ``max_steps`` too when the
    target is past that cap."""
    steps, kw = _steps(result)
    if steps <= 0:
        return "Run longer"
    target = max(int(math.ceil(target)) if target else 2 * steps, steps + 1)
    if str(result.sampler) == "emcee_jax" and kw == "nsteps":
        return f"Run longer (until_converged=True, or nsteps={target})"
    if kw == "max_steps":
        uc = result.info.get("until_converged") if isinstance(result.info, dict) else None
        uc = uc if isinstance(uc, dict) else {}
        if uc.get("stop_reason") == "max_time_s":
            chunks = uc.get("chunks") or [{}]
            elapsed = chunks[-1].get("elapsed_s") if isinstance(chunks[-1], dict) else None
            cap = uc.get("max_steps")
            more = f", max_steps={target}" if cap is not None and target > int(cap) else ""
            if elapsed is None or not elapsed > 0:
                return f"Run longer (a larger max_time_s{more})"
            secs = int(math.ceil(float(elapsed) * target / steps / 10.0) * 10)
            return f"Run longer (max_time_s={secs}{more})"
    return f"Run longer ({kw}={target})"


def _tau_param(result):
    """The parameter with the largest autocorrelation time, when the fit recorded them."""
    info = result.info if isinstance(result.info, dict) else {}
    taus = info.get("autocorr_time")
    fixed = info.get("fixed") or {}
    names = [p for p in result.parameters if p not in fixed]
    try:
        vals = [float(v) for v in taus]
    except (TypeError, ValueError):
        return None
    if len(vals) != len(names) or not any(math.isfinite(v) for v in vals):
        return None
    return names[max((i for i, v in enumerate(vals) if math.isfinite(v)), key=lambda i: vals[i])]


def _sentence(row, result, kind):
    """One plain sentence (or a few, for several parameters at a prior edge) for a failed row."""
    info = result.info if isinstance(result.info, dict) else {}
    check, v, thr = row.check, row.value, row.threshold
    p = _param(row)
    if check == "R-hat between independent ensembles (largest)":
        k = int(info.get("n_ensembles") or 0)
        if v is None:
            return [f"the {k} ensembles could not be compared on {p or 'a parameter'} "
                    f"(R-hat undefined, needs {thr}). {_longer(result)} and check that every ensemble "
                    f"moves."]
        return [f"the {k} ensembles disagree on {p} (R-hat {_rhat(v)}, needs {thr}). "
                f"{_longer(result)} or check for a second mode."]
    if check in ("split R-hat across walkers (largest)", "split R-hat (largest)"):
        who = "walkers" if "walkers" in check else "chains"
        if v is None and (row.reason or "").startswith("one chain"):
            return [f"R-hat could not be computed because there is only one {who[:-1]} (needs at "
                    f"least 2). Run more {who}."]
        if v is None:
            return [f"R-hat across the {who} could not be computed (needs {thr}). Install the "
                    f"[analysis] extra (arviz) and check the draws are finite."]
        what = ("the walkers disagree" if who == "walkers" else
                f"the {int(info.get('num_chains') or 0) or 'independent'} chains disagree")
        return [f"{what} on {p} (R-hat {_rhat(v)}, needs {thr}). {_longer(result)} or check for "
                f"{who} left in a second mode."]
    if check in ("bulk ESS (smallest)", "tail ESS (smallest)"):
        if v is None:
            return [f"the {check.split()[0]} effective sample size could not be computed (needs "
                    f"{_needs(thr)}). Install the [analysis] extra (arviz) and check the draws "
                    f"are finite."]
        need = _num(thr)
        steps = _steps(result)[0]
        target = steps * need / v if v and v > 0 and steps else None
        if check.startswith("bulk"):
            return [f"there are too few independent draws of {p} for its median and width (bulk "
                    f"ESS {v:.0f}, needs {_needs(thr)}). {_longer(result, target)}."]
        return [f"there are too few independent draws in the tails of {p} for its 16th and 84th "
                f"percentiles (tail ESS {v:.0f}, needs {_needs(thr)}). {_longer(result, target)}."]
    if check == "chain length / autocorrelation time":
        tau = info.get("max_autocorr_time")
        if v is None or tau is None or not math.isfinite(float(tau)):
            return [f"the chain is too short to estimate its autocorrelation time (needs at least "
                    f"{_num(thr):g} autocorrelation times). {_longer(result)}."]
        tp = _tau_param(result)
        on = f" on {tp}" if tp else ""
        return [f"the chain is too short for its autocorrelation time: {v:.1f} autocorrelation "
                f"times long (largest tau {float(tau):.0f} steps{on}, needs {_needs(thr)}). "
                f"{_longer(result, _num(thr) * float(tau))}."]
    if check == "stuck walkers":
        return [f"{v} of {info.get('nwalkers')} walkers are stuck more than 10 nats below the "
                f"best walker (needs none), and their draws widen the posterior. Start the walkers "
                f"around the likelihood peak (init=result.likelihood_max_opt(lc).params) or run "
                f"longer."]
    if check == "frozen walkers":
        return [f"{v} of {info.get('nwalkers')} walkers never moved across the kept draws "
                f"(needs none). {_longer(result)} or start the walkers closer to the posterior "
                f"(init=)."]
    if check == "prior-edge pile-up":
        piled = _EDGE.findall(row.reason or "")
        if not piled:
            return [f"draws pile up against a prior bound ({row.reason})."]
        return [f"{frac} of the {name} draws sit at its {side} prior bound ({bound}), needs at "
                f"most 5% in the outer 1% of the range, so the bound and not the data limits it. "
                f"Widen the prior if the bound is not physical, or report a limit."
                for frac, name, side, bound in piled]
    if check == "data points":
        k = int(result.n_params)
        return [f"there {'is' if v == 1 else 'are'} only {v} data point{'' if v == 1 else 's'} "
                f"for {k} free parameter{'' if k == 1 else 's'} (needs more points than "
                f"parameters), so the posterior mostly restates the prior. Wait "
                f"for more detections or fix some parameters."]
    if check == "posterior draws":
        return ["the sampler returned no posterior draws (needs at least one). Check the start "
                "(init=) and the prior."]
    if check == "divergences":
        if v is None:
            return ["divergent transitions were not recorded, so they cannot be ruled out (needs "
                    "0). Rerun with a sampler version that records them."]
        return [f"{v} transitions diverged (needs 0), so the draws may be biased. Raise "
                f"target_accept_prob or reparameterise."]
    if check == "log-likelihood R-hat":
        if v is None:
            return [f"the log-likelihood R-hat could not be computed (needs {thr}). "
                    f"{_longer(result)}."]
        return [f"the chains sit at different likelihood levels (log-likelihood R-hat {_rhat(v)}, "
                f"needs {thr}). {_longer(result)} or check for a second mode."]
    if check == "stranded chains":
        ids = ", ".join(str(c) for c in (info.get("stranded_chains") or []))
        return [f"{v} chain(s) ({ids}) are stranded more than 10 nats below the best chain (needs "
                f"none). Start from init_strategy='prior_scan' or from a previous result."]
    if check == "frozen chains":
        ids = ", ".join(str(c) for c in (info.get("frozen_chains") or []))
        return [f"{v} chain(s) ({ids}) froze and barely move (needs none). Enable x64 or shift "
                f"the clock."]
    if check == "gap to the prior scan's optimum":
        return [f"every draw is {v:.1f} nats below the prior scan's optimum (needs "
                f"{_needs(thr)}), so all chains missed the best region. Start from "
                f"init_strategy='prior_scan' or run longer."]
    if check == "gap to the likelihood maximum":
        return [f"the best draw is {v:.1f} nats below the optimised likelihood maximum (needs "
                f"{_needs(thr)}), so the sampler never reached the peak. Start from the likelihood "
                f"maximum (init=) and run again."]
    return [f"{check} failed ({row.reason})."]


def verdict_sentences(result, report):
    """One plain-English sentence per failed check of ``report`` (a ``DiagnosticsReport``).

    The sampler's own flag is used as a safety net: when it says not converged but no check of the
    report failed, its problems are given in one sentence.

    Parameters
    ----------
    result : SamplerResult
        The fit.
    report : DiagnosticsReport
        ``result.diagnostics()``.

    Returns
    -------
    list of str
        Empty when the fit passed every check.

    Examples
    --------
    >>> import pandas as pd
    >>> from whisper_cbpf.results import diagnose
    >>> from whisper_cbpf.samplers.base import SamplerResult
    >>> from whisper_cbpf.samplers._verdict import verdict_sentences
    >>> r = SamplerResult("mcmc", "toy", ["a"], pd.DataFrame({"a": [0.0, 1.0]}), {}, {}, 1, 1,
    ...                   0.1, {"nwalkers": 2})
    >>> verdict_sentences(r, diagnose(r))[0]                  # doctest: +ELLIPSIS
    'There is only 1 data point for 1 free parameter (needs more points than parameters), ...'
    """
    out = []
    for row in report.rows:
        if row.passed is False:
            out += _sentence(row, result, report.kind)
    info = result.info if isinstance(result.info, dict) else {}
    if not out and info.get("converged") is False:
        problems = "; ".join(info.get("convergence_problems") or []) or "no reason given"
        out.append(f"the sampler flagged a problem the checks did not catch: {problems}.")
    return [s[0].upper() + s[1:] for s in out]


def verdict_heading(result, report):
    """The verdict's first line: ``"Not converged:"``, or ``"Check the posterior:"`` when every
    failed check is one of :data:`POSTERIOR_CHECKS` and the sampler did not flag itself.

    A pile-up of draws at a prior bound (or a light curve with fewer points than parameters) is a
    property of the posterior, not a failure to converge: a longer run gives the same answer.

    Parameters
    ----------
    result : SamplerResult
        The fit.
    report : DiagnosticsReport
        ``result.diagnostics()``.

    Returns
    -------
    str

    Examples
    --------
    >>> import pandas as pd
    >>> from whisper_cbpf.results import DiagnosticRow, DiagnosticsReport
    >>> from whisper_cbpf.samplers.base import SamplerResult
    >>> from whisper_cbpf.samplers._verdict import verdict_heading
    >>> r = SamplerResult("mcmc", "toy", ["a"], pd.DataFrame({"a": [0.0, 1.0]}), {}, {}, 9, 1,
    ...                   0.1, {"converged": True})
    >>> edge = DiagnosticRow("prior-edge pile-up", 0.2, "<= 0.05", False, "20% of 'a' ...")
    >>> verdict_heading(r, DiagnosticsReport("mcmc", "toy", "ensemble", [edge]))
    'Check the posterior:'
    >>> tau = DiagnosticRow("chain length / autocorrelation time", 12.0, ">= 50", False, "short")
    >>> verdict_heading(r, DiagnosticsReport("mcmc", "toy", "ensemble", [edge, tau]))
    'Not converged:'
    """
    failed = [row.check for row in report.rows if row.passed is False]
    info = result.info if isinstance(result.info, dict) else {}
    if failed and all(c in POSTERIOR_CHECKS for c in failed) and info.get("converged") is not False:
        return "Check the posterior:"
    return "Not converged:"


def verdict_message(sentences, heading="Not converged:"):
    """``heading`` and one sentence per line, or ``None`` when there is nothing to say.

    Parameters
    ----------
    sentences : list of str
        :func:`verdict_sentences`'s.
    heading : str, default "Not converged:"
        The first line (:func:`verdict_heading`'s).

    Returns
    -------
    str or None

    Examples
    --------
    >>> from whisper_cbpf.samplers._verdict import verdict_message
    >>> print(verdict_message(["The chain is too short.", "Too few draws."]))
    Not converged:
    The chain is too short.
    Too few draws.
    >>> verdict_message([]) is None
    True
    >>> print(verdict_message(["Draws pile up at a bound."], heading="Check the posterior:"))
    Check the posterior:
    Draws pile up at a bound.
    """
    if not sentences:
        return None
    return "\n".join([heading, *sentences])
