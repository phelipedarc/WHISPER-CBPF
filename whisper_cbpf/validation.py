"""Inference-validation tools — recovery, posterior-predictive checks, and simulation-based calibration.

These quantify whether a fit **recovered the truth** and whether its **uncertainties are reliable**:

* :func:`recovery_metrics` — per-parameter bias, standardized z-score, and credible-interval coverage
  against a known truth (for synthetic-data recovery tests).
* :func:`posterior_predictive_check` — a posterior-predictive band, the reduced χ² of the best fit, and
  a Bayesian posterior-predictive *p*-value (a well-calibrated fit gives reduced χ² ≈ 1 and *p* ≈ 0.5).
* :func:`sbc_rank` / :func:`sbc_ranks` — Simulation-Based Calibration (Talts et al. 2018; Säilynoja et
  al. 2022): the rank of each true value within its posterior is **uniform** iff the posterior is
  calibrated. A χ²-of-uniformity *p*-value flags over-/under-confidence.

All operate on a :class:`~whisper_labia.samplers.base.SamplerResult` (or its ``.samples`` DataFrame) plus
the model + data, so they work identically for every sampler.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as _chi2_dist

from .models import get_model


def recovery_metrics(result, truth):
    """Per-parameter recovery of a known ``truth`` (dict) from a fit ``result``.

    Returns ``{param: {...}}`` with the posterior ``median``/``mean``/``std``, the 68% (16–84) and 95%
    (2.5–97.5) credible intervals, the ``bias`` (median − true), the standardized **``z_score``**
    (bias / std; ``|z| ≲ 2`` means recovered), and boolean 68%/95% ``within`` coverage. Also a top-level
    ``_summary`` with ``max_abs_z``, ``coverage68``/``coverage95`` (fraction of parameters covered), and
    ``rms_z``.
    """
    samples = result.samples if hasattr(result, "samples") else result
    names = [p for p in truth if p in samples.columns]
    out, z_all, c68, c95 = {}, [], [], []
    for p in names:
        s = np.asarray(samples[p], dtype=float)
        med, mean, std = float(np.median(s)), float(np.mean(s)), float(np.std(s, ddof=1))
        lo95, lo68, hi68, hi95 = np.percentile(s, [2.5, 16.0, 84.0, 97.5])
        t = float(truth[p])
        z = (med - t) / std if std > 0 else float("nan")
        w68, w95 = bool(lo68 <= t <= hi68), bool(lo95 <= t <= hi95)
        out[p] = {"true": t, "median": med, "mean": mean, "std": std,
                  "ci68": [float(lo68), float(hi68)], "ci95": [float(lo95), float(hi95)],
                  "bias": med - t, "z_score": z, "within_68": w68, "within_95": w95}
        z_all.append(z); c68.append(w68); c95.append(w95)
    finite = [abs(z) for z in z_all if np.isfinite(z)]
    out["_summary"] = {
        "max_abs_z": float(max(finite)) if finite else float("nan"),
        "rms_z": float(np.sqrt(np.mean(np.square(finite)))) if finite else float("nan"),
        "coverage68": float(np.mean(c68)) if c68 else float("nan"),
        "coverage95": float(np.mean(c95)) if c95 else float("nan"),
        "n_params": len(names),
    }
    return out


def posterior_predictive_check(result, lc, model=None, *, n_draws=300, time_grid=None, seed=0):
    """Posterior-predictive check for a flux-space fit.

    Draws ``n_draws`` posterior samples, forward-models each, and returns a predictive **band** (2.5/16/
    50/84/97.5 percentiles on ``time_grid``), the **reduced χ²** at the posterior median, and a **Bayesian
    posterior-predictive p-value** using the χ² discrepancy (fraction of replicated datasets with χ² ≥ the
    observed χ², per draw). Healthy fit ⇒ reduced χ² ≈ 1 and *p* ≈ 0.5 (near 0/1 flags mis-fit).
    """
    model = get_model(model if model is not None else result.model)
    lc = lc.add_flux()
    t_data = np.asarray(lc.time, dtype=float)
    bands_data = np.asarray(lc.band)
    obs = np.asarray(lc.flux, dtype=float)
    err = np.asarray(lc.flux_err, dtype=float)
    names = list(model.parameters)
    samples = result.samples if hasattr(result, "samples") else result
    S = samples[names].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(S), size=min(int(n_draws), len(S)), replace=len(S) < int(n_draws))
    draws = S[idx]

    # Smooth model band on a grid (parameter uncertainty only) — for plotting.
    if time_grid is None:
        time_grid = np.linspace(float(t_data.min()), float(t_data.max()), 200)
    grid_bands = np.full(time_grid.shape, bands_data[0] if len(bands_data) else "x")
    preds_grid = np.array([model.predict(dict(zip(names, row)), time_grid, grid_bands) for row in draws])
    band = np.percentile(preds_grid, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)

    # Goodness-of-fit at the BEST-FIT point (max-likelihood / min-distance) — measures whether the model
    # can fit the data, decoupled from posterior width (the median of a broad posterior can mis-fit).
    best = getattr(result, "best_params", None)
    point = best if best else {p: float(np.median(S[:, j])) for j, p in enumerate(names)}
    m_point = np.asarray(model.predict(point, t_data, bands_data), dtype=float)
    dof = max(len(obs) - len(names), 1)
    reduced_chi2 = float(np.sum(((obs - m_point) / err) ** 2) / dof)

    # Posterior-PREDICTIVE at the data times, WITH observation noise: y_rep = M(θ) + N(0, err). A
    # calibrated predictive contains ~68%/95% of the observed points -> the clean PPC calibration metric.
    preds_data = np.array([model.predict(dict(zip(names, row)), t_data, bands_data) for row in draws])
    y_rep = preds_data + rng.normal(0.0, err, size=preds_data.shape)
    lo68, hi68 = np.percentile(y_rep, [16.0, 84.0], axis=0)
    lo95, hi95 = np.percentile(y_rep, [2.5, 97.5], axis=0)
    cov68 = float(np.mean((obs >= lo68) & (obs <= hi68)))
    cov95 = float(np.mean((obs >= lo95) & (obs <= hi95)))
    # Bayesian χ² p-value (Gelman): compare observed vs replicated discrepancy per draw (≈0.5 healthy;
    # tends low for posteriors much wider than the noise — read alongside reduced_chi2 and coverage).
    chi2_obs = np.sum(((obs - preds_data) / err) ** 2, axis=1)
    chi2_rep = np.sum(((y_rep - preds_data) / err) ** 2, axis=1)
    p_value = float(np.mean(chi2_rep >= chi2_obs))

    return {"time_grid": time_grid, "lo95": band[0], "lo68": band[1], "median": band[2],
            "hi68": band[3], "hi95": band[4], "reduced_chi2": reduced_chi2, "dof": int(dof),
            "ppc_coverage68": cov68, "ppc_coverage95": cov95, "bayesian_p_value": p_value,
            "n_draws": int(len(draws))}


def sbc_rank(samples, true_value):
    """SBC rank of ``true_value`` among ``M`` posterior draws: the count of draws below it, in ``[0, M]``.
    Calibrated inference ⇒ these ranks are Uniform over ``{0, …, M}`` across many prior realizations."""
    return int(np.sum(np.asarray(samples, dtype=float) < float(true_value)))


def sbc_ranks(ranks_by_param, *, n_bins=20):
    """Simulation-Based Calibration diagnostics from collected ranks.

    ``ranks_by_param`` maps each parameter to its array of ``L`` ranks (each in ``[0, M]``, from
    :func:`sbc_rank` over ``L`` prior realizations). For each parameter returns the rank **histogram**, a
    **χ²-of-uniformity p-value** (low ⇒ mis-calibrated: a ∪-shape = over-confident/too-narrow, a ∩-shape =
    under-confident/too-wide, a slope = biased), and the sorted **fractional ranks** (rank/M) + empirical
    CDF for an ECDF-difference plot. Also a top-level ``_summary`` with the minimum p-value + verdict.
    """
    out, pvals = {}, []
    for p, r in ranks_by_param.items():
        r = np.asarray(r, dtype=float)
        L = len(r)
        M = int(np.max(r)) if L else 0
        nb = int(min(n_bins, max(M + 1, 2)))
        counts, _ = np.histogram(r, bins=nb, range=(-0.5, M + 0.5))
        expected = L / nb
        chi2_stat = float(np.sum((counts - expected) ** 2 / expected)) if expected > 0 else float("nan")
        pval = float(_chi2_dist.sf(chi2_stat, nb - 1)) if np.isfinite(chi2_stat) else float("nan")
        u = np.sort(r / M) if M > 0 else np.sort(r)
        out[p] = {"n_realizations": L, "M": M, "n_bins": nb, "counts": counts.tolist(),
                  "expected": float(expected), "chi2": chi2_stat, "uniformity_p": pval,
                  "frac_ranks": u.tolist(), "ecdf": (np.arange(1, L + 1) / L).tolist()}
        pvals.append(pval)
    finite_p = [q for q in pvals if np.isfinite(q)]
    min_p = float(min(finite_p)) if finite_p else float("nan")
    out["_summary"] = {"min_uniformity_p": min_p,
                       "calibrated": bool(min_p >= 0.05) if np.isfinite(min_p) else None,
                       "n_params": len(pvals)}
    return out
