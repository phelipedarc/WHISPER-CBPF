"""Bayesian model-selection metrics computed from a posterior sample.

**WAIC** — the Widely Applicable Information Criterion (Watanabe 2010; Gelman, Hwang & Vehtari 2014) —
is a fully-Bayesian alternative to AIC/BIC. Where AIC/BIC penalise a single best-fit point, WAIC uses
the **pointwise** log-likelihood averaged over the *whole posterior*, penalised by its posterior
variance (the effective number of parameters ``p_waic``). It therefore needs the posterior draws, not
just a point estimate — which is exactly why it complements (and is more honest than) a table of medians.

Lower WAIC is better. WAIC ``= -2 (lppd - p_waic)`` with
``lppd = Σ_i log mean_s p(y_i | θ_s)`` and ``p_waic = Σ_i Var_s log p(y_i | θ_s)``.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import logsumexp

from .likelihood import make_likelihood
from .models import get_model


def _resolve_samples(posterior, model):
    """Return (samples ndarray, column names or None, model name or None) from varied inputs."""
    import pandas as pd

    model_name = getattr(posterior, "model", None)
    if hasattr(posterior, "samples"):                          # SamplerResult
        df = posterior.samples
        return df.to_numpy(dtype=float), list(df.columns), model_name
    if isinstance(posterior, pd.DataFrame):
        return posterior.to_numpy(dtype=float), list(posterior.columns), model_name
    return np.asarray(posterior, dtype=float), None, model_name   # ndarray -> names from the model


def waic(posterior, lc, model=None, *, space="auto", likelihood="auto", fixed=None,
         max_samples=2000, seed=0):
    """Widely Applicable Information Criterion (WAIC) from a posterior sample.

    Parameters
    ----------
    posterior : SamplerResult | pandas.DataFrame | numpy.ndarray
        Posterior draws. A :class:`SamplerResult` uses its ``.samples`` (and ``.model`` when ``model``
        is omitted); a ``DataFrame`` uses its columns as parameter names; a bare array takes names from
        the model's ``parameters`` (so the column order must match).
    lc : LightCurve
        Data the model is scored against.
    model : str | Model, optional
        Model name/object; defaults to ``posterior.model``.
    space, likelihood : str
        Forwarded to :func:`~whisper_labia.likelihood.make_likelihood` (data space + likelihood kind).
    fixed : dict, optional
        Parameter values to merge into every draw — for parameters held fixed during the fit and so not
        present in the posterior columns (e.g. a pinned ``redshift``).
    max_samples : int
        Cap on posterior draws evaluated (one model call per draw — matters for slow simulators).
    seed : int
        Seed for subsampling when the posterior has more than ``max_samples`` draws.

    Returns
    -------
    dict
        ``waic`` (lower is better), ``lppd``, ``p_waic`` (effective #parameters), ``se`` (standard
        error of WAIC), ``n_samples``, ``n_data``.
    """
    samples, names, model_name = _resolve_samples(posterior, model)
    model = get_model(model if model is not None else model_name)
    if model is None:
        raise ValueError("No model given and the posterior has no .model; pass model=...")
    names = list(model.parameters) if names is None else names

    n_total = samples.shape[0]
    subsampled = n_total > max_samples
    if subsampled:
        sel = np.random.default_rng(seed).choice(n_total, size=max_samples, replace=False)
        samples = samples[sel]
        warnings.warn(f"WAIC: posterior has {n_total} draws; evaluated a random {max_samples}-draw "
                      "subsample (raise max_samples for the full set).", stacklevel=2)

    lik = make_likelihood(lc, kind=likelihood, space=space)
    if not hasattr(lik, "log_likelihood_pointwise"):
        raise TypeError(f"{type(lik).__name__} has no log_likelihood_pointwise(); WAIC needs the "
                        "pointwise log-likelihood. Use a Gaussian (with/without upper limits) likelihood.")
    times, bands = np.asarray(lc.time, float), np.asarray(lc.band)
    extra = dict(fixed or {})

    ll = np.vstack([
        np.asarray(lik.log_likelihood_pointwise(
            model.predict({**extra, **{nm: float(v) for nm, v in zip(names, row)}}, times, bands)), float)
        for row in samples])                                   # (n_samples, n_data)
    # Drop non-finite *draws* (rows), NOT data points (columns): WAIC is a sum over data points, so
    # every model in a comparison must be scored on the SAME points. Dropping a column when a single
    # bad draw makes it non-finite would shrink the dataset model-dependently and invalidate ΔWAIC.
    good = np.all(np.isfinite(ll), axis=1)
    n_dropped = int((~good).sum())
    if n_dropped:
        warnings.warn(f"WAIC: dropped {n_dropped}/{ll.shape[0]} posterior draws with a non-finite "
                      f"pointwise log-likelihood (all {ll.shape[1]} data points retained).", stacklevel=2)
    ll = ll[good]
    if ll.shape[0] < 2:
        raise ValueError("WAIC needs >= 2 posterior draws with finite pointwise log-likelihoods; "
                         f"only {ll.shape[0]} of {good.size} qualified.")

    n_samp = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(n_samp)            # log mean-likelihood per point
    p_waic_i = np.var(ll, axis=0, ddof=1)                      # posterior var of log-lik per point
    elpd_i = lppd_i - p_waic_i
    n_data = ll.shape[1]
    se = float(np.sqrt(n_data * np.var(-2.0 * elpd_i, ddof=1))) if n_data > 1 else float("nan")
    return {"waic": float(-2.0 * np.sum(elpd_i)), "lppd": float(np.sum(lppd_i)),
            "p_waic": float(np.sum(p_waic_i)), "se": se, "n_samples": int(n_samp), "n_data": int(n_data),
            "n_draws_dropped": n_dropped, "subsampled": bool(subsampled)}


def per_band_metrics(lc, model, params, *, space="auto", fixed=None):
    """Per-band goodness-of-fit residual metrics at a single parameter set (e.g. the best fit).

    Evaluates ``model`` at ``params`` on the observed ``(time, band)`` grid and, **per band**, reports
    the mean-squared error (MSE), root-mean-squared error (RMSE) and mean-absolute error (MAE) of the
    residuals ``observed - model``, computed in the Gaussian likelihood's **comparison space** — flux
    density [Jy] for a flux fit, apparent magnitude [mag] for a magnitude fit — so the metric matches
    the space the fit actually optimised. This is the deterministic point-estimate complement to the
    distributional :func:`waic` / posterior-predictive checks.

    Parameters
    ----------
    lc : LightCurve
        The observed data.
    model : str | Model
        Model name or object (its ``predict`` is called once at ``params``).
    params : dict
        Parameter values — typically ``result.best_params``. Only the model's own parameters are used;
        extra keys (e.g. a likelihood scatter term) are ignored.
    space : str
        ``'auto'`` | ``'flux'`` | ``'magnitude'`` — the residual space (default follows the data mode).
    fixed : dict, optional
        Parameters held fixed during the fit and absent from ``params`` (merged in before predicting).

    Returns
    -------
    dict
        ``{"space", "unit", "bands": {band: {"mse","rmse","mae","n"}}, "overall": {...}}``.
        ``unit`` is ``"Jy"`` (flux) or ``"mag"`` (magnitude).
    """
    from .likelihood import GaussianLikelihood

    m = get_model(model)
    if m is None:
        raise ValueError(f"Unknown model {model!r}.")
    lik = GaussianLikelihood(lc, space=space)
    times = np.asarray(lc.time, dtype=float)
    bands = np.asarray(lc.band).astype(str)
    p = dict(fixed or {})
    p.update({k: float(params[k]) for k in m.parameters if k in params})     # model params only
    model_flux = np.asarray(m.predict(p, times, bands), dtype=float)
    resid = np.asarray(lik.y, float) - np.asarray(lik.model_in_space(model_flux), float)

    def _stats(r):
        r = r[np.isfinite(r)]
        if r.size == 0:
            return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n": 0}
        mse = float(np.mean(r ** 2))
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(r))), "n": int(r.size)}

    per_band = {str(b): _stats(resid[bands == b]) for b in np.unique(bands)}
    unit = "mag" if lik.space == "magnitude" else "Jy"
    return {"space": lik.space, "unit": unit, "bands": per_band, "overall": _stats(resid)}


#: default nominal levels for the coverage-calibration curve.
DEFAULT_COVERAGE_LEVELS = (0.5, 0.68, 0.8, 0.9, 0.95, 0.99)


def predictive_metrics(result, lc, model=None, *, space="auto", likelihood="auto",
                       scatter_param="auto", fixed=None, levels=DEFAULT_COVERAGE_LEVELS,
                       n_draws=400, seed=0):
    r"""Posterior predictive / model-comparison metrics from a fit, for the JSON report.

    Evaluates the model across ``n_draws`` posterior samples once and returns, in one dict:

    * **rmse** — root-mean-squared error of ``observed - posterior_mean_prediction`` (per band + overall),
      in the fit's comparison space (Jy for flux, mag for magnitude).
    * **lpd** — log predictive density: ``sum_i log mean_s p(y_i | θ_s)`` (the ``lppd``; higher is
      better), with the per-point mean.
    * **elpd_loo** — expected log predictive density by **PSIS-LOO** cross-validation (Vehtari et al.
      2017; higher is better), with ``p_loo``, standard error ``se``, ``looic = -2·elpd_loo`` and the
      max Pareto-``k`` diagnostic (``k > 0.7`` ⇒ LOO estimate unreliable).
    * **waic** — WAIC on the deviance scale (``-2·elpd_waic``; lower is better), with ``elpd_waic``,
      ``p_waic`` and ``se``.
    * **aic** / **bic** — from the fit's best-fit likelihood (lower is better), carried through from
      ``result`` when available.
    * **coverage** — the calibration curve: for each nominal ``level`` the *empirical* fraction of
      observations inside the central posterior-predictive interval, overall and per band. A calibrated
      fit has empirical ≈ nominal at every level.

    **Scatter-aware predictive density.** The log-predictive quantities (LPD, WAIC, PSIS-LOO) and the
    coverage intervals must be evaluated under the *same* generative model the fit optimised — otherwise
    they score the data against a distribution the parameters were never tuned for. When the fit includes
    a **free extra-scatter term** :math:`\sigma` (Villar et al. 2017; a "jitter"/intrinsic-scatter
    parameter in the sense of Hogg, Bovy & Lang 2010), the pointwise density is the scatter-augmented
    Gaussian

    .. math::  p(y_i\mid\theta_s,\sigma_s) = \mathcal N\!\big(M_i(\theta_s),\; \sigma_i^2 + \sigma_s^2\big),

    with **each draw's own** :math:`\sigma_s`, and the posterior-predictive replications draw noise from
    the same inflated variance. Using only the reported errors :math:`\sigma_i` (i.e. dropping
    :math:`\sigma`) makes the density mis-specified: for high-SNR photometry it collapses, so the WAIC/LOO
    effective-parameter counts (``p_waic``/``p_loo``) explode and coverage collapses. Marginalising over
    the full posterior — nuisance scatter included — is the standard posterior-predictive prescription
    (Gelman et al., *Bayesian Data Analysis* 3rd ed., ch. 6–7; Vehtari, Gelman & Gabry 2017).

    ``scatter_param`` selects that column: ``"auto"`` (default) uses the **single posterior column that
    is not a model parameter** (e.g. ``sigma`` for a Villar fit) as :math:`\sigma`; a name forces it; and
    ``None`` disables scatter (reported errors only). Distance-based ABC fits carry no ``sigma`` column,
    so ``"auto"`` correctly falls back to the plain Gaussian for them.

    ``result`` is a :class:`SamplerResult` (uses ``.samples`` / ``.model`` / ``.aic`` / ``.bic``); a bare
    posterior (DataFrame/array) also works but then ``aic`` / ``bic`` are omitted. LOO/WAIC use ``arviz``
    when installed and fall back to a plug-in WAIC otherwise.

    References
    ----------
    * Villar et al. 2017, ApJL 851, L21 — the free extra-scatter term (Eq. 4, MOSFiT form).
    * Hogg, Bovy & Lang 2010, arXiv:1008.4686 — intrinsic scatter ("jitter") added in quadrature.
    * Gelman et al. 2013, *Bayesian Data Analysis* (3rd ed.), ch. 6–7 — posterior predictive checks.
    * Vehtari, Gelman & Gabry 2017, *Stat. Comput.* 27, 1413 — WAIC & PSIS-LOO from the pointwise
      predictive density; Watanabe 2010, *JMLR* 11, 3571 — WAIC.
    """
    from .likelihood import GaussianLikelihoodWithScatter

    samples, names, model_name = _resolve_samples(result, model)
    m = get_model(model if model is not None else model_name)
    if m is None:
        raise ValueError("No model given and the posterior has no .model; pass model=...")
    names = list(m.parameters) if names is None else names

    # Resolve the extra-scatter column. "auto" = the lone posterior column that is not a model
    # parameter (the fitted nuisance scatter sigma); explicit name forces it; None disables.
    extra_cols = [c for c in names if c not in set(m.parameters) and c not in (fixed or {})]
    if scatter_param == "auto":
        sp = extra_cols[0] if len(extra_cols) == 1 else None
    elif scatter_param and scatter_param in names:
        sp = scatter_param
    else:
        sp = None
    sp_idx = names.index(sp) if sp is not None else None

    n_total = samples.shape[0]
    if n_total > n_draws:
        samples = samples[np.random.default_rng(seed).choice(n_total, size=n_draws, replace=False)]
    if sp is not None:
        lik = GaussianLikelihoodWithScatter(lc, space=space, scatter_param=sp)
    else:
        lik = make_likelihood(lc, kind=likelihood, space=space)
    if not hasattr(lik, "log_likelihood_pointwise"):
        raise TypeError(f"{type(lik).__name__} has no log_likelihood_pointwise(); predictive_metrics "
                        "needs a Gaussian (with/without upper limits) likelihood.")
    times, bands = np.asarray(lc.time, float), np.asarray(lc.band).astype(str)
    y = np.asarray(lik.y, float)
    sigma = np.where(np.isfinite(lik.sigma) & (lik.sigma > 0), lik.sigma, 0.0)
    extra = dict(fixed or {})

    preds, lls, sig_tot = [], [], []
    for row in samples:
        s = float(row[sp_idx]) if sp_idx is not None else 0.0
        mf = m.predict({**extra, **{nm: float(v) for nm, v in zip(names, row)}}, times, bands)
        preds.append(np.asarray(lik.model_in_space(mf), float))
        lls.append(np.asarray(lik.log_likelihood_pointwise(mf, sigma_extra=s)
                              if sp_idx is not None else lik.log_likelihood_pointwise(mf), float))
        sig_tot.append(np.sqrt(sigma ** 2 + s ** 2))            # per-draw predictive sd (errors ⊕ σ)
    P = np.vstack(preds)                                        # (draws, data) in comparison space
    ll = np.vstack(lls)
    S = np.vstack(sig_tot)
    good = np.all(np.isfinite(ll), axis=1)
    ll, P_ok, S_ok = ll[good], P[good], S[good]
    n_used = ll.shape[0]

    # RMSE vs the posterior-MEAN prediction (per band + overall) — unaffected by the noise model.
    mean_pred = P_ok.mean(axis=0)
    resid = y - mean_pred

    def _rmse(r):
        r = r[np.isfinite(r)]
        return float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan")

    rmse = {"overall": _rmse(resid), "bands": {str(b): _rmse(resid[bands == b]) for b in np.unique(bands)}}

    # LPD (log pointwise predictive density).
    lppd_i = logsumexp(ll, axis=0) - np.log(n_used)
    lpd = {"total": float(np.sum(lppd_i)), "per_point": float(np.mean(lppd_i))}

    # ELPD (PSIS-LOO) and WAIC via arviz when available; plug-in WAIC otherwise.
    loo_out, waic_out = _loo_waic(ll)

    # Coverage-calibration curve: posterior-predictive intervals with the fit's own noise model
    # (reported errors ⊕ per-draw scatter σ).
    rng = np.random.default_rng(seed + 1)
    y_rep = P_ok + rng.normal(0.0, 1.0, size=P_ok.shape) * S_ok
    cov_overall, cov_bands = [], {str(b): [] for b in np.unique(bands)}
    for q in levels:
        lo, hi = np.percentile(y_rep, [50 * (1 - q), 50 * (1 + q)], axis=0)
        inside = (y >= np.minimum(lo, hi)) & (y <= np.maximum(lo, hi))
        cov_overall.append({"nominal": float(q), "empirical": float(np.mean(inside))})
        for b in np.unique(bands):
            sel = bands == b
            cov_bands[str(b)].append({"nominal": float(q), "empirical": float(np.mean(inside[sel]))})
    coverage = {"levels": [float(q) for q in levels], "overall": cov_overall, "bands": cov_bands}

    out = {"space": lik.space, "unit": "mag" if lik.space == "magnitude" else "Jy",
           "n_draws": int(n_used), "scatter_param": sp, "rmse": rmse, "lpd": lpd,
           "elpd_loo": loo_out, "waic": waic_out, "coverage": coverage}
    for key in ("aic", "bic"):
        v = getattr(result, key, None)
        if v is not None and np.isfinite(v):
            out[key] = float(v)
    return out


def _loo_waic(ll):
    """PSIS-LOO + WAIC from a (draws, data) pointwise log-likelihood matrix.

    WAIC is the standard plug-in estimator (Watanabe 2010 / Gelman et al. 2014). LOO uses ``arviz``'s
    Pareto-smoothed importance sampling (Vehtari et al. 2017) with its ``k`` diagnostic when installed;
    without arviz, ``elpd_loo`` is ``None`` (PSIS smoothing is not reimplemented here).
    """
    n_samp, n_data = ll.shape
    lppd_i = logsumexp(ll, axis=0) - np.log(n_samp)
    p_waic_i = np.var(ll, axis=0, ddof=1)
    elpd_waic_i = lppd_i - p_waic_i
    waic_out = {
        "waic": float(-2.0 * np.sum(elpd_waic_i)), "elpd_waic": float(np.sum(elpd_waic_i)),
        "p_waic": float(np.sum(p_waic_i)),
        "se": float(np.sqrt(n_data * np.var(-2.0 * elpd_waic_i, ddof=1))) if n_data > 1 else float("nan")}
    loo_out = None
    try:
        import arviz as az
        loo = az.loo(az.from_dict(posterior={"_": np.zeros((1, n_samp))},
                                  log_likelihood={"y": ll[None]}), pointwise=True)
        loo_out = {"elpd_loo": float(loo.elpd_loo), "p_loo": float(loo.p_loo), "se": float(loo.se),
                   "looic": float(-2.0 * float(loo.elpd_loo)),
                   "pareto_k_max": float(np.nanmax(np.asarray(loo.pareto_k)))}
    except Exception:
        pass
    return loo_out, waic_out
