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

from ..likelihood import make_likelihood
from ..models import get_model


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


def _model_and_rows(posterior, model, model_name, lc):
    """The model to score with and the rows to score: for a fit, its own Model object (a factory
    model is not registered by name) and its own rows (the pre-event rows it left out removed)."""
    if hasattr(posterior, "fitted_lc"):                       # a SamplerResult
        from ..samplers.base import fitted_model
        return fitted_model(posterior, model), posterior.fitted_lc(lc)
    if model is None and model_name is None:
        raise ValueError("No model given and the posterior has no .model; pass model=...")
    return get_model(model if model is not None else model_name), lc


#: Posterior columns a sampler writes as per-draw diagnostics, never as parameters, so ``"auto"``
#: must not take one for a scatter term. ABC's ``distance`` was taken: read as a sigma it gave a
#: predictive sd of ~1e4 mag per point and coverage 1.00 at every level on every ABC fit.
_DIAGNOSTIC_COLUMNS = ("distance",)


def _resolve_scatter(names, model, fixed, scatter_param):
    """Resolve the free extra-scatter column -> ``(name, column index)``, either ``None`` if absent.

    ``"auto"`` takes the **single** posterior column that is neither a model parameter nor a
    diagnostic (``_DIAGNOSTIC_COLUMNS``) -- the fitted nuisance sigma; an explicit name forces it;
    ``None`` disables scatter. Shared by :func:`waic` and :func:`predictive_metrics` so both score
    under the same density — see :func:`waic`.
    """
    if scatter_param == "auto":
        extra = [c for c in names if c not in set(model.parameters) and c not in (fixed or {})
                 and c not in _DIAGNOSTIC_COLUMNS]
        sp = extra[0] if len(extra) == 1 else None
    elif scatter_param and scatter_param in names:
        sp = scatter_param
    else:
        sp = None
    return sp, (names.index(sp) if sp is not None else None)


def _scoring_likelihood(lc, space, likelihood, sp, who):
    """Build the likelihood the draws are scored under (scatter-augmented when ``sp`` is set)."""
    from ..likelihood import GaussianLikelihoodWithScatter

    lik = (GaussianLikelihoodWithScatter(lc, space=space, scatter_param=sp) if sp is not None
           else make_likelihood(lc, kind=likelihood, space=space))
    if not hasattr(lik, "log_likelihood_pointwise"):
        raise TypeError(f"{type(lik).__name__} has no log_likelihood_pointwise(); {who} needs the "
                        "pointwise log-likelihood. Use a Gaussian (with/without upper limits) likelihood.")
    return lik


def _score_draws(model, lik, names, samples, sp_idx, fixed, times, bands):
    """Evaluate every draw once -> ``(predictions in comparison space, pointwise log-lik, sigma_s)``.

    One loop, one set of numbers: :func:`waic` and :func:`predictive_metrics` both go through here,
    so a WAIC computed either way is the same WAIC by construction rather than by coincidence.
    """
    extra = dict(fixed or {})
    preds, lls, sig = [], [], []
    for row in samples:
        s = float(row[sp_idx]) if sp_idx is not None else 0.0
        mf = model.predict({**extra, **{nm: float(v) for nm, v in zip(names, row)}}, times, bands)
        preds.append(np.asarray(lik.model_in_space(mf), float))
        lls.append(np.asarray(lik.log_likelihood_pointwise(mf, sigma_extra=s)
                              if sp_idx is not None else lik.log_likelihood_pointwise(mf), float))
        sig.append(s)
    return np.vstack(preds), np.vstack(lls), np.asarray(sig, float)


def _waic_from_ll(ll):
    """WAIC from a ``(draws, data)`` pointwise log-likelihood matrix — the one implementation.

    ``lppd_i = log mean_s p(y_i|θ_s)``, ``p_waic_i = Var_s log p(y_i|θ_s)``,
    ``elpd_waic = Σ_i (lppd_i - p_waic_i)`` and ``waic = -2 elpd_waic`` (Watanabe 2010; Vehtari,
    Gelman & Gabry 2017, Eq. 11). The variance is the **sample** variance, ``1/(S-1)``, as both
    papers define it — note ``arviz.waic`` uses the population variance ``1/S`` instead, so its
    ``p_waic`` is this one times ``(S-1)/S``.

    ``p_waic`` is returned with a reliability flag; see :func:`_loo_waic` for what it catches.
    Its warning points at the user's own line (a fit, or a call of :func:`waic`).
    """
    from ..samplers.base import _user_stacklevel

    n_samp, n_data = ll.shape
    lppd_i = logsumexp(ll, axis=0) - np.log(n_samp)            # log mean-likelihood per point
    p_waic_i = np.var(ll, axis=0, ddof=1)                      # posterior var of log-lik per point
    elpd_i = lppd_i - p_waic_i
    p_waic = float(np.sum(p_waic_i))
    reliable = bool(np.isfinite(p_waic) and p_waic <= 0.5 * n_data)
    if not reliable:
        warnings.warn(
            f"WAIC is unreliable here: p_waic = {p_waic:.4g} against n_data = {n_data} "
            f"(Gelman/Hwang/Vehtari's guidance is p_waic <= n_data/2). The posterior contains draws "
            f"whose pointwise log-likelihood varies by orders of magnitude. The usual cause is ONE "
            f"OR MORE STRANDED CHAINS -- a single walker parked in a bad region contributes a huge "
            f"per-point variance -- so check the chains before blaming the model. Do NOT compare "
            f"this WAIC against another model's, and do not read it on the same scale as AIC/BIC.",
            stacklevel=_user_stacklevel())
    return {"waic": float(-2.0 * np.sum(elpd_i)), "elpd_waic": float(np.sum(elpd_i)),
            "lppd": float(np.sum(lppd_i)), "p_waic": p_waic, "p_waic_reliable": reliable,
            "n_data": int(n_data),
            "se": float(np.sqrt(n_data * np.var(-2.0 * elpd_i, ddof=1))) if n_data > 1 else float("nan")}


def _fit_scoring_config(posterior):
    """How this fit's own metric block scored itself, so :func:`waic` can reproduce it exactly.

    ``{}`` for anything that is not a :class:`SamplerResult` carrying a successful
    ``info['predictive_metrics']`` block.
    """
    info = getattr(posterior, "info", None)
    block = info.get("predictive_metrics") if isinstance(info, dict) else None
    if not isinstance(block, dict):
        return {}
    cfg = {k: block[k] for k in ("space", "likelihood", "scatter_param") if k in block}
    if "n_draws_requested" in block:
        cfg["max_samples"] = int(block["n_draws_requested"])
    return cfg


def waic(posterior, lc, model=None, *, space="auto", likelihood="auto", scatter_param="auto",
         fixed=None, max_samples="auto", seed=0):
    """Widely Applicable Information Criterion (WAIC) from a posterior sample.

    **Handed a fitted** :class:`SamplerResult`, this reproduces the fit's own auto-attached number:
    ``wp.waic(result, lc)["waic"] == result.waic``. The defaults ``space``/``likelihood``/
    ``scatter_param``/``max_samples`` are all ``"auto"``, and ``"auto"`` means *whatever the fit
    recorded* — the space it was fitted in, the likelihood kind it was fitted under, the extra-scatter
    column it fitted, and the same draw count (and seed) the attached block scored. Before this, the
    manual call always used a plain Gaussian on 2000 draws while the attached block used the fit's own
    density on 200, so a scatter fit produced two WAICs that disagreed by a factor of 9182 (310496.50
    against 33.82 on an 80-point ``gaussian_rise`` fit) with nothing to say which was meant. Pass any
    argument explicitly to override.

    For a bare ``DataFrame``/array there is no fit to follow, so ``"auto"`` keeps the old meanings:
    the data-appropriate space and likelihood, and 2000 draws.

    Parameters
    ----------
    posterior : SamplerResult | pandas.DataFrame | numpy.ndarray
        Posterior draws. A :class:`SamplerResult` uses its ``.samples`` (and ``.model`` when ``model``
        is omitted); a ``DataFrame`` uses its columns as parameter names; a bare array takes names from
        the model's ``parameters`` (so the column order must match).
    lc : LightCurve
        Data the model is scored against. For a fit, the rows it left out as pre-event data
        (:meth:`SamplerResult.fitted_lc`) are left out here too, so the fit's light curve can be
        passed as it was.
    model : str | Model, optional
        Model name/object; defaults to the fit's own model (``posterior.model``).
    space, likelihood : str
        Forwarded to :func:`~whisper_cbpf.likelihood.make_likelihood` (data space + likelihood kind).
    scatter_param : str | None
        Posterior column holding a free extra-scatter sigma, added in quadrature to the reported
        errors. ``"auto"`` = the fit's own column (or, without a fit, the lone non-model column);
        ``None`` disables it and scores against the reported errors alone.
    fixed : dict, optional
        Parameter values to merge into every draw — for parameters held fixed during the fit and so not
        present in the posterior columns (e.g. a pinned ``redshift``).
    max_samples : int | "auto"
        Cap on posterior draws evaluated (one model call per draw — matters for slow simulators).
    seed : int
        Seed for subsampling when the posterior has more than ``max_samples`` draws.

    Returns
    -------
    dict
        ``waic`` (lower is better), ``elpd_waic``, ``lppd``, ``p_waic`` (effective #parameters),
        ``p_waic_reliable``, ``se`` (standard error of WAIC), ``n_samples``, ``n_data``.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t) + np.random.default_rng(1).normal(0, 0.2, 30)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.2))
    >>> res = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0)
    >>> w = wp.waic(res, lc)
    >>> w["n_data"], w["p_waic_reliable"], w["waic"] == res.waic   # the fit's own number
    (30, True, True)
    """
    samples, names, model_name = _resolve_samples(posterior, model)
    model, lc = _model_and_rows(posterior, model, model_name, lc)
    names = list(model.parameters) if names is None else names

    cfg = _fit_scoring_config(posterior)
    space = cfg.get("space", "auto") if space == "auto" else space
    likelihood = cfg.get("likelihood", "auto") if likelihood == "auto" else likelihood
    if scatter_param == "auto" and "scatter_param" in cfg:
        scatter_param = cfg["scatter_param"]
    from_fit = max_samples == "auto" and "max_samples" in cfg
    max_samples = cfg.get("max_samples", 2000) if max_samples == "auto" else int(max_samples)

    n_total = samples.shape[0]
    subsampled = n_total > max_samples
    if subsampled:
        # Same rng call as `predictive_metrics`, so the same seed picks the same draws.
        sel = np.random.default_rng(seed).choice(n_total, size=max_samples, replace=False)
        samples = samples[sel]
        if not from_fit:      # following the fit's own draw count is the documented default, not news
            warnings.warn(f"WAIC: posterior has {n_total} draws; evaluated a random {max_samples}-draw "
                          "subsample (raise max_samples for the full set).", stacklevel=2)

    sp, sp_idx = _resolve_scatter(names, model, fixed, scatter_param)
    lik = _scoring_likelihood(lc, space, likelihood, sp, "WAIC")
    times, bands = np.asarray(lc.time, float), np.asarray(lc.band)
    _, ll, _ = _score_draws(model, lik, names, samples, sp_idx, fixed, times, bands)

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

    out = _waic_from_ll(ll)
    out.update({"n_samples": int(ll.shape[0]), "scatter_param": sp,
                "n_draws_dropped": n_dropped, "subsampled": bool(subsampled)})
    return out


def per_band_metrics(lc, model, params, *, space="auto", fixed=None):
    """Per-band goodness-of-fit residual metrics at a single parameter set (e.g. the best fit).

    Evaluates ``model`` at ``params`` on the observed ``(time, band)`` grid and, **per band**, reports
    the mean-squared error (MSE), root-mean-squared error (RMSE) and mean-absolute error (MAE) of the
    residuals ``observed - model``, computed in the fit's **comparison space** — flux density [Jy] for
    a flux fit, apparent magnitude [mag] for a magnitude fit — so the metric matches the space the fit
    actually optimised. This is the deterministic point-estimate complement to the distributional
    :func:`waic` / posterior-predictive checks.

    **Detections only.** Rows flagged ``upper_limit=True`` are excluded, and ``n_upper_limits_excluded``
    reports how many. A non-detection's ``y`` is a *limit*, not a measurement, so ``observed - model``
    on that row is not a residual: a model sitting correctly *below* an upper limit still contributes
    ``(limit - model)**2`` to the MSE, and the deeper (more constraining) the limit, the worse it makes
    a correct fit look. Every band the light curve observes keeps an entry, so a band that is entirely
    non-detections reports ``n = 0`` rather than vanishing from the report.

    Parameters
    ----------
    lc : LightCurve
        The observed data. Censored rows are dropped from the metrics (see above); no error bars are
        read, so rows whose ``flux_err`` is NaN — which is how whisper's loader marks a non-detection —
        are fine.
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
        ``{"space", "unit", "n_upper_limits_excluded", "bands": {band: {"mse","rmse","mae","n"}},
        "overall": {...}}``. ``unit`` is ``"Jy"`` (flux) or ``"mag"`` (magnitude), and every ``n``
        counts detections only.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 20)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> lc = wp.LightCurve(time=t, band=["r", "g"] * 10,
    ...                    flux=wp.get_model("flare").predict(truth, t) + 0.1,
    ...                    flux_err=np.full(20, 0.2))
    >>> m = wp.per_band_metrics(lc, "flare", truth)
    >>> m["unit"], sorted(m["bands"]), round(m["overall"]["rmse"], 3)
    ('Jy', ['g', 'r'], 0.1)
    """
    from ..likelihood import flux_to_space, resolve_space

    m = get_model(model)
    if m is None:
        raise ValueError(f"Unknown model {model!r}.")
    # Resolve the space and convert directly instead of building a likelihood as a space converter.
    # A likelihood validates the data it is about to DIVIDE by -- GaussianLikelihood refuses a
    # censored row outright, and any NaN error bar with it -- and none of that applies to a
    # residual, which needs no sigma at all. Building one here made per-band metrics fail on
    # exactly the light curves that most need them: measured on a 24-point flux curve with 8
    # non-detections fitted through pymc_jax_gpu_vectorized with likelihood='upper_limits', the
    # ValueError left every result carrying info['band_metrics_error'] and no band_metrics.
    sp = resolve_space(lc, space)
    if sp == "magnitude":
        y = lc.magnitude if lc.magnitude is not None else lc.add_mag().magnitude
    else:
        y = lc.flux if lc.flux is not None else lc.add_flux().flux
    y = np.asarray(y, dtype=float)

    times = np.asarray(lc.time, dtype=float)
    bands = np.asarray(lc.band).astype(str)
    p = dict(fixed or {})
    p.update({k: float(params[k]) for k in m.parameters if k in params})     # model params only
    model_flux = np.asarray(m.predict(p, times, bands), dtype=float)
    resid = y - np.asarray(flux_to_space(model_flux, sp), dtype=float)

    ul = lc.upper_limit
    det = np.ones(resid.size, dtype=bool) if ul is None else ~np.asarray(ul, dtype=bool)

    def _stats(r):
        r = r[np.isfinite(r)]
        if r.size == 0:
            return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n": 0}
        mse = float(np.mean(r ** 2))
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(r))), "n": int(r.size)}

    per_band = {str(b): _stats(resid[det & (bands == b)]) for b in np.unique(bands)}
    return {"space": sp, "unit": "mag" if sp == "magnitude" else "Jy",
            "n_upper_limits_excluded": int(np.sum(~det)),
            "bands": per_band, "overall": _stats(resid[det])}


#: default nominal levels for the coverage-calibration curve.
DEFAULT_COVERAGE_LEVELS = (0.5, 0.68, 0.8, 0.9, 0.95, 0.99)


def predictive_metrics(result, lc, model=None, *, space="auto", likelihood="auto",
                       scatter_param="auto", fixed=None, levels=DEFAULT_COVERAGE_LEVELS,
                       n_draws=400, seed=0):
    r"""Posterior predictive / model-comparison metrics from a fit, for the JSON report.

    Evaluates the model across ``n_draws`` posterior samples once and returns, in one dict:

    * **rmse** — root-mean-squared error of ``observed - posterior_mean_prediction`` (per band + overall),
      in the fit's comparison space (Jy for flux, mag for magnitude). **Detections only** — see below.
    * **lpd** — log predictive density: ``sum_i log mean_s p(y_i | θ_s)`` (the ``lppd``; higher is
      better), with the per-point mean. Over **every** row, non-detections included.
    * **elpd_loo** — expected log predictive density by **PSIS-LOO** cross-validation (Vehtari et al.
      2017; higher is better), with ``p_loo``, standard error ``se``, ``looic = -2·elpd_loo`` and the
      max Pareto-``k`` diagnostic (``k > 0.7`` ⇒ LOO estimate unreliable).
    * **waic** — WAIC on the deviance scale (``-2·elpd_waic``; lower is better), with ``elpd_waic``,
      ``p_waic`` and ``se``.
    * **aic** / **bic** — from the fit's best-fit likelihood (lower is better), carried through from
      ``result`` when available.
    * **coverage** — the calibration curve: for each nominal ``level`` the *empirical* fraction of
      observations inside the central posterior-predictive interval, overall and per band. A calibrated
      fit has empirical ≈ nominal at every level. **Detections only** — see below.

    **Non-detections are in the density, not in the point comparisons.** ``rmse`` and ``coverage``
    score the **detections** alone, and ``n_upper_limits_excluded`` reports how many rows that dropped;
    ``lpd``, ``waic`` and ``elpd_loo`` score **every** row. The asymmetry is the point. A censored row's
    ``y`` is a *limit*: its principled contribution to a predictive *density* is the survival term
    ``log P(flux < limit)``, which ``log_likelihood_pointwise`` supplies and which is real information
    about the fit — but ``observed - model`` on that row is not a residual (a model correctly *below* an
    upper limit still contributes ``(limit - model)**2``, the more so the deeper the limit), and asking
    whether a bound falls inside a central predictive interval is not a calibration question. Scoring
    them in ``rmse``/``coverage`` reported inflated error and collapsed calibration for any censored fit.
    Bands with no detection are kept in both blocks with ``NaN`` rather than dropped.

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

    Parameters
    ----------
    result : SamplerResult or pandas.DataFrame or numpy.ndarray
        The fit (its rows and its own model are used), or bare posterior draws.
    lc : LightCurve
        The data; for a fit, the rows it left out as pre-event data are left out here too.
    model : str or Model, optional
        Default: the fit's own model.
    space, likelihood : str
        The comparison space and likelihood kind (``"auto"``: the data's).
    scatter_param : str or None, default "auto"
        The free-scatter column (see above).
    fixed : dict, optional
        Values of parameters held fixed and absent from the draws.
    levels : sequence of float
        Nominal levels of the coverage curve.
    n_draws : int, default 400
        Posterior draws evaluated.
    seed : int, default 0

    Returns
    -------
    dict
        ``rmse``, ``lpd``, ``elpd_loo``, ``waic``, ``aic``, ``bic``, ``coverage``, and how they were
        scored (``space``, ``likelihood``, ``scatter_param``, ``n_draws``, ...).

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t) + np.random.default_rng(1).normal(0, 0.2, 30)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.2))
    >>> res = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0)
    >>> m = wp.predictive_metrics(res, lc, n_draws=200)
    >>> m["unit"], [c["nominal"] for c in m["coverage"]["overall"]][:3]
    ('Jy', [0.5, 0.68, 0.8])

    References
    ----------
    * Villar et al. 2017, ApJL 851, L21 — the free extra-scatter term (Eq. 4, MOSFiT form).
    * Hogg, Bovy & Lang 2010, arXiv:1008.4686 — intrinsic scatter ("jitter") added in quadrature.
    * Gelman et al. 2013, *Bayesian Data Analysis* (3rd ed.), ch. 6–7 — posterior predictive checks.
    * Vehtari, Gelman & Gabry 2017, *Stat. Comput.* 27, 1413 — WAIC & PSIS-LOO from the pointwise
      predictive density; Watanabe 2010, *JMLR* 11, 3571 — WAIC.
    """
    samples, names, model_name = _resolve_samples(result, model)
    m, lc = _model_and_rows(result, model, model_name, lc)
    names = list(m.parameters) if names is None else names

    sp, sp_idx = _resolve_scatter(names, m, fixed, scatter_param)

    n_total = samples.shape[0]
    if n_total > n_draws:
        samples = samples[np.random.default_rng(seed).choice(n_total, size=n_draws, replace=False)]
    lik = _scoring_likelihood(lc, space, likelihood, sp, "predictive_metrics")
    times, bands = np.asarray(lc.time, float), np.asarray(lc.band).astype(str)
    y = np.asarray(lik.y, float)
    sigma = np.where(np.isfinite(lik.sigma) & (lik.sigma > 0), lik.sigma, 0.0)

    P, ll, sig_s = _score_draws(m, lik, names, samples, sp_idx, fixed, times, bands)
    S = np.sqrt(sigma ** 2 + sig_s[:, None] ** 2)               # per-draw predictive sd (errors ⊕ σ)
    good = np.all(np.isfinite(ll), axis=1)
    ll, P_ok, S_ok = ll[good], P[good], S[good]
    n_used = ll.shape[0]

    # WHICH ROWS EACH BLOCK SCORES. A non-detection's `y` is a LIMIT, not a measurement, so it
    # belongs in the DENSITY blocks (lpd, waic/loo) and NOT in the POINT-COMPARISON ones
    # (rmse, coverage). The asymmetry is deliberate, and it is the reverse of the intuition that
    # "a row with no measurement should be dropped everywhere":
    #
    #   * `lpd` / `waic` / `elpd_loo` keep EVERY row. They go through `log_likelihood_pointwise`,
    #     which gives a censored row its survival term log P(flux < limit) -- the correct, and
    #     genuinely informative, contribution of a non-detection to a predictive density. Dropping
    #     those rows would throw information away AND make the totals incomparable with a fit whose
    #     likelihood counted them.
    #   * `rmse` / `coverage` keep DETECTIONS only. `observed - model` on a limit is not a residual:
    #     a model correctly sitting BELOW an upper limit still contributes (limit - model)^2, and
    #     the deeper (more constraining) the limit the worse it makes a correct fit look. Coverage
    #     asks whether the observation falls inside the central predictive interval, which is not a
    #     calibration question when the "observation" is a bound rather than a draw from that
    #     interval. Running both over every row silently reports inflated error and
    #     collapsed calibration on any censored fit.
    #
    # `n_upper_limits_excluded` (top level of the returned dict) reports the rows this dropped.
    ul = lc.upper_limit
    det = np.ones(y.size, dtype=bool) if ul is None else ~np.asarray(ul, dtype=bool)

    # RMSE vs the posterior-MEAN prediction (per band + overall) — unaffected by the noise model.
    mean_pred = P_ok.mean(axis=0)
    resid = y - mean_pred

    def _rmse(r):
        r = r[np.isfinite(r)]
        return float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan")

    rmse = {"overall": _rmse(resid[det]),
            "bands": {str(b): _rmse(resid[det & (bands == b)]) for b in np.unique(bands)}}

    # LPD (log pointwise predictive density).
    lppd_i = logsumexp(ll, axis=0) - np.log(n_used)
    lpd = {"total": float(np.sum(lppd_i)), "per_point": float(np.mean(lppd_i))}

    # ELPD (PSIS-LOO) and WAIC via arviz when available; plug-in WAIC otherwise.
    loo_out, waic_out, loo_error = _loo_waic(ll)

    # Coverage-calibration curve: posterior-predictive intervals with the fit's own noise model
    # (reported errors ⊕ per-draw scatter σ), over the detections. Band keys stay the FULL observed
    # set, so a band with no detection reports NaN rather than vanishing from the curve.
    rng = np.random.default_rng(seed + 1)
    y_det, bands_det = y[det], bands[det]
    y_rep = P_ok[:, det] + rng.normal(0.0, 1.0, size=(n_used, int(np.sum(det)))) * S_ok[:, det]
    cov_overall, cov_bands = [], {str(b): [] for b in np.unique(bands)}
    for q in levels:
        lo, hi = np.percentile(y_rep, [50 * (1 - q), 50 * (1 + q)], axis=0)
        inside = (y_det >= np.minimum(lo, hi)) & (y_det <= np.maximum(lo, hi))
        cov_overall.append({"nominal": float(q),
                            "empirical": float(np.mean(inside)) if inside.size else float("nan")})
        for b in np.unique(bands):
            sel = bands_det == b
            cov_bands[str(b)].append({"nominal": float(q),
                                      "empirical": float(np.mean(inside[sel])) if sel.any()
                                      else float("nan")})
    coverage = {"levels": [float(q) for q in levels], "overall": cov_overall, "bands": cov_bands}

    # `space`/`likelihood`/`scatter_param`/`n_draws_requested` record HOW this block was scored, so
    # `waic(result, lc)` can reproduce exactly this number instead of a differently-specified one.
    out = {"space": lik.space, "unit": "mag" if lik.space == "magnitude" else "Jy",
           "likelihood": str(likelihood), "n_draws": int(n_used),
           "n_draws_requested": int(n_draws), "seed": int(seed),
           "n_upper_limits_excluded": int(np.sum(~det)),
           "scatter_param": sp, "rmse": rmse, "lpd": lpd,
           "elpd_loo": loo_out, "waic": waic_out, "coverage": coverage}
    if loo_error is not None:
        out["elpd_loo_error"] = loo_error
    for key in ("aic", "bic"):
        v = getattr(result, key, None)
        if v is not None and np.isfinite(v):
            out[key] = float(v)
    return out


def _loo_waic(ll):
    """PSIS-LOO + WAIC from a (draws, data) pointwise log-likelihood matrix.

    Returns ``(loo_out, waic_out, loo_error)``. WAIC is :func:`_waic_from_ll` — the same code
    :func:`waic` runs. LOO uses ``arviz``'s Pareto-smoothed importance sampling (Vehtari et al. 2017)
    with its ``k`` diagnostic when installed; without arviz, ``elpd_loo`` is ``None`` and
    ``loo_error`` is ``None`` too (PSIS smoothing is not reimplemented here).

    **arviz ABSENT and arviz FAILING are different things.** Collapsing them into one bare
    ``except Exception: pass``, so a genuine arviz error — a version whose ``loo`` signature moved, a
    matrix it rejects — was indistinguishable from the documented no-arviz degradation: both gave a
    silent ``elpd_loo = None``. The import is now caught on its own, and a failure of ``az.loo``
    itself is recorded in ``loo_error`` (surfaced by :func:`predictive_metrics` as
    ``elpd_loo_error``) so the caller can tell "you did not install arviz" from "arviz broke".

    **``p_waic`` is returned with a reliability flag.** ``p_waic`` estimates the effective number of
    parameters, so on a well-behaved fit it lands at or below the model's parameter count. When the
    posterior contains draws whose pointwise log-likelihood differs by orders of magnitude — an
    under-converged chain, or a badly misspecified model whose draws range over tens of thousands of
    sigma — the per-point variance explodes and WAIC stops being on a comparable scale with AIC/BIC
    at all. Measured on a 9-parameter redback kilonova against 88 AT2017GFO points:
    ``p_waic = 939505`` beside an AIC of 415. Gelman, Hwang & Vehtari (2014) §4 give ``p_waic >
    n_data / 2`` as the point where the estimator should not be trusted; that is what
    ``p_waic_reliable`` reports, and the caller is warned rather than handed a number that looks
    like the others.

    **It flags stranded chains, not misspecification.** Adversarially re-measured: on a 4-parameter
    fit to data generated FROM the fitted model, ``p_waic`` blew up on 5 of 12 seeds (max 2.79e8)
    with ``converged=True`` -- traced to 3.12% of draws (one emcee walker x 300 thinned steps) at
    logL -15950.8 against a posterior median of +112.6. Conversely the badly misspecified redback
    kilonova gives ``p_waic = 11.73``, flagged reliable, under ABC on the same 88 points.

    **Both arviz APIs.** arviz 1.0 moved ``from_dict`` to one dict of groups and renamed the LOO
    result's ``elpd_loo`` / ``p_loo`` to ``elpd`` / ``p``. Written for 0.x only, every fit on arviz
    1.3.0 carried ``elpd_loo_error = "TypeError: from_dict() got an unexpected keyword argument
    'posterior'"`` and no ELPD; :func:`_idata_from_ll` and the attribute fallbacks cover both.
    """
    waic_out = _waic_from_ll(ll)
    loo_out, loo_error = None, None
    try:
        import arviz as az
    except ImportError:
        return loo_out, waic_out, loo_error        # documented degraded mode, not a failure
    try:
        loo = az.loo(_idata_from_ll(az, ll), pointwise=True)
        elpd = float(loo.elpd if hasattr(loo, "elpd") else loo.elpd_loo)       # 1.x : 0.x
        p = float(loo.p if hasattr(loo, "p") else loo.p_loo)
        loo_out = {"elpd_loo": elpd, "p_loo": p, "se": float(loo.se),
                   "looic": float(-2.0 * elpd),
                   "pareto_k_max": float(np.nanmax(np.asarray(loo.pareto_k)))}
    except Exception as exc:                       # noqa: BLE001 - arviz is installed and broke
        loo_error = f"arviz {getattr(az, '__version__', '?')}: {type(exc).__name__}: {exc}"
    return loo_out, waic_out, loo_error


def _idata_from_ll(az, ll):
    """A (draws, data) log-likelihood matrix as the object ``az.loo`` takes, on arviz 0.x or >= 1.0.

    One chain; the posterior group holds a dummy variable because arviz needs one to exist.
    """
    groups = {"posterior": {"_": np.zeros((1, ll.shape[0]))}, "log_likelihood": {"y": ll[None]}}
    if int(str(az.__version__).split(".")[0]) >= 1:
        return az.from_dict(groups)                # 1.x: one dict of groups
    return az.from_dict(**groups)                  # 0.x: one keyword per group
