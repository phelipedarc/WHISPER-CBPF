"""Inference-validation tools — recovery, posterior-predictive checks, and simulation-based calibration.

These quantify whether a fit **recovered the truth** and whether its **uncertainties are reliable**:

* :func:`recovery_metrics` — per-parameter bias, standardized z-score, and credible-interval coverage
  against a known truth (for synthetic-data recovery tests).
* :func:`posterior_predictive_check` — a posterior-predictive band, the reduced χ² of the best fit, and
  a Bayesian posterior-predictive *p*-value (a well-calibrated fit gives reduced χ² ≈ 1 and *p* ≈ 0.5).
* :func:`sbc_rank` / :func:`sbc_ranks` — Simulation-Based Calibration (Talts et al. 2018; Säilynoja et
  al. 2022): the rank of each true value within its posterior is **uniform** iff the posterior is
  calibrated. A χ²-of-uniformity *p*-value flags over-/under-confidence.
* :func:`check_parity` — whether two models (a JAX port and its redback twin, say) predict the same
  band magnitudes at the same parameters: a per-band |Δmag| table and pass or fail against a stated
  tolerance.

All operate on a :class:`~whisper_cbpf.samplers.base.SamplerResult` (or its ``.samples`` DataFrame) plus
the model + data, so they work identically for every sampler.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2 as _chi2_dist

from .io.photometry import AB_ZEROPOINT_JY
from .models import get_model
from .priors import Prior


def recovery_metrics(result, truth):
    """Per-parameter recovery of a known ``truth`` (dict) from a fit ``result``.

    Returns ``{param: {...}}`` with the posterior ``median``/``mean``/``std``, the 68% (16–84) and 95%
    (2.5–97.5) credible intervals, the ``bias`` (median − true), the standardized **``z_score``**
    (bias / std; ``|z| ≲ 2`` means recovered), and boolean 68%/95% ``within`` coverage. Also a top-level
    ``_summary`` with ``max_abs_z``, ``coverage68``/``coverage95`` (fraction of parameters covered), and
    ``rms_z``.

    Parameters
    ----------
    result : SamplerResult or pandas.DataFrame
        The fit, or its posterior draws.
    truth : dict
        ``{parameter: true value}``; parameters absent from the draws are skipped.

    Returns
    -------
    dict

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t) + np.random.default_rng(1).normal(0, 0.2, 30)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.2))
    >>> res = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0)
    >>> rec = wp.recovery_metrics(res, truth)
    >>> rec["_summary"]["n_params"], rec["_summary"]["coverage95"]
    (3, 1.0)
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
    50/84/97.5 percentiles on ``time_grid``), the **reduced χ²** at the best fit (the posterior median
    for draws without one), and a **Bayesian posterior-predictive p-value** using the χ² discrepancy
    (fraction of replicated datasets with χ² ≥ the observed χ², per draw). Healthy fit ⇒ reduced χ² ≈ 1
    and *p* ≈ 0.5 (near 0/1 flags mis-fit).

    Parameters
    ----------
    result : SamplerResult or pandas.DataFrame
        The fit (its own model and rows are used), or posterior draws with ``model=``.
    lc : LightCurve
        The data; for a fit, the rows it left out as pre-event data are left out here too.
    model : str or Model, optional
        Default: the fit's own model.
    n_draws : int, default 300
    time_grid : array_like, optional
        Epochs of the predictive band; default 200 points across the data.
    seed : int, default 0

    Returns
    -------
    dict
        ``time_grid``, ``lo95``, ``lo68``, ``median``, ``hi68``, ``hi95``, ``reduced_chi2``,
        ``dof``, ``ppc_coverage68``, ``ppc_coverage95``, ``bayesian_p_value``, ``n_draws``.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t) + np.random.default_rng(1).normal(0, 0.2, 30)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.2))
    >>> res = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0)
    >>> ppc = wp.posterior_predictive_check(res, lc, n_draws=100)
    >>> ppc["dof"], bool(0.3 < ppc["reduced_chi2"] < 3.0)
    (27, True)
    """
    if hasattr(result, "fitted_lc"):              # a fit: its own model and its own rows
        from .samplers.base import fitted_model
        model, lc = fitted_model(result, model), result.fitted_lc(lc)
    else:
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

    Calibrated inference ⇒ these ranks are Uniform over ``{0, …, M}`` across many prior realizations.

    Parameters
    ----------
    samples : array_like
        One parameter's posterior draws.
    true_value : float
        The value the data were simulated with.

    Returns
    -------
    int

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> wp.sbc_rank([0.1, 0.5, 0.9], 0.6)
    2
    """
    return int(np.sum(np.asarray(samples, dtype=float) < float(true_value)))


def sbc_ranks(ranks_by_param, *, n_bins=20):
    """Simulation-Based Calibration diagnostics from collected ranks.

    ``ranks_by_param`` maps each parameter to its array of ``L`` ranks (each in ``[0, M]``, from
    :func:`sbc_rank` over ``L`` prior realizations). For each parameter returns the rank **histogram**, a
    **χ²-of-uniformity p-value** (low ⇒ mis-calibrated: a ∪-shape = over-confident/too-narrow, a ∩-shape =
    under-confident/too-wide, a slope = biased), and the sorted **fractional ranks** (rank/M) + empirical
    CDF for an ECDF-difference plot. Also a top-level ``_summary`` with the minimum p-value + verdict.

    Parameters
    ----------
    ranks_by_param : dict
        ``{parameter: array of L ranks}``.
    n_bins : int, default 20
        Histogram bins (at most ``M + 1``).

    Returns
    -------
    dict
        Per parameter the histogram and p-value; ``_summary``: ``min_uniformity_p``,
        ``calibrated`` (p >= 0.05) and ``n_params``.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> ranks = {"amplitude": np.random.default_rng(0).integers(0, 101, 400)}   # uniform ranks
    >>> wp.sbc_ranks(ranks)["_summary"]["calibrated"]
    True
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


# ---------------------------------------------------------------------------------------------------
# Model parity: do two models predict the same magnitudes at the same parameters?
# ---------------------------------------------------------------------------------------------------

#: Epochs where BOTH models are fainter than this (AB mag) are not compared. No survey reaches it, and
#: there a difference between two correct models is a difference between two negligible fluxes. The
#: redback comparison in the test suite uses the same cut. An epoch where one model is brighter is compared,
#: including against zero flux (an infinite difference).
PARITY_FAINT_MAG = 30.0

#: How many of the most discrepant draws a :class:`ParityReport` lists in ``worst``.
PARITY_N_WORST = 5


def _ab_mag(flux):
    """AB magnitude of a flux density in Jy; ``+inf`` where the flux is zero, negative or not finite."""
    f = np.asarray(flux, dtype=float)
    ok = np.isfinite(f) & (f > 0)
    out = np.full(f.shape, np.inf)
    out[ok] = -2.5 * np.log10(f[ok] / AB_ZEROPOINT_JY)
    return out


def _abs_stats(absd):
    """``(median, p99, max)`` of the |Δmag| values ``absd``; NaN when there are none.

    Linear-interpolated percentiles, as the redback comparisons in the test suite report them. With an infinite difference
    among the values, linear interpolation can give ``inf - inf = NaN``, so the percentiles are then
    read as the next data value up (``method="higher"``): still a real difference, never a NaN.
    """
    if absd.size == 0:
        return float("nan"), float("nan"), float("nan")
    method = "higher" if np.isinf(absd).any() else "linear"
    med, p99 = np.percentile(absd, [50.0, 99.0], method=method)
    return float(med), float(p99), float(np.max(absd))


def _posterior_rows(samples, k, rng):
    """Up to ``k`` rows of a samples DataFrame as dicts: all of them, or ``k`` drawn without replacement."""
    if k <= 0 or len(samples) == 0:
        return []
    if len(samples) <= k:
        idx = np.arange(len(samples))
    else:
        idx = np.sort(rng.choice(len(samples), size=k, replace=False))
    return samples.iloc[idx].to_dict("records")


def _parity_draws(source, n, seed):
    """The parameter dicts :func:`check_parity` compares at, and a phrase naming where they came from."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    if isinstance(source, Prior):
        # One prior.sample(rng) per draw, in sequence: the same draws as the redback comparisons in the test suite.
        return [source.sample(rng) for _ in range(n)], f"{n} prior draws (seed {seed})"
    if isinstance(source, Mapping):
        return [dict(source)], "1 parameter set"
    if isinstance(source, (list, tuple)) and source and all(isinstance(d, Mapping) for d in source):
        return [dict(d) for d in source], f"{len(source)} parameter sets"
    samples = getattr(source, "samples", None)
    if isinstance(samples, pd.DataFrame):                       # a SamplerResult: best fit first
        best = dict(getattr(source, "best_params", None) or {})
        rows = _posterior_rows(samples, n - 1 if best else n, rng)
        draws = ([best] if best else []) + rows
        if not draws:
            raise ValueError(
                "the fit has no posterior draws and no best fit (for ABC: no accepted draw), so there "
                "are no parameters to compare at. Refit with a larger budget, or pass a parameter dict.")
        return draws, (f"the best fit and {len(rows)} posterior draws" if best
                       else f"{len(rows)} posterior draws")
    if isinstance(source, pd.DataFrame):
        if len(source) == 0:
            raise ValueError("the posterior DataFrame has no rows, so there are no parameters to "
                             "compare at. Pass a non-empty posterior, a parameter dict or a Prior.")
        rows = _posterior_rows(source, n, rng)
        return rows, f"{len(rows)} posterior draws"
    raise TypeError(
        f"params_or_posterior must be a parameter dict, a list of dicts, a posterior (a DataFrame of "
        f"samples or a SamplerResult) or a Prior; got {type(source).__name__}.")


def _parity_params(model, other, draw):
    """``model``'s parameters out of ``draw``, pairing names through either model's ``param_aliases``."""
    out, missing = {}, []
    for p in model.parameters:
        keys = [p, model.param_aliases.get(p)] + [q for q, twin in other.param_aliases.items() if twin == p]
        key = next((k for k in keys if k is not None and k in draw), None)
        if key is None:
            missing.append(p)
        else:
            out[p] = float(draw[key])
    if missing:
        raise ValueError(
            f"model {model.name!r} needs {missing}, which the parameters to compare at do not name (they "
            f"name {sorted(draw)}). Give a value for every free parameter of both models; a parameter "
            f"named differently in the two models is paired through Model.param_aliases.")
    return out


@dataclass(frozen=True, eq=False)
class ParityReport:
    """What :func:`check_parity` measured: |Δmag| between two models, per band, and the verdict.

    Attributes
    ----------
    model_a, model_b : str
        The two models' names. Every difference is ``mag_a - mag_b``.
    passed : bool
        ``True`` when every compared point agrees within ``tolerance`` and neither model returned a
        non-finite flux. ``False`` also when nothing could be compared (see ``reason``).
    reason : str
        The verdict in one sentence: the largest difference against the tolerance, where it is, or why
        nothing was compared.
    tolerance : float
        The bar on max |Δmag|, in mag.
    n_draws, n_points : int
        Parameter sets evaluated, and (draw, epoch) pairs compared (one model brighter than
        :data:`PARITY_FAINT_MAG` there).
    median_abs, p99_abs, max_abs : float
        Median, 99th percentile and maximum of |Δmag| over the compared points (``inf`` where one
        model predicts zero flux and the other does not).
    per_band : dict
        ``{band: {"n", "median_abs", "p99_abs", "max_abs", "median_signed", "passed"}}``. Empty for a
        model called without bands.
    worst : list of dict
        Up to :data:`PARITY_N_WORST` draws, largest max |Δmag| first: ``draw`` (index into the draws),
        ``max_abs_dmag``, the ``time`` and ``band`` where it happens, ``mag_a`` and ``mag_b`` there, and
        ``params`` (``model_a``'s names).
    nonfinite : dict
        ``{"a": k, "b": j}``: points where a model returned NaN or infinite flux.
    draws : str
        Where the parameter sets came from.
    times, bands : numpy.ndarray
        The epochs compared, sorted by time (``bands`` is ``None`` for a band-free model).
    dmag : numpy.ndarray
        Every signed difference, shape ``(n_draws, n_epochs)``; NaN where not compared.
    """

    model_a: str
    model_b: str
    passed: bool
    reason: str
    tolerance: float
    n_draws: int
    n_points: int
    median_abs: float
    p99_abs: float
    max_abs: float
    per_band: dict
    worst: list
    nonfinite: dict
    draws: str
    times: np.ndarray
    bands: np.ndarray
    dmag: np.ndarray

    def table(self):
        """The per-band statistics, plus an ``all`` row, as a pandas DataFrame indexed by band."""
        import pandas as pd

        rows = {b: dict(s) for b, s in self.per_band.items()}
        signed = self.dmag[np.isfinite(self.dmag)]
        rows["all"] = dict(n=self.n_points, median_abs=self.median_abs, p99_abs=self.p99_abs,
                           max_abs=self.max_abs,
                           median_signed=float(np.median(signed)) if signed.size else float("nan"),
                           passed=self.passed)
        return pd.DataFrame.from_dict(rows, orient="index")[
            ["n", "median_abs", "p99_abs", "max_abs", "median_signed", "passed"]]

    def __repr__(self):
        verdict = "PASSED" if self.passed else "FAILED"
        head = (f"ParityReport: {self.model_a} vs {self.model_b} -- {verdict} "
                f"(tolerance {self.tolerance:g} mag)\n"
                f"{self.draws} x {self.times.size} epochs; {self.n_points} points compared "
                f"(|dmag| = |mag_a - mag_b| in AB mag)\n")
        body = self.table().to_string(float_format=lambda v: f"{v:.3g}")
        return (f"{head}{body}\n{self.reason}\n"
                f"Next: .worst lists the draws that disagree most; .dmag holds every difference.")


def check_parity(model_a, model_b, params_or_posterior, times, bands, *, tolerance=0.02, n=200, seed=0):
    """Check that two models predict the same band magnitudes at the same parameters.

    Evaluates both models at every parameter set and epoch, converts each flux to an AB magnitude and
    reports |Δmag| = |mag_a - mag_b| per band: median, 99th percentile, maximum, and pass or fail
    against ``tolerance``. Use it to check a JAX port against its redback twin (the model a GPU fit
    samples against the one its forecasts use), a grid or photometry setting against another, or any
    two whisper models.

    Parameters
    ----------
    model_a, model_b : Model or str
        The two models (a registered name or a :class:`~whisper_cbpf.models.Model`). Both are called
        through ``predict(params, times, bands)`` and must return flux density in Jy on the same time
        axis: build a JAX factory model and a redback-adapter model with the same redshift and time
        origin.
    params_or_posterior : dict, list of dict, Prior, pandas.DataFrame or SamplerResult
        Where to compare. A dict is one parameter set and a list is used as given. A
        :class:`~whisper_cbpf.priors.Prior` gives ``n`` draws (``prior.sample`` in sequence from
        ``numpy.random.default_rng(seed)``, as the redback comparisons in the test suite draw). A DataFrame of
        posterior samples gives ``n`` rows drawn without replacement (all rows if fewer); a
        :class:`~whisper_cbpf.samplers.base.SamplerResult` gives its best fit plus ``n - 1`` posterior
        rows. Each parameter set must name every free parameter of both models; names that differ
        between the two are paired through ``Model.param_aliases``.
    times : array_like
        Observer-frame epochs, on the models' own clock. Sorted before use, so the order of the input
        does not matter (some redback grids are sized by the last epoch).
    bands : array_like, str or None
        One band label per epoch, a single label for all epochs, or ``None`` for a model that takes no
        bands.
    tolerance : float, optional
        Largest |Δmag| that still passes, in mag. Default 0.02, the tolerance the supernova and
        TDE models meet against redback.
    n : int, optional
        Number of parameter sets taken from a Prior, DataFrame or SamplerResult. Default 200.
    seed : int, optional
        Seed for those draws. Default 0.

    Returns
    -------
    ParityReport
        ``passed``, ``reason``, the overall and per-band statistics (``table()``), the ``worst``
        draws, and every difference in ``dmag``.

    Raises
    ------
    ValueError
        Empty or mismatched ``times`` and ``bands``, a non-positive ``tolerance``, a parameter set that
        lacks a parameter of either model, or a posterior with no draws.
    TypeError
        ``params_or_posterior`` of an unsupported type.

    Notes
    -----
    A point is compared when at least one model is brighter than :data:`PARITY_FAINT_MAG` (30 mag)
    there. Zero or negative flux is an infinitely faint magnitude, so a model that goes dark where the
    other does not (an out-of-domain epoch, a constraint wall armed in one model only) fails with
    ``max_abs = inf``. A NaN or infinite flux fails the check wherever it occurs. When no point is
    brighter than 30 mag in either model, ``passed`` is ``False`` and ``reason`` says nothing was
    compared, rather than reporting a number.

    Examples
    --------
    Two copies of the same model agree exactly; a 2% flux error in one band is caught:

    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> base = wp.get_model("bazin")
    >>> def two_percent_bright_in_r(params, times, bands):
    ...     boost = np.where(np.asarray(bands) == "r", 1.02, 1.0)
    ...     return boost * base.predict(params, times, bands)
    >>> other = wp.Model("bazin_off", two_percent_bright_in_r, base.parameters)
    >>> t = np.linspace(-10.0, 60.0, 30)
    >>> b = np.where(np.arange(30) % 2 == 0, "g", "r")
    >>> wp.check_parity(base, base, base.default_prior, t, b, n=20).passed
    True
    >>> report = wp.check_parity(base, other, base.default_prior, t, b, n=20)
    >>> report.passed, round(report.per_band["r"]["max_abs"], 4), report.per_band["g"]["max_abs"]
    (False, 0.0215, 0.0)

    A JAX supernova port against redback's own model at 200 prior draws (needs the ``[gpu]`` and
    ``[models]`` extras and float64):

    >>> from whisper_cbpf.models import redback_adapter as ra          # doctest: +SKIP
    >>> z, bands = 0.051, ["ztfg", "ztfr"]                              # doctest: +SKIP
    >>> cpu = ra.redback_model("arnett", bands, redshift=z, constraint=None)  # doctest: +SKIP
    >>> gpu = wp.supernova_model("arnett", bands, z, ra.redback_luminosity_distance_cm(z),
    ...                          constraint=None)                       # doctest: +SKIP
    >>> t = np.linspace(3.0, 32.0, 28)                                  # doctest: +SKIP
    >>> wp.check_parity(gpu, cpu, cpu.default_prior, t, np.resize(bands, 28)).max_abs < 1e-11
    ... # doctest: +SKIP
    True
    """
    ma, mb = get_model(model_a), get_model(model_b)
    tol = float(tolerance)
    if not tol > 0:
        raise ValueError(f"tolerance={tolerance!r} must be a positive number of magnitudes (default 0.02).")
    if int(n) != n or n < 1:
        raise ValueError(f"n={n!r} must be a positive integer: the number of parameter sets to draw.")
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0:
        raise ValueError("times is empty: pass the epochs to compare the two models at.")
    if not np.all(np.isfinite(t)):
        raise ValueError("times contains NaN or infinite values; pass finite epochs.")
    if bands is None:
        b = None
    else:
        b = np.asarray(bands)
        if b.ndim == 0:
            b = np.full(t.shape, b.item())
        b = b.ravel()
        if b.size != t.size:
            raise ValueError(f"times has {t.size} epochs and bands has {b.size} labels: pass one band "
                             f"per epoch, or a single band name for all of them.")
    order = np.argsort(t, kind="stable")
    t = t[order]
    b = None if b is None else b[order]

    draws, source = _parity_draws(params_or_posterior, int(n), seed)
    flux = {"a": [], "b": []}
    for d in draws:
        for key, model, other in (("a", ma, mb), ("b", mb, ma)):
            f = np.asarray(model.predict(_parity_params(model, other, d), t, b), dtype=float)
            if f.shape != t.shape:
                raise ValueError(f"model {model.name!r} returned {f.shape} fluxes for {t.size} epochs; "
                                 f"predict must return one flux per epoch.")
            flux[key].append(f)
    fa, fb = np.array(flux["a"]), np.array(flux["b"])
    nonfinite = {"a": int((~np.isfinite(fa)).sum()), "b": int((~np.isfinite(fb)).sum())}
    mag_a, mag_b = _ab_mag(fa), _ab_mag(fb)
    compared = np.minimum(mag_a, mag_b) < PARITY_FAINT_MAG
    with np.errstate(invalid="ignore"):
        dmag = np.where(compared, mag_a - mag_b, np.nan)
    absd = np.abs(dmag[compared])
    median_abs, p99_abs, max_abs = _abs_stats(absd)

    per_band = {}
    if b is not None:
        band_grid = np.broadcast_to(b, dmag.shape)
        for name in sorted({str(x) for x in b.tolist()}):
            sel = compared & (band_grid.astype(str) == name)
            med, p99, mx = _abs_stats(np.abs(dmag[sel]))
            per_band[name] = dict(n=int(sel.sum()), median_abs=med, p99_abs=p99, max_abs=mx,
                                  median_signed=float(np.median(dmag[sel])) if sel.any() else float("nan"),
                                  passed=bool(sel.any() and mx <= tol))

    worst = []
    draw_max = np.where(compared, np.abs(np.nan_to_num(dmag, nan=0.0)), -1.0).max(axis=1)
    for i in np.argsort(-draw_max, kind="stable")[:PARITY_N_WORST]:
        if draw_max[i] < 0:
            break
        j = int(np.argmax(np.where(compared[i], np.abs(np.nan_to_num(dmag[i], nan=0.0)), -1.0)))
        worst.append(dict(draw=int(i), max_abs_dmag=float(draw_max[i]), time=float(t[j]),
                          band=None if b is None else str(b[j]), mag_a=float(mag_a[i, j]),
                          mag_b=float(mag_b[i, j]), params=_parity_params(ma, mb, draws[i])))

    n_points = int(compared.sum())
    if n_points == 0:
        passed = False
        reason = (f"Not enough data: no epoch is brighter than {PARITY_FAINT_MAG:g} mag in either model, "
                  f"so nothing was compared. Check that both models return flux density in Jy, and "
                  f"compare at epochs where the transient is bright.")
    else:
        w = worst[0]
        where = f"draw {w['draw']}, t = {w['time']:.4g}" + (f", band {w['band']}" if w["band"] else "")
        if nonfinite["a"] or nonfinite["b"]:
            passed = False
            bad = [f"{name} at {nonfinite[k]} points" for k, name in (("a", ma.name), ("b", mb.name))
                   if nonfinite[k]]
            reason = f"Non-finite flux (NaN or inf) from {' and '.join(bad)}: a model defect, whatever |dmag| is."
        elif np.isinf(max_abs):
            passed = False
            dark = mb.name if np.isinf(w["mag_b"]) else ma.name
            lit, mag = ((ma.name, w["mag_a"]) if dark == mb.name else (mb.name, w["mag_b"]))
            reason = (f"{dark} predicts no flux where {lit} is at {mag:.2f} mag ({where}): one model is "
                      f"dark where the other is not (an epoch outside one model's domain, or a "
                      f"constraint wall armed in one model only).")
        else:
            passed = bool(max_abs <= tol)
            reason = (f"max |dmag| {max_abs:.3g} mag {'<=' if passed else '>'} tolerance {tol:g} mag"
                      + ("" if passed else f"; largest at {where} ({w['mag_a']:.3f} vs {w['mag_b']:.3f} mag)")
                      + ".")
    return ParityReport(model_a=ma.name, model_b=mb.name, passed=passed, reason=reason, tolerance=tol,
                        n_draws=len(draws), n_points=n_points, median_abs=median_abs, p99_abs=p99_abs,
                        max_abs=max_abs, per_band=per_band, worst=worst, nonfinite=nonfinite, draws=source,
                        times=t, bands=b, dmag=dmag)
