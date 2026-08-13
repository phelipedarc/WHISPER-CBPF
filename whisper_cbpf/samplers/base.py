"""Sampler base class and the unified result container."""
from __future__ import annotations

import json
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


def attach_band_metrics(info, lc, model, best_params, space):
    """Populate ``info['band_metrics']`` with per-band MSE/MAE at the best fit (best-effort).

    Shared by every sampler so the per-band goodness-of-fit lands in ``SamplerResult.to_json``.
    Wrapped in a broad guard — a metric failure (e.g. a slow/failing forward model) must never
    break an otherwise-successful fit — but the failure is **reported**, not swallowed. See
    :func:`attach_predictive_metrics` for why.
    """
    try:
        from ..metrics import per_band_metrics
        info["band_metrics"] = per_band_metrics(lc, model, best_params, space=space)
    except Exception as exc:                                  # noqa: BLE001 - deliberate, see above
        info["band_metrics_error"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"per-band metrics could not be computed and are absent from this result "
                      f"({type(exc).__name__}: {exc}). The fit itself is unaffected.",
                      stacklevel=2)


def attach_predictive_metrics(result, lc, space, n_draws=200):
    """Populate ``result.info['predictive_metrics']`` with the posterior predictive metric block.

    RMSE (posterior-mean), LPD, ELPD (PSIS-LOO), WAIC, AIC/BIC and the coverage-calibration curve —
    see :func:`whisper_cbpf.metrics.predictive_metrics`. Called after the result is built (it reuses
    the fit's ``aic`` / ``bic``); ``n_draws`` bounds the extra forward-model evaluations.

    Best-effort: a failure (slow/failing model, no likelihood) leaves the fit untouched. It does
    **not** leave the user uninformed, which it used to — this was a bare ``except Exception:
    pass``, so a model whose ``predict`` raised produced a fit with no WAIC, no message, and no way
    to tell that from a model for which WAIC is simply undefined. Since WAIC is one of the three
    numbers this package exists to put side by side, "absent" and "absent because your bands did not
    resolve" have to be distinguishable. The reason now lands in
    ``info['predictive_metrics_error']`` and in a warning.
    """
    try:
        from ..metrics import predictive_metrics
        result.info["predictive_metrics"] = predictive_metrics(result, lc, space=space, n_draws=n_draws)
    except Exception as exc:                                  # noqa: BLE001 - deliberate, see above
        result.info["predictive_metrics_error"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"predictive metrics (WAIC / LPD / PSIS-LOO / coverage) could not be computed "
                      f"and are absent from this result ({type(exc).__name__}: {exc}). AIC and BIC "
                      f"are unaffected.", stacklevel=2)


@dataclass
class SamplerResult:
    """Unified result for every Whisper sampler.

    ``samples`` holds the accepted/posterior draws. Model-selection metrics (``aic``, ``bic``,
    ``max_log_likelihood``) come from the best fit. With a chi-square distance these use
    ``chi2 = -2 ln L`` **up to an additive constant** (the Gaussian normalization is dropped, so
    absolute values are offset, but model comparison on the same data is unaffected); a proper
    ``whisper_cbpf.likelihood`` gives exact values. ``info`` carries sampler-specific diagnostics.
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

    @property
    def n_samples(self):
        return int(len(self.samples))

    def to_dict(self):
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

    def __repr__(self):
        return (f"SamplerResult(sampler={self.sampler!r}, model={self.model!r}, "
                f"n_samples={self.n_samples}, AIC={self.aic:.1f}, runtime={self.runtime_s:.2f}s)")


class BaseSampler:
    """Contract for samplers: implement ``fit(lc, model, prior=None, **kwargs) -> SamplerResult``."""

    name = "base"

    def fit(self, lc, model, prior=None, **kwargs):
        raise NotImplementedError
