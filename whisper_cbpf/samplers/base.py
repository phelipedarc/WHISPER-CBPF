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


#: ``type(likelihood).__name__`` (what samplers record in ``info["likelihood"]``) -> registry key.
#: Anything unrecognised falls back to ``"auto"``, which is the pre-existing behaviour.
_LIKELIHOOD_KINDS = {
    "GaussianLikelihood": "gaussian",
    "GaussianLikelihoodWithUpperLimits": "gaussian_upper_limits",
    "GaussianLikelihoodWithScatter": "gaussian_scatter",
    "MixtureGaussianLikelihood": "mixture",
}


def _warn_without_raising(message):
    """Emit a UserWarning that cannot abort the caller, even under warnings-as-errors.

    Both `attach_*` helpers report their failure by warning from inside an `except` block. Under
    `-W error` / `simplefilter("error")` / a pytest `filterwarnings = error` that warning is itself
    an exception, so it escapes the handler and kills a fit that had ALREADY SUCCEEDED -- measured:
    on HEAD `simplefilter("error")` + `wp.fit_ABC(...)` raised out of `attach_predictive_metrics`
    where the pre-audit tree returned a result. A diagnostic that destroys the thing it is
    diagnosing is worse than the silence it replaced, so the filter is forced locally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, stacklevel=3)


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
        _warn_without_raising(f"per-band metrics could not be computed and are absent from this "
                              f"result ({type(exc).__name__}: {exc}). The fit itself is unaffected.")


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
    # Score under the density the fit was ACTUALLY run with. `predictive_metrics` defaults to
    # kind="auto", i.e. a plain Gaussian, so a `likelihood="mixture"` fit was being rescored under
    # a different density -- which then made the new p_waic flag blame the model. Measured on 40
    # points with one 30-sigma outlier: converged=True and medians within 2% of truth, yet
    # p_waic = 324.11 and "unreliable"; rescored under the mixture it was fitted with,
    # p_waic = 2.35 and reliable. `_LIKELIHOOD_KINDS` maps the class name samplers record in
    # info["likelihood"] back to a registry key.
    kind = _LIKELIHOOD_KINDS.get(str(result.info.get("likelihood", "")), "auto")
    try:
        from ..metrics import predictive_metrics
        result.info["predictive_metrics"] = predictive_metrics(
            result, lc, space=space, likelihood=kind, n_draws=n_draws)
    except Exception as exc:                                  # noqa: BLE001 - deliberate, see above
        result.info["predictive_metrics_error"] = f"{type(exc).__name__}: {exc}"
        _warn_without_raising(f"predictive metrics (WAIC / LPD / PSIS-LOO / coverage) could not be "
                              f"computed and are absent from this result ({type(exc).__name__}: "
                              f"{exc}). AIC and BIC are unaffected.")


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
