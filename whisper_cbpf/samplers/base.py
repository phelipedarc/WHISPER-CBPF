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


def aic_bic(max_log_likelihood, n_params, n_data):
    """``(AIC, BIC)`` from the maximum log-likelihood, both on the deviance scale."""
    k, n = int(n_params), int(n_data)
    return (float(-2.0 * max_log_likelihood + 2 * k),
            float(-2.0 * max_log_likelihood + k * np.log(n)))


def check_burnin(burnin, nsteps):
    """Refuse a burn-in that would discard the whole chain.

    ``get_chain(discard=burnin)`` returns zero rows when ``burnin >= nsteps``, and the AIC/BIC
    step that follows it calls ``np.nanargmax`` on that empty array. Raising here names the two
    numbers, and does it before the sampling rather than after.
    """
    if int(burnin) >= int(nsteps):
        raise ValueError(
            f"burnin={int(burnin)} discards the whole chain of nsteps={int(nsteps)}. "
            f"The burn-in must be shorter than the chain; raise nsteps or lower burnin.")


#: ``type(likelihood).__name__`` (what samplers record in ``info["likelihood"]``) -> registry key.
#: Anything unrecognised falls back to ``"auto"``.
_LIKELIHOOD_KINDS = {
    "GaussianLikelihood": "gaussian",
    "GaussianLikelihoodWithUpperLimits": "gaussian_upper_limits",
    "GaussianLikelihoodWithScatter": "gaussian_scatter",
    "MixtureGaussianLikelihood": "mixture",
}


def _warn_without_raising(message):
    """Emit a UserWarning that cannot abort the caller, even under warnings-as-errors.

    Both ``attach_*`` helpers warn from inside an ``except`` block. Under ``-W error`` the warning
    is itself an exception, so it would escape that handler and kill a fit that had already
    succeeded. A diagnostic must not destroy the thing it is diagnosing, so the filter is forced
    locally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, stacklevel=3)


def attach_band_metrics(info, lc, model, best_params, space):
    """Populate ``info['band_metrics']`` with per-band MSE/MAE at the best fit (best-effort).

    Shared by every sampler, so per-band goodness-of-fit lands in ``SamplerResult.to_json``. The
    broad guard is deliberate: a metric failure must not break an otherwise-successful fit. The
    failure is reported rather than swallowed, in ``info['band_metrics_error']`` and in a warning.
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

    Best-effort: a failure (slow or failing model, no likelihood) leaves the fit untouched but not
    the user uninformed. WAIC is one of the three numbers this package exists to put side by side,
    so "absent" and "absent because your bands did not resolve" have to be distinguishable: the
    reason lands in ``info['predictive_metrics_error']`` and in a warning.
    """
    # Score under the density the fit was run with. `predictive_metrics` defaults to kind="auto",
    # a plain Gaussian, which would rescore a `likelihood="mixture"` fit under a different density
    # and then blame the model for the resulting p_waic: on 40 points with one 30-sigma outlier
    # that reads as p_waic = 324.11 and "unreliable" against 2.35 and reliable under the mixture.
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

    ``aic`` and ``bic`` are plain fields; the posterior-predictive numbers live in the
    ``info['predictive_metrics']`` block and are surfaced beside them as the read-only properties
    :attr:`waic`, :attr:`waic_reliable`, :attr:`elpd_loo` and :attr:`rmse`, each ``None`` rather
    than raising when that block is missing or failed.
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

    def _metric(self, *path):
        """Read one number out of ``info['predictive_metrics']``; ``None`` if it is not there.

        The block is best-effort (see :func:`attach_predictive_metrics`): when it fails the key is
        absent and the reason is parked in ``info['predictive_metrics_error']``. Walking the path
        returns ``None`` at every level rather than raising ``KeyError`` on exactly the fits where
        something has already gone wrong -- including ``elpd_loo is None`` when arviz is absent.
        """
        node = self.info.get("predictive_metrics") if isinstance(self.info, dict) else None
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    @property
    def waic(self):
        """WAIC on the deviance scale (lower is better), or ``None`` if the metric block is absent.

        Equal by construction to ``whisper_cbpf.waic(result, lc)["waic"]`` -- the manual call adopts
        the likelihood, space, scatter column and draw count recorded here. Check :attr:`waic_reliable`
        before comparing it against another model's.
        """
        return self._metric("waic", "waic")

    @property
    def waic_reliable(self):
        """``p_waic <= n_data/2`` (Gelman/Hwang/Vehtari 2014 §4), or ``None`` if WAIC is absent.

        ``False`` means the posterior's pointwise log-likelihood varies by orders of magnitude --
        usually a stranded chain -- and this WAIC must not be compared or read beside AIC/BIC.
        """
        return self._metric("waic", "p_waic_reliable")

    @property
    def elpd_loo(self):
        """PSIS-LOO expected log predictive density (higher is better).

        ``None`` when the metric block is absent **or** ``arviz`` is not installed; in the latter
        case ``info['predictive_metrics']['elpd_loo']`` is ``None`` and, if arviz is installed but
        errored, ``info['predictive_metrics']['elpd_loo_error']`` says why.
        """
        return self._metric("elpd_loo", "elpd_loo")

    @property
    def rmse(self):
        """Overall RMSE of ``observed - posterior-mean prediction``, in the fit's comparison space.

        Jy for a flux fit, mag for a magnitude fit (``info['predictive_metrics']['unit']``); per-band
        values are under ``['rmse']['bands']``. ``None`` if the metric block is absent.
        """
        return self._metric("rmse", "overall")

    def to_dict(self):
        # waic/elpd_loo/rmse are views on info["predictive_metrics"], which this dict already
        # carries in full. Promoting them would duplicate the same numbers under new top-level
        # keys and change the JSON shape that existing readers parse.
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
