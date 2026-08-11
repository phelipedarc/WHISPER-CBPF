"""Throughput metrics for the benchmark: effective samples per second.

Raw draws/second is a misleading way to compare samplers. emcee draws are heavily
autocorrelated (many draws carry the information of one independent sample); NUTS draws are
near-independent; SNPE draws are exactly independent (iid from a trained density estimator).
Comparing wall-clock per raw draw therefore flatters whichever sampler produces the most
correlated output. Effective Sample Size normalizes that away, and the WORST parameter's ESS
is the honest summary — a posterior is only as well-characterized as its least-constrained
direction.

ArviZ is imported lazily and failures degrade to NaN with a warning, matching the
optional-arviz pattern in ``whisper_cbpf.metrics._loo_waic``.
"""
from __future__ import annotations

import warnings

import numpy as np


def ess_by_parameter(samples_by_chain, param_names, method="bulk"):
    """Per-parameter ESS from draws shaped ``(n_chains, n_draws, n_params)``.

    Samplers without a natural chain axis (SNPE's iid draws) should pass ``n_chains=1``.
    Returns ``{param: ess}``; all-NaN if arviz is unavailable.
    """
    samples_by_chain = np.asarray(samples_by_chain)
    if samples_by_chain.ndim != 3:
        raise ValueError(f"expected (n_chains, n_draws, n_params), got {samples_by_chain.shape}")
    if samples_by_chain.shape[2] != len(param_names):
        raise ValueError(f"{samples_by_chain.shape[2]} parameter columns but "
                         f"{len(param_names)} names given")
    try:
        import arviz as az
    except ImportError:
        warnings.warn("arviz not installed; ESS unavailable (pip install arviz).", stacklevel=2)
        return {nm: float("nan") for nm in param_names}

    try:
        # One arviz variable per parameter: az.ess then returns a scalar per name, avoiding any
        # ambiguity about how a multi-dimensional variable's ESS would be reduced.
        idata = az.convert_to_inference_data(
            {nm: samples_by_chain[:, :, i] for i, nm in enumerate(param_names)})
        ess = az.ess(idata, method=method)
        return {nm: float(np.asarray(ess[nm].values)) for nm in param_names}
    except Exception as exc:                                   # pragma: no cover
        warnings.warn(f"ESS computation failed: {exc!r}", stacklevel=2)
        return {nm: float("nan") for nm in param_names}


def ess_summary(samples_by_chain, param_names, wall_clock_s, method="bulk"):
    """ESS per parameter plus the worst-parameter ESS and ESS/sec.

    ``wall_clock_s`` should be the cost a user actually pays to obtain these draws. For SNPE
    that means simulation + training + sampling, since the network is not reused across curves.
    """
    by_param = ess_by_parameter(samples_by_chain, param_names, method=method)
    finite = [v for v in by_param.values() if np.isfinite(v)]
    worst = float(min(finite)) if finite else float("nan")
    worst_param = (min(((k, v) for k, v in by_param.items() if np.isfinite(v)),
                       key=lambda kv: kv[1])[0] if finite else None)
    wall = float(wall_clock_s)
    return {
        "ess_by_param": by_param,
        "ess_per_sec_by_param": {k: (v / wall if np.isfinite(v) and wall > 0 else float("nan"))
                                  for k, v in by_param.items()},
        "ess_worst": worst,
        "ess_worst_param": worst_param,
        "ess_worst_per_sec": (worst / wall if np.isfinite(worst) and wall > 0 else float("nan")),
        "wall_clock_s": wall,
        "n_draws": int(np.asarray(samples_by_chain).shape[0]
                       * np.asarray(samples_by_chain).shape[1]),
        "ess_method": method,
    }
