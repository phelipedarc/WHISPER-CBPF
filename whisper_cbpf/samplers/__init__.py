"""Pluggable samplers: a small registry + a generic ``fit`` dispatcher.

Add a new sampler by subclassing :class:`BaseSampler` and calling :func:`register_sampler`.
"""
from __future__ import annotations

from .abc import ABCSampler, fit_ABC
from .abc_smc import ABCSMCSampler, fit_ABC_SMC
from .base import (
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)
from .mcmc import MCMCSampler, fit_MCMC
from .snpe import SNPESampler, fit_SNPE

# "snpe" and "npe" both map to the same sampler (num_rounds=1 is amortized NPE, >1 is sequential).
_SAMPLERS = {
    "abc": ABCSampler, "abc_smc": ABCSMCSampler, "mcmc": MCMCSampler,
    "snpe": SNPESampler, "npe": SNPESampler,
}


def register_sampler(name, sampler_cls, *, overwrite=False):
    if name in _SAMPLERS and not overwrite:
        raise ValueError(f"Sampler {name!r} already registered (pass overwrite=True).")
    _SAMPLERS[name] = sampler_cls


def get_sampler(name):
    if name not in _SAMPLERS:
        raise KeyError(f"Unknown sampler {name!r}. Available: {list_samplers()}")
    return _SAMPLERS[name]()


def list_samplers():
    return sorted(_SAMPLERS)


def fit(lc, model, sampler="abc", **kwargs) -> SamplerResult:
    """Generic dispatch: ``fit(lc, model, sampler='abc'|'abc_smc'|'snpe', ...)``."""
    return get_sampler(sampler).fit(lc, model, **kwargs)


__all__ = [
    "BaseSampler", "SamplerResult", "summarize_posterior",
    "ABCSampler", "fit_ABC", "ABCSMCSampler", "fit_ABC_SMC", "MCMCSampler", "fit_MCMC",
    "SNPESampler", "fit_SNPE", "fit",
    "register_sampler", "get_sampler", "list_samplers",
]
