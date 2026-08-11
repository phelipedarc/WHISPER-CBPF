"""Whisper (``whisper_labia``): easy Bayesian model comparison of transient light curves.

**Four pluggable axes**, each a small name registry with matching ``register_*`` / ``list_*`` helpers:
**models** (``register_model`` / ``list_models``), **samplers** (``register_sampler`` / ``list_samplers``
— ABC, ABC-SMC, MCMC and SNPE today; Dynesty planned), **likelihoods** (``register_likelihood`` /
``list_likelihoods``) and **distances** (``register_distance`` / ``list_distances``). The data ingestion,
samplers, likelihoods, plots and outputs are Whisper's own and run standalone; physical models + priors
can optionally be supplied by the external redback package (the ``[models]`` extra), used only as a
source of models and priors.
"""

__version__ = "0.0.1.dev0"

from .io import (
    FILTER_LOOKUP,
    LSST_BAND_INFO,
    LightCurve,
    SvoUnavailable,
    clear_manual_bands,
    group_bands,
    load_lightcurve,
    register_manual_band,
    resolve_band,
    resolve_bands,
    unregister_manual_band,
)
from .plotting import CORNER_PALETTE, plot_calibration, plot_corner, plot_light_curve, plot_ppc
from .metrics import per_band_metrics, predictive_metrics, waic
from .validation import (
    posterior_predictive_check,
    recovery_metrics,
    sbc_rank,
    sbc_ranks,
)
from .priors import LogUniform, Prior, Uniform
from .distance import chi2_distance, get_distance, list_distances, register_distance
from .likelihood import (
    GaussianLikelihood,
    GaussianLikelihoodWithScatter,
    GaussianLikelihoodWithUpperLimits,
    MixtureGaussianLikelihood,
    list_likelihoods,
    make_likelihood,
    register_likelihood,
)
from .models import Model, get_model, list_models, register_model
from .samplers import (
    SamplerResult,
    fit,
    fit_ABC,
    fit_ABC_SMC,
    fit_MCMC,
    fit_SNPE,
    list_samplers,
    register_sampler,
)

__all__ = [
    "__version__",
    # data + plotting
    "LightCurve", "load_lightcurve", "plot_light_curve", "plot_ppc", "plot_calibration",
    "plot_corner", "CORNER_PALETTE",
    "group_bands", "FILTER_LOOKUP",
    "resolve_band", "resolve_bands", "LSST_BAND_INFO", "SvoUnavailable",
    "register_manual_band", "unregister_manual_band", "clear_manual_bands",
    # priors / models
    "Prior", "Uniform", "LogUniform",
    "Model", "register_model", "get_model", "list_models",
    # distances (registry)
    "chi2_distance", "register_distance", "get_distance", "list_distances",
    # likelihoods (registry)
    "GaussianLikelihood", "GaussianLikelihoodWithScatter", "GaussianLikelihoodWithUpperLimits",
    "MixtureGaussianLikelihood", "make_likelihood", "register_likelihood", "list_likelihoods",
    # samplers (registry)
    "fit_ABC", "fit_ABC_SMC", "fit_MCMC", "fit_SNPE", "fit", "SamplerResult", "register_sampler",
    "list_samplers",
    # metrics + validation
    "waic", "per_band_metrics", "predictive_metrics", "recovery_metrics", "posterior_predictive_check",
    "sbc_rank", "sbc_ranks",
]
