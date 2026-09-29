"""Whisper (``whisper_cbpf``): easy Bayesian model comparison of transient light curves.

**Four pluggable axes**, each a small name registry with matching ``register_*`` / ``list_*`` helpers:
**models** (``register_model`` / ``list_models``), **samplers** (``register_sampler`` / ``list_samplers``
— ABC, ABC-SMC, MCMC, nested sampling (dynesty) and SNPE), **likelihoods** (``register_likelihood`` /
``list_likelihoods``) and **distances** (``register_distance`` / ``list_distances``). The data ingestion,
samplers, likelihoods, plots and outputs are Whisper's own and run standalone; physical models + priors
can optionally be supplied by the external redback package (the ``[models]`` extra), used only as a
source of models and priors.
"""

__version__ = "0.2.0"

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
    check_parity,
    posterior_predictive_check,
    recovery_metrics,
    sbc_rank,
    sbc_ranks,
)
from .priors import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform
from .distance import (
    chi2_distance,
    get_distance,
    list_distances,
    max_abs_z_distance,
    register_distance,
)
from .models.cosmology import luminosity_distance_cm
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
# How a model SED becomes a band magnitude, the same on CPU and GPU (docs/PHOTOMETRY.md); bare
# u g r i z y are read as LSST unless the session default is changed here.
from .synphot import default_band_system, resolve_filter, set_default_band_system

# Binding factories for redback's own models, on the CPU. Like the JAX factories below these stay
# factories: the parameters and the prior are read out of redback at build time (never transcribed),
# which needs redback importable -- so naming a model here is a runtime call, not an import-time one,
# and `import whisper_cbpf` keeps working with the [models] extra absent.
from .models.redback_adapter import redback_model, register_redback  # noqa: E402
from .samplers import (
    SamplerResult,
    fit,
    fit_ABC,
    fit_ABC_SMC,
    fit_MCMC,
    fit_nested,
    fit_SNPE,
    list_samplers,
    register_sampler,
)
from .likelihood_max_opt import LikelihoodMaxOptResult, likelihood_max_opt  # noqa: E402
from .results import DiagnosticsReport, fit_cached, load_result  # noqa: E402
from .parallel import Job, run_jobs  # noqa: E402


def _unavailable(name, where, exc):
    """A stand-in for ``name`` when importing it from ``where`` failed (a module or an optional
    dependency this installation lacks): calling it raises an ImportError that says why."""
    reason = f"{type(exc).__name__}: {exc}"

    def unavailable(*args, **kwargs):
        raise ImportError(
            f"whisper_cbpf.{name} is not available in this installation: importing it from "
            f"{where} failed ({reason}). Install the extra that error names, or a whisper_cbpf "
            f"release that provides {name}.") from exc

    unavailable.__name__ = unavailable.__qualname__ = name
    unavailable.__doc__ = f"Not available: importing {name} from {where} failed ({reason})."
    unavailable.whisper_unavailable = reason
    return unavailable


# From an alert to a ranked, saved, explained answer (release 0.2.0). Imported here, not lazily:
# `compare`, `forecast`, `report` and `profile` are also the names of their modules, and a module
# first imported after its function is bound here would replace the function on the package. None
# of these modules imports jax, torch or redback at import time. A module missing from this
# installation leaves a stand-in that raises an ImportError saying so when called.
try:
    from .compare import Comparison, compare  # noqa: E402
except ImportError as _exc:
    Comparison, compare = (_unavailable(n, "whisper_cbpf.compare", _exc)
                           for n in ("Comparison", "compare"))
try:
    from .forecast import discriminate, forecast  # noqa: E402
except ImportError as _exc:
    discriminate, forecast = (_unavailable(n, "whisper_cbpf.forecast", _exc)
                              for n in ("discriminate", "forecast"))
try:
    from .plotting import plot_forecast, plot_model_comparison, plot_models, plot_widths  # noqa: E402
except ImportError as _exc:
    plot_forecast, plot_model_comparison, plot_models, plot_widths = (
        _unavailable(n, "whisper_cbpf.plotting", _exc)
        for n in ("plot_forecast", "plot_model_comparison", "plot_models", "plot_widths"))
try:
    from .facts import comparison_facts, result_facts, write_facts  # noqa: E402
except ImportError as _exc:
    comparison_facts, result_facts, write_facts = (
        _unavailable(n, "whisper_cbpf.facts", _exc)
        for n in ("comparison_facts", "result_facts", "write_facts"))
try:
    from .report import report  # noqa: E402
except ImportError as _exc:
    report = _unavailable("report", "whisper_cbpf.report", _exc)
# Speed for an LSST night: one compile for many alerts, and what a night holds.
try:
    from .samplers.jax._adapters import log_density  # noqa: E402  (numpy-only at import time)
except ImportError as _exc:
    log_density = _unavailable("log_density", "whisper_cbpf.samplers.jax._adapters", _exc)
try:
    from .samplers.jax.batch import fit_batch  # noqa: E402  (numpy-only at import time)
except ImportError as _exc:
    fit_batch = _unavailable("fit_batch", "whisper_cbpf.samplers.jax.batch", _exc)
try:
    from .profile import capacity, profile  # noqa: E402
except ImportError as _exc:
    capacity, profile = (_unavailable(n, "whisper_cbpf.profile", _exc)
                         for n in ("capacity", "profile"))

__all__ = [
    "__version__",
    # data + plotting
    "LightCurve", "load_lightcurve", "plot_light_curve", "plot_ppc", "plot_calibration",
    "plot_corner", "CORNER_PALETTE",
    "group_bands", "FILTER_LOOKUP",
    "resolve_band", "resolve_bands", "LSST_BAND_INFO", "SvoUnavailable",
    "register_manual_band", "unregister_manual_band", "clear_manual_bands",
    # photometry: label -> filter, and the bare-letter system
    "resolve_filter", "set_default_band_system", "default_band_system",
    # priors / models
    "Prior", "Uniform", "LogUniform", "Normal", "TruncatedNormal", "Fixed",
    "Model", "register_model", "get_model", "list_models",
    "redback_model", "register_redback", "luminosity_distance_cm",
    # distances (registry)
    "chi2_distance", "max_abs_z_distance", "register_distance", "get_distance", "list_distances",
    # likelihoods (registry)
    "GaussianLikelihood", "GaussianLikelihoodWithScatter", "GaussianLikelihoodWithUpperLimits",
    "MixtureGaussianLikelihood", "make_likelihood", "register_likelihood", "list_likelihoods",
    # samplers (registry)
    "fit_ABC", "fit_ABC_SMC", "fit_MCMC", "fit_nested", "fit_SNPE", "fit", "SamplerResult",
    "register_sampler", "list_samplers",
    # the likelihood peak behind AIC / BIC
    "likelihood_max_opt", "LikelihoodMaxOptResult",
    # saving, resuming and checking a fit
    "load_result", "fit_cached", "DiagnosticsReport",
    # metrics + validation
    "waic", "per_band_metrics", "predictive_metrics", "recovery_metrics", "posterior_predictive_check",
    "sbc_rank", "sbc_ranks", "check_parity",
    # many fits at once, over the GPUs and cores allowed
    "Job", "run_jobs",
    # from an alert to a ranked, saved, explained answer
    "compare", "Comparison", "forecast", "discriminate",
    "plot_forecast", "plot_models", "plot_model_comparison", "plot_widths",
    "result_facts", "comparison_facts", "write_facts", "report",
    # speed for an LSST night
    "log_density", "fit_batch", "profile", "capacity",
]

# --- JAX/GPU half ------------------------------------------------------------------------------
# Imported last, once the CPU registries above are populated. This registers the six GPU samplers
# and `flare_jax` WITHOUT importing jax: every entry is a lazy factory, so `import whisper_cbpf`
# works on a machine with no jax, no CUDA, no torch and no redback, and `list_samplers()` still
# names the GPU entries -- they are available, requiring the [gpu] extra.
from .backends import (  # noqa: E402
    check_gpu,
    env_report,
    env_script,
    gpu_list,
    n_jobs,
    require_jax,
    x64_enabled,
)
from .backends import _registration as _gpu_registration  # noqa: E402,F401

# Model-binding factories for the photometric JAX models. These stay factories on purpose: a band
# flux needs a filter set, a redshift and a luminosity distance, and a guessed distance silently
# rescales every fitted mass. Resolved lazily -- naming one without the [gpu] extra installed
# raises a message naming the extra.
from .models.jax import (  # noqa: E402
    flare_model,
    kilonova_model,
    kilonova_three_model,
    kilonova_two_model,
    register_kilonova,
    register_kilonova_three,
    register_kilonova_two,
    register_supernova,
    register_tde,
    supernova_model,
    supernova_models,
    tde_model,
)

__all__ += [
    # backends / GPU environment
    "check_gpu", "require_jax", "x64_enabled", "gpu_list", "n_jobs", "env_script", "env_report",
    # JAX model binding factories
    "register_kilonova", "register_kilonova_two", "register_kilonova_three",
    "register_tde", "register_supernova", "supernova_models",
    "flare_model", "kilonova_model", "kilonova_two_model", "kilonova_three_model",
    "tde_model", "supernova_model",
]
