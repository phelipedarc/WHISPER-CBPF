"""Pluggable samplers: a small registry + a generic ``fit`` dispatcher.

Add a new sampler by subclassing :class:`BaseSampler` and calling :func:`register_sampler`.
"""
from __future__ import annotations

from .abc import ABCSampler, fit_ABC
from .abc_smc import ABCSMCSampler, fit_ABC_SMC
from .base import (
    BaseSampler,
    SamplerResult,
    aic_bic,
    attach_band_metrics,
    attach_predictive_metrics,
    check_burnin,
    prepare_lc,
    summarize_posterior,
)
from .mcmc import MCMCSampler, fit_MCMC
from .nested import NestedSampler, fit_nested
from .snpe import SNPESampler, fit_SNPE

# "snpe" and "npe" both map to the same sampler (num_rounds=1 is amortized NPE, >1 is sequential).
# "nested" and "dynesty" are likewise the same sampler ("dynesty" names the library it wraps); a
# result from either reports sampler="nested", so the comparison table carries one label.
_SAMPLERS = {
    "abc": ABCSampler, "abc_smc": ABCSMCSampler, "mcmc": MCMCSampler,
    "nested": NestedSampler, "dynesty": NestedSampler,
    "snpe": SNPESampler, "npe": SNPESampler,
}


#: The samplers whose ``fit`` takes ``init=`` (named, so listing them imports no GPU stack).
_TAKE_INIT = ("mcmc", "emcee_jax", "nuts_gpu", "pymc_jax_gpu_vectorized",
              "pymc_jax_gpu_parallelized")


def register_sampler(name, sampler_cls, *, overwrite=False):
    """Register a sampler class so :func:`fit` and :func:`whisper_cbpf.compare` take it by name.

    Parameters
    ----------
    name : str
        The name to register it under.
    sampler_cls : type
        A :class:`BaseSampler` subclass whose ``fit(lc, model, prior=None, **kwargs)`` returns a
        :class:`SamplerResult`. Its ``fit`` applies the pre-event rule and records its provenance
        like every built-in sampler's.
    overwrite : bool, default False
        Replace a sampler already registered under ``name``.

    Raises
    ------
    ValueError
        ``name`` is taken and ``overwrite`` is False.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.samplers import ABCSampler
    >>> class QuickABC(ABCSampler):
    ...     name = "quick_abc"
    >>> wp.register_sampler("quick_abc", QuickABC, overwrite=True)
    >>> "quick_abc" in wp.list_samplers()
    True
    """
    if name in _SAMPLERS and not overwrite:
        raise ValueError(f"Sampler {name!r} already registered (pass overwrite=True).")
    _SAMPLERS[name] = sampler_cls


def get_sampler(name):
    """A new instance of the sampler registered under ``name``.

    Parameters
    ----------
    name : str
        A name from :func:`list_samplers`.

    Returns
    -------
    BaseSampler

    Raises
    ------
    KeyError
        No sampler is registered under ``name``; the message lists the ones that are.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.samplers import get_sampler
    >>> type(get_sampler("mcmc")).__name__
    'MCMCSampler'
    """
    if name not in _SAMPLERS:
        raise KeyError(f"Unknown sampler {name!r}. Available: {list_samplers()}")
    return _SAMPLERS[name]()


def list_samplers():
    """The names of every registered sampler, sorted.

    The JAX samplers (``emcee_jax``, ``nuts_gpu``, ``abc_gpu``, ...) are listed when the ``[gpu]``
    extra is installed.

    Returns
    -------
    list of str

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> {"abc", "mcmc", "nested"} <= set(wp.list_samplers())
    True
    """
    return sorted(_SAMPLERS)


def _gpu_visible():
    """True when JAX is installed and sees a GPU (never imports JAX for a CPU-only model)."""
    try:
        import jax
        return any(d.platform in ("gpu", "cuda") for d in jax.devices())
    except Exception:                                         # noqa: BLE001 - no JAX, no GPU
        return False


def _auto_sampler(model, gpu_visible=_gpu_visible):
    """``sampler="auto"``: ``emcee_jax`` for a model with a JAX half when JAX sees a GPU, else CPU
    ``mcmc``. Both are likelihood samplers, so they fit upper limits too."""
    from .base import _resolve_model

    m = _resolve_model(model)
    if getattr(m, "predict_jax", None) is not None and "emcee_jax" in _SAMPLERS \
            and gpu_visible():
        return "emcee_jax"
    return "mcmc"


def fit(lc, model, sampler="auto", *, likelihood_max_opt=False, **kwargs) -> SamplerResult:
    """Fit a light curve with a model, by any registered sampler.

    What every fit does by default, whatever the sampler:

    - **Pre-event data are never fitted.** Rows at or before the event -- day 0 of a light curve
      whose explosion or merger date is set (``lc.set_explosion_date``), or, for a model that
      fits its explosion time (``t_exp``), the rows before the first detection -- are left out of
      the likelihood and of the model evaluation. ``info["excluded_pre_event"]`` counts them and
      one warning says so (:func:`~whisper_cbpf.samplers.base.prepare_lc`).
    - **The explosion time's prior comes from the data** when the prior passed here does not
      name ``t_exp``: from the last non-detection before the first detection to the first
      detection (:meth:`~whisper_cbpf.LightCurve.explosion_time_prior`), cut to the model's own
      Uniform ``t_exp`` prior. ``info["t_exp_prior"]`` records it.
    - **Upper limits are fitted.** With ``space="auto"`` a light curve with limits after the
      event is fitted in flux space with the censored likelihood (the limits' significance is
      ``lc.meta["upper_limit_sigma"]``, 5 for the survey presets, else 5). ABC and SNPE cannot
      use a limit and say so.

    Parameters
    ----------
    lc : LightCurve
        The data.
    model : str or Model
        A registered model name or object.
    sampler : str, default "auto"
        A name from :func:`list_samplers`. ``"auto"`` (the same rule as
        :func:`whisper_cbpf.compare`): ``emcee_jax`` for a model with a JAX half (``predict_jax``)
        when JAX sees a GPU, CPU ``mcmc`` otherwise. Both fit upper limits, so a survey alert
        needs nothing but the data and the model name.
    likelihood_max_opt : bool, default False
        Also climb to the likelihood peak behind the fit's best draw
        (:meth:`SamplerResult.likelihood_max_opt
        <whisper_cbpf.samplers.base.SamplerResult.likelihood_max_opt>`), kept in
        ``info["likelihood_max_opt"]``: the maximum likelihood AIC and BIC should come from. The
        posterior is not changed. If the optimisation fails (a prior it cannot climb in, say), the
        fit is still returned, with the reason in ``info["likelihood_max_opt_error"]`` and a
        warning.
    **kwargs
        Passed to the sampler's ``fit``. ``init=`` (where the chains start: ``"prior_scan"``,
        ``"prior"``, a point, one start per chain, or a previous result such as an ABC fit) is
        taken by the MCMC-type samplers: ``mcmc``, ``emcee_jax``, ``nuts_gpu`` and
        ``pymc_jax_gpu_*``.

    Returns
    -------
    SamplerResult

    Raises
    ------
    ValueError
        When ``init=`` goes to a sampler that has no starting point (ABC, nested sampling, SNPE
        draw from the prior by construction); "not enough data" when no detection is left after
        the event; when ABC or SNPE is given a light curve with upper limits after the event.

    Examples
    --------
    >>> import numpy as np, whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
    >>> abc = wp.fit(lc, "flare", sampler="abc", n_simulations=5000, quantile=0.01)
    >>> post = wp.fit(lc, "flare", sampler="mcmc", init=abc, nsteps=1500, burnin=500,
    ...               likelihood_max_opt=True)
    >>> peak = post.info["likelihood_max_opt"]
    >>> post.info["init"], peak["max_log_likelihood"] >= post.max_log_likelihood
    ('result', True)

    Data and a model name alone (here CPU ``mcmc``: ``flare`` has no JAX half):

    >>> wp.fit(lc, "flare", nsteps=1000, burnin=300).sampler
    'mcmc'
    """
    if sampler == "auto":
        sampler = _auto_sampler(model)
    s = get_sampler(sampler)
    if "init" in kwargs:
        import inspect
        if "init" not in inspect.signature(s.fit).parameters:
            takers = [nm for nm in _TAKE_INIT if nm in _SAMPLERS]
            raise ValueError(
                f"sampler {sampler!r} has no starting point to set, so init= does nothing there "
                f"(it draws from the prior by construction). init= is for {takers}; to hand a "
                f"result over, fit it first and pass it as init= to one of those.")
    result = s.fit(lc, model, **kwargs)
    if likelihood_max_opt:
        from ..models import Model
        from .base import _warn_without_raising
        try:
            result.likelihood_max_opt(lc, model if isinstance(model, Model) else None)
        except (ValueError, TypeError, NotImplementedError) as exc:
            # The fit itself succeeded and may have taken hours: keep it, say why it has no peak.
            result.info["likelihood_max_opt_error"] = f"{type(exc).__name__}: {exc}"
            _warn_without_raising(f"the fit finished, but its maximum-likelihood optimisation "
                                  f"failed, so it has no info['likelihood_max_opt'] "
                                  f"({type(exc).__name__}: {exc}). AIC and BIC are the sampler's "
                                  f"best draw's.")
    return result


__all__ = [
    "BaseSampler", "SamplerResult", "summarize_posterior",
    "ABCSampler", "fit_ABC", "ABCSMCSampler", "fit_ABC_SMC", "MCMCSampler", "fit_MCMC",
    "NestedSampler", "fit_nested", "SNPESampler", "fit_SNPE", "fit",
    "register_sampler", "get_sampler", "list_samplers", "prepare_lc",
]

# Bind the JAX subpackage as an attribute so `whisper_cbpf.samplers.jax.<module>` resolves.
# Its __init__ imports nothing heavier than the standard library -- every JAX name inside
# it is lazy -- so this costs nothing on a CPU-only install.
from . import jax  # noqa: E402,F401
