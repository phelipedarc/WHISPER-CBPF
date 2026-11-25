"""Pluggable samplers: a small registry + a generic ``fit`` dispatcher.

Add a new sampler by subclassing :class:`BaseSampler` and calling :func:`register_sampler`.
"""
from __future__ import annotations

from .abc import ABCSampler, fit_ABC
from .abc_smc import ABCSMCSampler, fit_ABC_SMC
from .abc_window import abc_time_prior
from .base import (
    ZERO_FLUX_SNR,
    BaseSampler,
    SamplerResult,
    ZeroFluxWarning,
    aic_bic,
    attach_band_metrics,
    attach_predictive_metrics,
    check_burnin,
    prepare_lc,
    summarize_posterior,
    zero_flux_first_detections,
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


def fit(lc, model, sampler="auto", *, likelihood_max_opt=False, check_convergence=True,
        ABC_time_prior=False, ABC_time_prior_kwargs=None, **kwargs) -> SamplerResult:
    """Fit a light curve with a model, by any registered sampler.

    What every fit does by default, whatever the sampler:

    - **Pre-event data are never fitted.** Rows at or before the event -- day 0 of a light curve
      whose explosion or merger date is set (``lc.set_explosion_date``), or, for a model that
      fits its explosion time (``t_exp``), the rows before the first detection -- are left out of
      the likelihood and of the model evaluation. ``info["excluded_pre_event"]`` counts them and
      one warning says so (:func:`~whisper_cbpf.samplers.base.prepare_lc`).
    - **The explosion time's prior comes from the data** when the prior passed here does not
      name ``t_exp``: :func:`~whisper_cbpf.default_t_exp_prior`, from 30 days
      (:data:`~whisper_cbpf.io.schema.T_EXP_WINDOW_DAYS`) before the last non-detection that
      precedes the first detection -- or before the first detection when there is none -- to the
      first detection, cut to the model's own Uniform ``t_exp`` prior. A prior passed here is used
      as given. ``info["t_exp_prior"]`` records it.
    - **Upper limits are fitted.** With ``space="auto"`` a light curve with limits after the
      event is fitted in flux space with the censored likelihood (the limits' significance is
      ``lc.meta["upper_limit_sigma"]``, 5 for the survey presets, else 5). ABC and SNPE cannot
      use a limit and say so.
    - **An MCMC fit says whether it converged.** After an MCMC-type sampler (``mcmc``,
      ``emcee_jax``, ``nuts_gpu``, ``pymc_jax_gpu_*``) the fit runs :meth:`SamplerResult.diagnostics
      <whisper_cbpf.samplers.base.SamplerResult.diagnostics>` and, when a check fails, issues ONE
      short ``UserWarning``: a first line ``"Not converged:"``, then one plain sentence per failed
      check naming the metric, the parameter, the value and the threshold, and what to do. When
      the only failures are properties of the posterior itself (draws piled at a prior bound, too
      few data points), which a longer run does not change, the first line is ``"Check the
      posterior:"`` instead. For
      example ``"The 4 ensembles disagree on mstar (R-hat 1.18, needs < 1.01). Run longer
      (max_steps=8000) or check for a second mode."`` A converged fit is silent. The verdict is kept
      in ``info["convergence_verdict"]`` (``passed``, ``failed_checks``, ``message``); it replaces
      the sampler's own not-converged warning, whose problems stay in
      ``info["convergence_problems"]``.

    Parameters
    ----------
    lc : LightCurve
        The data.
    model : str or Model
        A registered model name or object.
    sampler : str, default "auto"
        A name from :func:`list_samplers`. ``"auto"``: ``emcee_jax`` for a model with a JAX half
        (``predict_jax``) when JAX sees a GPU, CPU ``mcmc`` otherwise. (:func:`whisper_cbpf.compare`
        has its own rule: nested sampling, for the evidence.) Both fit upper limits, so a survey alert
        needs nothing but the data and the model name.
    likelihood_max_opt : bool, default False
        Also climb to the likelihood peak behind the fit's best draw
        (:meth:`SamplerResult.likelihood_max_opt
        <whisper_cbpf.samplers.base.SamplerResult.likelihood_max_opt>`), kept in
        ``info["likelihood_max_opt"]``: the maximum likelihood AIC and BIC should come from. The
        posterior is not changed. If the optimisation fails (a prior it cannot climb in, say), the
        fit is still returned, with the reason in ``info["likelihood_max_opt_error"]`` and a
        warning.
    check_convergence : bool, default True
        Run the convergence checks after an MCMC-type fit and warn when one fails. ``False``: no
        check and no convergence warning at all (not even the sampler's own), and no
        ``info["convergence_verdict"]``.
    ABC_time_prior : bool, default False
        Before the fit, estimate the explosion time ``t_exp`` with whisper's rejection ABC on the
        GPU (``abc_gpu``, the model's JAX version, every detection in every band, no simulated
        noise) and fit with a flat ``t_exp`` prior over the accepted range plus a margin, cut to
        the default window (:func:`~whisper_cbpf.abc_time_prior`). A draw is accepted when the RMS
        of ``data - model`` in flux is at most 3 times the mean flux error of the detections.
        With fewer than 50 accepted draws it warns and keeps the default window. The window is
        chosen from the data the fit then uses, so intervals and evidences are mildly
        data-dependent. ``info["abc_time_prior"]`` records the step. Needs a model with ``t_exp``
        and a JAX version, and a ``prior`` that does not name ``t_exp``.
    ABC_time_prior_kwargs : dict, optional
        The ABC budget: ``n_simulations`` (1000000), ``seed`` (the fit's ``seed``),
        ``min_accepted`` (50), ``chunk`` and ``precision`` (``abc_gpu``'s).
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
        the event; when ABC or SNPE is given a light curve with upper limits after the event;
        with ``ABC_time_prior=True``, when the model has no ``t_exp`` or no JAX version, or
        ``prior`` names ``t_exp``.

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

    A run too short to converge warns, in plain English; ``check_convergence=False`` does not:

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     short = wp.fit(lc, "flare", sampler="mcmc", nsteps=200, burnin=50)
    >>> [str(w.message).splitlines()[0] for w in caught if "Not converged" in str(w.message)]
    ['Not converged:']
    >>> short.info["convergence_verdict"]["passed"]
    False
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     quiet = wp.fit(lc, "flare", sampler="mcmc", nsteps=200, burnin=50,
    ...                    check_convergence=False)
    >>> [w for w in caught if "onverged" in str(w.message)], "convergence_verdict" in quiet.info
    ([], False)

    The explosion-time window narrowed by an ABC run first (``ABC_time_prior=True``) is shown in
    :func:`~whisper_cbpf.abc_time_prior`'s example; ``result.info["abc_time_prior"]`` keeps it.
    """
    from . import abc_window as abc_t

    abc_on = abc_t._check_flag(ABC_time_prior)
    if not abc_on and ABC_time_prior_kwargs is not None:
        raise ValueError("ABC_time_prior_kwargs is the budget of the ABC_time_prior step: pass "
                         "ABC_time_prior=True with it, or leave it out.")
    options = abc_t._options(ABC_time_prior_kwargs, kwargs.get("seed")) if abc_on else None
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
    abc_record = None
    if abc_on:            # after the sampler checks above, so a bad call fails before the ABC run
        kwargs["prior"], abc_record = abc_t._fit_prior(lc, model, kwargs.get("prior"), options)
    from .jax._diagnostics import VERDICT_BY_CALLER

    token = VERDICT_BY_CALLER.set(True)          # this function gives the verdict, not the sampler
    try:
        result = s.fit(lc, model, **kwargs)
    finally:
        VERDICT_BY_CALLER.reset(token)
    if abc_record is not None:
        abc_t._stamp(result, abc_record)
    if likelihood_max_opt:
        from ..models import Model
        from .base import _warn_without_raising, _zero_flux_given
        try:
            with _zero_flux_given():            # the fit has given it already
                result.likelihood_max_opt(lc, model if isinstance(model, Model) else None)
        except (ValueError, TypeError, NotImplementedError) as exc:
            # The fit itself succeeded and may have taken hours: keep it, say why it has no peak.
            result.info["likelihood_max_opt_error"] = f"{type(exc).__name__}: {exc}"
            _warn_without_raising(f"the fit finished, but its maximum-likelihood optimisation "
                                  f"failed, so it has no info['likelihood_max_opt'] "
                                  f"({type(exc).__name__}: {exc}). AIC and BIC are the sampler's "
                                  f"best draw's.")
    if check_convergence:
        _convergence_verdict(result, sampler)
    return result


def _convergence_verdict(result, sampler):
    """``fit``'s verdict: after an MCMC-type fit, one plain-English warning when a check fails."""
    from ._verdict import MCMC_KINDS, verdict_heading, verdict_message, verdict_sentences
    from .base import _warn_user, _warn_without_raising

    info = result.info if isinstance(result.info, dict) else None
    peak = None
    if info is not None and isinstance(info.get("likelihood_max_opt"), dict):
        peak = info["likelihood_max_opt"].get("max_log_likelihood")
    try:
        report = result.diagnostics(likelihood_max_opt=peak)
    except Exception as exc:                                  # noqa: BLE001 - never lose the fit
        if info is not None:
            info["convergence_verdict"] = {"passed": None, "failed_checks": [],
                                           "message": None,
                                           "error": f"{type(exc).__name__}: {exc}"}
        _warn_without_raising(f"the fit finished, but its convergence could not be checked "
                              f"({type(exc).__name__}: {exc}). Run result.diagnostics() to see "
                              f"why.")
        return
    if report.kind not in MCMC_KINDS:
        # Not an MCMC fit: a sampler that flagged itself (a registered one, say) still warns.
        if info is not None and info.get("converged") is False and \
                info.get("convergence_problems"):
            from .jax._diagnostics import warn_if_unconverged
            warn_if_unconverged(str(sampler), info)
        return
    message = verdict_message(verdict_sentences(result, report),
                              heading=verdict_heading(result, report))
    if info is not None:
        info["convergence_verdict"] = {
            "passed": message is None,
            "failed_checks": [r.check for r in report.rows if r.passed is False],
            "message": message}
    if message is not None:
        _warn_user(message)


__all__ = [
    "BaseSampler", "SamplerResult", "summarize_posterior",
    "ABCSampler", "fit_ABC", "ABCSMCSampler", "fit_ABC_SMC", "MCMCSampler", "fit_MCMC",
    "NestedSampler", "fit_nested", "SNPESampler", "fit_SNPE", "fit",
    "register_sampler", "get_sampler", "list_samplers", "prepare_lc",
    "ZeroFluxWarning", "zero_flux_first_detections", "ZERO_FLUX_SNR", "abc_time_prior",
]

# Bind the JAX subpackage as an attribute so `whisper_cbpf.samplers.jax.<module>` resolves.
# Its __init__ imports nothing heavier than the standard library -- every JAX name inside
# it is lazy -- so this costs nothing on a CPU-only install.
from . import jax  # noqa: E402,F401
