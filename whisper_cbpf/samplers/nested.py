"""Nested sampling via ``dynesty`` -- the one sampler that returns the **log-evidence** ``ln Z``.

Same pluggable contract as every other sampler: ``fit(lc, model, prior=None, ...) -> SamplerResult``.
It reuses Whisper's own likelihood layer (:func:`~whisper_cbpf.likelihood.make_likelihood`), so it
respects the light curve's ``data_mode`` exactly as ``mcmc`` does -- flux space for flux data,
magnitude space for magnitude data -- and should converge to the same posterior as ABC / ABC-SMC /
MCMC / SNPE on the same data.

**The prior enters through the unit-cube transform, NOT through the likelihood.** Nested sampling
integrates ``Z = int L(theta) pi(theta) dtheta`` by mapping the unit hypercube through the prior's
inverse CDF (:meth:`whisper_cbpf.priors.Prior.rescale`) and evaluating the *likelihood alone* on the
result. So :func:`_log_likelihood` here deliberately does **not** add ``prior.log_prob`` -- that is
the one place this module differs from :func:`whisper_cbpf.samplers.mcmc._log_prob`, which sits one
module away and does add it. Adding it would count the prior twice and destroy ``ln Z``.
``tests/test_nested.py::test_prior_enters_exactly_once`` is the executable guard.

Two consequences worth knowing before you switch samplers:

* ``nested`` needs every prior distribution to implement ``rescale(u)`` and never calls
  ``log_prob``; ``mcmc`` needs ``log_prob`` and never calls ``rescale``. A user-supplied
  distribution with only ``log_prob`` is refused up front by :func:`_check_rescalable` rather than
  failing inside dynesty's live-point initialisation with a bare ``AttributeError``.
* ``max_log_likelihood`` here is **exact**, not reconstructed. ``dynesty``'s ``results.logl`` is the
  pure log-likelihood at every dead point, so there is no ``log_prob - prior.log_prob`` subtraction
  of the kind ``mcmc.py`` needs (and no LogUniform bias to defend against).

**Naming hazard:** Whisper's class is :class:`NestedSampler` and dynesty's factory is *also* called
``NestedSampler``. Never ``from dynesty import NestedSampler`` in this module -- ``import dynesty``
(lazily, inside :meth:`NestedSampler.fit`, matching ``mcmc.py``'s ``import emcee``) and go through
the module handle.

``dynesty`` is a core dependency (no extra needed). Exercised here against **3.0.0**; the module
uses only names present in **2.1.5** as well, and deliberately avoids the arguments 2.1.5 accepts
but 3.0.0 removed (``npdim``, ``gradient``, ``compute_jac``, ``fmove``, ``max_move``,
``update_func``, ``save_history``).
"""
from __future__ import annotations

import contextlib
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..likelihood import make_likelihood
from ..models import get_model
from .base import (
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)


def _prior_transform(u, names, prior):
    """dynesty's ``prior_transform``: unit hypercube -> parameter vector, via ``Prior.rescale``.

    ``Prior.rescale`` returns a dict; dynesty needs an array ordered like ``names``. The order is
    safe by construction: ``Prior.names`` is ``list(self.distributions)`` and ``Prior.rescale``
    enumerates ``self.distributions.items()``, so both follow the same insertion order.

    Module-level (not a closure or a method) so it survives pickling to a ``n_jobs > 1`` worker;
    the per-fit state travels in dynesty's ``ptform_args``, exactly as ``mcmc.py`` passes ``args=``
    to emcee.
    """
    v = prior.rescale(u)
    return np.array([v[n] for n in names], dtype=float)


def _log_likelihood(theta, names, predict, times, bands, likelihood, scatter_name=None):
    """log-LIKELIHOOD only -- **not** the log-posterior. See the module docstring.

    ``scatter_name`` routes that prior parameter to the likelihood as its extra-scatter term
    (:class:`~whisper_cbpf.likelihood.GaussianLikelihoodWithScatter`) instead of the model, exactly
    as ``mcmc`` does; the model's ``predict`` still receives the full dict (models ignore unknown
    keys).

    Returns ``-inf`` for a non-finite likelihood. **The ``isfinite`` guard is load-bearing, not
    decoration**: a ``NaN`` reaching dynesty raises ``ValueError: The log-likelihood of live point
    is invalid.`` and kills the run (measured -- a 2-d toy returning ``NaN`` on half its prior box
    raises exactly that). ``-inf`` is the choice because it is the honest value for a forbidden
    draw; a large finite sentinel measured *no worse* here (2-d Gaussian, half the prior box
    forbidden, analytic ``ln Z = -5.2983``: ``-inf`` gave ``-5.2042 +/- 0.1380`` in 1.28 s / 17 203
    calls, ``-1e300`` gave ``-5.2046 +/- 0.1379`` in 1.28 s / 17 218 calls -- indistinguishable),
    so this is a correctness preference, not a measured speed-up.
    """
    params = {nm: float(v) for nm, v in zip(names, theta)}
    model_flux = np.asarray(predict(params, times, bands), dtype=float)
    if scatter_name is not None:
        ll = likelihood.log_likelihood(model_flux, sigma_extra=params[scatter_name])
    else:
        ll = likelihood.log_likelihood(model_flux)
    return float(ll) if np.isfinite(ll) else -np.inf


def _check_rescalable(prior, names):
    """Refuse a prior whose distributions cannot map the unit cube, *before* any sampling starts.

    Without this, a distribution carrying only ``sample``/``log_prob``/``bounds`` fails inside
    dynesty's live-point initialisation with a bare ``AttributeError: 'X' object has no attribute
    'rescale'`` -- no parameter name, no sampler name, no remedy. ``TypeError`` (not ``ValueError``)
    to match ``samplers/jax/abc_gpu.py``'s choice for the same class of refusal.
    """
    bad = {nm: type(prior.distributions[nm]).__name__
           for nm in names if not callable(getattr(prior.distributions[nm], "rescale", None))}
    if bad:
        raise TypeError(
            "nested sampling maps the UNIT CUBE through the prior, so every distribution must "
            "implement rescale(u) -> value. These do not: "
            + ", ".join(f"{nm} ({cls})" for nm, cls in bad.items())
            + ". Add a rescale method (see whisper_cbpf.priors.Uniform.rescale, two lines), or use "
              "sampler='mcmc', which needs only log_prob.")


class NestedSampler(BaseSampler):
    """Nested sampling via ``dynesty`` -- posterior **and** log-evidence. See the module docstring."""

    name = "nested"

    def fit(self, lc, model, prior=None, *, nlive=None, dynamic=False, dlogz=None,
            bound="multi", sample="auto", maxiter=None, maxcall=None, maxbatch=None,
            pfrac=0.8, space="auto", likelihood="auto", seed=0, progress=False,
            n_jobs=None) -> SamplerResult:
        """Fit ``lc`` with ``model`` via dynesty nested sampling, returning a :class:`SamplerResult`.

        There is no ``initial_guess``, ``initial_scatter``, ``burnin``, ``thin``, ``nwalkers`` or
        ``moves`` here, and their absence is the point: nested sampling starts from the prior and
        has no burn-in to discard.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry errors (``flux_err`` / ``magnitude_err``).
        model : str or Model
            Registered model name or a :class:`~whisper_cbpf.models.Model`.
        prior : Prior, optional
            Parameter prior; defaults to the model's ``default_prior`` (``ValueError`` if neither).
            **Every distribution must implement ``rescale(u)``** -- see :func:`_check_rescalable`.
        nlive : int, optional
            Live points; default ``max(500, 25*ndim)``. 500 is dynesty's own default and the
            ``25*ndim`` floor only binds above ``ndim = 20``, where it keeps ``bound='multi'``'s
            covariance estimate well determined. **This is the knob that matters**: it sets both the
            evidence error (``~sqrt(H/nlive)``) and the cost.
        dynamic : bool, default False
            Use ``DynamicNestedSampler``. Static is the default because dynesty's dynamic weighting
            defaults to ``pfrac=0.8``, i.e. 80% of the refinement budget on the *posterior* and 20%
            on the *evidence* -- tuning for the number this sampler does not exist to produce.
            Measured (``flare``, 3 params, 40 points, ``nlive=400``, ``phe_sbi``, one CPU core):
            static ``dlogz=0.1`` 6.3 s / ncall 25 624 / ``lnZ = 52.8955 +/- 0.2092`` / n_eff 1 848;
            dynamic (defaults) 14.8 s / ncall 39 010 / ``lnZ = 52.8790 +/- 0.0918`` / n_eff 10 287.
            So dynamic costs 2.4x the wall time and 1.5x the likelihood calls, and buys 2.3x tighter
            ``ln Z`` and 5.6x the effective posterior sample. The two agree (``dlnZ = 0.017``
            against a combined error of 0.229). Use it when you want the posterior; leave it off
            when you want a budgetable evidence.
        dlogz : float, optional
            Static stopping criterion on the estimated remaining evidence. ``None`` uses dynesty's
            own default ``1e-3*(nlive-1) + 0.01`` (0.509 at ``nlive=500``). Pass ``dlogz=0.01`` for
            an evidence-grade run. **Ignored when ``dynamic=True``** (dynamic stops on its own rule,
            starting from ``dlogz_init=0.01``).
        bound : {'multi', 'single', 'balls', 'cubes', 'none'}, default 'multi'
            Bounding distribution for the live points; passed straight to dynesty.
        sample : str, default 'auto'
            Sampling method ('unif', 'rwalk', 'slice', 'rslice', ...); ``'auto'`` picks ``unif``
            below ~10 dimensions, ``rwalk`` to ~20 and ``rslice`` above.
        maxiter, maxcall : int, optional
            Hard budget caps. Hitting either truncates the run: ``info['converged']`` becomes False
            and ``log_evidence`` is then only a **lower bound**.
        maxbatch : int, optional
            Dynamic only; bounds the number of refinement batches.
        pfrac : float, default 0.8
            Dynamic only; fraction of the batch weight placed on the **posterior** rather than the
            **evidence** (dynesty's ``weight_function`` computes
            ``pfrac*pweight + (1-pfrac)*zweight``). 0.8 is dynesty's own default. Set ``pfrac=0.0``
            to spend the whole refinement budget on ``ln Z``; whether that beats a static run at
            matched ``ncall`` is **unknown** -- it was never measured. What would settle it: three
            runs at matched ``ncall`` (static, dynamic ``pfrac=0.8``, dynamic ``pfrac=0.0``)
            compared on ``log_evidence_err``. Ignored (and reported as ``None``) when
            ``dynamic=False``.
        space : {'auto', 'flux', 'magnitude'}, default 'auto'
            Comparison space passed to the likelihood (``'auto'`` follows the data's ``data_mode``).
        likelihood : str, default 'auto'
            Likelihood ``kind`` for :func:`make_likelihood` (e.g. ``'gaussian'``,
            ``'gaussian_upper_limits'``, ``'gaussian_scatter'``, ``'mixture'``).
        seed : int, default 0
            RNG seed (``rstate=np.random.default_rng(seed)``). Reproducible for a fixed
            ``(seed, n_jobs)`` pair -- **not across ``n_jobs``**, see below.
        progress : bool, default False
            dynesty's ``print_progress`` bar.
        n_jobs : int, optional
            Worker processes (a ``ProcessPoolExecutor`` handed to dynesty as ``pool``, with
            ``queue_size=n_jobs``). Default ``None`` = serial, because **parallelism is usually a
            loss for a cheap model**: measured on ``flare`` at ``nlive=400, dlogz=0.1``, serial
            6.28 s against ``n_jobs=4`` 13.80 s at essentially equal ``ncall`` (25 624 vs 25 072) --
            2.2x *slower*, process overhead dominating a microsecond-scale ``predict``. Worth it
            only when one likelihood call is expensive (a ~0.1 s radiative-transfer model); the
            break-even per-call cost is **unknown** and would need a sweep of artificial ``predict``
            costs against ``n_jobs`` to settle.
            For ``n_jobs > 1`` the model's ``predict``, the prior and the likelihood must all be
            **picklable** (defined at module level, not closures/lambdas -- the same rule
            ``whisper_cbpf.models`` states for parallel ABC); otherwise use ``n_jobs=None``.
            **Reproducibility is per-``(seed, n_jobs)``, not per-``seed``**, and that is weaker than
            ``abc``'s guarantee of an ``n_jobs``-independent result: dynesty gives each of the
            ``queue_size`` workers its own seed sequence, so the worker count is part of the RNG
            stream. Measured, same seed: serial ``lnZ = 52.8955 +/- 0.2092`` against ``n_jobs=4``
            ``lnZ = 53.3573 +/- 0.2055`` -- a 1.6-sigma stochastic difference, not a bug.

        Returns
        -------
        SamplerResult
            Equal-weight posterior samples + summary, ``best_params`` (the maximum-likelihood dead
            point), exact ``max_log_likelihood`` / ``aic`` / ``bic``, and in ``info``:
            ``log_evidence`` and ``log_evidence_err`` (see below), ``information_nats``,
            ``n_effective``, ``niter``, ``ncall``, ``efficiency_percent``. The raw
            ``dynesty.results.Results`` is attached as ``result.dynesty_results`` (an attribute, not
            an ``info`` entry -- ``info`` is JSON-serialised by ``to_json`` and ``Results`` holds
            ndarrays).

        Notes
        -----
        **Where ln Z lives.** ``info['log_evidence']`` and ``info['log_evidence_err']``, not a
        ``SamplerResult`` field -- promoting it to a first-class field alongside ``aic``/``bic``
        needs an edit to ``samplers/base.py``, which this change was not scoped to touch. It is a
        single copy and it reaches JSON through ``to_dict()['info']``.

        **Sign convention: higher ``ln Z`` is better; lower AIC/BIC is better.** Any table ranking
        models by evidence must select with ``max``, where the AIC column uses ``min``.

        **``ln Z`` is prior-dependent in a way AIC and BIC are not.** ``Z`` is the likelihood
        averaged *over the prior*, so widening a prior over a region the data exclude lowers ``ln Z``
        by construction even though the fit is unchanged. Measured, same data and same ``flare``
        model (``nlive=400``, same seed), only ``amplitude``'s prior widened from
        ``Uniform(0, 10)`` to ``Uniform(0, 1000)`` (100x): ``ln Z`` moved **-4.095 +/- 0.442** (the
        analytic Occam expectation is ``-ln 100 = -4.605``; 1.2 sigma) while AIC and BIC moved
        **-0.010**, i.e. by sampler noise. An arbitrary but defensible-looking prior width moved
        ``ln Z`` by 4 nats -- "strong evidence" on the Jeffreys scale -- and left AIC/BIC untouched.
        Three rules follow:

        1. Only compare ``ln Z`` between models whose priors you would defend *before* seeing the data.
        2. Never compare two ``ln Z`` for the *same* model under *different* priors and call it model
           comparison -- that is the measurement above, and it means nothing.
        3. ``ln Z`` obeys the package's existing rule that information criteria compare **models,
           not spaces** (``docs/CHOOSING.md``): a flux-space and a magnitude-space ``ln Z`` are
           integrals of different likelihoods over different representations of the data. Compare
           down a column, never across.

        Always quote ``log_evidence_err``: a Bayes factor ``ln B_12 = lnZ_1 - lnZ_2`` smaller than
        ~3x ``hypot(err_1, err_2)`` is not a result. **BIC is a crude Laplace approximation to
        ``-2 ln Z``** that drops the O(1) Occam factor; **AIC is not an approximation to ``ln Z`` at
        all** (it targets out-of-sample predictive accuracy, like the WAIC/LOO this package already
        reports), so AIC and ``ln Z`` may legitimately disagree.

        **``n_samples`` overstates independence.** ``samples`` has one row per dead point
        (``niter + nlive``) and contains duplicates from the weighted resampling; judge a posterior
        CI against ``info['n_effective']`` instead. Measured on ``flare``: 7 198 rows carrying 1 806
        effective samples.
        """
        import dynesty

        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        lik = make_likelihood(lc, kind=likelihood, space=space)   # reuse the shared likelihood layer
        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        predict = model.predict
        names = list(prior.names)
        ndim = len(names)
        k, n = ndim, int(len(times))

        # Refuse an unusable prior BEFORE any sampling, with a message naming the parameter.
        _check_rescalable(prior, names)

        nlive = int(nlive) if nlive else max(500, 25 * ndim)

        # A prior parameter named after the likelihood's scatter term (GaussianLikelihoodWithScatter)
        # is a LIKELIHOOD parameter: routed to log_likelihood(sigma_extra=...), sampled like the rest.
        # Unlike distance-based ABC (which cannot identify it at all, see docs/CHOOSING.md), a
        # likelihood-based sampler fits it correctly. Same two lines as mcmc.py, deliberately.
        scatter_name = getattr(lik, "scatter_param", None)
        scatter_name = scatter_name if (scatter_name and scatter_name in names) else None

        logl_args = (names, predict, times, bands, lik, scatter_name)
        ptform_args = (names, prior)
        rstate = np.random.default_rng(int(seed))

        pool_kw = {}
        ctx = contextlib.nullcontext()
        if n_jobs and int(n_jobs) > 1:
            ctx = ProcessPoolExecutor(max_workers=int(n_jobs))
        # The sampler must be BUILT and RUN inside the `with`: dynesty calls pool.map during
        # construction (drawing the initial live points), not only during run_nested.
        with ctx as ex:
            if ex is not None:
                pool_kw = {"pool": ex, "queue_size": int(n_jobs)}
            t0 = time.perf_counter()
            try:
                if dynamic:
                    sampler = dynesty.DynamicNestedSampler(
                        _log_likelihood, _prior_transform, ndim, nlive=nlive, bound=bound,
                        sample=sample, rstate=rstate, logl_args=logl_args,
                        ptform_args=ptform_args, **pool_kw)
                    sampler.run_nested(nlive_init=nlive, maxbatch=maxbatch, maxiter=maxiter,
                                       maxcall=maxcall, wt_kwargs={"pfrac": float(pfrac)},
                                       print_progress=bool(progress))
                else:
                    sampler = dynesty.NestedSampler(
                        _log_likelihood, _prior_transform, ndim, nlive=nlive, bound=bound,
                        sample=sample, rstate=rstate, logl_args=logl_args,
                        ptform_args=ptform_args, **pool_kw)
                    sampler.run_nested(dlogz=dlogz, maxiter=maxiter, maxcall=maxcall,
                                       print_progress=bool(progress))
            except RuntimeError as exc:
                if "could not find a single point" not in str(exc):
                    raise
                raise RuntimeError(
                    "nested sampling could not find one prior draw with a finite log-likelihood in "
                    "1000 attempts. The usual cause is a prior that cannot reach the data's scale "
                    "-- e.g. the analytic toys' amplitude prior is Uniform(0, 10), which cannot "
                    "reach a real transient's flux (see README). Check prior.bounds against the "
                    "light curve, or fit in magnitude space.") from exc
            runtime = time.perf_counter() - t0
            res = sampler.results

        # dynesty returns WEIGHTED dead points; every downstream consumer (summarize_posterior,
        # plot_corner, predictive_metrics) assumes equal weight. Resample with a FRESH generator
        # seeded from `seed` -- `rstate` has been advanced by the run, so reusing it would make the
        # resampling depend on run length. `samples_equal` IS dynesty.utils.resample_equal on
        # importance_weights(): verified bit-identical (tests/test_nested.py).
        resample_rng = np.random.default_rng(int(seed))
        samples = pd.DataFrame(np.asarray(res.samples_equal(rstate=resample_rng), dtype=float),
                               columns=names)

        # `res.logl` is the PURE log-likelihood at every dead point (the prior never entered it), so
        # the max-likelihood point is exact here -- no `log_prob - prior.log_prob` reconstruction of
        # the kind mcmc.py needs, and no LogUniform bias.
        i = int(np.nanargmax(res.logl))
        best_params = {nm: float(res.samples[i, j]) for j, nm in enumerate(names)}
        max_log_likelihood = float(res.logl[i])

        # logz / logzerr / information are arrays over iterations; the run's answer is the last entry.
        # ncall is a per-iteration array, NOT a scalar -- int(res.ncall) raises.
        ncall = int(np.sum(res.ncall))
        niter = int(res.niter)
        w = np.asarray(res.importance_weights(), dtype=float)
        hit_cap = ((maxiter is not None and niter >= int(maxiter))
                   or (maxcall is not None and ncall >= int(maxcall)))
        if hit_cap:
            warnings.warn(
                "nested sampling stopped on maxiter/maxcall, not on dlogz, so the run is truncated: "
                "log_evidence is a LOWER BOUND and its error is not trustworthy. Raise "
                "maxcall/maxiter or lower nlive.", stacklevel=2)
        info = {
            "nlive": int(nlive), "dynamic": bool(dynamic),
            "dlogz": (None if dynamic else (float(dlogz) if dlogz is not None
                                            else 1e-3 * (nlive - 1) + 0.01)),
            "pfrac": (float(pfrac) if dynamic else None),
            "bound": str(bound), "sample": str(sample),
            "space": lik.space, "likelihood": type(lik).__name__,
            "log_evidence": float(res.logz[-1]),          # higher is better (AIC/BIC: lower is better)
            "log_evidence_err": float(res.logzerr[-1]),
            "niter": niter, "ncall": ncall,
            "efficiency_percent": float(res.eff),
            "information_nats": float(np.asarray(res.information)[-1]),
            "n_effective": float(1.0 / np.sum(w ** 2)),
            "scatter_param": scatter_name,
            "n_jobs": int(n_jobs) if n_jobs and int(n_jobs) > 1 else 1,
            "converged": bool(not hit_cap), "seed": int(seed),
        }
        attach_band_metrics(info, lc, model.name, best_params, space)
        aic, bic = aic_bic(max_log_likelihood, k, n)
        result = SamplerResult(
            # "nested" even when reached through the "dynesty" alias -- one label in the comparison
            # table, matching how "npe" reports as "snpe".
            sampler="nested", model=model.name, parameters=list(names), samples=samples,
            summary=summarize_posterior(samples, names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=aic, bic=bic,
        )
        result.dynesty_results = res
        attach_predictive_metrics(result, lc, space)
        return result


def fit_nested(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via dynesty nested sampling. See :meth:`NestedSampler.fit`."""
    return NestedSampler().fit(lc, model, prior=prior, **kwargs)
