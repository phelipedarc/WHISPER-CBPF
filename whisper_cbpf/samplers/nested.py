"""Nested sampling via ``dynesty`` -- the one sampler that returns the **log-evidence** ``ln Z``.

Same pluggable contract as every other sampler: ``fit(lc, model, prior=None, ...) -> SamplerResult``.
It reuses Whisper's own likelihood layer (:func:`~whisper_cbpf.likelihood.make_likelihood`), so it
respects the light curve's ``data_mode`` exactly as ``mcmc`` does -- flux space for flux data,
magnitude space for magnitude data -- so it targets the same posterior as ``mcmc`` on the same data
and prior. That much is measured: ``tests/test_nested.py::test_nested_agrees_with_mcmc`` holds the
two medians to 20 % on a flare toy, and ``test_log_evidence_matches_an_analytic_integral`` holds
``ln Z`` to 4 of its quoted errors. ABC / ABC-SMC and SNPE approximate that posterior and, at the
budgets ``examples/compare_samplers.py`` ships, come out wider (ABC / ABC-SMC also shifted); nested
is not in that comparison.

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
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..likelihood import make_likelihood
from ..models import get_model
from .jax import _diagnostics as _dg                                # numpy-only; no JAX import
from .base import (
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
    _warn_user,
)

#: Worker start method for the ``n_jobs > 1`` pool. "spawn", not the Linux default "fork", for the
#: reason ``samplers/abc.py``'s ``_MP_CONTEXT`` measures: a parent with an initialised XLA runtime
#: forks children whose inherited mutexes stay locked, and the fit hangs instead of failing.
_MP_CONTEXT = multiprocessing.get_context("spawn")


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


#: Prior draws that estimate the fraction of a constrained model's prior inside its constraint wall
#: (:func:`_allowed_fraction`): the error on ``ln f`` is ``sqrt((1 - f) / (f N))``, 0.012 nats at
#: ``f = 0.3``.
CONSTRAINT_FRACTION_DRAWS = 20000


def _ctx_ok(ctx, params):
    """A JAX factory model's constraint wall at ``params`` (its parameters by name)."""
    return bool(ctx.physical([params[k] for k in ctx.params]))


def _adapter_ok(adapter, params):
    """The redback adapter's constraint wall at ``params`` (its free parameters by name)."""
    return bool(adapter.physical({**{k: params[k] for k in adapter.parameters}, **adapter.pinned}))


def _wall_check(model):
    """``ok(params) -> bool`` for a model that carries redback's constraint wall (the JAX supernova
    and TDE factories, the redback adapter), else ``None``. Module-level partials, so the check
    pickles to an ``n_jobs`` worker with the model."""
    import functools

    ctx = getattr(getattr(model, "predict_jax", None), "ctx", None)
    if ctx is not None and getattr(ctx, "constraint_model", None) is not None \
            and callable(getattr(ctx, "physical", None)):
        return functools.partial(_ctx_ok, ctx)
    pred = getattr(model, "predict", None)
    if callable(getattr(pred, "physical", None)) and hasattr(pred, "pinned") \
            and hasattr(pred, "parameters"):
        from ..models.redback_adapter import _constraint_predicate
        if _constraint_predicate(pred.model, pred.constraint) is not None:
            return functools.partial(_adapter_ok, pred)
    return None


def _allowed_fraction(wall, prior, names, fixed, seed, n=CONSTRAINT_FRACTION_DRAWS):
    """``(f, n_allowed, n)``: the fraction of ``n`` prior draws that pass ``wall``."""
    draws = _dg.prior_draws(prior, names, n, int(seed) + 7919)
    extra = dict(fixed or {})
    n_ok = sum(bool(wall({**dict(zip(names, row)), **extra})) for row in draws)
    return n_ok / float(n), int(n_ok), int(n)


def _log_likelihood(theta, names, predict, times, bands, likelihood, scatter_name=None,
                    fixed=None, wall=None):
    """log-LIKELIHOOD only -- **not** the log-posterior. See the module docstring.

    ``scatter_name`` routes that prior parameter to the likelihood as its extra-scatter term
    (:class:`~whisper_cbpf.likelihood.GaussianLikelihoodWithScatter`) instead of the model, exactly
    as ``mcmc`` does; the model's ``predict`` still receives the full dict (models ignore unknown
    keys). ``fixed`` (``{name: value}``) adds the prior's ``Fixed`` parameters, which are not
    dimensions of the unit cube.

    Returns ``-inf`` for a non-finite likelihood. **The ``isfinite`` guard is load-bearing, not
    decoration**: a ``NaN`` reaching dynesty raises ``ValueError: The log-likelihood of live point
    is invalid.`` and kills the run (measured -- a 2-d toy returning ``NaN`` on half its prior box
    raises exactly that). ``-inf`` is the choice because it is the honest value for a forbidden
    draw; a large finite sentinel measured *no worse* here (2-d Gaussian, half the prior box
    forbidden, analytic ``ln Z = -5.2983``: ``-inf`` gave ``-5.2042 +/- 0.1380`` in 1.28 s / 17 203
    calls, ``-1e300`` gave ``-5.2046 +/- 0.1379`` in 1.28 s / 17 218 calls -- indistinguishable),
    so this is a correctness preference, not a measured speed-up.

    ``wall`` (:func:`_wall_check`) makes a draw behind the model's constraint wall ``-inf`` too: the
    prior density is zero there. The model's own ``predict`` would return zero flux, whose finite
    likelihood would count the walled region as a dark source.
    """
    params = {nm: float(v) for nm, v in zip(names, theta)}
    if fixed:
        params.update(fixed)
    if wall is not None and not wall(params):
        return -np.inf
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
            A ``Fixed`` parameter is held at its value: it is not a dimension of the unit cube,
            not counted in AIC/BIC (``n_params``), and appears in ``samples`` / ``best_params``
            as a constant and in ``info["fixed"]``.
        nlive : int, optional
            Live points; default ``max(500, 25*ndim)``. 500 is dynesty's own default and the
            ``25*ndim`` floor only binds above ``ndim = 20``, where it keeps ``bound='multi'``'s
            covariance estimate well determined. **This is the knob that matters**: it sets both the
            evidence error (``~sqrt(H/nlive)``) and the cost.
        dynamic : bool, default False
            Use ``DynamicNestedSampler``. Static is the default because dynesty's dynamic weighting
            defaults to ``pfrac=0.8``, i.e. 80% of the refinement budget on the *posterior* and 20%
            on the *evidence* -- tuning for the number this sampler does not exist to produce.
            Measured (``flare``, 3 params, 40 points, ``nlive=400``, one CPU core):
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
            Sampling method ('unif', 'rwalk', 'slice', 'rslice', ...). ``'auto'`` resolves to
            ``'rwalk'`` up to 20 dimensions and ``'rslice'`` above: dynesty's own rule without its
            ``'unif'`` branch below 10 dimensions. On redback ``arnett`` against SN2025pgp
            (6 parameters, 29 points, ``nlive=100``) ``'unif'`` collapsed to 0.71% efficiency --
            its last 50 iterations cost 1 900 calls each -- and was still far from converging
            after 152 423 calls (745 s; uncapped, it stalled for 25+ minutes), while ``'rwalk'``
            converged in 85 593 calls (225 s serial). ``info['sample']`` records the method that
            ran. ``'unif'`` still runs when named and needs fewer calls on a simple posterior
            (``flare``: 25 469 calls against 119 547 for ``'rwalk'``, at about the same wall time,
            10.3 s against 11.2 s); the ``flare`` timings quoted elsewhere in this docstring were
            measured with it. A ``'unif'`` run whose efficiency ends below 1% warns.
        maxiter, maxcall : int, optional
            Hard budget caps. Hitting either truncates the run: ``info['converged']`` becomes False
            and ``log_evidence`` is then only a **lower bound**. **Neither bounds wall time**:
            dynesty checks them only between iterations (between batches of ``n_jobs`` proposals
            when parallel), and one ``'unif'`` iteration keeps redrawing until a point beats the
            likelihood threshold, however many calls that takes.
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
            Worker processes (a ``ProcessPoolExecutor`` started with ``spawn``, see ``_MP_CONTEXT``,
            handed to dynesty as ``pool``, with ``queue_size=n_jobs``). Default ``None`` = serial,
            because **parallelism is usually a loss for a cheap model**: measured on ``flare`` at
            ``nlive=400, dlogz=0.1`` with forked workers, serial 6.28 s against ``n_jobs=4``
            13.80 s at essentially equal ``ncall`` (25 624 vs 25 072) -- 2.2x *slower*, process
            overhead dominating a microsecond-scale ``predict``. Worth it
            only when one likelihood call is expensive (a ~0.1 s radiative-transfer model); the
            break-even per-call cost is **unknown** and would need a sweep of artificial ``predict``
            costs against ``n_jobs`` to settle.
            For ``n_jobs > 1`` the model's ``predict``, the prior and the likelihood must all be
            **picklable** (defined at module level, not closures/lambdas -- the same rule
            ``whisper_cbpf.models`` states for parallel ABC); otherwise use ``n_jobs=None``.
            ``spawn`` re-imports ``__main__``, so a script passing ``n_jobs > 1`` needs an
            ``if __name__ == "__main__":`` guard, and a ``predict`` defined in a notebook (or any
            ``__main__`` without a file) cannot be found by the workers -- the pool breaks with
            ``BrokenProcessPool``; put it in an importable module. Every worker pays a fresh
            interpreter and its imports before its first call. Measured, same seed, one session on
            a loaded machine:
            ``flare`` above at ``n_jobs=4`` 28.9 s spawned against 17.8 s forked; redback ``arnett``
            on SN2025pgp (``nlive=100``, ``n_jobs=16``) 54-69 s spawned against 59.7 s forked, the
            start-up lost in the noise once a call costs milliseconds. The results were
            bit-identical: the start method changes the time only.
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
            ``n_effective``, ``niter``, ``ncall``, ``sample`` (the method that ran) and
            ``efficiency_percent`` (dynesty's ``eff``: accepted points per 100 likelihood calls
            over the whole run, the number its progress line prints; ``'rwalk'`` sits near
            ``100/(ndim+20)``). The raw ``dynesty.results.Results`` is attached as
            ``result.dynesty_results`` (an attribute, not an ``info`` entry -- ``info`` is
            JSON-serialised by ``to_json`` and ``Results`` holds ndarrays).

        Notes
        -----
        **Where ln Z lives.** ``info['log_evidence']`` and ``info['log_evidence_err']``, not a
        ``SamplerResult`` field -- promoting it to a first-class field alongside ``aic``/``bic``
        needs an edit to ``samplers/base.py``, which this change was not scoped to touch. It is a
        single copy and it reaches JSON through ``to_dict()['info']``.

        **A model behind a constraint wall** (redback's ``Constraint`` priors, applied by the JAX
        supernova and TDE factories and the redback adapter): a draw behind the wall is ``-inf``
        and the prior is renormalised to the allowed region, as bilby normalises a constrained
        prior, so ``ln Z`` gains ``-ln f`` with ``f`` the allowed fraction of
        :data:`CONSTRAINT_FRACTION_DRAWS` prior draws (its error is added in quadrature to
        ``log_evidence_err``). ``info['constraint_prior']`` records ``allowed_fraction``,
        ``n_draws``, ``log_evidence_correction``, ``correction_err`` and
        ``log_evidence_before_correction``; it is ``None`` for a model without a wall. Without the
        renormalisation a walled model's ``ln Z`` is low by ``-ln f``: about 1 nat for Arnett, the
        magnetar and the TDE at their default priors.

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
        # A Fixed parameter is held at its value: it is not a dimension of the unit cube (a
        # zero-width dimension would only slow the bounding ellipsoids down) and it is not counted
        # in AIC/BIC; the likelihood sees it through `fixed`.
        all_names = list(prior.names)
        free_prior, names, fixed = _dg.split_fixed(prior, all_names, "nested")
        ndim = len(names)
        k, n = ndim, int(len(times))

        # Refuse an unusable prior BEFORE any sampling, with a message naming the parameter.
        _check_rescalable(free_prior, names)

        nlive = int(nlive) if nlive else max(500, 25 * ndim)

        # dynesty's own 'auto' picks 'unif' below 10 dimensions, and 'unif' redraws inside the
        # bounding ellipsoids until one point beats the likelihood threshold, with no cap: on a
        # posterior pressed against the prior edges one iteration can run for minutes. 'rwalk'
        # costs a fixed ndim + 20 calls per iteration. So 'auto' here is dynesty's rule without its
        # 'unif' branch; 'unif' runs only when asked for by name.
        if sample == "auto":
            sample = "rwalk" if ndim <= 20 else "rslice"

        # A prior parameter named after the likelihood's scatter term (GaussianLikelihoodWithScatter)
        # is a LIKELIHOOD parameter: routed to log_likelihood(sigma_extra=...), sampled like the rest.
        # Unlike distance-based ABC (which cannot identify it at all, see docs/CHOOSING.md), a
        # likelihood-based sampler fits it correctly. Same two lines as mcmc.py, deliberately.
        scatter_name = getattr(lik, "scatter_param", None)
        scatter_name = scatter_name if (scatter_name and scatter_name in all_names) else None

        wall = _wall_check(model)
        logl_args = (names, predict, times, bands, lik, scatter_name, fixed or None, wall)
        ptform_args = (names, free_prior)
        rstate = np.random.default_rng(int(seed))

        pool_kw = {}
        ctx = contextlib.nullcontext()
        if n_jobs and int(n_jobs) > 1:
            ctx = ProcessPoolExecutor(max_workers=int(n_jobs), mp_context=_MP_CONTEXT)
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
        equal = np.asarray(res.samples_equal(rstate=resample_rng), dtype=float)
        # every parameter as a column, a Fixed one constant at its value
        samples = pd.DataFrame(_dg.fill_fixed(equal, names, fixed, all_names), columns=all_names)

        # `res.logl` is the PURE log-likelihood at every dead point (the prior never entered it), so
        # the max-likelihood point is exact here -- no `log_prob - prior.log_prob` reconstruction of
        # the kind mcmc.py needs, and no LogUniform bias.
        i = int(np.nanargmax(res.logl))
        best = _dg.fill_fixed(np.asarray(res.samples[i], dtype=float), names, fixed, all_names)
        best_params = {nm: float(best[j]) for j, nm in enumerate(all_names)}
        max_log_likelihood = float(res.logl[i])

        # logz / logzerr / information are arrays over iterations; the run's answer is the last entry.
        # ncall is a per-iteration array, NOT a scalar -- int(res.ncall) raises.
        ncall = int(np.sum(res.ncall))
        niter = int(res.niter)
        w = np.asarray(res.importance_weights(), dtype=float)
        hit_cap = ((maxiter is not None and niter >= int(maxiter))
                   or (maxcall is not None and ncall >= int(maxcall)))
        if hit_cap:
            _warn_user(
                "nested sampling stopped on maxiter/maxcall, not on dlogz, so the run is truncated: "
                "log_evidence is a LOWER BOUND and its error is not trustworthy. Raise "
                "maxcall/maxiter or lower nlive.")
        # Only 'unif' has an unbounded cost per iteration. rwalk, slice and rslice spend a number of
        # calls set by ndim, so their efficiency is low by design (rwalk: about 100/(ndim+20) %).
        if sample == "unif" and float(res.eff) < 1.0:
            _warn_user(
                f"nested sampling efficiency was {float(res.eff):.2f}%: sample='unif' needed more "
                "than 100 likelihood calls per accepted point, the collapse that can stall a run "
                "for tens of minutes. Use sample='rwalk' (what 'auto' resolves to).")
        log_z, log_z_err = float(res.logz[-1]), float(res.logzerr[-1])
        wall_info = None
        if wall is not None:
            # The prior inside the wall, renormalised: Z = int_allowed L pi / f, the normalisation
            # bilby gives a constrained prior. Without it every walled model's ln Z is low by -ln f.
            frac, n_ok, n_draws = _allowed_fraction(wall, free_prior, names, fixed, seed)
            if n_ok == 0:
                raise RuntimeError(
                    f"none of {n_draws} prior draws of model {model.name!r} passes its constraint "
                    f"wall, so its prior cannot be normalised. Check the prior against the model's "
                    f"constraints (whisper_cbpf.models.constraints).")
            corr, corr_err = -float(np.log(frac)), float(np.sqrt((1.0 - frac) / n_ok))
            wall_info = {"allowed_fraction": frac, "n_draws": n_draws,
                         "log_evidence_correction": corr, "correction_err": corr_err,
                         "log_evidence_before_correction": log_z}
            log_z, log_z_err = log_z + corr, float(np.hypot(log_z_err, corr_err))
        info = {
            "nlive": int(nlive), "dynamic": bool(dynamic),
            "dlogz": (None if dynamic else (float(dlogz) if dlogz is not None
                                            else 1e-3 * (nlive - 1) + 0.01)),
            "pfrac": (float(pfrac) if dynamic else None),
            "bound": str(bound), "sample": str(sample),
            "space": lik.space, "likelihood": type(lik).__name__,
            "log_evidence": log_z,                        # higher is better (AIC/BIC: lower is better)
            "log_evidence_err": log_z_err,
            "constraint_prior": wall_info,
            "niter": niter, "ncall": ncall,
            "efficiency_percent": float(res.eff),
            "information_nats": float(np.asarray(res.information)[-1]),
            "n_effective": float(1.0 / np.sum(w ** 2)),
            "scatter_param": scatter_name,
            "n_jobs": int(n_jobs) if n_jobs and int(n_jobs) > 1 else 1,
            "converged": bool(not hit_cap), "seed": int(seed),
            "fixed": dict(fixed),
        }
        attach_band_metrics(info, lc, model, best_params, space)
        aic, bic = aic_bic(max_log_likelihood, k, n)
        result = SamplerResult(
            # "nested" even when reached through the "dynesty" alias -- one label in the comparison
            # table, matching how "npe" reports as "snpe".
            sampler="nested", model=model.name, parameters=list(all_names), samples=samples,
            summary=summarize_posterior(samples, all_names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=aic, bic=bic,
        )
        result.dynesty_results = res
        attach_predictive_metrics(result, lc, space, model=model)
        return result


def fit_nested(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via dynesty nested sampling. See :meth:`NestedSampler.fit`.

    Parameters
    ----------
    lc : LightCurve
        The data.
    model : str or Model, default "flare"
    prior : Prior, optional
        Default: the model's.
    **kwargs
        The sampler's settings (``nlive``, ``dlogz``, ``sample``, ``seed``, ...).

    Returns
    -------
    SamplerResult
        With the evidence in ``info["log_evidence"]`` and ``info["log_evidence_err"]``.

    Examples
    --------
    A ``Fixed`` parameter is held at its value: two free parameters, so AIC and BIC count two.

    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=wp.get_model("flare").predict(truth, t),
    ...                    flux_err=np.full(30, 0.2))
    >>> prior = wp.Prior({"amplitude": wp.Uniform(0.0, 10.0), "rise_time": wp.Fixed(3.0),
    ...                   "decay_time": wp.Uniform(1.0, 50.0)})
    >>> res = wp.fit_nested(lc, "flare", prior=prior, nlive=100, seed=0)
    >>> res.n_params, res.info["fixed"], bool((res.samples["rise_time"] == 3.0).all())
    (2, {'rise_time': 3.0}, True)
    >>> bool(np.isfinite(res.info["log_evidence"]))
    True
    """
    return NestedSampler().fit(lc, model, prior=prior, **kwargs)
