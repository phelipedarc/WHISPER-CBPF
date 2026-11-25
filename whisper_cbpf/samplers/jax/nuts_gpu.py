"""GPU NUTS sampling via NumPyro, following whisper_cbpf's ``BaseSampler``/``SamplerResult``
contract exactly (same shape as ``whisper_cbpf.samplers.mcmc.MCMCSampler``).

Registered into whisper's own sampler registry as ``nuts_gpu`` when :mod:`whisper_cbpf` is
imported, so ``wp.fit(lc, model, sampler="nuts_gpu", ...)`` works through the normal dispatcher.
Whisper instantiates samplers with zero arguments, so every GPU option travels through ``fit()``.

Takes an already-built, jitted ``log_prob_fn`` (e.g. from
``whisper_cbpf.models.jax.flare.make_log_prob_jax``) rather than building its own from ``model`` —
this is what lets the emcee-vectorized-JAX arm and this NUTS arm share the *literal same*
Python object for the fairness comparison, instead of just "the same code written twice". That
sharing is only correct while every prior is Uniform: emcee wants a log-posterior and this sampler
a log-likelihood, so a density flagged ``includes_prior=True`` is refused when any prior is not
Uniform (it would count that prior twice) and warned about otherwise.

**Where the chains start, and what ``converged`` means.** By default the best-scoring of 1000 prior
draws are climbed a short way uphill, and each chain starts near a distinct one of those that reach
the best basin (``init_strategy="prior_scan"``); the result is then checked chain by chain. See
:mod:`whisper_cbpf.samplers.jax._diagnostics`, which holds the evidence for both.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

# Imported relatively -- the CPU core and this sampler now live in one package. In the legacy
# split (whisper-GPU @ 10796a0, discontinued; superseded by whisper_cbpf) this file sat in a
# separate repo cloned INTO a checkout of the CPU package, and reached that package absolutely
# off sys.path; the merge removes that indirection.

from ...models import get_model  # noqa: E402
from ...priors._numpy import family  # noqa: E402
from ...samplers.base import (  # noqa: E402
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    check_not_empty,
    summarize_posterior,
)
from . import _diagnostics as _dg  # noqa: E402
from ._adapters import free_density as _free_density  # noqa: E402
from ._adapters import resolve_density as _resolve_density  # noqa: E402
from ._adapters import resolve_sampling_names as _resolve_sampling_names  # noqa: E402


def _numpyro_priors(prior, names):
    """Map whisper's prior onto NumPyro sample sites, ONE PER PARAMETER.

    Returns one ``(site_name, numpyro_distribution, to_linear)`` triple per entry of ``names``,
    in that order. ``to_linear`` is ``None`` when the site *is* the model parameter, and a callable
    when the site is a reparameterisation that has to be pushed back to the parameter's own units.

    Reading ``.bounds`` for every parameter and building a single vectorised
    ``dist.Uniform(lows, highs)``. That silently substituted a uniform density for any parameter
    whose prior was not uniform -- a ``LogUniform(100, 6000)`` temperature floor became
    ``Uniform(100, 6000)``, which is a *different posterior*, not a different parameterisation:
    log-uniform puts half its mass below 775 K and uniform puts half below 3050 K. Nothing warned,
    because ``.bounds`` is identical for both.

    WHY LOG-UNIFORM IS SAMPLED AS A UNIFORM ON log10, NOT AS ``dist.LogUniform``
    ---------------------------------------------------------------------------
    ``dist.LogUniform`` carries the correct *density*, but not the geometry NUTS needs, and the two
    are independent. NumPyro walks a site in the unconstrained space that ``biject_to(d.support)``
    defines, and ``dist.LogUniform(lo, hi).support`` is ``constraints.interval(lo, hi)`` -- exactly
    what a plain ``Uniform`` declares. So ``biject_to`` returns (measured, NumPyro 0.21)::

        biject_to(dist.LogUniform(100., 6000.).support)
        -> ComposeTransform([SigmoidTransform(), AffineTransform()])

    a sigmoid on the LINEAR value. There is no log anywhere in it. This module's docstring used to
    claim "NumPyro picks the right bijector ... For LogUniform that is a log transform"; it is not,
    and the consequences are not cosmetic:

    * The unconstrained origin lands on the ARITHMETIC midpoint. For (100, 6000), ``u = 0`` maps to
      x = 3050 K -- the 83rd percentile of the log-uniform prior -- instead of the prior median
      ``sqrt(100 * 6000) = 774.6`` K.
    * NumPyro's default ``init_to_uniform(radius=2)`` draws u ~ U(-2, 2), which under that transform
      is x in [803.3, 5296.7] K -- the whole init distribution sits ABOVE the prior median (measured
      over 400 seeds: lowest init 818.0 K). A temperature floor whose true value is 107 K cannot be
      initialised anywhere near.
    * The step size cannot serve both ends. Measured, ``dx/du`` at the init edge u = -2 is 619.5
      against 6.99 at the truth -- a factor 88.6. In log10 coordinates the same two points differ by
      9.84, so one adapted step size covers the traverse instead of being wrong at one end or the
      other.

    ``Uniform(log10 lo, log10 hi)`` pushed through ``10**u`` is the identical measure -- it is the
    definition of log-uniform -- and puts ``u = 0`` on the prior median by construction. It is also
    exactly what ``whisper_cbpf.samplers.jax.pymc_gpu._pymc_prior`` does, so the two GPU NUTS arms
    now walk the SAME geometry and are comparable as parameterisations rather than differing in one.

    Simulation-based calibration on data generated FROM the model is what exposed this;
    ``tests/test_nuts_gpu_loguniform_geometry.py`` pins the geometry directly, without sampling, so
    the regression cannot come back silently.

    NORMAL AND TRUNCATED NORMAL
    ---------------------------
    A ``Normal`` is a ``dist.Normal`` site: its support is the real line, so NumPyro walks it with
    the identity. A ``TruncatedNormal`` is a ``Uniform(0, 1)`` site on its own CDF, ``cdf_<name>``,
    pushed through the exact inverse CDF (``priors.ppf_jax``) -- the probability integral transform,
    so the parameter has exactly the truncated density, and the chain walks the logit of the CDF,
    which is O(1) wide whether the truncation is tight or far out in the tails (a sigmoid on the
    linear value, as ``dist.TruncatedNormal``'s support gives, makes N(0, 1) cut to [-100, 100] a
    posterior 0.02 units wide). ``pymc_gpu._pymc_prior`` does the same.

    Only the distributions whisper's samplers support are mapped (a ``Fixed`` parameter never
    reaches here: ``fit`` leaves it out of the sampled vector); anything else raises rather than
    being quietly approximated.
    """
    from ...priors import ppf_jax

    out = []
    for nm in names:
        d = prior.distributions[nm]
        kind = family(d)
        lo, hi = (float(x) for x in d.bounds)
        if kind == "Uniform":
            out.append((nm, dist.Uniform(lo, hi), None))
        elif kind == "LogUniform":
            out.append((f"log10_{nm}", dist.Uniform(np.log10(lo), np.log10(hi)),
                        lambda u: 10.0 ** u))
        elif kind == "Normal":
            out.append((nm, dist.Normal(float(d.mu), float(d.sigma)), None))
        elif kind == "TruncatedNormal":
            out.append((f"cdf_{nm}", dist.Uniform(0.0, 1.0), ppf_jax(d)))
        else:
            raise TypeError(
                f"nuts_gpu cannot express prior {kind!r} on parameter {nm!r} as a NumPyro "
                f"distribution. Supported: Uniform, LogUniform, Normal, TruncatedNormal (and Fixed, "
                f"held at its value). Extend _numpyro_priors rather than falling back to a uniform "
                f"box -- that changes the posterior silently.")
    return out


def _to_site(d, x):
    """A parameter value -> the value of its NumPyro / PyMC site (``_numpyro_priors``)."""
    kind = family(d)
    if kind == "LogUniform":
        return np.log10(x)
    if kind == "TruncatedNormal":
        return np.clip(d.cdf(x), 1e-12, 1.0 - 1e-12)
    return np.asarray(x, dtype=float)


#: ``init_strategy=`` accepts these names as well as any NumPyro init callable. Spelled out rather
#: than resolved with ``getattr(numpyro.infer, ...)`` so a typo names the valid options instead of
#: reaching an arbitrary attribute. ``"uniform"`` is NumPyro's own default, and what this sampler
#: did before ``"prior_scan"`` became the default.
_INIT_STRATEGIES = {
    "uniform": numpyro.infer.init_to_uniform,
    "median": numpyro.infer.init_to_median,
    "sample": numpyro.infer.init_to_sample,
    "feasible": numpyro.infer.init_to_feasible,
    "mean": numpyro.infer.init_to_mean,
}

#: Starts computed here, one per chain, and handed to NumPyro as ``init_params``:
#: ``"prior_scan"`` -- the default -- is ``_diagnostics.scan_starts``: score 1000 prior draws, climb
#: the best 32 a short way uphill, and start each chain near a distinct one of those that reach the
#: best basin. ``"prior"`` starts each chain at an independent prior draw, unscored.
_WHISPER_STARTS = ("prior_scan", "prior")

#: What NumPyro's ``MCMC`` accepts. Checked here so a typo raises before XLA spends a minute
#: compiling the kernel, and with a message naming the alternatives.
_CHAIN_METHODS = ("vectorized", "parallel", "sequential")


def _resolve_init_strategy(init_strategy):
    """``None`` -> the default ``"prior_scan"``; a short name -> its label and, for NumPyro's
    strategies, the matching callable; a callable -> itself; a ``(num_chains, k)`` array of starts
    in parameter units -> ``"per_chain"``; a point (a dict, a ``(k,)`` array, or
    ``(point, scale)``) -> ``"ball"``; a previous ``SamplerResult`` -> ``"result"``.

    Returns ``(callable_or_None, label)``; the label is what goes into ``result.info`` so a saved
    run records which strategy actually ran rather than a repr of a function object. The callable
    is ``None`` exactly when the start is computed here (``_WHISPER_STARTS``, ``"per_chain"``,
    ``"ball"`` or ``"result"``).
    """
    if init_strategy is None:
        return None, "prior_scan"
    if _dg._is_result(init_strategy):
        return None, "result"
    if isinstance(init_strategy, dict) or (
            isinstance(init_strategy, tuple) and len(init_strategy) == 2
            and np.ndim(init_strategy[1]) == 0
            and (isinstance(init_strategy[0], dict) or np.ndim(init_strategy[0]) >= 1)):
        return None, "ball"
    if isinstance(init_strategy, str):
        key = init_strategy.strip().lower()
        if key in _WHISPER_STARTS:
            return None, key
        key = key[len("init_to_"):] if key.startswith("init_to_") else key
        if key not in _INIT_STRATEGIES:
            raise ValueError(
                f"nuts_gpu: unknown init_strategy {init_strategy!r}. Use one of "
                f"{list(_WHISPER_STARTS) + sorted(_INIT_STRATEGIES)} (the NumPyro names with or "
                f"without the 'init_to_' prefix), a (num_chains, k) array of starts, or a NumPyro "
                f"init callable such as numpyro.infer.init_to_value(values=...).")
        return _INIT_STRATEGIES[key], f"init_to_{key}"
    if callable(init_strategy):
        # A partial (init_to_value(values=...) returns one) has no __name__; its .func does, and
        # the point-start warning needs the real name.
        return init_strategy, getattr(getattr(init_strategy, "func", init_strategy), "__name__",
                                      repr(init_strategy))
    arr = None
    try:
        arr = np.asarray(init_strategy, dtype=float)
    except (TypeError, ValueError):
        pass
    if arr is not None and arr.ndim == 1:
        return None, "ball"
    if arr is None or arr.ndim != 2:
        raise TypeError(
            f"nuts_gpu: init / init_strategy must be None, one of "
            f"{list(_WHISPER_STARTS) + sorted(_INIT_STRATEGIES)}, a point (dict or (k,) array), "
            f"(point, scale), a (num_chains, k) array of starts, a previous result, or a NumPyro "
            f"init callable; got {type(init_strategy).__name__}.")
    return None, "per_chain"


def _unconstrained_starts(site_specs, starts, num_chains, dtype, prior, names):
    """Per-chain starts in parameter units -> NumPyro ``init_params`` (unconstrained, per site).

    Each site is the parameter, its log10, or its CDF (``_numpyro_priors``), and NumPyro walks it
    through ``biject_to(support)``; its inverse is where a chain at that value starts.
    """
    from numpyro.distributions.transforms import biject_to

    out = {}
    for j, (site, d, to_linear) in enumerate(site_specs):
        v = _to_site(prior.distributions[names[j]], starts[:, j])
        u = biject_to(d.support).inv(jnp.asarray(v, dtype=dtype))
        out[site] = u if num_chains > 1 else u[0]
    return out


def _float_dtype():
    """Match JAX's configured precision instead of forcing float32.

    The kilonova likelihood is evaluated in float64 for production runs (``JAX_ENABLE_X64=1``);
    hard-coding float32 here downcast the sampled theta and the final argmax scan back to single
    precision, quietly discarding the very thing x64 was turned on for.
    """
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class NUTSGPUSampler(BaseSampler):
    """NUTS (NumPyro) on GPU. See module docstring for the shared-``log_prob_fn`` design."""

    name = "nuts_gpu"

    def fit(self, lc, model, prior=None, *, log_prob_fn=None, num_warmup=1000, num_samples=2000,
            num_chains=4, space="auto", likelihood="auto", seed=0, progress=False,
            target_accept_prob=0.8, init=None, init_strategy="prior_scan", dense_mass=False,
            max_tree_depth=10, step_size=1.0, chain_method="vectorized") -> SamplerResult:
        """Fit ``lc`` with ``model`` via NumPyro NUTS.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve. Used for ``n_data``/``attach_*`` metrics, and — when
            ``log_prob_fn`` is not supplied — to build the density.
        model : whisper_cbpf.models.Model or str
            A registered model, by object or by name.
        prior : Prior, optional
            Defaults to ``model.default_prior``. Each parameter may be ``Uniform``, ``LogUniform``,
            ``Normal``, ``TruncatedNormal`` (sampled exactly, through its inverse CDF) or
            ``Fixed`` (held at its value: not sampled, not counted in AIC/BIC, reported as a
            constant column, listed in ``info["fixed"]``); anything else raises rather than being
            approximated by its bounds.
        log_prob_fn : Callable[[jnp.ndarray], jnp.ndarray], optional
            A ``jax.jit``-compiled function mapping a flat ``theta`` vector (order =
            ``model.parameters``) to a scalar log-**likelihood**, ``-inf`` outside the prior box —
            not a log-posterior. The prior density is added separately by NumPyro's own per-
            parameter sample sites, so passing a posterior double-counts it: a density flagged
            ``includes_prior=True`` (``make_log_prob_jax(..., include_prior=True)``, what
            ``emcee_jax`` builds) raises when any prior is not Uniform, and warns otherwise.

            **Optional since the adapter layer landed.** When omitted, ``model.log_prob_jax`` is
            used if the factory filled it, and otherwise one is built from ``model.predict_jax``
            plus the likelihood implied by ``space`` (see
            :func:`whisper_cbpf.samplers.jax._adapters.make_log_prob_jax`). An explicitly passed
            callable always wins.
        num_warmup, num_samples, num_chains : int
            NUTS budget.
        init : str, dict, array, tuple or SamplerResult, optional
            Where the chains start: the name every MCMC-type sampler shares (``mcmc``,
            ``emcee_jax``, ``pymc_jax_gpu_*``), taking everything ``init_strategy`` takes plus a
            point -- a dict ``{name: value}`` or a ``(k,)`` array, started as a ball of
            ``DEFAULT_BALL_SCALE`` (1e-3) of each prior's width, or ``(point, scale)`` -- and a
            previous result: an ABC fit, a continuation, or an alert's earlier cut. From a result
            the chains start on ``num_chains`` distinct draws of it that are usable here (inside
            the box and every constraint wall, at a finite density); with fewer usable draws, in
            a ball around its optimised likelihood maximum or ``best_params`` sized by its own
            spread. Every start
            is checked and refused where the density is -inf or NaN. ``info["init"]`` records the
            kind that ran and ``info["init_detail"]`` how. Give ``init`` or ``init_strategy``,
            not both.
        init_strategy : str, callable or array, default "prior_scan"
            Where the chains start. ``"prior_scan"`` (the default; ``None`` means the same) scores
            1000 prior draws with the density in one device program, climbs the best 32 by a short
            gradient ascent to learn which basin each belongs to, and starts each chain a random
            ~2 posterior sd from a distinct one of those that reach the best basin -- overdispersed
            for R-hat, but not in a local optimum. It adds seconds to minutes, compile included:
            on the CPU 1-2 s for a Gaussian bump, 7-13 s for a kilonova or an arnett
            supernova; on one A6000 7-12 s for those. On the TDE it is mostly compiling the ODE's
            gradient for the climb: 1.6 min at ``n_time=500`` on one heavily loaded A6000, where
            the old ``jax.hessian`` curvature took it to 3.6-6 min (its NUTS run takes tens of
            minutes). The scan runs
            whatever the start, because its best point is also the independent optimum every run
            is checked against. It replaced NumPyro's ``init_to_uniform``, which drew u ~
            U(-2, 2) in unconstrained space and started chains wherever that fell: on a Gaussian
            bump 22 / 100 float64 runs had a chain stranded in a local optimum (the model hidden
            between epochs), and on a one-component kilonova 7 / 13 had chains on a prior-box
            corner. ``"prior"`` starts each chain at an independent prior draw: the scan's first
            ``num_chains`` draws at which the density is finite, not ranked by it. A
            ``(num_chains, k)`` array gives one start per chain, in parameter units and in
            ``model.parameters`` order; a start where the density is -inf or NaN is refused.
            NumPyro's strategies stay available as ``"uniform"`` (the
            old default), ``"median"``, ``"sample"``, ``"feasible"``, ``"mean"`` (the
            ``init_to_`` prefix is optional) or any NumPyro init callable, e.g.
            ``numpyro.infer.init_to_value(values={"mej": 0.03})``. The point strategies (median,
            mean, feasible, value) start every chain at one point and warn when
            ``num_chains > 1``: R-hat then cannot tell a stuck chain from a converged one.
            ``info["init_strategy"]`` records what ran.
        dense_mass : bool, default False
            ``False`` adapts a diagonal mass matrix (NumPyro's default), ``True`` a full one.
            Dense costs O(k^2) memory and a slower warmup, and pays for itself only when the
            posterior has strong linear correlations — which the kilonova's mej/vej/kappa do.
        max_tree_depth : int, default 10
            Doubling cap per NUTS trajectory: at most ``2**max_tree_depth`` leapfrog steps. Raise
            it when the sampler saturates (a badly conditioned posterior needs longer
            trajectories); every increment doubles the worst-case cost per draw.
        step_size : float, default 1.0
            *Initial* leapfrog step size. Adaptation is on, so this is a starting guess, not the
            step actually used.
        chain_method : {"vectorized", "parallel", "sequential"}, default "vectorized"
            How chains are placed. ``"vectorized"`` batches all chains onto one device via vmap —
            the standard single-GPU multi-chain NumPyro pattern, and the only one that uses a
            single card fully. ``"parallel"`` needs one device per chain and NumPyro **silently
            downgrades it to "sequential"** otherwise (see
            ``whisper_cbpf.samplers.jax.pymc_gpu._effective_chain_method``), so what you asked for
            and what runs can differ; ``info["chain_method"]`` records the request.
        space : {"auto", "flux", "magnitude"}
            **Live on the auto-built path**, where it selects the likelihood: ``"auto"`` follows
            ``lc.data_mode``, the same rule ``abc``/``abc_smc``/``mcmc`` use, so magnitude data is
            fitted in magnitude space. Defaulting to ``"flux"`` and recording it but never
            used, which meant a magnitude light curve was silently scored in flux by every
            predictive metric.

            When *you* pass ``log_prob_fn``, the space is fixed inside your density and this
            argument cannot change it — it is then only a label for the metrics, and
            ``info["space_source"]`` records which case applies.

        Returns
        -------
        SamplerResult
            ``runtime_s`` is the start, warmup and sampling (``info["init_time_s"]``,
            ``["warmup_time_s"]``, ``["sampling_time_s"]``); ``info["postprocess_s"]`` is what
            follows it -- the log-likelihood of every kept draw and the chain checks -- so a
            benchmark can add it back. ``info`` also carries divergences,
            rank-normalised R-hat and bulk/tail ESS per parameter (``rhat``, ``max_rhat``,
            ``ess_bulk``, ``ess_tail``, ``rhat_method``, ``rhat_error``), the log-likelihood R-hat,
            each chain's median log-likelihood and adapted step size, the ``stranded_chains`` and
            ``frozen_chains`` found, and ``convergence_problems``: one sentence per failed check.
            ``converged`` is True only when that list is empty -- no divergence, R-hat < 1.01 on
            every parameter and on the log-likelihood, ESS >= 100 per chain, no stranded or frozen
            chain, and no point found by the prior scan more than 10 nats above every chain's best
            draw (the all-chains-wrong case R-hat cannot see). A non-empty list is also a warning.
            The raw NumPyro ``MCMC`` object is attached as ``result.numpyro_mcmc``.

        Warns before sampling when float32 cannot resolve ``lc.time`` or a Uniform prior bound
        (absolute value >= 1e3: MJD clocks), the cause of frozen chains, and when >= 90 % of the
        prior scan's draws sit on one flat zero-signal level (a time prior much wider than the data
        window: a chain there has no gradient to follow). ``info["prior_scan"]`` records the scan.

        Examples
        --------
        A Gaussian prior on the peak epoch (from a last non-detection, say), a Fixed width, and
        the chains started on the draws of an earlier fit:

        >>> import numpy as np, jax.numpy as jnp, whisper_cbpf as wp
        >>> from whisper_cbpf.priors import Fixed, LogUniform, Normal, Prior
        >>> def bump(p, t, b=None):
        ...     return p["A"] * np.exp(-0.5 * ((np.asarray(t) - p["t0"]) / p["w"]) ** 2)
        >>> def bump_jax(th, t, bi=None):
        ...     return th[0] * jnp.exp(-0.5 * ((jnp.asarray(t) - th[1]) / th[2]) ** 2)
        >>> prior = Prior({"A": LogUniform(0.3, 10.0), "t0": Normal(15.0, 2.0), "w": Fixed(3.0)})
        >>> m = wp.register_model("doc_bump", bump, ["A", "t0", "w"], prior=prior,
        ...                       predict_jax=bump_jax, overwrite=True)
        >>> t = np.linspace(0.0, 30.0, 25)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 25, flux=bump({"A": 5, "t0": 14, "w": 3}, t),
        ...                    flux_err=np.full(25, 0.1))
        >>> first = wp.fit(lc, "doc_bump", sampler="mcmc", space="flux", nsteps=1500, burnin=500)
        >>> r = wp.fit(lc, "doc_bump", sampler="nuts_gpu", space="flux", init=first,
        ...            num_warmup=300, num_samples=300)
        >>> r.info["init"], r.info["fixed"], r.n_params
        ('result', {'w': 3.0}, 2)
        """
        # Only a string needs looking up. Any Model-like object passes straight through:
        # get_model() on a duck-typed object raises KeyError, or TypeError if it is unhashable.
        model = get_model(model) if isinstance(model, str) else model
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        check_not_empty(lc)             # a caller's log_prob_fn never meets the likelihood's check
        names, likelihood = _resolve_sampling_names(lc, model, prior, log_prob_fn, space,
                                                   likelihood)
        # A Fixed parameter is held at its value: from here on `prior` and `names` are the FREE
        # ones, and `all_names` / `full_prior` the model's, for the density and the result.
        all_names, full_prior = names, prior
        prior, names, fixed = _dg.split_fixed(full_prior, all_names, self.name)
        k, n = len(names), int(len(lc.time))
        # Validate the prior and the kernel options BEFORE building anything: an unsupported
        # distribution or a mistyped option should report itself, not hide behind a filter-set
        # build or a minute of XLA compilation.
        site_specs = _numpyro_priors(prior, names)
        init_spec = _dg.merge_init(init, init_strategy, "prior_scan", self.name)
        init_fn, init_label = _resolve_init_strategy(init_spec)
        if chain_method not in _CHAIN_METHODS:
            raise ValueError(f"nuts_gpu: chain_method must be one of {_CHAIN_METHODS}, got "
                             f"{chain_method!r}.")
        if init_label == "per_chain":           # the shape and the box, before anything compiles
            _dg.check_starts(init_spec, prior, names, num_chains, self.name)
        log_prob_fn, space, space_source, fn_source = _resolve_density(
            lc, model, full_prior, log_prob_fn, space, all_names, include_prior=False,
            sampler=self.name, likelihood=likelihood)
        log_prob_fn, _, _ = _free_density(log_prob_fn, full_prior, all_names)
        # Three hazards that are visible before a single step: a log-posterior here counts the
        # prior twice, float32 cannot resolve an MJD-scale clock, and point starts blind R-hat.
        _dg.check_likelihood_contract(log_prob_fn, prior, names, self.name)
        f32_hazard = _dg.float32_hazard(lc.time, prior, names, self.name,
                                        bool(jax.config.jax_enable_x64))
        _dg.warn_point_start(init_label, num_chains, self.name)
        dt = _float_dtype()

        # THE PRIOR SCAN, run whatever the start: it is where "prior_scan" starts its chains, and
        # its best point is the independent optimum every run is checked against afterwards (all
        # chains in one wrong mode agree with each other, so R-hat alone reads them as converged).
        t_scan = time.perf_counter()
        sc = _dg.scan_starts(log_prob_fn, prior, names, int(num_chains), seed, dt)
        reference_ll = sc["reference"]
        starts, start_ll, init_detail = None, None, {}
        if init_label == "prior_scan":
            starts, start_ll = sc["starts"], sc["start_scores"]
        elif init_label == "prior":
            starts, start_ll = _dg.prior_starts(sc, int(num_chains), self.name)
        elif init_fn is None:                   # per_chain, ball or result: checked where it lands
            starts, start_ll, init_label, init_detail = _dg.explicit_starts(
                init_spec, prior, names, int(num_chains), seed, sc["score"], self.name,
                what="chain")
        init_params = (None if starts is None
                       else _unconstrained_starts(site_specs, starts, int(num_chains), dt, prior,
                                                  names))
        init_time = time.perf_counter() - t_scan

        def numpyro_model():
            # One site per parameter so each keeps its OWN density and support transform. A
            # reparameterised site (log-uniform -> uniform on log10) is pushed back to the
            # parameter's own units by a `deterministic`, which NumPyro collects into
            # `get_samples()` alongside the sampled sites -- so `samples["temperature_floor"]` is
            # kelvin, exactly as before, while the chain walks log10 kelvin.
            values = []
            for nm, (site, d, to_linear) in zip(names, site_specs):
                v = numpyro.sample(site, d)
                values.append(v if to_linear is None else numpyro.deterministic(nm, to_linear(v)))
            # Stacked back into the flat theta vector log_prob_fn expects, in model.parameters order
            numpyro.factor("log_likelihood", log_prob_fn(jnp.stack(values)))

        kernel_kw = {} if init_fn is None else {"init_strategy": init_fn}
        kernel = NUTS(numpyro_model, target_accept_prob=float(target_accept_prob),
                      dense_mass=bool(dense_mass), max_tree_depth=int(max_tree_depth),
                      step_size=float(step_size), **kernel_kw)
        mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples),
                    num_chains=int(num_chains), progress_bar=progress, chain_method=chain_method)

        rng_key = jax.random.PRNGKey(int(seed))

        # NOTE ON TIMING: XLA compiles the NUTS kernel on the first warmup() call, so
        # `warmup_time_s` below is compile + warmup on a cold process. Compilation is a
        # one-time cost that does NOT scale with the number of curves fitted, and reporting
        # it as sampler cost badly misrepresents throughput (measured: ~8.4 s compile vs
        # ~4.8 s of actual warmup). The benchmark driver therefore fits each configuration
        # twice in one process -- XLA caches the executable, so run 2 is compile-free and
        # (run1 - run2) isolates compile time. See run_benchmark.py.
        t0 = time.perf_counter()
        mcmc.warmup(rng_key, collect_warmup=False, extra_fields=("diverging",),
                    init_params=init_params)
        jax.block_until_ready(mcmc.last_state)
        warmup_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        # extra_fields is REQUIRED for divergences to be collected at all. Without it
        # get_extra_fields() comes back empty and n_divergences falls through to its -1 sentinel --
        # which reads like "unknown" but in practice got treated as "none", silently disabling the
        # single most important NUTS health check. A divergent chain is biased, not merely noisy.
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("diverging",))
        # Per-parameter sites come back as a dict of (chains, draws); restack in parameter order.
        by_site = mcmc.get_samples(group_by_chain=True)
        samples_by_chain = jnp.stack([by_site[nm] for nm in names], axis=-1)  # (chains, draws, k)
        extra = mcmc.get_extra_fields(group_by_chain=True)
        jax.block_until_ready(samples_by_chain)
        sampling_time = time.perf_counter() - t1
        runtime = init_time + warmup_time + sampling_time
        t_post = time.perf_counter()

        n_chains_run, n_draws = samples_by_chain.shape[0], samples_by_chain.shape[1]
        theta_flat = np.asarray(samples_by_chain).reshape(-1, k)
        # Every model parameter, a Fixed one as a constant column (the metrics call the model).
        samples = pd.DataFrame(_dg.fill_fixed(theta_flat, names, fixed, all_names),
                               columns=all_names)

        n_divergences = int(np.asarray(extra["diverging"]).sum()) if "diverging" in extra else -1
        # Adapted step size per chain, for the frozen-chain check (one entry per chain for every
        # chain_method; None only if NumPyro's state layout ever changes -- then that check is off).
        try:
            steps = np.atleast_1d(np.asarray(mcmc.last_state.adapt_state.step_size, dtype=float))
        except AttributeError:
            steps = None

        # log_prob_fn is pure log-likelihood (the prior's density is contributed separately by
        # NumPyro's own sample sites), so the best draw is the argmax of log_prob_fn itself.
        #
        # SCANNED IN FIXED-WIDTH BLOCKS (`_diagnostics.block_scorer`, compiled once already for the
        # prior scan). This was first `jax.vmap(log_prob_fn)(theta_flat)` -- every post-warmup
        # sample batched into a single call, a vmap extent equal to the draw count. XLA's compile
        # time for a fused reduction grows steeply with that extent: on the two-component kilonova
        # a mere B=256 did not finish compiling in six minutes, so at 16,000 draws the sampler sat
        # at 0% GPU utilisation for over an hour AFTER sampling had finished. Then it was
        # `lax.map(log_prob_fn, theta_flat)`, one draw per step: constant compile, but a
        # latency-bound model paid its whole fixed cost per draw -- the TDE at n_time=5000 spent
        # 50.9 ms a draw, ~672 s for 13,200 draws, 4x its sampling time, outside `runtime_s`.
        # The time this takes is `info["postprocess_s"]`.
        ll_flat = sc["score"](theta_flat)
        best_idx = int(np.nanargmax(ll_flat))
        best_params = {nm: float(theta_flat[best_idx, j]) for j, nm in enumerate(names)}
        best_params = {nm: best_params.get(nm, fixed.get(nm)) for nm in all_names}
        max_log_likelihood = float(ll_flat[best_idx])

        # R-hat, ESS and the per-chain checks, all from arrays already in hand: the same
        # log-likelihood scan, reshaped by chain, is what exposes a stranded chain. Fails closed:
        # unknown is not converged, and max_rhat is a float (NaN when unknown), never None --
        # notebook 07 did round(info["max_rhat"], 4) and crashed on None.
        health = _dg.chain_health(
            np.asarray(samples_by_chain), names, ll_flat.reshape(n_chains_run, n_draws),
            n_divergences=n_divergences, step_size_by_chain=steps, reference_ll=reference_ll)
        _dg.warn_if_unconverged(self.name, health)
        postprocess_time = time.perf_counter() - t_post

        info = {
            "num_warmup": int(num_warmup), "num_samples": int(num_samples),
            "num_chains": int(n_chains_run), "space": space, "seed": int(seed),
            "space_source": space_source,
            # `_LIKELIHOOD_KINDS` in samplers/base.py maps this back to a registry name, so WAIC
            # and predictive_metrics re-score under the density that was actually fitted. None
            # when the caller supplied their own log_prob_fn: we cannot know then.
            "likelihood": getattr(log_prob_fn, "likelihood", None),
            "scatter_param": getattr(log_prob_fn, "scatter_param", None),
            "log_prob_fn": fn_source,
            "x64": bool(jax.config.jax_enable_x64),
            "target_accept_prob": float(target_accept_prob),
            # The kernel hyper-parameters that actually ran. Without these a saved run cannot be
            # reproduced or compared -- two fits differing only in dense_mass or max_tree_depth
            # were previously indistinguishable in the record.
            "init_strategy": init_label, "init": init_label, "init_detail": init_detail,
            "dense_mass": bool(dense_mass),
            "max_tree_depth": int(max_tree_depth), "step_size": float(step_size),
            "chain_method": chain_method,
            # The site each parameter is actually sampled on. Differs from the parameter name
            # exactly where a reparameterisation is in play, which is the thing worth recording.
            "sample_sites": {nm: site for nm, (site, _d, _f) in zip(names, site_specs)},
            # Parameters held at a Fixed prior's value: not sampled, not counted in AIC/BIC.
            "fixed": dict(fixed),
            # The scan behind the start and behind the independent-optimum check.
            "prior_scan": {"n_draws": int(len(sc["draws"])), "n_climbed": sc["n_climbed"],
                           "n_reaching_best": sc["n_good"], "climb_error": sc["climb_error"],
                           "plateau_fraction": sc["plateau_fraction"],
                           "best_log_likelihood": reference_ll,
                           "start_log_likelihood": (None if start_ll is None
                                                    else [float(v) for v in start_ll])},
            "float32_hazard": f32_hazard,
            "init_time_s": float(init_time),
            "warmup_time_s": float(warmup_time), "sampling_time_s": float(sampling_time),
            # After sampling, outside runtime_s: the log-likelihood of every kept draw, and R-hat,
            # ESS and the chain checks. The metrics attached below are not included.
            "postprocess_s": float(postprocess_time),
            "n_divergences": n_divergences,
            **health,
        }
        attach_band_metrics(info, lc, model, best_params, space)
        aic, bic = aic_bic(max_log_likelihood, k, n)
        result = SamplerResult(
            sampler="nuts_gpu", model=model.name, parameters=all_names, samples=samples,
            summary=summarize_posterior(samples, all_names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=aic, bic=bic,
        )
        result.numpyro_mcmc = mcmc
        # (chains, draws, len(parameters)) -- for ESS/sec; a Fixed parameter is a constant column
        result.samples_by_chain = _dg.fill_fixed(np.asarray(samples_by_chain), names, fixed,
                                                 all_names)
        attach_predictive_metrics(result, lc, space, model=model)
        return result


def fit_NUTSGPU(lc, model, prior=None, *, log_prob_fn=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via NumPyro NUTS on GPU. See :meth:`NUTSGPUSampler.fit`.

    ``log_prob_fn`` is optional here for the same reason it is on the method: omit it and one is
    built from ``model.predict_jax``. It stayed *required* on this wrapper after the method relaxed
    it, so the public function could not reach the auto-build path the method advertises --
    ``NUTSGPUSampler().fit(...)`` and ``wp.fit(...)`` worked while ``fit_NUTSGPU(...)`` raised
    ``TypeError: missing 1 required keyword-only argument``.
    """
    return NUTSGPUSampler().fit(lc, model, prior=prior, log_prob_fn=log_prob_fn, **kwargs)
