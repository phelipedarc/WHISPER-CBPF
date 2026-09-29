"""NUTS through PyMC's model API, executed by NumPyro on the GPU.

PyMC's JAX path does not implement its own sampler: ``pymc.sampling.jax.sample_numpyro_nuts``
lowers the PyTensor graph to JAX and hands it to **NumPyro's NUTS** -- the same kernel ``nuts_gpu``
calls directly. These are therefore not a competing algorithm. What they vary is the frontend
(PyMC's model API instead of a hand-written log-density) and the chain execution strategy:

``pymc_jax_gpu_vectorized``
    ``chain_method="vectorized"`` -- chains vmapped on one device. Same kernel and same strategy as
    ``nuts_gpu``, so it must reproduce that posterior to sampling noise, which makes it a free
    regression test on the whole PyTensor-to-JAX bridge.

``pymc_jax_gpu_parallelized``
    ``chain_method="parallel"`` -- chains ``pmap``-ed across devices. This is the multi-GPU path,
    and it needs more than one visible device to mean anything; with one it degenerates to
    sequential chains, so ``fit`` warns rather than reporting a fake result.

**Why a custom PyTensor Op is unavoidable.** The forward models are JAX and PyMC builds PyTensor
graphs, so bridging them needs an ``Op`` that PyTensor can place in a graph and that the JAX backend
knows how to lower -- which is what ``jax_funcify`` registration provides.

The subtle half is the gradient. PyTensor differentiates symbolically at the graph level and *then*
lowers, so it cannot see inside a JAX callable: an Op with no ``grad`` leaves NUTS with nothing to
differentiate, and the failure surfaces as a PyTensor error far from its cause. So the value Op
declares its gradient to be a second Op wrapping ``jax.grad`` of the same function, and both are
registered with the JAX backend. The chain is PyMC graph -> PyTensor grad -> JAX lowering -> one
fused XLA program per chain.

The log-density itself is passed in, not rebuilt. ``nuts_gpu``, ``emcee_jax`` and these samplers are
meant to receive the *same object*, so a disagreement between them is the sampler and never the
likelihood.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ...samplers.base import (
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    check_not_empty,
    summarize_posterior,
    _warn_user,
)

from ...backends._env import require_jax
from ...models import get_model
from ...priors._numpy import family
from . import _diagnostics as _dg
from ._adapters import free_density as _free_density
from ._adapters import resolve_density as _resolve_density
from ._adapters import resolve_sampling_names as _resolve_sampling_names


def _build_ops(log_prob_fn, jax, jnp):
    """A PyTensor Op for the JAX log-density, plus the gradient Op NUTS needs.

    Built per fit rather than at import: the Op closes over ``log_prob_fn``, and PyTensor is only
    imported here so ``import whisper_cbpf`` does not require PyMC to be installed.
    """
    import pytensor.tensor as pt
    from pytensor.graph import Apply, Op
    from pytensor.link.jax.dispatch import jax_funcify

    logp_jit = jax.jit(log_prob_fn)
    grad_jit = jax.jit(jax.grad(log_prob_fn))

    class JAXLogLikGrad(Op):
        """d(logp)/d(theta). Exists only so `JAXLogLik.grad` has something to return."""

        def make_node(self, theta):
            theta = pt.as_tensor_variable(theta)
            return Apply(self, [theta], [theta.type()])

        def perform(self, node, inputs, outputs):
            # NumPy fallback path, used if the graph is ever evaluated outside the JAX backend.
            outputs[0][0] = np.asarray(grad_jit(jnp.asarray(inputs[0])), dtype=inputs[0].dtype)

    class JAXLogLik(Op):
        """theta (D,) -> scalar log-density, evaluated by JAX."""

        def make_node(self, theta):
            theta = pt.as_tensor_variable(theta)
            return Apply(self, [theta], [pt.scalar(dtype=theta.dtype)])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(logp_jit(jnp.asarray(inputs[0])), dtype=inputs[0].dtype)

        def grad(self, inputs, output_grads):
            # Chain rule at the GRAPH level: PyTensor never looks inside the JAX callable, so the
            # derivative has to be handed to it as another Op.
            return [output_grads[0] * grad_op(inputs[0])]

    logp_op, grad_op = JAXLogLik(), JAXLogLikGrad()

    # Teach the JAX backend to lower each Op straight back to the original JAX function, so the
    # whole model becomes one XLA program instead of a graph with host callbacks in it.
    @jax_funcify.register(JAXLogLik)
    def _funcify_logp(op, **kwargs):
        def f(theta):
            return logp_jit(theta)
        return f

    @jax_funcify.register(JAXLogLikGrad)
    def _funcify_grad(op, **kwargs):
        def f(theta):
            return grad_jit(theta)
        return f

    return logp_op


def _pymc_prior(pm, pt, prior, names):
    """whisper's prior as PyMC variables, returning the stacked theta in ``names`` order.

    PyMC has no LogUniform, so a log-uniform parameter is sampled as a Uniform on log10 and
    exponentiated. That is the identical measure -- ``LogUniform(a, b)`` is exactly
    ``Uniform(log10 a, log10 b)`` pushed through ``10**x`` -- and it is also better geometry for
    NUTS on a parameter spanning decades. It matches what the AT2017GFO analysis already does for
    its JAX arms, so the two are directly comparable.
    """
    out = []
    for nm in names:
        d = prior.distributions[nm]
        kind = family(d)
        lo, hi = (float(x) for x in d.bounds)
        if kind == "Uniform":
            out.append(pm.Uniform(nm, lower=lo, upper=hi))
        elif kind == "LogUniform":
            log_v = pm.Uniform(f"log10_{nm}", lower=np.log10(lo), upper=np.log10(hi))
            out.append(pm.Deterministic(nm, 10.0 ** log_v))
        elif kind == "Normal":
            out.append(pm.Normal(nm, mu=float(d.mu), sigma=float(d.sigma)))
        elif kind == "TruncatedNormal":
            # The probability integral transform, as nuts_gpu._numpyro_priors does: a Uniform site
            # on the CDF and the exact inverse CDF, Phi^-1(q) = -sqrt(2) erfcinv(2 q), taken on
            # the upper tail's side when the range lies above mu (priors.TruncatedNormal.ppf).
            q = pm.Uniform(f"cdf_{nm}", lower=0.0, upper=1.0)
            base, mass = float(d._lo_cdf), float(d._mass)
            if d._flip:
                z = np.sqrt(2.0) * pt.erfcinv(2.0 * (base + (1.0 - q) * mass))
            else:
                z = -np.sqrt(2.0) * pt.erfcinv(2.0 * (base + q * mass))
            out.append(pm.Deterministic(nm, pt.clip(float(d.mu) + float(d.sigma) * z, lo, hi)))
        else:
            raise TypeError(
                f"pymc_gpu cannot express prior {kind!r} on parameter {nm!r}. Supported: Uniform, "
                f"LogUniform, Normal, TruncatedNormal (and Fixed, held at its value). Extend "
                f"_pymc_prior rather than substituting a uniform box -- that changes the posterior "
                f"silently.")
    return pt.stack(out)


def _pymc_site(prior, nm):
    """``(variable name, value -> site value)`` for parameter ``nm``: the variable ``_pymc_prior``
    creates, which is where a chain's start has to be given, and the map onto it."""
    from .nuts_gpu import _to_site

    d = prior.distributions[nm]
    kind = family(d)
    if kind in ("Uniform", "Normal"):
        return nm, (lambda v, d=d: _to_site(d, v))
    if kind == "LogUniform":
        return f"log10_{nm}", (lambda v, d=d: _to_site(d, v))
    if kind == "TruncatedNormal":
        return f"cdf_{nm}", (lambda v, d=d: _to_site(d, v))
    raise TypeError(
        f"pymc_gpu cannot express prior {kind!r} on parameter {nm!r}. Supported: Uniform, "
        f"LogUniform, Normal, TruncatedNormal (and Fixed, held at its value). Extend _pymc_prior "
        f"rather than substituting a uniform box -- that changes the posterior silently.")


#: ``init_strategy=`` names: whisper's two starts (see ``nuts_gpu._WHISPER_STARTS``) and PyMC's own.
_PYMC_STARTS = ("prior_scan", "prior", "jitter")


def _resolve_pymc_init(init_strategy):
    """``None`` -> ``"prior_scan"``; a name -> itself; a ``(num_chains, k)`` array -> ``"per_chain"``;
    a point (dict, ``(k,)`` array or ``(point, scale)``) -> ``"ball"``; a result -> ``"result"``."""
    if init_strategy is None:
        return "prior_scan"
    if _dg._is_result(init_strategy):
        return "result"
    if isinstance(init_strategy, dict) or (
            isinstance(init_strategy, tuple) and len(init_strategy) == 2
            and np.ndim(init_strategy[1]) == 0
            and (isinstance(init_strategy[0], dict) or np.ndim(init_strategy[0]) >= 1)):
        return "ball"
    if isinstance(init_strategy, str):
        key = init_strategy.strip().lower()
        if key not in _PYMC_STARTS:
            raise ValueError(f"pymc_gpu: unknown init_strategy {init_strategy!r}. Use one of "
                             f"{list(_PYMC_STARTS)}, a point, (point, scale), a (num_chains, k) "
                             f"array of starts, or a previous result.")
        return key
    try:
        ndim = np.asarray(init_strategy, dtype=float).ndim
    except (TypeError, ValueError):
        ndim = None
    if ndim == 1:
        return "ball"
    if ndim != 2:
        raise TypeError(f"pymc_gpu: init / init_strategy must be one of {list(_PYMC_STARTS)}, a "
                        f"point, (point, scale), a (num_chains, k) array of starts, or a previous "
                        f"result; got {type(init_strategy).__name__}.")
    return "per_chain"


def _effective_chain_method(requested, num_chains, n_local_devices):
    """What NumPyro will ACTUALLY do, which is not always what was asked for.

    ``numpyro.infer.MCMC.__init__`` contains, verbatim::

        if chain_method == "parallel" and local_device_count() < self.num_chains:
            chain_method = "sequential"

    so ``parallel`` parallelises only when every chain gets its own device, and the downgrade is a
    ``warnings.warn`` that is trivially lost in a benchmark log. It does **not** split into
    ``ceil(chains / devices)`` pmap passes. This module previously encoded that wrong belief by
    warning on ``num_chains % n_devices``, which is silent for the commonest failure of all: four
    chains on two GPUs has remainder zero, so nothing was reported, and the run was nonetheless
    fully sequential on a single device. Measuring "multi-GPU speedup" under those conditions
    compares a one-GPU run against a one-GPU run.
    """
    if requested == "parallel" and int(n_local_devices) < int(num_chains):
        return "sequential"
    return requested


class _PyMCJAXBase(BaseSampler):
    """Shared implementation; subclasses only pick ``chain_method``."""

    chain_method = "vectorized"

    def fit(self, lc, model, prior=None, *, log_prob_fn=None, num_warmup=1000, num_samples=2000,
            num_chains=4, space="auto", likelihood="auto", seed=0, progress=False,
            target_accept_prob=0.8, init=None, init_strategy="prior_scan") -> SamplerResult:
        """Fit ``lc`` with ``model`` via PyMC's model API, sampled by NumPyro NUTS on GPU.

        Parameters
        ----------
        model : whisper_cbpf.models.Model or str
            A registered model, by object or by name.
        log_prob_fn : callable, optional
            ``theta -> scalar`` pure log-LIKELIHOOD in ``model.parameters`` order. The prior's
            density is contributed by the PyMC variables, so passing a log-POSTERIOR here would
            double-count it. Same contract as ``nuts_gpu`` — and, like it, **optional**: when
            omitted one is built from ``model.predict_jax`` and the likelihood implied by
            ``space``. An explicitly passed callable always wins.
        space : {"auto", "flux", "magnitude"}
            Live on the auto-built path; a label only when you pass your own ``log_prob_fn``.
            ``info["space_source"]`` records which. See ``nuts_gpu.fit``.
        likelihood : str, default "auto"
            Registered likelihood ``kind`` for the auto-built density — ``"auto"`` (upper limits
            when the light curve carries them, else Gaussian), ``"gaussian"``, ``"upper_limits"``
            (flux space only) or ``"gaussian_scatter"``. Ignored when ``log_prob_fn`` is passed.
            ``"gaussian_scatter"`` needs its ``scatter_param`` (default ``"sigma"``) in ``prior``:
            it is then sampled as an ordinary site and routed to the likelihood rather than to
            ``model.predict``. The resolved class lands in ``info["likelihood"]``.
        num_warmup, num_samples, num_chains, seed, target_accept_prob
            NUTS budget and adaptation, as for ``nuts_gpu``.
        init_strategy : {"prior_scan", "prior", "jitter"} or array, default "prior_scan"
            Where the chains start, as for ``nuts_gpu``: ``"prior_scan"`` starts each chain near a
            distinct prior draw that was scored and climbed into the best basin found, ``"prior"``
            at an independent prior draw where the density is finite, a ``(num_chains, k)`` array
            at the given points (parameter units, ``model.parameters`` order; refused where the
            density is -inf or NaN). ``"jitter"`` is PyMC's own start and this
            sampler's old default: the prior's midpoint, jittered U(-1, 1) in the transformed space.
            Under it 10 of 100 float64 Gaussian-bump runs stranded a chain and 2 put every chain in
            the wrong mode with R-hat 1.002.
        init : dict, array, tuple, str or SamplerResult, optional
            The shared name for the start, as for ``nuts_gpu``: everything ``init_strategy`` takes,
            plus a point (a dict or ``(k,)`` array; ``(point, scale)`` sets the ball) and a previous
            result (an ABC fit, a continuation, an alert's earlier cut). ``info["init"]`` records
            the kind that ran. Give ``init`` or ``init_strategy``, not both.

        Returns the same ``info`` diagnostics as ``nuts_gpu`` (``convergence_problems``,
        ``stranded_chains``, ``frozen_chains``, rank-normalised ``rhat``, ESS, ...), with the same
        meaning of ``converged`` and of ``postprocess_s`` (the post-fit re-scan, outside
        ``runtime_s``), and gives the same warnings before sampling (a log-posterior passed as
        ``log_prob_fn``, a prior that is mostly a zero-signal plateau).

        **Precision.** PyTensor's JAX linker needs ``jax_enable_x64`` to match its ``floatX``
        (float64 by default), and switched it on for the whole process at import, silently turning
        every later ``nuts_gpu`` or ``emcee_jax`` call in the session into a float64 run. The fit
        now runs at the linker's precision and puts the session's flag back afterwards;
        ``info["x64"]`` records what this fit ran in, ``info["x64_session"]`` the flag it restored.

        **Priors.** ``Uniform``, ``LogUniform`` (a Uniform on log10), ``Normal`` (``pm.Normal``),
        ``TruncatedNormal`` (a Uniform on its CDF through the exact inverse CDF, as ``nuts_gpu``
        does) and ``Fixed`` (held at its value; ``info["fixed"]``).

        Examples
        --------
        >>> import numpy as np, jax.numpy as jnp, whisper_cbpf as wp
        >>> from whisper_cbpf.priors import Normal, Prior, TruncatedNormal
        >>> prior = Prior({"a": Normal(1.0, 0.5), "b": TruncatedNormal(0.0, 0.1, 0.0, 1.0)})
        >>> m = wp.register_model("doc_line", lambda p, t, b=None: p["a"] + p["b"] * np.asarray(t),
        ...                       ["a", "b"], prior=prior, overwrite=True,
        ...                       predict_jax=lambda th, t, bi=None: th[0] + th[1] * jnp.asarray(t))
        >>> t = np.linspace(0.0, 10.0, 20)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 20, flux=1.2 + 0.05 * t,
        ...                    flux_err=np.full(20, 0.2))
        >>> r = wp.fit(lc, "doc_line", sampler="pymc_jax_gpu_vectorized", space="flux",
        ...            init={"a": 1.2, "b": 0.05}, num_warmup=300, num_samples=300)
        >>> r.info["init"], bool((r.samples["b"] >= 0.0).all())
        ('ball', True)
        """
        jax, jnp = require_jax(self.name)
        x64_before = bool(jax.config.jax_enable_x64)
        try:
            return self._fit(jax, jnp, x64_before, lc, model, prior, log_prob_fn=log_prob_fn,
                             num_warmup=num_warmup, num_samples=num_samples,
                             num_chains=num_chains, space=space, likelihood=likelihood,
                             seed=seed, progress=progress, target_accept_prob=target_accept_prob,
                             init_strategy=_dg.merge_init(init, init_strategy, "prior_scan",
                                                          self.name))
        finally:
            # `import pytensor.link.jax` runs jax.config.update("jax_enable_x64", True) at module
            # scope and nothing ever turns it back. Measured at HEAD: x64 False before the first
            # pymc fit, True after it, for the rest of the process.
            jax.config.update("jax_enable_x64", x64_before)

    def _fit(self, jax, jnp, x64_before, lc, model, prior, *, log_prob_fn, num_warmup,
             num_samples, num_chains, space, likelihood, seed, progress, target_accept_prob,
             init_strategy):
        """The body of :meth:`fit`, which only wraps it to restore the session's x64 flag."""
        try:
            import pymc as pm
            import pytensor.tensor as pt
            from pymc.sampling.jax import sample_numpyro_nuts
            from pytensor import config as pytensor_config
        except ImportError as exc:                      # pragma: no cover - environment dependent
            raise ImportError(
                f"{self.name} needs PyMC on top of the JAX stack: "
                f"`pip install 'whisper-cbpf[gpu,pymc]'` (or `pip install pymc`). It is an optional "
                f"dependency -- no other sampler requires it, which is why the import is here "
                f"rather than at module scope.") from exc
        # The precision the PyTensor JAX linker lowers to -- what its import set, stated here so a
        # second fit in the same process (import cached, flag restored by `fit`) runs the same.
        run_x64 = pytensor_config.floatX == "float64"
        jax.config.update("jax_enable_x64", run_x64)

        model = get_model(model) if isinstance(model, str) else model
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        check_not_empty(lc)             # a caller's log_prob_fn never meets the likelihood's check
        names, likelihood = _resolve_sampling_names(lc, model, prior, log_prob_fn, space, likelihood)
        # Fixed parameters are held at their values: `prior` / `names` are the FREE ones from here.
        all_names, full_prior = names, prior
        prior, names, fixed = _dg.split_fixed(full_prior, all_names, self.name)
        k, n = len(names), int(len(lc.time))
        sites = [_pymc_site(prior, nm) for nm in names]      # refuses an unsupported prior up front
        init_label = _resolve_pymc_init(init_strategy)
        if init_label == "per_chain":           # the shape and the box, before anything compiles
            _dg.check_starts(init_strategy, prior, names, num_chains, self.name)
        log_prob_fn, space, space_source, fn_source = _resolve_density(
            lc, model, full_prior, log_prob_fn, space, all_names,
            include_prior=False, sampler=self.name, likelihood=likelihood)
        log_prob_fn, _, _ = _free_density(log_prob_fn, full_prior, all_names)
        _dg.check_likelihood_contract(log_prob_fn, prior, names, self.name)
        f32_hazard = _dg.float32_hazard(lc.time, prior, names, self.name, run_x64)
        dt = jnp.float64 if run_x64 else jnp.float32

        # The same prior scan as nuts_gpu: the start, and the independent optimum checked below.
        t_scan = time.perf_counter()
        sc = _dg.scan_starts(log_prob_fn, prior, names, int(num_chains), seed, dt)
        reference_ll = sc["reference"]
        starts, start_ll, init_detail = None, None, {}
        if init_label == "prior_scan":
            starts, start_ll = sc["starts"], sc["start_scores"]
        elif init_label == "prior":
            starts, start_ll = _dg.prior_starts(sc, int(num_chains), self.name)
        elif init_label != "jitter":            # per_chain, ball or result: checked where it lands
            starts, start_ll, init_label, init_detail = _dg.explicit_starts(
                init_strategy, prior, names, int(num_chains), seed, sc["score"], self.name,
                what="chain")
        # PyMC takes one {variable: value} per chain in the variables' own space -- the log10 site
        # for a LogUniform, the CDF for a TruncatedNormal -- and applies no jitter when told not to.
        initvals = (None if starts is None else
                    [{site: float(to_site(v)) for (site, to_site), v in zip(sites, row)}
                     for row in starts])
        init_time = time.perf_counter() - t_scan

        # local_device_count(), not device_count(): NumPyro checks the LOCAL count, and the two
        # differ on a multi-host setup.
        n_devices = jax.local_device_count()
        effective_chain_method = _effective_chain_method(self.chain_method, num_chains, n_devices)

        if effective_chain_method != self.chain_method:
            _warn_user(
                f"{self.name}: NumPyro will silently run chain_method='sequential', NOT "
                f"'parallel'. It requires one device per chain, and this call asks for "
                f"{int(num_chains)} chains with {n_devices} visible device(s). Wall time will "
                f"scale with num_chains, not num_chains/n_devices, and {max(n_devices - 1, 0)} "
                f"device(s) will idle while holding an allocation (measured on AT2017GFO: GPU 0 "
                f"at 97-98% util, GPU 3 at 0% with 262 MiB held by the same PID). Remedies: pass "
                f"num_chains={n_devices} for a true pmap, at the cost of fewer chains for r-hat; "
                f"or use the 'pymc_jax_gpu_vectorized' sampler, which vmaps all "
                f"{int(num_chains)} chains onto ONE device; or run "
                f"{max(int(num_chains) // max(n_devices, 1), 1)} processes of {n_devices} chains "
                f"pinned one per GPU, the only arrangement giving {int(num_chains)} chains AND "
                f"every device. result.info['chain_method'] records what actually ran.",
               )
        elif self.chain_method == "parallel" and n_devices > int(num_chains):
            _warn_user(
                f"{self.name}: {int(num_chains)} chains across {n_devices} visible devices leaves "
                f"{n_devices - int(num_chains)} idle -- pmap places exactly one chain per device "
                f"and does not split a chain. Raise num_chains to {n_devices} to use the hardware "
                f"this process is already holding.")

        logp_op = _build_ops(log_prob_fn, jax, jnp)

        with pm.Model() as pmodel:
            theta = _pymc_prior(pm, pt, prior, names)
            pm.Potential("jax_log_likelihood", logp_op(theta))

        t0 = time.perf_counter()
        idata = sample_numpyro_nuts(
            draws=int(num_samples), tune=int(num_warmup), chains=int(num_chains),
            target_accept=float(target_accept_prob), random_seed=int(seed),
            chain_method=self.chain_method, progressbar=bool(progress), model=pmodel,
            initvals=initvals, jitter=initvals is None,
            idata_kwargs={"log_likelihood": False})
        runtime = init_time + time.perf_counter() - t0
        t_post = time.perf_counter()

        post = idata.posterior
        # Every fitted parameter is present by NAME -- LogUniform ones as Deterministics -- so the
        # log10 reparameterisation never leaks into what the caller sees.
        by_chain = np.stack([np.asarray(post[nm].values) for nm in names], axis=-1)
        n_chains_run, n_draws = by_chain.shape[0], by_chain.shape[1]
        theta_flat = by_chain.reshape(-1, k)
        samples = pd.DataFrame(_dg.fill_fixed(theta_flat, names, fixed, all_names),
                               columns=all_names)

        stats = idata.sample_stats
        # -1 = not recorded, which chain_health treats as a failure rather than as "none".
        n_div = int(np.asarray(stats["diverging"].values).sum()) if "diverging" in stats else -1
        # The step size is constant after adaptation; its last value per chain is the adapted one.
        steps = (np.asarray(stats["step_size"].values, dtype=float)[:, -1]
                 if "step_size" in stats else None)

        # Best draw by the exact log-density, in the fixed-width blocks the prior scan compiled:
        # constant compile in the draw count (a wide vmap here is the trap that once stalled
        # nuts_gpu for over an hour), and not one draw per step, which on a latency-bound model
        # cost more than the sampling (see nuts_gpu). The time is info["postprocess_s"].
        ll = sc["score"](theta_flat)
        b = int(np.nanargmax(ll))
        best = {nm: float(theta_flat[b, j]) for j, nm in enumerate(names)}
        best = {nm: best.get(nm, fixed.get(nm)) for nm in all_names}
        max_log_likelihood = float(ll[b])
        # Same checks, same meaning of `converged`, as nuts_gpu: the two are compared arm by arm.
        health = _dg.chain_health(by_chain, names, ll.reshape(n_chains_run, n_draws),
                                  n_divergences=n_div, step_size_by_chain=steps,
                                  reference_ll=reference_ll)
        _dg.warn_if_unconverged(self.name, health)
        postprocess_time = time.perf_counter() - t_post

        info = {
            "frontend": "pymc",
            "kernel": "numpyro NUTS (pymc.sampling.jax.sample_numpyro_nuts)",
            # EFFECTIVE, not requested. A result that labels itself 'parallel' after NumPyro
            # quietly downgraded it to 'sequential' is how a single-GPU run gets reported as a
            # multi-GPU speedup. `chain_method_requested` keeps the asked-for value for auditing.
            "chain_method": effective_chain_method,
            "chain_method_requested": self.chain_method,
            "chain_method_downgraded": bool(effective_chain_method != self.chain_method),
            "n_devices": int(n_devices),
            # Devices actually engaged: pmap uses one per chain, everything else uses one.
            "n_devices_used": (min(int(num_chains), int(n_devices))
                               if effective_chain_method == "parallel" else 1),
            "devices": [str(d) for d in jax.devices()],
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(n_chains_run),
            "n_draws_per_chain": int(n_draws),
            "n_divergences": n_div,
            # Key names match nuts_gpu's exactly (rhat, max_rhat, converged, ...). These two are
            # meant to be compared arm by arm, and a report that reads info["rhat"] from one and
            # info["rhat_by_param"] from the other silently gets None for half its table.
            **health,
            "init_strategy": init_label, "init": init_label, "init_detail": init_detail,
            "fixed": dict(fixed),
            "prior_scan": {"n_draws": int(len(sc["draws"])), "n_climbed": sc["n_climbed"],
                           "n_reaching_best": sc["n_good"], "climb_error": sc["climb_error"],
                           "plateau_fraction": sc["plateau_fraction"],
                           "best_log_likelihood": reference_ll,
                           "start_log_likelihood": (None if start_ll is None
                                                    else [float(v) for v in start_ll])},
            "float32_hazard": f32_hazard,
            "init_time_s": float(init_time),
            # After sampling, outside runtime_s: every kept draw's log-likelihood and the chain
            # checks, as for nuts_gpu.
            "postprocess_s": float(postprocess_time),
            "target_accept_prob": float(target_accept_prob),
            "space": space,
            "space_source": space_source,
            # `type(likelihood).__name__`, the key `samplers.base._LIKELIHOOD_KINDS` maps back to a
            # registry name so `waic`/`predictive_metrics` re-score under the density that was
            # fitted. None when the caller supplied their own log_prob_fn -- we do not know then.
            "likelihood": getattr(log_prob_fn, "likelihood", None),
            "scatter_param": getattr(log_prob_fn, "scatter_param", None),
            "log_prob_fn": fn_source,
            "x64": bool(run_x64),
            "x64_session": bool(x64_before),
            "pymc_version": pm.__version__,
        }
        attach_band_metrics(info, lc, model, best, space)

        # The shared formula, not an inline copy: this one read log(max(n, 1)) where aic_bic read
        # log(n), so an empty light curve gave BIC 0.0 here and -inf from nuts_gpu.
        aic, bic = aic_bic(max_log_likelihood, k, n)
        result = SamplerResult(
            sampler=self.name, model=model.name, parameters=all_names, samples=samples,
            summary=summarize_posterior(samples, all_names), best_params=best, n_data=n,
            n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood, aic=aic, bic=bic)
        # min_distance is left at its default NaN rather than set to None, because
        # `SamplerResult.to_dict` does `float(self.min_distance)` -- so `result.to_json()`, the
        # documented way to save a fit, raised TypeError for these two samplers and no other.
        # NaN is the field's declared "this sampler has no distance" value; nuts_gpu, mcmc and
        # snpe all leave it that way.
        result.samples_by_chain = _dg.fill_fixed(by_chain, names, fixed, all_names)
        result.arviz_idata = idata
        attach_predictive_metrics(result, lc, space, model=model)
        return result


class PyMCJAXVectorizedSampler(_PyMCJAXBase):
    """PyMC frontend, NumPyro NUTS, chains vmapped on one device.

    Same kernel and chain strategy as ``nuts_gpu``, so it should reproduce that posterior to
    sampling noise — which is exactly what makes it a regression test on the PyTensor-to-JAX bridge.
    """

    name = "pymc_jax_gpu_vectorized"
    chain_method = "vectorized"


class PyMCJAXParallelizedSampler(_PyMCJAXBase):
    """PyMC frontend, NumPyro NUTS, chains ``pmap``ed across devices — the multi-GPU path.

    **Requires ``num_chains <= jax.local_device_count()``.** Above that NumPyro silently downgrades
    to ``sequential`` (see :func:`_effective_chain_method`) and this becomes a single-device
    sampler that is *slower* than ``pymc_jax_gpu_vectorized``, because sequential chains lose the
    vmap that the vectorized sampler keeps. ``fit`` warns and records the downgrade in
    ``info['chain_method']``, but it does not silently rewrite ``num_chains``: dropping chains
    weakens r-hat, and that is the caller's call to make.

    On a two-GPU budget the honest options are ``num_chains=2`` here, or ``num_chains=4`` on
    ``pymc_jax_gpu_vectorized``, or two 2-chain processes pinned one per GPU.
    """

    name = "pymc_jax_gpu_parallelized"
    chain_method = "parallel"


def fit_PyMCJAXVectorized(lc, model, prior=None, **kwargs) -> SamplerResult:
    """See :class:`PyMCJAXVectorizedSampler`."""
    return PyMCJAXVectorizedSampler().fit(lc, model, prior=prior, **kwargs)


def fit_PyMCJAXParallelized(lc, model, prior=None, **kwargs) -> SamplerResult:
    """See :class:`PyMCJAXParallelizedSampler`."""
    return PyMCJAXParallelizedSampler().fit(lc, model, prior=prior, **kwargs)
