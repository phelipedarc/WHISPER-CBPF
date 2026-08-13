"""NUTS through PyMC's model API, executed by NumPyro on the GPU.

ONE KERNEL, TWO KNOBS
    PyMC's JAX path does not implement its own sampler: ``pymc.sampling.jax.sample_numpyro_nuts``
    lowers the PyTensor graph to JAX and hands it to **NumPyro's NUTS** -- the same kernel
    ``nuts_gpu`` calls directly. So these samplers are not a competing algorithm. What they vary is
    the *frontend* (PyMC's model API instead of a hand-written log-density) and the *chain execution
    strategy*:

    ``pymc_jax_gpu_vectorized``   ``chain_method="vectorized"`` -- chains vmapped on ONE device.
                                  Same kernel and same strategy as ``nuts_gpu``, so it must
                                  reproduce that posterior to sampling noise. That makes it a free
                                  regression test on this whole PyTensor-to-JAX bridge rather than
                                  merely a new feature.
    ``pymc_jax_gpu_parallelized`` ``chain_method="parallel"`` -- chains ``pmap``ed ACROSS devices.
                                  This is the multi-GPU path. It needs more than one visible device
                                  to mean anything; with one it silently degenerates to sequential
                                  chains, so ``fit`` warns rather than reporting a fake result.

WHY A CUSTOM PyTensor Op IS UNAVOIDABLE
    The kilonova forward model is JAX. PyMC builds PyTensor graphs. Bridging them needs an ``Op``
    that PyTensor can place in a graph and that the JAX backend knows how to lower, which is what
    ``jax_funcify`` registration provides.

    The subtle half is the GRADIENT. PyTensor differentiates symbolically at the graph level and
    *then* lowers, so it cannot see inside a JAX callable -- an Op with no ``grad`` leaves NUTS with
    nothing to differentiate, and the failure is a confusing PyTensor error far from the cause. So
    the value Op declares its gradient to be a second Op wrapping ``jax.grad`` of the same function,
    and both are registered with the JAX backend. The chain is then: PyMC graph -> PyTensor grad ->
    JAX lowering -> one fused XLA program per chain.

    The log-density itself is passed in, not rebuilt. ``nuts_gpu``, ``emcee_jax`` and these samplers
    are meant to receive the *same object*, so a disagreement between them is the sampler and never
    the likelihood.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from ...samplers.base import (
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
)

from ...backends._env import require_jax
from ...models import get_model
from ._adapters import resolve_density as _resolve_density


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
        kind = type(d).__name__
        lo, hi = (float(x) for x in d.bounds)
        if kind == "Uniform":
            out.append(pm.Uniform(nm, lower=lo, upper=hi))
        elif kind == "LogUniform":
            log_v = pm.Uniform(f"log10_{nm}", lower=np.log10(lo), upper=np.log10(hi))
            out.append(pm.Deterministic(nm, 10.0 ** log_v))
        else:
            raise TypeError(
                f"pymc_gpu cannot express prior {kind!r} on parameter {nm!r}. Supported: Uniform, "
                f"LogUniform. Extend _pymc_prior rather than substituting a uniform box -- that "
                f"changes the posterior silently.")
    return pt.stack(out)


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
            num_chains=4, space="auto", seed=0, progress=False,
            target_accept_prob=0.8) -> SamplerResult:
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
        num_warmup, num_samples, num_chains, seed, target_accept_prob
            NUTS budget and adaptation, as for ``nuts_gpu``.
        """
        jax, jnp = require_jax(self.name)
        try:
            import pymc as pm
            import pytensor.tensor as pt
            from pymc.sampling.jax import sample_numpyro_nuts
        except ImportError as exc:                      # pragma: no cover - environment dependent
            raise ImportError(
                f"{self.name} needs PyMC on top of the JAX stack: "
                f"`pip install 'whisper-cbpf[gpu,pymc]'` (or `pip install pymc`). It is an optional "
                f"dependency -- no other sampler requires it, which is why the import is here "
                f"rather than at module scope.") from exc

        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        names = list(model.parameters)
        k, n = len(names), int(len(lc.time))
        log_prob_fn, space, space_source, fn_source = _resolve_density(
            lc, model, prior, log_prob_fn, space, names,
            include_prior=False, sampler=self.name)

        # local_device_count(), not device_count(): NumPyro checks the LOCAL count, and the two
        # differ on a multi-host setup.
        n_devices = jax.local_device_count()
        effective_chain_method = _effective_chain_method(self.chain_method, num_chains, n_devices)

        if effective_chain_method != self.chain_method:
            warnings.warn(
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
                stacklevel=2)
        elif self.chain_method == "parallel" and n_devices > int(num_chains):
            warnings.warn(
                f"{self.name}: {int(num_chains)} chains across {n_devices} visible devices leaves "
                f"{n_devices - int(num_chains)} idle -- pmap places exactly one chain per device "
                f"and does not split a chain. Raise num_chains to {n_devices} to use the hardware "
                f"this process is already holding.", stacklevel=2)

        logp_op = _build_ops(log_prob_fn, jax, jnp)

        with pm.Model() as pmodel:
            theta = _pymc_prior(pm, pt, prior, names)
            pm.Potential("jax_log_likelihood", logp_op(theta))

        t0 = time.perf_counter()
        idata = sample_numpyro_nuts(
            draws=int(num_samples), tune=int(num_warmup), chains=int(num_chains),
            target_accept=float(target_accept_prob), random_seed=int(seed),
            chain_method=self.chain_method, progressbar=bool(progress), model=pmodel,
            idata_kwargs={"log_likelihood": False})
        runtime = time.perf_counter() - t0

        post = idata.posterior
        # Every fitted parameter is present by NAME -- LogUniform ones as Deterministics -- so the
        # log10 reparameterisation never leaks into what the caller sees.
        by_chain = np.stack([np.asarray(post[nm].values) for nm in names], axis=-1)
        n_chains_run, n_draws = by_chain.shape[0], by_chain.shape[1]
        theta_flat = by_chain.reshape(-1, k)
        samples = pd.DataFrame(theta_flat, columns=names)

        n_div = -1
        try:
            n_div = int(np.asarray(idata.sample_stats["diverging"].values).sum())
        except Exception:
            pass
        rhat_by_param = {}
        try:
            import arviz as az
            rh = az.rhat(idata.posterior[names])
            rhat_by_param = {nm: float(np.asarray(rh[nm])) for nm in names}
        except Exception:
            pass

        # Best draw by the exact log-density, scanned with lax.map so compile stays constant in the
        # draw count (a wide vmap here is the trap that once stalled nuts_gpu for over an hour).
        ll = np.asarray(jax.block_until_ready(
            jax.lax.map(jax.jit(log_prob_fn), jnp.asarray(theta_flat))))
        b = int(np.nanargmax(ll))
        best = {nm: float(theta_flat[b, j]) for j, nm in enumerate(names)}
        max_log_likelihood = float(ll[b])

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
            # Key name matches nuts_gpu's exactly. These two are meant to be compared arm by arm,
            # and a report that reads info["rhat"] from one and info["rhat_by_param"] from the
            # other silently gets None for half its table.
            "rhat": rhat_by_param,
            "max_rhat": (float(max(rhat_by_param.values())) if rhat_by_param else None),
            "converged": bool(n_div == 0 and rhat_by_param
                              and max(rhat_by_param.values()) < 1.01),
            "target_accept_prob": float(target_accept_prob),
            "space": space,
            "space_source": space_source,
            "log_prob_fn": fn_source,
            "x64": bool(jax.config.jax_enable_x64),
            "pymc_version": pm.__version__,
        }
        attach_band_metrics(info, lc, model.name, best, space)

        aic = 2.0 * k - 2.0 * max_log_likelihood
        bic = k * np.log(max(n, 1)) - 2.0 * max_log_likelihood
        result = SamplerResult(
            sampler=self.name, model=model.name, parameters=names, samples=samples,
            summary=summarize_posterior(samples, names), best_params=best, n_data=n,
            n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood, aic=aic, bic=bic)
        # min_distance is LEFT AT ITS DEFAULT NaN, not set to None. It used to be None here, and
        # `SamplerResult.to_dict` does `float(self.min_distance)` -- so `result.to_json()`, the
        # documented way to save a fit, raised TypeError for these two samplers and no other.
        # NaN is the field's declared "this sampler has no distance" value; nuts_gpu, mcmc and
        # snpe all leave it that way.
        result.samples_by_chain = by_chain
        result.arviz_idata = idata
        attach_predictive_metrics(result, lc, space)
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
