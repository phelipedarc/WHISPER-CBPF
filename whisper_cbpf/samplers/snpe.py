"""Sequential Neural Posterior Estimation (SNPE / NPE) via ``sbi``.

Simulation-based inference: instead of an explicit likelihood, SNPE trains a neural density estimator
``q(theta | x)`` on (parameters, simulated light curve) pairs drawn from the prior, then conditions it
on the **observed** light curve to get the posterior. Running it over several *rounds* (each proposing
from the latest posterior) focuses simulations on the good region — the "Sequential" in SNPE.

How it plugs into Whisper:

* **Simulator** = Whisper's forward model. For a parameter vector it calls ``model.predict`` at the
  observed times/bands, maps the prediction into the data space (flux or magnitude, like
  :class:`~whisper_cbpf.likelihood.GaussianLikelihood`), and adds Gaussian noise with the per-point
  data error — so the implicit likelihood matches Whisper's Gaussian likelihood.
* **Prior** = Whisper's :class:`~whisper_cbpf.priors.Prior`, adapted to a torch prior (``Uniform`` →
  ``BoxUniform``; a mix with ``LogUniform``, ``Normal`` or ``TruncatedNormal`` →
  ``MultipleIndependent``). A ``Fixed`` parameter is held at its value, outside the torch prior.
* **Result** = the same :class:`~whisper_cbpf.samplers.SamplerResult` every other sampler returns,
  with exact Gaussian ``max_log_likelihood`` / ``AIC`` / ``BIC`` evaluated at the best posterior draw.
  The trained sbi posterior is attached as ``result.posterior`` (and ``result.posteriors`` per round)
  for resampling / sbi ``pairplot``.

``sbi`` + ``torch`` are the optional ``[sbi]`` extra; they are imported lazily, so importing Whisper
never requires them. ``num_rounds=1`` gives amortized NPE; ``num_rounds>1`` is sequential SNPE.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..likelihood import _MIN_FLUX_JY
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


def _snpe_logl_worker(args):
    """Exact log-likelihood of one posterior draw (module-level -> picklable for the parallel scan)."""
    predict, names, times, bands, lik, scatter_idx, row = args
    try:
        params = {nm: float(v) for nm, v in zip(names, row)}
        mf = predict(params, times, bands)
        if scatter_idx is not None:
            return float(lik.log_likelihood(mf, sigma_extra=float(row[scatter_idx])))
        return float(lik.log_likelihood(mf))
    except Exception:
        return float("-inf")


def _parallel_max_logl(worker_args, n_jobs, timeout):
    """Evaluate the best-fit scan in parallel with a hard wall-clock cap.

    The scan re-runs the (possibly expensive) forward model on each posterior draw; a single
    pathological draw can make that model hang indefinitely and, in a serial loop, freeze the whole
    fit. Running it in a process pool with a ``timeout`` means a hung draw costs at most ``timeout``
    seconds: the pool is terminated and the best likelihood among the draws that DID finish is used.
    Returns ``(logls_array_or_None, best_index)`` — ``None`` if nothing finished in time.
    """
    import multiprocessing as mp

    n = len(worker_args)
    n_jobs = max(1, min(int(n_jobs), n))
    if n_jobs == 1:
        return np.array([_snpe_logl_worker(a) for a in worker_args], dtype=float), None
    ctx = mp.get_context("fork")
    pool = ctx.Pool(n_jobs)
    try:
        out = pool.map_async(_snpe_logl_worker, worker_args)
        logls = np.array(out.get(timeout=timeout), dtype=float)
        return logls, None
    except mp.TimeoutError:
        return None, None                    # a draw hung; caller falls back to a bounded serial scan
    finally:
        pool.terminate()
        pool.join()


def _require_sbi():
    """Lazily import sbi/torch; raise a clear, actionable error if the ``[sbi]`` extra is missing."""
    try:
        import torch
        from sbi.inference import NPE, simulate_for_sbi
        from sbi.utils import (
            BoxUniform,
            MultipleIndependent,
            RestrictedPrior,
            get_density_thresholder,
        )
        from sbi.utils.user_input_checks import (
            check_sbi_inputs,
            process_prior,
            process_simulator,
        )
    except Exception as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "SNPE needs the optional 'sbi' + 'torch' dependencies (the `[sbi]` extra). Install with "
            "`pip install 'whisper-cbpf[sbi]'` (or `pip install sbi torch`).") from exc
    return SimpleNamespace(
        torch=torch, NPE=NPE, simulate_for_sbi=simulate_for_sbi, BoxUniform=BoxUniform,
        MultipleIndependent=MultipleIndependent, RestrictedPrior=RestrictedPrior,
        get_density_thresholder=get_density_thresholder, process_prior=process_prior,
        process_simulator=process_simulator, check_sbi_inputs=check_sbi_inputs)


_BOUNDED_LOG_UNIFORM_CLS = None


def _bounded_log_uniform_cls(torch):
    """The ``exp(Uniform(log a, log b))`` distribution class, with an **honest** ``[a, b]`` support.

    A plain ``TransformedDistribution(Uniform(log a, log b), ExpTransform())`` is the right *measure*
    but advertises the wrong *support*: torch derives ``.support`` from the last transform's codomain,
    so ``ExpTransform`` reports ``GreaterThan(0)`` and the ``[a, b]`` bounds are lost. That is not
    cosmetic. sbi's ``within_support`` -- the accept/reject test behind ``DirectPosterior.sample()``,
    i.e. the last thing that runs before draws are handed back -- prefers ``support.check(theta)`` over
    ``log_prob(theta)``, so with the codomain support the rejection step only checks positivity and
    every ``LogUniform`` parameter leaks draws from outside the prior box into the final posterior
    (measured at 28-74% of draws on the four AT2017gfo kilonova SNPE arms; the one plain ``Uniform``
    parameter, which has a real ``Interval`` support, leaked 0.00%).

    Only the *advertised support* changes here: same log-space uniform base, same ``ExpTransform``,
    so ``sample`` and ``log_prob`` are bit-for-bit what they were. ``log_prob`` was always correct
    (``-inf`` outside the box) -- it was simply not the test sbi ran.

    Built lazily and cached because torch is an optional dependency (imported only in ``_require_sbi``).
    """
    global _BOUNDED_LOG_UNIFORM_CLS
    if _BOUNDED_LOG_UNIFORM_CLS is not None:
        return _BOUNDED_LOG_UNIFORM_CLS

    constraints = torch.distributions.constraints

    class _BoundedLogUniform(torch.distributions.TransformedDistribution):
        """``exp(Uniform(log low, log high))`` whose ``.support`` is ``interval(low, high)``."""

        def __init__(self, base_distribution, transforms, low, high, validate_args=None):
            # Plain floats, not tensors: `constraints.interval` compares them against a tensor
            # elementwise, so the support stays device-agnostic and survives sbi's
            # `move_distribution_to_device` (which deep-moves tensor attributes) untouched.
            self._low, self._high = float(low), float(high)
            super().__init__(base_distribution, transforms, validate_args=validate_args)

        @constraints.dependent_property(is_discrete=False, event_dim=0)
        def support(self):
            return constraints.interval(self._low, self._high)

        def expand(self, batch_shape, _instance=None):
            # TransformedDistribution.expand() only copies base_dist/transforms onto the new
            # instance, so carry the bounds across or the expanded copy loses its support.
            new = self._get_checked_instance(_BoundedLogUniform, _instance)
            new._low, new._high = self._low, self._high
            return super().expand(batch_shape, _instance=new)

    _BOUNDED_LOG_UNIFORM_CLS = _BoundedLogUniform
    return _BoundedLogUniform


_TRUNCATED_NORMAL_CLS = None


def _truncated_normal_cls(torch):
    """A torch distribution for whisper's :class:`~whisper_cbpf.priors.TruncatedNormal`.

    torch has no truncated normal. This one is the whisper class moved into torch, term for term:

    * ``log_prob``: ``-z^2/2 - log sigma - log sqrt(2 pi) - log Z`` on the closed range
      ``[low, high]`` and ``-inf`` outside, ``Z`` the truncation mass (computed once, in float64, by
      the whisper class), so the density sbi's round-2 loss evaluates is exact;
    * ``sample``: the inverse CDF of ``U(0, 1)`` draws (the probability integral transform), in
      float64 and on the upper tail's side when the range lies above ``mu``, as the whisper class
      draws; ``u`` is kept inside ``(0, 1)`` and the result clamped into the range after the cast
      to torch's default dtype, so a draw never lands outside it;
    * ``support``: the real range (``interval``, ``greater_than_eq``, ``less_than`` or ``real``), the
      test sbi's rejection sampler runs, as :func:`_bounded_log_uniform_cls` does for LogUniform;
    * ``mean`` / ``variance``: the truncated distribution's (scipy), for sbi's transforms.

    Built lazily and cached because torch is an optional dependency.
    """
    global _TRUNCATED_NORMAL_CLS
    if _TRUNCATED_NORMAL_CLS is not None:
        return _TRUNCATED_NORMAL_CLS

    import math

    constraints = torch.distributions.constraints
    log_sqrt_2pi = 0.5 * math.log(2.0 * math.pi)

    class _TruncatedNormal(torch.distributions.Distribution):
        """whisper's ``TruncatedNormal(mu, sigma, low, high)`` as a 1-D torch distribution."""

        arg_constraints = {}
        has_rsample = False

        def __init__(self, dist, device="cpu", validate_args=False):
            from scipy.stats import truncnorm

            self._mu, self._sigma = float(dist.mu), float(dist.sigma)
            self._low, self._high = float(dist.low), float(dist.high)
            self._lo_cdf, self._mass = float(dist._lo_cdf), float(dist._mass)
            self._log_mass, self._flip = float(dist._log_mass), bool(dist._flip)
            ref = truncnorm(dist._alpha, dist._beta, loc=self._mu, scale=self._sigma)
            self._mean, self._var = float(ref.mean()), float(ref.var())
            self._device = torch.device(device)
            super().__init__(batch_shape=torch.Size([1]), validate_args=validate_args)

        def to(self, device):
            self._device = torch.device(device)
            return self

        @constraints.dependent_property(is_discrete=False, event_dim=0)
        def support(self):
            lo_inf, hi_inf = math.isinf(self._low), math.isinf(self._high)
            if lo_inf and hi_inf:
                return constraints.real
            if hi_inf:
                return constraints.greater_than_eq(self._low)
            if lo_inf:
                return constraints.less_than(self._high)
            return constraints.interval(self._low, self._high)

        @property
        def mean(self):
            return torch.full((1,), self._mean, device=self._device)

        @property
        def variance(self):
            return torch.full((1,), self._var, device=self._device)

        def icdf(self, value):
            p = torch.as_tensor(value, dtype=torch.float64, device=self._device)
            # torch.rand can return 0, whose quantile is -inf; eps keeps p * Z a normal number
            # (a subnormal one can be flushed to 0 on a GPU)
            eps = torch.finfo(torch.float64).eps
            p = p.clamp(eps, 1.0 - eps)
            if self._flip:
                z = -torch.special.ndtri(self._lo_cdf + (1.0 - p) * self._mass)
            else:
                z = torch.special.ndtri(self._lo_cdf + p * self._mass)
            return (self._mu + self._sigma * z).clamp(self._low, self._high)

        def sample(self, sample_shape=torch.Size()):
            shape = self._extended_shape(sample_shape)
            with torch.no_grad():
                u = torch.rand(shape, dtype=torch.float64, device=self._device)
                x = self.icdf(u).to(torch.get_default_dtype())
                return x.clamp(self._low, self._high)

        def log_prob(self, value):
            v = torch.as_tensor(value)
            x = v.to(torch.float64)
            z = (x - self._mu) / self._sigma
            lp = -0.5 * z * z - math.log(self._sigma) - log_sqrt_2pi - self._log_mass
            inside = (x >= self._low) & (x <= self._high)
            lp = torch.where(inside, lp, torch.full_like(lp, -math.inf))
            return lp.to(v.dtype if v.is_floating_point() else torch.get_default_dtype())

    _TRUNCATED_NORMAL_CLS = _TruncatedNormal
    return _TruncatedNormal


class _HoldFixed:
    """``predict`` with the prior's ``Fixed`` parameters added to every call: SNPE samples the free
    parameters only. A module-level class, so it pickles to ``num_workers > 1`` workers."""

    def __init__(self, predict, fixed):
        self.predict, self.fixed = predict, dict(fixed)

    def __call__(self, params, times, bands=None):
        full = dict(params)
        full.update(self.fixed)
        return self.predict(full, times, bands)


def _to_torch_prior(prior, sb, device="cpu"):
    """Adapt a Whisper :class:`Prior` to a torch prior sbi can use, on ``device``.

    All-``Uniform`` priors become a single ``BoxUniform`` (fast); any other mix becomes a
    ``MultipleIndependent`` of per-parameter 1-D distributions: ``LogUniform`` = exp of a uniform in
    log-space, via :func:`_bounded_log_uniform_cls` so the ``[low, high]`` bounds reach sbi's
    rejection sampler; ``Normal`` = ``torch.distributions.Normal``; ``TruncatedNormal`` via
    :func:`_truncated_normal_cls` (exact density and draws). A ``Fixed`` parameter is not a
    dimension of the torch prior (:meth:`SNPESampler.fit` holds it at its value), and other
    distribution types raise a clear error. The prior tensors are created on ``device`` so it
    matches the (GPU) training device sbi requires.
    """
    from ..priors import LogUniform, Normal, TruncatedNormal, Uniform
    from ..priors._numpy import family

    torch = sb.torch
    lows, highs, comps, all_uniform = [], [], [], True
    for name in prior.names:
        d = prior.distributions[name]
        if isinstance(d, Uniform):
            lows.append(float(d.low))
            highs.append(float(d.high))
            comps.append(torch.distributions.Uniform(
                torch.tensor([float(d.low)], device=device), torch.tensor([float(d.high)], device=device)))
        elif isinstance(d, LogUniform):
            all_uniform = False
            base = torch.distributions.Uniform(
                torch.tensor([float(np.log(d.low))], device=device),
                torch.tensor([float(np.log(d.high))], device=device))
            comps.append(_bounded_log_uniform_cls(torch)(
                base, torch.distributions.ExpTransform(), float(d.low), float(d.high)))
        elif isinstance(d, Normal):
            all_uniform = False
            comps.append(torch.distributions.Normal(
                torch.tensor([float(d.mu)], device=device),
                torch.tensor([float(d.sigma)], device=device)))
        elif isinstance(d, TruncatedNormal):
            all_uniform = False
            comps.append(_truncated_normal_cls(torch)(d, device=device))
        elif family(d) == "Fixed":
            raise TypeError(
                f"parameter {name!r} is Fixed, which is not a dimension of the torch prior: "
                f"SNPESampler.fit holds it at its value. Pass the prior to fit(), or leave "
                f"{name!r} out of the prior given to _to_torch_prior.")
        else:
            raise TypeError(
                f"SNPE prior adapter supports Uniform/LogUniform/Normal/TruncatedNormal (and "
                f"Fixed, held at its value by fit); got {type(d).__name__} for parameter "
                f"{name!r}. Express the prior in one of these, or use sampler='abc' / 'mcmc'.")
    if len(comps) == 1 and not all_uniform:
        # sbi's MultipleIndependent needs at least two components; one 1-D distribution with
        # batch_shape (1,) is already the joint.
        return comps[0]
    if all_uniform:
        return sb.BoxUniform(low=torch.tensor(lows, device=device), high=torch.tensor(highs, device=device))
    # device= is REQUIRED, not optional: sbi's MultipleIndependent does `self.to(device or "cpu")`
    # in __init__, which MOVES the per-parameter distributions built above back to the CPU. sbi then
    # refuses to train (`check_if_prior_on_device`: "Prior device 'cpu' must match training device
    # 'cuda:0'"), so any mixed Uniform/LogUniform prior -- i.e. every kilonova prior in this repo --
    # could not use device='cuda' at all. The kwarg is accepted from sbi 0.23; guard it for older ones.
    try:
        return sb.MultipleIndependent(comps, device=device)
    except TypeError:
        return sb.MultipleIndependent(comps)


def _build_simulator(predict, names, times, bands, model_in_space, sigma, torch, seed, extra=None,
                     scatter_idx=None):
    """A batched sbi simulator: parameter vector(s) -> noisy light curve in the data space.

    The observational noise for each parameter row is seeded from ``(seed, that row's values)``, so it
    is **reproducible** and **independent of how sbi chunks the batch across workers** — sharing a
    single RNG would make all ``num_workers>1`` processes add identical noise. When ``extra`` (a flat
    array of constant context channels: errors / times / band codes) is given, it is appended to every
    row — the ``x_format="stacked"`` layout. ``scatter_idx`` marks the theta column holding a free
    extra-scatter term (Villar+2017): that row's noise becomes ``N(0, sqrt(sigma² + scatter²))``, so
    the network *learns* the scatter dimension from its imprint on the simulations.
    """
    import zlib

    n_obs = len(times)
    sig = np.asarray(sigma, dtype=float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)   # guard degenerate/zero errors
    base_seed = int(seed)
    extra = None if extra is None else np.asarray(extra, dtype=float).ravel()

    def simulator(theta):
        th = np.asarray(theta.detach().cpu().numpy(), dtype=float)
        single = th.ndim == 1
        if single:
            th = th[None, :]
        width = n_obs if extra is None else n_obs + len(extra)
        out = np.empty((th.shape[0], width), dtype=float)
        for i in range(th.shape[0]):
            params = {nm: float(th[i, j]) for j, nm in enumerate(names)}
            model_flux = np.asarray(predict(params, times, bands), dtype=float)
            mf = np.nan_to_num(model_in_space(model_flux), nan=0.0, posinf=0.0, neginf=0.0)
            row_seed = (base_seed + zlib.crc32(th[i].astype(np.float64).tobytes())) & 0xFFFFFFFF
            row_sig = sig if scatter_idx is None else np.sqrt(sig ** 2 + th[i, scatter_idx] ** 2)
            out[i, :n_obs] = mf + np.random.default_rng(row_seed).normal(0.0, row_sig)
            if extra is not None:
                out[i, n_obs:] = extra
        res = torch.as_tensor(out, dtype=torch.float32)
        return res[0] if single else res

    return simulator


def _torch_model_in_space(space, zeropoint_jy, torch):
    """Torch counterpart of ``GaussianLikelihood.model_in_space``, for the GPU simulate path.

    Identity in flux space; in magnitude space ``-2.5 log10(clamp(F) / zp)``.

    THE FLOOR IS NOT ALWAYS 1e-300. The numpy likelihood clamps at ``_MIN_FLUX_JY = 1e-300``, which
    is fine in float64 and silently useless here: these tensors are usually float32, where 1e-300
    underflows to exactly 0 and the clamp stops clamping — log10(0) then gives -inf and the whole
    simulation batch becomes NaN. (The same defect, in the same form, is documented in the
    ``MAG_FLOOR`` note of :mod:`whisper_cbpf.models.jax.kilonova`.) Clamping at the dtype's smallest
    normal keeps a non-positive model flux finite at ~104 mag in float32, far fainter than any real
    detection and therefore harmless.

    The rule is ``max(_MIN_FLUX_JY, finfo.tiny)``, identical to
    :func:`whisper_cbpf.likelihood._jax.log_likelihood_jax` and
    :func:`whisper_cbpf.samplers.jax.abc_gpu._model_in_space_jnp`. A bare
    ``finfo.tiny``, which agreed with them in float32 but not in float64 — there ``tiny`` is
    2.2e-308, giving **778.03 mag** where every other backend gives 758.90. A 19.13-magnitude
    backend-dependent floor is exactly the sort of thing that makes a cross-sampler comparison
    incomparable, which is what this package exists to do.
    """
    if space == "flux":
        return lambda f: f

    def to_mag(f):
        tiny = torch.finfo(f.dtype).tiny
        floor = _MIN_FLUX_JY if _MIN_FLUX_JY > tiny else tiny
        return -2.5 * torch.log10(torch.clamp(f, min=floor) / zeropoint_jy)

    return to_mag


def _predict_torch_accepts_bands(predict_torch):
    """Does this batched torch forward map take a third, ``bands`` argument?

    The hook's original signature was ``predict_torch(theta, times)``, which is why **no photometric
    model could use it**: turning parameters into a flux in a filter needs to know which filter each
    observation is in, and there was nowhere to say so. Rather than break the two-argument callables
    that already exist (the flare's), the argument is passed only when the callable declares it.
    A ``*args``/``**kwargs`` signature counts as accepting it.
    """
    import inspect

    try:
        sig = inspect.signature(predict_torch)
    except (TypeError, ValueError):                    # builtins / C callables: assume the old shape
        return False
    positional = 0
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_POSITIONAL or p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 3


def _torch_simulate(proposal, num_simulations, predict_torch, times_t, sig_t, extra_t, torch, seed,
                    show_progress=False, scatter_idx=None, model_in_space=None, bands=None,
                    theta=None):
    """GPU-vectorized simulation: one batched ``predict_torch`` call replaces the per-row Python loop.

    ``predict_torch(theta, times[, bands])`` maps a ``(B, D)`` parameter tensor and ``(n,)`` time
    tensor to a ``(B, n)`` FLUX tensor **on the same device** — so 30k simulations are a single
    kernel launch instead of 30k Python iterations. ``bands`` (the light curve's own band labels, in
    row order) is passed only when the callable declares a third parameter, so a **photometric**
    model can resolve each observation's filter; see :func:`_predict_torch_accepts_bands`.

    ``model_in_space`` then maps that flux into the COMPARISON space before noise is added, exactly
    as the CPU simulator does (`mf = model_in_space(flux)`, then `mf + noise`). Getting that order
    wrong — adding magnitude-scale noise to a flux, or flux-scale noise to a magnitude — silently
    trains the flow on the wrong summaries.

    ``theta`` may be supplied already drawn. That is how the multi-round leakage guard reaches this
    path: the between-round draw from a trained posterior is plain rejection sampling against the
    prior and can stall indefinitely, so the caller draws it through
    :func:`_robust_proposal_draw` and hands the result in rather than letting ``proposal.sample``
    run unguarded here.
    """
    with torch.no_grad():
        if theta is None:
            try:
                theta = proposal.sample((int(num_simulations),), show_progress_bars=show_progress)
            except TypeError:                          # plain torch priors take no progress kwarg
                theta = proposal.sample((int(num_simulations),))
        theta = theta.to(times_t.device)               # e.g. RestrictedPrior samples on CPU (sbi 0.23)
        flux = (predict_torch(theta, times_t, bands) if _predict_torch_accepts_bands(predict_torch)
                else predict_torch(theta, times_t))
        flux = flux.to(times_t.device)
        # Space map FIRST, noise AFTER -- the order the CPU simulator uses. nan_to_num runs after
        # the map so a non-finite magnitude is scrubbed too, not just a non-finite flux.
        mf = flux if model_in_space is None else model_in_space(flux)
        mf = torch.nan_to_num(mf, nan=0.0, posinf=0.0, neginf=0.0)
        gen = torch.Generator(device=mf.device.type)
        gen.manual_seed(int(seed) & 0x7FFFFFFF)
        sig_row = (sig_t if scatter_idx is None
                   else torch.sqrt(sig_t ** 2 + theta[:, scatter_idx:scatter_idx + 1] ** 2))
        x = mf + torch.randn(mf.shape, generator=gen, device=mf.device) * sig_row
        if extra_t is not None:
            x = torch.cat([x, extra_t.unsqueeze(0).expand(x.shape[0], -1)], dim=1)
        return theta.float(), x.float()


def _build_density_estimator(density_estimator, embedding_net, hidden_features,
                             num_transforms, num_bins):
    """Resolve the sbi density estimator.

    * a callable (an already-built ``posterior_nn(...)`` factory) is used as-is;
    * a string with no extra options is passed straight to ``NPE`` (sbi builds the default);
    * a string **plus** an ``embedding_net`` and/or hyperparameters is wrapped in ``posterior_nn`` so
      a custom feature-extractor / architecture can be used (essential for high-dimensional,
      multi-band light curves).
    """
    if callable(density_estimator) and not isinstance(density_estimator, str):
        return density_estimator
    if embedding_net is None and hidden_features is None and num_transforms is None \
            and num_bins is None:
        return density_estimator
    from sbi.neural_nets import posterior_nn
    kwargs = {}
    if embedding_net is not None:
        kwargs["embedding_net"] = embedding_net
    if hidden_features is not None:
        kwargs["hidden_features"] = hidden_features
    if num_transforms is not None:
        kwargs["num_transforms"] = num_transforms
    if num_bins is not None:
        kwargs["num_bins"] = num_bins
    return posterior_nn(model=density_estimator, **kwargs)


class _CPUSampleProxy:
    """Proposal wrapper whose samples live on the CPU.

    sbi's parallel ``simulate_for_sbi`` (``num_workers > 1``) calls ``theta.numpy()`` on the
    proposal's samples, which raises for CUDA-resident proposals (the prior/posterior live on the
    training device). Only ``.sample`` is needed there, so this thin proxy moves the draws to CPU.
    """

    def __init__(self, proposal):
        self._proposal = proposal

    def sample(self, sample_shape, **kwargs):
        try:
            s = self._proposal.sample(sample_shape, **kwargs)
        except TypeError:                          # plain torch priors take no extra kwargs
            s = self._proposal.sample(sample_shape)
        return s.detach().cpu()


def _resolve_device(device, torch):
    """Resolve the requested SNPE training device, falling back to CPU when CUDA is unavailable.

    Accepts ``'cpu'``, ``'cuda'`` / ``'gpu'`` / ``'cuda:N'``, or ``'auto'`` (use CUDA when available,
    else CPU). Requesting a GPU without one available warns and uses CPU rather than crashing.
    """

    d = str(device).lower()
    cuda_ok = bool(torch.cuda.is_available())
    if d == "auto":
        return "cuda" if cuda_ok else "cpu"
    if d in ("gpu", "cuda") or d.startswith("cuda:"):
        if not cuda_ok:
            _warn_user(f"SNPE device={device!r} requested but CUDA is unavailable; using CPU.",
                         )
            return "cpu"
        return "cuda" if d == "gpu" else d
    return "cpu"


def _probe_acceptance_rate(posterior, torch_prior, torch, x_o_norm, n_probe=4000):
    """Bounded, single-shot estimate of a posterior's within-prior acceptance rate at ``x_o_norm``.

    sbi's own ``leakage_correction()``/``log_prob(norm_posterior=True)`` estimate this by
    accept-reject sampling *until N are accepted* — at a pathologically low acceptance rate that
    is exactly as slow as the failure it is meant to diagnose (confirmed: it hangs identically).
    This instead draws a FIXED ``n_probe`` sample directly from the density estimator (one
    unconditional forward pass — cost independent of the acceptance rate) and checks what fraction
    land in the prior support: cheap and hang-proof, at the cost of being an estimate rather than
    an exact count.
    """
    with torch.no_grad():
        cond = x_o_norm if x_o_norm.dim() > 1 else x_o_norm.unsqueeze(0)
        probe = posterior.posterior_estimator.sample((n_probe,), condition=cond)
        probe = probe.reshape(-1, probe.shape[-1])
        return float(torch.isfinite(torch_prior.log_prob(probe)).float().mean().item())


def _robust_final_sample(posterior, inference, de_net, torch_prior, torch, x_o_norm, num_samples,
                          show_progress, num_chains, min_acceptance=1e-3, n_probe=4000, seed=0):
    """Sample the final (real-data-conditioned) posterior, falling back to MCMC if needed.

    ``DirectPosterior.sample()`` (the sbi default) draws via rejection sampling against the prior,
    which requires the flow's inverse transform. Two failure modes surface only at THIS call — never
    during training, which conditions on simulated placeholder ``x``, not the real ``x_o`` — because
    the real observation can land in a region the estimator extrapolates poorly: (1) a pathologically
    low acceptance rate (the estimator puts most conditional mass outside the prior box), which makes
    rejection sampling impractically slow rather than raise, and (2) a numerically degenerate spline
    coefficient in the flow's inverse (nflows' ``rational_quadratic_spline`` assertion), for
    flexible estimators like NSF. Both are avoided by ``MCMCPosterior``, whose potential function
    calls the estimator's raw (unnormalised) ``log_prob`` directly — no rejection loop at all — at the
    cost of being slower per sample.

    The acceptance-rate probe must NOT use sbi's own ``leakage_correction``/``log_prob(norm_posterior=
    True)``: both estimate the correction factor by *accept-reject sampling until N are accepted*, so
    at a pathological acceptance rate they hang exactly like ``.sample()`` (confirmed: it timed out
    identically). Instead, draw a FIXED ``n_probe`` sample directly from the density estimator (one
    unconditional forward pass, cost independent of the acceptance rate) and check what fraction land
    in the prior support — a cheap, hang-proof estimate. Healthy runs are unaffected: the probe is fast
    and the common path (rejection sampling) is untouched.
    """

    def _mcmc_fallback(reason):
        _warn_user(
            f"SNPE: final posterior sampling {reason}; falling back to MCMC-based sampling "
            "(slice_np_vectorized) which conditions via log_prob instead of the flow's inverse.",
           )
        return _mcmc_sample(inference, de_net, x_o_norm, num_samples, num_chains, seed, torch,
                            show_progress)

    try:
        rate = _probe_acceptance_rate(posterior, torch_prior, torch, x_o_norm, n_probe)
    except Exception:
        rate = 1.0
    method = "rejection"
    if rate < min_acceptance:
        result = _mcmc_fallback(f"has a pathologically low acceptance rate (~{rate:.2e})")
        method = "mcmc_fallback"
    else:
        try:
            result = posterior.sample((num_samples,), x=x_o_norm, show_progress_bars=show_progress)
        except AssertionError:
            result = _mcmc_fallback("hit a numerically-degenerate flow transform (AssertionError)")
            method = "mcmc_fallback"
    samples_np = np.asarray(result.detach().cpu().numpy(), dtype=float)
    return samples_np, method, rate


def _mcmc_sample(inference, de_net, x_o_norm, num_draws, num_chains, seed, torch, show_progress):
    """``num_draws`` from an ``MCMCPosterior`` over the same trained estimator, at ``x_o_norm``.

    Shared by the final draw and the between-round proposal draw so both escape hatches are the
    same object: slice sampling on the estimator's raw ``log_prob``, with no rejection loop and
    therefore no acceptance rate to stall on.

    ``num_chains`` is its own setting. It used to be ``min(20, max(4, num_workers))``, so the
    parallelism knob changed the algorithm: 4 chains at ``num_workers=1`` (which ``snpe_gpu`` needs,
    to avoid forking a JAX process), 20 at ``num_workers=30``.

    SEEDED HERE, BECAUSE NOTHING UPSTREAM REACHES IT. sbi's ``slice_np_vectorized`` steps its chains
    with numpy's GLOBAL generator (``SliceSamplerVectorized.rng = np.random``) and picks their
    starting points with torch. The numpy simulate path hid this -- sbi's ``simulate_for_sbi``
    re-seeds every backend each round -- but the torch path (``predict_torch``, i.e. ``snpe_gpu``)
    never touches numpy's generator, so four seed-0 GPU runs of SN2025pgp/arnett ended on max ln L
    -1798.5, -1748.9, -1819.2 and -1805.9. Both generators are seeded, as ``simulate_for_sbi``
    seeds them, so the draw depends on ``seed`` and the trained estimator alone.
    """
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    mcmc_posterior = inference.build_posterior(
        de_net, sample_with="mcmc", mcmc_method="slice_np_vectorized",
        mcmc_parameters={"num_chains": int(num_chains), "warmup_steps": 100})
    mcmc_posterior.set_default_x(x_o_norm)
    return mcmc_posterior.sample((int(num_draws),), x=x_o_norm, show_progress_bars=show_progress)


def _nudge_into_prior_density(theta, torch_prior, torch, max_tries=4):
    """Move draws that sit EXACTLY on a prior bound one ULP inside, and say how many were left.

    THE TWO TESTS SBI APPLIES TO THE SAME DRAW DO NOT AGREE AT THE BOUNDARY, and this is what that
    costs. ``DirectPosterior.sample`` accepts a draw when ``support.check`` passes; torch's
    ``interval(low, high)`` constraint is CLOSED, so ``theta == high`` passes. sbi's NPE-C atomic
    loss then evaluates ``prior.log_prob`` on the accepted draws, and
    ``torch.distributions.Uniform.log_prob`` is HALF-OPEN (``high.gt(value)``), so ``theta == high``
    is ``-inf`` — and sbi's own ``assert_all_finite(log_prob_prior, "prior eval")`` turns the whole
    round-2 training into ``ValueError: NaN/Inf present in prior eval``. Measured on the AT2017GFO
    two-component kilonova: exactly **1 draw in 1500** landed on ``mej_red == 0.1``, with
    ``within_support`` reporting 1500/1500 inside, and the fit died after round 1 had already been
    paid for. This is not a leakage failure, it is a boundary-convention failure, and it kills
    ``num_rounds>1`` at random.

    ``nextafter`` toward the batch's own centre is the minimal repair: it touches ONLY rows the
    prior density rejects, and moves them by one unit in the last place — a relative change of
    ~1e-7, far below any physical resolution of these parameters — into the half-open interval where
    the two tests agree. Rows still rejected after ``max_tries`` are genuinely outside the box
    (which neither sampling path should produce) and are reported to the caller rather than nudged
    further. Returns ``(theta, n_nudged, n_still_bad)``.
    """
    lp = torch_prior.log_prob(theta)
    bad = ~torch.isfinite(lp.reshape(-1))
    n_bad = int(bad.sum())
    if n_bad == 0:
        return theta, 0, 0
    theta = theta.clone()
    good = ~bad
    centre = (theta[good].median(dim=0).values if bool(good.any())
              else theta.mean(dim=0))
    for _ in range(int(max_tries)):
        idx = torch.nonzero(bad).flatten()
        theta[idx] = torch.nextafter(theta[idx], centre.to(theta.dtype).expand(idx.numel(), -1))
        bad = ~torch.isfinite(torch_prior.log_prob(theta).reshape(-1))
        if not bool(bad.any()):
            break
    return theta, n_bad, int(bad.sum())


def _robust_proposal_draw(proposal, num_draws, inference, de_net, torch_prior, torch, x_o_norm,
                          show_progress, num_chains, min_acceptance=1e-3, n_probe=4000, seed=0):
    """Draw the NEXT round's parameters, escaping to MCMC when the proposal leaks.

    THE SAME HAZARD AS THE FINAL DRAW, ON THE SAME OBJECT, AND IT USED TO BE UNGUARDED HERE.
    Rounds 2+ of SNPE-C propose from the round-1 posterior conditioned on the real observation, and
    ``DirectPosterior.sample()`` gets those draws by rejection sampling against the prior. When the
    estimator puts most of its conditional mass outside the prior box — which is exactly what a real
    observation the network extrapolates to produces — that loop does not raise, it simply almost
    never accepts: observed at 0 of 1500 draws after 42 s, with no end in sight. ``num_rounds>1``
    was therefore a coin flip on whether the fit returned at all.

    :func:`_robust_final_sample` already had the answer for the *final* draw; this applies it one
    call earlier. Probe the within-prior acceptance rate with the bounded, hang-proof probe (a fixed
    ``n_probe`` unconditional sample, cost independent of the rate), and below ``min_acceptance``
    take the draws from an ``MCMCPosterior`` over the same estimator instead — slower per sample,
    but it conditions through ``log_prob`` and has no rejection loop to stall in.

    ``proposal`` objects with no ``posterior_estimator`` (the prior itself, or a ``RestrictedPrior``
    in truncated mode) have no flow inverse and no reject-against-the-prior loop, so they are drawn
    from directly. Returns ``(theta, method, rate)``.
    """

    def _direct(n):
        try:
            return proposal.sample((int(n),), show_progress_bars=show_progress)
        except TypeError:                              # plain torch priors take no progress kwarg
            return proposal.sample((int(n),))

    def _finish(theta, method, rate):
        """Every path leaves through here, so the boundary repair cannot be forgotten on one."""
        theta, n_nudged, n_bad = _nudge_into_prior_density(theta, torch_prior, torch)
        if n_bad:
            _warn_user(
                f"SNPE: {n_bad} of {int(num_draws)} proposal draws are outside the prior's own "
                "density and could not be nudged inside. sbi's atomic loss asserts a finite prior "
                "log-probability, so training this round would raise; replacing those rows with "
                "prior draws.")
            replacement = torch_prior.sample((n_bad,)).to(theta.device).to(theta.dtype)
            still = ~torch.isfinite(torch_prior.log_prob(theta).reshape(-1))
            theta[torch.nonzero(still).flatten()] = replacement
        return theta, method, rate

    if getattr(proposal, "posterior_estimator", None) is None:
        return _finish(_direct(num_draws), "direct", None)

    try:
        rate = _probe_acceptance_rate(proposal, torch_prior, torch, x_o_norm, n_probe)
    except Exception:
        rate = 1.0

    reason = None
    if rate < min_acceptance:
        reason = f"has a pathologically low acceptance rate (~{rate:.2e})"
    else:
        try:
            return _finish(_direct(num_draws), "rejection", rate)
        except AssertionError:
            reason = "hit a numerically-degenerate flow transform (AssertionError)"

    _warn_user(
        f"SNPE: the round-to-round proposal {reason}; drawing this round's simulations from an "
        "MCMC-based posterior (slice_np_vectorized) instead of the flow's rejection sampler, which "
        "would stall rather than fail.")
    theta = _mcmc_sample(inference, de_net, x_o_norm, num_draws, num_chains, seed, torch,
                         show_progress)
    return _finish(theta, "mcmc_fallback", rate)


class _PredrawnProposal:
    """Proposal whose ``.sample`` hands back parameters that were ALREADY drawn.

    sbi's ``simulate_for_sbi`` insists on drawing from a proposal object itself, so the only way to
    give the CPU simulate path a leakage-guarded draw (see :func:`_robust_proposal_draw`) is to
    present that draw as a proposal. ``.sample`` therefore ignores nothing and invents nothing — it
    returns the stored tensor, and refuses a size it was not built for rather than silently
    recycling rows.
    """

    def __init__(self, theta, to_cpu=False):
        self._theta = theta.detach().cpu() if to_cpu else theta.detach()

    def sample(self, sample_shape, **kwargs):
        n = int(sample_shape[0]) if len(tuple(sample_shape)) else 1
        if n != int(self._theta.shape[0]):
            raise ValueError(
                f"_PredrawnProposal holds {int(self._theta.shape[0])} parameter rows but was asked "
                f"for {n}. The guarded proposal draw and the simulation count must agree.")
        return self._theta


class SNPESampler(BaseSampler):
    """Sequential Neural Posterior Estimation (sbi ``NPE``). See the module docstring."""

    name = "snpe"

    def fit(self, lc, model, prior=None, *, num_rounds=2, num_simulations=1000, space="auto",
            density_estimator="maf", embedding_net=None, embedding_latent=32, x_format="value",
            predict_torch=None, scatter_param=None, hidden_features=None, num_transforms=None,
            num_bins=None, proposal_mode="posterior", truncate_quantile=1e-4,
            support_samples=10000, num_samples=10000, device="cpu", seed=0, show_progress=False,
            num_workers=1, max_logl_scan=2000, scan_timeout=300, standardize_x=True,
            proposal_min_acceptance=1e-3, num_chains=4, **train_kwargs):
        """Fit ``lc`` with ``model`` via SNPE/NPE.

        ``num_rounds=1`` is amortized NPE; ``num_rounds>1`` focuses simulations sequentially.
        ``num_simulations`` is *per round*; ``num_workers`` parallelizes simulation. ``space``
        ('auto'|'flux'|'magnitude') sets the data space for the simulator and the Gaussian noise model
        (errors required).

        **Priors:** ``Uniform``, ``LogUniform``, ``Normal`` and ``TruncatedNormal`` (exact density
        and draws; see :func:`_to_torch_prior`). A ``Fixed`` parameter is held at its value: it is
        not a dimension of the torch prior nor of the network, not counted in AIC/BIC
        (``n_params``), a constant column in ``samples``, and listed in ``info["fixed"]``. A
        ``predict_torch`` receives the free columns only, in the prior's order.

        **Input layout:** ``x_format="value"`` (default) conditions on the data-space values alone;
        ``x_format="stacked"`` appends two constant context channels — ``(value, error, time)`` per
        point — giving an embedding net the cadence/noise structure. The **band is NOT** a channel:
        every simulation is drawn on the same ``(time, band)`` grid as the observation, so each
        position's band is identical for the data and every simulation (encoded by position); a
        constant channel adds nothing for a single-object fit.

        **Input normalisation:** ``standardize_x`` (default ``True`` = ``"asinh"``; also ``"zscore"`` or
        ``"none"``/``False``) normalises the conditioning input before it reaches the estimator, with
        per-channel statistics fitted on the first round's simulations. ``"asinh"`` (``asinh(x/scale)``)
        VARIANCE-STABILISES a wide dynamic range — essential for **flux-space** data whose values span
        ~6 orders of magnitude (sbi z-scores the scale but not the skew), making flux inputs as
        well-conditioned as magnitude. The **same** transform is applied to the simulations, the
        observation ``x_o`` used to estimate the posterior, and ``result.format_x``.

        **Robust best-fit scan:** the post-fit max-likelihood scan (re-running the forward model on the
        posterior draws) runs in a process pool (``num_workers``) with a wall-clock cap ``scan_timeout``
        [s], so an expensive or occasionally-pathological model call cannot hang the fit.

        **Density estimator (flexible):** ``density_estimator`` is an sbi estimator name ('maf', 'nsf',
        'mdn', ...) **or** a pre-built ``posterior_nn(...)`` factory. ``embedding_net`` is ``None``
        (condition on the raw vector), a built-in name — ``"mlp"`` or ``"tcn"`` (Temporal Convolutional
        Network; see :mod:`whisper_cbpf.embeddings`), compressed to ``embedding_latent`` features —
        or any ``torch.nn.Module``. ``hidden_features`` / ``num_transforms`` / ``num_bins`` build a
        custom architecture via ``sbi``'s ``posterior_nn``.

        **GPU simulation:** pass ``predict_torch(theta, times)`` — or
        ``predict_torch(theta, times, bands)`` for a **photometric** model, which needs to know which
        filter each observation is in — a batched, device-agnostic torch implementation of the model
        (``(B, D)`` parameters + ``(n,)`` times → ``(B, n)`` FLUX) to replace the per-row Python
        simulator with a single on-device batched call. The three-argument form is detected from the
        signature (:func:`_predict_torch_accepts_bands`), so two-argument callables keep working.
        Whichever form it takes, the returned flux is mapped into the comparison space and only then
        given per-point white noise ``N(0, err)``, on-device — so **magnitude-space data is fully
        supported** (tested in ``tests/test_snpe.py``); the claim that this path was flux-only was
        never true after the space map landed. ``sampler="snpe_gpu"`` builds this callable for you
        from any model registered through the JAX factories.

        **Multi-round leakage guard:** ``proposal_min_acceptance`` (default ``1e-3``) is the
        within-prior acceptance rate below which the ROUND-TO-ROUND proposal draw, and the final
        draw, stop using the flow's rejection sampler and take their draws from an
        ``MCMCPosterior`` over the same estimator instead. Without it a ``num_rounds>1`` fit can
        stall indefinitely in a rejection loop that never raises (see
        :func:`_robust_proposal_draw`). That MCMC runs ``num_chains`` slice-sampling chains
        (default 4), seeded from ``seed``, so a fit that falls back is as reproducible as one that
        does not.

        **Reporting:** ``info["leakage"]`` is ``True`` when the final flow put so little mass inside
        the prior box that the final draw had to fall back (acceptance below
        ``proposal_min_acceptance``), and ``info["converged"]`` is ``True`` only when the final
        draw needed no fallback at all. Of 45 test fits with an emcee reference, the
        38 that fell back sat a median 1.61 emcee sigma from emcee (median worst parameter 19.8),
        the 7 that did not 1.00 (2.31). ``info["x_o_min_rms_z"]`` is the RMS distance, in units of
        the data errors, from the observation to the closest round-0 (prior-predictive)
        simulation: a large value says the prior cannot produce the data before any training is
        spent. ``runtime_s`` covers simulation and training; ``info["postprocess_s"]`` is the final
        draw plus the max-likelihood scan, which the fallback can make several times longer.

        **Free extra scatter (Villar+2017):** ``scatter_param`` names a prior parameter that is a
        LIKELIHOOD scatter term, not a model input: each simulation's noise becomes
        ``N(0, sqrt(err² + scatter²))`` with that draw's value, so the density estimator learns the
        scatter posterior from its imprint on the simulations — the SBI counterpart of
        :class:`~whisper_cbpf.likelihood.GaussianLikelihoodWithScatter`, which is also used for the
        exact ``max_log_likelihood``/AIC/BIC at the best draw.

        **Sequential scheme:** ``proposal_mode='posterior'`` (default) is SNPE-C (propose from the latest
        posterior). ``proposal_mode='restricted'`` is **truncated SNPE** — each round restricts the prior
        to the high-density region with ``get_density_thresholder(quantile=truncate_quantile)`` +
        ``RestrictedPrior`` (often more robust); the support is estimated from ``support_samples`` draws
        (kept modest; sbi's default of 1e6 can take hours). Extra keyword args pass through to ``NPE.train``
        (e.g. ``max_num_epochs``, ``training_batch_size``, ``stop_after_epochs``). The trained sbi
        posterior is attached as ``result.posterior`` (per round in ``result.posteriors``).

        **Device:** ``device`` selects where the neural density estimator trains — ``'cpu'`` (default),
        ``'cuda'`` / ``'gpu'`` / ``'cuda:N'``, or ``'auto'`` (CUDA when available, else CPU). The torch
        prior and observed data are placed on the device automatically; requesting a GPU without one
        warns and falls back to CPU. **Without** ``predict_torch`` the GPU accelerates *training*
        only — the simulator stays a per-row Python loop over the model's numpy ``predict`` — so it
        helps most with many simulations / large networks. **With** it (which is what
        ``sampler="snpe_gpu"`` supplies) the simulator is on-device too and the whole round is GPU
        resident.
        """
        if proposal_mode not in ("posterior", "restricted"):
            raise ValueError(f"proposal_mode must be 'posterior' or 'restricted'; got {proposal_mode!r}.")
        if x_format not in ("value", "stacked"):
            raise ValueError(f"x_format must be 'value' or 'stacked'; got {x_format!r}.")
        sb = _require_sbi()
        torch = sb.torch
        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")

        # Reuse the Gaussian likelihood for the data space, observation, errors, and exact metrics.
        from ..likelihood import GaussianLikelihood
        lik = GaussianLikelihood(lc, space=space)

        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        # A Fixed parameter is held at its value: it is not a dimension of the torch prior nor of
        # the network's output, and it is not counted in AIC/BIC. The simulator and the best-fit
        # scan see it through `_HoldFixed`; `predict_torch` receives the free columns only.
        all_names = list(prior.names)
        prior, param_names, fixed = _dg.split_fixed(prior, all_names, "snpe")
        predict = _HoldFixed(model.predict, fixed) if fixed else model.predict
        k, n = len(param_names), int(len(times))
        scatter_idx = None
        if scatter_param is not None:
            if scatter_param not in param_names:
                raise ValueError(f"scatter_param {scatter_param!r} is not in the prior "
                                 f"({param_names}).")
            scatter_idx = param_names.index(scatter_param)

        # Constant context channels for the stacked layout: per-point error + time. The BAND is NOT a
        # channel: every simulation is drawn on the SAME (time, band) grid as the observation, so the
        # band at each position is identical for X_input and every X_sim — its identity is encoded by
        # position, and a constant channel carries no information for a single-object fit.
        sig_clean = np.where(np.isfinite(np.asarray(lik.sigma, float)) & (np.asarray(lik.sigma) > 0),
                             np.asarray(lik.sigma, float), 1.0)
        if x_format == "stacked":
            extra = np.concatenate([sig_clean, times])
            n_channels = 3
        else:
            extra, n_channels = None, 1

        torch.manual_seed(int(seed))                       # BEFORE building nets: embedding init draws
        emb_spec = (embedding_net if isinstance(embedding_net, str)
                    else type(embedding_net).__name__ if embedding_net is not None else None)
        if isinstance(embedding_net, str):
            from ..embeddings import build_embedding
            embedding_net = build_embedding(embedding_net, n_points=n, n_channels=n_channels,
                                            latent_dim=int(embedding_latent))

        device = _resolve_device(device, torch)            # 'auto'/'gpu' -> 'cuda'/'cpu', with fallback
        torch_prior = _to_torch_prior(prior, sb, device)   # prior tensors must match the training device
        torch_prior, _, prior_returns_numpy = sb.process_prior(torch_prior)
        if predict_torch is None:
            simulator = _build_simulator(
                predict, param_names, times, bands, lik.model_in_space, lik.sigma, torch, seed, extra,
                scatter_idx)
            simulator = sb.process_simulator(simulator, torch_prior, prior_returns_numpy)
            sb.check_sbi_inputs(simulator, torch_prior)
            times_t = sig_t = extra_t = None
        else:
            times_t = torch.as_tensor(times, dtype=torch.float32, device=device)
            sig_t = torch.as_tensor(sig_clean, dtype=torch.float32, device=device)
            extra_t = (None if extra is None
                       else torch.as_tensor(extra, dtype=torch.float32, device=device))

        x_np = np.asarray(lik.y, dtype=float)
        if extra is not None:
            x_np = np.concatenate([x_np, extra])
        x_o = torch.as_tensor(x_np, dtype=torch.float32).to(device)

        # Input normalisation: a per-channel asinh transform that VARIANCE-STABILISES the conditioning
        # input before it reaches the network. sbi already z-scores x (fixing the scale), but not the
        # SKEW: flux-space values span ~6 orders of magnitude (~1e-6..1e-3 Jy), so the estimator trains
        # poorly. asinh(x / scale) compresses that dynamic range (≈ linear for |x|≲scale, ≈ log beyond)
        # while handling the small negatives that noise produces — making flux inputs as well-conditioned
        # as magnitude. The per-channel scale is set from the first round's simulations.
        # Input normalisation method. asinh(x/scale) variance-stabilises (best for wide-dynamic-range
        # flux); zscore standardises; none disables. Stats (on CPU) are set from round-0 sims and
        # applied to the sims, the observation x_o, AND result.format_x — consistently.
        _norm_method = ("asinh" if standardize_x is True
                        else "none" if standardize_x in (False, None)
                        else str(standardize_x).lower())
        if _norm_method not in ("asinh", "zscore", "none"):
            raise ValueError("standardize_x must be True/False or 'asinh'/'zscore'/'none'; "
                             f"got {standardize_x!r}")
        _norm_stats = {}

        def _fit_norm(x0):
            xc = x0.detach().to("cpu")
            if _norm_method == "asinh":
                _norm_stats["scale"] = xc.abs().median(dim=0).values.clamp_min(1e-30)
            elif _norm_method == "zscore":
                _norm_stats["mean"] = xc.mean(dim=0)
                _norm_stats["std"] = xc.std(dim=0).clamp_min(1e-30)

        def _norm(t):
            if _norm_method == "asinh" and "scale" in _norm_stats:
                return torch.asinh(t / _norm_stats["scale"].to(t.device))
            if _norm_method == "zscore" and "mean" in _norm_stats:
                return (t - _norm_stats["mean"].to(t.device)) / _norm_stats["std"].to(t.device)
            return t

        de_builder = _build_density_estimator(
            density_estimator, embedding_net, hidden_features, num_transforms, num_bins)
        inference = sb.NPE(prior=torch_prior, density_estimator=de_builder,
                           device=device, show_progress_bars=show_progress)
        restricted = proposal_mode == "restricted"

        t0 = time.perf_counter()
        proposal = torch_prior
        posteriors = []
        de_net = None
        proposal_draw_methods, proposal_draw_rates = [], []
        for r in range(int(num_rounds)):
            # LEAKAGE GUARD ON THE PROPOSAL DRAW. From round 1 on, `proposal` is a DirectPosterior
            # conditioned on the REAL observation, and drawing from it is rejection sampling against
            # the prior -- the same loop `_robust_final_sample` guards at the end, one call earlier,
            # and until now unguarded (observed: 0 of 1500 accepted after 42 s, no error, no end).
            # Draw it here, guarded, and hand the result to whichever simulate path runs.
            # Seeds: seed + r for the draw before round r, seed + num_rounds for the final draw.
            theta_pre = None
            if r > 0:
                theta_pre, _m, _rate = _robust_proposal_draw(
                    proposal, int(num_simulations), inference, de_net, torch_prior, torch,
                    _norm(x_o), show_progress, int(num_chains),
                    min_acceptance=float(proposal_min_acceptance), seed=int(seed) + r)
                proposal_draw_methods.append(_m)
                proposal_draw_rates.append(_rate)
            if predict_torch is not None:
                theta, x = _torch_simulate(proposal, num_simulations, predict_torch, times_t, sig_t,
                                           extra_t, torch, int(seed) + 7919 * r, show_progress,
                                           scatter_idx,
                                           _torch_model_in_space(lik.space, lik.zeropoint_jy, torch),
                                           bands, theta_pre)
            else:
                # Parallel simulation needs CPU theta (sbi calls theta.numpy() when chunking).
                if theta_pre is not None:
                    sim_proposal = _PredrawnProposal(theta_pre, to_cpu=int(num_workers) > 1)
                else:
                    sim_proposal = _CPUSampleProxy(proposal) if int(num_workers) > 1 else proposal
                theta, x = sb.simulate_for_sbi(
                    simulator, sim_proposal, num_simulations=int(num_simulations),
                    num_workers=num_workers, seed=int(seed) + r, show_progress_bar=show_progress)
            x = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
            if r == 0:
                # Out-of-distribution check, free before any training: RMS distance in data-error
                # units from x_o to the closest prior-predictive simulation (value columns only;
                # nanmin, because a failed simulation is a NaN row that training drops anyway).
                z = (x[:, :n].detach().cpu().double().numpy() - x_np[:n]) / sig_clean
                x_o_min_rms_z = float(np.nanmin(np.sqrt(np.mean(z ** 2, axis=1))))
            if _norm_method != "none" and not _norm_stats:   # fit the normaliser on round-0 sims
                _fit_norm(x)
            x = _norm(x)                                # normalised conditioning input
            if restricted:
                # Truncated SNPE: the proposal is a RestrictedPrior (not a posterior) -> first-round loss.
                de_net = inference.append_simulations(theta, x, exclude_invalid_x=True).train(
                    force_first_round_loss=True, show_train_summary=False, **train_kwargs)
            else:
                de_net = inference.append_simulations(
                    theta, x, proposal=proposal, exclude_invalid_x=True).train(
                    show_train_summary=False, **train_kwargs)
            posterior = inference.build_posterior(de_net)
            posterior.set_default_x(_norm(x_o))               # normalised obs; sample() needs no x
            # Pre-cache the leakage/normalising-constant factor with the bounded probe (see
            # _probe_acceptance_rate) so any internal `log_prob(norm_posterior=True)` call this round
            # or the next (e.g. inside `get_density_thresholder` below) reuses it instead of
            # re-deriving it via sbi's own accept-reject loop, which hangs at low acceptance.
            try:
                probed_rate = _probe_acceptance_rate(posterior, torch_prior, torch, _norm(x_o))
                posterior._leakage_density_correction_factor = torch.tensor(max(probed_rate, 1e-6))
            except Exception:
                pass
            posteriors.append(posterior)
            if r < int(num_rounds) - 1:                      # update proposal for the next round
                if restricted:
                    # num_samples_to_estimate_support defaults to 1e6 in sbi (hours of sampling);
                    # cap it to keep truncated SNPE practical.
                    accept_reject_fn = sb.get_density_thresholder(
                        posterior, quantile=float(truncate_quantile),
                        num_samples_to_estimate_support=int(support_samples))
                    # device= is REQUIRED: sbi 0.23's RestrictedPrior defaults to CPU and moves its
                    # samples there, which crashes the on-device predict_torch simulator (cuda vs cpu).
                    proposal = sb.RestrictedPrior(
                        torch_prior, accept_reject_fn, sample_with="rejection", device=device)
                else:
                    proposal = posterior
        runtime = time.perf_counter() - t0

        t_post = time.perf_counter()
        samples_np, final_sample_method, final_sample_acceptance = _robust_final_sample(
            posterior, inference, de_net, torch_prior, torch, _norm(x_o), int(num_samples),
            show_progress, int(num_chains), min_acceptance=float(proposal_min_acceptance),
            seed=int(seed) + int(num_rounds))
        samples = pd.DataFrame(_dg.fill_fixed(samples_np, param_names, fixed, all_names),
                               columns=all_names)

        # Exact Gaussian metrics at the best (max-likelihood) posterior draw — scatter-augmented
        # (with each draw's own scatter value) when a scatter parameter is fitted. The scan re-runs the
        # forward model per draw; it is done IN PARALLEL with a wall-clock cap so an expensive model
        # (e.g. redback ~0.2 s/call) is fast and a single pathological draw cannot hang the fit.
        if scatter_param is not None:
            from ..likelihood import GaussianLikelihoodWithScatter
            lik_scan = GaussianLikelihoodWithScatter(lc, space=space, scatter_param=scatter_param)
        else:
            lik_scan = lik
        scan = samples_np if len(samples_np) <= max_logl_scan else samples_np[:max_logl_scan]
        args = [(predict, param_names, times, bands, lik_scan, scatter_idx, scan[i])
                for i in range(len(scan))]
        n_jobs = max(1, int(num_workers))
        logls, _ = _parallel_max_logl(args, n_jobs, scan_timeout)
        if logls is None:                           # parallel scan hit the timeout (a draw hung)
            _warn_user(f"SNPE: max-likelihood scan exceeded scan_timeout={scan_timeout}s (a "
                          "posterior draw stalled the forward model); scoring a small subset instead. "
                          "This usually means a poorly-conditioned posterior — prefer magnitude space "
                          "for neural SBI on flux data.")
            logls, _ = _parallel_max_logl(args[:min(len(args), 4 * n_jobs)], n_jobs, scan_timeout)
            if logls is None:
                logls = np.array([float("-inf")])
        if np.any(np.isfinite(logls)):
            best_idx = int(np.nanargmax(logls))
            max_log_likelihood = float(logls[best_idx])
        else:                                       # every scanned draw gave a non-finite likelihood
            _warn_user("SNPE: all scanned posterior draws have non-finite log-likelihood; "
                          "AIC/BIC are -inf and best_params is the first draw.")
            best_idx, max_log_likelihood = 0, float("-inf")
        best_params = {nm: float(scan[best_idx][j]) for j, nm in enumerate(param_names)}
        best_params = {nm: best_params[nm] if nm in best_params else float(fixed[nm])
                       for nm in all_names}
        postprocess_s = time.perf_counter() - t_post

        info = {
            "num_rounds": int(num_rounds), "num_simulations": int(num_simulations),
            "total_simulations": int(num_rounds) * int(num_simulations),
            "density_estimator": density_estimator if isinstance(density_estimator, str) else "custom",
            "embedding_net": emb_spec,
            "x_format": x_format,
            "normalize_x": _norm_method,
            "final_sample_method": final_sample_method,
            "final_sample_acceptance_rate": final_sample_acceptance,
            "leakage": bool(final_sample_acceptance < float(proposal_min_acceptance)),
            "converged": final_sample_method == "rejection",
            "x_o_min_rms_z": x_o_min_rms_z,
            "proposal_draw_methods": list(proposal_draw_methods),
            "proposal_draw_acceptance_rates": list(proposal_draw_rates),
            "num_chains": int(num_chains),
            "postprocess_s": postprocess_s,
            "sim_backend": "torch" if predict_torch is not None else "numpy",
            "scatter_param": scatter_param,
            "proposal_mode": proposal_mode,
            "truncate_quantile": float(truncate_quantile) if proposal_mode == "restricted" else None,
            "space": lik.space, "num_samples": int(num_samples), "device": str(device),
            "seed": int(seed), "num_workers": int(num_workers),
            "fixed": dict(fixed),
        }
        attach_band_metrics(info, lc, model, best_params, space)
        aic, bic = aic_bic(max_log_likelihood, k, n)
        result = SamplerResult(
            sampler="snpe", model=model.name, parameters=list(all_names), samples=samples,
            summary=summarize_posterior(samples, all_names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=aic, bic=bic,
        )
        attach_predictive_metrics(result, lc, space, model=model)
        # Attach the trained sbi objects for resampling / pairplot (not part of to_json).
        result.posterior = posterior
        result.posteriors = posteriors

        def format_x(values):
            """Map a raw data-space vector (n,) to the network's conditioning input on ``device``,
            appending THIS fit's context channels (errors/times/bands) when ``x_format='stacked'``.
            Use it to condition the amortized posterior on a new observation **taken on the same
            observing grid** (same cadence, errors and bands — e.g. simulation-based calibration
            realizations). An observation with a different grid needs a re-trained network — do not
            feed it through this closure, and note the input length is checked only by the network."""
            v = np.asarray(values, dtype=float).ravel()
            if extra is not None:
                v = np.concatenate([v, extra])
            return _norm(torch.as_tensor(v, dtype=torch.float32).to(device))   # same normalisation

        result.format_x = format_x
        return result


def fit_SNPE(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via Sequential Neural Posterior Estimation.

    See :meth:`SNPESampler.fit` for options. Requires the optional ``[sbi]`` extra (sbi + torch).

    Parameters
    ----------
    lc : LightCurve
        The data (detections only: SNPE cannot use an upper limit).
    model : str or Model, default "flare"
    prior : Prior, optional
        Default: the model's.
    **kwargs
        The sampler's settings (``num_rounds``, ``num_simulations``, ``num_samples``, ``seed``,
        ...).

    Returns
    -------
    SamplerResult

    Examples
    --------
    A ``Normal`` prior on the amplitude and a ``Fixed`` rise time (held outside the network):

    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=wp.get_model("flare").predict(truth, t),
    ...                    flux_err=np.full(30, 0.2))
    >>> prior = wp.Prior({"amplitude": wp.Normal(5.0, 1.0), "rise_time": wp.Fixed(3.0),
    ...                   "decay_time": wp.Uniform(1.0, 50.0)})
    >>> res = wp.fit_SNPE(lc, "flare", prior=prior, num_rounds=1, num_simulations=500,
    ...                   max_num_epochs=20, num_samples=500, seed=0)
    >>> res.n_params, res.info["fixed"], list(res.samples.columns)
    (2, {'rise_time': 3.0}, ['amplitude', 'rise_time', 'decay_time'])
    """
    return SNPESampler().fit(lc, model, prior=prior, **kwargs)
