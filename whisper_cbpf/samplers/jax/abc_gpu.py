"""Rejection ABC on the GPU -- a JAX port of ``whisper_cbpf.samplers.abc``.

ABC is the case where a GPU forward model pays off most. It has no network to train and no chain to
advance: the entire cost is *simulate, measure distance, keep the closest*. Measured on the same
AT2017GFO problem, the JAX kilonova simulator ran ~33x faster per simulation than redback's; SNPE
converted that into only 2.05x end-to-end because neural training then dominated. ABC has no such
Amdahl term, so the end-to-end speedup should land far closer to the raw simulator ratio.

Semantics are matched to the CPU sampler deliberately and in detail, because a posterior that
differs for an unnoticed reason is worse than no port at all:

* **The simulator is generative.** ``x = model_in_space(predict(theta)) + N(0, sigma)``, with the
  noise added *after* the flux->space map, per point, using that draw's own scatter in quadrature
  when ``scatter_param`` is set. That is what makes ABC exact as epsilon -> 0 and keeps the
  posterior width calibrated rather than merely peaked.
* **The distance denominator is the RAW observational error**, even when a scatter parameter is
  fitted. The scatter enters the simulated noise only. Putting it in the denominator too would
  double-count it and shrink the recovered scatter.
* **Acceptance is inclusive** (``d <= epsilon``), matching ``abc.py``; ABC-SMC uses a strict ``<``.
* **``best_params`` is the argmax of the exact Gaussian log-likelihood over the accepted draws, not
  the argmin-distance draw.** With ``simulate_noise=True`` the closest draw is the luckiest noise
  realisation, not the best parameter vector.
* **No NaN scrubbing.** ``abc.py`` lets non-finite values propagate into the distance (whisper's
  SNPE path does scrub them, and that difference is intentional). A NaN distance therefore poisons
  the quantile, exactly as on CPU -- which is a signal worth seeing, not hiding. (``max_abs_z``
  maps a NaN point to ``inf`` by its own definition, on both backends.)

The GPU-specific parts are: prior draws by inverse CDF on ``jax.random.uniform`` so sampling
vectorises (the boxes analytically, ``Normal`` / ``TruncatedNormal`` through
:func:`whisper_cbpf.priors.ppf_jax`, ``Fixed`` at its value; :class:`_PriorDraw`), one PRNG key
folded per simulation index (matching the CPU's per-index streams), and the simulate/distance step
vmapped over particles and evaluated in chunks.

**One model evaluation per draw.** The sweep returns, for every draw, its distance AND its exact
log-likelihood, both from the one forward-model call (the likelihood through
:func:`whisper_cbpf.likelihood._jax.log_likelihood_jax`, the JAX twin of the numpy likelihood the
CPU sampler scores with). The best fit is then a lookup, not a second pass: 0.1.1 re-ran the model
over the accepted draws in a second compiled scan, which cost a second XLA compile of the model.
The parameter vectors and log-likelihoods stay on the device; only the distances (for the
acceptance quantile, computed on the host in float64 as before) and the ACCEPTED rows are
transferred.

**Precision.** ``precision="float32"`` runs the sweep in single precision whatever the session's
setting, for the models that are accurate in it (the kilonovae, the flare). The supernova and TDE
engines are float64-only and are refused by name.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

# Imported relatively -- the CPU core and this sampler now live in one package. In the legacy
# split (whisper-GPU @ 10796a0, discontinued; superseded by whisper_cbpf) this file sat in a
# separate repo cloned INTO a checkout of the CPU package, and reached that package absolutely
# off sys.path; the merge removes that indirection.
from ...likelihood import GaussianLikelihood, make_likelihood
from ...likelihood._jax import log_likelihood_jax
from ...models import get_model
from ...samplers.base import (
    aic_bic,
    BaseSampler,
    SamplerResult,
    attach_band_metrics,
    attach_predictive_metrics,
    summarize_posterior,
    _warn_user,
)

from ...backends._env import require_jax
from ...distance._jax import jnp_distance

#: Particles simulated per compiled call.
#:
#: Sized by COMPILE TIME, not by memory and not by throughput. The distance reduction is only
#: (B, n_points) -> (B,) and would tolerate thousands, but a photometric model builds a
#: (B, n_comp, n_obs, n_wave) spectrum on the way, and XLA's compile time for that fused reduction
#: grows far faster than its throughput improves. Measured on the two-component kilonova
#: (n_obs=126, n_wave=300, float64, A6000), against redback's 16.7 ms/sim:
#:
#:     B      per-sim      speedup    compile
#:     1     0.3197 ms       52x       0.9 s
#:    16     0.1251 ms      134x       4.0 s
#:    64     0.1162 ms      144x      59.3 s
#:   256          --          --      did not finish in 6 minutes
#:
#: Throughput saturates by B=16; past it you buy 7% more speed for 15x the compile. Solving for
#: where B=64 overtakes B=16 gives ~6.2 MILLION simulations -- far beyond any realistic ABC budget,
#: and at a typical n_sim=256000 the choice is 36 s versus 89 s end to end. Hence 16, until 250.
#:
#: RE-MEASURED (2026-09-26, current models, one A6000, float64, a fresh process per width). The rule that small widths are safe and large ones
#: slow does not hold: XLA's GPU compile time is erratic in the width, not monotone. The kilonovae
#: compile in 1.5-3.5 s at 16, 30, 60, 100, 250, 500 and 1000, but in 6-13 s at 128, 21-72 s at
#: 256 and more than 300 s at 1024 (killed); the supernovae and the TDE show no such spike any
#: more. And 16 is slow wherever the model is latency-bound: the TDE's ODE costs ~20 ms per call
#: at ``n_time=5000`` whatever the width, so 16 wide it scores 1.6 ms a simulation against 0.10 ms
#: at 250. Cold ``abc_gpu`` fits of 20000 simulations, chunk 16 against 250: kilonovae 9.1-10.8 s
#: against 8.5-8.8 s, the TDE 26.0 against 22.6 s (``n_time=500``) and 66.7 against 54.7 s
#: (5000), an arnett supernova 5.4 against 7.1 s (a slower compile); the gap grows with the
#: simulation count. Hence 250.
#:
#: Since the sweep became a single on-device ``lax.scan``, this is purely a compile-versus-
#: occupancy trade: there is no per-block host synchronisation left to amortise. Avoid powers of
#: two from 128 up on the kilonovae; lower it only if you OOM, which is unlikely to be the binding
#: limit. ``abc_smc_gpu`` uses the same default.
DEFAULT_CHUNK = 250

_MIN_FLUX_JY = 1e-300          # matches whisper_cbpf.likelihood._MIN_FLUX_JY
from ...io.photometry import AB_ZEROPOINT_JY


def _model_in_space_jnp(jnp, flux, space, zeropoint_jy):
    """jnp counterpart of ``GaussianLikelihood.model_in_space``.

    Identity in flux space; in magnitude space ``-2.5 log10(clip(F, floor) / zp)``. The clip is
    load-bearing: it maps a non-positive model flux to a finite magnitude (hence a huge chi2)
    instead of an ``inf`` that would silently poison the acceptance quantile.

    The floor is the working dtype's smallest normal whenever ``_MIN_FLUX_JY`` is not
    representable in it, exactly as :func:`whisper_cbpf.likelihood._jax.log_likelihood_jax` does.
    This module used a bare ``1e-300``, which is **exactly 0.0 in float32** and therefore clamped
    nothing at all: measured, ``_model_in_space_jnp(0.0, "magnitude")`` returned ``inf``, not the
    ~759 mag the docstring promised. ``abc_gpu`` never enables x64, so float32 is its default case,
    not its edge case.
    """
    if space == "flux":
        return flux
    mf = jnp.asarray(flux)
    tiny = jnp.finfo(mf.dtype).tiny
    floor = _MIN_FLUX_JY if _MIN_FLUX_JY > tiny else tiny
    return -2.5 * jnp.log10(jnp.maximum(mf, floor) / zeropoint_jy)


#: The prior families the GPU ABC samplers draw on the device.
DRAWABLE_PRIORS = ("Uniform", "LogUniform", "Normal", "TruncatedNormal", "Fixed")


class _PriorDraw:
    """The prior as a traceable map ``u -> theta``: each parameter's inverse CDF applied to its own
    ``U(0, 1)`` variate (the probability integral transform), so a vmapped sweep draws the prior
    exactly on the device.

    * ``Uniform`` / ``LogUniform``: ``lo + u (hi - lo)``, in log space for a LogUniform, the
      arithmetic 0.1.1 used (so the same seed gives the same draws);
    * ``Normal`` / ``TruncatedNormal``: :func:`whisper_cbpf.priors.ppf_jax`, with ``u`` kept in
      ``[eps, 1 - epsneg]`` (``jax.random.uniform`` can return exactly 0, whose Normal quantile is
      ``-inf``; ``eps`` is the smallest nonzero value it returns);
    * ``Fixed``: the value (its ``u`` is drawn and unused, so the other parameters' draws do not
      depend on which ones are fixed).

    ``nat_lo`` / ``nat_hi`` are the native bounds ABC-SMC's support test uses, ``free`` the indices
    of the parameters that are not Fixed, ``fixed`` ``{name: value}``.
    """

    def __init__(self, prior, names, sampler):
        from ...priors._numpy import family

        self.names = list(names)
        kinds, self.dists = [], []
        for nm in self.names:
            d = prior.distributions[nm]
            kind = family(d)
            if kind not in DRAWABLE_PRIORS:
                raise TypeError(
                    f"{sampler} draws the prior on the device through each distribution's inverse "
                    f"CDF and supports {', '.join(DRAWABLE_PRIORS)}; parameter {nm!r} is {kind}. "
                    f"Use the CPU sampler (`abc` / `abc_smc`), which draws any distribution with "
                    f"sample(rng), or express this prior in a supported family.")
            kinds.append(kind)
            self.dists.append(d)
        self.kinds = kinds
        box = [k in ("Uniform", "LogUniform") for k in kinds]
        self.is_log = np.array([k == "LogUniform" for k in kinds])
        # rescale space for the boxes (log for LogUniform); placeholders 0 / 1 elsewhere
        self.lows = np.array([(np.log(d.bounds[0]) if k == "LogUniform" else d.bounds[0])
                              if b else 0.0 for d, k, b in zip(self.dists, kinds, box)], dtype=float)
        self.highs = np.array([(np.log(d.bounds[1]) if k == "LogUniform" else d.bounds[1])
                               if b else 1.0 for d, k, b in zip(self.dists, kinds, box)], dtype=float)
        self.nat_lo = np.array([float(d.bounds[0]) for d in self.dists])
        self.nat_hi = np.array([float(d.bounds[1]) for d in self.dists])
        self.fixed = {nm: float(d.value) for nm, d, k in zip(self.names, self.dists, kinds)
                      if k == "Fixed"}
        self.free = np.array([i for i, k in enumerate(kinds) if k != "Fixed"], dtype=int)
        if not len(self.free):
            raise ValueError(f"{sampler}: every parameter is Fixed ({self.fixed}), so there is "
                             f"nothing to sample. Evaluate the model at those values directly.")

    def build(self, jnp):
        """``draw(u) -> theta``, traceable: ``u`` and ``theta`` are ``(D,)`` in ``names`` order."""
        from ...priors._jax import ppf_jax

        lo_j, hi_j, log_j = jnp.asarray(self.lows), jnp.asarray(self.highs), jnp.asarray(self.is_log)
        special = []
        for i, (kind, d) in enumerate(zip(self.kinds, self.dists)):
            if kind in ("Normal", "TruncatedNormal"):
                special.append((i, ppf_jax(d), None))
            elif kind == "Fixed":
                special.append((i, None, float(d.value)))

        def draw(u):
            raw = lo_j + u * (hi_j - lo_j)                  # inverse CDF in the parameter's space
            theta = jnp.where(log_j, jnp.exp(raw), raw)     # LogUniform: uniform in log, then exp
            if special:
                # u = 0 happens; its quantile is -inf. eps is the smallest nonzero value
                # jax.random.uniform returns, so the clip moves only that one lattice point, and
                # it keeps p * Z (Z the truncation mass) a normal number: `tiny` did not, and XLA
                # flushes the subnormal product to 0, whose quantile is -inf again.
                bottom = jnp.finfo(u.dtype).eps
                top = 1.0 - jnp.finfo(u.dtype).epsneg
                for i, ppf, value in special:
                    x = (jnp.asarray(value, dtype=u.dtype) if ppf is None
                         else ppf(jnp.clip(u[i], bottom, top)))
                    theta = theta.at[i].set(x)
            return theta

        return draw

    def exact_fixed(self, theta):
        """``theta`` (host rows, ``names`` order) with each Fixed column set to its float64 value
        (the device holds it in the sweep's precision)."""
        theta = np.array(theta, dtype=float, copy=True)
        for nm, v in self.fixed.items():
            theta[..., self.names.index(nm)] = v
        return theta


def _resolve_precision(jax, precision):
    """``precision=`` -> the ``jax_enable_x64`` value the sweep runs under."""
    if precision is None:
        return bool(jax.config.jax_enable_x64)
    if precision in ("float32", "float64"):
        return precision == "float64"
    raise ValueError(f"precision must be None (the session's precision), 'float32' or 'float64'; "
                     f"got {precision!r}.")


def _float64_only_error(model_name, exc):
    """The ValueError for ``precision="float32"`` on an engine that raised its float64 guard."""
    msg = str(exc)
    reason = msg.split("requires float64.", 1)[1].split(". ", 1)[0].strip()
    return ValueError(
        f"abc_gpu(precision='float32'): model {model_name!r} runs only in float64 ({reason}). "
        f"The supernova and TDE engines are float64-only; float32 is for the kilonovae and the "
        f"flare. Pass precision='float64', or leave precision=None in a float64 session.")


def _logl_scan_rows(dist, max_logl_scan):
    """Positions scored for the best fit: all of them, or, when ``max_logl_scan`` binds, the ones
    with the LOWEST ``dist`` (never the first ones accepted), and always at least one."""
    n_cap = len(dist) if max_logl_scan is None else max(int(max_logl_scan), 1)
    return (np.argsort(dist, kind="stable")[:n_cap] if len(dist) > n_cap
            else np.arange(len(dist)))


def _best_by_logl(jax, jnp, predict_jax, t_j, theta, dist, lik, scatter_idx, chunk,
                  max_logl_scan):
    """Best fit among the rows of ``theta`` by EXACT log-likelihood, not by distance.

    Returns ``(row, max ln L, n scored, capped)``. Every row is scored unless ``max_logl_scan``
    binds; then the rows with the LOWEST ``dist`` are kept, and always at least one. Used by
    ``abc_smc_gpu``, whose final population is assembled over several waves; ``abc_gpu`` needs no
    second pass, because its sweep already returns every draw's log-likelihood.

    The fluxes come from ONE on-device scan over ``chunk``-wide blocks, padded like the simulation
    sweep (a ragged last block would be a second compile) and transferred once. The python loop it
    replaces synchronised with the host once per block, which was tolerable while only the first
    2000 accepted draws were scored and is not now that every one is. The likelihood here is
    whisper's own ``lik.log_likelihood``, in numpy on the fluxes. (``abc_gpu``'s sweep uses its JAX
    twin, :func:`~whisper_cbpf.likelihood._jax.log_likelihood_jax`, which reads the data and the
    normalising constants off the same object, so the two cannot rank by different rules.)
    """
    scan = _logl_scan_rows(dist, max_logl_scan)
    width = max(min(int(chunk), len(scan)), 1)
    n_blocks = -(-len(scan) // width)
    rows = np.resize(scan, n_blocks * width)          # pads by wrapping; dropped after the scan
    blocks = jnp.asarray(theta[rows].reshape(n_blocks, width, theta.shape[1]))
    _, flux = jax.lax.scan(lambda _, th: (None, predict_jax(th, t_j)), None, blocks)
    flux = np.asarray(flux, dtype=float).reshape(n_blocks * width, -1)[:len(scan)]
    logls = np.asarray([lik.log_likelihood(f, sigma_extra=float(theta[i, scatter_idx]))
                        if scatter_idx is not None else lik.log_likelihood(f)
                        for f, i in zip(flux, scan)], dtype=float)
    b = int(np.nanargmax(logls)) if np.any(np.isfinite(logls)) else 0
    return int(scan[b]), float(logls[b]), len(scan), len(scan) < len(theta)


def _sweep(jax, jnp, block, n_blocks, width, precision, model_name):
    """Run every simulation as ONE on-device ``lax.scan`` of ``block`` over ``n_blocks`` blocks.

    Returns ``(theta, distance, logl, seconds)`` as device arrays of shape ``(n_blocks, width, D)``,
    ``(n_blocks, width)`` and ``(n_blocks, width)``, padded tail included, and the wall time of the
    sweep, compile included. Nothing is transferred here.

    A float64-only engine traced under ``precision="float32"`` raises its own RuntimeError at trace
    time, before any compute; it is re-raised as a ValueError that names the model and the fix.
    """
    # ONE on-device scan, not a Python loop with a host round trip per block.
    #
    # The loop this replaces did `block_until_ready` + `np.asarray` on every block, so a
    # 512,000-simulation run at chunk=16 made 32,000 device->host round trips and 32,000 python
    # dispatches. Measured on the two-component kilonova, that scaffolding cost MORE than the
    # physics: 0.5800 ms/sim total against a forward model of only 0.2723 ms/sim, i.e. 53% of
    # the runtime was overhead. Enough to turn a 61x-faster forward model into a dead heat with
    # 48 redback CPU cores.
    #
    # lax.scan keeps the whole sweep -- RNG, forward model, noise, distance, log-likelihood --
    # resident on the device. Compile is still governed by `width` (the vmap extent inside the
    # body), which is why DEFAULT_CHUNK is a measured width; what disappears is the per-block
    # synchronisation, not the per-block compile.
    #
    # PADDING: indices run to n_blocks*width, so up to width-1 simulations past n_total are
    # computed and then discarded. That is deliberate -- a ragged final block would be a second
    # shape and therefore a second XLA compile, which on this model costs far more than the
    # handful of wasted draws. The caller drops the padded draws before the quantile, so they
    # cannot influence epsilon or acceptance.
    # No explicit dtype: `fold_in` keys off this VALUE, so the default integer width must match
    # what the previous `jnp.arange(n_done, n_done + m)` produced or every draw changes.
    idx = jnp.arange(n_blocks * width).reshape(n_blocks, width)
    t0 = time.perf_counter()
    try:
        _, (theta_dev, dist_dev, logl_dev) = jax.lax.scan(block, None, idx)
    except RuntimeError as exc:
        if precision == "float32" and "requires float64." in str(exc):
            raise _float64_only_error(model_name, exc) from exc
        raise
    jax.block_until_ready(dist_dev)
    return theta_dev, dist_dev, logl_dev, time.perf_counter() - t0


class ABCGPUSampler(BaseSampler):
    """Rejection ABC with a vmapped JAX simulator. See the module docstring for matched semantics."""

    name = "abc_gpu"

    def fit(self, lc, model, prior=None, *, predict_jax=None, n_simulations=10000, quantile=0.01,
            threshold=None, distance="chi2", simulate_noise=True, space="auto", scatter_param=None,
            seed=0, chunk=DEFAULT_CHUNK, max_logl_scan=None, precision=None) -> SamplerResult:
        """Fit ``lc`` with ``model`` by rejection ABC on the GPU.

        Each prior draw costs one forward-model evaluation, which yields both its distance and its
        exact log-likelihood; the best fit is the accepted draw with the highest log-likelihood.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve, with errors in the comparison space.
        model : str or Model
            A registered model name or a :class:`~whisper_cbpf.models.Model` carrying a
            ``predict_jax`` (anything built by a JAX factory), unless ``predict_jax=`` is passed.
        prior : Prior, optional
            Defaults to ``model.default_prior``. ``Uniform``, ``LogUniform``, ``Normal``,
            ``TruncatedNormal`` and ``Fixed``, each drawn on the device through its inverse CDF
            (exact draws; see :class:`_PriorDraw`). A ``Fixed`` parameter is held at its value: a
            constant column in ``samples``, listed in ``info["fixed"]``, and not counted in
            AIC/BIC (``n_params``).
        predict_jax : callable, optional
            ``predict_jax(theta_2d, times) -> flux_2d`` mapping ``(B, D)`` parameters and ``(n,)``
            times to ``(B, n)`` model **flux**, on device. Parameter columns are in ``prior.names``
            order.

            **Optional since the adapter layer landed.** When omitted it is built from
            ``model.predict_jax`` by
            :func:`whisper_cbpf.samplers.jax._adapters.make_batched_predict_jax`, so any model
            registered through the JAX factories just works. An explicitly passed callable always
            wins. A model with no ``predict_jax`` still raises, naming ``sampler='abc'`` -- falling
            back to the numpy ``model.predict`` in a python loop would be the CPU sampler with
            extra steps.
        distance : str
            One of ``chi2`` (default, matching whisper's CPU ABC), ``mse``, ``rmse``, ``mae``,
            ``wmse``, ``wmae``, ``max_abs_z``. See :mod:`whisper_cbpf.distance` for what each
            implies -- the distance defines what "close" means, so it defines the posterior.
            ``max_abs_z`` with ``threshold=k`` accepts a draw only if every point is within ``k``
            of its own error (the v3 rule: ``threshold=5, simulate_noise=False,
            space="magnitude"``).
        n_simulations, quantile, threshold, simulate_noise, space, scatter_param, seed
            As ``whisper_cbpf.samplers.abc.ABCSampler.fit``. ``threshold`` overrides ``quantile``.
            ``scatter_param`` adds that draw's scatter to the SIMULATED noise; it does not make the
            parameter fittable (see the warning raised below).
        chunk : int
            Particles per compiled call. **Its limit is compile time, not memory**, and the compile
            time is erratic in the width rather than growing with it: the kilonovae compile in a
            few seconds at 250 (the default) but take minutes at 256 or 1024. See
            :data:`DEFAULT_CHUNK` for the measurements; prefer a width that is not a power of two.
        max_logl_scan : int, optional
            Cap on the accepted draws considered for ``best_params``. **Default ``None`` considers
            every accepted draw.** When a cap binds, the draws with the LOWEST distance are
            considered. Every draw's log-likelihood comes out of the sweep, so the cap no longer
            saves any time; it is kept so a capped run reproduces 0.1.1's choice.
            ``info['logl_scan_n']`` and ``info['logl_scan_capped']`` record what was done.
        precision : {None, "float32", "float64"}, default None
            Floating-point precision of the sweep. ``None`` follows the session
            (``jax_enable_x64``). ``"float32"`` runs single precision even in a float64 session,
            which is where a GPU is fastest (an RTX A6000 runs float64 at 1/32 of the float32
            rate); it is accurate for the kilonovae and the flare, whose float32 paths the package
            tests against float64. The supernova and TDE engines are float64-only, and
            ``precision="float32"`` on them raises a ValueError naming the reason. ``"float64"``
            runs double precision even in a float32 session. The posterior predictive and band
            metrics run in the session's precision. ``info['precision']`` records what ran.

        Returns
        -------
        SamplerResult
            Accepted draws (``samples``, with their ``distance``), ``best_params`` (the accepted
            draw with the highest exact log-likelihood, computed on the device in the sweep's
            precision), ``max_log_likelihood``, ``aic``, ``bic`` and ``min_distance`` (the closest
            draw's distance). ``runtime_s`` is the sweep, compile included;
            ``info['postprocess_s']`` the acceptance and best-fit step after it. If no draw is
            accepted, ``best_params`` is the closest rejected draw
            (``info['best_params_source']``) and the predictive metrics are skipped.

        Raises
        ------
        ValueError
            No ``predict_jax`` to build from; every parameter ``Fixed``; an unknown
            ``precision``; or ``precision="float32"`` on a float64-only engine.
        TypeError
            A prior family other than those listed under ``prior``.

        Examples
        --------
        The v3 rule (every point within 5 sigma) on a flare simulated from the model itself:

        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> from whisper_cbpf.models import get_model
        >>> from whisper_cbpf.samplers.jax.abc_gpu import ABCGPUSampler
        >>> flare = get_model("flare_jax")
        >>> t = np.linspace(1.0, 29.0, 30)
        >>> truth = {"log_amp": 0.5, "log_sigma": 0.3, "log_tau": 1.5, "t0": 8.0}
        >>> flux = np.asarray(flare.predict(truth, t, None), float)
        >>> lc = wp.LightCurve(time=t, band=np.array(["r"] * 30), flux=flux,
        ...                    flux_err=np.full(30, 0.05 * flux.max()))
        >>> res = ABCGPUSampler().fit(lc, "flare_jax", n_simulations=20000, distance="max_abs_z",
        ...                           threshold=5.0, simulate_noise=False, space="flux")
        >>> res.info["n_accepted"] > 0 and bool(np.all(res.samples["distance"] <= 5.0))
        True
        """
        jax, jnp = require_jax("abc_gpu")
        x64 = _resolve_precision(jax, precision)

        dist_fn = jnp_distance(distance)
        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        names = list(prior.names)

        predict_jax_source = "caller"
        if predict_jax is None:
            if getattr(model, "predict_jax", None) is None:
                raise ValueError(
                    f"abc_gpu needs predict_jax(theta_2d, times) -> flux_2d, a batched JAX forward "
                    f"model, and model {model.name!r} carries no predict_jax to build one from. "
                    f"Register it through a JAX factory (register_kilonova / register_kilonova_two "
                    f"/ register_tde / register_supernova), pass predict_jax= yourself, or use "
                    f"sampler='abc' instead.")
            predict_jax_source = "auto"          # built below, inside the sweep's precision
        if scatter_param is not None:
            if scatter_param not in names:
                raise ValueError(f"scatter_param {scatter_param!r} is not in the prior ({names}).")
            # A noise scale is a LIKELIHOOD parameter and is not identifiable by a rejection
            # distance: extra simulated noise only ever raises the expected residual, so every
            # distance here is monotonically penalised by it and the posterior rails to the
            # smallest allowed scatter. Fit it with a likelihood-based sampler, or leave it out of
            # the ABC prior (which is what whisper's own AT2017GFO analysis does).
            _warn_user(
                f"abc_gpu: {scatter_param!r} is in the ABC prior, but a noise scale cannot be "
                "inferred from a rejection distance -- its marginal will rail to the prior's lower "
                "edge and is not a measurement. Drop it from the ABC prior and fit it with mcmc / "
                "nuts_gpu instead.")

        lik_space = GaussianLikelihood(lc, space=space)
        times = np.asarray(lc.time, dtype=float)
        obs_y = np.asarray(lik_space.y, dtype=float)
        obs_err = np.asarray(lik_space.sigma, dtype=float)
        sp, zp = lik_space.space, lik_space.zeropoint_jy

        spec = _PriorDraw(prior, names, "abc_gpu")
        scatter_idx = names.index(scatter_param) if scatter_param else None
        k, n = len(names), lc.n_points
        k_free = len(spec.free)                          # a Fixed parameter is not counted in AIC/BIC
        # The likelihood the best fit is ranked by -- the scatter-augmented one when a scatter
        # parameter is fitted -- is evaluated INSIDE the sweep, through its JAX twin.
        lik = (make_likelihood(lc, kind="gaussian_scatter", space=space,
                               scatter_param=scatter_param) if scatter_param
               else make_likelihood(lc, space=space))
        n_total = int(n_simulations)
        width = max(min(int(chunk), n_total), 1)
        n_blocks = -(-n_total // width)                  # ceil; the last block is padded, see below

        # Everything that creates or traces a device array runs under the sweep's precision, so
        # `precision=` takes effect whatever the session's jax_enable_x64 says.
        with jax.enable_x64(x64):
            if predict_jax is None:
                from ._adapters import make_batched_predict_jax
                # names=prior.names: the adapter reorders into model.parameters order internally
                # and drops a prior-only column (scatter_param) rather than feeding it to the
                # physics. chunk stays None -- this sampler already bounds the batch through its
                # own `chunk`, so a lax.map here would only add a trip-count-1 scan.
                predict_jax = make_batched_predict_jax(lc, model, names=names,
                                                       chunk=None, max_batch=None)
            logl_fn = log_likelihood_jax(lik)
            t_j = jnp.asarray(times)
            y_j, e_j = jnp.asarray(obs_y), jnp.asarray(obs_err)
            draw = spec.build(jnp)

            def one(key):
                """One simulation: prior draw -> forward model -> (distance, ln L). Vmapped below.

                ONE model call per draw. The exact log-likelihood is taken from the same noiseless
                flux the simulation starts from, so the best-fit ranking needs no second pass.
                """
                k_theta, k_noise = jax.random.split(key)
                u = jax.random.uniform(k_theta, (k,))
                theta = draw(u)                              # the inverse CDF of every parameter
                flux = predict_jax(theta[None, :], t_j)[0]
                logl = (logl_fn(flux) if scatter_idx is None
                        else logl_fn(flux, sigma_extra=theta[scatter_idx]))
                sim = _model_in_space_jnp(jnp, flux, sp, zp)
                if simulate_noise:
                    # scatter enters the SIMULATED noise only, never the distance denominator
                    sig = (e_j if scatter_idx is None
                           else jnp.sqrt(e_j ** 2 + theta[scatter_idx] ** 2))
                    sim = sim + jax.random.normal(k_noise, sim.shape) * sig
                # the distance denominator is the RAW observational error even when a scatter
                # parameter is set -- the scatter enters the simulated noise only. Weighting the
                # residual by the inflated error too would double-count it.
                return theta, dist_fn(y_j, e_j, sim), logl

            base_key = jax.random.PRNGKey(int(seed))

            def block(_, block_idx):
                """One vmapped block of `width` simulations. The scan body -- compiled ONCE."""
                # fold_in per GLOBAL simulation index, mirroring the CPU's default_rng([seed, idx]):
                # the draw for index i depends only on (seed, i), never on how the run was blocked.
                # That property is what makes chunk a pure performance knob, and a test asserts it.
                keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(block_idx)
                return None, jax.vmap(one)(keys)

            theta_dev, dist_dev, logl_dev, runtime = _sweep(
                jax, jnp, block, n_blocks, width, precision, model.name)

            t1 = time.perf_counter()
            # The distances come to the host (float64, as before) for the acceptance quantile, so
            # epsilon and the accepted set are computed exactly as 0.1.1 computed them. The padded
            # tail is dropped here, before the quantile.
            distances = np.asarray(dist_dev, dtype=float).reshape(-1)[:n_total]
            epsilon = (float(np.quantile(distances, quantile)) if threshold is None
                       else float(threshold))
            keep = distances <= epsilon                  # inclusive, matching abc.py
            acc_idx = np.nonzero(keep)[0]
            # At least one candidate, always, even at `max_logl_scan=0` ("skip the ranking, I only
            # want timings"): an empty scan would die on `scan[0]` after all the sampling was done.
            cand = acc_idx if len(acc_idx) else np.array([int(np.argmin(distances))])
            # Only these rows leave the device: the accepted draws (or the closest rejected one)
            # and their log-likelihoods. 0.1.1 transferred every draw and then re-ran the model on
            # the accepted ones in a second compiled scan.
            rows = jnp.asarray(cand)
            theta_c = spec.exact_fixed(np.asarray(theta_dev.reshape(-1, k)[rows], dtype=float))
            logl_c = np.asarray(logl_dev.reshape(-1)[rows], dtype=float)

        if not keep.any():
            _warn_user(
                f"abc_gpu accepted 0 of {n_simulations} draws at epsilon={epsilon:g}; the posterior "
                "is empty. best_params / AIC / BIC reflect the single closest draw, which was NOT "
                "accepted, and the predictive metrics are skipped.")
        samples = pd.DataFrame(theta_c if len(acc_idx) else np.empty((0, k)), columns=names)
        samples["distance"] = distances[acc_idx]

        # Best fit by EXACT likelihood, not by distance: with simulate_noise the argmin-distance
        # draw is the luckiest noise realisation. Every accepted draw is considered by default.
        chi2_min = float(distances.min())               # over ALL draws, matching abc.py
        scan = _logl_scan_rows(distances[cand], max_logl_scan)
        b = (int(scan[np.nanargmax(logl_c[scan])]) if np.any(np.isfinite(logl_c[scan]))
             else int(scan[0]))
        max_log_likelihood = float(logl_c[b])
        best = {nm: float(theta_c[b, j]) for j, nm in enumerate(names)}
        n_scanned, capped = len(scan), len(scan) < len(cand)
        postprocess_s = time.perf_counter() - t1

        info = {
            "n_simulations": int(n_simulations),
            "n_accepted": int(keep.sum()),
            "acceptance_rate": float(keep.mean()),
            "epsilon": epsilon,
            "quantile": None if threshold is not None else float(quantile),
            "simulate_noise": bool(simulate_noise),
            "space": lik_space.space,
            "scatter_param": scatter_param,
            "distance": str(distance),
            "likelihood_space": lik.space,
            "backend": "jax-vmap",
            "chunk": int(chunk),
            "device": str(jax.devices()[0]),
            "precision": "float64" if x64 else "float32",
            "x64": bool(x64),                            # the sweep's precision
            "x64_session": bool(jax.config.jax_enable_x64),
            "predict_jax": predict_jax_source,
            "seed": int(seed),
            "logl_scan_n": int(n_scanned),
            "logl_scan_capped": bool(capped),
            "postprocess_s": float(postprocess_s),
            "best_params_source": "accepted_draws" if len(acc_idx) else "closest_rejected_draw",
            # points simulated at mag_floor. None = not counted: this sampler runs the
            # forward map inside one compiled scan, and a traced map counts nothing.
            "mag_floor_stats": (dict(fs) if (fs := getattr(predict_jax, "floor_stats", None))
                                and fs["n_points"] else None),
            "fixed": dict(spec.fixed),
        }
        attach_band_metrics(info, lc, model, best, space)
        aic, bic = aic_bic(max_log_likelihood, k_free, n)
        result = SamplerResult(
            sampler="abc_gpu", model=model.name, parameters=names, samples=samples,
            summary=summarize_posterior(samples, names), best_params=best,
            n_data=n, n_params=k_free, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=max_log_likelihood,
            aic=aic, bic=bic,
        )
        # (n_chains, n_draws, n_params) for the shared ESS helper; ABC draws are independent.
        result.samples_by_chain = samples[names].to_numpy()[None, :, :]
        attach_predictive_metrics(result, lc, space, model=model)
        return result


def fit_ABC_GPU(lc, model, prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` by rejection ABC on the GPU. See :meth:`ABCGPUSampler.fit`.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models import get_model
    >>> from whisper_cbpf.samplers.jax import fit_ABC_GPU
    >>> t = np.linspace(1.0, 29.0, 30)
    >>> truth = {"log_amp": 0.5, "log_sigma": 0.3, "log_tau": 1.5, "t0": 8.0}
    >>> flux = np.asarray(get_model("flare_jax").predict(truth, t, None), float)
    >>> lc = wp.LightCurve(time=t, band=np.array(["r"] * 30), flux=flux,
    ...                    flux_err=np.full(30, 0.05 * flux.max()))
    >>> res = fit_ABC_GPU(lc, "flare_jax", n_simulations=20000, quantile=0.01, precision="float32")
    >>> res.info["n_accepted"], res.info["precision"]
    (200, 'float32')
    """
    return ABCGPUSampler().fit(lc, model, prior=prior, **kwargs)
