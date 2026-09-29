"""Turn a registered ``Model`` into the callables the JAX samplers take.

Why this exists
---------------
The five JAX samplers each need one of two things, and until now the *user* had to build it:

* ``abc_gpu`` / ``abc_smc_gpu`` want a **batched forward map**, ``(B, D) theta + (n,) times ->
  (B, n) flux``.
* ``nuts_gpu`` / ``pymc_jax_gpu_*`` / ``emcee_jax`` want a **scalar density**, ``(D,) theta ->
  scalar``.

Meanwhile :class:`whisper_cbpf.models.Model` grew a ``predict_jax`` slot with a *different*
signature — one parameter set, an explicit band-index argument — and nothing read it. So a user
holding a perfectly good registered kilonova still had to hand-write a vmapped closure and a
likelihood before any GPU sampler would accept it. This module is the missing translation, and it
is what makes ``wp.fit(lc, "kilonova_one_jax", sampler="nuts_gpu")`` work.

The samplers call these only when the caller passed nothing — an explicit ``predict_jax=`` or
``log_prob_fn=`` always wins, so no existing call changes behaviour.

No JAX at module scope: ``require_jax`` is called inside each function, exactly as the model
factories do, so importing this module on a CPU-only install is free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...priors._numpy import family
from ..base import _warn_user

__all__ = ["MAX_UNCHUNKED_BATCH", "MAG_FLOOR_WARN_FRACTION", "float_dtype", "resolve_band_index",
           "make_batched_predict_jax", "make_log_prob_jax", "resolve_density",
           "resolve_sampling_names", "free_density", "log_density", "LogDensity", "bucket_size"]

#: Largest leading batch axis :func:`make_batched_predict_jax` will vmap in one compiled call.
#:
#: This is the fifth-instance guard for the trap ``docs/GPU_SETUP.md`` calls "the unbounded-vmap
#: trap": a ``jax.vmap`` whose extent is a *simulation count* builds one fused kernel sized by a
#: user-tunable number, and on a photometric model XLA's compile time explodes long before memory
#: does. ``abc_gpu`` measured B=256 as "did not finish compiling in 6 minutes", which is where this
#: number comes from. Raise it only for a cheap analytic model with no wavelength axis.
#:
#: Set to 64, not 256, and compared with >=. 256 was the width abc_gpu measured as "did not finish
#: compiling in 6 minutes", so a guard that ADMITS 256 admits the documented hang. Re-measured here
#: on the cited configuration (n_wave=1000, 213 observations, GPU): first-call compile+run was
#: 86.3 s at B=64 and 586.9 s at B=128. 64 is the last width that is merely slow.
MAX_UNCHUNKED_BATCH = 64

#: :func:`make_batched_predict_jax` warns (once per forward map) when more than this fraction of
#: the points it has simulated sit at the model's ``mag_floor``. The JAX models cap
#: magnitudes there and redback does not; a simulation-based sampler trains on prior draws, and on
#: ZTF20achncvv's magnetar 51% of the simulated points were exactly 40.000 mag (38.7% of curves
#: constant end to end) where the CPU simulator reached 93.9 mag. Nothing is changed -- the count
#: is kept on the map as ``.floor_stats`` and the warning says how to move the floor.
MAG_FLOOR_WARN_FRACTION = 0.10


def float_dtype():
    """``jnp.float64`` when ``jax_enable_x64`` is set, else ``jnp.float32``.

    Shared by the adapters and by ``nuts_gpu`` so precision is a stated function of the session
    flag rather than whatever ``jnp.asarray`` happens to do to a float64 numpy array.
    """
    from ...backends import require_jax
    jax, jnp = require_jax("float_dtype")
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def resolve_band_index(lc, model):
    """Map ``lc``'s band labels onto the integer indices ``model.predict_jax`` was bound to.

    Returns ``None`` for a *self-contained* model — one whose ``predict_jax`` carries no
    ``.band_index`` attribute, e.g. the flare — signalling that the caller should omit the band
    argument entirely.

    Relabelling lives on the MODEL, not here: pass ``band_aliases={"g": "sdssg", ...}`` to the
    factory. That is the only place it can go and still be honoured by ``per_band_metrics``,
    ``predictive_metrics`` and ``plot_ppc``, all of which call ``model.predict(..., lc.band)``
    directly. An alias known only to the sampler produces a fit whose WAIC silently fails.
    """
    from ...backends import require_jax
    _, jnp = require_jax("resolve_band_index")

    predict_jax = getattr(model, "predict_jax", None)
    if predict_jax is None:
        raise ValueError(
            f"model {getattr(model, 'name', model)!r} has no predict_jax, so there is no JAX "
            f"forward map to build from. Register it through one of the JAX factories "
            f"(register_kilonova / register_kilonova_two / register_tde / register_supernova), or "
            f"pass predict_jax=/log_prob_fn= explicitly.")

    index_of = getattr(predict_jax, "band_index", None)
    if index_of is None:
        return None                                    # self-contained model: no band axis

    return jnp.asarray(index_of(np.asarray(lc.band)))


def _gather_permutation(names, model_parameters, model_name):
    """Static index array reordering caller-supplied theta columns into ``model.parameters`` order.

    Load-bearing, and the reason is not cosmetic. ``abc_gpu`` takes its parameter order from
    ``prior.names`` (a dict's insertion order) while ``Model.predict_jax``'s contract is
    ``model.parameters`` order. Every in-tree factory happens to make the two agree, so nothing has
    ever caught it — but a user passing ``prior=Prior({...})`` in a different order would have the
    ``kappa`` prior evaluated at ``vej`` and get a converged fit to the wrong model, silently.

    The gather also *drops* any column that is in the prior but not in the model, which is exactly
    the ``scatter_param`` case: ``sigma`` is sampled and fed to the likelihood, never to the
    forward model.

    Returns ``None`` when the order already matches and nothing needs dropping.
    """
    missing = [p for p in model_parameters if p not in names]
    if missing:
        raise ValueError(
            f"model {model_name!r} needs parameter(s) {missing} which are not in the sampling "
            f"order {list(names)}. Pass a prior covering every model parameter.")
    perm = [list(names).index(p) for p in model_parameters]
    return None if perm == list(range(len(names))) else np.asarray(perm, dtype=int)


def make_batched_predict_jax(lc, model, *, names=None, times=None,
                             chunk=None, max_batch=MAX_UNCHUNKED_BATCH):
    """Build ``f(theta_2d, times=None) -> flux_2d`` in the shape ``abc_gpu`` documents.

    ``theta_2d`` is ``(B, D)`` with columns ordered by ``names`` (default ``model.parameters``);
    the result is ``(B, n)`` flux density in Jy, ``n = len(times)``.

    Parameters
    ----------
    names : sequence of str, optional
        The order the *caller* supplies theta columns in. Reconciled with ``model.parameters``
        internally; see :func:`_gather_permutation`.
    times : array, optional
        Defaults to ``lc.time``. The ``times`` argument of the returned function is **validated,
        not used** — the array is closed over, because the supernova family memoises its compiled
        core on the observation times and cannot accept a traced one. It reaches the model as the
        float64 numpy array, so a model's ``t_exp_days`` is subtracted before any float32 cast
        (casting first cost 7.2 mmag on a raw-MJD clock).
    chunk : int, optional
        Evaluate the batch in blocks of this width via ``jax.lax.map`` instead of one wide vmap,
        jitted once here, so it compiles once per batch size and not on every call. Leave ``None``
        when the caller already bounds the batch (``abc_gpu`` does, via its own ``chunk``), so no
        trip-count-1 scan is added.
    max_batch : int, optional
        Refuse an unchunked batch wider than this. ``None`` disables the guard.

    Two model hooks are honoured, both read off ``model.predict_jax`` like ``band_index``:
    ``constraint_ok`` (redback's ``Constraint`` priors) -- a draw that breaks one
    predicts zero flux in every epoch, which every distance rejects -- and ``mag_floor``:
    the points of the draws that pass, at the floor, are counted into ``.floor_stats``
    (``n_points``, ``n_floored``, ``fraction``, ``mag_floor``) on every call made outside a trace,
    with a warning once the fraction passes :data:`MAG_FLOOR_WARN_FRACTION`.
    """
    from ...backends import require_jax
    jax, jnp = require_jax("make_batched_predict_jax")

    names = list(names) if names is not None else list(model.parameters)
    perm = _gather_permutation(names, model.parameters, getattr(model, "name", "?"))
    bidx = resolve_band_index(lc, model)
    t_source = np.asarray(lc.time if times is None else times, dtype=float)
    predict_jax = model.predict_jax
    ok_fn = getattr(predict_jax, "constraint_ok", None)
    perm_j = None if perm is None else jnp.asarray(perm)

    def one(theta):
        th = theta if perm_j is None else theta[perm_j]
        # float64 numpy epochs: the model casts them, after subtracting its t_exp_days (1.3).
        flux = predict_jax(th, t_source) if bidx is None else predict_jax(th, t_source, bidx)
        return flux if ok_fn is None else jnp.where(ok_fn(th), flux, 0.0)

    vmapped = jax.jit(jax.vmap(one))
    # Built once, here, not per call: an eager lax.map recompiles on every call.
    mapped = None if chunk is None else _chunked_mapper(jax, jnp, vmapped, int(chunk))
    mag_floor = getattr(predict_jax, "mag_floor", None)
    stats = dict(mag_floor=mag_floor, n_points=0, n_floored=0, fraction=0.0)

    def record_floor(flux):
        """Count the floored points of a concrete result; a traced one is skipped."""
        if mag_floor is None or isinstance(flux, jax.core.Tracer):
            return
        from ...io.photometry import AB_ZEROPOINT_JY

        floor = AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag_floor)
        live = flux != 0.0                  # a walled draw is exactly zero; it is not a curve
        n_live = int(jnp.sum(live))
        n_floor = int(jnp.sum(live & jnp.isclose(flux, floor, rtol=1e-4, atol=0.0)))
        warned = stats["fraction"] > MAG_FLOOR_WARN_FRACTION
        stats["n_points"] += n_live
        stats["n_floored"] += n_floor
        stats["fraction"] = stats["n_floored"] / max(stats["n_points"], 1)
        if not warned and stats["fraction"] > MAG_FLOOR_WARN_FRACTION:
            _warn_user(
                f"{stats['n_floored']} of {stats['n_points']} simulated points "
                f"({100 * stats['fraction']:.1f}%) of model {getattr(model, 'name', '?')!r} are "
                f"at mag_floor={mag_floor:g}, the magnitude the JAX models cap at; redback has no "
                f"cap. A simulation-based fit trains on them as they are. Nothing was changed: "
                f"raise mag_floor= at the factory (float64 is clean to 100) to see the faint end, "
                f"or read .floor_stats on this forward map.", UserWarning)

    def batched(theta_2d, times=None):
        theta_2d = jnp.asarray(theta_2d)
        _check_times(times, t_source, np)
        b = int(theta_2d.shape[0])
        if chunk is None:
            if max_batch is not None and b >= max_batch:
                raise ValueError(
                    f"make_batched_predict_jax was asked to vmap {b} parameter sets in one "
                    f"compiled call, at or above max_batch={max_batch}. A vmap whose extent is a "
                    f"simulation count is the trap that has stalled this project's compiles five "
                    f"times. Pass chunk=<width> (abc_gpu's own default is 250), or raise max_batch "
                    f"if your forward model is analytic and has no wavelength axis.")
            flux = vmapped(theta_2d)
        else:
            flux = mapped(theta_2d)
        record_floor(flux)
        return flux

    batched.band_index = bidx
    batched.names = names
    batched.floor_stats = stats
    return batched


def _check_times(times, t_source, np_mod):
    """Validate a caller-supplied ``times`` against the array the adapter closed over.

    The samplers pass ``times`` positionally out of habit; silently ignoring a *different* array
    would fit the wrong epochs. Under a tracer the value check is skipped (a tracer cannot become
    a numpy array), but the shape check still applies.

    Compared against the **float64 numpy source**, not against the on-device ``t_ref``: in a
    float32 session ``t_ref`` has already been rounded, so an exact comparison against it would
    reject the very array it was built from.
    """
    if times is None:
        return
    # SHAPE FIRST, and unconditionally: a tracer has a .shape even though it has no value, so the
    # cheap check that catches "wrong number of epochs" must not sit behind the conversion that
    # only concrete arrays survive. Returning from inside the except would mean that, under a
    # tracer NO check ran at all and a 7-element traced `times` was silently evaluated on the
    # closed-over 24-epoch grid.
    shape = getattr(times, "shape", None)
    if shape is not None and tuple(shape) != t_source.shape:
        raise ValueError(
            f"times has shape {tuple(shape)} but this forward map was built for "
            f"{t_source.shape}. Rebuild the adapter for the new epochs.")
    try:
        arr = np_mod.asarray(times, dtype=float)
    except Exception:
        return                                          # traced: nothing concrete to compare
    if arr.shape != t_source.shape:
        raise ValueError(
            f"times has shape {arr.shape} but this forward map was built for {t_source.shape}. "
            f"Rebuild the adapter for the new epochs.")
    # Tolerance sized to the float32 round-trip and NOTHING more. The ABC samplers hand back their
    # own on-device copy of ``lc.time``, which in a float32 session has passed through single
    # precision and no longer compares equal to the float64 source; float32 eps is 1.2e-7, so 1e-6
    # covers it with margin.
    #
    # 1e-5 would be near-inert on MJD-valued times -- exactly what
    # loader produces. At MJD 58000 a relative 1e-5 permits a 0.58 DAY epoch error, so shifts of
    # 0.001-0.5 d were all accepted; on a days-since-explosion grid the same rtol correctly
    # rejected 1e-5 d. Four orders of magnitude of behaviour depending on the caller's time origin
    # is not a check. Scaling by the SPAN rather than the values removes the origin dependence.
    span = float(np_mod.ptp(t_source)) or 1.0
    if not np_mod.allclose(arr, t_source, rtol=0.0, atol=1e-6 * span):
        raise ValueError(
            "times differs from the array this forward map was built for. The epochs are closed "
            "over (the supernova family compiles its grid from them), so passing a different "
            "array would silently fit the wrong times.")


def _chunked_mapper(jax, jnp, vmapped, width):
    """Build a JITTED ``theta_2d -> flux_2d`` that ``lax.map``s ``vmapped`` over ``ceil(B/width)``
    padded blocks, then trims.

    Constant memory, linear cost, one compile per batch size -- the shape ``abc_gpu`` uses, built
    the way ``emcee_jax._make_walker_mapper`` is. Padding rather than a ragged tail keeps every
    block the same shape, so there is exactly one compiled kernel.

    **The jit is not optional.** An un-jitted ``lax.map`` is traced and compiled
    again on every call, and ``snpe_gpu`` makes one call per simulation round. Measured on one
    A6000, float64, B = 1000, per repeated call before the jit: arnett 0.59-0.89 s, kilonova
    1.17-1.26 s, the Gaussian-rise TDE 7.1-7.3 s -- each one XLA compile of a kernel already built.
    """
    @jax.jit
    def mapped(theta_2d):
        b = theta_2d.shape[0]
        if b == 0:                               # an ABC round can legitimately accept nothing;
            return vmapped(theta_2d)             # reshape(0, -1) cannot infer the trailing dim
        n_blocks = -(-b // width)
        pad = n_blocks * width - b
        if pad:
            theta_2d = jnp.concatenate([theta_2d, jnp.repeat(theta_2d[-1:], pad, axis=0)], axis=0)
        out = jax.lax.map(vmapped, theta_2d.reshape(n_blocks, width, theta_2d.shape[-1]))
        return out.reshape(n_blocks * width, out.shape[-1])[:b]

    return mapped


def make_log_prob_jax(lc, model, prior=None, *, space="auto", names=None,
                      include_prior=False, likelihood=None, times=None, jit=True):
    """Build ``f(theta) -> scalar``, the density the NUTS-family samplers take.

    ``theta`` is a flat ``(D,)`` vector ordered by ``names`` (default ``model.parameters``).

    Parameters
    ----------
    space : {"auto", "flux", "magnitude"}
        Which space the residuals are formed in. ``"auto"`` follows ``lc.data_mode``, so
        magnitude data is fitted in magnitude space — the same rule ``abc``/``abc_smc``/``mcmc``
        already use.
    include_prior : bool
        ``False`` (default) returns a pure **log-likelihood** with an ``-inf`` box indicator, which
        is what ``nuts_gpu`` and ``pymc_gpu`` need: their sample sites contribute the prior density
        themselves, so returning a log-posterior would count the prior twice. ``True`` adds the
        prior density, which is what ``emcee_jax`` needs — it samples this function directly and
        contributes nothing of its own.
    likelihood : str or object, optional
        A registered ``kind`` name (``"auto"``, ``"gaussian"``, ``"upper_limits"``,
        ``"gaussian_scatter"``, ...) or an already-built whisper likelihood object. ``None`` means
        ``"auto"``. The JAX backend implements three of the four registered classes; see
        :func:`whisper_cbpf.likelihood.log_likelihood_jax` for which, and what each refuses.
        ``"gaussian_scatter"`` additionally requires its ``scatter_param`` (default ``"sigma"``) to
        be in ``names`` — that column is read straight out of ``theta`` and passed as
        ``sigma_extra``, which is what makes the density non-constant in it.

    The returned function carries ``.space``, ``.names``, ``.includes_prior``, ``.likelihood``
    (``type(like).__name__``, for ``result.info["likelihood"]``), ``.scatter_param`` and
    ``.log_likelihood`` (the ``include_prior=False`` twin, so a caller that sampled the posterior
    can still rank draws by likelihood for AIC/BIC).

    Notes
    -----
    A model whose ``predict_jax`` carries ``constraint_ok`` (the supernova and TDE factories, for
    the models redback constrains) gets redback's ``Constraint`` priors as a hard wall: ``-inf``
    wherever one fails, exactly like the prior box. The epochs reach the model as the
    float64 numpy array, so its ``t_exp_days`` is subtracted before any float32 cast.

    The prior box is applied as ``clip`` *before* the model and ``where`` *after*. In support the
    clip is the identity, so no value changes; out of support it turns the ``0 * NaN`` of the
    backward pass into a clean ``0``. The physical engines — the TDE's Euler integrator, the
    supernovae's cgs luminosities — are a NaN factory at absurd theta, and a NaN cotangent poisons
    a whole NUTS trajectory. This is the same double-``where`` hazard already documented in
    ``priors/_jax.py`` and ``models/jax/flare.py``.
    """
    from ...backends import require_jax
    from ...likelihood import make_likelihood
    from ...likelihood._jax import log_likelihood_jax

    jax, jnp = require_jax("make_log_prob_jax")

    prior = prior if prior is not None else model.default_prior
    if prior is None:
        raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
    names = list(names) if names is not None else list(model.parameters)

    lows, highs = _box_bounds(prior, names, model.name)
    perm = _gather_permutation(names, model.parameters, getattr(model, "name", "?"))

    like = (likelihood if not isinstance(likelihood, (str, type(None)))
            else make_likelihood(lc, kind=(likelihood or "auto"), space=space))
    try:
        loglik = log_likelihood_jax(like)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{exc}\n\nThis density was built automatically from the light curve: "
            f"`make_likelihood(lc, kind={(likelihood or 'auto')!r}, space={space!r})` chose "
            f"{type(like).__name__}. Pass `likelihood=` naming a kind the JAX backend implements "
            f"(gaussian / upper_limits / gaussian_scatter), drop the rows that force this choice "
            f"(`lc.where(upper_limit=False)`), or build `log_prob_fn` yourself and pass it "
            f"explicitly.") from None

    # A prior-only column is fine for the FORWARD MAP (make_batched_predict_jax drops it, which is
    # the scatter_param case). Here it is fine only if the LIKELIHOOD consumes it. When it does not,
    # nothing downstream does, and the density would be exactly CONSTANT in that parameter.
    # Measured with a plain Gaussian: logL = 71.834030 identically at sigma = 0.02, 0.1 and 0.4,
    # with d(logL)/d(sigma) == 0.0, where the CPU twin gives 63.516 / 32.737 / -0.094. The sampler
    # would converge and report the PRIOR as that parameter's marginal, labelled as a fit. Refuse,
    # as the missing-parameter case does.
    scatter_param = getattr(loglik, "scatter_param", None)
    refused = [n for n in names if n not in set(model.parameters) and n != scatter_param]
    if refused:
        raise ValueError(
            f"parameter(s) {refused} are in the sampling order but neither in model "
            f"{model.name!r} nor consumed by {type(like).__name__}, so the automatic density would "
            f"be exactly constant in them and their posterior would be the prior. A free-scatter "
            f"column is the one exception, and it needs the likelihood that reads it: pass "
            f"likelihood='gaussian_scatter'. Otherwise drop them from the prior for this sampler, "
            f"or pass log_prob_fn= with a density that actually uses them.")
    if scatter_param is not None and scatter_param not in names:
        raise ValueError(
            f"{type(like).__name__} fits a free extra scatter on parameter {scatter_param!r}, but "
            f"that parameter is not in the sampling order {names}. The density would then evaluate "
            f"it at sigma_extra = 0, which is EXACTLY GaussianLikelihood -- a plain Gaussian fit "
            f"reported as a scatter fit. Add {scatter_param!r} to the prior (and to names=), or "
            f"use likelihood='gaussian'.")
    scatter_idx = None if scatter_param is None else names.index(scatter_param)

    bidx = resolve_band_index(lc, model)
    # float64 numpy epochs: the model casts them, after subtracting its t_exp_days on the host.
    t_ref = np.asarray(lc.time if times is None else times, dtype=float)
    predict_jax = model.predict_jax
    _check_epochs(predict_jax, t_ref, prior)
    ok_fn = getattr(predict_jax, "constraint_ok", None)     # redback's Constraint priors (1.6)
    dt = float_dtype()
    lo_j, hi_j = jnp.asarray(lows, dtype=dt), jnp.asarray(highs, dtype=dt)
    perm_j = None if perm is None else jnp.asarray(perm)

    def _log_likelihood(theta):
        theta = jnp.asarray(theta, dtype=dt)
        safe = jnp.clip(theta, lo_j, hi_j)              # never evaluate the physics outside the box
        th = safe if perm_j is None else safe[perm_j]
        flux = predict_jax(th, t_ref) if bidx is None else predict_jax(th, t_ref, bidx)
        inside = jnp.all((theta >= lo_j) & (theta <= hi_j))
        if ok_fn is not None:                           # the wall: -inf where redback rejects
            inside = inside & ok_fn(th)
        # `safe`, not `theta`: the clipped value is what the physics saw, and a scatter drawn
        # outside the box could be <= 0, whose log(var) is a NaN cotangent on the backward pass.
        ll = loglik(flux) if scatter_idx is None else loglik(flux, safe[scatter_idx])
        return jnp.where(inside, ll, -jnp.inf)

    if include_prior:
        from ...priors import log_prob_jax as prior_log_prob_jax
        lp = prior_log_prob_jax(prior, names)

        def _log_posterior(theta):
            theta = jnp.asarray(theta, dtype=dt)
            inside = jnp.all((theta >= lo_j) & (theta <= hi_j))
            return jnp.where(inside, _log_likelihood(theta) + lp(theta), -jnp.inf)

        density = _log_posterior
    else:
        density = _log_likelihood

    out = jax.jit(density) if jit else density
    out.space = like.space
    out.names = names
    out.includes_prior = bool(include_prior)
    # The class name, not the registry key: `samplers.base._LIKELIHOOD_KINDS` maps class name ->
    # key, and it is what every numpy sampler already records in info["likelihood"] and what
    # `waic`/`predictive_metrics` read back to score the fit under the density it was fitted with.
    out.likelihood = type(like).__name__
    out.scatter_param = scatter_param
    out.log_likelihood = jax.jit(_log_likelihood) if jit else _log_likelihood
    return out


def _check_epochs(predict_jax, times, prior):
    """Run a model's host-side epoch check (``predict_jax.check_epochs``, a supernova on fixed
    diffusion epochs) on the concrete epochs a density is built for, under the prior it samples:
    inside the density the epochs may be traced, where no check can run."""
    check = getattr(predict_jax, "check_epochs", None)
    if check is not None:
        check(times, prior)


#: The priors the automatic density (and ``priors.log_prob_jax``) can express. A Normal's support
#: is the whole line (``(-inf, inf)``: the clip and the box test below are then no-ops); a Fixed
#: parameter's is its value.
SUPPORTED_PRIORS = ("Uniform", "LogUniform", "Normal", "TruncatedNormal", "Fixed")


def _box_bounds(prior, names, model_name):
    """Per-parameter support ``(low, high)`` arrays, refusing a prior this cannot express.

    Mirrors ``nuts_gpu._numpyro_priors``' refusal: a distribution whose support this cannot express
    raises rather than being quietly approximated by its bounds, because substituting a uniform for
    a log-uniform is a *different posterior*, not a different parameterisation.
    """
    lows, highs = [], []
    for nm in names:
        d = prior.distributions[nm]
        kind = family(d)
        if kind not in SUPPORTED_PRIORS:
            raise TypeError(
                f"the automatic density for model {model_name!r} cannot express prior {kind!r} on "
                f"parameter {nm!r}. Supported: {', '.join(SUPPORTED_PRIORS)}. Build log_prob_fn "
                f"yourself and pass it explicitly rather than letting the bounds stand in for the "
                f"density -- that changes the posterior silently.")
        lo, hi = (float(x) for x in d.bounds)
        lows.append(lo)
        highs.append(hi)
    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def free_density(log_prob_fn, prior, names):
    """Hold a prior's ``Fixed`` parameters at their values -> ``(density, free_names, fixed)``.

    ``log_prob_fn`` takes the full ``theta`` in ``names`` order; the returned density takes only
    the free columns, in ``free_names`` order, and puts each fixed value back in place before
    calling it, so the samplers never move a Fixed parameter and never count it (AIC and BIC use
    ``len(free_names)``). ``fixed`` is ``{name: value}``. With no Fixed parameter everything comes
    back unchanged. The density's attributes (``space``, ``likelihood``, ``includes_prior``,
    ``scatter_param``, ``log_likelihood`` -- wrapped the same way) travel with it.
    """
    names = list(names)
    fixed = {nm: float(prior.distributions[nm].value) for nm in names
             if family(prior.distributions[nm]) == "Fixed"}
    if not fixed:
        return log_prob_fn, names, {}
    from ...backends import require_jax
    _, jnp = require_jax("free_density")

    free = [nm for nm in names if nm not in fixed]
    mask = np.array([nm in fixed for nm in names])
    values = np.array([fixed.get(nm, 0.0) for nm in names])
    source = np.clip(np.cumsum(~mask) - 1, 0, None)        # full column -> free column

    def wrap(fn):
        def density(theta_free):
            theta_free = jnp.asarray(theta_free)
            full = jnp.where(mask, jnp.asarray(values, theta_free.dtype), theta_free[source])
            return fn(full)
        return density

    out = wrap(log_prob_fn)
    for attr in ("space", "includes_prior", "likelihood", "scatter_param"):
        if hasattr(log_prob_fn, attr):
            setattr(out, attr, getattr(log_prob_fn, attr))
    out.names = free
    if getattr(log_prob_fn, "log_likelihood", None) is not None:
        out.log_likelihood = wrap(log_prob_fn.log_likelihood)
    return out, free, fixed


def resolve_sampling_names(lc, model, prior, log_prob_fn, space, likelihood):
    """``(names, likelihood)`` for the samplers that take a scalar density.

    ``names`` is ``model.parameters``, **plus** the resolved likelihood's free-scatter column when
    the prior supplies it. That column is not a model parameter — it is a likelihood parameter, and
    ``GaussianLikelihoodWithScatter`` is the only likelihood that reads one — so without this the
    samplers would build a ``gaussian_scatter`` density and never sample the very parameter it
    exists to fit, silently returning a plain Gaussian fit (``sigma_extra`` defaults to 0). Nothing
    is appended when the prior does not carry the column: ``make_log_prob_jax`` then refuses with a
    message naming it, which is better than a ``KeyError`` out of ``_box_bounds``.

    The likelihood object is returned alongside so the caller can hand it back to
    ``resolve_density`` instead of building a second one from the same light curve.

    A caller-supplied ``log_prob_fn`` (or a ``model.log_prob_jax``) is left completely alone: for
    those, ``names`` is the *caller's* contract — ``model.parameters`` order — and appending a
    column would feed their function a theta of the wrong width.
    """
    names = list(model.parameters)
    if log_prob_fn is not None or getattr(model, "log_prob_jax", None) is not None:
        return names, likelihood

    from ...likelihood import make_likelihood

    like = (likelihood if not isinstance(likelihood, (str, type(None)))
            else make_likelihood(lc, kind=(likelihood or "auto"), space=space))
    sp = getattr(like, "scatter_param", None)
    if sp is not None and sp not in names and prior is not None and sp in prior.distributions:
        names.append(sp)
    return names, like


def resolve_density(lc, model, prior, log_prob_fn, space, names, *,
                    include_prior=False, sampler="this sampler", likelihood="auto"):
    """Shared resolution order for the three samplers that take a scalar ``log_prob_fn``.

    Returns ``(log_prob_fn, space, space_source, fn_source)`` — four elements, unchanged, because
    all three samplers unpack it positionally. The resolved likelihood travels on the returned
    function instead, as ``.likelihood``; a caller-supplied density carries no such attribute, and
    ``getattr(fn, "likelihood", None)`` is what the samplers record.

    Order, identical in ``nuts_gpu``, ``pymc_gpu`` and ``emcee_jax``:

    1. an explicitly passed ``log_prob_fn`` — always wins, so no existing call changes;
    2. ``model.log_prob_jax``, if a factory filled it;
    3. built here from ``model.predict_jax`` and the likelihood implied by ``space``.

    ``space_source`` records how much the recorded ``space`` can be trusted:

    ``"density"``
        auto-built — ``space`` *is* the space the residuals were formed in.
    ``"declared"``
        the caller passed both a density and a ``space``; we take their word for the metrics.
    ``"assumed"``
        the caller passed a density and no ``space``; we resolved one from ``lc.data_mode`` for
        the metrics and warn, because it may not be the space their density scores in.
    """
    from ...likelihood import resolve_space

    if log_prob_fn is not None:
        declared = space is not None and str(space).lower() != "auto"
        resolved = resolve_space(lc, space)
        if not declared:
            _warn_user(
                f"{sampler}: a hand-built log_prob_fn was passed without space=, so the space it "
                f"scores in is unknown. Predictive metrics and band metrics will be computed in "
                f"{resolved!r} space (inferred from lc.data_mode), which may not match. Pass "
                f"space= to state it and silence this.")
        return log_prob_fn, resolved, ("declared" if declared else "assumed"), "caller"

    from_model = getattr(model, "log_prob_jax", None)
    if from_model is not None:
        if include_prior:
            # `Model.log_prob_jax` is contractually a pure log-LIKELIHOOD (models/__init__.py:40).
            # emcee has no prior mechanism of its own, so handing it this slot unchanged would
            # impose a flat box and reinstate the exact defect the auto-build path exists to fix --
            # measured on a LogUniform tau: -17.2056 from this branch against a correct posterior
            # -26.0258, and a tau median of 258.78 (P(tau>100) = 0.810) against 29.02 (0.264).
            # Adding the prior here is possible but would silently redefine a public slot's
            # meaning, so refuse and say which sampler wants what.
            raise ValueError(
                f"{sampler} needs a log-POSTERIOR (it has no prior mechanism of its own), but "
                f"model {model.name!r} supplies `log_prob_jax`, which is contractually a pure "
                f"log-LIKELIHOOD. Using it here would impose a flat box in place of your prior. "
                f"Pass log_prob_fn= with the prior already included, or clear model.log_prob_jax "
                f"so a correct density is built from model.predict_jax.")
        declared = space is not None and str(space).lower() != "auto"
        return (from_model, resolve_space(lc, space),
                ("declared" if declared else "assumed"), "model.log_prob_jax")

    if getattr(model, "predict_jax", None) is None:
        raise ValueError(
            f"{sampler} needs log_prob_fn(theta) -> scalar, and model {model.name!r} carries "
            f"neither log_prob_jax nor predict_jax to build one from. Register it through a JAX "
            f"factory (register_kilonova / register_kilonova_two / register_tde / "
            f"register_supernova), or pass log_prob_fn= yourself.")

    fn = make_log_prob_jax(lc, model, prior, space=space, names=names,
                           include_prior=include_prior, likelihood=likelihood)
    return fn, fn.space, "density", "auto"


# ================================================================================ log_density
# The public log-posterior of one light curve. `make_log_prob_jax` closes over the light curve, so
# every alert is a new function and a new XLA compile (measured: 24.3 s for the first
# supernova alert, 3.3-4.2 s for each later one once the data were passed as arguments). Here the
# data travel as ARGUMENTS of one jitted program, padded to a bucket size, so every light curve of
# the same model, likelihood and bucket reuses one compile -- and `fit_batch` can stack them.

#: The smallest observation count :func:`log_density` pads a light curve up to.
MIN_BUCKET = 16

#: How many compiled programs :func:`log_density` keeps (least recently used are dropped).
_PROGRAM_CACHE_SIZE = 64
_PROGRAMS = {}


def bucket_size(n):
    """The padded observation count for ``n`` observations: 16, 24, 32, 48, 64, 96, 128, ...

    Powers of two and 1.5 times powers of two, so padding costs at most a third of an evaluation
    more than the light curve needs, and light curves in the same bucket share one compiled
    program (:func:`log_density`).

    Parameters
    ----------
    n : int
        Number of observations.

    Returns
    -------
    int
        The bucket, at least :data:`MIN_BUCKET`.

    Examples
    --------
    >>> from whisper_cbpf.samplers.jax._adapters import bucket_size
    >>> [bucket_size(n) for n in (3, 16, 17, 30, 59, 65)]
    [16, 16, 24, 32, 64, 96]
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"bucket_size: the number of observations must be >= 0; got {n}.")
    b = MIN_BUCKET
    while True:
        if n <= b:
            return b
        if n <= 3 * b // 2:
            return 3 * b // 2
        b *= 2


@dataclass(frozen=True, repr=False)
class LogDensity:
    """The log-posterior of one light curve under one model, as JAX functions (:func:`log_density`).

    Attributes
    ----------
    fn : callable
        ``fn(theta) -> log posterior`` (log prior + log likelihood; ``-inf`` outside the prior's
        support and past a model's constraint wall). ``theta`` is a flat array ordered by
        :attr:`names`. Differentiable, and usable inside ``jax.jit`` / ``jax.vmap`` / ``jax.grad``.
    names : list of str
        The sampled parameters: the model's, plus a free-scatter column when the likelihood fits
        one; a ``Fixed`` prior's parameters are held at their values and left out.
    lows, highs : numpy.ndarray
        The prior's support per parameter (``-inf`` / ``inf`` for a Normal).
    n_data : int
        Observations in the likelihood (padding excluded).
    log_likelihood : callable
        ``log_likelihood(theta)``, the same density without the prior (for AIC/BIC).
    shared : callable
        The jitted program ``shared(theta, data) -> (log posterior, log likelihood)``. It is the
        same object for every light curve with the same model, prior families, likelihood and
        bucket, which is what lets one compile serve them all.
    data : dict
        This light curve's arrays, padded to :attr:`bucket` (``mask`` is 0 on the padding), and
        the prior's numbers.
    bucket : int
        The padded observation count (:func:`bucket_size`), or ``n_data`` when the epochs are
        closed over.
    data_as_argument : bool
        True when the data are arguments of :attr:`shared`. False when the model needs concrete
        epochs (a supernova with ``diffusion_grid="data"``, a kilonova on redback's time grid) or
        float32 cannot hold the clock; :attr:`reason` says which. The program is then compiled
        once per light curve.
    reason : str
        Why the epochs are closed over; empty otherwise.
    model, space, likelihood : str
        The model's name, the residual space and the likelihood class.
    fixed : dict
        ``{name: value}`` of the parameters a ``Fixed`` prior holds.
    prior : Prior
        The prior used (with the explosion-time prior the data set, for a free ``t_exp``).
    pre_event : dict
        The pre-event rule applied to the light curve, as a fit records it in
        ``info["pre_event"]``: rows at or before the event are not in the density.
    """

    fn: object
    names: list
    lows: np.ndarray
    highs: np.ndarray
    n_data: int
    log_likelihood: object
    shared: object
    data: dict
    bucket: int
    data_as_argument: bool
    reason: str
    model: str
    space: str
    likelihood: str
    fixed: dict
    prior: object
    pre_event: dict = None

    def __call__(self, theta):
        return self.fn(theta)

    def __repr__(self):
        shared = (f"data as arguments: one compiled program serves every light curve of up to "
                  f"{self.bucket} points" if self.data_as_argument else
                  f"epochs closed over (one compile per light curve): {self.reason}")
        fixed = f", fixed {self.fixed}" if self.fixed else ""
        return (f"LogDensity(model={self.model!r}, {len(self.names)} parameters {self.names}"
                f"{fixed}, {self.n_data} points padded to {self.bucket}, log-posterior in "
                f"{self.space} space ({self.likelihood}); {shared})\n"
                f"Read next: .fn(theta) -> log posterior, .log_likelihood(theta), .names / .lows "
                f"/ .highs, and fit_batch(light_curves, model) to fit many alerts in one call.")


def _program_key(model, spec, closure):
    return (id(model.predict_jax), spec, closure)


def _cached_program(model, spec, closure=None, t_closure=None, band_closure=None):
    """The jitted ``shared(theta, data)`` for ``spec``, from a small LRU cache.

    Keyed on the identity of ``model.predict_jax`` (held in the entry, so the id cannot be reused
    while cached) and on the spec; ``closure`` is ``None`` for the data-as-argument program and the
    epochs' bytes otherwise.
    """
    key = _program_key(model, spec, closure)
    hit = _PROGRAMS.pop(key, None)
    if hit is None:
        hit = {"predict_jax": model.predict_jax,
               "fn": _build_program(model, spec, t_closure, band_closure), "probe": {}}
    _PROGRAMS[key] = hit                                   # most recent last
    while len(_PROGRAMS) > _PROGRAM_CACHE_SIZE:
        _PROGRAMS.pop(next(iter(_PROGRAMS)))
    return hit


def _build_program(model, spec, t_closure=None, band_closure=None):
    """``jax.jit(shared)``, ``shared(theta, data) -> (log posterior, log likelihood)``.

    ``data`` holds the observations (``t``, ``band``, ``y``, ``sigma``, ``mask`` = 0 on padding,
    ``w_det`` / ``w_ul`` = the mask split into detections and upper limits, ``sigma_ul``,
    ``log_norm``), the prior table (``prior``: per parameter low, high, mu, sigma and the log
    density's constant) and ``fixed`` (a Fixed parameter's value). ``t_closure`` / ``band_closure``
    replace ``data["t"]`` / ``data["band"]`` for a model that needs concrete epochs.

    The likelihood is whisper's, term for term (``likelihood/_jax.py``): the Gaussian on detections
    with the data's own normalising constant, and in flux space the censoring term
    ``log Phi((limit - model) / (limit / upper_limit_sigma))`` on upper limits. The prior is
    ``priors.log_prob_jax``'s, with its numbers read from the table instead of closed over.
    """
    from ...backends import require_jax
    from ...likelihood._numpy import _LN2PI, _MIN_FLUX_JY, _MIN_PROB

    jax, jnp = require_jax("log_density")
    from jax.scipy.special import ndtr

    families = spec["families"]
    fixed_mask = np.array([f == "Fixed" for f in families])
    any_fixed = bool(fixed_mask.any())
    source = np.clip(np.cumsum(~fixed_mask) - 1, 0, None)          # full column -> free column
    terms = [(i, f) for i, f in enumerate(families) if f != "Fixed"]
    perm_j = None if spec["perm"] is None else jnp.asarray(spec["perm"])
    predict_jax = model.predict_jax
    ok_fn = getattr(predict_jax, "constraint_ok", None)
    magnitude = spec["space"] == "magnitude"
    zeropoint = float(spec["zeropoint"])
    scatter_idx = spec["scatter_idx"]
    has_band = spec["has_band"]
    band_c = None if band_closure is None else jnp.asarray(band_closure)

    def in_space(flux):
        if not magnitude:
            return flux
        tiny = float(jnp.finfo(flux.dtype).tiny)      # _MIN_FLUX_JY underflows in float32
        floor = _MIN_FLUX_JY if _MIN_FLUX_JY > tiny else tiny
        return -2.5 * jnp.log10(jnp.clip(flux, floor, None) / zeropoint)

    def log_prior(x, table):
        out = 0.0
        for i, fam in terms:
            if fam == "Uniform":
                out = out + table[i, 4]
            elif fam == "LogUniform":
                out = out - jnp.log(x[i]) + table[i, 4]
            else:                                           # Normal, TruncatedNormal
                z = (x[i] - table[i, 2]) / table[i, 3]
                out = out - 0.5 * z * z + table[i, 4]
        return out

    def shared(theta, d):
        table = d["prior"]
        theta = jnp.asarray(theta, dtype=table.dtype)
        lo, hi = table[:, 0], table[:, 1]
        full = jnp.where(fixed_mask, d["fixed"], theta[source]) if any_fixed else theta
        # The physics never sees a value outside the support (a NaN cotangent there would poison
        # a gradient); in support the clip is the identity (as in make_log_prob_jax).
        safe = jnp.clip(full, lo, hi)
        th = safe if perm_j is None else safe[perm_j]
        # float64 numpy epochs when closed over: the model subtracts its t_exp_days on the host
        #; traced epochs otherwise.
        t = d["t"] if t_closure is None else t_closure
        if has_band:
            flux = predict_jax(th, t, d["band"] if band_c is None else band_c)
        else:
            flux = predict_jax(th, t)
        inside = jnp.all((full >= lo) & (full <= hi))
        if ok_fn is not None:                               # redback's Constraint wall (1.6)
            inside = inside & ok_fn(th)
        m = in_space(flux)
        if scatter_idx is not None:
            var = d["sigma"] ** 2 + safe[scatter_idx] ** 2
            ll = -0.5 * jnp.sum(d["mask"] * ((d["y"] - m) ** 2 / var + _LN2PI + jnp.log(var)))
        else:
            res = (d["y"] - m) / d["sigma"]
            ll = -0.5 * jnp.sum(d["w_det"] * res * res) + d["log_norm"]
            if not magnitude:                               # upper limits exist in flux only
                prob = ndtr((d["y"] - m) / d["sigma_ul"])
                ll = ll + jnp.sum(d["w_ul"] * jnp.log(jnp.clip(prob, _MIN_PROB, 1.0 - _MIN_PROB)))
        ll = jnp.where(inside, ll, -jnp.inf)
        return jnp.where(inside, ll + log_prior(safe, table), -jnp.inf), ll

    return jax.jit(shared)


def _prior_table(prior, names):
    """``(n, 5)`` float64: low, high, mu, sigma and the log density's constant per parameter."""
    import math

    rows = []
    for nm in names:
        d = prior.distributions[nm]
        kind = family(d)
        lo, hi = (float(x) for x in d.bounds)
        if kind == "Uniform":
            rows.append((lo, hi, 0.0, 1.0, -math.log(hi - lo)))
        elif kind == "LogUniform":
            rows.append((lo, hi, 0.0, 1.0, -math.log(math.log(hi) - math.log(lo))))
        elif kind == "Normal":
            rows.append((lo, hi, float(d.mu), float(d.sigma),
                         -math.log(float(d.sigma)) - 0.5 * math.log(2.0 * math.pi)))
        elif kind == "TruncatedNormal":
            rows.append((lo, hi, float(d.mu), float(d.sigma),
                         -math.log(float(d.sigma)) - 0.5 * math.log(2.0 * math.pi)
                         - float(d._log_mass)))
        else:                                               # Fixed: support is its value
            v = float(d.value)
            rows.append((v, v, 0.0, 1.0, 0.0))
    return np.asarray(rows, dtype=float)


def _alert_arrays(lc, like, band, prior, names, bucket):
    """This light curve's likelihood arrays, padded to ``bucket`` rows, as float64 numpy.

    The padding repeats the LATEST epoch (its time and band), so a model whose value at one epoch
    depends on the call's last epoch (the supernovae size their engine grid from it, as redback
    does) gives the real rows exactly the numbers it gives the unpadded light curve; ``mask`` is 0
    there, so the padding adds nothing to the likelihood. No NaN is left anywhere: an upper limit's
    NaN error bar is replaced by 1 where it is masked out of the Gaussian term.
    """
    y = np.asarray(like.y, dtype=float)
    sigma = np.asarray(like.sigma, dtype=float)
    n = int(y.size)
    t = np.asarray(lc.time, dtype=float)
    if t.size != n:
        raise ValueError(
            f"log_density: the likelihood holds {n} rows but the light curve has {t.size} epochs, "
            f"so rows and epochs cannot be matched. Build the likelihood from this light curve.")
    det = np.asarray(getattr(like, "detections", np.ones(n, dtype=bool)), dtype=bool)
    ul_sigma = float(getattr(like, "upper_limit_sigma", 3.0))
    sig = np.where(det, sigma, 1.0)
    sig_ul = np.where(det, 1.0, y / ul_sigma)
    # read, not recomputed: the likelihood object's own constant (detections only when it has
    # upper limits), as likelihood/_jax.py does, so the two agree to the last digit.
    log_norm = getattr(like, "_log_norm_det", None)
    log_norm = float(getattr(like, "_log_norm", 0.0) if log_norm is None else log_norm)
    pad = int(bucket) - n
    last = int(np.argmax(t)) if n else 0

    def padded(a, fill=None):
        a = np.asarray(a)
        extra = np.repeat(a[last:last + 1] if fill is None else np.asarray([fill], a.dtype), pad)
        return np.concatenate([a, extra]) if pad else a

    mask = padded(np.ones(n), 0.0)
    detf = det.astype(float)
    return {"t": padded(t), "band": padded(np.asarray(band, dtype=np.int32)),
            "y": padded(y), "sigma": padded(sig), "sigma_ul": padded(sig_ul),
            "mask": mask, "w_det": mask * padded(detf, 0.0), "w_ul": mask * padded(1.0 - detf, 0.0),
            "log_norm": np.asarray(log_norm),
            "prior": _prior_table(prior, names),
            "fixed": np.asarray([float(prior.distributions[nm].value)
                                 if family(prior.distributions[nm]) == "Fixed" else 0.0
                                 for nm in names], dtype=float)}


def _to_device(arrays, jnp, dt):
    return {k: jnp.asarray(v, dtype=(jnp.int32 if k == "band" else dt)) for k, v in arrays.items()}


def log_density(lc, model, *, space="auto", likelihood="auto", prior=None, bucket="auto"):
    """The log-posterior of ``lc`` under ``model`` as a JAX function, with its names and box.

    One call gives what a sampler, an optimiser or a profiler needs: ``fn(theta) -> log
    posterior`` (differentiable; ``-inf`` outside the prior's support and past a model's
    constraint wall), the parameter names, the prior's support and the number of data points. The
    density is the one ``wp.fit`` would sample for this light curve: the same rows (pre-event data
    are left out, :func:`~whisper_cbpf.samplers.base.prepare_lc`), the same explosion-time prior
    for a free ``t_exp``, the same likelihood (``space`` / ``likelihood`` as there), and the prior
    density is the prior's own, normalising constants included.

    The data are ARGUMENTS of one compiled program, not constants inside it: the observations are
    padded to a bucket size (16, 24, 32, 48, 64, ...; :func:`bucket_size`) and the prior's numbers
    are passed as a table. So every light curve of the same model, prior families, likelihood and
    bucket reuses one compile, and ``fit_batch`` stacks them into one call. A model that needs
    concrete epochs (a supernova on ``diffusion_grid="data"``, a kilonova on redback's time grid)
    or a float32 session on a clock float32 cannot hold falls back to closing the epochs over,
    which compiles once per light curve; ``.data_as_argument`` and ``.reason`` say so.

    Parameters
    ----------
    lc : LightCurve
        The data. With ``space="auto"`` upper limits after the event are fitted in flux space with
        the censored likelihood, as by ``wp.fit``.
    model : str or Model
        A model with ``predict_jax`` (the JAX factories, ``flare_jax``).
    space : {"auto", "flux", "magnitude"}
        The residual space; ``"auto"`` follows ``lc.data_mode``.
    likelihood : str or likelihood object
        ``"auto"``, ``"gaussian"``, ``"upper_limits"`` or ``"gaussian_scatter"`` (its
        ``scatter_param``, default ``"sigma"``, must be in the prior), or a built object.
    prior : Prior, optional
        Default: the model's, with the explosion-time prior the data set when ``t_exp`` is free
        and the prior does not name it (as in ``wp.fit``). Uniform, LogUniform, Normal,
        TruncatedNormal and Fixed.
    bucket : "auto", int or None
        Pad the observations to this count; ``"auto"`` is :func:`bucket_size`, ``None`` no
        padding. Passing the same bucket for several light curves lets them share a program.

    Returns
    -------
    LogDensity
        ``.fn``, ``.names``, ``.lows``, ``.highs``, ``.n_data``, plus ``.log_likelihood``,
        ``.shared(theta, data)``, ``.data``, ``.bucket``, ``.data_as_argument``.

    Raises
    ------
    ValueError
        The model has no ``predict_jax``, no prior is available, the prior misses a parameter, or
        ``bucket`` is smaller than the light curve.
    NotImplementedError
        A likelihood the JAX backend does not implement (the mixture).

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> flare = wp.get_model("flare_jax")
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flare.predict(truth, t),
    ...                    flux_err=np.full(30, 0.1))
    >>> ld = wp.log_density(lc, "flare_jax")
    >>> ld.names, ld.n_data, ld.bucket, ld.data_as_argument
    (['log_amp', 'log_sigma', 'log_tau', 't0'], 30, 32, True)
    >>> bool(ld.fn(np.array([1.0, 0.5, 1.5, 10.0])) > ld.fn(np.array([0.0, 0.5, 1.5, 10.0])))
    True

    A second light curve of the same size runs on the same compiled program:

    >>> ld2 = wp.log_density(lc.where(time_min=2.0), "flare_jax")     # 28 points, same bucket
    >>> ld2.shared is ld.shared
    True
    """
    from ...backends import require_jax
    from ...likelihood._jax import log_likelihood_jax
    from ...models import get_model
    from ..base import _pre_event_plan, check_not_empty
    from ._diagnostics import F32_ABS_LIMIT

    jax, jnp = require_jax("log_density")
    model = get_model(model)
    check_not_empty(lc)
    plan = _pre_event_plan(lc, model, prior, likelihood=likelihood)
    plan.warn()
    lc = plan.lc
    if plan.prior is not None:
        prior = plan.prior
    if getattr(model, "predict_jax", None) is None:
        raise ValueError(
            f"log_density: model {model.name!r} has no predict_jax, so there is no JAX forward map "
            f"to build a density from. Use a JAX model: a factory (supernova_model, tde_model, "
            f"kilonova_model, ...) or 'flare_jax'.")
    if getattr(model, "log_prob_jax", None) is not None:
        raise ValueError(
            f"log_density: model {model.name!r} supplies its own log_prob_jax, a likelihood closed "
            f"over one data set, which cannot take the data as arguments or carry the prior. Clear "
            f"model.log_prob_jax so the density is built from model.predict_jax.")
    prior = prior if prior is not None else model.default_prior
    if prior is None:
        raise ValueError(f"log_density: no prior for model {model.name!r}; pass prior=Prior({{...}}).")
    names, like = resolve_sampling_names(lc, model, prior, None, space, likelihood)
    missing = [nm for nm in names if nm not in prior.distributions]
    if missing:
        raise ValueError(f"log_density: the prior has no distribution for {missing}; model "
                         f"{model.name!r} needs one for each of {list(names)}.")
    try:
        loglik = log_likelihood_jax(like)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{exc}\n\nlog_density built {type(like).__name__} from this light curve. Pass "
            f"likelihood='gaussian', 'upper_limits' or 'gaussian_scatter', or drop the rows that "
            f"force this choice.") from None
    scatter = getattr(loglik, "scatter_param", None)
    if scatter is not None and scatter not in names:
        raise ValueError(
            f"log_density: {type(like).__name__} fits a free scatter on {scatter!r}, which is not "
            f"in the prior. Add {scatter!r} to the prior, or use likelihood='gaussian'.")
    lows, highs = _box_bounds(prior, names, model.name)
    families = tuple(family(prior.distributions[nm]) for nm in names)
    fixed = {nm: float(prior.distributions[nm].value) for nm, f in zip(names, families)
             if f == "Fixed"}
    free = [nm for nm in names if nm not in fixed]
    if not free:
        raise ValueError(f"log_density: every parameter is Fixed ({fixed}); there is nothing to "
                         f"infer. Evaluate model.predict at those values instead.")
    perm = _gather_permutation(names, model.parameters, model.name)
    band_of = getattr(model.predict_jax, "band_index", None)
    n = int(len(lc.time))
    band = (np.zeros(n, dtype=int) if band_of is None
            else np.asarray(band_of(np.asarray(lc.band)), dtype=int))
    dt = float_dtype()
    _check_epochs(model.predict_jax, np.asarray(lc.time, dtype=float), prior)
    spec_items = (("families", families), ("perm", None if perm is None else tuple(perm)),
                  ("scatter_idx", None if scatter is None else names.index(scatter)),
                  ("space", like.space), ("zeropoint", float(like.zeropoint_jy)),
                  ("has_band", band_of is not None), ("names", tuple(names)),
                  ("dtype", str(np.dtype(dt))))
    t = np.asarray(lc.time, dtype=float)

    if bucket == "auto":
        width = bucket_size(n)
    elif bucket is None:
        width = n
    else:
        width = int(bucket)
        if width < n:
            raise ValueError(f"log_density: bucket={width} is smaller than the light curve's {n} "
                             f"observations. Pass bucket >= {n}, 'auto' or None.")

    reason = ""
    if dt != jnp.float64 and t.size and float(np.nanmax(np.abs(t))) >= F32_ABS_LIMIT:
        reason = (f"float32 session and lc.time reaches {float(np.nanmax(np.abs(t))):.6g}: the "
                  f"epochs are closed over as float64 so the model subtracts its t_exp_days on "
                  f"the host. Enable float64 (JAX_ENABLE_X64=1) to share one program.")
    entry = None
    if not reason:
        entry = _cached_program(model, _Frozen(spec_items))
        arrays = _alert_arrays(lc, like, band, prior, names, width)
        data = _to_device(arrays, jnp, dt)
        probe = entry["probe"].get(width)
        if probe is None:
            try:
                jax.eval_shape(entry["fn"], jax.ShapeDtypeStruct((len(free),), dt), data)
                probe = ""
            except TypeError as exc:           # the tracer-conversion errors are TypeErrors
                first = (str(exc).strip().splitlines() or [""])[0][:200]
                probe = (f"model {model.name!r} needs concrete epochs ({type(exc).__name__}: "
                         f"{first})")
            entry["probe"][width] = probe
        reason = probe
    if reason:
        closure = (t.tobytes(), band.tobytes())
        entry = _cached_program(model, _Frozen(spec_items), closure, t_closure=t,
                                band_closure=band)
        width = n
        data = _to_device(_alert_arrays(lc, like, band, prior, names, n), jnp, dt)
    program = entry["fn"]
    barrier = jax.lax.optimization_barrier

    # The barrier keeps XLA from constant-folding the model on the data when a caller jits `fn`
    # (the data are then constants of the caller's program), so that program compiles like the
    # one that takes the data as arguments.
    def fn(theta):
        return program(theta, barrier(data))[0]

    def log_likelihood(theta):
        return program(theta, barrier(data))[1]

    free_idx = [names.index(nm) for nm in free]
    return LogDensity(fn=fn, names=free, lows=np.asarray(lows)[free_idx],
                      highs=np.asarray(highs)[free_idx], n_data=n, log_likelihood=log_likelihood,
                      shared=program, data=data, bucket=int(width), data_as_argument=not reason,
                      reason=reason, model=model.name, space=like.space,
                      likelihood=type(like).__name__, fixed=fixed, prior=prior,
                      pre_event=dict(plan.rule))


class _Frozen(tuple):
    """A hashable spec: ``((key, value), ...)`` read back by name inside ``_build_program``."""

    def __getitem__(self, key):
        if isinstance(key, str):
            for k, v in tuple(self):
                if k == key:
                    return v
            raise KeyError(key)
        return tuple.__getitem__(self, key)
