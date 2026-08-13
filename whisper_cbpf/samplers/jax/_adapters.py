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

import numpy as np

__all__ = ["MAX_UNCHUNKED_BATCH", "float_dtype", "resolve_band_index",
           "make_batched_predict_jax", "make_log_prob_jax", "resolve_density"]

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
        core on the observation times and cannot accept a traced one.
    chunk : int, optional
        Evaluate the batch in blocks of this width via ``jax.lax.map`` instead of one wide vmap.
        Leave ``None`` when the caller already bounds the batch (``abc_gpu`` does, via its own
        ``chunk``), so no trip-count-1 scan is added.
    max_batch : int, optional
        Refuse an unchunked batch wider than this. ``None`` disables the guard.
    """
    from ...backends import require_jax
    jax, jnp = require_jax("make_batched_predict_jax")

    names = list(names) if names is not None else list(model.parameters)
    perm = _gather_permutation(names, model.parameters, getattr(model, "name", "?"))
    bidx = resolve_band_index(lc, model)
    t_source = np.asarray(lc.time if times is None else times, dtype=float)
    t_ref = jnp.asarray(t_source, dtype=float_dtype())
    predict_jax = model.predict_jax
    perm_j = None if perm is None else jnp.asarray(perm)

    def one(theta):
        th = theta if perm_j is None else theta[perm_j]
        return predict_jax(th, t_ref) if bidx is None else predict_jax(th, t_ref, bidx)

    vmapped = jax.jit(jax.vmap(one))

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
                    f"times. Pass chunk=<width> (abc_gpu's own default is 16), or raise max_batch "
                    f"if your forward model is analytic and has no wavelength axis.")
            return vmapped(theta_2d)
        return _mapped_in_chunks(jax, jnp, vmapped, theta_2d, int(chunk))

    batched.band_index = bidx
    batched.names = names
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
    # only concrete arrays survive. This used to `return` from inside the except, so under a
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
    # It used to be 1e-5, which is near-inert on MJD-valued times -- exactly what whisper's own
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


def _mapped_in_chunks(jax, jnp, vmapped, theta_2d, width):
    """``lax.map`` ``vmapped`` over ``ceil(B/width)`` padded blocks, then trim.

    Constant memory, linear cost, ONE compile — the shape ``abc_gpu`` uses. Padding rather than a
    ragged tail keeps every block the same shape, so there is exactly one compiled kernel.
    """
    b = int(theta_2d.shape[0])
    if b == 0:                                   # an ABC round can legitimately accept nothing;
        return vmapped(theta_2d)                 # reshape(0, -1) cannot infer the trailing dim
    n_blocks = -(-b // width)
    pad = n_blocks * width - b
    if pad:
        theta_2d = jnp.concatenate([theta_2d, jnp.repeat(theta_2d[-1:], pad, axis=0)], axis=0)
    blocks = theta_2d.reshape(n_blocks, width, theta_2d.shape[-1])
    out = jax.lax.map(vmapped, blocks)
    return out.reshape(n_blocks * width, out.shape[-1])[:b]


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
    likelihood : object, optional
        A pre-built whisper likelihood. Defaults to ``make_likelihood(lc, space=space)``.

    The returned function carries ``.space``, ``.names``, ``.includes_prior`` and
    ``.log_likelihood`` (the ``include_prior=False`` twin, so a caller that sampled the posterior
    can still rank draws by likelihood for AIC/BIC).

    Notes
    -----
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
    # A prior-only column is fine for the FORWARD MAP (make_batched_predict_jax drops it, which is
    # the scatter_param case). It is NOT fine here: nothing downstream consumes it -- the JAX
    # likelihood takes model_flux alone and refuses GaussianLikelihoodWithScatter -- so the density
    # would be exactly CONSTANT in that parameter. Measured: logL = 71.834030 identically at
    # sigma = 0.02, 0.1 and 0.4, with d(logL)/d(sigma) == 0.0, where the CPU twin gives 63.516 /
    # 32.737 / -0.094. The sampler would converge and report the PRIOR as that parameter's
    # marginal, labelled as a fit. Refuse, as the missing-parameter case does.
    extra = [n for n in names if n not in set(model.parameters)]
    if extra:
        raise ValueError(
            f"parameter(s) {extra} are in the sampling order but not in model {model.name!r}, so "
            f"the automatic density would be exactly constant in them and their posterior would be "
            f"the prior. This is what a `scatter_param` looks like here: the JAX likelihood scores "
            f"model flux only. Drop them from the prior for this sampler, or pass log_prob_fn= "
            f"with a density that actually uses them.")

    like = likelihood if likelihood is not None else make_likelihood(lc, space=space)
    try:
        loglik = log_likelihood_jax(like)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{exc}\n\nThis density was built automatically from the light curve. "
            f"`make_likelihood(lc, space={space!r})` chose {type(like).__name__}, most often "
            f"because the light curve carries upper limits. Either drop them "
            f"(`lc.select_detections()` or an equivalent mask) so a plain Gaussian is correct, or "
            f"build `log_prob_fn` yourself and pass it explicitly.") from None

    bidx = resolve_band_index(lc, model)
    t_ref = jnp.asarray(np.asarray(lc.time if times is None else times, dtype=float),
                        dtype=float_dtype())
    predict_jax = model.predict_jax
    dt = float_dtype()
    lo_j, hi_j = jnp.asarray(lows, dtype=dt), jnp.asarray(highs, dtype=dt)
    perm_j = None if perm is None else jnp.asarray(perm)

    def _log_likelihood(theta):
        theta = jnp.asarray(theta, dtype=dt)
        safe = jnp.clip(theta, lo_j, hi_j)              # never evaluate the physics outside the box
        th = safe if perm_j is None else safe[perm_j]
        flux = predict_jax(th, t_ref) if bidx is None else predict_jax(th, t_ref, bidx)
        inside = jnp.all((theta >= lo_j) & (theta <= hi_j))
        return jnp.where(inside, loglik(flux), -jnp.inf)

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
    out.log_likelihood = jax.jit(_log_likelihood) if jit else _log_likelihood
    return out


def _box_bounds(prior, names, model_name):
    """Per-parameter ``(low, high)`` arrays, refusing anything that is not a box.

    Mirrors ``nuts_gpu._numpyro_priors``' refusal: a distribution whose support this cannot express
    raises rather than being quietly approximated by its bounds, because substituting a uniform for
    a log-uniform is a *different posterior*, not a different parameterisation.
    """
    lows, highs = [], []
    for nm in names:
        d = prior.distributions[nm]
        kind = type(d).__name__
        if kind not in ("Uniform", "LogUniform"):
            raise TypeError(
                f"the automatic density for model {model_name!r} cannot express prior {kind!r} on "
                f"parameter {nm!r}. Supported: Uniform, LogUniform. Build log_prob_fn yourself and "
                f"pass it explicitly rather than letting the bounds stand in for the density -- "
                f"that changes the posterior silently.")
        lo, hi = (float(x) for x in d.bounds)
        lows.append(lo)
        highs.append(hi)
    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def resolve_density(lc, model, prior, log_prob_fn, space, names, *,
                    include_prior=False, sampler="this sampler"):
    """Shared resolution order for the three samplers that take a scalar ``log_prob_fn``.

    Returns ``(log_prob_fn, space, space_source, fn_source)``.

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
            import warnings
            warnings.warn(
                f"{sampler}: a hand-built log_prob_fn was passed without space=, so the space it "
                f"scores in is unknown. Predictive metrics and band metrics will be computed in "
                f"{resolved!r} space (inferred from lc.data_mode), which may not match. Pass "
                f"space= to state it and silence this.", stacklevel=3)
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
                           include_prior=include_prior)
    return fn, fn.space, "density", "auto"
