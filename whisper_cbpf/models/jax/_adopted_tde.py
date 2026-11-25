"""Whisper adapter shared by the two redback TDE models bound in :mod:`.tde_fallback` and
:mod:`.tde_analytical`.

Both turn ``(parameters, epochs, bands)`` into a flux the way the other photometric factories of
:mod:`whisper_cbpf.models.jax._factories` do (same context handling, same ``Model`` contract), so
this module only holds what the two have in common: the picklable predict object and the factory
body. Each physics module supplies four things, found by name from its ``kind``:

``PARAMETERS``
    every parameter of the redback model, in redback's order (pinned ones included);
``default_prior()``
    ``(Prior, pinned)``: redback's own prior file, read from redback, with a transcription for when
    redback is absent;
``<kind>_ab_magnitude(tau_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, **kw)``
    the traced magnitude at SOURCE-frame days ``tau_days`` since the model's time zero;
``<kind>_ab_magnitude(..., epoch_idx=None)`` (optional)
    ``(first, inv)`` int32, the distinct-epoch map of :func:`epoch_index`. A module accepts it only
    if a row's output depends on the epochs through the row's own ``tau`` and through the SET of
    epochs (the end of the dense grid), never through a count or an order of rows.

redback's ``Constraint`` priors for a model are in :mod:`whisper_cbpf.models.constraints` (keyed
by ``kind``), and are armed with ``set_constraint`` like the other factories'.

**Every distinct epoch is evaluated once, directly.** There is no fixed epoch grid and no
interpolation of the bolometric curve: the viscous or diffusion integral is computed at each epoch,
from a dense engine grid whose end follows the latest epoch (redback sizes it from its last epoch
too). When ``<kind>_ab_magnitude`` takes ``epoch_idx=`` (``tde_fallback``, ``tde_mosfit``),
observations that share a time (one epoch in several bands) share one evaluation. The engine, the
viscous integral and the photosphere run at the distinct epochs and are copied to their
observations before the band integral, as redback's and MOSFiT's ``Viscous`` do (``np.unique`` of
the times, then ``searchsorted``).

The map is built on the host from concrete times (:func:`epoch_index`), or arrives as data with
traced ones (``predict_jax(..., epoch_idx=)``, which
:func:`~whisper_cbpf.samplers.jax._adapters.log_density` passes). Traced times without a map are
evaluated per observation. ``unique_epochs=False`` at the factory evaluates every observation. The
epochs, the explosion time and the redshift remain ordinary traced arguments: one compiled function
serves any light curve with the same numbers of observations and of distinct epochs, and
``free=["t_exp", "redshift"]`` costs nothing extra.

Why the map is exact: equal times give equal ``tau`` (the same operations on equal operands), the
one quantity that mixes epochs (the end of the dense grid) is a maximum over the set of epochs,
which the distinct epochs leave unchanged, and everything else is computed row by row. ``np.unique``
merges ``+0.0`` and ``-0.0``, and all NaNs, and both merges are exact too: ``+0.0`` and ``-0.0``
give the same ``tau`` whenever a nonzero time is subtracted from them (``t_exp_days`` or a free
``t_exp``), and otherwise both are at the time zero, not alive, at ``mag_floor``; a NaN row is
never alive, at ``mag_floor``, whichever row is evaluated.
"""
from __future__ import annotations

import importlib
import inspect

import numpy as np

from ...io.photometry import AB_ZEROPOINT_JY
from ._factories import (_BandIndex, _PhotometricPredict, _PredictJax, _check_context, _check_free,
                         _constraint_note, _context, _days_since, _filter_set_dict, _floor_note,
                         _free_distributions, _free_note)

__all__ = ["DirectTDEPredict", "assemble", "epoch_index", "geometric_dense_times", "require_x64"]

#: First node of redback's engine grid for the interaction processes, days
#: (``np.geomspace(1e-5, time[-1] + 100, 1000)``), and the padding after the last epoch.
GRID_START = 1e-5
GRID_PAD = 100.0


def require_x64(what):
    """Raise, with the fix, unless ``jax_enable_x64`` is on. Resolved at trace time."""
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} requires float64: the bolometric luminosity is 1e43 to 1e46 erg/s in cgs, past "
            f"float32's 3.4e38 ceiling, so the light curve would come back at mag_floor. Enable it "
            f"BEFORE the first jax array is created:\n"
            f"    import jax; jax.config.update('jax_enable_x64', True)\n"
            f"or set JAX_ENABLE_X64=1 in the environment.")


def geometric_dense_times(t_last, n=1000):
    """redback's engine grid, ``geomspace(1e-5, t_last + 100, n)``, built so ``t_last`` may be traced.

    The endpoints are set exactly, so the grid is geometric to round-off and
    :func:`whisper_cbpf.models.jax.tde.diffusion` may find segments by arithmetic
    (``geometric=True``).
    """
    import jax.numpy as jnp

    u = jnp.linspace(0.0, 1.0, n)
    top = t_last + GRID_PAD
    dense = jnp.exp(np.log(GRID_START) + u * (jnp.log(top) - np.log(GRID_START)))
    return dense.at[0].set(GRID_START).at[-1].set(top)


def epoch_index(times, size=None):
    """The distinct-epoch map of one light curve's concrete epochs: ``(first, inv)``, int32.

    ``times[first]`` are the distinct epochs in increasing order (NaN last), ``first[k]`` is the
    first row holding epoch ``k``, and ``inv[i]`` is row ``i``'s slot, so ``times[first][inv]`` is
    ``times`` (NaN where NaN). Equal values are one epoch: ``+0.0`` and ``-0.0`` merge, and so do
    all NaNs (the module docstring says why that is exact). It is the structure of redback's and
    MOSFiT's ``Viscous`` (``np.unique``, then ``searchsorted``) and of the ``(uniq_times,
    gather_idx)`` of :func:`whisper_cbpf.models.jax.tde.build_interaction_grid`.

    Parameters
    ----------
    times : array_like, 1-D
        One light curve's epochs, on any clock: only equality matters.
    size : int or None
        Pad ``first`` to this length by repeating its last entry, an epoch already present (it
        changes no maximum and is never gathered back), so that light curves with different
        counts share a shape. ``None``: exactly the distinct count.

    Returns
    -------
    first : numpy.ndarray of int32, shape ``(size,)``, or ``(n_distinct,)`` for ``size=None``
    inv : numpy.ndarray of int32, shape ``(len(times),)``

    Raises
    ------
    ValueError
        ``times`` is not 1-D (call once per light curve, with a common ``size``); ``size`` is
        smaller than the distinct count; or ``size > 0`` for no epochs at all.

    Examples
    --------
    >>> from whisper_cbpf.models.jax._adopted_tde import epoch_index
    >>> epoch_index([5.0, 9.0, 5.0], size=3)
    (array([0, 1, 1], dtype=int32), array([0, 1, 0], dtype=int32))
    """
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"epoch_index: times must be 1-D, one light curve; got shape {t.shape}. "
                         f"Call it once per light curve with a common size=, and stack the maps.")
    _, first, inv = np.unique(t, return_index=True, return_inverse=True)
    # numpy 2 returns the inverse in the input's shape; one light curve is 1-D either way
    first, inv = first.astype(np.int32), np.asarray(inv).reshape(-1).astype(np.int32)
    if size is not None:
        size = int(size)
        if size < first.size:
            raise ValueError(f"epoch_index: size={size} is smaller than the {first.size} distinct "
                             f"epochs. Pass size >= {first.size}, or None for exactly {first.size}.")
        if size > first.size:
            if not first.size:
                raise ValueError(f"epoch_index: no epochs to pad to size={size}. An empty light "
                                 f"curve has no map; pass size=None.")
            first = np.concatenate([first, np.repeat(first[-1:], size - first.size)])
    return first, inv


def _at_epochs(x, epoch_idx):
    """``x`` at the distinct epochs (``x[first]``), or ``x`` itself without a map. Traced."""
    return x if epoch_idx is None else x[epoch_idx[0]]


def _to_rows(x, epoch_idx):
    """A per-epoch ``x`` copied back onto the observations (``x[inv]``), or ``x`` itself. Traced."""
    return x if epoch_idx is None else x[epoch_idx[1]]


def _guard_epochs(mag, tau, epoch_idx):
    """``mag`` unchanged, or NaN everywhere when ``epoch_idx`` does not describe ``tau``. Traced.

    The check is ``tau[first][inv] == tau`` (a NaN row matches a NaN), so a map from other epochs
    can never give a silently wrong light curve. The NaN is made only when the map is wrong
    (``inf - inf``), never as a constant: with a right map nothing in the evaluation is NaN, so
    ``jax_debug_nans`` stays quiet, jitted, eager or vmapped; the value and its cotangent pass
    through ``where`` unchanged.
    """
    if epoch_idx is None:
        return mag
    import jax.numpy as jnp

    first, inv = epoch_idx
    if tuple(inv.shape) != tuple(tau.shape):
        raise ValueError(f"epoch_idx: inv has shape {tuple(inv.shape)} for {tuple(tau.shape)} "
                         f"observations; build it with predict_jax.epoch_index(times[, size]).")
    back = tau[first][inv]
    ok = jnp.all((back == tau) | (jnp.isnan(back) & jnp.isnan(tau)))
    bad = jnp.where(ok, jnp.zeros((), mag.dtype), jnp.asarray(jnp.inf, mag.dtype))
    return jnp.where(ok, mag, bad - bad)


def _check_epoch_idx(times, first, inv):
    """Raise ``ValueError`` when concrete ``times`` and a concrete map disagree; a traced one is
    left to the trace's guard (:func:`_guard_epochs`)."""
    try:
        t = np.asarray(times, dtype=np.float64)
        f = np.asarray(first)
        i = np.asarray(inv)
    except TypeError:                     # a tracer: TracerArrayConversionError is a TypeError
        return
    hint = ("epoch_idx does not describe these times: build it with "
            "predict_jax.epoch_index(times[, size]) from the same array")
    if (t.ndim != 1 or f.ndim != 1 or i.shape != t.shape
            or not (np.issubdtype(f.dtype, np.integer) and np.issubdtype(i.dtype, np.integer))):
        raise ValueError(f"{hint} (times {t.shape}, first {f.shape} {f.dtype}, inv {i.shape} "
                         f"{i.dtype}; inv must have one entry per observation).")
    if t.size == 0:
        return
    if f.size == 0 or f.min() < 0 or f.max() >= t.size or i.min() < 0 or i.max() >= f.size:
        raise ValueError(f"{hint} (an index is out of range: first must index the {t.size} "
                         f"observations and inv the {f.size} slots of first).")
    back = t[f][i]
    if not np.all((back == t) | (np.isnan(back) & np.isnan(t))):
        raise ValueError(f"{hint} (times[first][inv] differs from times at "
                         f"{int(np.sum(~((back == t) | (np.isnan(back) & np.isnan(t)))))} rows).")


def _physics(kind):
    return importlib.import_module(f"{__package__}.{kind}")


class DirectTDEPredict(_PhotometricPredict):
    """``predict``/``predict_jax`` of :func:`assemble`'s models. See :class:`_PhotometricPredict`."""

    def __init__(self, name, params, band_index, *, kind, pinned, weights, norms, lam, redshift,
                 dl_cm, xi_mw, xi_host, common, t_exp_days=0.0, free=(), constraint=None,
                 unique_epochs=False):
        super().__init__(name, params, band_index)
        if unique_epochs:
            # set only when True: a per-observation model keeps the state (and so the
            # `model_record` identity) it had before the option existed; every read is a getattr
            self.unique_epochs = True
        self.kind = str(kind)
        self.free = tuple(free)
        self.t_exp_days = float(t_exp_days)
        self.pinned = dict(pinned)
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = None if redshift is None else float(redshift)
        self.dl_cm = None if dl_cm is None else float(dl_cm)
        self.xi_mw = None if xi_mw is None else np.asarray(xi_mw)
        self.xi_host = None if xi_host is None else np.asarray(xi_host)
        self.common = {k: v for k, v in common.items() if k not in ("xi_mw", "xi_host")}
        self.set_constraint(self.kind, constraint, redshift=redshift)

    def _physics_call(self, rel_days, bidx, free, **extra):
        """Observer-frame days since the time zero -> AB magnitude per observation. Traced.

        ``extra`` goes to the physics function unchanged (``epoch_idx=``). A physics function that
        takes ``geometric`` gets ``geometric=False`` unless the state (``common``) or ``extra``
        holds it: the factories store it only when True."""
        import jax.numpy as jnp

        c = _context(self, free)
        v = c.values
        v.update(self.pinned)                  # pinned values are compile-time constants
        common = dict(self.common,
                      xi_mw=None if self.xi_mw is None else jnp.asarray(self.xi_mw),
                      xi_host=None if self.xi_host is None else jnp.asarray(self.xi_host))
        tau = (rel_days - c.shift) / (1.0 + c.redshift)
        fn = getattr(_physics(self.kind), f"{self.kind}_ab_magnitude")
        if ("geometric" not in common and "geometric" not in extra
                and "geometric" in inspect.signature(fn).parameters):
            # the factories store `geometric` only when True, so its absence is the search,
            # whatever the physics function's default: a model built with geometric=False, or
            # pickled before the option existed, keeps its numbers
            common["geometric"] = False
        return fn(tau, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                  jnp.asarray(self.lam), c.redshift, c.dl_cm, **v, **common, **extra)

    def _core(self, rel_days, bidx, *free):
        """Every observation evaluated. Traced."""
        return self._physics_call(rel_days, bidx, free)

    def _core_epochs(self, rel_days, bidx, first, inv, *free):
        """Each distinct epoch evaluated once, with the map ``(first, inv)`` as ARGUMENTS of the
        jitted core (never closure constants, so nothing traced is memoised). Traced."""
        return self._physics_call(rel_days, bidx, free, epoch_idx=(first, inv))

    def _magnitude(self, times, bidx, free, epoch_idx=None):
        import jax.numpy as jnp

        rel = _days_since(times, self.t_exp_days)        # host float64, before any cast
        if (epoch_idx is None and getattr(self, "unique_epochs", False)
                and isinstance(rel, np.ndarray) and rel.ndim == 1 and rel.size):
            first, inv = epoch_index(rel)                # the map of the core's own input
            if first.size < rel.size:                    # all distinct: no map, the plain core
                epoch_idx = (first, inv)
        if epoch_idx is None:
            return self._jitted(None, lambda: self._core)(jnp.asarray(rel), jnp.asarray(bidx),
                                                          *free)
        first, inv = epoch_idx
        return self._jitted("epochs", lambda: self._core_epochs)(
            jnp.asarray(rel), jnp.asarray(bidx), jnp.asarray(first, jnp.int32),
            jnp.asarray(inv, jnp.int32), *free)

    def predict_jax(self, theta, times, band_idx=None, epoch_idx=None):
        """``predict`` for a traced caller: flat theta, integer bands, flux density in Jy.

        ``theta`` is ordered by ``Model.parameters``; ``band_idx`` indexes ``band_names``. Map the
        data's band strings once, outside the trace, with ``predict_jax.band_index(bands)``. One
        parameter set; ``jax.vmap`` it for a batch. ``times`` may be traced. **Requires float64**
        (the guard raises on the first trace). An epoch at or before the model's time zero is at
        ``mag_floor``.
        """
        if epoch_idx is None:
            return super().predict_jax(theta, times, band_idx)
        # the distinct-epoch map (see _EPOCH_DOC); ValueError, not TypeError, so that no caller's
        # `except TypeError` reads a contract error as "needs concrete epochs"
        if not getattr(self, "unique_epochs", False):
            raise ValueError(f"model {self.name!r} takes no epoch_idx: it evaluates every "
                             f"observation (built with unique_epochs=False, or {self.kind!r} has "
                             f"no per-epoch path). Call predict_jax(theta, times, band_idx).")
        if band_idx is None:
            raise ValueError(f"model {self.name!r} is photometric and needs a `band_idx` array")
        first, inv = epoch_idx
        _check_epoch_idx(times, first, inv)
        mag = self._magnitude(times, band_idx, [theta[i] for i in range(len(self.params))],
                              (first, inv))
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag)


_EPOCH_DOC = """
        For ``tde_fallback`` and ``tde_mosfit`` built with ``unique_epochs=True`` (the default),
        observations that share a time are evaluated once. With concrete ``times`` this is
        automatic. With traced ``times`` pass ``epoch_idx=predict_jax.epoch_index(times_host[,
        size])``, built on the host from the same epochs (one light curve per call); without it
        every observation is evaluated (the same numbers to round-off, slower). A map that does
        not describe the epochs gives NaN (``ValueError`` when both are concrete). With a map, the
        derivative with respect to the epochs themselves is per distinct epoch."""


class _DirectPredictJax(_PredictJax):
    """``_PredictJax`` plus the unique-epoch hook of a model built with ``unique_epochs=True``:
    ``epoch_index(times, size=None)`` and the ``epoch_idx=`` keyword. The JAX samplers' adapters
    read ``epoch_index`` like ``band_index``."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.__doc__ = (self.__doc__ or "") + _EPOCH_DOC
        self.epoch_index = epoch_index

    def __call__(self, theta, times, band_idx=None, epoch_idx=None):
        return self.ctx.predict_jax(theta, times, band_idx, epoch_idx=epoch_idx)


def assemble(kind, redback_model, band_names, redshift, dl_cm, *, name, n_wave, pin, prior,
             mag_floor, dilation, ebv_mw, ebv_host, r_v_mw, r_v_host, law, filter_set,
             band_aliases, default_system, t_exp_days, free, redshift_prior, constraint,
             engine_kw, summary, caller, unique_epochs=False):
    """Bind the dataset context to the physics module ``kind`` and return a whisper ``Model``.

    The body of :func:`tde_fallback_model` and :func:`tde_analytical_model`: the same steps as
    :func:`whisper_cbpf.models.jax.tde_model`, for a model whose parameters are redback's.
    ``unique_epochs=True`` evaluates each distinct epoch once (the physics function must take
    ``epoch_idx=``) and gives ``predict_jax`` the hook ``epoch_index``; ``False`` builds the
    per-observation model, with a plain ``predict_jax`` and no new state. ``engine_kw`` is stored
    in the model's state and passed to the physics function: the fall-back factories put
    ``geometric`` in it only when True, and :class:`DirectTDEPredict` reads its absence as
    ``geometric=False``.
    """
    import jax.numpy as jnp

    from ...models import Model
    from .. import constraints as C
    from . import kilonova as kn

    phys = _physics(kind)
    unique_epochs = bool(unique_epochs)
    fn = getattr(phys, f"{kind}_ab_magnitude")
    if unique_epochs and "epoch_idx" not in inspect.signature(fn).parameters:
        raise ValueError(f"{caller}: unique_epochs=True needs {kind}_ab_magnitude(..., "
                         f"epoch_idx=), which {kind!r} does not take. Pass unique_epochs=False.")
    C.check_mode(constraint)
    free = _check_free(free, caller)
    _check_context(free, redshift, dl_cm, caller)
    band_names = list(band_names)

    fs = _filter_set_dict(band_names, filter_set, n_wave, default_system)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    xi_mw = xi_host = None
    if ebv_mw is not None:
        xi_mw = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_mw, law=law, frame="observer")
    if ebv_host is not None and "redshift" not in free:    # a free z: the law runs in the trace
        xi_host = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_host, law=law, frame="rest")

    # redback's own prior, with its fixed parameters: whisper's prior layer has no delta, so a
    # pinned parameter is bound here and dropped from `parameters`, as `kilonova_model` does.
    rb_prior, pinned = phys.default_prior()
    dists = dict(rb_prior.distributions)
    if prior is not None:                      # caller-supplied distributions win
        dists.update(prior.distributions)
    pinned = dict(pinned)
    for k, v in (pin or {}).items():
        if k not in phys.PARAMETERS:
            raise ValueError(f"pin: {k!r} is not a parameter of {redback_model!r}. "
                             f"Known: {list(phys.PARAMETERS)}")
        if v is None:
            pinned.pop(k, None)
        else:
            pinned[k] = float(v)
    params = [k for k in phys.PARAMETERS if k not in pinned]
    missing = [k for k in params if k not in dists]
    if missing:
        raise ValueError(
            f"{missing} have no prior: redback's {redback_model!r} prior file pins them, so "
            f"unpinning leaves no distribution. Either leave them pinned, or pass "
            f"prior=Prior({{'{missing[0]}': Uniform(...), ...}}) with a range you can defend.")
    dists.update(_free_distributions(free, prior, redshift_prior, caller))
    params = params + list(free)
    model_prior = type(rb_prior)({k: dists[k] for k in params})

    common = dict(mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
                  xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law,
                  **engine_kw)
    ctx = DirectTDEPredict(
        name, params, _BandIndex(band_names, f"{caller}(band_names=[...])", band_aliases),
        kind=kind, pinned=pinned, weights=weights, norms=norms, lam=lam, redshift=redshift,
        dl_cm=dl_cm, xi_mw=xi_mw, xi_host=xi_host, common=common, t_exp_days=t_exp_days,
        free=free, constraint=constraint, unique_epochs=unique_epochs)
    predict = ctx.predict
    predict_jax = _DirectPredictJax(ctx) if unique_epochs else _PredictJax(ctx)

    pin_note = ("; redback pins " + ", ".join(f"{k}={v:g}" for k, v in sorted(pinned.items()))
                if pinned else "")
    wall = (_constraint_note(kind, constraint) if kind in C.MODELS
            else "redback declares no constraints for this model. ")
    return Model(name=name, predict=predict, parameters=params, default_prior=model_prior,
                 predict_jax=predict_jax,
                 description=(f"TDE: {summary} (JAX/GPU){pin_note}. Requires float64. {wall}"
                              f"{_free_note(free)}{_floor_note(mag_floor)}"))
