"""JAX transient models, and their adapters onto whisper's ``Model`` contract.

The physics modules (:mod:`whisper_cbpf.models.jax.kilonova`, :mod:`whisper_cbpf.models.jax.flare`) are kept
free of whisper coupling -- ``kilonova.py`` imports nothing but numpy, jax and the band-integral
weights of :mod:`whisper_cbpf.synphot.grid_rule`. That is deliberate: the JAX physics can then be validated standalone
against redback and against an independent numpy reference, with no inference framework in the way.
The whisper adapters live here instead.

Two kinds of model, and they register differently
-------------------------------------------------
*Self-contained* models (the flare) satisfy whisper's contract directly:
``predict(parameters: dict, times, bands) -> flux``. Everything they need is in ``parameters``, so
they are registered automatically when :mod:`whisper_cbpf` is imported.

*Photometric* models (the kilonova) are not self-contained. Turning ``(mej, vej, kappa,
temperature_floor)`` into a flux in a band requires a **filter set**, a **redshift** and a
**luminosity distance** -- none of which whisper's three-argument ``predict`` signature carries, and
none of which have a defensible default (a made-up distance silently rescales every fitted mass).
So the kilonova is exposed as a *factory*: you bind the dataset context once, then register the
result. See :func:`kilonova_model`.
"""
from __future__ import annotations

import inspect

import numpy as np

from ...io.photometry import AB_ZEROPOINT_JY

__all__ = ["flare_model", "kilonova_model", "register_kilonova",
           "kilonova_two_model", "register_kilonova_two",
           "tde_model", "register_tde",
           "supernova_model", "register_supernova", "supernova_models"]


class _BandIndex:
    """The band-string -> integer-index lookup every photometric factory needs.

    A callable **class** rather than a closure, and that is load-bearing: an instance is held by
    ``predict`` and by ``predict_jax``, and a local function cannot be pickled, so a closure here
    made every photometric JAX model unusable with a multiprocess sampler. See
    :class:`_PhotometricPredict` for the measurement.

    An instance is attached to each factory's ``predict_jax`` as ``.band_index`` so a traced caller
    can do the lookup **once, outside the trace** and hand the result in. ``rebuild_hint`` is the
    factory call to name in the error, e.g. ``"tde_model(band_names=[...])"``.

    ``band_aliases`` maps *your data's* labels onto the filter-set names, e.g.
    ``{"g": "sdssg", "r": "sdssr"}``. Bare ``g``/``r``/``i`` in ``band_names`` are read as LSST
    (:func:`whisper_cbpf.synphot.resolve_filter`; ``default_system="sdss"`` for SDSS), so data
    whose bare letters are another survey's -- ``tests/data/at2017gfo.csv`` is SDSS -- needs either
    that or an alias onto survey-prefixed ``band_names``.

    It lives on the **model**, not on the sampler call, on purpose. Everything downstream of a fit
    — ``per_band_metrics``, ``predictive_metrics``, ``plot_ppc`` — calls ``model.predict(params,
    times, lc.band)`` with the light curve's own labels. A mapping that existed only inside the
    sampler would give you a fit whose WAIC silently failed to compute.

    There is **no default alias table**. AT2017GFO's 29 labels include both ``i`` and 30 Johnson
    ``I`` points, which any case-folding table would merge into one filter. State the mapping and
    it is auditable; guess it and the wrong answer has no error.
    """

    def __init__(self, band_names, rebuild_hint, band_aliases=None):
        self.band_names = list(band_names)
        self.rebuild_hint = str(rebuild_hint)
        self.aliases = {str(k): str(v) for k, v in (band_aliases or {}).items()}
        self.index_of = {b: i for i, b in enumerate(self.band_names)}
        unknown = sorted(v for v in self.aliases.values() if v not in self.index_of)
        if unknown:
            raise ValueError(
                f"band_aliases maps onto {unknown}, which are not in band_names "
                f"{self.band_names}. "
                f"The alias target must be a band this model was bound to.")

    def __call__(self, bands):
        try:
            return np.array([self.index_of[self.aliases.get(str(b), str(b))]
                             for b in np.asarray(bands)], dtype=int)
        except KeyError as exc:
            raise KeyError(
                f"band {exc.args[0]!r} was not bound to this model. Bound: {self.band_names}"
                + (f"; aliases: {self.aliases}" if self.aliases else "")
                + f". Rebuild with {self.rebuild_hint} covering every band in the data, or pass "
                f"band_aliases={{'{exc.args[0]}': '<filter-set name>'}}."
            ) from None


class _PhotometricPredict:
    """A photometric JAX model's ``predict``/``predict_jax``, as a picklable object.

    Building these two entry points as *local closures*, which is the obvious way, does not work.
    That made them unpicklable, and so unusable with any multiprocess sampler: ``abc`` and
    ``abc_smc`` at their default ``n_jobs=8`` died with ``AttributeError: Can't pickle local
    object 'kilonova_model.<locals>.predict'`` while the same fit at ``n_jobs=1`` succeeded --
    measured for ``kilonova_one_jax``, ``kilonova_two_jax``, ``kilonova_three_jax``,
    ``tde_gaussianrise`` and ``arnett_jax``, all five.

    The fix is this class. The dataset context is held as **plain, picklable data** -- numpy
    arrays for the filter set (``weights``, ``norms``, ``lam``), floats for ``redshift`` /
    ``dl_cm`` / ``t_exp_days`` / ``mag_floor``, lists and dicts for the rest. The one piece that
    genuinely cannot be pickled is the ``jax.jit`` object, which is a closure over the traced
    function; it is therefore built **lazily on first call** and **dropped by ``__getstate__``**,
    so every worker process rebuilds (and recompiles) its own. Do not try to pickle it.

    ``weights`` / ``norms`` / ``lam`` are stored with ``np.asarray``, which preserves the dtype
    the factory produced (float32 when ``jax_enable_x64`` is off), and are handed back to
    ``jnp.asarray`` inside ``_core`` -- i.e. inside the trace, where they become the same
    compile-time constants the closure captured.

    Subclasses supply ``_core`` (the traced function, jitted once) and ``_magnitude`` (how a call
    reaches it: the time convention differs per model family).
    """

    def __init__(self, name, params, band_index):
        self.name = str(name)
        self.params = list(params)
        self.band_index = band_index
        self.free = ()                          # context values fitted as parameters (FREE_CONTEXT)
        self._jit_cache = None                  # process-local; see __getstate__
        # redback's Constraint priors: the redback model they are redback's for, the
        # mode, and the dataset values they read besides theta (redshift, xi, ...). Set by the
        # supernova and TDE factories; None means no wall.
        self.constraint_model = None
        self.constraint = None
        self.constraint_extra = {}

    def _jitted(self, key, build):
        """``jax.jit(build())``, memoised on ``key`` for the life of THIS process.

        ``build`` is a thunk rather than the function itself because the supernova family builds
        a dense grid per key, and that must not happen on a cache hit.
        """
        import jax

        if self._jit_cache is None:
            self._jit_cache = {}
        core = self._jit_cache.get(key)
        if core is None:
            core = self._jit_cache[key] = jax.jit(build())
        return core

    def __getstate__(self):
        """Pickle the context and never the compiled core -- ``jax.jit`` returns a closure."""
        state = self.__dict__.copy()
        state["_jit_cache"] = None
        return state

    def _core(self, *args):
        """The traced function: times/bands + free parameters -> AB magnitude. Subclass hook."""
        raise NotImplementedError

    def _magnitude(self, times, bidx, free):
        """Route one call through the jitted ``_core``. Subclass hook (time conventions differ)."""
        raise NotImplementedError

    def predict(self, parameters, times, bands=None):
        """whisper contract: (parameters, times, bands) -> flux density in Jy.

        A draw that breaks redback's ``Constraint`` priors (armed by the supernova and TDE
        factories' ``constraint=``) predicts zero flux in every epoch and the model is not
        evaluated -- the CPU redback adapter's wall. This is the entry point the CPU samplers
        (``abc``, ``abc_smc``, ``mcmc``, ``nested``, ``snpe``) call, so they reject the same draws
        the JAX samplers do; before, they sampled the unconstrained prior on a JAX model.

        **Silently**: no warning, because the samplers meet such draws all the time (33-97 % of
        redback's own prior, by model and ``constraint``). So a direct call at such a draw -- a
        plot at a prior's midpoint, say -- also returns all zeros without a word. Check the draw
        first with ``model.predict_jax.constraint_ok(theta)`` (``theta`` ordered by
        ``model.parameters``; the attribute is ``None`` when no wall is armed), or build the model
        with ``constraint=None`` for the raw physics.
        """
        if bands is None:
            raise ValueError(f"model {self.name!r} is photometric and needs a `bands` array")
        idx = self.band_index(bands)
        free = [float(parameters[k]) for k in self.params]
        if not self.physical(free):
            return np.zeros(np.shape(times), dtype=float)
        mag = self._magnitude(np.asarray(times, dtype=float), idx, free)
        # whisper models return FLUX; convert once, at the end. Never combine magnitudes.
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))

    def predict_jax(self, theta, times, band_idx=None):
        """``predict`` for a traced caller: flat theta, integer bands, flux density in Jy.

        ``theta`` is ordered by ``Model.parameters``; ``band_idx`` indexes ``band_names``. Map the
        data's band strings once, outside the trace, with ``predict_jax.band_index(bands)`` --
        that is the same lookup ``predict`` does, including its error message. One parameter set;
        ``jax.vmap`` it for a batch. Pass ``times`` as the float64 numpy epochs when you can: a
        ``t_exp_days`` is then subtracted on the host, before any float32 cast.
        """
        if band_idx is None:                    # same requirement, same message, as predict
            raise ValueError(f"model {self.name!r} is photometric and needs a `band_idx` array")
        mag = self._magnitude(times, band_idx, [theta[i] for i in range(len(self.params))])
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag)

    def constraint_ok(self, theta):
        """Whether flat ``theta`` passes redback's constraints for this model.

        Traceable; one parameter set. Pinned parameters and the dataset's ``redshift`` (and the
        TDE's ``xi``, ``binding_energy_const``) enter where redback's conversion function reads
        them. ``predict_jax`` does NOT apply it -- it is the physics, as redback's function is, and
        stays differentiable across the wall; the JAX samplers' adapters do
        (:mod:`whisper_cbpf.samplers.jax._adapters`). The host ``predict`` applies its NumPy twin,
        :meth:`physical`.
        """
        import jax.numpy as jnp

        from .. import constraints as C

        return C.constraint_ok(self.constraint_model, self._constraint_inputs(theta),
                               mode=self.constraint, xp=jnp)

    def physical(self, free):
        """:meth:`constraint_ok` on the host, in NumPy: True when no wall is armed."""
        if self.constraint_model is None:
            return True
        from .. import constraints as C

        return bool(C.constraint_ok(self.constraint_model, self._constraint_inputs(free),
                                    mode=self.constraint, xp=np))

    def _constraint_inputs(self, theta):
        """The values redback's conversion function reads: dataset, pinned, then free."""
        return {**self.constraint_extra, **getattr(self, "pinned", {}),
                **{k: theta[i] for i, k in enumerate(self.params)}}

    def set_constraint(self, redback_model, constraint, **extra):
        """Arm :meth:`constraint_ok` for ``redback_model`` (a no-op if redback declares none)."""
        from .. import constraints as C

        C.check_mode(constraint)
        if constraint is not None and redback_model in C.MODELS:
            self.constraint_model, self.constraint = str(redback_model), constraint
            # a free redshift is None here and arrives in theta instead
            self.constraint_extra = {k: float(v) for k, v in extra.items() if v is not None}


class _PredictJax:
    """``Model.predict_jax`` as an object, so that it carries ``.band_index`` *and* pickles.

    ``predict`` can be the plain bound method ``ctx.predict`` -- CPython pickles a bound method as
    ``getattr(instance, name)``, so it round-trips as long as the instance does. ``predict_jax``
    cannot: ``samplers/jax/_adapters.resolve_band_index`` and ``samplers/jax/snpe_gpu`` both read
    ``predict_jax.band_index``, and attribute lookup on a bound method falls through to the
    underlying *function*, which is shared by every instance of the class. Hence this adapter,
    which holds the per-model lookup as an ordinary instance attribute.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.band_index = ctx.band_index
        self.__doc__ = type(ctx).predict_jax.__doc__
        # The JAX samplers' adapters read these two, like `band_index`: the constraint wall
        # (None: nothing to apply) and the magnitude floor they count.
        self.constraint_ok = ctx.constraint_ok if ctx.constraint_model is not None else None
        floor = getattr(ctx, "mag_floor", None)
        self.mag_floor = float(ctx.common["mag_floor"] if floor is None else floor)
        # The host-side epoch check of a model on fixed epochs (None: nothing to check).
        self.check_epochs = getattr(ctx, "check_epochs", None)

    def __call__(self, theta, times, band_idx=None):
        return self.ctx.predict_jax(theta, times, band_idx)


def _days_since(times, t_exp_days):
    """``times - t_exp_days``, on the HOST in float64 whenever ``times`` is concrete.

    The factories take ``t_exp_days`` on the light curve's own clock, often MJD. Subtracted in the
    model's dtype, float32 resolves 2^-8 = 0.0039 d near MJD 58000: 7.2 mmag and a log-likelihood
    off by 8.7 on AT2017GFO's kilonova, for a subtraction of two data values that
    needs no precision the model has. So it happens here, before any cast; the result is days
    since ``t_exp_days``, which float32 holds to ~1e-6 d. Traced ``times`` (a caller vmapping or
    jitting over them) are subtracted in the trace as before.
    """
    try:
        return np.asarray(times, dtype=np.float64) - t_exp_days
    except TypeError:                       # a tracer: TracerArrayConversionError is a TypeError
        return times - t_exp_days if t_exp_days else times


def _kilonova_time_grid(ctx, times):
    """redback 1.20's kilonova grid for these epochs (days since ``t_exp_days``), or None for
    the converged quadrature.

    Built host-side from the CONCRETE epochs, the way redback builds it from the times it is
    handed -- it depends on their minimum and maximum only -- and passed to the jitted core as an
    ordinary argument, so a new light curve costs no recompile unless the node count changes.

    The epochs are clipped up to the CPU adapter's ``MIN_TIME_DAY`` first, as the adapter clips
    them before redback sees them: a pre-merger row (an upper limit, say) is then the earliest
    epoch on both sides, and moves every node the same way. Clipped at ``T_EVAL_MIN`` (1 ms)
    instead, one such row moved the JAX grid's lower edge to 0.01 s and the CPU's to 43 s, and
    the post-merger magnitudes apart by up to 0.27 mag (one component) and 0.57 mag (two) over
    20 prior draws.
    """
    if ctx.time_grid is None:
        return None
    import jax

    from ..redback_adapter import MIN_TIME_DAY
    from . import kilonova as kn

    try:
        t = np.asarray(times, dtype=np.float64)
    except (TypeError, jax.errors.TracerArrayConversionError) as exc:   # vmapped/jitted times
        raise TypeError(
            f"model {ctx.name!r}: time_grid='redback' builds redback's grid from CONCRETE times. "
            f"vmap over theta only (in_axes=(0, None, None)) and keep `times` outside the trace, "
            f"or build the model with time_grid=None.") from exc
    return kn.redback_time_grid(np.maximum(t, MIN_TIME_DAY) * kn.DAY_TO_S / (1.0 + ctx.redshift))


def _check_time_grid(time_grid):
    if time_grid not in ("redback", None):
        raise ValueError(f"time_grid must be 'redback' or None, got {time_grid!r}")
    return time_grid


def _floor_note(mag_floor):
    """The description's record of the magnitude cap: redback has none, and a
    simulation-based sampler trains on the capped values."""
    return (f"Magnitudes are capped at mag_floor={mag_floor:g} (redback has no cap; the JAX "
            f"samplers' batched forward map counts the capped points).")


def _constraint_note(redback_model, constraint):
    """The description's account of redback's constraints for this model."""
    from .. import constraints as C

    if redback_model not in C.MODELS:
        return ""
    bounds = ", ".join(f"{lo:g} < {k} < {hi:g}"
                       for k, (lo, hi) in C.MODELS[redback_model][1].items())
    if constraint is None:
        return f"redback's constraints ({bounds}) are NOT applied (constraint=None). "
    fix = (" (nuclear-burning energy corrected to 1.51e18 erg/g; redback: 1.91e19)"
           if constraint == "corrected" and redback_model == "arnett" else "")
    return (f"redback's constraints {bounds}{fix} are a hard wall: predict gives zero flux and "
            f"the JAX samplers reject the draw (predict_jax.constraint_ok). ")


# --- explosion time and redshift as free parameters ---
#: The context values a factory can turn into fitted parameters, in the order they are appended to
#: ``Model.parameters``.
FREE_CONTEXT = ("t_exp", "redshift")


def _check_free(free, what):
    """``free=`` as a tuple in :data:`FREE_CONTEXT` order; a clear error for anything else."""
    if free is None:
        return ()
    free = [free] if isinstance(free, str) else list(free)
    bad = [f for f in free if f not in FREE_CONTEXT]
    if bad:
        raise ValueError(f"{what}: free={free} names {bad}. Only 't_exp' (the explosion time) and "
                         f"'redshift' can be freed; the model's own parameters are free already.")
    return tuple(k for k in FREE_CONTEXT if k in free)


def _check_context(free, redshift, dl_cm, what):
    """The fixed-redshift arguments must be given exactly when the redshift is not free."""
    if "redshift" in free:
        if redshift is not None or dl_cm is not None:
            raise ValueError(
                f"{what}: redshift={redshift!r}, dl_cm={dl_cm!r} fix the redshift, and "
                f"free=[..., 'redshift'] fits it. Pass one: to fit it, redshift=None and dl_cm=None "
                f"with redshift_prior=lc.redshift_prior (or prior=Prior({{'redshift': ...}})); the "
                f"distance then follows the redshift (Planck18, whisper_cbpf.models.cosmology).")
        return
    if redshift is None or dl_cm is None:
        raise ValueError(
            f"{what}: redshift and dl_cm are required unless the redshift is fitted. With a known "
            f"redshift pass both (whisper_cbpf.models.cosmology.luminosity_distance_cm(z) is "
            f"Planck18's distance); without one pass free=['redshift'], "
            f"redshift_prior=lc.redshift_prior.")


def _free_distributions(free, prior, redshift_prior, what):
    """``{name: distribution}`` for the freed context values, and a check that each has one.

    ``t_exp`` needs ``prior=Prior({'t_exp': ...})``: its window depends on the first detection and
    the last non-detection, which a model factory does not see. ``redshift`` takes, in order,
    ``prior['redshift']``, ``redshift_prior`` (a :attr:`LightCurve.redshift_prior` hint or a
    distribution), then :data:`whisper_cbpf.io.schema.DEFAULT_REDSHIFT_PRIOR` (the package's
    unknown-redshift default, Uniform(0.001, 1)).
    """
    from ...io.schema import DEFAULT_REDSHIFT_PRIOR, redshift_distribution
    from ..cosmology import Z_MAX, Z_MIN

    given = dict(prior.distributions) if prior is not None else {}
    stray = [k for k in FREE_CONTEXT if k in given and k not in free]
    if stray:
        raise ValueError(f"{what}: prior= names {stray}, which this model holds fixed. Add them to "
                         f"free= to fit them (free={list(free) + stray}).")
    out = {}
    if "t_exp" in free:
        if "t_exp" not in given:
            raise ValueError(
                f"{what}: free=['t_exp'] fits the explosion time, which needs a prior on the light "
                f"curve's own clock (the clock of the times you predict at), e.g. "
                f"prior=Prior({{'t_exp': Uniform(t_first - 20, t_first)}}) with t_first the first "
                f"detection, or the last non-detection as the lower edge. There is no default: the "
                f"window depends on the data, which the model factory does not see.")
        out["t_exp"] = given["t_exp"]
    if "redshift" in free:
        dist = given.get("redshift")
        if dist is None:
            dist = redshift_distribution(redshift_prior if redshift_prior is not None
                                         else DEFAULT_REDSHIFT_PRIOR)
        lo, hi = _bounds(dist)
        if not (Z_MIN <= lo and hi <= Z_MAX):
            raise ValueError(
                f"{what}: the redshift prior {dist!r} reaches outside [{Z_MIN:g}, {Z_MAX:g}], the "
                f"range of the luminosity-distance table (whisper_cbpf.models.cosmology). Narrow "
                f"it; a redshift of 0 has no luminosity distance.")
        out["redshift"] = dist
    return out


def _bounds(dist):
    """``(low, high)`` of a prior distribution (``bounds``, else ``low``/``high``)."""
    b = getattr(dist, "bounds", None)
    return (float(b[0]), float(b[1])) if b is not None else (float(dist.low), float(dist.high))


def _free_note(free):
    """The description's account of the freed context values (and of I11's warning)."""
    if not free:
        return ""
    parts = []
    if "t_exp" in free:
        parts.append("t_exp (explosion time, days on the light curve's clock)")
    if "redshift" in free:
        parts.append("redshift (the luminosity distance follows it: Planck18, "
                     "whisper_cbpf.models.cosmology). A fitted redshift can bias a model "
                     "comparison -- a model can move the source to buy a fit; "
                     "check whether the redshift posterior is narrower than its prior and where it "
                     "sits")
    return "Free context: " + "; ".join(parts) + ". "


class _Context:
    """The dataset context of one call: (physical values, time shift [d], redshift, distance).

    With nothing free it is the factory's fixed values. ``t_exp`` enters as a shift of the epochs
    relative to ``t_exp_days`` (which the factories still subtract on the host), and a
    free redshift brings its own luminosity distance from the Planck18 table.
    """

    def __init__(self, values, shift, redshift, dl_cm):
        self.values, self.shift, self.redshift, self.dl_cm = values, shift, redshift, dl_cm


def _auto_time_grid(time_grid, free, what):
    """The kilonova ``time_grid``: ``"auto"`` is redback's grid unless a context value is free."""
    if time_grid == "auto":
        return None if free else "redback"
    _check_time_grid(time_grid)
    if time_grid == "redback" and free:
        raise ValueError(f"{what}: time_grid='redback' builds redback's grid from the concrete "
                         f"epochs, which free={list(free)} moves. Use time_grid=None (the "
                         f"converged quadrature), which the default 'auto' picks.")
    return time_grid


def _fold_extinction(kn, weights, lam, redshift, free, ebv_mw, ebv_host, r_v_mw, r_v_host, law):
    """``(weights, host_ext)``: a fixed extinction folded into the kilonova's AB weights.

    Host extinction is applied at the rest wavelength ``lam / (1 + z)``, so with a free redshift
    it cannot be folded at setup: it is returned as keywords for the traced magnitude instead
    (the Milky Way's, observer-frame, is still folded).
    """
    if "redshift" in free and ebv_host is not None:
        if ebv_mw is not None:
            weights = kn.extincted_weights(weights, lam, ebv_mw=ebv_mw, ebv_host=None,
                                           r_v_mw=r_v_mw, law=law)
        return weights, dict(ebv_host=ebv_host, r_v_host=r_v_host, law=law)
    if ebv_mw is not None or ebv_host is not None:
        weights = kn.extincted_weights(weights, lam, redshift=redshift or 0.0,
                                       ebv_mw=ebv_mw or 0.0, ebv_host=ebv_host or 0.0,
                                       r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)
    return weights, {}


def _context(ctx, free):
    """:class:`_Context` for ``ctx`` (a :class:`_PhotometricPredict`) at the flat values ``free``."""
    v = dict(zip(ctx.params, free))
    shift = 0.0
    if "t_exp" in ctx.free:
        shift = v.pop("t_exp") - ctx.t_exp_days
    if "redshift" in ctx.free:
        import jax.numpy as jnp

        from ..cosmology import luminosity_distance_cm
        z = v.pop("redshift")
        return _Context(v, shift, z, luminosity_distance_cm(z, xp=jnp))
    return _Context(v, shift, ctx.redshift, ctx.dl_cm)


# --- flare ---
def _filter_set_dict(band_names, filter_set, n_wave, default_system):
    """The ``{'lam', 'trans'}`` dict a photometric factory integrates with, one row per band name.

    ``filter_set`` wins: a :class:`whisper_cbpf.synphot.FilterSet` (converted with ``to_legacy``,
    so a CPU model built on the same FilterSet computes the same integral) or a legacy dict,
    used as is. Otherwise the names resolve through :func:`whisper_cbpf.synphot.resolve_filter`
    (bare ``u g r i z y`` are LSST unless ``default_system=`` says otherwise) and get the default
    rule, Gauss-16 per band (:func:`whisper_cbpf.synphot.filter_set_for`: shipped for LSST, ZTF and
    SDSS, else built from sncosmo). ``n_wave=`` selects the grid rule of whisper <= 0.1.0 instead
    (``make_filter_set`` on ``n_wave`` shared points). The two agree to <= 0.005 mmag; Gauss-16
    integrates over 16 x n_bands points instead of ``n_wave``.
    """
    from ...synphot import FilterSet, filter_set_for, resolve_filters

    if filter_set is not None and not isinstance(filter_set, FilterSet):
        return filter_set                       # a legacy dict: its rows ARE band_names' order
    known = filter_set.names if filter_set is not None else ()
    names = resolve_filters(band_names, None, default_system, known)
    if filter_set is None and n_wave is not None:
        from ...synphot.grid_rule import make_filter_set
        return make_filter_set(names, n_wave=n_wave)
    fs = filter_set if filter_set is not None else filter_set_for(names)
    missing = [n for n in names if n not in fs.names]
    if missing:
        raise ValueError(f"filter_set has no {missing} (band_names {list(band_names)}); it holds "
                         f"{list(fs.names)}")
    legacy = fs.to_legacy()
    legacy["trans"] = legacy["trans"][[fs.index(n) for n in names]]    # one row per band name
    legacy["names"] = np.array(names)
    return legacy


def flare_model():
    """The JAX flare as a ``whisper_cbpf.models.Model``. Self-contained, auto-registered.

    Needs no dataset context (no band integral), so it is registered at import as ``"flare_jax"``.

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> m = wp.flare_model()                          # needs the [gpu] extra
    >>> m.name, m.parameters
    ('flare_jax', ['log_amp', 'log_sigma', 'log_tau', 't0'])
    """
    from . import flare
    return flare.get_model()


# --- kilonova ---
class _KilonovaOnePredict(_PhotometricPredict):
    """``predict``/``predict_jax`` for :func:`kilonova_model`. See :class:`_PhotometricPredict`."""

    def __init__(self, name, params, band_index, *, weights, norms, lam, redshift, dl_cm,
                 t_exp_days, mag_floor, arnett_prefactor, temperature_floor, time_grid,
                 free=(), host_ext=None):
        super().__init__(name, params, band_index)
        self.free = tuple(free)
        self.time_grid = time_grid
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = None if redshift is None else float(redshift)
        self.dl_cm = None if dl_cm is None else float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.mag_floor = float(mag_floor)
        self.arnett_prefactor = float(arnett_prefactor)
        # None means temperature_floor is FREE and arrives in `free`; a float pins it.
        self.temperature_floor = None if temperature_floor is None else float(temperature_floor)
        # host extinction evaluated in the trace, at a FREE redshift (fixed: folded into weights)
        self.host_ext = dict(host_ext or {})

    def _core(self, times, bidx, grid, *free):
        """Observer-frame days since ``t_exp_days`` -> AB magnitude per observation. Traced."""
        import jax.numpy as jnp

        from . import kilonova as kn

        c = _context(self, free)
        v = c.values
        tf = v["temperature_floor"] if self.temperature_floor is None else self.temperature_floor
        t_src = kn.source_time_s(times - c.shift, c.redshift)
        return kn.ab_magnitude(t_src, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                               jnp.asarray(self.lam), c.redshift, c.dl_cm,
                               v["mej"], v["vej"], v["kappa"], tf,
                               mag_floor=self.mag_floor,
                               arnett_prefactor=self.arnett_prefactor, time_grid=grid,
                               **self.host_ext)

    def _magnitude(self, times, bidx, free):
        # JIT ONCE -- the same reason tde_model gives, and the same measurement: the unjitted
        # binding re-traces on every call, 65.3 ms per predict() against 0.14 ms for the identical
        # computation under jit. Only `times`, `band_idx` and the free parameters are traced; the
        # filter set, the redshift, the distance, `t_exp_days`, `mag_floor`, `arnett_prefactor`
        # and a PINNED temperature_floor are attributes read inside the trace, so this compiles
        # once per n_obs. Both entry points come through here, which keeps them from drifting.
        import jax.numpy as jnp

        rel = _days_since(times, self.t_exp_days)        # host float64, before any cast (1.3)
        grid = _kilonova_time_grid(self, rel)
        return self._jitted(None, lambda: self._core)(jnp.asarray(rel), jnp.asarray(bidx),
                                                      grid, *free)


def kilonova_model(band_names, redshift=None, dl_cm=None, *, name="kilonova_one_jax",
                   n_wave=None, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                   temperature_floor=None, mag_floor=40.0, t_exp_days=0.0, filter_set=None,
                   arnett_prefactor=1.0, time_grid="auto", band_aliases=None,
                   default_system=None, free=None, prior=None, redshift_prior=None):
    """Bind dataset context to the one-component JAX kilonova and return a whisper ``Model``.

    Parameters
    ----------
    band_names : sequence of str
        Filter names or labels (``lsstg``, ``ztfr``, ``sdssi``, ``bessellb``, an SVO ID, or a bare
        letter, read in ``default_system``), in the order the light curve's ``bands`` column will
        be mapped to. They resolve through :func:`whisper_cbpf.synphot.resolve_filter`.
    redshift, dl_cm : float
        Fixed for the dataset. ``dl_cm`` is NOT derived from ``redshift`` here on purpose -- the
        cosmology is the caller's choice, and quietly picking one would make two codes disagree for
        a reason that has nothing to do with the model. Leave both ``None`` when the redshift is
        fitted (``free=["redshift"]``).
    free : list of str or None
        Context values to fit as ordinary parameters: ``"t_exp"`` (the merger time, on the light
        curve's own clock) and/or ``"redshift"`` (the luminosity distance then follows it through
        Planck18, :func:`whisper_cbpf.models.cosmology.luminosity_distance_cm`). They are appended
        to ``parameters`` in that order and are traced like any other parameter, so ``jit`` and
        ``vmap`` over them compile once. ``t_exp`` needs ``prior=Prior({"t_exp": ...})``; the
        redshift's prior is ``prior["redshift"]``, else ``redshift_prior``, else Uniform(0.001, 1).
        A fitted redshift can bias a model comparison: a model can move the source to buy a fit,
        so read the redshift posterior against its prior before ranking.
    prior : whisper_cbpf.priors.Prior or None
        Distributions overriding the defaults below, per parameter (``t_exp`` and ``redshift``
        included).
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none: a
        :attr:`whisper_cbpf.LightCurve.redshift_prior` hint (read by
        :func:`whisper_cbpf.io.schema.redshift_distribution`) or a distribution.
    temperature_floor : float or None
        ``None`` (default) leaves it a FREE parameter, matching redback's default prior
        (``LogUniform(100, 6000)``). Pass a float to pin it.

    Notes on the opacity prior -- READ BEFORE "FIXING" IT
    ----------------------------------------------------
    This model's default prior gives ``kappa`` the broad ``Uniform(1, 30)``, and that is CORRECT.
    The two- and three-component models deliberately use *disjoint* opacity corridors (blue 0.1-1,
    purple 1-5, red 5-30, after Villar+2017) because with identical per-component priors the
    posterior is exactly symmetric under relabelling and the components can swap identities. A
    single component has nothing to be distinguished *from*, so a narrowed corridor here would be an
    arbitrary restriction, not a physical constraint.

    One consequence worth knowing when you COMPARE models: on its own default prior this model
    cannot reach the low-opacity blue regime, so a one-versus-two comparison in which only the
    two-component side is allowed there is not like-for-like. Widen this model's ``kappa`` to span
    both regimes *for that comparison* (see ``docs/MODEL_COMPARISON.md``). That is a property of
    the comparison, not a defect in this prior.
    filter_set : FilterSet, dict or None
        The band integral. Default ``None``: Gauss-16 per band
        (:func:`whisper_cbpf.synphot.filter_set_for`; shipped for LSST ugrizy, ZTF gri and SDSS
        ugriz, so no sncosmo needed). A :class:`whisper_cbpf.synphot.FilterSet` -- pass the one a
        CPU model uses to make both compute the same integral -- or a ``make_filter_set`` dict.
    n_wave : int or None
        ``None`` (default) uses ``filter_set``'s rule. An integer selects whisper <= 0.1.0's grid
        rule on ``n_wave`` shared points (it was 2000; within 0.005 mmag of Gauss-16, and slower).
    default_system : str or None
        The survey bare letters in ``band_names`` are read in (default: the session's, LSST).
    arnett_prefactor : float
        1.0 (default) reproduces redback, whose diffusion kernel integrates to 1/2 rather than 1
        -- a uniform 0.7526 mag too faint relative to the standard Arnett solution (measured:
        L/L_in -> 0.500055 at t = 100 t_diff, where it must -> 1). 2.0 selects Arnett 1982 /
        Villar+2017. **The default is redback's, by project rule: parity with redback outweighs
        the correction.** See kilonova.py's ARNETT_PREFACTOR block for the full measurement.
    time_grid : {"auto", "redback", None}
        ``"redback"`` solves the diffusion on redback 1.20's own time grid for these epochs
        (``kilonova.redback_time_grid``: 500 nodes, dense around the data) and interpolates the
        photosphere, exactly as redback's ``one_component_kilonova_model`` does, so the port
        matches redback at every epoch -- including where redback is under-resolved: past ~2.66
        t_diff that grid is too bright, by 1.19 mag at 20 d against a converged one (for
        parity with redback's default). ``None`` is the converged quadrature
        (kilonova.py identity 6), this factory's only behaviour up to whisper 0.1.0, and the one
        to use for late epochs. With ``"redback"`` the times passed to ``predict_jax`` must be
        concrete (vmap over theta only). ``"auto"`` (default) is ``"redback"`` with a fixed merger
        time and redshift, and ``None`` when either is free: redback's grid is built from the
        epochs, which a fitted ``t_exp`` or redshift moves.

    Notes
    -----
    A pre-merger epoch (``t <= t_exp``) is at ``mag_floor``, so an upper limit before the merger
    constrains a fitted ``t_exp``.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax import kilonova_model
    >>> from whisper_cbpf.priors import Prior, Uniform
    >>> m = kilonova_model(["lsstg", "lsstr"], free=["t_exp", "redshift"],
    ...                    prior=Prior({"t_exp": Uniform(-1.0, 0.0),
    ...                                 "redshift": Uniform(0.005, 0.05)}))
    >>> m.parameters
    ['mej', 'vej', 'kappa', 'temperature_floor', 't_exp', 'redshift']
    >>> p = dict(mej=0.03, vej=0.2, kappa=3.0, temperature_floor=2000.0, t_exp=-0.3,
    ...          redshift=0.01)
    >>> flux = m.predict(p, np.array([-0.5, 1.0, 2.0]), np.array(["lsstg", "lsstr", "lsstg"]))
    >>> round(float(-2.5 * np.log10(flux[0] / 3631.0)), 3)   # before the merger: mag_floor
    40.0
    """
    import jax.numpy as jnp

    from ...models import Model
    from ...priors import LogUniform, Prior, Uniform

    from . import kilonova as kn

    free = _check_free(free, "kilonova_model")
    _check_context(free, redshift, dl_cm, "kilonova_model")
    time_grid = _auto_time_grid(time_grid, free, "kilonova_model")
    band_names = list(band_names)
    fs = _filter_set_dict(band_names, filter_set, n_wave, default_system)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    weights, host_ext = _fold_extinction(kn, weights, fs["lam"], redshift, free, ebv_mw, ebv_host,
                                         r_v_mw, r_v_host, law)

    free_tf = temperature_floor is None
    params = ["mej", "vej", "kappa"] + (["temperature_floor"] if free_tf else []) + list(free)

    ctx = _KilonovaOnePredict(
        name, params, _BandIndex(band_names, "kilonova_model(band_names=[...])", band_aliases),
        weights=weights, norms=norms, lam=lam, redshift=redshift, dl_cm=dl_cm,
        t_exp_days=t_exp_days, mag_floor=mag_floor, arnett_prefactor=arnett_prefactor,
        temperature_floor=temperature_floor, time_grid=time_grid, free=free,
        host_ext=host_ext)
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    # redback's own defaults, verbatim from redback/priors/one_component_kilonova_model.prior,
    # so a redback-vs-whisper_cbpf (JAX port) comparison is apples to apples. Do not substitute
    # whisper's hand-chosen kilonova priors here -- they are disjoint from redback's in kappa.
    dists = {"mej": Uniform(1e-2, 0.05), "vej": Uniform(0.1, 0.5), "kappa": Uniform(1.0, 30.0)}
    if free_tf:
        dists["temperature_floor"] = LogUniform(100.0, 6000.0)
    dists.update(_free_distributions(free, prior, redshift_prior, "kilonova_model"))
    if prior is not None:
        dists.update({k: d for k, d in prior.distributions.items() if k in params})

    return Model(name=name, predict=predict, parameters=params,
                 default_prior=Prior({k: dists[k] for k in params}),
                 predict_jax=predict_jax,
                 description=("One-component kilonova (JAX/GPU). Physics identical to redback's "
                               "_one_component_kilonova_model"
                               + ("; solved on redback 1.20's own time grid, so it shares "
                                  "redback's late-time error beyond t ~ 2.66*t_diff (1.19 mag at "
                                  "20 d)" if time_grid == "redback" else
                                  "; the diffusion integral is re-quadratured (see kilonova.py "
                                  "identity 6), so it diverges from redback beyond t ~ "
                                  "2.66*t_diff, where redback is under-resolved")
                               + f". {_free_note(free)}{_floor_note(mag_floor)}"))


def _register(m):
    """Register a freshly built ``Model`` under its own name, replacing any earlier binding.

    ``overwrite=True`` is what makes re-running a factory with a different redshift or band set
    do the obvious thing rather than raise.
    """
    from ...models import register_model

    register_model(m.name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True,
                   predict_jax=m.predict_jax, log_prob_jax=m.log_prob_jax,
                   param_aliases=m.param_aliases)
    return m


def register_kilonova(band_names, redshift=None, dl_cm=None, *, name="kilonova_one_jax",
                      **kwargs):
    """Build a kilonova ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`kilonova_model`.
    name : str, default "kilonova_one_jax"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> m = wp.register_kilonova(["lsstg", "lsstr"], redshift=0.0098, dl_cm=1.3e26)
    >>> m.name, m.parameters, "kilonova_one_jax" in wp.list_models()
    ('kilonova_one_jax', ['mej', 'vej', 'kappa', 'temperature_floor'], True)
    """
    return _register(kilonova_model(band_names, redshift, dl_cm, name=name, **kwargs))


# --- kilonova (2-comp) ---
class _KilonovaMultiPredict(_PhotometricPredict):
    """``predict``/``predict_jax`` for the two- and three-component kilonovae.

    One class for both: they differ only in which ``kilonova_two`` entry point the core calls, so
    ``n_components`` selects it. See :class:`_PhotometricPredict` for why this is a class at all.
    """

    def __init__(self, name, params, band_index, *, n_components, weights, norms, lam,
                 redshift, dl_cm, t_exp_days, mag_floor, time_grid, free=(), host_ext=None):
        super().__init__(name, params, band_index)
        self.free = tuple(free)
        self.n_components = int(n_components)
        self.time_grid = time_grid
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = None if redshift is None else float(redshift)
        self.dl_cm = None if dl_cm is None else float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.mag_floor = float(mag_floor)
        self.host_ext = dict(host_ext or {})

    def _core(self, times_s, bidx, grid, *free):
        """SOURCE-frame seconds -> AB magnitude per observation. Traced, pure jnp."""
        import jax.numpy as jnp

        from . import kilonova_two as kn2

        fn = (kn2.two_component_magnitude if self.n_components == 2
              else kn2.three_component_magnitude)
        return fn(times_s, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                  jnp.asarray(self.lam), self.redshift, self.dl_cm, *free,
                  mag_floor=self.mag_floor, time_grid=grid)

    def _core_free(self, rel_days, bidx, *free):
        """Observer-frame days since ``t_exp_days`` -> AB magnitude, with a free ``t_exp`` and/or
        redshift among ``free``. Traced; the converged quadrature (``time_grid=None``) only."""
        import jax.numpy as jnp

        from . import kilonova as kn
        from . import kilonova_two as kn2

        c = _context(self, free)
        fn = (kn2.two_component_magnitude if self.n_components == 2
              else kn2.three_component_magnitude)
        t_src = kn.source_time_s(rel_days - c.shift, c.redshift)
        return fn(t_src, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                  jnp.asarray(self.lam), c.redshift, c.dl_cm,
                  *[c.values[k] for k in self.params if k not in FREE_CONTEXT],
                  mag_floor=self.mag_floor, time_grid=None, **self.host_ext)

    def _magnitude(self, times, bidx, free):
        # JIT ONCE -- the two-component factory was the only photometric one that called its
        # physics function unjitted, so every predict() re-traced `two_component_magnitude`;
        # `kilonova_two.py`'s own jitted-symbol list does not include it. Measured on 211
        # observations at n_wave=2000 on an A6000, best of five: 134.9 ms unjitted against
        # 1.893 ms here, a factor 71. Only `times`, `band_idx` and the free parameters are traced,
        # so this compiles once per n_obs, and both entry points come through here.
        import jax.numpy as jnp

        from . import kilonova as kn

        rel = _days_since(times, self.t_exp_days)        # host float64, before any cast (1.3)
        if self.free:
            return self._jitted("free", lambda: self._core_free)(jnp.asarray(rel),
                                                                 jnp.asarray(bidx), *free)
        grid = _kilonova_time_grid(self, rel)
        t_src = kn.source_time_s(rel, self.redshift)
        return self._jitted(None, lambda: self._core)(jnp.asarray(t_src), jnp.asarray(bidx),
                                                      grid, *free)


def _prior_in_own_names(prior, aliases, what):
    """``prior`` with each name spelled as an alias (a value of ``aliases``, redback's spelling)
    renamed to the model's own, so either spelling is accepted; the order is kept."""
    from ...priors import Prior

    back = {a: k for k, a in aliases.items()}
    renamed = {back.get(k, k): d for k, d in prior.distributions.items()}
    if len(renamed) < len(prior.distributions):
        raise ValueError(f"{what}: the prior names a parameter twice, once in each spelling "
                         f"({sorted(prior.distributions)}). Use {list(aliases)} or "
                         f"{list(aliases.values())}, or mix them, but each parameter once.")
    return Prior(renamed)


def _multi_prior(prior, free, default, aliases, redshift_prior, what):
    """A multi-component kilonova's prior: the physical part as before, plus the freed values.

    The physical distributions in ``prior`` replace the default as a whole (as they always did);
    ``t_exp`` / ``redshift`` in ``prior`` are read separately, so ``prior=Prior({"t_exp": ...})``
    alone keeps the default for everything else.
    """
    from ...priors import Prior

    given = dict(prior.distributions) if prior is not None else {}
    phys = {k: d for k, d in given.items() if k not in FREE_CONTEXT}
    base = (_prior_in_own_names(Prior(phys), aliases, what) if phys else default)
    ctx_prior = Prior({k: d for k, d in given.items() if k in FREE_CONTEXT})
    extra = _free_distributions(free, ctx_prior, redshift_prior, what)
    return Prior({**base.distributions, **extra})


def kilonova_two_model(band_names, redshift=None, dl_cm=None, *, name="kilonova_two_jax",
                       n_wave=None, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1,
                       law="f99", mag_floor=40.0, t_exp_days=0.0, filter_set=None, prior=None,
                       time_grid="auto", band_aliases=None, default_system=None, free=None,
                       redshift_prior=None):
    """Bind dataset context to the TWO-component JAX kilonova and return a whisper ``Model``.

    The two-component sibling of :func:`kilonova_model`, built the same way and for the same reason:
    ``predict(parameters, times, bands)`` carries no filter set, no redshift and no luminosity
    distance, so this is a factory rather than a registry entry.

    Parameters follow redback's positional order -- ``mej, vej, temperature_floor, kappa`` per
    component -- but are named for the component they describe: ``mej_blue ... kappa_blue`` then
    ``mej_red ... kappa_red``. The NAME is cosmetic. What makes a component blue or red is the
    opacity corridor its prior allows, and the default prior keeps those corridors **disjoint**
    (``kappa_blue`` U(0.1, 1.0), ``kappa_red`` U(1, 30)), so the two components are not
    exchangeable and a fit cannot label-switch.

    ``prior`` defaults to :func:`kilonova_two.villar_prior` (Villar+2017), NOT redback's box.
    redback's own prior cannot contain the published AT2017GFO solution -- Villar's
    ``kappa_blue = 0.5`` is below its ``Uniform(1, 30)`` floor and ``M_ej,red = 0.050`` is above its
    ``Uniform(0.01, 0.03)`` ceiling -- so a fit inside it rails on both and reaches chi2/N of order
    1e3. It is still available as :func:`kilonova_two.default_prior` for an apples-to-apples
    comparison against redback: ``prior=kilonova_two.default_prior()``. A ``prior`` may name the
    parameters either way, ``mej_blue ... kappa_red`` or redback's ``mej_1 ... kappa_2`` (the
    ``param_aliases`` below); the model's ``default_prior`` carries this model's names.
    Note this is **not** the prior used by ``whisper_cbpf.models.two_component_kilonova``, whose
    ``kappa_1`` range is disjoint from it -- the two are different models, not one model twice.

    ``time_grid`` is :func:`kilonova_model`'s: by default each component is solved on redback
    1.20's one-component grid for these epochs -- the sum of two one-component redback calls
    -- with redback's late-time error; ``None`` for the converged quadrature (and
    the default when ``free`` is not empty).

    ``free`` and ``redshift_prior`` are :func:`kilonova_model`'s (fit ``t_exp`` and/or the
    redshift; ``prior=Prior({"t_exp": ...})`` keeps the default for the other parameters).

    ``Model.param_aliases`` maps the names onto redback's (``mej_blue`` -> ``mej_1`` ...
    ``kappa_red`` -> ``kappa_2``), the names ``two_component_kilonova`` and the redback adapter
    use, so a CPU and a GPU posterior of this model pair column for column.

    Examples
    --------
    >>> from whisper_cbpf.models.jax import kilonova_two_model
    >>> from whisper_cbpf.priors import Prior, Uniform
    >>> m = kilonova_two_model(["lsstg", "lsstr"], 0.0098, 1.3e26, free=["t_exp"],
    ...                        prior=Prior({"t_exp": Uniform(-1.0, 0.0)}))
    >>> m.parameters[-2:]
    ['kappa_red', 't_exp']
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import kilonova_two as kn2

    free = _check_free(free, "kilonova_two_model")
    _check_context(free, redshift, dl_cm, "kilonova_two_model")
    time_grid = _auto_time_grid(time_grid, free, "kilonova_two_model")
    band_names = list(band_names)
    fs = _filter_set_dict(band_names, filter_set, n_wave, default_system)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    weights, host_ext = _fold_extinction(kn, weights, fs["lam"], redshift, free, ebv_mw, ebv_host,
                                         r_v_mw, r_v_host, law)
    # Villar+2017 naming. The label is cosmetic; what MAKES a component blue or red is its opacity
    # corridor in the prior (kappa_blue U(0.1, 1.0), kappa_red U(1, 30)), which is why the default
    # prior below is villar_prior and not redback's symmetric box.
    phys = ["mej_blue", "vej_blue", "temperature_floor_blue", "kappa_blue",
            "mej_red", "vej_red", "temperature_floor_red", "kappa_red"]
    params = phys + list(free)

    ctx = _KilonovaMultiPredict(
        name, params, _BandIndex(band_names, "kilonova_two_model(band_names=[...])", band_aliases),
        n_components=2, weights=weights, norms=norms, lam=lam, redshift=redshift, dl_cm=dl_cm,
        t_exp_days=t_exp_days, mag_floor=mag_floor, time_grid=time_grid, free=free,
        host_ext=host_ext)
    predict, predict_jax = ctx.predict, _PredictJax(ctx)
    # the redback spelling of each name, in the same positional order
    aliases = dict(zip(phys, kn2.PARAMETERS))

    if free:
        default_prior = _multi_prior(prior, free, kn2.villar_prior(2, with_sigma=False), aliases,
                                     redshift_prior, "kilonova_two_model")
    else:
        default_prior = (_prior_in_own_names(prior, aliases, "kilonova_two_model")
                         if prior is not None else kn2.villar_prior(2, with_sigma=False))
    return Model(name=name, predict=predict, parameters=params,
                 default_prior=default_prior,
                 predict_jax=predict_jax, param_aliases=aliases,
                 description=("Two-component (blue + red) kilonova, JAX/GPU. Same physics as "
                              "redback's two_component_kilonova_model; components are summed in "
                              "flux, never in magnitude. The default prior is Villar+2017's, with "
                              "DISJOINT opacity corridors (kappa_blue 0.1-1, kappa_red 1-30), so "
                              "the components are not exchangeable and cannot label-switch. "
                              + _free_note(free) + _floor_note(mag_floor)))


def register_kilonova_two(band_names, redshift=None, dl_cm=None, *, name="kilonova_two_jax",
                          **kwargs):
    """Build a two-component kilonova ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`kilonova_two_model`.
    name : str, default "kilonova_two_jax"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> m = wp.register_kilonova_two(["lsstg", "lsstr"], redshift=0.0098, dl_cm=1.3e26)
    >>> m.name, len(m.parameters)
    ('kilonova_two_jax', 8)
    """
    return _register(kilonova_two_model(band_names, redshift, dl_cm, name=name, **kwargs))


# --- three-component kilonova ---
def kilonova_three_model(band_names, redshift=None, dl_cm=None, *, name="kilonova_three_jax",
                         n_wave=None, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1,
                         law="f99", mag_floor=40.0, t_exp_days=0.0, filter_set=None, prior=None,
                         time_grid="auto", band_aliases=None, default_system=None, free=None,
                         redshift_prior=None):
    """Bind dataset context to the THREE-component JAX kilonova and return a whisper ``Model``.

    Villar+2017's blue + purple + red decomposition. The physics is unchanged from two components --
    independent blackbodies, each with its own photosphere and temperature floor, summed in FLUX
    before the band integral, never in magnitude. Villar+2017 preferred three components because the
    two-component model predicts a double-peaked near-infrared light curve at ~2-5 d that the
    AT2017GFO data does not show.

    Twelve parameters, ``mej, vej, temperature_floor, kappa`` per component in blue, purple, red
    order. As for two components the NAME is cosmetic: what makes a component blue, purple or red is
    the opacity corridor its prior allows. The default prior keeps all three **disjoint** --
    ``kappa_blue`` U(0.1, 1.0), ``kappa_purple`` U(1, 5), ``kappa_red`` U(5, 30), bracketing
    Villar+2017's fixed 0.5 / 3 / 10 cm^2 g^-1 -- so the components cannot be relabelled and the
    posterior has no mirror modes to average over.

    ``sigma``, Villar's white-noise term, is NOT a model parameter: it belongs to the likelihood.
    Fit it with ``likelihood="scatter"`` (see :class:`GaussianLikelihoodWithScatter`) rather than by
    adding a column the model cannot consume.

    Photometric, so a factory rather than a registry entry, for the same reason as its two-component
    sibling: ``predict(parameters, times, bands)`` carries no filter set, no redshift and no
    luminosity distance.

    ``time_grid`` is :func:`kilonova_model`'s, with the same default as the one- and
    two-component models, so the three are compared on one discretisation. ``free`` and
    ``redshift_prior`` are :func:`kilonova_model`'s.

    Parameters
    ----------
    band_names : sequence of str
        The bands the model will be evaluated in.
    redshift, dl_cm : float, optional
        Known redshift and luminosity distance [cm]; leave them out with ``free=["redshift"]``.
    name, n_wave, ebv_mw, ebv_host, r_v_mw, r_v_host, law, mag_floor, t_exp_days, filter_set,
    prior, time_grid, band_aliases, default_system, free, redshift_prior
        As :func:`kilonova_model`.

    Returns
    -------
    Model

    Examples
    --------
    >>> from whisper_cbpf.models.jax import kilonova_three_model
    >>> m = kilonova_three_model(["lsstg", "lsstr"], free=["redshift"])
    >>> m.parameters[-1], m.default_prior.distributions["redshift"]
    ('redshift', Uniform(0.001, 1.0))
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import kilonova_two as kn2

    free = _check_free(free, "kilonova_three_model")
    _check_context(free, redshift, dl_cm, "kilonova_three_model")
    time_grid = _auto_time_grid(time_grid, free, "kilonova_three_model")
    band_names = list(band_names)
    fs = _filter_set_dict(band_names, filter_set, n_wave, default_system)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    weights, host_ext = _fold_extinction(kn, weights, fs["lam"], redshift, free, ebv_mw, ebv_host,
                                         r_v_mw, r_v_host, law)
    params = list(kn2.PARAMETERS_3) + list(free)

    ctx = _KilonovaMultiPredict(
        name, params,
        _BandIndex(band_names, "kilonova_three_model(band_names=[...])", band_aliases),
        n_components=3, weights=weights, norms=norms, lam=lam, redshift=redshift, dl_cm=dl_cm,
        t_exp_days=t_exp_days, mag_floor=mag_floor, time_grid=time_grid, free=free,
        host_ext=host_ext)
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    if free:
        default_prior = _multi_prior(prior, free, kn2.villar_prior(3, with_sigma=False), {},
                                     redshift_prior, "kilonova_three_model")
    else:
        default_prior = prior if prior is not None else kn2.villar_prior(3, with_sigma=False)
    return Model(name=name, predict=predict, parameters=params,
                 default_prior=default_prior,
                 predict_jax=predict_jax,
                 description=("Three-component (blue + purple + red) kilonova, JAX/GPU, "
                              "Villar+2017. Components are summed in flux, never in magnitude. "
                              "The default prior gives each component a DISJOINT opacity corridor "
                              "(0.1-1 / 1-5 / 5-30), bracketing Villar's fixed 0.5 / 3 / 10, so "
                              "they cannot label-switch. " + _free_note(free)
                              + _floor_note(mag_floor)))


def register_kilonova_three(band_names, redshift=None, dl_cm=None, *, name="kilonova_three_jax",
                            **kwargs):
    """Build a three-component kilonova ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`kilonova_three_model`.
    name : str, default "kilonova_three_jax"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> m = wp.register_kilonova_three(["lsstg", "lsstr"], redshift=0.0098, dl_cm=1.3e26)
    >>> m.name, len(m.parameters)
    ('kilonova_three_jax', 12)
    """
    return _register(kilonova_three_model(band_names, redshift, dl_cm, name=name, **kwargs))


# --- TDE ---
class _TDEPredict(_PhotometricPredict):
    """``predict``/``predict_jax`` for :func:`tde_model`. See :class:`_PhotometricPredict`."""

    def __init__(self, name, params, band_index, *, rise, pinned, weights, norms, lam,
                 redshift, dl_cm, xi, n_pre, xi_mw, xi_host, common, t_exp_days=0.0, free=()):
        super().__init__(name, params, band_index)
        self.free = tuple(free)
        self.t_exp_days = float(t_exp_days)
        self.rise = str(rise)
        self.pinned = dict(pinned)
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = None if redshift is None else float(redshift)
        self.dl_cm = None if dl_cm is None else float(dl_cm)
        self.xi = xi
        self.n_pre = n_pre
        # The extinction shapes are arrays; the rest of `common` is plain scalars/strings.
        self.xi_mw = None if xi_mw is None else np.asarray(xi_mw)
        self.xi_host = None if xi_host is None else np.asarray(xi_host)
        self.common = {k: v for k, v in common.items() if k not in ("xi_mw", "xi_host")}

    def _core(self, t, bidx, *free):
        """Observer-frame days -> AB magnitude per observation. Traced, pure jnp."""
        import jax.numpy as jnp

        from . import tde

        c = _context(self, free)
        v = c.values
        v.update(self.pinned)                  # pinned values are compile-time constants
        common = dict(self.common,
                      xi_mw=None if self.xi_mw is None else jnp.asarray(self.xi_mw),
                      xi_host=None if self.xi_host is None else jnp.asarray(self.xi_host))
        args = (t - c.shift, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                jnp.asarray(self.lam), c.redshift, c.dl_cm)
        if self.rise == "gaussian":
            return tde.gaussianrise_cooling_envelope_ab_magnitude(
                *args, v["peak_time"], v["sigma_t"], v["mbh_6"], v["stellar_mass"],
                v["eta"], v["alpha"], v["beta"], xi=self.xi, n_pre=self.n_pre, **common)
        return tde.cooling_envelope_ab_magnitude(
            *args, v["mbh_6"], v["stellar_mass"], v["eta"], v["alpha"], v["beta"], **common)

    def _magnitude(self, times, bidx, free):
        # JIT ONCE. `cooling_envelope` builds its scan step with `partial(...)` over the parameter
        # values, so calling it OUTSIDE jit bakes those concrete floats into the jaxpr as
        # constants -- a cache miss on every new parameter set, i.e. a fresh 0.6 s XLA compile of
        # a 5000-step scan per likelihood evaluation. Measured: 831 ms per predict() against
        # 9.1 ms for the identical computation under jit, a factor 92. Everything that is not
        # traced (the filter set, the extinction shape, `law`, `n_time`) is read inside the trace
        # from this object's plain data, so this compiles once per (n_obs, n_band) shape.
        import jax.numpy as jnp

        rel = _days_since(times, self.t_exp_days)        # host float64, before any cast (1.3)
        return self._jitted(None, lambda: self._core)(jnp.asarray(rel), jnp.asarray(bidx), *free)

    def predict_jax(self, theta, times, band_idx=None):
        """``predict`` for a traced caller: flat theta, integer bands, flux density in Jy.

        ``theta`` is ordered by ``Model.parameters``; ``band_idx`` indexes ``band_names``. Map the
        data's band strings once, outside the trace, with ``predict_jax.band_index(bands)``. One
        parameter set; ``jax.vmap`` it for a batch.

        **Requires float64** -- the envelope ODE's guard raises on the first trace, not here. And
        note the termination cliff: past the envelope's own last valid epoch the flux is exactly
        zero and the magnitude pins to ``mag_floor``, a step ``jax.grad`` cannot see. Measured over
        10,000 draws of redback's ``gaussianrise`` prior, that cliff lands inside a 0-100 d window
        on 15.6% of draws, so a gradient-based sampler on this model is not a safe default.
        """
        return super().predict_jax(theta, times, band_idx)


def tde_model(band_names, redshift=None, dl_cm=None, *, name=None, rise="gaussian", n_wave=None,
              n_time=None, f_debris=1.0, xi=1.0, n_pre=100, pin=None, prior=None,
              mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None,
              r_v_mw=3.1, r_v_host=3.1, law="f99", filter_set=None, band_aliases=None,
              default_system=None, t_exp_days=0.0, constraint="corrected", free=None,
              redshift_prior=None, **engine_kwargs):
    """Bind dataset context to the JAX cooling-envelope TDE and return a whisper ``Model``.

    Photometric, so it is a factory rather than a registry entry -- for exactly the reason
    :func:`kilonova_model` is: ``predict(parameters, times, bands)`` carries no filter set,
    no redshift and no luminosity distance, and a guessed distance would silently rescale
    every fitted black-hole mass.

    Parameters
    ----------
    band_names : sequence of str
        Filter names or labels, in the order the light curve's ``bands`` column maps to; see
        :func:`kilonova_model` (bare letters are LSST unless ``default_system=``).
    redshift, dl_cm : float
        Fixed for the dataset. ``dl_cm`` is deliberately NOT derived from ``redshift``:
        the cosmology is the caller's choice, and picking one quietly would make two codes
        disagree for a reason unrelated to the model. Leave both ``None`` when the redshift is
        fitted (``free=["redshift"]``).
    free : list of str or None
        Context values to fit as ordinary parameters: ``"t_exp"`` (the model's time zero -- the
        start of the rise, or fallback for ``rise="none"`` -- on the light curve's own clock)
        and/or ``"redshift"`` (the luminosity distance follows it through Planck18,
        :func:`whisper_cbpf.models.cosmology.luminosity_distance_cm`; the constraint wall reads
        the fitted value). Appended to ``parameters`` in that order and traced like the others:
        the TDE already takes traced times, so nothing else changes. ``t_exp`` needs
        ``prior=Prior({"t_exp": ...})``; the redshift's prior is ``prior["redshift"]``, else
        ``redshift_prior``, else Uniform(0.001, 1). A fitted redshift can bias a model comparison:
        read its posterior against its prior before ranking.
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none: a
        :attr:`whisper_cbpf.LightCurve.redshift_prior` hint or a distribution.
    rise : {"gaussian", "none"}
        ``"gaussian"`` is redback's ``gaussianrise_cooling_envelope``: a Gaussian rise
        normalised to meet the envelope at ``xi * tfb``, which is the model people actually
        fit, and it adds ``peak_time`` and ``sigma_t`` to the parameters. ``"none"`` is bare
        ``cooling_envelope``, valid only AFTER circularisation -- with it, ``times`` are days
        SINCE FALLBACK, whereas with a rise they are days from the start of the light curve.
    pin : dict or None
        Override which parameters are held fixed. ``{"beta": 2.0}`` pins one redback leaves
        free; ``{"beta": None}`` frees one redback pins -- but redback's file replaced those
        parameters' distributions with deltas, so freeing one also needs ``prior=``. (Only
        redback <= 1.15's ``cooling_envelope.prior`` pins anything; 1.18+ pins nothing.)
    prior : whisper_cbpf.priors.Prior or None
        Distributions that override redback's, per parameter. The default is redback's own
        prior, read from redback and **not modified** -- including where redback 1.15's file
        leaves only ``stellar_mass`` free. Without redback it is the latest release's
        (:func:`whisper_cbpf.models.jax.tde.fallback_prior`). Widen it here, at the call site,
        where the choice is visible in the analysis rather than buried in a library default.
    n_time :
        Which redback grid to reproduce. Default ``None`` is the installed redback's
        (:func:`whisper_cbpf.models.jax.tde.default_n_time`): 500 for 1.15 and 1.20, and when
        redback is absent; 5000 for 1.12. Pass ``n_time=5000`` for 1.12.0's finer grid. See
        CHANGE 6 in :mod:`whisper_cbpf.models.jax.tde` for what the difference costs (1.7% in T,
        and a termination index that moves under a 1e-6 change of a parameter at 500).
    dilation : bool
        ``True`` (default) applies the ``(1+z)`` flux factor, matching redback 1.15.1 and
        :mod:`whisper_cbpf.models.jax.kilonova`. ``False`` reproduces redback 1.12.0, which has
        no such factor. Worth 2.5*log10(1+z) magnitudes -- 0.053 at z = 0.05, 0.44 at z = 0.5
        -- so it is not a rounding detail. See CHANGE 7 in :mod:`whisper_cbpf.models.jax.tde`.
    filter_set, n_wave, default_system :
        The band integral, as in :func:`kilonova_model`: Gauss-16 per band by default, a
        ``FilterSet`` or ``make_filter_set`` dict via ``filter_set=``, the old grid rule via
        ``n_wave=``.
    t_exp_days : float
        The model's time zero on the light curve's own clock -- fallback for ``rise="none"``, the
        start of the rise otherwise -- so a raw-MJD light curve can be fitted as it is. Subtracted
        on the host in float64, as in the other factories. Default 0: ``times`` are
        already days since it.
    constraint : {"corrected", "redback", None}
        redback's cooling-envelope constraints (``eta >= eta_min``, ``beta <= beta_max``, and for
        the Gaussian rise a stitch within 35 sigma of the peak) as a hard wall: ``predict`` gives
        zero flux, so the CPU samplers reject the draw, and the JAX samplers apply
        ``predict_jax.constraint_ok`` -- ``-inf`` log-density, zero flux in the batched forward map
        ``"corrected"`` and ``"redback"`` agree here (the correction is Arnett's);
        ``None`` applies none, as whisper <= 0.1.0 did. The zeros are silent, in a sampler and in
        a direct ``predict`` alike (62 % of redback's bare-TDE prior, 72 % of the Gaussian-rise
        one): test a draw with ``predict_jax.constraint_ok(theta)`` before plotting it.
    **engine_kwargs
        redback's ``cooling_envelope`` keywords the engine takes: ``t_0_init``,
        ``binding_energy_const``, ``zeta``, ``hoverR``. Anything else raises here, not at the
        first predict.

    Notes
    -----
    ``float64 is required`` -- the engine raises otherwise, because the envelope ODE
    accumulates increments float32 cannot resolve. Enable it before creating any array::

        import jax; jax.config.update("jax_enable_x64", True)

    ``predict`` jits its model once, at factory time, and recompiles only when the number of
    observations or bands changes. That matters more here than for any other model in this
    package: the engine bakes its parameters into the scan's jaxpr, so an unjitted call pays
    a fresh ~0.6 s XLA compile of a 5000-step scan EVERY time -- measured at 831 ms per call
    against 9.1 ms jitted. A single light curve costs ~11.5 ms at ``n_time=5000``; batched it
    is far better still, 512 curves in 4.6 ms, i.e. 9.0 us each, so prefer vmapped consumers
    (ABC, SNPE, ``chain_method="vectorized"`` NUTS) over a single sequential chain.

    Examples
    --------
    >>> from whisper_cbpf.models.jax import tde_model
    >>> from whisper_cbpf.priors import Prior, Uniform
    >>> m = tde_model(["lsstg", "lsstr"], free=["t_exp", "redshift"],
    ...               prior=Prior({"t_exp": Uniform(-30.0, 0.0)}),
    ...               redshift_prior={"type": "Uniform", "low": 0.01, "high": 0.2})
    >>> m.parameters[-2:]
    ['t_exp', 'redshift']
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import tde

    if rise not in ("gaussian", "none"):
        raise ValueError(f"rise must be 'gaussian' or 'none', got {rise!r}")
    engine = [k for k, v in inspect.signature(tde.cooling_envelope).parameters.items()
              if v.kind is inspect.Parameter.KEYWORD_ONLY and k not in ("f_debris", "n_time")]
    unknown = sorted(set(engine_kwargs) - set(engine))
    if unknown:
        raise ValueError(
            f"tde_model got unknown keyword argument(s) {unknown}. Besides its own arguments it "
            f"takes the engine's: {engine} (redback's cooling_envelope keywords). The time origin "
            f"is t_exp_days=.")
    free = _check_free(free, "tde_model")
    _check_context(free, redshift, dl_cm, "tde_model")
    band_names = list(band_names)
    if name is None:
        name = "tde_gaussianrise_jax" if rise == "gaussian" else "tde_cooling_envelope_jax"
    n_time = tde.default_n_time() if n_time is None else n_time

    fs = _filter_set_dict(band_names, filter_set, n_wave, default_system)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    xi_mw = xi_host = None
    if ebv_mw is not None:
        xi_mw = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_mw, law=law, frame="observer")
    if ebv_host is not None and "redshift" not in free:    # a free z: the law runs in the trace
        xi_host = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_host, law=law, frame="rest")

    # redback's OWN prior, read from redback rather than transcribed -- including its FIXED
    # parameters. redback <= 1.15's `cooling_envelope.prior` pins mbh_6=1, eta=0.1, alpha=0.1,
    # beta=0.9 (bilby returns them as DeltaFunction), leaving stellar_mass as the only free
    # parameter; 1.18+ frees all five, and `gaussianrise_cooling_envelope.prior` pins nothing
    # in any release. whisper's prior layer has no delta,
    # so a pinned parameter is bound here and dropped from `parameters` -- exactly how
    # `kilonova_model` handles a fixed `temperature_floor`. `pin=` overrides, in either
    # direction: pass a float to fix one redback leaves free, or None to free one it pins.
    rb_prior, pinned = (tde.default_prior_gaussianrise() if rise == "gaussian"
                        else tde.default_prior())
    dists = dict(rb_prior.distributions)
    if prior is not None:                      # caller-supplied distributions win
        dists.update(prior.distributions)
    pinned = dict(pinned)
    for k, v in (pin or {}).items():
        if k not in tde.PARAMETERS_GAUSSIANRISE:
            raise ValueError(f"pin: {k!r} is not a parameter of this model. "
                             f"Known: {tde.PARAMETERS_GAUSSIANRISE}")
        if v is None:
            pinned.pop(k, None)
        else:
            pinned[k] = float(v)
    all_params = list(tde.PARAMETERS_GAUSSIANRISE if rise == "gaussian" else tde.PARAMETERS)
    params = [k for k in all_params if k not in pinned]
    missing = [k for k in all_params if k not in pinned and k not in dists]
    if missing:
        # redback's file replaced these parameters' DISTRIBUTIONS with delta functions, so
        # freeing one leaves nothing to sample from -- there is no redback prior to fall back
        # on. Say so, rather than inventing a range.
        raise ValueError(
            f"{missing} have no prior: redback's {rise!r} TDE prior file pins them as delta "
            f"functions, so unpinning leaves no distribution. Either leave them pinned, or "
            f"pass prior=Prior({{'{missing[0]}': LogUniform(...), ...}}) with a range you can "
            f"defend. (redback's own gaussianrise_cooling_envelope.prior gives mbh_6 "
            f"LogUniform(0.1, 20), eta LogUniform(1e-4, 0.1), alpha LogUniform(0.1, 1), "
            f"beta Uniform(1, 5) for the same parameters, if you want its choice.)")
    dists.update(_free_distributions(free, prior, redshift_prior, "tde_model"))
    params = params + list(free)
    prior = type(rb_prior)({k: dists[k] for k in params})
    ekw = dict(n_time=n_time, f_debris=f_debris, **engine_kwargs)
    common = dict(mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
                  xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law,
                  **ekw)

    ctx = _TDEPredict(
        name, params, _BandIndex(band_names, "tde_model(band_names=[...])", band_aliases),
        rise=rise, pinned=pinned, weights=weights, norms=norms, lam=lam, redshift=redshift,
        dl_cm=dl_cm, xi=xi, n_pre=n_pre, xi_mw=xi_mw, xi_host=xi_host, common=common,
        t_exp_days=t_exp_days, free=free)
    rb_name = "gaussianrise_cooling_envelope" if rise == "gaussian" else "cooling_envelope"
    ctx.set_constraint(rb_name, constraint, redshift=redshift, xi=xi,
                       binding_energy_const=engine_kwargs.get("binding_energy_const", 0.8))
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    what = ("Gaussian rise + Sarin & Metzger cooling envelope"
            if rise == "gaussian" else "Sarin & Metzger cooling envelope (post-circularisation only)")
    pin_note = ("; redback pins " + ", ".join(f"{k}={v:g}" for k, v in sorted(pinned.items()))
                if pinned else "")
    return Model(name=name, predict=predict, parameters=params, default_prior=prior,
                 predict_jax=predict_jax,
                 description=(f"TDE: {what} (JAX/GPU){pin_note}. Physics identical to redback's "
                               f"_cooling_envelope at n_time={n_time}; the light curve ends at "
                               f"the envelope's own death rather than at redback's diverged "
                               f"Rv test (see CHANGE 3). Requires float64. "
                               f"{_constraint_note(rb_name, constraint)}{_free_note(free)}"
                               f"{_floor_note(mag_floor)}"))


def register_tde(band_names, redshift=None, dl_cm=None, *, name=None, **kwargs):
    """Build a TDE ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`tde_model`.
    name : str, optional
        The registry name; default :func:`tde_model`'s (``"tde_gaussianrise_jax"`` for the
        default Gaussian rise).

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> m = wp.register_tde(["ztfg", "ztfr"], redshift=0.02, dl_cm=2.8e26)
    >>> m.name, m.parameters[:3]
    ('tde_gaussianrise_jax', ['peak_time', 'sigma_t', 'mbh_6'])
    """
    return _register(tde_model(band_names, redshift, dl_cm, name=name, **kwargs))


# --- supernovae ---
def supernova_models():
    """The names :func:`supernova_model` accepts.

    :func:`whisper_cbpf.compare` also takes each of them as a family name, bound to the data.

    Returns
    -------
    list of str

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> {"arnett", "basic_magnetar_powered", "csm_shock_and_arnett"} <= set(wp.supernova_models())
    True
    """
    from . import supernova as sn

    return sn.model_names()


class _SupernovaPredict(_PhotometricPredict):
    """``predict``/``predict_jax`` for :func:`supernova_model`.

    The one family that does NOT fit the shape of the other four, and the difference is real: its
    dense diffusion grid is built FROM the observation times (CHANGE 1 in
    :mod:`whisper_cbpf.models.jax.supernova`), so the times are a compile-time constant rather
    than a traced argument and there is one compiled core PER set of epochs. That is why the base
    class's jit cache is keyed (``_jitted(key, build)``) instead of holding a single core: the key
    is ``t_src.tobytes()``, exactly as the old closure's ``_cores`` dict was. Everything else --
    plain data in, jit rebuilt lazily per process -- is unchanged from its siblings.

    With ``fixed_kw`` set (``diffusion_grid="fixed"``, the default when ``t_exp`` or the redshift
    is free) the diffusion runs on fixed source-frame epochs instead, the times are traced, and
    there is ONE compiled core per number of observations (the observed-epoch path of
    :func:`whisper_cbpf.models.jax.supernova.ab_magnitude_at`). ``phase_limits`` is
    ``(earliest t_exp, lowest z)`` the default prior allows, for the grid-coverage check.
    """

    def __init__(self, name, params, band_index, *, model, pinned, grid_kw, weights, norms, lam,
                 redshift, dl_cm, t_exp_days, xi_mw, xi_host, common, free=(), fixed_kw=None,
                 phase_limits=None):
        super().__init__(name, params, band_index)
        self.free = tuple(free)
        self.model = str(model)
        self.pinned = dict(pinned)
        self.grid_kw = dict(grid_kw)
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = None if redshift is None else float(redshift)
        self.dl_cm = None if dl_cm is None else float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.xi_mw = None if xi_mw is None else np.asarray(xi_mw)
        self.xi_host = None if xi_host is None else np.asarray(xi_host)
        self.common = {k: v for k, v in common.items() if k not in ("xi_mw", "xi_host")}
        self.spin_down_warned = False
        self.fixed_kw = None if fixed_kw is None else dict(fixed_kw)
        self.phase_limits = phase_limits
        self._fixed = None

    def fixed_grid(self):
        """The :class:`~whisper_cbpf.models.jax.supernova.FixedGrid` of the observed-epoch path,
        built once (as concrete data, whatever trace this is called from)."""
        import jax

        from . import supernova as sn

        if self._fixed is None:
            with jax.ensure_compile_time_eval():
                self._fixed = sn.build_fixed_grid(**self.fixed_kw)
        return self._fixed

    def check_epochs(self, times, prior=None):
        """Raise when ``times`` reach past the fixed diffusion epochs for a draw ``prior`` allows.

        The host-side check for a density built on concrete epochs (:func:`whisper_cbpf.log_density`
        and the JAX samplers' densities): the latest epoch at the earliest explosion time and the
        lowest redshift ``prior`` allows (the factory's own prior when ``None``). A traced caller
        cannot run it inside the trace, and past the fixed epochs the luminosity is held at its
        last value, so a density that skipped it would be silently wrong there. Nothing to check
        with ``diffusion_grid="data"``.
        """
        if self.fixed_kw is None:
            return
        t = np.asarray(times, dtype=float)
        if not t.size:
            return
        t_lo, z_lo = self.phase_limits
        dists = {} if prior is None else prior.distributions
        if "t_exp" in self.free and "t_exp" in dists:
            t_lo = _bounds(dists["t_exp"])[0]
            if not np.isfinite(t_lo):
                raise ValueError(
                    f"model {self.name!r}: its explosion-time prior {dists['t_exp']!r} has no lower "
                    f"bound, and the fixed diffusion epochs must cover the latest epoch at the "
                    f"earliest explosion. Give t_exp a prior with a finite lower bound (a Uniform, or "
                    f"TruncatedNormal(mu, sigma, low, high)).")
        if "redshift" in self.free and "redshift" in dists:
            z_lo = _bounds(dists["redshift"])[0]
        self._check_phase((float(np.max(t)) - t_lo) / (1.0 + z_lo),
                          "the epochs, at the earliest t_exp and lowest redshift the prior allows,")

    def _check_phase(self, tau_max, why):
        """Raise when a source-frame phase lies past the fixed epochs (the cubic would hold)."""
        top = self.fixed_kw["max_phase_days"]
        if tau_max > top * (1.0 + 1e-12):
            raise ValueError(
                f"model {self.name!r}: {why} reach {tau_max:.4g} source-frame days after the "
                f"explosion, past the {top:.4g} d this model's fixed diffusion epochs cover. "
                f"Rebuild it with max_phase_days={float(np.ceil(1.2 * tau_max)):g} (or times= "
                f"covering these epochs), or narrow the t_exp / redshift prior.")

    def predict(self, parameters, times, bands=None):
        """whisper contract. Also warns, once per model, when a magnetar's spin-down is too fast
        for the dense grid (CHANGE 1). This is the only place that check runs: ``predict_jax``
        (the GPU samplers) is traced and never checks, and neither does the CPU redback adapter;
        see :func:`~whisper_cbpf.models.jax.supernova.warn_if_spin_down_unresolved`."""
        from . import supernova as sn

        v = {**{k: parameters[k] for k in self.params}, **self.pinned}
        t_src = self._source_time(times, v)
        if self.fixed_kw is not None and t_src.size:
            self._check_phase(float(np.max(t_src)), "these epochs")
        if not self.spin_down_warned and "p0" in v:
            dense = sn.dense_grid(t_src,
                                  self.grid_kw.get("dense_resolution", sn.DENSE_RESOLUTION),
                                  spacing=self.grid_kw.get("spacing", "geometric"))
            frac = sn.warn_if_spin_down_unresolved(
                dense, *(float(v[k]) for k in ("p0", "bp", "mass_ns", "theta_pb")),
                convention=self.common["magnetar_convention"])
            self.spin_down_warned = frac > sn.SPIN_DOWN_UNRESOLVED_LIMIT
        return super().predict(parameters, times, bands)

    def _core_for(self, t_src):
        """The jitted core for these SOURCE-frame epochs, building the dense grid on a miss.

        JIT ONCE, per (grid, n_obs) shape. The grid is DATA -- built from the observation times at
        setup -- so it is a constant of the compilation rather than a traced argument, and a new
        time array means a new grid and one recompile. Memoising on the times is what lets a
        sampler calling predict() in a loop with the same light curve compile exactly once.
        """
        import jax.numpy as jnp

        from . import supernova as sn

        def build():
            g = sn.build_sn_grid(t_src, **self.grid_kw)
            common = dict(self.common,
                          xi_mw=None if self.xi_mw is None else jnp.asarray(self.xi_mw),
                          xi_host=None if self.xi_host is None else jnp.asarray(self.xi_host))

            def _core(bidx, *free):
                v = dict(zip(self.params, free))
                v.update(self.pinned)          # pinned values are compile-time constants
                return sn.ab_magnitude_of(self.model, g, v, bidx, jnp.asarray(self.weights),
                                          jnp.asarray(self.norms), jnp.asarray(self.lam),
                                          self.redshift, self.dl_cm, **common)

            return _core

        return self._jitted(t_src.tobytes(), build)

    def _fixed_core(self):
        """The one jitted core of the observed-epoch path: (days since ``t_exp_days``, band
        indices, *free) -> AB magnitude. ``t_exp`` and the redshift, when free, are in ``free``."""
        import jax.numpy as jnp

        from . import supernova as sn

        fixed = self.fixed_grid()

        def _core(rel_days, bidx, *free):
            c = _context(self, free)
            v = c.values
            v.update(self.pinned)              # pinned values are compile-time constants
            common = dict(self.common,
                          xi_mw=None if self.xi_mw is None else jnp.asarray(self.xi_mw),
                          xi_host=None if self.xi_host is None else jnp.asarray(self.xi_host))
            tau = (rel_days - c.shift) / (1.0 + c.redshift)
            return sn.ab_magnitude_at(self.model, fixed, tau, v, bidx, jnp.asarray(self.weights),
                                      jnp.asarray(self.norms), jnp.asarray(self.lam),
                                      c.redshift, c.dl_cm, **common)

        return _core

    def _source_time(self, times, values=None):
        """Source-frame days since the explosion; ``values`` supplies a free t_exp / redshift."""
        z = values["redshift"] if "redshift" in self.free else self.redshift
        t0 = values["t_exp"] if "t_exp" in self.free else self.t_exp_days
        return np.asarray(_source_time(times, float(z), float(t0)), dtype=float)

    def _magnitude(self, times, bidx, free):
        import jax.numpy as jnp

        if self.fixed_kw is not None:
            rel = _days_since(times, self.t_exp_days)    # host float64, before any cast (1.3)
            return self._jitted("fixed", self._fixed_core)(jnp.asarray(rel), jnp.asarray(bidx),
                                                           *free)
        return self._core_for(self._source_time(times))(jnp.asarray(bidx), *free)

    def predict_jax(self, theta, times, band_idx=None):
        """``predict`` for a traced caller: flat theta, integer bands, flux density in Jy.

        ``theta`` is ordered by ``Model.parameters``; ``band_idx`` indexes ``band_names``. Map the
        data's band strings once, outside the trace, with ``predict_jax.band_index(bands)``.

        **With ``diffusion_grid="data"`` (the default when nothing is free), ``times`` must be
        CONCRETE, not traced.** Unlike the kilonova and the TDE, that path's dense grid is built
        *from* the observation times (CHANGE 1) and the compiled core is memoised on
        ``t_src.tobytes()``, so the times are a closure constant of the compilation, not a traced
        argument. ``jax.vmap`` it over ``theta`` alone (``in_axes=(0, None, None)``) -- vmapping
        or jitting over ``times`` raises a JAX tracer-conversion error. With
        ``diffusion_grid="fixed"`` (the default with a free ``t_exp`` or redshift) the times may be
        traced, and concrete times are checked against the fixed epochs' span for every
        ``t_exp`` and redshift the default prior allows. Also requires float64: these engines carry
        cgs luminosities of 1e43-1e46 erg/s and float32 stops at 3.4e38.
        """
        import jax
        import jax.numpy as jnp

        if band_idx is None:                    # same requirement, same message, as predict
            raise ValueError(f"model {self.name!r} is photometric and needs a `band_idx` array")
        if self.fixed_kw is not None:
            try:
                t_host = np.asarray(times, dtype=float)
            except (TypeError, jax.errors.TracerArrayConversionError):
                t_host = None                   # traced epochs: nothing to check on the host
            if t_host is not None and t_host.size:
                t_lo, z_lo = self.phase_limits
                self._check_phase((float(np.max(t_host)) - t_lo) / (1.0 + z_lo),
                                  "the epochs, at the earliest t_exp and lowest redshift the "
                                  "prior allows,")
            self.fixed_grid()                   # concrete, outside the caller's trace
            mag = self._magnitude(times if t_host is None else t_host, band_idx,
                                  [theta[i] for i in range(len(self.params))])
            return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag)
        try:
            t_src = self._source_time(np.asarray(times, dtype=float))
        except (TypeError, ValueError, jax.errors.TracerArrayConversionError) as exc:
            raise TypeError(
                f"model {self.name!r}: predict_jax needs CONCRETE times -- the dense grid is built "
                f"from them and the compiled core is memoised on their bytes. vmap over theta only "
                f"(in_axes=(0, None, None)) and keep `times` outside the trace.") from exc
        # BUILD THE GRID OUTSIDE THE CALLER'S TRACE. `_core_for` may build a NEW dense grid, and
        # `build_sn_grid`'s jnp calls would then produce tracers belonging to whatever trace we are
        # being called from -- vmap's, or jit's. Those tracers are closed over by `_core` and
        # memoised in the jit cache FOREVER, so the entry is poisoned the moment that trace ends:
        # the next call raises UnexpectedTracerError, and because `predict` shares the same cache
        # it breaks too, taking per_band_metrics / predictive_metrics / plot_ppc with it. Measured
        # on HEAD before this guard: register_supernova(...) then a vmapped abc_gpu fit raised
        # `UnexpectedTracerError (float64[1000] = dense_times)`. It is order-dependent -- a
        # concrete predict() first leaves the entry clean -- which is why the tests missed it.
        # ensure_compile_time_eval forces the grid to be built as concrete data.
        with jax.ensure_compile_time_eval():
            core = self._core_for(t_src)
        mag = core(jnp.asarray(band_idx), *[theta[i] for i in range(len(self.params))])
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag)


def supernova_model(model, band_names, redshift=None, dl_cm=None, *, name=None, n_wave=None,
                    t_exp_days=0.0, dense_resolution=None, spacing="geometric",
                    csm_interp=True, magnetar_convention="1.15",
                    interaction=True, dilation=True, pin=None, prior=None, mag_floor=40.0,
                    ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                    filter_set=None, times=None, band_aliases=None, default_system=None,
                    constraint="corrected", free=None, redshift_prior=None,
                    diffusion_grid=None, max_phase_days=None, epochs_per_decade=None):
    """Bind dataset context to one of the JAX supernova models and return a whisper ``Model``.

    Photometric, so it is a factory rather than a registry entry -- for exactly the reason
    :func:`kilonova_model` and :func:`tde_model` are: ``predict(parameters, times, bands)``
    carries no filter set, no redshift and no luminosity distance, and a guessed distance would
    silently rescale every fitted ejecta mass.

    Parameters
    ----------
    model : str
        One of :func:`supernova_models` -- ``arnett``, ``shock_cooling_and_arnett``,
        ``basic_magnetar_powered``, ``slsn``, ``magnetar_nickel``, ``csm_shock_and_arnett``,
        ``sn_exponential_powerlaw``, ``sn_fallback``, ``sn_nickel_fallback``,
        ``general_magnetar_slsn``, ``type_1a``, ``type_1c``.
    band_names : sequence of str
        Filter names or labels, in the order the light curve's ``bands`` column maps to; see
        :func:`kilonova_model` (bare letters are LSST unless ``default_system=``). The band
        integral is ``filter_set``/``n_wave``'s, as there: Gauss-16 per band by default.
    redshift, dl_cm : float
        Fixed for the dataset. ``dl_cm`` is deliberately NOT derived from ``redshift``: the
        cosmology is the caller's choice, and picking one quietly would make two codes disagree
        for a reason unrelated to the model. Leave both ``None`` when the redshift is fitted
        (``free=["redshift"]``).
    free : list of str or None
        Context values to fit as ordinary parameters: ``"t_exp"`` (the explosion time, on the
        light curve's own clock, the clock of ``times`` and ``t_exp_days``) and/or ``"redshift"``
        (the luminosity distance then follows it through Planck18,
        :func:`whisper_cbpf.models.cosmology.luminosity_distance_cm`, within 1e-6 mag of
        astropy). They are appended to ``parameters`` in that order and are traced like any
        other parameter: ``jit`` and ``vmap`` over many explosion times and redshifts compile
        once. ``t_exp`` needs ``prior=Prior({"t_exp": ...})`` (its window depends on the first
        detection and the last non-detection); the redshift's prior is ``prior["redshift"]``,
        else ``redshift_prior``, else Uniform(0.001, 1). An epoch before the explosion is at
        ``mag_floor`` (no light). In a fit the rows before the first detection are left out of
        the likelihood (the pre-event rule) and set the default ``t_exp`` window instead, so a
        limit there never enters the likelihood. **A fitted redshift can
        bias a model comparison:** a model can move the source closer or farther to buy a fit it
        cannot make at the true distance, and a BIC charges every model the same for that
        freedom (on real ZTF supernovae, winning fits sat ~1 sigma_z low). Read the redshift
        posterior against its prior -- is it narrower, and where does it sit -- before ranking.
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none: a
        :attr:`whisper_cbpf.LightCurve.redshift_prior` hint (``lc.redshift_prior``, read by
        :func:`whisper_cbpf.io.schema.redshift_distribution`) or a distribution.
    diffusion_grid : {None, "data", "fixed"}
        Where the diffusion integral is evaluated. ``"data"`` is redback's own scheme, on the
        observation epochs: exact to redback (<= 8e-6 mag on 200 prior draws), but the epochs are a
        compile-time constant, so ``t_exp`` and the redshift cannot move. ``"fixed"`` evaluates
        it on fixed source-frame epochs (``epochs_per_decade`` to a decade, up to
        ``max_phase_days``), interpolates only the bolometric luminosity to the observations, and
        evaluates the photosphere and the SED at the observations only
        (:func:`whisper_cbpf.models.jax.supernova.ab_magnitude_at`): one compile for any epochs,
        within 9e-4 mag of redback over 200 prior draws per family. ``None`` (default) is
        ``"data"`` with nothing free and ``"fixed"`` otherwise.
    max_phase_days : float or None
        ``"fixed"`` only: the latest source-frame day since the explosion the model covers.
        Default: 200 d, or more if ``times`` reach later for the earliest ``t_exp`` and lowest
        redshift the prior allows. A later epoch raises, naming the value to rebuild with.
    epochs_per_decade : float or None
        ``"fixed"`` only: the epochs' resolution (default 30, 134 epochs to 200 d; at 24
        the magnetar was 2.6e-3 mag off redback, at 36 it is 2.8e-4, for 1.2x the cost).
    times : array or None
        With ``diffusion_grid="data"``: the epochs (on the light curve's clock) the dense
        diffusion grid will be sized from. Leave it ``None`` and the grid is rebuilt (and the
        model recompiled) the first time ``predict`` sees a new time array, which is correct but
        pays one XLA compile per distinct set of epochs. Pass the light curve's own times to
        build it once at the factory. With ``"fixed"``: only sizes ``max_phase_days``.
    t_exp_days : float
        Explosion epoch on the same clock as ``times``, subtracted before the redshift. With a
        free ``t_exp`` it is only the reference the epochs are shifted by on the host in
        float64; set it near the data when the clock is MJD.
    spacing : {"geometric", "linear"}
        The dense grids (CHANGE 1 and 4 in :mod:`whisper_cbpf.models.jax.supernova`).
        ``"geometric"`` (default) is redback 1.20's, ``"linear"`` redback <= 1.15's -- or pass
        ``**supernova.REDBACK_GRID_PRESETS[...]``. The linear grid cannot resolve a magnetar
        spin-down shorter than its ~0.3 d first cell and was up to 21 mag too bright against
        1.20; ``predict`` warns when either grid misses more than 10% of ``E_rot``.
    csm_interp : bool
        ``True`` (default) interpolates the CSM breakout off redback's 300 nodes, as redback
        does; ``False`` evaluates its closed form at the epochs (exact; whisper <= 0.1.0).
    magnetar_convention : {"1.15", "1.12"}
        Which redback's ``basic_magnetar`` (CHANGE 5 in :mod:`whisper_cbpf.models.jax.supernova`).
        Worth up to 0.75 mag, and not a constant offset. Ignored by models that have no
        magnetar. Default "1.15", which is also MOSFiT's.
    interaction : bool
        Whether ``sn_fallback`` / ``sn_nickel_fallback`` apply the diffusion they declare
        (CHANGE 7b). redback does NOT -- pass ``False`` to reproduce it, and know that
        ``kappa``, ``kappa_gamma`` and (for ``sn_fallback``) ``mej`` then do nothing at all.
        Worth up to 15 mag. Ignored by every other model.
    dilation : bool
        ``True`` (default) applies the ``(1+z)`` flux factor, matching redback 1.15.1 and
        :mod:`whisper_cbpf.models.jax.kilonova`. ``False`` reproduces 1.12.0 (CHANGE 6).
    pin : dict or None
        Override which parameters are held fixed. ``{"kappa": 0.1}`` pins one redback leaves
        free; ``{"line_width": None}`` frees one redback's prior file pins -- but a freed
        parameter needs a distribution, so that also needs ``prior=``.
    prior : whisper_cbpf.priors.Prior or None
        Distributions overriding redback's, per parameter. The default is redback's own prior,
        read from redback and not modified.
    constraint : {"corrected", "redback", None}
        redback's ``Constraint`` priors for ``arnett`` (nuclear burning >= kinetic energy),
        ``basic_magnetar_powered``, ``slsn`` and ``general_magnetar_slsn`` (rotational >= kinetic
        energy; ``slsn`` also a 100-500 d nebular time) as a hard wall: ``predict`` gives zero
        flux, so the CPU samplers reject the draw, and the JAX samplers apply
        ``predict_jax.constraint_ok`` -- ``-inf`` log-density, zero flux in the batched forward map
        ``"corrected"`` (default) bounds the Arnett
        kinetic energy by 1.51e18 erg/g of nickel; ``"redback"`` keeps redback's 1.91e19,
        12.6x too lenient; ``None`` applies none, as whisper <= 0.1.0 did. The other models have none. The
        zeros are silent, in a sampler and in a direct ``predict`` alike (60 % of redback's arnett
        prior, 93-97 % of the slsn ones): test a draw with ``predict_jax.constraint_ok(theta)``
        before plotting it.

    Notes
    -----
    ``float64 is required``, as for the TDE, but for a different reason. The TDE's is an
    accumulator: its envelope ODE adds increments below float32's epsilon. This family's is range --
    the engines carry bolometric luminosities in cgs, 1e43 to 1e46 erg/s, against float32's 3.4e38
    ceiling, so EVERY engine returns ``inf`` and every band comes back at ``mag_floor`` (CHANGE 8).

    **The guard fires at the first ``predict``, exactly like the TDE's**, so ``register_supernova``
    SUCCEEDS in float32 and the model enters the registry. This docstring claimed the opposite --
    that ``build_sn_grid`` raises at factory time so ``register_supernova`` itself fails -- until it
    was measured: in float32, 12/12 supernovae register and all twelve appear in ``list_models()``.
    ``build_sn_grid`` does carry the check, but it is only reached at factory time when ``times=``
    is passed, and the default is ``times=None``. Enable x64 before creating any array::

        import jax; jax.config.update("jax_enable_x64", True)

    redback's ``Constraint`` priors are rejection conditions rather than distributions, so they
    are not in the ``Prior``: ``constraint=`` arms ``predict_jax.constraint_ok``, which the JAX
    samplers apply, and the host ``predict`` the CPU samplers call returns zero flux for a draw
    that breaks one, as the CPU redback adapter does. ``predict_jax`` stays the physics, as
    redback's function is. Pass ``constraint=None`` to compare the physics at any draw.
    (redback 1.12's ``slsn.prior`` also carried two ``Constraint`` lines of its own; they are
    reported in the description when that redback is installed.)

    Examples
    --------
    An LSST alert with no spectroscopic redshift: fit the explosion time and the redshift.

    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax import supernova_model
    >>> from whisper_cbpf.priors import Prior, Uniform
    >>> m = supernova_model("arnett", ["lsstg", "lsstr"], free=["t_exp", "redshift"],
    ...                     prior=Prior({"t_exp": Uniform(-20.0, 0.0)}),
    ...                     redshift_prior={"type": "Uniform", "low": 0.02, "high": 0.2})
    >>> m.parameters[-2:]
    ['t_exp', 'redshift']
    >>> p = dict(f_nickel=0.1, mej=2.0, vej=5e3, kappa=0.1, kappa_gamma=10.0,
    ...          temperature_floor=5000.0, t_exp=-8.0, redshift=0.05)
    >>> flux = m.predict(p, np.array([-10.0, 0.0, 10.0]), np.array(["lsstg", "lsstr", "lsstg"]))
    >>> round(float(-2.5 * np.log10(flux[0] / 3631.0)), 6)   # before the explosion: mag_floor
    40.0
    >>> bool(flux[2] > flux[1] > flux[0])
    True
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import supernova as sn

    if model not in sn.MODELS:
        raise ValueError(f"unknown supernova model {model!r}. Known: {sn.model_names()}")
    free = _check_free(free, "supernova_model")
    _check_context(free, redshift, dl_cm, "supernova_model")
    if diffusion_grid is None:
        diffusion_grid = "fixed" if free else "data"
    if diffusion_grid not in ("data", "fixed"):
        raise ValueError(f"diffusion_grid must be 'data' or 'fixed', got {diffusion_grid!r}")
    if diffusion_grid == "data" and free:
        raise ValueError(f"supernova_model: diffusion_grid='data' builds the diffusion grid from "
                         f"the concrete epochs, which free={list(free)} moves. Use "
                         f"diffusion_grid='fixed' (the default with free=).")
    if diffusion_grid == "data" and (max_phase_days is not None or epochs_per_decade is not None):
        raise ValueError("supernova_model: max_phase_days= and epochs_per_decade= size the fixed "
                         "diffusion epochs; pass diffusion_grid='fixed' (or free=) with them.")
    band_names = list(band_names)
    name = name or f"{model}_jax"

    fs = _filter_set_dict(band_names, filter_set, n_wave, default_system)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    xi_mw = xi_host = None
    if ebv_mw is not None:
        xi_mw = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_mw, law=law, frame="observer")
    if ebv_host is not None and "redshift" not in free:    # a free z: the law runs in the trace
        xi_host = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_host, law=law, frame="rest")

    # redback's OWN prior, read from redback rather than transcribed, including its FIXED
    # parameters -- bilby turns a bare float in a prior file into a DeltaFunction, which is how
    # `type_1a.prior`'s line_wavelength/line_width/line_amplitude arrive. whisper's prior layer
    # has no delta, so a pinned parameter is bound here and dropped from `parameters`, exactly
    # as `kilonova_model` handles a fixed `temperature_floor`.
    rb_prior, pinned, constraints = sn.default_prior(model)
    dists = dict(rb_prior.distributions)
    if prior is not None:                      # caller-supplied distributions win
        dists.update(prior.distributions)
    pinned = dict(pinned)
    all_params = list(sn.PARAMETERS[model]) + sorted(
        k for k in pinned if k not in sn.PARAMETERS[model])
    for k, v in (pin or {}).items():
        if k not in all_params:
            raise ValueError(f"pin: {k!r} is not a parameter of {model!r}. "
                             f"Known: {all_params}")
        if v is None:
            pinned.pop(k, None)
        else:
            pinned[k] = float(v)
    params = [k for k in all_params if k not in pinned]
    missing = [k for k in params if k not in dists]
    if missing:
        raise ValueError(
            f"{missing} have no prior: redback's {model!r} prior file pins them as delta "
            f"functions, so unpinning leaves no distribution. Either leave them pinned, or "
            f"pass prior=Prior({{'{missing[0]}': Uniform(...), ...}}) with a range you can "
            f"defend.")
    dists.update(_free_distributions(free, prior, redshift_prior, "supernova_model"))
    params = params + list(free)
    model_prior = type(rb_prior)({k: dists[k] for k in params})

    common = dict(magnetar_convention=magnetar_convention, interaction=interaction,
                  mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
                  xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)

    grid_kw = dict(spacing=spacing, csm_interp=csm_interp)
    if dense_resolution is not None:
        grid_kw["dense_resolution"] = dense_resolution
    fixed_kw = phase_limits = None
    if diffusion_grid == "fixed":
        # the earliest explosion and the lowest redshift the prior allows give the latest phase
        t_lo = _bounds(dists["t_exp"])[0] if "t_exp" in free else float(t_exp_days)
        if not np.isfinite(t_lo):
            raise ValueError(
                f"supernova_model: the explosion-time prior {dists['t_exp']!r} has no lower bound, "
                f"and the fixed diffusion epochs a free t_exp needs are sized from the earliest "
                f"explosion it allows. Give t_exp a prior with a finite lower bound (a Uniform, or "
                f"TruncatedNormal(mu, sigma, low, high)).")
        z_lo = _bounds(dists["redshift"])[0] if "redshift" in free else float(redshift)
        phase_limits = (t_lo, z_lo)
        if max_phase_days is None:
            max_phase_days = sn.DEFAULT_MAX_PHASE_DAYS
            if times is not None and np.size(times):
                max_phase_days = max(max_phase_days,
                                     (float(np.max(times)) - t_lo) / (1.0 + z_lo))
        fixed_kw = dict(max_phase_days=float(max_phase_days),
                        epochs_per_decade=(sn.FIXED_GRID_EPOCHS_PER_DECADE
                                           if epochs_per_decade is None else epochs_per_decade),
                        dense_resolution=grid_kw.get("dense_resolution", sn.DENSE_RESOLUTION),
                        spacing=spacing, csm_interp=csm_interp)
    ctx = _SupernovaPredict(
        name, params,
        _BandIndex(band_names, f"supernova_model({model!r}, band_names=[...])", band_aliases),
        model=model, pinned=pinned, grid_kw=grid_kw, weights=weights, norms=norms, lam=lam,
        redshift=redshift, dl_cm=dl_cm, t_exp_days=t_exp_days, xi_mw=xi_mw, xi_host=xi_host,
        common=common, free=free, fixed_kw=fixed_kw, phase_limits=phase_limits)
    if times is not None and fixed_kw is None:  # build (and compile on first call) up front
        ctx._core_for(np.asarray(_source_time(times, redshift, t_exp_days), dtype=float))
    ctx.set_constraint(model, constraint, redshift=redshift)
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    notes = []
    if pinned:
        notes.append("redback pins " + ", ".join(f"{k}={v:g}" for k, v in sorted(pinned.items())))
    if constraints:
        notes.append("redback's prior file also declares Constraint(s) "
                     + ", ".join(f"{k} in [{lo:g}, {hi:g}]"
                                 for k, (lo, hi) in sorted(constraints.items()))
                     + ", NOT applied here")
    if model in ("sn_fallback", "sn_nickel_fallback"):
        notes.append(f"interaction={interaction} (redback applies none; worth up to 15 mag)")
    if any(p in sn.PARAMETERS[model] for p in ("p0", "bp")):
        notes.append(f"magnetar_convention={magnetar_convention}")
    tail = ("; " + "; ".join(notes)) if notes else ""
    grid_note = ""
    if fixed_kw is not None:
        grid_note = (f"Diffusion on fixed source-frame epochs ({fixed_kw['epochs_per_decade']:g} "
                     f"per decade to {fixed_kw['max_phase_days']:.4g} d), the SED at the "
                     f"observations only; within 9e-4 mag of redback. ")
    return Model(name=name, predict=predict, parameters=params, default_prior=model_prior,
                 predict_jax=predict_jax,
                 description=(f"Supernova: redback's {model} (JAX/GPU){tail}. Bolometric "
                              f"physics matches redback to ~1e-16; the band integral uses the "
                              f"real bandpass rather than redback's sncosmo spline. See "
                              f"whisper_cbpf.models.jax.supernova for the numbered deviations. "
                              f"{grid_note}{_constraint_note(model, constraint)}"
                              f"{_free_note(free)}{_floor_note(mag_floor)}"))


def _source_time(times, redshift, t_exp_days):
    """Observer-frame days -> SOURCE-frame days, redback's ``calc_kcorrected_properties``."""
    return (np.asarray(times, dtype=float) - t_exp_days) / (1.0 + redshift)


def register_supernova(model, band_names, redshift=None, dl_cm=None, *, name=None, **kwargs):
    """Build a supernova ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    model : str
        One of :func:`supernova_models`.
    band_names, redshift, dl_cm, **kwargs
        As :func:`supernova_model`.
    name : str, optional
        The registry name; default ``f"{model}_jax"``.

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.cosmology import luminosity_distance_cm
    >>> m = wp.register_supernova("arnett", ["ztfg", "ztfr"], redshift=0.05,
    ...                           dl_cm=float(luminosity_distance_cm(0.05)))
    >>> m.name, "arnett_jax" in wp.list_models()
    ('arnett_jax', True)
    """
    return _register(supernova_model(model, band_names, redshift, dl_cm, name=name, **kwargs))
