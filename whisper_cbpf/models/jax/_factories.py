"""JAX transient models, and their adapters onto whisper's ``Model`` contract.

The physics modules (:mod:`whisper_cbpf.models.jax.kilonova`, :mod:`whisper_cbpf.models.jax.flare`) are kept
free of whisper coupling -- ``kilonova.py`` imports nothing but numpy and jax (plus a lazy sncosmo
inside ``make_filter_set``). That is deliberate: the JAX physics can then be validated standalone
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

import numpy as np

AB_ZEROPOINT_JY = 3631.0

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
    ``{"g": "sdssg", "r": "sdssr"}``. This is needed more often than not: ``make_filter_set`` takes
    sncosmo names (``sdssg``, ``ztfr``, ``lsstg``) while survey CSVs — including
    ``tests/data/at2017gfo.csv`` — carry bare ``g``/``r``/``i``.

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

    Every photometric factory below used to build these two entry points as *local closures*.
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
        self._jit_cache = None                  # process-local; see __getstate__

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
        """whisper contract: (parameters, times, bands) -> flux density in Jy."""
        if bands is None:
            raise ValueError(f"model {self.name!r} is photometric and needs a `bands` array")
        idx = self.band_index(bands)
        mag = self._magnitude(np.asarray(times, dtype=float), idx,
                              [float(parameters[k]) for k in self.params])
        # whisper models return FLUX; convert once, at the end. Never combine magnitudes.
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))

    def predict_jax(self, theta, times, band_idx=None):
        """``predict`` for a traced caller: flat theta, integer bands, flux density in Jy.

        ``theta`` is ordered by ``Model.parameters``; ``band_idx`` indexes ``band_names``. Map the
        data's band strings once, outside the trace, with ``predict_jax.band_index(bands)`` --
        that is the same lookup ``predict`` does, including its error message. One parameter set;
        ``jax.vmap`` it for a batch.
        """
        if band_idx is None:                    # same requirement, same message, as predict
            raise ValueError(f"model {self.name!r} is photometric and needs a `band_idx` array")
        mag = self._magnitude(times, band_idx, [theta[i] for i in range(len(self.params))])
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag)


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

    def __call__(self, theta, times, band_idx=None):
        return self.ctx.predict_jax(theta, times, band_idx)


# --------------------------------------------------------------------------- flare
def flare_model():
    """The JAX flare as a ``whisper_cbpf.models.Model``. Self-contained, auto-registered."""
    from . import flare
    return flare.get_model()


# --------------------------------------------------------------------------- kilonova
class _KilonovaOnePredict(_PhotometricPredict):
    """``predict``/``predict_jax`` for :func:`kilonova_model`. See :class:`_PhotometricPredict`."""

    def __init__(self, name, params, band_index, *, weights, norms, lam, redshift, dl_cm,
                 t_exp_days, mag_floor, arnett_prefactor, temperature_floor):
        super().__init__(name, params, band_index)
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = float(redshift)
        self.dl_cm = float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.mag_floor = float(mag_floor)
        self.arnett_prefactor = float(arnett_prefactor)
        # None means temperature_floor is FREE and arrives in `free`; a float pins it.
        self.temperature_floor = None if temperature_floor is None else float(temperature_floor)

    def _core(self, times, bidx, *free):
        """Observer-frame days -> AB magnitude per observation. Traced, pure jnp."""
        import jax.numpy as jnp

        from . import kilonova as kn

        v = dict(zip(self.params, free))
        tf = v["temperature_floor"] if self.temperature_floor is None else self.temperature_floor
        t_src = kn.source_time_s(times, self.redshift, self.t_exp_days)
        return kn.ab_magnitude(t_src, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                               jnp.asarray(self.lam), self.redshift, self.dl_cm,
                               v["mej"], v["vej"], v["kappa"], tf,
                               mag_floor=self.mag_floor,
                               arnett_prefactor=self.arnett_prefactor)

    def _magnitude(self, times, bidx, free):
        # JIT ONCE -- the same reason tde_model gives, and the same measurement: the unjitted
        # binding re-traces on every call, 65.3 ms per predict() against 0.14 ms for the identical
        # computation under jit. Only `times`, `band_idx` and the free parameters are traced; the
        # filter set, the redshift, the distance, `t_exp_days`, `mag_floor`, `arnett_prefactor`
        # and a PINNED temperature_floor are attributes read inside the trace, so this compiles
        # once per n_obs. Both entry points come through here, which keeps them from drifting.
        import jax.numpy as jnp

        return self._jitted(None, lambda: self._core)(jnp.asarray(times), jnp.asarray(bidx), *free)


def kilonova_model(band_names, redshift, dl_cm, *, name="kilonova_one_jax", n_wave=2000,
                   ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                   temperature_floor=None, mag_floor=40.0, t_exp_days=0.0, filter_set=None,
                   arnett_prefactor=1.0, band_aliases=None):
    """Bind dataset context to the one-component JAX kilonova and return a whisper ``Model``.

    Parameters
    ----------
    band_names : sequence of str
        sncosmo bandpass names, in the order the light curve's ``bands`` column will be mapped to.
    redshift, dl_cm : float
        Fixed for the dataset. ``dl_cm`` is NOT derived from ``redshift`` here on purpose -- the
        cosmology is the caller's choice, and quietly picking one would make two codes disagree for
        a reason that has nothing to do with the model.
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
    both regimes *for that comparison* (see ``docs/MODEL_COMPARISON_GPU.md``). That is a property of
    the comparison, not a defect in this prior.
    filter_set : dict or None
        Pre-built ``make_filter_set`` output. Building it needs sncosmo and is slow, so a fit that
        re-registers repeatedly should build once and pass it here.
    arnett_prefactor : float
        1.0 (default) reproduces redback, whose diffusion kernel integrates to 1/2 rather than 1
        -- a uniform 0.7526 mag too faint relative to the standard Arnett solution (measured:
        L/L_in -> 0.500055 at t = 100 t_diff, where it must -> 1). 2.0 selects Arnett 1982 /
        Villar+2017. **The default is redback's, by project rule: parity with redback outweighs
        the correction.** See kilonova.py's ARNETT_PREFACTOR block for the full measurement.

    Notes
    -----
    ``redshift`` is a *fixed* argument rather than a fitted parameter even though redback's prior
    fits it, because ``dl_cm`` would then have to move with it through a cosmology on every
    likelihood call. Fit redshift by rebuilding the model, or pin it -- which is what any real
    kilonova fit does anyway, the host being identified.
    """
    import jax.numpy as jnp

    from ...models import Model
    from ...priors import LogUniform, Prior, Uniform

    from . import kilonova as kn

    band_names = list(band_names)
    fs = filter_set if filter_set is not None else kn.make_filter_set(band_names, n_wave=n_wave)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    if ebv_mw is not None or ebv_host is not None:
        weights = kn.extincted_weights(weights, fs["lam"], redshift=redshift,
                                        ebv_mw=ebv_mw or 0.0, ebv_host=ebv_host or 0.0,
                                        r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)

    free_tf = temperature_floor is None
    params = ["mej", "vej", "kappa"] + (["temperature_floor"] if free_tf else [])

    ctx = _KilonovaOnePredict(
        name, params, _BandIndex(band_names, "kilonova_model(band_names=[...])", band_aliases),
        weights=weights, norms=norms, lam=lam, redshift=redshift, dl_cm=dl_cm,
        t_exp_days=t_exp_days, mag_floor=mag_floor, arnett_prefactor=arnett_prefactor,
        temperature_floor=temperature_floor)
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    # redback's own defaults, verbatim from redback/priors/one_component_kilonova_model.prior,
    # so a redback-vs-whisper-GPU comparison is apples to apples. Do not substitute whisper's
    # hand-chosen kilonova priors here -- they are disjoint from redback's in kappa.
    dists = {"mej": Uniform(1e-2, 0.05), "vej": Uniform(0.1, 0.5), "kappa": Uniform(1.0, 30.0)}
    if free_tf:
        dists["temperature_floor"] = LogUniform(100.0, 6000.0)

    return Model(name=name, predict=predict, parameters=params, default_prior=Prior(dists),
                 predict_jax=predict_jax,
                 description=("One-component kilonova (JAX/GPU). Physics identical to redback's "
                               "_one_component_kilonova_model; the diffusion integral is "
                               "re-quadratured (see kilonova.py identity 6), so it diverges from "
                               "redback beyond t ~ 2.66*t_diff, where redback is under-resolved."))


def register_kilonova(band_names, redshift, dl_cm, *, name="kilonova_one_jax", **kwargs):
    """Build a kilonova ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = kilonova_model(band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True,
                   predict_jax=m.predict_jax, log_prob_jax=m.log_prob_jax)
    return m


# --------------------------------------------------------------------------- kilonova (2-comp)
class _KilonovaMultiPredict(_PhotometricPredict):
    """``predict``/``predict_jax`` for the two- and three-component kilonovae.

    One class for both: they differ only in which ``kilonova_two`` entry point the core calls, so
    ``n_components`` selects it. See :class:`_PhotometricPredict` for why this is a class at all.
    """

    def __init__(self, name, params, band_index, *, n_components, weights, norms, lam,
                 redshift, dl_cm, t_exp_days, mag_floor):
        super().__init__(name, params, band_index)
        self.n_components = int(n_components)
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = float(redshift)
        self.dl_cm = float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.mag_floor = float(mag_floor)

    def _core(self, times_s, bidx, *free):
        """SOURCE-frame seconds -> AB magnitude per observation. Traced, pure jnp."""
        import jax.numpy as jnp

        from . import kilonova_two as kn2

        fn = (kn2.two_component_magnitude if self.n_components == 2
              else kn2.three_component_magnitude)
        return fn(times_s, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                  jnp.asarray(self.lam), self.redshift, self.dl_cm, *free,
                  mag_floor=self.mag_floor)

    def _magnitude(self, times, bidx, free):
        # JIT ONCE -- the two-component factory was the only photometric one that called its
        # physics function unjitted, so every predict() re-traced `two_component_magnitude`;
        # `kilonova_two.py`'s own jitted-symbol list does not include it. Measured on 211
        # observations at n_wave=2000 on an A6000, best of five: 134.9 ms unjitted against
        # 1.893 ms here, a factor 71. Only `times`, `band_idx` and the free parameters are traced,
        # so this compiles once per n_obs, and both entry points come through here.
        import jax.numpy as jnp

        from . import kilonova as kn

        t_src = kn.source_time_s(times, self.redshift, self.t_exp_days)
        return self._jitted(None, lambda: self._core)(jnp.asarray(t_src), jnp.asarray(bidx), *free)


def kilonova_two_model(band_names, redshift, dl_cm, *, name="kilonova_two_jax", n_wave=2000,
                       ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                       mag_floor=40.0, t_exp_days=0.0, filter_set=None, prior=None,
                       band_aliases=None):
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
    comparison against redback.
    Note this is **not** the prior used by ``whisper_cbpf.models.two_component_kilonova``, whose
    ``kappa_1`` range is disjoint from it -- the two are different models, not one model twice.
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import kilonova_two as kn2

    band_names = list(band_names)
    fs = filter_set if filter_set is not None else kn.make_filter_set(band_names, n_wave=n_wave)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    if ebv_mw is not None or ebv_host is not None:
        weights = kn.extincted_weights(weights, fs["lam"], redshift=redshift,
                                       ebv_mw=ebv_mw or 0.0, ebv_host=ebv_host or 0.0,
                                       r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)
    # Villar+2017 naming. The label is cosmetic; what MAKES a component blue or red is its opacity
    # corridor in the prior (kappa_blue U(0.1, 1.0), kappa_red U(1, 30)), which is why the default
    # prior below is villar_prior and not redback's symmetric box.
    params = ["mej_blue", "vej_blue", "temperature_floor_blue", "kappa_blue",
              "mej_red", "vej_red", "temperature_floor_red", "kappa_red"]

    ctx = _KilonovaMultiPredict(
        name, params, _BandIndex(band_names, "kilonova_two_model(band_names=[...])", band_aliases),
        n_components=2, weights=weights, norms=norms, lam=lam, redshift=redshift, dl_cm=dl_cm,
        t_exp_days=t_exp_days, mag_floor=mag_floor)
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    return Model(name=name, predict=predict, parameters=params,
                 default_prior=(prior if prior is not None
                                else kn2.villar_prior(2, with_sigma=False)),
                 predict_jax=predict_jax,
                 description=("Two-component (blue + red) kilonova, JAX/GPU. Same physics as "
                              "redback's two_component_kilonova_model; components are summed in "
                              "flux, never in magnitude. The default prior is Villar+2017's, with "
                              "DISJOINT opacity corridors (kappa_blue 0.1-1, kappa_red 1-30), so "
                              "the components are not exchangeable and cannot label-switch."))


def register_kilonova_two(band_names, redshift, dl_cm, *, name="kilonova_two_jax", **kwargs):
    """Build a two-component kilonova ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = kilonova_two_model(band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True,
                   predict_jax=m.predict_jax, log_prob_jax=m.log_prob_jax)
    return m


# ------------------------------------------------------ three-component kilonova
def kilonova_three_model(band_names, redshift, dl_cm, *, name="kilonova_three_jax", n_wave=2000,
                         ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                         mag_floor=40.0, t_exp_days=0.0, filter_set=None, prior=None,
                         band_aliases=None):
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
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import kilonova_two as kn2

    band_names = list(band_names)
    fs = filter_set if filter_set is not None else kn.make_filter_set(band_names, n_wave=n_wave)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    if ebv_mw is not None or ebv_host is not None:
        weights = kn.extincted_weights(weights, fs["lam"], redshift=redshift,
                                       ebv_mw=ebv_mw or 0.0, ebv_host=ebv_host or 0.0,
                                       r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)
    params = list(kn2.PARAMETERS_3)

    ctx = _KilonovaMultiPredict(
        name, params,
        _BandIndex(band_names, "kilonova_three_model(band_names=[...])", band_aliases),
        n_components=3, weights=weights, norms=norms, lam=lam, redshift=redshift, dl_cm=dl_cm,
        t_exp_days=t_exp_days, mag_floor=mag_floor)
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    return Model(name=name, predict=predict, parameters=params,
                 default_prior=(prior if prior is not None
                                else kn2.villar_prior(3, with_sigma=False)),
                 predict_jax=predict_jax,
                 description=("Three-component (blue + purple + red) kilonova, JAX/GPU, "
                              "Villar+2017. Components are summed in flux, never in magnitude. "
                              "The default prior gives each component a DISJOINT opacity corridor "
                              "(0.1-1 / 1-5 / 5-30), bracketing Villar's fixed 0.5 / 3 / 10, so "
                              "they cannot label-switch."))


def register_kilonova_three(band_names, redshift, dl_cm, *, name="kilonova_three_jax", **kwargs):
    """Build a three-component kilonova ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = kilonova_three_model(band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True,
                   predict_jax=m.predict_jax, log_prob_jax=m.log_prob_jax)
    return m


# --------------------------------------------------------------------------- TDE
class _TDEPredict(_PhotometricPredict):
    """``predict``/``predict_jax`` for :func:`tde_model`. See :class:`_PhotometricPredict`."""

    def __init__(self, name, params, band_index, *, rise, pinned, weights, norms, lam,
                 redshift, dl_cm, xi, n_pre, xi_mw, xi_host, common):
        super().__init__(name, params, band_index)
        self.rise = str(rise)
        self.pinned = dict(pinned)
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = float(redshift)
        self.dl_cm = float(dl_cm)
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

        v = dict(zip(self.params, free))
        v.update(self.pinned)                  # pinned values are compile-time constants
        common = dict(self.common,
                      xi_mw=None if self.xi_mw is None else jnp.asarray(self.xi_mw),
                      xi_host=None if self.xi_host is None else jnp.asarray(self.xi_host))
        args = (t, bidx, jnp.asarray(self.weights), jnp.asarray(self.norms),
                jnp.asarray(self.lam), self.redshift, self.dl_cm)
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

        return self._jitted(None, lambda: self._core)(jnp.asarray(times), jnp.asarray(bidx), *free)

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


def tde_model(band_names, redshift, dl_cm, *, name=None, rise="gaussian", n_wave=2000,
              n_time=None, f_debris=1.0, xi=1.0, n_pre=100, pin=None, prior=None,
              mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None,
              r_v_mw=3.1, r_v_host=3.1, law="f99", filter_set=None, band_aliases=None,
              **engine_kwargs):
    """Bind dataset context to the JAX cooling-envelope TDE and return a whisper ``Model``.

    Photometric, so it is a factory rather than a registry entry -- for exactly the reason
    :func:`kilonova_model` is: ``predict(parameters, times, bands)`` carries no filter set,
    no redshift and no luminosity distance, and a guessed distance would silently rescale
    every fitted black-hole mass.

    Parameters
    ----------
    band_names : sequence of str
        sncosmo bandpass names, in the order the light curve's ``bands`` column maps to.
    redshift, dl_cm : float
        Fixed for the dataset. ``dl_cm`` is deliberately NOT derived from ``redshift``:
        the cosmology is the caller's choice, and picking one quietly would make two codes
        disagree for a reason unrelated to the model.
    rise : {"gaussian", "none"}
        ``"gaussian"`` is redback's ``gaussianrise_cooling_envelope``: a Gaussian rise
        normalised to meet the envelope at ``xi * tfb``, which is the model people actually
        fit, and it adds ``peak_time`` and ``sigma_t`` to the parameters. ``"none"`` is bare
        ``cooling_envelope``, valid only AFTER circularisation -- with it, ``times`` are days
        SINCE FALLBACK, whereas with a rise they are days from the start of the light curve.
    pin : dict or None
        Override which parameters are held fixed. ``{"beta": 2.0}`` pins one redback leaves
        free; ``{"beta": None}`` frees one redback pins -- but redback's file replaced those
        parameters' distributions with deltas, so freeing one also needs ``prior=``.
    prior : whisper_cbpf.priors.Prior or None
        Distributions that override redback's, per parameter. The default is redback's own
        prior, read from redback and **not modified** -- including where its file leaves only
        ``stellar_mass`` free. Widen it here, at the call site, where the choice is visible in
        the analysis rather than buried in a library default.
    n_time :
        Which redback grid to reproduce. Default is 1.12.0's 5000; pass
        ``**whisper_cbpf.models.jax.tde.REDBACK_PRESETS["1.15"]`` for 1.15.1's 500. See CHANGE 6
        in :mod:`whisper_cbpf.models.jax.tde` for what the difference costs (1.7% in T).
    dilation : bool
        ``True`` (default) applies the ``(1+z)`` flux factor, matching redback 1.15.1 and
        :mod:`whisper_cbpf.models.jax.kilonova`. ``False`` reproduces redback 1.12.0, which has
        no such factor. Worth 2.5*log10(1+z) magnitudes -- 0.053 at z = 0.05, 0.44 at z = 0.5
        -- so it is not a rounding detail. See CHANGE 7 in :mod:`whisper_cbpf.models.jax.tde`.
    filter_set : dict or None
        Pre-built ``kilonova.make_filter_set`` output; building it needs sncosmo and is
        slow, so pass it in if you re-register repeatedly.

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
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import tde

    if rise not in ("gaussian", "none"):
        raise ValueError(f"rise must be 'gaussian' or 'none', got {rise!r}")
    band_names = list(band_names)
    if name is None:
        name = "tde_gaussianrise_jax" if rise == "gaussian" else "tde_cooling_envelope_jax"
    n_time = tde.N_TIME if n_time is None else n_time

    fs = filter_set if filter_set is not None else kn.make_filter_set(band_names, n_wave=n_wave)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    xi_mw = xi_host = None
    if ebv_mw is not None:
        xi_mw = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_mw, law=law, frame="observer")
    if ebv_host is not None:
        xi_host = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_host, law=law, frame="rest")

    # redback's OWN prior, read from redback rather than transcribed -- including its FIXED
    # parameters. `cooling_envelope.prior` pins mbh_6=1, eta=0.1, alpha=0.1, beta=0.9 (bilby
    # returns them as DeltaFunction), leaving stellar_mass as the only free parameter;
    # `gaussianrise_cooling_envelope.prior` pins nothing. whisper's prior layer has no delta,
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
    prior = type(rb_prior)({k: dists[k] for k in params})
    ekw = dict(n_time=n_time, f_debris=f_debris, **engine_kwargs)
    common = dict(mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
                  xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law,
                  **ekw)

    ctx = _TDEPredict(
        name, params, _BandIndex(band_names, "tde_model(band_names=[...])", band_aliases),
        rise=rise, pinned=pinned, weights=weights, norms=norms, lam=lam, redshift=redshift,
        dl_cm=dl_cm, xi=xi, n_pre=n_pre, xi_mw=xi_mw, xi_host=xi_host, common=common)
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
                               f"Rv test (see CHANGE 3). Requires float64."))


def register_tde(band_names, redshift, dl_cm, *, name=None, **kwargs):
    """Build a TDE ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = tde_model(band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(m.name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True,
                   predict_jax=m.predict_jax, log_prob_jax=m.log_prob_jax)
    return m


# --------------------------------------------------------------------------- supernovae
def supernova_models():
    """The names :func:`supernova_model` accepts."""
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
    """

    def __init__(self, name, params, band_index, *, model, pinned, grid_kw, weights, norms, lam,
                 redshift, dl_cm, t_exp_days, xi_mw, xi_host, common):
        super().__init__(name, params, band_index)
        self.model = str(model)
        self.pinned = dict(pinned)
        self.grid_kw = dict(grid_kw)
        self.weights = np.asarray(weights)
        self.norms = np.asarray(norms)
        self.lam = np.asarray(lam)
        self.redshift = float(redshift)
        self.dl_cm = float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.xi_mw = None if xi_mw is None else np.asarray(xi_mw)
        self.xi_host = None if xi_host is None else np.asarray(xi_host)
        self.common = {k: v for k, v in common.items() if k not in ("xi_mw", "xi_host")}

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

    def _source_time(self, times):
        return np.asarray(_source_time(times, self.redshift, self.t_exp_days), dtype=float)

    def _magnitude(self, times, bidx, free):
        import jax.numpy as jnp

        return self._core_for(self._source_time(times))(jnp.asarray(bidx), *free)

    def predict_jax(self, theta, times, band_idx=None):
        """``predict`` for a traced caller: flat theta, integer bands, flux density in Jy.

        ``theta`` is ordered by ``Model.parameters``; ``band_idx`` indexes ``band_names``. Map the
        data's band strings once, outside the trace, with ``predict_jax.band_index(bands)``.

        **``times`` must be CONCRETE, not traced.** Unlike the kilonova and the TDE, this family's
        dense grid is built *from* the observation times (CHANGE 1) and the compiled core is
        memoised on ``t_src.tobytes()``, so the times are a closure constant of the compilation,
        not a traced argument. ``jax.vmap`` this over ``theta`` alone (``in_axes=(0, None, None)``)
        -- vmapping or jitting over ``times`` raises a JAX tracer-conversion error. Also requires
        float64: these engines carry cgs luminosities of 1e43-1e46 erg/s and float32 stops at
        3.4e38.
        """
        import jax
        import jax.numpy as jnp

        if band_idx is None:                    # same requirement, same message, as predict
            raise ValueError(f"model {self.name!r} is photometric and needs a `band_idx` array")
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


def supernova_model(model, band_names, redshift, dl_cm, *, name=None, n_wave=2000,
                    t_exp_days=0.0, dense_resolution=None, magnetar_convention="1.15",
                    interaction=True, dilation=True, pin=None, prior=None, mag_floor=40.0,
                    ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                    filter_set=None, times=None, band_aliases=None):
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
        sncosmo bandpass names, in the order the light curve's ``bands`` column maps to.
    redshift, dl_cm : float
        Fixed for the dataset. ``dl_cm`` is deliberately NOT derived from ``redshift``: the
        cosmology is the caller's choice, and picking one quietly would make two codes disagree
        for a reason unrelated to the model.
    times : array or None
        The SOURCE-frame epochs the dense diffusion grid will be sized from. Leave it ``None``
        and the grid is rebuilt (and the model recompiled) the first time ``predict`` sees a
        new time array, which is correct but pays one XLA compile per distinct set of epochs.
        Pass the light curve's own times to build it once at the factory.
    t_exp_days : float
        Explosion epoch on the same clock as ``times``, subtracted before the redshift.
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

    ``slsn``'s redback prior carries two bilby ``Constraint`` entries (``e_rot_constraint``,
    ``t_nebula_min``) which are rejection conditions rather than distributions. whisper's prior
    layer has no constraint concept, so they are NOT applied; they are reported in the model
    description and returned by ``whisper_cbpf.models.jax.supernova.default_prior`` if you want to
    enforce them at the call site.
    """
    import jax.numpy as jnp

    from ...models import Model

    from . import kilonova as kn
    from . import supernova as sn

    if model not in sn.MODELS:
        raise ValueError(f"unknown supernova model {model!r}. Known: {sn.model_names()}")
    band_names = list(band_names)
    name = name or f"{model}_jax"

    fs = filter_set if filter_set is not None else kn.make_filter_set(band_names, n_wave=n_wave)
    lam = jnp.asarray(fs["lam"])
    weights, norms = kn.ab_weights(fs["lam"], fs["trans"])
    xi_mw = xi_host = None
    if ebv_mw is not None:
        xi_mw = kn.extinction_shape(fs["lam"], redshift, r_v=r_v_mw, law=law, frame="observer")
    if ebv_host is not None:
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
    model_prior = type(rb_prior)({k: dists[k] for k in params})

    common = dict(magnetar_convention=magnetar_convention, interaction=interaction,
                  mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host,
                  xi_mw=xi_mw, xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)

    grid_kw = {} if dense_resolution is None else dict(dense_resolution=dense_resolution)
    ctx = _SupernovaPredict(
        name, params,
        _BandIndex(band_names, f"supernova_model({model!r}, band_names=[...])", band_aliases),
        model=model, pinned=pinned, grid_kw=grid_kw, weights=weights, norms=norms, lam=lam,
        redshift=redshift, dl_cm=dl_cm, t_exp_days=t_exp_days, xi_mw=xi_mw, xi_host=xi_host,
        common=common)
    if times is not None:                      # build (and compile on first call) up front
        ctx._core_for(np.asarray(_source_time(times, redshift, t_exp_days), dtype=float))
    predict, predict_jax = ctx.predict, _PredictJax(ctx)

    notes = []
    if pinned:
        notes.append("redback pins " + ", ".join(f"{k}={v:g}" for k, v in sorted(pinned.items())))
    if constraints:
        notes.append("redback's prior also declares Constraint(s) "
                     + ", ".join(f"{k} in [{lo:g}, {hi:g}]"
                                 for k, (lo, hi) in sorted(constraints.items()))
                     + ", NOT applied here")
    if model in ("sn_fallback", "sn_nickel_fallback"):
        notes.append(f"interaction={interaction} (redback applies none; worth up to 15 mag)")
    if any(p in sn.PARAMETERS[model] for p in ("p0", "bp")):
        notes.append(f"magnetar_convention={magnetar_convention}")
    tail = ("; " + "; ".join(notes)) if notes else ""
    return Model(name=name, predict=predict, parameters=params, default_prior=model_prior,
                 predict_jax=predict_jax,
                 description=(f"Supernova: redback's {model} (JAX/GPU){tail}. Bolometric "
                              f"physics matches redback to ~1e-16; the band integral uses the "
                              f"real bandpass rather than redback's sncosmo spline. See "
                              f"whisper_cbpf.models.jax.supernova for the numbered deviations."))


def _source_time(times, redshift, t_exp_days):
    """Observer-frame days -> SOURCE-frame days, redback's ``calc_kcorrected_properties``."""
    return (np.asarray(times, dtype=float) - t_exp_days) / (1.0 + redshift)


def register_supernova(model, band_names, redshift, dl_cm, *, name=None, **kwargs):
    """Build a supernova ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = supernova_model(model, band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(m.name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True,
                   predict_jax=m.predict_jax, log_prob_jax=m.log_prob_jax)
    return m
