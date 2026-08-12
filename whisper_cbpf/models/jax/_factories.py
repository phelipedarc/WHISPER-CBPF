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
           "tde_model", "register_tde",
           "supernova_model", "register_supernova", "supernova_models"]


# --------------------------------------------------------------------------- flare
def flare_model():
    """The JAX flare as a ``whisper_cbpf.models.Model``. Self-contained, auto-registered."""
    from . import flare
    return flare.get_model()


# --------------------------------------------------------------------------- kilonova
def kilonova_model(band_names, redshift, dl_cm, *, name="kilonova_one_jax", n_wave=2000,
                   ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                   temperature_floor=None, mag_floor=40.0, t_exp_days=0.0, filter_set=None,
                   arnett_prefactor=1.0):
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
    index_of = {b: i for i, b in enumerate(band_names)}

    free_tf = temperature_floor is None
    params = ["mej", "vej", "kappa"] + (["temperature_floor"] if free_tf else [])

    def predict(parameters, times, bands=None):
        """whisper contract: (parameters, times, bands) -> flux density in Jy."""
        if bands is None:
            raise ValueError(f"model {name!r} is photometric and needs a `bands` array")
        try:
            idx = np.array([index_of[str(b)] for b in np.asarray(bands)], dtype=int)
        except KeyError as exc:
            raise KeyError(
                f"band {exc.args[0]!r} was not bound to this model. Bound: {band_names}. "
                "Rebuild with kilonova_model(band_names=[...]) covering every band in the data."
            ) from None
        tf = float(parameters["temperature_floor"]) if free_tf else float(temperature_floor)
        t_src = kn.source_time_s(np.asarray(times, dtype=float), redshift, t_exp_days)
        mag = kn.ab_magnitude(jnp.asarray(t_src), jnp.asarray(idx), weights, norms, lam,
                              redshift, dl_cm,
                              float(parameters["mej"]), float(parameters["vej"]),
                              float(parameters["kappa"]), tf, mag_floor=mag_floor,
                              arnett_prefactor=arnett_prefactor)
        # whisper models return FLUX; convert once, at the end. Never combine magnitudes.
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))

    # redback's own defaults, verbatim from redback/priors/one_component_kilonova_model.prior,
    # so a redback-vs-whisper-GPU comparison is apples to apples. Do not substitute whisper's
    # hand-chosen kilonova priors here -- they are disjoint from redback's in kappa.
    dists = {"mej": Uniform(1e-2, 0.05), "vej": Uniform(0.1, 0.5), "kappa": Uniform(1.0, 30.0)}
    if free_tf:
        dists["temperature_floor"] = LogUniform(100.0, 6000.0)

    return Model(name=name, predict=predict, parameters=params, default_prior=Prior(dists),
                 description=("One-component kilonova (JAX/GPU). Physics identical to redback's "
                               "_one_component_kilonova_model; the diffusion integral is "
                               "re-quadratured (see kilonova.py identity 6), so it diverges from "
                               "redback beyond t ~ 2.66*t_diff, where redback is under-resolved."))


def register_kilonova(band_names, redshift, dl_cm, *, name="kilonova_one_jax", **kwargs):
    """Build a kilonova ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = kilonova_model(band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True)
    return m


# --------------------------------------------------------------------------- kilonova (2-comp)
def kilonova_two_model(band_names, redshift, dl_cm, *, name="kilonova_two_jax", n_wave=2000,
                       ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                       mag_floor=40.0, t_exp_days=0.0, filter_set=None, prior=None):
    """Bind dataset context to the TWO-component JAX kilonova and return a whisper ``Model``.

    The two-component sibling of :func:`kilonova_model`, built the same way and for the same reason:
    ``predict(parameters, times, bands)`` carries no filter set, no redshift and no luminosity
    distance, so this is a factory rather than a registry entry.

    Parameters are redback's, in redback's order: ``mej_1, vej_1, temperature_floor_1, kappa_1``
    then the same four for component 2. Component 1 is conventionally the blue / lanthanide-poor
    ejecta and component 2 the red, but **nothing enforces that** -- redback's default prior gives
    both components the same ``kappa`` range, so the components are exchangeable and a fit can
    label-switch. See ``OPEN_ITEMS.md`` on disjoint corridors versus an ordering constraint.

    ``prior`` defaults to :func:`kilonova_two.default_prior`, which is redback's own, verbatim.
    Note this is **not** the prior used by ``whisper_cbpf.models.two_component_kilonova``, whose
    ``kappa_1`` range is disjoint from it -- the two are different models, not one model twice.
    """
    import jax.numpy as jnp

    from ...io.photometry import AB_ZEROPOINT_JY
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
    index_of = {b: i for i, b in enumerate(band_names)}

    params = ["mej_1", "vej_1", "temperature_floor_1", "kappa_1",
              "mej_2", "vej_2", "temperature_floor_2", "kappa_2"]

    def predict(parameters, times, bands=None):
        """whisper contract: (parameters, times, bands) -> flux density in Jy."""
        if bands is None:
            raise ValueError(f"model {name!r} is photometric and needs a `bands` array")
        try:
            idx = np.array([index_of[str(b)] for b in np.asarray(bands)], dtype=int)
        except KeyError as exc:
            raise KeyError(
                f"band {exc.args[0]!r} was not bound to this model. Bound: {band_names}. "
                "Rebuild with kilonova_two_model(band_names=[...]) covering every band in the data."
            ) from None
        t_src = kn.source_time_s(np.asarray(times, dtype=float), redshift, t_exp_days)
        mag = kn2.two_component_magnitude(
            jnp.asarray(t_src), jnp.asarray(idx), weights, norms, lam, redshift, dl_cm,
            *(float(parameters[p]) for p in params), mag_floor=mag_floor)
        # whisper models return FLUX; convert once, at the end. Never combine magnitudes.
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))

    return Model(name=name, predict=predict, parameters=params,
                 default_prior=prior if prior is not None else kn2.default_prior(),
                 description=("Two-component (blue + red) kilonova, JAX/GPU. Same physics as "
                              "redback's two_component_kilonova_model; components are summed in "
                              "flux, never in magnitude. Both components share redback's kappa "
                              "range, so they are exchangeable under the default prior."))


def register_kilonova_two(band_names, redshift, dl_cm, *, name="kilonova_two_jax", **kwargs):
    """Build a two-component kilonova ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = kilonova_two_model(band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True)
    return m


# --------------------------------------------------------------------------- TDE
def tde_model(band_names, redshift, dl_cm, *, name=None, rise="gaussian", n_wave=2000,
              n_time=None, f_debris=1.0, xi=1.0, n_pre=100, pin=None, prior=None,
              mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None,
              r_v_mw=3.1, r_v_host=3.1, law="f99", filter_set=None, **engine_kwargs):
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
    import jax
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
    index_of = {b: i for i, b in enumerate(band_names)}

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

    # JIT ONCE, HERE. `cooling_envelope` builds its scan step with `partial(...)` over the
    # parameter values, so calling it OUTSIDE jit bakes those concrete floats into the
    # jaxpr as constants -- a cache miss on every new parameter set, i.e. a fresh 0.6 s XLA
    # compile of a 5000-step scan per likelihood evaluation. Measured: 831 ms per predict()
    # against 9.1 ms for the identical computation under jit, a factor 92. Everything that
    # is not traced (the filter set, the extinction shape, `law`, `n_time`) is captured in
    # the closure, so this compiles once per (n_obs, n_band) shape.
    def _core(t, bidx, *free):
        v = dict(zip(params, free))
        v.update(pinned)                       # pinned values are closure constants
        if rise == "gaussian":
            return tde.gaussianrise_cooling_envelope_ab_magnitude(
                t, bidx, weights, norms, lam, redshift, dl_cm,
                v["peak_time"], v["sigma_t"], v["mbh_6"], v["stellar_mass"],
                v["eta"], v["alpha"], v["beta"], xi=xi, n_pre=n_pre, **common)
        return tde.cooling_envelope_ab_magnitude(
            t, bidx, weights, norms, lam, redshift, dl_cm,
            v["mbh_6"], v["stellar_mass"], v["eta"], v["alpha"], v["beta"], **common)

    _core_jit = jax.jit(_core)

    def predict(parameters, times, bands=None):
        """whisper contract: (parameters, times, bands) -> flux density in Jy."""
        if bands is None:
            raise ValueError(f"model {name!r} is photometric and needs a `bands` array")
        try:
            idx = np.array([index_of[str(b)] for b in np.asarray(bands)], dtype=int)
        except KeyError as exc:
            raise KeyError(
                f"band {exc.args[0]!r} was not bound to this model. Bound: {band_names}. "
                "Rebuild with tde_model(band_names=[...]) covering every band in the data."
            ) from None
        t = jnp.asarray(np.asarray(times, dtype=float))
        bidx = jnp.asarray(idx)
        mag = _core_jit(t, bidx, *[float(parameters[k]) for k in params])
        # whisper models return FLUX; convert once, at the end. Never combine magnitudes.
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))

    what = ("Gaussian rise + Sarin & Metzger cooling envelope"
            if rise == "gaussian" else "Sarin & Metzger cooling envelope (post-circularisation only)")
    pin_note = ("; redback pins " + ", ".join(f"{k}={v:g}" for k, v in sorted(pinned.items()))
                if pinned else "")
    return Model(name=name, predict=predict, parameters=params, default_prior=prior,
                 description=(f"TDE: {what} (JAX/GPU){pin_note}. Physics identical to redback's "
                               f"_cooling_envelope at n_time={n_time}; the light curve ends at "
                               f"the envelope's own death rather than at redback's diverged "
                               f"Rv test (see CHANGE 3). Requires float64."))


def register_tde(band_names, redshift, dl_cm, *, name=None, **kwargs):
    """Build a TDE ``Model`` for this dataset and register it under ``name``."""
    from ...models import register_model

    m = tde_model(band_names, redshift, dl_cm, name=name, **kwargs)
    register_model(m.name, m.predict, m.parameters, prior=m.default_prior,
                   description=m.description, overwrite=True)
    return m


# --------------------------------------------------------------------------- supernovae
def supernova_models():
    """The names :func:`supernova_model` accepts."""
    from . import supernova as sn

    return sn.model_names()


def supernova_model(model, band_names, redshift, dl_cm, *, name=None, n_wave=2000,
                    t_exp_days=0.0, dense_resolution=None, magnetar_convention="1.15",
                    interaction=True, dilation=True, pin=None, prior=None, mag_floor=40.0,
                    ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
                    filter_set=None, times=None):
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
    ``float64 is required``, as for the TDE, but for a different reason and at a different moment.
    The TDE's is an accumulator: its envelope ODE adds increments below float32's epsilon. This
    family's is range -- the engines carry bolometric luminosities in cgs, 1e43 to 1e46 erg/s,
    against float32's 3.4e38 ceiling, so EVERY engine returns ``inf`` and every band comes back at
    ``mag_floor`` (CHANGE 8). ``build_sn_grid`` therefore raises at FACTORY time, so
    ``register_supernova`` itself fails; the TDE's guard is inside the engine and does not fire
    until the first ``predict``. Enable it before creating any array::

        import jax; jax.config.update("jax_enable_x64", True)

    ``slsn``'s redback prior carries two bilby ``Constraint`` entries (``e_rot_constraint``,
    ``t_nebula_min``) which are rejection conditions rather than distributions. whisper's prior
    layer has no constraint concept, so they are NOT applied; they are reported in the model
    description and returned by ``whisper_cbpf.models.jax.supernova.default_prior`` if you want to
    enforce them at the call site.
    """
    import jax
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
    index_of = {b: i for i, b in enumerate(band_names)}

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

    # JIT ONCE, per (grid, n_obs) shape. The grid is DATA -- it is built from the observation
    # times at setup (CHANGE 1) -- so it is captured in the closure rather than traced, and a
    # new time array means a new grid and one recompile. `_grids` memoises on the times so a
    # sampler calling predict() in a loop with the same light curve compiles exactly once.
    _cores = {}
    grid_kw = {} if dense_resolution is None else dict(dense_resolution=dense_resolution)

    def _core_for(t_src):
        key = t_src.tobytes()
        if key not in _cores:
            g = sn.build_sn_grid(t_src, **grid_kw)

            def _core(bidx, *free):
                v = dict(zip(params, free))
                v.update(pinned)               # pinned values are closure constants
                return sn.ab_magnitude_of(model, g, v, bidx, weights, norms, lam,
                                          redshift, dl_cm, **common)

            _cores[key] = jax.jit(_core)
        return _cores[key]

    if times is not None:                      # build (and compile on first call) up front
        _core_for(np.asarray(_source_time(times, redshift, t_exp_days), dtype=float))

    def predict(parameters, times, bands=None):
        """whisper contract: (parameters, times, bands) -> flux density in Jy."""
        if bands is None:
            raise ValueError(f"model {name!r} is photometric and needs a `bands` array")
        try:
            idx = np.array([index_of[str(b)] for b in np.asarray(bands)], dtype=int)
        except KeyError as exc:
            raise KeyError(
                f"band {exc.args[0]!r} was not bound to this model. Bound: {band_names}. "
                f"Rebuild with supernova_model({model!r}, band_names=[...]) covering every "
                f"band in the data.") from None
        t_src = np.asarray(_source_time(times, redshift, t_exp_days), dtype=float)
        mag = _core_for(t_src)(jnp.asarray(idx),
                               *[float(parameters[k]) for k in params])
        # whisper models return FLUX; convert once, at the end. Never combine magnitudes.
        return AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))

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
                   description=m.description, overwrite=True)
    return m
