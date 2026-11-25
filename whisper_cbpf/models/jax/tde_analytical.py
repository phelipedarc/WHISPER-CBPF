"""redback ``tde_analytical``: a ``t^-5/3`` fall-back luminosity through Arnett diffusion, in JAX.

Adapted from redback (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback, GPL-3.0;
modified for whisper_cbpf. redback-jax (Nikhil Sarin et al.),
https://github.com/nikhil-sarin/redback-jax, commit 3431e2e, GPL-3.0, also implements this model; no
code is taken from it. This module assembles redback's model from the whisper_cbpf ports of its
parts: the engine :func:`whisper_cbpf.models.jax.tde.analytic_fallback`, the diffusion kernel
:func:`whisper_cbpf.models.jax.tde.diffusion` on redback's grid, the temperature-floor photosphere
and the cut-off black body of :mod:`whisper_cbpf.models.jax.supernova`.

The model: the luminosity at source-frame time ``t`` (days since the fall-back began) is
``L = l0 / (t 86400)^(5/3)`` for ``t > t_0_turn`` and flat at its ``t_0_turn`` value before it, with
``l0`` the luminosity at 1 s in erg/s. It is delayed and broadened by the Arnett (1982) diffusion
of the ejecta, ``mej`` (Msun) at ``vej`` (km/s) with optical opacity ``kappa`` and gamma-ray
opacity ``kappa_gamma`` (cm^2/g), so the observed light curve rises while the photons diffuse out.
The photosphere is the temperature-floor one (it expands at ``vej`` until it reaches
``temperature_floor``, Villar et al. 2017) and the spectrum is the black body with the
short-wavelength cut-off at 3000 Angstrom of Nicholl et al. (2017).

**Time.** Source-frame days since the fall-back began, which is the model's time zero. In
:func:`tde_analytical_model` the epochs are observer-frame, ``t_exp`` (or ``t_exp_days``) is that
zero, and an epoch at or before it is at ``mag_floor``.

**Agreement with redback.** The bolometric luminosity is redback's to float64 round-off: the same
engine, the same dense grid ``geomspace(1e-5, t_last + 100, 1000)`` and the same adaptive
quadrature nodes, evaluated at every observation directly (no interpolation of the light curve),
so the epochs, ``t_exp`` and the redshift can be traced. The photometry differs from redback's only
by the band integral (whisper integrates the real bandpass).

**Constraints.** redback 1.20 attaches two to this model when ``get_priors(..., constraint=True)``:
the ejecta kinetic energy must lie below the fall-back energy ``5/2 l0 / (t_0_turn 86400)^(2/3)``
plus 1e51 erg, and the nebular time ``sqrt(3 kappa mej / (4 pi vej^2))`` must lie between 100 and
500 days. They are transcribed in :mod:`whisper_cbpf.models.constraints` and are a hard wall as for the
other whisper models (``constraint=``; there is no corrected variant, so ``"corrected"`` and
``"redback"`` are the same) and reject most of the box
prior; pass ``constraint=None`` for the bare physics.

**Prior: one deliberate difference from redback.** ``kappa`` is LogUniform(0.05, 100) cm^2/g
here, where redback's file has Uniform(0.05, 100); every other distribution is redback's
(:func:`default_prior` says why). ``tde_analytical_model(..., prior="redback")`` or
:func:`redback_default_prior` gives redback's prior unchanged.

**Precision.** float64 is required: the luminosity is 1e43 to 1e60 erg/s in cgs, past float32's
3.4e38 ceiling. The diffusion kernel raises rather than returning a light curve at ``mag_floor``::

    import jax; jax.config.update("jax_enable_x64", True)   # before the first array exists

References
----------
Arnett, W. D. 1982, ApJ 253, 785 (https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A)

Villar, V. A. et al. 2017, ApJ 851, L21 (https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V)

Nicholl, M., Guillochon, J., Berger, E. 2017, ApJ 850, 55
(https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N)

Sarin, N. et al. 2024, MNRAS 531, 1203 (redback; https://arxiv.org/abs/2308.12806)
"""
from __future__ import annotations

import jax.numpy as jnp

from ._adopted_tde import GRID_START, assemble, geometric_dense_times
from .supernova import _diffuse, sn_ab_magnitude
from .tde import analytic_fallback

__all__ = [
    "tde_analytical_bolometric", "tde_analytical_ab_magnitude", "PARAMETERS", "default_prior",
    "fallback_prior", "redback_default_prior", "LOG_UNIFORM_OVER_REDBACK", "tde_analytical_model",
    "register_tde_analytical",
]

#: redback's order of ``tde_analytical`` (``l0``, ``t_0_turn``), then its keyword arguments.
PARAMETERS = ("l0", "t_0_turn", "mej", "vej", "kappa", "kappa_gamma", "temperature_floor")


def tde_analytical_bolometric(grid, l0, t_0_turn, kappa, kappa_gamma, mej, vej):
    """redback's ``tde_analytical_bolometric`` [erg/s] on ``grid``.

    ``grid`` is :func:`whisper_cbpf.models.jax.supernova.build_sn_grid`'s (source-frame epochs
    ``time`` since the fall-back began, and their dense engine grid). ``l0`` in erg/s at 1 s,
    ``t_0_turn`` in days, ``mej`` in solar masses, ``vej`` in km/s.
    """
    return _diffuse(grid, analytic_fallback(grid["dense_times"], l0, t_0_turn),
                    kappa, kappa_gamma, mej, vej)


def tde_analytical_ab_magnitude(tau_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                                l0, t_0_turn, mej, vej, kappa, kappa_gamma, temperature_floor, *,
                                cutoff_wavelength=3000.0, dense_resolution=1000, mag_floor=40.0,
                                dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None,
                                xi_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99"):
    """AB magnitude of redback's ``tde_analytical`` per observation. Traced; **float64 only**.

    ``tau_days`` are SOURCE-frame days since the fall-back began (``(t - t_exp) / (1 + z)``); an
    epoch at or before 0 is at ``mag_floor``. ``weights, norms`` come from
    :func:`whisper_cbpf.models.jax.kilonova.ab_weights` and ``lam_obs_ang`` is its grid, as for
    :func:`whisper_cbpf.models.jax.supernova.sn_ab_magnitude`, whose ``mag_floor``, ``dilation`` and
    extinction arguments these are. ``cutoff_wavelength`` [Angstrom] is the black body's cut-off.
    """
    tau = jnp.asarray(tau_days)
    alive = tau > 0.0
    t_eval = jnp.where(alive, jnp.maximum(tau, GRID_START), 1.0)     # finite when dead
    grid = dict(time=t_eval, uniq_times=t_eval, gather_idx=jnp.arange(t_eval.size), tb=GRID_START,
                dense_times=geometric_dense_times(jnp.max(jnp.where(alive, tau, 0.0)),
                                                  dense_resolution),
                dense_is_geometric=1.0)          # a flag: the diffusion's lookup by arithmetic
    lbol = tde_analytical_bolometric(grid, l0, t_0_turn, kappa, kappa_gamma, mej, vej)
    mag = sn_ab_magnitude(grid, lbol, vej, temperature_floor, band_idx, weights, norms,
                          lam_obs_ang, redshift, dl_cm, sed_kind="cutoff",
                          cutoff_wavelength=cutoff_wavelength, mag_floor=mag_floor,
                          dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host, xi_mw=xi_mw,
                          xi_host=xi_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)
    return jnp.where(alive, mag, jnp.asarray(mag_floor, mag.dtype))


# --- priors ---------------------------------------------------------------------------------

#: Transcribed from redback 1.20's ``tde_analytical.prior``, for when redback is not importable.
_FALLBACK_PRIOR = (
    {"mej": ("LogUniform", 0.1, 100.0), "vej": ("LogUniform", 1e3, 1e5),
     "kappa": ("Uniform", 0.05, 100.0), "kappa_gamma": ("LogUniform", 1e-4, 1e4),
     "temperature_floor": ("LogUniform", 1e3, 1e4), "l0": ("LogUniform", 1e51, 1e60),
     "t_0_turn": ("LogUniform", 1e-4, 5e2)},
    {},
)

#: Where whisper's default prior (:func:`default_prior`) deliberately differs from redback's: each
#: parameter named here is LogUniform over the bounds redback's file gives it. redback has
#: ``kappa`` Uniform(0.05, 100). The other scale parameters (``mej``, ``vej``, ``kappa_gamma``,
#: ``l0``, ``t_0_turn``, ``temperature_floor``) are LogUniform in redback's file already.
LOG_UNIFORM_OVER_REDBACK = ("kappa",)


def fallback_prior():
    """:func:`redback_default_prior` without redback: the transcription of its 1.20 prior file.

    Returns
    -------
    tuple
        ``(Prior, pinned)``: redback's distributions (``kappa`` Uniform(0.05, 100)) and an empty
        dict.

    Examples
    --------
    >>> from whisper_cbpf.models.jax.tde_analytical import fallback_prior
    >>> prior, pinned = fallback_prior()
    >>> prior.distributions["kappa"], pinned
    (Uniform(0.05, 100.0), {})
    """
    from ...priors import LogUniform, Prior, Uniform

    dists, pinned = _FALLBACK_PRIOR
    cls = {"Uniform": Uniform, "LogUniform": LogUniform}
    return Prior({k: cls[c](lo, hi) for k, (c, lo, hi) in dists.items()}), dict(pinned)


def redback_default_prior():
    """``(Prior, pinned)`` of redback's ``tde_analytical``, unmodified: its prior file, read from
    redback where possible (``redshift`` dropped: it is the factory's argument), else
    :func:`fallback_prior`.

    ``mej`` LogUniform(0.1, 100) Msun, ``vej`` LogUniform(1e3, 1e5) km/s, ``kappa``
    Uniform(0.05, 100) cm^2/g, ``kappa_gamma`` LogUniform(1e-4, 1e4) cm^2/g,
    ``temperature_floor`` LogUniform(1e3, 1e4) K, ``l0`` LogUniform(1e51, 1e60) erg/s,
    ``t_0_turn`` LogUniform(1e-4, 500) d; nothing is pinned. ``tde_analytical_model(...,
    prior="redback")`` fits under it.

    Returns
    -------
    tuple
        ``(Prior, pinned)``.

    See Also
    --------
    default_prior : whisper's default, which differs in ``kappa``.

    Examples
    --------
    >>> from whisper_cbpf.models.jax.tde_analytical import redback_default_prior
    >>> prior, pinned = redback_default_prior()
    >>> type(prior.distributions["kappa"]).__name__, prior.distributions["kappa"].bounds
    ('Uniform', (0.05, 100.0))
    """
    from .tde import redback_prior

    try:
        return redback_prior("tde_analytical")
    except ImportError:
        return fallback_prior()


def default_prior():
    """``(Prior, pinned)`` whisper fits ``tde_analytical`` under: :func:`redback_default_prior`
    with each parameter of :data:`LOG_UNIFORM_OVER_REDBACK` (``kappa``) LogUniform over redback's
    bounds.

    **A deliberate difference from redback**, whose ``kappa`` is Uniform(0.05, 100). That
    Uniform spans 3.3 decades and leaves 0.45 % of the prior below 0.5 cm^2/g, where an ionised
    hydrogen-helium opacity (0.2-0.4 cm^2/g) lies. On injected light curves with ``kappa`` = 0.2,
    nested sampling and emcee_jax under redback's prior settled in a second solution (``kappa``
    60-100, ``mej`` 20-100) 42 to 253 nats below the true one, with ``t_exp`` 5 to 12 days late.
    With ``kappa`` LogUniform both recovered the truth, and emcee_jax converged in 46-217 s
    instead of not at all in 600 s. Pass ``prior="redback"`` to :func:`tde_analytical_model`
    (or use :func:`redback_default_prior`) for redback's prior.

    Returns
    -------
    tuple
        ``(Prior, pinned)``: ``kappa`` LogUniform(0.05, 100) cm^2/g, every other distribution
        as redback's; nothing is pinned.

    Examples
    --------
    >>> from whisper_cbpf.models.jax.tde_analytical import default_prior
    >>> prior, pinned = default_prior()
    >>> prior.distributions["kappa"]
    LogUniform(0.05, 100.0)
    """
    from ...priors import LogUniform, Prior

    rb, pinned = redback_default_prior()
    dists = dict(rb.distributions)
    for k in LOG_UNIFORM_OVER_REDBACK:
        if k in dists:
            lo, hi = dists[k].bounds
            dists[k] = LogUniform(lo, hi)
    return Prior(dists), dict(pinned)


def _resolve_prior(prior, caller):
    """``prior=`` of :func:`tde_analytical_model` as a ``Prior`` or ``None``: ``"redback"`` gives
    :func:`redback_default_prior`'s distributions; a ``Prior`` or ``None`` is returned as given."""
    if isinstance(prior, str):
        if prior.strip().lower() != "redback":
            raise ValueError(f"{caller}: prior={prior!r} is not understood. Pass prior=None for "
                             f"whisper's default (kappa LogUniform), prior='redback' for "
                             f"redback's (kappa Uniform), or a Prior.")
        return redback_default_prior()[0]
    return prior


# --- the whisper model ----------------------------------------------------------------------

def tde_analytical_model(band_names, redshift=None, dl_cm=None, *, name="tde_analytical_jax",
                         n_wave=None, dense_resolution=1000, cutoff_wavelength=3000.0, pin=None,
                         prior=None, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None,
                         r_v_mw=3.1, r_v_host=3.1, law="f99", filter_set=None, band_aliases=None,
                         default_system=None, t_exp_days=0.0, constraint="corrected", free=None,
                         redshift_prior=None):
    """Bind dataset context to redback's ``tde_analytical`` and return a whisper ``Model``.

    Photometric, so a factory rather than a registry entry: ``predict(parameters, times, bands)``
    carries no filter set, redshift or distance. See :func:`whisper_cbpf.models.jax.tde_model`,
    whose arguments these follow.

    Parameters
    ----------
    band_names : sequence of str
        Filter names or labels, in the order the light curve's ``bands`` column maps to (bare
        letters are LSST unless ``default_system=``).
    redshift, dl_cm : float
        Fixed for the dataset; leave both ``None`` when the redshift is fitted
        (``free=["redshift"]``).
    name : str
        The model's name (default ``"tde_analytical_jax"``).
    dense_resolution : int
        Points of the engine grid (redback's 1000).
    cutoff_wavelength : float
        The black body's cut-off, Angstrom (redback's default, 3000).
    pin : dict or None
        Override which parameters are held fixed, e.g. ``{"kappa": 0.2}``.
    prior : whisper_cbpf.priors.Prior, "redback" or None
        Distributions overriding the default, per parameter. The default (``None``) is
        :func:`default_prior`: redback's prior with ``kappa`` LogUniform(0.05, 100) instead of
        redback's Uniform(0.05, 100), a deliberate difference. ``"redback"`` fits under redback's
        prior unchanged (:func:`redback_default_prior`); with ``free=["t_exp"]``, pass
        ``Prior({**redback_default_prior()[0].distributions, "t_exp": ...})`` instead.
    constraint : {"corrected", "redback", None}
        redback's two constraints for this model as a hard wall (see the module docstring):
        ``predict`` gives zero flux, so the CPU samplers reject the draw, and the JAX samplers
        apply ``predict_jax.constraint_ok``. ``"corrected"`` and ``"redback"`` are the same here;
        ``None`` applies none. The zeros are silent in a direct ``predict``: test a draw with
        ``predict_jax.constraint_ok(theta)`` before plotting it.
    mag_floor, dilation, ebv_mw, ebv_host, r_v_mw, r_v_host, law, filter_set, n_wave, band_aliases, default_system :
        As :func:`whisper_cbpf.models.jax.tde_model` (Gauss-16 band integral by default, the
        ``(1 + z)`` flux factor of redback's TDE path, optional extinction).
    t_exp_days : float
        Time the fall-back began on the light curve's own clock (the model's time zero),
        subtracted on the host in float64. Default 0: ``times`` are days since it.
    free : list of str or None
        ``"t_exp"`` and/or ``"redshift"`` fitted as parameters, appended in that order; see
        :func:`whisper_cbpf.models.jax.tde_model`. ``t_exp`` needs ``prior=Prior({"t_exp": ...})``.
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none.

    Returns
    -------
    Model
        ``parameters`` ``["l0", "t_0_turn", "mej", "vej", "kappa", "kappa_gamma",
        "temperature_floor"]`` (plus the freed context values).

    References
    ----------
    Arnett 1982, ApJ 253, 785; Villar et al. 2017, ApJ 851, L21; Nicholl, Guillochon & Berger
    2017, ApJ 850, 55; Sarin et al. 2024, MNRAS 531, 1203 (redback). Full list in the module
    docstring.

    Notes
    -----
    **float64 is required** (``jax.config.update("jax_enable_x64", True)`` before the first array
    exists); the guard raises at the first ``predict``, not here.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax.tde_analytical import tde_analytical_model
    >>> m = tde_analytical_model(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.parameters[:2]
    ['l0', 't_0_turn']
    >>> m.default_prior.distributions["kappa"]
    LogUniform(0.05, 100.0)
    >>> rb = tde_analytical_model(["lsstg"], redshift=0.05, dl_cm=7.1e26, prior="redback")
    >>> rb.default_prior.distributions["kappa"]
    Uniform(0.05, 100.0)
    """
    prior = _resolve_prior(prior, "tde_analytical_model")
    own_kappa = ((prior is not None and "kappa" in prior.distributions)
                 or (pin or {}).get("kappa") is not None)     # the caller's prior, or pinned
    kappa_note = "" if own_kappa else "; kappa prior LogUniform, not redback's Uniform"
    return assemble(
        "tde_analytical", "tde_analytical", band_names, redshift, dl_cm, name=name, n_wave=n_wave,
        pin=pin, prior=prior, mag_floor=mag_floor, dilation=dilation, ebv_mw=ebv_mw,
        ebv_host=ebv_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law, filter_set=filter_set,
        band_aliases=band_aliases, default_system=default_system, t_exp_days=t_exp_days, free=free,
        redshift_prior=redshift_prior, constraint=constraint,
        engine_kw=dict(dense_resolution=dense_resolution, cutoff_wavelength=cutoff_wavelength),
        summary="t^-5/3 fall-back through Arnett diffusion, temperature-floor photosphere, "
                f"cut-off black body{kappa_note}",
        caller="tde_analytical_model")


def register_tde_analytical(band_names, redshift=None, dl_cm=None, *, name="tde_analytical_jax",
                            **kwargs):
    """Build a ``tde_analytical`` ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`tde_analytical_model`.
    name : str, default "tde_analytical_jax"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.jax.tde_analytical import register_tde_analytical
    >>> m = register_tde_analytical(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.name in wp.list_models()
    True
    """
    from ._factories import _register

    return _register(tde_analytical_model(band_names, redshift, dl_cm, name=name, **kwargs))
