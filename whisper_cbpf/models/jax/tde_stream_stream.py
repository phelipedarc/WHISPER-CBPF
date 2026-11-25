"""redback ``stream_stream_tde``: a TDE lit by stream-stream collisions, in JAX.

Adapted from redback (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback, GPL-3.0;
modified for whisper_cbpf. redback-jax (Nikhil Sarin et al.),
https://github.com/nikhil-sarin/redback-jax, commit 3431e2e, GPL-3.0, has no counterpart of this
model, so nothing is taken from it. This module assembles redback's photometric model from the
engine :func:`whisper_cbpf.models.jax.tde.stream_stream_collision` (redback's
``_stream_stream_collision``), the Gaussian rise, and the black-body photometry of
:mod:`whisper_cbpf.models.jax.tde`.

The model (Piran et al. 2015; Ryu, Krolik & Piran 2020; Krolik, Piran & Ryu 2024): light comes from
shocks where the returning debris streams intersect near the apocentre ``a0`` of the most bound
debris. At the peak, ``t_peak = 1.5 t_dyn``, the luminosity is ``L_max = Mdot_max Xi eps / c1`` with
``Mdot_max = M_star / (3 t_dyn)``, and afterwards ``L = L_max (t / t_peak)^(-5/3) F`` with the
cooling factor ``F = 2 / (1 + t_cool / t_dyn)`` when ``inc_tcool = 1`` (1 otherwise). The emitting
area is fixed, so ``T = (L / (del_omega sigma c1^2 a0^2 (1 + 2 h_r) / 4))^(1/4)`` falls as
``t^(-5/12)`` and the black-body radius is constant. Before the peak the model is a Gaussian of
width ``sigma_t`` and centre ``peak_time`` (days), normalised to meet the engine at ``t_peak``,
at the constant temperature the engine has at ``t_peak``.

**Time.** Source-frame days since the start of the Gaussian rise, which is the model's time zero.
In :func:`tde_stream_stream_model` the epochs are observer-frame, ``t_exp`` (or ``t_exp_days``) is
that zero, and an epoch at or before it is at ``mag_floor``. The engine grid ends 1500 days after
it, and past that end the model is masked (see D2).

**What redback does, reproduced by default.** The numbered behaviours below are properties of
redback 1.21.0 that the defaults of this module reproduce exactly; each has a keyword that removes
it, and each is an exact algebraic identity of redback's expression with the quirk removed. The
upstream write-ups are in ``reports/upstream_issues`` of the porting project (numbers in brackets).

* **Q1** (issue 30). The effective radius is ``sqrt(L / (pi sigma T^4))`` but the spectrum is that
  of a sphere, ``4 pi R^2``. Every band flux carries four times the energy of the bolometric light
  curve (1.5 mag too bright). ``sphere_radius=True`` uses ``sqrt(L / (4 pi sigma T^4))``.
* **Q2** (issue 31). ``del_omega`` is documented "in units of pi" and the prior file quotes 2.0, but
  the engine uses it in steradians. ``del_omega_in_pi=True`` multiplies it by pi first (T falls by
  pi^(1/4)).
* **Q3** (issue 32). :func:`tde_stream_stream_flux_density` builds its flux table on observer-frame
  times and looks it up at source-frame times: time dilation is applied twice and features sit at
  ``(1 + z)^2 t``. ``double_dilation=False`` looks it up at ``t / (1 + z)``. redback's band outputs
  (``magnitude``, ``flux``, ``spectra``) do NOT do this, so
  :func:`tde_stream_stream_ab_magnitude` defaults to ``double_dilation=False`` and
  ``double_dilation=True`` reproduces the ``flux_density`` output through the band integral.
* **Q4** (issue 36). The spectra behind redback's band outputs scale the black body to
  ``F_nu(nu_0) (nu / nu_0)^4 (e^(x_0) - 1) / (e^x - 1)`` with ``nu_0 = 6e14`` Hz, which is
  ``(nu / nu_0)`` times the Planck function (the exponent 4 belongs to nu L_nu, not F_nu). The band
  magnitudes are 0.4 mag too bright in u and 0.4 mag too faint in i. ``nu4_spectrum=False`` is the
  Planck function.
* **Q5** (issue 33). Past the end of the engine grid redback extrapolates the last segment linearly
  (the flux goes negative). ``extrapolate=True`` does the same (a negative flux is clipped to
  ``mag_floor``); the default masks it (D2).
* **Q6** (issue 37). The Gaussian is normalised as ``L_0 / f1`` with ``f1 = exp(-e_0)``,
  ``e_0 = (t_peak - peak_time)^2 / (2 sigma_t^2)``. ``L_0 / f1`` overflows to inf from
  ``e_0`` of about 610 to 640, and ``f1`` itself is zero from 745: the model returns inf/NaN for
  those draws (0.225 % of redback's prior box, 9 of 4000 draws, and 3 of the 16 corners). Here that is D1.
* **Q7** (issue 38). ``peak_time`` is a free parameter that is not tied to the engine's ``t_peak``.
  Away from it the Gaussian side that meets the engine is a falling one and the light curve peaks
  at ``exp((t_peak - peak_time)^2 / (2 sigma_t^2))`` times the engine's maximum luminosity (more
  than 20 times on 31 % of redback's prior box). Reproduced, not changed; see
  :func:`stream_stream_peak_time`.
* **Q8** (issue 34). The docstring of ``stream_stream_tde_bolometric`` says the time is observer-frame;
  there is no redshift in it, so it is source-frame. :func:`tde_stream_stream_bolometric` is
  source-frame.
* redback's piecewise-linear interpolation of the flux between 1100 nodes (100 rise nodes up to 1 ms
  before ``t_peak`` plus the engine's 1000) is reproduced node for node.

**Deviations from redback.**

* **D1** (rule a, and b where redback returns inf/NaN). The Gaussian normalisation is evaluated as
  ``L_0 exp(e_0 - e_1(t))`` instead of ``(L_0 / f1) exp(-e_1(t))``: the same number to a relative
  1e-15 wherever redback's is finite, and finite where redback's overflows. The rise luminosity is
  capped at 1e300 erg/s, which only the degenerate corner of the prior reaches (and where redback's
  own value is finite and below 1e300 the two agree).
* **D2** (rule b). Epochs after the end of the engine grid (1500 days of model time, observer
  ``1500 (1 + z)`` days; ``1500 (1 + z)^2`` for the ``double_dilation`` flux density) have zero flux
  (``mag_floor``) instead of redback's linear extrapolation of the last segment, which falls
  through zero (the flux density is 0.46 times its last value at 1.5 times the end of the grid
  and negative from about 1.9 times) and so returns negative luminosities. Redback's extrapolation
  also runs to negative times; those epochs are at ``mag_floor`` here too.
* **D3** (rule c, band integral). The band magnitudes integrate the black body against the real
  bandpass (as every photometric TDE of this package does) where redback evaluates the spectrum on
  a 100-point wavelength grid, splines it in time and calls sncosmo; the flux density of
  :func:`tde_stream_stream_flux_density` has no band integral and agrees with redback's to round-off.
  Measured against redback's ``magnitude`` output (sdss u g r i, 91 parameter sets whose Gaussian
  peaks within a few widths of the engine's): median 3.4e-5 mag; after the peak 99th percentile
  0.016 and maximum 0.044; before the peak, where redback's cubic spline crosses the steep
  Gaussian wing, 99th percentile 0.020 and maximum 3.0 mag (one parameter set).
* **D4** (rule c). redback's band path splines the spectra in time with a cubic; here the flux is
  interpolated linearly between the same nodes, as redback's own ``flux_density`` path does. The
  measured effect is inside D3's numbers.
* **D5** (rule b). The temperature is floored at 1 K inside the Planck function (as everywhere in
  this package) so that a zero cannot make a NaN.

**Precision.** float64 is required: the luminosity is 1e43 to 1e46 erg/s in cgs, past float32's
3.4e38 ceiling in the black-body step, and the functions raise rather than return a light curve at
``mag_floor``::

    import jax; jax.config.update("jax_enable_x64", True)   # before the first array exists

References
----------
Piran, T., Svirski, G., Krolik, J., Cheng, R. M., Shiokawa, H. 2015, ApJ 806, 164
(https://ui.adsabs.harvard.edu/abs/2015ApJ...806..164P)

Ryu, T., Krolik, J., Piran, T. 2020, ApJ 904, 73 (https://ui.adsabs.harvard.edu/abs/2020ApJ...904...73R)

Krolik, J., Piran, T., Ryu, T. 2024, ApJ 973, 12 (the cooling-time factor)

Sarin, N. et al. 2024, MNRAS 531, 1203 (redback; https://arxiv.org/abs/2308.12806)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._adopted_tde import assemble, require_x64
from .kilonova import (DAY_TO_S, INV_AB_ZEROPOINT, PLANCK_T3_AB, SIGMA_SB, SPEED_OF_LIGHT,
                       _ext_transmission)
from .tde import _flux_nu_tde, flux_density_mjy, stream_stream_collision

__all__ = [
    "PARAMETERS", "stream_stream_peak_time", "stream_stream_table", "tde_stream_stream_bolometric",
    "tde_stream_stream_flux_density", "tde_stream_stream_ab_magnitude", "default_prior",
    "fallback_prior", "tde_stream_stream_model", "register_tde_stream_stream",
]

#: redback's order for ``stream_stream_tde``, minus ``time`` and ``redshift``.
PARAMETERS = ("mbh_6", "mstar", "c1", "f", "h_r", "inc_tcool", "del_omega", "sigma_t", "peak_time")

#: Nodes before the engine grid (redback's ``np.linspace(0, t_peak - 0.001, 100)``) and the gap [s].
N_PRE = 100
PRE_GAP = 0.001
#: The reference frequency of redback's band spectra [Hz] (``freq_0 = 6e14``).
NU_REF = 6e14
#: Largest luminosity of the rise [erg/s], D1: well past anything physical, and finite in float64.
_L_CAP = 1e300


# --- the table ------------------------------------------------------------------------------

def _engine(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, n_time):
    """The engine with ``inc_tcool`` static when it is a Python number and selected otherwise."""
    if isinstance(inc_tcool, (bool, int, float, np.integer, np.floating)):
        return stream_stream_collision(mbh_6, mstar, c1, f, h_r, del_omega,
                                       inc_tcool=int(inc_tcool == 1), n_time=n_time)
    on = stream_stream_collision(mbh_6, mstar, c1, f, h_r, del_omega, inc_tcool=1, n_time=n_time)
    off = stream_stream_collision(mbh_6, mstar, c1, f, h_r, del_omega, inc_tcool=0, n_time=n_time)
    sel = inc_tcool == 1
    return tuple(jnp.where(sel, a, b) for a, b in zip(on, off))


def stream_stream_peak_time(mbh_6, mstar, c1=1.0, f=0.2, h_r=0.2, inc_tcool=1, del_omega=2.0, *,
                            del_omega_in_pi=False):
    """The engine's own peak time ``t_peak = 1.5 t_dyn`` in days: the natural ``peak_time``.

    redback's ``peak_time`` is a free parameter that is not tied to this value (Q7). The Gaussian
    rise meets the engine at ``t_peak``; ``peak_time = t_peak`` puts the maximum of the light curve
    there and nowhere else.

    Parameters
    ----------
    mbh_6 : float
        Black-hole mass, 1e6 solar masses.
    mstar : float
        Stellar mass, solar masses.
    c1, f, h_r, inc_tcool, del_omega : float
        As :func:`stream_stream_table` (redback's defaults).
    del_omega_in_pi : bool
        Read ``del_omega`` as the documented "units of pi" (Q2). It does not move the peak time.

    Returns
    -------
    float or jax.Array
        ``t_peak`` in days.

    References
    ----------
    Piran et al. 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73.

    Notes
    -----
    Needs float64 only through the engine, which raises without it.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax.tde_stream_stream import stream_stream_peak_time
    >>> round(float(stream_stream_peak_time(1.0, 1.0)), 2)
    32.17
    """
    om = del_omega * jnp.pi if del_omega_in_pi else del_omega
    return _engine(mbh_6, mstar, c1, f, h_r, inc_tcool, om, 2)[4] / DAY_TO_S


def stream_stream_table(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time, *,
                        n_time=1000, del_omega_in_pi=False, sphere_radius=False):
    """redback's ``dynamics_output`` of ``stream_stream_tde``: the 1100-node table, traced.

    Parameters
    ----------
    mbh_6 : float
        Black-hole mass, 1e6 solar masses.
    mstar : float
        Stellar mass, solar masses.
    c1 : float
        Size of the emitting region in units of the apocentre of the most bound debris.
    f : float
        Fraction of the bound mass within the semi-major axis of the most bound debris at peak.
    h_r : float
        Aspect ratio of the cooling time (used when ``inc_tcool == 1``).
    inc_tcool : float
        1 includes the photon-cooling factor ``2 / (1 + t_cool / t_dyn)``; anything else does not.
        A Python number is resolved at trace time; a traced value selects between the two engines.
    del_omega : float
        Solid angle of the emitting region; steradians in redback's code (Q2).
    sigma_t : float
        Width of the Gaussian rise, days.
    peak_time : float
        Centre of the Gaussian rise, days (not tied to the engine's peak, Q7).
    n_time : int
        Nodes of the engine grid (redback's 1000).
    del_omega_in_pi : bool
        Q2: multiply ``del_omega`` by pi.
    sphere_radius : bool
        Q1: the radius of a sphere, ``sqrt(L / (4 pi sigma T^4))``, instead of redback's
        ``sqrt(L / (pi sigma T^4))``.

    Returns
    -------
    dict
        ``time`` [s since the start of the rise, 100 + ``n_time`` nodes], ``luminosity`` [erg/s],
        ``temperature`` [K], ``radius`` [cm], and the scalars ``t_dyn`` [s], ``t_peak`` [s],
        ``r_tidal`` [cm].

    References
    ----------
    Piran et al. 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73; Krolik, Piran & Ryu
    2024, ApJ 973, 12.

    Notes
    -----
    The rise follows the module's D1: ``L_0 exp(e_0 - e_1)``, with the exponent capped.
    ``time`` is not redshifted. float64 is required by the engine.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax.tde_stream_stream import stream_stream_table
    >>> tab = stream_stream_table(1.0, 1.0, 1.0, 0.2, 0.2, 1, 2.0, 20.0, 32.17)
    >>> tab["time"].shape
    (1100,)
    """
    om = del_omega * jnp.pi if del_omega_in_pi else del_omega
    t_out, l_out, t_phot, t_dyn, t_peak, r_tidal = _engine(mbh_6, mstar, c1, f, h_r, inc_tcool, om,
                                                           n_time)
    # ORIGINAL (tde_models.py:1574-1590):
    #     f1 = pm.gaussian_rise(time=output.time_temp[0] / cc.day_to_s, a_1=1, peak_time=peak_time, sigma_t=sigma_t)
    #     norm = output.bolometric_luminosity[0] / f1
    #     tt_pre_fb = np.linspace(0, output.time_temp[0]-0.001, 100)
    #     f1_src = pm.gaussian_rise(time=tt_pre_fb, a_1=norm, peak_time=peak_time * cc.day_to_s,
    #                               sigma_t=sigma_t * cc.day_to_s)
    # f1 underflows to 0 (norm = inf, NaN rise) when its exponent passes 745; the product
    # norm * exp(-e_1) is rewritten as L_0 * exp(e_0 - e_1), see D1. The first call is in days and the
    # second in seconds; both are the same ratio, so nothing is lost (issue 35).
    tt_pre = jnp.linspace(0.0, t_out[0] - PRE_GAP, N_PRE)
    e_0 = (t_out[0] / DAY_TO_S - peak_time) ** 2. / (2 * sigma_t ** 2)
    e_1 = (tt_pre - peak_time * DAY_TO_S) ** 2. / (2 * (sigma_t * DAY_TO_S) ** 2)
    pre_l = l_out[0] * jnp.exp(jnp.minimum(e_0 - e_1, jnp.log(_L_CAP / l_out[0])))
    full_time = jnp.concatenate([tt_pre, t_out])
    full_lbol = jnp.concatenate([pre_l, l_out])
    full_temp = jnp.concatenate([jnp.ones(N_PRE) * t_phot[0], t_phot])
    # ORIGINAL: r_eff = np.sqrt(full_lbol / (np.pi * cc.sigma_sb * full_temp**4.0))
    area = 4.0 * jnp.pi if sphere_radius else jnp.pi
    r_eff = jnp.sqrt(full_lbol / (area * SIGMA_SB * full_temp ** 4.0))
    return dict(time=full_time, luminosity=full_lbol, temperature=full_temp, radius=r_eff,
                t_dyn=t_dyn, t_peak=t_peak, r_tidal=r_tidal)


def _locate(node_time, x):
    """Segment ``[lo, lo + 1]`` of the sorted ``node_time`` that holds (or is nearest to) ``x``."""
    lo = jnp.clip(jnp.searchsorted(node_time, x, side="right") - 1, 0, node_time.shape[0] - 2)
    return lo, lo + 1


def _lerp(y_lo, y_hi, x_lo, x_hi, x):
    """scipy ``interp1d`` linear: ``slope * (x - x_lo) + y_lo`` (it extrapolates past the ends)."""
    return (y_hi - y_lo) / (x_hi - x_lo) * (x - x_lo) + y_lo


def _inside(node_time, x, extrapolate):
    """True where the lookup is inside the table (everywhere if ``extrapolate``) and ``x`` >= 0."""
    ok = x >= 0.0
    return ok if extrapolate else ok & (x <= node_time[-1])


# --- bolometric -----------------------------------------------------------------------------

def tde_stream_stream_bolometric(time_days, mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega,
                                 sigma_t, peak_time, *, n_time=1000, del_omega_in_pi=False,
                                 extrapolate=False):
    """redback's ``stream_stream_tde_bolometric`` [erg/s] at SOURCE-frame ``time_days``.

    Parameters
    ----------
    time_days : array_like
        Days since the start of the Gaussian rise, source frame (redback's docstring says
        observer-frame; there is no redshift in the function, Q8).
    mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time : float
        As :func:`stream_stream_table`.
    n_time : int
        Nodes of the engine grid.
    del_omega_in_pi : bool
        Q2.
    extrapolate : bool
        Q5: extend the last (first) segment linearly, as redback does; the default returns 0 after
        the 1500-day end of the grid and before time zero (D2).

    Returns
    -------
    jax.Array
        Luminosity in erg/s, per entry of ``time_days``.

    References
    ----------
    Piran et al. 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73.

    Notes
    -----
    redback interpolates ``full_lbol`` linearly on its 1100 nodes, and so does this function.
    float64 required.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax.tde_stream_stream import tde_stream_stream_bolometric
    >>> lum = tde_stream_stream_bolometric(np.array([10.0, 40.0, 100.0]), 1.0, 1.0, 1.0, 0.2, 0.2,
    ...                                    1, 2.0, 20.0, 32.17)
    >>> bool(lum[1] > lum[2])
    True
    """
    require_x64("tde_stream_stream")
    tab = stream_stream_table(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time,
                              n_time=n_time, del_omega_in_pi=del_omega_in_pi)
    x = jnp.asarray(time_days) * DAY_TO_S
    lo, hi = _locate(tab["time"], x)
    val = _lerp(tab["luminosity"][lo], tab["luminosity"][hi], tab["time"][lo], tab["time"][hi], x)
    return jnp.where(_inside(tab["time"], x, extrapolate), val, 0.0)


# --- photometry -----------------------------------------------------------------------------

def tde_stream_stream_flux_density(t_obs_days, nu_obs_hz, redshift, dl_cm, mbh_6, mstar, c1, f, h_r,
                                   inc_tcool, del_omega, sigma_t, peak_time, *, n_time=1000,
                                   dilation=True, double_dilation=True, del_omega_in_pi=False,
                                   sphere_radius=False, extrapolate=False, ebv_mw=None,
                                   ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1,
                                   r_v_host=3.1, law="f99"):
    """redback's ``stream_stream_tde(..., output_format='flux_density')`` in mJy.

    Parameters
    ----------
    t_obs_days : array_like
        Observer-frame days since the start of the Gaussian rise.
    nu_obs_hz : float or array_like
        Observer-frame frequencies, broadcast against ``t_obs_days``.
    redshift, dl_cm : float
        Redshift and luminosity distance (cm).
    mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time : float
        As :func:`stream_stream_table`.
    n_time : int
        Nodes of the engine grid.
    dilation : bool
        The ``(1 + z)`` factor on the flux density that redback's TDE path applies.
    double_dilation : bool
        Q3. True (redback) looks the table up at ``t_obs / (1 + z)^2``; False at ``t_obs / (1 + z)``.
    del_omega_in_pi, sphere_radius : bool
        Q2 and Q1.
    extrapolate : bool
        Q5; the default masks the epochs outside the table to zero flux (D2). The table's end is
        ``1500 (1 + z)^2`` days with ``double_dilation``, ``1500 (1 + z)`` without.
    ebv_mw, ebv_host, xi_mw, xi_host, r_v_mw, r_v_host, law :
        Optional interstellar extinction, as :func:`whisper_cbpf.models.jax.tde.flux_density_mjy`.

    Returns
    -------
    jax.Array
        Flux density in mJy.

    References
    ----------
    Piran et al. 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73; Sarin et al. 2024,
    MNRAS 531, 1203 (redback).

    Notes
    -----
    With the defaults this is redback's output to float64 round-off for epochs inside the table,
    including Q1 (four times the energy of the bolometric light curve) and Q3. float64 required.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax.tde_stream_stream import tde_stream_stream_flux_density
    >>> fl = tde_stream_stream_flux_density(np.array([20.0, 60.0]), 6.5e14, 0.05, 7.09e26, 1.0, 1.0,
    ...                                     1.0, 0.2, 0.2, 1, 2.0, 20.0, 32.17)
    >>> bool((fl > 0).all())
    True
    """
    require_x64("tde_stream_stream")
    tab = stream_stream_table(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time,
                              n_time=n_time, del_omega_in_pi=del_omega_in_pi,
                              sphere_radius=sphere_radius)
    t_obs = jnp.asarray(t_obs_days)
    nu_obs = jnp.broadcast_to(jnp.asarray(nu_obs_hz), t_obs.shape)
    # ORIGINAL: time = time / (1 + z)  (calc_kcorrected_properties), then the table on
    # total_time = full_time * (1 + z) is read at time * day_to_s: the engine time is
    # t_obs / (1 + z)^2.
    x = t_obs * DAY_TO_S / (1.0 + redshift) ** (2 if double_dilation else 1)
    lo, hi = _locate(tab["time"], x)
    ext = dict(dilation=dilation, ebv_mw=ebv_mw, ebv_host=ebv_host, xi_mw=xi_mw, xi_host=xi_host,
               r_v_mw=r_v_mw, r_v_host=r_v_host, law=law)
    f_lo = flux_density_mjy(tab["temperature"][lo], tab["radius"][lo], nu_obs, redshift, dl_cm, **ext)
    f_hi = flux_density_mjy(tab["temperature"][hi], tab["radius"][hi], nu_obs, redshift, dl_cm, **ext)
    val = _lerp(f_lo, f_hi, tab["time"][lo], tab["time"][hi], x)
    return jnp.where(_inside(tab["time"], x, extrapolate), val, 0.0)


def tde_stream_stream_ab_magnitude(tau_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                                   mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t,
                                   peak_time, *, n_time=1000, mag_floor=40.0, dilation=True,
                                   double_dilation=False, nu4_spectrum=True,
                                   del_omega_in_pi=False, sphere_radius=False, extrapolate=False,
                                   ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                                   r_v_mw=3.1, r_v_host=3.1, law="f99"):
    """AB magnitude of redback's ``stream_stream_tde`` per observation. Traced; **float64 only**.

    Parameters
    ----------
    tau_days : array_like
        SOURCE-frame days since the start of the Gaussian rise (``(t - t_exp) / (1 + z)``); an epoch
        at or before 0, or after the 1500-day end of the engine grid, is at ``mag_floor``.
    band_idx : array_like of int
        Index of each observation's band in the filter set.
    weights, norms : array_like
        From :func:`whisper_cbpf.models.jax.kilonova.ab_weights`.
    lam_obs_ang : array_like
        The wavelength grid of ``weights`` (Angstrom, observer frame).
    redshift, dl_cm : float
        Redshift and luminosity distance (cm).
    mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time : float
        As :func:`stream_stream_table`.
    n_time : int
        Nodes of the engine grid.
    mag_floor, dilation, ebv_mw, ebv_host, xi_mw, xi_host, r_v_mw, r_v_host, law :
        As :func:`whisper_cbpf.models.jax.tde.ab_magnitude`.
    double_dilation : bool
        Q3. False (the default) is redback's band outputs (``magnitude``, ``flux``); True reads the
        table as its ``flux_density`` output does, at ``t_obs / (1 + z)^2``.
    nu4_spectrum : bool
        Q4. True (the default) is the ``(nu / nu_0)`` excess of redback's band spectra.
    del_omega_in_pi, sphere_radius, extrapolate : bool
        Q2, Q1 and Q5.

    Returns
    -------
    jax.Array
        AB magnitude per observation.

    References
    ----------
    Piran et al. 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73; Sarin et al. 2024,
    MNRAS 531, 1203 (redback).

    Notes
    -----
    The defaults reproduce redback's ``magnitude`` output: single time dilation, the ``(nu / nu_0)``
    excess of its spectra, and the four-fold energy of Q1. The flux is interpolated linearly between
    redback's 1100 nodes, and the band integral is the real bandpass (D3, D4).

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax import tde_stream_stream as S
    >>> m = S.tde_stream_stream_model(["lsstg"], redshift=0.05, dl_cm=7.1e26)
    >>> flux_jy = m.predict({"mbh_6": 1.0, "mstar": 1.0, "sigma_t": 20.0, "peak_time": 32.17},
    ...                     np.array([30.0, 3000.0]), np.array(["lsstg", "lsstg"]))
    >>> bool(flux_jy[0] > flux_jy[1])
    True
    """
    require_x64("tde_stream_stream")
    tab = stream_stream_table(mbh_6, mstar, c1, f, h_r, inc_tcool, del_omega, sigma_t, peak_time,
                              n_time=n_time, del_omega_in_pi=del_omega_in_pi,
                              sphere_radius=sphere_radius)
    tau = jnp.asarray(tau_days)
    alive = tau > 0.0
    # The engine time of the observation: t_obs / (1 + z) for redback's band outputs (the table is
    # read on the observer axis), t_obs / (1 + z)^2 for its flux_density output (Q3).
    x = jnp.where(alive, tau, 1.0) * DAY_TO_S / ((1.0 + redshift) if double_dilation else 1.0)
    lo, hi = _locate(tab["time"], x)
    inside = alive & _inside(tab["time"], x, extrapolate)

    nu_obs = SPEED_OF_LIGHT / (lam_obs_ang * 1e-8)
    nu_src = nu_obs[None, :] * (1.0 + redshift)

    def node_flux(idx):
        fl = _flux_nu_tde(tab["temperature"][idx][:, None], tab["radius"][idx][:, None], dl_cm,
                          nu_src, redshift, PLANCK_T3_AB, dilation=dilation)
        # ORIGINAL: flux_den[:, None] * (frequency / freq_0)**4 * (exp(h freq_0/kT) - 1) / (exp(h freq/kT) - 1)
        # = the Planck function times (frequency / freq_0), see Q4.
        return fl * (nu_src / NU_REF) if nu4_spectrum else fl

    f_lo, f_hi = node_flux(lo), node_flux(hi)
    f_ab = _lerp(f_lo, f_hi, tab["time"][lo][:, None], tab["time"][hi][:, None], x[:, None])
    f_ab = jax.lax.optimization_barrier(jnp.where(inside[:, None], f_ab, 0.0))
    tau_ext = _ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host, xi_mw, xi_host, r_v_mw,
                                r_v_host, law)
    w = weights if tau_ext is None else weights * tau_ext[None, :]
    num = jnp.sum(f_ab * w[band_idx], axis=1)
    den = jax.lax.optimization_barrier(INV_AB_ZEROPOINT * norms[band_idx])
    ratio_floor = jnp.asarray(10.0, dtype=num.dtype) ** (-0.4 * mag_floor)
    mag = -2.5 * jnp.log10(jnp.maximum(num / den, ratio_floor))
    return jnp.where(inside, mag, jnp.asarray(mag_floor, mag.dtype))


# --- priors ---------------------------------------------------------------------------------

#: Transcribed from redback 1.21's ``stream_stream_tde.prior`` (``redshift`` dropped), for when
#: redback is not importable. ``c1``, ``del_omega``, ``f``, ``h_r`` and ``inc_tcool`` are fixed
#: values there, so they are pinned, not sampled.
_FALLBACK_PRIOR = (
    {"peak_time": ("LogUniform", 0.1, 60.0), "sigma_t": ("LogUniform", 10.0, 60.0),
     "mbh_6": ("LogUniform", 0.01, 20.0), "mstar": ("LogUniform", 0.1, 10.0)},
    {"c1": 1.0, "del_omega": 2.0, "f": 0.2, "h_r": 0.2, "inc_tcool": 1.0},
)


def fallback_prior():
    """:func:`default_prior` without redback: the transcription of its 1.21 prior file.

    Returns
    -------
    tuple
        ``(Prior, pinned)`` as :func:`default_prior`.

    References
    ----------
    Sarin et al. 2024, MNRAS 531, 1203 (redback).

    Notes
    -----
    The shipped prior file is the source of truth; prefer :func:`default_prior`.

    Examples
    --------
    >>> from whisper_cbpf.models.jax.tde_stream_stream import fallback_prior
    >>> prior, pinned = fallback_prior()
    >>> sorted(pinned)
    ['c1', 'del_omega', 'f', 'h_r', 'inc_tcool']
    """
    from ...priors import LogUniform, Prior, Uniform

    dists, pinned = _FALLBACK_PRIOR
    cls = {"Uniform": Uniform, "LogUniform": LogUniform}
    return Prior({k: cls[c](lo, hi) for k, (c, lo, hi) in dists.items()}), dict(pinned)


def default_prior():
    """``(Prior, pinned)`` of redback's ``stream_stream_tde``: its prior file, read from redback
    where possible (``redshift`` dropped: it is the factory's argument), else :func:`fallback_prior`.

    Returns
    -------
    tuple
        ``(Prior, pinned)``: ``peak_time`` LogUniform(0.1, 60) d, ``sigma_t`` LogUniform(10, 60) d,
        ``mbh_6`` LogUniform(0.01, 20) 1e6 Msun, ``mstar`` LogUniform(0.1, 10) Msun; pinned
        ``c1 = 1``, ``del_omega = 2``, ``f = 0.2``, ``h_r = 0.2``, ``inc_tcool = 1``.

    References
    ----------
    Sarin et al. 2024, MNRAS 531, 1203 (redback).

    Notes
    -----
    The prior is redback's, not modified: ``peak_time`` is drawn independently of the engine's own
    peak (Q7).

    Examples
    --------
    >>> from whisper_cbpf.models.jax.tde_stream_stream import default_prior
    >>> prior, pinned = default_prior()
    >>> sorted(prior.distributions)
    ['mbh_6', 'mstar', 'peak_time', 'sigma_t']
    """
    from .tde import redback_prior

    try:
        return redback_prior("stream_stream_tde")
    except ImportError:
        return fallback_prior()


# --- the whisper model ----------------------------------------------------------------------

def tde_stream_stream_model(band_names, redshift=None, dl_cm=None, *, name="tde_stream_stream_jax",
                            n_wave=None, n_time=1000, double_dilation=False, nu4_spectrum=True,
                            del_omega_in_pi=False, sphere_radius=False, extrapolate=False,
                            pin=None, prior=None, mag_floor=40.0, dilation=True, ebv_mw=None,
                            ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99", filter_set=None,
                            band_aliases=None, default_system=None, t_exp_days=0.0, free=None,
                            redshift_prior=None):
    """Bind dataset context to redback's ``stream_stream_tde`` and return a whisper ``Model``.

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
        The model's name (default ``"tde_stream_stream_jax"``).
    n_time : int
        Nodes of the engine grid (redback's 1000).
    double_dilation, nu4_spectrum, del_omega_in_pi, sphere_radius, extrapolate : bool
        The switches of the module docstring (Q3, Q4, Q2, Q1, Q5). The defaults reproduce
        redback's ``magnitude`` output: ``double_dilation=False`` (that is the band path; the
        ``flux_density`` path is :func:`tde_stream_stream_flux_density`), ``nu4_spectrum=True``,
        ``del_omega_in_pi=False``, ``sphere_radius=False``, ``extrapolate=False`` (D2).
    pin : dict or None
        Override which parameters are held fixed, e.g. ``{"c1": 0.5}``; ``{"del_omega": None}``
        frees redback's pinned ``del_omega`` (and needs ``prior=`` for it).
    prior : whisper_cbpf.priors.Prior or None
        Distributions overriding redback's, per parameter. The default is redback's own prior
        (:func:`default_prior`), not modified.
    mag_floor, dilation, ebv_mw, ebv_host, r_v_mw, r_v_host, law, filter_set, n_wave, band_aliases, default_system :
        As :func:`whisper_cbpf.models.jax.tde_model` (Gauss-16 band integral by default, the
        ``(1 + z)`` flux factor of redback's TDE path, optional extinction).
    t_exp_days : float
        Time the Gaussian rise began on the light curve's own clock (the model's time zero),
        subtracted on the host in float64. Default 0: ``times`` are days since it.
    free : list of str or None
        ``"t_exp"`` and/or ``"redshift"`` fitted as parameters, appended in that order; see
        :func:`whisper_cbpf.models.jax.tde_model`. ``t_exp`` needs ``prior=Prior({"t_exp": ...})``.
    redshift_prior : dict, distribution or None
        The fitted redshift's prior when ``prior`` has none.

    Returns
    -------
    Model
        ``parameters`` ``["mbh_6", "mstar", "sigma_t", "peak_time"]`` (plus the freed context
        values); redback pins ``c1``, ``f``, ``h_r``, ``inc_tcool`` and ``del_omega``. redback
        declares no constraints for this model, so there is no wall.

    References
    ----------
    Piran et al. 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73; Krolik, Piran & Ryu
    2024, ApJ 973, 12; Sarin et al. 2024, MNRAS 531, 1203 (redback). Full list in the module
    docstring.

    Notes
    -----
    **float64 is required** (``jax.config.update("jax_enable_x64", True)`` before the first array
    exists); the guard raises at the first ``predict``, not here. The model ends 1500 days after its
    time zero (the epoch is at ``mag_floor`` after that). ``peak_time`` is not the engine's peak
    time (see :func:`stream_stream_peak_time`).

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax.tde_stream_stream import tde_stream_stream_model
    >>> m = tde_stream_stream_model(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.parameters
    ['mbh_6', 'mstar', 'sigma_t', 'peak_time']
    """
    return assemble(
        "tde_stream_stream", "stream_stream_tde", band_names, redshift, dl_cm, name=name,
        n_wave=n_wave, pin=pin, prior=prior, mag_floor=mag_floor, dilation=dilation,
        ebv_mw=ebv_mw, ebv_host=ebv_host, r_v_mw=r_v_mw, r_v_host=r_v_host, law=law,
        filter_set=filter_set, band_aliases=band_aliases, default_system=default_system,
        t_exp_days=t_exp_days, free=free, redshift_prior=redshift_prior, constraint=None,
        engine_kw=dict(n_time=n_time, double_dilation=double_dilation, nu4_spectrum=nu4_spectrum,
                       del_omega_in_pi=del_omega_in_pi, sphere_radius=sphere_radius,
                       extrapolate=extrapolate),
        summary="stream-stream collisions (Piran et al. 2015), Gaussian rise, fixed-area black body",
        caller="tde_stream_stream_model")


def register_tde_stream_stream(band_names, redshift=None, dl_cm=None, *,
                               name="tde_stream_stream_jax", **kwargs):
    """Build a ``stream_stream_tde`` ``Model`` for this dataset and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`tde_stream_stream_model`.
    name : str, default "tde_stream_stream_jax"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model

    References
    ----------
    Piran et al. 2015, ApJ 806, 164; Ryu, Krolik & Piran 2020, ApJ 904, 73; Sarin et al. 2024,
    MNRAS 531, 1203 (redback).

    Notes
    -----
    Registration is by name, replacing an existing entry of that name.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.jax.tde_stream_stream import register_tde_stream_stream
    >>> m = register_tde_stream_stream(["lsstg", "lsstr"], redshift=0.05, dl_cm=7.1e26)
    >>> m.name in wp.list_models()
    True
    """
    from ._factories import _register

    return _register(tde_stream_stream_model(band_names, redshift, dl_cm, name=name, **kwargs))
