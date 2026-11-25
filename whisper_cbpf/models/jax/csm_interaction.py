"""Adapted from redback-jax (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback-jax, commit 3431e2e, GPL-3.0, and from redback (Sarin et al.), https://github.com/nikhil-sarin/redback, GPL-3.0; corrected and modified for whisper_cbpf.

Chevalier-Chatzopoulos CSM interaction supernova, in JAX.

The model of redback's ``csm_interaction``: ejecta of mass ``mej`` and velocity ``vej`` run into a
circumstellar medium of mass ``csm_mass`` and density ``rho (r / r0)^-eta`` (``eta = 0`` a shell,
``eta = 2`` a wind). The forward and reverse shocks of the Chevalier (1982) self-similar solution
convert a fraction ``efficiency`` of the kinetic energy to radiation, with the shock coefficients
``AA``, ``Bf``, ``Br`` tabulated against the ejecta density slope ``nn`` and ``eta`` (Chatzopoulos
et al. 2012, 2013). That luminosity is delayed and smoothed by diffusion through the optically thick
CSM, ``L_obs(t) = (1 / t_d) int L(t') exp(-(t - t') / t_d) dt'`` with
``t_d = kappa M_thick / (4 pi^3 / 9 c r_ph)``. The photosphere is the temperature-floor one
(:func:`whisper_cbpf.models.jax.tde.temperature_floor_photosphere`, ejecta velocity ``vej``) and the
spectrum a black body, as in redback. This is a different model from the Margalit (2022) shell
breakout of ``csm_shock_and_arnett``.

Implementation notes
--------------------
* The table lookup is jnp (a bilinear interpolation, identical to scipy's to 4e-16), so the model is
  traceable and differentiable in every parameter including ``eta`` and ``nn``, and compiles once.
* The diffusion integral is the exact piecewise-linear recurrence of redback 1.20
  (``csm_diffusion_method="cumulative"``), solved with an associative scan, on a geometric grid that
  contains the epochs and evaluates the engine at them, so the answer at an epoch does not depend
  on the other epochs. Against a converged reference this grid (1000 nodes) is within
  2e-4 dex median, 9e-3 p99 and 0.07 dex worst case; redback's own default grid is off by up to
  1.03 dex.
  ``csm_diffusion_method="quadrature"`` is redback's legacy per-epoch integral.

The engine switches off at the forward- and reverse-shock times ``t_FS`` and ``t_RS`` by a hard mask,
as in redback, so the luminosity is discontinuous in the parameters there and the gradient of the
bolometric luminosity through the cut-offs is zero; the diffused light curve is smooth.

**float64 is required**: the luminosities are 1e43 to 1e46 erg/s in cgs, past float32's ceiling.
``eta = 1`` exactly is singular (the photosphere radius has a ``1 / (1 - eta)`` exponent), as in
redback; ``(nn, eta)`` outside the table's ``[6, 14] x [0, 2]`` are extrapolated linearly where
redback raises.

References
----------
Chevalier 1982; Chatzopoulos et al. 2012, 2013; Villar et al. 2017; Jacobson-Galan et al. 2020.
"""
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .kilonova import DAY_TO_S, SOLAR_MASS, SPEED_OF_LIGHT

__all__ = [
    "AU_CGS", "KM_CGS", "T_MIN", "TIME_PAD", "DENSE_RESOLUTION", "QUADRATURE_TIMESTEPS",
    "csm_coefficients", "csm_engine", "csm_diffusion_timescale", "exp_convolution",
    "csm_diffusion", "csm_dense_times", "csm_interaction_bolometric", "TABLE_FILE",
]

AU_CGS = 1.495978707e13          # cm, redback.constants.au_cgs (astropy)
KM_CGS = 1e5                     # cm
#: First node of the engine grid, days, and the padding past the last epoch.
T_MIN = 1e-3
TIME_PAD = 100.0
#: redback's ``dense_resolution`` default, and its ``csm_diffusion_timesteps`` default.
DENSE_RESOLUTION = 1000
QUADRATURE_TIMESTEPS = 3000
#: ``kappa M / (beta c r)`` of redback's ``CSMDiffusion._diffusion_timescale``.
_BETA = 4.0 * np.pi ** 3 / 9.0

TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "csm_table.txt"

@functools.lru_cache(maxsize=1)
def _table():
    """redback's ``utils.get_csm_properties`` reads the table as (eta, nn, Bf, Br, AA) rows, reshapes
    each coefficient to (10, 30).T and interpolates on the (nn, eta) axes. NumPy, not jnp: a jnp array
    built once would keep the dtype in force at that moment. Read on first use, so importing the
    supernova family does not need the file."""
    eta_col, nn_col, bf, br, aa = np.loadtxt(TABLE_FILE, delimiter=",", unpack=True)
    grids = tuple(np.reshape(c, (10, 30)).T for c in (aa, bf, br))
    return np.unique(nn_col), np.unique(eta_col), grids        # 30 values 6..14, 10 values 0..2


def _require_x64(what):
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} requires float64: the luminosities are 1e43 to 1e46 erg/s in cgs, past "
            f"float32's ceiling of 3.4e38, and in float32 the shock luminosity overflows and the "
            f"light curve comes back at the magnitude floor. Enable it BEFORE the first jax "
            f"array is created:\n"
            f"    import jax; jax.config.update('jax_enable_x64', True)\n"
            f"or set JAX_ENABLE_X64=1 in the environment.")


def csm_coefficients(nn, eta):
    """The shock coefficients ``(AA, Bf, Br)`` of Chevalier's solution at ``(nn, eta)``.

    Bilinear interpolation of redback's table (30 values of ``nn`` in 6-14 by 10 of ``eta`` in
    0-2), the same arithmetic as scipy's ``RegularGridInterpolator`` that redback uses but
    traceable and differentiable in both arguments. Outside the table the last cell is extrapolated
    linearly.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from whisper_cbpf.models.jax.csm_interaction import csm_coefficients
    >>> AA, Bf, Br = csm_coefficients(12.0, 0.0)
    >>> round(float(AA), 4), round(float(Bf), 4), round(float(Br), 4)
    (0.1921, 1.1213, 0.9742)
    """
    nn_np, eta_np, (aa_grid, bf_grid, br_grid) = _table()
    nn_axis, eta_axis = jnp.asarray(nn_np), jnp.asarray(eta_np)
    i = jnp.clip(jnp.searchsorted(nn_axis, nn, side="right") - 1, 0, nn_axis.size - 2)
    j = jnp.clip(jnp.searchsorted(eta_axis, eta, side="right") - 1, 0, eta_axis.size - 2)
    tx = (nn - nn_axis[i]) / (nn_axis[i + 1] - nn_axis[i])
    ty = (eta - eta_axis[j]) / (eta_axis[j + 1] - eta_axis[j])

    def bilinear(grid):
        g = jnp.asarray(grid)
        return (g[i, j] * (1 - tx) * (1 - ty) + g[i + 1, j] * tx * (1 - ty)
                + g[i, j + 1] * (1 - tx) * ty + g[i + 1, j + 1] * tx * ty)

    return bilinear(aa_grid), bilinear(bf_grid), bilinear(br_grid)


def csm_engine(time_days, mej, csm_mass, vej, eta, rho, kappa, r0, nn=12.0, delta=1.0,
               efficiency=0.5):
    """The shock luminosity of redback's ``_csm_engine``.

    Parameters
    ----------
    time_days : source-frame days since the explosion (>= 0).
    mej, csm_mass : ejecta and CSM mass, Msun.
    vej : ejecta velocity, km/s.
    eta : CSM density slope. ``rho`` : CSM density amplitude, g/cm^3 (at ``r0``).
    kappa : opacity, cm^2/g. ``r0`` : inner CSM radius, AU.
    nn, delta : outer and inner ejecta density slopes. ``efficiency`` : kinetic to radiated energy.

    Returns
    -------
    ``(lbol [erg/s], r_photosphere [cm], mass_csm_threshold [g])``, the optically thick CSM mass
    in grams, as redback returns it.
    """
    mej_g = mej * SOLAR_MASS
    csm_g = csm_mass * SOLAR_MASS
    r0_cm = r0 * AU_CGS
    vej_cm = vej * KM_CGS
    esn = 3.0 * vej_cm ** 2 * mej_g / 10.0
    AA, Bf, Br = csm_coefficients(nn, eta)

    qq = rho * r0_cm ** eta
    radius_csm = ((3.0 - eta) / (4.0 * jnp.pi * qq) * csm_g
                  + r0_cm ** (3.0 - eta)) ** (1.0 / (3.0 - eta))
    r_photosphere = jnp.abs((-2.0 * (1.0 - eta) / (3.0 * kappa * qq)
                             + radius_csm ** (1.0 - eta)) ** (1.0 / (1.0 - eta)))
    mass_csm_threshold = jnp.abs(4.0 * jnp.pi * qq / (3.0 - eta)
                                 * (r_photosphere ** (3.0 - eta) - r0_cm ** (3.0 - eta)))

    # g_n in log10: the numerator reaches 1e300 for the most massive, fastest ejecta at large nn.
    log_gn = (-jnp.log10(4.0 * jnp.pi * (nn - delta))
              + (nn - 3.0) / 2.0 * jnp.log10(2.0 * (5.0 - delta) * (nn - 5.0) * esn)
              - (nn - 5.0) / 2.0 * jnp.log10((3.0 - delta) * (nn - 3.0) * mej_g))
    g_n = 10.0 ** log_gn

    expo = (nn - eta) / ((nn - 3.0) * (3.0 - eta))
    t_fs = (jnp.abs((3.0 - eta) * qq ** ((3.0 - nn) / (nn - eta))
                    * (AA * g_n) ** ((eta - 3.0) / (nn - eta))
                    / (4.0 * jnp.pi * Bf ** (3.0 - eta))) ** expo
            * mass_csm_threshold ** expo)
    t_rs = (vej_cm / (Br * (AA * g_n / qq) ** (1.0 / (nn - eta)))
            * (1.0 - (3.0 - nn) * mej_g / (4.0 * jnp.pi * vej_cm ** (3.0 - nn) * g_n))
            ** (1.0 / (3.0 - nn))) ** ((nn - eta) / (eta - 3.0))

    t_s = time_days * DAY_TO_S
    t_eng = t_s + 1.0                                     # redback's ti = 1 s offset
    t_pow = t_eng ** ((2.0 * nn + 6.0 * eta - nn * eta - 15.0) / (nn - eta))
    lbol_fs = (2.0 * jnp.pi / (nn - eta) ** 3 * g_n ** ((5.0 - eta) / (nn - eta))
               * qq ** ((nn - 5.0) / (nn - eta)) * (nn - 3.0) ** 2 * (nn - 5.0)
               * Bf ** (5.0 - eta) * AA ** ((5.0 - eta) / (nn - eta)) * t_pow)
    lbol_rs = (2.0 * jnp.pi * (AA * g_n / qq) ** ((5.0 - nn) / (nn - eta))
               * Br ** (5.0 - nn) * g_n * ((3.0 - eta) / (nn - eta)) ** 3 * t_pow)
    lbol_fs = jnp.where(t_fs - t_s > 0, lbol_fs, 0.0)
    lbol_rs = jnp.where(t_rs - t_s > 0, lbol_rs, 0.0)
    return efficiency * (lbol_fs + lbol_rs), r_photosphere, mass_csm_threshold


def csm_diffusion_timescale(kappa, r_photosphere, mass_csm_threshold):
    """redback ``CSMDiffusion._diffusion_timescale``: ``kappa M / (4 pi^3 / 9 c r_ph)`` in days."""
    return kappa * mass_csm_threshold / (_BETA * SPEED_OF_LIGHT * r_photosphere) / DAY_TO_S


def exp_convolution(t, lum, t_diff):
    """``(1/t_d) int_t[0]^t[i] L(t') exp(-(t[i] - t') / t_d) dt'`` for ``L`` linear between nodes.

    Exact for a piecewise-linear ``L`` (the integral over each cell is closed form), starting from 0
    at ``t[0]``. The linear recurrence ``y[i+1] = a[i] y[i] + b[i]`` is solved in O(log n) depth
    with an associative scan, so it is a few microseconds on a GPU where redback's Python loop is
    sequential. ``t`` ascending (equal neighbours are allowed and add nothing), ``t_d`` in the units
    of ``t``.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.csm_interaction import exp_convolution
    >>> t = jnp.linspace(0.0, 50.0, 501)
    >>> y = exp_convolution(t, jnp.ones_like(t), 5.0)       # a constant source switched on at 0
    >>> bool(jnp.allclose(y, 1.0 - jnp.exp(-t / 5.0), atol=1e-12))
    True
    """
    dt = t[1:] - t[:-1]
    slope = (lum[1:] - lum[:-1]) / jnp.where(dt > 0, dt, 1.0)
    x = dt / t_diff
    decay = jnp.exp(-x)
    one_minus = -jnp.expm1(-x)
    # dt - t_d (1 - e^-x), expanded for small x where the subtraction cancels
    tail = jnp.where(x < 1e-4, t_diff * (x ** 2 / 2.0 - x ** 3 / 6.0), dt - t_diff * one_minus)
    drive = lum[:-1] * one_minus + slope * tail

    def combine(left, right):
        return left[0] * right[0], left[1] * right[0] + right[1]

    _, y = jax.lax.associative_scan(combine, (decay, drive))
    return jnp.concatenate([jnp.zeros(1, y.dtype), y])


def csm_dense_times(time_days, n=DENSE_RESOLUTION, t_min=T_MIN, t_pad=TIME_PAD):
    """The engine grid: ``n`` geometric nodes from ``t_min`` to ``time[-1] + t_pad`` merged with the epochs.

    Every epoch is a node, so the diffused luminosity there needs no interpolation. Epochs before
    ``t_min`` enter at ``t_min`` (the luminosity there is zero: nothing has been diffused yet).
    """
    time_days = jnp.asarray(time_days)
    nodes = jnp.exp(jnp.linspace(jnp.log(t_min), jnp.log(jnp.max(time_days) + t_pad), n))
    return jnp.sort(jnp.concatenate([nodes, jnp.maximum(time_days, t_min)]))


def csm_diffusion(time_days, dense_times, luminosity, t_diff, method="cumulative",
                  timesteps=QUADRATURE_TIMESTEPS):
    """redback's ``ip.CSMDiffusion``: the engine ``luminosity`` on ``dense_times``, diffused.

    ``method="cumulative"`` (redback 1.20's default) is the exact solution for a luminosity that is
    linear between the nodes and the epochs together; ``"quadrature"`` is the legacy trapezoid
    over ``timesteps`` nodes per epoch. ``t_diff`` in days. Returns the luminosity at ``time_days``.
    """
    if method == "cumulative":
        inside = jnp.clip(time_days, dense_times[0], dense_times[-1])
        nodes = jnp.sort(jnp.concatenate([dense_times, inside]))
        y = exp_convolution(nodes, jnp.interp(nodes, dense_times, luminosity), t_diff)
        return jnp.interp(time_days, nodes, y)
    if method == "quadrature":
        num = int(round(timesteps / 2.0))
        log_start = jnp.minimum(jnp.log10(t_diff / dense_times[-1]) - 3.0, 0.0)
        lsp = 10.0 ** (log_start + (0.0 - log_start) * jnp.linspace(0.0, 1.0, num))
        xm = jnp.sort(jnp.concatenate([jnp.zeros(1), lsp, 1.0 - lsp, jnp.ones(1)]))
        tb = jnp.maximum(0.0, dense_times[0])
        te = jnp.clip(time_days, tb, dense_times[-1])[:, None]
        nodes = tb + (te - tb) * xm
        args = jnp.interp(nodes, dense_times, luminosity) * jnp.exp((nodes - te) / t_diff)
        return jnp.sum(0.5 * (args[:, 1:] + args[:, :-1]) * (nodes[:, 1:] - nodes[:, :-1]),
                       axis=1) / t_diff
    raise ValueError(f"csm_diffusion_method must be 'cumulative' or 'quadrature', got {method!r}")


def _bolometric(grid, mej, csm_mass, vej, eta, rho, kappa, r0, nn, delta, efficiency,
                csm_diffusion_method, dense_resolution):
    _require_x64("csm_interaction")
    t = grid["time"]
    if dense_resolution is None:
        dense_resolution = grid["dense_times"].shape[0] if "dense_times" in grid else DENSE_RESOLUTION
    nodes = csm_dense_times(t, dense_resolution)
    lum, r_ph, m_thick = csm_engine(nodes, mej, csm_mass, vej, eta, rho, kappa, r0,
                                    nn, delta, efficiency)
    t_diff = csm_diffusion_timescale(kappa, r_ph, m_thick)
    return csm_diffusion(t, nodes, lum, t_diff, csm_diffusion_method)


def csm_interaction_bolometric(grid, mej, csm_mass, vej, eta, rho, kappa, r0, nn=12.0, delta=1.0,
                               efficiency=0.5, csm_diffusion_method="cumulative",
                               dense_resolution=None):
    """Bolometric luminosity of the CSM interaction model, erg/s, at ``grid['time']``.

    ``grid`` holds the SOURCE-frame epochs ``time`` in days since the explosion, in any order (the
    dict :func:`whisper_cbpf.models.jax.supernova.build_sn_grid` builds will do); only ``time`` is
    read, and the node count of ``dense_times`` when it is there. Units and meaning as :func:`csm_engine`.
    ``dense_resolution`` (default 1000, redback's) is the number of geometric engine nodes.
    ``csm_diffusion_method`` is ``"cumulative"`` (default) or ``"quadrature"``, static under ``jit``.

    Traceable and differentiable in every numeric argument, ``eta`` and ``nn`` included, and
    compiled once for any parameter values.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.csm_interaction import csm_interaction_bolometric
    >>> t = jnp.array([5.0, 20.0, 80.0, 200.0])
    >>> L = csm_interaction_bolometric({"time": t}, mej=10.0, csm_mass=5.0, vej=5000.0, eta=0.0,
    ...                                rho=1e-12, kappa=0.34, r0=100.0)
    >>> bool(jnp.all(L > 1e42)), bool(L[1] > L[3])
    (True, True)
    >>> g = jax.grad(lambda e: csm_interaction_bolometric(                      # d L / d eta
    ...     {"time": t}, 10.0, 5.0, 5000.0, e, 1e-12, 0.34, 100.0)[1])(0.5)
    >>> bool(jnp.isfinite(g))
    True
    """
    return _jitted_bolometric(grid, mej, csm_mass, vej, eta, rho, kappa, r0, nn, delta,
                              efficiency, csm_diffusion_method, dense_resolution)


_jitted_bolometric = jax.jit(_bolometric,
                             static_argnames=("csm_diffusion_method", "dense_resolution"))
