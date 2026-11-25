"""Adapted from redback-jax (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback-jax, commit 3431e2e, GPL-3.0, and from redback (Sarin et al.), https://github.com/nikhil-sarin/redback, GPL-3.0; corrected and modified for whisper_cbpf.

Relativistic magnetar-driven supernova ejecta, in JAX.

The model of redback's ``general_magnetar_driven_supernova`` (Sarin et al. 2022; Omand & Sarin
2024): a magnetar with spin-down luminosity ``L(t) = l0 (1 + t / tau_sd)^((1 + nn) / (1 - nn))``
(braking index ``nn``, ``tau_sd`` in s) injects energy into ejecta of mass ``mej`` and kinetic energy
``E_sn`` that expand from a radius of 1e11 cm into a medium of 1e-5 cm^-3. The state ``(gamma, r, V,
E)`` -- bulk Lorentz factor, radius, comoving volume and internal energy -- obeys an ODE with the
magnetar energy thermalised with efficiency ``1 - exp(-3 kappa_gamma mej / (4 pi v^2 t^2))``,
nickel heating ``f_nickel mej`` (Ni-56 and Co-56), adiabatic losses, radiation from the surface
(optical depth ``kappa mej r / (V gamma)``) and the swept-up medium. The observed luminosity is the
comoving one boosted by the Doppler factor squared. The photosphere is the temperature-floor one
whose velocity is the ejecta's own ``v(t)`` (Villar et al. 2017) and the spectrum is the black
body with the 3000 A cut-off of Nicholl et al. (2017); there is no pair cascade and no r-process
heating, as in redback's default.

Solvers
-------
``solver`` selects how the ODE is integrated:

``"auto"`` (default)
    ``"diffrax"`` when diffrax is installed, else ``"rk4"`` with a one-time warning.
``"diffrax"``
    Adaptive Tsit5 with a PID controller (``rtol=1e-5``, ``atol=1e-8``, ``max_steps=8192``,
    ``throw=False``), evaluated at the epochs. About 80 steps; 7.8e-6 dex median, 1.2e-4 p99 and
    8e-4 dex worst case against a tight-tolerance reference over 300 draws of redback's prior. diffrax
    is imported inside the model, so it is an optional dependency: ``pip install "whisper_cbpf[ode]"``.
    A solve that runs out of steps does not raise (it must be traceable): the light curve of that
    draw is returned at the floor and the draw is reported by ``ok=False`` from :func:`magnetar_ode_solve`.
``"rk4"``
    The dependency-free fallback: classical fourth-order Runge-Kutta with ``n_grid`` (default 1000)
    steps spaced evenly in ``log t`` from 1 s to 1e8 s. At 1000 steps its error equals Tsit5's at
    ``rtol=1e-5`` (1e-5 dex median, 3e-3 dex worst case against the converged solution), with 4 000
    right-hand-side evaluations to Tsit5's 500, which is 2.8 times the cost on a GPU at batch 4096
    (28 against 10 microseconds per light curve) and 15 times on a CPU. It is unstable for too few steps (one draw in 300 of
    redback's prior is wrong by 18 dex at 250), so ``n_grid`` below 500 is not safe.
``"euler"``
    redback's own explicit Euler recursion on redback's own time grid (``n_grid=None``, 1998 nodes
    from 1 s to 1e8 s), reproducing redback to round-off. It is first order and differs from the
    true solution by 0.008 dex median and 0.09 dex worst case, and only 0.004 dex at 50 000 steps:
    use it to compare with redback, not to fit.

Precision
---------
**float64 is required**: the state spans 1e11 to 1e50 and Euler increments fall below float32's
epsilon. Every entry point raises in float32 rather than returning a plausible wrong answer.

References
----------
Sarin et al. 2022; Omand & Sarin 2024; Nicholl et al. 2017; Villar et al. 2017.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .kilonova import DAY_TO_S, SIGMA_SB, SOLAR_MASS, SPEED_OF_LIGHT
from .tde import temperature_floor_photosphere

__all__ = [
    "SOLVERS", "DEFAULT_N_GRID", "T_FIRST", "T_LAST", "redback_time_grid", "resolve_solver",
    "magnetar_ode_solve", "general_magnetar_driven_model",
]

#: Constants of redback's ODE (``constants.py``, astropy cgs; and the model's fixed inputs).
PROTON_MASS = 1.67262192595e-24                 # g
RADIATION_CONSTANT = 4.0 * SIGMA_SB / SPEED_OF_LIGHT
KM_CGS = 1e5
NI56_LUM, CO56_LUM = 6.45e43, 1.45e43           # erg/s per Msun of Ni-56 / Co-56
NI56_LIFE, CO56_LIFE = 8.8 * DAY_TO_S, 111.3 * DAY_TO_S
R0_CM = 1.0e11                                  # initial ejecta radius
N_ISM = 1.0e-5                                  # ambient density, cm^-3

SOLVERS = ("auto", "diffrax", "rk4", "euler")
#: Steps of the ``"rk4"`` fallback (log-spaced between ``T_FIRST`` and ``T_LAST`` seconds).
DEFAULT_N_GRID = 1000
T_FIRST, T_LAST = 1.0, 1.0e8
_L_FLOOR = 1e-30
_warned = []


def _require_x64(what):
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} requires float64: the ejecta state spans 1e11 to 1e50 and the Euler "
            f"increments fall below float32's epsilon, so in float32 every output is non-finite. "
            f"Enable it BEFORE the first jax array is created:\n"
            f"    import jax; jax.config.update('jax_enable_x64', True)\n"
            f"or set JAX_ENABLE_X64=1 in the environment.")


def redback_time_grid(t_min=T_FIRST, t_max=T_LAST, resolution=2000):
    """redback's ``get_optimal_time_array(1, 1e8, 2000)``: three geometric segments of 500, 1000 and
    500 nodes joined at ``t_min (t_max / t_min)^(1/4)`` and ``^(3/4)``, duplicates removed (1998
    nodes). Seconds."""
    n_early, n_mid = int(resolution * 0.25), int(resolution * 0.50)
    ratio = t_max / t_min
    trans1, trans2 = t_min * ratio ** 0.25, t_min * ratio ** 0.75
    return np.unique(np.concatenate([np.geomspace(t_min, trans1, n_early),
                                     np.geomspace(trans1, trans2, n_mid),
                                     np.geomspace(trans2, t_max, resolution - n_early - n_mid)]))


def resolve_solver(solver):
    """``"auto"`` -> ``"diffrax"`` if it is installed, else ``"rk4"`` (warning once); validates others."""
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if solver == "diffrax":
        try:
            import diffrax  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "solver='diffrax' needs diffrax, an optional dependency: "
                "pip install \"whisper_cbpf[ode]\" (or solver='rk4', which needs nothing)") from exc
    if solver == "auto":
        try:
            import diffrax  # noqa: F401
            return "diffrax"
        except ImportError:
            if not _warned:
                _warned.append(True)
                warnings.warn(
                    "diffrax is not installed, so the magnetar-driven ejecta ODE is integrated with "
                    f"a fixed-step RK4 on {DEFAULT_N_GRID} log-spaced steps from 1 s to 1e8 s. That "
                    "matches the adaptive Tsit5 of the [ode] extra to 1e-5 dex median and 3e-3 dex "
                    "worst case over redback's prior, but fewer than ~500 steps are unstable and it "
                    "costs 3 to 15 times as much as the adaptive solver. pip install \"whisper_cbpf[ode]\" for the "
                    "adaptive solver.", UserWarning, stacklevel=3)
            return "rk4"
    return solver


# --- the right-hand side -------------------------------------------------------------------------

def _initial_state(mej_g, e_sn):
    beta0 = jnp.minimum(jnp.sqrt(e_sn / (0.5 * mej_g)) / SPEED_OF_LIGHT, 0.9999)
    gamma0 = 1.0 / jnp.sqrt(1.0 - beta0 ** 2)
    return jnp.stack([gamma0, jnp.asarray(R0_CM), jnp.asarray(4.0 / 3.0 * np.pi * R0_CM ** 3),
                      0.5 * beta0 ** 2 * mej_g * SPEED_OF_LIGHT ** 2])


def _clamp(gamma, r, v, e):
    """Keep the state physical (guards redback does not have; inactive inside its prior)."""
    return (jnp.maximum(gamma, 1.0 + 1e-15), jnp.maximum(r, 1e8), jnp.maximum(v, 1e24),
            jnp.maximum(e, 1e30))


def _beta_doppler(gamma):
    """``beta`` (floored at 1e-8 so its square root has a gradient) and the Doppler factor."""
    beta = jnp.sqrt(jnp.maximum(1.0 - 1.0 / gamma ** 2, 1e-16))
    return beta, 1.0 / (gamma * jnp.maximum(1.0 - beta, 1e-15))


def _terms(t, gamma, beta, dop, r, v, e, p):
    """``((dgamma, dr, dV, dE)/dt, L_comoving)`` at time ``t`` [s].

    ``beta`` and ``dop`` are passed in because redback's recursion takes them from the Lorentz
    factor of the START of a step and everything else from the updated one.
    """
    mej_g, kappa, kappa_gamma, nickel_msun, l0, tau_sd, nn = p
    c = SPEED_OF_LIGHT
    magnetar = l0 * (1.0 + t / tau_sd) ** ((1.0 + nn) / (1.0 - nn))
    swept_mass = 4.0 / 3.0 * jnp.pi * r ** 3 * N_ISM * PROTON_MASS
    pressure = e / (3.0 * v)
    t_comoving = dop * t
    l_nickel = nickel_msun * (
        NI56_LUM * jnp.exp(jnp.maximum(-t_comoving / NI56_LIFE, -700.0))
        + CO56_LUM * jnp.exp(jnp.maximum(-t_comoving / CO56_LIFE, -700.0)))
    tau = kappa * (mej_g / v) * (r / gamma)
    r_over_g = r / gamma
    l_emit = jnp.where(tau <= 1.0, e * c / r_over_g, e * c / (jnp.maximum(tau, 1e-15) * r_over_g))

    vej = jnp.maximum(jnp.sqrt(jnp.maximum(1.0 - 1.0 / gamma ** 2, 0.0)) * c, 1e5)
    prefactor = 3.0 * kappa_gamma * mej_g / (4.0 * jnp.pi * vej ** 2)
    efficiency = 1.0 - jnp.exp(-prefactor / jnp.maximum(t, 1.0) ** 2)

    drdt = jnp.maximum(beta, 1e-15) * c / jnp.maximum(1.0 - beta, 1e-15)
    dswept = 4.0 * jnp.pi * r ** 2 * N_ISM * PROTON_MASS * drdt
    dvdt_comoving = 4.0 * jnp.pi * r ** 2 * jnp.maximum(beta, 1e-15) * c
    de_total = efficiency * magnetar + dop ** 2 * (l_nickel - l_emit)
    de_comoving = efficiency * dop ** -2 * magnetar + l_nickel - l_emit - pressure * dvdt_comoving
    denominator = jnp.maximum(jnp.abs(mej_g * c ** 2 + e + 2.0 * gamma * swept_mass * c ** 2), 1e30)
    dgamma = (de_total - gamma * dop * de_comoving - (gamma ** 2 - 1.0) * c ** 2 * dswept) / denominator
    return (dgamma, drdt, dvdt_comoving * dop, de_comoving * dop), l_emit


def _rhs(t, y, p):
    """The ODE ``dy/dt`` with the Doppler factor of the current state (RK4 and Tsit5)."""
    gamma, r, v, e = _clamp(y[0], y[1], y[2], y[3])
    beta, dop = _beta_doppler(gamma)
    return jnp.stack(_terms(t, gamma, beta, dop, r, v, e, p)[0])


def _luminosity(y, p):
    """``(L_obs [erg/s], v [km/s])`` of states ``y`` (``(..., 4)``), the current Doppler factor."""
    gamma, r, v, e = _clamp(y[..., 0], y[..., 1], y[..., 2], y[..., 3])
    beta, dop = _beta_doppler(gamma)
    mej_g, kappa = p[0], p[1]
    tau = kappa * (mej_g / v) * (r / gamma)
    r_over_g = r / gamma
    l_emit = jnp.where(tau <= 1.0, e * SPEED_OF_LIGHT / r_over_g,
                       e * SPEED_OF_LIGHT / (jnp.maximum(tau, 1e-15) * r_over_g))
    return l_emit * dop ** 2, jnp.sqrt(jnp.maximum(1.0 - 1.0 / gamma ** 2, 0.0)) * SPEED_OF_LIGHT / KM_CGS


def _parameters(mej, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel):
    mej_g = mej * SOLAR_MASS
    return (mej_g, kappa, kappa_gamma, f_nickel * mej, l0, tau_sd, nn)


# --- the three integrators -----------------------------------------------------------------------

def _euler_grid(p, y0, ts):
    """redback's explicit Euler recursion on the nodes ``ts`` [s]: ``(L_obs, v_kms)`` at every node."""
    dts = jnp.concatenate([jnp.zeros(1), jnp.diff(ts)])
    zero = jnp.zeros(())

    def step(carry, xs):
        gamma, r, v, e, dg, dr, dv, de = carry
        t, dt = xs
        beta_old, dop_old = _beta_doppler(gamma)               # from the Lorentz factor BEFORE the update
        gamma, r, v, e = _clamp(gamma + dg * dt, r + dr * dt, v + dv * dt, e + de * dt)
        derivs, l_emit = _terms(t, gamma, beta_old, dop_old, r, v, e, p)
        out = (l_emit * dop_old ** 2, jnp.sqrt(jnp.maximum(1.0 - 1.0 / gamma ** 2, 0.0))
               * SPEED_OF_LIGHT / KM_CGS)
        return (gamma, r, v, e) + derivs, out

    carry0 = (y0[0], y0[1], y0[2], y0[3], zero, zero, zero, zero)
    _, (lum, vkms) = jax.lax.scan(step, carry0, (ts, dts))
    return lum, vkms


def _rk4_grid(p, y0, ts):
    """Classical RK4 over the node pairs of ``ts`` [s]: the states at every node, ``(n, 4)``."""
    def step(y, tt):
        t0, t1 = tt
        h = t1 - t0
        k1 = _rhs(t0, y, p)
        k2 = _rhs(t0 + h / 2.0, y + h / 2.0 * k1, p)
        k3 = _rhs(t0 + h / 2.0, y + h / 2.0 * k2, p)
        k4 = _rhs(t1, y + h * k3, p)
        y_new = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_new, y_new

    _, ys = jax.lax.scan(step, y0, (ts[:-1], ts[1:]))
    return jnp.concatenate([y0[None], ys])


def _diffrax_states(p, y0, t_eval, rtol, atol, max_steps):
    """Tsit5 states at ``t_eval`` [s] (any order, repeats allowed) and whether the solve finished."""
    from diffrax import ODETerm, PIDController, RESULTS, SaveAt, Tsit5, diffeqsolve

    order = jnp.argsort(t_eval)
    ts = t_eval[order]
    t0 = jnp.minimum(T_FIRST, 0.9 * ts[0])
    t1 = jnp.maximum(1.001 * ts[-1], t0 + 1.0)
    sol = diffeqsolve(ODETerm(_rhs), Tsit5(), t0=t0, t1=t1, dt0=10.0, y0=y0, args=p,
                      saveat=SaveAt(ts=ts), stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      max_steps=max_steps, throw=False)
    ok = sol.result == RESULTS.successful
    ys = jnp.where(ok, sol.ys, y0)                  # a failed solve leaves inf in the unreached nodes
    return ys[jnp.argsort(order)], ok


def _grid_seconds(solver, n_grid):
    if solver == "euler":
        return jnp.asarray(redback_time_grid() if n_grid is None else np.geomspace(T_FIRST, T_LAST, n_grid))
    return jnp.asarray(np.geomspace(T_FIRST, T_LAST, DEFAULT_N_GRID + 1 if n_grid is None else n_grid + 1))


def _solve(time_days, mej, e_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, solver, n_grid, rtol,
           atol, max_steps):
    """``(L [erg/s], v [km/s], ok)`` at ``time_days``; the Euler grid values are returned too."""
    p = _parameters(mej, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel)
    y0 = _initial_state(p[0], e_sn)
    t_eval = jnp.maximum(jnp.asarray(time_days) * DAY_TO_S, T_FIRST)
    if solver == "diffrax":
        ys, ok = _diffrax_states(p, y0, t_eval, rtol, atol, max_steps)
        lum, vkms = _luminosity(ys, p)
        return lum, vkms, ok, None
    ts = _grid_seconds(solver, n_grid)
    if solver == "euler":
        lum_g, v_g = _euler_grid(p, y0, ts)
        return (jnp.interp(t_eval, ts, lum_g), jnp.interp(t_eval, ts, v_g), jnp.asarray(True),
                (ts / DAY_TO_S, lum_g, v_g))
    lum_g, v_g = _luminosity(_rk4_grid(p, y0, ts), p)
    return (jnp.exp(jnp.interp(t_eval, ts, jnp.log(jnp.maximum(lum_g, _L_FLOOR)))),
            jnp.interp(t_eval, ts, v_g), jnp.asarray(True), None)


def _static(solver, n_grid):
    return resolve_solver(solver)


def magnetar_ode_solve(time_days, mej, e_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel=0.0, *,
                       solver="auto", n_grid=None, rtol=1e-5, atol=1e-8, max_steps=8192):
    """Integrate the ejecta ODE and return ``(lbol [erg/s], vej [km/s], ok)`` at ``time_days``.

    ``time_days`` are source-frame days since the explosion, any order, repeats allowed (epochs
    before 1 s are held at 1 s). ``mej`` Msun, ``e_sn`` erg, ``kappa`` and ``kappa_gamma`` cm^2/g,
    ``l0`` erg/s, ``tau_sd`` s, ``nn`` the braking index, ``f_nickel`` the Ni-56 mass fraction.
    ``ok`` is False when the ``"diffrax"`` solver ran out of ``max_steps`` (the outputs are then
    meaningless); the fixed-step solvers always report True. See the module for ``solver`` and
    ``n_grid`` (``None``: 1000 steps for ``"rk4"``, redback's 1998-node grid for ``"euler"``).

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.magnetar_ode import magnetar_ode_solve
    >>> t = jnp.array([10.0, 40.0, 100.0])
    >>> L, v, ok = magnetar_ode_solve(t, 5.0, 1e51, 0.1, 1e44, 1e5, 3.0, 0.1, 0.05, solver="rk4")
    >>> bool(ok), bool(jnp.all(L > 1e41)), bool(jnp.all(v > 1e3))
    (True, True, True)
    """
    _require_x64("general_magnetar_driven")
    lum, vkms, ok, _ = _jitted_solve(time_days, mej, e_sn, kappa, l0, tau_sd, nn, kappa_gamma,
                                     f_nickel, _static(solver, n_grid), n_grid, rtol, atol, max_steps)
    return lum, vkms, ok


_STATIC = ("solver", "n_grid", "max_steps")
_jitted_solve = jax.jit(_solve, static_argnames=_STATIC)


def _model(grid, mej, e_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, temperature_floor,
           solver, n_grid, rtol, atol, max_steps):
    t = grid["time"]
    lum, vkms, ok, euler = _solve(t, mej, e_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel,
                                  solver, n_grid, rtol, atol, max_steps)
    if euler is not None:
        # redback builds the photosphere on its grid and interpolates T, R and L from there
        t_g, lum_g, v_g = euler
        temp_g, rad_g = temperature_floor_photosphere(t_g, lum_g, v_g, temperature_floor)
        return (lum, jnp.interp(t, t_g, temp_g), jnp.interp(t, t_g, rad_g))
    lum = jnp.where(ok, lum, _L_FLOOR)
    temp, rad = temperature_floor_photosphere(t, lum, jnp.where(ok, vkms, 0.0), temperature_floor)
    return lum, temp, rad


_jitted_model = jax.jit(_model, static_argnames=_STATIC)


def general_magnetar_driven_model(grid, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel,
                                  temperature_floor, *, solver="auto", n_grid=None, rtol=1e-5,
                                  atol=1e-8, max_steps=8192):
    """The magnetar-driven supernova at ``grid['time']``: ``(lbol, T, R_photosphere)``.

    ``grid['time']`` are source-frame days since the explosion. ``lbol`` [erg/s], ``T`` [K] and
    ``R`` [cm] are those of the temperature-floor photosphere with the ejecta's own velocity, for
    :func:`whisper_cbpf.models.jax.supernova.sn_ab_magnitude` with the cut-off black body. Parameters
    and ``solver`` as :func:`magnetar_ode_solve`, plus ``temperature_floor`` [K]. Traceable and
    differentiable (a failed ``"diffrax"`` solve gives the floor luminosity and a zero gradient).
    """
    _require_x64("general_magnetar_driven")
    return _jitted_model(grid, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel,
                         temperature_floor, _static(solver, n_grid), n_grid, rtol, atol, max_steps)
