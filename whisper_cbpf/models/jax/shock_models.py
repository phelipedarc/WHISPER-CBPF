"""Adapted from redback-jax (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback-jax, commit 3431e2e, GPL-3.0, and from redback (Sarin et al.), https://github.com/nikhil-sarin/redback, GPL-3.0; corrected and modified for whisper_cbpf.

Shocked jet cocoon (Piro & Kollmeier 2018), in JAX.

The model of redback's ``shocked_cocoon``: a jet drives a shock through ``mej`` of ejecta
(``vej`` in units of c, density slope ``eta``), of which a fraction ``shocked_fraction`` is shocked
out to ``rshock = c * tshock`` within the opening angle ``arccos(cos_theta_cocoon)``. The shocked
shell radiates as its photons diffuse out: ``L = L0 (t / tau_diff)^(-4 / (eta + 2))`` with the
diffusion time ``tau_diff = sqrt(Msun kappa M_sh / (4 pi c vej))``  and a taper once the shell is
transparent at ``t_thin = sqrt(c / vej) tau_diff``, ``(1 + tanh(t_thin - t)) / 2``. The photosphere
is the cocoon's (:func:`whisper_cbpf.models.jax.tde.cocoon_photosphere`: velocity
``vej (t / t_thin)^(-2 / (eta + 3))``), with no temperature floor, and the spectrum a black body.

Differences from redback
------------------------
* The taper is evaluated as ``sigmoid(2 (t_thin - t))``, which equals ``(1 + tanh(.)) / 2``
  exactly but does not cancel. ``1 + tanh(x)`` loses every digit once ``x < -9`` (it is zero to
  round-off, exactly zero beyond 18 d), and its gradient with respect to ``mej``, ``vej`` or
  ``kappa`` disagrees with finite differences on 60 to 90% of draws over 0.01 to 100 d. The stable
  form removes both: the luminosity is positive and its gradient is right in the deep taper.
* whisper's constants (astropy's) instead of rounded literals, which moves L by 9e-5 dex.
* The photosphere is the cocoon's, as in redback, not the generic temperature-floor one.

**float64 is required** for the cgs luminosities (1e38 to 1e45 erg/s) and the diffusion constant.

References
----------
Piro & Kollmeier 2018.
"""
import jax
import jax.numpy as jnp

from .kilonova import DAY_TO_S, SOLAR_MASS, SPEED_OF_LIGHT
from .tde import cocoon_photosphere

__all__ = ["KM_CGS", "shocked_cocoon_engine", "shocked_cocoon_model"]

KM_CGS = 1e5
_C_KMS = SPEED_OF_LIGHT / KM_CGS


def _require_x64(what):
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} requires float64: the luminosities and the diffusion constant "
            f"(Msun / (4 pi c km) = 1e18 in cgs) are far outside what float32 resolves. Enable it "
            f"BEFORE the first jax array is created:\n"
            f"    import jax; jax.config.update('jax_enable_x64', True)\n"
            f"or set JAX_ENABLE_X64=1 in the environment.")


def shocked_cocoon_engine(time_days, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon,
                          kappa):
    """redback's ``_shocked_cocoon`` luminosity, and the two scales the photosphere needs.

    Parameters
    ----------
    time_days : source-frame days.
    mej : ejecta mass, Msun. vej : ejecta velocity in units of c (redback's docstring says km/s;
        its code multiplies by c / km, so it is c). eta : density slope.
    tshock : shock time, s. shocked_fraction : shocked mass over ``mej``.
    cos_theta_cocoon : cosine of the cocoon opening angle. kappa : opacity, cm^2/g.

    Returns
    -------
    ``(lbol [erg/s], t_thin [d], vej [km/s])``.
    """
    vej_kms = vej * _C_KMS
    diff_const = SOLAR_MASS / (4.0 * jnp.pi * SPEED_OF_LIGHT * KM_CGS)
    shocked_mass = mej * shocked_fraction
    rshock = tshock * SPEED_OF_LIGHT
    theta = jnp.arccos(jnp.clip(cos_theta_cocoon, -1.0, 1.0))
    tau_diff = jnp.sqrt(diff_const * kappa * shocked_mass / vej_kms) / DAY_TO_S
    t_thin = (_C_KMS / vej_kms) ** 0.5 * tau_diff
    l0 = ((theta ** 2 / 2.0) ** (1.0 / 3.0)
          * (shocked_mass * SOLAR_MASS * vej_kms * KM_CGS * rshock / (tau_diff * DAY_TO_S) ** 2))
    t_safe = jnp.maximum(time_days, 1e-30)
    log_l = (jnp.log(jnp.maximum(l0, 1e-300)) - 4.0 / (eta + 2.0) * jnp.log(t_safe / tau_diff)
             + jax.nn.log_sigmoid(2.0 * (t_thin - time_days)))
    return jnp.exp(log_l), t_thin, vej_kms


@jax.jit
def shocked_cocoon_model(grid, mej, vej, eta, tshock, shocked_fraction, cos_theta_cocoon, kappa):
    """The shocked cocoon at ``grid['time']`` (source-frame days): ``(lbol, T, R_photosphere)``.

    ``lbol`` in erg/s, ``T`` in K and ``R`` in cm from the cocoon's own photosphere. Parameters as
    :func:`shocked_cocoon_engine`.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.shock_models import shocked_cocoon_model
    >>> t = jnp.array([0.1, 0.5, 2.0, 10.0])
    >>> L, T, R = shocked_cocoon_model({"time": t}, 0.01, 0.3, 3.0, 1.5, 0.5, 0.7, 10.0)
    >>> bool(jnp.all(L > 0)), bool(L[0] > L[3])
    (True, True)
    >>> g = jax.grad(lambda m: shocked_cocoon_model({"time": t}, m, 0.3, 3.0, 1.5, 0.5, 0.7,
    ...                                             10.0)[0].sum())(0.01)
    >>> bool(jnp.isfinite(g))
    True
    """
    _require_x64("shocked_cocoon")
    t = grid["time"]
    lbol, t_thin, vej_kms = shocked_cocoon_engine(t, mej, vej, eta, tshock, shocked_fraction,
                                                  cos_theta_cocoon, kappa)
    temperature, r_photosphere = cocoon_photosphere(t, lbol, t_thin, vej_kms, eta)
    return lbol, temperature, r_photosphere
