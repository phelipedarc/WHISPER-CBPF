"""Two-component (and N-component) kilonova in JAX -- redback's ``two_component_kilonova_model``.

**How the components combine.** redback (``kilonova_models.py:1092-1181``) evaluates
``_one_component_kilonova_model`` once per component and sums the emitted radiation, not the
luminosities. Each component keeps its own photosphere ``(T, R)`` and its own
``temperature_floor``, so they are two independent blackbodies; there is no shared photosphere, no
opacity mixing and no reprocessing. Summing L and forming a single blackbody at some mean
temperature would be a different, and wrong, model.

So **fluxes add before the band integral and before the log**, and magnitudes of separate
components must never be summed or averaged. Because the band integral and the AB zero point are
both linear in f_nu, summing the spectra and integrating once equals integrating each and summing
-- but only in flux.

**Why this imports the one-component core rather than copying it.** ``kilonova.py`` carries eight
algebraic identities, each removing a specific numerical defect that a copy made before them would
silently reintroduce -- among them an arctan branch applied on the wrong side of t0 (58% error in
L), an XLA reassociation that makes float32 flux density NaN, a float32 gradient overflow that
flips the sign of d/d(vej), and a magnitude clamp that does nothing because
``float32(1e-300) == 0``. Sharing the core means a fix lands once, and this file cannot drift.

**Validity horizon -- read this before comparing against redback.** With ``time_grid=None``,
``kilonova.py``'s converged quadrature diverges from redback past ``t ~ 2.66 * t_diff``, where
redback's trapezoid is under-resolved; with ``time_grid=redback_time_grid(...)`` (the factories'
default) it reproduces redback there, error included. With two components there are two ``t_diff``
values, and under redback's own prior they can differ by up to a factor of 21. So:

* use ``2.66 * min(t_diff_1, t_diff_2)`` as the horizon -- not the max, and not a flux-weighted
  average, because the *short* component enters the bad regime first;
* cross-check per component and per band, never on the summed light curve. A bright, correct
  component can mask a 10x-wrong faint one, letting a broken port pass an H/Ks test while g-band is
  a magnitude off. The blue (short-``t_diff``) component both dominates the blue bands and breaks
  first.

Note also that ``2comp(p1, p2) != 1comp(p1) + 1comp(p2)`` *in redback*, because its two-component
path ends its dense grid at 6 d and its one-component path at 81 d (redback 1.20; in 1.15 they
started at 1e-2 s and 1e-3 s). Do not assert that identity against redback's two-component model.
Here the two do agree, which is a property worth testing -- and the sum of one-component calls is
what whisper takes as the two-component model.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .kilonova import (
    INV_AB_ZEROPOINT,
    MAG_FLOOR,
    MJY,
    PLANCK_T3,
    PLANCK_T3_AB,
    SPEED_OF_LIGHT,
    TDIFF_CONST,
    _ext_transmission,
    _flux_nu,
    bolometric,
    source_time_s,
)

__all__ = ["bolometric_multi", "flux_density_mjy_multi", "ab_magnitude_multi",
           "n_component_magnitude", "two_component_magnitude",
           "three_component_magnitude", "two_component_flux_density", "t_diff_days",
           "validity_horizon_days", "PARAMETERS", "PARAMETERS_3", "default_prior",
           "villar_prior"]

#: redback's positional order for two_component_kilonova_model, minus `time`/`redshift`. Kept
#: verbatim so a call can be checked against redback argument by argument -- note it interleaves
#: (mej, vej, temperature_floor, kappa) per component rather than grouping by quantity.
PARAMETERS = ["mej_1", "vej_1", "temperature_floor_1", "kappa_1",
              "mej_2", "vej_2", "temperature_floor_2", "kappa_2"]


def t_diff_days(mej, vej, kappa):
    """Diffusion timescale in days, per component."""
    return float(np.sqrt(TDIFF_CONST * kappa * mej / vej)) / 86400.0


def validity_horizon_days(*args, factor=2.66):
    """Beyond this, redback's quadrature is under-resolved for at least one component.

    The MINIMUM over components, not the maximum: the short-t_diff component fails first and its
    error is added to still-correct siblings, so the summed curve is already wrong.

    Accepts either the flat two-component form ``(mej_1, vej_1, kappa_1, mej_2, vej_2, kappa_2)``
    or, for any component count, three equal-length sequences ``(mej, vej, kappa)``.
    """
    if len(args) == 3 and all(np.ndim(a) > 0 for a in args):
        mej, vej, kappa = (np.atleast_1d(a) for a in args)
    elif len(args) % 3 == 0:
        mej = np.array(args[0::3], dtype=float)
        vej = np.array(args[1::3], dtype=float)
        kappa = np.array(args[2::3], dtype=float)
    else:
        raise TypeError("pass (mej, vej, kappa) sequences, or a flat (mej, vej, kappa) per "
                        f"component; got {len(args)} positional arguments")
    return factor * float(min(t_diff_days(m, v, k) for m, v, k in zip(mej, vej, kappa)))


def n_component_magnitude(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                          mej, vej, kappa, temperature_floor, **kw):
    """AB magnitude of the summed SED for ANY number of components.

    ``mej``, ``vej``, ``kappa``, ``temperature_floor`` are ``(n_comp,)``. This is just
    :func:`ab_magnitude_multi` under a name that says what it does; the two- and three-component
    wrappers below exist only to accept flat scalar arguments in redback's ordering.
    """
    return ab_magnitude_multi(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                              jnp.asarray(mej), jnp.asarray(vej), jnp.asarray(kappa),
                              jnp.asarray(temperature_floor), **kw)


def three_component_magnitude(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                              mej_blue, vej_blue, temperature_floor_blue, kappa_blue,
                              mej_purple, vej_purple, temperature_floor_purple, kappa_purple,
                              mej_red, vej_red, temperature_floor_red, kappa_red, **kw):
    """Blue + purple + red, summed in FLUX before the band integral (Villar+2017 3-component).

    The components are independent blackbodies with their own photosphere and temperature floor --
    exactly as for two -- so nothing about the physics changes with the extra shell. `vmap` over the
    component axis already handled N; this only names the parameters.
    """
    st = lambda *v: jnp.stack([jnp.asarray(x) for x in v])   # noqa: E731
    return n_component_magnitude(
        t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
        st(mej_blue, mej_purple, mej_red), st(vej_blue, vej_purple, vej_red),
        st(kappa_blue, kappa_purple, kappa_red),
        st(temperature_floor_blue, temperature_floor_purple, temperature_floor_red), **kw)


def bolometric_multi(t_src_s, mej, vej, kappa, temperature_floor,
                     arnett_prefactor=None, time_grid=None):
    """Per-component ``(L/LSCALE, T, R)``, each ``(n_comp, n_obs)``.

    ``mej, vej, kappa, temperature_floor`` are ``(n_comp,)``. A plain vmap over the one-component
    core, so it generalises to N components for free. ``arnett_prefactor`` is per the
    one-component core (None = its ARNETT_PREFACTOR default = redback's 1.0; see kilonova.py) --
    passed OUTSIDE the vmap because it is a trace-time python float, not a traced value.
    ``time_grid`` is the one-component core's too, shared by every component: redback 1.20's
    grid depends on the epochs only, so the sum of one-component redback calls solves each
    component on the same grid.
    """
    from .kilonova import ARNETT_PREFACTOR
    ap = ARNETT_PREFACTOR if arnett_prefactor is None else arnett_prefactor
    fn = lambda t, m, v, k, tf: bolometric(t, m, v, k, tf, arnett_prefactor=ap,  # noqa: E731
                                           time_grid=time_grid)
    return jax.vmap(fn, in_axes=(None, 0, 0, 0, 0))(
        t_src_s, mej, vej, kappa, temperature_floor)


def flux_density_mjy_multi(t_src_s, nu_obs_hz, redshift, dl_cm,
                           mej, vej, kappa, temperature_floor, arnett_prefactor=None,
                           time_grid=None):
    """Flux density in mJy, summed over components."""
    _, temp, rad = bolometric_multi(t_src_s, mej, vej, kappa, temperature_floor,
                                    arnett_prefactor=arnett_prefactor, time_grid=time_grid)
    nu_src = nu_obs_hz * (1.0 + redshift)
    f_nu = _flux_nu(temp, rad, dl_cm, nu_src, redshift, PLANCK_T3)   # (n_comp, n_obs)
    return jnp.sum(f_nu, axis=0) / MJY


def ab_magnitude_multi(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                       mej, vej, kappa, temperature_floor, *, mag_floor=MAG_FLOOR,
                       arnett_prefactor=None, time_grid=None,
                       ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                       r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """AB magnitude of the SUMMED SED.

    Fluxes are added before the band integral and before the log. Extinction applies to the summed
    spectrum -- it is a property of the sightline, not of a component, so reddening each component
    separately and adding would be the same thing only by linearity, and reddening after the log
    would be wrong.

    MEMORY: the intermediate is ``(n_comp, n_obs, n_wave)``. That is the driver here -- with 2
    components, 300 epochs and 5000 wavelengths it is already 24 MB per batch element, so a vmapped
    caller must chunk. See ``abc_gpu.DEFAULT_CHUNK`` for the same trap.
    """
    _, temp, rad = bolometric_multi(t_src_s, mej, vej, kappa, temperature_floor,
                                    arnett_prefactor=arnett_prefactor, time_grid=time_grid)
    nu_obs = SPEED_OF_LIGHT / (lam_obs_ang * 1e-8)
    nu_src = nu_obs[None, None, :] * (1.0 + redshift)

    # AB-zero-point units (identity 8), matching the one-component path: this is what keeps
    # d(mag)/d(flux) inside float32 rather than overflowing to inf and poisoning the gradient.
    f_nu = _flux_nu(temp[:, :, None], rad[:, :, None], dl_cm, nu_src, redshift, PLANCK_T3_AB)
    f_tot = jnp.sum(f_nu, axis=0)                                   # (n_obs, n_wave), IN FLUX
    tau = _ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host,
                            xi_mw, xi_host, r_v_mw, r_v_host, law)
    w = weights if tau is None else weights * tau[None, :]
    num = jnp.sum(f_tot * w[band_idx], axis=1)
    den = jax.lax.optimization_barrier(INV_AB_ZEROPOINT * norms[band_idx])
    floor = 10.0 ** (-0.4 * mag_floor)
    return -2.5 * jnp.log10(jnp.maximum(num / den, floor))


def _stack(a, b):
    return jnp.stack([jnp.asarray(a), jnp.asarray(b)])


def two_component_magnitude(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
                            mej_1, vej_1, temperature_floor_1, kappa_1,
                            mej_2, vej_2, temperature_floor_2, kappa_2, **kw):
    """AB magnitude of the summed two-component SED. Argument order follows redback's.

    Component 1 is conventionally the blue / lanthanide-poor (low kappa) ejecta and component 2 the
    red / lanthanide-rich (high kappa), but nothing here enforces that -- redback's own default
    prior gives both components the SAME kappa range, U(1, 30).
    """
    return ab_magnitude_multi(
        t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm,
        _stack(mej_1, mej_2), _stack(vej_1, vej_2), _stack(kappa_1, kappa_2),
        _stack(temperature_floor_1, temperature_floor_2), **kw)


def two_component_flux_density(t_src_s, nu_obs_hz, redshift, dl_cm,
                               mej_1, vej_1, temperature_floor_1, kappa_1,
                               mej_2, vej_2, temperature_floor_2, kappa_2, time_grid=None):
    """Flux density in mJy, summed over both components."""
    return flux_density_mjy_multi(
        t_src_s, nu_obs_hz, redshift, dl_cm,
        _stack(mej_1, mej_2), _stack(vej_1, vej_2), _stack(kappa_1, kappa_2),
        _stack(temperature_floor_1, temperature_floor_2), time_grid=time_grid)


def default_prior():
    """redback's defaults, verbatim from ``priors/two_component_kilonova_model.prior``.

    Both components share the SAME bounds -- redback does not break the blue/red symmetry in its
    prior, and neither do we. (whisper's own ``models/two_component_kilonova.py`` hand-picks
    ``kappa_1 U(0.1, 0.5)``, which is *disjoint* from redback's ``U(1, 30)``; that wrapper is not
    the reference for an apples-to-apples comparison.)

    ``redshift`` is omitted: it is fixed per dataset here, because ``dl_cm`` would otherwise have to
    move with it through a cosmology on every likelihood call.
    """
    from ...priors import LogUniform, Prior, Uniform

    return Prior({
        "mej_1": Uniform(1e-2, 0.03), "vej_1": Uniform(0.1, 0.5),
        "temperature_floor_1": LogUniform(100.0, 6000.0), "kappa_1": Uniform(1.0, 30.0),
        "mej_2": Uniform(1e-2, 0.03), "vej_2": Uniform(0.1, 0.5),
        "temperature_floor_2": LogUniform(100.0, 6000.0), "kappa_2": Uniform(1.0, 30.0),
    })


#: Villar+2017 parameter order, blue -> purple -> red, matching `villar_prior`.
PARAMETERS_3 = ["mej_blue", "vej_blue", "temperature_floor_blue", "kappa_blue",
                "mej_purple", "vej_purple", "temperature_floor_purple", "kappa_purple",
                "mej_red", "vej_red", "temperature_floor_red", "kappa_red"]


def villar_prior(n_components=2, with_sigma=True):
    """Villar+2017 multi-component kilonova priors, all opacities free.

    The opacity ranges are disjoint, and that is the point::

        kappa_blue    U(0.1, 1.0)
        kappa_purple  U(1.0, 5.0)     (three-component only)
        kappa_red     U(5.0, 30.0)    (U(1.0, 30.0) when there is no purple component)

    With identical per-component priors the posterior is exactly symmetric under relabelling: every
    mode has a mirror twin, chains lock into different labellings, r-hat measures which one each
    chain fell into rather than convergence, and per-parameter medians are medians of a mixture. The
    usual patch is to sort each draw afterwards. Disjoint opacities remove the problem at the
    source: the components are no longer exchangeable, so there is nothing to relabel, and the
    ordering constraint costs no extra machinery. A downstream sort-by-kappa becomes a harmless
    no-op.

    WHY THIS REPLACES redback's DEFAULT AS THE DEFAULT HERE
        redback's `two_component_kilonova_model.prior` cannot contain the published AT2017GFO
        solution: Villar's kappa_blue = 0.5 is below its Uniform(1, 30) floor and M_ej_red = 0.050
        is above its Uniform(0.01, 0.03) ceiling. Fitting inside it rails on both parameters and
        reaches chi2/N ~ 1355; this prior reaches ~1.

    Note `sigma` is Uniform(0, 1) magnitudes here, not log-uniform. It is an extra scatter that may
    genuinely be near zero, and a log-uniform prior cannot represent that -- it puts infinite
    density at the lower edge and none at zero.
    """
    from ...priors import LogUniform, Prior, Uniform

    if n_components == 2:
        d = {
            "mej_blue": Uniform(1e-3, 0.1), "vej_blue": Uniform(0.03, 0.40),
            "temperature_floor_blue": LogUniform(100.0, 5000.0), "kappa_blue": Uniform(0.1, 1.0),
            "mej_red": Uniform(1e-3, 0.1), "vej_red": Uniform(0.01, 0.30),
            "temperature_floor_red": LogUniform(100.0, 5000.0), "kappa_red": Uniform(1.0, 30.0),
        }
    elif n_components == 3:
        d = {
            "mej_blue": Uniform(1e-3, 0.1), "vej_blue": Uniform(0.03, 0.40),
            "temperature_floor_blue": LogUniform(100.0, 5000.0), "kappa_blue": Uniform(0.1, 1.0),
            "mej_purple": Uniform(1e-3, 0.1), "vej_purple": Uniform(0.01, 0.30),
            "temperature_floor_purple": LogUniform(100.0, 5000.0),
            "kappa_purple": Uniform(1.0, 5.0),
            "mej_red": Uniform(1e-3, 0.1), "vej_red": Uniform(0.01, 0.30),
            "temperature_floor_red": LogUniform(100.0, 5000.0), "kappa_red": Uniform(5.0, 30.0),
        }
    else:
        raise ValueError(f"villar_prior supports 2 or 3 components, got {n_components}")
    if with_sigma:
        d["sigma"] = Uniform(0.0, 1.0)          # magnitudes; may legitimately be ~0
    return Prior(d)


bolometric_multi_jit = jax.jit(bolometric_multi)
flux_density_multi_jit = jax.jit(flux_density_mjy_multi)
two_component_flux_density_jit = jax.jit(two_component_flux_density)
