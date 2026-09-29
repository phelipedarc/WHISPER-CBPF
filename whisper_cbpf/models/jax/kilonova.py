"""JAX implementation of redback's ``one_component_kilonova_model``.

Depends on jax only; sncosmo is used once, offline, to export filter curves (see
:func:`make_filter_set`). This module is also the shared substrate for
:mod:`whisper_cbpf.models.jax.tde` and :mod:`whisper_cbpf.models.jax.supernova`: the Planck
function, the AB band integral, the diffusion quadrature and the interstellar extinction laws all
live here. The filter export (``make_filter_set``, ``ab_weights``) moved to
:mod:`whisper_cbpf.synphot.grid_rule` in whisper 0.1.1 and is re-exported here unchanged; the
factories now default to :func:`whisper_cbpf.synphot.gauss_rule` (``docs/PHOTOMETRY.md``).

The physics is redback's. Eight expressions are rewritten using exact algebraic identities that
remove overflow or catastrophic cancellation present in the original; five of them are what make
the float32 path usable at all. They, and the deliberate deviations -- chiefly that the band
integral is taken against the real bandpass at the observation times rather than splined off a
dense grid -- are listed in ``docs/PORTING_NOTES.md``.

Optional extras, all default-off and all free when unused: an explosion time, interstellar
extinction (F99 or CCM89, Milky Way and/or host), and exact filter cell integrals.
"""

import warnings

import numpy as np
import jax
import jax.numpy as jnp

# --- constants (redback/constants.py, astropy cgs) -------------------------
DAY_TO_S = 86400.0
SOLAR_MASS = 1.988409870698051e33      # g
SPEED_OF_LIGHT = 2.99792458e10         # cm / s
PLANCK = 6.62607015e-27                # erg s
BOLTZMANN = 1.380649e-16               # erg / K
SIGMA_SB = 5.6703744191844314e-5       # erg cm^-2 s^-1 K^-4
BETA = 13.7
AB_ZEROPOINT = 3631.0e-23              # erg s^-1 cm^-2 Hz^-1
MJY = 1e-26                            # erg s^-1 cm^-2 Hz^-1 per mJy

# Frequencies are carried in units of NUSCALE. nu**3 in cgs is ~1e44 and
# overflows float32; MANUAL REGROUPING IS NOT ENOUGH, because XLA's algebraic
# simplifier reassociates the product under jit and reconstitutes nu**3 (eager
# gives the right answer, jit gives NaN). Folding the scale into a single
# precomputed literal leaves nothing to reassociate.
NUSCALE = 1e14
PLANCK_PREF = 2.0 * np.pi * PLANCK * NUSCALE ** 3 / SPEED_OF_LIGHT ** 2
PLANCK_ARG = PLANCK * NUSCALE / BOLTZMANN

# NUSCALE alone is NOT enough for the nu^3 prefactor (identity 4 below). XLA rewrites
# (c*nu)**3 -> c**3 * nu**3, hoisting the scale back out: nu_source**3 = 1.3e44 -> +inf in
# float32, while PLANCK_PREF*RATIO_CONST/NUSCALE**3 = 4.6e-71 constant-folds to exactly 0.0
# (below float32's 1.4e-45 subnormal). inf * 0 = NaN. Re-expressing the prefactor in terms of
# x = h nu / (k T) leaves only O(1) quantities to cube, so no reassociation can overflow.
PLANCK_T3 = 2.0 * np.pi * BOLTZMANN ** 3 / (PLANCK ** 2 * SPEED_OF_LIGHT ** 2)

# GRADIENT-safety scale for synthetic photometry (identity 8, float32 backward pass).
# f_nu in cgs is ~1e-38 erg/s/cm^2/Hz at AB 44 -- already SUBNORMAL in float32 -- and
#     d(mag)/d(num) = 2.5 / (ln10 * u * norms),   u = num/norms = 10**(-0.4 mag)
# carries BOTH 1/u and 1/norms, with norms = 3631 Jy * int T dlam/lam ~ 5e-22..4e-21.
# Measured at AB 43.8: 7.0e38 > FLT_MAX = 3.4e38. It overflows to +inf, meets the
# T = 0 wings of the bandpass, and inf*0 = NaN -- so the WHOLE gradient vector is NaN
# while the forward magnitudes look perfectly reasonable. Nothing in the model is wrong;
# the units are. Measuring the spectrum in AB-zero-point units instead makes the same
# cotangent 10**(0.4 mag)/int T dlam/lam ~ 1e19 at AB 45, i.e. 19 decades of headroom.
# This only became reachable once extinction could push bands past AB ~43.
INV_AB_ZEROPOINT = 1.0 / AB_ZEROPOINT                             # 2.7540622418e+19
PLANCK_T3_AB = PLANCK_T3 / AB_ZEROPOINT                           # 1.1541174374e+04

# Faintest magnitude ab_magnitude will report; USER-TUNABLE per call via mag_floor=.
# It is applied as a floor on the flux RATIO, 10**(-0.4*mag_floor), which makes the
# returned value exactly min(mag, mag_floor) with zero gradient in the capped region.
# Clamping in magnitude space directly does NOT work: log10(0) = -inf is computed first
# and its derivative is inf, so jnp.minimum would return 40.0 with a NaN gradient.
# TWO separate float32 constraints bound how large mag_floor may be:
#  (a) the clamp must be a representable float32. The clamp this replaces -- 1e-300 on
#      the ratio -- is np.float32(1e-300) == 0.0 EXACTLY, so it clamped nothing: the
#      shipped module returned mag = +inf with -inf gradients whenever a band underflowed
#      (temperature floors below ~700 K reach that with no extinction at all).
#  (b) the backward pass carries 1/u with u = 10**(-0.4*mag_floor). In AB-zeropoint units
#      (identity 8) that cotangent is 10**(0.4*mag_floor)/int T dlam/lam.
# mag_floor caps the largest UNCAPPED magnitude, hence the largest 1/u the backward pass
# ever sees, which is what makes it the tunable knob. Measured on the PATCHED denominator
# (300 prior draws, distance shifted so the epochs straddle each floor, forward AND gradient
# checked): float32 is clean through mag_floor = 55 and first fails at 60, where 258/900
# gradient components go non-finite; float64 is clean through 100. The underlying limit is
# the magnitude itself -- f32 gradients survive to AB ~ 58 and f32 VALUES to AB ~ 73.
# The old pre-identity-8 budget rule (mag_floor <= 45) was derived on the cgs denominator
# and no longer applies; it is superseded by the measurement above.
# 40 is ~15 mag below anything any telescope will ever report and leaves 15 mag of margin.
MAG_FLOOR = 40.0

# Same reassociation hazard: (R/dl)**2 can be rewritten as R**2/dl**2, and
# dl**2 ~ 1e52 overflows float32. Pre-scale both into a literal ratio.
RSCALE = 1e14                          # cm, typical photosphere radius
DLSCALE = 1e26                         # cm, ~30 Mpc
RATIO_CONST = (RSCALE / DLSCALE) ** 2

# Luminosities reach ~1e41 erg/s; float32 max is 3.4e38.
LSCALE = 1e40

# GRADIENT-safety scales (identity 5). JAX differentiates x/y w.r.t. y as -x*integer_pow(y,-2),
# lowered as 1/(y*y): in float32 that squaring overflows for |y| > sqrt(FLT_MAX) = 1.84e19 and
# returns exactly 0.0 -- silently, with the forward pass unaffected. Two divisors in this model
# cross that cliff, both functions of vej alone (BETA*v0*c ~ 2.5e21..8.6e21 and
# 4*pi*sigma_sb*r_free**2 ~ 1.2e23..2.1e30), which is why d/dvej was wrong (and sign-flipped)
# while d/dmej and d/dkappa -- which enter through NUMERATORS, whose VJP is a single division --
# stayed correct to six digits. Folding the constants keeps every differentiated divisor O(1).
TDIFF_CONST = 2.0 * SOLAR_MASS / (BETA * SPEED_OF_LIGHT ** 2)     # 3.2297873782e+11
T_CONST = (LSCALE / (4.0 * np.pi * SIGMA_SB)) ** 0.25             # 6.1206084825e+10
# identity 5c, needed once temperature_floor becomes fittable:
#     (L/(4 pi sigma Tf^4))^0.5 * LSCALE^0.5  ==  R_FLOOR_CONST * sqrt(L) / Tf^2
# Exactly the same value. The point is the backward pass: in the original grouping the
# DIFFERENTIATED divisor is 4*pi*sigma*Tf^4, which reaches 9.2e11 at Tf = 6000 K and is squared
# to 8.5e23 by the VJP of x/y. That still fits float32, but only by ~14 orders of magnitude, and
# it forces the quotient L/(...) down to 1e-42 -- SUBNORMAL in float32 -- for small L. After the
# regrouping the differentiated divisor is Tf^2 <= 3.6e7 and nothing is subnormal.
R_FLOOR_CONST = (LSCALE / (4.0 * np.pi * SIGMA_SB)) ** 0.5        # 3.7461848196e+21

# --- diffusion quadrature (identity 6) --------------------------------------
# Substituting s = (t^2 - t'^2)/td^2 (so t' dt' = -(td^2/2) ds) turns
#     L(t) = (1/td) int_0^t f(t') (t'/td) exp(-(t^2-t'^2)/td^2) dt'
# into
#     L(t) = 1/2 int_0^{t^2/td^2} f(t'(s)) exp(-s) ds,   t'(s) = sqrt(t^2 - s td^2)
# which is exact in the continuum. The point is the DISCRETISATION: the original kernel has
# width td^2/(2t) in t', which SHRINKS as t grows while a geomspace grid's spacing GROWS, so
# the trapezoid rule silently over-estimates once t >> td (measured: 8.6x too high at 20 d for
# td = 1.47 d, and redback has the identical defect). After the substitution the kernel is
# exp(-s) on a FIXED O(1)-wide domain, so a fixed node count resolves it for ANY td.
# exp(-S_MAX) = 4e-18 is below float32 resolution, so truncating there is exact.
# The integral is split at t' = SPLIT_FRAC * t because its two ends are hard in opposite ways:
#   OUTER (t' near t): the kernel is narrow -- width td^2/(2t) -- so it needs the s-substitution.
#   INNER (t' -> 0):   the heating rate diverges as t'^-1.3 (radioactive decay), making the
#                      integrand ~ u^-0.3. That is integrable but is an endpoint singularity, and
#                      Gauss-Legendre converges only algebraically on it (measured: 2.5e-2 error
#                      at t = 0.5 d, with 2.8% of the mass in u < 1e-4). Integrating in log t'
#                      turns the power law into a smooth exponential, which GL nails -- this is
#                      exactly why the original geomspace grid was accurate at early times.
# Handling both is what makes the scheme uniformly accurate; either alone fails at one end.
S_MAX = 40.0            # exp(-40) = 4e-18, below float32 resolution -> truncation is exact
SPLIT_FRAC = 0.5
# Lower cutoff of the diffusion integral, seconds. A CONSTANT, deliberately not grid_s[0]:
# tying it to the caller's output grid means passing observation times directly as `grid_s`
# silently truncates the integral instead of failing. The neglected [0, T_FLOOR) tail is
# ~ (T_FLOOR/t)^0.7 (the integrand goes as t'^-0.3 there) = 4e-8 relative at t = 1 d.
T_FLOOR = 1e-3
# Smallest source-frame evaluation time, seconds. Times at or below this are treated as
# PRE-EXPLOSION and return identically zero flux. Set equal to T_FLOOR: the diffusion
# integral's own lower cutoff, so an evaluation time below it would ask for an answer on a
# domain the quadrature does not cover. Flux at t = T_FLOOR is ~1e-45 mJy (see `bolometric`),
# so the discontinuity introduced at the mask edge is numerically exactly zero.
T_EVAL_MIN = T_FLOOR
# The nodes, the Barnes-Kasen table and the F99 spline below are float64 NumPy, cast by `_const`
# where they are used: a jnp array built here takes the dtype in force at IMPORT, and a module
# imported before x64 is enabled froze them in float32 (up to 1.9e-6 relative in L_bol).


def _const(x):
    """A float64 NumPy constant as a jnp array of the session's float dtype, at trace time.

    The dtype is explicit: a bare ``jnp.asarray`` under ``jit`` can hand back a float64 constant
    JAX cached while x64 was on, which a float32 session then truncates with a UserWarning.
    """
    return jnp.asarray(x, dtype=jnp.result_type(float))


_GLX_O, _GLW_O = np.polynomial.legendre.leggauss(128)      # outer, in s
GL_X = 0.5 * (_GLX_O + 1.0)
GL_W = 0.5 * _GLW_O
# Inner order 256, not 96: the residual quadrature error is entirely this panel at
# t = 0.10-0.30 d, because arctan(0.11/(t'-1.3)) has a branch point 0.0845 off the real axis
# inside a z-interval of half-length ~7.8. Measured max |dL/L|: 96 -> 2.00e-3, 192 -> 1.37e-4,
# 256 -> 2.57e-5 (0.03 mmag). Free on a single curve; ~15% at batch 1000, roughly cancelled by
# the searchsorted removal above. Raising the OUTER order does nothing (128 == 512).
_GLX_I, _GLW_I = np.polynomial.legendre.leggauss(256)      # inner, in log t'
GL_XI = 0.5 * (_GLX_I + 1.0)
GL_WI = 0.5 * _GLW_I

# --- diffusion normalisation (redback's, and the standard Arnett one) --------------------
# redback's kilonova writes the diffusion integral with a prefactor of 1/td^2:
#
#     L(t) = exp(-t^2/td^2)/td * cumtrapz[ f(t') (t'/td) exp(t'^2/td^2) ]      (redback)
#
# Arnett (1982), Chatzopoulos+2012 eq. 3 and Villar+2017 all carry 2/td^2. The factor matters
# and it is not subtle: substituting u = (t^2 - t'^2)/td^2 gives
#
#     int_0^inf (t'/td^2) exp(-(t^2 - t'^2)/td^2) dt'  =  1/2
#
# so redback's diffusion kernel integrates to 1/2 rather than 1, and at late times -- where a
# slowly varying heating rate means all deposited energy escapes and L must approach L_in --
# it converges to L_in/2 instead. Measured on this module at t = 100 t_diff: **L/L_in =
# 0.500055**. Against a direct evaluation of the standard Arnett integral at t = 0.3, 0.5, 1,
# 3 and 10 t_diff for two parameter sets: **0.499994 - 0.500000**. The observable consequence
# is a uniform 2.5*log10(2) = **0.7526 mag** too faint in bolometric, T low by 2^0.25 on the
# hot branch and R_phot low by sqrt(2) on the floored branch, which biases every inferred mej
# and kappa. It is not a corner case and no prior excludes it.
#
# THE DEFAULT IS REDBACK'S, BY PROJECT RULE: these ports exist so a redback-vs-whisper_cbpf
# (JAX port) comparison is apples to apples, and correcting the physics unilaterally would
# break parity with every kilonova fit already run. `arnett_prefactor=2.0` selects the standard
# Arnett normalisation for anyone who wants it, and is threaded through every entry point
# including the N-component model and the whisper factory.
#
# The switch is a PYTHON float compared at TRACE time, not a traced multiply, so the default
# path compiles to byte-identical HLO -- the same discipline the extinction terms use with
# their None sentinel.
ARNETT_PREFACTOR = 1.0            # 1.0 = redback; 2.0 = Arnett 1982 / Villar+2017

# --- Barnes & Kasen thermalisation table -----------------------------------
# Verbatim from redback/utils.py:1176-1186. Grid is (mass, velocity) = (5, 4).
_BK_MEJ = np.array([1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2, 1.0e-1])
_BK_VEJ = np.array([0.1, 0.2, 0.3, 0.4])
_BK_A = np.array([[2.01, 4.52, 8.16, 16.3], [0.81, 1.90, 3.20, 5.00],
                  [0.56, 1.31, 2.19, 3.00], [0.27, 0.55, 0.95, 2.00],
                  [0.20, 0.39, 0.65, 0.90]])
_BK_B = np.array([[0.28, 0.62, 1.19, 2.40], [0.19, 0.28, 0.45, 0.65],
                  [0.17, 0.21, 0.31, 0.45], [0.10, 0.13, 0.15, 0.17],
                  [0.06, 0.11, 0.12, 0.12]])
_BK_D = np.array([[1.12, 1.39, 1.52, 1.65], [0.86, 1.21, 1.39, 1.50],
                  [0.74, 1.13, 1.32, 1.40], [0.60, 0.90, 1.13, 1.25],
                  [0.63, 0.79, 1.04, 1.50]])


def _bilinear_extrap(table, xg, yg, x, y):
    """Bilinear interpolation with LINEAR EXTRAPOLATION outside the grid.

    Matches scipy RegularGridInterpolator(bounds_error=False, fill_value=None),
    which redback uses. It extrapolates rather than clamping -- kilonova priors
    routinely exceed vej = 0.4c, and clamping there would change the physics.
    """
    # Comparison count rather than jnp.searchsorted: identical result (searchsorted with
    # side='left' IS the count of grid points strictly below x), but searchsorted lowers to a
    # sequential scan that XLA swallows into this model's big reduce-fusion, making ptxas time
    # explode with the vmap dimension -- measured 65.7 s -> 0.79 s compile at batch 1000, and
    # 10% faster execution. Forward values and gradients are BITWISE identical.
    # The comparison must be STRICT (x > xg): with >= the value is unchanged everywhere but the
    # one-sided derivative flips exactly AT a node, and the canonical test point
    # (mej=0.01, vej=0.2) sits on nodes of both grids.
    i = jnp.clip(jnp.sum(x > xg) - 1, 0, xg.size - 2)
    j = jnp.clip(jnp.sum(y > yg) - 1, 0, yg.size - 2)
    tx = (x - xg[i]) / (xg[i + 1] - xg[i])      # may be <0 or >1: extrapolates
    ty = (y - yg[j]) / (yg[j + 1] - yg[j])
    return ((1 - tx) * (1 - ty) * table[i, j]
            + tx * (1 - ty) * table[i + 1, j]
            + (1 - tx) * ty * table[i, j + 1]
            + tx * ty * table[i + 1, j + 1])


def thermalisation_coeffs(mej, vej):
    """Barnes+2016 (a, b, d). redback/utils.py:1168."""
    xg, yg = _const(_BK_MEJ), _const(_BK_VEJ)
    return tuple(_bilinear_extrap(_const(table), xg, yg, mej, vej)
                 for table in (_BK_A, _BK_B, _BK_D))


def _inverse_expm1(arg):
    """1/(exp(x)-1), stable for large x. Mirrors redback/sed.py:10."""
    small = 1.0 / jnp.expm1(jnp.minimum(arg, 50.0))
    e = jnp.exp(-jnp.maximum(arg, 0.0))
    large = e / (1.0 - e)
    return jnp.where(arg > 50.0, large, small)


# --- setup-time construction (NumPy, run once per dataset) -----------------

def source_time_s(t_obs_days, redshift, t_exp_days=0.0):
    """Observer-frame days -> source-frame seconds SINCE EXPLOSION.

        t_src = (t_obs - t_exp) * 86400 / (1 + z)

    This replaces build_time_grid. It is a pure affine map, so t_exp_days may be a
    traced JAX scalar and is differentiable end to end (dt_src/dt_exp = -86400/(1+z),
    a constant). Entries with t_src <= T_EVAL_MIN are pre-explosion; the model returns
    zero flux there (see `bolometric`). Negative values are allowed and are safe --
    they never reach a log or a fractional power.

    t_exp_days is an OBSERVER-frame epoch on the same clock as t_obs_days (typically MJD
    if t_obs_days is MJD, or days-since-trigger if t_obs_days is that). Do not redshift it
    yourself; the (1+z) below does it.
    """
    return (jnp.asarray(t_obs_days) - t_exp_days) * DAY_TO_S / (1.0 + redshift)


def pre_explosion(t_src_s):
    """Boolean mask, True where the model returns identically zero flux.

    Provided so a likelihood can route these epochs to an upper-limit / censored term
    instead of a Gaussian residual against a floor magnitude.
    """
    return jnp.asarray(t_src_s) <= T_EVAL_MIN


def build_time_grid(t_obs_days, redshift, t_min=1e-2, n_dense=300):
    """DEPRECATED. Kept only so old call sites keep running.

    Since identity 6 the diffusion integral generates its own Gauss-Legendre nodes from the
    evaluation time alone, so `grid_s` was never read except as `grid_s[row]`: verified
    bit-identical output (and gradients) for grids of 4 to 5004 points, for t_min in
    {1e-6, 1e-2, 1e3}, and with NaN/inf stuffed into every unselected entry, in both
    precisions. Use source_time_s() instead -- it is the same numbers with the dead
    indirection removed, and it is what makes t_exp fittable.
    """
    t_src = np.asarray(t_obs_days, dtype=np.float64) * DAY_TO_S / (1.0 + redshift)
    if np.any(t_src <= 0):
        raise ValueError("observation times must be > 0 in the source frame")
    dense = np.geomspace(t_min, t_src.max(), n_dense)
    grid = np.unique(np.concatenate([dense, t_src]))
    row = np.searchsorted(grid, t_src)
    assert np.allclose(grid[row], t_src)
    return jnp.asarray(grid), jnp.asarray(row)


def redback_time_grid(t_src_s, t_min=1e-2, t_max=7e6, dense_resolution=500):
    """The source-frame grid [s] redback 1.20's ``one_component_kilonova_model`` solves on. NumPy.

    Transcribes ``get_optimal_time_array(1e-2, 7e6, dense_resolution, user_times=t_src_s)``
    (``redback/utils.py:114``, the ``user_times`` branch): 15% of the nodes geometric below the
    epochs, 70% geometric from half the first epoch to twice the last, 15% above, deduplicated.
    It depends on the epochs only through their minimum and maximum. Pass it to
    :func:`bolometric` (and everything that calls it) as ``time_grid=`` to reproduce redback's
    discretisation instead of the converged quadrature (identity 6); see ``time_grid`` there
    for what that costs at late times.

    Epochs past ``t_max`` (81 d at z = 0) are outside redback's grid -- redback raises there --
    and this warns: the model then holds the last node's photosphere. Epochs at or before the
    explosion enter as ``T_EVAL_MIN``: the model masks them anyway, the grid is the one redback
    builds from the same raw epochs, and a light curve with none still gets a finite grid (redback
    has no answer). Any such epoch is the earliest, so it sets the lower edge and moves every
    node; the factories therefore clip days at the CPU adapter's ``MIN_TIME_DAY`` before calling
    this, as the adapter does before calling redback (``_factories._kilonova_time_grid``).
    """
    t = np.maximum(np.asarray(t_src_s, dtype=np.float64), T_EVAL_MIN)
    eval_min = max(t_min, float(t.min()) * 0.5)
    eval_max = min(t_max, float(t.max()) * 2.0)
    n_before = int(dense_resolution * 0.15)
    n_user = int(dense_resolution * 0.70)
    n_after = dense_resolution - n_before - n_user
    if eval_min > t_min:
        before = np.geomspace(t_min, eval_min, n_before)
    else:
        before = np.array([t_min])
        n_user += n_before
    user = np.geomspace(eval_min, eval_max, n_user)
    if eval_max < t_max:
        after = np.geomspace(eval_max, t_max, n_after)
    else:
        after = np.array([t_max])
        n_user += n_after
        user = np.geomspace(eval_min, eval_max, n_user)
    if t.max() > t_max:
        warnings.warn(
            f"{int((t > t_max).sum())} epochs lie past redback's kilonova grid (t_src > {t_max:g} "
            f"s); redback raises there, and the model holds the last node's photosphere.",
            RuntimeWarning, stacklevel=2)
    return np.unique(np.concatenate([before, user, after]))


# --- the grid-rule band integral: moved VERBATIM to whisper_cbpf.synphot.grid_rule ----------
# (whisper 0.1.1: one photometry module for the CPU and GPU paths). Re-exported here so
# `kilonova.make_filter_set` / `kilonova.ab_weights` -- and `tde.` / `supernova.`, which re-export
# them from this module -- keep working and keep reproducing every stored filter set bit for bit.
from ...synphot.grid_rule import _cell_widths, ab_weights, make_filter_set  # noqa: E402,F401


# --- interstellar extinction ------------------------------------------------
# Both laws below return XI(lam) = A(lam)/E(B-V), the SHAPE of the extinction curve.
# At fixed R_V the attenuation is EXACTLY LINEAR IN E(B-V):
#       F_obs(lam) = F(lam) * 10**(-0.4 * E(B-V) * XI(lam))
# so the wavelength-dependent part can be precomputed once and only the scalar E(B-V)
# needs to live inside the model. That is what makes the sampled-E(B-V) path cost one
# exp() per grid point and the fixed-E(B-V) path cost exactly nothing (fold it into the
# AB weights, which are already precomputed -- see extincted_weights).
#
# LAW CHOICE: Fitzpatrick (1999) is the default.
#   * it is redback's default for BOTH the MW and the host term
#     (extinction_models._perform_extinction, mw_law=host_law='fitzpatrick99'), so the
#     cross-check against redback is like-for-like;
#   * it is defined from 910 A to 6 um, covering the whole 1000-30000 A filter grid;
#     CCM89's claimed validity stops at 1250 A / 3.3 um, i.e. INSIDE that grid;
#   * F99 is the law the SFD / Schlafly-Finkbeiner E(B-V) maps are calibrated against,
#     which is where an MW E(B-V) prior actually comes from.
# CCM89 is provided too (it is four polynomials, so it costs ~40 lines) for compatibility
# with older analyses; it is NOT the recommended default.
#
# FRAMES (the classic sign-of-the-redshift error):
#   MW dust sits between us and everything, so it attenuates at the OBSERVED wavelength.
#   Host dust sits at the source, so it attenuates at the REST wavelength lam_obs/(1+z).
# Hence xi_mw = XI(lam_obs) but xi_host = XI(lam_obs/(1+z)). At z = 0.5 the two differ by
# more than a magnitude per unit E(B-V) in the blue -- see test_extinction_frames.

_LN10_04 = 0.4 * np.log(10.0)          # 0.9210340371976184, A_mag -> e-folds

# F99 spline anchors in inverse microns. These are the updated E. Fitzpatrick optical
# points used by IDL astrolib FM_UNRED and by the `extinction` package -- NOT F99 Table 4.
_F99_XK = np.array([0.0, 1e4 / 26500., 1e4 / 12200., 1e4 / 6000., 1e4 / 5470.,
                    1e4 / 4670., 1e4 / 4110., 1e4 / 2700., 1e4 / 2600.])
_F99_XSPLIT = float(_F99_XK[7])        # 3.7037 invum = 2700 A: FM90 above, spline below


def _natural_spline_matrix(xk):
    """Constant linear map (knot values) -> (knot second derivatives), natural cubic spline.

    The knot ABSCISSAE are fixed constants, so M = S y with S = A^-1 B a fixed 9x9 matrix.
    Precomputing S turns the spline construction into one small matvec that is exactly
    differentiable w.r.t. the knot values -- hence w.r.t. R_V, which is the only thing they
    depend on. Same discipline as the Gauss-Legendre nodes above: solve once in NumPy.
    """
    n = xk.size
    h = np.diff(xk)
    amat = np.zeros((n, n))
    bmat = np.zeros((n, n))
    amat[0, 0] = 1.0                                     # natural BC: M_0 = 0
    amat[-1, -1] = 1.0                                   # natural BC: M_{n-1} = 0
    for i in range(1, n - 1):
        amat[i, i - 1] = h[i - 1]
        amat[i, i] = 2.0 * (h[i - 1] + h[i])
        amat[i, i + 1] = h[i]
        bmat[i, i - 1] = 6.0 / h[i - 1]
        bmat[i, i] = -6.0 * (1.0 / h[i - 1] + 1.0 / h[i])
        bmat[i, i + 1] = 6.0 / h[i]
    return np.linalg.solve(amat, bmat)


_F99_S = _natural_spline_matrix(_F99_XK)                  # (9, 9), float64 NumPy (see GL_X)
_F99_H = np.diff(_F99_XK)


def _f99_uv_k(x, r_v):
    """FM90 UV parametrisation, k(x) = A(x)/E(B-V) - R_V, with the F99 coefficients.

    x0 = 4.596 and gamma = 0.99 are folded as x0**2 = 21.123216 and gamma**2 = 0.9801 so
    the Lorentzian denominator is one fused expression; it has a floor of
    (x0**2)**2 = 446 at x = 0 and is never small, so no branch is needed.
    """
    c2 = -0.824 + 4.717 / r_v
    c1 = 2.030 - 3.007 * c2
    d = x * x / ((x * x - 21.123216) ** 2 + 0.9801 * x * x)
    y = jnp.maximum(x - 5.9, 0.0)                        # F(x), zero below 5.9 by definition
    return c1 + c2 * x + 3.23 * d + 0.41 * (0.5392 * y ** 2 + 0.05644 * y ** 3)


def _f99_knot_k(r_v):
    """k = A/E(B-V) - R_V at the nine F99 anchors.

    The published anchors are A/E(B-V); subtracting R_V costs precision, because e.g. anchor 3
    is 1.00270*r_v - r_v -- two numbers agreeing to 3 digits, leaving a result of size 0.008
    with an absolute error of ulp(3.1) = 2.4e-7 in float32, i.e. ~3e-5 relative ON THAT TERM.
    Folding the subtraction into the LITERAL (1.00270 - 1 = 0.00270) removes it exactly and
    for free, the same trick as identity 1 in the heating rate. Measured effect on the anchor:
    8.3e-8 absolute, 2.0e-7 relative -- small, but there is no reason to carry it.
    """
    return jnp.stack([
        -r_v,
        -0.9146161290322581 * r_v,                       # 0.26469/3.1 - 1
        -0.7325 * r_v,                                   # 0.82925/3.1 - 1 (exact)
        -0.422809 + 0.00270 * r_v + 2.13572e-4 * r_v ** 2,
        -5.13540e-2 + 0.00216 * r_v - 7.35778e-5 * r_v ** 2,
        0.700127 + 0.00184 * r_v - 3.32598e-5 * r_v ** 2,
        (1.19456 + 0.01707 * r_v - 5.46959e-3 * r_v ** 2
         + 7.97809e-4 * r_v ** 3 - 4.45636e-5 * r_v ** 4),
        _f99_uv_k(_F99_XK[7], r_v),                      # continuity with the FM90 branch
        _f99_uv_k(_F99_XK[8], r_v),
    ])


def f99_a_over_ebv(lam_aa, r_v=3.1):
    """Fitzpatrick (1999) A(lambda)/E(B-V). JAX-native, differentiable in r_v.

    Two branches, split at 2700 A: the FM90 analytic form BELOW it, a NATURAL cubic spline
    through the nine anchors ABOVE it. That is exactly what `extinction` 0.4.7 and IDL
    astrolib FM_UNRED do, and it reproduces `extinction.fitzpatrick99(lam, r_v*ebv, r_v)/ebv`
    to 3.6e-15 mag in float64 / 6.1e-6 mag in float32 over 1000-30000 A and R_V = 2.0-5.5.
    `extinction` is a compiled C extension with no JVP rule, so it can only ever be the
    reference, never the model.
    """
    x = 1e4 / jnp.asarray(lam_aa)                        # inverse microns
    kk = _f99_knot_k(r_v)
    mm = _const(_F99_S) @ kk                             # second derivatives at the anchors
    xs = jnp.minimum(x, _F99_XSPLIT)                     # keep the spline branch in range
    # Comparison count, not jnp.searchsorted -- the same substitution as in _bilinear_extrap
    # and for the same reason: searchsorted lowers to a sequential scan whose ptxas time
    # explodes under vmap. Here it is a (n_wave, 9) boolean reduction, which is free.
    # Exactly identical: searchsorted(side='left') IS the count of knots strictly below xs.
    # This branch only runs when R_V is SAMPLED (otherwise pass a precomputed xi_*), but that
    # is precisely the case that gets vmapped over a batch of parameter draws.
    xk = _const(_F99_XK)
    i = jnp.clip(jnp.sum(xs[..., None] > xk, axis=-1) - 1, 0, _F99_XK.size - 2)
    h = _const(_F99_H)[i]
    a = (xk[i + 1] - xs) / h
    b = 1.0 - a
    k_spl = (a * kk[i] + b * kk[i + 1]
             + ((a ** 3 - a) * mm[i] + (b ** 3 - b) * mm[i + 1]) * (h * h / 6.0))
    k_uv = _f99_uv_k(jnp.maximum(x, _F99_XSPLIT), r_v)   # both branches finite everywhere
    return jnp.where(x < _F99_XSPLIT, k_spl, k_uv) + r_v


def ccm89_a_over_ebv(lam_aa, r_v=3.1):
    """Cardelli, Clayton & Mathis (1989) A(lambda)/E(B-V) = R_V*a(x) + b(x).

    Provided for compatibility; F99 is the recommended default (see the block comment).
    Claimed validity 0.3 <= x <= 10 invum, i.e. 1000-33333 A.
    """
    x = 1e4 / jnp.asarray(lam_aa)

    y = x - 1.82                                         # optical/NIR expansion point
    a_opt = (1. + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4
             + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7)
    b_opt = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4
             - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7)

    xir = jnp.maximum(x, 1e-6) ** 1.61                   # x**1.61 has an infinite slope at 0
    a_ir, b_ir = 0.574 * xir, -0.527 * xir

    yu = jnp.maximum(x - 5.9, 0.0)
    fa = -0.04473 * yu ** 2 - 0.009779 * yu ** 3
    fb = 0.2130 * yu ** 2 + 0.1207 * yu ** 3
    a_uv = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) + fa
    b_uv = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + fb

    yf = x - 8.0
    a_fuv = -1.073 - 0.628 * yf + 0.137 * yf ** 2 - 0.070 * yf ** 3
    b_fuv = 13.670 + 4.257 * yf - 0.420 * yf ** 2 + 0.374 * yf ** 3

    # Every branch above is a polynomial or a Lorentzian with a positive floor, so all four
    # are finite at every x and the nested where cannot leak a NaN through the unused side.
    a = jnp.where(x < 1.1, a_ir, jnp.where(x < 3.3, a_opt,
                  jnp.where(x < 8.0, a_uv, a_fuv)))
    b = jnp.where(x < 1.1, b_ir, jnp.where(x < 3.3, b_opt,
                  jnp.where(x < 8.0, b_uv, b_fuv)))
    return r_v * a + b


_EXT_LAWS = {'f99': f99_a_over_ebv, 'ccm89': ccm89_a_over_ebv}


def extinction_shape(lam_obs_ang, redshift=0.0, r_v=3.1, law='f99', frame='observer'):
    """A(lambda)/E(B-V) evaluated in the requested frame, on the OBSERVER wavelength grid.

    frame='observer' (Milky Way): the law is evaluated at lam_obs.
    frame='rest'     (host):      the law is evaluated at lam_obs/(1+z).
    The returned array is always indexed by the observer-frame grid, so it multiplies the
    observed spectrum element-wise either way -- the redshift only moves the SAMPLING POINT.
    """
    lam = jnp.asarray(lam_obs_ang)
    if frame == 'rest':
        lam = lam / (1.0 + redshift)
    elif frame != 'observer':
        raise ValueError("frame must be 'observer' or 'rest'")
    return _EXT_LAWS[law](lam, r_v)


def _ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host,
                      xi_mw, xi_host, r_v_mw, r_v_host, law):
    """10**(-0.4 A_tot(lam)) on the observer grid, or None if no extinction was requested.

    Returning None (rather than a vector of ones) is deliberate: it lets the caller drop the
    multiply at TRACE time, so an un-extincted call compiles to exactly the HLO it compiled to
    before extinction existed. Backward compatibility is then a property of the graph, not a
    promise.
    """
    tau = None
    if ebv_mw is not None:
        xi = _EXT_LAWS[law](lam_obs_ang, r_v_mw) if xi_mw is None else xi_mw
        tau = ebv_mw * xi
    if ebv_host is not None:
        if xi_host is None:
            xi = _EXT_LAWS[law](jnp.asarray(lam_obs_ang) / (1.0 + redshift), r_v_host)
        else:
            xi = xi_host
        tau = ebv_host * xi if tau is None else tau + ebv_host * xi
    if tau is None:
        return None
    # exp(-0.4 ln10 A) instead of 10**(-0.4 A): one exp, no pow, and the constant is folded
    # so nothing intermediate can overflow. The exponent is <= 0 by construction (E(B-V) >= 0,
    # XI > 0 over 1000-30000 A), so this underflows smoothly to 0 rather than overflowing.
    return jnp.exp(-_LN10_04 * tau)


def extincted_weights(weights, lam_obs_ang, redshift=0.0, ebv_mw=0.0, ebv_host=0.0,
                      r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """Fold a FIXED extinction into the precomputed AB weights. Zero model-runtime cost.

    >>> W, N = ab_weights(lam, trans)
    >>> W = extincted_weights(W, lam, redshift=z, ebv_mw=0.02)   # setup, once
    >>> mag = ab_magnitude_jit(t_src, bidx, W, N, lam, z, dl, mej, vej, kappa)

    ONLY the weights are reddened, never `norms`: norms is the AB zero-point integral
    3631 Jy * int T dlam/lam, a property of the FILTER. Reddening it too would divide the
    extinction back out and return the unextincted magnitude.
    """
    tau = _ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host,
                            None, None, r_v_mw, r_v_host, law)
    return weights if tau is None else jnp.asarray(weights) * tau[None, :]


# --- backward compatibility with the retired (grid_s, row) calling convention ---
# `bolometric`, `flux_density_mjy` and `ab_magnitude` also accept the older (grid_s, row) form and
# evaluate at grid_s[row]. Since identity 6 that indirection is dead weight (grid_s was
# never read any other way), and it is also what blocked a fittable t_exp, so the times
# are now passed directly. The old form still works: the shim below detects it and does
# the one thing the old code did, grid_s[row].

def _is_row_index(grid, x):
    """True only for something that can be a `row` index INTO `grid`, never a physical argument.

    `row` is the ONLY integer ARRAY the old signatures ever took at the probed position, and
    every FLOATING alternative there (mej, weights) is excluded by dtype alone. But dtype is
    NOT sufficient: nu_obs_hz may legitimately be an integer array, in which case
    flux_density_mjy(t_src, nu_int, z, dl, mej, vej, kappa, Tf) -- eight positional args --
    would be silently reinterpreted as the legacy (grid_s, row, nu, z, dl, mej, vej, kappa)
    and return a finite answer wrong by ~1e177. So also require that `grid` is a real array
    and that every value of x is a valid index into it, which frequencies in Hz (~1e14) never
    are. Under jit the values are traced and cannot be inspected -- shapes and dtypes are
    static, values are not -- so the bound check is skipped there and the dtype+rank test
    stands alone; pass times as floats (they always are in practice) and the ambiguity
    cannot arise.
    """
    if not (hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.integer)
            and jnp.ndim(x) > 0):
        return False
    if not (hasattr(grid, "__len__") or (hasattr(grid, "ndim") and jnp.ndim(grid) > 0)):
        return False                                  # legacy form needs an indexable grid_s
    try:                                              # concrete values only; traced -> skip
        n = int(jnp.shape(grid)[0])
        xv = np.asarray(x)
        return bool(xv.min() >= 0 and xv.max() < n)
    except Exception:
        return True                                   # traced: fall back to the dtype test


def _grid_row_shim(fn, probe):
    """Wrap fn so it ALSO accepts the retired (grid_s, row, ...) positional form.

    `probe` is the positional index that disambiguates the two conventions:
      bolometric / flux_density_mjy : 1  (old row vs new mej / nu_obs_hz)
      ab_magnitude                  : 2  (old band_idx vs new weights) -- position 1 is an
                                         integer array in BOTH conventions, so it cannot
                                         be the probe.
    A DeprecationWarning IS raised, once per call site. The detection is a heuristic on
    dtype/rank/bounds, and a heuristic that silently rewrites your arguments is exactly the
    kind of thing that produced the other six bugs in this file: an integer-typed physical
    argument at the probe position gets reinterpreted, and the result is finite and wrong
    rather than an error. Warning once makes the rewrite visible without spamming a sampler
    loop. (Known residual: with x64 DISABLED, an integer frequency array is truncated to
    int32 on the way in -- 5e14 does not fit -- so its values are already meaningless before
    this check runs and may land inside the valid index range. Pass frequencies as floats.)
    """
    def wrapper(*args, **kwargs):
        if len(args) > probe and _is_row_index(args[0], args[probe]):
            warnings.warn(
                f"{fn.__name__}: the (grid_s, row, ...) calling convention is deprecated; "
                f"pass source-frame times directly, e.g. source_time_s(t_obs, z, t_exp). "
                f"Interpreting argument {probe} as a row index into argument 0.",
                DeprecationWarning, stacklevel=2)
            args = (args[0][args[1]],) + args[2:]
        return fn(*args, **kwargs)
    # Deliberately NOT functools.wraps: it sets __wrapped__, which makes inspect.signature
    # (and hence jax.jit's static_argnames resolution) report the WRAPPED signature while
    # the wrapper is actually variadic.
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__doc__ = fn.__doc__
    wrapper.__module__ = fn.__module__
    return wrapper


# --- core physics ----------------------------------------------------------

# Below this, log1p(x)/x is evaluated by its Taylor series (identity 7). The switch point is
# a two-sided trade-off, measured: BELOW it the naive quotient's VJP loses significance (it
# is a difference of two ~1/x terms), ABOVE it the 4-term series truncates. At x = 1e-2 the
# series derivative is off by 1.6e-6 relative while the naive FLOAT32 derivative is off by
# 1.4e-5 -- the series is the better of the two by 10x -- and the two curves cross at
# x ~ 2e-2. Anything in [1e-3, 2e-2] is defensible; 1e-2 minimises the larger of the two.
BT_SMALL = 1e-2


def _log1p_over_x(x):
    """log1p(x)/x, evaluated so that the DERIVATIVE survives small x. Identity 7.

    The naive form differentiates to  1/(x(1+x)) - log1p(x)/x**2, and JAX lowers the second
    term so that the incoming cotangent is divided by x**2. In `_heating` that cotangent is
    lum_in*0.36 ~ 2.9e11*mej, so the quotient overflows float32 for

        x < sqrt(0.36*4e18*mej*M_sun/LSCALE / FLT_MAX)   = 2.9e-16 (mej=1e-4)
                                                         ... 9.2e-15 (mej=0.1)

    x = 2 b t^d reaches ~1e-17 at the inner quadrature's first node (t' = T_FLOOR = 1 ms)
    whenever the extrapolated Barnes exponent d exceeds ~1.8, i.e. at high mej AND high vej.
    The two overflowing paths then have OPPOSITE signs -- d/dbv gets -inf, d/ddv gets +inf,
    because d(x)/d(dv) = x*log(tdays) < 0 -- and -inf + inf = NaN. That NaN reaches
    d(mag)/d(everything) and kills the whole gradient, silently, on ~19% of the prior box.

    A jnp.maximum(x, floor) does NOT fix this: the floor would have to sit above 1e-14, where
    it would distort a value the model actually uses. Worse, the naive VJP is badly wrong long
    before it overflows -- it is a difference of two ~1/x terms, so it loses all significance
    (in FLOAT64 it returns -128 instead of -0.5 at x = 1e-18). The series has no cancellation
    and no division, and is exact in the limit:  log1p(x)/x -> 1,  d/dx -> -1/2.

    Series: 1 - x/2 + x^2/3 - x^3/4 (Horner). Truncation is x^4/5 = 2.0e-9 relative at the
    switch point, ~60x below float32 eps; the derivative there is 1.6e-6 relative, against
    1.4e-5 for the naive float32 quotient it replaces.

    COST: in float64 this shifts L(t) by up to 7.8e-11 relative against the pre-series module
    (the series truncation at the inner quadrature's early nodes); in float32 187/12000 L
    entries differ, by at most 2.0e-7 relative (~1 ulp). That is the whole regression
    footprint of identity 7.

    The branch test is on |x|, NOT x. A one-sided `x < BT_SMALL` would also route large
    NEGATIVE x into the series, where 1 - x/2 + x^2/3 - x^3/4 diverges instead of clamping --
    and it stays FINITE while doing so, so no NaN sweep would ever catch it. x = 2 b t^d goes
    negative wherever the Barnes-Kasen table is EXTRAPOLATED far enough for b to change sign,
    which starts at mej ~ 0.155 (vej = 0.7) / 0.22 (vej = 0.4) / 0.375 (vej = 0.2). That is
    outside this model's mej <= 0.1 prior (where the effect is 5.7e-11, invisible), but it is
    one prior-widening away from silently corrupting L by a factor 6e18.
    """
    small = jnp.abs(x) < BT_SMALL
    x_big = jnp.where(small, jnp.asarray(BT_SMALL, dtype=x.dtype), x)   # double-where
    series = 1.0 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x))
    return jnp.where(small, series, jnp.log1p(x_big) / x_big)


def _heating(t_s, m0, av, bv, dv):
    """Radioactive heating rate lum_in * e_th at source-frame times t_s (seconds).

    Returned in LSCALE units. Shape-agnostic: t_s may be any shape.
    """
    # heating rate, cancellation-free (identity 1) -- BUT ONLY VALID FOR t > t0.
    # arctan(a) + arctan(1/a) = +pi/2 for a > 0 and -pi/2 for a < 0, so
    # 0.5 - arctan(x/sig)/pi == arctan(sig/x)/pi holds only on x > 0. Clamping dt to 1e-30
    # does not dodge the invalid region: it maps EVERY node with t < t0 to
    # arctan(0.11/1e-30)/pi = 0.5 exactly, freezing the heating factor at 0.5**1.3 = 0.406
    # where redback rises to 0.965. Below t0 the original form has no cancellation to remove
    # (it lies in (0.5, 1]), so use it as-is there.
    dt = t_s - 1.3                                       # t0 = 1.3 s
    dt_safe = jnp.where(dt > 0.0, dt, 1.0)               # keep the unselected branch finite
    frac = jnp.where(dt > 0.0,
                     jnp.arctan(0.11 / dt_safe) / jnp.pi,
                     0.5 - jnp.arctan(dt / 0.11) / jnp.pi)
    # Scale FIRST: 4e18 * m0 is ~8e49 and overflows float32 before the divide.
    lum_in = (4.0e18 / LSCALE) * m0 * frac ** 1.3

    # Barnes+2016 thermalisation. No floor on bt: identity 7 removes the 1/bt**2 term from
    # the backward pass entirely, so the 0/0 limit and its derivative are both exact.
    tdays = t_s / DAY_TO_S
    bt = 2.0 * bv * tdays ** dv
    e_th = 0.36 * (jnp.exp(-av * tdays) + _log1p_over_x(bt))
    return lum_in * e_th


def bolometric(t_src_s, mej, vej, kappa, temperature_floor=4000.0, *,
               arnett_prefactor=ARNETT_PREFACTOR, time_grid=None):
    """Bolometric luminosity, temperature, photosphere radius at t_src_s.

    t_src_s : source-frame times in SECONDS SINCE EXPLOSION, any shape, need not be
              sorted, may be <= 0 (pre-explosion, see below). Build it with
              source_time_s(t_obs_days, z, t_exp_days); it may be a traced array, so
              t_exp is differentiable.
    mej     : ejecta mass, solar masses
    vej     : minimum initial velocity, units of c
    kappa   : grey opacity, cm^2/g
    temperature_floor : K. May be traced (fittable); enters through r_floor and the
              free-expansion / floor switch below.
    arnett_prefactor : 1.0 (default, = redback) or 2.0 (= Arnett 1982 / Villar+2017).
              redback's diffusion kernel integrates to 1/2, so its L is a uniform factor 2
              -- 0.7526 mag -- below the standard Arnett solution. See ARNETT_PREFACTOR for
              the measurement. Must be a PYTHON float: it is compared at trace time.
    time_grid : None (default here) or a source-frame grid in seconds, from
              :func:`redback_time_grid`. None evaluates the diffusion integral with identity 6's
              quadrature, converged at every epoch. A grid reproduces redback 1.20 instead: its
              trapezoid on that grid, then T and R interpolated to t_src_s -- equal to redback to
              ~1e-9, and as under-resolved as redback past ~2.66 t_diff (redback's
              ``dense_resolution=500`` is 1.19 mag too bright at 20 d). The whisper
              kilonova factories default to the grid, for parity with redback.

    Returns (L_bol / LSCALE, T [K], R_phot [cm]), each with t_src_s.shape.
    Luminosity is returned in units of LSCALE = 1e40 erg/s: the cgs value is
    ~1e41 and exceeds float32's 3.4e38 ceiling. Multiply by LSCALE only under
    x64. T and R_phot are in plain cgs and are safe in either precision.

    PRE-EXPLOSION MASK. t_src <= T_EVAL_MIN returns (0, temperature_floor, 0). R = 0 makes
    every downstream flux exactly 0 -- the flux scales as (R/dl)^2 -- with no special case
    needed in _flux_nu. The mask is a DOUBLE where(): the time fed to the physics is clamped
    to T_EVAL_MIN BEFORE any log or fractional power, so the dead branch produces no NaN in
    either the forward or the backward pass, and the live branch is untouched. Without the
    clamp, t <= 0 gives log(t_split) = NaN and tdays**dv = NaN, and JAX's where() propagates
    NaN through the reverse pass even when the forward value is masked away.

    The mask is not a physical approximation: the model has F ~ t^2 as t -> 0 (L ~ t^2 and
    T -> const, so R ~ t and F ~ R^2), giving ~1e-45 mJy at t = T_EVAL_MIN = 1 ms. So both
    the value AND the derivative go to zero at the explosion, the model is C^1 in t_exp
    across the boundary, and the step introduced at the mask edge is below any float epsilon.

    NOTE: `grid_s`/`row` are gone. Since identity 6 the diffusion integral carries its own
    Gauss-Legendre nodes in s derived from the evaluation time alone, so grid_s was only ever
    read as grid_s[row] and `n_dense` controlled nothing. Verified bit-identical for grids of
    4 to 5004 points, for t_min in {1e-6,1e-2,1e3}, and with NaN/inf in every unselected grid
    entry, in both precisions. The old signature still works -- see _grid_row_shim.
    """
    if time_grid is not None:
        return _bolometric_on_grid(time_grid, t_src_s, mej, vej, kappa, temperature_floor,
                                   arnett_prefactor)
    v0 = vej * SPEED_OF_LIGHT
    m0 = mej * SOLAR_MASS
    t_src_s = jnp.asarray(t_src_s)
    live = t_src_s > T_EVAL_MIN
    # jnp.where, not jnp.maximum: maximum propagates NaN, and a caller who hands us a NaN
    # time (e.g. from an unconstrained t_exp reparametrisation) should get a masked zero,
    # not a poisoned reverse pass.
    t_live = jnp.where(live, t_src_s, T_EVAL_MIN)
    # identity 5a: 2*kappa*m0/(BETA*v0*c) == TDIFF_CONST*kappa*mej/vej, since
    # m0 = mej*M_sun and v0 = vej*c. Same value; the divisor is now vej (O(0.1)) instead of
    # BETA*v0*c ~ 1e21, which float32 cannot square during the backward pass.
    tdiff = jnp.sqrt(TDIFF_CONST * kappa * mej / vej)
    av, bv, dv = thermalisation_coeffs(mej, vej)

    # diffusion (identity 6): split at t' = SPLIT_FRAC * t, each side in the variable that makes
    # it well-conditioned (see the constant block above for why both are needed).
    t_eval = t_live[..., None]                           # (..., 1)
    amp_a = t_eval ** 2 / tdiff ** 2                     # A = t^2 / td^2
    t_split = SPLIT_FRAC * t_eval

    # OUTER, t' in [t_split, t], in s = (t^2 - t'^2)/td^2 so the kernel becomes exp(-s) on a
    # fixed domain. s runs to A(1 - SPLIT_FRAC^2) (which lands exactly on t_split), capped at
    # S_MAX beyond which exp(-s) is below float resolution anyway.
    s_hi = jnp.minimum(amp_a * (1.0 - SPLIT_FRAC ** 2), S_MAX)
    s = s_hi * _const(GL_X)                              # (..., 1)*(128,) -> (..., 128)
    tp_o = jnp.sqrt(jnp.maximum(t_eval ** 2 - s * tdiff ** 2, 0.0))
    lum_o = 0.5 * s_hi[..., 0] * jnp.sum(
        _const(GL_W) * _heating(tp_o, m0, av, bv, dv) * jnp.exp(-s), axis=-1)

    # INNER, t' in [T_FLOOR, t_split], in z = log t'. With t' = e^z, dt' = t' dz, so
    #   L_inner = (1/td^2) int f(t') t'^2 exp(-(t^2 - t'^2)/td^2) dz
    # and the t'^-1.3 heating divergence becomes a smooth exponential in z, which GL resolves.
    z_lo = jnp.log(jnp.asarray(T_FLOOR, dtype=t_eval.dtype))
    z_hi = jnp.maximum(jnp.log(t_split), z_lo)           # empty if t_split <= t_min
    tp_i = jnp.exp(z_lo + (z_hi - z_lo) * _const(GL_XI))
    expo_i = jnp.minimum(-(t_eval ** 2 - tp_i ** 2) / tdiff ** 2, 0.0)
    lum_i = (z_hi - z_lo)[..., 0] / tdiff ** 2 * jnp.sum(
        _const(GL_WI) * _heating(tp_i, m0, av, bv, dv) * tp_i ** 2 * jnp.exp(expo_i), axis=-1)

    lum = _arnett_rescaled(lum_o + lum_i, arnett_prefactor)

    temperature, r_photosphere = _photosphere(lum, t_live, v0, temperature_floor)
    valid = (mej > 0.0) & live                           # redback line 1864, plus t_exp mask
    return (jnp.where(valid, lum, 0.0),
            jnp.where(valid, temperature, temperature_floor),
            jnp.where(valid, r_photosphere, 0.0))


def _arnett_rescaled(lum, arnett_prefactor):
    """``L`` times ``arnett_prefactor``, shared by both diffusion paths of :func:`bolometric`."""
    # See ARNETT_PREFACTOR. Compared at trace time: at the default the multiply is not emitted
    # and the graph is byte-identical to the version that had no flag.
    if arnett_prefactor != 1.0:
        lum = arnett_prefactor * lum
    return lum


def _photosphere(lum, t_s, v0, temperature_floor):
    """redback's temperature-floor photosphere ``(T [K], R [cm])`` from ``L/LSCALE`` at ``t_s``."""
    # L is ~1e41 erg/s and CANNOT be represented in float32 (max 3.4e38), so it
    # stays in LSCALE units and the constants absorb the factor.
    lum_safe = jnp.maximum(lum, 1e-30)                   # keeps d/dL of L^0.25 finite
    r_free = v0 * t_s
    # identity 5b: (L/(4 pi sigma R^2))^0.25 * S^0.25 == (S/(4 pi sigma))^0.25 * L^0.25 / sqrt(R).
    # Same value, but the differentiated divisor becomes sqrt(r_free) ~ 1e7 rather than
    # 4*pi*sigma*r_free**2 ~ 1e23..1e30, which float32 cannot square in the backward pass.
    t_free = T_CONST * lum_safe ** 0.25 / jnp.sqrt(r_free)
    # identity 5c: r_floor regrouped so the differentiated divisor is Tf^2, not
    # 4 pi sigma Tf^4 -- see the constant block. Required now that temperature_floor is
    # fittable. Keeping LSCALE inside R_FLOOR_CONST is safe here (unlike 4*pi*sigma/LSCALE
    # = 7e-44, which underflows float32) because the constant is 3.7e21, a normal float32.
    r_floor = R_FLOOR_CONST * jnp.sqrt(lum_safe) / temperature_floor ** 2

    # The switch is CONTINUOUS in value: t_free == temperature_floor implies r_free == r_floor
    # identically (both say L = 4 pi sigma R^2 T^4), so L, T, R and every flux are C^0 across
    # it. Only the DERIVATIVE w.r.t. temperature_floor jumps (0 on the hot side, nonzero on the
    # floored side).
    hot = t_free > temperature_floor
    return jnp.where(hot, t_free, temperature_floor), jnp.where(hot, r_free, r_floor)


def _bolometric_on_grid(time_grid, t_src_s, mej, vej, kappa, temperature_floor, arnett_prefactor):
    """``bolometric`` with redback 1.20's discretisation. See ``time_grid`` in :func:`bolometric`.

    ORIGINAL (kilonova_models.py:1628, ``_kilonova_diffusion_luminosity``, on ``time_temp``):
        decay = np.exp(-np.diff(t**2/td**2))
        L[1:] = 0.5*np.diff(t)/td**2 * (decay*L_in[:-1]*t[:-1] + L_in[1:]*t[1:])
        L = solve_banded((1, 0), [[1...], [-decay...]], L)     # L[i] += decay[i-1]*L[i-1]
        L[0] = L[1]*np.exp(s[1] - s[0])
    then ``T`` and ``R`` on the grid, and ``interp1d`` of each to the epochs.

    CHANGES: the first-order recursion is a ``lax.associative_scan`` rather than a sequential
    solve (the same sums, reassociated: ~1e-16 per step, and log-depth on a GPU); redback's
    non-finite-prefix guard is dropped, as nothing here produces a non-finite value (identities
    1-8); the pre-explosion mask is :func:`bolometric`'s. ``e_th`` goes through identity 7's
    series, which is where the ~1e-9 residual against redback comes from.
    """
    v0 = vej * SPEED_OF_LIGHT
    t_src_s = jnp.asarray(t_src_s)
    grid = jnp.asarray(time_grid, dtype=t_src_s.dtype)
    tdiff = jnp.sqrt(TDIFF_CONST * kappa * mej / vej)
    av, bv, dv = thermalisation_coeffs(mej, vej)
    f = _heating(grid, mej * SOLAR_MASS, av, bv, dv) * grid        # L_in e_th t, LSCALE units
    s = grid ** 2 / tdiff ** 2
    decay = jnp.exp(-jnp.diff(s))                                  # <= 1: nothing overflows
    rhs = 0.5 * jnp.diff(grid) / tdiff ** 2 * (decay * f[:-1] + f[1:])
    _, tail = jax.lax.associative_scan(lambda a, b: (a[0] * b[0], b[0] * a[1] + b[1]),
                                       (decay, rhs))
    lum = _arnett_rescaled(jnp.concatenate([tail[:1] * jnp.exp(s[1] - s[0]), tail]),
                           arnett_prefactor)
    temperature, r_photosphere = _photosphere(lum, grid, v0, temperature_floor)

    valid = (mej > 0.0) & (t_src_s > T_EVAL_MIN)
    return (jnp.where(valid, jnp.interp(t_src_s, grid, lum), 0.0),
            jnp.where(valid, jnp.interp(t_src_s, grid, temperature), temperature_floor),
            jnp.where(valid, jnp.interp(t_src_s, grid, r_photosphere), 0.0))


bolometric = _grid_row_shim(bolometric, probe=1)


def bolometric_grid(grid_s, row, mej, vej, kappa, temperature_floor=4000.0, **kw):
    """DEPRECATED explicit spelling of the old (grid_s, row) signature."""
    return bolometric(grid_s[row], mej, vej, kappa, temperature_floor, **kw)


def _flux_nu(temperature, r_photosphere, dl_cm, nu_source, redshift,
             pref_const=PLANCK_T3):
    """Blackbody F_nu in erg/s/Hz/cm^2, observer frame. redback/sed.py:210.

    Every large quantity is pre-scaled into a precomputed literal constant.
    Manual regrouping is NOT sufficient: XLA reassociates float products under
    jit, so an expression that is finite in eager mode can overflow once
    compiled. Scale the variables, not the parentheses.

    pref_const selects the OUTPUT UNIT: PLANCK_T3 (default) gives cgs, PLANCK_T3_AB
    gives the same spectrum divided by the AB zero point (identity 8).
    """
    ratio = RATIO_CONST * ((r_photosphere / RSCALE) / (dl_cm / DLSCALE)) ** 2
    # The barrier stops XLA hoisting 1/NUSCALE back out of the cube below. Without it the
    # scalar-nu path (flux_density_mjy) returns NaN under jit while the array-nu path
    # (ab_magnitude) happens to survive -- i.e. correctness by compiler heuristic. Identity
    # JVP/transpose, so gradients are unaffected.
    nu14 = jax.lax.optimization_barrier(nu_source / NUSCALE)
    arg = PLANCK_ARG * nu14 / temperature                # x = h nu / (k T), O(1..40)
    # identity 4: 2 pi h nu^3 / c^2 == (2 pi k^3 / (h^2 c^2)) * x^3 * T^3, using nu = x k T / h.
    # PLANCK_T3 == PLANCK_PREF / PLANCK_ARG**3 exactly (verified to 1 double ulp). Both x and T
    # are O(1)-by-construction, so no XLA reassociation of this product can overflow float32 --
    # unlike nu^3 ~ 1e44, which does.
    # pref_const carries the unit (identity 8). It is a FOLDED LITERAL, not a multiply applied
    # afterwards, so f_nu never passes through a subnormal float32 intermediate on the way.
    pref = pref_const * arg ** 3 * temperature ** 3
    return pref * ratio * _inverse_expm1(arg) * (1.0 + redshift)


def flux_density_mjy(t_src_s, nu_obs_hz, redshift, dl_cm,
                     mej, vej, kappa, temperature_floor=4000.0, *,
                     arnett_prefactor=ARNETT_PREFACTOR, time_grid=None,
                     ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                     r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """Flux density in mJy at observer-frame frequencies nu_obs_hz.

    nu_obs_hz must broadcast against t_src_s: one frequency per observation, or a scalar.
    Extinction arguments are documented on ab_magnitude; they default to None, in which
    case not a single extinction op enters the traced graph. ``time_grid`` is
    :func:`bolometric`'s.

    Pre-explosion epochs return EXACTLY 0.0 mJy with exactly zero gradient. This is the
    interface to fit through: zero is a perfectly good flux, it is C^1 in t_exp, and it needs
    no floor constant. Magnitudes cannot do any of that (see ab_magnitude).
    """
    _, temp, rad = bolometric(t_src_s, mej, vej, kappa, temperature_floor,
                              arnett_prefactor=arnett_prefactor, time_grid=time_grid)
    nu_src = nu_obs_hz * (1.0 + redshift)               # k-correction
    fd = _flux_nu(temp, rad, dl_cm, nu_src, redshift) / MJY
    # c in Angstrom/s, folded into one literal (2.99792458e18) so nothing intermediate
    # is 1e10 * 1e8 in float32.
    lam_obs = 2.99792458e18 / nu_obs_hz
    tau = _ext_transmission(lam_obs, redshift, ebv_mw, ebv_host,
                            xi_mw, xi_host, r_v_mw, r_v_host, law)
    return fd if tau is None else fd * tau


flux_density_mjy = _grid_row_shim(flux_density_mjy, probe=1)


def ab_magnitude(t_src_s, band_idx, weights, norms, lam_obs_ang,
                 redshift, dl_cm, mej, vej, kappa, temperature_floor=4000.0, *,
                 mag_floor=MAG_FLOOR, arnett_prefactor=ARNETT_PREFACTOR, time_grid=None,
                 ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None,
                 r_v_mw=3.1, r_v_host=3.1, law='f99'):
    """AB magnitude in the band given by band_idx, for each observation.

    weights, norms : from ab_weights(). lam_obs_ang : the wavelength grid. time_grid : see
    :func:`bolometric`.

    mag_floor (default MAG_FLOOR = 40.0, USER-TUNABLE) caps the returned magnitude at
    min(mag, mag_floor), which is what keeps a zero flux -- a pre-explosion epoch, or a
    band pushed under float resolution by extinction or a low temperature floor -- finite.
    The cap is continuous and has zero gradient in the capped region. Measured safe up to
    mag_floor = 55 in float32 (first failure at 60) and 100 in float64 -- see the MAG_FLOOR
    block for the measurement and for why the limit exists at all.

    Extinction (all optional, keyword-only). None means "not requested", and the whole term
    then disappears at TRACE time, so a call without extinction compiles to the same graph
    it compiled to before extinction existed.
      ebv_mw   : Milky Way E(B-V) mag, applied at the OBSERVED wavelength.
      ebv_host : host-galaxy E(B-V) mag, applied at the REST wavelength lam_obs/(1+z).
      xi_mw, xi_host : precomputed A(lam)/E(B-V) shape vectors on the lam_obs_ang grid,
          from extinction_shape(..., frame='observer'/'rest'). Supply these whenever R_V
          is FIXED (the usual case): the law is then never evaluated inside the model and
          the E(B-V) dependence is a single scalar multiply -- which is EXACT, because
          A(lam) is exactly linear in E(B-V) at fixed R_V.
      r_v_mw, r_v_host, law : used only where the matching xi is None, i.e. when R_V is
          itself sampled. `law` is a Python string: jit it with static_argnames=('law',).

    For a FIXED E(B-V) prefer extincted_weights() at setup: it folds the correction into
    `weights` and costs zero extra model runtime.

    Prefer flux_density_mjy for the likelihood. Magnitude space is genuinely singular at the
    explosion: F ~ t^2 gives mag ~ -5 log10(t - t_exp), so d(mag)/d(t_exp) -> +inf as
    t_obs -> t_exp from above. The cap bounds the value but not that approach, and a Gaussian
    residual against an arbitrary mag_floor is not a censored-data likelihood. Use
    pre_explosion() to route those epochs to an upper-limit term instead.
    """
    _, temp, rad = bolometric(t_src_s, mej, vej, kappa, temperature_floor,
                              arnett_prefactor=arnett_prefactor, time_grid=time_grid)
    nu_obs = SPEED_OF_LIGHT / (lam_obs_ang * 1e-8)      # Hz, observer frame
    nu_src = nu_obs[None, :] * (1.0 + redshift)

    # PLANCK_T3_AB, not PLANCK_T3: the spectrum is carried in AB-zero-point units, which is
    # what keeps d(mag)/d(flux) inside float32 (identity 8). Forward-identical to the cgs
    # form -- norms is rescaled by the same literal five lines below.
    f_ab = _flux_nu(temp[:, None], rad[:, None], dl_cm, nu_src, redshift, PLANCK_T3_AB)
    tau = _ext_transmission(lam_obs_ang, redshift, ebv_mw, ebv_host,
                            xi_mw, xi_host, r_v_mw, r_v_host, law)
    # Redden the WEIGHTS, not the spectrum: weights is (n_band, n_wave) and f_nu is
    # (n_obs, n_wave) with n_obs >= n_band, so this is the cheaper of the two and it keeps
    # `norms` -- the pure filter zero-point integral -- untouched. Reddening norms as well
    # would divide the extinction straight back out.
    w = weights if tau is None else weights * tau[None, :]
    num = jnp.sum(f_ab * w[band_idx], axis=1)
    # The barrier stops XLA undoing the rescaling by reassociating num/(INV_AB*norms) into
    # (num/INV_AB)/norms, which would put the 1e-39 cgs value back in the denominator.
    den = jax.lax.optimization_barrier(INV_AB_ZEROPOINT * norms[band_idx])
    # Clamp the RATIO at 10**(-0.4*mag_floor), not the magnitude: -2.5*log10(max(u, f))
    # IS min(mag, mag_floor), but it never evaluates log10(0), whose derivative is inf and
    # would poison the backward pass through a jnp.minimum. The clamp this replaces was
    # 1e-300, which is exactly 0.0 in float32 and therefore clamped nothing at all.
    ratio_floor = jnp.asarray(10.0, dtype=num.dtype) ** (-0.4 * mag_floor)
    return -2.5 * jnp.log10(jnp.maximum(num / den, ratio_floor))


ab_magnitude = _grid_row_shim(ab_magnitude, probe=2)


bolometric_jit = jax.jit(bolometric, static_argnames=('arnett_prefactor',))
# `law` is a Python string and must stay static; everything else, E(B-V) included, traces.
flux_density_jit = jax.jit(flux_density_mjy, static_argnames=('law', 'arnett_prefactor'))
ab_magnitude_jit = jax.jit(ab_magnitude, static_argnames=('law', 'arnett_prefactor'))


# --- self-test -------------------------------------------------------------
