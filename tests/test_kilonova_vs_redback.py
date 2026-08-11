"""The project's central claim, as a test.

whisper-GPU's kilonova must reproduce redback's physics **where redback is correct**, and diverge
from it in the *documented* way where redback is not.

The subtlety that makes this test worth writing carefully: redback's diffusion integral uses a
300-point geomspace trapezoid whose kernel width shrinks as ``t_diff^2 / 2t`` while the grid
spacing grows, so past roughly ``t > 2.66 * t_diff`` it is under-resolved and too bright. Asserting
"matches redback" there would enshrine redback's quadrature error as the specification.

So the reference is **not** redback-at-defaults. It is an adaptive ``scipy.quad`` evaluation of the
same integral, which redback's own grid converges onto once refined (ratio 7.37 -> 1.00004 at
200,000 points). Inside the horizon all three agree; outside it, redback is the outlier.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.jax import kilonova as kn  # noqa: E402

DAY = 86400.0
#: redback's quadrature is good to <0.003 mag below 1.4*t_diff and exceeds 0.05 mag beyond
#: ~2.66*t_diff (median over 9 crossing cases).
HORIZON_FACTOR = 2.66


def t_diff_days(mej, vej, kappa):
    """Diffusion timescale in days: t_diff = sqrt(TDIFF_CONST * kappa * mej / vej)."""
    return float(np.sqrt(kn.TDIFF_CONST * kappa * mej / vej)) / DAY


def reference_bolometric(t_eval_s, mej, vej, kappa):
    """Adaptive float64 reference for L(t), independent of the JAX code path.

    Integrates the u-form  L = A * int_0^1 f(t*u) * u * exp(-A(1-u^2)) du  with A = t^2/t_d^2,
    which is smooth on [0, 1]. A fixed grid is deliberately NOT used here: at the short-t_diff
    corner it needs (t/t_d)^2 ~ 1e6 nodes, i.e. it has the very resolution limit this test exists
    to detect.
    """
    from scipy.integrate import quad

    m0 = mej * kn.SOLAR_MASS
    td = np.sqrt(2.0 * kappa * m0 / (kn.BETA * vej * kn.SPEED_OF_LIGHT * kn.SPEED_OF_LIGHT))
    av, bv, dv = (float(np.asarray(x)) for x in kn.thermalisation_coeffs(mej, vej))

    def f(tp):
        d = tp - 1.3
        frac = np.arctan(0.11 / d) / np.pi if d > 0 else 0.5 - np.arctan(d / 0.11) / np.pi
        tday = tp / DAY
        bt = max(2.0 * bv * tday ** dv, 1e-18)
        e_th = 0.36 * (np.exp(-av * tday) + np.log1p(bt) / bt)
        return 4.0e18 * m0 * frac ** 1.3 * e_th

    out = []
    for te in np.atleast_1d(t_eval_s):
        amp = te * te / (td * td)
        integ = lambda u: f(te * u) * u * np.exp(-amp * (1.0 - u * u))  # noqa: E731
        lo = max(0.0, 1.0 - 20.0 / max(amp, 1e-30))                     # concentrate near u = 1
        v1, _ = quad(integ, lo, 1.0, limit=400, epsabs=0, epsrel=1e-12)
        v0 = quad(integ, 0.0, lo, limit=400, epsabs=0, epsrel=1e-12)[0] if lo > 0 else 0.0
        out.append(amp * (v1 + v0))
    return np.array(out)


CASES = [                              # (mej, vej, kappa, label)
    (0.01, 0.20, 1.0, "typical"),
    (0.05, 0.10, 3.0, "long t_diff"),
    (1e-4, 0.70, 0.1, "corner: shortest t_diff"),
    (0.10, 0.05, 30.0, "corner: longest t_diff"),
    (1e-3, 0.50, 0.5, "low mass, fast"),
]


@pytest.mark.parametrize("mej,vej,kappa,label", CASES)
def test_bolometric_matches_a_converged_reference(mej, vej, kappa, label):
    """Accurate everywhere, including well past redback's validity horizon."""
    t_days = np.array([0.5, 1.0, 3.0, 5.0, 10.0, 15.0])
    t_s = t_days * DAY
    got = np.asarray(kn.bolometric(jnp.asarray(t_s), mej, vej, kappa)[0],
                     dtype=np.float64) * kn.LSCALE
    ref = reference_bolometric(t_s, mej, vej, kappa)
    ok = ref > 0
    rel = np.abs(got[ok] - ref[ok]) / ref[ok]
    assert rel.max() < 1e-3, f"{label}: max rel err {rel.max():.3e} at t={t_days[ok][rel.argmax()]} d"


def test_flux_is_non_negative_everywhere():
    """The model is a Planck function times R^2 -- both non-negative, so F >= 0 by construction.

    Any negative flux in a simulation therefore comes from added observational noise, never from
    the model. This matters because it is what makes flux space the safe space to simulate in.
    """
    rng = np.random.default_rng(0)
    t_s = np.geomspace(0.5, 21.0, 30) * DAY
    for _ in range(40):
        mej = float(np.exp(rng.uniform(np.log(1e-4), np.log(0.1))))
        vej = float(rng.uniform(0.05, 0.7))
        kap = float(np.exp(rng.uniform(np.log(0.1), np.log(30.0))))
        L, T, R = kn.bolometric(jnp.asarray(t_s), mej, vej, kap)
        for arr in (L, T, R):
            a = np.asarray(arr)
            assert np.all(np.isfinite(a)), (mej, vej, kap)
            assert np.all(a >= 0.0), (mej, vej, kap)


def test_documented_divergence_from_redback_is_late_time_only():
    """Inside redback's horizon the two agree; outside it, redback is the one that is wrong.

    IMPORTANT: redback's `_one_component_kilonova_model` runs `cumulative_trapezoid` over EXACTLY
    the array it is handed, so it must be given its own dense grid -- passing only the epochs of
    interest silently reduces its quadrature to that many nodes and makes it look wrong everywhere.
    We therefore hand it its internal default, `geomspace(1e-3, 7e6, 300)`, and read off the
    nearest nodes. Skipped when redback is unavailable; the converged-reference test above is the
    binding one.
    """
    redback = pytest.importorskip("redback")  # noqa: F841
    from redback.transient_models.kilonova_models import _one_component_kilonova_model

    mej, vej, kappa = 0.01, 0.2, 1.0
    td = t_diff_days(mej, vej, kappa)

    grid = np.geomspace(1e-3, 7e6, 300)                       # redback's own default grid
    rb_L = _one_component_kilonova_model(grid, mej, vej, kappa, temperature_floor=4000.0)[0]

    want = np.array([0.3 * td, 0.8 * td, 8.0 * td]) * DAY     # inside, inside, well outside
    idx = np.searchsorted(grid, want).clip(1, grid.size - 1)
    t_s = grid[idx]

    jax_L = np.asarray(kn.bolometric(jnp.asarray(t_s), mej, vej, kappa)[0],
                       dtype=np.float64) * kn.LSCALE
    ref_L = reference_bolometric(t_s, mej, vej, kappa)
    rb = rb_L[idx]

    # whisper-GPU tracks the converged reference at every epoch, inside the horizon and outside
    assert np.max(np.abs(jax_L - ref_L) / ref_L) < 1e-3, np.abs(jax_L - ref_L) / ref_L

    # redback agrees inside its validity horizon ...
    assert np.max(np.abs(rb[:2] - ref_L[:2]) / ref_L[:2]) < 5e-2, rb[:2] / ref_L[:2]
    # ... and is substantially TOO BRIGHT beyond it, which is the documented divergence
    assert rb[-1] / ref_L[-1] > 2.0, (
        f"expected redback well above the converged reference past {HORIZON_FACTOR * td:.2f} d; "
        f"got ratio {rb[-1] / ref_L[-1]:.3f}")


# ---------------------------------------------------------------------------------------
def test_arnett_prefactor_default_is_redback_and_flag_is_exactly_two():
    """The Arnett factor-of-2, kept as redback BY PROJECT RULE, exposed as a flag.

    redback's diffusion kernel integrates to 1/2, not 1: substituting u = (t^2-t'^2)/td^2
    gives int (t'/td^2) exp(-(t^2-t'^2)/td^2) dt' = 1/2, so at late times -- where all
    deposited energy escapes -- L converges to L_in/2 instead of L_in. The default
    reproduces that (parity with redback outweighs the correction); arnett_prefactor=2.0
    selects the standard Arnett 1982 / Villar+2017 normalisation.
    """
    t_s = jnp.asarray(np.geomspace(0.5, 20.0, 24) * DAY)
    mej, vej, kappa = 0.01, 0.2, 1.0

    base = np.asarray(kn.bolometric(t_s, mej, vej, kappa)[0])
    same = np.asarray(kn.bolometric(t_s, mej, vej, kappa, arnett_prefactor=1.0)[0])
    twice = np.asarray(kn.bolometric(t_s, mej, vej, kappa, arnett_prefactor=2.0)[0])

    # the default IS redback's, bitwise -- the flag must cost nothing when unused
    assert np.array_equal(base, same)
    # and the flag is exactly the advertised factor (a single trace-time multiply by 2.0
    # is exact in binary floating point)
    assert np.array_equal(twice, 2.0 * base)

    # the physics claim: with the Arnett normalisation the kernel integrates to 1, so
    # L -> L_in at late times; with redback's it converges to L_in/2.
    td_d = t_diff_days(mej, vej, kappa)
    t_late = jnp.asarray([100.0 * td_d * DAY])
    m0 = mej * kn.SOLAR_MASS
    av, bv, dv = kn.thermalisation_coeffs(mej, vej)
    L_in = float(kn._heating(t_late, m0, av, bv, dv)[0])
    ratio_rb = float(kn.bolometric(t_late, mej, vej, kappa)[0][0]) / L_in
    ratio_ar = float(kn.bolometric(t_late, mej, vej, kappa, arnett_prefactor=2.0)[0][0]) / L_in
    assert abs(ratio_rb - 0.5) < 1e-3, f"redback normalisation: L/L_in = {ratio_rb}"
    assert abs(ratio_ar - 1.0) < 2e-3, f"Arnett normalisation: L/L_in = {ratio_ar}"

    # Threaded through the photometry. The BAND magnitude does not shift uniformly: doubling
    # L also raises T by 2^0.25 on the hot branch, moving the SED, so the band change there
    # is chromatic (measured 0.81-1.27 mag in this g-like band). Only on the FLOORED branch,
    # where T is pinned at temperature_floor and the flux scales purely as R_phot^2 ~ L, is
    # the shift exactly 2.5 log10(2) -- and that is itself a physics check of the floor.
    lam = np.geomspace(1000.0, 30000.0, 500)
    trans = np.zeros((1, lam.size)); trans[0][(lam > 4000) & (lam < 5500)] = 1.0
    W, N = kn.ab_weights(lam, trans)
    bidx = jnp.zeros(24, dtype=int)
    m1 = np.asarray(kn.ab_magnitude(t_s, bidx, W, N, jnp.asarray(lam), 0.01, 1.34e26,
                                    mej, vej, kappa, 4000.0))
    m2 = np.asarray(kn.ab_magnitude(t_s, bidx, W, N, jnp.asarray(lam), 0.01, 1.34e26,
                                    mej, vej, kappa, 4000.0, arnett_prefactor=2.0))
    # every epoch must brighten (more L can never dim a blackbody band)
    ok = m1 < 39.0
    assert np.all((m1 - m2)[ok] > 0.5), (m1 - m2)[ok]
    # floored branch: both runs at the floor -> exactly the bolometric factor
    floored1 = np.isclose(np.asarray(kn.bolometric(t_s, mej, vej, kappa, 4000.0)[1]), 4000.0)
    floored2 = np.isclose(np.asarray(kn.bolometric(t_s, mej, vej, kappa, 4000.0,
                                                   arnett_prefactor=2.0)[1]), 4000.0)
    both = floored1 & floored2 & ok
    assert both.any(), "no epoch on the floored branch; adjust the time grid"
    assert np.allclose((m1 - m2)[both], 2.5 * np.log10(2.0), atol=1e-4), (m1 - m2)[both]
