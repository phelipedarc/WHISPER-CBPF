"""The project's central claim, as a test.

The JAX kilonova (``whisper_cbpf.models.jax.kilonova``) must reproduce redback's physics **where
redback is correct**, and diverge from it in the *documented* way where redback is not.

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

    # the JAX port tracks the converged reference at every epoch, inside the horizon and outside
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


# ---------------------------------------------------------------------------------------
# the factories solve on redback 1.20's own grid by default, for parity with it.

def _top_hat_filter_set():
    lam = np.geomspace(3000.0, 11000.0, 300)
    trans = np.zeros((2, lam.size))
    trans[0][(lam > 4000) & (lam < 5500)] = 1.0
    trans[1][(lam > 7000) & (lam < 9000)] = 1.0
    return {"lam": lam, "trans": trans, "names": np.array(["a", "b"])}


def _redback_band_magnitudes(model, t_obs, bidx, fs, z, **p):
    """redback's SED through the JAX band integral (same nodes, same weights).

    One flattened call, as the reference comparison makes: its grid is the one a call on ``t_obs`` builds,
    because redback 1.20 sizes the kilonova grid from the extremes of the epochs only.
    """
    from redback.model_library import all_models_dict

    lam = fs["lam"]
    w, norms = (np.asarray(a) for a in kn.ab_weights(lam, fs["trans"]))
    mjy = np.asarray(all_models_dict[model](
        np.repeat(t_obs, lam.size), redshift=z, output_format="flux_density",
        frequency=np.tile(2.99792458e18 / lam, t_obs.size), **p)).reshape(t_obs.size, lam.size)
    return -2.5 * np.log10(np.sum(mjy * 1e-26 * w[bidx], axis=1) / norms[bidx])


KN_CASES = [                           # (mej, vej, kappa, temperature_floor)
    (0.03, 0.20, 3.0, 4000.0),
    (0.01, 0.30, 1.0, 2000.0),         # t_diff = 1.2 d: 20 d is 17 t_diff, redback's worst
    (0.05, 0.10, 25.0, 1000.0),
]


@pytest.mark.parametrize("mej,vej,kappa,tf", KN_CASES)
def test_factory_default_is_redback_at_every_epoch(mej, vej, kappa, tf):
    """``kilonova_model``'s default reproduces redback 1.20 out to 30 d.

    It used to follow the converged quadrature only (identity 6), which leaves redback past
    ~2.66 t_diff. Now the default is redback's own 500-node grid and the converged one is
    ``time_grid=None`` -- and the size of redback's late-time error is what separates them.
    """
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as RA
    from whisper_cbpf.models.jax import kilonova_model

    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        z = 0.0098
        dl = RA.redback_luminosity_distance_cm(z, "one_component_kilonova_model")
        fs = _top_hat_filter_set()
        t = np.geomspace(0.5, 30.0, 30)
        bidx = np.arange(t.size) % 2
        bands = np.array(["a", "b"])[bidx]
        p = dict(mej=mej, vej=vej, kappa=kappa, temperature_floor=tf)
        ref = _redback_band_magnitudes("one_component_kilonova_model", t, bidx, fs, z, **p)

        # mag_floor lifted: the red, late epochs of the kappa = 25 case sit past 40 mag, and the
        # default floor would clamp them (a separate deviation)
        mag = -2.5 * np.log10(kilonova_model(["a", "b"], z, dl, filter_set=fs, mag_floor=99.0)
                              .predict(p, t, bands) / 3631.0)
        assert np.max(np.abs(mag - ref)) < 1e-6, np.max(np.abs(mag - ref))

        conv = -2.5 * np.log10(kilonova_model(["a", "b"], z, dl, filter_set=fs, time_grid=None,
                                              mag_floor=99.0).predict(p, t, bands) / 3631.0)
        late = t > 2.66 * t_diff_days(mej, vej, kappa)     # none for kappa = 25: t_diff = 23 d
        if late.any():
            print(f"\nmej={mej} kappa={kappa}: converged - redback at t > 2.66 t_diff: "
                  f"max {np.max(np.abs(conv - ref)[late]):.3f} mag, brighter redback by "
                  f"{np.max((conv - ref)[late]):+.3f}; inside: "
                  f"{np.max(np.abs(conv - ref)[~late]):.4f}")
        assert np.max(np.abs(conv - ref)[~late]) < 0.02
    finally:
        jax.config.update("jax_enable_x64", was)


def test_two_component_default_is_the_sum_of_two_redback_one_component_calls():
    """whisper defines the two-component kilonova as two one-component redback calls summed
    (redback's own two-component model stops at 6 d). The JAX default reproduces that sum, past 6 d
    included, because each component is solved on the same one-component grid."""
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as RA
    from whisper_cbpf.models.jax import kilonova_two_model

    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        z = 0.0098
        dl = RA.redback_luminosity_distance_cm(z, "one_component_kilonova_model")
        fs = _top_hat_filter_set()
        t = np.geomspace(0.5, 20.0, 24)
        bidx = np.arange(t.size) % 2
        comps = [dict(mej=0.02, vej=0.25, kappa=1.0, temperature_floor=3000.0),
                 dict(mej=0.04, vej=0.12, kappa=12.0, temperature_floor=1500.0)]
        flux = sum(10 ** (-0.4 * _redback_band_magnitudes("one_component_kilonova_model", t,
                                                           bidx, fs, z, **c)) for c in comps)
        m = kilonova_two_model(["a", "b"], z, dl, filter_set=fs)
        p = {f"{k}_{c}": v for c, comp in zip(("blue", "red"), comps) for k, v in comp.items()}
        mag = -2.5 * np.log10(m.predict(p, t, np.array(["a", "b"])[bidx]) / 3631.0)
        assert np.max(np.abs(mag + 2.5 * np.log10(flux))) < 1e-6
    finally:
        jax.config.update("jax_enable_x64", was)


def test_redback_time_grid_is_redbacks():
    """The transcription of ``get_optimal_time_array`` against redback's own function."""
    pytest.importorskip("redback")
    from redback.utils import get_optimal_time_array

    for t_days in (np.geomspace(0.5, 5.5, 36), np.geomspace(0.1, 90.0, 12), np.array([3.0])):
        t_s = t_days * DAY / 1.0098
        want = get_optimal_time_array(1e-2, 7e6, 500, user_times=t_s)
        if t_days.max() * DAY / 1.0098 > 7e6:
            with pytest.warns(RuntimeWarning, match="past redback's kilonova grid"):
                got = kn.redback_time_grid(t_s)
        else:
            got = kn.redback_time_grid(t_s)
        assert np.array_equal(got, want)

    # epochs before the explosion: redback cannot evaluate them, and the grid must neither move
    # for them nor break when there is nothing else (it raised at t_src = 0, NaN below)
    t_s = np.array([-3.0, 0.0, 5.0, 40.0]) * DAY
    assert np.array_equal(kn.redback_time_grid(t_s), get_optimal_time_array(
        1e-2, 7e6, 500, user_times=t_s))
    for t_s in (np.array([-2.0, -1.0]) * DAY, np.zeros(2)):
        assert np.all(np.isfinite(kn.redback_time_grid(t_s)))


def test_factory_default_masks_epochs_before_explosion_as_the_converged_one_does():
    """A fixed ``t_exp_days`` after some (or all) epochs: those epochs are exactly the mag_floor
    flux on either grid, and the later ones are unaffected by them."""
    import warnings

    from whisper_cbpf.models.jax import kilonova_model

    fs = _top_hat_filter_set()
    p = dict(mej=0.03, vej=0.2, kappa=3.0, temperature_floor=4000.0)
    for t in (np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0]), np.array([1.0, 6.0, 9.0])):
        b = np.array(["a"] * t.size)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = kilonova_model(["a", "b"], 0.0098, 1.3e26, filter_set=fs, t_exp_days=5.0,
                                 ).predict(p, t, b)
        conv = kilonova_model(["a", "b"], 0.0098, 1.3e26, filter_set=fs, t_exp_days=5.0,
                              time_grid=None).predict(p, t, b)
        dead = t <= 5.0
        assert np.array_equal(got[dead], conv[dead]), (t, got, conv)
        assert np.all(got[~dead] > 100.0 * conv[dead].max(initial=0.0))


def test_a_pre_merger_epoch_moves_the_jax_and_cpu_grids_alike():
    """REGRESSION: an epoch at or before the merger -- a pre-merger upper limit -- is the
    earliest, so it sets the lower edge of redback's grid. The CPU adapter and
    ``two_component_kilonova`` clip it to ``MIN_TIME_DAY`` = 1e-3 d before redback builds that
    grid; the JAX factories clipped it to ``T_EVAL_MIN`` = 1e-3 s, and every post-merger epoch
    moved: up to 0.27 mag (one component) and 0.57 mag (two) over 20 prior draws, where without
    the pre-merger rows the two sides agree to 7e-9 mag."""
    pytest.importorskip("redback")
    import warnings

    import whisper_cbpf as wp
    from whisper_cbpf.models import redback_adapter as RA
    from whisper_cbpf.models.jax import kilonova_model, kilonova_two_model

    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        z, bands = 0.0098, ["lsstg", "lsstr", "lssti"]
        dl = RA.redback_luminosity_distance_cm(z, "one_component_kilonova_model")
        t = np.concatenate([[-2.0, -0.5], np.geomspace(0.3, 15.0, 15)])
        b = np.array(["lsstg", "lsstr"] + bands * 5)
        one = kilonova_model(bands, z, dl, mag_floor=99.0)
        two = kilonova_two_model(bands, z, dl, mag_floor=99.0)
        pairs = [(RA.redback_model("one_component_kilonova_model", bands, redshift=z), one, {}),
                 (wp.get_model("two_component_kilonova"), two, two.param_aliases)]
        rng = np.random.default_rng(0)
        for cpu, gpu, aliases in pairs:
            for _ in range(3):
                p = gpu.default_prior.sample(rng)
                q = {aliases.get(k, k): v for k, v in p.items()}
                if aliases:
                    q["redshift"] = z
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    want = -2.5 * np.log10(cpu.predict(q, t, b)[2:] / 3631.0)
                got = -2.5 * np.log10(gpu.predict(p, t, b)[2:] / 3631.0)
                ok = want < 30.0
                assert ok.sum() >= 5, want
                assert np.max(np.abs(got - want)[ok]) < 1e-6, (gpu.name, got - want)
    finally:
        jax.config.update("jax_enable_x64", was)


def test_constants_do_not_freeze_at_the_import_precision():
    """REGRESSION: the quadrature nodes (``GL_*``), the Barnes-Kasen thermalisation table
    (``_BK_*``) and the F99 spline (``_F99_*``) were jnp arrays built AT IMPORT, so a module
    imported before float64 was enabled -- as any script that imports first and configures second
    does -- froze them in float32 and carried 3.6e-8 median, up to 1.9e-6 relative error into
    float64 L_bol (50 draws x 20 epochs; 4.8e-8 at this point), 1.1e-7 into the F99 law: the bug
    ``supernova._NXCS`` had. They are float64 NumPy now and take the session's precision where
    they are used. Measured in fresh interpreters, one per import order."""
    import os
    import subprocess
    import sys

    code = ("import sys, jax\n"
            "if sys.argv[1] == 'after': jax.config.update('jax_enable_x64', True)\n"
            "from whisper_cbpf.models.jax import kilonova as kn\n"
            "jax.config.update('jax_enable_x64', True)\n"
            "import jax.numpy as jnp\n"
            "t = jnp.asarray([0.3, 1.0, 3.0, 10.0]) * 86400.0\n"
            "lum = kn.bolometric(t, 0.03, 0.23, 3.0)[0]\n"
            "ext = kn.f99_a_over_ebv(jnp.asarray([3500.0, 4800.0, 6200.0, 9000.0]), 3.3)\n"
            "print(repr([lum.tolist(), ext.tolist()]))")
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    got = {order: [np.array(a) for a in eval(subprocess.run(
        [sys.executable, "-c", code, order], capture_output=True, text=True, check=True,
        env=env).stdout)] for order in ("before", "after")}
    for what, before, after in zip(("L_bol", "F99"), got["before"], got["after"]):
        rel = np.max(np.abs(before - after) / np.abs(after))
        assert rel < 1e-14, f"{what}: {rel:.2e} relative between the two import orders"


def test_two_component_names_pair_with_the_cpu_model_through_param_aliases():
    """The JAX port names its parameters ``mej_blue ... kappa_red`` and the CPU
    models (``two_component_kilonova``, the redback adapter) ``mej_1 ... kappa_2``, so a CPU and a
    GPU posterior of the same model shared none of its 8 columns (168 of 176 paired). The names
    stay; ``Model.param_aliases`` maps them, survives registration, and renaming a JAX draw with it
    gives a parameter set the CPU model evaluates to the same light curve."""
    pytest.importorskip("redback")
    import whisper_cbpf as wp
    from whisper_cbpf.models import redback_adapter as RA
    from whisper_cbpf.models import two_component_kilonova as tck
    from whisper_cbpf.models.jax import register_kilonova_two

    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        z = 0.0098
        dl = RA.redback_luminosity_distance_cm(z, "one_component_kilonova_model")
        m = register_kilonova_two(["ztfg", "ztfr"], z, dl, name="kn2_alias_test")
        aliases = wp.get_model("kn2_alias_test").param_aliases
        assert aliases == m.param_aliases and sorted(aliases) == sorted(m.parameters)
        cpu_names = [p for p in tck.PARAMETERS if p != "redshift"]
        assert sorted(aliases.values()) == sorted(cpu_names)
        assert set(aliases.values()) <= set(RA.redback_parameters("two_component_kilonova_model"))
        assert wp.get_model("two_component_kilonova").param_aliases == {}   # already redback's

        draw = m.default_prior.sample(np.random.default_rng(3))
        cpu_draw = {aliases[k]: v for k, v in draw.items()}
        t = np.geomspace(0.5, 12.0, 10)
        b = np.array(["ztfg", "ztfr"] * 5)
        gpu = m.predict(draw, t, b)
        cpu = wp.get_model("two_component_kilonova").predict({**cpu_draw, "redshift": z}, t, b)
        assert np.max(np.abs(2.5 * np.log10(gpu / cpu))) < 1e-3
    finally:
        jax.config.update("jax_enable_x64", was)


def test_two_component_prior_takes_either_spelling():
    """The docstring's apples-to-apples recipe, ``kilonova_two_model(...,
    prior=kilonova_two.default_prior())``, gave the model a prior named ``mej_1 ... kappa_2`` for
    parameters named ``mej_blue ... kappa_red``, so building its density failed looking up
    ``mej_blue``. Through ``param_aliases`` a prior in either spelling, or a mix, now comes out in
    the model's names with the same distributions in the same order; a parameter named both ways
    raises."""
    import whisper_cbpf as wp
    from whisper_cbpf.models.jax import kilonova_two as kn2
    from whisper_cbpf.models.jax import kilonova_two_model
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        rb = kn2.default_prior()
        m = kilonova_two_model(["ztfg", "ztfr"], 0.0098, 1.23e26, prior=rb)
        assert m.default_prior.names == m.parameters
        assert [m.default_prior.distributions[p] for p in m.parameters] == [
            rb.distributions[m.param_aliases[p]] for p in m.parameters]
        t = np.linspace(0.5, 5.0, 6)
        lc = wp.LightCurve(time=t, band=np.array(["ztfg", "ztfr"] * 3), flux=np.full(6, 1e-4),
                           flux_err=np.full(6, 1e-5))
        draw = m.default_prior.sample(np.random.default_rng(0))
        assert np.isfinite(float(make_log_prob_jax(lc, m)(
            np.array([draw[p] for p in m.parameters]))))

        mixed = dict(rb.distributions)
        mixed["kappa_red"] = mixed.pop("kappa_2")
        m2 = kilonova_two_model(["ztfg", "ztfr"], 0.0098, 1.23e26, prior=wp.Prior(mixed))
        assert sorted(m2.default_prior.names) == sorted(m2.parameters)
        mixed["kappa_2"] = rb.distributions["kappa_2"]
        with pytest.raises(ValueError, match="twice"):
            kilonova_two_model(["ztfg", "ztfr"], 0.0098, 1.23e26, prior=wp.Prior(mixed))
    finally:
        jax.config.update("jax_enable_x64", was)


def test_constants_raise_no_dtype_warning_after_x64_is_switched_off():
    """The same constants, the other way round: after a float64 call, a float32 one must not see a
    float64 constant JAX cached meanwhile (a bare ``jnp.asarray`` of the NumPy table under ``jit``
    did, and JAX truncated it with a UserWarning, which ``-W error`` callers and this file's
    ``test_factory_default_masks_epochs_before_explosion_as_the_converged_one_does`` turn into a
    failure). Fresh interpreter: x64 is process-wide."""
    import os
    import subprocess
    import sys

    code = ("import warnings, numpy as np, jax\n"
            "from whisper_cbpf.models.jax import kilonova_model\n"
            "lam = np.geomspace(1000.0, 30000.0, 400)\n"
            "trans = np.zeros((2, lam.size))\n"
            "trans[0][(lam > 4000) & (lam < 5500)] = 1.0\n"
            "trans[1][(lam > 5500) & (lam < 7000)] = 1.0\n"
            "fs = {'lam': lam, 'trans': trans, 'names': np.array(['a', 'b'])}\n"
            "p = dict(mej=0.03, vej=0.23, kappa=3.0, temperature_floor=4000.0)\n"
            "t, b = np.array([1.0, 2.0, 3.0]), np.array(['a', 'b', 'a'])\n"
            "build = lambda: kilonova_model(['a', 'b'], 0.0098, 1.3e26, filter_set=fs)\n"
            "jax.config.update('jax_enable_x64', True)\n"
            "build().predict(p, t, b)\n"
            "jax.config.update('jax_enable_x64', False)\n"
            "warnings.simplefilter('error')\n"
            "print(build().predict(p, t, b).dtype)")
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-600:]
