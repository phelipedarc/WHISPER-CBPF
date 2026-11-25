"""The TDE port's central claims, as tests.

There are three of them, and they are not the same claim:

1. **It reproduces redback.** The envelope ODE is a forward-Euler integration, so "the same
   physics" is not enough -- the same grid and the same arithmetic spelling are needed too.
   Verified against redback's own ``_cooling_envelope`` where redback is importable, and
   against an independent NumPy transcription (``_reference_engine`` below) always.

2. **It is differentiable where it matters.** The reference port this module replaced had
   NaN gradients through the photosphere, and its own smoke test could not see that because
   the one output it tested -- the bolometric luminosity -- is also the one output that
   never touches the diverging variable. :func:`test_gradients_finite_through_photosphere`
   is the regression test for that, and it deliberately does NOT test ``L``.

3. **"Agrees with redback" is ambiguous until the version is named.** redback 1.12.0 (the
   clone this project pins) and 1.15.1 (installed in the GPU container) have IDENTICAL
   ``_cooling_envelope`` loop bodies but differ in the grid size (5000 vs 500), the
   termination guards, ``f_debris``, and a ``(1+z)`` flux factor (1.20 is 1.15's grid). A
   test that hardcoded one of them would fail in the other container for a reason that has
   nothing to do with this code, so the package's own selector,
   ``redback_adapter.installed_redback_preset``, picks the matching preset at run time -- the
   one the module's default ``n_time`` follows too.

FLOAT64. The engine requires it and raises otherwise (see CHANGE 8 in the module). The
fixture below enables it for this module and restores the previous setting afterwards, so
the float32 kilonova tests in the same session are unaffected.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.jax import tde as T  # noqa: E402

DAY = 86400.0

#: (mbh_6, stellar_mass, eta, alpha, beta). Chosen to span the prior box and to include
#: both a long-lived envelope (case 0, constraint ~2400 at n_time=5000) and one that
#: terminates almost immediately (case 3, ~190).
CASES = [
    (1.0, 1.0, 0.05, 0.1, 1.0),
    (5.0, 0.5, 0.03, 0.2, 1.5),
    (0.5, 3.0, 0.08, 0.05, 0.8),
    (10.0, 0.2, 0.1, 0.5, 2.0),
]
PNAMES = ["mbh_6", "stellar_mass", "eta", "alpha", "beta"]


@pytest.fixture(scope="module", autouse=True)
def _x64():
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


def _preset_for_installed_redback():
    """The package's selector (keyed on redback's source, read without importing it)."""
    from whisper_cbpf.models.redback_adapter import installed_redback_preset

    return installed_redback_preset()


# ---------------------------------------------------------------------------------------
def _reference_engine(mbh_6, stellar_mass, eta, alpha, beta, n=500,
                      t_0_init=1.0, bec=0.8, zeta=2.0, hoverR=0.3):
    """redback's loop, transcribed to NumPy, independent of the JAX code path.

    This exists so claim 1 can be tested with no redback installed at all. It is a literal
    transcription of ``redback/transient_models/tde_models.py:97-149`` -- deliberately
    unclever, so that a disagreement points at the port and not at this.
    """
    G, MSUN, RSUN = T.GRAVITATIONAL_CONSTANT, T.SOLAR_MASS, T.SOLAR_RADIUS
    C, SSB = T.SPEED_OF_LIGHT, T.SIGMA_SB

    Mstar = stellar_mass * MSUN
    Rstar = stellar_mass ** 0.8 * RSUN
    Rt = Rstar * (mbh_6 * 1e6 / stellar_mass) ** (1 / 3)
    Rcirc = 2.0 * Rt / beta
    tfb = float(T.calc_tfb(bec, mbh_6, stellar_mass))
    Ledd40 = 1.4e4 * mbh_6
    t = np.logspace(np.log10(tfb), np.log10(5000 * tfb), n)
    Mdotfb = (0.8 * Mstar / (3.0 * tfb)) * (t / tfb) ** (-5 / 3)

    e = lambda: np.empty(n)                                                  # noqa: E731
    Me, Ee, Rv, Rph, Lamb = e(), e(), e(), e(), e()
    Racc, Edfb, Lrad, Teff, tacc, MdBH, Edbh = (e() for _ in range(7))

    Me[0] = 0.1 * Mstar + 0.4 * Mstar * (1 - t_0_init ** (-2 / 3))
    Rv[0] = (2 * Rt ** 2 / (5 * bec * Rstar)) * (Me[0] / Mstar)
    Ee[0] = ((2 * G * mbh_6 * 1e6 * Me[0]) / (5 * Rv[0])) * 2e-7
    Lamb[0] = 0.38 * Me[0] / (10 * np.pi * Rv[0] ** 2)
    Rph[0] = Rv[0] * (1 + np.log(Lamb[0]))
    Racc[0] = zeta * Rv[0]
    Edfb[0] = (G * mbh_6 * 1e6 * Mdotfb[0] / Racc[0]) * 2e-7
    Lrad[0] = Ledd40 + Edfb[0]
    tacc[0] = (2.2e-17 * (10 / (3 * alpha)) * Rv[0] ** 2
               / (G * mbh_6 * 1e6 * Rcirc) ** 0.5 * hoverR ** -2)
    MdBH[0] = Me[0] / tacc[0]
    Edbh[0] = eta * C ** 2 * (Me[0] / tacc[0]) * 1e-40
    Teff[0] = 1e10 * ((Ledd40 + Edfb[0]) / (4 * np.pi * SSB * Rph[0] ** 2)) ** 0.25

    with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
        for ii in range(1, n):
            dt = t[ii] - t[ii - 1]
            Me[ii] = Me[ii - 1] - (MdBH[ii - 1] - Mdotfb[ii - 1]) * dt
            Ee[ii] = Ee[ii - 1] + (Ledd40 - Edbh[ii - 1]) * dt
            Rv[ii] = ((2 * G * mbh_6 * 1e6 * Me[ii]) / (5 * Ee[ii])) * 2e-7
            Lamb[ii] = 0.38 * Me[ii] / (10 * np.pi * Rv[ii] ** 2)
            Rph[ii] = Rv[ii] * (1 + np.log(Lamb[ii]))
            Racc[ii] = zeta * Rv[0] * (t[ii] / tfb) ** (2 / 3)
            Edfb[ii] = (G * mbh_6 * 1e6 * Mdotfb[ii] / Racc[ii]) * 2e-7
            Lrad[ii] = Ledd40 + Edfb[ii]
            Teff[ii] = 1e10 * ((Ledd40 + Edfb[ii]) / (4 * np.pi * SSB * Rph[ii] ** 2)) ** 0.25
            tacc[ii] = (2.2e-17 * (10 / (3.0 * alpha)) * Rv[ii] ** 2
                        / (G * mbh_6 * 1e6 * Rcirc) ** 0.5 * hoverR ** -2)
            MdBH[ii] = Me[ii] / tacc[ii]
            Edbh[ii] = eta * C ** 2 * (Me[ii] / tacc[ii]) * 1e-40

    try:
        c1 = int(np.min(np.where(Rv < Rcirc / 2.)))
        c2 = int(np.min(np.where(Me < 0.0)))
    except ValueError:
        c1 = c2 = n
    return dict(t=t, L=Lrad * 1e40, T=Teff, Rph=Rph, Rv=Rv, Me=Me,
                constraint=min(c1, c2), tfb=tfb)


def _trim(k):
    """Index below which two implementations of this recursion can be asked to agree.

    Forward Euler amplifies round-off, and near the end the ODE is genuinely
    ill-conditioned -- Rv oscillates between +1e14 and -1e15 on successive steps, in redback
    too. Measured, the drift is still at float64 round-off (< 1e-11) at 80% of the curve and
    reaches O(1) at the final step. 80% is therefore where agreement is a claim about the
    port rather than about rounding. :func:`test_ulp_amplification_is_confined_to_the_tail`
    is what stops this becoming a rug to sweep errors under.
    """
    return max(int(0.8 * k), 2)


# ================================================================= claim 1: reproduces redback
@pytest.mark.parametrize("p", CASES)
def test_engine_matches_independent_numpy_transcription(p):
    """No redback required: the JAX scan must equal a plain Python loop, term for term."""
    ref = _reference_engine(*p, n=500)
    got = T.cooling_envelope(*p, n_time=500)
    k = _trim(min(ref["constraint"], int(got["constraint"])))
    assert k > 10, f"nothing to compare: constraint {ref['constraint']}"
    for key, name in (("L", "bolometric_luminosity"), ("T", "photosphere_temperature"),
                      ("Rph", "photosphere_radius"), ("Rv", "envelope_radius"),
                      ("Me", "envelope_mass")):
        a = np.asarray(got[name], dtype=np.float64)[:k]
        b = ref[key][:k]
        rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
        assert rel < 1e-9, f"{name}: max rel {rel:.3e} at index {np.argmax(np.abs(a - b))}"


@pytest.mark.parametrize("p", CASES)
def test_engine_matches_redback(p):
    """The real thing, against whichever redback is installed."""
    pytest.importorskip("redback")
    from redback.transient_models.tde_models import _cooling_envelope

    preset = T.REDBACK_ENGINE_PRESETS[_preset_for_installed_redback()]
    rb = _cooling_envelope(*p)
    got = T.cooling_envelope(*p, **preset)
    k_rb, k_jx = len(rb.time_temp), int(got["constraint"])
    # +-2 steps: the termination index is ill-conditioned by construction (see _trim)
    assert abs(k_jx - k_rb) <= 2, f"constraint {k_jx} vs redback {k_rb}"

    k = _trim(min(k_rb, k_jx))
    for ours, theirs in (("bolometric_luminosity", rb.bolometric_luminosity),
                         ("photosphere_temperature", rb.photosphere_temperature),
                         ("photosphere_radius", rb.photosphere_radius)):
        a = np.asarray(got[ours], dtype=np.float64)[:k]
        b = np.asarray(theirs)[:k]
        rel = np.max(np.abs(a - b) / np.abs(b))
        assert rel < 1e-6, f"{ours}: max rel {rel:.3e}"


@pytest.mark.parametrize("p", CASES[:2])
def test_valid_mask_reproduces_redbacks_slice(p):
    """CHANGE 2: ``arr[valid]`` is what redback's ``arr[:constraint]`` returns."""
    got = T.cooling_envelope(*p, n_time=500)
    valid = np.asarray(got["valid"])
    k = int(got["constraint"])
    assert valid.sum() == k
    assert valid[:k].all() and not valid[k:].any()
    L = np.asarray(got["bolometric_luminosity"])
    assert np.array_equal(L[valid], L[:k])
    # nothing inside the mask may be a dead point
    assert np.all(np.asarray(got["envelope_mass"])[valid] > 0)
    assert np.all(np.asarray(got["envelope_energy"])[valid] > 0)


def test_meaningful_is_the_models_domain_not_a_percentage():
    """`meaningful` narrows `valid` by Rph > 0 -- the optically-thick condition Lamb > 1/e.

    Below it redback's photosphere radius goes NEGATIVE and is then squared into a blackbody,
    reporting a positive flux from a negative radius. A fraction-of-the-curve trim could not
    express that: it removes good points on short curves and keeps bad ones on long.
    """
    for p in CASES:
        o = T.cooling_envelope(*p, n_time=500)
        v, m = np.asarray(o["valid"]), np.asarray(o["meaningful"])
        assert np.all(v[m]), "meaningful must be inside valid"
        if m.any():
            assert np.all(np.asarray(o["photosphere_radius"])[m] > 0)
            assert np.all(np.asarray(o["photosphere_temperature"])[m] > 0)


@pytest.mark.parametrize("p", CASES)
def test_ulp_amplification_is_confined_to_the_tail(p):
    """CHANGE 5, quantified -- and the test that stops ``_trim`` becoming a rug.

    Forward Euler amplifies round-off, so this implementation and a NumPy transcription of
    the same loop drift apart even in float64. What has to be true is that the drift is
    (a) at float64 round-off through the bulk of the curve, and (b) confined to the last
    few steps. If either half of that stops holding, the trim is hiding a real defect.
    """
    ref = _reference_engine(*p, n=500)
    got = T.cooling_envelope(*p, n_time=500)
    k = min(ref["constraint"], int(got["constraint"]))
    if k < 40:
        pytest.skip(f"curve too short to have a bulk and a tail (constraint {k})")

    def worst(j):
        out = 0.0
        for key, name in (("L", "bolometric_luminosity"), ("T", "photosphere_temperature"),
                          ("Rph", "photosphere_radius"), ("Me", "envelope_mass")):
            a = np.asarray(got[name], dtype=np.float64)[:j]
            b = ref[key][:j]
            out = max(out, np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        return out

    # (a) the bulk is at round-off ...
    assert worst(int(0.8 * k)) < 1e-11, f"drift starts too early: {worst(int(0.8 * k)):.2e}"
    # ... (b) and the trim point is still clean, so it is not positioned to hide anything
    assert worst(_trim(k)) < 1e-9, f"trim point already diverged: {worst(_trim(k)):.2e}"


# ================================================================= claim 3: version presets
def test_presets_are_the_documented_difference():
    """What the two redbacks differ by, and what is NOT a preset.

    The grid and the (1+z) factor are version differences and are reproducible. The
    termination GUARDS are not offered, because both versions of them are artefacts of
    redback's exception handling rather than statements about the model -- see CHANGE 3.
    """
    assert T.REDBACK_ENGINE_PRESETS["1.12"] == dict(n_time=5000)
    assert T.REDBACK_ENGINE_PRESETS["1.15"] == dict(n_time=500)
    assert T.REDBACK_PRESETS["1.12"]["dilation"] is False
    assert T.REDBACK_PRESETS["1.15"]["dilation"] is True
    assert "termination" not in T.REDBACK_ENGINE_PRESETS["1.12"]


def test_default_n_time_follows_the_installed_redback(monkeypatch):
    """1.2: the default grid is the installed redback's -- 500 for 1.15 and 1.20, and when
    redback is absent (the latest release); 5000 only for 1.12. It was 1.12's 5000 whatever was
    installed, which put the port up to 19.6 mag from redback 1.20 where the envelope ends."""
    from whisper_cbpf.models import redback_adapter as RA

    assert T.REDBACK_ENGINE_PRESETS["1.20"] == dict(n_time=500)
    assert T.REDBACK_PRESETS["1.20"]["dilation"] is True
    for release, n in (("1.12", 5000), ("1.15", 500), ("1.20", 500), (None, 500)):
        monkeypatch.setattr(RA, "installed_redback_preset", lambda r=release: r)
        assert T.default_n_time() == n, release
    monkeypatch.undo()

    pytest.importorskip("redback")
    want = T.REDBACK_ENGINE_PRESETS[_preset_for_installed_redback()]["n_time"]
    assert T.default_n_time() == want
    got = T.cooling_envelope(*CASES[0])
    assert got["time_temp"].shape == (want,)
    ref = T.cooling_envelope(*CASES[0], n_time=want)
    assert np.array_equal(np.asarray(got["bolometric_luminosity"]),
                          np.asarray(ref["bolometric_luminosity"]))


@pytest.mark.parametrize("model", ["cooling_envelope", "gaussianrise_cooling_envelope"])
def test_fallback_prior_is_the_latest_redbacks_file(model):
    """Without redback the prior is a transcription, and it must be the LATEST release's -- the
    one ``default_n_time`` already follows when redback is absent. It was 1.15.1's, whose
    ``cooling_envelope.prior`` pins four of five parameters; redback 1.18 (and so 1.20) frees them
    with ranges of its own, so the same factory call had one free parameter without redback and
    five with it. Checked against the installed file when that file IS the latest release."""
    from whisper_cbpf.models.redback_adapter import LATEST_REDBACK_PRESET

    pytest.importorskip("redback")
    if _preset_for_installed_redback() != LATEST_REDBACK_PRESET:
        pytest.skip("the transcription is of the latest redback; this one is older")
    rb, rb_pinned = T.redback_prior(model)
    fb, fb_pinned = T.fallback_prior(model)
    assert rb_pinned == fb_pinned, model
    assert set(rb.distributions) == set(fb.distributions), model
    for k, d in rb.distributions.items():
        assert type(d).__name__ == type(fb.distributions[k]).__name__, f"{model}.{k}"
        assert d.bounds == pytest.approx(fb.distributions[k].bounds, rel=1e-12), f"{model}.{k}"


def test_factory_default_matches_the_installed_redback():
    """``tde_model``'s default reproduces the installed redback's envelope, end included."""
    pytest.importorskip("redback")
    import whisper_cbpf

    lam = np.geomspace(1000.0, 30000.0, 1000)
    fs = {"lam": lam, "trans": np.stack([((lam > 3000) & (lam < 4000)).astype(float),
                                         ((lam > 5000) & (lam < 6000)).astype(float)])}
    n = T.REDBACK_ENGINE_PRESETS[_preset_for_installed_redback()]["n_time"]
    default = whisper_cbpf.tde_model(["a", "b"], 0.05, 7e26, rise="none", filter_set=fs)
    explicit = whisper_cbpf.tde_model(["a", "b"], 0.05, 7e26, rise="none", filter_set=fs,
                                      n_time=n)
    t = np.geomspace(1.0, 300.0, 24)
    b = np.array(["a", "b"] * 12)
    for p in CASES:
        par = {k: v for k, v in zip(PNAMES, p) if k in default.parameters}
        assert np.array_equal(default.predict(par, t, b), explicit.predict(par, t, b)), p


def test_default_grid_stability_probe():
    """THE RISK OF THE 500-POINT DEFAULT, measured: a 1e-6 nudge of ``mbh_6`` (1.2's risk line).

    ``OPEN_ITEMS.md`` warned "do not fit at 500" (0.55 mag under a 1e-6 relative change in
    ``mbh_6``): the termination index moves by grid steps under such a nudge, so at a FIXED epoch
    near the envelope's end the magnitude steps by the whole ``mag_floor`` gap. Probed at fixed
    epochs, fractions of the nominal life, over ``mbh_6 (1 + k 1e-7)``, ``|k| <= 10``, AB magnitude
    at 6e14 Hz (zero flux read as ``mag_floor`` = 40). Max spread over ``CASES``:

        fraction of the life     0.05    0.25    0.5     0.75    0.9     0.95    0.99
        port, n_time=500         1.2e-6  1.4e-6  2.7e-6  8.8e-6  7.9e-3  23.7    24.2
        redback 1.20 (500)       1.2e-6  1.4e-6  2.7e-6  8.8e-6  1.4e-2  24.2    24.7
        port, n_time=5000        1.4e-6  1.3e-6  2.5e-6  7.7e-6  2.3e-5  4.9e-5  0.31

    (jitted; the last two columns are the chaotic tail, where an XLA scheduling change alone
    moves the numbers -- CHANGE 1's unroll note.)

    So the default is smooth where the envelope is and a ``mag_floor``-sized step where it ends,
    and redback 1.20 has the same step at the same epochs: parity inherits it, the port adds
    nothing. ``n_time=5000`` pushes it into the last per cent. Asserted: both grids smooth in the
    bulk, the step bounded by ``mag_floor``, and -- where redback is installed at 500 points --
    redback just as smooth and just as stepped.
    """
    fracs = np.array([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    ks = np.arange(-10, 11) * 1e-7
    z, dl, nu = 0.05, 7e26, 6e14

    def mag(mjy):
        mjy = np.asarray(mjy, dtype=float)
        return np.where(mjy > 0, -2.5 * np.log10(np.maximum(mjy, 1e-300) * 1e-3 / 3631.0), 40.0)

    def probe(n, flux):
        spread = []
        for p in CASES:
            o = T.cooling_envelope(*p, n_time=n)
            if int(o["constraint"]) < 20:
                continue
            t = fracs * float(o["termination_time"]) * (1 + z) / DAY
            m = np.array([mag(flux(t, (p[0] * (1 + k), *p[1:]))) for k in ks]).clip(max=40.0)
            spread.append(m.max(0) - m.min(0))
        s = np.max(spread, axis=0)
        print(f"\n[n_time={n}] max |dmag| under 1e-6 in mbh_6, by fraction of the life: "
              + ", ".join(f"{f:.2f}: {x:.2e}" for f, x in zip(fracs, s)))
        return s

    port = {n: probe(n, jax.jit(lambda t, v, n=n: T.cooling_envelope_flux_density(
        t, nu, z, dl, *v, n_time=n))) for n in (500, 5000)}
    assert np.all(port[500][fracs <= 0.75] < 1e-4), port[500]
    assert port[500][fracs == 0.9][0] < 0.05, port[500]
    assert np.all(port[500] < 40.0 - 10.0), port[500]       # bounded by the mag_floor step
    assert np.all(port[5000][fracs <= 0.95] < 1e-3), port[5000]

    pytest.importorskip("redback")
    if T.REDBACK_ENGINE_PRESETS[_preset_for_installed_redback()]["n_time"] != 500:
        return
    from redback.transient_models.tde_models import cooling_envelope

    rb = probe(500, lambda t, v: cooling_envelope(t, z, *v, output_format="flux_density",
                                                   frequency=np.full(t.size, nu)))
    assert np.all(rb[fracs <= 0.75] < 1e-4), rb
    assert np.all(rb[fracs >= 0.95] > 1.0), rb          # redback steps where the port does


def test_termination_is_the_models_own_criterion():
    """CHANGE 3: the curve stops where the ENVELOPE stops, and `valid` proves it.

    Not one point inside `valid` may have a dead envelope. That is the property redback
    cannot state -- its stopping index is read off a trajectory that has already diverged,
    so it keeps between one and three post-mortem steps.
    """
    for p in CASES:
        o = T.cooling_envelope(*p, n_time=500)
        v = np.asarray(o["valid"])
        if not v.any():
            continue
        assert np.all(np.asarray(o["envelope_energy"])[v] > 0), p
        assert np.all(np.asarray(o["envelope_mass"])[v] > 0), p
        k = int(o["constraint"])
        if k < 500:                       # it stopped, so the stopping condition must hold there
            Rv = float(np.asarray(o["envelope_radius"])[k])
            Ee = float(np.asarray(o["envelope_energy"])[k])
            Me = float(np.asarray(o["envelope_mass"])[k])
            assert (Ee <= 0) or (Me <= 0) or (Rv < float(o["rcirc"]) / 2), (p, Rv, Ee, Me)


# ================================================================= CHANGE 6: the grid IS the model
def test_n_time_error_is_first_order_in_the_step():
    """Forward Euler: halving the step must roughly halve the error. This is what makes
    ``n_time`` part of the model rather than a tolerance -- redback 1.15.1's default of 500
    is ~1.7% from converged in the photosphere temperature."""
    p = CASES[0]
    ref = T.cooling_envelope(*p, n_time=20000)
    kr = int(ref["constraint"])
    t_ref = np.asarray(ref["time_since_fb"])[:kr]
    probe = np.geomspace(t_ref[1], t_ref[int(0.8 * kr)], 30)
    T_ref = np.interp(probe, t_ref, np.asarray(ref["photosphere_temperature"])[:kr])

    errs = []
    for n in (500, 1000, 2000):
        o = T.cooling_envelope(*p, n_time=n)
        k = int(o["constraint"])
        tt = np.asarray(o["time_since_fb"])[:k]
        got = np.interp(probe, tt, np.asarray(o["photosphere_temperature"])[:k])
        errs.append(np.max(np.abs(got - T_ref) / T_ref))

    assert errs[0] > 5e-3, f"n=500 should be ~1.7% off, got {errs[0]:.2e}"
    for a, b in zip(errs, errs[1:]):
        assert 1.5 < a / b < 3.0, f"not first order: {errs}"


# ================================================================= claim 2: differentiability
@pytest.mark.parametrize("p", CASES)
def test_gradients_finite_through_photosphere(p):
    """REGRESSION TEST for the defect this module exists to fix.

    Deliberately does NOT test ``bolometric_luminosity``: L depends on the envelope radius
    only through its INITIAL value, so it stays finite even when every other output has
    gone NaN -- which is exactly why the bug survived the reference port's own smoke test.
    """
    t_obs = jnp.asarray(np.geomspace(1.0, 200.0, 15) * DAY)

    for key in ("photosphere_temperature", "photosphere_radius",
                "envelope_radius", "envelope_mass"):
        def loss(v, key=key):
            o = T.cooling_envelope(v[0], v[1], v[2], v[3], v[4], n_time=500)
            n = o["time_since_fb"].shape[0]
            idx = jnp.minimum(jnp.arange(n), jnp.maximum(o["constraint"] - 1, 0))
            return jnp.sum(jnp.interp(t_obs, o["time_since_fb"][idx], o[key][idx]))

        g = np.asarray(jax.grad(loss)(jnp.asarray(p, dtype=jnp.float64)))
        assert np.all(np.isfinite(g)), f"{key}: non-finite gradient {g}"
        assert np.any(g != 0.0), f"{key}: gradient is identically zero -- nothing is flowing"


def test_gradients_agree_with_finite_differences():
    """Finite and CORRECT are different claims; this is the second one.

    The observable is the light curve interpolated onto FIXED epochs, which is what a
    likelihood computes. A masked *sum* would not do: its summation limit moves with the
    parameters, so central differences see points enter and leave the sum and disagree with
    AD by design -- that is a property of that objective, not of the model.
    """
    p0 = jnp.asarray(CASES[0], dtype=jnp.float64)
    t_obs = jnp.asarray(np.geomspace(1.0, 100.0, 12) * DAY)

    def loss(v):
        o = T.cooling_envelope(v[0], v[1], v[2], v[3], v[4], n_time=500)
        n = o["time_since_fb"].shape[0]
        idx = jnp.minimum(jnp.arange(n), jnp.maximum(o["constraint"] - 1, 0))
        return jnp.sum(jnp.interp(t_obs, o["time_since_fb"][idx],
                                  o["photosphere_temperature"][idx]))

    g = np.asarray(jax.grad(loss)(p0))
    for j, name in enumerate(PNAMES):
        h = 1e-6 * float(p0[j])
        fd = float((loss(p0.at[j].add(h)) - loss(p0.at[j].add(-h))) / (2 * h))
        rel = abs(g[j] - fd) / max(abs(fd), 1e-30)
        assert rel < 1e-3, f"d/d{name}: AD {g[j]:.6e} vs FD {fd:.6e} (rel {rel:.2e})"


def test_float32_is_refused_with_an_actionable_message():
    """CHANGE 8. In float32 the Euler increments are ~1e-6 of the state, below the 1.2e-7
    epsilon: measured, ``constraint`` collapses to 1 and the luminosity is inf. Returning
    that would be worse than raising."""
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="requires float64"):
            T.cooling_envelope(*CASES[0], n_time=100)
    finally:
        jax.config.update("jax_enable_x64", was)


# ================================================================= photometry
def _flat_filters():
    lam = np.geomspace(1000.0, 30000.0, 1000)
    trans = np.zeros((2, lam.size))
    trans[0][(lam > 3000) & (lam < 4000)] = 1.0
    trans[1][(lam > 5000) & (lam < 6000)] = 1.0
    W, N = T.ab_weights(lam, trans)
    return jnp.asarray(lam), W, N


def test_dilation_is_exactly_one_plus_z():
    """CHANGE 7: the whole difference between the two redbacks' photometry is this factor."""
    lam, W, N = _flat_filters()
    t = jnp.asarray(np.geomspace(1.0, 100.0, 8))
    for z in (0.01, 0.1, 0.5):
        dl = 1e27
        a = np.asarray(T.cooling_envelope_flux_density(t, 6e14, z, dl, *CASES[0],
                                                       n_time=500, dilation=False))
        b = np.asarray(T.cooling_envelope_flux_density(t, 6e14, z, dl, *CASES[0],
                                                       n_time=500, dilation=True))
        assert np.allclose(b / a, 1.0 + z, rtol=1e-12), (z, (b / a)[:3])


def test_flux_density_matches_redback():
    pytest.importorskip("redback")
    from redback.transient_models.tde_models import _cooling_envelope, cooling_envelope
    astropy_cosmo = pytest.importorskip("astropy.cosmology")

    tag = _preset_for_installed_redback()
    preset = T.REDBACK_PRESETS[tag]
    z, nu = 0.05, 6.0e14
    dl = astropy_cosmo.Planck18.luminosity_distance(z).cgs.value

    for p in CASES[:3]:
        rb_out = _cooling_envelope(*p)
        tmax = rb_out.time_since_fb[-1] * (1 + z) / DAY
        t_obs = np.geomspace(max(tmax * 1e-3, 0.5), tmax * 0.9, 12)
        want = cooling_envelope(t_obs, z, p[0], p[1], p[2], p[3], p[4],
                                output_format="flux_density",
                                frequency=np.full(t_obs.size, nu))
        got = np.asarray(T.cooling_envelope_flux_density(
            jnp.asarray(t_obs), nu, z, dl, *p, **preset))
        rel = np.max(np.abs(got - want) / np.abs(want))
        assert rel < 2e-3, f"{p}: max rel {rel:.3e}"


#: Parameter sets that the tests above do not reach, kept because each one once failed.
#: Both are inside redback's own prior and both produced a finite, smooth, ordinary-looking
#: likelihood for a light curve that does not exist.
REGRESSIONS = {
    # envelope dies 0.0155 d after fallback -> constraint == 1 at n_time=5000. The clamped
    # index collapsed the time axis to one point and jnp.interp returned (T[0], R[0]) at
    # every epoch: a flat 400-day curve at AB 20.5. 9.85% of the prior at n_time=500.
    "constraint_one": (0.010498878379993507, 8.344290585513178, 0.054451670132515374,
                       0.7370240289465599, 1.931913362151394),
    # Rph = -5.18e16 cm at an ISOLATED INTERIOR index (3820 of 3824), where the envelope
    # goes optically thin. redback squares it into a blackbody and reports 0.465 mJy from a
    # negative radius; interpolating through it moved the AB magnitude by up to 3.89 mag.
    "negative_rph": (0.1797195540229069, 5.911796441834461, 0.020097471754105388,
                     0.23263844467821457, 2.084113453497437),
}


def test_short_lived_envelope_is_not_a_flat_light_curve():
    """REGRESSION. A one-point light curve must be zero flux, not a 400-day plateau.

    The failure this guards against was finite, differentiable and bright (AB 20.5), i.e.
    perfectly capable of being fitted -- which is what makes it worse than a NaN.
    """
    p = REGRESSIONS["constraint_one"]
    o = T.cooling_envelope(*p, n_time=5000)
    assert int(o["constraint"]) < 2, f"pick a new regression case: constraint {int(o['constraint'])}"

    lam, W, N = _flat_filters()
    t = jnp.asarray(np.geomspace(5.0, 400.0, 8))
    mag = np.asarray(T.cooling_envelope_ab_magnitude(t, jnp.zeros(8, dtype=int), W, N, lam,
                                                     0.05, 7e26, *p, n_time=5000))
    assert np.allclose(mag, 40.0), f"expected mag_floor, got {mag}"
    fd = np.asarray(T.cooling_envelope_flux_density(t, 6e14, 0.05, 7e26, *p, n_time=5000))
    assert np.all(fd == 0.0), fd
    # The Gaussian RISE keeps its normaliser, the envelope's first sample (redback's; see
    # test_gaussian_rise_survives_an_envelope_whose_integration_dies_at_its_first_step), so only
    # the epochs after the stitch are the floor -- and before it the rise is a Gaussian, not a
    # plateau. (This asserted the floor at every epoch while the rise needed two samples.)
    gr = np.asarray(T.gaussianrise_cooling_envelope_ab_magnitude(
        t, jnp.zeros(8, dtype=int), W, N, lam, 0.05, 7e26, 20.0, 15.0, *p, n_time=5000))
    after = np.asarray(t) >= float(o["tfb"]) * 1.05 / DAY
    assert 0 < after.sum() < t.size, after
    assert np.all(gr[after] == 40.0), gr
    assert np.all(gr[~after] < 40.0) and np.ptp(gr[~after]) > 0.1, gr


def test_zero_flux_outside_the_models_own_time_span():
    """Past the last valid epoch the model says nothing, so the flux is zero -- not the
    saturated edge value, which reads as a source that simply stopped changing."""
    p = CASES[1]
    o = T.cooling_envelope(*p, n_time=500)
    k = int(o["constraint"])
    t_end = float(np.asarray(o["time_since_fb"])[k - 1]) / DAY
    t = jnp.asarray([0.5 * t_end, 0.99 * t_end, 1.5 * t_end, 10.0 * t_end])
    fd = np.asarray(T.cooling_envelope_flux_density(t, 6e14, 0.05, 7e26, *p, n_time=500))
    assert np.all(fd[:2] > 0), fd
    assert np.all(fd[2:] == 0.0), fd


def test_negative_photosphere_radius_never_reaches_the_blackbody():
    """REGRESSION. Rph < 0 must not be squared into a flux.

    It happens where the envelope goes optically thin (``Lamb < 1/e``, so
    ``Rph = Rv(1 + ln Lamb) < 0``), at ISOLATED INTERIOR indices rather than in a tail.
    redback reports a positive flux from a negative radius there; interpolating through it
    moved the AB magnitude by up to 3.89 mag. Clamping at zero is continuous (``Rph -> 0``
    as ``Lamb -> 1/e``) and says the physical thing: no photosphere, no thermal emission.

    Written as a PROPERTY over a seeded sweep rather than against one pinned parameter set.
    The first version of this test pinned the draw an auditor found, and its premise stopped
    holding on a different backend -- ``constraint`` moved by 4 steps, which is inside the
    conditioning this model has anyway (see _trim). A regression test that depends on a
    chaotic index is not a regression test.
    """
    rng = np.random.default_rng(4)
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))     # noqa: E731
    seen_negative = 0
    for _ in range(40):
        p = (lu(0.01, 20), lu(0.1, 10), lu(1e-4, 0.1), lu(0.1, 1.0), float(rng.uniform(1, 5)))
        o = T.cooling_envelope(*p, n_time=500)
        k = int(o["constraint"])
        if k < 2:
            continue
        rph = np.asarray(o["photosphere_radius"])
        neg = rph < 0
        seen_negative += int(neg.any())
        # `meaningful` must never contain a negative-radius epoch ...
        assert not np.asarray(o["meaningful"])[neg].any(), p
        # ... and no interpolated radius may be negative, whether or not this draw has one
        t_src = jnp.asarray(np.asarray(o["time_since_fb"])[:k])
        temp, rad = T._interp_photosphere(o, t_src)
        rad = np.asarray(rad)
        assert np.all(rad >= 0.0), (p, rad.min())
        fd = np.asarray(T.flux_density_mjy(temp, jnp.asarray(rad), 6e14, 0.05, 7e26))
        assert np.all(np.isfinite(fd)) and np.all(fd >= 0.0), p
    assert seen_negative > 0, "sweep never produced a negative Rph -- the test proves nothing"


def test_envelope_exists_predicts_constraint_zero_in_closed_form():
    """The degenerate case that IS excludable by a prior, and exactly which one.

    ``constraint == 0`` means the envelope is born inside its own circularisation radius --
    a statement about the parameters alone, with no integration in it, so a prior can carry
    it. The predicate must agree with the integrator, both ways.
    """
    rng = np.random.default_rng(11)
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))     # noqa: E731
    saw_dead = saw_alive = 0
    for _ in range(60):
        p = (lu(0.01, 20), lu(0.1, 10), lu(1e-4, 0.1), lu(0.1, 1.0), float(rng.uniform(1, 5)))
        k = int(T.cooling_envelope(*p, n_time=500)["constraint"])
        alive = bool(T.envelope_exists(p[0], p[1], p[4]))
        assert alive == (k > 0), (p, k, alive)
        saw_dead += int(not alive); saw_alive += int(alive)
    assert saw_alive > 0
    # and the boundary itself: beta * (Rt/Rstar) == 25 * bec
    mbh, mstar = 1.0, 1.0
    beta_crit = 20.0 / (mbh * 1e6 / mstar) ** (1 / 3)
    assert not T.envelope_exists(mbh, mstar, beta_crit * 0.999)
    assert T.envelope_exists(mbh, mstar, beta_crit * 1.001)


def test_rise_peaks_near_fallback_flags_the_absurd_regime():
    """A stitch far past the peak means a rise exp(n_sigma^2/2) brighter than the envelope."""
    assert T.rise_peaks_near_fallback(20.0, 15.0, 1.0, 1.0)            # 2.5 sigma, fine
    assert not T.rise_peaks_near_fallback(20.0, 15.0, 20.0, 10.0)      # 26.1 sigma
    assert not T.rise_peaks_near_fallback(0.1, 10.0, 20.0, 10.0)       # 41.1 sigma
    # one-sided: a peak AFTER the stitch is a legitimate monotonic rise
    assert T.rise_peaks_near_fallback(1e4, 10.0, 1.0, 1.0)


def test_band_magnitudes_are_physical():
    lam, W, N = _flat_filters()
    t = jnp.asarray(np.geomspace(1.0, 300.0, 25))
    z, dl = 0.05, 2.3e27
    for bi in (0, 1):
        mag = np.asarray(T.cooling_envelope_ab_magnitude(
            t, jnp.full(25, bi), W, N, lam, z, dl, *CASES[0], n_time=500))
        assert np.all(np.isfinite(mag))
        assert np.all((5.0 < mag) & (mag < 40.0)), mag


def test_colour_tracks_temperature_the_right_way():
    """A hotter blackbody must be BLUER. This is the sign check that catches a swapped
    band index, a wrong wavelength-to-frequency conversion, or an inverted Planck argument
    -- none of which a "magnitudes are finite" test would notice."""
    lam, W, N = _flat_filters()
    z, dl = 0.05, 2.3e27
    rad = jnp.full(4, 1e15)
    temps = jnp.asarray([5e3, 1e4, 3e4, 1e5])
    blue = np.asarray(T.ab_magnitude(temps, rad, jnp.zeros(4, dtype=int), W, N, lam, z, dl))
    red = np.asarray(T.ab_magnitude(temps, rad, jnp.ones(4, dtype=int), W, N, lam, z, dl))
    colour = blue - red
    assert np.all(np.diff(colour) < 0), f"hotter must be bluer, got colour {colour}"
    # and a bigger photosphere must be brighter, at fixed temperature, as R^2
    m1 = np.asarray(T.ab_magnitude(temps, rad, jnp.zeros(4, dtype=int), W, N, lam, z, dl))
    m2 = np.asarray(T.ab_magnitude(temps, 2 * rad, jnp.zeros(4, dtype=int), W, N, lam, z, dl))
    assert np.allclose(m1 - m2, 2.5 * np.log10(4.0), rtol=1e-10)


def test_gaussianrise_flux_density_never_overflows():
    """REGRESSION. The FLUX branch of the rise must not overflow where the magnitude one can't.

    Magnitude can represent an absurdly bright rise (-800 mag is a fine float); flux cannot
    represent exp(931), which is what the exponent reaches at the prior corner
    (peak_time = 0.1 d against tfb = 411 d). The analytic cancellation removed the NaN from
    the magnitude path and left it in the flux path, at essentially the same rate: 0.31% of
    the module's own default prior at z = 0.5 returned +inf with NaN/-inf gradients.
    """
    p = (20.0, 10.0, 0.05, 0.1, 1.0)                # the corner, inside redback's prior
    t = jnp.asarray([0.5, 1.0, 5.0, 20.0])
    fd = np.asarray(T.gaussianrise_cooling_envelope_flux_density(
        t, 6e14, 0.05, 7e26, 0.1, 10.0, *p, n_time=500))
    assert np.all(np.isfinite(fd)), fd
    assert np.all(fd >= 0.0), fd

    def loss(v):
        return jnp.sum(T.gaussianrise_cooling_envelope_flux_density(
            t, 6e14, 0.05, 7e26, v[0], v[1], *p, n_time=500))

    g = np.asarray(jax.grad(loss)(jnp.array([0.1, 10.0])))
    assert np.all(np.isfinite(g)), g

    # and the dead-envelope case, where f_stitch is exactly 0 and 0 * exp(huge) would be NaN
    dead = REGRESSIONS["constraint_one"]
    fd0 = np.asarray(T.gaussianrise_cooling_envelope_flux_density(
        t, 6e14, 0.05, 7e26, 0.1, 10.0, *dead, xi=5.0, n_time=5000))
    assert np.all(np.isfinite(fd0)) and np.all(fd0 == 0.0), fd0


def test_adapter_does_not_recompile_per_parameter_set():
    """REGRESSION. The engine bakes its parameters into the scan's jaxpr via ``partial``, so
    an unjitted ``predict`` is a cache miss -- and a fresh ~0.6 s XLA compile of a 5000-step
    scan -- on EVERY likelihood evaluation. Measured before the fix: 831 ms per call against
    9.1 ms jitted.

    Timed rather than introspected, because the jitted callable is a closure. The margin is
    ~30x, not 2x: ten calls take ~0.1 s jitted and ~8 s unjitted.
    """
    import time

    import whisper_cbpf

    m = whisper_cbpf.tde_model(["sdssg", "sdssr"], redshift=0.05, dl_cm=7e26,
                              n_wave=300, n_time=5000)
    t = np.linspace(1.0, 150.0, 12)
    b = np.array(["sdssg", "sdssr"] * 6)
    base = {"peak_time": 20.0, "sigma_t": 15.0, "mbh_6": 1.0, "stellar_mass": 1.0,
            "eta": 0.05, "alpha": 0.1, "beta": 1.0}
    m.predict(base, t, b)                                   # pay the one compile

    t0 = time.perf_counter()
    for i in range(10):
        m.predict({**base, "mbh_6": 1.0 + 0.01 * i}, t, b)  # a NEW parameter set each time
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, (
        f"10 predict() calls took {elapsed:.2f} s; that is a recompile per parameter set "
        f"(~0.6 s each), not a cache hit")


def test_gaussian_rise_is_continuous_at_the_stitch():
    """The rise is normalised to meet the envelope at ``xi * tfb``; a jump there would mean
    the normalisation is wrong."""
    lam, W, N = _flat_filters()
    z, dl = 0.05, 2.3e27
    out = T.cooling_envelope(*CASES[0], n_time=500)
    tfb_obs = float(out["tfb"]) * (1 + z) / DAY
    eps = tfb_obs * 1e-4
    t = jnp.asarray([tfb_obs - eps, tfb_obs, tfb_obs + eps])
    mag = np.asarray(T.gaussianrise_cooling_envelope_ab_magnitude(
        t, jnp.zeros(3, dtype=int), W, N, lam, z, dl, 20.0, 15.0, *CASES[0], n_time=500))
    assert np.all(np.isfinite(mag))
    assert np.max(np.abs(np.diff(mag))) < 0.02, f"discontinuous at the stitch: {mag}"


#: A ``gaussianrise_cooling_envelope.prior`` draw (redback 1.20; the worst prior draw
#: after the 0.1.1 fix) whose envelope integration dies after ONE step at n_time=500 -- the first
#: forward-Euler step drives ``Ee`` through zero -- and lives 2637 steps at 5000. Measured over
#: 4000 draws of that prior: 4.8% end at ``constraint == 1`` at 500, none at 5000.
FIRST_STEP_DEATH = (1.2766, 2.741, 0.0602, 0.7035, 4.8821)


def test_gaussian_rise_survives_an_envelope_whose_integration_dies_at_its_first_step():
    """REGRESSION, found against redback: at the 500-point default the rise vanished.

    redback normalises the rise at the envelope's FIRST sample (``photosphere_temperature[0]``,
    closed form, no integration in it), so its rise exists whenever the envelope is born. The
    port asked for two live samples before it had a normaliser, so on these draws it returned
    ``mag_floor`` at EVERY epoch -- 47.7 mag from redback 1.20 on the reference ZTF epochs, which
    all lie on the rise. Now the rise is the converged curve's (it never depends on the grid)
    and redback's; after the stitch the envelope is over (CHANGE 3): zero flux, ``mag_floor``.
    """
    p = FIRST_STEP_DEATH
    assert int(T.cooling_envelope(*p, n_time=500)["constraint"]) == 1
    assert int(T.cooling_envelope(*p, n_time=5000)["constraint"]) > 2000
    z, dl, nu, rise = 0.05, 7e26, 6e14, (20.0, 15.0)
    tfb_obs = float(T.calc_tfb(0.8, p[0], p[1])) * (1 + z) / DAY
    t = np.array([0.1, 0.5, 0.9, 1.5]) * tfb_obs
    pre = t < tfb_obs

    f = {n: np.asarray(T.gaussianrise_cooling_envelope_flux_density(
        jnp.asarray(t), nu, z, dl, *rise, *p, n_time=n)) for n in (500, 5000)}
    assert np.all(f[500][pre] > 0.0), f[500]
    assert _maxrel_np(f[500][pre], f[5000][pre]) < 1e-12
    assert np.all(f[500][~pre] == 0.0), f[500]
    lam, W, N = _flat_filters()
    m = {n: np.asarray(T.gaussianrise_cooling_envelope_ab_magnitude(
        jnp.asarray(t), jnp.zeros(t.size, dtype=int), W, N, lam, z, dl, *rise, *p, n_time=n))
        for n in (500, 5000)}
    assert np.max(np.abs(m[500][pre] - m[5000][pre])) < 1e-10, m
    assert np.all(m[500][~pre] == 40.0), m[500]
    # a stitch PAST the envelope's one sample still has nothing to normalise to
    late = np.asarray(T.gaussianrise_cooling_envelope_flux_density(
        jnp.asarray(t), nu, z, dl, *rise, *p, xi=2.0, n_time=500))
    assert np.all(late == 0.0), late

    pytest.importorskip("redback")
    from redback.transient_models.tde_models import gaussianrise_cooling_envelope
    astropy_cosmo = pytest.importorskip("astropy.cosmology")

    dl = astropy_cosmo.Planck18.luminosity_distance(z).cgs.value
    ours = np.asarray(T.gaussianrise_cooling_envelope_flux_density(
        jnp.asarray(t[pre]), nu, z, dl, *rise, *p, n_time=500))
    # redback evaluated at t (1+z): its double time dilation taken out. Its rise is
    # a linear interpolant through 200 nodes, hence the 1e-3.
    theirs = gaussianrise_cooling_envelope(t[pre] * (1 + z), z, *rise, *p,
                                           output_format="flux_density",
                                           frequency=np.full(pre.sum(), nu))
    assert _maxrel_np(ours, theirs) < 1e-3, (ours, theirs)


def _maxrel_np(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


# ================================================================= the smaller ports
def test_analytic_fallback_plateaus_then_decays_as_five_thirds():
    t = jnp.asarray(np.geomspace(0.1, 500.0, 200))
    l0, t0 = 1e50, 10.0
    lb = np.asarray(T.analytic_fallback(t, l0, t0))
    tn = np.asarray(t)
    assert np.allclose(lb[tn < t0], l0 / (t0 * DAY) ** (5 / 3))
    late = tn > 2 * t0
    slope = np.polyfit(np.log(tn[late]), np.log(lb[late]), 1)[0]
    assert abs(slope + 5 / 3) < 1e-8, slope


def test_analytic_fallback_survives_nonpositive_times():
    """A grid that starts at the trigger contains t = 0; t**(-5/3) is NaN there and its
    derivative is infinite, so the unselected branch has to be sanitised."""
    t = jnp.asarray([-5.0, 0.0, 1.0, 50.0])
    lb = np.asarray(T.analytic_fallback(t, 1e50, 10.0))
    assert np.all(np.isfinite(lb))
    g = jax.grad(lambda x: jnp.sum(T.analytic_fallback(t, x, 10.0)))(1e50)
    assert np.isfinite(float(g))


def test_stream_stream_collision_is_finite_and_five_thirds():
    for inc in (0, 1):
        t, L, Temp, t_dyn, t_peak, r_t = T.stream_stream_collision(
            1.0, 1.0, 0.1, 0.5, 0.3, 1.0, inc_tcool=inc)
        for a in (L, Temp):
            assert np.all(np.isfinite(np.asarray(a))) and np.all(np.asarray(a) > 0)
        slope = np.polyfit(np.log(np.asarray(t)), np.log(np.asarray(L)), 1)[0]
        assert abs(slope + 5 / 3) < 1e-6, slope
        assert float(t_peak) == pytest.approx(1.5 * float(t_dyn))
    # the cooling correction can only reduce the luminosity
    l_off = T.stream_stream_collision(1.0, 1.0, 0.1, 0.5, 0.3, 1.0, inc_tcool=0)[1]
    l_on = T.stream_stream_collision(1.0, 1.0, 0.1, 0.5, 0.3, 1.0, inc_tcool=1)[1]
    assert np.all(np.asarray(l_on) <= np.asarray(l_off))


@pytest.mark.parametrize("vej,tf", [(1e4, 6000.0), (3e3, 1000.0)])
def test_temperature_floor_photosphere_matches_redback(vej, tf):
    photosphere = pytest.importorskip("redback.photosphere")
    t = np.geomspace(0.1, 300.0, 200)
    lum = 1e43 * (t / 10.0) ** -1.5
    lum[:20] = 0.0                                   # redback's zero-padded region
    ref = photosphere.TemperatureFloor(time=t, luminosity=lum, vej=vej, temperature_floor=tf)
    got_T, got_R = T.temperature_floor_photosphere(jnp.asarray(t), jnp.asarray(lum), vej, tf)
    assert np.allclose(np.asarray(got_T), ref.photosphere_temperature, rtol=1e-12)
    assert np.allclose(np.asarray(got_R), ref.r_photosphere, rtol=1e-12)


def test_tde_photosphere_matches_redback():
    photosphere = pytest.importorskip("redback.photosphere")
    t = np.geomspace(0.1, 300.0, 200)
    lum = 1e43 * (t / 10.0) ** -1.5
    lum[:20] = 0.0
    kw = dict(mass_bh=1e6, mass_star=1.0, star_radius=1.0, tpeak=20.0, beta=1.0,
              rph_0=1.0, lphoto=1.0)
    ref = photosphere.TDEPhotosphere(time=t, luminosity=lum, **kw)
    got_T, got_R, got_rp = T.tde_photosphere(jnp.asarray(t), jnp.asarray(lum), **kw)
    assert np.allclose(np.asarray(got_R), ref.r_photosphere, rtol=1e-12)
    ok = lum > 0                                     # redback returns 0**0.25 = 0 where L = 0
    assert np.allclose(np.asarray(got_T)[ok], ref.photosphere_temperature[ok], rtol=1e-12)
    assert float(got_rp) == pytest.approx(ref.rp, rel=1e-12)


def test_photospheres_have_finite_gradients_at_zero_luminosity():
    """redback's TDE luminosities contain exact zeros, and ``0**0.25`` differentiates to
    infinity. The floor is what keeps a zero-flux epoch from poisoning the whole gradient."""
    t = jnp.asarray(np.geomspace(0.1, 300.0, 50))
    lum = jnp.asarray(np.concatenate([np.zeros(10), 1e43 * np.geomspace(1, 0.01, 40)]))
    g1 = jax.grad(lambda v: jnp.sum(T.temperature_floor_photosphere(t, lum, v[0], v[1])[0])
                  )(jnp.array([1e4, 6000.0]))
    g2 = jax.grad(lambda v: jnp.sum(T.tde_photosphere(t, lum, v[0], v[1], v[2], v[3],
                                                       1.0, v[4], v[5])[0])
                  )(jnp.array([1e6, 1.0, 1.0, 20.0, 1.0, 1.0]))
    assert np.all(np.isfinite(np.asarray(g1))), g1
    assert np.all(np.isfinite(np.asarray(g2))), g2


def test_diffusion_matches_redback():
    ip = pytest.importorskip("redback.interaction_processes")
    dense = np.geomspace(0.1, 300.0, 400)
    lum = 1e43 * np.exp(-dense / 40.0)
    t = np.geomspace(0.5, 250.0, 37)
    kw = dict(kappa=0.2, kappa_gamma=0.03, mej=5.0, vej=1e4)
    ref = ip.Diffusion(time=t, dense_times=dense, luminosity=lum, **kw).new_luminosity
    ut, gi, tb = T.build_interaction_grid(t, dense)
    got = np.asarray(T.diffusion(ut, gi, tb, jnp.asarray(dense), jnp.asarray(lum), **kw))
    assert np.allclose(got, ref, rtol=1e-9), np.max(np.abs(got / ref - 1))


def test_viscous_matches_redback():
    ip = pytest.importorskip("redback.interaction_processes")
    dense = np.geomspace(0.1, 300.0, 400)
    lum = 1e43 * np.exp(-dense / 40.0)
    t = np.geomspace(0.5, 250.0, 37)
    ref = ip.Viscous(time=t, dense_times=dense, luminosity=lum, t_viscous=5.0).new_luminosity
    ut, gi, tb = T.build_interaction_grid(t, dense)
    got = np.asarray(T.viscous(ut, gi, tb, jnp.asarray(dense), jnp.asarray(lum), 5.0))
    assert np.allclose(got, ref, rtol=1e-9), np.max(np.abs(got / ref - 1))


def test_csm_diffusion_does_not_overflow_where_redback_does():
    """The exponent is folded so it can never be positive. redback's form evaluates
    ``exp(t'/t0)`` on its own, which overflows float64 for a small enough ``t0`` -- a value
    an ordinary prior reaches."""
    dense = np.geomspace(0.1, 300.0, 400)
    lum = 1e43 * np.exp(-dense / 40.0)
    t = np.geomspace(0.5, 250.0, 37)
    ut, gi, tb = T.build_interaction_grid(t, dense)
    kw = dict(kappa=0.2, r_photosphere=1e16, mass_csm_threshold=1e30)
    got = np.asarray(T.csm_diffusion(ut, gi, tb, jnp.asarray(dense), jnp.asarray(lum), **kw))
    assert np.all(np.isfinite(got)), "overflowed"

    t0 = kw["kappa"] * kw["mass_csm_threshold"] / (
        4 * np.pi ** 3 / 9 * T.SPEED_OF_LIGHT * kw["r_photosphere"]) / DAY
    with np.errstate(over='ignore'):
        assert not np.isfinite(np.exp(dense[-1] / t0)), (
            f"pick a smaller t0 for this test to be meaningful (t0 = {t0:.3e} d)")


def test_build_interaction_grid_warns_instead_of_crashing_out_of_range():
    dense = np.geomspace(1.0, 100.0, 50)
    with pytest.warns(RuntimeWarning, match="outside the dense grid"):
        ut, gi, tb = T.build_interaction_grid(np.array([0.5, 10.0, 500.0]), dense)
    assert int(np.asarray(gi).max()) <= int(np.asarray(ut).size) - 1
