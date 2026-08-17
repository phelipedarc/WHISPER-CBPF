"""Tests for the generic redback adapter, :mod:`whisper_cbpf.models.redback_adapter`.

One module binds every redback model, so this one file replaces ``test_tde_redback.py`` (11 tests)
and ``test_supernova_redback.py`` (22). The claims are the adapter's, not any one model's, and each
is checked through at least two *different* redback models so that nothing can pass by accident of
``arnett``'s or ``cooling_envelope``'s particular shape:

1. **Parameters and priors are read from redback, never transcribed.** ``redback_parameters`` must
   recover the four parameters ``arnett`` hides in ``**kwargs``, and ``redback_prior`` must equal
   ``redback.priors.get_priors`` distribution by distribution -- with ``DeltaFunction`` entries
   becoming *pinned* parameters and ``Constraint`` entries dropped.
2. **It is a thin adapter with the right units.** ``test_mjy_to_jy_is_exactly_redbacks_own_mjy``
   pins the 1e-3 against redback's own return value on the identical call, for two models, so a
   1000x slip cannot pass as a plausible offset on a log plot.
3. **One call, never a per-band loop** (``arnett``'s diffusion grid depends on ``time[-1]``).
4. **Out-of-domain epochs are zero and the rest of the band is not** -- the ``cooling_envelope``
   regression that blanked 9.83% of prior draws.
5. **It agrees with the independent JAX ports** for both models, to float64 round-off.
6. **It generalises.** ``basic_magnetar_powered`` and ``one_component_kilonova_model`` -- neither of
   which ever had a wrapper -- bind and produce finite positive Jy with no new code.
7. **It survives ``pickle``** (multiprocess ABC) and reports usable errors without redback.
"""
from __future__ import annotations

import builtins
import importlib
import pickle
import sys

import numpy as np
import pytest

from whisper_cbpf.models import get_model, list_models
from whisper_cbpf.models import redback_adapter as RA
from whisper_cbpf.priors import LogUniform, Prior, Uniform

# --- arnett (supernova) --------------------------------------------------------------------------
SN = "arnett"
SN_PARAMS = ["redshift", "f_nickel", "mej", "vej", "kappa", "kappa_gamma", "temperature_floor"]
SN_FREE6 = [p for p in SN_PARAMS if p != "redshift"]
Z = 0.0098
#: Inside redback's ``arnett.prior`` everywhere.
SN_SAMPLE = dict(f_nickel=0.1, mej=2.0, vej=1e4, kappa=0.1, kappa_gamma=0.03,
                 temperature_floor=4000.0)
#: Four corners of the prior. The third is deliberately cold and thin (the SED is steep across a
#: band there), which is where a monochromatic-vs-band-integral difference is largest.
SN_CASES = [
    dict(f_nickel=0.1, mej=2.0, vej=1e4, kappa=0.1, kappa_gamma=0.03, temperature_floor=4000.0),
    dict(f_nickel=0.5, mej=10.0, vej=5e3, kappa=0.2, kappa_gamma=0.01, temperature_floor=8000.0),
    dict(f_nickel=1e-3, mej=1e-2, vej=3e4, kappa=1.5, kappa_gamma=1e3, temperature_floor=1500.0),
    dict(f_nickel=0.9, mej=50.0, vej=1e3, kappa=0.05, kappa_gamma=1e-3, temperature_floor=2e4),
]
T_OBS = np.geomspace(0.5, 60.0, 45)
BANDS = np.array([["g", "r", "i"][i % 3] for i in range(T_OBS.size)])

# --- cooling_envelope (TDE) ----------------------------------------------------------------------
TDE = "cooling_envelope"
TDE_PHYS = ["mbh_6", "stellar_mass", "eta", "alpha", "beta"]
ZT = 0.05
#: (mbh_6, stellar_mass, eta, alpha, beta) -- the same cases ``test_tde_vs_redback.py`` uses, so a
#: disagreement here can be read against that file's numbers.
TDE_CASES = [
    (1.0, 1.0, 0.05, 0.1, 1.0),
    (5.0, 0.5, 0.03, 0.2, 1.5),
    (0.5, 3.0, 0.08, 0.05, 0.8),
    (10.0, 0.2, 0.1, 0.5, 2.0),
]
#: redback's ``cooling_envelope.prior`` pins ``mbh_6``/``eta``/``alpha``/``beta`` with
#: ``DeltaFunction`` entries, so the adapter pins them too -- that is what a delta prior *means*, and
#: ``test_delta_prior_entries_become_pinned_parameters`` asserts it. To vary all five (which parity
#: needs) the caller unpins them with bounds it can defend; these are redback 1.15.1's
#: ``gaussianrise_cooling_envelope`` ranges, the shipped TDE prior that leaves all five free.
TDE_UNPIN = Prior({"mbh_6": LogUniform(0.1, 20.0), "eta": LogUniform(1e-4, 0.1),
                   "alpha": LogUniform(0.1, 1.0), "beta": Uniform(1.0, 5.0)})
DAY = 86400.0


def _maxrel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _sn_model(**kw):
    return RA.redback_model(SN, ["g", "r", "i"], redshift=Z, **kw)


def _tde_model(**kw):
    return RA.redback_model(TDE, ["g", "r", "i"], redshift=ZT, prior=TDE_UNPIN, **kw)


def _tde_pars(case):
    return dict(zip(TDE_PHYS, case))


def _preset_for_installed_redback():
    """Which redback is importable, and therefore which JAX preset must be compared to it.

    Keyed on the SOURCE, not ``__version__``: 1.15.1 reports "unknown" in this environment. The
    ``f_debris`` rewrite is the same commit that took the TDE grid from 5000 points to 500.
    Identical to ``tests/test_tde_vs_redback.py``'s helper, deliberately -- they must not disagree.
    """
    import inspect

    from redback.transient_models import tde_models
    return "1.15" if "f_debris" in inspect.getsource(tde_models._cooling_envelope) else "1.12"


def _epochs_inside_envelope(case, n, z=ZT):
    """Observer-frame days since fallback, inside the envelope's own span.

    redback's ``interp1d`` carries no ``fill_value`` and raises outside it; the adapter turns that
    into zero flux, which is correct behaviour but useless for a parity measurement.
    """
    from redback.transient_models import tde_models
    tmax = tde_models._cooling_envelope(*case).time_since_fb[-1] * (1 + z) / DAY
    return np.geomspace(max(tmax * 1e-3, 0.5), tmax * 0.9, n)


# --- CLAIM 1: parameters and priors are redback's ------------------------------------------------

def test_parameters_include_the_ones_hidden_in_kwargs():
    """``arnett``'s signature is ``(time, redshift, f_nickel, mej, **kwargs)``.

    ``vej``, ``kappa``, ``kappa_gamma`` and ``temperature_floor`` are read out of ``**kwargs``, so a
    signature-only parameter list would silently drop four of the seven parameters and every fit
    would run at redback's defaults for them.
    """
    pytest.importorskip("redback")
    assert RA.redback_parameters(SN) == SN_PARAMS
    assert RA.redback_parameters(TDE) == ["redshift"] + TDE_PHYS


@pytest.mark.parametrize("model", [SN, TDE, "basic_magnetar_powered"])
def test_prior_is_redbacks_own_prior_file(model):
    """Distribution by distribution against ``redback.priors.get_priors``.

    Nothing is transcribed here, so this is checking the *translation*: bilby ``Uniform`` ->
    WHISPER ``Uniform``, bilby ``LogUniform`` -> ``LogUniform``, ``DeltaFunction`` -> pinned,
    ``Constraint`` -> dropped, and every parameter accounted for by exactly one of those.
    """
    pytest.importorskip("redback")
    from redback.priors import get_priors

    rb = dict(get_priors(model=model))
    prior, pinned = RA.redback_prior(model), RA.redback_pinned(model)
    for key, dist in rb.items():
        kind = type(dist).__name__
        if kind == "Constraint":
            assert key not in prior.names and key not in pinned
        elif kind == "DeltaFunction":
            assert pinned[key] == pytest.approx(float(dist.peak), rel=1e-14)
        else:
            assert type(prior.distributions[key]).__name__ == kind, key
            assert prior.distributions[key].bounds == pytest.approx(
                (float(dist.minimum), float(dist.maximum)), rel=1e-12), key
    assert set(prior.names) | set(pinned) | {k for k, v in rb.items()
                                             if type(v).__name__ == "Constraint"} == set(rb)


def test_delta_prior_entries_become_pinned_parameters():
    """``cooling_envelope.prior`` pins four of five physics parameters. The adapter must too.

    This is the behaviour change from the per-model wrapper this replaced, which transcribed a
    *different* model's prior file (``gaussianrise_cooling_envelope``) to keep all five free. Reading
    redback's answer for the model you asked for is the point; unpinning is the caller's explicit
    choice, made with ``prior=``.
    """
    pytest.importorskip("redback")
    assert RA.redback_pinned(TDE) == {"mbh_6": 1.0, "eta": 0.1, "alpha": 0.1, "beta": 0.9}
    assert RA.redback_prior(TDE).names == ["redshift", "stellar_mass"]

    default = RA.redback_model(TDE, redshift=ZT)
    assert default.parameters == ["stellar_mass"]
    unpinned = _tde_model()
    assert unpinned.parameters == TDE_PHYS


def test_unsupported_prior_type_is_named_not_silently_dropped():
    """``tophat.prior`` gives ``thv`` a bilby ``Sine``, which WHISPER cannot represent.

    Guessing a Uniform in its place would be a silent change of prior. The error must name the
    parameter and the bilby class, and both escape hatches must work.
    """
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match=r"thv.*Sine"):
        RA.redback_model("tophat", redshift=0.05)
    assert "thv" not in RA.redback_model("tophat", redshift=0.05, pin={"thv": 0.1}).parameters
    assert "thv" in RA.redback_model("tophat", redshift=0.05,
                                     prior={"thv": Uniform(0.0, 0.4)}).parameters


def test_unknown_model_name_lists_the_registry():
    pytest.importorskip("redback")
    with pytest.raises(KeyError, match="all_models_dict"):
        RA.redback_parameters("not_a_redback_model_xyz")


def test_pin_and_prior_reject_names_that_are_not_parameters():
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match="not a parameter"):
        RA.redback_model(SN, pin={"nope": 1.0})
    with pytest.raises(ValueError, match="not a parameter"):
        RA.redback_model(SN, prior={"nope": Uniform(0, 1)})
    with pytest.raises(ValueError, match="redshift once"):
        RA.redback_model(SN, redshift=Z, pin={"redshift": Z})


# --- CLAIM 2: units -------------------------------------------------------------------------------

def test_mjy_to_jy_constant():
    """The one place the factor lives. 1e-3, not 1e3."""
    assert RA.MJY_TO_JY == 1e-3


@pytest.mark.parametrize("model", [SN, TDE])
def test_mjy_to_jy_is_exactly_redbacks_own_mjy(model):
    """WHISPER's Jy is EXACTLY 1e-3 x redback's own mJy on the identical call.

    This is the test a 1000x error cannot survive: it compares against redback's return value, not
    against a remembered magnitude.
    """
    pytest.importorskip("redback")
    from redback.model_library import all_models_dict

    if model == SN:
        t, bands, pars = T_OBS, BANDS, dict(SN_SAMPLE, redshift=Z)
    else:
        t = _epochs_inside_envelope(TDE_CASES[0], 9)
        bands, pars = np.array(["r"] * t.size), dict(_tde_pars(TDE_CASES[0]), redshift=ZT)
    nu = RA._frequencies_hz(bands)
    got = RA.redback_flux_jy(model, pars, t, bands)
    want = np.asarray(all_models_dict[model](t, output_format="flux_density", frequency=nu, **pars),
                      dtype=float)
    assert np.all(got > 0)
    assert np.allclose(got, 1e-3 * want, rtol=0, atol=0)


def test_jy_scale_anchored_against_redbacks_own_magnitude():
    """A 1000x unit error cannot survive this either, by a second and independent route.

    redback's ``magnitude`` branch is a different code path from the ``flux_density`` branch the
    adapter uses (it splines an sncosmo source off a 300-node time grid and integrates a real
    bandpass), so the two are not expected to agree exactly -- but agreement to a few tenths of a
    magnitude pins the absolute scale, where a factor of 1000 would show as ~7.5 mag.
    """
    pytest.importorskip("redback")
    from redback.model_library import all_models_dict

    t = np.geomspace(2.0, 40.0, 15)
    flux_jy = RA.redback_flux_jy(SN, dict(SN_SAMPLE, redshift=Z), t, np.array(["r"] * t.size))
    mag_ours = -2.5 * np.log10(flux_jy / 3631.0)
    mag_rb = np.asarray(all_models_dict[SN](t, output_format="magnitude", bands=["sdssr"],
                                            redshift=Z, **SN_SAMPLE), dtype=float)
    assert np.nanmax(np.abs(mag_ours - mag_rb)) < 0.2, (
        f"absolute scale is off: ours {mag_ours[:3]} vs redback {mag_rb[:3]}. A 1000x unit error "
        f"would show as ~7.5 mag.")


# --- band resolution (point 4/5 of the module docstring) -----------------------------------------

def test_band_frequencies_are_redbacks_own():
    """The frequency is redback's ``wavelength [Hz]`` column, reached via its own converter."""
    pytest.importorskip("redback")
    from redback.utils import bands_to_frequency

    got = RA._frequencies_hz(np.array(["g", "r", "i"]))
    assert np.allclose(got, np.asarray(bands_to_frequency(["g", "r", "i"])).ravel(),
                       rtol=0, atol=0)
    assert got[0] == pytest.approx(6.380e14)            # SDSS g, redback's table


def test_bare_letters_are_sdss_because_redbacks_table_says_so():
    """Deliberately unlike :mod:`whisper_cbpf.models.two_component_kilonova`, which remaps to LSST.

    Asserted rather than commented, because the two modules disagreeing looks like a bug until you
    know that this one is reading redback's own answer for a bare letter and the other is choosing a
    filter system for an sncosmo bandpass.
    """
    pytest.importorskip("redback")
    from redback.utils import bands_to_frequency

    got = RA._frequencies_hz(np.array(["g", "r", "i"]))
    assert np.allclose(got, bands_to_frequency(["sdssg", "sdssr", "sdssi"]), rtol=0, atol=0)


def test_band_resolution():
    pytest.importorskip("redback")
    for band in ("g", "r", "i", "sdssg", "lsstg", "ztfr", "bessellb", "2massj"):
        assert RA._redback_band(band) == band
    assert RA._redback_band("g-band") == "g"             # WHISPER's own effective-band label
    with pytest.raises(ValueError, match="not in redback's filters table"):
        RA._redback_band("not_a_band_xyz")


def test_band_names_are_validated_at_build_time():
    """``band_names=`` is a fail-fast: a typo must raise here, not inside the first likelihood."""
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match="not_a_band_xyz"):
        RA.redback_model(SN, ["g", "not_a_band_xyz"], redshift=Z)


def test_band_and_time_shapes_must_match():
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match="same shape"):
        RA.redback_flux_jy(SN, dict(SN_SAMPLE, redshift=Z), T_OBS, BANDS[:5])


def test_requires_bands():
    """The argument check must come BEFORE the optional dependency is resolved."""
    with pytest.raises(ValueError, match="band-dependent"):
        RA.redback_flux_jy(SN, dict(SN_SAMPLE, redshift=Z), T_OBS, bands=None)


# --- CLAIM 3: one call, not a per-band loop ------------------------------------------------------

def test_one_call_not_a_per_band_loop():
    """``arnett_bolometric``'s dense diffusion grid is ``linspace(0, time[-1] + 100, 1000)`` -- it
    depends on the time array. Splitting the light curve by band would give each band a different
    grid, so the whole curve must go through in one call. Detected by asking whether a point's flux
    depends on the OTHER bands' epochs, which it must (through ``time[-1]``) and would not under a
    per-band loop that re-derived the grid from that band's own times."""
    pytest.importorskip("redback")
    t = np.array([5.0, 5.0, 300.0])                     # the last epoch sets the grid
    pars = dict(SN_SAMPLE, redshift=Z)
    a = RA.redback_flux_jy(SN, pars, t, np.array(["g", "r", "i"]))
    b = RA.redback_flux_jy(SN, pars, t[:2], np.array(["g", "r"]))
    assert not np.allclose(a[:2], b), (
        "the grid is not responding to time[-1]; is this looping per band?")


def test_flux_finite_positive_and_band_dependent():
    pytest.importorskip("redback")
    m = _sn_model()
    flux = m.predict(SN_SAMPLE, T_OBS, BANDS)
    assert flux.shape == T_OBS.shape
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
    t = np.full(3, 5.0)
    fg, fr, fi = (m.predict(SN_SAMPLE, t, np.array([b] * 3))[0] for b in "gri")
    assert fg != fr and fr != fi

    tde = _tde_model()
    t = _epochs_inside_envelope(TDE_CASES[0], 4)
    f = {b: tde.predict(_tde_pars(TDE_CASES[0]), t, np.array([b] * t.size)) for b in "gri"}
    for v in f.values():
        assert np.all(np.isfinite(v)) and np.all(v > 0)
    assert not np.allclose(f["g"], f["r"]) and not np.allclose(f["r"], f["i"])
    mixed = tde.predict(_tde_pars(TDE_CASES[0]), np.full(3, t[1]), np.array(["g", "r", "i"]))
    assert np.allclose(mixed, [f["g"][1], f["r"][1], f["i"][1]])


# --- CLAIM 4: out-of-domain epochs ----------------------------------------------------------------

def test_out_of_domain_epochs_are_zero_but_the_band_survives():
    """redback raises past the envelope's termination; only THOSE epochs may become zero.

    Zeroing everything instead was measured to blank an otherwise-real light curve on 9.83% of
    ``cooling_envelope`` prior draws over AT2017GFO's 0.45-10.47 d grid -- a forward-model difference
    against :mod:`whisper_cbpf.models.jax.tde`, which zeroes per epoch. This is the regression test.

    The adapter finds the epochs redback will answer for by bisecting over redback's own
    accept/reject, so the recovered values must be *bit-identical* to asking for those epochs alone.
    """
    pytest.importorskip("redback")
    from redback.transient_models import tde_models

    case = TDE_CASES[3]                                 # short-lived envelope
    tmax = tde_models._cooling_envelope(*case).time_since_fb[-1] * (1 + ZT) / DAY
    t = np.array([tmax * 0.5, tmax * 0.9, tmax * 10.0])
    m = _tde_model()
    out = m.predict(_tde_pars(case), t, np.array(["r", "r", "r"]))
    assert np.all(np.isfinite(out))
    assert out[0] > 0.0 and out[1] > 0.0                # inside the span: untouched
    assert out[2] == 0.0                                # past termination: zero, not an exception
    alone = m.predict(_tde_pars(case), t[:2], np.array(["r", "r"]))
    assert np.allclose(out[:2], alone, rtol=0, atol=0)


def test_out_of_domain_epochs_survive_in_the_middle_of_a_multiband_curve():
    """The realistic shape of the failure: 3 bands interleaved, a tail past termination.

    Only the tail may be zero, and the surviving epochs must be untouched -- which is exactly what
    the whole-band version got wrong.
    """
    pytest.importorskip("redback")
    from redback.transient_models import tde_models

    case = TDE_CASES[3]
    tmax = tde_models._cooling_envelope(*case).time_since_fb[-1] * (1 + ZT) / DAY
    t = np.concatenate([np.geomspace(tmax * 0.05, tmax * 0.95, 15), tmax * np.array([1.5, 4.0])])
    bands = np.array([["g", "r", "i"][i % 3] for i in range(t.size)])
    m = _tde_model()
    out = m.predict(_tde_pars(case), t, bands)
    assert np.all(np.isfinite(out))
    assert np.all(out[:15] > 0.0), "the in-domain epochs were blanked with the out-of-domain ones"
    assert np.all(out[15:] == 0.0)
    inside = m.predict(_tde_pars(case), t[:15], bands[:15])
    assert np.allclose(out[:15], inside, rtol=0, atol=0)


def test_no_solution_at_all_is_zero_flux():
    """``envelope_exists`` is False here (``beta*(Rt/Rstar) < 20``): constraint == 0, no light curve.

    redback 1.12.0 raises ``IndexError`` from an empty slice rather than ``ValueError``; both must
    become finite zeros, which is what a sampler has to reject.
    """
    pytest.importorskip("redback")
    case = (0.010498878379993507, 8.344290585513178, 0.054451670132515374,
            0.7370240289465599, 1.931913362151394)      # test_tde_vs_redback's "constraint_one"
    out = _tde_model().predict(_tde_pars(case), np.geomspace(0.5, 50.0, 6),
                              np.array(["g", "r", "i", "g", "r", "i"]))
    assert np.all(np.isfinite(out)) and np.all(out == 0.0)


# --- registration, pinning, pickling -------------------------------------------------------------

def test_register_unpinned():
    pytest.importorskip("redback")
    m = RA.register_redback(SN, name="arnett_redback_test_unpinned", overwrite=True)
    assert "arnett_redback_test_unpinned" in list_models()
    assert get_model("arnett_redback_test_unpinned") is m
    assert m.parameters == SN_PARAMS
    assert set(m.default_prior.names) == set(SN_PARAMS)


def test_default_registry_name_is_the_redback_name():
    pytest.importorskip("redback")
    assert RA.redback_model(SN).name == "arnett_redback"
    assert RA.redback_model(TDE, prior=TDE_UNPIN).name == "cooling_envelope_redback"


def test_register_pinned_redshift_drops_it_without_changing_the_curve():
    """Pinning is what makes this comparable with a JAX factory, which never fits z."""
    pytest.importorskip("redback")
    m = RA.register_redback(SN, redshift=Z, name="arnett_redback_test_pinned", overwrite=True)
    assert m.parameters == SN_FREE6
    assert "redshift" not in m.default_prior.names
    assert np.array_equal(m.predict(SN_SAMPLE, T_OBS, BANDS),
                          RA.redback_flux_jy(SN, dict(SN_SAMPLE, redshift=Z), T_OBS, BANDS))


def test_pinned_value_wins_over_anything_passed_in():
    """A pinned parameter is pinned: a stale ``redshift`` left in the parameter dict must be ignored,
    not silently used, or a fit would sample a parameter the model reports as fixed."""
    pytest.importorskip("redback")
    m = _sn_model()
    assert np.array_equal(m.predict(dict(SN_SAMPLE, redshift=2.0), T_OBS, BANDS),
                          m.predict(SN_SAMPLE, T_OBS, BANDS))


@pytest.mark.parametrize("model,kw", [(SN, {}), (TDE, {"prior": TDE_UNPIN})])
def test_predict_and_model_are_picklable(model, kw):
    """``n_jobs > 1`` pickles ``predict``; a closure here fails while ``n_jobs=1`` succeeds.

    ``_RedbackPredict`` is a module-level class whose state is plain data, so the round trip must
    reproduce the curve exactly and not merely survive.
    """
    pytest.importorskip("redback")
    m = RA.redback_model(model, redshift=Z if model == SN else ZT, **kw)
    back = pickle.loads(pickle.dumps(m.predict))
    assert back.model == model and back.pinned == m.predict.pinned
    if model == SN:
        t, bands, pars = T_OBS, BANDS, SN_SAMPLE
    else:
        t = _epochs_inside_envelope(TDE_CASES[0], 6)
        bands, pars = np.array(["g", "r", "i"] * 2), _tde_pars(TDE_CASES[0])
    assert np.array_equal(back(pars, t, bands), m.predict(pars, t, bands))
    pickle.loads(pickle.dumps(m))                       # the whole Model, prior included


# --- redback's cosmology and version conventions -------------------------------------------------

def test_luminosity_distance_is_redbacks_own():
    """redback takes no ``dl_cm``; a JAX twin does. 1.23e26 vs 1.34991e26 at z=0.0098 is 0.20 mag."""
    pytest.importorskip("redback")
    from redback.transient_models.supernova_models import cosmo

    assert RA.redback_luminosity_distance_cm(Z, SN) == pytest.approx(
        float(cosmo.luminosity_distance(Z).cgs.value), rel=1e-14)
    assert RA.redback_luminosity_distance_cm(Z, TDE) == pytest.approx(
        float(cosmo.luminosity_distance(Z).cgs.value), rel=1e-14)


def test_redback_kwargs_reach_redback():
    """``redback_kwargs=`` is how a model that needs a fixed extra argument gets bound.

    ``cooling_envelope`` reads ``kwargs.get('cosmology', cosmo)``, so a different cosmology must
    change the flux by the distance ratio squared -- proving the kwarg is forwarded rather than
    swallowed.
    """
    pytest.importorskip("redback")
    from astropy.cosmology import Planck15, Planck18

    t = _epochs_inside_envelope(TDE_CASES[0], 5)
    bands = np.array(["r"] * t.size)
    base = _tde_model().predict(_tde_pars(TDE_CASES[0]), t, bands)
    other = _tde_model(redback_kwargs={"cosmology": Planck15}).predict(
        _tde_pars(TDE_CASES[0]), t, bands)
    ratio = (Planck18.luminosity_distance(ZT) / Planck15.luminosity_distance(ZT)).value ** 2
    assert np.allclose(other / base, ratio, rtol=1e-10)


# --- CLAIM 7: no redback installed ---------------------------------------------------------------

def test_module_usable_without_redback(monkeypatch):
    """Blocks every redback import. The module must still import (it is part of the minimal
    install), the band resolver must degrade to pass-through, and every call must raise
    ``ImportError`` naming redback rather than dying inside pandas.

    NOTE what is no longer claimed: the per-model wrappers this replaced could *register* without
    redback, because their prior was transcribed into the file. This adapter reads redback's prior
    file instead, so building a model needs redback -- which is the trade the architecture makes,
    and it is asserted here rather than left to be discovered.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "redback":
            raise ImportError("simulated: redback is not installed")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in list(sys.modules) if m.split(".")[0] == "redback"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(RA, "_MODEL_CACHE", {})
    monkeypatch.setattr(RA, "_FREQ_TABLE", None)
    monkeypatch.setattr(RA, "_BAND_CACHE", {})
    monkeypatch.setattr(RA, "_PRIOR_CACHE", {})

    mod = importlib.reload(RA)                          # a stray top-level import would fail here
    try:
        assert mod.MJY_TO_JY == 1e-3                    # plain data needs no backend
        assert mod._redback_band("g") == "g"            # resolver degrades to pass-through
        with pytest.raises(ImportError, match="requires the optional 'redback' package"):
            mod.redback_flux_jy(SN, dict(SN_SAMPLE, redshift=Z), T_OBS, BANDS)
        with pytest.raises(ImportError, match="requires the optional 'redback' package"):
            mod.redback_model(SN, redshift=Z)
    finally:
        monkeypatch.undo()
        importlib.reload(RA)                            # restore the shared module object


# --- CLAIM 6: it generalises ---------------------------------------------------------------------

@pytest.mark.parametrize("model", ["basic_magnetar_powered", "one_component_kilonova_model"])
def test_binds_a_model_that_never_had_a_wrapper(model):
    """The claim the architecture rests on: any redback model, no new code.

    ``basic_magnetar_powered`` is a magnetar central engine and
    ``one_component_kilonova_model`` an r-process kilonova -- neither shares a parameter set, a
    prior file or a physics family with ``arnett`` or ``cooling_envelope``, and neither has ever had
    a module in this package. Both must bind, sample their own redback prior, and return finite
    positive Jy that depends on the band.
    """
    pytest.importorskip("redback")
    m = RA.redback_model(model, ["g", "r", "i"], redshift=0.05)
    assert m.parameters and "redshift" not in m.parameters
    assert set(m.parameters) == set(m.default_prior.names)
    t = np.geomspace(1.0, 40.0, 12)
    bands = np.array([["g", "r", "i"][i % 3] for i in range(t.size)])
    flux = m.predict(m.default_prior.sample(np.random.default_rng(0)), t, bands)
    assert flux.shape == t.shape
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
    same = m.predict(m.default_prior.sample(np.random.default_rng(0)), np.full(3, 10.0),
                     np.array(["g", "r", "i"]))
    assert len(set(same.tolist())) == 3, "flux does not depend on the band"


# --- CLAIM 5: parity with the independent JAX ports ----------------------------------------------

@pytest.fixture()
def _x64():
    """float64 is MANDATORY: these engines are cgs luminosities of 1e43-1e46 erg/s and float32
    stops at 3.4e38. Restored afterwards so the float32 kilonova tests are unaffected."""
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


@pytest.mark.parametrize("case", range(len(SN_CASES)))
def test_supernova_parity_with_the_jax_twin(_x64, case):
    """redback's own ``arnett`` through the adapter, against the JAX port.

    Same parameters, epochs and frequencies. Every step between them is closed-form arithmetic on
    the same grid -- no quadrature, no interpolation, no spline -- so the bar is float64 round-off.
    Measured through this adapter: max relative 2.54e-15 over the four corners.
    """
    pytest.importorskip("redback")
    import jax.numpy as jnp

    from whisper_cbpf.models.jax import supernova as S

    p = SN_CASES[case]
    dl = RA.redback_luminosity_distance_cm(Z, SN)
    nu = RA._frequencies_hz(BANDS)
    grid = S.build_sn_grid(T_OBS / (1.0 + Z))
    ours = _sn_model().predict(p, T_OBS, BANDS)
    theirs = np.asarray(S.flux_density("arnett", grid, p, jnp.asarray(nu), Z, dl,
                                       dilation=RA.redback_applies_dilation(SN))) * RA.MJY_TO_JY
    assert _maxrel(ours, theirs) < 1e-12, f"case {case}: {_maxrel(ours, theirs):.3e}"


def test_tde_parity_with_the_jax_twin(_x64):
    """The adapter over ``cooling_envelope`` against the standalone JAX port.

    The preset must match the *installed* redback (:func:`_preset_for_installed_redback`): 1.12.0 and
    1.15.1 differ in the ODE grid (5000 vs 500 points) and in a ``(1+z)`` flux factor, so a
    hardcoded pairing would fail in one container for a reason unrelated to this code. Measured
    through this adapter against redback 1.15.1 (n_time=500, dilation=True): 8.53e-4 max relative
    over four cases x 12 epochs x 3 bands, the residual living entirely in the last epoch of the
    coarse grid; the median is ~1e-15.

    ``dl_cm`` must be Planck18's: redback derives the luminosity distance from the redshift with its
    own cosmology, and the JAX side takes it as an argument.
    """
    pytest.importorskip("redback")
    import jax.numpy as jnp
    cosmo = pytest.importorskip("astropy.cosmology")

    from whisper_cbpf.models.jax import tde as T

    preset = T.REDBACK_PRESETS[_preset_for_installed_redback()]
    dl = cosmo.Planck18.luminosity_distance(ZT).cgs.value
    bands = np.array(["g", "r", "i"] * 4)
    nu = RA._frequencies_hz(bands)
    m = _tde_model()
    worst = 0.0
    for case in TDE_CASES:
        t = _epochs_inside_envelope(case, bands.size)
        got = m.predict(_tde_pars(case), t, bands)
        want = 1e-3 * np.asarray(T.cooling_envelope_flux_density(
            jnp.asarray(t), jnp.asarray(nu), ZT, dl, *case, **preset))
        assert np.all(got > 0), case
        worst = max(worst, float(np.max(np.abs(got - want) / np.abs(want))))
    assert worst < 2e-3, f"max relative difference {worst:.3e}"


def test_no_arnett_prefactor_style_convention_offset(_x64):
    """CONVENTION CHECK. :mod:`whisper_cbpf.models.two_component_kilonova`'s JAX twin needs an
    ``arnett_prefactor`` because redback's *kilonova* diffusion kernel integrates to 1/2 -- a uniform
    0.7526 mag offset. The supernova has no analogue: both sides go through redback's
    ``ip.Diffusion``, so the flux ratio is exactly 1, not 2 and not 0.5. A future prefactor would
    show here as a constant ratio and would have to be argued for."""
    pytest.importorskip("redback")
    import jax.numpy as jnp

    from whisper_cbpf.models.jax import supernova as S

    p = SN_CASES[0]
    dl = RA.redback_luminosity_distance_cm(Z, SN)
    nu = RA._frequencies_hz(BANDS)
    grid = S.build_sn_grid(T_OBS / (1.0 + Z))
    ratio = (np.asarray(S.flux_density("arnett", grid, p, jnp.asarray(nu), Z, dl,
                                       dilation=RA.redback_applies_dilation(SN)))
             * RA.MJY_TO_JY / _sn_model().predict(p, T_OBS, BANDS))
    assert abs(np.median(ratio) - 1.0) < 1e-12
    assert abs(-2.5 * np.log10(np.median(ratio))) < 1e-11


def test_monochromatic_costs_more_than_the_port_does(_x64):
    """WHAT THE ``flux_density`` CHOICE COSTS, as a number rather than as a caveat.

    The adapter is monochromatic at each band's redback reference frequency; the shipped JAX factory
    integrates the real SDSS bandpass. That is a different OBSERVABLE, not a different
    implementation -- but it is 4+ orders of magnitude larger than the parity above, so anyone
    overlaying the two posteriors has to know which of the two they matched. Both bounds are
    asserted: it must not be negligible (or the module docstring is overcautious) and it must not be
    enormous in the well-behaved corner (or the band mapping is wrong, not approximate).
    """
    pytest.importorskip("redback")
    pytest.importorskip("sncosmo")
    import jax.numpy as jnp

    from whisper_cbpf.models.jax import kilonova as kn
    from whisper_cbpf.models.jax import supernova as S

    dl = RA.redback_luminosity_distance_cm(Z, SN)
    grid = S.build_sn_grid(T_OBS / (1.0 + Z))
    fs = kn.make_filter_set(["sdssg", "sdssr", "sdssi"], n_wave=2000)
    w, n = kn.ab_weights(fs["lam"], fs["trans"])
    bidx = jnp.asarray(np.array([["g", "r", "i"].index(x) for x in BANDS]))

    p = SN_CASES[0]
    mag = np.asarray(S.sn_ab_magnitude(grid, S.bolometric("arnett", grid, p), p["vej"],
                                       p["temperature_floor"], bidx, w, n,
                                       jnp.asarray(fs["lam"]), Z, dl,
                                       dilation=RA.redback_applies_dilation(SN)))
    f_band = 3631.0 * 10.0 ** (-0.4 * mag)
    f_mono = _sn_model().predict(p, T_OBS, BANDS)
    rel = _maxrel(f_band, f_mono)
    assert rel > 1e-3, ("band-integrated and monochromatic now agree to better than 1e-3; "
                        "re-measure, the module docstring says they do not")
    assert rel < 0.1, f"{rel:.3e} is too large for a bandpass-width effect; check the band mapping"


def test_dilation_detection_matches_what_redback_computes(_x64):
    """:func:`~whisper_cbpf.models.redback_adapter.redback_applies_dilation` reads the model's
    source. Check the source says what the numbers do: only ONE of ``dilation=True/False`` can match
    redback, and it must be the one reported. At z = 0.0098 the factor is 0.0106 mag -- small, and
    100x the parity bar."""
    pytest.importorskip("redback")
    import jax.numpy as jnp

    from whisper_cbpf.models.jax import supernova as S

    p = SN_CASES[0]
    dl = RA.redback_luminosity_distance_cm(Z, SN)
    nu = RA._frequencies_hz(BANDS)
    grid = S.build_sn_grid(T_OBS / (1.0 + Z))
    ours = _sn_model().predict(p, T_OBS, BANDS)
    rel = {}
    for dil in (True, False):
        jx = np.asarray(S.flux_density("arnett", grid, p, jnp.asarray(nu), Z, dl,
                                       dilation=dil)) * RA.MJY_TO_JY
        rel[dil] = _maxrel(ours, jx)
    reported = RA.redback_applies_dilation(SN)
    assert rel[reported] < 1e-12, f"reported dilation={reported} but rel={rel}"
    assert rel[not reported] > 1e-3, f"the two conventions are indistinguishable here: {rel}"
