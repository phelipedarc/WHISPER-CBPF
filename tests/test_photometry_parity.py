"""One photometry for the CPU and the GPU.

Up to whisper 0.1.0 a band magnitude was made four ways: the redback CPU adapter took redback's
SED at ONE reference frequency per band, the two-component kilonova used redback's magnitude branch,
``mck19`` used the effective wavelength (6000 A for anything unresolved) and the JAX models
integrated the bandpass. A CPU fit and a GPU fit of the same model disagreed by 4-40 mmag median and
up to 0.65 mag, by construction. What this file holds the package to:

1. **CPU = the band integral of redback's own SED**, to <= 2e-5 mag, for every family whisper binds
   (the reference here is an independent 1 A trapezoid on sncosmo's curve, not whisper's code).
2. **CPU = GPU** when both use the same FilterSet: the bolometric physics agrees to round-off, so
   what is left is float64 noise.
3. **The double-dilation TDEs** are redback called at ``t (1+z)``.
4. A silent revert to monochromatic, or a grouped label reaching a model, is caught.
5. pyphot, an independent implementation of the band integral, agrees on the same curve.
"""
from __future__ import annotations

import numpy as np
import pytest

from whisper_cbpf.models import redback_adapter as RA

C_AA = 2.99792458e18
Z = 0.051
T_OBS = np.geomspace(1.0, 60.0, 24)
BANDS = np.array(["ztfg", "ztfr", "lsstu", "lsstg", "lssti", "lssty"] * 4)
ARNETT = dict(f_nickel=0.1, mej=2.0, vej=1e4, kappa=0.1, kappa_gamma=0.03,
              temperature_floor=4000.0)


@pytest.fixture()
def _x64():
    """float64 for a CPU-against-JAX comparison; every one here also needs redback (the CPU side,
    or ``redback_luminosity_distance_cm``), so a ``[gpu]`` install without ``[models]`` skips."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("redback")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


def fine_band_mag(fn, t, bands, kw, step=1.0):
    """redback's SED integrated over each observation's sncosmo bandpass: a 1 A trapezoid of the
    piecewise-linear curve, photon-counting AB. ONE redback call, each observation's nodes
    contiguous and the last observation last, so redback builds the grid the model builds."""
    sncosmo = pytest.importorskip("sncosmo")
    nodes = {}
    for b in set(bands):
        bp = sncosmo.get_bandpass(b)
        lam = np.arange(bp.wave[0], bp.wave[-1] + 0.5 * step, step)
        w = np.interp(lam, bp.wave, bp.trans, left=0.0, right=0.0) / lam * step
        w[[0, -1]] *= 0.5
        nodes[b] = (lam[w > 0], w[w > 0])
    tt = np.concatenate([np.full(nodes[b][0].size, ti) for ti, b in zip(t, bands)])
    nu = np.concatenate([C_AA / nodes[b][0] for b in bands])
    mjy = np.asarray(fn(tt, output_format="flux_density", frequency=nu, **kw), dtype=float)
    out, i = np.empty(len(t)), 0
    for k, b in enumerate(bands):
        n = nodes[b][0].size
        out[k] = nodes[b][1] @ mjy[i:i + n] / nodes[b][1].sum()
        i += n
    return -2.5 * np.log10(1e-3 * out / 3631.0)


def _mag(jy):
    return -2.5 * np.log10(np.asarray(jy) / 3631.0)


def _redback(model):
    pytest.importorskip("redback")
    from redback.model_library import all_models_dict
    return all_models_dict[model]


CASES = {
    "arnett": (ARNETT, T_OBS),
    "basic_magnetar_powered": (dict(p0=2.0, bp=1.0, mass_ns=1.4, theta_pb=1.0, mej=2.0, vej=1e4,
                                    kappa=0.1, kappa_gamma=0.03, temperature_floor=4000.0), T_OBS),
    "csm_shock_and_arnett": (dict(mej=2.0, f_nickel=0.1, csm_mass=1.0, v_min=1e4, beta=0.45,
                                  kappa=0.2, shell_radius=1.0, shell_width_ratio=0.1,
                                  kappa_gamma=0.03, temperature_floor=4000.0), T_OBS),
    "one_component_kilonova_model": (dict(mej=0.03, vej=0.2, kappa=3.0, temperature_floor=3000.0),
                                     np.geomspace(0.3, 12.0, 24)),
    "cooling_envelope": (dict(mbh_6=1.0, stellar_mass=1.0, eta=0.05, alpha=0.1, beta=1.0),
                         np.geomspace(5.0, 200.0, 24)),
}


# --- 1. the CPU path is the band integral --------------------------------------------------------

@pytest.mark.parametrize("model", sorted(CASES))
def test_cpu_is_the_fine_band_integral_of_redbacks_own_sed(model):
    """<= 2e-5 mag (0.02 mmag) against an independent fine integral. The monochromatic path of
    whisper <= 0.1.0 missed this by 1-40 mmag on these same points."""
    p, t = CASES[model]
    kw = dict(p, redshift=Z)
    fn = _redback(model)                    # first: it skips without redback
    got = _mag(RA.redback_flux_jy(model, kw, t, BANDS))
    want = fine_band_mag(fn, t, BANDS, kw)
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - want)) < 2e-5, f"{model}: {np.max(np.abs(got - want)):.2e} mag"


#: The one family over 2e-5 (docs/PHOTOMETRY.md): ``type_1a``'s CutoffBlackbody has a
#: kink at 3000 A rest frame, which at z = 0.35 lands at 4050 A, inside LSST g. A piecewise-smooth
#: SED is where a 16-node Gauss rule is least exact: this draw (the worst of 60 of redback's prior)
#: is 3.6e-5 mag off, and n_nodes=32 gives 8.9e-6.
#: Gauss-16 stays the default; this family's budget is 5e-5.
TYPE_1A_BUDGET = 5e-5
TYPE_1A_KINK_IN_LSSTG = dict(f_nickel=0.0209475, mej=53.4002, vej=9995.2, kappa=0.879196,
                             kappa_gamma=9.15604, temperature_floor=97767.2,
                             line_wavelength=6500.0, line_width=500.0, line_amplitude=0.3,
                             redshift=0.35)


def test_type_1a_cutoff_kink_in_lsst_g_is_within_its_documented_budget():
    from whisper_cbpf.synphot import filter_set_for

    bands = np.array(["lsstu", "lsstg", "lsstr", "lssti"] * 6)
    t = np.repeat(np.geomspace(2.0, 60.0, 6), 4)
    kw = TYPE_1A_KINK_IN_LSSTG
    want = fine_band_mag(_redback("type_1a"), t, bands, kw)
    g16 = np.abs(_mag(RA.redback_flux_jy("type_1a", kw, t, bands)) - want)
    fs32 = filter_set_for(sorted(set(bands)), n_nodes=32)
    g32 = np.abs(_mag(RA.redback_flux_jy("type_1a", kw, t, bands, filter_set=fs32)) - want)
    assert g16.max() < TYPE_1A_BUDGET, f"Gauss-16 {g16.max():.2e} mag"
    assert bands[np.argmax(g16)] == "lsstg", "the kink's band"
    assert g32.max() < 2e-5, f"Gauss-32 {g32.max():.2e} mag: the documented way below 2e-5"


def test_monochromatic_is_an_explicit_opt_in_and_differs_by_more_than_1_mmag():
    """The old path is kept, named, and measurably different -- so a silent revert of the default
    to it (or of this opt-in to the band integral) fails here."""
    pytest.importorskip("redback")
    kw = dict(ARNETT, redshift=Z)
    band = _mag(RA.redback_flux_jy("arnett", kw, T_OBS, BANDS))
    mono = _mag(RA.redback_flux_jy("arnett", kw, T_OBS, BANDS, photometry="monochromatic"))
    assert np.median(np.abs(band - mono)) > 1e-3
    with pytest.raises(ValueError, match="photometry must be one of"):
        RA.redback_flux_jy("arnett", kw, T_OBS, BANDS, photometry="bandpass")


def test_a_grouped_label_raises_instead_of_reaching_a_model():
    """``g-band`` (``load_lightcurve(band_lookup=...)``) merges every g filter: there is no curve to
    integrate. whisper <= 0.1.0 modelled it as SDSS g -- also for ZTF g data."""
    pytest.importorskip("redback")
    with pytest.raises(ValueError, match="grouped effective-band label"):
        RA.redback_flux_jy("arnett", dict(ARNETT, redshift=Z), T_OBS[:2],
                           np.array(["g-band", "r-band"]))
    from whisper_cbpf.synphot import resolve_filter
    with pytest.raises(ValueError, match="grouped effective-band label"):
        resolve_filter("g-band")


# --- 2. CPU = GPU on one FilterSet ------------------------------------------------------------------

def test_cpu_and_gpu_supernova_agree_to_round_off(_x64):
    """Same physics, same band integral: redback's arnett through the adapter against the JAX
    factory at its defaults. 5e-15 mag measured; whisper <= 0.1.0 differed by ~10 mmag."""
    import whisper_cbpf as wp

    bands = sorted(set(BANDS))
    dl = RA.redback_luminosity_distance_cm(Z, "arnett")
    cpu = RA.redback_model("arnett", bands, redshift=Z).predict(ARNETT, T_OBS, BANDS)
    gpu = wp.supernova_model("arnett", bands, Z, dl).predict(ARNETT, T_OBS, BANDS)
    assert np.max(np.abs(_mag(cpu) - _mag(gpu))) < 1e-9


def test_cpu_and_gpu_kilonova_agree_on_one_filter_set(_x64):
    """Passing the SAME FilterSet object to both sides: the one-component kilonova, redback's grid."""
    import whisper_cbpf as wp
    from whisper_cbpf.synphot import filter_set_for

    fs = filter_set_for(["lsstg", "lsstr", "lssti"])
    p, t = CASES["one_component_kilonova_model"]
    bands = np.array(["lsstg", "lsstr", "lssti"] * 8)
    dl = RA.redback_luminosity_distance_cm(Z, "one_component_kilonova_model")
    cpu = RA.redback_model("one_component_kilonova_model", list(fs.names), redshift=Z,
                           filter_set=fs).predict(p, t, bands)
    gpu = wp.kilonova_model(list(fs.names), Z, dl, filter_set=fs, mag_floor=99.0).predict(
        p, t, bands)
    assert np.max(np.abs(_mag(cpu) - _mag(gpu))) < 1e-6


def test_a_filter_set_is_matched_by_name_not_by_position(_x64):
    """A FilterSet passed to a factory in another order than ``band_names``, holding a curve of
    the user's own under any name, must integrate each band with its own filter -- on both sides."""
    import whisper_cbpf as wp
    from whisper_cbpf.synphot import FilterSet, filter_set_for, gauss_rule

    wave = np.linspace(5400.0, 7000.0, 161)
    mine = gauss_rule(["mycam_r"], curves={"mycam_r": (wave, np.exp(-0.5 * ((wave - 6200) / 300) ** 2))})
    fs = FilterSet.concat([mine, filter_set_for(["ztfg"])])            # order: mycam_r, ztfg
    dl = RA.redback_luminosity_distance_cm(Z, "arnett")
    t, b = T_OBS[:8], np.array(["ztfg", "mycam_r"] * 4)
    gpu = wp.supernova_model("arnett", ["ztfg", "mycam_r"], Z, dl, filter_set=fs).predict(ARNETT, t, b)
    # same epochs (the SN grid is sized by the last one), every point in ztfg
    ref = wp.supernova_model("arnett", ["ztfg"], Z, dl).predict(ARNETT, t, np.array(["ztfg"] * 8))
    assert np.max(np.abs(_mag(gpu[::2]) - _mag(ref[::2]))) < 1e-12        # ztfg is ztfg
    cpu = RA.redback_model("arnett", ["ztfg", "mycam_r"], redshift=Z, filter_set=fs).predict(
        ARNETT, t, b)
    assert np.max(np.abs(_mag(cpu) - _mag(gpu))) < 1e-9


def test_jax_factories_default_to_gauss16_and_n_wave_keeps_the_grid(_x64):
    """``n_wave=None`` is the sentinel: the default is Gauss-16 per band (16 x n_bands nodes), an
    explicit ``n_wave`` is whisper <= 0.1.0's shared grid. The two agree to <= 0.005 mmag."""
    import whisper_cbpf as wp

    bands = ["ztfg", "ztfr"]
    dl = RA.redback_luminosity_distance_cm(Z, "arnett")
    gauss = wp.supernova_model("arnett", bands, Z, dl)
    grid = wp.supernova_model("arnett", bands, Z, dl, n_wave=2000)
    assert gauss.predict.__self__.lam.size == 16 * len(bands)
    assert grid.predict.__self__.lam.size == 2000
    b = np.array(bands * 12)
    d = _mag(gauss.predict(ARNETT, T_OBS, b)) - _mag(grid.predict(ARNETT, T_OBS, b))
    assert np.max(np.abs(d)) < 5e-6


def test_bare_letters_in_a_jax_factory_are_lsst(_x64):
    """The JAX factories took sncosmo names only (a bare ``g`` failed inside sncosmo); a bare
    letter is now LSST's filter there as everywhere, or the survey ``default_system=`` names."""
    import whisper_cbpf as wp

    dl = RA.redback_luminosity_distance_cm(Z, "arnett")
    t, p = T_OBS[:6], ARNETT
    bare = wp.supernova_model("arnett", ["g", "r"], Z, dl).predict(p, t, np.array(["g", "r"] * 3))
    lsst = wp.supernova_model("arnett", ["lsstg", "lsstr"], Z, dl).predict(
        p, t, np.array(["lsstg", "lsstr"] * 3))
    sdss = wp.supernova_model("arnett", ["g", "r"], Z, dl, default_system="sdss").predict(
        p, t, np.array(["g", "r"] * 3))
    assert np.array_equal(bare, lsst)
    assert np.min(np.abs(_mag(sdss) - _mag(lsst))) > 1e-3


# --- 3. redback double time dilation ------------------------------------------------------------------

#: A draw from redback's own gaussianrise prior (seed 3, the 7th), 19-22 mag at z = 0.35.
GAUSSIANRISE = dict(peak_time=16.3727, sigma_t=43.4653, mbh_6=0.9686, stellar_mass=3.2916,
                    eta=0.0432, alpha=0.1266, beta=4.3991)


def test_double_dilation_is_detected_in_exactly_the_three_models():
    pytest.importorskip("redback")
    for model in ("gaussianrise_cooling_envelope", "bpl_cooling_envelope", "stream_stream_tde"):
        assert RA.redback_double_dilation(model), model
    for model in ("cooling_envelope", "arnett", "one_component_kilonova_model",
                  "tde_analytical"):
        assert not RA.redback_double_dilation(model), model


def test_double_dilation_tde_is_redback_at_t_times_one_plus_z(_x64):
    """redback's gaussianrise evolves (1+z) too slowly; the adapter calls it at t(1+z),
    which equals the JAX port (always right) to its own discretisation: 1.6e-5 mag at this draw,
    0.3 mmag median over prior draws, with a tail to 0.06 mag that is the port's, not the
    photometry's. redback uncorrected is 0.83 mag off here."""
    import whisper_cbpf as wp

    z, model = 0.35, "gaussianrise_cooling_envelope"
    t = np.geomspace(2.0, 90.0, 24)
    bands = np.array(["lsstg", "lsstr", "lssti"] * 8)
    kw = dict(GAUSSIANRISE, redshift=z)
    got = _mag(RA.redback_flux_jy(model, kw, t, bands))
    fn = _redback(model)
    fixed = fine_band_mag(fn, t * (1 + z), bands, kw)
    raw = fine_band_mag(fn, t, bands, kw)
    assert np.max(np.abs(got - fixed)) < 2e-5
    assert np.max(np.abs(got - raw)) > 0.3
    port = wp.tde_model(["lsstg", "lsstr", "lssti"], z,
                        RA.redback_luminosity_distance_cm(z, model), rise="gaussian")
    assert np.max(np.abs(got - _mag(port.predict(GAUSSIANRISE, t, bands)))) < 1e-3


# --- 5. pyphot, an independent band integral --------------------------------------------------------

@pytest.mark.parametrize("band", ["lsstg", "ztfr", "sdssi", "lssty"])
def test_pyphot_agrees_with_the_band_integral_on_the_same_curve(band):
    """Role C of the pyphot trial: pyphot's own photon-counting integral of the SAME sncosmo curve,
    on blackbodies from 3000 to 30000 K. Measured <= 0.0015 mmag (the target was 1 mmag)."""
    sncosmo = pytest.importorskip("sncosmo")
    pytest.importorskip("pyphot")
    import astropy.units as u
    from pyphot import Filter

    from whisper_cbpf.synphot import filter_set_for

    bp = sncosmo.get_bandpass(band)
    filt = Filter(bp.wave * u.AA, bp.trans, name=band, dtype="photon")
    fs = filter_set_for([band])
    x, w = fs.nodes[0], fs.weights[0]
    lam = np.linspace(2000.0, 12000.0, 20001)
    for temp in (3000.0, 6000.0, 1e4, 3e4):
        flam = lambda l: 1.0 / l ** 5 / np.expm1(1.4387768775039337e8 / (l * temp))  # noqa: E731
        mag_pyphot = -2.5 * np.log10(filt.get_flux(lam * u.AA, flam(lam)).value) - filt.AB_zero_mag
        fnu = flam(x) * x ** 2 / C_AA                    # erg/s/cm2/Hz for f_lam in erg/s/cm2/A
        mag_whisper = -2.5 * np.log10(w @ fnu / w.sum()) - 48.6
        assert abs(mag_pyphot - mag_whisper) < 1e-5, (band, temp)
