"""``whisper_cbpf.models.tide``: the TiDE tidal-disruption model (CPU, calls libtide through tidepy).

Fourteen claims, each with a number. The first group needs no library (it reads the golden, or
replaces the library by a stub); the second group runs the real library and is **skipped, not
failed**, where libtide/tidepy are absent (the JAX-only container).

Without libtide
---------------
1. **The golden is the oracle's.** ``tests/goldens/tide_oracle.npz`` holds the unmodified library's
   output for 59 parameter sets; its provenance names TiDE commit ``9bd5b89`` and the SHA256 of
   ``libtide.so`` and of the two fall-back tables.
2. **Known answers in the golden**: ``t_min`` = 41.0076 d at ``beta_fulldisrupt = 1`` and 6.4766 d at
   1.85, whose ratio is ``1.85^3`` to 2e-5 relative; the notebook anchor (1e6 Msun, 1 Msun,
   eta = 0.1) peaks at 37.68 d with 3.734e44 erg/s (the notebook's 37.7 d, 3.73e44).
3. **The conversion logic**, with the library replaced by a lookup of the golden ``L_nu``: the
   model's flux is the oracle's ``F_nu = (1+z) L_nu(nu (1+z), t/(1+z)) / (4 pi d_L^2)`` at every one
   of the 59 x 30 x 6 points, to rounding; an epoch at or before ``t_min (1 + 1e-4)`` is 0 Jy and
   never reaches the library; ``t_exp`` and the redshift enter as ``tau = (t - t_exp)/(1+z)``.
4. **Bands**: a label in ``band_wavelengths``, a bare number (Angstrom) or a filter (the shipped SDSS
   curves: ``g`` at 4685 A) resolve to the wavelength the library is called at; ``n_nodes`` Gauss
   nodes average with weights that sum to one.
5. **Parameters**: ``pin`` / ``free`` give the documented parameter lists and priors; parameter
   sets outside the library's domain give zero flux; bad arguments raise.
6. **No library at import, a clear error without it**, and the model pickles by value (432 bytes)
   with no library object inside.

With libtide (skipped without it)
---------------------------------
7. **Parity with the oracle**: the model's ``F_nu`` against the oracle's at the 53 anchor, variant
   and Latin-hypercube sets, evaluated in a shuffled order through ONE reused library object per
   configuration (so a stale-state error would show): max relative difference 4e-14.
8. **L_nu, t_min, bolometric luminosity and wind photosphere** equal the oracle's (bit for bit up
   to the integration, 1e-9 budgets), and the six variants (``Mdotfb_calc`` default and L09_const,
   ``tmin_calc`` with_rp and GR13, the 5/3 polytrope, a white dwarf) do too.
9. **Known answers live**: the two ``t_min`` values, the anchor's peak, ``t_min`` ~ ``M_BH^1/2``
   exactly, ``t_peak`` ~ ``M_BH^0.4999`` and independent of ``eta`` and ``fout`` to 3e-5.
10. **Physics**: the wind photosphere shrinks (1.2e14 -> 1.6e11 cm) and heats (7e4 -> 9e5 K) after
    the peak; the wind dominates the luminosity at the peak (disc/wind 0.39) and the disc late (3.7 at
    640 d).
11. **Pickling and ``spawn``**: a worker built by ``spawn`` gives the parent's flux to the bit.
12. **Registration and CPU samplers**: ``register_tide`` under a unique name, ``predict_jax`` is
    ``None``, and ``fit_ABC`` (two processes) and ``fit_MCMC`` run on it.
13. **Robustness**: 300 draws of the default prior, and 300 of a box 10x wider, give finite flux and
    never crash the process; the same library object is reused (at most two per process).
14. **Upstream behaviour reproduced, not hidden**: ``tidepy.Parameters()`` ignores a later
    ``Mdotfb_calc = "default"`` and a later numeric ``fout`` (U8 and its sibling), the disc ring
    count ``N`` is read when the disc object is created, and the diffusion step returns NaN at
    its first epoch. Each is asserted here so a fixed upstream shows up as a failing test.
"""
from __future__ import annotations

import contextlib
import io
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.models import tide as T

GOLDEN = Path(__file__).parent / "goldens" / "tide_oracle.npz"
LIB = T.libtide_available()
needs_lib = pytest.mark.skipif(not LIB, reason="libtide/tidepy not importable (use the tide container)")

Z = 0.05
C_AA = 2.99792458e18


@pytest.fixture(scope="module")
def gold():
    d = np.load(GOLDEN, allow_pickle=False)
    return {k: d[k] for k in d.files}


def _prov(gold):
    import json
    return json.loads(str(gold["provenance"]))


def _index(gold, name):
    return list(gold["names"]).index(name)


def _spec(gold, k):
    """The golden's parameter set ``k`` as ``(physical dict for the model, pins, config kwargs)``."""
    bif = bool(gold["fout_is_bifout"][k])
    phys = dict(bh_M6=float(gold["bh_M6"][k]), star_mstar=float(gold["star_mstar"][k]),
                bh_eta=float(gold["bh_eta"][k]), fv=float(gold["fv"][k]))
    pin = dict(eta_r=float(gold["eta_r"][k]), beta_fulldisrupt=float(gold["beta_fulldisrupt"][k]))
    if bif:
        pin["fout"] = "bifout"
    else:
        phys["fout"] = float(gold["fout"][k])
    cfg = dict(Mdotfb_calc=str(gold["mdotfb_calc"][k]), tmin_calc=str(gold["tmin_calc"][k]),
               star_rstar=str(gold["star_rstar"][k]), star_politrop=str(gold["star_politrop"][k]))
    return phys, pin, cfg


def _model(gold, k, **kw):
    phys, pin, cfg = _spec(gold, k)
    wl = dict(zip(gold["band_names"], map(float, gold["band_lambda_aa"])))
    return (T.tide_model(redshift=Z, dl_cm=float(gold["dl_cm"]), pin=pin, band_wavelengths=wl,
                         **cfg, **kw), phys)


def _grid(gold, k):
    """Observer-frame (times, bands) of the golden's z = 0.05 grid for set ``k``, and its Jy."""
    t = gold["t_rel"][k] * (1 + Z)
    return (np.repeat(t, 6), np.tile(gold["band_names"], t.size), gold["fnu_jy_z"][k].reshape(-1))


# --------------------------------------------------------------------------------------------
# 1-2. the golden
# --------------------------------------------------------------------------------------------
def test_01_golden_provenance(gold):
    prov = _prov(gold)
    assert prov["tide_commit"].startswith("9bd5b89"), prov["tide_commit"]
    for key in ("libtide_so_sha256", "table_n3_dat_sha256", "table_n3per2_dat_sha256",
                "tidepy_tide_py_sha256"):
        assert len(prov[key]) == 64 and int(prov[key], 16) >= 0, key
    assert prov["tidepy_tide_py_sha256"] == prov["pristine_tide_py_sha256"]   # unmodified wrapper
    assert len(gold["names"]) == prov["n_sets"] == 59
    assert prov["route_standard_vs_new_max_rel"] == 0.0 and prov["order_max_rel"] == 0.0
    assert prov["wind_blackbody_residual_max"] < 1e-12     # measured 1.5e-13


def test_02_known_answers_in_the_golden(gold):
    a185, a100 = _index(gold, "anchor_beta185"), _index(gold, "anchor_beta100")
    assert gold["tmin"][a185] == pytest.approx(6.4766, abs=5e-5)
    assert gold["tmin"][a100] == pytest.approx(41.0076, abs=5e-5)
    assert gold["tmin"][a100] / gold["tmin"][a185] == pytest.approx(1.85 ** 3, rel=2e-5)  # 1.5e-5
    for name, (tp, lp) in {"anchor_beta185": (37.68, 3.734e44)}.items():
        t, L = gold[f"notebook_t_{name}"], gold[f"notebook_lbol_{name}"]
        i = int(np.argmax(L))
        assert t[i] == pytest.approx(tp, abs=0.01) and L[i] == pytest.approx(lp, rel=1e-3)
    assert 37.6 < t[i] < 37.8 and 3.72e44 < L[i] < 3.75e44           # the notebook's 37.7 d, 3.73e44


# --------------------------------------------------------------------------------------------
# 3-5. the conversion logic, the library replaced by the golden's L_nu
# --------------------------------------------------------------------------------------------
class _GoldenEngine:
    """Stands in for the library: ``L_nu`` is looked up in the golden's z = 0.05 grid."""

    def __init__(self, gold, k):
        self.g, self.k = gold, k
        self.tmin = float(gold["tmin"][k])
        self.calls = []
        self.nu_z = gold["nu_bands"] * (1 + Z)

    def configure(self, v):
        return self.tmin

    def lnu(self, tau, nu):
        self.calls.append((tau, nu))
        j = int(np.argmin(np.abs(self.nu_z / nu - 1)))
        assert abs(self.nu_z[j] / nu - 1) < 1e-12, "nu is not one of the six band frequencies (1+z)"
        i = int(np.argmin(np.abs(self.g["t_rel"][self.k] / tau - 1)))
        assert abs(self.g["t_rel"][self.k][i] / tau - 1) < 1e-12, "tau is not on the golden grid"
        return self.g["lw_rel_z"][self.k][i, j], self.g["ld_rel_z"][self.k][i, j]


def test_03_conversion_matches_the_oracle_flux(gold, monkeypatch):
    worst, n = 0.0, 0
    for k in range(len(gold["names"])):
        if str(gold["kinds"][k]) == "variant" and gold["names"][k] != "var_fout_0p3_fv2_etar0p2":
            continue
        eng = _GoldenEngine(gold, k)
        monkeypatch.setattr(T, "_engine", lambda cfg, bifout, eng=eng: eng)
        model, phys = _model(gold, k)
        t, b, ref = _grid(gold, k)
        f = model.predict(phys, t, b)
        lit = gold["t_rel"][k] > eng.tmin * 1.0001
        lit6 = np.repeat(lit, 6)
        assert np.all(f[~lit6] == 0.0)
        # the epochs at or below t_min (1 + 1e-4) never reached the library
        assert all(c[0] > eng.tmin * 1.0001 for c in eng.calls)
        ok = lit6 & (ref > 0)
        worst = max(worst, float(np.max(np.abs(f[ok] / ref[ok] - 1))))
        n += int(ok.sum())
    assert n > 59 * 20 and worst < 1e-12, (n, worst)        # measured ~2e-16


def test_03b_t_exp_and_redshift(gold, monkeypatch):
    k = _index(gold, "anchor_beta185")
    eng = _GoldenEngine(gold, k)
    monkeypatch.setattr(T, "_engine", lambda cfg, bifout: eng)
    model, phys = _model(gold, k)
    t, b, ref = _grid(gold, k)
    # t_exp as a constant: a shift of the epochs, exactly
    shifted = T.tide_model(["4686"], redshift=Z, dl_cm=float(gold["dl_cm"]), t_exp_days=1234.5,
                           pin={"eta_r": 0.0, "beta_fulldisrupt": 1.85, "fout": "bifout"})
    f0 = model.predict(phys, t, b)
    assert np.array_equal(shifted.predict(phys, t + 1234.5, np.full(t.shape, "4686")),
                          model.predict(phys, t, np.full(t.shape, "4686")))
    # a free t_exp and a free redshift are read from the parameter dict
    mz = T.tide_model(["g"], free=["t_exp", "redshift"], dl_cm=None, redshift=None, default_system="sdss",
                      pin={"eta_r": 0.0, "beta_fulldisrupt": 1.85, "fout": "bifout"},
                      prior=wp.Prior({"t_exp": wp.Uniform(-5, 0)}),
                      band_wavelengths={"g": 4686.0})
    assert mz.parameters == ["bh_M6", "star_mstar", "bh_eta", "fv", "t_exp", "redshift"]
    seen = []

    class Rec:
        tmin = 10.0

        def configure(self, v):
            return self.tmin

        def lnu(self, tau, nu):
            seen.append((tau, nu))
            return 1e28, 0.0

    monkeypatch.setattr(T, "_engine", lambda cfg, bifout: Rec())
    p = dict(bh_M6=1.0, star_mstar=1.0, bh_eta=0.1, fv=1.0, t_exp=-2.0, redshift=0.1)
    f = mz.predict(p, [48.0, 5.0], ["g", "g"])
    # (48 - (-2)) / 1.1 = 45.45 d > t_min; (5 + 2) / 1.1 = 6.4 d <= t_min -> 0 Jy, not asked of the library
    assert len(seen) == 1 and seen[0][0] == pytest.approx(50.0 / 1.1, rel=1e-14)
    assert seen[0][1] == pytest.approx(C_AA / 4686.0 * 1.1, rel=1e-14)
    dl = float(T.luminosity_distance_cm(0.1))
    assert f[0] == pytest.approx(1e23 * 1.1 * 1e28 / (4 * np.pi * dl ** 2), rel=1e-14) and f[1] == 0.0
    assert f0.shape == t.shape


def test_03c_distance_is_planck18(gold):
    assert float(T.luminosity_distance_cm(Z)) == pytest.approx(float(gold["dl_cm"]), rel=5e-7)


def test_04_bands(monkeypatch):
    seen = []

    class Rec:
        tmin = 5.0

        def configure(self, v):
            return self.tmin

        def lnu(self, tau, nu):
            seen.append(nu)
            return 1e28 * (nu / 1e15), 0.0

    monkeypatch.setattr(T, "_engine", lambda cfg, bifout: Rec())
    pin = {"fout": "bifout"}
    p = dict(bh_M6=1.0, star_mstar=1.0, bh_eta=0.1, fv=1.0)
    # explicit wavelength, bare number, filter: the wavelength the library is called at (z = 0)
    m = T.tide_model(["myband"], redshift=1e-4, dl_cm=1e26, band_wavelengths={"myband": 5000.0}, pin=pin)
    m.predict(p, [50.0], ["myband"])
    assert seen[-1] == pytest.approx(C_AA / 5000.0 * (1 + 1e-4), rel=1e-14)
    m.predict(p, [50.0], ["6166"])
    assert seen[-1] == pytest.approx(C_AA / 6166.0 * (1 + 1e-4), rel=1e-14)
    ms = T.tide_model(["sdssg"], redshift=1e-4, dl_cm=1e26, pin=pin)
    ms.predict(p, [50.0], ["sdssg"])
    lam_eff = C_AA / (seen[-1] / (1 + 1e-4))
    assert abs(lam_eff - 4686.0) < 3.0, lam_eff                      # measured 4685.2 A
    # n_nodes: the real passband, 16 Gauss nodes, weights summing to one
    seen.clear()
    m16 = T.tide_model(["sdssg"], redshift=1e-4, dl_cm=1e26, pin=pin, n_nodes=16)
    f16 = m16.predict(p, [50.0], ["sdssg"])
    assert len(seen) == 16
    lam, w = T._nodes("sdssg", 16, {}, None, None)
    assert w.sum() == pytest.approx(1.0, rel=1e-14) and lam.size == 16
    ref = sum(wk * 1e28 * (C_AA / lk * (1 + 1e-4) / 1e15) for lk, wk in zip(lam, w))
    assert f16[0] == pytest.approx(1e23 * (1 + 1e-4) * ref / (4 * np.pi * 1e52), rel=1e-12)
    with pytest.raises(ValueError):                                  # a band that names no filter
        T.tide_model(["not_a_filter_name"], redshift=0.05, pin=pin)


def test_05_parameters_and_arguments():
    m = T.tide_model(redshift=0.05)
    assert m.parameters == ["bh_M6", "star_mstar", "bh_eta", "fout", "fv"] and m.predict_jax is None
    assert m.default_prior.names == m.parameters
    b = m.default_prior.bounds
    assert b["bh_M6"] == (0.1, 30.0) and b["star_mstar"] == (0.1, 10.0) and b["bh_eta"] == (0.01, 0.4)
    assert b["fout"] == (0.05, 1.0) and b["fv"] == (1.0, 3.0)
    assert m.predict.pinned == {"eta_r": 0.0, "beta_fulldisrupt": 1.85}     # tidepy's default (U8)
    full = T.tide_model(redshift=0.05, pin={"eta_r": None, "beta_fulldisrupt": None})
    assert full.parameters == list(T.PARAMETERS)
    assert T.default_prior().names == list(T.PARAMETERS)
    c1 = T.tide_model(redshift=0.05, beta_fulldisrupt=1.0)
    assert c1.predict.pinned["beta_fulldisrupt"] == 1.0
    with pytest.raises(ValueError, match="redshift"):
        T.tide_model()                                               # neither fixed nor free
    with pytest.raises(ValueError, match="t_exp"):
        T.tide_model(redshift=0.05, free=["t_exp"])                  # needs a prior
    with pytest.raises(ValueError):
        T.tide_model(redshift=0.05, free=["mass"])
    with pytest.raises(ValueError):
        T.tide_model(redshift=0.05, pin={"mass": 1.0})
    with pytest.raises(ValueError):
        T.tide_model(redshift=0.05, Mdotfb_calc="nope")
    with pytest.raises(ValueError):
        T.tide_model(redshift=0.05, prior=wp.Prior({"eta_r": wp.Uniform(0, 1)}))   # pinned
    with pytest.raises(ValueError):
        T.tide_model(redshift=0.05, free=["redshift"])               # fixed AND free
    # invalid physics is rejected as zero flux, before the library is asked
    good = dict(bh_M6=1.0, star_mstar=1.0, bh_eta=0.1, fout=0.3, fv=1.0)
    for bad in (dict(bh_M6=-1.0), dict(star_mstar=0.0), dict(bh_eta=0.0), dict(bh_eta=1.5),
                dict(fv=0.5), dict(fout=1.5), dict(fout=-0.1), dict(bh_M6=np.nan), dict(fv=np.inf)):
        out = m.predict(dict(good, **bad), [50.0, 100.0], ["6166", "6166"])
        assert np.array_equal(out, [0.0, 0.0]), bad
    with pytest.raises(ValueError, match="bands"):
        m.predict(good, [50.0])
    with pytest.raises(ValueError, match="same shape"):
        m.predict(good, [50.0, 60.0], ["6166"])
    assert m.predict(good, [], []).shape == (0,)


def test_06_no_library_at_import_and_pickling(monkeypatch):
    code = ("import sys, whisper_cbpf.models.tide as t; "
            "assert 'tidepy' not in sys.modules and 'tidepy.tide' not in sys.modules; "
            "assert not t.ENGINES; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-400:]
    m = T.tide_model(["g"], redshift=0.05, pin={"fout": "bifout"}, band_wavelengths={"g": 4686.0})
    blob = pickle.dumps(m.predict)
    assert len(blob) < 1000                                         # measured 432 bytes: plain data
    back = pickle.loads(blob)
    assert repr(back) == repr(m.predict) and back.band_wavelengths == m.predict.band_wavelengths
    assert pickle.loads(pickle.dumps(m)).parameters == m.parameters
    # a missing library is an ImportError that says what to do, not a crash
    monkeypatch.setattr(T, "_TIDE", None)
    monkeypatch.setitem(sys.modules, "tidepy", None)
    monkeypatch.setitem(sys.modules, "tidepy.tide", None)
    with pytest.raises(ImportError, match="libtide.*Dockerfile.tide"):
        T._load_tide()
    assert T.libtide_available() is False
    monkeypatch.undo()


# --------------------------------------------------------------------------------------------
# 7-14. the real library
# --------------------------------------------------------------------------------------------
@needs_lib
def test_07_parity_with_the_oracle_flux(gold):
    names = list(gold["names"])
    ks = [k for k in range(len(names)) if str(gold["kinds"][k]) != "variant"
          or names[k] == "var_fout_0p3_fv2_etar0p2"]
    assert len(ks) == 53
    rng = np.random.default_rng(3)
    worst, count = 0.0, 0
    for k in rng.permutation(ks):                                    # reuse of the library objects
        model, phys = _model(gold, k)
        t, b, ref = _grid(gold, k)
        f = model.predict(phys, t, b)
        lit = np.repeat(gold["t_rel"][k] > gold["tmin"][k] * 1.0001, 6)
        assert np.all(f[~lit] == 0.0)
        ok = lit & (ref > 0)
        assert np.all(f[ok] > 0)
        worst = max(worst, float(np.max(np.abs(f[ok] / ref[ok] - 1))))
        count += int(ok.sum())
    assert count > 9000 and worst < 1e-9, (count, worst)           # measured 4.3e-14
    assert len(T.ENGINES) <= 4


@needs_lib
def test_08_luminosity_tmin_bolometric_photosphere_and_variants(gold):
    names = list(gold["names"])
    bn = np.array(gold["band_names"])
    nu = gold["nu_bands"]
    worst = dict(lnu=0.0, tmin=0.0, bol=0.0, T=0.0, R=0.0)
    for k in range(len(names)):
        phys, pin, cfg = _spec(gold, k)
        kw = dict(bh_M6=phys["bh_M6"], star_mstar=phys["star_mstar"], bh_eta=phys["bh_eta"],
                  fv=phys["fv"], fout=phys.get("fout", "bifout"), eta_r=pin["eta_r"],
                  beta_fulldisrupt=pin["beta_fulldisrupt"], **cfg)
        tmin = T.tide_t_min(**kw)
        worst["tmin"] = max(worst["tmin"], abs(tmin / gold["tmin"][k] - 1))
        t = gold["t_rel"][k]
        lw, ld = T.tide_luminosity_nu(t[:, None], nu[None, :], **kw)
        lit = (t > tmin * 1.0001)[:, None] & np.ones(6, bool)
        for got, ref in ((lw, gold["lw_rel"][k]), (ld, gold["ld_rel"][k])):
            ok = lit & (ref > 0)
            assert np.all(got[~lit] == 0.0) and np.all(got[ok] > 0)
            assert np.array_equal(got[lit] > 0, ref[lit] > 0)         # a zero disc is a zero disc
            if ok.any():
                worst["lnu"] = max(worst["lnu"], float(np.max(np.abs(got[ok] / ref[ok] - 1))))
        if k % 6 == 0 or str(gold["kinds"][k]) == "variant":
            sub = slice(None, None, 3)
            bw, bd = T.tide_bolometric_luminosity(t[sub], **kw)
            for got, ref in ((bw, gold["lbol_wind_rel"][k][sub]), (bd, gold["lbol_disc_rel"][k][sub])):
                ok = ref > 0
                if ok.any():
                    worst["bol"] = max(worst["bol"], float(np.max(np.abs(got[ok] / ref[ok] - 1))))
            Tw, Rw = T.tide_wind_photosphere(t[sub], **kw)
            ok = np.isfinite(gold["T_wind_rel"][k][sub])
            assert np.array_equal(np.isfinite(Tw), ok)
            worst["T"] = max(worst["T"], float(np.max(np.abs(Tw[ok] / gold["T_wind_rel"][k][sub][ok] - 1))))
            worst["R"] = max(worst["R"], float(np.max(np.abs(Rw[ok] / gold["R_wind_rel"][k][sub][ok] - 1))))
    assert max(worst.values()) < 1e-9, worst                         # measured: all below 1e-12
    for name in ("var_mdotfb_default", "var_mdotfb_L09_const", "var_tmin_with_rp", "var_tmin_GR13",
                 "var_politrop_5per3", "var_white_dwarf"):
        assert name in names


@needs_lib
def test_09_known_answers_live(gold):
    t100 = T.tide_t_min(beta_fulldisrupt=1.0)
    t185 = T.tide_t_min(beta_fulldisrupt=1.85)
    assert t100 == pytest.approx(41.0076, abs=5e-5) and t185 == pytest.approx(6.4766, abs=5e-5)
    assert t100 / t185 == pytest.approx(1.85 ** 3, rel=2e-5)
    # the notebook anchor, on the golden's own grid points around the peak
    tt = gold["notebook_t_anchor_beta185"]
    L = gold["notebook_lbol_anchor_beta185"]
    i = int(np.argmax(L))
    sel = np.arange(i - 4, i + 5)
    lw, ld = T.tide_bolometric_luminosity(tt[sel])
    assert np.max(np.abs((lw + ld) / L[sel] - 1)) < 1e-9
    assert int(np.argmax(lw + ld)) == 4 and 37.6 < tt[i] < 37.8
    # scalings: t_min ~ M_BH^(1/2) (exact), t_peak ~ M_BH^0.5, t_peak independent of eta and fout
    M = np.array([0.3, 1.0, 3.0, 10.0])
    tmin = np.array([T.tide_t_min(bh_M6=m) for m in M])
    assert np.polyfit(np.log(M), np.log(tmin), 1)[0] == pytest.approx(0.5, abs=1e-9)

    def peak(**kw):
        tmn = T.tide_t_min(**kw)
        tg = np.geomspace(tmn * 1.2, tmn * 15, 40)
        lw, ld = T.tide_bolometric_luminosity(tg, **kw)
        j = int(np.argmax(lw + ld))
        c = np.polyfit(np.log(tg[j - 1:j + 2]), np.log((lw + ld)[j - 1:j + 2]), 2)
        return float(np.exp(-c[1] / (2 * c[0])))

    tp = np.array([peak(bh_M6=m) for m in M])
    assert np.polyfit(np.log(M), np.log(tp), 1)[0] == pytest.approx(0.5, abs=0.01)   # measured 0.4999
    base = peak()
    for kw in (dict(bh_eta=0.03), dict(bh_eta=0.3), dict(fout=0.1), dict(fout=1.0)):
        assert abs(peak(**kw) / base - 1) < 1e-3, kw                 # measured below 3e-5


@needs_lib
def test_10_physics_invariants():
    t = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0])
    Tw, Rw = T.tide_wind_photosphere(t)
    lw, ld = T.tide_bolometric_luminosity(t)
    assert np.all(np.isfinite(Tw)) and np.all(np.isfinite(Rw))
    # after the peak (~40 d) the wind photosphere shrinks and heats, monotonically
    assert np.all(np.diff(Rw[2:]) < 0) and np.all(np.diff(Tw[2:]) > 0)
    assert Rw[2] > 1e14 > Rw[-1] * 100 and Tw[-1] > 10 * Tw[2]        # 1.2e14 -> 1.6e11 cm; 7e4 -> 9e5 K
    # the wind dominates at the peak, the disc late
    assert (ld / lw)[2] < 1.0 and (ld / lw)[6] > 2.0 and (ld / lw)[7] > 2.0     # 0.39, 3.7, 2.7
    # a more efficient (larger eta) disc is brighter, the peak time is not moved
    lo = T.tide_bolometric_luminosity([40.0], bh_eta=0.03)
    hi = T.tide_bolometric_luminosity([40.0], bh_eta=0.3)
    assert sum(hi) > sum(lo)
    assert np.all(np.isfinite(lw + ld)) and np.all(lw + ld > 0)


@needs_lib
def test_11_pickling_and_spawn(gold):
    import multiprocessing as mp

    model = T.tide_model(["g", "r"], redshift=0.05, default_system="sdss",
                         band_wavelengths={"g": 4686.0, "r": 6166.0})
    rng = np.random.default_rng(5)
    ps = [model.default_prior.sample(rng) for _ in range(3)]
    t = np.array([20.0, 50.0, 90.0, 200.0, 400.0])
    b = np.array(["g", "r", "g", "r", "g"])
    mine = [model.predict(p, t, b) for p in ps]
    back = pickle.loads(pickle.dumps(model.predict))
    assert all(np.array_equal(back(p, t, b), m) for p, m in zip(ps, mine))
    with mp.get_context("spawn").Pool(2) as pool:
        jobs = [pool.apply_async(model.predict, (p, t, b)) for p in ps + ps]
        theirs = [j.get(timeout=300) for j in jobs]
    assert all(np.array_equal(a, m) for a, m in zip(theirs, mine + mine))     # bit for bit
    assert all(np.all(m > 0) for m in mine)


@needs_lib
def test_12_registration_and_cpu_samplers():
    name = "tide_test_registration_unique"
    m = T.register_tide(["sdssg", "sdssr"], redshift=0.05, name=name,
                        pin={"fout": "bifout"}, band_wavelengths={"sdssg": 4686.0, "sdssr": 6166.0})
    assert name in wp.list_models() and wp.get_model(name) is m
    assert m.predict_jax is None and m.parameters == ["bh_M6", "star_mstar", "bh_eta", "fv"]
    truth = dict(bh_M6=2.0, star_mstar=1.0, bh_eta=0.1, fv=1.5)
    t = np.linspace(30.0, 300.0, 24)
    b = np.array(["sdssg", "sdssr"] * 12)
    flux = m.predict(truth, t, b)
    assert np.all(flux > 0)
    err = 0.05 * flux
    lc = wp.LightCurve(time=t, band=b, flux=flux + np.random.default_rng(0).normal(0, err), flux_err=err,
                       name="tide_synth")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        abc = wp.fit_ABC(lc, name, n_simulations=200, quantile=0.1, n_jobs=2, seed=1)
        mc = wp.fit_MCMC(lc, name, nwalkers=12, nsteps=30, burnin=10, seed=1, init="prior")
    assert abc.n_samples > 0 and set(abc.samples.columns) >= set(m.parameters)
    assert mc.n_samples > 0 and set(mc.samples.columns) >= set(m.parameters)
    wp.models._REGISTRY.pop(name, None)


@needs_lib
def test_13_robustness():
    m = T.tide_model(redshift=0.05, band_wavelengths={"g": 4686.0, "r": 6166.0})
    rng = np.random.default_rng(11)
    n_eng = len(T.ENGINES)
    for scale in (1.0, 10.0):                                        # the prior box, then 10x wider
        bad = 0
        for _ in range(300):
            p = m.default_prior.sample(rng)
            if scale > 1.0:
                p = dict(bh_M6=10 ** rng.uniform(-2, 2.5), star_mstar=10 ** rng.uniform(-1.3, 1.5),
                         bh_eta=10 ** rng.uniform(-3, -0.3), fout=rng.uniform(0, 1), fv=rng.uniform(1, 10))
            t = np.sort(10 ** rng.uniform(0, 5.5, 12))
            f = m.predict(p, t, rng.choice(["g", "r"], 12))
            bad += int(not np.all(np.isfinite(f)) or np.any(f < 0))
        assert bad == 0, (scale, bad)
    assert len(T.ENGINES) <= max(n_eng, 2) + 2                       # the same objects, reused


@needs_lib
def test_14_upstream_behaviour_is_reproduced():
    """The quirks the model keeps, asserted on the original library so a fixed upstream shows up."""
    from tidepy import tide as tm
    from tidepy import Disk_part_of_lc, Parameters, Wind_part_of_lc, Light_curve_of_tde

    q = contextlib.redirect_stdout(io.StringIO())
    nu = 6e14
    with q:
        p = Parameters()
        w = Wind_part_of_lc(p)
        l09 = w.wind_luminosity_at_time_nu(60.0, nu)
        p.Mdotfb_calc = "default"
        p.param_init()
        after_default = w.wind_luminosity_at_time_nu(60.0, nu)
        p.fout = "0.3"
        p.param_init()
        after_fout = w.wind_luminosity_at_time_nu(60.0, nu)
    assert after_default == l09           # U8: 'default' set after __init__ is ignored (L09 stays)
    assert after_fout == l09              # a numeric fout set after __init__ is ignored (bifout stays)
    # ... and the model honours both (the notebook's route), measured 3.1e27 -> 9.6e26 at 60 d
    d = T.tide_luminosity_nu(60.0, nu, Mdotfb_calc="default")[0]
    n = T.tide_luminosity_nu(60.0, nu, fout=0.3)[0]
    b = T.tide_luminosity_nu(60.0, nu)[0]
    assert abs(b / l09 - 1) < 1e-12 and d < 0.5 * b and abs(n / b - 1) > 0.5
    # N is read when the disc object is created
    with q:
        p = Parameters()
        disc = Disk_part_of_lc(p)
        ref = disc.disk_luminosity_at_time_nu(100.0, nu)
        p.N = 50
        p.param_init()
        same = disc.disk_luminosity_at_time_nu(100.0, nu)
        coarse = Disk_part_of_lc(p).disk_luminosity_at_time_nu(100.0, nu)
    assert same == ref and abs(coarse / ref - 1) > 1e-4               # 2.7e-3
    # diffusion: NaN at the first epoch, one of 388
    with q, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = Parameters()
        p.tend = 200.0
        t, diff = Light_curve_of_tde(p).diffused_light_curve()
    assert np.isnan(diff[0]) and int(np.isnan(diff).sum()) == 1 and np.all(np.isfinite(diff[1:]))
    assert tm.lib is not None
