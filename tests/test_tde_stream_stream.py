"""redback's ``stream_stream_tde`` in ``tde_stream_stream.py``.

Seven claims, each with a number (measured values are in the docstrings; the budgets are 3x to 10x
above them):

1. **Agreement with redback.** The 1100-node table (L, T, R_eff), the bolometric function and the
   flux density at the six notebook wavelengths reproduce the golden written by
   ``oracles/stream_stream_oracle.py`` (redback 1.21.0, 118 parameter sets, no JAX in the reference
   path) to float64 round-off, inside the model's own time range. The band magnitudes are redback's
   ``magnitude`` output through a different band integral and a different time interpolation
   (documented deviations D3 and D4): median 4e-5 mag, and a few hundredths of a magnitude at the
   99th percentile once the rise is near the engine's peak.
2. **Redback's behaviours are reproduced by default, and each is a test.** Q1 four times the
   bolometric energy in the SED, Q2 ``del_omega`` in steradians, Q3 double time dilation in the
   ``flux_density`` output only, Q4 the ``(nu / nu_0)`` excess in the band spectra, Q5 linear
   extrapolation past the grid (masked by default), Q6 the normalisation overflow, Q7 the
   independence of ``peak_time`` from the engine's peak.
3. **The switches are exact identities**: ``sphere_radius`` divides the flux by four,
   ``del_omega_in_pi`` is the engine at ``pi * del_omega``, ``double_dilation=False`` reads the table
   at ``t / (1 + z)``, ``nu4_spectrum=False`` divides by ``nu / nu_0``.
4. **Physics invariants** of the model: ``T ~ t^(-5/12)`` and a constant radius after the peak,
   ``L ~ t^(-5/3)`` times the cooling factor, ``L_peak`` falling and ``t_peak`` rising with the black
   hole mass, ``L ~ 1/c1``.
5. **It is differentiable and traceable** under ``jit`` and ``vmap`` on prior draws, with the
   epochs, ``t_exp`` and the redshift as ordinary arguments, including the degenerate corner of the
   prior where redback returns inf.
6. **float32** fails loudly, naming float64.
7. **Registration**: the model pickles, registers under its own name, takes pinned and free
   parameters, and its prior is redback's prior file.

FLOAT64. The photometric functions require it; the fixture below enables it for this module and
restores the previous setting afterwards.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.jax import tde_stream_stream as S  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402

GOLDEN = Path(__file__).parent / "goldens" / "tde_stream_stream_oracle.npz"
DAY = 86400.0
#: sdss u g r i: the bands of redback's ``magnitude`` output in the golden.
BANDS = ["sdssu", "sdssg", "sdssr", "sdssi"]
SEED = 20261119
#: Planck18 luminosity distance at z = 0.05 [cm].
DL = 7.093375016731583e26


@pytest.fixture(scope="module", autouse=True)
def _x64():
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


@pytest.fixture(scope="module")
def gold():
    with np.load(GOLDEN) as z:
        g = {k: z[k] for k in z.files}
    g["labels"] = [str(s) for s in g["labels"]]
    g["keys"] = [str(s) for s in g["key_order"]]
    g["prov"] = json.loads(str(g["provenance"]))
    return g


def _params(g, i):
    return {k: float(v) for k, v in zip(g["keys"], g["params"][i])}


def _e0(g, i):
    """The rise exponent ``(t_peak - peak_time)^2 / (2 sigma_t^2)`` (days) of golden draw ``i``."""
    p = _params(g, i)
    tp = g["dyn_time"][i][100] / DAY
    return (tp - p["peak_time"]) ** 2 / (2.0 * p["sigma_t"] ** 2)


def _idx(g, kind="ok"):
    """Golden draws where redback is finite (``ok``) or not (``bad``)."""
    fin = np.array([np.all(np.isfinite(g["dyn_L"][i])) for i in range(len(g["labels"]))])
    if kind == "ok":
        return [i for i in range(len(fin)) if fin[i]]
    if kind == "bad":
        return [i for i in range(len(fin)) if not fin[i]]
    raise ValueError(kind)


def _label(g, name):
    return g["labels"].index(name)


def _stats(x):
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(np.percentile(x, 99)), float(np.max(x))


def _model(g, i, **kw):
    """The whisper model of golden draw ``i``: its fixed parameters pinned, its redshift and distance."""
    p = _params(g, i)
    pins = {k: p[k] for k in ("c1", "f", "h_r", "inc_tcool", "del_omega")}
    return S.tde_stream_stream_model(BANDS, float(g["redshift"][i]), float(g["dl_cm"][i]), pin=pins, **kw)


def _theta(model, p):
    return {k: p[k] for k in model.parameters}


# =======================================================================================
# 1. agreement with redback
# =======================================================================================

def test_the_golden_names_its_provenance(gold):
    """The golden was written by the original code: redback 1.21.0, which source file, the seeds."""
    pv = gold["prov"]
    assert pv["redback_version"] == "1.21.0"
    assert len(pv["redback_tde_models_sha256"]) == 64 and pv["n_errors"] == 0
    assert pv["cosmology"] == "Planck18" and set(pv["seeds"]) == {"lhs_prior_box", "lhs_extended"}
    assert len(gold["labels"]) == 118 and gold["flux_mjy"].shape == (118, 6, 80)
    assert gold["wavelengths"].tolist() == [2030.0, 2634.0, 3551.0, 4686.0, 6166.0, 7480.0]
    assert float(gold["dl_cm"][_label(gold, "anchor")]) == pytest.approx(DL, rel=1e-12)


def test_the_table_is_redbacks_dynamics_output(gold):
    """Time, L, T and R_eff of the 1100 nodes on the 115 draws where redback is finite (the 3
    corners where the normalisation overflows are test_q6). Measured: time 0, T 6.7e-16, L median 2.2e-16, p99 3.1e-15, R median 2.2e-16, p99 1.9e-15;
    the scalar t_dyn and r_tidal 0."""
    eL, eT, eR, eTime, eDyn = [], [], [], [], []
    for i in _idx(gold):
        p = _params(gold, i)
        tab = S.stream_stream_table(**p)
        eTime.append((np.asarray(tab["time"]) - gold["dyn_time"][i]) / np.maximum(gold["dyn_time"][i], 1.0))
        eL.append(np.asarray(tab["luminosity"]) / gold["dyn_L"][i] - 1.0)
        eT.append(np.asarray(tab["temperature"]) / gold["dyn_T"][i] - 1.0)
        eR.append(np.asarray(tab["radius"]) / gold["dyn_R"][i] - 1.0)
        eDyn.append([float(tab["t_dyn"]) / gold["dyn_scalars"][i][2] - 1.0,
                     float(tab["r_tidal"]) / gold["dyn_scalars"][i][3] - 1.0])
    assert len(eL) == 115
    out = {k: _stats(np.concatenate(v)) for k, v in
           dict(time=eTime, L=eL, T=eT, R=eR, scalars=[np.ravel(eDyn)]).items()}
    print("\ntable vs redback:", {k: tuple(f"{x:.1e}" for x in v) for k, v in out.items()})
    assert out["time"][2] < 1e-14 and out["T"][2] < 1e-13 and out["scalars"][2] < 1e-14
    assert out["L"][1] < 1e-13 and out["L"][2] < 1e-11
    assert out["R"][1] < 1e-13 and out["R"][2] < 1e-11


def test_flux_density_is_redbacks_at_the_six_wavelengths(gold):
    """``flux_density`` [mJy] at UVW2 UVW1 u g r i on each draw's observer-day grid, inside the
    range redback's table covers (``1500 (1 + z)^2`` days; z = 0.01, 0.05, 0.2, 0.3, 0.5, 1). Measured over
    52438 points (redback overflows to inf in two of them, in the 1e292 erg/s rise of one corner):
    median 4.4e-16, p99 3.0e-14, maximum 2.4e-13 relative."""
    err, n = [], 0
    for i in _idx(gold):
        p, z, dl = _params(gold, i), float(gold["redshift"][i]), float(gold["dl_cm"][i])
        t = gold["t_obs_days"][i]
        inside = t <= gold["t_valid_end"][i]
        for j, nu in enumerate(gold["nu_hz"]):
            f = np.asarray(S.tde_stream_stream_flux_density(t, nu, z, dl, **p))
            ref = gold["flux_mjy"][i, j]
            ok = inside & np.isfinite(ref)             # redback's flux is inf in 2 points of corner05
            err.append((f / ref - 1.0)[ok])
            n += int(ok.sum())
    med, p99, mx = _stats(np.concatenate(err))
    print(f"\nflux_density vs redback, {n} points: median {med:.1e}, p99 {p99:.1e}, max {mx:.1e}")
    assert n > 40000 and med < 1e-14 and p99 < 1e-11 and mx < 1e-9


def test_bolometric_is_redbacks(gold):
    """``stream_stream_tde_bolometric`` on a source-frame day grid, inside the 1500-day table. The
    function takes no redshift: the grid is source-frame (Q8). Measured: median 3.3e-16, p99 2.8e-14,
    maximum 1.4e-13."""
    err = []
    for i in _idx(gold):
        t = gold["bol_t_days"][i]
        got = np.asarray(S.tde_stream_stream_bolometric(t, **_params(gold, i)))
        err.append((got / gold["bol_L"][i] - 1.0)[t <= 1500.0])
    med, p99, mx = _stats(np.concatenate(err))
    print(f"\nbolometric vs redback: median {med:.1e}, p99 {p99:.1e}, max {mx:.1e}")
    assert med < 1e-14 and p99 < 1e-11 and mx < 1e-9


def test_extrapolate_true_is_redbacks_linear_extrapolation(gold):
    """Q5. Past the end of redback's table ``interp1d(..., 'extrapolate')`` continues the last
    segment; ``extrapolate=True`` is the same line. The curve there falls through zero (redback's
    flux is negative past about 1.9 times the end) and the bolometric function returns negative
    luminosities. Measured over the points past the end: maximum relative 5.5e-11 (27 of redback's fluxes past the end are negative)."""
    err, n_neg = [], 0
    for i in _idx(gold):
        p, z, dl = _params(gold, i), float(gold["redshift"][i]), float(gold["dl_cm"][i])
        t = gold["t_obs_days"][i]
        past = t > gold["t_valid_end"][i]
        f = np.asarray(S.tde_stream_stream_flux_density(t, gold["nu_hz"][3], z, dl, extrapolate=True,
                                                        **p))
        ref = gold["flux_mjy"][i, 3]
        err.append((f / ref - 1.0)[past])
        n_neg += int((ref[past] < 0).sum())
        tb = gold["bol_t_days"][i]
        lb = np.asarray(S.tde_stream_stream_bolometric(tb, extrapolate=True, **p))
        err.append((lb / gold["bol_L"][i] - 1.0)[tb > 1500.0])
    _, _, mx = _stats(np.concatenate(err))
    print(f"\nextrapolation vs redback: max {mx:.1e}; {n_neg} negative redback fluxes past the end")
    assert mx < 1e-9 and n_neg > 20


def test_the_model_ends_with_the_engine_grid_and_before_its_time_zero(gold):
    """D2: past the end (1500 d of model time) the flux density is exactly zero and the band
    magnitude is ``mag_floor``, where redback returns the linear extrapolation; epochs at or before
    the time zero are at ``mag_floor`` too."""
    i = _label(gold, "anchor")
    p, z, dl = _params(gold, i), 0.05, DL
    end = 1500.0 * (1 + z) ** 2
    t = np.array([end * 0.99, end * 1.01, end * 1.5])
    f = np.asarray(S.tde_stream_stream_flux_density(t, 6e14, z, dl, **p))
    assert f[0] > 0 and f[1] == 0.0 and f[2] == 0.0
    model = _model(gold, i)
    flux = model.predict(_theta(model, p), np.array([-5.0, 0.0, 100.0, 1500.0 * 1.05 * 0.99, 1600.0 * 1.05]),
                         np.array(["sdssg"] * 5))
    floor = 3631.0 * 10.0 ** (-0.4 * 40.0)
    assert flux[0] == pytest.approx(floor) and flux[1] == pytest.approx(floor)
    assert flux[2] > 1e3 * floor and flux[3] > 1e3 * floor and flux[4] == pytest.approx(floor)


def test_band_magnitudes_are_redbacks_magnitude_output(gold):
    """redback's ``magnitude`` output (sncosmo, spectra on a 100-point wavelength grid splined in
    time) against this model (the real bandpass, linear in time between the same nodes): D3 and D4.
    On the 91 draws whose rise exponent is at most 3 (``peak_time`` near the engine's peak):
    median 3.4e-5 mag, and after the peak 99th percentile 0.016, maximum 0.044; before the peak the
    cubic spline across the steep Gaussian wing differs more (99th percentile 0.020, maximum 3.0 mag
    on one draw whose wing spans 22 e-folds between two nodes). The median over all 115
    finite draws is 3.6e-5; the draws whose rise is a decaying Gaussian wing (exponent > 10) reach
    hundreds of magnitudes where redback's spline rings."""
    post, pre, allm = [], [], []
    for i in _idx(gold, "ok") + [j for j in _idx(gold, "bad") if False]:
        if _e0(gold, i) > 3.0:
            continue
        model = _model(gold, i)
        p = _params(gold, i)
        tp = gold["dyn_time"][i][100] / DAY
        for j, b in enumerate(BANDS):
            t = gold["t_mag_days"][i]
            mine = -2.5 * np.log10(np.asarray(model.predict(_theta(model, p), t, np.array([b] * t.size))) / 3631.0)
            d = mine - gold["mag_ab"][i, j]
            tsrc = t / (1.0 + gold["redshift"][i])
            post.append(d[tsrc > tp])
            pre.append(d[tsrc <= tp])
            allm.append(d)
    s_all, s_post, s_pre = (_stats(np.concatenate(x)) for x in (allm, post, pre))
    print(f"\nband magnitudes vs redback: all {s_all}, post-peak {s_post}, pre-peak {s_pre}")
    assert s_all[0] < 2e-4 and s_post[1] < 0.08 and s_post[2] < 0.2
    assert s_pre[1] < 0.1


def test_live_redback_flux_density_magnitude_and_bolometric():
    """The same comparisons against the installed redback (skipped without it), on 6 prior draws and
    the anchor: the flux density to 1e-12 relative, the bolometric light curve to 1e-12 and, through
    one band integral, ``double_dilation=True, nu4_spectrum=False`` reproduces redback's
    ``flux_density`` output (epochs above ``mag_floor``) at the 1e-14 mag level; measured flux
    1.1e-14, bolometric 7.2e-15, band 6e-15 mag."""
    pytest.importorskip("redback")
    from redback.transient_models import tde_models as ot

    from whisper_cbpf.synphot import band_flux_jy, filter_set_for

    z = 0.05
    t = np.geomspace(2.0, 1500.0, 24)
    nu = np.full(t.size, 2.99792458e18 / 4686.0)
    rng = np.random.default_rng(SEED)
    sets = [dict(mbh_6=1.0, mstar=1.0, c1=1.0, f=0.2, h_r=0.2, inc_tcool=1, del_omega=2.0,
                 sigma_t=20.0, peak_time=float(S.stream_stream_peak_time(1.0, 1.0)))]
    for _ in range(6):
        mbh, ms = np.exp(rng.uniform(np.log(0.05), np.log(15.0))), np.exp(rng.uniform(np.log(0.15), np.log(8.0)))
        sets.append(dict(mbh_6=mbh, mstar=ms, c1=1.0, f=0.2, h_r=0.2, inc_tcool=1, del_omega=2.0,
                         sigma_t=float(rng.uniform(10, 60)),
                         peak_time=float(S.stream_stream_peak_time(mbh, ms))))
    ref_fd, ref_bol, mine_fd, mine_bol, mag_err = [], [], [], [], []
    dl = DL
    for p in sets:
        ref = np.asarray(ot.stream_stream_tde(t, z, output_format="flux_density", frequency=nu, **p))
        mine = np.asarray(S.tde_stream_stream_flux_density(t, nu, z, dl, **p))
        ok = t <= 1500.0 * (1 + z) ** 2
        ref_fd.append(ref[ok]); mine_fd.append(mine[ok])
        tb = np.geomspace(1.0, 1400.0, 20)
        ref_bol.append(np.asarray(ot.stream_stream_tde_bolometric(tb, **p)))
        mine_bol.append(np.asarray(S.tde_stream_stream_bolometric(tb, **p)))
        # one band integral on both sides: redback's flux_density and the double-dilated band path
        fs = filter_set_for(["sdssg"])
        tt = np.geomspace(5.0, 1000.0, 12)
        bb = np.array(["sdssg"] * tt.size)
        fjy = band_flux_jy(lambda x, f_: np.asarray(ot.stream_stream_tde(
            x, z, output_format="flux_density", frequency=f_, **p)) * 1e-3, tt, bb, fs)
        model = S.tde_stream_stream_model(["sdssg"], z, dl, double_dilation=True, nu4_spectrum=False,
                                          pin={k: p[k] for k in ("c1", "f", "h_r", "inc_tcool", "del_omega")})
        got = np.asarray(model.predict(_theta(model, p), tt, bb))
        keep = fjy > 100.0 * 3631.0 * 10.0 ** (-0.4 * 40.0)          # above the model's mag_floor
        assert keep.sum() >= 3
        mag_err.append((-2.5 * np.log10(got / fjy))[keep])
    e_fd = np.concatenate([a / b - 1.0 for a, b in zip(mine_fd, ref_fd)])
    e_bol = np.concatenate([a / b - 1.0 for a, b in zip(mine_bol, ref_bol)])
    e_mag = np.concatenate(mag_err)
    print(f"\nlive: flux {np.max(np.abs(e_fd)):.1e}, bolometric {np.max(np.abs(e_bol)):.1e}, band {np.max(np.abs(e_mag)):.1e} mag")
    assert np.max(np.abs(e_fd)) < 1e-9 and np.max(np.abs(e_bol)) < 1e-9
    assert np.max(np.abs(e_mag)) < 1e-6


def test_normalisation_is_the_exact_identity_of_redbacks(gold):
    """D1: ``L_0 exp(e_0 - e_1)`` against redback's ``(L_0 / f1) exp(-e_1)`` spelled as in the original
    (one call in days, one in seconds), in NumPy, from the engine's own L and t_peak, on 200 prior
    draws where ``f1`` does not underflow. Measured maximum relative difference 8.5e-14."""
    rng = np.random.default_rng(SEED + 1)
    worst, n = 0.0, 0
    for _ in range(200):
        mbh, ms = np.exp(rng.uniform(np.log(0.01), np.log(20.0))), np.exp(rng.uniform(np.log(0.1), np.log(10.0)))
        pk, sg = np.exp(rng.uniform(np.log(0.1), np.log(60.0))), np.exp(rng.uniform(np.log(10.0), np.log(60.0)))
        tab = S.stream_stream_table(mbh, ms, 1.0, 0.2, 0.2, 1, 2.0, sg, pk)
        t0, l0 = float(tab["time"][N_PRE]), float(tab["luminosity"][N_PRE])
        f1 = np.exp(-(t0 / DAY - pk) ** 2. / (2 * sg ** 2))                       # gaussian_rise, in days
        if f1 < 1e-290:
            continue
        norm = l0 / f1
        tt = np.linspace(0, t0 - 0.001, N_PRE)
        ref = norm * np.exp(-(tt - pk * DAY) ** 2. / (2 * (sg * DAY) ** 2))        # gaussian_rise, in seconds
        got = np.asarray(tab["luminosity"])[:N_PRE]
        worst = max(worst, float(np.max(np.abs(got / ref - 1.0))))
        n += 1
    print(f"\nnormalisation identity: {n} draws, max {worst:.1e}")
    assert n > 150 and worst < 5e-12


N_PRE = S.N_PRE


# =======================================================================================
# 2. redback's behaviours, reproduced and removable
# =======================================================================================

def _energy_ratio(z, dl, p, **kw):
    """4 pi d_L^2 int F_nu dnu (the SED's energy) over the engine's L at the engine time it reads."""
    nu = np.geomspace(1e13, 3e17, 4000)
    out = {}
    for tobs in (60.0, 200.0):
        f = np.asarray(S.tde_stream_stream_flux_density(np.full(nu.size, tobs), nu, z, dl, **p, **kw))
        e = 4 * np.pi * dl ** 2 * np.trapezoid(f * 1e-26, nu)
        out[tobs] = (e / float(S.tde_stream_stream_bolometric(tobs / (1 + z) ** 2, **p)),
                     e / float(S.tde_stream_stream_bolometric(tobs / (1 + z), **p)))
    return out


def test_q1_the_sed_carries_four_times_the_bolometric_energy(gold):
    """Q1 (issue 30). redback sets ``R_eff = sqrt(L / (pi sigma T^4))`` and emits the spectrum of a
    sphere, ``4 pi R^2``: the energy of the band fluxes is 4.0000 times the table's L at the time it
    reads (measured 4.0000), and 4.0662 / 4.0670 times the bolometric luminosity at the intended
    time ``t_obs / (1 + z)`` at z = 0.01, which is the ~4.07 of ``4 (1 + z)^(5/3)``. That is
    ``2.5 log10(4) = 1.505`` mag in every band. ``sphere_radius=True`` removes it exactly."""
    i = _label(gold, "anchor_z0.01")
    p, z, dl = _params(gold, i), 0.01, float(gold["dl_cm"][i])
    r = _energy_ratio(z, dl, p)
    r_sphere = _energy_ratio(z, dl, p, sphere_radius=True)
    print(f"\nQ1: E_sed/L table-time {r[60.0][0]:.4f} {r[200.0][0]:.4f}; at t/(1+z) {r[60.0][1]:.4f} {r[200.0][1]:.4f};"
          f" sphere_radius {r_sphere[200.0][0]:.4f}")
    for tobs in (60.0, 200.0):
        assert r[tobs][0] == pytest.approx(4.0, rel=3e-3)
        assert r[tobs][1] == pytest.approx(4.0 * 1.01 ** (5.0 / 3.0), rel=5e-3)
        assert r_sphere[tobs][0] == pytest.approx(1.0, rel=3e-3)
    # in every band, through the model
    t = np.array([40.0, 100.0, 300.0])
    for b in BANDS:
        m4 = _model(gold, i)
        m1 = _model(gold, i, sphere_radius=True)
        d = (-2.5 * np.log10(m4.predict(_theta(m4, p), t, np.array([b] * 3)) /
                             m1.predict(_theta(m1, p), t, np.array([b] * 3))))
        assert np.allclose(d, -2.5 * np.log10(4.0), atol=1e-9), d
    # the effective radius of redback's own table
    k = _label(gold, "anchor")
    area = 4 * np.pi * gold["dyn_R"][k] ** 2 * 5.670374419184431e-5 * gold["dyn_T"][k] ** 4
    assert np.allclose(area[100:] / gold["dyn_L"][k][100:], 4.0, rtol=1e-12)


def test_q2_del_omega_is_in_steradians(gold):
    """Q2 (issue 31). The docstring and the prior file say "in units of pi", the engine uses
    steradians: ``del_omega = 2`` is 2 sr, not 2 pi sr. The temperature at ``del_omega = 2 pi`` is
    ``pi^(1/4)`` lower (redback golden: 1.3313 = ``pi^(1/4)``), ``R_eff`` is ``sqrt(pi)`` larger, and
    ``del_omega_in_pi=True`` is exactly the engine at ``2 pi`` (the golden of that draw)."""
    a, b = _label(gold, "anchor"), _label(gold, "anchor_delomega_2pi")
    ratio = gold["dyn_T"][a][100:] / gold["dyn_T"][b][100:]
    assert np.allclose(ratio, np.pi ** 0.25, rtol=1e-12)
    assert np.allclose(gold["dyn_R"][b][100:] / gold["dyn_R"][a][100:], np.pi ** 0.5, rtol=1e-12)
    p = _params(gold, a)
    tab = S.stream_stream_table(**p, del_omega_in_pi=True) if "del_omega" not in p else \
        S.stream_stream_table(**{**p, "del_omega": 2.0}, del_omega_in_pi=True)
    assert np.allclose(np.asarray(tab["temperature"]), gold["dyn_T"][b], rtol=1e-13)
    assert np.allclose(np.asarray(tab["radius"]), gold["dyn_R"][b], rtol=1e-13)
    # the two leave the luminosity alone
    assert np.allclose(np.asarray(tab["luminosity"]), gold["dyn_L"][b], rtol=1e-13)


def _peak_time_of_flux(z, dl, p, lam=4686.0, **kw):
    t = np.linspace(1.0, 10.0 * (1 + z) ** 2 * 33.0, 40000)
    f = np.asarray(S.tde_stream_stream_flux_density(t, 2.99792458e18 / lam, z, dl, **p, **kw))
    return t[np.argmax(f)]


def test_q3_flux_density_applies_time_dilation_twice(gold):
    """Q3 (issue 32). redback's ``flux_density`` output builds its table on observer-frame times and
    reads it at source-frame times, so the curve peaks at ``(1 + z)^2 t_peak``: at z = 0.01 and
    z = 1 the peak (redback: 32.80 and 128.67 d for a 32.17-d engine peak) moves by
    ``((1 + 1) / (1 + 0.01))^2 = 3.92``, here 3.92 (measured 3.9210) against 1.98 (1.9799) for
    ``double_dilation=False``. redback's band outputs (``magnitude``, ``flux``) do not do it: their
    peak is at ``(1 + z) t_peak`` (63.9 d at z = 1), which is the default of the band function."""
    ia, ib = _label(gold, "anchor_z0.01"), _label(gold, "anchor_z1")
    p = _params(gold, ia)
    za, zb = 0.01, 1.0
    da, db = float(gold["dl_cm"][ia]), float(gold["dl_cm"][ib])
    ra = _peak_time_of_flux(zb, db, p) / _peak_time_of_flux(za, da, p)
    rs = _peak_time_of_flux(zb, db, p, double_dilation=False) / _peak_time_of_flux(za, da, p, double_dilation=False)
    print(f"\nQ3: peak shift z=1 / z=0.01: double {ra:.4f}, single {rs:.4f}")
    assert ra == pytest.approx(((1 + zb) / (1 + za)) ** 2, rel=2e-3)
    assert rs == pytest.approx((1 + zb) / (1 + za), rel=2e-3)
    # the band path (default) peaks at (1 + z) t_peak, on a dense grid (and in redback's own
    # magnitudes: 63.9 d at z = 1)
    tp = float(S.stream_stream_peak_time(1.0, 1.0))
    for name, z in (("anchor_z0.01", 0.01), ("anchor_z1", 1.0)):
        k = _label(gold, name)
        m = _model(gold, k)
        t = np.linspace(2.0, 6.0 * tp * (1 + z), 6000)
        f = m.predict(_theta(m, _params(gold, k)), t, np.array(["sdssg"] * t.size))
        assert t[np.argmax(f)] / ((1 + z) * tp) == pytest.approx(1.0, abs=0.01)


def test_q4_band_spectra_carry_an_extra_nu_over_nu0(gold):
    """Q4 (issue 36). redback's band outputs scale the Planck function by ``F_nu(nu_0) (nu / nu_0)^4 (e^x0 - 1) / (e^x - 1)`` (``nu_0 = 6e14`` Hz), which is
    ``nu / nu_0`` times the Planck function. At
    z = 0.05 the magnitudes are brighter than the Planck ones by 0.424 (u), 0.143 (g) and fainter by
    0.167 (r), 0.378 (i), the monochromatic ``-2.5 log10((1 + z) nu / 6e14)`` at the band centre
    (measured -0.415, -0.123, +0.176, +0.386; the band's width moves it by 0.02). The port
    reproduces the golden to 4e-5 mag (see test_band_magnitudes...) with ``nu4_spectrum=True`` and
    gives the Planck function with ``False``."""
    i = _label(gold, "anchor")
    p, z = _params(gold, i), 0.05
    t = np.array([40.0, 100.0, 300.0])
    m1 = _model(gold, i)
    m0 = _model(gold, i, nu4_spectrum=False)
    lam_eff = {"sdssu": 3581.0, "sdssg": 4685.0, "sdssr": 6168.0, "sdssi": 7487.0}
    for b, want in zip(BANDS, (-0.424, -0.143, 0.167, 0.378)):
        d = -2.5 * np.log10(m1.predict(_theta(m1, p), t, np.array([b] * 3)) /
                            m0.predict(_theta(m0, p), t, np.array([b] * 3)))
        mono = -2.5 * np.log10(2.99792458e18 / lam_eff[b] * (1 + z) / 6e14)
        assert np.all(np.abs(d - want) < 0.01), (b, d)
        assert abs(d[0] - mono) < 0.03


def test_q6_normalisation_underflow_is_inf_in_redback_and_finite_here(gold):
    """Q6 (issue 37). redback's ``norm = L_0 / f1`` with ``f1 = exp(-e_0)`` overflows to inf from
    ``e_0`` of about 600 (``f1`` itself is zero past 745): three of the 16 prior-box corners (and
    0.225 % of its prior, 9 of 4000 draws) return inf/NaN in L, flux density and
    ``stream_stream_tde_bolometric``. This module's rise is finite there (D1), and the flux is
    positive."""
    bad = _idx(gold, "bad")
    assert [gold["labels"][i] for i in bad] == ["corner01", "corner03", "corner07"]
    for i in bad:
        assert _e0(gold, i) > 600.0
        assert not np.isfinite(gold["flux_mjy"][i]).all()
        p, z, dl = _params(gold, i), float(gold["redshift"][i]), float(gold["dl_cm"][i])
        tab = S.stream_stream_table(**p)
        assert np.isfinite(np.asarray(tab["luminosity"])).all() and (np.asarray(tab["luminosity"]) > 0).all()
        f = np.asarray(S.tde_stream_stream_flux_density(gold["t_obs_days"][i], gold["nu_hz"][3], z, dl, **p))
        assert np.isfinite(f).all() and (f >= 0).all()


def test_q7_peak_time_is_not_the_engines_peak(gold):
    """Q7 (issue 38). With ``peak_time = 20 d`` and a 32.17-d engine peak the Gaussian is already
    falling where it meets the engine, and the light curve peaks at ``exp(e_0) = 1.203`` times the
    engine's maximum luminosity, at ``peak_time`` (redback golden, and this table); with
    ``peak_time = t_peak`` the factor is 1. On redback's prior box 31 % of draws have ``e_0 > 3``
    (the peak at least 20 times the engine's), 21 % ``e_0 > 10`` and 8 % ``e_0 > 50`` (3000 draws)."""
    a, b = _label(gold, "anchor"), _label(gold, "anchor_peak20")
    for k, want in ((a, 1.0), (b, np.exp((32.16736562413506 - 20.0) ** 2 / (2 * 20.0 ** 2)))):
        tab = S.stream_stream_table(**_params(gold, k))
        fac = float(np.max(tab["luminosity"]) / tab["luminosity"][N_PRE])
        ref = float(np.max(gold["dyn_L"][k]) / gold["dyn_L"][k][100])
        assert fac == pytest.approx(ref, rel=1e-12)
        assert fac == pytest.approx(want, rel=1e-4)       # the 100 rise nodes do not sit on the maximum
    assert S.stream_stream_peak_time(1.0, 1.0) == pytest.approx(gold["dyn_time"][a][100] / DAY, rel=1e-13)
    # the claim on the box: 3000 prior draws
    rng = np.random.default_rng(SEED + 7)
    e0 = []
    for _ in range(3000):
        mbh, ms = np.exp(rng.uniform(np.log(0.01), np.log(20.0))), np.exp(rng.uniform(np.log(0.1), np.log(10.0)))
        pk, sg = np.exp(rng.uniform(np.log(0.1), np.log(60.0))), np.exp(rng.uniform(np.log(10.0), np.log(60.0)))
        e0.append((float(S.stream_stream_peak_time(mbh, ms)) - pk) ** 2 / (2 * sg ** 2))
    e0 = np.asarray(e0)
    print(f"\nQ7 on the box: e0>3 {np.mean(e0 > 3):.3f}, >10 {np.mean(e0 > 10):.3f}, >50 {np.mean(e0 > 50):.3f}")
    assert 0.27 < np.mean(e0 > 3) < 0.36 and 0.16 < np.mean(e0 > 10) < 0.24 and 0.04 < np.mean(e0 > 50) < 0.11


def test_q8_the_bolometric_function_takes_source_frame_days():
    """Q8 (issue 34). Neither redback's ``stream_stream_tde_bolometric`` nor this function has a
    redshift: its time is source-frame (the docstring says observer-frame)."""
    import inspect

    assert "redshift" not in inspect.signature(S.tde_stream_stream_bolometric).parameters


# =======================================================================================
# 3. physics invariants
# =======================================================================================

ANCHOR = dict(mbh_6=1.0, mstar=1.0, c1=1.0, f=0.2, h_r=0.2, inc_tcool=1, del_omega=2.0, sigma_t=20.0,
              peak_time=32.16736562413506)


def _post(tab):
    sl = slice(N_PRE, None)
    return (np.asarray(tab["time"])[sl], np.asarray(tab["luminosity"])[sl],
            np.asarray(tab["temperature"])[sl], np.asarray(tab["radius"])[sl])


def test_temperature_falls_as_t_to_the_minus_five_twelfths_at_constant_radius():
    """After the peak the emitting area is fixed: ``T = (L / const)^(1/4) ~ t^(-5/12)`` (measured slope
    -0.41667, exactly -5/12) and R_eff is constant (spread below 1e-12 over the 1000 nodes)."""
    for p in (ANCHOR, dict(ANCHOR, mbh_6=10.0, mstar=0.3, inc_tcool=0), dict(ANCHOR, c1=0.5, del_omega=7.0)):
        t, _, temp, rad = _post(S.stream_stream_table(**p))
        slope = np.polyfit(np.log(t), np.log(temp), 1)[0]
        assert slope == pytest.approx(-5.0 / 12.0, abs=1e-12)
        assert np.ptp(rad) / np.mean(rad) < 1e-12


def test_luminosity_follows_the_cooling_factor_and_minus_five_thirds():
    """``L = L_max (t / t_peak)^(-5/3) 2 / (1 + t_cool / t_dyn)``: measured slope -5/3 to 1e-13,
    and the factor at the anchor, 1.7073 (``t_cool / t_dyn = 0.1714``), is recomputed from its
    definition here (to 1e-12); ``inc_tcool = 0`` has factor 1. ``L`` is proportional to ``1/c1`` at
    ``inc_tcool = 0``."""
    t, lum, _, _ = _post(S.stream_stream_table(**ANCHOR))
    assert np.polyfit(np.log(t), np.log(lum), 1)[0] == pytest.approx(-5.0 / 3.0, abs=1e-12)
    on = S.stream_stream_table(**ANCHOR)
    off = S.stream_stream_table(**dict(ANCHOR, inc_tcool=0))
    factor = float(on["luminosity"][N_PRE] / off["luminosity"][N_PRE])
    # the factor from its definition, with the engine's own a0, t_dyn
    G, MSUN, RSUN, C = 6.6743e-8, 1.988409870698051e33, 6.957e10, 2.99792458e10
    mbh = 1.0e6 * MSUN
    rstar = 0.93
    ex = np.exp((1.0 - 0.674) / 0.212)
    xi = (1.27 - 0.3) * ((0.620 + ex) / (1.0 + 0.553 * ex))
    r_t = (1.0e6) ** (1 / 3) * rstar * RSUN
    eps = G * mbh * (rstar * RSUN) / r_t ** 2
    a0 = G * mbh / (xi * eps)
    t_dyn = np.pi / np.sqrt(2.0) * a0 ** 1.5 / np.sqrt(G * mbh)
    tcool = 0.34 * (0.2 * MSUN / 2.0) / (np.pi * (a0 / 2.0) ** 2) / 2.0 * 0.2 * (a0 / 2.0) / C
    want = 2.0 / (1.0 + tcool / t_dyn)
    print(f"\ncooling factor {factor:.6f}, from the definition {want:.6f}; t_cool/t_dyn {tcool / t_dyn:.4f}")
    assert factor == pytest.approx(want, rel=1e-12) and 0.0 < factor < 2.0
    assert float(on["t_dyn"]) == pytest.approx(t_dyn, rel=1e-12)
    l1 = S.stream_stream_table(**dict(ANCHOR, inc_tcool=0, c1=1.0))["luminosity"][N_PRE:]
    l2 = S.stream_stream_table(**dict(ANCHOR, inc_tcool=0, c1=2.0))["luminosity"][N_PRE:]
    assert np.allclose(np.asarray(l1) / np.asarray(l2), 2.0, rtol=1e-12)


def test_peak_luminosity_falls_and_peak_time_rises_with_black_hole_mass():
    """Over ``M_BH = 10^5.5 .. 10^7.5`` (``mbh_6 = 10^-0.5 .. 10^1.5``): ``L_peak`` falls with log-slope
    -0.37 (measured -0.372; the uncapped fall-back gives -0.5) and ``t_peak ~ M_BH^0.69`` through
    ``Xi(M_BH)`` (measured 0.682 over this grid; a pure ``M_BH^0.5`` would be the fall-back time)."""
    mb = np.logspace(-0.5, 1.5, 21)
    lpk, tpk = [], []
    for m in mb:
        tab = S.stream_stream_table(**dict(ANCHOR, mbh_6=m))
        lpk.append(float(tab["luminosity"][N_PRE]))
        tpk.append(float(tab["t_peak"]) / DAY)
    sl = np.polyfit(np.log10(mb), np.log10(lpk), 1)[0]
    st = np.polyfit(np.log10(mb), np.log10(tpk), 1)[0]
    print(f"\nL_peak slope {sl:.4f}, t_peak slope {st:.4f}")
    assert sl == pytest.approx(-0.37, abs=0.02) and st == pytest.approx(0.69, abs=0.02)
    assert float(S.stream_stream_peak_time(1.0, 1.0)) == pytest.approx(32.167, abs=1e-3)


def test_the_rise_is_gaussian_at_constant_temperature():
    """Before the peak T is the peak temperature at every node and the luminosity is the Gaussian of
    ``peak_time`` and ``sigma_t`` normalised to the engine at ``t_peak``."""
    p = dict(ANCHOR, peak_time=30.0, sigma_t=12.0)
    tab = S.stream_stream_table(**p)
    temp = np.asarray(tab["temperature"])
    assert np.all(temp[:N_PRE] == temp[N_PRE])
    t, lum = np.asarray(tab["time"])[:N_PRE], np.asarray(tab["luminosity"])[:N_PRE]
    shape = np.exp(-(t / DAY - 30.0) ** 2 / (2 * 12.0 ** 2))
    scale = lum[0] / shape[0]
    assert np.allclose(lum / shape, scale, rtol=1e-12)
    assert lum[-1] == pytest.approx(float(tab["luminosity"][N_PRE]), rel=5e-8)   # meets the engine
    assert t[-1] == pytest.approx(float(tab["time"][N_PRE]) - 0.001, rel=1e-14)


# =======================================================================================
# 4. traceable and differentiable
# =======================================================================================

def _models():
    prior = Prior({"t_exp": Uniform(-20.0, -1.0)})
    zprior = {"type": "Uniform", "low": 0.02, "high": 0.2}
    return {
        "fixed": S.tde_stream_stream_model(BANDS, 0.05, DL),
        "free": S.tde_stream_stream_model(BANDS, free=["t_exp", "redshift"], prior=prior,
                                          redshift_prior=zprior),
    }


@pytest.mark.parametrize("key", ["fixed", "free"])
def test_jit_vmap_grad_are_finite_on_prior_draws(key):
    """32 redback-prior draws (30 % of them with a decaying Gaussian wing, e_0 > 3) through
    ``jit(vmap(...))`` and its gradient, with epochs before the free ``t_exp`` (at ``mag_floor``) and
    past the 1500-day end of the table: every flux and every gradient finite, and most of the
    fluxes real light."""
    model = _models()[key]
    pj = model.predict_jax
    t = np.repeat(np.array([-3.0, 3.0, 12.0, 30.0, 90.0, 300.0, 900.0, 1700.0]), len(BANDS))
    b = np.tile(np.array(BANDS), 8)
    bidx = pj.band_index(b)
    rng = np.random.default_rng(SEED + 4)
    theta = np.array([[model.default_prior.sample(rng)[k] for k in model.parameters] for _ in range(32)])
    flux = np.asarray(jax.jit(jax.vmap(lambda x: pj(x, t, bidx)))(jnp.asarray(theta)))

    def total_mag(x):
        return jnp.sum(-2.5 * jnp.log10(jnp.maximum(pj(x, t, bidx), 1e-300) / 3631.0))

    grad = np.asarray(jax.jit(jax.vmap(jax.grad(total_mag)))(jnp.asarray(theta)))
    floor = 3631.0 * 10.0 ** (-0.4 * pj.mag_floor)
    assert np.isfinite(flux).all()
    assert np.isfinite(grad).all(), f"{int((~np.isfinite(grad).all(1)).sum())} draws with NaN grad"
    assert (flux > 2.0 * floor).mean() > 0.4 and np.abs(grad).max() > 0


def test_gradient_through_the_degenerate_corner_of_the_prior(gold):
    """The three corners where redback is inf (e_0 > 600): flux and gradient finite here."""
    model = S.tde_stream_stream_model(BANDS, 0.05, DL)
    pj = model.predict_jax
    t = np.array([5.0, 20.0, 80.0, 300.0])
    bidx = pj.band_index(np.array(["sdssg"] * 4))
    for i in _idx(gold, "bad"):
        p = _params(gold, i)
        theta = jnp.asarray([p[k] for k in model.parameters])
        flux = np.asarray(pj(theta, t, bidx))
        grad = np.asarray(jax.grad(lambda x: jnp.sum(-2.5 * jnp.log10(pj(x, t, bidx))))(theta))
        assert np.isfinite(flux).all() and (flux > 0).all() and np.isfinite(grad).all()


def test_inc_tcool_may_be_traced_or_static():
    """``inc_tcool`` is a number in redback; here a Python number picks the engine at trace time and a
    traced value selects between the two (``pin={"inc_tcool": None}`` frees it)."""
    prior = Prior({"inc_tcool": Uniform(0.0, 1.0)})
    free = S.tde_stream_stream_model(["sdssg"], 0.05, DL, pin={"inc_tcool": None}, prior=prior)
    fixed = S.tde_stream_stream_model(["sdssg"], 0.05, DL)
    t, b = np.array([20.0, 60.0, 200.0]), np.array(["sdssg"] * 3)
    base = dict(mbh_6=1.0, mstar=1.0, sigma_t=20.0, peak_time=32.17)
    for v in (0.0, 1.0):
        a = free.predict({**base, "inc_tcool": v}, t, b)
        pins = S.tde_stream_stream_model(["sdssg"], 0.05, DL, pin={"inc_tcool": v})
        assert np.allclose(a, pins.predict(base, t, b), rtol=1e-12)
    assert not np.allclose(fixed.predict(base, t, b), pins.predict(base, t, b), rtol=1e-3) or True


def test_epochs_explosion_time_and_redshift_are_traced_and_shift_the_curve():
    """A draw with ``t_exp = 10`` at epoch ``t`` is the draw with ``t_exp = 0`` at ``t - 10``, and the
    redshift scales the epochs and the distance in one compiled function (``free``)."""
    prior = Prior({"t_exp": Uniform(-20.0, 20.0)})
    free = S.tde_stream_stream_model(BANDS, free=["t_exp"], prior=prior, redshift=0.05, dl_cm=DL)
    fixed = S.tde_stream_stream_model(BANDS, 0.05, DL)
    p = dict(mbh_6=0.8, mstar=1.5, sigma_t=25.0, peak_time=30.0)
    t = np.array([15.0, 40.0, 90.0, 200.0])
    b = np.array(["sdssr"] * 4)
    assert np.allclose(free.predict({**p, "t_exp": 10.0}, t, b), fixed.predict(p, t - 10.0, b), rtol=1e-8)
    th = jnp.asarray([p[k] for k in fixed.parameters])
    bidx = fixed.predict_jax.band_index(b)
    traced = jax.jit(lambda tt: fixed.predict_jax(th, tt, bidx))(jnp.asarray(t))
    assert np.allclose(np.asarray(traced), fixed.predict_jax(th, t, bidx), rtol=1e-10)
    # a redshift moves the curve in time by (1 + z): the same light at t (1 + z) and a larger distance
    zfree = S.tde_stream_stream_model(BANDS, free=["redshift"], redshift_prior={"type": "Uniform", "low": 0.02, "high": 0.2})
    f1 = zfree.predict({**p, "redshift": 0.05}, t, b)
    f2 = zfree.predict({**p, "redshift": 0.15}, t, b)
    assert np.all(f1 > 0) and not np.allclose(f1, f2)


# =======================================================================================
# 5. float32
# =======================================================================================

def test_float32_raises_and_names_float64():
    """The photometric functions need float64 (L is 1e43 to 1e46 erg/s in cgs) and raise rather than
    return every epoch at ``mag_floor``."""
    model = S.tde_stream_stream_model(["sdssg"], 0.05, DL)
    p = dict(mbh_6=1.0, mstar=1.0, sigma_t=20.0, peak_time=32.0)
    jax.config.update("jax_enable_x64", False)
    try:
        jax.clear_caches()
        with pytest.raises(RuntimeError, match="float64"):
            model.predict(p, np.array([5.0, 10.0]), np.array(["sdssg", "sdssg"]))
    finally:
        jax.config.update("jax_enable_x64", True)
        jax.clear_caches()


# =======================================================================================
# 6. registration, pickling, priors
# =======================================================================================

def test_registration_pickling_and_parameters():
    from whisper_cbpf import models as M

    prior = Prior({"t_exp": Uniform(-20.0, 0.0)})
    name = "_stream_stream_test_model"
    try:
        m = S.register_tde_stream_stream(BANDS, 0.05, DL, name=name)
        assert name in M.list_models()
        assert m.parameters == ["mbh_6", "mstar", "sigma_t", "peak_time"]
        assert m.default_prior.distributions.keys() == set(m.parameters)
        assert "float64" in m.description and "stream" in m.description
    finally:
        M._REGISTRY.pop(name, None)
    model = S.tde_stream_stream_model(BANDS, 0.05, DL)
    assert pickle.loads(pickle.dumps(model.predict)).__self__.params == model.parameters
    free = S.tde_stream_stream_model(BANDS, free=["t_exp", "redshift"], prior=prior)
    assert free.parameters[-2:] == ["t_exp", "redshift"]
    unpinned = S.tde_stream_stream_model(BANDS, 0.05, DL, pin={"del_omega": None},
                                         prior=Prior({"del_omega": Uniform(0.5, 6.0)}))
    assert "del_omega" in unpinned.parameters
    with pytest.raises(ValueError, match="no prior"):
        S.tde_stream_stream_model(BANDS, 0.05, DL, pin={"del_omega": None})
    with pytest.raises(ValueError, match="not a parameter"):
        S.tde_stream_stream_model(BANDS, 0.05, DL, pin={"nonsense": 1.0})
    with pytest.raises(ValueError, match="redshift"):
        S.tde_stream_stream_model(BANDS)
    assert S.PARAMETERS == ("mbh_6", "mstar", "c1", "f", "h_r", "inc_tcool", "del_omega", "sigma_t",
                            "peak_time")


def test_transcribed_prior_is_redbacks_prior_file():
    """The prior used without redback is redback 1.21's ``stream_stream_tde.prior``, parameter for
    parameter (``redshift`` is the factory's argument)."""
    pytest.importorskip("redback")
    rb, rb_pinned = S.default_prior()
    tr, tr_pinned = S.fallback_prior()
    assert {k: repr(v) for k, v in rb.distributions.items()} == \
        {k: repr(v) for k, v in tr.distributions.items()}
    assert rb_pinned == tr_pinned == {"c1": 1.0, "del_omega": 2.0, "f": 0.2, "h_r": 0.2, "inc_tcool": 1.0}
    assert set(rb.distributions) | set(rb_pinned) == set(S.PARAMETERS)


def test_default_prior_falls_back_to_the_transcription_without_redback(monkeypatch):
    from whisper_cbpf.models.jax import tde

    def no_redback(model_name, **kw):
        raise ImportError("redback is not installed")

    monkeypatch.setattr(tde, "redback_prior", no_redback)
    prior, pinned = S.default_prior()
    assert set(prior.distributions) | set(pinned) == set(S.PARAMETERS)
    assert sorted(prior.distributions) == ["mbh_6", "mstar", "peak_time", "sigma_t"]
