"""MOSFiT's ``tde`` model in JAX (``whisper_cbpf.models.jax.tde_mosfit``).

Six claims, each with the number measured when this file was written (budgets are 3 to 10 times
the measured value):

1. **Parity with the original MOSFiT modules.** ``tests/goldens/tde_mosfit_oracle.npz`` is written
   by running MOSFiT 2.1.1's own ``Fallback``, ``Viscous``, ``TdePhotosphere``, ``TDEConstraints``
   and ``Blackbody`` unmodified, on 88 parameter sets (the canonical point, 64 Latin-hypercube
   draws over MOSFiT's prior box and 23 edge sets) at 64 epochs each, from 0.03 to 30 times the
   peak time of the return rate. Against it, over those 88 x 64 points: luminosity median 4.9e-5
   dex, 99th percentile 2.2e-4, maximum 1.7e-3; temperature 1.1e-5 / 6.0e-5 / 4.2e-4; photosphere
   radius 1.4e-5 / 1.8e-4 / 8.1e-4; stellar radius, ``t_peak``, ``beta`` and the pericentre to
   round-off (3e-15); the ``tde_constraints`` score to a relative 7.8e-4 at worst; monochromatic
   F_nu at the six UVOT/SDSS wavelengths of the comparison notebook (z = 0.05) median 4.1e-5 dex,
   99th percentile 3.9e-4, maximum 3.6e-3 (where F_nu > 1e-6 of the set's maximum; the Wien tail
   amplifies a temperature error). The fall-back engine, shared with ``tde_fallback``, is MOSFiT's
   ``Fallback`` node for node (its luminosity before the viscous delay is MOSFiT's to 1.64e-13,
   relative); the residual is the viscous delay, redback's quadrature on a 1000-node grid against
   MOSFiT's exact solution on the golden's 3000-node grid, largest on the earliest epochs of the
   rise. (With the 512-step resampled tables of 0.3.0 the luminosity differed by 8.8e-5 / 8.2e-3 /
   1.8e-2 dex, and that tail was the engine.)
2. **Known answers** at the canonical point (1e6 Msun, 1 Msun, b = 1, efficiency 0.1,
   Tviscous = 1 d, Rph0 = 6.31, lphoto = 1.5): the first fall-back 6.24 d after disruption, the
   peak 25.1 d after disruption (the top is flat to 1e-4: MOSFiT's grid puts it at 25.2),
   L = 1.43e44 erg/s, T = 1.50e4 K; the Eddington cap.
3. **Physics invariants**: the late decline tends to ``t^-5/3`` (slope -1.70 for b = 2, -1.57 for
   b = 1 within 150 peak times); ``T ~ L^((1 - 2 lphoto) / 4)`` (measured exponent -0.49965 for
   lphoto = 1.5, so T rises after the peak); ``t_peak ~ M_BH^(1/2)`` exactly and independent of the
   efficiency; the photosphere is ``Rph0 a_p (L / Leddlim L_Edd)^lphoto + r_isco`` with nothing else.
4. **The relation to the Redback-style photosphere of ``tde_fallback``** (upstream issue U7):
   same luminosity, the redback photosphere has 0.2500 of MOSFiT's radius and 2.0001 times its
   temperature at the peak.
5. **JAX behaviour**: finite jit/vmap/grad on prior draws, traced epochs, ``t_exp`` and redshift,
   epochs before the first fall-back at ``mag_floor``, the float64 guard, pickling, registration,
   the constraint wall.
6. **Live MOSFiT**, when MOSFiT can run (skipped otherwise): the golden is reproduced by the
   installed MOSFiT, the prior transcription equals ``parameters.json``, the port agrees with
   a fresh run, and the fall-back tables and final nodes are MOSFiT's ``Fallback``'s (the
   ``__init__`` arrays bit for bit, the nodes to 2.0e-13). The test file needs ``jax`` and ``pytest``; MOSFiT lives in its own environment
   in this project, so the live tests run MOSFiT in a subprocess: set ``MOSFIT_PYTHON`` to a Python
   that imports ``mosfit`` (``MOSFIT_PYTHON=/path/to/python pytest tests/test_tde_mosfit.py -k live``),
   or run the whole file in an environment that has jax, pytest and mosfit together.

FLOAT64. The photometric functions require it; the fixture below enables it for this module and
restores the previous setting afterwards.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models import constraints as C  # noqa: E402
from whisper_cbpf.models.jax import tde_fallback as TF  # noqa: E402
from whisper_cbpf.models.jax import tde_mosfit as M  # noqa: E402
from whisper_cbpf.models.jax._adopted_tde import geometric_dense_times  # noqa: E402
from whisper_cbpf.models.jax.tde import tde_photosphere  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "goldens" / "tde_mosfit_oracle.npz"
Z = 0.05
#: Planck18 luminosity distance at z = 0.05 [cm].
DL = 7.093375016731583e26
BANDS = ["lsstg", "lsstr", "lssti", "ztfg", "ztfr"]
SEED = 20261119
ANCHOR = (1.0e6, 1.0, 1.0, 0.1, 1.0, 6.31, 1.5, 1.0)       # M.PARAMETERS order
#: Eddington luminosity of 1e6 solar masses (kappa = 0.2 (1 + 0.74)), erg/s.
LEDD_1E6 = 1.4366896299274836e44


@pytest.fixture(scope="module", autouse=True)
def _x64():
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


@pytest.fixture(scope="module")
def gold():
    with np.load(GOLDEN, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _stats(x):
    x = np.abs(np.asarray(x, dtype=float))
    return float(np.median(x)), float(np.percentile(x, 99)), float(np.max(x))


@pytest.fixture(scope="module")
def port(gold):
    """The port on every golden parameter set: ``jit(vmap(...))``, once for the module."""
    theta = jnp.asarray(gold["theta"])
    tau = jnp.asarray(gold["tau"])
    curve = jax.jit(jax.vmap(lambda t, th: M.tde_mosfit_curve(t, *th)))
    lum, temp, rad, rstar, t_peak, beta, rp = (np.asarray(a) for a in curve(tau, theta))
    nu = jnp.asarray(gold["nu_hz"])
    n_t = tau.shape[1]

    def flux(t, th):
        tt = jnp.repeat(t, 6)
        return M.tde_mosfit_flux_density_mjy(tt, jnp.tile(nu, n_t), Z, DL, *th).reshape(n_t, 6)

    fnu = np.asarray(jax.jit(jax.vmap(flux))(tau, theta))
    score = np.asarray(jax.jit(jax.vmap(lambda t, th: M.tde_mosfit_constraint_score(t, *th)))(
        tau, theta))
    return dict(L=lum, T=temp, R=rad, rstar=rstar, t_peak=t_peak, beta=beta, rp=rp, fnu=fnu,
                score=score)


# =======================================================================================
# 1. parity with the original MOSFiT modules
# =======================================================================================

def test_golden_provenance_and_shape(gold):
    """The golden says what made it: MOSFiT 2.1.1, hashes of the modules it ran, the seed."""
    prov = json.loads(str(gold["provenance"]))
    assert prov["mosfit_version"] == "2.1.1" and prov["seed_lhs"] == SEED
    assert len(prov["sha256"]["mosfit/modules/engines/fallback.py"]) == 64
    assert gold["theta"].shape == (88, 8) and gold["tau"].shape == (88, 64)
    assert list(gold["kind"][:2]) == ["anchor", "lhs"] and (gold["kind"] == "lhs").sum() == 64
    assert prov["parameter_order"] == list(M.PARAMETERS)
    gate = prov["sanity_gate_anchor_notebook_grid"]          # the notebook's numbers, reproduced
    assert abs(gate["first_fallback_d"] - 6.2) < 0.05 and abs(gate["peak_day_since_disruption"] - 25.2) < 0.1
    assert abs(gate["L_peak"] / 1e44 - 1.43) < 0.01 and abs(gate["T_at_peak"] / 1e4 - 1.50) < 0.01


def test_the_oracle_grid_is_converged(gold):
    """The oracle's own discretisation: 3000 nodes against 12000 (max 1.3e-4 dex), and MOSFiT's
    default 100-step ``DenseTimes`` grid against the converged answer (median 5e-5, p99 2.5e-3,
    max 3.9e-3 dex): the golden is not limited by its grid."""
    assert float(gold["max_dlog_selfconv"].max()) < 5e-4
    d = np.abs(np.log10(gold["L_dense100"]) - np.log10(gold["L"]))
    med, p99, mx = _stats(d)
    print(f"\nMOSFiT default dense grid vs converged: median {med:.1e}, p99 {p99:.1e}, max {mx:.1e} dex")
    assert med < 3e-4 and p99 < 1e-2 and mx < 2e-2


def test_luminosity_temperature_radius_match_mosfit(gold, port):
    """88 sets x 64 epochs. Measured (dex): L 4.9e-5 / 2.2e-4 / 1.7e-3 (median / p99 / max),
    T 1.1e-5 / 6.0e-5 / 4.2e-4, R 1.4e-5 / 1.8e-4 / 8.1e-4. The anchor alone: L max 6.3e-5."""
    out = {}
    for key in ("L", "T", "R"):
        d = np.log10(port[key]) - np.log10(gold[key])
        out[key] = _stats(d)
        print(f"{key}: median {out[key][0]:.1e}, p99 {out[key][1]:.1e}, max {out[key][2]:.1e} dex")
    assert out["L"][0] < 2e-4 and out["L"][1] < 1e-3 and out["L"][2] < 6e-3
    assert out["T"][0] < 5e-5 and out["T"][1] < 3e-4 and out["T"][2] < 2e-3
    assert out["R"][0] < 6e-5 and out["R"][1] < 8e-4 and out["R"][2] < 4e-3
    anchor = np.abs(np.log10(port["L"][0]) - np.log10(gold["L"][0]))
    assert anchor.max() < 3e-4


@pytest.mark.parametrize("kind,budget", [("anchor", 3e-4), ("lhs", 6e-3), ("edge", 2e-3)])
def test_parity_holds_in_each_group(gold, port, kind, budget):
    """The anchor, the 64 draws and the 23 edge sets (gamma-blend limits, b = 0 and 2, the Eddington
    limits, the box corners), each with its own budget. Measured L max: 6.3e-5, 1.7e-3, 4.0e-4 dex;
    medians 4.5e-5, 5.4e-5, 4.4e-5."""
    m = gold["kind"] == kind
    d = np.abs(np.log10(port["L"][m]) - np.log10(gold["L"][m]))
    assert d.max() < budget and np.median(d) < 2e-4


def test_engine_scalars_match_mosfit_to_roundoff(gold, port):
    """Stellar radius, time of the return-rate peak, ``beta`` and pericentre: same formulas on the
    same nodes (measured relative 2.4e-15 at worst over the 88 sets)."""
    for mine, key in (("rstar", "Rstar"), ("t_peak", "tpeak"), ("beta", "beta"), ("rp", "rp")):
        rel = np.abs(port[mine] / gold[key] - 1.0)
        assert rel.max() < 1e-12, (mine, rel.max())


def test_constraint_score_matches_mosfit(gold, port):
    """MOSFiT's ``-10 (Rs/rp)^3 - 10 max(R/Rmax)^2`` (``rphotmaxwind = 1``, ``vphotmaxwind = 0.5``,
    epochs = the 64 above). Relative to max(|score|, 1e-2): median 1.2e-6, p99 4.9e-4, max 7.8e-4
    (the squared photosphere ratio doubles the radius error). The scores span -1.8e6 to -9e-6."""
    rel = np.abs(port["score"] - gold["score_wind"]) / np.maximum(np.abs(gold["score_wind"]), 1e-2)
    med, p99, mx = _stats(rel)
    print(f"\nconstraint score: median {med:.1e}, p99 {p99:.1e}, max {mx:.1e} (relative)")
    assert med < 5e-6 and p99 < 2e-3 and mx < 3e-3
    assert port["score"][0] == pytest.approx(gold["score_wind"][0], rel=1e-4)   # anchor -0.7258


def test_constraint_terms_and_the_apocentre_variant(gold, port):
    """``Rs / rp`` is exact; the apocentre variant (``rphotmaxwind = 0``, MOSFiT's old default) is
    the same function with the flag off (measured relative 2.8e-7 on this set)."""
    th = jnp.asarray(gold["theta"][5])
    tau = jnp.asarray(gold["tau"][5])
    _, _, rad, _, _, _, rp = M.tde_mosfit_curve(tau, *th)
    rs_over_rp, ratio = M.tde_mosfit_constraint_terms(tau, rad, th[0], rp, False)
    assert float(rs_over_rp) == pytest.approx(gold["Rs_over_rp"][5], rel=1e-12)
    score = -10.0 * rs_over_rp ** 3 - 10.0 * ratio ** 2
    assert float(score) == pytest.approx(gold["score_apocenter"][5], rel=2e-6)
    assert float(M.tde_mosfit_constraint_score(tau, *th, rphotmaxwind=False)) == pytest.approx(
        float(score), rel=1e-12)


def test_flux_density_at_the_six_wavelengths_matches_mosfit(gold, port):
    """Monochromatic F_nu [mJy] of MOSFiT's pure black body at z = 0.05, UVW2 UVW1 u g r i, in the
    convention of the comparison notebook (``(1+z) pi B_nu(T, nu (1+z)) R^2 / d_L^2``). Where F_nu
    exceeds 1e-6 of its set's maximum (93 % of the points): median 4.1e-5, p99 3.9e-4, max 3.6e-3 dex."""
    ref = gold["fnu_mjy"]
    big = ref > 1e-6 * ref.max(axis=(1, 2), keepdims=True)
    d = (np.log10(port["fnu"]) - np.log10(np.maximum(ref, 1e-300)))[big]
    med, p99, mx = _stats(d)
    print(f"\nF_nu, {d.size} points: median {med:.1e}, p99 {p99:.1e}, max {mx:.1e} dex")
    assert big.mean() > 0.9
    assert med < 2e-4 and p99 < 2e-3 and mx < 1.5e-2


def test_the_golden_flux_convention_is_the_band_path(gold):
    """MOSFiT's ``Blackbody`` module followed by the frequency-row arithmetic of ``Photometry`` is
    ``(1 + z)`` times the notebook's physical convention, which is what its band path gives (issue
    12). The golden stores the notebook convention; the module's own factor was 1.05 on every set."""
    assert np.allclose(gold["fnu_freqpath_over_notebook"], 1.0 + Z, rtol=1e-12)
    assert float(gold["max_rel_notebook_formula"].max()) < 1e-9


def test_kroupa_matches_mosfit(gold):
    """MOSFiT's Kroupa stellar-mass distribution (``prior_icdf``, ``lnprior_pdf``) for the three
    lower limits of the golden: the quantile function by relative 1e-10 at worst and the
    log-density to 1e-12."""
    u = gold["kroupa_u"]
    for tag, lo in (("001", 0.01), ("01", 0.1), ("06", 0.6)):
        mass = M.kroupa_icdf(u, lo, 30.0)
        assert np.allclose(mass, gold[f"kroupa_mass_min{tag}"], rtol=1e-9), tag
        assert np.allclose(M.kroupa_log_pdf(gold[f"kroupa_lnpdf_mass_min{tag}"]),
                           gold[f"kroupa_lnpdf_min{tag}"], rtol=0, atol=1e-12), tag
    # the median star of MOSFiT's prior is 0.12 Msun: most of its draws are below 0.5 Msun
    assert M.kroupa_icdf(0.5) == pytest.approx(0.1195, abs=0.002)


# =======================================================================================
# 2. known answers at the canonical point
# =======================================================================================

def _first_fallback_days(mbh, mstar, b=1.0):
    """The first fall-back of the 4/3 simulation at beta = 1.85 (b = 1, a simulated beta, so its own
    nodes), scaled as the engine scales it."""
    tab = TF.fallback_tables()["4-3"]
    i = int(np.argmin(np.abs(tab["sim_beta"] - (0.6 + 1.25 * b if b < 1 else 1.85 + 2.15 * (b - 1)))))
    rstar = float(TF.stellar_radius(mstar))
    return (10.0 ** float(tab["premaptime"][0][i, 0]) * np.sqrt(mbh / 1e6) / mstar * rstar ** 1.5
            / 86400.0)


def test_anchor_first_fallback_peak_luminosity_and_temperature(gold):
    """MOSFiT's numbers for 1e6 Msun, 1 Msun, b = 1: first fall-back 6.24 d after disruption (the
    port reads it from the same table: 6.2354 d), peak at 25.24 d after disruption (MOSFiT: 25.2;
    the top is flat to 1e-4), L = 1.426e44 erg/s (1.43e44), T = 1.496e4 K (1.50e4)."""
    t_first = _first_fallback_days(1e6, 1.0)
    assert t_first == pytest.approx(6.2354, abs=2e-3) and t_first == pytest.approx(
        gold["tfallback"][0], rel=1e-9)
    tau = jnp.geomspace(1.0, 100.0, 4000)
    lum, temp, *_ = M.tde_mosfit_curve(tau, *ANCHOR)
    j = int(np.argmax(np.asarray(lum)))
    assert abs(float(tau[j]) + t_first - 25.2) < 0.1
    assert float(lum[j]) == pytest.approx(1.43e44, rel=0.01)
    assert float(temp[j]) == pytest.approx(1.50e4, rel=0.005)
    assert float(lum[j]) < LEDD_1E6


@pytest.mark.parametrize("leddlim", [0.1, 1.0])
def test_eddington_cap(leddlim):
    """``L -> L Llim / (L + Llim)``: with efficiency 0.4 the peak sits just under ``Leddlim L_Edd``
    (measured 0.99995 and 0.99825 of it) and never above it, however large the rate."""
    tau = jnp.geomspace(0.5, 3000.0, 600)
    lum = np.asarray(M.tde_mosfit_curve(tau, 1e6, 1.0, 1.0, 0.4, 1.0, 6.31, 1.5, leddlim)[0])
    assert lum.max() < leddlim * LEDD_1E6 and lum.max() > 0.99 * leddlim * LEDD_1E6


def test_the_uncapped_luminosity_scales_with_the_efficiency():
    """Far below the cap ``L`` is linear in the efficiency: the peak L is 1.68e43, 4.09e43, 8.19e43
    for 1e-4, 3e-4, 1e-3 (the cap still takes 11 % at 1e-4, so those are not 3:1), and the ratio of
    the peaks for 1e-5 and 1e-6 is 9.88 (10 less the 1.2 % the cap still takes at 1e-5; budget 2 %); the time of the peak does not move."""
    tau = jnp.geomspace(0.5, 3000.0, 800)
    peaks, times = [], []
    for eff in (1e-5, 1e-6):
        lum = np.asarray(M.tde_mosfit_curve(tau, 1e6, 1.0, 1.0, eff, 1.0, 6.31, 1.5, 1.0)[0])
        peaks.append(lum.max())
        times.append(float(tau[lum.argmax()]))
    assert peaks[0] / peaks[1] == pytest.approx(10.0, rel=2e-2) and times[0] == times[1]


# =======================================================================================
# 3. physics invariants
# =======================================================================================

@pytest.mark.parametrize("b,lo,hi,expect,tol", [(2.0, 5, 60, -1.69, 0.06), (1.0, 20, 150, -1.57, 0.15)])
def test_late_decline_tends_to_t_to_the_minus_five_thirds(b, lo, hi, expect, tol):
    """Far below the cap (efficiency 1e-4) the decline tends to ``t^-5/3`` (-1.667). Measured slopes
    between ``lo`` and ``hi`` peak times: b = 2: -1.69 (-1.71 and -1.70 in the other windows);
    b = 1, a 4/3 polytrope, steeper with time: -1.39 (5 to 20), -1.40 (10 to 60), -1.57 (20 to 150)."""
    tau = jnp.geomspace(1.0, 1.5e4, 800)
    out = M.tde_mosfit_curve(tau, 1e6, 1.0, b, 1e-4, 1.0, 6.31, 1.5, 1.0)
    lum, t_peak = np.asarray(out[0]), float(out[4])
    t = np.asarray(tau)
    sel = (t > lo * t_peak) & (t < hi * t_peak)
    slope = np.polyfit(np.log10(t[sel]), np.log10(lum[sel]), 1)[0]
    assert abs(slope - expect) < tol, slope
    assert abs(slope + 5.0 / 3.0) < 0.3


@pytest.mark.parametrize("lphoto", [0.25, 0.5, 1.0, 1.5, 2.0])
def test_temperature_follows_L_to_the_one_minus_two_l_over_four(lphoto):
    """With ``R ~ L^l`` and ``T = (L / 4 pi sigma R^2)^(1/4)``, ``T ~ L^((1 - 2 l) / 4)`` while
    ``R`` is well above ``r_isco`` (measured slope in ln T against ln L between 2 and 5 peak times:
    0.1251, 0.0001, -0.2498, -0.4996, -0.7495 against 0.125, 0, -0.25, -0.5, -0.75)."""
    tau = jnp.geomspace(1.0, 1.5e4, 800)
    out = M.tde_mosfit_curve(tau, 1e6, 1.0, 1.0, 0.1, 1.0, 6.31, lphoto, 1.0)
    lum, temp, t_peak = np.asarray(out[0]), np.asarray(out[1]), float(out[4])
    t = np.asarray(tau)
    sel = (t > 2 * t_peak) & (t < 5 * t_peak)
    slope = np.polyfit(np.log(lum[sel]), np.log(temp[sel]), 1)[0]
    assert slope == pytest.approx((1.0 - 2.0 * lphoto) / 4.0, abs=2e-3)


def test_temperature_rises_after_the_peak_for_lphoto_above_one_half():
    """For ``lphoto = 1.5`` the photosphere shrinks faster than ``L`` falls, so ``T`` rises after the
    peak (measured T(3 t_peak) / T(peak) = 1.0129); for ``lphoto = 0.25`` it falls (0.9968)."""
    tau = jnp.geomspace(1.0, 1.5e4, 800)
    ratio = {}
    for lphoto in (0.25, 1.5):
        out = M.tde_mosfit_curve(tau, 1e6, 1.0, 1.0, 0.1, 1.0, 6.31, lphoto, 1.0)
        lum, temp, t_peak = np.asarray(out[0]), np.asarray(out[1]), float(out[4])
        ratio[lphoto] = np.interp(3 * t_peak, np.asarray(tau), temp) / temp[lum.argmax()]
    assert ratio[1.5] > 1.005 and ratio[0.25] < 0.999


def test_peak_time_scales_as_the_square_root_of_the_black_hole_mass():
    """``t_peak ~ M_BH^(1/2)`` (measured 18.0785 d at 1e6 Msun: 5.717, 18.078, 57.17, 180.79 d from
    1e5 to 1e8, to 1e-15), whatever the efficiency, and the first fall-back scales as
    ``M_BH^(1/2) M_*^-1 R_*^(3/2)``."""
    t_peak = []
    for mbh in (1e5, 1e6, 1e7, 1e8):
        for eff in (1e-4, 0.4):
            t_peak.append(float(M.tde_mosfit_curve(jnp.ones(1), mbh, 1.0, 1.0, eff, 1.0, 6.31,
                                                   1.5, 1.0)[4]))
    t_peak = np.array(t_peak).reshape(4, 2)
    assert np.all(t_peak[:, 0] == t_peak[:, 1])
    assert np.allclose(t_peak[:, 0] / t_peak[1, 0], np.sqrt([0.1, 1.0, 10.0, 100.0]), rtol=1e-12)
    assert _first_fallback_days(4e6, 1.0) / _first_fallback_days(1e6, 1.0) == pytest.approx(2.0, rel=1e-12)
    ratio = _first_fallback_days(1e6, 4.0) / _first_fallback_days(1e6, 1.0)
    rstar = float(TF.stellar_radius(4.0) / TF.stellar_radius(1.0))
    assert ratio == pytest.approx(rstar ** 1.5 / 4.0, rel=1e-12)


def test_light_curve_peak_does_not_move_with_the_efficiency():
    """The top of ``L`` is flat to 1e-4, so its position is only defined to about 0.05 d: 18.959,
    18.971, 18.995, 19.018 d for efficiencies 1e-4, 1e-3, 0.1, 0.4 (a spread of 0.06 d, 0.3 % of the peak
    time, for a factor 4000 in efficiency; budget 0.2 d)."""
    tau = jnp.geomspace(5.0, 60.0, 4000)
    peak = {}
    for eff in (1e-4, 1e-3, 0.1, 0.4):
        lum = np.asarray(M.tde_mosfit_curve(tau, 1e6, 1.0, 1.0, eff, 1.0, 6.31, 1.5, 1.0)[0])
        peak[eff] = float(tau[lum.argmax()])
    assert max(peak.values()) - min(peak.values()) < 0.2


def test_the_photosphere_is_a_power_law_plus_the_isco_and_nothing_else():
    """``R = Rph0 a_p (L / (Leddlim L_Edd))^lphoto + r_isco`` with ``a_p`` from the peak time of the
    return rate: recomputed here from the outputs and astropy-free constants, equal to 1e-12 for
    ``Leddlim = 3``, a ``Leddlim`` where redback's normalisation (by ``L_Edd``) would differ by
    ``3^1.5``, and no soft maximum anywhere (R exceeds the apocentre-like ``r_p + 2 a_t`` at 300 d)."""
    G, MSUN_, C_ = 6.6743e-8, 1.988409870698051e33, 2.99792458e10
    mbh, leddlim = 3e6, 3.0
    tau = jnp.geomspace(2.0, 300.0, 50)
    lum, temp, rad, _, t_peak, beta, rp = (np.asarray(a) for a in M.tde_mosfit_curve(
        tau, mbh, 1.0, 1.0, 0.1, 1.0, 6.31, 1.5, leddlim))
    a_p = (G * mbh * MSUN_ * (float(t_peak) * 86400.0 / np.pi) ** 2) ** (1.0 / 3.0)
    ledd = 4 * np.pi * G * mbh * MSUN_ * C_ / (0.2 * 1.74)
    r_isco = 6 * G * mbh * MSUN_ / C_ ** 2
    want = 6.31 * a_p * (lum / (leddlim * ledd)) ** 1.5 + r_isco
    assert np.allclose(rad, want, rtol=1e-12)
    assert np.allclose(temp, (lum / (4 * np.pi * 5.6703744191844314e-5 * rad ** 2)) ** 0.25, rtol=1e-12)
    a_t = (G * mbh * MSUN_ * (np.asarray(tau) * 86400.0 / np.pi) ** 2) ** (1.0 / 3.0)
    assert (rad > float(rp) + 2 * a_t + r_isco).any()      # a soft maximum could not exceed it


def test_b_outside_the_table_holds_the_edge():
    """MOSFiT raises (a fresh engine) or reuses the previous call's ``beta`` (issue 10); here ``b``
    is held in [0, 2], rate, ``beta`` and ``rp`` consistently: the curve at b = 2.5 is the curve at
    b = 2, and b = -0.3 the curve at b = 0, bit for bit; the gradient in ``b`` is finite (zero)."""
    tau = jnp.geomspace(1.0, 200.0, 20)
    f = lambda b: np.concatenate([np.asarray(a).ravel() for a in M.tde_mosfit_curve(
        tau, 1e6, 1.0, b, 0.1, 1.0, 6.31, 1.5, 1.0)])
    assert np.array_equal(f(2.5), f(2.0)) and np.array_equal(f(-0.3), f(0.0))
    g = jax.grad(lambda b: jnp.sum(jnp.log10(M.tde_mosfit_curve(
        tau, 1e6, 1.0, b, 0.1, 1.0, 6.31, 1.5, 1.0)[0])))(2.5)
    assert float(g) == 0.0


# =======================================================================================
# 4. the relation to the Redback-style photosphere (U7)
# =======================================================================================

def test_redback_style_photosphere_is_a_quarter_the_radius_and_twice_the_temperature():
    """Same luminosity (identical engine), different photosphere: at the canonical peak the
    ``tde_fallback`` photosphere (normalised by ``L_Edd``, soft maximum ``r_p + 2 a_t``) is 0.24998 of
    MOSFiT's radius and 2.00008 times its temperature (measured), and its radius is smaller at every
    epoch from 1 to 400 d (ratios 0.17 at 3 d, 0.26 at 20 d, 0.53 at 100 d, 0.78 at 300 d)."""
    tau = jnp.geomspace(1.0, 400.0, 400)
    lum, temp, rad, rstar, t_peak, beta, _ = M.tde_mosfit_curve(tau, *ANCHOR)
    dense = geometric_dense_times(jnp.max(tau), 1000)
    log_l, rstar2, t_peak2, beta2 = TF.fallback_engine(dense, 1.0, 1.0, 1.0, 0.1, 1.0)
    lum2 = 10.0 ** TF.viscous_log10(tau, dense, log_l, 1.0)
    assert float(jnp.max(jnp.abs(lum2 / lum - 1.0))) < 1e-12
    temp2, rad2, _ = tde_photosphere(tau, lum2, 1e6, 1.0, rstar2, t_peak2, beta2, 6.31, 1.5)
    j = int(np.argmax(np.asarray(lum)))
    assert float(rad2[j] / rad[j]) == pytest.approx(0.25, abs=0.01)
    assert float(temp2[j] / temp[j]) == pytest.approx(2.0, abs=0.05)
    assert bool(jnp.all(rad2 < rad)) and bool(jnp.all(temp2 > temp))


# =======================================================================================
# 5. JAX behaviour
# =======================================================================================

def _model(**kw):
    return M.tde_mosfit_model(BANDS, Z, DL, **kw)


def _draws(model, n, seed=SEED):
    rng = np.random.default_rng(seed)
    return [model.default_prior.sample(rng) for _ in range(n)]


def test_parameters_prior_and_pinned_value():
    """MOSFiT's order and units, the box of ``parameters.json``, ``Leddlim`` pinned at 1."""
    m = _model()
    assert m.parameters == ["bhmass", "starmass", "b", "efficiency", "Tviscous", "Rph0", "lphoto"]
    assert M.PARAMETERS == ("bhmass", "starmass", "b", "efficiency", "Tviscous", "Rph0", "lphoto",
                            "Leddlim")
    _, pinned = M.default_prior()
    assert pinned == {"Leddlim": 1.0}
    d = m.default_prior.distributions
    assert (d["bhmass"].low, d["bhmass"].high) == (5e4, 5e8)
    assert (d["starmass"].low, d["starmass"].high) == (0.01, 30.0)
    assert (d["b"].low, d["b"].high) == (0.0, 2.0) and (d["lphoto"].high, d["Rph0"].low) == (2.0, 1e-4)
    assert "float64" in m.description and "MOSFiT" in m.description
    assert _model(pin={"b": 1.0}).parameters == m.parameters[:2] + m.parameters[3:]
    with pytest.raises(ValueError, match="not a parameter"):
        _model(pin={"nonsense": 1.0})
    with pytest.raises(ValueError, match="no prior"):
        _model(pin={"Leddlim": None})                   # unpinning needs a range


@pytest.mark.parametrize("free", [None, ["t_exp", "redshift"]])
def test_jit_vmap_grad_are_finite_on_prior_draws(free):
    """32 prior draws (stars from 0.01 to 30 Msun, black holes 5e4 to 5e8) through ``jit(vmap(...))``
    and the gradient of the summed magnitudes, with an epoch before the free ``t_exp`` of some
    draws (at ``mag_floor``): every flux and every gradient finite, no dead model."""
    if free:
        model = M.tde_mosfit_model(BANDS, free=free, prior=Prior({"t_exp": Uniform(-20.0, -1.0)}),
                                   redshift_prior={"type": "Uniform", "low": 0.02, "high": 0.2})
    else:
        model = _model()
    t = np.repeat(np.array([-3.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0]), len(BANDS))
    b = np.tile(np.array(BANDS), 8)
    pj = model.predict_jax
    bidx = pj.band_index(b)
    theta = np.array([[p[k] for k in model.parameters] for p in _draws(model, 32, SEED + 4)])
    flux = np.asarray(jax.jit(jax.vmap(lambda x: pj(x, t, bidx)))(jnp.asarray(theta)))

    def total_mag(x):
        return jnp.sum(-2.5 * jnp.log10(jnp.maximum(pj(x, t, bidx), 1e-300) / 3631.0))

    grad = np.asarray(jax.jit(jax.vmap(jax.grad(total_mag)))(jnp.asarray(theta)))
    floor = 3631.0 * 10.0 ** (-0.4 * pj.mag_floor)
    assert np.isfinite(flux).all()
    assert np.isfinite(grad).all(), f"{int((~np.isfinite(grad).all(1)).sum())} draws with NaN grad"
    assert (flux > 2.0 * floor).mean() > 0.4 and np.abs(grad).max() > 0
    t_exp = theta[:, model.parameters.index("t_exp")] if free else 0.0
    dead = np.broadcast_to(t_exp >= -3.0, (theta.shape[0],))
    assert np.allclose(flux[dead][:, :len(BANDS)], floor)


def test_epochs_before_the_first_fallback_are_at_mag_floor():
    model = _model()
    p = _draws(model, 1)[0]
    flux = model.predict(p, np.array([-5.0, 0.0, 30.0]), np.array(["lsstg"] * 3))
    floor = 3631.0 * 10.0 ** (-0.4 * 40.0)
    assert flux[0] == pytest.approx(floor) and flux[1] == pytest.approx(floor) and flux[2] > floor
    lum, temp, rad, *_ = M.tde_mosfit_curve(jnp.array([-5.0, 0.0, 30.0]), *ANCHOR)
    assert float(lum[0]) == 1e-30 and float(lum[1]) == 1e-30 and float(lum[2]) > 1e40
    assert bool(jnp.all(jnp.isfinite(temp))) and bool(jnp.all(rad > 0))


def test_explosion_time_is_traced_and_shifts_the_curve():
    """A draw with ``t_exp = 10`` at epoch ``t`` is the draw with ``t_exp = 0`` at ``t - 10``."""
    free = M.tde_mosfit_model(BANDS, Z, DL, free=["t_exp"], prior=Prior({"t_exp": Uniform(-20.0, 20.0)}))
    fixed = _model()
    p = _draws(fixed, 1)[0]
    t = np.array([15.0, 40.0, 90.0, 200.0])
    b = np.array(["ztfr"] * 4)
    assert np.allclose(free.predict({**p, "t_exp": 10.0}, t, b), fixed.predict(p, t - 10.0, b),
                       rtol=1e-8)


def test_epochs_may_be_traced():
    model = _model()
    p = _draws(model, 1, SEED + 6)[0]
    theta = jnp.asarray([p[k] for k in model.parameters])
    t = np.array([4.0, 9.0, 33.0, 120.0])
    bidx = model.predict_jax.band_index(np.array(["lsstg", "lsstr", "lssti", "ztfr"]))
    traced = jax.jit(lambda tt: model.predict_jax(theta, tt, bidx))(jnp.asarray(t))
    assert np.allclose(np.asarray(traced), model.predict_jax(theta, t, bidx), rtol=1e-10)


def test_float64_is_required_and_the_message_says_so():
    model = _model()
    p = _draws(model, 1)[0]
    jax.config.update("jax_enable_x64", False)
    try:
        jax.clear_caches()
        with pytest.raises(RuntimeError, match="float64"):
            model.predict(p, np.array([5.0, 10.0]), np.array(["lsstg", "lsstg"]))
        with pytest.raises(RuntimeError, match="float64"):
            M.tde_mosfit_curve(jnp.array([5.0, 10.0]), *ANCHOR)
    finally:
        jax.config.update("jax_enable_x64", True)
        jax.clear_caches()


def test_pickling_and_registration():
    from whisper_cbpf import models as Mod

    model = _model()
    assert pickle.loads(pickle.dumps(model.predict)).__self__.params == model.parameters
    name = "_test_tde_mosfit_unique"
    try:
        m = M.register_tde_mosfit(BANDS, Z, DL, name=name)
        assert name in Mod.list_models() and m.name == name
        assert Mod.get_model(name).parameters == m.parameters
    finally:
        Mod._REGISTRY.pop(name, None)


@pytest.fixture
def wall():
    """The wall's entry in the constraints table is removed again after the test."""
    had = "tde_mosfit" in C.MODELS
    yield
    if not had:
        C.MODELS.pop("tde_mosfit", None)


def test_constraint_wall_rejects_what_the_score_penalises(gold, wall):
    """``constraint="corrected"`` (``"redback"`` is the same here) arms the wall of
    ``tde_mosfit_constraint``: ``Rs / rp < 1`` and ``max R / R_max < 1``. On the 88 golden sets
    its accept / reject agrees with the golden's own terms on all 88 (45 % of the 64 draws are
    rejected: 23 % swallow the star, 27 % have too large a photosphere); ``rs_over_rp`` agrees to
    1e-15 and the photosphere term to a relative 3.9e-4 at worst (median 1.2e-5)."""
    model = _model(constraint="corrected")
    assert "tde_mosfit" in C.MODELS and "hard wall" in model.description
    assert _model().predict_jax.constraint_ok is None            # default: no wall
    ok_fn = model.predict_jax.constraint_ok
    assert ok_fn is not None
    ok_port, rs, ratio = [], [], []
    for th in gold["theta"]:
        ok_port.append(bool(ok_fn(jnp.asarray(th[:7]))))
        w = M.tde_mosfit_constraint(dict(zip(M.PARAMETERS, th)), np, 0.0)
        rs.append(float(w["rs_over_rp"]))
        ratio.append(float(w["rphot_over_rmax"]))
    ok_gold = (gold["Rs_over_rp"] < 1) & (gold["ratio_max_wind"] < 1)
    assert np.array_equal(np.array(ok_port), ok_gold)
    assert 0.3 < (~ok_gold[gold["kind"] == "lhs"]).mean() < 0.6
    assert np.allclose(rs, gold["Rs_over_rp"], rtol=1e-12)
    rel = np.abs(np.array(ratio) / gold["ratio_max_wind"] - 1.0)
    assert np.median(rel) < 5e-5 and rel.max() < 2e-3
    # a walled draw predicts zero flux on the host, a free one does not
    bad = dict(zip(model.parameters, gold["theta"][np.argmin(ok_gold)][:7]))
    good = dict(zip(model.parameters, gold["theta"][0][:7]))
    t, b = np.array([20.0, 40.0]), np.array(["lsstg", "lsstr"])
    assert (model.predict(bad, t, b) == 0).all() and (model.predict(good, t, b) > 0).all()
    assert model.predict_jax.constraint_ok is not None
    with pytest.raises(ValueError, match="constraint"):
        _model(constraint="strict")


# =======================================================================================
# 6. live MOSFiT
# =======================================================================================

_LIVE_SCRIPT = textwrap.dedent("""
    import json, sys
    import numpy as np
    import mosfit
    from pathlib import Path
    from mosfit.modules.engines.fallback import Fallback
    from mosfit.modules.transforms.viscous import Viscous
    from mosfit.modules.photospheres.tde_photosphere import TdePhotosphere
    from mosfit.modules.constraints.tde_constraints import TDEConstraints

    class _Host:
        def pool(self): return None
        def printer(self): return None

    names = ["bhmass", "starmass", "b", "efficiency", "Tviscous", "Rph0", "lphoto", "Leddlim"]
    sets = np.array(json.loads(sys.argv[1]))
    taus = np.array(json.loads(sys.argv[2]))
    fb, vi = Fallback(name="fallback", model=_Host()), Viscous(name="viscous", model=_Host())
    ph, co = TdePhotosphere(name="tde_photosphere", model=_Host()), TDEConstraints(name="tde_constraints", model=_Host())
    out = []
    for th, tau in zip(sets, taus):
        p = dict(zip(names, th))
        te = np.unique(np.concatenate((np.geomspace(1e-4, tau[-1] + 10.0, 3000), tau)))
        f = fb.process(dense_times=te, rest_times=te, bhmass=p["bhmass"], starmass=p["starmass"],
                       b=p["b"], efficiency=p["efficiency"], Leddlim=p["Leddlim"], resttexplosion=0.0)
        lum = vi.process(rest_times=te, resttexplosion=0.0, dense_times=te,
                         dense_luminosities=f["luminosities"], Tviscous=p["Tviscous"])["luminosities"]
        o = ph.process(rest_times=te, bhmass=p["bhmass"], starmass=p["starmass"], lphoto=p["lphoto"],
                       Rph0=p["Rph0"], luminosities=lum, resttexplosion=0.0, beta=f["beta"],
                       Rstar=f["Rstar"], tpeak=f["tpeak"], Ledd=f["Ledd"], Leddlim=p["Leddlim"])
        i = np.searchsorted(te, tau)
        sc = co.process(rp=o["rp"], bhmass=p["bhmass"], rphotmaxwind=1, vphotmaxwind=0.5,
                        rest_times=tau, resttexplosion=0.0, radiusphot=o["radiusphot"][i],
                        rphotmin=o["rphotmin"])["score_modifier"]
        out.append(dict(L=lum[i].tolist(), T=o["temperaturephot"][i].tolist(),
                        R=o["radiusphot"][i].tolist(), score=float(sc), tpeak=float(f["tpeak"]),
                        tfallback=float(f["tfallback"])))
    pj = json.loads((Path(mosfit.__file__).parent / "models" / "tde" / "parameters.json").read_text())
    print("RESULT" + json.dumps(dict(sets=out, parameters=pj, version=mosfit.__version__)))
""")


def _mosfit_python():
    """The interpreter that runs MOSFiT: this one if it can import ``mosfit``, else ``$MOSFIT_PYTHON``."""
    if importlib.util.find_spec("mosfit") is not None:
        return sys.executable
    candidate = os.environ.get("MOSFIT_PYTHON")
    if candidate:
        probe = subprocess.run([candidate, "-c", "import mosfit"], capture_output=True)
        if probe.returncode == 0:
            return candidate
    pytest.skip("MOSFiT is not importable here and MOSFIT_PYTHON does not point to a Python with it")


@pytest.fixture(scope="module")
def live(gold):
    """A fresh MOSFiT run on six golden sets (the anchor, three draws, two edges)."""
    python = _mosfit_python()
    idx = [0, 1, 17, 40, 65 + 3, 65 + 14]
    run = subprocess.run(
        [python, "-c", _LIVE_SCRIPT, json.dumps(gold["theta"][idx].tolist()),
         json.dumps(gold["tau"][idx].tolist())], capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.split("RESULT", 1)[1])
    out["idx"] = idx
    return out


def test_live_mosfit_reproduces_the_golden(gold, live):
    """The installed MOSFiT gives the golden's numbers (to float round-off when it is the version
    that wrote the golden: the module hashes in the provenance are compared and reported)."""
    prov = json.loads(str(gold["provenance"]))
    if live["version"] != prov["mosfit_version"]:
        pytest.skip(f"golden written by MOSFiT {prov['mosfit_version']}, installed {live['version']}")
    for i, s in zip(live["idx"], live["sets"]):
        assert np.allclose(s["L"], gold["L"][i], rtol=1e-9), i
        assert np.allclose(s["T"], gold["T"][i], rtol=1e-9) and np.allclose(s["R"], gold["R"][i], rtol=1e-9)
        assert s["score"] == pytest.approx(gold["score_wind"][i], rel=1e-9)


def test_live_mosfit_agrees_with_the_port(gold, live):
    """The port against a fresh MOSFiT run: the same budgets as the golden's maxima."""
    for i, s in zip(live["idx"], live["sets"]):
        out = M.tde_mosfit_curve(jnp.asarray(gold["tau"][i]), *gold["theta"][i])
        for key, got, budget in (("L", out[0], 6e-3), ("T", out[1], 2e-3), ("R", out[2], 4e-3)):
            d = np.log10(np.asarray(got)) - np.log10(np.asarray(s[key]))
            assert np.abs(d).max() < budget, (i, key)
        assert float(out[4]) == pytest.approx(s["tpeak"], rel=1e-12)


def test_live_prior_equals_parameters_json(live):
    """The transcribed prior is MOSFiT's ``parameters.json``: bounds and which are logarithmic
    (the stellar mass is Kroupa-distributed in ``tde.json`` and log-uniform here)."""
    pj = live["parameters"]
    prior, pinned = M.mosfit_prior()
    from whisper_cbpf.priors import LogUniform

    for name, dist in prior.distributions.items():
        box = pj[name]
        assert (dist.low, dist.high) == (box["min_value"], box["max_value"]), name
        assert isinstance(dist, LogUniform) == bool(box.get("log", False)) or name == "starmass"
    assert pinned == {"Leddlim": 1.0}


_LIVE_NODES_SCRIPT = textwrap.dedent("""
    import json, sys
    import numpy as np
    import mosfit.modules.engines.fallback as FB

    class _Host:
        def pool(self): return None
        def printer(self): return None

    captured = {}
    original = FB._lin_interp

    def spy(x, xp, fp):                 # the last call in process: _lin_interp(mid, time, dmdt)
        captured["t"], captured["d"] = np.array(xp, dtype=float), np.array(fp, dtype=float)
        return original(x, xp, fp)

    FB._lin_interp = spy
    fb = FB.Fallback(name="fallback", model=_Host())
    out = {}
    for g in ("4-3", "5-3"):
        k = g.replace("-", "_")
        out[f"sim_beta_{k}"] = np.array(fb._sim_beta[g])
        for name in ("premaptime", "premapdmdt", "mapped_time", "beta_slope", "beta_yinter"):
            for i, row in enumerate(getattr(fb, "_" + name)[g]):
                for j in (0, 1):
                    out[f"{name}_{k}_{i}_{j}"] = np.array(row[j], dtype=float)
    sets = json.loads(sys.argv[2])
    te = np.geomspace(1e-3, 1e3, 50)
    for n, (bhmass, starmass, b) in enumerate(sets):
        f = fb.process(dense_times=te, rest_times=te, bhmass=bhmass, starmass=starmass, b=b,
                       efficiency=0.1, Leddlim=1.0, resttexplosion=0.0)
        out[f"node_t_{n}"], out[f"node_d_{n}"] = captured["t"], captured["d"]
        out[f"scalars_{n}"] = np.array([f["Rstar"], f["tpeak"], f["beta"]])
    np.savez(sys.argv[1], **out)
    print("RESULT")
""")


def test_live_mosfit_fallback_nodes_are_the_ports(gold, tmp_path):
    """MOSFiT's own ``Fallback``: its ``__init__`` arrays are ``build_fallback_tables``'s bit for bit,
    and on the 88 golden sets plus a star at an exactly simulated 5/3 beta (MOSFiT's equality
    branch) and a 4/3-5/3 blend there, its final ``(time, dM/dt)`` nodes are the port's
    (``_fallback_nodes``): measured 2.0e-13 in time and 3.8e-14 in ``dM/dt`` (relative) at worst
    over 634 sets; ``R_*``, ``t_peak`` and ``beta`` to 3e-15."""
    python = _mosfit_python()
    sets = [(float(th[0]), float(th[1]), float(th[2])) for th in gold["theta"]]
    b_085 = (0.85 - 0.5) / 0.4                     # 0.5 + 0.4 b == 0.85 exactly for this float
    assert 0.5 + 0.4 * b_085 == 0.85
    sets += [(1e6, 0.2, b_085), (1e6, 0.6, b_085), (1e6, 18.0, 1.0)]
    path = tmp_path / "mosfit_nodes.npz"
    run = subprocess.run([python, "-c", _LIVE_NODES_SCRIPT, str(path), json.dumps(sets)],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-2000:]
    with np.load(path) as theirs:
        tables = TF.fallback_tables()
        for gamma in ("4-3", "5-3"):
            k = gamma.replace("-", "_")
            tab = tables[gamma]
            assert np.array_equal(tab["sim_beta"], theirs[f"sim_beta_{k}"])
            for name in ("premaptime", "premapdmdt", "mapped_time", "beta_slope", "beta_yinter"):
                count = tab["premap_count" if name.startswith("premap") else "mapped_count"]
                for i in range(count.shape[0]):
                    for j in (0, 1):
                        assert np.array_equal(tab[name][j][i, :count[i, j]],
                                              theirs[f"{name}_{k}_{i}_{j}"]), (gamma, name, i, j)
        nodes = jax.jit(lambda p: TF._fallback_nodes(p[0] / 1e6, p[1], p[2]))
        for n, p in enumerate(sets):
            t, log_d, valid, rstar, t_peak, beta = (np.asarray(a) for a in nodes(jnp.asarray(p)))
            mt, md = theirs[f"node_t_{n}"], theirs[f"node_d_{n}"]
            assert int(valid.sum()) == mt.size, n
            t, d = t[valid], 10.0 ** log_d[valid]
            assert t[0] == 0.0 and mt[0] == 0.0
            assert np.max(np.abs(t[1:] / mt[1:] - 1.0)) < 2e-12, n
            assert np.max(np.abs(d / md - 1.0)) < 2e-12, n
            assert np.allclose([rstar, t_peak, beta], theirs[f"scalars_{n}"], rtol=1e-13, atol=0)
