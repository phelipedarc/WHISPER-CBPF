"""T0 numerical parity: the JAX ports against goldens generated from redback 1.15.1.

The goldens (``tests/goldens/*.npz``) were generated ONCE, from redback only, inside the
reference container (see ``make_goldens.py`` and each file's ``provenance`` field). These
tests need no redback at all: the point of a golden is that the reference is frozen.

REGIME-AWARE BY CONSTRUCTION. Parity is asserted where the port claims parity, and each
documented deviation is asserted AS a deviation with its measured bound:

  TDE engine        parity on the first 80% of both curves (the Euler recursion's own
                    ULP-amplification tail is documented in the module and in
                    test_tde_vs_redback); termination within the documented +-4 steps.
  TDE flux density  parity at epochs clear of the last ZONE_STEPS = 24 grid steps of the
                    shorter curve (the module's VALIDITY block measures the two-
                    implementation drift extending 23 steps before termination at
                    n_time=500); the zone itself is bounded as the documented deviation,
                    including the port-zero-past-its-own-termination cases (CHANGE 3).
  TDE band mags     DEVIATION bound, not parity: redback splines a 100-point SED through
                    an sncosmo TimeSeriesSource, the port integrates the exact band.
  Kilonova          parity for t < 1.4*t_diff (the port's <0.3% claim); past 2.66*t_diff
                    the divergence is asserted IN THE DOCUMENTED DIRECTION (redback's
                    under-resolved trapezoid is too bright -- kilonova.py identity 6);
                    past ~26.6*t_diff redback is NaN (exp overflow, identity 2) and the
                    port must be finite.

TOLERANCES, derived from the protocol's dchi2 template. A model error dF biases the fit
statistic by dchi2 = sum_i (dF_i / sigma_i)^2 = sum_i (rel_i * SNR_i)^2. For a harsh
reference dataset -- N = 100 epochs at SNR = 20 each -- a uniform relative error eps gives
dchi2 = N * (eps * SNR)^2 = 4e4 * eps^2, so eps < 5e-4 keeps dchi2 < 0.01, invisible
against sampling noise. Hence:
  * interior flux parity gate p99.9 < 1e-6 (dchi2 < 4e-8; measured max is 1.2e-12);
  * the termination-zone deviation bound 5e-3 on <= a handful of epochs contributes at
    worst ~3*(5e-3*20)^2 = 0.03, at epochs the module itself calls ill-conditioned;
  * for magnitudes, sigma_m = 1.0857/SNR = 54 mmag at SNR 20: the 25 mmag deviation bound
    is 0.21 chi^2 per affected point -- a DEVIATION to model, not noise to ignore, which
    is why it is labelled a deviation bound and not parity.

Measured-headroom gates (tighter than the protocol gates) are asserted alongside them so a
real regression trips long before it matters to a fit.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
# BEFORE any jax array exists: the TDE engine requires float64 (tde.py CHANGE 8), and the
# kilonova quadrature nodes are baked at import time in the then-active precision. This
# module sorts before tests/test_*.py, so a full-suite run imports everything in f64.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from whisper_cbpf.models.jax import kilonova as kn  # noqa: E402
from whisper_cbpf.models.jax import tde as T  # noqa: E402

DAY = 86400.0
GOLDENS = Path(__file__).resolve().parent.parent / "goldens"

#: The flux-density termination zone, in grid steps of the SHORTER curve. The module's
#: VALIDITY block measures the drift between two implementations of the same Euler
#: recursion becoming visible up to 23 steps before termination at n_time = 500; inside
#: that zone agreement is a claim about round-off amplification, not about the port.
ZONE_STEPS = 24

#: Documented termination-index agreement: "within the +-4 steps that redback agrees with
#: itself" (tde.py CHANGE 3; redback-vs-itself under a one-ulp nudge moves up to 4 steps).
DK_MAX = 4


def _need(name):
    p = GOLDENS / name
    if not p.exists():
        pytest.fail(f"golden file missing: {p} -- run tests/t0_parity/make_goldens.py "
                    f"inside the reference container (see its docstring)")
    return np.load(p, allow_pickle=False)


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(n=int(x.size), median=float(np.median(x)),
                p999=float(np.percentile(x, 99.9)), max=float(x.max()),
                argmax=int(np.argmax(x)))


# ======================================================================= provenance
def test_goldens_provenance_pins_the_reference():
    """The goldens must say exactly which redback produced them -- dist-info version plus
    the SHA256 of the two model source files (``redback.__version__`` is 'unknown' in the
    reference container, so the hashes are the binding record)."""
    for name in ("tde_cooling_envelope.npz", "tde_cooling_envelope_magnitude.npz",
                 "kilonova_one_component.npz"):
        prov = json.loads(str(_need(name)["provenance"]))
        assert prov["redback_dist_version"] == "1.15.1", (name, prov)
        assert len(prov["redback_tde_models_sha256"]) == 64
        assert len(prov["redback_kilonova_models_sha256"]) == 64
        assert prov["dtype"] == "float64"
        assert prov["numpy"] and prov["scipy"] and prov["generated_utc"]
    fs_prov = json.loads(str(_need("filterset_t0.npz")["provenance"]))
    assert "NOT a redback reference output" in fs_prov["model"]


# ======================================================================= TDE fixtures
@pytest.fixture(scope="module")
def tde_gold():
    return _need("tde_cooling_envelope.npz")


@pytest.fixture(scope="module")
def tde_port(tde_gold):
    """The port evaluated at every golden point: engine curves + flux densities."""
    g = tde_gold
    P = g["params"]
    z, dl = float(g["redshift"]), float(g["dl_cm"])
    n = P.shape[0]
    freq_j = jnp.asarray(g["frequency_hz"])

    @jax.jit
    def fd_fn(t, p):
        return T.cooling_envelope_flux_density(
            t, freq_j, z, dl, p[0], p[1], p[2], p[3], p[4],
            **T.REDBACK_PRESETS["1.15"])

    out = dict(k=np.zeros(n, dtype=int), L=np.zeros((n, 500)), T=np.zeros((n, 500)),
               Rph=np.zeros((n, 500)), t_fb=np.zeros((n, 500)),
               fd=np.zeros_like(g["flux_density_mjy"]))
    for i in range(n):
        o = T.cooling_envelope_jit(*[float(v) for v in P[i]], n_time=500)
        out["k"][i] = int(o["constraint"])
        out["L"][i] = np.asarray(o["bolometric_luminosity"])
        out["T"][i] = np.asarray(o["photosphere_temperature"])
        out["Rph"][i] = np.asarray(o["photosphere_radius"])
        out["t_fb"][i] = np.asarray(o["time_since_fb"])
        out["fd"][i] = np.asarray(fd_fn(jnp.asarray(g["t_obs_days"][i]),
                                        jnp.asarray(P[i])))
    return out


# ======================================================================= TDE engine
def test_tde_engine_termination_is_within_documented_steps(tde_gold, tde_port):
    """|constraint_port - constraint_redback| <= 4 on every draw -- the documented bound.

    Note the k_rb == 4 draws are redback's own ``max(min(c1, c2), 4)`` FLOOR: the envelope
    died within a step or two and redback pads the answer to 4 points. The port reports
    its own (smaller) death index there; the difference still sits inside DK_MAX.
    """
    dk = tde_port["k"] - tde_gold["k_rb"]
    n = dk.size
    exact = int(np.sum(dk == 0))
    print(f"\n[T0 tde engine] dk: exact {exact}/{n}, max |dk| {np.abs(dk).max()}, "
          f"range [{dk.min()}, {dk.max()}], k_rb==4 floored draws "
          f"{int(np.sum(tde_gold['k_rb'] == 4))}")
    assert np.abs(dk).max() <= DK_MAX, np.argmax(np.abs(dk))
    # the majority must agree EXACTLY (measured 239/289 = 83%)
    assert exact / n > 0.7, f"only {exact}/{n} exact"
    # sentinel check: this golden set contains no full-curve (k_rb == 500) draws, so a
    # port curve reaching 500 would be a real disagreement
    assert not np.any(tde_gold["k_rb"] == 500)
    assert not np.any(tde_port["k"] == 500)


def test_tde_engine_parity_L_T_Rph(tde_gold, tde_port):
    """Parity on the first 80% of both curves. Measured (289 draws pooled, n=19681):

        L   median 1.7e-16  p99.9 9.1e-15  max 1.04e-14
        T   median 2.0e-16  p99.9 1.0e-13  max 3.80e-12
        Rph median 3.7e-16  p99.9 2.0e-13  max 7.60e-12  (worst: corner draw 287)

    The last 20% is excluded as the documented ULP-amplification tail of the forward-Euler
    recursion (tde.py CHANGE 5; test_tde_vs_redback._trim), where redback disagrees with
    ITSELF under a one-ulp nudge. Gates carry ~100x headroom over the measurement so an
    XLA scheduling change does not flip them, while remaining ~1e6 below anything a fit
    could feel.
    """
    g, p = tde_gold, tde_port
    gates = {"L": (1e-12, 1e-10), "T": (1e-11, 1e-9), "Rph": (1e-11, 1e-9)}
    for key, gold_key in (("L", "L"), ("T", "T"), ("Rph", "Rph")):
        pool, worst = [], (0.0, -1)
        for i in range(g["params"].shape[0]):
            k = min(int(g["k_rb"][i]), int(p["k"][i]))
            j = max(int(0.8 * k), 2)
            if k < 4:
                continue                      # dead-on-arrival draws: no engine curve
            a, b = p[key][i][:j], g[gold_key][i][:j]
            rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
            pool.append(rel)
            if rel.max() > worst[0]:
                worst = (float(rel.max()), i)
        s = _stats(np.concatenate(pool))
        print(f"[T0 tde engine] {key}: {s} worst draw {worst[1]} "
              f"params {g['params'][worst[1]]}")
        p999_gate, max_gate = gates[key]
        assert s["p999"] < p999_gate, (key, s)
        assert s["max"] < max_gate, (key, s, g["params"][worst[1]])


# ======================================================================= TDE flux density
def _flux_regimes(g, p):
    """Split every (draw, epoch) into interior / termination-zone / dead-draw."""
    z = float(g["redshift"])
    rel = np.abs(p["fd"] - g["flux_density_mjy"]) / np.maximum(
        np.abs(g["flux_density_mjy"]), 1e-300)
    interior, zone, dead = [], [], []
    zone_zero = 0
    for i in range(g["params"].shape[0]):
        finite = np.isfinite(g["flux_density_mjy"][i])
        if p["k"][i] < 2:                    # port: envelope never lived (CHANGE 2/3)
            dead.append(i)
            continue
        kmin = min(int(g["k_rb"][i]), int(p["k"][i]))
        t_src = g["t_obs_days"][i] * DAY / (1.0 + z)
        t_edge = g["time_since_fb"][i][kmin - ZONE_STEPS - 1] if kmin > ZONE_STEPS else -1.0
        inside = (t_src <= t_edge) & finite
        interior.append(rel[i][inside])
        zsel = (~(t_src <= t_edge)) & finite
        zone.append(rel[i][zsel])
        zone_zero += int(np.sum((p["fd"][i][zsel] == 0.0)
                                & (g["flux_density_mjy"][i][zsel] > 0.0)))
    return (np.concatenate(interior), np.concatenate(zone), zone_zero, dead, rel)


def test_tde_flux_density_interior_parity(tde_gold, tde_port):
    """PARITY where the port claims parity. Measured: n=2231, median 5.1e-16,
    p99.9 2.5e-13, max 1.16e-12 (draw 243, epoch 0).

    Protocol gates (dchi2 template, module docstring): p99.9 < 1e-6, max < 2e-3.
    Measured-headroom gate: max < 1e-9 (a real regression trips this first).
    """
    interior, _, _, _, _ = _flux_regimes(tde_gold, tde_port)
    s = _stats(interior)
    print(f"\n[T0 tde flux] interior: {s}")
    assert s["n"] > 1500, "regime split degenerated -- interior nearly empty"
    assert s["p999"] < 1e-6, s          # protocol gate
    assert s["max"] < 2e-3, s           # protocol gate
    assert s["max"] < 1e-9, s           # measured headroom (~1000x above 1.2e-12)
    assert s["median"] < 1e-14, s


def test_tde_flux_density_termination_zone_is_the_documented_deviation(tde_gold, tde_port):
    """The zone (last 24 steps of the shorter curve) is bounded AS a deviation.

    Measured: 2322 zone points; the nonzero ones have median 9.6e-16 and max 1.6e-3 --
    the 'last-epoch constraint+-1' cases, where the two interpolants read
    the ill-conditioned Euler tail through termination indices differing by <= 4 steps.
    203 points (8.7% of the zone) are port-zero-vs-redback-positive: epochs past the
    port's own termination on short curves, i.e. exactly the CHANGE 3 statement that the
    envelope is over -- redback pads those curves to >= 4 points (its max(.., 4) floor)
    and the port refuses to.
    """
    _, zone, zone_zero, dead, _ = _flux_regimes(tde_gold, tde_port)
    nonzero = zone[zone < 1.0]
    s = _stats(nonzero)
    print(f"[T0 tde flux] zone: n={zone.size}, port-zero-vs-gold-positive={zone_zero}, "
          f"nonzero {s}")
    assert s["max"] < 5e-3, s                        # deviation bound (measured 1.6e-3)
    assert np.median(nonzero) < 1e-12                # the zone is still mostly exact
    assert zone_zero / max(zone.size, 1) < 0.15      # measured 8.7%
    # dead draws: redback's constraint floor invents a 4-point curve; the port returns
    # exactly zero flux. Both facts are asserted, as the documented CHANGE 2/3 deviation.
    print(f"[T0 tde flux] dead-envelope draws (port k<2): {dead}")
    assert len(dead) <= 20, dead                     # measured 16/289 = 5.5% of this grid
    for i in dead:
        assert tde_gold["k_rb"][i] == 4, (i, tde_gold["k_rb"][i])   # redback's floor
        assert np.all(tde_port["fd"][i] == 0.0), i
        assert np.nanmax(tde_gold["flux_density_mjy"][i]) > 0.0, i


# ======================================================================= TDE magnitudes
def test_tde_band_magnitudes_deviation_bound(tde_gold):
    """DEVIATION bound, not parity -- and a FINDING beyond the expected envelope.

    The documented deviation: redback evaluates the SED on a 100-point wavelength grid,
    builds an sncosmo TimeSeriesSource and splines it (in wavelength AND time); the port
    integrates the exact Planck band and interpolates (T, R) linearly in time.

    Measured on this golden set (65 draws x 12 epochs, sdss ugri):

        all draws            median 0.076 mmag   p99 36 mmag   max 86.3 mmag
        k_rb >= 100 (29)     median 0.013 mmag                 max  2.3 mmag
        k_rb in [50,100) (8)                                   max 22.4 mmag
        k_rb < 50 (28)       median 0.33 mmag                  max 86.3 mmag

    FINDING (reported, bounded here): the worst case is 86.3 mmag, NOT within the ~25 mmag
    envelope previously quoted for the band-integral deviation -- and it is NOT the
    wavelength grid: regenerating redback's SED with n_lambda = 100 -> 3000 moves it by
    < 0.1 mmag. It is the TIME-interpolation scheme on SHORT curves (k_rb <= 45): sncosmo
    splines band flux through <= 45 time nodes where the port interpolates (T, R)
    linearly between the same nodes. Both are interpolants of the same engine output --
    the deviation measures grid coarseness, not an error in either integrand. On
    well-resolved curves (k_rb >= 100) the two agree to 2.3 mmag, which is BELOW the
    band-integral deviation bound of 25 mmag.
    """
    gm = _need("tde_cooling_envelope_magnitude.npz")
    fs = _need("filterset_t0.npz")
    W, N = kn.ab_weights(fs["lam"], fs["trans"])
    lam_j = jnp.asarray(fs["lam"])
    band_idx = jnp.asarray(gm["band_idx"])
    z, dl = float(gm["redshift"]), float(gm["dl_cm"])

    @jax.jit
    def mag_fn(t, p):
        return T.cooling_envelope_ab_magnitude(
            t, band_idx, W, N, lam_j, z, dl, p[0], p[1], p[2], p[3], p[4],
            **T.REDBACK_PRESETS["1.15"])

    P, tm, gold = gm["params"], gm["t_obs_days"], gm["magnitude"]
    dm = np.zeros_like(gold)
    for i in range(P.shape[0]):
        dm[i] = np.abs(np.asarray(mag_fn(jnp.asarray(tm[i]), jnp.asarray(P[i]))) - gold[i])
    assert np.all(np.isfinite(dm))

    krb = gm["k_rb"]
    long_curves = krb >= 100
    s_all, s_long = _stats(dm), _stats(dm[long_curves])
    im = np.unravel_index(np.argmax(dm), dm.shape)
    print(f"\n[T0 tde mags] all: median {s_all['median']*1e3:.3f} mmag, "
          f"max {s_all['max']*1e3:.1f} mmag at draw {im[0]} (k_rb={krb[im[0]]}, "
          f"params {P[im[0]]}) epoch {im[1]} band {gm['bands'][im[1] % 4]}")
    print(f"[T0 tde mags] k_rb>=100 ({int(long_curves.sum())} draws): "
          f"median {s_long['median']*1e3:.3f} mmag, max {s_long['max']*1e3:.1f} mmag")

    # the documented band-integral deviation bound, on curves the time grid resolves:
    assert np.median(dm) < 5e-3, s_all               # 5 mmag, measured 0.076
    assert long_curves.sum() >= 20
    assert dm[long_curves].max() < 25e-3, s_long     # 25 mmag envelope, measured 2.3
    # the FINDING, bounded so it cannot silently grow: short-curve time-interpolation
    # deviation. 0.12 mag = 1.4x the measured 86.3 mmag worst case.
    assert dm.max() < 0.12, s_all
    # and it must remain attributable to short curves, not leak into resolved ones
    assert krb[im[0]] < 50, (im, krb[im[0]])


# ======================================================================= kilonova
@pytest.fixture(scope="module")
def kn_gold():
    return _need("kilonova_one_component.npz")


@pytest.fixture(scope="module")
def kn_port(kn_gold):
    grid_j = jnp.asarray(kn_gold["grid_s"])

    @jax.jit
    def fn(p):
        return kn.bolometric(grid_j, p[0], p[1], p[2], p[3])

    P = kn_gold["params"]
    n, m = P.shape[0], kn_gold["grid_s"].size
    out = dict(L=np.zeros((n, m)), T=np.zeros((n, m)), Rph=np.zeros((n, m)))
    for i in range(n):
        L, Tt, R = (np.asarray(x) for x in fn(jnp.asarray(P[i])))
        out["L"][i] = L * kn.LSCALE
        out["T"][i] = Tt
        out["Rph"][i] = R
    return out


def _kn_tdiff(params):
    return np.sqrt(kn.TDIFF_CONST * params[:, 2] * params[:, 0] / params[:, 1])


def test_kilonova_parity_inside_validity_horizon(kn_gold, kn_port):
    """t < 1.4 * t_diff: the region where redback's own quadrature is resolved and the
    port claims < 0.3% agreement. Measured (145 draws pooled, 4276 nodes):

        L   median 4.5e-04  p99.9 1.9e-03  max 2.07e-03  (worst at t = 1.38 t_diff,
                                                          i.e. the horizon edge itself)
        T   median 9.1e-05  p99.9 4.6e-04  max 5.19e-04
        Rph median 0        p99.9 9.2e-04  max 1.04e-03

    Nodes with index < 2 are excluded: redback copies L[0] = L[1] by construction. The
    0.3% is redback's trapezoid error inside its horizon, not the port's -- against a
    converged adaptive reference the port holds < 1e-3 everywhere
    (test_kilonova_vs_redback.test_bolometric_matches_a_converged_reference).
    """
    grid = kn_gold["grid_s"]
    td = _kn_tdiff(kn_gold["params"])
    gates = {"L": 3e-3, "T": 1e-3, "Rph": 2e-3}
    for key in ("L", "T", "Rph"):
        pool, worst = [], (0.0, None)
        for i in range(td.size):
            inside = ((np.arange(grid.size) >= 2) & (grid >= 0.15 * td[i])
                      & (grid < 1.4 * td[i]))
            b = kn_gold[key][i][inside]
            rel = np.abs(kn_port[key][i][inside] - b) / np.maximum(np.abs(b), 1e-300)
            pool.append(rel)
            if rel.size and rel.max() > worst[0]:
                worst = (float(rel.max()),
                         (i, float(grid[inside][np.argmax(rel)] / td[i])))
        s = _stats(np.concatenate(pool))
        print(f"\n[T0 kilonova] {key} inside t<1.4*t_diff: {s} "
              f"worst (draw, t/td) {worst[1]} params {kn_gold['params'][worst[1][0]]}")
        assert s["max"] < gates[key], (key, s, worst)


def test_kilonova_documented_divergence_beyond_horizon(kn_gold, kn_port):
    """Past ~2.66 * t_diff redback's geomspace trapezoid is under-resolved and TOO BRIGHT
    (kilonova.py identity 6) -- asserted as a DEVIATION, in the documented direction.

    Measured ratio redback / port, pooled over every golden node beyond the horizon:

        t/t_diff in [2.8, 4)  n=673   min 1.074  median 1.17  max 1.4
        t/t_diff in [4, 8)    n=1218  min 1.369  median 2.24  max 4.6
        t/t_diff >= 8         n=823   min 4.589  median 9.2   max 43.5

    and past t' > 26.6 * t_diff redback's own integrand exp(t'^2/td^2) overflows float64
    to inf, poisoning its cumulative sum to NaN (13 of 145 draws reach it on redback's
    grid); the port (identity 2's folded exponent) must stay finite exactly there.
    """
    grid = kn_gold["grid_s"]
    td = _kn_tdiff(kn_gold["params"])
    bins = {(2.8, 4.0): 1.05, (4.0, 8.0): 1.3, (8.0, np.inf): 4.0}
    nan_draws = 0
    for (f0, f1), floor in bins.items():
        pool = []
        for i in range(td.size):
            sel = ((grid >= f0 * td[i]) & (grid < f1 * td[i])
                   & np.isfinite(kn_gold["L"][i]))
            if sel.any():
                pool.append(kn_gold["L"][i][sel] / kn_port["L"][i][sel])
        ratio = np.concatenate(pool)
        print(f"\n[T0 kilonova] rb/port L at t/td in [{f0},{f1}): n={ratio.size} "
              f"min={ratio.min():.4f} median={np.median(ratio):.2f} "
              f"max={ratio.max():.1f}")
        assert ratio.size >= 3, "not enough beyond-horizon nodes"
        assert ratio.min() > floor, (f0, f1, ratio.min())     # redback too bright, always

    for i in range(td.size):
        bad = ~np.isfinite(kn_gold["L"][i])
        if bad.any():
            nan_draws += 1
            assert grid[np.argmax(bad)] > 20.0 * td[i], i     # NaN starts near 26.6 td
            assert np.all(np.isfinite(kn_port["L"][i][bad])), i
    print(f"[T0 kilonova] redback-NaN draws (t' > ~26.6 t_diff overflow): {nan_draws}")
    assert nan_draws >= 5      # the regime is exercised, not vacuously true


# ======================================================================= precision matrix
# Reported, not gated on f32 quality: the T0 gate is everything above, in float64 on CPU.
# The float32 STATUS of each model is itself a documented claim, so it is verified:
# kilonova is float32-clean by construction (no accumulator); the TDE engine is not
# float32-representable at all (Euler increments ~1e-6 of the state vs eps 1.2e-7) and
# must refuse rather than return a plausible wrong answer.

def test_precision_matrix_tde_requires_float64_and_raises():
    """CHANGE 8, asserted: with x64 off the engine raises at trace time with the fix in
    the message. (Also covered by test_tde_vs_redback; repeated here so the precision
    matrix is complete in one place.)"""
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="requires float64"):
            T.cooling_envelope(1.0, 1.0, 0.05, 0.1, 1.0, n_time=100)
    finally:
        jax.config.update("jax_enable_x64", was)


def test_precision_matrix_kilonova_float32_clean_on_16_draws(tmp_path):
    """Kilonova's documented float32 claim, verified on 16 golden draws: the forward pass
    is finite and band magnitudes agree with float64 to < 1e-3 mag.

    Run in a SUBPROCESS with JAX_ENABLE_X64=0: x64 is a process-wide flag and this test
    module (like the TDE engine) needs it ON everywhere else.
    """
    gk = _need("kilonova_one_component.npz")
    fs = _need("filterset_t0.npz")
    P = gk["params"][:16]
    out = tmp_path / "f32.npz"
    probe = Path(__file__).with_name("_f32_probe.py")
    # the directory that holds whisper_cbpf/ (kn is whisper_cbpf/models/jax/kilonova.py), so the
    # probe imports this package even where it is not pip-installed
    pkg_root = Path(kn.__file__).resolve().parents[3]
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu",
           "PYTHONPATH": str(pkg_root)}
    res = subprocess.run([sys.executable, str(probe), str(GOLDENS), str(out)],
                         env=env, cwd=str(pkg_root), capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0, f"f32 probe failed:\n{res.stdout}\n{res.stderr}"
    f32 = np.load(out)
    assert str(f32["dtype"]) == "float32", f32["dtype"]
    assert np.all(np.isfinite(f32["mag"]))
    assert np.all(np.isfinite(f32["L"])) and np.all(np.isfinite(f32["temp"]))

    # float64 reference for the same points, in this (x64) process
    W, N = kn.ab_weights(fs["lam"], fs["trans"])
    lam_j = jnp.asarray(fs["lam"])
    t_obs = np.asarray(f32["t_obs_days"])
    z, dl = float(f32["redshift"]), float(f32["dl_cm"])
    bidx = jnp.asarray(f32["band_idx"])
    dm_max = 0.0
    for i in range(P.shape[0]):
        t_src = kn.source_time_s(t_obs, z)
        m64 = np.asarray(kn.ab_magnitude(t_src, bidx, W, N, lam_j, z, dl,
                                         float(P[i, 0]), float(P[i, 1]),
                                         float(P[i, 2]), float(P[i, 3])))
        sel = m64 < 39.0                       # off the mag floor
        dm_max = max(dm_max, float(np.max(np.abs(f32["mag"][i][sel] - m64[sel]))))
    print(f"\n[T0 precision] kilonova f32 vs f64: max |dmag| = {dm_max*1e3:.3f} mmag "
          f"on 16 draws (claim: < 1e-3 mag)")
    assert dm_max < 1e-3, dm_max
