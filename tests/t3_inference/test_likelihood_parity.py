"""T3.3 -- likelihood parity: the JAX ports and redback 1.15.1 must agree in logL, in nats.

TWO MODELS, TWO SHAPES OF DEVIATION. ``test_tde_likelihood_parity_vs_redback`` splits on
the TDE's documented termination behaviour (D1/D2/D3/D9 -- a cliff in the parameters);
``test_kilonova_likelihood_parity_vs_redback`` splits on the kilonova's documented
quadrature horizon (K1 -- a horizon in time). Both report |dlogL| absolutely AND against
the scale that decides an inference: 0.5 nats, one sigma of a one-parameter posterior.

Same synthetic-dataset discipline as T3.2, but the quantity compared is the thing a
sampler actually consumes: a Gaussian log-likelihood over a fixed 40-point ZTF-like
dataset, computed once through ``whisper_cbpf.models.jax.tde.cooling_envelope_flux_density``
with ``REDBACK_PRESETS["1.15"]`` (n_time=500, dilation=True) and once through redback
1.15.1's ``cooling_envelope(..., output_format='flux_density')``, on 64 Latin-hypercube
draws over the five engine parameters.

WHY logL AND NOT THE FLUX. Away from termination the two engines agree to ~1e-9 relative
(same grid, same arithmetic spelling -- test_tde_vs_redback.py), but logL multiplies each
flux difference by the residual/sigma^2, and at a random prior draw the residuals are
enormous (the data sit at the fiducial). So |dlogL| < 1e-2 nats is a much STRICTER claim
than per-point flux agreement, and it is the claim model selection depends on.

BOUNDARY DRAWS ARE COUNTED, NOT AVERAGED AWAY. The two implementations are DOCUMENTED to
disagree in three places (tde.py CHANGES 2-3), all of them at or past the envelope's
death:
  * constraint < ~5: redback 1.15.1 floors its termination index at 4 points and, when
    its first search finds nothing, returns a 5000-sentinel "post-mortem" curve, where
    this port returns zero flux (CHANGE 2 -- "ours 0 vs redback 389 at n=500");
  * epochs in the last ~20% of the curve: forward-Euler round-off amplification makes
    the tail irreproducible even between redback and itself under a one-ulp nudge
    (CHANGE 3 / the _trim discipline in test_tde_vs_redback.py);
  * epochs PAST termination: redback's interp1d raises ValueError where this port
    returns exactly zero flux (_interp_photosphere).
A draw in any of those regimes is classified BOUNDARY from the models' own outputs
(constraint, termination times), counted, bounded -- and exempt from the 1e-2 assert,
because the disagreement there is the documented, deliberate divergence, not a port bug.

WHY THE EPOCHS STOP AT 30 DAYS. Measured on this exact LHS (seed 78, n_time=500), the
termination time over the prior box has observer-frame percentiles

    5%: 1.0 d   25%: 20.5 d   50%: 59.3 d   75%: 338.7 d   95%: 4786 d

i.e. HALF of redback's own prior box terminates within ~60 observer days, and 5 of the
64 draws never live at all (constraint < 5). A 400-day grid -- natural as it looks for
a TDE -- puts 52/64 draws in boundary territory and the parity claim would be about
nothing. With 40 points on 2-30 d (where every surviving envelope overlaps the data)
the same LHS gives 41 clean draws. That distribution is a statement about the MODEL
under its shipped prior, and any fit design against real data should check its epochs
against it.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import _t3_common as C  # noqa: E402  (flips x64 BEFORE jax arrays exist)

import jax.numpy as jnp  # noqa: E402

N_DRAWS = 64
SEED = 77
TOL_NATS = 1e-2
TRIM = 0.8               # the _trim discipline: beyond 80% of the curve is tail
#: nominal ZTF g/r effective frequencies, Hz. Both implementations receive the SAME
#: numbers, so parity does not depend on their exact values.
NU_G, NU_R = 6.32e14, 4.79e14
FID = (1.0, 1.0, 0.05, 0.1, 1.0)          # CASES[0] of test_tde_vs_redback.py
PNAMES = ["mbh_6", "stellar_mass", "eta", "alpha", "beta"]
#: the 5 engine-parameter ranges of redback's own gaussianrise prior (its bare
#: cooling_envelope prior pins 4 of the 5, which would make parity trivial)
RANGES = {"mbh_6": (0.1, 20.0, True), "stellar_mass": (0.1, 10.0, True),
          "eta": (1e-4, 0.1, True), "alpha": (0.1, 1.0, True), "beta": (1.0, 5.0, False)}


def _lhs(n, rng):
    """Latin hypercube over RANGES: one stratified sample per dimension, permuted."""
    cols = []
    for nm in PNAMES:
        lo, hi, is_log = RANGES[nm]
        u = (rng.permutation(n) + rng.uniform(0.0, 1.0, n)) / n
        cols.append(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if is_log
                    else lo + u * (hi - lo))
    return np.stack(cols, axis=1)


def _dataset(T):
    """40 ZTF-like flux epochs at the fiducial, all inside the fiducial envelope span.

    2-30 observer days SINCE FALLBACK: early enough that most of the prior box still has
    a live envelope there (see module docstring -- the prior's median termination is
    ~59 d, so a longer grid drowns the comparison in documented boundary cases).
    """
    t_days = np.repeat(np.geomspace(2.0, 30.0, 20), 2)      # observer days SINCE FALLBACK
    nu = np.tile([NU_G, NU_R], 20)
    preset = dict(T.REDBACK_PRESETS["1.15"])
    f_fid = np.asarray(T.cooling_envelope_flux_density(
        jnp.asarray(t_days), jnp.asarray(nu), C.Z, C.DL_CM, *FID, **preset))
    assert np.all(f_fid > 0), "a fiducial epoch fell outside the envelope span"
    sigma = f_fid * (np.log(10.0) / 2.5) * 0.05             # 0.05 mag, in flux
    rng = np.random.default_rng(SEED)
    f_obs = f_fid + rng.normal(0.0, sigma)
    return t_days, nu, f_obs, sigma, preset


def _logl(f_model, f_obs, sigma):
    r = (np.asarray(f_model, dtype=np.float64) - f_obs) / sigma
    return float(-0.5 * np.sum(r * r))


#: What "large enough to move a posterior" means, in nats. 0.5 is the log-likelihood
#: difference between the maximum and the 1-sigma point of a one-parameter Gaussian
#: posterior; a systematic offset of that size RELOCATES a credible interval by 1 sigma.
#: 0.05 -- a tenth of it -- is the threshold used here for "cannot move a posterior".
NAT_1SIGMA = 0.5
NAT_NEGLIGIBLE = 0.05


def _region_table(title, rows, key="dll", scale_key="ll_rb"):
    """Print |dlogL| per region, absolutely and as a fraction of the likelihood's scale.

    Two scales are reported because they answer different questions:

    * ``|dlogL| / |logL|`` -- relative accuracy of the number itself. Reassuring but
      almost meaningless for inference: at a random prior draw |logL| is astronomically
      large (the data sit at the fiducial), so ANY difference looks small against it.
    * ``|dlogL| / 0.5 nats`` -- the operational scale. 0.5 nats is one sigma of a
      one-parameter posterior, so this column reads directly as "how much of a sigma".
      This is the column that decides whether a deviation matters.
    """
    print(f"\n  {title}")
    print(f"  {'region':>22} | {'n':>4} | {'median |dlogL|':>14} | {'max |dlogL|':>12} | "
          f"{'max |dlogL|/|logL|':>18} | {'max / 0.5 nat':>13}")
    out = {}
    for name in REGIONS:
        sel = [r for r in rows if r["region"] == name]
        if not sel:
            print(f"  {name:>22} | {0:>4} | {'--':>14} | {'--':>12} | {'--':>18} | {'--':>13}")
            out[name] = None
            continue
        d = np.array([r[key] for r in sel], dtype=float)
        s = np.array([abs(r[scale_key]) for r in sel], dtype=float)
        finite = np.isfinite(d)
        if not finite.any():
            print(f"  {name:>22} | {len(sel):>4} | {'n/a (redback raised)':>14} | "
                  f"{'--':>12} | {'--':>18} | {'--':>13}")
            out[name] = dict(n=len(sel), median=np.nan, max=np.nan, max_frac=np.nan)
            continue
        dd, ss = d[finite], s[finite]
        frac = dd / np.maximum(ss, 1e-300)
        print(f"  {name:>22} | {len(sel):>4} | {np.median(dd):14.4e} | {dd.max():12.4e} | "
              f"{frac.max():18.3e} | {dd.max() / NAT_1SIGMA:13.3e}")
        out[name] = dict(n=len(sel), median=float(np.median(dd)), max=float(dd.max()),
                         max_frac=float(frac.max()),
                         n_finite=int(finite.sum()), n_nan=int((~finite).sum()))
    return out


#: Exclusive regions, in priority order. Every draw gets exactly one, and each name is a
#: row of PHYSICS_NOTES, not a bucket invented here.
REGIONS = ("parity", "tail (D9/D1)", "past-termination (D3/D10)", "dead-envelope (D2)")


def _classify(k, t_src_max, t_end_jax, t_end_rb, rb_err, n_port_zero):
    """Which documented regime this draw lives in (highest-priority match wins)."""
    if k < 5:
        return REGIONS[3]                      # D2: envelope never lived / redback's floor
    if rb_err is not None or n_port_zero:
        return REGIONS[2]                      # D3/D10: outside the model's own time span
    if t_end_rb > 0 and t_src_max > TRIM * min(t_end_jax, t_end_rb):
        return REGIONS[1]                      # D9/D1: forward-Euler amplification tail
    return REGIONS[0]


def test_tde_likelihood_parity_vs_redback():
    """logL parity against redback 1.15.1 on one fixed synthetic dataset, 64 LHS draws.

    WHAT IT WOULD CATCH. A port bug that per-point flux comparisons cannot see. logL
    multiplies every flux difference by residual/sigma^2, and at a random prior draw the
    residuals are enormous (the data sit at the fiducial), so a relative flux error of
    1e-9 that looks negligible per point becomes a measurable number of nats here -- which
    is the quantity a Bayes factor is made of. It would also catch the regime
    classification itself rotting: every draw is assigned exactly one documented region
    from the models' OWN outputs (constraint, termination times, redback's exception), and
    the parity region is required to keep at least 32 of the 64 draws, so a change that
    quietly pushed the whole LHS into "documented deviation" would fail rather than pass
    vacuously.
    """
    pytest.importorskip("redback")
    import inspect

    from redback.transient_models import tde_models
    from whisper_cbpf.models.jax import tde as T

    # this comparison is against 1.15.1's photometry; refuse to "pass" against 1.12.0
    assert "f_debris" in inspect.getsource(tde_models._cooling_envelope), (
        "installed redback is 1.12.x; this parity test is written against 1.15.1 "
        "(REDBACK_PRESETS['1.15'])")

    t_days, nu, f_obs, sigma, preset = _dataset(T)
    n_time = preset["n_time"]
    theta = _lhs(N_DRAWS, np.random.default_rng(SEED + 1))

    rows, t0 = [], time.perf_counter()
    for i in range(N_DRAWS):
        p = tuple(float(v) for v in theta[i])

        f_jax = np.asarray(T.cooling_envelope_flux_density(
            jnp.asarray(t_days), jnp.asarray(nu), C.Z, C.DL_CM, *p, **preset))
        out = T.cooling_envelope(*p, n_time=n_time)
        k = int(out["constraint"])
        t_end_jax = (float(np.asarray(out["time_since_fb"])[k - 1]) if k >= 1 else 0.0)

        rb_err, f_rb, t_end_rb = None, None, 0.0
        try:
            eng = tde_models._cooling_envelope(*p)
            t_end_rb = float(eng.time_since_fb[-1])
            f_rb = np.asarray(tde_models.cooling_envelope(
                t_days, C.Z, *p, output_format="flux_density", frequency=nu))
        except Exception as exc:                             # documented: interp1d raises
            rb_err = f"{type(exc).__name__}: {exc}"

        t_src = t_days * C.DAY / (1.0 + C.Z)
        n_port_zero = int(np.sum((f_jax == 0.0)
                                 & (np.asarray(f_rb) > 0.0 if f_rb is not None else True)))
        region = _classify(k, t_src.max(), t_end_jax, t_end_rb, rb_err, n_port_zero)

        ll_jax = _logl(f_jax, f_obs, sigma)
        ll_rb = _logl(f_rb, f_obs, sigma) if f_rb is not None else np.nan
        rows.append(dict(p=p, k=k, ll_jax=ll_jax, ll_rb=ll_rb, region=region,
                         n_port_zero=n_port_zero, rb_err=rb_err,
                         dll=abs(ll_jax - ll_rb) if f_rb is not None else np.nan))
    print(f"parity: {N_DRAWS} LHS draws in {time.perf_counter() - t0:.1f} s")

    stats = _region_table(
        f"TDE cooling_envelope flux-density logL, {N_DRAWS} LHS draws, "
        f"{len(t_days)} epochs, n_time={n_time}", rows)
    for r in rows:
        if r["region"] != REGIONS[0]:
            d = f"{r['dll']:.4g}" if np.isfinite(r["dll"]) else "n/a"
            why = (f"redback raised {r['rb_err'].splitlines()[0][:48]}" if r["rb_err"]
                   else (f"{r['n_port_zero']} epoch(s) zero in the port, positive in redback"
                         if r["n_port_zero"] else "epochs in the last 20% of the curve"))
            print(f"    [{r['region']}] k={r['k']:>4}  |dlogL|={d:>11}  <- {why}")

    clean = [r for r in rows if r["region"] == REGIONS[0]]
    dll_clean = np.array([r["dll"] for r in clean])
    ll_all = np.array([r["ll_jax"] for r in rows if np.isfinite(r["ll_jax"])])
    print(f"\n  scale of the likelihood over this LHS: logL spans "
          f"[{ll_all.min():.4g}, {ll_all.max():.4g}] nats "
          f"(range {ll_all.max() - ll_all.min():.4g}); one posterior sigma = "
          f"{NAT_1SIGMA} nats")
    verdict = ("NO deviation in the parity region can move a posterior"
               if dll_clean.max() < NAT_NEGLIGIBLE else
               "a parity-region deviation EXCEEDS a tenth of a posterior sigma")
    print(f"  VERDICT (parity region): max |dlogL| = {dll_clean.max():.3e} nats = "
          f"{dll_clean.max() / NAT_1SIGMA:.2e} sigma -- {verdict}")
    for name in REGIONS[1:]:
        s = stats[name]
        if s is None:
            continue
        if not np.isfinite(s["max"]):
            print(f"  VERDICT ({name}): redback returns NO likelihood at all "
                  f"({s['n']} draw(s)) -- not a difference in logL, an absence of one")
        else:
            print(f"  VERDICT ({name}): max |dlogL| = {s['max']:.3e} nats = "
                  f"{s['max'] / NAT_1SIGMA:.2e} sigma -- "
                  f"{'moves a posterior' if s['max'] > NAT_1SIGMA else 'does not'}")

    # the parity claim, on the draws where the two implementations claim to be the same
    # (measured on this seed: 41 clean; the margin absorbs the +-4-step termination
    # conditioning flipping borderline draws on another backend)
    assert len(clean) >= 32, (
        f"only {len(clean)} clean draws -- the epoch grid is spending the whole LHS in "
        f"boundary territory and the test has lost its teeth")
    assert dll_clean.max() < TOL_NATS, (
        f"|dlogL| = {dll_clean.max():.3e} nats on a NON-boundary draw (tolerance "
        f"{TOL_NATS}): the two implementations disagree away from the documented "
        f"divergence regions -- that is a port bug, not conditioning")
    # ... and, more strongly, it must be small on the scale that decides a posterior
    assert dll_clean.max() < NAT_NEGLIGIBLE, (
        f"parity-region |dlogL| reaches {dll_clean.max():.3e} nats, more than a tenth of "
        f"the {NAT_1SIGMA}-nat one-sigma scale: this could relocate a credible interval")
    # the deviation set must stay a minority, and every member must carry a documented tag
    bdry = [r for r in rows if r["region"] != REGIONS[0]]
    assert len(bdry) <= N_DRAWS // 2, (
        f"{len(bdry)}/{N_DRAWS} draws are boundary cases -- the dataset should be "
        f"redesigned, this is no longer 'counted separately', it is the population")
    assert all(r["region"] in REGIONS for r in rows)


# =======================================================================================
# Kilonova: the same question for the OTHER documented deviation, K1
#
# The TDE test above splits on termination. The kilonova's documented deviation has a
# completely different shape -- it is a QUADRATURE horizon in TIME, not a cliff in the
# parameters -- so it needs its own likelihood-parity statement, with its own regime split:
#
#   PHYSICS_NOTES K1 / kilonova.py identity 6: redback's diffusion integral is a 300-point
#   geomspace trapezoid whose kernel narrows as t_diff^2/2t while its spacing grows. Below
#   ~1.4 t_diff it is resolved (parity region); beyond ~2.66 t_diff it is under-resolved
#   and TOO BRIGHT, by up to ~11x in L, and redback's own grid refined to 2e5 points
#   converges onto the port (ratio 7.37 -> 1.00004).
#
# WHY BOLOMETRIC AND NOT BAND MAGNITUDES. redback's photometric path adds K2 on top (its
# 100-point SED splined through an sncosmo TimeSeriesSource), so a band-magnitude
# comparison measures K1 and K2 together and can attribute neither. The T0 tier already
# bounds K2 separately. Here the likelihood is built on log10 L_bol, which isolates K1.
#
# WHY THE EPOCHS ARE REDBACK'S OWN GRID NODES. `_one_component_kilonova_model` runs
# `cumulative_trapezoid` over EXACTLY the array it is handed, so handing it 20 observing
# epochs would not evaluate redback's model at those epochs -- it would evaluate a
# different, far coarser quadrature. "Identical data" therefore means: epochs chosen from
# redback's own default grid, geomspace(1e-3, 7e6, 300).
# =======================================================================================
KN_PARITY_FACTOR = 1.4        # resolved below this multiple of t_diff
KN_HORIZON_FACTOR = 2.66      # measurably too bright above it
KN_SIGMA_DEX = 0.05           # ~0.125 mag, the same S/N as the TDE dataset above
KN_DRAWS = 24
KN_REGIONS = ("parity  t < 1.4 t_diff", "transition 1.4-2.66", "K1 horizon t > 2.66")


def test_kilonova_likelihood_parity_vs_redback():
    """logL parity for the kilonova, split at redback's own quadrature horizon.

    WHAT IT WOULD CATCH. (a) A port that silently drifted from redback INSIDE the horizon,
    where the two are supposed to be the same integral -- that would be a port bug. (b) The
    deviation OUTSIDE the horizon changing sign or vanishing: it is documented as
    "redback too bright", and a port that agreed with redback out there would mean the
    re-quadrature (identity 6) had been undone. (c) The size of the deviation in the only
    units that matter for model selection -- nats of log-likelihood on a real dataset.

    Not a pass/fail on the horizon side: that difference is deliberate and its magnitude is
    reported, together with a plain statement of whether it can move a posterior.
    """
    pytest.importorskip("redback")
    from redback.transient_models.kilonova_models import _one_component_kilonova_model

    from whisper_cbpf.models.jax import kilonova as kn

    grid = np.geomspace(1e-3, 7e6, 300)                 # redback's own default grid, seconds
    rng = np.random.default_rng(SEED + 5)
    # the factory prior's own box (models/__init__.py kilonova_model): mej U(0.01, 0.05),
    # vej U(0.1, 0.5), kappa U(1, 30). temperature_floor does not enter L_bol.
    draws = np.stack([rng.uniform(0.01, 0.05, KN_DRAWS), rng.uniform(0.1, 0.5, KN_DRAWS),
                      rng.uniform(1.0, 30.0, KN_DRAWS)], axis=1)

    rows, t0 = [], time.perf_counter()
    for mej, vej, kappa in draws:
        t_diff_s = float(np.sqrt(kn.TDIFF_CONST * kappa * mej / vej))
        # 24 epochs spread over 0.2-8 t_diff, SNAPPED to redback's own grid nodes so both
        # codes are evaluated at literally the same times
        want = np.geomspace(0.2 * t_diff_s, 8.0 * t_diff_s, 24)
        idx = np.unique(np.clip(np.searchsorted(grid, want), 1, grid.size - 1))
        t_s = grid[idx]

        L_rb = np.asarray(_one_component_kilonova_model(
            grid, mej, vej, kappa, temperature_floor=4000.0)[0])[idx]
        # kn.bolometric returns L in the module's SCALED units; LSCALE puts it in erg/s,
        # which is what redback returns (tests/test_kilonova_vs_redback.py does the same).
        L_jx = np.asarray(kn.bolometric(jnp.asarray(t_s), mej, vej, kappa, 4000.0)[0],
                          dtype=np.float64) * kn.LSCALE
        ok = np.isfinite(L_rb) & (L_rb > 0) & np.isfinite(L_jx) & (L_jx > 0)

        # data simulated from the PORT (the model under test), 0.05 dex noise
        y = np.log10(np.where(ok, L_jx, 1.0)) + rng.normal(0.0, KN_SIGMA_DEX, t_s.size)
        r_jx = (np.log10(np.where(ok, L_jx, 1.0)) - y) / KN_SIGMA_DEX
        r_rb = (np.log10(np.where(ok, L_rb, 1.0)) - y) / KN_SIGMA_DEX

        tau = t_s / t_diff_s
        masks = (ok & (tau < KN_PARITY_FACTOR),
                 ok & (tau >= KN_PARITY_FACTOR) & (tau <= KN_HORIZON_FACTOR),
                 ok & (tau > KN_HORIZON_FACTOR))
        for name, m in zip(KN_REGIONS, masks):
            if not m.any():
                continue
            ll_jx = float(-0.5 * np.sum(r_jx[m] ** 2))
            ll_rb = float(-0.5 * np.sum(r_rb[m] ** 2))
            rows.append(dict(region=name, n_ep=int(m.sum()), ll_jax=ll_jx, ll_rb=ll_rb,
                             dll=abs(ll_jx - ll_rb),
                             max_ratio=float(np.max(L_rb[m] / L_jx[m])),
                             min_ratio=float(np.min(L_rb[m] / L_jx[m])),
                             p=(mej, vej, kappa)))
        rows.append(dict(region="__nan__", n_ep=int((~ok).sum()), ll_jax=np.nan,
                         ll_rb=np.nan, dll=np.nan, max_ratio=np.nan, min_ratio=np.nan,
                         p=(mej, vej, kappa)))
    print(f"kilonova parity: {KN_DRAWS} draws in {time.perf_counter() - t0:.1f} s")

    n_bad = sum(r["n_ep"] for r in rows if r["region"] == "__nan__")
    print(f"  epochs where redback's own integrand is non-finite or non-positive: {n_bad} "
          f"(identity 2: its exp(t'^2/t_d^2) overflows past t' ~ 26.6 t_diff; the port is "
          f"finite there -- see T0 finding 2)")
    print(f"\n  {'region':>22} | {'draws':>5} | {'median |dlogL|':>14} | {'max |dlogL|':>12} "
          f"| {'max |dlogL|/|logL|':>18} | {'L_rb/L_port: min..max':>24}")
    summary = {}
    for name in KN_REGIONS:
        sel = [r for r in rows if r["region"] == name]
        if not sel:
            continue
        d = np.array([r["dll"] for r in sel])
        frac = d / np.maximum(np.abs([r["ll_rb"] for r in sel]), 1e-300)
        rmax = np.array([r["max_ratio"] for r in sel])
        rmin = np.array([r["min_ratio"] for r in sel])
        print(f"  {name:>22} | {len(sel):>5} | {np.median(d):14.4e} | {d.max():12.4e} | "
              f"{frac.max():18.3e} | {rmin.min():10.6f} .. {rmax.max():10.4f}")
        summary[name] = dict(n=len(sel), median=float(np.median(d)), max=float(d.max()),
                             max_ratio=float(rmax.max()), min_ratio=float(rmin.min()))

    par, hor = summary[KN_REGIONS[0]], summary.get(KN_REGIONS[2])
    # The physics statement, decoupled from this dataset's noise: how far apart are the two
    # INTEGRALS inside the region where both are converged? (T0 measured max 0.21 % in L.)
    dev_par = max(abs(par["max_ratio"] - 1.0), abs(par["min_ratio"] - 1.0))
    print(f"\n  VERDICT (parity region, t < {KN_PARITY_FACTOR} t_diff): the two integrals "
          f"differ by at most {dev_par:.3%} in L -- that is redback's own trapezoid error "
          f"inside its horizon, not a port deviation")
    print(f"    consequence for logL at {KN_SIGMA_DEX} dex per epoch: max |dlogL| = "
          f"{par['max']:.3e} nats = {par['max'] / NAT_1SIGMA:.2e} sigma -- "
          f"{'cannot move' if par['max'] < NAT_NEGLIGIBLE else 'CAN move'} a posterior")
    if hor:
        print(f"  VERDICT (K1 horizon, t > {KN_HORIZON_FACTOR} t_diff): L_redback/L_port "
              f"reaches {hor['max_ratio']:.3f}x and max |dlogL| = {hor['max']:.4g} nats = "
              f"{hor['max'] / NAT_1SIGMA:.3g} sigma. This is a DIFFERENT MODEL, not a "
              f"rounding difference -- and redback is the one that is wrong (its geomspace "
              f"trapezoid under-resolves the narrowing kernel; identity 6 / K1)")

    # inside redback's own horizon the two integrals must agree to redback's own quadrature
    # error there -- NOT to machine precision, because the port is being compared against a
    # coarser quadrature of the same integral. T0 measured max 0.21 % in L; 1 % is the gate.
    assert dev_par < 0.01, (
        f"the two integrals differ by {dev_par:.3%} in L INSIDE redback's resolved region "
        f"(t < {KN_PARITY_FACTOR} t_diff), where T0 measures redback's own trapezoid error "
        f"at 0.21 %: the port and redback are no longer the same integral where both "
        f"converge")
    # ... and outside it, the documented divergence must still be there, in the documented
    # direction (redback brighter). A port that agreed here would have lost identity 6.
    assert hor is not None and hor["min_ratio"] > 1.0, (
        f"redback is no longer strictly brighter than the port beyond its quadrature "
        f"horizon (min ratio {hor['min_ratio'] if hor else float('nan'):.4f}): either the "
        f"re-quadrature (identity 6) was undone, or the horizon moved")


if __name__ == "__main__":
    for fn in (test_tde_likelihood_parity_vs_redback,
               test_kilonova_likelihood_parity_vs_redback):
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"=== {fn.__name__}: PASS ===")
