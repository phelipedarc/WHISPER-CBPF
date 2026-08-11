"""T3.2 -- continuity: where is the model not smooth, and does every kink have a name?

Two halves, and they answer different questions.

**(1) The likelihood scan** -- ``test_kilonova_continuity`` / ``test_tde_continuity`` /
``test_tde_cliff_demo_observation_past_termination``. Scans logL along every parameter
axis over the FULL prior and demands that every detected transition map to a documented
cause. This is the "is there an undocumented kink anywhere" sweep, described below.

**(2) The targeted measurements** -- three tests that put numbers on the three continuity
claims the modules and PHYSICS_NOTES actually make, rather than only checking that a kink
is where it should be:

* ``test_tde_parameter_jump_scales_with_the_time_grid`` -- the ppm-scale parameter
  discontinuity, in MAGNITUDES, for all 7 parameters at n_time 500 and 5000. Checks
  tde.py CHANGE 6's own figures (0.55 mag at n=500 vs 8.3e-6 mag at n=5000 for mbh_6
  under a +-2e-6 relative change at 99 % of the span).
* ``test_tde_gaussian_rise_stitch_is_structurally_continuous`` -- the Gaussian-rise /
  envelope join, claimed exact by construction (CHANGE 2). Asserted BITWISE, plus the
  size of the C1 kink.
* ``test_kilonova_barnes_kasen_nodes_are_c0_with_a_derivative_kink`` -- PHYSICS_NOTES K5,
  including the part that is easy to get wrong: the kink is at the INTERIOR nodes only.

For each model, a Gaussian log-likelihood is built on a synthetic 40-point ZTF-like
dataset (simulated at a fiducial theta, sigma = 0.05 mag), and logL is scanned along
EVERY parameter axis over the full prior range -- 20,000 points per axis -- plus 3
random diagonal slices. Two detectors run on each slice:

* KINKS:  |second difference of logL| > 100x its median over the slice.
* CLIFFS: |first difference| jumps of order the mag_floor penalty (genuine steps).

Every detected transition is then MAPPED to a documented cause using diagnostics
computed alongside logL at every scan point. A transition that maps to nothing is a
FINDING and fails the test loudly. (One reclassification is allowed first: the global
100x-median flag also fires on smooth-but-steep C2 curvature on an otherwise-flat axis;
:func:`locally_smooth` demotes those -- and ONLY those -- because a genuine kink
concentrates |d2| in one or two grid points while smooth curvature blends into its
neighbours. See its docstring for the measured example.)

Documented causes, kilonova (whisper_cbpf/models/kilonova.py):
  BK-node        the Barnes & Kasen thermalisation table is bilinear, so its derivative
                 jumps at interior grid lines: vej = 0.2, 0.3 (0.1 and 0.4 are edge nodes
                 -- linear extrapolation continues the edge cell's slope, so they are C1;
                 _bilinear_extrap). mej's nodes 0.01 and 0.05 ARE the prior edges, so the
                 mej axis has no interior node to cross.
  floor-crossing the photosphere switches to the temperature_floor branch when the hot-
                 branch temperature of some epoch crosses the floor (a max(), C0 not C1).
  magfloor-cap   an epoch's magnitude reaches the mag_floor cap (min(), C0 not C1;
                 kilonova.py MAG_FLOOR block).

Documented causes, TDE gaussianrise (whisper_cbpf/models/tde.py):
  termination-cliff  an OBSERVATION crosses the envelope's termination time, which moves
                 with the parameters; past it the flux is exactly zero -> mag_floor, a
                 ~20-mag step (documented at length in _interp_photosphere: "a genuine
                 discontinuity in time AND in the parameters ... which HMC cannot see").
  dead-envelope  constraint < 2: the whole curve snaps to mag_floor (CHANGE 2).
  constraint-jitter  the termination index is conditioned to +-4 grid steps (CHANGE 3 /
                 unroll note); at n_time = 500 it moves under ppm-scale parameter changes
                 (CHANGE 6). Small unless an epoch sits near termination.
  stitch-crossing  an observation crosses xi*tfb, where the Gaussian rise hands over to
                 the envelope: continuous by construction, C1 kink.
  grid-node      an observation crosses a node of the n_time interpolation grid, whose
                 positions scale with tfb: piecewise-linear interpolation is C0 there.
                 "THE GRID IS PART OF THE MODEL, not a tolerance" (CHANGE 6).
  magfloor-cap   as for the kilonova.

.. warning:: OBSERVATION PAST TERMINATION (measured by
   ``test_tde_cliff_demo_observation_past_termination``, numbers from this machine's run
   at n_time=500). The T3 datasets keep every epoch inside the envelope span at the
   fiducial, as tde.py instructs ("keep observations inside the envelope's own span").
   Add ONE observation past it -- here a detection at mag 21 placed at 2600 d observer
   frame, ~7% past the fiducial envelope's end -- and the likelihood inherits the cliff:
   scanning stellar_mass across [0.1, 10], logL jumps by ~7.2e4 nats in a single grid
   step (delta ~ 0.5*((40-21)^2 - residual^2)/0.05^2) at m* ~ 1.13, the point where the
   termination time sweeps through that epoch. jax.grad is FINITE AND SMOOTH on both
   sides -- the gradient never sees the step coming -- so NUTS/HMC will happily propose
   across it and silently reject, or worse, never cross. A fit whose termination time
   pushes through its own data must be treated as unconverged, not as a solution.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import _t3_common as C  # noqa: E402  (flips x64 BEFORE jax arrays exist)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
# AFTER anything that might import redback (redback turns usetex on globally); harmless
# if redback was never imported.
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt  # noqa: E402

N_POINTS = 20_000        # per slice; 200k per axis is CPU-prohibitive through the TDE engine
SIGMA = 0.05             # mag
SEED = 4242
KINK_FACTOR = 100.0      # |d2 logL| threshold, in units of the slice median
MERGE_GAP = 50           # grid points; flagged indices closer than this join one cluster
NODE_TOL = 3             # grid points; how close a BK node must be to a cluster to claim it
FIGDIR = Path(__file__).resolve().parent / "figures"

FID_KN = dict(mej=0.03, vej=0.25, kappa=5.0, temperature_floor=2500.0)
FID_TDE = dict(peak_time=20.0, sigma_t=15.0, mbh_6=1.0, stellar_mass=1.0,
               eta=0.05, alpha=0.1, beta=1.0)
BK_VEJ_INTERIOR = (0.2, 0.3)


# =======================================================================================
# slice machinery
# =======================================================================================
def _grids(prior, names):
    """Per-parameter (lo, hi, is_log): scan uniformly in the prior's own space."""
    out = {}
    for nm in names:
        d = prior.distributions[nm]
        lo, hi = (float(x) for x in d.bounds)
        out[nm] = (lo, hi, type(d).__name__ == "LogUniform")
    return out


def _axis_thetas(fid_vec, j, lo, hi, is_log, n=N_POINTS):
    u = np.linspace(np.log(lo) if is_log else lo, np.log(hi) if is_log else hi, n)
    x = np.exp(u) if is_log else u
    theta = np.tile(fid_vec, (n, 1))
    theta[:, j] = x
    return x, theta


def _diagonal_thetas(fid_vec, grids, names, rng, n=N_POINTS):
    """A random straight line through the fiducial, clipped to the prior box.

    Direction drawn in the prior's own space (log for LogUniform); the line runs from
    boundary to boundary through the fiducial.
    """
    lo = np.array([grids[nm][0] for nm in names])
    hi = np.array([grids[nm][1] for nm in names])
    is_log = np.array([grids[nm][2] for nm in names])
    u0 = np.where(is_log, np.log(fid_vec), fid_vec)
    ulo = np.where(is_log, np.log(lo), lo)
    uhi = np.where(is_log, np.log(hi), hi)
    scale = uhi - ulo
    d = rng.standard_normal(len(names)) * scale
    d /= np.linalg.norm(d / scale)
    # largest t range with ulo <= u0 + t*d <= uhi in every coordinate
    with np.errstate(divide="ignore"):
        t1, t2 = (ulo - u0) / d, (uhi - u0) / d
    t_lo = np.max(np.minimum(t1, t2))
    t_hi = np.min(np.maximum(t1, t2))
    tt = np.linspace(t_lo, t_hi, n)
    u = u0[None, :] + tt[:, None] * d[None, :]
    return tt, np.where(is_log[None, :], np.exp(u), u)


def _clusters(flagged_idx, gap=MERGE_GAP):
    if flagged_idx.size == 0:
        return []
    out, start, prev = [], flagged_idx[0], flagged_idx[0]
    for i in flagged_idx[1:]:
        if i - prev > gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def detect_transitions(logl):
    """(clusters, d1, d2, med2): kink clusters from the second difference of logL.

    A cliff IS a kink of enormous size, so one detector covers both; the caller reads
    max|d1| inside the cluster to tell them apart.

    The median is taken over the |d2| that are ABOVE the slice's round-off floor
    (1e-11 x max|logL|). Without that guard an axis with a flat half -- e.g. the
    kilonova's temperature_floor below every epoch's hot-branch temperature, where logL
    is exactly constant and d2 is exactly 0 -- has median 0, and "100x the median" then
    flags every point that is not identically flat.
    """
    d1 = np.diff(logl)
    d2 = np.diff(logl, n=2)
    a2 = np.abs(d2)
    eps = 1e-11 * max(float(np.max(np.abs(logl))), 1.0)
    nz = a2[a2 > eps]
    med2 = float(np.median(nz)) if nz.size else eps
    thr = KINK_FACTOR * max(med2, eps)
    flagged = np.where(a2 > thr)[0] + 1                # d2[i] lives at grid point i+1
    return _clusters(flagged), d1, d2, med2


def locally_smooth(d2, i0, i1, margin=10, window=300, factor=10.0):
    """True where a flagged cluster is SMOOTH CURVATURE, not a derivative discontinuity.

    "100x the slice median |d2|" is a GLOBAL criterion, and it fires on one thing that is
    not a kink: a steeply curving but perfectly C2 stretch of logL on an axis whose
    median curvature is tiny (measured: the TDE peak_time axis near its upper bound,
    where d1 walks -298.99, -299.15, -299.30 ... per step -- a smooth quadratic fall,
    flagged only because the rest of the axis is flat). A genuine kink or cliff
    concentrates |d2| in one or two grid points, so it towers over its NEIGHBOURS; a
    smooth region does not. This classifier compares the cluster's peak |d2| to the
    median |d2| in flanking windows, and only clusters that fail every documented-cause
    match AND this test are reported as findings.
    """
    a2 = np.abs(d2)
    j0, j1 = max(i0 - 2, 0), min(i1, a2.size)
    inside = float(np.max(a2[j0:j1])) if j1 > j0 else 0.0
    flanks = np.concatenate([a2[max(j0 - window, 0):max(j0 - margin, 0)],
                             a2[min(j1 + margin, a2.size):min(j1 + window, a2.size)]])
    if flanks.size < 20:                                # cluster fills the slice: not smooth
        return False
    return inside <= factor * float(np.median(flanks))


def summarise_cluster(xgrid, logl, d1, d2, med2, i0, i1):
    lo, hi = max(i0 - 1, 0), min(i1 + 1, len(logl) - 1)
    seg1 = d1[max(lo - 1, 0):hi]
    seg2 = d2[max(i0 - 2, 0):i1]
    return dict(i0=i0, i1=i1, x_lo=xgrid[lo], x_hi=xgrid[hi],
                max_d1=float(np.max(np.abs(seg1))) if seg1.size else 0.0,
                d2_over_med=float(np.max(np.abs(seg2)) / max(med2, 1e-300)))


def _plot_slice(tag, xlabel, x, logl, clusters, med2, fname, logx=False):
    # redback flips text.usetex on GLOBALLY when imported, and the TDE slices import
    # redback (its prior is read from redback's own files) AFTER this module set the
    # rcParam at import time. Re-assert it here, immediately before drawing -- this is
    # the "AFTER importing redback" rule, enforced at the last possible moment.
    matplotlib.rcParams["text.usetex"] = False
    FIGDIR.mkdir(exist_ok=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1]))
    a.plot(x, logl, lw=0.6, color="#1f5fa8")
    a.set_ylabel("logL")
    a.set_title(tag, fontsize=10)
    d2 = np.abs(np.diff(logl, n=2))
    b.semilogy(x[1:-1], np.maximum(d2, 1e-300), lw=0.4, color="#666666")
    b.axhline(KINK_FACTOR * max(med2, 1e-300), color="#c23b22", lw=0.8, ls="--",
              label=f"{KINK_FACTOR:.0f} x median")
    for cl in clusters:
        for ax in (a, b):
            ax.axvspan(cl["x_lo"], cl["x_hi"], color="#c23b22", alpha=0.25, lw=0)
    if logx:
        a.set_xscale("log")
    b.set_xlabel(xlabel)
    b.set_ylabel("|$\\Delta^2$ logL|")
    b.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, dpi=110)
    plt.close(fig)


def _report(model, slices):
    """Print every transition and its cause; return the unmapped ones."""
    unmapped = []
    for s in slices:
        print(f"  [{model}] slice {s['tag']}: {len(s['clusters'])} transition(s), "
              f"median|d2|={s['med2']:.3g}")
        show = s["clusters"][:15]
        for cl in show:
            causes = cl["causes"] if cl["causes"] else ["UNMAPPED"]
            kind = "CLIFF" if cl["max_d1"] > 100.0 else "kink"
            print(f"      {kind} at {s['pname']} ~ [{cl['x_lo']:.6g}, {cl['x_hi']:.6g}]"
                  f"  max|d1|={cl['max_d1']:.3g}  |d2|/med={cl['d2_over_med']:.3g}"
                  f"  -> {', '.join(causes)}")
            if not cl["causes"]:
                unmapped.append((model, s["tag"], cl))
        if len(s["clusters"]) > 15:
            print(f"      ... and {len(s['clusters']) - 15} more, all mapped"
                  if all(c["causes"] for c in s["clusters"][15:]) else
                  f"      ... and {len(s['clusters']) - 15} more, SOME UNMAPPED")
            unmapped += [(model, s["tag"], c) for c in s["clusters"][15:] if not c["causes"]]
    return unmapped


# =======================================================================================
# kilonova
# =======================================================================================
def _kn_setup():
    from whisper_cbpf.models.jax import kilonova as kn
    from whisper_cbpf.models.jax import kilonova_model

    lam, W, N, fs = C.filters()
    t_obs, band_idx = C.epochs_two_bands(0.5, 25.0, 20)      # 40 observations
    t_src = jnp.asarray(kn.source_time_s(t_obs, C.Z))
    bidx = jnp.asarray(band_idx)
    model = kilonova_model(C.BANDS, C.Z, C.DL_CM, n_wave=C.N_WAVE, filter_set=fs)
    names = list(model.parameters)
    fid = np.array([FID_KN[nm] for nm in names])

    def mags_fn(th):
        return kn.ab_magnitude(t_src, bidx, W, N, lam, C.Z, C.DL_CM,
                               th[0], th[1], th[2], th[3])

    rng = np.random.default_rng(SEED)
    obs = np.asarray(mags_fn(jnp.asarray(fid))) + rng.normal(0.0, SIGMA, t_obs.size)
    obs_j = jnp.asarray(obs)

    def eval_fn(th):
        mags = mags_fn(th)
        logl = C.gaussian_loglike(mags, obs_j, SIGMA)
        # diagnostics for the cause-mapping: how many epochs sit on the floor branch,
        # how many at the cap
        t_hot = kn.bolometric(t_src, th[0], th[1], th[2], 1.0)[1]   # floor 1 K: never binds
        n_floor = jnp.sum(t_hot <= th[3])
        n_cap = jnp.sum(mags >= 40.0 - 1e-9)
        return logl, n_floor, n_cap

    return names, fid, model.default_prior, eval_fn


def _kn_causes(cl, x, thetas, names, n_floor, n_cap):
    """Map one cluster to the documented kilonova causes."""
    i_lo, i_hi = max(cl["i0"] - NODE_TOL, 0), min(cl["i1"] + NODE_TOL, len(x) - 1)
    causes = []
    vej = thetas[:, names.index("vej")]
    v_lo, v_hi = sorted((vej[i_lo], vej[i_hi]))
    if any(v_lo - 1e-12 <= node <= v_hi + 1e-12 for node in BK_VEJ_INTERIOR):
        causes.append("BK-node (vej)")
    if n_floor[i_lo] != n_floor[i_hi]:
        causes.append(f"floor-crossing ({int(n_floor[i_lo])}->{int(n_floor[i_hi])} epochs)")
    if n_cap[i_lo] != n_cap[i_hi]:
        causes.append(f"magfloor-cap ({int(n_cap[i_lo])}->{int(n_cap[i_hi])} epochs)")
    return causes


def test_kilonova_continuity():
    """Sweep logL along every kilonova axis and demand a documented cause for every kink.

    WHAT IT WOULD CATCH. An UNDOCUMENTED derivative discontinuity anywhere in the prior
    box -- the failure mode that makes an HMC posterior quietly wrong rather than loudly
    broken, because the sampler still runs and still returns samples. The three legal
    causes here (Barnes-Kasen node, temperature-floor branch switch, mag_floor cap) are
    each identified from a DIAGNOSTIC computed alongside logL at the same theta, not from
    the kink's position, so a kink that merely happens to sit near a node cannot be
    mislabelled. It also catches the detector going blind: the vej axis must find both
    interior BK nodes, and the mej axis (whose table nodes are the prior's own edges) must
    find none.
    """
    names, fid, prior, eval_fn = _kn_setup()
    grids = _grids(prior, names)
    rng = np.random.default_rng(SEED + 7)
    slices, t0 = [], time.perf_counter()

    jobs = [("axis_" + nm, nm, *_axis_thetas(fid, j, *grids[nm]))
            for j, nm in enumerate(names)]
    for d in range(3):
        tt, thetas = _diagonal_thetas(fid, grids, names, rng)
        jobs.append((f"diag{d}", "t (slice coordinate)", tt, thetas))

    for tag, pname, x, thetas in jobs:
        logl, n_floor, n_cap = C.chunked_vmap(eval_fn, thetas, chunk=512)
        clusters, d1, d2, med2 = detect_transitions(logl)
        summarised = []
        for i0, i1 in clusters:
            cl = summarise_cluster(x, logl, d1, d2, med2, i0, i1)
            cl["causes"] = _kn_causes(cl, x, thetas, names, n_floor, n_cap)
            if not cl["causes"] and locally_smooth(d2, i0, i1):
                cl["causes"] = ["smooth-curvature (C2, no discontinuity; "
                                "global-median flag only)"]
            summarised.append(cl)
        is_log = tag.startswith("axis_") and grids.get(pname, (0, 0, False))[2]
        _plot_slice(f"kilonova {tag}", pname, x, logl, summarised, med2,
                    f"kn_{tag}.png", logx=is_log)
        slices.append(dict(tag=tag, pname=pname, clusters=summarised, med2=med2))
    print(f"kilonova: {len(jobs)} slices x {N_POINTS} points in "
          f"{time.perf_counter() - t0:.1f} s")

    unmapped = _report("kilonova", slices)

    # the detector must PROVE it works: the vej axis crosses both interior BK nodes,
    # and bilinear interpolation guarantees a derivative jump there
    vej_slice = next(s for s in slices if s["tag"] == "axis_vej")
    for node in BK_VEJ_INTERIOR:
        assert any(c["x_lo"] - 1e-3 <= node <= c["x_hi"] + 1e-3
                   and any("BK-node" in cc for cc in c["causes"])
                   for c in vej_slice["clusters"]), (
            f"vej scan failed to detect the documented BK node at {node} -- the detector "
            f"is not working")
    # and the mej axis must be CLEAN of BK kinks: its nodes are the prior edges
    mej_slice = next(s for s in slices if s["tag"] == "axis_mej")
    assert not any(any("BK" in cc for cc in c["causes"]) for c in mej_slice["clusters"]), \
        "BK kink detected INSIDE the mej axis, where the table has no interior node"

    assert not unmapped, (
        f"{len(unmapped)} UNMAPPED transition(s) -- an undocumented kink is a finding:\n"
        + "\n".join(f"  {m} {t}: x in [{c['x_lo']:.6g}, {c['x_hi']:.6g}], "
                    f"max|d1|={c['max_d1']:.3g}" for m, t, c in unmapped))


# =======================================================================================
# TDE gaussianrise
# =======================================================================================
def _tde_setup():
    from whisper_cbpf.models.jax import tde as T

    lam, W, N, _ = C.filters()
    t_obs, band_idx = C.epochs_two_bands(1.0, 400.0, 20)     # 40 observations, 1-400 d
    t = jnp.asarray(t_obs)
    bidx = jnp.asarray(band_idx)
    t_obs_s = jnp.asarray(t_obs * C.DAY)
    preset = dict(T.REDBACK_PRESETS["1.15"])                 # n_time=500, dilation=True
    n_time = preset["n_time"]

    prior, _ = T.default_prior_gaussianrise()
    names = list(T.PARAMETERS_GAUSSIANRISE)
    fid = np.array([FID_TDE[nm] for nm in names])

    def mags_fn(th):
        return T.gaussianrise_cooling_envelope_ab_magnitude(
            t, bidx, W, N, lam, C.Z, C.DL_CM,
            th[0], th[1], th[2], th[3], th[4], th[5], th[6], **preset)

    rng = np.random.default_rng(SEED + 1)
    obs = np.asarray(mags_fn(jnp.asarray(fid))) + rng.normal(0.0, SIGMA, t_obs.size)
    # every fiducial epoch must be inside the envelope span (the doc says fits must)
    assert obs.max() < 39.0, "fiducial dataset has an epoch outside the envelope span"
    obs_j = jnp.asarray(obs)

    def eval_fn(th):
        mags = mags_fn(th)
        logl = C.gaussian_loglike(mags, obs_j, SIGMA)
        out = T.cooling_envelope(th[2], th[3], th[4], th[5], th[6], n_time=n_time)
        k = out["constraint"]
        alive = k >= 2
        tt_last = out["time_temp"][jnp.maximum(k - 1, 0)] * (1.0 + C.Z)
        n_past = jnp.sum((t_obs_s > tt_last) & alive)
        stitch_s = out["tfb"] * (1.0 + C.Z)
        n_pre = jnp.sum(t_obs_s < stitch_s)
        # which interpolation-grid cell each observation sits in; changes when an epoch
        # crosses a node of the (tfb-scaled) grid
        node_sum = jnp.sum(jnp.searchsorted(out["time_temp"] * (1.0 + C.Z), t_obs_s))
        n_cap = jnp.sum(mags >= 40.0 - 1e-9)
        return logl, k, alive, n_past, n_pre, node_sum, n_cap

    return names, fid, prior, eval_fn, mags_fn, obs, (t, bidx, t_obs_s)


def _tde_causes(cl, x, i_lo_hi, k, alive, n_past, n_pre, node_sum, n_cap):
    i_lo, i_hi = i_lo_hi
    causes = []
    if alive[i_lo] != alive[i_hi]:
        causes.append("dead-envelope (constraint < 2)")
    if n_past[i_lo] != n_past[i_hi]:
        causes.append(f"termination-cliff ({int(n_past[i_lo])}->{int(n_past[i_hi])} "
                      f"epochs past)")
    if n_pre[i_lo] != n_pre[i_hi]:
        causes.append("stitch-crossing")
    if node_sum[i_lo] != node_sum[i_hi]:
        causes.append("grid-node")
    if n_cap[i_lo] != n_cap[i_hi]:
        causes.append("magfloor-cap")
    if not causes and np.any(np.abs(np.diff(k[i_lo:i_hi + 1])) >= 1):
        causes.append("constraint-jitter (+-4-step conditioning)")
    return causes


def test_tde_continuity():
    """Same sweep for the TDE, where the kinks are steps rather than slope changes.

    WHAT IT WOULD CATCH. (a) An undocumented transition -- the same failure as the kilonova
    case, but here the legal causes are six, and one of them (termination-cliff) is a
    ~2e5-nat STEP rather than a kink, so a mis-mapping is not a cosmetic error. (b) The
    cliff DISAPPEARING: ``saw_cliff`` fails the test if no slice found a termination or
    dead-envelope transition anywhere, which would mean the zero-flux-outside-the-span
    rule (D3) had stopped applying and the model had gone back to inventing light curves
    past its own death. Both directions matter: a scan that finds nothing is not a pass.
    """
    names, fid, prior, eval_fn, _, _, _ = _tde_setup()
    grids = _grids(prior, names)
    rng = np.random.default_rng(SEED + 13)
    slices, t0 = [], time.perf_counter()

    jobs = [("axis_" + nm, nm, *_axis_thetas(fid, j, *grids[nm]))
            for j, nm in enumerate(names)]
    for d in range(3):
        tt, thetas = _diagonal_thetas(fid, grids, names, rng)
        jobs.append((f"diag{d}", "t (slice coordinate)", tt, thetas))

    saw_cliff = False
    for tag, pname, x, thetas in jobs:
        logl, k, alive, n_past, n_pre, node_sum, n_cap = \
            C.chunked_vmap(eval_fn, thetas, chunk=512)
        clusters, d1, d2, med2 = detect_transitions(logl)
        summarised = []
        for i0, i1 in clusters:
            cl = summarise_cluster(x, logl, d1, d2, med2, i0, i1)
            lo = max(cl["i0"] - NODE_TOL, 0)
            hi = min(cl["i1"] + NODE_TOL, len(x) - 1)
            cl["causes"] = _tde_causes(cl, x, (lo, hi), k, alive, n_past, n_pre,
                                       node_sum, n_cap)
            if not cl["causes"] and locally_smooth(d2, i0, i1):
                cl["causes"] = ["smooth-curvature (C2, no discontinuity; "
                                "global-median flag only)"]
            if any("termination-cliff" in c or "dead-envelope" in c
                   for c in cl["causes"]):
                saw_cliff = True
            summarised.append(cl)
        is_log = tag.startswith("axis_") and grids.get(pname, (0, 0, False))[2]
        _plot_slice(f"tde {tag}", pname, x, logl, summarised, med2,
                    f"tde_{tag}.png", logx=is_log)
        slices.append(dict(tag=tag, pname=pname, clusters=summarised, med2=med2))
        print(f"    tde slice {tag} done ({time.perf_counter() - t0:.0f} s elapsed)",
              flush=True)

    unmapped = _report("tde", slices)

    assert saw_cliff, (
        "no termination cliff detected on any slice: the scan cannot claim to work -- "
        "the documented cliff (zero flux past the envelope's termination) must be found")
    assert not unmapped, (
        f"{len(unmapped)} UNMAPPED transition(s) -- an undocumented kink is a finding:\n"
        + "\n".join(f"  {m} {t}: x in [{c['x_lo']:.6g}, {c['x_hi']:.6g}], "
                    f"max|d1|={c['max_d1']:.3g}" for m, t, c in unmapped))


# =======================================================================================
def test_tde_cliff_demo_observation_past_termination():
    """One deliberate violation: an epoch PAST the fiducial termination, as a warning.

    Measures what the module docstring warns about (see the module-level warning block):
    the logL step when the moving termination time sweeps through a real detection. Not a
    pass/fail scan -- the numbers are printed and the values asserted only to be finite.
    """
    from whisper_cbpf.models.jax import tde as T

    names, fid, prior, _, mags_fn0, obs0, (t, bidx, _) = _tde_setup()
    lam, W, N, _ = C.filters()
    preset = dict(T.REDBACK_PRESETS["1.15"])

    # fiducial envelope, for placing the offending epoch just past its end
    out = T.cooling_envelope(fid[2], fid[3], fid[4], fid[5], fid[6],
                             n_time=preset["n_time"])
    k = int(out["constraint"])
    t_end_obs = float(np.asarray(out["time_temp"])[k - 1]) * (1 + C.Z) / C.DAY
    t_bad = 2600.0
    assert t_bad > t_end_obs, (t_bad, t_end_obs)

    t_all = jnp.concatenate([t, jnp.asarray([t_bad])])
    b_all = jnp.concatenate([bidx, jnp.asarray([0])])
    obs_all = jnp.asarray(np.concatenate([obs0, [21.0]]))    # a real detection out there

    def logl_fn(th):
        mags = T.gaussianrise_cooling_envelope_ab_magnitude(
            t_all, b_all, W, N, lam, C.Z, C.DL_CM,
            th[0], th[1], th[2], th[3], th[4], th[5], th[6], **preset)
        return C.gaussian_loglike(mags, obs_all, SIGMA)

    j = names.index("stellar_mass")
    lo, hi, is_log = _grids(prior, names)["stellar_mass"]
    x, thetas = _axis_thetas(fid, j, lo, hi, is_log)
    logl = C.chunked_vmap(logl_fn, thetas, chunk=512)
    assert np.all(np.isfinite(logl))

    d1 = np.diff(logl)
    i_max = int(np.argmax(np.abs(d1)))
    print(f"  cliff demo: fiducial envelope ends {t_end_obs:.0f} d (observer); extra "
          f"detection at {t_bad:.0f} d, mag 21")
    print(f"  scanning stellar_mass in [{lo}, {hi}]: largest single-step logL jump "
          f"= {abs(d1[i_max]):.4g} nats at m* = {x[i_max]:.4f}")
    g = np.asarray(jax.grad(logl_fn)(jnp.asarray(thetas[i_max])))
    print(f"  jax.grad at the LEFT edge of that step: {g} (finite: "
          f"{bool(np.all(np.isfinite(g)))} -- the gradient cannot see the cliff)")

    _plot_slice("tde cliff demo: detection past termination (WARNING case)",
                "stellar_mass", x, logl, [], float(np.median(np.abs(np.diff(logl, n=2)))),
                "tde_cliff_demo_stellar_mass.png", logx=True)
    assert abs(d1[i_max]) > 1e3, (
        "the demonstration epoch never produced its cliff -- either the epoch is not "
        "past termination or the scan range no longer sweeps termination through it")


# =======================================================================================
# T3.1 QUANTITATIVE -- how far does the MAGNITUDE move under a ppm parameter change?
#
# The logL scans above answer "where are the kinks and does each have a name". This block
# answers the number the module itself quotes, in the units an observer reads:
#
#     tde.py CHANGE 6: "scanning mbh_6 over +-2e-6 relative, the AB magnitude at 99% of
#     the curve spreads by 0.55 mag at n=500 and by 8.3e-6 mag at n=5000."
#
# It is checked here for ALL SEVEN parameters at BOTH grids, at five fixed epochs, because
# a claim about one parameter at one epoch is not a property. The mechanism is the
# termination index: it is an integer read off the ODE trajectory, so a ppm parameter
# change can move it by one grid step, which moves the END of the curve by one step --
# and a FIXED observation at 99% of the span then lands in a different interpolation cell,
# or outside the span altogether (zero flux -> mag_floor). Refining the grid shrinks the
# step, hence the effect; it does not remove it.
# =======================================================================================
PPM_HALFWIDTH = 2e-6     # the module's own figure: "+-2e-6 relative"
PPM_POINTS = 401         # -> relative step 1e-8, i.e. a scan far finer than any sampler
PPM_GRIDS = (500, 5000)  # REDBACK_PRESETS["1.15"] and ["1.12"] / this module's default
#: fractions of the FIDUCIAL envelope's own observer-frame span at which the magnitude is
#: read. 0.99 is the module's quoted point; the rest give the depth profile, and the rise
#: epoch shows that the pre-fallback branch inherits the same jitter through mag_stitch.
SPAN_FRACS = (0.10, 0.50, 0.90, 0.99)


def _tde_fid_vec():
    from whisper_cbpf.models.jax import tde as T

    return list(T.PARAMETERS_GAUSSIANRISE), np.array(
        [FID_TDE[nm] for nm in T.PARAMETERS_GAUSSIANRISE])


def _ppm_epochs(n_time, fid):
    """Fixed observer-frame epochs tied to the FIDUCIAL curve at this ``n_time``.

    Fixed is the point: an observation happens when it happens, and the model's own span
    then slides underneath it as the parameters move. Because the span itself depends on
    ``n_time`` (median 10 % shorter at 500 -- CHANGE 6), each grid gets its own epochs and
    both are reported, rather than pretending one set of days means the same thing on both.
    """
    from whisper_cbpf.models.jax import tde as T

    out = T.cooling_envelope(*fid[2:], n_time=n_time)
    k = int(out["constraint"])
    t_end = float(np.asarray(out["time_temp"])[max(k - 1, 0)]) * (1.0 + C.Z) / C.DAY
    stitch = float(out["tfb"]) * (1.0 + C.Z) / C.DAY
    times = np.array([0.5 * stitch] + [f * t_end for f in SPAN_FRACS])
    labels = ["rise 0.50 x stitch"] + [f"envelope {f:.2f} x span" for f in SPAN_FRACS]
    return times, np.zeros(times.size, dtype=int), labels, k, t_end, stitch


def _ppm_scan(n_time):
    """Two scans over the same +-PPM_HALFWIDTH relative window, for every parameter.

    They measure DIFFERENT THINGS and the difference is the whole point:

    * **FIXED epochs** -- observer-frame days that do not move, tied to the fiducial curve.
      This is what an observation is. A fixed epoch near the end of the span falls OUTSIDE
      the span as soon as the termination index retreats, and outside the span the flux is
      exactly zero (D3), so the magnitude steps straight to ``mag_floor``.
    * **CO-MOVING epochs** -- the same FRACTION of whatever span this theta produces. These
      can never leave the span, so they isolate the interpolation-cell effect from the
      zero-flux cliff. This is the quantity tde.py's "0.55 mag" is measured on.

    Also returned: ``kspan``, how many grid steps the termination index itself moves over
    the window -- the mechanism behind both numbers.
    """
    from whisper_cbpf.models.jax import tde as T

    names, fid = _tde_fid_vec()
    lam, W, N, _ = C.filters()
    t_days, bidx, labels, k, t_end, stitch = _ppm_epochs(n_time, fid)
    t_j, b_j = jnp.asarray(t_days), jnp.asarray(bidx)
    fracs = jnp.asarray(np.array(SPAN_FRACS))

    def probe(th):
        m_fix = T.gaussianrise_cooling_envelope_ab_magnitude(
            t_j, b_j, W, N, lam, C.Z, C.DL_CM,
            th[0], th[1], th[2], th[3], th[4], th[5], th[6],
            n_time=n_time, dilation=True)
        out = T.cooling_envelope(th[2], th[3], th[4], th[5], th[6], n_time=n_time)
        kk = out["constraint"]
        te = out["time_temp"][jnp.maximum(kk - 1, 0)] * (1.0 + C.Z) / C.DAY
        st = out["tfb"] * (1.0 + C.Z) / C.DAY
        t_co = jnp.concatenate([jnp.asarray([0.5]) * st, fracs * te])
        m_co = T.gaussianrise_cooling_envelope_ab_magnitude(
            t_co, b_j, W, N, lam, C.Z, C.DL_CM,
            th[0], th[1], th[2], th[3], th[4], th[5], th[6],
            n_time=n_time, dilation=True)
        return m_fix, m_co, kk

    rel = np.linspace(-PPM_HALFWIDTH, PPM_HALFWIDTH, PPM_POINTS)
    step_rel = float(rel[1] - rel[0])
    out = {}
    for j, nm in enumerate(names):
        theta = np.tile(fid, (PPM_POINTS, 1))
        theta[:, j] = fid[j] * (1.0 + rel)
        m_fix, m_co, kk = C.chunked_vmap(probe, theta, chunk=PPM_POINTS)
        rec = dict(kspan=int(kk.max() - kk.min()), mags=m_fix, mags_co=m_co)
        for tag, m in (("", m_fix), ("_co", m_co)):
            step = np.max(np.abs(np.diff(m, axis=0)), axis=0)
            rec["ptp" + tag] = np.ptp(m, axis=0)
            rec["max_step" + tag] = step
            rec["per_unit_rel" + tag] = step / step_rel
        out[nm] = rec
    return out, labels, k, t_end, stitch, step_rel


def _ppm_property_sweep(n_draws=32, n_pts=101, n_time=500, k_min=50):
    """The fixed-epoch ppm measurement repeated at random draws from the shipped prior.

    One fiducial proves an effect exists; it does not prove it is a property of the model
    over its prior. Here `mbh_6` is scanned over the same +-PPM_HALFWIDTH window at
    `n_draws` prior draws whose envelope is live and resolved (constraint >= k_min), with
    the epoch fixed at 0.99 x THAT DRAW's own span.
    """
    from whisper_cbpf.models.jax import tde as T

    lam, W, N, _ = C.filters()
    prior, _ = T.default_prior_gaussianrise()
    names = list(T.PARAMETERS_GAUSSIANRISE)
    rng = np.random.default_rng(SEED + 303)
    pool = C.sample_prior_matrix(prior, names, 2048, rng)

    def kfun(th):
        out = T.cooling_envelope(th[2], th[3], th[4], th[5], th[6], n_time=n_time)
        kk = out["constraint"]
        return kk, out["time_temp"][jnp.maximum(kk - 1, 0)] * (1.0 + C.Z) / C.DAY

    k_all, span_all = C.chunked_vmap(kfun, pool, chunk=512)
    sel = np.where(k_all >= k_min)[0][:n_draws]
    assert sel.size == n_draws, (
        f"only {sel.size} of 2048 prior draws have constraint >= {k_min}; the sweep cannot "
        f"be built (that itself would be a finding about the prior)")

    rel = np.linspace(-PPM_HALFWIDTH, PPM_HALFWIDTH, n_pts)
    rows = []
    for i in sel:                                    # theta in cols 0-6, epoch in col 7
        block = np.tile(np.append(pool[i], 0.99 * span_all[i]), (n_pts, 1))
        block[:, 2] = pool[i, 2] * (1.0 + rel)
        rows.append(block)
    big = np.concatenate(rows, axis=0)

    def probe(row):
        m = T.gaussianrise_cooling_envelope_ab_magnitude(
            row[7:8], jnp.zeros(1, dtype=jnp.int32), W, N, lam, C.Z, C.DL_CM,
            row[0], row[1], row[2], row[3], row[4], row[5], row[6],
            n_time=n_time, dilation=True)
        return m[0]

    mags = C.chunked_vmap(probe, big, chunk=n_pts).reshape(n_draws, n_pts)
    spreads = np.ptp(mags, axis=1)
    return float(np.mean(spreads > 10.0)), spreads, k_all[sel].astype(int)


def test_tde_parameter_jump_scales_with_the_time_grid():
    """The ppm-scale parameter discontinuity, measured in magnitudes, at n_time 500 vs 5000.

    WHAT IT WOULD CATCH. (a) A port that has quietly become CONTINUOUS in its parameters --
    which here would mean the termination criterion stopped being an index into the
    trajectory, i.e. a different model. (b) The opposite and more dangerous failure: the
    n_time = 5000 grid developing a ppm-scale cliff of its own, which would make the
    project's DEFAULT grid unfittable rather than only redback 1.15.1's. (c) A regression
    in the documented magnitudes: the module states 0.55 mag at n=500 and 8.3e-6 mag at
    n=5000 for mbh_6 at 99 % of the span, and this test recomputes both from scratch.

    ON (c), MEASURED: neither quoted figure is reproduced under the recipe as written.
    In the CO-MOVING convention (the one the quote must mean -- see _ppm_scan) this
    machine measures **5.86 mag at n=500 and 3.67e-4 mag at n=5000** through the exact
    ztfg band integral: 10x and 44x the quoted numbers, same 1e4-scale ratio between the
    grids. In the FIXED convention -- what an observation actually is -- n=500 gives
    **24.5 mag**, the whole mag_floor step. The module's figures are therefore an
    UNDER-statement of the effect, not an over-statement; they are not asserted, the
    direction and the grid scaling are.

    Reported (not asserted) for all 7 parameters x 5 epochs x 2 grids: peak-to-peak spread
    over the window, largest single-step jump, and that jump per unit RELATIVE parameter
    change -- the last is the "mag per unit parameter change" figure, normalised so that
    parameters with different units are comparable. The mechanism (movement of the
    termination INDEX over the same window) is printed alongside, so the magnitudes are
    attributed rather than merely observed.
    """
    names, _ = _tde_fid_vec()
    res = {}
    for n in PPM_GRIDS:
        t0 = time.perf_counter()
        res[n] = _ppm_scan(n)
        print(f"  ppm scan at n_time={n}: {len(names)} params x {PPM_POINTS} points "
              f"in {time.perf_counter() - t0:.1f} s")

    for n in PPM_GRIDS:
        rows, labels, k, t_end, stitch, step_rel = res[n]
        print(f"\n  --- n_time = {n}: fiducial constraint = {k}, span = {t_end:.2f} d "
              f"(observer), stitch = {stitch:.2f} d, relative step = {step_rel:.1e}")
        for tag, what in (("", "FIXED epochs (an observation does not move)"),
                          ("_co", "CO-MOVING epochs (same fraction of each theta's span)")):
            print(f"  {what}")
            print(f"  {'parameter':>13} | " + " | ".join(f"{lab:>19}" for lab in labels))
            for nm in names:
                cells = " | ".join(f"{p:9.3e} /{s:8.1e}" for p, s in
                                   zip(rows[nm]["ptp" + tag], rows[nm]["max_step" + tag]))
                print(f"  {nm:>13} | {cells}    (ptp / max step, mag)")
            worst = max((float(np.max(rows[nm]["per_unit_rel" + tag])), nm) for nm in names)
            print(f"  largest jump per unit RELATIVE parameter change: "
                  f"{worst[0]:.3e} mag at {worst[1]}")
        print("  termination index moves over the window: "
              + ", ".join(f"{nm} {rows[nm]['kspan']:+d} steps" for nm in names))

    i99 = res[500][1].index("envelope 0.99 x span")
    fix500 = float(res[500][0]["mbh_6"]["ptp"][i99])
    fix5000 = float(res[5000][0]["mbh_6"]["ptp"][i99])
    co500 = float(res[500][0]["mbh_6"]["ptp_co"][i99])
    co5000 = float(res[5000][0]["mbh_6"]["ptp_co"][i99])
    print(f"\n  THE DOCUMENTED CLAIM (mbh_6, +-2e-6 relative, 99 % of the span; tde.py "
          f"CHANGE 6 quotes 0.55 mag at n=500 and 8.3e-6 mag at n=5000)")
    print(f"    co-moving epoch  n_time= 500: {co500:.4g} mag   |  n_time=5000: "
          f"{co5000:.4g} mag   (ratio {co500 / max(co5000, 1e-300):.3g}x)")
    print(f"    FIXED epoch      n_time= 500: {fix500:.4g} mag   |  n_time=5000: "
          f"{fix5000:.4g} mag   (ratio {fix500 / max(fix5000, 1e-300):.3g}x)")
    print(f"    -> the quote must mean the CO-MOVING convention; measured here it is "
          f"{co500 / 0.55:.0f}x the quoted 0.55 mag at n=500 and {co5000 / 8.3e-6:.0f}x the "
          f"quoted 8.3e-6 at n=5000, i.e. the module UNDER-states its own effect.")
    print(f"    -> and holding the epoch still, as a real observation does, the same ppm "
          f"change moves the magnitude by {fix500:.1f} mag at n=500 ({fix500 / co500:.1f}x "
          f"the co-moving figure): the fixed epoch leaves the span entirely and the flux "
          f"is exactly zero (D3).")

    for n in PPM_GRIDS:
        for nm in names:
            assert np.all(np.isfinite(res[n][0][nm]["mags"])), (n, nm)
            assert np.all(np.isfinite(res[n][0][nm]["mags_co"])), (n, nm)
    # (a) the coarse grid IS effectively discontinuous at the ppm scale -- if this fails the
    #     termination index has stopped being read off the trajectory
    assert co500 > 0.05, (
        f"mbh_6 co-moving spread at n_time=500 is {co500:.3e} mag, far below the documented "
        f"0.55: the coarse-grid parameter discontinuity has disappeared, which means the "
        f"termination criterion changed")
    # (b) the project's default grid is NOT -- the whole reason 5000 is the default
    assert co5000 < 1e-2 and fix5000 < 1e-2, (
        f"mbh_6 spread at n_time=5000 is {co5000:.3e} (co-moving) / {fix5000:.3e} (fixed) "
        f"mag, against a documented 8.3e-6: the DEFAULT grid has acquired a ppm-scale "
        f"cliff, and no sampler should be pointed at it")
    assert co500 > 30.0 * co5000, (
        f"refining the grid 10x only bought {co500 / max(co5000, 1e-300):.1f}x: the effect "
        f"is no longer the termination-step quantisation the module attributes it to")
    # (c) and the fixed-epoch consequence must be the FULL mag_floor cliff, on every engine
    #     parameter -- this is the number a fit actually meets, and it is not 0.55 mag
    for nm in ("mbh_6", "stellar_mass", "eta", "alpha", "beta"):
        got = float(res[500][0][nm]["ptp"][i99])
        assert got > 10.0, (
            f"a fixed epoch at 99 % of the span no longer sees the zero-flux cliff under a "
            f"ppm change in {nm} at n_time=500 (spread {got:.3e} mag). Either the cliff "
            f"moved or the termination index stopped jittering -- both are findings")

    # --- is this ONE fiducial, or the prior? -----------------------------------------
    frac_cliff, spreads, ks = _ppm_property_sweep()
    print(f"\n  PROPERTY SWEEP -- the same measurement at {len(spreads)} random draws from "
          f"the shipped prior (live, resolved envelopes, constraint >= 50), mbh_6 axis, "
          f"fixed epoch at 0.99 x each draw's OWN span, n_time = 500:")
    print(f"    spread over the +-2e-6 window: min {spreads.min():.3e}, median "
          f"{np.median(spreads):.3e}, max {spreads.max():.3e} mag")
    print(f"    draws where the epoch fell off the cliff (spread > 10 mag): "
          f"{frac_cliff:.1%} ({int(round(frac_cliff * len(spreads)))}/{len(spreads)});  "
          f"constraint over the sweep: {ks.min()}-{ks.max()}")
    print(f"    -> the cliff is NOT the typical case (median {np.median(spreads):.2e} mag: "
          f"most epochs at 0.99 x span stay several grid steps inside it). It is a "
          f"minority of draws, and on those the magnitude is unusable.")
    # The property claim is about the MINORITY, not the median: what must survive is that a
    # ppm change can put a real epoch outside the model's own span on a non-negligible
    # share of the prior, not that it usually does.
    assert spreads.max() > 10.0 and frac_cliff > 0.0, (
        f"over {len(spreads)} random prior draws with a live, resolved envelope, no fixed "
        f"epoch at 0.99 x span fell off the cliff under a +-2e-6 change in mbh_6 "
        f"(max spread {spreads.max():.3e} mag). The effect is then specific to the "
        f"fiducial, not a property of the model over its prior -- which would contradict "
        f"the +4..+6-step termination jitter measured above")


# =======================================================================================
STITCH_DELTAS = (1e-3, 1e-5, 1e-7, 1e-9, 1e-11)
N_STITCH_DRAWS = 128


def test_tde_gaussian_rise_stitch_is_structurally_continuous():
    """The Gaussian-rise / cooling-envelope join: C0 by construction, C1 kinked.

    WHAT IT WOULD CATCH. The rise is normalised to meet the envelope AT ``xi*tfb`` by
    cancelling the normalisation on paper (tde.py CHANGE 2), so the two branches must agree
    at the join EXACTLY -- not to a tolerance. Any regression that reintroduces redback's
    ``norm = F_env(stitch)/exp(...)`` ratio, or evaluates the envelope at a different time
    on the two sides (redback's own two branches disagree about this -- D13), shows up here
    as a step. The claim is checked two ways:

    * STRUCTURALLY: ``_gaussian_rise_magnitude(stitch, stitch, m, ...)`` must return ``m``
      BITWISE -- its parabola term is an exact zero at its own stitch. That is the identity
      that says the normalisation is still cancelled on paper, and it is asserted exactly.
      The two BRANCH values at the node are then compared and required to agree to a few
      ulp of the magnitude (measured 1.421e-14 mag = 4.0 ulp, bitwise on 103 of 121; the
      residual is vectorised-vs-scalar ``jnp.interp`` contraction, and the test rules the
      day<->second epoch conversion out explicitly rather than assuming it).
    * NUMERICALLY: |m(stitch(1-d)) - m(stitch(1+d))| must fall LINEARLY to zero with d.
      The limiting slope of that line is the C1 kink, and it is reported (mag/day) rather
      than asserted away -- a kink is what a stitched model is entitled to have.
    """
    from whisper_cbpf.models.jax import tde as T

    lam, W, N, _ = C.filters()
    preset = dict(T.REDBACK_PRESETS["1.15"])
    prior, _ = T.default_prior_gaussianrise()
    names = list(T.PARAMETERS_GAUSSIANRISE)
    rng = np.random.default_rng(SEED + 101)
    theta = C.sample_prior_matrix(prior, names, N_STITCH_DRAWS, rng)
    deltas = jnp.asarray(np.array(STITCH_DELTAS))
    nd = len(STITCH_DELTAS)

    def probe(th):
        # NOTE `preset` carries `dilation`, which is a PHOTOMETRY option; the engine takes
        # only n_time. Splatting the preset into cooling_envelope raises.
        out = T.cooling_envelope(th[2], th[3], th[4], th[5], th[6],
                                 n_time=preset["n_time"])
        k = out["constraint"]
        stitch_s = out["tfb"] * (1.0 + C.Z)               # xi = 1
        t_end_s = out["time_temp"][jnp.maximum(k - 1, 0)] * (1.0 + C.Z)
        ts = jnp.concatenate([stitch_s * (1.0 - deltas), stitch_s * (1.0 + deltas),
                              jnp.asarray([stitch_s])]) / C.DAY
        b = jnp.zeros(ts.size, dtype=jnp.int32)
        m = T.gaussianrise_cooling_envelope_ab_magnitude(
            ts, b, W, N, lam, C.Z, C.DL_CM,
            th[0], th[1], th[2], th[3], th[4], th[5], th[6], **preset)
        # The rise branch's own value AT the node, rebuilt EXACTLY as the model builds it:
        # scalar (T, R) at the stitch, broadcast to the observation array's shape, then one
        # ab_magnitude call (the shape matters: the band integral is a reduction over
        # n_wave, and a (1, n_wave) block reassociates the sum differently from an
        # (n_obs, n_wave) one).
        def _m_at(t_seconds):
            tp_, rd_ = T._envelope_TR_observer(out, t_seconds, C.Z)
            return T.ab_magnitude(jnp.broadcast_to(tp_, ts.shape),
                                  jnp.broadcast_to(rd_, ts.shape), b,
                                  W, N, lam, C.Z, C.DL_CM,
                                  mag_floor=40.0, dilation=preset["dilation"])[2 * nd]

        m_stitch = _m_at(stitch_s)
        m_rise_at_node = T._gaussian_rise_magnitude(stitch_s, stitch_s, m_stitch,
                                                    th[0], th[1])
        # ...and the same thing evaluated at the epoch the model ACTUALLY receives from an
        # observation table: the public API takes DAYS, so `stitch_s` makes a round trip
        # through /DAY and *DAY_TO_S before `jnp.interp` sees it, and that round trip is not
        # the identity in float64. This isolates that ulp from a real branch disagreement.
        m_stitch_roundtrip = _m_at((stitch_s / C.DAY) * T.DAY_TO_S)
        return (m[:nd], m[nd:2 * nd], m[2 * nd], m_rise_at_node, m_stitch,
                m_stitch_roundtrip, k, stitch_s / C.DAY, t_end_s / C.DAY)

    (m_lo, m_hi, m_node, m_rise_node, m_stitch, m_stitch_rt,
     k, stitch_d, t_end_d) = C.chunked_vmap(probe, theta, chunk=128)

    # a stitch that sits past the envelope's own death has no envelope value to meet: the
    # whole curve is mag_floor there, which is the documented dead-envelope regime, not a
    # statement about the join. Keep the draws where the join is a join.
    usable = (k >= 2) & (stitch_d <= t_end_d) & np.isfinite(m_node) & (m_node < 39.9)
    print(f"  stitch probe: {int(usable.sum())}/{N_STITCH_DRAWS} draws have a live envelope "
          f"whose span reaches the stitch (the rest are the documented dead/short-envelope "
          f"regime: whole curve at mag_floor)")

    # --- structural, part 1: the paper cancellation, bitwise ---------------------------
    # _gaussian_rise_magnitude(s, s, m, tp, sigma) must return m EXACTLY: the parabola term
    # is ((s-tp)^2 - (s-tp)^2)/(2 sigma^2), an exact zero for every finite input. This is
    # the claim that would break the instant anyone re-formed redback's normalising ratio.
    n_paper = int(np.sum(m_rise_node[usable] == m_stitch[usable]))
    print(f"  rise formula at its own stitch returns mag_stitch BITWISE on "
          f"{n_paper}/{int(usable.sum())} draws (the on-paper cancellation, CHANGE 2)")

    # --- structural, part 2: the two branch values at the node ------------------------
    #
    # This one is NOT bitwise, and the reason is worth stating precisely rather than
    # loosening the tolerance until it passes. The envelope branch reads
    # `jnp.interp(t_array, ...)` and picks the element at the stitch; the rise branch's
    # normalisation reads `jnp.interp(stitch_scalar, ...)`. Same formula, same inputs, but
    # XLA is free to contract `f_lo + (x - x_lo) * slope` into an FMA in the vectorised
    # form and not in the scalar one, which moves the result by ~1 ulp. Re-evaluating the
    # scalar form at the day<->second ROUND-TRIPPED epoch (printed below) does NOT recover
    # the missing draws, which rules the epoch conversion out and leaves the vectorised
    # interpolation as the source. The measured residual, 1.4e-14 mag, is ~3 ulp of a
    # 20-mag number -- twelve orders below the 0.05 mag noise of any T3 dataset.
    n_exact = int(np.sum(m_node[usable] == m_rise_node[usable]))
    worst = float(np.max(np.abs(m_node[usable] - m_rise_node[usable])))
    n_rt = int(np.sum(m_node[usable] == m_stitch_rt[usable]))
    ulp = float(np.max(np.spacing(np.abs(m_node[usable]))))
    print(f"  branch values AT the node: bitwise equal on {n_exact}/{int(usable.sum())} "
          f"draws, max |difference| = {worst:.3e} mag = {worst / ulp:.1f} ulp")
    print(f"  same value via the SCALAR interpolation at the round-tripped epoch: bitwise "
          f"on {n_rt}/{int(usable.sum())} -- the day<->second conversion is not the cause; "
          f"the residual is vectorised-vs-scalar jnp.interp contraction")

    # --- numerical: the jump falls linearly in d --------------------------------------
    jumps = np.abs(m_hi[usable] - m_lo[usable])                      # (n_usable, nd)
    print(f"  {'delta (relative)':>18} | {'median |jump|':>13} | {'max |jump|':>11} | "
          f"{'max kink, mag/day':>17}")
    for i, d in enumerate(STITCH_DELTAS):
        kink = jumps[:, i] / (2.0 * d * stitch_d[usable])
        print(f"  {d:>18.0e} | {np.median(jumps[:, i]):13.3e} | "
              f"{jumps[:, i].max():11.3e} | {np.nanmax(kink):17.3e}")

    assert n_paper == int(usable.sum()), (
        f"_gaussian_rise_magnitude no longer returns mag_stitch exactly at the stitch on "
        f"{int(usable.sum()) - n_paper} draw(s): the normalisation is supposed to cancel "
        f"on paper, and this is the identity that says it still does")
    assert worst < 10.0 * ulp, (
        f"the rise and envelope branches differ by {worst:.3e} mag at the stitch = "
        f"{worst / ulp:.1f} ulp of the magnitude itself. Last-place noise from the "
        f"vectorised interpolation is expected; anything above ~10 ulp is a real "
        f"disagreement between the two branches")
    assert jumps[:, -1].max() < 1e-6, (
        f"|m(stitch-) - m(stitch+)| = {jumps[:, -1].max():.3e} mag at a relative offset of "
        f"{STITCH_DELTAS[-1]:.0e}: the join is not C0")
    # linear, not jump-like: shrinking d by 1e8 must shrink the gap by ~1e8
    ratio = jumps[:, 0].max() / max(jumps[:, -2].max(), 1e-300)
    print(f"  jump({STITCH_DELTAS[0]:.0e}) / jump({STITCH_DELTAS[-2]:.0e}) = {ratio:.3e} "
          f"(linear scaling would give {STITCH_DELTAS[0] / STITCH_DELTAS[-2]:.0e})")


# =======================================================================================
BK_VEJ_NODES = (0.2, 0.3, 0.4)          # table nodes inside the prior's vej range (0.1-0.5)
BK_C1_NODES = (0.4,)                    # table EDGE: the extrapolation continues the cell
BK_DELTAS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)


def test_kilonova_barnes_kasen_nodes_are_c0_with_a_derivative_kink():
    """Barnes-Kasen table nodes: value-continuous, derivative-discontinuous, and only
    at the INTERIOR nodes.

    WHAT IT WOULD CATCH. ``_bilinear_extrap`` picks its cell with a STRICT ``x > xg``
    comparison count. A regression to ``>=`` leaves every value unchanged and silently
    flips the one-sided derivative exactly AT a node (kilonova.py says so in as many
    words); a regression to clamping instead of extrapolating would flatten the derivative
    beyond vej = 0.4 -- and both are invisible to any value-only parity test. This one
    reads the derivative directly. It also refutes the lazy version of the claim: the
    kink is NOT at every node. vej = 0.4 is the table's edge, where linear extrapolation
    continues the last cell's slope, so the coefficient is C1 there -- asserted bitwise.

    Reported: the value jump as d -> 0 (C0), the one-sided derivative jump in per cent
    (PHYSICS_NOTES K5 documents 31-46 %), and the same two quantities propagated to the AB
    magnitude a sampler actually sees.
    """
    from whisper_cbpf.models.jax import kilonova as kn

    # --- coefficient level ------------------------------------------------------------
    mej_grid = np.linspace(0.01, 0.05, 9)          # the factory prior's own mej range
    # per-COEFFICIENT derivatives (a, b, d separately), not a summed proxy: PHYSICS_NOTES K5
    # quotes "31-46 %", and that is a statement about the individual table entries.
    _stack = lambda v, m: jnp.stack(kn.thermalisation_coeffs(m, v))  # noqa: E731
    coef = jax.jit(jax.vmap(_stack, in_axes=(None, 0)))
    dcoef = jax.jit(jax.vmap(jax.jacfwd(_stack, argnums=0), in_axes=(None, 0)))

    print(f"  {'node':>6} | {'d':>7} | {'max |value jump|':>16} | {'max deriv jump':>14} | "
          f"{'per-coefficient kink, % of the left slope':>42}")
    kink_pct = {}
    for node in BK_VEJ_NODES:
        for d in BK_DELTAS:
            lo = np.asarray(coef(node - d, mej_grid))
            hi = np.asarray(coef(node + d, mej_grid))
            g_lo = np.asarray(dcoef(node - d, mej_grid))     # (n_mej, 3)
            g_hi = np.asarray(dcoef(node + d, mej_grid))
            vjump = float(np.max(np.abs(hi - lo)))
            gjump = float(np.max(np.abs(g_hi - g_lo)))
            pct = np.max(np.abs(g_hi - g_lo) / np.maximum(np.abs(g_lo), 1e-300),
                         axis=0) * 100.0
            if d == BK_DELTAS[-1]:
                kink_pct[node] = float(np.max(pct))
                print(f"  {node:>6.2f} | {d:>7.0e} | {vjump:16.3e} | {gjump:14.3e} | "
                      f"a {pct[0]:8.2f}  b {pct[1]:8.2f}  d {pct[2]:8.2f}")
            else:
                print(f"  {node:>6.2f} | {d:>7.0e} | {vjump:16.3e} | {gjump:14.3e} |")

    # The value must be continuous at every node. "Continuous" is asserted the only way it
    # can be with finite differences: the observed change across +-d must be ENTIRELY
    # EXPLAINED BY THE LOCAL SLOPE, |f(x+d) - f(x-d)| <= C * max|f'| * 2d with C of order 1.
    # A step would leave a d-independent residual and blow this up as d shrinks. (An
    # absolute tolerance would not distinguish the two: at d = 1e-10 a slope of 8 already
    # produces 1.6e-9 of perfectly continuous variation.)
    for node in BK_VEJ_NODES:
        d = BK_DELTAS[-1]
        vjump = float(np.max(np.abs(np.asarray(coef(node + d, mej_grid))
                                    - np.asarray(coef(node - d, mej_grid)))))
        slope = float(np.max(np.abs(np.asarray(dcoef(node - d, mej_grid)))))
        assert vjump <= 4.0 * slope * 2.0 * d, (
            f"BK table is NOT C0 at vej = {node}: the value moves {vjump:.3e} across "
            f"+-{d:.0e}, more than the local slope {slope:.3g} can account for "
            f"({4.0 * slope * 2 * d:.3e}) -- that residual is a step")
    # ...and the derivative must jump at the INTERIOR nodes only
    for node in BK_VEJ_NODES:
        d = BK_DELTAS[-1]
        gj = float(np.max(np.abs(np.asarray(dcoef(node + d, mej_grid))
                                 - np.asarray(dcoef(node - d, mej_grid)))))
        if node in BK_C1_NODES:
            assert gj == 0.0, (
                f"vej = {node} is the table's EDGE node: linear extrapolation continues the "
                f"last cell's slope, so the derivative must be identical on both sides -- "
                f"got a jump of {gj:.3e}. Has _bilinear_extrap started clamping?")
        else:
            assert kink_pct[node] > 5.0, (
                f"no derivative kink at the interior node vej = {node} "
                f"({kink_pct[node]:.2f} %): bilinear interpolation must kink there, so the "
                f"cell selection is no longer switching cells at the node")

    # one-sided AT the node: the strict `>` puts vej = node in the LEFT cell
    d = 1e-9
    g_at = np.asarray(dcoef(BK_VEJ_NODES[0], mej_grid))
    g_left = np.asarray(dcoef(BK_VEJ_NODES[0] - d, mej_grid))
    g_right = np.asarray(dcoef(BK_VEJ_NODES[0] + d, mej_grid))
    print(f"  AT vej = {BK_VEJ_NODES[0]}: d(a+b+d)/dvej matches the LEFT cell to "
          f"{np.max(np.abs(g_at - g_left)):.2e} and the right cell to "
          f"{np.max(np.abs(g_at - g_right)):.2e}")
    assert np.max(np.abs(g_at - g_left)) < np.max(np.abs(g_at - g_right)), (
        "the derivative exactly AT a node is supposed to be the LEFT cell's (strict `>` in "
        "_bilinear_extrap); it now follows the right cell")

    # --- propagated to the observable -------------------------------------------------
    lam, W, N, _ = C.filters()
    t_obs, band_idx = C.epochs_two_bands(0.5, 25.0, 20)
    t_src = jnp.asarray(kn.source_time_s(t_obs, C.Z))
    bidx = jnp.asarray(band_idx)

    def mag_sum(vej, th):
        m = kn.ab_magnitude(t_src, bidx, W, N, lam, C.Z, C.DL_CM,
                            th[0], vej, th[1], th[2])
        return jnp.sum(m)

    rng = np.random.default_rng(SEED + 202)
    others = np.stack([rng.uniform(0.01, 0.05, 32), rng.uniform(1.0, 30.0, 32),
                       np.exp(rng.uniform(np.log(100.0), np.log(6000.0), 32))], axis=1)
    f = jax.jit(jax.vmap(mag_sum, in_axes=(None, 0)))
    g = jax.jit(jax.vmap(jax.grad(mag_sum, argnums=0), in_axes=(None, 0)))
    print(f"  {'node':>6} | {'d':>7} | {'max |sum-mag jump|':>18} | "
          f"{'slope x 2d (smooth)':>19} | {'d(sum mag)/dvej jump':>21}")
    for node in BK_VEJ_NODES:
        for d in (1e-6, 1e-9):
            vj = float(np.max(np.abs(np.asarray(f(node + d, others))
                                     - np.asarray(f(node - d, others)))))
            gl = np.asarray(g(node - d, others))
            gr = np.asarray(g(node + d, others))
            smooth = float(np.max(np.abs(gl))) * 2.0 * d
            gj = float(np.max(np.abs(gr - gl)))
            print(f"  {node:>6.2f} | {d:>7.0e} | {vj:18.3e} | {smooth:19.3e} | {gj:21.3e}")
            # same C0 criterion as at the coefficient level: no residual beyond the slope
            assert vj <= 4.0 * smooth, (
                f"the AB magnitude is not continuous across the BK node vej = {node}: it "
                f"moves {vj:.3e} mag across +-{d:.0e}, against {smooth:.3e} explainable by "
                f"the local slope (32 draws x 40 epochs)")
        if node not in BK_C1_NODES:
            assert gj > 0.0, f"no gradient kink reached the magnitude at vej = {node}"


# =======================================================================================
if __name__ == "__main__":
    for fn in (test_kilonova_continuity, test_tde_continuity,
               test_tde_parameter_jump_scales_with_the_time_grid,
               test_tde_gaussian_rise_stitch_is_structurally_continuous,
               test_kilonova_barnes_kasen_nodes_are_c0_with_a_derivative_kink,
               test_tde_cliff_demo_observation_past_termination):
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"=== {fn.__name__}: PASS ===")
