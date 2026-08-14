#!/usr/bin/env python3
"""Simulation-Based Calibration for the kilonova: are the posteriors CALIBRATED, and do the
samplers agree with each other?

Why this exists. `scripts/check_sampler_agreement.py` compares samplers against a long reference
run, and on AT2017GFO that failed for a reason no amount of sampling fixes: the one-component
kilonova cannot fit that data, so the "reference" posterior collapses onto a corner of the prior
(mej pinned at its 0.05 ceiling with a standard deviation of 2.2e-06). Comparing against a delta
function measures misspecification, not agreement.

Here the data is SIMULATED FROM THE MODEL, so the posterior is interior, informative, and has a
known truth. That converts two vague questions into sharp ones:

  CALIBRATION -- are the credible intervals honest? Draw theta* from the prior, simulate a light
      curve at theta*, fit it, and record where theta* falls among the posterior draws. Repeat.
      If the sampler samples the posterior the model and prior imply, those RANKS ARE UNIFORM
      (Talts et al. 2018). A U-shaped rank histogram means over-confident (intervals too narrow);
      an inverted-U means under-confident; a slope means biased. Equivalently, the fraction of
      simulations whose truth falls inside the nominal q% interval should equal q -- that is the
      COVERAGE / P-P plot, and the diagonal is the answer.

  AGREEMENT -- given the SAME simulated dataset, do different samplers return the same posterior?
      Here the normalised distances mean something, because the reference posterior has real width.

ABC is deliberately excluded. A rejection tolerance makes the posterior broader than the true one by
construction, so its ranks are expected to be non-uniform; including it would report a known and
intended property as a calibration failure. Only likelihood-based samplers and SNPE are run.

Noise is realistic by construction: the per-point sigmas come from the real AT2017GFO photometry
(each epoch inherits the median magnitude_err of the real points in that band within +/-0.5 d), so
the error bars grow as the source genuinely fades rather than being a convenient invention.

The default observing setup is a SINGLE band on a NIGHTLY cadence for 10 days -- what a survey
actually delivers for most events, and a far easier inference problem than AT2017GFO's 212 points
across three bands. That matters twice over: looser constraints keep the posterior interior instead
of pinned to a prior face, and cheap fits are what make 1000 simulations affordable. Where the
requested epochs run past the real coverage in that band, the last measured sigma is carried
forward and the run says so on stdout.

Usage::

    python scripts/sbc_kilonova.py --n-sims 1000 --device gpu
    python scripts/sbc_kilonova.py --n-sims 4 --device gpu --samplers mcmc,nested   # smoke
    python scripts/sbc_kilonova.py --merge shardA,shardB --out merged                # combine

Outputs in ``--out``: ranks + coverage as PNG, an overlay corner for one simulation, and
``sbc.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from compare_at2017gfo import (  # noqa: E402
    BAND_ALIASES, DL_CM, FILTER_SET, REDSHIFT, _load_lc,
)

#: Likelihood-based samplers plus SNPE. ABC is excluded on purpose -- see the module docstring.
DEFAULT_SAMPLERS = ["mcmc", "nested", "nuts_gpu", "pymc_jax_gpu_vectorized", "snpe"]

#: Budgets sized so a full SBC study finishes overnight rather than never. SBC needs MANY fits, so
#: each one is deliberately modest; the rank statistic is valid at any budget provided the sampler
#: has actually converged, which is why `converged` is recorded per fit and reported.
#: Raised from the smoke budget after MEASURING that it did not converge: at nsteps=3000 emcee
#: reported converged=False on 2/2 simulations, and at (600, 600, 2 chains) so did NUTS/PyMC, which
#: also warned that 4 chains are needed for a robust r-hat. A rank from an unconverged fit measures
#: the failure to converge rather than the posterior, so the cheaper budget would have bought a
#: study that could not be read. Roughly 2x the cost.
SBC_BUDGET = {
    "mcmc": dict(nsteps=6000, burnin=2000, nwalkers=32, thin=5),
    "nested": dict(nlive=300, dlogz=0.5),           # already converged=True at this budget
    "nuts_gpu": dict(num_warmup=1000, num_samples=1000, num_chains=4),
    "emcee_jax": dict(nwalkers=32, nsteps=6000, burnin=2000, thin=5),
    "pymc_jax_gpu_vectorized": dict(num_warmup=1000, num_samples=1000, num_chains=4),
    "pymc_jax_gpu_parallelized": dict(num_warmup=1000, num_samples=1000, num_chains=4),
    #: three rounds, on the GPU, as asked. num_rounds>1 is the "sequential" in SNPE: each round
    #: proposes from the latest posterior, so simulations concentrate where the data points.
    "snpe": dict(num_rounds=3, num_simulations=1500, num_samples=4000, device="cuda"),
}

#: Nominal credible levels for the coverage / P-P plot.
LEVELS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.75, 0.8, 0.9, 0.95, 0.99])

#: Below this many simulations, NO calibration verdict is issued.
#:
#: A chi-square-of-uniformity test on a handful of ranks has essentially no power, so it returns a
#: large p-value and the harness would print "calibrated: True" -- a pass earned by having too
#: little data to fail. The smoke run did exactly that from TWO simulations. Talts et al. use of
#: order 1000; 100 is the usual practical floor for spotting a gross miscalibration, and the
#: binomial band on the coverage plot is +/-9 percentage points even there.
MIN_SIMS_FOR_VERDICT = 100


def simulate(model, template, theta, rng):
    """One synthetic light curve: the model at ``theta``, on the template's epochs and errors.

    Returns a magnitude-mode LightCurve. The noise is Gaussian in MAGNITUDE with the real per-point
    ``magnitude_err``, which is what the Gaussian likelihood in magnitude space assumes -- so the
    generative model and the inference model match exactly. Any mismatch there would show up as a
    calibration failure that is the harness's fault rather than the sampler's.
    """
    import whisper_cbpf as wp
    from whisper_cbpf.io.photometry import AB_ZEROPOINT_JY

    times, bands, err = template
    flux = np.asarray(model.predict(theta, times, bands), dtype=float)
    mag = -2.5 * np.log10(np.clip(flux, 1e-300, None) / AB_ZEROPOINT_JY)
    obs = mag + rng.normal(0.0, err)
    return wp.LightCurve(time=times, band=bands, magnitude=obs, magnitude_err=err,
                         name="sbc", redshift=REDSHIFT)


def _build_template(args):
    """The observing setup every simulation is generated on: epochs, bands, per-point sigma.

    A single band on a nightly cadence, which is what a survey actually delivers for most events
    and a far easier inference problem than AT2017GFO's 212 points across three bands. Fewer, wider
    constraints keep the posterior INTERIOR rather than pinned to a prior face, which is the regime
    SBC needs -- and the cheap fits are what make 1000 simulations affordable.

    Sigmas are taken from the real AT2017GFO data rather than invented: each epoch inherits the
    median magnitude_err of the real points in that band within +/-0.5 d, so the noise tracks the
    source genuinely fading. IMPORTANT AND STATED IN THE OUTPUT: AT2017GFO's g band stops at
    7.47 d, so epochs beyond it have NO real coverage and carry the last measured error forward.
    Those sigmas are extrapolated, not measured, and the printed template says which.
    """
    real = _load_lc()
    full = real.add_mag(3631.0)
    rt = np.asarray(real.time, float)
    rb = np.asarray([BAND_ALIASES.get(str(b), str(b)) for b in np.asarray(real.band)])
    re_ = np.asarray(full.magnitude_err, float)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    epochs = np.arange(1, args.n_days + 1, dtype=float) * args.cadence

    times, bnames, errs, extrapolated = [], [], [], 0
    for b in bands:
        canon = BAND_ALIASES.get(b, b)
        sel = rb == canon
        if not sel.any():
            raise SystemExit(f"band {b!r} (canonical {canon!r}) is not in the reference data; "
                             f"available: {sorted(set(rb))}")
        bt, be = rt[sel], re_[sel]
        for t in epochs:
            near = np.abs(bt - t) <= 0.5
            if near.any():
                sig = float(np.median(be[near]))
            else:                                   # past the real coverage -- carry last forward
                sig = float(be[np.argmax(bt)])
                extrapolated += 1
            times.append(t), bnames.append(canon), errs.append(sig)

    times = np.asarray(times, float)
    errs = np.asarray(errs, float)
    last = float(rt[np.isin(rb, [BAND_ALIASES.get(b, b) for b in bands])].max())
    print(f"template: {len(times)} epochs, bands {bands}, days 1-{args.n_days} "
          f"at {args.cadence:g} d cadence, sigma {errs.min():.3f}-{errs.max():.3f} mag "
          f"(median {np.median(errs):.3f})")
    if extrapolated:
        print(f"  !! {extrapolated} epoch(s) lie beyond the real {bands} coverage (ends {last:.2f} d); "
              f"their sigma is the last measured value carried forward, i.e. EXTRAPOLATED.")
    return times, np.asarray(bnames), errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sims", type=int, default=32)
    ap.add_argument("--bands", default="g",
                    help="comma-separated bands to simulate (default: g only)")
    ap.add_argument("--n-days", type=int, default=10,
                    help="number of epochs, one per cadence step (default: 10)")
    ap.add_argument("--cadence", type=float, default=1.0,
                    help="days between epochs (default: 1.0, i.e. nightly)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--samplers", default=",".join(DEFAULT_SAMPLERS))
    ap.add_argument("--n-wave", type=int, default=300,
                    help="filter-set resolution; 300 keeps a full study affordable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "_scratch" / "sbc"))
    ap.add_argument("--merge", default=None,
                    help="comma-separated shard dirs: re-render ranks/coverage from "
                         "their raw arrays and exit (no fitting)")
    args = ap.parse_args(argv)

    if args.merge:
        return _merge([Path(d.strip()) for d in args.merge.split(",") if d.strip()],
                      Path(args.out), args)

    if args.device == "gpu":
        import jax
        jax.config.update("jax_enable_x64", True)          # BEFORE the first array

    import whisper_cbpf as wp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    if args.device == "gpu":
        import jax
        ok, msg = wp.check_gpu()
        print(f"device: {jax.devices()}  x64={jax.config.jax_enable_x64}")
        if not ok:
            print(f"\n!! JAX IS NOT ON A GPU -- correct but ~50x slow.\n{msg}\n")

    template = _build_template(args)
    print(f"template: {len(template[0])} epochs, bands {sorted(set(template[1]))}, "
          f"median sigma {np.median(template[2]):.3f} mag")

    model = wp.register_kilonova(FILTER_SET, redshift=REDSHIFT, dl_cm=DL_CM,
                                 n_wave=args.n_wave, band_aliases=BAND_ALIASES)
    names = list(model.parameters)
    prior = model.default_prior
    samplers = [s for s in args.samplers.split(",") if s]
    print(f"model {model.name} params {names}\nsamplers {samplers}\nsimulations {args.n_sims}\n")

    rng = np.random.default_rng(args.seed)
    ranks = {s: {p: [] for p in names} for s in samplers}
    covered = {s: {p: [] for p in names} for s in samplers}   # per level, per sim
    recov, failures, keep_corner = {s: [] for s in samplers}, [], None
    truths, fingerprints = [], []

    for i in range(args.n_sims):
        theta = {p: float(v) for p, v in prior.rescale(rng.random(len(names))).items()}
        lc = simulate(model, template, theta, rng)
        print(f"[sim {i + 1}/{args.n_sims}] truth " +
              " ".join(f"{p}={theta[p]:.4g}" for p in names), flush=True)
        truths.append(theta)
        # Fingerprint the actual data, not just the seed: sharding is only valid if every shard
        # simulated the SAME light curves, and _merge refuses to combine shards that disagree.
        fingerprints.append(float(np.sum(np.asarray(lc.magnitude, float) ** 2)))
        per_sim = {}
        for s in samplers:
            t0 = time.perf_counter()
            try:
                r = wp.fit(lc, model.name, sampler=s, space="magnitude", seed=args.seed + i,
                           **SBC_BUDGET.get(s, {}))
                for p in names:
                    d = np.asarray(r.samples[p], dtype=float)
                    ranks[s][p].append(wp.sbc_rank(d, theta[p]))
                    lo = np.percentile(d, 50 * (1 - LEVELS))
                    hi = np.percentile(d, 50 * (1 + LEVELS))
                    covered[s][p].append(((theta[p] >= lo) & (theta[p] <= hi)).astype(int))
                recov[s].append({"sim": i, "converged": r.info.get("converged"),
                                 "metrics": wp.recovery_metrics(r, theta)["_summary"]})
                per_sim[s] = np.column_stack([np.asarray(r.samples[p], float) for p in names])
                print(f"    {s:26s} ok   {time.perf_counter() - t0:6.1f}s  "
                      f"max|z|={recov[s][-1]['metrics'].get('max_abs_z', float('nan')):5.2f}  "
                      f"converged={r.info.get('converged')}")
            except Exception as exc:
                failures.append({"sim": i, "sampler": s, "error": f"{type(exc).__name__}: {exc}"})
                print(f"    {s:26s} FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:70]}")
        if i == 0 and per_sim:
            keep_corner = (theta, per_sim)

    report = _summarise(ranks, covered, recov, failures, names, samplers, args)
    report["truths"] = truths
    report["data_fingerprints"] = fingerprints
    (out / "sbc.json").write_text(json.dumps(report, indent=2, default=str))
    _plot_ranks(ranks, names, samplers, out)
    _plot_coverage(covered, names, samplers, out)
    if keep_corner:
        _plot_corner(keep_corner, names, out)
    print(f"\nwrote {out}/sbc.json and the figures")
    return 0


def _summarise(ranks, covered, recov, failures, names, samplers, args):
    import whisper_cbpf as wp

    rep = {"n_sims": args.n_sims, "params": names, "samplers": samplers,
           "n_failures": len(failures), "failures": failures[:20], "per_sampler": {}}
    print("\n" + "=" * 78)
    print(f"{'sampler':26s} {'min uniformity p':>17s} {'calibrated':>11s} {'cov68':>7s} {'cov95':>7s}")
    print("-" * 78)
    for s in samplers:
        by_p = {p: np.asarray(ranks[s][p]) for p in names if len(ranks[s][p])}
        if not by_p:
            print(f"{s:26s} {'no fits':>17s}")
            continue
        sbc = wp.sbc_ranks(by_p, n_bins=min(20, max(3, args.n_sims // 2)))
        cov = {p: np.mean(np.vstack(covered[s][p]), axis=0).tolist() for p in by_p}
        i68 = int(np.argmin(np.abs(LEVELS - 0.68)))
        i95 = int(np.argmin(np.abs(LEVELS - 0.95)))
        c68 = float(np.mean([cov[p][i68] for p in cov]))
        c95 = float(np.mean([cov[p][i95] for p in cov]))
        rep["per_sampler"][s] = {
            "sbc": {k: v for k, v in sbc.items() if k == "_summary"},
            "uniformity_p": {p: sbc[p]["uniformity_p"] for p in by_p},
            # RAW ranks and per-simulation coverage, so shards run on different cards can be
            # merged. Safe by construction: theta* and the noise are drawn per simulation BEFORE
            # the sampler loop, so one --seed gives identical datasets in every shard.
            "ranks_raw": {p: by_p[p].tolist() for p in by_p},
            "covered_raw": {p: np.vstack(covered[s][p]).tolist() for p in by_p},
            "coverage": cov, "levels": LEVELS.tolist(),
            "coverage68_mean": c68, "coverage95_mean": c95,
            # A sampler with NO convergence diagnostic (SNPE has no r-hat -- there is no chain to
            # converge) reports None. That is "not applicable", not "failed", so it is counted
            # separately: lumping None in with False made SNPE read as 0/2 converged.
            "n_converged": sum(1 for r in recov[s] if r.get("converged") is True),
            "n_diagnosed": sum(1 for r in recov[s] if r.get("converged") is not None),
            "n_fits": len(recov[s]),
            # Persist the real per-fit flags. Without these a merge has to invent them, and an
            # invented True would let a merged report claim full convergence it never measured.
            "converged_raw": [r.get("converged") for r in recov[s]],
        }
        mp = sbc["_summary"]["min_uniformity_p"]
        enough = args.n_sims >= MIN_SIMS_FOR_VERDICT
        verdict = str(sbc["_summary"]["calibrated"]) if enough else "n/a"
        rep["per_sampler"][s]["verdict_withheld"] = not enough
        nconv = rep["per_sampler"][s]["n_converged"]
        ndiag = rep["per_sampler"][s]["n_diagnosed"]
        if ndiag == 0:
            flag = "  <- no convergence diagnostic for this sampler"
        elif nconv == ndiag:
            flag = ""
        else:
            flag = f"  <- only {nconv}/{ndiag} fits converged"
        print(f"{s:26s} {mp:17.4f} {verdict:>11s} {c68:7.3f} {c95:7.3f}{flag}")
    print("-" * 78)
    if args.n_sims < MIN_SIMS_FOR_VERDICT:
        print(f"!! NO CALIBRATION VERDICT: {args.n_sims} simulations is below the "
              f"{MIN_SIMS_FOR_VERDICT} needed for the uniformity test to have power.")
        print("   At this size the test passes because it cannot fail, not because the sampler is")
        print("   calibrated. The ranks and coverage below are shown as a smoke check only.")
    print("calibrated: chi2-of-uniformity p >= 0.05 on EVERY parameter.")
    print("cov68/cov95: fraction of simulations whose truth fell inside the nominal interval;")
    print("             0.68 and 0.95 are the honest values. Lower = over-confident.")
    print("A rank from an UNCONVERGED fit measures the sampler's failure to converge, not the")
    print("posterior -- check the converged counts above before reading any verdict.")
    return rep


def _plot_ranks(ranks, names, samplers, out):
    """Rank histograms. Uniform = calibrated. U = too narrow. Inverted-U = too wide. Slope = biased."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in samplers if any(len(ranks[s][p]) for p in names)]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), len(names), squeeze=False,
                             figsize=(3.0 * len(names), 2.3 * len(rows)))
    for i, s in enumerate(rows):
        for j, p in enumerate(names):
            ax = axes[i][j]
            r = np.asarray(ranks[s][p], dtype=float)
            if not len(r):
                ax.set_axis_off(); continue
            M = max(r.max(), 1)
            nb = min(12, max(3, len(r) // 2))
            ax.hist(r / M, bins=nb, range=(0, 1), color="#0E6E5C", alpha=.75)
            ax.axhline(len(r) / nb, color="#A33A28", lw=1.2, ls="--")   # uniform expectation
            if i == 0:
                ax.set_title(p, fontsize=10)
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            ax.set_yticks([]); ax.set_xticks([0, .5, 1])
    fig.suptitle("SBC rank histograms — flat is calibrated; U = over-confident, ∩ = under-confident",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "sbc_ranks.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ranks     -> sbc_ranks.png")


def _plot_coverage(covered, names, samplers, out):
    """The P-P / coverage plot: nominal vs empirical. The diagonal is calibration."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in samplers if any(len(covered[s][p]) for p in names)]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), squeeze=False, figsize=(3.3 * len(rows), 3.3))
    for i, s in enumerate(rows):
        ax = axes[0][i]
        n = 0
        for p in names:
            if not len(covered[s][p]):
                continue
            emp = np.mean(np.vstack(covered[s][p]), axis=0)
            n = len(covered[s][p])
            ax.plot(LEVELS, emp, marker="o", ms=3, lw=1.3, label=p)
        ax.plot([0, 1], [0, 1], color="#A33A28", ls="--", lw=1.2)
        if n:   # binomial 95% band around the diagonal -- inside it is consistent with calibrated
            band = 1.96 * np.sqrt(LEVELS * (1 - LEVELS) / n)
            ax.fill_between(LEVELS, LEVELS - band, LEVELS + band, color="#A33A28", alpha=.12)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("nominal credible level")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("fraction containing the truth")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Coverage (P-P): on the diagonal = honest intervals; below = over-confident",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "sbc_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  coverage  -> sbc_coverage.png")


def _plot_corner(keep, names, out):
    """Every sampler's posterior for ONE simulated dataset, with the truth marked."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import whisper_cbpf as wp

    theta, per_sim = keep
    labels = list(per_sim)
    try:
        fig = wp.plot_corner([pd.DataFrame(per_sim[s], columns=names) for s in labels],
                             labels=labels, truths=[theta[p] for p in names],
                             title="One simulated kilonova — every sampler, truth marked")
        fig.savefig(out / "sbc_corner_sim0.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("  corner    -> sbc_corner_sim0.png")
    except Exception as exc:
        print(f"  corner failed: {type(exc).__name__}: {exc}")


def _merge(dirs, out, args):
    """Re-render ranks and coverage from several shards' raw arrays. No fitting."""
    import whisper_cbpf as wp

    out.mkdir(parents=True, exist_ok=True)
    ranks, covered, recov, names, samplers, n_sims = {}, {}, {}, [], [], 0
    failures, ref_fp, ref_dir = [], None, None
    for d in dirs:
        j = json.loads((d / "sbc.json").read_text())
        names = names or j["params"]
        n_sims = max(n_sims, j["n_sims"])
        failures.extend(j.get("failures", []))
        # The whole point of sharding is that one --seed gives every shard the same simulated data.
        # That is an ASSUMPTION about rng consumption order, so check it rather than trust it:
        # merging shards built on different light curves would silently pool unrelated ranks.
        fp = j.get("data_fingerprints")
        if fp is None:
            print(f"  [warn] {d.name}: no data fingerprint (built before this check existed); "
                  f"cannot confirm it saw the same simulations")
        elif ref_fp is None:
            ref_fp, ref_dir = fp, d.name
        else:
            n = min(len(fp), len(ref_fp))
            if not np.allclose(fp[:n], ref_fp[:n], rtol=1e-12, atol=0.0):
                bad = int(np.argmax(~np.isclose(fp[:n], ref_fp[:n], rtol=1e-12, atol=0.0)))
                raise SystemExit(
                    f"REFUSING TO MERGE: {d.name} and {ref_dir} simulated DIFFERENT data "
                    f"(first mismatch at simulation {bad}: {fp[bad]!r} vs {ref_fp[bad]!r}). "
                    f"Shards must share --seed, --n-sims and --n-wave; pooling ranks from "
                    f"different light curves would be meaningless.")
        for s_name, blk in j.get("per_sampler", {}).items():
            if "ranks_raw" not in blk or "converged_raw" not in blk:
                print(f"  [skip] {d.name}/{s_name}: no raw arrays (built before --merge existed)")
                continue
            samplers.append(s_name)
            ranks.setdefault(s_name, {p: [] for p in names})
            covered.setdefault(s_name, {p: [] for p in names})
            recov.setdefault(s_name, [])
            for p, v in blk["ranks_raw"].items():
                ranks[s_name][p].extend(v)
            for p, v in blk["covered_raw"].items():
                covered[s_name][p].extend([np.asarray(r) for r in v])
            # carry the MEASURED flags across, never a fabricated True
            recov[s_name].extend({"converged": c} for c in blk["converged_raw"])  # None stays None
    samplers = sorted(set(samplers))
    print(f"merged {len(dirs)} shard(s): {n_sims} simulations x {samplers}")
    args.n_sims = n_sims
    rep = _summarise(ranks, covered, recov, failures, names, samplers, args)
    (out / "sbc.json").write_text(json.dumps(rep, indent=2, default=str))
    _plot_ranks(ranks, names, samplers, out)
    _plot_coverage(covered, names, samplers, out)
    print(f"\nwrote {out}/sbc.json and the figures")
    return 0

if __name__ == "__main__":
    sys.exit(main())

