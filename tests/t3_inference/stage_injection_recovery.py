"""STAGED JOB (T3, sampler half) -- injection-recovery for the TDE gaussianrise model
through whisper's own ``nuts_gpu`` sampler. NOT a test; the runnable artefact the CPU
tier stages for the GPU tier. Companion to stage_sbc.py -- same model context, same
likelihood, same container contract.

WHAT IT CHECKS. 100 injections drawn from the prior, each fitted end to end; for every
parameter the empirical coverage of the central 50% and 90% credible intervals is
compared to nominal with a two-sided binomial test. Where SBC (the companion) detects
any distortion of the posterior SHAPE, this answers the blunter operational question --
"do my reported intervals contain the truth as often as they claim?" -- at a third of
the sampler cost, and its per-injection table doubles as a recovery atlas (which corners
of the prior the fit loses).

RUNS ON GPU, with the env sourced and CUDA_VISIBLE_DEVICES set. From the repo root, with the
``[gpu,models,dev]`` extras installed::

    export CUDA_VISIBLE_DEVICES=0
    source "$(whisper-cbpf-env)"
    python -u tests/t3_inference/stage_injection_recovery.py --n-injections 100

The script refuses to start on CPU unless ``--allow-cpu`` is passed; see stage_sbc.py's
docstring for why CPU is nonetheless a reasonable place to run this, and shard the same
way (one ``--outdir`` per worker). REPORT.md §4 records what was actually run.

``--sane-only`` restricts the INJECTED TRUTH to the region the model's own predicates
accept -- see :func:`sane_theta`. It does not touch the prior the sampler is given.

EXPECTED WALL-CLOCK, 1x A6000, defaults (100 injections, 500 warmup + 1000 samples, 4
vectorized chains, n_time=500): ~15-25 min per injection plus ~1 min compile each,
i.e. **~27-45 h total** (same per-fit cost model as stage_sbc.py: ~20-25 ms per
leapfrog through the 500-step scan, ~32 leapfrogs per NUTS tree). Shard across GPUs
with ``--injection-offset``/``--n-injections``; results append to ``coverage.csv`` so
partial runs are analysable at any time with ``--analyse-only``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

EST_SEC_PER_LEAPFROG = 0.022
EST_TREE_LEAPFROGS = 32
EST_COMPILE_S = 60.0
#: measured CPU cost model -- see the same constants in stage_sbc.py for why CPU is viable
EST_CPU_SEC_PER_LEAPFROG = 0.010
EST_CPU_COMPILE_S = 10.0
LEVELS = (0.50, 0.90)

DEFAULT_OUT = Path(__file__).resolve().parent / "injection_results"


def sane_theta(sample_theta, rng, n_time, t_max_days, z, day, max_tries=200_000):
    """Draw from the prior until the MODEL'S OWN predicates accept the draw.

    WHAT THIS IS, AND WHAT IT IS NOT. It restricts the INJECTED TRUTH, not the prior the
    sampler is given -- that stays redback's, unmodified, by project rule. So the coverage
    measured with ``--sane-only`` answers "does the fit recover a TDE this model can
    actually produce over this observing window", which is the operational question, and
    NOT "is the posterior calibrated over the shipped prior", which is SBC's job and needs
    truths drawn from exactly the prior the sampler uses.

    The three conditions are the model's own, not invented here:

    * ``envelope_exists``  -- the envelope is born outside R_circ/2 (tde.py, closed form);
    * ``rise_peaks_near_fallback`` -- the Gaussian rise peaks within 3 sigma of the stitch,
      i.e. the draw is outside the documented absurd-bright-rise regime;
    * the light curve's own span reaches the last epoch, which is what tde.py's
      ``_interp_photosphere`` instructs ("keep observations inside the envelope's own
      span") and what T3.1 measured 40 % of the prior failing for a 180-day campaign.
    """
    import numpy as np

    from whisper_cbpf.models.jax import tde as T

    for tries in range(1, max_tries + 1):
        th = sample_theta(rng)
        if not bool(T.envelope_exists(th[2], th[3], th[6])):
            continue
        if not bool(T.rise_peaks_near_fallback(th[0], th[1], th[2], th[3])):
            continue
        out = T.cooling_envelope(th[2], th[3], th[4], th[5], th[6], n_time=n_time)
        k = int(out["constraint"])
        if k < 2:
            continue
        span = float(np.asarray(out["time_temp"])[k - 1]) * (1.0 + z) / day
        if span < t_max_days:
            continue
        return th, k, span, tries
    raise SystemExit(f"no draw satisfied the model's own validity predicates in "
                     f"{max_tries} tries -- report the acceptance rate, do not loosen them")


def _load_rows(outdir):
    """Every ``coverage.csv`` under ``outdir`` (one per shard), concatenated.

    Same reason as stage_sbc._load_rank_rows: parallel workers must not append to one file.
    """
    paths = sorted(set(list(outdir.glob("coverage.csv"))
                       + list(outdir.glob("*/coverage.csv"))))
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    return rows, paths


def analyse(outdir, names, alpha=0.05):
    from scipy import stats

    rows, paths = _load_rows(outdir)
    if not rows:
        raise SystemExit(f"no injections found under {outdir} (looked for coverage.csv)")
    n = len(rows)
    div = np.array([int(r["n_divergences"]) for r in rows])
    rhat = np.array([float(r["max_rhat"]) if r["max_rhat"] not in ("", "None") else np.nan
                     for r in rows])
    print(f"analysing {n} injections from {len(paths)} shard file(s); "
          f"{int((div > 0).sum())} had divergences, "
          f"{int(np.nansum(rhat > 1.01))} had max r-hat > 1.01")
    bonf = alpha / (len(names) * len(LEVELS))
    ok = True
    table = {}
    for nm in names:
        for lev in LEVELS:
            hits = np.array([r[f"in{int(lev * 100)}_{nm}"] == "True" for r in rows])
            k = int(hits.sum())
            p = float(stats.binomtest(k, n, lev).pvalue)
            verdict = "ok" if p >= bonf else "FAIL"
            ok &= p >= bonf
            table[(nm, lev)] = (k / n, p)
            print(f"  {nm:>13} {int(lev * 100)}%: coverage {k}/{n} = {k / n:.3f} "
                  f"(binomial p = {p:.4f}) [{verdict}]")
    # the single-injection statement the T3 report has to be able to make
    per_inj = [(r["injection"], sum(r[f"in90_{nm}"] == "True" for nm in names))
               for r in rows]
    best = max(per_inj, key=lambda x: x[1])
    print(f"  best single injection: #{best[0]} recovered {best[1]}/{len(names)} truths "
          f"inside their 90% credible intervals")
    _plot(outdir, names, table, n)
    print(f"injection-recovery coverage {'PASS' if ok else 'FAIL'} "
          f"at Bonferroni alpha = {alpha}")
    return ok


def _plot(outdir, names, table, n):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    xs = np.arange(len(names))
    for lev, color, dx in zip(LEVELS, ("#1f5fa8", "#c23b22"), (-0.12, 0.12)):
        cov = [table[(nm, lev)][0] for nm in names]
        err = 1.96 * np.sqrt(lev * (1 - lev) / n)
        ax.errorbar(xs + dx, cov, yerr=err, fmt="o", ms=4, color=color,
                    label=f"{int(lev * 100)}% interval", capsize=3)
        ax.axhline(lev, color=color, lw=0.8, ls="--")
    ax.set_xticks(xs, names, rotation=30, fontsize=8)
    ax.set_ylabel("empirical coverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "injection_coverage.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-injections", type=int, default=100)
    ap.add_argument("--injection-offset", type=int, default=0)
    ap.add_argument("--num-warmup", type=int, default=500)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--n-time", type=int, default=500)
    ap.add_argument("--seed", type=int, default=9291)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--sane-only", action="store_true",
                    help="draw the injected TRUTH from the subset of the prior the model's "
                         "own predicates accept (see sane_theta). The sampler's prior is "
                         "unchanged; this makes the run a recovery test on producible "
                         "TDEs, not a calibration test over the whole prior.")
    args = ap.parse_args(argv)

    import _stage_common as S                      # flips x64 before any jax array

    names = ["peak_time", "sigma_t", "mbh_6", "stellar_mass", "eta", "alpha", "beta"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.analyse_only:
        sys.exit(0 if analyse(args.outdir, names) else 1)

    backend = S.require_gpu(args.allow_cpu)
    n_steps = args.num_warmup + args.num_samples
    per_lf, comp, tag = ((EST_SEC_PER_LEAPFROG, EST_COMPILE_S, "1x A6000") if backend == "gpu"
                         else (EST_CPU_SEC_PER_LEAPFROG, EST_CPU_COMPILE_S, "CPU, measured"))
    est = args.n_injections * (n_steps * EST_TREE_LEAPFROGS * per_lf + comp)
    print(f"injection-recovery: {args.n_injections} injections x ({args.num_warmup}+"
          f"{args.num_samples} steps, {args.num_chains} chains) on backend={backend!r}")
    print(f"ESTIMATED WALL-CLOCK ({tag} cost model): {est / 3600:.1f} h "
          f"({est / max(args.n_injections, 1) / 60:.1f} min/injection). Shard with "
          f"--injection-offset + a per-shard --outdir; --analyse-only globs every "
          f"coverage.csv under the outdir.", flush=True)

    from whisper_cbpf.models.jax import tde as T

    model, mags_fn, t_obs, bands, names, sample_theta = S.build_context(args.n_time)
    csv_path = args.outdir / "coverage.csv"
    new_file = not csv_path.exists()
    fields = (["injection", "n_divergences", "max_rhat", "constraint", "rise_sane",
               "span_days", "n_tries", "wall_s"]
              + [f"truth_{nm}" for nm in names]
              + [f"in{int(l * 100)}_{nm}" for l in LEVELS for nm in names])
    (args.outdir / "config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))

    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for r in range(args.injection_offset, args.injection_offset + args.n_injections):
            rng = np.random.default_rng(args.seed + 1000 * r)
            n_tries, span = 1, float("nan")
            if args.sane_only:
                theta, _k, span, n_tries = sane_theta(
                    sample_theta, rng, args.n_time, float(t_obs.max()), S.Z, S.DAY)
            else:
                theta = sample_theta(rng)
            lc, mag_obs = S.simulate_lightcurve(mags_fn, theta, t_obs, bands, rng)
            log_prob = S.make_log_prob(mags_fn, mag_obs)
            t0 = time.perf_counter()
            res = S.fit_one(lc, model, log_prob,
                            num_warmup=args.num_warmup, num_samples=args.num_samples,
                            num_chains=args.num_chains, seed=args.seed + r,
                            target_accept_prob=args.target_accept)
            wall = time.perf_counter() - t0

            post = res.samples[names].to_numpy()
            eng = T.cooling_envelope(*theta[2:], n_time=args.n_time)
            row = dict(injection=r,
                       n_divergences=res.info.get("n_divergences", -1),
                       max_rhat=res.info.get("max_rhat"),
                       constraint=int(eng["constraint"]),
                       rise_sane=bool(T.rise_peaks_near_fallback(*theta[:4])),
                       span_days=round(span, 2) if span == span else "",
                       n_tries=n_tries,
                       wall_s=round(wall, 2))
            row.update({f"truth_{nm}": theta[j] for j, nm in enumerate(names)})
            for lev in LEVELS:
                q = (0.5 - lev / 2, 0.5 + lev / 2)
                lo = np.quantile(post, q[0], axis=0)
                hi = np.quantile(post, q[1], axis=0)
                for j, nm in enumerate(names):
                    row[f"in{int(lev * 100)}_{nm}"] = bool(lo[j] <= theta[j] <= hi[j])
            writer.writerow(row)
            fh.flush()
            print(f"  injection {r}: {wall / 60:.1f} min, "
                  f"div={row['n_divergences']}, rhat={row['max_rhat']}", flush=True)

    analyse(args.outdir, names)


if __name__ == "__main__":
    main()
