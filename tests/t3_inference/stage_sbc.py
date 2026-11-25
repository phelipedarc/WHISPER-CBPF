"""STAGED JOB (T3, sampler half) -- Simulation-Based Calibration of the TDE gaussianrise
model through whisper's own ``nuts_gpu`` sampler. NOT a test; nothing here runs under
pytest. This file is the runnable artefact the CPU tier stages for the GPU tier.

WHAT IT CHECKS. SBC (Talts et al. 2018): draw theta* from the prior, simulate a dataset,
sample the posterior, record the rank of each theta*_i among L thinned posterior draws.
If the sampler samples the posterior the model+prior imply, ranks are uniform on
{0..L}; the rank-ECDF check at the end quantifies the departure per parameter. This is
the one check that catches a WRONG posterior (biased sampler, broken likelihood
gradient, prior/parameterisation mismatch) rather than a merely noisy one.

RUNS ON GPU, with the env sourced and CUDA_VISIBLE_DEVICES set. From the repo root, with the
``[gpu,models,dev]`` extras installed::

    export CUDA_VISIBLE_DEVICES=0
    source "$(whisper-cbpf-env)"
    python -u tests/t3_inference/stage_sbc.py --n-replicates 300

The script refuses to start on CPU unless ``--allow-cpu`` is passed. **It is, however,
perfectly runnable on CPU, and that is how the numbers in REPORT.md §4 were produced.**
The 500-step scan is launch-bound on a GPU and has no launches to pay for on a CPU, so
the measured penalty is ~4x per leapfrog rather than the ~100x ``require_gpu`` warns
about -- and a many-core host runs a dozen replicates side by side. Shard like this
(one outdir per worker; ``--analyse-only`` globs them all back together)::

    for i in 0 1 2 ... ; do
      taskset -c $((4*i))-$((4*i+3)) python -u .../stage_sbc.py --allow-cpu \
        --n-replicates 4 --replicate-offset $((4*i)) \
        --num-warmup 400 --num-samples 600 --num-chains 2 --rank-bins 99 \
        --outdir .../sbc_results/shard$i &
    done; wait

Do NOT let two workers share an ``--outdir``: 13 processes appending to one ``ranks.csv``
interleave at the byte level and corrupt rows.

EXPECTED WALL-CLOCK, 1x A6000, defaults (300 replicates, 500 warmup + 1000 samples,
4 vectorized chains, n_time=500): ~15-25 min per replicate -- the engine is a 500-step
sequential scan, so a leapfrog step costs ~20-25 ms (launch-bound, forward+VJP) and an
average NUTS tree spends ~32 of them -- plus ~1 min of per-replicate XLA compile (each
replicate builds a fresh NumPyro MCMC). That is **~80-130 h for the full 300**. It is
embarrassingly parallel across replicates: shard with ``--replicate-offset``/
``--n-replicates`` over GPUs (e.g. 4 x 75 replicates ~= a day per GPU). Results append
to ``ranks.csv`` per replicate, so a killed run resumes with ``--replicate-offset`` and
partial runs are analysable at any time with ``--analyse-only``.

The prior is redback's shipped gaussianrise prior, UNTRUNCATED -- including the ~5% of
draws whose envelope never lives (data = noise around mag_floor) and the ~40% in the
documented absurd-bright-rise regime (T3.1 measured both). SBC remains valid over the
full prior; slice the saved per-replicate diagnostics (constraint, rise_sane columns)
to study the calibrated subsets separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

#: measured/estimated per-replicate cost model on 1x A6000, n_time=500, 4 vectorized
#: chains. Used ONLY for the startup estimate; the loop reports true times as it goes.
EST_SEC_PER_LEAPFROG = 0.022
EST_TREE_LEAPFROGS = 32
EST_COMPILE_S = 60.0
#: ... and the MEASURED CPU cost model (JAX_PLATFORMS=cpu, n_time=500, 2
#: vectorized chains): a 500-step lax.scan has no kernel launches to pay for, so the CPU is
#: only ~4x slower per leapfrog than the A6000 rather than the 100x `require_gpu` warns
#: about -- and a 96-core host runs ~24 replicates side by side. That is what makes a
#: reduced SBC possible without a GPU. See REPORT.md for what was actually run.
EST_CPU_SEC_PER_LEAPFROG = 0.010
EST_CPU_COMPILE_S = 10.0

DEFAULT_OUT = Path(__file__).resolve().parent / "sbc_results"


def _load_rank_rows(outdir):
    """Every ``ranks.csv`` under ``outdir``, concatenated.

    Sharding across processes writes one CSV per shard (``outdir/shardNN/ranks.csv``)
    rather than appending to a shared file: concurrent appends from 24 workers interleave
    at the byte level and silently corrupt rows. A single un-sharded run still writes
    ``outdir/ranks.csv`` and is picked up by the same glob.
    """
    paths = sorted(set(list(outdir.glob("ranks.csv")) + list(outdir.glob("*/ranks.csv"))))
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    return rows, paths


def rank_ecdf_check(ranks, n_bins, alpha=0.05):
    """Per-parameter rank-uniformity: KS test + worst pointwise ECDF deviation.

    ranks: (n_replicates, n_params) integer ranks in {0..n_bins}. Returns a dict per
    parameter with the KS p-value and the max |ECDF - uniform| in units of the
    binomial-envelope width; p < alpha/n_params (Bonferroni) fails.
    """
    from scipy import stats

    n_rep, n_par = ranks.shape
    out = {}
    for j in range(n_par):
        u = (ranks[:, j] + 0.5) / (n_bins + 1)
        ks = stats.kstest(u, "uniform")
        grid = np.linspace(0.0, 1.0, 201)
        ecdf = np.searchsorted(np.sort(u), grid, side="right") / n_rep
        env = np.sqrt(grid * (1 - grid) / n_rep)
        worst = float(np.max(np.abs(ecdf - grid) / np.maximum(env, 1e-12)))
        out[j] = dict(ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
                      worst_env_units=worst)
    return out


def analyse(outdir, names, alpha=0.05):
    rows, paths = _load_rank_rows(outdir)
    if not rows:
        raise SystemExit(f"no replicates found under {outdir} (looked for ranks.csv)")
    n_bins_all = {int(r["n_bins"]) for r in rows}
    if len(n_bins_all) > 1:
        raise SystemExit(
            f"replicates were run with DIFFERENT posterior-draw counts {sorted(n_bins_all)}: "
            f"ranks from different L are not on the same scale and must not be pooled")
    n_bins = n_bins_all.pop()
    ranks = np.array([[int(r[f"rank_{nm}"]) for nm in names] for r in rows])
    div = np.array([int(r["n_divergences"]) for r in rows])
    rhat = np.array([float(r["max_rhat"]) if r["max_rhat"] not in ("", "None") else np.nan
                     for r in rows])
    print(f"analysing {len(rows)} replicates from {len(paths)} shard file(s) "
          f"(L = {n_bins} posterior bins)")
    print(f"  replicates with divergences: {(div > 0).sum()} "
          f"(divergent replicates bias SBC; investigate before trusting the ranks)")
    with np.errstate(invalid="ignore"):
        print(f"  replicates with max r-hat > 1.01: "
              f"{int(np.nansum(rhat > 1.01))}/{int(np.isfinite(rhat).sum())} "
              f"(median r-hat {np.nanmedian(rhat):.3f}, max {np.nanmax(rhat):.3f}) -- an "
              f"unconverged replicate contributes a rank that means nothing")
    res = rank_ecdf_check(ranks, n_bins, alpha)
    bonf = alpha / len(names)
    ok = True
    for j, nm in enumerate(names):
        r = res[j]
        verdict = "ok" if r["ks_pvalue"] >= bonf else "FAIL"
        ok &= r["ks_pvalue"] >= bonf
        print(f"  {nm:>13}: KS p = {r['ks_pvalue']:.4f}  worst ECDF dev = "
              f"{r['worst_env_units']:.2f} envelope units  [{verdict}]")
    _plot(outdir, names, ranks, n_bins)
    print(f"rank-ECDF {'PASS' if ok else 'FAIL'} at Bonferroni alpha = {alpha}")
    return ok


def _plot(outdir, names, ranks, n_bins):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt

    n_rep = ranks.shape[0]
    fig, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 2.6), sharey=True)
    grid = np.linspace(0, 1, 201)
    env = 1.96 * np.sqrt(grid * (1 - grid) / n_rep)
    for j, (nm, ax) in enumerate(zip(names, np.atleast_1d(axes))):
        u = np.sort((ranks[:, j] + 0.5) / (n_bins + 1))
        ecdf = np.searchsorted(u, grid, side="right") / n_rep
        ax.fill_between(grid, -env, env, color="#cccccc", alpha=0.6, lw=0)
        ax.plot(grid, ecdf - grid, lw=1.0, color="#1f5fa8")
        ax.set_title(nm, fontsize=8)
        ax.set_xlabel("rank quantile", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("ECDF - uniform")
    fig.tight_layout()
    fig.savefig(outdir / "sbc_rank_ecdf.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-replicates", type=int, default=300)
    ap.add_argument("--replicate-offset", type=int, default=0,
                    help="first replicate index (for sharding across GPUs / resuming)")
    ap.add_argument("--num-warmup", type=int, default=500)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-chains", type=int, default=4)
    ap.add_argument("--n-time", type=int, default=500)
    ap.add_argument("--rank-bins", type=int, default=1023,
                    help="L: posterior draws kept per replicate, thinned uniformly. SBC "
                         "ranks are uniform on {0..L} only if the kept draws are close to "
                         "independent, so L must be at or below the chain's effective "
                         "sample size -- 1023 assumes a long, well-mixed GPU run; a short "
                         "CPU run should pass something like 99.")
    ap.add_argument("--seed", type=int, default=8123)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--analyse-only", action="store_true",
                    help="skip sampling; run the rank-ECDF check on outdir/ranks.csv")
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
    est = args.n_replicates * (n_steps * EST_TREE_LEAPFROGS * per_lf + comp)
    print(f"SBC: {args.n_replicates} replicates x ({args.num_warmup}+{args.num_samples} "
          f"steps, {args.num_chains} vectorized chains) on backend={backend!r}")
    print(f"ESTIMATED WALL-CLOCK ({tag} cost model): {est / 3600:.1f} h "
          f"({est / max(args.n_replicates, 1) / 60:.1f} min/replicate). "
          f"Shard with --replicate-offset + a per-shard --outdir if that is unacceptable "
          f"(--analyse-only globs every ranks.csv under the outdir).", flush=True)

    import jax.numpy as jnp

    from whisper_cbpf.models.jax import tde as T

    model, mags_fn, t_obs, bands, names, sample_theta = S.build_context(args.n_time)
    csv_path = args.outdir / "ranks.csv"
    new_file = not csv_path.exists()
    fields = (["replicate", "n_bins", "n_divergences", "max_rhat", "constraint",
               "rise_sane", "wall_s"]
              + [f"truth_{nm}" for nm in names] + [f"rank_{nm}" for nm in names])
    (args.outdir / "config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))

    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for r in range(args.replicate_offset, args.replicate_offset + args.n_replicates):
            rng = np.random.default_rng(args.seed + 1000 * r)
            theta = sample_theta(rng)
            lc, mag_obs = S.simulate_lightcurve(mags_fn, theta, t_obs, bands, rng)
            log_prob = S.make_log_prob(mags_fn, mag_obs)
            t0 = time.perf_counter()
            res = S.fit_one(lc, model, log_prob,
                            num_warmup=args.num_warmup, num_samples=args.num_samples,
                            num_chains=args.num_chains, seed=args.seed + r)
            wall = time.perf_counter() - t0

            post = res.samples[names].to_numpy()
            # thin to at most --rank-bins so ties/autocorrelation do not fake uniformity
            n_bins = min(args.rank_bins, post.shape[0])
            take = np.linspace(0, post.shape[0] - 1, n_bins).astype(int)
            ranks = (post[take] < theta[None, :]).sum(axis=0)

            eng = T.cooling_envelope(*theta[2:], n_time=args.n_time)
            row = dict(replicate=r, n_bins=n_bins,
                       n_divergences=res.info.get("n_divergences", -1),
                       max_rhat=res.info.get("max_rhat"),
                       constraint=int(eng["constraint"]),
                       rise_sane=bool(T.rise_peaks_near_fallback(*theta[:4])),
                       wall_s=round(wall, 2))
            row.update({f"truth_{nm}": theta[j] for j, nm in enumerate(names)})
            row.update({f"rank_{nm}": int(ranks[j]) for j, nm in enumerate(names)})
            writer.writerow(row)
            fh.flush()
            print(f"  replicate {r}: {wall / 60:.1f} min, "
                  f"div={row['n_divergences']}, rhat={row['max_rhat']}", flush=True)

    analyse(args.outdir, names)


if __name__ == "__main__":
    main()
