#!/usr/bin/env python3
"""Do the samplers agree on the ANSWER, not just on how fast they get there?

``scripts/benchmark_samplers.py`` measures wall time. It says nothing about whether two samplers
that both "worked" returned the same posterior — and a sampler that is 33x faster and subtly wrong
is worse than one that is slow. Every cell of the production benchmark reported
``accuracy: not measured``, because no reference posteriors had ever been built. This closes that.

What it does, per (model, space):

1. Loads the **reference** posterior cached by ``benchmark_samplers.py --build-references`` — a long,
   converged run that had to pass an explicit bar (for NUTS: r-hat < 1.01, zero divergences,
   ESS_worst > 2000). A reference that failed its bar is **refused**, not used: comparing against an
   unconverged reference manufactures a verdict instead of measuring one.
2. Fits the same data with every sampler that can legitimately run it.
3. Scores each against the reference with **c2st** — train a classifier to tell the two sample sets
   apart. 0.5 means indistinguishable; 1.0 means trivially separable. The honest comparison is not
   against 0.5 but against the **floor**: c2st of the reference against *itself*, split in half.
   That is what "indistinguishable" actually costs at this sample size, and it is never exactly 0.5.
4. Writes an **overlay corner plot** per (model, space) — every sampler on shared axes with the
   reference — plus a marginal-comparison figure. A number can hide a shifted mode; a corner plot
   cannot.

Usage::

    python scripts/check_sampler_agreement.py --device gpu --models kn1 --space magnitude
    python scripts/check_sampler_agreement.py --device cpu --models bazin --space flux

Outputs land in ``--out`` (default ``_scratch/agreement/``): one PNG per (model, space), a
``agreement.json`` sidecar, and a printed table.

**What a failure here means.** A high c2st is not automatically a bug in the faster sampler — ABC
with a loose tolerance is *expected* to be broader than the exact posterior, and that is a
documented property, not a defect. What this catches is the case nobody expects: a shifted mode, a
collapsed dimension, or a marginal that is confidently wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from compare_at2017gfo import (  # noqa: E402
    BAND_ALIASES, CPU_SAMPLERS, DL_CM, FILTER_SET, GPU_SAMPLERS, REDSHIFT,
    _budget, _load_lc, _register_models, impossible_reason,
)

#: c2st at or below this counts as indistinguishable. Mirrors benchmark_samplers.C2ST_PASS so the
#: two tools cannot disagree about what "agrees" means.
C2ST_PASS = 0.60


#: Where ``benchmark_samplers.py --build-references`` caches them (its OUT_ROOT / "_references").
REF_DIR = REPO / "_scratch" / "bench" / "_references"


def _reference_path(model_key, space, lc):
    """Locate the cached reference for this (model, space).

    Found by GLOB, not by recomputing the builder's hash. That hash covers the cell's RESOLVED
    space, which the builder decides internally; recomputing it here would mean guessing that
    resolution and silently reporting "no reference" whenever the guess was wrong. The filename
    carries model and space in clear text, so a glob is both simpler and harder to get subtly wrong.
    """
    hits = sorted(REF_DIR.glob(f"{model_key}__{space}__*.npz"))
    return hits[-1] if hits else REF_DIR / f"{model_key}__{space}__MISSING.npz"


def _c2st(a, b, seed=0):
    """Classifier two-sample test. 0.5 = indistinguishable, 1.0 = trivially separable.

    Uses sbi's implementation when available so this agrees with the benchmark harness; falls back
    to a plain logistic-regression CV otherwise rather than silently reporting nothing.
    """
    try:
        import torch
        from sbi.utils.metrics import c2st as _sbi_c2st
        return float(np.asarray(_sbi_c2st(torch.as_tensor(a, dtype=torch.float32),
                                          torch.as_tensor(b, dtype=torch.float32))).ravel()[0])
    except Exception:
        pass
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        n = min(len(a), len(b))
        rng = np.random.default_rng(seed)
        X = np.vstack([a[rng.choice(len(a), n, replace=False)],
                       b[rng.choice(len(b), n, replace=False)]])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="accuracy")))
    except Exception:
        return float("nan")


def _w1_max(a, b):
    """Largest per-parameter 1-Wasserstein distance, normalised by the reference's own spread."""
    try:
        from scipy.stats import wasserstein_distance
        return float(max(wasserstein_distance(a[:, j], b[:, j]) / (np.std(b[:, j]) or 1.0)
                         for j in range(a.shape[1])))
    except Exception:
        return float("nan")


def _draws(result, names):
    return np.column_stack([np.asarray(result.samples[n], dtype=float) for n in names])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--models", default="kn1", help="comma-separated: kn1,kn2,bazin,kn2_redback,...")
    ap.add_argument("--space", choices=["flux", "magnitude", "both"], default="magnitude")
    ap.add_argument("--samplers", default="", help="subset; default = every legitimate sampler")
    ap.add_argument("--quick", action="store_true", help="pilot budgets; posteriors NOT converged")
    ap.add_argument("--out", default=str(REPO / "_scratch" / "agreement"))
    args = ap.parse_args(argv)

    import whisper_cbpf as wp

    has_jax = False
    if args.device == "gpu":
        try:
            import jax
            jax.config.update("jax_enable_x64", True)   # BEFORE the first array
            has_jax = True
            ok, msg = wp.check_gpu()
            print(f"device: {jax.devices()}  x64={jax.config.jax_enable_x64}")
            if not ok:
                print(f"\n!! JAX IS NOT ON A GPU -- results correct, ~50x slow.\n{msg}\n")
        except ImportError:
            print("!! --device gpu but JAX is not importable; falling back to the CPU set.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lc = _load_lc()
    print(f"\n{lc.n_points} points, bands {sorted(set(np.asarray(lc.band)))}, "
          f"data_mode={lc.data_mode}")

    models = _register_models(has_jax, set(args.models.split(",")))
    samplers = CPU_SAMPLERS + (GPU_SAMPLERS if args.device == "gpu" else [])
    if args.samplers:
        samplers = [s for s in samplers if s in set(args.samplers.split(","))]
    spaces = ["flux", "magnitude"] if args.space == "both" else [args.space]
    budget = _budget(args.quick)

    rows, report = [], {}
    for label, name, kind in models:
        model_key = args.models.split(",")[0] if len(models) == 1 else name
        for space in spaces:
            ref_path = _reference_path(model_key, space, lc)
            if not ref_path.exists():
                print(f"\n[{label} / {space}] NO REFERENCE at {ref_path.name}")
                print("   build one first:  python scripts/benchmark_samplers.py --build-references"
                      f" --device {args.device}")
                continue
            ref = np.load(ref_path, allow_pickle=True)
            ref_draws = np.asarray(ref["draws"], dtype=float)
            ref_names = [str(x) for x in ref["params"]]      # the archive calls them `params`
            qualifies = bool(np.asarray(ref["qualifies"]).ravel()[0]) if "qualifies" in ref else True
            print(f"\n=== {label} / {space} ===")
            print(f"  reference: {ref_path.name}  {ref_draws.shape[0]} draws  qualifies={qualifies}")
            if not qualifies:
                print("  REFUSED: this reference did not meet its convergence bar. Comparing "
                      "against it would manufacture a verdict, not measure one.")
                continue

            # The floor: the reference against ITSELF. c2st's power grows with sample size, so the
            # floor MUST be computed at the same sizes the real comparison uses, or it is not a
            # floor. The first version split the reference in half (8000 vs 8000) and compared
            # samplers at 4000 vs 16000 -- a more powerful test against a weaker baseline, which
            # made every sampler look like it disagreed. `_floor_at` is called per sampler with
            # that sampler's own draw count.
            def _floor_at(n_a, n_b, seed=0):
                rng = np.random.default_rng(seed)
                idx = rng.permutation(len(ref_draws))
                a_i, b_i = idx[:n_a], idx[n_a:n_a + n_b]
                if len(b_i) < n_b:                      # not enough draws to split disjointly
                    return float("nan")
                return _c2st(ref_draws[a_i], ref_draws[b_i])

            # A posterior pinned at a prior corner cannot discriminate anything: the normalised W1
            # denominator is the reference's own spread, and a railed parameter has none.
            ref_std = ref_draws.std(axis=0)
            railed = [ref_names[j] for j in range(len(ref_names))
                      if ref_std[j] < 1e-4 * (abs(ref_draws[:, j].mean()) or 1.0)]
            if railed:
                print(f"  !! REFERENCE IS RAILED on {railed} (std/|mean| < 1e-4).")
                print("     Its posterior is effectively a delta at a prior corner, so normalised")
                print("     W1 divides by ~0 and c2st asks only 'did you collapse to the same")
                print("     corner'. This measures misspecification, not sampler agreement.")
                print("     Use --synthetic, or a model that actually fits the data.")

            posteriors, plot_labels = [ref_draws], ["reference"]
            for s in samplers:
                why = impossible_reason(kind, s, space, has_jax)
                if why:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = wp.fit(lc, name, sampler=s, space=space, seed=0, **budget.get(s, {}))
                    d = _draws(r, ref_names)
                    c, w = _c2st(d, ref_draws), _w1_max(d, ref_draws)
                    floor = _floor_at(len(d), min(len(d), len(ref_draws) - len(d)))
                    # Read against the SIZE-MATCHED floor, not against 0.5.
                    thresh = max(C2ST_PASS, (floor + 0.05) if np.isfinite(floor) else C2ST_PASS)
                    verdict = "agrees" if c <= thresh else "DIFFERS"
                    rows.append(dict(model=label, space=space, sampler=s, n=len(d),
                                     c2st=c, c2st_floor=floor, w1_max=w, verdict=verdict,
                                     ess=(r.info.get("predictive_metrics") or {}) and None,
                                     aic=r.aic))
                    posteriors.append(d)
                    plot_labels.append(s)
                    print(f"  {s:26s} c2st={c:.3f} (floor {floor:.3f})  w1_max={w:9.3f}  {verdict}")
                except Exception as exc:
                    print(f"  {s:26s} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")

            if len(posteriors) > 1:
                _plot(posteriors, plot_labels, ref_names, label, space, out)
                report[f"{label}|{space}"] = dict(
                    reference=ref_path.name, n_reference=int(ref_draws.shape[0]),
                    c2st_floor=floor, rows=[r for r in rows if r["model"] == label
                                            and r["space"] == space])

    (out / "agreement.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out / 'agreement.json'}")
    if rows:
        bad = [r for r in rows if r["verdict"] != "agrees"]
        print(f"\n{len(rows) - len(bad)}/{len(rows)} sampler-cells agree with the reference.")
        for r in bad:
            print(f"  DIFFERS: {r['model']} / {r['space']} / {r['sampler']} "
                  f"c2st={r['c2st']:.3f} (floor {r['c2st_floor']:.3f}) w1_max={r['w1_max']:.3f}")
    return 0


def _plot(posteriors, labels, names, model_label, space, out):
    """Overlay corner plot + per-parameter marginals. The picture the numbers cannot replace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import whisper_cbpf as wp
    import pandas as pd

    stem = f"{model_label.replace(' ', '_').replace('/', '-')}__{space}"

    # 1. the overlay corner -- plot_corner exists precisely to put posteriors on shared axes
    try:
        frames = [pd.DataFrame(p, columns=names) for p in posteriors]
        fig = wp.plot_corner(frames, labels=labels,
                             title=f"{model_label} — {space} space — every sampler vs reference")
        fig.savefig(out / f"{stem}__corner.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  corner  -> {stem}__corner.png")
    except Exception as exc:
        print(f"  corner plot failed: {type(exc).__name__}: {exc}")

    # 2. marginals, which make a shifted mode obvious at a glance even when the corner is dense
    try:
        k = len(names)
        fig, axes = plt.subplots(1, k, figsize=(3.1 * k, 2.8))
        axes = np.atleast_1d(axes)
        for j, nm in enumerate(names):
            ax = axes[j]
            for p, lab in zip(posteriors, labels):
                ax.hist(p[:, j], bins=45, density=True, histtype="step", lw=2.0 if lab == "reference" else 1.2,
                        label=lab, alpha=0.95 if lab == "reference" else 0.8)
            ax.set_xlabel(nm)
            ax.set_yticks([])
        axes[0].set_ylabel("density")
        axes[-1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{model_label} — {space} — marginals against the reference", y=1.04)
        fig.tight_layout()
        fig.savefig(out / f"{stem}__marginals.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  marginals -> {stem}__marginals.png")
    except Exception as exc:
        print(f"  marginal plot failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
