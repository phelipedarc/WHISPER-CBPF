#!/usr/bin/env python3
"""End-to-end Bayesian model comparison on AT2017GFO — the acceptance test for this package.

Fits a TDE, a supernova, and one- and two-component kilonovae (plus the analytic toys) to
``tests/data/at2017gfo.csv``, on CPU or GPU, in flux space or apparent-magnitude space, with every
sampler that can legitimately run each combination, and prints one table of AIC / BIC / WAIC.

Combinations that **cannot** work are not skipped silently: each prints the reason it cannot work,
so the table is a complete statement about the matrix rather than a list of what happened to run.

    # the minimal install: analytic models + the redback kilonova, CPU samplers
    python scripts/compare_at2017gfo.py --device cpu --space both

    # the full matrix, needs [gpu] and the environment bootstrap sourced FIRST
    source scripts/whisper-gpu-env.sh
    python scripts/compare_at2017gfo.py --device gpu --space both

    # a fast smoke of the whole matrix (seconds per cell, posteriors NOT converged)
    python scripts/compare_at2017gfo.py --device gpu --quick

Read ``docs/CHOOSING.md`` before interpreting the output, and note two things the table cannot say
for you:

* **AIC/BIC/WAIC compare MODELS, not SPACES.** A flux-space fit and a magnitude-space fit of the
  same model have different likelihoods over different data representations, so their information
  criteria are not on one scale. Compare down a column, never across.
* **A WAIC flagged unreliable is not a number.** ``p_waic`` above ``n_data/2`` means the posterior
  contains draws whose pointwise log-likelihood varies by orders of magnitude — an under-converged
  chain or a badly misspecified model. The table marks those with ``!``.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "tests" / "data" / "at2017gfo.csv"

#: AT2017GFO, Abbott+2017. Redshift and distance are the caller's choice everywhere in this
#: package -- a guessed distance silently rescales every fitted mass -- so they are stated here.
REDSHIFT = 0.0098
DL_CM = 1.23e26                       # ~40 Mpc
T_EXPLOSION_MJD = 57982.529           # GW170817 merger

#: The light curve's own labels -> the sncosmo filter names ``make_filter_set`` needs. Stated, not
#: guessed: this file carries both ``i`` and 30 Johnson ``I`` points, which any case-folding alias
#: table would merge into one filter.
_ALL_ALIASES = {"g": "sdssg", "r": "sdssr", "i": "sdssi", "z": "sdssz"}
FIT_BANDS = ["g", "r", "i"]
#: Only the bands we actually fit. Aliasing a band the filter set was not built for is refused by
#: `_make_band_indices`, which is the correct behaviour and which this script tripped over first.
BAND_ALIASES = {b: _ALL_ALIASES[b] for b in FIT_BANDS}
FILTER_SET = [BAND_ALIASES[b] for b in FIT_BANDS]


def _budget(quick):
    """Sampler budgets. ``--quick`` trades convergence for a matrix you can run in minutes."""
    if quick:
        return {
            "abc": dict(n_simulations=400, quantile=0.1, n_jobs=8),
            "abc_smc": dict(n_particles=48, n_rounds=2, n_jobs=8),
            "mcmc": dict(nsteps=300, burnin=100, nwalkers=24),
            "snpe": dict(num_rounds=1, num_simulations=300, num_samples=500),
            "abc_gpu": dict(n_simulations=512, quantile=0.05),
            "abc_smc_gpu": dict(n_particles=48, n_rounds=2),
            "nuts_gpu": dict(num_warmup=150, num_samples=150, num_chains=2),
            "emcee_jax": dict(nwalkers=16, nsteps=400, burnin=150, thin=5),
            "pymc_jax_gpu_vectorized": dict(num_warmup=150, num_samples=150, num_chains=2),
            "pymc_jax_gpu_parallelized": dict(num_warmup=150, num_samples=150, num_chains=2),
        }
    return {
        "abc": dict(n_simulations=20000, quantile=0.01, n_jobs=8),
        "abc_smc": dict(n_particles=400, n_rounds=5, n_jobs=8),
        "mcmc": dict(nsteps=4000, burnin=1000, nwalkers=32),
        "snpe": dict(num_rounds=2, num_simulations=2000, num_samples=4000),
        "abc_gpu": dict(n_simulations=50000, quantile=0.01),
        "abc_smc_gpu": dict(n_particles=400, n_rounds=5),
        "nuts_gpu": dict(num_warmup=1000, num_samples=1000, num_chains=4),
        "emcee_jax": dict(nwalkers=32, nsteps=4000, burnin=1000, thin=10),
        "pymc_jax_gpu_vectorized": dict(num_warmup=1000, num_samples=1000, num_chains=4),
        "pymc_jax_gpu_parallelized": dict(num_warmup=1000, num_samples=1000, num_chains=2),
    }


CPU_SAMPLERS = ["abc", "abc_smc", "mcmc", "snpe"]
GPU_SAMPLERS = ["abc_gpu", "abc_smc_gpu", "nuts_gpu", "emcee_jax",
                "pymc_jax_gpu_vectorized", "pymc_jax_gpu_parallelized"]


def impossible_reason(model_kind, sampler, space, has_jax):
    """Why this cell cannot work, or ``None`` if it can. Every string here is a documented fact.

    Kept as one function so the script, ``docs/CHOOSING.md`` and ``DEPLOYMENT_AUDIT.md`` cannot
    drift: this is the executable copy of CHOOSING's "which combinations are impossible" table.
    """
    numpy_only = model_kind in ("toy", "redback")
    if sampler in GPU_SAMPLERS and not has_jax:
        return "needs the [gpu] extra (JAX is not importable in this interpreter)"
    if numpy_only and sampler in ("nuts_gpu", "pymc_jax_gpu_vectorized", "pymc_jax_gpu_parallelized"):
        return "NUTS needs gradients; a numpy simulator has none and jax.grad cannot differentiate it"
    if numpy_only and sampler in ("abc_gpu", "abc_smc_gpu", "emcee_jax"):
        return "the JAX samplers need a batched predict_jax; a numpy model has none"
    if model_kind in ("tde", "supernova") and sampler in ("abc_gpu", "abc_smc_gpu"):
        return ("float64 is mandatory for this family, and batched simulation in float64 costs "
                "~57x (docs/GPU_SETUP.md) -- runnable but not a defensible use of a GPU budget")
    return None


def _load_lc():
    import whisper_cbpf as wp

    lc = wp.load_lightcurve(CSV, redshift=REDSHIFT)
    # ORDER MATTERS. set_explosion_date first: select_time_window works on whatever `time`
    # currently holds, so windowing on [0.4, 12] before the shift compares days-since-explosion
    # bounds against raw MJD (~57983) and silently returns an EMPTY light curve -- which every
    # sampler then fits without complaint, reporting AIC = 2k and BIC = -inf.
    lc = lc.select_bands(FIT_BANDS).set_explosion_date(T_EXPLOSION_MJD)
    lc = lc.select_time_window(0.4, 12.0)
    if lc.n_points == 0:
        raise SystemExit("the selection left no data points; check the band list and time window")
    return lc


def _register_models(has_jax, want):
    """Return ``[(label, model_name, kind), ...]`` for the models that can exist here."""
    import whisper_cbpf as wp

    out = []
    if "toys" in want:
        for nm in ("gaussian_rise", "bazin", "mck19"):
            out.append((nm, nm, "toy"))
    if "kn2_redback" in want:
        try:
            wp.get_model("two_component_kilonova")
            out.append(("kilonova-2c (redback, CPU)", "two_component_kilonova", "redback"))
        except Exception as exc:
            print(f"  [skip] two_component_kilonova: {exc}")
    if not has_jax:
        return out

    common = dict(redshift=REDSHIFT, dl_cm=DL_CM, band_aliases=BAND_ALIASES)
    builders = [
        ("kn1", "kilonova-1c (JAX)", lambda: wp.register_kilonova(FILTER_SET, n_wave=1000, **common), "kilonova"),
        ("kn2", "kilonova-2c (JAX)", lambda: wp.register_kilonova_two(FILTER_SET, n_wave=1000, **common), "kilonova"),
        ("tde", "TDE gaussianrise", lambda: wp.register_tde(FILTER_SET, rise="gaussian", n_time=5000, **common), "tde"),
        ("tde_ce", "TDE cooling_envelope", lambda: wp.register_tde(FILTER_SET, rise="none", n_time=5000, **common), "tde"),
        ("sn", "SN arnett", lambda: wp.register_supernova("arnett", FILTER_SET, **common), "supernova"),
    ]
    for key, label, build, kind in builders:
        if key not in want:
            continue
        try:
            out.append((label, build().name, kind))
        except Exception as exc:
            print(f"  [skip] {label}: {type(exc).__name__}: {str(exc)[:120]}")
    return out


def _run_cell(lc, model_name, sampler, space, kwargs):
    """One fit. Returns a row dict; never raises."""
    import whisper_cbpf as wp

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = wp.fit(lc, model_name, sampler=sampler, space=space, seed=0, **kwargs)
        pm = res.info.get("predictive_metrics") or {}
        w = pm.get("waic") if isinstance(pm.get("waic"), dict) else None
        return {
            "status": "works", "aic": res.aic, "bic": res.bic,
            "waic": (w or {}).get("waic"),
            "waic_ok": (w or {}).get("p_waic_reliable"),
            "waic_err": res.info.get("predictive_metrics_error"),
            "space": res.info.get("space"), "n_params": res.n_params,
            "runtime": time.perf_counter() - t0,
            "converged": res.info.get("converged"),
            "warnings": len([c for c in caught if issubclass(c.category, UserWarning)]),
        }
    except Exception as exc:
        return {"status": "broken", "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}",
                "runtime": time.perf_counter() - t0}


def _print_table(rows, space, device):
    hdr = f"  {'model':<26} {'sampler':<26} {'k':>2} {'AIC':>13} {'BIC':>13} {'WAIC':>15} {'s':>7}"
    print(f"\n=== {device.upper()} / {space} space " + "=" * (len(hdr) - 24 - len(space) - len(device)))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        if r["status"] != "works":
            tag = "IMPOSSIBLE" if r["status"] == "impossible" else "BROKEN"
            print(f"  {r['model']:<26} {r['sampler']:<26} {tag:>10}  {r['reason'][:60]}")
            continue
        w = r["waic"]
        wtxt = "n/a" if w is None else (f"{w:.1f}" + ("" if r["waic_ok"] else " !"))
        print(f"  {r['model']:<26} {r['sampler']:<26} {r['n_params']:>2} "
              f"{r['aic']:>13.1f} {r['bic']:>13.1f} {wtxt:>15} {r['runtime']:>7.1f}")
    ok = [r for r in rows if r["status"] == "works"]
    if ok:
        best = min(ok, key=lambda r: r["aic"])
        print(f"\n  Lowest AIC in {space} space: {best['model']} / {best['sampler']} "
              f"({best['aic']:.1f}). '!' marks a WAIC its own p_waic says is unreliable.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--space", choices=["flux", "magnitude", "both"], default="both")
    ap.add_argument("--quick", action="store_true", help="tiny budgets; posteriors NOT converged")
    ap.add_argument("--models", default="toys,kn2_redback,kn1,kn2,tde,sn",
                    help="comma-separated: toys,kn2_redback,kn1,kn2,tde,tde_ce,sn")
    ap.add_argument("--samplers", default="", help="comma-separated subset; default = all legitimate")
    ap.add_argument("--plot", default=None, help="write the light curve + best fit here (PNG)")
    args = ap.parse_args(argv)

    import whisper_cbpf as wp

    has_jax = False
    if args.device == "gpu":
        try:
            import jax
            # x64 BEFORE the first array exists. The TDE and all twelve supernovae require it, and
            # the flag is global and session-wide, so it can never be a model default.
            jax.config.update("jax_enable_x64", True)
            has_jax = True
            ok, msg = wp.check_gpu()
            print(f"device: {jax.devices()}  x64={jax.config.jax_enable_x64}")
            if not ok:
                print(f"\n!! JAX IS NOT ON A GPU. Results will be correct and ~50x slow.\n{msg}\n")
        except ImportError:
            print("!! --device gpu but JAX is not importable; falling back to the CPU matrix.\n"
                  "   pip install 'whisper-cbpf[gpu]' and source scripts/whisper-gpu-env.sh first.")

    print(f"\nloading {CSV.relative_to(REPO)} ...")
    lc = _load_lc()
    print(f"  {lc.n_points} points, bands {sorted(set(np.asarray(lc.band)))}, "
          f"data_mode={lc.data_mode}, z={REDSHIFT}")

    print("registering models ...")
    models = _register_models(has_jax, set(args.models.split(",")))
    print(f"  {len(models)} model(s): " + ", ".join(lbl for lbl, _, _ in models))

    samplers = CPU_SAMPLERS + (GPU_SAMPLERS if args.device == "gpu" else [])
    if args.samplers:
        samplers = [s for s in samplers if s in set(args.samplers.split(","))]
    budget = _budget(args.quick)
    spaces = ["flux", "magnitude"] if args.space == "both" else [args.space]

    all_rows = {}
    for space in spaces:
        rows = []
        for label, name, kind in models:
            for s in samplers:
                why = impossible_reason(kind, s, space, has_jax)
                if why:
                    rows.append({"model": label, "sampler": s, "status": "impossible", "reason": why})
                    continue
                print(f"  fitting {label} / {s} / {space} ...", flush=True)
                r = _run_cell(lc, name, s, space, budget.get(s, {}))
                r.update(model=label, sampler=s)
                rows.append(r)
        all_rows[space] = rows
        _print_table(rows, space, args.device)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        fig = wp.plot_light_curve(lc, quantity="apparent_mag")
        fig.savefig(args.plot, dpi=130, bbox_inches="tight")
        print(f"\nplot written to {args.plot}")

    n_ok = sum(r["status"] == "works" for rows in all_rows.values() for r in rows)
    n_broken = sum(r["status"] == "broken" for rows in all_rows.values() for r in rows)
    n_imp = sum(r["status"] == "impossible" for rows in all_rows.values() for r in rows)
    print(f"\nmatrix: {n_ok} works, {n_broken} broken, {n_imp} structurally impossible")
    if n_broken:
        print("\nBROKEN cells (these are defects, not structural limits):")
        for rows in all_rows.values():
            for r in rows:
                if r["status"] == "broken":
                    print(f"  {r['model']} / {r['sampler']}: {r['reason']}")
    return 1 if n_broken else 0


if __name__ == "__main__":
    sys.exit(main())
