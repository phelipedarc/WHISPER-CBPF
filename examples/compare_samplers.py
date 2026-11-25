#!/usr/bin/env python
"""ABC, ABC-SMC, MCMC and SNPE on one synthetic light curve: how close they come, at these settings.

All four samplers share Whisper's model, prior and Gaussian likelihood, so they target the same
posterior -- but ABC and ABC-SMC reach it only as the acceptance distance goes to zero, and SNPE only
as its simulation budget grows, so at a finite budget they return something wider, and ABC can be
shifted. This fits the ``gaussian_rise`` model with each, prints median +/- 68 % per parameter, and
overlays the four posteriors in one corner plot (truth marked), saved as
``sampler_comparison_corner.png`` in the current directory. SNPE needs the ``[sbi]`` extra.

What it printed (whisper 0.1.0 plus the 0.1.1 fixes, seed 0; truth amplitude 5, t0 8,
sigma_rise 3, tau_decay 15)::

    parameter    ABC                ABC-SMC             MCMC             SNPE
    amplitude    4.42 +0.86/-0.50   3.85 +0.94/-0.61    5.02 +/-0.03     4.96 +0.45/-0.37
    t0           8.31 +2.60/-2.10   9.62 +3.33/-3.00    7.92 +/-0.08     7.75 +0.46/-0.51
    sigma_rise   3.59 +2.41/-1.88   6.05 +4.67/-3.56    2.91 +/-0.07     2.77 +0.23/-0.24
    tau_decay   18.33 +9.65/-5.97  22.16 +19.1/-9.80   15.14 +/-0.23    16.10 +4.45/-3.87

MCMC, the likelihood-based reference, is tight and on the truth. SNPE is centred within one of its
own sd of the truth but 3-18x wider than MCMC. ABC and ABC-SMC are 20-65x wider, their medians
shifted from MCMC's by up to 1.5 of their own sd (ABC-SMC's 68 % interval on the amplitude misses
the truth). Tightening their acceptance (a smaller ``quantile`` with more simulations, or
``min_epsilon``) is what brings them toward MCMC; this script does not measure how far. So it shows
how far each method is from the likelihood posterior at a practical budget, not that they agree.

Ported from WHISPER_AI @ 8ba3843 (discontinued; superseded by whisper_cbpf),
``sanity_check/compare_samplers.py``. From the repo root::

    python examples/compare_samplers.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import whisper_cbpf as wp
from whisper_cbpf.models import get_model

MODEL = "gaussian_rise"
TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}


def main():
    # ------------------------------------------------------------ synthetic gaussian_rise light curve
    t = np.linspace(0.1, 30.0, 60)
    flux = get_model(MODEL).predict(TRUE, t, None)
    err = np.full_like(flux, 0.1)
    obs = flux + np.random.default_rng(42).normal(0.0, err)
    lc = wp.LightCurve(time=t, band=["r"] * len(t), flux=obs, flux_err=err, name="sanity")

    prior = get_model(MODEL).default_prior    # all samplers share the same model + prior + likelihood

    # ------------------------------------------------------------ run the four samplers
    print("Running ABC ...")
    r_abc = wp.fit_ABC(lc, MODEL, prior=prior, n_simulations=200_000, quantile=0.003, n_jobs=8, seed=0)
    print("Running ABC-SMC ...")
    r_smc = wp.fit_ABC_SMC(lc, MODEL, prior=prior, n_particles=1000, n_rounds=6, quantile=0.5,
                           n_jobs=8, seed=0)
    print("Running MCMC ...")
    r_mcmc = wp.fit_MCMC(lc, MODEL, prior=prior, nsteps=4000, burnin=1000, thin=5, seed=0)
    print("Running SNPE (trains a neural net) ...")
    r_snpe = wp.fit_SNPE(lc, MODEL, prior=prior, num_rounds=2, num_simulations=2500, num_samples=6000,
                         space="flux", seed=0)

    results = [("ABC", r_abc, "#1f77b4"), ("ABC-SMC", r_smc, "#2ca02c"),
               ("MCMC", r_mcmc, "#d62728"), ("SNPE", r_snpe, "#9467bd")]
    names = list(prior.names)
    truths = [TRUE[p] for p in names]

    # ------------------------------------------------------------ comparison table
    print(f"\n{'parameter':12s} {'true':>7s}" + "".join(f"{lbl:>22s}" for lbl, _, _ in results))
    for p in names:
        row = f"{p:12s} {TRUE[p]:7.2f}"
        for _, r, _ in results:
            s = r.summary[p]
            row += f"   {s['median']:7.3f} +{s['ci84'] - s['median']:.3f}/-{s['median'] - s['ci16']:.3f}"
        print(row)

    # ------------------------------------------------------------ overlaid corner plot
    # Common per-parameter range so the four corner plots align, computed from the union of all samples.
    allsamp = np.vstack([r.samples[names].to_numpy() for _, r, _ in results])
    rng = [(np.percentile(allsamp[:, i], 0.5), np.percentile(allsamp[:, i], 99.5))
           for i in range(len(names))]

    fig = None
    for i, (lbl, r, color) in enumerate(results):
        fig = corner.corner(
            r.samples[names].to_numpy(), fig=fig, color=color, labels=names, range=rng, bins=30,
            smooth=1.0, plot_datapoints=False, plot_density=False, fill_contours=False,
            levels=(0.39, 0.86), contour_kwargs=dict(colors=color),
            hist_kwargs=dict(density=True, color=color, lw=1.6),
            truths=truths if i == 0 else None, truth_color="black")   # truth (dashed black) once
    fig.legend(handles=[Patch(color=c, label=lbl) for lbl, _, c in results],
               loc="upper right", frameon=True, fontsize=12, title="sampler")
    fig.suptitle(f"{MODEL}: four samplers' posteriors (truth in black)", y=1.02)

    out = "sampler_comparison_corner.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")


# The guard is required: the parallel ABC samplers start their workers with "spawn", which
# re-imports this file in every worker.
if __name__ == "__main__":
    main()
