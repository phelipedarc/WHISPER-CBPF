#!/usr/bin/env python
"""Demo: Sequential Neural Posterior Estimation (SNPE / NPE) in Whisper.

The same workflow as a bare-``sbi`` script, but the simulator is Whisper's forward model, the prior is
a Whisper ``Prior``, and the result is the usual ``SamplerResult`` (with the trained sbi posterior
attached as ``result.posterior``). Needs the optional ``[sbi]`` extra (sbi + torch).

Ported from WHISPER_AI @ 8ba3843 (discontinued; superseded by whisper_cbpf), ``dev/demo_snpe.py``.
From the repo root::

    python examples/demo_snpe.py
"""
from __future__ import annotations

import numpy as np

import whisper_cbpf as wp
from whisper_cbpf.models import get_model

# ---------------------------------------------------------------- synthetic observation
TRUE = {"amplitude": 1.0, "rise_time": 1.0, "decay_time": 3.0}
t = np.linspace(0.1, 10.0, 50)
flux = get_model("flare").predict(TRUE, t, None)
err = np.full_like(flux, 0.05)
obs = flux + np.random.default_rng(0).normal(0.0, err)

lc = wp.LightCurve(time=t, band=["r"] * len(t), flux=obs, flux_err=err, name="snpe-demo")

# ---------------------------------------------------------------- prior (Whisper, any sampler reuses it)
prior = wp.Prior({
    "amplitude":  wp.Uniform(0.0, 5.0),
    "rise_time":  wp.Uniform(0.1, 5.0),
    "decay_time": wp.Uniform(0.5, 10.0),
})

# ---------------------------------------------------------------- run SNPE
print("Running SNPE (this trains a neural density estimator — ~1 min)...")
res = wp.fit_SNPE(
    lc, "flare", prior=prior,
    num_rounds=2,            # 1 = amortized NPE; >1 = sequential SNPE-C
    num_simulations=2000,    # per round
    num_samples=10000,
    space="auto",            # flux space here (flux data)
    seed=0,
)

print(res)
print(f"space={res.info['space']}  total_simulations={res.info['total_simulations']}  "
      f"runtime={res.runtime_s:.1f}s")
print(f"max_logL={res.max_log_likelihood:.2f}  AIC={res.aic:.2f}  BIC={res.bic:.2f}\n")

print(f"{'parameter':12s} {'true':>6s} {'median':>8s} {'ci16':>8s} {'ci84':>8s}")
for name in res.parameters:
    s = res.summary[name]
    print(f"{name:12s} {TRUE[name]:6.2f} {s['median']:8.3f} {s['ci16']:8.3f} {s['ci84']:8.3f}")

# ---------------------------------------------------------------- corner plot (optional)
try:
    import corner
    fig = corner.corner(res.samples[res.parameters].to_numpy(), labels=res.parameters,
                        truths=[TRUE[p] for p in res.parameters], show_titles=True)
    out = "snpe_flare_corner.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved corner plot -> {out}")
except Exception as exc:   # corner optional / headless quirks must not fail the demo
    print(f"\n(corner plot skipped: {exc})")

# You can also resample / use sbi's own pairplot directly on the trained posterior:
#   samples = res.posterior.sample((10000,))
#   from sbi.analysis import pairplot; pairplot(samples, labels=res.parameters)
