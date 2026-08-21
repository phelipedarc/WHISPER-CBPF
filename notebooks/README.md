# Notebooks

Ten notebooks, one per subsystem. Each states what it covers and what it needs, and each has been
executed end to end — the stored outputs are real.

| | notebook | needs |
|---|---|---|
| 1 | [Light curves and I/O](01_light_curves.ipynb) — loading, selecting, magnitudes and flux, phase, absolute magnitude | — |
| 2 | [Bands, filters and SVO](02_bands_and_filters.ipynb) — how a band name becomes a wavelength and a zero point | — |
| 3 | [Models](03_models.ipynb) — the built-ins, binding a redback model, registering your own | `[models]` for the redback section |
| 4 | [Priors, likelihoods and distances](04_priors_and_likelihoods.ipynb) — three of the four pluggable axes | — |
| 5 | [CPU samplers](05_cpu_samplers.ipynb) — `mcmc`, `nested`, `abc`, `abc_smc`, `snpe` | `[sbi]` for SNPE |
| 6 | [JAX physical models](06_jax_physical_models.ipynb) — kilonovae, TDEs, supernovae | `[gpu]` |
| 7 | [GPU samplers](07_gpu_samplers.ipynb) — NUTS three ways, GPU ABC, emcee-over-JAX, SNPE | `[gpu]` |
| 8 | [Metrics and model comparison](08_metrics_and_comparison.ipynb) — AIC, BIC, WAIC, PSIS-LOO, coverage | `arviz` for PSIS-LOO |
| 9 | [Validation and calibration](09_validation.ipynb) — recovery, posterior predictive checks, SBC | — |
| 10 | [Plotting](10_plotting.ipynb) — light curves, PPCs, corner plots, calibration | — |

Start at 1 if you are new. If you only want to fit something, 3 → 5 → 8 is the short path.

## Running them

They resolve the repository root themselves, so they work from this directory or from the root:

```bash
jupyter lab notebooks/
```

Notebooks 6 and 7 need the GPU environment sourced **before** Python starts, or JAX falls back to
the CPU silently:

```bash
source "$(whisper-cbpf-env)"
```

## Budgets

Every fit in these notebooks uses a small, illustrative budget so the set runs in minutes. Before
quoting any number, raise the budget until `result.info["converged"]` is `True`, and only compare
fits that used the same `space=`.

## Data

`../tests/data/at2017gfo.csv` is the raw AT2017GFO photometry the notebooks load.
`data/at2017gfo_full_preprocessed.csv` is a cleaned 133-point reduction of the same event, if you
want one that needs no selection.
