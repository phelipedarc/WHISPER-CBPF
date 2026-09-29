# Notebooks

Eleven notebooks: one per subsystem, and a quickstart from an LSST alert to a ranked answer. Each
states what it covers and what it needs, and each has been executed end to end, so the stored
outputs are real.

| | notebook | needs |
|---|---|---|
| 1 | [Light curves and I/O](01_light_curves.ipynb) — loading, selecting, magnitudes and flux, phase, absolute magnitude, ZTF and LSST alerts | — |
| 2 | [Bands, filters and SVO](02_bands_and_filters.ipynb) — how a band name becomes a wavelength, a zero point and a filter curve | — |
| 3 | [Models](03_models.ipynb) — the built-ins, binding a redback model, registering your own | `[models]` for the redback section |
| 4 | [Priors, likelihoods and distances](04_priors_and_likelihoods.ipynb) — three of the four pluggable axes | — |
| 5 | [CPU samplers](05_cpu_samplers.ipynb) — `mcmc`, `nested`, `abc`, `abc_smc`, `snpe`; the convergence report; the likelihood peak | `[sbi]` for SNPE |
| 6 | [JAX physical models](06_jax_physical_models.ipynb) — kilonovae, TDEs, supernovae; explosion time and redshift as parameters | `[gpu]` |
| 7 | [GPU samplers](07_gpu_samplers.ipynb) — NUTS three ways, GPU ABC, emcee-over-JAX, SNPE | `[gpu]` |
| 8 | [Metrics and model comparison](08_metrics_and_comparison.ipynb) — `compare`, AIC, BIC, WAIC, PSIS-LOO, coverage, saving | `arviz` for PSIS-LOO |
| 9 | [Validation and calibration](09_validation.ipynb) — recovery, posterior predictive checks, SBC | — |
| 10 | [Plotting](10_plotting.ipynb) — light curves, PPCs, every model over the data, the ranking, corner plots, posterior widths, forecasts, calibration | — |
| 11 | [From an LSST alert to a ranked answer](11_lsst_alert_quickstart.ipynb) — a Rubin alert packet, three supernova families compared, diagnostics, forecast, facts file and HTML report | `[gpu]` (runs on the CPU, slower) |

Start at 1 if you are new. If you only want to fit something, 3 → 5 → 8 is the short path. If you
have an alert, start at 11.

## Running them

They resolve the repository root themselves, so they work from this directory or from the root:

```bash
jupyter lab notebooks/
```

Notebooks 6, 7 and 11 need the GPU environment sourced **before** Python starts, or JAX falls back
to the CPU silently:

```bash
source "$(whisper-cbpf-env)"
```

## Budgets and priors

Every fit in these notebooks uses a small, illustrative budget so the set runs in minutes. Before
quoting any number, raise the budget until `result.diagnostics()` passes; when it fails, it says
which check failed and why. Only compare fits that used the same `space=`.

The empirical models (`bazin`, `gaussian_rise`, `flare`, `flare_jax`) ship generic priors whose
amplitudes run to 10 Jy and beyond. Every fit to AT2017GFO passes a prior on its flux scale, and a
fit to your own data should too.

## Data

`../tests/data/at2017gfo.csv` is the raw AT2017GFO photometry the notebooks load.
`data/at2017gfo_full_preprocessed.csv` is a cleaned 133-point reduction of the same event, if you
want one that needs no selection. Notebook 11 simulates its alert from a known supernova, so it
needs no data file.
