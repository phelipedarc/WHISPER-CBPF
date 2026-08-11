# WHISPER-CBPF

**Bayesian model comparison for astronomical transient light curves — one package, CPU and GPU.**

Load a light curve, fit candidate models with several samplers, get posteriors plus AIC/BIC and
predictive metrics, and answer *"which model best fits this transient?"*

```python
import whisper_cbpf as wp

lc  = wp.load_lightcurve("transient.csv")
res = wp.fit(lc, "gaussian_rise", sampler="abc")      # CPU, no extras needed
print(res.posterior.describe(), res.aic, res.bic)
```

## Install

```bash
pip install whisper-cbpf
```

That is a complete, working CPU package. It imports with **no JAX, no CUDA, no torch and no
redback**, and every GPU model and sampler still appears in `list_models()` / `list_samplers()` /
`list_distances()` — you can discover what exists before deciding to install it.

| Extra | Command | What it adds |
|---|---|---|
| *(none)* | `pip install whisper-cbpf` | ingestion, plotting, priors, likelihoods, ABC, ABC-SMC, MCMC |
| `[gpu]` | `pip install "whisper-cbpf[gpu]"` | JAX/CUDA-12 models and samplers, NumPyro NUTS, ArviZ ESS |
| `[pymc]` | `pip install "whisper-cbpf[pymc]"` | the two PyMC-fronted NUTS samplers |
| `[sbi]` | `pip install "whisper-cbpf[sbi]"` | SNPE / NPE (torch) |
| `[models]` | `pip install "whisper-cbpf[models]"` | redback physical models and priors (needs a C compiler) |
| `[svo]` | `pip install "whisper-cbpf[svo]"` | SVO filter lookup for unknown bands |
| `[loo]` | `pip install "whisper-cbpf[loo]"` | PSIS-LOO / ELPD in `predictive_metrics` |
| `[dev]` | `pip install -e ".[dev]"` | pytest and the build tooling |

**GPU users: `pip install "whisper-cbpf[gpu]"` is not sufficient on its own.** You must source the
environment bootstrap *before starting Python*, because JAX reads device visibility and memory
fraction at import time:

```bash
source scripts/whisper-gpu-env.sh
```

Skip it and JAX falls back to CPU **silently, at roughly 50×, with no error**. Check with
`whisper_cbpf.check_gpu()`. See [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md).

## Models

Analytic models are registered on import. The photometric models are **not** — turning parameters
into a band flux needs a filter set, a redshift and a luminosity distance, none of which have an
honest default (a guessed distance silently rescales every fitted mass), so you bind the dataset
context yourself with a factory.

| Model | Backend | Requires | How to get it |
|---|---|---|---|
| `gaussian_rise` | cpu | — | registered |
| `flare` | cpu | — | registered |
| `bazin` | cpu | — | registered |
| `mck19` | cpu | — | registered — BBH-in-AGN-disk kicked-hotspot flare |
| `two_component_kilonova` | cpu | `[models]` | registered — redback-backed |
| `flare_jax` | gpu | `[gpu]` | registered |
| one/two/three-component kilonova | gpu | `[gpu]` | `wp.register_kilonova(bands, redshift, dl_cm)` |
| TDE (cooling envelope, Gaussian rise) | gpu | `[gpu]` | `wp.register_tde(bands, redshift, dl_cm)` |
| twelve redback supernovae | gpu | `[gpu]` | `wp.register_supernova(name, bands, redshift, dl_cm)` — see `wp.supernova_models()` |

> **`flare` and `flare_jax` are different models, not a CPU/GPU pair.** `flare` is
> `A(1−e^{−t/t_rise})·e^{−t/t_decay}`; `flare_jax` is a Gaussian rise with exponential decay in log
> parameters, which makes it the GPU counterpart of **`gaussian_rise`**. Their parameter names are
> disjoint, so a mistake raises rather than silently fitting the wrong thing — but do not read the
> shared name stem as a claim of equivalence. See [`DUPLICATES.md`](DUPLICATES.md).

> **The two kilonovae disagree on purpose.** `two_component_kilonova` (redback) and the JAX kilonova
> are validated against different references and use **disjoint priors**. They are two models, not
> one model twice.

## Samplers

Every sampler returns the same `SamplerResult`, so results are comparable across backends.

| Sampler | Backend | Requires | Notes |
|---|---|---|---|
| `abc` | cpu | — | rejection ABC, parallel over cores |
| `abc_smc` | cpu | — | importance-weighted sequential ABC |
| `mcmc` | cpu | — | emcee over Whisper's likelihood |
| `snpe` / `npe` | cpu/gpu | `[sbi]` | torch; has its own `device='cuda'` path |
| `abc_gpu` | gpu | `[gpu]` | JAX port of `abc`; crossover vs CPU at ~50k simulations |
| `abc_smc_gpu` | gpu | `[gpu]` | JAX port of `abc_smc` |
| `emcee_jax` | gpu | `[gpu]` | emcee over a jitted JAX log-density |
| `nuts_gpu` | gpu | `[gpu]` | NumPyro NUTS |
| `pymc_jax_gpu_vectorized` | gpu | `[gpu]`+`[pymc]` | vmap over chains — the 2-GPU default |
| `pymc_jax_gpu_parallelized` | gpu | `[gpu]`+`[pymc]` | true pmap; fastest per chain |
| `snpe_gpu` | gpu | `[gpu]` | **not registered** — flare-only; import directly |

Which to use, and which combinations are structurally impossible (a numpy simulator has no
gradients for NUTS and cannot be `vmap`ped by `abc_gpu`), is in
[`docs/CHOOSING.md`](docs/CHOOSING.md).

## Where things came from

WHISPER-CBPF merges two repositories that remain live and citable:

- [WHISPER_AI](https://github.com/phelipedarc/WHISPER_AI) — the CPU core, imported as `whisper_labia`
- [whisper-GPU](https://github.com/phelipedarc/whisper-GPU) — the JAX/GPU extension

Old import → new import for every public symbol is in [`docs/MIGRATION.md`](docs/MIGRATION.md).
Every file's origin repo, path and commit is in [`PROVENANCE.md`](PROVENANCE.md).

## Docs

| | |
|---|---|
| [`docs/TUTORIAL.md`](docs/TUTORIAL.md) | start here |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | the full public surface |
| [`docs/CHOOSING.md`](docs/CHOOSING.md) | which model and sampler for which problem |
| [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) | the silent-CPU-fallback trap, float64, multi-GPU |
| [`docs/EXTENDING.md`](docs/EXTENDING.md) | add a model, sampler, likelihood or distance |
| [`docs/DESIGN.md`](docs/DESIGN.md) | why the architecture is shaped this way |
| [`docs/PHYSICS_NOTES.md`](docs/PHYSICS_NOTES.md) | the physics behind the JAX models |
| [`notebooks/`](notebooks/) | five worked notebooks |

## License

GPL-3.0 — inherited from the redback dependency.
