# WHISPER-CBPF

**Bayesian model comparison for astronomical transient light curves — one package, CPU and GPU.**

Load a light curve, fit candidate models with several samplers, get posteriors plus AIC/BIC and
predictive metrics, and answer *"which model best fits this transient?"*

```python
import numpy as np
import whisper_cbpf as wp

# a light curve we know the answer to, so you can see the fit work
truth = {"amplitude": 5.0, "t0": 10.0, "sigma_rise": 3.0, "tau_decay": 12.0}
t     = np.linspace(0.5, 40, 60)
flux  = wp.get_model("gaussian_rise").predict(truth, t, None)
flux += np.random.default_rng(0).normal(0, 0.05, flux.shape)

lc  = wp.LightCurve(time=t, band=np.array(["ztfg"] * len(t)), flux=flux,
                    flux_err=np.full(len(t), 0.05), name="demo", redshift=0.01)

res = wp.fit(lc, "gaussian_rise", sampler="mcmc")      # CPU, no extras needed
print(res.aic, res.bic, res.max_log_likelihood)
print({k: round(v["median"], 2) for k, v in res.summary.items()})
```

`res.summary` is `{parameter: {median, ci16, ci84, mean, std}}`:

```
-200.7 -192.3 104.3
{'amplitude': 4.96, 't0': 10.06, 'sigma_rise': 3.05, 'tau_decay': 12.13}
```

Against a truth of `5.0 / 10.0 / 3.0 / 12.0`, every parameter is recovered to within 1.3%. Plots:

```python
wp.plot_light_curve(lc)          # the data
wp.plot_corner([res])            # the posterior — note the list, see below
wp.plot_ppc(res, lc)             # posterior predictive check
```

**Your own data** goes through `load_lightcurve` instead:

```python
lc = wp.load_lightcurve("transient.csv")        # auto-maps columns, groups bands, mag<->flux
lc = lc.select_bands(["r"]).set_explosion_date(58598.0)
```

Two things to know before fitting real transients, because the defaults will not do it for you:

- **Scale the prior to your flux units.** The analytic models ship
  `amplitude ~ Uniform(0, 10)`, and if your fluxes are ~1e-5 Jy the prior cannot reach the
  right amplitude at all — you will get a converged fit to a wrong answer. Pass your own
  `Prior({...})`.
- **Analytic models are toys.** `gaussian_rise`, `flare` and `bazin` are shape fitters, not
  physics. On a real kilonova they leave a χ²/N in the tens no matter how you tune them. For real
  transients use the physical models: `mck19`, `two_component_kilonova`, or the JAX kilonova / TDE /
  supernova families.

The plotters take what you'd expect, with one exception worth knowing up front:
**`plot_corner` takes a *sequence* of posteriors, even for a single fit** — it exists to overlay
several on shared axes, so `plot_corner([res])` works and `plot_corner(res)` raises
`TypeError: ... has no len()`. Comparing samplers is the point:

```python
wp.plot_corner([r_abc, r_mcmc], labels=["abc", "mcmc"])
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
