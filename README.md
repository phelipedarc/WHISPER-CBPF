# WHISPER-CBPF

Bayesian model comparison for astronomical transient light curves — supernovae, kilonovae and tidal
disruption events — on CPU or GPU.


```bash
pip install "git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

> **Status: alpha (0.1.0).** The package installs and runs, but not every sampler has been validated
> to the same level. Read [Known limitations](#known-limitations) before publishing a number.

## Your first fit

```python
import whisper_cbpf as wp

# 1. Load a light curve. AT2017GFO is in the repository, so this path works from a checkout.
lc = wp.load_lightcurve("tests/data/at2017gfo.csv", redshift=0.0098)
lc = lc.select_bands(["g"]).set_explosion_date(57982.529).select_time_window(0.0, 10.0)
print(lc.n_points)          # 53

# 2. Fit a model.
result = wp.fit(lc, "bazin", sampler="mcmc", space="magnitude", nsteps=2000, burnin=500, seed=0)

# 3. Read the answer.
print(result.summary["t0"]["median"])     # each parameter: median, mean, std, ci16, ci84
print(result.aic, result.bic)             # 279.3  287.2
```

> **One ordering rule that bites everyone.** Call `set_explosion_date` *before*
> `select_time_window`. The window filters whatever `time` currently holds, so windowing first
> compares days-since-explosion against raw MJD (~57983) and quietly leaves you with an empty light
> curve.

## Comparing models

Fit each candidate to the same data, then compare information criteria. **Lower is better.**

```python
for name in ["bazin", "gaussian_rise"]:
    r = wp.fit(lc, name, sampler="mcmc", space="magnitude", nsteps=2000, burnin=500, seed=0)
    print(f"{name:16s} AIC={r.aic:8.1f}  BIC={r.bic:8.1f}")

# bazin            AIC=   279.3  BIC=   287.2
# gaussian_rise    AIC=   275.5  BIC=   283.4   <- preferred
```

> **Only compare fits that converged.** These demo budgets are small and neither fit passes its
> convergence check. `bazin` moves from AIC 279.3 to 491.2 just by changing `nsteps` from 2000 to
> 1500, because AIC is taken from the best draw. Raise the budget until `r.info["converged"]` is
> `True` before you compare anything you intend to publish.

Full worked examples, with a personalised prior on the two-component kilonova:

* [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) — CPU and GPU, with a personalised prior

**AIC, BIC and WAIC compare models, not spaces.** Numbers from a flux-space fit are not comparable
with numbers from a magnitude-space fit. Keep `space=` the same across everything you compare.

## Plots

```python
wp.plot_light_curve(lc, save="lc.png")               # the data
wp.plot_ppc(result, lc, model="bazin", save="ppc.png")   # posterior predictive check
wp.plot_corner([result], save="corner.png")          # note the list -- overlay several results
```

## Custom priors

Every sampler accepts `prior=`. Build one from `Uniform` and `LogUniform`:

```python
prior = wp.Prior({
    "amplitude": wp.Uniform(0.0, 5.0, name="amplitude"),
    "t0":        wp.Uniform(0.0, 10.0, name="t0"),
    "tau_rise":  wp.LogUniform(0.1, 20.0, name="tau_rise"),
    "tau_fall":  wp.LogUniform(0.1, 50.0, name="tau_fall"),
})
result = wp.fit(lc, "bazin", sampler="nested", prior=prior, space="magnitude", seed=0)
```

Omit `prior=` and the model's own default is used.

## Models

```python
wp.list_models()
```

| Model | Kind | Needs |
|---|---|---|
| `bazin`, `gaussian_rise`, `flare` | empirical rise/fall shapes | — |
| `mck19` | analytic kilonova | — |
| `two_component_kilonova` | blue + red kilonova (redback, CPU) | `[models]` |
| `flare_jax` | JAX flare | `[gpu]` |

Physical JAX models are built on demand, because they need your bands and distance:

```python
import jax; jax.config.update("jax_enable_x64", True)   # FIRST -- before any array exists
import whisper_cbpf as wp

kn  = wp.register_kilonova(["sdssg"], redshift=0.0098, dl_cm=1.23e26)        # one component
kn2 = wp.register_kilonova_two(["sdssg"], redshift=0.0098, dl_cm=1.23e26)    # blue + red
kn3 = wp.register_kilonova_three(["sdssg"], redshift=0.0098, dl_cm=1.23e26)  # blue + purple + red
tde = wp.register_tde(["sdssg"], redshift=0.0098, dl_cm=1.23e26)
sn  = wp.register_supernova(["sdssg"], redshift=0.0098, dl_cm=1.23e26)
```

The multi-component kilonovae default to **Villar et al. (2017)** priors, with each component given
its own **disjoint opacity corridor**:

| component | `kappa` prior | Villar's fixed value |
|---|---|---|
| blue (lanthanide-poor) | U(0.1, 1.0) | 0.5 cm² g⁻¹ |
| purple (3-component only) | U(1.0, 5.0) | 3 cm² g⁻¹ |
| red (lanthanide-rich) | U(5.0, 30.0) — U(1.0, 30.0) with no purple | 10 cm² g⁻¹ |

The name of a component is cosmetic; **the opacity corridor is what makes it blue, purple or red**.
Keeping the corridors disjoint also means the components cannot swap labels, so the posterior has no
mirror modes to average over. Villar's white-noise term `sigma` is a *likelihood* parameter, not a
model one — fit it with `likelihood="scatter"`.

`float64` is **mandatory** for the TDE and every supernova — they raise at the first `predict`
without it, rather than returning a wrong answer.

## Samplers

```python
wp.list_samplers()
```

| Sampler | Device | Needs | Use it for |
|---|---|---|---|
| `mcmc` | CPU | — | the sensible default; emcee ensemble |
| `nested` (`dynesty`) | CPU | — | when you want **log-evidence** for Bayes factors |
| `abc`, `abc_smc` | CPU | — | likelihood-free; tolerant of awkward models |
| `emcee_jax` | GPU | `[gpu]` | emcee, vectorised on the GPU |
| `nuts_gpu` | GPU | `[gpu]` | gradient-based NUTS — **see Known limitations** |
| `pymc_jax_gpu_vectorized` / `_parallelized` | GPU | `[gpu]` | NUTS via a PyMC front end |
| `abc_gpu`, `abc_smc_gpu` | GPU | `[gpu]` | ABC with GPU-batched simulation |
| `snpe` / `npe` | CPU/GPU | `[sbi]` | neural simulation-based inference |

Which one to pick, and which combinations cannot work, is in
[`docs/CHOOSING.md`](docs/CHOOSING.md).

## Install options

Three extras, combinable:

```bash
pip install "whisper-cbpf[gpu,models,sbi] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

| Extra | Gives you |
|---|---|
| `[gpu]` | all JAX models and GPU samplers (includes pymc, arviz, astroquery) |
| `[models]` | redback physical models and the TDE / supernova priors |
| `[sbi]` | the SNPE / NPE sampler |

**GPU users:** `[gpu]` alone is not enough. Run `source "$(whisper-cbpf-env)"` *before* starting
Python, or JAX silently falls back to CPU at ~50× slower with no error. Full detail in
[`INSTALL.md`](INSTALL.md) and [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md).

## Known limitations

Honest list, so you do not discover these in review:

* **`nuts_gpu` can return a confidently wrong posterior.** On simulated kilonovae with a known
  truth, it placed the truth outside its entire posterior while reporting `r-hat ≈ 1.03` and zero
  divergences. Under investigation. Prefer `pymc_jax_gpu_vectorized`, `nested` or `mcmc` for
  anything you intend to publish, and check `result.info["converged"]`.
* **No sampler has a published calibration result yet.** Simulation-based calibration is in
  progress; until it lands, treat credible intervals as unvalidated.
* **The redback CPU model `two_component_kilonova` has exchangeable components.** Its prior gives
  both the same `kappa` range, so fits can label-switch. The JAX models
  (`kilonova_two_jax`, `kilonova_three_jax`) do not have this problem — their default Villar priors
  use disjoint corridors. Impose disjoint `kappa` ranges yourself if you use the redback model.
* **A WAIC whose `p_waic` is flagged unreliable is not a number** — check the flag before quoting it.

## Documentation

| | |
|---|---|
| [`INSTALL.md`](INSTALL.md) | installing, GPU setup, verifying |
| [`docs/TUTORIAL.md`](docs/TUTORIAL.md) | guided walkthrough |
| [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) | worked comparison on CPU and GPU |
| [`docs/CHOOSING.md`](docs/CHOOSING.md) | picking a sampler; impossible combinations |
| [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) | CUDA, float64, multi-GPU |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | every public symbol |
| [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) | how the JAX models differ from redback, and why |
| [`notebooks/`](notebooks/) | ten runnable notebooks, one per subsystem |

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

GPL-3.0 — see [`LICENSE`](LICENSE).
