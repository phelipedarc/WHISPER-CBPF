# Model comparison on GPU

The same job as [`MODEL_COMPARISON_CPU.md`](MODEL_COMPARISON_CPU.md) — the **two-component
kilonova** with a **personalised prior** — but with the JAX models and GPU samplers, where larger
budgets become affordable.

```bash
pip install "whisper-cbpf[gpu] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

## Before any Python: the two things that silently cost you

**1. Source the environment bootstrap.** JAX reads device visibility at *import* time. Get it wrong
and JAX falls back to CPU — roughly 50× slower, with correct results and no warning at all.

```bash
source "$(whisper-cbpf-env)"
python your_fit.py
```

**2. Turn on float64 before the first array exists.**

```python
import jax
jax.config.update("jax_enable_x64", True)   # FIRST LINE. Not after importing whisper_cbpf.
import whisper_cbpf as wp

ok, msg = wp.check_gpu()
print(ok, msg)          # verify -- never assume
```

float64 is *mandatory* for the TDE and every supernova; for the kilonova it is strongly advised.

## The samplers you get on GPU

| Sampler | What it gives you |
|---|---|
| `emcee_jax` | emcee's ensemble moves, vectorised across walkers on the GPU. No gradients needed. |
| `pymc_jax_gpu_vectorized` | NUTS through a PyMC front end, `vmap` over chains. |
| `pymc_jax_gpu_parallelized` | the same, with true `pmap` — fastest per chain across several cards. |
| `abc_gpu` / `abc_smc_gpu` | ABC with simulation batched on the GPU; enormous simulation counts become cheap. |
| `nuts_gpu` | NumPyro NUTS directly. **See the warning below.** |
| `snpe` | neural simulation-based inference (needs `[sbi]`); `device="cuda"` to train on the GPU. |

> ### Warning: `nuts_gpu` is not currently trustworthy
>
> On simulated kilonovae where the true parameters are known, `nuts_gpu` returned posteriors that
> excluded the truth entirely — the true value fell below *every* posterior draw on nearly every
> parameter — while reporting `r-hat ≈ 1.03`, **zero divergences**, and in one case
> `converged=True`. The chains settled roughly 2×10⁵ log-units below the likelihood at the true
> parameters, welded against a prior bound.
>
> Two other NUTS front ends (`pymc_jax_gpu_vectorized`) and `snpe` handled the identical datasets
> correctly, so this is specific to that sampler, not to NUTS or to the model.
>
> **Use `pymc_jax_gpu_vectorized` instead** until this is resolved. If you do run `nuts_gpu`, check
> that the posterior is not pinned to a prior edge before believing anything.

## 1. Register the JAX models

JAX models are built on demand, because they need your band set and luminosity distance:

```python
REDSHIFT, DL_CM = 0.0098, 1.23e26      # AT2017GFO / GW170817, ~40 Mpc

two = wp.register_kilonova_two(["sdssg"], redshift=REDSHIFT, dl_cm=DL_CM,
                               n_wave=300, band_aliases={"g": "sdssg"})
one = wp.register_kilonova(["sdssg"], redshift=REDSHIFT, dl_cm=DL_CM,
                           n_wave=300, band_aliases={"g": "sdssg"})

print(two.parameters)
print(one.parameters)
```

`band_aliases` maps the band names *in your data* to the filter names the model was built with —
here the file says `g` and the model uses `sdssg`. Without it you get a `KeyError` naming the band,
which is deliberate: guessing an alias silently is how you end up fitting the wrong filter.

`n_wave` is the spectral resolution of the synthetic photometry. 300 is a good default; higher costs
time for very little change in a broad band.

> ### The JAX and redback two-component models are not interchangeable
>
> ```
> two_component_kilonova (redback, CPU):  mej_1, vej_1, kappa_1, temperature_floor_1, ... + redshift   (9)
> kilonova_two_jax       (JAX,     GPU):  mej_1, vej_1, temperature_floor_1, kappa_1, ...              (8)
> ```
>
> `kappa` and `temperature_floor` are **in the opposite order**, and the JAX model has no `redshift`
> parameter — it is fixed when you register it. A prior written for one and reused on the other will
> fit something you did not intend. Always build the prior from `model.parameters`.

## 2. A personalised prior, in the model's own order

```python
prior = wp.Prior({
    "mej_1":               wp.Uniform(0.01, 0.03,     name="mej_1"),
    "vej_1":               wp.Uniform(0.15, 0.45,     name="vej_1"),
    "temperature_floor_1": wp.LogUniform(1000., 6000., name="temperature_floor_1"),
    "kappa_1":             wp.Uniform(1.0,  5.0,      name="kappa_1"),
    "mej_2":               wp.Uniform(0.01, 0.03,     name="mej_2"),
    "vej_2":               wp.Uniform(0.05, 0.20,     name="vej_2"),
    "temperature_floor_2": wp.LogUniform(100., 1000.,  name="temperature_floor_2"),
    "kappa_2":             wp.Uniform(10.0, 30.0,     name="kappa_2"),
})

assert list(prior.names) == list(two.parameters)     # cheap, and worth it
```

**The `kappa` corridors are disjoint** — `[1, 5]` for the blue component and `[10, 30]` for the red.
The model's default prior gives both components the *same* `[1, 30]` range, which makes them
exchangeable: swap component 1 and component 2 and the likelihood is identical. That is label
switching, and it yields a bimodal posterior that more sampling will never fix. Disjoint corridors
remove the symmetry by construction.

## 3. Load the data and fit

```python
lc = wp.load_lightcurve("tests/data/at2017gfo.csv", redshift=REDSHIFT)
lc = lc.select_bands(["g"]).set_explosion_date(57982.529)    # BEFORE windowing
lc = lc.select_time_window(0.0, 10.0)

runs = [
    ("abc_gpu",                 dict(n_simulations=20000, quantile=0.02)),
    ("abc_smc_gpu",             dict(n_particles=500, n_rounds=3)),
    ("emcee_jax",               dict(nwalkers=32, nsteps=3000, burnin=1000, thin=5)),
    ("pymc_jax_gpu_vectorized", dict(num_warmup=800, num_samples=800, num_chains=4)),
    ("snpe",                    dict(num_rounds=2, num_simulations=1500,
                                     num_samples=4000, device="cuda")),
]

results = {}
for sampler, kwargs in runs:
    r = wp.fit(lc, two.name, sampler=sampler, prior=prior, space="magnitude", seed=0, **kwargs)
    results[sampler] = r
    print(f"{sampler:26s} AIC={r.aic:9.2f} BIC={r.bic:9.2f} conv={r.info.get('converged')}")
```

Measured on one NVIDIA GPU, 53 g-band points, `kilonova_two_jax`, float64:

| sampler | wall clock | AIC | BIC | `converged` |
|---|---:|---:|---:|---|
| `abc_gpu` | 18.1 s | 260.49 | 276.25 | n/a |
| `abc_smc_gpu` | 17.2 s | 1370.26 | 1386.02 | n/a |
| `emcee_jax` | 26.7 s | −7.98 | 7.78 | False |
| `snpe` | 93.2 s | 57.10 | 72.86 | n/a |
| `nuts_gpu` | 396.8 s | −12.07 | 3.69 | False |
| `pymc_jax_gpu_vectorized` | 856.8 s | **−33.23** | **−17.47** | False |

### Do not read that table as a sampler ranking

Two things in it are traps.

**ABC's AIC is not comparable with a likelihood-based sampler's.** ABC never evaluates the
likelihood — it accepts draws whose simulated light curve is *close enough* under a distance. The
AIC you get back is computed from whichever accepted draw happened to be best, so `abc_gpu`'s 260
and `emcee_jax`'s −8 are not measuring the same thing. Compare ABC with ABC, and likelihood samplers
with likelihood samplers.

**`converged=False` on every MCMC row.** At these budgets none of the chains passed the r-hat test,
which means the intervals are not yet trustworthy even though the point estimates look sensible.
This is a documentation example sized to finish in minutes, not a publication run. Raise
`num_samples` / `nsteps` until `converged` is `True`.

Where two comparable samplers *can* be checked against each other, they agree:

| parameter | `emcee_jax` median | `pymc_jax_gpu_vectorized` median |
|---|---:|---:|
| `mej_1` | 0.0295 | 0.0293 |
| `vej_1` | 0.2526 | 0.2301 |
| `kappa_1` | 0.1653 | 0.1398 |
| `mej_2` | 0.0258 | 0.0245 |
| `vej_2` | 0.1959 | 0.1381 |
| `kappa_2` | 1.0261 | 1.3319 |

### The prior is not a detail — it is most of the answer

The identical `abc_gpu` run, changing **only** the prior:

| prior on `kappa_1` | AIC |
|---|---:|
| `[1, 30]` — the shape of the model's own default | 20006.76 |
| `[0.1, 0.5]` — a real blue component | **260.49** |

The model's default prior gives *both* components `kappa ∈ [1, 30]`, so it cannot represent
low-opacity blue ejecta — and early g-band emission from a kilonova is exactly that. Searching 400
random draws from each box, the best χ² was 38,450 under the default versus 1,574 with the split, a
**24× difference**, and only the split reaches the observed brightness at all. If you take one thing
from this page: **check that your prior can express the physics before blaming the sampler.**

### Check whether the posterior is railed

`mej_1` comes back at 0.0293–0.0295 against a prior ceiling of 0.03, and `kappa_1` at 0.14–0.17
against a floor of 0.1. Both are pressed against their bounds, which means the prior — not the data
— is setting them. The honest response is to widen those bounds and refit, and to say so in the
paper. A posterior median sitting on a prior edge is a result about your prior.

## 4. One component or two?

The question a kilonova paper actually asks. Same sampler, same data, same space — only the model
changes:

```python
for label, model, pr in [("one-component", one, None), ("two-component", two, prior)]:
    r = wp.fit(lc, model.name, sampler="emcee_jax", prior=pr, space="magnitude", seed=0,
               nwalkers=32, nsteps=3000, burnin=1000, thin=5)
    print(f"{label:16s} k={len(model.parameters)}  AIC={r.aic:9.2f}  BIC={r.bic:9.2f}")
```

Measured, `emcee_jax`, same 53 points, `space="magnitude"`:

| model | k | AIC | BIC |
|---|---:|---:|---:|
| one-component | 4 | 146.23 | 154.11 |
| two-component | 8 | **−7.98** | **7.78** |

ΔAIC = **154.2** in favour of two components. Anything above ~10 is decisive, so the data clearly
wants two components — it pays for four extra parameters many times over. BIC agrees.

### Give both models a fair prior, or the answer is meaningless

The first time this page was written, the one-component model was run with its **default** prior
(`kappa ∈ [1, 30]`, which cannot be blue) while the two-component model got the personalised prior
above. That produced ΔAIC = **13,681** — a result that looks overwhelming and is largely an artefact
of handicapping one side.

Rerunning with `kappa ∈ [0.1, 30]` for the one-component model, so it can reach the blue solution
too, gives the honest **154.2**. Same conclusion, but the inflated figure was **89× too large**.

When you compare models, give each one the prior it deserves. A comparison where one model was
denied the parameter range it needs is not a comparison.

> The fair one-component fit lands at `mej` = 0.030 (ceiling 0.03), `kappa` = 0.111 (floor 0.1),
> `temperature_floor` = 5996 K (ceiling 6000) — railed against three bounds at once. That is the
> model straining to do a job it cannot do, which is itself the physical answer here: one component
> cannot reproduce a blue *and* a red phase.

The two-component model has twice the parameters, so it will always fit at least as well in raw
likelihood. AIC and BIC ask whether it fits *enough* better to justify them — BIC more harshly.

## 5. Checks worth doing before you believe it

```python
print(r.info["converged"], r.info.get("max_rhat"), r.info.get("n_divergences"))
```

* **`converged`** — `False` means the rank statistic and the credible intervals are not
  interpretable, whatever the AIC says. `None` means the sampler has no such diagnostic (SNPE has no
  chains, so nothing to converge).
* **Divergences** — for the NUTS-family samplers, a nonzero count means the geometry defeated the
  integrator somewhere. Curiously, zero divergences with a high `r-hat` is worse news than a few
  divergences: it usually means the chains never explored at all.
* **Overlay the posteriors.** If two samplers targeting the same posterior disagree visibly, at
  least one has not converged:

```python
wp.plot_corner([results["emcee_jax"], results["pymc_jax_gpu_vectorized"]], save="overlay.png")
wp.plot_ppc(results["emcee_jax"], lc, model=two.name, save="ppc.png")
```

* **AIC / BIC / WAIC compare models, not spaces.** Never compare a `space="flux"` number with a
  `space="magnitude"` one.

## Next

* [`CHOOSING.md`](CHOOSING.md) — which sampler for which problem, and combinations that cannot work.
* [`GPU_SETUP.md`](GPU_SETUP.md) — CUDA, float64 cost, multi-GPU.
