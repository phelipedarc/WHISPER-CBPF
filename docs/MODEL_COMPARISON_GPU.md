# Model comparison on GPU

The same job as [`MODEL_COMPARISON_CPU.md`](MODEL_COMPARISON_CPU.md) — fitting and comparing
multi-component kilonovae — but with the JAX models and GPU samplers, where larger budgets become
affordable. Here the defaults are already the physically motivated Villar+2017 priors, so the page
also answers the question a kilonova paper asks: **one component, two, or three?**

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
> two_component_kilonova (redback, CPU):  mej_1,    vej_1,    kappa_1,    temperature_floor_1,    ... + redshift  (9)
> kilonova_two_jax       (JAX,     GPU):  mej_blue, vej_blue, temperature_floor_blue, kappa_blue, ...             (8)
> ```
>
> Three differences, any one of which silently changes what you fit: the names differ, `kappa` and
> `temperature_floor` are **in the opposite order**, and the JAX model has no `redshift` parameter —
> it is fixed when you register it. Always build the prior from `model.parameters`.

## 2. The default prior is already the physical one

You do not have to build a prior to get a sensible fit. The JAX kilonovae ship with
**Villar et al. (2017)** priors, and the thing that makes them physical is that each component gets
its own **disjoint opacity corridor**:

| component | `kappa` prior | Villar's fixed value |
|---|---|---|
| blue (lanthanide-poor) | U(0.1, 1.0) | 0.5 cm² g⁻¹ |
| purple (three-component only) | U(1.0, 5.0) | 3 cm² g⁻¹ |
| red (lanthanide-rich) | U(5.0, 30.0) — U(1.0, 30.0) when there is no purple | 10 cm² g⁻¹ |

**The label is cosmetic; the corridor is what makes a component blue, purple or red.** Keeping them
disjoint also removes label switching at the source: with identical per-component priors the
posterior is exactly symmetric under relabelling, every mode has a mirror twin, chains lock into
different labellings, and `r-hat` then measures which labelling each chain fell into rather than
convergence. Disjoint corridors mean there is nothing to relabel.

Inspect what you are about to fit:

```python
for n in two.parameters:
    d = two.default_prior.distributions[n]
    print(f"{n:24s} {type(d).__name__}({d.low}, {d.high})")
```

To personalise it — a measured redshift, a mass range you trust — build a `Prior` **in the model's
own parameter order** and pass `prior=`:

```python
prior = wp.Prior({
    "mej_blue":               wp.Uniform(1e-3, 0.1,      name="mej_blue"),
    "vej_blue":               wp.Uniform(0.03, 0.40,     name="vej_blue"),
    "temperature_floor_blue": wp.LogUniform(100., 5000., name="temperature_floor_blue"),
    "kappa_blue":             wp.Uniform(0.1,  1.0,      name="kappa_blue"),
    "mej_red":                wp.Uniform(1e-3, 0.1,      name="mej_red"),
    "vej_red":                wp.Uniform(0.01, 0.30,     name="vej_red"),
    "temperature_floor_red":  wp.LogUniform(100., 5000., name="temperature_floor_red"),
    "kappa_red":              wp.Uniform(1.0,  30.0,     name="kappa_red"),
})

assert list(prior.names) == list(two.parameters)     # cheap, and worth it
```

If you narrow a bound, check afterwards that the posterior has not simply moved onto it — see
[Check whether the posterior is railed](#check-whether-the-posterior-is-railed).

> **`sigma` is not a model parameter.** Villar's white-noise term belongs to the likelihood. Fit it
> with `likelihood="scatter"`; adding it as a prior column the model cannot consume is refused by
> the density adapter, deliberately.

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
    r = wp.fit(lc, two.name, sampler=sampler, space="magnitude", seed=0, **kwargs)
    results[sampler] = r
    print(f"{sampler:26s} AIC={r.aic:9.2f} BIC={r.bic:9.2f} conv={r.info.get('converged')}")
```

Measured on one NVIDIA GPU, 53 g-band points, `kilonova_two_jax`, float64, **default prior**:

| sampler | wall clock | AIC | BIC | `converged` |
|---|---:|---:|---:|---|
| `abc_gpu` | 15.1 s | 416.24 | 432.00 | n/a |
| `abc_smc_gpu` | 15.9 s | 1489.51 | 1505.27 | n/a |
| `emcee_jax` | 21.3 s | −30.64 | −14.88 | False |
| `snpe` | 100.5 s | 52.26 | 68.02 | n/a |
| `pymc_jax_gpu_vectorized` | 770.8 s | **−37.65** | **−21.89** | False |

### Do not read that table as a sampler ranking

Two things in it are traps.

**ABC's AIC is not comparable with a likelihood-based sampler's.** ABC never evaluates the
likelihood — it accepts draws whose simulated light curve is *close enough* under a distance, so the
AIC comes from whichever accepted draw happened to be best. Compare ABC with ABC.

**`converged=False` on both MCMC rows.** These budgets are sized to finish in minutes, not to
converge. And you can see the cost directly:

| parameter | `emcee_jax` | `pymc_jax_gpu_vectorized` |
|---|---:|---:|
| `mej_blue` | 0.0388 | 0.0461 |
| `vej_blue` | 0.2882 | 0.3701 |
| `kappa_blue` | 0.1886 | 0.2731 |
| `temperature_floor_blue` | **4462** | **825** |
| `mej_red` | 0.0477 | 0.0347 |
| `vej_red` | 0.1326 | 0.2178 |
| `kappa_red` | 1.6293 | 1.1921 |
| `temperature_floor_red` | **520** | **3443** |

Masses and opacities agree to a few tens of percent. The **temperature floors do not** — they are
almost exchanged between the two fits. Disjoint opacity corridors stop the *components* swapping,
but they do not make every other direction unimodal, and neither chain has converged. This is what
`converged=False` looks like from the outside, and it is why the flag is worth more than the AIC:
raise the budget until it is `True` before quoting an interval.

### Why the default prior is Villar's and not redback's

redback's two-component prior gives **both** components `kappa` U(1, 30), so it cannot represent
low-opacity blue ejecta at all — and early g-band emission is exactly that. It also cannot contain
Villar's published AT2017GFO solution: `kappa_blue = 0.5` is below its floor and `M_ej,red = 0.050`
is above its `Uniform(0.01, 0.03)` ceiling, so a fit inside it rails on both.

Measured on this dataset, searching 400 random draws from each box:

| prior | best χ² (53 points) | reduced χ² |
|---|---:|---:|
| redback's, both `kappa` U(1, 30) | 38450.8 | 725.5 |
| Villar's, `kappa_blue` U(0.1, 1.0) | **1574.3** | **29.7** |

A factor of 24, and only the Villar box reaches the observed brightness. redback's prior is still
available as `kilonova_two.default_prior()` — it is the right choice for one job only, an
apples-to-apples comparison against redback itself.

### Check whether the posterior is railed

Look at where the medians sit relative to their bounds. Here `mej_red` comes back at 0.048 against a
ceiling of 0.1 and `kappa_blue` at 0.19 against a floor of 0.1 — comfortable. If a median sits *on*
a bound, the prior is setting it, not the data. Widen and refit, and say so in the paper: a
posterior median on a prior edge is a result about your prior.

## 4. One, two, or three components?

The question a kilonova paper actually asks. Same sampler, same data, same space — only the model
changes:

```python
one   = wp.register_kilonova(["sdssg"], redshift=REDSHIFT, dl_cm=DL_CM, n_wave=300,
                             band_aliases={"g": "sdssg"})
three = wp.register_kilonova_three(["sdssg"], redshift=REDSHIFT, dl_cm=DL_CM, n_wave=300,
                                   band_aliases={"g": "sdssg"})

for label, model in [("one", one), ("two", two), ("three", three)]:
    r = wp.fit(lc, model.name, sampler="emcee_jax", space="magnitude", seed=0,
               nwalkers=32, nsteps=3000, burnin=1000, thin=5)
    print(f"{label:6s} k={len(model.parameters):2d}  AIC={r.aic:9.2f}  BIC={r.bic:9.2f}")
```

Measured:

| model | k | AIC | BIC |
|---|---:|---:|---:|
| one-component | 4 | 26.93 | 34.81 |
| **two-component** | 8 | **−30.64** | **−14.88** |
| three-component | 12 | −23.14 | 0.50 |

**Two components win, and three do not.** ΔAIC = 57.6 for two over one — decisive, and worth the
four extra parameters. But going to three *costs* 7.5 AIC and 15.4 BIC: the extra component buys no
fit and is charged for anyway.

That is not a contradiction of Villar+2017, which preferred three. They had UV through near-infrared
coverage; this is **one band over ten days**. A purple component is distinguished from blue and red
mainly by where it emits, so a single filter cannot separate it. The lesson is the general one:
*model complexity you cannot constrain is complexity you should not fit* — and an information
criterion will tell you so if you ask it.

### Give every model a prior it can actually use

The one-component number above is **26.93** because it was given `kappa` U(0.1, 30), spanning both
regimes. On its own default prior — `kappa` U(1, 30) — the same fit returns **13673.37**, and ΔAIC
against two components inflates from 57.6 to 13704, a factor of 238.

**That is not a defect in the one-component model's prior.** A single component has nothing to be
distinguished *from*, so it correctly gets the broad opacity range; disjoint corridors only make
sense once there are two components to keep apart. Widening it here is a **device for a like-for-like
comparison**, not a fix — for a one-component fit on its own, the default is the right prior.

The lesson is about the comparison, not the model: a candidate denied the parameter range it needs
cannot compete, and the inflated margin is the one that looks most impressive. Whenever you compare
models, check that each can actually reach the solution.

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
