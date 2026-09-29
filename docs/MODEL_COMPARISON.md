# Model comparison, on CPU and on GPU

[Part 0](#part-0--one-call-wpcompare) is the one-call comparison, `wp.compare`, and the rules it
ranks by. The rest is a complete worked example done by hand: fit a **multi-component kilonova** to
AT2017GFO with a **personalised prior**, using each available sampler, then compare it against a
simpler model.

The task is the same on both backends and so is the arithmetic. What differs is which samplers you
have, how the models are built, and — on the GPU — two setup steps that cost you silently if you
skip them. Read [Part 1](#part-1--on-cpu) or [Part 2](#part-2--on-gpu) depending on what you have;
[Part 3](#part-3--reading-the-numbers) applies to both.

**The rule that governs every comparison below:** AIC, BIC and WAIC compare *models*, not *spaces*.
Numbers from a flux-space fit are not comparable with numbers from a magnitude-space fit. Keep
`space=` identical across everything you put in one table, and only compare fits that converged.

---

## Part 0 — one call: `wp.compare`

```python
import numpy as np
import whisper_cbpf as wp

t = np.linspace(0.5, 30.0, 30)                                   # simulated from flare
truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
flux = wp.get_model("flare").predict(truth, t) + np.random.default_rng(1).normal(0.0, 0.1, 30)
toy = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
cmp = wp.compare(toy, ["flare", "bazin", "gaussian_rise"], sampler="mcmc")
print(cmp.summary())        # 'flare' first, "decisive" over 'bazin' (ln B = 10.34); all converged
```

`compare` does, for every model, what Parts 1 and 2 do by hand, and keeps the rule above for you:
the same light curve, the same sampler, the same settings and the same space for every model. Then:

1. **It finds each fit's likelihood peak** (`wp.likelihood_max_opt`). AIC and BIC are defined at
   the maximum likelihood; a sampler's best draw is below it, by an amount that differs between models and runs
   (on a simulated LSST alert, 0.005 ln L for the 6-parameter Arnett model and 0.3-1 ln L for the
   9-11-parameter families: [`VALIDATION.md`](VALIDATION.md) section 5.3). `compare` takes AIC
   and BIC from the peak; the posterior is not changed. `cmp.peaks[model].gain` is how far the
   sampler was from it.
2. **It checks each fit** (`result.diagnostics()`): `converged` in the table is the report's verdict,
   and `problems` lists every failed check, a likelihood maximum more than 1 ln L above the
   sampler's best draw, and a peak on a prior edge.
3. **It ranks.** By ln Z when every ranked model has a nested-sampling evidence that stopped on
   dlogz (`sampler="nested"`), otherwise by BIC at the likelihood maximum. `delta` is the gap to
   the best,
   `weight` is `exp(-delta/2)` normalised over the ranked models, and `grade` is Jeffreys' scale on
   ln B (`delta / 2` under BIC): below 1.15 inconclusive, below 2.30 substantial, below 4.61 strong,
   decisive above. The winner is graded against the runner-up.
4. **It leaves out** what cannot be compared, with the reason: a failed fit, no draws, a fit to
   other rows or in another space than the most common ones, `k >= n` ("not enough data"), no
   finite BIC.
5. **It checks a close call.** When the top two are within ln B < ln 10, `evidence_check="auto"` runs
   nested sampling on them; if ln Z disagrees with BIC, the winner's grade becomes "inconclusive".
   ln Z weighs the priors as well as the fits: when the ln Z gap is much larger than the gap between
   the two likelihood peaks, the summary says it comes from the priors, and it is then only as
   meaningful as they are. A model behind a constraint wall (the supernova and TDE families) has
   its prior renormalised to the allowed region, so the wall does not cost it evidence
   (`info["constraint_prior"]` of its nested fit).

`cmp.table` has every number; `cmp.save(dir)` / `wp.Comparison.load(dir)`, `cmp.facts()`,
`cmp.report(path)`, `cmp.forecast(times, bands)` and `cmp.discriminate(times, bands)` take it
further. Pass `likelihood_max_opt=False` to rank on the samplers' best draws instead, and `cache_dir=` to resume
an interrupted comparison. For survey alerts and physical families (`"arnett"`, `"magnetar"`,
`"tde"`, ...), see [`LSST_ALERTS.md`](LSST_ALERTS.md); every argument is in
[`API_REFERENCE.md` A5](API_REFERENCE.md#a5-compare-models).

---

## Part 1 — on CPU

Everything in this part runs on the base install plus `[models]`:

```bash
pip install "whisper-cbpf[models] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

No GPU, no JAX, no CUDA.

### The samplers you get on CPU

| Sampler | Needs | What it gives you |
|---|---|---|
| `mcmc` | — | emcee ensemble MCMC. The sensible default. |
| `nested` (alias `dynesty`) | — | nested sampling — the only CPU sampler that returns **log-evidence**, so the only one that gives you Bayes factors. |
| `abc` | — | likelihood-free rejection ABC. Works when a likelihood is awkward to write. |
| `abc_smc` | — | sequential ABC; reaches the same tolerance in far fewer simulations than flat `abc`. |
| `snpe` / `npe` | `[sbi]` | neural simulation-based inference. |

`wp.list_samplers()` shows all 14 registered names; the GPU ones are listed too and tell you which
extra they need if you ask for them.

### 1. Load the data

```python
import whisper_cbpf as wp

wp.set_default_band_system("sdss")      # this file's bare g, r, i are SDSS photometry
lc = wp.load_lightcurve("tests/data/at2017gfo.csv", redshift=0.0098)
lc = lc.select_bands(["g"]).set_explosion_date(57982.529)   # merger epoch, MJD
lc = lc.select_time_window(0.0, 10.0)
print(lc.n_points)      # 53
```

A bare band letter means LSST unless you say otherwise ([`PHOTOMETRY.md`](PHOTOMETRY.md) section 2),
and AT2017GFO's `g` is SDSS g: without the first line the physical model below would integrate its
SED over the LSST g filter.

> **Order matters.** `set_explosion_date` must come *before* `select_time_window`. The window
> filters whatever `time` currently holds, so windowing first compares days-since-explosion bounds
> against raw MJD (~57983) and silently returns an **empty** light curve.

### 2. Build a personalised prior

The model's default prior is redback's own, and it is deliberately wide. If you know something about
your event — a measured redshift, a plausible ejecta mass — encode it. A tighter, honest prior is
the cheapest way to make sampling converge.

`two_component_kilonova` takes nine parameters: four per component plus redshift. Component 1 is
conventionally the **blue**, lanthanide-poor ejecta (low `kappa`) and component 2 the **red**.

```python
prior = wp.Prior({
    # --- component 1: blue, lanthanide-poor ---
    "mej_1":               wp.Uniform(0.001, 0.05,   name="mej_1"),
    "vej_1":               wp.Uniform(0.10,  0.40,   name="vej_1"),
    "kappa_1":             wp.Uniform(0.1,   0.5,    name="kappa_1"),
    "temperature_floor_1": wp.LogUniform(1000., 5000., name="temperature_floor_1"),
    # --- component 2: red, lanthanide-rich ---
    "mej_2":               wp.Uniform(0.001, 0.10,   name="mej_2"),
    "vej_2":               wp.Uniform(0.05,  0.25,   name="vej_2"),
    "kappa_2":             wp.Uniform(1.0,   10.0,   name="kappa_2"),
    "temperature_floor_2": wp.LogUniform(100.,  2000., name="temperature_floor_2"),
    # --- known from the GW event, so pinned tight ---
    "redshift":            wp.Uniform(0.009, 0.011,  name="redshift"),
})
```

Two things worth noticing.

**The `kappa` ranges are disjoint** — `[0.1, 0.5]` and `[1.0, 10.0]`. redback's default gives both
components the *same* `kappa` range, which makes them exchangeable: the sampler can swap component 1
and component 2 and the likelihood is unchanged. That is label switching, and it produces a bimodal
posterior that no amount of sampling fixes. Disjoint corridors remove it by construction.

**`temperature_floor` uses `LogUniform`.** It spans decades, so a `Uniform` would put half its prior
mass above 3000 K. Use `LogUniform` for any scale parameter.

### 3. Fit with each sampler

Pass `prior=` to any sampler. Keep `space=` identical everywhere you intend to compare.

```python
runs = [
    ("abc",     dict(n_simulations=2000, quantile=0.05)),
    ("abc_smc", dict(n_particles=200, n_rounds=3)),
    ("mcmc",    dict(nsteps=1500, burnin=500, nwalkers=24, thin=5)),
    ("nested",  dict(nlive=100, dlogz=1.0, maxcall=40000)),   # capped -- see below
]

results = {}
for sampler, kwargs in runs:
    r = wp.fit(lc, "two_component_kilonova", sampler=sampler, prior=prior,
               space="magnitude", seed=0, **kwargs)
    results[sampler] = r
    print(f"{sampler:8s} AIC={r.aic:9.2f} BIC={r.bic:9.2f}")
```

These are deliberately small budgets so the example finishes in minutes. For a result you intend to
publish, raise them and **check `r.info["converged"]`** before quoting anything.

Measured on CPU (one process, a shared 96-core host), 53 SDSS g points, `two_component_kilonova`
(redback), 9 parameters:

| sampler | wall clock | AIC | BIC | ln Z | `converged` |
|---|---:|---:|---:|---:|---|
| `abc_smc` | 10.2 s | 934.82 | 952.55 | — | n/a |
| `abc` | 12.1 s | 599.47 | 617.21 | — | n/a |
| `mcmc` | 205.1 s | **−26.22** | **−8.49** | — | False |
| `nested` | 140.9 s | −21.32 | −3.59 | 0.17 (capped: a lower bound) | False |

#### Budget these fits from a measured cost, not a guess

One `two_component_kilonova` prediction on these 53 points costs:

* **1.5 s on the first call** — redback and sncosmo build their grids once.
* **4.9 ms warm** (median of 20).

Multiply by the number of likelihood evaluations before you start. `mcmc` at 24 walkers × 1500 steps
is 36,000 evaluations ≈ 3 minutes. Nested sampling on 9 parameters needs far more calls than a
back-of-envelope suggests — the run above is capped with `maxcall=40000` for exactly that reason,
which is why it did not converge and its ln Z is only a lower bound.

#### Why `nested`'s AIC is worse while its medians agree

`mcmc` reports AIC −26.22 and `nested` −21.32, and the two posteriors' medians are close, except
`mej_2`, which neither capped run pins down:

| parameter | `mcmc` | `nested` |
|---|---:|---:|
| `mej_1` | 0.0404 | 0.0471 |
| `vej_1` | 0.2921 | 0.3013 |
| `kappa_1` | 0.1731 | 0.1423 |
| `mej_2` | 0.0700 | 0.0416 |
| `kappa_2` | 1.4266 | 1.3079 |
| `redshift` | 0.0094 | 0.0100 |

A sampler's AIC is computed from its **best** draw, so a capped run that never reached the
likelihood peak scores worse even when its *typical* draws are in the right place.
`wp.likelihood_max_opt(r, lc)` climbs from those draws to the peak and recomputes AIC and BIC there (Part 0); `compare` does it for
every model. Both reported `converged=False` at these deliberately small budgets. Two conclusions
follow, and both matter more than the AIC: compare AIC only between fits of comparable quality, and
raise the budget until `converged` is `True` before quoting anything.

**ABC's AIC is not comparable with `mcmc`'s at all.** ABC never evaluates the likelihood; it accepts
draws whose simulated curve is close enough under a distance, and the reported AIC comes from
whichever accepted draw was best. Compare ABC with ABC.

#### A cross-check, and what it can and cannot tell you

Part 2 fits a *different implementation* of the same physics — `kilonova_two_jax`, with its
parameters in another order, no free redshift and Villar's default prior instead of the prior
above. On these 53 SDSS g points the likelihood maxima are ln L 22.46 for this page's `mcmc` fit
(AIC −26.92, 9 parameters) and 27.79 for an `emcee_jax` fit of `kilonova_two_jax` at Part 2's
settings (AIC −39.57, 8 parameters), both measured on a CPU. That gap does not show that the
implementations disagree: the two fits search different prior boxes with different parameters (a
free redshift on one side only), so their peaks need not coincide. To check that two implementations
compute the same light curve, compare them at the
same parameters, `wp.check_parity(model_a, model_b, params, times, bands)`
([`API_REFERENCE.md` A11](API_REFERENCE.md#a11-check-that-two-models-agree)); compare their fits
only under the same prior.

### 4. Compare against a simpler model

Model comparison needs at least two models on the *same data, in the same space, with the same
sampler and the same budget*. Change any of those and you are measuring the change, not the models.

```python
BUDGET = dict(nsteps=1500, burnin=500, nwalkers=24, thin=5)

for name in ["two_component_kilonova", "gaussian_rise", "bazin"]:
    pr = prior if name == "two_component_kilonova" else None
    r = wp.fit(lc, name, sampler="mcmc", prior=pr, space="magnitude", seed=0, **BUDGET)
    print(f"{name:24s} AIC={r.aic:9.2f}  BIC={r.bic:9.2f}")
```

Measured:

| model | k | AIC | BIC | wall clock |
|---|---:|---:|---:|---:|
| `two_component_kilonova` | 9 | **−26.22** | **−8.49** | 222.8 s |
| `gaussian_rise` | 4 | 275.51 | 283.39 | 3.3 s |
| `bazin` | 4 | 277.52 | 285.40 | 3.1 s |

ΔAIC ≈ **302** in favour of the physical model over the best empirical one — decisive, and it earns
its five extra parameters many times over. It also costs about 70× more wall clock, which is the
real trade.

> **An AIC from the best draw depends on where the chain stopped.** None of these fits passed its
> convergence report at this budget. Here the best draws came close to the peak — `bazin` scores
> 277.52 at `nsteps=1500` and 277.49 at `nsteps=2000`, and its optimised likelihood maximum is
> 277.42 at both — but a chain that has not reached the peak gives an AIC that is not a property of
> the model. Check `r.diagnostics()` before you compare anything, and take AIC and BIC from the
> likelihood maximum (`wp.likelihood_max_opt`, or `wp.compare`, Part 0), which does not depend on where the chain happened to
> stop.

---

## Part 2 — on GPU

### Before any Python: the two things that silently cost you

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

### The samplers you get on GPU

| Sampler | What it gives you |
|---|---|
| `emcee_jax` | emcee's ensemble moves, vectorised across walkers on the GPU. No gradients needed. |
| `pymc_jax_gpu_vectorized` | NUTS through a PyMC front end, `vmap` over chains. |
| `pymc_jax_gpu_parallelized` | the same, with true `pmap` — fastest per chain across several cards. |
| `abc_gpu` / `abc_smc_gpu` | ABC with simulation batched on the GPU; enormous simulation counts become cheap. |
| `nuts_gpu` | NumPyro NUTS directly. **Read the caveat below.** |
| `snpe` | neural simulation-based inference (needs `[sbi]`); `device="cuda"` to train on the GPU. |

> ### Caveat: local samplers and local optima — check `r.info["converged"]`
>
> NUTS (`nuts_gpu`, `pymc_jax_gpu_*`) and emcee only move locally, and these likelihoods have
> optima a chain cannot leave: on the one-component kilonova, chains parked on a corner of the prior
> box (typically `mej` at its upper bound and `vej` at its lower) 972-65 618 log-units below the
> best chain; on flux models, a chain shrank the model until it hid between the epochs. This is a
> property of local samplers on these likelihoods, on CPU and GPU alike, not a `nuts_gpu` coding
> bug — `pymc_jax_gpu_vectorized` and `emcee_jax` failed the same way on the same mock problems.
> (This caveat replaces a warning that `nuts_gpu` returned "confidently wrong" kilonova
> posteriors — the truth outside the whole posterior, R-hat ≈ 1, no divergences — for an unknown
> reason. The cause was most likely a log-uniform parameter walked in linear coordinates, so no
> chain could start near a low truth: 42ea833 fixed that and reproduced the signature on the GPU,
> but the report's own simulations were not re-run, so the link is likely rather than proven.)
>
> What changed: each chain now **starts** at a distinct prior draw that was scored and climbed
> into the best basin found (`init_strategy="prior_scan"`, the default; `init=` for `emcee_jax`),
> and each fit **checks itself** — stranded, frozen or stuck chains, rank-normalised R-hat on
> every parameter and on the log-likelihood, ESS, divergences, and whether any chain reached the
> best optimum the scan found. On the one-component kilonova mocks, broken runs fell from 7 / 13
> to 0 on the CPU and from 3 / 4 to 0 / 13 on the GPU (`nuts_gpu`), and from 3 / 3 to 0 / 13
> for `pymc_jax_gpu_vectorized` on the same GPU simulations, whose posteriors then agree with
> `nuts_gpu`'s to 0.07 sd (median; at most 0.25 sd). With the old start every broken run was
> flagged. At the reduced 500 / 500 budget of that test 9 of 13 fits of each sampler still read
> `converged=False` — for R-hat and divergences, which a longer run and a higher
> `target_accept_prob` address — and say so. In float32 (JAX's default) an MJD-scale clock or
> prior bound freezes chains; the samplers warn before sampling, and float64 as above avoids it. A
> start can still miss a basin no prior draw reached, so **read `r.info["convergence_problems"]`**
> (empty when `converged` is `True`) before believing a posterior, from any of these samplers.

### 1. Register the JAX models

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

### 2. The default prior is already the physical one

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

### 3. Load the data and fit

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

#### Do not read that table as a sampler ranking

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

#### Why the default prior is Villar's and not redback's

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
apples-to-apples comparison against redback itself:
`kilonova_two_model(..., prior=kilonova_two.default_prior())`. Its names are redback's
(`mej_1 … kappa_2`); the factory maps them onto its own through `Model.param_aliases`.

#### Check whether the posterior is railed

Look at where the medians sit relative to their bounds. Here `mej_red` comes back at 0.048 against a
ceiling of 0.1 and `kappa_blue` at 0.19 against a floor of 0.1 — comfortable. If a median sits *on*
a bound, the prior is setting it, not the data. Widen and refit, and say so in the paper: a
posterior median on a prior edge is a result about your prior.

### 4. One, two, or three components?

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

#### Give every model a prior it can actually use

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

---

## Part 3 — reading the numbers

### Reading the numbers

* **AIC / BIC — lower is better.** BIC penalises extra parameters harder, so it favours simpler
  models more aggressively than AIC. A two-component kilonova has 9 parameters against `bazin`'s 4;
  BIC will make it earn them. Both are defined at the likelihood peak: take them from
  `wp.likelihood_max_opt(r, lc)` (or `wp.compare`), not from the sampler's best draw, which falls further short
  of the peak for the model with more parameters.
* **Weights and grades.** `exp(-dBIC/2)`, normalised, is each model's weight; `dBIC/2` is the log
  Bayes factor BIC approximates, graded on Jeffreys' scale (1.15 / 2.30 / 4.61 for substantial /
  strong / decisive). `wp.compare` reports both.
* **Log-evidence — higher is better.** Only `nested` returns it, in
  `r.info["log_evidence"]` with `r.info["log_evidence_err"]`. The difference between two models'
  `ln Z` is the log Bayes factor.
* **AIC, BIC and WAIC compare models, not spaces.** A flux-space fit and a magnitude-space fit of
  the *same* model differ by a fixed Jacobian term. Never compare across `space=`.
* **A WAIC flagged unreliable is not a number.** Check
  `r.info["predictive_metrics"]["waic"]["p_waic_reliable"]` before quoting WAIC. A blown-up `p_waic`
  usually means chains stranded away from the bulk, not a bad model.

### Look at the fit, do not just read the number

An information criterion cannot tell you the fit is nonsense. Plot it:

```python
best = results["nested"]
wp.plot_ppc(best, lc, model="two_component_kilonova", save="ppc.png")
wp.plot_corner([best], save="corner.png")
wp.plot_light_curve(lc, save="data.png")
```

To overlay several samplers on one corner and see whether they agree, pass a list:

```python
wp.plot_corner([results["mcmc"], results["nested"]], save="overlay.png")
```

If two samplers targeting the same posterior disagree visibly, at least one has not converged —
that is worth more than any AIC.


### Checks worth doing before you believe it

```python
print(r.diagnostics())                             # every check, pass or fail, and why
print(r.info["converged"], r.info.get("max_rhat"), r.info.get("n_divergences"))
print(r.info.get("convergence_problems"))          # one sentence per failed check
```

* **`converged`** — `False` means the rank statistic and the credible intervals are not
  interpretable, whatever the AIC says. For `nuts_gpu`, `pymc_jax_gpu_*`, `emcee_jax` and `mcmc`,
  `convergence_problems` names the reason — a stranded or frozen chain, stuck walkers, R-hat, ESS,
  divergences, too short a chain — and the fit warned with the same text. For `snpe` / `snpe_gpu`
  it means the final posterior draw needed no MCMC fallback (`info["leakage"]` says whether the flow
  put its mass outside the prior box); for `nested`, that the run stopped before its iteration cap.
  The ABC samplers have no such diagnostic and record no `converged` (use `r.info.get(...)`).
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

---

## Next

* [`docs/CHOOSING.md`](CHOOSING.md) — which sampler, and which combinations cannot work.
* [`docs/PORTING_NOTES.md`](PORTING_NOTES.md) — where the JAX models differ from redback.
* [`notebooks/08_metrics_and_comparison.ipynb`](../notebooks/08_metrics_and_comparison.ipynb) —
  the same material as a runnable notebook.
