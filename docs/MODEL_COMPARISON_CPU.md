# Model comparison on CPU

A complete worked example: fit the **two-component kilonova** to AT2017GFO with a **personalised
prior**, using each CPU sampler, then compare it against a simpler model.

Everything here runs on the base install plus `[models]`:

```bash
pip install "whisper-cbpf[models] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

No GPU, no JAX, no CUDA.

## The samplers you get on CPU

| Sampler | Needs | What it gives you |
|---|---|---|
| `mcmc` | — | emcee ensemble MCMC. The sensible default. |
| `nested` (alias `dynesty`) | — | nested sampling — the only CPU sampler that returns **log-evidence**, so the only one that gives you Bayes factors. |
| `abc` | — | likelihood-free rejection ABC. Works when a likelihood is awkward to write. |
| `abc_smc` | — | sequential ABC; reaches the same tolerance in far fewer simulations than flat `abc`. |
| `snpe` / `npe` | `[sbi]` | neural simulation-based inference. |

`wp.list_samplers()` shows all 13 registered names; the GPU ones are listed too and tell you which
extra they need if you ask for them.

## 1. Load the data

```python
import whisper_cbpf as wp

lc = wp.load_lightcurve("tests/data/at2017gfo.csv", redshift=0.0098)
lc = lc.select_bands(["g"]).set_explosion_date(57982.529)   # merger epoch, MJD
lc = lc.select_time_window(0.0, 10.0)
print(lc.n_points)      # 53
```

> **Order matters.** `set_explosion_date` must come *before* `select_time_window`. The window
> filters whatever `time` currently holds, so windowing first compares days-since-explosion bounds
> against raw MJD (~57983) and silently returns an **empty** light curve.

## 2. Build a personalised prior

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

## 3. Fit with each sampler

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

Measured on CPU, 53 g-band points, `two_component_kilonova` (redback), 9 parameters:

| sampler | wall clock | AIC | BIC | ln Z | `converged` |
|---|---:|---:|---:|---:|---|
| `abc_smc` | 12.1 s | 745.33 | 763.06 | — | n/a |
| `abc` | 33.3 s | 719.46 | 737.20 | — | n/a |
| `mcmc` | 230.5 s | **−3.06** | **14.67** | — | False |
| `nested` | 464.0 s | 186.07 | 203.80 | **−96.22** | False |

### Budget these fits from a measured cost, not a guess

One `two_component_kilonova` prediction on these 53 points costs:

* **97.6 s on the first call** — redback and sncosmo build their grids once. Your first fit appears
  to hang for a minute and a half. It has not.
* **14.7 ms warm** (median of 20).

Multiply by the number of likelihood evaluations before you start. `mcmc` at 24 walkers × 1500 steps
is 36,000 evaluations ≈ 9 minutes. Nested sampling on 9 parameters needs far more calls than a
back-of-envelope suggests — the run above is capped with `maxcall=40000` for exactly that reason,
and an earlier uncapped attempt was still going after 50 minutes.

### Why `nested`'s AIC is worse while its medians agree

`mcmc` reports AIC −3.06 and `nested` 186.07, yet the two posteriors nearly coincide:

| parameter | `mcmc` | `nested` |
|---|---:|---:|
| `mej_1` | 0.0467 | 0.0424 |
| `vej_1` | 0.3443 | 0.3141 |
| `kappa_1` | 0.2316 | 0.1947 |
| `mej_2` | 0.0793 | 0.0791 |
| `kappa_2` | 1.7134 | 2.3929 |
| `redshift` | 0.0091 | 0.0091 |

AIC is computed from the **best** draw, so a capped run that never reached the likelihood peak
scores worse even when its *typical* draws are in the right place. Both reported
`converged=False` at these deliberately small budgets. Two conclusions follow, and both matter more
than the AIC: compare AIC only between fits of comparable quality, and raise the budget until
`converged` is `True` before quoting anything.

**ABC's AIC is not comparable with `mcmc`'s at all.** ABC never evaluates the likelihood; it accepts
draws whose simulated curve is close enough under a distance, and the reported AIC comes from
whichever accepted draw was best. Compare ABC with ABC.

### A cross-check worth noticing

The GPU page fits a *different implementation* of the same physics — `kilonova_two_jax`, which even
orders its parameters differently — and `emcee_jax` there returns AIC **−7.98** against this page's
`mcmc` **−3.06**. Two independent implementations, two different samplers, two devices, agreeing to
about 5 AIC units. That is the kind of agreement that should precede belief.

## 4. Compare against a simpler model

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
| `two_component_kilonova` | 9 | **−3.06** | 14.67 | 230.5 s |
| `gaussian_rise` | 4 | 275.51 | 283.39 | 1.9 s |
| `bazin` | 4 | 491.22 | 499.11 | 3.4 s |

ΔAIC ≈ **279** in favour of the physical model over the best empirical one — decisive, and it earns
its five extra parameters many times over. It also costs about 100× more wall clock, which is the
real trade.

> **A caution this table earned the hard way.** `bazin` scores 491.22 at `nsteps=1500` and 279.3 at
> `nsteps=2000` — the same model on the same data, moving by 212 AIC units purely with budget,
> because the fit has not converged and AIC is taken from the best draw. `gaussian_rise` was stable
> across the same change (275.51 vs 275.5). **An AIC from an unconverged fit is not a property of
> the model.** Check `r.info["converged"]` before you compare anything.

## 5. Reading the numbers

* **AIC / BIC — lower is better.** BIC penalises extra parameters harder, so it favours simpler
  models more aggressively than AIC. A two-component kilonova has 9 parameters against `bazin`'s 4;
  BIC will make it earn them.
* **Log-evidence — higher is better.** Only `nested` returns it, in
  `r.info["log_evidence"]` with `r.info["log_evidence_err"]`. The difference between two models'
  `ln Z` is the log Bayes factor.
* **AIC, BIC and WAIC compare models, not spaces.** A flux-space fit and a magnitude-space fit of
  the *same* model differ by a fixed Jacobian term. Never compare across `space=`.
* **A WAIC flagged unreliable is not a number.** Check
  `r.info["predictive_metrics"]["waic"]["p_waic_reliable"]` before quoting WAIC. A blown-up `p_waic`
  usually means chains stranded away from the bulk, not a bad model.

## 6. Look at the fit, do not just read the number

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

## Next

* [`MODEL_COMPARISON_GPU.md`](MODEL_COMPARISON_GPU.md) — the same comparison on GPU, where the JAX
  models make larger budgets affordable.
* [`CHOOSING.md`](CHOOSING.md) — picking a sampler, and combinations that cannot work.
