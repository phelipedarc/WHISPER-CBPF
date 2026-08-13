# Choosing a model and a sampler

A decision guide. If you read one section, read
[Which combinations are impossible](#which-combinations-are-impossible) — those failures are
structural, not tuning problems.

## Start here

| Your situation | Use |
|---|---|
| Toy or analytic light curve, want a posterior fast | `abc` or `mcmc` on CPU |
| Analytic model, want a *good* posterior | `abc_smc` — same accuracy as `abc` at ~4× fewer simulations |
| Physical model, gradients available (JAX) | `nuts_gpu` — `wp.fit(lc, "kilonova_one_jax", sampler="nuts_gpu")` builds the density for you |
| Physical model, no gradients, many simulations | `abc_gpu` — but see the crossover below |
| Multimodal posterior | `abc_smc_gpu`, or dynesty on CPU. **Not NUTS** |
| You want a **Bayes factor**, not an information criterion | `nested` (alias `dynesty`) — the only sampler that returns `ln Z`, in `result.info["log_evidence"]`. Report `log_evidence_err` with it, and remember `ln Z` **depends on your prior widths in a way AIC/BIC do not**: measured on the same `flare` fit, widening one prior 100× moved `ln Z` by −4.1 ± 0.4 nats (analytic Occam expectation −ln 100 = −4.61) and moved AIC and BIC by −0.01. Only compare models whose priors you would defend *before* seeing the data. **Higher `ln Z` is better; lower AIC/BIC is better.** |
| Amortised inference over many transients | `snpe` |

## CPU or GPU: the crossover is real and it is high

**The GPU crosses over at roughly 50,000 simulations.** Below that, the CPU ABC wins — you pay JAX
compile time and never amortise it. Do not reach for `abc_gpu` on a 5,000-simulation run.

For chain-level parallelism the news is worse: **vmap over chains buys about 1.05×** on a model that
already saturates the GPU. One chain of the 2-component kilonova is 284 observations × 256
quadrature nodes × 2 components, which leaves no spare capacity for batching. The vectorized sampler
is convenience on that problem, not speed. See [`GPU_SETUP.md`](GPU_SETUP.md) for the measurements.

Where the GPU genuinely pays: large simulation counts (ABC past the crossover), batched model
evaluation (9.0 µs/curve at batch 512 against 11.5 ms for one), and gradient-based sampling that has
no CPU equivalent.

## Which combinations are impossible

These are not tuning problems. They cannot be made to work by adjusting parameters.

| Combination | Why it cannot work |
|---|---|
| numpy model + `nuts_gpu` or `pymc_jax_gpu_*` | NUTS needs gradients. A numpy simulator has none, and `jax.grad` cannot differentiate through it. |
| numpy model + `abc_gpu`, `abc_smc_gpu` or `emcee_jax` | They need a batched JAX forward map (`Model.predict_jax`). A numpy model has none, and a numpy function cannot be `vmap`ped. This includes `two_component_kilonova`, which is redback + sncosmo behind a numpy `predict`. |
| `flare_jax` + any JAX sampler | It *has* a JAX implementation, but `models/jax/flare.py`'s `predict_jax` takes a parameter **dict** rather than the flat theta `Model.predict_jax` specifies, so `flare_jax` registers without the slot. Known gap, tracked in `../OPEN_ITEMS.md`. |
| TDE or any of the twelve supernovae in float32 | The envelope ODE's increments are below float32 epsilon; the supernova engines carry cgs luminosities of 1e43–1e46 erg/s against float32's 3.4e38 ceiling. Both raise. **Both raise at the first `predict`, not at registration** — a registry containing `tde_gaussianrise_jax` or `arnett_jax` is not evidence the session can run it. |
| a non-`Uniform`/`LogUniform` prior + any JAX sampler | Refused rather than approximated by its bounds. Substituting a uniform for a log-uniform is a *different posterior*, not a different parameterisation: log-uniform puts half its mass below 775 K where uniform puts half below 3050 K. |
| comparing AIC/BIC/WAIC **across spaces** | Not a sampler limit — an arithmetic one. A flux-space and a magnitude-space fit have different likelihoods over different representations of the data, so their information criteria are not on one scale. Compare models *within* a space. |
| Any non-flare model + `snpe_gpu` | It always passes `flare_jax.predict_torch`. It raises `NotImplementedError` rather than returning flare draws under your model's parameter names. |
| Fitting a scatter parameter with distance-based ABC | Extra simulated noise only ever increases the expected residual, so the posterior rails to the smallest allowed scatter. **The parameter is not identifiable by ABC at all** — it is a likelihood parameter. `abc_gpu` warns if you try. |

## The two SNPE paths

This is the most confusing thing a new user meets, so be explicit about which you want.

| | `snpe` / `npe` | `snpe_gpu` |
|---|---|---|
| Backend | torch (has its own `device='cuda'`) | JAX |
| Registered? | yes | **no** — direct import only |
| Works with | any registered model | **the flare only** |
| Extra | `[sbi]` | `[gpu]` |

**Use `snpe`.** It is the general path and it already runs on a GPU through torch — you do not need
`snpe_gpu` to get GPU-accelerated SNPE. `snpe_gpu` exists for the flare benchmark and stays
unregistered because registering it would let `fit(lc, "kilonova_one_jax", sampler="snpe_gpu")`
return flare simulations labelled with kilonova parameter names — a wrong answer with no error.

```python
from whisper_cbpf.samplers.jax import fit_snpe_gpu     # deliberate, explicit
```

## The CPU/GPU sampler pairs

Each GPU sampler has a CPU sibling that computes the same thing. Keeping both is what lets you check
a GPU result against a CPU one.

| CPU | GPU | Relationship |
|---|---|---|
| `abc` | `abc_gpu` | JAX port of the same rejection scheme |
| `abc_smc` | `abc_smc_gpu` | JAX port; the importance weights are *literally the same code* |
| `mcmc` (emcee) | `emcee_jax` | same emcee frontend, jitted JAX log-density |
| `nested` (dynesty) | — | no GPU equivalent, and this one is not a gap to fill: dynesty is a CPU library, and a GPU nested sampler (`jaxns`) is a *different algorithm*, not a port. Cross-check `nested` against `mcmc` instead — both are exact-likelihood samplers over the same prior, so they must agree (`tests/test_nested.py::test_nested_agrees_with_mcmc`) |
| — | `nuts_gpu` | no CPU equivalent |

To compare CPU against GPU fairly, fix the distance. All six metrics (`chi2`, `mse`, `rmse`, `mae`,
`wmse`, `wmae`) exist in both backends and are registered for both samplers.

## Three NUTS frontends, one kernel

`nuts_gpu`, `pymc_jax_gpu_vectorized` and `pymc_jax_gpu_parallelized` all drive the **same NumPyro
kernel**. That is deliberate: one acts as a free regression test on the others.

- `nuts_gpu` — NumPyro directly. Fewest layers; start here.
- `pymc_jax_gpu_vectorized` — vmap over chains. **The default on a 2-GPU budget**; leaves a GPU free
  at ~1.22× wall-time cost.
- `pymc_jax_gpu_parallelized` — true pmap, one chain per device. Fastest per chain, but asking for
  more chains than you have devices is *worse than not asking* — see [`GPU_SETUP.md`](GPU_SETUP.md).

## Choosing a distance

The distance defines what "close" means, so it defines the posterior. The three families are not
interchangeable.

| Family | Members | Use when |
|---|---|---|
| Error-weighted | `chi2`, `wmse`, `wmae` | You trust your uncertainties. `chi2` equals `−2 ln L` for an independent Gaussian, which is what lets ABC report AIC/BIC on the same scale as the likelihood-based samplers. |
| Unweighted | `mse`, `rmse`, `mae` | You want the *shape* of the light curve. Sensible in magnitude space (already logarithmic); in flux space, where a kilonova spans decades, an unweighted metric is dominated by peak epochs and effectively ignores the tail. |
| Absolute-residual | `mae`, `wmae` | The photometry has outliers a Gaussian likelihood would fight. |

`rmse` is monotone in `mse`, so it selects the **same** draws at a matched quantile — it changes only
the numeric scale of ε, which matters if you set an absolute `threshold` rather than a quantile.

## Choosing a kilonova — read this before comparing results

There are **two kilonova implementations and they disagree on purpose**:

- `two_component_kilonova` — redback-backed, CPU, `[models]` extra
- the JAX kilonova — `register_kilonova(...)`, GPU, `[gpu]` extra

They are validated against different references and use **disjoint priors**. They are two models,
not one model twice. Comparing a fit from one against a fit from the other is not a like-for-like
comparison. See [`../DUPLICATES.md`](../DUPLICATES.md).

The same warning applies to `flare` and `flare_jax`: **`flare_jax` is a Gaussian rise with
exponential decay**, which makes it the counterpart of `gaussian_rise`, not of `flare`. Their
parameter names are disjoint, so a mistake raises rather than silently fitting the wrong model.

## When a fit has not converged

- **r-hat above ~1.1, ESS in the single digits, and chains in different places** → genuine
  multimodality, not a warmup problem. NUTS is a local sampler and **cannot** hop between separated
  modes at any warmup budget. Switch to `abc_smc_gpu` or dynesty, or report per-mode rather than
  marginally. A marginal median over three modes is not a physical solution.
- **Parameters railing at prior bounds** → the prior is shaping the answer. Widen it, or ask whether
  a missing physical term (extinction, say) is being absorbed by the railing parameter.
- **A TDE fit that pushes the termination time through your data** → unconverged, not converged. The
  model returns zero flux outside the envelope's own time span, which is a genuine cliff that
  `jax.grad` cannot see.
