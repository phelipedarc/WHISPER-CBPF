# Choosing a model and a sampler

A decision guide. If you read one section, read
[Which combinations are impossible](#which-combinations-are-impossible) — those failures are
structural, not tuning problems.

## Start here

| Your situation | Use |
|---|---|
| A survey alert and several physical models, want a ranked answer | `wp.compare(lc, models)`: `sampler="auto"` is `emcee_jax` on a GPU, `mcmc` on a CPU, and every fit is optimised to its likelihood maximum and checked ([`LSST_ALERTS.md`](LSST_ALERTS.md)) |
| Many alerts, one model | `wp.fit_batch` (emcee, all alerts in one compiled GPU loop) |
| Many fits over several GPUs and CPU cores | `wp.run_jobs` |
| A Bayes factor for the top two of a comparison | `wp.compare(..., evidence_check=True)` runs `nested` on them |
| Toy or analytic light curve, want a posterior fast | `abc` or `mcmc` on CPU |
| Analytic model, want a *good* posterior | `abc_smc` — same accuracy as `abc` at ~4× fewer simulations |
| Physical model, gradients available (JAX) | `nuts_gpu` — `wp.fit(lc, "kilonova_one_jax", sampler="nuts_gpu")` builds the density for you |
| Physical model, no gradients, many simulations | `abc_gpu` — but see the crossover below |
| Multimodal posterior | `abc_smc_gpu`, or dynesty on CPU. **Not NUTS** |
| You want a **Bayes factor**, not an information criterion | `nested` (alias `dynesty`) — the only sampler that returns `ln Z`, in `result.info["log_evidence"]`. Report `log_evidence_err` with it, and remember `ln Z` **depends on your prior widths in a way AIC/BIC do not**: measured on the same `flare` fit, widening one prior 100× moved `ln Z` by −4.1 ± 0.4 nats (analytic Occam expectation −ln 100 = −4.61) and moved AIC and BIC by −0.01. Only compare models whose priors you would defend *before* seeing the data. **Higher `ln Z` is better; lower AIC/BIC is better.** |
| Amortised inference over many transients | `snpe` |

## CPU or GPU: there is no single crossover

It is tempting to quote one number — "the GPU wins past 50,000 simulations". Measured, that is not
true: **the crossover moves by a factor of 60** depending on two choices, so both have to be stated
alongside any figure.

Measured on the one-component JAX kilonova, AT2017GFO, one idle A6000, 3 repeats:

| simulations | CPU `abc`, `n_jobs=1` | GPU `abc_gpu` | ratio |
|---|---|---|---|
| 1,000 | 4.84 s | 2.18 s | 2.2× |
| 10,000 | 50.34 s | 3.19 s | 15.8× |
| 30,000 | 154.75 s | 3.33 s | 46.5× |
| 50,000 | 271.47 s | 3.57 s | 76.0× |

Fitting `wall(N) = a + b·N` per arm and solving for the crossover `N*`:

| CPU baseline | `N*` warm (compile excluded) | `N*` cold (compile included) |
|---|---|---|
| `n_jobs=1` — **measured** | **1,067** | **4,047** |
| `n_jobs=8` at the 3.8× this repo measured — *extrapolated* | 2,410 | 19,916 |
| `n_jobs=8` at a perfect 8× — *extrapolated* | 4,485 | 63,992 |

So the old "roughly 50,000" is **defensible only in the cold-start, well-parallelised corner**, and
is wrong by ~50× for a warm serial baseline. Read it as: *if you are paying the XLA compile on every
run and your CPU ABC is using all its cores, the GPU takes tens of thousands of simulations to pay
off; if you are amortising the compile, it pays off in the low thousands.*

Two caveats that matter more than the number:

* **Only the `n_jobs=1` rows are measured.** The parallel rows are linear extrapolations from this
  repo's own 3.8×-at-`n_jobs=8` ABC measurement, not new timings.
* **This compares whisper's CPU ABC against whisper's GPU ABC, not CPU silicon against GPU
  silicon.** The CPU arm calls `predict` (jitted JAX, one curve per call); the GPU arm calls
  `vmap(predict_jax)` at `chunk=16` (the default then; 250 now). Some of the gap is batching, not
  the device.

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
| TDE or any of the twelve supernovae in float32 | The envelope ODE's increments are below float32 epsilon; the supernova engines carry cgs luminosities of 1e43–1e46 erg/s against float32's 3.4e38 ceiling. Both raise. **Both raise at the first `predict`, not at registration** — a registry containing `tde_gaussianrise_jax` or `arnett_jax` is not evidence the session can run it. |
| a prior distribution other than `Uniform`, `LogUniform`, `Normal`, `TruncatedNormal` or `Fixed` + any JAX sampler | Refused rather than approximated by its bounds. Substituting one family for another is a *different posterior*, not a different parameterisation: log-uniform puts half its mass below 775 K where uniform puts half below 3050 K. The five families work in every sampler ([`API_REFERENCE.md` A2](API_REFERENCE.md#priors-beyond-boxes-normal-truncatednormal-fixed)). |
| comparing AIC/BIC/WAIC **across spaces** | Not a sampler limit — an arithmetic one. A flux-space and a magnitude-space fit have different likelihoods over different representations of the data, so their information criteria are not on one scale. Compare models *within* a space. |
| numpy model + `snpe_gpu` | Its simulator is built from `Model.predict_jax`, and a numpy model has none. It raises `ValueError` naming `sampler='snpe'`, whose simulator runs the numpy `predict` (its network can still train on the GPU with `device='cuda'`). |
| Fitting a scatter parameter with distance-based ABC | Extra simulated noise only ever increases the expected residual, so the posterior rails to the smallest allowed scatter. **The parameter is not identifiable by ABC at all** — it is a likelihood parameter. `abc_gpu` warns if you try. |

## The two SNPE paths

This is the most confusing thing a new user meets, so be explicit about which you want.

| | `snpe` / `npe` | `snpe_gpu` |
|---|---|---|
| Simulator | the model's numpy `predict`, one draw at a time on the CPU | the model's `predict_jax`, batched on the GPU |
| Training | torch, `device='cpu'` by default (`'cuda'` works) | torch, `device='cuda'` by default |
| Registered? | yes | yes |
| Works with | any registered model | any model with a `predict_jax` (the JAX factories, `flare_jax`) |
| Extra | `[sbi]` | `[sbi]` + `[gpu]` |

**They are one sampler with two simulators.** `snpe_gpu` hands its batched simulator to the same
`SNPESampler` code, with the same defaults (`num_rounds=2`) and the same `embedding_net=` names, so
a CPU/GPU pair differs only in its simulator and its default training device. Use `snpe` for a
numpy model; for a model built by a JAX factory, `snpe_gpu` moves the simulator onto the GPU too.

```python
res = wp.fit(lc, "kilonova_one_jax", sampler="snpe_gpu")
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
kernel**. That is deliberate: one acts as a free regression test on the others. They also share the
default start (`init_strategy="prior_scan"`) and the chain checks behind `info["converged"]`, so a
disagreement between them is the frontend, not the start or the diagnostics.

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
comparison.

The same warning applies to `flare` and `flare_jax`: **`flare_jax` is a Gaussian rise with
exponential decay**, which makes it the counterpart of `gaussian_rise`, not of `flare`. Their
parameter names are disjoint, so a mistake raises rather than silently fitting the wrong model.

## When a fit has not converged

`result.diagnostics()` is one report for every sampler: each check with its value, threshold, pass
or fail and a plain reason, then a "Why" section. `result.info["convergence_problems"]` lists every
check the sampler itself failed, one sentence each, and the fit warned with the same text
(`nuts_gpu`, `pymc_jax_gpu_*`, `emcee_jax`, `mcmc`). `converged` is `True` only when the list is
empty.

- **`stranded_chains` or `stuck_walkers` not empty** → a chain (or emcee walker) sits in a local
  optimum, typically the model switched off between the epochs or a corner of the prior box, and
  its draws are pooled into the posterior. The default start (`init_strategy="prior_scan"` for the
  NUTS samplers, `init="prior_scan"` for `emcee_jax`) makes this rare; if it persists, tighten the
  prior the optimum lives on (a width floor below the largest cadence gap lets a bump hide), or
  start the chains yourself with `init=` (every MCMC-type sampler, CPU `mcmc` included): one row per
  chain or walker, a point, or `init=previous_result` (an ABC fit, or the same alert's earlier cut).
  The `"uniform"` / `"box"` / `"jitter"` starts reproduce old runs. `"prior"` (independent prior
  draws) is the fallback when the default start cannot spread the walkers: an ensemble start that
  does not span every parameter is refused with a message naming it.
- **`stuck_walkers` on a fit run with `walker_coordinates="linear"`** → in linear coordinates a
  LogUniform parameter's `-ln x` prior density can put walkers spread over a flat direction "more
  than 10 nats below the best walker". The default, `walker_coordinates="own"`, moves such a
  parameter in its log, where that direction is flat.
- **A large gain of the likelihood maximum** (`result.likelihood_max_opt(lc).gain` above 1 ln L, or
  the report's "gap to the likelihood maximum" above 5) → the sampler never reached the likelihood peak, so its posterior may miss the
  best region. Restart from the peak: `init=(peak.params, 0.01)`, then run longer.
- **"every chain's best draw is ... below the best point the prior scan found"** → the chains agree
  with each other and are all wrong, which R-hat cannot see. Same remedies.
- **`frozen_chains` not empty** → a chain adapted a tiny step and stopped moving. In float32 this
  is an MJD-scale clock or parameter (the sampler warned before starting): enable x64, or use
  `lc.set_explosion_date(mjd)` with a t0 prior in days.
- **A warning that most prior draws "score the same log-density"** (before sampling) → over most of
  the prior box the model puts no signal where the data are, typically a `t0` or explosion-epoch
  prior much wider than the data window. A chain there has no gradient to follow, and the start
  depends on the few draws that landed near the data (with t0 ~ U(-10 000, 10 000) around 40 d of
  data, 1 seed in 4 had none). Give the time parameter a box around the data.
- **Divergences with nothing stranded** → the posterior has a kink or a funnel the integrator
  cannot follow (`flare_jax`'s rise/decay join at `t0` diverges in most fits). Raise
  `target_accept_prob`, or treat the posterior near that feature with suspicion.
- **R-hat just above 1.01 with a small ESS, nothing stranded** → the chains are too short. Raise
  `num_samples` (for emcee, `nsteps`: it needs `nsteps >= 50 x` its largest autocorrelation time).
- **r-hat above ~1.1, ESS in the single digits, and chains in different places** → genuine
  multimodality, not a warmup problem. NUTS is a local sampler and **cannot** hop between separated
  modes at any warmup budget. Switch to `abc_smc_gpu` or dynesty, or report per-mode rather than
  marginally. A marginal median over three modes is not a physical solution.
- **Parameters railing at prior bounds** → the prior is shaping the answer. Widen it, or ask whether
  a missing physical term (extinction, say) is being absorbed by the railing parameter.
- **A TDE fit that pushes the termination time through your data** → unconverged, not converged. The
  model returns zero flux outside the envelope's own time span, which is a genuine cliff that
  `jax.grad` cannot see.
