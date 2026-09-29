# WHISPER-CBPF

Bayesian model comparison for astronomical transient light curves (supernovae, kilonovae and tidal
disruption events) on CPU or GPU, from a survey alert to a ranked, checked and explained answer.

```bash
pip install "git+https://github.com/phelipedarc/WHISPER-CBPF.git"
pip install jax           # the supernova and TDE models, on a CPU; on an NVIDIA GPU use [gpu]
```

The supernova and TDE families (`"arnett"`, `"magnetar"`, `"tde"`, ...), and so the quickstart
below, are JAX models, which the base install alone cannot run. `pip install jax` adds them on a
CPU; the `[gpu]` extra ([Install options](#install-options)) on an NVIDIA GPU.

> **Status: alpha (0.2.0).** The package installs and runs, but not every sampler has been validated
> to the same level. Read [Known limitations](#known-limitations) before publishing a number.

## From an LSST alert to a ranked answer

```python
import jax; jax.config.update("jax_enable_x64", True)    # supernova and TDE models need float64
import whisper_cbpf as wp
lc  = wp.load_lightcurve("tests/data/lsst_alert_sn.json", survey="lsst", redshift=0.1)
cmp = wp.compare(lc, ["arnett", "magnetar"])
print(cmp.summary())
cmp.report("out/")
```

This reads a Rubin alert packet (a synthetic supernova shipped in `tests/data`, so the path works
from a checkout), fits both supernova models, finds each fit's likelihood peak, checks each fit's
convergence, ranks the models by BIC with weights and a Jeffreys grade, and writes `out/report.html`,
one self-contained page with the ranking, the figures, the diagnostics and the facts behind every
number.

- ZTF alerts load the same way with `survey="ztf"` (ALeRCE, Fink or the alert packet).
- Only difference-imaging photometry is fitted (`psfFlux`, `magpsf`), upper limits included, and
  rows before the explosion are never fitted: they only set the explosion-time prior.
- With the host redshift unknown, omit `redshift=`: it is then fitted, and the summary says that a
  fitted redshift can bias the comparison.

The whole workflow, how to read the answer, forecasts, and measured seconds per alert on a CPU and
on one GPU: [`docs/LSST_ALERTS.md`](docs/LSST_ALERTS.md).

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
print(result.diagnostics())               # every convergence check, pass or fail, and why
```

> **One ordering rule that bites everyone.** Call `set_explosion_date` *before*
> `select_time_window`. The window filters whatever `time` currently holds, so windowing first
> compares days-since-explosion against raw MJD (~57983) and quietly leaves you with an empty light
> curve.

## Comparing models

`wp.compare` fits every model to the same data with the same settings and ranks them. On a light
curve simulated from `flare`, whose answer is known:

```python
import numpy as np
import whisper_cbpf as wp

t = np.linspace(0.5, 30.0, 30)                                   # 30 epochs, one band
truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
flux = wp.get_model("flare").predict(truth, t) + np.random.default_rng(1).normal(0.0, 0.1, 30)
toy = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))

cmp = wp.compare(toy, ["flare", "bazin", "gaussian_rise"], sampler="mcmc")
print(cmp.summary())        # winner, grade, table, models left out and why, caveats
cmp.table                   # every number: BIC, delta, weight, grade, converged, problems
```

```text
Comparison of 3 models on 30 points by BIC: winner 'flare' (decisive over 'bazin', ln B = 10.34)

  rank  model          sampler  k  n   max ln L  BIC     ln Z  dBIC   weight  grade     converged
  1     flare          mcmc     3  30  31.88     -53.55  -     0.00   1.000   decisive  yes
  2     bazin          mcmc     4  30  23.24     -32.88  -     20.67  0.000   decisive  yes
  3     gaussian_rise  mcmc     4  30  15.47     -17.33  -     36.22  0.000   decisive  yes
```

The true model wins, and every fit passed its convergence report (25 s on a CPU).

- AIC and BIC come from each fit's **optimised likelihood maximum** (`wp.likelihood_max_opt`), not
  from the sampler's best draw, which falls short of the peak by a different amount for each model.
  The posterior is not changed. Why this matters, with numbers:
  [`docs/LSST_ALERTS.md` §5](docs/LSST_ALERTS.md#5-why-the-ranking-uses-the-likelihood-maximum).
- The ranking is by ln Z when every model has a converged nested-sampling evidence
  (`sampler="nested"`), and by BIC otherwise. ln Z depends on the priors, so rank by it only
  between priors you would defend; the summary says when a ln Z gap comes from the priors rather
  than from the fits.
- A model with at least as many parameters as points, or fitted to other data points, is left out
  with its reason.
- Every fit carries its convergence report; `converged` in the table says whether it passed.

Rules and worked examples on CPU and GPU: [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

## Plots, facts and the report

```python
wp.plot_models(cmp, lc)                  # every model over the data, with residuals
wp.plot_model_comparison(cmp)            # weights, the gap to the best, and the grade
wp.plot_ppc(result, lc, model="bazin")   # posterior predictive, 68 % and 95 % bands
wp.plot_corner([result])                 # note the list: overlay several results
facts = cmp.facts()                      # every quoted number, computed by a stated rule
cmp.report("out/")                       # all of it on one HTML page
```

## Custom priors

Every sampler accepts `prior=`. Build one from `Uniform`, `LogUniform`, `Normal`,
`TruncatedNormal` and `Fixed`:

```python
prior = wp.Prior({
    "amplitude": wp.Uniform(0.0, 5.0),
    "t0":        wp.TruncatedNormal(4.0, 1.0, 0.0, 10.0),
    "tau_rise":  wp.LogUniform(0.1, 20.0),
    "tau_fall":  wp.LogUniform(0.1, 50.0),
})
result = wp.fit(lc, "bazin", sampler="nested", prior=prior, space="magnitude", seed=0)
```

Omit `prior=` and the model's own default is used. A `Fixed` parameter is held at its value and not
counted in AIC or BIC.

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
| `arnett`, `magnetar`, `csm_shock_arnett`, `shock_cooling_arnett`, `tde` | supernova and TDE families, bound to your light curve by `wp.compare` | `[gpu]` |

Physical JAX models can also be built by hand, for your bands and distance:

```python
import jax; jax.config.update("jax_enable_x64", True)   # FIRST -- before any array exists
import whisper_cbpf as wp

kn  = wp.register_kilonova(["sdssg"], redshift=0.0098, dl_cm=1.23e26)        # one component
kn2 = wp.register_kilonova_two(["sdssg"], redshift=0.0098, dl_cm=1.23e26)    # blue + red
kn3 = wp.register_kilonova_three(["sdssg"], redshift=0.0098, dl_cm=1.23e26)  # blue + purple + red
tde = wp.register_tde(["sdssg"], redshift=0.0098, dl_cm=1.23e26)
sn  = wp.register_supernova("arnett", ["sdssg"], redshift=0.0098, dl_cm=1.23e26)
```

The supernova, TDE and kilonova factories take `free=["t_exp", "redshift"]` to fit the explosion
time and the redshift as ordinary parameters.

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
| `emcee_jax` | GPU | `[gpu]` | emcee, vectorised on the GPU; `compare`'s default on a GPU |
| `nuts_gpu` | GPU | `[gpu]` | gradient-based NUTS (NumPyro directly) |
| `pymc_jax_gpu_vectorized` / `_parallelized` | GPU | `[gpu]` | NUTS via a PyMC front end |
| `abc_gpu`, `abc_smc_gpu` | GPU | `[gpu]` | ABC with GPU-batched simulation |
| `snpe` / `npe`, `snpe_gpu` | CPU/GPU | `[sbi]` | neural simulation-based inference |

Many alerts with one model in one GPU call: `wp.fit_batch`. Many fits over several GPUs and CPU
cores: `wp.run_jobs`. Which sampler to pick, and which combinations cannot work:
[`docs/CHOOSING.md`](docs/CHOOSING.md).

## Install options

Four extras, combinable:

```bash
pip install "whisper-cbpf[gpu,models,sbi,analysis] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

| Extra | Gives you |
|---|---|
| `[gpu]` | all JAX models and GPU samplers (includes pymc, which brings arviz) |
| `[models]` | redback physical models and the TDE / supernova priors, plus pyphot for SVO filters |
| `[sbi]` | the SNPE / NPE sampler |
| `[analysis]` | arviz (ESS, PSIS-LOO) and astroquery (SVO search) for a CPU-only install |

**GPU users:** `[gpu]` alone is not enough. Run `source "$(whisper-cbpf-env)"` *before* starting
Python, or JAX silently falls back to CPU at ~50× slower with no error. Full detail in
[`INSTALL.md`](INSTALL.md) and [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md).

## Known limitations

Honest list, so you do not discover these in review:

* **A local sampler can leave a chain in a local optimum.** NUTS (`nuts_gpu`, `pymc_jax_gpu_*`)
  and emcee (`emcee_jax`, `mcmc`) only move locally, and these likelihoods have optima a chain
  cannot leave: a model shrunk until it hides between the epochs, or a corner of the prior box.
  Every chain therefore starts from a scan of the prior climbed into the best basin found
  (`init="prior_scan"`, the default), and **every fit checks itself**: `result.diagnostics()` and
  `result.info["converged"]` fail on divergences, R-hat, ESS, stranded, frozen or stuck chains, and
  on a prior-scan optimum no chain reached. Check it before using a posterior. The checks cannot
  see every failure: when every walker settles in the same poorer peak, the fit can pass them.
  When a number matters, compare it with a fit from independent prior draws (`init="prior"`).
* **The maximum-likelihood optimisation is local.** It finds the peak of the mode the sampler found;
  it cannot find a mode no chain visited. A large gain over the sampler's best draw is flagged, and
  a small BIC gap triggers the evidence check in `compare`.
* **A fitted redshift can bias a model comparison**: a model can move the source to buy a fit.
  Pass a host redshift when there is one, and read each redshift posterior against its prior.
* **float32 cannot hold an MJD clock.** With `jax_enable_x64` off, a time near MJD 60000 moves in
  0.0039 d steps and chains freeze on the staircase. The JAX samplers warn before sampling; enable
  x64, or measure time from an epoch near the data (`lc.set_explosion_date(mjd)`).
* **Supernova and TDE error bars are somewhat too narrow at the default settings.** On simulated
  LSST alerts ([`docs/VALIDATION.md`](docs/VALIDATION.md) section 4) the 95 % intervals of Arnett,
  the shock-cooling and CSM families and the TDE hold the truth 87-89 % of the time (their 68 %
  intervals 62-71 %), and the magnetar's only 73 % (48 %); a 4 times longer chain moved Arnett only
  from 87 % to 88 %. The kilonova intervals are calibrated. Widen a supernova or TDE interval before
  quoting it (VALIDATION.md section 9 says how).
* **An unknown explosion date gets a narrow default prior.** It runs from the last non-detection
  (any band, any depth) to the first detection, and at LSST's single-visit depth that window
  excluded the true explosion in 24 of 30 simulated alerts, which biases the fit and can make the
  wrong supernova family win. Unless the limits before the first detection are deep, pass a wider
  `t_exp` prior ([`docs/LSST_ALERTS.md`](docs/LSST_ALERTS.md) section 2).
* **A GPU fit is not bit-reproducible by default.** Two `emcee_jax` runs at the same seed on a GPU
  give different draws; set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` before starting Python for
  identical ones (about 35 % slower). CPU runs repeat exactly.
* **The redback CPU model `two_component_kilonova` has exchangeable components.** Its prior gives
  both the same `kappa` range, so fits can label-switch. The JAX models
  (`kilonova_two_jax`, `kilonova_three_jax`) do not have this problem — their default Villar priors
  use disjoint corridors. Impose disjoint `kappa` ranges yourself if you use the redback model.
* **A WAIC whose `p_waic` is flagged unreliable is not a number** — check the flag before quoting it.

## Documentation

| | |
|---|---|
| [`INSTALL.md`](INSTALL.md) | installing, GPU setup, verifying |
| [`docs/LSST_ALERTS.md`](docs/LSST_ALERTS.md) | from an alert to a ranked answer: fields, photometry, reading the answer, forecasts, speed |
| [`docs/TUTORIAL.md`](docs/TUTORIAL.md) | guided walkthrough |
| [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) | the ranking rules, and worked comparisons on CPU and GPU |
| [`docs/CHOOSING.md`](docs/CHOOSING.md) | picking a sampler; impossible combinations |
| [`docs/PHOTOMETRY.md`](docs/PHOTOMETRY.md) | how a band magnitude is computed, from alert fields to filter integrals |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | what the science-validation suite measured on simulated LSST alerts: calibration, model selection, speed, and what does not pass yet |
| [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) | CUDA, float64, multi-GPU |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | every public symbol, by task |
| [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) | how the JAX models differ from redback, and why |
| [`docs/MIGRATION.md`](docs/MIGRATION.md) | whisper_cbpf supersedes the discontinued WHISPER_AI (`whisper_labia`) and whisper-GPU (`whisper_gpu`), merged from WHISPER_AI @ 8ba3843 and whisper-GPU @ 10796a0 |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed in each release |
| [`notebooks/`](notebooks/) | runnable notebooks, one per subsystem |

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

GPL-3.0 — see [`LICENSE`](LICENSE).
