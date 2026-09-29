# WHISPER-CBPF (`whisper_cbpf`) — API Reference

Generated for **v0.2.0**. Every signature here was checked against the installed package.

- **[Part A](#part-a--by-task)** is organised by what you want to do: load alert photometry, fit,
  find the likelihood peak, check a fit, compare models, save and resume, forecast, plot, explain,
  and scale to many alerts. The alert workflow end to end, with measured costs, is in
  [`LSST_ALERTS.md`](LSST_ALERTS.md).
- **[Part B](#part-b--by-subsystem)** is the reference by subsystem: the light curve, bands,
  photometry, priors, models, distances, samplers, likelihoods, metrics, validation, and the JAX/GPU
  half.

- **Install:** `pip install whisper-cbpf` is CPU-complete; `[gpu]` adds the JAX half. Every GPU entry
  is listed by `list_models()` / `list_samplers()` / `list_distances()` even without the extra.
- **Run the tests:** `python -m pytest -q`. The JAX tiers need the `[gpu]` extra and
  `source "$(whisper-cbpf-env)"` before Python starts — see [`GPU_SETUP.md`](GPU_SETUP.md).

## Package map

```
whisper_cbpf/
  __init__.py          # the public API
  compare.py           # compare, Comparison: fit, optimise the likelihood, check, rank, weigh, grade
  likelihood_max_opt.py
                       # likelihood_max_opt, LikelihoodMaxOptResult: the likelihood peak behind AIC/BIC
  results.py           # save / load_result / fit_cached, DiagnosticsReport, provenance
  forecast.py          # forecast, discriminate
  facts.py             # result_facts, comparison_facts, write_facts
  report.py            # report: one self-contained HTML page
  profile.py           # profile, capacity
  parallel.py          # run_jobs, Job
  plotting.py          # plot_light_curve, plot_ppc, plot_calibration, plot_corner, and the plot kit
  validation.py        # recovery_metrics, posterior_predictive_check, sbc_rank(s), check_parity
  embeddings.py        # MLPEmbedding, TCNEmbedding, build_embedding (SNPE conditioning nets)
  io/                  # LightCurve, load_lightcurve, survey presets, bands, photometry, units, svo
  synphot/             # the band integral for every model: FilterSet, gauss_rule, resolve_filter
  priors/              # Uniform, LogUniform, Normal, TruncatedNormal, Fixed, Prior (_numpy + lazy _jax)
  likelihood/          # the four likelihood classes + make_likelihood   (_numpy.py + lazy _jax.py)
  distance/            # chi2, max_abs_z + the distance registry         (_numpy.py + lazy _jax.py)
  metrics/             # waic, per_band_metrics, predictive_metrics; ArviZ ESS in _jax.py
  models/              # Model + register/get/list, the CPU built-ins, the redback adapter,
                       # constraints, cosmology (luminosity_distance_cm)
  models/jax/          # kilonova (1/2/3-component), tde, supernova, flare + binding factories
  samplers/            # BaseSampler, SamplerResult, fit, prepare_lc, and the CPU samplers
  samplers/jax/        # nuts_gpu, abc_gpu, abc_smc_gpu, emcee_jax, pymc_gpu, snpe_gpu,
                       # log_density (_adapters), fit_batch (batch)
  backends/            # check_gpu, require_jax, gpu_list, device policy, env.sh
```

## Top-level names

`import whisper_cbpf as wp` exposes 104 names (`wp.__all__`), grouped here by task. `import
whisper_cbpf` works with no jax, torch, sbi or redback installed; a name whose module cannot be
imported raises an ImportError naming the module and the cause when it is called.

| task | names |
|---|---|
| light curves and bands | `LightCurve`, `load_lightcurve`, `group_bands`, `FILTER_LOOKUP`, `resolve_band`, `resolve_bands`, `LSST_BAND_INFO`, `SvoUnavailable`, `register_manual_band`, `unregister_manual_band`, `clear_manual_bands`, `resolve_filter`, `set_default_band_system`, `default_band_system` |
| priors | `Prior`, `Uniform`, `LogUniform`, `Normal`, `TruncatedNormal`, `Fixed` |
| models | `Model`, `register_model`, `get_model`, `list_models`, `redback_model`, `register_redback`, `luminosity_distance_cm` |
| JAX models | `kilonova_model`, `kilonova_two_model`, `kilonova_three_model`, `tde_model`, `supernova_model`, `supernova_models`, `flare_model`, `register_kilonova`, `register_kilonova_two`, `register_kilonova_three`, `register_tde`, `register_supernova` |
| distances and likelihoods | `chi2_distance`, `max_abs_z_distance`, `register_distance`, `get_distance`, `list_distances`, `GaussianLikelihood`, `GaussianLikelihoodWithScatter`, `GaussianLikelihoodWithUpperLimits`, `MixtureGaussianLikelihood`, `make_likelihood`, `register_likelihood`, `list_likelihoods` |
| fitting | `fit`, `fit_ABC`, `fit_ABC_SMC`, `fit_MCMC`, `fit_nested`, `fit_SNPE`, `SamplerResult`, `register_sampler`, `list_samplers` |
| the likelihood peak | `likelihood_max_opt`, `LikelihoodMaxOptResult` |
| saving and checking fits | `load_result`, `fit_cached`, `DiagnosticsReport`, `check_parity` |
| comparing models | `compare`, `Comparison` |
| forecasts | `forecast`, `discriminate` |
| plots | `plot_light_curve`, `plot_ppc`, `plot_calibration`, `plot_corner`, `CORNER_PALETTE`, `plot_forecast`, `plot_models`, `plot_model_comparison`, `plot_widths` |
| facts and report | `result_facts`, `comparison_facts`, `write_facts`, `report` |
| metrics and validation | `waic`, `per_band_metrics`, `predictive_metrics`, `recovery_metrics`, `posterior_predictive_check`, `sbc_rank`, `sbc_ranks` |
| many alerts and speed | `log_density`, `fit_batch`, `profile`, `capacity`, `Job`, `run_jobs` |
| GPU environment | `check_gpu`, `require_jax`, `x64_enabled`, `gpu_list`, `n_jobs`, `env_script`, `env_report` |
| version | `__version__` |

`compare`, `forecast`, `report`, `profile` and `likelihood_max_opt` are both functions and module names:
`wp.compare` is the function (and so is `import whisper_cbpf.compare as C`). Reach a module's other
names with `from whisper_cbpf.likelihood_max_opt import STD_SPAN`.

---

# Part A — by task

## A1. Load alert photometry: survey presets

`load_lightcurve(path, *, survey=None, limits=None, name=None, redshift=None, luminosity_distance=None, data_mode=None, flux_unit=None, magnitude_unit=None, column_map=None, band_aliases=None, band_lookup=None, normalize=True, default_band=None, quality_cuts=True, drop_nonfinite=True, flag_filters=None, time_min=None, time_max=None, bands=None, min_snr=None, explosion_date=None, delimiter=None, resolve_band_info=True, svo_fallback=True)` → `LightCurve`
(`io.loader`, `io.surveys`)

With `survey="ztf"` or `"lsst"` it reads alert photometry in the brokers' own fields and returns a
magnitude-mode `LightCurve`, sorted by time. The input `path` can be:
- a path (`.csv`, or `.json` with `survey=`), an open file, a DataFrame, one record, or a list of records;
- a ZTF alert packet (`candidate` + `prv_candidates`);
- a Rubin alert packet (`diaSource` + `prvDiaSources` + `prvDiaForcedSources`);
- ALeRCE's `{"detections": [...], "non_detections": [...]}`.

Field names match case-insensitively, with Fink's prefixes (`i:`, `d:`, `r:`) removed. A table
already in whisper's columns (`time, band, magnitude, magnitude_err, upper_limit`) is read as it is,
with its bands mapped to the survey labels.

| | `survey="ztf"` | `survey="lsst"` |
|---|---|---|
| Detections | `magpsf` / `sigmapsf` where `isdiffpos` is positive (`t`, `1`) | diaSource `psfFlux` / `psfFluxErr` [nJy]: `m = 31.4 - 2.5 log10 f`, `sigma_m = (2.5/ln 10) err/f` |
| Never used | `magpsf_corr` (raises if it is the only magnitude) | `scienceFlux` (raises if it is the only flux) |
| Upper limits (5 sigma) | `diffmaglim` of rows without `magpsf` | `limits=` forced photometry (e.g. Fink `/fp`) or the packet's forced sources, at epochs with no diaSource: `31.4 - 2.5 log10(5 psfFluxErr)` |
| Dropped and counted in `lc.meta["n_dropped"]` | negative subtraction (`isdiffpos` `f`/`0`/`-1`), Fink `tag == "badquality"` | `isNegative` or `psfFlux <= 0`, `psfFlux_flag`, `timeWithdrawnMjdTai` set |
| Time | `mjd`, or `jd - 2400000.5` (UTC) | `midpointMjdTai` (TAI) |
| Bands | `fid` 1/2/3 or a band field → `ztfg ztfr ztfi` | `band` → `lsstu lsstg lsstr lssti lsstz lssty` |

Both presets also drop and count repeats of one exposure (same time, band and kind), an upper limit
at a detection's epoch (same visit, or within 1e-4 d), and detections below `min_snr` (`low_snr`).
A warning is raised when negative fluxes were dropped. Band labels are always survey-prefixed, never
bare letters; `whisper_cbpf.io.SURVEY_BANDS` lists them.

Selections run on the input's MJD clock, in this order: `bands=` (with `survey=`, `"g"` means the
survey's g); `time_min` / `time_max`; `min_snr` (with `survey=` it cuts detections only); then
`explosion_date`, which shifts the clock last.

A preset records in `lc.meta`: `survey`, `photometry`, `time_system`, `upper_limit_sigma` (5.0),
`n_dropped` and `first_detection_mjd` (the MJD of the first detection left after the cuts; use it as
day 0 with `lc.set_time_reference(lc.meta["first_detection_mjd"], "first detection")`).

**Raises `ValueError`** when no detection is left ("not enough data: no detection is left ...",
with the counts); the survey, a band or a ZTF `fid` is unknown; `magpsf_corr` or `scienceFlux` is
present without the difference-image field; forced photometry is passed as the detections, or
`limits=` lacks `psfFluxErr`; an argument the preset fixes itself is passed (`column_map`,
`band_aliases`, `band_lookup`, `default_band`, `flux_unit`, `magnitude_unit`, `flag_filters`,
`normalize=False`, a non-magnitude `data_mode`); `limits=` is given without `survey=`.

A preset light curve fits with no argument: its upper limits use the censored flux likelihood at
`lc.meta["upper_limit_sigma"]` ([A2](#a2-fit-one-model)).

```python
lc = wp.load_lightcurve("sources.json", survey="lsst", limits="fp.json", redshift=0.35)
lc = wp.load_lightcurve(alerce_response, survey="ztf", redshift=0.05, bands=["g", "r"], min_snr=3)
```

`whisper_cbpf.io.surveys`: `survey_table(x, survey, *, limits=None, delimiter=None)` returns
`(table, info)`, the engine behind the preset, with the drop counts; `survey_band(label, survey)`
maps a label (`"g"` → `"lsstg"`, `"zg"` → `"ztfg"`) and raises outside the survey;
`check_survey(survey)` validates the name. Constants: `SURVEYS`, `LSST_ZP_NJY = 31.4`,
`LIMIT_SIGMA = 5.0`, `SAME_EPOCH_DAYS = 1e-4`, `DROP_REASONS`.

Without `survey=`, `load_lightcurve(bands=...)` reads the names through the same alias map and
grouping as the band column (`bands=["zg"]` finds `ztfg`), raises when nothing matches, and `path`
may be a DataFrame or a list of records.

## A2. Fit one model

### What every fit does

`fit(lc, model, sampler="auto", *, likelihood_max_opt=False, **kwargs)` → `SamplerResult` dispatches to any
registered sampler ([§6.4](#64-samplers--whisper_cbpfsamplers)). `sampler="auto"` (the same rule
as `compare`) is `emcee_jax` for a model with `predict_jax` when JAX sees a GPU, CPU `mcmc`
otherwise; both fit upper limits, so a survey alert needs only the data and the model. Every
sampler, whichever way it is called, applies three rules before it fits.

**Pre-event data are never fitted.** Rows at or before the event are left out of the likelihood
and of the model evaluation; they can only define a prior.

| light curve / model | event | rows left out |
|---|---|---|
| model with a free `t_exp` (`free=["t_exp"]`) | first detection | the non-detections before it; they set the `t_exp` prior |
| model with `t_exp` = `Fixed(v)` | `v` | `time <= v` |
| explosion or merger declared: `lc.set_explosion_date(mjd)`, `set_time_reference(mjd, "explosion"/"merger")`, or `lc.meta["merger_mjd"]` (an MJD) | the event on the curve's clock | `time <=` event |
| other reference (`set_time_reference(mjd, "first detection"/"peak"/...)`), model without an epoch parameter | 0 | `time <= 0` |
| the same, but the model fits its own epoch (`t0`/`tpeak`/`t_peak`: `bazin`, `gaussian_rise`, `flare_jax`) | none | nothing (`info["pre_event"]["reason"]`) |
| raw MJD clock, no declared event | none | nothing |

Every fit records `info["excluded_pre_event"]` (the count), `info["pre_event"]` (`rule`,
`reference`, `inclusive`, `label`, `reference_mjd`, `n_excluded`, `n_detections`,
`n_upper_limits`) and, for a model with a `t_exp`, `info["t_exp_prior"]`. One UserWarning says what
was left out. It raises "not enough data" when no row, or no detection, is left after the event, and
refuses a likelihood object built on the full light curve (build it on `prepare_lc(lc, model)`).

**The explosion-time prior comes from the data** when the prior given to the fit does not name
`t_exp`: `LightCurve.explosion_time_prior(*, fallback_days=30.0)` returns `Uniform(last non-detection
before the first detection, first detection)` on the curve's own clock, every band counted; with no
earlier non-detection, `Uniform(first - fallback_days, first)`
(`whisper_cbpf.io.schema.EXPLOSION_FALLBACK_DAYS = 30`). A model with no `t_exp` prior takes this
window; a model's `Uniform` `t_exp` prior is cut to it (no overlap raises); a `Normal`,
`TruncatedNormal` or `Fixed` model prior is kept; a prior passed to `fit` wins. A limit shallower
than the first detection does not rule out an earlier explosion: pass a wider prior for shallow
limits. `whisper_cbpf.io.schema.explosion_window(time, upper_limit=None)` returns the numbers behind
it (`first_detection`, `last_non_detection`, `n_before`).

**Upper limits are fitted by default.** `space="auto"` resolves to flux for a light curve that
carries upper limits after the event (`likelihood.resolve_space(lc, space)`), so `make_likelihood`
and `likelihood="auto"` use the censored `GaussianLikelihoodWithUpperLimits`: detections are scored
on flux (`sigma_F = 0.921 F sigma_m`), limits by the probability that the flux was below them. The
limits' significance, `GaussianLikelihoodWithUpperLimits(lc, space="auto", upper_limit_sigma=None,
zeropoint_jy=3631.0)`, reads `lc.meta["upper_limit_sigma"]` (5 for both presets), else
`likelihood.DEFAULT_UPPER_LIMIT_SIGMA = 5`. An explicit `space="magnitude"` is refused.
`lc.where(upper_limit=False)` keeps a magnitude fit of the detections. ABC, ABC-SMC, `abc_gpu`,
`abc_smc_gpu`, SNPE and `snpe_gpu` compare values point by point and refuse limits after the event,
by name, before any simulation (`whisper_cbpf.samplers.base.NO_UPPER_LIMIT_SAMPLERS`).

`likelihood_max_opt=True` also climbs to the likelihood peak behind the fit's best draw and keeps it in
`result.info["likelihood_max_opt"]` ([A3](#a3-find-the-likelihood-peak)). If the optimisation
fails, the fit is still returned, with `info["likelihood_max_opt_error"]` and a warning.

| helper | signature | returns |
|---|---|---|
| `whisper_cbpf.samplers.prepare_lc` | `(lc, model=None, *, prior=None)` | the rows of `lc` a fit of `model` uses (`lc` itself when nothing is left out) |
| `SamplerResult.fitted_lc` | `(lc)` | the rows of `lc` the fit used (its recorded cut re-applied); pass it to `wp.waic`, `predictive_metrics` and your own checks |
| `whisper_cbpf.samplers.base.fitted_model` | `(result, model=None)` | the `Model` behind a fit: `model` when given (a name or an object), else the Model object the fit ran with (kept on every result in this session, so a factory-built model is found although it is not registered), else the registered model `result.model` names; `KeyError` for a loaded result of an unregistered model (pass `model=`) |
| `LightCurve.explosion_time_prior` | `(*, fallback_days=30.0)` | the `Uniform` above; raises "not enough data" with no detection |

```python
from whisper_cbpf.samplers import prepare_lc
rows = prepare_lc(lc.set_explosion_date(60000.0), "flare")     # the rows a fit would use
res = wp.fit(lc, "flare", sampler="mcmc", likelihood_max_opt=True)
res.info["excluded_pre_event"], res.info["likelihood_max_opt"]["max_log_likelihood"]
```

### Where the chains start: `init=`

`mcmc`, `emcee_jax`, `nuts_gpu`, `pymc_jax_gpu_vectorized` and `pymc_jax_gpu_parallelized` take
`init=`, through `wp.fit(lc, model, sampler=..., init=...)` or each sampler's own `fit`
(`MCMCSampler.fit(..., init="prior_scan", initial_guess=None, ...)`, `fit_emcee_jax(...,
init="prior_scan")`, `NUTSGPUSampler.fit(..., init=None, init_strategy="prior_scan", ...)`).

| `init=` | where the chains / walkers start | `info["init"]` |
|---|---|---|
| `"prior_scan"` (default) | Score max(1000, 4 n) prior draws in each prior's own coordinate and climb the best (by gradient for the JAX samplers, derivative-free for CPU `mcmc`). Start ~2 posterior sd around the climbed points that reach the best basin, farthest-first. | `"prior_scan"` |
| `"prior"` | Independent prior draws, each redrawn while its density is -inf. For CPU `mcmc` these are the pre-0.2.0 draws for the same seed. | `"prior"` |
| `{name: value}` or a `(k,)` array | A ball of 1e-3 of each prior's width (`DEFAULT_BALL_SCALE`), in its own coordinate. | `"ball"` |
| `(point, scale)` | The same ball, `scale` times the width. | `"ball"` |
| `(n, k)` array | One start per chain or walker, as given, in the sampled parameters' order. | `"per_chain"` / `"per_walker"` |
| a `SamplerResult` (ABC, a continuation, an alert's earlier cut) | With at least n usable draws (finite, strictly inside the box, finite density here): a random distinct subset. Otherwise a ball around its optimised likelihood maximum (`info["likelihood_max_opt"]`) or `best_params`, sized by the sd of its draws in each prior's own coordinate. | `"result"` |

Every start is checked against the box, the constraint wall and a finite density (CPU `mcmc` also
checks the model predicts some flux): a random start is redrawn, a given one is refused with the
reason. A ball start more than max(10, 2k) nats below its centre is pulled halfway back, up to 6
times. `info["init_detail"]` describes the start. The NUTS samplers keep `init_strategy=` (NumPyro's
`"uniform"`, `"median"`, ..., PyMC's `"jitter"`), and passing both raises; `emcee_jax` keeps
`init="box"`; CPU `mcmc` keeps `initial_guess=` / `initial_scatter=`, and combining it with `init=`
raises. `init=` to `abc`, `abc_smc`, `nested` or `snpe` raises: they draw from the prior.

The `"prior_scan"` spread tries each start again when its move is refused (more than max(10, 2k)
nats down, or behind a constraint wall): a fresh direction and its mirror image, at half the length,
for up to 20 rounds (`_diagnostics.SPREAD_ROUNDS`); a coordinate whose curvature cannot be read
across a wall gets a short step. An ensemble start (`emcee_jax`, `mcmc`, `fit_batch`) that still
does not span every parameter (walkers on one point, or on a few collinear ones) is refused with
the way out (`init="prior"`), before emcee's own "Initial state has a large condition number".

**Where the walkers move: `walker_coordinates=`** (`emcee_jax`, `mcmc`, `fit_batch`).
`"own"` (default) moves each walker in each prior's own coordinate: the natural log of a
`LogUniform` parameter, the parameter itself otherwise, with the log-Jacobian added, so the
posterior is the same. `"linear"` moves them in the parameters themselves (whisper 0.1.x). The
stretch move is affine-invariant only in the coordinates it moves in: in linear ones a parameter
spanning decades is explored slowly, and the supernova and TDE posteriors came out too narrow
([`VALIDATION.md`](VALIDATION.md) section 4). `info["walker_coordinates"]` records
`{"coordinates", "log"}` (the log columns). The draws, the summary and `samples_by_chain` are in
parameter units; `result.emcee_sampler`'s chain is in walker coordinates. The stuck-walker check
compares the walkers' log posterior in walker coordinates, where a flat `LogUniform` direction is
flat.

```python
abc  = wp.fit(lc, "flare", sampler="abc", n_simulations=5000, quantile=0.01)
post = wp.fit(lc, "flare", sampler="mcmc", init=abc)          # the ABC -> MCMC handoff
post.info["init"], post.info["init_detail"]["mode"]           # ('result', 'draws')
later = wp.fit(lc_more_data, "flare", sampler="mcmc", init=post)   # an alert's next cut
```

### Priors beyond boxes: `Normal`, `TruncatedNormal`, `Fixed`

`Normal(mu, sigma, name=None)`, `TruncatedNormal(mu, sigma, low, high, name=None)` (`low` or `high`
may be infinite; exact density with its truncation normalisation, inverse-CDF draws in either tail)
and `Fixed(value, name=None)` (a pinned parameter), beside `Uniform` and `LogUniform`. Each has
`sample(rng)`, `log_prob(x)`, `rescale(u)` and `bounds`; the Normal pair also `cdf`, `ppf` and
`std`. `Prior.fixed` returns `{name: value}`. `priors.log_prob_jax(prior, names=None)` takes all
five families, and `priors.ppf_jax(dist)` is the traceable inverse CDF.

| sampler | Normal / TruncatedNormal | Fixed |
|---|---|---|
| `mcmc`, `emcee_jax` | exact density | held at its value |
| `nuts_gpu`, `pymc_jax_gpu_*` | Normal as a Normal site; TruncatedNormal as a Uniform site on its CDF (`cdf_<name>`) through the exact inverse CDF | held at its value |
| `nested` | inverse CDF (`rescale`) | not a dimension of the unit cube |
| `abc` | drawn from the prior | drawn as its value |
| `abc_smc` | drawn from the prior; perturbed particles tested against it | never perturbed; left out of the kernel and the weights |
| `abc_gpu`, `abc_smc_gpu` | on the device through `priors.ppf_jax` | its exact value (`abc_smc_gpu`: kernel width 0) |
| `snpe`, `snpe_gpu` | torch Normal, or an exact torch TruncatedNormal | outside the torch prior and the network |

A `Fixed` parameter is not counted in AIC/BIC (`result.n_params` counts the free parameters only)
and stays a constant column of `result.samples`, in `result.best_params` and in `info["fixed"]`. A
prior where every parameter is Fixed raises.

```python
prior = wp.Prior({"mej": wp.LogUniform(1e-3, 1.0), "t_exp": wp.Normal(-3.0, 1.0),
                  "kappa": wp.TruncatedNormal(0.1, 0.05, 0.0, 1.0), "redshift": wp.Fixed(0.051)})
r = wp.fit(lc, model, sampler="nuts_gpu", prior=prior)
r.info["fixed"], r.n_params                                    # ({'redshift': 0.051}, 3)
```

### Free explosion time and redshift (JAX factories)

`supernova_model(model, band_names, redshift=None, dl_cm=None, *, ..., free=None,
redshift_prior=None, diffusion_grid=None, max_phase_days=None, epochs_per_decade=None)`,
`tde_model(band_names, redshift=None, dl_cm=None, *, ..., free=None, redshift_prior=None,
**engine_kwargs)`, `kilonova_model(band_names, redshift=None, dl_cm=None, *, ..., time_grid="auto",
free=None, prior=None, redshift_prior=None)`, and `kilonova_two_model` / `kilonova_three_model` with
the same `free=`, `redshift_prior=` and `time_grid="auto"` (full signatures in
[§10](#jax-model-factories--whisper_cbpfmodelsjax)).

- `free=["t_exp", "redshift"]` (either or both) makes the explosion (merger) time, on the light
  curve's own clock, and the redshift ordinary parameters, appended to `Model.parameters` in that
  order and traced: jit or vmap over many values compiles once.
- `t_exp` needs a prior: `prior=Prior({"t_exp": ...})`. In a fit, a `Uniform` model prior is cut to
  the data window; a prior passed to the fit is used as given. A supernova on fixed diffusion
  epochs needs a `t_exp` prior with a finite lower bound (a `Uniform`, or a `TruncatedNormal` with a
  finite `low`): the epochs are sized from the earliest explosion it allows, and a `Normal` raises.
- With the redshift free, pass `redshift=None, dl_cm=None`: the distance follows the redshift
  (Planck18). Its prior is `prior["redshift"]`, else `redshift_prior` (a `LightCurve.redshift_prior`
  hint or a distribution), else Uniform(0.001, 1).
- Epochs before the explosion carry no light (`mag_floor`). In a fit the rows before the first
  detection are left out of the likelihood (the pre-event rule above) and set the default `t_exp`
  window instead, so an upper limit there does not enter the likelihood.
- **A fitted redshift can bias a model comparison**: a model can move the source to buy a fit. Read
  the redshift posterior against its prior before ranking.
- `supernova_model(diffusion_grid=)`: `"data"` is redback's scheme on the observed epochs (exact to
  redback, epochs fixed at compile time); `"fixed"` diffuses on fixed source-frame epochs
  (`epochs_per_decade`, default 30, up to `max_phase_days`, default 200 d), interpolates only the
  bolometric luminosity and evaluates the SED at the observations: within 9e-4 mag of redback, and
  one compile for any epochs. `None` means `"data"` with nothing free and `"fixed"` otherwise. A
  concrete epoch past `max_phase_days` raises, naming the value to rebuild with; `log_density` (and
  so `fit_batch`, `likelihood_max_opt` and `profile`, which pass the epochs traced) runs the same check on the
  light curve's epochs under the fit's prior before it builds the density
  (`model.predict_jax.check_epochs(times, prior=None)`). `times=` sizes `max_phase_days` to the
  given epochs; `compare` passes the light curve's.
- The kilonova factories' `time_grid="auto"` is redback's grid with nothing free, the converged
  quadrature otherwise. `register_*` take `redshift=None, dl_cm=None` and forward them.

`luminosity_distance_cm(z, xp=numpy)` (`whisper_cbpf.models.cosmology`, also `wp.`): the Planck18
luminosity distance in cm from a 2000-node log-log table over [1e-4, 10], within 1e-6 mag of astropy;
`xp=jax.numpy` makes it traceable and differentiable; numpy input outside the table raises.
`whisper_cbpf.io.schema.redshift_distribution(hint)` maps a `redshift_prior` hint to a distribution
(`Uniform`/`LogUniform`, or `Normal`/`TruncatedNormal` → `TruncatedNormal(mu, sigma, low, high)`).

```python
model = wp.supernova_model("arnett", ["lsstg", "lsstr"], free=["t_exp", "redshift"],
                           prior=wp.Prior({"t_exp": wp.Uniform(-20.0, 0.0)}),
                           redshift_prior=lc.redshift_prior)
round(wp.luminosity_distance_cm(0.1) / 3.0857e24)           # 476 (Mpc)
```

The observed-epoch supernova photometry is `whisper_cbpf.models.jax.supernova`:
`build_fixed_grid(max_phase_days=200.0, epochs_per_decade=30, dense_resolution=1000,
first_epoch=0.01, t_pad=100.0, spacing="geometric", csm_interp=True)` → `FixedGrid` (`arrays`,
`n_epochs`, `max_phase_days`, `dense_times(t_last)`); `bolometric_at(model, fixed, tau_days, params,
*, t_last=None, magnetar_convention="1.15", interaction=True)`; `ab_magnitude_at(model, fixed,
tau_days, params, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, *, magnetar_convention="1.15",
interaction=True, mag_floor=40.0, **kw)`. `tde.diffusion(..., timesteps=100, geometric=False)`:
`geometric=True` finds each segment of a geometric grid by arithmetic (same values).

### ABC: the worst-point distance and single precision

`max_abs_z_distance(obs, obs_err, sim, bands=None)` (also `wp.`), registered as `"max_abs_z"`
(`"max_abs_z_jax"` on the GPU): `max_i |obs_i - sim_i| / err_i`, so `threshold=k` accepts a draw only
if every point is within k sigma of its own error. `abc` and `abc_gpu` accept `d <= k`, `abc_smc` and
`abc_smc_gpu` accept `d < epsilon`; a NaN or infinite simulated point is infinitely far. AIC/BIC still
come from the exact Gaussian likelihood. ABC-SMC's `min_epsilon="auto"` floor is derived for `chi2`
and warns with any other distance.

`ABCGPUSampler.fit(lc, model, prior=None, *, predict_jax=None, n_simulations=10000, quantile=0.01,
threshold=None, distance="chi2", simulate_noise=True, space="auto", scatter_param=None, seed=0,
chunk=250, max_logl_scan=None, precision=None)`: each draw costs one model evaluation, which yields
its distance and its exact log-likelihood; draws stay on the device. `precision=None` follows the
session, `"float32"` runs the sweep in single precision (kilonovae and the flare; the float64-only
supernova and TDE engines raise), `"float64"` runs double. `info["precision"]`, `info["x64_session"]`,
`info["postprocess_s"]` record what ran.

```python
res = wp.fit_ABC(lc, "flare", distance="max_abs_z", threshold=5.0, simulate_noise=False,
                 space="magnitude")
res.info["n_accepted"], res.min_distance      # worst point of the closest draw, in sigma
```

## A3. Find the likelihood peak

`likelihood_max_opt(result, lc, model=None, *, n_candidates=300, n_starts=5, tol=0.0001,
max_rounds=20, space=None, likelihood=None, backend="auto", seed=0, prior=None)` →
`LikelihoodMaxOptResult` (`wp.likelihood_max_opt`; the module is `whisper_cbpf.likelihood_max_opt`).
Also `SamplerResult.likelihood_max_opt(lc, model=None, **kwargs)` and
`fit(..., likelihood_max_opt=True)`, which keep the peak in `result.info["likelihood_max_opt"]`.

A sampler's best draw is a lower bound on the maximum likelihood, short by a different amount for
each model, so a ranking on the sampler's own BIC carries sampler noise. `likelihood_max_opt`
climbs from the fit's best draws to the peak of the fit's own likelihood (the space, likelihood
class and scatter term the fit recorded, the model's constraint wall, the rows the fit used, and
the prior the fit recorded unless `prior=` is given) and recomputes AIC and BIC there. The
posterior is not changed, and `wp.likelihood_max_opt` does not modify `result`. Why this matters, with numbers:
[`LSST_ALERTS.md` §5](LSST_ALERTS.md#5-why-the-ranking-uses-the-likelihood-maximum).

- **Coordinates.** Uniform, LogUniform and two-sided TruncatedNormal climb on [0, 1] in their own
  coordinate (log10 for LogUniform); Normal and one-sided TruncatedNormal in `(x - mu) / sigma`,
  bounded at the finite end. A finite bound is reached exactly. `Fixed` parameters are held.
- **Candidates.** Up to `n_candidates` draws, plus the fit's best draw and any earlier peak in
  `result.info["likelihood_max_opt"]`.
- **Climbs.** From the `n_starts` best: projected Adam (JAX backend), bounded L-BFGS-B (exact
  gradients on the JAX backend, central differences on the CPU), Nelder-Mead where L-BFGS-B did not
  converge; the two best restart until a round gains less than `tol`.
- **Never lower.** Every evaluated point is kept, so `max_log_likelihood >= start_log_likelihood`.
- **Backends.** `backend="auto"` uses the JAX density when the model has `predict_jax` and JAX
  implements the likelihood, else `model.predict` with scipy. Its compiled programs are kept in an
  LRU of `PROGRAM_CACHE_SIZE = 16` densities. Enable float64 for a JAX-backend optimisation.

`LikelihoodMaxOptResult` (frozen dataclass): `params`, `max_log_likelihood`, `start_log_likelihood`,
`gain`, `at_edge`, `aic`, `bic`, `n_data`, `n_params`, `n_evals`, `runtime_s`, `method`,
`sampler_log_likelihood`; `.enough_data`, `.to_dict()`, `LikelihoodMaxOptResult.from_dict(d)`.

| field | how to read it |
|---|---|
| `gain` | peak minus the sampler's best draw. Above ~1, the sampler never reached the peak |
| `at_edge` | parameters within 0.1 % of a prior bound (`EDGE_TOL`), in their own coordinate: widen that prior, or quote a bound |
| `aic`, `bic` | `nan`, with a "not enough data" warning, when `n_data <= n_params` |
| `sampler_log_likelihood` | the ln L the fit's own BIC used |

Errors name the cause and the fix: a light curve with another number of points, a best draw outside
the prior box (pass `prior=`), an unregistered model (pass `model=`), a fitted parameter neither in
the model nor read by the likelihood. A density-mismatch warning says to pass `space=` /
`likelihood=`.

```python
res  = wp.fit(lc, "flare", sampler="mcmc")
peak = wp.likelihood_max_opt(res, lc)
peak.max_log_likelihood, peak.gain, peak.at_edge, peak.bic
```

## A4. Check a fit

`SamplerResult.diagnostics(*, prior=None, likelihood_max_opt=None)` → `DiagnosticsReport`: one convergence
report with the checks that apply to the sampler that ran.

| sampler | checks |
|---|---|
| every fit | posterior draws > 0; data points > free parameters ("not enough data"); prior-edge pile-up: at most 5 % of draws in the outer 1 % of a prior range (log10 for LogUniform) |
| NUTS (`nuts_gpu`, `pymc_jax_gpu_*`) | divergences = 0; split R-hat < 1.01; bulk and tail ESS >= 100 per chain; log-likelihood R-hat < 1.01; no stranded chain; no frozen chain; prior-scan optimum no more than 10 nats above the best draw |
| emcee (`mcmc`, `emcee_jax`, `fit_batch`) | no stuck walker; nsteps / largest tau >= 50; R-hat across walkers < 1.05; bulk and tail ESS >= 400 |
| ABC | accepted draws >= 100 |
| ABC-SMC | distinct particles >= 100 |
| nested | stopped on dlogz; effective sample size >= 100 |
| SNPE | final draw by rejection |
| other samplers | their own `converged` flag |

`prior=` sets the prior for the edge check (default: the prior recorded with the fit).
`likelihood_max_opt=` takes a `LikelihoodMaxOptResult` or a float and adds a check ("gap to the
likelihood maximum") that the best draw is within 5 nats of the peak.
R-hat and ESS skip `Fixed` parameters. Thresholds are constants of `whisper_cbpf.results`
(`ENSEMBLE_RHAT_MAX = 1.05`, `ENSEMBLE_ESS_MIN = 400`, `N_OVER_TAU_MIN = 50`,
`MIN_INDEPENDENT_DRAWS = 100`, `EDGE_BAND = 0.01`, `EDGE_FRACTION_MAX = 0.05`, `LIKELIHOOD_MAX_GAP_MAX = 5.0`).

`DiagnosticsReport` (frozen dataclass: `sampler`, `model`, `kind`, `rows`): `.passed` (no check
failed; a check that could not run is "not checked", `passed=None`, and does not fail the report),
`.rows` (`DiagnosticRow(check, value, threshold, passed, reason)`), `.reasons` (`"check: reason"`
for every failed check), `.to_dict()`, and `print(report)`: the table, then a "Why:" section.

```python
report = res.diagnostics()
report.passed, report.reasons
```

## A5. Compare models

`compare(lc, models, sampler="auto", *, likelihood_max_opt=True, cache_dir=None,
evidence_check="auto", n_jobs=None, prior=None, seed=0, **fit_kwargs)` → `Comparison`

Fits several models to one light curve with the same settings, finds each fit's likelihood peak
(`wp.likelihood_max_opt`, so AIC and BIC come from the peak), attaches each fit's convergence
report, and ranks.

**`models`**: `Model` objects, registered names, or family names bound to `lc`: any
`wp.supernova_models()` name, `"magnetar"`, `"csm_shock_arnett"`, `"shock_cooling_arnett"`, and
`"tde"` / `"tde_gaussianrise"` (`whisper_cbpf.compare.FAMILY_ALIASES`, `TDE_NAMES`). A bound family
takes the light curve's bands, fixes the redshift at `lc.redshift` when it is known and fits it from
`lc.redshift_prior` otherwise, and fits the explosion time over `lc.explosion_time_prior()` unless the
time reference is the explosion.

**`sampler`**: `"auto"` is `emcee_jax` on the GPU for a model with `predict_jax` when JAX sees a GPU,
CPU `mcmc` otherwise; any registered name runs every model with it; `"nested"` gives every model a
ln Z, and the ranking is then by ln Z.

**Ranking.** By ln Z when every ranked model has a nested run that stopped on dlogz; otherwise by
BIC at the optimised likelihood maximum, `-2 ln L_max + k ln n`. `delta` is `BIC - BIC_best` (or `ln Z_best - ln Z`); `weight`
is `exp(-delta/2)` (or the evidence weight), normalised over the ranked models; `grade` is Jeffreys'
scale on ln B (`delta / 2` under BIC): below 1.15 "inconclusive", below 2.30 "substantial", below
4.61 "strong", otherwise "decisive" (`GRADE_CUTS`). The winner's row is graded against the
runner-up, every other row against the winner.

**ln Z of a model behind a constraint wall** (the supernova and TDE families, the redback adapter)
is computed with its prior renormalised to the region the wall allows, as bilby normalises a
constrained prior: nested sampling treats a walled draw as `-inf`, and `ln Z` gains `-ln f`, `f`
the allowed fraction of 20 000 prior draws (`samplers.nested.CONSTRAINT_FRACTION_DRAWS`; its error
is added to `log_evidence_err`). Without it every walled model's ln Z was low by `-ln f` (about 1
nat for Arnett, the magnetar and the TDE). `info["constraint_prior"]` records `allowed_fraction`,
`n_draws`, `log_evidence_correction`, `correction_err` and `log_evidence_before_correction`
(`None` for a model without a wall).

**Left out, each with `left_out_reason`**: a failed fit; no posterior draws; a fit to another number
of points than the most common one (for example after its fit dropped pre-event rows); among the
fits with that number, a fit to other rows or scored in another space (flux or magnitude) than the
most common rows and space; `k >= n` ("not enough data"); no finite BIC. If nothing can be ranked, `winner` is `None` and the headline
says "not enough data".

**`evidence_check`**: `"auto"` runs CPU nested sampling on the top two when their BIC ln B is below
ln 10 (`EVIDENCE_CHECK_LN_B`); `True` always; `False` never. If ln Z disagrees with BIC, the
winner's grade becomes "inconclusive" and the reason is given. About 8-12 minutes per supernova
family on a 34-point alert, serial: use `n_jobs=`, `cache_dir=`, or `False`.

**`cache_dir`**: fits go through `wp.fit_cached`, so a rerun loads them and an interrupted comparison
resumes. **`prior`**: a `Prior` for every model, or `{model: Prior}`; for a bound family it
overrides the named parameters of the default prior, and a `t_exp` prior given this way is used as
given (not cut to the data window): `prior={"arnett": Prior({"t_exp": Uniform(t1 - 30, t1)})}`
widens the window when the non-detections are shallow. A supernova family's `t_exp` prior needs a
finite lower bound. **`fit_kwargs`**: passed to every fit; a
setting one of the samplers does not take is refused before any fit runs.

`Comparison`:
- `.table`: one row per model, ranked first, best first: `model, sampler, n_params, n_data,
  max_log_likelihood, aic, bic, log_evidence, log_evidence_err, delta, weight, grade, converged,
  problems, status, left_out_reason` (`whisper_cbpf.compare.COLUMNS`).
- `.results`, `.peaks` (`{model: LikelihoodMaxOptResult}`), `.winner`, `.criterion` (`"ln Z"` or
  `"BIC"`), `.diagnostics`, `.evidence`, `.evidence_check`, `.n_data`, `.problems`, `.timing`
  (`fit_s`, `likelihood_max_opt_s` and `evidence_s` per model), `.settings`, `.lc`, `.models`.
- `.summary()`: the winner and its grade, the table, the models left out and why, the evidence check,
  and the caveats, as text.
- `.save(path, *, overwrite=False)` and `Comparison.load(path, *, models=None)` round-trip the table,
  peaks, results, diagnostics and data. A family bound by name is bound again from the saved light
  curve and the prior its fit recorded; pass `models=` for models you built with a factory.
- `.facts(**kwargs)`, `.report(path, **kwargs)`, `.forecast(times, bands, **kwargs)` and
  `.discriminate(times, bands, survey_depth=None)` hand off to [A7](#a7-forecast-and-choose-the-next-observation)
  and [A9](#a9-facts-and-the-html-report).

A likelihood maximum more than 1 ln L above the sampler's best draw (`LIKELIHOOD_MAX_GAIN_FLAG`)
and a peak on a prior edge are listed in `problems`.
When a ranked model fits the redshift, the comparison says so.

```python
cmp = wp.compare(lc, ["arnett", "magnetar", "csm_shock_arnett", "tde"])
print(cmp.summary())
cmp.save("sn_d30"); cmp.report("sn_d30/report.html")
```

## A6. Save, reload and resume

| name | signature | what it does |
|---|---|---|
| `SamplerResult.save` | `(path, *, overwrite=False)` → `Path` | A `.npz` path writes one file; any other path a directory with `manifest.json`, `result.json` and `arrays.npz`. Saves the draws (by chain for NUTS, PyMC and emcee), the summary, best fit, AIC/BIC and `info`, and a manifest: whisper version and git state, package versions, the model and the prior used, the sampler with its seed, explicit settings and applied defaults, devices, and timing (`runtime_s`, `run_s`, `compile_s`, `init_s`, `warmup_s`, `sampling_s`, `postprocess_s`, `wall_s`). Live objects are listed under `not_saved`. `FileExistsError` unless `overwrite=True`. |
| `load_result` | `(path)` → `SamplerResult` | Every field equal to the saved one; `result.samples_by_chain` restored; `result.loaded_from` set. `FileNotFoundError`; `ValueError` for a path that is not a saved result, a newer format, or a hash that no longer matches ("modified or truncated"). |
| `SamplerResult.provenance` | dict | Recorded by every registered sampler's `fit`: environment, data hash, model and prior, sampler settings and seed, JAX devices, wall time. Empty for a result built by hand. |
| `fit_cached` | `(lc, model, sampler, cache_dir, **fit_kwargs)` → `SamplerResult` | Loads an identical fit saved in `cache_dir`, or fits and saves. The hash covers the data, the model (name, parameters, prior, the identity of `predict`), the sampler with its explicit and default settings, the whisper version, and the band system and float64 setting; `progress` is ignored. Saves go to a hidden directory renamed into place, so an interrupted batch finishes only what is missing. `result.provenance["cache"]` holds the configuration and hash; `whisper_cbpf.results.cache_config(lc, model, sampler, **fit_kwargs)` returns `(config, hash)` without fitting. |

The light curve itself is not saved with a fit, only its hash: pass it again for
`likelihood_max_opt`, plots and predictive checks. `Comparison.save` does save it.

```python
res = wp.fit(lc, "flare", sampler="mcmc", nsteps=600, burnin=200, seed=0)
res.save("out/flare_mcmc")                        # or "out/flare_mcmc.npz"
same = wp.load_result("out/flare_mcmc")
for seed in range(6):
    wp.fit_cached(lc, "flare", "mcmc", "fits/", nsteps=4000, seed=seed)
```

## A7. Forecast and choose the next observation

`forecast(result, times, bands, *, lc=None, model=None, n_draws=400, quantiles=(0.025, 0.16, 0.5,
0.84, 0.975), survey_depth=None, seed=0)` → `pandas.DataFrame` (`wp.forecast`; also
`SamplerResult.forecast(times, bands, **kw)` and `Comparison.forecast(times, bands, **kw)`, which
adds a leading `model` column)

Predicted AB magnitudes at (time, band) cells from a fit's posterior draws. Every combination of
`times` x `bands` is a cell, time-major; `times` are on the fit's clock.
- **Draws:** up to `n_draws` posterior rows, without replacement, chosen by `seed`.
- **Model:** `model=`, or the registered model `result.model` names; a JAX twin reads the draws
  through `param_aliases`.
- **Evaluation:** a model with `predict_jax` runs in one compiled batch (the GPU when JAX has one),
  constraint wall applied, compiled once per (model, epochs, bands) and cached; any other model uses
  `Model.predict` once per draw.
- **Columns:** `time, band, mean_mag, sd_mag, q2.5, q16, q50, q84, q97.5, frac_too_faint, frac_dark,
  n_draws, depth`. Statistics are over the draws that predict light; dark draws (zero or non-finite
  flux, a JAX `mag_floor`, or fainter than `DARK_MAG = 35`) are counted in `frac_dark`. With fewer
  than two lit draws a cell is NaN, with a "not enough data" warning.
- **`survey_depth`:** one 5-sigma magnitude or `{band: mag}`; it sets `frac_too_faint` (draws fainter
  than the depth, dark included) and `depth`.
- **`.attrs`:** `model, sampler, n_draws, n_samples, seed, quantiles, survey_depth, time_label,
  backend, eval_s, dark_mag, lc_name`.
- **With `lc=`:** times on another clock are refused, and the message gives the shift.

`discriminate(results, times, bands, *, survey_depth=None, phot_sigma=None)` → `pandas.DataFrame`
(`wp.discriminate`; `Comparison.discriminate(times, bands, survey_depth=None)` does it for the top
two ranked models)

For every pair of models and cell, `D = |mean_a - mean_b| / sqrt(sd_a^2 + sd_b^2 + sd_phot^2)`.
- `results`: `{name: SamplerResult}`, or `{name: forecast DataFrame}` over the same cells and depth.
  Fits with no draw are named in `.attrs["left_out"]`.
- `sd_phot`: `phot_sigma` (one value or per band); otherwise, with a depth,
  `SIGMA_AT_LIMIT * 10**(0.4 (m - depth))` (0.217 mag at the 5-sigma depth); otherwise 0.
- `observable`: both means brighter than the depth and fewer than half of each model's draws too
  faint (without a depth: fewer than half dark).
- Columns: `model_a, model_b, time, band, mean_a, mean_b, sd_a, sd_b, sd_phot, D, observable`.
  `.attrs["best"]` is the observable row with the largest D (or `None` with `.attrs["reason"]`).
- Reading D: near 1 or below, one measurement cannot tell the two models apart; 3 or more, it can.

```python
fc = wp.forecast(res, lc.time.max() + np.array([0.5, 1, 3]), ["lsstg", "lsstr"], lc=lc,
                 survey_depth=24.5)
d = cmp.discriminate(lc.time.max() + np.array([1, 3, 7]), ["lsstg", "lsstr"], survey_depth=24.5)
d.attrs["best"]
```

## A8. Plot

Every plot returns `Axes` (never calls `plt.show()`, so a bare call in a notebook draws exactly one
image), is in AB magnitudes with brighter up, and labels the time axis from the light curve's day 0.

| function | what it draws |
|---|---|
| `plot_forecast(forecast_df, ax=None)` | per band, the median with the 68 % and 95 % intervals, the 5-sigma depths (dashed), open markers where at least half the draws are too faint. One model at a time: select one model's rows of a `Comparison.forecast` table |
| `plot_models(results_or_comparison, lc, ax=None, bands=None, *, n_draws=200, seed=0)` | one column per band: the data and each model's median and 68 % band (rank order and weights for a Comparison, drawn with its own model objects); below, residuals at the detections each fit used. Pre-event rows are open grey squares, with no residual and no curve before a declared event. With `ax=`: one band, no residuals |
| `plot_model_comparison(comparison, history=None)` | weight bars, best on top, with the gap to the best and the grade, and the reasons models were left out. `history={label: comparison}` (oldest first) adds each model's weight at each decision point |
| `plot_widths(result, prior=None, *, ax=None)` | posterior 68 % width over prior 68 % width per free parameter, in the prior's own coordinate; a line at 1 (posterior = prior) and the facts file's `prior_dominated` threshold. `whisper_cbpf.plotting.posterior_width_ratios(result, prior=None)` returns the numbers |
| `plot_ppc(..., intervals=(68, 95))` | shades both predictive intervals by default; `intervals=(95,)` gives the 0.1.1 figure |
| `plot_corner(..., log_params="auto")` | parameters with a LogUniform prior in the fits passed in go on log10 axes, labelled `log10 <name>`; `log_params=None` turns it off |

```python
wp.plot_models(cmp, lc)
wp.plot_model_comparison(cmp)
wp.plot_forecast(fc)
```

## A9. Facts and the HTML report

`result_facts(result, lc, *, model=None, prior=None, thresholds=None)` → dict (also
`SamplerResult.facts(lc, **kwargs)`): every number a reader
quotes from one fit, each computed by a stated rule, JSON-ready (no NaN, no time stamp, no path) and
sealed with a `sha256`. Keys: `fit` (n, k, draws, the sampler's max ln L, the optimised likelihood
maximum and its gain as `optimised_max_log_likelihood` and `likelihood_max_gain`, AIC and BIC or
"not enough data", ln Z); `parameters` (median, p16, p84, the prior, the
posterior/prior 68 % width ratio, `prior_dominated` (ratio above 1/sqrt(2)), `at_prior_edge`, a
one-line `reading`); `fixed`; `narrowing` (redshift and `t_exp`); `data` (per band the brightest
detection, the state, rise and decline rates with errors; the latest colours with errors; the last
non-detection before the first detection); `absolute_magnitude` (asymmetric, Planck18);
`diagnostics`; `caveats` (`converged`, `stranded_walkers`, `not_enough_data`, `no_posterior_draws`,
`large_likelihood_max_gain`, `prior_dominated`, `at_prior_edge`, `data_differs_from_fit`); `readings`,
`thresholds`, `rules`, `inputs` (SHA-256 of the data, the draws, the model identity and the prior),
`sha256`. `thresholds` overrides `whisper_cbpf.facts.DEFAULT_THRESHOLDS`
(`prior_dominated_ratio`, `edge_band`, `edge_fraction`, `likelihood_max_gain_large`,
`colour_max_separation_days`, `grade_cuts_ln_b`); an unknown key raises.

`comparison_facts(comparison, lc=None, *, thresholds=None)` → dict (also `Comparison.facts()`):
`ranking` (criterion, winner, runner-up, ln B and grade, and per model its rank, delta, weight,
grade and diagnostics; the models left out with the reason), `models` (`{name: result facts}`),
`data`, `absolute_magnitude`, `caveats` (`models_left_out`, `winner_converged`,
`winner_prior_dominated_parameters`, `evidence_check_disagrees`, `models_with` each flag,
`comparison_problems`), `thresholds`, `rules`, `inputs`, `sha256`.

`write_facts(facts, path)` → `Path`: indented, NaN-free UTF-8 JSON, the same bytes for the same facts;
a path that is not `.json` is a directory that gets `facts.json`.

`report(comparison, lc, out, *, forecast_times=None, decision=None, title=None)` → `Path` (also
`Comparison.report(path, **kw)`): one self-contained HTML page (inline CSS, base64 PNG figures, no
script, no external URL; readable on a phone, light or dark). Sections: the answer and caveats; an
optional decision card (only when `decision=` is passed, labelled as supplied by the caller); the
ranking; the figures (models over the data, weights, posterior/prior widths, the winner's corner
plot, the forecast); the parameters; what the data show; every diagnostic; the forecast, when
`forecast_times=` is given, with where the leading models differ most; and the provenance, with the
facts file as a download. Every number comes from `comparison_facts`; the same inputs give the same
bytes. A figure or forecast that cannot be made is left out with a warning and its reason on the
page. `out` ending in `.html` is the file; otherwise a directory that gets `report.html`.

```python
facts = cmp.facts()
facts["ranking"]["winner"], facts["ranking"]["grade"]
wp.write_facts(facts, "out/")
cmp.report("out/", forecast_times=[t_next, t_next + 3])
```

## A10. Many alerts, and how fast

### `log_density(lc, model, *, space="auto", likelihood="auto", prior=None, bucket="auto")` → `LogDensity`

The log-posterior of one light curve under one JAX model (`whisper_cbpf.samplers.jax._adapters`),
with the likelihood `wp.fit` would use and the pre-event rule applied.
- `.fn(theta)`: the log posterior, `-inf` outside the prior's support and past a constraint wall;
  differentiable, and works inside `jax.jit`, `jax.vmap` and `jax.grad`.
- `.names` (sampled parameters; `Fixed` ones are in `.fixed`), `.lows` / `.highs`, `.n_data`
  (observations, padding excluded), `.log_likelihood(theta)`, `.shared(theta, data)` (the jitted
  program, returning log posterior and log likelihood), `.data`, and `.pre_event` (the pre-event
  rule applied, as a fit records it in `info["pre_event"]`: the rows at or before the event are
  not in the density).
- **Data as arguments.** Observations are padded to a bucket (16, 24, 32, 48, 64, 96, ...:
  `bucket_size(n)`) by repeating the latest epoch with mask 0, and the prior's numbers travel as a
  table, so every light curve with the same model, prior families, likelihood and bucket reuses one
  compiled `.shared`. A model that needs concrete epochs (a supernova on `diffusion_grid="data"`, a
  kilonova on redback's grid) or a float32 session on a clock with |t| >= 1e3 closes over its
  epochs instead; `.data_as_argument` is then `False` and `.reason` says why.
- Raises `ValueError` for a model without `predict_jax` or with its own `log_prob_jax`, no prior, a
  prior missing a parameter, every parameter Fixed, a `bucket` smaller than the light curve, an
  empty light curve, or a supernova on fixed diffusion epochs whose epochs, at the earliest `t_exp`
  and lowest redshift the prior allows, reach past its `max_phase_days` (the traced epochs cannot
  be checked inside the density); `NotImplementedError` for the mixture likelihood.

### `fit_batch(lcs, model, *, sampler="emcee_jax", nwalkers=32, nsteps=5000, burnin=1000, thin=10, seed=0, prior=None, space="auto", likelihood="auto", init="prior_scan", metrics=True, walker_coordinates="own")` → `list[SamplerResult]`

One model fitted to many light curves in one compiled on-device loop
(`whisper_cbpf.samplers.jax.batch`). Each light curve gets its own ensemble of `nwalkers` walkers,
moved by emcee's stretch move (a = 2) with its randomised red-blue split; the kept states are
emcee's `get_chain(discard=burnin, thin=thin)`. Light curve i uses its own stream, `seed + i` (or
`seed[i]`), so `fit_batch([a, b], m, seed=0)[1]` equals `fit_batch([b], m, seed=1)[0]`. `prior=` is
one Prior or one per light curve (same families; their numbers are data, so they share the compile);
the pre-event rule sets each light curve's `t_exp` window. `init=`: `"prior_scan"` (default,
`emcee_jax`'s start, every light curve's climb batched), `"prior"`, a `(K, nwalkers, ndim)` array, or
one start per light curve in any form `emcee_jax` takes. `walker_coordinates=` as for `emcee_jax`
(see "Where the walkers move" in A2).

Each result has `sampler="emcee_batch"`, draws, summary, best fit, AIC/BIC from the best kept draw,
`samples_by_chain`, the `emcee_jax` diagnostics plus `info["frozen_walkers"]` and
`info["n_nan_proposals"]`, and `info["batch"]` (`n_alerts`, `alert_index`, `bucket`, `program`, and
the batch's `compile_s`, `start_s`, `run_s`, `run_s_per_alert`); `runtime_s` is this light curve's
share. `diagnostics()`, `save()` and `load_result()` work. One warning names the unconverged fits.
Raises `ValueError` for a sampler other than `emcee_jax`, an unknown `init`, `burnin >= nsteps`, a
`thin` that keeps no draw, fewer than `2 * ndim` walkers, mismatched `seed` / `prior` / `init`
lengths, light curves that do not share the sampled parameters, or a light curve whose density is
`-inf` at every prior draw; `TypeError` for a single `LightCurve`.

Measured on one A6000, float64, 60 x 10 000 chain, per alert at 512 alerts per call: free `arnett`
5.33 s chain + 0.31 s start; TDE 3.26 s + 0.25 s; alone (K = 1) 10.7 s and 53.7 s.

### `profile(model, lc, *, batch_sizes=(1, 64, 480, 4096), grad=True, prior=None, space="auto", likelihood="auto", repeats=5, seed=0)` → `ProfileReport`

Times `jit(vmap(log density))` and, with `grad=True`, `jit(vmap(value_and_grad))` on B prior draws
per batch size, with the data as arguments (`whisper_cbpf.profile`). The compile is timed apart; the
call time is the median of `repeats` calls. `ProfileReport`: `rows` (per batch size and kind:
`compile_s`, `call_s`, `per_eval_us`, `temp_bytes`, `temp_bytes_per_eval`, `skipped`),
`finite_fraction`, `gradient` (finite fraction, zero fraction, median |g|), `peak_bytes`;
`seconds_per_eval(kind="value")`, `seconds_per_alert(*, nwalkers=32, nsteps=5000)` (chain plus the
start's scan draws, not its climb), `to_dict()`. A program whose scratch memory exceeds 90 % of the
device's free memory is skipped, with the reason.

### `capacity(cost, *, hours=12.0, n_gpus=1, nwalkers=32, nsteps=5000)` → dict

Alerts that fit in `hours` on `n_gpus` busy cards: `alerts_exact = hours x 3600 x n_gpus /
seconds_per_alert`. `cost` is seconds per alert (for example a `fit_batch` result's `runtime_s`), a
`ProfileReport` (converted at `nwalkers` / `nsteps`), or a list of them, summed. Returns `alerts`,
`alerts_exact`, `alerts_per_gpu_hour`, `seconds_per_alert`, `hours`, `n_gpus`, `basis`,
`assumptions`; device time only.

```python
round(wp.capacity(636.7, hours=12)["alerts_exact"])      # 68: five families, one alert at a time
wp.capacity([5.64, 3.51], hours=12, n_gpus=5)["alerts"]  # 23606: arnett + TDE per alert, batched
```

### `run_jobs(jobs, *, gpus="auto", cpu_cores=None, cache_dir=None, retries=1, timeout=None)` → `JobsReport` and `Job(lc, model, sampler, kwargs={}, name=None)`

Many fits, one fresh Python process per job, over the GPUs and CPU cores this run may use
(`whisper_cbpf.parallel`).
- **Jobs**: a `Job`, a tuple `(lc, model, sampler[, kwargs])`, or a dict. Names default to
  `<lc.name>__<model>__<sampler>` and must be unique.
- **Devices**: jobs with a JAX sampler run one per card at a time, with `CUDA_VISIBLE_DEVICES=<uuid>`
  and `JAX_PLATFORMS=cuda`; other jobs run on CPU cores beside them (`JAX_PLATFORMS=cpu`). Every job
  is pinned to its own block of cores before it imports anything, and gets `NCCL_P2P_DISABLE=1`.
- **`gpus`**: `"auto"` (the idle cards this process may see: under 512 MiB and 10 % utilisation,
  checked again before every launch; with none, GPU jobs run on the CPU with a warning), a list of
  `nvidia-smi` indices, or `None` (CPU only).
- **`cpu_cores`**: `None` (60 % of the allowed cores), an int, or a string such as `"0-15,32"` or a
  list.
- **`cache_dir`**: each job calls `wp.fit_cached`, so a rerun fits only what had not finished; logs
  and the summary are written there (a new temporary directory with `None`).
- **`retries`**: how many more times a failed job runs. **`timeout`**: seconds per attempt, start-up
  and compile included; the job gets SIGTERM, then SIGKILL after `STOP_GRACE_S = 10` s, and counts as
  a failed attempt.
- **Returns** `JobsReport`: `rows` (name, sampler, model, device, cores, status, attempts, wall_s,
  resumed, error, log, result), `passed`, `failed`, `table()`, `result(name)`, `results`.
- **Files**: `run_jobs.log`, `jobs/<name>/log.txt`, `run_jobs_summary.json`.
- **Requirements**: everything a job carries is pickled (a factory-built model is picklable; a
  `predict` defined in a notebook cell or as a lambda is not). A script must call `run_jobs` under
  `if __name__ == "__main__":`. Ctrl-C stops the running jobs, writes the summary and raises again.

```python
jobs = [wp.Job(lc, model, "emcee_jax", kwargs={"seed": 0}) for lc in alerts]
report = wp.run_jobs(jobs, cache_dir="runs/", timeout=3600)
fits = report.results                               # {name: SamplerResult}
```

## A11. Check that two models agree

`check_parity(model_a, model_b, params_or_posterior, times, bands, *, tolerance=0.02, n=200, seed=0)`
→ `ParityReport` (`whisper_cbpf.validation`)

Whether two models predict the same band magnitudes at the same parameters: a JAX port against its
redback twin, two grid settings, two photometry modes. Both are called through `predict(params,
times, bands)` and must return Jy on the same clock. `params_or_posterior` is a dict, a list of
dicts, a `Prior` (`n` draws in sequence from `default_rng(seed)`), a DataFrame of samples (`n` rows),
or a `SamplerResult` (its best fit plus `n - 1` posterior rows). Parameters named differently are
paired through `Model.param_aliases`; `times` are sorted first.

A point is compared when either model is brighter than 30 mag (`PARITY_FAINT_MAG`). One model dark
where the other is bright gives an infinite difference and fails; so does a NaN flux. With nothing
brighter than 30 mag the check fails with "Not enough data". `ParityReport`: `passed` (`max_abs <=
tolerance`), `reason`, `n_draws`, `n_points`, `median_abs`, `p99_abs`, `max_abs`, `per_band`
(`{band: {n, median_abs, p99_abs, max_abs, median_signed, passed}}`), `worst` (`PARITY_N_WORST = 5`
draws), `nonfinite`, `dmag` (signed `mag_a - mag_b`, NaN where not compared), `table()`.

```python
from whisper_cbpf.models import redback_adapter as ra
cpu = ra.redback_model("arnett", ["ztfg", "ztfr"], redshift=0.051, constraint=None)
gpu = wp.supernova_model("arnett", ["ztfg", "ztfr"], 0.051, ra.redback_luminosity_distance_cm(0.051),
                         constraint=None)
report = wp.check_parity(gpu, cpu, cpu.default_prior, t, band_per_epoch)   # 200 prior draws
report.passed, report.max_abs            # (True, 5.3e-13) on SN2025pgp's ZTF epochs
```

---

# Part B — by subsystem

---

## 1. `LightCurve`  (`io.schema`)

**`LightCurve` is a subclass of `astropy.table.Table`.** Per-point quantities are **columns** (`time`,
`band`, `magnitude`, `magnitude_err`, `flux`, `flux_err`, `upper_limit`, `system`, `lambda_eff`,
`zero_point`, plus any you add) and scalar metadata lives in **`.meta`** (`name`, `redshift`,
`data_mode`, `luminosity_distance`, `redshift_prior`, `dm`, `refmjd`, …). So full table semantics work:

```python
lc['absmag_shift'] = lc['magnitude'] + 5        # add / compute columns
bright = lc[lc['magnitude'] < 18]               # boolean-mask slicing (keeps the subclass + .meta)
lc.sort('time'); lc.group_by('band')            # any astropy Table method
lc()                                            # __call__ -> the table itself
```

Construct from arrays — `LightCurve(time=, band=, magnitude=|flux=, …, name=, redshift=, data_mode=)`
(requires `band` + at least one of `magnitude`/`flux`; `flux` is flux density in Jy) — or from anything
`Table` accepts. `data_mode` ∈ `{flux_density, magnitude, flux}` (inferred from the columns when
omitted). Redshift is validated at construction: finite & `≥ 0`; `z == 0` requires `luminosity_distance`
(Mpc); negative/NaN raises; `None` → *unknown* (`redshift_known=False` + a default `redshift_prior`).

For convenience the common quantities are **also** attributes: `lc.time` / `lc.flux` / `lc.band` … return
the column data (`None` if absent; settable); `lc.redshift` / `lc.data_mode` / `lc.name` /
`lc.redshift_known` / `lc.luminosity_distance` / `lc.output_format` read `.meta`; `lc.n_points`,
`lc.bands`, `lc.snr` are derived.

Methods (return a new `LightCurve` unless noted): **`where(**constraints)`** (`col` / `col_min` /
`col_max` / `col_not`, list = OR — `lc.where(band='r', time_min=58000, upper_limit=False)`);
`select_bands` / `select_time_window` / `select_snr`; `add_flux(zeropoint_jy=3631.0)` / `add_mag(...)`
(constant **AB** zero point — pass `zeropoint_jy=lc.zero_point` to opt into per-band; raise for
`data_mode='flux'`); `resolve_bands(svo_fallback=True)` (fill `lambda_eff`/`zero_point`);
`set_explosion_date(mjd)` (observer-frame days since explosion); `set_time_reference(mjd, label)`
(the same shift when day 0 is not the explosion, e.g. `label="first detection"`: kept in
`meta['time_reference']` / `meta['time_reference_mjd']`, so plots name it and `calc_phase` counts
from it; a model still reads `time` as days since its own day 0); **`calc_phase(reference=, redshift=,
peak=, hours=)`** (rest-frame phase `(t − ref)/(1+z)`); **`calc_absmag(dm=, redshift=, ebv=, rv=3.1,
extinction=)`** (distance modulus from `z`/`luminosity_distance` + Milky-Way extinction via CCM89 or an
explicit `{band: A_mag}` dict → `absmag` column); `to_dataframe()` (→ `to_pandas()`); `lc()` returns the
table itself.

## 2. `load_lightcurve(path, *, ...)` → `LightCurve`  (`io.loader`)

Auto-maps columns; key options: `column_map`, `band_lookup` (broadband grouping), `default_band`,
`quality_cuts`, `flag_filters={'catflags':0}`, `time_min/time_max`, `bands`, `min_snr`,
`explosion_date`, `delimiter`. Detects an `upper_limit` column. Raises `ValueError` for a missing
time / band / measurement column. `path` may also be a DataFrame or a list of records. For ZTF and
LSST alert photometry, `survey="ztf" | "lsst"` and `limits=` read the brokers' own fields: see
[A1](#a1-load-alert-photometry-survey-presets).

`LightCurve.explosion_time_prior(*, fallback_days=30.0)` returns the explosion-time prior the data
define ([A2](#what-every-fit-does)).

Ingestion options:

| Argument | Default | Description |
|---|---|---|
| `redshift` | `None` | Explicit redshift; otherwise a `redshift` column is used, else unknown (warns). |
| `luminosity_distance` | `None` | Mpc; required when `redshift == 0`. |
| `data_mode` | `None` | `flux_density`/`magnitude`/`flux`; inferred from the columns when omitted. |
| `flux_unit` | `None` | astropy unit of the flux column — F_ν (Jy/mJy/µJy) or F_λ (erg/s/cm²/Å). `None` → warn + assume Jy. |
| `magnitude_unit` | `None` | Must be dimensionless AB; a flux unit raises. `None` → warn + assume dimensionless. |
| `resolve_band_info` | `True` | Fill per-point `lambda_eff` + `zero_point` from the bands. |
| `svo_fallback` | `True` | Query SVO for bands missing from `FILTER_LOOKUP`. |
| `normalize` | `True` | Apply the band-alias map (`DEFAULT_BAND_ALIASES` + `band_aliases`) to the band column. `False` keeps the raw survey labels — use it when your labels are already canonical, or when an alias would collide with a band you mean literally. |
| `band_aliases` | `None` | Extra `{raw: canonical}` alias entries, **merged over** `DEFAULT_BAND_ALIASES` (so you can override a built-in alias). Ignored when `normalize=False`. |
| `drop_nonfinite` | `True` | Drop rows with non-finite values / non-positive errors (upper limits are kept). `False` keeps them — you then own the NaN handling downstream. |

Column auto-detection uses the case-insensitive synonym table `io.loader.CANONICAL_SYNONYMS`
(`{canonical_name: [accepted spellings]}`); `column_map` overrides it per column. When the redshift is
unknown the light curve records `redshift_known=False` and `io.schema.DEFAULT_REDSHIFT_PRIOR`.

## 3. Bands & SVO resolution  (`io.bands`, `io.svo`)
`DEFAULT_BAND_ALIASES` and `FILTER_LOOKUP` (97 labels → 11 effective bands; ZTF `ZTF_g`/`ztf_g`-style names included, aliases matched case-insensitively). `lsstu` groups to U-band and `y`/`Y`/`lssty`/`y-p1` to their own **y-band** (LSST y, 9710 Å), which up to 0.1.0 was folded into z-band with z's 8679 Å.

These groups are for ingestion, plotting and quick looks. **A model integrates over a filter, not a
group**: models resolve labels with `whisper_cbpf.resolve_filter` (§4b), which refuses a grouped label.

- **`normalize_band(band, aliases=None, warn_unknown=False, known=None)`** → the canonical label
  (whitespace-trimmed, alias applied case-sensitively first, then case-insensitively). `aliases` is
  merged **over** `DEFAULT_BAND_ALIASES`. `warn_unknown=True` warns for a label absent from `known`
  (a collection of accepted labels) and returns it unchanged — it never raises.
- **`normalize_bands(bands, aliases=None, warn_unknown=False, known=None)`** → `np.ndarray`, the
  vectorized form of the above (this is what `load_lightcurve(normalize=True)` applies).
- **`group_bands(bands, lookup=None, default=None, warn_unknown=False)`** → `np.ndarray` of effective
  broadbands. `lookup=None` means `FILTER_LOOKUP`. Labels absent from the lookup pass through unchanged
  (or become `default`); `warn_unknown=True` emits one warning listing the distinct unmapped labels.
- **`unmapped_bands(bands, lookup=None)`** → the sorted distinct labels not covered by `lookup` —
  the "what will not group?" check to run before a fit.

- **`resolve_band(band, *, lookup=None, svo_fallback=True, lambda_eff_hint=None, warn=True)`** →
  `{group, lambda_eff (Å), zero_point (Jy), filter_id, source}`. Order: `FILTER_LOOKUP` group →
  `LSST_BAND_INFO` (optical anchored to **LSST ugrizy**; NIR documented) → SVO fallback → unresolved
  (warn). `source` ∈ `{lsst, documented, svo, manual, unresolved}`.
- **`resolve_bands(bands, *, lookup=None, svo_fallback=True, warn=True)`** → `(lambda_eff, zero_point,
  info)` arrays (NaN where unresolved); resolves each distinct band once.
- **`LSST_BAND_INFO`** — effective wavelength + zero point per effective band.

**SVO** (`io.svo`, needs `pyphot` or `astroquery`; pyphot is used first when installed, astroquery is
the fallback and is needed for the wavelength search): `resolve_band_svo(band, *,
lambda_eff_hint=None)`, `get_filter_metadata(filter_id, *, use_cache=True)` (cached by ID; offline-safe
disk cache under `$WHISPER_SVO_CACHE` — `use_cache=False` forces a fresh fetch, e.g. after SVO revises a
filter), `get_transmission_data(filter_id, *, use_cache=True)` (the curve a model bound to an SVO ID
integrates; cached in `svo_curves/` beside the metadata), `transmission_backend()`,
`find_filter_id(band, *, lambda_eff_hint=None, tol_frac=0.05)`,
`register_manual_band(band, lambda_eff, zero_point)`, `clear_cache(disk=False)`. `find_filter_id`
resolves in priority order **`DEFAULT_SVO_IDS`** (the documented band → SVO-ID table) → wavelength
search within `±tol_frac` of `lambda_eff_hint`; widen `tol_frac` for more candidates (and more
ambiguity), narrow it for fewer false matches. Several candidates ⇒ warn, list them, return the
closest in wavelength. Network failure /
unknown filter raises `SvoUnavailable`, which `resolve_band` turns into a warning + the manual-override
path. Corrupt/unusable cache entries are ignored, never crash a load. All network access goes through
`_svo_fetch_metadata` / `_svo_fetch_index` / `_svo_fetch_transmission` (the points the tests mock).

## 4. Photometry & units  (`io.photometry`, `io.units`)
`mag_to_flux_density`, `flux_density_to_mag`, `mag_err_to_snr`; `AB_ZEROPOINT_JY=3631.0`,
`POGSON=2.5/ln10`.

`io.units` stores **one canonical unit per mode**: `flux_density` → Jy, `flux` → erg/s/cm², `magnitude`
→ dimensionless AB (the table is `DEFAULT_UNITS`, keyed by `data_mode`; `VALID_DATA_MODES` lists the
modes, and the two astropy unit objects themselves are `CANON_FD` = Jy and `CANON_F` = erg/s/cm²). `to_canonical(values, unit, data_mode, *, lambda_eff=None, warn_default=True)`
validates + converts and is the entry point the loader uses; it dispatches to the three below.

| Function | Purpose |
|---|---|
| `as_unit(unit)` | Coerce `str` \| astropy unit \| `None` → `UnitBase`. `None`/`""` mean **dimensionless**; an unparseable string raises `ValueError`. Use it when accepting a unit from user input or a file header. |
| `to_flux_density_jy(values, unit, lambda_eff=None)` | → Jy. Accepts F_ν directly and F_λ via `u.spectral_density(λ_eff)` — the F_λ path **requires** a per-point wavelength and errors clearly, naming the offending points, when it is missing or NaN. |
| `to_flux_cgs(values, unit)` | → erg s⁻¹ cm⁻² for `data_mode='flux'` (band-**integrated** energy flux). Raises if the unit's physical type is not an energy flux — i.e. it rejects a flux *density* passed to the integrated mode. |
| `check_magnitude_unit(unit)` | Rejects a flux unit on a magnitude column (must be dimensionless AB). |

A no-unit column warns and applies the `DEFAULT_UNITS` entry for its mode.

## 4b. Synthetic photometry  (`whisper_cbpf.synphot`)
How every model turns an SED into a band flux, the same on CPU and GPU: `m_b = -2.5 log10[Σ w_k
F_ν(λ_k) / (3631 Jy Σ w_k)]` with the nodes and weights of one `FilterSet`. The full account, with
the measurements, is [`docs/PHOTOMETRY.md`](PHOTOMETRY.md).

| Name | Signature | Description |
|---|---|---|
| `resolve_filter` | `(label, aliases=None, default_system=None, known=())` | A data label → a filter name (sncosmo name, shipped filter or SVO ID; a name in `known` — the filters of a FilterSet you pass a model — as is). Bare `u g r i z y` are **LSST** (one warning per session); a grouped label (`g-band`) raises. Also `wp.resolve_filter`. |
| `set_default_band_system` / `default_band_system` | `(system)` / `()` | The system bare letters are read in for the session: `lsst` (default), `sdss`, `ztf`, `ps1`, `des`. Returns the previous one; mirrored in `$WHISPER_BAND_SYSTEM` so spawned sampler workers agree. Also `wp.set_default_band_system`. |
| `FilterSet` | `(names, nodes, weights, *, rule, provenance=None)` | Per-band nodes [Å] and weights. `.norms`, `.hash`, `.select(names)`, `FilterSet.concat(sets)`, `.save(path)` / `FilterSet.load(path)` (npz, hash-checked), `.to_legacy()` (the JAX factories' `filter_set=` dict), `FilterSet.from_legacy(dict)`. Plain numpy: pickles by value. |
| `gauss_rule` | `(names, n_nodes=16, *, curves=None)` | The band-adapted Gauss rule on `T dλ/λ`; curves from sncosmo, SVO IDs, or `curves={name: (wave, trans)}`. Warns if its build-time self-check on 1500–50 000 K blackbodies exceeds 0.02 mmag. |
| `filter_set_for` | `(names, n_nodes=None)` | The default FilterSet: Gauss-16, from the shipped LSST ugrizy / ZTF gri / SDSS ugriz files where possible (no sncosmo needed), else built. Memoised. |
| `band_flux_jy` | `(flux_jy, times, names, filter_set)` | CPU band flux: `flux_jy(t_flat, nu_flat_hz)` called once on every observation's nodes; an observation with any non-finite node returns 0. |
| `grid_rule.make_filter_set`, `grid_rule.ab_weights` | as in `models.jax.kilonova` | The shared-grid rule of whisper ≤ 0.1.0, moved verbatim (re-exported by `kilonova`/`tde`/`supernova`); a factory's `n_wave=` selects it. |

The CPU redback adapter takes the same knobs: `redback_model(model, band_names=None, *, ...,
photometry="band", filter_set=None, default_system=None, band_aliases=None)` (and
`register_redback`; the rest of its signature is in [§6.2b](#62b-the-redback-adapter--whisper_cbpfmodelsredback_adapter)).
`photometry="monochromatic"` is whisper ≤ 0.1.0's reference-frequency path.
For redback's double-dilation TDEs (`redback_double_dilation(model)`) it calls redback at `t (1+z)`,
which undoes that double time dilation.

## 5. Plotting

Every plot function returns its matplotlib `Axes` (the figure is `np.ravel(axes)[0].figure`) and leaves
the figure open in pyplot, so a bare call shows it exactly once in a notebook, and `plt.show()`,
`plt.savefig` and `save=` work as usual.

**`plot_light_curve(lc, *, layout="report", quantity="apparent_mag", bands=None, ncols=3, figsize=None, title=None, save=None)`**
— `layout`: `"report"` (mag + flux panels) or `"grid"` (per band). `quantity`: `apparent_mag` /
`absolute_mag` (needs redshift) / `flux`. Markers: detections = circles, SNR<3 = △, upper limits = ▽.
Returns the `Axes` array: the 2 panels for `"report"`, the `nrows x ncols` grid for `"grid"` (unused
cells hidden).

**`plot_ppc(results, lc, model=None, *, quantity="apparent_mag", panel_by="auto", n_draws=200, bands=None, tmin=None, tmax=None, ncols=None, colors=None, figsize=None, title=None, seed=0, save=None, intervals=(68, 95))`**
— **posterior-predictive check as a grid.** For each fit, draws `n_draws` posterior samples, evaluates
the model on a smooth time grid, and shades each central predictive interval in `intervals` (by
default the **68 % and 95 % bands**, darker for the narrower; `intervals=(95,)` gives the 0.1.1
figure) + median over the photometry, per band. A model that is not registered needs `model=`. `results` is one `SamplerResult`, a `{label: SamplerResult}` dict, or a
list. `panel_by`: `"method"` (one panel per fit, bands overlaid — the multi-sampler grid),
`"band"` (one panel per band, fits overlaid), or `"auto"` (method when several fits, else band).
`quantity`: `"apparent_mag"` (inverted axis) or `"flux"` (flux density [Jy]). Returns the 2-D
`nrows x ncols` `Axes` array.

**`plot_calibration(results, lc, model=None, *, levels=(0.5, 0.68, 0.8, 0.9, 0.95, 0.99), space="auto", per_band=False, n_draws=400, colors=None, figsize=None, title=None, seed=0, save=None)`**
— **coverage-calibration curve** (reliability / pp-plot): empirical posterior-predictive coverage vs the
nominal credible level, for one or more fits (or one fit broken out `per_band=True`). On the diagonal =
calibrated; below = over-confident, above = under-confident. Coverage from `predictive_metrics`.
Returns the `Axes`.

**`plot_corner(posteriors, *, labels=None, parameters=None, colors=None, truths=None, bins=30,
levels=(0.39, 0.86), smooth=1.0, log_params="auto", title=None, legend_loc="upper right", save=None, **corner_kwargs)`**
— overlay a **list of posteriors** on one publication-ready corner plot.

> **`posteriors` is a sequence, even for one fit.** Pass `[res]`, not `res` — a bare `SamplerResult`
> raises `TypeError: object of type 'SamplerResult' has no len()`, and a bare `DataFrame` iterates its
> column *names* and raises `ValueError: could not convert string to float`. Neither message points at
> the fix, so:
>
> ```python
> wp.plot_corner([res])                                        # one posterior
> wp.plot_corner([r_abc, r_mcmc], labels=["abc", "mcmc"])      # the comparison this is built for
> ```
 Each posterior is a
`SamplerResult`, a `DataFrame`, a `{name: array}` dict, or a 2-D array (then pass `parameters`).
Shared per-parameter ranges align the panels; each posterior gets a distinct dark colour
(`CORNER_PALETTE`); 2-D panels are contour **lines** (default `levels` ≈ 1σ/2σ) and the diagonals are
step histograms, so several posteriors stay readable overlaid. `parameters` defaults to the columns
common to all inputs; `log_params` puts those on a `log10` axis (the default `"auto"` takes every
parameter with a LogUniform prior in the `SamplerResult`s passed, labelled `log10 <name>`; `None`
or `[]` turns it off); `truths` (dict or list) draws dashed
reference lines; a colour→label legend is added. Returns the `K x K` array of corner `Axes` (row-major,
as `corner` lays them out). Ideal for comparing samplers on
the same data — the posteriors (with uncertainties) show whether methods are *compatible*, which a table
of point estimates cannot.

The plots for comparisons and forecasts (`plot_models`, `plot_model_comparison`, `plot_forecast`,
`plot_widths`) are in [A8](#a8-plot).

---

## 6. Inference: priors, models, distance, samplers

### 6.1 Priors  (`whisper_cbpf.priors`)
Small, **picklable** distributions (so they cross process boundaries in parallel ABC).

| Class | Constructor | Methods |
|---|---|---|
| `Uniform` | `(low, high, name=None)` | `sample(rng)`, `log_prob(x)`, `rescale(u)`, `bounds` |
| `LogUniform` | `(low, high, name=None)` (`low>0`) | same |
| `Normal` | `(mu, sigma, name=None)` | same, plus vectorised `cdf`, `ppf`, `std`; `bounds` is `(-inf, inf)` |
| `TruncatedNormal` | `(mu, sigma, low, high, name=None)` (`low` or `high` may be infinite) | same as `Normal`; exact density with its truncation normalisation |
| `Fixed` | `(value, name=None)` | `sample`, `log_prob`, `rescale`, `std`, `bounds`: a parameter held at `value` |
| `Prior` | `(distributions: dict)` | `sample(rng=None) -> dict`, `log_prob(params)`, `rescale(unit_cube)`, `names`, `bounds`, `fixed` (`{name: value}` of the `Fixed` entries) |

Which sampler takes which family, and how a `Fixed` parameter is counted:
[A2](#priors-beyond-boxes-normal-truncatednormal-fixed).

```python
prior = wp.Prior({"amplitude": wp.Uniform(0, 10), "rise_time": wp.Uniform(1, 10)})
prior.sample(np.random.default_rng(0))   # {'amplitude': ..., 'rise_time': ...}
```

### 6.2 Models  (`whisper_cbpf.models`)
A model maps parameters to predicted flux: `predict(params: dict, times: np.ndarray, bands) -> np.ndarray`.

| Function | Signature | Description |
|---|---|---|
| `register_model` | `(name, predict, parameters, prior=None, description="", *, overwrite=False, predict_jax=None, log_prob_jax=None, param_aliases=None)` | Register a model by name. The two `*_jax` slots and `param_aliases` are optional; a factory that built them must pass them through or the registered model is the CPU half only. |
| `get_model` | `(model)` | Resolve a name (or pass a `Model`). |
| `list_models` | `()` | Sorted registered model names. |
| `Model` | dataclass: `name, predict, parameters, default_prior, description, predict_jax, log_prob_jax, param_aliases` | Callable: `model(params, times, bands)`. `predict_jax(theta, times, band_idx) -> flux` is `predict` with a flat theta and integer bands, traceable and differentiable; `log_prob_jax(theta) -> scalar` is a data-bound log-*likelihood*. Both `None` unless a factory filled them — see [§10](#10-the-jaxgpu-half). `param_aliases` (`{}` unless set) maps parameter names onto the redback-named twin's ([§6.2b](#62b-the-redback-adapter--whisper_cbpfmodelsredback_adapter)). |

Built-in **toy** models (band-independent, vectorized; `t` = days since explosion; scale `amplitude` to
your flux units for real data):

| Name | Form | Parameters |
|---|---|---|
| `flare` | `A·(1 − e^(−t/t_rise))·e^(−t/t_decay)` | amplitude, rise_time, decay_time |
| `bazin` | `A·e^(−(t−t0)/τ_fall) / (1 + e^(−(t−t0)/τ_rise))` | amplitude, t0, tau_rise, tau_fall |
| `gaussian_rise` | Gaussian rise to peak at `t0`, then exp decay | amplitude, t0, sigma_rise, tau_decay |

Built-in **physical** models:

| Name | Physics | Parameters |
|---|---|---|
| `mck19` | EM flare from a **BBH merger in an AGN disk** — a GW-recoil-kicked remnant shocks a bound-gas **hotspot** that radiates as a blackbody: a `sin²` rise to the ram-pressure delay `t_ram`, then exponential decay back to the disk baseline. McKernan et al. 2019 ([ApJL 884, L50](https://iopscience.iop.org/article/10.3847/2041-8213/ab4886)), implementation of [Darc 2025](https://arxiv.org/abs/2506.02224). | v_kick, M_smbh, M_bh, r_bh, redshift |
| `two_component_kilonova` | **Kilonova** (NS–NS merger) with a blue (low-κ) + red (high-κ) ejecta component, via the optional **redback** backend (`[models]` extra). The sum of two one-component redback flux densities, band-integrated (§4b) → flux density (Jy). | mej_1, vej_1, kappa_1, temperature_floor_1, mej_2, vej_2, kappa_2, temperature_floor_2, redshift |

`two_component_kilonova` is the **first redback-backed model** — redback is imported lazily, so WHISPER
and `list_models()` work without it; only `predict` needs the `[models]` extra. It is redback's
`two_component_kilonova_model` computed as **two `one_component_kilonova_model` calls summed in flux
density** and integrated over each band's filter (bare letters are LSST; `B`, `J`, `uvot::uvw1` …
resolve through redback's filter table). Since whisper 0.1.1: redback's two-component model stops at 6
days source frame, where 0.1.0 returned 99 mag; the sum is finite at every epoch,
agrees with redback's two-component model to 1.3–2.0 mmag inside 6 d (its grid differs), and costs
~9 ms per 200-observation predict against 177 ms for the old per-band magnitude calls. See
[`dev/demo_kilonova.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_kilonova.py)
and [`dev/fit_kilonova_at2017gfo.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/fit_kilonova_at2017gfo.py)
(pinned history: WHISPER_AI @ 8ba3843, discontinued; superseded by whisper_cbpf).

`mck19` is **band-dependent** — it returns flux density (Jy) at each `(time, band)`, integrating the
hotspot and disk blackbodies over each band's filter (§4b) at the redshift (Planck18 luminosity
distance + time dilation). `t = 0` is the merger; the flare peaks at the observer-frame delay `t_ram`.
Self-contained (astropy constants/cosmology only — no `speclite`, no sncosmo for the shipped LSST,
ZTF and SDSS filters). A band that names no filter raises; up to 0.1.0 every band was taken at its
effective wavelength and an unresolved one silently at 6000 Å (the g peak at the test point moves
25.900 → 25.842). Fixed
disk constants (`mdot=0.05` Edd, `alpha=0.1`) do not enter the light-curve *shape*. See
[`dev/demo_mck19.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_mck19.py)
(pinned history: WHISPER_AI @ 8ba3843, discontinued) for the g/r/i light curve.

**Any redback model, on the CPU**: `register_redback("arnett", redshift=0.0098)` binds any of redback's 347 models
through one adapter — see [§6.2b](#62b-the-redback-adapter--whisper_cbpfmodelsredback_adapter).

> For **parallel** ABC (`n_jobs>1`) the `predict` function must be picklable (module-level, not a
> closure/lambda). Closures work with `n_jobs=1`.

**Built-in model module contract.** Every module in `whisper_cbpf/models/` exposes the four names that
`models/__init__.py` feeds to `register_model`, and a custom model file is expected to follow suit:

| Name | Type | Role |
|---|---|---|
| `PARAMETERS` | `list[str]` | Ordered parameter names — becomes `Model.parameters` and fixes the column order of every posterior. |
| `PRIOR` | `Prior` | The model's `default_prior`, used when a sampler is called without `prior=`. |
| `DESCRIPTION` | `str` | One-line description carried on `Model.description`. |
| `<name>_flux` | callable | The `predict(params, times, bands) -> flux (Jy)` implementation, e.g. `flare_flux`, `bazin_flux`, `gaussian_rise_flux`, `mck19_flux`, `two_component_kilonova_flux`. |

`two_component_kilonova` additionally exposes `REDBACK_MODEL` (the redback function it wraps). Import
these directly (`from whisper_cbpf.models import flare; flare.PRIOR`) to reuse or widen a built-in
prior instead of retyping its bounds.

### 6.2b The redback adapter  (`whisper_cbpf.models.redback_adapter`)
Any key of `redback.model_library.all_models_dict` becomes a whisper `Model` on the CPU, with its
parameters and prior read from redback itself (the `[models]` extra; redback is imported lazily, so
this module loads without it). `register_redback` is also `wp.register_redback`.

```python
import whisper_cbpf as wp
lc.set_explosion_date(60800.2)                       # the adapter's clock: days since t = 0
wp.register_redback("arnett", ["ztfg", "ztfr"], redshift=0.05, times=lc.time)
res = wp.fit(lc, "arnett_redback", sampler="mcmc")
```

| Function | Signature | Description |
|---|---|---|
| `redback_model` / `register_redback` | `(model, band_names=None, *, redshift=None, name=None, prior=None, pin=None, min_time_day=1e-3, redback_kwargs=None, description=None, photometry="band", filter_set=None, default_system=None, band_aliases=None, constraint="corrected", times=None)` (+ `overwrite=False` to register) | Build (or build and register, as `f"{model}_redback"` unless `name=`) the model. Arguments below. A `*_bolometric` engine (erg/s, no `redshift`) raises, naming its photometric wrapper, and so does a model whose `flux_density` is the same at two frequencies (a luminosity engine or a unitless curve). |
| `redback_flux_jy` | `(model, parameters, times, bands=None, *, min_time_day=1e-3, redback_kwargs=None, photometry="band", filter_set=None, default_system=None, band_aliases=None)` | The physics at one parameter set, in Jy: no constraint wall, no domain probe. Epochs redback refuses (any exception) or answers with NaN/inf are 0, with a warning once per model and span. |
| `redback_parameters` / `redback_prior` / `redback_pinned` | `(model)` | Every parameter redback's model takes (signature plus the `.prior` file's `**kwargs` entries), in redback's order; its prior as a whisper `Prior`; the `DeltaFunction` entries as `{name: value}`. |
| `redback_luminosity_distance_cm` | `(redshift, model="arnett")` | The distance redback uses for `model` (its module's own cosmology): pass it as `dl_cm` to a JAX twin. |
| `redback_applies_dilation` / `redback_double_dilation` | `(model)` | Whether the installed redback multiplies `flux_density` by `(1+z)`; whether it dilates time twice (`gaussianrise_cooling_envelope`, `bpl_cooling_envelope`, `stream_stream_tde` in 1.20). For the latter the adapter calls redback at `t (1+z)`, and the model description says so. Both read redback's source. |
| `installed_redback_preset` | `()` | `"1.20"`, `"1.15"`, `"1.12"` or `None`: the preset key of the JAX ports (`tde.REDBACK_ENGINE_PRESETS`, `supernova.REDBACK_GRID_PRESETS`) that reproduces the **installed** redback, read from its source without importing it. The TDE's default `n_time` follows it. |
| `LATEST_REDBACK_PRESET` | `"1.20"` | The newest release the presets reproduce; the defaults when no redback is installed. |
| `redback_package_dir` | `()` | redback's package directory, found without importing it (a redback clone on `sys.path` does not fool it); `None` if absent. |

Constants: `MJY_TO_JY = 1e-3` (redback's `flux_density` is mJy), `MIN_TIME_DAY = 1e-3` (an epoch
before the explosion, `t < 0`, predicts zero flux; epochs from 0 up to `MIN_TIME_DAY` are clipped up
to it, since redback silently mis-evaluates `t <= 0`), `PHOTOMETRY_MODES = ("band",
"monochromatic")`, `DOMAIN_PROBE_DRAWS = 8`.

**Arguments of `redback_model` / `register_redback`:**

| Argument | Meaning |
|---|---|
| `band_names` | Labels to resolve now, so a typo fails here and not inside a likelihood. Does not restrict what `predict` accepts. |
| `redshift` | Sugar for `pin={"redshift": z}`, with the same check: it must be a parameter of the model. |
| `prior` / `pin` | Per-parameter prior overrides (a named parameter becomes free, even if redback pins it) / fixed values (they leave `Model.parameters`). |
| `photometry` | `"band"` (default since 0.1.1): redback's SED integrated over each observation's filter, one redback call on every Gauss node (§4b, [`PHOTOMETRY.md`](PHOTOMETRY.md)). `"monochromatic"`: whisper ≤ 0.1.0's single reference frequency per band, 4–40 mmag median and up to 0.65 mag off the band integral; for physics-parity tests. |
| `filter_set` | The `FilterSet` to integrate with (default: Gauss-16 for the filters the labels resolve to). Pass the one a JAX model was built with and the two compute the same integral. |
| `default_system` | The system bare `u g r i z y` are read in, for this model only; default the session's (`wp.set_default_band_system`, LSST). |
| `band_aliases` | Your labels → filter names, applied first: `{"g": "sdssg"}`, `{"g-band": "ztfg"}`. |
| `constraint` | redback's `Constraint` priors (Arnett nuclear burning ≥ kinetic energy, magnetar rotational ≥ kinetic energy, cooling-envelope `eta`/`beta` bounds, …) as a hard wall: a draw that breaks one predicts **zero flux** and redback is not called. `"corrected"` (default) bounds the Arnett kinetic energy by 1.51e18 erg/g of nickel; `"redback"` keeps redback's 1.91e19; `None` applies none (whisper ≤ 0.1.0). Formulas and modes: `whisper_cbpf.models.constraints` (`constraint_ok(model, params, mode="corrected", xp=np)`, `E_BURN_PER_G`, `E_BURN_PER_G_REDBACK`, `MODELS`). |
| `times` | The epochs you will fit, on the model's clock. The domain probe then runs at registration (needs `band_names`): epochs redback refuses at every one of `DOMAIN_PROBE_DRAWS` prior draws, with the same message — a limit of the model, like `two_component_kilonova_model`'s 6-day grid — raise a `ValueError` naming the span. Without `times` the probe runs at the first `predict` that loses epochs. Epochs lost at some parameters only are zero for that draw (a rejection) and warn once per span. |
| `min_time_day`, `redback_kwargs`, `description` | Epoch clip; extra fixed keywords for redback on every call (`cosmology=`, `base_model=`, …); a description override. |

**Time.** `predict`'s `times` are observer-frame **days since the model's own t = 0** — the explosion
for a supernova, the merger for a kilonova, the fallback for `cooling_envelope` — not MJD. redback's
photometric models have no explosion-time parameter, so `pin={"t0": mjd}` (or `t_exp`,
`explosion_time`, …) raises, and the error says this. Put day 0 on the light curve instead:
`lc.set_explosion_date(mjd)`.

**Bands.** Labels go through `wp.resolve_filter` (§4b): bare letters are LSST unless
`wp.set_default_band_system(...)`, `default_system=` or `band_aliases=` say otherwise; a grouped label
(`g-band`) raises; any sncosmo name or SVO ID works. The integral is `synphot.band_flux_jy` over a
`FilterSet` (`synphot.gauss_rule`, `synphot.filter_set_for`) — the same object the JAX factories take
as `filter_set=`.

**The same wall on the JAX models.** `supernova_model(..., constraint="corrected")` (for `arnett`,
`basic_magnetar_powered`, `slsn`, `general_magnetar_slsn`) and `tde_model(..., constraint=...)` (both
TDEs) take the same three modes: the host `predict` — what the CPU samplers call — gives zero flux
for a draw that breaks one, as this adapter does, and the JAX samplers apply
`predict_jax.constraint_ok` ([§10](#10-the-jaxgpu-half)). Pass `constraint=None` to compare the
physics at any draw. The zeros are silent — the samplers meet such draws on 33-97 % of redback's
prior — so a direct `predict` at a rejected draw (a plot at a prior's midpoint, say) returns all
zeros without a warning; test the draw with `model.predict_jax.constraint_ok(theta)` first.

**Parameter names across backends.** `Model.param_aliases` maps a model's parameter names onto its
redback-named twin's where they differ: the JAX `kilonova_two_model`'s `mej_blue … kappa_red` →
`mej_1 … kappa_2`, the names of `two_component_kilonova` and of
`register_redback("two_component_kilonova_model")`. Rename a CPU or GPU sample table with it to pair
the two posteriors: `samples.rename(columns=model.param_aliases)`. `kilonova_two_model(prior=...)`
takes either spelling (or a mix), so `prior=kilonova_two.default_prior()`, redback's own box, works
as it is; the model's `default_prior` comes back in its own names.

**Warnings.** Importing redback runs `warnings.simplefilter("ignore")` for the whole process (redback
1.20 `result.py:17`) and sets `text.usetex`. The adapter imports it through `_import_redback`, which
restores your warning filters and matplotlib settings afterwards, so whisper's own warnings (out of
domain, SNPE fallback, …) still reach you after a redback model is bound.

### 6.3 Distance  (`whisper_cbpf.distance`)
`chi2_distance(obs_flux, obs_flux_err, sim_flux, bands=None) -> float` = `sum(((obs-sim)/err)**2)`.
Any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float` can be passed as a custom distance.

Distances are also a **name registry**, so a custom distance can be selected by string
(`fit_ABC(lc, "flare", distance="my_distance")`) and survives pickling into parallel ABC workers:

| Function | Signature | Description |
|---|---|---|
| `register_distance` | `(name, fn, *, overwrite=False)` | Register a distance by name. |
| `get_distance` | `(distance)` | Resolve a name (a callable passes straight through). |
| `list_distances` | `()` | Sorted registered names — **16** out of the box: `chi2`, `chi_square` (an alias), `mse`, `rmse`, `mae`, `wmse`, `wmae`, `max_abs_z`, and a `*_jax` twin of each. The `_jax` entries are listed even without the `[gpu]` extra. |

Names are matched **case-insensitively**. `max_abs_z_distance(obs, obs_err, sim, bands=None)`, the
worst point in units of its own error, is described in
[A2](#abc-the-worst-point-distance-and-single-precision).

### 6.4 Samplers  (`whisper_cbpf.samplers`)

`fit_ABC(lc, model="flare", ...)`, `fit_ABC_SMC(lc, model="flare", ...)`, `fit_MCMC(lc, model="flare",
...)`, `fit_nested(lc, model="flare", ...)` and `fit_SNPE(lc, model="flare", ...)` → `SamplerResult`.
`fit(lc, model, sampler="auto", *, likelihood_max_opt=False, **kwargs)` is the generic dispatcher
(`"auto"`: `emcee_jax` for a JAX model when JAX sees a GPU, CPU `mcmc` otherwise); what every fit
does before sampling (the pre-event rule, the explosion-time prior, upper limits) and
`likelihood_max_opt=` are in
[A2](#what-every-fit-does). `list_samplers()` →
`['abc', 'abc_gpu', 'abc_smc', 'abc_smc_gpu', 'dynesty', 'emcee_jax', 'mcmc', 'nested', 'npe',
'nuts_gpu', 'pymc_jax_gpu_parallelized', 'pymc_jax_gpu_vectorized', 'snpe', 'snpe_gpu']` — **14**, the
seven GPU entries included even without the `[gpu]` extra (`npe` is an alias of `snpe`, `dynesty` of
`nested`); `register_sampler(name, cls)` adds your own. **ABC/ABC-SMC** use the χ² distance; **MCMC/SNPE**
use the shared likelihood layer. All four target the same posterior, but at practical budgets only
MCMC samples it closely: in [`examples/compare_samplers.py`](../examples/compare_samplers.py)
(`gaussian_rise`, shipped settings) SNPE came out 3-18× wider than MCMC, and ABC / ABC-SMC 20-65× wider
with medians shifted by up to 1.5 of their own sd. ABC approaches it only as its acceptance tightens.

Registry helpers: `register_sampler(name, sampler_cls, *, overwrite=False)`,
**`get_sampler(name)`** → a *fresh instance* of the registered class (`KeyError` listing the available
names if unknown — this is what `fit` dispatches through, and the way to reach a sampler's own
`fit` signature by name), `list_samplers()`.

**`ABCSampler.fit(lc, model, prior=None, *, ...)`**:

| Argument | Default | Description |
|---|---|---|
| `n_simulations` | `10000` | Total prior draws / simulations. |
| `quantile` | `0.01` | Accept the best fraction by distance (robust default). |
| `threshold` | `None` | Fixed acceptance distance ε (overrides `quantile` if set). **Scale warning:** with `simulate_noise=True`, `E[D] ≈ χ² + n_points` — re-derive old noiseless thresholds. |
| `distance` | `chi2_distance` | Distance function. |
| `simulate_noise` | `True` | Add per-point `N(0, flux_err)` white noise to each simulation so it matches the data's generative model — makes ABC exact as ε→0 and its **width calibrated** (`False` = old noiseless shell behaviour). |
| `space` | `"auto"` | Comparison space (`'flux'`/`'magnitude'`): data, simulations, noise and distance all live here. |
| `scatter_param` | `None` | Prior parameter used as a free extra-scatter term in the simulation noise (see §6.5; the scatter *level* is not identifiable by a χ² distance — use MCMC/SBI for it). |
| `n_jobs` | `None` (→ min(cpu, 8)) | Processes for parallel simulation. |
| `seed` | `0` | RNG seed (independent streams per worker via `SeedSequence`). |

Sampled parameters (and the `samples` columns / `n_params` in AIC/BIC) are **`prior.names`** — a prior
may carry more than the model's own parameters (e.g. the scatter term).

`best_params` (and the AIC/BIC evaluated there) are selected by the **exact Gaussian log-likelihood**
over the accepted draws — never by the noisy distance, whose argmin is the luckiest noise draw.

**`ABCSMCSampler.fit(lc, model, prior=None, *, n_particles=500, n_rounds=5, epsilon_schedule=None,
quantile=0.5, min_epsilon=None, simulate_noise=True, space="auto", scatter_param=None,
perturbation_scale=0.1, distance=chi2_distance, n_jobs=None, seed=0,
max_attempts_per_round=None)`** — `space`/`scatter_param`
as in `ABCSampler.fit` (the scatter is perturbed and importance-weighted like every particle
dimension).
— sequential rejection: round 0 draws from the prior; later rounds resample + Gaussian-perturb accepted
particles under a shrinking epsilon (explicit `epsilon_schedule`, or adaptive `quantile` of the previous
round's distances). Perturbs only parameters and rejects proposals outside the prior; `info` carries
per-round epsilon / acceptance / `total_simulations`. **`min_epsilon`** floors the adaptive epsilon so it
is not driven to `χ²_min` (which collapses the posterior onto the MLE → overconfident): `"auto"` floors
it at **`χ²_min + 2(k+2)`** (`k` = #parameters), reproducing the Gaussian posterior width; a float sets a
fixed floor (default `None` = no floor). **`simulate_noise=True`** (default) adds per-point
`N(0, flux_err)` noise to every simulation — the smooth acceptance kernel that keeps the SMC posterior
width calibrated; note it shifts the distance scale (`E[D] ≈ χ² + n_points`), so old noiseless
`epsilon_schedule`/float-`min_epsilon` values must be re-derived (the adaptive quantile handles it).
**`max_attempts_per_round`** is the safety cap on proposals per round (default
`max(200·n_particles, 200_000)`) that stops an unreachable epsilon from looping forever: a round that
hits it **warns and continues with however many particles it accepted** — so an
`ABC-SMC round N: only k/n_particles accepted` warning means the round was truncated and its epsilon
is too tight for the noise floor, not that the fit failed.

**`MCMCSampler.fit(lc, model, prior=None, *, nwalkers=None, nsteps=5000, burnin=1000, thin=10,
init="prior_scan", initial_guess=None, initial_scatter=1e-3, space="auto", likelihood="auto", seed=0,
progress=False, moves=None, n_jobs=None, walker_coordinates="own")`** — `n_jobs` runs the walkers' likelihood evaluations in a process pool
(worth it only for expensive simulators, e.g. the ~0.1 s kilonova model); with
`likelihood="gaussian_scatter"` a prior parameter named `sigma` is routed to the likelihood as the
free Villar+17 extra-scatter term (see §6.5). — affine-invariant ensemble MCMC via `emcee` (`samplers.mcmc`; emcee is a **core**
dependency). The log-posterior is the Whisper prior + the **shared likelihood layer**
(`make_likelihood(lc, kind=likelihood, space=space)`), so MCMC uses the same physically consistent,
`data_mode`-aware likelihood as the others (flux data → flux space, magnitude data → magnitude space).
`nwalkers` defaults to `max(2·ndim+2, 4·ndim)` (forced even); walkers start from `init=`
(default `"prior_scan"`: the best prior draws climbed without a gradient, [A2](#where-the-chains-start-init);
`init="prior"` gives the pre-0.2.0 independent prior draws), or from `initial_guess` when it is
given. Sampling is **seeded/reproducible**. `best_params` is the max-likelihood draw
(each draw's log-posterior minus its log-prior, so a LogUniform prior does not bias AIC/BIC);
`max_log_likelihood`/`AIC`/`BIC` are exact Gaussian values; `info` carries `mean_acceptance_fraction`,
`mean_autocorr_time` and **`max_autocorr_time`** (the largest over parameters), **`stuck_walkers`**
(walkers whose median log-posterior after burn-in is more than 10 nats below the best walker's: parked
in a separate optimum, their draws pooled into the posterior) with `walker_median_log_prob`, and
**`convergence_problems`**, one sentence per failed check. **`converged`** is `True` only when that list
is empty: no stuck walker and `nsteps ≥ 50 ×` the *largest* autocorrelation time (an unknown time is not
a pass); a non-empty list also warns. It used to be the autocorrelation test alone, on the mean time,
which read `True` on 55 of 78 Gaussian-bump fits with stuck walkers (measured on `emcee_jax`, which
used the same rule). The `n_jobs` pool starts its workers
with `spawn` (never `fork` after JAX has started), which costs process start-up: worth it only when one
likelihood call is expensive. The `emcee.EnsembleSampler` is attached as `result.emcee_sampler`.
`fit_MCMC(...)` is the convenience wrapper.

**`SNPESampler.fit(lc, model, prior=None, *, num_rounds=2, num_simulations=1000, space="auto",
density_estimator="maf", embedding_net=None, embedding_latent=32, x_format="value", predict_torch=None,
scatter_param=None, hidden_features=None, num_transforms=None, num_bins=None,
proposal_mode="posterior", truncate_quantile=1e-4, support_samples=10000, num_samples=10000,
device="cpu", seed=0, show_progress=False, num_workers=1, max_logl_scan=2000, scan_timeout=300,
standardize_x=True, proposal_min_acceptance=1e-3, num_chains=4, **train_kwargs)`** —
`scatter_param` names a prior parameter used as the free Villar+17 extra-scatter term: it enters the
simulation noise as `N(0, √(σᵢ²+σ²))` per draw, so the density estimator learns its posterior from
the noise imprint (§6.5).
Sequential Neural Posterior Estimation via `sbi` (`samplers.snpe`). Needs the optional **`[sbi]`** extra
(sbi + torch; imported lazily). The simulator is Whisper's forward model + **per-point Gaussian noise
from the data errors**; the prior is adapted automatically (`Uniform`→`BoxUniform`, mixed
`Uniform`/`LogUniform`→`MultipleIndependent`, both built **on the training device**). A `LogUniform`
component is `exp(Uniform(log a, log b))` carrying an explicit `interval(a, b)` **support**, not just
the matching density: sbi's `within_support` — the accept/reject test inside `DirectPosterior.sample()`
— consults `support.check(theta)` rather than `log_prob`, so the advertised support is what actually
keeps out-of-prior draws out of the returned posterior. `num_rounds=1` is amortized NPE, `>1` is sequential;
`num_simulations` is per round; `space` ('auto'|'flux'|'magnitude') matches the likelihood;
`num_workers` parallelizes simulation.

- **Input layout:** `x_format="value"` conditions on the data-space values alone; `"stacked"` appends
  per-point error + time channels (the same information the likelihood-based samplers receive, so an
  embedding net can exploit cadence/noise structure). The **band is not a channel**: every simulation is
  drawn on the identical `(time, band)` grid as the observation, so band identity is already encoded by
  position — a constant per-position channel would carry no information (and empirically hurt flux-space
  fits; removing it improved recovery of the red kilonova opacity in the AT2017GFO application).
- **Input normalisation:** `standardize_x` (default `True` → `"asinh"`; also `"zscore"` or
  `"none"`/`False`) rescales the conditioning input before it reaches the estimator, with per-channel
  statistics fitted on the first round's simulations and applied identically to the observation and
  `result.format_x`. `"asinh"` (`asinh(x/scale)`) variance-stabilises wide dynamic ranges — essential
  for flux-space data spanning several orders of magnitude, where sbi's built-in z-scoring fixes scale
  but not skew.
- **Density estimator + embeddings:** `density_estimator` is an estimator name **or** a pre-built
  `posterior_nn(...)` factory. **`embedding_net`** is `None`, a built-in name — **`"mlp"`** or
  **`"tcn"`** (Temporal Convolutional Network: dilated causal convolutions for time series; see
  `whisper_cbpf.embeddings`), compressed to `embedding_latent` features and trained jointly — or any
  `torch.nn.Module`; `hidden_features` / `num_transforms` / `num_bins` build a custom architecture.
  The built-ins come from **`build_embedding(spec, n_points, n_channels=1, latent_dim=32)`**
  (`whisper_cbpf.embeddings`), which returns `MLPEmbedding(n_points*n_channels, latent_dim=...)` or
  `TCNEmbedding(n_points, n_channels=..., latent_dim=...)` and raises `ValueError` for any other name.
  `fit` calls it for you with `n_channels` set by `x_format` (1 for `"value"`, 3 for `"stacked"`) and
  `latent_dim=embedding_latent`; call it directly only to build/inspect a net outside a fit.
- **GPU simulation:** `predict_torch(theta, times) -> flux`, or `predict_torch(theta, times, bands)`
  for a photometric model (the three-argument form is detected from the signature; batched torch
  model: `(B, D)` params + `(n,)` times → `(B, n)` flux) replaces the per-row Python simulator with
  one on-device batched call (~10³× faster simulation). The flux is mapped into the comparison space
  before the noise is added, so magnitude data are supported. `sampler="snpe_gpu"` builds this
  callable from any model's `predict_jax`. `result.format_x(values)` maps a raw vector to the
  network's conditioning input (same observing grid) for amortized reuse.
- **Sequential scheme:** `proposal_mode='posterior'` (SNPE-C, default) or `'restricted'` (truncated SNPE
  via `RestrictedPrior` + `get_density_thresholder(quantile=truncate_quantile)`; support estimated from
  `support_samples` draws — kept modest, since sbi's default 1e6 can take hours; rejection sampling makes
  it compute-heavy).
- **Robust best-fit scan:** the post-fit max-likelihood scan (re-running the forward model on posterior
  draws) runs in a process pool (`num_workers`) with a wall-clock cap `scan_timeout` [s], so an expensive
  or occasionally-pathological model call cannot hang the fit — it degrades to scoring a smaller subset
  instead.
- **Robust final sampling:** conditioning a trained estimator on the *real* observation (as opposed to
  simulated placeholders seen during training) can occasionally expose numerical pathologies sbi's
  default rejection sampling doesn't handle gracefully — a near-zero acceptance rate (impractically slow
  rather than an error) or a degenerate flow transform (`AssertionError` deep in `nflows`). Both are
  detected (a bounded, hang-proof acceptance probe; an `AssertionError` catch) and fall back to
  MCMC-based posterior sampling (`sample_with="mcmc"`, which conditions via `log_prob` instead of the
  flow's inverse). `result.info["final_sample_method"]` records which path was used
  (`"rejection"` or `"mcmc_fallback"`) and `["final_sample_acceptance_rate"]` the probed rate.
  `proposal_min_acceptance` (default `1e-3`) is the acceptance below which the final draw, and every
  round-to-round proposal draw, falls back; the fallback runs `num_chains` (default 4, recorded in
  `info["num_chains"]`) slice-sampling chains seeded from `seed`, so a fit that falls back is
  reproducible and its chain count no longer depends on `num_workers`.
- **Is this posterior usable?** `info["leakage"]` is `True` when the final flow put so little mass
  inside the prior box that the final draw fell back (acceptance below `proposal_min_acceptance`), and
  `info["converged"]` is `True` only when the final draw needed no fallback. The flag separates the two
  populations of 45 test fits with an emcee reference: the 38 that fell back sat a median 1.61 emcee-σ from an emcee
  reference (median worst parameter 19.8σ), the 7 that did not 1.00σ (2.31σ). `info["x_o_min_rms_z"]`
  is the RMS distance, in data errors, from the observation to the closest round-0 (prior-predictive)
  simulation: large means the prior cannot produce the data before any training is spent.
  `runtime_s` covers simulation and training; `info["postprocess_s"]` is the final draw plus the
  max-likelihood scan, which the fallback can make several times longer.

- **Device (GPU):** `device` = `'cpu'` (default), `'cuda'`/`'gpu'`/`'cuda:N'`, or `'auto'` (CUDA when
  available, else CPU). The torch prior + observed data are placed on the device; a GPU request without
  one warns and falls back to CPU. The GPU accelerates *training* (not the CPU simulator), so it helps
  most with many simulations / large nets — see
  [`sanity_check/benchmark_snpe_device.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/benchmark_snpe_device.py)
  and [its figure](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/sanity_check/figures/snpe_device_benchmark.png)
  (pinned history: WHISPER_AI @ 8ba3843, discontinued).

Extra kwargs pass to `NPE.train` (e.g. `max_num_epochs`, `training_batch_size`, `stop_after_epochs`).
`max_log_likelihood`/`AIC`/`BIC` are the exact Gaussian values at the best posterior draw. The trained
sbi posterior is attached as `result.posterior` (and `result.posteriors` per round) for resampling /
`sbi.analysis.pairplot`. `fit_SNPE(...)` is the convenience wrapper; `"snpe"` and `"npe"` both dispatch here.

**`SamplerResult`** fields: `sampler`, `model`, `parameters`, `samples` (DataFrame of accepted draws
+ `distance`), `summary` (median/ci16/ci84/mean/std per param), `best_params`, `n_data`, `n_params`,
`runtime_s`, `info` (n_simulations, n_accepted, acceptance_rate, epsilon, quantile, n_jobs, plus
**`band_metrics`** — per-band MSE/RMSE/MAE at the best fit — and **`predictive_metrics`** — RMSE / LPD /
ELPD-LOO / WAIC / AIC / BIC + the coverage-calibration curve; see §6.6), `min_distance`,
`max_log_likelihood`, `aic`, `bic`, and `provenance` (how the fit was made: versions, model and prior,
sampler settings and seed, devices, timing; [A6](#a6-save-reload-and-resume)). Methods: `n_samples`,
`to_dict()`, `to_json(path=None, indent=2)` —
every field above (`band_metrics` and `predictive_metrics` included) is serialised to the JSON, written
to `path` when given and always returned as a string; `indent=None` emits compact one-line JSON for
machine-read outputs.

| method | signature | see |
|---|---|---|
| `save` | `(path, *, overwrite=False)` → `Path` | [A6](#a6-save-reload-and-resume) |
| `diagnostics` | `(*, prior=None, likelihood_max_opt=None)` → `DiagnosticsReport` | [A4](#a4-check-a-fit) |
| `likelihood_max_opt` | `(lc, model=None, **kwargs)` → `LikelihoodMaxOptResult`, kept in `info["likelihood_max_opt"]` | [A3](#a3-find-the-likelihood-peak) |
| `forecast` | `(times, bands, **kwargs)` → DataFrame | [A7](#a7-forecast-and-choose-the-next-observation) |
| `facts` | `(lc, **kwargs)` → dict (`result_facts(self, lc, **kwargs)`) | [A9](#a9-facts-and-the-html-report) |
| `fitted_lc` | `(lc)` → the rows of `lc` the fit used | [A2](#what-every-fit-does) |

Read-only **metric properties** sit beside `aic`/`bic` so the predictive numbers take one hop instead of
four: **`waic`** (deviance scale, lower is better), **`waic_reliable`** (`p_waic ≤ n_data/2` — check it
before comparing WAICs), **`elpd_loo`** (PSIS-LOO, higher is better) and **`rmse`** (overall, in the
fit's comparison space). Each returns `None` — never raises — when the metric block is missing or failed
(the reason is then in `info["predictive_metrics_error"]`), and `elpd_loo` is `None` without `arviz`.
They are views on `info["predictive_metrics"]`, deliberately **not** added to `to_dict()`/`to_json()`,
which already carry the whole block and whose shape is unchanged.

**Writing a sampler that produces the standard output** (`whisper_cbpf.samplers.base`): the two hooks
below are what every built-in sampler calls, and a custom sampler should call them too so its JSON
carries the same metric blocks as the rest. Both are **best-effort** — they swallow any exception, so a
slow or failing forward model degrades the metrics instead of breaking an otherwise-successful fit.

| Helper | Signature | Effect |
|---|---|---|
| `summarize_posterior` | `(df, parameters)` | → the `summary` dict (median / ci16 / ci84 / mean / std per parameter). |
| `attach_band_metrics` | `(info, lc, model, best_params, space)` | Fills `info["band_metrics"]` with `per_band_metrics` at the best fit. Call **before** building the `SamplerResult`, on the dict you pass as `info`. |
| `attach_predictive_metrics` | `(result, lc, space, n_draws=200)` | Fills `result.info["predictive_metrics"]` with `predictive_metrics` (§6.6). Call **after** the result exists — it reuses the fit's `aic`/`bic`. `n_draws` bounds the extra forward-model evaluations. |

> Metrics note: for the χ² distance, `chi2 = -2 ln L` (Gaussian), so `max_log_likelihood = -0.5·χ²_min`,
> `AIC = χ²_min + 2k`, `BIC = χ²_min + k·ln(n)`.

```python
res = wp.fit_ABC(r_band_lc, "flare", prior=prior, n_simulations=200_000, quantile=0.005, n_jobs=16)
res.summary["amplitude"]    # {'median':..., 'ci16':..., 'ci84':...}
res.best_params; res.aic; res.bic
res.to_json("fit.json")
```

---

### 6.5 Likelihoods  (`whisper_cbpf.likelihood`)

Models predict **flux**; a likelihood compares it to the data in a chosen **space** —
`space='flux'` (residuals/errors in Jy; the **only** space in which upper limits are usable),
`space='magnitude'` (model flux → AB mag vs observed mag/err), or `space='auto'` (magnitude data →
magnitude space, flux data → flux space; the correct default). Each exposes
`log_likelihood(model_flux) -> float` and is picklable.

**Upper limits are a flux-space statement.** A non-detection says the *flux* was below a limiting
flux, and the magnitude of zero flux is undefined (`−2.5 log₁₀ 0 = +∞`), so there is no
magnitude-space censoring integral over the same quantity. `space='auto'` on a light curve with
upper limits (after the pre-event cut) therefore resolves to **flux**, and the censored likelihood is
used with no argument: detections are scored on flux, limits by the probability that the flux was
below them. An explicit `space='magnitude'` raises. Fit the detections alone in magnitude space with
`lc.where(upper_limit=False)`.

**A likelihood that does not model censoring refuses censored data.** Building `GaussianLikelihood`,
`MixtureGaussianLikelihood` or `GaussianLikelihoodWithScatter` on a light curve with any
`upper_limit=True` row raises. Previously the limits' NaN error bars flowed into the normalising
constant and every log-likelihood came back `NaN` — silently, so the fit completed and reported NaN
`max_log_likelihood`/AIC/BIC. There is no combined free-scatter **and** upper-limit likelihood, so
those two are mutually exclusive rather than silently resolved in favour of scatter. Non-finite or
non-positive error bars on *detections* are refused for the same reason.

| Class / function | Purpose |
|---|---|
| `GaussianLikelihood(lc, space="auto")` | Independent Gaussian in the chosen space. |
| `GaussianLikelihoodWithScatter(lc, space="auto", scatter_param="sigma")` | Gaussian with a **free extra-scatter term added in quadrature** (Villar+2017): `lnL = −½Σ[(O−M)²/(σᵢ²+σ²) + ln(2π(σᵢ²+σ²))]`; `log_likelihood(model_flux, sigma_extra=…)`. `kind="gaussian_scatter"` / `"villar"`. |
| `GaussianLikelihoodWithUpperLimits(lc, space="auto", upper_limit_sigma=None, zeropoint_jy=3631.0)` | Gaussian for detections + a censoring term `P(true flux < limit) = Φ((limit − model)/(limit/upper_limit_sigma))` for the non-detections. **Flux space only** — `space="magnitude"` raises. `upper_limit_sigma=None` reads `lc.meta["upper_limit_sigma"]` (5 for the survey presets), else `DEFAULT_UPPER_LIMIT_SIGMA = 5`. |
| `MixtureGaussianLikelihood(lc, space="auto", alpha=0.9, sigma_out_scale=10.0)` | Outlier-robust two-component mixture (α, σ_out fixed). |
| `make_likelihood(lc, kind="auto", space="auto", **kw)` | Build the data-appropriate likelihood (auto-selects upper-limits when present). |

Likelihoods are a **name registry** like models/samplers/distances — `kind=` on `make_likelihood` (and
`likelihood=` on `MCMCSampler.fit`) is a registered name:

| Function | Signature | Description |
|---|---|---|
| `register_likelihood` | `(name, likelihood_cls, *, overwrite=False)` | Register your own. The class must accept `(lc, space=..., **kwargs)` and expose `log_likelihood(model_flux) -> float`; subclass `GaussianLikelihood` for the easy path (you then inherit `log_likelihood_pointwise`, and WAIC/LOO keep working). |
| `list_likelihoods` | `()` | Every registered `kind` name **including aliases** — `gaussian`/`normal`, `gaussian_scatter`/`scatter`/`villar`, `gaussian_upper_limits`/`upper_limits`/`ul`, `mixture`/`mixture_gaussian`/`outlier`. |

Names are matched case-insensitively; an unknown `kind` raises `ValueError` listing the available names.

**Free scatter routing:** a prior parameter named after the scatter term (default `"sigma"`) is a
*likelihood* parameter, sampled with the rest — MCMC routes it via `likelihood="gaussian_scatter"`;
ABC/ABC-SMC/SNPE take `scatter_param="sigma"` and fold it into their **generative noise**
(`N(0, √(σᵢ²+σ²))` with each draw's value), so every method fits the same model. Caveat (verified on
synthetic data): a plain χ² rejection *distance* is monotonically penalised by extra noise, so the
scatter level is **not identifiable by distance-based ABC** — fit σ with MCMC or neural SBI.

> **ABC/ABC-SMC comparison space:** `space="auto"|"flux"|"magnitude"` now routes the ABC acceptance
> itself — data, simulations, noise and distance all live in the chosen space, and
> `AIC`/`BIC`/`max_log_likelihood` come from the exact likelihood there (comparable across samplers).

Each Gaussian likelihood also exposes **`log_likelihood_pointwise(model_flux) -> array`** (the per-data-
point log-likelihood, summing to `log_likelihood`) — the ingredient WAIC needs.

### 6.6 Metrics  (`whisper_cbpf.metrics`)

**`waic(posterior, lc, model=None, *, space="auto", likelihood="auto", scatter_param="auto", fixed=None, max_samples="auto", seed=0)`**
— the **Widely Applicable Information Criterion** (Watanabe 2010; Gelman et al. 2014): a fully-Bayesian
fit score that, unlike AIC/BIC (which use one best-fit point), uses the **whole posterior**. It evaluates
the model's *pointwise* log-likelihood across the posterior draws and returns a dict with `waic`
(`= -2(lppd - p_waic)`, **lower is better**), `elpd_waic`, `lppd`, `p_waic` (effective # parameters),
`p_waic_reliable`, `se` (standard error), `n_samples`, `n_data`, `scatter_param`.

**Given a `SamplerResult`, this returns the fit's own number:** `wp.waic(res, lc)["waic"] == res.waic`.
Each `"auto"` means *whatever the fit recorded* — the space it was fitted in, the likelihood kind it was
fitted under, the extra-scatter column it fitted, and the same draw count and seed the auto-attached
block scored. Pass any argument explicitly to override. For a bare `DataFrame`/array there is no fit to
follow, so `"auto"` keeps the plain meanings (data-appropriate space and likelihood, lone non-model
column as σ, 2000 draws). `fixed=` supplies values for parameters pinned during the fit (so absent from
the posterior columns); `max_samples` caps the per-draw model evaluations (matters for slow simulators).

`p_waic` uses the **sample** variance (`1/(S−1)`), as Watanabe 2010 and Vehtari, Gelman & Gabry 2017
Eq. 11 define it; `arviz.waic` uses the population variance `1/S`, so its `p_waic` is this one times
`(S−1)/S` (0.5 % at S = 200, O(1/S) in general). Note `p_waic` (and hence WAIC) inflates for posteriors
much broader than the likelihood — e.g. ABC's tolerance posterior or an under-converged SNPE run — which
is itself a useful diagnostic; magnitude space is numerically gentler than flux space (whose tiny errors
make the likelihood very sharp).

**`per_band_metrics(lc, model, params, *, space="auto", fixed=None)`** — deterministic **per-band
goodness of fit** at a single parameter set (typically `result.best_params`). Evaluates the model once
and reports, **per band** and overall, the `mse` / `rmse` / `mae` of the residuals `observed − model`
in the fit's comparison space (`"flux"` → Jy, `"magnitude"` → mag). Returns
`{"space", "unit", "n_upper_limits_excluded", "bands": {band: {"mse","rmse","mae","n"}}, "overall": {...}}`.
Every sampler calls this at its best fit and stores the result in `result.info["band_metrics"]`, so it
is reported in `to_json()` automatically — no extra call needed for the standard fit output.

**Detections only.** Rows flagged `upper_limit=True` are excluded and counted in
`n_upper_limits_excluded`; every `n` counts detections. A non-detection's `y` is a *limit*, not a
measurement, so `observed − model` on that row is not a residual — a model correctly *below* an upper
limit still contributes `(limit − model)²`, and the deeper the limit the worse it makes a correct fit
look. Bands with no detection are kept with `n = 0` rather than dropped from the report. No error bars
are read (the space conversion goes through `likelihood.resolve_space` / `likelihood.flux_to_space`,
not a likelihood object), so the NaN `flux_err` the loader writes for a non-detection is fine — before
this, building a `GaussianLikelihood` purely as a space converter made every censored fit report
`info["band_metrics_error"]` instead of `info["band_metrics"]`.

**`predictive_metrics(result, lc, model=None, *, space="auto", likelihood="auto", scatter_param="auto", fixed=None, levels=(0.5, 0.68, 0.8, 0.9, 0.95, 0.99), n_draws=400, seed=0)`**
— the **posterior-predictive metric block** reported for every fit. Evaluates the model across `n_draws`
posterior samples once and returns:
- **`rmse`** — root-mean-squared error of `observed − posterior_mean_prediction`, per band + overall.
  **Detections only** (see below).
- **`lpd`** — log predictive density `Σ_i log mean_s p(y_i|θ_s)` (the `lppd`; higher is better), `total`
  and `per_point`. Over **every** row, non-detections included.
- **`elpd_loo`** — expected log predictive density by **PSIS-LOO** cross-validation (Vehtari 2017;
  higher is better) with `p_loo`, `se`, `looic = −2·elpd_loo`, and `pareto_k_max` (>0.7 ⇒ unreliable);
  uses `arviz` when installed (`None` otherwise).
- **`waic`** — `waic` (deviance, lower is better), `elpd_waic`, `p_waic`, `se`.
- **`aic` / `bic`** — carried through from the fit's best-fit likelihood.
- **`coverage`** — the **calibration curve**: for each nominal `level`, the empirical fraction of
  observations inside the central posterior-predictive interval, overall and per band (empirical ≈
  nominal ⇒ calibrated). **Detections only** (see below).
- **`n_upper_limits_excluded`** — rows dropped from `rmse` / `coverage` because they are non-detections.
- **`scatter_param`** — the posterior column used as the extra-scatter σ (or `null`).
- **`space` / `likelihood` / `scatter_param` / `n_draws_requested` / `seed`** — how this block was
  scored. `waic(result, lc)` reads them back so the manual call reproduces exactly this number.
- **`elpd_loo_error`** — present only when `arviz` **is** installed and `az.loo` raised; `elpd_loo` is
  `null` both when arviz is absent (documented degradation) and when it failed, and this key is what
  tells the two apart.

**Non-detections are in the density, not in the point comparisons.** `rmse` and `coverage` score the
**detections** alone, with `n_upper_limits_excluded` reporting the rest; `lpd`, `waic` and `elpd_loo`
score **every** row. The asymmetry is deliberate. A censored row's `y` is a *limit*: its principled
contribution to a predictive *density* is the survival term `log P(flux < limit)`, which
`log_likelihood_pointwise` supplies and which is real information about the fit — but `observed − model`
on that row is not a residual (a model correctly *below* an upper limit still contributes
`(limit − model)²`, the more so the deeper the limit), and asking whether a bound falls inside a central
predictive interval is not a calibration question. Before this, both blocks ran over every row and
silently reported inflated error and collapsed calibration on any censored fit. Bands with no detection
stay in both blocks as `NaN` rather than being dropped. Same rule as `per_band_metrics` (§6.6).

`levels` defaults to **`DEFAULT_COVERAGE_LEVELS = (0.5, 0.68, 0.8, 0.9, 0.95, 0.99)`** — import it from
`whisper_cbpf.metrics` to extend rather than retype the set (e.g.
`levels=DEFAULT_COVERAGE_LEVELS + (0.999,)`). Note `plot_calibration` does **not** read the stored
block: it takes its own `levels=` (same values, and recomputes the coverage), so a custom set must be
passed to both.

**Scatter-aware predictive density.** The log-predictive quantities (LPD, WAIC, PSIS-LOO) and the
coverage intervals are evaluated under the *same* generative model the fit optimised. When the fit has a
free **extra-scatter** σ (Villar+17; a "jitter"/intrinsic-scatter term à la Hogg, Bovy & Lang 2010), the
pointwise density is the scatter-augmented Gaussian `𝒩(M_i, σ_i²+σ_s²)` with **each draw's own** σ_s,
and the predictive replications draw from the same inflated variance. Dropping σ mis-specifies the
density: for high-SNR photometry it collapses, so `p_waic`/`p_loo` explode and coverage collapses —
folding σ back in restores near-nominal coverage and sane WAIC/LOO. `scatter_param`: `"auto"` (default)
uses the lone posterior column that is not a model parameter (e.g. `sigma`); a name forces it; `None`
disables (reported errors only — the honest default for distance-based ABC, which fits no σ). Marginalising
the predictive over the full posterior including nuisance scatter is the standard prescription (Gelman
et al., *BDA3*, ch. 6–7; Vehtari, Gelman & Gabry 2017; Watanabe 2010).

Every sampler attaches this to **`result.info["predictive_metrics"]`** (with `n_draws=200`) so it lands
in `to_json()`. Note that broad tolerance/under-converged posteriors (ABC, un-converged SNPE) inflate
`p_waic`/`p_loo` and their information criteria — a known, useful diagnostic, not a bug.

### 6.7 Validation — recovery, PPC, SBC  (`whisper_cbpf.validation`)

Sampler-agnostic checks that a fit **recovered the truth** with **reliable uncertainties** (used by
[`sanity_check/sanity_check.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/sanity_check.py),
pinned history: WHISPER_AI @ 8ba3843, discontinued); all take a `SamplerResult` (or its `.samples`) so they work for every sampler.

**`recovery_metrics(result, truth)`** — per-parameter recovery of a known `truth` (dict): posterior
`median`/`mean`/`std`, 68% (16–84) and 95% (2.5–97.5) credible intervals, `bias = median − true`, the
standardized **`z_score` = bias/std** (`|z|≲2` ⇒ recovered), and boolean 68/95% `within` coverage; a
top-level `_summary` gives `max_abs_z`, `rms_z`, `coverage68`/`coverage95`.

**`posterior_predictive_check(result, lc, model=None, *, n_draws=300, time_grid=None, seed=0)`** — a
posterior-predictive **band** on a grid, the **reduced χ² at the best fit** (goodness-of-fit, decoupled
from posterior width), noise-inflated **predictive coverage** `ppc_coverage68`/`95` (fraction of data in
the predictive band — the clean calibration metric), and a Bayesian χ² `bayesian_p_value` (≈0.5 healthy).

**`sbc_rank(samples, true_value)`** / **`sbc_ranks(ranks_by_param, *, n_bins=20)`** — Simulation-Based
Calibration (Talts 2018; Säilynoja 2022). Over `L` prior→data→fit realizations the rank of each true
value within its posterior is **uniform** iff the posterior is calibrated; `sbc_ranks` returns the rank
histogram + a **χ²-of-uniformity p-value** per parameter (∪-shape = overconfident, ∩-shape =
underconfident, slope = biased) and a `_summary` with `min_uniformity_p` + a `calibrated` verdict.

`check_parity(model_a, model_b, params_or_posterior, times, bands, *, tolerance=0.02, n=200,
seed=0)` compares two models' band magnitudes: [A11](#a11-check-that-two-models-agree).
`posterior_predictive_check` and the metrics that take `(result, lc)` score the rows the fit used
(`result.fitted_lc(lc)`).

Exposed as `wp.recovery_metrics`, `wp.posterior_predictive_check`, `wp.sbc_rank`, `wp.sbc_ranks`. See
[`sanity_check/figures/REPORT.md`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/figures/REPORT.md)
(pinned history: WHISPER_AI @ 8ba3843, discontinued) for the full synthetic-recovery benchmark across all five samplers.

## Notes & limitations (review findings)

- **Metrics are cross-sampler comparable:** ABC/ABC-SMC still *accept* on the flux χ² distance, but
  their `max_log_likelihood`/`AIC`/`BIC` are now the **exact Gaussian log-likelihood at the best fit** in
  the data's natural space (`info['likelihood_space']`) — the same convention as MCMC/SNPE — so AIC/BIC
  can be compared across samplers. (The ABC *posterior* is still set by the flux-space acceptance; only
  the reported best-fit metric uses the natural-space likelihood.)
- **ABC-SMC is importance-weighted** (Beaumont 2009 / Toni 2009): weighted resampling + an adaptive
  diagonal-Gaussian kernel + weights `w_i ∝ π(θ_i)/Σ_j w_j K(θ_i|θ_j)`; the returned posterior is the
  equal-weight resample, and per-round effective sample size is in `info['rounds']`.
- **ABC posteriors are approximate** (broadened by the acceptance ε); tighten ε / use SMC for sharper ones.
- **Toy models** are band-independent analytic forms: `flare` = 0 before explosion; `bazin` computed
  stably in log-space; `gaussian_rise` has a derivative kink at the peak. **`mck19`** is a built-in
  *physical*, **band-dependent** model (blackbody hotspot + AGN disk, redshift-aware). Further physical,
  band-dependent models can optionally be supplied by the external redback `[models]` extra (a
  models/priors source only).
- **SNR(magnitude)** uses `(2.5/ln10)/σ_m`, valid for small magnitude errors.

## 7. Internals
`io.loader._resolve_columns`, `schema.LightCurve._subset/_copy`, `plotting._categories/_scatter`,
`samplers.abc._simulate_batch/_worker`, `samplers.base.summarize_posterior`,
`io.units.to_canonical`, `io.svo._svo_fetch_metadata/_svo_fetch_index/_svo_fetch_transmission`
(network boundary), [`examples/demo_ingestion.py`](../examples/demo_ingestion.py), and — pinned
history, WHISPER_AI @ 8ba3843, discontinued —
[`dev/phase0_smoke.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/phase0_smoke.py),
[`dev/demo_abc_at2017gfo.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_abc_at2017gfo.py).

## 9. End-to-end example

```python
import whisper_cbpf as wp

lc = wp.load_lightcurve("at2017gfo.csv", explosion_date=57982.0, min_snr=3)
r  = lc.select_bands("r")
fmax = r.add_flux().flux.max()
prior = wp.Prior({"amplitude": wp.Uniform(0, 10*fmax),
                  "rise_time": wp.Uniform(0.05, 10), "decay_time": wp.Uniform(0.5, 40)})
res = wp.fit_ABC(r, "flare", prior=prior, n_simulations=200_000, quantile=0.005, n_jobs=16)
print(res, res.best_params)
```

---

## 10. The JAX/GPU half

Everything below needs the `[gpu]` extra (`pip install "whisper-cbpf[gpu]"`) **and** the
environment bootstrap sourced before Python starts. Signatures and summaries below are
produced by introspecting the installed package, not transcribed.

Naming: GPU samplers keep their registered names (`nuts_gpu`, `abc_gpu`, …) and JAX distances
take a `_jax` suffix (`mse` → `mse_jax`). The photometric models do **not** auto-register —
bind them with a factory. See [`CHOOSING.md`](CHOOSING.md) for which to use when.

### Backends and the GPU environment  (`whisper_cbpf.backends`)

- **`check_gpu(strict=False)`**  
  Return ``(ok, message)`` describing whether JAX can actually see a GPU.
- **`require_jax(feature='this sampler')`**  
  Import and return ``(jax, jnp)``, or raise with an actionable message.
- **`x64_enabled()`**  
  Whether JAX is in float64 mode. Must be decided BEFORE any jax array is created.
- **`gpu_list(n=2, prefer_idle=True)`**  
  The device ids to expose, as a CUDA_VISIBLE_DEVICES string.
- **`n_jobs(fraction=0.6)`**  
  Worker count at the production CPU budget (60% of cores by default).
- **`env_script()`**  
  Absolute path to `whisper_cbpf/backends/env.sh`, the shell bootstrap. It lives inside the package
  and is listed in `package-data`, so it exists in a wheel install as well as a checkout. The
  console script `whisper-cbpf-env` prints it, which is why
  `source "$(whisper-cbpf-env)"` works everywhere.
- **`env_report()`**  
  A dict describing the current GPU environment. Never raises, never imports jax.

### JAX model factories  (`whisper_cbpf.models.jax`)

- **`flare_model()`**  
  The JAX flare as a ``whisper_cbpf.models.Model``. Self-contained, auto-registered.
Every photometric factory below takes `redshift=None, dl_cm=None` (required unless the redshift is
free), `free=["t_exp", "redshift"]` and `redshift_prior=`: the explosion (merger) time and the
redshift as traced parameters ([A2](#free-explosion-time-and-redshift-jax-factories)).

- **`kilonova_model(band_names, redshift=None, dl_cm=None, *, name='kilonova_one_jax', n_wave=None, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', temperature_floor=None, mag_floor=40.0, t_exp_days=0.0, filter_set=None, arnett_prefactor=1.0, time_grid='auto', band_aliases=None, default_system=None, free=None, prior=None, redshift_prior=None)`**  
  Bind dataset context to the one-component JAX kilonova and return a whisper ``Model``. **Band
  integral (every factory):** `filter_set=None, n_wave=None` is Gauss-16 per band (§4b); pass a
  `FilterSet` (the one a CPU model uses, to compute the same integral) or a `make_filter_set` dict
  as `filter_set=`, or `n_wave=` for whisper ≤ 0.1.0's shared grid (it was `n_wave=2000`; ≤ 0.005
  mmag apart). `band_names` are filter names or labels resolved by `resolve_filter`: bare letters are
  LSST unless `default_system=` says otherwise. Every factory fills `Model.predict_jax`:
  `predict_jax(theta, times, band_idx)` returns Jy
  from a flat theta, is differentiable, and carries `predict_jax.band_index(bands)` for the
  string → index map. Both it and `predict` go through one core jitted at factory time.
  `time_grid='auto'` is redback's 500-node grid with nothing free and the converged quadrature when
  anything is free; `prior=` overrides the default distributions per parameter.
- **`register_kilonova(band_names, redshift=None, dl_cm=None, *, name='kilonova_one_jax', **kwargs)`**  
  Build a kilonova ``Model`` for this dataset and register it under ``name``.
- **`kilonova_two_model(band_names, redshift=None, dl_cm=None, *, name='kilonova_two_jax', n_wave=None, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', mag_floor=40.0, t_exp_days=0.0, filter_set=None, prior=None, time_grid='auto', band_aliases=None, default_system=None, free=None, redshift_prior=None)`**  
  The blue + red kilonova (Villar et al. 2017 priors with disjoint opacity corridors;
  `prior=Prior({"t_exp": ...})` alone keeps them). `register_kilonova_two(band_names, redshift=None,
  dl_cm=None, *, name='kilonova_two_jax', **kwargs)` builds and registers it.
- **`kilonova_three_model(band_names, redshift=None, dl_cm=None, *, name='kilonova_three_jax', n_wave=None, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', mag_floor=40.0, t_exp_days=0.0, filter_set=None, prior=None, time_grid='auto', band_aliases=None, default_system=None, free=None, redshift_prior=None)`**  
  Blue + purple + red. `register_kilonova_three(band_names, redshift=None, dl_cm=None, *,
  name='kilonova_three_jax', **kwargs)` builds and registers it.
- **`tde_model(band_names, redshift=None, dl_cm=None, *, name=None, rise='gaussian', n_wave=None, n_time=None, f_debris=1.0, xi=1.0, n_pre=100, pin=None, prior=None, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', filter_set=None, band_aliases=None, default_system=None, t_exp_days=0.0, constraint='corrected', free=None, redshift_prior=None, **engine_kwargs)`**  
  Bind dataset context to the JAX cooling-envelope TDE and return a whisper ``Model``.
  `**engine_kwargs` are the engine's own (`t_0_init`, `binding_energy_const`, `zeta`, `hoverR`);
  anything else raises here rather than at the first predict. `t_exp_days` is the model's time
  zero on the light curve's clock (fallback for `rise='none'`), subtracted on the host in float64.
- **`register_tde(band_names, redshift=None, dl_cm=None, *, name=None, **kwargs)`**  
  Build a TDE ``Model`` for this dataset and register it under ``name``.
- **`supernova_model(model, band_names, redshift=None, dl_cm=None, *, name=None, n_wave=None, t_exp_days=0.0, dense_resolution=None, spacing='geometric', csm_interp=True, magnetar_convention='1.15', interaction=True, dilation=True, pin=None, prior=None, mag_floor=40.0, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', filter_set=None, times=None, band_aliases=None, default_system=None, constraint='corrected', free=None, redshift_prior=None, diffusion_grid=None, max_phase_days=None, epochs_per_decade=None)`**  
  Bind dataset context to one of the JAX supernova models and return a whisper ``Model``.
  `diffusion_grid`: `"data"` (redback's scheme on the observed epochs, fixed at compile time) or
  `"fixed"` (fixed source-frame epochs, the SED at the observations only; the default when anything
  is free), [A2](#free-explosion-time-and-redshift-jax-factories).

**What the JAX samplers read off `predict_jax`** besides `band_index` (every photometric factory):
`predict_jax.constraint_ok(theta) -> bool` — redback's `Constraint` priors for `arnett`,
`basic_magnetar_powered`, `slsn`, `general_magnetar_slsn` and both TDEs, armed by `constraint=`
(same modes as the CPU adapter; `None` elsewhere) — which `samplers.jax._adapters` applies as
`-inf` log-density and zero flux in the batched forward map; and `predict_jax.mag_floor`, the
magnitude cap (redback has none). `make_batched_predict_jax` counts the points at the cap into
`.floor_stats` (`n_points`, `n_floored`, `fraction`, `mag_floor`) and warns once above
`MAG_FLOOR_WARN_FRACTION = 0.10`. `predict_jax` itself is the physics, with no wall; the host `predict`
(what the CPU samplers call) applies the same wall as zero flux, like the CPU redback adapter.
`t_exp_days` is subtracted on the host in float64 in every factory, and the adapters pass the
float64 epochs, so a raw-MJD clock is safe in float32 (kilonovae).
- **`supernova_models()`**  
  The names :func:`supernova_model` accepts.
- **`register_supernova(model, band_names, redshift=None, dl_cm=None, *, name=None, **kwargs)`**  
  Build a supernova ``Model`` for this dataset and register it under ``name``.

### JAX/GPU samplers  (`whisper_cbpf.samplers.jax`)

**`likelihood=` on the scalar-density samplers.** `nuts_gpu`, `pymc_jax_gpu_*` and `emcee_jax` build
their density from `model.predict_jax` when no `log_prob_fn` is passed, and `likelihood="auto"`
(default) selects which one — the same registered `kind` names `make_likelihood` takes, restricted
to what the JAX backend implements (`auto`, `gaussian`, `upper_limits`, `gaussian_scatter`;
`mixture` is refused). It is ignored when you pass your own `log_prob_fn`. The resolved class is
recorded in `result.info["likelihood"]` — the same key the numpy samplers write, and the one
`attach_predictive_metrics` reads back so WAIC/LOO re-score a fit under the density it was fitted
with. `likelihood="gaussian_scatter"` needs the scatter parameter (default `"sigma"`) in the
`prior`: the sampled vector then grows by that column, which is routed to the likelihood's
`sigma_extra` instead of to `model.predict`, and `info["scatter_param"]` names it.

**Where the chains start.** `NUTSGPUSampler.fit` and the two PyMC samplers take
**`init_strategy="prior_scan"`** (the default; `None` means the same), `fit_emcee_jax` takes
**`init="prior_scan"`**: 1000 prior draws (`max(1000, 4·nwalkers)` for emcee) are scored in one device
program, the best 32 climbed a short monotone gradient ascent to learn which basin each belongs to, and
each chain (walker) starts ~2 posterior sd from a distinct climbed point that reached the best basin. It
replaced NumPyro's `init_to_uniform`, PyMC's jitter and emcee's uniform box, which left chains in local
optima (a model hidden between the epochs, a prior-box corner): broken runs fell from 22 / 100 to 0
(`nuts_gpu`, Gaussian bump) and 78 / 100 to 0 (`emcee_jax`). The old starts are one name away —
`"uniform"` (NumPyro), `"jitter"` (PyMC), `init="box"` (emcee) — beside `"prior"` (independent prior
draws, not ranked, at which the density is finite), NumPyro's `"median"`, `"sample"`, `"feasible"`,
`"mean"` or any NumPyro init callable, and one start per chain as a `(num_chains, k)` array in
parameter units. `emcee_jax` also takes a point (a dict or `(k,)` array) or `(point, scale)` for a
ball, a `(nwalkers, k)` array, or a previous `SamplerResult` to continue from. A start where the
density is `-inf` is skipped or redrawn (`"prior"`, emcee's random starts) or refused (your per-chain
or per-walker starts, a previous result). A point start with more than
one chain warns (R-hat then cannot tell a stuck chain from a converged one). `info["init_strategy"]`
(`info["init"]`, `info["init_detail"]` for emcee) records what ran, and `info["prior_scan"]` the scan
(`n_draws`, `n_climbed`, `n_reaching_best`, `plateau_fraction`, `climb_error`,
`best_log_likelihood`, `start_log_likelihood`). The start costs seconds to minutes, compile included,
on the CPU or one A6000 alike: 7-13 s for a kilonova or an arnett supernova. On the TDE it is mostly
compiling the climb's gradient through the ODE, so it still dominates a short `emcee_jax` run there
(`info["init_time_s"]`): a 60 × 300 fit at `n_time=500` on one heavily loaded A6000 spent 96-97 s on
the start and 3-4 s sampling. The curvature that spreads the starts is a finite difference through
the scan's compiled scorer; as a `jax.hessian` it took the same start to 219-361 s. The scan
runs whatever the start, because its best point is the independent optimum every run is checked
against. Since 0.2.0 every one of these samplers, and CPU `mcmc`, also takes the shared `init=`
keyword (a point, `(point, scale)`, one start per chain, or a previous result such as an ABC fit):
[A2](#where-the-chains-start-init). `nuts_gpu` and the PyMC samplers refuse `init=` together with
`init_strategy=`.

**What `converged` means** (`nuts_gpu`, `pymc_jax_gpu_*`). `info["converged"]` is `True` only when
`info["convergence_problems"]` — one sentence per failed check, repeated in a warning — is empty: no
divergence (`n_divergences`); rank-normalised R-hat < 1.01 on every parameter (`rhat`, `max_rhat`: a
float, NaN when unknown) and on the per-chain log-likelihood (`rhat_log_likelihood`); bulk and tail ESS ≥
100 per chain (`ess_bulk`, `ess_tail`, `min_ess`); no **stranded** chain, i.e. none whose median
log-likelihood is more than 10 nats below the best chain's (`stranded_chains`,
`chain_median_log_likelihood`); no **frozen** chain, i.e. none whose adapted step is below 1e-2 of the
median while it barely moves (`frozen_chains`, `step_size_by_chain`); and no point from the prior scan
more than 10 nats above every chain's best draw (`reference_log_likelihood`) — the all-chains-wrong case
R-hat cannot see. `rhat_method` says how R-hat was computed (arviz's rank-normalised statistic on arviz
0.x and 1.x; NumPyro's split R-hat if arviz fails, with the reason in `rhat_error`). `emcee_jax` reports
`stuck_walkers`, `max_autocorr_time` and `convergence_problems` as `MCMCSampler` does (§6.4).

**Guards before sampling.** With `jax_enable_x64` off, the three samplers warn when `lc.time` or a
Uniform prior bound reaches 1e3 in absolute value (an MJD clock: float32 moves it in 0.0039 d steps and
chains freeze) and record the message in `info["float32_hazard"]`; enable x64 or
`lc.set_explosion_date(mjd)`. Under the default start they warn when ≥ 90 % of the scan's draws score
one flat level (a time prior much wider than the data window). A density flagged `includes_prior`
(a log-*posterior*, what `emcee_jax` builds) passed to `nuts_gpu` or `pymc_jax_gpu_*` is refused when any
prior is not Uniform (the prior would be counted twice) and warned about otherwise. An empty light curve
is refused by every sampler, a caller-supplied `log_prob_fn` included.

**Timing keys.** `runtime_s` is the start plus warmup and sampling (`info["init_time_s"]`,
`["warmup_time_s"]`, `["sampling_time_s"]`; emcee: the start plus `run_mcmc`, its compile at every
ensemble shape emcee uses in `["compile_time_s"]`). **`info["postprocess_s"]`** is what follows:
the log-likelihood of every kept draw, in blocks of 250 (`_diagnostics.SCORE_BLOCK`), and the chain
checks. The block widths are measured defaults, and compile time on the GPU is erratic in the width
(the kilonovae compile in seconds at 250 and 1000 but in minutes at 256 or 1024), so prefer widths that
are not powers of two: `fit_emcee_jax(walker_chunk="half")` (one vmapped block per emcee call; an int
fixes it, `None` is one wide vmap, and `info["walker_chunk"]` records the width), `abc_gpu` /
`abc_smc_gpu` `chunk=250`, `snpe_gpu` `sim_chunk=250`. The simulation-based samplers record
`info["mag_floor_stats"]` — the points simulated at the models' `mag_floor` (see the factories above);
`snpe_gpu` counts every simulation, while the ABC pair run the forward map inside one compiled scan,
where nothing is counted, and report `None`.

- **`NUTSGPUSampler()`** — *class*  
  NUTS (NumPyro) on GPU. See module docstring for the shared-``log_prob_fn`` design.
- **`NUTSGPUSampler.fit(lc, model, prior=None, *, log_prob_fn=None, num_warmup=1000, num_samples=2000, num_chains=4, space='auto', likelihood='auto', seed=0, progress=False, target_accept_prob=0.8, init=None, init_strategy='prior_scan', dense_mass=False, max_tree_depth=10, step_size=1.0, chain_method='vectorized')`**  
  The PyMC samplers' `fit` takes the same arguments up to `init_strategy`.
- **`fit_NUTSGPU(lc, model, prior=None, *, log_prob_fn=None, **kwargs) -> 'SamplerResult'`**  
  Fit ``lc`` with ``model`` via NumPyro NUTS on GPU. See :meth:`NUTSGPUSampler.fit`.
- **`ABCGPUSampler()`** — *class*  
  Rejection ABC with a vmapped JAX simulator. See the module docstring for matched semantics.
- **`fit_ABC_GPU(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  Fit ``lc`` with ``model`` by rejection ABC on the GPU. See :meth:`ABCGPUSampler.fit`, whose
  signature, one model evaluation per draw and `precision=` are in
  [A2](#abc-the-worst-point-distance-and-single-precision).
- **`ABCSMCGPUSampler()`** — *class*  
  Importance-weighted ABC-SMC with a vmapped JAX simulator. See the module docstring.
- **`fit_ABCSMCGPU(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  Fit ``lc`` with ``model`` via ABC-SMC on GPU. See :meth:`ABCSMCGPUSampler.fit`.
- **`EmceeJAXSampler()`** — *class*  
  whisper registry adapter for :func:`fit_emcee_jax`.
- **`fit_emcee_jax(lc, model, log_prob_fn=None, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000, thin=10, seed=0, progress=False, space='auto', likelihood='auto', walker_chunk='half', init='prior_scan', walker_coordinates='own') -> 'SamplerResult'`**  
  Arm A(1): emcee vectorized against the shared jitted JAX log-density (a log-**posterior**: emcee
  has no prior of its own; built from `model.predict_jax` when omitted). `init=`, `walker_coordinates=` and `walker_chunk=`
  as above.
- **`fit_emcee_numpy(lc, model, times, values, sigmas, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000, thin=10, seed=0, n_jobs=8, progress=False, space='flux', init='prior_scan') -> 'SamplerResult'`**  
  Arm A(2): plain-numpy emcee on N CPU cores — the realistic no-GPU baseline. Its default start is
  the `nwalkers` best-scoring of `max(1000, 4·nwalkers)` prior draws (no gradient to climb with).
- **`PyMCJAXVectorizedSampler()`** — *class*  
  PyMC frontend, NumPyro NUTS, chains vmapped on one device.
- **`fit_PyMCJAXVectorized(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  See :class:`PyMCJAXVectorizedSampler`.
- **`PyMCJAXParallelizedSampler()`** — *class*  
  PyMC frontend, NumPyro NUTS, chains ``pmap``ed across devices — the multi-GPU path.
- **`fit_PyMCJAXParallelized(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  See :class:`PyMCJAXParallelizedSampler`.
- **`fit_snpe_gpu(lc, model, *, prior=None, num_rounds=2, num_simulations=10000, embedding_net=None, embedding_latent=32, density_estimator='maf', num_samples=10000, seed=0, device='cuda', show_progress=False, embedding=None, **snpe_kwargs)`**  
  Fit one light curve with SNPE on GPU, using the JAX model for GPU-batched simulation.
  `num_rounds` and `embedding_net` mean what they mean in `SNPESampler.fit`; `embedding=` is a
  deprecated alias of `embedding_net=`. `SNPEGPUSampler.fit` (registered as `snpe_gpu`) takes
  `sim_chunk=250` simulations per compiled block and passes every other keyword
  (`proposal_min_acceptance`, `num_chains`, …) to `SNPESampler.fit`, so its `info` carries the same
  `leakage`, `converged`, `x_o_min_rms_z` and `postprocess_s`, plus `sim_chunk`, `sim_transfer` and
  `mag_floor_stats`.
- **`ContextEmbedding(sigma, band_codes, n_bands, spec='mlp', latent_dim=32)`** — *class*  
  Concatenate constant per-point context (sigma, band one-hot) to the varying flux channel.
- **`encode_bands(band_array, labels=('u', 'g', 'r', 'i', 'z', 'y'))`**  
  Map band labels to integer codes against a FIXED label list.

### GPU-side prior and likelihood  (`whisper_cbpf.priors`, `whisper_cbpf.likelihood`)

The JAX twins of `Prior.log_prob` and `GaussianLikelihood.log_likelihood`. With a model's
`predict_jax` (above) these are the three pieces of a differentiable log-posterior, so a gradient
sampler no longer needs a hand-written density per model. Both resolve lazily — a CPU-only install
never imports JAX.

- **`priors.log_prob_jax(prior, names=None)`**
  Build `f(theta) -> scalar`, the log prior density at a flat parameter vector ordered by `names`
  (default `prior.names` — pass it explicitly, insertion order is not a contract). Same value as
  `Prior.log_prob`, normalising constants included, so **do not** add it to `nuts_gpu`'s
  `log_prob_fn`, which already gets the prior from NumPyro's sample sites. `-inf` outside the
  support with a *finite* gradient there. Takes `Uniform`, `LogUniform`, `Normal`,
  `TruncatedNormal` and `Fixed`. **`priors.ppf_jax(dist)`** is the traceable inverse CDF of a
  `Normal` or `TruncatedNormal`, for exact draws from U(0, 1) on any device.
- **`likelihood.log_likelihood_jax(like)`**
  Build the JAX twin of `like.log_likelihood` from an already-constructed likelihood **object**,
  reading `y`, `sigma`, `space`, the detection mask and the normalising constants off it so the two
  arms cannot disagree about the fit. Not jitted: jit the assembled log-posterior instead. Dispatch
  is on the **exact** type — a user subclass is refused by name rather than silently scored under
  its base class's density.

  | numpy class | JAX | signature |
  |---|---|---|
  | `GaussianLikelihood` | flux, magnitude | `f(model_flux)` |
  | `GaussianLikelihoodWithUpperLimits` | flux | `f(model_flux)` |
  | `GaussianLikelihoodWithScatter` | flux, magnitude | `f(model_flux, sigma_extra=0.0)` |
  | `MixtureGaussianLikelihood` | — | refused: a deliberate non-goal, not an omission |

  The returned function carries `.scatter_param` — the prior parameter that supplies `sigma_extra`,
  or `None`. Agreement with the numpy half, measured over 13 cases spanning both spaces, all three
  classes, both empty-half edge cases of the censoring term and several `sigma_extra`: worst
  relative difference **8.96e-14** with `jax_enable_x64`, **3.69e-5** without — and in float32 the
  worst cases are all magnitude space, where the limiting step is the float32 `flux → mag` `log10`
  map, not the density (flux space stays at 2.2e-7).

  The censoring term uses `ndtr`, not `0.5(1 + erf(z/√2))`, on **both** backends. The `erf` form
  cancels catastrophically in the left tail — which is precisely where this term lives — and in
  float64 `erf` saturates at exactly −1 for `z ≲ −8.3`, flooring every such non-detection at the
  `_MIN_PROB` clip. That gave a flat −69.0776 plateau with zero gradient across the whole range a
  gradient sampler must cross to pull an over-bright model back under a limit; measured on one
  light curve, `log P` at `z = −8.94` was −69.0776 against a true −43.0721.

```python
ll = wp.likelihood.log_likelihood_jax(wp.GaussianLikelihood(lc, space="flux"))
lp = wp.priors.log_prob_jax(model.default_prior, model.parameters)
bidx = jnp.asarray(model.predict_jax.band_index(np.asarray(lc.band)))
log_posterior = jax.jit(lambda th: ll(model.predict_jax(th, times, bidx)) + lp(th))
```

A sampler that probes far outside the prior box must clamp `theta` into it before calling
`predict_jax`: the prior term is `-inf` there, but the *model* can be non-finite, and
`nan + -inf = nan` because `jnp.where` picks its branch only after computing both.

### GPU-side metrics  (`whisper_cbpf.metrics`)

- **`ess_by_parameter(samples_by_chain, param_names, method='bulk')`**  
  Per-parameter ESS from draws shaped ``(n_chains, n_draws, n_params)``.
- **`ess_summary(samples_by_chain, param_names, wall_clock_s, method='bulk')`**  
  ESS per parameter plus the worst-parameter ESS and ESS/sec.

### JAX model modules

Physics implementations. Bind them to a dataset with the factories above rather than calling
them directly unless you know what you are doing.

**Kilonova (one component)** — `whisper_cbpf.models.jax.kilonova` (15 public functions)

- `ab_magnitude(*args, **kwargs)` — AB magnitude in the band given by band_idx, for each observation.
- `ab_weights(lam, trans)` — Photon-counting AB weights: w = T(lam)/lam * dlam, norm = 3631Jy*sum(w).
- `bolometric(*args, **kwargs)` — Bolometric luminosity, temperature, photosphere radius at t_src_s.
- `bolometric_grid(grid_s, row, mej, vej, kappa, temperature_floor=4000.0, **kw)` — DEPRECATED explicit spelling of the old (grid_s, row) signature.
- `build_time_grid(t_obs_days, redshift, t_min=0.01, n_dense=300)` — DEPRECATED. Kept only so old call sites keep running.
- `ccm89_a_over_ebv(lam_aa, r_v=3.1)` — Cardelli, Clayton & Mathis (1989) A(lambda)/E(B-V) = R_V*a(x) + b(x).
- `extincted_weights(weights, lam_obs_ang, redshift=0.0, ebv_mw=0.0, ebv_host=0.0, r_v_mw=3.1, r_v_host=3.1, law='f99')` — Fold a FIXED extinction into the precomputed AB weights. Zero model-runtime cost.
- `extinction_shape(lam_obs_ang, redshift=0.0, r_v=3.1, law='f99', frame='observer')` — A(lambda)/E(B-V) evaluated in the requested frame, on the OBSERVER wavelength grid.
- `f99_a_over_ebv(lam_aa, r_v=3.1)` — Fitzpatrick (1999) A(lambda)/E(B-V). JAX-native, differentiable in r_v.
- `flux_density_mjy(*args, **kwargs)` — Flux density in mJy at observer-frame frequencies nu_obs_hz.
- `make_filter_set(band_names, lam_min=1000.0, lam_max=30000.0, n_wave=5000)` — Export sncosmo bandpasses to plain arrays. Run ONCE, offline, then save. (Re-exported from `whisper_cbpf.synphot.grid_rule`, like `ab_weights`; the factories default to `synphot.gauss_rule` instead, see §4b.)
- `pre_explosion(t_src_s)` — Boolean mask, True where the model returns identically zero flux.
- `redback_time_grid(t_src_s, t_min=0.01, t_max=7000000.0, dense_resolution=500)` — The source-frame grid [s] redback 1.20's ``one_component_kilonova_model`` solves on. Pass it as ``time_grid=`` to ``bolometric``/``flux_density_mjy``/``ab_magnitude``; the kilonova factories do by default (``time_grid='redback'``).
- `source_time_s(t_obs_days, redshift, t_exp_days=0.0)` — Observer-frame days -> source-frame seconds SINCE EXPLOSION.
- `thermalisation_coeffs(mej, vej)` — Barnes+2016 (a, b, d). redback/utils.py:1168.

**Kilonova (multi-component)** — `whisper_cbpf.models.jax.kilonova_two` (11 public functions)

- `ab_magnitude_multi(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, mej, vej, kappa, temperature_floor, *, mag_floor=40.0, arnett_prefactor=None, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99')` — AB magnitude of the SUMMED SED.
- `bolometric_multi(t_src_s, mej, vej, kappa, temperature_floor, arnett_prefactor=None)` — Per-component ``(L/LSCALE, T, R)``, each ``(n_comp, n_obs)``.
- `default_prior()` — redback's defaults, verbatim from ``priors/two_component_kilonova_model.prior``.
- `flux_density_mjy_multi(t_src_s, nu_obs_hz, redshift, dl_cm, mej, vej, kappa, temperature_floor, arnett_prefactor=None)` — Flux density in mJy, summed over components.
- `n_component_magnitude(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, mej, vej, kappa, temperature_floor, **kw)` — AB magnitude of the summed SED for ANY number of components.
- `t_diff_days(mej, vej, kappa)` — Diffusion timescale in days, per component.
- `three_component_magnitude(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, mej_blue, vej_blue, temperature_floor_blue, kappa_blue, mej_purple, vej_purple, temperature_floor_purple, kappa_purple, mej_red, vej_red, temperature_floor_red, kappa_red, **kw)` — Blue + purple + red, summed in FLUX before the band integral (Villar+2017 3-component).
- `two_component_flux_density(t_src_s, nu_obs_hz, redshift, dl_cm, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2)` — Flux density in mJy, summed over both components.
- `two_component_magnitude(t_src_s, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kw)` — AB magnitude of the summed two-component SED. Argument order follows redback's.
- `validity_horizon_days(*args, factor=2.66)` — Beyond this, redback's quadrature is under-resolved for at least one component.
- `villar_prior(n_components=2, with_sigma=True)` — Villar+2017 multi-component kilonova priors, all opacities free.

**Tidal disruption events** — `whisper_cbpf.models.jax.tde` (23 public functions)

- `ab_magnitude(temperature, r_photosphere, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, *, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99')` — AB magnitude per observation, from a photosphere ``(T, R)`` and a filter set.
- `analytic_fallback(time, l0, t_0)` — Bolometric luminosity: t^-5/3 fall-back with a flat plateau before ``t_0``.
- `build_interaction_grid(time, dense_times)` — Setup-time (NumPy) companion to :func:`diffusion` / :func:`viscous`.
- `calc_tfb(binding_energy_const, mbh_6, stellar_mass)` — Fall-back time of the most tightly bound debris, in SECONDS. Pure arithmetic.
- `cocoon_photosphere(time, luminosity, t_thin, vej, nn)` — redback ``photosphere.CocoonPhotosphere``. time days, luminosity erg/s, vej km/s.
- `default_n_time()` — ``n_time`` of the INSTALLED redback's grid: the default of every function here (500 for 1.15/1.20 and without redback, 5000 for 1.12).
- `cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, *, f_debris=1.0, t_0_init=1.0, binding_energy_const=0.8, zeta=2.0, hoverR=0.3, n_time=None)` — Sarin & Metzger (2024) cooling-envelope TDE engine.
- `cooling_envelope_ab_magnitude(t_obs_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, mbh_6, stellar_mass, eta, alpha, beta, *, n_time=None, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', **kw)` — redback ``cooling_envelope(..., output_format='magnitude')``, AB magnitudes.
- `cooling_envelope_flux_density(t_obs_days, nu_obs_hz, redshift, dl_cm, mbh_6, stellar_mass, eta, alpha, beta, *, n_time=None, dilation=True, **kw)` — redback ``cooling_envelope(..., output_format='flux_density')``, in mJy.
- `default_prior(model_name='cooling_envelope')` — ``(Prior, pinned)`` for a TDE model: redback's, read from redback where possible.
- `default_prior_gaussianrise()` — ``(Prior, pinned)`` for ``gaussianrise_cooling_envelope``. All seven free (1.15.1, 1.20).
- `envelope_exists(mbh_6, stellar_mass, beta, binding_energy_const=0.8)` — True where the envelope is born OUTSIDE the circularisation radius, i.e. exists at all.
- `exponential_powerlaw(time, a_1, alpha_1, alpha_2, tpeak)` — a_1 (1 - exp(-t/tpeak))^alpha_1 (t/tpeak)^-alpha_2.
- `fallback_prior(model_name)` — :func:`redback_prior` without redback: the latest release's (1.20) files, transcribed. Same return shape; see _FALLBACK.
- `flux_density_mjy(temperature, r_photosphere, nu_obs_hz, redshift, dl_cm, *, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99')` — Flux density in mJy from a photosphere ``(T, R)`` at observer frequencies.
- `gaussian_rise(time, a_1, peak_time, sigma_t)` — a_1 * exp(-(t - t_peak)^2 / 2 sigma^2). `time` and the two scales share units.
- `gaussianrise_cooling_envelope_ab_magnitude(t_obs_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, peak_time, sigma_t, mbh_6, stellar_mass, eta, alpha, beta, *, xi=1.0, n_pre=100, n_time=None, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', **kw)` — Gaussian rise stitched onto the cooling envelope, AB magnitudes per observation.
- `gaussianrise_cooling_envelope_flux_density(t_obs_days, nu_obs_hz, redshift, dl_cm, peak_time, sigma_t, mbh_6, stellar_mass, eta, alpha, beta, *, xi=1.0, n_pre=200, n_time=None, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', **kw)` — Gaussian rise stitched onto the cooling envelope, flux density in mJy.
- `redback_prior(model_name, *, drop=('redshift',))` — redback's OWN prior for ``model_name``, read from redback, not transcribed.
- `rise_peaks_near_fallback(peak_time, sigma_t, mbh_6, stellar_mass, *, n_sigma=3.0, xi=1.0, binding_energy_const=0.8)` — True where the Gaussian rise peaks within ``n_sigma`` of the stitch point.
- `ryu_f_debris(mbh_6, stellar_mass, beta)` — Ryu et al. partial-disruption debris fraction, clipped to [0, 1].
- `tde_photosphere(time, luminosity, mass_bh, mass_star, star_radius, tpeak, beta, rph_0, lphoto)` — redback ``photosphere.TDEPhotosphere``. Photosphere that expands as a power of Mdot.
- `temperature_floor_photosphere(time, luminosity, vej, temperature_floor)` — redback ``photosphere.TemperatureFloor``. time days, luminosity erg/s, vej km/s.

**Supernovae** — `whisper_cbpf.models.jax.supernova` (24 public functions)

- `ab_magnitude_of(model, grid, params, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, *, magnetar_convention='1.15', interaction=True, **kw)` — AB magnitude per observation for any model in the family. ``model`` is static.
- `basic_magnetar(time_s, p0, bp, mass_ns, theta_pb, convention='1.15')` — Dipole spin-down luminosity, erg/s. ``time_s`` SECONDS, ``p0`` ms, ``bp`` 1e14 G.
- `bolometric(model, grid, params, *, magnetar_convention='1.15', interaction=True)` — Bolometric luminosity in erg/s for any model in the family. ``model`` is static.
- `build_sn_grid(time_days, dense_resolution=1000, t_pad=100.0, spacing='geometric', csm_interp=True)` — Setup-time (NumPy) companion to every model here. Build once per dataset, reuse. redback 1.20's grids by default; `spacing='linear'` for redback <= 1.15 (`REDBACK_GRID_PRESETS`).
- `cutoff_norm(luminosity, temperature, r_photosphere, cutoff_wavelength_ang)` — redback ``CutoffBlackbody._set_norm``. Dimensionless renormalisation, per epoch.
- `cutoff_shape(lam_obs_ang, redshift=0.0, cutoff_wavelength_ang=3000.0)` — ``min(lam_source/lam_cut, 1)`` on the OBSERVER wavelength grid. Setup-time-able.
- `default_prior(model)` — ``(Prior, pinned, constraints)``: redback's, read from redback where possible.
- `dense_grid(time_days, dense_resolution=1000, t_pad=100.0, spacing='geometric')` — The NumPy ``ip.Diffusion`` dense grid in days (redback 1.20's geometric, or ``spacing='linear'`` for <= 1.15).
- `exponential_powerlaw_engine(time_days, lbol_0, alpha_1, alpha_2, tpeak_d)` — Phenomenological rise-and-decay, erg/s. ``time`` and ``tpeak`` share units (days).
- `exponential_powerlaw_integrable(alpha_1, alpha_2)` — True where the diffusion integral of this engine EXISTS. A prior condition, not a guard.
- `fallback_lbol(time_days, logl1, tr)` — t^-5/3 fallback with a flat plateau before ``tr``. Both times in DAYS, erg/s.
- `fallback_prior(model)` — :func:`redback_prior` without redback. Same three-tuple return shape; see ``_FALLBACK``.
- `flux_density(model, grid, params, nu_obs_hz, redshift, dl_cm, *, magnetar_convention='1.15', interaction=True, **kw)` — Flux density in mJy for any model in the family. ``model`` is static.
- `line_band_term(lam_obs_ang, weights, redshift=0.0, line_wavelength=7500.0, line_width=500.0)` — Per-band wavelength integral of the additive line profile, ``(n_band,)``.
- `magnetar_only(time_s, l0, tau_s, nn)` — Generalised magnetar spin-down, erg/s. ``time_s`` and ``tau_s`` in SECONDS.
- `model_names()` — The eleven (twelve, counting ``slsn`` and ``basic_magnetar_powered`` separately) names.
- `nickelcobalt_engine(time_days, f_nickel, mej)` — Ni56 -> Co56 -> Fe56 radioactive heating, erg/s. ``time`` days, ``mej`` Msun.
- `photosphere(grid, lbol, vej, temperature_floor)` — ``photosphere.TemperatureFloor`` on this module's time grid. Returns ``(T [K], R [cm])``.
- `redback_prior(model, *, drop=('redshift',))` — redback's OWN prior for ``model``, READ FROM redback rather than transcribed.
- `shock_cooling(time_s, mass, radius, energy, nn=10.0, delta=1.1)` — Piro+2021 shock cooling. ``time_s`` SECONDS, ``mass`` Msun, ``radius``/``energy`` cgs.
- `sn_ab_magnitude(grid, lbol, vej, temperature_floor, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, *, sed_kind='blackbody', cutoff_wavelength=3000.0, line_wavelength=7500.0, line_width=500.0, line_time=50.0, line_duration=25.0, line_amplitude=0.3, pp=3.0, nu_max=1000000000.0, source_radius=10000000000000.0, f0=1e-26, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99')` — AB magnitude per observation from a bolometric light curve.
- `sn_flux_density(grid, lbol, vej, temperature_floor, nu_obs_hz, redshift, dl_cm, *, sed_kind='blackbody', cutoff_wavelength=3000.0, line_wavelength=7500.0, line_width=500.0, line_time=50.0, line_duration=25.0, line_amplitude=0.3, pp=3.0, nu_max=1000000000.0, source_radius=10000000000000.0, f0=1e-26, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99')` — Flux density in mJy from a bolometric light curve. ``nu_obs_hz`` OBSERVER frame.
- `synchrotron_f_nu(nu_src_hz, pp, nu_max=1000000000.0, source_radius=10000000000000.0, f0=1e-26, dl_cm=1.0)` — redback ``sed.Synchrotron`` as a flux density in erg/s/cm^2/Hz.
- `warn_if_spin_down_unresolved(dense_times, p0, bp, mass_ns, theta_pb, convention='1.15', limit=0.1)` — Warn when the magnetar spin-down is too fast for the dense grid. Host-side NumPy; returns the fraction of ``E_rot`` released before the grid's first resolved node.

**Flare (JAX)** — `whisper_cbpf.models.jax.flare` (6 public functions)

- `get_model()` — A ``whisper_cbpf.models.Model`` instance for this model — NOT registered in whisper_cbpf's
- `make_log_prob_jax(times, values, sigmas, prior_bounds)` — Build a ``jax.jit``-compiled ``log_prob(theta_vec) -> scalar`` log-posterior.
- `predict_jax(params, times)` — Batched/vmapped flux.
- `predict_numpy(parameters, times, bands=None)` — whisper_cbpf's ``Model.predict(parameters: dict, times, bands) -> flux`` contract.
- `predict_torch(theta_batch, times, device='cuda')` — GPU-batched simulator for SNPE: ``(B, 4)`` torch params -> ``(B, n)`` torch flux.
- `prior_bounds_flat()` — ``[(low, high), ...]`` in PARAMETERS order (from the shared JAX-free spec).

