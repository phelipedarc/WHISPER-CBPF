# Changelog

All notable changes to WHISPER-CBPF are recorded here, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Where a fix was measured, the entry quotes
before → after on the case named. Changes before 0.1.1 were not recorded.

## [0.2.0] - 2026-09-28

From a survey alert to a ranked, checked and explained answer: survey ingest presets, a one-call
model comparison at optimised likelihood maxima, one convergence report for every sampler, forecasts,
a facts file and a one-page HTML report, and batched fitting for an LSST night. Read **Changed**
before upgrading, because these defaults move results: `wp.fit` chooses its sampler
(`sampler="auto"`, it was ABC); upper limits are fitted; rows at or before the explosion are not;
CPU `mcmc` starts from a prior scan; `emcee_jax`, `mcmc` and `fit_batch` move LogUniform parameters
in log space; and `nested` renormalises the prior of a model behind a constraint wall. Read
**Known issues** before relying on the supernova and TDE error bars, or on the default prior for an
unknown explosion date.

### Added

#### From an alert to an answer
- **Survey ingest presets:** `load_lightcurve(x, survey="lsst" | "ztf", limits=...)` reads Rubin
  alert packets, ZTF alert packets, ALeRCE and Fink (CSV or JSON) in their own field names, and
  returns difference-imaging AB magnitudes with survey band labels (`lsstu`…`lssty`, `ztfg ztfr
  ztfi`), sorted by time.
  - LSST: diaSource `psfFlux`/`psfFluxErr` in nJy as `31.4 - 2.5 log10 f`; forced photometry as
    5-sigma limits at `31.4 - 2.5 log10(5 psfFluxErr)`. ZTF: `magpsf`/`sigmapsf` with positive
    `isdiffpos`; `diffmaglim` non-detections as 5-sigma limits.
  - `scienceFlux` and `magpsf_corr` are never used; the loader raises when one is the only
    measurement.
  - Negative, flagged, withdrawn and repeated rows, and limits at a detection's epoch, are dropped
    and counted in `lc.meta["n_dropped"]`; `min_snr` cuts detections only;
    `lc.meta["first_detection_mjd"]` is recorded; no detection left raises "not enough data".
  - On five real alerts (four ZTF, one LSST) it reproduces all 28 hand-built decision-point light
    curves row for row from the raw broker data, in about 10 ms per object.
  - `whisper_cbpf.io.SURVEY_BANDS` and `whisper_cbpf.io.surveys` (`survey_table`, `survey_band`,
    `check_survey`). `load_lightcurve` also takes a DataFrame or a list of records.
- **`wp.compare(lc, models, ...)`** and `wp.Comparison`: fits every model to the same light curve,
  finds each fit's likelihood maximum, attaches its convergence report, and ranks by ln Z when every
  ranked model has a converged one, by BIC at the likelihood maximum otherwise, with dBIC or evidence
  weights and a Jeffreys grade.
  - Models left out are listed with the reason: a failed fit, no draws, another number of points,
    `k >= n` ("not enough data"), no finite BIC.
  - `evidence_check="auto"` runs nested sampling on the top two when their BIC gap is under 4.61,
    and grades the winner "inconclusive" when ln Z disagrees.
  - Family names (`arnett`, `magnetar`, `csm_shock_arnett`, `shock_cooling_arnett`, `tde`, any
    supernova model) are bound to the light curve's bands, redshift (fixed if known, fitted
    otherwise) and explosion-time window.
  - `summary()`, `save()` / `load()`, `facts()`, `report()`, `forecast()`, `discriminate()`;
    `cache_dir=` resumes an interrupted comparison.
  - It reproduces the hand-built rankings of 60 decision points exactly from their own numbers.
    Refitted at a reduced budget on one GPU, it finds the same winner on all 5 alerts and the same
    order on 4 of 5.
- **`wp.likelihood_max_opt(result, lc)`** (maximum-likelihood optimisation) returns
  `wp.LikelihoodMaxOptResult`: the peak of the fit's own likelihood, the gain over the sampler's
  best draw, the parameters on a prior edge, and AIC/BIC from the peak. The posterior (samples,
  medians, error bars) is not changed, and the peak is never below the best draw. It climbs in each
  prior's own coordinate (log10 for LogUniform, standardised for Normal), so an optimum on a prior
  edge is reached exactly. Also `wp.fit(..., likelihood_max_opt=True)` and
  `result.likelihood_max_opt(lc)`, which keep the peak in `info["likelihood_max_opt"]`.
  - redback `arnett`, SN2025pgp 30-day cut: a 32 x 600 `emcee_jax` run's best draw, ln L 28.80,
    climbs to 31.49 (4.1 s on one GPU), the peak a 1.8 million-evaluation CPU chain approaches
    (31.38 → 31.49). redback scores that peak the same to 4e-14.
  - Compiled programs are reused between optimisations of the same density: 6.8 → 2.6 s on
    `arnett`.
- **`result.diagnostics()`** returns `wp.DiagnosticsReport`: one convergence report for every
  sampler, each check with its value, threshold, pass or fail and a plain reason.
  - NUTS and PyMC: divergences, R-hat, bulk and tail ESS, log-likelihood R-hat, stranded and frozen
    chains, gap to the prior-scan optimum. emcee: stuck walkers, N/tau from the largest tau, R-hat
    and ESS across walkers. ABC: accepted draws. ABC-SMC: distinct particles. Nested: dlogz stop and
    effective sample size. SNPE: fallback. Every fit: no draws, `n <= k`, prior-edge pile-up.
  - `likelihood_max_opt=` adds a check that the best draw is within 5 nats of the likelihood
    maximum.
- **Save, reload, resume:** `result.save(path)` and `wp.load_result(path)` (a directory or one
  `.npz`) round-trip a fit exactly, draws by chain included. `SamplerResult.provenance` records how
  every fit was made (versions, git state, model and prior, sampler settings and seed, devices,
  timing), with no argument. Loading checks hashes and refuses a modified save.
  `wp.fit_cached(lc, model, sampler, cache_dir, **kw)` loads a fit whose configuration was already
  saved: a 6-job batch killed after 2 fits finished only the rest when run again.
- **Forecasts:** `wp.forecast(result, times, bands, ...)` and `result.forecast(...)`: predicted AB
  magnitudes per (time, band) cell, with the mean, spread, percentiles, the fraction of draws too
  faint for a survey depth and the fraction dark. A JAX model runs as one compiled batch: 400 draws
  x 15 cells in about 1 ms after a 1.5 s compile on one GPU, equal to redback's CPU forecast to
  round-off.
- `wp.discriminate(results, times, bands, ...)`: for every pair of models and cell,
  `D = |dmean| / sqrt(sd_a^2 + sd_b^2 + sd_phot^2)`, with an observable flag and the best cell in
  `.attrs["best"]`.
- **Plot kit:** `wp.plot_forecast`, `wp.plot_models` (every model over the data, with residuals,
  pre-event rows marked), `wp.plot_model_comparison` (weights, gap and grade, and their evolution
  with `history=`), `wp.plot_widths` and `plotting.posterior_width_ratios` (posterior width over
  prior width), `plot_ppc(..., intervals=)`.
- **Facts and report:** `wp.result_facts` / `SamplerResult.facts(lc)`, `wp.comparison_facts` /
  `Comparison.facts()` and `wp.write_facts`: every quoted number computed by a stated rule
  (ranking, per-parameter medians and widths with `prior_dominated` and `at_prior_edge`, redshift
  and explosion-time narrowing, rates and colours, the absolute magnitude with its asymmetric range,
  caveats as booleans), with the thresholds, the rules in words and SHA-256 hashes of every input.
  `wp.report(comparison, lc, out)` / `Comparison.report(path)` writes one self-contained HTML page
  (inline CSS, base64 figures, no script, no network, readable on a phone); the same inputs give the
  same bytes.

#### Modelling
- **Explosion time and redshift as free parameters** in `supernova_model`, `tde_model`,
  `kilonova_model`, `kilonova_two_model` and `kilonova_three_model`: `free=["t_exp", "redshift"]`.
  With the redshift fixed at the truth the free model equals the baked-in one to 4.3e-14 mag; jit
  and vmap over 200 redshifts compile once. The redshift prior defaults to `lc.redshift_prior`.
  `whisper_cbpf.models.cosmology.luminosity_distance_cm` (`wp.luminosity_distance_cm`), a Planck18
  distance JAX can trace, within 8.3e-7 mag of astropy; `io.schema.redshift_distribution`.
- **Supernova colours at the observed epochs only** (`supernova_model(..., diffusion_grid="fixed")`,
  the default when anything is free): within 8.8e-4 mag of redback over 200 prior draws per family,
  6-8x faster per evaluation on one GPU than a magnitude grid with a free redshift.
- **Pre-event data are never fitted.** Every sampler leaves the rows at or before the event out of
  the likelihood and the model evaluation, records `info["excluded_pre_event"]` and
  `info["pre_event"]`, and warns once. Adding pre-event rows changes a posterior by exactly zero.
  `whisper_cbpf.samplers.prepare_lc(lc, model)` returns the rows a fit uses;
  `SamplerResult.fitted_lc(lc)` re-applies a fit's cut, and `wp.waic(result, lc)`,
  `predictive_metrics` and `posterior_predictive_check` score the fit's own rows. `log_density`
  and `fit_batch` apply the same rule.
- **Explosion-time prior from the data:** `LightCurve.explosion_time_prior()`, Uniform from the last
  non-detection before the first detection to the first detection (30-day fallback,
  `io.schema.EXPLOSION_FALLBACK_DAYS`). A fit of a model with a free `t_exp` uses it when its prior
  names none; `info["t_exp_prior"]` records the prior used.
- **Priors beyond boxes:** `wp.Normal`, `wp.TruncatedNormal` (exact density, inverse-CDF draws) and
  `wp.Fixed`, in every sampler; `Prior.fixed`, `priors.ppf_jax`. NUTS and PyMC sample a
  TruncatedNormal through its CDF; the GPU ABC samplers draw by the probability integral transform;
  SNPE uses an exact torch TruncatedNormal. A `Fixed` parameter is held at its value and not counted
  in AIC/BIC (`info["fixed"]`).
- **Where chains start:** `init=` on `mcmc`, `emcee_jax`, `nuts_gpu` and `pymc_jax_gpu_*`:
  `"prior_scan"` (default), `"prior"`, a point, `(point, scale)`, one start per chain or walker, or
  a previous result, such as an ABC fit or an alert's earlier cut. With fewer usable draws than
  walkers the start is a ball around the result's likelihood maximum. Every start is checked against the
  box, the constraint wall and a finite density; `info["init"]` and `info["init_detail"]` record it.

#### Inference and speed
- ABC worst-point distance `distance="max_abs_z"` (`wp.max_abs_z_distance`; `max_abs_z_jax` on the
  GPU): a draw is accepted only if every point is within k sigma of its own error.
- `abc_gpu(..., precision=None | "float32" | "float64")`: single precision for the kilonovae and
  the flare (an alert 9.7 → 6.6 s); the float64-only supernova and TDE engines are refused by name.
- **`wp.log_density(lc, model)`**: the log-posterior of one light curve as a JAX function, with the
  data as arguments of one compiled program, padded to length buckets (16, 24, 32, 48, 64, 96, ...),
  so alerts of one model and bucket share one compile. It equals the density `emcee_jax` samples
  to 1e-12.
- **`wp.fit_batch(lcs, model, ...)`**: one emcee ensemble per light curve in one compiled on-device
  loop; an alert fitted in a batch equals the same alert fitted alone. Each result carries emcee's
  diagnostics, with frozen walkers named in `info["frozen_walkers"]`. On one A6000, float64, per
  alert at 512 alerts per call (60 x 10 000): free `arnett` 5.3 s + 0.3 s start, TDE 3.3 s + 0.25 s.
- `wp.profile(model, lc)`: compile and call time per evaluation by batch size, value and gradient,
  memory, and the density's and gradient's health. `wp.capacity(cost, hours=12, n_gpus=1)`: alerts
  per night from a measured cost.
- **`wp.run_jobs(jobs, ...)`** and `wp.Job`: many fits, one fresh process per job, over the idle
  GPUs this process may see and pinned CPU cores; failed jobs are retried, `timeout=` stops a hung
  job, and `cache_dir=` resumes an interrupted batch.
- `wp.check_parity(model_a, model_b, params_or_posterior, times, bands)`: a per-band |Δmag| table
  and a pass or fail at a stated tolerance, for a JAX port against its redback twin or any two
  models (JAX `arnett` against the redback adapter: max 5.3e-13 mag).
- `walker_coordinates=` on `emcee_jax`, `mcmc` and `fit_batch` (see **Changed**), recorded in
  `info["walker_coordinates"]`; `info["constraint_prior"]` on a nested fit (see **Changed**);
  `LogDensity.pre_event`, the pre-event rule a density applies; `samplers.base.fitted_model(result)`,
  the Model behind a fit; `model.predict_jax.check_epochs(times, prior)` on a supernova on fixed
  diffusion epochs.

#### Package
- Every new name is exported from `whisper_cbpf` (104 names in `__all__`); `import whisper_cbpf`
  still works without jax, torch, sbi or redback.
- `docs/LSST_ALERTS.md` (the alert workflow end to end, with seconds per alert), a synthetic LSST
  alert in `tests/data`, `tests/test_docs_coverage.py` (every public name documented, the
  documentation's code parses, the README quickstart runs) and `tests/test_no_silent_failures.py`.
- `notebooks/11_lsst_alert_quickstart.ipynb`: a Rubin alert packet to a ranked comparison,
  diagnostics, a forecast, the facts file and the HTML report, checked against a known truth.
  The other notebooks show the new API where it applies: survey presets (01), the new priors (04,
  06), the diagnostics and `likelihood_max_opt` (05), `compare`, save and load (08) and the plot
  kit (10).
- The science-validation suite (`tests/science`, `docs/VALIDATION.md`): simulated LSST alerts with
  a known answer, fitted and compared as a user does it, for posterior calibration, model
  selection, the alert data rules, CPU/GPU agreement and speed budgets.
- The sdist carries `CHANGELOG.md`, and leaves out sbi's training logs (`sbi-logs/`) that SNPE
  runs write into a checkout.

### Changed
- **`wp.fit(lc, model)` defaults to `sampler="auto"`** (it was `"abc"`): `emcee_jax` for a JAX model
  when JAX sees a GPU, CPU `mcmc` otherwise, the rule `compare` uses. Both fit upper limits, so a
  survey alert needs only the data and the model name. Pass `sampler="abc"` for the old default.
- **Upper limits are fitted by default.** `space="auto"` on a light curve with upper limits after
  the event resolves to flux space with the censored likelihood; it used to refuse them. The limits'
  significance defaults to `lc.meta["upper_limit_sigma"]` (5 for the survey presets), else 5; it was
  3.
- **Rows at or before the event are no longer fitted**, so a light curve with pre-event rows gives
  a different `n_data`, AIC and BIC than in 0.1.1. A `set_time_reference(..., "peak")` clock leaves
  out nothing for `bazin`, `gaussian_rise` and `flare_jax`, which fit their own epoch.
- ABC, ABC-SMC, SNPE and their GPU versions refuse a light curve with upper limits after the event
  by name, before any simulation; they used to fail inside the likelihood.
- **CPU `mcmc` starts from `init="prior_scan"`**, climbed without a gradient, instead of independent
  prior draws (`init="prior"` reproduces the old start). redback `arnett` on SN2025pgp, 60 walkers x
  10 000 steps: stuck walkers 20 of 60 → 0, for a 26-29 s start that `runtime_s` includes.
  `fit_emcee_numpy`'s start climbs too.
- `emcee_jax(init=<result>)` with fewer usable draws than walkers starts in a ball; it used to
  raise. `mcmc(initial_guess=...)` refuses a start where the density is `-inf` or the model predicts
  no flux; such a walker used to sit unmoved.
- `nested`, `abc` and `abc_smc` no longer count a `Fixed` parameter in `n_params`, so AIC and BIC
  use the free parameters only; `abc_smc` perturbs only the free parameters.
- `SamplerResult.likelihood_max_opt` stores its peak in `info["likelihood_max_opt"]` and optimises
  on the rows and prior the fit used; `wp.likelihood_max_opt(prior=None)` uses the prior the fit
  recorded.
- `abc_gpu` evaluates each draw once (one model trace per fit): an `arnett` alert at 3e6 draws 10.2
  → 7.7 s, results unchanged in float64. `info["x64"]` records the sweep's precision
  (`info["x64_session"]` the session's). ABC-SMC's `min_epsilon="auto"` warns with a distance other
  than `chi2`, whose scale it was derived for.
- `load_lightcurve(bands=...)` spells the bands the way the band column is spelled, and raises,
  naming the available bands, when nothing matches (it returned an empty light curve).
- `plot_ppc` shades the 68 % and 95 % bands by default (`intervals=(95,)` gives the old figure);
  `plot_corner` puts parameters with a LogUniform prior on log10 axes (`log_params=None` for the old
  axes).
- The JAX factories' `redshift` and `dl_cm` default to `None` (required unless the redshift is
  free). The kilonova factories' `time_grid` defaults to `"auto"`: redback's grid with nothing free
  (results unchanged), the converged quadrature otherwise. `kilonova_model` takes `prior=`.
- A warning raised inside a sampler's `fit` points at `whisper_cbpf/samplers/base.py`, not at the
  calling line; the message is unchanged.
- **`emcee_jax`, `mcmc` and `fit_batch` move their walkers in each prior's own coordinate**
  (`walker_coordinates="own"`): the natural log of a LogUniform parameter, with the log-Jacobian
  added, so the posterior is the same; `walker_coordinates="linear"` reproduces the old moves. The
  stretch move is affine-invariant only in the coordinates it moves in. On simulated LSST alerts
  (`emcee_jax`, 32 walkers x 5 000 steps, explosion date known) the 68 % / 95 % intervals hold the
  truth: Arnett 53 / 73 % → 62 / 87 %, shock cooling + Arnett 60 / 85 → 65 / 88, CSM + Arnett
  49 / 72 → 63 / 89, TDE 55 / 79 → 71 / 87, magnetar 47 / 66 → 48 / 73; the kilonovae stay
  calibrated (74 / 93 and 74 / 95). The stuck-walker check now compares the
  walkers' log posterior in these coordinates: a flat LogUniform direction was reported as stuck
  walkers (8-26 of 32 on a toy with two unconstrained LogUniform parameters), with advice to use
  the start that was already the default.
- **ln Z of a model behind a constraint wall** (the JAX supernova and TDE families, the redback
  adapter) is computed with its prior renormalised to the allowed region: `nested` treats a walled
  draw as `-inf` (it scored the zero-flux model) and adds `-ln f`, `f` the allowed fraction of
  20 000 prior draws. At the default priors that is +0.94 for Arnett, +1.06 for the magnetar and
  +1.22 for the TDE, by which each was handicapped in `compare`'s evidence check against the
  shock-cooling and CSM families (no wall). `info["constraint_prior"]` records it.
- `load_lightcurve` reads a `jd` or `hjd` time column holding Julian dates (median above 2400000.5)
  as MJD, `time = JD - 2400000.5`, and records it in `lc.meta["time_converted"]`; the column was a
  plain time alias. `explosion_date=`, `time_min=` and `time_max=` are then on the MJD clock: a
  value in JD selects nothing.
- The redback CPU adapter predicts zero flux before the explosion (`t < 0`); it returned the flux
  at `MIN_TIME_DAY` (1e-3 d) there, which showed in plots and forecasts. Epochs from 0 to 1e-3 d
  are still clipped up to it.
- `compare` ranks only fits on the same rows scored in the same space (flux or magnitude): among
  the fits with the most common number of points, a fit to other rows or in another space is left
  out with its reason.
- `likelihood_max_opt` on a JAX density in a float32 session warns that ln L is resolved only to
  float32.
- The default start (`init="prior_scan"`) retries a refused spread move in a fresh direction and
  its mirror image, at half the length, for up to 20 rounds, and gives a coordinate walled on one
  side a short step instead of the widest one. An ensemble start that still does not span every
  parameter is refused, naming `init="prior"`, before emcee's "Initial state has a large condition
  number".
- `supernova_model(free=["t_exp"])` on fixed diffusion epochs refuses a `t_exp` prior with no finite
  lower bound (a `Normal`), since the epochs are sized from the earliest explosion it allows.
- `compare`: the evidence check's reason no longer blames a fitted redshift when no model fits one;
  a caveat says when a ln Z gap between the top two is not explained by their likelihood peaks (it
  then comes from the priors); an unknown family name without JAX installed says to install it.

### Fixed
- `load_lightcurve(bands=["zg", "zr", "zi"], min_snr=3)` on ZTF25aaxwrva returned 0 of 29 points
  with no error; it returns 29.
- `LightCurve.select_snr` (and `load_lightcurve(min_snr=...)` without `survey=`) dropped every upper
  limit, whose signal-to-noise is NaN; it now cuts detections only and keeps the limits.
- A model built by a factory (every family `compare` binds) had no per-band or posterior-predictive
  metrics, and each fit warned "Unknown model" twice; the metrics are now computed with the model
  object.
- `plot_ppc` with a model that is not registered said only `KeyError`; it says to pass `model=`.
- `abc_smc_gpu`: the `chunk` docstring said the default is 16; it is 250.
- `compare(prior={family: Prior({"t_exp": ...})})` for a family named by string was cut back to the
  data window by the fit (the draws of the bundled alert stayed within 61000.66-61000.68); the
  explosion-time prior given is now passed to the fit as given.
- A family bound by `compare` with an explosion-time prior with no upper bound (a `Normal`) was
  shifted by an infinite reference and predicted the magnitude floor at every epoch: `mcmc` ran to
  max ln L = -3.4e6 with no error. The reference is now the first detection, and a supernova
  family refuses a prior with no lower bound.
- `compare` built its supernova families on diffusion epochs sized to 200 d: on a light curve
  reaching further after the earliest allowed explosion (an Arnett alert followed for 298 d) every
  supernova family failed with "Rebuild it with max_phase_days=..." and was left out. They are now
  sized to the light curve.
- `log_density` (and so `fit_batch`, the JAX `likelihood_max_opt` and `profile`) passes the epochs to a
  supernova on fixed diffusion epochs traced, where the phase check cannot run, and past the last
  epoch the luminosity is held: Arnett at 320 d was 0.375 mag too bright with no error. The check
  now runs on the host before the density is built.
- `emcee_jax`'s default start put every walker on one point in 27 of 503 fits of the validation
  suite, and emcee refused to run; `compare` then left the model out (the true model, in 2 of 8
  TDE alerts). The two reproducers (a TDE and a magnetar injection) now start and run, and none of
  290 fits of the recovery studies failed to start.
- A numpy "overflow encountered in power" warning reached the user from the coordinate maps of the
  start and of `likelihood_max_opt` on an MJD-valued explosion time.

### Known issues
- The default explosion-time prior when the explosion date is unknown (last non-detection to first
  detection, any band and depth) excludes the true explosion on 24 of 30 simulated LSST alerts;
  pass a wider `t_exp` prior (`docs/VALIDATION.md` section 4.2).
- The supernova and TDE 95 % intervals hold the truth 87-89 % of the time on simulated LSST alerts
  at the default settings (the magnetar's 73 %, its 68 % intervals 48 %); 4 times longer chains
  move Arnett from 87 % to 88 % (`docs/VALIDATION.md` section 4).
- Two `emcee_jax` fits at the same seed on a GPU differ (the start's gradient climb uses
  non-deterministic GPU reductions); `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes them
  identical at about 35 % more time.
- The evidence check in `compare` is CPU nested sampling: 8-12 minutes per supernova family on a
  34-point alert. Pass `evidence_check=False` or `cache_dir=` in a stream.
- `likelihood_max_opt` compiles its programs for every new alert, about half of a GPU comparison's
  wall time.
- `fit_batch` is emcee only.
- LSST times stay on TAI (`midpointMjdTai`), 37 s ahead of ZTF's UTC.
- `abc_smc_gpu` scores its final population in a second compiled pass and has no `precision=`.

## [0.1.1] - 2026-09-26

A correctness release. The CPU and GPU paths now compute the same band magnitude, the redback
models follow redback 1.20 (and refuse or flag what redback gets wrong), and several samplers that
reported confident, wrong or empty diagnostics now say what went wrong. Read the first section
before upgrading: some defaults changed, and some results change with them.

### Breaking and behaviour changes

- **Plots return their Axes, not the Figure** (`plot_light_curve`, `plot_ppc`, `plot_calibration`,
  `plot_corner`), so a bare last-line call in a notebook shows one image, not two. The figure is
  still open in pyplot: use `np.ravel(axes)[0].figure.savefig(...)`, or pass `save=`.
- **Bare `u g r i z y` mean LSST in every model**, with one warning per session. The redback
  adapter used to read them as SDSS (`y` as PS1). Use `wp.set_default_band_system("sdss")`
  (mirrored in `$WHISPER_BAND_SYSTEM` for worker processes), or `default_system=` / `band_aliases=`
  per model. AT2017GFO's g, r, i are SDSS: pass `default_system="sdss"`.
- **A grouped label such as `g-band` raises** in the band-integrating models. It used to be modelled
  as SDSS g, even for ZTF g data.
- **Photometry is integrated over the filter by default**, on the CPU as on the GPU. The redback
  adapter used to take redback's SED at one reference frequency per band;
  `photometry="monochromatic"` keeps that.
- **redback's Constraint priors are enforced by default**: a draw that breaks one predicts zero flux
  (CPU adapter, JAX supernova and TDE `predict`) or has log-density `-inf` (JAX samplers). The
  default `constraint="corrected"` bounds the Arnett kinetic energy by the physical nuclear-burning
  energy, 1.51e18 erg per gram of Ni-56; redback uses 1.91e19, 12.6× too lenient.
  `constraint="redback"` reproduces redback's decisions exactly; `constraint=None` is 0.1.0.
- **The JAX supernova and TDE `predict` returns zeros at unphysical draws**, for the same reason:
  `abc`, `abc_smc`, `mcmc`, `nested` and `snpe` on a JAX model no longer sample the unconstrained
  prior, and a direct call at such a draw (a plot at a prior's midpoint, say) returns zeros. Pass
  `constraint=None` for the raw physics; `predict_jax` is unchanged.
- **`nuts_gpu` and `pymc_jax_gpu_*` start from `init_strategy="prior_scan"`, and `converged` is
  stricter**: no divergences, R-hat < 1.01 on every parameter and on the log-likelihood, ESS ≥ 100
  per chain, no stranded or frozen chain, and no prior-scan optimum more than 10 nats above every
  chain. `info["convergence_problems"]` says which check failed. `init_strategy="uniform"`
  (`"jitter"` for PyMC) restores the old start.
- **`emcee_jax` / `fit_emcee_numpy` start from `init="prior_scan"`** (`init="box"` is the old
  start), and they and CPU `mcmc` no longer call a run with stuck walkers converged; the
  autocorrelation rule uses the largest τ, not the mean. Some fits that read `converged=True` now
  read `False`.
- **`snpe_gpu` / `fit_snpe_gpu` default to `num_rounds=2`** (was 1), as `snpe` does: a default call
  is now sequential SNPE with twice the simulations. `num_rounds=1` is the old amortised run.
- **`snpe_gpu` records `info["embedding_net"]`, not `info["embedding"]`**; code reading the old key
  gets a KeyError.
- **CPU `snpe`: same seed, different (still reproducible) answer** when the final draw falls back to
  MCMC, because the fallback is now seeded from `seed`.
- **The Gaussian-rise TDE's rise is normalised to the envelope's initial photosphere** where the
  envelope integration ends after its first step, as redback does. 0.1.0 required two live
  envelope samples and returned the magnitude floor at every epoch on those draws.
- **The JAX TDE's default `n_time` follows the installed redback**: 500 for redback 1.15 and 1.20
  (and without redback), 5000 for 1.12. Pass `n_time=5000` for the finer grid.
- **The JAX supernova models use redback 1.20's geometric diffusion grid**, and the CSM breakout
  interpolates redback's 300 nodes. `spacing="linear"` (or `**REDBACK_GRID_PRESETS["1.15"]`) and
  `csm_interp=False` restore 0.1.0.
- **The JAX kilonova factories solve on redback's own 500-node time grid** (`time_grid="redback"`),
  which is too bright late on, as redback is: up to 1.19 mag at 20 d. `time_grid=None` is the
  converged quadrature of 0.1.0; use it for late epochs. The grid is built from concrete times, so
  `predict_jax` of `kilonova_model`, `kilonova_two_model` and `kilonova_three_model` now raises a
  TypeError when the times are traced (`jit` or `vmap` over them, `grad` with respect to a time
  shift or an explosion time), which 0.1.0 accepted: pass `time_grid=None` to trace times.
- **`nested` and CPU `mcmc` with `n_jobs > 1` start workers with `spawn`**, as ABC already did.
  Scripts need an `if __name__ == "__main__":` guard, and a `predict` defined in a notebook fails
  in the pool (BrokenProcessPool): move it into a module.
- **`nested`'s default `sample="auto"` is dynesty's `"rwalk"`** up to 20 parameters (`"rslice"`
  above), not `"unif"`. `info["sample"]` records the method that ran.
- **New `[analysis]` extra** (`arviz`, `astroquery`), for PSIS-LOO, ESS and SVO wavelength search
  on a CPU-only install. Both left `[gpu]` (which still installs arviz through pymc); `pyphot`
  joined `[models]`.

### Fixed

#### Photometry and bands
- The redback CPU adapter's band magnitudes against a fine band integral of redback's own SED:
  4–39 mmag median, 17–653 mmag max → ≤ 8e-6 mag (the one exception, `type_1a` in LSST g at
  z = 0.35, is 3.6e-5 mag, from redback's cutoff-blackbody kink). CPU minus GPU for the same
  supernova model: ~10 mmag median, up to 0.55 mag → ≤ 4e-12 mag.
- `mck19` integrates its disk and hotspot blackbodies over the filter (g peak at the reference point
  25.900 → 25.842 mag), and an unresolvable band raises; it used to be evaluated at 6000 Å silently.
- `io.bands`: `lsstu` resolves (it did not), and `y` / `lssty` have their own y band at 9710 Å
  (they were folded into z at 8679 Å).
- The redback adapter's band and filter tables and its installed-redback check no longer come back
  empty when a redback clone's folder is on `sys.path` (redback then imports as a namespace
  package).

#### redback adapter
- redback's `gaussianrise_cooling_envelope`, `bpl_cooling_envelope` and `stream_stream_tde` dilate
  time twice. The adapter calls them at `t (1+z)`, detected from redback's source, so the shim turns
  itself off once redback is fixed. CPU minus the JAX port: 1.0 / 2.2 mag max (ZTF / LSST) →
  1.9e-5 / 3.0e-5 mag median; the remaining tail, up to 0.11 mag, is the port's own residual.
- `two_component_kilonova` is the band-integrated sum of two one-component redback calls. Past 6
  days it returned 99 mag (4e-37 Jy); it is now physical and continuous. Inside 6 days it agrees
  with redback's two-component model to 1.3–2.0 mmag. A 200-observation predict: 177 → 8.7 ms.
- Epochs redback cannot compute are no longer silent. A domain limit every draw shares raises,
  naming the span, at the first `predict` (or at registration with the new `times=`): redback's
  `two_component_kilonova_model` on AT2017GFO returned 0 Jy for the 19 of 213 points past 5.49 d.
  Limits that move with the parameters still give zero flux for that draw, and warn once per model
  and span. Any exception inside redback's function rejects the draw instead of aborting the fit.
- Bolometric engines are refused. `register_redback("shock_cooling_and_arnett_bolometric")` fitted
  erg/s as Jy (median 1.2e37 "Jy"); it now raises and names the photometric wrapper
  `shock_cooling_and_arnett`. 55 `*_bolometric` engines and 23 frequency-independent functions are
  refused, and `redshift=` is checked like `pin=`.
- whisper's warnings survive `import redback`. redback 1.20 installs a process-wide
  `warnings.simplefilter("ignore")`, which hid every whisper warning once a redback model was bound:
  the SNPE fallback notice appeared in 0 of 46 run logs although 39 runs fell back.
- `register_redback(..., pin={"t0": mjd})` explains that redback's times are days since the
  model's own t = 0, not MJD, and points to `lc.set_explosion_date(mjd)`; it only said the name was
  not a parameter.
- `register_tde` raises on unknown engine keywords at registration (they failed at the first
  predict with a TypeError).

#### JAX models
- Supernova magnetar: up to 21 mag too bright against redback 1.20. Over 200 prior draws, max |Δmag|
  18.3 (ZTF) / 13.0 (LSST) → < 5e-5. Arnett through the redback adapter: 6.7e-5 → 5.4e-15 relative.
- CSM shock breakout against redback 1.20: 0.059 / 0.037 mag → < 5e-5.
- Bare cooling-envelope TDE against redback 1.20: 22.9 / 18.2 mag → < 5e-5 (the `n_time` change).
- Gaussian-rise TDE: no longer returns the magnitude floor at every epoch when the envelope
  integration dies after its first step (4.8% of redback's prior at 500 points). Against redback,
  with redback's double time dilation taken out: up to 47.7 / 60.2 mag → 0.057 / 0.108 mag.
- Kilonovae against redback 1.20: 3.2 / 3.0 mmag → 0.0 / 1.9 mmag (redback's time grid). That grid
  is built from the epochs clipped at the CPU adapter's 1e-3 d, so a pre-merger row (an upper
  limit) moves the JAX and CPU grids alike.
- A raw-MJD clock is exact in float32: every factory subtracts `t_exp_days` on the host in float64,
  and the JAX samplers hand the model float64 epochs. Two-component kilonova, float32 against
  float64: 7.3 mmag → 4.8e-6 mag, max |Δ log L| 5.2 → 0.010, median relative gradient error
  1.8e-2 → 2.7e-5.
- Precision no longer depends on import order (importing whisper before enabling float64):
  `slsn` / `type_1a` 7.6e-8 → < 1e-14 relative; kilonovae 3.6e-8 median, up to 1.9e-6 →
  bit-identical.
- `flare_jax` follows the session precision: `predict` and `predict_torch` cast to float32 in an x64
  session (2.3e-2 relative error at MJD 59000), and `flare.make_log_prob_jax` scored an MJD-clock
  flare −212.67 instead of −220.06.

#### ABC (`abc`, `abc_smc`, `abc_gpu`, `abc_smc_gpu`)
- The best fit is taken from every accepted draw (every final particle for ABC-SMC), not the first
  2000. redback `arnett`, SN2025pgp 30-day cut, 10,000 accepted draws: ln L_max −55.44 → −5.95,
  BIC 130.9 → 31.9. JAX `arnett` on `abc_gpu`: ln L_max −246.6 → −22.6, BIC 513.2 → 65.2, at the
  same wall time. `max_logl_scan=` still caps it; `info["logl_scan_n"]` and
  `info["logl_scan_capped"]` report what was scored.
- The `distance` column is no longer scored as a free scatter term: every ABC fit reported
  posterior-predictive coverage 1.00 at every level. On a redback `arnett` fit, coverage 1.00 at
  all six levels → 0.00–0.96 and WAIC 704.6 → an honestly flagged unreliable value; on a flare toy
  the 50% interval covers 0.83, not 1.00. The predictive metrics now use the scatter parameter
  each fit records (`info["scatter_param"]`), for every sampler that records one (all but CPU
  `mcmc`).
- A fit that accepts no draw says so: `info["predictive_metrics_skipped"] = "no accepted draws"`
  and `info["best_params_source"] = "closest_rejected_draw"`, instead of a "need at least one array
  to concatenate" warning.

#### Nested sampling
- The `"rwalk"` default converges where `"unif"` stalled. redback `arnett` on SN2025pgp (29 points,
  6 parameters, nlive=100): not converged after 152,423 calls and 745 s (lnZ −1929.6, n_effective 1;
  uncapped it ran 25+ minutes) → converged in 85,593 calls (225 s serial, 54–69 s with
  `n_jobs=16`), lnZ −1727.2 ± 0.6. At nlive=400, lnZ −1728.86 ± 0.41 against the reference
  −1728.73 ± 0.45. On cheap toys it takes ~4.7× the calls at about the same wall time.
- `n_jobs > 1` no longer hangs on a JAX-on-CPU model (`kilonova_one_jax`, `n_jobs=8`: no progress in
  240 s, every worker idle). Start-up costs a few seconds (flare, `n_jobs=4`: 17.8 → 28.9 s).
- An explicit `sample="unif"` run that ends below 1% efficiency warns and suggests `"rwalk"`.

#### MCMC (`mcmc`, `emcee_jax`, `fit_emcee_numpy`)
- CPU `mcmc` with `n_jobs > 1` no longer hangs after JAX has started (a forked pool sat at 0% CPU
  for 48 minutes). Draws are identical to a serial run; worker start-up is ~11 s for a cheap model.
- Stuck walkers: emcee said `converged=True` on 77 of 100 bump fits, 55 of them with stuck walkers.
  With the old start, 78/78 broken bump runs and 57/57 broken flare runs are now flagged. With the
  new start, runs with stuck walkers: bump 78 → 0 of 100, flare 57 → 0, kilonova 2 → 0 of 10.
- A log-posterior passed as `log_prob_fn` is refused for non-Uniform priors (it counted the prior
  twice, 2.5 dex off on a LogUniform), and warned about for all-Uniform priors.

#### NUTS and PyMC on the GPU (`nuts_gpu`, `pymc_jax_gpu_*`)
- R-hat and ESS work on arviz ≥ 1.0. Every run read `rhat={}`, `max_rhat=None`, `converged=False`,
  good or bad; now rank-normalised R-hat, bulk and tail ESS per parameter and the log-likelihood
  R-hat (`rhat_method` / `rhat_error` say how). A clean Gaussian-bump run reads `converged=True`
  (84 of 100 mock fits).
- Wrong answers are flagged. With the old start every broken mock run is now caught: 22/22 bump,
  50/50 bump float32-MJD, 29/29 flare float32-MJD, 12/12 PyMC, including two PyMC runs with every
  chain in the wrong mode at R-hat 1.00. The `prior_scan` start then removes most of them: bump
  22 → 0 of 100, bump float32-MJD 50 → 0, flare float32-MJD 29 → 1 (flagged), PyMC bump 12 → 0,
  kilonova 7 → 0 of 13, arnett 3 → 0 of 9.
- New warnings: a prior that is mostly a zero-signal plateau (a bump with t0 ~ U(−1000, 1000) around
  40 d of data stranded 4 of 4 seeds; all 4 are now correct), and float32 that cannot resolve the
  clock (float32-MJD fits froze chains in 19–25 of 100). `info["frozen_chains"]` and
  `info["step_size_by_chain"]` record frozen chains. Point starts warn with more than one chain.
- `pymc_jax_gpu_*` no longer switches the whole session to float64; `info["x64"]` and
  `info["x64_session"]` record both.

#### SNPE (`snpe`, `snpe_gpu`)
- A fit whose posterior draw falls back to MCMC is reproducible: two seed-0 `snpe_gpu` fits of
  SN2025pgp / arnett gave max ln L −1837.3 and −1927.4; both now give −1893.9, with identical
  samples. `snpe` on the CPU already repeated, and now gives a different value: SN2025pgp / redback
  arnett, 2×1000 simulations, seed 0, max ln L −1761.15 → −1844.42 (each twice), posterior medians
  within 4%.
- The MCMC fallback's chain count no longer depends on the parallelism: it was
  `min(20, max(4, num_workers))`, 4 chains at 1 worker and 20 at 30. It is `num_chains` (default 4).
- `snpe_gpu` takes `embedding_net=`, as `snpe` does (it raised `TypeError: multiple values for
  keyword argument 'embedding_net'`), and `"mlp"` / `"tcn"` build the same network on both
  samplers. `x_format="stacked"` with `"tcn"` fits; it raised a RuntimeError after the simulations
  had run. `embedding=` still works, with a DeprecationWarning.

#### Every sampler
- An empty light curve is refused with a ValueError that names the usual cause (`select_time_window`
  before `set_explosion_date`), also with your own `log_prob_fn`. `nuts_gpu` and `emcee_jax`
  returned AIC 2k with BIC −inf, the best model in any BIC ranking; the PyMC pair returned BIC 0.0.
- PSIS-LOO works on arviz ≥ 1.0: every fit carried `elpd_loo=None` with a `from_dict()` error.
  `ess_by_parameter` / `ess_summary` are finite (they were NaN).

#### Packaging, environment and examples
- `tests/t3_inference/_stage_common.py` was a SyntaxError, so the staged SBC and
  injection-recovery jobs could not import it. `python -m compileall` goes from 1 error to 0 and
  is a CI step.
- `env.sh` no longer puts the current directory on `sys.path` or on the library path: it set
  `PYTHONPATH=<repo>:` and could leave `::` in `LD_LIBRARY_PATH`. CUDA libraries are found from the
  running interpreter's site-packages; in a python 3.12 environment that is 15 nvidia library
  folders where there were 0.
- The tutorial's runnable examples imported a discontinued package and failed with
  `ModuleNotFoundError`; they are ported to `examples/` and run to the end. The ingestion demo no
  longer deletes your SVO filter cache.
- The quickstart notebook linked the discontinued repository; it links WHISPER-CBPF, and its stored
  outputs no longer show private paths. The test docstrings' run recipes work from any checkout.

### Added

- `whisper_cbpf.synphot`: one photometry definition for the CPU and the GPU. `FilterSet` (save,
  load, hash), `gauss_rule` (16 band-adapted Gauss nodes per band, within 0.02 mmag of the fine
  integral), `filter_set_for` with shipped sets for LSST ugrizy, ZTF gri and SDSS ugriz (no sncosmo
  needed), `band_flux_jy`, `resolve_filter`, and `set_default_band_system` /
  `default_band_system`. The JAX factories take `filter_set=`.
- `LightCurve.set_time_reference(mjd, label)`: plots and `calc_phase` count from a named day 0.
  SN2025pgp shifted to its discovery date was labelled "days since explosion (MJD 60853.300)"; with
  `set_time_reference(60853.30, "discovery")` it reads "days since discovery (MJD 60853.300)".
- redback models: `constraint=`, `photometry=`, `filter_set=`, `default_system=`, `band_aliases=`
  and `times=` on the builders; `whisper_cbpf.models.constraints`; `register_tde(t_exp_days=)` (a
  raw-MJD light curve fits directly and equals days since t0 to 2e-13).
- `Model.param_aliases`: the JAX two-component kilonova's `mej_blue … kappa_red` map onto
  `mej_1 … kappa_2`, so CPU and GPU posteriors pair on all 8 columns (they shared 0):
  `samples.rename(columns=model.param_aliases)`.
- Starting points: `init_strategy=` values `"prior_scan"`, `"prior"` (independent prior draws
  where the density is finite) and one start per chain (refused where the density is `-inf`) for
  NUTS / PyMC; `init=` on `emcee_jax` / `fit_emcee_numpy` (`"prior_scan"`, `"prior"`, `"box"`, a
  point, `(point, scale)`, one start per walker, or a previous result).
- Diagnostics in `info`: `convergence_problems`, `rhat_method`, `frozen_chains`,
  `step_size_by_chain`, `init_time_s`, `postprocess_s` (GPU samplers and SNPE); SNPE's `leakage`,
  `converged` and `x_o_min_rms_z` (in a 45-fit study, the 38 fits that fell back, a median 1.61 σ
  off the emcee reference, looked the same as the 7 that did not, 1.00 σ); `mag_floor_stats` on
  `snpe_gpu`; nested's `sample`.
- `mag_floor` is recorded: `predict_jax.mag_floor`, and the batched forward map counts points at the
  floor and warns above 10% (ZTF20achncvv's magnetar: 46% of simulated points, 26% with the
  constraints).
- A warning when a magnetar spin-down releases more than 10% of its energy before the grid's first
  node (redback 1.20 delivers only 45% of E_rot at t_p = 1.43 s): once per model, in the JAX
  supernova factory's `predict`.
- `examples/demo_ingestion.py`, `examples/compare_samplers.py` and `examples/demo_snpe.py`.
- `io.svo` reads SVO through pyphot when installed (astroquery as the fallback) and caches curves on
  disk.
- Docs: `docs/MIGRATION.md` (old names to new), `docs/PHOTOMETRY.md` (the photometry defaults; SVO curves
  differ from sncosmo's by up to 43 mmag in LSST g), a redback-adapter section in API_REFERENCE.
- Tests for three 0.1.0 fixes that had none: `burnin >= nsteps` is refused by name,
  `LightCurve.where` names a missing column, binding a redback model leaves `text.usetex` alone.
- `MANIFEST.in`: the sdist carries the full test suite, docs, notebooks and examples (126 → 214
  files).

### Changed

- GPU post-fit likelihood re-scan (`nuts_gpu`, `pymc_jax_gpu_*`, `emcee_jax`) runs in vmapped blocks
  of 250: per draw, TDE (n_time=5000) 51.3 → 0.10 ms, kilonova / arnett 0.17 → 0.006–0.014 ms. An
  `emcee_jax` TDE fit of 13,200 draws spent 727 s outside `runtime_s`, now 51 s. Values unchanged
  to 5e-15 relative.
- `make_batched_predict_jax(chunk=…)`, which `snpe_gpu` calls every round, compiles once. At
  B = 1000, float64: arnett 0.59–0.89 → 0.006–0.021 s, kilonova 1.2 → 0.003–0.012 s,
  Gaussian-rise TDE 7.1–7.3 → 0.015–0.17 s. Results are bitwise identical.
- The `prior_scan` start climbs its candidates as one vmapped block: the TDE climb 198 → 46 s
  (n_time=5000: ~30 min → 92 s). The whole default start costs 7–13 s on a kilonova or arnett and
  1.6–1.7 min on the TDE.
- `emcee_jax` warms up every ensemble shape: compile inside `runtime_s`, TDE 9.4 s / 6.7 s and
  arnett 0.76 s → 0.
- Measured block widths (pass the old value to restore it): `emcee_jax(walker_chunk="half")` (was
  16), a TDE fit 317.5 → 147.5 s; `abc_gpu` / `abc_smc_gpu` `chunk=250` (was 16), 20k-simulation TDE
  fit 66.7 → 54.7 s, kilonovae 9–11 → 8.5–8.8 s, arnett 5.4 → 7.1 s (a slower compile);
  `snpe_gpu` `sim_chunk=250` (was 16), TDE 2.25 → 0.73 s per 1000 simulations.
- JAX factories integrate bands with Gauss-16 by default; `n_wave=` gives the old grid. Results move
  by ≤ 0.0015 mmag, and JAX-on-CPU `predict` is 1.5–3.1× faster.
- The JAX TDE prior used without redback is redback 1.20's: all five `cooling_envelope` parameters
  free, `mbh_6` LogUniform(0.1, 10), `stellar_mass` LogUniform(0.5, 10). `pin=` rebuilds 1.15.1's.
- `env.sh` reads `WHISPER_CBPF_VENV`, unset by default, and changes PATH only when you name a venv
  (the discontinued `WHISPER_GPU_VENV` still works, and the old `/opt` default is gone).
- Docs: the NUTS caveat (README, MODEL_COMPARISON, CHOOSING, notebook 07) explains the local-optimum
  mechanism, the new start and how to read `convergence_problems`; "ABC, ABC-SMC, MCMC and SNPE
  reach the same posterior" is replaced by what `examples/compare_samplers.py` measures
  (`sigma_rise`: MCMC 2.91 ± 0.07, SNPE 2.77 ± 0.23, ABC 3.59 +2.41/−1.88, ABC-SMC 6.05 +4.67/−3.56;
  truth 3.0);
  CHOOSING and API_REFERENCE list `snpe_gpu` as registered and `flare_jax` as usable on the GPU;
  notebook 10 matches the plotting API.

### Known issues

- `fit_snpe_gpu` still defaults to `num_simulations=10000`; `snpe` and `SNPEGPUSampler.fit` use
  1000.
- `nested` with `n_jobs > 1` runs a JAX model, but every pool task traces it again:
  `kilonova_one_jax`, nlive=50, 150 iterations, took 199 s with `n_jobs=2` against 2.6 s serially
  (on a loaded host). Fit JAX models with nested sampling serially.
- A `ContextEmbedding` module with `x_format="stacked"` still fails; the `"mlp"` / `"tcn"` strings
  work.
- The magnetar spin-down warning runs only in the JAX factory's `predict`, not in `predict_jax` or
  the CPU redback adapter, whose grid has the same limit.
- An SNPE fit that falls back to MCMC reseeds numpy's and torch's global generators.
