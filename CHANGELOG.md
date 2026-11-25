# Changelog

All notable changes to WHISPER-CBPF are recorded here, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Where a fix was measured, the entry quotes
before → after on the case named. Changes before 0.1.1 were not recorded.

## [Unreleased]

Proposed as 0.4.0: three new TDE models, an `emcee_jax` that runs on the device, a convergence
verdict on every MCMC fit, and `wp.compare` with nested sampling by default and an option to run its
fits in parallel. Read **Changed** before upgrading: `emcee_jax` draws for a given seed,
`wp.compare(sampler="auto")`, the convergence warnings, the `identity_hash` of models that carry
functions, the default numbers of `tde_fallback` and `tde_mosfit`, and the default explosion-time
prior of a free `t_exp` all change.

### Added
- **`ABC_time_prior=True`** in `wp.fit` and `wp.compare` (budget in `ABC_time_prior_kwargs`), and
  the helper **`wp.abc_time_prior(lc, model, prior=None, ...)`** (module
  `whisper_cbpf.samplers.abc_window`), which returns the window and a record. Before the fit,
  whisper's rejection ABC on the GPU (`abc_gpu`) runs the model's JAX version on every detection
  (all bands, no simulated noise), with every parameter drawn from its prior and `t_exp` from the
  default window. A draw is accepted when the RMS of `data - model` in flux is at most 3 times the
  mean flux error of the detections. The fit then uses a flat `t_exp` prior over the accepted range
  plus `max(1 d, 10 %)` on each side, cut to the default window (never wider, never after the first
  detection). With fewer than 50 accepted draws it warns and keeps the default window. The step is
  recorded in `info["abc_time_prior"]`, and `info["t_exp_prior"]["source"]` is `"ABC_time_prior"`.
  A `t_exp` prior passed with it is refused (`ValueError`), and so is a model with no JAX version
  in `wp.fit`; `wp.compare` keeps the default window for such a model (TiDE) and warns once. The
  window comes from the data the fit then uses, so the intervals and evidences are mildly
  data-dependent. A budget count (`n_simulations`, `min_accepted`) that is not a whole number of at
  least 1 is refused, and `wp.fit` checks its sampler and `init=` before the ABC step runs. Tests:
  `tests/test_abc_time_prior.py` (CPU) and `tests/test_abc_time_prior_gpu.py` (skipped without a
  GPU).
- **`wp.default_t_exp_prior(lc, window=None)`** (also `whisper_cbpf.io.default_t_exp_prior`): the
  explosion-time prior a fit gives a free `t_exp` by default, `Uniform(anchor - window, first
  detection)`, named `"t_exp"`. The anchor is the last upper limit before the first detection when
  the light curve has one (any band; limits after the first detection do not count), otherwise the
  first detection. `window` defaults to `whisper_cbpf.io.schema.T_EXP_WINDOW_DAYS` = 30 days,
  observer frame, read at call time; a window that is not finite and > 0 is refused. Tests:
  `tests/test_default_t_exp_prior.py`.
- **MOSFiT's `tde` model**: `wp.tde_mosfit_model` / `wp.register_tde_mosfit` (and `"tde_mosfit"` in `wp.compare`). The fall-back engine of `tde_fallback` with MOSFiT's own photosphere
  (`R = Rph0 a_p (L / Leddlim L_Edd)^lphoto + r_isco`, 4 times larger and 2 times cooler at the peak than redback's), MOSFiT's `tde_constraints` score
  (`whisper_cbpf.models.jax.tde_mosfit.tde_mosfit_constraint_score`; an optional hard wall through `constraint=`) and its Kroupa stellar-mass distribution (`kroupa_icdf`). Against MOSFiT 2.1.1's
  own modules (golden `tests/goldens/tde_mosfit_oracle.npz`, 88 parameter sets): luminosity 8.8e-5 dex median, 1.8e-2 dex at worst. Needs float64. Host extinction is not ported.
- **TiDE, as a CPU model**: `wp.tide_model` / `wp.register_tide` (`whisper_cbpf.models.tide`; and `"tide"` in `wp.compare`) call the C++ library TiDE
  (Kovacs-Stermeczky & Vinko 2023; libtide through tidepy) -- fall-back rate from the Lodato et al. (2009) polytrope tables, a
  wind photosphere and a thin disc -- and return flux density in Jy. CPU samplers only (no JAX half). `import whisper_cbpf`
  does not need libtide; the first light curve does, and its absence raises an `ImportError` that says how to get it. Agrees
  with the unmodified library to 4e-14 (flux) on 59 parameter sets. 0.27 ms per observation, scaling to 16 processes with
  74-98 % efficiency; the model keeps one library object per process because libtide never frees its objects (10.5 MB each).
  `beta_fulldisrupt` defaults to tidepy's 1.85, never silently to libtide's 1.0 (the first-fall-back time differs by 6.33x).
- **Redback's stream-stream TDE**: `wp.tde_stream_stream_model` / `wp.register_tde_stream_stream` (and `"tde_stream_stream"` in `wp.compare`): Piran et al. 2015 / Ryu, Krolik & Piran 2020 stream-stream
  collisions with the Krolik et al. 2024 cooling-time factor, a Gaussian rise and a fixed-area black body. Reproduces redback 1.21.0's flux density to float64 round-off and its sncosmo band output to a median 3.4e-5 mag
  (golden `tests/goldens/tde_stream_stream_oracle.npz`, 118 parameter sets), including the behaviours the keywords `sphere_radius`, `del_omega_in_pi`, `double_dilation`, `nu4_spectrum` and `extrapolate` remove.
  The model ends 1500 days after its time zero. Needs float64.
- **`emcee_jax` runs until converged, on request**: `wp.fit(..., sampler="emcee_jax",
  until_converged=True)` (and `fit_emcee_jax`) with `max_steps="auto"`, `max_time_s="auto"` and
  `check_every="auto"`. The run is chunked: first the pilot, the burn-in and the main run to
  `nsteps`, then `check_every` more steps at a time (default half of `nsteps - burnin`). Each chunk
  is one call of the same compiled program, and the chain's state stays on the device. Between
  chunks the checks of `diagnostics()` run on every draw kept so far, the cheap ones first (stuck
  and frozen walkers, chain length >= 50 tau). The run stops when all pass, or when the next chunk
  would pass `max_steps` (`"auto"`: `UNTIL_MAX_STEPS_FACTOR` = 40 x `nsteps`, 200 000 steps at the
  default 5000) or `max_time_s` (`"auto"`: `UNTIL_MAX_TIME_S` = 600 s, compile included).
  `info["until_converged"]` records the steps run, the stop reason and every chunk's checks;
  `emcee_jax.until_plan(...)` returns the plan. **Off by default**, because on most TDE models it
  runs to the cap without converging. Measured on each JAX TDE model's own 175-point light curve
  (35 epochs x 5 LSST bands), one RTX A6000, 4 x 32 walkers, a 900 s cap, three seeds, compile
  included:

  | model | s per 1000 steps | converged | steps / wall at the stop |
  |---|---|---|---|
  | `tde_stream_stream` | 1.63 | 3 of 3 | 7 000-9 000 steps, 25-29 s |
  | `tde_mosfit` | 6.3-6.7 | 2 of 3 | 33 000 and 73 000 steps, 291 and 556 s |
  | `tde_cooling_envelope` | 4.9 | 0 of 3 | N/tau 34-44 at the cap |
  | `tde_analytical` | 2.15 | 0 of 3 | N/tau 26-29 at the cap; `kappa` piles on its lower bound |
  | `tde_fallback` | 6.45 | 0 of 3 | N/tau 12-14, walker R-hat 1.16-1.20 at the cap |

  N/tau >= 50 is always the last check to fail. For `tde_fallback` and `tde_analytical` the
  estimated tau grows in proportion to the chain, so a larger budget does not help. `max_time_s`
  stops new chunks, not the fit: the last checks and the post-processing add 10-55 s.
- **`wp.fit(..., check_convergence=True)`**: after an MCMC fit (`mcmc`, `emcee_jax`, `nuts_gpu`,
  `pymc_jax_gpu_*`) `wp.fit` runs `result.diagnostics()` and, when a check fails, issues one short
  `UserWarning` in plain English, one sentence per failed check naming the metric, the parameter,
  the value, the threshold and what to do, e.g. `Not converged:` / `The 4 ensembles disagree
  on mstar (R-hat 1.18, needs < 1.01). Run longer (max_steps=10000) or check for a second mode.`
  The first line is `Check the posterior:` instead when only the posterior checks fail (a prior-edge
  pile-up, too few data points), which a longer run does not fix. The verdict is kept in
  `info["convergence_verdict"]` (`passed`, `failed_checks`, `message`). `check_convergence=False`
  turns it off.
- **`diagnostics()` rows for ensemble fits**: `frozen walkers` (`emcee_jax`, `fit_batch`) and
  `R-hat between independent ensembles (largest)` (< 1.01, `emcee_jax` with two or more
  ensembles). Before, `diagnostics()` passed an `emcee_jax` fit whose ensembles disagreed.
- **`wp.compare(..., parallel=True, gpus="auto", cpu_cores=None)`**: every model's fit (and the
  evidence check's runs) goes to its own fresh process through `wp.run_jobs`. A JAX-sampler fit
  gets one card to itself, and the other fits (nested sampling, `mcmc`, TiDE) get their own blocks
  of CPU cores. The fits come back to the calling process, which finds the likelihood maxima, the
  reports and the ranking as the serial run does: with the same seed the comparison is the same
  (tested on two CPU models, nested sampling). `settings["jobs"]` records where each fit ran.
  Measured on the `tde_mosfit` light curve, five JAX TDE models, seed 0: serial nested on one core
  5 870 s; parallel nested on CPU cores 2 429 s; parallel nested with the JAX likelihood on GPUs
  3 522 s (dynesty calls the likelihood one point at a time, and `tde_cooling_envelope` is 4x slower
  on a GPU); parallel `emcee_jax` with `until_converged=True` and a 900 s cap 1 574 s, of which 573
  s are the parent's serial likelihood maxima and reports, and two of five fits stopped at the cap.
  All four rank `tde_mosfit` first, "decisive", and the three nested modes give the same ln Z to
  1e-12. Parallel nested on CPU cores is the fastest mode in which every fit converges.
- **`wp.check_device_transfers(devices=None, *, refresh=False)`**: copies a 4 MiB array with a
  pattern unique to the pair and the call (`parallel.TRANSFER_CHECK_BYTES`) between every ordered
  pair of devices, compares it bit for bit, and raises `RuntimeError` naming every bad pair and
  pointing to one process per card (`run_jobs`). The result is kept per process and per device set.
  On some multi-GPU hosts a card-to-card copy returns zeros or stale data with no error; on the
  host this package is validated on, the check finds such a pair (all values zero one way, stale
  data the other) and passes the intact pairs, in about 1 s including the CUDA start. No fit uses
  two cards in one process (`emcee_jax` and `fit_batch` run on one device, `run_jobs` and
  `compare(parallel=True)` one process per card), so nothing calls it automatically: call it before
  your own multi-device code (sharded arrays, `pmap`, chains split over cards).
- `emcee_jax`: `n_ensembles=4` (`DEFAULT_N_ENSEMBLES`), `pilot="auto"`, `pilot_steps="auto"`
  (`nsteps // 5`); module-level helpers `ensemble_seeds(seed, n_ensembles)`, `pilot_plan(...)` and
  `until_plan(...)`; new `info` keys `n_ensembles`, `walkers_per_ensemble`, `ensemble_seeds`,
  `ensemble_rhat`, `ensemble_ess_bulk`, `ensemble_ess_tail`, `ensemble_rhat_log_likelihood`,
  `ensemble_rhat_method`, `pilot`, `autocorr_time_by_ensemble`, `frozen_walkers`,
  `n_nan_proposals`, `program`, `device_timing` and `until_converged`. `init=` also takes one start
  per ensemble, an `(n_ensembles, nwalkers, ndim)` array.
- `info["pilot"]` of `emcee_jax` gains `restarted_ensembles` (always every ensemble,
  `list(range(n_ensembles))`) and `independent` (always `False`): informational keys that say the
  main-run ensembles are no longer independent starts once the pilot has run. No check reads them,
  and the draws do not change.
- **A warning when a fit starts on a detection consistent with zero flux.** `wp.fit`, every
  sampler's `fit`, `fit_batch` and `wp.compare` issue one `ZeroFluxWarning` before the fit when the
  earliest detection(s) the fit uses have S/N below 3, for a model with an explosion time (free or
  fixed) or a light curve with a declared explosion date. S/N is `|flux| / flux_err`, or
  `(2.5 / ln 10) / magnitude_err` for a row with only a magnitude; the leading detections are listed
  up to the first significant one, with MJD, band and S/N. A model that switches on as a step
  (`tde_stream_stream`) cannot fit a zero-flux point after `t_exp`: on such a light curve every
  sampler settled about 18 d before the true explosion and `emcee_jax` still reported converged;
  marking the first epoch as upper limits recovered the truth. The data are not changed; mark the
  point as an upper limit (`upper_limit=True`) if it is a non-detection. The warning is given once
  per fit on every path (`emcee_jax` and `likelihood_max_opt=True` included), once for a whole
  `wp.compare`, and once for a `fit_batch`, naming the light curves by index. New in
  `whisper_cbpf.samplers`: `ZeroFluxWarning`, `zero_flux_first_detections(lc, threshold=3.0)` and
  `ZERO_FLUX_SNR` = 3.0. Silence it with
  `warnings.filterwarnings("ignore", category=whisper_cbpf.samplers.ZeroFluxWarning)`. Tests:
  `tests/test_zero_flux_first_point.py`.
- `tde_fallback_model(unique_epochs=True)` and `tde_mosfit_model(unique_epochs=True)`; their
  `predict_jax.epoch_index(times, size=None)` (the host-side epoch map) and the `epoch_idx=` keyword of
  `predict_jax`, `tde_fallback_ab_magnitude` and `tde_mosfit_ab_magnitude`.
- `log_density(epoch_bucket="auto")` and `LogDensity.n_distinct_epochs` / `.epoch_bucket`;
  `fit_batch` reports `info["batch"]["epoch_bucket"]` and `["n_distinct_epochs"]`;
  `ProfileReport.n_distinct_epochs` / `.epoch_bucket`.
- `geometric=` (default `True`) on both factories and on the physics functions that build their
  own dense grid with `geometric_dense_times`: `tde_fallback_bolometric_log10`,
  `tde_fallback_ab_magnitude`, `tde_mosfit_curve`, `tde_mosfit_ab_magnitude`,
  `tde_mosfit_constraint_score` and `tde_mosfit_flux_density_mjy`. `True` finds each
  viscous-kernel node's segment on the geometric dense grid by arithmetic instead of a search
  (porting rule (a), the same interpolant). `viscous_log10` takes `geometric=` too, with default
  `False`: it takes the caller's grid, and on a valid grid that is not geometric the arithmetic
  picks wrong segments (on a linear 1000-node grid, 20 prior draws x 48 epochs, log10 L off by a
  median 0.52 dex and up to 3.95 dex). The constraint wall (`tde_mosfit_constraint`, one function
  for every model) keeps the search. `geometric=True` needs at least 2 dense nodes and raises
  `ValueError` otherwise (at the factory call and in `viscous_log10`). `geometric=False` keeps
  the search, and `unique_epochs=False, geometric=False` is the code before these options
  existed: those models keep their `model_record` identity, while the default models are
  identified as geometric (see Changed).

  Against the search, on 35 epochs x 5 LSST bands, CPU: magnitudes bitwise over 200 prior draws in
  the batched programs (host `predict`, one parameter set per call, 98.7-99.3 % bitwise, at most
  5.4e-14 and 3.4e-13 mag), the log posterior of `log_density`'s batched programs bitwise (one
  parameter set per call, as `likelihood_max_opt`'s value program evaluates it, to a relative
  8.6e-15), gradients within 3.0e-15 of their largest component; one A6000 GPU: host `predict`
  and the log posterior of `log_density`'s batched programs bitwise, gradients within 1.0e-14.
  The bolometric function, the score and the flux density (200 prior draws x 48 epochs, CPU):
  values bitwise, the gradients of the first two within 3.5e-15 and 5.3e-15 of their largest
  component; in float32 the
  bolometric function's values are the search's, bit for bit (360 draws x 60 epochs).
  The default (unique epochs and the arithmetic search) against the code before both options, on
  five light curves (35, 28 and 21 epochs x 5 bands, an edge-case curve and the shipped alert;
  fixed and free `t_exp` and redshift; every call path), CPU: magnitudes at most 8.9e-14 mag
  (`tde_fallback`) and 2.5e-13 mag (`tde_mosfit`), host `predict` at least 98.1 % and 98.5 %
  bitwise, log densities to 7.8e-15 and 1.2e-14 of the size of their summed terms, gradients
  within 1.9e-12 and 1.11e-12 of their largest component, `fit_batch` started with
  `init="prior"` the same chains, bit for bit; one A6000 GPU under XLA's deterministic flags:
  host `predict` and `log_density`'s value programs bitwise (the value it returns with its
  gradient within 4.2e-16 of the size of its summed terms, through `make_log_prob_jax` within
  3.5e-15), magnitudes at most 2.0e-14 mag, gradients within 7.4e-13 and 1.6e-13, the same
  `fit_batch` chains. These maxima are those curves'. Over 200 fresh random light curves (1-79
  distinct epochs, 1-317 rows in up to six LSST bands; both models, fixed and free `t_exp` and
  redshift), CPU: magnitudes at most 1.9e-13 and 2.8e-13 mag, host `predict` at least 99.0 %
  bitwise, gradients within 1.6e-12 and 5.1e-12 of their largest component; on the GPU under the
  deterministic flags, on 17 of those curves (every clock and configuration), every flux path and
  every value program of the log density bitwise (the value returned with a gradient within
  3.0e-16), gradients within 8.0e-14.

  Speed on one A6000, float64, 35 epochs x 5 bands, free `t_exp`, against unique epochs alone
  (`geometric=False`), in one run (host load 13-20): 1.10-1.50x per evaluation on the 175-row
  curve (1.06-1.38x on the shipped alert, 1.12-1.49x with every time distinct; one program is not
  faster, `tde_fallback`'s `jit(vmap(predict_jax))` at 4 sets, 0.95x). `fit_batch` (32 walkers x
  5000 steps) for one light curve: in that run (host load 13-28 at the start of the median
  rounds) 40.7 → 38.1 s (`tde_fallback`) and 37.9 → 34.6 s (`tde_mosfit`), the sampling 12.6 →
  10.1 s and 12.5 → 10.0 s; in an earlier run (that of the 68 → 37 s under Changed; host load 10-15
  at the start of its fits) 36.4-37.1 → 32.7-34.1 s over four
  (truth, model) pairs, and for 64 light curves (load 11) 620-624 → 449.6-454.9 s. Compile: the
  summed compile time of 18 programs 69.8 → 70.6 s (`tde_fallback`) and 65.3 → 69.3 s
  (`tde_mosfit`), single programs 0.86-1.34x (`jit(vmap(...))` at 256 sets on the all-distinct
  curve 3.14 → 4.22 s and 3.25 → 4.36 s); against the code before both options, median 0.98-0.99 per
  program (0.78-1.58). Host `predict` on a CPU, one parameter set on the 175-row curve (the
  earlier run, load about 13): 9.76 → 6.67 ms and 8.64 → 5.68 ms.

### Changed
- **`tde_analytical`: the default prior has `kappa` LogUniform(0.05, 100) instead of redback's
  Uniform(0.05, 100)**, a deliberate difference from redback (docs/PORTING_NOTES.md,
  "`tde_analytical`: one deliberate prior difference"). The bounds are read from redback's prior
  file at run time; every other distribution is redback's. Under the Uniform, 99.5 % of the prior
  lies above 0.5 cm^2/g (0.45 % below; LogUniform 30 %), and the samplers settled in a wrong
  solution (`kappa` 60-100, `mej` 20-100) 42-253 nats below the true one, with `t_exp` 4.5-11.7 d
  late; with the LogUniform, 14 of 14 fits on those light curves sat in the true solution and
  `emcee_jax` converged in 36-55 s. New: `redback_default_prior()` (redback's prior unchanged),
  `tde_analytical_model(prior="redback")` (any other string is refused) and
  `LOG_UNIFORM_OVER_REDBACK = ("kappa",)` in `whisper_cbpf.models.jax.tde_analytical`; a caller's
  `kappa` prior wins over both. `wp.compare` binds `tde_analytical` with the new default; pass
  `prior={"tde_analytical": redback_default_prior()[0]}` for redback's. The model description
  notes the LogUniform `kappa` when the default is in use. A cost: on light curves whose best
  `tde_analytical` solution needs a large `kappa` (51-80, 6 % of the new prior mass, 29 % of the
  old), nested sampling can miss that mode: on the MOSFiT-truth curve of the model-comparison
  matrix 4 of 4 runs stopped 1262 nats below it, so the model's evidence there is far too low.
  Fits cached with `fit_cached` under the old default are not reused (the prior is in the cache
  key); a saved `Comparison` re-binds with the prior it was fitted under.
- **Breaking: the default explosion-time prior is 30 days wider.** When the prior passed to a fit
  names no `t_exp`, `wp.fit`, every sampler's `fit`, `fit_batch`, `log_density` and `wp.compare`
  now use `wp.default_t_exp_prior(lc)`: from 30 days before the last non-detection that precedes
  the first detection (or before the first detection when there is none) to the first detection.
  Before, the window started *at* the last non-detection (`lc.explosion_time_prior()`, which is
  unchanged). A model's own `Uniform` `t_exp` prior is cut to the new window on both sides (never
  widened); a model with no `t_exp` prior takes it; `Normal`, `TruncatedNormal` and `Fixed` model
  priors are kept; a prior passed to `fit` (or to `compare` for one model) is used as given.
  `wp.compare` binds a family's `t_exp` to the new window. `info["t_exp_prior"]` gains `anchor`
  and `window_days`; its `source` is now `"default window"` or `"model prior, cut to the default
  window"` (was `"data"`, `"data, fallback window"`, `"model prior, cut to the data window"`).
  Without a non-detection before the first detection nothing changes (the window was already
  30 days before the first detection), except that a model's Uniform prior is now cut from below
  too. Why: on simulated LSST alerts the old window excluded the true explosion in 24 of 30
  (docs/VALIDATION.md section 4.2), and a non-detection shallower than the source does not prove
  that it had not exploded. A posterior piled against the lower edge means the rise was not
  observed and 30 days is too short: pass a wider prior. To keep the old behaviour pass
  `prior=Prior({"t_exp": lc.explosion_time_prior()})`. Results whose explosion-time prior came from
  the old default change and should be re-run.
- **Cooling envelope, time dilation (documentation only).** The JAX Gaussian-rise cooling envelope
  dilates time once, by `(1+z)`, where redback 1.20-1.21's `gaussianrise_cooling_envelope` dilates
  it twice. This is now stated as a deliberate, documented deviation from redback (a project
  decision; the double dilation was reported upstream as a redback bug) in `tde_model`'s
  docstring, the `whisper_cbpf.models.jax.tde` module docstring and `docs/PORTING_NOTES.md`
  section 4.5. The code is unchanged.
- **Breaking: `emcee_jax` runs entirely on the device.** It no longer drives the emcee library
  from a Python loop: emcee's stretch move (a = 2, emcee's randomised red-blue split) runs as one
  compiled loop on the device, the `fit_batch` engine, with no return to Python between steps.
  The call and the `SamplerResult` it returns are the same; what changes:
  - draws for a given seed differ from earlier versions (other random streams);
  - by default 4 independent ensembles (`n_ensembles=4`) of `nwalkers=32` walkers each, so 128
    walkers. `nwalkers` is now per ensemble, and `info["nwalkers"]` is the total;
  - with two or more ensembles a pilot of `nsteps // 5` steps runs first. Every ensemble then
    restarts from the kept pilot states of the ensemble with the highest log likelihood, which
    drops an ensemble stranded in a lower mode, and the main run follows, in the same compiled
    loop. The ensemble followed is the one with the highest likelihood peak, not the most
    probability mass: a second peak of comparable (or larger) probability is dropped, and the fit
    can still report converged. On a two-mode toy with equal mass the main-run R-hat was 1.000 and
    the mass of the dropped mode 0 (truth 0.5), whether 1, 2 or 3 of the 4 ensembles started in
    it; seen on MOSFiT's `starmass`-`efficiency` modes, 0-0.5 nats apart. After the restart the
    between-ensemble R-hat tests mixing within one peak, not that independent starts agree; only
    the pilot R-hat (`info["pilot"]["rhat"]`) records the disagreement, and no check reads it. Use
    `pilot=False` or nested sampling when a second mode matters. The `fit_emcee_jax` docstring now
    says this;
  - `converged` also fails when the R-hat between the ensembles is 1.01 or more;
  - `init="prior_scan"` (the default) is the derivative-free scan `fit_batch` uses, no longer the
    gradient climb;
  - `progress=` is accepted and ignored, and `result.emcee_sampler` is gone (there is no emcee
    object);
  - on a GPU a fixed seed gives identical draws only under XLA's deterministic flags
    (`XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"`, set before JAX
    starts). Without them two runs with the same seed sample the same posterior but differ draw by
    draw (on `tde_stream_stream`, every draw of two runs differed), and with `until_converged`
    they can stop at different lengths. On the CPU a fixed seed is reproduced exactly.

  Measured on `tde_analytical`'s own 175-point light curve, 4000 steps, one RTX A6000, a cold
  process: the old 32-walker run 27.1-27.9 s (sampling 21.0-22.0 s); the same work on the device
  14.9 s (sampling 5.0 s); the new default, 4 x 32 walkers and an 800-step pilot, 25.0 s. The old
  run's best ln L was 47 nats below the device runs' (1697.7 against 1745.1). `fit_batch`'s public
  behaviour and its draws are unchanged (its loop is now one `lax.fori_loop`; on the same curve
  with 4 light curves x 4000 steps the compile went from 9.0 to 7.2 s).
- **Breaking: `wp.compare(sampler="auto")` is nested sampling for every model**
  (`compare.AUTO_SAMPLER = "nested"`), CPU and JAX models alike, because it gives each model its ln
  Z and the ranking is then by ln Z. It was `emcee_jax` on a GPU and CPU `mcmc` otherwise, ranked
  by BIC. Settings such as `nsteps=` or `nwalkers=` passed to `compare` with the default sampler are
  now refused (nested sampling takes `nlive=`, `dlogz=`); pass `sampler="emcee_jax"` or
  `sampler="mcmc"` to keep the old behaviour. `wp.fit(sampler="auto")` keeps its own rule
  (`emcee_jax` for a JAX model when JAX sees a GPU, CPU `mcmc` otherwise). The three ways to run a
  comparison are in the API reference (A5).
- **`wp.fit`'s convergence warnings start "Not converged:"** (or "Check the posterior:"), with one
  sentence per failed check, instead of "<sampler>: this posterior is not converged -- ...". The
  sampler's own warning is no longer repeated inside `wp.fit`; direct calls (`fit_MCMC`,
  `fit_emcee_jax`, ...) warn as before. `wp.compare`, `run_jobs` and `fit_cached` call `wp.fit` and
  carry the same verdict.
- **The `dev` extra pins `redback>=1.20,<1.21`**, the version the test suite's redback comparisons
  are validated against. The `models` extra keeps `redback>=1.12` for users.
- **`tde_fallback` and `tde_mosfit` evaluate each distinct epoch once.** A multi-band light curve
  observes an epoch in several bands, and the engine, the viscous integral (1000 nodes, each
  interpolated on the 1000-point dense grid) and the photosphere were computed again for every
  band. They now run once per distinct epoch and are copied to the observations before the band
  integral, as redback's and MOSFiT's `Viscous` do (porting rule (a)).

  With concrete epochs the map is automatic. `log_density` passes it as data, so `fit_batch`,
  `likelihood_max_opt` and `profile` get it. Traced epochs without a map are evaluated per
  observation, as before.

  The figures in this entry are for unique epochs alone (`geometric=False`); the default also
  finds the kernel's segments by arithmetic (Added, above, with the two together against the code
  before them, and its speed on top of unique epochs: in the run of the `fit_batch` figures at
  host load 10-12 below, one light curve 36.4-37.1 → 32.7-34.1 s). Against the per-observation
  evaluation, on 35 epochs x 5 LSST bands (200 prior
  draws, fixed and free `t_exp` and redshift, every call path, CPU):
  - magnitudes to at most 6.4e-14 mag (`tde_fallback`) and 1.11e-13 mag (`tde_mosfit`), host
    `predict` bitwise; log posterior through `log_density` to a relative 7.9e-16 and 2.1e-16 (at
    least 98.5 % and 99.5 % bitwise);
  - gradients within 9.3e-14 and 2.8e-13 of their largest component;
  - `fit_batch` started with `init="prior"` (16 walkers x 200 steps, 35, 28 and 21 epochs in one
    call) gives the same chains, bit for bit. With the default `init="prior_scan"` the start's climb
    amplifies the round-off: in 5 of these 6 fits the starting walkers moved by 2.1e-10 to 8.7e-5,
    so the chains differ from the first step while they sample the same density.

  These maxima are those curves'. Over 200 random light curves (1-81 distinct epochs, 2-311 rows
  in up to six LSST bands: unsorted rows, same-band repeats, rows before the time zero, upper
  limits; both models, 16 prior draws each), on the CPU the magnitudes reached 3.2e-13 mag and the
  gradients 2.7e-12 of their largest component, and host `predict` was bitwise on all but one
  `tde_fallback` curve (91.3 % bitwise there, at most 3.6e-15 mag); on one A6000 GPU under XLA's
  default flags, on 50 of those curves, host `predict` differed in 5 of the 100 (curve, model)
  pairs, by up to 4.3e-14 mag. All of it is within the package's tolerances (1e-12 mag, 1e-10 of a
  gradient's largest component).

  **`likelihood_max_opt`'s peak, and the AIC and BIC from it, are not reproduced to round-off.**
  Its climbs follow the density's round-off and can end at another point, or in another basin:
  from the same fit (the same draws, the same starting ln L), one of the two `tde_mosfit` members
  tried moved from ln L 580.33 to 586.15 (AIC and BIC 11.6 lower, `starmass` 0.36 → 0.10), the
  other by 2.1e-8; on a GPU two calls of the code before this option, from one fit, differ by up
  to 3.0e-3. Model-comparison tables built on these peaks must be re-run.

  On one A6000 GPU, float64, 35 epochs x 5 LSST bands with free `t_exp`: the log posterior through
  `log_density` (its default epoch bucket, 48) 2.0-2.4× faster per evaluation at 16 parameter sets
  and 3.5× at 1024, and 4.4× at 256 with the 35 slots `fit_batch` pads to; its gradient
  17.2-19.2× at 16 sets and 25× at 64; the XLA scratch memory of these programs 3.5-6× smaller
  (the value at 2048 sets 15.8 → 4.0 GB); `fit_batch` (32 walkers x 5000 steps) 68 → 37 s for one
  light curve at host load 10-12 and 74.5-78.5 → 44.8-46.2 s at load 41-50 (the call alone; from
  the process's start, its imports and the model's build included, 80.9-87.5 → 51.0-54.6 s there),
  and 2665 → 620 s for 64 (the sampling 41.6 → 12.7 s and 2293 → 492 s), although such chains did
  not converge by the package's diagnostics on these curves; `likelihood_max_opt` from the same fit
  2.1-3.2× faster (98-143 → 32-61 s at load 31-50: some 26 000-45 000 sequential evaluations, whose
  number varies from call to call, so its time follows the host load and the starting chain);
  compile times unchanged. Exactness there, under XLA's deterministic flags: magnitudes to 2.0e-14 mag, host
  `predict` and `log_density`'s value programs bitwise (the value it returns with its gradient
  within 4.2e-16 of the size of its summed terms; through `make_log_prob_jax` to a relative
  2.3e-14), gradients within 7.4e-13, the same `fit_batch` chains with `init="prior"`. A
  light curve whose times are all distinct gains nothing and pays nothing measurable: with 175
  distinct times the medians of the paired per-round ratios before/after are 1.00-1.03x (the value
  at 256 sets; the value and gradient at 16 sets for `tde_fallback`, at 1 set for `tde_mosfit`,
  1.03x); on the shipped LSST alert, whose 12
  distinct epochs fill its 16 slots, 0.98-1.01x, while single timings of its millisecond-scale
  programs scatter by up to 25 % either way. On a CPU (4 cores, the two codes called alternately), with every time
  distinct, host `predict` and the one-set value and gradient within 1.3 %.

  `unique_epochs=False, geometric=False` evaluates every observation with the search, the code
  before these options existed (with MOSFiT's engine, below). The derivative with respect to traced
  epochs given a map is per distinct epoch.
- **The viscous kernel of `tde_fallback` and `tde_mosfit` finds its segments by arithmetic by
  default** (`geometric=True`, Added). The default numbers of both factories and of
  `tde_fallback_bolometric_log10`, `tde_fallback_ab_magnitude`, `tde_mosfit_curve`,
  `tde_mosfit_ab_magnitude`, `tde_mosfit_constraint_score` and `tde_mosfit_flux_density_mjy`
  move at round-off against the search (the figures under Added); pass `geometric=False` for the
  search's numbers. The functions that keep the search are deliberate exceptions to
  `geometric=True`: `viscous_log10` (default `geometric=False`, because it takes the caller's grid,
  which need not be geometric) and the constraint wall `tde_mosfit_constraint`.
- **`log_density` data of the fall-back TDEs carry the epoch map** (`epoch_first`, `epoch_inv`).
  With the default `bucket="auto"`, two light curves in one bucket can differ in `epoch_bucket`:
  they then compile separately and their data do not stack. An explicit `bucket=` pads the distinct
  epochs to it as well, which keeps one compile per bucket and stackable data at the
  per-observation cost; a common `epoch_bucket=` keeps the speed-up. `fit_batch` does this itself.
- **Cached fits of `tde_fallback` and `tde_mosfit` re-run**: their `predict_jax` changed and the
  default model's state records `geometric=True`, so `model_record`'s `identity_hash` and
  `fit_cached`'s key changed. `unique_epochs=False, geometric=False` models, and every other
  family, keep theirs (`geometric` is in a model's state only when True), apart from the one-time
  change of every key that holds a function (the install-path fix, under Fixed). A model built with
  `unique_epochs=False` alone now has `geometric=True`: its key changed too, and it is no longer
  the code before these options (pass `geometric=False` as well).
- **`tde_fallback` and `tde_mosfit`: the fall-back engine is MOSFiT's `Fallback`, node for node.** Up to 0.3.0 the
  Guillochon & Ramirez-Ruiz tables were resampled once onto 512 equal steps of normalised log time per branch (log10
  `dM/dt` drawn as a straight line in log time between the simulation's nodes), interpolated in `beta` and in gamma at
  fixed phase, and log10 `dM/dt` interpolated linearly in time. MOSFiT (and redback, which transcribes it) maps the
  neighbouring curve with fewer nodes onto the other's own nodes, interpolates in `beta` and gamma there, and
  interpolates `dM/dt` itself linearly in time. Now MOSFiT's `Fallback.__init__` runs once, offline
  (`tde_fallback.build_fallback_tables`; its output is the new `whisper_cbpf/data/tde_fallback_nodes.npz`, 1.3 MB,
  equal bit for bit to MOSFiT's and redback's arrays), and `Fallback.process` runs in the trace. Before → after, on
  the same inputs:
  - luminosity before the viscous delay against redback's `_tde_mosfit_engine` (634 parameter sets: the 88 MOSFiT
    and 65 redback golden sets, `bb` = 0, 1, 2 and exactly at simulated betas, every gamma regime and its limits,
    stars below 0.1 solar masses): median 5.0e-5, p99 1.4e-2, max 2.8e-2 dex → 0, 1.4e-14, 7.1e-14 dex (relative
    1.64e-13 at worst; the same against MOSFiT's `Fallback`); `t_peak`, `beta` and the stellar radius stay at 3e-15;
  - `tde_fallback` against redback: bolometric (60 prior draws x 60 epochs) 5.4e-5 / 1.4e-2 / 2.1e-2 dex → 1.1e-5 /
    1.4e-4 / 2.2e-4 dex (over 200 draws x 701 epochs from 1e-3 to 1e3 d the maximum is 2.6e-4); band magnitudes
    (300 prior draws, LSST griz and ZTF gr, z = 0.05) 6.9e-5 / 1.0e-2 / 4.0e-2 mag → 2.2e-5 / 3.7e-4 / 7.9e-4 mag; the
    redback golden (`tests/goldens/tde_redback_oracle.npz`, 65 sets) L
    6.2e-5 / 1.2e-2 / 2.0e-2 dex → 1.1e-5 / 1.4e-4 / 2.3e-4 dex and F_nu (relative) 9.6e-5 / 1.7e-2 / 4.6e-2 →
    2.3e-5 / 4.2e-4 / 8.0e-4 (median / p99 / max). What is left is the viscous kernel's interpolation of log10 L
    between the dense nodes (redback's own kernel on the new engine gives 2e-14);
  - `tde_mosfit` against MOSFiT's golden (88 sets x 64 epochs): L 8.8e-5 / 8.2e-3 / 1.8e-2 dex → 4.9e-5 / 2.2e-4 /
    1.7e-3; T 2.4e-5 / 1.8e-3 / 3.9e-3 → 1.1e-5 / 6.0e-5 / 4.2e-4; R 4.4e-5 / 4.8e-3 / 9.8e-3 → 1.4e-5 / 1.8e-4 /
    8.1e-4; F_nu 9.3e-5 / 1.0e-2 / 5.7e-2 → 4.1e-5 / 3.9e-4 / 3.6e-3; the constraint score at worst 6.7e-3 → 7.8e-4
    (relative). What is left is the viscous delay (MOSFiT's exact filter applied to the new engine on the golden's
    grid gives 4e-14);
  - the rate now has MOSFiT's (and redback's) two discontinuities, which the 0.3.0 engine did not have: it jumps where
    `beta` equals a simulated value (before the peak up to 1.47e-2 dex, over every simulated beta a float `bb`
    reaches, at two stars per gamma) and at the limits of the gamma regimes (at 1e6 solar masses and `bb` = 0.25,
    0.5, 0.77, 1, 1.5: up to 2.3e-2 dex just above `mstar` = 0.3, 1.5e-2 just below 22, 8e-5 below 1, 5e-5 above 15;
    the 0.3.0 engine was continuous there to 2e-9). The model is piecewise smooth in `bb` and in `mstar`. Gradients
    stay finite; at a jump the derivative is that of the side selected, not a one-sided limit (at a simulated beta, a
    star of one gamma has a derivative of exactly 0 in `bb`; across `mstar` = 0.3 or 22 a finite difference sees the
    jump), which matters to a sampler that starts on such a point, e.g. `bb` = 1;
  - redback-jax, which keeps the 512-step resampling, is no longer matched to round-off (6e-14 → engine median 6.0e-5,
    max 2.4e-2 dex; bolometric 4.7e-5 / 2.3e-2), exactly its own difference from redback;
  - float32 still works: the bolometric light curve within 1.2e-4 dex of float64 over 60 prior draws (1.24e-4
    before; 1.1e-4 over 300 draws). New: in float32 the equality branch compares float32 betas, so a pinned round `bb`
    can take it where float64, MOSFiT and redback do not, or miss it (of `bb` = 0.00, 0.01, ..., 2.00: 0.04, 0.08,
    0.24, 0.88, 0.96), moving the rate by up to 3.6e-3 dex (at most a jump's size); pin `bb` in float64;
  - cost, `jit(vmap(...))` at 30 epochs x 6 bands, old and new engine run alternately (medians): on a GPU (one RTX
    A6000) at one parameter set `tde_fallback` 0.63-0.71 → 0.74-0.81 ms and `tde_mosfit` 0.67-0.68 → 0.78-0.80 ms
    (+13 % and +18 % on the means of three rounds of 300 runs), and the gradient of a Gaussian log-likelihood 1.21-1.36
    → 1.56-1.66 ms and 1.19-1.25 → 1.61-1.66 ms (+23 % and +34 %); at 64 sets +1 to +3 %, at 1024 and 2048 sets +0.2
    to +0.6 %, the gradient at 64 and 1024 sets unchanged (within 0.3 %). On four CPU cores `tde_fallback` +6 % at
    one set (20.2 → 21.3-21.6 ms) and +3 to +4 % at 64 and 1024 sets, `tde_mosfit` within the run-to-run spread at 1
    and 64 sets and +1 % at 1024; the gradient at one set +5 to +8 %. Compilation (lower + compile, 1 to 2048 sets):
    on the GPU 1.3-2.1 → 2.2-3.2 s for the prediction and 2.4-3.2 → 4.0-5.2 s for the gradient, on the CPU 0.5-1.0
    → 1.3-2.4 s and 1.4-2.5 → 3.1-4.1 s. The generated GPU code is 0.85 → 1.97 MB and, as before, the same at every
    batch from 1 to 2048.
- `tde_fallback.TABLE_FILE` is now `whisper_cbpf/data/tde_fallback_nodes.npz`; `build_fallback_tables(table_dir)`
  returns MOSFiT's `Fallback.__init__` arrays (its `n_phase` argument is gone); `fallback_tables()` returns
  `{gamma: {name: array or (rising, falling) pair of arrays}}` (`sim_beta` and the valid-node counts `premap_count`
  and `mapped_count` as arrays; `premaptime`, `premapdmdt`, `mapped_time`, `beta_slope` and `beta_yinter` as one array
  per branch) instead of `{gamma: (beta, log_time, log_dmdt)}`.
- `tde_fallback.py` and `tde_mosfit.py` carry MOSFiT's MIT copyright and permission notice in their headers, so that it
  ships with the code in a wheel too (the wheel's metadata holds only `LICENSE`; `THIRD_PARTY_NOTICES.md` is in the
  source distribution).

### Deprecated
- `whisper_cbpf/data/tde_fallback_tables.npz`, the 0.3.0 resampled tables, is no longer read by any code and the
  package can no longer write it (0.3.0's `build_fallback_tables` is gone); it still ships in the package, and
  will be removed in the next release.

### Removed
- `tde_fallback.N_PHASE`: the resampling it sized is gone.
- `SamplerResult.emcee_sampler` on `emcee_jax` fits: `emcee_jax` no longer builds an emcee object.

### Fixed
- **A model's `identity_hash` no longer depends on where the package is installed.** A function in
  a model's record (a closure, a nested function, a lambda) was identified by its code object,
  which carries the file name and line numbers, so the same model imported from two install paths
  had two hashes, and `fit_cached` / `compare(cache_dir=...)` keys did not carry over between
  machines. The code part is now a SHA-256 of the bytecode, the names it uses and its constants,
  recursively, independent of the path and of the string-hash seed; a real code change (another
  global or attribute name, constant or body, or another closure value) still changes it. The keys
  of models whose record holds a function change once with this release: such cached fits re-run.
- **`test_fit_batch` claim 2 (a fit does not depend on the batch) no longer flakes on a GPU**: it
  runs in a fresh process under `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`,
  without which the batch program and the one-curve program can part at the last bit after a few
  thousand steps. On the CPU it runs as before.
- **`abc_gpu`: a draw the model returns NaN for is no longer the "closest draw".** `min_distance`
  was NaN as soon as one draw's distance was NaN, and with nothing accepted the closest rejected
  draw (`best_params`) could be that NaN draw. Both now skip NaN distances (`min_distance` is NaN
  only when every draw is). On a toy model that is NaN for 13 % of its prior draws:
  `min_distance` NaN → the smallest finite RMS.

## [0.3.0] - 2026-10-01

More physical models, adapted from redback and redback-jax, and a GPU compile time that no longer
grows with the batch. Nothing in the existing models changed: their magnitudes agree with 0.2.0 to
1e-12 mag.

### Added
- **Four more supernova-family models**, through `supernova_model` / `register_supernova` and by name
  in `wp.compare` (`wp.supernova_models()` lists sixteen): `csm_interaction` (Chevalier–Chatzopoulos
  CSM interaction; a different model from `csm_shock_and_arnett`, which is Margalit's), `shock_cooling`
  (Piro, alone, with its own photosphere), `shocked_cocoon` and `general_magnetar_driven` (a
  relativistic magnetar-driven ejecta ODE). They are evaluated at the observations themselves and need
  float64. `supernova_model(engine_options=...)` sets one model's engine: `csm_diffusion_method` and
  `dense_resolution` for `csm_interaction`; `solver`, `n_grid`, `rtol`, `atol` and `max_steps` for
  `general_magnetar_driven`.
- **Two more TDE models**: `wp.tde_fallback_model` / `wp.register_tde_fallback` (MOSFiT fall-back
  tables with a viscous delay) and `wp.tde_analytical_model` / `wp.register_tde_analytical` (a
  `t^-5/3` fall-back through diffusion, with redback's two constraints as a hard wall). Both need
  float64, and are bound by name in `wp.compare` (`"tde_fallback"`, `"tde_analytical"`).
- **Two Metzger kilonovae**: `wp.metzger_kilonova_model` / `wp.register_metzger_kilonova` and
  `wp.magnetar_boosted_kilonova_model` / `wp.register_magnetar_boosted_kilonova`, bound in
  `wp.compare` as `"metzger_kilonova"` and `"magnetar_boosted_kilonova"`. They are not comparable
  parameter for parameter with `kilonova_one_jax`. float32 is safe.
- **Four phenomenological light-curve models** with one amplitude per band: `wp.phenomenological_models`,
  `wp.phenomenological_model`, `wp.register_phenomenological` (`bazin_sne`, `villar_sne`,
  `smooth_exponential_powerlaw`, `exp_rise_powerlaw_decline`). Their default amplitude prior is a
  placeholder: pass `prior=`. `wp.compare` takes the `Model`, not a name.
- **The `[ode]` extra** (`pip install "whisper-cbpf[ode]"`): `diffrax`, the adaptive solver of
  `general_magnetar_driven`. Without it the model uses a fixed-step solver built into the package and
  warns once. The core install does not need it.
- `data/*.npz` and `data/*.txt` are package data (the TDE tables and the CSM-interaction table).
- `THIRD_PARTY_NOTICES.md`: what is adapted from redback and redback-jax, the files that carry it, and
  the papers the models implement; `CITATION.cff` lists both repositories.

### Changed
- `wp.supernova_models()` lists sixteen names, not twelve: a loop over it now meets the four new
  models, which need float64 like the others.
- The float64 error of the supernova engines no longer calls every engine a supernova luminosity, so
  it reads correctly for `tde_analytical`, which shares them.

### Fixed
- **GPU compile time of batched supernova, TDE and kilonova evaluations.** The Planck array now sits
  behind `jax.lax.optimization_barrier` before the band sum: `jit(vmap(predict_jax))` over 480
  parameter sets and 480 observations in six bands compiled in 563 s on a GPU and compiles in 1.2 s,
  at every batch size tried (to 4096). Values and gradients are unchanged (|dmag| below 1e-12); CPU
  was never affected.
- **A zero-temperature photosphere is zero flux, not NaN** (as in redback): the blackbody magnitude
  is `mag_floor`. The temperature is floored at 1 K inside the spectrum, which changes no physical
  temperature.
- **The float32 gradient of the kilonova magnitude with respect to a free redshift was NaN**, because
  XLA reassociated `(R / dl)^2` and the backward pass cubed `dl`; it is finite now.

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
