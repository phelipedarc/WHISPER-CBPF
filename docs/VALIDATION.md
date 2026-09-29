# Validation: what whisper gets right, what it does not yet, and how we know

This page reports the science-validation suite (`tests/science/`, `tests/test_acceptance.py`):
simulated LSST alerts whose answer is known, fitted and compared the way a user does it. Every number
below was measured on 2026-09-28 with the 0.2.0 development tree, redback 1.20.0 and JAX 0.11.2, on
one NVIDIA RTX A6000 per run and a 96-core host shared with other work (so CPU times are upper
bounds).

## Summary

| Check | Result | Verdict |
|---|---|---|
| Band magnitudes against redback (4 supernova families, LSST and ZTF, 8 prior draws each) | CPU vs redback's filter-integrated SED: max 3.9e-6 mag. CPU vs JAX: max 7.5e-14 mag on the CPU, 1.1e-14 mag on the GPU | pass |
| Pre-event rows never fitted (forced photometry and a detection before the event added to an alert) | every table number and every posterior draw of `compare` unchanged, bit for bit | pass |
| Upper limits fitted by default | an LSST alert's non-detections after the event enter the likelihood with no argument (censored flux likelihood, 5 sigma) | pass |
| Survey fields | ZTF `magpsf`/`sigmapsf`/`diffmaglim` and LSST `psfFlux`/`psfFluxErr` read exactly; `magpsf_corr` and `scienceFlux` never read | pass |
| Right class first at good SNR (supernova / TDE / kilonova) | 22 of 24 alerts, every graded winner "decisive"; both misses are TDE alerts on which the TDE and Arnett fits failed to start, a start failure since fixed (section 4.1; this study was not rerun) | pass |
| Few detections | all 9 alerts cut at their third detection: "not enough data", no model ranked | pass |
| Ranking from the optimised likelihood maximum vs the sampler's best draw | same winner in every seed. For a model with one peak the optimised max ln L is the same in all 6 seeds (to 1e-4) while the best draw varies by up to 0.76 ln L; the best draw falls short by 0.3-1 ln L on average for the larger models (0.005 for Arnett) | pass |
| Same fit on CPU and GPU | the same best ln L to 0.007; posterior medians within 1.5 Monte Carlo standard errors in one session, 4.2 on the unconverged nickel-mass ridge in another (section 6) | pass, marginal |
| Same seed, same answer | CPU: identical draws. GPU: the draws differed between two runs in one of two sessions; identical with `XLA_FLAGS=--xla_gpu_deterministic_ops=true` | **fail** by default |
| Posterior calibration, kilonovae (one and two components) | 68 % intervals hold the truth 74 % and 74 % of the time, 95 % intervals 93 % and 95 %; SBC ranks uniform (p = 0.03, 0.24) | pass |
| Posterior calibration, supernovae and TDE, default settings | 68 % intervals hold the truth 62-71 % of the time for Arnett, shock cooling + Arnett, CSM + Arnett and the TDE, 95 % intervals 87-89 % (should be 68 % and 95 %); the magnetar 48 % and 73 % | **fail** (95 % intervals slightly narrow; the magnetar far off) |
| Explosion-time prior when the explosion date is unknown | the default window (last non-detection to first detection) excludes the true explosion in 24 of 30 alerts; a 30-day window restores 57 % / 87 % coverage of the explosion time | **fail** |
| Default sampler start (`init="prior_scan"`) | all walkers on one point, and emcee refused to run, in 27 of 503 fits; since the fix, both reproducers start and none of 290 recovery fits failed to start | pass |
| Speed, per alert | 5-family comparison: 400 s on one GPU, 29-32 min on the CPU; `fit_batch` of one model over 64 alerts: 3.0 s per alert on one GPU | recorded, 2x regression gate |
| Six lines from an LSST alert to a ranked table and an HTML report | runs in a fresh Python process | pass |

**What to take from it.** The light curves, the photometry and the data rules are right, and at good
signal-to-noise the true class of transient (supernova, TDE or kilonova) wins. Two things are not
right yet: the posterior error bars of the supernova and TDE families at the default settings are
somewhat too narrow at 95 % (and the magnetar's clearly so); and an unknown explosion date gets a
prior window that usually excludes the truth, which can also make the wrong supernova family win.
Section 9 says what to do meanwhile.

---

## 1. Simulated LSST alerts

`tests/science/_sim.py` makes alerts whose answer is known:

- **Cadence**: the wide-fast-deep pattern, two visits per night in two of g, r, i, z, nights 2-4 days
  apart (a kilonova gets nightly follow-up; the model-selection study uses visits every 1-2 days for
  every class, so the cadence does not give the class away).
- **Depths**: LSST single-visit 5-sigma depths (Ivezic et al. 2019, table 2) with 0.2 mag of scatter.
- **Noise**: LSST's photometric error model (Ivezic et al. 2019, eqs. 4-5): background-limited for a
  faint source, source noise and a 0.005 mag floor for a bright one.
- **Alerts**: a visit measured at 5 sigma or more is a diaSource; every visit has forced photometry.
  The result is a Rubin alert packet, read with `wp.load_lightcurve(packet, survey="lsst")`.

With this noise the censored flux likelihood whisper fits by default is the exact likelihood of the
data, so a calibrated sampler must reach nominal coverage.

## 2. Model light curves against redback

`test_science_redback.py`: for `arnett`, `basic_magnetar_powered`, `shock_cooling_and_arnett` and
`csm_shock_and_arnett`, 8 draws of redback's own prior at an LSST cadence (z = 0.1) and a ZTF cadence
(z = 0.05). The reference is redback's SED integrated over the sncosmo bandpass on a 1 A grid,
computed without whisper's photometry code.

| family | survey | points | CPU vs reference, max (mag) | CPU vs JAX, max (mag), JAX on the GPU |
|---|---|---|---|---|
| arnett | LSST | 171 | 3.2e-8 | 1.1e-14 |
| arnett | ZTF | 319 | 3.9e-6 | 1.1e-14 |
| magnetar | LSST | 85 | 2.3e-8 | 7.1e-15 |
| magnetar | ZTF | 79 | 2.1e-7 | 7.1e-15 |
| csm + Arnett | LSST | 385 | 1.1e-7 | 1.1e-14 |
| csm + Arnett | ZTF | 361 | 2.3e-6 | 1.1e-14 |
| shock cooling + Arnett | LSST | 106 | 3.0e-8 | 1.1e-14 |
| shock cooling + Arnett | ZTF | 315 | 5.9e-7 | 1.1e-14 |

Gates: 2e-5 mag against the reference, 1e-10 mag between CPU and JAX. Both hold with orders of
magnitude to spare. A larger run of the same comparison (200 draws per family, every family whisper
binds) also shows that `type_1a` in LSST g reaches 3.6e-5 mag, and that the JAX Gaussian-rise TDE
differs from the time-dilation-corrected redback model by up to 0.1 mag at the faint end (median
3e-5 mag).

## 3. The alert rules, end to end

`test_science_alert_rules.py` runs `wp.compare` on simulated alerts:

- **Pre-event rows**: forced photometry before the explosion and a 6-sigma detection at the last
  visit before it are added to an alert. The comparison table (every number) and every posterior
  draw of every model are identical, bit for bit, at the same seed; each fit records the rows it
  left out. The kilonova version (slow) does the same with pre-merger rows.
- **Upper limits**: every non-detection after the event is a data point
  (`GaussianLikelihoodWithUpperLimits`, flux space, the preset's 5 sigma); dropping them changes the
  peak likelihood.
- **Survey fields**: a ZTF alert packet with a decoy `magpsf_corr` 2 mag off, and an LSST packet
  with a decoy `scienceFlux` 50 times `psfFlux`, load to exactly `magpsf` and
  `31.4 - 2.5 log10(psfFlux)` and go straight into `compare`.

## 4. Posterior calibration (known-answer recovery)

`test_science_recovery.py`: parameters drawn from each family's own prior (inside its constraint
walls), observed as an LSST alert with the explosion (or merger) date and the redshift known, fitted
with `wp.fit(..., sampler="auto")` (`emcee_jax` on a GPU, 32 walkers x 5 000 steps, 1 000 burn-in,
the walkers moving in each prior's own coordinate, `walker_coordinates="own"`). Coverage is pooled
over parameters; the band is 3 binomial standard errors; SBC is the rank of the truth among 49
posterior draws, tested for uniformity (chi-square p).

| family | settings | injections (failed) | 68 % intervals hold the truth | 95 % intervals hold the truth | SBC p | converged | median fit (s) |
|---|---|---|---|---|---|---|---|
| kilonova, one component | default | 30 | **74 %** (55-81 % expected) | **93 %** (89-100 %) | 0.03 | 30 of 30 | 27 |
| kilonova, two components | default | 30 | **74 %** (59-77 %) | **95 %** (91-99 %) | 0.24 | 0 of 30 | 32 |
| Arnett | default | 30 | **62 %** (58-78 %) | 87 % (90-100 %) | 0.01 | 0 of 30 | 23 |
| magnetar | default | 30 | 48 % (59-77 %) | 73 % (91-99 %) | 1e-8 | 0 of 30 | 25 |
| shock cooling + Arnett | default | 30 | **65 %** (60-76 %) | 88 % (91-99 %) | 0.03 | 0 of 30 | 30 |
| CSM + Arnett | default | 30 | **63 %** (60-76 %) | 89 % (91-99 %) | 0.07 | 0 of 30 | 29 |
| TDE (Gaussian rise) | default | 20 | **71 %** (56-80 %) | 87 % (89-100 %) | 0.92 | 0 of 20 | 137 |
| Arnett | 20 000 steps, 5 000 burn-in | 30 | **66 %** (58-78 %) | 88 % (90-100 %) | 0.66 | 0 of 30 | 73 |
| Arnett | walkers in linear coordinates (`walker_coordinates="linear"`) | 30 | 53 % (58-78 %) | 73 % (90-100 %) | 1e-8 | 0 of 30 | 29 |
| Arnett | linear coordinates, 20 000 steps, 5 000 burn-in | 30 | 59 % (58-78 %) | 83 % (90-100 %) | 3e-4 | 0 of 30 | 112 |
| Arnett | linear coordinates, start from prior draws (`init="prior"`) | 30 | 49 % (58-78 %) | 82 % (90-100 %) | 2e-6 | 0 of 30 | 26 |

With the walkers in linear coordinates the same injections gave, at 68 % / 95 %: magnetar 47 / 66,
shock cooling + Arnett 60 / 85, CSM + Arnett 49 / 72, TDE 55 / 79 (17 of 20 fitted: 3 failed to
start), kilonovae 74 / 93 and 70 / 96.

Per parameter at the default settings (68 % / 95 % coverage):

- **Arnett**: f_nickel 67/83, mej 63/87, vej 60/93, kappa 47/83, kappa_gamma 53/77,
  temperature_floor 80/97.
- **magnetar**: p0 33/67, bp 27/50, mass_ns 60/77, theta_pb 27/43, mej 47/73, vej 60/87, kappa 57/83,
  kappa_gamma 50/80, temperature_floor 73/97.
- **shock cooling + Arnett**: log10_mass 73/93, log10_radius 77/97, log10_energy 67/93, nn 73/97,
  delta 73/93, f_nickel 63/80, mej 57/83, vej 57/87, kappa 57/77, kappa_gamma 50/80,
  temperature_floor 67/93.
- **CSM + Arnett**: mej 63/80, f_nickel 60/87, csm_mass 57/83, v_min 77/100, beta 60/93, kappa 53/80,
  shell_radius 60/90, shell_width_ratio 67/97, kappa_gamma 53/80, temperature_floor 83/100.
- **TDE**: peak_time 60/90, sigma_t 60/90, mbh_6 75/90, stellar_mass 80/85, eta 80/90, alpha 75/80,
  beta 70/85.
- **kilonova, one component**: mej 77/87, vej 63/93, kappa 77/97, temperature_floor 80/97.
- **kilonova, two components**: every parameter 67-77 / 90-100.

**Reading it.**

- Moving the walkers in each prior's own coordinate (log for a LogUniform parameter, with the
  Jacobian) is what changed the supernova and TDE rows: the stretch move is affine-invariant only in
  the coordinates it moves in, and the nickel-mass / ejecta-mass / diffusion-time ridge along which
  the Arnett posteriors lie is straight in the logs, not in the parameters. In linear coordinates
  neither 4 times longer chains nor independent prior-draw starts helped (the last three rows).
- The 68 % intervals are now calibrated or nearly so for every family but the magnetar; the 95 %
  intervals still hold the truth only 87-89 % of the time for the supernovae and the TDE. 4 times
  longer chains move Arnett from 62 / 87 to 66 / 88: the tails converge slowly.
- The magnetar is not calibrated: `p0`, `bp` and `theta_pb` (27-33 % at 68 %) trade off along
  ridges bounded by its rotational-energy wall, where the walkers do not mix in 5 000 steps.
- The kilonovae are calibrated at the default settings.
- The true parameters score about as well as the best draw (the truth beats the best draw by more
  than 1 ln L in at most 3 of the injections of any family), so the fits find the right region;
  what is off is the width of the posterior.
- `diagnostics()` fails every supernova fit at these settings (autocorrelation length, R-hat, ESS)
  and says how many steps it wants: the report is honest about it.

### 4.1 The default start

`init="prior_scan"` (the default of `emcee_jax` and CPU `mcmc`) scores prior draws, climbs the best
and spreads the walkers around them. In 27 of the 503 fits made for sections 5.1 and 5.3 (all
`emcee_jax` on a GPU) it put all 32 walkers on exactly one point, and emcee refused to run
("Initial state has a large condition number"); in a comparison the model was then left out. It
happened most inside comparisons with the explosion time fitted over a narrow window (23 of 87
Arnett and TDE fits) and for a model far from the data (the Arnett model on a CSM light curve, 5 of
6 seeds), but also with the explosion date known (1 of 30 magnetar and 3 of 20 TDE injections).

The cause was an optimum pressed against a constraint wall (the magnetar's rotational-energy bound,
the TDE's `eta` bound): the curvature cannot be read across the wall, the spread was then set to
its widest, and every move went through the wall. The spread now gives such a coordinate a short
step and retries a refused move in a fresh direction and its mirror image, and a start that still
does not span every parameter is refused with the way out. The two reproducers
(`test_the_default_start_gives_the_walkers_room_to_move`, a TDE and a magnetar injection) now start
and run on the GPU, and the magnetar injection's 32 starts are distinct and span all 9 parameters.
None of the 290 fits of sections 4 and 4.2 failed to start. The model-selection studies of
section 5 were not rerun.

### 4.2 An unknown explosion date

Most alerts do not come with an explosion date. `compare` then fits the explosion time with the prior
`lc.explosion_time_prior()`: uniform from the last non-detection before the first detection to the
first detection. On 30 simulated Arnett alerts (the same simulation, explosion date not given):

| quantity | value |
|---|---|
| alerts whose true explosion lies inside the window | 6 of 30 |
| start of the window after the true explosion | median 4.3 d, 90th percentile 15.6 d |
| width of the window | median 2.9 d, narrowest 0.02 d |
| 68 % / 95 % intervals holding the true explosion time | 13 % / 20 % |
| 68 % / 95 % intervals holding the truth, all parameters | 41 % / 63 % (SBC p = 2e-14) |

A supernova is fainter than a 24.5 mag single-visit limit for days after it explodes, so the last
non-detection usually comes after the explosion (often on the same night as the first detection, in
another filter). The window then excludes the truth, the explosion time is forced late, and the other
parameters follow. The effect on a comparison is decisive: on one simulated Arnett alert the true
model loses to the shock-cooling and CSM models by more than 600 in BIC with the default window, and
wins with the explosion date known (section 5.3). On the bundled alert of section 8 (Arnett, 60
detections), the five-family comparison at the default settings ranks the shock-cooling model first
on the GPU and the CSM model first on the CPU; the window there starts 0.66 d after the true
explosion.

The same 30 alerts fitted with a window of the 30 days before the first detection (passed as the
prior) give 57 % / 87 % for the explosion time and 59 % / 84 % for all parameters (SBC p = 0.29):
the explosion-time bias is gone, and what remains is close to the sampler's under-coverage of
section 4 (62 % / 87 % with the date known). Pass such a window yourself (section 9).

## 5. Model selection

### 5.1 The right class at good SNR

`test_science_selection.py`: alerts simulated from Arnett, the Gaussian-rise TDE and the
one-component kilonova, on one cadence (visits every 1-2 days from 10 days before to 60 days after the
event), at least 12 detections in 3 filters, redshift known, explosion time unknown (the default
binding), compared with `wp.compare(lc, ["arnett", "tde", kilonova_model])`.

| simulated from \ ranked first | Arnett | TDE | kilonova |
|---|---|---|---|
| Arnett (8 alerts) | **8** | 0 | 0 |
| TDE (8 alerts) | 0 | **6** | 2 |
| kilonova (8 alerts) | 0 | 0 | **8** |

Every winner ranked against another model was graded "decisive". The two TDE alerts won by the
kilonova are not ranking errors: on both, the TDE and the Arnett fits failed to start (all walkers
on one point, a failure since fixed: section 4.1), so the kilonova was the only model left to rank, and the comparison says
so ("the only model ranked"). The same start failure left out a model in 9 more of the 24
comparisons without changing the winner. A comparison took 5-7 minutes on a GPU shared with
other runs.

### 5.2 Few detections

The same 9 alerts (3 per class) cut at their third detection: in all 9 every model was left out
with "not enough data: k free parameters >= n points", and the headline reads "not enough data --
no model could be ranked". No number is printed where the data cannot support one.

### 5.3 Why the ranking uses the likelihood maximum, measured

BIC is defined at the maximum of the likelihood. The traditional shortcut takes it from the best
posterior draw, which is always below the peak by an amount that depends on the model and on the run
(`docs/LSST_ALERTS.md`, section 5). `compare` climbs to the peak (`wp.likelihood_max_opt`) instead. The same alert
was compared with six seeds (explosion date known, the four supernova families, default settings);
the table gives, for each model, how much its ln L varies from seed to seed (range over the seeds)
and how far the best draw falls below the peak on average (the gain).

| simulated from | model (free parameters) | best draw: range over seeds | max ln L (optimised): range over seeds | mean gain |
|---|---|---|---|---|
| Arnett | Arnett (6) | 0.012 | 0.0001 | 0.005 |
| Arnett | magnetar (9) | 0.76 | 0.0001 | 0.36 |
| Arnett | shock cooling + Arnett (11) | 0.82 | 0.92 | 0.43 |
| Arnett | CSM + Arnett (10) | 1.94 | 2.90 | 0.97 |
| CSM + Arnett | magnetar (9) | 1.34 | 0.0000001 | 0.70 |
| CSM + Arnett | CSM + Arnett (10) | 0.60 | 0.42 | 0.42 |
| CSM + Arnett | shock cooling + Arnett (11) | 0.69 | 0.51 | 0.34 |

What this shows:

- **The shortcut penalises the larger models twice.** The best draw falls 0.005 ln L short for the
  6-parameter Arnett model and 0.3-1 ln L short for the 9-11-parameter families, so BIC from the
  best draw adds 0.6-2 to the larger models' BIC on top of BIC's own `k ln n`. On short chains it is
  worse: `docs/LSST_ALERTS.md` (section 5) shows a 300-step run whose best draws graded a comparison
  "decisive" where the peaks say "strong". (On the CSM alert the Arnett fit failed to start in 5 of
  6 seeds, section 4.1, so Arnett is not in that half of the table.)
- **For a model with one peak the optimised maximum is exact and the same in every run** (Arnett and
  magnetar: identical to 1e-4 over six seeds, while the best draw moved by up to 0.76). That noise is
  gone from the ranking.
- **The maximum-likelihood optimisation is a local climb.** The shock-cooling and CSM families have
  several local peaks of nearly equal height; each run climbs to the one its best draws were near,
  so their optimised maxima still vary by 0.4-2.9 ln L. The winner was the same in every seed either way here (its lead was 11 in
  BIC); the order of the models behind it was not.

With 1 000-step chains (6 seeds, the Arnett alert) the best draws fell further short (mean gain 0.6
ln L for the magnetar, shock-cooling and CSM families); the magnetar's best draw varied by 0.80 ln L
and its optimised likelihood maximum by 0.17.

## 6. CPU and GPU

`test_science_devices.py`: one simulated LSST supernova alert, `emcee_jax` at its default settings,
twice per device, each device in a fresh process.

| | CPU (JAX) | GPU | GPU, deterministic XLA ops |
|---|---|---|---|
| two runs at the same seed | identical | **differ** in one of two sessions | identical |
| wall per fit (s, shared machine) | 114-118 | 17-24 | 29-33 |
| largest median difference from the CPU, in Monte Carlo standard errors | - | 1.5 and 4.2 in two sessions | 2.4 |
| best ln L | 768.5782 | 768.5772 and 768.5819 | 768.5748 |

The 4.2 is on `f_nickel` and `kappa`, the nickel-mass ridge of section 4, along which these chains
are not converged; its Monte Carlo error, estimated from the effective sample size of an
unconverged chain, is too small. The physics and the density agree on the two devices to round-off
(section 2).

On the GPU the density itself repeats exactly; what differs is the start, whose gradient climb uses
GPU reductions that are not bitwise reproducible (the best start differs at 5e-11 ln L, and the
chains then part). `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes the whole fit reproducible at
about 35 % more time.

## 7. Speed

`test_science_speed.py` records these and fails when a new measurement is more than twice its
budget (scale with `WHISPER_SPEED_SCALE` on other hardware).

The alert is the bundled one of section 8 (74 rows, 60 fitted), explosion date unknown, the default
settings, `evidence_check=False` (when the nested-sampling check fires it adds CPU time of its own,
about 20 minutes on a 34-point alert).

| what | where | wall | of which |
|---|---|---|---|
| `compare` of 5 families (Arnett, magnetar, shock cooling + Arnett, CSM + Arnett, TDE), first call | one RTX A6000 | **400 s** | fits 25-31 s per supernova family and 132 s for the TDE; likelihood maximum 7-13 s per supernova family and 118 s for the TDE |
| the same, second call in the same process | one RTX A6000 | **392 s** | nothing compiled is reused between calls |
| the same | CPU (`sampler="auto"` is CPU `mcmc`), two runs | **1 723 s and 1 938 s** | fits 250-400 s per family; likelihood maximum 12-73 s |
| `fit_batch` of 64 alerts (Arnett, free explosion time, 32 walkers x 5 000 steps) | one RTX A6000 | **194 s, 3.0 s per alert** | compile 27 s, starts 11 s, chains 65 s (1.3 s per alert) |
| one Arnett fit, 32 walkers x 1 000 steps, CPU: `mcmc` against `emcee_jax` | CPU | 82 s against 51 s | the TDE: 98 s against 50 s |

- The TDE is 63 % of a GPU comparison (its fit and its likelihood maximum).
- On the CPU, `emcee_jax` (the JAX density vectorised over the walkers) is 1.6-2 times faster than
  `mcmc`, which `sampler="auto"` picks there.
- `fit_batch` is the way to run one model over many alerts: 3 s per alert against 30-45 s for one
  supernova fit and its likelihood maximum in `compare`. It does not optimise to the likelihood
  maximum or rank.

## 8. Six lines, from an alert to a report

`tests/test_acceptance.py` runs, in a fresh Python process with nothing configured:

```python
import jax; jax.config.update("jax_enable_x64", True)
import whisper_cbpf as wp
lc = wp.load_lightcurve("tests/science/data/lsst_alert_sn.json", survey="lsst", redshift=0.1)
cmp = wp.compare(lc, ["arnett", "magnetar"])
print(cmp.summary())
cmp.report("out/")
```

and checks the printed ranking and the report (one self-contained HTML file with inline figures).
The fast version adds `nsteps=300, burnin=100` (about 165 s on the CPU, most of it the two fits and
their likelihood maxima); the slow one runs the defaults as written (85 s on one GPU). The first line is
needed: without float64 the supernova and TDE models refuse to run.

## 9. What this means for your science

- **Rankings**: trust the class-level answer (supernova vs TDE vs kilonova) at good SNR. Between
  supernova families, rank only with the explosion date known or with the wider explosion-time prior
  below. Read the grade together with `converged` and the gain of each likelihood maximum
  (`cmp.peaks[model].gain`).
- **A model left out because its start "does not span all k parameters"**: the default start could
  not spread the walkers. Fit that model again with `init="prior"` (none of the 30 Arnett fits
  started that way failed here).
- **Error bars of supernova and TDE parameters**: at the default settings the 68 % intervals are
  about right (62-71 %) but the 95 % intervals hold the truth only 87-89 % of the time; the
  magnetar's are too narrow at both levels (48 % and 73 %). Treat a 95 % interval as closer to a
  90 % one, and do not quote a magnetar interval as it is. A 4 times longer chain helps little
  (Arnett 87 → 88 %). If a number matters, compare it with a nested-sampling fit
  (`sampler="nested"`, independent draws; its calibration was not measured here) and quote the wider
  interval.
- **Kilonova parameters**: calibrated at the default settings.
- **Unknown explosion date**: give the explosion time a prior that starts well before the first
  detection. The default window excludes the truth in most LSST alerts; a 30-day window removed that
  bias (section 4.2). A `{"t_exp": ...}` entry in `compare(prior=...)` is passed to the fit as given:

  ```python
  t1 = lc.meta["first_detection_mjd"]
  window = wp.Prior({"t_exp": wp.Uniform(t1 - 30.0, t1)})
  cmp = wp.compare(lc, ["arnett", "magnetar"], prior={"arnett": window, "magnetar": window})
  ```
- **Reproducibility on a GPU**: set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` before starting
  Python if you need the same draws at the same seed.

## 10. How to rerun

The fast checks run with the normal suite:

```bash
python -m pytest tests/science tests/test_acceptance.py -m "not slow"      # ~4 minutes on a CPU
```

The studies need a GPU and hours; keep their records so a rerun resumes and the tables can be rebuilt:

```bash
export WHISPER_VALIDATION_OUT=validation_runs      # one JSON line per injection / comparison
python -m pytest tests/science -m slow             # all studies (about 10 GPU-hours)
python -m pytest tests/science/test_science_recovery.py -m slow -k "kilonova"
WHISPER_SCIENCE_N=10 python -m pytest tests/science/test_science_selection.py -m slow
```

`JAX_PLATFORMS=cpu` runs the CPU speed budget and skips the GPU studies. `WHISPER_SCIENCE_N` sets the
number of injections (or alerts per class); `WHISPER_SPEED_SCALE` scales the speed budgets.

The slow tests that fail today are the "fail" rows of the summary, each a reproducer:
`test_known_answer_recovery_at_the_default_settings` for Arnett, magnetar, shock cooling + Arnett,
CSM + Arnett and the TDE (their 95 % coverage is below its band);
`test_explosion_time_prior_from_the_data_window_covers_the_truth`; and
`test_the_gpu_repeats_itself_at_a_fixed_seed` (in one of two sessions). `test_the_default_start_gives_the_walkers_room_to_move`
(the two injections on which the start used to collapse) passes;
`test_cpu_and_gpu_posteriors_agree_within_monte_carlo_error` passed in one session of two (section
6). The model-selection (section 5) and speed (section 7) studies were measured with the walkers
in linear coordinates and have not been rerun with the default `walker_coordinates="own"`.
