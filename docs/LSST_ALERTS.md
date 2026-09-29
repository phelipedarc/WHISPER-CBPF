# From an LSST alert to a ranked answer

This page takes one alert from a broker to a ranked table of physical models and a one-file HTML
report, then explains how to read the answer, what to observe next, and what it costs per alert
on a CPU and on one GPU.

**Contents**

1. [Quickstart](#1-quickstart)
2. [Reading the alert](#2-reading-the-alert)
3. [Choosing the models](#3-choosing-the-models)
4. [Reading the answer](#4-reading-the-answer)
5. [Why the ranking uses the likelihood maximum](#5-why-the-ranking-uses-the-likelihood-maximum)
6. [What to observe next](#6-what-to-observe-next)
7. [Seconds per alert](#7-seconds-per-alert)
8. [Many alerts](#8-many-alerts)
9. [Saving, resuming, provenance](#9-saving-resuming-provenance)

---

## 1. Quickstart

```python
import jax; jax.config.update("jax_enable_x64", True)    # supernova and TDE models need float64
import whisper_cbpf as wp
lc  = wp.load_lightcurve("tests/data/lsst_alert_sn.json", survey="lsst", redshift=0.1)
cmp = wp.compare(lc, ["arnett", "magnetar"])
print(cmp.summary())
cmp.report("out/")
```

The file is a Rubin alert packet (a synthetic supernova shipped with the repository, written by
`tests/data/make_lsst_alert_sn.py`): 12 detections in LSST g, r and i, and three forced-photometry
epochs before the explosion. What the six lines do:

| line | result |
|---|---|
| `load_lightcurve(..., survey="lsst")` | 12 detections as AB magnitudes (`lsstg`, `lsstr`, `lssti`) and 3 upper limits at 5 sigma |
| `compare(lc, [...])` | each model fitted, its likelihood peak found, its convergence checked; the models ranked, weighed and graded |
| `cmp.summary()` | the winner, the grade, the table, the models left out and why, the caveats |
| `cmp.report("out/")` | `out/report.html`: one self-contained page (no network, no script), readable on a phone |

The summary on this alert (CPU, default settings; the caveat lines are shortened here):

```text
Comparison of 2 models on 12 points by BIC: winner 'arnett' (strong over 'magnetar', ln B = 3.46)

  rank  model     sampler  k   n   max ln L  BIC     ln Z  dBIC  weight  grade   converged
  1     arnett    mcmc     7   12  29.17     -40.95  -     0.00  0.970   strong  no
  2     magnetar  mcmc     10  12  29.44     -34.03  -     6.92  0.030   strong  no

Caveats:
  - the winner's fit ('arnett') did not pass its convergence report (its failed checks are listed
    below). Its likelihood maximum and BIC stand; do not quote its error bars from this run.
  - arnett: chain length / autocorrelation time: the chain is 10.8 autocorrelation times long
    (largest tau 461 steps), below 50: too short to trust. Raise nsteps to at least 23065; ...
  - magnetar: chain length / autocorrelation time: the chain is 8.6 autocorrelation times long
    (largest tau 580 steps), below 50: too short to trust. Raise nsteps to at least 29008; ...

Read next: .table (every number), .results[model].diagnostics(), .peaks[model], .report(path)
```

How to read it:

- The alert was simulated from `arnett`, and `arnett` wins, graded "strong" (ln B = 3.46, weight
  0.97). The magnetar fits the 12 points slightly better (max ln L 29.44 against 29.17) but spends
  three more parameters, and BIC charges `3 ln 12 = 7.5` for them.
- The 3 limits before the first detection were not fitted (`n = 12`); they set the explosion-time
  prior.
- Neither fit passed its convergence report at the default 5 000 steps: the chains are 9-11
  autocorrelation times long (the report asks for 50) and the walkers disagree (R-hat 2.5-3.1).
  The ranking rests on the likelihood maxima (section 5), which a longer run moves little. Do not
  quote the error bars from this run: `nsteps=25000` addresses the chain length, and
  [`VALIDATION.md`](VALIDATION.md) section 4 says how far the supernova intervals can be trusted
  even then.
- No evidence check ran: ln B = 3.46 is above its ln 10 trigger.

Omit `redshift=` when the host redshift is unknown: the redshift is then fitted (section 3).

---

## 2. Reading the alert

`load_lightcurve(x, survey="lsst" | "ztf", limits=None)` reads broker photometry in the brokers' own
field names and returns a magnitude-mode `LightCurve`, sorted by time. `x` can be a path (`.csv`, or
`.json`), an open file, a DataFrame, one record or alert packet, or a list of them:

- a Rubin alert packet: `diaSource` + `prvDiaSources` + `prvDiaForcedSources`;
- a ZTF alert packet: `candidate` + `prv_candidates`;
- ALeRCE's `{"detections": [...], "non_detections": [...]}`;
- Fink's CSV or JSON (the `i:`, `d:`, `r:` prefixes are removed);
- a table already in whisper's columns (`time, band, magnitude, magnitude_err, upper_limit`).

Field names match case-insensitively. Loading takes about 10 ms per object.

### Field mapping

| | `survey="lsst"` | `survey="ztf"` |
|---|---|---|
| Detection | diaSource `psfFlux`, `psfFluxErr` [nJy] | `magpsf`, `sigmapsf`, where `isdiffpos` is positive (`t`, `1`) |
| Magnitude | `m = 31.4 - 2.5 log10(psfFlux)`, `sigma_m = (2.5 / ln 10) psfFluxErr / psfFlux` | `magpsf` |
| Upper limit (5 sigma) | a forced-photometry epoch with no diaSource: `31.4 - 2.5 log10(5 psfFluxErr)` | a row with no `magpsf`: its `diffmaglim` |
| Where the limits come from | the packet's `prvDiaForcedSources`, or `limits=` (e.g. Fink's forced photometry) | the packet's `prv_candidates`, ALeRCE's `non_detections` |
| Time | `midpointMjdTai` (MJD, TAI) | `mjd`, or `jd - 2400000.5` (MJD, UTC) |
| Band | `band` -> `lsstu lsstg lsstr lssti lsstz lssty` | `fid` 1/2/3 or a band field -> `ztfg ztfr ztfi` |
| Never read | `scienceFlux` | `magpsf_corr` |
| Dropped and counted | `isNegative` or `psfFlux <= 0`, `psfFlux_flag`, withdrawn rows (`timeWithdrawnMjdTai`) | negative subtractions (`isdiffpos` `f`, `0`, `-1`), Fink `tag == "badquality"` |

Both presets also drop, and count, repeats of one exposure and an upper limit at a detection's
epoch (the same visit, or within 1e-4 d). The counts are in `lc.meta["n_dropped"]`; a warning is
raised when negative fluxes were dropped.

`lc.meta` records `survey`, `photometry`, `time_system`, `upper_limit_sigma` (5.0),
`n_dropped` and `first_detection_mjd`. LSST times stay on TAI, which is 37 s ahead of UTC: a light
curve that mixes ZTF and LSST points carries that offset.

**Band labels are always survey-prefixed** (`lsstg`, `ztfg`), never bare letters: a bare `g` names
no survey. `whisper_cbpf.io.SURVEY_BANDS` lists them. With `survey=`, `bands=["g", "r"]` means that
survey's g and r.

**Selections** run on the input's MJD clock, in this order: `bands=`, `time_min` / `time_max`,
`min_snr` (detections only; limits have no signal-to-noise and are kept), then `explosion_date`.

**Not enough data.** With no detection left, the loader raises "not enough data: no detection is
left ...", with the counts.

### The photometry rule: difference imaging only

An alert is a detection on a **difference image**: the science image minus a reference image of the
same field. The difference-image PSF flux (`psfFlux`, `magpsf`) is the transient's own light, with
the host galaxy subtracted. That is what every model predicts, so it is the only photometry whisper
fits.

- `scienceFlux` (LSST) is the flux on the direct image: transient plus host.
- `magpsf_corr` (ZTF) adds the reference image's flux back: a total magnitude, meant for variable
  stars.

Neither is ever read. When one of them is the only measurement in the input, the loader raises and
says which field it needs. A negative difference has no AB magnitude, so it is dropped and counted.

### Pre-event rows are never fitted

Rows at or before the event are left out of the likelihood and out of the model evaluation. They
can only set a prior. The event is:

| light curve and model | event | what happens to the earlier rows |
|---|---|---|
| a model that fits its explosion time (`t_exp`), such as every family `compare` binds to an alert | the first detection | the non-detections before it set the explosion-time prior |
| `lc.set_explosion_date(mjd)`, or `set_time_reference(mjd, "explosion")` / `"merger"` | that date | left out |
| a model whose `t_exp` is `Fixed(v)` | `v` | left out |
| another day 0 (`set_time_reference(mjd, "first detection")`), for a model whose clock starts there | day 0 | left out |

The **explosion-time prior** is `lc.explosion_time_prior()`: Uniform from the last non-detection
before the first detection to the first detection, every band counted. With no earlier
non-detection it falls back to the 30 days before the first detection
(`whisper_cbpf.io.schema.EXPLOSION_FALLBACK_DAYS`), and `compare` adds a caveat. A limit shallower
than the first detection does not prove the source was not there yet, and at LSST's single-visit
depth a supernova stays below the limit for days after it explodes: on simulated LSST alerts the
default window excludes the true explosion most of the time ([`VALIDATION.md`](VALIDATION.md)
section 4.2). Unless the limits before the first detection are deep, give the explosion time a
wider prior of your own; a `t_exp` prior passed this way is used as given, not cut to the window:

```python
t1 = lc.meta["first_detection_mjd"]
cmp = wp.compare(lc, ["arnett", "magnetar"],
                 prior={m: wp.Prior({"t_exp": wp.Uniform(t1 - 30.0, t1)})
                        for m in ("arnett", "magnetar")})
```

Every fit records what it left out: `result.info["excluded_pre_event"]` (a count),
`result.info["pre_event"]` (the rule), `result.info["t_exp_prior"]` (the prior used and where it
came from), and one warning says so. Adding pre-event rows to an alert changes the posterior, the
peak and the predictions by exactly zero. `result.fitted_lc(lc)` returns the rows a fit used.

On the quickstart alert, the 3 forced-photometry limits precede the first detection: they are left
out of both fits and set `t_exp ~ Uniform(60997.0, 61002.5)`.

### Upper limits

A light curve with upper limits after the event is fitted in flux space with the censored
likelihood, with no argument: a detection contributes a Gaussian term, a limit the probability that
the flux was below it. The limits' significance is `lc.meta["upper_limit_sigma"]` (5 for both
presets). The likelihood samplers (`mcmc`, `emcee_jax`, `nuts_gpu`, `nested`) take limits; ABC and
SNPE compare values point by point and refuse them by name. `lc.where(upper_limit=False)` keeps the
detections only.

---

## 3. Choosing the models

`compare` takes registered model names, `Model` objects, and **family names that it binds to the
alert**:

| name | model |
|---|---|
| `"arnett"` | nickel-powered supernova |
| `"magnetar"` | magnetar-powered supernova |
| `"csm_shock_arnett"` | CSM shock breakout plus nickel |
| `"shock_cooling_arnett"` | shock cooling plus nickel |
| `"tde"`, `"tde_gaussianrise"` | tidal disruption event: Gaussian rise onto a cooling envelope |
| any name in `wp.supernova_models()` | that supernova model |

A bound family:

- uses the alert's bands;
- **fixes the redshift** at `lc.redshift` when it is known, and **fits it** from
  `lc.redshift_prior` (Uniform(0.001, 1) by default) when it is not;
- **fits the explosion time** over `lc.explosion_time_prior()`, unless day 0 is the explosion;
- uses the model's default prior (redback's) for everything else. `prior=` overrides named
  parameters, per model: `prior={"arnett": wp.Prior({"f_nickel": wp.Uniform(0.01, 0.5)})}`. A
  `t_exp` entry is used as given (section 2); for a supernova family it needs a finite lower bound
  (a `Uniform`, or a `TruncatedNormal` with a finite `low`).

**A fitted redshift can bias the comparison.** A model can move the source in distance to buy a fit
it cannot make at the true distance, and BIC charges every model the same for that freedom. Pass a
host redshift when there is one. When the redshift is fitted, the summary says so, and the facts
flag each redshift posterior that only restates its prior (section 4).

**Few points.** A model with at least as many free parameters as fitted points is left out ("not
enough data"), and so is any model fitted to a different number of points than the rest (only fits
of the same data are ranked). With the redshift known and the explosion time fitted, `arnett` has
7 free parameters, `magnetar` 10 and the TDE 8; with the redshift fitted, one more each. An alert
with 7 fitted points therefore cannot rank `arnett`, and says so instead of printing a number.

**Cost.** The TDE is the most expensive family: on one GPU it compiles for 30-45 s and one
evaluation costs about 20 times a supernova's when evaluated alone (section 7).

---

## 4. Reading the answer

### The ranking

`cmp.table` has one row per model, ranked models first, best first:

| column | meaning |
|---|---|
| `n_params`, `n_data` | free parameters, fitted points |
| `max_log_likelihood`, `aic`, `bic` | at the optimised likelihood maximum (section 5) |
| `log_evidence`, `log_evidence_err` | ln Z, when a nested-sampling run exists for the model |
| `delta` | BIC minus the best BIC (or best ln Z minus ln Z) |
| `weight` | `exp(-delta / 2)` (or the evidence weight), normalised over the ranked models |
| `grade` | Jeffreys' scale on ln B = `delta / 2`: below 1.15 "inconclusive", below 2.30 "substantial", below 4.61 "strong", above "decisive". The winner's row is graded against the runner-up, every other row against the winner |
| `converged` | the fit's convergence report passed |
| `problems` | one sentence per failed check or caveat |
| `status`, `left_out_reason` | "ranked", or "left out" with the reason |

`cmp.winner` is the best model, or `None` when nothing could be ranked. `cmp.criterion` is `"BIC"`,
or `"ln Z"` when every ranked model has a nested-sampling evidence that converged
(`wp.compare(..., sampler="nested")`, or the evidence check below when only two models are ranked).

**The evidence check.** When the top two are within ln B < ln 10 (a BIC gap under 4.61),
`evidence_check="auto"` runs nested sampling on those two. If ln Z disagrees with BIC, the winner's
grade becomes "inconclusive" and the summary says why. On a CPU it costs 8-12 minutes per supernova
family on a 34-point alert, so pass `evidence_check=False` in a stream and run it on the alerts that
matter, or pass `cache_dir=` to pay it once.

### The convergence report

`cmp.diagnostics[model]` (or `result.diagnostics()` for a single fit) is a `DiagnosticsReport`: one
row per check, with its value, threshold, pass or fail and a plain reason, and a "Why" section.

| sampler | checks |
|---|---|
| every fit | posterior draws exist; more data points than free parameters; at most 5 % of the draws in the outer 1 % of a prior range |
| emcee (`mcmc`, `emcee_jax`) | no stuck walker; `nsteps / largest tau >= 50`; R-hat across walkers < 1.05; bulk and tail ESS >= 400 |
| NUTS (`nuts_gpu`, `pymc_jax_gpu_*`) | no divergence; R-hat < 1.01 on every parameter and on the log-likelihood; ESS >= 100 per chain; no stranded or frozen chain; no prior-scan optimum 10 nats above every chain |
| with an optimised likelihood maximum | the sampler's best draw within 5 nats of the peak |

A fit that fails its report keeps its likelihood maximum and BIC (the ranking stands); do not quote its
error bars from that run. A longer run (`nsteps=`, passed through `compare`) fixes a report that
failed only on chain length; it does not make an error bar calibrated
([`VALIDATION.md`](VALIDATION.md) section 4 says which families' intervals are).

### The facts

`cmp.facts()` returns every number a reader quotes, each computed by a stated rule, as JSON-ready
data sealed with a SHA-256 (`wp.write_facts(facts, "out/")` writes `facts.json`):

- `ranking`: the criterion, winner, runner-up, ln B and grade, and per model its delta, weight and
  grade, and the models left out with the reason;
- `models[name]["parameters"]`: per parameter the median and 16th/84th percentiles, the posterior
  68 % width over the prior 68 % width, `prior_dominated` (ratio above 1/sqrt(2): the data barely
  narrowed it) and `at_prior_edge`;
- `models[name]["narrowing"]`: how much the redshift and the explosion time narrowed;
- `data`: per band the brightest detection, rising, peaked or declining, rise and decline rates with
  errors, the latest colours with errors, the last non-detection before the first detection;
- `absolute_magnitude`: with its asymmetric range from the redshift;
- `caveats`: `winner_converged`, `winner_prior_dominated_parameters`, `evidence_check_disagrees`,
  `models_left_out`, and `models_with`, the models that raised each flag (`not_converged`,
  `stranded_walkers`, `not_enough_data`, `large_likelihood_max_gain`, `prior_dominated`,
  `at_prior_edge`);
- `thresholds`, `rules` (in words) and `inputs` (hashes), so every flag can be recomputed.

The report shows the same facts as tables.

### The report

`cmp.report("out/", forecast_times=[t1, t2])` writes `out/report.html`: the answer and its caveats,
the ranking, the figures (every model over the data with residuals, the weights, posterior width
over prior width, the winner's corner plot, the forecast), the parameters, what the data show, every
diagnostic, the forecast tables and the provenance, with `facts.json` embedded as a download. Every
number on the page comes from the facts, and the same inputs give the same bytes. A figure or
forecast that cannot be made is left out, with its reason on the page.

---

## 5. Why the ranking uses the likelihood maximum

BIC and AIC are defined at the **maximum** of the likelihood:
`BIC = k ln n - 2 ln L_max`. The usual shortcut takes `ln L_max` from the sampler's best posterior
draw. `compare` does not: it climbs from the best draws to the peak of the same likelihood
(`wp.likelihood_max_opt`, a maximum-likelihood optimisation) and computes BIC there.

**The best draw is always below the peak, by an amount that depends on the model.** A sampler is
built to map the posterior, not to find its summit. Near the peak, `ln L_max - ln L` of a posterior
draw behaves like half a chi-square with `k` degrees of freedom, so the best of `N` independent draws
falls short by an amount that grows with the number of parameters. Simulated, for an ideal Gaussian
posterior:

| free parameters `k` | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| best of 400 independent draws, mean shortfall (ln L) | 0.06 | 0.24 | 0.50 | 0.83 |
| best of 4 000 independent draws | 0.02 | 0.10 | 0.27 | 0.50 |

So, even for a perfectly mixed chain, taking BIC from the best draw penalises the model with more
parameters a second time, by up to about 1.5 in BIC (0.83 ln L at `k = 10` against 0.06 at
`k = 4`). Real chains do worse: their draws are correlated, a peak on a prior edge is rarely
visited, and a chain that has not converged may never have reached the peak at all.

**Measured on the quickstart alert** (simulated from `arnett`; `arnett` has 7 free parameters
here, `magnetar` 10). The same comparison at two chain lengths, ranked from the likelihood maxima and
from the samplers' best draws:

| chain | gain of the maximum, `arnett` / `magnetar` (ln L) | ln B, `arnett` over `magnetar`: likelihood maxima | ln B: sampler's best draws |
|---|---|---|---|
| 24 walkers x 300 steps | 0.49 / 2.45 | 3.63, strong | 5.60, **decisive** |
| default, 28 / 40 walkers x 5 000 steps | 0.20 / 0.23 | 3.46, strong | 3.49, strong |

From the likelihood maxima the answer barely moves with the chain length (3.63 against 3.46). From the
best draws, the short run misses the magnetar's peak by 2.45 ln L and `arnett`'s by only 0.49, so it
adds 3.9 to the BIC gap and grades the comparison one step too high. The shortcut's error is the
difference between two models' shortfalls, and nothing in the sampler's output reports it.

**Measured on a real alert**, redback `arnett` on SN2025pgp's 30-day ZTF cut:

| fit | sampler's best ln L | max ln L (optimised) | BIC error of the shortcut |
|---|---|---|---|
| `emcee_jax`, 32 walkers x 600 steps | 28.80 | 31.49 | 5.4 |
| CPU `mcmc`, 1.8 million evaluations | 31.38 | 31.49 | 0.22 |

A BIC error of 5.4 is more than a whole grade of Jeffreys' scale. On other real alerts, a fit that
stopped 1.8 ln L short of its peak swapped the top two models, and another turned "decisive" into
"strong". Optimising to the likelihood maximum removes that sampler noise from the ranking.

**What the optimisation changes, and what it does not.**

- It changes `max_log_likelihood`, AIC and BIC in the table: they are the peak's.
- It does not change the posterior: samples, medians and error bars are the sampler's.
- It is never lower than the sampler's best draw (the best draw is one of its starts, and every
  evaluated point is kept).
- It climbs inside the prior box in each prior's own coordinate (log10 for LogUniform), so a peak on
  a prior edge is reached exactly; `peak.at_edge` names such parameters.
- It costs a few seconds to a few minutes per model, compile included: 13 s (`arnett`) and 37-43 s
  (`magnetar`) on the quickstart alert on a CPU; on one GPU, 4 s for `arnett` on SN2025pgp and up to
  2 minutes for the other supernova families and 4 minutes for the TDE on real alerts. Its compiled
  programs are reused between optimisations of the same density.

**The gain is a diagnostic.** `cmp.peaks[model].gain` is the peak minus the sampler's best draw, and
`sampler_log_likelihood` is the number the shortcut would have used. A gain above 1 ln L is listed
in `problems` ("the sampler never reached the peak") and in the facts' `large_likelihood_max_gain`; the
convergence report fails a fit whose best draw is more than 5 ln L below the peak. A large gain says
the posterior itself may be incomplete, and a longer run is needed for the error bars.

**What the optimisation cannot do.** It is a local climb. A chain stuck in the wrong mode stays
there: a short run started from plain prior draws that found ln L 3.8 on SN2025pgp climbed to 4.1,
against a peak of 31.49. That is why every
chain starts from a scan of the prior, every fit carries its convergence report, and a small BIC
gap triggers the nested-sampling evidence check, which needs no peak at all.

To rank the traditional way, pass `likelihood_max_opt=False`: the table then uses the samplers' best draws, and
the summary says the ranking carries sampler noise.

---

## 6. What to observe next

**Forecasts.** `cmp.forecast(times, bands, survey_depth=...)` predicts AB magnitudes at future
(time, band) cells from each ranked model's posterior draws; `result.forecast(...)` and
`wp.forecast(result, ...)` do the same for one fit. Every combination of `times` x `bands` is a cell.

```python
import numpy as np
t_last = float(lc.time.max())
fc = cmp.forecast(t_last + np.array([1.0, 3.0, 7.0]), ["lsstg", "lsstr"],
                  survey_depth={"lsstg": 24.5, "lsstr": 24.2})
fc[["model", "time", "band", "q50", "q16", "q84", "frac_too_faint"]]
```

Each row has the mean and spread, the 2.5/16/50/84/97.5 percentiles, the fraction of draws fainter
than the depth, and the fraction with no light. `wp.plot_forecast` draws one model's rows.

**Where the models differ most.** `cmp.discriminate(times, bands, survey_depth=...)` scores, for the
top two models and every cell, how well one more measurement there would tell them apart:

`D = |mean_a - mean_b| / sqrt(sd_a^2 + sd_b^2 + sd_phot^2)`

where `sd_phot` is the photometric error expected at that magnitude for the given depth (0.217 mag
at the 5-sigma depth). D near 1 or below: one measurement cannot separate the models; 3 or more: it
can. `.attrs["best"]` is the observable cell with the largest D, and `wp.report(...,
forecast_times=...)` shows the five best cells.

On the quickstart alert (a 40 walkers x 300 steps comparison), the best cell over the next week is
LSST g, 7 days after the last point, with D = 0.47: the two models predict g = 21.97 and 21.88 with
spreads of 0.16 and 0.11 mag. No single point in that week separates them; the ranking rests on the
data already in hand.

A JAX model's forecast runs as one compiled batch: 400 draws x 15 cells take about 1 ms of model time
after a compile of about 1.5 s on one GPU, and agree with redback's CPU forecast to round-off.

---

## 7. Seconds per alert

**On a CPU.** The quickstart as written (12 fitted points, `arnett` and `magnetar`, redshift known,
free explosion time, default settings, CPU `mcmc`), three runs in one process each on a shared
2 x Intel Xeon Gold 5318Y host:

| step | seconds |
|---|---|
| `import`, then `load_lightcurve` | 3-5, then 0.01 |
| `arnett`: fit (28 walkers x 5 000 steps, prior-scan start), then the likelihood maximum | 209-258, then 13-17 |
| `magnetar`: fit (40 walkers x 5 000 steps), then the likelihood maximum | 275-331, then 19-43 |
| `report` | 4-5 |
| **one alert, two models** | **544-636** |
| evidence check, when the top two are close | 8-12 minutes more per model (nested sampling) |

About 9-11 minutes per alert for two supernova families, and neither chain long enough for error
bars (section 1): a converged run needs about five times the steps. On a CPU, rank with the defaults
and rerun the alerts that matter longer.

**On one GPU.** Measured on one NVIDIA RTX A6000 in float64 (`sampler="auto"` picks `emcee_jax`
when JAX sees a GPU), on real ZTF and LSST alerts of 13-59 points:

| step | seconds per alert |
|---|---|
| `compare` of five families (`arnett`, `magnetar`, `csm_shock_arnett`, `shock_cooling_arnett`, `tde_gaussianrise`), 60 walkers x 3 000 steps, no evidence check | 515-562 |
| of which one supernova fit / its likelihood maximum | 22-32 / 7-115 |
| of which the TDE fit / its likelihood maximum | 101-121 / 96-235 |
| the evidence check, when it runs (CPU nested sampling, top two) | about 1 200 more |
| `fit_batch`, free `arnett`, 60 x 10 000, 512 alerts per call | 5.3 chain + 0.3 start |
| `fit_batch`, TDE, 60 x 10 000, 512 alerts per call | 3.3 chain + 0.25 start |
| `fit_batch`, one alert alone (`arnett` / TDE) | 10.7 / 53.7 |
| `abc_gpu`, 3 million draws, `arnett` | 7.7 |
| `forecast`, 400 draws x 15 cells, after compile | 0.001 |

The single-alert path (`compare`) spends about half its time finding the likelihood maxima, whose
programs compile for
every new alert; the batched path compiles once per model and bucket of light-curve length and
shares it across alerts. Section 8 turns these numbers into alerts per night.

---

## 8. Many alerts

**One model, many alerts, one GPU call.** `wp.fit_batch(lcs, model)` runs one emcee ensemble per
light curve inside one compiled loop: each alert has its own walkers and random stream, so a batched
fit equals the same alert fitted alone at the same seed. The model is shared, so build it once for
the survey's bands, with the explosion time and the redshift free, and put every alert on its own
first-detection clock:

```python
bands = list(wp.io.SURVEY_BANDS["lsst"])                      # lsstu ... lssty
model = wp.supernova_model("arnett", bands, free=["t_exp", "redshift"],
                           prior=wp.Prior({"t_exp": wp.Uniform(-30.0, 0.0)}))
alerts = [wp.load_lightcurve(p, survey="lsst") for p in paths]  # paths: your alert files
alerts = [lc.set_time_reference(lc.meta["first_detection_mjd"], "first detection") for lc in alerts]
fits = wp.fit_batch(alerts, model, nwalkers=60, nsteps=10000, burnin=2000, seed=0)
[f.best_params for f in fits]
fits[0].info["batch"]["run_s_per_alert"]
```

Each alert's `t_exp` prior is the model's `Uniform(-30, 0)` cut to that alert's own window (the
pre-event rule; `fits[i].info["t_exp_prior"]`). `prior=` also takes one prior per light curve, for
example each host's redshift as a `TruncatedNormal`. Alerts share one compile when they share the
model, the prior families, the likelihood and the length bucket (16, 24, 32, 48, 64, 96, ...
points; `wp.log_density(lc, model).bucket`). Each result is a `SamplerResult`
(`sampler="emcee_batch"`) with emcee's diagnostics and its share of the batch time in
`info["batch"]`; `diagnostics()`, `save()` and `wp.load_result()` work on it. `fit_batch` runs emcee
only.

**How many alerts fit in a night.** `wp.profile(model, lc)` times the log-density by batch size;
`wp.capacity(cost, hours=12, n_gpus=1)` converts seconds per alert into alerts per night:

```python
round(wp.capacity(636.7, hours=12)["alerts_exact"])      # 68: five families, one alert at a time
wp.capacity([5.64, 3.51], hours=12, n_gpus=5)["alerts"]  # 23606: arnett + TDE per alert, batched
```

`capacity` counts device time only; the likelihood maxima, figures and reports run on the host.

**Many fits on several GPUs and CPU cores.** `wp.run_jobs(jobs, gpus="auto", cache_dir="runs/")`
runs one fresh process per job, one job per idle GPU at a time and CPU jobs on their own cores
beside them, retries failures, stops a job that runs past `timeout=`, and resumes an interrupted
batch through `fit_cached`:

```python
jobs = [wp.Job(lc, model, "emcee_jax", kwargs={"seed": 0}, name=f"alert{i}")
        for i, lc in enumerate(alerts)]
report = wp.run_jobs(jobs, cache_dir="runs/", timeout=3600)
fits = report.results                                        # {name: SamplerResult}
```

A script that calls `run_jobs` needs an `if __name__ == "__main__":` guard, and every model a job
carries must be picklable (a factory-built model is; a `predict` defined in a notebook cell is not).
With no idle GPU, GPU jobs run on the CPU with a warning.

---

## 9. Saving, resuming, provenance

```python
cmp.save("out/alert_3068394823507361793")
again = wp.Comparison.load("out/alert_3068394823507361793")
```

`save` writes the table, the peaks, every fit (`SamplerResult.save`), the diagnostics and the light
curve. Every fit carries its provenance (`result.provenance`: whisper and package versions, git
state, the model and its prior, the sampler settings and seed, devices, timing) and every saved file
is hashed, so `load` refuses a modified save. A family that `compare` bound by name (`"arnett"`,
`"magnetar"`, `"tde"`, ...) is bound again from the saved light curve and the prior its fit
recorded, so forecasts and reports work after a reload; pass `models=` only for models you built
yourself with a factory.

`wp.compare(..., cache_dir="fits/")` saves each fit under a hash of its configuration (data, model,
prior, sampler settings, whisper version): running the same comparison again loads the fits instead
of refitting, and an interrupted comparison finishes only what is missing.

---

**Next:** [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) for the ranking rules on any data,
[`API_REFERENCE.md`](API_REFERENCE.md) for every signature, [`CHOOSING.md`](CHOOSING.md) for the
samplers, [`PHOTOMETRY.md`](PHOTOMETRY.md) for how a band magnitude is computed.
