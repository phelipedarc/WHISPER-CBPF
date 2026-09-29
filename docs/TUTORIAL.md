# Whisper Tutorial — working with transient light curves

A hands-on tour of what Whisper can do — **data ingestion, plotting, fitting** (ABC, ABC-SMC, MCMC,
nested sampling and SNPE), and **ranking models, checking the fits and explaining the answer**
(§8). For a survey alert end to end, start with [`LSST_ALERTS.md`](LSST_ALERTS.md). Import once:

> **Most figures and scripts linked below are not in this package.** This repository ships code, not
> analyses, so `dev/` and `sanity_check/` were deliberately left behind at the merge; the three demos
> this tutorial calls runnable are ported to [`examples/`](../examples/). Every other such link is
> history, in the legacy repositories
> [WHISPER_AI @ `8ba3843`](https://github.com/phelipedarc/WHISPER_AI/tree/8ba3843) and
> [whisper-GPU @ `10796a0`](https://github.com/phelipedarc/whisper-GPU/tree/10796a0)
> (discontinued; superseded by whisper_cbpf), pinned at those commits so they cannot rot. Coming
> from the discontinued `whisper_labia` or `whisper_gpu`? See [`MIGRATION.md`](MIGRATION.md).

```python
import whisper_cbpf as wp
```

---

## 1. Load a light curve

`load_lightcurve` reads almost any photometry CSV and works out the columns for you.

```python
lc = wp.load_lightcurve("tests/data/at2017gfo.csv")
print(lc)
# LightCurve(name='at2017gfo', n_points=645, bands=[...29 labels...], mode='magnitude')
```

It auto-detects `time/MJD`, `magnitude`/`flux`, their errors, `band` and `system` columns (override
with `column_map={'time': 'MJD', ...}`), sniffs comma- vs semicolon-separated files, and drops
obviously bad rows (non-finite values, non-positive errors).

### Survey alerts

`survey="lsst"` or `"ztf"` reads alert photometry in the brokers' own field names (a Rubin or ZTF
alert packet, ALeRCE, Fink), keeps only difference-imaging photometry (`psfFlux`, `magpsf`), turns
forced photometry and non-detections into 5-sigma upper limits, and labels the bands `lsstg`,
`ztfr`, ...:

```python
alert = wp.load_lightcurve("tests/data/lsst_alert_sn.json", survey="lsst", redshift=0.1)
print(alert.n_points, alert.bands)            # 15 ['lsstg', 'lssti', 'lsstr']: 12 detections, 3 limits
alert.meta["first_detection_mjd"], alert.meta["n_dropped"]
```

The field mapping and the photometry rule are in [`LSST_ALERTS.md`](LSST_ALERTS.md#2-reading-the-alert).

## 2. Inspect it

`LightCurve` **is an `astropy.table.Table`** — per-point quantities are columns, scalar metadata lives
in `.meta`, and every table operation works directly:

```python
lc['mag_plus_5'] = lc['magnitude'] + 5     # add / compute columns
bright = lc[lc['magnitude'] < 18]          # boolean-mask slicing (keeps the LightCurve + .meta)
lc.sort('time'); lc.group_by('band')       # any astropy Table method
lc()                                       # __call__ -> the table itself
```

The common quantities are also attributes (handy, and what the samplers use):

```python
lc.n_points              # number of points (== len(lc))
lc.bands                 # sorted unique band labels
lc.time, lc.flux, lc.magnitude   # column data as arrays (None if absent; settable)
lc.data_mode             # 'flux_density' | 'magnitude' | 'flux'  (in .meta)
lc.output_format         # forward-model comparison space: 'magnitude' | 'flux_density'
lc.redshift_known        # True/False — False means a redshift prior must be sampled
lc.snr                   # per-point signal-to-noise (computed from the errors)
lc.to_dataframe()        # a pandas view (== lc.to_pandas())
```

**Select** with `where(...)` (`col` / `col_min` / `col_max` / `col_not`, list = OR), or `select_*`:

```python
lc.where(band='r', time_min=58000, time_max=58020, upper_limit=False)
```

## 3. Clean & shape — chainable, each call returns a **new** `LightCurve`

```python
lc = (wp.load_lightcurve("tests/data/at2017gfo.csv", band_lookup=True)  # group bands
        .select_snr(min_snr=5)               # keep SNR >= 5
        .select_time_window(time_max=57990)  # MJD window
        .set_explosion_date(57982.0))        # time -> days since explosion (day 0)
```

…or do it all at load time:

```python
lc = wp.load_lightcurve("tests/data/at2017gfo.csv",
                        band_lookup=True, min_snr=5, time_max=57990, explosion_date=57982.0)
```

### Band grouping
Surveys label filters inconsistently. `band_lookup=True` collapses them into an effective ladder
(`U/g/r/i/z/J/H/K-band` + JWST) using `wp.FILTER_LOOKUP` — e.g. `B→g-band`, `V→r-band`, `Ks→K-band`,
`F606W→r-band`. Clear/white-light bands (`C`, `W`, `w`) are kept as-is. AT2017GFO's 29 raw labels
collapse to 11 effective bands this way.

### Signal-to-noise cut
```python
wp.load_lightcurve("tests/data/at2017gfo.csv").n_points              # 645
wp.load_lightcurve("tests/data/at2017gfo.csv", min_snr=3).n_points   # 632
wp.load_lightcurve("tests/data/at2017gfo.csv", min_snr=5).n_points   # 578
```

## 4. Convert magnitude ↔ flux

```python
lc.add_flux()      # AB magnitudes -> flux density (Jy), with error propagation
flux_lc.add_mag()  # flux -> AB magnitudes
```

`add_flux`/`add_mag` use the constant **AB 3631 Jy** zero point (so the modelling flux the samplers see
stays on one zero point). Pass `zeropoint_jy=lc.zero_point` to opt into the per-band LSST/SVO zero
points instead.

### Rest-frame phase and absolute magnitude

```python
ph = lc.set_explosion_date(57982.0).calc_phase()     # rest-frame phase = (t − ref)/(1+z); adds 'phase'
pk = lc.calc_phase(peak=True)                         # phase relative to the brightest detection
ab = lc.calc_absmag(ebv=0.1)                          # absmag = mag − dm − A_band
```

`calc_phase` uses the curve's redshift for the `(1+z)` time dilation (reference defaults to the
explosion epoch, the peak, or the first detection). `calc_absmag` gets the distance modulus from the
redshift (Planck18) or `luminosity_distance`, and Milky-Way extinction from `ebv`/`rv` (CCM89, per
band's effective wavelength) or an explicit `extinction={'r': 0.3, ...}` dict. Both add a column and
record what they used in `.meta`.

## 5. Data mode, redshift, units & band resolution

These four are wired into `load_lightcurve` and the `LightCurve` itself.

### Data mode
`data_mode` is `flux_density` (canonical unit Jy, default), `magnitude` (dimensionless AB), or `flux`
(band-integrated erg/s/cm²). It is inferred from the columns but you can set it explicitly. To fill in
the other photometry column, call `add_flux()` / `add_mag()` (the light curve is a table, so the new
column is just added):

```python
lc = lc.add_mag()        # adds a 'magnitude' column from 'flux' (constant AB zero point)
lc.output_format         # 'flux_density' / 'magnitude' — the forward-model comparison space
```

### Redshift — argument > column > unknown (never silently assumed)
```python
wp.load_lightcurve("sn.csv", redshift=0.034)        # explicit
wp.load_lightcurve("sn.csv")                        # a 'redshift' column is picked up automatically
```
If neither is present the load **does not fail** — the curve is flagged `redshift_known=False`, carries a
default `redshift_prior` you can override, and warns that *z will be sampled, not assumed*. Validation:
`z ≥ 0`; `z == 0` needs an explicit `luminosity_distance=` (Mpc); negative/NaN is a hard error.

```python
lc.redshift_known        # False
lc.redshift_prior        # {'type': 'Uniform', 'low': 0.001, 'high': 1.0, 'name': 'redshift'}
wp.load_lightcurve("z0.csv", redshift=0.0, luminosity_distance=40.0)   # z=0 case
```

### Units (astropy) — F_ν or F_λ in, canonical Jy out
Flux density may arrive as **F_ν** (Jy/mJy/µJy) or **F_λ** (erg/s/cm²/Å). Pass the unit and Whisper
stores Jy internally; the F_λ→F_ν conversion uses each band's effective wavelength
(`u.spectral_density`):

```python
wp.load_lightcurve("fnu.csv",  flux_unit="mJy")                  # F_nu
wp.load_lightcurve("flam.csv", flux_unit="erg/(s cm2 AA)")       # F_lambda -> Jy via band wavelength
```
Magnitudes must be dimensionless AB (a flux unit on a magnitude column is a clear error). A flux column
with **no** unit warns and assumes the documented default (Jy) — pass `flux_unit=` to silence it.

### Bands — FILTER_LOOKUP, then SVO fallback
Each band resolves to an effective wavelength + zero point. Known filters come from `FILTER_LOOKUP`
(optical bands anchored to LSST ugrizy); an unknown band warns and falls back to the **SVO Filter
Profile Service**:

```python
wp.resolve_band("g")                 # {'source':'lsst', 'lambda_eff':4866.0, 'zero_point':3631.0, ...}
wp.resolve_band("PAN-STARRS/PS1.w")  # warns, then queries SVO (cached by filter ID; offline-safe)
```
SVO results are cached locally, so re-runs never re-query. If SVO is unavailable (no network /
pyphot / astroquery), the load degrades gracefully and you can supply the band by hand:

```python
wp.register_manual_band("my_filter", lambda_eff=9000.0, zero_point=3631.0)
```
> SVO needs `pyphot` (in the `[models]` extra) or, as the fallback, `astroquery` (in `[analysis]`);
> neither needs a GPU, so on a CPU-only install add one with `pip install pyphot`.
> A runnable, **offline** tour of all four features is in
> [`examples/demo_ingestion.py`](../examples/demo_ingestion.py).

## 6. Plot

### Report (overview): apparent magnitude **and** flux, all bands overlaid
```python
wp.plot_light_curve(lc, layout="report")
```
![report](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/dev/figures/at2017gfo_report.png)

### Per-band grid: one box per band, choose the quantity
```python
wp.plot_light_curve(lc, layout="grid", quantity="apparent_mag", ncols=4)
```
![grid](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/dev/figures/at2017gfo_grid_mag.png)

`quantity` can be `"apparent_mag"`, `"flux"`, or `"absolute_mag"` (the latter needs `redshift=` set on
the curve, e.g. `load_lightcurve(..., redshift=0.0099)`).

**Marker conventions:** each band gets a distinct color; **detections** are circles with a black edge;
**SNR < 3** points are up-triangles (△); **upper limits** are down-triangles (▽); magnitude axes are
inverted (brighter = up).

**Time axis:** it names day 0 as the curve declares it. Raw MJD reads `time [MJD]`;
`set_explosion_date(mjd)` reads "days since explosion (MJD …)". When day 0 is not the explosion, say
what it is: `lc.set_time_reference(mjd, "first detection")` shifts the clock the same way and reads
"days since first detection (MJD …)" (and `calc_phase` counts from it).

### Any survey format works
The same one-liner handles raw ZTF photometry — `zg`/`zr` are recognised as `ztfg`/`ztfr`, and
`flag_filters` drops rows by any quality column the file carries:
```python
ztf = wp.load_lightcurve("my_ztf_photometry.csv", flag_filters={"catflags": 0})
wp.plot_light_curve(ztf, layout="report")
```

---

## 7. Fit a model with ABC

Whisper has two pluggable axes — **models** and **samplers**:

```python
wp.list_models()     # 6: ['bazin', 'flare', 'flare_jax', 'gaussian_rise', 'mck19',
                     #     'two_component_kilonova']            (+ your own via register_model)
wp.list_samplers()   # 14: ['abc', 'abc_gpu', 'abc_smc', 'abc_smc_gpu', 'dynesty', 'emcee_jax',
                     #      'mcmc', 'nested', 'npe', 'nuts_gpu', 'pymc_jax_gpu_parallelized',
                     #      'pymc_jax_gpu_vectorized', 'snpe', 'snpe_gpu']
                     #      (+ your own via register_sampler)

# The photometric JAX models (kilonova / TDE / the twelve supernovae) are NOT in list_models()
# until you bind them to a dataset: wp.register_kilonova(bands, redshift, dl_cm). They need the
# [gpu] extra even to register. See INSTALL.md.
```

The built-in **flare** model is `flux = A·(1 − e^(−t/t_rise))·e^(−t/t_decay)`. Fit it to AT2017GFO's
r-band with **Approximate Bayesian Computation** (parallel rejection sampling):

```python
lc = wp.load_lightcurve("at2017gfo.csv", explosion_date=57982.0, min_snr=3)
r  = lc.select_bands("r")

fmax  = r.add_flux().flux.max()                       # scale the amplitude prior to the data
prior = wp.Prior({"amplitude":  wp.Uniform(0, 10*fmax),
                  "rise_time":  wp.Uniform(0.05, 10),
                  "decay_time": wp.Uniform(0.5, 40)})

res = wp.fit_ABC(r, "flare", prior=prior, n_simulations=200_000, quantile=0.005, n_jobs=16)
print(res)                 # SamplerResult(sampler='abc', model='flare', n_samples=1000, AIC=..., runtime=1.2s)
res.summary["amplitude"]   # {'median':..., 'ci16':..., 'ci84':...}
res.best_params            # best-fit parameter dict
res.to_json("fit.json")    # AIC, BIC, max-likelihood, posterior summary, diagnostics
```

![ABC flare fit](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/dev/figures/at2017gfo_abc_flare_r.png)

The flare model tracks the r-band decline well. (Its reduced χ² is high only because the photometry
is high-SNR — tiny error bars magnify any model imperfection; physically-motivated models come later.)

- **Acceptance** is by `quantile` (keep the best fraction — robust default) or a fixed `threshold`.
- **Noise-matched simulations** (`simulate_noise=True`, default): every simulation gets per-point
  white noise drawn from the reported `flux_err`, so it matches the generative model of the data —
  this is what makes ABC exact as ε→0 and keeps the posterior width honest. Distances therefore
  include the simulation noise (`E[D] ≈ χ² + n_points`): re-derive any fixed `threshold` from the old
  noiseless scale, or just use `quantile`.
- **Metrics:** ABC reports `max_log_likelihood`, `AIC` and `BIC` for model comparison, evaluated with
  the exact Gaussian likelihood at the best **accepted** draw (selected by likelihood, not by the
  noisy distance).

### It runs in parallel
Simulations are split across processes (`n_jobs`). On this machine (200k simulations):

| n_jobs | time | sims/s |
|---|---|---|
| 1 | 7.25 s | 27,600 |
| 8 | 1.90 s | 105,000 |
| 32 | 1.73 s | 115,000 |

(~4× here; for expensive physical models the speedup scales further.)

### Bring your own model / distance
```python
import numpy as np
def my_model(params, times, bands=None):
    return params["a"] * np.exp(-times / params["tau"])

wp.register_model("expdecay", my_model, ["a", "tau"],
                  prior=wp.Prior({"a": wp.Uniform(0, 1), "tau": wp.Uniform(1, 50)}))
res = wp.fit_ABC(lc, "expdecay", n_jobs=1)   # n_jobs=1 for closures; module-level fns run in parallel
```
A custom distance is any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float` passed as `distance=`.

### ABC-SMC and more models

There's also **ABC-SMC** (sequential rejection over rounds of shrinking threshold) and two more
built-in models — **`bazin`** (SN rise+fall) and **`gaussian_rise`** (Gaussian rise + exp decay):

```python
wp.fit_ABC_SMC(r, "bazin", prior=prior, n_particles=1000, n_rounds=8, quantile=0.4, n_jobs=32)
```

Worked model comparisons, end to end with a personalised prior, are in
[`MODEL_COMPARISON.md`](MODEL_COMPARISON.md).

![model comparison](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/dev/figures/at2017gfo_model_comparison.png)

### A physical, band-dependent model: `mck19`

The toy models above ignore the band. **`mck19`** is a built-in *physical* model — the optical flare
from a **binary-black-hole merger in an AGN disk** (McKernan 2019; implementation of Darc 2025). A
GW-recoil-kicked remnant shocks a bound-gas hotspot that radiates as a blackbody: a `sin²` rise to the
ram-pressure delay `t_ram`, then exponential decay back to the disk baseline. It returns **flux density
per band** (each blackbody, at the source redshift, is integrated over the observation's filter), so
g/r/i differ — and it fits with any sampler through the same likelihood. Every band must name a filter
to integrate over, and a band that names none raises (AT2017GFO's `C`, `W` and `F814W`), so fit the
bands you mean. Bare letters are LSST's filters unless you say otherwise, and AT2017GFO's are SDSS
photometry ([`PHOTOMETRY.md`](PHOTOMETRY.md) §2):

```python
wp.set_default_band_system("sdss")              # this session: bare u g r i z are SDSS
opt = lc.select_bands(["g", "r", "i", "z"])     # 261 points
res = wp.fit_MCMC(opt, "mck19", nsteps=2000)    # params: v_kick, M_smbh, M_bh, r_bh, redshift
```

Because the data is in magnitude space the likelihood compares in magnitude automatically (the model
predicts flux).
[`dev/demo_mck19.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_mck19.py)
(pinned history: WHISPER_AI @ 8ba3843, discontinued) renders the light curve:

![mck19 light curve](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/dev/figures/mck19_lightcurve.png)

### A redback-backed model: `two_component_kilonova`

WHISPER can also drive **redback** models (the optional `[models]` extra). **`two_component_kilonova`**
is redback's blue + red kilonova, computed as the sum of two one-component redback calls and integrated
over each observation's filter into WHISPER's flux density, with the band rules `mck19` uses — so it
fits through the *same* samplers and likelihood as everything else. redback is imported lazily, so the
package works without it (only `predict` needs the extra, and raises without it).

```python
res = wp.fit_SNPE(opt, "two_component_kilonova", num_simulations=3000)   # the g r i z points above
```

Each call runs redback twice (~4 ms on those 261 points), so **SNPE** (which amortizes simulation
cost) is the natural sampler; ABC/MCMC work with modest budgets. Because AT2017GFO is a real
kilonova, this model fits it well (low residual, no prior-railing) — the clean counterpart to the
`mck19` exercise above.
[`dev/demo_kilonova.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_kilonova.py)
(pinned history: WHISPER_AI @ 8ba3843, discontinued) renders the light curve (note the blue bands
fading faster — kilonova reddening):

![kilonova light curve](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/dev/figures/kilonova_lightcurve.png)

### Likelihoods & space (flux vs magnitude)

All inference can run in **flux** or **apparent-magnitude** space, with Gaussian, upper-limit, and
mixture (outlier-robust) likelihoods (`whisper_cbpf.likelihood`). The default is chosen by the data:
a light curve with **upper limits** is fitted in flux space with the censored likelihood, with no
argument (a non-detection bounds the flux, and the magnitude of zero flux is undefined), at the
limits' significance in `lc.meta["upper_limit_sigma"]` (5 for the survey presets). An explicit
`space="magnitude"` with limits raises, and a likelihood that does not model censoring refuses a
light curve that carries non-detections rather than silently fitting the limits as measurements:

```python
from whisper_cbpf.likelihood import make_likelihood, GaussianLikelihoodWithUpperLimits
lik = make_likelihood(lc, space="magnitude")                 # default by data type; override space/kind
lik = GaussianLikelihoodWithUpperLimits(lc, space="flux")    # use non-detections in flux space
```

(ABC compares simulations with the data through a distance, in the space `space=` names; **MCMC,
nested sampling and SNPE use these likelihoods directly** — see below. ABC and SNPE cannot use upper
limits: fit `lc.where(upper_limit=False)` with them.)

**Rows before the event are never fitted.** Once `set_explosion_date` (or a merger date) is set,
every fit leaves out the rows at or before it, and a model that fits its explosion time leaves out
the non-detections before the first detection, which then set that time's prior
(`result.info["excluded_pre_event"]`, [`API_REFERENCE.md` A2](API_REFERENCE.md#what-every-fit-does)).

### MCMC (emcee)

Likelihood-based posterior sampling with affine-invariant ensemble MCMC. It reuses the **same**
likelihood as the other samplers (so it respects the data's `data_mode` — flux data is fit in flux
space, magnitude data in magnitude space), and `emcee` is a core dependency (no extra needed):

```python
res = wp.fit_MCMC(r, "flare", nsteps=5000, burnin=1000, thin=10, seed=0)
print(res)                              # SamplerResult(sampler='mcmc', ..., AIC=..., runtime=...s)
res.summary["amplitude"]               # median / ci16 / ci84, same as every sampler
res.info["mean_acceptance_fraction"]   # diagnostics; res.emcee_sampler is the raw emcee object
```

Walkers start from a scan of the prior, climbed into the best basin found (`init="prior_scan"`, no
starting guess required); pass `init=` a point, a previous result (an ABC fit, say), or
`init="prior"` for independent prior draws. Sampling is **seeded and reproducible**, and
`res.diagnostics()` says whether the chain can be trusted (§8).

### How close the samplers come — a sanity check

ABC, ABC-SMC, MCMC and SNPE share Whisper's model + prior + likelihood, so they **target** the same
posterior, but only MCMC samples it directly: ABC and ABC-SMC reach it as the acceptance distance goes
to zero, SNPE as its simulation budget grows. At the settings in
[`examples/compare_samplers.py`](../examples/compare_samplers.py), which fits all four to
`gaussian_rise` and overlays them in one corner plot, MCMC is tight and on the truth (`sigma_rise`
2.91 ± 0.07, truth 3); SNPE is centred within one of its own sd but 3-18× wider (2.77 ± 0.23); ABC
and ABC-SMC are 20-65× wider and shifted by up to 1.5 of their own sd (ABC 3.59 +2.41/−1.88,
ABC-SMC 6.05 +4.67/−3.56). The script's docstring has the full table. Treat MCMC (or `nested`) as
the reference and tighten ABC's acceptance before comparing widths. The output figure is not
committed, so run the script to regenerate it; a four-sampler AT2017GFO corner *is* published in
[`sanity_check/BENCHMARK.md`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/BENCHMARK.md)
(pinned history: WHISPER_AI @ 8ba3843, discontinued).

### Neural posterior estimation (SNPE)

Whisper also ships **Sequential Neural Posterior Estimation** (`snpe` / `npe`), a simulation-based
inference method powered by [`sbi`](https://sbi-dev.github.io/sbi/). Instead of an explicit likelihood,
it trains a neural density estimator on `(parameters, simulated light curve)` pairs and conditions it on
your data. The same model + prior + `LightCurve` you use everywhere else just work:

```python
res = wp.fit_SNPE(r, "flare", prior=prior,
                  num_rounds=2,            # 1 = amortized NPE; >1 = sequential SNPE
                  num_simulations=2000,    # per round
                  space="auto")            # 'flux' | 'magnitude' | 'auto', like the likelihoods
print(res)                                  # SamplerResult(sampler='snpe', ..., AIC=..., runtime=...s)
res.summary["amplitude"]                    # median / ci16 / ci84, same as every sampler
res.best_params; res.aic; res.bic           # exact Gaussian AIC/BIC at the best posterior draw
res.to_json("snpe_fit.json")
```

The simulator is Whisper's forward model (`model.predict` at the observed times/bands) with Gaussian
noise matching the data errors — so SNPE's implicit likelihood agrees with `GaussianLikelihood`. The
trained sbi posterior is attached for resampling or an sbi corner plot:

```python
samples = res.posterior.sample((10000,))    # resample the trained posterior
from sbi.analysis import pairplot
pairplot(samples, labels=res.parameters)
# or use Whisper's own samples DataFrame with corner: corner.corner(res.samples)
```

**Advanced / flexible options** (for harder, high-dimensional or multi-band problems):

```python
res = wp.fit_SNPE(
    r, "flare", prior=prior,
    x_format="stacked",                            # condition on (value, err, time) per point
    embedding_net="tcn", embedding_latent=32,      # built-in: "mlp" | "tcn" (or any torch.nn.Module)
    density_estimator="nsf", hidden_features=64, num_transforms=8,                # custom architecture
    proposal_mode="restricted", truncate_quantile=1e-4, support_samples=10_000,   # truncated SNPE
    num_workers=8,                                                                # parallel simulation
)
```

- **`x_format="stacked"`** — condition the network on `(value, error, time)` per point instead of
  the values alone, which is what an embedding net needs to exploit cadence/noise structure.
  The **band is deliberately not a channel**: every simulation is drawn on the observation's own
  `(time, band)` grid in its own row order, so position *i* is the same filter for the data and for
  every simulation — band identity is carried positionally, and a constant channel would carry no
  information for a single-object fit (sbi's z-scoring maps it to exactly 0.0 anyway).
- **`embedding_net`** — `"mlp"` or `"tcn"` build the built-in compressors (`whisper_cbpf.embeddings`;
  the TCN is a Temporal Convolutional Network — dilated causal convolutions specialized for time
  series), trained jointly with the estimator to `embedding_latent` features; or pass any
  `torch.nn.Module`. In the Bazin benchmark the MLP was the *fastest* config and the TCN the *most
  accurate* (see
  [`sanity_check/figures/REPORT.md`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/figures/REPORT.md),
  pinned history: WHISPER_AI @ 8ba3843, discontinued).
- **`density_estimator`** — a name (`'maf'`/`'nsf'`/`'mdn'`) **or** a pre-built `posterior_nn(...)`
  factory; `hidden_features` / `num_transforms` / `num_bins` tune the built-in architectures.
- **`proposal_mode='restricted'`** — truncated SNPE (`RestrictedPrior` + `get_density_thresholder`),
  more robust than SNPE-C when between-round posterior sampling leaks; keep `support_samples` modest.
- **`predict_torch`** — a batched torch forward model (`(B, D)` params + `(n,)` times → `(B, n)` flux,
  or `(theta, times, bands)` for a photometric model) replaces the per-row Python simulator with one
  on-device call. The returned flux is mapped into the comparison space **before** the per-point noise
  is added, on-device, so magnitude-space data is supported as well as flux. You rarely need to write
  one: **`sampler="snpe_gpu"` builds it for you** from any model registered through the JAX factories
  (see [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md)).
- **`device`** — train on a GPU: `'cpu'` (default), `'cuda'` / `'gpu'` / `'cuda:N'`, or **`'auto'`**
  (CUDA when available, else CPU; a GPU request with no CUDA warns and falls back). With
  `predict_torch` the GPU also runs the simulator; `training_batch_size`/`stop_after_epochs` (passed
  through to `sbi`) are the training-speed levers —
  [`sanity_check/benchmark_snpe_device.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/benchmark_snpe_device.py)
  and [`sanity_check/BENCHMARK.md`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/BENCHMARK.md)
  (pinned history: WHISPER_AI @ 8ba3843, discontinued) carry the measurements.
- **Amortized reuse** — with `num_rounds=1` (NPE) the trained `result.posterior` infers a **new**
  same-grid observation in ~10–100 ms: `result.posterior.sample((2000,), x=result.format_x(new_flux))`
  — no refit per object.

```python
res = wp.fit_SNPE(opt, "two_component_kilonova", device="auto", num_rounds=2, num_simulations=2000)
```

> `snpe` needs the optional `[sbi]` extra (`pip install 'whisper-cbpf[sbi]'`, adds `sbi` + `torch`).
> Runnable: [`examples/demo_snpe.py`](../examples/demo_snpe.py) and the notebook
> [`examples/at2017gfo_quickstart.ipynb`](../examples/at2017gfo_quickstart.ipynb). Training is the slow
> part — its tests are marked `slow` (`pytest -m "not slow"` skips them).

## 8. Rank models, check the fits, explain the answer

The supernova and TDE models need float64, set before anything creates an array:

```python
import jax; jax.config.update("jax_enable_x64", True)
import whisper_cbpf as wp
alert = wp.load_lightcurve("tests/data/lsst_alert_sn.json", survey="lsst", redshift=0.1)
```

### Compare models in one call

```python
cmp = wp.compare(alert, ["arnett", "magnetar"])
print(cmp.summary())       # winner 'arnett', strong over 'magnetar' (ln B = 3.46), weight 0.97
cmp.table                  # BIC, delta, weight, grade, converged, problems, per model
```

`compare` fits every model to the same data with the same settings (`emcee_jax` on a GPU, `mcmc` on a
CPU), finds each fit's likelihood peak, checks each fit's convergence, and ranks by BIC (by ln Z when
every model has a converged nested-sampling evidence). Family names such as `"arnett"` are bound to
the light curve: its bands, its redshift (fitted when unknown), and an explosion time fitted from the
last non-detection to the first detection (with shallow limits, pass a wider window:
[`LSST_ALERTS.md`](LSST_ALERTS.md) section 2). Models fitted to other points, or with at least as many
parameters as points, are left out with the reason. The rules are in
[`MODEL_COMPARISON.md`](MODEL_COMPARISON.md#part-0--one-call-wpcompare).

### The likelihood peak behind AIC and BIC

```python
peak = cmp.peaks["arnett"]      # the same as wp.likelihood_max_opt(cmp.results["arnett"], alert)
peak.max_log_likelihood, peak.gain, peak.at_edge, peak.bic
```

A sampler's best draw falls short of the likelihood peak by a different amount for each model;
`likelihood_max_opt` climbs to the peak and computes AIC and BIC there. The posterior is not
changed. `gain` is how far the sampler was from the peak: above 1, it never reached it.
[Why this matters, with numbers](LSST_ALERTS.md#5-why-the-ranking-uses-the-likelihood-maximum).
For a single fit: `wp.fit(..., likelihood_max_opt=True)` or `result.likelihood_max_opt(lc)`.

### Can this fit be trusted?

```python
report = cmp.results["arnett"].diagnostics()     # or cmp.diagnostics["arnett"]
print(report)                                    # every check, pass or fail, and why
report.passed, report.reasons
```

Every sampler gets the checks that apply to it: R-hat, ESS, stuck or stranded chains and N/tau for
MCMC, accepted draws for ABC, the dlogz stop for nested sampling, pile-up at a prior edge for every
fit ([`API_REFERENCE.md` A4](API_REFERENCE.md#a4-check-a-fit)).

### Save, reload, resume

```python
res = cmp.results["arnett"]
res.save("out/arnett")                       # draws, summary, metrics, and how the fit was made
same = wp.load_result("out/arnett")          # identical; refuses a modified save
cmp.save("out/alert")                        # the whole comparison, with the light curve
fit = wp.fit_cached(r, "flare", "mcmc", "fits/", prior=prior, seed=0)   # loads, not refits, next time
```

### Forecasts, and where the models differ most

```python
import numpy as np
t_next = float(alert.time.max()) + np.array([1.0, 3.0, 7.0])
fc = cmp.forecast(t_next, ["lsstg", "lsstr"], survey_depth=24.5)   # magnitudes, per model and cell
d = cmp.discriminate(t_next, ["lsstg", "lsstr"], survey_depth=24.5)
d.attrs["best"]                               # the cell where one more point separates them best
```

### Plots, facts and the report

```python
wp.plot_models(cmp, alert)                    # every model over the data, with residuals
wp.plot_model_comparison(cmp)                 # weights, gap to the best, grade
wp.plot_widths(cmp.results["arnett"])         # how much the data narrowed each parameter
wp.plot_forecast(fc[fc["model"] == "arnett"]) # one model's forecast with its 68 % and 95 % bands
facts = cmp.facts()                           # every quoted number, computed by a stated rule
cmp.report("out/", forecast_times=list(t_next))   # one self-contained HTML page
```

### Priors beyond boxes, and where the chains start

```python
prior = wp.Prior({"amplitude": wp.Uniform(0.0, 10 * fmax),
                  "rise_time": wp.TruncatedNormal(3.0, 1.0, 0.05, 10.0),
                  "decay_time": wp.Fixed(15.0)})            # held at 15, not counted in AIC/BIC
abc  = wp.fit(r, "flare", sampler="abc", prior=prior, n_simulations=20000, quantile=0.01)
post = wp.fit(r, "flare", sampler="mcmc", prior=prior, init=abc)   # chains start on ABC's draws
```

`Normal`, `TruncatedNormal` and `Fixed` work in every sampler. `init=` takes a previous result (an
ABC fit, or the same alert's earlier cut), a point, or one start per chain.

### Explosion time and redshift as parameters

```python
model = wp.supernova_model("arnett", ["lsstg", "lsstr", "lssti"], free=["t_exp", "redshift"],
                           prior=wp.Prior({"t_exp": wp.Uniform(61000.0 - 30, 61002.5)}))
model.parameters[-2:]                         # ['t_exp', 'redshift']
```

A fitted redshift can bias a comparison, since a model can move the source to buy a fit: read the
redshift posterior against its prior (`wp.plot_widths`, and `prior_dominated` in the facts).

### Many alerts, and checking a model against its twin

- `wp.fit_batch(alerts, model)`: one model, many alerts, one compiled GPU loop;
  `wp.run_jobs(jobs)`: many fits over the idle GPUs and pinned CPU cores; `wp.profile(model, lc)`
  and `wp.capacity(cost, hours=12)`: seconds per evaluation and alerts per night.
  [`LSST_ALERTS.md` §7-8](LSST_ALERTS.md#7-seconds-per-alert).
- `wp.check_parity(model_a, model_b, prior_or_posterior, times, bands)`: the per-band |Δmag| between
  two models at the same parameters, for example a JAX port against its redback twin
  ([`API_REFERENCE.md` A11](API_REFERENCE.md#a11-check-that-two-models-agree)).

## 9. Compare posteriors & select models — `plot_corner` and `waic`

Every sampler returns a `SamplerResult` with the same interface, so comparing them is easy. To judge
whether methods are **compatible**, overlay their posteriors (a table of medians hides the uncertainty):

```python
axes = wp.plot_corner(
    [res_abc, res_mcmc, res_snpe],                       # any SamplerResult / DataFrame / dict / array
    labels=["ABC", "MCMC", "SNPE"], log_params=["mej_1"], # log axes for wide-range parameters
    truths={"amplitude": 5.0}, title="AT2017GFO posteriors", save="corner.png")
```

It uses a dark, colourblind-distinct palette (`wp.CORNER_PALETTE`), shared axis ranges, contour lines +
filled marginals, and a legend — publication-ready. See
[`sanity_check/BENCHMARK.md`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/BENCHMARK.md)
(pinned history: WHISPER_AI @ 8ba3843, discontinued) for the four-sampler AT2017GFO corner.

For **model selection**, `AIC`/`BIC` on each result are now computed from the exact Gaussian likelihood
and are comparable across samplers (lower = better). For a fully-Bayesian score that uses the *whole*
posterior, use **WAIC**:

```python
w = wp.waic(res, opt, model="two_component_kilonova", fixed={"redshift": 0.00984})
w["waic"], w["p_waic"]   # lower WAIC is better; p_waic is the effective number of parameters
```

WAIC is most reliable for well-converged posteriors — `p_waic` inflates for very broad ones (ABC's
tolerance posterior, an under-trained SNPE), which is itself a useful warning sign.

## What's next

- [`LSST_ALERTS.md`](LSST_ALERTS.md): a survey alert end to end, how to read the answer, and seconds
  per alert on a CPU and on one GPU.
- [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md): the ranking rules, and worked comparisons on CPU and
  GPU.
- [`API_REFERENCE.md`](API_REFERENCE.md): every public name, by task.
