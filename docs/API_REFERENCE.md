# WHISPER-CBPF (`whisper_cbpf`) — API Reference

Generated for **v0.1.0**. Covers data ingestion + plotting and the inference layer — four pluggable
axes (models, samplers, likelihoods, distances) with the ABC / ABC-SMC / MCMC / SNPE samplers on CPU,
and the JAX/GPU half (NUTS, GPU ABC/ABC-SMC, emcee-over-JAX, PyMC frontends, and the JAX kilonova,
TDE and supernova families) in [§10](#10-the-jaxgpu-half).

- **Environments:** `phe_sbi` (CPU, Python 3.11) and `phe_sbi_2` (GPU/JAX, needs
  `source "$(whisper-cbpf-env)"` before Python starts — see [`GPU_SETUP.md`](GPU_SETUP.md)).
- **Run the CPU tests:** `python -m pytest -q -m "not slow"` → **239 passed, 36 skipped,
  6 deselected** (measured 2026-08-13 in `phe_sbi`, ~77 s). §8 has the caveat about `--collect-only`.
- **Run the GPU tests:** in `phe_sbi_2`, with the env sourced, using the interpreter its banner names.
- **Install:** `pip install whisper-cbpf` is CPU-complete; `[gpu]` adds the JAX half. Every GPU entry
  is listed by `list_models()` / `list_samplers()` / `list_distances()` even without the extra.

## Package map

```
whisper_cbpf/
  __init__.py          # public API exports
  plotting.py          # plot_light_curve, plot_ppc, plot_calibration, plot_corner
  metrics.py           # waic, per_band_metrics, predictive_metrics (RMSE/LPD/ELPD-LOO/coverage)
  validation.py        # recovery_metrics, posterior_predictive_check, sbc_rank, sbc_ranks
  priors/              # Uniform, LogUniform, Prior; _numpy.py (default) + _jax.py (lazy: log_prob_jax)
  distance.py          # chi2_distance + the distance registry
  embeddings.py        # MLPEmbedding, TCNEmbedding, build_embedding (SNPE conditioning nets)
  likelihood/          # GaussianLikelihood, ...WithUpperLimits, Mixture..., make_likelihood;
                       #   _numpy.py (default) + _jax.py (lazy: log_likelihood_jax)
  models/              # Model + register/get/list + built-ins flare/bazin/gaussian_rise + physical mck19 + redback two_component_kilonova
  samplers/            # BaseSampler, SamplerResult, ABCSampler, ABCSMCSampler, MCMCSampler, SNPESampler, fit_ABC(_SMC)/fit_MCMC/fit_SNPE, fit
  io/                  # LightCurve, load_lightcurve, bands, photometry, units, svo
  distance/            # registry; _numpy.py (default) + _jax.py (lazy)
  metrics/             # _numpy.py (waic, per-band, predictive) + _jax.py (ArviZ ESS)
  models/jax/          # kilonova, kilonova_two, tde, supernova, flare + binding factories
  samplers/jax/        # nuts_gpu, abc_gpu, abc_smc_gpu, emcee_jax, pymc_gpu, snpe_gpu
  backends/            # check_gpu, require_jax, gpu_list, device policy
```

Top-level (`import whisper_cbpf as wp`): `LightCurve`, `load_lightcurve`, `plot_light_curve`,
`plot_ppc`, `plot_calibration`, `plot_corner`, `CORNER_PALETTE`, `waic`, `per_band_metrics`, `predictive_metrics`, `recovery_metrics`, `posterior_predictive_check`, `sbc_rank`,
`sbc_ranks`, `group_bands`, `FILTER_LOOKUP`, `resolve_band`, `resolve_bands`, `LSST_BAND_INFO`, `SvoUnavailable`,
`register_manual_band`, `unregister_manual_band`, `clear_manual_bands`, `Prior`, `Uniform`, `LogUniform`,
`Model`, `register_model`, `get_model`, `list_models`, `chi2_distance`, `register_distance`,
`get_distance`, `list_distances`, `GaussianLikelihood`, `GaussianLikelihoodWithUpperLimits`,
`MixtureGaussianLikelihood`, `make_likelihood`, `register_likelihood`, `list_likelihoods`, `fit_ABC`,
`fit_ABC_SMC`, `fit_MCMC`, `fit_SNPE`, `fit`, `SamplerResult`, `register_sampler`, `list_samplers`.

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
`set_explosion_date(mjd)` (observer-frame days since explosion); **`calc_phase(reference=, redshift=,
peak=, hours=)`** (rest-frame phase `(t − ref)/(1+z)`); **`calc_absmag(dm=, redshift=, ebv=, rv=3.1,
extinction=)`** (distance modulus from `z`/`luminosity_distance` + Milky-Way extinction via CCM89 or an
explicit `{band: A_mag}` dict → `absmag` column); `to_dataframe()` (→ `to_pandas()`); `lc()` returns the
table itself.

## 2. `load_lightcurve(path, *, ...)` → `LightCurve`  (`io.loader`)

Auto-maps columns; key options: `column_map`, `band_lookup` (broadband grouping), `default_band`,
`quality_cuts`, `flag_filters={'catflags':0}`, `time_min/time_max`, `bands`, `min_snr`,
`explosion_date`, `delimiter`. Detects an `upper_limit` column. Raises `ValueError` for a missing
time / band / measurement column.

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
`DEFAULT_BAND_ALIASES` and `FILTER_LOOKUP` (96 labels → 10 effective bands; ZTF `ZTF_g`/`ztf_g`-style names included, aliases matched case-insensitively).

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

**SVO** (`io.svo`, needs `astroquery`, shipped with `[gpu]`): `resolve_band_svo(band, *,
lambda_eff_hint=None)`, `get_filter_metadata(filter_id, *, use_cache=True)` (cached by ID; offline-safe
disk cache under `$WHISPER_SVO_CACHE` — `use_cache=False` forces a fresh fetch, e.g. after SVO revises a
filter), `get_transmission_data(filter_id)` (Phase-2 spectral integration),
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

## 5. Plotting

**`plot_light_curve(lc, *, layout="report", quantity="apparent_mag", bands=None, ncols=3, figsize=None, title=None, save=None)`**
— `layout`: `"report"` (mag + flux panels) or `"grid"` (per band). `quantity`: `apparent_mag` /
`absolute_mag` (needs redshift) / `flux`. Markers: detections = circles, SNR<3 = △, upper limits = ▽.

**`plot_ppc(results, lc, model=None, *, quantity="apparent_mag", panel_by="auto", n_draws=200, bands=None, tmin=None, tmax=None, ncols=None, colors=None, figsize=None, title=None, seed=0, save=None)`**
— **posterior-predictive check as a grid.** For each fit, draws `n_draws` posterior samples, evaluates
the model on a smooth time grid, and shades the **95% predictive band** (2.5–97.5 percentiles) + median
over the photometry, per band. `results` is one `SamplerResult`, a `{label: SamplerResult}` dict, or a
list. `panel_by`: `"method"` (one panel per fit, bands overlaid — the multi-sampler grid),
`"band"` (one panel per band, fits overlaid), or `"auto"` (method when several fits, else band).
`quantity`: `"apparent_mag"` (inverted axis) or `"flux"` (flux density [Jy]). Returns the `Figure`.

**`plot_calibration(results, lc, model=None, *, levels=(0.5, 0.68, 0.8, 0.9, 0.95, 0.99), space="auto", per_band=False, n_draws=400, colors=None, figsize=None, title=None, seed=0, save=None)`**
— **coverage-calibration curve** (reliability / pp-plot): empirical posterior-predictive coverage vs the
nominal credible level, for one or more fits (or one fit broken out `per_band=True`). On the diagonal =
calibrated; below = over-confident, above = under-confident. Coverage from `predictive_metrics`.

**`plot_corner(posteriors, *, labels=None, parameters=None, colors=None, truths=None, bins=30,
levels=(0.39, 0.86), smooth=1.0, log_params=None, title=None, legend_loc="upper right", save=None, **corner_kwargs)`**
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
common to all inputs; `log_params` puts those on a `log10` axis; `truths` (dict or list) draws dashed
reference lines; a colour→label legend is added. Returns the `Figure`. Ideal for comparing samplers on
the same data — the posteriors (with uncertainties) show whether methods are *compatible*, which a table
of point estimates cannot.

---

## 6. Inference: priors, models, distance, samplers

### 6.1 Priors  (`whisper_cbpf.priors`)
Small, **picklable** distributions (so they cross process boundaries in parallel ABC).

| Class | Constructor | Methods |
|---|---|---|
| `Uniform` | `(low, high, name=None)` | `sample(rng)`, `log_prob(x)`, `rescale(u)`, `bounds` |
| `LogUniform` | `(low, high, name=None)` (`low>0`) | same |
| `Prior` | `(distributions: dict)` | `sample(rng=None) -> dict`, `log_prob(params)`, `rescale(unit_cube)`, `names`, `bounds` |

```python
prior = wp.Prior({"amplitude": wp.Uniform(0, 10), "rise_time": wp.Uniform(1, 10)})
prior.sample(np.random.default_rng(0))   # {'amplitude': ..., 'rise_time': ...}
```

### 6.2 Models  (`whisper_cbpf.models`)
A model maps parameters to predicted flux: `predict(params: dict, times: np.ndarray, bands) -> np.ndarray`.

| Function | Signature | Description |
|---|---|---|
| `register_model` | `(name, predict, parameters, prior=None, description="", *, overwrite=False, predict_jax=None, log_prob_jax=None)` | Register a model by name. The two `*_jax` slots are optional; a factory that built them must pass them through or the registered model is the CPU half only. |
| `get_model` | `(model)` | Resolve a name (or pass a `Model`). |
| `list_models` | `()` | Sorted registered model names. |
| `Model` | dataclass: `name, predict, parameters, default_prior, description, predict_jax, log_prob_jax` | Callable: `model(params, times, bands)`. `predict_jax(theta, times, band_idx) -> flux` is `predict` with a flat theta and integer bands, traceable and differentiable; `log_prob_jax(theta) -> scalar` is a data-bound log-*likelihood*. Both `None` unless a factory filled them — see [§10](#10-the-jaxgpu-half). |

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
| `two_component_kilonova` | **Kilonova** (NS–NS merger) with a blue (low-κ) + red (high-κ) ejecta component, via the optional **redback** backend (`[models]` extra). Band-integrated AB magnitudes (redshift-aware) → flux density (Jy). | mej_1, vej_1, kappa_1, temperature_floor_1, mej_2, vej_2, kappa_2, temperature_floor_2, redshift |

`two_component_kilonova` is the **first redback-backed model** — redback is imported lazily, so WHISPER
and `list_models()` work without it; only `predict` needs the `[models]` extra. It wraps redback's
`two_component_kilonova_model`, mapping WHISPER bands → redback LSST filters and converting redback's AB
magnitude to flux density (a machine-precision round-trip). It is an **expensive** simulator (~50 ms per
band-call), so **SNPE** (amortized) is the natural sampler; ABC/MCMC want modest budgets. See
[`dev/demo_kilonova.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_kilonova.py)
and [`dev/fit_kilonova_at2017gfo.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/fit_kilonova_at2017gfo.py)
(WHISPER_AI, not in this package).

`mck19` is **band-dependent** — it returns flux density (Jy) at each `(time, band)`, evaluating the
hotspot + disk blackbody at the band's effective wavelength (via Whisper's band system) and the redshift
(Planck18 luminosity distance + time dilation). `t = 0` is the merger; the flare peaks at the
observer-frame delay `t_ram`. Self-contained (astropy constants/cosmology only — no `speclite`); the AB
magnitude is the monochromatic-at-`λ_eff` approximation to the original LSST filter integration. Fixed
disk constants (`mdot=0.05` Edd, `alpha=0.1`) do not enter the light-curve *shape*. See
[`dev/demo_mck19.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_mck19.py)
(WHISPER_AI, not in this package) for the g/r/i light curve.

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

### 6.3 Distance  (`whisper_cbpf.distance`)
`chi2_distance(obs_flux, obs_flux_err, sim_flux, bands=None) -> float` = `sum(((obs-sim)/err)**2)`.
Any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float` can be passed as a custom distance.

Distances are also a **name registry**, so a custom distance can be selected by string
(`fit_ABC(lc, "flare", distance="my_distance")`) and survives pickling into parallel ABC workers:

| Function | Signature | Description |
|---|---|---|
| `register_distance` | `(name, fn, *, overwrite=False)` | Register a distance by name. |
| `get_distance` | `(distance)` | Resolve a name (a callable passes straight through). |
| `list_distances` | `()` | Sorted registered names — **14** out of the box: `chi2`, `chi_square` (an alias), `mse`, `rmse`, `mae`, `wmse`, `wmae`, and a `*_jax` twin of each. The `_jax` entries are listed even without the `[gpu]` extra. |

Names are matched **case-insensitively**.

### 6.4 Samplers  (`whisper_cbpf.samplers`)

`fit_ABC(lc, model="flare", ...)`, `fit_ABC_SMC(lc, model="flare", ...)`, `fit_MCMC(lc, model="flare",
...)`, and `fit_SNPE(lc, model="flare", ...)` → `SamplerResult`. `fit(lc, model,
sampler="abc"|"abc_smc"|"mcmc"|"snpe", **kwargs)` is the generic dispatcher. `list_samplers()` →
`['abc', 'abc_gpu', 'abc_smc', 'abc_smc_gpu', 'emcee_jax', 'mcmc', 'npe', 'nuts_gpu',
'pymc_jax_gpu_parallelized', 'pymc_jax_gpu_vectorized', 'snpe']` — **11**, the six GPU entries
included even without the `[gpu]` extra (`npe` is an alias of `snpe`); `register_sampler(name, cls)`
adds your own. **ABC/ABC-SMC** use the χ² distance; **MCMC/SNPE** use the shared likelihood layer, so all
four reach the same posterior on the same data (see
[`sanity_check/compare_samplers.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/compare_samplers.py),
WHISPER_AI, not in this package).

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
initial_guess=None, initial_scatter=1e-3, space="auto", likelihood="auto", seed=0, progress=False,
moves=None, n_jobs=None)`** — `n_jobs` runs the walkers' likelihood evaluations in a process pool
(worth it only for expensive simulators, e.g. the ~0.1 s kilonova model); with
`likelihood="gaussian_scatter"` a prior parameter named `sigma` is routed to the likelihood as the
free Villar+17 extra-scatter term (see §6.5). — affine-invariant ensemble MCMC via `emcee` (`samplers.mcmc`; emcee is a **core**
dependency). The log-posterior is the Whisper prior + the **shared likelihood layer**
(`make_likelihood(lc, kind=likelihood, space=space)`), so MCMC uses the same physically consistent,
`data_mode`-aware likelihood as the others (flux data → flux space, magnitude data → magnitude space).
`nwalkers` defaults to `max(2·ndim+2, 4·ndim)` (forced even); walkers init from the prior unless
`initial_guess` is given. Sampling is **seeded/reproducible**. `best_params` is the max-posterior draw;
`max_log_likelihood`/`AIC`/`BIC` are exact Gaussian values; `info` carries acceptance fraction +
autocorrelation time; the `emcee.EnsembleSampler` is attached as `result.emcee_sampler`. `fit_MCMC(...)`
is the convenience wrapper.

**`SNPESampler.fit(lc, model, prior=None, *, num_rounds=2, num_simulations=1000, space="auto",
density_estimator="maf", embedding_net=None, embedding_latent=32, x_format="value", predict_torch=None,
scatter_param=None, hidden_features=None, num_transforms=None, num_bins=None,
proposal_mode="posterior", truncate_quantile=1e-4, support_samples=10000, num_samples=10000,
device="cpu", seed=0, show_progress=False, num_workers=1, max_logl_scan=2000, scan_timeout=300,
standardize_x=True, **train_kwargs)`** —
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
- **GPU simulation:** `predict_torch(theta, times) -> flux` (batched torch model: `(B, D)` params +
  `(n,)` times → `(B, n)` flux) replaces the per-row Python simulator with one on-device batched call
  (~10³× faster simulation; flux-space only). `result.format_x(values)` maps a raw vector to the
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

- **Device (GPU):** `device` = `'cpu'` (default), `'cuda'`/`'gpu'`/`'cuda:N'`, or `'auto'` (CUDA when
  available, else CPU). The torch prior + observed data are placed on the device; a GPU request without
  one warns and falls back to CPU. The GPU accelerates *training* (not the CPU simulator), so it helps
  most with many simulations / large nets — see
  [`sanity_check/benchmark_snpe_device.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/benchmark_snpe_device.py)
  and [its figure](https://raw.githubusercontent.com/phelipedarc/WHISPER_AI/8ba3843/sanity_check/figures/snpe_device_benchmark.png)
  (WHISPER_AI, not in this package).

Extra kwargs pass to `NPE.train` (e.g. `max_num_epochs`, `training_batch_size`, `stop_after_epochs`).
`max_log_likelihood`/`AIC`/`BIC` are the exact Gaussian values at the best posterior draw. The trained
sbi posterior is attached as `result.posterior` (and `result.posteriors` per round) for resampling /
`sbi.analysis.pairplot`. `fit_SNPE(...)` is the convenience wrapper; `"snpe"` and `"npe"` both dispatch here.

**`SamplerResult`** fields: `sampler`, `model`, `parameters`, `samples` (DataFrame of accepted draws
+ `distance`), `summary` (median/ci16/ci84/mean/std per param), `best_params`, `n_data`, `n_params`,
`runtime_s`, `info` (n_simulations, n_accepted, acceptance_rate, epsilon, quantile, n_jobs, plus
**`band_metrics`** — per-band MSE/RMSE/MAE at the best fit — and **`predictive_metrics`** — RMSE / LPD /
ELPD-LOO / WAIC / AIC / BIC + the coverage-calibration curve; see §6.6), `min_distance`,
`max_log_likelihood`, `aic`, `bic`. Methods: `n_samples`, `to_dict()`, `to_json(path=None, indent=2)` —
every field above (`band_metrics` and `predictive_metrics` included) is serialised to the JSON, written
to `path` when given and always returned as a string; `indent=None` emits compact one-line JSON for
machine-read outputs.

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

**Upper limits are a flux-space statement.** `GaussianLikelihoodWithUpperLimits` used to carry a
magnitude branch as well (a survival term `1 − Φ((limit − model)/(POGSON/upper_limit_sigma))`,
i.e. "fainter than the limit"); it was **removed**, and `space='magnitude'` — including `'auto'` on
magnitude data — now raises. A non-detection says the *flux* was below a limiting flux, and the
magnitude of zero flux is undefined (`−2.5 log₁₀ 0 = +∞`), so there is no magnitude-space censoring
integral over the same quantity; the removed branch also took its width from the Pogson constant
rather than from the data. Call `lc.add_flux()` and pass `space='flux'`, or fit the detections alone
with `lc.where(upper_limit=False)`.

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
| `GaussianLikelihoodWithUpperLimits(lc, space="auto", upper_limit_sigma=3.0)` | Gaussian for detections + a censoring term `P(true flux < limit) = Φ((limit − model)/(limit/upper_limit_sigma))` for the non-detections. **Flux space only** — `space="magnitude"` raises. |
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
- **`lpd`** — log predictive density `Σ_i log mean_s p(y_i|θ_s)` (the `lppd`; higher is better), `total`
  and `per_point`.
- **`elpd_loo`** — expected log predictive density by **PSIS-LOO** cross-validation (Vehtari 2017;
  higher is better) with `p_loo`, `se`, `looic = −2·elpd_loo`, and `pareto_k_max` (>0.7 ⇒ unreliable);
  uses `arviz` when installed (`None` otherwise).
- **`waic`** — `waic` (deviance, lower is better), `elpd_waic`, `p_waic`, `se`.
- **`aic` / `bic`** — carried through from the fit's best-fit likelihood.
- **`coverage`** — the **calibration curve**: for each nominal `level`, the empirical fraction of
  observations inside the central posterior-predictive interval, overall and per band (empirical ≈
  nominal ⇒ calibrated).
- **`scatter_param`** — the posterior column used as the extra-scatter σ (or `null`).
- **`space` / `likelihood` / `scatter_param` / `n_draws_requested` / `seed`** — how this block was
  scored. `waic(result, lc)` reads them back so the manual call reproduces exactly this number.
- **`elpd_loo_error`** — present only when `arviz` **is** installed and `az.loo` raised; `elpd_loo` is
  `null` both when arviz is absent (documented degradation) and when it failed, and this key is what
  tells the two apart.

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
WHISPER_AI, not in this package); all take a `SamplerResult` (or its `.samples`) so they work for every sampler.

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

Exposed as `wp.recovery_metrics`, `wp.posterior_predictive_check`, `wp.sbc_rank`, `wp.sbc_ranks`. See
[`sanity_check/figures/REPORT.md`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/sanity_check/figures/REPORT.md)
(WHISPER_AI, not in this package) for the full synthetic-recovery benchmark across all five samplers.

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
(network boundary), and — in WHISPER_AI, not in this package —
[`dev/phase0_smoke.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/phase0_smoke.py),
[`dev/demo_abc_at2017gfo.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_abc_at2017gfo.py),
[`dev/demo_ingestion.py`](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/dev/demo_ingestion.py).

## 8. Test coverage

> **239 passed, 36 skipped, 6 deselected** on a base + `[dev]` install (`phe_sbi`, `-m "not slow"`,
> measured 2026-08-13, ~77 s). Note `--collect-only` reports **260**, not 270: ten test modules
> `importorskip("jax")` at *module* scope, so they are never collected yet each still reports one
> skip. The per-file table below has drifted and is not re-measured here; treat the totals above as
> the source of truth. Regenerate with
> `docker exec phe_sbi bash -lc 'cd <repo> && python -m pytest tests --collect-only -q | tail -1'`
> after adding tests, and update this section — three rows had silently drifted before this check.

| File | Tests | Focus |
|---|---|---|
| `test_photometry.py` | 5 | AB zeropoint, mag↔flux, error propagation, SNR. |
| `test_bands.py` | 13 | Aliases (incl. ZTF `ZTF_g`-style, case-insensitive), case-sensitivity, `group_bands`/`FILTER_LOOKUP` (ZTF resolution + finite λ_eff/ZP), `unmapped_bands`. |
| `test_schema.py` | 11 | Validation, subsetting, `add_*`, `snr`/`select_snr`, `set_explosion_date`, upper limits. |
| `test_loader.py` | 12 | AT2017GFO load, window/subset, grouping, `min_snr`, `explosion_date`, upper limits. |
| `test_plotting.py` | 11 | report/grid layouts, flux/absolute-mag, redshift guard, upper-limit markers; `plot_corner` overlay/legend, common-params + log axes + array/empty errors; `plot_ppc` single/multi × flux/magnitude × panel-by band/method; `plot_calibration` multi + per-band. |
| `test_metrics.py` | 21 | WAIC keys + finite + better-fit-lower ordering, `fixed=`/subsampling, pointwise log-lik summing to the total; **WAIC pinned to a closed-form value** (two-point Gaussian posterior, derivation in the docstring) both through `_loo_waic` and end-to-end through `wp.waic`, and cross-checked against `arviz.waic` on the same matrix (exact up to arviz's ddof=0); **`wp.waic(res, lc)` == `res.waic` on a scatter fit**; the `waic`/`waic_reliable`/`elpd_loo`/`rmse` properties returning `None` instead of raising on a failed block; arviz-failing distinguished from arviz-absent; `per_band_metrics` zero-at-truth / offset / JSON, **and on a censored light curve** — it builds no likelihood (so NaN `flux_err` on a non-detection is fine), scores detections only with `n_upper_limits_excluded` reporting the rest, keeps an all-censored band at `n = 0`, and `flux_to_space` matching the bound `model_in_space`; `predictive_metrics` RMSE/LPD/ELPD-LOO/WAIC/AIC-BIC/coverage block + auto-attach, coverage-calibration for a good fit, and scatter-aware WAIC/coverage vs mis-specified. |
| `test_validation.py` | 4 | recovery z-score + coverage signs, PPC (reduced χ²≈1, predictive coverage, p≈0.5), SBC rank uniformity (calibrated vs edge-biased), `sbc_rank` bounds. |
| `test_embeddings.py` | 6 | MLP/TCN embedding shapes + finite output, TCN receptive field covers the input, `build_embedding` dispatch + unknown-name error, `x_format` validation, ABC `simulate_noise` (n_jobs-reproducible, optional, noisy-vs-noiseless distance floor). |
| `test_scatter.py` | 4 | Villar+17 scatter likelihood (formula vs hand calc, σ=0 reduces to Gaussian, pointwise sums, registry), MCMC recovers injected extra scatter (σ counted in AIC), ABC scatter validation + posterior column, ABC magnitude-space acceptance. |
| `test_review_fixes.py` | 6 | scientific-review fixes: ABC exact-likelihood AIC, ABC-SMC importance weighting, WAIC drops draws not data points, mck19 π factor, CCM89 out-of-range clamp, non-positive-flux→mag guard. |
| `test_priors.py` | 6 | Uniform/LogUniform/Prior sampling, log_prob, rescale, picklability. |
| `test_models.py` | 8 | flare/bazin/gaussian_rise (incl. flare pre-explosion=0, bazin tail→0), custom model, errors. |
| `test_distance.py` | 2 | chi² zero/known value. |
| `test_abc.py` | 6 | parameter recovery, serial+parallel, acceptance count, JSON, dispatch, custom model. |
| `test_abc_smc.py` | 9 | SMC registration, recovery, epsilon tightening, explicit schedule, dispatch, model-agnostic, and the `min_epsilon` floor (broadens the posterior; `"auto"` runs). |
| `test_likelihood.py` | 15 | Gaussian flux/mag, space-auto, upper-limit CDF (incl. the tail that used to flatten at the `_MIN_PROB` clip), mixture, make_likelihood, picklable; and the refusals — magnitude-space upper limits, censored rows under a likelihood that ignores them, free scatter + upper limits, non-finite error bars. |
| `test_units.py` | 12 | F_ν/F_λ→Jy, per-point λ, NaN-λ error, mag rejects flux unit, no-unit default, flux dimensionality. |
| `test_svo.py` | 15 | FILTER_LOOKUP→SVO, mocked metadata/index, cache hit + disk cache, corrupt-cache (×4 params), graceful degrade, manual override (no spurious warn), transmission, ambiguity. |
| `test_ingestion.py` | 25 | data_mode/output_format, `__call__`, redshift (arg/column/unknown/0/neg/NaN/all-NaN), units, band-info, subset preservation, backward-compat ZP, flux-mode. |
| `test_snpe.py` | 11 | registry/alias + proposal-mode validation (no sbi), torch-prior adapter — including the **bounded `LogUniform` support** regression (`support.check` and `log_prob` must agree, and out-of-prior draws must be rejected) — density-estimator dispatch; SNPE fit end-to-end / multi-round / embedding-net + custom estimator (`slow`; need `[sbi]`). |
| `test_table_lc.py` | 7 | LightCurve as astropy.Table: column ops + native methods, property setters, slicing keeps subclass/meta, `where()`, rest-frame `calc_phase`, `calc_absmag` (distance modulus + CCM89/explicit extinction). |
| `test_registries.py` | 4 | likelihood + distance registries (`register_*`/`list_*`/`get_distance`, distance-by-name in `fit`), manual-band register/unregister/clear. |
| `test_mcmc.py` | 4 | MCMC recovery + reproducibility, data-mode-consistent likelihood space, and ABC↔MCMC posterior agreement. |
| `test_mck19.py` | 7 | `mck19` AGN-disk BBH flare: registration, finite/positive flux, band-dependence, peak at `t_ram`, redshift dimming, no-bands warning, and end-to-end MCMC recovery in magnitude space (`slow`). |
| `test_two_component_kilonova.py` | 9 | redback kilonova: registry/prior + band-mapping + no-bands error (no redback needed); flux finite/positive, band-dependence, machine-precision redback round-trip, mixed-band predict, and SNPE recovery (`slow`) — all guarded by `importorskip("redback")`. |
| `test_benchmark_kilonova.py` | 3, **none collected here** | flux-vs-magnitude benchmark: `setup` (data + prior) and the magnitude-space χ² distance (no redback), plus a `slow` end-to-end fit→publication-report render. It loads `sanity_check/benchmark_kilonova_modes.py` by path, and `sanity_check/` is not part of this package, so `conftest.py` excludes the file from collection; run it against [WHISPER_AI](https://github.com/phelipedarc/WHISPER_AI/blob/8ba3843/tests/test_benchmark_kilonova.py) instead. |

Fixtures: `tests/data/at2017gfo.csv`, `tests/data/ztf18aarlhfw.csv`. The figures + ABC JSON that
`test_benchmark_kilonova.py`'s render produced live in
[`sanity_check/figures/`](https://github.com/phelipedarc/WHISPER_AI/tree/8ba3843/sanity_check/figures)
(WHISPER_AI, not in this package — see the row above).
All SVO/network calls are **mocked** — no live network in CI.

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
generated by introspecting the installed package, not transcribed.

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
- **`kilonova_model(band_names, redshift, dl_cm, *, name='kilonova_one_jax', n_wave=2000, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', temperature_floor=None, mag_floor=40.0, t_exp_days=0.0, filter_set=None, arnett_prefactor=1.0)`**  
  Bind dataset context to the one-component JAX kilonova and return a whisper ``Model``. The only
  factory that fills `Model.predict_jax` so far: `predict_jax(theta, times, band_idx)` returns Jy
  from a flat theta, is differentiable, and carries `predict_jax.band_index(bands)` for the
  string → index map. Both it and `predict` go through one core jitted at factory time.
- **`register_kilonova(band_names, redshift, dl_cm, *, name='kilonova_one_jax', **kwargs)`**  
  Build a kilonova ``Model`` for this dataset and register it under ``name``.
- **`tde_model(band_names, redshift, dl_cm, *, name=None, rise='gaussian', n_wave=2000, n_time=None, f_debris=1.0, xi=1.0, n_pre=100, pin=None, prior=None, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', filter_set=None, **engine_kwargs)`**  
  Bind dataset context to the JAX cooling-envelope TDE and return a whisper ``Model``.
- **`register_tde(band_names, redshift, dl_cm, *, name=None, **kwargs)`**  
  Build a TDE ``Model`` for this dataset and register it under ``name``.
- **`supernova_model(model, band_names, redshift, dl_cm, *, name=None, n_wave=2000, t_exp_days=0.0, dense_resolution=None, magnetar_convention='1.15', interaction=True, dilation=True, pin=None, prior=None, mag_floor=40.0, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', filter_set=None, times=None)`**  
  Bind dataset context to one of the JAX supernova models and return a whisper ``Model``.
- **`supernova_models()`**  
  The names :func:`supernova_model` accepts.
- **`register_supernova(model, band_names, redshift, dl_cm, *, name=None, **kwargs)`**  
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

- **`NUTSGPUSampler()`** — *class*  
  NUTS (NumPyro) on GPU. See module docstring for the shared-``log_prob_fn`` design.
- **`fit_NUTSGPU(lc, model, prior=None, *, log_prob_fn, **kwargs) -> 'SamplerResult'`**  
  Fit ``lc`` with ``model`` via NumPyro NUTS on GPU. See :meth:`NUTSGPUSampler.fit`.
- **`ABCGPUSampler()`** — *class*  
  Rejection ABC with a vmapped JAX simulator. See the module docstring for matched semantics.
- **`fit_ABC_GPU(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  Fit ``lc`` with ``model`` by rejection ABC on the GPU. See :meth:`ABCGPUSampler.fit`.
- **`ABCSMCGPUSampler()`** — *class*  
  Importance-weighted ABC-SMC with a vmapped JAX simulator. See the module docstring.
- **`fit_ABCSMCGPU(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  Fit ``lc`` with ``model`` via ABC-SMC on GPU. See :meth:`ABCSMCGPUSampler.fit`.
- **`EmceeJAXSampler()`** — *class*  
  whisper registry adapter for :func:`fit_emcee_jax`.
- **`fit_emcee_jax(lc, model, log_prob_fn, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000, thin=10, seed=0, progress=False, space='flux') -> 'SamplerResult'`**  
  Arm A(1): emcee vectorized against the shared jitted JAX log-density.
- **`fit_emcee_numpy(lc, model, times, values, sigmas, *, prior=None, nwalkers=32, nsteps=5000, burnin=1000, thin=10, seed=0, n_jobs=8, progress=False, space='flux') -> 'SamplerResult'`**  
  Arm A(2): plain-numpy emcee on N CPU cores — the realistic no-GPU baseline.
- **`PyMCJAXVectorizedSampler()`** — *class*  
  PyMC frontend, NumPyro NUTS, chains vmapped on one device.
- **`fit_PyMCJAXVectorized(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  See :class:`PyMCJAXVectorizedSampler`.
- **`PyMCJAXParallelizedSampler()`** — *class*  
  PyMC frontend, NumPyro NUTS, chains ``pmap``ed across devices — the multi-GPU path.
- **`fit_PyMCJAXParallelized(lc, model, prior=None, **kwargs) -> 'SamplerResult'`**  
  See :class:`PyMCJAXParallelizedSampler`.
- **`fit_snpe_gpu(lc, model, *, prior=None, num_rounds=3, num_simulations=10000, embedding='none', embedding_latent=32, density_estimator='maf', num_samples=10000, seed=0, device='cuda', show_progress=False, **snpe_kwargs)`**  
  Fit one light curve with SNPE on GPU, using the JAX model for GPU-batched simulation.
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
  support with a *finite* gradient there. Refuses anything but `Uniform` / `LogUniform`.
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

**Kilonova (one component)** — `whisper_cbpf.models.jax.kilonova` (14 public functions)

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
- `make_filter_set(band_names, lam_min=1000.0, lam_max=30000.0, n_wave=5000)` — Export sncosmo bandpasses to plain arrays. Run ONCE, offline, then save.
- `pre_explosion(t_src_s)` — Boolean mask, True where the model returns identically zero flux.
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

**Tidal disruption events** — `whisper_cbpf.models.jax.tde` (22 public functions)

- `ab_magnitude(temperature, r_photosphere, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, *, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99')` — AB magnitude per observation, from a photosphere ``(T, R)`` and a filter set.
- `analytic_fallback(time, l0, t_0)` — Bolometric luminosity: t^-5/3 fall-back with a flat plateau before ``t_0``.
- `build_interaction_grid(time, dense_times)` — Setup-time (NumPy) companion to :func:`diffusion` / :func:`viscous`.
- `calc_tfb(binding_energy_const, mbh_6, stellar_mass)` — Fall-back time of the most tightly bound debris, in SECONDS. Pure arithmetic.
- `cocoon_photosphere(time, luminosity, t_thin, vej, nn)` — redback ``photosphere.CocoonPhotosphere``. time days, luminosity erg/s, vej km/s.
- `cooling_envelope(mbh_6, stellar_mass, eta, alpha, beta, *, f_debris=1.0, t_0_init=1.0, binding_energy_const=0.8, zeta=2.0, hoverR=0.3, n_time=5000)` — Sarin & Metzger (2024) cooling-envelope TDE engine.
- `cooling_envelope_ab_magnitude(t_obs_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, mbh_6, stellar_mass, eta, alpha, beta, *, n_time=5000, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', **kw)` — redback ``cooling_envelope(..., output_format='magnitude')``, AB magnitudes.
- `cooling_envelope_flux_density(t_obs_days, nu_obs_hz, redshift, dl_cm, mbh_6, stellar_mass, eta, alpha, beta, *, n_time=5000, dilation=True, **kw)` — redback ``cooling_envelope(..., output_format='flux_density')``, in mJy.
- `default_prior(model_name='cooling_envelope')` — ``(Prior, pinned)`` for a TDE model: redback's, read from redback where possible.
- `default_prior_gaussianrise()` — ``(Prior, pinned)`` for ``gaussianrise_cooling_envelope``. All seven free in 1.15.1.
- `envelope_exists(mbh_6, stellar_mass, beta, binding_energy_const=0.8)` — True where the envelope is born OUTSIDE the circularisation radius, i.e. exists at all.
- `exponential_powerlaw(time, a_1, alpha_1, alpha_2, tpeak)` — a_1 (1 - exp(-t/tpeak))^alpha_1 (t/tpeak)^-alpha_2.
- `fallback_prior(model_name)` — :func:`redback_prior` without redback. Same return shape; see _FALLBACK.
- `flux_density_mjy(temperature, r_photosphere, nu_obs_hz, redshift, dl_cm, *, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99')` — Flux density in mJy from a photosphere ``(T, R)`` at observer frequencies.
- `gaussian_rise(time, a_1, peak_time, sigma_t)` — a_1 * exp(-(t - t_peak)^2 / 2 sigma^2). `time` and the two scales share units.
- `gaussianrise_cooling_envelope_ab_magnitude(t_obs_days, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, peak_time, sigma_t, mbh_6, stellar_mass, eta, alpha, beta, *, xi=1.0, n_pre=100, n_time=5000, mag_floor=40.0, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', **kw)` — Gaussian rise stitched onto the cooling envelope, AB magnitudes per observation.
- `gaussianrise_cooling_envelope_flux_density(t_obs_days, nu_obs_hz, redshift, dl_cm, peak_time, sigma_t, mbh_6, stellar_mass, eta, alpha, beta, *, xi=1.0, n_pre=200, n_time=5000, dilation=True, ebv_mw=None, ebv_host=None, xi_mw=None, xi_host=None, r_v_mw=3.1, r_v_host=3.1, law='f99', **kw)` — Gaussian rise stitched onto the cooling envelope, flux density in mJy.
- `redback_prior(model_name, *, drop=('redshift',))` — redback's OWN prior for ``model_name``, read from redback, not transcribed.
- `rise_peaks_near_fallback(peak_time, sigma_t, mbh_6, stellar_mass, *, n_sigma=3.0, xi=1.0, binding_energy_const=0.8)` — True where the Gaussian rise peaks within ``n_sigma`` of the stitch point.
- `ryu_f_debris(mbh_6, stellar_mass, beta)` — Ryu et al. partial-disruption debris fraction, clipped to [0, 1].
- `tde_photosphere(time, luminosity, mass_bh, mass_star, star_radius, tpeak, beta, rph_0, lphoto)` — redback ``photosphere.TDEPhotosphere``. Photosphere that expands as a power of Mdot.
- `temperature_floor_photosphere(time, luminosity, vej, temperature_floor)` — redback ``photosphere.TemperatureFloor``. time days, luminosity erg/s, vej km/s.

**Supernovae** — `whisper_cbpf.models.jax.supernova` (22 public functions)

- `ab_magnitude_of(model, grid, params, band_idx, weights, norms, lam_obs_ang, redshift, dl_cm, *, magnetar_convention='1.15', interaction=True, **kw)` — AB magnitude per observation for any model in the family. ``model`` is static.
- `basic_magnetar(time_s, p0, bp, mass_ns, theta_pb, convention='1.15')` — Dipole spin-down luminosity, erg/s. ``time_s`` SECONDS, ``p0`` ms, ``bp`` 1e14 G.
- `bolometric(model, grid, params, *, magnetar_convention='1.15', interaction=True)` — Bolometric luminosity in erg/s for any model in the family. ``model`` is static.
- `build_sn_grid(time_days, dense_resolution=1000, t_pad=100.0)` — Setup-time (NumPy) companion to every model here. Build once per dataset, reuse.
- `cutoff_norm(luminosity, temperature, r_photosphere, cutoff_wavelength_ang)` — redback ``CutoffBlackbody._set_norm``. Dimensionless renormalisation, per epoch.
- `cutoff_shape(lam_obs_ang, redshift=0.0, cutoff_wavelength_ang=3000.0)` — ``min(lam_source/lam_cut, 1)`` on the OBSERVER wavelength grid. Setup-time-able.
- `default_prior(model)` — ``(Prior, pinned, constraints)``: redback's, read from redback where possible.
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

**Flare (JAX)** — `whisper_cbpf.models.jax.flare` (6 public functions)

- `get_model()` — A ``whisper_cbpf.models.Model`` instance for this model — NOT registered in whisper_cbpf's
- `make_log_prob_jax(times, values, sigmas, prior_bounds)` — Build a ``jax.jit``-compiled ``log_prob(theta_vec) -> scalar`` log-posterior.
- `predict_jax(params, times)` — Batched/vmapped flux.
- `predict_numpy(parameters, times, bands=None)` — whisper_cbpf's ``Model.predict(parameters: dict, times, bands) -> flux`` contract.
- `predict_torch(theta_batch, times, device='cuda')` — GPU-batched simulator for SNPE: ``(B, 4)`` torch params -> ``(B, n)`` torch flux.
- `prior_bounds_flat()` — ``[(low, high), ...]`` in PARAMETERS order (from the shared JAX-free spec).

