# Changelog

All notable changes to WHISPER-CBPF (`whisper_cbpf`). The project is in early development; nothing is
released to PyPI yet — install from GitHub
(`pip install git+https://github.com/phelipedarc/WHISPER-CBPF.git`).

## [Unreleased]

### Fixed — five places that returned a wrong number with no error

Each was measured, not inferred. Full report in [`DEPLOYMENT_AUDIT.md`](DEPLOYMENT_AUDIT.md).

- **`abc_gpu`'s magnitude floor did not floor.** `_model_in_space_jnp` clamped at a bare `1e-300`,
  which is exactly `0.0` in float32, so `jnp.maximum` clamped nothing. Measured:
  `_model_in_space_jnp(0.0, "magnitude")` returned `inf`, not the "~759 mag" its docstring promised.
  `abc_gpu` never enables x64, so float32 is its default case. Now uses the same
  `max(_MIN_FLUX_JY, finfo.tiny)` rule `likelihood/_jax.py` already had.
- **`MixtureGaussianLikelihood` scored WAIC under a plain Gaussian.** It overrode `log_likelihood`
  but not `log_likelihood_pointwise`, and both `waic` and `predictive_metrics` dispatch on the
  latter. Measured on 25 points: total 29.1446 against a pointwise sum of 31.4662, so
  `fit(..., likelihood="mixture")` reported an AIC from the mixture beside a WAIC from a Gaussian.
- **The eleven flux→magnitude converters disagreed.** SNPE's torch path clamped at a bare
  `finfo.tiny`, giving 769.80 mag in float64 against everyone else's 758.90. All model-side
  converters now agree. `io.photometry.flux_density_to_mag` still returns `NaN` and now documents
  why it is the exception: it is the **data** converter, and a negative *measured* flux must not be
  given an invented magnitude.
- **WAIC did not say when it was meaningless.** Measured on a 9-parameter redback kilonova against
  88 AT2017GFO points: `p_waic = 939505` beside an `AIC = 415`, returned silently. `_loo_waic` now
  reports `p_waic_reliable` against Gelman/Hwang/Vehtari's `p_waic <= n_data/2`, and warns.
- **A fit on an empty light curve reported `AIC = 2k`, `BIC = -inf`.** `GaussianLikelihood` now
  raises and names the usual cause — `select_time_window` before `set_explosion_date`.

`attach_band_metrics` / `attach_predictive_metrics` were bare `except Exception: pass`. The reason a
metric block is missing now lands in `info` and in a warning.

### Added — `wp.fit` reaches the JAX/GPU samplers, and `space=` works there

New `whisper_cbpf/samplers/jax/_adapters.py` turns a registered `Model` into the two callables the
JAX samplers already document: a batched `(B,D) -> (B,n)` forward map for `abc_gpu`/`abc_smc_gpu`,
and a scalar `theta -> scalar` density for `nuts_gpu`/`pymc_jax_gpu_*`/`emcee_jax`. Resolution order
is explicit argument → `model.log_prob_jax` → auto-build, so it fires only where the code used to
raise and **no existing call changes**.

Measured: all five now fit `kilonova_one_jax` on AT2017GFO through `wp.fit` with nothing hand-built,
and the two NUTS frontends agree to 4 significant figures (AIC 1222722.6 vs 1222724.0).

- **`space=` is now live on the NUTS family.** It defaulted to `"flux"` and was recorded but never
  used, so a magnitude light curve was scored in flux by every predictive metric. Default is now
  `"auto"`. `info` records `space_source` (`density` / `declared` / `assumed`) and `log_prob_fn`.
- **`tde_model`, `supernova_model` and `kilonova_two_model` gained `predict_jax`**, and the three
  registrars now forward it. The supernova's requires **concrete** `times` — its grid is built from
  the observation times and its compiled core memoised on their bytes.
- **`band_aliases=`** on all four photometric factories, e.g. `{"g": "sdssg"}`: `make_filter_set`
  takes sncosmo names while survey CSVs carry bare `g`/`r`/`i`. It lives on the *model* because
  `per_band_metrics`, `predictive_metrics` and `plot_ppc` all call `model.predict(..., lc.band)`.
- **`kilonova_two_model` now jits its core** — it was the only photometric factory that did not.
  Measured on 211 observations at `n_wave=2000` (A6000, best of five): **134.9 ms → 1.893 ms**, a
  factor 71.
- **`likelihood.resolve_space` is public** (was `_resolve_space`).

### Fixed — three `emcee_jax` defects

- **It sampled a flat box where `nuts_gpu` used the real prior.** `fit_emcee_jax` reads only
  `prior.distributions[nm].bounds`, so its stationary distribution is `L(θ)·1_box` — correct only if
  every prior is Uniform. `kilonova_model` gives `temperature_floor` a `LogUniform(100, 6000)`, and
  `nuts_gpu._numpyro_priors` documents that exact substitution as a bug it fixed (median 775 K vs
  3050 K). The auto path now builds a log-posterior and ranks draws for AIC/BIC by the likelihood
  twin. **If you pass your own `log_prob_fn` to `emcee_jax`, pass a log-posterior.**
- **It hard-cast theta to `jnp.float32`**, discarding x64 where it was enabled.
- **Its vmap extent was `nwalkers`**, user-scaled and unchunked — the sixth instance of the
  unbounded-vmap trap. Now chunked at 16; `walker_chunk=None` restores the old behaviour.

### Added — `scripts/compare_at2017gfo.py` and `INSTALL.md`

The acceptance test: fits a TDE, a supernova and one- and two-component kilonovae to
`tests/data/at2017gfo.csv`, on CPU or GPU, in flux or apparent-magnitude space, with every sampler
that can legitimately run each combination, and prints one AIC/BIC/WAIC table. Combinations that
cannot work print the reason rather than being skipped.

`INSTALL.md` documents every extra and what needs it, the GPU bootstrap and why it cannot be a
Python call, the float64 flag and what it costs, and a verification step per layer.

### Fixed — documentation that contradicted the code

**`register_supernova` does NOT raise at registration.** Measured in float32: all **12/12**
supernovae register successfully and enter `list_models()`; the guard fires at the first `predict`,
exactly like the TDE's. `docs/GPU_SETUP.md`, `models/jax/__init__.py` and `_factories.py` all said
otherwise. Also corrected: `API_REFERENCE`'s two contradictory test counts, its duplicate `## 9`
heading, its `list_distances()` (2 vs 14) and `list_samplers()` (5 vs 11) claims, its `env_script()`
description; `TUTORIAL`'s model and sampler listings (3 vs 6, 4 vs 11); `CHOOSING`'s "all five
metrics" over a list of six; and `CONTRIBUTING`'s `whisper_labia` title and coverage command.

### Added — a JAX half for the prior, the likelihood and the kilonova `Model`

A gradient sampler previously had to be handed a hand-written log-density per model
(`models/jax/flare.make_log_prob_jax` is the only one that exists). The three pieces it needs are
now available separately and can be composed:

- **`Model.predict_jax` and `Model.log_prob_jax`**, two optional dataclass fields, `None` on a model
  that has no JAX implementation. `predict_jax(theta, times, band_idx) -> flux` is `predict` with
  exactly two changes — the parameter dict becomes a flat array ordered by `parameters`, and band
  names become integer indices — and it takes one parameter set, so batching stays the caller's
  `jax.vmap`. `register_model` grew matching keyword arguments; a factory that does not pass them
  registers the CPU half only.
- **`priors.log_prob_jax(prior, names=None)`** — `Prior.log_prob` as a traceable function of a flat
  vector, normalising constants included. `-inf` outside the support with a **finite** gradient
  there: each branch's input is clamped into the support before the log, because `jnp.where`
  evaluates both branches and `0 * inf = nan` otherwise survives the mask into the backward pass.
  Refuses any distribution it cannot express rather than substituting a uniform box.
- **`likelihood.log_likelihood_jax(like)`** — the JAX twin of `GaussianLikelihood.log_likelihood`,
  built *from an existing likelihood object* so the two arms cannot disagree about which points are
  in the fit, which space they are compared in, or the normalising constant. The three subclasses
  are refused by name (`isinstance` would silently accept all three and evaluate a different
  density). In magnitude space the flux floor is the working dtype's smallest normal when
  `_MIN_FLUX_JY = 1e-300` is not representable in it — the same float32 trap the SNPE torch path hit.

`priors.py` and `likelihood.py` became `priors/` and `likelihood/` packages with `_numpy.py` +
`_jax.py` and a lazy `__getattr__`, matching `distance/` and `metrics/`. Every existing import path
still resolves, including the private `_MIN_FLUX_JY` / `_MIN_PROB` / `_LN2PI` that `samplers/snpe.py`
and `tests/test_snpe.py` name.

### Fixed

- **`kilonova_model`'s `predict` is now jitted**, closing the `OPEN_ITEMS.md` entry: the factory
  called the unjitted `kn.ab_magnitude` binding and re-traced on every call — 65.3 ms per `predict`
  against 0.14 ms for the identical computation under `jax.jit`. Both entry points now go through
  one `_core` jitted at factory time, the way `tde_model` already did, which is also what keeps
  `predict` and `predict_jax` from drifting (measured agreement: 2.2e-16 relative in float64).

### Documentation

- **`docs/GPU_SETUP.md`'s float64 cost was wrong.** It said `jax_enable_x64` makes the kilonova
  "11–23× slower", a figure carried over from `whisper-GPU/HANDOFF.md` and never measured. Measured
  on one idle A6000 at `n_wave=2000`, 100 epochs, 4 bands: **1.9–2.5×** for a forward pass at batch 1
  (and that case is launch-bound — the float32 kernel is ~12 µs inside a ~45 µs launch floor, so the
  ratio moves ±30% between runs), **4.3×** for `value_and_grad` at batch 1, which is what NUTS pays
  per leapfrog step, and **52–57×** from batch 100 up, stable to <1%. The plateau is the hardware:
  the A6000's FP64 rate is 1/32 of FP32 and float64 transcendentals cost more than that again.
  The single "11–23×" number was wrong at both ends because the cost is a function of batch size.

## [0.1.0] — 2026-08-11 — the merge

`whisper_labia` and `whisper_gpu` become one package, `whisper_cbpf`. **This is a reorganisation, not
a rewrite:** almost every migrated file is byte-identical to its source apart from its import lines.
Sources: `WHISPER_AI` @ `8ba3843` and `whisper-GPU` @ `10796a0`, both frozen, both untouched, both
still citable. Per-file origins are in `PROVENANCE.md`.

### Breaking

- **Import name is `whisper_cbpf`.** No compatibility shim and no alias — `whisper_labia` and
  `whisper_gpu` do not resolve. `docs/MIGRATION.md` maps every public symbol old → new.
- **The GPU sampler registry stores zero-argument factories, not class objects.** `get_sampler(name)`
  still does `_SAMPLERS[name]()` and still returns a sampler instance, so the documented contract is
  unchanged; only code introspecting `_SAMPLERS["nuts_gpu"]` *as a class* is affected. This is what
  lets the package import without JAX.
- **The whisper version guard is gone.** `_REQUIRED_WHISPER_SYMBOLS` / `_check_whisper()` verified an
  *external* `whisper_labia` checkout. The CPU core is now the same package, so that skew cannot
  occur.

### Added

- `pip install whisper-cbpf` is a complete CPU package that imports with **no JAX, no CUDA, no torch
  and no redback** — while `list_models()`, `list_samplers()` and `list_distances()` still name every
  GPU entry, so the GPU half is discoverable before it is installed. Asking for one without the extra
  raises a message naming the extra, never a bare `ImportError`.
- New extras: `[gpu]` (`jax[cuda12]`, numpyro, arviz) and `[pymc]`. CUDA 12 is pinned deliberately —
  plain `jax` silently installs the CPU build.
- Paired-backend subpackages: `distance/` and `metrics/`, each `_numpy.py` + `_jax.py` behind one
  registry, numpy the default, `_jax` imported lazily. JAX distances take a `_jax` suffix
  (`mse` → `mse_jax`), matching the sampler naming.
- Documentation the GPU half never had: `docs/API_REFERENCE.md` §9 covers its ~110 public symbols,
  plus `docs/CHOOSING.md`, `docs/GPU_SETUP.md`, `docs/MIGRATION.md`, `DUPLICATES.md`, `FINDINGS.md`,
  `OPEN_ITEMS.md`, `INVENTORY.md` and `PROVENANCE.md`.

### Changed

- The five numpy distances (`mse`, `rmse`, `mae`, `wmse`, `wmae`) are now core and register normally.
  They used to be pushed into the registry from the GPU repo on import via `register_into_whisper()`,
  which no longer exists. The CPU ABC keeps them — that is what makes a CPU/GPU comparison at a fixed
  distance possible.
- `require_jax()`'s message now names the extra (`pip install 'whisper-cbpf[gpu]'`) instead of
  directing users to install jax and numpyro by hand. The environment-bootstrap half is unchanged and
  still essential.
- `env.sh` → `scripts/whisper-gpu-env.sh`. Still a sourceable shell script, deliberately: JAX reads
  device visibility at import, so this cannot become import-time logic.

### Not included

Analyses, benchmarks and run results — this package ships functionality only. 217 files stay in the
legacy repos: `analysis/`, `sanity_check/`, `dev/`, `jax_gpu/` and `whisper-GPU/benchmarks/`.

### Known

- `tests/test_benchmark_kilonova.py` (3 tests) is not collected: it loads a file from the excluded
  `sanity_check/` tree. The only reduction against the pre-merge CPU count.
- `test_flare_gradients.py::test_naive_single_where_would_fail_here` fails on JAX 0.10.2 — **and fails
  identically in the source repo.** A control that has stopped testing anything; see `FINDINGS.md`.
- The two kilonova implementations, and `flare` vs `flare_jax`, are **different models** — not
  CPU/GPU pairs. See `DUPLICATES.md`.

---

## [Unreleased] — 0.0.1.dev0 (legacy `whisper_labia` history below)

### SNPE's GPU simulate path now works in magnitude space (it used to refuse)
- **`_torch_simulate` maps the model flux into the comparison space before adding noise**, via a new
  `_torch_model_in_space` — the torch counterpart of `GaussianLikelihood.model_in_space`. `predict_torch`
  returns **flux** by contract, but the observation the flow trains against lives in `lik.space`, and
  the GPU path was adding noise to the raw flux and returning that. Rather than produce wrong
  summaries the sampler simply refused: `fit_SNPE(..., predict_torch=...)` raised
  *"GPU simulation currently supports flux-space data only"* for any magnitude-space light curve, so
  the fastest simulator was unavailable to exactly the datasets that are reported in magnitudes. The
  guard is gone and the order now matches the CPU simulator line for line — `mf = model_in_space(flux)`,
  then `mf + noise`. Getting that order wrong is silent: the flow still trains, on the wrong data.
- **The float32 floor is `finfo.tiny`, not `_MIN_FLUX_JY`.** The numpy likelihood clamps a
  non-positive model flux at `1e-300` before `log10`, which is correct in float64 and useless here:
  the simulate tensors are **float32**, where `1e-300` underflows to exactly `0.0`, so the clamp stops
  clamping, `log10(0)` gives `-inf`, and the whole simulation batch becomes NaN. Clamping at the
  dtype's smallest normal instead leaves a non-positive flux at ~103.7 mag — far fainter than any real
  detection, and finite.
- **`nan_to_num` runs after the space map, not before.** Scrubbing the flux only would let a
  non-finite *magnitude* through untouched.
- Regression tests in `tests/test_snpe.py`: `test_torch_model_in_space_matches_the_numpy_likelihood`
  (agrees with the CPU path), `test_torch_model_in_space_floor_survives_float32` (finite output, and
  asserts `_MIN_FLUX_JY` really is 0 in float32), `test_torch_simulate_maps_space_before_adding_noise`
  (noiseless simulate returns the magnitude, not the flux). The `predict_torch` path had **no** test
  before this.

### Fix: SNPE/TSNPE returned posterior draws from outside a `LogUniform` prior's bounds
- **`_to_torch_prior` now gives each `LogUniform` component a bounded `interval(low, high)` support.**
  It previously built them as `TransformedDistribution(Uniform(log a, log b), ExpTransform())`, and
  torch derives `TransformedDistribution.support` from the **last transform's codomain** — so an
  `ExpTransform` reports `GreaterThan(0)` and the `[a, b]` bounds silently vanish. That matters
  because sbi's `within_support()` — the accept/reject test inside `DirectPosterior.sample()`, the
  last thing that runs before draws are returned — prefers `distribution.support.check(theta)` over
  `log_prob(theta)`. The final rejection step was therefore only checking *positivity* for every
  `LogUniform` parameter, and out-of-prior draws were handed back as posterior samples. On the
  AT2017gfo kilonova benchmark (10,000 draws per arm) **35.6% / 27.8% / 29.9% / 73.8%** of
  `snpe_cpu` / `snpe_jax` / `tsnpe_cpu` / `tsnpe_jax` draws violated at least one prior bound — and
  `tsnpe_jax`'s M_ej *median* (0.1007 M☉) sat outside its own prior maximum (0.1). The plain
  `Uniform` parameter, which has a real `Interval` support, leaked 0.00% in all four arms.
  Re-running the four arms after the fix gives **0.00%** leakage in every one, all still drawn by
  rejection (no MCMC fallback), all still `converged`.
- The prior's *measure* is unchanged — same log-space `Uniform` base, same `ExpTransform`, so `sample`
  and `log_prob` are bit-for-bit what they were (`log_prob` was always correct, returning `-inf`
  outside the box; it simply was not the test sbi ran). Only the advertised support changed. The
  single-round arms confirm this: their probed acceptance rate is identical to four decimal places
  (0.4360, 0.1390) because the trained flow is unchanged and only the rejection step moved. The
  multi-round TSNPE arms *do* shift, as expected — their restricted proposal also rejects against the
  prior support, so rounds 2+ train on different simulations.
- Implemented as a small cached `TransformedDistribution` subclass (`_bounded_log_uniform_cls`) that
  overrides `support`; bounds are stored as Python floats so the constraint is device-agnostic under
  sbi's `move_distribution_to_device`, and `expand()` is overridden to carry them across. Regression
  test in `tests/test_snpe.py::test_torch_prior_loguniform_support_is_bounded`.

### Fix: a mixed `Uniform`/`LogUniform` prior could not run SNPE on a GPU at all
- **`_to_torch_prior` now passes `device=` to `MultipleIndependent`.** sbi's `MultipleIndependent`
  runs `self.to(device or "cpu")` inside `__init__`, which **moved the per-parameter distributions
  back to the CPU** after the adapter had deliberately built them on the training device. sbi then
  refused to train — `check_if_prior_on_device`: *"Prior device 'cpu' must match training device
  'cuda:0'"* — so `device='cuda'` was unusable for **any** prior that mixes `Uniform` and `LogUniform`,
  i.e. every kilonova prior in this repo, even though §6.4 of the API reference documents the GPU path.
  All-`Uniform` priors were unaffected (they take the `BoxUniform` branch, whose tensors are built on
  `device` directly). The `device` kwarg is accepted from sbi 0.23; a `TypeError` fallback keeps older
  versions working.

### Documentation: undocumented public API + drift
- **Documented the public symbols that appeared in no `.md` file**, chiefly `io.units.as_unit` /
  `to_flux_cgs` / `DEFAULT_UNITS`, `io.bands.normalize_bands`, `samplers.get_sampler`,
  `samplers.base.attach_band_metrics` / `attach_predictive_metrics`, `metrics.DEFAULT_COVERAGE_LEVELS`,
  `io.svo.DEFAULT_SVO_IDS`, and the per-model module contract (`PARAMETERS` / `PRIOR` / `DESCRIPTION` /
  `<name>_flux`) that `register_model` consumes — the convention `docs/EXTENDING.md` tells you to copy
  from `flare.py` but never named.
- **Documented keyword arguments that existed only in the source**: `load_lightcurve(band_aliases=,
  normalize=, drop_nonfinite=)`, `normalize_band` / `normalize_bands` / `group_bands(warn_unknown=)`,
  `ABCSMCSampler.fit(max_attempts_per_round=)`, `get_filter_metadata(use_cache=)`,
  `find_filter_id(tol_frac=)`, `build_embedding(n_channels=, latent_dim=)`,
  `SamplerResult.to_json(indent=)`.
- **`docs/EXTENDING.md` now covers the likelihood axis.** It opened with "two pluggable axes" and had
  no likelihood section, while README links it as *"add your own model, sampler, likelihood, or
  distance"* and `__init__` / README / CONTRIBUTING all say **four**. Added an *Add a likelihood*
  section and corrected the count.
- **API reference:** `embeddings.py` added to the package map (it was described in the SNPE prose but
  absent from the module listing); the `register_distance` / `get_distance` / `list_distances` and
  `register_likelihood` / `list_likelihoods` registries documented in §6.3 / §6.5 instead of only being
  named in the export list; the SNPE prior-adapter paragraph now states that `LogUniform` components
  carry a bounded `[low, high]` support and that mixed priors are placed on the training device.
- **Reconciled the test count**, which simultaneously read 189 (README), 195 (API-reference header) and
  218 (API-reference §8). The verified figure is **219 collected** (212 with `-m "not slow"`), and three
  per-file rows in §8 had drifted (`test_abc_smc` 8→9, `test_registries` 5→4,
  `test_two_component_kilonova` 8→9, plus `test_snpe` 8→11).
- **README install section** now lists every optional extra (`[models]`, `[svo]`, `[sbi]`, `[loo]`,
  `[dev]`); `[svo]`, `[sbi]` and `[loo]` previously appeared only in the tutorial, API reference and
  CONTRIBUTING, never where a user installs.

### Scatter-aware predictive metrics (WAIC / PSIS-LOO / coverage)
- **`predictive_metrics` now folds the fitted extra-scatter σ into the predictive density.** For a fit
  with a free Villar+2017 scatter term (a "jitter"/intrinsic-scatter parameter, Hogg, Bovy & Lang 2010),
  the pointwise density used for LPD / WAIC / PSIS-LOO and the coverage intervals is the scatter-augmented
  Gaussian `𝒩(M_i, σ_i² + σ_s²)` with **each posterior draw's own** σ_s — the same generative model the
  fit optimised. Previously these used the reported errors only, which for high-SNR photometry collapsed
  the density: `p_waic`/`p_loo` exploded (10⁴–10⁶) and coverage under-shot badly. Folding σ back in
  restores sane effective-parameter counts and near-nominal coverage (verified on synthetic data:
  p_waic 152072 → 4.1, 95%-coverage 0.33 → 0.99). New `scatter_param` argument: `"auto"` (default; uses
  the lone non-model posterior column, e.g. `sigma`), an explicit name, or `None` to disable. This is the
  standard posterior-predictive prescription — marginalise over the full posterior, nuisance scatter
  included (Gelman et al., *Bayesian Data Analysis* 3rd ed. ch. 6–7; Vehtari, Gelman & Gabry 2017;
  Watanabe 2010). Distance-based ABC (no σ) correctly falls back to the plain Gaussian.

### Fix: plots no longer crash after importing redback (LaTeX rendering)
- **redback** enables matplotlib's `text.usetex` globally on import, which made every subsequent
  WHISPER plot crash with `RuntimeError: ... latex could not be found` on systems without a TeX
  install. Each plotting function now calls a small `_safe_usetex()` guard that resets the flag
  **only when `latex` is genuinely missing** (a user with a working TeX keeps `usetex=True`). Fixes the
  order-dependent `test_plotting` failures and the model-comparison notebook's corner plot.

### Band lookup: ZTF-prefixed names + case-insensitive aliases
- **`ZTF_g` / `ZTF_r` / `ZTF_i` / `ZTF_z`** (and `ztf_g`-style / mixed-case variants) now resolve to the
  LSST g/r/i/z bands — added to `FILTER_LOOKUP` and `DEFAULT_BAND_ALIASES`, so their effective wavelength
  and zero point fill in instead of `NaN`. `normalize_band` now matches aliases **case-insensitively**
  (the alias keys are survey prefixes that never collide with case-sensitive single-letter bands), and
  `resolve_band` **composes** the alias step (a label absent from the lookup is normalised first), so
  survey codes resolve without hardcoding every case variant.

### Predictive & model-comparison metrics + coverage calibration in every fit's JSON
- **`predictive_metrics(result, lc, ...)`** (`whisper_cbpf.metrics`) — a posterior-predictive metric
  block computed from the posterior sample and **auto-attached to `result.info["predictive_metrics"]`**
  by every sampler (so it is written to `to_json`):
  - **RMSE** of `observed − posterior-mean prediction`, per band + overall.
  - **LPD** (log predictive density, the `lppd`).
  - **ELPD** by **PSIS-LOO** cross-validation (Vehtari et al. 2017) with `p_loo`, `se`, `looic` and the
    Pareto-`k` diagnostic — via the optional **`arviz`** extra (`pip install 'whisper-cbpf[loo]'`);
    degrades to `None` without it.
  - **WAIC** (deviance scale), **AIC**, **BIC**.
  - a **coverage-calibration curve**: empirical posterior-predictive coverage vs nominal at
    50/68/80/90/95/99%, overall and per band.
- **`plot_calibration(results, lc, ...)`** (`whisper_cbpf.plotting`) — the coverage-calibration curve /
  reliability pp-plot (empirical vs nominal coverage; on the diagonal ⇒ calibrated), for one or more
  fits or one fit broken out per band.

### Per-band goodness-of-fit + posterior-predictive grid plot
- **`per_band_metrics(lc, model, params, *, space, fixed)`** (`whisper_cbpf.metrics`) — per-band and
  overall **MSE / RMSE / MAE** of the best-fit residuals in the fit's comparison space (Jy for flux,
  mag for magnitude). **Every sampler** now computes it at its best fit and stores it in
  `result.info["band_metrics"]`, so it is written to `SamplerResult.to_json()` automatically — the
  per-band error is part of the standard fit output.
- **`plot_ppc(results, lc, model=None, *, quantity="apparent_mag"|"flux", panel_by="method"|"band"|"auto", ...)`**
  (`whisper_cbpf.plotting`) — the grid posterior-predictive-check plot (previously bespoke to the
  AT2017GFO study) is now a first-class feature: 95% predictive band + median over the data, one panel
  per fit (bands overlaid) or one panel per band (fits overlaid), in **magnitude or flux** space. Takes
  one `SamplerResult`, a `{label: result}` dict, or a list.

### Tutorial notebooks
- **`notebooks/`** — five hands-on Jupyter tutorials covering the main functionality, executed with
  embedded outputs: **1** light curves (`load_lightcurve`, SNR cuts, band selection, magnitude↔flux,
  phase, plotting), **2** models (built-ins, `predict`, the redback kilonova, `register_model`),
  **3** samplers (`fit_ABC`/`fit_ABC_SMC`/`fit_MCMC`/`fit_SNPE`, `SamplerResult`, AIC/BIC), **4**
  visualizing results (`plot_corner`, `recovery_metrics`, `waic`), and **5** a full Bayesian model
  comparison — AT2017GFO fit with one/two/three-component kilonova models × ABC/SNPE/MCMC, ranked by
  AIC/BIC. Cells are generated from `notebooks/_build_notebooks.py`; linked from the README.

### Repository reorganization — analysis / benchmarks / dev separated from the package
- **`analysis/at2017gfo_villar/`** now holds the full AT2017GFO Villar+17 study — `villar.py`,
  `villar_plots.py`, `fetch_at2017gfo_full.py`, `preprocess_at2017gfo.py`, `runtime_plot.py`, plus its
  own `data/`, `figures/` and `reports/`. **`sanity_check/`** holds the synthetic-recovery benchmarks
  (`sanity_check.py`, `benchmark_*.py`, `corner_kilonova_benchmark.py`, `compare_samplers.py`,
  `sanity_mck19_modes.py`, `BENCHMARK.md`) and their figures. **`dev/`** holds one-off demos and
  superseded fit/report scripts. The importable package (`whisper_cbpf/`) and the `tests/` suite are
  untouched; only the g/r/i `tests/data/at2017gfo.csv` stays a shared fixture.
- **Preprocessed AT2017GFO reduction** (`analysis/at2017gfo_villar/preprocess_at2017gfo.py`): drops the
  lone Swift-UVOT `uvw1` band, cuts to SNR > 5, and collapses near-simultaneous same-band duplicates to
  one point per `round(MJD, 2)` epoch (607 → 416 detections). Fit by all 7 methods, giving a fourth
  AT2017GFO variant alongside g/r/i, full-UVOIR magnitude, and full-UVOIR flux. A standalone
  wall-clock benchmark bar chart (`villar_runtime.png`) is linked from the README.
- **Early-time peak-timing diagnostic** added to the report interpretation: the best-fit curve peaks a
  few tenths of a day after the brightest observed point in the reddest optical bands — present in both
  magnitude and flux space, so a model-adequacy limitation rather than a units/weighting artifact.

### Full UV–optical–NIR AT2017GFO application + SNPE robustness against pathological real-data conditioning
- **Full-UVOIR data.** `analysis/at2017gfo_villar/fetch_at2017gfo_full.py` pulls the complete AT2017GFO photometry from
  the Open Astronomy Catalog (607 detections, 18 bands, 0.5–25 d, Vega→AB converted where needed) into
  `analysis/at2017gfo_villar/data/at2017gfo_full.csv`. `two_component_kilonova._redback_band` now maps any band redback's
  filter table knows (`H`/`J`/`Ks`/`Y` → 2MASS, `U`/`B`/`V`/`R`/`I` → Bessell, `uvot::*` → Swift UVOT),
  not just optical grizy, so the model can be fit to the full UV→optical→NIR range.
  `analysis/at2017gfo_villar/PLAN_fullband.md` records the acquisition plan and cost budget.
- **Physical ejecta-velocity prior.** `analysis/at2017gfo_villar/villar.py`'s real-data prior tightened
  `vej_1`/`vej_2` from `Uniform(0.01, 0.7)` to `Uniform(0.05, 0.3)` (physical kilonova range) — the old
  bound let the MAP rail to an unphysical 0.7 c, a genuine-but-unphysical optimum caused by the
  permissive prior combined with g/r/i-only data underconstraining the red component. With the full
  UVOIR data the red component (κ_red, v_red, T_red) now moves off the prior edges toward
  Villar+2017-consistent values — the payoff of the NIR coverage g/r/i alone cannot supply.
- **Flux-vs-magnitude comparison.** `space="flux"|"magnitude"` now routes through the whole real-data
  application (`VILLAR_SPACE` env var): data, simulator noise, likelihood and the σ scatter prior
  (`LogUniform` bounds are space-appropriate — Jy for flux, mag for magnitude) are all consistent in
  the chosen space. Fixed a units bug where MCMC's ABC-seeded σ warm-start was hardcoded to a
  magnitude-scale value (0.2) regardless of space, silently placing flux-space MCMC's σ walker outside
  its own `LogUniform(1e-8, 1e-3)` Jy prior.
- **SNPE input normalisation (`standardize_x="asinh"|"zscore"|"none"`).** Flux-space light curves span
  ~6 orders of magnitude; sbi's built-in z-scoring fixes scale but not skew, which left flux-space
  neural SBI poorly conditioned. `asinh(x/scale)` (per-channel `scale` fit on round-0 simulations,
  applied identically to simulations, the observation and `result.format_x`) variance-stabilises the
  input, making flux as well-conditioned as magnitude space.
- **Band removed as a stacked-input channel.** Every SNPE simulation is drawn on the identical
  `(time, band)` grid as the observation, so band identity is already encoded positionally — a constant
  per-position band channel added no information and empirically hurt flux-space fits (removing it
  improved recovered κ_red from 2.34 to 3.37 cm²/g, closer to Villar+17's 3.65).
- **Robust post-fit max-likelihood scan.** The scan re-runs the forward model on up to `max_logl_scan`
  posterior draws to get the exact best-fit/AIC/BIC; a single pathological draw calling an expensive
  model (e.g. redback, ~0.2–0.3 s/call) could hang the entire fit for hours. Now runs in a process pool
  with a wall-clock `scan_timeout` (default 300 s), degrading to a smaller scored subset on timeout
  rather than hanging (root-caused a 10.8 h flux-space SNPE hang; fixed to ~200 s).
  Confirmed **not** a GPU or simulation-parallelism issue (both were already healthy).
- **Robust final posterior sampling against real-data pathologies.** Conditioning a trained density
  estimator on the *real* observation (rather than the simulated placeholders seen in training) can
  expose two failure modes sbi's default rejection sampling does not handle gracefully: a near-zero
  acceptance rate (technically not an error, just impractically slow — sbi's own diagnosis loop is
  itself just as slow, since it estimates the correction factor by accept-reject sampling until enough
  are accepted) and a numerically degenerate spline coefficient in the flow's inverse transform
  (`AssertionError` deep in `nflows`, for flexible estimators like NSF). `SNPESampler.fit` now probes
  the acceptance rate with a single bounded, hang-proof sample (drawn directly from the density
  estimator, cost independent of the acceptance rate) and catches the `AssertionError`, falling back to
  MCMC-based posterior sampling (conditions via the numerically-stable forward `log_prob` instead of the
  flow's inverse) in either case. The same acceptance factor is pre-cached on each round's posterior so
  sequential/truncated SNPE's internal proposal-restriction step (`get_density_thresholder`) reuses it
  instead of re-deriving it the slow way. `result.info["final_sample_method"]`
  (`"rejection"`/`"mcmc_fallback"`) and `["final_sample_acceptance_rate"]` record which path was taken.
  Also fixed the same exposure in `analysis/at2017gfo_villar/villar.py`'s amortized-resample timing benchmark.
- **PPC plot readability.** Taller per-band posterior-predictive panels (shared y-axis, range set from
  data + model medians rather than σ-inflated tails) so bands separate cleanly. New supplementary
  **`villar_ppc_grid.png`**: the same check zoomed to the first 10 days, laid out as one square panel
  per method with larger labels and bolder, edge-outlined data markers, for a closer read of band-by-band
  structure where the two components pull apart fastest.
- **Rail/σ reporting fixes** in `analysis/at2017gfo_villar/villar_plots.py`: prior-rail detection now judges proximity
  to the bound *value* (not fraction of prior range, which misreported well-constrained values under
  wide priors like κ∈[1,30]) and is keyed on the MCMC reference posterior rather than any single
  (possibly broad neural) method; the reported σ is MCMC's own value with space-aware units (mag vs Jy)
  rather than a cross-method median dragged up by broad neural posteriors.

### Villar+2017 AT2017GFO application: free-scatter likelihood, ABC comparison space, parallel MCMC
- **`GaussianLikelihoodWithScatter`** (`kind="gaussian_scatter"`/`"villar"`): Gaussian likelihood with
  a **free extra-scatter term σ added in quadrature** to the reported errors — Villar et al. 2017
  (ApJL 851 L21) Eq. 4 in its correctly normalized form, `lnL = −½Σ[(O−M)²/(σᵢ²+σ²) +
  ln(2π(σᵢ²+σ²))]`. σ is a *likelihood* parameter sampled with the rest: MCMC routes a prior
  parameter named `sigma` via `likelihood="gaussian_scatter"`; ABC/ABC-SMC/SNPE take
  `scatter_param="sigma"` and fold it into their **generative simulation noise**
  (`N(0, √(σᵢ²+σ²))` per draw). Verified on synthetic data with mis-reported errors: MCMC and the
  exact likelihood recover the injected scatter; neural SBI constrains it weakly (a noise level is a
  distributional feature of one realization); and a plain χ² rejection distance **cannot** identify
  it (monotonically penalised by extra noise — its ABC posterior collapses to zero), so the ABC
  family fits the physical parameters only, documented in the report.
- **ABC/ABC-SMC `space="auto"|"flux"|"magnitude"`**: the comparison space now routes the acceptance
  itself — data, simulations, noise and distance all live in the chosen space (previously flux-only).
  Sampled parameters are now `prior.names` (a prior may carry non-model parameters).
- **MCMC `n_jobs`**: emcee walker likelihoods in a process pool — makes an 8-D fit with the ~0.1 s
  redback kilonova likelihood run in minutes.
- **`analysis/at2017gfo_villar/villar.py` + `villar_plots.py`**: the real-world application — the
  `two_component_kilonova` with **κ_blue = 0.5 fixed**, z fixed, κ_red + both temperature floors
  free + σ, fit in magnitude space by 7 methods (MCMC, ABC, ABC-SMC, NPE-MDN, NPE-NSF, SNPE-5r-NSF,
  SNPE-5r-NSF+TCN), rendered to `analysis/at2017gfo_villar/reports/REPORT_at2017gfo_villar.md` with annotated posterior
  histograms, an all-method corner, magnitude-space posterior-predictive light curves and a
  parameter/runtime summary.

### Calibrated likelihood-free inference + neural-SBI performance upgrade
- **Noise-matched ABC/ABC-SMC (`simulate_noise=True`, new default).** Every simulation now adds
  per-point white noise from the reported errors (`N(0, flux_err)`), matching the generative model of
  the data — this is what makes ABC exact as ε→0 and its posterior **width calibrated** (a noiseless
  simulator under a hard cut targets a likelihood shell). Reproducibility across `n_jobs` is preserved
  (noise drawn from each simulation's own RNG stream). **Scale note:** distances now include the
  simulation noise (`E[D] ≈ χ² + n_points`), so fixed `threshold`/`epsilon_schedule`/float
  `min_epsilon` values from the noiseless era must be re-derived (adaptive quantiles handle it).
  `best_params` (and AIC/BIC there) are now selected by the **exact Gaussian log-likelihood** over
  accepted draws — never by the noisy distance, whose argmin is the luckiest noise realization.
- **`whisper_cbpf.embeddings` (new):** `MLPEmbedding` and `TCNEmbedding` (Temporal Convolutional
  Network — dilated causal 1-D convolutions with residual blocks, avg+max-pool head), buildable via
  `fit_SNPE(..., embedding_net="mlp"|"tcn", embedding_latent=...)` and trained jointly with the
  density estimator.
- **`fit_SNPE` input & speed:** `x_format="stacked"` conditions the network on the full observation
  tuple `(value, error, time, band)` — the same information the likelihood-based samplers get;
  `predict_torch=` accepts a batched torch forward model and replaces the per-row Python simulator
  with a single on-device call (**~2000× faster simulation**; 30k Bazin simulations in milliseconds);
  `result.format_x` maps a raw vector to the network input for amortized reuse (same observing grid).
  Fixed a CUDA device mismatch when truncated SNPE (`proposal_mode="restricted"`) is combined with
  `predict_torch` (sbi's `RestrictedPrior` samples on CPU by default). Embedding weights are now
  initialized after `torch.manual_seed`, making embedding benchmarks run-to-run reproducible.
  Multi-GPU data-parallel training was evaluated and rejected: sbi 0.23 has no native support and
  these networks are far too small to amortize synchronization — the effective multi-GPU strategy is
  one method/config per GPU (how the benchmarks run).

### Inference-validation tools + synthetic-recovery sanity check
- **`whisper_cbpf.validation`** — reusable, sampler-agnostic checks that a fit *recovered the truth*
  with *reliable uncertainties*: `recovery_metrics` (per-parameter bias, standardized z-score, 68/95%
  credible-interval coverage vs a known truth), `posterior_predictive_check` (predictive band, best-fit
  reduced χ², noise-inflated predictive coverage, Bayesian p-value) and `sbc_rank` / `sbc_ranks`
  (Simulation-Based Calibration — Talts 2018 / Säilynoja 2022; rank uniformity with a χ²-of-uniformity
  p-value flags over-/under-confidence). Exposed as `wp.recovery_metrics` etc.
- **ABC-SMC `min_epsilon` (ε floor).** The adaptive schedule could drive ε all the way to χ²_min,
  collapsing the posterior onto the MLE — spuriously **overconfident** (on the synthetic 2-param recovery
  the raw run gave |z|≈8 with 0% coverage). `min_epsilon="auto"` floors ε at **χ²_min + 2(k+2)** (k =
  #parameters), which reproduces the Gaussian posterior width and restores |z|≲2 with nominal coverage;
  a float sets a fixed floor. Default `None` keeps the old behavior.
- **`sanity_check/sanity_check.py` + `sanity_check/sanity_check_plots.py`** — end-to-end recovery benchmark on
  synthetic data with known ground truth: fits a physically-motivated **Bazin (2009) supernova** light
  curve (headline showcase), a 4-param damped sinusoid (correlated/oscillatory stress test) and a
  2/4/6-param Gaussian-pulse sweep, timing every sampler, and renders posterior histograms, an
  all-sampler corner, posterior-predictive checks, SBC rank histograms, a recovery/speed/scaling summary
  and `REPORT.md` into `sanity_check/figures/`. Showcase noise seeds are screened non-adversarial
  (worst |MLE−truth|/σ_Fisher ≲ 1) and the choice is disclosed in the report — single-realization tables
  compare methods; **SBC over many unscreened realizations is the calibration evidence**. On the damped
  sine, exact MCMC calibrates, NPE-MAF trails mildly over-confident, and the ABC family shows its
  width errors (rejection ABC under-confident; the diagonal-kernel ABC-SMC can't fully capture the
  correlated posterior) — exactly the approximation error SBC is designed to expose.
- **Neural SBI at scale: MDN + NSF, GPU-parallel, no embedding net.** The Bazin showcase fits
  `npe_mdn`/`npe_nsf` (amortized, 1 round × 30k simulations) and `snpe_mdn`/`snpe_nsf` (sequential,
  10 rounds × 3k) with the density estimators conditioning **directly on the raw light-curve vector**;
  each method trains on its own GPU, so all four neural fits run in parallel. Results: **every method
  recovers the truth within 1σ** (reduced χ² ≈ 1); **NPE-NSF is formally SBC-calibrated** (min
  rank-uniformity p = 0.057; three of four parameters p > 0.5) alongside exact MCMC (p = 0.229);
  NPE-MDN is the fastest accurate neural method (371 s end-to-end, best point recovery 0.24σ) but
  marginally over-confident (p = 0.036). Note: sbi 0.23.3's SNPE-C **non-atomic MoG loss has a CUDA
  device-mismatch bug** (triggered when the proposal is an MDN posterior), so `snpe_mdn` runs the
  **truncated sequential scheme** (`proposal_mode="restricted"`) — documented in the script.

### Scientific-review fixes (physical + Bayesian correctness)
A systematic, adversarially-verified review (physical consistency, Bayesian rigor, numerics) fixed:
- **Cross-sampler AIC/BIC now comparable.** ABC/ABC-SMC previously reported `-2 lnL` from the bare
  flux χ² (Gaussian normalisation dropped, flux-only), so their AIC/BIC were on a different scale than
  MCMC/SNPE — invalidating any model-selection table that mixed samplers. They now evaluate the **exact
  Gaussian log-likelihood at the best fit** in the data's natural space (`info['likelihood_space']`).
- **ABC-SMC is now importance-weighted** (Beaumont 2009 / Toni 2009): weighted resampling, an adaptive
  diagonal-Gaussian kernel (2× weighted population variance), and weights
  `w_i ∝ π(θ_i)/Σ_j w_j K(θ_i|θ_j)`; the equal-weight resampled posterior is returned and per-round
  effective sample size is recorded. The old **unweighted** variant biased the posterior.
- **`mck19` blackbody was missing a factor of π** (`F = πB(R/D)²`) — every predicted flux was π (~1.24
  mag) too faint (an error inherited from the reference). Fixed.
- **WAIC now drops non-finite *draws*, not data *points*** — a single bad draw no longer removes an
  entire data point, which had made ΔWAIC between models incomparable (different `n_data`); returns
  `n_draws_dropped` and `subsampled` and warns on both.
- **MCMC AIC/BIC use the max-*likelihood* draw**, not the max-*posterior* draw (these differ under a
  LogUniform prior); plus a convergence warning when `nsteps < 50·τ` (`info['converged']`).
- **CCM89 extinction clamps** out-of-range bands to the nearest regime (was silently `A=0`, e.g. for
  JWST/mid-IR) with a warning; `flux_density_to_mag` returns `NaN` + warns on non-positive flux (was a
  silent NaN with a sign-flipped error); SNPE guards an all-non-finite likelihood scan; ABC warns on an
  empty (0-accepted) posterior; the magnitude-space flux floor is documented.

### `plot_corner` + WAIC
- **`wp.plot_corner(posteriors, ...)`** — a built-in, documented corner plot for **overlaying a list of
  posteriors** (SamplerResults, DataFrames, dicts, or arrays) on one publication-ready figure: shared
  per-parameter ranges so panels align, a **dark, distinct palette** (`CORNER_PALETTE`), contour lines
  (not filled) so overlaps stay legible, `log_params=` for log axes, `truths=`, and a colour→label
  legend. `sanity_check/corner_kilonova_benchmark.py` uses it for the AT2017GFO benchmark posteriors
  (`sanity_check/figures/at2017gfo_corner_flux.png`) — the corner shows whether the samplers are *compatible*
  (full posteriors + uncertainties), which the point-estimate table cannot.
- **`wp.waic(posterior, lc, model, ...)`** — the **Widely Applicable Information Criterion** (Watanabe
  2010), a fully-Bayesian fit score (lower is better) that uses the *whole* posterior: returns `waic`,
  `lppd`, `p_waic` (effective #params), `se`, with a `fixed=` hook for pinned parameters. Backed by a
  new pointwise log-likelihood (`GaussianLikelihood[WithUpperLimits].log_likelihood_pointwise`).

### SNPE on GPU
- **SNPE can now train on a GPU.** `fit_SNPE(..., device=...)` accepts `'cpu'` (default), `'cuda'` /
  `'gpu'` / `'cuda:N'`, or **`'auto'`** (CUDA when available, else CPU). The torch prior and observed
  data are now placed on the chosen device (fixing the sbi *"prior device must match training device"*
  error), and requesting a GPU without one **warns and falls back to CPU** instead of crashing.
- The GPU accelerates the neural-network **training**, not the (CPU) simulator — so it helps most with
  many simulations / large networks. `sanity_check/benchmark_snpe_device.py` measures GPU-vs-CPU runtime
  across a ladder of simulation counts (estimating each tier's time before running it, with a
  `--budget` guard) and saves a log-log plot (`sanity_check/figures/snpe_device_benchmark.png`).
- Tested: `_resolve_device` mapping/fallback (no GPU needed) + a slow on-GPU recovery test
  (`skipif` no CUDA).

### Kilonova flux-vs-magnitude benchmark
- New **timed benchmark + sanity check** ([`sanity_check/BENCHMARK.md`](sanity_check/BENCHMARK.md),
  `sanity_check/benchmark_kilonova_modes.py`): `two_component_kilonova` fit to AT2017GFO (g/r/i) in
  **flux** vs **magnitude** space with ABC / MCMC / SNPE (6 configs). Records per-config **runtime**,
  AIC, RMS and posterior size; each config writes its own result file so the six run in parallel.
- **Publication-quality report figure** (`sanity_check/figures/kilonova_benchmark_report.png`): larger fonts,
  colourblind-safe (Okabe–Ito) band colours, line-style per sampler, inward ticks, clear unit-labelled
  axes, and an **embedded table of the best-fit ejecta parameters + metrics per configuration**.
- Tested (`tests/test_benchmark_kilonova.py`): `setup` + magnitude-space distance (no redback) + a
  slow, guarded end-to-end fit→report.

### `two_component_kilonova` — first redback-backed model
- New built-in **`two_component_kilonova`** (`whisper_cbpf/models/two_component_kilonova.py`): a
  blue (low-κ) + red (high-κ) kilonova via the optional **redback** package (`[models]` extra), wrapping
  redback's `two_component_kilonova_model`. Parameters: `mej_1/2`, `vej_1/2`, `kappa_1/2`,
  `temperature_floor_1/2`, `redshift` (default prior follows the Darc kilonova-simulation setup).
- **redback is imported lazily** — WHISPER and `list_models()` work without it; only `predict` needs the
  extra (clear `ImportError` pointing to `pip install 'whisper-cbpf[models]'` otherwise). This is the
  template for a planned series of redback-backed models.
- Band-dependent: WHISPER bands → redback LSST filters (`g→lsstg`…); redback's band-integrated AB
  magnitude is converted to WHISPER's canonical **flux density (Jy)** as an exact (machine-precision)
  round-trip, so the shared likelihood/samplers treat it like any other model. Expensive simulator
  (~50 ms/call) → **SNPE** is the natural sampler; `predict` is module-level (parallel-ABC safe).
- `dev/demo_kilonova.py` (light curve) and `dev/fit_kilonova_at2017gfo.py` (ABC/MCMC/SNPE fit of
  AT2017GFO — which, being a real kilonova, this model fits well, unlike `mck19`).

### `mck19` physical model — BBH merger in an AGN disk
- New built-in **`mck19`** model (`whisper_cbpf/models/mck19.py`): the optical flare from a
  binary-black-hole merger embedded in an AGN accretion disk. A GW-recoil-kicked remnant shocks a
  bound-gas **hotspot** that radiates as a blackbody — a `sin²` rise to the ram-pressure delay `t_ram`,
  then exponential decay back to the disk baseline. McKernan et al. 2019
  ([ApJL 884, L50](https://iopscience.iop.org/article/10.3847/2041-8213/ab4886)); implementation of
  [Darc 2025](https://arxiv.org/abs/2506.02224). Parameters: `v_kick`, `M_smbh`, `M_bh`, `r_bh`,
  `redshift` (default prior spans the Darc 2025 grid).
- **First band-dependent built-in:** returns flux density (Jy) per `(time, band)`, evaluating the
  hotspot + disk blackbody at each band's effective wavelength (via Whisper's band system) and the
  source redshift (Planck18 luminosity distance + time dilation). Self-contained — astropy
  constants/cosmology only (no `speclite`/`extinction`); the AB magnitude is the monochromatic-at-`λ_eff`
  approximation to the original LSST filter integration. Fits with every sampler through the shared
  likelihood (an MCMC recovery test confirms data-mode-consistent magnitude-space fitting).
- `dev/demo_mck19.py` renders the g/r/i light curve (`dev/figures/mck19_lightcurve.png`).

### MCMC sampler (emcee)
- New **`MCMCSampler`** (`mcmc`, `fit_MCMC`) — affine-invariant ensemble MCMC via `emcee` (a core
  dependency). The log-posterior is the Whisper prior + the **shared likelihood layer**
  (`make_likelihood`), so MCMC uses the *same physically consistent, `data_mode`-aware likelihood* as
  ABC/ABC-SMC/SNPE (flux data → flux space, magnitude data → magnitude space). Walkers init from the
  prior (or a given `initial_guess`); sampling is **seeded/reproducible**; exact Gaussian AIC/BIC; the
  `emcee.EnsembleSampler` is attached as `result.emcee_sampler`.
- `sanity_check/compare_samplers.py` — sanity check that ABC / ABC-SMC / MCMC / SNPE converge to the **same
  posterior** on `gaussian_rise`, with an overlaid corner plot.

### Production-readiness pass
- **Reproducibility (scientific bug fix):** ABC and ABC-SMC results are now **independent of `n_jobs`**.
  Each simulation/attempt owns an RNG stream derived from its *global index* (not the worker), so a fixed
  `seed` gives identical posteriors regardless of core count. SNPE seeds the simulator's observational
  noise per parameter row (same reason — a shared RNG made `num_workers>1` add identical noise). Locked
  by new determinism tests.
- **Four pluggable axes, all symmetric:** added `register_likelihood`/`list_likelihoods` and
  `register_distance`/`get_distance`/`list_distances` to match `register_model`/`register_sampler`;
  samplers now accept `distance=` by name. `register_manual_band` documents its process-global scope and
  gains `unregister_manual_band`/`clear_manual_bands`.
- **Packaging / open-source-ready:** added a `LICENSE` file (GPL-3.0), a `py.typed` marker (shipped via
  package-data) + `Typing :: Typed` classifier, single-sourced the version (`dynamic`), Python-version
  classifiers, and Documentation/Issues/Changelog URLs. Return-type hints on the public entry points.
- **Docs:** new `docs/DESIGN.md` (design rationale + known limitations), `CONTRIBUTING.md`, `CITATION.cff`;
  ABC/ABC-SMC `fit` docstrings; magic-number guards promoted to named constants; doc accuracy fixes.

### LightCurve is now an `astropy.table.Table`
- `LightCurve` **subclasses `astropy.table.Table`**: per-point quantities are columns, scalar metadata
  lives in `.meta`, and full table semantics work — `lc['new'] = lc['flux'] + 5`, boolean-mask slicing
  (`lc[lc['magnitude'] < 18]`, which keeps the subclass + `.meta`), `sort` / `group_by`, etc. `lc()`
  returns the table itself. The common quantities stay available as attributes (`lc.time`, `lc.flux`,
  `lc.redshift`, …) so the samplers/likelihood/plotting are unchanged. `to_dataframe()` → `to_pandas()`.
- New methods: **`where(**constraints)`** (`col` / `col_min` / `col_max` / `col_not`, list = OR);
  **`calc_phase(reference=, redshift=, peak=, hours=)`** — rest-frame phase `(t − ref)/(1+z)`;
  **`calc_absmag(dm=, redshift=, ebv=, rv=3.1, extinction=)`** — distance modulus (from `z` /
  `luminosity_distance`) + Milky-Way extinction (CCM89 from `ebv`/`rv`, or an explicit `{band: A_mag}`
  dict) → an `absmag` column. Inspired by `lightcurve_fitting.LC`.

### Survey-CSV ingestion: data_mode, redshift, astropy units, SVO band fallback
- `LightCurve.data_mode` is now a stored attribute in `{flux_density, magnitude, flux}` (default
  inferred from the columns; explicit override) with an `output_format` property (`magnitude` /
  `flux_density`) — the forward-model comparison space (the optional redback backend uses the same).
- **Redshift** is resolved `redshift=` argument > `redshift` column > unknown. Unknown does *not* fail:
  the curve records `redshift_known=False` + a default `redshift_prior` and warns that `z` will be
  sampled. Validation: `z >= 0`; `z == 0` requires an explicit `luminosity_distance`; negative/NaN is fatal.
- **astropy.units throughout** (`io/units.py`): `flux_density` accepts F_nu (Jy/mJy/µJy) or F_lambda
  (erg/s/cm²/Å) and stores Jy canonically, converting F_lambda via `u.spectral_density(lambda_eff)`
  (errors clearly when the per-point effective wavelength is missing); `magnitude` must be dimensionless
  AB; band-integrated `flux` validates erg/s/cm² dimensionality. No-unit columns warn and apply a
  documented per-mode default.
- **Bands** resolve to per-point effective wavelength + zero point via `FILTER_LOOKUP` anchored to the
  LSST ugrizy zero points (`resolve_band` / `resolve_bands`, `LSST_BAND_INFO`). A miss warns and falls
  back to the **SVO Filter Profile Service** (`io/svo.py`, `astroquery.svo_fps`) — results cached by
  filter ID (offline-safe; corrupt/unusable cache entries are ignored, never crash a load), graceful
  degradation on network failure with a `register_manual_band` override path. astroquery is an optional
  `[svo]` extra; all SVO/network calls are mocked in tests.
- Note: `add_flux`/`add_mag` keep the constant **AB 3631 Jy** zero point so the modelling flux the
  samplers/likelihood consume stays on one zero point across bands; pass `zeropoint_jy=lc.zero_point` to
  opt into the per-band (LSST/SVO) zero points.

### Data ingestion & plotting
- `LightCurve` canonical container with pandas-style `select_*` / `add_*` methods, an `snr` property,
  `set_explosion_date`, and upper-limit support.
- `load_lightcurve` — flexible CSV loader: column auto-detection, band normalization + `FILTER_LOOKUP`
  broadband grouping, mag↔flux conversion, quality cuts, SNR cut (`min_snr`), time window, explosion date.
- `plot_light_curve` — report (magnitude + flux) and per-band grid layouts; markers for detections
  (circles), SNR<3 (△) and upper limits (▽); apparent/absolute magnitude or flux.

### Inference (pluggable models + samplers)
- Priors: `Uniform`, `LogUniform`, `Prior` (sample / log_prob / rescale; picklable for parallel ABC).
- Models: registry (`register_model` / `list_models`) + built-ins **`flare`**, **`bazin`**,
  **`gaussian_rise`**.
- Distance: `chi2_distance` (pluggable; any `f(obs, err, sim, bands) -> float`).
- Samplers: registry (`register_sampler` / `list_samplers`) + **`ABCSampler`** (parallel rejection),
  **`ABCSMCSampler`** (sequential, adaptive or explicit epsilon), and **`SNPESampler`** — **Sequential
  Neural Posterior Estimation** (`snpe`/`npe`, `fit_SNPE`) via `sbi`: the simulator is Whisper's forward
  model + Gaussian noise, the prior is auto-adapted to torch (`Uniform`→`BoxUniform`, mixed→
  `MultipleIndependent`), `num_rounds>1` is sequential, and the trained sbi posterior is attached as
  `result.posterior`. **Flexible:** a custom `embedding_net` (`torch.nn.Module`), a custom
  `density_estimator` (or `posterior_nn` `hidden_features`/`num_transforms`/`num_bins`), parallel
  `num_workers`, and `proposal_mode="restricted"` for truncated SNPE (`RestrictedPrior` +
  `get_density_thresholder`, with `support_samples` capping the otherwise-1e6 support estimate). Lazy
  optional `[sbi]` extra. `SamplerResult` with posterior summary, best-fit, AIC / BIC / max-log-likelihood
  (χ² ≙ −2 ln L; SNPE uses the exact Gaussian value), `to_json`.
- Notebook quick-start: `examples/at2017gfo_quickstart.ipynb` — load AT2017GFO, register a **custom
  model**, fit with ABC, compare models, and run SNPE.
- Likelihoods (`likelihood.py`): `GaussianLikelihood`, `GaussianLikelihoodWithUpperLimits`,
  `MixtureGaussianLikelihood`, `make_likelihood` — **flux or apparent-magnitude space**, default by
  data type, upper limits in flux/mag. _Standalone + tested; sampler integration pending._

### Fixed
- `flare` returns 0 before the explosion (was negative for `t<0`); `bazin` is evaluated in log-space
  (numerically stable — no early-time plateau for `tau_rise < tau_fall`).
- SNPE: the trained `result.posterior` now has its default `x` set every round, so
  `result.posterior.sample()` works without re-passing `x` (regressed in the truncated-proposal refactor
  for `num_rounds=1`).

### Packaging & docs
- pip-installable from GitHub; relaxed dependency pins (no forced numpy/scipy downgrade); redback is an
  optional `[models]` extra — Phase-1 data + plotting + ABC run with no redback and no compiler.
- Tutorial, API reference, design rationale, extensibility + contributing guides, an AT2017GFO
  model-comparison report, and a quick-start notebook. LICENSE (GPL-3.0), CITATION.cff, py.typed. 189 tests (redback-backed model + benchmark tests skip without the [models] extra).
