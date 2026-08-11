# Whisper — design rationale & architecture

This document explains the **why** behind Whisper's structure so a contributor can understand, trust,
extend, and maintain it without reverse-engineering the source. For the *how-to*, see
[`TUTORIAL.md`](TUTORIAL.md), [`API_REFERENCE.md`](API_REFERENCE.md), and [`EXTENDING.md`](EXTENDING.md).

## 1. Goals & philosophy

Whisper answers one question — *"which model best fits this transient light curve, and what are the
posteriors?"* — and is built for use in real transient-characterization pipelines. Design priorities, in
order: **clarity > cleverness; explicit > hidden magic; reproducibility > convenience; extensibility >
hardcoding; user experience > developer convenience; long-term maintainability > short-term speed.**

## 2. Architecture at a glance

```
load_lightcurve ──► LightCurve (astropy.Table)
                         │
        ┌────────────────┼─────────────────────────────┐
     models           samplers          likelihoods / distances
   (register_model) (register_sampler)  (register_likelihood / register_distance)
        │                │                         │
        └──────► fit(lc, model, sampler) ──► SamplerResult (posterior, AIC/BIC, JSON)
```

**Four pluggable axes**, each a small name registry with `register_*` / `list_*` / `get_*` helpers:
**models**, **samplers**, **likelihoods**, **distances**. "Easy to add a new one" is a core goal —
adding any axis member is ~1 function and needs no edits elsewhere.

## 3. Key decisions (and why)

- **`LightCurve` *is* an `astropy.table.Table`.** Astronomy-native (units, masked columns, `group_by`,
  ascii/FITS I/O), and lets users compute on columns directly (`lc['x'] = lc['flux'] + 5`). The common
  quantities are mirrored as attributes (`lc.flux`, `lc.time`, `lc.redshift`) so consumers and
  beginners keep a simple interface. Scalar metadata lives in `.meta`, which astropy propagates through
  slicing — so subsetting preserves redshift/data_mode for free.
- **redback is optional and auxiliary.** Everything (ingestion, samplers, likelihoods, plots) is
  Whisper's own and runs standalone. redback (the `[models]` extra) is used *only* as a source of
  physical models + priors — never as a runtime dependency of the core.
- **Flux is the canonical model output; likelihoods choose the space.** Models predict flux density;
  the likelihood compares in `flux` or `magnitude` space. This keeps models space-agnostic and makes
  the comparison space a user choice (`space='auto'|'flux'|'magnitude'`).
- **χ² distance ≙ −2 ln L (Gaussian),** so ABC can report `max_log_likelihood`, `AIC`, `BIC` for model
  comparison without an explicit likelihood evaluation.
- **Reproducibility is seed-based and `n_jobs`-independent.** Each simulation/attempt owns an RNG stream
  derived from its *global index* (not the worker), so a fixed `seed` gives identical posteriors
  regardless of core count or `n_jobs`. SNPE seeds the simulator's noise per parameter row for the same
  reason. Parallelism affects speed only, never the science.
- **Picklability constraint.** For parallel ABC, model `predict`, priors, and distances must be
  picklable (module-level, not closures) — this shapes those APIs (small picklable classes/functions).
- **SVO band fallback behind one mockable boundary.** All network access goes through three private
  `_svo_fetch_*` functions, cached by filter ID (offline-safe). Tests mock that boundary, so CI never
  hits the network and the package imports without `astroquery`.

## 4. Known limitations

- **ABC posteriors are approximate** (broadened by the acceptance ε); tighten ε or use ABC-SMC.
- **ABC acceptance is flux-χ² only.** ABC/ABC-SMC still *accept* on the flux-space χ² distance (requires
  errors); selecting `GaussianLikelihoodWithUpperLimits` / `MixtureGaussianLikelihood` or a magnitude
  space for the *acceptance* is a planned `likelihood=` / `space=` parameter (see §5). The reported
  `AIC`/`BIC`/`max_log_likelihood`, however, now use the **exact Gaussian log-likelihood at the best fit**
  in the data's natural space — the same convention as MCMC/SNPE, so they are **comparable across
  samplers** (`info['likelihood_space']`).
- **Built-in models** are the analytic toys (`flare`, `bazin`, `gaussian_rise`), the physical
  band-dependent **`mck19`** (BBH-in-AGN flare), and the redback-backed **`two_component_kilonova`**
  (`[models]` extra); further physical models come from redback.
- **`calc_absmag` extinction** uses the CCM89 law (clamped to its 0.3–8 µm⁻¹ validity, with a warning
  for out-of-range bands) and applies no Vega→AB offsets for NIR bands; host extinction beyond an
  explicit `extinction` dict is not modeled.
- **WAIC is posterior-quality-sensitive:** `p_waic` (and hence WAIC) inflates for posteriors much
  broader than the likelihood (ABC tolerance / under-converged SNPE) — a useful diagnostic, but treat
  WAIC as reliable only for well-converged posteriors (a robust PSIS-LOO is a possible addition).
- **SVO mapping** uses a ±5% wavelength window and documented default filter IDs; ambiguous matches warn
  and pick the closest, which may not be the intended filter — override with `register_manual_band`.

## 5. Planned extensions

- `likelihood=` / `space=` routed into the ABC/ABC-SMC **acceptance** (not just the reported metric).
- Unified sampler keyword names across ABC/ABC-SMC/MCMC/SNPE (with deprecation aliases).
- Nested sampling (Dynesty) for direct evidence; PSIS-LOO as a robust complement to WAIC.
- Full type-hint coverage across all internal modules (the public surface is annotated and `py.typed`).
- More redback model adapters; a `fit_all` grid (transient × model × sampler) with a comparison report.

*Delivered since the first draft: MCMC (emcee), importance-weighted ABC-SMC, the `mck19` and
`two_component_kilonova` models, GPU SNPE, `plot_corner`, and `waic`.*
