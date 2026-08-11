# INVENTORY.md — WHISPER-CBPF merge, Phase 0

Read-only inventory. Nothing was written to either source repo; nothing was written to the target.

## 0. What was inspected

| Repo | Ref | SHA | Date |
|---|---|---|---|
| `WHISPER_AI` | `origin/main` | `8ba3843` (`8ba3843…`) | 2026-08-11 |
| `whisper-GPU` | `origin/main` | `10796a0` (`10796a0…`) | 2026-08-11 |

Both fetched after the user's push. Fresh clones in scratch; the user's working checkouts were read
(`git log` / `rev-parse` / `diff`) but never modified.

**Gate check (settled decision #3): PASSES.** `models/tde.py`, `tests/test_tde_vs_redback.py` and
`benchmarks/tde_vs_redback/` are all tracked on `whisper-GPU` `origin/main`.

**Baselines superseded.** The brief's pre-merge numbers were taken before the push:

| | Brief says | Actual at the inspected SHAs |
|---|---|---|
| CPU tests | 219, of which 7 `slow` | **222 collected** (221 passed, 1 skipped — verified in `phe_sbi`), 7 `slow` markers |
| GPU tests | 84 passed, 2 skipped | **144 test functions** in 21 files — needs a GPU-host run to establish pass/skip |

The CPU 219 was correct for `dc1e331`; `main` has since advanced by two commits that add 3 SNPE
tests. The GPU 84/2 predates both the TDE and supernova work.

## 1. Scope decisions taken (user, this session)

1. **`jax_gpu/` is a legacy precursor — excluded.** It is now *on* `WHISPER_AI` `origin/main`
   (14 tracked files) after the feature branch merged, so exclusion must be **deliberate and
   recorded in `FINDINGS.md`**, not silent. It overlaps `whisper-GPU` by filename and concept
   (own `env.sh`, `snpe_gpu.py`, `metrics_gpu.py`, JAX flare + flare spec).
2. **The package ships functionality only — no analysis, no benchmarks, no run results.** This
   overrides the brief's Phase-2 bullet "Keep `benchmarks/`, `analysis/`, `sanity_check/`, `dev/`
   and their figures." Consequence: any doc text that links to those artefacts must be rewritten to
   cite the legacy repos, which stay live and citable. Directly excluded by file count:

   | Excluded | Files |
   |---|---|
   | `WHISPER_AI/analysis/` | 88 |
   | `WHISPER_AI/sanity_check/` | 63 |
   | `WHISPER_AI/dev/` | 30 |
   | `WHISPER_AI/jax_gpu/` | 14 |
   | `whisper-GPU/benchmarks/` | 22 |
   | **Total** | **217** |

   This also settles the TDE-figures question: `benchmarks/tde_vs_redback/figures/` is gitignored
   and exists untracked on this machine only — it is out of scope and stays out.

## 2. CPU inventory — `whisper_labia/` @ `8ba3843`

27 Python modules, 5,306 lines (`samplers/snpe.py` grew by 49 lines on this ref).

| Module | Lines | Public classes | Public functions |
|---|---|---|---|
| `__init__.py` | 80 | — | (re-exports only) |
| `distance.py` | 48 | — | `chi2_distance`, `register_distance`, `get_distance`, `list_distances` |
| `embeddings.py` | 96 | `MLPEmbedding`, `TCNEmbedding` | `build_embedding` |
| `likelihood.py` | 246 | `GaussianLikelihood`, `GaussianLikelihoodWithUpperLimits`, `MixtureGaussianLikelihood`, `GaussianLikelihoodWithScatter` | `register_likelihood`, `list_likelihoods`, `make_likelihood` |
| `metrics.py` | 344 | — | `waic`, `per_band_metrics`, `predictive_metrics` |
| `plotting.py` | 507 | — | `plot_light_curve`, `plot_ppc`, `plot_calibration`, `plot_corner` |
| `priors.py` | 94 | `Uniform`, `LogUniform`, `Prior` | — |
| `validation.py` | 150 | — | `recovery_metrics`, `posterior_predictive_check`, `sbc_rank`, `sbc_ranks` |
| `io/bands.py` | 246 | — | `resolve_band(s)`, `normalize_band(s)`, `group_bands`, `unmapped_bands` |
| `io/loader.py` | 228 | — | `load_lightcurve` |
| `io/photometry.py` | 53 | — | `mag_to_flux_density`, `flux_density_to_mag`, `mag_err_to_snr` |
| `io/schema.py` | 448 | `LightCurve` | — |
| `io/svo.py` | 275 | `SvoUnavailable` | 8 incl. `resolve_band_svo`, `find_filter_id`, `register_manual_band` |
| `io/units.py` | 171 | — | `as_unit`, `to_flux_density_jy`, `to_flux_cgs`, `check_magnitude_unit`, `to_canonical` |
| `models/__init__.py` | 73 | `Model` | `register_model`, `get_model`, `list_models` |
| `models/{bazin,flare,gaussian_rise}.py` | 37/39/41 | — | `bazin_flux`, `flare_flux`, `gaussian_rise_flux` |
| `models/mck19.py` | 191 | — | `mck19_flux` |
| `models/two_component_kilonova.py` | 166 | — | `two_component_kilonova_flux` |
| `samplers/__init__.py` | 52 | — | `register_sampler`, `get_sampler`, `list_samplers`, `fit` |
| `samplers/base.py` | 124 | `SamplerResult`, `BaseSampler` | `summarize_posterior`, `attach_band_metrics`, `attach_predictive_metrics` |
| `samplers/abc.py` | 235 | `ABCSampler` | `fit_ABC` |
| `samplers/abc_smc.py` | 355 | `ABCSMCSampler` | `fit_ABC_SMC` |
| `samplers/mcmc.py` | 212 | `MCMCSampler` | `fit_MCMC` |
| `samplers/snpe.py` | 737 | `SNPESampler` | `fit_SNPE` |

Registered at import: models `bazin`, `flare`, `gaussian_rise`, `mck19`, `two_component_kilonova`;
samplers `abc`, `abc_smc`, `mcmc`, `npe`, `snpe`.

## 3. GPU inventory — `whisper-GPU/` @ `10796a0` (repo root *is* the package)

18 Python modules, **7,954 lines** — up from 5,819 before the push.

| Module | Lines | Public classes | Public functions |
|---|---|---|---|
| `__init__.py` | 182 | — | `register` (+ `GPU_SAMPLERS`, `GPU_MODELS`) |
| `_env.py` | 158 | — | `env_script`, `env_report`, `require_jax`, `check_gpu`, `x64_enabled`, `n_jobs`, `gpu_list` |
| `distances.py` | 142 | — | `chi2_distance`, `mse_distance`, `rmse_distance`, `mae_distance`, `wmse_distance`, `wmae_distance`, `register_into_whisper`, `jnp_distance` |
| `metrics.py` | 74 | — | `ess_by_parameter`, `ess_summary` |
| `models/__init__.py` | **528** | — | `flare_model`, `kilonova_model`, `register_kilonova`, `tde_model`, `register_tde`, **`supernova_models`, `supernova_model`, `register_supernova`** |
| `models/_flare_spec.py` | 77 | — | `default_prior`, `flare_flux_numpy`, `get_model_numpy` |
| `models/flare.py` | 189 | — | `predict_jax`, `predict_numpy`, `make_log_prob_jax`, `predict_torch`, `prior_bounds_flat`, `get_model` |
| `models/kilonova.py` | 1179 | — | 14 incl. `bolometric`, `flux_density_mjy`, `ab_magnitude`, `make_filter_set` |
| `models/kilonova_two.py` | 301 | — | 11 incl. `two_component_magnitude`, `three_component_magnitude`, `villar_prior` |
| **`models/supernova.py`** | **1672** | — | **32**, incl. `arnett_bolometric`, `shock_cooling`, `basic_magnetar`, `csm_shock_breakout`, `sn_fallback_bolometric`, `general_magnetar_slsn_bolometric`, `synchrotron_f_nu`, `model_names`, `redback_prior` |
| `models/tde.py` | 1724 | — | 27 incl. `cooling_envelope`, `stream_stream_collision`, `tde_photosphere`, `diffusion`, `viscous`, **`redback_prior`, `fallback_prior`** |
| `samplers/abc_gpu.py` | 335 | `ABCGPUSampler` | `fit_ABC_GPU` |
| `samplers/abc_smc_gpu.py` | 352 | `ABCSMCGPUSampler` | `fit_ABCSMCGPU` |
| `samplers/emcee_jax.py` | 253 | `EmceeJAXSampler` | `fit_emcee_jax`, `fit_emcee_numpy` |
| `samplers/nuts_gpu.py` | 238 | `NUTSGPUSampler` | `fit_NUTSGPU` |
| `samplers/pymc_gpu.py` | 348 | `PyMCJAXVectorizedSampler`, `PyMCJAXParallelizedSampler` | `fit_PyMCJAXVectorized`, `fit_PyMCJAXParallelized` |
| `samplers/snpe_gpu.py` | 179 | `ContextEmbedding` | `encode_bands`, `fit_snpe_gpu` |

Registered into **whisper's** registries on `import whisper_gpu`: samplers `nuts_gpu`, `abc_gpu`,
`abc_smc_gpu`, `emcee_jax`, `pymc_jax_gpu_vectorized`, `pymc_jax_gpu_parallelized`; model
`flare_jax`; plus five numpy distances via `register_into_whisper`.
Deliberately **not** registered: `snpe_gpu` (`samplers/snpe_gpu.py:137`), and the photometric
kilonova / TDE / supernova models (`__init__.py:128`), bound instead through `register_kilonova` /
`register_tde` / `register_supernova`.

## 4. Symbol collisions — module-level public names defined in **both** packages

Still only two after the push:

| Symbol | CPU | GPU | Note |
|---|---|---|---|
| `chi2_distance` | `whisper_labia/distance.py` | `distances.py` | Same name, same role. Verify byte-identical before either becomes `distance/_numpy.py`; the GPU copy is what `register_into_whisper` pushes into whisper's registry today. |
| `get_model` | `whisper_labia/models/__init__.py` (registry accessor) | `models/flare.py` (returns the flare model) | **Same name, unrelated content.** Colliding these in one namespace would be a live bug. |

Note `redback_prior` is now defined in **both** `models/tde.py` and `models/supernova.py` *within*
the GPU package — an intra-package repeat, not a cross-package collision, but it needs one home or
two distinct names in the merged tree.

Filename-only collisions: `metrics.py` (disjoint contents), `models/flare.py` (different physics,
§8), `models/__init__.py`, `samplers/__init__.py`, `__init__.py`.

## 5. Import graph

**GPU → CPU, one direction, heavy.** `whisper_gpu` imports `whisper_labia` at module scope in
`__init__.py`, `distances.py`, `samplers/{abc_gpu,abc_smc_gpu,nuts_gpu,pymc_gpu}.py` and the tests:

- `samplers.base` → `BaseSampler`, `SamplerResult`, `summarize_posterior`, `attach_*` (all samplers)
- `models` → `Model`, `register_model`, `get_model`, `list_models`
- `priors` → `Prior`, `Uniform`, `LogUniform`; `io.schema` → `LightCurve`
- `samplers` → `register_sampler`, `get_sampler`, `list_samplers`
- `likelihood` → `GaussianLikelihood`, `make_likelihood` (`abc_gpu`, `abc_smc_gpu`)
- `samplers.abc_smc` → **private** `_importance_log_weights`, `_kernel_std` (`abc_smc_gpu.py:55`).
  A private cross-package import; becomes an intra-package relative import after the merge.

`whisper_gpu/__init__.py:38-43` guards this with a version-compat check naming the exact symbols and
raises a readable error with the found `whisper_labia.__file__` when the checkout is stale.

**CPU → GPU: none.** Strictly one-way at import time.

**Reverse-write path.** `distances.py:94-100` calls `register_distance(..., overwrite=True)` into
whisper's registry from `register_into_whisper()`, invoked by `whisper_gpu.register()` at import.
After the move these are core and register normally — the CPU ABC must keep them so a CPU/GPU
comparison at fixed distance stays possible.

## 6. Test inventory

**CPU @ `8ba3843` — 222 collected**, 221 passed / 1 skipped, verified in `phe_sbi`. 7 `@pytest.mark.slow`.
The skip is `test_snpe.py::test_snpe_gpu_recovers` ("no CUDA device available" — `phe_sbi` is CPU-only).
26 files; largest are `test_ingestion` 25, `test_svo` 15, `test_bands` 13, `test_units` 12,
`test_loader` 12, `test_snpe` 14, `test_schema` 11, `test_plotting` 11, `test_metrics` 10.
Optional-dep gates: `torch` (module-level, `test_embeddings`), `sbi` (module-level, `test_snpe`),
`redback` (function-level, `test_two_component_kilonova`, `test_benchmark_kilonova`).

**GPU @ `10796a0` — 144 test functions in 21 files**, organised into tiers:

| File | Test fns |
|---|---|
| `test_tde_vs_redback.py` | 33 |
| `test_supernova_vs_redback.py` | 31 |
| `t2_autodiff/test_gradients.py` | 14 |
| `test_import_and_registry.py` | 13 |
| `t0_parity/test_goldens.py` | 10 |
| `t0_parity/test_conventions.py` | 8 |
| `test_kilonova_two.py` | 7 |
| `t3_inference/test_continuity.py` | 6 |
| `test_abc_smc_gpu.py`, `test_flare_gradients.py`, `test_kilonova_vs_redback.py` | 4 each |
| `test_abc_gpu_scan.py`, `test_pymc_chain_method.py` | 3 each |
| `t3_inference/test_likelihood_parity.py`, `t3_inference/test_prior_predictive.py` | 2 each |

Non-test helper modules that ship inside `tests/`: `t0_parity/{_f32_probe,make_goldens}.py`,
`t3_inference/{_stage_common,_t3_common,stage_injection_recovery,stage_sbc}.py`. These are
infrastructure, not results — they belong with the tests under the functionality-only rule, but the
`t3_inference/sbc_results/` shard directories do **not**.

Every GPU test gates on `importorskip("jax")`; `test_{kilonova,tde,supernova}_vs_redback` also on
`redback`, `test_pymc_chain_method` on `numpyro`. All need `phe_sbi_2` with `env.sh` sourced.
`test_tde_vs_redback` needs float64 and selects a `REDBACK_PRESETS` preset by detecting which redback
is importable — 1.12.0 in `phe_sbi` vs 1.15.1 in `phe_sbi_2`, which disagree (`HANDOFF.md:259`).
**No `slow` marker exists in the GPU repo**; the CPU convention does not carry over. Needs a decision
at merge time.

## 7. Docs vs disk

- **CPU: complete.** Every public class/function on disk appears in `docs/API_REFERENCE.md` — 0 gaps.
  (`API_REFERENCE.md:7` still states 219 tests / 212 not-slow, now one ref stale.)
- **GPU: 69 public symbols had no mention** in `README.md`/`HANDOFF.md` at the pre-push ref, and the
  push added ~40 more public functions (`supernova.py` alone has 32). whisper-GPU has **no API
  reference**. Phase 4's "merge the two API references into one" is really "write the GPU half from
  scratch" — the single largest item in the plan.
- `HANDOFF.md` is now stale on repo state (says six commits unpushed at `45bfe0f`, "Nothing is
  committed" for the TDE). Its technical content still reads current.
- `HANDOFF.md` open items partly overtaken: #49 marked DONE; #44 (push) now done; the kilonova
  Arnett factor-of-2 has been acted on by commit `da2c00f` (`arnett_prefactor`). `OPEN_ITEMS.md`
  must be written against the *current* state, not HANDOFF's list verbatim.

## 8. Seed-list verification — two entries are wrong

| Seed claim | Verdict |
|---|---|
| Distances: CPU `chi2` only; GPU adds 5 in numpy+jnp, registers numpy into whisper on import | **Confirmed.** GPU also defines its own `chi2_distance` (collision, §4). |
| Metrics: disjoint contents, shared filename | **Confirmed.** CPU `waic`/`per_band_metrics`/`predictive_metrics`; GPU `ess_by_parameter`/`ess_summary`. |
| Flare: "same physics, two implementations" | **WRONG — different models.** CPU `flare_flux` = `A(1−e^{−t/t_rise})·e^{−t/t_decay}`, params `[amplitude, rise_time, decay_time]`. GPU `_flare_spec.flare_flux_numpy` = **Gaussian rise / exponential decay** in log params `[log_amp, log_sigma, log_tau, t0]`. The GPU "flare" is functionally the CPU's **`gaussian_rise`** (`[amplitude, t0, sigma_rise, tau_decay]`), log-parameterised — not the CPU's `flare`. Post-merge `flare` and `flare_jax` sit adjacent in one registry, shared name stem, different physics, disjoint parameter names. Flagging, not deciding. |
| ABC/ABC-SMC: intentional ports | **Confirmed.** `abc_gpu.py:1` says "a JAX port of `whisper_labia.samplers.abc`"; `abc_smc_gpu` reuses the CPU's private `_importance_log_weights`/`_kernel_std`. |
| SNPE: two GPU stories | **Confirmed.** CPU `SNPESampler` (737 lines, torch, own `device='cuda'`); GPU `snpe_gpu` unregistered, flare-only, comment at `samplers/snpe_gpu.py:137`. |
| MCMC/emcee | **Confirmed.** CPU `MCMCSampler` (emcee); GPU `EmceeJAXSampler` + `fit_emcee_jax`/`fit_emcee_numpy`. |
| Kilonova: two distinct models, do not merge | **Confirmed.** Disjoint priors documented at `models/kilonova_two.py`. |
| Priors: GPU "reads redback `.prior` files" | **Now correct — it was not before the push.** Commit `5ace064` added the loader; `models/tde.py` and `models/supernova.py` both now expose `redback_prior`. On the pre-push ref the priors were transcribed verbatim in code. |
| NUTS frontends | **Confirmed**, with a correction: `pymc_gpu.py` ships `_vectorized` **and** `_parallelized`, so it is three entry points over the NumPyro kernel, not two. |

## 9. Deferred to Phase 1+

- Byte-level `chi2_distance` CPU-vs-GPU comparison for `DUPLICATES.md`.
- Whether the GPU `redback_prior` in `tde.py` and `supernova.py` are the same function twice.
- A GPU-host run to establish the pre-merge GPU pass/skip baseline (replacing the stale 84/2).
