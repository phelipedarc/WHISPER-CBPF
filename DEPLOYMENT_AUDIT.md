# DEPLOYMENT_AUDIT.md

A deployment-readiness audit of `whisper_cbpf`, against one question: **can a working astronomer,
given only this repository's documentation, fit a TDE, a supernova and one- and two-component
kilonovae to one light curve on either device in either space, compare them, and get a number they
can defend?**

Audited at `3f23e66`; fixes land in `9ee55b3`, `d19c3f3`, `15de940`. Every number below was measured
in `phe_sbi` (CPU, no JAX — the minimal install the package advertises) or `phe_sbi_2` (GPU, JAX
0.10.2, one RTX A6000 idle in both memory and utilisation). Where a claim could not be measured it
says so.

**Answer: yes for the kilonovae and the toys, on both devices and in both spaces. Yes for the TDE
and supernovae on GPU with float64. The remaining gaps are documented below, and the ones that are
structural are documented in the place a user will look.**

---

## 1. The matrix

Rows are model families; columns are (space × device). Each cell states the status **per sampler**.
`wp.list_samplers()` returns 11: `abc`, `abc_gpu`, `abc_smc`, `abc_smc_gpu`, `emcee_jax`, `mcmc`,
`npe`, `nuts_gpu`, `pymc_jax_gpu_parallelized`, `pymc_jax_gpu_vectorized`, `snpe` (`npe` is an alias
of `snpe`, so there are 10 distinct samplers). `snpe_gpu` exists but is deliberately unregistered.

**"CPU" splits in two, and the docs did not say so.** The minimal install (no JAX) **cannot register
any photometric JAX model at all** — `register_kilonova`, `register_kilonova_two`, `register_tde`
and `register_supernova` each raise `ImportError: ... requires the [gpu] extra`
(`backends/_registration.py:54-57`). So "kilonova on CPU" means either `two_component_kilonova`
(redback, `[models]`) or the JAX kilonova running on **CPU-JAX** with `[gpu]` installed. Both columns
below mean the latter unless the row says redback.

Legend: **W** works · **U** needs undocumented setup · **B** broken · **X** structurally impossible.
`(run)` = executed this session; `(code)` = determined from source with the file:line reason given.

### TDE — `register_tde`, both `rise=` values

| sampler | flux/CPU | flux/GPU | mag/CPU | mag/GPU |
|---|---|---|---|---|
| `abc`, `abc_smc`, `mcmc` | W (code) | W (code) | W (code) | W (code) |
| `snpe`/`npe` | W (code) | W (code) | W (code) | W (code) |
| `abc_gpu`, `abc_smc_gpu` | W (code) | W (code) | W (code) | W (code) |
| `nuts_gpu`, `pymc_jax_gpu_*`, `emcee_jax` | W (code) | W (code) | W (code) | W (code) |

**What was actually executed on this family, and what was not.** `rise="gaussian"` was registered in
float64, and its `predict` and the new `predict_jax` agree to **1.5e-16** relative on a 4-point
2-band evaluation — so the forward map the samplers consume is verified. The **full sampler sweep
was launched but had not completed when this report was written**; the `W (code)` cells are
therefore reasoned from the shared adapter path, which *is* executed end to end on the kilonova
below. `rise="none"` was registered only.

Treat this row as the weakest evidence in the matrix, and re-run
`scripts/compare_at2017gfo.py --device gpu --models tde --space both` to convert it.

The TDE is also the family where a **W** deserves the most caution for physical reasons, not
software ones — see the termination-cliff note below.

**Requires float64 in every cell.** `jax.config.update("jax_enable_x64", True)` before the first JAX
array; the guard raises at the first `predict` (`models/jax/tde.py:410`). Was **U** before this
audit for the five JAX samplers — they required a hand-built density or forward map.

**Caveat that is not a defect:** the termination cliff. Past the envelope's own last epoch the flux
is exactly zero → `mag_floor`, a ~20-mag step `jax.grad` cannot see. Measured over 10,000 draws of
redback's `gaussianrise` prior it lands inside a 0–100 d window on **15.6%** of draws
(`OPEN_ITEMS.md`). Gradient-based sampling on this model is not a safe default; ABC is safer.

### Supernovae — `register_supernova`, all 12

Identical to the TDE row, with the same evidence caveat. **All twelve** were registered (in float32
*and* float64 — see §2 for why that matters), and `arnett`'s `predict` agrees with its new
`predict_jax` to **1.9e-16**. The full sampler sweep was launched but incomplete at the time of
writing, so these are `W (code)` too. **Requires float64**, for a different reason from the TDE —
cgs luminosities of 1e43–1e46 erg/s against float32's 3.4e38 ceiling
(`models/jax/supernova.py:328`).

The eleven non-`arnett` engines were **not** individually fitted. Their `Model` construction, prior
loading from redback and `predict`/`predict_jax` wiring are shared code, so the risk is low, but it
is not zero and it is not measured.

`abc_gpu` / `abc_smc_gpu` are marked **W**, not **X**, but come with a cost note: batched simulation
in float64 pays 52–57× (`docs/GPU_SETUP.md`), so a large-N ABC on this family is runnable and rarely
the right call.

### Kilonova, 1-component — `register_kilonova`

| sampler | flux/CPU | flux/GPU | mag/CPU | mag/GPU |
|---|---|---|---|---|
| all 10 | W (run for 5 GPU + 3 CPU; code for snpe) | W (run) | W (run) | W (run) |

**Safe in float32**, unlike the TDE and supernovae. Measured through `wp.fit` with nothing
hand-built, x64 on, verified GPU: `nuts_gpu` AIC 1222722.6, `pymc_jax_gpu_vectorized` 1222724.0 —
the two NUTS frontends agreeing to 4 significant figures, which is the cross-check they exist to be.

### Kilonova, 2-component — `register_kilonova_two` (JAX) and `two_component_kilonova` (redback)

| sampler | flux/CPU | flux/GPU | mag/CPU | mag/GPU |
|---|---|---|---|---|
| `abc`, `abc_smc`, `mcmc` (redback model) | W (run) | W (run) | W (run) | W (run) |
| `abc`, `abc_smc`, `mcmc` (JAX model) | W (run) | W (run) | W (run) | W (run) |
| the five JAX samplers (JAX model) | W (run) | W (run) | W (run) | W (run) |
| the five JAX samplers (redback model) | **X** | **X** | **X** | **X** |

**X reason:** `two_component_kilonova` is numpy + redback + sncosmo, with in-place boolean-mask
assignment; it cannot be `vmap`ped and `jax.grad` cannot differentiate it. `models/two_component_kilonova.py:145`.

**The two kilonovae are two models, not one model twice** — different references, disjoint priors
(`DUPLICATES.md`). Comparing a fit of one against a fit of the other is not like-for-like.

### Toy models — `bazin`, `gaussian_rise`, `flare`, `mck19`, `flare_jax`

| sampler | flux/CPU | flux/GPU | mag/CPU | mag/GPU |
|---|---|---|---|---|
| `abc`, `abc_smc`, `mcmc`, `snpe` | W (run) | W (run) | W (run) | W (run) |
| `nuts_gpu`, `pymc_jax_gpu_*` | **X** | **X** | **X** | **X** |
| `abc_gpu`, `abc_smc_gpu`, `emcee_jax` | **X** | **X** | **X** | **X** |

**X reasons.** NUTS needs gradients and a numpy simulator has none. The JAX samplers need a batched
`predict_jax`, which the numpy toys do not carry. `flare_jax` is the exception — it has a JAX
implementation — but it is registered from `_flare_spec` **without** its `predict_jax` slot
(`backends/_registration.py:134-139`), so it too falls into **X** through `wp.fit`. That is a real
gap, listed under non-blocking below.

Two toy-model traps the README already warns about and that the matrix cannot: the analytic
amplitude prior is `Uniform(0, 10)`, which cannot reach a real transient's flux scale, and these are
shape fitters, not physics.

### Cells that are impossible for every model

| Combination | Why | Where |
|---|---|---|
| numpy model + `nuts_gpu` / `pymc_jax_gpu_*` | NUTS needs gradients; `jax.grad` cannot differentiate a numpy simulator | `docs/CHOOSING.md` |
| numpy model + `abc_gpu` / `abc_smc_gpu` / `emcee_jax` | they need a batched JAX forward map | `abc_gpu.py:152` |
| any non-flare model + `snpe_gpu` | hard-wired to `flare_jax.predict_torch`; raises `NotImplementedError` | `snpe_gpu.py:143` |
| TDE or any supernova in float32 | envelope ODE below float32 epsilon / cgs luminosities above its ceiling | `tde.py:410`, `supernova.py:328` |
| fitting a scatter parameter with distance-based ABC | extra simulated noise only raises the expected residual, so the posterior rails to the smallest allowed scatter | `abc_gpu.py:181` |
| a non-Uniform/LogUniform prior + any JAX sampler | refused rather than approximated by its bounds | `nuts_gpu.py:67`, `_adapters.py:_box_bounds` |

**Rows `docs/CHOOSING.md` was missing, now added:** the `emcee_jax` row, the float32+supernova row,
the non-Uniform-prior row, and the statement that **AIC/BIC/WAIC compare models, not spaces**.

---

## 2. Blocking defects — all fixed, each with a regression test

Ordered silent-wrong first, as instructed.

### 2.1 `abc_gpu`'s magnitude floor did not floor (silent-wrong) — FIXED `9ee55b3`

`samplers/jax/abc_gpu.py:96` clamped at a bare `1e-300`, which is **exactly `0.0` in float32**, so
`jnp.maximum` clamped nothing. Its own docstring (`:90-92`) claimed it "maps a non-positive model
flux to a finite ~759 mag … instead of a NaN that would silently poison the acceptance quantile".

**Measured:** `_model_in_space_jnp(jnp.zeros(3, jnp.float32), "magnitude", 3631.0)` → `[inf, inf,
inf]`. `abc_gpu` never enables x64, so float32 is its default case. `likelihood/_jax.py:62` and
`samplers/snpe.py:265` both already carried the `finfo.tiny` guard it lacked.

### 2.2 `MixtureGaussianLikelihood` scored WAIC under the wrong density (silent-wrong) — FIXED `9ee55b3`

`likelihood/_numpy.py:139-156` overrode `log_likelihood` but not `log_likelihood_pointwise`, and
both `metrics.waic` (`metrics/_numpy.py:81`) and `predictive_metrics` (`:255`) dispatch on the
latter. **Measured on 25 points:** total `29.1446`, pointwise sum `31.4662`. So
`wp.fit(..., likelihood="mixture")` returned an AIC from the mixture beside a WAIC from a plain
Gaussian, presented as comparable, with no warning.

### 2.3 The eleven flux→magnitude converters disagreed (silent-wrong) — FIXED `9ee55b3`

The AB zero point (3631 Jy) and the magnitude→flux direction agree everywhere. The flux→magnitude
direction had **four** distinct non-positive-flux behaviours:

| implementation | float64 floor | note |
|---|---|---|
| `io/photometry.py:43` | `NaN` + warning | the **data** converter — deliberate, now documented |
| `likelihood/_numpy.py:55`, `_jax.py:63`, `plotting.py:169`, `models/mck19.py:136` | 758.90 mag | |
| `samplers/snpe.py:265` | **769.80 mag** | bare `finfo.tiny`; agreed in f32, disagreed by ~11 mag in f64 |
| `samplers/jax/abc_gpu.py:96` | `inf` in f32 | §2.1 |
| `models/jax/{kilonova,tde,supernova,kilonova_two}.py` | `mag_floor` = 40 | a different layer; the likelihood floor never fires on these |

Now one rule everywhere on the model side: `max(_MIN_FLUX_JY, finfo.tiny)` → 758.90 in f64, 103.72
in f32. `flux_density_to_mag` keeps returning NaN and now says why it is the exception.

### 2.4 WAIC did not say when it was meaningless (silent-wrong) — FIXED `9ee55b3`

**Measured**, redback 2-component kilonova, 88 AT2017GFO points, 9 parameters:
`p_waic = 939505` beside an `AIC = 415`, returned with no warning. `p_waic` estimates the effective
number of parameters; 939505 against 88 data points is not a number. Cross-checked against a
well-specified toy at the same budgets, where AIC = −200.66 and WAIC = −201.64 with `p_waic = 2.75`
for 4 parameters — so the estimator is fine and the blow-up is real signal about the fit.

`_loo_waic` now returns `p_waic_reliable` against Gelman/Hwang/Vehtari's `p_waic ≤ n_data/2` and
warns when it fails. `scripts/compare_at2017gfo.py` marks those cells `!`.

### 2.5 A fit on an EMPTY light curve reported an AIC (silent-wrong) — FIXED `9ee55b3`

Found by the acceptance script, whose own bug it was. `select_time_window(0.4, 12)` applied *before*
`set_explosion_date` compares day bounds against raw MJD (~57983) and drops every row. Every sampler
then completed and reported `AIC = 2k`, `BIC = -inf` (`k·log(0)`). `GaussianLikelihood.__init__` now
raises and names that ordering.

### 2.6 The GPU samplers were unreachable through `wp.fit` — FIXED `d19c3f3`

`Model.predict_jax` / `log_prob_jax` existed and **no sampler read either slot**
(`models/__init__.py:26-29`). Only `register_kilonova` filled `predict_jax` (`_factories.py:170`);
`tde_model`, `supernova_model` and `kilonova_two_model` never *built* one.
`wp.fit(lc, m, sampler="nuts_gpu")` died with a bare CPython `TypeError`.

New `samplers/jax/_adapters.py`; resolution order explicit → `model.log_prob_jax` → auto-build, so
the auto-build fires only where the code used to raise.

### 2.7 `space=` was inert on the whole NUTS family — FIXED `d19c3f3`

`nuts_gpu`, `pymc_gpu` and `emcee_jax` accepted `space=`, defaulted it to `"flux"`, and only
recorded it (`nuts_gpu.py:115-116`). A magnitude light curve was therefore scored in **flux** by
`attach_band_metrics` and `attach_predictive_metrics`. Default is now `"auto"`;
`info["space_source"]` distinguishes `density` / `declared` / `assumed`.

### 2.8 `emcee_jax` sampled a flat box where its documented twin used the real prior (silent-wrong) — FIXED `d19c3f3`

`fit_emcee_jax` reads only `prior.distributions[nm].bounds` (`:111-112`), so its stationary
distribution is `L(θ)·1_box`. `nuts_gpu` adds the real prior through its sample sites. These agree
**only if every prior is Uniform** — and `kilonova_model` gives `temperature_floor` a
`LogUniform(100, 6000)` (`_factories.py:153`). `nuts_gpu._numpyro_priors`' own docstring (`:46-54`)
documents exactly this substitution as a bug it fixed, with the measurement: log-uniform puts half
its mass below 775 K, uniform below 3050 K. `emcee_jax.py:135-136` stated the Uniform-only
precondition as a fact.

**Consequence for existing results:** any AT2017GFO comparison that handed one `log_prob_fn` to both
arms with `temperature_floor` free was comparing two different posteriors. Not re-run here.

### 2.9 `attach_*_metrics` swallowed every failure (silent-wrong) — FIXED `9ee55b3`

Bare `except Exception: pass` (`base.py:39-40`, `:54-55`). A model whose `predict` raised produced a
fit with no WAIC, no message, and no way to distinguish that from a model for which WAIC is
undefined. This is how §2.6's band-alias problem was found.

### 2.10 `emcee_jax` downcast theta to float32 under x64 — FIXED `d19c3f3`

`emcee_jax.py:118` hard-cast to `jnp.float32` regardless of `jax_enable_x64` — the exact defect
`nuts_gpu._float_dtype` (`:74-81`) exists to prevent.

### 2.11 The unbounded-vmap trap, fifth instance — FIXED `d19c3f3`

`emcee_jax.py:115` vmapped over `nwalkers`, a user-scaled count, unchunked. Harmless while the only
density was the analytic flare; not once a photometric model can reach that path. Now chunked at 16.
`make_batched_predict_jax` also refuses an unchunked batch above 256 with a message naming `chunk=`.

### 2.12 `kilonova_two_model` was the only photometric factory not jitting its core — FIXED `d19c3f3`

**Measured**, 211 observations, `n_wave=2000`, A6000, best of five: **134.9 ms unjitted against
1.893 ms jitted, a factor 71.**

---

## 3. Non-blocking findings

### 3.1 Dead code — reported, NOT deleted

| Item | Evidence | Risk of removal |
|---|---|---|
| `pyproject.toml:52` `abc = ["pyabc"]` | `grep -rn pyabc` over the package, tests and docs returns only this line | none identified; it is an extra nobody can be using for anything |
| `pyproject.toml:42` `dynesty>=2.0` — a **core** dependency | imported nowhere; the docs describe Dynesty as *planned* (`__init__.py:5`, `docs/DESIGN.md:80`) | none, but it is a required install for every user |
| `register_kilonova_two` / `kilonova_two_model` | exported at `__init__.py:107,105`; zero callers, zero tests, zero docs before this audit | now reachable — the acceptance script uses it |
| 9 stale `.ipynb_checkpoints/*.py` inside `whisper_cbpf/` | verified **not** a packaging leak: no `__init__.py`, invalid dotted name, and `SOURCES.txt` contains zero checkpoint entries | none; they are gitignored and untracked |
| `tests/test_benchmark_kilonova.py` (3 tests) | excluded by `conftest.py:28`; loads `sanity_check/` which is not in this repo | the dependency stays visible, which is why it was kept |
| `tests/test_import_and_registry.py:181` | permanently skipped; asserts on a guard that cannot exist post-merge (`FINDINGS.md` §2.4) | none |
| `redback_prior` defined in both `tde.py:1549` and `supernova.py:1471` | first parameter named differently (`model_name` vs `model`); **still not diffed** — `DUPLICATES.md`'s one pending decision | unknown until diffed |
| `models/jax/flare.py:51` `predict_jax` | exists but takes a parameter **dict**, so it does not satisfy the `Model.predict_jax` contract, and `flare_jax` registers without it | this is why `flare_jax` is **X** for the JAX samplers |

### 3.2 Documentation drift

Fixed in `15de940`: the supernova-guard claim (four places), `API_REFERENCE`'s two contradictory
test counts, its duplicate `## 9` heading and ambiguous TOC anchor, its `list_distances()` (2 vs 14)
and `list_samplers()` (5 vs 11) claims, its `env_script()` description, `TUTORIAL`'s model/sampler
listings (3 vs 6, 4 vs 11) in the document the README calls "start here", `CHOOSING`'s "all five
metrics" over a list of six, and `CONTRIBUTING`'s `whisper_labia` title and coverage command.

**Not fixed — ~25 links into directories that have never existed in this repository.** `dev/`,
`sanity_check/`, `analysis/`, `benchmarks/` and `docs/figures/` were excluded by the merge's
scope rule (`FINDINGS.md` §2.7), and the migrated docs kept the links. Worst affected:
`docs/TUTORIAL.md` (12), `docs/API_REFERENCE.md` (11), `docs/PHYSICS_NOTES.md` (6),
`docs/REPORT_at2017gfo.md:20` (its only figure). `PROVENANCE.md:29,30,39,56` give four destination
paths that do not exist and are self-contradicted by `:79-81` of the same file. These need a
decision — re-point at the legacy repositories, or remove — which is §5.

**Numbers not re-measured.** Every timing and ratio in `docs/TUTORIAL.md` (10 of them),
`API_REFERENCE.md:255,358,414` and `docs/REPORT_at2017gfo.md`'s results table is carried from the
pre-merge repos and was not re-measured here. `TUTORIAL.md:415` says "~2000×" where
`API_REFERENCE.md:414` says "~10³×" for the same quantity. The `GPU_SETUP.md` float64 table and the
TDE-cliff percentages **were** measured, in the session before this one.

### 3.3 Duplicated logic

The flux↔magnitude map has **eleven** implementations (§2.3). They now agree, but nothing enforces
that: a twelfth would not be caught. `tests/test_audit_fixes.py::test_every_model_side_flux_to_mag_converter_agrees_on_zero_flux`
pins four of them.

`_make_band_indices` collapsed four identical copies of the band lookup (`_factories.py`).

### 3.4 Missing docstrings — 41 public symbols

Highest-impact, because they are top-level registry API and their `distance`/`model` counterparts
are documented: **`register_sampler`, `get_sampler`, `list_samplers`** (`samplers/__init__.py:26,32,38`)
and **`list_models`** (`models/__init__.py:88`). Then `SamplerResult.to_dict` / `to_json` /
`n_samples` and `BaseSampler.fit` (`base.py:85,88,106,123`), 12 in `priors/_numpy.py`, 10 in
`likelihood/_numpy.py` (4 fixed here), 8 `LightCurve` properties in `io/schema.py`.

`Prior.sample`'s missing docstring cost real time in this session: its signature is
`sample(self, rng=None)`, so `prior.sample(1)` silently binds `1` as the rng and fails downstream.

`CONTRIBUTING.md:47-48` states the convention is to "document every public symbol".

### 3.6 `FINDINGS.md` §1.1 misdiagnoses the one failing GPU test — measured

`tests/test_flare_gradients.py::test_naive_single_where_would_fail_here` is the control that proves
the flare's double-`where` guard is necessary: it asserts a *naive* `jnp.where` still overflows.

`FINDINGS.md` §1.1 says it fails because **"on JAX 0.10.2 the naive form no longer fails"** — a
JAX-version story. That is wrong. **The control is dtype-dependent, and therefore
test-order-dependent.** Measured directly, same machine, same JAX, same inputs:

| | gradient | control assertion |
|---|---|---|
| float32 (`jax_enable_x64` off) | `{'log_amp': nan, 'log_sigma': 0.0, 'log_tau': nan, 't0': nan}` | **passes** |
| float64 (`jax_enable_x64` on) | `{'log_amp': 0.0, 'log_sigma': 0.0, 'log_tau': -0.0, 't0': -0.0}` | **fails** |

The arithmetic: the test's unselected branch is `amp·exp(-dt/tau)` with `dt = -510`, `tau = e`, so
the exponent is **187.62**. float32 overflows above ~88.7 → `inf` → `0 · inf = nan` in the backward
pass → the assertion holds. float64 overflows only above ~709.8, so `exp(187.62) ≈ 6.7e81` is
finite, the cotangent is a clean `0`, and the assertion fails with its own message.

**`tests/t2_autodiff/test_gradients.py:86` calls `jax.config.update("jax_enable_x64", True)` at
module import, and that flag is process-global.** So this test passes when run alone and fails when
`t2_autodiff` is collected before it in the same pytest process — which is what the GPU regression
selection does. Confirmed: run in isolation on `phe_sbi_2` it is **1 passed**; in the combined
selection it is the sole failure.

Not caused by this audit — nothing here touches the flare model, that test, or x64 configuration.

**The fix is now obvious, which it was not under the version story:** pin the control's own dtype
(evaluate it in float32 explicitly), or choose parameters that overflow in float64 too — the
exponent must exceed ~709, e.g. a larger `|dt|` or a smaller `tau`. Either makes the control assert
something again regardless of collection order. **Not done here** — it is a decision about what the
control should assert, which is the reason `FINDINGS.md` left it alone.

### 3.5 CI

`.github/workflows/tests.yml:63` runs `pytest tests -q` with **no `-m "not slow"`**, contradicting
`CONTRIBUTING.md:25`. In practice the slow tests self-skip because `sbi`/`redback` are absent, so it
is latent. There is no GPU runner, so `t0_parity`, `t2_autodiff`, `t3_inference` and every
`*_vs_redback` comparison are ungated.

---

## 4. Verification

| Check | Before (`3f23e66`) | After |
|---|---|---|
| CPU, `phe_sbi`, `-m "not slow"` | 235 passed, 35 skipped, 6 deselected | **239 passed, 36 skipped, 6 deselected** (+4 new tests, +1 JAX-only module skip) |
| `tests/test_jax_backend.py`, `phe_sbi_2` | 19 passed | 19 passed |
| `tests/t0_parity` + `tests/t2_autodiff`, `phe_sbi_2` | 37 passed, 25m48s | **37 passed** (same selection, before the fixes) |
| GPU regression: `test_jax_backend` + `t0_parity` + `t2_autodiff` + `abc_gpu_scan` + `abc_smc_gpu` + `kilonova_two` + `pymc_chain_method` + `flare_gradients` | — | **89 passed, 1 failed, 27m35s** — the one failure is pre-existing and order-dependent, see §3.6 |
| `tests/t3_inference`, `phe_sbi_2` (~23 min) | not re-run at `3f23e66` either (`WHISPER_STATUS.md`) | **not run** |
| `tests/test_jax_adapters.py` (new) | — | 21 passed |
| Clean-checkout import (`git archive HEAD`) | ok | ok — 6 models, 11 samplers, 14 distances |
| `scripts/compare_at2017gfo.py --device cpu --quick` | did not exist | 16 works, 0 broken, 0 impossible |
| Five GPU samplers via `wp.fit`, nothing hand-built | all raised | all fit; the two NUTS frontends agree to 4 s.f. |

The documented CPU baseline reproduced **exactly**. An earlier note in this audit that it was
"arithmetically impossible" against `--collect-only`'s 260 was wrong and is withdrawn: ten test
modules `importorskip("jax")` at module scope, so they are never collected yet each reports one
skip. 260 + 10 = 270 = 235 + 35.

---

## 5. Not fixed — needs your decision

**Deletions (nothing was deleted; each needs permission):**

1. `pyproject.toml:52`'s `abc = ["pyabc"]` extra — `pyabc` is imported nowhere.
2. `pyproject.toml:42`'s core `dynesty>=2.0` — imported nowhere, and it is a required install for
   every user. Move to an extra, or drop until the Dynesty sampler exists.
3. The ~25 doc links into `dev/`, `sanity_check/`, `analysis/`, `benchmarks/`, `docs/figures/`:
   re-point at the legacy repositories (which remain live and citable) or remove.
4. `PROVENANCE.md:29,30,39,56` — four wrong destination paths, self-contradicted by `:79-81`.
5. The 9 stale `.ipynb_checkpoints/*.py` inside the package (harmless, verified not packaged).

**Behaviour changes I did not make:**

6. **`flare_jax` → `gaussian_rise_jax`.** You ruled rename-only, keep `flare`. Not executed yet —
   it touches ~41 references across 6 test files, 5 docs and 2 notebooks and belongs in its own
   commit. This audit's diff was kept to deployment blockers.
7. **`flare_jax` carries no `predict_jax` slot**, so it is the one JAX model unreachable by the JAX
   samplers through `wp.fit`. Fixing it means adapting `models/jax/flare.py:51` (which takes a
   parameter dict) to the flat-theta contract — a small change to a model's public shape.
8. **`Model.requires_x64`.** Would move the TDE/supernova float64 failure from mid-trace to `fit()`
   entry, naming the model. One dataclass field; declined as scope beyond the blockers.
9. **Shipping `scripts/whisper-gpu-env.sh` in the wheel.** `OPEN_ITEMS.md:138-155` lays out the two
   options and says not to solve it by keeping two copies. Still open; `INSTALL.md` now states the
   gap plainly.
10. **`snpe_gpu` generalisation.** Still flare-only and unregistered, by design.
11. **`abc_smc_gpu` does not set `result.samples_by_chain`** — the only GPU sampler that doesn't, so
    `metrics._jax.ess_by_parameter` has nothing to consume for it.
12. **`abc_smc_gpu`'s `max_attempts_per_round`** is still a CPU-era cap (`OPEN_ITEMS.md:75-80`).

**Measurements I could not run:**

13. Whether the **redback prior-file loaders** in `tde.py:1549` and `supernova.py:1471` are the same
    function twice. `DUPLICATES.md`'s one pending decision; needs the two diffed.
14. Whether **`jax_gpu/`** (excluded from the merge as a legacy precursor) holds anything unique.
15. Re-measurement of the ~15 carried-over timings in `TUTORIAL.md`, `API_REFERENCE.md` and
    `REPORT_at2017gfo.md`. Each needs its original script, and several of those scripts are in the
    excluded trees.
16. **Whether existing AT2017GFO results are affected by §2.8.** The emcee arm of any comparison that
    shared one `log_prob_fn` with NUTS while `temperature_floor` was free was sampling a different
    prior. Re-running that comparison would settle it; not done here.
