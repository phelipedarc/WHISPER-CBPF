# OPEN_ITEMS.md

Open work carried across from `whisper-GPU/HANDOFF.md` so the merge does not lose it, plus items
opened by the merge itself. Numbers in `#nn` are the legacy issue ids from HANDOFF.

Status as of the merge against `WHISPER_AI` @ `8ba3843` and `whisper-GPU` @ `10796a0`.

---

## Physics and inference

### #37 — redback's extinction wrapper is passed days where seconds are expected

**Extinction has been OFF in every fit so far because of this.** HANDOFF promotes it to a priority:
the AT2017GFO fits show every opacity railing to its prior floor and `mej_red` railing to its
ceiling, which is what an unmodelled reddening term looks like. Cheapest change that could move the
physics.

**Next:** fix the units at the call boundary, refit AT2017GFO, and see whether the railing relaxes.

### Disjoint opacity corridors vs an ordering constraint

The Villar priors use disjoint opacities — blue `κ U(0.1,1)`, purple `U(1,5)`, red `U(5,30)` — to
break label-switching between components. It works: chains land in the same mode rather than in
relabelled copies. But in the 3-component AT2017GFO fit **all three opacities rail on their corridor
floors**, so the components are held apart by the priors rather than by the data, which undercuts
the physical reading of the three-way split.

**Next:** test an ordering constraint (`kappa_blue < kappa_purple < kappa_red` over one shared wide
range). It preserves non-exchangeability without pinning where each boundary sits. HANDOFF lists
this at the top.

### The kilonova Arnett factor of 2

redback's diffusion kernel carries `1/td²` where Arnett / Chatzopoulos / Villar carry `2/td²`, so it
integrates to 1/2 rather than 1 and `L/L_in → 0.500055` at `t = 100 t_diff`. That is a uniform
**0.753 mag too faint, everywhere in parameter space** — not a corner case, so no prior protects it.
The JAX port reproduces redback faithfully, so correcting it breaks parity with every existing
kilonova fit.

**Partly addressed:** commit `da2c00f` in whisper-GPU added an `arnett_prefactor` flag exposing the
factor rather than changing it silently — the approach HANDOFF recommended.

**Next:** verify against a published AT2017GFO solution before changing any default.

### The 2-component AT2017GFO posterior is multimodal, not misconfigured

`max_rhat 2.41`, `ESS_worst 4.9`. Four chains in three distinct modes, confirmed by reshaping draws
chain-major vs step-major. **Do not respond by raising warmup** — NUTS is a local sampler and cannot
hop between separated modes at any budget; HANDOFF shows the warmup arithmetic confirming 6000 was
applied. `chi2/N 0.848` remains a fair statement about model quality, but **no credible interval
from that run is usable.**

**Next, in HANDOFF's order of expected value:** fix extinction and refit; use a multimodal sampler
(`abc_smc_gpu`, or dynesty on CPU); report per-mode rather than marginally; multi-start NUTS then
cluster.

### An unexplained better optimum

Random search found `χ²/N 1.07` while NUTS's posterior sat at a worse optimum, suggesting a local
mode on the κ–M_ej ridge that 6000 warmup steps does not escape.

**Next:** initialise a chain at the random-search optimum and see whether it stays.

---

## Samplers

### #39 — SNPE flows return draws outside prior support

Roughly **0.29% acceptance on LogUniform parameters**. Partly addressed on the CPU side by
`dc1e331` ("SNPE returned posterior draws from outside a LogUniform prior's bounds"); whether the
GPU path is affected was not checked during the merge.

### #48 — `abc_smc_gpu`'s `max_attempts_per_round` is a CPU-era cap

200k attempts was 46 minutes on CPU and 44 seconds on GPU, so the cap now **underfills rounds**
rather than protecting against a runaway.

**Next:** scale it to the backend, or express it as a time budget.

### `snpe_gpu` cannot be registered as-is

It takes a `model` argument but always passes `flare_jax.predict_torch`, so registering it would
return flare draws labelled with another model's parameter names. It raises `NotImplementedError` on
a non-flare model and stays reachable by direct import.

**Next:** generalise the `predict_torch` hook — that is what would make it registrable.

---

## Environment

### Two redbacks are installed and they disagree

`phe_sbi` has 1.12.0; `phe_sbi_2` has 1.15.1. Their `_cooling_envelope` loop bodies are identical,
but four things around them are not: grid size (5000 vs 500), an `f_debris` term, the termination
rule, and a `(1 + redshift)` flux factor. **A redback comparison run in the wrong container will
disagree for reasons that have nothing to do with this code.** The test suite detects which redback
is importable and picks the matching `REDBACK_PRESETS` entry.

### Do not fit at `REDBACK_PRESETS["1.15"]`

At `n_time=500` the AB magnitude spreads **0.55 mag** under a 1e-6 relative change in `mbh_6`
(8.3e-6 mag at 5000), photometry-interpolation error reaches 0.6 mag, termination time is 10% from
converged at the median, and 10% of the prior gives a one-point light curve. **Reproducible, not
fittable.**

### #44 — push whisper-GPU

**Done.** All commits landed; `origin/main` is `10796a0`.

---

## Opened by the merge

See `FINDINGS.md` for the full list. The ones that need a decision:

| Item | Where |
|---|---|
| Remove `flare`, rename `flare_jax` → `gaussian_rise_jax` — user has ruled, execution deferred | `DUPLICATES.md` |
| `redback_prior` defined in both `tde.py` and `supernova.py` — same function twice? | `DUPLICATES.md` §Priors |
| `jax_gpu/` excluded as a legacy precursor, not verified file-by-file against whisper-GPU | `FINDINGS.md` §2.6 |
| Migrated docs still link to `benchmarks/`, `analysis/`, `sanity_check/` — excluded from this package | `FINDINGS.md` §2.7 |
| A control test in `test_flare_gradients.py` no longer tests anything | **diagnosed 2026-08-13 — the cause in `FINDINGS.md` §1.1 is wrong; see below** |
| No `slow` marker in the GPU tests; the suite takes 47 minutes | `FINDINGS.md` §3.4 |
| The pre-merge GPU baseline is stale — no like-for-like regression check has been run | `FINDINGS.md` §3.5 |
| The merged API reference does not yet cover the ~110 GPU public symbols | — |

---

## Opened by the AT2017GFO GPU end-to-end run (2026-08-12)

All six GPU samplers and the three photometric registration factories were run for real against
`tests/data/at2017gfo.csv` at `4a5458a`; see `_scratch/gpu_e2e/REPORT.md` for the full run. Three
defects found there are fixed; these three are not.

### `scripts/whisper-gpu-env.sh` is not shipped in the wheel

`pyproject.toml`'s `packages.find` includes only `whisper_cbpf*`, and `package-data` only
`py.typed`, so a `pip install whisper-cbpf[gpu]` user gets **no `scripts/` directory at all** — and
the shell bootstrap is the one thing standing between them and the silent CPU fallback that runs
~50× slower with no error. `docs/GPU_SETUP.md` opens with `pip install` and then says
`source scripts/whisper-gpu-env.sh`, which that user cannot do.

`env_script()` now derives the real checkout path and `env_script_hint()` says plainly when the file
is absent instead of printing an unusable `source` line, so the failure is at least legible. That is
a mitigation, not the fix.

**Next:** decide where the single copy lives. Moving it to `whisper_cbpf/backends/env.sh` and adding
it to `package-data` ships it and makes the pre-merge `_env.py` path correct again, at the cost of
`source <site-packages>/...` being an odd thing to type; keeping it at `scripts/` and adding a
`whisper-cbpf-env` console script that prints the path is friendlier but is new surface. Do not
solve it by keeping two copies — see `DUPLICATES.md` for why this project treats that as the worse
outcome.

### `docs/GPU_SETUP.md`'s "interpreter gotcha" and `FINDINGS.md` §3.3 describe a failure that cannot happen

Both state that sourcing the env script does not put the venv interpreter on `PATH`, and that plain
`python` afterwards gives `ModuleNotFoundError: No module named 'jax'`. Measured in `phe_sbi_2` in a
`bash -lc` shell: before sourcing `python` → `/opt/conda/bin/python`; after sourcing → 
`/opt/jax_gpu_venv/bin/python`, and `import jax` returns 0.10.2. `scripts/whisper-gpu-env.sh:24-27`
exports `PATH` explicitly and has done so in its only commit (`0ab9d0b`), which predates
`FINDINGS.md` (`c176ed9`) — so the observation was already wrong when it was written.

Harmless in practice, because naming the interpreter explicitly always works and is what every run
command in this repo does. But it is a documented trap that does not exist, which costs a reader
trust in the traps that do.

**Next:** work out what the original observation actually was (a second `docker exec` losing the
sourced environment is the likely candidate, and *that* is a real trap worth documenting) and
rewrite both passages, or delete them.

### ~~The unbounded-vmap trap, fifth instance: the kilonova band integral~~ — PARTLY CLOSED 2026-08-13

`docs/GPU_SETUP.md` lists four instances. Here is a fifth, and unlike the others it is reachable
through a **public tuning knob**. A `jax.vmap` of `kilonova.ab_magnitude` whose batch axis is a
simulation count blows XLA's compile time up between extent 16 and extent 128 — measured at
`n_wave=1000` over 211 observations, first-call compile+run: batch 1 → 1.0 s, batch 16 → 2.6 s,
batch 128 → `slow_operation_alarm` on `input_reduce_fusion` and still compiling after 5 minutes.

`abc_gpu` / `abc_smc_gpu` expose exactly that extent as `chunk`. The default 16 is safely inside;
**raising `chunk` for throughput on a photometric model is the trap**, and nothing says so — the
parameter is documented as "particles per compiled call", which reads like a pure performance knob.

**Both ABC samplers' `chunk` docstrings now say it is bounded by XLA compile time, not by memory**,
and carry the measurement (`d19c3f3`). A SIXTH instance was found and fixed in the same commit:
`emcee_jax.py:115` vmapped over `nwalkers`, user-scaled and unchunked — harmless while the only
density there was the analytic flare, not once a photometric model can reach that path (it now can).
It is chunked at 16, and `make_batched_predict_jax` refuses an unchunked batch above 256 with a
message naming `chunk=`.

**Still next:** add both cases to `docs/GPU_SETUP.md`'s list, which still says "hit four times".

### ~~`kilonova_model`'s `predict` is not jitted, unlike the TDE's and the supernova's~~ — CLOSED 2026-08-12

**Fixed** while adding `Model.predict_jax` (see `CHANGELOG.md`, Unreleased). The factory now defines
one `_core(times, bidx, *free)` jitted at factory time, exactly as `tde_model` does, and both
`predict` and the new `predict_jax` go through it. The free/pinned `temperature_floor` switch is
resolved inside `_core` (pinned → closure constant, free → a traced argument), and the band-index
array stays a traced argument, so it compiles once per `n_obs`. Regression test:
`tests/test_jax_backend.py::test_predict_jax_compiles_once`. Original report below.

Measured on 211 points: the registered kilonova's `predict` takes **65.3 ms** per call (310 µs per
point), against **0.14 ms** for the identical computation inside a `jax.jit`. The factory calls
`kn.ab_magnitude` — the unjitted binding; `kn.ab_magnitude_jit` sits next to it — so every call
re-traces.

`tde_model` jits `_core` at factory time and documents this exact hazard ("831 ms against 9.1 ms
jitted, a factor 92"); `supernova_model` jits per grid shape and lands at 1.23 ms. The kilonova has
the same structure without the jit. It is invisible to the GPU samplers, which build their own
jitted density, and it is paid by every CPU-side consumer: `attach_predictive_metrics`' 200 draws
work out at ~13 s, consistent with the 23 s gap between `nuts_gpu`'s 33.7 s sampler time and its
56.5 s wall time.

**Next:** jit the kilonova factory's core the way `tde_model` does. It is not a one-liner — the
free/pinned `temperature_floor` switch and the band-index array both have to stay outside the traced
signature — which is why it is here rather than in the fixes.

---

## Opened by the float64 / TDE-cliff measurements (2026-08-12)

### The TDE termination cliff lands inside a real data window on 16% of prior draws

`tde.py`'s `_interp_photosphere` documents a genuine discontinuity: past the envelope's own last
valid epoch the flux is exactly zero → `mag_floor`, a ~20-mag step that `jax.grad` cannot see
because it only ever sees the smooth side. Until now nobody had measured **how often a fit actually
straddles it**, so it read as a theoretical caveat. It is not one.

10,000 draws, `z = 0.05`, both grids, seed 20260812
(`_scratch/f64_bench/tde_termination_prior.py`, git-ignored; JSON alongside it):

| prior | clock | cliff inside 0–100 d | 0–30 d | 0–200 d |
|---|---|---|---|---|
| `gaussianrise` (7 free, what `tde_model(rise="gaussian")` ships) | days since **disruption** (this model's own t=0) | **15.6%** | 0.4% | 40.1% |
| `gaussianrise` | days since **fallback** | **49.6%** | 27.6% | 58.7% |
| `cooling_envelope` (1 free — redback's file self-overrides 4 of 5) | days since fallback | **0.000%** | 0.000% | 0.000% |

Numbers are at `n_time=5000`; `n_time=500` moves them by <0.5 pp (27.6→27.0, 49.6→49.2), so this is
a property of the prior, not of the grid. Two more facts from the same sweep:

- **38.0% of `gaussianrise` draws have `tfb ≥ 100 d`**, so a 100-day campaign sits entirely inside
  the analytic Gaussian rise and never touches the envelope at all. Median `tfb` is 72.8 d.
- **Dead envelopes** (`constraint < 2`, zero flux at *every* epoch and therefore zero gradient in
  all parameters) are **4.96% at `n_time=500` and 0.02% at `n_time=5000`** over this prior. The
  docstring's "empty at n_time=5000 in 512 draws" is a small-sample artefact — it is 2 in 10,000,
  not 0. Its 6.45% / 3.1% figures are for other priors and were not reproduced here.
- The bare `cooling_envelope` prior terminates at a median of 5993 d (`n_time=5000`) / 1819 d
  (`n_time=500`). A factor 3.3 between grids, far more grid-sensitive than the 10.1% median quoted
  for the other case — but at either value the cliff is nowhere near a real campaign.

**Next:** the caveat in `_interp_photosphere` ("keep observations inside the envelope's own span, and
treat a fit that pushes the termination time through the data as unconverged") needs to become a
runtime diagnostic for `rise="gaussian"`, not a docstring — one draw in six is too many to leave to
the reader. `envelope_exists()` / `rise_peaks_near_fallback()` are the existing closed-form
predicates; neither bounds this.


---

## Opened / settled by the deployment-readiness audit (2026-08-13)

Full report: [`DEPLOYMENT_AUDIT.md`](DEPLOYMENT_AUDIT.md). Commits `9ee55b3`, `d19c3f3`, `15de940`.

### Settled

| Item | Settled by |
|---|---|
| `Model.predict_jax` / `log_prob_jax` exist but no sampler reads them | `d19c3f3` — `samplers/jax/_adapters.py`; all five JAX samplers now auto-build from the model when the caller passes nothing. `wp.fit(lc, "kilonova_one_jax", sampler="nuts_gpu")` works |
| `kilonova_two_model` is not jitted (the same defect closed for the 1-component model) | `d19c3f3` — measured 134.9 ms → 1.893 ms on 211 points at `n_wave=2000`, a factor 71 |
| The merged API reference does not cover the GPU public symbols | **partial** — `15de940` fixed the wrong claims (`list_samplers`, `list_distances`, `env_script`, the duplicate `## 9`, two contradictory test counts). The ~110 GPU symbols are still not individually documented |
| No `slow` marker in the GPU tests; the suite takes 47 minutes | **still open** |
| A control test in `test_flare_gradients.py` no longer tests anything | **still open** — untouched |
| Migrated docs link to `benchmarks/`, `analysis/`, `sanity_check/` | **still open** — ~25 links; needs a decision, see `DEPLOYMENT_AUDIT.md` §5 |
| `jax_gpu/` not verified file-by-file | **still open** |
| The pre-merge GPU baseline is stale | **still open** |
| Remove `flare`, rename `flare_jax` → `gaussian_rise_jax` | **user has re-ruled 2026-08-13: rename only, do NOT remove `flare`.** Not yet executed |
| `redback_prior` defined in both `tde.py` and `supernova.py` | **still open** — first parameter is named differently (`model_name` vs `model`); still not diffed |

### The supernova float64 guard does NOT fire at registration — this document was wrong too

`docs/GPU_SETUP.md`, `models/jax/__init__.py`, `_factories.py` and `../WHISPER_STATUS.md` all stated
that `register_supernova` raises during registration so the model never enters the registry, and
`WHISPER_STATUS.md` recorded it as a *defect found and fixed*.

**Measured, float32, `phe_sbi_2`: all 12/12 supernovae register successfully and all twelve appear
in `list_models()`.** The guard fires at the first `predict`, exactly like the TDE's.
`build_sn_grid` does carry the check, but it is only reached at factory time when `times=` is
passed, and the default is `times=None`. All four documents corrected in `15de940`.

### WAIC is returned on a scale that is not comparable with AIC/BIC, and now says so

Measured on the redback 2-component kilonova against 88 AT2017GFO points, 9 parameters:
**`p_waic = 939505`**, returned silently beside an `AIC = 415`. Cross-checked against a
well-specified toy at the same budget, where AIC = −200.66, WAIC = −201.64 and `p_waic = 2.75` for
4 parameters — so the estimator is correct and the blow-up is real signal about the fit.

`_loo_waic` now reports `p_waic_reliable` against Gelman/Hwang/Vehtari's `p_waic ≤ n_data/2` and
warns. **Next:** decide whether an unreliable WAIC should be omitted from `predictive_metrics`
entirely rather than returned with a flag — a number that is present is a number someone will quote.

### `emcee_jax` samples a flat box where `nuts_gpu` uses the real prior

`fit_emcee_jax` reads only `prior.distributions[nm].bounds`, so its stationary distribution is
`L(θ)·1_box`. The two agree **only if every prior is Uniform**, and `kilonova_model` gives
`temperature_floor` a `LogUniform(100, 6000)` — the exact substitution `nuts_gpu._numpyro_priors`
documents as a bug it fixed (median 775 K vs 3050 K). Fixed for the auto-built path in `d19c3f3`.

**Next:** any existing AT2017GFO comparison that handed one `log_prob_fn` to both arms with
`temperature_floor` free was comparing two different posteriors. Re-running it would settle how much
that moved the emcee arm. Not done.

### The eleven flux→magnitude converters agree now, but nothing enforces it

A twelfth would not be caught. `tests/test_audit_fixes.py` pins four of them.
**Next:** a single shared helper, or a test that enumerates them by import.

### `flare_jax` is the one JAX model the JAX samplers cannot reach

`models/jax/flare.py:51`'s `predict_jax` takes a parameter **dict**, not the flat theta
`Model.predict_jax` specifies, and `backends/_registration.py:134-139` registers `flare_jax` without
the slot. So it is structurally impossible through `wp.fit` for `abc_gpu`/`nuts_gpu`/etc., which is
odd for the one model those samplers were originally benchmarked on.

**Next:** adapt the flare's hook to the flat-theta contract. Small, but it changes a model's public
shape.


### `FINDINGS.md` §1.1 has the wrong cause for the failing flare control test

`FINDINGS.md` says `test_naive_single_where_would_fail_here` fails because "on JAX 0.10.2 the naive
form no longer fails". **Measured: it is float64, not the JAX version.** Same machine, same JAX,
same inputs:

* **float32** — gradient `{log_amp: nan, log_sigma: 0.0, log_tau: nan, t0: nan}` → the control
  **passes**.
* **float64** — gradient `{log_amp: 0.0, log_sigma: 0.0, log_tau: -0.0, t0: -0.0}` → the control
  **fails**.

The unselected branch is `amp·exp(-dt/tau)` with `dt = -510`, `tau = e`, i.e. exponent **187.62**.
float32 overflows above ~88.7 (→ `inf` → `0·inf = nan`, which is exactly what the control wants to
see); float64 overflows only above ~709.8, so the value is finite and the cotangent is a clean zero.

`tests/t2_autodiff/test_gradients.py:86` sets `jax_enable_x64=True` at module import and the flag is
**process-global**, so this test passes in isolation (verified: 1 passed) and fails whenever
`t2_autodiff` is collected first in the same pytest process (verified: the sole failure in an
89-passed GPU selection). It is order-dependent, not version-dependent.

**Next:** the fix follows from the diagnosis — either evaluate the control in float32 explicitly, or
pick parameters whose exponent exceeds ~709 so it overflows in float64 too. Either restores the
control regardless of collection order. Left undone deliberately: it is a judgement about what the
control should assert, which is why `FINDINGS.md` left it alone in the first place. `FINDINGS.md`
§1.1 itself still needs its explanation corrected.
