# DUPLICATES.md

Every place the two source repositories overlapped. **Nothing was resolved by deleting**, except the
one case marked below, which is recorded with the evidence that made it safe.

Source refs: `WHISPER_AI` @ `8ba3843`, `whisper-GPU` @ `10796a0`.

---

## flare vs flare_jax — same name stem, different models

**CPU:** `whisper_cbpf/models/flare.py::flare_flux` (39 lines)
**GPU:** `whisper_cbpf/models/jax/flare.py` + `_flare_spec.py::flare_flux_numpy` (189 + 77 lines)

**Relationship:** same name, unrelated content.

**Evidence:** CPU `flare_flux` computes `A(1 − e^{−t/t_rise})·e^{−t/t_decay}` over parameters
`[amplitude, rise_time, decay_time]`. `_flare_spec.flare_flux_numpy` computes a **Gaussian rise with
exponential decay** over `[log_amp, log_sigma, log_tau, t0]`. Functional forms and parameter names
are both disjoint. The GPU model is the counterpart of the CPU's **`gaussian_rise`**
(`[amplitude, t0, sigma_rise, tau_decay]`), log-parameterised — not of the CPU's `flare`.

**Recommendation:** rename `flare_jax` to `gaussian_rise_jax`, and separately remove `flare`.

**Risk if merged or renamed carelessly:** a user comparing "the flare model" CPU-vs-GPU would
compare two different models. The disjoint parameter names mean a mistake raises rather than
silently fitting the wrong thing, which is the only reason this is a documentation problem and not
a correctness one. Removing `flare` touches ~41 references across 6 test files, 5 docs and 2
notebooks.

**Decision:** ☑ **The user has ruled that `flare` should not exist and only `gaussian_rise` should.**
Deferred out of this migration deliberately — it is a behaviour change, and this task is
reorganisation only. Both models migrated unchanged. Execute as a separate, reviewable change.

---

## chi2_distance — defined in both

**CPU:** `whisper_labia/distance.py::chi2_distance` @ `8ba3843`
**GPU:** `distances.py::chi2_distance` @ `10796a0`

**Relationship:** identical intent; the GPU copy is a self-declared re-export.

**Evidence:** the whisper-GPU copy's own docstring reads *"identical to whisper's built-in,
re-exported for symmetry"* (`distances.py:50`). Both compute `sum(((obs − sim)/err)**2)` and return
a float; only the parameter names differ (`obs_flux, obs_flux_err, sim_flux` vs `obs, obs_err, sim`).

**Recommendation:** keep one. The brief's prescribed `distance/` layout specifies
"chi2 (from CPU) + mse rmse mae wmse wmae (from GPU)".

**Risk if merged:** none identified. Both are pure functions over the same formula.

**Decision:** ☑ **Collapsed — the only one in this migration.** `distance/_numpy.py` carries the
WHISPER_AI implementation. The whisper-GPU copy remains in this repository's history at commit
`3b12c1b`, and in the whisper-GPU repo, which is untouched. Flagged here rather than done silently.

---

## The two kilonovae — two models, not one model twice

**CPU:** `whisper_cbpf/models/two_component_kilonova.py` (166 lines, redback-backed)
**GPU:** `whisper_cbpf/models/jax/kilonova.py` + `kilonova_two.py` (1179 + 301 lines, standalone JAX)

**Relationship:** same physical phenomenon, deliberately divergent implementations.

**Evidence:** validated against different references and using **disjoint priors** — the CPU wrapper
uses redback's `kappa_1 U(0.1, 0.5)`; the JAX version carries redback's own `U(1, 30)` and a Villar
variant with disjoint opacity corridors (`kilonova_two.py:212, 255`). `HANDOFF.md` documents the
divergence as intentional.

**Recommendation:** keep both, permanently. A selector that *picks one and records which* is planned
separate work; it must never reconcile, average, or present them as interchangeable.

**Risk if merged:** every existing kilonova fit loses comparability, and the disjoint-prior trap
(documented in `OPEN_ITEMS.md`) becomes invisible.

**Decision:** ☑ keep both. Not negotiable — stated as a hard constraint of the migration.

---

## metrics.py — same filename, disjoint contents

**CPU:** `whisper_labia/metrics.py` — `waic`, `per_band_metrics`, `predictive_metrics` (344 lines)
**GPU:** `whisper-GPU/metrics.py` — `ess_by_parameter`, `ess_summary` (74 lines)

**Relationship:** same name, unrelated content. Not a numpy/JAX pair; neither is a port of the other.

**Evidence:** zero symbol overlap.

**Recommendation:** both survive in `metrics/_numpy.py` and `metrics/_jax.py`.

**Risk if merged:** nothing to merge.

**Decision:** ☑ both migrated unchanged, nothing reconciled. Note the `_jax` slot name is the
layout's naming for "the GPU-side half", not a dependency claim — that module imports numpy and
lazily arviz, and no JAX at all.

---

## ABC and ABC-SMC — intentional CPU/GPU ports

**CPU:** `samplers/abc.py::ABCSampler`, `samplers/abc_smc.py::ABCSMCSampler`
**GPU:** `samplers/jax/abc_gpu.py::ABCGPUSampler`, `abc_smc_gpu.py::ABCSMCGPUSampler`

**Relationship:** same intent, different implementation, by design.

**Evidence:** `abc_gpu.py:1` states it is "a JAX port of `whisper_labia.samplers.abc`";
`abc_smc_gpu` imports the CPU's `_importance_log_weights` and `_kernel_std` directly, so the
weighting scheme is literally shared rather than reimplemented.

**Recommendation:** keep both and document the pairing. The GPU crosses over at roughly 50k
simulations (`HANDOFF.md`); below that the CPU is faster.

**Decision:** ☑ keep both, paired in `docs/CHOOSING.md`.

---

## SNPE — two GPU stories for one method

**CPU:** `samplers/snpe.py::SNPESampler` (737 lines, torch, its own `device='cuda'` path)
**GPU:** `samplers/jax/snpe_gpu.py::fit_snpe_gpu` (179 lines, unregistered, flare-only)

**Relationship:** same method, two unrelated implementations.

**Evidence:** `snpe_gpu.py:137` explains that it always passes `flare_jax.predict_torch`, so
registering it would let `fit(lc, "kilonova_one_jax", sampler="snpe_gpu")` return **flare draws
labelled with kilonova parameter names** — a wrong answer with no error. It raises
`NotImplementedError` on a non-flare model.

**Recommendation:** `SNPESampler` is the general path and already runs on GPU via torch. Use
`snpe_gpu` only for the flare, by direct import. Generalising its `predict_torch` hook is what would
make it registrable.

**Decision:** ☑ both kept; `snpe_gpu` stays unregistered with its comment carried across verbatim.
This is the most confusing thing a new user meets — called out in `docs/CHOOSING.md`.

---

## MCMC — same frontend, different density

**CPU:** `samplers/mcmc.py::MCMCSampler` (emcee over Whisper's likelihood)
**GPU:** `samplers/jax/emcee_jax.py::EmceeJAXSampler` (emcee over a jitted JAX log-density)

**Relationship:** same sampler library, different log-probability implementation.

**Decision:** ☑ keep both, documented as a pair.

---

## NUTS — one kernel, three frontends

**GPU only:** `samplers/jax/nuts_gpu.py` (NumPyro directly), `pymc_gpu.py`
(`PyMCJAXVectorizedSampler`, `PyMCJAXParallelizedSampler`)

**Relationship:** the same NumPyro kernel behind three entry points, by design — one acts as a free
regression test on the others.

**Note:** the seed list called this two frontends; it is three. `pymc_gpu.py` ships both a
`_vectorized` and a `_parallelized` variant.

**Decision:** ☑ keep all three, documented as intentional in `docs/CHOOSING.md`.

---

## Priors — a loader, not a duplication

**CPU:** `priors.py` — `Uniform`, `LogUniform`, `Prior`
**GPU:** `models/jax/tde.py::redback_prior`, `models/jax/supernova.py::redback_prior`

**Relationship:** the GPU side reads redback's `.prior` files and maps them onto the CPU prior
classes. Not a duplication of the classes.

**Note:** this changed during the migration window. On the pre-push ref the priors were transcribed
verbatim into code with comments citing the source `.prior` file; commit `5ace064` replaced that
with a real loader.

**Open sub-item:** `redback_prior` is now defined in **both** `tde.py` and `supernova.py` — an
intra-package repeat, not a cross-package collision. Whether these are the same function twice was
not verified. Recorded in `FINDINGS.md`.

**Decision:** ☐ pending — needs the two `redback_prior` definitions diffed.
