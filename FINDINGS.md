# FINDINGS.md

Everything noticed during the merge and **not** acted on, plus every deviation from the migration
brief, with the reason. The migration was reorganisation only: no function body was changed, no
parameter renamed, no default altered, no documented bug fixed.

Source refs: `WHISPER_AI` @ `8ba3843`, `whisper-GPU` @ `10796a0`.

---

## 1. Defects that pre-date the merge

### 1.1 A control test has stopped testing anything

`tests/test_flare_gradients.py::test_naive_single_where_would_fail_here` asserts that a naive
`jnp.where` formulation still overflows, proving the guarded form is necessary. On JAX 0.10.2 the
naive form no longer fails, so the control asserts nothing and the test errors with its own message:
*"the naive form no longer fails -- this control has stopped testing anything"*.

**Verified pre-existing:** it fails identically when run against `whisper-GPU` @ `10796a0` in
`phe_sbi_2`. Not caused by the migration. Left alone — the fix is a judgement about what the control
should assert on a modern JAX, which is not a reorganisation decision.

### 1.2 The TDE evidence figures exist in no repository

`HANDOFF.md:296-306` cites `benchmarks/tde_vs_redback/figures/cooling_envelope_vs_redback.png` as
the evidence behind three documented deviations. `.gitignore` in whisper-GPU ignores `figures/`
globally and un-ignores only `benchmarks/two_component_at2017gfo/figures/`, so those three PNGs were
never committed. They exist untracked on the machine that generated them and nowhere else.

Out of scope here anyway (this package ships code, not analyses), but the *legacy* repo's own docs
reference files it does not contain.

### 1.3 `csm_diffusion`, `cocoon_photosphere` and `aspherical_diffusion` have no caller

Ported and tested against redback, but used by no TDE model (`HANDOFF.md`). Carried across unchanged.

---

## 2. Deviations from the brief, with reasons

### 2.1 No metrics registry was invented

The Phase-1 layout describes `metrics/__init__.py` as "registry + re-exports". **Neither source
package had a metrics registry** — metrics are called directly, never looked up by name. Building
one would be new behaviour, which the anti-goals forbid. `metrics/__init__.py` does re-exports only.

### 2.2 `is_gpu_available()` was not added

The layout names `backends/__init__.py` as exposing `is_gpu_available(), check_gpu()`. Only
`check_gpu()` exists in the sources. Adding `is_gpu_available()` would be a new public function.
`backends/` re-exports the existing helpers instead.

### 2.3 Registration is lazy — the registry stores factories, not classes

Hard constraint 4 requires `import whisper_cbpf` to work with no JAX. The legacy
`whisper_gpu/__init__.py` imported the six sampler classes eagerly, and each imports jax at module
scope. The registry therefore holds a zero-argument `_LazySampler` rather than the class object.

`get_sampler(name)` still does `_SAMPLERS[name]()` and still returns a sampler instance — the
documented contract is unchanged. What differs: **anything introspecting `_SAMPLERS["nuts_gpu"]` as
a class now sees a callable.** Explicitly approved during the migration.

### 2.4 The whisper compatibility guard was deleted

`whisper_gpu/__init__.py` carried `_REQUIRED_WHISPER_SYMBOLS` and `_check_whisper()`, which verified
that an *external* `whisper_labia` checkout was importable and new enough. The CPU core is now the
same package, so that skew is not expressible and the guard has nothing to check. Removed.
`tests/test_import_and_registry.py::test_version_guard_names_the_symbols_it_needs` asserts on it and
is marked skip with that reason — kept rather than deleted, so the decision stays visible.

### 2.5 `register_into_whisper()` was not carried across

whisper-GPU's `distances.py` pushed its five numpy metrics into whisper's registry from outside.
After the merge they are core and register normally in `distance/__init__.py`. The CPU ABC still
gains all five, which is what keeps a CPU/GPU comparison at a fixed distance possible.

### 2.6 `jax_gpu/` was excluded deliberately

`WHISPER_AI` `origin/main` carries a self-contained `jax_gpu/` package (14 tracked files: its own
`env.sh`, `snpe_gpu.py`, `metrics_gpu.py`, a JAX flare and a flare spec) that overlaps whisper-GPU by
filename and concept. The user identified it as the **legacy precursor** that became the whisper-GPU
repo. Excluded from WHISPER-CBPF. It remains in `WHISPER_AI`, which is untouched.

**Not verified file-by-file** against whisper-GPU. If anything in it is genuinely unique, it is not
in this package.

### 2.7 Analyses, benchmarks and run results were excluded

Per the user's scope rule — this package ships functionality, not the analyses we ran. 217 files
excluded: `analysis/` (88), `sanity_check/` (63), `dev/` (30), `jax_gpu/` (14),
`whisper-GPU/benchmarks/` (22), plus `tests/t3_inference/sbc_results/` (27),
`injection_results/` (11) and three tier `REPORT.md` files.

**Consequence:** the migrated docs inherited text that links to those artefacts. Those links now
point at nothing in this repo and should be re-pointed at the legacy repositories, which remain live
and citable. Not yet done.

### 2.8 Two tests are excluded from collection

`conftest.py` skips:

- `tests/test_benchmark_kilonova.py` (3 tests) — loads `sanity_check/benchmark_kilonova_modes.py`
  by path, and that tree is excluded by 2.7. **This is the only reduction against the pre-merge CPU
  count**: 222 → 219 contributed.
- `tests/t3_inference/` when JAX is absent — `_t3_common` does a bare `import jax` because it must
  set `jax_enable_x64` before the first array exists, so `importorskip` would defeat it and
  collection aborts the whole run on a CPU box.

### 2.9 Notebook stored outputs still contain the old package names

Four occurrences across `examples/at2017gfo_quickstart.ipynb` and `notebooks/01_lightcurves.ipynb`,
all inside **stored outputs**; zero in source cells. The brief requires not re-executing notebooks
or touching outputs. Rewriting them would falsify a record of what was actually run.

### 2.10 `require_jax()`'s install message was changed

It said `pip install 'jax[cuda12]' numpyro`, routing users around the packaged extra. It now says
`pip install 'whisper-cbpf[gpu]'`. The environment-bootstrap half of the message is unchanged and
still essential. This is a one-string change inside an otherwise byte-identical file.

---

## 3. Noticed, not investigated

### 3.1 `redback_prior` is defined twice inside the GPU package

Both `models/jax/tde.py` and `models/jax/supernova.py` define `redback_prior`. Whether these are the
same function twice or two different loaders was **not** verified. Tracked in `DUPLICATES.md`.

### 3.2 GPU samplers fail at different moments

`nuts_gpu` imports jax at module scope, so `get_sampler("nuts_gpu")` raises immediately without the
extra. `abc_gpu` defers jax to `require_jax()` inside `fit()`, so it constructs fine and fails later.
Both messages are actionable; the asymmetry is inherited from the sources, not introduced here.

### 3.3 The env script does not put its interpreter on `PATH`

`scripts/whisper-gpu-env.sh` reports `python=/opt/jax_gpu_venv/bin/python3` but sourcing it does not
make plain `python` resolve there in a `bash -lc` shell. Running `python` after sourcing gives
`ModuleNotFoundError: No module named 'jax'`. Documented in `docs/GPU_SETUP.md`; the script is
unchanged.

### 3.4 The GPU suite is slow

The selected GPU files take **47 minutes** in `phe_sbi_2` — the redback comparison tests each build
a redback light curve and a JAX one and diff them. `tests/t3_inference/` was not run at all and is
likely heavier. There is no `slow` marker in the GPU tests; the CPU convention did not carry over.
Worth adding, but that is restructuring tests.

### 3.5 The pre-merge GPU baseline in the brief is stale

The brief cites `84 passed, 2 skipped`. That predates both the TDE and the twelve supernova models;
the suite is now 144 test functions. The GPU tally here is the **first** measurement of the merged
suite, not a like-for-like comparison. A true regression check would run the same selection against
`whisper-GPU` @ `10796a0` and diff the tallies.

---

## 4. Recommendations for the legacy repositories

Not acted on — the source repos are frozen and were never written to.

- Both would benefit from a deprecation note pointing at WHISPER-CBPF.
- `whisper-GPU/HANDOFF.md` is stale on repo state: it says six commits are unpushed with
  `origin/main` at `45bfe0f` and that the TDE work is uncommitted. All of that has since landed.
- The `.gitignore` rule that excluded the TDE evidence figures (1.2) is worth revisiting there.
