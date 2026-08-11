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
| A control test in `test_flare_gradients.py` no longer tests anything on JAX 0.10.2 | `FINDINGS.md` §1.1 |
| No `slow` marker in the GPU tests; the suite takes 47 minutes | `FINDINGS.md` §3.4 |
| The pre-merge GPU baseline is stale — no like-for-like regression check has been run | `FINDINGS.md` §3.5 |
| The merged API reference does not yet cover the ~110 GPU public symbols | — |
