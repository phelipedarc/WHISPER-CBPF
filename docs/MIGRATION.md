# Migrating from `whisper_labia` / `whisper_gpu`

WHISPER-CBPF replaces two packages with one. **There is no compatibility shim and no alias** —
`import whisper_labia` and `import whisper_gpu` do not work. Rewrite the import.

```python
import whisper_cbpf as wp        # was: import whisper_labia as wp
```

Legacy sources: `WHISPER_AI` @ `8ba3843` (`whisper_labia/`) and `whisper-GPU` @ `10796a0`
(`whisper_gpu/`). Both repositories are discontinued and archived.

---

## The one-line rule

| Old | New |
|---|---|
| `whisper_labia.<anything>` | `whisper_cbpf.<anything>` — unchanged below the top level |
| `whisper_gpu` (top level) | `whisper_cbpf` |
| `whisper_gpu.models` | `whisper_cbpf.models.jax` |
| `whisper_gpu.samplers` | `whisper_cbpf.samplers.jax` |
| `whisper_gpu._env` | `whisper_cbpf.backends` (or `.backends._env`) |
| `whisper_gpu.distances` | `whisper_cbpf.distance` |
| `whisper_gpu.metrics` | `whisper_cbpf.metrics` |

Everything from the CPU package keeps its module path. Only the GPU half moved, and it moved so each
component sits beside its CPU sibling.

---

## CPU package — paths unchanged

```python
from whisper_cbpf.io.schema      import LightCurve          # was whisper_labia.io.schema
from whisper_cbpf.io.loader      import load_lightcurve
from whisper_cbpf.priors         import Prior, Uniform, LogUniform
from whisper_cbpf.likelihood     import GaussianLikelihood, make_likelihood
from whisper_cbpf.models         import register_model, get_model, list_models
from whisper_cbpf.samplers       import fit, get_sampler, list_samplers
from whisper_cbpf.samplers.base  import SamplerResult, BaseSampler
```

Two modules became subpackages. **The public names did not change**, so imports through the package
root are unaffected:

| Old module | New location | Import that still works |
|---|---|---|
| `whisper_labia.distance` | `whisper_cbpf/distance/` (`_numpy.py`, `_jax.py`) | `from whisper_cbpf.distance import chi2_distance, register_distance, get_distance, list_distances` |
| `whisper_labia.metrics` | `whisper_cbpf/metrics/` (`_numpy.py`, `_jax.py`) | `from whisper_cbpf.metrics import waic, per_band_metrics, predictive_metrics` |

---

## GPU package — every public symbol

### Samplers

| Old | New |
|---|---|
| `whisper_gpu.samplers.nuts_gpu.NUTSGPUSampler` | `whisper_cbpf.samplers.jax.NUTSGPUSampler` |
| `whisper_gpu.samplers.nuts_gpu.fit_NUTSGPU` | `whisper_cbpf.samplers.jax.fit_NUTSGPU` |
| `whisper_gpu.samplers.abc_gpu.ABCGPUSampler` | `whisper_cbpf.samplers.jax.ABCGPUSampler` |
| `whisper_gpu.samplers.abc_gpu.fit_ABC_GPU` | `whisper_cbpf.samplers.jax.fit_ABC_GPU` |
| `whisper_gpu.samplers.abc_smc_gpu.ABCSMCGPUSampler` | `whisper_cbpf.samplers.jax.ABCSMCGPUSampler` |
| `whisper_gpu.samplers.abc_smc_gpu.fit_ABCSMCGPU` | `whisper_cbpf.samplers.jax.fit_ABCSMCGPU` |
| `whisper_gpu.samplers.emcee_jax.EmceeJAXSampler` | `whisper_cbpf.samplers.jax.EmceeJAXSampler` |
| `whisper_gpu.samplers.emcee_jax.fit_emcee_jax` | `whisper_cbpf.samplers.jax.fit_emcee_jax` |
| `whisper_gpu.samplers.emcee_jax.fit_emcee_numpy` | `whisper_cbpf.samplers.jax.fit_emcee_numpy` |
| `whisper_gpu.samplers.pymc_gpu.PyMCJAXVectorizedSampler` | `whisper_cbpf.samplers.jax.PyMCJAXVectorizedSampler` |
| `whisper_gpu.samplers.pymc_gpu.PyMCJAXParallelizedSampler` | `whisper_cbpf.samplers.jax.PyMCJAXParallelizedSampler` |
| `whisper_gpu.samplers.snpe_gpu.fit_snpe_gpu` | `whisper_cbpf.samplers.jax.fit_snpe_gpu` |
| `whisper_gpu.samplers.snpe_gpu.ContextEmbedding` | `whisper_cbpf.samplers.jax.ContextEmbedding` |

Registered sampler **names are unchanged** — `nuts_gpu`, `abc_gpu`, `abc_smc_gpu`, `emcee_jax`,
`pymc_jax_gpu_vectorized`, `pymc_jax_gpu_parallelized` all still work through
`wp.fit(lc, model, sampler=...)`. `snpe_gpu`, unregistered before, is now registered too.

### Models and factories

| Old | New |
|---|---|
| `whisper_gpu.register_kilonova` | `whisper_cbpf.register_kilonova` |
| `whisper_gpu.register_tde` | `whisper_cbpf.register_tde` |
| `whisper_gpu.register_supernova` | `whisper_cbpf.register_supernova` |
| `whisper_gpu.kilonova_model` | `whisper_cbpf.kilonova_model` |
| `whisper_gpu.tde_model` | `whisper_cbpf.tde_model` |
| `whisper_gpu.supernova_model` / `supernova_models` | `whisper_cbpf.supernova_model` / `supernova_models` |
| `whisper_gpu.flare_model` | `whisper_cbpf.flare_model` |
| `whisper_gpu.models.kilonova` | `whisper_cbpf.models.jax.kilonova` |
| `whisper_gpu.models.kilonova_two` | `whisper_cbpf.models.jax.kilonova_two` |
| `whisper_gpu.models.tde` | `whisper_cbpf.models.jax.tde` |
| `whisper_gpu.models.supernova` | `whisper_cbpf.models.jax.supernova` |
| `whisper_gpu.models.flare` | `whisper_cbpf.models.jax.flare` |

### Environment and distances

| Old | New |
|---|---|
| `whisper_gpu._env.check_gpu` | `whisper_cbpf.check_gpu` |
| `whisper_gpu._env.require_jax` | `whisper_cbpf.require_jax` |
| `whisper_gpu._env.x64_enabled` | `whisper_cbpf.x64_enabled` |
| `whisper_gpu._env.gpu_list` / `n_jobs` | `whisper_cbpf.gpu_list` / `n_jobs` |
| `whisper_gpu._env.env_script` / `env_report` | `whisper_cbpf.env_script` / `env_report` |
| `whisper_gpu._env.MAX_GPUS` / `CPU_FRACTION` | `whisper_cbpf.backends.MAX_GPUS` / `CPU_FRACTION` |
| `whisper_gpu.distances.mse_distance` (etc.) | `whisper_cbpf.distance.mse_distance` |
| `whisper_gpu.distances.jnp_distance` | `whisper_cbpf.distance._jax.jnp_distance`, or `get_distance("mse_jax")` |
| `whisper_gpu.metrics.ess_summary` / `ess_by_parameter` | `whisper_cbpf.metrics.ess_summary` / `ess_by_parameter` |
| `whisper_gpu/env.sh` | `whisper_cbpf/backends/env.sh`; `source "$(whisper-cbpf-env)"` |
| `WHISPER_GPU_VENV` (read by env.sh) | `WHISPER_CBPF_VENV`, unset by default; the old name is still honoured |

---

## Behaviour changes you may notice

The merge itself (0.1.0) changed only three; apart from them, code that worked under the old packages
worked under 0.1.0 with only its import lines changed. 0.1.1 changes much more — plots return their
Axes, bare `u g r i z y` mean LSST, the NUTS/PyMC/emcee default starts, block widths and `snpe_gpu`
rounds, redback's constraint priors enforced by default, and others. Read
[`CHANGELOG.md`](../CHANGELOG.md) before upgrading past 0.1.0.

### 1. Importing no longer requires JAX

`import whisper_gpu` required jax to be installed. `import whisper_cbpf` does not — it works with no
jax, no CUDA, no torch and no redback. GPU entries still appear in `list_samplers()`,
`list_models()` and `list_distances()`, so you can discover them before installing the extra.
Asking for one without `[gpu]` raises a message naming the extra.

### 2. The sampler registry stores factories, not classes

`get_sampler(name)` still does `_SAMPLERS[name]()` and still returns a sampler instance. But
`_SAMPLERS["nuts_gpu"]` is now a zero-argument callable rather than the class object, so code that
introspected it *as a class* needs adjusting. Nothing in the public API is affected.

### 3. The whisper version guard is gone

`whisper_gpu._REQUIRED_WHISPER_SYMBOLS` and its `_check_whisper()` verified that an **external**
`whisper_labia` checkout was importable and new enough. The CPU core is now the same package, so
that mismatch cannot occur and the guard has been removed.

---

## Distances gained names

whisper-GPU used to push `mse`, `rmse`, `mae`, `wmse` and `wmae` into whisper's registry from
outside, on import. They are now core, registered normally, and **the CPU ABC keeps them** — which
is what makes a CPU/GPU comparison at a fixed distance possible.

The JAX versions are reached by a `_jax` suffix, matching the sampler convention:

```python
from whisper_cbpf.distance import get_distance, list_distances
get_distance("mse")        # numpy — the default
get_distance("mse_jax")    # jnp — needs [gpu]
list_distances()           # lists both; *_jax entries are available-requiring-the-extra
```

---

## Two things that did not change and matter

- **`flare` and `flare_jax` are still different models.** The GPU one is a Gaussian rise with
  exponential decay in log parameters — the counterpart of `gaussian_rise`, not of `flare`. Their
  parameter names are disjoint.
- **The TDE still needs `jax_enable_x64` before the first JAX array exists**, and that flag is
  global and session-wide. See [`GPU_SETUP.md`](GPU_SETUP.md).
