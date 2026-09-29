# Installing WHISPER-CBPF

## 1. Quick start

```bash
pip install "git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

That gives you a working CPU package: data ingestion (survey alerts included), plotting, priors,
likelihoods, the samplers `abc`, `abc_smc`, `mcmc` and `nested`, and the comparison tools
(`compare`, `likelihood_max_opt`, `diagnostics`, `forecast`, the facts file and the HTML report)
for the CPU models. The supernova and TDE families that `compare` binds to an alert (`"arnett"`, `"magnetar"`,
`"tde"`, ...), and so the README quickstart, are JAX models. Add JAX for them:

```bash
pip install jax           # CPU: Linux, macOS, Windows (the platforms JAX publishes wheels for)
```

That is enough for the quickstart on a CPU (`compare` then uses CPU `mcmc`); it was checked with JAX
installed and numpyro, PyMC, redback, sbi and torch absent. On an NVIDIA GPU use the `[gpu]` extra
instead (section 3), which pins JAX's CUDA 12 build and adds the GPU samplers; it has no macOS or
Windows wheels.

Verify it:

```bash
python -c "import whisper_cbpf as wp; print(wp.__version__, len(wp.list_models()), len(wp.list_samplers()))"
```

Expect the version, then `6 14`. All 14 samplers are listed even on the base install — the ones
needing an extra say so when you ask for them, rather than vanishing from the registry.

> **PyPI:** nothing is published yet, so `pip install whisper-cbpf` does **not** work. Install from
> GitHub with the command above. (The GitHub route was verified on 2026-08-14: anonymous
> `git clone`, then `pip install .`, then import from the installed copy.)

## 2. Requirements

| | |
|---|---|
| Python | ≥ 3.10 |
| OS | Linux, macOS, Windows (the `[gpu]` extra is Linux + CUDA 12; on a CPU add `pip install jax`) |
| Compiler | not needed for the base install; `[models]` builds `sncosmo` |

The base install is pure-Python wheels: numpy, scipy, pandas, matplotlib, astropy, emcee, dynesty,
corner.

## 3. The four extras

There are four, and you can combine them.

| Extra | Command | Gives you |
|---|---|---|
| `[gpu]` | `pip install "whisper-cbpf[gpu] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"` | every JAX model (kilonova with one, two or three components, TDE, 12 supernovae), the samplers `nuts_gpu`, `abc_gpu`, `abc_smc_gpu`, `emcee_jax`, `pymc_jax_gpu_vectorized`, `pymc_jax_gpu_parallelized`, and `fit_batch`, `log_density` and `profile` for many alerts |
| `[models]` | `…[models] @ git+…` | redback's physical models — `two_component_kilonova` on CPU, and the TDE / supernova priors read from redback's own `.prior` files — plus `pyphot`, the SVO filter client |
| `[sbi]` | `…[sbi] @ git+…` | the `snpe` / `npe` simulation-based sampler (pulls torch); with `[gpu]` as well, `snpe_gpu` |
| `[analysis]` | `…[analysis] @ git+…` | `arviz` (ESS, PSIS-LOO) and `astroquery` (SVO wavelength search, and SVO filters when `pyphot` is absent), neither GPU-specific |

Everything at once:

```bash
pip install "whisper-cbpf[gpu,models,sbi,analysis] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

`[gpu]` installs `pymc`, which requires `arviz`, so a GPU install has the PyMC samplers and the ESS /
PSIS-LOO metrics without `[analysis]`. On a **CPU-only** install add `[analysis]` for them. Without
`arviz`, `elpd_loo` is `None`, not an error.

The SVO lookup (`io.svo`) uses `pyphot` when it is installed and falls back to `astroquery`; either
one is enough for filter curves by SVO ID, and a search by wavelength needs `astroquery`. With
neither, a band given by SVO ID cannot be resolved and you supply it by hand. The built-in LSST, ZTF
and SDSS bands need neither.

### redback pins numpy

redback pins `numpy==1.26` / `scipy<1.14`, so installing `[models]` into a numpy-2.x environment
will downgrade it. WHISPER itself works fine on numpy 2.x — only redback constrains you. Leave
`[models]` out if you do not need its models or the TDE / supernova priors.

## 4. GPU setup — the part that fails silently

**`[gpu]` alone is not enough.** JAX reads device visibility and memory settings **at import time**.
If `LD_LIBRARY_PATH` is wrong when Python starts, JAX finds no GPU, falls back to CPU, and runs
roughly **50× slower with no error, no warning, and correct results**. You will not notice until you
compare wall-clock with someone else.

That is why the bootstrap is a shell script — by the time Python is running, it is too late:

```bash
source "$(whisper-cbpf-env)"     # BEFORE starting Python
python your_fit.py
```

It sets `CUDA_DEVICE_ORDER`, `CUDA_VISIBLE_DEVICES`, `XLA_PYTHON_CLIENT_MEM_FRACTION` and
`LD_LIBRARY_PATH`, all overridable from your environment. The script ships inside the wheel at
`whisper_cbpf/backends/env.sh`, and `whisper-cbpf-env` prints its path, so the one command works
from a pip install and from a checkout alike.

On a shared machine, pick a card idle in **both** memory and utilisation:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

```bash
CUDA_VISIBLE_DEVICES=3 source "$(whisper-cbpf-env)"
```

Then confirm from Python — never assume:

```python
import whisper_cbpf as wp
ok, msg = wp.check_gpu()
print(ok, msg)
```

### float64 is mandatory for the TDE and every supernova

The TDE's envelope ODE accumulates increments ~1e-6 of the state, below float32's 1.2e-7 epsilon.
The twelve supernova engines carry bolometric luminosities of 1e43–1e46 erg/s, and float32 stops at
3.4e38. Both **raise** rather than returning a plausible wrong answer — but at the first `predict`,
**not** at registration, so a registry containing `tde_gaussianrise_jax` is not evidence that this
session can run it.

The flag is global and must be set before the first JAX array exists, so it can never be a model
default:

```python
import jax
jax.config.update("jax_enable_x64", True)     # FIRST, before anything creates an array
import whisper_cbpf as wp
```

Check the current state with `wp.x64_enabled()`. What it costs depends on batch size — about 4.3×
for `value_and_grad` at batch 1 (what NUTS pays), 52–57× from batch 100 up. Details in
[`docs/GPU_SETUP.md`](docs/GPU_SETUP.md).

## 5. Verify what you installed

Run the checks that match your install.

```bash
# base package
python -c "import whisper_cbpf as wp; print(wp.__version__, len(wp.list_models()), len(wp.list_samplers()))"
```

```bash
# [gpu] -- is JAX actually on a GPU, and is float64 on?
python -c "
import jax; jax.config.update('jax_enable_x64', True)
import whisper_cbpf as wp
print('devices:', jax.devices())
print('gpu ok :', wp.check_gpu())
print('x64    :', wp.x64_enabled())
"
```

```bash
# [models] -- redback reachable
python -c "import whisper_cbpf as wp; print(wp.get_model('two_component_kilonova').parameters)"
```

```bash
# [sbi] -- torch reachable
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

```bash
# [gpu], from a checkout -- the README quickstart, from an LSST alert to an HTML report
python -m pytest tests/test_docs_coverage.py -q
```

The last check runs the quickstart on the synthetic alert in `tests/data` with a short chain (about
two minutes on a CPU), and checks that every public name is documented.

## 6. Where to go next

* [`README.md`](README.md) — from an LSST alert to a ranked answer in six lines, and a first fit.
* [`docs/LSST_ALERTS.md`](docs/LSST_ALERTS.md) — the alert workflow end to end, with seconds per
  alert on a CPU and on one GPU.
* [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — the guided walkthrough.
* [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) — comparing models on CPU and on
  GPU, with a personalised prior.
* [`docs/CHOOSING.md`](docs/CHOOSING.md) — which sampler for which problem, and which combinations
  cannot work.
