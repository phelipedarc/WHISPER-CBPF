# Installing WHISPER-CBPF

## 1. Quick start

```bash
pip install "git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

That gives you a working CPU package: data ingestion, plotting, priors, likelihoods, the model
comparison metrics, and the samplers `abc`, `abc_smc`, `mcmc` and `nested`.

Verify it:

```bash
python -c "import whisper_cbpf as wp; print(wp.__version__, len(wp.list_models()), len(wp.list_samplers()))"
```

Expect `0.1.0 6 13`. All 13 samplers are listed even on the base install — the ones needing an extra
say so when you ask for them, rather than vanishing from the registry.

> **PyPI:** nothing is published yet, so `pip install whisper-cbpf` does **not** work. Install from
> GitHub with the command above. (The GitHub route was verified on 2026-08-14: anonymous
> `git clone`, then `pip install .`, then import from the installed copy.)

## 2. Requirements

| | |
|---|---|
| Python | ≥ 3.10 |
| OS | Linux, macOS, Windows (GPU extra is Linux + CUDA 12) |
| Compiler | not needed for the base install; `[models]` builds `sncosmo` |

The base install is pure-Python wheels: numpy, scipy, pandas, matplotlib, astropy, emcee, dynesty,
corner.

## 3. The three extras

There are exactly three, and you can combine them.

| Extra | Command | Gives you |
|---|---|---|
| `[gpu]` | `pip install "whisper-cbpf[gpu] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"` | every JAX model (kilonova 1- and 2-component, TDE, 12 supernovae) and the samplers `nuts_gpu`, `abc_gpu`, `abc_smc_gpu`, `emcee_jax`, `pymc_jax_gpu_vectorized`, `pymc_jax_gpu_parallelized` |
| `[models]` | `…[models] @ git+…` | redback's physical models — `two_component_kilonova` on CPU, and the TDE / supernova priors read from redback's own `.prior` files |
| `[sbi]` | `…[sbi] @ git+…` | the `snpe` / `npe` simulation-based sampler (pulls torch) |

Everything at once:

```bash
pip install "whisper-cbpf[gpu,models,sbi] @ git+https://github.com/phelipedarc/WHISPER-CBPF.git"
```

`[gpu]` also carries `pymc`, `arviz` and `astroquery`, so the PyMC samplers, the ESS / PSIS-LOO
metrics and the SVO band lookup need no separate extra. On a **CPU-only** install those three are
not pulled in; add them directly if you want them:

```bash
pip install arviz astroquery      # PSIS-LOO metrics, SVO band resolution, on CPU
```

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

## 6. Where to go next

* [`README.md`](README.md) — a five-minute first fit.
* [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — the guided walkthrough.
* [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) — comparing models on CPU and on
  GPU, with a personalised prior.
* [`docs/CHOOSING.md`](docs/CHOOSING.md) — which sampler for which problem, and which combinations
  cannot work.
