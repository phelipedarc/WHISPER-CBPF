# Installing WHISPER-CBPF

Whisper installs in layers. The base package is a complete, working CPU toolkit; each extra adds a
capability you may not need. **Install the smallest thing that does your job** — the GPU stack alone
is ~3 GB of CUDA wheels.

If you only want the answer: for a full-capability install with redback physics and GPU sampling,

```bash
pip install "whisper-cbpf[gpu,models,pymc,sbi,svo,loo]"
source scripts/whisper-gpu-env.sh          # BEFORE starting Python — see §4
```

then verify with §5. Everything below explains the pieces.

> **Neither published install route works yet — install from a local checkout.**
>
> * **PyPI:** nothing published. `pip install whisper-cbpf` fails.
> * **GitHub:** `origin` is `https://github.com/phelipedarc/WHISPER-CBPF.git`. The repository exists
>   and is reachable, but it is **empty** — verified 2026-08-13 with `git ls-remote origin`, which
>   exits 0 and returns **zero refs**. So `pip install git+https://…` and `git clone https://…`
>   both give you nothing. Nothing has been pushed.
>
> The only route that works today is from the working tree:
> `pip install -e ".[gpu,models]"` at the repository root. Every `pip install whisper-cbpf …`
> command below is the shape the released package will take, not something you can run yet.

## 1. Requirements

| | |
|---|---|
| Python | **3.10, 3.11 or 3.12** (`requires-python = ">=3.10"`) |
| OS | Linux is what this is developed and tested on. macOS works for the CPU half; `jax[cuda12]` has no macOS build |
| GPU (optional) | NVIDIA with **CUDA 12**. Developed on RTX A6000 |
| Compiler | none needed — every dependency ships wheels |

Core dependencies, installed automatically: `numpy>=1.23`, `scipy>=1.9`, `pandas>=1.5`,
`matplotlib>=3.5`, `astropy>=5.0`, `emcee>=3.1`, `dynesty>=2.0`, `corner>=2.2`, `tqdm`.

## 2. The base install

```bash
pip install whisper-cbpf
```

That is a complete package: ingestion, plotting, priors, likelihoods, and the `abc`, `abc_smc` and
`mcmc` samplers. It imports with **no JAX, no CUDA, no torch and no redback**, and every GPU model
and sampler still appears in `list_models()` / `list_samplers()` / `list_distances()` — you can see
what exists before deciding to install it.

What you can do with only this: fit the analytic models (`gaussian_rise`, `bazin`, `flare`,
`flare_jax`) and `mck19`, in flux or apparent-magnitude space, and compare them by AIC/BIC/WAIC.

What you cannot: any photometric JAX model (kilonova, TDE, supernova) — those need `[gpu]` even to
*register*, because their forward model is written in JAX. `register_kilonova` and friends raise
`ImportError: ... requires the [gpu] extra` on a base install.

## 3. The extras

| Extra | Command | What it adds | Needed for |
|---|---|---|---|
| `[gpu]` | `pip install "whisper-cbpf[gpu]"` | `jax[cuda12]>=0.4.30`, `numpyro>=0.15`, `arviz>=0.15` | **every** JAX model (kilonova / TDE / 12 supernovae) and the samplers `abc_gpu`, `abc_smc_gpu`, `nuts_gpu`, `emcee_jax` |
| `[models]` | `pip install "whisper-cbpf[models]"` | `redback>=1.12` | `two_component_kilonova`; and the TDE/supernova **priors**, which are read from redback's own `.prior` files at runtime |
| `[pymc]` | `pip install "whisper-cbpf[gpu,pymc]"` | `pymc>=5.16` | `pymc_jax_gpu_vectorized`, `pymc_jax_gpu_parallelized`. They run on JAX, so `[gpu]` is required too |
| `[sbi]` | `pip install "whisper-cbpf[sbi]"` | `sbi`, `torch` | `snpe` / `npe` (and `snpe_gpu`) |
| `[svo]` | `pip install "whisper-cbpf[svo]"` | `astroquery>=0.4.6` | resolving band names not in `FILTER_LOOKUP` via the SVO filter service |
| `[loo]` | `pip install "whisper-cbpf[loo]"` | `arviz>=0.15` | PSIS-LOO / ELPD in `predictive_metrics` (WAIC works without it) |
| `[dev]` | `pip install -e ".[dev]"` | `pytest`, `pytest-cov`, `build`, `astroquery` | running the test suite |

**The combination for the full §1 acceptance matrix** — all three model families, both spaces, both
devices, every sampler:

```bash
pip install "whisper-cbpf[gpu,models,pymc,sbi]"
```

### A note on redback and numpy

redback pins `numpy==1.26` / `scipy<1.14`. Installing `[models]` into an environment with numpy 2.x
will downgrade it. Whisper itself works on numpy 2.x; only redback constrains you. If you do not
need redback's models or the TDE/supernova priors, leave `[models]` out.

## 4. GPU setup — the part that fails silently

**`pip install "whisper-cbpf[gpu]"` is not sufficient on its own.** JAX reads device visibility and
memory settings **at import time**. If `LD_LIBRARY_PATH` is wrong when Python starts, JAX finds no
GPU, falls back to CPU, and runs roughly **50× slower with no error, no warning, and correct
results**. You will not notice until you compare wall-clock with someone else.

This is why the bootstrap is a shell script and not a Python call — by the time Python is running it
is too late:

```bash
source scripts/whisper-gpu-env.sh     # BEFORE starting Python
python your_fit.py
```

The script sets `CUDA_DEVICE_ORDER`, `CUDA_VISIBLE_DEVICES`, `XLA_PYTHON_CLIENT_MEM_FRACTION` and
`LD_LIBRARY_PATH`, all overridable from your environment. On a shared machine, pick a card that is
idle in **both** memory and utilisation:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=3 source scripts/whisper-gpu-env.sh
```

> **Known gap:** `scripts/` is not shipped inside the wheel (`pyproject.toml`'s `packages.find`
> takes only `whisper_cbpf*`), so a `pip install whisper-cbpf[gpu]` user has no `scripts/`
> directory. Until that is resolved, get the script from a checkout of this repository.
> `whisper_cbpf.env_script()` prints where it would live and `env_script_hint()` says plainly when
> it is absent. Tracked in `OPEN_ITEMS.md`.

### float64: mandatory for the TDE and every supernova

The TDE's envelope ODE accumulates increments ~1e-6 of the state, below float32's 1.2e-7 epsilon.
The twelve supernova engines carry bolometric luminosities of 1e43–1e46 erg/s and float32 stops at
3.4e38. Both raise rather than returning a plausible wrong answer — **at the first `predict`, not at
registration**, so a registry containing `tde_gaussianrise_jax` is not evidence that this session can
run it.

The flag is global and must be set **before the first JAX array exists**, so it can never be a model
default:

```python
import jax
jax.config.update("jax_enable_x64", True)     # FIRST, before anything creates an array
import whisper_cbpf as wp
```

What it costs depends entirely on batch size — ~4.3× for `value_and_grad` at batch 1 (what NUTS
pays) and 52–57× from batch 100 up. The table is in [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md). Check
the current state with `wp.x64_enabled()`.

## 5. Verify the install

Each step checks one layer. Run the ones that match what you installed.

```bash
# 1. base package
python -c "import whisper_cbpf as wp; print(wp.__version__, len(wp.list_models()), len(wp.list_samplers()))"
#    -> 0.1.0 6 11

# 2. [gpu] — is JAX actually on a GPU?
source scripts/whisper-gpu-env.sh
python -c "import whisper_cbpf as wp; ok, msg = wp.check_gpu(); print(ok or msg)"
#    -> True   (anything else prints an actionable diagnostic; do NOT ignore it)

# 3. [models] — redback importable, and which version
python -c "import importlib.metadata as m; print('redback', m.version('redback'))"

# 4. everything, end to end: the acceptance test
python scripts/compare_at2017gfo.py --device cpu --space both --quick
python scripts/compare_at2017gfo.py --device gpu --space both --quick   # needs [gpu] + step 2
```

`scripts/compare_at2017gfo.py` fits a TDE, a supernova and one- and two-component kilonovae to
`tests/data/at2017gfo.csv` with every sampler that can legitimately run each combination, and prints
one AIC/BIC/WAIC table. Combinations that cannot work print the reason instead of being skipped, so
the output is a complete statement about your install.

```bash
# 5. the test suite ([dev])
python -m pytest -q -m "not slow"
```

On a base + `[dev]` install this reports **253 passed, 36 skipped, 6 deselected** (measured 2026-08-13; it was 239 before the `nested` sampler added 14 tests). The skips are the
JAX/redback/sbi-gated tests. Note that `--collect-only` reports 260 rather than 270 there: ten test
modules `importorskip("jax")` at module scope, so they are never *collected* yet each still reports
one skip.

## 6. Which redback do you have?

`redback` 1.12.0 and 1.15.1 have identical `_cooling_envelope` loop bodies but differ in grid size
(5000 vs 500), an `f_debris` term, the termination rule and a `(1 + redshift)` flux factor. **A
comparison run against the wrong version will disagree for reasons that have nothing to do with this
code.** The test suite detects which is importable and selects the matching `REDBACK_PRESETS` entry;
your own scripts should record the version. Do not fit at `REDBACK_PRESETS["1.15"]` — see
[`docs/GPU_SETUP.md`](docs/GPU_SETUP.md).

## 7. Development install

```bash
# NOTE: cloning gives you an empty directory today - the remote has zero refs (see the box at the
# top). Work from an existing checkout until the first push.
cd WHISPER-CBPF
pip install -e ".[dev]"
python -m pytest -q -m "not slow"
```

CI runs the **base + `[dev]`** install on Python 3.10/3.11/3.12 and asserts that none of `jax`,
`torch`, `redback`, `numpyro`, `pymc` or `sbi` is importable — that is the configuration most likely
to break by accident, because nobody develops in it. The GPU tiers (`tests/t0_parity`,
`tests/t2_autodiff`, `tests/t3_inference`) have no CI runner and stay a manual run.

## 8. Where to go next

| | |
|---|---|
| [`README.md`](README.md) | the 30-second tour |
| [`docs/TUTORIAL.md`](docs/TUTORIAL.md) | start here |
| [`docs/CHOOSING.md`](docs/CHOOSING.md) | which model and sampler, and which combinations are impossible |
| [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) | the silent-CPU-fallback trap, float64, multi-GPU |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | the full public surface |
