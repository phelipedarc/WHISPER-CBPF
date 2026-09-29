# GPU setup

Read this before your first GPU fit. Most of it is about one failure mode that produces **no error
message at all**.

## Install

```bash
pip install "whisper-cbpf[gpu]"          # jax[cuda12], numpyro, pymc (which brings arviz)
# pymc ships inside [gpu] -- the two pymc_jax_gpu_* samplers need no extra install
```

CUDA 12 is pinned deliberately. Installing plain `jax` instead gives you the CPU build, and see
below for why that is the expensive mistake.

## The trap: JAX falls back to CPU silently

**JAX reads device visibility and memory fraction at import time.** If `LD_LIBRARY_PATH` is wrong
when Python starts, JAX finds no GPU, falls back to CPU, and runs **roughly 50× slower with no
error, no warning and correct results**. You will not notice until you compare wall-clock against
someone else's run.

This is why the environment bootstrap is a shell script and not a Python call. It cannot be replaced
by import-time logic — by the time Python is running, it is too late.

```bash
source "$(whisper-cbpf-env)"     # BEFORE starting Python
python your_fit.py
```

Then confirm, every time, before paying for a long fit:

```python
import whisper_cbpf as wp
wp.check_gpu()          # raises with an actionable message if the GPU is not visible
```

### The interpreter gotcha

Sourcing the script prints the interpreter it will use, for example
`python=/path/to/venv/bin/python3`, and looks for the CUDA libraries in that interpreter's own
site-packages. It puts a virtualenv first on your `PATH` **only when you name it**; otherwise it
uses whatever `python3` is already there. In a container where JAX lives in a virtualenv that is
not on your `PATH`, running `python` after sourcing gives:

```
ModuleNotFoundError: No module named 'jax'
```

Name the virtualenv before sourcing (the discontinued name `WHISPER_GPU_VENV` still works):

```bash
export WHISPER_CBPF_VENV=/path/to/venv
source "$(whisper-cbpf-env)"
python -m pytest tests/t0_parity -q
```

## float64 and the TDE — and every supernova

**The TDE model cannot run in float32.** Its envelope ODE accumulates Euler increments around 1e-6
of the state, below float32's 1.2e-7 epsilon; in f32 the curve terminates after one step and the
luminosity comes out `inf`. It raises rather than returning a plausible wrong answer.

**Neither can any of the twelve supernova models**, for an unrelated reason: their engines carry
bolometric luminosities in cgs, 1e43 to 1e46 erg/s, and float32 stops at 3.4e38. Every engine in
the family returns `inf` — including at the faintest corner of redback's prior — and the guards
would turn that into `mag_floor` in every band, so this one raises too.

Only the kilonovae and the flare are safe in float32. That includes a raw-MJD light curve with
`t_exp_days=` on the same clock: since whisper 0.1.1 the factories subtract it on the host in
float64 before any cast, and the samplers' adapters hand them the float64 epochs. Up to 0.1.0 it
was subtracted in float32, which resolves 0.0039 d at MJD 58000: 7.3 mmag and a log-likelihood off
by 5.2 on the two-component kilonova, where days since t0 gave 5e-6 mag and 0.01.

```python
import jax
jax.config.update("jax_enable_x64", True)     # BEFORE the first JAX array exists
import whisper_cbpf as wp
wp.register_tde(["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)
wp.register_supernova("arnett", ["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)
```

**Both guards fire at the first `predict`, not at registration.** `register_tde` and
`register_supernova` both **succeed** in float32 and the model **enters the registry**; the guard is
inside the engine and raises when it is first traced. A registry that contains
`tde_gaussianrise_jax` or `arnett_jax` is therefore not evidence that the session can run it.

> Registration itself succeeds: in float32, all **12/12** supernovae register and all twelve appear
> in `list_models()`. `build_sn_grid` does carry the float64 check, but it is only reached at factory
> time when `times=` is passed, and the default is `times=None`. So the guard fires at the first
> `predict`, not at registration.

Two things make this awkward, and neither is hidden from you:

1. **The flag is global and session-wide**, so a session that fits a TDE or a supernova pays for
   float64 on the kilonova too — and what that costs depends almost entirely on the batch size:

   | `kilonova.ab_magnitude` | float32 | float64 | ratio |
   |---|---|---|---|
   | forward, batch 1 | 62–83 µs | 157 µs | **1.9–2.5×** |
   | `value_and_grad`, batch 1 (one NUTS leapfrog step) | 102 µs | 437 µs | **4.3×** |
   | forward, `vmap` batch 10 | 75 µs | 992 µs | **13×** |
   | forward, `vmap` batch 100 | 158 µs | 8.3 ms | **52×** |
   | forward, `vmap` batch 1000 | 1.45 ms | 81.8 ms | **57×** |
   | forward, `vmap` batch 10000 | 14.4 ms | 823 ms | **57×** |

   One RTX A6000, `n_wave=2000` (the `kilonova_model` default up to whisper 0.1.0; since 0.1.1 the
   default band integral is 16 Gauss nodes per band, docs/PHOTOMETRY.md), 100 epochs, 4 bands, jitted, best
   of three timed loops; both dtypes on identical inputs, agreeing to 6e-8 relative. **Batch 1 is
   not a measurement of arithmetic**: the launch floor on this machine is ~45 µs, so the float32
   kernel is only ~12 µs of that 62 and the ratio there moves ±30% between runs. Everything from
   batch 10 up is stable to <1%. The plateau at ~57× is the hardware: the A6000's FP64 rate is
   1/32 of FP32, and float64 `exp`/`log`/`pow` cost more than that ratio again.

   So: **NUTS on one light curve pays ~4×, batched simulation pays ~57×.** float64 is affordable
   in the first case and is the reason SBI/ABC simulation should not run in it.
2. **It must be set before the first array is created**, so it cannot be a model default. Nothing in
   this package sets it for you — a silent default here would quietly change the cost of every other
   model in the session. Importing whisper first is safe: since 0.1.1 no model module builds a JAX
   array at import (the kilonova's quadrature nodes and the supernova's series constants did, and
   froze in float32 when imported before the flag, 1e-8–2e-6 relative in float64 light curves).

Check the current state with `wp.x64_enabled()`.

## Multi-GPU: ask for less than you think

The device budget lives in `whisper_cbpf.backends`: `MAX_GPUS = 2`, `CPU_FRACTION = 0.60`
(58 of 96 cores). `gpu_list()` only offers a device that is idle in **both** memory (< 512 MiB) and
utilisation (< 10%) — a device at 100% utilisation with 5 MiB allocated will happily accept work and
then contend for SMs.

### `parallel` with more chains than devices is worse than not asking

NumPyro contains:

```python
if chain_method == "parallel" and local_device_count() < self.num_chains:
    chain_method = "sequential"
```

So `parallel` gives every chain its own device or **gives up entirely** — there is no second `pmap`
pass. Asking for 4 chains on 2 GPUs runs fully sequentially on one device while the second GPU holds
an idle allocation. Measured on the 2-component kilonova, 1000 warmup + 1000 draws/chain:

| Arm | asked | actually ran | GPUs | chains | wall | s/chain |
|---|---|---|---|---|---|---|
| A `parallel`, 4 chains | parallel | **sequential** | 1 | 4 | 209.7 s | 52.4 |
| B `parallel`, 2 chains | parallel | parallel | 2 | 2 | 81.8 s | 40.9 |
| C `vectorized`, 4 chains | vectorized | vectorized | 1 | 4 | 199.8 s | 49.9 |

Three conclusions:

1. **Arm A is strictly worse than arm C** — 209.7 s against 199.8 s on the *same single GPU*, and it
   holds a second GPU idle.
2. **vmap over chains buys almost nothing here** — 1.05× over running them one at a time. One chain
   of this model already saturates the GPU, so chain-level batching has no spare capacity to
   exploit. The vectorized sampler is convenience on this problem, not speed.
3. **True pmap is fastest per chain** (40.9 s vs 49.9 vmapped vs 52.4 sequential).

`fit` deliberately does **not** auto-reduce `num_chains` — dropping 4 chains to 2 halves your r-hat
evidence, and that is your call, not the library's. `info["chain_method"]` records what *actually*
ran; `info["chain_method_requested"]` keeps what you asked for. Check the former before labelling a
run multi-GPU.

**Default for a 2-GPU budget:** `pymc_jax_gpu_vectorized` with 4 chains — it leaves a GPU free at
about a 1.22× wall-time cost. Use two pinned 2-chain processes only when a single fit's wall time is
the bottleneck.

> An earlier "2.26× on 4 GPUs" figure was 4 chains on 4 devices — the one regime where `parallel`
> parallelises. It does not transfer to 2 GPUs.

## The unbounded-vmap trap

Hit four times in this project. `jax.vmap` whose batch axis is a **draw or simulation count** builds
one fused kernel sized by a user-tunable number. A best-draw scan over 6400 draws × 2 components ×
284 observations × 256 quadrature nodes is 7.4 GB in a single kernel, and XLA fails during
autotuning — *after* sampling has already succeeded.

The fix each time was `jax.lax.map` over a scalar scorer: constant memory, linear cost, one compile.
**Treat "vmap whose extent is a draw or simulation count" as a bug on sight.**

## Two redbacks, and they disagree

If you compare against redback, check which one you have. `redback` 1.12.0 and 1.15.1 have identical
`_cooling_envelope` loop bodies but differ in four things around them:

| | 1.12.0 | 1.15.1 |
|---|---|---|
| grid | `logspace(..., 5000)` | `logspace(..., 500)` |
| debris | — | `f_debris` / `calculate_f_debris` |
| termination | `min(c1, c2)` | `if c1==0: c1=5000`; `max(min(c1,c2), 4)` |
| flux | no redshift factor | `× (1 + redshift)` |

**A comparison run against the wrong version will disagree for reasons that have nothing to do with
this code.** redback 1.20 keeps 1.15.1's TDE column. The JAX TDE's default `n_time` follows the
installed redback (`tde.default_n_time()`: 500 for 1.15 and 1.20, and without redback; 5000 for
1.12), and the tests select the matching `REDBACK_PRESETS` entry the same way.

**The 500-point grid is redback's, and it is coarse.** Under a 1e-6 relative change in `mbh_6` the AB
magnitude at a fixed epoch moves by ≤ 8.8e-6 mag through 75% of the envelope's life, 7.9e-3 mag at
90%, and by the whole ~24 mag `mag_floor` step at 95-99%, where the termination index moves by grid
steps (at 5000: ≤ 4.9e-5 mag through 95%, 0.31 mag at 99%); redback 1.20 steps at the same epochs.
Photometry-interpolation error reaches 0.6 mag, and 5-10% of the prior yields a one-point envelope
(the Gaussian rise of `rise="gaussian"` survives it; `PORTING_NOTES.md` §4.3). Pass `n_time=5000`
when you fit near the envelope's end and parity with redback is not the point.

## Performance reference

TDE, `n_time=5000`, on an A6000: **11.5 ms** for one light curve (the scan is launch-bound, hence
`unroll=16`), **9.0 µs/curve at batch 512**. Prefer vmapped consumers.

One trap worth knowing: a model that closes over parameter values in a `functools.partial` bakes
them into the scan's jaxpr, so calling it outside `jit` is a cache miss — a fresh ~0.6 s XLA compile
*per likelihood evaluation* (831 ms against 9.1 ms jitted). The factories jit once. Any new model
built the same way inherits the same trap.
