# GPU setup

Read this before your first GPU fit. Most of it is about one failure mode that produces **no error
message at all**.

## Install

```bash
pip install "whisper-cbpf[gpu]"          # jax[cuda12], numpyro, arviz
pip install "whisper-cbpf[pymc]"         # only for the two pymc_jax_gpu_* samplers
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
source scripts/whisper-gpu-env.sh     # BEFORE starting Python
python your_fit.py
```

Then confirm, every time, before paying for a long fit:

```python
import whisper_cbpf as wp
wp.check_gpu()          # raises with an actionable message if the GPU is not visible
```

### The interpreter gotcha

Sourcing the script prints the interpreter it expects, for example
`python=/opt/jax_gpu_venv/bin/python3` — but it **does not put that interpreter on your `PATH`**. In
a container where JAX lives in a virtualenv, running plain `python` after sourcing gives:

```
ModuleNotFoundError: No module named 'jax'
```

Use the interpreter the banner names, explicitly:

```bash
source scripts/whisper-gpu-env.sh
/opt/jax_gpu_venv/bin/python3 -m pytest tests/t0_parity -q
```

## float64 and the TDE — and every supernova

**The TDE model cannot run in float32.** Its envelope ODE accumulates Euler increments around 1e-6
of the state, below float32's 1.2e-7 epsilon; in f32 the curve terminates after one step and the
luminosity comes out `inf`. It raises rather than returning a plausible wrong answer.

**Neither can any of the twelve supernova models**, for an unrelated reason: their engines carry
bolometric luminosities in cgs, 1e43 to 1e46 erg/s, and float32 stops at 3.4e38. Every engine in
the family returns `inf` — including at the faintest corner of redback's prior — and the guards
would turn that into `mag_floor` in every band, so this one raises too.

Only the kilonovae and the flare are safe in float32.

```python
import jax
jax.config.update("jax_enable_x64", True)     # BEFORE the first JAX array exists
import whisper_cbpf as wp
wp.register_tde(["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)
wp.register_supernova("arnett", ["ztfg", "ztfr"], redshift=0.0206, dl_cm=2.8e26)
```

**The two guards fire at different moments.** `register_supernova` raises during registration
(`build_sn_grid` checks before it builds the dense grid), so the model never enters the registry.
`register_tde` **succeeds** in float32 and raises only on the first `predict`, because its guard is
inside the engine. A registry that contains `tde_gaussianrise_jax` is therefore not evidence that
the session can run it.

Two things make this awkward, and neither is hidden from you:

1. **The flag is global and session-wide.** A session that fits a TDE or a supernova runs the
   kilonova **11–23× slower**.
2. **It must be set before the first array is created**, so it cannot be a model default. Nothing in
   this package sets it for you — a silent default here would quietly change the cost of every other
   model in the session.

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
this code.** The test suite detects which redback is importable and selects the matching
`REDBACK_PRESETS` entry.

**Do not fit at `REDBACK_PRESETS["1.15"]`.** At `n_time=500` the AB magnitude spreads 0.55 mag under
a 1e-6 relative change in `mbh_6` (8.3e-6 mag at 5000), photometry-interpolation error reaches
0.6 mag, and 10% of the prior yields a one-point light curve. It is reproducible, not fittable.

## Performance reference

TDE, `n_time=5000`, on an A6000: **11.5 ms** for one light curve (the scan is launch-bound, hence
`unroll=16`), **9.0 µs/curve at batch 512**. Prefer vmapped consumers.

One trap worth knowing: a model that closes over parameter values in a `functools.partial` bakes
them into the scan's jaxpr, so calling it outside `jit` is a cache miss — a fresh ~0.6 s XLA compile
*per likelihood evaluation* (831 ms against 9.1 ms jitted). The factories jit once. Any new model
built the same way inherits the same trap.
