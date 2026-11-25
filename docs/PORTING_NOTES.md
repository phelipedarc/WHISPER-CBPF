# Porting notes: the JAX re-implementations of redback models

`whisper_cbpf.models.jax` re-implements a subset of [redback](https://github.com/nikhil-sarin/redback)'s
transient models in JAX, so that they can be jitted, vmapped and differentiated. The modules depend
on `jax` and `numpy` only — no redback, no astropy, no scipy, no sncosmo at runtime.

Re-implementing physics invites two questions: *is it the same model?* and *where is it not?* This
document answers both. It is the reference for every deviation from redback, with the measurement
that motivated it. The source modules carry short docstrings and point here.

If you want redback's own code rather than a port, use
[`whisper_cbpf.models.redback_adapter`](#9-the-redback-cpu-adapter) — it binds any of redback's 347
models directly, on the CPU.

**Contents**

1. [What is ported](#1-what-is-ported)
2. [The three rules a deviation must satisfy](#2-the-three-rules-a-deviation-must-satisfy)
3. [Numerical identities in the kilonova core](#3-numerical-identities-in-the-kilonova-core)
4. [The TDE cooling envelope](#4-the-tde-cooling-envelope)
5. [The supernova family](#5-the-supernova-family)
6. [Which redback? Version differences that change numbers](#6-which-redback-version-differences-that-change-numbers)
7. [Precision requirements](#7-precision-requirements)
8. [Performance, and what each family is good for](#8-performance-and-what-each-family-is-good-for)
9. [The redback CPU adapter](#9-the-redback-cpu-adapter)
10. [TiDE (CPU): calling libtide, not porting it](#10-tide-cpu-calling-libtide-not-porting-it)

---

## 1. What is ported

### Kilonova — `models/jax/kilonova.py`, `kilonova_two.py`

redback's `one_component_kilonova_model`, plus the shared substrate the other two families reuse:
the Planck function, the AB band integral, the filter export, the diffusion quadrature and the
interstellar-extinction laws. `kilonova_two.py` sums *N* such components in flux.

### TDE — `models/jax/tde.py`

Ported and validated: `_analytic_fallback`, `_cooling_envelope` (the Sarin & Metzger envelope ODE),
`_stream_stream_collision`, `TemperatureFloor`, `TDEPhotosphere`, `CocoonPhotosphere`, `Diffusion`,
`AsphericalDiffusion`, `Viscous`, `CSMDiffusion`, a photometric layer (flux density and AB band
magnitude), and the Gaussian-rise stitched light curve.

**Not ported — `DenseCore`.** It is broken in redback and unused by any TDE model.
`mask_all = mask_2 & mask_3 & mask_4` ANDs two boolean arrays of length *n* with
`np.arange(peak_idx, n)`, an integer index array of length *n − peak_idx*. For `peak_idx > 0` the
shapes do not broadcast and it raises; for `peak_idx == 0` numpy's bitwise `&` yields an integer
array whose only values are 0 and 1, which is then used as a fancy index — so it writes to rows 0
and 1 instead of selecting "after peak and optically thick". Verified against
`redback/photosphere.py:274-309`. Porting it would mean guessing the intent, which would change the
physics.

**Not portable** — `fitted` / `fitted_pl_decay` / `fitted_exp_decay` (they need an external compiled
package). `_tde_mosfit_engine`, whose ragged `dtype=object` tables are loaded from disk on every call,
is ported in `tde_fallback.py` with those tables built once and padded (see `tde_fallback` below).

### Supernovae — `models/jax/supernova.py`

Twelve model names over nine engines: `_nickelcobalt_engine`, `basic_magnetar`, `magnetar_only`,
`exponential_powerlaw`, `fallback_lbol`, `_shock_cooling` (Piro+ 2021) and `_csm_shock_breakout`
(Margalit 2022), assembled into `arnett`, `shock_cooling_and_arnett`, `basic_magnetar_powered`
(= `slsn`), `magnetar_nickel`, `csm_shock_and_arnett`, `exponential_powerlaw`, `sn_fallback`,
`sn_nickel_fallback`, `general_magnetar_slsn`, `type_1a` (= `arnett`) and `type_1c` (= `arnett`).
Four SED kinds: blackbody, cutoff blackbody, line, synchrotron.

**Not ported:** the `homologous_expansion` / `thin_shell` family and the Morag and Sapir–Waxman
shock-cooling variants. Each needs machinery this module does not have — a root-find or a table.
They are separate ports, not omissions of convenience.

### Further models, adapted from redback and redback-jax

Sixteen supernova names (`supernova_models()`), the two further TDEs, the two Metzger kilonovae and
four phenomenological shapes were added in 0.3.0; which files are adapted, and from what, is in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Each reproduces its redback model, and the
agreement is a test (`tests/test_adopted_*.py`).

- `csm_interaction` (`csm_interaction.py`): Chevalier–Chatzopoulos. redback's shock-coefficient table
  is read once and interpolated bilinearly in JAX, so the model is differentiable in `eta` and `nn`.
  The diffusion integral is redback 1.20's cumulative piecewise-linear recurrence, solved with an
  associative scan on a geometric grid that contains the epochs, so the answer at an epoch does not
  depend on the other epochs. It is not `csm_shock_and_arnett` (Margalit).
- `shock_cooling` and `shocked_cocoon`: evaluated at the epochs, each with its own photosphere. The
  cocoon's taper is written `sigmoid(2 (t_thin - t))`, equal to `(1 + tanh(.)) / 2` but without the
  cancellation that loses every digit in the deep taper.
- `general_magnetar_driven` (`magnetar_ode.py`): the ejecta ODE. `solver="auto"` uses `diffrax`
  (Tsit5, `rtol=1e-5`, `atol=1e-8`) when it is installed (the `[ode]` extra) and a fixed-step RK4 on
  1000 log-spaced steps otherwise, with a one-time warning; `"euler"` reproduces redback's own
  fixed-step scheme. float64 is required.
- `tde_fallback` and `tde_analytical`: the first has MOSFiT's fall-back engine node for node (MOSFiT's
  `Fallback`, which redback transcribes as `_tde_mosfit_engine`; see
  [its section](#tde_fallback-the-engine-and-deviations-from-redback) below): its luminosity before
  the viscous delay is redback's to a relative 1.64e-13 at worst, and after it differs by about 2e-4 dex
  at most (maxima 2.2e-4 over 60 prior draws x 60 epochs, 2.3e-4 on the 65 redback golden sets, 2.6e-4
  over 200 draws x 701 epochs).
  The second is redback's to float64 round-off (its default prior differs in `kappa`, [below](#tde_analytical-one-deliberate-prior-difference)). Both evaluate the epochs directly (no epoch grid);
  `tde_fallback` evaluates each distinct epoch once (item 6 of
  [its section](#tde_fallback-the-engine-and-deviations-from-redback)). The epochs, `t_exp` and the
  redshift are ordinary traced arguments.
- `tde_mosfit` (`tde_mosfit.py`): MOSFiT 2.1.1's `tde` model (MIT licence; the engine and viscous kernel are `tde_fallback`'s). It is **not** redback's `tde_fallback`
  with other names: MOSFiT's photosphere is `Rph0 a_p (L / (Leddlim L_Edd))^lphoto + r_isco` with no soft maximum and no `L_Edd`-only normalisation; at the canonical
  point redback's photosphere has 0.2500 of its radius and 2.0001 times its temperature at the peak (notebook issue U7; upstream finding 13). Against MOSFiT's own modules
  (88 parameter sets x 64 epochs, `tests/goldens/tde_mosfit_oracle.npz`): `L` median 4.9e-5 / p99 2.2e-4 / max 1.7e-3 dex, `T` 1.1e-5 / 6.0e-5 / 4.2e-4, `R` 1.4e-5 / 1.8e-4 / 8.1e-4,
  monochromatic F_nu median 4.1e-5 / p99 3.9e-4 / max 3.6e-3 dex, constraint score to a relative 7.8e-4 at worst.
- `tde_stream_stream` (`tde_stream_stream.py`): redback's `stream_stream_tde`. The engine is `tde.stream_stream_collision`; the Gaussian rise, the 100 + 1000 node table and redback's
  linear interpolation of the flux between the nodes are reproduced node for node. Against redback 1.21.0 (`tests/goldens/tde_stream_stream_oracle.npz`, 118 parameter sets):
  table L, T and R_eff to a median 2e-16 and 3e-15 at the 99th percentile; flux density at six wavelengths median 4e-16, p99 3e-14, max 2.4e-13 (relative; epochs inside the 1500-day table);
  band magnitudes against redback's sncosmo `magnitude` output (91 sets whose Gaussian peaks near the engine's) median 3.4e-5 mag, after the peak p99 0.016 / max 0.044 mag.
- `metzger_kilonova` and `magnetar_boosted_kilonova` (`kilonova_metzger.py`): redback's explicit
  Euler loop over 200 velocity shells, on a fixed internal grid instead of one built from the
  requested epochs, so a light curve does not depend on which epochs accompany it. They are not
  comparable parameter for parameter with `kilonova_one_jax`.
- The phenomenological shapes (`phenomenological.py`) are redback's functions with the logarithm of
  `exp_rise_powerlaw_decline` clamped (its gradient was NaN whenever an epoch preceded `t0`) and
  `bazin_sne` written without the overflow of `exp(-dt/tau_rise)` far before `t0`.

### The band integral on a GPU

Every band integral holds the Planck array behind `jax.lax.optimization_barrier` before the band
sum. The barrier changes no number (the identity in value and in both derivatives; |dmag| below
1e-12). Without it XLA fuses the double-precision `exp`/`expm1` into the reduction over
(batch, observations, wavelength nodes), and its GPU fusion autotuner then compiles several
candidate kernels of that reduction: 563 s for 480 parameter sets over 480 observations in six bands,
1.2 s with the barrier. A blackbody at `T = 0` is zero flux (the temperature is floored at 1 K in the
spectrum, below which the flux is zero in float64 anyway), as in redback, so the magnitude is
`mag_floor`, not NaN. The kilonova spectrum also pins the scaled distance, so the float32 gradient
with respect to a free redshift is finite.

### Reuse, not copying

The TDE and supernova modules import the blackbody, the AB weights and the filter export from
`kilonova.py`, exactly as redback shares `interaction_processes.py`, `photosphere.py` and `sed.py`
across model families. Copying them would silently reintroduce every numerical defect
[§3](#3-numerical-identities-in-the-kilonova-core) removes, the next time the original is fixed.
Nothing is shared for its *physics* — what is shared is arithmetic.

### `tde_fallback`: the engine, and deviations from redback

The fall-back engine is MOSFiT 2.1.1's `Fallback` (`mosfit/modules/engines/fallback.py`, MIT; redback's `_initialize_mosfit_tde_model` and `_tde_mosfit_engine` are the same code), ported
under rule **(a)**. `Fallback.__init__` runs once, offline, in `build_fallback_tables`: each simulation's own `(log10 t, log10 dM/dt)` nodes split at the peak of `dM/dt` into a rising and a
falling branch, and, for each pair of neighbouring simulated betas and each branch, the normalised log-time nodes of the curve with more nodes and the slope and intercept in `beta` of
log10 `dM/dt` on them. The output, `whisper_cbpf/data/tde_fallback_nodes.npz` (1.3 MB), equals MOSFiT's and redback's arrays bit for bit (it is built with astropy's
`G = 6.674299999999999e-8`, one unit in the last place below the package's constant, which would move the slopes by 7e-13). `Fallback.process` runs in the trace in `fallback_engine`:
the pair by `searchsorted`; MOSFiT's equality branch (a `beta` equal to a simulated one takes that simulation's own nodes) as a `where`; for stars between 0.3 and 1 and between 15 and 22
solar masses both gamma mappings (the curve with fewer nodes interpolated onto the other's mapped times) at the padded size, chosen by a `where` on the two node counts; MOSFiT's scalings;
then `dM/dt` linear in time between the nodes and zero outside them. Measured against redback's `_tde_mosfit_engine` and MOSFiT's `Fallback` at the same epochs, over 634 parameter sets
(the 88 MOSFiT and the 65 redback golden sets; `bb` = 0, 1, 2; `bb` giving exactly a simulated beta, for the 19 of 23 (4/3) and 18 of 20 (5/3) that a float `bb` near the exact
inverse reaches, and one unit in the last place either side; stars in each gamma regime, on its limits 0.3, 1, 15, 22, and below 0.1 solar masses): the luminosity before the viscous delay to a relative 1.64e-13 at worst, the final nodes to 2.0e-13, `t_peak`, `beta` and `R_*`
to 3e-15. The rate inherits MOSFiT's two discontinuities, the same in MOSFiT, redback and here: at each simulated beta (the equality branch; before the peak up to 1.47e-2 dex, over every
simulated beta a float `bb` reaches, at two stars per gamma) and at the limits of the gamma regimes, where a pure gamma's own nodes meet the blend's mapped ones (at 1e6 solar masses and
`bb` = 0.25, 0.5, 0.77, 1, 1.5: up to 2.3e-2 dex just above `mstar` = 0.3, 1.5e-2 just below 22, 8e-5 below 1, 5e-5 above 15; the 0.3.0 engine was continuous there to 2e-9). So the model
is piecewise smooth in `bb` and in `mstar`. Its gradient is finite everywhere; at a jump it is the derivative of the side the `where` selects, not a one-sided limit: at a simulated beta it is
that simulation's own curve, so for a star of one gamma the derivative in `bb` is exactly 0 there (at `bb` = 1 and 2, for instance).

1. **(c) The viscous kernel interpolates log10 `L` between the dense nodes**, where redback's `Viscous` interpolates `L` (this keeps the kernel float32-safe). It is the whole of the
   remaining difference: against `tde_fallback_bolometric` (60 prior draws x 60 epochs, 0.5 to 1000 d) median 1.1e-5, p99 1.4e-4, max 2.2e-4 dex, and 2e-14 with redback's own kernel
   on this engine (over 200 draws x 701 epochs, 1e-3 to 1e3 d: max 2.6e-4 dex); band magnitudes over 300 prior draws (LSST griz, ZTF gr, z = 0.05) median 2.2e-5, p99 3.7e-4, max 7.9e-4 mag. Before this release, with the tables resampled onto 512 equal
   steps of normalised log time, the same comparisons gave 5.4e-5 / 1.4e-2 / 2.1e-2 dex and 6.9e-5 / 1.0e-2 / 4.0e-2 mag.
2. **(b) Floor instead of zero.** Before the first fall-back and after the last node redback's luminosity is exactly 0; here it is 1e-30 erg/s, which keeps every derivative finite.
3. **(b) `bb` outside [0, 2]** (redback raises, MOSFiT reuses its previous call's `beta`): the rate holds the table edge (the node lookup clips `beta` to the simulated range, so the
   first or last simulation is used unchanged); the `beta` returned, which reaches the photosphere, is the unclipped mapping of `bb` (`bb` = 2.5 at 5 solar masses: the rate of `bb` = 2,
   `beta` 5.075). `tde_mosfit` clips `b` to [0, 2] before the engine, so there both agree.
4. **(b) Padding.** The ragged node arrays are padded to a fixed length (166 rising and 925 falling nodes) by repeating their last node, with the number of valid nodes stored beside them
   (the mask); a repeated node is a zero-length segment, which changes no interpolated value.
5. **(a) Exact spelling.** The products of the `bb`-to-`beta` mapping are held apart from their sums by `jax.lax.optimization_barrier`: under `jit` XLA on the CPU fuses them into a
   multiply-add with one rounding, which moved 18 % of the betas by one unit in the last place and so decided the equality branch differently from MOSFiT. With the barrier `beta43` and
   `beta53`, which the equality branch compares, are MOSFiT's to the last bit. A second barrier holds the node times before the first is subtracted, so that XLA cannot fuse the subtraction
   with the last product and the first node is exactly 0, as MOSFiT's strict `times < time[0]` needs (a guarantee: without it the first node was also 0 in 2000 random sets on the CPU and a GPU). In a gamma blend the `beta` returned is MOSFiT's to one unit in the last place, not to the bit (under `jit` it differs
   in 21 % of 2000 random blends on the CPU, 10 % on a GPU: XLA divides by a constant as a product with its reciprocal); it reaches only the photosphere. The final interpolation's search
   is written out (`_search_right`, the same nodes as `jnp.searchsorted`) because `method="scan_unrolled"` let XLA fold its first step into a constant of batch x epochs entries, and the
   compiled program grew with the batch. `dM/dt` is interpolated relative to its maximum and scaled in log10, and the luminosity carried in log10 through the cap, so the engine holds in
   float32: the bolometric light curve within 1.2e-4 dex of float64 over 60 prior draws (1.1e-4 over 300), the engine alone within 2.8e-5 (4.6e-5). Two exceptions, for which float64 is
   needed. In float32 the equality branch compares float32 betas, so it fires at other `bb` than in float64: of the round values `bb` = 0.00, 0.01, ..., 2.00 the branch differs at 0.04,
   0.08, 0.24, 0.88 and 0.96, which moved the rate by up to 3.6e-3 dex (0.88, at 0.6 solar masses; at most a jump's size), against 4.4e-5 dex at the other 196, so a pinned `bb` should be
   evaluated in float64. And the float32 derivative in `mstar` is wrong above about 11 solar masses (the values are right): the quotient rule of `stellar_radius` overflows float32 (`mass**19.5` squared), which predates the new engine. This is documented, not fixed: the package enforces float64 for `tde_fallback` and `tde_mosfit` (their physics functions raise without `jax_enable_x64`), so no fit sees the float32 gradient.
6. **(a) Each distinct epoch is evaluated once.** Observations that share a time (one epoch in several bands, a repeated exposure) share the engine, the viscous integral and the
   photosphere, which run at the distinct epochs `tau[first]`; `T` and `R` are copied back with `inv` before the band integral, which stays per observation. This is redback's own
   structure: `Viscous.convert_input_luminosity` evaluates at `np.unique(time)` and gathers back with `searchsorted`. The first port had dropped it.

   It is an identity:
   - an observation enters only through its own `tau = (t - t_exp)/(1 + z)`, and equal times give equal `tau`;
   - the only quantity that mixes epochs, the end of the dense grid, is a maximum over the set of epochs, which the distinct epochs (padded by repeating one of them) leave unchanged;
   - `np.unique` merges `+0/-0` and NaNs, which is exact too: `+0` and `-0` give the same `tau` whenever a nonzero time is subtracted from them (`t_exp_days` or a free `t_exp`),
     and otherwise both are at the time zero, not alive, at `mag_floor`; a NaN row is never alive, at `mag_floor`.

   The map `(first, inv)` is built on the host with `np.unique`:
   - from the concrete epochs in `predict` and under a caller's `jit` with numpy epochs;
   - by `log_density` from the padded epochs it passes as data;
   - traced epochs without a map are evaluated per observation, as before;
   - a map that does not describe the epochs gives NaN, not a wrong light curve.

   Measured against per-observation evaluation on the CPU, 200 prior draws, fixed and free `t_exp` and redshift, on every call path (host `predict`, `jit`/`vmap` of `predict_jax` on
   numpy and on traced epochs, `make_log_prob_jax`, `make_batched_predict_jax`, `log_density`, `likelihood_max_opt`, `profile`, `fit_batch`), 35 epochs x 5 LSST bands:
   - magnitudes: host `predict` bitwise; at most 6.4e-14 mag on any path (at least 71 % bitwise);
   - log posterior through `log_density`: at least 98.5 % bitwise, to a relative 7.9e-16;
   - gradients: within 9.3e-14 of their largest component. Only the summation order over the observations of an epoch changes;
   - `fit_batch` on 35, 28 and 21 epochs in one call, 16 walkers x 200 steps started with `init="prior"`: the same chains, bit for bit. With the default `init="prior_scan"` the
     start's climb amplifies the round-off: in 5 of these 6 fits (both models) the starting walkers moved by 2.1e-10 to 8.7e-5, so the chains differ from the first step while they
     sample the same density; any chain can also part after an accept/reject decision at round-off.

   Over five light curves (adding 28 and 21 epochs, an unsorted curve with repeats, rows before the time zero and upper limits, and the shipped LSST alert, whose epochs are all
   distinct): magnitudes 6.4e-14 mag, log densities to a relative 3.7e-14, gradients within 1.9e-12. These maxima are those curves':
   over 200 random light curves (1-81 distinct epochs, 2-311 rows in up to six bands, unsorted, with repeats, rows before the time zero and upper limits; 16 prior draws each), on
   the CPU the magnitudes reached 3.2e-13 mag (`tde_mosfit`) and the gradients 2.7e-12, and host `predict` was bitwise on all but one `tde_fallback` curve (91.3 %, at most
   3.6e-15 mag); on one A6000 GPU under XLA's default flags, on 50 of them, host `predict` differed in 5 of the 100 (curve, model) pairs, by up to 4.3e-14 mag.

   Those differences are XLA's shape-dependent rounding: the per-observation code itself moves by up to 6.4e-14 mag between a 35-row and a 175-row call (82 % bitwise).

   `likelihood_max_opt`'s peak, and the AIC and BIC from it, are not reproduced to round-off: its climbs follow the density's round-off and can end at another point or in another
   basin. From the same fit (the same draws and starting ln L, CPU), one of two `tde_mosfit` members tried reached ln L 586.15 where the per-observation code reached 580.33 (AIC
   and BIC 11.6 lower; `starmass` 0.10 against 0.36, `efficiency` 0.21 against 0.06), the other the same peak to 2.1e-8; on a GPU two calls of the per-observation code from one fit
   differ by up to 3.0e-3. A model-comparison table built on these peaks must be re-run.

   On one A6000 GPU, under XLA's deterministic flags (`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`; without them the per-observation code itself is not
   reproducible from run to run there), the same comparison on the five light curves: host `predict` and `log_density`'s value programs bitwise on every curve (the value it
   returns with its gradient within 4.2e-16 of the size of its summed terms; through `make_log_prob_jax` to a relative 2.3e-14); magnitudes at most 2.0e-14 mag (at least 71 % bitwise); log densities to 3.5e-15 of the size of their summed terms; gradients
   within 7.4e-13 (3.7e-14 on the 35-epoch curve); `fit_batch` started with `init="prior"` the same chains, bit for bit; no NaN from the map's check.

   The derivative with respect to the observation times (traced, with a map; no sampler takes it) is per distinct epoch, carried by its first observation. `unique_epochs=False` at the
   factory evaluates every observation (with `geometric=False`, item 7, the code before these options).

   The figures of this item are for unique epochs alone (`geometric=False`); item 7 gives the default, the two together, against the code before both.
7. **(a) The viscous kernel finds each node's segment by arithmetic** (`geometric=True`; `tde._interp_geometric`, as the supernova diffusion does) instead of by a
   search. It is the default (`geometric=True`), chosen after the measurements below; `geometric=False` keeps the search. The dense grid is geometric (`geometric_dense_times` sets its endpoints exactly), and the interpolation is `jnp.interp`'s own arithmetic in the same order. The value is
   therefore identical when the same segment is chosen, and differs by round-off when a node within rounding of a grid point takes the neighbouring segment.

   Measured on the CPU against the search: the kernel's log10 `L` bitwise over 200 prior draws x 48 epochs (its gradient within 1.4e-14 of the largest component); magnitudes over
   200 prior draws x 175 observations bitwise, with and without the epoch map; host `predict` (one parameter set per call) 99 % bitwise, at most 5.4e-14 mag (in the one-set programs
   only; the batched programs are bitwise); the log posterior of `log_density`'s batched programs bitwise (one parameter set per call, as `likelihood_max_opt`'s value program
   evaluates it, 45 to 48 of 48 draws bitwise, to a relative 8.6e-15), its gradient within 3.0e-15. On one A6000 GPU, with XLA's default flags and with its deterministic ones,
   host `predict` and the log posterior of `log_density`'s batched programs bitwise, the gradient within 1.0e-14. Together with item 6, against the code before both, on the five
   light curves and every call path of item 6: magnitudes at most 8.9e-14 mag, host `predict` at least 98.1 % bitwise, log densities to 7.8e-15 of the size of their summed terms,
   gradients within 1.9e-12, `fit_batch` started with `init="prior"` the same chains, bit for bit; on the GPU (deterministic flags, item 6) every value as item 6's GPU numbers
   (host `predict` and `log_density`'s value programs bitwise, the value returned with a gradient within 4.2e-16), gradients within 7.4e-13, the same chains. These maxima are
   those curves': over 200 fresh random light curves (1-79 distinct epochs, 1-317 rows in up to six bands; fixed and free `t_exp` and redshift), on the CPU the magnitudes
   reached 1.9e-13 mag and the gradients 1.6e-12 of their largest component (host `predict` at least 99.0 % bitwise); on the GPU under the deterministic flags, on 17 of those
   curves, every flux path and every value program of the log density was bitwise (the value returned with a gradient within 3.0e-16) and the gradients within 8.0e-14.

   It is the default of both factories and of every function that builds its own dense grid with `geometric_dense_times`: `tde_fallback_bolometric_log10`,
   `tde_fallback_ab_magnitude`, `tde_mosfit_curve`, `tde_mosfit_ab_magnitude`, `tde_mosfit_constraint_score` and `tde_mosfit_flux_density_mjy` (against the search, the bolometric
   function, the score and the flux density were bitwise over 200 prior draws x 48 epochs, the gradients of the first two within 3.5e-15 and 5.3e-15; in float32 the bolometric function is the
   search bit for bit over 360 draws x 60 epochs, so its float32 figures, measured on the search, hold). Two keep the search. `viscous_log10` takes the caller's `dense_times`, so
   it keeps `jnp.interp` unless asked (`geometric=False` by default): on a valid grid that is not geometric the arithmetic picks wrong segments (a linear 1000-node grid from 1e-5 d
   to the last epoch + 100 d, 20 prior draws x 48 epochs: log10 `L` off by a median 0.52 dex, up to 3.95 dex). The constraint wall (`tde_mosfit_constraint`) is one function for
   every model and must keep the numbers of `unique_epochs=False, geometric=False` with a wall, so it always passes `geometric=False`. The arithmetic needs at least 2 dense nodes
   (with one it divided by zero and gave NaN for every observation): `geometric=True` with `dense_resolution` below 2 raises `ValueError` at the factory call and in
   `viscous_log10`; `geometric=False` with one node is unchanged. `unique_epochs=False, geometric=False` is the code before both options. `geometric` enters a model's state only
   when True, so a `geometric=False` model keeps the `model_record` identity it had before the option, and the default model is identified as geometric.

   Speed on one A6000 (float64, default XLA flags, 35 epochs x 5 bands, free `t_exp`), against unique epochs alone (`geometric=False`), in one run (host load 13-20): 1.10-1.50x per
   evaluation on the 175-row curve, 1.06-1.38x on the shipped alert and 1.12-1.49x with every time distinct (one program is not faster: `tde_fallback`'s `jit(vmap(predict_jax))` at 4
   sets, 0.95x, rounds 0.87-1.07). `fit_batch`, 32 walkers x 5000 steps, one light curve: in that run (host load 13-28 at the start of the median rounds) 40.7 → 38.1 s
   (`tde_fallback`) and 37.9 → 34.6 s (`tde_mosfit`), the sampling 12.6 → 10.1 s and 12.5 → 10.0 s; in an earlier run (host load 10-15) 36.4-37.1 → 32.7-34.1 s over four
   (truth, model) pairs, and 64 light curves (load 11) 620-624 → 449.6-454.9 s. Compile: the summed compile time of 18 programs 69.8 → 70.6 s (`tde_fallback`) and 65.3 → 69.3 s
   (`tde_mosfit`), single programs 0.86-1.34x (`jit(vmap(...))` at 256 sets on the all-distinct curve 3.14 → 4.22 s and 3.25 → 4.36 s); against the code before both options,
   median 0.98-0.99 per program (0.78-1.58). Host `predict` on a CPU, one parameter set on the 175-row curve (the earlier run, load about 13): 9.76 → 6.67 ms (`tde_fallback`) and 8.64 → 5.68 ms
   (`tde_mosfit`).

### `tde_mosfit`: deviations from MOSFiT

1. **(c) The engine is MOSFiT's; the viscous delay is redback's.** The engine is `tde_fallback`'s, which is MOSFiT's `Fallback` node for node (above; `t_peak`, `beta`, `R_*` and `r_p`
   agree to 3e-15). The viscous delay is redback's node quadrature (log10 `L` interpolated between the nodes, trapezoid rule on 500 + 500 log-mirror nodes) on the dense grid
   `geomspace(1e-5, t_last + 100, 1000)`, not MOSFiT's exact piecewise-linear solution on its `DenseTimes` grid, and it is the whole of the measured difference: `L` median 4.9e-5, p99
   2.2e-4, max 1.7e-3 dex (the maximum at the earliest epoch, 0.03 `t_peak`, of a 0.24 solar-mass star). MOSFiT's exact filter applied to this engine gives 1.2e-5 / 2.1e-4 / 1.6e-3 dex on the
   same 1000-node grid, 3.1e-7 / 3.9e-5 / 2.3e-4 on 6000 nodes and 4e-14 on the golden's own grid: the tail is the resolution of the 1000-node grid on the early rise, the median the
   quadrature. Before this release the engine read tables resampled onto 512 equal steps of normalised log time, interpolated in `beta` and gamma at fixed phase and log-linearly in time,
   and the difference was 8.8e-5 / 8.2e-3 / 1.8e-2 dex. Its tail was that engine (the resampling between the sparse nodes before the peak and the blend at fixed phase), not the
   discretisation, as an earlier version of this note said. MOSFiT's own default 100-step dense grid differs from its converged answer by a median 5e-5, p99 2.5e-3, max 3.9e-3 dex.
2. **(b) Floors instead of exact zeros.** MOSFiT's luminosity is exactly 0 before the first fall-back (and after the end of its table); here an epoch at or before 0 is at 1e-30 erg/s and
   `mag_floor` in magnitude, and the photosphere power uses `max(L, 1e-30)` (its derivative at 0 is infinite; at the floor the term is 1e-70 of `r_isco`).
3. **(b) `b` is held in [0, 2].** MOSFiT raises, or reuses the previous call's `beta` (finding 10); here the table edge is used for the rate and for `beta`, consistently.
4. **(b) A guard at `tau <= 0` in the constraint's `a_t`**, where MOSFiT sets `a_t = 0` by assignment (same value, finite derivative).
5. **(c) Not ported: the early-time power-law extrapolation of `Fallback`** (active only when the first fall-back is later than the first dense epoch). Measured with the first fall-back 10 days
   after the first epoch: at most 0.018 dex, within 6 days of the first fall-back, zero after 20 days.
6. **(c) Not ported: host extinction** (`losextinction`). `ebv_mw` / `ebv_host` are whisper's laws.
7. **(c) The stellar-mass prior is log-uniform on MOSFiT's box (0.01 to 30), not Kroupa** (the prior layer has no such family; MOSFiT's Kroupa puts 44 % of the draws below 0.1 solar masses,
   log-uniform 29 %). `kroupa_icdf` and `kroupa_log_pdf` are MOSFiT's distribution (**(a)**: the quantile function inverts the analytic cumulative segment by segment; 1e-9 relative to MOSFiT's).
8. **(c) The constraint.** MOSFiT adds `-10 (Rs/rp)^3 - 10 max_t (R/R_max)^2` to the log-likelihood; `tde_mosfit_constraint_score` is that function (epochs as MOSFiT's observation epochs).
   The hard wall `constraint=` arms is whisper's addition (`Rs/rp < 1` and `max R/R_max < 1` on 64 epochs from 0.03 to 30 peak times), default off; it rejects 45 % of MOSFiT's box prior.
9. **(c) Flux convention.** `dilation=True` gives `(1+z) pi B_nu R^2 / d_L^2`, which is MOSFiT's band path; MOSFiT's frequency-row path of `Photometry` is `(1+z)` larger (finding 12).
10. **(a) Each distinct epoch is evaluated once**, as in MOSFiT's own `Viscous` (`mosfit/modules/transforms/viscous.py`: `np.unique` of the times, then `searchsorted`), which the
    first port had dropped: `tde_mosfit_curve` runs at the distinct epochs, and `T` and `R` are copied to the observations before the band integral. The argument is `tde_fallback`'s
    item 6. The constraint wall evaluates `tde_mosfit_curve` on its own 64 epochs, unchanged, with the search. The viscous kernel finds its segments by arithmetic by default
    (`geometric=True`, `tde_fallback`'s item 7; `geometric=False` keeps the search): against the search on the CPU, magnitudes bitwise over 200 prior draws in the batched programs, host `predict` 98.7 % bitwise and at most 3.4e-13 mag,
    the log posterior of `log_density`'s batched programs bitwise (one parameter set per call 47 or 48 of 48 draws bitwise, to a relative 2.0e-15) and its gradient within 3.0e-15
    (on the GPU host `predict` and the log posterior of `log_density`'s batched programs bitwise); together with the map, against the code before both, magnitudes at most
    2.5e-13 mag, host `predict` at least 98.5 % bitwise, log densities to 1.2e-14, gradients within 1.11e-12, `fit_batch` started with `init="prior"` the same chains (CPU; over
    200 fresh random light curves, as `tde_fallback`'s item 7, the magnitudes reached 2.8e-13 mag and the gradients 5.1e-12, host `predict` at least 99.2 % bitwise); on one A6000
    GPU (deterministic flags) every value equal to the default call's (the GPU numbers below), gradients within 1.6e-13, the same chains.

    Measured (CPU, as `tde_fallback`'s item 6, 35 epochs x 5 bands): magnitudes at most 1.11e-13 mag (at least 75 % bitwise; host `predict` bitwise); log posterior through
    `log_density` at least 99.5 % bitwise, to a relative 2.1e-16; gradients within 2.8e-13; `fit_batch` started with `init="prior"` the same chains, bit for bit (with the default
    `init="prior_scan"` the chains differ from the first step, as item 6 says). Over the five light curves: magnitudes 1.11e-13 mag, log densities to a relative 3.6e-15,
    gradients within 1.11e-12. The per-observation code itself moves by up to 8.6e-14 mag between a 35-row and a 175-row call (75 % bitwise). On one A6000 GPU (deterministic
    flags, as item 6): host `predict` and `log_density`'s value programs bitwise on all five curves (the value it returns with its gradient within 4.2e-16 of the size of its
    summed terms; through `make_log_prob_jax` to a relative 4.5e-15), magnitudes at
    most 7.11e-15 mag, log densities to 1.8e-15, gradients within 1.6e-13, `fit_batch` started with `init="prior"` the same chains, bit for bit. `likelihood_max_opt`'s peak
    is not reproduced to round-off (item 6).

### `tde_stream_stream`: redback behaviours reproduced, and deviations

Redback behaviours the defaults reproduce (each has a keyword that removes it and is an exact identity of the original expression with the quirk removed; upstream notes 30-38):

* **Q1** `sphere_radius`: `R_eff = sqrt(L / (pi sigma T^4))` against a sphere SED, 4x the bolometric energy, 1.505 mag. (30)
* **Q2** `del_omega_in_pi`: `del_omega` used in steradians although documented in units of pi; T is `pi^(1/4)` hotter than the documented model. (31)
* **Q3** `double_dilation`: only redback's `flux_density` output dilates time twice (features at `(1 + z)^2 t`; 3.92 between z = 0.01 and z = 1); its band outputs do not, so the band function defaults to `False`
  and `tde_stream_stream_flux_density` to `True`. (32)
* **Q4** `nu4_spectrum`: the band spectra are `(nu / 6e14)` times the Planck function: -0.42 mag in u, +0.38 in i at z = 0.05, -1.03 at 2030 A. (36)
* **Q5** `extrapolate`: past the 1500-day grid redback extrapolates linearly (negative luminosities). (33)
* **Q6, Q7**: the Gaussian normalisation `L_0 / f1` overflows (0.225 % of redback's prior box returns inf, 37); `peak_time` is independent of the engine's peak, so 31 % of the prior box peaks at more than
  20 times the engine's maximum luminosity. (38) Reproduced (Q7) / made finite (D1).

Deviations:

1. **(a) / (b)** The rise is `L_0 exp(e_0 - e_1)`, not `(L_0 / f1) exp(-e_1)`: the same number to a relative 8.5e-14 wherever redback is finite, and finite (capped at 1e300 erg/s) where redback's `L_0 / f1` is inf/NaN.
2. **(b)** Epochs after the end of the table (1500 days of model time) and at or before time zero are masked (flux 0 / `mag_floor`) instead of linearly extrapolated. `extrapolate=True` reproduces redback to 5.5e-11.
3. **(c)** The band integral is the real bandpass (as in the other TDE models) where redback's `magnitude` output evaluates the spectrum on 100 wavelengths and splines it in time with a cubic;
   here the flux is interpolated linearly between the same nodes (redback's own `flux_density` path). Measured size in the paragraph above; before the peak, across a steep Gaussian wing, redback's spline differs by up to 3 mag in one parameter set of 91.
4. **(b)** `inc_tcool` may be a traced value (the two engines are computed and selected); a Python number picks the engine at trace time.
5. **(b)** The temperature is floored at 1 K inside the Planck function (package-wide).

### `tde_analytical`: one deliberate prior difference

The physics is redback's to float64 round-off; the **default prior** is not. `default_prior()` is
redback's `tde_analytical.prior` (read from the installed redback at run time, else the
transcription `fallback_prior()`) with **`kappa` LogUniform(0.05, 100)** where redback has
**Uniform(0.05, 100)**, over the same bounds. It is a rule (c) change: a change of default, stated,
measured and justified here. `LOG_UNIFORM_OVER_REDBACK = ("kappa",)` names the parameters this
applies to; `redback_default_prior()` and `tde_analytical_model(prior="redback")` give redback's
prior unchanged. Precedent: `kilonova_two` defaults to Villar's prior, not redback's
([MODEL_COMPARISON](MODEL_COMPARISON.md#why-the-default-prior-is-villars-and-not-redbacks)).

**Why.** `kappa` is the only Uniform in redback's file, over 3.3 decades; the other wide scale
parameters (`mej`, `vej`, `kappa_gamma`, `l0`, `t_0_turn`, `temperature_floor`) are already
LogUniform there. Under the Uniform, P(`kappa` < 0.5) = 0.45 % and P(`kappa` < 0.2) = 0.15 %
(LogUniform: 30 % and 18 %). On injected `tde_analytical` light curves with `kappa` = 0.2, nested
sampling and `emcee_jax` settled in a second solution (`kappa` 60-100, `mej` 20-100,
`kappa_gamma` near 1e-4) 42-253 nats below the truth, with `t_exp` 4.5-11.7 d late; with the
LogUniform 14 of 14 fits on the same curves sat in the true solution (`kappa` median 0.23-0.28) and
`emcee_jax` converged in 36-55 s, where before it ran to its cap.

**What it costs.** A prior that favours small `kappa` makes a solution that needs a large one
harder to find. On the model-comparison matrix's MOSFiT-truth curve the best allowed
`tde_analytical` solution needs `kappa` 51-80 (6 % of the LogUniform prior mass, 29 % of the
Uniform's); nested sampling under the LogUniform missed it in 4 of 4 runs (nlive 500 and 2000),
1262 nats below. Everywhere else in that matrix the winners, ranks and margins (within 2.2 nats)
are those of the Uniform prior. Pass `prior="redback"` to reproduce redback's fits.

---

## 2. The three rules a deviation must satisfy

Every difference from redback in these modules is one of:

- **(a)** an exact algebraic identity;
- **(b)** a change in how an *invalid* region is *represented* — a mask instead of a slice, a
  guarded value instead of `inf`/`NaN` — never a change to values inside the valid region;
- **(c)** a change of default or of discretisation that is stated, measured and justified here.

Two disciplines follow from JAX rather than from physics, and they apply throughout:

**Data-dependent slices become boolean masks.** redback returns `Lrad[:constraint]`, whose length
depends on the data; under `jit` an array shape must be known at trace time. The full arrays are
returned alongside a `valid` mask, and `arr[np.asarray(valid)]` outside `jit` reproduces redback's
slice.

**Guards go on the inputs, not on the outputs.** redback wraps loops in
`np.errstate(invalid='ignore', divide='ignore')` and slices the resulting `inf`/`NaN` away
afterwards. JAX cannot: `jnp.where(c, f(x), g(x))` evaluates *both* branches, and its VJP multiplies
the discarded one by zero — and `0 * inf = NaN`. A `NaN` that never appears in the forward output
still destroys the gradient. So every fractional or negative power has its *argument* floored
before the power is taken, never its result repaired afterwards, and the floors sit strictly
outside the region redback evaluates.

**The arithmetic is kept in redback's exact spelling.** A forward-Euler recursion amplifies
round-off: writing `Rv * Rv` where redback writes `Rv ** 2` — the same number, a different XLA op —
changed `Rv` in the 16th digit at step 4 and `Me` by 5e-6 *relative* by the end of the integration.
The guards are pure `where` selections that do not touch the expressions. Do not "simplify" them.

---

## 3. Numerical identities in the kilonova core

Each is a rearrangement, not an approximation. Identities 1, 4, 5, 7 and 8 are what make the
float32 path usable.

1. **Heating rate** (`redback/kilonova_models.py:1884`).
   `0.5 − arctan((t − t0)/sig)/π  ≡  arctan(sig/(t − t0))/π` for `t > t0`. The original subtracts
   two numbers that agree to ~7 digits by *t* = 1 d, losing ~9 of float64's 16 digits and all of
   float32's.

2. **Diffusion integral** (`:1885-1889`).
   `exp(t'²/td²)` overflows float64 once `t' > 26.6 td`, giving `inf * 0 = NaN`. Folding the
   constant prefactor inside the integral gives `1/td ∫ f(t') (t'/td) exp(−(t² − t'²)/td²) dt'`.
   Expanding `cumulative_trapezoid` and distributing the prefactor reproduces redback's *discrete*
   sum term by term, so the quadrature is identical.

3. **Planck denominator.** redback's `_inverse_expm1` split at argument > 50, reproduced exactly.

4. **Planck prefactor** (float32 under `jit`).
   `2πhν³/c²  ≡  (2πk³/(h²c²)) x³ T³`, with `x = hν/(kT)`. Scaling `ν` alone does not survive
   `jit`: XLA rewrites `(c·ν)**3` to `c**3 · ν**3`, so `ν³ = 1.3e44` overflows float32 to `+inf`
   while the folded constant underflows to exactly 0, and `inf * 0 = NaN`. Both `x` and `T` are
   O(1), so the rewritten form has nothing left to overflow; `jax.lax.optimization_barrier` pins
   the scaling.

5. **Gradient-safe divisors** (float32 backward pass).
   `2κm₀/(βv₀c) ≡ TDIFF_CONST·κ·mej/vej`; `(L/(4πσR²))^¼ S^¼ ≡ T_CONST·L^¼/√R`;
   `(L/(4πσT_f⁴))^½ S^½ ≡ R_FLOOR_CONST·√L/T_f²`. JAX differentiates `x/y` with respect to `y` as
   `−x/(y·y)`; in float32 that square overflows for `|y| > 1.84e19` and silently returns 0. Both
   original divisors exceed that, and both depend only on `vej` — which is exactly why `d/dvej` was
   wrong, and sign-flipped, while `d/dmej` and `d/dκ` were correct to six digits.

6. **Diffusion quadrature.** `s = (t² − t'²)/td²` on the outer panel and `z = log t'` on the inner,
   replacing the geomspace trapezoid grid. Documented in full at the quadrature constant block in
   the source. Not an identity in the discrete sense — it converges where redback's grid does not
   — so it is **not the factories' default** (below).

**The factories' default is redback's own grid.** redback 1.20 solves
`one_component_kilonova_model` with a trapezoid on
`get_optimal_time_array(1e-2, 7e6, dense_resolution=500, user_times=epochs)` — 500 geometric nodes,
70% of them between half the first epoch and twice the last — then interpolates `T` and `R` to the
epochs. Its kernel width `td²/2t` shrinks as the node spacing grows, so past ~2.66 `t_diff` that
grid is too bright: 1.19 mag at 20 d against a converged grid, and the error grows with `t/t_diff`
(2.45 mag at 30 d for `t_diff` = 1.2 d, `test_factory_default_is_redback_at_every_epoch`; before
2.66 `t_diff` the two agree to ≤ 3 mmag). The project chose parity with
redback's default over the correction, so `kilonova_model`, `kilonova_two_model` and
`kilonova_three_model` take `time_grid="redback"` by default: `kilonova.redback_time_grid` rebuilds
redback's grid from the concrete epochs, and `bolometric(..., time_grid=grid)` solves the same
trapezoid (as a `lax.associative_scan`) and interpolates the photosphere. It matches redback's
`flux_density` to ~3e-9 relative out to 30 d, the residual being identity 7's series; the
two-component default is the sum of two one-component redback calls, the definition whisper
adopts (redback's own two-component model stops at 6 d). `time_grid=None` is the converged
quadrature — whisper ≤ 0.1.0's only behaviour, and the one to use for late epochs. With the
default, `predict_jax` needs concrete times, as the supernova's does. The low-level
`bolometric` / `flux_density_mjy` / `ab_magnitude` keep `time_grid=None` as their default, so a
traced explosion time still works there.

7. **Series for `log1p(x)/x`** (float32 and float64 backward pass).
   `log1p(x)/x ≡ 1 − x/2 + x²/3 − x³/4 + …`. The naive quotient differentiates to a term in `1/x²`,
   and `x = 2bt^d` reaches 1.2e-17 at the inner quadrature's first node. `d/dbv` and `d/ddv` then
   overflow with *opposite* signs, and `−inf + inf = NaN`. No floor can fix it; see
   `_log1p_over_x`.

8. **AB-zeropoint units for synthetic photometry** (float32 backward pass).
   `f_ν/AB_ZEROPOINT ≡ (PLANCK_T3/AB_ZEROPOINT) x³ T³ · ratio/(e^x − 1)`. `d(mag)/d(f_ν)` carries
   `1/(u · norms)` and reaches 7.0e38 > `FLT_MAX` in cgs units once extinction can push a band past
   AB ≈ 43. Measuring the spectrum in AB-zeropoint units instead leaves ~19 decades of headroom.

**Optional extras**, all default-off and all costing exactly nothing when unused: an explosion time
(`source_time_s()` / `pre_explosion()` map observer days to source-frame seconds; `t_exp` is a
traced, differentiable scalar); interstellar extinction (F99 by default, CCM89 available, Milky Way
and/or host, with the wavelength shape precomputable at setup — the no-extinction sentinel is
`None`, so the term vanishes at *trace* time and the compiled graph is byte-identical); and exact
filter cell integrals in `make_filter_set` (2.127 → 0.127 mmag at `n_wave = 1000`; the default is
5000).

---

## 4. The TDE cooling envelope

### 4.1 The sequential loop becomes `lax.scan`

`_cooling_envelope` is a forward-Euler integration whose step reads exactly four quantities from
the previous step: `(Me, Ee40, MdotBH, Edotbh40)`. Those become the scan carry; `Mdotfb` is
precomputable and is passed as a scan input at both `ii` and `ii−1`. Same `dt`, same arithmetic,
term for term. Against redback's own function, the maximum relative difference in `L` is **6e-15**
over the whole valid region, and in `T` and `Rv` **1e-14** over the first 95% of it.

### 4.2 The light curve stops where the envelope stops

This is the one place the port deliberately does not chase redback, and the reason is measurable.

redback stops at `min( first(Rv < Rcirc/2), first(Me < 0) )`, inside a shared `try/except` so that
if *either* search comes up empty *both* indices become `len(time_temp)`. But `Me < 0` is only
reached one to three steps *after* `Ee` crosses zero — i.e. after `Rv` has already diverged, at
which point `Rv` oscillates between +1e14 and −1e15 on successive Euler steps (measured:
`Rv[3434:3442]` = 3.2e14, 2.4e14, 4.5e14, 1.9e14, −9.5e15, 2.1e14, 9.6e14, 1.8e14). The index is
read off a chaotic trajectory, and it is not reproducible:

| comparison | identical | max deviation |
|---|---|---|
| redback's loop vs itself, `Rv**2` respelled as `Rv*Rv` | 100.0% | 0 steps |
| redback's loop vs itself, `beta` nudged by one ULP | 90.0% | 4 steps |
| an implementation carrying redback's raw trajectory alongside the guarded one, purely to reproduce that index | 77.3% | 1793 steps |

The last row is the point: an implementation built specifically to reproduce the index could not,
because there is no stable answer to reproduce. What the model actually says is simple — the
envelope is over when it runs out of energy, runs out of mass, or has shrunk to the circularisation
radius:

```
stop = (Ee <= 0) | (Me <= 0) | (Rv < Rcirc/2)
```

Each term is exact while the envelope lives, which is the only place any of them needs to be, and
none reads a diverged quantity.

**The deviation, measured.** redback's shared `try/except` fires when *either* search comes up
empty. Not the branch you would expect: in 3000 draws at `n_time=500`, `Rv < Rcirc/2` firing while
alive with `Me` never going negative occurred **zero** times. What does occur (0.17%) is the mirror
image — `Me < 0` fires and `Rv < Rcirc/2` never does — because once `Me` goes negative, `Ee` and
`Me` flip sign *together* on every step (`Ee`: −1.85e11, +2.15e13, −2.44e15 …; `Me`: −1.76e33,
+2.15e35, −2.43e37 …), so their ratio pins `Rv` at a constant +1.57e12 cm, permanently above
`Rcirc/2`. redback then returns all 500 points, including 53 with `Me < 0` and 53 `NaN`
temperatures. A further 0.30% of draws at `n_time=5000` have *neither* condition firing. This
module stops at the death index in both cases. Elsewhere the two agree within redback's own
irreproducibility: restricted to draws where both of redback's conditions do fire, this module is
exact on 85.3% (n=500) / 81.9% (n=5000) with a maximum deviation of 5 and 14 steps, against
redback-vs-itself-under-one-ULP at 93.5% / 89.3% and maxima of 6 and 12.

`meaningful` narrows `valid` by the same discipline — the model's own domain rather than a
percentage of it. The envelope must still be optically thick enough to put its photosphere outside
itself, `Lamb > 1/e`, i.e. `Rph > 0`; below that redback's `Rph` goes negative and is then squared
into a blackbody, reporting a positive flux from a negative radius (1.77% of prior draws). Trimming
a fixed fraction of the curve instead is not a physical statement: it removes good points from
short curves and keeps bad ones on long.

### 4.3 An empty envelope is a mask, not a crash

redback's `constraint` can be 0 — 0.43% of redback's shipped prior, by Monte Carlo over the
grid-independent condition `Rv0 < Rcirc/2` — and redback 1.12.0's
`output.time_since_fb = output.time_temp - output.time_temp[0]` then raises `IndexError` on an
empty array. Here it is simply an all-`False` mask, and the photometry returns exactly zero flux.

redback 1.15.1 does *not* raise there: it sets `constraint_1 = 5000` and returns a full curve, so
this is the single largest divergence from that version (measured: ours 0 vs redback 389 at n=500,
ours 0 vs 4817 at n=5000). Ours is the defensible answer — the envelope starts at 0.75 × `Rcirc/2`,
i.e. already inside the circularisation radius — but it is a behaviour change, not only crash
avoidance.

**An envelope that dies after its first sample still normalises the rise.** The envelope's own
light curve needs two live samples to exist (`_interp_photosphere`: one sample would otherwise be
a flat 400-day plateau), so `constraint == 1` is zero flux there. The *Gaussian rise* of
`gaussianrise_cooling_envelope` needs only its normaliser, the photosphere at the stitch — and for
a stitch at or before fallback (`xi ≤ 1`, the default) that is the first sample, the closed-form
initial envelope, which is redback's normaliser in every case (`photosphere_temperature[0]`).
Up to whisper 0.1.0 the port asked for two samples before it had a normaliser, so those draws were
`mag_floor` at every epoch. At `n_time=5000` that was rare; at redback 1.20's 500 points
([§6.1](#61-the-tde-grid-n_time-5000-vs-500)), the first forward-Euler step drives `Ee` through
zero on **4.8%** of `gaussianrise_cooling_envelope.prior` (4000 draws; none at 5000, where the same
draws live ~3100 steps), and the port sat up to 47.7 mag from redback 1.20 on epochs that all lie
on the rise (200 prior draws against redback). `_rise_normaliser` now takes the first sample there: the
rise equals the converged curve's and redback's, and after the stitch the envelope is over (zero
flux, as in [§4.2](#42-the-light-curve-stops-where-the-envelope-stops)).
`test_gaussian_rise_survives_an_envelope_whose_integration_dies_at_its_first_step`.

### 4.4 Why the input guard is safe

Guarding only `log(Lamb)` is not enough. When `Ee` crosses zero, `Rv = 2GM·Me/(5Ee)` goes to ±inf
and everything downstream of it — `Lamb`, `Rph`, `Teff`, `tacc`, `MdotBH` — inherits `inf`/`NaN`.
Measured on an unguarded implementation at `(mbh_6, m*, eta, alpha, beta) = (1, 1, 0.05, 0.1, 1)`:

| derivative | result |
|---|---|
| `d/d(params)` of `sum(L)` | finite |
| `d/d(params)` of `sum(T)` | NaN |
| `d/d(params)` of `sum(Rph)` | NaN |
| `d/d(params)` of `sum(Rv)` | NaN |
| `d/d(params)` of `sum(Me)` | NaN |

`L` is the exception precisely because it is the one output that never touches `Rv`:
`Lrad = Ledd40 + Edotfb`, and `Edotfb` depends on `Racc = ζ·Rv[0]·(t/tfb)^(2/3)` — the *initial*
radius only. So the natural smoke test is the one test that cannot see the bug, and on that same
test `d/dη`, `d/dα` and `d/dβ` come back exactly 0.0 because `Lrad` does not depend on them at all.
Three zeros and two finite numbers reads like a pass. Every photometric use of the model goes
through `T` and `Rph`, so an unguarded version is unusable with NUTS, HMC or any gradient-based
sampler.

The fix is the kilonova's double-`where` discipline: sanitise the *input* to the singular operation
so neither branch can produce a non-finite value, then select. `alive = (Ee > 0) & (Me > 0)` gates
the two divisions; where it holds, the arithmetic is textually identical to redback's and the
result is bit-identical. After the fix, autodiff agrees with central finite differences to 1.1e-4
(`T`), 9.7e-6 (`Rph`) and 7.2e-7 (`L`) on an interpolated light curve.

**A guard may move the singular point; it may not move the dynamics.** An intermediate
implementation violated that, and the failure was silent and severe: with `Ee` replaced by a
positive sentinel after death, `Rv` jumped to +1.1e23, `tacc ~ Rv²` exploded, `MdotBH` collapsed to
~0 and the envelope *grew* instead of draining, so `Me < 0` never fired and — through redback's
shared `try/except` — 6.55% of the prior box never terminated at all. It returned photosphere radii
to −5e24 cm and temperatures of 0.06 K inside the region it called valid, with gradients that were
finite, smooth, confident and wrong by 3–5 orders of magnitude and in sign. [§4.2](#42-the-light-curve-stops-where-the-envelope-stops)
is what makes the guard safe: the stopping test no longer reads anything downstream of the guarded
division, so guarding it cannot change where the curve ends.

### 4.5 Time dilation and `xi`: what the port keeps, against redback

**Decision: correct physics, one time dilation, is the default.** This is a deliberate, documented
deviation from redback (a project decision), and the double dilation has been reported upstream
as a redback bug. The port does not reproduce redback's behaviour and has no keyword that
does; a fit with the JAX cooling envelope and a fit with redback's own function at the same
parameters give light curves that differ by the factor `(1+z)` in time.

The port's default is unchanged in this release, and two redback behaviours stay as they were:
(c) **one time dilation**, where redback 1.20-1.21's `gaussianrise_cooling_envelope` dilates time
twice on both its `flux_density` and its band outputs (features at `(1+z)^2 t`: 5 % late at
z = 0.05, a factor 2 at z = 1), and (c) **the band branch's `xi` convention on both branches**,
where redback applies `xi` to the post-fall-back times only on its `flux_density` branch. The engine
`L` matches redback to 5.3e-15; `F_nu` at face value differs by a median 8.0e-3 (relative, z = 0.05),
and by 1.2e-5 with the second dilation emulated, which is how the parity tests compare (the port
evaluated at `t / (1+z)`). There is no `double_dilation=` keyword here (`tde_stream_stream` has one),
the golden does not vary `xi`, and the CPU redback adapter undoes redback's double dilation as well
([§9](#9-the-redback-cpu-adapter)).

---

## 5. The supernova family

**Why these port easily:** the engines are pure vectorised arithmetic — no ODE integration, no
sequential loop, no root-finding. The only control flow anywhere is two masks, in `shock_cooling`
and `fallback_lbol`. Compare the TDE cooling envelope, which needed `lax.scan` over a 5000-step
Euler integration. The consequence is that the whole family is cheap, batches perfectly, and is
differentiable end to end with no guard heroics.

### 5.1 The dense grid moves to setup time

Every bolometric model that passes through `ip.Diffusion` builds
`dense_times = np.geomspace(1e-5, time[-1] + 100, dense_resolution)` (1000 points) *inside the
likelihood* — redback 1.20 (`supernova_models.py:371, 961, 1461, 1664, 1687, 2402`; from 1.18). It
depends on the observation times, which are data rather than fitted parameters, so its shape is
fixed for a given dataset. `build_sn_grid` builds it once, together with the `uniq_times` /
`searchsorted` bookkeeping `ip.Diffusion` needs, and the models take that grid as their first
argument. Nothing about it varies during sampling, and the values are byte-identical to redback's.

**Which redback.** Up to 1.17 the grid was `np.linspace(0, time[-1] + 100, dense_resolution)`
(1.12.0 `supernova_models.py:390`, 1.15.1 `:251`): linear, so early times were resolved at ~0.3 d,
and starting at exactly 0, so `tb = max(0, min(dense_times))` was 0 rather than 1e-5. The default
here is 1.20's; `build_sn_grid(..., spacing="linear")`, or `**REDBACK_GRID_PRESETS["1.15"]` (and
`supernova_model(spacing="linear")`), reproduces the older one. `installed_redback_preset()` in the
redback adapter names the preset of whichever redback is installed, read from its source.

**Why it matters: the magnetar.** Every engine but one is smooth at `t = 0`, and on those the two
grids differ by ≤ 1.4e-4 mag. The dipole spin-down is not: it is a spike of height `2 E_rot / t_p`
and width `t_p`, and `t_p` spans decades inside the prior. When `t_p` is shorter than the linear
grid's first cell (0.28 d for ZTF20achncvv), the grid holds one sample of the spike, at `t = 0`,
and the trapezoid delivers `Δt/t_p` times `E_rot` — up to 2.3e5. The port on the linear grid sat up
to 21.5 mag above redback 1.20 at identical parameters, always too bright (200
prior draws: 18.3 mag max on ZTF epochs, 13.0 on LSST). On 1.20's grid the port equals redback
to float64 round-off at `t_p` from 1e-6 to 1e3 d (`test_magnetar_parity_with_the_jax_twin_across_spin_down_decades`).

**Matching redback does not make the fast spin-down right.** 1.20's grid never integrates
`[0, 1e-5 d] = [0, 0.864 s]`, and the spin-down releases `E_rot · x/(1+x)` before `t` with
`x = 2t/t_p`. At `t_p = 1.43 s` redback (and so the port) delivers 45% of `E_rot`; at 0.105 s,
5.7%. The grid fix turns "far too bright" into "quietly too faint", so it is not left quiet:
`warn_if_spin_down_unresolved` computes that fraction in closed form, and the supernova factory's
`predict` warns, once per model, when more than 10% of `E_rot` falls before the grid's first
resolved node (`dense_times[0]`, or `dense_times[1]` on the linear grid). It is host-side: under
`jit` the parameters are traced and there is nothing to test.

**That factory `predict` is the only place the warning runs.** A fit on a GPU sampler calls
`predict_jax`, which never checks, and the CPU redback adapter (`register_redback("basic_magnetar_powered")`,
`"slsn"`, ...) does not check either, although redback's grid has exactly the same limit. So a
magnetar fit on those paths is silent about it; check the posterior yourself with
`warn_if_spin_down_unresolved(dense_grid(t_src_days), p0, bp, mass_ns, theta_pb)`, where
`t_src_days = (t - t_exp) / (1 + z)`.

`exponential_powerlaw`'s first-interval deviation (CHANGE 3 in the source) is a property of the
linear grid only: on 1.20's grid both codes evaluate the formula at 1e-5 d and agree exactly.

### 5.2 Masked assignment becomes `jnp.where`

`lbol[time < td] = …` is an in-place write into a preallocated array, which JAX arrays do not
support. The *selection* condition is unchanged in every case, so each point takes the same branch
redback gives it.

### 5.3 The CSM shock breakout: redback's interpolation by default, the closed form on request

```python
# redback 1.20, shock_powered_models.py:544
time_temp = get_optimal_time_array(1e-2, 200, 300)   # == np.geomspace(1e-2, 200, 300)
func = interp1d(time_temp, lbol, fill_value='extrapolate')
return func(time)
# redback <= 1.15: time_temp = np.linspace(1e-2, 200, 300), 0.67 d spacing
```

`_csm_shock_breakout` is a closed-form expression — there is no integration to discretise — so the
300-node interpolation is pure error. Up to whisper 0.1.0 the port evaluated the closed form at the
epochs instead, and that was the one place it did not reproduce redback: against redback 1.20 it
was 0.12 mag apart at t ≥ 1 d and 0.42 mag below 1 d (200 prior draws:
0.059 mag max). The default now reproduces redback's interpolation, on the nodes of the preset in
use (geometric for 1.20, linear for ≤ 1.15, as for the diffusion grid), and
`build_sn_grid(..., csm_interp=False)` / `supernova_model(csm_interp=False)` restores the closed
form. **Outside the nodes, [0.01 d, 200 d], the closed form is used either way**: redback
extrapolates linearly there, and past 200 d that goes negative (from 201.09 d, where the light
curve decays exponentially) — so the port does not follow
it there.

What the interpolation costs, measured on redback ≤ 1.15's *linear* nodes over four parameter sets
spanning the shipped prior, on 100 epochs from 0.5 to 200 d:

| `csm_mass` | `v_min` | `beta` | `R_shell` | max \|Δmag\| | median \|Δmag\| |
|---|---|---|---|---|---|
| 1.0 | 1e4 | 0.30 | 1.0 | 1.150 | 0.008 |
| 0.1 | 5e3 | 0.45 | 0.1 | (L → 0) | — |
| 3.0 | 2e4 | 0.40 | 5.0 | 0.075 | 0.002 |
| 0.5 | 1e4 | 0.45 | 0.01 | 1.294 | 0.005 |

The maximum always lands at the *first* epoch, inside the first grid interval, where the breakout
luminosity varies by more than a factor of two across 0.67 d. 1.20's geometric nodes are dense
there, which is why its error is smaller (the 0.12 / 0.42 mag above), but it is still redback's
error, and a fit that exists to constrain the breakout should know it can switch it off.

### 5.4 Three redback models declare `ip.Diffusion` and then do not apply it

Measured, by sweeping a parameter that *only* the interaction process consumes and asking whether
the light curve moves at all (redback 1.15.1, `flux_density` branch):

| model | `d/dκ` | `d/dκ_γ` | `d/dmej` |
|---|---|---|---|
| `arnett` | 9.6e-1 | 1.7e+0 | 1.8e+1 |
| `sn_exponential_powerlaw` | 9.6e-1 | 3.1e+0 | 9.8e-1 |
| `magnetar_nickel` | 5.7e-1 | 1.6e+0 | 1.3e+0 |
| `sn_fallback` | **0.0** | **0.0** | **0.0** |
| `sn_nickel_fallback` | **0.0** | **0.0** | 1.8e+0 |

Exact zeros. `sn_fallback` and `sn_nickel_fallback` set
`kwargs["interaction_process"] = ip.Diffusion` and then never reach the block that would use it:
both branches call `fallback_lbol` and hand the raw engine straight to the photosphere. So `κ` and
`κ_γ` are **dead parameters** in redback's fallback fits — three of `sn_fallback`'s seven, since
`mej` is dead too — sampled from their priors, counted in every AIC/BIC, and constrained by
nothing. (`mej` survives in `sn_nickel_fallback` only because it also sets the nickel mass.)
`magnetar_nickel` has a milder version of the same defect: it diffuses in its `flux_density` branch
and *not* in its magnitude branch, so the same parameters give two different light curves depending
on the requested output format.

The size of the difference decides whether this can be treated as a detail. Applying diffusion to
`fallback_lbol` moves the light curve by:

| parameters | shift |
|---|---|
| `logl1=54, tr=1, κ=0.1, mej=2` | −7.65 … +0.79 mag |
| `logl1=52, tr=20, κ=0.5, mej=10` | −11.89 … +0.66 mag |
| `logl1=55, tr=0.1, κ=2.0, mej=50` | −15.59 … −2.11 mag |

It is a different model, not a correction. So it is a **switch**, not a silent choice:
`interaction=True` (the default) diffuses, matching every sibling model and redback's own declared
`interaction_process`; `interaction=False` reproduces redback bit for bit and leaves `κ`, `κ_γ` and
`mej` doing nothing. If you are comparing evidences against a published redback fallback fit, pass
`interaction=False` — and drop the dead parameters from the count while you are there.
`magnetar_nickel` here always diffuses, matching redback's `flux_density` branch.

### 5.5 The band integral is the real bandpass, not an sncosmo spline

redback's magnitude branch evaluates the SED on `np.geomspace(100, 60000, 100)` at
`np.geomspace(0.1, 3000, 300)` days, builds an sncosmo `TimeSeriesSource` and splines onto the
observations. All three JAX modules integrate the spectrum against the tabulated bandpass directly
*at* the observation times, making them rows of the diffusion kernel — no spline, no interpolation
error, and the kernel is `(n_obs, n_grid)` instead of `(n_grid, n_grid)`.

Expect small differences from redback's *magnitudes* traceable to the removed spline: 4.6–18 mmag
at worst, from its 300-node time grid. The flux-density path has no such step and is the
like-for-like comparison.

**One integral for both backends (whisper 0.1.1).** The integral itself lives in
`whisper_cbpf.synphot` ([`docs/PHOTOMETRY.md`](PHOTOMETRY.md)). By default every factory integrates
with **16 Gauss nodes per band**, adapted to `T dλ/λ` (`synphot.gauss_rule`), handed to the
unchanged kernels through `filter_set=` as a `{'lam', 'trans'}` dict (`FilterSet.to_legacy()`,
weights reproduced to 2.7e-16). It agrees with the 0.1.0 default — the grid rule on 2000 shared
points, `make_filter_set`/`ab_weights`, moved verbatim to `synphot.grid_rule` and re-exported here —
to ≤ 0.005 mmag, and makes JAX-on-CPU predict 1.6–2.5× faster. `n_wave=None` is the sentinel: pass
`n_wave=` to get the grid rule back. The CPU redback adapter integrates redback's own SED with the
same FilterSet (§9), so a CPU and a GPU model of the same physics now differ by the port's residual
only — 5e-15 mag for `arnett` — instead of the ~10 mmag the monochromatic CPU path added.

---

## 6. Which redback? Version differences that change numbers

"Agrees with redback" is ambiguous until the version is named. Three differences between redback
1.12.0 and 1.15.1 change numbers a fit would report; 1.20 (the latest) keeps 1.15.1's TDE grid,
magnetar and `(1+z)`, and changes the supernova, CSM breakout and kilonova grids
([§5.1](#51-the-dense-grid-moves-to-setup-time), [§5.3](#53-the-csm-shock-breakout-redbacks-interpolation-by-default-the-closed-form-on-request),
[§3](#3-numerical-identities-in-the-kilonova-core) identity 6). The TDE and supernova ports keep
preset tables keyed `"1.12"`, `"1.15"`, `"1.20"`, and
`whisper_cbpf.models.redback_adapter.installed_redback_preset()` names the key of the installed
redback by reading its source (without importing it), or `None` when redback is absent. The TDE's
default `n_time` follows it (the latest release when redback is absent); the supernova and kilonova
defaults are 1.20's whatever is installed.

### 6.1 The TDE grid: `n_time` 5000 vs 500

The `_cooling_envelope` loop bodies are identical — diffed line by line, only whitespace separates
them — but four things around the loop are not:

| 1.12.0 | 1.15.1 |
|---|---|
| `np.logspace(..., 5000)` | `np.logspace(..., 500)` — 10× coarser |
| no `f_debris` | `f_debris` / `calculate_f_debris` |
| `constraint = min(c1, c2)` | `if c1 == 0: c1 = 5000`; `constraint = max(min(c1, c2), 4)` |
| `tfb = calc_tfb(bec, mbh_6, m*)` | `tfb = calc_tfb(bec, mbh_6, m* * f_debris)` |

`f_debris = 1.0` (the default, and 1.15.1's own default) makes rows 2 and 4 agree, so only the grid
and the termination rule really differ. Both are reproducible: `REDBACK_PRESETS["1.12"]` and
`["1.15"]`, or the `n_time` argument directly.

**The grid is part of the model, not a tolerance** — this is a first-order ODE solve. Measured
against a converged *n* = 40000 (max relative error over the light curve):

| `n_time` | 500 | 1000 | 2500 | 5000 | 10000 | 20000 |
|---|---|---|---|---|---|---|
| `L` | 1.3e-4 | 2.5e-5 | 5.5e-6 | 1.6e-6 | 3.3e-7 | 7.2e-8 |
| `T` | 1.7e-2 | 7.9e-3 | 3.0e-3 | 1.4e-3 | 5.9e-4 | 2.0e-4 |

Error falls as O(1/n), as forward Euler must. redback 1.15.1's own default is **1.7% wrong in the
photosphere temperature** — a different model, not a cheaper one, and the change from 5000 to 500
does not appear in its changelog.

The termination time moves too. Across 200 prior draws,
`|t_term(n) − t_term(20000)| / t_term(20000)` has median 10.1% at n=500 (98.5% of draws above 1%,
50.3% above 10%), 1.6% at n=2500, and 0.60% at n=5000. One draw gives `constraint = 1` at every n
up to 10000 and 19039 at n=20000. The grid decides when the light curve ends, not only how accurate
it is while it lasts.

A second reason to prefer 5000, measured on the light curve rather than the engine: scanning
`mbh_6` over ±2e-6 relative, the AB magnitude at 99% of the curve spreads by 5.86 mag at n=500 and
by 3.67e-4 mag at n=5000 — and that is the *forgiving* measure. Following the epoch as it moves
understates the effect; hold the epoch fixed, as an observation does, and the same 1e-6 nudge moves
the magnitude by **24.52 mag**, the full `mag_floor` step, on all five engine parameters. The
mechanism is direct: the termination index moves +4 to +6 grid steps over ±2e-6 at n=500 and **0
steps** at n=5000. On a property sweep, 7 of 32 live prior draws show the full cliff at the coarse
grid. At the coarse grid the model is discontinuous in its own parameters at the 1e-6 scale, which
no sampler should be asked to work with.

**Default here: the installed redback's grid** (`tde.default_n_time()`): 500 for 1.15 and 1.20,
and when redback is not installed (the latest release); 5000 for 1.12. Up to whisper 0.1.0 the
default was 5000 whatever was installed, the more accurate of the two — and against redback 1.20
that put the port up to 19.6 mag away where the envelope ends at a different step (200
prior draws: 22.9 mag max). Pass `n_time=5000` for the finer grid
(`**REDBACK_PRESETS["1.12"]` reproduces redback 1.12 whole, `dilation=False` included), and know
what the coarse one costs: probed at fixed epochs under a ±1e-6 change of `mbh_6`
(`test_default_grid_stability_probe`, AB magnitude at 6e14 Hz), the magnitude at 500 points moves
by ≤ 8.8e-6 mag through 75% of the envelope's life and 7.9e-3 mag at 90%, then steps by the full
~24 mag `mag_floor` gap at 95-99%; the same probe at 5000 points gives ≤ 4.9e-5 mag through 95% and
0.31 mag at 99%. redback 1.20 itself, probed the same way, gives the same numbers through 75%,
1.4e-2 mag at 90% and the same 24 mag step at 95-99%: the default inherits redback's fragility and
adds none. The coarse grid also ends 4.8% of `gaussianrise_cooling_envelope.prior` after one Euler
step; [§4.3](#43-an-empty-envelope-is-a-mask-not-a-crash) says why the rise survives that.
Row 3 — the termination guards — is
reproduced by *neither* preset, because both versions of it are artefacts: 1.15.1's exist only to
stop `Lrad[:constraint]` returning an empty array and crashing on `time_temp[0]`, which
[§4.3](#43-an-empty-envelope-is-a-mask-not-a-crash) removed by returning a mask instead, and its
`constraint_1 = 5000` sentinel is stale on its own 500-point grid.

### 6.2 `basic_magnetar` differs by a factor of two, worth 0.75 mag

`E_rot` and `t_p` are identical in both versions; the luminosity is not:

| version | luminosity |
|---|---|
| 1.12.0 | `L = E_rot/t_p / (1 + t/t_p)²` |
| 1.15.1 | `L = 2 E_rot/t_p / (1 + 2 t/t_p)²` — and MOSFiT's magnetar engine |

Both integrate to exactly `E_rot`, so neither is losing energy; they are the same one-parameter
family with the spin-down time redefined by a factor of two. Measured over 0.1–300 d, the ratio
spans 0.514–1.954 at `(p0, bp, m_ns, θ) = (2, 1, 1.4, 1.0)` and 1.362–2.000 at `(5, 0.5, 2.0, 0.5)`
— i.e. −0.72 to +0.75 mag — and it is *not* a constant offset: the two curves cross, so it changes
the shape as well as the normalisation.

**Default here: `magnetar_convention="1.15"`**, because it is what MOSFiT's
`modules/engines/magnetar.py` computes and redback 1.15.1 changed *to* it, not away from it. Pass
`"1.12"` for the other. This affects `basic_magnetar_powered_bolometric`, `slsn_bolometric` and
`magnetar_nickel_bolometric`; `general_magnetar_slsn` uses `magnetar_only`, which is identical in
both.

### 6.3 The `(1+z)` flux factor

Both versions call `sed.blackbody_to_flux_density(temperature, r_photosphere, dl, frequency)` on
k-corrected inputs (`ν_src = ν_obs (1+z)`, `t_src = t_obs/(1+z)`), and that function contains no
redshift term in either. But 1.15.1's `cooling_envelope` and supernova models then return

```python
flux_density.to(uu.mJy).value * (1 + redshift)
```

where 1.12.0 returns the same expression without the factor, and 1.15.1's spectrum branch likewise
routes through a new `blackbody_to_spectrum(..., redshift=...)`.

**This is a relativity correction, not a convention.** Flux density is per unit *observed*
frequency, and observed frequency intervals are compressed by the expansion:
`dν_obs = dν_src/(1+z)`. So

```
F_ν(ν_obs) = (1 + z) L_ν([1+z] ν_obs) / (4π d_L²)
```

with `d_L` already absorbing the energy redshift and the time dilation. Drop the `(1+z)` and the
model is not using a different convention, it is missing a term. redback 1.15.1 added it;
`kilonova._flux_nu` has always had it.

**Default here: `dilation=True`.** Measured: with `dilation=False` the TDE module sits exactly
`4.7619e-2 = 1 − 1/1.05` below redback 1.15.1 at z = 0.05, at every epoch and every parameter set
tried, and with `dilation=True` it agrees. `dilation=False` exists only to reproduce redback 1.12.0
bit for bit, and reproducing it means reproducing the missing correction — worth
`2.5 log₁₀(1+z)` = 0.011 mag at z = 0.01, 0.053 at z = 0.05, 0.43 at z = 0.5. Use it to check a
port, not to fit.

---

## 7. Precision requirements

### The kilonova is float32-clean by construction

It has no accumulator, and it carries `L` in units of `LSCALE = 1e40` throughout. The identities in
[§3](#3-numerical-identities-in-the-kilonova-core) are what remove the remaining float32 hazards.

One hazard was not the model's: the clock. The factories take `t_exp_days=` on the light curve's
own clock, often MJD, and up to whisper 0.1.0 subtracted it inside the trace, in the model's dtype —
after the samplers' adapters had already cast the epochs to float32. Near MJD 58000 float32 resolves
2^-8 = 0.0039 d. Measured on the two-component kilonova (211 AT2017GFO-like epochs, 200 draws near
Villar's solution, float32 against float64 on days since t0): **7.3 mmag, a log-likelihood off by
5.2, and a median relative gradient error of 1.8e-2** on the raw-MJD clock, against 3.4e-6 mag, 0.010
and 2.3e-5 on days since t0 (7.2 mmag and 8.7 on AT2017GFO posterior draws). Since 0.1.1
`_factories._days_since` subtracts it on the host in float64, before any cast, and the adapters
(`samplers/jax/_adapters.py`) hand `predict_jax` the float64 numpy epochs: the raw-MJD clock now
gives 4.8e-6 mag, 0.010 and 2.7e-5 — the days clock's numbers. Only a caller that traces the epochs
themselves (vmapping over `times`) still subtracts in the trace. The TDE factory's `t_exp_days=`
(new in 0.1.1) takes the same path.

A third hazard was the import order. The quadrature nodes (`GL_X`, `GL_W`, `GL_XI`, `GL_WI`), the
Barnes–Kasen table (`_BK_*`) and the F99 spline (`_F99_*`) were `jnp` arrays built at import, so a
module imported before `jax_enable_x64` was switched on — as any script that imports first and
configures second does — froze them in float32, and a float64 session carried that into every
light curve: 3.6e-8 median and 1.9e-6 max relative error in `L_bol` (50 draws × 20 epochs), 1.1e-7
in the F99 law. Since 0.1.1 they are float64 NumPy, cast where they are used, so they take the
session's precision and the two import orders agree bit for bit
(`test_constants_do_not_freeze_at_the_import_precision`). The supernova's `_NXCS` had the same bug
(7.6e-8 in the cutoff SED) and the same fix.

### The TDE requires float64 — because of accumulation

Not preferred: required. The state variables are `Me ~ 2e32` g and `Ee ~ 5e13`, and each Euler
increment is ~1e-6 of the state, which is at the edge of float32's 1.2e-7 epsilon. The integration
is a sum of increments float32 cannot resolve. Measured in float32: `constraint` collapses to 1 and
the luminosity is `inf` for every case tried. There is no rescaling that fixes this, because the
problem is the ratio of increment to state, not the magnitude of either.

### The supernovae require float64 — because of dynamic range

A different reason. Nothing here accumulates, and rescaling would not help, because the problem is
plain dynamic range: these engines are bolometric luminosities in cgs, and float32 stops at
3.403e38.

| engine (redback's own prior medians) | value | float32 |
|---|---|---|
| `nickelcobalt(f_nickel=0.1, mej=2)` | 1.58e43 | inf |
| `nickelcobalt(f_nickel=1e-3, mej=1e-4)` | 7.90e36 | **inf** |
| `basic_magnetar(p0=2, bp=1)` | 1.77e46 | inf |
| `fallback_lbol(logl1=54, tr=1)` | 5.92e45 | inf |
| csm shock breakout `e0 = 0.5 M v0²` | 9.94e50 | inf |

Note the second row. 7.9e36 is comfortably inside float32, and it still overflows, because
`ni56_lum * np.exp(-t/τ)` = 6.45e43 is formed *before* the nickel mass multiplies it, and XLA is
free to reassociate the product — the same hazard as identity 4 in
[§3](#3-numerical-identities-in-the-kilonova-core). So there is no corner of the prior where
float32 works, and the failure is silent: `L` becomes `inf`, then 0 through the guards, and every
band comes back at `mag_floor`. A whole light curve of exactly 40.0 mag, with finite gradients.

Carrying `L` in `LSCALE` units here, as the kilonova does, would mean rescaling the engines, the
diffusion kernel and the photosphere — and it would end the property that every expression is
textually redback's, which is what makes this port checkable. The engines are cheap enough that
float64 costs little.

**Both families raise if `x64` is off, rather than returning a plausible wrong answer:**

```python
import jax; jax.config.update("jax_enable_x64", True)   # BEFORE the first array
```

The supernova SED layer *is* float32-clean and stays that way, because it never sees a bare
luminosity: `cutoff_norm` carries only O(1) quantities and the Line term is grouped through
`LINE_AB_CONST` so that neither `4π d_L²` (1.3e55) nor `L_bol` (1e43) is ever formed.

---

## 8. Performance, and what each family is good for

Measured on an A6000 at float64.

| model | one light curve | vmapped batch of 1024 | speedup |
|---|---|---|---|
| TDE `cooling_envelope` (5000 steps) | 38 ms | 42.5 ms total, 41.6 µs each | 926× |
| `arnett_bolometric` (100 epochs, 1000 grid points) | 0.279 ms | 10.0 ms total, 9.78 µs each | 29× |

The TDE's scan is sequential and launch-bound, so a single likelihood call is expensive and
**effectively requires a vectorised consumer** — ABC, SNPE, `emcee_jax` (which runs on the device:
its K ensembles' walkers are one batched evaluation per half-step inside one compiled loop), or NUTS
with `chain_method="vectorized"` over many chains. A plain sequential chain will be dominated by the
38 ms per call; that includes dynesty's nested sampling, which calls the likelihood one point at a
time, so on a GPU it gains only for the heaviest models.

The supernovae have nothing sequential: one call is already 136× cheaper, and there is far less to
win by batching. They are fast enough for a plain sequential NUTS chain and still batch well when
you want them. The dominant cost in a real supernova fit was the band integral (`n_obs × n_wave`,
159 of 187 µs per evaluation at `n_wave = 2000`), not the engine. Since whisper 0.1.1 the default is
16 Gauss nodes per band (§5.5), so it is `n_obs × 16 n_bands`, and the diffusion quadrature's fixed
100 nodes are no longer free by comparison.

**`tde_fallback` / `tde_mosfit` on multi-band light curves.** The viscous kernel runs once per distinct
epoch (`tde_fallback`'s deviation 6). The gain is at most the ratio of observations to distinct epochs:
the band integral, two gathers and a check stay per observation. Measured on one A6000, float64, on the
model-comparison light curves (35 epochs × 5 LSST bands: 175 observations, bucket 192, epoch bucket
48; free `t_exp`), `unique_epochs=True` against `False`, the two models alike:

- the log posterior through `log_density`, with its default epoch bucket (48): 2.0-2.4× faster at 16
  parameter sets, 2.9× at 64, 3.4× at 256 and 3.5× at 1024 (230 → 65 ms). `fit_batch` evaluates
  the same program with the distinct epochs padded exactly (35 slots here), which gains more: 4.4×
  at 256 sets;
- its value and gradient (`likelihood_max_opt`, NUTS): 1.7-4.4× at one set (a latency-bound
  program: the medians of interleaved runs move with the host load), 17.2-19.2× at 16 and 25× at 64
  (585-637 → 24-25 ms); through `make_log_prob_jax` 28× at 64;
- `make_batched_predict_jax` (ABC, SNPE): 1.8-2.7× at 32 sets, 4.0× at 1000 (chunks of 250);
- the XLA scratch memory 3.5-6× smaller: the value at 2048 sets 15.8 → 4.0 GB, `predict_jax`
  vmapped over 1024 sets 7.2 → 1.2 GB, the value and gradient at 16 sets 260 → 75 MB; the device's
  peak during a one-curve `fit_batch` 269 → 84 MB and during its `likelihood_max_opt` 1541 → 391 MB;
- `fit_batch`, 32 walkers × 5000 steps: one light curve 68-69 → 36-37 s at host load 10-12 and
  74.5-78.5 → 44.8-46.2 s at load 41-50, the call alone (from the process's start, imports and the
  model's build included, 80.9-87.5 → 51.0-54.6 s at that load); the sampling 41.2-41.7 → 12.4-12.7
  s; the compile, 16-18 s at load 10-12 and 22-26 s at 41-50, is unchanged. 64 light curves in one
  call 2665-2669 → 620-624 s (sampling 2293 → 492 s). These chains did not converge on these curves
  by the package's diagnostics (12.6 and 44 autocorrelation times long, a prior-edge pile-up,
  stuck walkers), with either code;
- `likelihood_max_opt` from the same fit: 2.1-3.2× faster (98-143 → 32-61 s at load 31-50). It is
  some 26 000-45 000 sequential evaluations, whose number varies from call to call, so its time
  follows the host load and the chain it starts from;
- compile seconds the same in every variant (2.1-5.1 s per program at load 10-12), the optimised
  program text 4-6 % longer.

These are unique epochs alone (`geometric=False`). The default also finds the viscous kernel's
segments by arithmetic (`geometric=True`, `tde_fallback`'s deviation 7), which adds 1.10-1.50x per
evaluation on these curves. In the run of the `fit_batch` figures above (host load 10-15 at the
start of its fits, the arithmetic search's included) it took one light curve from 36.4-37.1 s with
unique epochs alone to 32.7-34.1 s, and 64 light curves from 620-624 s to 449.6-454.9 s. In a later run (host load 13-28) unique epochs alone took 40.7 s
(`tde_fallback`) and 37.9 s (`tde_mosfit`), and the default 38.1 s and 34.6 s, the sampling 12.6 →
10.1 s and 12.5 → 10.0 s: the runs differ in host load, so compare figures within a run.

A light curve whose observations all have different times (one band per visit) gains nothing and pays
nothing measurable on the GPU: with every one of the 175 times distinct, the medians of the paired
per-round ratios before/after are 1.00-1.03x (the value at 256 sets; the value and gradient at 16 sets
for `tde_fallback`, at 1 set for `tde_mosfit`, 1.03x); on the shipped LSST alert (12 observations after the pre-event cut,
whose 12 distinct epochs fill its 16 slots, so the map saves nothing), the medians of interleaved
timings within 3 %, while single timings of its millisecond-scale programs scatter by up to 25 %
either way (one such timing gave 6.5 → 6.9 ms for `tde_fallback`'s value at 256 sets). On a CPU (4
cores, the two codes called alternately, host load 39-44), with every time distinct, host `predict`
and the one-set value and gradient within 1.3 %. Host `predict` on the CPU, one parameter set on the
175-point curve: 15.8 → 9.8 ms (`tde_fallback`) and 16.1 → 8.6 ms (`tde_mosfit`) at load 13, 2.8-2.9×
at load 41-44; the one-set value and gradient 4.3-4.4× there; the first call (its compile) 1.4-1.7 s
either way.

---

## 9. The redback CPU adapter

`whisper_cbpf.models.redback_adapter` is the other half of the story: instead of re-implementing a
model, it binds redback's own implementation to WHISPER's `Model` contract, on the CPU.

```python
from whisper_cbpf import register_redback

register_redback("arnett", redshift=0.0098)             # -> model "arnett_redback"
register_redback("cooling_envelope", redshift=0.05)     # -> model "cooling_envelope_redback"
register_redback("basic_magnetar_powered", redshift=0.1)
```

One module covers all 347 of redback's models, because redback ships them behind a single calling
convention. The only per-model facts are the parameter names and the prior, and redback ships both:
`redback_parameters` reads the first from the function signature plus the prior file, and
`redback_prior` reads the second from redback's own `.prior` file. Nothing is transcribed, so
nothing can drift.

A per-model file is reserved for the JAX *ports*, which are re-implementations and therefore have
per-model code to hold.

### The eight things the adapter must not get wrong

It maps WHISPER's `predict(parameters, times, bands) -> flux [Jy]` onto redback's
`fn(time, output_format="flux_density", frequency=..., **params) -> mJy`, integrated over each
observation's filter. That is all. Each of the following has a test.

1. **mJy → Jy is 1e-3** (`MJY_TO_JY`). redback's `flux_density` branch ends in `.to(uu.mJy).value`.
   A 1000× error reads as a plausible offset on a log-flux plot and lands entirely on the fitted
   masses, so the factor is one named constant, used once, and pinned by a test against redback's
   own return value on the identical call.

2. **One redback call for the whole light curve, with a per-point `frequency` array — never a
   per-band loop.** `arnett_bolometric` builds its diffusion grid as
   `geomspace(1e-5, time[-1] + 100, 1000)` (redback 1.20; `linspace(0, …)` up to 1.17), and the
   kilonova builds its grid from the extremes of the epochs: both depend on the time array they are
   handed. Splitting the
   curve by band would hand each band a *different* dense grid and silently change the physics.
   `test_one_call_not_a_per_band_loop` detects a regression by checking that a point's flux depends
   on the *other* bands' epochs, which it must.

3. **Out-of-domain epochs return zero flux; the rest of the band does not — and never silently.**
   redback interpolates its solution with `scipy.interp1d` and no `fill_value`, so one epoch past
   the model's own termination makes the *whole* call raise `ValueError` (redback 1.12.0's TDE path
   also raises `IndexError` when the solution is empty). Zeroing everything on any exception blanked
   an otherwise-real light curve on **9.83% of `cooling_envelope` prior draws** over AT2017GFO's
   212-epoch grid, where the JAX twin returned a real curve — a forward-model difference that would
   have surfaced later as a posterior difference. So `_accepted_span` recovers the epochs redback
   *will* answer for, by bisection over redback's own accept/reject, and only the rest becomes
   zero. It asks redback where its domain ends rather than knowing anything about a particular
   model's internals. Since whisper 0.1.1:

   - **any** exception raised inside redback's function refuses those epochs (only `ValueError` /
     `IndexError` were caught, so any other failure on one draw aborted the run); an exception in
     whisper's own band integral still propagates;
   - epochs redback refuses, or answers with NaN/inf, **warn once per model and span**, naming the
     epochs lost and the span kept, with redback's own message;
   - a limit **every draw shares is a configuration error and raises**. The first time a light
     curve loses epochs to an exception, `DOMAIN_PROBE_DRAWS = 8` prior draws are evaluated on it;
     if redback refuses the same epochs with the same message at all of them and at the draw that
     triggered it, no draw a sampler could propose escapes the limit. `interp1d` names the end of
     the grid it was built on, so a limit that moves with the parameters (an envelope's death)
     gives different messages and passes. `redback_model(..., times=lc.time)` runs the same probe
     at registration. The case it exists for: `two_component_kilonova_model` stops at 6 d source
     frame (`kilonova_models.py:1241`), so on AT2017GFO's 211 points to 12.5 d every
     draw lost the 18 past 5.49 d and every CPU sampler reached AIC ≈ 2.5e9 with no word; it now
     raises, naming the span and redback's `518400.0` s. (whisper's own `two_component_kilonova`
     sums two one-component calls and has no such ceiling.)

4. **Bands resolve through `whisper_cbpf.resolve_filter`, the map every model uses — so bare
   `u g r i z y` are LSST** (the bare-letter decision, whisper 0.1.1; one warning per session).
   redback's `tables/filters.csv` reads them as **SDSS** (`y` as PS1), and up to 0.1.0 this adapter
   passed them through while the two-component kilonova and the JAX factories read LSST: one label,
   two filters. `default_system="sdss"` (per model) or `whisper_cbpf.set_default_band_system("sdss")`
   gives redback's answer back; `band_aliases=` maps any label. A grouped label (`g-band`) raises in
   band mode — it names no curve — where 0.1.0 modelled it as SDSS g, also for ZTF g data.

5. **The band table and the filter sets are memoised.** `redback.utils.bands_to_frequency`
   re-reads a 264-row CSV on every call; the adapter reads it once per process. The band integral
   (`synphot.band_flux_jy`) expands every observation into its 16 nodes and calls redback ONCE on
   all of them, each observation's nodes contiguous and the last observation last, so redback's
   `time[-1]`-sized grids are the ones the monochromatic call builds (a row-order change moves results
   by up to 0.8 mmag); the expansion is memoised on the bands and the FilterSet hash.
   Cost: 1.0–1.3× the monochromatic call for most models; `gaussianrise_cooling_envelope` loops in
   Python per element and costs ~6×.

6. **redback's double time dilation is undone.** `gaussianrise_cooling_envelope`,
   `bpl_cooling_envelope` and `stream_stream_tde` convert the epochs to the source frame and then read
   them off a light curve built in the observer frame, so the curve evolves `(1+z)` too slowly (0.49
   mag at z = 0.062, 2.70 mag at z = 0.35). `redback_double_dilation(model)` detects the pattern in
   redback's source — exactly those three of 347 models in redback 1.20 — and the adapter then calls
   redback at `t (1+z)`: equal to the fine band integral of redback at `t (1+z)` to 1e-8 mag, and to
   the JAX port, which was always right, to 0.3 mmag median. It switches itself off in a redback that
   fixes either half of the pattern.

7. **A bolometric engine is refused.** redback's `*_bolometric` functions take
   `**kwargs`, so `output_format="flux_density"` and `frequency=` are swallowed and the return value
   is L_bol in erg/s: `shock_cooling_and_arnett_bolometric` bound as an 11-parameter model predicting
   a median 1.2e37 "Jy" (max 9.5e39), where its photometric wrapper predicts 1.8e-8 Jy. The
   builder now refuses, before reading the prior, any name ending in `_bolometric` whose signature
   has no `redshift` (all 55 in redback 1.20), naming the photometric wrapper when redback has one;
   then, on a prior draw, it evaluates `flux_density` at 4.0e14 and 6.5e14 Hz and refuses an output
   that is the same at both — which catches the 23 luminosity engines and unitless curves without
   the suffix (`basic_magnetar`, `general_magnetar`, `bazin_sne`, `villar_sne`, …).
   `test_redback_gate_lists_exactly_which_models_are_refused` loops over `all_models_dict` and pins
   both lists. `redshift=` now goes through the same membership check as `pin=`, where it used to be
   pinned unchecked and swallowed by `**kwargs`.

8. **redback's `Constraint` priors are a hard wall.** redback 1.20 keeps them in
   `redback.priors._constraint_settings` (a conversion function and `Constraint(lo, hi)` bounds per
   model, attached by `get_priors(model, constraint=True)`), not in the `.prior` files: Arnett
   nuclear burning ≥ kinetic energy, magnetar rotational ≥ kinetic energy (`slsn` also a 100–500 d
   nebular time), cooling-envelope `eta ≥ eta_min`, `beta ≤ beta_max` and, with a Gaussian rise, the
   stitch within 35 σ of the peak. whisper ≤ 0.1.0 dropped them, and the samplers explored the
   corners they exclude (6–13% of GPU TDE draws broke the `eta` bound; Arnett fits won through
   `f_nickel → 1`). `constraint=` on `redback_model` (default `"corrected"`) makes a draw that breaks
   one predict zero flux without calling redback. The decision is bilby's, strict `lo < v < hi`,
   from `whisper_cbpf.models.constraints`: a transcription of redback's formulas (the JAX samplers
   need it traceable) that makes **identical accept/reject decisions to redback on 20 000 prior
   draws per model, numpy and JAX**; for a model it does not transcribe (`csm_interaction`, the BNS
   ejecta relations, …) redback's own conversion function is called. `"corrected"` changes one
   number: redback bounds the Arnett kinetic energy by `-(14·2.4249 − 53.9037) MeV / m_p` =
   1.91e19 erg per gram of nickel, a sign slip on Δ(⁵⁶Ni) and one nucleon mass for 56 u; burning 14
   ⁴He to ⁵⁶Ni releases 87.85 MeV per 56 u, **1.51e18 erg/g**, 12.6× less. On redback's own arnett
   prior 62% of draws break the corrected bound and 31% redback's (200 prior draws). `"redback"`
   keeps redback's number; `None` is whisper ≤ 0.1.0. `redback_flux_jy` (one parameter set) applies
   no wall: it is the physics. The JAX supernova and TDE factories take the same `constraint=`: their
   host `predict`, which the CPU samplers call, gives zero flux for a draw that breaks it, exactly as
   this adapter does, so `abc`, `mcmc` or `nested` on a JAX model see the same wall; the JAX
   samplers' adapters apply `predict_jax.constraint_ok` (`-inf` log-density). `predict_jax` itself
   stays the physics.

### Conventions that are redback's, not ours

- **Band-integrated, from redback's `flux_density` branch.** redback's `magnitude` branch splines
  an sncosmo `TimeSeriesSource` off a coarse time grid (4.6–18 mmag error), so the adapter integrates
  the `flux_density` SED itself, with the FilterSet the JAX factories use: 16 Gauss nodes per band,
  ≤ 8e-6 mag from a 1 Å band integral of redback's SED on every family bar one row: `type_1a` in
  LSST g at z = 0.35 (2.6e-5 median, 3.6e-5 max), where redback's `CutoffBlackbody` puts its
  3000 Å rest-frame kink inside the band
  ([PHOTOMETRY.md §3](PHOTOMETRY.md#3-the-quadrature-16-gauss-nodes-per-band)). Up to whisper 0.1.0 it
  evaluated the SED at one reference frequency per band instead — 4–40 mmag median and up to 0.65
  mag off that integral on redback's own priors. That path stays available as
  `photometry="monochromatic"`, for the parity tests that compare single frequencies
  (`test_monochromatic_costs_more_than_the_port_does` pins the difference).
- **Distance is redback's cosmology.** redback models take `redshift` and derive
  `dl = cosmology.luminosity_distance(redshift).cgs.value` from their own default (Planck18); there
  is no `dl_cm` argument. A JAX twin takes one explicitly, so a comparison must be handed redback's:
  `redback_luminosity_distance_cm`. At z = 0.0098 that is 1.34991e26 cm; assuming 1.23e26 cm instead
  is 9.8% in distance, 20% in flux, 0.20 mag, all of it landing on `mej`.
- **The `(1+z)` flux dilation is a live version difference** ([§6.3](#63-the-1z-flux-factor); 0.0106
  mag at z = 0.0098). `redback_applies_dilation` reports which you have by *reading the model's
  source*, not by comparing version numbers — `redback.__version__` is `"unknown"` in some installs.

### Priors come from redback's `.prior` file

`redback_prior` translates `redback.priors.get_priors(model)`:

- bilby `Uniform` / `LogUniform` → WHISPER `Uniform` / `LogUniform`;
- bilby `DeltaFunction` → a **pinned** parameter (`redback_pinned`), which is what a delta prior
  means: redback fixed it. `type_1a.prior` pins its three line parameters in every release. Up to
  redback 1.15, `cooling_envelope.prior` pinned `mbh_6 = 1`, `eta = 0.1`, `alpha = 0.1` and
  `beta = 0.9`, leaving `stellar_mass` as the only free physics parameter; redback 1.18 (so 1.20)
  frees all five, with `mbh_6` LogUniform(0.1, 10) and `stellar_mass` LogUniform(0.5, 10). The
  adapter reads whichever file is installed (the JAX TDE factory without redback uses the latest
  release's, transcribed in `tde.fallback_prior`). Pass `prior={"mbh_6": LogUniform(0.1, 20), ...}`
  to unpin one with bounds you can defend;
- bilby `Constraint` → not a parameter, so not in the prior (redback 1.12's `slsn.prior` carries
  `e_rot_constraint` and `t_nebula_min`; 1.20 attaches them through `_constraint_settings`
  instead), and passing it to the model would be wrong. redback's constraints are applied as a wall
  by the model instead (point 8 above);
- anything else (`Sine`, `Gaussian`, …) has no WHISPER equivalent, so it is left out and
  `redback_model` raises, naming the parameter and the bilby type. Supply your own with `prior=` or
  fix it with `pin=`.

### Picklability

`predict` is `_RedbackPredict`, a **module-level callable class** whose whole state is plain data —
a model name, a list of names, a dict of floats. Not a closure, and not a `functools.partial` over a
per-model function: closures made the JAX photometric models fail under `abc`/`abc_smc` at
`n_jobs > 1` while the same fit at `n_jobs=1` succeeded. The caches — redback's function object and
the band table — are module-level, so an instance carries nothing unpicklable and needs no
`__getstate__`; each worker process fills its own on first call. The same reasoning drives
`models/jax/_factories._PhotometricPredict`.

redback is imported lazily, inside the functions, so `import whisper_cbpf` works without the
`[models]` extra and only calling `predict` needs it.

---

## 10. TiDE (CPU): calling libtide, not porting it

`whisper_cbpf.models.tide` runs the C++ library TiDE (Kovacs-Stermeczky & Vinko 2023) through tidepy. It is not a JAX port: the
owner's decision, taken from the measured cost (`reports/tide_cpu_cost.md`: 0.27 ms per observation, 17 ms for a 60-point light
curve, near-linear scaling to 16 processes, ~95 % of the time in the disc ring integral).

**What is checked.** `tests/goldens/tide_oracle.npz` holds the unmodified library's output for 59 parameter sets (the anchor at both
beta defaults, six variants of the library's options, 48 Latin-hypercube draws of the prior box, two box corners): `L_nu` of the
wind and the disc at the six notebook wavelengths, `t_min`, bolometric luminosities, the wind photosphere. The model's flux agrees
with it to 4e-14 (relative, 9,500 points, library objects reused in a shuffled order); L_nu, t_min, bolometric and photosphere
quantities to better than 1e-12. The notebook anchor peaks at 37.68 d with 3.734e44 erg/s.

**Deviations from the original, with the rule of section 2 each satisfies.**
1. (a) One `Parameters` object per process and configuration, refreshed in place (`param_set_params`, 0.3 ms) instead of a new object
   per parameter set (32 ms, 10.5 MB, never freed). Identity: 4e-14 against fresh objects.
2. (a) A numeric `fout` is changed with `param_set_allparams` using the calculation names `"default"` (which leaves the fall-back
   calculation and its table alone; 0.7 ms, against 32 ms and +11 MB). 4e-14.
3. (b) An epoch at or before `t_min (1 + 1e-4)` is 0 Jy, not the library's non-physical tiny number (issue 26), as in the notebook.
4. (b) A parameter set outside the library's domain (non-finite, non-positive mass or efficiency, `fout` outside [0, 1],
   `fv < 1`) is zero flux at every epoch.
5. (c) `beta_fulldisrupt` is a pinned parameter, default 1.85 (tidepy's, the notebook's): libtide's own default is 1.0 and moves
   `t_min` by 1.85^3 (issue 20). Never overridden silently.
6. (c) `Mdotfb_calc` and a numeric `fout` are honoured as given. `tidepy.Parameters()` ignores `Mdotfb_calc = "default"` and a numeric
   `fout` set after construction (issues 20 and 21); the model builds its objects the notebook's way (`Parameters.__new__`). The
   oracle uses the same route for those sets and the standard route for the 'bifout' L09 ones (they are identical: 0.0).
7. (c) Bands are monochromatic at the filter's effective wavelength by default (the notebook's convention); the other whisper models
   integrate the passband. Against a 16-node integral: 0.4 to 2.1 % in flux (4 to 21 mmag) for SDSS ugi and LSST z. `n_nodes=16`
   restores the integral at 16x the cost.
8. Not offered: diffusion (NaN at the first epoch, stale timescale: issue 22), `d` and `i` (ignored by the library: issue 24).
   `N` (disc rings) is part of the configuration because libtide reads it when the disc object is created (issue 23).

**What is not tested here.** The wind photosphere and bolometric luminosity are the library's through a blackbody inversion of its
`L_nu` (residual at a third frequency 1.5e-13); libtide's internals (the wind radius options other than "rph", the reprocessing
module, spectra) are not exercised.
