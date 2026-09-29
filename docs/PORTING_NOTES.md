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
package) and `_tde_mosfit_engine` (ragged `dtype=object` tables loaded from disk; only its final
scaling and interpolation could move).

### Supernovae — `models/jax/supernova.py`

Twelve model names over nine engines: `_nickelcobalt_engine`, `basic_magnetar`, `magnetar_only`,
`exponential_powerlaw`, `fallback_lbol`, `_shock_cooling` (Piro+ 2021) and `_csm_shock_breakout`
(Margalit 2022), assembled into `arnett`, `shock_cooling_and_arnett`, `basic_magnetar_powered`
(= `slsn`), `magnetar_nickel`, `csm_shock_and_arnett`, `exponential_powerlaw`, `sn_fallback`,
`sn_nickel_fallback`, `general_magnetar_slsn`, `type_1a` (= `arnett`) and `type_1c` (= `arnett`).
Four SED kinds: blackbody, cutoff blackbody, line, synchrotron.

**Not ported:** the `homologous_expansion` / `thin_shell` / `csm_interaction` (Chatzopoulos) family,
the Morag and Sapir–Waxman shock-cooling variants, and `general_magnetar_driven_supernova`. Each
needs machinery this module does not have — a root-find, a table, or an ODE. They are separate
ports, not omissions of convenience.

### Reuse, not copying

The TDE and supernova modules import the blackbody, the AB weights and the filter export from
`kilonova.py`, exactly as redback shares `interaction_processes.py`, `photosphere.py` and `sed.py`
across model families. Copying them would silently reintroduce every numerical defect
[§3](#3-numerical-identities-in-the-kilonova-core) removes, the next time the original is fixed.
Nothing is shared for its *physics* — what is shared is arithmetic.

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
**effectively requires a vectorised consumer** — ABC, SNPE, or NUTS with
`chain_method="vectorized"` over many chains. A plain sequential chain will be dominated by the
38 ms per call.

The supernovae have nothing sequential: one call is already 136× cheaper, and there is far less to
win by batching. They are fast enough for a plain sequential NUTS chain and still batch well when
you want them. The dominant cost in a real supernova fit was the band integral (`n_obs × n_wave`,
159 of 187 µs per evaluation at `n_wave = 2000`), not the engine. Since whisper 0.1.1 the default is
16 Gauss nodes per band (§5.5), so it is `n_obs × 16 n_bands`, and the diffusion quadrature's fixed
100 nodes are no longer free by comparison.

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
