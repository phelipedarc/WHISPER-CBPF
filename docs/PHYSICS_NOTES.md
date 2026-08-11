# PHYSICS_NOTES — deviation ledger

One row per place the JAX code is **not arithmetically identical** to the reference
(redback; the installed 1.15.1 in `phe_sbi_2` unless a row says 1.12.0). Every impact
number is *measured* by a test or an audit script in `_scratch/`, not estimated.
Empty is the goal; honest is the requirement.

Project rules governing this ledger (user directives, 2026-08-08):
- **Stay as close as possible to the redback implementation.** Deviations exist only where
  JAX makes identity impossible (shapes, both-branch `where`, chaotic indices) or where the
  user approved a flag.
- **Priors are never changed.** They are read from `redback.priors.get_priors()` at run
  time, delta functions included.

## Kilonova (`models/kilonova.py`, `kilonova_two.py`)

| ID | What differs | Why | Max impact | Test that bounds it | Approved |
|----|--------------|-----|-----------|---------------------|----------|
| K1 | Diffusion quadrature: s-substitution + log-t′ Gauss–Legendre replaces redback's 300-pt geomspace trapezoid (identity 6) | redback under-resolved past `t ≈ 2.66 t_diff` (its own grid refined to 2×10⁵ pts converges onto ours, ratio 7.37→1.00004) | agrees with adaptive `scipy.quad` to **1.78e-5**; vs redback-at-defaults diverges up to ~11× in L beyond the horizon, redback the bright outlier | `test_kilonova_vs_redback.py::test_documented_divergence…` | pre-existing design |
| K2 | Physics evaluated *at* observation times; no sncosmo `TimeSeriesSource` spline; exact filter-cell band integral | removes interpolation/aliasing error | band mags agree with a 4×10⁵-pt sncosmo integration to **0.023 mmag**; vs redback's splined path, mmag-level | `test_kilonova_vs_redback.py`, audit `aud_0*` | pre-existing design |
| K3 | Identities 1,4,5,7,8: algebraically exact regroupings for f32/gradient safety | JAX reverse pass overflows where NumPy's forward pass does not | forward f64 shifted ≤ ~1 ulp (documented per identity) | audit `aud_12_final.py` | pre-existing design |
| K4 | `arnett_prefactor` flag. **Default 1.0 = redback**, whose diffusion kernel integrates to ½ (measured `L/L_in → 0.500055` at 100 t_diff where Arnett/Chatzopoulos/Villar → 1). `2.0` selects the standard normalisation | the reference is wrong (uniform **0.7526 mag** too faint; T low by 2^0.25, R_floor low by √2) but §8.4 applies: do not fix the reference in the port | 0 by default (bitwise, compared at trace time); 2× in L when enabled | `test_kilonova_vs_redback.py::test_arnett_prefactor…` | **user, 2026-08-08: "keep as redback, expose a flag"** |
| K5 | Barnes–Kasen bilinear table uses a strict `>` count | reproduces scipy RGI values exactly; derivative is one-sided **at** nodes | value C⁰ everywhere; derivative kink **up to 162.5 %** (T3; an earlier row said 31–46 % from a smaller probe) — but **bitwise C¹ at the vej = 0.4 edge node**, so "it kinks at every node" is false | `t3_inference/test_continuity.py` | inherited from reference |

## TDE (`models/tde.py`)

| ID | What differs | Why | Max impact | Test that bounds it | Approved |
|----|--------------|-----|-----------|---------------------|----------|
| D1 | Termination: `stop = (Ee≤0)|(Me≤0)|(Rv<Rcirc/2)` on one trajectory, replacing redback's `min(first(Rv<Rcirc/2), first(Me<0))` + shared try/except | redback's index is read off a **diverged** trajectory: redback vs itself under a 1-ulp β nudge differs on 10 % of draws (max 4–12 steps); a port built to reproduce the index missed by up to 1793 | constraint exact 81.9–85.3 % (vs redback-self 89.3–93.5 %), max 5/14 steps at n=500/5000; termination-time p99 1.2–5.0 % | `test_tde_vs_redback.py::test_termination_is_the_models_own_criterion`, `_scratch/constraint_conditioning.py` | forced by JAX + measurement |
| D2 | `valid`/`meaningful` masks replace the data-dependent slice; `constraint == 0` → zero flux where redback 1.15.1 returns a **full curve** (its `constraint_1 = 5000` sentinel) | shapes must be static under jit; the envelope is born inside R_circ/2 on **0.431 %** of the prior (closed form: `β·(R_t/R★) < 20`, predicate `envelope_exists`) | largest single divergence from installed redback: 0 pts vs 389–4817 pts on 0.43 % of the prior | `t…::test_envelope_exists_predicts…`, `…short_lived_envelope…` | forced (jit) + physical rule |
| D3 | Zero flux outside the envelope's own time span; `R_phot` clamped ≥ 0 before interpolation | redback: flat/extrapolated curves past termination (a `constraint==1` draw produced a flat 400-day curve at AB 20.5 on **9.85 %** of the prior at n=500); negative R_phot squared into a blackbody on **3.6 %** of draws (≤ 3.89 mag local) | a genuine cliff in t and θ that `jax.grad` cannot see — documented; keep data inside the span | `…::test_short_lived…`, `…::test_zero_flux_outside…`, `…::test_negative_photosphere…` | physical rule, user-endorsed 2026-08-08 |
| D4 | Exact bandpass integral vs redback's 100-pt SED → sncosmo spline | same choice as K2 | envelope mags: median 0.02–3.9 mmag, **max 19.2 mmag**; gaussianrise ≤ 100 mmag at the final epoch | `benchmarks/tde_vs_redback/compare.py` (figures) | pre-existing design |
| D5 | Interpolate (T, R) onto observation epochs, then integrate the band — redback's own `cooling_envelope` convention (its `gaussianrise` integrates then interpolates) | 12.5× memory/compute; and measured *more* accurate vs a cubic reference in 15/17 cases | ≤ 7.2e-3 mag at n=5000 (3-day cadence 3.8e-3); up to **0.596 mag at n=500** — one more reason not to fit at `REDBACK_PRESETS["1.15"]` | verifier round 3, `_scratch/v*` | convention chosen from redback itself |
| D6 | Gaussian rise evaluated analytically in magnitude (a parabola), normalisation cancelled on paper | redback samples 100 nodes and interpolates (error 0.0128 mag at σ_t=2 d) and its normalising ratio underflows → NaN mags *and* grads on 0.27 % of the prior at z=0.5 | exact to **5.4e-13 mag** vs 60-digit mpmath; NaN rate now 0 in 2.88 M evaluations | verifier round 3; `…::test_gaussian_rise_is_continuous…` | forced (NaN grads) |
| D7 | Flux-branch rise exponent clamped at 700 (saturates ≈1e304) | flux cannot represent exp(931) that magnitude represents as −800 mag; unclamped: +inf forward, NaN/−inf grads on 0.31 % of prior at z=0.5 | monotone saturation in a regime no observation reaches | `…::test_gaussianrise_flux_density_never_overflows` | forced (overflow) |
| D8 | `dilation` flag; **default True** | the (1+z) is a relativity correction (`dν_obs = dν_src/(1+z)`); 1.12.0 is missing the term, 1.15.1 has it | `False` reproduces 1.12.0 exactly (ratio ≡ 1/(1+z), ≤1e-10); worth 0.011–0.43 mag at z = 0.01–0.5 | `…::test_dilation_is_exactly_one_plus_z` | user framing 2026-08-08 |
| D9 | `lax.scan(..., unroll=16)` | 5.2× speed; scan is launch-bound (13.7–17.4 µs/step, flat in n and batch) | **not bit-identical**: constraint moves on 5.0 %/10.0 % of draws (≤3/≤12 steps), photometric p99 **5.9e-6 mag** at n=500 and 3.6e-15 at 5000, confined to the ill-conditioned tail (≤4e-14 over the first half). *An earlier row said 8.7 %/18.7 % and p99 5.0e-3 mag — a smaller sample, pessimistic by three orders on the photometry* | `t2_autodiff/test_gradients.py` | speed, within D1's noise floor |
| D10 | `jnp.interp` (clamps) vs `interp1d` (raises) / `fill_value='extrapolate'` | jit cannot raise on data; extrapolating a terminated model invents a light curve | subsumed by D3's zero-outside-span rule | `…::test_zero_flux_outside…` | forced (jit) |
| D11 | `csm_diffusion`: exponentials folded so the exponent ≤ 0; `aspherical_diffusion`: `(2+x)/(1+eˣ)` → `(2+x)e⁻ˣ/(e⁻ˣ+1)` (exact identity) | redback's CSM form overflows f64 once `t_max/t0 ≳ 700` (reachable at 0.01 M☉ of CSM at 10¹⁵ cm); the aspherical form survives only because inf lands in a denominator | agrees with redback to ≤ 5.3e-14 wherever redback is finite; finite where redback is not | `…::test_csm_diffusion_does_not_overflow…`; verifier round 1 | exact identity |
| D12 | `build_interaction_grid` clips out-of-range epochs with a `RuntimeWarning` | redback raises IndexError | saturated value instead of a crash, warned | `…::test_build_interaction_grid_warns…` | quality-of-life, warned |
| D13 | `gaussianrise` follows the `cooling_envelope` frame convention in **both** branches | redback's two branches disagree with each other: its flux branch queries an observer-frame interpolator at source-frame times (**×397 ≡ 6.5 mag at z=0.5**), and its magnitude branch feeds source-frame times to `cooling_envelope` (self-inconsistency 0.125 mag at z=0.5) | vs redback's magnitude branch: rise ≤ 0.002 mag; envelope region up to **0.20 mag at z=0.5** — entirely attributable to redback's internal inconsistency (whisper matches `rb(t·(1+z))` to ≤1e-3) | verifier round 3 (A6/A6b) | §8.4: reference bug documented, not reproduced |

## Gradient-side properties (T2/T3, 2026-08-09)

Not deviations from redback — redback has no gradients — but properties a gradient sampler
meets and the forward-side ledger above does not describe.

| ID | Property | Measure | Consequence | Test |
|----|----------|---------|-------------|------|
| G1 | **Zero-gradient plateau.** Where `constraint < 2` the flux is exactly 0 at every epoch, so ∂/∂θ is **exactly 0 in all five engine parameters** | **6.45 %** of redback's own prior at `n_time=500`; **empty** at 5000 (512 draws); 3.1 % of `tde_model(rise="gaussian", n_time=500)`'s prior | a flat region HMC can neither escape nor detect the edge of. `envelope_exists()` bounds only the closed-form 0.431 % | `t2_autodiff/test_gradients.py` |
| G2 | **The zero-flux cliff moves with the parameters** and AD is blind to it (D3, gradient side) | epoch held fixed: **24.52 mag** under a 1e-6 relative nudge on any of the five, at n=500; termination index moves +4…+6 steps, **0** at n=5000; 7/32 live prior draws | a likelihood straddling the cliff is discontinuous in a way HMC cannot see | `t3_inference/test_continuity.py` |
| G3 | `Model.predict` is **not differentiable** — both factories cast with `float(...)` and return NumPy | `jax.grad` raises `ConcretizationTypeError`; the two paths are pinned equal (kilonova rtol < 1e-12, TDE jitted-vs-eager 4.2e-15) | gradient samplers must call the module functions, not `predict` | `t2_autodiff/test_gradients.py` |
| G4 | **NUTS does not converge over redback's untruncated shipped prior** | **0 of 12** fits; median r-hat **2.55**, median divergence rate **50 %**. Injection #5 recovered 7/7 truths in its 90 % CI, but every fit has r-hat ≥ 1.83 | truncate the prior before any production fit; see HANDOFF | `t3_inference/stage_sbc.py`, `stage_injection_recovery.py` |

## Reference defects found, NOT fixed in the ports (§8.4 register)

| Where | Defect | Status |
|-------|--------|--------|
| redback kilonova diffusion | kernel integrates to ½, uniform 0.753 mag | reproduced by default; `arnett_prefactor=2.0` flag (K4) |
| redback `photosphere.DenseCore` | `mask_2 & mask_3 & mask_4` ANDs booleans with an integer index array; raises unless peak index is 0, else writes rows 0–1 | **not ported**; documented |
| redback `gaussianrise_cooling_envelope` flux branch | frame error, ×397 at z=0.5 | not reproduced (D13) |
| redback `gaussianrise` vs its own `cooling_envelope` | source-frame times into an observer-frame API, 0.125 mag at z=0.5 | not reproduced (D13) |
| redback 1.15.1 `cooling_envelope` magnitude branch | `TypeError` whenever the user passes `lambda_array` | not applicable here |
| redback 1.12.0 `cooling_envelope.prior` | self-overriding file → 4 of 5 parameters are deltas | **reproduced faithfully** via `get_priors()` (not a deviation) |
| redback 1.12.0 flux | missing (1+z) | reproducible via `dilation=False` (D8) |

## Version-difference register (not deviations: pick with `REDBACK_PRESETS`)

`n_time` 5000 (1.12.0) vs 500 (1.15.1) — 1.7 % in T, termination time median 10 % off at 500,
parameter-discontinuity 24.52 mag at 1e-6 relative, epoch held fixed (T3); `f_debris` (1.15.1 only, default 1.0 ≡ 1.12.0);
`(1+z)` (D8); termination guards (subsumed by D1). **Do not fit at `REDBACK_PRESETS["1.15"]`.**
