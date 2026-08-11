"""T3.1 -- prior-predictive sweep: the model must survive its OWN prior.

20,000 draws from each model's own prior are pushed through the band-magnitude path --
the exact code a sampler evaluates -- and every value and a 512-draw gradient subsample
must come back finite. This is the cheapest possible statement of "the sampler will not
step on a NaN", and it is a statement about the PRIOR BOX, not about nice parameters:
the draws go where the sampler goes, including the corners.

What is asserted, and what is only reported:

* Kilonova (4 free parameters: mej, vej, kappa, temperature_floor -- the factory's own
  redback-verbatim prior): every magnitude finite and in (5, 40]; every gradient
  component of a scalar loss finite on the 512-draw subsample. The fraction of
  (draw, epoch) pairs sitting AT the mag_floor cap is reported -- the cap is documented
  behaviour (kilonova.py MAG_FLOOR block), not a failure.

* TDE gaussianrise (7 free parameters, redback 1.15.1's shipped prior read from redback):
  every magnitude finite and <= mag_floor; gradients finite on the 512-draw subsample.
  THREE documented pathologies of the prior box are REPORTED as fractions rather than
  failed on, because the model's own docstring predicts each one:
    - ``constraint < 2``: the envelope never lives; the whole curve is mag_floor
      (tde.py CHANGE 2 / _interp_photosphere -- ~9.85% of this prior at n_time=500).
    - epochs past the envelope's termination time: the DOCUMENTED zero-flux region
      (tde.py CHANGE 3 / _interp_photosphere: "outside the model's own time span there
      is no model").
    - the absurd-bright rise: draws whose stitch point sits > 3 sigma past the Gaussian
      peak genuinely predict magnitudes far brighter than 5 (measured -887 at z=0.05 over
      120k draws in tde.py). ``rise_peaks_near_fallback`` is the model's own predicate
      for that regime, so the (5, 40] assertion is applied to the draws it accepts, and
      the regime's size and brightest magnitude are reported for the rest. Anything
      brighter than 5 on an ACCEPTED draw is still a failure.

Both tests also print a PRIOR-USABILITY block (:func:`usability_report`), which is
description, not judgement: the fraction of draws and of (draw, epoch) pairs that come
back non-finite, EXACTLY zero, or outside a magnitude window any real survey could record
-- plus, for the TDE, the distribution of the light curve's own observer-frame span
against three campaign lengths. The project rule is that priors are never modified, so the
point of these numbers is to tell whoever designs a fit how much of the prior box produces
something an instrument could have seen.

Budget: 20,000 draws (not 1e6) because this runs on CPU; the TDE engine alone is
~0.2 ms/draw at n_time=500 and the photometry ~1 ms/draw at n_wave=1000.
"""
from __future__ import annotations

import time

import numpy as np

import _t3_common as C  # noqa: E402  (flips x64 BEFORE jax arrays exist)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

N_DRAWS = 20_000
N_GRAD = 512
SEED = 20260808
MAG_FLOOR = 40.0
CHUNK_VAL = 256          # ab_magnitude builds an (n_obs, n_wave) spectrum per draw
CHUNK_GRAD = 64


#: What a ground-based optical survey can actually record. Bright end: ZTF saturates near
#: 12.5 and LSST near 16, so 14 is a generous middle; faint end: 27 is deeper than any
#: wide-field survey reaches in a single visit. A draw outside this window is not "wrong",
#: it is a light curve NO instrument would ever have in its data -- which is a statement
#: about how much of the prior box is doing useful work.
OBSERVABLE_MAG = (14.0, 27.0)
#: Observer-frame campaign lengths a TDE light curve is compared against, days.
CAMPAIGN_DAYS = (30.0, 60.0, 180.0)


def _finite_report(name, arr):
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"  {name}: {arr.size} values, NaN={n_nan}, inf={n_inf}, "
          f"min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}")
    return n_nan, n_inf


def usability_report(label, flux, mags):
    """The four prior-usability fractions, per DRAW and per (draw, epoch).

    DESCRIPTIVE, NOT A GATE. Nothing here is asserted: the question is not "is the prior
    correct" (it is redback's, read at run time, and the project rule is that priors are
    never changed) but "what does a sampler that respects it actually spend its time on".

    ``flux`` and ``mags`` are (n_draws, n_epochs) from the SAME parameter matrix, so
    "exactly zero" is read off the flux -- where the model's own zero-outside-the-span rule
    writes it -- rather than inferred from a magnitude hitting the cap, which the cap can
    also produce for a merely very faint source.
    """
    n_draws, n_ep = mags.shape
    nonfinite_ep = ~np.isfinite(mags) | ~np.isfinite(flux)
    zero_ep = flux == 0.0
    lo, hi = OBSERVABLE_MAG
    out_ep = np.isfinite(mags) & ((mags < lo) | (mags > hi))
    stats = dict(
        frac_nonfinite_ep=float(np.mean(nonfinite_ep)),
        frac_nonfinite_draws=float(np.mean(nonfinite_ep.any(axis=1))),
        frac_zero_ep=float(np.mean(zero_ep)),
        frac_zero_curves=float(np.mean(zero_ep.all(axis=1))),
        frac_out_ep=float(np.mean(out_ep)),
        frac_out_curves=float(np.mean(out_ep.all(axis=1))),
        frac_too_bright_ep=float(np.mean(np.isfinite(mags) & (mags < lo))),
        frac_too_faint_ep=float(np.mean(np.isfinite(mags) & (mags > hi))),
    )
    print(f"\n  --- {label}: prior usability over {n_draws} draws x {n_ep} epochs")
    print(f"  non-finite flux or magnitude:        "
          f"{stats['frac_nonfinite_ep']:.4%} of epochs, "
          f"{stats['frac_nonfinite_draws']:.4%} of draws (any epoch)")
    print(f"  EXACTLY zero flux:                   "
          f"{stats['frac_zero_ep']:.4%} of epochs, "
          f"{stats['frac_zero_curves']:.4%} of draws are zero at EVERY epoch")
    print(f"  outside the observable window {lo}-{hi} mag: "
          f"{stats['frac_out_ep']:.4%} of epochs "
          f"({stats['frac_too_bright_ep']:.4%} too bright, "
          f"{stats['frac_too_faint_ep']:.4%} too faint); "
          f"{stats['frac_out_curves']:.4%} of draws have NO observable epoch")
    return stats


# =======================================================================================
def test_kilonova_prior_predictive():
    """20,000 draws from the kilonova factory's own prior, through the band-magnitude path.

    WHAT IT WOULD CATCH. A NaN or inf anywhere in the prior box -- forward OR in the
    gradient. That is the single failure that stops a sampler dead, and it is invisible to
    any test that evaluates "reasonable" parameters: the corners of a LogUniform(100, 6000)
    temperature floor crossed with vej = 0.5 and kappa = 30 are where the Barnes-Kasen
    extrapolation drives the exponent d past 1.8 and the heating term changes sign
    (kilonova.py). It would also catch a broken distance/zero-point path, which shows up
    as magnitudes brighter than 5 at z = 0.05 -- physically impossible for this source
    class, so it is asserted rather than reported.
    """
    from whisper_cbpf.models.jax import kilonova as kn
    from whisper_cbpf.models.jax import kilonova_model

    lam, W, N, fs = C.filters()
    # 30 epochs, 0.5-25 d, both ZTF bands at every epoch -> 60 observations
    t_obs, band_idx = C.epochs_two_bands(0.5, 25.0, 30)
    t_src = jnp.asarray(kn.source_time_s(t_obs, C.Z))
    bidx = jnp.asarray(band_idx)

    # THE MODEL'S OWN PRIOR: taken from the factory, not retyped here, so this test keeps
    # tracking whatever the factory ships. 4 parameters (temperature_floor left free).
    model = kilonova_model(C.BANDS, C.Z, C.DL_CM, n_wave=C.N_WAVE, filter_set=fs)
    names = list(model.parameters)
    assert names == ["mej", "vej", "kappa", "temperature_floor"], names
    rng = np.random.default_rng(SEED)
    theta = C.sample_prior_matrix(model.default_prior, names, N_DRAWS, rng)

    def mags_fn(th):
        return kn.ab_magnitude(t_src, bidx, W, N, lam, C.Z, C.DL_CM,
                               th[0], th[1], th[2], th[3], mag_floor=MAG_FLOOR)

    t0 = time.perf_counter()
    mags = C.chunked_vmap(mags_fn, theta, chunk=CHUNK_VAL, desc="kilonova mags")
    print(f"kilonova: {N_DRAWS} draws x {t_obs.size} obs in {time.perf_counter()-t0:.1f} s")

    n_nan, n_inf = _finite_report("magnitudes", mags)
    frac_cap = float(np.mean(mags >= MAG_FLOOR - 1e-9))
    print(f"  fraction of (draw, epoch) AT the mag_floor cap: {frac_cap:.4%}")
    # The kilonova has NO termination and NO zero-flux rule -- its only exact zero is the
    # pre-explosion window (kilonova.pre_explosion, t_exp = 0 here so no epoch is in it), so
    # the "exactly zero" line below is expected to read 0.0000% and would be a finding if it
    # did not. Flux is taken from the model's own magnitude output, in AB units of 3631 Jy.
    flux = np.power(10.0, -0.4 * mags)
    usability_report("kilonova one-component", flux, mags)

    # gradient of a scalar loss on a 512-draw subsample
    sub = theta[rng.choice(N_DRAWS, N_GRAD, replace=False)]
    grad_fn = jax.grad(lambda th: jnp.mean(mags_fn(th)))
    t0 = time.perf_counter()
    grads = C.chunked_vmap(grad_fn, sub, chunk=CHUNK_GRAD)
    print(f"  {N_GRAD} gradients in {time.perf_counter()-t0:.1f} s")
    g_nan, g_inf = _finite_report("gradients", grads)

    assert n_nan == 0 and n_inf == 0, f"kilonova magnitudes: {n_nan} NaN, {n_inf} inf"
    assert g_nan == 0 and g_inf == 0, f"kilonova gradients: {g_nan} NaN, {g_inf} inf"
    assert mags.max() <= MAG_FLOOR + 1e-9, f"cap violated: {mags.max()}"
    assert mags.min() > 5.0, (
        f"kilonova magnitude {mags.min():.3f} <= 5: brighter than any transient at "
        f"z={C.Z} has a right to be -- check the distance/zero-point plumbing")


# =======================================================================================
def test_tde_gaussianrise_prior_predictive():
    """20,000 draws from redback 1.15.1's own 7-parameter gaussianrise prior.

    WHAT IT WOULD CATCH. (a) NaN/inf in the forward pass or the gradient through the
    500-step scan -- the regime tde.py CHANGES 6-7 exist to remove (measured before the
    fixes: 0.27 % NaN magnitudes AND gradients, 0.31 % +inf flux). A single reappearance
    stops NUTS. (b) A bright-end regression: magnitudes brighter than 5 are asserted to
    occur ONLY on draws the model's own ``rise_peaks_near_fallback`` predicate rejects, so
    a numerical overflow leaking into the sane region cannot hide inside the (large,
    documented) absurd-rise population. (c) The sweep going blind: ``frac_dead`` and
    ``frac_past`` are asserted NONZERO, because a prior-predictive that never reaches the
    documented pathologies is not covering the prior it claims to cover.
    """
    from whisper_cbpf.models.jax import tde as T

    lam, W, N, _ = C.filters()
    # 24 epochs, 1-400 d from the light-curve origin, both bands -> 48 observations
    t_obs, band_idx = C.epochs_two_bands(1.0, 400.0, 24)
    t = jnp.asarray(t_obs)
    bidx = jnp.asarray(band_idx)
    t_obs_s = jnp.asarray(t_obs * C.DAY)

    preset = dict(T.REDBACK_PRESETS["1.15"])       # n_time=500, dilation=True
    n_time = preset["n_time"]

    # redback 1.15.1's own 7-parameter prior, read from redback (not transcribed).
    #
    # WHICH PRIOR, AND WHY NOT THE OTHER ONE. `T.default_prior()` -- the bare
    # `cooling_envelope` prior -- is ALSO read from redback, and redback's own file for it is
    # self-overriding, so 4 of its 5 parameters come back as delta functions (PHYSICS_NOTES,
    # reference-defect register: "reproduced faithfully via get_priors()"). Sweeping it would
    # sweep one axis and call it a prior-predictive check. The 7-parameter gaussianrise prior
    # is the one the fitted model actually ships with, and the one swept here; the pinned set
    # of the other is printed so the difference is on the record rather than assumed.
    bare_prior, bare_pinned = T.default_prior()
    print(f"  NOTE tde.default_prior('cooling_envelope') pins {sorted(bare_pinned)} "
          f"(redback's own self-overriding prior file); free: {sorted(bare_prior.names)}. "
          f"This sweep uses the 7-parameter gaussianrise prior instead.")
    prior, pinned = T.default_prior_gaussianrise()
    assert not pinned, f"gaussianrise prior unexpectedly pins {pinned}"
    names = list(T.PARAMETERS_GAUSSIANRISE)
    assert sorted(names) == sorted(prior.names), (names, prior.names)
    rng = np.random.default_rng(SEED + 1)
    theta = C.sample_prior_matrix(prior, names, N_DRAWS, rng)

    def mags_fn(th):
        return T.gaussianrise_cooling_envelope_ab_magnitude(
            t, bidx, W, N, lam, C.Z, C.DL_CM,
            th[0], th[1], th[2], th[3], th[4], th[5], th[6],
            mag_floor=MAG_FLOOR, **preset)

    def flux_fn(th):
        return T.gaussianrise_cooling_envelope_flux_density(
            t, jnp.where(bidx == 0, 6.32e14, 4.79e14), C.Z, C.DL_CM,
            th[0], th[1], th[2], th[3], th[4], th[5], th[6], **preset)

    def diag_fn(th):
        out = T.cooling_envelope(th[2], th[3], th[4], th[5], th[6], n_time=n_time)
        k = out["constraint"]
        tt_last = out["time_temp"][jnp.maximum(k - 1, 0)] * (1.0 + C.Z)
        alive = k >= 2
        n_past = jnp.sum((t_obs_s > tt_last) & alive)
        # observer-frame span of the whole light curve, measured from its own origin
        # (t = 0 is the start of the Gaussian rise, so the span INCLUDES the rise)
        span_days = jnp.where(alive, tt_last / C.DAY, 0.0)
        return k, n_past, span_days

    t0 = time.perf_counter()
    mags = C.chunked_vmap(mags_fn, theta, chunk=CHUNK_VAL, desc="tde mags")
    print(f"tde: {N_DRAWS} draws x {t_obs.size} obs in {time.perf_counter()-t0:.1f} s")
    constraint, n_past, span_days = C.chunked_vmap(diag_fn, theta, chunk=1024)
    t0 = time.perf_counter()
    flux = C.chunked_vmap(flux_fn, theta, chunk=1024, desc="tde flux")
    print(f"tde flux path: {N_DRAWS} draws in {time.perf_counter()-t0:.1f} s")

    n_nan, n_inf = _finite_report("magnitudes", mags)
    _finite_report("flux density (mJy)", flux)

    # --- the three DOCUMENTED fractions: reported, not failed on -----------------------
    dead = constraint < 2
    frac_dead = float(np.mean(dead))
    alive = ~dead
    frac_past = float(n_past[alive].sum() / (alive.sum() * t_obs.size))
    frac_cap = float(np.mean(mags >= MAG_FLOOR - 1e-9))
    sane = np.asarray(T.rise_peaks_near_fallback(theta[:, 0], theta[:, 1],
                                                 theta[:, 2], theta[:, 3]))
    frac_absurd = float(np.mean(~sane))
    print(f"  fraction of draws with constraint < 2 (envelope never lives): {frac_dead:.4%}")
    print(f"  fraction of epochs past termination, among live draws:        {frac_past:.4%}")
    print(f"  fraction of (draw, epoch) AT the mag_floor cap:               {frac_cap:.4%}")
    print(f"  fraction of draws in the absurd-bright-rise regime (>3 sigma): {frac_absurd:.4%}")
    if (~sane).any():
        m_abs = mags[~sane]
        print(f"    brightest magnitude in that regime: {m_abs.min():.1f} "
              f"(documented: reaches -887 at z=0.05; not a numerical artefact)")
        print(f"    fraction of its epochs brighter than mag 5: "
              f"{float(np.mean(m_abs < 5.0)):.4%}")

    usability_report("TDE gaussianrise", flux, mags)

    # --- HOW LONG IS THE LIGHT CURVE? --------------------------------------------------
    # The single most consequential property of this prior for fit design: the model's own
    # span is what decides whether a real campaign's epochs fall inside it, and the
    # zero-flux-outside-the-span rule (D3) turns every epoch beyond it into a mag_floor
    # cliff. Reported against three campaign lengths; DESCRIPTIVE.
    live = constraint >= 2
    sp = span_days[live]
    pct = np.percentile(sp, [1, 5, 25, 50, 75, 95, 99])
    print(f"\n  --- TDE light-curve span (observer frame, from the light-curve origin), "
          f"{int(live.sum())} live draws")
    print(f"  percentiles  1%: {pct[0]:.2f}  5%: {pct[1]:.2f}  25%: {pct[2]:.2f}  "
          f"50%: {pct[3]:.2f}  75%: {pct[4]:.2f}  95%: {pct[5]:.1f}  99%: {pct[6]:.1f} d")
    for camp in CAMPAIGN_DAYS:
        f_short = float(np.mean(span_days < camp))       # dead draws count as span 0
        print(f"  fraction of ALL draws whose light curve is shorter than a "
              f"{camp:5.0f}-day campaign: {f_short:.4%}")

    # gradients on a 512-draw subsample, ALL SEVEN parameters
    sub = theta[rng.choice(N_DRAWS, N_GRAD, replace=False)]
    grad_fn = jax.grad(lambda th: jnp.mean(mags_fn(th)))
    t0 = time.perf_counter()
    grads = C.chunked_vmap(grad_fn, sub, chunk=CHUNK_GRAD)
    print(f"  {N_GRAD} gradients (7 params, through the 500-step scan) "
          f"in {time.perf_counter()-t0:.1f} s")
    g_nan, g_inf = _finite_report("gradients", grads)

    assert n_nan == 0 and n_inf == 0, f"tde magnitudes: {n_nan} NaN, {n_inf} inf"
    assert g_nan == 0 and g_inf == 0, f"tde gradients: {g_nan} NaN, {g_inf} inf"
    assert mags.max() <= MAG_FLOOR + 1e-9, f"cap violated: {mags.max()}"
    ok = sane[:, None] & np.ones_like(mags, dtype=bool)
    assert mags[ok].min() > 5.0, (
        f"magnitude {mags[ok].min():.3f} <= 5 on a draw whose rise peaks within 3 sigma "
        f"of its stitch point -- OUTSIDE the documented absurd-rise regime, so this is "
        f"a finding, not the known corner")
    # the documented fractions must be nonzero, or this sweep is not exercising the
    # regions it claims to cover
    assert frac_dead > 0.0, "no dead-envelope draws in 20k: sweep is not covering the prior"
    assert frac_past > 0.0, "no past-termination epochs in 20k: sweep is not covering the prior"


# =======================================================================================
if __name__ == "__main__":
    for fn in (test_kilonova_prior_predictive, test_tde_gaussianrise_prior_predictive):
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"=== {fn.__name__}: PASS ===")
