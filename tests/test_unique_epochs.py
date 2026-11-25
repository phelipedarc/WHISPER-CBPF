"""Each distinct epoch evaluated once: ``tde_fallback`` and ``tde_mosfit`` with an epoch map.

The claims, each a test (numbers measured on a CPU, x64, 35 epochs x 5 LSST bands). The model
under test is the factory's default: the epoch map and the arithmetic segment search
(``geometric=True``, the default since it was measured; claim 13). The reference is the same
factory call with ``unique_epochs=False, geometric=False``, the per-observation code before these
options existed (claim 13 shows it is that code); claim 2 holds the search fixed to isolate the map:

1. **The map is the light curve's.** ``epoch_index(times, size)`` gives int32 ``(first, inv)`` with
   ``times[first][inv] == times`` (NaN-aware), ``first`` in increasing time and padded by repeating
   its last entry, over 200 random arrays (repeats, unsorted, +-0, NaN, +-inf); a 2-D input, a size
   below the distinct count and padding an empty input raise ``ValueError``; its docstring example
   runs, and so do ``log_density``'s (the fall-back TDE example gives 36 observations in bucket 48,
   12 distinct epochs padded to 16). ``log_density`` pads the map with the light curve: the pad
   rows join the latest epoch.
2. **The map changes nothing but rounding**: with the identity map and with the distinct-epoch map
   (padded to 35, 48 and 175 slots) the physics is the per-observation physics to 1e-12 mag, over
   200 prior draws x 175 observations: at most 3.2e-14 mag (``tde_fallback``, 98.9 % bitwise; the
   identity map 7.11e-15, 99.1 %) and 3.2e-14 (``tde_mosfit``, 99.9 %; the identity map bitwise).
   The per-observation code itself moves by up to 6.4e-14 and 8.6e-14 mag between a 35-row and a
   175-row call (82 % and 75 % bitwise): that is XLA's shape-dependent rounding, not the map.
3. **Every call path agrees with the reference** to 1e-12 mag and 1e-12 of the size of the summed
   terms in the log density (fixed and free ``t_exp`` and redshift): host ``predict``,
   ``jit(vmap(predict_jax))`` on numpy epochs, traced epochs without a map (bitwise the code of
   ``unique_epochs=False, geometric=True``, which it is, and bitwise the reference on these draws)
   and with one, ``make_log_prob_jax`` (likelihood and posterior), ``make_batched_predict_jax``,
   ``log_density`` (``fn`` and ``log_likelihood``, another light curve's data),
   ``likelihood_max_opt``'s programs, ``profile``'s, and the closed-over program: magnitudes at
   most 5.7e-14 (``tde_fallback``) and 7.9e-14 mag (``tde_mosfit``; host ``predict`` with free
   ``t_exp`` and redshift 2.8e-13), 72-100 % bitwise, host ``predict`` 98.4-99.2 % bitwise (the
   segment search's round-off in its one-set programs, claim 13); log densities within 2.0e-15 of
   the size of their summed terms, through ``log_density`` at least 98 % (``tde_fallback``) and
   99.5 % (``tde_mosfit``) bitwise. With every epoch distinct host ``predict`` takes the plain core:
   bitwise that of ``unique_epochs=False, geometric=True``, and here the reference's.
4. **Every call path uses the map**: the viscous kernel's arrays have E rows (the distinct count on
   concrete paths, the epoch bucket on ``log_density`` / ``likelihood_max_opt``, the exact count on
   ``profile`` and the batch's largest in ``fit_batch``'s stacked program) on each path above, and
   the observation count (or its bucket) with ``unique_epochs=False``.
5. **Gradients agree to round-off**: ``value_and_grad`` of the log posterior gives the same values,
   bit for bit, and gradients within 3.1e-13 (``tde_fallback``) and 1.03e-13 (``tde_mosfit``) of
   each draw's largest component (48 draws, fixed and free redshift); ``jax.grad`` of
   ``make_log_prob_jax`` within 2.0e-13 and 1.8e-13 (free ``t_exp`` and redshift). With respect to
   traced epochs given a map, the derivative is per distinct epoch, carried by its first
   observation; the per-epoch sums agree to 1.9e-13 and 4.5e-11 of the largest. Not to 1e-12: in
   the epoch of the largest difference the rows do not cancel (their terms all have one sign, so
   the sum of their sizes is the size of their sum); the difference arises inside that epoch's
   backward pass.
6. **Edge cases and random light curves** (unsorted, same-band repeats, rows at and before the time
   zero, +-0, a NaN time, every row before the time zero, one epoch in five bands, one row, no
   repeats, upper limits in flux space, the shipped LSST alert, and seeded random curves with free
   ``t_exp`` and redshift): within the claim-3 tolerance; magnitudes at most 7.11e-15 mag (96-100 %
   bitwise), every row at or before the time zero (and the NaN time) at ``mag_floor``; the alert's
   log posterior 80 % and 85 % bitwise at a relative 1.7e-16; 29 of the 30 random curves bitwise,
   the other at 1.7e-16.
7. **fit_batch stacks light curves with different epoch counts** (35, 28, 21 distinct; one width
   192; epochs padded to 35) with every start form (``"prior"``, ``"prior_scan"``, explicit
   starts): a toy model with the hook gives the chains of the same toy without it, bit for bit; a
   permuted batch gives each member the same chain; a member alone (its own 28 epochs) re-scores
   the batch's draws to a relative 1.1e-15 (53-94 % bitwise). With the two models (``slow``: 16
   walkers, 24 steps, every start form) the per-observation density re-scores the kept draws to
   the chains' own log posterior within 8.3e-15 relative.
8. **Compiles**: one ``shared`` program for every light curve of a model; XLA compiles per (bucket,
   epoch bucket) (cache sizes 1, 1, 2 for epoch buckets 48, 48, 64); an explicit bucket keeps one
   compile and stackable data; the band integral keeps its barrier on repeated epochs; the
   closed-over program carries no map (it agrees with the data path to 2.3e-16); no compiled
   constant grows with the batch (21 275 entries at 4 and 64 sets); the program text grows by 2.3 %
   (``tde_fallback``) and 2.2 % (``tde_mosfit``) and the scratch memory of the 64-set log-density
   program drops from 593 MB to 128 MB; the host core compiles once per shape;
   ``likelihood_max_opt`` reuses its programs only for the same epoch bucket; on a GPU (one A6000,
   ``slow``, skipped without a GPU) the 480 x 480 batch and ``log_density``'s ``value_and_grad`` at
   480 sets compile, after the default flipped (host load 10-11), in 3.4 s and 8.2 s
   (``tde_fallback``) and 2.6 s and 5.4 s (``tde_mosfit``) with ``geometric=False``, and in 3.0 s
   and 5.7 s, 2.8 s and 5.2 s with ``geometric=True``; over that run and two earlier ones, 2.6-5.4 s
   and 5.2-11.5 s, every one under 60 s. Neither setting compiles faster in every run:
   ``tde_mosfit``'s batch took 2.6 -> 2.8 s and 3.3 -> 4.0 s with ``geometric=True`` in two of them.
9. **A wrong map is loud, a right one is quiet**: a map from another light curve's epochs or a
   scrambled one gives NaN in every draw, a same-pattern map gives the right numbers; a concrete
   wrong map, a model built without the option (through its context's ``predict_jax``; the plain
   ``predict_jax`` such a model carries raises Python's ``TypeError``), a hook without the keyword,
   ``unique_epochs=True`` for physics without ``epoch_idx`` and an ``epoch_bucket`` out of range or
   not ``"auto"`` raise ``ValueError`` with the fix; a right map raises nothing under
   ``jax_debug_nans``, jitted or op by op, and a wrong one raises ``FloatingPointError`` there
   (``slow``: op by op, every operation compiles alone, 46-53 s).
10. **Nothing else moves**: the other families and ``unique_epochs=False, geometric=False`` models
    keep a plain ``predict_jax`` with no hook, no new state (no ``geometric`` key) and their
    docstring, get no data keys, and receive no ``epoch_idx``.
11. **Pickling and worker processes keep the hook**: a pickled model keeps ``epoch_index`` and its
    fluxes, bit for bit; one ``spawn`` worker (the CPU samplers' ``n_jobs`` path) builds the
    per-epoch core and returns the parent's fluxes bit for bit (both models, 6.8-10.5 s with its
    start);
    an object pickled before the options existed (no ``unique_epochs`` on its context, no
    ``geometric`` in its state, a plain ``predict_jax``) takes the per-observation path with the
    search, bit for bit the reference's.
12. **float32 still raises the float64 error** on every new path: ``predict``, a jitted
    ``predict_jax`` with a map, ``log_density``'s data path, ``likelihood_max_opt``'s programs, and
    the closed-over program of an MJD clock when it is called; ``epoch_index`` (numpy) works.
13. (``geometric=True``, the default) **The arithmetic segment search is the interpolant**:
    against the search, the viscous kernel's log10 L is bitwise over 200 prior draws x 48 epochs
    (its gradient within 1.4e-14 of the largest component); magnitudes over 200 draws x 175
    observations bitwise, with and without the map; the default model's host ``predict`` with
    free ``t_exp`` and redshift against ``geometric=False`` 99.0 % (``tde_fallback``) and 99.2 %
    (``tde_mosfit``) bitwise, at most 5.0e-14 and 2.8e-13 mag; through ``log_density``'s vmapped
    ``value_and_grad`` the log posterior bitwise and its gradient within 2.9e-15 (48 draws, fixed
    and free). **The defaults** (``inspect.signature``, the docstrings and
    ``docs/API_REFERENCE.md``): both factories ``unique_epochs=True, geometric=True``; the
    functions that build their own dense grid (``tde_fallback_bolometric_log10``,
    ``tde_fallback_ab_magnitude``, ``tde_mosfit_curve``, ``tde_mosfit_ab_magnitude``,
    ``tde_mosfit_constraint_score``, ``tde_mosfit_flux_density_mjy``) ``geometric=True``, and they
    find the segments by arithmetic, as the default factories do; ``viscous_log10``, which takes the
    caller's grid, ``geometric=False``: on a linear 1000-node grid ``geometric=True`` is wrong, its
    largest error per draw 0.25 to 3.8 dex over 20 draws (median 0.41 dex over the 20 x 48 values),
    while its default is the search to the bit. The constraint wall, one function for every model,
    takes no ``geometric`` and keeps the search, to the bit, as do these functions and the factories
    given ``geometric=False``. ``geometric=True`` with fewer than 2 dense nodes raises
    ``ValueError`` naming the fix, at the factory call and in ``viscous_log10`` (with one node the
    arithmetic search gave NaN for every observation), while ``geometric=False`` with one node
    still gives finite numbers. ``geometric`` enters a model's state only when True:
    ``unique_epochs=False, geometric=False`` has the state and the ``model_record`` identity of an
    object from before the options (the default model with both removed) and its numbers bit for
    bit on the CPU (host ``predict``, ``jit(vmap(predict_jax))``, the physics called with
    ``geometric=False``; the same jaxpr, with no ``floor``), and the default model's identity
    differs from that of ``geometric=False``.

FLOAT64: the fixture enables it for the module, calling ``jax.clear_caches()`` on entry and exit,
and restores it afterwards.

TOLERANCES: 1e-12 mag, 1e-12 of the size of the summed terms of a log density and 1e-10 of a
gradient's largest component are the package's padding standard (``tests/test_log_density.py``),
not a budget of 3-10 x the numbers measured here. The round-off depends on the light curve: over 200
random light curves (both models, 16 prior draws each, CPU) the magnitudes reached 3.2e-13 mag and
the gradients 2.7e-12, 4 x and 9 x the largest measured in this file, so a budget fitted to these
seeds would hold for these seeds only (and ``tde_jax120``'s redback prior draws other parameters).
The time derivative's per-epoch sums (claim 5) are budgeted on their measurement, 9 x, and the
linear-grid error of claim 13 from below, at a fifth of the smallest per-draw error measured; the
GPU compile bound of claim 8 (60 s) is ``test_compile_barrier.py``'s.
"""
from __future__ import annotations

import contextlib
import doctest
import functools
import importlib
import io
import re
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.models.cosmology import luminosity_distance_cm  # noqa: E402
from whisper_cbpf.models.jax import _adopted_tde as AT  # noqa: E402
from whisper_cbpf.models.jax._factories import _BandIndex, _PredictJax  # noqa: E402
from whisper_cbpf.priors import Prior, Uniform  # noqa: E402
from whisper_cbpf.samplers.jax import _adapters as A  # noqa: E402
from whisper_cbpf.samplers.jax import _diagnostics as DG  # noqa: E402
from whisper_cbpf.samplers.jax import batch as B  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

KINDS = ("tde_fallback", "tde_mosfit")
CONFIGS = ("fixed", "free")
BANDS = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz"]
Z = 0.05
DL = float(luminosity_distance_cm(Z))
T0 = 59988.0                         # the synthetic curves' time zero, on an MJD clock
SEED = 20261120
N_KERNEL = 1000                      # the viscous kernel's nodes (500 + 500) and the dense grid
TOL_MAG = 1e-12
TOL_LOGD = 1e-12
TOL_GRAD = 1e-10
#: The time derivative's per-epoch sums (claim 5), relative to the largest: 9 x the measured.
TOL_TIME_DERIVATIVE = 4e-10
#: Claim 13: on a linear grid ``viscous_log10(geometric=True)`` is wrong in every draw by at least
#: this much (dex): a fifth of the smallest per-draw error measured (0.247 dex over 20 draws).
MIN_LINEAR_GRID_ERROR_DEX = 0.05
BLOCK = 50
#: The free config's priors: t_exp before the first epoch of every curve, z around the truth.
FREE_PRIOR = {"t_exp": (T0 - 40.0, T0 + 1.0), "redshift": (0.02, 0.2)}
TRUTH = {
    "tde_fallback": dict(mbh6=1.5, mstar=1.0, tvisc=2.0, bb=1.0, eta=0.1, rph0=6.31, lphoto=1.5),
    "tde_mosfit": dict(bhmass=1.5e6, starmass=1.0, b=1.0, efficiency=0.1, Tviscous=2.0, Rph0=6.31,
                       lphoto=1.5),
}
#: (first, last, number) of the epochs, in days after T0, each observed in the five bands.
CURVES = {"LC35": (3.0, 300.0, 35), "LC35b": (4.0, 280.0, 35), "LC28": (2.0, 200.0, 28),
          "LC21": (4.0, 250.0, 21)}


@pytest.fixture(scope="module", autouse=True)
def _x64():
    was = jax.config.jax_enable_x64
    jax.clear_caches()
    jax.config.update("jax_enable_x64", True)
    yield
    jax.clear_caches()
    jax.config.update("jax_enable_x64", was)


class _Stop(Exception):
    """Raised by a spy once it has what it needs, so nothing is compiled or run past it."""


@contextlib.contextmanager
def _patched(obj, attr, value):
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


def _measured(label, **numbers):
    """Print a measured number in a form the docstring's claims are filled from (pytest -s)."""
    print("MEASURED " + label + ": " + ", ".join(f"{k}={v}" for k, v in numbers.items()))


# --- models, light curves, draws -------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _model(kind, config="fixed", unique_epochs=True, tag="", t_exp_days=T0, geometric=None):
    """The factory call; ``tag`` gives an object of its own (its own program cache).

    ``geometric=None``: with ``unique_epochs=True`` no ``geometric`` keyword reaches the factory,
    so the model is the factory's default (the map and the arithmetic segment search); with
    ``unique_epochs=False`` the factory gets ``geometric=False``: the per-observation reference,
    the code before these options existed. A bool is passed as given (claim 13)."""
    fac = {"tde_fallback": wp.tde_fallback_model, "tde_mosfit": wp.tde_mosfit_model}[kind]
    if geometric is None and not unique_epochs:
        geometric = False
    geo_tag = {None: "", False: "_search", True: "_geo"}[geometric]
    name = f"{kind}_{config}_{'ue' if unique_epochs else 'obs'}{geo_tag}{tag}"
    geo = {} if geometric is None else {"geometric": bool(geometric)}
    if config == "fixed":
        return fac(BANDS, Z, DL, name=name, t_exp_days=t_exp_days, unique_epochs=unique_epochs,
                   **geo)
    prior = Prior({k: Uniform(*v) for k, v in FREE_PRIOR.items()})
    return fac(BANDS, name=name, free=["t_exp", "redshift"], prior=prior,
               unique_epochs=unique_epochs, **geo)


def _grid(name):
    lo, hi, n = CURVES[name]
    tau = np.geomspace(lo, hi, n)
    return np.repeat(T0 + tau, len(BANDS)), np.tile(np.array(BANDS), n)


def _observe(kind, t, b, seed, t_exp_days=T0, upper=None):
    """A light curve of ``kind`` at its truth, with seeded noise (5 % plus 1 % of the peak)."""
    m = _model(kind, "fixed", False, t_exp_days=t_exp_days)
    flux = np.asarray(m.predict(TRUTH[kind], t, b), dtype=float)
    rng = np.random.default_rng(seed)
    sig = 0.05 * flux + 0.01 * np.nanmax(flux)
    obs = flux + rng.normal(0.0, sig)
    if upper is not None:
        obs = np.where(upper, 3.0 * sig, obs)
        sig = np.where(upper, np.nan, sig)
    return wp.LightCurve(time=t, band=b, flux=obs, flux_err=sig, upper_limit=upper,
                         redshift=Z, name=f"{kind}_{seed}")


@functools.lru_cache(maxsize=None)
def _lc(kind, name):
    t, b = _grid(name)
    return _observe(kind, t, b, seed=sorted(CURVES).index(name) + 1)


@functools.lru_cache(maxsize=None)
def _ld(kind, config, unique_epochs, name, geometric=None, **kw):
    # the model object of `_model(kind, config, unique_epochs)` (one lru_cache key) unless
    # `geometric` is given, so that the paths share its programs
    m = (_model(kind, config, unique_epochs) if geometric is None
         else _model(kind, config, unique_epochs, geometric=geometric))
    return A.log_density(_lc(kind, name), m, **kw)


def _draws(ld, n, seed=SEED):
    """``n`` seeded draws of the density's own prior, in ``ld.names`` order."""
    free_prior = DG.split_fixed(ld.prior, list(ld.names) + list(ld.fixed), "test")[0]
    return DG.prior_draws(free_prior, list(ld.names), n, seed)


def _blocks(fn, th, width=BLOCK):
    out = [fn(jnp.asarray(th[i:i + width])) for i in range(0, th.shape[0], width)]
    if isinstance(out[0], tuple):
        return tuple(np.concatenate([np.asarray(o[j], dtype=float) for o in out])
                     for j in range(len(out[0])))
    return np.concatenate([np.asarray(o, dtype=float) for o in out])


# --- comparisons ------------------------------------------------------------------------------

def _bits(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    return float(np.mean((a.view(np.int64) == b.view(np.int64)) | (np.isnan(a) & np.isnan(b))))


def _mag(flux):
    with np.errstate(all="ignore"):
        return -2.5 * np.log10(np.asarray(flux, dtype=float) / 3631.0)


def _close_flux(new, ref, what):
    """Magnitudes within 1e-12, the same finite pattern; returns (bitwise fraction, max |dmag|)."""
    new, ref = np.asarray(new, dtype=float), np.asarray(ref, dtype=float)
    assert new.shape == ref.shape, (what, new.shape, ref.shape)
    mn, mr = _mag(new), _mag(ref)
    assert np.array_equal(np.isfinite(mn), np.isfinite(mr)), f"{what}: finite pattern differs"
    ok = np.isfinite(mr)
    d = float(np.max(np.abs(mn - mr)[ok])) if ok.any() else 0.0
    frac = _bits(new, ref)
    _measured(what, bitwise=f"{frac:.4f}", max_dmag=f"{d:.3g}")
    assert d <= TOL_MAG, f"{what}: |dmag| up to {d:.3g}"
    return frac, d


def _scale(ll_ref, log_norm):
    """Per draw, the size of the terms a log density sums: its normalising constant and the
    residual sum (``ll - log_norm``). 1e-12 of it is the round-off bound where they cancel."""
    ll_ref = np.asarray(ll_ref, dtype=float)
    with np.errstate(invalid="ignore"):
        s = np.maximum(abs(float(log_norm)), np.abs(ll_ref - float(log_norm)))
    return np.where(np.isfinite(s), s, 0.0)


def _close_logd(new, ref, what, scale=None):
    """|d| <= 1e-12 max(1, |ref|, scale), identical non-finite values; (bitwise, max rel)."""
    new, ref = np.asarray(new, dtype=float), np.asarray(ref, dtype=float)
    assert new.shape == ref.shape, (what, new.shape, ref.shape)
    fn, fr = np.isfinite(new), np.isfinite(ref)
    assert np.array_equal(fn, fr), f"{what}: finite pattern differs ({np.sum(fn != fr)} draws)"
    assert np.array_equal(np.where(fn, 0.0, new), np.where(fr, 0.0, ref), equal_nan=True), what
    s = np.maximum(1.0, np.abs(ref))
    if scale is not None:
        s = np.maximum(s, np.asarray(scale, dtype=float))
    r = (np.abs(new - ref) / s)[fr]
    m = float(r.max()) if r.size else 0.0
    frac = _bits(new, ref)
    d = float(np.max(np.abs(new - ref)[fr])) if fr.any() else 0.0
    _measured(what, bitwise=f"{frac:.4f}", max_rel=f"{m:.3g}", max_abs=f"{d:.3g}",
              max_abs_ref=f"{float(np.max(np.abs(ref[fr]))) if fr.any() else 0.0:.3g}")
    assert m <= TOL_LOGD, f"{what}: relative difference up to {m:.3g}"
    return frac, m


def _close_grad(new, ref, what):
    """Per draw max |dg| <= 1e-10 ||g_ref||_inf on the draws finite in both; the same pattern."""
    new, ref = np.asarray(new, dtype=float), np.asarray(ref, dtype=float)
    assert np.array_equal(np.isfinite(new), np.isfinite(ref)), f"{what}: finite pattern differs"
    rows = np.all(np.isfinite(ref), axis=1)
    size = np.max(np.abs(ref[rows]), axis=1)
    d = np.max(np.abs(new[rows] - ref[rows]), axis=1)
    r = np.where(size > 0, d / np.where(size > 0, size, 1.0), np.where(d > 0, np.inf, 0.0))
    m = float(r.max()) if r.size else 0.0
    _measured(what, bitwise=f"{_bits(new, ref):.4f}", max_rel_to_inf_norm=f"{m:.3g}",
              n_rows=int(rows.sum()))
    assert m <= TOL_GRAD, f"{what}: gradient differs by {m:.3g} of its largest component"
    return m


# --- jaxpr walking (claim 4, claim 8) -----------------------------------------------------------

def _walk(jaxpr, found, prim=None):
    """Every outvar shape of ``jaxpr`` and the jaxprs inside it (or, with ``prim``, the input
    shapes of that primitive), as ``test_compile_barrier._barriers`` walks."""
    for eqn in jaxpr.eqns:
        if prim is None:
            found.extend(tuple(v.aval.shape) for v in eqn.outvars if hasattr(v, "aval"))
        elif eqn.primitive.name == prim:
            found.extend(tuple(v.aval.shape) for v in eqn.invars if hasattr(v, "aval"))
        stack = list(eqn.params.values())
        while stack:
            p = stack.pop()
            if hasattr(p, "jaxpr"):
                _walk(p.jaxpr, found, prim)
            elif hasattr(p, "eqns"):
                _walk(p, found, prim)
            elif isinstance(p, (list, tuple)):
                stack.extend(p)


def _kernel_rows(closed):
    """The leading sizes of every ``(..., rows, 1000)`` intermediate: the viscous kernel's rows."""
    shapes = []
    _walk(closed.jaxpr, shapes)
    return {s[-2] for s in shapes if len(s) >= 2 and s[-1] == N_KERNEL}


# =============================================================================================
# 1. the map
# =============================================================================================

def _random_times(rng):
    n = int(rng.integers(1, 301))
    pool = rng.choice([0.0, -0.0, np.nan, np.inf, -np.inf, 1.5, -2.25, 1e5], size=8)
    pool = np.concatenate([pool, rng.normal(0.0, 50.0, int(rng.integers(1, 40)))])
    return rng.choice(pool, size=n)


def test_epoch_index_contract():
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        t = _random_times(rng)
        first, inv = AT.epoch_index(t)
        assert first.dtype == np.int32 and inv.dtype == np.int32 and inv.shape == t.shape
        back = t[first][inv]
        assert np.all((back == t) | (np.isnan(back) & np.isnan(t)))
        u = t[first]
        v = u[~np.isnan(u)]
        assert np.all(v[:-1] < v[1:]), u                                     # increasing
        assert np.sum(np.isnan(u)) <= 1 and (not np.isnan(u).any() or np.isnan(u[-1]))
        for k, f in enumerate(first):                    # the first row holding epoch k
            same = (t == t[f]) | (np.isnan(t) & np.isnan(t[f]))
            assert f == np.flatnonzero(same)[0]
        size = first.size + int(rng.integers(0, 5))
        fp, ip = AT.epoch_index(t, size)
        assert fp.shape == (size,) and np.array_equal(fp[:first.size], first)
        assert np.all(fp[first.size:] == first[-1]) and np.array_equal(ip, inv)
    with pytest.raises(ValueError, match="size=1 is smaller"):
        AT.epoch_index([1.0, 2.0], size=1)
    with pytest.raises(ValueError, match="once per light curve"):
        AT.epoch_index(np.zeros((3, 4)))
    first, inv = AT.epoch_index([])
    assert first.shape == inv.shape == (0,) and first.dtype == inv.dtype == np.int32
    with pytest.raises(ValueError, match="no epochs to pad"):
        AT.epoch_index([], size=4)
    first, inv = AT.epoch_index([0.0, -0.0, np.nan, np.nan])     # +-0 merge, NaNs merge
    assert first.tolist() == [0, 2] and inv.tolist() == [0, 0, 1, 1]


def test_epoch_index_docstring_example_runs():
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    tests = doctest.DocTestFinder().find(AT.epoch_index, "epoch_index", globs={})
    assert tests and tests[0].examples
    for test in tests:
        runner.run(test)
    assert runner.summarize(verbose=False).failed == 0


def test_log_density_docstring_examples_run():
    """``log_density``'s examples, the fall-back TDE one included (claim 1). That example switches
    x64 on; the session's setting is put back, and the caches cleared, if it changed."""
    was = jax.config.jax_enable_x64
    tests = doctest.DocTestFinder().find(A.log_density, "log_density", globs={})
    assert any("n_distinct_epochs" in ex.source for t in tests for ex in t.examples)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for test in tests:
                runner.run(test, out=io.StringIO().write)
    finally:
        if jax.config.jax_enable_x64 != was:
            jax.config.update("jax_enable_x64", was)
            jax.clear_caches()
    assert runner.tries > 0 and runner.failures == 0


def test_padded_rows_join_the_latest_epoch():
    ld = _ld("tde_fallback", "fixed", True, "LC35")
    d = {k: np.asarray(v) for k, v in ld.data.items()}
    assert ld.n_distinct_epochs == 35 and ld.epoch_bucket == 48 and ld.bucket == 192
    assert d["epoch_first"].shape == (48,) and d["epoch_inv"].shape == (192,)
    assert d["epoch_first"].dtype == d["epoch_inv"].dtype == np.int32
    latest = int(np.argmax(d["t"][:175]))
    assert np.all(d["epoch_inv"][175:] == d["epoch_inv"][latest])           # pad rows
    assert np.all(d["epoch_first"][35:] == d["epoch_first"][34])            # pad slots
    assert d["epoch_first"][34] == latest                                   # its first row
    t = d["t"]
    assert np.array_equal(t[d["epoch_first"]][d["epoch_inv"]], t)
    assert "35 distinct epochs padded to 48" in repr(ld)


# =============================================================================================
# 2. the physics with a map
# =============================================================================================

@functools.lru_cache(maxsize=None)
def _physics(kind, variant):
    """Magnitudes (200 draws x 175 rows, LC35, fixed z) of the physics function, through the
    model's own context, with ``variant``: "none" (per observation), "identity", or E (int)."""
    m = _model(kind, "fixed", False)
    ctx = m.predict.__self__
    t, b = _grid("LC35")
    rel = jnp.asarray(t - T0)
    bidx = jnp.asarray(m.predict_jax.band_index(b))
    k = len(m.parameters)
    if variant == "none":
        def one(x):
            return ctx._physics_call(rel, bidx, [x[i] for i in range(k)])
    else:
        n = t.size
        f, i = ((np.arange(n), np.arange(n)) if variant == "identity"
                else AT.epoch_index(t - T0, int(variant)))
        f, i = jnp.asarray(f, jnp.int32), jnp.asarray(i, jnp.int32)

        def one(x):
            return ctx._physics_call(rel, bidx, [x[j] for j in range(k)], epoch_idx=(f, i))
    th = _draws(_ld(kind, "fixed", False, "LC35"), 200, SEED + 2)
    return _blocks(jax.jit(jax.vmap(one)), th)


def _mags_close(new, ref, what):
    assert np.array_equal(np.isfinite(new), np.isfinite(ref)), what
    d = float(np.max(np.abs(new - ref)))
    frac = _bits(new, ref)
    _measured(what, bitwise=f"{frac:.4f}", max_dmag=f"{d:.3g}")
    assert d <= TOL_MAG, f"{what}: |dmag| up to {d:.3g}"


@pytest.mark.parametrize("kind", KINDS)
def test_identity_map_matches(kind):
    """The identity map adds two gathers and the check: the bitwise fraction is recorded, not
    asserted (the gathers can move XLA's fusion boundaries)."""
    _mags_close(_physics(kind, "identity"), _physics(kind, "none"), f"claim2 {kind} identity")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("size", [35, 48, pytest.param(175, marks=pytest.mark.slow)])
def test_distinct_epoch_map_matches(kind, size):
    ref = _physics(kind, "none")
    new = _physics(kind, size)
    assert np.all(np.isfinite(ref))
    _mags_close(new, ref, f"claim2 {kind} E={size}")


# =============================================================================================
# 3. every call path
# =============================================================================================

FLUX_PATHS = ("predict", "jit_vmap_concrete", "traced_no_map", "traced_with_map",
              "batched_chunk_none", "batched_chunk_50")
LOGD_PATHS = ("make_log_prob_jax", "log_density", "shared_other_data", "lmo_score", "lmo_value",
              "profile_program", "closure_program")
#: The paths of the fast set: the rest, and 200 draws on every path, are ``slow``.
FAST_PATHS = ("predict", "jit_vmap_concrete", "traced_no_map", "traced_with_map",
              "batched_chunk_none", "make_log_prob_jax", "log_density", "shared_other_data",
              "lmo_score", "profile_program", "closure_program")


def _lmo():
    return importlib.import_module("whisper_cbpf.likelihood_max_opt")


def _lmo_programs(m, lc, ld):
    """``likelihood_max_opt._jax_programs`` as an optimisation builds them, with the density it
    built (captured) and its cache entry."""
    from whisper_cbpf.likelihood import make_likelihood
    from whisper_cbpf.samplers.base import _pre_event_plan

    LMO = _lmo()
    lc_fit = _pre_event_plan(lc, m, None).lc
    lik = make_likelihood(lc_fit, kind="auto", space="auto")
    names = list(ld.names)
    box = LMO._Box(ld.prior, names)
    seen = []
    orig = A.log_density

    def spy(*a, **k):
        seen.append(orig(*a, **k))
        return seen[-1]

    with _patched(A, "log_density", spy):
        progs, reused = LMO._jax_programs(lc_fit, m, box, names, lik)
    entry = LMO._PROGRAMS[next(reversed(LMO._PROGRAMS))]
    return progs, reused, seen[-1], entry


def _profile_density(m, lc):
    """The density ``profile`` builds (captured; nothing is compiled), and its keywords."""
    seen = []
    orig = A.log_density

    def spy(*a, **k):
        seen.append((orig(*a, **k), k))
        raise _Stop

    with _patched(A, "log_density", spy), pytest.raises(_Stop):
        wp.profile(m, lc, batch_sizes=(1,), grad=False, repeats=1)
    return seen[0]


def _closure_density(m, lc):
    """``log_density`` on its closed-over path, forced: the data-as-argument probe fails."""
    def fail(*a, **k):
        raise TypeError("forced: the epochs are closed over")

    with _patched(jax, "eval_shape", fail):
        ld = A.log_density(lc, m)
    assert not ld.data_as_argument and "forced" in ld.reason
    return ld


@functools.lru_cache(maxsize=None)
def _path(kind, config, unique_epochs, path, n, geometric=None):
    """``path``'s output for LC35 on ``n`` draws of the per-observation density's prior, on the
    model of ``_model(kind, config, unique_epochs, geometric=...)`` (``None``: its default)."""
    m = (_model(kind, config, unique_epochs) if geometric is None
         else _model(kind, config, unique_epochs, geometric=geometric))
    lc = _lc(kind, "LC35")
    th = _draws(_ld(kind, config, False, "LC35"), n)
    t_np, b_np = np.asarray(lc.time, dtype=float), np.asarray(lc.band)
    pj = m.predict_jax
    bidx = pj.band_index(b_np)
    if path == "predict":
        names = list(m.parameters)
        return np.stack([np.asarray(m.predict(dict(zip(names, x)), t_np, b_np), dtype=float)
                         for x in th])
    if path == "jit_vmap_concrete":
        return _blocks(jax.jit(jax.vmap(lambda x: pj(x, t_np, bidx))), th)
    if path == "traced_no_map":
        f = jax.jit(jax.vmap(lambda x, t: pj(x, t, bidx), in_axes=(0, None)))
        return _blocks(lambda x: f(x, jnp.asarray(t_np)), th)
    if path == "traced_with_map":
        fi, iv = (jnp.asarray(a) for a in pj.epoch_index(t_np))
        f = jax.jit(jax.vmap(lambda x, t, a, b: pj(x, t, bidx, epoch_idx=(a, b)),
                             in_axes=(0, None, None, None)))
        return _blocks(lambda x: f(x, jnp.asarray(t_np), fi, iv), th)
    if path == "batched_chunk_none":
        return _blocks(A.make_batched_predict_jax(lc, m), th)
    if path == "batched_chunk_50":
        return np.asarray(A.make_batched_predict_jax(lc, m, chunk=50)(jnp.asarray(th)), float)
    if path == "make_log_prob_jax":                    # (log likelihood, log posterior)
        like = A.make_log_prob_jax(lc, m, include_prior=False)
        post = A.make_log_prob_jax(lc, m, include_prior=True)
        return _blocks(jax.jit(jax.vmap(lambda x: (like(x), post(x)))), th)
    if path == "log_density":                          # (fn, log_likelihood)
        ld = _ld(kind, config, unique_epochs, "LC35")
        return _blocks(jax.jit(jax.vmap(lambda x: (ld.fn(x), ld.log_likelihood(x)))), th)
    if path == "shared_other_data":                    # LC35's program on LC28's data
        ld_a, ld_b = (_ld(kind, config, unique_epochs, nm) for nm in ("LC35", "LC28"))
        assert ld_a.shared is ld_b.shared
        f = jax.jit(jax.vmap(ld_a.shared, in_axes=(0, None)))
        return _blocks(lambda x: f(x, ld_b.data), th)
    if path in ("lmo_score", "lmo_value"):
        progs = _lmo_programs(m, lc, _ld(kind, config, unique_epochs, "LC35"))[0]
        if path == "lmo_score":
            return np.asarray(progs["score"](th), dtype=float)
        return np.array([progs["value"](x) for x in th[:8]], dtype=float)
    if path == "profile_program":
        pld = _profile_density(m, lc)[0]
        f = jax.jit(jax.vmap(lambda x, d: pld.shared(x, d)[0], in_axes=(0, None)))
        return _blocks(lambda x: f(x, pld.data), th)
    if path == "closure_program":
        cm = _model(kind, config, unique_epochs, tag="_closure")
        cld = _closure_density(cm, lc)
        return _blocks(jax.jit(jax.vmap(cld.fn)), th)
    raise ValueError(path)


#: The fast set reads the per-observation value of the same quantity off one program where several
#: paths compute it (a compile each saved); the slow sweep runs every path on both models.
SHARED_REFERENCE = {"traced_with_map": ("traced_no_map", None),
                    "batched_chunk_none": ("jit_vmap_concrete", None),
                    "lmo_score": ("log_density", 1), "lmo_value": ("log_density", 1),
                    "profile_program": ("log_density", 0), "closure_program": ("log_density", 0)}
N_SCALE = 200                 # the draws of the per-observation log likelihood behind the scale


def _reference(kind, config, path, n, same_path):
    """The per-observation model's value of the quantity ``path`` computes (by the same path, or
    by the one that shares it), and the per-draw scale of the log density's summed terms."""
    ref_path, part = (path, None) if same_path else SHARED_REFERENCE.get(path, (path, None))
    if ref_path == "traced_with_map":                  # the per-observation model has no map
        ref_path = "traced_no_map"
    if ref_path == "log_density" and part is not None:
        ref = _path(kind, config, False, "log_density", N_SCALE)[part][:n]
    else:
        ref = _path(kind, config, False, ref_path, n)
    if path in FLUX_PATHS:
        return ref, None
    if path == "lmo_value":
        ref = ref[:8]
    if path == "shared_other_data":                    # LC28's own log likelihood
        ll0, ld0 = ref[1], _ld(kind, config, False, "LC28")
    else:
        ll0, ld0 = _path(kind, config, False, "log_density", N_SCALE)[1], \
            _ld(kind, config, False, "LC35")
    scale = _scale(ll0, float(np.asarray(ld0.data["log_norm"])))
    m = np.shape(ref[0] if isinstance(ref, tuple) else ref)[0]
    return ref, scale[:m]


PARTS = {"make_log_prob_jax": ("likelihood", "posterior"), "log_density": ("fn", "loglik"),
         "shared_other_data": ("lp", "ll")}


def _check_path(kind, config, path, n, same_path=False):
    new = _path(kind, config, True, path, n)
    ref, scale = _reference(kind, config, path, n, same_path)
    what = f"claim3 {kind} {config} {path}"
    if path == "traced_no_map":
        # without a map the default model evaluates every observation: the code of the
        # per-observation model with the same segment search, bit for bit
        same = _path(kind, config, False, path, n, geometric=True)
        assert np.array_equal(new, same), what
        _measured(what + " vs unique_epochs=False, geometric=True", bitwise="1.0000 (array_equal)")
    if path in FLUX_PATHS:
        _close_flux(new, ref, what)
    elif isinstance(new, tuple):
        for j, part in enumerate(PARTS[path]):
            _close_logd(new[j], ref[j], f"{what} {part}", scale)
    else:
        _close_logd(new, ref, what, scale)


@pytest.mark.parametrize("path", FAST_PATHS)
@pytest.mark.parametrize("kind", KINDS)
def test_call_path_matches_per_observation(kind, path):
    """Free t_exp and redshift; 200 draws on ``predict`` and ``log_density``, 50 elsewhere."""
    _check_path(kind, "free", path, 200 if path in ("predict", "log_density") else 50)


@pytest.mark.slow
@pytest.mark.parametrize("path", FLUX_PATHS + LOGD_PATHS)
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("kind", KINDS)
def test_call_path_sweep_200_draws(kind, config, path):
    """Every path, both configs, 200 draws, each against the same path's per-observation run."""
    _check_path(kind, config, path, 200, same_path=True)


def test_host_map_is_used_and_skipped():
    m = _model("tde_mosfit", "fixed", True, tag="_host")
    ctx = m.predict.__self__
    t, b = _grid("LC35")
    m.predict(TRUTH["tde_mosfit"], t, b)
    assert list(ctx._jit_cache) == ["epochs"]
    dense = np.linspace(T0 + 1.0, T0 + 300.0, 40)                 # every epoch distinct
    m.predict(TRUTH["tde_mosfit"], dense, np.array(["lsstg"] * 40))
    assert sorted(ctx._jit_cache, key=str) == [None, "epochs"]
    flux = m.predict(TRUTH["tde_mosfit"], dense, np.array(["lsstg"] * 40))
    same = _model("tde_mosfit", "fixed", False, geometric=True).predict(
        TRUTH["tde_mosfit"], dense, np.array(["lsstg"] * 40))
    assert np.array_equal(flux, same)              # the same core (and search), the same bits
    ref = _model("tde_mosfit", "fixed", False).predict(TRUTH["tde_mosfit"], dense,
                                                        np.array(["lsstg"] * 40))
    _close_flux(flux, ref, "claim3 tde_mosfit all distinct, host predict vs per observation")


# =============================================================================================
# 4. every path uses the map
# =============================================================================================

PATHS_4 = ("predict", "jit_vmap_concrete", "traced_with_map", "make_log_prob_jax",
           "batched_chunk_none", "log_density", "lmo_vg", "profile_program", "closure_program",
           "fit_batch")


def _jaxpr_of(kind, unique_epochs, path):
    """``(closed jaxpr, E expected with the map)`` of ``path`` (free config, LC35), traced only."""
    m = _model(kind, "free", unique_epochs, tag="_jaxpr")
    lc = _lc(kind, "LC35")
    ld0 = _ld(kind, "free", False, "LC35")
    th = jnp.asarray(_draws(ld0, 4))
    t_np, b_np = np.asarray(lc.time, dtype=float), np.asarray(lc.band)
    pj = m.predict_jax
    bidx = pj.band_index(b_np)
    if path == "predict":
        seen = []
        orig = AT.DirectTDEPredict._jitted

        def spy(self, key, build):
            def call(*args):
                seen.append((key, build(), args))
                raise _Stop
            return call

        with _patched(AT.DirectTDEPredict, "_jitted", spy), pytest.raises(_Stop):
            m.predict(dict(zip(m.parameters, np.asarray(th[0]))), t_np, b_np)
        assert orig is AT.DirectTDEPredict._jitted
        key, fn, args = seen[0]
        assert key == ("epochs" if unique_epochs else None)
        return jax.make_jaxpr(fn)(*args), 35
    if path == "jit_vmap_concrete":
        return jax.make_jaxpr(jax.vmap(lambda x: pj(x, t_np, bidx)))(th), 35
    if path == "traced_with_map":
        if not unique_epochs:
            return jax.make_jaxpr(jax.vmap(lambda x, t: pj(x, t, bidx), in_axes=(0, None)))(
                th, jnp.asarray(t_np)), 35
        fi, iv = (jnp.asarray(a) for a in pj.epoch_index(t_np))
        return jax.make_jaxpr(jax.vmap(lambda x, t, a, b: pj(x, t, bidx, epoch_idx=(a, b)),
                                       in_axes=(0, None, None, None)))(
            th, jnp.asarray(t_np), fi, iv), 35
    if path == "make_log_prob_jax":
        return jax.make_jaxpr(A.make_log_prob_jax(lc, m, include_prior=True))(th[0]), 35
    if path == "batched_chunk_none":
        return jax.make_jaxpr(A.make_batched_predict_jax(lc, m))(th), 35
    if path == "log_density":
        return jax.make_jaxpr(A.log_density(lc, m).fn)(th[0]), 48
    if path == "lmo_vg":
        _, _, lld, entry = _lmo_programs(m, lc, A.log_density(lc, m))
        return jax.make_jaxpr(entry["vg"])(th, lld.data), 48
    if path == "profile_program":
        pld, kw = _profile_density(m, lc)
        assert kw.get("epoch_bucket", "auto") is None
        return jax.make_jaxpr(jax.vmap(lambda x, d: pld.shared(x, d)[0], in_axes=(0, None)))(
            th, pld.data), 35
    if path == "closure_program":
        cm = _model(kind, "free", unique_epochs, tag="_jaxpr_closure")
        return jax.make_jaxpr(_closure_density(cm, lc).fn)(th[0]), 35
    if path == "fit_batch":
        program, data = _fit_batch_program(m, [_lc(kind, nm) for nm in ("LC35", "LC28", "LC21")])
        if unique_epochs:
            assert data["epoch_first"].shape == (3, 35) and data["epoch_inv"].shape == (3, 192)
        x = jnp.broadcast_to(th[:2], (3, 2, th.shape[1]))
        lnp = jax.vmap(jax.vmap(program, in_axes=(0, None)), in_axes=(0, 0))
        return jax.make_jaxpr(lnp)(x, data), 35
    raise ValueError(path)


def _fit_batch_program(model, curves, **kw):
    """The walker program and the stacked data ``fit_batch`` compiles (captured; nothing runs)."""
    seen = {}
    orig = B._make_loop

    class _Lower:
        def __init__(self, run):
            self.run = run

        def lower(self, keys, x_spec, data):
            seen["data"] = data
            raise _Stop

    def spy(program, *a, **k):
        run, score = orig(program, *a, **k)
        seen["program"] = program
        return _Lower(run), score

    with _patched(B, "_make_loop", spy), pytest.raises(_Stop):
        B.fit_batch(curves, model, nwalkers=2 * len(model.parameters), nsteps=4, burnin=2,
                    thin=1, init="prior", metrics=False, **kw)
    return seen["program"], seen["data"]


@pytest.mark.parametrize("path", PATHS_4)
@pytest.mark.parametrize("kind", KINDS)
def test_each_path_runs_the_kernel_at_distinct_epochs(kind, path):
    """LC35 has n = 175 observations, bucket 192, 35 distinct epochs, epoch bucket 48, and the
    kernel 1000 nodes: all different, so the kernel's row count names the path's evaluation."""
    closed, e = _jaxpr_of(kind, True, path)
    rows = _kernel_rows(closed)
    assert e in rows and not rows & {175, 192}, (path, rows)
    closed0, _ = _jaxpr_of(kind, False, path)
    rows0 = _kernel_rows(closed0)
    assert rows0 & {175, 192} and e not in rows0, (path, rows0)


# =============================================================================================
# 5. gradients
# =============================================================================================

@functools.lru_cache(maxsize=None)
def _vg(kind, config, unique_epochs, n=48):
    """``(log posterior, log likelihood, gradient)`` on ``n`` draws, vmapped value_and_grad."""
    ld = _ld(kind, config, unique_epochs, "LC35")
    th = _draws(_ld(kind, config, False, "LC35"), n, SEED + 5)
    f = jax.jit(jax.vmap(jax.value_and_grad(lambda x, d: ld.shared(x, d), has_aux=True),
                         in_axes=(0, None)))
    (v, ll), g = f(jnp.asarray(th), ld.data)
    return (np.asarray(v, dtype=float), np.asarray(ll, dtype=float),
            np.asarray(g, dtype=float))


@pytest.mark.parametrize("config", ["fixed", pytest.param("free", marks=pytest.mark.slow)])
@pytest.mark.parametrize("kind", KINDS)
def test_gradients_match(kind, config):
    """Vmapped ``value_and_grad`` of the log posterior (``likelihood_max_opt``'s shape)."""
    v1, _, g1 = _vg(kind, config, True)
    v0, ll0, g0 = _vg(kind, config, False)
    scale = _scale(ll0, float(np.asarray(_ld(kind, config, False, "LC35").data["log_norm"])))
    assert np.isfinite(v0).sum() >= 40
    _close_logd(v1, v0, f"claim5 {kind} {config} value", scale)
    _close_grad(g1, g0, f"claim5 {kind} {config} grad")


@pytest.mark.parametrize("kind", KINDS)
def test_make_log_prob_jax_gradients_match(kind):
    """``jax.grad`` of ``make_log_prob_jax`` (the NUTS paths), free t_exp and redshift."""
    lc = _lc(kind, "LC35")
    th = jnp.asarray(_draws(_ld(kind, "free", False, "LC35"), 48, SEED + 6))
    g = {ue: np.asarray(jax.jit(jax.vmap(jax.grad(A.make_log_prob_jax(
        lc, _model(kind, "free", ue)))))(th), dtype=float) for ue in (True, False)}
    _close_grad(g[True], g[False], f"claim5 {kind} make_log_prob_jax grad")


@pytest.mark.parametrize("kind", KINDS)
def test_time_derivative_is_per_epoch(kind):
    """With traced epochs and a map, d/dt of an epoch sits on its first observation: the other
    rows get exactly 0, and the per-epoch sums are the per-observation ones."""
    ld1, ld0 = (_ld(kind, "fixed", ue, "LC35") for ue in (True, False))
    th = jnp.asarray(_draws(ld0, 4, SEED + 7))

    def dt(ld):
        def lp(t, x):
            return ld.shared(x, {**ld.data, "t": t})[0]
        return np.asarray(jax.jit(jax.vmap(jax.grad(lp), in_axes=(None, 0)))(ld.data["t"], th))

    g1, g0 = dt(ld1), dt(ld0)
    first = np.asarray(ld1.data["epoch_first"])
    inv = np.asarray(ld1.data["epoch_inv"])
    not_first = np.arange(inv.size) != first[inv]
    assert np.all(g1[:, not_first] == 0.0) and np.all(np.isfinite(g1))
    s1 = np.stack([np.bincount(inv, weights=r, minlength=first.size) for r in g1])
    s0 = np.stack([np.bincount(inv, weights=r, minlength=first.size) for r in g0])
    top = np.max(np.abs(s0), axis=1)
    d = np.max(np.abs(s1 - s0), axis=1)
    r = np.where(top > 0, d / np.where(top > 0, top, 1.0), np.where(d > 0, np.inf, 0.0))
    _measured(f"claim5 {kind} d/dt per-epoch sums", max_rel=f"{float(r.max()):.3g}",
              per_draw=[f"{v:.2g}" for v in r])
    # Not 1e-12: inside one epoch's backward pass the viscous kernel's node terms cancel (the
    # rise of log10 L from its floor makes steep segments), so a summation order moves the
    # result by more than the round-off of its own size (4.5e-11 at worst on these draws).
    assert np.all(r <= TOL_TIME_DERIVATIVE), r


# =============================================================================================
# 6. edge cases and random light curves
# =============================================================================================

def _edge_case(name):
    """``(times, bands)`` on a clock whose time zero is 0."""
    rng = np.random.default_rng(SEED + 11)
    if name == "mixed":       # unsorted, same-band repeats, rows at/before 0, +-0, a NaN time
        ep = np.array([-3.0, 0.0, 2.5, 7.0, 15.0, 40.0, 90.0])
        t = np.concatenate([np.repeat(ep, 5), [40.0, 90.0, 0.0, -0.0, np.nan]])
        b = np.concatenate([np.tile(np.array(BANDS), ep.size), ["lsstg", "lsstr", "lsstg",
                                                                 "lsstr", "lssti"]])
        perm = rng.permutation(t.size)
        return t[perm], b[perm]
    if name == "all_before":
        return np.repeat([-20.0, -5.0, 0.0], 3), np.tile(np.array(BANDS[:3]), 3)
    if name == "one_epoch":
        return np.full(5, 12.0), np.array(BANDS)
    if name == "one_row":
        return np.array([12.0]), np.array(["lsstr"])
    if name == "no_repeats":
        return np.linspace(2.0, 150.0, 20), np.array([BANDS[i % 5] for i in range(20)])
    raise ValueError(name)


@pytest.mark.parametrize("case", ["mixed", "all_before", "one_epoch",
                                  pytest.param("one_row", marks=pytest.mark.slow),
                                  pytest.param("no_repeats", marks=pytest.mark.slow)])
@pytest.mark.parametrize("kind", KINDS)
def test_edge_cases(kind, case):
    """``no_repeats`` is also ``test_host_map_is_used_and_skipped``'s dense grid (fast)."""
    t, b = _edge_case(case)
    m1, m0 = (_model(kind, "fixed", ue, t_exp_days=0.0) for ue in (True, False))
    th = _draws(_ld(kind, "fixed", False, "LC35"), 20, SEED + 12)
    pj1, pj0 = m1.predict_jax, m0.predict_jax
    bidx = pj1.band_index(b)
    f1 = np.asarray(jax.jit(jax.vmap(lambda x: pj1(x, t, bidx)))(jnp.asarray(th)))
    f0 = np.asarray(jax.jit(jax.vmap(lambda x: pj0(x, t, bidx)))(jnp.asarray(th)))
    if case == "no_repeats":    # no map: the per-observation core itself, with the same search
        pjg = _model(kind, "fixed", False, t_exp_days=0.0, geometric=True).predict_jax
        assert np.array_equal(f1, np.asarray(jax.jit(jax.vmap(lambda x: pjg(x, t, bidx)))(
            jnp.asarray(th))))
    _close_flux(f1, f0, f"claim6 {kind} {case}")
    floor = 3631.0 * 10.0 ** (-0.4 * 40.0)
    dead = ~(t > 0.0)                               # at or before 0, and NaN
    np.testing.assert_allclose(f1[:, dead], floor, rtol=1e-12)
    if case == "all_before":
        g = jax.jit(jax.vmap(jax.grad(lambda x: jnp.sum(pj1(x, t, bidx)))))(jnp.asarray(th))
        assert np.all(np.isfinite(np.asarray(g)))


def _check_log_density_case(kind, name):
    """Through ``log_density``: ``"upper_limits"``, an unsorted curve with repeats and upper
    limits after the event (flux space, censored likelihood), or ``"alert"``, the shipped LSST
    alert (15 rows, every epoch distinct, its forced-photometry epochs upper limits): the map's
    no-gain case."""
    if name == "alert":
        curve = wp.load_lightcurve(str(_alert_path()), survey="lsst", redshift=0.1)
        z, kw = 0.1, {"t_exp_days": 60990.0}
    else:
        t, b = _edge_case("mixed")
        keep = np.isfinite(t)
        t, b = t[keep], b[keep]
        upper = (t > 30.0) & np.isin(b, ["lsstu", "lsstz"])
        curve = _observe(kind, t, b, seed=SEED + 13, t_exp_days=0.0, upper=upper)
        z, kw = Z, {"t_exp_days": 0.0}
    fac = {"tde_fallback": wp.tde_fallback_model, "tde_mosfit": wp.tde_mosfit_model}[kind]
    # the default model, and the per-observation reference (both options off)
    spell = {True: {}, False: {"unique_epochs": False, "geometric": False}}
    lds = {ue: A.log_density(curve, fac(BANDS, z, float(luminosity_distance_cm(z)), **spell[ue],
                                        **kw)) for ue in (True, False)}
    ld = lds[True]
    t_fit = np.asarray(ld.data["t"])[:ld.n_data]                      # after the pre-event rule
    assert ld.n_distinct_epochs == np.unique(t_fit).size
    if name == "alert":
        assert ld.n_distinct_epochs == ld.n_data == 15                # every epoch distinct
    else:                                                             # the censored term is used
        assert ld.space == "flux" and float(np.sum(np.asarray(ld.data["w_ul"]))) > 0
    th = jnp.asarray(_draws(lds[False], 20, SEED + 14))
    out = {ue: [np.asarray(v) for v in jax.jit(jax.vmap(
        lambda x, ld=ld: (ld.fn(x), ld.log_likelihood(x))))(th)] for ue, ld in lds.items()}
    scale = _scale(out[False][1], float(np.asarray(lds[False].data["log_norm"])))
    _close_logd(out[True][0], out[False][0], f"claim6 {kind} {name} lp", scale)
    _close_logd(out[True][1], out[False][1], f"claim6 {kind} {name} ll", scale)


@pytest.mark.parametrize("kind", KINDS)
def test_the_shipped_alert(kind):
    _check_log_density_case(kind, "alert")


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_upper_limits(kind):
    _check_log_density_case(kind, "upper_limits")


def _alert_path():
    from pathlib import Path

    return Path(__file__).parent / "data" / "lsst_alert_sn.json"


def _random_curve(kind, seed):
    """1-60 epochs, 1-6 rows per epoch (bands drawn with repeats), random drops, order and
    0-30 % of the epochs before the time zero, observed at the truth."""
    rng = np.random.default_rng(seed)
    n_ep = int(rng.integers(1, 61))
    before = int(round(rng.uniform(0.0, 0.3) * n_ep))
    ep = np.concatenate([T0 - rng.uniform(0.5, 30.0, before),
                         T0 + rng.uniform(1.0, 250.0, n_ep - before)])
    rows_t, rows_b = [], []
    for e in ep:
        k = int(rng.integers(1, 7))
        rows_t.extend([e] * k)
        rows_b.extend(rng.choice(BANDS, size=k))
    t, b = np.asarray(rows_t), np.asarray(rows_b)
    keep = rng.uniform(size=t.size) > 0.1 if t.size > 3 else np.ones(t.size, bool)
    perm = rng.permutation(int(keep.sum()))
    t, b = t[keep][perm], b[keep][perm]
    if not np.any(t > T0):                                  # at least one detection after it
        t, b = np.append(t, T0 + 20.0), np.append(b, "lsstr")
    return _observe(kind, t, b, seed=seed)


def _check_random(kind, seed):
    lc = _random_curve(kind, seed)
    lds = {ue: A.log_density(lc, _model(kind, "free", ue)) for ue in (True, False)}
    assert lds[True].n_distinct_epochs == np.unique(np.asarray(lds[True].data["t"])
                                                    [:lds[True].n_data]).size
    th = jnp.asarray(_draws(lds[False], 5, seed))
    out = {ue: [np.asarray(v) for v in jax.jit(jax.vmap(
        lambda x, ld=ld: (ld.fn(x), ld.log_likelihood(x))))(th)] for ue, ld in lds.items()}
    scale = _scale(out[False][1], float(np.asarray(lds[False].data["log_norm"])))
    what = (f"claim6 {kind} random seed={seed} n={lds[True].n_data} "
            f"E={lds[True].n_distinct_epochs}/{lds[True].epoch_bucket}")
    _close_logd(out[True][0], out[False][0], what + " lp", scale)


@pytest.mark.parametrize("seed", range(5))
def test_random_light_curves(seed):
    _check_random(KINDS[seed % 2], SEED + 100 + seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5, 30))
def test_random_light_curves_slow(seed):
    _check_random(KINDS[seed % 2], SEED + 100 + seed)


# =============================================================================================
# 7. fit_batch with different epoch counts
# =============================================================================================

TOY_PARAMS = ["amp", "t_peak", "width"]
TOY_PRIOR = {"amp": (0.5, 2.0), "t_peak": (20.0, 80.0), "width": (10.0, 60.0)}


def _toy_flux_rows(theta, x_rows, band_idx):
    return theta[0] * jnp.exp(-0.5 * x_rows * x_rows) * (1.0 + 0.1 * band_idx) + 1e-3


class _Toy:
    """A cheap photometric model WITHOUT the hook: a Gaussian bump, every row evaluated."""

    def __init__(self):
        self.band_index = _BandIndex(BANDS, "toy")
        self.seen = []

    def __call__(self, theta, times, band_idx=None):
        x = (times - theta[1]) / theta[2]
        return _toy_flux_rows(theta, x, band_idx)


class _ToyWithHook(_Toy):
    """The same toy WITH the hook: the scaled epoch at the distinct epochs, gathered back, checked;
    the rest per row, on the shapes of the toy without it (so the two are bitwise)."""

    def __init__(self):
        super().__init__()
        self.epoch_index = AT.epoch_index

    def __call__(self, theta, times, band_idx=None, epoch_idx=None):
        if epoch_idx is None:
            return super().__call__(theta, times, band_idx)
        first, inv = epoch_idx
        self.seen.append(tuple(first.shape))
        x = ((times[first] - theta[1]) / theta[2])[inv]
        return AT._guard_epochs(_toy_flux_rows(theta, x, band_idx), times, epoch_idx)


class _ToyHookNoKeyword(_Toy):
    """A broken contract: the hook, but a call without ``epoch_idx``."""

    def __init__(self):
        super().__init__()
        self.epoch_index = AT.epoch_index


def _toy_model(cls):
    def predict(p, t, bands):
        x = (np.asarray(t, float) - p["t_peak"]) / p["width"]
        bi = np.array([BANDS.index(str(v)) for v in np.asarray(bands)])
        return p["amp"] * np.exp(-0.5 * x * x) * (1.0 + 0.1 * bi) + 1e-3

    return wp.Model(name=f"toy_{cls.__name__}", predict=predict, parameters=list(TOY_PARAMS),
                    default_prior=Prior({k: Uniform(*v) for k, v in TOY_PRIOR.items()}),
                    predict_jax=cls())


def _toy_curves():
    out = []
    for i, n_ep in enumerate((35, 28, 21)):
        t = np.repeat(np.geomspace(2.0, 150.0, n_ep), len(BANDS))
        b = np.tile(np.array(BANDS), n_ep)
        p = {"amp": 1.0 + 0.2 * i, "t_peak": 40.0 + 5 * i, "width": 25.0}
        f = _toy_model(_Toy).predict(p, t, b)
        rng = np.random.default_rng(SEED + 20 + i)
        out.append(wp.LightCurve(time=t, band=b, flux=f + rng.normal(0, 0.02, f.size),
                                 flux_err=np.full(f.size, 0.02), name=f"toy{i}"))
    return out


def _fit_capture(curves, model, init, seeds):
    """``fit_batch`` with every member's density and kept log posterior captured."""
    captured = []
    orig = B._result

    def spy(lc, model_, ld, chain, lp_chain, ll_chain, *a, **k):
        captured.append({"ld": ld, "lp": np.array(lp_chain), "chain": np.array(chain)})
        return orig(lc, model_, ld, chain, lp_chain, ll_chain, *a, **k)

    with _patched(B, "_result", spy):
        fits = B.fit_batch(curves, model, nwalkers=8, nsteps=40, burnin=20, thin=1, seed=seeds,
                           init=init, metrics=False)
    return fits, captured


def _toy_init(init, order=(0, 1, 2)):
    if init != "explicit":
        return init
    rng = np.random.default_rng(SEED + 30)
    starts = [np.column_stack([rng.uniform(0.9, 1.1, 8) * (1.0 + 0.2 * i),
                               rng.uniform(38.0, 42.0, 8) + 5 * i, rng.uniform(24.0, 26.0, 8)])
              for i in range(3)]
    return [starts[i] for i in order]


@pytest.mark.parametrize("init", ["prior", "prior_scan", "explicit"])
def test_fit_batch_toy_common_epoch_width(init):
    curves = _toy_curves()
    hooked = _toy_model(_ToyWithHook)
    fits, cap = _fit_capture(curves, hooked, _toy_init(init), [5, 6, 7])
    assert {s for s in hooked.predict_jax.seen} == {(35,)}
    for i, (f, c, n_u) in enumerate(zip(fits, cap, (35, 28, 21))):
        assert f.info["batch"]["epoch_bucket"] == 35 and f.info["batch"]["n_alerts"] == 3
        assert f.info["batch"]["n_distinct_epochs"] == n_u == c["ld"].n_distinct_epochs
        assert c["ld"].data["epoch_first"].shape == (35,) and c["ld"].bucket == 192
    plain_fits, plain = _fit_capture(curves, _toy_model(_Toy), _toy_init(init), [5, 6, 7])
    for f, g in zip(fits, plain_fits):                    # the same shapes for the arithmetic
        assert np.array_equal(f.samples_by_chain, g.samples_by_chain)
    order = (2, 0, 1)
    perm_fits, _ = _fit_capture([curves[i] for i in order], _toy_model(_ToyWithHook),
                                _toy_init(init, order), [7, 5, 6])
    for j, i in enumerate(order):                         # the same width and epoch bucket
        assert np.array_equal(perm_fits[j].samples_by_chain, fits[i].samples_by_chain)
    alone = A.log_density(curves[1], hooked, bucket=192, epoch_bucket=None)
    assert alone.epoch_bucket == 28
    lp = np.asarray(jax.jit(jax.vmap(alone.fn))(jnp.asarray(cap[1]["chain"].reshape(-1, 3))))
    _close_logd(lp, cap[1]["lp"].reshape(-1), f"claim7 toy {init} member alone (E=28)")


@pytest.mark.slow
@pytest.mark.parametrize("init", ["prior", "prior_scan", "explicit"])
@pytest.mark.parametrize("kind", KINDS)
def test_fit_batch_mixed_epoch_counts(kind, init):
    curves = [_lc(kind, nm) for nm in ("LC35", "LC28", "LC21")]
    m = _model(kind, "fixed", True)
    if init == "explicit":
        ld0 = _ld(kind, "fixed", False, "LC35")
        init = [_draws(ld0, 16, SEED + 40 + i) for i in range(3)]
    captured = []
    orig = B._result

    def spy(lc, model_, ld, chain, lp_chain, ll_chain, *a, **k):
        captured.append({"lc": lc, "ld": ld, "lp": np.array(lp_chain), "chain": np.array(chain)})
        return orig(lc, model_, ld, chain, lp_chain, ll_chain, *a, **k)

    with _patched(B, "_result", spy):
        fits = B.fit_batch(curves, m, nwalkers=16, nsteps=24, burnin=8, seed=3, init=init,
                           metrics=False)
    ref_m = _model(kind, "fixed", False)
    for f, c in zip(fits, captured):
        assert f.info["batch"]["epoch_bucket"] == 35
        ref = A.log_density(c["lc"], ref_m, prior=c["ld"].prior, bucket=c["ld"].bucket)
        is_log = DG.walker_log_columns(DG.split_fixed(ref.prior, list(ref.names), "t")[0],
                                       list(ref.names), "own", "t")
        y = DG.to_walker(c["chain"], is_log).reshape(-1, len(ref.names))
        il = jnp.asarray(is_log)

        def walker_lp(yy, ref=ref):
            lp, _ = ref.shared(jnp.where(il, jnp.exp(jnp.where(il, yy, 0.0)), yy), ref.data)
            return lp + jnp.sum(jnp.where(il, yy, 0.0))

        lp = np.asarray(jax.jit(jax.vmap(walker_lp))(jnp.asarray(y)))
        ll0 = np.asarray(jax.jit(jax.vmap(lambda yy, ref=ref: ref.shared(
            jnp.where(il, jnp.exp(jnp.where(il, yy, 0.0)), yy), ref.data)[1]))(jnp.asarray(y)))
        _close_logd(c["lp"].reshape(-1), lp, f"claim7 {kind} {init if isinstance(init, str) else 'explicit'} rescore",
                    _scale(ll0, float(np.asarray(ref.data["log_norm"]))))


# =============================================================================================
# 8. compiles
# =============================================================================================

@pytest.mark.parametrize("kind", KINDS)
def test_one_program_per_bucket_and_epoch_bucket(kind):
    m = _model(kind, "fixed", True, tag="_programs")
    t60 = np.repeat(T0 + np.geomspace(3.0, 300.0, 60), 3)        # 180 rows, 60 distinct epochs
    b60 = np.tile(np.array(BANDS[1:4]), 60)
    lcs = [_lc(kind, "LC35"), _lc(kind, "LC35b"), _observe(kind, t60, b60, SEED + 50)]
    lds = [A.log_density(lc, m) for lc in lcs]
    assert len({id(ld.shared) for ld in lds}) == 1
    assert [ld.bucket for ld in lds] == [192, 192, 192]
    assert [ld.epoch_bucket for ld in lds] == [48, 48, 64]
    th = jnp.asarray(_draws(lds[0], 1))[0]
    sizes = []
    for ld in lds:
        jax.block_until_ready(ld.shared(th, ld.data))
        sizes.append(ld.shared._cache_size())
    assert sizes == [1, 1, 2], sizes


def test_explicit_bucket_keeps_one_compile_and_stacks():
    kind = "tde_mosfit"
    m = _model(kind, "fixed", True, tag="_explicit")
    t30 = np.repeat(T0 + np.geomspace(3.0, 300.0, 30), 5)        # 150 rows, bucket 192
    b30 = np.tile(np.array(BANDS), 30)
    t60 = np.repeat(T0 + np.geomspace(3.0, 300.0, 60), 3)        # 180 rows, bucket 192
    b60 = np.tile(np.array(BANDS[1:4]), 60)
    lcs = [_observe(kind, t30, b30, SEED + 51), _observe(kind, t60, b60, SEED + 52)]
    auto = [A.log_density(lc, m) for lc in lcs]
    assert [ld.epoch_bucket for ld in auto] == [32, 64]
    with pytest.raises(ValueError):                               # the ladder does not stack
        jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[ld.data for ld in auto])
    explicit = [A.log_density(lc, m, bucket=192) for lc in lcs]
    assert [ld.epoch_bucket for ld in explicit] == [192, 192]
    jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[ld.data for ld in explicit])
    common = [A.log_density(lc, m, bucket=192, epoch_bucket=64) for lc in lcs]
    assert [ld.epoch_bucket for ld in common] == [64, 64]
    jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[ld.data for ld in common])
    shared = explicit[0].shared
    assert all(ld.shared is shared for ld in explicit + common + auto)
    th = jnp.asarray(_draws(explicit[0], 1))[0]
    before = shared._cache_size()
    for ld in explicit:
        jax.block_until_ready(shared(th, ld.data))
    assert shared._cache_size() == before + 1                    # one compile for both


@pytest.mark.parametrize("kind", KINDS)
def test_repeated_epochs_keep_the_barrier(kind):
    """24 epochs x 3 bands, concrete: the band integral's ``(n_obs, n_wave)`` barrier is there
    (``test_compile_barrier.py`` uses one band per epoch, so it never traces the map)."""
    m = _model(kind, "fixed", True, t_exp_days=0.0)
    t = np.repeat(np.linspace(2.0, 90.0, 24), 3)
    b = np.tile(np.array(BANDS[1:4]), 24)
    bidx = m.predict_jax.band_index(b)
    n_wave = m.predict.__self__.weights.shape[1]
    th = jnp.asarray(_draws(_ld(kind, "fixed", False, "LC35"), 1))[0]
    closed = jax.make_jaxpr(lambda x: m.predict_jax(x, t, bidx))(th)
    barriers = []
    _walk(closed.jaxpr, barriers, "optimization_barrier")
    assert any(s[-2:] == (72, n_wave) for s in barriers), barriers
    rows = _kernel_rows(closed)                     # 1: the kernel's (1, 1000) node fractions
    assert 24 in rows and 72 not in rows, rows


@pytest.mark.parametrize("kind", KINDS)
def test_closure_path_has_no_map(kind):
    cld = _closure_density(_model(kind, "fixed", True, tag="_closure8"), _lc(kind, "LC35"))
    assert "epoch_first" not in cld.data and "epoch_inv" not in cld.data
    assert cld.n_distinct_epochs == cld.epoch_bucket == 35 and cld.bucket == 175
    ld = _ld(kind, "fixed", True, "LC35")
    th = jnp.asarray(_draws(ld, 8, SEED + 8))
    a = np.asarray(jax.jit(jax.vmap(cld.fn))(th))
    b = np.asarray(jax.jit(jax.vmap(ld.fn))(th))
    _close_logd(a, b, f"claim8 {kind} closure vs data path")


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_constants_do_not_grow(kind):
    """As ``test_adopted_tde.py``'s engine test: the largest HLO constant of
    ``jit(vmap(predict_jax))`` on numpy epochs (LC35, the map's gathers in it) is the same at 4
    and 64 parameter sets."""
    m = _model(kind, "fixed", True)
    lc = _lc(kind, "LC35")
    t, bidx = np.asarray(lc.time, float), m.predict_jax.band_index(np.asarray(lc.band))
    th = _draws(_ld(kind, "fixed", False, "LC35"), 64, SEED + 9)

    def largest(n):
        hlo = jax.jit(jax.vmap(lambda x: m.predict_jax(x, t, bidx))).lower(
            jnp.asarray(th[:n])).compile().as_text()
        shapes = re.findall(r"= [a-z]+\d*\[([\d,]*)\]\S* constant\(", hlo)
        return max(int(np.prod([int(d) for d in s.split(",") if d])) for s in shapes)

    small, large = largest(4), largest(64)
    _measured(f"claim8 {kind} largest constant", at_4=small, at_64=large)
    assert large == small and large < 64 * 1000, (small, large)


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_size_and_scratch(kind):
    """The ``log_density`` value program at 64 sets: the program text grows by at most 10 %, and
    XLA's scratch memory is not larger than the per-observation program's."""
    out = {}
    for ue in (True, False):
        ld = _ld(kind, "fixed", ue, "LC35")
        th = jnp.asarray(_draws(ld, 64, SEED + 10))
        compiled = jax.jit(jax.vmap(lambda x, d, ld=ld: ld.shared(x, d)[0],
                                    in_axes=(0, None))).lower(th, ld.data).compile()
        try:
            temp = int(compiled.memory_analysis().temp_size_in_bytes)
        except Exception:                                  # noqa: BLE001 - backend-dependent
            temp = None
        out[ue] = (len(compiled.as_text()), temp)
    ratio = out[True][0] / out[False][0]
    _measured(f"claim8 {kind} program text", ratio=f"{ratio:.3f}", ue=out[True][0],
              per_obs=out[False][0], scratch_ue=out[True][1], scratch_per_obs=out[False][1])
    assert ratio <= 1.10, ratio
    if out[True][1] is None or out[False][1] is None:
        pytest.skip("this backend reports no scratch memory")
    assert out[True][1] <= out[False][1], out


def test_host_core_compiles_once_per_shape():
    kind = "tde_fallback"
    m = _model(kind, "fixed", True, tag="_host_once")
    ctx = m.predict.__self__
    t, b = _grid("LC35")
    t2 = np.repeat(T0 + np.geomspace(5.0, 250.0, 35), 5)          # same (175 rows, 35 epochs)
    for times in (t, t2, t):
        m.predict(TRUTH[kind], times, b)
    assert ctx._jit_cache["epochs"]._cache_size() == 1


def test_lmo_reuse_flag_follows_epoch_bucket():
    """No compile needed: the flag is read off the program cache's key."""
    kind = "tde_fallback"
    m = _model(kind, "fixed", True, tag="_lmo")
    t60 = np.repeat(T0 + np.geomspace(3.0, 300.0, 60), 3)
    b60 = np.tile(np.array(BANDS[1:4]), 60)
    first, second, other = _lc(kind, "LC35"), _lc(kind, "LC35b"), _observe(kind, t60, b60, 53)
    flags = []
    for lc in (first, second, other):
        ld = A.log_density(lc, m)
        flags.append((_lmo_programs(m, lc, ld)[1], ld.epoch_bucket))
    assert flags == [(False, 48), (True, 48), (False, 64)], flags


@pytest.mark.slow
@pytest.mark.parametrize("geometric", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_fall_back_tdes_compile_in_under_a_minute_on_a_gpu(kind, geometric):
    """``test_compile_barrier``'s 480 x 480 test, 80 epochs x 6 bands: the concrete path, and
    ``log_density``'s vmapped ``value_and_grad`` with traced epochs and the map; with the search
    (``geometric=False``) and with the default ``geometric=True`` (claim 13)."""
    import time

    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("needs a GPU: this is the XLA GPU fusion autotuner")
    bands = BANDS + ["lssty"]
    fac = {"tde_fallback": wp.tde_fallback_model, "tde_mosfit": wp.tde_mosfit_model}[kind]
    m = fac(bands, Z, DL, t_exp_days=0.0, geometric=geometric)
    t = np.repeat(np.geomspace(2.0, 300.0, 80), 6)
    b = np.tile(np.array(bands), 80)
    th = jax.device_put(jnp.asarray(_draws(_ld(kind, "fixed", False, "LC35"), 480, 1)), device)
    bidx = m.predict_jax.band_index(b)
    start = time.perf_counter()
    compiled = jax.jit(jax.vmap(lambda x: m.predict_jax(x, t, bidx))).lower(th).compile()
    concrete = time.perf_counter() - start
    assert concrete < 60.0, f"{kind}: compiling the concrete batch took {concrete:.0f} s"
    assert np.all(np.isfinite(np.asarray(compiled(th))))
    flux = np.asarray(m.predict(TRUTH[kind], t, b))
    lc = wp.LightCurve(time=t, band=b, flux=flux, flux_err=0.05 * flux)
    ld = A.log_density(lc, m)
    f = jax.jit(jax.vmap(jax.value_and_grad(lambda x, d: ld.shared(x, d)[0]), in_axes=(0, None)))
    start = time.perf_counter()
    compiled = f.lower(th, ld.data).compile()
    elapsed = time.perf_counter() - start
    _measured(f"claim8 {kind} geometric={geometric} GPU compile", concrete_s=f"{concrete:.1f}",
              log_density_value_and_grad_s=f"{elapsed:.1f}")
    assert elapsed < 60.0, f"{kind}: compiling log_density's value_and_grad took {elapsed:.0f} s"
    v, g = compiled(th, ld.data)
    assert np.all(np.isfinite(np.asarray(v)))


# =============================================================================================
# 9. a wrong map is loud
# =============================================================================================

@pytest.mark.parametrize("kind", KINDS)
def test_wrong_map_is_nan(kind):
    ld_a, ld_b = (_ld(kind, "fixed", True, nm) for nm in ("LC35", "LC35b"))
    t44 = np.repeat(T0 + np.geomspace(3.0, 300.0, 44), 4)[:175]     # another pattern, 175 rows
    b44 = np.tile(np.array(BANDS[:4]), 44)[:175]
    ld_c = A.log_density(_observe(kind, t44, b44, SEED + 54), _model(kind, "fixed", True),
                         bucket=192, epoch_bucket=48)
    assert ld_c.n_distinct_epochs == 44
    th = jnp.asarray(_draws(ld_a, 4, SEED + 15))
    f = jax.jit(jax.vmap(ld_a.shared, in_axes=(0, None)))
    good = np.asarray(f(th, ld_a.data)[0])
    assert np.all(np.isfinite(good))
    swapped = {**ld_a.data, "epoch_inv": ld_c.data["epoch_inv"],
               "epoch_first": ld_c.data["epoch_first"]}
    assert np.all(np.isnan(np.asarray(f(th, swapped)[0])))
    rng = np.random.default_rng(SEED + 16)
    scrambled = {**ld_a.data, "epoch_inv": jnp.asarray(rng.permutation(
        np.asarray(ld_a.data["epoch_inv"])), jnp.int32)}
    assert np.all(np.isnan(np.asarray(f(th, scrambled)[0])))
    # LC35b repeats each epoch in the same five bands, so LC35's map describes its epochs too
    same = {**ld_b.data, "epoch_first": ld_a.data["epoch_first"],
            "epoch_inv": ld_a.data["epoch_inv"]}
    np.testing.assert_array_equal(np.asarray(f(th, same)[0]), np.asarray(f(th, ld_b.data)[0]))


def _assemble(kind, unique_epochs):
    return AT.assemble(
        kind, kind, BANDS, Z, DL, name="x", n_wave=None, pin=None, prior=None, mag_floor=40.0,
        dilation=True, ebv_mw=None, ebv_host=None, r_v_mw=3.1, r_v_host=3.1, law="f99",
        filter_set=None, band_aliases=None, default_system=None, t_exp_days=0.0, free=None,
        redshift_prior=None, constraint=None, engine_kw={}, summary="x", caller="test",
        unique_epochs=unique_epochs)


def test_errors():
    m1 = _model("tde_fallback", "fixed", True)
    m0 = _model("tde_fallback", "fixed", False)
    t, b = _grid("LC21")
    bidx = m1.predict_jax.band_index(b)
    th = jnp.asarray(_draws(_ld("tde_fallback", "fixed", False, "LC35"), 1))[0]
    first, inv = m1.predict_jax.epoch_index(t)
    with pytest.raises(ValueError, match="epoch_index"):                  # concrete, wrong
        m1.predict_jax(th, t, bidx, epoch_idx=(first, np.roll(inv, 1)))
    with pytest.raises(ValueError, match="epoch_index"):                  # out of range
        m1.predict_jax(th, t, bidx, epoch_idx=(first + 1000, inv))
    with pytest.raises(ValueError, match="epoch_index"):                  # one entry per row
        m1.predict_jax(th, t, bidx, epoch_idx=(first, inv[:-1]))
    with pytest.raises(ValueError, match=re.escape(repr(m0.name))):      # a ctx without the option
        m0.predict.__self__.predict_jax(th, t, bidx, epoch_idx=(first, inv))
    for m in (m0, wp.tde_analytical_model(BANDS, Z, DL, constraint=None)):
        assert type(m.predict_jax) is _PredictJax
        with pytest.raises(TypeError):                                    # a plain predict_jax
            m.predict_jax(th, t, bidx, epoch_idx=(first, inv))
    toy = _toy_model(_ToyHookNoKeyword)
    with pytest.raises(ValueError, match="carries epoch_index but its call does not take"):
        A.log_density(_toy_curves()[0], toy)
    with pytest.raises(ValueError, match="unique_epochs=False"):
        _assemble("tde_analytical", True)
    assert _assemble("tde_analytical", False).predict_jax.__class__ is _PredictJax
    lc = _lc("tde_fallback", "LC35")
    for eb in (34, 193):
        with pytest.raises(ValueError, match=r"epoch_bucket=\d+ must lie between .*35.* 192"):
            A.log_density(lc, m1, epoch_bucket=eb)
    with pytest.raises(ValueError, match="epoch_bucket must be"):
        A.log_density(lc, m1, epoch_bucket="exact")
    ld = A.log_density(lc, m1, epoch_bucket=192)
    assert ld.epoch_bucket == 192 and ld.data["epoch_first"].shape == (192,)


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_guard_is_quiet_under_debug_nans(kind):
    """``jax_debug_nans`` raises at the first NaN an operation makes: the guard makes none for a
    right map, op by op (``disable_jit``), eager-vmapped over a free t_exp, and jitted. ``slow``:
    the first op-by-op run compiles every operation alone (52 s on the CPU)."""
    m = _model(kind, "free", True)
    t = np.repeat(T0 + np.array([5.0, 10.0, 20.0, 40.0]), 3)
    b = np.tile(np.array(BANDS[1:4]), 4)
    bidx = m.predict_jax.band_index(b)
    x = np.array([TRUTH[kind][k] for k in m.parameters[:-2]] + [T0 - 1.0, Z])
    shifts = jnp.asarray([T0 - 3.0, T0 - 2.0, T0 - 1.0])
    j = m.parameters.index("t_exp")
    with jax.debug_nans(True):
        with jax.disable_jit():
            f_host = m.predict(dict(zip(m.parameters, x)), t, b)
            f_vmap = jax.vmap(lambda s: m.predict_jax(jnp.asarray(x).at[j].set(s), t, bidx))(
                shifts)
        f_jit = jax.jit(lambda s: m.predict_jax(jnp.asarray(x).at[j].set(s), t, bidx))(
            shifts[2])
    assert np.all(np.isfinite(f_host)) and np.all(np.isfinite(np.asarray(f_vmap)))
    np.testing.assert_allclose(np.asarray(f_jit), f_host, rtol=1e-10)
    # a wrong map: NaN, which jax_debug_nans reports where it is made
    ctx = m.predict.__self__
    rel = jnp.asarray(t - 0.0)
    first, inv = AT.epoch_index(t)
    wrong = (jnp.asarray(first), jnp.asarray(np.roll(inv, 1)))
    free = [jnp.asarray(v) for v in x]
    out = jax.jit(lambda: ctx._physics_call(rel, jnp.asarray(bidx), free, epoch_idx=wrong))()
    assert np.all(np.isnan(np.asarray(out)))
    with jax.debug_nans(True), jax.disable_jit(), pytest.raises(FloatingPointError):
        ctx._physics_call(rel, jnp.asarray(bidx), free, epoch_idx=wrong)


# =============================================================================================
# 10. nothing else moves
# =============================================================================================

#: ``log_density``'s data keys for a model that evaluates every observation, as before.
PLAIN_KEYS = {"t", "band", "y", "sigma", "sigma_ul", "mask", "w_det", "w_ul", "log_norm",
              "prior", "fixed"}


class _Recorder:
    """A ``predict_jax`` that records the keywords it is called with."""

    def __init__(self, inner):
        self.inner = inner
        self.keywords = []
        for attr in ("band_index", "constraint_ok", "mag_floor", "check_epochs"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    def __call__(self, *args, **kwargs):
        self.keywords.append(sorted(kwargs))
        return self.inner(*args, **kwargs)


def _others():
    return {
        "tde_analytical": wp.tde_analytical_model(BANDS, Z, DL, constraint=None, t_exp_days=0.0),
        "tde_stream_stream": wp.tde_stream_stream_model(BANDS, Z, DL),
        "tde_fallback_per_obs": _model("tde_fallback", "fixed", False, t_exp_days=0.0),
        "tde_mosfit_per_obs": _model("tde_mosfit", "fixed", False, t_exp_days=0.0),
        "arnett": wp.supernova_model("arnett", BANDS, Z, DL, constraint=None),
        "flare_jax": wp.get_model("flare_jax"),
        "kilonova_one": wp.kilonova_model(BANDS, Z, DL),
    }


@pytest.mark.parametrize("name", ["tde_analytical", "tde_stream_stream", "tde_fallback_per_obs",
                                  "tde_mosfit_per_obs", "arnett", "flare_jax", "kilonova_one"])
def test_other_families_untouched(name):
    m = _others()[name]
    pj = m.predict_jax
    assert getattr(pj, "epoch_index", None) is None
    if name.startswith("tde_"):                          # the families `assemble` builds
        ctx = m.predict.__self__
        assert isinstance(ctx, AT.DirectTDEPredict)
        assert type(pj) is _PredictJax
        assert "unique_epochs" not in vars(ctx) and "geometric" not in ctx.common
        assert pj.__doc__ == AT.DirectTDEPredict.predict_jax.__doc__
    t = np.repeat(np.geomspace(3.0, 60.0, 8), 3)
    b = np.tile(np.array(BANDS[1:4]), 8)
    lc = wp.LightCurve(time=t, band=b, flux=np.full(t.size, 1e-5), flux_err=np.full(t.size, 1e-6))
    rec = _Recorder(pj)
    copy = wp.Model(name=m.name + "_rec", predict=m.predict, parameters=list(m.parameters),
                    default_prior=m.default_prior, predict_jax=rec)
    ld = A.log_density(lc, copy)
    assert set(ld.data) == PLAIN_KEYS
    assert ld.n_distinct_epochs is None and ld.epoch_bucket is None
    assert "distinct epochs" not in repr(ld)
    th = jnp.asarray(_draws(ld, 4, SEED + 17))
    jax.eval_shape(jax.vmap(ld.fn), th)
    assert rec.keywords and all("epoch_idx" not in k for k in rec.keywords), rec.keywords
    plain = A.log_density(lc, m)
    assert set(plain.data) == PLAIN_KEYS and plain.n_distinct_epochs is None


# =============================================================================================
# 11. pickling and worker processes
# =============================================================================================

def _predict_rows(m, kind, n, seed):
    """``m.predict`` on LC35's rows for ``n`` draws of the per-observation density's prior."""
    t, b = _grid("LC35")
    th = _draws(_ld(kind, "fixed", False, "LC35"), n, seed)
    return np.stack([np.asarray(m.predict(dict(zip(m.parameters, x)), t, b), dtype=float)
                     for x in th])


@pytest.mark.parametrize("kind", KINDS)
def test_pickle_round_trip(kind):
    """A pickled model keeps the hook, its ``geometric`` and its numbers; an object pickled before
    the options existed (no ``unique_epochs`` on its context, no ``geometric`` in its state, a
    plain ``_PredictJax``) takes the per-observation path with the search: the reference's bits."""
    import pickle

    m = _model(kind, "fixed", True, tag="_pickle")
    ref = _predict_rows(m, kind, 4, SEED + 60)
    back = pickle.loads(pickle.dumps(m))
    assert type(back.predict_jax) is AT._DirectPredictJax
    assert back.predict_jax.epoch_index is AT.epoch_index
    ctx = back.predict.__self__
    assert ctx.unique_epochs is True and ctx._jit_cache is None    # no compiled core pickled
    assert ctx.common["geometric"] is True
    new = _predict_rows(back, kind, 4, SEED + 60)
    assert np.array_equal(new, ref)                                 # the same code, the same shapes
    assert list(ctx._jit_cache) == ["epochs"]
    # as pickled before the options existed: no attribute, no key, a plain predict_jax
    old = pickle.loads(pickle.dumps(m))
    octx = old.predict.__self__
    del octx.unique_epochs
    del octx.common["geometric"]
    opj = _PredictJax(octx)
    assert getattr(opj, "epoch_index", None) is None
    flux_old = _predict_rows(old, kind, 4, SEED + 60)
    assert list(octx._jit_cache) == [None]                          # the per-observation core
    per_obs = _predict_rows(_model(kind, "fixed", False), kind, 4, SEED + 60)
    assert np.array_equal(flux_old, per_obs)
    _close_flux(flux_old, ref, f"claim11 {kind} pickled before the options vs the default")
    t, b = _grid("LC35")
    x = jnp.asarray(_draws(_ld(kind, "fixed", False, "LC35"), 1))[0]
    with pytest.raises(TypeError):                         # a plain predict_jax, as before
        opj(x, t, opj.band_index(b), epoch_idx=AT.epoch_index(t))
    with pytest.raises(ValueError, match="takes no epoch_idx"):
        octx.predict_jax(x, t, opj.band_index(b), epoch_idx=AT.epoch_index(t))


def _spawned_predict(jobs):
    """Run in a ``spawn`` worker, as the CPU samplers' ``n_jobs`` pools run a model: for each
    pickled ``(model, parameters, times, bands)``, the flux, whether ``predict_jax`` carries the
    hook, and the keys of the cores the worker compiled."""
    import jax as jax_

    jax_.config.update("jax_enable_x64", True)
    out = []
    for model, params, times, bands in jobs:
        flux = np.asarray(model.predict(params, times, bands), dtype=float)
        ctx = model.predict.__self__
        out.append((flux, getattr(model.predict_jax, "epoch_index", None) is not None,
                    sorted(map(str, ctx._jit_cache or {}))))
    return out


def test_spawn_worker_predict_equals_parent():
    """One ``spawn`` worker (the CPU samplers' ``n_jobs`` path) evaluates both models on LC35, two
    parameter sets each: the parent's fluxes, and the worker builds the ``"epochs"`` core."""
    import multiprocessing
    import time

    t, b = _grid("LC35")
    jobs, parent = [], []
    for kind in KINDS:
        m = _model(kind, "fixed", True, tag="_spawn")
        for x in _draws(_ld(kind, "fixed", False, "LC35"), 2, SEED + 61):
            p = dict(zip(m.parameters, map(float, x)))
            jobs.append((m, p, t, b))
            parent.append(np.asarray(m.predict(p, t, b), dtype=float))
    start = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        got = pool.apply(_spawned_predict, (jobs,))
    elapsed = time.perf_counter() - start
    _measured("claim11 spawn worker", seconds=f"{elapsed:.1f}")
    for (flux, hook, keys), ref in zip(got, parent):
        assert hook and keys == ["epochs"], (hook, keys)
        frac, d = _close_flux(flux, ref, "claim11 spawn worker vs parent")
        assert frac == 1.0, frac                                     # the same code and shapes


# =============================================================================================
# 12. float32
# =============================================================================================

def test_float32_session_still_raises_float64_on_every_new_path():
    """x64 off (``jax.clear_caches()`` around the toggle), 3 epochs x 2 bands: every path the map
    adds raises the float64 error, as the per-observation paths do; ``epoch_index`` itself is
    numpy. On a clock whose zero is 0 ``log_density`` takes the data path and raises when it
    traces the program; on an MJD clock float32 cannot hold the epochs, so it closes them over
    and the program raises when it is called."""
    from whisper_cbpf.likelihood import make_likelihood

    LMO = _lmo()
    ep = np.repeat(np.array([5.0, 20.0, 60.0]), 2)
    b = np.tile(np.array(BANDS[1:3]), 3)

    def curve(t):
        return wp.LightCurve(time=t, band=b, flux=np.full(t.size, 1e-5),
                             flux_err=np.full(t.size, 1e-6))

    jax.clear_caches()
    jax.config.update("jax_enable_x64", False)
    try:
        for kind in KINDS:
            fac = {"tde_fallback": wp.tde_fallback_model, "tde_mosfit": wp.tde_mosfit_model}[kind]
            m = fac(BANDS[1:3], Z, DL, name=f"{kind}_f32", t_exp_days=0.0)
            pj = m.predict_jax
            first, inv = pj.epoch_index(ep)
            assert first.dtype == inv.dtype == np.int32 and first.tolist() == [0, 2, 4]
            params = {k: float(TRUTH[kind][k]) for k in m.parameters}
            with pytest.raises(RuntimeError, match="float64"):
                m.predict(params, ep, b)
            bidx = pj.band_index(b)
            th = jnp.asarray([params[k] for k in m.parameters])
            with pytest.raises(RuntimeError, match="float64"):
                jax.jit(lambda x: pj(x, ep, bidx, epoch_idx=(first, inv)))(th)
            with pytest.raises(RuntimeError, match="float64"):          # the data path
                A.log_density(curve(ep), m)
            lik = make_likelihood(curve(ep), kind="auto", space="auto")
            names = list(m.parameters)
            with pytest.raises(RuntimeError, match="float64"):
                LMO._jax_programs(curve(ep), m, LMO._Box(m.default_prior, names), names, lik)
            m_mjd = fac(BANDS[1:3], Z, DL, name=f"{kind}_f32_mjd", t_exp_days=T0)
            ld = A.log_density(curve(T0 + ep), m_mjd)                  # epochs closed over
            assert not ld.data_as_argument and "epoch_first" not in ld.data
            with pytest.raises(RuntimeError, match="float64"):
                ld.fn(th)
    finally:
        jax.clear_caches()
        jax.config.update("jax_enable_x64", True)


# =============================================================================================
# 13. (stage B) the geometric segment search
# =============================================================================================

def _prims(closed):
    """Every primitive name in ``closed`` and the jaxprs inside it."""
    found = set()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            found.add(eqn.primitive.name)
            stack = list(eqn.params.values())
            while stack:
                p = stack.pop()
                if hasattr(p, "jaxpr"):
                    walk(p.jaxpr)
                elif hasattr(p, "eqns"):
                    walk(p)
                elif isinstance(p, (list, tuple)):
                    stack.extend(p)

    walk(closed.jaxpr)
    return found


#: 48 source-frame epochs, days since the first fall-back, for the kernel-level test.
KERNEL_EPOCHS = np.geomspace(0.5, 400.0, 48)


@functools.lru_cache(maxsize=None)
def _kernel(geometric, n=200):
    """``viscous_log10`` (log10 L at 48 epochs) on the engine of ``n`` ``tde_fallback`` prior
    draws, on the dense grid of :func:`geometric_dense_times`, and its gradient (of the sum over
    the epochs) in ``mbh6, mstar, tvisc, bb, eta``; jitted, vmapped."""
    TF = importlib.import_module("whisper_cbpf.models.jax.tde_fallback")
    t = jnp.asarray(KERNEL_EPOCHS)

    def one(p):
        dense = AT.geometric_dense_times(jnp.max(t))
        log_l, _, _, _ = TF.fallback_engine(dense, p[0], p[1], p[3], p[4], 1.0)
        return TF.viscous_log10(t, dense, log_l, p[2], geometric=geometric)

    th = _draws(_ld("tde_fallback", "fixed", False, "LC35"), n, SEED + 70)[:, :5]
    v = _blocks(jax.jit(jax.vmap(one)), th)
    g = _blocks(jax.jit(jax.vmap(jax.grad(lambda p: jnp.sum(one(p))))), th)
    return v, g


def test_geometric_kernel_is_the_interpolant():
    """``viscous_log10(geometric=True)`` against the search, 200 prior draws x 48 epochs."""
    v1, g1 = _kernel(True)
    v0, g0 = _kernel(False)
    assert np.all(np.isfinite(v0)) and np.array_equal(np.isfinite(g1), np.isfinite(g0))
    d = float(np.max(np.abs(v1 - v0)))
    _measured("claim13 kernel log10 L", bitwise=f"{_bits(v1, v0):.4f}", max_ddex=f"{d:.3g}")
    assert d <= 1e-12, d
    _close_grad(g1, g0, "claim13 kernel gradient")


@functools.lru_cache(maxsize=None)
def _physics_geo(kind, variant, geometric, n):
    """As :func:`_physics` (LC35, fixed z, the model's own context), on ``n`` draws, with
    ``geometric`` passed to the physics function: ``variant`` "none" or E (the map's size)."""
    m = _model(kind, "fixed", False)
    ctx = m.predict.__self__
    t, b = _grid("LC35")
    rel = jnp.asarray(t - T0)
    bidx = jnp.asarray(m.predict_jax.band_index(b))
    k = len(m.parameters)
    extra = {"geometric": bool(geometric)}
    if variant != "none":
        f, i = (jnp.asarray(a, jnp.int32) for a in AT.epoch_index(t - T0, int(variant)))
        extra["epoch_idx"] = (f, i)

    def one(x):
        return ctx._physics_call(rel, bidx, [x[j] for j in range(k)], **extra)

    th = _draws(_ld(kind, "fixed", False, "LC35"), n, SEED + 71)
    return _blocks(jax.jit(jax.vmap(one)), th)


def _check_geometric_magnitudes(kind, n):
    for variant in (35, "none"):          # with the distinct-epoch map, and per observation
        new = _physics_geo(kind, variant, True, n)
        ref = _physics_geo(kind, variant, False, n)
        assert np.all(np.isfinite(ref))
        _mags_close(new, ref, f"claim13 {kind} magnitudes geometric (map={variant}, {n} draws)")


@functools.lru_cache(maxsize=None)
def _vg_geo(kind, config, geometric, n=48):
    """:func:`_vg` of the unique-epoch model: the default (``geometric=True``) or built with
    ``geometric=False``."""
    ld = (_ld(kind, config, True, "LC35") if geometric
          else _ld(kind, config, True, "LC35", geometric=False))
    th = _draws(_ld(kind, config, False, "LC35"), n, SEED + 72)
    f = jax.jit(jax.vmap(jax.value_and_grad(lambda x, d: ld.shared(x, d), has_aux=True),
                         in_axes=(0, None)))
    (v, ll), g = f(jnp.asarray(th), ld.data)
    return (np.asarray(v, dtype=float), np.asarray(ll, dtype=float),
            np.asarray(g, dtype=float))


@pytest.mark.parametrize("kind", KINDS)
def test_geometric_magnitudes(kind):
    """``*_ab_magnitude(geometric=True)`` against the search (50 draws x 175 rows, with the map
    and per observation), and the default factories against ``geometric=False``: the log
    posterior and its gradient through ``log_density`` (48 draws, fixed redshift)."""
    _check_geometric_magnitudes(kind, 50)
    v1, _, g1 = _vg_geo(kind, "fixed", True)
    v0, ll0, g0 = _vg_geo(kind, "fixed", False)
    scale = _scale(ll0, float(np.asarray(_ld(kind, "fixed", False, "LC35").data["log_norm"])))
    _close_logd(v1, v0, f"claim13 {kind} fixed log posterior geometric", scale)
    _close_grad(g1, g0, f"claim13 {kind} fixed gradient geometric")


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_geometric_magnitudes_200_draws(kind):
    """The 200-draw sweep; host ``predict`` of the default model with free t_exp and redshift
    (200 draws, against the unique-epoch model with ``geometric=False`` and the per-observation
    reference); and the free config through ``log_density``."""
    _check_geometric_magnitudes(kind, 200)
    host = _path(kind, "free", True, "predict", 200)
    _close_flux(host, _path(kind, "free", True, "predict", 200, geometric=False),
                f"claim13 {kind} free host predict geometric vs unique epochs")
    _close_flux(host, _path(kind, "free", False, "predict", 200),
                f"claim13 {kind} free host predict geometric vs per observation")
    v1, _, g1 = _vg_geo(kind, "free", True)
    v0, ll0, g0 = _vg_geo(kind, "free", False)
    scale = _scale(ll0, float(np.asarray(_ld(kind, "free", False, "LC35").data["log_norm"])))
    _close_logd(v1, v0, f"claim13 {kind} free log posterior geometric", scale)
    _close_grad(g1, g0, f"claim13 {kind} free gradient geometric")


def test_generic_kernels_keep_interp():
    """Which paths find segments by arithmetic (``floor``). The default of ``viscous_log10``
    (False: it takes the caller's grid), the wall, and ``tde_fallback_bolometric_log10``,
    ``tde_mosfit_curve``, ``tde_mosfit_constraint_score``, ``tde_mosfit_flux_density_mjy`` and the
    factories given ``geometric=False`` keep the search, to the bit; the defaults of those
    functions and of the factories find the segments by arithmetic, each its own composition."""
    TF = importlib.import_module("whisper_cbpf.models.jax.tde_fallback")
    TM = importlib.import_module("whisper_cbpf.models.jax.tde_mosfit")
    t = jnp.asarray(KERNEL_EPOCHS[::2])
    th = _draws(_ld("tde_fallback", "fixed", False, "LC35"), 20, SEED + 73)[:, :5]

    def kernel(p, **kw):
        dense = AT.geometric_dense_times(jnp.max(t))
        log_l, _, _, _ = TF.fallback_engine(dense, p[0], p[1], p[3], p[4], 1.0)
        return TF.viscous_log10(t, dense, log_l, p[2], **kw)

    def bolo(p, **kw):
        return TF.tde_fallback_bolometric_log10(t, p[0], p[1], p[2], p[3], p[4], 1.0, **kw)

    run = functools.partial(_blocks, width=20)
    v_default = run(jax.jit(jax.vmap(kernel)), th)
    assert np.array_equal(v_default, run(jax.jit(jax.vmap(
        lambda p: kernel(p, geometric=False))), th))
    # the bolometric function is its own composition: the search with geometric=False, the
    # arithmetic search by default
    assert np.array_equal(run(jax.jit(jax.vmap(lambda p: bolo(p, geometric=False))), th),
                          v_default)
    assert np.array_equal(run(jax.jit(jax.vmap(bolo)), th),
                          run(jax.jit(jax.vmap(lambda p: kernel(p, geometric=True))), th))
    tm = _draws(_ld("tde_mosfit", "fixed", False, "LC35"), 20, SEED + 74)

    def curve(p, **kw):
        return jnp.stack(TM.tde_mosfit_curve(t, *[p[i] for i in range(7)], **kw)[:3])

    assert np.array_equal(run(jax.jit(jax.vmap(curve)), tm),
                          run(jax.jit(jax.vmap(lambda p: curve(p, geometric=True))), tm))

    def score(p, **kw):                                  # the score's own composition
        _, _, rad, _, _, _, rp = TM.tde_mosfit_curve(t, *[p[i] for i in range(7)], **kw)
        a, r = TM.tde_mosfit_constraint_terms(t, rad, p[0], rp)
        return -10.0 * a ** 3 - 10.0 * r ** 2

    for kw, geo in (({}, True), ({"geometric": False}, False)):
        assert np.array_equal(
            run(jax.jit(jax.vmap(lambda p, kw=kw: TM.tde_mosfit_constraint_score(
                t, *[p[i] for i in range(7)], **kw))), tm),
            run(jax.jit(jax.vmap(lambda p, geo=geo: score(p, geometric=geo))), tm)), kw
    p0 = jnp.asarray(th[0])
    q0 = jnp.asarray(tm[0])
    wall = {k: q0[i] for i, k in enumerate(TM.PARAMETERS[:7])}
    wall["Leddlim"] = 1.0
    nu = jnp.full(t.shape, 5e14)

    def flux(q, **kw):
        return TM.tde_mosfit_flux_density_mjy(t, nu, Z, DL, *[q[i] for i in range(7)], **kw)

    def score_fn(q, **kw):
        return TM.tde_mosfit_constraint_score(t, *[q[i] for i in range(7)], **kw)

    searching = {
        "viscous_log10": jax.make_jaxpr(kernel)(p0),
        "tde_mosfit_constraint": jax.make_jaxpr(lambda: TM.tde_mosfit_constraint(wall, None,
                                                                                None))(),
        "tde_fallback_bolometric_log10(geometric=False)": jax.make_jaxpr(
            lambda p: bolo(p, geometric=False))(p0),
        "tde_mosfit_constraint_score(geometric=False)": jax.make_jaxpr(
            lambda q: score_fn(q, geometric=False))(q0),
        "tde_mosfit_flux_density_mjy(geometric=False)": jax.make_jaxpr(
            lambda q: flux(q, geometric=False))(q0),
        "tde_mosfit_curve(geometric=False)": jax.make_jaxpr(
            lambda q: curve(q, geometric=False))(q0),
    }
    arithmetic = {
        "tde_fallback_bolometric_log10": jax.make_jaxpr(bolo)(p0),
        "tde_mosfit_constraint_score": jax.make_jaxpr(score_fn)(q0),
        "tde_mosfit_flux_density_mjy": jax.make_jaxpr(flux)(q0),
    }
    for name, closed in arithmetic.items():
        assert "floor" in _prims(closed), name
    lc35 = {kind: _lc(kind, "LC35") for kind in KINDS}
    for kind in KINDS:
        for geo in (None, False):                        # the default, and the search
            m = (_model(kind, "fixed", True) if geo is None
                 else _model(kind, "fixed", True, geometric=False))
            lc = lc35[kind]
            tt, bidx = np.asarray(lc.time, float), m.predict_jax.band_index(np.asarray(lc.band))
            x = jnp.asarray(_draws(_ld(kind, "fixed", False, "LC35"), 1))[0]
            closed = jax.make_jaxpr(lambda y, m=m: m.predict_jax(y, tt, bidx))(x)
            if geo is None:
                assert "floor" in _prims(closed), kind
            else:
                searching[f"{kind} model, geometric=False"] = closed
    for name, closed in searching.items():
        assert "floor" not in _prims(closed), name
    assert "floor" in _prims(jax.make_jaxpr(lambda p: kernel(p, geometric=True))(p0))
    assert "floor" in _prims(jax.make_jaxpr(curve)(q0))


@pytest.mark.parametrize("kind", KINDS)
def test_geometric_state_only_when_asked(kind):
    """``geometric`` enters the model's state only when True, the default: a model built with
    ``geometric=False`` keeps the state and the ``model_record`` identity it had before the option
    existed (as ``unique_epochs=False`` does, claim 10), and the default model is identified as
    geometric."""
    from whisper_cbpf.results import model_record

    fac = {"tde_fallback": wp.tde_fallback_model, "tde_mosfit": wp.tde_mosfit_model}[kind]
    built = {label: fac(BANDS, Z, DL, name=f"{kind}_state", **kw) for label, kw in (
        ("default", {}), ("on", {"geometric": True}), ("off", {"geometric": False}),
        ("per_obs", {"unique_epochs": False}),
        ("per_obs_on", {"unique_epochs": False, "geometric": True}),
        ("per_obs_off", {"unique_epochs": False, "geometric": False}))}
    common = {label: m.predict.__self__.common for label, m in built.items()}
    for label, c in common.items():
        if label.endswith("off"):
            assert "geometric" not in c, label
        else:
            assert c["geometric"] is True, label
    ids = {label: model_record(m)["identity_hash"] for label, m in built.items()}
    assert ids["default"] == ids["on"] != ids["off"]
    assert ids["per_obs"] == ids["per_obs_on"] != ids["per_obs_off"]
    assert len({ids["default"], ids["off"], ids["per_obs"], ids["per_obs_off"]}) == 4


def test_defaults_are_the_documented_ones():
    """Read with ``inspect.signature``: both factories default to ``unique_epochs=True,
    geometric=True``, the physics functions that build their own dense grid (the bolometric
    function, both ``*_ab_magnitude``, ``tde_mosfit_curve``, the score and the flux density) to
    ``geometric=True``, and ``viscous_log10``, which takes the caller's grid, to False; the wall
    takes no ``geometric``. Each docstring states its default, and so do the factories'
    signatures in ``docs/API_REFERENCE.md``."""
    import inspect
    from pathlib import Path

    TF = importlib.import_module("whisper_cbpf.models.jax.tde_fallback")
    TM = importlib.import_module("whisper_cbpf.models.jax.tde_mosfit")

    def default(fn, name):
        return inspect.signature(fn).parameters[name].default

    def text(fn):
        return " ".join(fn.__doc__.split())

    api = (Path(__file__).resolve().parents[1] / "docs" / "API_REFERENCE.md").read_text(
        encoding="utf-8")
    for fac in (wp.tde_fallback_model, wp.tde_mosfit_model):
        assert default(fac, "unique_epochs") is True and default(fac, "geometric") is True
        assert "geometric : bool Default True:" in text(fac), fac.__name__
        assert "unique_epochs : bool" in text(fac) and "(default True;" in text(fac)
        line = next(ln for ln in api.splitlines() if ln.startswith(f"- **`{fac.__name__}("))
        assert line.rstrip().endswith("unique_epochs=True, geometric=True)`**"), line
    for fn in (TF.tde_fallback_bolometric_log10, TF.tde_fallback_ab_magnitude,
               TM.tde_mosfit_ab_magnitude, TM.tde_mosfit_curve, TM.tde_mosfit_constraint_score,
               TM.tde_mosfit_flux_density_mjy):
        assert default(fn, "geometric") is True, fn.__name__
        assert "``geometric`` (default True)" in text(fn), fn.__name__
    assert default(TF.viscous_log10, "geometric") is False
    assert "``geometric=False`` (default)" in text(TF.viscous_log10)
    assert "geometric" not in inspect.signature(TM.tde_mosfit_constraint).parameters


@pytest.mark.parametrize("kind", KINDS)
def test_geometric_needs_two_dense_nodes(kind):
    """``geometric=True`` with fewer than 2 dense nodes raises ``ValueError`` naming the fix: at the
    factory call (``dense_resolution`` 1, 0 and -1) and in ``viscous_log10`` itself, so in every
    function that reaches it (with one node the arithmetic search divided by ``n - 1 = 0`` and
    gave NaN for every observation). ``geometric=False`` with one node is untouched: it builds,
    and host ``predict`` and the functions give finite numbers, as before the option; 2 nodes is
    enough for the default."""
    TF = importlib.import_module("whisper_cbpf.models.jax.tde_fallback")
    TM = importlib.import_module("whisper_cbpf.models.jax.tde_mosfit")
    fac = {"tde_fallback": wp.tde_fallback_model, "tde_mosfit": wp.tde_mosfit_model}[kind]
    for n in (1, 0, -1):
        for kw in ({}, {"geometric": True}, {"unique_epochs": False, "geometric": True}):
            with pytest.raises(ValueError, match="at least 2 dense nodes"):
                fac(BANDS, Z, DL, dense_resolution=n, **kw)
    t, b = _grid("LC21")
    th = _draws(_ld(kind, "fixed", False, "LC35"), 2, SEED + 77)
    for n, kw in ((1, {"unique_epochs": False, "geometric": False}),
                  (1, {"geometric": False}), (2, {})):
        m = fac(BANDS, Z, DL, name=f"{kind}_nodes", t_exp_days=T0, dense_resolution=n, **kw)
        flux = np.stack([np.asarray(m.predict(dict(zip(m.parameters, x)), t, b)) for x in th])
        assert np.all(np.isfinite(flux)), (n, kw)
    tau = jnp.asarray(KERNEL_EPOCHS[:5])
    if kind == "tde_fallback":
        one = jnp.asarray([3.0])
        with pytest.raises(ValueError, match="at least 2 dense nodes"):
            TF.viscous_log10(tau, one, jnp.asarray([43.0]), 1.0, geometric=True)
        assert np.all(np.isfinite(np.asarray(TF.viscous_log10(tau, one, jnp.asarray([43.0]),
                                                              1.0))))
        p = TRUTH["tde_fallback"]
        args = (p["mbh6"], p["mstar"], p["tvisc"], p["bb"], p["eta"], 1.0)
        calls = {"tde_fallback_bolometric_log10": lambda **kw: TF.tde_fallback_bolometric_log10(
            tau, *args, dense_resolution=1, **kw)}
    else:
        q = [TRUTH["tde_mosfit"][k] for k in TM.PARAMETERS[:7]]
        calls = {
            "tde_mosfit_curve": lambda **kw: TM.tde_mosfit_curve(tau, *q, dense_resolution=1,
                                                                 **kw)[0],
            "tde_mosfit_constraint_score": lambda **kw: TM.tde_mosfit_constraint_score(
                tau, *q, dense_resolution=1, **kw),
            "tde_mosfit_flux_density_mjy": lambda **kw: TM.tde_mosfit_flux_density_mjy(
                tau, jnp.full(tau.shape, 5e14), Z, DL, *q, dense_resolution=1, **kw),
        }
    for name, call in calls.items():
        with pytest.raises(ValueError, match="at least 2 dense nodes"):
            call()
        assert np.all(np.isfinite(np.asarray(call(geometric=False)))), name


def test_viscous_log10_keeps_the_search_by_default():
    """``viscous_log10`` takes the caller's ``dense_times``, so its default stays the search:
    on a valid grid that is not geometric (linear from 1e-5 d to the last epoch + 100 d, 1000
    nodes) ``geometric=True`` picks wrong segments, while the default is ``geometric=False`` to
    the bit; on the grid of ``geometric_dense_times`` the two agree (claim 13's kernel test)."""
    TF = importlib.import_module("whisper_cbpf.models.jax.tde_fallback")
    t = jnp.asarray(KERNEL_EPOCHS)
    th = _draws(_ld("tde_fallback", "fixed", False, "LC35"), 20, SEED + 76)[:, :5]

    def kernel(p, **kw):
        dense = jnp.linspace(AT.GRID_START, jnp.max(t) + AT.GRID_PAD, N_KERNEL)
        log_l, _, _, _ = TF.fallback_engine(dense, p[0], p[1], p[3], p[4], 1.0)
        return TF.viscous_log10(t, dense, log_l, p[2], **kw)

    x = jnp.asarray(th)
    v_def = np.asarray(jax.jit(jax.vmap(kernel))(x))
    assert np.all(np.isfinite(v_def))
    assert np.array_equal(v_def, np.asarray(jax.jit(jax.vmap(
        lambda p: kernel(p, geometric=False)))(x)))
    wrong = np.abs(np.asarray(jax.jit(jax.vmap(lambda p: kernel(p, geometric=True)))(x)) - v_def)
    per_draw = wrong.max(axis=1)
    _measured("claim13 viscous_log10 geometric=True on a linear grid",
              max_ddex=f"{wrong.max():.3g}", median_ddex=f"{np.median(wrong):.3g}",
              smallest_per_draw_max_ddex=f"{per_draw.min():.3g}")
    assert per_draw.min() > MIN_LINEAR_GRID_ERROR_DEX, per_draw


@pytest.mark.parametrize("kind", KINDS)
def test_per_observation_spelling_is_the_code_before_the_options(kind):
    """``unique_epochs=False, geometric=False``: no attribute, no ``geometric`` key and a plain
    ``_PredictJax``, so the state, and the ``model_record`` identity, of an object from before the
    options (the default model with both removed); bit for bit on the CPU that object's host
    ``predict`` and ``jit(vmap(predict_jax))``, and the physics function called with
    ``geometric=False`` and no map; the same jaxpr as that object's, without ``floor``."""
    import dataclasses
    import pickle

    from whisper_cbpf.results import model_record

    fac = {"tde_fallback": wp.tde_fallback_model, "tde_mosfit": wp.tde_mosfit_model}[kind]
    name = f"{kind}_before_the_options"
    m0 = fac(BANDS, Z, DL, name=name, t_exp_days=T0, unique_epochs=False, geometric=False)
    ctx0 = m0.predict.__self__
    assert "unique_epochs" not in vars(ctx0) and "geometric" not in ctx0.common
    assert type(m0.predict_jax) is _PredictJax
    # the default model with its two options removed: the state before they existed
    old = pickle.loads(pickle.dumps(fac(BANDS, Z, DL, name=name, t_exp_days=T0)))
    octx = old.predict.__self__
    assert octx.unique_epochs is True and octx.common["geometric"] is True
    del octx.unique_epochs
    del octx.common["geometric"]
    old = dataclasses.replace(old, predict_jax=_PredictJax(octx))
    assert model_record(old)["identity_hash"] == model_record(m0)["identity_hash"]
    t, b = _grid("LC35")
    th = _draws(_ld(kind, "fixed", False, "LC35"), 8, SEED + 75)
    host = [np.stack([np.asarray(m.predict(dict(zip(m.parameters, x)), t, b), dtype=float)
                      for x in th]) for m in (m0, old)]
    assert np.array_equal(host[0], host[1])
    bidx = m0.predict_jax.band_index(b)
    jv = [np.asarray(jax.jit(jax.vmap(lambda x, m=m: m.predict_jax(x, t, bidx)))(
        jnp.asarray(th))) for m in (m0, old)]
    assert np.array_equal(jv[0], jv[1])
    rel, bj, k = jnp.asarray(t - T0), jnp.asarray(bidx), len(m0.parameters)
    mags = [np.asarray(jax.jit(jax.vmap(lambda x, kw=kw: ctx0._physics_call(
        rel, bj, [x[j] for j in range(k)], **kw)))(jnp.asarray(th))) for kw in (
        {}, {"geometric": False})]
    assert np.array_equal(mags[0], mags[1])
    closed = [jax.make_jaxpr(lambda x, m=m: m.predict_jax(x, t, bidx))(jnp.asarray(th[0]))
              for m in (m0, old)]
    assert "floor" not in _prims(closed[0]) and str(closed[0]) == str(closed[1])
