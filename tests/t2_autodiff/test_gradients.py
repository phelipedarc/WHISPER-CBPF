"""T2 -- AUTODIFF CORRECTNESS for the JAX transient models.

T0 asks "is the forward pass right?". T2 asks the two questions a gradient-based sampler
actually depends on:

    (1) does ``jax.grad`` return a NUMBER, everywhere the model's own prior can go, and
    (2) is that number the derivative?

Both are answered as measured distributions over each model's OWN prior -- read from
``tde.default_prior_gaussianrise()`` / ``kilonova_two.default_prior()`` / the
``kilonova_model`` factory, never invented here -- rather than at pinned parameter sets. That
choice is deliberate: an earlier regression test in this repo pinned a chaotic termination
index and its premise stopped holding on a different backend, so every sweep below is a
property over the prior box and every threshold bounds a DISTRIBUTION.

WHAT EACH TEST WOULD CATCH is stated in its own docstring. In outline:

  * finiteness -- a NaN/inf gradient from a guard that fires inside the light curve, from
    ``0 ** 0.25``, from ``log(0)``, from an overflowing normalisation, or from a ``where``
    whose unselected branch poisons the reverse pass. This is the failure the CHANGE 4 guard
    discipline in ``tde.py`` exists to prevent, and it is silent: a NaN gradient stops a NUTS
    chain without stopping the forward model.
  * AD vs central finite differences -- a derivative that is finite and WRONG: a mis-transposed
    rule, a ``stop_gradient`` that should not be there, an ``optimization_barrier`` placed
    where it changes the JVP, or a ``where`` whose selected branch is not the value returned.
    A wrong-but-finite gradient is worse than a NaN, because the sampler converges on it.
  * guard bit-identity -- ``tde.py`` CHANGE 4's claim that the ``_EE_DEAD``/``_ME_DEAD``
    sentinels are free INSIDE ``valid``. If they were not, every fitted number would depend on
    a numerical guard.
  * ``unroll=16`` -- the module bounds what an XLA scheduling constant costs. Verified here,
    not repeated.
  * ``arnett_prefactor`` -- that the default really is byte-for-byte redback.
  * f32 vs f64 -- which precisions each model may be fitted in.

FINITE DIFFERENCES ARE NOT A GROUND TRUTH, and this file does not treat them as one. Four
regimes make a central difference meaningless while AD is exactly right. Each is DETECTED and
reported separately rather than averaged into a headline number:

  A. A MOVING INTEGER. The TDE's ``constraint`` (the termination index) is an integer function
     of the parameters; so is the Barnes-Kasen table cell in the kilonova (PHYSICS_NOTES K5)
     and the count of epochs sitting on the temperature floor. If one differs between the ends
     of an FD stencil, the two evaluations are of two different curves and their difference is
     not the derivative of anything. Every stencil is fingerprinted with the integers its value
     depends on, and classified before it is used.
  B. FD RESOLUTION. A central difference resolves a slope only down to
     ``~eps*|f| / (h*|theta|)``. Components neither AD nor FD puts above that floor cannot be
     validated by FD at any step size, and are reported as unresolved, not as failures.
  C. A KINK. Where the model is C0 but not C1 in a parameter, the two ONE-SIDED slopes
     converge to different numbers, the central difference converges to their average, and AD
     correctly returns one of the two. Detected by the one-sided gap FAILING TO SHRINK when h
     shrinks by 1000x -- never by the gap merely being large, so ordinary curvature is not
     mistaken for a kink.
  D. NO FD PLATEAU. Where the model is chaotic (the TDE's post-98% tail, D1/D9) the FD
     estimate GROWS as h shrinks, so "best over h" would always find something and it would
     always be noise. FD is used only where two adjacent step sizes agree with each other.

Everything outside A-D is a real disagreement and is asserted on.

Run (CPU; float64 is mandatory -- the TDE engine raises without it) from the repo root, with the
``[gpu,models,dev]`` extras installed::

    export JAX_PLATFORMS=cpu
    source "$(whisper-cbpf-env)"
    python -u -m pytest tests/t2_autodiff -v

``T2_DUMP=/path.json`` writes every measured table (REPORT.md was built from it); by default
the tests write nothing. ``T2_SCALE=0.25`` shrinks every sweep for a quick pass -- the
committed defaults are the sizes REPORT.md quotes.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
# BEFORE any array is created: the cooling-envelope ODE accumulates increments ~1e-6 of the
# state, which float32 cannot resolve, and tde.py raises rather than returning inf later.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from whisper_cbpf.models.jax import kilonova as kn  # noqa: E402
from whisper_cbpf.models.jax import kilonova_two as kn2  # noqa: E402
from whisper_cbpf.models.jax import tde as T  # noqa: E402

DAY = 86400.0
GOLDENS = Path(__file__).resolve().parent.parent / "goldens"
Z = 0.05
DL_CM = 7.0e26                      # ~z = 0.05; the exact cosmology is irrelevant to a gradient
EPS64 = float(np.finfo(np.float64).eps)

#: relative FD steps, best-over-h. eps^(1/3) ~ 6e-6 is the central-difference optimum for a
#: well-scaled f64 function; the sweep brackets it by two decades either way because the
#: optimum moves with the function's own curvature.
H_REL = (1e-3, 1e-4, 1e-5, 1e-6)
#: the pass mark for a gradient component FD can actually see (regimes A and B excluded)
FD_TOL = 1e-3
#: how well two adjacent step sizes must agree before FD is treated as having CONVERGED and
#: therefore as being usable to judge AD at all. Loose on purpose: a smooth f64 function
#: reaches 1e-6 here, so 5% only rejects the genuinely unresolved.
FD_CONV_TOL = 0.05
#: fraction of each draw's OWN curve summed by the "window" TDE observables, frozen at the
#: central parameters and then held fixed like the observation epochs. The scan's trajectory
#: does not depend on `constraint`, so a frozen window is a smooth function of the parameters
#: -- unlike a `meaningful`-masked sum, whose mask is an integer that moves with them. Half
#: the curve leaves 2x margin against the few steps `constraint` can move.
WINDOW_FRAC = 0.5

_SCALE = float(os.environ.get("T2_SCALE", "1.0"))


def _n(x):
    return max(8, int(round(x * _SCALE)))


N_FINITE = {500: _n(512), 5000: _n(192)}    # finiteness sweep, per grid
N_FD = {500: _n(96), 5000: _n(32)}          # AD-vs-FD sweep (x n_param x 4 steps x 2 signs)
N_GUARD = _n(300)                           # the guard claim says 300 draws
N_UNROLL = _n(300)
N_KN = _n(384)
N_EPOCH = 16
#: draws per XLA call. The reverse pass through a 5000-step scan tapes 13 arrays per step, so
#: the whole sweep does not fit in one batch; chunking changes no number, only the peak RSS.
CHUNK = {500: 128, 5000: 32}

_RECORD: dict = {}
_DUMP = os.environ.get("T2_DUMP")


def _record(key, payload):
    _RECORD[key] = payload
    if _DUMP:
        Path(_DUMP).write_text(json.dumps(_RECORD, indent=1, default=float))
    return payload


# =======================================================================================
# helpers
# =======================================================================================

def prior_draws(prior, n, seed):
    """``(names, theta (n, p))`` -- a Latin hypercube over the model's OWN prior.

    ``Prior.rescale`` maps the unit cube through each named distribution, so the box, the
    log/linear spacing and the parameter order are redback's, not this file's.
    """
    from scipy.stats import qmc

    names = list(prior.names)
    u = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    return names, np.array([[prior.rescale(row)[k] for k in names] for row in u])


def stats(x):
    """median / p99 / max / n of a sample, ignoring non-finite entries."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, median=None, p99=None, max=None)
    return dict(n=int(x.size), median=float(np.median(x)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


def batched(fn, size):
    """Run a vmapped ``fn`` over leading-axis chunks and concatenate, as NumPy."""
    def run(*arrays):
        n = int(np.shape(arrays[0])[0])
        outs = [jax.tree_util.tree_map(np.asarray, fn(*[jnp.asarray(np.asarray(a)[i:i + size])
                                                        for a in arrays]))
                for i in range(0, n, size)]
        if len(outs) == 1:
            return outs[0]
        return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *outs)
    return run


def vmap_value(fn, size):
    """Batched ``fn(theta, *data) -> (value, integer signature)``."""
    return batched(jax.jit(jax.vmap(fn)), size)


def vmap_value_and_grad(fn, size):
    """Batched ``value_and_grad`` of the value half of ``fn``, w.r.t. ``theta``."""
    g = jax.jit(jax.vmap(jax.value_and_grad(lambda v, *d: fn(v, *d)[0])))
    return batched(g, size)


@pytest.fixture(scope="session")
def filters():
    """Four real SDSS bandpasses, from the committed T0 golden. No sncosmo, no network."""
    p = GOLDENS / "filterset_t0.npz"
    if not p.exists():
        pytest.skip(f"golden filter set missing: {p}")
    fs = np.load(p, allow_pickle=False)
    W, N = kn.ab_weights(fs["lam"], fs["trans"])
    return dict(lam=jnp.asarray(fs["lam"]), W=W, N=N, n_band=int(fs["trans"].shape[0]),
                lam_np=np.asarray(fs["lam"]), trans_np=np.asarray(fs["trans"]))


def _weights(rng, size):
    """Random O(1) signed weights: a random projection of the Jacobian, so a wrong row cannot
    cancel against another the way a plain sum would let it."""
    return jnp.asarray(rng.uniform(-1.0, 1.0, size))


def _sig(*xs):
    """Pack the integers an FD stencil must be fingerprinted with."""
    return jnp.stack([jnp.asarray(x, dtype=jnp.int32) for x in xs])


# --------------------------------------------------------------------------- FD machinery

def fd_sweep(fun, theta, extra, h_rel=H_REL):
    """Central differences at every ``theta`` row, for every parameter and every step size.

    ``fun(theta_batch, *extra) -> (value (n,), signature (n, k) int)``. Returns
    ``(fd, moved, f0, onesided)``: the central differences, the stencils across which an
    integer moved (regime A), the base values, and the relative disagreement between the two
    ONE-SIDED slopes -- which is what distinguishes a kink from mere truncation error, since
    it shrinks with h for a differentiable function and does not at a kink.
    """
    theta = np.asarray(theta, dtype=float)
    n, p = theta.shape
    f0, s0 = fun(jnp.asarray(theta), *extra)
    f0, s0 = np.asarray(f0), np.asarray(s0)
    fd = np.zeros((n, p, len(h_rel)))
    onesided = np.zeros((n, p, len(h_rel)))
    moved = np.zeros((n, p, len(h_rel)), dtype=bool)
    for j in range(p):
        for k, h in enumerate(h_rel):
            step = h * np.abs(theta[:, j])
            tp, tm = theta.copy(), theta.copy()
            tp[:, j] += step
            tm[:, j] -= step
            fp, sp = fun(jnp.asarray(tp), *extra)
            fm, sm = fun(jnp.asarray(tm), *extra)
            fp, fm = np.asarray(fp), np.asarray(fm)
            fd[:, j, k] = (fp - fm) / (2.0 * step)
            dp, dm = (fp - f0) / step, (f0 - fm) / step
            d = np.maximum(np.abs(dp), np.abs(dm))
            onesided[:, j, k] = np.where(d > 0, np.abs(dp - dm) / np.where(d > 0, d, 1.0), 0.0)
            moved[:, j, k] = (np.any(np.asarray(sp) != s0, axis=1)
                              | np.any(np.asarray(sm) != s0, axis=1))
    return fd, moved, f0, onesided


def fd_compare(ad, fd, moved, f0, theta, onesided=None, h_rel=H_REL, conv_tol=FD_CONV_TOL):
    """AD/FD comparison ON A CONVERGED FD PLATEAU, with the exclusions split out.

    "Best over h" alone is not a criterion: at a discontinuity the FD estimate GROWS as h
    shrinks (the numerator stops shrinking), so some h always looks best and the number it
    produces is meaningless. What separates the two cases is whether FD converges in h at all.
    So the plateau is found first -- the adjacent pair of step sizes whose FD estimates agree
    best -- and AD is compared only where that pair agrees to ``conv_tol``. Where it does not,
    FD has not resolved a derivative and cannot be used to judge one; that fraction is
    reported, together with how much of it coincides with a moved integer (regime A).

    Returns ``(n, p)`` arrays: ``rel`` (NaN where excluded), ``clean``, ``moved``,
    ``unconverged``, ``unresolved``, plus per-h detail for diagnosis.
    """
    ad = np.asarray(ad, dtype=float)
    theta = np.asarray(theta, dtype=float)
    h = np.asarray(h_rel)[None, None, :]
    # FD resolution floor: the difference is formed at ~eps*|f| absolute, so a central
    # difference cannot resolve a slope below eps*|f| / (h*|theta_j|). x100 for margin.
    floor = 100.0 * EPS64 * np.abs(f0)[:, None, None] / np.maximum(
        h * np.abs(theta)[:, :, None], 1e-300)
    den = np.maximum(np.abs(ad)[:, :, None], np.abs(fd))
    rel_h = np.where(den > 0.0, np.abs(ad[:, :, None] - fd) / np.where(den > 0.0, den, 1.0), 0.0)

    a, b = fd[:, :, :-1], fd[:, :, 1:]
    den_adj = np.maximum(np.abs(a), np.abs(b))
    adj = np.where(den_adj > 0.0, np.abs(a - b) / np.where(den_adj > 0.0, den_adj, 1.0), 0.0)
    adj = np.where(moved[:, :, :-1] | moved[:, :, 1:], np.inf, adj)
    best_adj = np.min(np.where(np.isfinite(adj), adj, np.inf), axis=2)
    converged = np.isfinite(best_adj) & (best_adj <= conv_tol)
    # a slope neither AD nor FD puts above the noise floor at ANY step is not something a
    # central difference can adjudicate. `max(|ad|, |fd|)`, not `|ad|`: a wrongly-zero AD
    # against an FD that clears the floor is a real disagreement and must stay in.
    resolved = np.any(np.maximum(np.abs(ad)[:, :, None], np.abs(fd)) >= floor, axis=2)

    # A KINK. The two one-sided slopes of a differentiable function converge to each other
    # linearly in h; at a kink they converge to two different numbers, and the central
    # difference converges to their average -- which is not a derivative of anything, while AD
    # returns one of the two one-sided values (correctly: it takes the branch the `where` /
    # comparison selects). Detected as "the one-sided gap failed to shrink when h shrank by
    # 1000x", never as "the gap is large", so ordinary curvature is not mistaken for a kink.
    if onesided is None:
        kinked = np.zeros(ad.shape, dtype=bool)
    else:
        os_lo, os_hi = onesided[:, :, -1], onesided[:, :, 0]
        # h shrinks by h_rel[-1]/h_rel[0] (1e-3 here); a differentiable function's one-sided
        # gap shrinks with it, so "still >5% of its value at the largest h" is a kink.
        kinked = (os_lo > 1e-7) & (os_lo > 0.05 * os_hi)

    clean = converged & resolved & ~kinked
    # once FD is known to have converged, take its best h: the pair that agrees best with
    # ITSELF is not always the most accurate, since truncation error falls with h while
    # cancellation error rises.
    rel = np.where(clean, np.min(np.where(moved, np.inf, rel_h), axis=2), np.nan)
    all_moved = np.all(moved, axis=2)
    unresolved = ~resolved & ~all_moved
    return dict(rel=rel, clean=np.isfinite(rel), moved=all_moved, kinked=kinked & ~all_moved,
                unconverged=~converged & ~all_moved & resolved & ~kinked, unresolved=unresolved,
                any_moved=np.any(moved, axis=2), fd_plateau=best_adj, per_h=rel_h, fd=fd,
                onesided=onesided)


def table_by_param(rel, clean, names):
    """``{param: {n, median, p99, max, frac_over_tol, frac_excluded}}`` -- the T2 headline."""
    out = {}
    for j, nm in enumerate(names):
        r = rel[:, j][clean[:, j]]
        s = stats(r)
        s["frac_over_tol"] = float(np.mean(r > FD_TOL)) if r.size else None
        s["frac_excluded"] = float(np.mean(~clean[:, j]))
        out[nm] = s
    return out


def summarise_fd(name, grad, fd, moved, f0, theta, names, onesided=None):
    """One observable's AD/FD table, plus its five worst clean disagreements."""
    cmp = fd_compare(np.asarray(grad), fd, moved, f0, theta, onesided=onesided)
    r = cmp["rel"][cmp["clean"]]
    excl = ~cmp["clean"]
    table = dict(by_param=table_by_param(cmp["rel"], cmp["clean"], names),
                 overall=stats(r),
                 frac_excluded=float(np.mean(excl)),
                 frac_stencil_moved=float(np.mean(cmp["moved"])),
                 frac_kinked=float(np.mean(cmp["kinked"])),
                 frac_fd_unconverged=float(np.mean(cmp["unconverged"])),
                 frac_below_fd_resolution=float(np.mean(cmp["unresolved"])),
                 frac_excluded_with_moved_integer=(
                     float(np.mean(cmp["any_moved"][excl])) if excl.any() else None),
                 frac_over_tol=float(np.mean(r > FD_TOL)) if r.size else None)
    order = np.argsort(-np.where(cmp["clean"], np.nan_to_num(cmp["rel"]), -1.0), axis=None)[:5]
    ii, jj = np.unravel_index(order, cmp["rel"].shape)
    table["worst"] = [dict(param=names[j], rel=float(cmp["rel"][i, j]),
                           theta=dict(zip(names, theta[i].tolist())),
                           ad=float(np.asarray(grad)[i, j]),
                           fd_per_h=fd[i, j].tolist(),
                           rel_per_h=[None if not np.isfinite(x) else float(x)
                                      for x in cmp["per_h"][i, j]])
                      for i, j in zip(ii, jj) if cmp["clean"][i, j]]
    return table, r


# --------------------------------------------------------------------------- source variants

def source_variant(name, subs, tmpdir, source=None):
    """Import a copy of a model module with ``subs`` applied to its SOURCE.

    Builds the counterfactuals T2 must compare against: ``tde.py`` with the CHANGE-4 sentinels
    removed, with the ``unroll=16`` token changed, and ``kilonova.py`` with the
    ``arnett_prefactor`` multiply deleted outright. Each substitution asserts it matched
    exactly once, so a refactor that moves the code fails loudly instead of silently comparing
    a module against itself.
    """
    src = Path(source or T.__file__).read_text()
    for old, new in subs:
        assert src.count(old) == 1, f"{name}: anchor matched {src.count(old)}x, not 1: {old!r}"
        src = src.replace(old, new)
    src = src.replace("from .kilonova import", "from whisper_cbpf.models.jax.kilonova import")
    # kilonova.py re-exports the grid rule from whisper_cbpf.synphot since whisper 0.1.1
    src = src.replace("from ...synphot.", "from whisper_cbpf.synphot.")
    assert "\nfrom ." not in src, f"{name}: unhandled relative import"
    path = Path(tmpdir) / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"_t2_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


UNGUARDED_SUBS = [(
    "    Ee_safe = jnp.where(alive, Ee, _EE_DEAD)\n"
    "    Me_safe = jnp.where(alive, Me, _ME_DEAD)\n",
    "    Ee_safe = Ee\n"
    "    Me_safe = Me\n",
)]
UNROLL1_SUBS = [(", xs, unroll=16)", ", xs)")]
NOFLAG_SUBS = [("    if arnett_prefactor != 1.0:\n        lum = arnett_prefactor * lum\n", "")]


@pytest.fixture(scope="session")
def tde_unguarded(tmp_path_factory):
    """``tde.py`` with the ``_EE_DEAD``/``_ME_DEAD`` sentinels removed, nothing else."""
    return source_variant("tde_unguarded", UNGUARDED_SUBS,
                          tmp_path_factory.mktemp("t2_variants"))


@pytest.fixture(scope="session")
def tde_unroll1(tmp_path_factory):
    """``tde.py`` with only the ``unroll=16`` token removed (XLA's default schedule)."""
    return source_variant("tde_unroll1", UNROLL1_SUBS,
                          tmp_path_factory.mktemp("t2_variants"))


# =======================================================================================
# TDE observables
# =======================================================================================

CE_NAMES = list(T.PARAMETERS)                       # mbh_6, stellar_mass, eta, alpha, beta
GR_NAMES = list(T.PARAMETERS_GAUSSIANRISE)          # peak_time, sigma_t, then the five above


def ce_prior_box():
    """redback's ``gaussianrise_cooling_envelope`` box, restricted to the five ENGINE
    parameters.

    redback's own ``cooling_envelope.prior`` collapses four of those five to delta functions
    (see ``tde.redback_prior``), which would leave a ONE-dimensional gradient sweep of a
    five-parameter engine. The seven-parameter file is the same model's prior with the same
    four free, so it is the honest box in which to ask whether the engine's gradients are
    finite. Both are redback's; neither is invented here, and no bound is widened.
    """
    prior, pinned = T.default_prior_gaussianrise()
    assert not pinned, pinned
    assert list(prior.names) == GR_NAMES, list(prior.names)
    return type(prior)({k: prior.distributions[k] for k in CE_NAMES})


@partial(jax.jit, static_argnames="n_time")
def _engine_scalars(v, n_time):
    o = T.cooling_envelope(v[0], v[1], v[2], v[3], v[4], n_time=n_time)
    return o["tfb"], o["termination_time"], o["constraint"]


def envelope_epochs(theta_ce, n_time, kind, n_epoch=N_EPOCH, hi=0.98):
    """Observation epochs INSIDE each draw's own envelope, then FROZEN as data.

    They are computed once from the CENTRAL parameters and held fixed across every FD stencil
    and every gradient, because observation times are data, not parameters. Choosing them per
    draw rather than as one absolute grid is what ``_interp_photosphere``'s docstring asks for
    ("keep observations inside the envelope's own span"): tfb spans 58 d to 411 d over this
    prior, so a single absolute grid would put most draws entirely outside their own model and
    measure the gradient of a constant.

    ``kind='ce'``  observer-frame days SINCE FALLBACK, 2%..``hi`` of the envelope's life.
    ``kind='gr'``  observer-frame days from the light-curve ORIGIN: half on the Gaussian rise
                   (before the stitch at ``tfb*(1+z)``), half on the envelope.

    ``hi`` is 0.98 for the finiteness sweep (which should stress the edge) and 0.90 for the
    FD sweep, where the last few per cent of the curve is chaotic in the parameters -- a
    property of the model, measured by
    ``test_tde_tail_conditioning_is_what_bounds_finite_differences``, not of the gradient.
    """
    f = batched(jax.jit(jax.vmap(partial(_engine_scalars, n_time=n_time))), CHUNK[n_time])
    tfb, tterm, cons = f(np.asarray(theta_ce))
    dead = cons < 2
    if kind == "ce":
        frac = np.linspace(0.02, hi, n_epoch)[None, :]
        t = (tterm[:, None] * frac) * (1.0 + Z) / DAY
    else:
        stitch = tfb * (1.0 + Z) / DAY
        end = (tfb + tterm) * (1.0 + Z) / DAY
        n_rise = n_epoch // 2
        fr = np.linspace(0.30, 0.97, n_rise)[None, :]
        fe = np.linspace(0.01, hi, n_epoch - n_rise)[None, :]
        t = np.concatenate([stitch[:, None] * fr,
                            stitch[:, None] + (end - stitch)[:, None] * fe], axis=1)
    # a draw whose envelope never lived has no span: give it a nominal grid and count it
    t = np.where(dead[:, None], np.linspace(0.1, 10.0, n_epoch)[None, :], t)
    # the frozen grid window: the first WINDOW_FRAC of THIS draw's own curve, as data
    kwin = np.maximum((WINDOW_FRAC * cons).astype(np.int64), 1)
    wmask = (np.arange(n_time)[None, :] < kwin[:, None]).astype(float)
    return t, wmask, dict(dead=dead, constraint=cons, tfb_days=tfb / DAY,
                          term_days=tterm / DAY, window_median=float(np.median(kwin)))


def tde_losses(n_time, filt, rng, n_epoch=N_EPOCH, mode="masked"):
    """The six TDE observables a fit consumes, each reduced by a random projection.

    Every one returns ``(value, signature)``, the signature being the integers that can jump
    between two FD evaluations: the termination index, and the number of epochs the model
    declares outside its own span.

    ``mode`` selects how the raw grid arrays are reduced.
      ``'masked'`` -- sum over ``meaningful``. This is what a consumer of the whole light
        curve sees, and it is what the finiteness sweep uses, because the mask is exactly
        where a guard could leak. Its mask is an integer function of the parameters.
      ``'window'`` -- sum over a FROZEN leading window of each draw's own curve (``wmask``,
        built once at the central parameters, exactly as the epochs are). The scan's
        trajectory does not depend on ``constraint``, so this is a genuinely smooth function
        of the parameters and is therefore the form FD can validate.
    """
    wg = _weights(rng, n_time)
    we = _weights(rng, n_epoch)
    bidx = jnp.asarray(np.arange(n_epoch) % filt["n_band"])
    lam, W, N = filt["lam"], filt["W"], filt["N"]

    def _out(v):
        return T.cooling_envelope(v[0], v[1], v[2], v[3], v[4], n_time=n_time)

    def _grid(v, tobs, wmask, key=None, scale=1.0):
        o = _out(v)
        if mode == "window":
            return jnp.sum(wg * wmask * o[key]) * scale, _sig(0)
        val = jnp.sum(wg * jnp.where(o["meaningful"], o[key], 0.0)) * scale
        return val, _sig(o["constraint"], jnp.sum(~o["meaningful"]))

    def _interp(v, tobs, wmask, which=None):
        o = _out(v)
        temp, rad = T._interp_photosphere(o, tobs * DAY / (1.0 + Z))
        val = jnp.sum(we * (temp if which == "T" else rad * 1e-14))
        return val, _sig(o["constraint"], jnp.sum(rad == 0.0))

    def _mag_ce(v, tobs, wmask):
        o = _out(v)
        _, rad = T._interp_photosphere(o, tobs * DAY / (1.0 + Z))
        m = T.cooling_envelope_ab_magnitude(tobs, bidx, W, N, lam, Z, DL_CM,
                                            v[0], v[1], v[2], v[3], v[4], n_time=n_time)
        return jnp.sum(we * m), _sig(o["constraint"], jnp.sum(rad == 0.0))

    def _mag_gr(v, tobs, wmask):
        # v is the 7-vector: peak_time, sigma_t, then the five engine parameters
        o = T.cooling_envelope(v[2], v[3], v[4], v[5], v[6], n_time=n_time)
        m = T.gaussianrise_cooling_envelope_ab_magnitude(
            tobs, bidx, W, N, lam, Z, DL_CM, v[0], v[1], v[2], v[3], v[4], v[5], v[6],
            n_time=n_time)
        return (jnp.sum(we * m),
                _sig(o["constraint"], jnp.sum(tobs * DAY < o["tfb"] * (1.0 + Z))))

    tag = f"first {WINDOW_FRAC:.0%} of curve" if mode == "window" else "grid"
    ce = {
        f"T_phot ({tag})": partial(_grid, key="photosphere_temperature"),
        f"R_phot ({tag})": partial(_grid, key="photosphere_radius", scale=1e-14),
        f"L_bol ({tag})": partial(_grid, key="bolometric_luminosity", scale=1e-44),
        "T_phot (interpolated)": partial(_interp, which="T"),
        "R_phot (interpolated)": partial(_interp, which="R"),
        "mag_AB (cooling_envelope)": _mag_ce,
    }
    return ce, {"mag_AB (gaussianrise)": _mag_gr}


def tde_cases(n_time, n_draw, filt, rng, seed, mode="masked", hi=0.98):
    """``[(name, fn, theta, data, names, meta)]`` for both TDE entry points, where ``data`` is
    the frozen ``(epochs, window mask)`` pair each loss is evaluated against. No draw is
    filtered out anywhere: the whole prior box is swept."""
    _, theta_ce = prior_draws(ce_prior_box(), n_draw, seed=seed)
    gr_names, theta_gr = prior_draws(T.default_prior_gaussianrise()[0], n_draw, seed=seed + 1)
    assert gr_names == GR_NAMES, gr_names
    t_ce, w_ce, meta_ce = envelope_epochs(theta_ce, n_time, "ce", hi=hi)
    t_gr, w_gr, meta_gr = envelope_epochs(theta_gr[:, 2:], n_time, "gr", hi=hi)
    ce, gr = tde_losses(n_time, filt, rng, mode=mode)
    cases = ([(k, f, theta_ce, (t_ce, w_ce), CE_NAMES, meta_ce) for k, f in ce.items()]
             + [(k, f, theta_gr, (t_gr, w_gr), GR_NAMES, meta_gr) for k, f in gr.items()])
    return cases, meta_ce, meta_gr


# =======================================================================================
# kilonova observables
# =======================================================================================

def _kilonova_prior(filt):
    """The prior the ``kilonova_model`` factory publishes -- redback's own file, verbatim."""
    import whisper_cbpf.models.jax as M

    m = M.kilonova_model(["sdssu", "sdssg", "sdssr", "sdssi"], Z, DL_CM,
                         filter_set=dict(lam=filt["lam_np"], trans=filt["trans_np"]))
    return m.default_prior


def kn_epochs(n_draw, n=N_EPOCH):
    """Observer-frame days after explosion: a real kilonova cadence, 0.5 -> 15 d."""
    return np.tile(np.geomspace(0.5, 15.0, n)[None, :], (n_draw, 1))


def kn_losses(filt, rng):
    """L_bol, T_phot, R_phot, band AB magnitude and flux density, 1-component kilonova."""
    we = _weights(rng, N_EPOCH)
    bidx = jnp.asarray(np.arange(N_EPOCH) % filt["n_band"])
    lam, W, N = filt["lam"], filt["W"], filt["N"]
    nu = jnp.asarray(kn.SPEED_OF_LIGHT / (float(np.mean(filt["lam_np"])) * 1e-8))

    def _src(tobs):
        return jnp.asarray(tobs) * DAY / (1.0 + Z)

    def _floored(v, tobs):
        """How many epochs sit on the temperature floor. The free-expansion / floor switch is
        C0 but not C1 in ``temperature_floor`` (kilonova.py says so: "only the DERIVATIVE
        w.r.t. temperature_floor jumps"), so a stencil that moves an epoch across it compares
        two sides of a kink -- the same regime as the Barnes-Kasen nodes."""
        _, Tp, _ = kn.bolometric(_src(tobs), v[0], v[1], v[2], v[3])
        return jnp.sum(Tp <= v[3] * (1.0 + 1e-14))

    def bol(v, tobs, which=None):
        L, Tp, R = kn.bolometric(_src(tobs), v[0], v[1], v[2], v[3])
        x = {"L": L, "T": Tp, "R": R * 1e-15}[which]
        return jnp.sum(we * x), _sig(jnp.sum(Tp <= v[3] * (1.0 + 1e-14)))

    def mag(v, tobs):
        m = kn.ab_magnitude(_src(tobs), bidx, W, N, lam, Z, DL_CM, v[0], v[1], v[2], v[3])
        return jnp.sum(we * m), _sig(jnp.sum(m >= kn.MAG_FLOOR), _floored(v, tobs))

    def flux(v, tobs):
        f = kn.flux_density_mjy(_src(tobs), nu, Z, DL_CM, v[0], v[1], v[2], v[3])
        return jnp.sum(we * f * 1e3), _sig(jnp.sum(f == 0.0), _floored(v, tobs))

    return {"L_bol": partial(bol, which="L"), "T_phot": partial(bol, which="T"),
            "R_phot": partial(bol, which="R"), "mag_AB": mag, "flux_mJy": flux}


def kn2_losses(filt, rng):
    """The two-component stitched magnitude -- what ``kilonova_two`` actually exposes."""
    we = _weights(rng, N_EPOCH)
    bidx = jnp.asarray(np.arange(N_EPOCH) % filt["n_band"])
    lam, W, N = filt["lam"], filt["W"], filt["N"]

    def mag2(v, tobs):
        t = jnp.asarray(tobs) * DAY / (1.0 + Z)
        m = kn2.two_component_magnitude(t, bidx, W, N, lam, Z, DL_CM,
                                        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
        nfl = [jnp.sum(kn.bolometric(t, v[4 * c], v[4 * c + 1], v[4 * c + 3],
                                     v[4 * c + 2])[1] <= v[4 * c + 2] * (1.0 + 1e-14))
               for c in (0, 1)]
        return jnp.sum(we * m), _sig(jnp.sum(m >= kn.MAG_FLOOR), *nfl)

    return {"mag_AB (two_component)": mag2}


_BK_MEJ = np.asarray(kn._BK_MEJ, dtype=float)
_BK_VEJ = np.asarray(kn._BK_VEJ, dtype=float)


def kn_signature(vf, n_comp=1):
    """Wrap a batched kilonova loss so its FD signature also carries the Barnes-Kasen cell.

    PHYSICS_NOTES K5: the (a, b, d) table is bilinear with a one-sided derivative AT its nodes,
    and ``vej = 0.2, 0.3, 0.4`` are interior to the prior ``U(0.1, 0.5)``. A stencil that
    straddles a node compares two sides of a kink, so FD is not the derivative there and AD --
    which takes the ``x > xg`` branch -- is the right one-sided answer.
    """
    def run(th, *extra):
        val, sig = vf(np.asarray(th), *extra)
        thn = np.asarray(th)
        cells = []
        for c in range(n_comp):
            cells += [np.searchsorted(_BK_MEJ, thn[:, 4 * c + 0], side="left"),
                      np.searchsorted(_BK_VEJ, thn[:, 4 * c + 1], side="left")]
        return val, np.concatenate([np.asarray(sig), np.stack(cells, axis=1)], axis=1)
    return run


# =======================================================================================
# 1. GRADIENTS ARE FINITE
# =======================================================================================

@pytest.mark.parametrize("n_time", [500, 5000])
def test_tde_gradients_are_finite_over_its_own_prior(n_time, filters):
    """Every TDE observable a fit consumes has a finite gradient over redback's own box.

    WOULD CATCH: a guard that fires INSIDE ``valid`` (the 6.55% failure the CHANGE 4 block
    records), ``0 ** 0.25`` at a zero-padded luminosity, ``log(0)`` at a vanished photosphere,
    D6's underflowing Gaussian normalisation restored (0.27% NaN gradients in redback's own
    form), or D7's unclamped rise exponent restored (0.31%). Each of those returns NaN or inf
    from ``jax.grad`` while the forward model still looks healthy -- which stops a NUTS chain
    with no other symptom.

    Both grids are swept because ``constraint``, and hence how much of the curve the reverse
    pass differentiates through, is a strong function of ``n_time``.
    """
    n_draw = N_FINITE[n_time]
    rng = np.random.default_rng(20260809 + n_time)
    cases, meta_ce, meta_gr = tde_cases(n_time, n_draw, filters, rng, seed=11 + n_time)

    out, bad = {}, {}
    for name, fn, theta, data, names, meta in cases:
        val, grad = vmap_value_and_grad(fn, CHUNK[n_time])(theta, *data)
        g, v = np.asarray(grad), np.asarray(val)
        finite = np.isfinite(g)
        out[name] = dict(n_draw=int(theta.shape[0]), n_param=len(names),
                         frac_nonfinite=float(np.mean(~finite)),
                         frac_draws_any_nonfinite=float(np.mean(~finite.all(axis=1))),
                         frac_value_nonfinite=float(np.mean(~np.isfinite(v))),
                         frac_grad_exactly_zero=float(np.mean(np.all(g == 0.0, axis=1))))
        if not finite.all():
            rows = np.where(~finite.all(axis=1))[0][:10]
            bad[name] = [dict(theta=dict(zip(names, theta[r].tolist())), grad=g[r].tolist(),
                              constraint=int(meta["constraint"][r]),
                              term_days=float(meta["term_days"][r])) for r in rows]
    out["_prior_context"] = dict(
        n_time=n_time, n_draw=n_draw,
        frac_dead_envelope_ce=float(np.mean(meta_ce["dead"])),
        frac_dead_envelope_gr=float(np.mean(meta_gr["dead"])),
        median_constraint_ce=float(np.median(meta_ce["constraint"])),
        frac_constraint_lt_6_ce=float(np.mean(meta_ce["constraint"] < 6)))
    _record(f"tde_finite_n{n_time}", dict(table=out, offenders=bad))
    assert not bad, ("non-finite TDE gradients inside the model's own prior:\n"
                     + json.dumps(bad, indent=1))


def test_kilonova_gradients_are_finite_over_its_own_prior(filters):
    """The one- and two-component kilonovae, over the priors their own factories publish.

    WOULD CATCH: any of the float32-safety regroupings (identities 1, 4, 5, 7, 8) being undone
    -- they exist because the REVERSE pass overflows where the forward pass does not, so a
    forward-only test cannot see it; the ``lum_safe`` floor removed (``L**0.25`` has an
    infinite derivative at the exact zero every pre-explosion epoch produces); the magnitude
    floor clamped after the log rather than on the ratio (finite value, NaN gradient); and a
    NaN leaking out of the unselected branch of the pre-explosion ``where``.
    """
    rng = np.random.default_rng(4242)
    names, theta = prior_draws(_kilonova_prior(filters), N_KN, seed=21)
    names2, theta2 = prior_draws(kn2.default_prior(), N_KN, seed=22)
    todo = [("kilonova", kn_losses(filters, rng), theta, names),
            ("kilonova_two", kn2_losses(filters, rng), theta2, names2)]

    out, bad = {}, {}
    for tag, losses, th, nms in todo:
        tobs = kn_epochs(th.shape[0])
        for name, fn in losses.items():
            val, grad = vmap_value_and_grad(fn, 128)(th, tobs)
            g = np.asarray(grad)
            finite = np.isfinite(g)
            key = f"{tag}: {name}"
            out[key] = dict(n_draw=int(th.shape[0]), n_param=len(nms),
                            frac_nonfinite=float(np.mean(~finite)),
                            frac_value_nonfinite=float(np.mean(~np.isfinite(np.asarray(val)))))
            if not finite.all():
                rows = np.where(~finite.all(axis=1))[0][:10]
                bad[key] = [dict(theta=dict(zip(nms, th[r].tolist())), grad=g[r].tolist())
                            for r in rows]
    _record("kilonova_finite_f64", dict(table=out, offenders=bad))
    assert not bad, json.dumps(bad, indent=1)


# =======================================================================================
# 2. GRADIENTS ARE CORRECT
# =======================================================================================

@pytest.mark.parametrize("n_time", [500, 5000])
def test_tde_ad_matches_finite_differences(n_time, filters):
    """AD vs central FD for every TDE observable x parameter, best over four step sizes.

    WOULD CATCH a derivative that is finite and wrong: a mis-transposed rule, a
    ``stop_gradient`` on something that should carry one, an ``optimization_barrier`` that is
    not JVP-identity (``tde.py`` uses two), or a ``where`` whose SELECTED branch is not the
    value returned. None of those show up in forward parity or in a finiteness sweep; they
    show up as a sampler that converges confidently on the wrong parameters.

    The exclusions of the module docstring are applied BEFORE anything is asserted, and each
    is MEASURED per stencil rather than assumed: a stencil across which an integer moved, one
    on which FD never converges in h, and a component below the FD resolution floor. All three
    fractions are reported, and a large one is itself a finding.

    The observables are posed so that FD CAN see them: the grid sums run over a FROZEN leading
    window of each draw's own curve (the scan's trajectory does not depend on ``constraint``,
    so that sum is smooth, whereas a ``meaningful``-masked sum has a moving integer in it), and
    the epochs stop at 90% of each envelope's life. Both are restrictions on the VALIDATOR, not
    on the model, and no draw is dropped: the finiteness sweep above uses the masked sums and
    runs to 98%, and what happens in the last few per cent is MEASURED by the conditioning test
    below rather than swept under the carpet.
    """
    rng = np.random.default_rng(777 + n_time)
    cases, meta_ce, meta_gr = tde_cases(n_time, N_FD[n_time], filters, rng, seed=31 + n_time,
                                        mode="window", hi=0.90)
    tables, failures = {}, {}
    tables["_validator_setup"] = dict(
        n_time=n_time, window_frac=WINDOW_FRAC, n_draw=N_FD[n_time],
        median_window_points_ce=meta_ce["window_median"],
        median_constraint_ce=float(np.median(meta_ce["constraint"])),
        frac_dead_envelope_ce=float(np.mean(meta_ce["dead"])))
    for name, fn, theta, data, names, _meta in cases:
        _, grad = vmap_value_and_grad(fn, CHUNK[n_time])(theta, *data)
        vf = vmap_value(fn, CHUNK[n_time])
        fd, moved, f0, os1 = fd_sweep(vf, theta, data)
        table, r = summarise_fd(name, grad, fd, moved, f0, theta, names, onesided=os1)
        tables[name] = table
        if r.size and np.percentile(r, 99) > FD_TOL:
            failures[name] = table
    _record(f"tde_fd_n{n_time}", tables)
    assert not failures, (f"AD and FD disagree by more than {FD_TOL} at p99 on a converged "
                          f"stencil:\n" + json.dumps(failures, indent=1))


@pytest.mark.parametrize("n_time", [500, 5000])
def test_tde_tail_conditioning_is_what_bounds_finite_differences(n_time):
    """MEASURES the model's own conditioning along the curve -- the reason the FD validator
    stops at 90% and uses a leading window, stated as a number instead of as a caveat.

    D1/D9 in PHYSICS_NOTES claim the trajectory is well conditioned early and chaotic late
    ("<= 4e-14 relative over the first half ... O(1) past 98%", "redback disagrees with itself
    by up to 12-14 steps under a one-ulp parameter nudge"). This nudges alpha and beta by 1e-12
    RELATIVE -- far below any sampler step -- and reports how far that propagates at each
    fraction of the curve, plus how often the termination index moves at all.

    WOULD CATCH the conditioning getting worse: if the amplification at 50% of the curve ever
    rose to where a sampler's own step sizes live, then AD would be reporting the derivative
    of noise over the FITTED part of the light curve and not merely over its last few points.
    It also documents WHY an AD-vs-FD number quoted over the whole curve is meaningless,
    without which the exclusions above would look like an excuse.
    """
    _, theta = prior_draws(ce_prior_box(), _n(128), seed=101 + n_time)
    eng = batched(jax.jit(jax.vmap(partial(T.cooling_envelope, n_time=n_time))), CHUNK[n_time])
    base = eng(*[theta[:, i] for i in range(5)])
    T0 = np.asarray(base["photosphere_temperature"])
    c0 = np.asarray(base["constraint"]).astype(np.int64)
    h = 1e-12
    fracs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    out = {}
    for j, nm in ((3, "alpha"), (4, "beta")):
        th = theta.copy()
        th[:, j] *= (1.0 + h)
        pert = eng(*[th[:, i] for i in range(5)])
        T1 = np.asarray(pert["photosphere_temperature"])
        c1 = np.asarray(pert["constraint"]).astype(np.int64)
        alive = c0 >= 10
        rel = np.abs(T1 - T0) / np.maximum(np.abs(T0), 1e-300)
        prof = {}
        for f in fracs:
            idx = np.clip((f * c0).astype(int) - 1, 0, T0.shape[1] - 1)
            r = rel[np.arange(T0.shape[0]), idx][alive]
            prof[f"{f:.2f}"] = dict(median=float(np.median(r)), p99=float(np.percentile(r, 99)),
                                    max=float(r.max()),
                                    amplification_median=float(np.median(r) / h))
        out[nm] = dict(profile=prof,
                       frac_constraint_moved=float(np.mean((c0 != c1)[alive])),
                       max_steps_moved=int(np.abs(c0 - c1)[alive].max()))
    out["_setup"] = dict(n_time=n_time, n_draw=int(theta.shape[0]), relative_nudge=h)
    _record(f"tde_conditioning_n{n_time}", out)
    # the first half must stay well conditioned: this is what makes the FITTED part of the
    # light curve -- and therefore its gradient -- reproducible at all
    for nm, rec in out.items():
        if nm == "_setup":
            continue
        assert rec["profile"]["0.50"]["p99"] < 1e-6, (nm, rec["profile"]["0.50"])


def test_kilonova_ad_matches_finite_differences(filters):
    """AD vs central FD for the one- and two-component kilonovae, over their own priors.

    WOULD CATCH the same class of finite-but-wrong derivative as the TDE case -- in particular
    identity 7 (``log1p(x)/x`` evaluated by its series so the VJP survives small x) and
    identity 5 (the regrouped ``T``/``R`` divisors), both of which are changes made FOR the
    reverse pass and so cannot be validated by any forward test.

    The Barnes-Kasen node crossings (K5) are excluded by the signature mechanism, and the
    excluded fraction is reported: if it ever grew far past the few per cent a 3-node grid
    implies, that would be the finding.
    """
    rng = np.random.default_rng(9090)
    names, theta = prior_draws(_kilonova_prior(filters), N_FD[500], seed=41)
    names2, theta2 = prior_draws(kn2.default_prior(), N_FD[500], seed=42)
    todo = [(kn_losses(filters, rng), theta, names, 1),
            (kn2_losses(filters, rng), theta2, names2, 2)]
    tables, failures = {}, {}
    for losses, th, nms, ncomp in todo:
        tobs = kn_epochs(th.shape[0])
        for name, fn in losses.items():
            _, grad = vmap_value_and_grad(fn, 128)(th, tobs)
            vf = kn_signature(vmap_value(fn, 128), n_comp=ncomp)
            fd, moved, f0, os1 = fd_sweep(vf, th, (tobs,))
            table, r = summarise_fd(name, grad, fd, moved, f0, th, nms, onesided=os1)
            tables[name] = table
            if r.size and np.percentile(r, 99) > FD_TOL:
                failures[name] = table
    _record("kilonova_fd_f64", tables)
    assert not failures, json.dumps(failures, indent=1)


# =======================================================================================
# 3. THE GUARD BIT-IDENTITY CLAIM (tde.py CHANGE 4)
# =======================================================================================

@pytest.mark.parametrize("n_time", [500, 5000])
def test_guard_sentinels_are_bit_identical_inside_valid(n_time, tde_unguarded):
    """CHANGE 4 claims the ``_EE_DEAD``/``_ME_DEAD`` sentinels cost nothing INSIDE ``valid``.
    Checked against a copy of the module with exactly those two lines removed.

    WOULD CATCH the failure the block describes in the other direction: a sentinel leaking into
    the live trajectory. Every fitted photosphere radius and temperature would then depend on a
    numerical guard rather than on the physics -- and no forward parity test would necessarily
    notice, because both sentinels are O(1) in their own units and the result stays plausible.

    Bit-identity, not a tolerance: the guards sit inside a ``where`` whose predicate is exactly
    ``(Ee > 0) & (Me > 0)``, so if the claim holds there is no rounding to allow for.
    """
    _, theta = prior_draws(ce_prior_box(), N_GUARD, seed=51 + n_time)

    def run(mod):
        f = jax.jit(jax.vmap(partial(mod.cooling_envelope, n_time=n_time)))
        return batched(f, CHUNK[n_time])(*[theta[:, i] for i in range(5)])

    a, b = run(T), run(tde_unguarded)
    valid = np.asarray(a["valid"])
    report = dict(n_draw=int(theta.shape[0]), n_time=n_time,
                  constraint_identical=bool(np.array_equal(a["constraint"], b["constraint"])),
                  mean_valid_fraction=float(valid.mean()),
                  n_arrays_checked=0, arrays={})
    diffs = {}
    for k, va in a.items():
        va, vb = np.asarray(va), np.asarray(b[k])
        if va.ndim == 2 and va.shape[1] == n_time:
            eq = (va == vb) | (~valid)
            report["arrays"][k] = dict(bit_identical_inside_valid=bool(eq.all()),
                                       n_diff_inside_valid=int((~eq).sum()),
                                       n_diff_anywhere=int((va != vb).sum()))
            report["n_arrays_checked"] += 1
            if not eq.all():
                diffs[k] = int((~eq).sum())
        elif va.ndim == 1:
            same = bool(np.array_equal(va, vb))
            report["arrays"][k] = dict(bit_identical=same)
            if not same:
                diffs[k] = "scalar"
    _record(f"guard_identity_n{n_time}", report)
    assert report["constraint_identical"], "removing the sentinels moved `constraint`"
    assert not diffs, f"guarded and unguarded differ inside `valid`: {diffs}"


# =======================================================================================
# 4. lax.scan(..., unroll=16)
# =======================================================================================

@pytest.mark.parametrize("n_time,claim_frac,claim_steps", [(500, 0.087, 4), (5000, 0.187, 18)])
def test_unroll16_effect_stays_inside_the_documented_bound(n_time, claim_frac, claim_steps,
                                                           tde_unroll1, filters):
    """``unroll=16`` is NOT bit-identical, and ``tde.py`` bounds what it costs. Checked here,
    not repeated: the claim is that ``constraint`` moves on 8.7% / 18.7% of draws (max 4 / 18
    steps) at n_time 500 / 5000, with a photometric p99 of 5e-3 mag over 5-95% of the curve.

    WOULD CATCH a compiler or JAX upgrade that turned a documented sub-noise-floor scheduling
    effect into a real one -- i.e. the moment an XLA constant starts moving fitted parameters.
    Asserted with margin, because the quoted numbers are a measurement on one draw set and not
    a supremum; the measured values are recorded so drift is visible rather than hidden by the
    margin.
    """
    n_epoch = 32
    _, theta = prior_draws(ce_prior_box(), N_UNROLL, seed=61 + n_time)
    t_obs, _wmask, meta = envelope_epochs(theta, n_time, "ce", n_epoch=n_epoch)
    bidx = jnp.asarray(np.arange(n_epoch) % filters["n_band"])
    chunk = max(8, CHUNK[n_time] // 4)          # the (n, n_epoch, n_wave) spectrum dominates

    def run(mod):
        eng = batched(jax.jit(jax.vmap(partial(mod.cooling_envelope, n_time=n_time))), chunk)
        cons = eng(*[theta[:, i] for i in range(5)])["constraint"]

        def one(tt, *p):
            return mod.cooling_envelope_ab_magnitude(
                tt, bidx, filters["W"], filters["N"], filters["lam"], Z, DL_CM,
                *p, n_time=n_time)

        m = batched(jax.jit(jax.vmap(one)), chunk)(t_obs, *[theta[:, i] for i in range(5)])
        return np.asarray(cons), np.asarray(m)

    c16, m16 = run(T)
    c1, m1 = run(tde_unroll1)
    d = np.abs(c16.astype(np.int64) - c1.astype(np.int64))
    lo, hi = int(round(0.05 * n_epoch)), int(round(0.95 * n_epoch))
    dm = np.abs(m16[:, lo:hi] - m1[:, lo:hi])
    dm = dm[np.isfinite(dm)]
    got = dict(n_draw=int(theta.shape[0]), n_time=n_time,
               frac_constraint_moved=float(np.mean(d > 0)), max_steps_moved=int(d.max()),
               mag_median=float(np.median(dm)), mag_p99=float(np.percentile(dm, 99)),
               mag_max=float(dm.max()),
               claim=dict(frac=claim_frac, max_steps=claim_steps, mag_p99=5.0e-3))
    _record(f"unroll_n{n_time}", got)
    assert got["frac_constraint_moved"] <= 2.0 * claim_frac + 0.02, got
    assert got["max_steps_moved"] <= 4 * claim_steps, got
    assert got["mag_p99"] <= 3 * 5.0e-3, got


# =======================================================================================
# 5. arnett_prefactor
# =======================================================================================

def test_arnett_default_path_is_elided_at_trace_time(tmp_path):
    """K4 claims the default ``arnett_prefactor=1.0`` is byte-identical to code that never had
    the flag -- "compared at trace time", so the multiply is never emitted.

    WOULD CATCH the flag becoming a traced value (or being compared with ``jnp`` rather than
    Python ``!=``), which would insert a multiply into every kilonova jaxpr and quietly break
    the project rule that the default path IS redback.
    """
    noflag = source_variant("kn_noflag", NOFLAG_SUBS, tmp_path, source=kn.__file__)
    args = (jnp.asarray(kn_epochs(1)[0] * DAY), 0.02, 0.2, 3.0, 4000.0)
    ref = str(jax.make_jaxpr(noflag.bolometric)(*args))
    got = str(jax.make_jaxpr(kn.bolometric)(*args))
    two = str(jax.make_jaxpr(partial(kn.bolometric, arnett_prefactor=2.0))(*args))
    _record("arnett_trace", dict(default_identical_to_flagless_module=bool(got == ref),
                                 two_differs_from_default=bool(two != got),
                                 extra_mul_ops_at_2=int(two.count("mul") - got.count("mul")),
                                 jaxpr_eqns_default=got.count("\n")))
    assert got == ref, "arnett_prefactor=1.0 emitted ops the flagless module does not"
    assert two != got, "arnett_prefactor=2.0 emitted the same graph as 1.0"


def test_arnett_two_is_exactly_double_and_both_paths_differentiate(filters):
    """``arnett_prefactor=2.0`` must be exactly 2x in L -- a power-of-two multiply is exact in
    binary floating point, so this is a BITWISE claim, not a tolerance -- and both settings
    must give finite, FD-correct gradients.

    WOULD CATCH the factor being applied in the wrong place (after the temperature/radius
    split, say, where it would stop being a pure luminosity rescaling), or a flag that changes
    the conditioning of the reverse pass.
    """
    _, theta = prior_draws(_kilonova_prior(filters), _n(128), seed=71)
    tobs = kn_epochs(theta.shape[0])
    rng = np.random.default_rng(3)
    we = _weights(rng, N_EPOCH)

    def L(v, tt, pref):
        return kn.bolometric(jnp.asarray(tt) * DAY / (1.0 + Z), v[0], v[1], v[2], v[3],
                             arnett_prefactor=pref)[0]

    l1 = batched(jax.jit(jax.vmap(partial(L, pref=1.0))), 128)(theta, tobs)
    l2 = batched(jax.jit(jax.vmap(partial(L, pref=2.0))), 128)(theta, tobs)
    exact = bool(np.array_equal(l2, 2.0 * l1))

    nonfinite, fds = {}, {}
    for pref in (1.0, 2.0):
        def fn(v, tt, p=pref):
            return jnp.sum(we * L(v, tt, p)), _sig(jnp.sum(jnp.zeros(1)))
        _, g = vmap_value_and_grad(fn, 128)(theta, tobs)
        vf = kn_signature(vmap_value(fn, 128))
        fd, moved, f0, os1 = fd_sweep(vf, theta, (tobs,))
        cmp = fd_compare(np.asarray(g), fd, moved, f0, theta, onesided=os1)
        r = cmp["rel"][cmp["clean"]]
        nonfinite[pref] = float(np.mean(~np.isfinite(np.asarray(g))))
        fds[pref] = dict(stats(r), frac_over_tol=float(np.mean(r > FD_TOL)) if r.size else None,
                         frac_excluded=float(np.mean(~cmp["clean"])))
    _record("arnett_values", dict(exactly_double_bitwise=exact,
                                  frac_nonfinite_grad=nonfinite, fd=fds))
    assert exact, "arnett_prefactor=2.0 is not bitwise 2x in L"
    assert all(v == 0.0 for v in nonfinite.values()), nonfinite
    for pref, s in fds.items():
        assert s["p99"] is None or s["p99"] <= FD_TOL, (pref, s)


# =======================================================================================
# 6. PRECISION
# =======================================================================================

_F32_PROBE = r'''
"""Written by test_gradients.py. Runs with JAX_ENABLE_X64=0.

x64 is process-wide and the rest of T2 needs it ON, so float32 can only be measured honestly
in a child process: an in-process "float32" run would silently promote against the modules'
float64 table constants (_BK_A, GL nodes) and measure nothing.
"""
import json, sys
import numpy as np
import jax
assert not jax.config.jax_enable_x64
import jax.numpy as jnp
from whisper_cbpf.models.jax import kilonova as kn
from whisper_cbpf.models.jax import tde as T

theta = np.load(sys.argv[1])["theta"].astype(np.float32)
fs = np.load(sys.argv[2], allow_pickle=False)
W, N = kn.ab_weights(fs["lam"].astype(np.float32), fs["trans"].astype(np.float32))
lam = jnp.asarray(fs["lam"], dtype=jnp.float32)
DAY, Z, DL = 86400.0, 0.05, 7.0e26
tobs = np.geomspace(0.5, 15.0, 16).astype(np.float32)
t = jnp.asarray(tobs * DAY / (1.0 + Z), dtype=jnp.float32)
bidx = jnp.asarray(np.arange(16) % int(fs["trans"].shape[0]))
we = jnp.asarray(np.random.default_rng(5).uniform(-1, 1, 16).astype(np.float32))

def mk(which):
    def f(v):
        if which == "mag_AB":
            y = kn.ab_magnitude(t, bidx, W, N, lam, Z, DL, v[0], v[1], v[2], v[3])
        else:
            L, Tp, R = kn.bolometric(t, v[0], v[1], v[2], v[3])
            y = {"L_bol": L, "T_phot": Tp, "R_phot": R * jnp.float32(1e-15)}[which]
        return jnp.sum(we * y)
    return f

out = {}
for which in ("L_bol", "T_phot", "R_phot", "mag_AB"):
    f = mk(which)
    val, g = jax.jit(jax.vmap(jax.value_and_grad(f)))(jnp.asarray(theta))
    g, val = np.asarray(g), np.asarray(val)
    vf = jax.jit(jax.vmap(f))
    best = None
    for h in (1e-2, 3e-3, 1e-3, 3e-4):
        rel = np.zeros_like(g)
        for j in range(theta.shape[1]):
            step = (h * np.abs(theta[:, j])).astype(np.float32)
            tp, tm = theta.copy(), theta.copy()
            tp[:, j] += step; tm[:, j] -= step
            fd = (np.asarray(vf(jnp.asarray(tp))).astype(np.float64)
                  - np.asarray(vf(jnp.asarray(tm))).astype(np.float64)) / (2.0 * step)
            den = np.maximum(np.abs(g[:, j]).astype(np.float64), np.abs(fd))
            rel[:, j] = np.where(den > 0, np.abs(g[:, j] - fd) / np.where(den > 0, den, 1), 0.0)
        best = rel if best is None else np.minimum(best, rel)
    out[which] = dict(dtype=str(g.dtype), n_draw=int(theta.shape[0]),
                      frac_nonfinite=float(np.mean(~np.isfinite(g))),
                      frac_value_nonfinite=float(np.mean(~np.isfinite(val))),
                      fd_median=float(np.nanmedian(best)),
                      fd_p99=float(np.nanpercentile(best, 99)), fd_max=float(np.nanmax(best)))

try:
    T.cooling_envelope(1.0, 1.0, 0.05, 0.1, 1.0, n_time=500)
    out["tde_f32"] = dict(raised=False, message=None)
except RuntimeError as exc:
    out["tde_f32"] = dict(raised=True, message=str(exc)[:300])
print("T2_F32_JSON " + json.dumps(out))
'''


@pytest.fixture(scope="session")
def f32_probe(tmp_path_factory, filters):
    """Run the float32 probe in a child process with x64 OFF; return its JSON."""
    d = tmp_path_factory.mktemp("t2_f32")
    _, theta = prior_draws(_kilonova_prior(filters), _n(192), seed=81)
    np.savez(d / "theta.npz", theta=theta)
    script = d / "probe_f32.py"
    script.write_text(_F32_PROBE)
    env = dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS="cpu",
               PYTHONPATH=os.pathsep.join(sys.path))
    r = subprocess.run([sys.executable, "-u", str(script), str(d / "theta.npz"),
                        str(GOLDENS / "filterset_t0.npz")],
                       capture_output=True, text=True, env=env, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("T2_F32_JSON ")]
    if not line:
        pytest.fail("float32 probe failed:\nSTDOUT\n"
                    f"{r.stdout[-3000:]}\nSTDERR\n{r.stderr[-3000:]}")
    return json.loads(line[0][len("T2_F32_JSON "):])


def test_kilonova_gradients_are_usable_in_float32(f32_probe):
    """The kilonova supports both precisions, so both must be fit-safe.

    WOULD CATCH the loss of any float32 regrouping (identities 1, 4, 5, 7, 8), each of which
    exists because the REVERSE pass overflows float32 where the forward pass does not --
    ``4 pi sigma R^2`` at R ~ 1e15, ``nu^3`` at 1e44, ``L ~ 1e41`` against a 3.4e38 ceiling.
    T0's forward-only f32 probe cannot see any of them.

    The FD bar is looser than f64's by design: float32 carries ~7 decimal digits, so a central
    difference of two f32 evaluations resolves a slope to ~1e-3 at best, and the number being
    bounded here is "AD and FD agree to FD's own precision", not "AD is accurate to 1e-3".
    """
    tab = {k: v for k, v in f32_probe.items() if k != "tde_f32"}
    _record("kilonova_f32", tab)
    for which, s in tab.items():
        assert s["dtype"] == "float32", (which, s)
        assert s["frac_nonfinite"] == 0.0, (which, s)
        assert s["fd_p99"] <= 5e-2, (which, s)


def test_tde_refuses_float32_rather_than_returning_inf(f32_probe):
    """The TDE engine raises in f32 BY DESIGN (CHANGE 8), so that is asserted, not measured:
    the ODE accumulates increments ~1e-6 of the state against a float32 eps of 1.2e-7, and the
    documented failure is ``constraint`` collapsing to 1 with an infinite luminosity.

    WOULD CATCH the guard being removed, or moved to after the first array is created, which
    turns a loud RuntimeError at trace time into a silent, plausible, wrong light curve.
    """
    got = f32_probe["tde_f32"]
    _record("tde_f32", got)
    assert got["raised"], got
    assert "float64" in got["message"], got


# =======================================================================================
# 7. SUPERNOVAE (the third model family; same two questions, one sweep)
# =======================================================================================

def sn_case(model, filt, rng, t_src_days):
    """``(names, prior, loss)`` for one supernova model, over redback's own prior for it."""
    from whisper_cbpf.models.jax import supernova as sn

    prior, pinned, _constraints = sn.default_prior(model)
    names = [k for k in sn.PARAMETERS[model] if k not in pinned]
    dists = type(prior)({k: prior.distributions[k] for k in names})
    grid = sn.build_sn_grid(t_src_days)
    n_obs = int(np.size(t_src_days))
    we = _weights(rng, n_obs)
    bidx = jnp.asarray(np.arange(n_obs) % filt["n_band"])

    vej_name = sn.MODELS[model]["vej_name"]

    def loss(v, _dummy):
        p = dict(zip(names, [v[i] for i in range(len(names))]))
        p.update(pinned)                      # redback's delta functions, as constants
        mag = sn.ab_magnitude_of(model, grid, p, bidx, filt["W"], filt["N"], filt["lam"],
                                 Z, DL_CM)
        # the free-expansion / temperature-floor switch is C0 but not C1, exactly as in the
        # kilonova: count the floored epochs so a stencil that crosses it is detected
        lbol = sn.bolometric(model, grid, p)
        temp, _ = sn.photosphere(grid, lbol, p[vej_name], p["temperature_floor"])
        return (jnp.sum(we * mag),
                _sig(jnp.sum(mag >= 40.0),
                     jnp.sum(temp <= p["temperature_floor"] * (1.0 + 1e-14))))

    return names, dists, loss


def test_supernova_gradients_are_finite_and_fd_correct(filters):
    """All twelve supernova models, each over redback's own prior for it.

    WOULD CATCH the same two failures as the other families, in the family with the most
    parameters (up to 11) and the widest priors -- redback's ``vej`` runs to 1e5 km/s, i.e.
    0.33c, and ``general_magnetar_slsn``'s ``tsd`` is labelled seconds and consumed as days.
    Wide priors are where the reverse pass overflows, and where an interpolation or a
    fractional power is asked for a value the forward pass merely saturates on.

    This family REQUIRES float64 (CHANGE 8) -- not for the TDE's reason, it has no accumulator,
    but because its cgs luminosities of 1e43-1e46 erg/s overflow float32's 3.4e38 outright, and
    ``build_sn_grid`` raises rather than let every band return ``mag_floor``. So it is swept in the
    ambient x64 session like the rest; the per-model table is recorded either way.
    """
    from whisper_cbpf.models.jax import supernova as sn

    rng = np.random.default_rng(5150)
    t_src = np.geomspace(0.5, 120.0, 24)
    table, bad, failures = {}, {}, {}
    for model in sn.model_names():
        names, prior, loss = sn_case(model, filters, rng, t_src)
        _, theta = prior_draws(prior, _n(96), seed=hash(model) % 10000)
        dummy = np.zeros((theta.shape[0], 1))
        _, grad = vmap_value_and_grad(loss, 96)(theta, dummy)
        g = np.asarray(grad)
        fd, moved, f0, os1 = fd_sweep(vmap_value(loss, 96), theta, (dummy,))
        tab, r = summarise_fd(model, g, fd, moved, f0, theta, names, onesided=os1)
        tab["n_param"] = len(names)
        tab["frac_nonfinite_grad"] = float(np.mean(~np.isfinite(g)))
        table[model] = tab
        if not np.isfinite(g).all():
            rows = np.where(~np.isfinite(g).all(axis=1))[0][:5]
            bad[model] = [dict(theta=dict(zip(names, theta[i].tolist())), grad=g[i].tolist())
                          for i in rows]
        if r.size and np.percentile(r, 99) > FD_TOL:
            failures[model] = tab
    _record("supernova", dict(table=table, offenders=bad))
    assert not bad, json.dumps(bad, indent=1)
    assert not failures, json.dumps(failures, indent=1)


# =======================================================================================
# 8. THE FACTORIES
# =======================================================================================

def test_factory_predict_is_a_concrete_numpy_boundary(filters):
    """``models/__init__.py``'s ``predict`` casts every parameter with ``float(...)`` and
    returns NumPy, so it is NOT differentiable -- by construction, not by accident.

    Asserted rather than lamented, because it is a real constraint on T3: a gradient-based
    sampler cannot be pointed at ``Model.predict``; it must call the module functions
    (``tde.gaussianrise_cooling_envelope_ab_magnitude``, ``kilonova.ab_magnitude``) that the
    rest of this file proves differentiable. The test also pins that the two paths agree, so
    the differentiable path is the same model the factory publishes.

    WOULD CATCH a refactor that made ``predict`` traceable-looking but subtly different from
    the module function -- a silent divergence between what is fitted and what is
    differentiated -- and it documents the boundary so nobody rediscovers it inside a sampler.
    """
    import whisper_cbpf.models.jax as M

    fs = dict(lam=filters["lam_np"], trans=filters["trans_np"])
    bands = ["sdssu", "sdssg", "sdssr", "sdssi"]
    m_kn = M.kilonova_model(bands, Z, DL_CM, filter_set=fs)
    m_tde = M.tde_model(bands, Z, DL_CM, filter_set=fs, rise="gaussian", n_time=500)
    m_ce = M.tde_model(bands, Z, DL_CM, filter_set=fs, rise="none", n_time=500)

    _, pinned = T.default_prior()          # redback's own file pins four of the five
    assert m_ce.parameters == [k for k in T.PARAMETERS if k not in pinned]
    assert m_tde.parameters == T.PARAMETERS_GAUSSIANRISE
    assert m_kn.parameters == ["mej", "vej", "kappa", "temperature_floor"]

    t = np.geomspace(1.0, 60.0, 8)
    bb = np.array(bands * 2)
    p_kn = dict(mej=0.02, vej=0.2, kappa=3.0, temperature_floor=4000.0)
    t_src = kn.source_time_s(t, Z)
    direct = kn.ab_magnitude(jnp.asarray(t_src),        # the factory's default: redback's grid
                             jnp.asarray(np.arange(8) % 4), filters["W"], filters["N"],
                             filters["lam"], Z, DL_CM, 0.02, 0.2, 3.0, 4000.0,
                             time_grid=kn.redback_time_grid(t_src))
    assert np.allclose(m_kn.predict(p_kn, t, bb),
                       3631.0 * 10 ** (-0.4 * np.asarray(direct)), rtol=1e-12)

    p_tde = dict(peak_time=20.0, sigma_t=20.0, mbh_6=1.0, stellar_mass=1.0,
                 eta=0.05, alpha=0.1, beta=1.5)
    direct_tde = T.gaussianrise_cooling_envelope_ab_magnitude(
        jnp.asarray(t), jnp.asarray(np.arange(8) % 4), filters["W"], filters["N"],
        filters["lam"], Z, DL_CM, 20.0, 20.0, 1.0, 1.0, 0.05, 0.1, 1.5, n_time=500)
    f_tde = m_tde.predict(p_tde, t, bb)
    ref_tde = 3631.0 * 10 ** (-0.4 * np.asarray(direct_tde))
    assert np.all(np.isfinite(f_tde))
    # the factory jits its core and the reference here is eager, so this also measures what
    # XLA's reassociation is worth on this path -- recorded, not hidden behind the tolerance
    rel_jit = float(np.max(np.abs(f_tde - ref_tde) / np.abs(ref_tde)))
    assert rel_jit < 1e-6, rel_jit

    raised = {}
    for tag, model, params in (("kilonova", m_kn, p_kn), ("tde", m_tde, p_tde)):
        key = list(params)[0]

        def loss(x, model=model, params=params, key=key):
            q = dict(params)
            q[key] = x
            return jnp.sum(jnp.asarray(model.predict(q, t, bb)))

        with pytest.raises(Exception) as exc:      # noqa: PT011 - the type IS the finding
            jax.grad(loss)(jnp.asarray(params[key]))
        raised[tag] = type(exc.value).__name__
    _record("factory_boundary", dict(predict_not_differentiable=raised,
                                     tde_parameters=m_tde.parameters,
                                     cooling_envelope_parameters=m_ce.parameters,
                                     cooling_envelope_pinned=pinned,
                                     tde_jit_vs_eager_max_rel=rel_jit))
    for tag, nm in raised.items():
        assert "Tracer" in nm or "Concretization" in nm, (tag, nm)


def test_factory_bound_model_is_differentiable_through_the_module_functions(filters):
    """The gradient a sampler would need, through exactly the arguments the factory binds --
    its filter set, its redshift, its ``mag_floor``, its ``n_time`` -- over its own prior.

    WOULD CATCH a factory that wired the module function with an argument which breaks the
    reverse pass (an extinction shape in the wrong frame, a filter set of the wrong dtype, a
    ``mag_floor`` that clamps every epoch to a zero-gradient plateau). The module-level sweeps
    above would not see it, because they build their own arguments.
    """
    import whisper_cbpf.models.jax as M

    fs = dict(lam=filters["lam_np"], trans=filters["trans_np"])
    bands = ["sdssu", "sdssg", "sdssr", "sdssi"]
    m = M.tde_model(bands, Z, DL_CM, filter_set=fs, rise="gaussian", n_time=500)
    names, theta = prior_draws(m.default_prior, _n(96), seed=91)
    assert names == m.parameters
    t = jnp.asarray(np.geomspace(1.0, 120.0, 12))
    bidx = jnp.asarray(np.arange(12) % 4)
    we = _weights(np.random.default_rng(6), 12)

    def loss(v):
        mag = T.gaussianrise_cooling_envelope_ab_magnitude(
            t, bidx, filters["W"], filters["N"], filters["lam"], Z, DL_CM,
            *[v[i] for i in range(7)], n_time=500)
        return jnp.sum(we * mag)

    g = batched(jax.jit(jax.vmap(jax.grad(loss))), 96)(theta)
    _record("factory_grad", dict(n_draw=int(theta.shape[0]),
                                 frac_nonfinite=float(np.mean(~np.isfinite(g))),
                                 frac_all_zero=float(np.mean(np.all(g == 0, axis=1)))))
    assert np.isfinite(g).all()
