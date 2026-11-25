"""The four phenomenological models, adapted from redback-jax and redback, as tests.

``smooth_exponential_powerlaw``, ``exp_rise_powerlaw_decline``, ``bazin_sne`` and ``villar_sne``
(``whisper_cbpf.models.jax.phenomenological``) are closed-form per-band functions. The claims:

1. **They reproduce redback.** 150 seeded draws of redback's own prior boxes, each on an epoch grid
   placed around the features of that draw. Budgets are set from the measured worst case (in the
   comments) with a factor of 3 to 10 of headroom for another libm. The one wide budget is
   ``smooth_exponential_powerlaw``'s, and it is redback's: ``0.5 (1 + tanh(.))`` and
   ``1 - 0.5 (1 + tanh(.))`` cancel in the blended tail, so redback is off its own exact value by up
   to 7e-6 there; the port is within 4e-15 of a 60-digit evaluation.
2. **They reproduce redback-jax** at identical parameters, in a subprocess: importing
   ``redback_jax.models`` turns ``jax_enable_x64`` on for the whole process, and this suite shares one.
3. **Gradients are finite everywhere the prior reaches**, including epochs before ``t0``, where the
   unclamped logarithm of ``exp_rise_powerlaw_decline`` has NaN gradients (a control reproduces that).
4. **float32 behaves as documented.**
5. **The whisper ``Model``** carries redback's parameter names, one amplitude per band, flux density
   in Jy, a ``predict_jax`` equal to ``predict``, and survives pickling.
6. **A fit recovers a known truth** with the default sampler on a CPU.

float64 is switched on per test (and restored), never for the session.
"""
from __future__ import annotations

import doctest
import json
import os
import pickle
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import whisper_cbpf as wp  # noqa: E402
from whisper_cbpf.io.photometry import AB_ZEROPOINT_JY  # noqa: E402
from whisper_cbpf.models.jax import _phenomenological_spec as spec  # noqa: E402
from whisper_cbpf.models.jax import phenomenological as ph  # noqa: E402
from whisper_cbpf.priors import LogUniform, Uniform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEYS = list(spec.MODELS)
N_DRAWS = 150


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _args(key, p, t, bands_amp=None):
    """The positional arguments of the public function, in redback's order."""
    if key == "smooth_exponential_powerlaw":
        return (t, p["a_1"], p["tpeak"], p["alpha_1"], p["alpha_2"], p["smoothing_factor"])
    if key == "exp_rise_powerlaw_decline":
        return (t, p["t0"], p["m_peak"], p["tau_rise"], p["alpha"], p["t_peak"])
    if key == "bazin_sne":
        return (t, p["aa"], p["bb"], p["t0"], p["tau_rise"], p["tau_fall"])
    return (t, p["aa"], p["cc"], p["t0"], p["tau_rise"], p["tau_fall"], p["gamma"], p["nu"])


def _grid(key, p):
    """Epochs placed around this draw's features, including epochs before the start."""
    if key == "smooth_exponential_powerlaw":
        return p["tpeak"] * np.geomspace(1e-3, 100.0, 80)
    if key == "exp_rise_powerlaw_decline":
        return p["t0"] + (p["t_peak"] - p["t0"]) * np.linspace(-0.05, 3.0, 80)
    return np.linspace(-20.0, 400.0, 80)


def _draws(key, n=N_DRAWS, seed=12345):
    rng = np.random.default_rng(seed)
    prior = spec.default_prior(key)
    return [prior.sample(rng) for _ in range(n)]


def _rel(a, b):
    """Elementwise ``|a - b| / |b|`` over the points where both are finite; the NaN masks must agree."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert (np.isnan(a) == np.isnan(b)).all(), "the NaN pattern differs"
    ok = np.isfinite(a) & np.isfinite(b)
    return np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300)


# --- 0. the spec, and redback's own boxes ------------------------------------------------------

def _redback_prior_file(name):
    from whisper_cbpf.models.redback_adapter import _import_redback

    redback = _import_redback()
    bilby = pytest.importorskip("bilby")
    path = Path(redback.__file__).parent / "priors" / f"{name}.prior"
    return bilby.prior.PriorDict(filename=str(path))


def _bounds(dist):
    return float(dist.bounds[0]), float(dist.bounds[1])


@pytest.mark.parametrize("key,file,rename", [
    ("bazin_sne", "bazin_sne", {}),
    ("villar_sne", "villar_sne", {}),
    ("exp_rise_powerlaw_decline", "exp_rise_powerlaw_decline", {}),
    ("smooth_exponential_powerlaw", "smooth_exponential_powerlaw_cooling_envelope_bolometric",
     {"peak_time": "tpeak"}),
])
def test_default_prior_is_redbacks_box(key, file, rename):
    """The transcribed boxes equal the installed redback's ``.prior`` files (type and bounds)."""
    pytest.importorskip("redback")
    theirs = _redback_prior_file(file)
    ours = spec.default_prior(key).distributions
    for rb_name, dist in theirs.items():
        name = rename.get(rb_name, rb_name)
        if name not in ours:
            continue                       # tau_decline, and the cooling envelope's own parameters
        assert type(ours[name]).__name__ == type(dist).__name__, name
        assert _bounds(ours[name]) == (float(dist.minimum), float(dist.maximum)), name
    shared = set(ours) - {"a_1"} if key == "smooth_exponential_powerlaw" else set(ours)
    assert shared <= {rename.get(k, k) for k in theirs}, "a parameter of the model has no redback box"


def test_parameter_names_are_redbacks_with_bands_appended():
    import inspect

    pytest.importorskip("redback")
    from whisper_cbpf.models.redback_adapter import _import_redback

    _import_redback()
    import redback.transient_models.phenomenological_models as pm

    for key in KEYS:
        sig = list(inspect.signature(getattr(pm, key)).parameters)[1:]
        sig = [s for s in sig if s != "kwargs"]
        assert spec.parameter_names(key) == sig, key
        banded = spec.parameter_names(key, ["g", "r"])
        expected = []
        for s in sig:
            expected += [f"{s}_g", f"{s}_r"] if s in spec.SPECS[key]["per_band"] else [s]
        assert banded == expected


# --- 1. parity with redback --------------------------------------------------------------------

@pytest.fixture(scope="module")
def pm():
    pytest.importorskip("redback")
    from whisper_cbpf.models.redback_adapter import _import_redback

    _import_redback()
    import redback.transient_models.phenomenological_models as module

    return module


def _quiet():
    return np.errstate(all="ignore")


#: Measured over 200 draws (seed 12345): bazin 5.3e-15, villar 7.3e-15 (relative); exp_rise 1.3e-12 mag.
@pytest.mark.parametrize("key,budget", [("bazin_sne", 5e-14), ("villar_sne", 5e-14)])
def test_flux_functions_match_redback_to_round_off(x64, pm, key, budget):
    worst = 0.0
    with _quiet():
        for p in _draws(key):
            t = _grid(key, p)
            args = _args(key, {k: float(v) for k, v in p.items()}, t)
            worst = max(worst, _rel(getattr(ph, key)(*args), getattr(pm, key)(*args)).max())
    assert worst < budget, worst


def test_exp_rise_matches_redback_in_magnitude_and_in_where_it_is_nan(x64, pm):
    worst, nan_epochs = 0.0, 0
    with _quiet():
        for p in _draws("exp_rise_powerlaw_decline"):
            t = _grid("exp_rise_powerlaw_decline", p)
            m_peak = np.array([p["m_peak"], p["m_peak"] + 0.5, p["m_peak"] - 0.3])   # three filters
            args = (t, p["t0"], m_peak, p["tau_rise"], p["alpha"], p["t_peak"])
            ours, theirs = np.asarray(ph.exp_rise_powerlaw_decline(*args)), pm.exp_rise_powerlaw_decline(*args)
            assert ours.shape == theirs.shape == (80, 3)
            assert (np.isnan(ours) == np.isnan(theirs)).all()
            nan_epochs += int(np.isnan(ours).any(axis=1).sum())
            worst = max(worst, float(np.nanmax(np.abs(ours - theirs))))
    assert nan_epochs > 0, "the grid never went before t0"
    assert worst < 1e-11, worst


def test_exp_rise_scalar_and_delta_follow_redback(x64, pm):
    t = np.linspace(58990.0, 59050.0, 300)
    ours = ph.exp_rise_powerlaw_decline(t, 59000.0, 17.0, 3.0, 1.5, 59010.0)
    with _quiet():
        theirs = pm.exp_rise_powerlaw_decline(t, 59000.0, 17.0, 3.0, 1.5, 59010.0)
    assert ours.shape == theirs.shape == (300,)
    assert np.nanmax(np.abs(np.asarray(ours) - theirs)) < 1e-11
    ours = ph.exp_rise_powerlaw_decline(t, 59000.0, 17.0, 3.0, 1.5, 59010.0, delta=0.2)
    with _quiet():
        theirs = pm.exp_rise_powerlaw_decline(t, 59000.0, 17.0, 3.0, 1.5, 59010.0, delta=0.2)
    assert np.nanmax(np.abs(np.asarray(ours) - theirs)) < 1e-11


def test_smooth_power_law_matches_redback_to_redbacks_own_cancellation(x64, pm):
    """Measured 1.1e-5 over 200 draws. The difference is redback's rounding, not the port's: see
    :func:`test_smooth_power_law_matches_a_high_precision_evaluation`."""
    key, worst = "smooth_exponential_powerlaw", 0.0
    with _quiet():
        for p in _draws(key):
            t = _grid(key, p)
            args = _args(key, {k: float(v) for k, v in p.items()}, t)
            worst = max(worst, _rel(ph.smooth_exponential_powerlaw(*args), pm.smooth_exponential_powerlaw(*args)).max())
    assert worst < 1e-4, worst


def test_smooth_power_law_matches_a_high_precision_evaluation(x64):
    """Against 60-digit arithmetic: the port 3.5e-15 (measured), redback 7.4e-6."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    worst = 0.0
    for p in _draws("smooth_exponential_powerlaw", n=40, seed=5):
        t = _grid("smooth_exponential_powerlaw", p)[::4]
        ours = np.asarray(ph.smooth_exponential_powerlaw(t, p["a_1"], p["tpeak"], p["alpha_1"],
                                                         p["alpha_2"], p["smoothing_factor"]))
        for ti, fi in zip(t, ours):
            x = mp.mpf(float(ti)) / mp.mpf(float(p["tpeak"]))
            w = (1 + mp.tanh(mp.mpf(float(p["smoothing_factor"])) * mp.log(x))) / 2
            exact = mp.mpf(float(p["a_1"])) * (x ** mp.mpf(float(p["alpha_1"])) * (1 - w)
                                               + x ** mp.mpf(float(p["alpha_2"])) * w)
            worst = max(worst, float(abs((mp.mpf(float(fi)) - exact) / exact)))
    assert worst < 1e-13, worst


def test_per_band_amplitudes_follow_redbacks_matrix_layout(x64, pm):
    """``bazin_sne`` with array ``aa``/``bb`` is ``(n_band, n_time)``, as in redback; ``villar_sne``
    takes the same arrays."""
    t = np.linspace(-10.0, 200.0, 50)
    aa, bb = np.array([3.0, 2.0, 1.0]), np.array([0.1, 0.0, 0.3])
    ours = np.asarray(ph.bazin_sne(t, aa, bb, 5.0, 3.0, 30.0))
    theirs = pm.bazin_sne(t, list(aa), list(bb), 5.0, 3.0, 30.0)
    assert ours.shape == theirs.shape == (3, 50)
    assert _rel(ours, theirs).max() < 5e-14
    v = np.asarray(ph.villar_sne(t, aa, bb, 5.0, 3.0, 30.0, 20.0, 0.4))
    assert v.shape == (3, 50)
    for i in range(3):
        row = pm.villar_sne(t, aa[i], bb[i], 5.0, 3.0, 30.0, 20.0, 0.4)
        assert _rel(v[i], row).max() < 5e-14
    with pytest.raises(ValueError, match="same shape"):
        ph.bazin_sne(t, aa, bb[:2], 5.0, 3.0, 30.0)


def test_villar_keeps_the_input_order_where_redback_concatenates(x64, pm):
    t = np.array([100.0, 5.0, 50.0, 20.0, 300.0, -4.0])
    args = (2.0, 0.1, 0.0, 3.0, 30.0, 25.0, 0.5)
    ours = np.asarray(ph.villar_sne(t, *args))
    theirs = pm.villar_sne(t, *args)               # branch one first, then branch two
    early = t < 0.0 + 25.0
    order = np.concatenate([np.flatnonzero(early), np.flatnonzero(~early)])
    assert not np.array_equal(order, np.arange(t.size))
    assert _rel(ours[order], theirs).max() < 1e-13
    assert np.array_equal(ours, np.asarray(ph.villar_sne(np.sort(t), *args))[np.argsort(np.argsort(t))])


def test_smooth_power_law_is_zero_where_redback_is_zero_or_nan(x64, pm):
    t = np.array([-5.0, 0.0, 3.0])
    ours = np.asarray(ph.smooth_exponential_powerlaw(t, 2.0, 10.0, 1.5, -2.0, 1.0))
    assert np.array_equal(ours[:2], [0.0, 0.0]) and ours[2] > 0
    with _quiet():
        theirs = pm.smooth_exponential_powerlaw(t, 2.0, 10.0, 1.5, -2.0, 1.0)
        shallow = pm.smooth_exponential_powerlaw(t, 2.0, 10.0, 1.5, -0.8, 1.0)
    assert np.isnan(theirs[:2]).all() and np.array_equal(shallow[:2], [0.0, 0.0])
    assert abs(ours[2] - theirs[2]) / theirs[2] < 1e-12


# --- 2. parity with redback-jax, in a subprocess ----------------------------------------------

_RBJAX_CHILD = r"""
import json, sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import whisper_cbpf.models.jax.phenomenological as P
import redback_jax.phenomenological_models as R
from whisper_cbpf.models.jax._phenomenological_spec import default_prior
rng = np.random.default_rng(7)
out = {}
for key in ["smooth_exponential_powerlaw", "exp_rise_powerlaw_decline", "bazin_sne", "villar_sne"]:
    prior, worst = default_prior(key), 0.0
    for _ in range(100):
        p = prior.sample(rng)
        if key == "bazin_sne":
            t = np.linspace(-20, 400, 80); a = (t, p["aa"], p["bb"], p["t0"], p["tau_rise"], p["tau_fall"])
        elif key == "villar_sne":
            t = np.linspace(-20, 400, 80)
            a = (t, p["aa"], p["cc"], p["t0"], p["tau_rise"], p["tau_fall"], p["gamma"], p["nu"])
        elif key == "smooth_exponential_powerlaw":
            t = p["tpeak"] * np.geomspace(1e-3, 100, 80)
            a = (t, p["a_1"], p["tpeak"], p["alpha_1"], p["alpha_2"], p["smoothing_factor"])
        else:
            t = p["t0"] + (p["t_peak"] - p["t0"]) * np.linspace(-0.05, 3, 80)
            a = (t, p["t0"], np.array([p["m_peak"], p["m_peak"] + 0.5]), p["tau_rise"], p["alpha"], p["t_peak"])
        x, y = np.asarray(getattr(P, key)(*a)), np.asarray(getattr(R, key)(*a))
        assert (np.isnan(x) == np.isnan(y)).all(), key
        ok = np.isfinite(x) & np.isfinite(y)
        err = np.abs(x[ok] - y[ok]) if key == "exp_rise_powerlaw_decline" else \
            np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1e-300)
        worst = max(worst, float(err.max()))
    out[key] = worst
print(json.dumps(out))
"""

#: Measured over 200 draws: smooth 3.3e-5 (the tanh cancellation of the blended tail), exp_rise 0 (the same
#: arithmetic), bazin 3.2e-15, villar 4.2e-16.
_RBJAX_BUDGET = {"smooth_exponential_powerlaw": 2e-4, "exp_rise_powerlaw_decline": 1e-11,
                 "bazin_sne": 5e-14, "villar_sne": 5e-14}


@pytest.mark.skipif(find_spec("redback_jax") is None, reason="redback-jax is not installed")
def test_functions_match_redback_jax_at_identical_parameters():
    done = subprocess.run([sys.executable, "-c", _RBJAX_CHILD], capture_output=True, text=True,
                          cwd=ROOT, env={**os.environ, "JAX_PLATFORMS": "cpu"}, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    worst = json.loads(done.stdout.strip().splitlines()[-1])
    for key, budget in _RBJAX_BUDGET.items():
        assert worst[key] < budget, (key, worst)


# --- 3. gradients ------------------------------------------------------------------------------

def _gradient_params(key):
    """Names and a vector of the differentiable arguments (the epochs excluded) for one draw."""
    return [n for n in spec.SPECS[key]["parameters"]]


def _grad_fn(key):
    """``jit(grad)`` of the summed finite output with respect to every argument but the epochs."""
    names = _gradient_params(key)

    def loss(theta, t):
        out = getattr(ph, key)(*_args(key, dict(zip(names, theta)), t))
        return jnp.sum(jnp.where(jnp.isfinite(out), out, 0.0))
    return jax.jit(jax.grad(loss))


@pytest.mark.parametrize("key", KEYS)
def test_gradients_are_finite_across_the_prior_including_epochs_before_the_start(x64, key):
    bad, grad = [], _grad_fn(key)
    for p in _draws(key, n=60, seed=3):
        t = jnp.asarray(_grid(key, p))
        theta = jnp.asarray([float(p[n]) for n in _gradient_params(key)])
        g = grad(theta, t)
        if not bool(jnp.all(jnp.isfinite(g))):
            bad.append(({k: float(v) for k, v in p.items()}, np.asarray(g)))
    assert not bad, bad[:2]


@pytest.mark.parametrize("key", KEYS)
def test_gradients_are_finite_in_float32_too(key):
    bad, grad = 0, _grad_fn(key)
    with jax.enable_x64(False):
        for p in _draws(key, n=30, seed=4):
            t = _grid(key, p)
            q = {k: float(v) for k, v in p.items()}
            if key == "exp_rise_powerlaw_decline":           # a relative clock, as documented
                t, q["t_peak"], q["t0"] = t - q["t0"], q["t_peak"] - q["t0"], 0.0
            theta = jnp.asarray([q[n] for n in _gradient_params(key)])
            bad += int(not bool(jnp.all(jnp.isfinite(grad(theta, jnp.asarray(t))))))
    assert bad == 0


def test_the_unclamped_logarithm_has_nan_gradients_so_the_clamp_is_load_bearing(x64):
    """Control. A body that evaluates ``log10`` at the negative ratios before ``t0`` and masks
    afterwards: ``0 * nan`` then poisons the backward pass. The ported function does not."""
    def unclamped(t, t0, m_peak, tau_rise, alpha, t_peak):
        width = (t_peak - t0) * 0.5
        weight = 0.5 * (1 + jnp.tanh((t - t_peak) / width))
        m_rise = m_peak + 1.086 * ((t_peak - t) / tau_rise)
        m_decline = m_peak + 2.5 * alpha * jnp.log10((t - t0) / (t_peak - t0))
        return jnp.where(t >= t0, (1 - weight) * m_rise + weight * m_decline, jnp.nan)

    t = jnp.array([-3.0, -1.0, 2.0, 8.0, 20.0])          # two epochs precede t0 = 0

    def grad_of(fn, argnum):
        def total(*theta):
            m = fn(t, *theta)
            return jnp.sum(jnp.where(jnp.isnan(m), 0.0, m))
        return jax.grad(total, argnums=argnum)(0.0, 17.0, 3.0, 1.5, 10.0)

    # m_peak enters the logarithm only additively, so its gradient survives; alpha and t0 do not
    for argnum, name in ((3, "alpha"), (0, "t0")):
        assert not bool(jnp.isfinite(grad_of(unclamped, argnum))), name
        assert bool(jnp.isfinite(grad_of(ph.exp_rise_powerlaw_decline, argnum))), name


def test_gradient_of_the_exp_rise_magnitude_is_the_blend_weight_complement(x64):
    """d m / d m_peak is 1 for every valid epoch (both phases add m_peak): a value check of the
    gradient that the fix must not have disturbed."""
    t = jnp.array([-3.0, 2.0, 8.0, 20.0])
    g = jax.grad(lambda m: jnp.sum(jnp.where(jnp.isnan(
        ph.exp_rise_powerlaw_decline(t, 0.0, m, 3.0, 1.5, 10.0)), 0.0,
        ph.exp_rise_powerlaw_decline(t, 0.0, m, 3.0, 1.5, 10.0))))(17.0)
    assert float(g) == pytest.approx(3.0, abs=1e-12)


def test_extreme_corners_stay_finite_in_value_and_gradient(x64):
    """``tau_rise = 0.01`` overflows ``exp(-dt/tau_rise)`` 7 d before ``t0``; the plateau branch of
    Villar overflows far before ``t0`` with ``tau_fall = 1``."""
    t = jnp.linspace(-300.0, 500.0, 200)
    f = jax.jit(jax.grad(lambda th: jnp.sum(ph.bazin_sne(t, th[0], th[1], th[2], th[3], th[4]))))
    g = f(jnp.array([3.0, 0.1, 40.0, 0.01, 1.0]))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert bool(jnp.all(jnp.isfinite(ph.bazin_sne(t, 3.0, 0.1, 40.0, 0.01, 1.0))))
    f = jax.jit(jax.grad(lambda th: jnp.sum(ph.villar_sne(t, th[0], th[1], th[2], th[3], th[4], th[5], th[6]))))
    g = f(jnp.array([3.0, 0.1, 40.0, 0.01, 1.0, 0.1, 0.99]))
    assert bool(jnp.all(jnp.isfinite(g)))


# --- 4. float32 --------------------------------------------------------------------------------

def _both_precisions(key, p, relative_clock=False):
    t = _grid(key, p)
    q = {k: float(v) for k, v in p.items()}
    if relative_clock:
        t = t - q["t0"]
        q["t_peak"] -= q["t0"]
        q["t0"] = 0.0
    args = _args(key, q, t)
    with jax.enable_x64(False):
        low = np.asarray(getattr(ph, key)(*args))
        assert low.dtype == np.float32
    with jax.enable_x64(True):
        high = np.asarray(getattr(ph, key)(*args))
        assert high.dtype == np.float64
    return low.astype(float), high


#: Measured over 200 draws, float32 against float64: smooth 2.2e-6, bazin 6.9e-6 (maximum relative
#: error); villar's 99th percentile is 6.9e-6 (its maximum is 1 where the flux underflows float32).
@pytest.mark.parametrize("key,budget", [("smooth_exponential_powerlaw", 3e-5), ("bazin_sne", 5e-5)])
def test_float32_relative_error_is_small(key, budget):
    worst = 0.0
    with _quiet():
        for p in _draws(key, n=100, seed=3):
            low, high = _both_precisions(key, p)
            ok = np.isfinite(low) & np.isfinite(high) & (np.abs(high) > 1e-30)
            worst = max(worst, float((np.abs(low[ok] - high[ok]) / np.abs(high[ok])).max()))
    assert worst < budget, worst


def test_float32_villar_is_accurate_except_where_the_flux_underflows():
    errs = []
    with _quiet():
        for p in _draws("villar_sne", n=100, seed=3):
            low, high = _both_precisions("villar_sne", p)
            ok = np.isfinite(low) & np.isfinite(high) & (np.abs(high) > 1e-30)
            errs.append(np.abs(low[ok] - high[ok]) / np.abs(high[ok]))
    assert np.quantile(np.concatenate(errs), 0.99) < 5e-5


def test_float32_smooth_power_law_keeps_its_tails():
    """The tanh form lost the tail in float32 (relative error up to 1); the logistic weights do not."""
    worst = 0.0
    with _quiet():
        for p in _draws("smooth_exponential_powerlaw", n=100, seed=9):
            low, high = _both_precisions("smooth_exponential_powerlaw", p)
            ok = np.isfinite(low) & (np.abs(high) > 1e-30)
            worst = max(worst, float((np.abs(low[ok] - high[ok]) / np.abs(high[ok])).max()))
    assert worst < 1e-4, worst


def test_float32_exp_rise_needs_a_clock_near_the_event():
    """Measured: within 1.6e-3 mag on a relative clock, 1.3e-2 mag on raw MJD (worst of 200 draws)."""
    worst = {True: 0.0, False: 0.0}
    for rel in worst:
        for p in _draws("exp_rise_powerlaw_decline", n=100, seed=3):
            low, high = _both_precisions("exp_rise_powerlaw_decline", p, relative_clock=rel)
            ok = np.isfinite(low) & np.isfinite(high)
            worst[rel] = max(worst[rel], float(np.abs(low[ok] - high[ok]).max()))
    assert worst[True] < 5e-3, worst
    assert worst[False] > worst[True]            # the raw clock is the worse one


def test_float32_predict_moves_the_clock_on_the_host():
    """``Model.predict`` subtracts the first epoch in float64 before the cast, so MJD epochs and
    MJD ``t0``/``t_peak`` cost nothing a relative clock would not."""
    m = ph.phenomenological_model("exp_rise_powerlaw_decline")
    p = dict(t0=59000.0, m_peak=17.0, tau_rise=3.0, alpha=1.5, t_peak=59010.0)
    t = 59000.0 + np.linspace(-2.0, 50.0, 40)
    with jax.enable_x64(True):
        truth = m.predict(p, t)
    with jax.enable_x64(False):
        low = m.predict(p, t)
    live = truth > 0
    assert np.abs(2.5 * np.log10(low[live] / truth[live])).max() < 5e-3


# --- 5. the whisper Model ----------------------------------------------------------------------

BANDS = ["g", "r", "i"]
TRUTH = {
    "smooth_exponential_powerlaw": dict(a_1=3e-4, tpeak=12.0, alpha_1=1.5, alpha_2=-1.2,
                                        smoothing_factor=1.0),
    "exp_rise_powerlaw_decline": dict(t0=0.0, m_peak=18.0, tau_rise=3.0, alpha=1.5, t_peak=12.0),
    "bazin_sne": dict(aa=3e-4, bb=1e-5, t0=2.0, tau_rise=3.0, tau_fall=25.0),
    "villar_sne": dict(aa=3e-4, cc=1e-5, t0=2.0, tau_rise=3.0, tau_fall=25.0, gamma=15.0, nu=0.3),
}


def _flux_fn(key):
    """The model's reference flux density [Jy] from the public function."""
    def flux(t, **p):
        if key == "exp_rise_powerlaw_decline":
            m = np.asarray(ph.exp_rise_powerlaw_decline(t, p["t0"], p["m_peak"], p["tau_rise"],
                                                        p["alpha"], p["t_peak"]))
            return np.where(np.isnan(m), 0.0, AB_ZEROPOINT_JY * 10.0 ** (-0.4 * np.nan_to_num(m)))
        return np.asarray(getattr(ph, key)(*_args(key, p, t)))
    return flux


@pytest.mark.parametrize("key", KEYS)
def test_band_independent_model_is_the_function_in_jy(x64, key):
    m = ph.phenomenological_model(key)
    assert m.name == f"{key}_jax" and m.parameters == spec.parameter_names(key)
    assert m.predict_jax is not None and not hasattr(m.predict_jax, "band_index")
    t = np.linspace(-2.0, 80.0, 40)
    flux = m.predict(TRUTH[key], t, None)
    assert flux.dtype == np.float64 and flux.shape == t.shape and np.isfinite(flux).all()
    np.testing.assert_allclose(flux, _flux_fn(key)(t, **TRUTH[key]), rtol=1e-12)
    # the band is ignored
    np.testing.assert_array_equal(flux, m.predict(TRUTH[key], t, np.array(["g"] * t.size)))


@pytest.mark.parametrize("key", KEYS)
def test_predict_jax_equals_predict_and_batches_with_vmap(x64, key):
    m = ph.phenomenological_model(key, BANDS)
    rng = np.random.default_rng(2)
    prior = m.default_prior
    rows = [prior.sample(rng) for _ in range(5)]
    if key == "exp_rise_powerlaw_decline":
        rows = [dict(r, t0=0.0, t_peak=12.0) for r in rows]
    t = np.linspace(-2.0, 90.0, 30)
    bands = np.array(BANDS * 10)
    bidx = m.predict_jax.band_index(bands)
    theta = jnp.asarray([[r[k] for k in m.parameters] for r in rows])
    batched = jax.jit(jax.vmap(lambda th: m.predict_jax(th, t, bidx)))(theta)
    for r, row in zip(rows, np.asarray(batched)):
        np.testing.assert_allclose(row, m.predict(r, t, bands), rtol=1e-12, atol=1e-300)


def _banded_params(key):
    """TRUTH with each per-band argument spread over BANDS (``aa_g, aa_r, aa_i``), in model order."""
    params = {}
    for name, value in TRUTH[key].items():
        if name in spec.SPECS[key]["per_band"]:
            for k, b in enumerate(BANDS):
                params[f"{name}_{b}"] = value + (0.5 * k if key == "exp_rise_powerlaw_decline"
                                                 else value * 0.5 * k)
        else:
            params[name] = value
    return params


@pytest.mark.parametrize("key", KEYS)
def test_each_band_takes_its_own_amplitude_and_shares_the_shape(x64, key):
    m = ph.phenomenological_model(key, BANDS)
    spec_ = spec.SPECS[key]
    params = _banded_params(key)
    assert list(params) == m.parameters
    t = np.linspace(1.0, 60.0, 12)
    for k, b in enumerate(BANDS):
        own = {n: (params[f"{n}_{b}"] if n in spec_["per_band"] else v) for n, v in TRUTH[key].items()}
        np.testing.assert_allclose(m.predict(params, t, np.array([b] * t.size)),
                                   _flux_fn(key)(t, **own), rtol=1e-12)
    mixed = m.predict(params, np.tile(t, 3), np.repeat(BANDS, t.size))
    assert not np.allclose(mixed[:12], mixed[12:24])


def test_exp_rise_model_is_ab_flux_density_and_zero_before_t0(x64):
    m = ph.phenomenological_model("exp_rise_powerlaw_decline")
    p = TRUTH["exp_rise_powerlaw_decline"]
    t = np.array([-5.0, -0.1, 0.5, 12.0, 60.0])
    flux = m.predict(p, t, None)
    assert (flux[:2] == 0.0).all() and (flux[2:] > 0).all() and np.isfinite(flux).all()
    peak_mag = float(ph.exp_rise_powerlaw_decline(np.array([12.0]), 0.0, 18.0, 3.0, 1.5, 12.0)[0])
    assert flux[3] == pytest.approx(AB_ZEROPOINT_JY * 10 ** (-0.4 * peak_mag), rel=1e-12)
    assert peak_mag == pytest.approx(18.0, abs=1e-9)         # both phases equal m_peak at t_peak


@pytest.mark.parametrize("key", KEYS)
def test_model_pickles_and_works_in_a_fresh_copy(key):
    m = ph.phenomenological_model(key, BANDS, band_aliases={"G": "g"})
    params = _banded_params(key)
    t, b = np.linspace(1.0, 50.0, 9), np.array(["G", "r", "i"] * 3)
    before = m.predict(params, t, b)
    clone = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(before, clone.predict(params, t, b))
    assert clone.predict_jax.band_index(b).tolist() == [0, 1, 2] * 3


def test_band_errors_name_the_problem():
    m = ph.phenomenological_model("bazin_sne", ["g", "r"])
    params = dict(aa_g=1.0, aa_r=1.0, bb_g=0.0, bb_r=0.0, t0=0.0, tau_rise=2.0, tau_fall=20.0)
    t = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="needs a `bands` array"):
        m.predict(params, t, None)
    with pytest.raises(KeyError, match="was not bound"):
        m.predict(params, t, np.array(["g", "z"]))
    with pytest.raises(ValueError, match="needs a `band_idx` array"):
        m.predict_jax(jnp.ones(7), t, None)
    with pytest.raises(ValueError, match="unknown phenomenological model"):
        ph.phenomenological_model("not_a_model")
    with pytest.raises(ValueError, match="band_aliases needs band_names"):
        ph.phenomenological_model("bazin_sne", band_aliases={"g": "g"})
    with pytest.raises(ValueError, match="without repeats"):
        ph.phenomenological_model("bazin_sne", ["g", "g"])
    assert m.predict(params, np.array([]), np.array([])).shape == (0,)


def test_prior_override_by_full_name_or_by_redbacks_name():
    custom = {"aa": LogUniform(1e-6, 1e-2), "t0": Uniform(-5, 5), "bb_r": Uniform(0, 1e-5)}
    m = ph.phenomenological_model("bazin_sne", ["g", "r"], prior=custom)
    d = m.default_prior.distributions
    assert d["aa_g"].bounds == d["aa_r"].bounds == (1e-6, 1e-2)
    assert d["bb_r"].bounds == (0.0, 1e-5) and d["bb_g"].bounds == (1e-15, 1e15)
    assert d["t0"].bounds == (-5.0, 5.0) and list(d) == m.parameters
    with pytest.raises(ValueError, match="not a parameter"):
        ph.phenomenological_model("bazin_sne", prior={"nope": Uniform(0, 1)})


def test_register_phenomenological_registers_a_usable_model():
    from whisper_cbpf.models import _REGISTRY

    name = "adopted_phenomenological_test_model"
    try:
        m = ph.register_phenomenological("villar_sne", ["g", "r"], name=name)
        got = wp.get_model(name)
        assert got is not None and got.parameters == m.parameters and got.predict_jax is not None
        assert got.default_prior is not None
    finally:
        _REGISTRY.pop(name, None)


def test_docstring_examples_run():
    from whisper_cbpf.models import _REGISTRY

    before = set(_REGISTRY)
    try:
        for module in (ph, spec):
            result = doctest.testmod(module, optionflags=doctest.ELLIPSIS, verbose=False)
            assert result.failed == 0 and result.attempted > 0, (module.__name__, result)
    finally:
        for name in set(_REGISTRY) - before:
            _REGISTRY.pop(name, None)


def test_the_module_sets_no_global_precision():
    code = ("import jax; import whisper_cbpf.models.jax.phenomenological as P; "
            "assert not jax.config.jax_enable_x64; P.bazin_sne([1.0], 1.0, 0.0, 0.0, 1.0, 1.0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"}, timeout=300)
    assert done.returncode == 0, done.stderr[-1500:]


# --- 6. a fit that recovers a known truth --------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cpu_mcmc_recovers_the_truth_of_a_two_band_bazin():
    """Simulate with known parameters, fit with ``wp.fit`` and CPU ``mcmc`` (the default on a CPU).

    The chain is short on purpose (about 10 s), too short for the convergence check, so the tolerances
    are loose: 15% on the well-determined amplitudes and timescales, 3 d on ``t0``, 4 sigma on all."""
    m = ph.phenomenological_model("bazin_sne", ["g", "r"], prior={
        "aa": LogUniform(1e-5, 1e-2), "bb": Uniform(0.0, 2e-5), "t0": Uniform(-10.0, 10.0),
        "tau_rise": Uniform(0.5, 15.0), "tau_fall": Uniform(5.0, 80.0)})
    truth = dict(aa_g=2e-4, aa_r=3e-4, bb_g=2e-6, bb_r=4e-6, t0=0.0, tau_rise=4.0, tau_fall=30.0)
    t, band = np.tile(np.linspace(-5.0, 90.0, 24), 2), np.repeat(["g", "r"], 24)
    flux = m.predict(truth, t, band)
    sigma = np.full_like(flux, 0.05 * flux.max())
    rng = np.random.default_rng(1)
    lc = wp.LightCurve(time=t, band=band, flux=flux + rng.normal(0.0, sigma), flux_err=sigma)
    res = wp.fit(lc, m, sampler="mcmc", nwalkers=16, nsteps=900, burnin=300, seed=0)
    assert res.sampler == "mcmc"
    for name in ("aa_g", "aa_r", "tau_rise", "tau_fall"):
        med = res.summary[name]["median"]
        assert abs(med - truth[name]) < 0.15 * truth[name], (name, med, truth[name])
    assert abs(res.summary["t0"]["median"] - truth["t0"]) < 3.0
    for name, value in truth.items():                 # no truth is more than 4 posterior sigma off
        s = res.summary[name]
        assert abs(s["median"] - value) < 4.0 * s["std"] + 1e-12, (name, s["median"], s["std"], value)
