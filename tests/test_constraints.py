"""redback's ``Constraint`` priors as a hard wall.

Up to whisper 0.1.0 they were dropped (``redback_adapter._translated_prior`` skipped them), so the
samplers explored parameters redback rejects. Three claims:

1. **The transcription decides exactly as redback does.** ``whisper_cbpf.models.constraints`` is
   compared with redback's own ``get_priors(model, constraint=True).evaluate_constraints`` on 20 000
   draws of redback's prior per model, in numpy and in JAX: identical accept/reject in
   ``constraint="redback"`` mode.
2. **The default corrects one number.** ``"corrected"`` bounds the Arnett kinetic energy by
   1.51e18 erg/g of nickel (14 He-4 -> Ni-56, 87.85 MeV per 56 u), not redback's 1.91e19; every
   other decision is redback's.
3. **The wall is applied where the samplers look.** The CPU adapter predicts zero flux for a draw
   that breaks one; the JAX samplers' batched forward map predicts zero and their log-density is
   ``-inf``. ``constraint=None`` restores whisper <= 0.1.0.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

N_DRAWS = 20_000
#: The reference comparison families by redback name, plus the two JAX-
#: ported magnetar models redback also constrains. The first four have constraints in redback 1.20.
FAMILIES = ["arnett", "basic_magnetar_powered", "cooling_envelope",
            "gaussianrise_cooling_envelope", "slsn", "general_magnetar_slsn",
            "shock_cooling_and_arnett", "csm_shock_and_arnett", "type_1a", "type_1c",
            "one_component_kilonova_model", "two_component_kilonova_model"]
#: Inside arnett.prior; breaks the corrected bound (kinetic/burning = 10.3) and passes redback's
#: (0.82): the draw that tells the two modes apart.
ARNETT_BETWEEN = dict(f_nickel=0.05, mej=2.0, vej=2.5e4, kappa=0.1, kappa_gamma=0.03,
                      temperature_floor=4000.0)
#: ... and one that breaks both (kinetic/burning 2.1e3 corrected, 163 redback).
ARNETT_BAD = dict(f_nickel=1e-3, mej=2.0, vej=5e4, kappa=0.1, kappa_gamma=0.03,
                  temperature_floor=4000.0)
#: ... and one that passes both (0.83 corrected).
ARNETT_OK = dict(f_nickel=0.1, mej=2.0, vej=1e4, kappa=0.1, kappa_gamma=0.03,
                 temperature_floor=4000.0)
Z = 0.05
#: The redback models whose constraints are transcribed for the JAX ports.
TRANSCRIBED = ["arnett", "basic_magnetar_powered", "cooling_envelope",
               "gaussianrise_cooling_envelope", "general_magnetar_slsn", "slsn"]


def _C():
    """The constraints module, imported per test so the behaviour tests (claim 3) run -- and fail
    on their assertions -- against a whisper that predates it."""
    return importlib.import_module("whisper_cbpf.models.constraints")


def _redback_constraint_or_skip():
    pytest.importorskip("redback")
    import redback.priors as rp
    if not hasattr(rp, "_constraint_settings"):
        pytest.skip("the installed redback predates _constraint_settings (1.20)")
    return rp


def _draws(model, n=N_DRAWS, seed=0):
    """``n`` draws of redback's own (unconstrained) prior for ``model``, seeded, as arrays."""
    from redback.priors import get_priors

    rng = np.random.default_rng(seed)
    pri = get_priors(model=model)
    return {k: np.asarray(p.rescale(rng.uniform(size=n)), dtype=float) * np.ones(n)
            for k, p in pri.items() if type(p).__name__ != "Constraint"}


def _redback_decisions(model, s):
    """redback's own accept (1) / reject (0) for each draw."""
    from redback.priors import get_priors

    return np.asarray(get_priors(model=model, constraint=True).evaluate_constraints(dict(s)),
                      dtype=bool)


# --- CLAIM 1: the transcription is redback's, decision for decision ------------------------------

@pytest.mark.parametrize("model", TRANSCRIBED)
def test_transcription_decides_exactly_as_redback_on_20000_draws(model):
    C = _C()
    _redback_constraint_or_skip()
    s = _draws(model)
    rb = _redback_decisions(model, s)
    # The test must be able to fail: redback accepts some draws and rejects others.
    assert 0 < rb.sum() < N_DRAWS, (model, rb.mean())
    mine = np.asarray(C.constraint_ok(model, s, mode="redback"), dtype=bool)
    assert np.array_equal(mine, rb), f"{model}: {int((mine != rb).sum())} of {N_DRAWS} differ"


@pytest.mark.parametrize("model", TRANSCRIBED)
def test_jax_transcription_decides_exactly_as_redback_on_20000_draws(model):
    """The JAX samplers evaluate the same formulas with ``xp=jax.numpy``, in float64."""
    C = _C()
    _redback_constraint_or_skip()
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        s = _draws(model)
        rb = _redback_decisions(model, s)
        f = jax.jit(lambda p: C.constraint_ok(model, p, mode="redback", xp=jnp))
        mine = np.asarray(f({k: jnp.asarray(v) for k, v in s.items()}), dtype=bool)
    finally:
        jax.config.update("jax_enable_x64", was)
    assert np.array_equal(mine, rb), f"{model}: {int((mine != rb).sum())} of {N_DRAWS} differ"


def test_every_transcribed_model_is_one_redback_constrains():
    """No invented constraint: each transcribed model has a redback entry with the same bounds."""
    C = _C()
    assert sorted(C.MODELS) == TRANSCRIBED
    rp = _redback_constraint_or_skip()
    for model, (_, bounds) in C.MODELS.items():
        conv, rb_bounds = rp._constraint_settings[model]
        assert {k: tuple(v) for k, v in rb_bounds.items()} == bounds, model


# --- CLAIM 2: the corrected bound ----------------------------------------------------------------

def test_corrected_burning_energy_is_14_helium_to_nickel_56():
    """Q = 14 Delta(He4) - Delta(Ni56) = 87.85 MeV per 56 u = 1.51e18 erg/g; redback's is 12.6x."""
    C = _C()
    q_mev = 14 * 2.4249 + 53.9037
    assert q_mev == pytest.approx(87.8523, abs=1e-4)
    assert C.E_BURN_PER_G == pytest.approx(1.5137e18, rel=1e-4)
    assert C.E_BURN_PER_G_REDBACK == pytest.approx(1.9114e19, rel=1e-4)
    assert C.E_BURN_PER_G_REDBACK / C.E_BURN_PER_G == pytest.approx(12.63, rel=1e-3)


def test_corrected_mode_is_the_independent_formula_and_changes_only_arnett():
    C = _C()
    _redback_constraint_or_skip()
    s = _draws("arnett")
    msun, km = 1.988409870698051e33, 1e5
    ke = 0.5 * s["mej"] * msun * (s["vej"] * km / 2.0) ** 2
    expect = ke / (C.E_BURN_PER_G * s["mej"] * msun * s["f_nickel"]) < 1.0
    got = np.asarray(C.constraint_ok("arnett", s), dtype=bool)          # default mode
    assert np.array_equal(got, expect)
    rb = _redback_decisions("arnett", s)
    assert not np.any(got & ~rb), "the corrected bound must be the stricter one"
    assert got.sum() < rb.sum()
    for model in set(C.MODELS) - {"arnett"}:
        s = _draws(model, n=2000)
        assert np.array_equal(np.asarray(C.constraint_ok(model, s), dtype=bool),
                              np.asarray(C.constraint_ok(model, s, mode="redback"), dtype=bool))


def test_modes_are_validated():
    C = _C()
    with pytest.raises(ValueError, match="constraint must be one of"):
        C.constraint_ok("arnett", ARNETT_OK, mode="strict")
    assert C.constraint_ok("arnett", ARNETT_BAD, mode=None) is True
    assert C.constraint_ok("type_1a", ARNETT_BAD) is True           # redback declares none


# --- CLAIM 3a: the CPU adapter --------------------------------------------------------------------

def test_cpu_adapter_predicts_zero_where_redback_rejects():
    """The bug: whisper <= 0.1.0 returned a light curve for a draw redback's prior rejects."""
    pytest.importorskip("redback")
    from whisper_cbpf.models import redback_adapter as RA

    t = np.array([5.0, 10.0, 20.0])
    b = np.array(["ztfg", "ztfr", "ztfi"])
    m = RA.redback_model("arnett", ["ztfg", "ztfr", "ztfi"], redshift=Z)
    assert np.all(m.predict(ARNETT_OK, t, b) > 0)
    assert np.all(m.predict(ARNETT_BAD, t, b) == 0.0)
    assert np.all(m.predict(ARNETT_BETWEEN, t, b) == 0.0)            # default: corrected bound
    rbm = RA.redback_model("arnett", ["ztfg"], redshift=Z, constraint="redback")
    assert np.all(rbm.predict(ARNETT_BETWEEN, t, b) > 0)
    assert np.all(rbm.predict(ARNETT_BAD, t, b) == 0.0)
    off = RA.redback_model("arnett", ["ztfg"], redshift=Z, constraint=None)
    assert np.all(off.predict(ARNETT_BAD, t, b) > 0)
    assert "1.51e18" in m.description and "constraint" in m.description


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_adapter_decisions_are_redbacks_on_20000_draws(family):
    """Per reference family, through the adapter's own predicate (``constraint="redback"``), with the
    redshift pinned as the adapter pins it. Families redback does not constrain accept every draw
    in both."""
    _redback_constraint_or_skip()
    from redback.priors import get_priors
    from whisper_cbpf.models import redback_adapter as RA

    s = _draws(family)
    if "redshift" in s:
        s["redshift"] = np.full(N_DRAWS, Z)
    ok = RA._constraint_predicate(family, "redback")
    if ok is None:                          # nothing declared: redback accepts every draw too
        assert np.all(get_priors(model=family, constraint=True).evaluate_constraints(dict(s)))
        return
    rb = _redback_decisions(family, s)
    mine = np.array([ok({k: v[i] for k, v in s.items()}) for i in range(N_DRAWS)])
    assert np.array_equal(mine, rb), f"{family}: {int((mine != rb).sum())} differ"


def test_cpu_adapter_uses_redbacks_own_function_for_models_it_does_not_transcribe():
    C = _C()
    rp = _redback_constraint_or_skip()
    from whisper_cbpf.models import redback_adapter as RA

    assert "csm_interaction" in rp._constraint_settings and "csm_interaction" not in C.MODELS
    ok = RA._constraint_predicate("csm_interaction", "corrected")
    s = _draws("csm_interaction", n=500)
    rb = _redback_decisions("csm_interaction", s)
    mine = np.array([ok({k: v[i] for k, v in s.items()}) for i in range(500)])
    assert np.array_equal(mine, rb) and 0 < rb.sum() < 500


# --- CLAIM 3b: the JAX samplers -------------------------------------------------------------------

@pytest.fixture()
def _x64():
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", was)


def _arnett_jax(**kw):
    from whisper_cbpf.models.jax import _factories as F
    return F.supernova_model("arnett", ["ztfg", "ztfr", "ztfi"], Z, 6.9e26, **kw)


def _lc():
    import whisper_cbpf as wp

    t = np.linspace(3.0, 40.0, 9)
    b = np.array(["ztfg", "ztfr", "ztfi"] * 3)
    return wp.LightCurve(time=t, band=b, flux=np.full(9, 1e-5), flux_err=np.full(9, 1e-6))


def test_jax_log_density_is_minus_inf_where_redback_rejects(_x64):
    """The bug: the auto-built density was finite at a draw redback's prior rejects."""
    from whisper_cbpf.samplers.jax._adapters import make_log_prob_jax

    m = _arnett_jax()
    lp = make_log_prob_jax(_lc(), m)
    theta = lambda d: np.array([d[p] for p in m.parameters])        # noqa: E731
    assert np.isfinite(float(lp(theta(ARNETT_OK))))
    assert float(lp(theta(ARNETT_BAD))) == -np.inf
    assert float(lp(theta(ARNETT_BETWEEN))) == -np.inf
    lp_rb = make_log_prob_jax(_lc(), _arnett_jax(constraint="redback"))
    assert np.isfinite(float(lp_rb(theta(ARNETT_BETWEEN))))
    lp_off = make_log_prob_jax(_lc(), _arnett_jax(constraint=None))
    assert np.isfinite(float(lp_off(theta(ARNETT_BAD))))


def test_jax_batched_forward_map_is_zero_where_redback_rejects(_x64):
    from whisper_cbpf.samplers.jax._adapters import make_batched_predict_jax

    m = _arnett_jax()
    f = make_batched_predict_jax(_lc(), m)
    rows = np.array([[d[p] for p in m.parameters] for d in (ARNETT_OK, ARNETT_BAD)])
    flux = np.asarray(f(rows))
    assert np.all(flux[0] > 0) and np.all(flux[1] == 0.0)


def test_jax_model_host_predict_is_zero_where_redback_rejects(_x64):
    """The CPU samplers (``abc``, ``mcmc``, ``nested``, ...) call ``Model.predict``, not the JAX
    adapters, so on a JAX model they sampled the unconstrained prior: ``predict`` computed the
    light curve at every draw. It now gives zero flux where ``constraint_ok`` fails, as the CPU
    redback adapter does, and ``predict_jax`` stays the physics."""
    m = _arnett_jax()
    t, b = np.linspace(3.0, 40.0, 9), np.array(["ztfg", "ztfr", "ztfi"] * 3)
    assert np.all(m.predict(ARNETT_OK, t, b) > 0)
    assert np.all(m.predict(ARNETT_BAD, t, b) == 0.0)
    assert np.all(m.predict(ARNETT_BETWEEN, t, b) == 0.0)            # default: corrected bound
    assert np.all(_arnett_jax(constraint="redback").predict(ARNETT_BETWEEN, t, b) > 0)
    off = _arnett_jax(constraint=None).predict(ARNETT_BAD, t, b)
    assert np.all(off > 0)
    theta = np.array([ARNETT_BAD[p] for p in m.parameters])
    physics = np.asarray(m.predict_jax(theta, t, m.predict_jax.band_index(b)))
    assert np.allclose(physics, off, rtol=1e-12, atol=0.0), "predict_jax must stay the physics"
    with pytest.raises(ValueError, match="needs a `bands` array"):
        m.predict(ARNETT_BAD, t)                                     # bands still checked first


def test_jax_model_zeros_are_silent_and_the_documented_check_tells_them_apart(_x64):
    """The zeros carry no warning -- the samplers meet a rejected draw at most draws of redback's
    prior (60 % for arnett) -- so ``predict``'s docstring points a direct caller to
    ``predict_jax.constraint_ok``, which must say which draws those are."""
    import warnings

    m = _arnett_jax()
    t, b = np.linspace(3.0, 40.0, 9), np.array(["ztfg", "ztfr", "ztfi"] * 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(m.predict(ARNETT_BAD, t, b) == 0.0)
    ok = m.predict_jax.constraint_ok
    assert bool(ok(np.array([ARNETT_OK[p] for p in m.parameters])))
    assert not bool(ok(np.array([ARNETT_BAD[p] for p in m.parameters])))


@pytest.mark.parametrize("family,build", [
    ("arnett", ("supernova", "arnett")),
    ("basic_magnetar_powered", ("supernova", "basic_magnetar_powered")),
    ("slsn", ("supernova", "slsn")),
    ("general_magnetar_slsn", ("supernova", "general_magnetar_slsn")),
    ("cooling_envelope", ("tde", "none")),
    ("gaussianrise_cooling_envelope", ("tde", "gaussian")),
])
def test_jax_model_decisions_are_redbacks_on_20000_draws(_x64, family, build):
    """Through the JAX model's own ``predict_jax.constraint_ok`` (what the adapters apply), vmapped
    over 20 000 draws of redback's prior at the model's fixed redshift."""
    _redback_constraint_or_skip()
    jax = _x64
    import jax.numpy as jnp
    from whisper_cbpf.models.jax import _factories as F

    kind, arg = build
    if kind == "supernova":
        m = F.supernova_model(arg, ["ztfg"], Z, 6.9e26, constraint="redback")
    else:
        m = F.tde_model(["ztfg"], Z, 6.9e26, rise=arg, constraint="redback")
    s = _draws(family)
    s["redshift"] = np.full(N_DRAWS, Z)
    rb = _redback_decisions(family, s)
    theta = jnp.asarray(np.stack([s[p] for p in m.parameters], axis=1))
    mine = np.asarray(jax.jit(jax.vmap(m.predict_jax.constraint_ok))(theta), dtype=bool)
    assert 0 < rb.sum() < N_DRAWS
    assert np.array_equal(mine, rb), f"{family}: {int((mine != rb).sum())} differ"


def test_unconstrained_jax_models_carry_no_predicate(_x64):
    from whisper_cbpf.models.jax import _factories as F

    assert getattr(_arnett_jax(constraint=None).predict_jax, "constraint_ok", None) is None
    assert getattr(F.supernova_model("type_1c", ["ztfg"], Z, 6.9e26).predict_jax,
                   "constraint_ok", None) is None
    with pytest.raises(ValueError, match="constraint must be one of"):
        _arnett_jax(constraint="strict")
