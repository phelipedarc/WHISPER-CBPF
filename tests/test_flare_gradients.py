"""The flare model's NaN-safety property, which is easy to lose in a refactor.

``jnp.where(cond, f(x), g(x))`` evaluates BOTH branches on the raw ``x`` and masks afterwards, so
if the unselected branch overflows the resulting inf survives into the backward pass as
``0 * inf = nan`` -- contaminating the gradient while the forward value looks perfectly fine. The
model therefore sanitises each branch's input BEFORE evaluating it (the "double-where" pattern).
These tests pin that behaviour down, including a control showing the naive form really does fail.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from whisper_cbpf.models.jax import _flare_spec as spec  # noqa: E402
from whisper_cbpf.models.jax import flare  # noqa: E402


def test_gradients_finite_across_the_prior_including_the_dt0_boundary():
    rng = np.random.default_rng(0)
    # t == t0 exactly is the branch boundary; the far points exercise the overflow-prone side
    times = jnp.asarray([-500.0, -50.0, -5.0, 0.0, 10.0, 10.0 + 1e-6, 25.0, 200.0])
    for _ in range(60):
        p = {nm: float(rng.uniform(lo, hi))
             for nm, (lo, hi) in zip(spec.PARAMETERS, spec.PRIOR_BOUNDS)}
        g = jax.grad(lambda q: jnp.sum(flare.predict_jax(q, times)))(
            {k: jnp.asarray(v) for k, v in p.items()})
        for k, v in g.items():
            assert np.isfinite(float(v)), f"non-finite d/d{k} at {p}"


def test_naive_single_where_would_fail_here():
    """Control: the property under test is real, not vacuous.

    THE OFFSET HAS TO OVERFLOW IN float64 TOO, NOT JUST float32.
        ``jax_enable_x64`` is a global, irreversible flag, and ``tests/t2_autodiff/`` turns it on.
        Pytest collects ``t2_autodiff`` before this file (``t2`` sorts before ``test``), so by the
        time this control runs the session may be in float64 -- whereas run alone it is float32.

        ``exp`` overflows once its argument passes ``log(finfo.max)``: about 88.7 in float32 but
        709.8 in float64. The original ``t = -500`` gives ``dt/tau = 510/e = 187.6``, which
        overflows float32 and is a perfectly finite 1e81 in float64 -- so this control PASSED
        alone and FAILED in the full suite, reporting that the naive form "no longer fails" when
        all that had changed was the precision it inherited.

        ``t = -2500`` gives ``dt/tau = 2510/e = 923``, past both limits, so the control now means
        the same thing whatever an earlier test did to the global flag.
    """
    def naive(la, ls, lt, t0, t):
        amp, s, tau = jnp.exp(la), jnp.exp(ls), jnp.exp(lt)
        dt = t - t0
        return jnp.where(dt < 0, amp * jnp.exp(-(dt ** 2) / (2 * s ** 2)),
                          amp * jnp.exp(-dt / tau))       # evaluated on RAW dt -> overflows

    times = jnp.asarray([-2500.0])
    p = dict(log_amp=jnp.asarray(1.0), log_sigma=jnp.asarray(0.5),
             log_tau=jnp.asarray(1.0), t0=jnp.asarray(10.0))
    g = jax.grad(lambda q: jnp.sum(jax.vmap(naive, in_axes=(None,) * 4 + (0,))(
        q["log_amp"], q["log_sigma"], q["log_tau"], q["t0"], times)))(p)
    assert any(not np.isfinite(float(v)) for v in g.values()), \
        "the naive form no longer fails -- this control has stopped testing anything"


def test_jax_matches_the_numpy_reference():
    rng = np.random.default_rng(11)
    times = np.sort(rng.uniform(-10.0, 70.0, 200))
    for _ in range(20):
        p = {nm: float(rng.uniform(lo, hi))
             for nm, (lo, hi) in zip(spec.PARAMETERS, spec.PRIOR_BOUNDS)}
        a = np.asarray(flare.predict_jax({k: jnp.asarray(v) for k, v in p.items()},
                                          jnp.asarray(times, dtype=jnp.float32)))
        b = spec.flare_flux_numpy(p, times)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=np.exp(p["log_amp"]) * 1e-6)


def test_f32_and_f64_gradients_agree_to_better_than_one_percent():
    """Whether x64 is on is a process-wide setting, so compare against a numpy central difference
    rather than trying to flip it mid-test."""
    rng = np.random.default_rng(3)
    times = np.sort(rng.uniform(0.0, 30.0, 120))
    for _ in range(15):
        p = np.array([rng.uniform(lo, hi) for lo, hi in spec.PRIOR_BOUNDS])
        keys = spec.PARAMETERS

        def total(vec):
            return float(np.sum(spec.flare_flux_numpy(dict(zip(keys, vec)), times)))

        ad = jax.grad(lambda v: jnp.sum(flare.predict_jax(
            dict(zip(keys, v)), jnp.asarray(times, dtype=jnp.float32))))(
            jnp.asarray(p, dtype=jnp.float32))
        ad = np.asarray(ad, dtype=np.float64)
        for j in range(len(p)):
            h = 1e-5 * max(abs(p[j]), 1.0)
            hi, lo = p.copy(), p.copy()
            hi[j] += h; lo[j] -= h
            fd = (total(hi) - total(lo)) / (2 * h)
            if abs(fd) > 1e-6:
                assert abs(ad[j] - fd) / abs(fd) < 1e-2, (keys[j], ad[j], fd)
