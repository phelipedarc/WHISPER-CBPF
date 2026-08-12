"""Prior log-density — JAX backend.

The numpy half (``_numpy.Prior.log_prob``) takes a parameter **dict** and returns a Python float.
A gradient sampler needs the same number as a traceable function of a **flat vector**, so that is
what :func:`log_prob_jax` builds. The density is the same one, term for term, including the
normalising constants — this is ``Prior.log_prob``'s value, not a box indicator.

Never imported at package import time: ``whisper_cbpf.priors.__init__`` resolves it lazily, so a
CPU-only install never sees JAX.
"""
from __future__ import annotations

import math

__all__ = ["log_prob_jax"]


def log_prob_jax(prior, names=None):
    """Build ``f(theta) -> scalar``, the log prior density at a flat parameter vector.

    ``theta`` is ordered by ``names`` (default ``prior.names``). Pass ``names`` explicitly whenever
    the order matters to a caller — a sampler's ``model.parameters`` order and a prior's insertion
    order are two different things, and silently trusting them to coincide is how a fit ends up
    evaluating the ``kappa`` prior at ``vej``.

    Outside the support the value is ``-inf``, matching ``Prior.log_prob``. The gradient there is
    zero rather than NaN: each branch's input is clamped into the support *before* the log, because
    ``jnp.where`` evaluates both branches on the raw input and ``0 * inf = nan`` would otherwise
    survive the mask into the backward pass (the same double-where hazard ``models/jax/flare.py``
    documents).

    Supported distributions are ``Uniform`` and ``LogUniform`` — the two the CPU classes define.
    Anything else raises rather than being approximated by a box, which would be a different
    posterior and not a different parameterisation (see ``samplers/jax/nuts_gpu._numpyro_priors``
    for the same refusal and the measurement behind it).

    NOTE for ``nuts_gpu``: do **not** add this to its ``log_prob_fn``. That sampler declares the
    prior to NumPyro as per-parameter sample sites and expects a pure log-*likelihood*; adding this
    on top double-counts the prior.
    """
    # Validate BEFORE importing jax, so a prior this cannot express is reported as that rather
    # than as a missing dependency on a machine that happens not to have JAX.
    names = list(prior.names) if names is None else list(names)
    missing = [n for n in names if n not in prior.distributions]
    if missing:
        raise KeyError(f"prior has no distribution for {missing}; it has {list(prior.names)}")

    spec = []
    for i, nm in enumerate(names):
        d = prior.distributions[nm]
        kind = type(d).__name__
        if kind not in ("Uniform", "LogUniform"):
            raise TypeError(
                f"log_prob_jax cannot express prior {kind!r} on parameter {nm!r}. Supported: "
                f"Uniform, LogUniform. Add the distribution here rather than falling back to a "
                f"uniform box -- that changes the posterior silently.")
        spec.append((kind, i) + tuple(float(x) for x in d.bounds))

    import jax.numpy as jnp

    terms = [(_uniform_term if kind == "Uniform" else _loguniform_term)(jnp, i, lo, hi)
             for kind, i, lo, hi in spec]

    def log_prob(theta):
        theta = jnp.asarray(theta)
        return sum(term(theta) for term in terms)

    return log_prob


def _uniform_term(jnp, i, lo, hi):
    """-log(hi - lo) inside [lo, hi], -inf outside. No theta dependence, so zero gradient."""
    # a Python float, not a jnp scalar: it then promotes to whatever dtype theta has, instead of
    # freezing the precision that happened to be configured when the prior function was built.
    density = -math.log(hi - lo)

    def term(theta):
        x = theta[i]
        return jnp.where((x >= lo) & (x <= hi), density, -jnp.inf)

    return term


def _loguniform_term(jnp, i, lo, hi):
    """-log(x) - log(log(hi) - log(lo)) inside [lo, hi], -inf outside."""
    log_range = math.log(math.log(hi) - math.log(lo))

    def term(theta):
        x = theta[i]
        inside = (x >= lo) & (x <= hi)
        # clamp BEFORE the log: lo > 0 is guaranteed by LogUniform's constructor, so the
        # unselected branch can never see a non-positive argument and d/dx = -1/x stays finite.
        safe = jnp.where(inside, x, lo)
        return jnp.where(inside, -jnp.log(safe) - log_range, -jnp.inf)

    return term
