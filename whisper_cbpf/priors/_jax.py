"""Prior log-density and inverse CDF — JAX backend.

The numpy half (``_numpy.Prior.log_prob``) takes a parameter **dict** and returns a Python float.
A gradient sampler needs the same number as a traceable function of a **flat vector**, so that is
what :func:`log_prob_jax` builds. The density is the same one, term for term, including the
normalising constants — this is ``Prior.log_prob``'s value, not a box indicator.
:func:`ppf_jax` is a distribution's inverse CDF as a traceable function, the probability integral
transform the NUTS samplers sample a TruncatedNormal by.

Never imported at package import time: ``whisper_cbpf.priors.__init__`` resolves it lazily, so a
CPU-only install never sees JAX.
"""
from __future__ import annotations

import math

from ._numpy import family

__all__ = ["log_prob_jax", "ppf_jax"]

_SUPPORTED = ("Uniform", "LogUniform", "Normal", "TruncatedNormal", "Fixed")
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


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

    Supported distributions: ``Uniform``, ``LogUniform``, ``Normal``, ``TruncatedNormal`` (its
    truncation normalisation included, a constant computed once in float64) and ``Fixed`` (0 at its
    value, ``-inf`` elsewhere; the samplers leave a Fixed parameter out of the sampled vector, so
    this term is only met by a caller who keeps it in). Anything else raises rather than being
    approximated by a box, which would be a different posterior and not a different
    parameterisation (see ``samplers/jax/nuts_gpu._numpyro_priors`` for the same refusal and the
    measurement behind it).

    NOTE for ``nuts_gpu``: do **not** add this to its ``log_prob_fn``. That sampler declares the
    prior to NumPyro as per-parameter sample sites and expects a pure log-*likelihood*; adding this
    on top double-counts the prior.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.priors import LogUniform, Normal, Prior, log_prob_jax
    >>> prior = Prior({"z": Normal(0.05, 0.01), "mej": LogUniform(1e-3, 1.0)})
    >>> f = log_prob_jax(prior)
    >>> x = {"z": 0.052, "mej": 0.1}
    >>> abs(float(f(jnp.array([0.052, 0.1]))) - prior.log_prob(x)) < 1e-5
    True
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
        kind = family(d)
        if kind not in _SUPPORTED:
            raise TypeError(
                f"log_prob_jax cannot express prior {kind!r} on parameter {nm!r}. Supported: "
                f"{', '.join(_SUPPORTED)}. Add the distribution here rather than falling back to "
                f"a uniform box -- that changes the posterior silently.")
        spec.append((kind, i, d))

    import jax.numpy as jnp

    build = {"Uniform": _uniform_term, "LogUniform": _loguniform_term, "Normal": _normal_term,
             "TruncatedNormal": _truncated_normal_term, "Fixed": _fixed_term}
    terms = [build[kind](jnp, i, d) for kind, i, d in spec]

    def log_prob(theta):
        theta = jnp.asarray(theta)
        return sum(term(theta) for term in terms)

    return log_prob


def _uniform_term(jnp, i, d):
    """-log(hi - lo) inside [lo, hi], -inf outside. No theta dependence, so zero gradient."""
    lo, hi = (float(x) for x in d.bounds)
    # a Python float, not a jnp scalar: it then promotes to whatever dtype theta has, instead of
    # freezing the precision that happened to be configured when the prior function was built.
    density = -math.log(hi - lo)

    def term(theta):
        x = theta[i]
        return jnp.where((x >= lo) & (x <= hi), density, -jnp.inf)

    return term


def _loguniform_term(jnp, i, d):
    """-log(x) - log(log(hi) - log(lo)) inside [lo, hi], -inf outside."""
    lo, hi = (float(x) for x in d.bounds)
    log_range = math.log(math.log(hi) - math.log(lo))

    def term(theta):
        x = theta[i]
        inside = (x >= lo) & (x <= hi)
        # clamp BEFORE the log: lo > 0 is guaranteed by LogUniform's constructor, so the
        # unselected branch can never see a non-positive argument and d/dx = -1/x stays finite.
        safe = jnp.where(inside, x, lo)
        return jnp.where(inside, -jnp.log(safe) - log_range, -jnp.inf)

    return term


def _normal_term(jnp, i, d):
    """-(x - mu)^2 / (2 sigma^2) - log(sigma sqrt(2 pi)); finite everywhere."""
    mu, sigma = float(d.mu), float(d.sigma)
    const = -math.log(sigma) - _LOG_SQRT_2PI

    def term(theta):
        z = (theta[i] - mu) / sigma
        return -0.5 * z * z + const

    return term


def _truncated_normal_term(jnp, i, d):
    """The Normal term minus log Z inside [low, high], -inf outside. ``log Z`` is the numpy
    class's, computed once in float64 from ``log_ndtr`` (exact in either tail)."""
    mu, sigma, lo, hi = float(d.mu), float(d.sigma), float(d.low), float(d.high)
    const = -math.log(sigma) - _LOG_SQRT_2PI - float(d._log_mass)

    def term(theta):
        x = theta[i]
        inside = (x >= lo) & (x <= hi)
        z = (jnp.clip(x, lo, hi) - mu) / sigma        # clamped: a finite cotangent outside
        return jnp.where(inside, -0.5 * z * z + const, -jnp.inf)

    return term


def _fixed_term(jnp, i, d):
    """0 at the value, -inf elsewhere (a point mass, relative to counting measure)."""
    v = float(d.value)

    def term(theta):
        return jnp.where(theta[i] == v, 0.0, -jnp.inf)

    return term


def ppf_jax(dist):
    """The inverse CDF of a ``Normal`` or ``TruncatedNormal`` as a traceable ``p -> x``.

    For a TruncatedNormal it is the probability integral transform
    ``x = mu + sigma * Phi^-1(Phi(alpha) + p Z)``, taken on the upper tail's side when the range
    lies above ``mu`` (as the numpy class does), clipped into ``[low, high]``: exact, so ``p`` drawn
    ``U(0, 1)`` gives exact draws, on any device. The constants are float64 numbers from the numpy
    class, which promote to the dtype of ``p``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.priors import TruncatedNormal, ppf_jax
    >>> d = TruncatedNormal(0.0, 1.0, 0.0, 2.0)
    >>> x = ppf_jax(d)(jnp.array([0.25, 0.5, 0.75]))
    >>> bool(jnp.allclose(x, d.ppf([0.25, 0.5, 0.75]), atol=1e-6))
    True
    """
    from jax.scipy.special import ndtri
    import jax.numpy as jnp

    kind = family(dist)
    if kind == "Normal":
        mu, sigma = float(dist.mu), float(dist.sigma)
        return lambda p: mu + sigma * ndtri(p)
    if kind != "TruncatedNormal":
        raise TypeError(f"ppf_jax takes a Normal or a TruncatedNormal; got {kind!r}.")
    mu, sigma, lo, hi = float(dist.mu), float(dist.sigma), float(dist.low), float(dist.high)
    base, mass = float(dist._lo_cdf), float(dist._mass)
    if dist._flip:
        return lambda p: jnp.clip(mu - sigma * ndtri(base + (1.0 - p) * mass), lo, hi)
    return lambda p: jnp.clip(mu + sigma * ndtri(base + p * mass), lo, hi)
