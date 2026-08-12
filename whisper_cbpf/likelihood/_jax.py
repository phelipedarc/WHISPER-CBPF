"""Gaussian log-likelihood — JAX backend.

:func:`log_likelihood_jax` does not re-derive the likelihood; it is handed an already-built
likelihood **object** and reads ``y``, ``sigma``, ``space`` and the normalising constant off it.
That is deliberate. The two arms then cannot disagree about which points are in the fit, which
space they are compared in, or how the errors were resolved from the light curve — the one place
a CPU/GPU pair of likelihoods actually drifts.

Never imported at package import time: ``whisper_cbpf.likelihood.__init__`` resolves it lazily, so
a CPU-only install never sees JAX.
"""
from __future__ import annotations

__all__ = ["log_likelihood_jax"]


def log_likelihood_jax(like):
    """Build ``f(model_flux) -> scalar``, the JAX twin of ``like.log_likelihood(model_flux)``.

    ``like`` must be a :class:`~whisper_cbpf.likelihood.GaussianLikelihood` **exactly** — the
    subclasses are refused by name rather than silently evaluated as a plain Gaussian, which is
    what an ``isinstance`` check would do. ``GaussianLikelihoodWithUpperLimits`` needs the
    survival term, ``MixtureGaussianLikelihood`` the log-sum-exp, and
    ``GaussianLikelihoodWithScatter`` a second argument this signature does not carry; each is a
    different density, not a different implementation of this one.

    Returned function is *not* jitted. Jit the assembled log-posterior instead: jitting this piece
    alone would put a compilation boundary in the middle of the model-plus-likelihood graph, and
    the piece is a handful of vector ops that XLA fuses into the model's own kernels.

    Gradient note for magnitude space. The map is ``mag = -2.5 log10(flux / zeropoint)``, whose
    derivative ``-POGSON / flux`` diverges as the model flux goes to zero. The numpy half clamps at
    ``_MIN_FLUX_JY = 1e-300``, which is right in float64 and **underflows to exactly 0.0 in
    float32** — the same trap the SNPE torch path hit (see CHANGELOG, "The float32 floor is
    finfo.tiny"). The floor here is therefore the working dtype's smallest normal whenever
    ``_MIN_FLUX_JY`` is not representable in it. For a model with a magnitude cap (the JAX
    kilonova's ``mag_floor = 40`` gives flux >= 3.6e-16 Jy) the clamp never fires at all.
    """
    import jax.numpy as jnp

    from ._numpy import GaussianLikelihood, _MIN_FLUX_JY

    if type(like) is not GaussianLikelihood:
        raise NotImplementedError(
            f"log_likelihood_jax supports GaussianLikelihood; got {type(like).__name__}, which is "
            f"a different density (upper-limit survival term / mixture / free scatter), not a "
            f"different implementation of this one. Add it here explicitly when it is needed.")

    y = jnp.asarray(like.y)
    sigma = jnp.asarray(like.sigma)
    # read, not recomputed: -0.5 * sum(log(2 pi sigma^2)) is a constant of the DATA, and taking the
    # numpy object's own value is what makes the two log-likelihoods comparable to the last digit.
    log_norm = float(like._log_norm)
    magnitude = like.space == "magnitude"
    zeropoint = float(like.zeropoint_jy)

    def model_in_space(model_flux):
        mf = jnp.asarray(model_flux)
        if not magnitude:
            return mf
        tiny = jnp.finfo(mf.dtype).tiny
        floor = _MIN_FLUX_JY if _MIN_FLUX_JY > tiny else tiny
        return -2.5 * jnp.log10(jnp.clip(mf, floor, None) / zeropoint)

    def log_likelihood(model_flux):
        res = (y - model_in_space(model_flux)) / sigma
        return -0.5 * jnp.sum(res * res) + log_norm

    return log_likelihood
