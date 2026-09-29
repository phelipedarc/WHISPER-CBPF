"""Gaussian log-likelihoods — JAX backend.

:func:`log_likelihood_jax` does not re-derive the likelihood; it is handed an already-built
likelihood **object** and reads ``y``, ``sigma``, ``space``, the detection mask and the normalising
constants off it. That is deliberate. The two arms then cannot disagree about which points are in
the fit, which space they are compared in, or how the errors were resolved from the light curve —
the one place a CPU/GPU pair of likelihoods actually drifts.

Never imported at package import time: ``whisper_cbpf.likelihood.__init__`` resolves it lazily, so
a CPU-only install never sees JAX.

Parity with ``_numpy.py``
-------------------------
=================================== ============== ==========================================
class                               JAX            note
=================================== ============== ==========================================
``GaussianLikelihood``              flux, mag      ``f(model_flux)``
``GaussianLikelihoodWithUpperLimits`` flux         ``f(model_flux)``; magnitude space does not
                                                   exist on either backend (see ``_numpy.py``)
``GaussianLikelihoodWithScatter``   flux, mag      ``f(model_flux, sigma_extra=0.0)``
``MixtureGaussianLikelihood``       --             deliberate non-goal, refused by name
=================================== ============== ==========================================
"""
from __future__ import annotations

__all__ = ["log_likelihood_jax"]


def log_likelihood_jax(like):
    """Build the JAX twin of ``like.log_likelihood``.

    ``like`` must be one of the three supported classes **exactly** — dispatch is on ``type(like)``
    rather than ``isinstance``, so a user's own subclass is refused by name instead of being
    silently evaluated as whichever base it happens to inherit. Each of these is a different
    density, not a different implementation of one.

    Returns
    -------
    callable
        ``f(model_flux) -> scalar`` for :class:`~whisper_cbpf.likelihood.GaussianLikelihood` and
        :class:`~whisper_cbpf.likelihood.GaussianLikelihoodWithUpperLimits`;
        ``f(model_flux, sigma_extra=0.0) -> scalar`` for
        :class:`~whisper_cbpf.likelihood.GaussianLikelihoodWithScatter`, matching that class's own
        arity. The returned function carries ``.scatter_param`` — the name of the prior parameter
        that supplies ``sigma_extra``, or ``None`` — which is how
        ``samplers.jax._adapters.make_log_prob_jax`` knows whether a sampled column that is not a
        model parameter is consumed here or is a flat direction to refuse.

    Returned function is *not* jitted. Jit the assembled log-posterior instead: jitting this piece
    alone would put a compilation boundary in the middle of the model-plus-likelihood graph, and
    the piece is a handful of vector ops that XLA fuses into the model's own kernels.

    Gradient note for magnitude space. The map is ``mag = -2.5 log10(flux / zeropoint)``, whose
    derivative ``-POGSON / flux`` diverges as the model flux goes to zero. The numpy half clamps at
    ``_MIN_FLUX_JY = 1e-300``, which is right in float64 and **underflows to exactly 0.0 in
    float32** — the same trap the SNPE torch path hit (see CHANGELOG, "The float32 floor is
    finfo.tiny"). The floor here is therefore the working dtype's smallest normal whenever
    ``_MIN_FLUX_JY`` is not representable in it. For a model with a magnitude cap (the JAX
    kilonova's ``mag_floor = 40`` gives flux >= 3.631e-13 Jy) the clamp never fires at all.
    """
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.special import ndtr

    from ._numpy import (
        _LN2PI,
        _MIN_FLUX_JY,
        _MIN_PROB,
        GaussianLikelihood,
        GaussianLikelihoodWithScatter,
        GaussianLikelihoodWithUpperLimits,
        MixtureGaussianLikelihood,
    )

    kind = type(like)
    if kind is MixtureGaussianLikelihood:
        raise NotImplementedError(
            "log_likelihood_jax does not implement MixtureGaussianLikelihood, and that is a "
            "decision rather than an omission: its density is a two-component log-sum-exp whose "
            "mixture weight and outlier scale are FIXED, so on the gradient path it buys a "
            "heavier-tailed likelihood at the cost of a posterior that depends on two numbers "
            "nobody fitted. Use a numpy sampler (mcmc/nested) for it, or pass log_prob_fn= with "
            "your own JAX mixture.")
    if kind not in (GaussianLikelihood, GaussianLikelihoodWithUpperLimits,
                    GaussianLikelihoodWithScatter):
        raise NotImplementedError(
            f"log_likelihood_jax supports GaussianLikelihood, GaussianLikelihoodWithUpperLimits "
            f"and GaussianLikelihoodWithScatter; got {kind.__name__}. Dispatch is on the exact "
            f"type, not isinstance, so a subclass is refused here rather than silently scored "
            f"under its base class's density. Add it explicitly when it is needed.")

    y = jnp.asarray(like.y)
    sigma = jnp.asarray(like.sigma)
    magnitude = like.space == "magnitude"
    zeropoint = float(like.zeropoint_jy)

    def model_in_space(model_flux):
        mf = jnp.asarray(model_flux)
        if not magnitude:
            return mf
        tiny = jnp.finfo(mf.dtype).tiny
        floor = _MIN_FLUX_JY if _MIN_FLUX_JY > tiny else tiny
        return -2.5 * jnp.log10(jnp.clip(mf, floor, None) / zeropoint)

    if kind is GaussianLikelihoodWithUpperLimits:
        if magnitude:                       # unreachable through __init__, which refuses it there
            raise NotImplementedError(
                "GaussianLikelihoodWithUpperLimits is flux-only on both backends; this object "
                "reports space='magnitude'. Rebuild it with space='flux'.")
        det = np.asarray(like.detections, dtype=bool)
        # Static gather indices, not boolean masks: the mask is a property of the DATA and is known
        # at trace time, so the two terms become two fixed-shape gathers with no branching inside.
        i_det, i_ul = jnp.asarray(np.flatnonzero(det)), jnp.asarray(np.flatnonzero(~det))
        y_det, sigma_det = y[i_det], sigma[i_det]
        # read, not recomputed: the detection-only normaliser is a constant of the DATA, and taking
        # the numpy object's own value is what makes the two log-likelihoods comparable to the last
        # digit. It is computed over detections ONLY so the NaN sigmas of the censored rows -- what
        # whisper's loader stores for a non-detection -- cannot poison it.
        log_norm_det = float(like._log_norm_det)
        limit = y[i_ul]
        sigma_ul = limit / float(like.upper_limit_sigma)
        n_det, n_ul = int(det.sum()), int((~det).sum())

        def log_likelihood(model_flux):
            m = model_in_space(model_flux)
            out = jnp.asarray(0.0, dtype=m.dtype)
            if n_det:
                res = (y_det - m[i_det]) / sigma_det
                out = out - 0.5 * jnp.sum(res * res) + log_norm_det
            if n_ul:
                # P(true flux < limit), through the SAME `ndtr` the numpy half uses -- not
                # `0.5 * (1 + erf(z / sqrt 2))`, whose cancellation in the left tail is worth 0.14
                # in log-likelihood at z = -4.9 in float32 (see GaussianLikelihoodWithUpperLimits.
                # _cdf for the measured table). The clip is the numpy half's too, and it is what
                # keeps log(prob) finite for a model far above the limit.
                prob = ndtr((limit - m[i_ul]) / sigma_ul)
                out = out + jnp.sum(jnp.log(jnp.clip(prob, _MIN_PROB, 1.0 - _MIN_PROB)))
            return out

        log_likelihood.scatter_param = None
        return log_likelihood

    if kind is GaussianLikelihoodWithScatter:
        var_data = sigma * sigma

        def log_likelihood(model_flux, sigma_extra=0.0):
            # sigma_extra is TRACED (it is a sampled parameter); the numpy twin's float() cast
            # would concretise it and break the gradient this whole class exists to provide.
            var = var_data + jnp.asarray(sigma_extra) ** 2
            res2 = (y - model_in_space(model_flux)) ** 2 / var
            return -0.5 * jnp.sum(res2 + _LN2PI + jnp.log(var))

        log_likelihood.scatter_param = str(like.scatter_param)
        return log_likelihood

    # read, not recomputed: -0.5 * sum(log(2 pi sigma^2)) is a constant of the DATA, and taking the
    # numpy object's own value is what makes the two log-likelihoods comparable to the last digit.
    log_norm = float(like._log_norm)

    def log_likelihood(model_flux):
        res = (y - model_in_space(model_flux)) / sigma
        return -0.5 * jnp.sum(res * res) + log_norm

    log_likelihood.scatter_param = None
    return log_likelihood
