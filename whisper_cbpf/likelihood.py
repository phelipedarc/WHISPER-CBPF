"""Likelihoods for transient inference, in flux **or** apparent-magnitude space.

Whisper models predict **flux**; a likelihood compares that prediction to the data in a chosen
``space``:

* ``space='flux'`` — residuals/errors in flux (Jy). Upper limits (non-detections) can be used here.
* ``space='magnitude'`` — the model flux is converted to AB magnitude and compared to the observed
  magnitudes/errors.
* ``space='auto'`` (default) — magnitude data -> magnitude space, flux data -> flux space. This is the
  "correct" default; users override only for edge cases (e.g. outliers, non-detections).

Likelihoods are picklable so they can cross process boundaries for parallel inference. Each exposes
``log_likelihood(model_flux) -> float``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from .io.photometry import AB_ZEROPOINT_JY, POGSON, flux_density_to_mag

_LN2PI = np.log(2.0 * np.pi)
_MIN_FLUX_JY = 1e-300       # floor a (clipped) model flux before log10 in the flux->magnitude transform
_MIN_PROB = 1e-30          # floor a probability before log() (upper-limit / mixture terms)


def _resolve_space(lc, space):
    if space is None or str(space).lower() == "auto":
        return "magnitude" if lc.data_mode == "magnitude" else "flux"
    s = str(space).lower()
    if s in ("mag", "magnitude"):
        return "magnitude"
    if s in ("flux", "flux_density", "luminosity"):
        return "flux"
    raise ValueError(f"space must be 'flux', 'magnitude', or 'auto'; got {space!r}")


class GaussianLikelihood:
    """Independent-Gaussian likelihood in flux or magnitude space."""

    def __init__(self, lc, space="auto", zeropoint_jy=AB_ZEROPOINT_JY):
        self.space = _resolve_space(lc, space)
        self.zeropoint_jy = float(zeropoint_jy)
        full = lc.add_flux(self.zeropoint_jy).add_mag(self.zeropoint_jy)
        if self.space == "magnitude":
            self.y, self.sigma = full.magnitude, full.magnitude_err
        else:
            self.y, self.sigma = full.flux, full.flux_err
        if self.y is None or self.sigma is None:
            raise ValueError(f"GaussianLikelihood in {self.space} space requires y values and errors.")
        self.y = np.asarray(self.y, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        self._log_norm = -0.5 * np.sum(_LN2PI + 2.0 * np.log(self.sigma))

    def model_in_space(self, model_flux):
        mf = np.asarray(model_flux, dtype=float)
        if self.space == "magnitude":
            # A non-positive model flux has no magnitude; clip to a tiny floor so it maps to a very
            # faint magnitude (~+759) -> a large but FINITE chi-square penalty. This keeps the
            # log-likelihood finite (so it effectively rejects, rather than NaN-ing, such draws) and
            # avoids spurious non-finite values that would otherwise be dropped by WAIC.
            return flux_density_to_mag(np.clip(mf, _MIN_FLUX_JY, None), zeropoint_jy=self.zeropoint_jy)
        return mf

    def log_likelihood(self, model_flux):
        res = (self.y - self.model_in_space(model_flux)) / self.sigma
        return float(-0.5 * np.sum(res * res) + self._log_norm)

    def log_likelihood_pointwise(self, model_flux):
        """Per-data-point log-likelihood (length ``n_data``). Sums to :meth:`log_likelihood`; the
        pointwise terms are what WAIC needs (see :func:`whisper_labia.metrics.waic`)."""
        res = (self.y - self.model_in_space(model_flux)) / self.sigma
        return -0.5 * (res * res) - 0.5 * (_LN2PI + 2.0 * np.log(self.sigma))

    def summary(self):
        return {"likelihood": "gaussian", "space": self.space, "n_data": int(self.y.size)}


class GaussianLikelihoodWithUpperLimits(GaussianLikelihood):
    """Gaussian for detections + a CDF/survival term for upper limits (non-detections).

    Upper-limit ``y`` values are the limiting flux/magnitude. ``upper_limit_sigma`` is the N-sigma
    level of those limits (e.g. 3.0 for 3-sigma). Flux space treats a limit as "true value < limit";
    magnitude space as "true value > limit" (fainter than the limit).
    """

    def __init__(self, lc, space="auto", upper_limit_sigma=3.0, zeropoint_jy=AB_ZEROPOINT_JY):
        super().__init__(lc, space=space, zeropoint_jy=zeropoint_jy)
        ul = lc.upper_limit
        self.detections = np.ones(self.y.size, dtype=bool) if ul is None else ~np.asarray(ul, dtype=bool)
        self.upper_limit_sigma = float(upper_limit_sigma)
        if np.any(~self.detections) and np.any(~np.isfinite(self.y[~self.detections])):
            raise ValueError("Upper limits require finite limiting values (flux or magnitude).")
        det = self.detections
        self._log_norm_det = (-0.5 * np.sum(_LN2PI + 2.0 * np.log(self.sigma[det]))
                              if np.any(det) else 0.0)

    @staticmethod
    def _cdf(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    def log_likelihood(self, model_flux):
        m = self.model_in_space(model_flux)
        ll = 0.0
        det = self.detections
        if np.any(det):
            res = (self.y[det] - m[det]) / self.sigma[det]
            ll += -0.5 * np.sum(res * res) + self._log_norm_det
        ul = ~det
        if np.any(ul):
            limit, model_ul = self.y[ul], m[ul]
            if self.space == "magnitude":
                sigma_ul = POGSON / self.upper_limit_sigma
                prob = 1.0 - self._cdf((limit - model_ul) / sigma_ul)   # true mag > limit (fainter)
            else:
                sigma_ul = limit / self.upper_limit_sigma
                prob = self._cdf((limit - model_ul) / sigma_ul)         # true flux < limit
            ll += float(np.sum(np.log(np.clip(prob, _MIN_PROB, 1.0 - _MIN_PROB))))
        return float(ll)

    def log_likelihood_pointwise(self, model_flux):
        """Per-point log-likelihood: Gaussian at detections, log upper-limit probability elsewhere."""
        m = self.model_in_space(model_flux)
        out = np.empty(self.y.size, dtype=float)
        det = self.detections
        if np.any(det):
            res = (self.y[det] - m[det]) / self.sigma[det]
            out[det] = -0.5 * (res * res) - 0.5 * (_LN2PI + 2.0 * np.log(self.sigma[det]))
        ul = ~det
        if np.any(ul):
            limit, model_ul = self.y[ul], m[ul]
            if self.space == "magnitude":
                prob = 1.0 - self._cdf((limit - model_ul) / (POGSON / self.upper_limit_sigma))
            else:
                prob = self._cdf((limit - model_ul) / (limit / self.upper_limit_sigma))
            out[ul] = np.log(np.clip(prob, _MIN_PROB, 1.0 - _MIN_PROB))
        return out

    def summary(self):
        return {"likelihood": "gaussian_upper_limits", "space": self.space,
                "n_data": int(self.y.size), "detections": int(np.sum(self.detections)),
                "upper_limits": int(np.sum(~self.detections)),
                "upper_limit_sigma": self.upper_limit_sigma}


class MixtureGaussianLikelihood(GaussianLikelihood):
    """Outlier-robust two-component Gaussian mixture (inlier ``sigma`` + wide ``sigma*scale``)."""

    def __init__(self, lc, space="auto", alpha=0.9, sigma_out_scale=10.0, zeropoint_jy=AB_ZEROPOINT_JY):
        super().__init__(lc, space=space, zeropoint_jy=zeropoint_jy)
        self.alpha = float(alpha)
        self.sigma_out = self.sigma * float(sigma_out_scale)

    def log_likelihood(self, model_flux):
        res = self.y - self.model_in_space(model_flux)
        logp_in = -0.5 * _LN2PI - np.log(self.sigma) - 0.5 * (res / self.sigma) ** 2
        logp_out = -0.5 * _LN2PI - np.log(self.sigma_out) - 0.5 * (res / self.sigma_out) ** 2
        return float(np.sum(np.logaddexp(np.log(self.alpha) + logp_in,
                                         np.log(1.0 - self.alpha) + logp_out)))

    def summary(self):
        return {"likelihood": "mixture_gaussian", "space": self.space,
                "n_data": int(self.y.size), "alpha": self.alpha}


class GaussianLikelihoodWithScatter(GaussianLikelihood):
    """Gaussian likelihood with a FREE additional scatter term added in quadrature (Villar+2017).

    .. math::

        \\ln\\mathcal{L} = -\\tfrac12 \\sum_i \\left[ \\frac{(O_i - M_i)^2}{\\sigma_i^2 + \\sigma^2}
                          + \\ln\\!\\big(2\\pi(\\sigma_i^2 + \\sigma^2)\\big) \\right]

    where :math:`\\sigma` is a fitted parameter absorbing extra model/data uncertainty beyond the
    reported per-point errors :math:`\\sigma_i` (Villar et al. 2017, ApJL 851 L21; as implemented in
    MOSFiT — the correctly normalized form of their Eq. 4). With :math:`\\sigma = 0` this reduces
    exactly to :class:`GaussianLikelihood`.

    ``scatter_param`` names the prior parameter that carries :math:`\\sigma` (default ``"sigma"``);
    samplers route that parameter here (``sigma_extra``) instead of into ``model.predict``, and the
    simulation-based samplers add it to their generative noise, so every method fits the same model.
    """

    def __init__(self, lc, space="auto", scatter_param="sigma", zeropoint_jy=AB_ZEROPOINT_JY):
        super().__init__(lc, space=space, zeropoint_jy=zeropoint_jy)
        self.scatter_param = str(scatter_param)

    def log_likelihood(self, model_flux, sigma_extra=0.0):
        var = self.sigma ** 2 + float(sigma_extra) ** 2
        res2 = (self.y - self.model_in_space(model_flux)) ** 2 / var
        return float(-0.5 * np.sum(res2 + _LN2PI + np.log(var)))

    def log_likelihood_pointwise(self, model_flux, sigma_extra=0.0):
        var = self.sigma ** 2 + float(sigma_extra) ** 2
        res2 = (self.y - self.model_in_space(model_flux)) ** 2 / var
        return -0.5 * (res2 + _LN2PI + np.log(var))

    def summary(self):
        return {"likelihood": "gaussian_scatter", "space": self.space,
                "n_data": int(self.y.size), "scatter_param": self.scatter_param}


_LIKELIHOODS = {
    "gaussian": GaussianLikelihood, "normal": GaussianLikelihood,
    "gaussian_scatter": GaussianLikelihoodWithScatter, "scatter": GaussianLikelihoodWithScatter,
    "villar": GaussianLikelihoodWithScatter,
    "gaussian_upper_limits": GaussianLikelihoodWithUpperLimits,
    "upper_limits": GaussianLikelihoodWithUpperLimits, "ul": GaussianLikelihoodWithUpperLimits,
    "mixture": MixtureGaussianLikelihood, "mixture_gaussian": MixtureGaussianLikelihood,
    "outlier": MixtureGaussianLikelihood,
}


def register_likelihood(name, likelihood_cls, *, overwrite=False):
    """Register a likelihood class under ``name`` so ``make_likelihood(kind=name)`` can build it.

    The class must accept ``(lc, space=..., **kwargs)`` and expose ``log_likelihood(model_flux) -> float``
    (subclass :class:`GaussianLikelihood` for the easy path). Mirrors ``register_model`` /
    ``register_sampler``.
    """
    key = str(name).lower()
    if key in _LIKELIHOODS and not overwrite:
        raise ValueError(f"Likelihood {name!r} already registered (pass overwrite=True).")
    _LIKELIHOODS[key] = likelihood_cls


def list_likelihoods():
    """Sorted list of registered likelihood ``kind`` names (incl. aliases)."""
    return sorted(_LIKELIHOODS)


def make_likelihood(lc, kind="auto", space="auto", **kwargs):
    """Build the data-appropriate likelihood (override with ``kind`` / ``space``).

    ``kind='auto'`` picks ``gaussian_upper_limits`` when the light curve has upper limits, else
    ``gaussian``. ``space='auto'`` picks magnitude space for magnitude data, flux space otherwise.
    Use :func:`list_likelihoods` to see the available ``kind`` names and :func:`register_likelihood`
    to add your own.
    """
    if kind == "auto":
        has_ul = lc.upper_limit is not None and bool(np.any(lc.upper_limit))
        kind = "gaussian_upper_limits" if has_ul else "gaussian"
    key = str(kind).lower()
    if key not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {kind!r}. Available: {list_likelihoods()}")
    return _LIKELIHOODS[key](lc, space=space, **kwargs)
