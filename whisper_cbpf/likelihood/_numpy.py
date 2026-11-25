"""Likelihoods for transient inference — numpy backend (the default).

The classes and the ``make_likelihood`` registry, function bodies unchanged from the single-module
``likelihood.py`` this replaced. What a likelihood *is*, and what ``space`` means, is documented once
in the package docstring (``whisper_cbpf/likelihood/__init__.py``); the JAX twin of
``GaussianLikelihood.log_likelihood`` is ``_jax.log_likelihood_jax``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ..io.photometry import AB_ZEROPOINT_JY, flux_density_to_mag

_LN2PI = np.log(2.0 * np.pi)
_MIN_FLUX_JY = 1e-300       # floor a (clipped) model flux before log10 in the flux->magnitude transform
_MIN_PROB = 1e-30          # floor a probability before log() (upper-limit / mixture terms)

#: Significance of an upper limit when neither the caller nor ``lc.meta["upper_limit_sigma"]``
#: states it: 5 sigma, the depth ZTF's ``diffmaglim`` and LSST's forced-photometry limits are quoted
#: at (``whisper_cbpf.io.surveys.LIMIT_SIGMA``).
DEFAULT_UPPER_LIMIT_SIGMA = 5.0


def _has_upper_limits(lc):
    ul = getattr(lc, "upper_limit", None)
    return ul is not None and bool(np.any(np.asarray(ul, dtype=bool)))


def resolve_space(lc, space):
    """Normalise a ``space=`` argument against the light curve's own data mode.

    ``"auto"`` (or ``None``) follows the data: magnitude data is fitted in magnitude space,
    everything else in flux, **except** that a light curve carrying upper limits is fitted in flux
    space, the only space the censored likelihood exists in
    (:class:`GaussianLikelihoodWithUpperLimits`): a non-detection is a statement about flux, and
    the magnitude of zero flux is undefined. ``"mag"``/``"magnitude"`` and
    ``"flux"``/``"flux_density"``/``"luminosity"`` are the accepted explicit spellings; anything
    else raises.

    Public because the samplers need the rule *without* building a likelihood — a caller who
    supplies their own ``log_prob_fn`` still needs a resolved space to label the predictive metrics
    with, and a second copy of this rule is how the two would drift.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.likelihood import resolve_space
    >>> lc = wp.LightCurve(time=[1.0, 2.0], band=["lsstr"] * 2, magnitude=[22.0, 24.5],
    ...                    magnitude_err=[0.1, float("nan")], upper_limit=[False, True])
    >>> resolve_space(lc, "auto"), resolve_space(lc.where(upper_limit=False), "auto")
    ('flux', 'magnitude')
    """
    if space is None or str(space).lower() == "auto":
        if lc.data_mode == "magnitude" and not _has_upper_limits(lc):
            return "magnitude"
        return "flux"
    s = str(space).lower()
    if s in ("mag", "magnitude"):
        return "magnitude"
    if s in ("flux", "flux_density", "luminosity"):
        return "flux"
    raise ValueError(f"space must be 'flux', 'magnitude', or 'auto'; got {space!r}")


#: Back-compat alias — this was private until the samplers needed the rule too.
_resolve_space = resolve_space


def flux_to_space(model_flux, space, zeropoint_jy=AB_ZEROPOINT_JY):
    """Convert a **model** flux [Jy] into an already-resolved comparison ``space``.

    Flux space is the identity. Magnitude space clips at ``_MIN_FLUX_JY`` before converting, so a
    non-positive model flux maps to a very faint magnitude (~+759) — a large but FINITE chi-square
    penalty. That keeps the log-likelihood finite (such draws are effectively rejected rather than
    NaN-ed) and avoids spurious non-finite values that WAIC would otherwise drop. It is the
    model-side counterpart of :func:`whisper_cbpf.io.photometry.flux_density_to_mag`, which returns
    NaN instead because inventing a magnitude for a *measured* negative flux would fabricate a
    detection.

    ``space`` must already be ``'flux'`` or ``'magnitude'`` — pass ``resolve_space(lc, space)``.
    Module-level for the same reason :func:`resolve_space` is: a caller that needs only the
    CONVERSION should not have to build a likelihood to reach it, and a second copy of the floor is
    how the two would drift. :meth:`GaussianLikelihood.model_in_space` is this function bound to the
    instance's space and zero point.
    """
    if space not in ("flux", "magnitude"):
        raise ValueError(f"flux_to_space needs a resolved space ('flux' or 'magnitude'), got "
                         f"{space!r}; pass resolve_space(lc, space).")
    mf = np.asarray(model_flux, dtype=float)
    if space == "magnitude":
        return flux_density_to_mag(np.clip(mf, _MIN_FLUX_JY, None), zeropoint_jy=zeropoint_jy)
    return mf


class GaussianLikelihood:
    """Independent-Gaussian likelihood in flux or magnitude space.

    ``ln L = -1/2 sum[((y - m) / sigma)^2 + ln(2 pi sigma^2)]``, with its normalising constant, so
    AIC and BIC from it are absolute. A light curve with upper limits is refused (use
    :class:`GaussianLikelihoodWithUpperLimits`, which :func:`make_likelihood` picks for it).

    Parameters
    ----------
    lc : LightCurve
        The data, detections only.
    space : {"auto", "flux", "magnitude"}, default "auto"
        The space residuals are taken in (:func:`resolve_space`).
    zeropoint_jy : float, default 3631
        Zero point used to convert between flux and magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[1.0, 2.0, 3.0],
    ...                    flux_err=[0.1] * 3)
    >>> lik = wp.GaussianLikelihood(lc, space="flux")
    >>> round(lik.log_likelihood(np.array([1.0, 2.0, 3.0])), 4)       # a perfect model
    4.1509
    >>> round(lik.log_likelihood(np.array([1.1, 2.0, 3.0])), 4)       # one point 1 sigma off
    3.6509
    """

    #: Does this class READ ``lc.upper_limit``? Only :class:`GaussianLikelihoodWithUpperLimits`
    #: does. Every other likelihood here would score a limiting flux as though it were a
    #: measurement, so instead of doing that quietly it refuses the data -- see :meth:`__init__`.
    _handles_upper_limits = False

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
        # An EMPTY light curve is not a degenerate fit, it is a mistake upstream -- and without
        # this guard every sampler completes on it and reports `AIC = 2k` with `BIC = -inf`
        # (k*log(0)), which looks like a result. The usual cause is ordering: `select_time_window`
        # applied before `set_explosion_date` compares day-since-explosion bounds against raw MJD
        # and silently drops every row.
        if self.y.size == 0:
            raise ValueError(
                "the light curve has no data points, so there is nothing to fit. A common cause is "
                "calling select_time_window(...) BEFORE set_explosion_date(...) -- the window is "
                "then compared against raw MJD. Check lc.n_points after each selection step.")
        self._check_censoring(lc)
        self._check_errors(lc)
        self._log_norm = -0.5 * np.sum(_LN2PI + 2.0 * np.log(self.sigma))

    def _check_censoring(self, lc):
        """Refuse a light curve carrying non-detections, unless this class models them.

        Two silent wrongs, one rule. (1) Whisper's loader stores **NaN** in the error column of an
        ``upper_limit=True`` row, and ``_log_norm`` sums ``log(sigma)`` over every row -- so one
        such row makes ``log_likelihood`` NaN at every parameter value, and every sampler completes
        on that and reports NaN ``max_log_likelihood``/``aic``/``bic`` as a result. (2) When the
        limits happen to carry a *finite* placeholder error instead, nothing is NaN and the
        limiting flux is fitted as if it were a measurement -- the censoring information is
        dropped without a word. That second case is how a ``gaussian_scatter`` fit on a censored
        light curve silently becomes a plain scatter fit: there is no combined
        scatter-and-censoring likelihood in this package, and picking one of the two in silence is
        the worst of the three available answers.
        """
        ul = lc.upper_limit
        if self._handles_upper_limits or ul is None:
            return
        n_ul = int(np.sum(np.asarray(ul, dtype=bool)))
        if n_ul:
            raise ValueError(
                f"{type(self).__name__} does not model censoring, but {n_ul} of "
                f"{int(np.size(ul))} rows are flagged upper_limit=True. A non-detection is a "
                f"CENSORED measurement, not a measurement with an error bar: its y value is a "
                f"limit and its error bar is typically NaN, which would make the log-likelihood "
                f"NaN at every parameter value. Use likelihood='auto' (the default) or "
                f"'upper_limits' with space='auto' or 'flux', which models the censoring, or drop "
                f"those rows with lc.where(upper_limit=False) and fit the detections alone. (There "
                f"is no combined free-scatter + upper-limit likelihood; the two are mutually "
                f"exclusive here.)")

    def _check_errors(self, lc, mask=None):
        """Refuse an error bar that is not finite and positive, BEFORE it becomes a NaN fit.

        ``_log_norm`` sums ``log(sigma)`` over every row and ``log_likelihood`` divides the residual
        by it, so ONE NaN sigma makes the log-likelihood NaN at every parameter value. The censored
        rows are handled earlier by :meth:`_check_censoring`; this catches the rest — a detection
        with a missing or zero error bar, which is just as fatal and much easier to overlook.

        ``mask`` restricts the check to the rows this likelihood actually divides by; see
        :meth:`GaussianLikelihoodWithUpperLimits._check_errors`, the only override.
        """
        bad = ~(np.isfinite(self.sigma) & (self.sigma > 0.0))
        if mask is not None:
            bad = bad & np.asarray(mask, dtype=bool)
        if not np.any(bad):
            return
        raise ValueError(
            f"{type(self).__name__} in {self.space} space was given {int(np.sum(bad))} of "
            f"{self.sigma.size} error bar(s) that are not finite and positive, so the "
            f"log-likelihood would be NaN at every parameter value and the fit would report NaN "
            f"scores rather than failing. Fix or drop those rows.")

    def model_in_space(self, model_flux):
        """This instance's :func:`flux_to_space` — the model flux in the comparison space."""
        return flux_to_space(model_flux, self.space, self.zeropoint_jy)

    def log_likelihood(self, model_flux):
        res = (self.y - self.model_in_space(model_flux)) / self.sigma
        return float(-0.5 * np.sum(res * res) + self._log_norm)

    def log_likelihood_pointwise(self, model_flux):
        """Per-data-point log-likelihood (length ``n_data``). Sums to :meth:`log_likelihood`; the
        pointwise terms are what WAIC needs (see :func:`whisper_cbpf.metrics.waic`)."""
        res = (self.y - self.model_in_space(model_flux)) / self.sigma
        return -0.5 * (res * res) - 0.5 * (_LN2PI + 2.0 * np.log(self.sigma))

    def summary(self):
        return {"likelihood": "gaussian", "space": self.space, "n_data": int(self.y.size)}


class GaussianLikelihoodWithUpperLimits(GaussianLikelihood):
    """Gaussian for detections + a censoring (CDF) term for upper limits — **flux space only**.

    Upper-limit ``y`` values are the limiting flux in Jy. ``upper_limit_sigma`` is the N-sigma level
    of those limits (e.g. 5.0 for 5-sigma), so a limit contributes
    ``P(true flux < limit) = Phi((limit - model) / (limit / upper_limit_sigma))``. When it is not
    given it is ``lc.meta["upper_limit_sigma"]`` (the survey presets record 5), else
    :data:`DEFAULT_UPPER_LIMIT_SIGMA` (5; it was 3 before 0.2.0). ``summary()`` reports the value
    used.

    ``space='magnitude'`` is **refused**. A non-detection is a statement about FLUX -- "the source
    was fainter than this limiting flux" -- and the magnitude of zero flux is undefined
    (``-2.5 log10(0) = +inf``), so the censoring integral has no magnitude-space counterpart that
    is a bound on the same quantity. A magnitude branch (a survival term
    ``1 - Phi((limit - model) / (POGSON / upper_limit_sigma))`` with a width that came from the
    Pogson constant rather than from the data); it was removed deliberately rather than ported to
    the JAX backend, so the two backends implement one density and not two. ``space='auto'``
    resolves to flux for a light curve with upper limits (:func:`resolve_space`), so magnitude data
    with limits needs no conversion: its detections' errors become flux errors
    (``sigma_F = 0.921 F sigma_m``).

    Parameters
    ----------
    lc : LightCurve
        Detections and upper limits (``lc.upper_limit``).
    space : {"auto", "flux"}, default "auto"
        ``"magnitude"`` is refused (see above).
    upper_limit_sigma : float, optional
        The limits' significance; default ``lc.meta["upper_limit_sigma"]``, else 5.
    zeropoint_jy : float, default 3631

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["lsstr"] * 3,
    ...                    magnitude=[24.5, 22.0, 22.3],
    ...                    magnitude_err=[float("nan"), 0.1, 0.1], upper_limit=[True, False, False])
    >>> lik = wp.make_likelihood(lc)                   # space='auto' -> flux, limits censored
    >>> lik.summary()["space"], lik.summary()["upper_limits"], lik.upper_limit_sigma
    ('flux', 1, 5.0)
    """

    _handles_upper_limits = True

    def __init__(self, lc, space="auto", upper_limit_sigma=None, zeropoint_jy=AB_ZEROPOINT_JY):
        # Refused BEFORE super().__init__ so nothing is half-built. `space='auto'` resolves to flux
        # for a light curve with limits, so only an explicit magnitude space (or a curve without a
        # single limit, fitted as magnitudes) reaches this.
        if _resolve_space(lc, space) == "magnitude":
            raise ValueError(
                f"{type(self).__name__} is flux-only; space={space!r} resolves to 'magnitude'. A "
                f"non-detection is a statement about FLUX -- the source was fainter than a "
                f"limiting flux -- and the magnitude of zero flux is undefined "
                f"(-2.5*log10(0) = +inf), so there is no magnitude-space censoring integral over "
                f"the same quantity. Pass space='auto' (the default; flux for a light curve with "
                f"upper limits) or space='flux', or fit the detections alone with "
                f"lc.where(upper_limit=False).")
        super().__init__(lc, space=space, zeropoint_jy=zeropoint_jy)
        ul = lc.upper_limit
        self.detections = np.ones(self.y.size, dtype=bool) if ul is None else ~np.asarray(ul, dtype=bool)
        if upper_limit_sigma is None:
            meta = getattr(lc, "meta", None) or {}
            upper_limit_sigma = meta.get("upper_limit_sigma")
            if upper_limit_sigma is None:
                upper_limit_sigma = DEFAULT_UPPER_LIMIT_SIGMA
        self.upper_limit_sigma = float(upper_limit_sigma)
        if not (np.isfinite(self.upper_limit_sigma) and self.upper_limit_sigma > 0.0):
            raise ValueError(f"upper_limit_sigma must be finite and > 0 (the N-sigma level of the "
                             f"limits); got {upper_limit_sigma!r}.")
        if np.any(~self.detections) and np.any(~np.isfinite(self.y[~self.detections])):
            raise ValueError("Upper limits require finite limiting fluxes.")
        det = self.detections
        self._log_norm_det = (-0.5 * np.sum(_LN2PI + 2.0 * np.log(self.sigma[det]))
                              if np.any(det) else 0.0)

    def _check_errors(self, lc, mask=None):
        """Only the DETECTIONS need a usable error bar.

        The censored rows carry their own width, ``limit / upper_limit_sigma``, and whisper's loader
        stores NaN in ``flux_err`` for a non-detection -- which is exactly why ``_log_norm_det`` is
        computed over detections only.
        """
        ul = lc.upper_limit
        det = np.ones(self.sigma.size, dtype=bool) if ul is None else ~np.asarray(ul, dtype=bool)
        super()._check_errors(lc, mask=det if mask is None else (det & np.asarray(mask, dtype=bool)))

    @staticmethod
    def _cdf(x):
        """Standard normal CDF, via ``ndtr`` rather than ``0.5 * (1 + erf(x / sqrt(2)))``.

        Mathematically the same function; numerically not. ``1 + erf(w)`` cancels catastrophically
        in the left tail, and this term lives in that tail — it is evaluated exactly when a model
        is far ABOVE an upper limit, which is where the likelihood has to push back hardest.
        Measured on the same eight non-detections, ``log P`` per row:

        ==========  =====================  ============  ===================
        ``z``       ``0.5(1+erf)`` float64  ``ndtr``      what changed
        ==========  =====================  ============  ===================
        -4.90       -14.5464826            -14.5464826   nothing
        -8.94       **-69.0775528**        -43.0720974   26 log units
        -10.24      **-69.0775528**        -55.6570429   13 log units
        ==========  =====================  ============  ===================

        ``erf`` returns exactly -1 for ``w <= -8.3`` in float64, so the old form floored every such
        row at the ``_MIN_PROB`` clip: a FLAT plateau of -69.0776 with zero gradient across the
        whole range a gradient sampler has to cross to bring the model back under the limit.
        ``ndtr`` calls ``erfc`` on the tail and has no cancellation, which is also what makes the
        JAX twin agree in float32 (measured: -55.6570396 against this function's -55.6570429,
        where the ``erf`` form gave -14.6896 against -14.5465 at ``z = -4.90`` alone).
        """
        return ndtr(x)

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
            sigma_ul = limit / self.upper_limit_sigma
            prob = self._cdf((limit - model_ul) / sigma_ul)             # true flux < limit
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
            prob = self._cdf((limit - model_ul) / (limit / self.upper_limit_sigma))
            out[ul] = np.log(np.clip(prob, _MIN_PROB, 1.0 - _MIN_PROB))
        return out

    def summary(self):
        return {"likelihood": "gaussian_upper_limits", "space": self.space,
                "n_data": int(self.y.size), "detections": int(np.sum(self.detections)),
                "upper_limits": int(np.sum(~self.detections)),
                "upper_limit_sigma": self.upper_limit_sigma}


class MixtureGaussianLikelihood(GaussianLikelihood):
    """Outlier-robust two-component Gaussian mixture (inlier ``sigma`` + wide ``sigma*scale``).

    Each point is an inlier with probability ``alpha`` (its reported error) or an outlier (the
    error times ``sigma_out_scale``), so one bad point cannot dominate the fit.

    Parameters
    ----------
    lc : LightCurve
    space : {"auto", "flux", "magnitude"}, default "auto"
    alpha : float, default 0.9
        Inlier fraction.
    sigma_out_scale : float, default 10.0
        Width of the outlier component, in units of each point's error.
    zeropoint_jy : float, default 3631

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[1.0, 2.0, 3.0],
    ...                    flux_err=[0.1] * 3)
    >>> model = np.array([1.0, 2.0, 5.0])                            # a 20-sigma outlier
    >>> round(wp.MixtureGaussianLikelihood(lc, space="flux").log_likelihood(model), 4)
    -2.6429
    >>> round(wp.GaussianLikelihood(lc, space="flux").log_likelihood(model), 4)
    -195.8491
    """

    def __init__(self, lc, space="auto", alpha=0.9, sigma_out_scale=10.0, zeropoint_jy=AB_ZEROPOINT_JY):
        super().__init__(lc, space=space, zeropoint_jy=zeropoint_jy)
        self.alpha = float(alpha)
        self.sigma_out = self.sigma * float(sigma_out_scale)

    def log_likelihood(self, model_flux):
        """Total log-likelihood: the log-sum-exp of the inlier and outlier components."""
        return float(np.sum(self.log_likelihood_pointwise(model_flux)))

    def log_likelihood_pointwise(self, model_flux):
        """Per-point mixture log-density; sums exactly to :meth:`log_likelihood`.

        This override is load-bearing. Without it the class inherited
        :meth:`GaussianLikelihood.log_likelihood_pointwise`, a *plain Gaussian*, while
        :meth:`log_likelihood` computed the mixture — so ``waic`` and ``predictive_metrics``,
        which both dispatch on ``hasattr(lik, "log_likelihood_pointwise")``, scored a mixture fit
        under a density it was not fitted with, and reported it beside an AIC that came from the
        mixture. Measured on 25 points: total 29.1446 against a pointwise sum of 31.4662.
        """
        res = self.y - self.model_in_space(model_flux)
        logp_in = -0.5 * _LN2PI - np.log(self.sigma) - 0.5 * (res / self.sigma) ** 2
        logp_out = -0.5 * _LN2PI - np.log(self.sigma_out) - 0.5 * (res / self.sigma_out) ** 2
        return np.logaddexp(np.log(self.alpha) + logp_in,
                            np.log(1.0 - self.alpha) + logp_out)

    def summary(self):
        """Identifying metadata for the fit record: kind, space, point count and ``alpha``."""
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

    Parameters
    ----------
    lc : LightCurve
    space : {"auto", "flux", "magnitude"}, default "auto"
    scatter_param : str, default "sigma"
        The prior parameter carrying the extra scatter.
    zeropoint_jy : float, default 3631

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[1.0, 2.0, 3.0],
    ...                    flux_err=[0.1] * 3)
    >>> lik = wp.GaussianLikelihoodWithScatter(lc, space="flux")
    >>> round(lik.log_likelihood(np.array([1.1, 2.0, 3.0]), sigma_extra=0.1), 4)
    2.8612
    >>> lik.log_likelihood(np.array([1.0, 2.0, 3.0])) == wp.GaussianLikelihood(
    ...     lc, space="flux").log_likelihood(np.array([1.0, 2.0, 3.0]))   # no scatter: the Gaussian
    True
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

    Parameters
    ----------
    name : str
        The ``kind`` name (matched case-insensitively).
    likelihood_cls : type
    overwrite : bool, default False
        Replace a likelihood already registered under ``name``.

    Raises
    ------
    ValueError
        ``name`` is taken and ``overwrite`` is False.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> class Heavier(wp.MixtureGaussianLikelihood):
    ...     def __init__(self, lc, space="auto", **kw):
    ...         super().__init__(lc, space=space, alpha=0.8, **kw)
    >>> wp.register_likelihood("heavier_tails", Heavier, overwrite=True)
    >>> lc = wp.LightCurve(time=[1.0, 2.0], band=["r"] * 2, flux=[1.0, 2.0], flux_err=[0.1] * 2)
    >>> wp.make_likelihood(lc, kind="heavier_tails").alpha
    0.8
    """
    key = str(name).lower()
    if key in _LIKELIHOODS and not overwrite:
        raise ValueError(f"Likelihood {name!r} already registered (pass overwrite=True).")
    _LIKELIHOODS[key] = likelihood_cls


def list_likelihoods():
    """Sorted list of registered likelihood ``kind`` names (incl. aliases).

    Returns
    -------
    list of str

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> {"gaussian", "upper_limits", "gaussian_scatter", "mixture"} <= set(wp.list_likelihoods())
    True
    """
    return sorted(_LIKELIHOODS)


def make_likelihood(lc, kind="auto", space="auto", **kwargs):
    """Build the data-appropriate likelihood (override with ``kind`` / ``space``).

    ``kind='auto'`` picks ``gaussian_upper_limits`` when the light curve has upper limits, else
    ``gaussian``. ``space='auto'`` picks magnitude space for magnitude data without upper limits,
    flux space otherwise (:func:`resolve_space`), so a survey light curve with limits gets the
    censored likelihood with no argument; its limits' significance is
    ``lc.meta["upper_limit_sigma"]`` (5 for the survey presets), else 5, unless
    ``upper_limit_sigma=`` is passed. Use :func:`list_likelihoods` to see the available ``kind``
    names and :func:`register_likelihood` to add your own.

    Parameters
    ----------
    lc : LightCurve
    kind : str, default "auto"
        A name from :func:`list_likelihoods`.
    space : {"auto", "flux", "magnitude"}, default "auto"
    **kwargs
        Passed to the likelihood class (``upper_limit_sigma``, ``scatter_param``, ``alpha``, ...).

    Returns
    -------
    GaussianLikelihood or a subclass

    Raises
    ------
    ValueError
        An unknown ``kind``.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=[1.0, 2.0], band=["ztfg"] * 2, magnitude=[20.6, 19.1],
    ...                    magnitude_err=[float("nan"), 0.05], upper_limit=[True, False])
    >>> type(wp.make_likelihood(lc)).__name__
    'GaussianLikelihoodWithUpperLimits'
    >>> type(wp.make_likelihood(lc.where(upper_limit=False))).__name__
    'GaussianLikelihood'
    """
    if kind == "auto":
        has_ul = lc.upper_limit is not None and bool(np.any(lc.upper_limit))
        kind = "gaussian_upper_limits" if has_ul else "gaussian"
    key = str(kind).lower()
    if key not in _LIKELIHOODS:
        raise ValueError(f"Unknown likelihood {kind!r}. Available: {list_likelihoods()}")
    return _LIKELIHOODS[key](lc, space=space, **kwargs)
