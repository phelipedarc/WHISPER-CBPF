"""Prior distributions for model parameters — numpy backend (the default).

The distribution classes themselves. ``Uniform``, ``LogUniform`` and ``Prior`` keep the bodies of
the single-module ``priors.py`` this replaced; ``Normal``, ``TruncatedNormal`` and ``Fixed`` were
added for 0.2.0. Rationale for the design is in the package docstring
(``whisper_cbpf/priors/__init__.py``); the JAX twin of ``Prior.log_prob`` is ``_jax.log_prob_jax``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special as _special

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class Uniform:
    """Uniform prior on ``[low, high]``.

    Parameters
    ----------
    low, high : float
        The range.
    name : str, optional

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.priors import Uniform
    >>> d = Uniform(0.0, 10.0)
    >>> d.bounds, d.rescale(0.25), round(float(d.log_prob(3.0)), 4)
    ((0.0, 10.0), 2.5, -2.3026)
    >>> 0.0 <= d.sample(np.random.default_rng(0)) <= 10.0
    True
    """

    def __init__(self, low, high, name=None):
        self.low = float(low)
        self.high = float(high)
        self.name = name

    def sample(self, rng):
        return float(rng.uniform(self.low, self.high))

    def log_prob(self, x):
        return -np.log(self.high - self.low) if self.low <= x <= self.high else -np.inf

    def rescale(self, u):
        return self.low + float(u) * (self.high - self.low)

    @property
    def bounds(self):
        return (self.low, self.high)

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


class LogUniform:
    """Log-uniform (Jeffreys) prior on ``[low, high]``, ``low > 0`` (good for scale params).

    Uniform in ``log x``: each decade gets the same prior mass.

    Parameters
    ----------
    low, high : float
        The range, ``0 < low < high``.
    name : str, optional

    Raises
    ------
    ValueError
        ``low <= 0``.

    Examples
    --------
    >>> from whisper_cbpf.priors import LogUniform
    >>> d = LogUniform(1e-3, 1.0)
    >>> round(d.rescale(0.5), 6)                     # the geometric midpoint
    0.031623
    """

    def __init__(self, low, high, name=None):
        if low <= 0:
            raise ValueError("LogUniform requires low > 0.")
        self.low = float(low)
        self.high = float(high)
        self.name = name
        self._lnlow = np.log(self.low)
        self._lnhigh = np.log(self.high)

    def sample(self, rng):
        return float(np.exp(rng.uniform(self._lnlow, self._lnhigh)))

    def log_prob(self, x):
        if self.low <= x <= self.high:
            return -np.log(x) - np.log(self._lnhigh - self._lnlow)
        return -np.inf

    def rescale(self, u):
        return float(np.exp(self._lnlow + float(u) * (self._lnhigh - self._lnlow)))

    @property
    def bounds(self):
        return (self.low, self.high)

    def __repr__(self):
        return f"LogUniform({self.low}, {self.high})"


def _check_normal(mu, sigma, what):
    mu, sigma = float(mu), float(sigma)
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"{what} needs a finite mu and a finite sigma > 0; got mu={mu!r}, "
                         f"sigma={sigma!r}.")
    return mu, sigma


class Normal:
    """Normal (Gaussian) prior ``N(mu, sigma)`` on the whole real line.

    For a measured quantity: a redshift from a host spectrum, an explosion epoch from a
    non-detection, an extinction. Every sampler samples it exactly: ``nested`` through
    :meth:`rescale` (the inverse CDF), ``abc`` / ``abc_smc`` through :meth:`sample`, ``abc_gpu`` /
    ``abc_smc_gpu`` through the same inverse CDF on the device, ``snpe`` / ``snpe_gpu`` through a
    torch Normal, and ``mcmc``, ``emcee_jax``, ``nuts_gpu`` and ``pymc_jax_gpu_*`` through
    :meth:`log_prob` (or its JAX twin). For a quantity that must stay in a range (a positive mass),
    use :class:`TruncatedNormal`.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation, > 0.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.priors import Normal
    >>> z = Normal(0.051, 0.002)
    >>> round(z.log_prob(0.051), 3)
    5.296
    >>> round(z.rescale(0.975), 4)
    0.0549
    >>> z.bounds
    (-inf, inf)
    """

    def __init__(self, mu, sigma, name=None):
        self.mu, self.sigma = _check_normal(mu, sigma, "Normal")
        self.name = name

    def sample(self, rng):
        return float(rng.normal(self.mu, self.sigma))

    def log_prob(self, x):
        z = (float(x) - self.mu) / self.sigma
        return -0.5 * z * z - math.log(self.sigma) - _LOG_SQRT_2PI

    def rescale(self, u):
        return float(self.ppf(u))

    def cdf(self, x):
        """``P(X <= x)``, vectorised."""
        return _special.ndtr((np.asarray(x, dtype=float) - self.mu) / self.sigma)

    def ppf(self, p):
        """The inverse CDF, vectorised: what :meth:`rescale` applies to a unit-cube coordinate."""
        return self.mu + self.sigma * _special.ndtri(np.asarray(p, dtype=float))

    @property
    def std(self):
        return self.sigma

    @property
    def bounds(self):
        return (-math.inf, math.inf)

    def __repr__(self):
        return f"Normal({self.mu}, {self.sigma})"


class TruncatedNormal:
    """``N(mu, sigma)`` restricted to ``[low, high]`` and renormalised; one side may be infinite.

    The density is exact, truncation normalisation included:
    ``log p(x) = log phi((x - mu) / sigma) - log sigma - log Z`` with
    ``Z = Phi(beta) - Phi(alpha)``, ``alpha = (low - mu) / sigma``, ``beta = (high - mu) / sigma``,
    computed in log space. Draws come from the inverse CDF (the probability integral transform:
    ``x = mu + sigma * Phi^-1(Phi(alpha) + u Z)`` for ``u ~ U(0, 1)``), evaluated on the upper tail's
    side when the range lies above ``mu``, so a range deep in either tail keeps its precision. The
    NUTS samplers sample it the same way, as a Uniform site on the CDF pushed through the inverse
    CDF, and the JAX samplers' density is this one. Which samplers take it: as for :class:`Normal`.

    Parameters
    ----------
    mu, sigma : float
        Location and scale of the untruncated normal (not the mean and sd of the result).
    low, high : float
        The range, ``low < high``; ``-inf`` / ``inf`` for a one-sided truncation.

    Examples
    --------
    >>> from whisper_cbpf.priors import TruncatedNormal
    >>> mej = TruncatedNormal(0.05, 0.1, 0.0, 1.0)          # a mass: positive
    >>> mej.bounds
    (0.0, 1.0)
    >>> mej.log_prob(-0.01)
    -inf
    >>> 0.0 < mej.rescale(0.01) < mej.rescale(0.5) < mej.rescale(0.99) < 1.0
    True
    """

    def __init__(self, mu, sigma, low, high, name=None):
        self.mu, self.sigma = _check_normal(mu, sigma, "TruncatedNormal")
        self.low, self.high = float(low), float(high)
        if not self.low < self.high:
            raise ValueError(f"TruncatedNormal needs low < high; got low={self.low!r}, "
                             f"high={self.high!r}.")
        self.name = name
        self._alpha = (self.low - self.mu) / self.sigma
        self._beta = (self.high - self.mu) / self.sigma
        # Upper tail: work with -z, whose range [-beta, -alpha] is below 0, where Phi keeps its
        # relative precision (Phi(8) rounds to 1, Phi(-8) = 6.2e-16 does not).
        self._flip = self._alpha > 0.0
        a, b = (-self._beta, -self._alpha) if self._flip else (self._alpha, self._beta)
        self._lo_cdf = float(_special.ndtr(a))                    # Phi of the range's lower end
        self._mass = float(_special.ndtr(b) - self._lo_cdf)       # Z
        la, lb = float(_special.log_ndtr(a)), float(_special.log_ndtr(b))
        self._log_mass = lb + math.log1p(-math.exp(la - lb)) if la < lb else -math.inf
        if not (math.isfinite(self._log_mass) and self._mass > 0.0):
            raise ValueError(
                f"TruncatedNormal({mu}, {sigma}, {low}, {high}) keeps no probability that float64 "
                f"can resolve: [low, high] lies {min(abs(self._alpha), abs(self._beta)):.0f} sigma "
                f"from mu. Move mu or sigma so that the range carries the prior's mass.")

    def sample(self, rng):
        return self.rescale(rng.uniform())

    def log_prob(self, x):
        x = float(x)
        if not self.low <= x <= self.high:
            return -math.inf
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - math.log(self.sigma) - _LOG_SQRT_2PI - self._log_mass

    def rescale(self, u):
        return float(self.ppf(u))

    def cdf(self, x):
        """``P(X <= x)`` of the truncated distribution, vectorised; 0 below ``low``, 1 above."""
        z = (np.asarray(x, dtype=float) - self.mu) / self.sigma
        if self._flip:
            p = 1.0 - (_special.ndtr(-z) - self._lo_cdf) / self._mass
        else:
            p = (_special.ndtr(z) - self._lo_cdf) / self._mass
        return np.clip(p, 0.0, 1.0)

    def ppf(self, p):
        """The inverse CDF, vectorised, clipped into ``[low, high]``."""
        p = np.asarray(p, dtype=float)
        if self._flip:
            z = -_special.ndtri(self._lo_cdf + (1.0 - p) * self._mass)
        else:
            z = _special.ndtri(self._lo_cdf + p * self._mass)
        return np.clip(self.mu + self.sigma * z, self.low, self.high)

    @property
    def std(self):
        from scipy.stats import truncnorm
        return float(truncnorm(self._alpha, self._beta, loc=self.mu, scale=self.sigma).std())

    @property
    def bounds(self):
        return (self.low, self.high)

    def __repr__(self):
        return f"TruncatedNormal({self.mu}, {self.sigma}, {self.low}, {self.high})"


class Fixed:
    """A parameter held at one value: pinned, not fitted.

    Every sampler leaves it out of the sampled dimensions (and out of the parameter count behind
    AIC and BIC), evaluates the model at the value, and reports it as a constant column
    (``result.info["fixed"]``). :func:`whisper_cbpf.likelihood_max_opt` holds it at its value too.

    Parameters
    ----------
    value : float
        The value.

    Examples
    --------
    >>> from whisper_cbpf.priors import Fixed, LogUniform, Prior
    >>> prior = Prior({"mej": LogUniform(1e-3, 1.0), "redshift": Fixed(0.051)})
    >>> prior.fixed
    {'redshift': 0.051}
    >>> prior.sample()["redshift"]
    0.051
    """

    def __init__(self, value, name=None):
        self.value = float(value)
        if not math.isfinite(self.value):
            raise ValueError(f"Fixed needs a finite value; got {value!r}.")
        self.name = name

    def sample(self, rng):
        return self.value

    def log_prob(self, x):
        if float(x) == self.value:
            return 0.0
        raise ValueError(
            f"{self!r}: the prior density was asked for at {float(x)!r}. A Fixed parameter is not "
            f"sampled, and the caller moved it. Every whisper sampler holds it at its value; a "
            f"sampler of your own must too (leave it out of the sampled dimensions), or pin it in "
            f"the model instead (e.g. register_redback(..., pin={{...}})).")

    def rescale(self, u):
        return self.value

    @property
    def std(self):
        return 0.0

    @property
    def bounds(self):
        return (self.value, self.value)

    def __repr__(self):
        return f"Fixed({self.value})"


def family(d):
    """The family name the samplers dispatch on: the class name, except that ``Normal``,
    ``TruncatedNormal`` and ``Fixed`` must be instances of these classes (a look-alike of that
    name, say one with only ``bounds``, is reported as ``"foreign <name>"`` and refused)."""
    name = type(d).__name__
    if name in ("Normal", "TruncatedNormal", "Fixed") and not isinstance(
            d, (Normal, TruncatedNormal, Fixed)):
        return f"foreign {name}"
    return name


class Prior:
    """A set of named, independent parameter priors.

    Parameters
    ----------
    distributions : dict
        ``{parameter: distribution}``: :class:`Uniform`, :class:`LogUniform`, :class:`Normal`,
        :class:`TruncatedNormal` or :class:`Fixed`.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> prior = wp.Prior({"amplitude": wp.Uniform(0.1, 10.0), "tau": wp.LogUniform(1.0, 100.0),
    ...                   "z": wp.Fixed(0.05)})
    >>> prior.names, prior.fixed
    (['amplitude', 'tau', 'z'], {'z': 0.05})
    >>> draw = prior.sample(np.random.default_rng(1))
    >>> bool(np.isfinite(prior.log_prob(draw)))
    True
    """

    def __init__(self, distributions):
        self.distributions = dict(distributions)

    @property
    def names(self):
        return list(self.distributions)

    def sample(self, rng=None):
        """Draw a parameter dict. ``rng`` is a ``numpy.random.Generator`` (made if None)."""
        rng = np.random.default_rng() if rng is None else rng
        return {name: dist.sample(rng) for name, dist in self.distributions.items()}

    def log_prob(self, params):
        return float(sum(self.distributions[n].log_prob(params[n]) for n in self.distributions))

    def rescale(self, unit_cube):
        return {n: d.rescale(unit_cube[i]) for i, (n, d) in enumerate(self.distributions.items())}

    @property
    def bounds(self):
        return {n: d.bounds for n, d in self.distributions.items()}

    @property
    def fixed(self):
        """``{name: value}`` of the :class:`Fixed` parameters (empty when there are none)."""
        return {n: d.value for n, d in self.distributions.items() if isinstance(d, Fixed)}

    def __repr__(self):
        return f"Prior({self.distributions})"
