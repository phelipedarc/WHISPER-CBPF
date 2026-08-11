"""Prior distributions for model parameters.

Small, picklable distribution classes (so priors can cross process boundaries for parallel ABC) with
the hooks future samplers need: ``sample`` (ABC/MCMC init), ``log_prob`` (MCMC), ``rescale``
(unit-cube -> value, for nested sampling).
"""
from __future__ import annotations

import numpy as np


class Uniform:
    """Uniform prior on ``[low, high]``."""

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
    """Log-uniform (Jeffreys) prior on ``[low, high]``, ``low > 0`` (good for scale params)."""

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


class Prior:
    """A set of named, independent parameter priors."""

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

    def __repr__(self):
        return f"Prior({self.distributions})"
