"""Phenomenological per-band light-curve models in JAX: smooth power law, exponential rise with
power-law decline, Bazin, Villar.

Adapted from redback-jax (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback-jax,
commit 3431e2e, GPL-3.0, and from redback (Sarin et al.), https://github.com/nikhil-sarin/redback,
GPL-3.0; modified for whisper_cbpf.

Four closed-form functions of time, with no photosphere, no SED, no redshift and no distance. They
reproduce redback's ``smooth_exponential_powerlaw``, ``exp_rise_powerlaw_decline``, ``bazin_sne``
and ``villar_sne`` to float64 round-off (the agreement is measured in
``tests/test_adopted_phenomenological.py``), with these changes:

* ``exp_rise_powerlaw_decline`` clamps the ratio inside its logarithm. The logarithm was evaluated
  at negative arguments for every epoch before ``t0`` and masked afterwards, and ``0 * nan`` made
  the gradient NaN whenever one epoch preceded ``t0``.
* ``smooth_exponential_powerlaw`` blends with the two logistic weights ``sigmoid(-z)`` and
  ``sigmoid(z)``, ``z = 2 s ln(t / tpeak)``, in log space. This is the same function as
  ``0.5 (1 + tanh(s ln x))`` and ``1 - 0.5 (1 + tanh(s ln x))``, but the second form rounds to 0 in
  float32 while the term it multiplies is huge, which lost the tail of the curve. It returns 0 for
  ``t <= 0``, where redback returns 0 or NaN depending on ``alpha_2``.
* ``bazin_sne`` is evaluated as ``exp(-dt/tau_fall - softplus(-dt/tau_rise))``, which is the same
  quotient without the overflow of ``exp(-dt/tau_rise)`` far before ``t0``.
* ``villar_sne`` keeps the input order of the epochs; redback concatenates its two branches, which
  reorders an unsorted array. It also takes an array ``aa`` and ``cc`` (one per band), as
  ``bazin_sne`` does.

**Bands.** redback evaluates these models with shared shape parameters and one amplitude per band
(``aa`` and ``bb`` of ``bazin_sne``, ``m_peak`` of ``exp_rise_powerlaw_decline``). The model
factory :func:`phenomenological_model` does the same: bound to ``band_names`` it carries
``aa_<band>``, ``bb_<band>`` and so on, and ``band_idx`` picks the amplitude of each observation,
as for the supernova and kilonova factories. Bound to no bands it is band-independent, with
redback's parameter names, and ignores the band like ``flare_jax``.

**Units.** ``bazin_sne``, ``villar_sne`` and ``smooth_exponential_powerlaw`` return the units of
their amplitudes, so the model's amplitudes are flux densities in Jy.
``exp_rise_powerlaw_decline`` returns a magnitude; the model reads it as AB and returns Jy, with
zero flux before ``t0`` (the function itself returns NaN there, as redback does).

**Precision.** Everything follows the session's precision (``jax_enable_x64``); nothing sets it.
In float64 the functions agree with redback at round-off. In float32, against float64 over
200 draws of redback's priors on an 80-epoch grid, the relative error is at worst 2e-6
(``smooth_exponential_powerlaw``) and 7e-6 (``bazin_sne``), 99th percentile 7e-6 for
``villar_sne``; it is unbounded only where the flux underflows float32 or crosses zero.
``exp_rise_powerlaw_decline`` needs a clock that starts near the event: on a relative clock the
magnitude is within 2e-3 mag, but float32 resolves 0.004 d at MJD 59000, and ``t0``, ``t_peak`` and
the epochs are each rounded, so on a raw MJD clock it is off by up to 1.3e-2 mag. Put the light
curve on a clock that starts near the event (``lc.set_explosion_date``) and move ``t0`` and
``t_peak`` with it. ``Model.predict`` also subtracts the first epoch on the host in float64 before it
casts, in a float32 session; ``Model.predict_jax`` receives its parameters already in the session's
precision and cannot.

References
----------
Bazin et al. 2009, https://ui.adsabs.harvard.edu/abs/2009A%26A...499..653B/abstract
Villar et al. 2019, https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract;
Arnett 1982, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract

redback cites no reference for ``smooth_exponential_powerlaw`` or ``exp_rise_powerlaw_decline``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ...io.photometry import AB_ZEROPOINT_JY
from . import _phenomenological_spec as _spec
from ._phenomenological_spec import SPECS, default_prior, parameter_names  # noqa: F401

__all__ = ["smooth_exponential_powerlaw", "exp_rise_powerlaw_decline", "bazin_sne", "villar_sne",
           "phenomenological_models", "phenomenological_model", "register_phenomenological"]


# --- the four functions ---------------------------------------------------------------------------
#
# Each ``_*_core`` is elementwise: every argument may be a scalar or an array, and the result has
# their broadcast shape. The public functions arrange the band axis around them; the model factory
# calls them with one amplitude per observation.

def _smooth_core(time, a_1, tpeak, alpha_1, alpha_2, smoothing_factor):
    """a_1 * (x**alpha_1 * sigmoid(-z) + x**alpha_2 * sigmoid(z)), x = t/tpeak, z = 2 s ln x."""
    t_norm = time / tpeak
    positive = t_norm > 0
    x = jnp.where(positive, t_norm, 1.0)               # double where: no log(<=0) in the backward pass
    ln_x = jnp.log(x)
    z = 2.0 * smoothing_factor * ln_x
    pre = jnp.exp(alpha_1 * ln_x + jax.nn.log_sigmoid(-z))
    post = jnp.exp(alpha_2 * ln_x + jax.nn.log_sigmoid(z))
    return jnp.where(positive, a_1 * (pre + post), 0.0)


def _exp_rise_core(t, t0, m_peak, tau_rise, alpha, t_peak, delta):
    """``(magnitude, valid)``; the magnitude is finite for every epoch, ``valid`` is ``t >= t0``."""
    width = (t_peak - t0) * delta
    weight = 0.5 * (1.0 + jnp.tanh((t - t_peak) / width))
    m_rise = m_peak + 1.086 * ((t_peak - t) / tau_rise)
    valid = t >= t0
    ratio = (t - t0) / (t_peak - t0)
    # Clamp the ratio *inside* the logarithm: a negative ratio before t0 made the log NaN, and the
    # NaN survived the mask as 0 * nan in the backward pass.
    ratio = jnp.maximum(jnp.where(valid, ratio, 1.0), jnp.finfo(ratio.dtype).tiny)
    m_decline = m_peak + 2.5 * alpha * jnp.log10(ratio)
    return (1.0 - weight) * m_rise + weight * m_decline, valid


def _bazin_core(time, aa, bb, t0, tau_rise, tau_fall):
    """aa * exp(-dt/tau_fall) / (1 + exp(-dt/tau_rise)) + bb, dt = t - t0, in log space."""
    dt = time - t0
    return aa * jnp.exp(-dt / tau_fall - jax.nn.softplus(-dt / tau_rise)) + bb


def _villar_core(time, aa, cc, t0, tau_rise, tau_fall, gamma, nu):
    """Villar et al. 2019: a plateau-and-decline branch after ``t0 + gamma``, input order kept."""
    early = time < t0 + gamma
    norm = cc + aa * jax.nn.sigmoid((time - t0) / tau_rise)
    flux_early = norm * (1.0 - nu * ((time - t0) / gamma))
    late_arg = jnp.where(early, 0.0, time - t0 - gamma)  # the late branch overflows before the plateau
    flux_late = norm * ((1.0 - nu) * jnp.exp(-late_arg / tau_fall))
    return jnp.where(early, flux_early, flux_late)


def _per_band_grid(time, *amplitudes):
    """Amplitudes shaped to broadcast against ``time`` with a leading band axis, or scalars as is."""
    amps = [jnp.asarray(a, dtype=float) for a in amplitudes]
    if amps[0].ndim == 0:
        return amps
    if any(a.shape != amps[0].shape for a in amps):
        raise ValueError("the per-band amplitudes must have the same shape, got "
                         f"{[tuple(a.shape) for a in amps]}")
    return [a.reshape(a.shape + (1,) * jnp.ndim(time)) for a in amps]


@jax.jit
def smooth_exponential_powerlaw(time, a_1, tpeak, alpha_1, alpha_2, smoothing_factor):
    """Two power laws joined smoothly at ``tpeak``: rises as ``alpha_1``, falls as ``alpha_2``.

    ``a_1 * (x**alpha_1 * (1 - T) + x**alpha_2 * T)`` with ``x = time / tpeak`` and
    ``T = 0.5 * (1 + tanh(smoothing_factor * ln x))``. The value at ``tpeak`` is ``a_1``.

    Parameters
    ----------
    time : array_like
        Time since the start, in the units of ``tpeak``.
    a_1 : float or array_like
        Amplitude, in the units of the result.
    tpeak : float
        Peak time.
    alpha_1, alpha_2 : float
        Power-law index before and after the peak.
    smoothing_factor : float
        Width of the transition in ``ln x``; larger is sharper.

    Returns
    -------
    ndarray
        In the units of ``a_1``, shape of ``time``; zero where ``time <= 0``. Everything is
        broadcast, so an array ``a_1`` of the shape of ``time`` is one amplitude per epoch.

    References
    ----------
    redback cites no reference for this function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.phenomenological import smooth_exponential_powerlaw
    >>> t = jnp.array([1.0, 5.0, 10.0, 40.0])
    >>> f = smooth_exponential_powerlaw(t, 2.0, 10.0, 1.5, -1.5, 1.0)
    >>> f.shape, bool(f[2] == f.max()), round(float(f[2]), 6)
    ((4,), True, 2.0)
    """
    time = jnp.asarray(time, dtype=float)
    return _smooth_core(time, a_1, tpeak, alpha_1, alpha_2, smoothing_factor)


@jax.jit
def exp_rise_powerlaw_decline(t, t0, m_peak, tau_rise, alpha, t_peak, delta=0.5):
    """Magnitude of an exponential rise that turns into a power-law decline, blended at the peak.

    For ``t >= t0``, ``m = (1 - w) * m_rise + w * m_decline`` with
    ``m_rise = m_peak + 1.086 (t_peak - t) / tau_rise``,
    ``m_decline = m_peak + 2.5 alpha log10((t - t0) / (t_peak - t0))`` and
    ``w = 0.5 (1 + tanh((t - t_peak) / (delta (t_peak - t0))))``. The shape is the same in every
    filter; only ``m_peak`` differs.

    Parameters
    ----------
    t : array_like
        Epochs, flattened to one dimension, on the clock ``t0`` and ``t_peak`` are on.
    t0 : float
        Start of the event.
    m_peak : float or array_like
        Peak magnitude; one entry per filter if an array.
    tau_rise : float
        Rise time, in the units of ``t``.
    alpha : float
        Power-law decline index.
    t_peak : float
        Epoch of the peak; must be later than ``t0``.
    delta : float, default 0.5
        Width of the blend, as a fraction of ``t_peak - t0``.

    Returns
    -------
    ndarray
        NaN for ``t < t0``. Shape ``(n_times,)`` for a scalar ``m_peak`` or one-entry array,
        ``(n_times, n_filters)`` otherwise. The gradient is finite for every epoch, including the
        ones before ``t0`` (the ratio is clamped inside the logarithm). At ``t == t0`` the
        magnitude is finite, where redback's is ``-inf``.

    References
    ----------
    redback cites no reference for this function.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.phenomenological import exp_rise_powerlaw_decline
    >>> t = jnp.linspace(-5.0, 40.0, 10)
    >>> m = exp_rise_powerlaw_decline(t, 0.0, jnp.array([17.0, 17.5]), 3.0, 1.5, 10.0)
    >>> m.shape, bool(jnp.isnan(m[0, 0]))
    ((10, 2), True)
    >>> g = jax.grad(lambda p: jnp.sum(jnp.nan_to_num(
    ...     exp_rise_powerlaw_decline(t, 0.0, p, 3.0, 1.5, 10.0))))(17.0)
    >>> bool(jnp.isfinite(g))
    True
    """
    t = jnp.asarray(t, dtype=float).flatten()
    m_peak = jnp.atleast_1d(jnp.asarray(m_peak, dtype=float))
    mag, valid = _exp_rise_core(t[:, None], t0, m_peak[None, :], tau_rise, alpha, t_peak, delta)
    mag = jnp.where(valid, mag, jnp.nan)
    return mag.flatten() if m_peak.shape[0] == 1 else mag


@jax.jit
def bazin_sne(time, aa, bb, t0, tau_rise, tau_fall):
    """Bazin et al. 2009 supernova light curve, ``aa * exp(-dt/tau_fall) / (1 + exp(-dt/tau_rise)) + bb``.

    ``dt = time - t0``.

    Parameters
    ----------
    time : array_like
        Epochs, in the units of ``t0``.
    aa : float or array_like
        Normalisation; one entry per band if an array.
    bb : float or array_like
        Additive constant, same shape as ``aa``.
    t0 : float
        Start time (the time of the logistic's midpoint).
    tau_rise, tau_fall : float
        Rise and fall times.

    Returns
    -------
    ndarray
        In the units of ``aa``. Shape of ``time`` for a scalar ``aa``, ``aa.shape + time.shape``
        (``(n_band, n_time)``) for an array.

    References
    ----------
    Bazin et al. 2009, https://ui.adsabs.harvard.edu/abs/2009A%26A...499..653B/abstract

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.phenomenological import bazin_sne
    >>> t = jnp.linspace(-10.0, 60.0, 8)
    >>> bazin_sne(t, 3.0, 0.1, 0.0, 3.0, 20.0).shape
    (8,)
    >>> bazin_sne(t, jnp.array([3.0, 2.0]), jnp.array([0.1, 0.0]), 0.0, 3.0, 20.0).shape
    (2, 8)
    """
    time = jnp.asarray(time, dtype=float)
    aa, bb = _per_band_grid(time, aa, bb)
    return _bazin_core(time, aa, bb, t0, tau_rise, tau_fall)


@jax.jit
def villar_sne(time, aa, cc, t0, tau_rise, tau_fall, gamma, nu):
    """Villar et al. 2019 supernova light curve: rise, plateau of length ``gamma``, exponential fall.

    ``norm = cc + aa / (1 + exp(-(t - t0)/tau_rise))``; before ``t0 + gamma`` the flux is
    ``norm * (1 - nu (t - t0)/gamma)``, after it ``norm * (1 - nu) * exp(-(t - t0 - gamma)/tau_fall)``.

    Parameters
    ----------
    time : array_like
        Epochs, in the units of ``t0``. Any order; the result follows the input order.
    aa : float or array_like
        Amplitude; one entry per band if an array.
    cc : float or array_like
        Baseline flux, same shape as ``aa``.
    t0, tau_rise, tau_fall, gamma : float
        Start time, rise time, fall time, plateau duration.
    nu : float
        Between 0 and 1; ``nu = -beta / gamma / A`` in the notation of the paper.

    Returns
    -------
    ndarray
        In the units of ``aa``. Shape of ``time`` for a scalar ``aa``, ``aa.shape + time.shape``
        for an array.

    References
    ----------
    Villar et al. 2019, https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract;
    Arnett 1982, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from whisper_cbpf.models.jax.phenomenological import villar_sne
    >>> t = jnp.array([40.0, 2.0, 10.0, 25.0])
    >>> f = villar_sne(t, 2.0, 0.05, 0.0, 3.0, 20.0, 15.0, 0.5)
    >>> f.shape, bool(f[1] < f[2])
    ((4,), True)
    """
    time = jnp.asarray(time, dtype=float)
    aa, cc = _per_band_grid(time, aa, cc)
    return _villar_core(time, aa, cc, t0, tau_rise, tau_fall, gamma, nu)


# --- flux densities in Jy, one amplitude per observation ----------------------------------------

def _flux_exp_rise(t, t0, m_peak, tau_rise, alpha, t_peak):
    mag, valid = _exp_rise_core(t, t0, m_peak, tau_rise, alpha, t_peak, 0.5)
    return jnp.where(valid, AB_ZEROPOINT_JY * 10.0 ** (-0.4 * mag), 0.0)


_FLUX = {
    "smooth_exponential_powerlaw": _smooth_core,
    "exp_rise_powerlaw_decline": _flux_exp_rise,
    "bazin_sne": _bazin_core,
    "villar_sne": _villar_core,
}


def _session_dtype():
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class _PhenomPredict:
    """``predict`` / ``predict_jax`` of one bound phenomenological model, as a picklable object.

    Holds only strings, lists and index arrays; the jitted function is built on the first call and
    dropped by ``__getstate__``, so a multiprocess sampler recompiles in each worker rather than
    failing to pickle a closure.
    """

    def __init__(self, key, name, params, band_index):
        spec = SPECS[key]
        self.key, self.name, self.params = key, str(name), list(params)
        self.band_index = band_index
        self.n_bands = 0 if band_index is None else len(band_index.band_names)
        # for each redback argument: an index into theta, or the indices of its per-band entries
        self._slots = []
        for p in spec["parameters"]:
            if band_index is not None and p in spec["per_band"]:
                self._slots.append(np.array([self.params.index(f"{p}_{b}")
                                             for b in band_index.band_names]))
            else:
                self._slots.append(self.params.index(p))
        self._epoch_idx = [self.params.index(p) for p in spec["epochs"]]
        self._jit = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_jit"] = None
        return state

    def _flux(self, theta, times, band_idx):
        args = [theta[s] if isinstance(s, int) else theta[s][band_idx] for s in self._slots]
        return _FLUX[self.key](times, *args)

    def _need_bands(self, band_idx, what):
        if self.band_index is not None and band_idx is None:
            raise ValueError(f"model {self.name!r} carries one amplitude per band and needs a "
                             f"`{what}` array")

    def predict(self, parameters, times, bands=None):
        """whisper contract: ``(parameters, times, bands) -> flux density [Jy]``, as NumPy."""
        self._need_bands(bands, "bands")
        t = np.asarray(times, dtype=float)
        theta = np.array([float(parameters[k]) for k in self.params])
        if t.size == 0:
            return np.zeros(t.shape)
        bidx = None if self.band_index is None else self.band_index(bands)
        if not jax.config.jax_enable_x64 and self._epoch_idx:
            # float32 resolves 0.004 d at MJD 59000: move the clock to the first epoch in float64
            origin = float(t.min())
            t = t - origin
            theta[self._epoch_idx] -= origin
        if self._jit is None:
            self._jit = jax.jit(self._flux)
        dtype = _session_dtype()
        flux = self._jit(jnp.asarray(theta, dtype=dtype), jnp.asarray(t, dtype=dtype),
                         None if bidx is None else jnp.asarray(bidx))
        return np.asarray(flux, dtype=float)

    def predict_jax(self, theta, times, band_idx=None):
        """``predict`` for a traced caller: flat ``theta`` in ``Model.parameters`` order, integer
        ``band_idx`` (``predict_jax.band_index(bands)``, outside the trace), flux density in Jy.
        One parameter set; ``jax.vmap`` it for a batch."""
        self._need_bands(band_idx, "band_idx")
        return self._flux(jnp.asarray(theta), jnp.asarray(times), band_idx)


class _PredictJaxBands:
    """``Model.predict_jax`` of a band-bound model: a picklable object that carries ``.band_index``.

    The samplers read ``predict_jax.band_index`` to map the data's band labels once, outside the
    trace; a bound method would hand them the attributes of the class's shared function instead.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.band_index = ctx.band_index
        self.__doc__ = type(ctx).predict_jax.__doc__

    def __call__(self, theta, times, band_idx=None):
        return self.ctx.predict_jax(theta, times, band_idx)


# --- factories ------------------------------------------------------------------------------------

def phenomenological_models():
    """The phenomenological models :func:`phenomenological_model` builds, by redback's name.

    Examples
    --------
    >>> from whisper_cbpf.models.jax.phenomenological import phenomenological_models
    >>> phenomenological_models()
    ['smooth_exponential_powerlaw', 'exp_rise_powerlaw_decline', 'bazin_sne', 'villar_sne']
    """
    return list(_spec.MODELS)


def phenomenological_model(model, band_names=None, *, name=None, prior=None, band_aliases=None):
    """A whisper ``Model`` for one phenomenological function, with a JAX half.

    ``predict(parameters, times, bands)`` returns flux density in Jy as NumPy;
    ``predict_jax(theta, times, band_idx)`` is the traced twin. Nothing is registered: use
    :func:`register_phenomenological` for that.

    Parameters
    ----------
    model : str
        One of :func:`phenomenological_models`.
    band_names : sequence of str, optional
        The bands the model is bound to. Each gets its own amplitude (``aa_<band>``, ``bb_<band>``,
        ``cc_<band>``, ``a_1_<band>``, ``m_peak_<band>``); the shape parameters are shared, as in
        redback. ``None`` (default) makes the model band-independent: one amplitude, redback's
        parameter names, the band ignored.
    name : str, optional
        The registry name; default ``"<model>_jax"``.
    prior : Prior or dict, optional
        Distributions that replace the default ones. A key is a parameter name or, for a per-band
        argument, redback's name alone, which applies to every band. The default amplitude prior
        is redback's unit-free ``LogUniform(1e-15, 1e15)``, far wider than any flux in Jy.
    band_aliases : dict, optional
        Your labels onto ``band_names``, e.g. ``{"g": "ztfg"}``.

    Returns
    -------
    whisper_cbpf.models.Model

    References
    ----------
    Bazin et al. 2009, https://ui.adsabs.harvard.edu/abs/2009A%26A...499..653B/abstract
    (``bazin_sne``); Villar et al. 2019, https://ui.adsabs.harvard.edu/abs/2019ApJ...884...83V/abstract
    (``villar_sne``); Arnett 1982, https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract.
    redback cites no reference for ``smooth_exponential_powerlaw`` or ``exp_rise_powerlaw_decline``.

    Examples
    --------
    >>> import numpy as np
    >>> from whisper_cbpf.models.jax.phenomenological import phenomenological_model
    >>> m = phenomenological_model("bazin_sne", ["g", "r"])
    >>> m.name, m.parameters
    ('bazin_sne_jax', ['aa_g', 'aa_r', 'bb_g', 'bb_r', 't0', 'tau_rise', 'tau_fall'])
    >>> p = dict(aa_g=2e-4, aa_r=3e-4, bb_g=0.0, bb_r=0.0, t0=0.0, tau_rise=3.0, tau_fall=25.0)
    >>> flux = m.predict(p, np.array([5.0, 5.0, 30.0]), np.array(["g", "r", "r"]))
    >>> flux.shape, bool(flux[1] > flux[0] > 0)
    ((3,), True)
    """
    from ...models import Model
    from ._factories import _BandIndex

    if model not in SPECS:
        raise ValueError(f"unknown phenomenological model {model!r}; "
                         f"available: {phenomenological_models()}")
    name = f"{model}_jax" if name is None else str(name)
    params = parameter_names(model, band_names)
    index = None if band_names is None else _BandIndex(
        band_names, f"phenomenological_model({model!r}, band_names=[...])", band_aliases)
    if band_names is None and band_aliases:
        raise ValueError("band_aliases needs band_names")
    ctx = _PhenomPredict(model, name, params, index)
    # band-independent: a bound method with no .band_index, like the flare's
    predict_jax = ctx.predict_jax if index is None else _PredictJaxBands(ctx)
    return Model(name=name, predict=ctx.predict, parameters=params,
                 default_prior=default_prior(model, band_names, prior),
                 description=(f"{model} (redback), JAX; {SPECS[model]['unit']}. "
                              + ("One amplitude per band, shared shape."
                                 if band_names is not None else "Band-independent.")),
                 predict_jax=predict_jax)


def register_phenomenological(model, band_names=None, *, name=None, **kwargs):
    """Build a phenomenological ``Model`` (see :func:`phenomenological_model`) and register it.

    Replaces an earlier registration under the same name, so binding the same model to another
    band list does the obvious thing.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.jax.phenomenological import register_phenomenological
    >>> m = register_phenomenological("villar_sne", ["g", "r"], name="villar_gr")
    >>> wp.get_model("villar_gr") is not None
    True
    """
    from ._factories import _register
    return _register(phenomenological_model(model, band_names, name=name, **kwargs))
