"""``ABC_time_prior``: narrow the explosion-time window with a rejection ABC run on the GPU.

A light curve from an alert stream is on an MJD clock, so every model fits its explosion or
merger time ``t_exp``. The default prior of that time is a 30-day window
(:func:`~whisper_cbpf.default_t_exp_prior`). With ``ABC_time_prior=True``, :func:`whisper_cbpf.fit`
and :func:`whisper_cbpf.compare` first run whisper's rejection ABC on the GPU (``abc_gpu``) with
the model's JAX version on the whole light curve (every detection, every band), and fit inside
the range of ``t_exp`` the ABC accepted.

The rule is fixed:

- every parameter is drawn from the model's prior, and ``t_exp`` from the default window;
- the simulation is noiseless (``simulate_noise=False``), and the distance is the RMS of
  ``data - model`` over the detections, in flux space (whisper's ``rmse`` distance);
- a draw is accepted when that RMS is at most :data:`N_SIGMA` (3) times the mean flux error of
  the detections;
- the new prior is a flat ``Uniform`` from the lowest to the highest accepted ``t_exp``, widened
  on each side by the margin ``max(MARGIN_DAYS, MARGIN_FRACTION * range)`` and cut to the default
  window (never wider than it, never after the first detection);
- with fewer than :data:`MIN_ACCEPTED` accepted draws it warns and keeps the default window.

The data are used twice: the window is chosen from the same light curve it is then fitted to, so
the credible intervals and the evidences of the fit are mildly data-dependent.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

#: A draw is accepted when the RMS of ``data - model`` (flux, over the detections) is at most
#: this many times the mean flux error of the detections.
N_SIGMA = 3.0

#: Fewer accepted draws than this and the step keeps the default window (with a warning).
MIN_ACCEPTED = 50

#: The margin added on each side of the accepted ``t_exp`` range: at least this many days ...
MARGIN_DAYS = 1.0

#: ... or this fraction of the accepted range, whichever is larger.
MARGIN_FRACTION = 0.1

#: The ABC budget ``ABC_time_prior_kwargs`` can change, and its defaults. ``seed=None`` takes the
#: fit's seed. One million draws take 11-42 s on one RTX A6000 (float64) for the five JAX TDE
#: models on a 35-epoch, 5-band light curve; 200 000 took 4-18 s but kept the default window
#: (fewer than 50 accepted) in 53 of 80 runs, against 18 of 40 with one million.
DEFAULTS = {"n_simulations": 1_000_000, "seed": None, "min_accepted": MIN_ACCEPTED,
            "chunk": None, "precision": None}


def abc_time_prior(lc, model, prior=None, *, n_simulations=1_000_000, seed=0,
                   min_accepted=MIN_ACCEPTED, chunk=None, precision=None):
    """The explosion-time prior ``ABC_time_prior=True`` gives: a flat window from a GPU ABC run.

    Every parameter of ``model`` is drawn from its prior (``prior``, else the model's default)
    and ``t_exp`` from the default window of a fit (:func:`~whisper_cbpf.default_t_exp_prior`,
    cut to the model's own ``Uniform`` ``t_exp`` prior, as a fit cuts it). Each draw is simulated
    without noise by the model's JAX version on every detection of ``lc`` (all bands) with
    whisper's rejection ABC on the GPU (``abc_gpu``), and accepted when the RMS of
    ``data - model`` in flux is at most :data:`N_SIGMA` (3) times the mean flux error of the
    detections. The window returned is the range of the accepted ``t_exp``, widened on each side
    by ``max(MARGIN_DAYS, MARGIN_FRACTION * range)`` (1 d or 10 %), and cut to the default window.

    The data are used twice: the window is chosen from the light curve the fit then uses, so a
    fit's credible intervals and evidence are mildly data-dependent.

    Parameters
    ----------
    lc : LightCurve
        The data. Upper limits set the default window (its anchor) and are not simulated.
    model : str or Model
        A model with a ``t_exp`` parameter and a JAX version (``predict_jax``).
    prior : Prior, optional
        The prior of the other parameters. Default ``model.default_prior``. It must not name
        ``t_exp``: the window is what this function sets.
    n_simulations : int, default 1000000
        Prior draws simulated.
    seed : int, default 0
        Seed of the draws. The same seed gives the same window on the same device.
    min_accepted : int, default 50
        Fewer accepted draws than this: warn and return the default window.
    chunk : int, optional
        Draws per compiled block (``abc_gpu``'s ``chunk``; default its own).
    precision : {None, "float32", "float64"}, optional
        ``abc_gpu``'s ``precision``. ``None`` follows the session.

    Returns
    -------
    window : Uniform
        The ``t_exp`` prior, named ``"t_exp"``.
    record : dict
        What was done: ``window``, ``default_window``, ``accepted_range`` (``None`` on a fall
        back), ``margin_days``, ``n_simulations``, ``n_accepted``, ``min_accepted``,
        ``threshold`` (flux, Jy), ``n_sigma``, ``mean_flux_err``, ``min_rms``, ``n_detections``,
        ``runtime_s``, ``fell_back``, ``seed``, ``device``, ``precision``, ``sampler``,
        ``distance``, ``warnings`` (the first line of each warning of the ABC fit).

    Raises
    ------
    ValueError
        The model has no ``t_exp`` parameter, no JAX version (a CPU-only model such as TiDE),
        ``prior`` names ``t_exp``, or ``n_simulations`` or ``min_accepted`` is not a whole
        number of at least 1.

    Warns
    -----
    UserWarning
        Fewer than ``min_accepted`` draws were accepted: the default window is returned.

    Examples
    --------
    A rise of known shape whose first point is 4 days after the explosion: the ABC window is
    narrower than the default 30 days and holds the truth.

    >>> import numpy as np, jax.numpy as jnp, whisper_cbpf as wp
    >>> def rise(p, t, bands=None):
    ...     dt = np.clip(np.asarray(t, float) - p["t_exp"], 0.0, None)
    ...     return p["amp"] * 1e-4 * (1 - np.exp(-dt / 3.0)) * np.exp(-dt / 20.0)
    >>> def rise_jax(theta, t):
    ...     dt = jnp.clip(jnp.asarray(t) - theta[1], 0.0, None)
    ...     return theta[0] * 1e-4 * (1 - jnp.exp(-dt / 3.0)) * jnp.exp(-dt / 20.0)
    >>> model = wp.Model(name="rise", predict=rise, parameters=["amp", "t_exp"],
    ...                  predict_jax=rise_jax,
    ...                  default_prior=wp.Prior({"amp": wp.Uniform(0.5, 2.0),
    ...                                          "t_exp": wp.Uniform(59900.0, 60100.0)}))
    >>> t = 60000.0 + np.arange(0.0, 40.0, 2.0)
    >>> lc = wp.LightCurve(time=t, band=["r"] * t.size,
    ...                    flux=rise({"amp": 1.0, "t_exp": 59996.0}, t),
    ...                    flux_err=np.full(t.size, 2e-6))
    >>> window, record = wp.abc_time_prior(lc, model, n_simulations=5000)
    >>> record["default_window"], record["fell_back"], record["n_accepted"] >= 50
    ([59970.0, 60000.0], False, True)
    >>> bool(window.low < 59996.0 < window.high), bool(window.high - window.low < 30.0)
    (True, True)
    """
    from ..io import schema
    from ..likelihood import GaussianLikelihood
    from ..models import get_model
    from ..priors import Prior, Uniform
    from .base import _t_exp_prior, _upper_limits, _warn_user, _zero_flux_given

    start = time.perf_counter()
    n_simulations = _whole(n_simulations, "n_simulations")
    min_accepted = _whole(min_accepted, "min_accepted")
    model = get_model(model)
    _check_model(model)
    if prior is not None and "t_exp" in prior.distributions:
        raise ValueError(_conflict_message(model.name))
    base = prior if prior is not None else model.default_prior
    if base is None:
        raise ValueError(f"ABC_time_prior: model {model.name!r} has no default prior; pass "
                         f"prior= (without t_exp) for its other parameters.")
    window, _ = _t_exp_prior(lc, model, base, None)        # the default window of a fit
    if not isinstance(window, Uniform):                    # a model's Normal t_exp prior, say
        window = schema.default_t_exp_prior(lc)
    default = (float(window.low), float(window.high))

    det = lc[~_upper_limits(lc)]
    sigma = np.asarray(GaussianLikelihood(det, space="flux").sigma, dtype=float)
    mean_err = float(np.mean(sigma))
    threshold = N_SIGMA * mean_err
    abc_prior = Prior({**base.distributions, "t_exp": Uniform(*default, name="t_exp")})

    from .jax.abc_gpu import DEFAULT_CHUNK, ABCGPUSampler
    with warnings.catch_warnings(record=True) as caught:
        # The ABC fit's own warnings (an empty acceptance, its WAIC on draws that are not a
        # posterior) are kept in the record, not shown: this step says what matters below.
        warnings.simplefilter("always")
        with _zero_flux_given():                           # the fit gives that warning
            res = ABCGPUSampler().fit(
                det, model, prior=abc_prior, n_simulations=int(n_simulations),
                threshold=threshold, distance="rmse", simulate_noise=False, space="flux",
                seed=int(seed), chunk=DEFAULT_CHUNK if chunk is None else int(chunk),
                precision=precision)
    accepted = np.asarray(res.samples["t_exp"], dtype=float)
    n_acc = int(accepted.size)
    fell_back = n_acc < int(min_accepted)
    if fell_back:
        low, high, acc_range, margin = default[0], default[1], None, None
        closest = (f"the closest draw has {res.min_distance * 1e6:.3g} uJy"
                   if np.isfinite(res.min_distance) else "no draw gave a finite RMS")
        _warn_user(
            f"ABC_time_prior: only {n_acc:,} of {int(n_simulations):,} draws fit within "
            f"{N_SIGMA:g} x the mean error (RMS needs <= {threshold * 1e6:.3g} uJy, {closest}; "
            f"at least {int(min_accepted)} are needed); using the default t_exp window "
            f"[{default[0]:.10g}, {default[1]:.10g}].")
    else:
        acc_range = [float(accepted.min()), float(accepted.max())]
        margin = window_margin(acc_range[1] - acc_range[0])
        low = max(default[0], acc_range[0] - margin)
        high = min(default[1], acc_range[1] + margin)
    record = {
        "window": [float(low), float(high)], "default_window": [default[0], default[1]],
        "accepted_range": acc_range, "margin_days": margin,
        "n_simulations": int(n_simulations), "n_accepted": n_acc,
        "min_accepted": int(min_accepted), "threshold": float(threshold),
        "n_sigma": float(N_SIGMA), "mean_flux_err": mean_err,
        "min_rms": float(res.min_distance), "n_detections": int(len(det)),
        "runtime_s": float(time.perf_counter() - start), "fell_back": bool(fell_back),
        "seed": int(seed), "device": res.info.get("device"),
        "precision": res.info.get("precision"), "sampler": "abc_gpu", "distance": "rmse",
        "warnings": [str(w.message).splitlines()[0] for w in caught],
    }
    return Uniform(float(low), float(high), name="t_exp"), record


def window_margin(accepted_range_days):
    """The margin added on each side of the accepted ``t_exp`` range, in days.

    ``max(MARGIN_DAYS, MARGIN_FRACTION * range)``: 1 d, or 10 % of the range when that is larger.

    Parameters
    ----------
    accepted_range_days : float
        Highest minus lowest accepted ``t_exp``.

    Returns
    -------
    float

    Examples
    --------
    >>> from whisper_cbpf.samplers.abc_window import window_margin
    >>> window_margin(4.0), window_margin(25.0)
    (1.0, 2.5)
    """
    return float(max(MARGIN_DAYS, MARGIN_FRACTION * float(accepted_range_days)))


def _whole(value, name):
    """``value`` as an ``int`` >= 1, or a ValueError naming ``name`` and the value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if isinstance(value, bool) or not (np.isfinite(number) and number >= 1
                                       and number == int(number)):
        raise ValueError(f"ABC_time_prior: {name} must be a whole number of at least 1; got "
                         f"{value!r}.")
    return int(number)


def _check_model(model):
    """ValueError unless ``model`` fits ``t_exp`` and has a JAX version."""
    from .base import _has_t_exp

    if not _has_t_exp(model):
        raise ValueError(f"ABC_time_prior: model {model.name!r} has no t_exp parameter, so there "
                         f"is no explosion time to estimate. Fit without ABC_time_prior.")
    if getattr(model, "predict_jax", None) is None:
        raise ValueError(f"ABC_time_prior: model {model.name!r} has no JAX version "
                         f"(predict_jax), and the ABC step runs the JAX model on the GPU. Fit it "
                         f"without ABC_time_prior (it then uses the default t_exp window), or "
                         f"pass prior= with a t_exp prior of your own.")


def _conflict_message(name):
    return (f"ABC_time_prior=True sets the t_exp prior of {name!r} from the data, but the prior "
            f"passed also names t_exp. Pass one or the other: drop t_exp from prior=, or leave "
            f"ABC_time_prior=False to use your own t_exp prior.")


def _options(kwargs, seed):
    """``ABC_time_prior_kwargs`` checked and completed: ``seed=None`` takes the fit's ``seed``."""
    kwargs = dict(kwargs or {})
    stray = sorted(set(kwargs) - set(DEFAULTS))
    if stray:
        raise TypeError(f"ABC_time_prior_kwargs takes {sorted(DEFAULTS)}; got {stray}.")
    out = {**DEFAULTS, **kwargs}
    if out["seed"] is None:
        out["seed"] = 0 if seed is None else int(seed)
    return out


def _fit_prior(lc, model, prior, options):
    """``(prior for the fit, record)``: ``prior`` (or the model's) with the ABC ``t_exp`` window."""
    from ..models import get_model
    from ..priors import Prior

    model = get_model(model)
    window, record = abc_time_prior(lc, model, prior, **options)
    base = prior if prior is not None else model.default_prior
    return Prior({**base.distributions, "t_exp": window}), record


def _stamp(result, record):
    """Keep ``record`` in ``result.info["abc_time_prior"]``; name the source of ``t_exp_prior``."""
    if not isinstance(getattr(result, "info", None), dict):
        return
    result.info["abc_time_prior"] = dict(record)
    if isinstance(result.info.get("t_exp_prior"), dict):
        result.info["t_exp_prior"]["source"] = "ABC_time_prior"


def _check_flag(value):
    if value not in (True, False):
        raise ValueError(f"ABC_time_prior must be True or False; got {value!r}.")
    return bool(value)
