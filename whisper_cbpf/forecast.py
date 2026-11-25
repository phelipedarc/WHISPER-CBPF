"""Forecasts in magnitudes from a fit's posterior, and where two models differ most.

:func:`forecast` answers "what will this transient look like at these times, in these bands?" for
one fitted model: it pushes posterior draws through the model and summarises the predicted AB
magnitude of every (time, band) cell by its mean, spread, percentiles and the fraction of draws
too faint to be seen. :func:`discriminate` compares those forecasts between models and scores each
cell by

    D = |mean_a - mean_b| / sqrt(sd_a**2 + sd_b**2 + sd_phot**2),

the gap between two models' predictions in units of their combined uncertainty (the posterior
spread of each model plus the photometric error of the measurement), so the best next observation
is the observable cell with the largest D.

A model with a JAX half (``Model.predict_jax``) is evaluated in one batched, compiled call over all
draws and cells, on the GPU when JAX has one: 400 draws x 15 cells of the Arnett model take about a
millisecond after a compile of about a second, against seconds for the same draws through redback on
the CPU. Other models are called once per draw through ``Model.predict``. The two paths give the
same magnitudes; ``tests/test_forecast.py`` checks the JAX Arnett model against redback's at the
same draws.

Draws that predict no light at a cell (zero or non-finite flux, a JAX model's ``mag_floor``, or
fainter than :data:`DARK_MAG`) are "dark": they are counted in ``frac_dark`` and left out of the
mean, spread and percentiles, which therefore describe the transient *when it shines*. With
``survey_depth`` the ``frac_too_faint`` column counts the draws fainter than the depth, dark ones
included.
"""
from __future__ import annotations

import itertools
import math
import time as _time
import warnings
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .io.photometry import AB_ZEROPOINT_JY
from .models import get_model

__all__ = ["forecast", "discriminate", "DARK_MAG", "SIGMA_AT_LIMIT", "DEFAULT_QUANTILES",
           "OBSERVABLE_MAX_FAINT", "CLOCK_TOLERANCE_DAYS"]

#: A predicted magnitude at or beyond this is "no light" (far below any survey's limit).
DARK_MAG = 35.0
#: Magnitude error of a detection at the survey's 5-sigma depth: 2.5 / ln(10) / 5.
SIGMA_AT_LIMIT = 2.5 / math.log(10.0) / 5.0
#: Percentiles reported by :func:`forecast`: the 95 % and 68 % central intervals and the median.
DEFAULT_QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)
#: :func:`discriminate` calls a cell observable when both models' mean magnitudes are brighter than
#: the depth and less than this fraction of each model's draws is too faint.
OBSERVABLE_MAX_FAINT = 0.5
#: Forecast times further than this from the light curve's span are refused as another clock.
CLOCK_TOLERANCE_DAYS = 1.0e4
#: Largest number of draws evaluated in one compiled JAX call; more are evaluated in blocks of this
#: size (the last one padded), so memory stays bounded and there is still one compile.
MAX_JAX_BLOCK = 1024
#: Compiled batched forward maps kept for reuse, keyed by model, epochs and bands.
_JAX_CACHE_SIZE = 32
_JAX_CACHE = OrderedDict()


# --- inputs ----------------------------------------------------------------------------------------

def _model_of(result, model):
    """The model to forecast with: ``model`` if given, else the one the fit names."""
    if model is not None:
        return get_model(model)
    name = getattr(result, "model", None)
    if name is None:
        raise ValueError("forecast needs a model: this result names none. Pass model=<the fitted "
                         "Model or its registered name>.")
    obj = getattr(result, "_model_object", None)    # the fit's own Model (see fitted_model)
    if obj is not None:
        return obj
    try:
        return get_model(name)
    except KeyError:
        raise ValueError(
            f"forecast needs the fitted model, and {name!r} is not registered in this session. "
            f"Pass model=<the Model the fit used> (for a wp.compare result: "
            f"comparison.forecast(...), or model=comparison.models[{name!r}]), or build it again "
            f"with the call that made it (e.g. wp.register_supernova(...)).") from None


def _columns(samples, model):
    """Sample columns in ``model.parameters`` order, through ``model.param_aliases`` if needed."""
    aliases = dict(getattr(model, "param_aliases", None) or {})
    cols, missing = [], []
    for p in model.parameters:
        if p in samples.columns:
            cols.append(p)
        elif aliases.get(p) in samples.columns:
            cols.append(aliases[p])
        else:
            missing.append(p)
    if missing:
        raise ValueError(
            f"the posterior has no column for {missing} of model {model.name!r} (columns: "
            f"{list(samples.columns)}). Forecast with the model the fit sampled (model=), or rename "
            f"the columns, e.g. result.samples.rename(columns={{...}}).")
    return cols


def _draws(result, model, n_draws, seed):
    """``(theta (n, k) float64, n_total)``: up to ``n_draws`` posterior rows, without replacement."""
    samples = getattr(result, "samples", None)
    if not isinstance(samples, pd.DataFrame):
        raise TypeError(f"forecast takes a fit result with a .samples DataFrame (a SamplerResult); "
                        f"got {type(result).__name__}.")
    n_total = int(len(samples))
    if n_total == 0:
        raise ValueError(
            f"not enough data: the fit of {getattr(result, 'model', model.name)!r} has no posterior "
            f"draws (for ABC: no draw was accepted), so there is nothing to forecast. Refit with "
            f"more simulations, a looser threshold, or another sampler.")
    if int(n_draws) < 1:
        raise ValueError(f"n_draws must be at least 1; got {n_draws}.")
    theta = samples[_columns(samples, model)].to_numpy(dtype=float)
    if n_total > int(n_draws):
        idx = np.sort(np.random.default_rng(seed).choice(n_total, int(n_draws), replace=False))
        theta = theta[idx]
    return theta, n_total


def _times(times):
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if t.ndim != 1 or t.size == 0:
        raise ValueError(f"times must be a non-empty 1-D sequence of epochs; got shape {t.shape}.")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite; NaN or inf found.")
    return t


def _bands(bands):
    if isinstance(bands, (str, np.str_)):
        bands = [bands]
    try:
        out = [str(b) for b in bands]
    except TypeError:
        raise TypeError(f"bands must be a band label or a sequence of labels; got "
                        f"{type(bands).__name__}.") from None
    if not out:
        raise ValueError("bands is empty; pass at least one band label, e.g. ['lsstg', 'lsstr'].")
    return list(dict.fromkeys(out))              # unique, first-seen order


def _depths(survey_depth, bands):
    """``{band: depth or nan}`` from ``None``, one number, or ``{band: number}``."""
    if survey_depth is None:
        return {b: float("nan") for b in bands}
    if isinstance(survey_depth, Mapping):
        missing = [b for b in bands if b not in survey_depth]
        if missing:
            raise ValueError(f"survey_depth has no depth for {missing}; it covers "
                             f"{list(survey_depth)}. Give every forecast band a 5-sigma depth "
                             f"(AB mag), or one number for all.")
        out = {b: float(survey_depth[b]) for b in bands}
    else:
        try:
            d = float(survey_depth)
        except (TypeError, ValueError):
            raise TypeError(f"survey_depth must be None, a magnitude, or {{band: magnitude}}; got "
                            f"{type(survey_depth).__name__}.") from None
        out = {b: d for b in bands}
    bad = [b for b, d in out.items() if not np.isfinite(d)]
    if bad:
        raise ValueError(f"survey_depth for {bad} is not a finite magnitude.")
    return out


def _quantile_name(q):
    return f"q{round(100.0 * float(q), 6):g}"


def _check_quantiles(quantiles):
    qs = [float(q) for q in np.atleast_1d(quantiles)]
    bad = [q for q in qs if not 0.0 < q < 1.0]
    if bad:
        raise ValueError(f"quantiles must lie strictly between 0 and 1 (e.g. 0.16, 0.84); got "
                         f"{bad}.")
    return qs


def _check_clock(times, lc):
    """Refuse forecast times on another clock than the light curve's (MJD against days since)."""
    t_lc = np.asarray(lc.time, dtype=float)
    if t_lc.size == 0:
        return
    lo, hi = float(np.min(t_lc)), float(np.max(t_lc))
    gap = float(np.max(np.abs(times - np.clip(times, lo, hi))))
    if gap <= CLOCK_TOLERANCE_DAYS:
        return
    ref = lc.meta.get("time_reference_mjd")
    hint = (f" The light curve counts days since MJD {ref:.3f}: subtract it, times - {ref:.3f}."
            if ref is not None and np.median(times) > 3.0e4 else "")
    raise ValueError(
        f"the forecast times ({times.min():.6g} to {times.max():.6g}) are on another clock than "
        f"the light curve ({lo:.6g} to {hi:.6g}): they lie {gap:.3g} days from its span. Pass the "
        f"times on the light curve's own clock, e.g. lc.time.max() + np.array([0.25, 1.0, 3.0])."
        + hint)


# --- the model, over many draws ----------------------------------------------------------------

def _flux_to_mag(flux):
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux / AB_ZEROPOINT_JY)
    return np.where(flux > 0, mag, np.inf)


def _jax_forward(model, t, b, n_cols):
    """A compiled ``theta (B, k) -> flux (B, n)`` for these concrete epochs and bands, cached."""
    from .backends import require_jax
    jax, jnp = require_jax("a batched JAX forecast")

    pj = model.predict_jax
    index_of = getattr(pj, "band_index", None)
    bidx = None if index_of is None else np.asarray(index_of(b), dtype=int)
    key = (id(pj), t.tobytes(), None if bidx is None else bidx.tobytes(), int(n_cols),
           bool(jax.config.jax_enable_x64))
    hit = _JAX_CACHE.get(key)
    if hit is not None:
        _JAX_CACHE.move_to_end(key)
        return hit[1]
    ok_fn = getattr(pj, "constraint_ok", None)

    def one(theta):
        # concrete float64 epochs: a model subtracts its t_exp on the host, and the supernova
        # family builds its grid from them (it cannot take traced times)
        flux = pj(theta, t) if bidx is None else pj(theta, t, bidx)
        return flux if ok_fn is None else jnp.where(ok_fn(theta), flux, 0.0)

    fn = jax.jit(jax.vmap(one))
    _JAX_CACHE[key] = (pj, fn)                   # pj kept alive, so its id is not reused
    while len(_JAX_CACHE) > _JAX_CACHE_SIZE:
        _JAX_CACHE.popitem(last=False)
    return fn


def _jax_flux(model, theta, t, b):
    from .backends import require_jax
    jax, jnp = require_jax("a batched JAX forecast")

    fn = _jax_forward(model, t, b, theta.shape[1])
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    n = theta.shape[0]
    width = min(n, MAX_JAX_BLOCK)
    out = []
    for start in range(0, n, width):
        block = theta[start:start + width]
        pad = width - block.shape[0]
        if pad:                                  # same shape every block: one compile
            block = np.concatenate([block, np.repeat(block[-1:], pad, axis=0)])
        out.append(np.asarray(fn(jnp.asarray(block, dtype=dtype)), dtype=float)[:width - pad])
    return np.concatenate(out, axis=0)


def _numpy_flux(model, theta, t, b):
    names = list(model.parameters)
    out = np.empty((theta.shape[0], t.size), dtype=float)
    for i, row in enumerate(theta):
        out[i] = np.asarray(model.predict(dict(zip(names, map(float, row))), t, b), dtype=float)
    return out


def _mag_floor(model):
    floor = getattr(getattr(model, "predict_jax", None), "mag_floor", None)
    return None if floor is None else float(floor)


def _draw_magnitudes(model, theta, times, bands):
    """``(mags (n_draws, n_points), dark (same shape), backend)`` at paired ``times`` / ``bands``.

    The points are evaluated in time order (redback sizes some grids by the last epoch it is given)
    and returned in the caller's order.
    """
    t = np.asarray(times, dtype=float)
    b = np.asarray([str(x) for x in bands])
    order = np.argsort(t, kind="stable")
    ts, bs = np.ascontiguousarray(t[order]), b[order]
    if getattr(model, "predict_jax", None) is not None:
        flux, backend = _jax_flux(model, theta, ts, bs), "jax"
    else:
        flux, backend = _numpy_flux(model, theta, ts, bs), "numpy"
    if flux.shape != (theta.shape[0], t.size):
        raise ValueError(f"model {model.name!r} returned {flux.shape[1:]} fluxes for {t.size} "
                         f"epochs; a model must return one flux density per epoch.")
    mag = np.empty_like(flux)
    mag[:, order] = _flux_to_mag(flux)
    dark = ~np.isfinite(mag) | (mag >= DARK_MAG)
    floor = _mag_floor(model)
    if floor is not None:
        dark |= mag >= floor - 1e-6
    return mag, dark, backend


# --- forecast ---------------------------------------------------------------------------------------

def _time_label(lc):
    if lc is None:
        return None
    from .plotting import _time_label as label
    return label(lc)


def forecast(result, times, bands, *, lc=None, model=None, n_draws=400,
             quantiles=DEFAULT_QUANTILES, survey_depth=None, seed=0):
    """Predicted AB magnitudes at future (time, band) cells, from a fit's posterior draws.

    Every combination of ``times`` and ``bands`` is a cell. Up to ``n_draws`` posterior draws are
    pushed through the model; each cell reports the mean, spread and percentiles of the predicted
    magnitude over the draws that predict light there, and the fractions of draws that predict no
    light or light too faint for the survey. A model with ``predict_jax`` is evaluated in one
    compiled batch (on the GPU when JAX has one); any other model once per draw on the CPU.

    Parameters
    ----------
    result : SamplerResult
        The fit. Its ``samples`` are the draws; its ``model`` names the model.
    times : array_like
        Epochs on the fit's clock, the light curve's ``time`` (days since its reference, or MJD).
    bands : str or sequence of str
        Band labels the model was built for, e.g. ``["lsstg", "lsstr"]``.
    lc : LightCurve, optional
        The fitted light curve. When given, times on another clock (MJD against days since a
        reference) are refused, and the time-axis label is kept for :func:`plot_forecast`.
    model : str or Model, optional
        The model to evaluate; by default the one ``result.model`` names. A JAX twin of the fitted
        model reads the draws through its ``param_aliases``.
    n_draws : int, default 400
        Posterior draws used, taken without replacement (all of them if there are fewer).
    quantiles : sequence of float, default (0.025, 0.16, 0.5, 0.84, 0.975)
        Percentiles reported, as columns ``q2.5``, ``q16``, ``q50``, ``q84``, ``q97.5``.
    survey_depth : float or dict, optional
        The 5-sigma limiting magnitude, one for all bands or ``{band: mag}``. Sets
        ``frac_too_faint`` and the ``depth`` column.
    seed : int, default 0
        Chooses the draws.

    Returns
    -------
    pandas.DataFrame
        One row per cell, time-major: ``time``, ``band``, ``mean_mag``, ``sd_mag``, one column per
        quantile, ``frac_too_faint`` (draws fainter than ``depth``, or predicting no light when no
        depth is given), ``frac_dark`` (draws predicting no light: zero or non-finite flux, a JAX
        model's ``mag_floor``, or fainter than :data:`DARK_MAG`), ``n_draws`` (draws that predict
        light, behind the mean) and ``depth``. The statistics are over the draws that predict light;
        with fewer than two such draws they are NaN ("not enough data"), not a number. ``.attrs``
        holds ``model``, ``sampler``, ``n_draws`` (draws used), ``n_samples`` (posterior size),
        ``seed``, ``quantiles``, ``survey_depth``, ``time_label``, ``backend`` ("jax" or "numpy"),
        ``eval_s`` and ``dark_mag``.

    Raises
    ------
    ValueError
        The fit has no draws ("not enough data"); the model is not registered; a model parameter has
        no posterior column; ``times`` are empty, non-finite or on another clock than ``lc``;
        ``survey_depth`` lacks a band; a quantile is outside (0, 1).

    See Also
    --------
    discriminate : where two models' forecasts differ most.
    whisper_cbpf.plotting.plot_forecast : the forecast as a figure.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.forecast import forecast
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
    >>> fc = forecast(res, [31.0, 33.0, 36.0], "r", lc=lc, survey_depth=10.0)
    >>> fc.shape
    (3, 13)
    >>> list(fc.columns[:6])
    ['time', 'band', 'mean_mag', 'sd_mag', 'q2.5', 'q16']
    >>> bool((fc["mean_mag"].diff().dropna() > 0).all())    # fading after the peak
    True
    """
    m = _model_of(result, model)
    t = _times(times)
    bs = _bands(bands)
    qs = _check_quantiles(quantiles)
    depth = _depths(survey_depth, bs)
    if lc is not None:
        _check_clock(t, lc)
    theta, n_total = _draws(result, m, n_draws, seed)

    cell_t = np.repeat(t, len(bs))
    cell_b = np.tile(np.asarray(bs, dtype=object), t.size)
    start = _time.perf_counter()
    mag, dark, backend = _draw_magnitudes(m, theta, cell_t, cell_b)
    eval_s = _time.perf_counter() - start

    lit = np.where(dark, np.nan, mag)
    n_lit = (~dark).sum(axis=0)
    enough = n_lit >= 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns -> NaN
        mean = np.where(enough, np.nanmean(lit, axis=0), np.nan)
        sd = np.where(enough, np.nanstd(lit, axis=0, ddof=1), np.nan)
        qv = np.nanquantile(lit, qs, axis=0) if lit.size else np.full((len(qs), 0), np.nan)
    qv = np.where(enough[None, :], qv, np.nan)
    d_cell = np.array([depth[b] for b in cell_b], dtype=float)
    faint = dark | (np.isfinite(d_cell)[None, :] & (mag > d_cell[None, :]))

    df = pd.DataFrame({"time": cell_t, "band": cell_b.astype(str), "mean_mag": mean,
                       "sd_mag": sd})
    for q, row in zip(qs, qv):
        df[_quantile_name(q)] = row
    df["frac_too_faint"] = faint.mean(axis=0)
    df["frac_dark"] = dark.mean(axis=0)
    df["n_draws"] = n_lit.astype(int)
    df["depth"] = d_cell
    if not enough.all():
        warnings.warn(
            f"not enough data at {int((~enough).sum())} of {enough.size} cells: fewer than two of "
            f"the {theta.shape[0]} draws of {m.name!r} predict light there, so their mean, spread "
            f"and percentiles are NaN (see frac_dark).", UserWarning, stacklevel=2)
    df.attrs.update(
        model=m.name, sampler=str(getattr(result, "sampler", "")), n_draws=int(theta.shape[0]),
        n_samples=n_total, seed=int(seed), quantiles=qs,
        survey_depth=None if survey_depth is None else {b: depth[b] for b in bs},
        time_label=_time_label(lc), backend=backend, eval_s=float(eval_s), dark_mag=DARK_MAG,
        lc_name=None if lc is None else lc.name)
    return df


# --- discriminate ----------------------------------------------------------------------------------

def _phot_sigmas(phot_sigma, bands):
    if phot_sigma is None:
        return None
    if isinstance(phot_sigma, Mapping):
        missing = [b for b in bands if b not in phot_sigma]
        if missing:
            raise ValueError(f"phot_sigma has no value for {missing}; give one number for all "
                             f"bands or {{band: sigma}} covering {bands}.")
        out = {b: float(phot_sigma[b]) for b in bands}
    else:
        out = {b: float(phot_sigma) for b in bands}
    if any(not (np.isfinite(v) and v >= 0) for v in out.values()):
        raise ValueError(f"phot_sigma must be finite and >= 0 mag; got {out}.")
    return out


def _forecast_of(name, value, times, bands, survey_depth):
    """A model's forecast: computed from a result, or a forecast DataFrame checked for the cells."""
    if isinstance(value, pd.DataFrame):
        cells = list(zip(np.repeat(times, len(bands)), np.tile(bands, len(times))))
        got = list(zip(value["time"].to_numpy(float), value["band"].astype(str)))
        if len(got) != len(cells) or any(abs(a[0] - b[0]) > 1e-9 or a[1] != b[1]
                                         for a, b in zip(got, cells)):
            raise ValueError(f"the forecast given for {name!r} has other cells than times x "
                             f"bands; recompute it with forecast(result, times, bands).")
        have = value.attrs.get("survey_depth")
        if survey_depth is not None and have != _depths(survey_depth, bands):
            raise ValueError(f"the forecast given for {name!r} was made with survey_depth="
                             f"{have}; recompute it with survey_depth={survey_depth!r}.")
        return value
    return forecast(value, times, bands, survey_depth=survey_depth)


def discriminate(results, times, bands, *, survey_depth=None, phot_sigma=None):
    """Where two models' forecasts differ most, cell by cell, in units of their joint uncertainty.

    For every pair of models and every (time, band) cell,
    ``D = |mean_a - mean_b| / sqrt(sd_a**2 + sd_b**2 + sd_phot**2)``, from each model's
    :func:`forecast` (the mean and spread of its predicted magnitude). ``D`` near 1 or below: one
    measurement there cannot tell the two models apart; ``D`` of 3 or more: it can, if the cell is
    observable. The most discriminating observable cell is ``.attrs["best"]``.

    Parameters
    ----------
    results : dict
        ``{name: SamplerResult}`` (each forecast with its own model, 400 draws, seed 0), or
        ``{name: forecast DataFrame}`` for forecasts already made over the same cells. A model whose
        fit has no posterior draw is left out, with its reason in ``.attrs["left_out"]``.
    times : array_like
        Epochs on the fits' clock.
    bands : str or sequence of str
        Band labels.
    survey_depth : float or dict, optional
        5-sigma limiting magnitude, one for all bands or ``{band: mag}``. A cell is observable when
        both mean magnitudes are brighter than it and fewer than half of each model's draws are
        fainter (:data:`OBSERVABLE_MAX_FAINT`). Without it, observable means that most draws of both
        models predict light.
    phot_sigma : float or dict, optional
        The photometric error of one measurement, in mag. By default it follows the depth for a
        background-limited survey, :data:`SIGMA_AT_LIMIT` (0.217 mag at the 5-sigma depth) times
        ``10**(0.4 * (m - depth))`` at the pair's mean magnitude ``m``; zero with no depth.

    Returns
    -------
    pandas.DataFrame
        One row per pair and cell: ``model_a``, ``model_b``, ``time``, ``band``, ``mean_a``,
        ``mean_b``, ``sd_a``, ``sd_b``, ``sd_phot``, ``D`` and ``observable``. ``D`` is NaN where a
        model has too few lit draws or no spread at all ("not enough data"). ``.attrs``: ``best``
        (the observable row with the largest D, as a dict, or None), ``reason`` (why ``best`` is
        None), ``left_out`` ({model: reason}), ``survey_depth``, ``phot_sigma``.

    Raises
    ------
    ValueError
        Fewer than two models, or fewer than two with posterior draws; a forecast DataFrame with
        other cells or another depth; ``phot_sigma`` negative or missing a band.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.forecast import discriminate
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> fits = {m: wp.fit(lc, m, sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
    ...         for m in ("flare", "bazin")}
    >>> d = discriminate(fits, [35.0, 45.0, 60.0], "r", survey_depth=12.0)
    >>> list(d.columns)
    ['model_a', 'model_b', 'time', 'band', 'mean_a', 'mean_b', 'sd_a', 'sd_b', 'sd_phot', 'D', 'observable']
    >>> sorted(d.attrs["best"])[:3]
    ['D', 'band', 'mean_a']
    """
    if not isinstance(results, Mapping):
        raise TypeError("discriminate takes {model name: SamplerResult}, e.g. comparison.results.")
    if len(results) < 2:
        raise ValueError(f"discriminate needs at least two models to compare; got "
                         f"{len(results)} ({list(results)}).")
    t = _times(times)
    bs = _bands(bands)
    depth = _depths(survey_depth, bs)
    sig = _phot_sigmas(phot_sigma, bs)

    fcs, left_out = {}, {}
    for name, value in results.items():
        if not isinstance(value, pd.DataFrame) and getattr(value, "n_samples", None) == 0:
            left_out[str(name)] = "no posterior draws"
            continue
        fcs[str(name)] = _forecast_of(name, value, t, bs, survey_depth)
    if len(fcs) < 2:
        raise ValueError(f"not enough data: fewer than two models have posterior draws to "
                         f"forecast (left out: {left_out}).")

    rows = []
    for a, b in itertools.combinations(fcs, 2):
        fa, fb = fcs[a], fcs[b]
        ma, mb = fa["mean_mag"].to_numpy(float), fb["mean_mag"].to_numpy(float)
        sa, sb = fa["sd_mag"].to_numpy(float), fb["sd_mag"].to_numpy(float)
        band = fa["band"].astype(str).to_numpy()
        d_cell = np.array([depth[x] for x in band])
        m_pair = 0.5 * (ma + mb)
        if sig is not None:
            sp = np.array([sig[x] for x in band])
        elif survey_depth is not None:
            sp = SIGMA_AT_LIMIT * 10.0 ** (0.4 * (m_pair - d_cell))
        else:
            sp = np.zeros_like(ma)
        den = np.sqrt(sa ** 2 + sb ** 2 + sp ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.where(den > 0, np.abs(ma - mb) / den, np.nan)
        lit = np.isfinite(ma) & np.isfinite(mb)
        if survey_depth is not None:
            obs = (lit & (ma < d_cell) & (mb < d_cell)
                   & (fa["frac_too_faint"].to_numpy(float) < OBSERVABLE_MAX_FAINT)
                   & (fb["frac_too_faint"].to_numpy(float) < OBSERVABLE_MAX_FAINT))
        else:
            obs = (lit & (fa["frac_dark"].to_numpy(float) < OBSERVABLE_MAX_FAINT)
                   & (fb["frac_dark"].to_numpy(float) < OBSERVABLE_MAX_FAINT))
        rows.append(pd.DataFrame({
            "model_a": a, "model_b": b, "time": fa["time"].to_numpy(float), "band": band,
            "mean_a": ma, "mean_b": mb, "sd_a": sa, "sd_b": sb, "sd_phot": sp, "D": D,
            "observable": obs.astype(bool)}))
    out = pd.concat(rows, ignore_index=True)

    ok = out["observable"].to_numpy(bool) & np.isfinite(out["D"].to_numpy(float))
    best = reason = None
    if ok.any():
        i = int(np.flatnonzero(ok)[np.argmax(out["D"].to_numpy(float)[ok])])
        best = {k: (v.item() if hasattr(v, "item") else v) for k, v in out.iloc[i].items()}
    elif not out["observable"].any():
        reason = ("no cell where both models of a pair are observable"
                  + (" (brighter than the survey depth)" if survey_depth is not None else
                     " (most draws predict light)"))
    else:
        reason = "not enough data: no observable cell has a finite D (too few lit draws, no spread)"
    out.attrs.update(best=best, reason=reason, left_out=left_out,
                     survey_depth=None if survey_depth is None else depth,
                     phot_sigma=sig)
    return out
