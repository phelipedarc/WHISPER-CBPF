"""Light-curve report plots.

``plot_light_curve`` renders a transient's photometry with consistent, explainable styling:

* each band gets a distinct color;
* **detections** (SNR >= 3) are circles with a black edge;
* **low-SNR** points (SNR < 3) are up-triangles;
* **upper limits** are down-triangles;
* magnitude axes are inverted (brighter = up).

Two layouts:

* ``layout='report'`` -- one figure, two stacked panels (apparent magnitude vs time and flux density
  vs time), all selected bands overlaid.
* ``layout='grid'`` -- one panel per band, with the y-axis chosen by ``quantity``
  (``'apparent_mag'`` | ``'absolute_mag'`` (needs redshift) | ``'flux'``).
"""
from __future__ import annotations

import shutil
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

_MARKER = {"det": "o", "lowsnr": "^", "ul": "v"}
_SNR_LOW = 3.0


def _safe_usetex():
    """Disable matplotlib's LaTeX text rendering when no ``latex`` executable is available.

    Some optional backends (notably **redback**) enable ``text.usetex`` globally on import, which then
    makes every subsequent plot crash with ``RuntimeError: ... latex could not be found`` on systems
    without a TeX install. Called at the start of each plot function, this resets the flag only when
    LaTeX is genuinely missing — a user with a working TeX keeps ``usetex=True`` untouched.
    """
    if matplotlib.rcParams.get("text.usetex") and shutil.which("latex") is None:
        matplotlib.rcParams["text.usetex"] = False


def _band_colors(bands):
    cmap = plt.get_cmap("turbo")
    n = max(len(bands), 1)
    return {b: cmap((i + 0.5) / n) for i, b in enumerate(bands)}


def _categories(lc):
    """Classify each point as 'det', 'lowsnr', or 'ul'."""
    n = lc.n_points
    ul = lc.upper_limit if lc.upper_limit is not None else np.zeros(n, dtype=bool)
    try:
        snr = lc.snr
    except ValueError:
        snr = np.full(n, np.inf)
    return np.where(ul, "ul", np.where(snr < _SNR_LOW, "lowsnr", "det")).astype(object)


def _time_label(lc):
    """Time-axis label naming the day 0 the curve declares (``LightCurve.set_time_reference``).

    Any shifted curve used to read "days since explosion", including one whose day 0 is the first
    detection. A curve with only ``explosion_mjd`` (saved before ``time_reference`` existed) still
    reads as the explosion.
    """
    ref = lc.meta.get("time_reference")
    if ref is None and "explosion_mjd" in lc.meta:
        ref = "explosion"
    if ref is None:
        return "time [MJD]"
    mjd = lc.meta.get("time_reference_mjd", lc.meta.get("explosion_mjd"))
    return f"days since {ref}" + ("" if mjd is None else f" (MJD {mjd:.3f})")


def _scatter(ax, time, y, yerr, cat, color, label=None):
    """Plot one band's points, split by marker category."""
    labeled = False
    for c in ("det", "lowsnr", "ul"):
        m = cat == c
        if not m.any():
            continue
        lab = label if (label and not labeled) else None
        style = dict(color=color, marker=_MARKER[c], linestyle="none",
                     markeredgecolor="black", markeredgewidth=0.6, markersize=6, label=lab)
        if c == "ul" or yerr is None:
            ax.plot(time[m], y[m], **style)
        else:
            ax.errorbar(time[m], y[m], yerr=yerr[m], ecolor=color, elinewidth=0.8,
                        capsize=0, **style)
        labeled = labeled or (lab is not None)


def _quantity(full, quantity):
    """Return (y, yerr, ylabel, invert_axis) for the requested quantity."""
    q = quantity.lower()
    if q in ("flux", "flux_density"):
        return full.flux, full.flux_err, "flux density [Jy]", False
    if q in ("absolute_mag", "absolute", "abs_mag"):
        if full.redshift is None:
            raise ValueError("quantity='absolute_mag' requires the light curve's redshift.")
        from astropy.cosmology import Planck18
        mu = float(Planck18.distmod(full.redshift).value)
        return full.magnitude - mu, full.magnitude_err, "absolute magnitude (AB)", True
    return full.magnitude, full.magnitude_err, "apparent magnitude (AB)", True


def plot_light_curve(lc, *, layout="report", quantity="apparent_mag", bands=None,
                     ncols=3, figsize=None, title=None, save=None):
    """Report plot of a light curve. Returns the ``Axes`` array.

    See the module docstring for layouts, the ``quantity`` options, and the marker conventions.
    ``layout='report'`` returns the 2 panels (magnitude, flux); ``layout='grid'`` returns the
    ``nrows x ncols`` grid (unused cells hidden). The figure is ``np.ravel(axes)[0].figure``; it is
    left open in pyplot, so it appears exactly once in a notebook and ``plt.show()`` /
    ``plt.savefig`` still work.

    Parameters
    ----------
    lc : LightCurve
    layout : {"report", "grid"}, default "report"
    quantity : str, default "apparent_mag"
        What the grid layout shows (see the module docstring).
    bands : str or list of str, optional
        Plot only these bands.
    ncols : int, default 3
        Columns of the grid layout.
    figsize : tuple, optional
    title : str, optional
    save : str or Path, optional
        Also write the figure there.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=[1.0, 2.0, 4.0, 6.0], band=["ztfg", "ztfr", "ztfg", "ztfr"],
    ...                    magnitude=[19.0, 18.8, 19.2, 19.1], magnitude_err=[0.1] * 4)
    >>> axes = wp.plot_light_curve(lc)
    >>> len(axes), axes[0].get_ylabel()
    (2, 'apparent magnitude (AB)')
    >>> plt.close("all")
    """
    _safe_usetex()
    if bands is not None:
        lc = lc.select_bands(bands)
    full = lc.add_flux().add_mag()      # ensure both magnitude and flux are available
    band_list = full.bands
    colors = _band_colors(band_list)
    cat = _categories(full)

    if layout == "report":
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize or (9, 8))
        ax_m, ax_f = axes
        for b in band_list:
            m = full.band == b
            merr = None if full.magnitude_err is None else full.magnitude_err
            ferr = None if full.flux_err is None else full.flux_err
            _scatter(ax_m, full.time[m], full.magnitude[m],
                     None if merr is None else merr[m], cat[m], colors[b], label=b)
            _scatter(ax_f, full.time[m], full.flux[m],
                     None if ferr is None else ferr[m], cat[m], colors[b])
        ax_m.invert_yaxis()
        ax_m.set_ylabel("apparent magnitude (AB)")
        ax_f.set_ylabel("flux density [Jy]")
        ax_f.set_xlabel(_time_label(full))
        for ax in (ax_m, ax_f):
            ax.grid(alpha=0.3)
        ax_m.legend(ncol=4, fontsize=8, loc="best")

    elif layout == "grid":
        y, yerr, ylabel, invert = _quantity(full, quantity)
        nb = len(band_list)
        ncols = min(ncols, nb) or 1
        nrows = int(np.ceil(nb / ncols))
        fig, axes = plt.subplots(nrows, ncols, sharex=True,
                                 figsize=figsize or (4 * ncols, 2.6 * nrows), squeeze=False)
        axflat = axes.ravel()
        for ax, b in zip(axflat, band_list):
            m = full.band == b
            _scatter(ax, full.time[m], y[m], None if yerr is None else yerr[m],
                     cat[m], colors[b])
            if invert:
                ax.invert_yaxis()
            ax.set_title(b, fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axflat[nb:]:
            ax.set_visible(False)
        fig.supxlabel(_time_label(full))
        fig.supylabel(ylabel)

    else:
        raise ValueError(f"Unknown layout {layout!r} (use 'report' or 'grid').")

    fig.suptitle(title or (full.name or "light curve"))
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return axes


# --- posterior-predictive check --------------------------------------------------------------------

from .io.photometry import AB_ZEROPOINT_JY as _AB_ZP_JY


def _flux_to_quantity(flux, quantity):
    """Map model flux density [Jy] to the plotted quantity."""
    if quantity in ("flux", "flux_density"):
        return np.asarray(flux, float)
    return -2.5 * np.log10(np.clip(np.asarray(flux, float), 1e-300, None) / _AB_ZP_JY)   # AB mag


def _check_intervals(intervals):
    """Central credible levels in percent, widest first; each strictly between 0 and 100."""
    levels = sorted({float(i) for i in np.atleast_1d(intervals)}, reverse=True)
    bad = [i for i in levels if not 0.0 < i < 100.0]
    if bad or not levels:
        raise ValueError(f"intervals are central credible levels in percent, each strictly between "
                         f"0 and 100, e.g. (68, 95); got {intervals!r}.")
    return levels


def _ppc_curves(result, model, tgrid, band_list, quantity, n_draws, seed, intervals=(95.0,)):
    """Posterior-predictive curves per band over ``tgrid``: ``(median, [(lo, hi) per interval])``,
    the intervals in the order given."""
    from .forecast import _model_of

    m = _model_of(result, model)                # names model= when the fit's model is unknown here
    samples = result.samples
    cols = [c for c in m.parameters if c in samples.columns]
    draws = samples[cols].to_numpy(dtype=float)
    idx = np.random.default_rng(seed).choice(
        len(draws), size=min(n_draws, len(draws)), replace=len(draws) < n_draws)
    pct = [50.0] + [p for i in intervals for p in (50.0 - i / 2.0, 50.0 + i / 2.0)]
    curves = {}
    for b in band_list:
        gb = np.array([b] * len(tgrid))
        stack = np.empty((len(idx), len(tgrid)), dtype=float)
        for i, j in enumerate(idx):
            p = {c: float(draws[j, k]) for k, c in enumerate(cols)}
            stack[i] = _flux_to_quantity(np.asarray(m.predict(p, tgrid, gb), float), quantity)
        q = np.nanpercentile(stack, pct, axis=0)
        curves[b] = (q[0], [(q[1 + 2 * k], q[2 + 2 * k]) for k in range(len(intervals))])
    return curves


def plot_ppc(results, lc, model=None, *, quantity="apparent_mag", panel_by="auto", n_draws=200,
             bands=None, tmin=None, tmax=None, ncols=None, colors=None, figsize=None, title=None,
             seed=0, save=None, intervals=(68, 95)):
    """Posterior-predictive check: model band(s) over the data, in a **grid** of panels.

    For each fit, draws ``n_draws`` posterior samples, evaluates the model on a smooth time grid, and
    shades the **68 % and 95 % posterior-predictive bands** (the 16-84 and 2.5-97.5 percentiles;
    ``intervals=``) with the median curve, over the observed photometry — per band. Two panel
    layouts:

    * ``panel_by='method'`` — one panel per fit, all bands overlaid (band-coloured); this is the
      multi-sampler grid (as used for the AT2017GFO study).
    * ``panel_by='band'`` — one panel per band, all fits overlaid (fit-coloured).

    ``'auto'`` picks ``'method'`` when several fits are given, else ``'band'``.

    Parameters
    ----------
    results : SamplerResult | dict[str, SamplerResult] | list[SamplerResult]
        One or more fits. A dict keys the panels/legend by label; a list uses each result's sampler name.
    lc : LightCurve
        The observed data (its ``time`` / ``band`` grid and the plotted photometry).
    model : str | Model, optional
        Forward model; defaults to each result's ``.model``.
    quantity : str
        ``'apparent_mag'`` (default, inverted axis) or ``'flux'`` (flux density [Jy]).
    panel_by : str
        ``'auto'`` | ``'method'`` | ``'band'``.
    n_draws : int
        Posterior draws per fit used to build the predictive band.
    bands : list, optional
        Restrict to these bands (default: all bands in ``lc``, blue→red order preserved).
    tmin, tmax : float, optional
        Time-axis limits (data units); the predictive grid spans this range.
    ncols : int, optional
        Grid columns (default: near-square).
    colors : dict, optional
        Override colours — by band (``panel_by='method'``) or by fit label (``panel_by='band'``).
    intervals : sequence of float, default (68, 95)
        Central credible levels, in percent, shaded around the median (darker for the narrower).

    Returns the 2-D ``nrows x ncols`` ``Axes`` array (unused cells hidden). The figure is
    ``np.ravel(axes)[0].figure``; it is left open in pyplot, so it appears exactly once in a
    notebook.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
    >>> axes = wp.plot_ppc(res, lc)                       # 68 % and 95 % bands, magnitudes
    >>> len(axes[0, 0].collections) >= 2
    True
    """
    if quantity.lower() in ("flux", "flux_density"):
        quantity, invert = "flux", False
    else:
        quantity, invert = "apparent_mag", True
    levels = _check_intervals(intervals)

    _safe_usetex()
    if hasattr(results, "samples"):                            # a single SamplerResult
        fits = {getattr(results, "sampler", "fit"): results}
    elif isinstance(results, dict):
        fits = dict(results)
    else:
        fits = {getattr(r, "sampler", f"fit{i}"): r for i, r in enumerate(results)}
    if panel_by == "auto":
        panel_by = "method" if len(fits) > 1 else "band"

    full = lc.add_flux().add_mag() if bands is None else \
        lc.select_bands(bands).add_flux().add_mag()
    band_list = list(bands) if bands is not None else full.bands
    t = np.asarray(full.time, float)
    obs_band = np.asarray(full.band).astype(str)
    obs_y = np.asarray(full.flux if quantity == "flux" else full.magnitude, float)
    obs_e = np.asarray((full.flux_err if quantity == "flux" else full.magnitude_err), float)
    lo_t = float(np.min(t)) if tmin is None else float(tmin)
    hi_t = float(np.max(t)) if tmax is None else float(tmax)
    tgrid = np.linspace(max(lo_t, 1e-3), hi_t, 120)

    curves = {lbl: _ppc_curves(r, model, tgrid, band_list, quantity, n_draws, seed, levels)
              for lbl, r in fits.items()}
    # widest first, lightest; each narrower level is drawn over it, darker
    alphas = [0.14 + 0.16 * k for k in range(len(levels))]
    band_col = _band_colors(band_list) if colors is None or panel_by == "method" else None
    if panel_by == "method" and colors is not None:
        band_col = {**band_col, **colors}
    fit_col = colors if (colors is not None and panel_by == "band") else \
        {lbl: CORNER_PALETTE[i % len(CORNER_PALETTE)] for i, lbl in enumerate(fits)}

    panels = list(fits) if panel_by == "method" else band_list
    ncols = ncols or int(np.ceil(np.sqrt(len(panels))))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4.6 * ncols, 3.6 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axflat = axes.ravel()
    ylabel = "flux density [Jy]" if quantity == "flux" else "apparent magnitude (AB)"

    for ax, panel in zip(axflat, panels):
        if panel_by == "method":
            for b in band_list:
                med, bands_lo_hi = curves[panel][b]
                for (lo, hi), a in zip(bands_lo_hi, alphas):
                    ax.fill_between(tgrid, lo, hi, color=band_col[b], alpha=a, lw=0)
                ax.plot(tgrid, med, color=band_col[b], lw=1.8, label=b)
                sel = obs_band == b
                ax.errorbar(t[sel], obs_y[sel], yerr=None if obs_e is None else obs_e[sel],
                            fmt="o", ms=5, mfc=band_col[b], mec="black", mew=0.5,
                            ecolor="black", elinewidth=0.7, alpha=0.9, zorder=5)
            ax.set_title(panel, fontsize=13, weight="bold")
        else:                                                  # panel_by == 'band'
            b = panel
            for lbl, r in fits.items():
                med, bands_lo_hi = curves[lbl][b]
                for (lo, hi), a in zip(bands_lo_hi, alphas):
                    ax.fill_between(tgrid, lo, hi, color=fit_col[lbl], alpha=0.8 * a, lw=0)
                ax.plot(tgrid, med, color=fit_col[lbl], lw=1.8, label=lbl)
            sel = obs_band == b
            ax.errorbar(t[sel], obs_y[sel], yerr=None if obs_e is None else obs_e[sel],
                        fmt="o", ms=5, mfc="0.2", mec="black", mew=0.5, ecolor="0.4",
                        elinewidth=0.7, alpha=0.9, zorder=5)
            ax.set_title(b, fontsize=13, weight="bold")
        ax.grid(alpha=0.25)
    if invert and len(panels):
        axflat[0].invert_yaxis()               # sharey=True -> inverting one inverts all (invert once)
    for ax in axflat[len(panels):]:
        ax.set_visible(False)
    axflat[0].legend(fontsize=9, frameon=True,
                     title=("band" if panel_by == "method" else "fit")
                     + f" (median; {' and '.join(f'{i:g} %' for i in levels[::-1])} bands)")
    fig.supxlabel(_time_label(full))
    fig.supylabel(ylabel)
    fig.suptitle(title or f"Posterior-predictive check — {full.name or lc.name or 'light curve'}",
                 weight="bold")
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return axes


# --- coverage-calibration curve --------------------------------------------------------------------

def plot_calibration(results, lc, model=None, *, levels=(0.5, 0.68, 0.8, 0.9, 0.95, 0.99),
                     space="auto", per_band=False, n_draws=400, colors=None, figsize=None,
                     title=None, seed=0, save=None):
    """Coverage-calibration curve (reliability / pp-plot) for one or more fits.

    For each nominal credible level, plots the **empirical** posterior-predictive coverage — the
    fraction of observations inside the central predictive interval (model + observation noise) — against
    the nominal level. A well-calibrated fit lies on the diagonal; below ⇒ over-confident (intervals too
    narrow), above ⇒ under-confident. Coverage is computed by
    :func:`whisper_cbpf.metrics.predictive_metrics`.

    ``results`` is a `SamplerResult`, a `{label: result}` dict, or a list. With ``per_band=True`` a
    single fit is broken out into one line per band. Returns the ``Axes``; its figure (``ax.figure``) is
    left open in pyplot, so it appears exactly once in a notebook.

    Parameters
    ----------
    results : SamplerResult, dict or list
    lc : LightCurve
        The data the fits were run on.
    model : str or Model, optional
        Default: each fit's own model.
    levels : sequence of float
        Nominal credible levels.
    space : str, default "auto"
    per_band : bool, default False
    n_draws : int, default 400
    colors, figsize, title, save, seed
        Presentation, and the seed of the predictive draws.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t) + np.random.default_rng(1).normal(0, 0.2, 30)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.2))
    >>> res = wp.fit(lc, "flare", sampler="nested", nlive=100, seed=0)
    >>> ax = wp.plot_calibration(res, lc, n_draws=100)
    >>> bool(ax.get_xlim()[1] <= 1.05)
    True
    >>> plt.close("all")
    """
    from .metrics import predictive_metrics

    _safe_usetex()
    if hasattr(results, "samples"):
        fits = {getattr(results, "sampler", "fit"): results}
    elif isinstance(results, dict):
        fits = dict(results)
    else:
        fits = {getattr(r, "sampler", f"fit{i}"): r for i, r in enumerate(results)}

    fig, ax = plt.subplots(figsize=figsize or (5.4, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1, label="perfect calibration")
    palette = colors or CORNER_PALETTE

    def _line(cov_list, label, color, **kw):
        nominal = [c["nominal"] for c in cov_list]
        empirical = [c["empirical"] for c in cov_list]
        ax.plot(nominal, empirical, marker="o", ms=5, color=color, label=label, **kw)

    if per_band and len(fits) == 1:
        (lbl, r), = fits.items()
        pm = predictive_metrics(r, lc, model=model, space=space, levels=levels, n_draws=n_draws, seed=seed)
        _line(pm["coverage"]["overall"], "overall", "black", lw=2.2)
        band_items = list(pm["coverage"]["bands"].items())
        for i, (b, cov) in enumerate(band_items):
            _line(cov, b, palette[i % len(palette)], lw=1.4, alpha=0.85)
    else:
        for i, (lbl, r) in enumerate(fits.items()):
            pm = predictive_metrics(r, lc, model=model, space=space, levels=levels, n_draws=n_draws,
                                    seed=seed)
            _line(pm["coverage"]["overall"], lbl, palette[i % len(palette)], lw=2.0)

    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(title or f"Coverage calibration — {getattr(lc, 'name', None) or 'light curve'}",
                 weight="bold")
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return ax


# --- corner plot -----------------------------------------------------------------------------------

#: Dark, distinct, print-friendly palette for overlaying posteriors (dark blue, dark red, dark green,
#: deep purple, dark orange, dark slate). Saturated/dark and well-separated in hue so the contours and
#: marginals stay legible when several posteriors are overlaid.
CORNER_PALETTE = ["#08306b", "#a50026", "#006d2c", "#54278f", "#993404", "#252525"]


def _posterior_to_frame(p, parameters):
    """Coerce one posterior (SamplerResult / DataFrame / dict / 2-D array) to (DataFrame, label)."""
    import pandas as pd

    if hasattr(p, "samples") and hasattr(p, "sampler"):        # SamplerResult
        return p.samples, str(p.sampler)
    if isinstance(p, pd.DataFrame):
        return p, None
    if isinstance(p, dict):
        return pd.DataFrame(p), None
    arr = np.asarray(p, dtype=float)
    if parameters is None or arr.ndim != 2 or arr.shape[1] != len(parameters):
        raise ValueError("array posteriors need a matching `parameters` list (one name per column).")
    return pd.DataFrame(arr, columns=list(parameters)), None


def _prior_from_record(rec):
    """A :class:`~whisper_cbpf.priors.Prior` rebuilt from ``provenance["model"]["prior"]``, or None."""
    from .priors import Fixed, LogUniform, Normal, Prior, TruncatedNormal, Uniform

    params = (rec or {}).get("parameters") if isinstance(rec, dict) else None
    if not params:
        return None
    build = {"Uniform": lambda r: Uniform(r["low"], r["high"]),
             "LogUniform": lambda r: LogUniform(r["low"], r["high"]),
             "Normal": lambda r: Normal(r["mu"], r["sigma"]),
             "TruncatedNormal": lambda r: TruncatedNormal(r["mu"], r["sigma"], r["low"], r["high"]),
             "Fixed": lambda r: Fixed(r["value"])}
    dists = {}
    for name, r in params.items():
        try:
            dists[name] = build[r["type"]](r)
        except (KeyError, TypeError, ValueError):
            return None                          # a family this reader does not know: no guess
    return Prior(dists)


def _fit_prior(result, prior=None):
    """The prior of a fit: ``prior`` if given, else the one recorded with it, else its model's."""
    if prior is not None:
        return prior
    prov = getattr(result, "provenance", None)
    rec = ((prov.get("model") or {}).get("prior")) if isinstance(prov, dict) else None
    recorded = _prior_from_record(rec)
    if recorded is not None:
        return recorded
    from .samplers.base import fitted_model
    try:
        return fitted_model(result).default_prior
    except (KeyError, TypeError):
        return None


def _log_uniform_names(prior):
    return {n for n, d in getattr(prior, "distributions", {}).items()
            if type(d).__name__ == "LogUniform"}


def plot_corner(posteriors, *, labels=None, parameters=None, colors=None, truths=None,
                bins=30, levels=(0.39, 0.86), smooth=1.0, log_params="auto", title=None,
                legend_loc="upper right", save=None, **corner_kwargs):
    """Overlay one or more posteriors on a single publication-ready corner plot.

    A thin, well-styled wrapper over :mod:`corner` for comparing posteriors (e.g. several samplers on
    the same data): shared per-parameter ranges so the panels align, a dark distinct colour per
    posterior, contour lines (not filled) so overlaps stay readable, and a legend.

    Parameters
    ----------
    posteriors : sequence
        Each item is a :class:`~whisper_cbpf.samplers.base.SamplerResult`, a ``pandas.DataFrame`` of
        samples, a ``{name: array}`` dict, or a 2-D array (then pass ``parameters`` for the columns).
    labels : sequence of str, optional
        Legend label per posterior (defaults to each ``SamplerResult``'s sampler name, else
        ``"posterior i"``).
    parameters : sequence of str, optional
        Parameters (columns) to plot, in order. Defaults to the columns common to every posterior.
    colors : sequence, optional
        One colour per posterior; defaults to :data:`CORNER_PALETTE`.
    truths : dict | sequence, optional
        Reference values drawn once as solid black lines (a dict is keyed by parameter name).
    bins, smooth : int, float
        Histogram bins and Gaussian contour smoothing.
    levels : tuple
        2-D enclosed-probability contour levels (default ``(0.39, 0.86)`` ≈ 1σ/2σ in 2-D).
    log_params : "auto" or sequence of str, default "auto"
        Parameters to display on a ``log10`` axis (samples are log10-transformed; the label reads
        ``log10 <name>``). ``"auto"``: those with a LogUniform prior in any ``SamplerResult`` given
        (the prior recorded with the fit, else its model's default), none for plain sample tables;
        ``None`` or ``[]``: none.
    title : str, optional
        Figure suptitle.
    save : str, optional
        If given, save the figure there (PNG, 200 dpi).

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        The ``K x K`` corner axes in :mod:`corner`'s row-major layout (``axes[i, j]`` is row ``i``,
        column ``j``; the other triangle is blank). The figure is ``np.ravel(axes)[0].figure``;
        it is left open in pyplot, so it appears exactly once in a notebook.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> a = pd.DataFrame({"mej": 10 ** rng.normal(-2, 0.3, 500), "vej": rng.normal(0.2, 0.02, 500)})
    >>> b = pd.DataFrame({"mej": 10 ** rng.normal(-1.8, 0.3, 500), "vej": rng.normal(0.22, 0.02, 500)})
    >>> axes = wp.plot_corner([a, b], labels=["ABC", "MCMC"], log_params=["mej"])
    >>> axes.shape
    (2, 2)
    >>> print(axes[1, 0].get_xlabel())
    $\\log_{10}\\,$mej

    With fits (``SamplerResult``), ``log_params="auto"`` puts every LogUniform-prior parameter on a
    log axis: ``wp.plot_corner([res_abc, res_mcmc], labels=["ABC", "MCMC"])``.
    """
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _safe_usetex()
    if not len(posteriors):
        raise ValueError("plot_corner needs at least one posterior.")
    frames, auto_labels = [], []
    for i, p in enumerate(posteriors):
        frame, lab = _posterior_to_frame(p, parameters)
        frames.append(frame)
        auto_labels.append(lab or f"posterior {i + 1}")
    labels = list(labels) if labels is not None else auto_labels

    if parameters is None:
        parameters = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
        if not parameters:
            raise ValueError("posteriors share no common parameter columns; pass parameters=...")
    if isinstance(log_params, str) and log_params == "auto":
        log_params = set()
        for p in posteriors:
            if hasattr(p, "samples") and hasattr(p, "sampler"):
                log_params |= _log_uniform_names(_fit_prior(p))
    elif isinstance(log_params, str):
        raise ValueError(f"log_params is 'auto', None or a list of parameter names; got "
                         f"{log_params!r}.")
    log_params = set(log_params or []) & set(parameters)
    disp = [(r"$\log_{10}\,$" + p) if p in log_params else p for p in parameters]

    def _array(frame):
        return np.column_stack([
            np.log10(np.asarray(frame[p], float)) if p in log_params else np.asarray(frame[p], float)
            for p in parameters])

    samples = [_array(f) for f in frames]
    union = np.vstack(samples)
    rng = []
    for j in range(union.shape[1]):
        lo, hi = np.percentile(union[:, j], 0.5), np.percentile(union[:, j], 99.5)
        rng.append((lo, hi) if hi > lo else (lo - 0.5, hi + 0.5))

    if colors is None:
        colors = [CORNER_PALETTE[i % len(CORNER_PALETTE)] for i in range(len(samples))]
    if isinstance(truths, dict):
        truths = [(np.log10(truths[p]) if p in log_params else truths[p]) if p in truths else None
                  for p in parameters]

    base = dict(bins=bins, smooth=smooth, range=rng, plot_datapoints=False, plot_density=False,
                fill_contours=False, levels=levels,
                label_kwargs=dict(fontsize=14, fontweight="bold"))
    base.update(corner_kwargs)
    fig = None
    for i, X in enumerate(samples):
        fig = corner.corner(
            X, fig=fig, color=colors[i], labels=disp,
            hist_kwargs=dict(density=True, color=colors[i], lw=1.8,
                             histtype="stepfilled", alpha=0.30),
            contour_kwargs=dict(colors=colors[i], linewidths=2.0),
            truths=truths if i == 0 else None, truth_color="0.1",
            truth_kwargs=dict(lw=1.4, ls="--"), **base)
    for ax in fig.get_axes():
        ax.tick_params(labelsize=11)
    fig.legend(handles=[Line2D([], [], color=c, lw=2.6, label=l) for c, l in zip(colors, labels)],
               loc=legend_loc, frameon=True, fontsize=13, title="posterior", title_fontsize=13)
    if title:
        fig.suptitle(title, y=1.02, fontsize=16, weight="bold")
    if save is not None:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return np.asarray(fig.get_axes(), dtype=object).reshape(len(parameters), len(parameters))


# --- forecasts, models over the data, the ranking, posterior widths ---------------------------------

_MAG_LABEL = "apparent magnitude (AB)"
_NO_CLOCK_LABEL = "time (the fit's clock)"


def _invert_y(ax):
    """Brighter up, once: an Axes handed in already inverted stays inverted."""
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def _interval_columns(df, level):
    """The quantile columns of a central ``level`` % interval in a forecast table, or None."""
    from .forecast import _quantile_name

    lo, hi = _quantile_name((50.0 - level / 2.0) / 100.0), _quantile_name((50.0 + level / 2.0) / 100.0)
    return (lo, hi) if lo in df.columns and hi in df.columns else None


def plot_forecast(forecast_df, ax=None):
    """A forecast in magnitudes: median, 68 % and 95 % bands per band, and the survey depth.

    Parameters
    ----------
    forecast_df : pandas.DataFrame
        What :func:`whisper_cbpf.forecast.forecast` returns (one model; from a
        ``Comparison.forecast`` table, select one model's rows). The 68 % band needs its ``q16``
        and ``q84`` columns, the 95 % band ``q2.5`` and ``q97.5`` (the default quantiles); without
        ``q50`` the line is the mean.
    ax : matplotlib.axes.Axes, optional
        Draw here; by default a new figure.

    Returns
    -------
    matplotlib.axes.Axes
        Magnitude against time, brighter up, the time axis labelled with the light curve's day 0
        when the forecast was made with ``lc=``. A band's line joins its forecast times; a single
        time is drawn as error bars. Dashed lines are the 5-sigma depths; open markers are cells
        where at least half of the draws are too faint. The figure is left open in pyplot, so it
        appears exactly once in a notebook.

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
    >>> fc = forecast(res, np.linspace(31.0, 60.0, 30), "r", lc=lc, survey_depth=10.5)
    >>> from whisper_cbpf.plotting import plot_forecast
    >>> ax = plot_forecast(fc)
    >>> ax.get_ylabel(), bool(ax.yaxis_inverted())
    ('apparent magnitude (AB)', True)
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    missing = {"time", "band", "mean_mag"} - set(getattr(forecast_df, "columns", []))
    if missing:
        raise ValueError(f"plot_forecast takes the table forecast() returns; this one has no "
                         f"{sorted(missing)} column.")
    df = forecast_df
    names = list(dict.fromkeys(df["model"].astype(str))) if "model" in df.columns else []
    if len(names) > 1:
        raise ValueError(f"this table holds the forecasts of {len(names)} models {names} (a "
                         f"Comparison.forecast table); plot one at a time, e.g. "
                         f"plot_forecast(fc[fc['model'] == {names[0]!r}]).")
    _safe_usetex()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 4.8))
    bands = list(dict.fromkeys(df["band"].astype(str)))
    colors = _band_colors(bands)
    alpha = {95.0: 0.16, 68.0: 0.34}
    shown, depth_shown, faint_shown = set(), False, False
    for b in bands:
        d = df[df["band"].astype(str) == b].sort_values("time", kind="stable")
        t = d["time"].to_numpy(float)
        centre = (d["q50"] if "q50" in d.columns else d["mean_mag"]).to_numpy(float)
        c = colors[b]
        for level in (95.0, 68.0):
            cols = _interval_columns(d, level)
            if cols is None:
                continue
            lo, hi = d[cols[0]].to_numpy(float), d[cols[1]].to_numpy(float)
            if len(d) > 1:
                ax.fill_between(t, lo, hi, color=c, alpha=alpha[level], lw=0)
            else:
                ax.errorbar(t, centre, yerr=[centre - lo, hi - centre], fmt="none", ecolor=c,
                            elinewidth=2.0 if level == 95.0 else 5.0, alpha=0.5, capsize=0)
            shown.add(level)
        ax.plot(t, centre, "-o", color=c, lw=1.8, ms=4, label=b)
        if "frac_too_faint" in d.columns:
            faint = d["frac_too_faint"].to_numpy(float) >= 0.5
            if faint.any():
                ax.plot(t[faint], centre[faint], "o", mfc="white", mec=c, mew=1.4, ms=7, zorder=4)
                faint_shown = True
        if "depth" in d.columns:
            depth = d["depth"].to_numpy(float)
            if np.isfinite(depth).any():
                ax.axhline(float(np.nanmedian(depth)), color=c, ls="--", lw=1.1, alpha=0.9)
                depth_shown = True
    _invert_y(ax)
    ax.set_xlabel(df.attrs.get("time_label") or _NO_CLOCK_LABEL)
    ax.set_ylabel(_MAG_LABEL)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    for level in sorted(shown):
        handles.append(Patch(color="0.4", alpha=alpha[level] + 0.1))
        labels.append(f"{level:g} % interval")
    if depth_shown:
        handles.append(Line2D([], [], color="0.3", ls="--"))
        labels.append("5-sigma depth")
    if faint_shown:
        handles.append(Line2D([], [], ls="", marker="o", mfc="white", mec="0.3"))
        labels.append("most draws too faint")
    ax.legend(handles, labels, fontsize=8, ncol=2, loc="best",
              title="median" if "q50" in df.columns else "mean", title_fontsize=8)
    name = names[0] if names else df.attrs.get("model")      # a Comparison's label, if present
    n = df.attrs.get("n_draws")
    ax.set_title("Forecast" + (f": {name}" if name else "")
                 + (f" ({n} posterior draws)" if n else ""), fontsize=11)
    return ax


def _fits_and_weights(results_or_comparison):
    """``({label: result}, {label: weight or nan}, {label: reason left out}, {label: Model})``, in
    rank order for a Comparison. The models are a Comparison's own objects (a family bound by
    ``compare`` is not registered by name); elsewhere each result's ``.model`` is looked up."""
    import pandas as pd

    obj = results_or_comparison
    weights, left, models = {}, {}, {}
    if isinstance(getattr(obj, "results", None), dict):
        fits = dict(obj.results)
        models = {str(k): m for k, m in (getattr(obj, "models", None) or {}).items()
                  if m is not None and not isinstance(m, str)}
        table = getattr(obj, "table", None)
        if isinstance(table, pd.DataFrame) and "model" in table.columns:
            ranked = _ranked_rows(table) if "weight" in table.columns else table
            order = [str(m) for m in ranked["model"] if str(m) in fits]
            fits = {**{k: fits[k] for k in order}, **fits}
            if "weight" in table.columns:
                w = _numbers(table["weight"])
                weights = {str(m): float(v) for m, v in zip(table["model"], w)}
    elif hasattr(obj, "samples"):
        fits = {str(getattr(obj, "model", "fit")): obj}
    elif isinstance(obj, dict):
        fits = {str(k): v for k, v in obj.items()}
    else:
        fits = {}
        for i, r in enumerate(obj):
            label = str(getattr(r, "model", f"fit {i + 1}"))
            fits[label if label not in fits else f"{label} ({getattr(r, 'sampler', i + 1)})"] = r
    for k in list(fits):
        if getattr(fits[k], "n_samples", 1) == 0:
            left[k] = "no posterior draws"
            del fits[k]
    return fits, weights, left, models


def _pre_event_rule(result):
    """The pre-event rule a fit applied (``info["pre_event"]``), ``{}`` when none is recorded."""
    info = result.info if isinstance(getattr(result, "info", None), dict) else {}
    rule = info.get("pre_event")
    return rule if isinstance(rule, dict) else {}


def _kept_rows(time, rule):
    """``True`` for the epochs a fit under ``rule`` keeps (all of them without a rule)."""
    try:
        from .samplers.base import _pre_event_mask
    except ImportError:                          # an installation without the pre-event rule
        return np.ones(np.size(time), bool)
    return np.asarray(_pre_event_mask(time, rule), bool)


def plot_models(results_or_comparison, lc, ax=None, bands=None, *, n_draws=200, seed=0):
    """Every model over the data, in magnitudes, with residual panels.

    One column per band. Top: the photometry (upper limits as down-triangles) and, for each model,
    its posterior median and 68 % band. Bottom: data minus each model's median at the detections
    the fit used, with the data's error bars, so a systematic misfit reads as a trend. Rows a fit
    left out as pre-event data (``info["pre_event"]``) are drawn as open grey squares and have no
    residual; before a declared event (day 0, or a fixed explosion time) no curve is drawn.

    Parameters
    ----------
    results_or_comparison : Comparison, dict, list or SamplerResult
        A :func:`whisper_cbpf.compare` result (models in rank order, labelled with their weights),
        ``{label: SamplerResult}``, a list of results, or one result. Each is evaluated with the
        model its ``.model`` names; fits without posterior draws are left out and named.
    lc : LightCurve
        The fitted light curve (magnitudes or flux density; the plot is in AB magnitudes).
    ax : matplotlib.axes.Axes, optional
        Draw one band's top panel here, without the residuals (pass ``bands=[one band]``). By
        default a new figure with both rows.
    bands : sequence of str, optional
        Bands to show (default: every band of ``lc``).
    n_draws : int, default 200
        Posterior draws per model (without replacement), evaluated as in
        :func:`whisper_cbpf.forecast.forecast`: batched on the GPU for a JAX model.
    seed : int, default 0
        Chooses the draws.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes or matplotlib.axes.Axes
        The ``(2, n_bands)`` grid (top: light curves, bottom: residuals), or ``ax``. The figure is
        left open in pyplot, so it appears exactly once in a notebook.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> fits = {m: wp.fit(lc, m, sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
    ...         for m in ("flare", "bazin")}
    >>> from whisper_cbpf.plotting import plot_models
    >>> axes = plot_models(fits, lc)
    >>> axes.shape, axes[1, 0].get_ylabel()
    ((2, 1), 'data - model [mag]')
    """
    from .forecast import _draw_magnitudes, _draws, _model_of

    _safe_usetex()
    fits, weights, left, models = _fits_and_weights(results_or_comparison)
    if not fits:
        raise ValueError(f"plot_models has no model with posterior draws to draw (left out: "
                         f"{left}).")
    if getattr(lc, "data_mode", None) == "flux":
        raise ValueError("plot_models draws AB magnitudes, and this light curve is band-integrated "
                         "flux (data_mode='flux'), which has no magnitude.")
    sel = lc if bands is None else lc.select_bands(list(bands))
    if sel.n_points == 0:
        raise ValueError(f"the light curve has no point in bands {list(bands)}; it has "
                         f"{lc.bands}.")
    full = sel.add_mag()
    band_list = list(bands) if bands is not None else full.bands
    if ax is not None and len(band_list) != 1:
        raise ValueError(f"with ax= plot_models draws one band without residuals; pass "
                         f"bands=[one of {band_list}], or ax=None for one column per band.")
    t = np.asarray(full.time, float)
    b = np.asarray(full.band).astype(str)
    mag = np.asarray(full.magnitude, float)
    err = (np.asarray(full.magnitude_err, float) if full.magnitude_err is not None
           else np.zeros_like(mag))
    ul = (np.asarray(full.upper_limit, bool) if full.upper_limit is not None
          else np.zeros(t.size, bool))
    det = ~ul & np.isfinite(mag)
    span = float(np.ptp(t)) or 1.0
    tgrid = np.linspace(t.min() - 0.02 * span, t.max() + 0.02 * span, 160)
    nb = len(band_list)

    curves, resid = {}, {}
    fitted_any = np.zeros(t.size, bool)
    for label, res in fits.items():
        m = _model_of(res, models.get(str(label)))
        theta, _ = _draws(res, m, n_draws, seed)
        rule = _pre_event_rule(res)
        kept = _kept_rows(t, rule)                 # the rows this fit used (pre-event rule)
        fitted_any |= kept
        use = det & kept
        # before a declared event the model is not evaluated (it would move a model's grid)
        on = (_kept_rows(tgrid, rule) if rule.get("rule") in ("day 0", "fixed t_exp")
              else np.ones(tgrid.size, bool))
        tg = tgrid[on]
        pts_t = np.concatenate([np.repeat(tg, nb), t[use]])
        pts_b = np.concatenate([np.tile(np.asarray(band_list, dtype=object), tg.size), b[use]])
        mags, dark, _ = _draw_magnitudes(m, theta, pts_t, pts_b)
        lit = np.where(dark, np.nan, mags)
        enough = (~dark).sum(axis=0) >= 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-dark columns -> NaN
            q = np.nanpercentile(lit, [16.0, 50.0, 84.0], axis=0)
        q = np.where(enough[None, :], q, np.nan)
        ng = tg.size * nb
        curves[label] = np.full((3, tgrid.size, nb), np.nan)
        curves[label][:, on, :] = q[:, :ng].reshape(3, tg.size, nb)
        resid[label] = np.full(t.size, np.nan)
        resid[label][use] = mag[use] - q[1, ng:]

    if ax is None:
        fig, axes = plt.subplots(2, nb, figsize=(4.6 * nb + 0.8, 6.4), sharex="col", squeeze=False,
                                 gridspec_kw={"height_ratios": [3, 1]})
        top, bottom = axes[0], axes[1]
        out = axes
    else:
        fig, top, bottom, out = ax.figure, [ax], None, ax
    colors = {lbl: CORNER_PALETTE[i % len(CORNER_PALETTE)] for i, lbl in enumerate(fits)}

    def name(lbl):
        w = weights.get(lbl, float("nan"))
        return f"{lbl} (weight {w:.2f})" if np.isfinite(w) else str(lbl)

    for j, band in enumerate(band_list):
        axm = top[j]
        in_band = b == band
        for k, lbl in enumerate(fits):
            lo, med, hi = curves[lbl][:, :, j]
            axm.fill_between(tgrid, lo, hi, color=colors[lbl], alpha=0.18, lw=0)
            axm.plot(tgrid, med, color=colors[lbl], lw=1.8, label=name(lbl))
        m_det = in_band & det & fitted_any
        m_ul = in_band & ul & fitted_any
        m_pre = in_band & ~fitted_any & np.isfinite(mag)
        axm.errorbar(t[m_det], mag[m_det], yerr=err[m_det], fmt="o", ms=4.5, mfc="0.2", mec="black",
                     mew=0.5, ecolor="0.4", elinewidth=0.8, zorder=5, label="data")
        if m_ul.any():
            axm.plot(t[m_ul], mag[m_ul], "v", ms=6, mfc="white", mec="0.2", zorder=5,
                     label="upper limit")
        if m_pre.any():
            axm.plot(t[m_pre], mag[m_pre], "s", ms=5, mfc="white", mec="0.55", zorder=4,
                     label="pre-event (not fitted)")
        _invert_y(axm)
        axm.set_title(band, fontsize=11, weight="bold")
        axm.grid(alpha=0.25)
        if bottom is not None:
            axr = bottom[j]
            for k, lbl in enumerate(fits):
                rows = in_band & np.isfinite(resid[lbl])
                dx = (k - (len(fits) - 1) / 2.0) * 0.006 * span
                axr.errorbar(t[rows] + dx, resid[lbl][rows], yerr=err[rows], fmt="o", ms=3.5,
                             color=colors[lbl], elinewidth=0.7, alpha=0.9)
            axr.axhline(0.0, color="0.4", lw=0.9)
            axr.grid(alpha=0.25)
    top[0].set_ylabel(_MAG_LABEL)
    title = "left out: " + ", ".join(f"{k} ({v})" for k, v in left.items()) if left else None
    top[0].legend(fontsize=8, title=title, title_fontsize=7, loc="best")
    if bottom is not None:
        bottom[0].set_ylabel("data - model [mag]")
        fig.supxlabel(_time_label(full))
        fig.suptitle(f"Models over the data: {full.name or lc.name or 'light curve'}",
                     weight="bold")
        fig.tight_layout()
    else:
        ax.set_xlabel(_time_label(full))
    return out


_TABLE_NEEDS = ("model", "weight")


def _comparison_table(obj):
    """``(table, criterion, winner)`` from a Comparison or its table."""
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        table, crit, win = obj, obj.attrs.get("criterion"), obj.attrs.get("winner")
    else:
        table = getattr(obj, "table", None)
        crit, win = getattr(obj, "criterion", None), getattr(obj, "winner", None)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"plot_model_comparison takes a Comparison (wp.compare) or its .table; "
                            f"got {type(obj).__name__}.")
    missing = [c for c in _TABLE_NEEDS if c not in table.columns]
    if missing:
        raise ValueError(f"the comparison table has no {missing} column (columns: "
                         f"{list(table.columns)}); pass what wp.compare returns.")
    return table, crit, win


def _numbers(column):
    """A table column as floats, NaN for anything missing or not a number."""
    import pandas as pd

    return pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)


def _ranked_rows(table):
    """Rows best first: by ``delta`` ascending (by weight descending without it), models without a
    number last."""
    t = table.reset_index(drop=True)
    key = _numbers(t["delta"]) if "delta" in t.columns else -_numbers(t["weight"])
    key = np.where(np.isfinite(key), key, np.inf)
    return t.iloc[np.argsort(key, kind="stable")].reset_index(drop=True)


def _history_points(history, comparison, table):
    """``[(label, table)]`` for the evolution panel, ``comparison`` last."""
    items = list(history.items()) if isinstance(history, dict) else list(history)
    points, last = [], None
    for i, it in enumerate(items):
        label, comp = it if isinstance(it, tuple) and len(it) == 2 else (str(i + 1), it)
        points.append((str(label), _comparison_table(comp)[0]))
        last = comp
    if last is not comparison:
        points.append(("latest", table))
    return points


def plot_model_comparison(comparison, history=None):
    """The ranking: each model's weight, its gap to the best and its grade; optionally over time.

    Parameters
    ----------
    comparison : Comparison or pandas.DataFrame
        What :func:`whisper_cbpf.compare` returns, or its ``.table``. Read: ``model``, ``weight``
        and, when present, ``delta``, ``grade``, ``status`` and ``left_out_reason``; the criterion
        (``ln Z`` or ``BIC``) from ``.criterion`` or ``table.attrs["criterion"]``.
    history : dict or sequence, optional
        Earlier comparisons of the same transient, oldest first: ``{label: comparison}`` or a list
        of ``(label, comparison)`` pairs (labels such as "+3 d" or "6 detections"). Adds a second
        panel with each model's weight at every decision point; ``comparison`` is the last point
        unless it already is the last entry of ``history``.

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray of Axes
        The ranking panel, or ``[ranking, evolution]`` with ``history``. Bars are weights (best on
        top), annotated with the gap in the criterion and the grade; models left out are listed
        with their reason. The figure is left open in pyplot, so it appears exactly once in a
        notebook.

    Examples
    --------
    >>> import pandas as pd
    >>> table = pd.DataFrame({"model": ["arnett", "magnetar", "csm"],
    ...                       "delta": [0.0, 3.1, float("nan")], "weight": [0.83, 0.17, float("nan")],
    ...                       "grade": ["", "positive", ""], "status": ["ok", "ok", "left out"],
    ...                       "left_out_reason": ["", "", "k >= n"]})
    >>> table.attrs["criterion"] = "BIC"
    >>> from whisper_cbpf.plotting import plot_model_comparison
    >>> ax = plot_model_comparison(table)
    >>> ax.get_xlabel()
    'model weight'
    >>> earlier = table.assign(weight=[0.4, 0.6, float("nan")], delta=[0.8, 0.0, float("nan")])
    >>> axes = plot_model_comparison(table, history={"+3 d": earlier, "+6 d": table})
    >>> len(axes)
    2
    """
    _safe_usetex()
    table, crit, win = _comparison_table(comparison)
    rows = _ranked_rows(table)
    points = [] if history is None else _history_points(history, comparison, table)
    models = list(dict.fromkeys([str(m) for m in rows["model"]]
                                + [str(m) for _, t in points for m in t["model"]]))
    colors = {m: CORNER_PALETTE[i % len(CORNER_PALETTE)] for i, m in enumerate(models)}

    if points:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.55 * len(rows) + 2.8),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
        ax, ax2 = axes
    else:
        fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 2.2))
        axes = ax
    y = np.arange(len(rows))
    w = _numbers(rows["weight"])
    delta = _numbers(rows["delta"]) if "delta" in rows.columns else np.full(len(rows), np.nan)
    ok = np.isfinite(w)
    ax.barh(y[ok], w[ok], color=[colors[str(m)] for m in rows["model"][ok]], alpha=0.85)
    for i, r in rows.iterrows():
        if np.isfinite(w[i]):
            grade = r["grade"] if "grade" in rows.columns else None
            grade = grade if isinstance(grade, str) and grade else None
            if win is not None and str(r["model"]) == str(win):
                # the winner's grade is its preference over the runner-up, not a gap to itself
                parts = ["best"] + ([f"{grade} over the next"] if grade else [])
            else:
                parts = ([f"delta {delta[i]:.1f}"] if np.isfinite(delta[i]) else []) + \
                        ([grade] if grade else [])
            ax.text(w[i] + 0.01, i, "  ".join(parts), va="center", fontsize=8)
        else:
            why = r["left_out_reason"] if "left_out_reason" in rows.columns else None
            ax.text(0.01, i, "left out" + (f": {why}" if isinstance(why, str) and why else ""),
                    va="center", fontsize=8, color="0.45", style="italic")
    ax.set_yticks(y, [str(m) for m in rows["model"]])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.3)
    ax.set_xlabel("model weight")
    ax.grid(axis="x", alpha=0.3)
    what = crit or "the comparison's criterion"
    head = f"Ranking by {what}" + (f": {win} preferred" if win else "")
    ax.set_title(head + "\n(delta: gap to the best, in the criterion's units)", fontsize=10)

    if points:
        x = np.arange(len(points))
        for m in models:
            ys = []
            for _, t in points:
                hit = _numbers(t.loc[t["model"].astype(str) == m, "weight"])
                ys.append(hit[0] if hit.size else np.nan)
            ax2.plot(x, ys, "-o", color=colors[m], lw=1.6, ms=5, label=m)
        ax2.set_xticks(x, [lbl for lbl, _ in points])
        ax2.set_ylim(-0.02, 1.05)
        ax2.set_ylabel("model weight")
        ax2.set_xlabel("decision point")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc="best")
        ax2.set_title("How the weights evolve as data arrive", fontsize=10)
        axes = np.asarray([ax, ax2], dtype=object)
    fig.tight_layout()
    return axes


def _own_coordinate(dist, x):
    """``x`` in the prior's own coordinate: log10 for a LogUniform, as it is otherwise."""
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, 1e-300, None)) if type(dist).__name__ == "LogUniform" else x


def posterior_width_ratios(result, prior=None):
    """``{parameter: posterior 68 % width / prior 68 % width}``, in each prior's own coordinate.

    The rule of the facts file (``whisper_cbpf.facts``, ``width_ratio``): the width is ``p84 -
    p16`` of the finite posterior draws (``numpy.percentile``), taken in log10 for a LogUniform
    prior and linearly otherwise, over the same range of the prior from its inverse CDF
    (``rescale(0.16)``, ``rescale(0.84)``). A ratio near 1 means the data did not narrow that
    parameter (the posterior repeats the prior); well below 1, they did. Fixed or pinned parameters
    and those without a prior or a posterior column are skipped.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from types import SimpleNamespace
    >>> from whisper_cbpf.priors import Prior, Uniform
    >>> res = SimpleNamespace(samples=pd.DataFrame({"a": np.linspace(0.0, 1.0, 10001)}))
    >>> round(posterior_width_ratios(res, Prior({"a": Uniform(0.0, 1.0)}))["a"], 4)
    1.0
    """
    prior = _fit_prior(result, prior)
    if prior is None:
        raise ValueError("the fit's prior is unknown (none recorded, and its model is not "
                         "registered in this session); pass prior=<the Prior the fit used>.")
    samples = result.samples
    if len(samples) < 2:
        raise ValueError(f"not enough data: {len(samples)} posterior draw(s), and a width needs at "
                         f"least two.")
    info = result.info if isinstance(getattr(result, "info", None), dict) else {}
    pinned = info.get("fixed") if isinstance(info.get("fixed"), dict) else {}
    out = {}
    for name, dist in prior.distributions.items():
        if type(dist).__name__ == "Fixed" or name in pinned or name not in samples.columns:
            continue
        x = samples[name].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            out[name] = float("nan")
            continue
        p16, p84 = _own_coordinate(dist, np.percentile(x, [16.0, 84.0]))
        lo, hi = (_own_coordinate(dist, dist.rescale(u)) for u in (0.16, 0.84))
        prior_w = float(hi - lo)
        out[name] = float(p84 - p16) / prior_w if prior_w > 0 else float("nan")
    return out


def _prior_dominated_ratio():
    """The facts file's ``prior_dominated`` threshold, or None where that module is absent."""
    try:
        from .facts import DEFAULT_THRESHOLDS
        return float(DEFAULT_THRESHOLDS["prior_dominated_ratio"])
    except (ImportError, KeyError, TypeError, ValueError):
        return None


def plot_widths(result, prior=None, *, ax=None):
    """How much the data narrowed each parameter: posterior 68 % width over prior 68 % width.

    Widths are taken in each prior's own coordinate (log10 for a LogUniform), so a posterior that
    repeats its prior reads 1 whatever the prior's shape, and a well-measured parameter reads close
    to 0. See :func:`posterior_width_ratios` for the numbers.

    Parameters
    ----------
    result : SamplerResult
        The fit.
    prior : Prior, optional
        The prior the fit used. By default the one recorded with the fit, else its model's default.
    ax : matplotlib.axes.Axes, optional
        Draw here; by default a new figure.

    Returns
    -------
    matplotlib.axes.Axes
        One bar per free parameter (``(log10)`` marks a log coordinate), with a dotted line at 1
        (posterior = prior) and a dashed line at the facts file's ``prior_dominated`` threshold
        (``whisper_cbpf.facts.DEFAULT_THRESHOLDS``). The figure is left open in pyplot, so it
        appears exactly once in a notebook.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=2000, quantile=0.05, seed=0)
    >>> from whisper_cbpf.plotting import plot_widths
    >>> ax = plot_widths(res)
    >>> [tick.get_text() for tick in ax.get_yticklabels()]
    ['amplitude', 'rise_time', 'decay_time']
    """
    _safe_usetex()
    prior = _fit_prior(result, prior)
    ratios = posterior_width_ratios(result, prior)
    if not ratios:
        raise ValueError(f"no free parameter of the prior ({list(prior.distributions)}) has a "
                         f"posterior column ({list(result.samples.columns)}).")
    logs = _log_uniform_names(prior)
    names = list(ratios)
    vals = np.array([ratios[n] for n in names], dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.45 * len(names) + 1.8))
    y = np.arange(len(names))
    ax.barh(y, vals, color=CORNER_PALETTE[0], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v + 0.02 if np.isfinite(v) else 0.02, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(1.0, color="0.3", ls=":", lw=1.2, label="posterior = prior")
    dominated = _prior_dominated_ratio()
    if dominated is not None:
        ax.axvline(dominated, color="#a50026", ls="--", lw=1.0,
                   label=f"prior-dominated above {dominated:.2f} (facts rule)")
    ax.set_yticks(y, [n + (" (log10)" if n in logs else "") for n in names])
    ax.invert_yaxis()
    top = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    ax.set_xlim(0.0, max(1.15, 1.1 * top))
    ax.set_xlabel("posterior 68 % width / prior 68 % width")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"What the data constrained: {getattr(result, 'model', '')}"
                 f" ({getattr(result, 'sampler', '')})", fontsize=10)
    ax.figure.tight_layout()
    return ax
