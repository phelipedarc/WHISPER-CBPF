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
    if "explosion_mjd" in lc.meta:
        return f"days since explosion (MJD {lc.meta['explosion_mjd']:.3f})"
    return "time [MJD]"


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
    """Report plot of a light curve. Returns the matplotlib ``Figure``.

    See the module docstring for layouts, the ``quantity`` options, and the marker conventions.
    """
    _safe_usetex()
    if bands is not None:
        lc = lc.select_bands(bands)
    full = lc.add_flux().add_mag()      # ensure both magnitude and flux are available
    band_list = full.bands
    colors = _band_colors(band_list)
    cat = _categories(full)

    if layout == "report":
        fig, (ax_m, ax_f) = plt.subplots(2, 1, sharex=True, figsize=figsize or (9, 8))
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
    return fig


# --- posterior-predictive check --------------------------------------------------------------------

_AB_ZP_JY = 3631.0


def _flux_to_quantity(flux, quantity):
    """Map model flux density [Jy] to the plotted quantity."""
    if quantity in ("flux", "flux_density"):
        return np.asarray(flux, float)
    return -2.5 * np.log10(np.clip(np.asarray(flux, float), 1e-300, None) / _AB_ZP_JY)   # AB mag


def _ppc_curves(result, model, tgrid, band_list, quantity, n_draws, seed):
    """Posterior-predictive percentile curves (2.5/50/97.5) per band over ``tgrid``."""
    from .models import get_model

    m = get_model(model if model is not None else getattr(result, "model", None))
    if m is None:
        raise ValueError("plot_ppc needs a model (pass model=... or use a result with .model).")
    samples = result.samples
    cols = [c for c in m.parameters if c in samples.columns]
    draws = samples[cols].to_numpy(dtype=float)
    idx = np.random.default_rng(seed).choice(
        len(draws), size=min(n_draws, len(draws)), replace=len(draws) < n_draws)
    curves = {}
    for b in band_list:
        gb = np.array([b] * len(tgrid))
        stack = np.empty((len(idx), len(tgrid)), dtype=float)
        for i, j in enumerate(idx):
            p = {c: float(draws[j, k]) for k, c in enumerate(cols)}
            stack[i] = _flux_to_quantity(np.asarray(m.predict(p, tgrid, gb), float), quantity)
        curves[b] = np.nanpercentile(stack, [2.5, 50, 97.5], axis=0)
    return curves


def plot_ppc(results, lc, model=None, *, quantity="apparent_mag", panel_by="auto", n_draws=200,
             bands=None, tmin=None, tmax=None, ncols=None, colors=None, figsize=None, title=None,
             seed=0, save=None):
    """Posterior-predictive check: model band(s) over the data, in a **grid** of panels.

    For each fit, draws ``n_draws`` posterior samples, evaluates the model on a smooth time grid, and
    shades the **95% posterior-predictive band** (2.5–97.5 percentiles) with the median curve, over the
    observed photometry — per band. Two panel layouts:

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

    Returns the matplotlib ``Figure``.
    """
    if quantity.lower() in ("flux", "flux_density"):
        quantity, invert = "flux", False
    else:
        quantity, invert = "apparent_mag", True

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

    curves = {lbl: _ppc_curves(r, model, tgrid, band_list, quantity, n_draws, seed)
              for lbl, r in fits.items()}
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
                lo, med, hi = curves[panel][b]
                ax.fill_between(tgrid, lo, hi, color=band_col[b], alpha=0.22, lw=0)
                ax.plot(tgrid, med, color=band_col[b], lw=1.8, label=b)
                sel = obs_band == b
                ax.errorbar(t[sel], obs_y[sel], yerr=None if obs_e is None else obs_e[sel],
                            fmt="o", ms=5, mfc=band_col[b], mec="black", mew=0.5,
                            ecolor="black", elinewidth=0.7, alpha=0.9, zorder=5)
            ax.set_title(panel, fontsize=13, weight="bold")
        else:                                                  # panel_by == 'band'
            b = panel
            for lbl, r in fits.items():
                lo, med, hi = curves[lbl][b]
                ax.fill_between(tgrid, lo, hi, color=fit_col[lbl], alpha=0.18, lw=0)
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
                     title="band" if panel_by == "method" else "fit")
    fig.supxlabel(_time_label(full))
    fig.supylabel(ylabel)
    fig.suptitle(title or f"Posterior-predictive check — {full.name or lc.name or 'light curve'}",
                 weight="bold")
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig


# --- coverage-calibration curve --------------------------------------------------------------------

def plot_calibration(results, lc, model=None, *, levels=(0.5, 0.68, 0.8, 0.9, 0.95, 0.99),
                     space="auto", per_band=False, n_draws=400, colors=None, figsize=None,
                     title=None, seed=0, save=None):
    """Coverage-calibration curve (reliability / pp-plot) for one or more fits.

    For each nominal credible level, plots the **empirical** posterior-predictive coverage — the
    fraction of observations inside the central predictive interval (model + observation noise) — against
    the nominal level. A well-calibrated fit lies on the diagonal; below ⇒ over-confident (intervals too
    narrow), above ⇒ under-confident. Coverage is computed by
    :func:`whisper_labia.metrics.predictive_metrics`.

    ``results`` is a `SamplerResult`, a `{label: result}` dict, or a list. With ``per_band=True`` a
    single fit is broken out into one line per band. Returns the ``Figure``.
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
    return fig


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


def plot_corner(posteriors, *, labels=None, parameters=None, colors=None, truths=None,
                bins=30, levels=(0.39, 0.86), smooth=1.0, log_params=None, title=None,
                legend_loc="upper right", save=None, **corner_kwargs):
    """Overlay one or more posteriors on a single publication-ready corner plot.

    A thin, well-styled wrapper over :mod:`corner` for comparing posteriors (e.g. several samplers on
    the same data): shared per-parameter ranges so the panels align, a dark distinct colour per
    posterior, contour lines (not filled) so overlaps stay readable, and a legend.

    Parameters
    ----------
    posteriors : sequence
        Each item is a :class:`~whisper_labia.samplers.base.SamplerResult`, a ``pandas.DataFrame`` of
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
    log_params : sequence of str, optional
        Parameters to display on a ``log10`` axis (samples are log10-transformed; the label is prefixed).
    title : str, optional
        Figure suptitle.
    save : str, optional
        If given, save the figure there (PNG, 200 dpi).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import whisper_labia as wp
    >>> fig = wp.plot_corner([res_abc, res_mcmc, res_snpe],
    ...                      labels=["ABC", "MCMC", "SNPE"], log_params=["mej_1"],
    ...                      title="AT2017GFO posteriors")
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
    log_params = set(log_params or [])
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
    return fig
