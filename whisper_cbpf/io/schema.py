"""Canonical light-curve container: a subclass of :class:`astropy.table.Table`.

``LightCurve`` **is** an astropy ``Table`` -- per-point quantities are columns (``time``, ``band``,
``magnitude``, ``flux``, ...) and scalar metadata lives in ``.meta`` (``name``, ``redshift``,
``data_mode``, ``luminosity_distance``, ``dm``, ``refmjd``, ...). So all the usual table ergonomics
work directly::

    lc['absmag_shift'] = lc['magnitude'] + 5      # add/compute columns
    bright = lc[lc['magnitude'] < 18]             # boolean-mask slicing (keeps the subclass + meta)
    lc.sort('time'); lc.group_by('band')          # any Table method

For convenience and backward compatibility the common quantities are **also** exposed as attributes:
``lc.time`` / ``lc.flux`` / ``lc.band`` ... return the column data (or ``None`` if absent), and
``lc.redshift`` / ``lc.data_mode`` / ``lc.name`` read from ``.meta``. Calling the object (``lc()``)
returns the table itself.

``data_mode`` is one of :data:`VALID_DATA_MODES`; ``flux`` is stored in janskys (Jy, flux density) or
erg/s/cm^2 for band-integrated ``flux`` mode. Subsetting (``select_*`` / :meth:`where`) returns a new
``LightCurve``.
"""
from __future__ import annotations

import warnings

import numpy as np
from astropy.table import Table

from .photometry import (
    AB_ZEROPOINT_JY,
    flux_density_to_mag,
    mag_err_to_snr,
    mag_to_flux_density,
)

#: Allowed values for :attr:`LightCurve.data_mode`.
from .units import VALID_DATA_MODES  # noqa: F401  (re-exported by io/__init__)

#: data_mode -> the forward-model comparison space ('magnitude' | 'flux_density'); band-integrated
#: ``flux`` maps onto ``flux_density``. (Optional physical-model backends such as redback use the same.)
_OUTPUT_FORMAT = {"flux_density": "flux_density", "magnitude": "magnitude", "flux": "flux_density"}

#: Default prior spec attached when the redshift is unknown (override per object). A
#: serialisable hint.
DEFAULT_REDSHIFT_PRIOR = {"type": "Uniform", "low": 0.001, "high": 1.0, "name": "redshift"}


def redshift_distribution(hint):
    """The prior distribution a redshift hint (:attr:`LightCurve.redshift_prior`) describes.

    The JAX model factories call this when their redshift is free
    (``supernova_model(..., free=["redshift"], redshift_prior=lc.redshift_prior)``).

    Parameters
    ----------
    hint : dict, distribution or None
        A dict with ``"type"``: ``"Uniform"`` or ``"LogUniform"`` (keys ``low``, ``high``), or
        ``"Normal"`` / ``"TruncatedNormal"`` (keys ``mu``, ``sigma`` and optionally ``low``,
        ``high``; ``loc`` / ``scale`` also read). A distribution object (anything with ``sample``
        and ``bounds``) is returned as it is; ``None`` gives ``None``.

    Returns
    -------
    distribution or None

    Raises
    ------
    ValueError
        An unknown ``type``, or missing bounds / width.
    NotImplementedError
        A Normal hint when this install's :mod:`whisper_cbpf.priors` has no ``TruncatedNormal``;
        the message says what to pass instead.

    Examples
    --------
    >>> from whisper_cbpf.io.schema import DEFAULT_REDSHIFT_PRIOR, redshift_distribution
    >>> redshift_distribution(DEFAULT_REDSHIFT_PRIOR)
    Uniform(0.001, 1.0)
    >>> redshift_distribution({"type": "LogUniform", "low": 0.01, "high": 0.5})
    LogUniform(0.01, 0.5)
    """
    if hint is None:
        return None
    if hasattr(hint, "sample") and hasattr(hint, "bounds"):
        return hint
    if not isinstance(hint, dict):
        raise ValueError(f"a redshift prior hint is a dict such as {DEFAULT_REDSHIFT_PRIOR} or a "
                         f"prior distribution; got {hint!r}")
    from .. import priors as P

    kind = str(hint.get("type", "")).replace("_", "").replace(" ", "").lower()
    if kind in ("uniform", "loguniform"):
        try:
            low, high = float(hint["low"]), float(hint["high"])
        except KeyError as exc:
            raise ValueError(f"redshift prior hint {hint!r} has no {exc.args[0]!r}; a "
                             f"{hint['type']} hint needs 'low' and 'high'.") from None
        return (P.Uniform if kind == "uniform" else P.LogUniform)(low, high)
    if kind in ("normal", "gaussian", "truncatednormal", "truncnorm"):
        mu = hint.get("mu", hint.get("loc", hint.get("mean")))
        sigma = hint.get("sigma", hint.get("scale", hint.get("std")))
        if mu is None or sigma is None:
            raise ValueError(f"redshift prior hint {hint!r} needs 'mu' and 'sigma'.")
        mu, sigma = float(mu), float(sigma)
        low = float(hint.get("low", max(1e-3, mu - 5.0 * sigma)))
        high = float(hint.get("high", mu + 5.0 * sigma))
        cls = getattr(P, "TruncatedNormal", None)
        if cls is None:
            raise NotImplementedError(
                f"redshift prior hint {hint!r} is a Normal, and this install's whisper_cbpf.priors "
                f"has no TruncatedNormal. Pass a box instead, e.g. prior=Prior({{'redshift': "
                f"Uniform({low:g}, {high:g})}}), or a hint {{'type': 'Uniform', 'low': {low:g}, "
                f"'high': {high:g}}}.")
        return cls(mu, sigma, low, high)
    raise ValueError(f"redshift prior hint {hint!r}: unknown type {hint.get('type')!r}. Known: "
                     f"'Uniform', 'LogUniform', 'Normal' / 'TruncatedNormal'.")

#: Width [days] of :meth:`LightCurve.explosion_time_prior` when no non-detection precedes the first
#: detection: the window then runs from this many days before the first detection to it. 30 d covers
#: the observer-frame rise of most supernovae; it is too wide for a kilonova and too narrow for some
#: superluminous supernovae and tidal disruption events -- pass ``fallback_days=`` for those.
EXPLOSION_FALLBACK_DAYS = 30.0


def explosion_window(time, upper_limit=None):
    """The first detection and the last non-detection before it, on the light curve's own clock.

    The data behind :meth:`LightCurve.explosion_time_prior` and the pre-event rule of the fits
    (:func:`whisper_cbpf.samplers.base.prepare_lc`).

    Parameters
    ----------
    time : array_like
        Epochs, on the light curve's clock.
    upper_limit : array_like of bool, optional
        ``True`` for a non-detection. ``None`` means every row is a detection.

    Returns
    -------
    dict
        ``first_detection`` (float), ``last_non_detection`` (float, or ``None`` when no
        non-detection precedes the first detection) and ``n_before`` (rows before the first
        detection, all non-detections).

    Raises
    ------
    ValueError
        No detection ("not enough data").

    Examples
    --------
    >>> from whisper_cbpf.io.schema import explosion_window
    >>> explosion_window([1.0, 2.0, 3.5, 4.0], [True, True, False, False])
    {'first_detection': 3.5, 'last_non_detection': 2.0, 'n_before': 2}
    """
    t = np.asarray(time, dtype=float)
    ul = np.zeros(t.size, dtype=bool) if upper_limit is None else np.asarray(upper_limit, bool)
    if not np.any(~ul):
        raise ValueError(
            f"not enough data: this light curve has no detection ({int(ul.sum())} upper limit(s)), "
            f"so there is no first detection to place the explosion before. Wait for a detection.")
    t_first = float(np.min(t[~ul]))
    before = ul & (t < t_first)
    last = float(np.max(t[before])) if np.any(before) else None
    return {"first_detection": t_first, "last_non_detection": last, "n_before": int(before.sum())}


#: Per-point columns Whisper understands (others may be added freely as ordinary columns).
_FLOAT_COLS = ("time", "magnitude", "magnitude_err", "flux", "flux_err", "lambda_eff", "zero_point")
_SCALAR_META = ("name", "redshift", "luminosity_distance", "data_mode", "redshift_prior")


class LightCurve(Table):
    """A multi-band transient light curve, stored as an :class:`astropy.table.Table`.

    Construct from arrays (``LightCurve(time=..., band=..., flux=..., ...)``) or from anything
    ``Table`` accepts. See the module docstring for the column / metadata layout.
    :func:`whisper_cbpf.load_lightcurve` builds one from a file, a table or survey alerts.

    Parameters
    ----------
    time : array_like
        Epochs in days (MJD, or days since a reference once one is set).
    band : array_like of str
        One band label per epoch (``"ztfr"``, ``"lsstg"``, ...).
    magnitude, magnitude_err : array_like, optional
        AB magnitudes and their errors. Give these or ``flux``.
    flux, flux_err : array_like, optional
        Flux densities in Jy and their errors.
    upper_limit : array_like of bool, optional
        True for a non-detection; its magnitude or flux is the limit.
    system, lambda_eff, zero_point : array_like, optional
        Photometric system, effective wavelength [Angstrom] and zero point [Jy] per epoch.
    name : str, optional
    redshift : float, optional
        Known redshift. Unknown (None) records ``redshift_known=False`` and a default
        ``redshift_prior`` that fits sample.
    luminosity_distance : float, optional
        Mpc; needed when ``redshift == 0``.
    data_mode : {"magnitude", "flux_density", "flux"}, optional
        Inferred from the columns when omitted.
    redshift_prior : dict or Prior, optional
        The prior a fitted redshift takes when ``redshift`` is unknown.
    meta : dict, optional
        Extra metadata.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=[60000.0, 60002.0, 60005.0], band=["ztfg", "ztfr", "ztfg"],
    ...                    magnitude=[19.1, 18.7, 18.9], magnitude_err=[0.1, 0.08, 0.1],
    ...                    redshift=0.05, name="SN demo")
    >>> lc.n_points, lc.bands
    (3, ['ztfg', 'ztfr'])
    >>> lc.set_explosion_date(59999.0).time.tolist()
    [1.0, 3.0, 6.0]
    """

    def __init__(self, data=None, *, time=None, band=None, magnitude=None, magnitude_err=None,
                 flux=None, flux_err=None, upper_limit=None, system=None, lambda_eff=None,
                 zero_point=None, name=None, redshift=None, luminosity_distance=None, data_mode=None,
                 redshift_prior=None, meta=None, **kwargs):
        building = data is None and any(v is not None for v in (time, band, magnitude, flux))
        if not building:
            # astropy path: slice / copy / construct-from-table-or-columns. meta (incl. our scalars)
            # is propagated by astropy; just backfill defaults for a bare table.
            super().__init__(data, meta=meta, **kwargs)
            self._ensure_meta_defaults()
            return

        if band is None:
            raise ValueError("LightCurve requires a 'band' array.")
        cols = {"time": np.asarray(time, dtype=float),
                "band": np.array([str(b) for b in band])}
        n = len(cols["time"])
        for nm, v in (("magnitude", magnitude), ("magnitude_err", magnitude_err),
                      ("flux", flux), ("flux_err", flux_err),
                      ("lambda_eff", lambda_eff), ("zero_point", zero_point)):
            if v is not None:
                cols[nm] = np.asarray(v, dtype=float)
        if upper_limit is not None:
            cols["upper_limit"] = np.asarray(upper_limit, dtype=bool)
        if system is not None:
            cols["system"] = np.array([str(s) for s in system])
        for nm, arr in cols.items():
            if len(arr) != n:
                raise ValueError(f"{nm} length {len(arr)} != time length {n}")
        if "magnitude" not in cols and "flux" not in cols:
            raise ValueError("LightCurve requires either 'magnitude' or 'flux'.")

        super().__init__(cols, meta=dict(meta) if meta else {})
        self.meta["name"] = name
        self.meta["redshift"] = None if redshift is None else float(redshift)
        self.meta["luminosity_distance"] = luminosity_distance
        self.meta["data_mode"] = self._resolve_data_mode(data_mode)
        self.meta["redshift_prior"] = redshift_prior
        self._validate_redshift()
        if self.meta["redshift"] is None and self.meta["redshift_prior"] is None:
            self.meta["redshift_prior"] = dict(DEFAULT_REDSHIFT_PRIOR)

    # --- setup helpers ---
    def _ensure_meta_defaults(self):
        for key in _SCALAR_META:
            self.meta.setdefault(key, None)
        if self.meta.get("data_mode") is None:
            self.meta["data_mode"] = self._resolve_data_mode(None)

    def _resolve_data_mode(self, data_mode):
        if data_mode is not None:
            if data_mode not in VALID_DATA_MODES:
                raise ValueError(
                    f"data_mode={data_mode!r} invalid; expected one of {VALID_DATA_MODES}.")
            return data_mode
        return "magnitude" if "magnitude" in self.colnames else "flux_density"

    def _validate_redshift(self):
        z = self.meta.get("redshift")
        if z is None:
            return
        z = float(z)
        if not np.isfinite(z):
            raise ValueError(f"redshift must be finite; got {z!r}.")
        if z < 0:
            raise ValueError(f"redshift must be >= 0; got {z}.")
        if z == 0 and self.meta.get("luminosity_distance") is None:
            raise ValueError(
                "redshift == 0 leaves the luminosity distance undefined; pass an explicit "
                "luminosity_distance (Mpc) instead of z=0.")
        self.meta["redshift"] = z

    # --- scalar metadata (properties) ---
    @property
    def name(self):
        return self.meta.get("name")

    @name.setter
    def name(self, value):
        self.meta["name"] = value

    @property
    def redshift(self):
        return self.meta.get("redshift")

    @property
    def redshift_known(self):
        """``True`` when a redshift was supplied; ``False`` means a prior must be sampled."""
        return self.meta.get("redshift") is not None

    @property
    def redshift_prior(self):
        return self.meta.get("redshift_prior")

    @property
    def luminosity_distance(self):
        return self.meta.get("luminosity_distance")

    @property
    def data_mode(self):
        return self.meta.get("data_mode")

    @property
    def output_format(self):
        """Forward-model comparison space ('magnitude' | 'flux_density'); see :data:`_OUTPUT_FORMAT`."""
        return _OUTPUT_FORMAT[self.data_mode]

    # --- per-point columns (attrs) ---
    def _col(self, name):
        return self[name].data if name in self.colnames else None

    def _set_col(self, name, value):
        if value is None:
            if name in self.colnames:
                self.remove_column(name)
        else:
            self[name] = value

    time = property(lambda self: self._col("time"),
                    lambda self, v: self._set_col("time", v))
    band = property(lambda self: self._col("band"),
                    lambda self, v: self._set_col("band", v))
    magnitude = property(lambda self: self._col("magnitude"),
                         lambda self, v: self._set_col("magnitude", v))
    magnitude_err = property(lambda self: self._col("magnitude_err"),
                             lambda self, v: self._set_col("magnitude_err", v))
    flux = property(lambda self: self._col("flux"),
                    lambda self, v: self._set_col("flux", v))
    flux_err = property(lambda self: self._col("flux_err"),
                        lambda self, v: self._set_col("flux_err", v))
    upper_limit = property(lambda self: self._col("upper_limit"),
                           lambda self, v: self._set_col("upper_limit", v))
    system = property(lambda self: self._col("system"),
                      lambda self, v: self._set_col("system", v))
    lambda_eff = property(lambda self: self._col("lambda_eff"),
                          lambda self, v: self._set_col("lambda_eff", v))
    zero_point = property(lambda self: self._col("zero_point"),
                          lambda self, v: self._set_col("zero_point", v))

    # --- derived info ---
    @property
    def n_points(self):
        return len(self)

    @property
    def bands(self):
        return sorted(set(np.asarray(self["band"]).tolist()))

    @property
    def snr(self):
        """Per-point signal-to-noise (``flux/flux_err`` or ``(2.5/ln10)/magnitude_err``)."""
        if "flux" in self.colnames and "flux_err" in self.colnames:
            return np.abs(self["flux"].data) / self["flux_err"].data
        if "magnitude_err" in self.colnames:
            return mag_err_to_snr(self["magnitude_err"].data)
        raise ValueError("Cannot compute SNR: need flux_err or magnitude_err.")

    # --- selection ---
    def where(self, **constraints):
        """Return the subset matching ``column=value`` constraints (astropy-table-fitting style).

        Each key is a column name, optionally suffixed: ``col`` (==), ``col_not`` (!=), ``col_min``
        (>=), ``col_max`` (<=). A list value means "match any of" (or, with ``_not``, "match none of").
        E.g. ``lc.where(band='r', time_min=58000, time_max=58020, upper_limit=False)``.
        """
        mask = np.ones(len(self), dtype=bool)
        for key, val in constraints.items():
            col, op = key, "in"
            for suffix in ("_min", "_max", "_not"):
                if key.endswith(suffix):
                    col, op = key[:-4], suffix
                    break
            # Resolve the column BEFORE dereferencing it, or astropy's own KeyError fires first
            # and this message never reaches the caller.
            if col not in self.colnames:
                raise KeyError(f"where(): no column {col!r}; have {self.colnames}")
            data = self[col].data
            if op == "_min":
                m = data >= val
            elif op == "_max":
                m = data <= val
            else:
                vals = val if isinstance(val, (list, tuple, set, np.ndarray)) else [val]
                m = np.isin(data, list(vals))
                if op == "_not":
                    m = ~m
            mask &= m
        return self[mask]

    def select_bands(self, bands):
        """Keep only the given band(s) (str or iterable)."""
        bands = [bands] if isinstance(bands, str) else list(bands)
        return self[np.isin(self["band"].data, bands)]

    def select_time_window(self, time_min=None, time_max=None):
        """Keep points with ``time_min <= time <= time_max``."""
        mask = np.ones(len(self), dtype=bool)
        if time_min is not None:
            mask &= self["time"].data >= time_min
        if time_max is not None:
            mask &= self["time"].data <= time_max
        return self[mask]

    def select_snr(self, min_snr=5.0):
        """Keep the detections with signal-to-noise ratio >= ``min_snr``, and every upper limit.

        A non-detection has no signal-to-noise (its error is not a measurement error), and a fit
        uses it as a limit, so the cut applies to detections only.

        Parameters
        ----------
        min_snr : float, default 5.0

        Returns
        -------
        LightCurve

        Examples
        --------
        >>> import numpy as np, whisper_cbpf as wp
        >>> lc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[0.3, 1.0, 5.0],
        ...                    flux_err=[np.nan, 0.5, 0.5], upper_limit=[True, False, False])
        >>> lc.select_snr(5.0).time.tolist()        # the 2-sigma point goes, the limit stays
        [1.0, 3.0]
        """
        ul = self.upper_limit
        ul = np.zeros(len(self), dtype=bool) if ul is None else np.asarray(ul, dtype=bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            keep = ul | (self.snr >= min_snr)
        return self[keep]

    # --- band resolution ---
    def _resolved_zero_point(self, svo_fallback=False):
        """Per-point zero point (Jy): stored if present, else resolved locally; NaN -> AB 3631."""
        if "zero_point" in self.colnames:
            zp = self["zero_point"].data
        else:
            from .bands import resolve_bands
            _, zp, _ = resolve_bands(self["band"].data, svo_fallback=svo_fallback, warn=False)
        zp = np.asarray(zp, dtype=float)
        return np.where(np.isfinite(zp), zp, AB_ZEROPOINT_JY)

    def resolve_bands(self, *, svo_fallback=True):
        """Copy with ``lambda_eff`` / ``zero_point`` columns filled (FILTER_LOOKUP + SVO fallback)."""
        from .bands import resolve_bands as _resolve
        lam, zp, _ = _resolve(self["band"].data, svo_fallback=svo_fallback)
        out = self.copy()
        out["lambda_eff"] = lam
        out["zero_point"] = zp
        return out

    def _zp_for_conversion(self, zeropoint_jy):
        return AB_ZEROPOINT_JY if zeropoint_jy is None else zeropoint_jy

    # --- derived columns ---
    def add_flux(self, zeropoint_jy=AB_ZEROPOINT_JY):
        """Copy with ``flux`` (+ ``flux_err``) from magnitude (constant AB zero point). No-op if present."""
        if self.data_mode == "flux":
            raise ValueError("add_flux is for flux-density data; data_mode='flux' is band-integrated.")
        if "flux" in self.colnames:
            return self.copy()
        if "magnitude" not in self.colnames:
            raise ValueError("Cannot add flux: no magnitude available.")
        out = self.copy()
        if "magnitude_err" in self.colnames:
            out["flux"], out["flux_err"] = mag_to_flux_density(
                self["magnitude"].data, self["magnitude_err"].data, zeropoint_jy)
        else:
            out["flux"] = mag_to_flux_density(self["magnitude"].data, None, zeropoint_jy)
        return out

    def add_mag(self, zeropoint_jy=AB_ZEROPOINT_JY):
        """Copy with ``magnitude`` (+ ``magnitude_err``) from flux (constant AB zero point). No-op if present."""
        if self.data_mode == "flux":
            raise ValueError("add_mag is for flux-density data; data_mode='flux' is band-integrated.")
        if "magnitude" in self.colnames:
            return self.copy()
        if "flux" not in self.colnames:
            raise ValueError("Cannot add magnitude: no flux available.")
        out = self.copy()
        if "flux_err" in self.colnames:
            out["magnitude"], out["magnitude_err"] = flux_density_to_mag(
                self["flux"].data, self["flux_err"].data, zeropoint_jy)
        else:
            out["magnitude"] = flux_density_to_mag(self["flux"].data, None, zeropoint_jy)
        return out

    def set_time_reference(self, mjd, label):
        """Copy with ``time`` measured in **days since ``mjd`` (MJD)**, recording what that date is.

        ``label`` names day 0, e.g. ``"first detection"``, ``"discovery"`` or ``"peak"``. It is kept
        in ``meta['time_reference']`` (the date in ``meta['time_reference_mjd']``), so plots label
        the time axis "days since first detection (MJD ...)" and :meth:`calc_phase` counts from it.
        A model still reads ``time`` as days since its own day 0 (the explosion, for redback's
        models). A fit therefore leaves out the rows at or before day 0 (pre-event data are never
        fitted; :func:`whisper_cbpf.samplers.base.prepare_lc`), except for a model that fits its
        own epoch (a ``t0`` parameter, as ``bazin``) when ``label`` is not ``"explosion"`` or
        ``"merger"``. For the explosion itself use :meth:`set_explosion_date`.
        """
        out = self.copy()
        out["time"] = out["time"].data - float(mjd)
        out.meta["time_reference"] = str(label)
        out.meta["time_reference_mjd"] = float(mjd)
        return out

    def set_explosion_date(self, explosion_date):
        """Copy with ``time`` measured in **days since ``explosion_date`` (MJD)**; day 0 = explosion.

        :meth:`set_time_reference` with ``label="explosion"``, plus ``meta['explosion_mjd']``. When
        the date is only a stand-in for the explosion (the first detection, say), use
        :meth:`set_time_reference` so the plots do not call it the explosion.
        """
        out = self.set_time_reference(explosion_date, "explosion")
        out.meta["explosion_mjd"] = float(explosion_date)
        return out

    def explosion_time_prior(self, *, fallback_days=EXPLOSION_FALLBACK_DAYS):
        """The explosion-time prior the data define: from the last non-detection to the first detection.

        ``Uniform(last non-detection before the first detection, first detection)``, on this
        light curve's own clock (the clock a model's ``t_exp`` is on: MJD for a raw curve, days
        since the reference after :meth:`set_time_reference`). A fit of a model with a free
        explosion time (``free=["t_exp"]``) uses it by default; see Notes.

        Parameters
        ----------
        fallback_days : float, default :data:`EXPLOSION_FALLBACK_DAYS` (30)
            Width of the window when no non-detection precedes the first detection: the prior
            then runs from ``fallback_days`` before the first detection to it.

        Returns
        -------
        Uniform

        Raises
        ------
        ValueError
            The light curve has no detection ("not enough data"), or ``fallback_days <= 0``.

        Notes
        -----
        Every band counts: the first detection is the earliest detection in any band, the lower
        edge the latest non-detection, in any band, before it. A non-detection shallower than the
        first detection does not prove the source had not yet exploded, only that it was fainter
        than that limit; with shallow limits pass a wider prior of your own.

        In a fit (:func:`whisper_cbpf.fit` and every sampler's ``fit``), when the prior passed to
        the fit names no ``t_exp``: a model prior that is a ``Uniform`` is cut to this window (a
        fit never explores explosion times the data rule out, nor ones the model's own prior rules
        out); a model with no ``t_exp`` prior takes this one; any other family is kept as it is.
        The rows before the first detection are never fitted (they set this prior instead);
        ``result.info["t_exp_prior"]`` records the prior used and where it came from.

        Examples
        --------
        >>> import whisper_cbpf as wp
        >>> lc = wp.LightCurve(time=[60000.0, 60002.5, 60003.1, 60005.0],
        ...                    band=["lsstr", "lsstg", "lsstr", "lsstg"],
        ...                    magnitude=[24.5, 24.6, 23.2, 22.4],
        ...                    magnitude_err=[float("nan"), float("nan"), 0.12, 0.06],
        ...                    upper_limit=[True, True, False, False])
        >>> lc.explosion_time_prior()
        Uniform(60002.5, 60003.1)
        >>> lc.where(upper_limit=False).explosion_time_prior(fallback_days=20)
        Uniform(59983.1, 60003.1)
        """
        from ..priors import Uniform

        if not float(fallback_days) > 0.0:
            raise ValueError(f"fallback_days must be > 0; got {fallback_days!r}.")
        w = explosion_window(self.time, self.upper_limit)
        low = w["last_non_detection"]
        if low is None:
            low = w["first_detection"] - float(fallback_days)
        return Uniform(low, w["first_detection"])

    def calc_phase(self, reference=None, redshift=None, peak=False, hours=False):
        """Copy with a rest-frame ``phase`` column: ``(time - reference) / (1 + z)``.

        ``reference`` (MJD) defaults to ``meta['explosion_mjd']`` / ``meta['refmjd']`` / the earliest
        detection; a curve already shifted by :meth:`set_time_reference` or
        :meth:`set_explosion_date` is phased from its declared day 0 (``reference = 0``).
        ``redshift`` defaults to the curve's redshift (0 if unknown). With ``peak=True`` the
        reference is the brightest detection (``meta['peakdate']``); ``hours=True`` gives rest-frame
        hours. The reference and redshift used are recorded in ``meta``.
        """
        out = self.copy()
        z = redshift if redshift is not None else (out.redshift or 0.0)
        if reference is None:
            if peak:
                if "magnitude" not in out.colnames:
                    raise ValueError("peak=True needs a 'magnitude' column.")
                det = out.where(upper_limit=False) if "upper_limit" in out.colnames else out
                reference = float(det["time"].data[np.argmin(det["magnitude"].data)])
                out.meta["peakdate"] = reference
            elif "time_reference_mjd" in out.meta or "explosion_mjd" in out.meta:
                reference = 0.0   # set_time_reference already shifted time to days since day 0
            else:
                det = out.where(upper_limit=False) if "upper_limit" in out.colnames else out
                reference = float(np.min(det["time"].data))
        out.meta["refmjd"] = float(reference)
        out.meta["redshift_for_phase"] = float(z)
        phase = (out["time"].data - float(reference)) / (1.0 + float(z))
        out["phase"] = phase * 24.0 if hours else phase
        return out

    def calc_absmag(self, dm=None, redshift=None, ebv=None, rv=3.1, extinction=None):
        """Copy with an ``absmag`` column = ``magnitude - dm - A_band``.

        ``dm`` (distance modulus) defaults to ``5 log10(luminosity_distance) + 25`` if a luminosity
        distance is set, else ``Planck18.distmod(redshift)``. Milky-Way extinction ``A_band`` is taken
        from the ``extinction`` dict (``{band: A_mag}``) if given, otherwise computed per band from
        ``ebv`` / ``rv`` via the CCM89 law using each band's effective wavelength (no correction where
        the wavelength is unknown). The ``dm`` / ``ebv`` used are recorded in ``meta``.
        """
        if "magnitude" not in self.colnames:
            raise ValueError("calc_absmag needs a 'magnitude' column (call add_mag() first).")
        out = self.copy()
        z = redshift if redshift is not None else out.redshift
        if dm is None:
            ld = out.luminosity_distance
            if ld is not None:
                dm = 5.0 * np.log10(float(ld) * 1e6) - 5.0
            elif z:
                from astropy.cosmology import Planck18
                dm = float(Planck18.distmod(z).value)
            else:
                dm = 0.0
        out.meta["dm"] = float(dm)

        a_band = np.zeros(len(out), dtype=float)
        if extinction is not None:
            a_band = np.array([float(extinction.get(str(b), 0.0)) for b in out["band"].data])
            out.meta["extinction"] = dict(extinction)
        elif ebv:
            lam = out._resolved_lambda_eff()
            a_band = _ccm89_a_lambda(lam, float(ebv), float(rv))
            out.meta["ebv"] = float(ebv)
            out.meta["rv"] = float(rv)
        out["absmag"] = out["magnitude"].data - float(dm) - a_band
        return out

    def _resolved_lambda_eff(self):
        if "lambda_eff" in self.colnames:
            return self["lambda_eff"].data
        from .bands import resolve_bands
        lam, _, _ = resolve_bands(self["band"].data, svo_fallback=False, warn=False)
        return lam

    # --- export ---
    def to_dataframe(self):
        """Return a :class:`pandas.DataFrame` view (alias for astropy's ``to_pandas``)."""
        return self.to_pandas()

    def __call__(self):
        """Return the light curve itself (it *is* a table -- assign/compute columns directly)."""
        return self

    def __repr__(self):
        z = "unknown" if self.redshift is None else self.redshift
        return (f"LightCurve(name={self.name!r}, n_points={self.n_points}, "
                f"bands={self.bands}, mode={self.data_mode!r}, z={z})")


def _ccm89_a_lambda(lambda_eff_aa, ebv, rv):
    """Milky-Way extinction A(lambda) in mag from the Cardelli, Clayton & Mathis (1989) law.

    ``lambda_eff_aa`` is the effective wavelength per point (Angstrom; NaN -> no extinction). Valid for
    the optical/NIR (1.1 <= x <= 3.3 um^-1) and UV; outside that range it clamps to the nearest regime.
    """
    lam = np.asarray(lambda_eff_aa, dtype=float)
    x = 1.0 / (lam * 1e-4)                       # inverse microns
    finite = np.isfinite(x)
    # The CCM89 parametrisation is defined for 0.3 <= x <= 8 um^-1. Outside that (e.g. mid-IR / JWST
    # F356W, F444W), CLAMP x to the nearest valid edge so the nearest regime is used (as documented),
    # rather than silently returning A=0. Warn because the clamped value is an extrapolation.
    out_of_range = finite & ((x < 0.3) | (x > 8.0))
    if np.any(out_of_range):
        warnings.warn(f"CCM89: {int(out_of_range.sum())} band(s) outside the valid 0.3-8 um^-1 range; "
                      "clamping to the nearest regime (extinction there is approximate).", stacklevel=2)
    x = np.where(finite, np.clip(x, 0.3, 8.0), x)
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    # optical/NIR: 1.1 <= x <= 3.3
    opt = finite & (x >= 1.1) & (x <= 3.3)
    y = x[opt] - 1.82
    a[opt] = (1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + 0.72085 * y**4
              + 0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7)
    b[opt] = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4
              - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7)
    # near-IR: 0.3 <= x < 1.1
    nir = finite & (x >= 0.3) & (x < 1.1)
    a[nir] = 0.574 * x[nir]**1.61
    b[nir] = -0.527 * x[nir]**1.61
    # UV: 3.3 < x <= 8 (simplified Fa/Fb=0 base term)
    uv = finite & (x > 3.3) & (x <= 8.0)
    xu = x[uv]
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341)
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263)
    av = rv * ebv
    a_lambda = av * (a + b / rv)
    return np.where(finite, a_lambda, 0.0)
