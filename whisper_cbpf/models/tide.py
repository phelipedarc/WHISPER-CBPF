"""``tide`` -- the TiDE tidal-disruption model (Kovacs-Stermeczky & Vinko 2023), a CPU model.

TiDE is a C++ library (``libtide``) with a Python wrapper (``tidepy``). It is **not** ported to JAX:
whisper calls it. The model is therefore a plain CPU ``predict(params, times, bands) -> flux [Jy]``
that runs through the CPU samplers (``fit_MCMC``/emcee, nested/dynesty, ABC). It has no
``predict_jax``. Importing this module needs neither libtide nor tidepy; the library is loaded the
first time a light curve is computed, and a missing library raises an ``ImportError`` that says so.

**The physics.** The star's fall-back rate is Lodato, King & Pringle (2009)'s polytrope table
(``Mdotfb_calc="L09"``, tidepy's default) from the first fall-back at ``t_min`` on. Two components
radiate: a **wind** launched at ``R_L = R_g / eta`` with the fraction ``fout`` of the fall-back
mass, whose photosphere is a single blackbody (Strubbe & Quataert 2009; Lodato & Rossi 2011:
``T_ph = T_L (r_ph/R_L)^(-2/3) (fout/fv)^(1/3)``), and a thin accretion **disc** fed by the rest. The
library returns the monochromatic luminosity ``L_nu`` [erg/s/Hz] of each. ``fout="bifout"`` is
TiDE's built-in time-dependent outflow fraction (Lodato & Rossi 2011, eq. 28).

**Flux convention.** TiDE has no cosmology. With ``tau = (t - t_exp) / (1 + z)`` the source-frame
days since the disruption and ``d_L`` the Planck18 luminosity distance,

    F_nu(nu_obs, t) = (1 + z) L_nu(nu_obs (1 + z), tau) / (4 pi d_L^2),

converted to Jy. This is the convention of the model-comparison notebook the port was checked
against, and it is checked against an oracle that calls the unmodified library
(``tests/goldens/tide_oracle.npz``).

**Bands.** Each band is evaluated **monochromatically** at its effective wavelength (the
photon-weighted mean wavelength of the band's filter, ``band_wavelengths=`` overrides it per label)
by default, which is the convention the comparison used and costs one library call per epoch.
``n_nodes > 1`` integrates the real passband with that many Gauss nodes instead
(:mod:`whisper_cbpf.synphot`), at ``n_nodes`` times the cost.

**Time.** Epochs are observer-frame days; ``t_exp`` is the disruption on the same clock. TiDE's
light curve starts at ``t_min`` (the first fall-back, ``41.0076 d`` at ``beta_fulldisrupt = 1`` and
``6.4766 d`` at 1.85 for the 1e6 Msun, 1 Msun anchor). Before it the library returns small
non-physical numbers (the fall-back table is extrapolated), so an epoch with
``tau <= t_min (1 + 1e-4)`` is returned at **0 Jy**. After it there is no end: the fall-back table
is held at its last value, and the light curve is finite to at least 1e6 days.

**Persistent library state.** libtide never frees its objects and every ``Parameters`` object loads
an 11 MB table. The model therefore keeps ONE ``Parameters`` object per process and
configuration (:data:`ENGINES`), created lazily, and refreshes its fields for every parameter set
(0.3 ms, against 32 ms and 10.5 MB for a new object). Nothing from the library lives in the
:class:`TidePredict` instance, which pickles by value: a ``spawn`` worker builds its own.

**The two defaults of ``beta_fulldisrupt``.** tidepy's wrapper sets 1.85 (the beta of the full
disruption of a gamma = 4/3 polytrope); the C++ default and the command-line program use 1.0. The
first-fall-back time scales as ``beta_fulldisrupt^3`` (6.4766 d against 41.0076 d), so the choice
moves the whole light curve in time. This model exposes it as ``beta_fulldisrupt``, default **1.85**
(the tidepy default, the notebook's), and never overrides it silently. Likewise ``Mdotfb_calc`` is an
explicit option: ``tidepy.Parameters()`` fixes ``"L09"`` in its constructor and ignores a later
``Mdotfb_calc = "default"``, so this model builds its objects the way the notebook does
(``Parameters.__new__``) to let every value through.

**Diffusion is not offered.** libtide's diffusion step returns NaN at the first epoch and keeps the
diffusion timescale of a default parameter set; see the upstream issue notes.

References
----------
Kovacs-Stermeczky, Zs. V. & Vinko, J. 2023, PASP 135, 034102 (https://arxiv.org/abs/2302.08441)

Lodato, G., King, A. R. & Pringle, J. E. 2009, MNRAS 392, 332

Strubbe, L. E. & Quataert, E. 2009, MNRAS 400, 2070

Lodato, G. & Rossi, E. M. 2011, MNRAS 410, 359

Notes
-----
TiDE is MIT-licensed. This module does not contain or adapt any TiDE source: it calls the installed
library through ``tidepy``.
"""
from __future__ import annotations

import contextlib
import ctypes
import io
import os
import threading
import warnings
from pathlib import Path

import numpy as np

from ..priors import LogUniform, Prior, Uniform
from . import Model, register_model
from .cosmology import luminosity_distance_cm

__all__ = [
    "PARAMETERS", "FREE_CONTEXT", "DEFAULT_PINNED", "DEFAULT_FREE", "ENGINES", "TidePredict",
    "default_prior", "tide_model", "register_tide", "libtide_available", "tide_luminosity_nu",
    "tide_bolometric_luminosity", "tide_t_min", "tide_wind_photosphere",
]

#: The model's physical parameters, in this order, then the context values of :data:`FREE_CONTEXT`.
#: ``bh_M6``: black-hole mass [1e6 Msun]; ``star_mstar``: stellar mass [Msun]; ``bh_eta``:
#: radiative efficiency (sets the wind launch radius ``R_g / eta``); ``fout``: outflow fraction in
#: [0, 1] or ``"bifout"`` (pinned only); ``fv``: wind speed in units of the escape speed at the
#: launch radius (>= 1); ``eta_r``: reprocessing efficiency; ``beta_fulldisrupt``: beta of the full
#: disruption (U8).
PARAMETERS = ("bh_M6", "star_mstar", "bh_eta", "fout", "fv", "eta_r", "beta_fulldisrupt")
#: Context values a model can fit as ordinary parameters, appended in this order.
FREE_CONTEXT = ("t_exp", "redshift")
#: Pinned unless ``pin={name: None}`` frees them: no reprocessing, and tidepy's own beta.
DEFAULT_PINNED = {"eta_r": 0.0, "beta_fulldisrupt": 1.85}
#: The physical parameters a default model fits.
DEFAULT_FREE = ("bh_M6", "star_mstar", "bh_eta", "fout", "fv")

#: Fraction of ``t_min`` an epoch must exceed to be lit (the library's first points after ``t_min``
#: are a vertical rise from zero; the comparison notebook uses the same 1e-4).
FLOOR_REL = 1e-4
_C_AA = 2.99792458e18                   # speed of light [Angstrom/s]
_NU_T1, _NU_T2 = 3e14, 1.5e15           # the two frequencies the wind temperature is read from [Hz]
_NU_BOL = np.geomspace(1e13, 3e17, 90)  # default frequency grid of the bolometric luminosity [Hz]
_MAX_ENGINES = 8

#: Library objects of THIS process, keyed by configuration. Never pickled, never shared.
ENGINES: dict = {}
_LOCK = threading.RLock()
_TIDE = None

_INSTALL_HINT = (
    "The 'tide' model needs TiDE's C++ library (libtide) and its Python wrapper (tidepy). "
    "Neither is a pip dependency of whisper_cbpf: build libtide from the TiDE sources, set "
    "TIDE_LIB=<prefix>/lib and TIDE_PATH=<prefix> (the directory whose share/ holds table_n3.dat "
    "and table_n3per2.dat) and pip-install the python/tidepy package. The repository's "
    "docker/Dockerfile.tide builds an image that has all of it (container name tde_tide).")


# --- loading the library -------------------------------------------------------------------------

def _load_tide():
    """``tidepy.tide`` (the wrapper module, with ``lib``), imported once, quietly.

    Raises ``ImportError`` with :data:`_INSTALL_HINT` if tidepy or libtide is missing, or if
    ``TIDE_PATH`` does not hold the two fall-back tables (libtide would otherwise throw a C++
    exception on the first parameter set, which aborts the interpreter).
    """
    global _TIDE
    if _TIDE is None:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                from tidepy import tide as mod
        except (ImportError, OSError) as exc:
            raise ImportError(f"{_INSTALL_HINT} ({type(exc).__name__}: {exc})") from exc
        share = Path(os.environ.get("TIDE_PATH", "")) / "share"
        missing = [f for f in ("table_n3.dat", "table_n3per2.dat") if not (share / f).is_file()]
        if missing:
            raise ImportError(f"{_INSTALL_HINT} TIDE_PATH={os.environ.get('TIDE_PATH')!r} has no "
                              f"share/{missing}.")
        _TIDE = mod
    return _TIDE


def libtide_available():
    """Whether libtide and tidepy can be loaded in this process.

    Returns
    -------
    bool

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    Loads the library on the first call (quietly) and remembers the answer.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import libtide_available
    >>> isinstance(libtide_available(), bool)
    True
    """
    try:
        _load_tide()
    except ImportError:
        return False
    return True


# --- one persistent library object per process and configuration ---------------------------------

def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


class _Engine:
    """One tidepy ``Parameters`` + wind + disc per configuration, refreshed in place.

    Built the way the comparison notebook builds it: ``Parameters.__new__`` skips the wrapper's
    ``__init__``, which fixes ``Mdotfb_calc = "L09"`` before it can be set (U8). Changing the
    numeric ``fout`` goes through ``param_set_allparams`` with the calculation names ``"default"``,
    which leaves the fall-back, peak and ``t_min`` calculations as they are and only sets the
    value: ``refresh_all_params`` would reload and append the 11 MB table on every call.
    """

    def __init__(self, cfg, bifout):
        tide = _load_tide()
        self.lib = tide.lib
        self.cfg = cfg
        self.bifout = bool(bifout)
        mdotfb, tmin_calc, rstar, politrop, n_rings = cfg
        with _quiet():
            par = tide.Parameters.__new__(tide.Parameters)
            par.param = tide.lib.Cparameters()
            par.bh_M6, par.star_mstar, par.bh_eta = 1.0, 1.0, 0.1
            par.star_rstar, par.star_politrop = rstar, politrop
            par.fout = "bifout" if bifout else "0.5"
            par.fv, par.eta_r, par.d, par.i = 1.0, 0.0, 0.0, 0.0     # d = 0: L_nu, not F_nu
            par.diff_timescale, par.beta_fulldisrupt, par.N = "default", 1.85, int(n_rings)
            par.Mdotpeakcalc, par.Mdotfb_calc, par.tmin_calc = "default", mdotfb, tmin_calc
            par.nu, par.tend, par.dt, par.tstart = 6.3e14, 3000.0, 0.5, "default"
            par.refresh_all_params()
            par.set_tstart()
            self.par = par
            self.wind = tide.Wind_part_of_lc(par)
            self.disk = tide.Disk_part_of_lc(par)
        self._fout = None if bifout else 0.5
        self._last = None
        self.tmin = float(par.tstart_value)

    def configure(self, v):
        """Set the physical values ``v`` and return ``t_min`` [days]."""
        key = tuple(v[k] for k in ("bh_M6", "star_mstar", "bh_eta", "eta_r", "fv",
                                   "beta_fulldisrupt", "fout"))
        if key == self._last:
            return self.tmin
        par = self.par
        par.bh_M6, par.star_mstar, par.bh_eta = v["bh_M6"], v["star_mstar"], v["bh_eta"]
        par.eta_r, par.fv, par.beta_fulldisrupt = v["eta_r"], v["fv"], v["beta_fulldisrupt"]
        with _quiet():
            if not self.bifout and v["fout"] != self._fout:
                par.fout = repr(float(v["fout"]))
                self._set_all_keep_calculations()
                self._fout = float(v["fout"])
            else:
                par.refresh_params()
            par.set_tstart()
        self.tmin = float(par.tstart_value)
        self._last = key
        return self.tmin

    def _set_all_keep_calculations(self):
        """``refresh_all_params`` without re-creating the fall-back calculation (and its table)."""
        p, lib = self.par, self.lib
        lib.param_set_allparams(
            p.param, ctypes.c_double(p.bh_M6), ctypes.c_double(p.star_mstar),
            ctypes.c_char_p(str(p.star_rstar).encode()), ctypes.c_char_p(p.star_politrop.encode()),
            ctypes.c_double(p.bh_eta), ctypes.c_double(p.eta_r), ctypes.c_char_p(p.fout.encode()),
            ctypes.c_double(p.fv), ctypes.c_double(p.d), ctypes.c_double(p.i),
            ctypes.c_char_p(p.diff_timescale.encode()), ctypes.c_double(p.beta_fulldisrupt),
            ctypes.c_int(p.N), ctypes.c_char_p(b"default"), ctypes.c_char_p(b"default"),
            ctypes.c_char_p(b"default"), ctypes.c_double(p.nu))
        lib.param_refresh(p.param)

    def lnu(self, tau, nu):
        """``(L_wind, L_disc)`` [erg/s/Hz] at ``tau`` days since the disruption and ``nu`` [Hz]."""
        with _quiet():
            return (self.wind.wind_luminosity_at_time_nu(float(tau), float(nu)),
                    self.disk.disk_luminosity_at_time_nu(float(tau), float(nu)))


def _engine(cfg, bifout):
    """This process's engine for ``cfg`` (created on first use)."""
    key = (cfg, bool(bifout))
    eng = ENGINES.get(key)
    if eng is None:
        if len(ENGINES) >= _MAX_ENGINES:
            warnings.warn(
                f"{len(ENGINES) + 1} TiDE configurations in one process: libtide never frees its "
                f"objects (about 10.5 MB each), so each new configuration keeps its memory.",
                RuntimeWarning, stacklevel=3)
        eng = ENGINES[key] = _Engine(cfg, bifout)
    return eng


def _config(Mdotfb_calc, tmin_calc, star_rstar, star_politrop, N=1000):
    for name, value, allowed in (("Mdotfb_calc", Mdotfb_calc, ("L09", "L09_const", "default")),
                                 ("tmin_calc", tmin_calc, ("default", "with_rp", "GR13")),
                                 ("star_politrop", star_politrop, ("default", "4per3", "5per3"))):
        if value not in allowed:
            raise ValueError(f"{name}={value!r}: TiDE accepts {allowed}.")
    if int(N) < 2:
        raise ValueError(f"N={N}: the disc integration needs at least 2 rings.")
    return (str(Mdotfb_calc), str(tmin_calc), str(star_rstar), str(star_politrop), int(N))


# --- parameters and priors -----------------------------------------------------------------------

def _distributions():
    return {
        "bh_M6": LogUniform(0.1, 30.0),
        "star_mstar": LogUniform(0.1, 10.0),
        "bh_eta": LogUniform(0.01, 0.4),
        "fout": Uniform(0.05, 1.0),
        "fv": Uniform(1.0, 3.0),
        "eta_r": Uniform(0.0, 0.5),
        "beta_fulldisrupt": Uniform(0.9, 2.0),
    }


def default_prior(names=None):
    """The default prior box of the TiDE parameters.

    Parameters
    ----------
    names : sequence of str, optional
        The parameters to include (default: every name in :data:`PARAMETERS`).

    Returns
    -------
    whisper_cbpf.priors.Prior
        ``bh_M6`` LogUniform(0.1, 30) (black holes of 1e5 to 3e7 Msun), ``star_mstar`` LogUniform(0.1,
        10) Msun, ``bh_eta`` LogUniform(0.01, 0.4), ``fout`` Uniform(0.05, 1), ``fv`` Uniform(1, 3),
        ``eta_r`` Uniform(0, 0.5), ``beta_fulldisrupt`` Uniform(0.9, 2).

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    TiDE publishes no prior; this box is whisper's. It spans the black-hole masses of observed TDE
    hosts and main-sequence stars, with the efficiency range of a thin disc, ``fv >= 1`` (the wind
    cannot be slower than the escape speed) and ``fout`` away from 0, where the wind vanishes. The
    library gave a finite light curve at every one of 48 Latin-hypercube draws and the two box
    corners tried; the box is a statement of where it was exercised, not of where TiDE is valid.
    ``t_exp`` and ``redshift`` have no default: their windows depend on the data.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import default_prior
    >>> sorted(default_prior(["bh_M6", "fv"]).names)
    ['bh_M6', 'fv']
    """
    d = _distributions()
    names = list(PARAMETERS if names is None else names)
    unknown = [n for n in names if n not in d]
    if unknown:
        raise KeyError(f"no default prior for {unknown}; the parameters are {list(PARAMETERS)}.")
    return Prior({n: d[n] for n in names})


# --- the predict callable ------------------------------------------------------------------------

_NODE_CACHE: dict = {}      # (label, n_nodes, default_system) -> (lambda [A], weights); per process


def _nodes(label, n_nodes, band_wavelengths, default_system, aliases):
    """``(wavelength [Angstrom] nodes, weights summing to 1)`` of one band label."""
    raw = str(label).strip()
    if raw in band_wavelengths:
        return np.array([float(band_wavelengths[raw])]), np.ones(1)
    try:                                       # a bare number is a wavelength in Angstrom
        return np.array([float(raw)]), np.ones(1)
    except ValueError:
        pass
    key = (raw, int(n_nodes), default_system, tuple(sorted((aliases or {}).items())))
    hit = _NODE_CACHE.get(key)
    if hit is None:
        from ..synphot import filter_set_for, resolve_filter

        name = resolve_filter(raw, aliases, default_system)
        fs = filter_set_for([name], n_nodes=None if n_nodes == 1 else int(n_nodes))
        lam, w = fs.nodes[0], fs.weights[0]
        if n_nodes == 1:
            lam, w = np.array([float((w * lam).sum() / w.sum())]), np.ones(1)
        else:
            w = w / w.sum()
        hit = _NODE_CACHE[key] = (lam, w)
    return hit


class TidePredict:
    """A bound TiDE model as WHISPER's ``predict``: ``predict(parameters, times, bands) -> Jy``.

    A module-level class of plain data (strings, numbers, dicts), so it pickles by value and a
    ``spawn`` worker rebuilds the library state on first use. Build one through :func:`tide_model`;
    the constructor does no validation.

    Parameters
    ----------
    free : tuple of str
        The physical parameters read from ``parameters`` (a subset of :data:`PARAMETERS`).
    pinned : dict
        The other physical parameters' values (``fout`` may be ``"bifout"`` here).
    free_context : tuple of str
        ``"t_exp"`` and/or ``"redshift"``, read from ``parameters``.
    redshift, dl_cm : float or None
        The fixed redshift and its luminosity distance [cm] (``dl_cm=None``: Planck18 at the
        redshift).
    t_exp_days : float
        The disruption time on the epochs' clock when ``t_exp`` is not free.
    config : tuple
        ``(Mdotfb_calc, tmin_calc, star_rstar, star_politrop, N)``; ``N`` is the number of disc
        rings of the library's integration, which libtide reads when its disc object is created
        (setting it later has no effect), so it is part of the configuration.
    n_nodes : int
        1: monochromatic at the band's effective wavelength; more: Gauss nodes over the passband.
    band_wavelengths : dict
        ``{label: Angstrom}`` overriding the filter-derived effective wavelength.
    default_system, band_aliases
        As :func:`whisper_cbpf.synphot.resolve_filter`.

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    Parameter sets outside the library's domain (non-finite values, ``bh_M6``, ``star_mstar`` or
    ``bh_eta`` not positive, ``fout`` outside [0, 1], ``fv < 1``, ``eta_r`` outside [0, 1]) give
    zero flux at every epoch, which a likelihood rejects. Not thread-safe across threads that use
    different parameter sets at once: calls take a per-process lock.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import tide_model
    >>> m = tide_model(["4686"], redshift=0.05)            # a bare number is Angstrom
    >>> p = dict(bh_M6=1.0, star_mstar=1.0, bh_eta=0.1, fout=0.3, fv=1.0)   # doctest: +SKIP
    >>> m.predict(p, [60.0, 120.0], ["4686", "4686"])      # doctest: +SKIP
    """

    def __init__(self, free, pinned, free_context=(), redshift=None, dl_cm=None, t_exp_days=0.0,
                 config=("L09", "default", "ms", "default", 1000), n_nodes=1,
                 band_wavelengths=None, default_system=None, band_aliases=None):
        self.free = tuple(free)
        self.pinned = dict(pinned)
        self.free_context = tuple(free_context)
        self.redshift = None if redshift is None else float(redshift)
        self.dl_cm = None if dl_cm is None else float(dl_cm)
        self.t_exp_days = float(t_exp_days)
        self.config = tuple(config)
        self.n_nodes = int(n_nodes)
        self.band_wavelengths = {str(k): float(v) for k, v in dict(band_wavelengths or {}).items()}
        self.default_system = default_system
        self.band_aliases = dict(band_aliases or {})

    # -- parameter handling
    def values(self, parameters):
        """The physical values (a dict over :data:`PARAMETERS`), or ``None`` if invalid."""
        v = dict(self.pinned)
        for k in self.free:
            v[k] = parameters[k]
        try:
            nums = {k: float(x) for k, x in v.items() if not isinstance(x, str)}
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(x) for x in nums.values()):
            return None
        v.update(nums)
        bif = isinstance(v["fout"], str)
        if bif and v["fout"] != "bifout":
            raise ValueError(f"fout={v['fout']!r}: a number in [0, 1] or 'bifout'.")
        ok = (v["bh_M6"] > 0 and v["star_mstar"] > 0 and 0 < v["bh_eta"] <= 1 and v["fv"] >= 1
              and 0 <= v["eta_r"] <= 1 and v["beta_fulldisrupt"] > 0 and (bif or 0 <= v["fout"] <= 1))
        return v if ok else None

    def _engine_for(self, v):
        return _engine(self.config, isinstance(v["fout"], str))

    def _context(self, parameters):
        t_exp = float(parameters["t_exp"]) if "t_exp" in self.free_context else self.t_exp_days
        z = float(parameters["redshift"]) if "redshift" in self.free_context else self.redshift
        dl = self.dl_cm if (self.dl_cm is not None and "redshift" not in self.free_context) \
            else float(luminosity_distance_cm(z))
        return t_exp, z, dl

    # -- the model
    def luminosity_nu(self, parameters, tau_days, nu_hz):
        """``(L_wind, L_disc, t_min)``: source-frame ``L_nu`` [erg/s/Hz] at ``tau_days`` (days since
        the disruption) and ``nu_hz`` (rest-frame Hz), broadcast against each other; zero up to
        ``t_min (1 + FLOOR_REL)``, and zero with ``t_min = nan`` for an invalid parameter set."""
        tau, nu = np.broadcast_arrays(np.asarray(tau_days, float), np.asarray(nu_hz, float))
        lw, ld = np.zeros(tau.shape), np.zeros(tau.shape)
        v = self.values(parameters)
        if v is None:
            return lw, ld, np.nan
        with _LOCK:
            eng = self._engine_for(v)
            tmin = eng.configure(v)
            if not np.isfinite(tmin) or tmin <= 0:
                return lw, ld, tmin
            for idx in np.flatnonzero(tau.ravel() > tmin * (1.0 + FLOOR_REL)):
                lw.flat[idx], ld.flat[idx] = eng.lnu(tau.flat[idx], nu.flat[idx])
        return lw, ld, tmin

    def __call__(self, parameters, times, bands=None):
        times = np.asarray(times, dtype=float)
        if bands is None:
            raise ValueError("the 'tide' model is band-dependent; `bands` is required.")
        bands = np.asarray(bands)
        if bands.shape != times.shape:
            raise ValueError(f"`bands` and `times` must have the same shape, got {bands.shape} and "
                             f"{times.shape}")
        out = np.zeros(times.shape, dtype=float)
        if times.size == 0:
            return out
        v = self.values(parameters)
        if v is None:
            return out
        t_exp, z, dl = self._context(parameters)
        tau = (times - t_exp) / (1.0 + z)
        nodes = {}
        for label in np.unique(bands):
            nodes[label] = _nodes(label, self.n_nodes, self.band_wavelengths, self.default_system,
                                  self.band_aliases)
        with _LOCK:
            eng = self._engine_for(v)
            tmin = eng.configure(v)
            if not np.isfinite(tmin) or tmin <= 0:
                return out
            lit = np.flatnonzero(tau.ravel() > tmin * (1.0 + FLOOR_REL))
            flat, b = out.ravel(), bands.ravel()
            for i in lit:
                lam, w = nodes[b[i]]
                total = 0.0
                for lk, wk in zip(lam, w):
                    lw, ld = eng.lnu(tau.flat[i], _C_AA / lk * (1.0 + z))
                    total += wk * (lw + ld)
                flat[i] = 1e23 * (1.0 + z) * total / (4.0 * np.pi * dl ** 2)
        return np.where(np.isfinite(out), out, 0.0)

    def __repr__(self):
        return (f"TidePredict(free={list(self.free)}, pinned={self.pinned}, "
                f"context={list(self.free_context)}, config={self.config}, n_nodes={self.n_nodes})")


# --- the factory ---------------------------------------------------------------------------------

def _check_free(free):
    free = () if free is None else ([free] if isinstance(free, str) else tuple(free))
    bad = [f for f in free if f not in FREE_CONTEXT]
    if bad:
        raise ValueError(f"free={list(free)} names {bad}. Only 't_exp' (the disruption time) and "
                         f"'redshift' can be freed; the model's own parameters are free already.")
    return tuple(k for k in FREE_CONTEXT if k in free)


def tide_model(band_names=None, redshift=None, dl_cm=None, *, name="tide", free=None, pin=None,
               prior=None, redshift_prior=None, t_exp_days=0.0, beta_fulldisrupt=1.85,
               Mdotfb_calc="L09", tmin_calc="default", star_rstar="ms", star_politrop="default",
               N=1000, n_nodes=1, band_wavelengths=None, band_aliases=None, default_system=None):
    """Bind dataset context to TiDE and return a whisper ``Model`` (CPU, no ``predict_jax``).

    Same call shape as the JAX TDE factories, so a comparison can bind it to a light curve:
    ``band_names`` first, the fixed redshift and distance, ``free=`` for ``t_exp``/``redshift``,
    ``pin=`` for the physical parameters.

    Parameters
    ----------
    band_names : sequence of str, optional
        Band labels the light curve will use. Resolved now (a label that names no filter raises
        here, not inside a likelihood); ``None`` skips the check.
    redshift, dl_cm : float
        Fixed redshift and luminosity distance [cm]. ``dl_cm=None`` takes Planck18's. Leave both
        ``None`` when the redshift is fitted (``free=["redshift"]``).
    name : str
        The model's name (default ``"tide"``).
    free : list of str or None
        ``"t_exp"`` and/or ``"redshift"`` fitted as parameters, appended in that order. ``t_exp``
        needs ``prior=Prior({"t_exp": ...})`` on the light curve's clock.
    pin : dict or None
        ``{name: value}`` holds a physical parameter fixed (``{"fout": "bifout"}`` selects TiDE's
        built-in outflow fraction); ``{name: None}`` frees one pinned by default (``eta_r``,
        ``beta_fulldisrupt``).
    prior : Prior or None
        Distributions overriding :func:`default_prior`'s, per parameter.
    redshift_prior : distribution or None
        The fitted redshift's prior when ``prior`` has none.
    t_exp_days : float
        Disruption time on the epochs' clock when ``t_exp`` is not free (default 0: epochs are days
        since the disruption).
    beta_fulldisrupt : float
        Beta of the full disruption: **1.85** (tidepy's default, the notebook's) or 1.0 (libtide's
        default); ``t_min`` scales as its cube (U8). Pinned; free it with ``pin={"beta_fulldisrupt":
        None}``.
    Mdotfb_calc : {"L09", "L09_const", "default"}
        Fall-back rate: Lodato et al. (2009) tables, their constant-density variant, or the
        classical power law. Honoured as given (tidepy's constructor would override ``"default"``).
    tmin_calc : {"default", "with_rp", "GR13"}
        Time of first fall-back: from the tidal radius (``R_t / beta_fulldisrupt``), with the
        pericentre, or from Guillochon & Ramirez-Ruiz (2013).
    star_rstar : str or float
        ``"ms"`` main-sequence radius law, ``"wd"`` white dwarf, or a radius in Rsun.
    star_politrop : {"default", "4per3", "5per3"}
        Polytrope; ``"default"`` follows the star type.
    N : int
        Rings of the disc integration (default 1000, the library's). libtide reads it when its disc
        object is created, so it is fixed per configuration, not refreshed.
    n_nodes : int
        1 (default): each band monochromatic at its effective wavelength; more: that many Gauss
        nodes over the passband, at that multiple of the cost.
    band_wavelengths : dict or None
        ``{label: Angstrom}``; overrides the filter-derived effective wavelength. A label that is
        a number is read as Angstrom.
    band_aliases, default_system
        As :func:`whisper_cbpf.synphot.resolve_filter`.

    Returns
    -------
    Model
        ``parameters`` ``["bh_M6", "star_mstar", "bh_eta", "fout", "fv"]`` by default (plus the
        freed context values); ``default_prior`` :func:`default_prior`; flux density in Jy.

    Raises
    ------
    ValueError
        An option outside its list, a context argument given both fixed and free, or a freed
        ``t_exp`` without a prior.
    ImportError
        Only when a light curve is computed and libtide is missing.

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102 (https://arxiv.org/abs/2302.08441); Lodato,
    King & Pringle 2009, MNRAS 392, 332; Strubbe & Quataert 2009, MNRAS 400, 2070; Lodato & Rossi
    2011, MNRAS 410, 359.

    Notes
    -----
    One call costs about 0.06 ms (wind) plus 0.27 ms (disc) per epoch and node, plus 0.3 ms to
    refresh the parameters; see ``reports/tide_cpu_cost.json`` of the porting notes for the
    scaling with epochs and processes. An epoch at or before ``t_min`` is at 0 Jy.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import tide_model
    >>> m = tide_model(["g", "r"], redshift=0.05, pin={"fout": "bifout"})
    >>> m.parameters
    ['bh_M6', 'star_mstar', 'bh_eta', 'fv']
    >>> tide_model(redshift=0.05, name="tide_z").default_prior.names
    ['bh_M6', 'star_mstar', 'bh_eta', 'fout', 'fv']
    """
    free = _check_free(free)
    config = _config(Mdotfb_calc, tmin_calc, star_rstar, star_politrop, N)
    pins = dict(DEFAULT_PINNED, beta_fulldisrupt=float(beta_fulldisrupt))
    for key, value in dict(pin or {}).items():
        if key not in PARAMETERS:
            raise ValueError(f"pin names {key!r}, which is not a TiDE parameter {list(PARAMETERS)}.")
        if value is None:
            pins.pop(key, None)
        else:
            pins[key] = value if isinstance(value, str) else float(value)
    pinned = {k: v for k, v in pins.items() if k in PARAMETERS}
    physical = [k for k in PARAMETERS if k not in pinned]
    if "fout" in pinned and isinstance(pinned["fout"], str) and pinned["fout"] != "bifout":
        raise ValueError(f"pin fout={pinned['fout']!r}: a number in [0, 1] or 'bifout'.")

    if "redshift" in free:
        if redshift is not None or dl_cm is not None:
            raise ValueError("redshift/dl_cm fix the redshift, and free=['redshift'] fits it: pass one.")
    elif redshift is None:
        raise ValueError("redshift is required unless it is fitted (free=['redshift']).")

    dists = _distributions()
    given = dict(prior.distributions) if isinstance(prior, Prior) else dict(prior or {})
    stray = [k for k in FREE_CONTEXT if k in given and k not in free]
    if stray:
        raise ValueError(f"prior= names {stray}, which this model holds fixed. Add them to free=.")
    unknown = [k for k in given if k not in PARAMETERS and k not in FREE_CONTEXT]
    if unknown:
        raise ValueError(f"prior= names {unknown}, which are not parameters of this model "
                         f"({list(PARAMETERS) + list(FREE_CONTEXT)}).")
    for k in given:
        if k in pinned:
            raise ValueError(f"prior= names {k!r}, which is pinned; free it with pin={{'{k}': None}}.")
    dists.update({k: v for k, v in given.items() if k in PARAMETERS})
    parameters = list(physical)
    out_prior = {k: dists[k] for k in physical}
    if "t_exp" in free:
        if "t_exp" not in given:
            raise ValueError("free=['t_exp'] needs prior=Prior({'t_exp': Uniform(...)}) on the light "
                             "curve's own clock, e.g. Uniform(t_first - 60, t_first); the window "
                             "depends on the data.")
        parameters.append("t_exp")
        out_prior["t_exp"] = given["t_exp"]
    if "redshift" in free:
        if "redshift" in given:
            dist = given["redshift"]
        elif redshift_prior is not None:
            from ..io.schema import redshift_distribution
            dist = redshift_distribution(redshift_prior)
        else:
            from ..io.schema import DEFAULT_REDSHIFT_PRIOR, redshift_distribution
            dist = redshift_distribution(DEFAULT_REDSHIFT_PRIOR)
        parameters.append("redshift")
        out_prior["redshift"] = dist

    predict = TidePredict(physical, pinned, free, redshift, dl_cm, t_exp_days, config, n_nodes,
                          band_wavelengths, default_system, band_aliases)
    if band_names is not None:
        for label in band_names:             # fail here, not inside a likelihood
            _nodes(label, predict.n_nodes, predict.band_wavelengths, default_system, band_aliases)
    pin_note = "; pinned " + ", ".join(f"{k}={v}" for k, v in pinned.items())
    description = (f"TiDE (libtide, CPU): wind + thin-disc tidal-disruption light curve, "
                   f"Mdotfb={config[0]}, tmin={config[1]}{pin_note}; flux density (Jy), "
                   f"{'monochromatic at the band effective wavelength' if n_nodes == 1 else f'{n_nodes}-node band integral'}"
                   f", F_nu = (1+z) L_nu(nu(1+z), t/(1+z)) / (4 pi d_L^2); 0 Jy before t_min.")
    return Model(name=str(name), predict=predict, parameters=parameters,
                 default_prior=Prior(out_prior), description=description)


def register_tide(band_names=None, redshift=None, dl_cm=None, *, name="tide", **kwargs):
    """Build a TiDE ``Model`` with :func:`tide_model` and register it under ``name``.

    Parameters
    ----------
    band_names, redshift, dl_cm, **kwargs
        As :func:`tide_model`.
    name : str, default "tide"
        The registry name (an existing one is replaced).

    Returns
    -------
    Model
        The registered model: ``fit_MCMC``/``nested``/``abc`` find it by name. CPU samplers only; its
        ``predict_jax`` is ``None``.

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    Registering needs no library: the library is loaded at the first ``predict``.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.tide import register_tide
    >>> m = register_tide(["g", "r"], redshift=0.05, name="tide_example")
    >>> "tide_example" in wp.list_models() and m.predict_jax is None
    True
    """
    m = tide_model(band_names, redshift, dl_cm, name=name, **kwargs)
    return register_model(name, m.predict, m.parameters, prior=m.default_prior,
                          description=m.description, overwrite=True)


# --- the library's own quantities ----------------------------------------------------------------

def _physical(bh_M6, star_mstar, bh_eta, fout, fv, eta_r, beta_fulldisrupt):
    pinned = dict(bh_M6=bh_M6, star_mstar=star_mstar, bh_eta=bh_eta, fout=fout, fv=fv, eta_r=eta_r,
                  beta_fulldisrupt=beta_fulldisrupt)
    pinned = {k: (v if isinstance(v, str) else float(v)) for k, v in pinned.items()}
    return TidePredict((), pinned)


_ARGS = dict(bh_M6=1.0, star_mstar=1.0, bh_eta=0.1, fout="bifout", fv=1.0, eta_r=0.0,
             beta_fulldisrupt=1.85, Mdotfb_calc="L09", tmin_calc="default", star_rstar="ms",
             star_politrop="default", N=1000)


def _bound(kw):
    unknown = set(kw) - set(_ARGS)
    if unknown:
        raise TypeError(f"unexpected keyword(s) {sorted(unknown)}; TiDE takes {sorted(_ARGS)}.")
    a = dict(_ARGS, **kw)
    p = _physical(a["bh_M6"], a["star_mstar"], a["bh_eta"], a["fout"], a["fv"], a["eta_r"],
                  a["beta_fulldisrupt"])
    p.config = _config(a["Mdotfb_calc"], a["tmin_calc"], a["star_rstar"], a["star_politrop"], a["N"])
    return p


def tide_luminosity_nu(tau_days, nu_hz, **kwargs):
    """TiDE's wind and disc monochromatic luminosities ``L_nu`` [erg/s/Hz] (source frame).

    Parameters
    ----------
    tau_days : float or array
        Days since the disruption (source frame).
    nu_hz : float or array
        Rest-frame frequency [Hz], broadcast against ``tau_days``.
    **kwargs
        ``bh_M6=1``, ``star_mstar=1``, ``bh_eta=0.1``, ``fout="bifout"``, ``fv=1``, ``eta_r=0``,
        ``beta_fulldisrupt=1.85``, ``Mdotfb_calc="L09"``, ``tmin_calc="default"``,
        ``star_rstar="ms"``, ``star_politrop="default"``, ``N=1000`` (see :func:`tide_model`).

    Returns
    -------
    (wind, disc) : tuple of ndarray
        Zero at ``tau_days <= t_min (1 + 1e-4)``. Face-on, isotropic-equivalent (``d = 0``,
        ``i = 0`` in the library).

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    Uses the same persistent library object as :class:`TidePredict`.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import tide_luminosity_nu
    >>> wind, disc = tide_luminosity_nu([30.0, 300.0], 6e14)    # doctest: +SKIP
    >>> bool(wind[0] > wind[1])                                 # doctest: +SKIP
    True
    """
    lw, ld, _ = _bound(kwargs).luminosity_nu({}, tau_days, nu_hz)
    return lw, ld


def tide_t_min(**kwargs):
    """The first-fall-back time ``t_min`` [days] the library computes for these parameters.

    Parameters
    ----------
    **kwargs
        As :func:`tide_luminosity_nu`.

    Returns
    -------
    float
        ``41.0076`` at ``beta_fulldisrupt=1`` and ``6.4766`` at 1.85 for the 1e6 Msun, 1 Msun anchor.

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    ``t_min`` scales as ``beta_fulldisrupt^3`` (``1.85^3 = 6.33``): the tidal radius enters as
    ``R_t / beta_fulldisrupt``.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import tide_t_min
    >>> round(tide_t_min(beta_fulldisrupt=1.0), 4)     # doctest: +SKIP
    41.0076
    """
    return float(_bound(kwargs).luminosity_nu({}, [1.0], 1e15)[2])


def tide_bolometric_luminosity(tau_days, nu_grid=None, **kwargs):
    """Bolometric wind and disc luminosities [erg/s]: ``L_nu`` integrated over frequency.

    Parameters
    ----------
    tau_days : array
        Days since the disruption (source frame).
    nu_grid : array, optional
        Frequencies [Hz] of the trapezoid (default ``geomspace(1e13, 3e17, 90)``).
    **kwargs
        As :func:`tide_luminosity_nu`.

    Returns
    -------
    (wind, disc) : tuple of ndarray

    References
    ----------
    Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    90 frequencies per epoch: about 30 ms an epoch. The default grid is the comparison notebook's.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import tide_bolometric_luminosity
    >>> wind, disc = tide_bolometric_luminosity([40.0])    # doctest: +SKIP
    >>> bool(wind[0] > 1e43)                               # doctest: +SKIP
    True
    """
    grid = _NU_BOL if nu_grid is None else np.asarray(nu_grid, float)
    tau = np.atleast_1d(np.asarray(tau_days, float))
    p = _bound(kwargs)
    t2 = np.repeat(tau[:, None], grid.size, axis=1)
    lw, ld, _ = p.luminosity_nu({}, t2, np.broadcast_to(grid, t2.shape))
    return np.trapezoid(lw, grid, axis=1), np.trapezoid(ld, grid, axis=1)


def tide_wind_photosphere(tau_days, **kwargs):
    """The wind photosphere ``(T [K], R [cm])`` read from the library's wind ``L_nu``.

    Parameters
    ----------
    tau_days : array
        Days since the disruption (source frame).
    **kwargs
        As :func:`tide_luminosity_nu`.

    Returns
    -------
    (T, R) : tuple of ndarray
        NaN where the wind has no light or its temperature is outside 2e3 to 5e6 K.

    References
    ----------
    Strubbe & Quataert 2009, MNRAS 400, 2070; Kovacs-Stermeczky & Vinko 2023, PASP 135, 034102.

    Notes
    -----
    tidepy does not return the photosphere. The wind is one blackbody, ``L_nu = 4 pi^2 R^2 B_nu(T)``,
    so ``T`` follows from ``L_nu`` at 3e14 and 1.5e15 Hz and ``R`` from the amplitude; the
    residual at a third frequency was below 1.5e-13 in every one of the 59 parameter sets of the
    oracle.

    Examples
    --------
    >>> from whisper_cbpf.models.tide import tide_wind_photosphere
    >>> T, R = tide_wind_photosphere([30.0, 300.0])    # doctest: +SKIP
    >>> bool(T[1] > T[0] and R[1] < R[0])              # doctest: +SKIP
    True
    """
    from scipy import constants as sc
    from scipy.optimize import brentq

    tau = np.atleast_1d(np.asarray(tau_days, float))
    p = _bound(kwargs)
    nus = np.array([_NU_T1, _NU_T2])
    T, R = np.full(tau.shape, np.nan), np.full(tau.shape, np.nan)

    def planck(Tk, nu):
        x = np.clip(sc.h * nu / (sc.k * Tk), 1e-8, 700.0)
        return 2 * sc.h * nu ** 3 / sc.c ** 2 / np.expm1(x)

    for j, t in enumerate(tau):
        lw, _, _ = p.luminosity_nu({}, np.full(2, t), nus)
        if not (lw[0] > 0 and lw[1] > 0):
            continue

        def g(Tk, lw=lw):
            return np.log(planck(Tk, nus[1]) / planck(Tk, nus[0])) - np.log(lw[1] / lw[0])

        if g(2e3) * g(5e6) >= 0:
            continue
        T[j] = brentq(g, 2e3, 5e6, xtol=1e-9, rtol=1e-13)
        R[j] = 100.0 * np.sqrt(lw[0] / (1e7 * 4 * np.pi ** 2 * planck(T[j], nus[0])))
    return T, R
