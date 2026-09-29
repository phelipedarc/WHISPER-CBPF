"""Simulated LSST alerts for the science-validation suite.

Everything a science test needs to make data whose answer is known:

- **Cadence.** Nights separated by ``gap`` days, two visits per night in two different filters
  (the LSST wide-fast-deep pattern); a kilonova follow-up uses nightly visits.
- **Depths.** LSST single-visit 5-sigma depths (Ivezic et al. 2019, ApJ 873, 111, table 2), with a
  per-visit scatter of 0.2 mag for seeing and sky.
- **Noise.** LSST's photometric error model (Ivezic et al. 2019, eqs. 4-5): background-limited
  for a faint source, source noise and a 0.005 mag floor for a bright one. The measured flux is
  the true flux plus a Gaussian draw of that width, and the alert reports that width.
- **Alerts.** A visit whose measured flux is at least 5 sigma is a diaSource; every visit also has
  forced photometry. :func:`alert_packet` writes the Rubin alert packet (``diaSource``,
  ``prvDiaSources``, ``prvDiaForcedSources``), so the tests read it through
  ``load_lightcurve(..., survey="lsst")`` exactly as a user reads a broker's alert.

With this noise model the censored flux likelihood whisper fits by default (detections as
Gaussians, a non-detection as ``P(flux < 5 sigma)``) is the exact likelihood of the data, so a
calibrated sampler must reach nominal coverage.
"""
from __future__ import annotations

import math

import numpy as np

#: LSST single-visit 5-sigma depths (AB mag; Ivezic et al. 2019, table 2).
LSST_M5 = {"lsstu": 23.78, "lsstg": 24.81, "lsstr": 24.35, "lssti": 23.92, "lsstz": 23.34,
           "lssty": 22.45}
#: ZTF median single-visit 5-sigma depths (Bellm et al. 2019, PASP 131, 018002).
ZTF_M5 = {"ztfg": 20.8, "ztfr": 20.6, "ztfi": 19.9}
M5 = {**LSST_M5, **ZTF_M5}
#: The filters the simulated wide-fast-deep visits use.
WFD_BANDS = ("lsstg", "lsstr", "lssti", "lsstz")
#: AB zero point of a flux in nJy.
ZP_NJY = 31.4
#: Detection threshold (sigma) of a diaSource.
DETECTION_SIGMA = 5.0
#: LSST error model: systematic floor (mag) and gamma (Ivezic et al. 2019, table 2).
SIGMA_SYS = 0.005
GAMMA = 0.039
#: Epoch of the simulated explosions (MJD, TAI); the surveys run around it.
MJD_EVENT = 61000.0


def lsst_cadence(rng, t_start, t_end, *, bands=WFD_BANDS, gap=(2.0, 4.0), per_night=2,
                 depth_scatter=0.2):
    """Visit epochs, filters and 5-sigma depths between ``t_start`` and ``t_end`` (days).

    Returns
    -------
    times, bands, m5 : numpy arrays
    """
    times, bnds, m5 = [], [], []
    t = t_start + rng.uniform(0.0, gap[1])
    while t < t_end:
        chosen = rng.choice(len(bands), size=min(per_night, len(bands)), replace=False)
        for j, k in enumerate(chosen):
            b = bands[k]
            times.append(t + 0.021 * j)            # the second visit ~30 min later
            bnds.append(b)
            m5.append(M5[b] + rng.normal(0.0, depth_scatter))
        t += rng.uniform(*gap)
    return np.asarray(times), np.asarray(bnds), np.asarray(m5)


def flux_error(true_njy, m5):
    """The 1-sigma flux error (nJy) of a visit: LSST's photometric error model.

    ``sigma_m^2 = sigma_sys^2 + (0.04 - gamma) x + gamma x^2`` with ``x = 10^(0.4 (m - m5))``,
    ``gamma = 0.039`` and ``sigma_sys = 0.005`` mag (Ivezic et al. 2019, eqs. 4-5), turned into flux.
    A faint source is background-limited (about a fifth of the 5-sigma flux); a bright one reaches
    the 0.005 mag floor.
    """
    f5 = 10.0 ** (-0.4 * (np.asarray(m5, dtype=float) - ZP_NJY))
    f = np.maximum(np.asarray(true_njy, dtype=float), 1e-3 * f5)
    x = f5 / f
    sigma_m = np.sqrt(SIGMA_SYS ** 2 + (0.04 - GAMMA) * x + GAMMA * x ** 2)
    return f * sigma_m / 1.0857362047581294


def observe(true_jy, m5, rng):
    """Measured flux, its error (both nJy) and the detection flag of each visit."""
    true_njy = np.asarray(true_jy, dtype=float) * 1e9
    err = flux_error(true_njy, m5)
    flux = true_njy + rng.normal(0.0, 1.0, np.shape(err)) * err
    return flux, err, flux >= DETECTION_SIGMA * err


def alert_packet(mjd, bands, flux, err, detected, *, object_id=1):
    """A Rubin alert packet (dict) for these visits: the latest detection is the ``diaSource``."""
    det = np.flatnonzero(detected)
    if det.size == 0:
        raise ValueError("no detection: a simulated transient below every visit's 5-sigma depth "
                         "does not raise an alert.")
    letters = [str(b).replace("lsst", "") for b in bands]

    def src(i):
        return {"diaSourceId": int(object_id) * 100000 + int(i), "diaObjectId": int(object_id),
                "visit": int(i), "midpointMjdTai": float(mjd[i]), "band": letters[i],
                "psfFlux": float(flux[i]), "psfFluxErr": float(err[i]), "isNegative": False}

    def forced(i):
        return {"diaForcedSourceId": int(object_id) * 100000 + int(i),
                "diaObjectId": int(object_id), "visit": int(i),
                "midpointMjdTai": float(mjd[i]), "band": letters[i],
                "psfFlux": float(flux[i]), "psfFluxErr": float(err[i])}

    return {"diaObject": {"diaObjectId": int(object_id)},
            "diaSource": src(det[-1]),
            "prvDiaSources": [src(i) for i in det[:-1]],
            "prvDiaForcedSources": [forced(i) for i in range(len(mjd))]}


# --------------------------------------------------------------------------------- the families
#: family -> (factory kind, factory name or None). The keys are what the tests call the families.
FAMILIES = {
    "arnett": ("supernova", "arnett"),
    "magnetar": ("supernova", "basic_magnetar_powered"),
    "shock_cooling_arnett": ("supernova", "shock_cooling_and_arnett"),
    "csm_shock_arnett": ("supernova", "csm_shock_and_arnett"),
    "tde": ("tde", None),
    "kilonova": ("kilonova", None),
    "kilonova_two": ("kilonova_two", None),
}
#: Survey window (days from the event) and cadence of each family's simulated alert.
WINDOWS = {"supernova": dict(t_start=-20.0, t_end=90.0, gap=(2.0, 4.0)),
           "tde": dict(t_start=-30.0, t_end=160.0, gap=(2.0, 4.0)),
           "kilonova": dict(t_start=-3.0, t_end=12.0, gap=(0.8, 1.2)),
           "kilonova_two": dict(t_start=-3.0, t_end=12.0, gap=(0.8, 1.2))}
#: Redshift range each family is injected over (LSST-detectable at good SNR).
REDSHIFTS = {"supernova": (0.03, 0.12), "tde": (0.03, 0.12), "kilonova": (0.01, 0.03),
             "kilonova_two": (0.01, 0.03)}


def build_model(family, bands, redshift, **kw):
    """The family's JAX model, bound to ``bands`` at a known ``redshift``, explosion at day 0.

    ``**kw`` goes to the factory (``free=``, ``prior=``, ``t_exp_days=``, ...).
    """
    import whisper_cbpf as wp

    kind, name = FAMILIES[family]
    bands = list(dict.fromkeys(str(b) for b in bands))
    zkw = {} if redshift is None else dict(redshift=float(redshift),
                                           dl_cm=float(wp.luminosity_distance_cm(redshift)))
    if kind == "supernova":
        return wp.supernova_model(name, bands, name=family, **zkw, **kw)
    if kind == "tde":
        return wp.tde_model(bands, name=family, rise="gaussian", **zkw, **kw)
    if kind == "kilonova":
        return wp.kilonova_model(bands, name=family, **zkw, **kw)
    return wp.kilonova_two_model(bands, name=family, **zkw, **kw)


def draw_physical(model, rng, *, max_tries=10000):
    """One draw of ``model.default_prior`` that passes the model's constraint wall."""
    prior = model.default_prior
    ok = getattr(model.predict_jax, "constraint_ok", None) if model.predict_jax else None
    ctx = getattr(model.predict_jax, "ctx", None)
    for _ in range(max_tries):
        theta = {k: float(prior.distributions[k].sample(rng)) for k in model.parameters
                 if k in prior.distributions}
        if ok is None or ctx is None or ctx.physical([theta[k] for k in ctx.params]):
            return theta
    raise RuntimeError(f"no physical draw of {model.name} in {max_tries} tries")


def simulate_alert(family, rng, *, redshift=None, min_detections=6, min_bands=2,
                   max_tries=200, event_mjd=MJD_EVENT, object_id=1, cadence=None, truth=None):
    """Draw a physical parameter set from the family's prior (or take ``truth``) and observe it
    as an LSST alert.

    Draws until the alert has at least ``min_detections`` detections in at least ``min_bands``
    filters (a selection on the data alone, so posterior calibration is unaffected).

    Returns
    -------
    dict
        ``family``, ``truth`` (parameters), ``redshift``, ``packet`` (Rubin alert),
        ``event_mjd``, ``n_detections``, ``bands`` (filters observed).
    """
    kind = FAMILIES[family][0]
    win = dict(WINDOWS[kind], **(cadence or {}))
    for _ in range(max_tries):
        z = float(rng.uniform(*REDSHIFTS[kind])) if redshift is None else float(redshift)
        t, b, m5 = lsst_cadence(rng, win["t_start"], win["t_end"], gap=win["gap"])
        model = build_model(family, b, z)
        theta = dict(truth) if truth is not None else draw_physical(model, rng)
        true_jy = np.zeros(t.size)
        after = t > 0
        true_jy[after] = model.predict(theta, t[after], b[after])
        flux, err, det = observe(true_jy, m5, rng)
        if det.sum() >= min_detections and len(set(b[det])) >= min_bands:
            return {"family": family, "truth": theta, "redshift": z, "event_mjd": event_mjd,
                    "packet": alert_packet(event_mjd + t, b, flux, err, det,
                                           object_id=object_id),
                    "n_detections": int(det.sum()), "bands": sorted(set(b))}
    raise RuntimeError(f"{family}: no detectable alert in {max_tries} draws")


def truncate_packet(packet, n_detections):
    """The alert as it stood at its ``n_detections``-th detection (the earlier alert)."""
    det = sorted([packet["diaSource"]] + list(packet["prvDiaSources"]),
                 key=lambda r: r["midpointMjdTai"])[:n_detections]
    t_cut = det[-1]["midpointMjdTai"]
    forced = [f for f in packet["prvDiaForcedSources"] if f["midpointMjdTai"] <= t_cut]
    return dict(packet, diaSource=det[-1], prvDiaSources=det[:-1], prvDiaForcedSources=forced)


# ------------------------------------------------------------- a fast toy transient (numpy models)
#: A flare peaking near 20.7 mag (flux in Jy), for tests that must run in seconds on a CPU.
TOY_TRUTH = {"amplitude": 3e-5, "rise_time": 3.0, "decay_time": 12.0}


def toy_priors():
    """Priors in Jy for the registered ``flare`` and ``bazin`` models (theirs are unitless)."""
    from whisper_cbpf.priors import LogUniform, Prior, Uniform

    return {"flare": Prior({"amplitude": LogUniform(1e-7, 1e-3),
                            "rise_time": Uniform(1.0, 10.0), "decay_time": Uniform(5.0, 30.0)}),
            "bazin": Prior({"amplitude": LogUniform(1e-7, 1e-3), "t0": Uniform(-10.0, 30.0),
                            "tau_rise": Uniform(0.1, 20.0), "tau_fall": Uniform(0.5, 60.0)})}


def toy_visits(rng, *, bands=("lsstg", "lsstr"), t_start=-12.0, t_end=75.0, gap=(1.5, 3.0),
               truth=None):
    """Visits of the toy flare (exploding at day 0): ``t, bands, m5, flux, err, detected``."""
    import whisper_cbpf as wp

    t, b, m5 = lsst_cadence(rng, t_start, t_end, bands=bands, gap=gap)
    true = np.zeros(t.size)
    after = t > 0
    true[after] = wp.get_model("flare").predict(truth or TOY_TRUTH, t[after], None)
    flux, err, det = observe(true, m5, rng)
    return t, b, m5, flux, err, det


def ztf_packet(mjd, bands, flux, err, detected, *, object_id="ZTF26aatoyab", corr_offset=-2.0):
    """A ZTF alert packet (``candidate`` + ``prv_candidates``) for these visits (flux in nJy).

    Detections carry ``magpsf``/``sigmapsf`` with ``isdiffpos="t"``; non-detections carry only
    ``diffmaglim``, their 5-sigma depth. A decoy ``magpsf_corr`` (``corr_offset`` mag off) is on
    every detection: the ZTF preset must never read it.
    """
    fid = {"ztfg": 1, "ztfr": 2, "ztfi": 3}
    rows = []
    for i in range(len(mjd)):
        lim = float(ZP_NJY - 2.5 * np.log10(DETECTION_SIGMA * err[i]))
        row = {"jd": float(mjd[i]) + 2400000.5, "fid": fid[str(bands[i])], "diffmaglim": lim,
               "candid": 1000 + i, "magpsf": None, "sigmapsf": None, "isdiffpos": None}
        if detected[i]:
            mag = float(ZP_NJY - 2.5 * np.log10(flux[i]))
            row.update(magpsf=mag, sigmapsf=float(1.0857362047581294 * err[i] / flux[i]),
                       isdiffpos="t", magpsf_corr=mag + corr_offset, sigmapsf_corr=0.01)
        rows.append(row)
    last = int(np.flatnonzero(detected)[-1])          # the alert of the latest detection
    return {"objectId": object_id, "candidate": rows[last], "prv_candidates": rows[:last]}


# --------------------------------------------------------------------------- calibration numbers
def interval_hits(samples, truth, level):
    """Whether ``truth`` lies inside the central ``level`` interval of ``samples``."""
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [0.5 - level / 2, 0.5 + level / 2])
    return bool(lo <= truth <= hi)


def sbc_rank(samples, truth, n_draws=49):
    """Rank of ``truth`` among ``n_draws`` evenly spaced draws (0..n_draws; uniform if calibrated).

    Evenly spaced picks from the flattened chain reach every walker, which keeps the draws close
    to independent.
    """
    s = np.asarray(samples, dtype=float)
    pick = s[np.linspace(0, s.size - 1, n_draws).round().astype(int)]
    return int(np.sum(pick < truth))


def binomial_band(p, n, z=3.0):
    """``p +/- z`` binomial standard errors over ``n`` trials."""
    sd = math.sqrt(p * (1.0 - p) / max(n, 1))
    return p - z * sd, p + z * sd


def rank_uniformity_p(ranks, n_draws=49, n_bins=5):
    """Chi-square p-value of the ranks' histogram against uniform (``n_bins`` equal bins)."""
    from scipy import stats

    ranks = np.asarray(ranks)
    edges = np.linspace(-0.5, n_draws + 0.5, n_bins + 1)
    counts, _ = np.histogram(ranks, bins=edges)
    expected = ranks.size * np.diff(edges) / (n_draws + 1)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return float(stats.chi2.sf(chi2, n_bins - 1)), counts.tolist()
