"""JAX-free definition of the phenomenological models: parameter names, per-band layout, priors.

Adapted from redback-jax (Nikhil Sarin et al.), https://github.com/nikhil-sarin/redback-jax,
commit 3431e2e, GPL-3.0, and from redback (Sarin et al.), https://github.com/nikhil-sarin/redback,
GPL-3.0; modified for whisper_cbpf.

Kept apart from :mod:`.phenomenological` for the reason :mod:`._flare_spec` is kept apart from the
flare: the parameter names and the prior boxes can be read, and a model can be registered, without
importing JAX.

Four per-band flux or magnitude functions, with no photosphere and no SED. In every one the shape
parameters are shared by all bands and only the amplitude differs from band to band, which is how
redback evaluates them (``bazin_sne`` takes one ``aa`` and ``bb`` per band;
``exp_rise_powerlaw_decline`` takes one ``m_peak`` per filter and "only ``m_peak`` differs from
filter to filter"). A model bound to a band list therefore carries one amplitude per band, named
after redback's argument with the band appended: ``aa_ztfg``, ``bb_ztfg``, ``m_peak_ztfg``.

The prior boxes are redback's own, transcribed from its ``.prior`` files
(``bazin_sne.prior``, ``villar_sne.prior``, ``exp_rise_powerlaw_decline.prior``); a test compares
them with the installed redback so they cannot drift. Two of them are not redback's:

* ``smooth_exponential_powerlaw`` has no prior file of its own. Its shape prior is the one redback
  ships for the cooling-envelope model that wraps it (``peak_time`` there, ``tpeak`` here), and its
  amplitude ``a_1`` takes the amplitude prior of the other flux models.
* The amplitude prior ``LogUniform(1e-15, 1e15)`` of ``bazin_sne`` and ``villar_sne`` is redback's
  unit-free placeholder. Here amplitudes are flux densities in Jy, so replace it with a box that
  covers your data (``prior=`` of the factory).
"""
from __future__ import annotations

__all__ = ["MODELS", "SPECS", "parameter_names", "default_prior"]

_AMPLITUDE = ("loguniform", 1e-15, 1e15)

#: One entry per model, in redback's argument order (the epoch array excluded):
#:
#: ``parameters``  redback's argument names;
#: ``per_band``    the arguments that take one value per band (the rest are shared);
#: ``epochs``      the arguments that are epochs on the light curve's clock, as opposed to durations;
#: ``prior``       ``{name: (kind, low, high)}``;
#: ``unit``        what the model returns, before whisper's flux density in Jy.
SPECS = {
    "smooth_exponential_powerlaw": dict(
        parameters=["a_1", "tpeak", "alpha_1", "alpha_2", "smoothing_factor"],
        per_band=("a_1",),
        epochs=(),
        prior={"a_1": _AMPLITUDE,
               "tpeak": ("loguniform", 0.1, 60.0),
               "alpha_1": ("uniform", 0.5, 3.0),
               "alpha_2": ("uniform", -3.0, -0.5),
               "smoothing_factor": ("uniform", 0.5, 2.0)},
        unit="flux density in Jy (the units of a_1); time is measured from the start, in days",
    ),
    "exp_rise_powerlaw_decline": dict(
        parameters=["t0", "m_peak", "tau_rise", "alpha", "t_peak"],
        per_band=("m_peak",),
        epochs=("t0", "t_peak"),
        prior={"t0": ("uniform", 58000.0, 59000.0),
               "m_peak": ("uniform", 16.0, 20.0),
               "tau_rise": ("uniform", 0.1, 10.0),
               "alpha": ("uniform", 0.1, 5.0),
               "t_peak": ("uniform", 59000.0, 59500.0)},
        unit="AB magnitude, converted to flux density in Jy; zero flux before t0",
    ),
    "bazin_sne": dict(
        parameters=["aa", "bb", "t0", "tau_rise", "tau_fall"],
        per_band=("aa", "bb"),
        epochs=("t0",),
        prior={"aa": _AMPLITUDE,
               "bb": _AMPLITUDE,
               "t0": ("uniform", 0.0, 50.0),
               "tau_rise": ("uniform", 0.01, 50.0),
               "tau_fall": ("uniform", 1.0, 300.0)},
        unit="flux density in Jy (the units of aa and bb)",
    ),
    "villar_sne": dict(
        parameters=["aa", "cc", "t0", "tau_rise", "tau_fall", "gamma", "nu"],
        per_band=("aa", "cc"),
        epochs=("t0",),
        prior={"aa": _AMPLITUDE,
               "cc": _AMPLITUDE,
               "t0": ("uniform", 0.0, 50.0),
               "tau_rise": ("uniform", 0.01, 50.0),
               "tau_fall": ("uniform", 1.0, 300.0),
               "gamma": ("uniform", 0.1, 100.0),
               "nu": ("uniform", 0.0, 1.0)},
        unit="flux density in Jy (the units of aa and cc)",
    ),
}

MODELS = tuple(SPECS)


def _spec(model):
    try:
        return SPECS[model]
    except KeyError:
        raise ValueError(f"unknown phenomenological model {model!r}; "
                         f"available: {list(MODELS)}") from None


def parameter_names(model, band_names=None):
    """The model's parameters, in redback's argument order.

    With ``band_names=None`` the model is band-independent and the names are redback's own. With a
    band list each per-band argument is replaced, in place, by one entry per band
    (``aa`` -> ``aa_g, aa_r``).

    >>> parameter_names("bazin_sne")
    ['aa', 'bb', 't0', 'tau_rise', 'tau_fall']
    >>> parameter_names("bazin_sne", ["g", "r"])
    ['aa_g', 'aa_r', 'bb_g', 'bb_r', 't0', 'tau_rise', 'tau_fall']
    """
    spec = _spec(model)
    if band_names is None:
        return list(spec["parameters"])
    bands = [str(b) for b in band_names]
    if not bands or len(set(bands)) != len(bands):
        raise ValueError(f"band_names must be a non-empty list without repeats, got {bands}")
    out = []
    for p in spec["parameters"]:
        out += [f"{p}_{b}" for b in bands] if p in spec["per_band"] else [p]
    return out


def _distribution(kind, low, high):
    from ...priors import LogUniform, Uniform
    return {"uniform": Uniform, "loguniform": LogUniform}[kind](low, high)


def default_prior(model, band_names=None, override=None):
    """The model's prior as a :class:`~whisper_cbpf.priors.Prior`, redback's boxes.

    ``override`` is a ``Prior`` or a ``dict`` of distributions. A key is either a full parameter
    name or, for a per-band argument, redback's name, which then applies to every band
    (``{"aa": LogUniform(1e-6, 1e-3)}``).

    >>> p = default_prior("bazin_sne", ["g", "r"])
    >>> p.names[:3]
    ['aa_g', 'aa_r', 'bb_g']
    """
    spec = _spec(model)
    names = parameter_names(model, band_names)
    dists = {}
    for p in spec["parameters"]:
        dist = _distribution(*spec["prior"][p])
        if band_names is not None and p in spec["per_band"]:
            dists.update({f"{p}_{b}": _distribution(*spec["prior"][p]) for b in map(str, band_names)})
        else:
            dists[p] = dist
    if override is not None:
        given = dict(getattr(override, "distributions", override))
        for key, dist in given.items():
            if key in dists:
                dists[key] = dist
            elif band_names is not None and key in spec["per_band"]:
                dists.update({f"{key}_{b}": dist for b in map(str, band_names)})
            else:
                raise ValueError(f"prior names {key!r}, which is not a parameter of {model!r}. "
                                 f"Parameters: {names}; a per-band argument "
                                 f"{list(spec['per_band'])} may be given without its band.")
    from ...priors import Prior
    return Prior({n: dists[n] for n in names})
