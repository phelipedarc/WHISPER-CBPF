"""Generic single-component flare model (analytic; no redback needed).

``flux(t) = amplitude * (1 - exp(-t / rise_time)) * exp(-t / decay_time)``

Band-independent and vectorized over ``times``. ``t`` should be days since explosion (use
``LightCurve.set_explosion_date`` / ``load_lightcurve(..., explosion_date=...)``).
"""
from __future__ import annotations

import numpy as np

from ..priors import Prior, Uniform

PARAMETERS = ["amplitude", "rise_time", "decay_time"]
DESCRIPTION = "Generic flare: A*(1 - exp(-t/t_rise))*exp(-t/t_decay)."

#: Clip the exponent of ``exp()`` to keep it finite for O(1)-scaled time/rate arguments.
_EXP_ARG_CLIP = 50.0

# Default prior matches the example; scale `amplitude` to your flux units when fitting real data.
PRIOR = Prior({
    "amplitude": Uniform(0.0, 10.0),
    "rise_time": Uniform(1.0, 10.0),
    "decay_time": Uniform(5.0, 30.0),
})


def flare_flux(parameters, times, bands=None):
    """Predicted flux at ``times`` (bands ignored -- this toy model is band-independent).

    Flux is **zero for t <= 0** (no emission before the explosion) and non-negative everywhere.
    """
    amplitude = parameters["amplitude"]
    rise_time = parameters["rise_time"]
    decay_time = parameters["decay_time"]
    t = np.asarray(times, dtype=float)
    flux = (amplitude * (1.0 - np.exp(np.clip(-t / rise_time, -_EXP_ARG_CLIP, _EXP_ARG_CLIP)))
            * np.exp(np.clip(-t / decay_time, -_EXP_ARG_CLIP, _EXP_ARG_CLIP)))
    return np.where(t > 0.0, flux, 0.0)
