"""Transient model registry: built-in models + user-registered custom models.

A model maps parameters to predicted observables:
``predict(parameters: dict, times: np.ndarray, bands: np.ndarray) -> flux np.ndarray``.

Register your own with :func:`register_model`. NOTE: for parallel ABC (``n_jobs > 1``) the predict
function must be picklable (i.e. defined at module level, not a closure/lambda); otherwise use
``n_jobs=1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..priors import Prior


@dataclass
class Model:
    name: str
    predict: Callable                       # (params, times, bands) -> flux array
    parameters: List[str]
    default_prior: Optional[Prior] = None
    description: str = ""

    def __call__(self, parameters, times, bands=None):
        return np.asarray(self.predict(parameters, times, bands), dtype=float)


_REGISTRY: dict = {}


def register_model(name, predict, parameters, prior=None, description="", *, overwrite=False):
    """Register a model so it can be used by name (e.g. ``fit_ABC(lc, "my_model")``)."""
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Model {name!r} already registered (pass overwrite=True to replace).")
    model = Model(name=name, predict=predict, parameters=list(parameters),
                  default_prior=prior, description=description)
    _REGISTRY[name] = model
    return model


def get_model(model):
    """Resolve a model name (or pass a :class:`Model` through)."""
    if isinstance(model, Model):
        return model
    if model in _REGISTRY:
        return _REGISTRY[model]
    raise KeyError(f"Unknown model {model!r}. Available: {list_models()}")


def list_models():
    return sorted(_REGISTRY)


# --- register built-in models ---
# two_component_kilonova imports redback lazily (only inside predict), so importing it here is safe
# even when the optional [models] extra is not installed.
from . import bazin, flare, gaussian_rise, mck19, two_component_kilonova  # noqa: E402

register_model("flare", flare.flare_flux, flare.PARAMETERS,
               prior=flare.PRIOR, description=flare.DESCRIPTION)
register_model("bazin", bazin.bazin_flux, bazin.PARAMETERS,
               prior=bazin.PRIOR, description=bazin.DESCRIPTION)
register_model("gaussian_rise", gaussian_rise.gaussian_rise_flux, gaussian_rise.PARAMETERS,
               prior=gaussian_rise.PRIOR, description=gaussian_rise.DESCRIPTION)
register_model("mck19", mck19.mck19_flux, mck19.PARAMETERS,
               prior=mck19.PRIOR, description=mck19.DESCRIPTION)
register_model("two_component_kilonova", two_component_kilonova.two_component_kilonova_flux,
               two_component_kilonova.PARAMETERS, prior=two_component_kilonova.PRIOR,
               description=two_component_kilonova.DESCRIPTION)
