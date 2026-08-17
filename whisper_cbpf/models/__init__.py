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
    """A transient model, optionally carrying a JAX half.

    ``predict`` is the contract every model satisfies. ``predict_jax`` and ``log_prob_jax`` are the
    optional GPU-side slots, ``None`` on a model that has no JAX implementation. **No sampler reads
    them yet** — the GPU samplers still take a hand-built density (``nuts_gpu``'s ``log_prob_fn``) or
    a hand-passed forward map (``abc_gpu``'s ``predict_jax=``). The slots exist so that a model can
    carry its own instead of every call site rebuilding one; wiring the samplers to prefer them is
    the next step, and until it happens a filled slot changes nothing on its own.

    ``predict_jax(theta, times, band_idx=None) -> flux`` is ``predict`` with the two changes a
    traced function needs, and no others: the parameter **dict** becomes a flat array ordered by
    ``parameters``, and the **band-name** array becomes integer indices into the band list the model
    was built with (the third argument mirrors ``predict``'s ``bands``, including being required by
    a photometric model and ignorable by a self-contained one). Times, and the returned flux density
    in Jy, are unchanged. It takes ONE parameter set; batching is the caller's ``jax.vmap``, which
    keeps one contract instead of two (``samplers/jax/abc_gpu`` wants ``(B, D) -> (B, n)``, NUTS
    wants the scalar form).

    ``log_prob_jax(theta) -> scalar`` is a data-bound log-*likelihood*, ``-inf`` outside the prior
    box, matching what ``samplers/jax/nuts_gpu`` takes as ``log_prob_fn`` (the prior's own density
    is added there by NumPyro's sample sites, so this must not include it). Only a factory that has
    already seen the light curve can fill this slot; a model built without data leaves it ``None``
    and the sampler assembles one from ``predict_jax`` plus
    ``likelihood.log_likelihood_jax`` / ``priors.log_prob_jax``.
    """

    name: str
    predict: Callable                       # (params, times, bands) -> flux array
    parameters: List[str]
    default_prior: Optional[Prior] = None
    description: str = ""
    predict_jax: Optional[Callable] = None      # (theta, times, band_idx) -> flux, jax-traceable
    log_prob_jax: Optional[Callable] = None     # (theta) -> scalar log-likelihood

    def __call__(self, parameters, times, bands=None):
        return np.asarray(self.predict(parameters, times, bands), dtype=float)


_REGISTRY: dict = {}


def register_model(name, predict, parameters, prior=None, description="", *, overwrite=False,
                   predict_jax=None, log_prob_jax=None):
    """Register a model so it can be used by name (e.g. ``fit_ABC(lc, "my_model")``).

    ``predict_jax`` / ``log_prob_jax`` are the optional JAX slots (see :class:`Model`). A factory
    that built them must pass them through here, or the registered model is the CPU half only.
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Model {name!r} already registered (pass overwrite=True to replace).")
    model = Model(name=name, predict=predict, parameters=list(parameters),
                  default_prior=prior, description=description,
                  predict_jax=predict_jax, log_prob_jax=log_prob_jax)
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
# two_component_kilonova and redback_adapter import redback lazily (only inside the functions that
# need it), so importing them here is safe even when the optional [models] extra is not installed.
from . import bazin, flare, gaussian_rise, mck19, redback_adapter  # noqa: E402
from . import two_component_kilonova  # noqa: E402

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

# Bind the JAX subpackage as an attribute so `whisper_cbpf.models.jax.<module>` resolves.
# Its __init__ imports nothing heavier than the standard library -- every JAX name inside
# it is lazy -- so this costs nothing on a CPU-only install.
from . import jax  # noqa: E402,F401
