"""Transient model registry: built-in models plus any you register yourself.

A model maps parameters to predicted observables:
``predict(parameters: dict, times: np.ndarray, bands: np.ndarray) -> flux np.ndarray``.

Register your own with :func:`register_model`. For parallel ABC (``n_jobs > 1``) the predict
function must be picklable — defined at module level, not a closure or lambda — otherwise use
``n_jobs=1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..priors import Prior


@dataclass
class Model:
    """A transient model, optionally carrying a JAX half.

    ``predict`` is the contract every model satisfies. ``predict_jax`` and ``log_prob_jax`` are the
    optional GPU-side slots, ``None`` on a model with no JAX implementation. The JAX samplers read
    them through :mod:`whisper_cbpf.samplers.jax._adapters`, which prefers an explicit argument,
    then a filled slot, then builds one.

    ``predict_jax(theta, times, band_idx=None) -> flux`` is ``predict`` with the two changes a
    traced function needs, and no others: the parameter **dict** becomes a flat array ordered by
    ``parameters``, and the **band-name** array becomes integer indices into the band list the model
    was built with. Times, and the returned flux density in Jy, are unchanged. It takes one
    parameter set; batching is the caller's ``jax.vmap``, so there is one contract rather than two.

    ``log_prob_jax(theta) -> scalar`` is a data-bound log-*likelihood*, ``-inf`` outside the prior
    box. It must not include the prior's own density: NumPyro adds that at its sample sites. Only a
    factory that has already seen the light curve can fill this slot; a model built without data
    leaves it ``None`` and the sampler assembles one from ``predict_jax`` plus
    ``likelihood.log_likelihood_jax`` and ``priors.log_prob_jax``.

    ``param_aliases`` maps a parameter's name here to the name the same parameter carries in the
    model's redback-named twin, where the two differ -- ``kilonova_two_jax``'s ``mej_blue ...
    kappa_red`` onto ``mej_1 ... kappa_2``, the names of ``two_component_kilonova`` and of
    ``register_redback("two_component_kilonova_model")``. Empty when they already agree. It renames
    nothing by itself; a CPU-vs-GPU comparison pairs the two posteriors with it, e.g.
    ``samples.rename(columns=model.param_aliases)``.

    Attributes
    ----------
    name : str
    predict : callable
        ``predict(params: dict, times, bands) -> flux [Jy]``.
    parameters : list of str
    default_prior : Prior, optional
    description : str
    predict_jax, log_prob_jax : callable, optional
        The JAX slots described above.
    param_aliases : dict

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> def decline(p, t, bands=None):
    ...     return p["amp"] * np.exp(-np.asarray(t, float) / p["tau"])
    >>> m = wp.Model(name="decline", predict=decline, parameters=["amp", "tau"],
    ...              default_prior=wp.Prior({"amp": wp.LogUniform(1e-6, 1e-2),
    ...                                      "tau": wp.Uniform(1.0, 50.0)}))
    >>> m({"amp": 1e-4, "tau": 10.0}, [0.0, 10.0]).round(8).tolist()
    [0.0001, 3.679e-05]
    """

    name: str
    predict: Callable                       # (params, times, bands) -> flux array
    parameters: List[str]
    default_prior: Optional[Prior] = None
    description: str = ""
    predict_jax: Optional[Callable] = None      # (theta, times, band_idx) -> flux, jax-traceable
    log_prob_jax: Optional[Callable] = None     # (theta) -> scalar log-likelihood
    param_aliases: dict = field(default_factory=dict)   # {name here: redback twin's name}

    def __call__(self, parameters, times, bands=None):
        return np.asarray(self.predict(parameters, times, bands), dtype=float)


_REGISTRY: dict = {}


def register_model(name, predict, parameters, prior=None, description="", *, overwrite=False,
                   predict_jax=None, log_prob_jax=None, param_aliases=None):
    """Register a model so it can be used by name (e.g. ``fit_ABC(lc, "my_model")``).

    ``predict_jax`` / ``log_prob_jax`` are the optional JAX slots and ``param_aliases`` the
    optional name map (see :class:`Model`). A factory that built them must pass them through here,
    or the registered model is the CPU half only. A model you only pass as an object (to
    :func:`whisper_cbpf.fit` or :func:`whisper_cbpf.compare`) needs no registration.

    Parameters
    ----------
    name : str
        The name to fit it by.
    predict : callable
        ``predict(params: dict, times, bands) -> flux [Jy]``. For parallel ABC (``n_jobs > 1``) it
        must be picklable: defined at module level, not a closure or lambda.
    parameters : list of str
        The parameter names, in the order samplers report them.
    prior : Prior, optional
        The default prior of a fit.
    description : str, optional
    overwrite : bool, default False
        Replace a model already registered under ``name``.
    predict_jax, log_prob_jax, param_aliases : optional
        See :class:`Model`.

    Returns
    -------
    Model

    Raises
    ------
    ValueError
        ``name`` is taken and ``overwrite`` is False.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> def plateau(p, t, bands=None):
    ...     return np.full(np.shape(t), p["level"])
    >>> m = wp.register_model("plateau_example", plateau, ["level"],
    ...                       prior=wp.Prior({"level": wp.Uniform(0.0, 5.0)}), overwrite=True)
    >>> wp.get_model("plateau_example") is m
    True
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"Model {name!r} already registered (pass overwrite=True to replace).")
    model = Model(name=name, predict=predict, parameters=list(parameters),
                  default_prior=prior, description=description,
                  predict_jax=predict_jax, log_prob_jax=log_prob_jax,
                  param_aliases=dict(param_aliases or {}))
    _REGISTRY[name] = model
    return model


def get_model(model):
    """Resolve a model name (or pass a :class:`Model` through).

    Parameters
    ----------
    model : str or Model

    Returns
    -------
    Model

    Raises
    ------
    KeyError
        No model is registered under that name; the message lists the ones that are.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> flare = wp.get_model("flare")
    >>> flare.parameters
    ['amplitude', 'rise_time', 'decay_time']
    >>> wp.get_model(flare) is flare
    True
    """
    if isinstance(model, Model):
        return model
    if model in _REGISTRY:
        return _REGISTRY[model]
    raise KeyError(f"Unknown model {model!r}. Available: {list_models()}")


def list_models():
    """The names of every registered model, sorted.

    The JAX factories (:func:`whisper_cbpf.supernova_model`, ...) build models that are passed as
    objects and are not listed unless registered (``register_supernova`` and friends).

    Returns
    -------
    list of str

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> {"bazin", "flare", "gaussian_rise"} <= set(wp.list_models())
    True
    """
    return sorted(_REGISTRY)


# two_component_kilonova and redback_adapter import redback lazily, inside the functions that need
# it, so importing them here is safe without the optional [models] extra.
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
