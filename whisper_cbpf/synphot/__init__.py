"""Synthetic photometry: how whisper turns a model SED into a band magnitude, on the CPU and GPU.

One definition everywhere (whisper 0.1.1): the model's F_nu is integrated over the
photon-counting bandpass,

    m_b = -2.5 log10[ sum_k w_bk F_nu(lam_bk) / (3631 Jy sum_k w_bk) ],

with the nodes ``lam_bk`` and weights ``w_bk`` of ONE :class:`FilterSet` that both backends read.
Before 0.1.1 the redback CPU adapter evaluated F_nu at a single reference frequency per band
(10 mmag median, up to 0.65 mag off the band integral) while the JAX models integrated, so a CPU
and a GPU fit of the same model disagreed by construction.

- :func:`resolve_filter` -- a data label to a filter name (bare ``u g r i z y`` are LSST).
- :func:`gauss_rule` -- the default quadrature: 16 Gauss nodes per band, adapted to ``T dlam/lam``.
- :mod:`.grid_rule` -- the shared-grid rule the JAX models used up to 0.1.0 (``n_wave=``).
- :func:`filter_set_for` -- the default FilterSet for some filters (shipped for LSST ugrizy,
  ZTF gri and SDSS ugriz, so no sncosmo is needed for those).
- :func:`band_flux_jy` -- the CPU band flux of any model that returns F_nu at (time, frequency).

Imports nothing heavier than numpy: jax, sncosmo, pyphot and redback are all optional and lazy.
See ``docs/PHOTOMETRY.md``.
"""
from .band_flux import band_flux_jy
from .filterset import FilterSet, build_library, filter_set_for, shipped_filter_names
from .gauss_rule import DEFAULT_N_NODES, curve, fine_measure, gauss_rule
from .labels import (
    BAND_SYSTEMS,
    default_band_system,
    resolve_filter,
    resolve_filters,
    set_default_band_system,
)

__all__ = ["FilterSet", "gauss_rule", "filter_set_for", "shipped_filter_names", "build_library",
           "band_flux_jy", "resolve_filter", "resolve_filters", "set_default_band_system",
           "default_band_system", "BAND_SYSTEMS", "curve", "fine_measure", "DEFAULT_N_NODES"]
