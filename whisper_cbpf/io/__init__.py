"""Data ingestion: canonical light-curve container, CSV loader, band/photometry/unit helpers."""
from . import svo
from .bands import (
    DEFAULT_BAND_ALIASES,
    FILTER_LOOKUP,
    LSST_BAND_INFO,
    group_bands,
    normalize_band,
    normalize_bands,
    resolve_band,
    resolve_bands,
    unmapped_bands,
)
from .loader import load_lightcurve
from .photometry import (
    AB_ZEROPOINT_JY,
    flux_density_to_mag,
    mag_err_to_snr,
    mag_to_flux_density,
)
from .schema import VALID_DATA_MODES, LightCurve
from .svo import (
    SvoUnavailable,
    clear_manual_bands,
    get_transmission_data,
    register_manual_band,
    resolve_band_svo,
    unregister_manual_band,
)
from .units import to_canonical

__all__ = [
    "LightCurve",
    "VALID_DATA_MODES",
    "load_lightcurve",
    "normalize_band",
    "normalize_bands",
    "group_bands",
    "resolve_band",
    "resolve_bands",
    "unmapped_bands",
    "DEFAULT_BAND_ALIASES",
    "FILTER_LOOKUP",
    "LSST_BAND_INFO",
    "mag_to_flux_density",
    "flux_density_to_mag",
    "mag_err_to_snr",
    "AB_ZEROPOINT_JY",
    "to_canonical",
    # SVO fallback
    "svo",
    "SvoUnavailable",
    "register_manual_band",
    "unregister_manual_band",
    "clear_manual_bands",
    "resolve_band_svo",
    "get_transmission_data",
]
