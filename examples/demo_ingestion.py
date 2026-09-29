#!/usr/bin/env python
"""Demo: survey-CSV ingestion upgrade — data_mode, redshift, astropy units, SVO band fallback.

Runs **fully offline**: the one place that would hit the network (the SVO Filter Profile Service)
is stubbed here with a fake metadata function, exactly as the test-suite mocks it. In production that
boundary queries SVO through pyphot when it is installed (the ``[models]`` extra), else through
``astroquery.svo_fps.SvoFps`` (the ``[analysis]`` extra).

Ported from WHISPER_AI @ 8ba3843 (discontinued; superseded by whisper_cbpf), ``dev/demo_ingestion.py``.
From the repo root::

    python examples/demo_ingestion.py
"""
from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import astropy.units as u

import whisper_cbpf as wp
from whisper_cbpf.io import svo
from whisper_cbpf.io.units import to_flux_density_jy

# Magnitudes are dimensionless by convention; silence that one no-unit notice to keep the demo tidy
# (the units section below focuses on flux). Every other warning is shown intentionally.
warnings.filterwarnings("ignore", message="Column for data_mode='magnitude'")

# A throwaway SVO disk cache: the stubbed answers below must not be written into, and
# clear_cache(disk=True) must not delete, the real one (~/.cache/whisper_cbpf/svo_cache.json).
os.environ["WHISPER_SVO_CACHE"] = str(Path(tempfile.mkdtemp()) / "svo_cache.json")


def rule(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def write_csv(text):
    path = Path(tempfile.mkdtemp()) / "lc.csv"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
rule("1. data_mode  +  __call__  -> enriched dataframe (both flux and magnitude)")

mag_csv = write_csv(
    "time,magnitude,e_magnitude,band\n"
    "1.0,20.0,0.05,g\n2.0,20.4,0.05,r\n3.0,20.8,0.06,i\n")
lc = wp.load_lightcurve(mag_csv, redshift=0.1)
print(f"data_mode       = {lc.data_mode!r}   (stored in .meta; inferred from the columns)")
print(f"output_format   = {lc.output_format!r}   (forward-model comparison space: magnitude / flux_density)")
print("LightCurve is an astropy Table; add the derived flux column with add_flux():")
print(lc.add_flux().to_dataframe().to_string(index=False))


# ---------------------------------------------------------------------------
rule("2. Redshift: argument > column > unknown (validated, never fatal when unknown)")

print("a) explicit argument      ->", wp.load_lightcurve(mag_csv, redshift=0.12).redshift)

zcol = write_csv("time,magnitude,e_magnitude,band,redshift\n1.0,20.0,0.05,g,0.34\n2.0,20.4,0.05,r,0.34\n")
print("b) 'redshift' column      ->", wp.load_lightcurve(zcol).redshift)
print("c) argument beats column  ->", wp.load_lightcurve(zcol, redshift=0.99).redshift)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    lcu = wp.load_lightcurve(mag_csv)            # neither -> unknown
print(f"d) neither -> unknown     ->  redshift_known={lcu.redshift_known}, "
      f"prior={lcu.redshift_prior}")
print(f"   warned: {any('redshift' in str(x.message) for x in w)}")

for bad, why in [(0.0, "z==0 needs a luminosity distance"), (-0.1, "negative"), (np.nan, "NaN")]:
    try:
        wp.LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[0.1], redshift=bad)
        print(f"e) z={bad!r:>5} -> NO error (unexpected!)")
    except ValueError as exc:
        print(f"e) z={bad!r:>5} -> ValueError ({why})")
ok = wp.LightCurve(time=[1.0], band=["r"], flux=[1.0], flux_err=[0.1],
                   redshift=0.0, luminosity_distance=40.0)
print(f"   z=0 + luminosity_distance=40 Mpc -> OK (redshift_known={ok.redshift_known})")


# ---------------------------------------------------------------------------
rule("3. astropy units: F_nu / F_lambda -> canonical Jy; magnitude must be dimensionless")

lam = 6215.0  # r-band effective wavelength (Angstrom)
flam = (1.0 * u.mJy).to_value(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(lam * u.AA))
print(f"1 mJy as F_lambda @ {lam} AA = {flam:.3e} erg/s/cm^2/AA")
print(f"  -> via Jy (F_nu)          = {to_flux_density_jy([1.0], 'mJy')[0]:.6e} Jy")
print(f"  -> via spectral_density   = {to_flux_density_jy([flam], 'erg/(s cm2 AA)', lambda_eff=[lam])[0]:.6e} Jy")

flam_csv = write_csv(f"time,flux,flux_err,band\n1.0,{flam},{flam*0.1},r\n")
lcf = wp.load_lightcurve(flam_csv, redshift=0.1, flux_unit="erg/(s cm2 AA)")
print(f"loader F_lambda -> Jy        = {lcf.flux[0]:.6e} Jy  (band wavelength pulled automatically)")

try:
    wp.load_lightcurve(mag_csv, redshift=0.1, magnitude_unit="Jy")
except ValueError as exc:
    print(f"magnitude with a flux unit   -> ValueError: {str(exc)[:60]}...")


# ---------------------------------------------------------------------------
rule("4. Bands: FILTER_LOOKUP -> warn -> SVO fallback (cached, graceful, manual override)")

# Stub the SVO network boundary (in production this calls astroquery.svo_fps.SvoFps).
_calls = {"n": 0}
def fake_svo_metadata(filter_id):
    _calls["n"] += 1
    print(f"      [network] SVO queried for {filter_id!r}  (call #{_calls['n']})")
    return {"filter_id": filter_id, "WavelengthEff": 6200.0, "ZeroPoint": 3600.0}
svo._svo_fetch_metadata = fake_svo_metadata
svo.clear_cache(disk=True)

print("a) known band 'g'  (LSST table, no SVO call):")
g = wp.resolve_band("g")
print(f"     {g['source']}: lambda_eff={g['lambda_eff']} AA, zero_point={g['zero_point']} Jy")

print("b) unknown band 'PAN-STARRS/PS1.w'  (warns, then SVO):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    w1 = wp.resolve_band("PAN-STARRS/PS1.w")
    w2 = wp.resolve_band("PAN-STARRS/PS1.w")   # second time: served from cache
print(f"     {w1['source']}: lambda_eff={w1['lambda_eff']} AA, zero_point={w1['zero_point']} Jy")
print(f"     repeated lookup -> total SVO network calls = {_calls['n']} (cache hit on the 2nd)")

print("c) SVO down + manual override:")
def svo_down(filter_id):
    raise svo.SvoUnavailable("network unreachable")
svo._svo_fetch_metadata = svo_down
svo.clear_cache(disk=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    unr = wp.resolve_band("my_weird_filter")
print(f"     graceful degrade -> source={unr['source']!r}, lambda_eff={unr['lambda_eff']}")
wp.register_manual_band("my_weird_filter", lambda_eff=9000.0, zero_point=3631.0)
man = wp.resolve_band("my_weird_filter")
print(f"     after register_manual_band -> source={man['source']!r}, lambda_eff={man['lambda_eff']} AA")

print("\nDone — every step above ran offline (SVO stubbed). Install pyphot or astroquery for the "
      "live service.")
