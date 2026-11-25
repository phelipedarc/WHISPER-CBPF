"""Write ``lsst_alert_sn.json``: a synthetic Rubin alert packet of a nickel-powered supernova.

The quickstart in ``README.md`` and ``docs/LSST_ALERTS.md`` loads this file, and
``tests/test_docs_coverage.py`` runs that quickstart. It is a Rubin alert packet in the alert schema's
own field names (``diaSource`` + ``prvDiaSources`` + ``prvDiaForcedSources``; ``midpointMjdTai``,
``band``, ``psfFlux`` and ``psfFluxErr`` in nJy, ``visit``).

The truth: the JAX ``arnett`` supernova at redshift 0.1 (Planck18 distance), exploding at
MJD 61000.0, with ``f_nickel=0.1, mej=2.0, vej=9000, kappa=0.1, kappa_gamma=10,
temperature_floor=5000``. It is observed in LSST g, r and i every ~2.3 days from +2.5 d to +30 d
(12 detections). The noise is a 24.5 mag 5-sigma depth plus a 2 % calibration floor, drawn with
seed 20260928. Three forced-photometry epochs before the explosion (-9, -6, -3 d) carry no
diaSource, so the LSST preset reads them as 5-sigma upper limits.

Run it in the package's environment (float64 is needed by the supernova model)::

    python tests/data/make_lsst_alert_sn.py
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import whisper_cbpf as wp  # noqa: E402

REDSHIFT = 0.1
T_EXPLOSION = 61000.0
TRUTH = {"f_nickel": 0.1, "mej": 2.0, "vej": 9000.0, "kappa": 0.1, "kappa_gamma": 10.0,
         "temperature_floor": 5000.0}
DEPTH_5SIGMA = 24.5                      # AB mag
CALIBRATION_FLOOR = 0.02                 # fractional
ZP_NJY = 31.4                            # AB magnitude of 1 nJy
OBJECT_ID = 3068394823507361793
SEED = 20260928
OUT = Path(__file__).with_name("lsst_alert_sn.json")


def main():
    rng = np.random.default_rng(SEED)
    bands = ["lsstg", "lsstr", "lssti"]
    model = wp.supernova_model("arnett", bands, REDSHIFT, wp.luminosity_distance_cm(REDSHIFT))
    phase = np.round(np.linspace(2.5, 30.0, 12), 3)                 # days since explosion
    band = [bands[i % 3] for i in range(phase.size)]
    flux_njy = np.asarray(model.predict(TRUTH, phase, band)) * 1e9
    sky = 10 ** ((ZP_NJY - DEPTH_5SIGMA) / 2.5) / 5.0              # 1-sigma sky noise, nJy
    err = np.sqrt(sky ** 2 + (CALIBRATION_FLOOR * flux_njy) ** 2)
    obs = flux_njy + rng.normal(0.0, err)

    sources = []
    for i, (t, b, f, e) in enumerate(zip(phase, band, obs, err)):
        sources.append({"diaSourceId": 1000 + i, "diaObjectId": OBJECT_ID,
                        "visit": 700000 + i, "midpointMjdTai": round(T_EXPLOSION + float(t), 5),
                        "band": b[-1], "psfFlux": round(float(f), 3),
                        "psfFluxErr": round(float(e), 3), "isNegative": False})
    forced = []
    for j, t in enumerate((-9.0, -6.0, -3.0)):
        forced.append({"diaForcedSourceId": 5000 + j, "diaObjectId": OBJECT_ID,
                       "visit": 690000 + j, "midpointMjdTai": T_EXPLOSION + t,
                       "band": "gri"[j], "psfFlux": round(float(rng.normal(0.0, sky)), 3),
                       "psfFluxErr": round(float(sky), 3)})
    packet = {"diaObject": {"diaObjectId": OBJECT_ID},
              "diaSource": sources[-1], "prvDiaSources": sources[:-1],
              "prvDiaForcedSources": forced}
    OUT.write_text(json.dumps(packet, indent=1) + "\n")
    print(f"wrote {OUT} ({len(sources)} detections, {len(forced)} forced epochs)")


if __name__ == "__main__":
    main()
