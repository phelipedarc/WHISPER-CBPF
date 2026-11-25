"""Write the simulated LSST alert the acceptance and speed tests read.

``lsst_alert_sn.json``: a Rubin alert packet of a nickel-powered supernova (the ``arnett`` model
with ``TRUTH`` below) at redshift 0.1, observed on the LSST wide-fast-deep cadence from 20 days
before to 90 days after the explosion (MJD 61000), with LSST's photometric error model
(``tests/science/_sim.py``). The pre-explosion visits are forced photometry only.
``lsst_alert_sn.truth.json`` records the truth.

Run from the repository root::

    python tests/science/data/make_alerts.py
"""
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import _sim  # noqa: E402

TRUTH = {"f_nickel": 0.4, "mej": 1.4, "vej": 11000.0, "kappa": 0.15, "kappa_gamma": 30.0,
         "temperature_floor": 5000.0}


def main():
    rng = np.random.default_rng(20260928)
    alert = _sim.simulate_alert("arnett", rng, redshift=0.1, min_detections=20, min_bands=3,
                                truth=TRUTH, object_id=20260928)
    out = HERE / "lsst_alert_sn.json"
    out.write_text(json.dumps(alert["packet"], indent=1) + "\n")
    meta = {k: alert[k] for k in ("family", "truth", "redshift", "event_mjd", "n_detections")}
    (HERE / "lsst_alert_sn.truth.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}: {alert['n_detections']} detections, truth {alert['truth']}")


if __name__ == "__main__":
    main()
