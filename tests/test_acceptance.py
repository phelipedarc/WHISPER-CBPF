"""Usability acceptance: from an LSST alert to a ranked table and an HTML report in six lines.

The user code below is run exactly as a user runs it: in a fresh Python process, from the alert
file, with nothing imported or configured beforehand. It must stay within six lines, print the
ranking, and write one self-contained HTML report that shows the ranking.

- The fast test runs the six lines on the CPU with a short chain (``nsteps=300``), which is how a
  user takes a first look; it belongs to the normal suite.
- The slow test runs them at the default settings (the ``README`` quickstart), on a GPU when one
  is visible.

The alert is ``tests/science/data/lsst_alert_sn.json``: a simulated Rubin alert packet of a
supernova at redshift 0.1 (``tests/science/data/make_alerts.py``).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

pytestmark = pytest.mark.science

ALERT = Path(__file__).parent / "science" / "data" / "lsst_alert_sn.json"

USER_CODE = """\
import jax; jax.config.update("jax_enable_x64", True)
import whisper_cbpf as wp
lc = wp.load_lightcurve("{alert}", survey="lsst", redshift=0.1)
cmp = wp.compare(lc, ["arnett", "magnetar"]{settings})
print(cmp.summary())
cmp.report("{out}")
"""


def _user_lines(code):
    return [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _run(tmp_path, settings, *, cpu):
    code = USER_CODE.format(alert=ALERT, out=tmp_path / "out", settings=settings)
    assert len(_user_lines(code)) <= 6, code
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    if cpu:
        env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=5400)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout, tmp_path / "out" / "report.html"


def _check(stdout, report):
    assert re.search(r"Comparison of 2 models on \d+ points by BIC: winner '(arnett|magnetar)'",
                     stdout), stdout[:2000]
    assert "rank" in stdout and "weight" in stdout and "grade" in stdout
    assert report.is_file()
    html = report.read_text()
    assert "<script" not in html.lower()                          # self-contained, no code
    assert not re.search(r"(src|href)=\"https?://", html)          # nothing fetched from the web
    for word in ("arnett", "magnetar", "BIC"):
        assert word in html
    assert "data:image/png;base64," in html                       # the figures are inline


def test_six_lines_from_an_lsst_alert_to_a_ranked_table_and_report(tmp_path):
    stdout, report = _run(tmp_path, ", nsteps=300, burnin=100", cpu=True)
    _check(stdout, report)


@pytest.mark.slow
def test_six_lines_at_the_default_settings(tmp_path):
    stdout, report = _run(tmp_path, "", cpu=False)
    _check(stdout, report)
