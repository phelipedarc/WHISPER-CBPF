"""The one-file HTML report: offline, deterministic, results only, facts-backed."""
import base64
import json
import re
import warnings
from html.parser import HTMLParser

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf import plotting
from whisper_cbpf.compare import Comparison
from whisper_cbpf.report import REPORT_NAME, report

FORECAST_TIMES = [32.0, 36.0]


def toy_lc(n=30, name="toy flare"):
    t = np.linspace(0.5, 30.0, n)
    flux = wp.get_model("flare").predict({"amplitude": 5.0, "rise_time": 3.0,
                                          "decay_time": 15.0}, t, None)
    bands = (["ztfg", "ztfr"] * n)[:n]
    return wp.LightCurve(time=t, band=bands, flux=flux, flux_err=np.full(n, 0.1), name=name,
                         redshift=0.05)


@pytest.fixture(scope="module")
def cmp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.compare(toy_lc(), ["flare", "bazin"], nsteps=1500, burnin=500,
                          evidence_check=False)


def build(comparison, out, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(comparison, None, out, **kw)


class Page(HTMLParser):
    """Every tag with its attributes, and the text, of an HTML page."""

    def __init__(self):
        super().__init__()
        self.tags, self.text, self.open = [], [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in ("meta", "img", "br", "hr", "input", "link"):
            self.open.append(tag)

    def handle_endtag(self, tag):
        assert self.open and self.open[-1] == tag, f"</{tag}> closes <{self.open[-1:]}>"
        self.open.pop()

    def handle_data(self, data):
        self.text.append(data)


def parse(text):
    page = Page()
    page.feed(text)
    page.close()
    assert page.open == [], f"unclosed tags: {page.open}"
    return page


# =================================================================================== offline
def test_the_page_is_self_contained_and_renders_offline(cmp, tmp_path):
    path = build(cmp, tmp_path, forecast_times=FORECAST_TIMES)
    assert path == tmp_path / REPORT_NAME
    text = path.read_text(encoding="utf-8")
    page = parse(text)
    names = [t for t, _ in page.tags]
    assert "script" not in names and "link" not in names and "iframe" not in names
    assert not re.search(r"https?://|@import|url\(", text)
    pngs = 0
    for tag, attrs in page.tags:
        if "src" in attrs:
            assert attrs["src"].startswith("data:image/png;base64,")
            assert base64.b64decode(attrs["src"].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
            pngs += 1
        if "href" in attrs:
            assert attrs["href"].startswith(("#", "data:application/json;base64,"))
    assert pngs >= 3
    # readable on a phone: viewport, a bounded column, scrolling tables, fluid images
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in text
    assert "max-width:60rem" in text and "overflow-x:auto" in text and "width:100%" in text
    tables = [a for t, a in page.tags if t == "table"]
    assert len(tables) == text.count('<div class="scroll"><table>')
    for anchor in ("answer", "ranking", "figures", "parameters", "data", "diagnostics",
                   "forecast", "provenance"):
        assert f'id="{anchor}"' in text


def test_the_embedded_facts_file_is_the_comparisons_facts(cmp, tmp_path):
    text = build(cmp, tmp_path / "r.html").read_text()
    href = re.search(r'href="data:application/json;base64,([^"]+)"', text).group(1)
    embedded = json.loads(base64.b64decode(href))
    facts = cmp.facts()
    assert embedded == facts
    assert facts["sha256"] in text


# ============================================================================== determinism
def test_two_builds_are_byte_identical(cmp, tmp_path):
    a = build(cmp, tmp_path / "a", forecast_times=FORECAST_TIMES).read_bytes()
    b = build(cmp, tmp_path / "b", forecast_times=FORECAST_TIMES).read_bytes()
    assert a == b


def test_rebuilding_from_saved_results_gives_identical_html(cmp, tmp_path):
    first = build(cmp, tmp_path / "first", forecast_times=FORECAST_TIMES).read_bytes()
    cmp.save(tmp_path / "saved")
    loaded = Comparison.load(tmp_path / "saved")
    again = build(loaded, tmp_path / "again", forecast_times=FORECAST_TIMES).read_bytes()
    assert first == again
    assert str(tmp_path) not in first.decode()                    # no path on the page


# ============================================================================== the content
def test_ranking_table_shows_the_facts(cmp, tmp_path):
    text = build(cmp, tmp_path).read_text()
    facts = cmp.facts()
    section = text[text.index('id="ranking"'):text.index('id="figures"')]
    for row in facts["ranking"]["models"]:
        assert f"<b>{row['model']}</b>" in section
        assert f"{row['bic']:.1f}" in section and f"{row['delta_bic']:.1f}" in section
        assert f"{row['max_log_likelihood']:.2f}" in section
    assert facts["ranking"]["reading"] in text
    diag = text[text.index('id="diagnostics"'):text.index('id="forecast"')]
    for name, core in facts["models"].items():
        for r in core["diagnostics"]["rows"]:
            assert r["check"] in diag


def test_decision_slot_appears_only_when_passed_and_is_escaped(cmp, tmp_path):
    plain = build(cmp, tmp_path / "plain").read_text()
    assert "Decision (supplied by the caller" not in plain
    text = build(cmp, tmp_path / "d1", decision="<b>observe</b> tonight & again").read_text()
    assert "Decision (supplied by the caller, not computed by whisper)" in text
    assert "&lt;b&gt;observe&lt;/b&gt; tonight &amp; again" in text
    text = build(cmp, tmp_path / "d2", decision={"action": "spectrum", "priority": 1}).read_text()
    assert "<td>action</td><td>spectrum</td>" in text and "<td>priority</td><td>1</td>" in text


def test_forecast_section(cmp, tmp_path):
    text = build(cmp, tmp_path / "no").read_text()
    assert "No forecast was asked for" in text
    text = build(cmp, tmp_path / "yes", forecast_times=FORECAST_TIMES).read_text()
    section = text[text.index('id="forecast"'):text.index('id="provenance"')]
    assert "<td class=\"num\">32.00</td>" in section and "Where the models differ most" in section
    assert 'id="fig-forecast"' in text


def test_a_missing_plot_function_is_reported_not_fatal(cmp, tmp_path, monkeypatch):
    monkeypatch.delattr(plotting, "plot_widths")
    with pytest.warns(UserWarning, match="'widths' figure is left out"):
        path = report(cmp, None, tmp_path)
    text = path.read_text()
    assert 'id="fig-widths"' not in text
    assert "plot_widths is not part of this installation" in text


def test_not_enough_data_is_said_not_printed(tmp_path):
    lc = toy_lc(n=3, name="three points")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        few = wp.compare(lc, ["flare"], nsteps=400, burnin=100, evidence_check=False)
        text = report(few, None, tmp_path).read_text()
    assert "Not enough data: no model could be ranked." in text
    assert "Left out of the ranking: flare (not enough data" in text
    params = text[text.index('id="parameters"'):text.index('id="data"')]
    assert "No parameter values are shown" in params and "<th>Median</th>" not in params
    parse(text)


def _toy_shape(params, t, bands, decay, rise=2.0):
    """A rise-and-fade light curve in Jy that starts at ``t_exp``; bluer bands are brighter."""
    dt = np.clip(np.asarray(t, dtype=float) - params["t_exp"], 0.0, None)
    col = np.array([{"lsstg": 1.0, "lsstr": 0.8}.get(str(b), 1.0) for b in bands])
    return np.where(dt > 0, params["amplitude"] * 1e-4 * col * (1.0 - np.exp(-dt / rise))
                    * np.exp(-dt / decay), 0.0)


@pytest.fixture(scope="module")
def alert_cmp():
    """An LSST-like alert (two bands, limits before the first detection) and three toy models:
    one fits the redshift, which it ignores (so its posterior is the prior), and one has a fixed
    explosion time, so it keeps the pre-event limits and is left out (different n)."""
    from whisper_cbpf.priors import LogUniform, Prior, Uniform
    wp.register_model("report_toy_z", lambda p, t, b: _toy_shape(p, t, b, 12.0),
                      ["amplitude", "t_exp", "redshift"], overwrite=True,
                      prior=Prior({"amplitude": LogUniform(0.1, 10.0), "t_exp": Uniform(-6.0, 0.0),
                                   "redshift": Uniform(0.01, 0.2)}))
    wp.register_model("report_toy_decay", lambda p, t, b: _toy_shape(p, t, b, p["decay"], 6.0),
                      ["amplitude", "t_exp", "decay"], overwrite=True,
                      prior=Prior({"amplitude": LogUniform(0.1, 10.0), "t_exp": Uniform(-6.0, 0.0),
                                   "decay": Uniform(2.0, 40.0)}))
    wp.register_model("report_toy_fixed", lambda p, t, b: _toy_shape(dict(p, t_exp=-1.0), t, b,
                                                                     p["decay"]),
                      ["amplitude", "decay"], overwrite=True,
                      prior=Prior({"amplitude": LogUniform(0.1, 10.0),
                                   "decay": Uniform(2.0, 40.0)}))
    t = np.array([-5.0, -3.0, 0.0, 0.3, 2.0, 2.4, 5.0, 5.5, 9.0, 9.2, 14.0, 14.5])
    b = np.array(["lsstg", "lsstr"] * 6)
    flux = _toy_shape({"amplitude": 2.0, "t_exp": -2.0}, t, b, 12.0)
    mag = -2.5 * np.log10(np.clip(flux, 1e-30, None) / 3631.0) \
        + np.random.default_rng(0).normal(0.0, 0.05, t.size)
    ul = t < -2.5
    mag[ul], err = 21.0, np.where(ul, np.nan, 0.05)
    lc = wp.LightCurve(time=t, band=b, magnitude=mag, magnitude_err=err, upper_limit=ul,
                       name="toy LSST alert")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wp.compare(lc, ["report_toy_z", "report_toy_decay", "report_toy_fixed"],
                          nsteps=800, burnin=200, evidence_check=False)


def test_an_alert_with_limits_and_a_fitted_redshift(alert_cmp, tmp_path):
    text = build(alert_cmp, tmp_path, forecast_times=[16.0, 20.0]).read_text()
    parse(text)
    facts = alert_cmp.facts()
    rk = facts["ranking"]
    assert {m["model"] for m in rk["models"]} == {"report_toy_z", "report_toy_decay"}
    assert [r["model"] for r in rk["left_out"]] == ["report_toy_fixed"]
    assert "Left out of the ranking: report_toy_fixed (fitted to n = 12" in text
    # the limits before the first detection are named, the colour is the latest pair
    data = text[text.index('id="data"'):text.index('id="diagnostics"')]
    assert "Last non-detection before it: lsstr fainter than 21.00 mag at -3.000." in data
    colour = facts["data"]["colours"][0]
    assert f"{colour['colour']:.2f} &#177; {colour['error']:.2f}" in data
    # redshift and explosion time narrowing, with the prior-dominated redshift flagged
    params = text[text.index('id="parameters"'):text.index('id="data"')]
    nz = facts["models"]["report_toy_z"]["narrowing"]
    assert nz["redshift"]["prior_dominated"] is True and nz["t_exp"]["prior_dominated"] is False
    assert ("<td>report_toy_z</td><td>redshift</td>" in params
            and "<td>report_toy_z</td><td>t_exp</td>" in params)
    assert f"{nz['t_exp']['width_ratio']:.2f}" in params
    assert "Not enough data to measure redshift" in params
    # the winner (the right rise time) fits the redshift: its absolute magnitude is asymmetric
    # and the page says the range restates the redshift prior
    assert rk["winner"] == "report_toy_z"
    am = facts["models"]["report_toy_z"]["absolute_magnitude"]["per_band"]["lsstg"]
    assert f"&#8722;{am['brighter_by']:.2f} / +{am['fainter_by']:.2f}" in data
    assert am["fainter_by"] > am["brighter_by"]
    assert "is prior-dominated, so this range restates the redshift prior" in data
    # forecasts for every band with a detection, both ranked models
    fc = text[text.index('id="forecast"'):text.index('id="provenance"')]
    for name in ("report_toy_z", "report_toy_decay"):
        assert f"<h3>{name}</h3>" in fc
    assert "<td>lsstg</td>" in fc and "<td>lsstr</td>" in fc


def test_title_and_output_paths(cmp, tmp_path):
    nested = tmp_path / "deep" / "dir" / "custom.html"
    path = build(cmp, nested, title="SN 2026abc <draft>")
    assert path == nested and path.exists()
    assert "<title>SN 2026abc &lt;draft&gt;</title>" in path.read_text()
    assert "<title>toy flare: model comparison</title>" in build(cmp, tmp_path).read_text()


def test_errors_name_the_fix(cmp, tmp_path):
    with pytest.raises(TypeError, match="Comparison from wp.compare"):
        report(object(), toy_lc(), tmp_path)

    class NoCurve:
        table, results, criterion = cmp.table, cmp.results, cmp.criterion
    with pytest.raises(TypeError, match="pass lc="):
        report(NoCurve(), None, tmp_path)
