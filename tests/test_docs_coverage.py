"""The documentation covers the public API, and its code runs.

The claims, each a test:

1. **Every public name is documented with an example.** Each function and class in
   ``whisper_cbpf.__all__`` has a docstring with a numpy-style ``Examples`` section. Names that do
   not have one yet are listed in :data:`MISSING_EXAMPLES` and reported as expected failures, so the
   list can only shrink: a name added to ``__all__`` without an example fails here.
2. **Every public name is in the API reference**: it appears in a code span of
   ``docs/API_REFERENCE.md``.
3. **The code blocks parse**: every ``python`` block of ``README.md``, ``INSTALL.md`` and
   ``docs/*.md`` (``README.md`` and ``docs/LSST_ALERTS.md`` must have some).
4. **The README quickstart runs end to end** on the synthetic LSST alert shipped in ``tests/data``
   (``make_lsst_alert_sn.py`` writes it): load the alert, compare two models, print the summary,
   write the HTML report. The quickstart runs as written, except that ``wp.compare`` gets a short
   chain and no evidence check, so the test takes about two minutes on a CPU instead of the
   documented 9-10 minutes per alert.
"""
from __future__ import annotations

import ast
import functools
import inspect
import re
import shutil
import warnings
from pathlib import Path

import pytest

import whisper_cbpf as wp

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
API_REFERENCE = ROOT / "docs" / "API_REFERENCE.md"
LSST_ALERTS = ROOT / "docs" / "LSST_ALERTS.md"
ALERT = ROOT / "tests" / "data" / "lsst_alert_sn.json"

#: Public data objects (a version string, tables, a palette): they have no docstring of their own,
#: and are only required to appear in the API reference.
DATA_NAMES = {"__version__", "CORNER_PALETTE", "FILTER_LOOKUP", "LSST_BAND_INFO"}

#: Public functions and classes whose docstring has no ``Examples`` section yet. Each is an expected
#: failure of the example check; remove a name here once its docstring has one.
MISSING_EXAMPLES = {
    "LightCurve", "plot_light_curve", "plot_calibration", "group_bands", "resolve_band",
    "resolve_bands", "SvoUnavailable", "register_manual_band", "unregister_manual_band",
    "clear_manual_bands", "resolve_filter", "set_default_band_system", "default_band_system",
    "Prior", "Uniform", "LogUniform", "Model", "register_model", "get_model", "list_models",
    "redback_model", "register_redback", "chi2_distance", "GaussianLikelihood",
    "GaussianLikelihoodWithScatter", "MixtureGaussianLikelihood", "register_likelihood",
    "list_likelihoods", "fit_MCMC", "SamplerResult", "waic",
    "per_band_metrics", "predictive_metrics", "recovery_metrics", "posterior_predictive_check",
    "sbc_rank", "sbc_ranks", "check_gpu", "require_jax", "x64_enabled", "gpu_list", "n_jobs",
    "env_script", "env_report", "register_kilonova", "register_kilonova_two",
    "register_kilonova_three", "register_tde", "register_supernova", "supernova_models",
    "flare_model",
}

_EXAMPLES = re.compile(r"^\s*Examples\s*\n\s*-{8,}\s*$", re.MULTILINE)


def _documented_names():
    return [n for n in wp.__all__ if n not in DATA_NAMES]


def _python_blocks(path):
    """The ``python`` fenced code blocks of a Markdown file, as ``(first line, source)``."""
    text = path.read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"^```python[^\n]*\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL):
        out.append((text[:m.start()].count("\n") + 1, m.group(1)))
    return out


def _as_source(block):
    """A block as plain Python: doctest prompts removed, their output lines dropped."""
    lines = block.splitlines()
    if not any(line.lstrip().startswith(">>>") for line in lines):
        return block
    src = []
    for line in lines:
        s = line.lstrip()
        if s.startswith(">>> ") or s == ">>>":
            src.append(s[4:])
        elif s.startswith("... ") or s == "...":
            src.append(s[4:])
    return "\n".join(src)


# --- 1. every public name has an example ------------------------------------------------------------
def test_the_lists_name_only_public_names():
    stale = sorted((MISSING_EXAMPLES | DATA_NAMES) - set(wp.__all__))
    assert not stale, f"not in whisper_cbpf.__all__ any more; remove them from this file: {stale}"


@pytest.mark.parametrize("name", _documented_names())
def test_every_public_name_has_a_docstring_with_an_example(name):
    obj = getattr(wp, name)
    doc = inspect.getdoc(obj) or ""
    has_example = bool(_EXAMPLES.search(doc))
    if name in MISSING_EXAMPLES and not has_example:
        pytest.xfail(f"{name}: its docstring has no Examples section yet")
    assert doc.strip(), f"whisper_cbpf.{name} has no docstring."
    assert has_example, (f"the docstring of whisper_cbpf.{name} has no 'Examples' section. Add a "
                         f"numpy-style Examples section with a runnable example.")


# --- 2. every public name is in the API reference ---------------------------------------------------
@pytest.fixture(scope="module")
def api_reference_spans():
    text = API_REFERENCE.read_text(encoding="utf-8")
    return re.findall(r"`([^`\n]+)`", text)


@pytest.mark.parametrize("name", list(wp.__all__))
def test_every_public_name_is_in_the_api_reference(name, api_reference_spans):
    word = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    assert any(word.search(span) for span in api_reference_spans), (
        f"whisper_cbpf.{name} is not in docs/API_REFERENCE.md (as a `code` span). Document it "
        f"there.")


# --- 3. the code blocks parse -------------------------------------------------------------------------
DOCS = [README, ROOT / "INSTALL.md"] + sorted((ROOT / "docs").glob("*.md"))


def test_the_quickstart_pages_exist():
    assert README.is_file() and LSST_ALERTS.is_file() and ALERT.is_file()


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_every_python_block_parses(path):
    blocks = _python_blocks(path)
    if path in (README, LSST_ALERTS):
        assert blocks, f"{path.name} has no python code block."
    for line, block in blocks:
        try:
            ast.parse(_as_source(block))
        except SyntaxError as exc:
            pytest.fail(f"{path.name}, the python block at line {line} does not parse: {exc}")


# --- 4. the README quickstart runs ----------------------------------------------------------------------
def _quickstart():
    for _, block in _python_blocks(README):
        if 'survey="lsst"' in block and "wp.compare(" in block and ".report(" in block:
            return block
    raise AssertionError("README.md has no quickstart block (load_lightcurve(survey='lsst') -> "
                         "compare -> report).")


def test_the_readme_quickstart_is_at_most_six_lines():
    code = [line for line in _quickstart().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    assert len(code) <= 6, f"the README quickstart has {len(code)} lines of code: {code}"


def test_the_readme_quickstart_runs_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    import jax

    (tmp_path / "tests" / "data").mkdir(parents=True)
    shutil.copy(ALERT, tmp_path / "tests" / "data" / ALERT.name)
    monkeypatch.chdir(tmp_path)                    # "out/" and the alert path resolve here
    fast = functools.partial(wp.compare, nwalkers=24, nsteps=300, burnin=150,
                             evidence_check=False)
    monkeypatch.setattr(wp, "compare", fast)
    x64 = jax.config.jax_enable_x64
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # short chains: the fits say they are unconverged
            namespace = {}
            exec(compile(_quickstart(), "README quickstart", "exec"), namespace)
    finally:
        jax.config.update("jax_enable_x64", x64)
    cmp = namespace["cmp"]
    assert isinstance(cmp, wp.Comparison)
    assert list(cmp.table["model"]) and cmp.winner in set(cmp.table["model"])
    assert cmp.lc.meta["survey"] == "lsst"
    html = tmp_path / "out" / "report.html"
    assert html.is_file() and html.stat().st_size > 10_000
    assert cmp.winner in html.read_text(encoding="utf-8")
