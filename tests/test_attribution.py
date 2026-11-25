"""Attribution: what ``THIRD_PARTY_NOTICES.md`` lists is in the package and says where it is from.

1. Every file the notices list exists.
2. Every adapted source file begins with the attribution header of its origin: "Adapted from
   redback-jax" (for the files adapted from redback-jax and redback) or "Adapted from redback" (for
   the file adapted from redback alone), with the licence and the commit; the two files adapted
   from MOSFiT carry MOSFiT's copyright and MIT permission notice, which then ships in a wheel too.
3. Every adopted model's docstring has a References section.
4. The README names both projects and links the notices, and ``CITATION.cff`` lists both
   repositories and MOSFiT.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
_PATH = re.compile(r"`(whisper_cbpf/[\w/.]+\.(?:py|txt|npz))`")
_REFERENCES = re.compile(r"^\s*References\s*\n\s*-{8,}\s*$", re.MULTILINE)


def _sections():
    """``{heading: [path, ...]}`` of the table rows and prose of THIRD_PARTY_NOTICES.md."""
    out, heading = {}, None
    for line in NOTICES.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            out[heading] = []
        elif heading is not None:
            out[heading].extend(_PATH.findall(line))
    return out


def _head(path, n=40):
    """The first ``n`` lines of a source file, one line, whitespace collapsed."""
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()[:n]
    return " ".join(" ".join(lines).split())


SECTIONS = _sections()
BOTH = SECTIONS["Adapted from redback-jax and from redback"]
REDBACK_ONLY = SECTIONS["Adapted from redback only"]
SOURCE_BOTH = [p for p in BOTH if p.endswith(".py")]


def test_the_notices_list_the_adapted_files():
    assert {Path(p).name for p in SOURCE_BOTH} == {
        "phenomenological.py", "_phenomenological_spec.py", "tde_fallback.py",
        "kilonova_metzger.py", "csm_interaction.py", "shock_models.py", "magnetar_ode.py"}
    assert [Path(p).name for p in REDBACK_ONLY] == [
        "tde_analytical.py", "tde_stream_stream.py"]
    assert {Path(p).name for p in SECTIONS["Data tables"]} == {
        "csm_table.txt", "tde_fallback_nodes.npz", "tde_fallback_tables.npz"}
    assert {Path(p).name for p in SECTIONS["Adapted from MOSFiT"] if p.endswith(".py")} == {
        "tde_mosfit.py", "tde_fallback.py"}


@pytest.mark.parametrize("path", sorted({p for paths in SECTIONS.values() for p in paths}))
def test_every_file_the_notices_list_exists(path):
    assert (ROOT / path).is_file(), f"THIRD_PARTY_NOTICES.md lists {path}, which does not exist"


@pytest.mark.parametrize("path", SOURCE_BOTH)
def test_files_adapted_from_redback_jax_say_so(path):
    head = _head(path)
    assert "Adapted from redback-jax" in head and "https://github.com/nikhil-sarin/redback-jax" in head
    assert "commit 3431e2e" in head and "GPL-3.0" in head and "redback (Sarin et al.)" in head


def test_the_file_adapted_from_redback_alone_says_so():
    head = _head("whisper_cbpf/models/jax/tde_analytical.py")
    assert "Adapted from redback (Nikhil Sarin et al.)" in head and "GPL-3.0" in head
    assert "https://github.com/nikhil-sarin/redback," in head
    assert "no code is taken from it" in head          # from redback-jax


def test_the_supernova_module_says_which_models_are_adapted():
    head = _head("whisper_cbpf/models/jax/supernova.py", 60).lower()
    assert "adapted from redback-jax" in head and "commit 3431e2e" in head and "gpl-3.0" in head


def test_the_data_tables_are_not_empty():
    assert (ROOT / "whisper_cbpf/data/csm_table.txt").stat().st_size > 10_000
    assert (ROOT / "whisper_cbpf/data/tde_fallback_nodes.npz").stat().st_size > 1_000_000
    assert (ROOT / "whisper_cbpf/data/tde_fallback_tables.npz").stat().st_size > 100_000


#: The MIT permission notice, as MOSFiT's LICENSE states it (collapsed whitespace).
_MIT_PERMISSION = ("Permission is hereby granted, free of charge, to any person obtaining a copy of "
                   "this software and associated documentation files")
_MIT_CONDITION = ("The above copyright notice and this permission notice shall be included in all "
                  "copies or substantial portions of the Software.")


@pytest.mark.parametrize("path", ["whisper_cbpf/models/jax/tde_fallback.py",
                                  "whisper_cbpf/models/jax/tde_mosfit.py"])
def test_the_files_adapted_from_mosfit_say_so(path):
    """The fall-back engine and the ``tde`` model are MOSFiT's: each file's header names MOSFiT, the
    commit and the copyright, and the file itself carries MOSFiT's copyright and permission notice
    (the wheel ships the code but not THIRD_PARTY_NOTICES.md); the notices reproduce the licence."""
    head = _head(path)
    assert "dapted from MOSFiT" in head and "(James Guillochon et al.)" in head
    assert "https://github.com/guillochon/MOSFiT" in head and "commit 0bc772e" in head
    assert "MIT licence, Copyright (c) 2016-2026 James Guillochon" in head
    body = " ".join((ROOT / path).read_text(encoding="utf-8").replace("#", " ").split())
    assert "MIT License Copyright (c) 2016-2026 James Guillochon" in body
    assert _MIT_PERMISSION in body and _MIT_CONDITION in body
    assert 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND' in body
    notices = NOTICES.read_text(encoding="utf-8")
    assert "> Copyright (c) 2016-2026 James Guillochon" in notices


#: (module, attribute or None for the module docstring) of each adopted model.
ADOPTED_DOCSTRINGS = [
    ("phenomenological", "phenomenological_model"),
    ("tde_fallback", "tde_fallback_model"),
    ("tde_mosfit", "tde_mosfit_model"),
    ("tde_analytical", "tde_analytical_model"),
    ("tde_stream_stream", "tde_stream_stream_model"),
    ("kilonova_metzger", "metzger_kilonova_model"),
    ("kilonova_metzger", "magnetar_boosted_kilonova_model"),
    ("csm_interaction", None),
    ("shock_models", None),
    ("magnetar_ode", None),
    ("supernova", "shock_cooling_model"),
]


@pytest.mark.parametrize("module,attr", ADOPTED_DOCSTRINGS)
def test_every_adopted_model_cites_its_references(module, attr):
    pytest.importorskip("jax")
    mod = importlib.import_module(f"whisper_cbpf.models.jax.{module}")
    doc = mod.__doc__ if attr is None else getattr(mod, attr).__doc__
    assert doc and _REFERENCES.search(doc), f"{module}.{attr or '<module>'} has no References section"


def test_the_readme_and_citation_name_both_projects():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "redback" in readme and "redback-jax" in readme and "THIRD_PARTY_NOTICES.md" in readme
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "https://github.com/nikhil-sarin/redback\"" in cff or \
        "https://github.com/nikhil-sarin/redback\n" in cff
    assert "https://github.com/nikhil-sarin/redback-jax" in cff
    assert "https://github.com/guillochon/MOSFiT" in cff


def test_the_notices_ship_in_the_sdist():
    assert "THIRD_PARTY_NOTICES.md" in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
