"""Legacy names stay out of the tree, and every tracked Python file compiles.

whisper_cbpf supersedes two discontinued repositories, WHISPER_AI (package ``whisper_labia``) and
whisper-GPU (``whisper_gpu``); see docs/MIGRATION.md. Their names, and the private containers and
paths they were developed in, may appear only as commit-pinned provenance. Private paths are never
an instruction: a user cannot ``cd`` into someone else's host.

The compile check lives here because a find-and-replace of those names once left
``tests/t3_inference/_stage_common.py`` a SyntaxError, and nothing noticed: pytest never collects
that file, so only a check over every tracked file can see it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The old package and repository names, the old containers, and their private paths. No
#: look-ahead after ``gpu``: ``WHISPER_GPU_VENV`` and ``/opt/jax_gpu_venv`` must match too.
LEGACY = re.compile(
    r"whisper[_-]?(?:ai|labia|gpu)|phe_sbi|/tf/astrodados|/opt/jax_gpu_venv"
    r"|/opt/(?:conda|jax_gpu_venv)/lib/python3\.11",
    re.IGNORECASE)

#: A line may name them only as pinned provenance, or say they are discontinued.
PINNED = re.compile(
    r"@ ?`?(?:8ba3843|10796a0)\b"
    r"|/(?:blob|tree)/(?:8ba3843|10796a0)\b"
    r"|raw\.githubusercontent\.com/phelipedarc/WHISPER_AI/8ba3843/"
    r"|discontinued"
    r"|\$\{WHISPER_GPU_VENV:-",               # env.sh's fallback to the old variable name
    re.IGNORECASE)

#: The migration guide is a table of old names, and this file spells out the patterns.
EXEMPT = {"docs/MIGRATION.md", "tests/test_no_legacy_names.py"}


def _tracked(suffix=""):
    """Tracked files that exist in the working tree; skips outside a git checkout (an sdist)."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    paths = [ROOT / p for p in out.stdout.decode().split("\0") if p.endswith(suffix)]
    return [p for p in paths if p.is_file()]


def test_legacy_names_appear_only_as_pinned_provenance():
    # Notebooks are scanned as stored: nbformat writes one source or output line per JSON line,
    # so the stored outputs are checked too, and they are where private paths leak.
    offences = []
    for path in _tracked():
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:                 # binary: goldens, images
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if LEGACY.search(line) and not PINNED.search(line):
                offences.append(f"{rel}:{n}: {line.strip()[:140]}")
    assert not offences, (f"{len(offences)} unpinned legacy name(s):\n" + "\n".join(offences))


def test_every_tracked_python_file_compiles():
    errors = []
    for path in _tracked(".py"):
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT).as_posix()}:{exc.lineno}: {exc.msg}")
    assert not errors, "\n".join(errors)
