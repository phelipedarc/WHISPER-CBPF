"""The GPU bootstrap, ``whisper_cbpf/backends/env.sh``, sourced against a throwaway virtualenv.

A venv named before sourcing must go first on ``PATH``, under the current variable name and the
discontinued one it still honours, and its nvidia libraries must be found from wherever that
interpreter's site-packages actually is. ``LD_LIBRARY_PATH`` must never gain an empty entry: the
dynamic loader reads one as the current directory.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ENV_SH = Path(__file__).resolve().parents[1] / "whisper_cbpf" / "backends" / "env.sh"
VENV_VARS = ("WHISPER_CBPF_VENV", "WHISPER_GPU_VENV")    # the second is discontinued, still honoured

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="env.sh is a bash script")


@pytest.fixture(scope="module")
def venv(tmp_path_factory):
    """A real venv (no pip) with an empty ``nvidia/cudnn/lib`` in its site-packages."""
    root = tmp_path_factory.mktemp("venv")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True)
    site = subprocess.run(
        [str(root / "bin" / "python3"), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True, text=True, check=True).stdout.strip()
    libdir = Path(site) / "nvidia" / "cudnn" / "lib"
    libdir.mkdir(parents=True)
    return root, libdir


def _source(**env):
    """``PATH`` and ``LD_LIBRARY_PATH`` after sourcing env.sh, each split on ':'."""
    base = {k: v for k, v in os.environ.items() if k not in VENV_VARS + ("LD_LIBRARY_PATH",)}
    out = subprocess.run(
        ["bash", "-c", 'source "$0" >/dev/null && printf "%s\\n%s\\n" "$PATH" "$LD_LIBRARY_PATH"',
         str(ENV_SH)],
        env={**base, **env}, capture_output=True, text=True, check=True)
    path, ld = out.stdout.split("\n")[:2]
    return path.split(":"), ld.split(":")


@pytest.mark.parametrize("var", VENV_VARS)
def test_named_venv_goes_first_and_its_nvidia_libs_are_found(venv, var):
    root, libdir = venv
    path, ld = _source(**{var: str(root)})
    assert path[0] == str(root / "bin")
    assert os.path.realpath(ld[0]) == os.path.realpath(libdir), ld
    assert "" not in ld, ld


def test_no_venv_leaves_path_alone_and_adds_no_empty_entry():
    path, ld = _source()
    assert ":".join(path) == os.environ["PATH"]
    assert "" not in ld, ld


def test_an_existing_ld_library_path_is_kept_last(venv):
    root, _ = venv
    _, ld = _source(WHISPER_CBPF_VENV=str(root), LD_LIBRARY_PATH="/opt/mine")
    assert ld[-1] == "/opt/mine"
    assert "" not in ld, ld


@pytest.mark.parametrize("before,after", [(None, []), ("/opt/mine", ["/opt/mine"])])
def test_pythonpath_gains_the_checkout_and_no_empty_entry(before, after):
    """The same hazard in ``PYTHONPATH``: ``"<repo>:"`` puts the current directory on ``sys.path``."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if before is not None:
        env["PYTHONPATH"] = before
    out = subprocess.run(["bash", "-c", 'source "$0" >/dev/null && printf "%s" "$PYTHONPATH"',
                          str(ENV_SH)], env=env, capture_output=True, text=True, check=True).stdout
    entries = out.split(":")
    assert os.path.realpath(entries[0]) == os.path.realpath(ENV_SH.parents[2])   # the checkout
    assert entries[1:] == after, entries
