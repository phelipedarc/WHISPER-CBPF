"""The package's public names: every 0.2.0 entry point is exported, real, and importable without the
optional extras.

* Every name of waves 1 and 2 is in ``whisper_cbpf.__all__`` and resolves to the real object, not
  to the stand-in the package binds when a module or an optional dependency is missing.
* ``import whisper_cbpf`` works with no jax, torch, sbi or redback (the ``[gpu]``, ``[sbi]`` and
  ``[models]`` extras), and those names are still the real functions.
* ``compare``, ``forecast``, ``report``, ``profile`` and ``likelihood_max_opt`` are also module
  names; importing the modules must not replace the functions on the package.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import whisper_cbpf as wp

NEW_CORE = ["Normal", "TruncatedNormal", "Fixed", "max_abs_z_distance", "luminosity_distance_cm",
          "likelihood_max_opt", "LikelihoodMaxOptResult", "load_result", "fit_cached",
          "DiagnosticsReport",
          "check_parity", "run_jobs", "Job"]
NEW_WORKFLOWS = ["compare", "Comparison", "forecast", "discriminate", "plot_forecast", "plot_models",
          "plot_model_comparison", "plot_widths", "result_facts", "comparison_facts",
          "write_facts", "report", "log_density", "fit_batch", "profile", "capacity"]


@pytest.mark.parametrize("name", NEW_CORE + NEW_WORKFLOWS)
def test_every_new_name_is_exported_and_real(name):
    assert name in wp.__all__
    obj = getattr(wp, name)
    assert callable(obj)
    assert getattr(obj, "whisper_unavailable", None) is None, obj.whisper_unavailable
    assert obj.__doc__ and obj.__doc__.strip(), f"wp.{name} has no docstring"


def test_all_is_unique_and_every_entry_resolves():
    assert len(wp.__all__) == len(set(wp.__all__))
    missing = [n for n in wp.__all__ if not hasattr(wp, n)]
    assert missing == []


def test_function_names_that_are_also_module_names_stay_functions():
    import importlib

    for name in ("compare", "forecast", "report", "profile", "likelihood_max_opt"):
        importlib.import_module(f"whisper_cbpf.{name}")
        assert callable(getattr(wp, name)) and not isinstance(getattr(wp, name), type(sys)), name


def test_import_needs_no_jax_torch_sbi_or_redback():
    """A fresh interpreter where those packages cannot be imported at all."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('jax', 'jaxlib', 'torch', 'sbi', 'redback', 'numpyro',\n"
        "                                  'pymc', 'bilby'):\n"
        "            raise ImportError(f'{name} is blocked for this test')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import whisper_cbpf as wp\n"
        f"names = {NEW_CORE + NEW_WORKFLOWS!r}\n"
        "stubs = [n for n in names if getattr(getattr(wp, n), 'whisper_unavailable', None)]\n"
        "assert stubs == [], stubs\n"
        "assert 'nuts_gpu' in wp.list_samplers()\n"
        "assert not any(m.split('.')[0] in ('jax', 'torch', 'redback') for m in sys.modules)\n"
        "print('ok')\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), run.stderr[-2000:]


def test_a_missing_module_leaves_a_stand_in_that_says_why():
    stub = wp._unavailable("compare", "whisper_cbpf.compare",
                           ModuleNotFoundError("No module named 'whisper_cbpf.compare'"))
    assert stub.__name__ == "compare" and "Not available" in stub.__doc__
    with pytest.raises(ImportError, match=r"whisper_cbpf.compare is not available.*No module"):
        stub(None, ["arnett"])
