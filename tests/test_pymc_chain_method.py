"""`parallel` is not always parallel, and a benchmark must not claim it was.

NumPyro downgrades ``chain_method="parallel"`` to ``"sequential"`` whenever it has fewer devices
than chains. The downgrade is a ``warnings.warn`` deep inside ``MCMC.__init__``, which is invisible
in a benchmark log, so a four-chain run on two GPUs reports itself as the multi-GPU arm while
executing entirely on one device. That is not a slow path, it is a wrong measurement: it compares a
single-GPU run against a single-GPU run and calls the ratio a speedup.

The suite this file belongs to previously encoded the opposite belief -- that pmap performs
``ceil(chains / devices)`` passes -- and warned on ``num_chains % n_devices``. Four chains on two
GPUs has remainder zero, so the commonest failure was exactly the silent one.

These tests need neither PyMC nor a GPU: the rule is arithmetic, and the one test that does need
NumPyro reads its source rather than sampling.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

# The repo root, where whisper_cbpf lives. Not its parent: in a checkout whose parent also holds a
# redback clone, that directory would shadow the installed redback as an empty namespace package for
# every test that runs after this module is collected.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper_cbpf.samplers.jax.pymc_gpu import _effective_chain_method  # noqa: E402


@pytest.mark.parametrize("chains,devices,expected", [
    # The case that shipped silently: remainder zero, and still fully sequential.
    (4, 2, "sequential"),
    (4, 1, "sequential"),
    (2, 1, "sequential"),
    (8, 3, "sequential"),
    # One device per chain is the only regime that actually pmaps.
    (4, 4, "parallel"),
    (1, 1, "parallel"),
    # More devices than chains still parallelises; the spares just idle.
    (2, 4, "parallel"),
])
def test_parallel_downgrades_unless_every_chain_has_a_device(chains, devices, expected):
    assert _effective_chain_method("parallel", chains, devices) == expected


@pytest.mark.parametrize("devices", [1, 2, 4, 8])
def test_vectorized_and_sequential_are_never_downgraded(devices):
    """Only `parallel` carries a device requirement; the others run anywhere."""
    for method in ("vectorized", "sequential"):
        assert _effective_chain_method(method, 4, devices) == method


def test_matches_numpyros_actual_source():
    """Pin the rule to NumPyro's code, not to my reading of its docs.

    If a future NumPyro changes the comparison (say to `<=`, or to pad chains across devices),
    `_effective_chain_method` becomes wrong in a way no amount of local unit testing would reveal
    -- the sampler would keep reporting a chain method it no longer runs. Reading the source is
    ugly, but it fails loudly at the moment the assumption breaks.
    """
    numpyro_mcmc = pytest.importorskip(
        "numpyro.infer.mcmc", reason="numpyro not importable in this container")
    import inspect

    src = inspect.getsource(numpyro_mcmc.MCMC)
    src = re.sub(r"\s+", " ", src)
    guard = ('if chain_method == "parallel" and local_device_count() < self.num_chains: '
             'chain_method = "sequential"')
    assert guard in src, (
        "NumPyro's parallel-to-sequential downgrade rule is no longer the one "
        "_effective_chain_method models. Re-read MCMC.__init__ and update both.")


def test_result_survives_to_json():
    """These two samplers were the only ones whose result could not be saved.

    ``fit`` passed ``min_distance=None``; ``SamplerResult.to_dict`` does
    ``float(self.min_distance)``; ``None`` is not a float. So ``result.to_json()`` -- the documented
    way to persist a fit -- raised ``TypeError`` after a multi-minute GPU run, and only here. NaN is
    the field's declared "this sampler has no distance" value, and ``nuts_gpu``, ``mcmc`` and
    ``snpe`` all leave it at that default.

    Checked without PyMC or a GPU, the way `test_matches_numpyros_actual_source` is: against the
    source of the construction, plus the serialisation it has to survive.
    """
    import inspect
    import json

    import pandas as pd

    from whisper_cbpf.samplers.base import SamplerResult
    from whisper_cbpf.samplers.jax import pymc_gpu

    assert "min_distance=None" not in inspect.getsource(pymc_gpu._PyMCJAXBase.fit)

    result = SamplerResult(
        sampler="pymc_jax_gpu_vectorized", model="m", parameters=["a"],
        samples=pd.DataFrame({"a": [1.0]}), summary={}, best_params={"a": 1.0},
        n_data=1, n_params=1, runtime_s=1.0, max_log_likelihood=-1.0, aic=1.0, bic=1.0)
    assert json.loads(result.to_json())["sampler"] == "pymc_jax_gpu_vectorized"
