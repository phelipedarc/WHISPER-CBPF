"""Keeping a fit: save and reload it with its provenance, skip it when it has already run, check it.

Three things a fit needs once it has finished, each built on the same record:

* **Save / load** (:meth:`SamplerResult.save <whisper_cbpf.samplers.base.SamplerResult.save>`,
  :func:`load_result`). A directory (``manifest.json``, ``result.json``, ``arrays.npz``) or one
  ``.npz`` file holding the draws (and, for the chain samplers, the draws by chain), the best fit,
  the metrics, ``info`` and a manifest: whisper version and git state, package versions, the model
  (name, description, parameters, prior), the sampler and its settings, the seed, the devices, the
  time split into compile and run where the sampler records it, and a hash of the data. Loading
  checks the draws and the result against the hashes written with them.
* **Resumable runs** (:func:`fit_cached`). A fit whose configuration (data, model and prior, sampler
  and settings, whisper version) matches a saved one is loaded instead of run, so a batch that was
  interrupted finishes only what is missing.
* **One convergence report** (:meth:`SamplerResult.diagnostics
  <whisper_cbpf.samplers.base.SamplerResult.diagnostics>`, :class:`DiagnosticsReport`). Every
  applicable check for the sampler that ran, each with its value, its threshold, pass or fail and
  a plain reason.

The provenance is recorded when the fit runs: every sampler class registered with whisper records
the call (see :class:`~whisper_cbpf.samplers.base.BaseSampler`) in ``result.provenance``.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .samplers.base import SamplerResult
from .samplers.jax import _diagnostics as _dg

#: What a saved result says it is, and the layout version this code writes and reads.
FORMAT = "whisper_cbpf.result"
FORMAT_VERSION = 1
MANIFEST, RESULT, ARRAYS = "manifest.json", "result.json", "arrays.npz"

#: Packages whose versions go into every manifest (``None`` when not installed).
PACKAGES = ("numpy", "scipy", "pandas", "astropy", "jax", "jaxlib", "numpyro", "arviz", "emcee",
            "dynesty", "pymc", "torch", "sbi", "redback", "pyphot")
#: Environment variables that decide devices and precision, recorded with the call.
ENV_VARIABLES = ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "JAX_ENABLE_X64",
                 "XLA_PYTHON_CLIENT_PREALLOCATE", "WHISPER_BAND_SYSTEM")
#: Keyword arguments that change what is displayed, never the draws: left out of the cache key.
DISPLAY_ONLY = frozenset({"progress", "progress_bar", "show_progress"})

# --- thresholds of the convergence report ------------------------------------------------------
#: R-hat across emcee walkers. Walkers are not independent chains (each move is built from the
#: others), so the 1.01 of independent chains over-flags: healthy runs of a correlated 2-D Gaussian
#: at N/tau = 100 read 1.008-1.013 over 32-60 walkers. 1.05 still catches walkers split between
#: modes, where R-hat is far above it.
ENSEMBLE_RHAT_MAX = 1.05
#: Bulk and tail ESS for an ensemble, counted over all walkers (Vehtari et al. 2021: at least 400).
ENSEMBLE_ESS_MIN = 400
#: emcee's own rule: the chain must be at least this many of the LARGEST autocorrelation times long.
N_OVER_TAU_MIN = 50
#: Independent draws a sampler without chains needs (accepted ABC draws, distinct SMC particles,
#: nested sampling's effective sample size): with 100, a 16th or 84th percentile is known to about
#: 0.1 sigma.
MIN_INDEPENDENT_DRAWS = 100
#: Prior-edge pile-up: more than EDGE_FRACTION_MAX of the draws in the outer EDGE_BAND of a prior
#: range (in the prior's own coordinate, log10 for LogUniform). A posterior as flat as the prior puts
#: 1 % there; one limited by the bound puts several times that.
EDGE_BAND = 0.01
EDGE_FRACTION_MAX = 0.05
#: The sampler's best draw may sit this far below the optimised likelihood maximum: beyond it the
#: sampler never reached the peak (the demos' chain-health rule).
LIKELIHOOD_MAX_GAP_MAX = 5.0

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_MAX_DEPTH = 4


# =========================================================================================== hashes
def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _array_record(a):
    a = np.ascontiguousarray(a)
    if a.dtype.kind in "USO":
        body = "\x1f".join(str(v) for v in a.ravel().tolist()).encode()
    else:
        body = a.tobytes()
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": _sha(body)}


def _qualname(obj):
    module = getattr(obj, "__module__", None) or type(obj).__module__
    name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    return f"{module}.{name}"


def _function_record(fn, depth):
    """A Python function by name, bytecode and closure: two closures over different data differ."""
    code = fn.__code__
    digest = hashlib.sha256(code.co_code)
    digest.update(_ADDRESS.sub("", repr(code.co_consts)).encode())
    cells = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            cells.append(_canonical(cell.cell_contents, depth))
        except ValueError:                                  # an empty cell
            cells.append(None)
    return {"function": _qualname(fn), "code": digest.hexdigest()[:16], "closure": cells,
            "defaults": _canonical(getattr(fn, "__defaults__", None), depth)}


def _canonical(obj, depth=0):
    """A JSON-able description of ``obj`` that identifies it: equal inputs, equal descriptions.

    Numbers exactly, arrays and tables by content hash, a prior by its distributions, a function by
    name, bytecode and closure, any other object by its class and public attributes (private ones,
    ``_``-prefixed, and sets are caches). Nested at most four levels deep.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if depth > _MAX_DEPTH:
        return {"object": _qualname(type(obj))}
    d = depth + 1
    if isinstance(obj, SamplerResult):
        return {"sampler_result": {"sampler": obj.sampler, "model": obj.model,
                                   "samples": samples_hash(obj.samples)}}
    if isinstance(obj, pd.DataFrame):
        return {"dataframe": {"columns": [str(c) for c in obj.columns],
                              "values": _array_record(obj.to_numpy())}}
    if hasattr(obj, "colnames") and hasattr(obj, "meta"):          # an astropy Table / LightCurve
        return {"table": data_hash(obj)}
    if isinstance(obj, np.ndarray):
        return {"array": _array_record(obj)}
    if isinstance(obj, dict):
        return {str(k): _canonical(v, d) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v, d) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonical(v, d) for v in obj), key=lambda x: _json_bytes(x))
    if hasattr(obj, "distributions") and hasattr(obj, "names"):   # a whisper Prior
        return {"prior": prior_record(obj)}
    from .models import Model
    if isinstance(obj, Model):
        return {"model": model_record(obj, _depth=d)}
    if hasattr(obj, "__array__") and type(obj).__module__.split(".")[0] in ("jax", "jaxlib"):
        try:
            return {"array": _array_record(np.asarray(obj))}
        except Exception:                                   # noqa: BLE001 - fall through to a name
            pass
    if isinstance(obj, functools.partial):
        return {"partial": _canonical(obj.func, d), "args": _canonical(obj.args, d),
                "keywords": _canonical(obj.keywords, d)}
    if inspect.ismethod(obj):
        return {"method": _function_record(obj.__func__, d), "self": _canonical(obj.__self__, d)}
    if inspect.isfunction(obj):
        return _function_record(obj, d)
    if hasattr(obj, "__wrapped__"):                         # jax.jit, functools.wraps
        return {"wrapped": _canonical(obj.__wrapped__, d)}
    if inspect.isbuiltin(obj) or inspect.isclass(obj) or inspect.ismodule(obj):
        return {"callable": _qualname(obj)}
    if hasattr(obj, "__dict__"):
        state = {k: _canonical(v, d) for k, v in vars(obj).items()
                 if not k.startswith("_") and not isinstance(v, (set, frozenset))}
        return {"object": _qualname(type(obj)), "state": state}
    return {"object": _qualname(type(obj)), "repr": _ADDRESS.sub("", repr(obj))[:200]}


def _hash(obj):
    return _sha(_json_bytes(_canonical(obj)))


def data_hash(lc):
    """SHA-256 of a light curve: every column (values, dtype, mask, unit) and its metadata.

    Two light curves with the same hash hold the same data, time origin and metadata, so a result
    that records it can be matched to its data later. Column order counts; row order counts.

    Parameters
    ----------
    lc : LightCurve or astropy.table.Table
        The data a fit was run on.

    Returns
    -------
    str
        64 hexadecimal characters.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.results import data_hash
    >>> lc = wp.LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 2.0], flux_err=[0.1, 0.1])
    >>> data_hash(lc) == data_hash(lc.copy())
    True
    >>> lc2 = lc.copy(); lc2["flux"][0] = 1.5
    >>> data_hash(lc) == data_hash(lc2)
    False
    """
    if not (hasattr(lc, "colnames") and hasattr(lc, "meta")):
        return _hash(lc)
    h = hashlib.sha256()
    for name in lc.colnames:
        col = lc[name]
        h.update(_json_bytes([name, str(getattr(col, "unit", None) or "")]))
        h.update(_json_bytes(_array_record(np.asarray(col))))
        mask = getattr(col, "mask", None)
        if mask is not None and np.any(mask):
            h.update(np.packbits(np.asarray(mask, dtype=bool)).tobytes())
    h.update(_json_bytes(_canonical(dict(lc.meta))))
    return h.hexdigest()


def samples_hash(samples):
    """SHA-256 of a posterior table: column names, values (bit for bit), dtypes and index.

    Examples
    --------
    >>> import pandas as pd
    >>> from whisper_cbpf.results import samples_hash
    >>> a = pd.DataFrame({"x": [1.0, 2.0]})
    >>> samples_hash(a) == samples_hash(a.copy()), samples_hash(a) == samples_hash(a * 2)
    (True, False)
    """
    rec = {"columns": [str(c) for c in samples.columns],
           "values": [_array_record(samples[c].to_numpy()) for c in samples.columns],
           "index": _array_record(np.asarray(samples.index))}
    return _sha(_json_bytes(rec))


# ======================================================================================= provenance
def prior_record(prior):
    """``{"repr": ..., "parameters": {name: {"type", "low", "high", ...}}}`` for a whisper prior.

    Examples
    --------
    >>> from whisper_cbpf.priors import LogUniform, Prior
    >>> from whisper_cbpf.results import prior_record
    >>> prior_record(Prior({"m": LogUniform(0.01, 1.0)}))["parameters"]
    {'m': {'type': 'LogUniform', 'low': 0.01, 'high': 1.0}}
    """
    if prior is None:
        return None
    params = {}
    for name, dist in getattr(prior, "distributions", {}).items():
        rec = {"type": type(dist).__name__}
        bounds = getattr(dist, "bounds", None)
        if bounds is not None:
            rec["low"], rec["high"] = float(bounds[0]), float(bounds[1])
        rec.update({k: _canonical(v, 1) for k, v in vars(dist).items()
                    if not k.startswith("_") and k not in ("low", "high", "name")})
        params[str(name)] = rec
    extra = {k: _canonical(v, 1) for k, v in vars(prior).items()
             if k != "distributions" and not k.startswith("_")}
    out = {"repr": _ADDRESS.sub("", repr(prior)), "parameters": params}
    if extra:
        out["extra"] = extra
    return out


def model_record(model, prior=None, *, _depth=0):
    """What identifies a fitted model: name, description, parameters, prior and its predict function.

    ``prior`` is the prior the fit used; ``None`` means the model's default. ``identity_hash`` covers
    everything here, so two models that share a name but not a redshift, a filter set or a
    constraint setting (all held by their ``predict``) hash differently.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.results import model_record
    >>> rec = model_record(wp.get_model("flare"))
    >>> rec["name"], rec["parameters"], rec["prior_source"]
    ('flare', ['amplitude', 'rise_time', 'decay_time'], "the model's default")
    """
    used = prior if prior is not None else model.default_prior
    rec = {"name": model.name, "description": model.description,
           "parameters": list(model.parameters),
           "prior": prior_record(used),
           "prior_source": "passed to fit" if prior is not None else "the model's default",
           "predict": _canonical(model.predict, _depth),
           "predict_jax": _canonical(model.predict_jax, _depth),
           "param_aliases": dict(model.param_aliases or {})}
    rec["identity_hash"] = _sha(_json_bytes({k: v for k, v in rec.items() if k != "prior_source"}))
    return rec


@functools.lru_cache(maxsize=1)
def _package_versions():
    from importlib import metadata
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


@functools.lru_cache(maxsize=1)
def _git_state():
    """Commit, branch and whether tracked files differ from it, for a whisper git checkout.

    ``None`` for an installed (non-checkout) package or without git. Read once per process, with
    ``--no-optional-locks`` so it never writes to the repository.
    """
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return None

    def git(*args):
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return None


def _x64():
    """The float64 setting a JAX computation in this session gets (without importing jax)."""
    jax = sys.modules.get("jax")
    if jax is not None:
        return bool(jax.config.jax_enable_x64)
    return os.environ.get("JAX_ENABLE_X64", "").strip().lower() in ("1", "true", "yes", "on")


def _result_settings():
    """Session settings that change a fit's numbers: bare-letter band system and JAX float64."""
    from .synphot import default_band_system
    return {"band_system": default_band_system(), "x64": _x64()}


def _environment():
    import whisper_cbpf
    return {"whisper": {"version": whisper_cbpf.__version__, "git": _git_state()},
            "packages": dict(_package_versions()),
            "python": platform.python_version(), "platform": platform.platform(),
            "environment": {"variables": {k: os.environ.get(k) for k in ENV_VARIABLES},
                            **_result_settings()}}


def _split_call(signature, arguments):
    """``(lc, model, explicit kwargs, defaults)`` from a sampler ``fit``'s bound arguments."""
    params = [p for p in signature.parameters.values() if p.name != "self"]
    names = [p.name for p in params]
    lc = arguments.get(names[0]) if names else None
    model = arguments.get(names[1]) if len(names) > 1 else None
    explicit, defaults = {}, {}
    for p in params[2:]:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            explicit.update(arguments.get(p.name, {}))
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            if arguments.get(p.name):
                explicit["*args"] = list(arguments[p.name])
        elif p.name in arguments:
            explicit[p.name] = arguments[p.name]
        elif p.default is not inspect.Parameter.empty:
            defaults[p.name] = p.default
    return lc, model, explicit, defaults


def _call_record(sampler_name, sampler_cls, lc, model, explicit, defaults):
    """The part of the provenance that identifies a fit: data, model and prior, sampler, settings."""
    from .models import get_model
    explicit = dict(explicit)
    prior = explicit.pop("prior", None)
    defaults = {k: v for k, v in defaults.items() if k != "prior"}
    m = get_model(model)
    seed = explicit.get("seed", defaults.get("seed"))
    return {
        "data": {"hash": data_hash(lc), "n_points": int(len(lc)),
                 "name": (lc.meta.get("name") if hasattr(lc, "meta") else None)},
        "model": model_record(m, prior),
        "sampler": {"name": sampler_name, "class": sampler_cls, "seed": _canonical(seed),
                    "kwargs": _canonical(explicit), "defaults": _canonical(defaults)},
    }


def record_call(sampler, bound, wall_s):
    """Provenance of one ``fit`` call: environment, data, model and prior, sampler settings, time.

    Called by :class:`~whisper_cbpf.samplers.base.BaseSampler` after every successful ``fit``
    with the sampler, the call's ``inspect.BoundArguments`` and its wall time; stored in
    ``result.provenance`` and written to the manifest by ``result.save``.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> lc = wp.LightCurve(time=np.linspace(1, 30, 20), band=["r"] * 20, flux=np.ones(20),
    ...                    flux_err=np.full(20, 0.1))
    >>> res = wp.fit(lc, "flare", sampler="abc", n_simulations=500, quantile=0.1, seed=0)
    >>> res.provenance["sampler"]["kwargs"], res.provenance["model"]["name"]
    ({'n_simulations': 500, 'quantile': 0.1, 'seed': 0}, 'flare')
    """
    lc, model, explicit, defaults = _split_call(bound.signature, bound.arguments)
    out = {"recorded_at": "fit", **_environment()}
    out.update(_call_record(getattr(sampler, "name", None), _qualname(type(sampler)),
                            lc, model, explicit, defaults))
    jax = sys.modules.get("jax")
    if jax is not None and type(sampler).__module__.startswith("whisper_cbpf.samplers.jax"):
        out["jax_devices"] = {"backend": jax.default_backend(),
                              "devices": [str(d) for d in jax.devices()]}
    out["wall_s"] = float(wall_s)
    return out


# ============================================================================= JSON with arrays
def _encode(obj, path, lossy):
    """JSON-able form of ``info``-like data that :func:`_decode` turns back into the same objects.

    Arrays keep their dtype and shape, tuples stay tuples, dicts with non-string keys keep their
    keys. Anything else is written as ``{"__repr__": ..., "type": ...}`` and listed in ``lossy``.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "biuf":
        return {"__ndarray__": obj.tolist(), "dtype": obj.dtype.str, "shape": list(obj.shape)}
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            if "__repr__" in obj and "type" in obj and len(obj) == 2:
                return dict(obj)                            # already a placeholder: keep it
            return {k: _encode(v, f"{path}.{k}", lossy) for k, v in obj.items()}
        return {"__items__": [[_encode(k, path, lossy), _encode(v, f"{path}[{k!r}]", lossy)]
                              for k, v in obj.items()]}
    if isinstance(obj, list):
        return [_encode(v, f"{path}[{i}]", lossy) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, f"{path}[{i}]", lossy) for i, v in enumerate(obj)]}
    lossy.append(path)
    return {"__repr__": _ADDRESS.sub("", repr(obj))[:500], "type": _qualname(type(obj))}


def _decode(obj):
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.array(obj["__ndarray__"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        if "__tuple__" in obj:
            return tuple(_decode(v) for v in obj["__tuple__"])
        if "__items__" in obj:
            return {_decode(k): _decode(v) for k, v in obj["__items__"]}
        return {k: _decode(v) for k, v in obj.items()}
    return obj


# ==================================================================================== save / load
def chains_of(result):
    """The draws by chain, ``(chains, draws, parameters)``, or ``None`` when the sampler has none.

    ``result.samples_by_chain`` where the sampler set it; for the CPU ``mcmc`` sampler, rebuilt
    from its draws, which emcee stores step by step (all walkers of step 0, then step 1, ...).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from whisper_cbpf.results import chains_of
    >>> from whisper_cbpf.samplers.base import SamplerResult
    >>> res = SamplerResult("mcmc", "toy", ["a"], pd.DataFrame({"a": np.arange(6.0)}), {}, {},
    ...                     10, 1, 0.0, {"nwalkers": 2})
    >>> chains_of(res)[:, :, 0]                      # walker 0 made draws 0, 2 and 4
    array([[0., 2., 4.],
           [1., 3., 5.]])
    """
    chains = getattr(result, "samples_by_chain", None)
    if chains is not None:
        return np.asarray(chains)
    info = result.info if isinstance(result.info, dict) else {}
    nw = info.get("nwalkers")
    names = [p for p in result.parameters if p in result.samples.columns]
    if nw and result.n_samples and result.n_samples % int(nw) == 0 and names:
        flat = result.samples[names].to_numpy()
        return flat.reshape(result.n_samples // int(nw), int(nw), len(names)).swapaxes(0, 1)
    return None


def _timing(result, prov):
    info = result.info if isinstance(result.info, dict) else {}

    def num(key):
        v = info.get(key)
        return float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else None
    t = {"runtime_s": float(result.runtime_s), "run_s": float(result.runtime_s),
         "compile_s": num("compile_time_s"), "init_s": num("init_time_s"),
         "warmup_s": num("warmup_time_s"), "sampling_s": num("sampling_time_s"),
         "postprocess_s": num("postprocess_s"), "wall_s": prov.get("wall_s")}
    if t["compile_s"] is not None:
        t["note"] = "compile_s is outside run_s (= runtime_s); wall_s is the whole fit() call"
    elif t["warmup_s"] is not None:
        t["note"] = ("this sampler compiles during warmup, so compile time is inside warmup_s and "
                     "run_s; wall_s is the whole fit() call")
    else:
        t["note"] = ("this sampler does not time compilation separately: run_s (= runtime_s) "
                     "includes any; wall_s is the whole fit() call")
    return t


def _device(result, prov):
    info = result.info if isinstance(result.info, dict) else {}
    keys = ("device", "devices", "sim_device", "n_devices", "n_devices_used", "chain_method",
            "x64", "n_jobs", "num_workers")
    out = {k: _canonical(info[k]) for k in keys if k in info}
    if prov.get("jax_devices") is not None:
        out["jax_devices"] = prov["jax_devices"]
    return out


def _payload(result):
    """``(manifest, result_json_bytes, arrays)`` -- what both layouts store."""
    import whisper_cbpf
    prov = dict(getattr(result, "provenance", None) or {})
    if not prov:
        prov = {"recorded_at": "save",
                "note": ("this result did not come from a registered sampler's fit(), so the data, "
                         "the model and the sampler settings were not recorded"),
                **_environment()}
    info = result.info if isinstance(result.info, dict) else {}
    samples = result.samples
    arrays = {f"samples/{i}": np.asarray(samples[c].to_numpy())
              for i, c in enumerate(samples.columns)}
    for key, arr in list(arrays.items()):
        if arr.dtype.kind == "O":
            arrays[key] = arr.astype(str)
    idx = samples.index
    default_index = isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1
    if not default_index:
        arrays["samples_index"] = np.asarray(idx)
        if arrays["samples_index"].dtype.kind == "O":
            arrays["samples_index"] = arrays["samples_index"].astype(str)
    chains = chains_of(result)
    if chains is not None:
        arrays["samples_by_chain"] = chains
    lossy = []
    body = {
        "sampler": result.sampler, "model": result.model, "parameters": list(result.parameters),
        "n_data": int(result.n_data), "n_params": int(result.n_params),
        "runtime_s": float(result.runtime_s), "min_distance": float(result.min_distance),
        "max_log_likelihood": float(result.max_log_likelihood),
        "aic": float(result.aic), "bic": float(result.bic),
        "best_params": _encode(dict(result.best_params), "best_params", lossy),
        "summary": _encode(result.summary, "summary", lossy),
        "info": _encode(result.info, "info", lossy),
        "samples_columns": [_encode(c, "samples.columns", lossy) for c in samples.columns],
        "samples_index_saved": not default_index,
    }
    body_bytes = json.dumps(body, indent=1).encode()
    fields = {f.name for f in SamplerResult.__dataclass_fields__.values()}
    # private attributes (the fitted Model object, see samplers.base.fitted_model) are session
    # state, not results: a loaded result finds its model by name
    not_saved = sorted(k for k in vars(result) if k not in fields and k != "samples_by_chain"
                       and k != "loaded_from" and not k.startswith("_"))
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "saved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "saved_with_whisper": whisper_cbpf.__version__,
        "result": {"sampler": result.sampler, "model": result.model,
                   "n_samples": result.n_samples, "n_data": int(result.n_data),
                   "n_params": int(result.n_params),
                   "seed": _canonical(info.get("seed", (prov.get("sampler") or {}).get("seed"))),
                   "chains_shape": None if chains is None else list(chains.shape)},
        "provenance": prov,
        "timing": _timing(result, prov),
        "device": _device(result, prov),
        "hashes": {"samples": samples_hash(samples),
                   "samples_by_chain": None if chains is None else _array_record(chains)["sha256"],
                   "summary": _sha(_json_bytes(_encode(result.summary, "summary", []))),
                   "result_json": _sha(body_bytes),
                   "data": (prov.get("data") or {}).get("hash")},
        "not_saved": not_saved,
        "lossy_info_keys": lossy,
    }
    return manifest, body_bytes, arrays


def _write_atomic(path, write):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _json_array(data):
    return np.frombuffer(data, dtype=np.uint8)


def save_result(result, path, *, overwrite=False):
    """Save ``result`` to ``path``; see :meth:`SamplerResult.save
    <whisper_cbpf.samplers.base.SamplerResult.save>` (this is its implementation).

    Examples
    --------
    >>> import pandas as pd
    >>> from whisper_cbpf.results import load_result, save_result
    >>> from whisper_cbpf.samplers.base import SamplerResult
    >>> res = SamplerResult("custom", "toy", ["a"], pd.DataFrame({"a": [1.0, 2.0]}),
    ...                     {"a": {"median": 1.5}}, {"a": 2.0}, 10, 1, 0.1)
    >>> load_result(save_result(res, "toy.npz", overwrite=True)).summary
    {'a': {'median': 1.5}}
    """
    path = Path(path)
    manifest, body, arrays = _payload(result)
    manifest_bytes = json.dumps(manifest, indent=1).encode()
    if path.suffix == ".npz":
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists. Pass overwrite=True to replace it, or save to "
                                  f"another path.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**arrays, "__manifest__": _json_array(manifest_bytes),
                   "__result__": _json_array(body)}
        _write_atomic(path, lambda fh: np.savez(fh, **payload))
        return path
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"{path} is a file. Save to a directory path, or to a path ending in "
                              f".npz for a single file.")
    if (path / MANIFEST).exists() and not overwrite:
        raise FileExistsError(f"{path} already holds a saved result. Pass overwrite=True to replace "
                              f"it, or save to another path.")
    path.mkdir(parents=True, exist_ok=True)
    # The manifest goes last: it carries the hashes of the other two, so a save interrupted
    # half-way is refused by load_result rather than read as a mix of two results.
    _write_atomic(path / ARRAYS, lambda fh: np.savez(fh, **arrays))
    _write_atomic(path / RESULT, lambda fh: fh.write(body))
    _write_atomic(path / MANIFEST, lambda fh: fh.write(manifest_bytes))
    return path


def _read(path):
    if not path.exists():
        raise FileNotFoundError(f"no saved result at {path}.")
    if path.is_dir():
        if not (path / MANIFEST).exists():
            raise ValueError(f"{path} is not a saved whisper result: it has no {MANIFEST}. Save one "
                             f"with result.save(path).")
        manifest = json.loads((path / MANIFEST).read_text())
        body = (path / RESULT).read_bytes()
        with np.load(path / ARRAYS, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return manifest, body, arrays
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as exc:                                    # noqa: BLE001 - reworded below
        raise ValueError(f"{path} is not a saved whisper result ({type(exc).__name__}: {exc}).") \
            from exc
    if "__manifest__" not in arrays:
        raise ValueError(f"{path} is an .npz file but not a saved whisper result (no manifest). "
                         f"Save one with result.save('name.npz').")
    manifest = json.loads(arrays.pop("__manifest__").tobytes())
    body = arrays.pop("__result__").tobytes()
    return manifest, body, arrays


def load_result(path):
    """Load a result saved with ``result.save(path)``: draws, best fit, metrics, info and provenance.

    The draws, the draws by chain, the summary and the result file are checked against the hashes
    written with them, so a modified or truncated save is refused rather than read.

    Parameters
    ----------
    path : str or Path
        A directory written by ``result.save("dir")`` or a file written by
        ``result.save("name.npz")``.

    Returns
    -------
    SamplerResult
        Equal to the saved one in every field; ``result.provenance`` is the recorded provenance
        (the manifest's ``"provenance"`` block), ``result.samples_by_chain`` is restored where the
        sampler had chains, and ``result.loaded_from`` is ``path``. Live sampler objects
        (``emcee_sampler``, ``numpyro_mcmc``, ...) are not saved; the manifest lists them under
        ``"not_saved"``.

    Raises
    ------
    FileNotFoundError
        Nothing at ``path``.
    ValueError
        ``path`` is not a saved whisper result, was written by a newer whisper, or no longer
        matches its hashes.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> res = wp.fit(lc, "flare", sampler="mcmc", nsteps=600, burnin=200, seed=0)
    >>> path = res.save("flare_mcmc", overwrite=True)      # "flare_mcmc.npz" for one file
    >>> again = wp.load_result(path)
    >>> again.summary == res.summary, again.samples.equals(res.samples)
    (True, True)
    >>> again.provenance["model"]["name"], again.provenance["sampler"]["kwargs"]["nsteps"]
    ('flare', 600)
    """
    path = Path(path)
    manifest, body_bytes, arrays = _read(path)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a saved whisper result (format {manifest.get('format')!r}).")
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise ValueError(f"{path} was saved by whisper {manifest.get('saved_with_whisper')} in "
                         f"format {manifest['format_version']}; this whisper reads up to format "
                         f"{FORMAT_VERSION}. Upgrade whisper_cbpf to load it.")
    hashes = manifest.get("hashes", {})

    def refuse(what):
        raise ValueError(f"{path}: the hash of {what} differs from the one written when it was "
                         f"saved, so the save was modified or truncated. Re-save the result, or "
                         f"re-run the fit.")
    if _sha(body_bytes) != hashes.get("result_json"):
        refuse("the result file")
    body = json.loads(body_bytes)
    columns = [_decode(c) for c in body["samples_columns"]]
    samples = pd.DataFrame({c: arrays[f"samples/{i}"] for i, c in enumerate(columns)},
                           columns=columns)
    if body.get("samples_index_saved"):
        samples.index = pd.Index(arrays["samples_index"])
    if samples_hash(samples) != hashes.get("samples"):
        refuse("the posterior draws")
    chains = arrays.get("samples_by_chain")
    if (None if chains is None else _array_record(chains)["sha256"]) != hashes.get("samples_by_chain"):
        refuse("the draws by chain")
    summary = _decode(body["summary"])
    if _sha(_json_bytes(body["summary"])) != hashes.get("summary"):
        refuse("the summary")
    result = SamplerResult(
        sampler=body["sampler"], model=body["model"], parameters=list(body["parameters"]),
        samples=samples, summary=summary, best_params=_decode(body["best_params"]),
        n_data=int(body["n_data"]), n_params=int(body["n_params"]),
        runtime_s=float(body["runtime_s"]), info=_decode(body["info"]),
        min_distance=float(body["min_distance"]),
        max_log_likelihood=float(body["max_log_likelihood"]),
        aic=float(body["aic"]), bic=float(body["bic"]),
        provenance=manifest.get("provenance") or {})
    if chains is not None:
        result.samples_by_chain = chains
    result.loaded_from = str(path)
    return result


# ==================================================================================== fit_cached
def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._") or "x"


def cache_config(lc, model, sampler, **fit_kwargs):
    """The configuration :func:`fit_cached` hashes: what decides a fit's numbers.

    The data (hash), the model (name, description, parameters, prior, predict function), the
    sampler and its settings (explicit and default), the whisper version and the two session
    settings that change results (bare-letter band system, JAX float64). Display-only keywords
    (``progress``) are left out.

    Returns
    -------
    (dict, str)
        The configuration and its SHA-256.

    Examples
    --------
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.results import cache_config
    >>> lc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3, flux=[1.0, 2.0, 1.5],
    ...                    flux_err=[0.1] * 3)
    >>> _, a = cache_config(lc, "flare", "mcmc", nsteps=600, seed=0)
    >>> _, b = cache_config(lc, "flare", "mcmc", nsteps=600, seed=1)
    >>> a == b
    False
    """
    import whisper_cbpf
    from .samplers import get_sampler, list_samplers
    if not isinstance(sampler, str):
        raise TypeError(f"name the sampler by its registry name (one of {list_samplers()}), not "
                        f"{type(sampler).__name__}.")
    instance = get_sampler(sampler)
    kwargs = {k: v for k, v in fit_kwargs.items() if k not in DISPLAY_ONLY}
    signature = inspect.signature(type(instance).fit)
    try:
        bound = signature.bind_partial(instance, lc, model, **kwargs)
    except TypeError as exc:
        raise TypeError(f"sampler {sampler!r}: {exc}. Its settings are listed in "
                        f"help(whisper_cbpf.samplers.get_sampler({sampler!r}).fit).") from None
    _, _, explicit, defaults = _split_call(signature, bound.arguments)
    defaults = {k: v for k, v in defaults.items() if k not in DISPLAY_ONLY}
    call = _call_record(sampler, _qualname(type(instance)), lc, model, explicit, defaults)
    config = {"data_hash": call["data"]["hash"], "model": call["model"],
              "sampler": {"name": sampler, "kwargs": call["sampler"]["kwargs"],
                          "defaults": call["sampler"]["defaults"]},
              "whisper_version": whisper_cbpf.__version__, "session": _result_settings()}
    return config, _sha(_json_bytes(config))


def fit_cached(lc, model, sampler, cache_dir, **fit_kwargs):
    """Fit, unless an identical fit was already saved in ``cache_dir``: then load it.

    The configuration -- the data (hash), the model and its prior, the sampler and its settings, the
    whisper version and the session's band system and float64 setting (:func:`cache_config`) -- is
    hashed. A saved result with that hash is loaded and returned without running anything;
    otherwise the fit runs and is saved under ``cache_dir`` before it is returned. A save is written
    to a temporary directory and renamed into place, so a run killed half-way leaves no result that
    could be mistaken for a finished one: re-running an interrupted batch finishes only what is
    missing.

    Parameters
    ----------
    lc : LightCurve
        The data.
    model : str or Model
        A registered model name or a :class:`~whisper_cbpf.models.Model`.
    sampler : str
        A registered sampler name (:func:`~whisper_cbpf.list_samplers`).
    cache_dir : str or Path
        Where results are kept, one directory per configuration, named
        ``<light curve>__<model>__<sampler>__<hash prefix>``.
    **fit_kwargs
        Passed to :func:`~whisper_cbpf.fit` (``prior=``, ``seed=``, ``nsteps=`` ...). Every one is
        part of the configuration except the display-only ``progress``.

    Returns
    -------
    SamplerResult
        The fresh result, or the saved one (then ``result.loaded_from`` names its directory).
        ``result.provenance["cache"]`` holds the configuration and its hash.

    Notes
    -----
    A code change that keeps the whisper version does not change the hash unless it changes the
    model's own code or settings, a sampler default, or the prior: delete the directory to refit.
    Callables passed as settings are identified by name, bytecode and closure.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1), name="toy")
    >>> first = wp.fit_cached(lc, "flare", "mcmc", "fits", nsteps=600, burnin=200, seed=0)
    >>> again = wp.fit_cached(lc, "flare", "mcmc", "fits", nsteps=600, burnin=200, seed=0)
    >>> again.loaded_from is not None, again.summary == first.summary
    (True, True)
    """
    from .models import get_model
    from .samplers import fit
    config, key = cache_config(lc, model, sampler, **fit_kwargs)
    m = get_model(model)
    name = "__".join(_slug(p) for p in (lc.meta.get("name") if hasattr(lc, "meta") else None,
                                         m.name, sampler, key[:16]) if p)
    path = Path(cache_dir) / name
    if (path / MANIFEST).exists():
        result = load_result(path)
        recorded = (result.provenance.get("cache") or {}).get("config_hash")
        if recorded != key:
            raise ValueError(f"{path} holds a result of another configuration (hash {recorded} "
                             f"against {key}). Move or delete that directory to refit here.")
        return result
    result = fit(lc, model, sampler=sampler, **fit_kwargs)
    result.provenance = {**(getattr(result, "provenance", None) or {}),
                         "cache": {"config_hash": key, "config": config}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        save_result(result, tmp)
        try:
            os.rename(tmp, path)
        except OSError:
            if not (path / MANIFEST).exists():               # not a finished twin: a real error
                raise
            # Another process saved the same configuration first; its result stands.
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return result


# ================================================================================== diagnostics
class DiagnosticRow(NamedTuple):
    """One check of a :class:`DiagnosticsReport`: ``passed`` is ``None`` when it could not be run.

    Examples
    --------
    >>> from whisper_cbpf.results import DiagnosticRow
    >>> check, value, threshold, passed, reason = DiagnosticRow("divergences", 0, "0", True, "none")
    >>> passed
    True
    """

    check: str
    value: object
    threshold: str
    passed: object
    reason: str


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else f"{float(v):.4g}"
    return str(v)


@dataclass(frozen=True)
class DiagnosticsReport:
    """Whether a fit's posterior can be used: every applicable check, with a plain reason each.

    Made by :meth:`SamplerResult.diagnostics <whisper_cbpf.samplers.base.SamplerResult.diagnostics>`.
    ``rows`` holds one :class:`DiagnosticRow` ``(check, value, threshold, passed, reason)`` per
    check; ``passed`` is True when no check failed (checks that could not be run, ``passed=None``,
    are listed but do not fail the report); ``reasons`` gives the reason of every failed check.

    Attributes
    ----------
    sampler, model : str
        The fit's sampler and model.
    kind : str
        The family of checks run (NUTS, emcee, ABC, ABC-SMC, nested, SNPE).
    rows : list of DiagnosticRow
    passed : bool
        No check failed.
    reasons : list of str
        ``"check: reason"`` for every failed check.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.models.flare import flare_flux
    >>> t = np.linspace(0.5, 30.0, 40)
    >>> flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 40, flux=flux, flux_err=np.full(40, 0.1))
    >>> report = wp.fit(lc, "flare", sampler="mcmc", nsteps=600, burnin=200, seed=0).diagnostics()
    >>> report.passed                       # 600 steps is too short for emcee's N/tau >= 50
    False
    >>> "chain length / autocorrelation time" in [r.check for r in report.rows if r.passed is False]
    True
    >>> report.reasons[0]                                   # doctest: +ELLIPSIS
    'chain length / autocorrelation time: the chain is ... Raise nsteps to at least ...'
    """

    sampler: str
    model: str
    kind: str
    rows: list

    @property
    def passed(self):
        return all(r.passed is not False for r in self.rows)

    @property
    def reasons(self):
        return [f"{r.check}: {r.reason}" for r in self.rows if r.passed is False]

    def to_dict(self):
        """JSON-able form: sampler, model, kind, passed, rows (as dicts) and reasons."""
        return {"sampler": self.sampler, "model": self.model, "kind": self.kind,
                "passed": self.passed, "reasons": self.reasons,
                "rows": [{"check": r.check, "value": _canonical(r.value),
                          "threshold": r.threshold, "passed": r.passed, "reason": r.reason}
                         for r in self.rows]}

    def __repr__(self):
        failed = sum(r.passed is False for r in self.rows)
        unrun = sum(r.passed is None for r in self.rows)
        verdict = "PASSED" if self.passed else f"FAILED ({failed} of {len(self.rows)} checks)"
        head = f"Diagnostics of the {self.sampler} fit of {self.model!r}: {verdict}"
        if unrun:
            head += f"; {unrun} not checked"
        table = [("check", "value", "threshold", "result")] + [
            (r.check, _fmt(r.value), r.threshold,
             {True: "pass", False: "FAIL", None: "not checked"}[r.passed]) for r in self.rows]
        widths = [max(len(row[i]) for row in table) for i in range(3)]
        lines = [head] + ["  " + "  ".join(c.ljust(w) for c, w in zip(row[:3], widths))
                          + "  " + row[3] for row in table]
        notes = [(r, "FAIL") for r in self.rows if r.passed is False] + \
                [(r, "not checked") for r in self.rows if r.passed is None]
        if notes:
            lines.append("Why:")
            lines += [f"  - {r.check} [{tag}]: {r.reason}" for r, tag in notes]
        lines.append("Every check with its reason: report.rows; as JSON: report.to_dict().")
        return "\n".join(lines)


_NUTS = ("nuts_gpu", "pymc_jax_gpu_vectorized", "pymc_jax_gpu_parallelized")
_ENSEMBLE = ("mcmc", "emcee_jax", "emcee_numpy")


def _kind(result):
    info = result.info if isinstance(result.info, dict) else {}
    s = str(result.sampler)
    if s in _NUTS or "n_divergences" in info:
        return "nuts"
    if s in _ENSEMBLE or "nwalkers" in info:
        return "ensemble"
    if s in ("abc", "abc_gpu") or "n_accepted" in info:
        return "abc"
    if s in ("abc_smc", "abc_smc_gpu") or "n_particles" in info:
        return "abc_smc"
    if s == "nested" or "log_evidence" in info:
        return "nested"
    if s in ("snpe", "npe", "snpe_gpu"):
        return "snpe"
    return "other"


def _common_rows(result):
    n, k, d = result.n_samples, int(result.n_params), int(result.n_data)
    rows = [DiagnosticRow(
        "posterior draws", n, "> 0", n > 0,
        f"{n} draws to summarise" if n > 0 else
        "the sampler returned no posterior draws, so every summary of this fit is undefined (for "
        "ABC: no draw was accepted; loosen the threshold or run more simulations)")]
    rows.append(DiagnosticRow(
        "data points", d, f"> {k} (free parameters)", d > k,
        f"{d} points for {k} free parameters" if d > k else
        f"not enough data: {d} points for {k} free parameters, so the posterior and BIC mostly "
        f"restate the prior. Wait for more detections, or fix (pin) parameters"))
    return rows


def _rank_stats(result, chains, from_info):
    """R-hat and ESS per parameter: from ``info`` where the sampler computed them, else computed."""
    info = result.info if isinstance(result.info, dict) else {}
    if from_info and all(isinstance(info.get(k), dict) and info.get(k) for k in
                         ("rhat", "ess_bulk", "ess_tail")):
        return {"rhat": info["rhat"], "ess_bulk": info["ess_bulk"], "ess_tail": info["ess_tail"],
                "rhat_log_likelihood": info.get("rhat_log_likelihood"),
                "method": info.get("rhat_method"), "error": info.get("rhat_error")}
    names = [p for p in result.parameters if p in result.samples.columns]
    if chains is None or chains.ndim != 3 or chains.shape[2] != len(names):
        return None
    d = _dg.rank_diagnostics(chains, names)
    return {"rhat": d["rhat"], "ess_bulk": d["ess_bulk"], "ess_tail": d["ess_tail"],
            "rhat_log_likelihood": None, "method": d["rhat_method"], "error": d["rhat_error"]}


def _extreme(values, pick):
    finite = {k: float(v) for k, v in (values or {}).items()
              if v is not None and np.isfinite(float(v))}
    if not finite:
        return None, float("nan")
    name = pick(finite, key=finite.get)
    return name, finite[name]


def _rank_rows(stats, n_chains, rhat_max, ess_min, unknown, walkers=False):
    unit = "walkers" if walkers else "chains"
    label = "split R-hat across walkers (largest)" if walkers else "split R-hat (largest)"
    if stats is None:
        why = "the draws by chain were not kept with this result"
        return [DiagnosticRow(label, None, f"< {rhat_max}", None, why),
                DiagnosticRow("bulk ESS (smallest)", None, f">= {ess_min}", None, why),
                DiagnosticRow("tail ESS (smallest)", None, f">= {ess_min}", None, why)]
    rows = []
    worst, rhat = _extreme(stats["rhat"], max)
    if n_chains < 2:
        rows.append(DiagnosticRow(label, None, f"< {rhat_max}", False,
                                  f"one chain: R-hat compares {unit}, so it needs at least 2 (4 "
                                  f"recommended); run more {unit}"))
    elif worst is None:
        rows.append(DiagnosticRow(label, None, f"< {rhat_max}", unknown,
                                  f"could not be computed ({stats['method']}; {stats['error']}). "
                                  f"Install the [analysis] extra (arviz) to compute it"))
    elif rhat < rhat_max:
        rows.append(DiagnosticRow(label, rhat, f"< {rhat_max}", True,
                                  f"the {unit} agree (largest on {worst!r})"))
    else:
        rows.append(DiagnosticRow(label, rhat, f"< {rhat_max}", False,
                                  f"R-hat {rhat:.3f} on {worst!r}: the {unit} disagree about where "
                                  f"the posterior is. Run longer, and look for {unit} left in "
                                  f"another mode"))
    for kind, what in (("bulk", "medians and widths"), ("tail", "16th/84th and 5th/95th "
                                                        "percentiles")):
        worst, ess = _extreme(stats[f"ess_{kind}"], min)
        check = f"{kind} ESS (smallest)"
        if worst is None:
            rows.append(DiagnosticRow(check, None, f">= {ess_min}", unknown,
                                      f"could not be computed ({stats['method']}; "
                                      f"{stats['error']})"))
        elif ess >= ess_min:
            rows.append(DiagnosticRow(check, ess, f">= {ess_min}", True,
                                      f"enough effective draws for the {what}"))
        else:
            rows.append(DiagnosticRow(check, ess, f">= {ess_min}", False,
                                      f"{ess:.0f} effective draws on {worst!r}: too few for "
                                      f"reliable {what}. Draw more samples"))
    return rows


def _nuts_rows(result, chains):
    info = result.info
    n_chains = int(info.get("num_chains") or (chains.shape[0] if chains is not None else 0))
    rows = []
    ndiv = info.get("n_divergences")
    if ndiv is None:
        rows.append(DiagnosticRow("divergences", None, "0", None,
                                  "not recorded by this sampler"))
    elif int(ndiv) < 0:
        rows.append(DiagnosticRow("divergences", None, "0", False,
                                  "divergences were not recorded, so they cannot be ruled out"))
    else:
        rows.append(DiagnosticRow(
            "divergences", int(ndiv), "0", int(ndiv) == 0,
            "no divergent transition" if int(ndiv) == 0 else
            f"{int(ndiv)} divergent transition(s): the sampler could not follow the posterior's "
            f"curvature somewhere, so the draws may be biased. Raise target_accept_prob, or "
            f"reparameterise"))
    need = _dg.ESS_PER_CHAIN_MIN * max(n_chains, 1)
    stats = _rank_stats(result, chains, from_info=True)
    rows += _rank_rows(stats, n_chains, _dg.RHAT_MAX, need, unknown=False)
    rll = None if stats is None else stats.get("rhat_log_likelihood")
    if rll is None:
        rows.append(DiagnosticRow("log-likelihood R-hat", None, f"< {_dg.RHAT_MAX}", None,
                                  "not recorded by this sampler"))
    elif not np.isfinite(float(rll)):
        rows.append(DiagnosticRow("log-likelihood R-hat", None, f"< {_dg.RHAT_MAX}",
                                  False if n_chains >= 2 else None,
                                  "could not be computed" if n_chains >= 2 else "one chain"))
    else:
        rows.append(DiagnosticRow(
            "log-likelihood R-hat", float(rll), f"< {_dg.RHAT_MAX}", float(rll) < _dg.RHAT_MAX,
            "the chains sit at the same likelihood level" if float(rll) < _dg.RHAT_MAX else
            f"log-likelihood R-hat {float(rll):.3f}: the chains sit at different likelihood levels"))
    stranded = info.get("stranded_chains")
    if stranded is None:
        rows.append(DiagnosticRow("stranded chains", None, "none", None, "not recorded"))
    else:
        rows.append(DiagnosticRow(
            "stranded chains", len(stranded), "none", not stranded,
            "every chain reached the best region" if not stranded else
            f"chain(s) {', '.join(map(str, stranded))} sit more than "
            f"{_dg.STRANDED_NATS:g} nats below the best chain or the prior scan's optimum: a local "
            f"optimum, pooled into the posterior. Start from init_strategy='prior_scan' or a "
            f"previous result"))
    frozen = info.get("frozen_chains")
    if frozen is None:
        rows.append(DiagnosticRow("frozen chains", None, "none", None, "not recorded"))
    else:
        rows.append(DiagnosticRow(
            "frozen chains", len(frozen), "none", not frozen,
            "every chain moves" if not frozen else
            f"chain(s) {', '.join(map(str, frozen))} froze (adapted step below "
            f"{_dg.FROZEN_STEP_RATIO:g} of the median chain's, and barely moving): in float32 the "
            f"signature of an MJD-scale clock. Enable x64 or shift the clock"))
    rows.append(_scan_gap_row(result))
    return rows


def _scan_gap_row(result):
    scan = result.info.get("prior_scan") if isinstance(result.info, dict) else None
    ref = scan.get("best_log_likelihood") if isinstance(scan, dict) else None
    thr = f"<= {_dg.STRANDED_NATS:g} nats"
    if ref is None or not np.isfinite(float(ref)) or not np.isfinite(result.max_log_likelihood):
        return DiagnosticRow("gap to the prior scan's optimum", None, thr, None,
                             "no independent optimum was recorded (the run did not start from a "
                             "prior scan)")
    gap = float(ref) - float(result.max_log_likelihood)
    if gap <= _dg.STRANDED_NATS:
        return DiagnosticRow("gap to the prior scan's optimum", gap, thr, True,
                             "the best draw reached the prior scan's optimum")
    return DiagnosticRow("gap to the prior scan's optimum", gap, thr, False,
                         f"the prior scan found a log-likelihood {gap:.1f} nats above every draw: "
                         f"all chains missed the best region, which R-hat cannot see. Start from "
                         f"init_strategy='prior_scan', or run longer")


def _ensemble_rows(result, chains):
    info = result.info
    rows = []
    stuck = info.get("stuck_walkers")
    nw = info.get("nwalkers")
    if stuck is None:
        rows.append(DiagnosticRow("stuck walkers", None, "none", None, "not recorded"))
    else:
        advice = ("Rerun with the walkers started around the likelihood peak "
                  "(init=result.likelihood_max_opt(lc).params), or longer"
                  if info.get("init") == "prior_scan"
                  else "Start from init='prior_scan' (the default) or around the likelihood peak "
                       "(init=result.likelihood_max_opt(lc).params)")
        rows.append(DiagnosticRow(
            "stuck walkers", len(stuck), "none", not stuck,
            "every walker reached the best region" if not stuck else
            f"{len(stuck)} of {nw} walkers sit more than {_dg.STRANDED_NATS:g} nats below the best "
            f"walker (median log-posterior in the coordinates the walkers move in): their draws "
            f"are pooled into the posterior and widen it. {advice}"))
    tau, nsteps = info.get("max_autocorr_time"), info.get("nsteps")
    thr = f">= {N_OVER_TAU_MIN} (largest tau)"
    if tau is None or nsteps is None:
        rows.append(DiagnosticRow("chain length / autocorrelation time", None, thr, None,
                                  "not recorded"))
    elif not np.isfinite(float(tau)) or float(tau) <= 0:
        rows.append(DiagnosticRow("chain length / autocorrelation time", None, thr, False,
                                  "the autocorrelation time could not be estimated: the chain is "
                                  "too short. Raise nsteps"))
    else:
        ratio = float(nsteps) / float(tau)
        rows.append(DiagnosticRow(
            "chain length / autocorrelation time", ratio, thr, ratio >= N_OVER_TAU_MIN,
            f"{ratio:.0f} autocorrelation times long (largest tau {float(tau):.0f} steps)"
            if ratio >= N_OVER_TAU_MIN else
            f"the chain is {ratio:.1f} autocorrelation times long (largest tau {float(tau):.0f} "
            f"steps), below {N_OVER_TAU_MIN}: too short to trust. Raise nsteps to at least "
            f"{int(np.ceil(N_OVER_TAU_MIN * float(tau)))}"))
    stats = _rank_stats(result, chains, from_info=False)
    n_chains = 0 if chains is None else int(chains.shape[0])
    rows += _rank_rows(stats, n_chains, ENSEMBLE_RHAT_MAX, ENSEMBLE_ESS_MIN, unknown=None,
                       walkers=True)
    return rows


def _independent_row(check, value, reason_ok, reason_low):
    thr = f">= {MIN_INDEPENDENT_DRAWS}"
    if value is None:
        return DiagnosticRow(check, None, thr, None, "not recorded")
    ok = value >= MIN_INDEPENDENT_DRAWS
    return DiagnosticRow(check, value, thr, ok, reason_ok if ok else reason_low)


def _abc_rows(result):
    n = result.info.get("n_accepted")
    low = ("ABC accepted no draw: there is no posterior, and best_params is the closest REJECTED "
           "draw. Loosen the threshold (larger quantile) or run more simulations"
           if n == 0 else
           f"only {n} accepted draws: percentiles of fewer than {MIN_INDEPENDENT_DRAWS} draws are "
           f"uncertain by more than ~0.1 sigma. Run more simulations or loosen the threshold")
    return [_independent_row("accepted draws", None if n is None else int(n),
                             "enough accepted draws for the percentiles", low)]


def _smc_rows(result):
    names = [p for p in result.parameters if p in result.samples.columns]
    n = int(len(result.samples[names].drop_duplicates())) if names else 0
    return [_independent_row(
        "distinct particles", n, "enough distinct particles for the percentiles",
        f"the final population holds {n} distinct particles: the resampling collapsed onto a few "
        f"points. Raise n_particles or stop at a larger epsilon")]


def _nested_rows(result):
    info = result.info
    rows = []
    conv = info.get("converged")
    rows.append(DiagnosticRow(
        "stopped on dlogz", conv, "True", None if conv is None else bool(conv),
        "not recorded" if conv is None else
        "the run reached its evidence tolerance" if conv else
        "the run stopped on maxiter/maxcall, not on dlogz: log_evidence is a lower bound and its "
        "error is not trustworthy. Raise maxcall/maxiter or lower nlive"))
    neff = info.get("n_effective")
    rows.append(_independent_row(
        "effective sample size", None if neff is None else float(neff),
        "enough effective draws for the percentiles",
        f"{0 if neff is None else neff:.0f} effective draws: too few for reliable percentiles. "
        f"Raise nlive"))
    return rows


def _snpe_rows(result):
    info = result.info
    conv = info.get("converged")
    if conv is None:
        return [DiagnosticRow("final draw by rejection", None, "True", None, "not recorded")]
    z = info.get("x_o_min_rms_z")
    hint = "" if z is None else f" (closest prior simulation {float(z):.2g} sigma RMS from the data)"
    return [DiagnosticRow(
        "final draw by rejection", bool(conv), "True", bool(conv),
        "the posterior was drawn from the flow without a fallback" if conv else
        f"the final draw fell back to MCMC because the flow put too little mass inside the prior "
        f"box (leakage={info.get('leakage')}){hint}. In a 45-fit study such fits sat a median 1.61 "
        f"emcee sigma from emcee, against 1.00 without the fallback. Add simulations or rounds, "
        f"or check that the prior can produce the data")]


def _other_rows(result):
    info = result.info if isinstance(result.info, dict) else {}
    if "converged" not in info:
        return []
    conv = info["converged"]
    problems = info.get("convergence_problems") or []
    return [DiagnosticRow("sampler's own convergence flag", conv, "True",
                          None if conv is None else bool(conv),
                          "; ".join(problems) if problems else
                          ("the sampler reports convergence" if conv else
                           "the sampler reports no convergence"))]


def _prior_box(result, prior):
    """``{name: (type, low, high)}`` from ``prior`` or the prior recorded with the fit."""
    if prior is not None:
        rec = prior_record(prior)["parameters"]
    else:
        prov = getattr(result, "provenance", None)
        prov = prov if isinstance(prov, dict) else {}
        rec = (((prov.get("model") or {}).get("prior")) or {}).get("parameters")
    if not rec:
        return None
    return {n: (r.get("type"), r.get("low"), r.get("high")) for n, r in rec.items()
            if r.get("low") is not None and r.get("high") is not None}


def _edge_row(result, prior):
    thr = (f"<= {EDGE_FRACTION_MAX:.0%} of draws in the outer {EDGE_BAND:.0%} of a prior range")
    box = _prior_box(result, prior)
    if box is None:
        return DiagnosticRow("prior-edge pile-up", None, thr, None,
                             "the prior was not recorded with this result: pass prior= to check it")
    if result.n_samples == 0:
        return DiagnosticRow("prior-edge pile-up", None, thr, None, "no draws")
    piled, worst = [], 0.0
    for name, (kind, lo, hi) in box.items():
        if name not in result.samples.columns or not hi > lo:
            continue
        x = result.samples[name].to_numpy(dtype=float)
        if kind == "LogUniform" and lo > 0:
            x, lo, hi = np.log10(np.clip(x, 1e-300, None)), np.log10(lo), np.log10(hi)
        band = EDGE_BAND * (hi - lo)
        for side, frac, bound in (("lower", np.mean(x <= lo + band), lo),
                                  ("upper", np.mean(x >= hi - band), hi)):
            worst = max(worst, float(frac))
            if frac > EDGE_FRACTION_MAX:
                shown = 10 ** bound if kind == "LogUniform" else bound
                piled.append(f"{frac:.0%} of {name!r} at its {side} bound ({shown:g})")
    if not piled:
        return DiagnosticRow("prior-edge pile-up", worst, thr, True,
                             "no parameter piles up against a prior bound")
    return DiagnosticRow("prior-edge pile-up", worst, thr, False,
                         "; ".join(piled) + ": the bound, not the data, limits these parameters. "
                         "Widen the prior if the bound is not physical; otherwise report a limit")


def _likelihood_max_row(result, likelihood_max_opt):
    peak = float(getattr(likelihood_max_opt, "max_log_likelihood", likelihood_max_opt))
    gap = peak - float(result.max_log_likelihood)
    thr = f"<= {LIKELIHOOD_MAX_GAP_MAX:g} nats"
    if gap <= LIKELIHOOD_MAX_GAP_MAX:
        return DiagnosticRow("gap to the likelihood maximum", gap, thr, True,
                             "the sampler's best draw is close to the likelihood peak")
    return DiagnosticRow("gap to the likelihood maximum", gap, thr, False,
                         f"the optimised likelihood maximum is {gap:.1f} nats above the sampler's "
                         f"best draw: the sampler never reached the peak, so its posterior may sit "
                         f"in another region. Start from the likelihood maximum (init=) and run "
                         f"again")


def diagnose(result, *, prior=None, likelihood_max_opt=None):
    """The convergence report of ``result``; see :meth:`SamplerResult.diagnostics
    <whisper_cbpf.samplers.base.SamplerResult.diagnostics>` (this is its implementation).

    Examples
    --------
    >>> import pandas as pd
    >>> from whisper_cbpf.results import diagnose
    >>> from whisper_cbpf.samplers.base import SamplerResult
    >>> empty = SamplerResult("abc", "toy", ["a"], pd.DataFrame({"a": []}), {}, {}, 10, 1, 0.1,
    ...                       {"n_accepted": 0})
    >>> diagnose(empty).reasons[0]                          # doctest: +ELLIPSIS
    'posterior draws: the sampler returned no posterior draws...'
    """
    kind = _kind(result)
    chains = chains_of(result)
    rows = _common_rows(result)
    if kind == "nuts":
        rows += _nuts_rows(result, chains)
    elif kind == "ensemble":
        rows += _ensemble_rows(result, chains)
    elif kind == "abc":
        rows += _abc_rows(result)
    elif kind == "abc_smc":
        rows += _smc_rows(result)
    elif kind == "nested":
        rows += _nested_rows(result)
    elif kind == "snpe":
        rows += _snpe_rows(result)
    else:
        rows += _other_rows(result)
    rows.append(_edge_row(result, prior))
    if likelihood_max_opt is not None:
        rows.append(_likelihood_max_row(result, likelihood_max_opt))
    return DiagnosticsReport(sampler=str(result.sampler), model=str(result.model), kind=kind,
                             rows=rows)
