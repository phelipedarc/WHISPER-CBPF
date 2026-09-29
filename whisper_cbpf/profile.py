"""What one model evaluation costs on this hardware, and how many alerts fit in a night.

:func:`profile` times a model's log-density (value, and value with gradient) at several batch
sizes, reads XLA's memory accounting for each compiled program, and checks that the gradient is
finite where the density is; :func:`capacity` turns a measured cost per alert into alerts per night
on N GPUs. They replace hand-built timing tables, where this arithmetic is easy to get wrong
(serial against parallel time, wall against GPU seconds, per model against per alert).

No JAX at module scope.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["profile", "capacity", "ProfileReport"]

#: Scan draws ``fit_batch``'s default start scores per light curve, beyond the chain itself.
def _scan_draws(nwalkers):
    return max(1000, 4 * int(nwalkers))


@dataclass
class ProfileReport:
    """Cost of one model's log-density on one device (:func:`profile`).

    Attributes
    ----------
    model : str
        The model's name.
    device : str
        The JAX device the programs ran on.
    x64 : bool
        Whether the session computes in float64.
    n_data, bucket : int
        Observations in the light curve, and the padded count the program was compiled for.
    names : list of str
        The sampled parameters.
    rows : list of dict
        One per (batch size, ``"value"`` or ``"value_and_grad"``): ``batch``, ``kind``,
        ``compile_s``, ``call_s`` (median of the repeats), ``per_eval_us``, ``temp_bytes`` and
        ``temp_bytes_per_eval`` (XLA's scratch memory for the program, ``None`` where the
        backend does not report it), and ``skipped`` (why a program was not run, else ``None``).
    finite_fraction : float
        Fraction of the prior draws at which the log-density is finite (the rest are outside a
        constraint wall, or the physics failed).
    gradient : dict
        On the prior draws where the density is finite: ``n``, ``finite_fraction`` (gradients with
        every component finite), ``zero_fraction`` (gradients exactly zero: a flat or walled
        region) and ``median_abs`` per parameter. Empty when ``grad=False``.
    peak_bytes : int or None
        The device's peak memory in use in this process so far (``memory_stats``), where JAX
        reports it.
    """

    model: str
    device: str
    x64: bool
    n_data: int
    bucket: int
    names: list
    rows: list = field(default_factory=list)
    finite_fraction: float = float("nan")
    gradient: dict = field(default_factory=dict)
    peak_bytes: object = None

    def seconds_per_eval(self, kind="value"):
        """The lowest measured cost of one evaluation (seconds), over the batch sizes that ran.

        Examples
        --------
        >>> from whisper_cbpf.profile import ProfileReport
        >>> r = ProfileReport("m", "cpu", True, 30, 32, ["a"],
        ...                   rows=[{"batch": 1, "kind": "value", "per_eval_us": 40.0},
        ...                         {"batch": 64, "kind": "value", "per_eval_us": 5.0}])
        >>> r.seconds_per_eval()
        5e-06
        """
        vals = [r["per_eval_us"] for r in self.rows
                if r.get("kind") == kind and r.get("per_eval_us") is not None]
        if not vals:
            raise ValueError(f"profile of {self.model!r} has no measured {kind!r} row; run "
                             f"profile(..., grad=True) for gradients, or check .rows for why "
                             f"each batch size was skipped.")
        return float(min(vals)) / 1e6

    def seconds_per_alert(self, *, nwalkers=32, nsteps=5000):
        """Device seconds for one ``fit_batch`` fit: ``(nsteps * nwalkers + scan) x`` the lowest
        cost per evaluation, the batch being full (many alerts per call).

        The scan is the first stage of ``fit_batch``'s default start, ``max(1000, 4 nwalkers)``
        draws; the climb that follows is not counted (measured on one A6000 at 64-512 light
        curves per call: 0.31 s per light curve for the free Arnett, 0.25 s for the TDE, against
        5.3 s and 3.3 s for a 60 x 10 000 chain). Compile time, host work (metrics, saving) and a
        batch too small to fill the card are not included either.

        Examples
        --------
        >>> from whisper_cbpf.profile import ProfileReport
        >>> r = ProfileReport("m", "gpu", True, 30, 32, ["a"],
        ...                   rows=[{"batch": 4096, "kind": "value", "per_eval_us": 23.0}])
        >>> round(r.seconds_per_alert(nwalkers=60, nsteps=10000), 2)
        13.82
        """
        n_evals = int(nsteps) * int(nwalkers) + _scan_draws(nwalkers)
        return n_evals * self.seconds_per_eval("value")

    def to_dict(self):
        """A JSON-able dict of every field."""
        return {"model": self.model, "device": self.device, "x64": self.x64,
                "n_data": self.n_data, "bucket": self.bucket, "names": list(self.names),
                "rows": [dict(r) for r in self.rows], "finite_fraction": self.finite_fraction,
                "gradient": dict(self.gradient), "peak_bytes": self.peak_bytes}

    def __repr__(self):
        head = (f"Profile of {self.model!r} on {self.device} ({'float64' if self.x64 else 'float32'}"
                f", {self.n_data} points padded to {self.bucket}, {len(self.names)} parameters)")
        lines = [head, f"  {'batch':>6}  {'kind':<15} {'compile s':>9} {'call ms':>10} "
                       f"{'us / eval':>10} {'scratch MB':>10}"]
        for r in self.rows:
            if r.get("skipped"):
                lines.append(f"  {r['batch']:>6}  {r['kind']:<15} skipped: {r['skipped']}")
                continue
            mb = ("" if r.get("temp_bytes") is None else f"{r['temp_bytes'] / 1e6:10.1f}")
            lines.append(f"  {r['batch']:>6}  {r['kind']:<15} {r['compile_s']:9.2f} "
                         f"{1e3 * r['call_s']:10.3f} {r['per_eval_us']:10.2f} {mb:>10}")
        lines.append(f"  finite log-density on {100 * self.finite_fraction:.1f}% of prior draws")
        if self.gradient:
            g = self.gradient
            lines.append(f"  gradient: finite on {100 * g['finite_fraction']:.1f}%, exactly zero "
                         f"on {100 * g['zero_fraction']:.1f}% of {g['n']} finite draws")
        if self.peak_bytes is not None:
            lines.append(f"  device peak memory in this process: {self.peak_bytes / 1e9:.2f} GB")
        lines.append("Read next: .seconds_per_alert(nwalkers=, nsteps=) and "
                     "capacity(report, hours=, n_gpus=).")
        return "\n".join(lines)


def _time_calls(fn, args, repeats):
    """Median wall seconds of ``repeats`` calls of an already-compiled ``fn`` (after one warm-up)."""
    import jax

    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _free_bytes(device):
    """Bytes the device can still allocate, or None when JAX does not report it."""
    try:
        stats = device.memory_stats()
    except Exception:                                # noqa: BLE001 - backend-dependent
        return None
    if not stats or "bytes_limit" not in stats:
        return None
    return int(stats["bytes_limit"]) - int(stats.get("bytes_in_use", 0))


def profile(model, lc, *, batch_sizes=(1, 64, 480, 4096), grad=True, prior=None, space="auto",
            likelihood="auto", repeats=5, seed=0):
    """Cost of one evaluation of ``model``'s log-density on ``lc``, by batch size, on this device.

    For each batch size B, the log-density (and, with ``grad=True``, its value and gradient) is
    ``vmap``-ped over B parameter sets drawn from the prior, compiled once (the compile time is
    reported apart), then called ``repeats`` times; the median call over B is the cost per
    evaluation. XLA's own memory accounting of each compiled program gives its scratch memory,
    and a program whose scratch would not fit in the device's free memory is skipped with the
    reason instead of run. The gradient is checked on the prior draws where the density is finite:
    the fraction with every component finite, and the fraction exactly zero.

    Parameters
    ----------
    model : str or Model
        A JAX model (``predict_jax``).
    lc : LightCurve
        A light curve typical of the data: the cost grows with its padded length.
    batch_sizes : sequence of int
        Parameter sets per call. A sampler's batch is ``alerts x walkers / 2`` for ``fit_batch``.
    grad : bool
        Also time ``value_and_grad`` and check the gradient.
    prior, space, likelihood
        As for :func:`~whisper_cbpf.samplers.jax._adapters.log_density`.
    repeats : int
        Timed calls per program (the median is kept).
    seed : int
        Seed of the prior draws.

    Returns
    -------
    ProfileReport
        ``.rows`` (compile s, call s, us per evaluation, scratch bytes per batch size and kind),
        ``.finite_fraction``, ``.gradient``, ``.peak_bytes``; ``.seconds_per_alert(nwalkers=,
        nsteps=)`` for :func:`capacity`.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> flare = wp.get_model("flare_jax")
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"log_amp": 1.0, "log_sigma": 0.5, "log_tau": 1.5, "t0": 10.0}
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flare.predict(truth, t),
    ...                    flux_err=np.full(30, 0.1))
    >>> report = wp.profile("flare_jax", lc, batch_sizes=(1, 64))
    >>> [(r["batch"], r["kind"]) for r in report.rows]
    [(1, 'value'), (1, 'value_and_grad'), (64, 'value'), (64, 'value_and_grad')]
    >>> report.gradient["finite_fraction"]
    1.0
    >>> wp.capacity(report, hours=12, n_gpus=1)["alerts"] > 0
    True
    """
    from .backends import require_jax
    from .samplers.jax import _diagnostics as _dg
    from .samplers.jax._adapters import float_dtype, log_density

    jax, jnp = require_jax("profile")
    sizes = [int(b) for b in batch_sizes]
    if not sizes or min(sizes) < 1:
        raise ValueError(f"profile: batch_sizes must be positive integers; got {batch_sizes!r}.")
    ld = log_density(lc, model, space=space, likelihood=likelihood, prior=prior)
    dt = float_dtype()
    free_prior = _dg.split_fixed(ld.prior, list(ld.names) + list(ld.fixed), "profile")[0]
    draws = _dg.prior_draws(free_prior, ld.names, max(sizes), seed)
    device = jax.devices()[0]

    def post(theta, data):
        return ld.shared(theta, data)[0]

    # The data are ARGUMENTS of the timed programs, as in fit_batch, not constants of them.
    value = jax.vmap(post, in_axes=(0, None))
    both = jax.vmap(jax.value_and_grad(post), in_axes=(0, None))
    report = ProfileReport(model=ld.model, device=str(device), x64=bool(dt == jnp.float64),
                           n_data=ld.n_data, bucket=ld.bucket, names=list(ld.names))
    kinds = [("value", value)] + ([("value_and_grad", both)] if grad else [])
    finite, grad_programs = None, {}
    for b in sizes:
        theta = jnp.asarray(draws[:b], dtype=dt)
        for kind, f in kinds:
            row = {"batch": b, "kind": kind, "compile_s": None, "call_s": None,
                   "per_eval_us": None, "temp_bytes": None, "temp_bytes_per_eval": None,
                   "skipped": None}
            t0 = time.perf_counter()
            compiled = jax.jit(f).lower(theta, ld.data).compile()
            row["compile_s"] = float(time.perf_counter() - t0)
            try:
                temp = int(compiled.memory_analysis().temp_size_in_bytes)
                row["temp_bytes"], row["temp_bytes_per_eval"] = temp, temp / b
            except Exception:                        # noqa: BLE001 - backend-dependent
                temp = None
            free = _free_bytes(device)
            if temp is not None and free is not None and temp > 0.9 * free:
                row["skipped"] = (f"needs {temp / 1e9:.1f} GB of scratch memory, "
                                  f"{free / 1e9:.1f} GB free on {device}")
                report.rows.append(row)
                continue
            row["call_s"] = _time_calls(compiled, (theta, ld.data), repeats)
            row["per_eval_us"] = 1e6 * row["call_s"] / b
            report.rows.append(row)
            if kind == "value" and (finite is None or b > finite[0]):
                finite = (b, np.asarray(compiled(theta, ld.data), dtype=float))
            elif kind == "value_and_grad":
                grad_programs[b] = compiled
    if finite is not None:
        report.finite_fraction = float(np.mean(np.isfinite(finite[1])))
    if grad:
        report.gradient = _gradient_health(jnp, grad_programs, ld.data, draws, finite, dt)
    try:
        stats = device.memory_stats() or {}
        report.peak_bytes = int(stats["peak_bytes_in_use"]) if "peak_bytes_in_use" in stats else None
    except Exception:                                # noqa: BLE001 - backend-dependent
        report.peak_bytes = None
    return report


def _gradient_health(jnp, programs, data, draws, finite, dt, n_max=480):
    """Finite and zero gradient fractions on up to ``n_max`` prior draws where the value is
    finite, with the gradient program of the largest measured batch size not above ``n_max``
    (the smallest one when every one is larger)."""
    if finite is None or not programs:
        return {}
    ok = np.flatnonzero(np.isfinite(finite[1]))[:n_max]
    if ok.size == 0:
        return {"n": 0, "finite_fraction": float("nan"), "zero_fraction": float("nan"),
                "median_abs": None}
    width = max([b for b in programs if b <= n_max] or [min(programs)])
    f = programs[width]
    rows = draws[ok]
    pad = (-rows.shape[0]) % width
    padded = np.concatenate([rows, np.repeat(rows[-1:], pad, axis=0)]) if pad else rows
    grads = []
    for i in range(0, padded.shape[0], width):
        _, g = f(jnp.asarray(padded[i:i + width], dtype=dt), data)
        grads.append(np.asarray(g, dtype=float))
    g = np.concatenate(grads)[:rows.shape[0]]
    fin = np.all(np.isfinite(g), axis=1)
    zero = np.all(g == 0.0, axis=1)
    med = np.median(np.abs(g[fin]), axis=0) if fin.any() else None
    return {"n": int(rows.shape[0]), "finite_fraction": float(np.mean(fin)),
            "zero_fraction": float(np.mean(zero)),
            "median_abs": None if med is None else [float(v) for v in med]}


def capacity(cost, *, hours=12.0, n_gpus=1, nwalkers=32, nsteps=5000):
    """How many alerts fit in ``hours`` on ``n_gpus`` cards, from a measured cost per alert.

    Parameters
    ----------
    cost : float, ProfileReport, or a sequence of them
        Device seconds per alert: a measured number (for example a ``fit_batch`` result's
        ``runtime_s``, its share of the batch's start and chain), or a :class:`ProfileReport`,
        converted with :meth:`ProfileReport.seconds_per_alert` at ``nwalkers`` / ``nsteps``. A
        sequence is summed: one alert fitted with each model of a comparison.
    hours : float
        The observing window, 12 h by default.
    n_gpus : int
        Cards working in parallel, each kept busy.
    nwalkers, nsteps : int
        Chain size for a :class:`ProfileReport` (``fit_batch``'s defaults).

    Returns
    -------
    dict
        ``alerts`` (whole alerts finished, rounded down), ``alerts_exact``,
        ``alerts_per_gpu_hour``, ``seconds_per_alert``, ``hours``, ``n_gpus``, ``basis`` (where
        the cost came from) and ``assumptions``.

    Examples
    --------
    At 636.7 s per alert, one alert at a time, a 12-hour night holds ~68 alerts:

    >>> import whisper_cbpf as wp
    >>> round(wp.capacity(636.7, hours=12)["alerts_exact"])
    68

    A batched Arnett chain at 13.93 GPU s per alert and model:

    >>> round(wp.capacity(13.928)["alerts_per_gpu_hour"], 1)
    258.5
    """
    items = list(cost) if isinstance(cost, (list, tuple)) else [cost]
    if not items:
        raise ValueError("capacity: no cost given.")
    seconds, basis = 0.0, []
    for c in items:
        if isinstance(c, ProfileReport):
            s = c.seconds_per_alert(nwalkers=nwalkers, nsteps=nsteps)
            basis.append(f"{c.model}: profile on {c.device}, {c.seconds_per_eval() * 1e6:.2f} us "
                         f"per evaluation x ({nsteps} steps x {nwalkers} walkers + "
                         f"{_scan_draws(nwalkers)} scan draws) = {s:.2f} s")
        else:
            try:
                s = float(c)
            except (TypeError, ValueError):
                raise TypeError(f"capacity: cost must be seconds per alert (a number), a "
                                f"ProfileReport, or a list of them; got {type(c).__name__}.") from None
            basis.append(f"measured: {s:.4g} s per alert")
        if not (math.isfinite(s) and s > 0):
            raise ValueError(f"capacity: a cost of {s!r} s per alert is not a positive number.")
        seconds += s
    hours, n_gpus = float(hours), int(n_gpus)
    if not (hours > 0 and n_gpus >= 1):
        raise ValueError(f"capacity: hours must be > 0 and n_gpus >= 1; got {hours}, {n_gpus}.")
    exact = hours * 3600.0 * n_gpus / seconds
    return {"alerts": int(math.floor(exact + 1e-9)), "alerts_exact": float(exact),
            "alerts_per_gpu_hour": float(3600.0 / seconds), "seconds_per_alert": float(seconds),
            "hours": hours, "n_gpus": n_gpus, "basis": basis,
            "assumptions": ["every card is busy for the whole window (a full queue, no gaps)",
                            "the cost is device time only: compile time, host work (ingest, "
                            "metrics, reports, saving) and CPU stages are not counted",
                            "an alert that does not finish inside the window is not counted"]}
