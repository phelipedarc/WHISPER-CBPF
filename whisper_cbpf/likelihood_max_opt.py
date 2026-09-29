"""Maximum-likelihood optimisation: the maximum likelihood behind AIC and BIC.

AIC and BIC are defined at the maximum of the likelihood, ``-2 ln L_max + k (2 or ln n)``. A
sampler maps the bulk of the posterior and rarely lands on its summit, so the ``max_log_likelihood``
a fit reports is a lower bound, short by a different amount for each model. A ranking built on it
carries sampler noise: on real ZTF supernovae one pass stopped 1.8 ln L short of the peak and
swapped the top two models, another turned a "decisive" preference into a "strong" one, and a GPU
climb in logit coordinates, which cannot reach a prior edge, fell 4-31 ln L short on the fits
whose optimum sits on one.

:func:`likelihood_max_opt` climbs from the fit's own best draws to the peak of the fit's own
likelihood and recomputes AIC and BIC there. It never changes the posterior: samples, medians and
error bars are the sampler's. The optimiser works in **box coordinates** -- each parameter mapped
linearly onto ``[0, 1]`` in its prior's own coordinate (``log10`` for a LogUniform) -- so a parameter can rest
exactly on a prior edge and is still pulled back in by the likelihood. A parameter whose prior has
no finite box (a ``Normal``, a one-sided ``TruncatedNormal``) is climbed in standardised
coordinates ``(x - mu) / sigma`` instead, bounded only where the prior is; a ``Fixed`` one is held
at its value and never optimised.
"""
from __future__ import annotations

import math
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields

import numpy as np

from .models import Model, get_model
from .samplers.base import _LIKELIHOOD_KINDS, aic_bic

__all__ = ["LikelihoodMaxOptResult", "likelihood_max_opt"]

#: A parameter whose peak value lies within this fraction of its box width (in the prior's own
#: coordinate) of a box face is reported in :attr:`LikelihoodMaxOptResult.at_edge`.
EDGE_TOL = 1e-3
#: A parameter with an unbounded prior (a Normal, or a TruncatedNormal with an infinite end) is
#: climbed in standardised coordinates, ``u = (x - mu) / sigma``. This width in ``u`` (+-2 prior
#: sigma) plays the role of a box's [0, 1] for the optimisers' step sizes and for the edge test of
#: a one-sided TruncatedNormal.
STD_SPAN = 4.0
#: Largest number of compiled program sets the JAX backend keeps between calls, oldest dropped
#: first (see :func:`_jax_programs`): one per model, prior families, likelihood and bucket.
PROGRAM_CACHE_SIZE = 16
#: The batched value-and-gradient program is compiled at a multiple of this width: the Adam batch
#: (``n_starts`` plus earlier peaks) and L-BFGS-B's single points are padded to it, so an
#: optimisation with up to this many starts compiles it once.
VG_WIDTH = 8
#: A best draw this far outside its box (in box units) is refused as "fitted with another prior";
#: closer than this it is clipped (a float32 draw can land one rounding step past a bound).
BOX_SLACK = 1e-6
#: The fit's recorded ``max_log_likelihood`` and its best draw re-scored here may differ by this
#: much (or 1e-6 relative) before :func:`likelihood_max_opt` warns that the two densities differ.
DENSITY_MISMATCH = 1e-3
#: Largest number of posterior draws the JAX backend scores to choose its candidates.
MAX_SCORED = 20000
#: Projected Adam (JAX backend): steps, and the cosine-decayed step size in box units.
ADAM_STEPS = 300
ADAM_LR = (1e-2, 1e-5)
#: L-BFGS-B iteration cap per climb.
LBFGS_MAXITER = 500
#: Nelder-Mead: initial simplex edge in box units (the reference implementation's 5 % of the box),
#: position tolerance in box units, and evaluation cap per parameter.
NM_STEP = 0.05
NM_XATOL = 1e-7
NM_MAXFEV_PER_PARAM = 400
#: Objective value (a minimisation of -ln L) for a point whose ln L is not finite.
_BAD = 1e300


@dataclass(frozen=True)
class LikelihoodMaxOptResult:
    """The peak of a fit's likelihood, found by :func:`likelihood_max_opt`.

    Attributes
    ----------
    params : dict
        Parameter values at the peak, one entry per fitted parameter (a ``Fixed`` one at its
        value).
    max_log_likelihood : float
        ln L at the peak. Never below :attr:`start_log_likelihood`.
    start_log_likelihood : float
        ln L of the sampler's best draw (``result.best_params``), re-scored by the same density.
    gain : float
        ``max_log_likelihood - start_log_likelihood`` (>= 0). A gain above ~1 means the sampler
        never reached the peak, and a ranking by the sampler's own BIC would have been off by
        twice that.
    at_edge : list of str
        Parameters whose peak value lies within ``EDGE_TOL`` (0.1 %) of a prior-box face, in the
        prior's own coordinate (for a one-sided TruncatedNormal, within 0.1 % of ``STD_SPAN``
        prior sigmas of its finite end). The data push them against the prior: widen that prior,
        or report the bound as a limit.
    aic, bic : float
        ``-2 ln L_max + 2k`` and ``-2 ln L_max + k ln n`` at the peak. ``nan`` when there are not
        more data points than parameters (``n_data <= n_params``): the peak is then not constrained
        by the data, and a number would mislead.
    n_data, n_params : int
        ``n`` and ``k`` used by AIC and BIC; ``k`` counts the free parameters only.
    n_evals : int
        Likelihood evaluations spent (a value-and-gradient evaluation counts as one).
    runtime_s : float
        Wall time of the optimisation, compilation included.
    method : str
        The optimiser chain that ran, and where.
    sampler_log_likelihood : float
        The ``max_log_likelihood`` the fit itself recorded, the ln L its own AIC and BIC used.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
    >>> peak = wp.likelihood_max_opt(wp.fit_ABC(lc, "flare", n_simulations=2000, seed=0), lc)
    >>> sorted(peak.params)
    ['amplitude', 'decay_time', 'rise_time']
    >>> peak.gain >= 0.0 and peak.n_data == 30
    True
    """

    params: dict
    max_log_likelihood: float
    start_log_likelihood: float
    gain: float
    at_edge: list
    aic: float
    bic: float
    n_data: int
    n_params: int
    n_evals: int
    runtime_s: float
    method: str
    sampler_log_likelihood: float = float("nan")

    @property
    def enough_data(self):
        """``True`` when there are more data points than parameters, so AIC and BIC exist."""
        return self.n_data > self.n_params

    def to_dict(self):
        """Plain JSON-ready dict of every field, e.g. to keep it as
        ``result.info["likelihood_max_opt"]``.

        Examples
        --------
        >>> import numpy as np
        >>> import whisper_cbpf as wp
        >>> t = np.linspace(0.5, 30.0, 30)
        >>> flux = wp.get_model("flare").predict(
        ...     {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
        >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
        >>> res = wp.fit_ABC(lc, "flare", n_simulations=2000, seed=0)
        >>> res.info["likelihood_max_opt"] = wp.likelihood_max_opt(res, lc).to_dict()
        >>> wp.LikelihoodMaxOptResult.from_dict(res.info["likelihood_max_opt"]).n_params
        3
        """
        out = asdict(self)
        out["params"] = {k: float(v) for k, v in self.params.items()}
        out["at_edge"] = list(self.at_edge)
        return out

    @classmethod
    def from_dict(cls, data):
        """Rebuild a :class:`LikelihoodMaxOptResult` from :meth:`to_dict`'s output (unknown keys
        ignored).

        Examples
        --------
        >>> import whisper_cbpf as wp
        >>> d = {"params": {"a": 1.0}, "max_log_likelihood": -3.0, "start_log_likelihood": -3.5,
        ...      "gain": 0.5, "at_edge": [], "aic": 8.0, "bic": 8.9, "n_data": 10, "n_params": 1,
        ...      "n_evals": 40, "runtime_s": 0.1, "method": "example"}
        >>> wp.LikelihoodMaxOptResult.from_dict(d).gain
        0.5
        """
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(data).items() if k in names})

    def __repr__(self):
        k, n = self.n_params, self.n_data
        lines = [f"LikelihoodMaxOptResult: peak ln L = {self.max_log_likelihood:.4f} "
                 f"({k} parameters, {n} data points)",
                 f"  gain over the sampler's best draw: {self.gain:+.4f} "
                 f"({self.start_log_likelihood:.4f} -> {self.max_log_likelihood:.4f})"]
        if self.enough_data:
            lines.append(f"  from the peak: AIC = {self.aic:.2f}, BIC = {self.bic:.2f}")
        else:
            lines.append(f"  AIC, BIC: not enough data (k = {k} >= n = {n}); fit fewer "
                         f"parameters or wait for more points")
        lines.append("  on a prior edge: " + (", ".join(self.at_edge) if self.at_edge else "none"))
        lines.append("  peak: " + ", ".join(f"{p}={v:.6g}" for p, v in self.params.items()))
        lines.append(f"  {self.n_evals} evaluations in {self.runtime_s:.1f} s; {self.method}")
        lines.append("  read next: .gain (> ~1: the sampler never reached the peak), .at_edge "
                     "(widen that prior or quote a bound), .bic (rank models on it)")
        return "\n".join(lines)


def likelihood_max_opt(result, lc, model=None, *, n_candidates=300, n_starts=5, tol=1e-4,
                       max_rounds=20, space=None, likelihood=None, backend="auto", seed=0,
                       prior=None):
    """Find the peak of a fit's likelihood and recompute AIC and BIC there.

    Starting from the fit's best draws, a bounded local optimiser climbs the same likelihood the
    fit was scored with (``model.predict`` or, when the model has one, its JAX density; the same
    ``space`` and likelihood class; the model's constraint wall). The posterior is not touched:
    samples, medians and error bars stay the sampler's.

    1. **Candidates.** Up to ``n_candidates`` of the fit's draws, scored by the fit's likelihood:
       the best by a ``log_likelihood`` column when the samples carry one; else, on the JAX
       backend, the best of every draw (up to ``MAX_SCORED``); else, on the CPU, the
       lowest-``distance`` draws of an ABC fit or a random subset (``seed``). The fit's best draw,
       and any earlier peak kept in ``result.info["likelihood_max_opt"]``, are always candidates.
    2. **Climbs.** From the ``n_starts`` best candidates, and from every earlier peak: projected
       Adam (JAX backend only, all starts in one batch), then bounded L-BFGS-B (exact gradients on
       the JAX backend, central differences on the CPU), then Nelder-Mead where L-BFGS-B stopped
       without converging. All in box coordinates, so an optimum on a prior edge is reached
       exactly (a logit map puts the edge at infinity).
    3. **Restarts.** The two best climbs are restarted (a fresh Nelder-Mead simplex, then
       L-BFGS-B) until a round gains less than ``tol`` in ln L, at most ``max_rounds`` times.

    Every evaluated point is kept, so the peak is never below the sampler's best draw scored by
    the same density. A ``Normal`` or one-sided ``TruncatedNormal`` parameter is climbed in
    standardised coordinates, ``(x - mu) / sigma``, bounded only on the prior's finite side; a
    ``Fixed`` parameter (in the prior, or in the fit's ``info["fixed"]``) is held at its value,
    never optimised and not counted in AIC/BIC. The JAX backend's compiled programs take the data
    and the prior's numbers as arguments (:func:`~whisper_cbpf.samplers.jax._adapters.log_density`)
    and are kept per forward model, prior families, likelihood and size bucket (up to
    ``PROGRAM_CACHE_SIZE``), so optimising again -- the same fit, or another light curve fitted
    with the same model object, as after :func:`whisper_cbpf.fit_batch` -- does not compile again.

    Parameters
    ----------
    result : SamplerResult
        The fit to optimise. Its ``parameters``, ``best_params``, ``samples``, ``n_data`` and
        ``info`` (``space``, ``likelihood``, ``scatter_param``, ``likelihood_max_opt``) are read.
    lc : LightCurve
        The light curve the fit was run on, after the same selection steps. The rows the fit left
        out as pre-event data (``result.fitted_lc``) are left out here too.
    model : str or Model, optional
        The fitted model. Default: ``result.model`` looked up in the registry. Pass another model
        with the same parameters to score the same draws with it, e.g. a redback CPU model for a
        fit run on its JAX twin.
    n_candidates : int, default 300
        Draws scored to choose the starts.
    n_starts : int, default 5
        Local climbs from the best candidates.
    tol : float, default 1e-4
        A restart round that gains less than this in ln L ends the restarts.
    max_rounds : int, default 20
        Restart rounds per climb, at most.
    space : {"flux", "magnitude", "auto"}, optional
        Comparison space. Default: the one the fit recorded (``result.info["space"]``).
    likelihood : str or likelihood object, optional
        A registered likelihood kind (``"gaussian"``, ``"upper_limits"``, ``"gaussian_scatter"``,
        ``"mixture"``, ...) or a built likelihood. Default: the class the fit recorded
        (``result.info["likelihood"]``; a free-scatter fit's ``scatter_param`` is kept).
    backend : {"auto", "jax", "cpu"}, default "auto"
        ``"auto"`` uses the JAX density (batched scoring, gradients) when the model has a
        ``predict_jax`` and JAX implements the likelihood, else the CPU (scipy). A JAX-backend
        optimisation in a float32 session carries float32 rounding in ln L; enable float64 to
        resolve ``tol``.
    seed : int, default 0
        Seed for the random candidate subset (CPU backend, samples without a score).
    prior : Prior, optional
        The prior the fit used: it sets the box. Default: the prior the fit recorded
        (``result.provenance``, for a fit run through a registered sampler), else the model's
        default. ``Uniform``, ``LogUniform``, ``Normal``, ``TruncatedNormal`` and ``Fixed``.

    Returns
    -------
    LikelihoodMaxOptResult
        The peak parameters and ln L, the gain over the sampler's best draw, the parameters on a
        prior edge, and AIC/BIC from the peak (``nan`` with a warning when ``n <= k``: not enough
        data).

    Raises
    ------
    ValueError
        ``lc`` is not the fit's light curve (another number of points); the fit's best draw lies
        outside the prior box (pass ``prior=``); a fitted parameter is neither in the model nor
        read by the likelihood; ``backend="jax"`` where the JAX density is unavailable; every
        parameter ``Fixed``; a setting out of range.
    TypeError
        A prior family other than ``Uniform``, ``LogUniform``, ``Normal``, ``TruncatedNormal`` or
        ``Fixed``.

    Warns
    -----
    UserWarning
        The fit's recorded ``max_log_likelihood`` differs from its best draw re-scored here, so
        the fit was scored under another density (pass ``space=`` / ``likelihood=`` to match it);
        ``n <= k`` (AIC and BIC are ``nan``); or the JAX backend ran in float32.

    Examples
    --------
    >>> import numpy as np
    >>> import whisper_cbpf as wp
    >>> t = np.linspace(0.5, 30.0, 30)
    >>> truth = {"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}
    >>> flux = wp.get_model("flare").predict(truth, t, None)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full(30, 0.1))
    >>> res = wp.fit_ABC(lc, "flare", n_simulations=2000, seed=0)
    >>> peak = wp.likelihood_max_opt(res, lc)
    >>> peak.max_log_likelihood >= res.max_log_likelihood
    True
    >>> bool(abs(peak.params["amplitude"] - 5.0) < 0.05)       # noiseless data: the truth
    True
    >>> peak.bic <= res.bic
    True
    """
    t_start = time.perf_counter()
    _check_settings(n_candidates, n_starts, tol, max_rounds)
    fitted = getattr(result, "fitted_lc", None)
    if callable(fitted):                     # the rows the fit used: its pre-event rows left out
        lc = fitted(lc)
    model = _resolve_model(result, model)
    names = [str(p) for p in result.parameters]
    lik = _resolve_likelihood(result, lc, space, likelihood)
    _check_names(names, model, lik)
    box = _Box(_resolve_prior(model, prior, names, result), names)   # free parameters only
    n_data = _check_data(result, lc)
    use_jax = _use_jax(backend, model, lik)
    obj = (_JaxObjective if use_jax else _CpuObjective)(lc, model, box, names, lik)

    # --- candidates: the best draw, earlier peaks, then the pool --------------------------------
    x_best = box.checked(result.best_params, "the fit's best draw (result.best_params)", model)
    earlier = _earlier_peaks(getattr(result, "info", None) or {}, box.names, box)
    pool = _candidate_pool(getattr(result, "samples", None), box.names, box, n_candidates, seed,
                           score_all=use_jax)
    X = np.vstack([x_best[None, :]] + [e[None, :] for e in earlier] + [pool])
    vals = obj.score(X)
    start = float(vals[0])
    if not np.isfinite(start):
        raise ValueError(
            f"the fit's best draw scores ln L = {start} under this density ({type(lik).__name__}, "
            f"space={lik.space!r}, backend={'jax' if use_jax else 'cpu'}): it is outside the "
            f"model's constraint wall or its prior box, so the fit was scored some other way. "
            f"Pass the model, space= and likelihood= the fit used (backend='cpu' scores a draw "
            f"through model.predict, as the CPU samplers do).")
    recorded = float(getattr(result, "max_log_likelihood", float("nan")))
    _check_density(recorded, start, lik, result)
    if use_jax and np.dtype(obj.dtype) == np.float32:
        warnings.warn(
            f"likelihood_max_opt ran on the JAX density in this session's float32: ln L = "
            f"{start:.6g} is resolved to about "
            f"{max(abs(start), 1.0) * np.finfo(np.float32).eps:.1g}, against tol = {tol:g}, so the peak and the gain are uncertain at that level. Enable float64 "
            f"before any JAX computation (jax.config.update('jax_enable_x64', True)), or pass "
            f"backend='cpu'.", stacklevel=2)

    # --- climbs from the best candidates and from every earlier peak ----------------------------
    order, seen = [], set()         # best first, each point once (the best draw is usually in
    for i in np.argsort(-vals, kind="stable"):                      # the pool as well)
        if np.isfinite(vals[i]) and X[i].tobytes() not in seen:
            seen.add(X[i].tobytes())
            order.append(int(i))
    chosen = order[:n_starts] + [1 + j for j in range(len(earlier))
                                 if 1 + j in order and 1 + j not in order[:n_starts]]
    U = box.to_u(X)
    ends = obj.climb([_Point(U[i], X[i], float(vals[i])) for i in chosen], tol)

    # --- restart the two best until a round gains less than tol ---------------------------------
    ends.sort(key=lambda p: -p.f)
    for i, p in enumerate(ends[:2]):
        for _ in range(int(max_rounds)):
            q = obj.restart(p, tol)
            gain, p = q.f - p.f, q
            if gain < tol:
                break
        ends[i] = p
    best_i = int(np.argmax(vals))
    peak = max(ends + [_Point(U[best_i], X[best_i], float(vals[best_i]))], key=lambda p: p.f)

    k = len(box.names)                                   # a Fixed parameter is not counted
    if n_data > k:
        aic, bic = aic_bic(peak.f, k, n_data)
    else:
        aic = bic = float("nan")
        warnings.warn(
            f"not enough data: {n_data} data point(s) for {k} free parameter(s), so the peak is "
            f"not constrained by the data and AIC/BIC are left as nan. Fit fewer parameters (pin "
            f"some with pin=) or wait for more points.", stacklevel=2)
    at_edge = box.at_edge(box.to_u(peak.x[None, :])[0])
    return LikelihoodMaxOptResult(
        params={nm: float(v) for nm, v in zip(box.all_names, box.full(peak.x))},
        max_log_likelihood=float(peak.f), start_log_likelihood=start,
        gain=float(peak.f - start), at_edge=at_edge, aic=float(aic), bic=float(bic),
        n_data=int(n_data), n_params=int(k), n_evals=int(obj.n_evals),
        runtime_s=float(time.perf_counter() - t_start), method=obj.method,
        sampler_log_likelihood=recorded)


# --- inputs ---------------------------------------------------------------------------------------

def _check_settings(n_candidates, n_starts, tol, max_rounds):
    if int(n_candidates) < 0 or int(n_starts) < 1 or int(max_rounds) < 0:
        raise ValueError(f"likelihood_max_opt needs n_candidates >= 0, n_starts >= 1 and "
                         f"max_rounds >= 0; got n_candidates={n_candidates}, n_starts={n_starts}, "
                         f"max_rounds={max_rounds}.")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(f"tol must be a positive number of ln L units (default 1e-4); got "
                         f"{tol!r}.")


def _resolve_model(result, model):
    if model is None:
        from .samplers.base import fitted_model
        try:
            return fitted_model(result)
        except KeyError:
            raise ValueError(
                f"the fit's model {result.model!r} is not registered in this session, so "
                f"likelihood_max_opt cannot evaluate it. Pass model= (the Model object the fit "
                f"used, or register it again under the same name).") from None
    return model if isinstance(model, Model) else get_model(model)


def _resolve_likelihood(result, lc, space, likelihood):
    """The fit's own likelihood: the class and space it recorded, unless overridden."""
    from .likelihood import make_likelihood

    if likelihood is not None and not isinstance(likelihood, str):
        if not hasattr(likelihood, "log_likelihood"):
            raise TypeError(f"likelihood= must be a registered kind name or a likelihood object "
                            f"with log_likelihood(model_flux); got {type(likelihood).__name__}.")
        return likelihood
    info = getattr(result, "info", None) or {}
    sp = space if space is not None else (info.get("space") or "auto")
    kind = likelihood if likelihood is not None else _LIKELIHOOD_KINDS.get(
        str(info.get("likelihood")))
    scatter = info.get("scatter_param")
    if kind is None:
        kind = "gaussian_scatter" if scatter else "auto"
    extra = ({"scatter_param": scatter}
             if scatter and str(kind).lower() in ("gaussian_scatter", "scatter", "villar") else {})
    return make_likelihood(lc, kind=kind, space=sp, **extra)


def _check_names(names, model, lik):
    """Every fitted parameter must reach the model or the likelihood, or its peak is arbitrary."""
    scatter = getattr(lik, "scatter_param", None)
    stray = [n for n in names if n not in set(model.parameters) and n != scatter]
    if stray:
        raise ValueError(
            f"parameter(s) {stray} were fitted but are neither parameters of model "
            f"{model.name!r} nor read by {type(lik).__name__}, so ln L does not depend on them. "
            f"Pass the model the fit used (model=), or likelihood='gaussian_scatter' if "
            f"{stray[0]!r} is a free-scatter term.")
    if scatter is not None and scatter not in names:
        raise ValueError(
            f"{type(lik).__name__} fits a free scatter on {scatter!r}, which the fit did not "
            f"sample (it sampled {names}). Pass the likelihood the fit used (likelihood=).")
    missing = [p for p in model.parameters if p not in names]
    if missing:
        raise ValueError(f"model {model.name!r} needs {missing}, which the fit did not sample "
                         f"(it sampled {names}). Pass the model the fit used (model=).")


def _resolve_prior(model, prior, names, result=None):
    """The prior that sets the climb's coordinates: ``prior=``, else the one the fit recorded
    (``result.provenance``), else the model's default. A parameter the fit held fixed
    (``result.info["fixed"]``) is held at that value whatever the prior says."""
    from .priors import Fixed, Prior
    from .priors._numpy import family
    from .samplers.base import _recorded_prior

    if prior is None and result is not None:
        prior = _recorded_prior(result)
    if prior is None:
        prior = model.default_prior
        if prior is None:
            raise ValueError(f"model {model.name!r} has no default prior, so the box to climb in "
                             f"is unknown. Pass prior= (the Prior the fit used).")
    elif not isinstance(prior, Prior):
        prior = Prior(prior)
    held = (getattr(result, "info", None) or {}).get("fixed") if result is not None else None
    extra = {nm: float(v) for nm, v in (held or {}).items()
             if nm in names and (nm not in prior.distributions
                                 or family(prior.distributions[nm]) != "Fixed")}
    if extra:
        prior = Prior({**prior.distributions, **{nm: Fixed(v) for nm, v in extra.items()}})
    missing = [n for n in names if n not in prior.distributions]
    if missing:
        raise ValueError(f"the prior has no distribution for {missing}, which the fit sampled. "
                         f"Pass prior= (the Prior the fit used, e.g. with its scatter term).")
    return prior


def _check_data(result, lc):
    n = int(len(np.asarray(lc.time)))
    if int(result.n_data) != n:
        raise ValueError(
            f"the fit was run on {int(result.n_data)} data points but this light curve has {n}. "
            f"Pass the light curve the fit used, after the same selection steps "
            f"(select_time_window, where, ...).")
    return n


def _use_jax(backend, model, lik):
    b = str(backend).lower()
    if b not in ("auto", "jax", "cpu", "numpy"):
        raise ValueError(f"backend must be 'auto', 'jax' or 'cpu'; got {backend!r}.")
    if b in ("cpu", "numpy"):
        return False
    reason = None
    if getattr(model, "predict_jax", None) is None:
        reason = f"model {model.name!r} has no predict_jax (a CPU model, such as a redback binding)"
    else:
        try:
            import jax  # noqa: F401
        except ImportError:
            reason = "jax is not installed (the [gpu] extra)"
        else:
            from .likelihood._jax import log_likelihood_jax

            try:
                log_likelihood_jax(lik)
            except NotImplementedError as exc:
                reason = f"the JAX backend does not implement {type(lik).__name__} ({exc})"
    if reason is None:
        return True
    if b == "jax":
        raise ValueError(f"backend='jax' is not available: {reason}. Use backend='cpu' or "
                         f"'auto'.")
    return False


def _check_density(recorded, start, lik, result):
    """Warn when the fit's own max ln L is not its best draw re-scored here."""
    if not np.isfinite(recorded):
        return
    if abs(recorded - start) > max(DENSITY_MISMATCH, 1e-6 * abs(recorded)):
        how = (result.info or {}).get("max_log_likelihood_is")
        why = (" The fit recorded a sampled density, not a likelihood (a hand-built "
               "log_prob_fn)." if how == "sampled density" else "")
        warnings.warn(
            f"the fit recorded max ln L = {recorded:.6g} at its best draw, which scores "
            f"{start:.6g} under the density used here ({type(lik).__name__}, space="
            f"{lik.space!r}), so the fit was scored under another density.{why} Pass space= and "
            f"likelihood= to match it. The peak, its gain and AIC/BIC below are under the density "
            f"used here.", stacklevel=3)


# --- box coordinates ------------------------------------------------------------------------------

class _Box:
    """The climb's coordinates ``u``, one per FREE parameter (a ``Fixed`` one is held at its value,
    in ``fixed``, and never optimised):

    * a bounded prior (``Uniform``, ``LogUniform``, a ``TruncatedNormal`` with both ends finite):
      mapped linearly onto ``[0, 1]`` in its own coordinate (log10 for a LogUniform);
    * an unbounded one (``Normal``, a ``TruncatedNormal`` with an infinite end): standardised,
      ``u = (x - mu) / sigma``, bounded only where the prior is (``ulo``, ``uhi``, possibly
      infinite). ``span`` (:data:`STD_SPAN` here, 1 in a box) is the width in ``u`` the optimisers'
      step sizes and the edge test are scaled by.

    ``to_x`` maps a finite face exactly onto the prior bound, even where ``10 ** log10(bound)`` or
    ``lo + (hi - lo) * 1`` rounds past or short of it. ``names`` are the free parameters and
    ``all_names`` every parameter in the fit's order; :meth:`full` puts the fixed values back."""

    def __init__(self, prior, names):
        from .priors._numpy import family

        self.prior, self.all_names = prior, list(names)
        self.fixed, free, kinds = {}, [], []
        for nm in self.all_names:
            d = prior.distributions[nm]
            kind = family(d)
            if kind == "Fixed":
                self.fixed[nm] = float(d.value)
                continue
            if kind not in ("Uniform", "LogUniform", "Normal", "TruncatedNormal"):
                raise TypeError(
                    f"likelihood_max_opt climbs in each prior's own coordinate and cannot express "
                    f"prior {kind!r} on parameter {nm!r}. Supported: Uniform, LogUniform, "
                    f"Normal, TruncatedNormal, Fixed (held at its value).")
            free.append(nm)
            kinds.append(kind)
        if not free:
            raise ValueError(f"every parameter is Fixed ({self.fixed}), so there is nothing to "
                             f"climb. Evaluate the model at those values directly.")
        self.names = free
        dists = [prior.distributions[nm] for nm in free]
        self.is_log = np.array([k == "LogUniform" for k in kinds])
        self.lo = np.array([float(d.bounds[0]) for d in dists])
        self.hi = np.array([float(d.bounds[1]) for d in dists])
        self.is_std = ~(np.isfinite(self.lo) & np.isfinite(self.hi))
        mu = np.array([float(getattr(d, "mu", 0.0)) for d in dists])
        sigma = np.array([float(getattr(d, "sigma", 1.0)) for d in dists])
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            lo_s = np.where(self.is_log, np.log10(np.where(self.is_log, self.lo, 1.0)), self.lo)
            hi_s = np.where(self.is_log, np.log10(np.where(self.is_log, self.hi, 1.0)), self.hi)
            self.offset = np.where(self.is_std, mu, lo_s)
            self.scale = np.where(self.is_std, sigma, hi_s - lo_s)
            self.ulo = np.where(self.is_std, (self.lo - mu) / sigma, 0.0)
            self.uhi = np.where(self.is_std, (self.hi - mu) / sigma, 1.0)
        self.span = np.where(self.is_std, STD_SPAN, 1.0)
        flat = [nm for nm, w, s in zip(self.names, self.scale, self.is_std) if not s and not w > 0]
        if flat:
            raise ValueError(f"the prior box of {flat} has zero width; a fixed value is not a "
                             f"fitted parameter. Give it a Fixed(value) prior or pin it in the "
                             f"model instead.")

    def clip_u(self, u):
        return np.clip(np.asarray(u, dtype=float), self.ulo, self.uhi)

    def to_u(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):    # x <= 0 on a log axis -> nan
            s = np.where(self.is_log, np.log10(np.where(self.is_log & (x > 0), x, np.nan)), x)
        return (s - self.offset) / self.scale

    def to_x(self, u):
        u = self.clip_u(u)
        s = self.offset + self.scale * u
        # 10 ** s only on the log columns: on a linear one s can be an MJD, where it overflows
        x = np.clip(np.where(self.is_log, 10.0 ** np.where(self.is_log, s, 0.0), s),
                    self.lo, self.hi)
        # a face is the bound itself: lo + (hi - lo) * 1 can round one step short of hi
        return np.where(u <= self.ulo, self.lo, np.where(u >= self.uhi, self.hi, x))

    def dx_du(self, u):
        """d x / d u, unclipped, for the chain rule of a gradient taken in x."""
        s = self.offset + self.scale * self.clip_u(u)
        return self.scale * np.where(self.is_log,
                                     math.log(10.0) * 10.0 ** np.where(self.is_log, s, 0.0), 1.0)

    def full(self, x):
        """Free-parameter rows ``(..., k_free)`` -> every parameter ``(..., k)`` in ``all_names``
        order, each Fixed one at its value."""
        x = np.asarray(x, dtype=float)
        if not self.fixed:
            return x
        out = np.empty(x.shape[:-1] + (len(self.all_names),))
        for j, nm in enumerate(self.all_names):
            out[..., j] = self.fixed[nm] if nm in self.fixed else x[..., self.names.index(nm)]
        return out

    def at_edge(self, u):
        """Free parameters within ``EDGE_TOL`` spans of a finite face, in their own coordinate."""
        u = np.asarray(u, dtype=float)
        tol = EDGE_TOL * self.span
        with np.errstate(invalid="ignore"):
            near = ((np.isfinite(self.ulo) & (u - self.ulo < tol))
                    | (np.isfinite(self.uhi) & (self.uhi - u < tol)))
        return [nm for nm, e in zip(self.names, near) if e]

    def _ok(self, U):
        with np.errstate(invalid="ignore"):
            return np.isfinite(U) & (U >= self.ulo - BOX_SLACK) & (U <= self.uhi + BOX_SLACK)

    def checked(self, params, what, model):
        """``params`` (a dict) as a free-parameter vector inside the prior's support, refused if
        it lies outside."""
        missing = [nm for nm in self.names if nm not in (params or {})]
        if missing:
            raise ValueError(f"{what} has no value for {missing}; nothing to start from.")
        x = np.array([float(params[nm]) for nm in self.names])
        bad = [nm for nm, ok in zip(self.names, self._ok(self.to_u(x))) if not ok]
        if bad:
            nm = bad[0]
            i = self.names.index(nm)
            raise ValueError(
                f"{what} has {nm}={x[i]:g}, outside the prior box [{self.lo[i]:g}, "
                f"{self.hi[i]:g}] of model {model.name!r}, so the fit used another prior. Pass "
                f"prior= (the Prior the fit used).")
        return np.clip(x, self.lo, self.hi)

    def inside(self, X):
        return np.all(self._ok(self.to_u(X)), axis=1)


def _earlier_peaks(info, names, box):
    """Peaks already found for this fit (``info["likelihood_max_opt"]``: a
    LikelihoodMaxOptResult, its dict, or a list of either), those that name every parameter and lie
    inside the box."""
    found = info.get("likelihood_max_opt") if isinstance(info, dict) else None
    items = found if isinstance(found, (list, tuple)) else [found]
    out = []
    for it in items:
        params = it.params if isinstance(it, LikelihoodMaxOptResult) else (
            (it.get("params") or it.get("best_params")) if isinstance(it, dict) else None)
        if isinstance(params, dict) and all(nm in params for nm in names):
            x = np.array([float(params[nm]) for nm in names])
            if box.inside(x[None, :])[0]:
                out.append(np.clip(x, box.lo, box.hi))
    return out


def _candidate_pool(samples, names, box, n_candidates, seed, score_all):
    """The draws to score, before the best draw and the earlier peaks are added."""
    k = len(names)
    if samples is None or len(samples) == 0 or not all(nm in samples for nm in names) \
            or n_candidates == 0:
        return np.empty((0, k))
    X = samples[names].to_numpy(dtype=float)
    keep = box.inside(X)
    X, df = np.clip(X[keep], box.lo, box.hi), samples[keep]
    _, first = np.unique(X, axis=0, return_index=True)          # chains repeat rejected steps
    first = np.sort(first)
    X, df = X[first], df.iloc[first]
    if len(X) <= n_candidates:
        return X
    if "log_likelihood" in df:
        ll = df["log_likelihood"].to_numpy(dtype=float)
        return X[np.argsort(-np.where(np.isfinite(ll), ll, -np.inf), kind="stable")[:n_candidates]]
    rng = np.random.default_rng(int(seed))
    if score_all:
        return X if len(X) <= MAX_SCORED else X[np.sort(rng.choice(len(X), MAX_SCORED, False))]
    if "distance" in df:
        return X[np.argsort(df["distance"].to_numpy(dtype=float), kind="stable")[:n_candidates]]
    return X[np.sort(rng.choice(len(X), n_candidates, replace=False))]


# --- objectives and climbs ------------------------------------------------------------------------

@dataclass
class _Point:
    u: np.ndarray       # box coordinates
    x: np.ndarray       # parameter values, exactly those scored
    f: float            # ln L at x


class _Tracker:
    """Wraps an objective for scipy (minimising -ln L) and keeps the best point it ever saw, so a
    climb can only return a point at least as good as its start."""

    def __init__(self, obj, p):
        self.obj, self.best = obj, _Point(p.u.copy(), p.x.copy(), float(p.f))

    def _keep(self, u, x, f):
        if np.isfinite(f) and f > self.best.f:
            self.best = _Point(np.array(u, dtype=float), np.array(x, dtype=float), float(f))

    def neg(self, u):
        u = self.obj.box.clip_u(u)
        x, f = self.obj.value(u)
        self._keep(u, x, f)
        return -f if np.isfinite(f) else _BAD

    def neg_grad(self, u):
        u = self.obj.box.clip_u(u)
        x, f, g = self.obj.value_grad(u)
        self._keep(u, x, f)
        if not np.isfinite(f):
            return _BAD, np.zeros_like(u)
        return -f, -np.where(np.isfinite(g), g, 0.0)


class _Objective:
    """The shared climb: L-BFGS-B, then Nelder-Mead where it did not converge; a restart is a fresh
    Nelder-Mead simplex, then L-BFGS-B. Subclasses provide ``value``, ``score`` and, for exact
    gradients, ``value_grad``."""

    method = ""
    has_grad = False

    def __init__(self):
        self.n_evals = 0

    def lbfgsb(self, p):
        from scipy.optimize import Bounds, minimize

        tr = _Tracker(self, p)
        box = self.box
        opts = {"maxiter": LBFGS_MAXITER}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.has_grad:
                r = minimize(tr.neg_grad, p.u, jac=True, method="L-BFGS-B",
                             bounds=Bounds(box.ulo, box.uhi), options=opts)
            else:
                r = minimize(tr.neg, p.u, jac="3-point", method="L-BFGS-B",
                             bounds=Bounds(box.ulo, box.uhi), options=opts)
        return tr.best, bool(r.success)

    def nelder_mead(self, p, tol):
        from scipy.optimize import Bounds, minimize

        tr = _Tracker(self, p)
        box = self.box
        k = len(p.u)
        simplex = np.repeat(p.u[None, :], k + 1, axis=0)
        for i in range(k):
            step = NM_STEP * box.span[i]
            step = step if p.u[i] + step <= box.uhi[i] else -step
            simplex[i + 1, i] = np.clip(p.u[i] + step, box.ulo[i], box.uhi[i])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            minimize(tr.neg, p.u, method="Nelder-Mead", bounds=Bounds(box.ulo, box.uhi),
                     options={"initial_simplex": simplex, "xatol": NM_XATOL, "fatol": 0.1 * tol,
                              "maxfev": NM_MAXFEV_PER_PARAM * k})
        return tr.best

    def refine(self, p, tol):
        q, converged = self.lbfgsb(p)
        return q if converged else self.nelder_mead(q, tol)

    def restart(self, p, tol):
        return self.lbfgsb(self.nelder_mead(p, tol))[0]

    def climb(self, starts, tol):
        return [self.refine(p, tol) for p in starts]


class _CpuObjective(_Objective):
    """ln L through ``model.predict`` and the numpy likelihood: what the CPU samplers score."""

    method = ("bounded L-BFGS-B (central differences) + Nelder-Mead in box coordinates "
              "(scipy, model.predict)")

    def __init__(self, lc, model, box, names, lik):
        super().__init__()
        self.predict, self.box, self.lik = model.predict, box, lik
        self.t = np.asarray(lc.time, dtype=float)
        self.b = np.asarray(lc.band)
        self.scatter = getattr(lik, "scatter_param", None)

    def loglike(self, x):
        """ln L at the free-parameter vector ``x``, the Fixed parameters at their values."""
        self.n_evals += 1
        p = {nm: float(v) for nm, v in zip(self.box.names, x)}
        p.update(self.box.fixed)
        with np.errstate(all="ignore"):
            flux = np.asarray(self.predict(p, self.t, self.b), dtype=float)
            ll = (self.lik.log_likelihood(flux, sigma_extra=p[self.scatter]) if self.scatter
                  else self.lik.log_likelihood(flux))
        return float(ll) if np.isfinite(ll) else -np.inf

    def score(self, X):
        return np.array([self.loglike(x) for x in X], dtype=float)

    def value(self, u):
        x = self.box.to_x(u)
        return x, self.loglike(x)


#: Compiled JAX-backend programs kept between calls: key -> programs (see :func:`_jax_programs`).
_PROGRAMS = OrderedDict()


def _jax_programs(lc, model, box, names, lik):
    """The JAX backend's compiled programs for this density -> ``(programs, reused)``.

    ``programs`` holds ``score(X)``, ``value(x)`` and ``vg(X)`` on free-parameter rows in the box's
    order, and ``dtype``. An optimisation needs three compiled programs (the block scorer, the
    scalar value, the batched value and gradient): about 4 s on an arnett supernova, and 30-45 s on
    the TDE, compile included. They are built on the data-as-argument density of
    :func:`~whisper_cbpf.samplers.jax._adapters.log_density` (the same likelihood, box and
    constraint wall as the JAX samplers), so every light curve fitted with the same forward model
    (one model object, as in :func:`whisper_cbpf.fit_batch`), prior families, likelihood and bucket
    reuses them. :func:`whisper_cbpf.compare` binds a new model to each alert (its explosion-time
    reference and redshift are part of it), so there each alert still compiles its own. They are
    kept, up to :data:`PROGRAM_CACHE_SIZE`, by the identity of that shared density, the bucket and
    the precision. A model ``log_density`` cannot take (one carrying its own ``log_prob_jax``)
    falls back to programs closed over this light curve, cached per model, data, prior over the
    fitted parameters, likelihood and precision.
    """
    from .samplers.jax._adapters import float_dtype, log_density

    dtype = float_dtype()
    try:
        ld = log_density(lc, model, space=lik.space, likelihood=lik, prior=box.prior)
    except (ValueError, NotImplementedError, TypeError):
        ld = None
    if ld is not None and sorted(ld.names) == sorted(box.names):
        return _shared_programs(ld, box, dtype)
    return _closed_programs(lc, model, box, names, lik, dtype)


def _cache(key, build):
    """``(entry, reused)`` from :data:`_PROGRAMS`; ``build()`` makes and stores a missing one."""
    entry = _PROGRAMS.get(key) if key is not None else None
    if entry is not None:
        _PROGRAMS.move_to_end(key)
        return entry, True
    entry = build()
    if key is not None:
        _PROGRAMS[key] = entry
        while len(_PROGRAMS) > PROGRAM_CACHE_SIZE:
            _PROGRAMS.popitem(last=False)
    return entry, False


def _blocks(batched, rows, width):
    """``batched`` over ``rows`` in blocks of exactly ``width`` (the last padded): one compile."""
    n = rows.shape[0]
    pad = (-n) % width
    if pad:
        rows = np.concatenate([rows, np.repeat(rows[-1:], pad, axis=0)])
    out = [batched(rows[i:i + width]) for i in range(0, rows.shape[0], width)]
    return np.concatenate([np.asarray(o, dtype=float) for o in out])[:n]


def _shared_programs(ld, box, dtype):
    import jax
    import jax.numpy as jnp

    from .samplers.jax._diagnostics import SCORE_BLOCK

    shared = ld.shared

    def build():
        def loglik(theta, data):
            return shared(theta, data)[1]
        return {"shared": shared,          # held, so its id cannot be reused while cached
                "batch": jax.jit(jax.vmap(loglik, in_axes=(0, None))),
                "value": jax.jit(loglik),
                "vg": jax.jit(jax.vmap(jax.value_and_grad(loglik), in_axes=(0, None)))}

    # one jitted `shared` serves every bucket, but XLA compiles each bucket's shapes: key on both
    entry, reused = _cache(("shared", id(shared), int(ld.bucket), np.dtype(dtype).name), build)
    perm = np.array([box.names.index(nm) for nm in ld.names], dtype=int)   # box -> density order
    data = ld.data

    def theta(X):
        return jnp.asarray(np.asarray(X, dtype=float)[..., perm], dtype=dtype)

    def vg(X):
        f, g = entry["vg"](theta(X), data)
        g = np.asarray(g, dtype=float)
        out = np.empty_like(g)
        out[:, perm] = g
        return np.asarray(f, dtype=float), out

    return {"dtype": dtype, "vg": vg,
            "score": lambda X: _blocks(lambda R: entry["batch"](theta(R), data),
                                       np.asarray(X, dtype=float), SCORE_BLOCK),
            "value": lambda x: float(entry["value"](theta(x), data))}, reused


def _closed_programs(lc, model, box, names, lik, dtype):
    import jax
    import jax.numpy as jnp

    from .samplers.jax._adapters import make_log_prob_jax
    from .samplers.jax._diagnostics import block_scorer

    try:
        from .results import _hash, data_hash, prior_record

        sub = type(box.prior)({nm: box.prior.distributions[nm] for nm in names})
        key = ("closed", data_hash(lc), id(model), tuple(names), _hash(prior_record(sub)),
               _hash(lik), np.dtype(dtype).name)
    except Exception:                                     # noqa: BLE001 - uncacheable, not wrong
        key = None

    def build():
        dens = make_log_prob_jax(lc, model, box.prior, space=lik.space, names=names,
                                 include_prior=False, likelihood=lik, jit=False)
        return {"model": model, "score": block_scorer(dens, dtype), "value": jax.jit(dens),
                "vg": jax.jit(jax.vmap(jax.value_and_grad(dens)))}

    entry, reused = _cache(key, build)
    if reused and entry["model"] is not model:            # a model object re-created in place
        _PROGRAMS.pop(key, None)
        entry, reused = _cache(key, build)
    free = np.array([list(names).index(nm) for nm in box.names], dtype=int)

    def vg(X):
        f, g = entry["vg"](jnp.asarray(box.full(X), dtype=dtype))
        return np.asarray(f, dtype=float), np.asarray(g, dtype=float)[:, free]

    return {"dtype": dtype, "vg": vg,
            "score": lambda X: entry["score"](box.full(np.asarray(X, dtype=float))),
            "value": lambda x: float(entry["value"](jnp.asarray(box.full(x), dtype=dtype)))}, \
        reused


class _JaxObjective(_Objective):
    """The JAX samplers' density (``make_log_prob_jax``: box, constraint wall, same likelihood):
    batched scoring, exact gradients, and a projected Adam pass over all starts at once."""

    method = ("projected Adam + bounded L-BFGS-B (exact gradients) + Nelder-Mead in box "
              "coordinates (JAX density)")
    has_grad = True

    def __init__(self, lc, model, box, names, lik):
        super().__init__()
        progs, reused = _jax_programs(lc, model, box, names, lik)
        self.box, self.dtype = box, progs["dtype"]
        # every program takes the FREE parameters, in the box's order
        self._score, self._value, self._vg = progs["score"], progs["value"], progs["vg"]
        self._width = VG_WIDTH
        if reused:
            self.method = self.method + "; compiled programs reused from an earlier optimisation"

    def _rows(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        pad = self._width - n
        if pad > 0:
            X = np.concatenate([X, np.repeat(X[-1:], pad, axis=0)])
        f, g = self._vg(X)
        return f[:n], g[:n]

    def score(self, X):
        self.n_evals += len(X)
        return self._score(np.asarray(X, dtype=float))

    def value(self, u):
        self.n_evals += 1
        x = self.box.to_x(u)
        v = self._value(x)
        return x, (v if np.isfinite(v) else -np.inf)

    def value_grad(self, u):
        self.n_evals += 1
        x = self.box.to_x(u)
        f, g = self._rows(x[None, :])
        f0 = float(f[0])
        return x, (f0 if np.isfinite(f0) else -np.inf), g[0] * self.box.dx_du(u)

    def adam(self, starts):
        """Projected Adam in box coordinates, every start in one batch: each step clips u into
        [0, 1], so a parameter can sit exactly on its edge, and each start keeps its best point."""
        box = self.box
        U = np.array([p.u for p in starts], dtype=float)
        best = [_Point(p.u.copy(), p.x.copy(), float(p.f)) for p in starts]
        m, v = np.zeros_like(U), np.zeros_like(U)
        lr0, lr1 = ADAM_LR
        for i in range(ADAM_STEPS + 1):
            X = box.to_x(U)
            f, gx = self._rows(X)
            self.n_evals += len(U)
            for j in np.nonzero(np.isfinite(f) & (f > np.array([b.f for b in best])))[0]:
                best[j] = _Point(U[j].copy(), X[j].copy(), float(f[j]))
            if i == ADAM_STEPS:
                break
            g = -gx * box.dx_du(U)
            g = np.where(np.isfinite(g), g, 0.0)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh, vh = m / (1.0 - 0.9 ** (i + 1)), v / (1.0 - 0.999 ** (i + 1))
            lr = lr1 + 0.5 * (lr0 - lr1) * (1.0 + math.cos(math.pi * i / ADAM_STEPS))
            U = box.clip_u(U - lr * box.span * mh / (np.sqrt(vh) + 1e-8))
        return best

    def climb(self, starts, tol):
        self._width = VG_WIDTH * -(-len(starts) // VG_WIDTH)
        return [self.refine(p, tol) for p in self.adam(starts)]
