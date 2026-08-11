# Extending Whisper

Whisper has **four pluggable axes** — **models**, **samplers**, **likelihoods** and **distances**. Each
is a tiny name registry with matching `register_*` / `list_*` helpers, so you can add your own without
touching the core.

## Add a model

A model maps parameters to predicted flux: `predict(params: dict, times, bands) -> np.ndarray`.
Keep it **vectorized** (no per-point loops), and for **parallel** ABC define it at **module level**
(closures/lambdas only work with `n_jobs=1`, because parallel workers pickle the function).

```python
import numpy as np
import whisper_labia as wp

def powerlaw(params, times, bands=None):
    return params["amplitude"] * np.clip(times, 1e-6, None) ** (-params["index"])

wp.register_model(
    "powerlaw", powerlaw, ["amplitude", "index"],
    prior=wp.Prior({"amplitude": wp.Uniform(0, 10), "index": wp.Uniform(0, 3)}),
    description="Simple power-law decay.",
)

print("powerlaw" in wp.list_models())     # True
res = wp.fit_ABC(lc, "powerlaw")          # usable by name with any sampler
```

`bands` is passed for generality; ignore it for band-independent models. See
`whisper_labia/models/{flare,bazin,gaussian_rise}.py` for references.

**To ship a model as a module** (the convention every built-in follows, and what
`models/__init__.py` feeds to `register_model`), expose these four names:

```python
PARAMETERS  = ["amplitude", "index"]      # ordered -> fixes every posterior's column order
DESCRIPTION = "Simple power-law decay."   # one line, shown by list_models()/Model.description
PRIOR       = wp.Prior({...})             # becomes Model.default_prior
def powerlaw_flux(params, times, bands=None): ...   # the predict callable, returns flux in Jy
```

then register it once at import. Keeping them at module level means a user can widen a built-in prior
without retyping its bounds — `from whisper_labia.models import flare; flare.PRIOR`.

## Add a sampler

Subclass `BaseSampler`, implement `fit(lc, model, prior=None, **kwargs) -> SamplerResult`, and register
it. Reuse the building blocks: `get_model`, the model's `predict`, the prior's `sample` / `log_prob` /
`rescale`, a distance (or your own likelihood), and `summarize_posterior`.

```python
import time
import numpy as np
import pandas as pd
import whisper_labia as wp
from whisper_labia.samplers import register_sampler
from whisper_labia.samplers.base import BaseSampler, SamplerResult, summarize_posterior
from whisper_labia.models import get_model
from whisper_labia.distance import chi2_distance

class RandomSearch(BaseSampler):
    name = "random_search"

    def fit(self, lc, model, prior=None, *, n=2000, keep=0.05, seed=0, **kw):
        model = get_model(model)
        prior = prior or model.default_prior
        lc = lc.add_flux()
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        rows = []
        for _ in range(n):
            theta = prior.sample(rng)
            sim = model.predict(theta, lc.time, lc.band)
            theta["distance"] = chi2_distance(lc.flux, lc.flux_err, sim)
            rows.append(theta)

        df = pd.DataFrame(rows).nsmallest(int(n * keep), "distance")
        chi2_min = float(df["distance"].min())
        k, ndata = len(model.parameters), lc.n_points
        best = df.iloc[0][model.parameters].to_dict()
        return SamplerResult(
            sampler=self.name, model=model.name, parameters=model.parameters,
            samples=df, summary=summarize_posterior(df, model.parameters),
            best_params={p: float(v) for p, v in best.items()},
            n_data=ndata, n_params=k, runtime_s=time.perf_counter() - t0, info={"n": n},
            min_distance=chi2_min, max_log_likelihood=-0.5 * chi2_min,
            aic=chi2_min + 2 * k, bic=chi2_min + k * np.log(ndata),
        )

register_sampler("random_search", RandomSearch)
res = wp.fit(lc, "flare", sampler="random_search", n=5000)   # dispatch by name
```

The full ABC and ABC-SMC implementations live in `whisper_labia/samplers/abc.py` and `abc_smc.py`.

## Add a likelihood

A likelihood compares the model's predicted **flux** to the data in a chosen **space**. The contract is
a class taking `(lc, space="auto", **kwargs)` that exposes `log_likelihood(model_flux) -> float`.
Subclass `GaussianLikelihood` for the easy path: you inherit the space resolution (`"auto"` → magnitude
space for magnitude data, flux otherwise), the `self.y` / `self.sigma` arrays it prepares, and
`model_in_space()`, which maps model flux into the comparison space.

Also override **`log_likelihood_pointwise(model_flux) -> array`** (per-point terms summing to
`log_likelihood`) — WAIC, PSIS-LOO and the whole `predictive_metrics` block are built on it, so without
it your likelihood fits but scores `None`. `summary()` is what lands in the result JSON.

```python
import numpy as np
import whisper_labia as wp
from whisper_labia.likelihood import GaussianLikelihood, register_likelihood

class StudentTLikelihood(GaussianLikelihood):
    """Heavy-tailed Student-t: outlier-tolerant without a mixture component."""

    def __init__(self, lc, space="auto", nu=4.0, **kw):
        super().__init__(lc, space=space, **kw)
        self.nu = float(nu)

    def log_likelihood_pointwise(self, model_flux):
        from scipy.special import gammaln
        res = (self.y - self.model_in_space(model_flux)) / self.sigma
        return (gammaln(0.5 * (self.nu + 1)) - gammaln(0.5 * self.nu)
                - 0.5 * np.log(np.pi * self.nu) - np.log(self.sigma)
                - 0.5 * (self.nu + 1) * np.log1p(res * res / self.nu))

    def log_likelihood(self, model_flux):
        return float(np.sum(self.log_likelihood_pointwise(model_flux)))

    def summary(self):
        return {"likelihood": "student_t", "space": self.space, "nu": self.nu,
                "n_data": int(self.y.size)}

register_likelihood("student_t", StudentTLikelihood)

print("student_t" in wp.list_likelihoods())                    # True
res = wp.fit_MCMC(lc, "flare", likelihood="student_t")         # usable by name
lik = wp.make_likelihood(lc, kind="student_t", space="magnitude", nu=3.0)   # or build directly
```

Registered names are matched case-insensitively; pass `overwrite=True` to replace an existing one.
Likelihoods must be **picklable** for parallel MCMC (`n_jobs>1`) — module-level classes, no closures.
The samplers that use a likelihood are MCMC (via `likelihood=`) and SNPE/ABC for their exact
`max_log_likelihood`/AIC/BIC; distance-based ABC acceptance uses a *distance*, below.

## Add a distance

A distance is any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float`. Pass it as `distance=` to a
sampler:

```python
def mae_distance(obs, err, sim, bands=None):
    return float(np.sum(np.abs((obs - sim) / err)))

wp.fit_ABC(lc, "flare", distance=mae_distance)
```

Register it to select it by **name** (case-insensitive), so it can be driven from a config or a script
argument instead of an import:

```python
wp.register_distance("mae", mae_distance)
print(wp.list_distances())                        # ['chi2', 'chi_square', 'mae']
wp.fit_ABC(lc, "flare", distance="mae", n_jobs=8)
```

Registering does **not** exempt it from pickling: the sampler resolves the name to the callable before
starting the process pool, so for `n_jobs > 1` the function must still be module-level (not a closure
or lambda) — same rule as `predict`.
