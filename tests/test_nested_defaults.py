"""Nested sampler (dynesty): the defaults that decide whether a run finishes at all.

Three guards, each for a failure that looked like a hang rather than an error:

1. ``sample="auto"`` must not resolve to dynesty's ``'unif'``. dynesty's own ``'auto'`` picks
   ``'unif'`` below 10 parameters, and ``'unif'`` redraws inside a bounding ellipsoid until one
   point beats the likelihood threshold -- an unbounded loop that collapsed to 0.65% efficiency and
   stalled for 25+ minutes on redback ``arnett`` (6 free parameters) fit to SN2025pgp.
2. dynesty's efficiency is in ``info``, and a collapsed ``'unif'`` run says so.
3. ``n_jobs > 1`` workers are *spawned*, not forked: a parent that has started JAX's XLA runtime
   forks children whose inherited mutexes stay locked (see ``samplers/abc.py``'s ``_MP_CONTEXT``).
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.models.flare import flare_flux
from whisper_cbpf.priors import Prior, Uniform
from whisper_cbpf.samplers import nested as nested_module

# --- a thin Gaussian shell: the textbook case where ellipsoid ('unif') sampling collapses ---------
# Module level (not a closure) so the model is picklable, per whisper_cbpf.models' rule.
_SHELL_D, _SHELL_R, _SHELL_W = 4, 0.3, 1e-3


def _shell_predict(parameters, times, bands=None):
    """Distance from the unit cube's centre; one datum at ``_SHELL_R`` makes the likelihood a shell
    of radius 0.3 and width 1e-3 -- a region no ellipsoid bounds tightly."""
    x = np.array([parameters[f"x{i}"] for i in range(_SHELL_D)])
    return np.full(np.shape(times), np.sqrt(np.sum((x - 0.5) ** 2)))


def _shell():
    names = [f"x{i}" for i in range(_SHELL_D)]
    wp.register_model("_nested_test_shell", _shell_predict, names, overwrite=True)
    prior = Prior({nm: Uniform(0.0, 1.0) for nm in names})
    lc = wp.LightCurve(time=[1.0], band=["r"], flux=[_SHELL_R], flux_err=[_SHELL_W])
    return lc, prior


# --- 1. 'auto' resolves to rwalk, and info records the method that ran ---------------------------
def test_auto_resolves_to_rwalk_for_a_redback_model(monkeypatch):
    """The common alert shape: redback ``arnett`` with the redshift pinned has 6 free parameters, where
    dynesty's own ``'auto'`` would pick ``'unif'``. Checked on what dynesty is actually handed, not
    only on the label in ``info``. ``maxiter=5``: the method is chosen before the first iteration,
    so a converged run would only cost time."""
    pytest.importorskip("redback")
    import dynesty

    model = wp.register_redback("arnett", band_names=["ztfg", "ztfr", "ztfi"], redshift=0.051,
                                name="_nested_test_arnett", overwrite=True)
    t = np.linspace(2.0, 40.0, 12)
    b = np.array(["ztfg", "ztfr", "ztfi"] * 4)
    truth = dict(f_nickel=0.1, mej=2.0, vej=1e4, kappa=0.1, kappa_gamma=0.03,
                 temperature_floor=4000.0)
    flux = np.asarray(model.predict(truth, t, b), dtype=float)
    lc = wp.LightCurve(time=t, band=b, flux=flux, flux_err=0.05 * flux)

    handed = []
    real = dynesty.NestedSampler

    def spy(*args, **kwargs):
        handed.append(kwargs["sample"])
        return real(*args, **kwargs)

    monkeypatch.setattr(dynesty, "NestedSampler", spy)
    with pytest.warns(UserWarning, match="truncated"):
        r = wp.fit_nested(lc, "_nested_test_arnett", nlive=30, maxiter=5, seed=0)
    assert len(r.parameters) == 6                      # dynesty's 'auto' -> 'unif' below 10
    assert handed == ["rwalk"]
    assert r.info["sample"] == "rwalk"


def test_explicit_unif_is_still_honoured():
    lc, prior = _shell()
    r = wp.fit_nested(lc, "_nested_test_shell", prior=prior, nlive=25, sample="unif", maxiter=20,
                      seed=0)
    assert r.info["sample"] == "unif"


# --- 2. efficiency is reported; a collapsed 'unif' run warns, the default converges --------------
def test_unif_collapse_warns_and_the_default_converges():
    """Same 4-parameter shell, same ``nlive=25``. Measured over seeds 0-4: ``'unif'`` capped at
    ``maxcall=30000`` ends at 0.73-0.85% efficiency (at seed 0 its last 50 iterations cost ~500
    calls each) without converging; the default ``'rwalk'`` converges at 5.5-5.9% in ~5 000
    calls. The ``'unif'`` run is what the old default did: dynesty's own ``'auto'`` is ``'unif'``
    at 4 parameters."""
    lc, prior = _shell()
    with pytest.warns(UserWarning, match="efficiency"):
        ru = wp.fit_nested(lc, "_nested_test_shell", prior=prior, nlive=25, sample="unif",
                           maxcall=30000, seed=0)
    assert ru.info["efficiency_percent"] < 1.0
    assert ru.info["converged"] is False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rd = wp.fit_nested(lc, "_nested_test_shell", prior=prior, nlive=25, seed=0)
    assert rd.info["sample"] == "rwalk" and rd.info["converged"] is True
    assert rd.info["efficiency_percent"] > 1.0
    assert not [w for w in caught if "efficiency" in str(w.message)]


# --- 3. n_jobs > 1 workers are spawned -----------------------------------------------------------
def test_parallel_workers_are_spawned_not_forked(monkeypatch):
    """Reads the start method off the context the pool actually holds, not off wall time."""
    methods = []

    class SpyPool(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            methods.append(self._mp_context.get_start_method())

    monkeypatch.setattr(nested_module, "ProcessPoolExecutor", SpyPool)
    t = np.linspace(0.5, 30, 30)
    flux = flare_flux({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0}, t, None)
    lc = wp.LightCurve(time=t, band=["r"] * 30, flux=flux, flux_err=np.full_like(flux, 0.1))
    r = wp.fit_nested(lc, "flare", nlive=50, dlogz=1.0, seed=0, n_jobs=2)
    assert methods == ["spawn"]
    assert r.info["n_jobs"] == 2 and np.isfinite(r.info["log_evidence"])
