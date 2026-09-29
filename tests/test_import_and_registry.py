"""Import, registration and the whisper contract.

These are the tests that fail loudly if the JAX/GPU half has drifted from the core's extension API
-- the seam where two formerly separate packages were joined, and so the thing most likely to break.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import whisper_cbpf as wp
import whisper_cbpf
from whisper_cbpf.samplers import get_sampler, list_samplers


def test_registers_into_whispers_own_registries():
    # not a parallel registry -- the GPU samplers must show up in whisper's own listing
    assert "nuts_gpu" in list_samplers()
    assert "flare_jax" in wp.list_models()


def test_sampler_constructs_with_zero_arguments():
    pytest.importorskip("jax")   # nuts_gpu imports jax at module scope
    # whisper's get_sampler does _SAMPLERS[name]() -- a sampler that needs constructor arguments
    # is unreachable through the registry, so all config must go through fit(**kwargs).
    s = get_sampler("nuts_gpu")
    assert s.__class__.__name__ == "NUTSGPUSampler"
    assert hasattr(s, "fit")


def test_reimport_is_idempotent():
    # register_sampler raises on a duplicate name unless overwrite=True; a reload must not explode
    importlib.reload(whisper_cbpf)
    importlib.reload(whisper_cbpf)
    assert "nuts_gpu" in list_samplers()


def test_flare_model_satisfies_the_predict_contract():
    pytest.importorskip("jax")   # flare_jax's deferred predict resolves the JAX module
    m = wp.get_model("flare_jax")
    t = np.linspace(0.0, 30.0, 25)
    f = m.predict({"log_amp": 1.0, "log_sigma": 0.3, "log_tau": 1.2, "t0": 12.0}, t, None)
    assert isinstance(f, np.ndarray) and f.shape == t.shape
    assert np.all(np.isfinite(f)) and np.all(f >= 0.0)
    assert f.max() > 0.0


def test_kilonova_is_not_auto_registered_and_says_why():
    # It is photometric: it needs a filter set, redshift and distance that predict() cannot carry.
    # Auto-registering with a guessed distance would silently rescale every fitted mass.
    assert "kilonova_one_jax" not in wp.list_models()
    assert callable(whisper_cbpf.kilonova_model)
    assert callable(whisper_cbpf.register_kilonova)


def test_kilonova_factory_uses_redback_default_priors():
    pytest.importorskip("jax")   # builds a JAX model
    kn = whisper_cbpf.kilonova_model(["sdssg", "sdssr"], redshift=0.01, dl_cm=1.3e26, n_wave=400)
    b = {k: v.bounds for k, v in kn.default_prior.distributions.items()}
    # verbatim from redback/priors/one_component_kilonova_model.prior
    assert b["mej"] == pytest.approx((1e-2, 0.05))
    assert b["vej"] == pytest.approx((0.1, 0.5))
    assert b["kappa"] == pytest.approx((1.0, 30.0))
    assert b["temperature_floor"] == pytest.approx((100.0, 6000.0))
    assert type(kn.default_prior.distributions["temperature_floor"]).__name__ == "LogUniform"


def test_kilonova_predict_returns_flux_and_rejects_unbound_bands():
    pytest.importorskip("jax")   # builds a JAX model
    kn = whisper_cbpf.kilonova_model(["sdssg", "sdssr"], redshift=0.01, dl_cm=1.3e26, n_wave=400)
    t = np.array([1.0, 3.0, 7.0, 15.0])
    p = {"mej": 0.03, "vej": 0.2, "kappa": 1.0, "temperature_floor": 3000.0}
    f = kn.predict(p, t, np.array(["sdssg", "sdssr", "sdssg", "sdssr"]))
    assert np.all(np.isfinite(f)) and np.all(f > 0.0)          # kilonova flux is positive by construction
    mag = -2.5 * np.log10(f / 3631.0)
    assert np.all((10.0 < mag) & (mag < 40.0)), mag            # physically plausible, not saturated
    assert np.all(np.diff(mag[::2]) > 0)                        # g-band fades between 1 d and 7 d
    with pytest.raises(KeyError, match="not bound"):
        kn.predict(p, t, np.array(["sdssg", "sdssr", "sdssg", "lsstz"]))


def test_tde_is_not_auto_registered_and_says_why():
    # Same reasoning as the kilonova: photometric, so predict() cannot carry the filter set,
    # redshift and distance it needs. It additionally requires float64, which a zero-argument
    # registry factory could not communicate either.
    assert "tde_gaussianrise_jax" not in wp.list_models()
    assert callable(whisper_cbpf.tde_model)
    assert callable(whisper_cbpf.register_tde)


def test_tde_factory_names_itself_after_the_rise_it_was_given():
    pytest.importorskip("jax")   # builds a JAX model
    a = whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200)
    b = whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200, rise="none")
    assert a.name == "tde_gaussianrise_jax" and b.name == "tde_cooling_envelope_jax"
    # The two differ by more than the rise, because redback's two prior files differ: the
    # gaussianrise one pins nothing, the cooling_envelope one pins four up to redback 1.15. Compare
    # the FULL parameter sets by freeing everything.
    from whisper_cbpf.priors import LogUniform, Prior, Uniform
    free = {k: None for k in ("mbh_6", "eta", "alpha", "beta")}
    # freeing a delta-pinned parameter needs a distribution: redback's file has none for them
    pr = Prior({"mbh_6": LogUniform(0.1, 20.0), "eta": LogUniform(1e-4, 0.1),
                "alpha": LogUniform(0.1, 1.0), "beta": Uniform(1.0, 5.0)})
    a2 = whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200,
                               pin=free, prior=pr)
    b2 = whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200,
                               rise="none", pin=free, prior=pr)
    assert set(a2.parameters) - set(b2.parameters) == {"peak_time", "sigma_t"}
    assert b2.parameters == ["mbh_6", "stellar_mass", "eta", "alpha", "beta"]
    with pytest.raises(ValueError, match="rise must be"):
        whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, rise="linear")


def test_tde_factory_uses_redbacks_prior_including_what_redback_pins():
    pytest.importorskip("jax")   # builds a JAX model
    """redback's prior, read from redback -- deltas and all -- not a transcription of it.

    `gaussianrise_cooling_envelope.prior` pins nothing (1.15.1, 1.20), so all seven are free.
    redback <= 1.15's `cooling_envelope.prior` pins four (bilby parses its trailing scalar
    assignments as DeltaFunction), leaving stellar_mass the only free parameter; 1.18 freed
    them, so under 1.20 all five are free. whisper has no delta prior, so a pinned parameter
    is bound at the factory and absent from `parameters`. (This test used to assert 1.15's
    answer whatever redback was installed, and failed under 1.20 for that reason alone.)
    """
    m = whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200)
    b = {k: v.bounds for k, v in m.default_prior.distributions.items()}
    assert set(m.parameters) == set(b), "every free parameter needs a prior"
    assert b["stellar_mass"] == pytest.approx((0.1, 10.0))
    assert b["sigma_t"] == pytest.approx((10.0, 60.0))
    assert type(m.default_prior.distributions["beta"]).__name__ == "Uniform"

    bare = whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200, rise="none")
    pytest.importorskip("redback")
    from whisper_cbpf.models.redback_adapter import installed_redback_preset
    pins = installed_redback_preset() in ("1.12", "1.15")
    # redback's own file decides how many are free here; reproducing that is the point
    if pins:
        assert bare.parameters == ["stellar_mass"], bare.parameters
        assert "redback pins" in bare.description
    else:
        assert bare.parameters == ["mbh_6", "stellar_mass", "eta", "alpha", "beta"]
        assert "redback pins" not in bare.description
        assert bare.default_prior.distributions["stellar_mass"].bounds == pytest.approx((0.5, 10.0))

    # ... and `pin` overrides in both directions
    from whisper_cbpf.priors import LogUniform, Prior, Uniform
    freed = whisper_cbpf.tde_model(
        ["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200, rise="none",
        pin={"beta": None, "mbh_6": None},
        prior=Prior({"beta": Uniform(1.0, 5.0), "mbh_6": LogUniform(0.1, 20.0)}))
    assert set(freed.parameters) >= {"stellar_mass", "beta", "mbh_6"}
    assert "beta" not in whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200,
                                                rise="none", pin={"beta": 2.0}).parameters
    # ... and unpinning WITHOUT a distribution must fail loudly, not invent a range (only a
    # release whose file pins something has such a parameter)
    if pins:
        with pytest.raises(ValueError, match="no prior"):
            whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200,
                                  rise="none", pin={"beta": None})
    fixed = whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200,
                                  pin={"sigma_t": 15.0})
    assert "sigma_t" not in fixed.parameters
    with pytest.raises(ValueError, match="not a parameter"):
        whisper_cbpf.tde_model(["sdssg"], redshift=0.05, dl_cm=7e26, n_wave=200,
                              pin={"nonsense": 1.0})


def test_tde_predict_returns_flux_and_rejects_unbound_bands():
    jax = pytest.importorskip("jax")
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)          # the engine refuses float32 by design
    try:
        m = whisper_cbpf.tde_model(["sdssg", "sdssr"], redshift=0.05, dl_cm=7e26,
                                  n_wave=300, n_time=200)
        p = {"peak_time": 20.0, "sigma_t": 15.0, "mbh_6": 1.0, "stellar_mass": 1.0,
             "eta": 0.05, "alpha": 0.1, "beta": 1.0}
        p = {k: p[k] for k in m.parameters}
        t = np.linspace(1.0, 150.0, 8)
        f = m.predict(p, t, np.array(["sdssg", "sdssr"] * 4))
        assert isinstance(f, np.ndarray) and f.shape == t.shape
        assert np.all(np.isfinite(f)) and np.all(f > 0.0)     # blackbody flux is positive
        mag = -2.5 * np.log10(f / 3631.0)
        assert np.all((5.0 < mag) & (mag < 40.0)), mag
        with pytest.raises(KeyError, match="not bound"):
            m.predict(p, t, np.array(["sdssg", "lsstz"] * 4))
        with pytest.raises(ValueError, match="needs a `bands`"):
            m.predict(p, t, None)
    finally:
        jax.config.update("jax_enable_x64", was)


def test_env_helpers_never_import_jax_at_module_scope():
    # importing whisper_cbpf on a CPU-only box must work, so `wp.list_samplers()` still tells the
    # user what exists; the GPU requirement belongs at fit() time with an actionable message.
    import ast
    import inspect

    from whisper_cbpf.backends import _env

    # AST, not a substring search: the module docstring legitimately contains the words
    # "import jax" while describing when NOT to do it.
    tree = ast.parse(inspect.getsource(_env))
    top_level = []
    for node in tree.body:                       # module scope only; function bodies may import
        if isinstance(node, ast.Import):
            top_level += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_level.append(node.module or "")
    assert not [m for m in top_level if m.split(".")[0] in ("jax", "torch", "numpyro")], top_level

    rep = _env.env_report()
    assert set(_env.REQUIRED_ENV).issubset(rep)


def test_env_script_names_the_file_that_actually_exists():
    """The path in every "you are on CPU, here is the fix" message must be real AND installed.

    ``packages.find`` ships only ``whisper_cbpf*``, so a bootstrap living anywhere else is absent
    from a pip install -- and its absence has no symptom, just JAX quietly on the CPU at ~50x. So
    the assertion is the strong one: the file exists, and it is inside the package directory.
    """
    import os

    from whisper_cbpf.backends import _env

    path = _env.env_script()
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(_env.__file__)))

    assert os.path.isabs(path)
    assert os.path.basename(path) == "env.sh"
    assert os.path.exists(path), f"the GPU bootstrap is missing at {path}"
    assert os.path.commonpath([path, pkg_root]) == pkg_root, (
        f"{path} is outside the package, so `pip install` will not ship it")
    assert _env.env_script_hint() == f"source {path}"
    assert _env.env_report()["env_script_exists"] is True
