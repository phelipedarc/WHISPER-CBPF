"""SNPE / NPE sampler (sbi). Training tests skip cleanly when the [sbi] extra is absent."""
import numpy as np
import pytest

import whisper_cbpf as wp
from whisper_cbpf.models import get_model


def test_snpe_registered_without_sbi():
    """Registry + import work even without sbi installed (sbi is imported lazily)."""
    assert "snpe" in wp.list_samplers()
    assert "npe" in wp.list_samplers()


def test_snpe_invalid_proposal_mode():
    """proposal_mode is validated up-front (before any sbi import) -> works without the extra."""
    lc = wp.LightCurve(time=[1.0, 2.0], band=["r", "r"], flux=[1.0, 0.5], flux_err=[0.1, 0.1])
    prior = wp.Prior({"amplitude": wp.Uniform(0, 5), "rise_time": wp.Uniform(0.1, 5),
                      "decay_time": wp.Uniform(0.5, 10)})
    with pytest.raises(ValueError, match="proposal_mode"):
        wp.fit_SNPE(lc, "flare", prior=prior, proposal_mode="bogus")


# Everything below needs the optional [sbi] extra (sbi + torch).
sbi = pytest.importorskip("sbi")
from whisper_cbpf.samplers.snpe import (  # noqa: E402
    _build_density_estimator,
    _require_sbi,
    _to_torch_prior,
)


def test_build_density_estimator_dispatch():
    import torch.nn as nn
    # plain string with no extras -> passed straight to NPE
    assert _build_density_estimator("maf", None, None, None, None) == "maf"
    # an already-built factory (callable) -> used as-is
    def fake_builder(*a, **k):
        return None
    assert _build_density_estimator(fake_builder, None, None, None, None) is fake_builder
    # string + embedding/hyperparameters -> wrapped via posterior_nn (a callable, not the string)
    built = _build_density_estimator("nsf", nn.Identity(), 16, 2, 4)
    assert callable(built) and built != "nsf"


def test_torch_prior_adapter_uniform_and_loguniform():
    sb = _require_sbi()
    # all-Uniform -> BoxUniform
    tp = _to_torch_prior(wp.Prior({"a": wp.Uniform(0, 2), "b": wp.Uniform(-1, 1)}), sb)
    s = tp.sample((200,))
    assert tuple(s.shape) == (200, 2)
    assert float(s[:, 0].min()) >= 0 and float(s[:, 0].max()) <= 2
    assert float(s[:, 1].min()) >= -1 and float(s[:, 1].max()) <= 1
    # mixed Uniform + LogUniform -> MultipleIndependent; log-param stays within (positive) bounds
    tp2 = _to_torch_prior(wp.Prior({"a": wp.Uniform(0, 1), "tau": wp.LogUniform(1, 100)}), sb)
    s2 = tp2.sample((200,))
    assert tuple(s2.shape) == (200, 2)
    assert float(s2[:, 1].min()) >= 1 and float(s2[:, 1].max()) <= 100


def test_torch_prior_loguniform_support_is_bounded():
    """A LogUniform component must advertise ``[low, high]``, not the ExpTransform codomain.

    Regression test. sbi's ``within_support`` -- the accept/reject test inside
    ``DirectPosterior.sample()`` -- prefers ``support.check(theta)`` over ``log_prob(theta)``. A bare
    ``TransformedDistribution(Uniform(log a, log b), ExpTransform())`` reports ``GreaterThan(0)`` as
    its support, so that check passes for ANY positive value and out-of-prior draws land in the final
    posterior (28-74% of them on the AT2017gfo kilonova arms). ``log_prob`` was never the problem, so
    this asserts the two tests now agree -- and that the density itself did not move.
    """
    import torch
    from sbi.utils.sbiutils import within_support

    sb = _require_sbi()
    prior = wp.Prior({"a": wp.Uniform(0.0, 1.0), "tau": wp.LogUniform(1.0, 100.0)})
    tp, _, _ = sb.process_prior(_to_torch_prior(prior, sb))

    tau = tp.dists[1]
    assert float(tau.support.lower_bound) == 1.0 and float(tau.support.upper_bound) == 100.0

    inside = torch.tensor([[0.5, 10.0]])
    outside = torch.tensor([[0.5, 500.0], [0.5, 0.01]])   # 5x above / 100x below tau's bounds
    assert bool(within_support(tp, inside).all())
    assert not bool(within_support(tp, outside).any())    # the leak: these used to come back True
    # support.check and log_prob must agree, and log_prob must still be the log-uniform density
    assert torch.equal(within_support(tp, torch.cat([inside, outside])),
                       torch.isfinite(tp.log_prob(torch.cat([inside, outside]))))
    expected = -np.log(10.0) - np.log(np.log(100.0) - np.log(1.0))   # tau term; the Uniform adds 0
    assert float(tp.log_prob(inside)) == pytest.approx(expected, rel=1e-6)


def test_torch_prior_adapter_rejects_unknown_dist():
    sb = _require_sbi()

    class Weird:
        pass

    p = wp.Prior({"a": wp.Uniform(0, 1)})
    p.distributions["a"] = Weird()
    with pytest.raises(TypeError, match="Uniform/LogUniform"):
        _to_torch_prior(p, sb)


def test_torch_model_in_space_matches_the_numpy_likelihood():
    """The GPU simulate path must map flux -> comparison space exactly as the CPU path does.

    Regression test. ``predict_torch`` returns FLUX, but the observation the flow is trained
    against lives in ``lik.space``. Until this was fixed the GPU path added noise to the raw flux
    and the sampler simply refused to run in magnitude space; now it maps first and adds noise
    after, so the two simulators must agree.
    """
    import torch

    from whisper_cbpf.likelihood import GaussianLikelihood
    from whisper_cbpf.samplers.snpe import _torch_model_in_space

    lc = wp.LightCurve(time=[1.0, 2.0, 3.0], band=["r"] * 3,
                       magnitude=[20.0, 20.5, 21.0], magnitude_err=[0.05] * 3)
    lik = GaussianLikelihood(lc, space="magnitude")
    assert lik.space == "magnitude"

    flux = np.array([1e-5, 3.631e-6, 1e-7])
    ref = lik.model_in_space(flux)
    got = _torch_model_in_space(lik.space, lik.zeropoint_jy, torch)(
        torch.as_tensor(flux, dtype=torch.float64))
    assert np.allclose(np.asarray(got), ref, rtol=1e-12)

    # flux space is the identity, and must not copy-convert the tensor's meaning
    t = torch.as_tensor(flux)
    assert torch.equal(_torch_model_in_space("flux", lik.zeropoint_jy, torch)(t), t)


def test_torch_model_in_space_floor_survives_float32():
    """A non-positive model flux must give a finite faint magnitude, not -inf -> NaN.

    Regression test for the floor value specifically. The numpy likelihood clamps at
    ``_MIN_FLUX_JY = 1e-300``, which is fine in float64 and useless in float32 -- the simulate
    path's dtype -- because 1e-300 underflows to exactly 0.0 there, so the clamp stops clamping.
    ``log10(0)`` is then -inf and ``nan_to_num`` cannot tell that apart from a real value.
    """
    import torch

    from whisper_cbpf.likelihood import _MIN_FLUX_JY
    from whisper_cbpf.samplers.snpe import _torch_model_in_space

    to_mag = _torch_model_in_space("magnitude", 3631.0, torch)
    bad = torch.tensor([0.0, -1e-12, 1e-30], dtype=torch.float32)
    out = to_mag(bad)
    assert bool(torch.isfinite(out).all()), f"non-finite magnitudes from {bad.tolist()}: {out}"
    assert float(out[0]) == pytest.approx(103.7, abs=0.5)     # ~104 mag: fainter than any detection
    # ...and demonstrate why the numpy floor could not have been reused here
    assert torch.tensor(_MIN_FLUX_JY, dtype=torch.float32) == 0.0


def test_torch_simulate_maps_space_before_adding_noise():
    """Order matters: ``mf = model_in_space(flux)`` THEN ``mf + noise``, as the CPU simulator does.

    Regression test. With zero reported errors the simulation is noiseless, so the returned x must
    be the magnitude of the model flux exactly. Adding flux-scale noise to a magnitude (or the
    reverse) is silent -- the flow still trains, on the wrong summaries.
    """
    import torch

    from whisper_cbpf.samplers.snpe import _torch_model_in_space, _torch_simulate

    n_obs, zp = 4, 3631.0
    flux_row = torch.tensor([1e-5, 5e-6, 2e-6, 1e-6], dtype=torch.float64)

    class _Proposal:
        def sample(self, shape):
            return torch.zeros((int(shape[0]), 1), dtype=torch.float64)

    def predict_torch(theta, times):                     # constant, so x is fully determined
        return flux_row.expand(theta.shape[0], n_obs)

    times_t = torch.zeros(n_obs, dtype=torch.float64)
    sig_t = torch.zeros(n_obs, dtype=torch.float64)      # noiseless
    to_mag = _torch_model_in_space("magnitude", zp, torch)
    _, x = _torch_simulate(_Proposal(), 3, predict_torch, times_t, sig_t, None, torch, seed=0,
                           model_in_space=to_mag)

    expected = np.asarray(to_mag(flux_row), dtype=float)
    assert x.shape == (3, n_obs)
    assert np.allclose(np.asarray(x, dtype=float), expected, rtol=1e-6)
    # and the flux the map was applied to is NOT what comes back -- i.e. the map really ran
    assert not np.allclose(np.asarray(x[0], dtype=float), np.asarray(flux_row, dtype=float))


def _synthetic_flare_lc(seed=0, n=40):
    m = get_model("flare")
    t = np.linspace(0.1, 10, n)
    flux = m.predict({"amplitude": 1.0, "rise_time": 1.0, "decay_time": 3.0}, t, None)
    err = np.full_like(flux, 0.05)
    obs = flux + np.random.default_rng(seed).normal(0, err)
    return wp.LightCurve(time=t, band=["r"] * n, flux=obs, flux_err=err, name="synth")


@pytest.mark.slow
def test_snpe_fit_end_to_end():
    """One NPE round on synthetic flare data: structural correctness + prior-support sanity.

    (Tight parameter recovery needs many more simulations/rounds; that's exercised in the demo, not
    in this fast CI test.)
    """
    lc = _synthetic_flare_lc()
    prior = wp.Prior({"amplitude": wp.Uniform(0, 5),
                      "rise_time": wp.Uniform(0.1, 5),
                      "decay_time": wp.Uniform(0.5, 10)})
    res = wp.fit_SNPE(lc, "flare", prior=prior, num_rounds=1, num_simulations=400,
                      num_samples=500, seed=0, max_num_epochs=40)

    assert res.sampler == "snpe" and res.model == "flare"
    assert res.samples.shape == (500, 3)
    assert list(res.samples.columns) == ["amplitude", "rise_time", "decay_time"]
    assert set(res.best_params) == {"amplitude", "rise_time", "decay_time"}
    assert np.isfinite(res.aic) and np.isfinite(res.bic) and np.isfinite(res.max_log_likelihood)
    assert res.n_data == len(lc) and res.n_params == 3
    # samples respect the prior support -> validates the prior adapter (bounds + ordering)
    assert res.samples["amplitude"].between(0, 5).all()
    assert res.samples["rise_time"].between(0.1, 5).all()
    assert res.samples["decay_time"].between(0.5, 10).all()
    # trained sbi posterior is attached for resampling / pairplot
    assert hasattr(res, "posterior") and len(res.posteriors) == 1
    extra = res.posterior.sample((50,), show_progress_bars=False)   # resample from the trained posterior
    assert tuple(extra.shape) == (50, 3)
    import json
    json.loads(res.to_json())   # JSON-serializable (no torch objects leak into to_dict)


@pytest.mark.slow
def test_snpe_dispatch_and_alias_multiround():
    lc = _synthetic_flare_lc(seed=1)
    prior = wp.Prior({"amplitude": wp.Uniform(0, 5),
                      "rise_time": wp.Uniform(0.1, 5),
                      "decay_time": wp.Uniform(0.5, 10)})
    res = wp.fit(lc, "flare", sampler="npe", prior=prior, num_rounds=2, num_simulations=250,
                 num_samples=300, seed=1, max_num_epochs=25)
    assert res.sampler == "snpe" and res.info["num_rounds"] == 2
    assert res.info["total_simulations"] == 500
    assert res.n_samples == 300


def test_resolve_device_auto_gpu_and_fallback():
    """`_resolve_device` maps auto/gpu/cuda/cpu and falls back (with a warning) when CUDA is absent."""
    from whisper_cbpf.samplers.snpe import _resolve_device

    class _Cuda:
        def __init__(self, ok): self._ok = ok
        def is_available(self): return self._ok

    class _Torch:
        def __init__(self, ok): self.cuda = _Cuda(ok)

    have, none = _Torch(True), _Torch(False)
    assert _resolve_device("auto", have) == "cuda"
    assert _resolve_device("gpu", have) == "cuda"
    assert _resolve_device("cuda:2", have) == "cuda:2"
    assert _resolve_device("cpu", have) == "cpu"
    assert _resolve_device("auto", none) == "cpu"
    with pytest.warns(UserWarning, match="CUDA is unavailable"):
        assert _resolve_device("cuda", none) == "cpu"


@pytest.mark.slow
def test_snpe_gpu_recovers():
    """SNPE trains on the GPU (prior + observed data on-device) and recovers the injection."""
    pytest.importorskip("sbi")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    m = get_model("gaussian_rise")
    true = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}
    t = np.linspace(0.1, 30, 50)
    flux = m.predict(true, t, None)
    obs = flux + np.random.default_rng(0).normal(0, 0.1, flux.shape)
    lc = wp.LightCurve(time=t, band=["r"] * 50, flux=obs, flux_err=np.full_like(flux, 0.1), name="syn")
    res = wp.fit_SNPE(lc, "gaussian_rise", num_rounds=1, num_simulations=400, num_samples=600,
                      space="flux", device="cuda", seed=0, max_num_epochs=60)
    assert res.info["device"].startswith("cuda")
    assert np.isfinite(res.aic)
    assert abs(res.summary["amplitude"]["median"] - 5.0) / 5.0 < 0.5


@pytest.mark.slow
def test_snpe_embedding_net_and_custom_estimator():
    """A custom embedding net and custom density-estimator architecture both run end-to-end."""
    import torch.nn as nn
    lc = _synthetic_flare_lc()
    prior = wp.Prior({"amplitude": wp.Uniform(0, 5), "rise_time": wp.Uniform(0.1, 5),
                      "decay_time": wp.Uniform(0.5, 10)})
    emb = nn.Sequential(nn.Linear(len(lc), 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
    res = wp.fit_SNPE(lc, "flare", prior=prior, num_rounds=1, num_simulations=200, num_samples=200,
                      embedding_net=emb, max_num_epochs=20, seed=0)
    assert res.samples.shape == (200, 3) and res.info["embedding_net"] == "Sequential"

    res2 = wp.fit_SNPE(lc, "flare", prior=prior, num_rounds=1, num_simulations=200, num_samples=150,
                       density_estimator="nsf", hidden_features=20, num_transforms=2, num_bins=4,
                       max_num_epochs=20, seed=0)
    assert res2.samples.shape == (150, 3) and res2.info["density_estimator"] == "nsf"
