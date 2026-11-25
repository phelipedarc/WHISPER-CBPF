"""``sampler="snpe_gpu"``: the batched GPU simulator, band threading, and the leakage guard.

The two things worth the most here are the ones that would produce **a plausible wrong number with
no error**:

* the batched simulator evaluating the wrong FILTER for a row — which is what the old
  ``predict_torch(theta, times)`` hook could not even express, and why the sampler was hard-wired to
  the flare; and
* the round-to-round proposal draw stalling forever in a rejection loop that never raises.

Both are tested directly and first. Everything down to the ``slow`` marker runs on CPU with no CUDA
and no sncosmo; the marked tests train a real network on the real AT2017GFO light curve.
"""
from __future__ import annotations

import numpy as np
import pytest

import whisper_cbpf as wp


# ------------------------------------------------------- no jax, no torch, no sbi required
def test_snpe_gpu_is_a_registered_sampler():
    """It used to be deliberately absent from the registry; registering it IS the feature."""
    assert "snpe_gpu" in wp.list_samplers()


def test_predict_torch_bands_argument_is_detected_from_the_signature():
    """A two-argument callable must keep working; a three-argument one must be given the bands.

    This dispatch is the whole reason a photometric model can use the hook at all, and getting it
    backwards is silent: a two-argument flare callable handed a third argument raises, while a
    photometric one starved of it evaluates every row in filter 0.
    """
    from whisper_cbpf.samplers.snpe import _predict_torch_accepts_bands

    assert not _predict_torch_accepts_bands(lambda theta, times: None)
    assert _predict_torch_accepts_bands(lambda theta, times, bands: None)
    assert _predict_torch_accepts_bands(lambda theta, times=None, bands=None: None)
    assert _predict_torch_accepts_bands(lambda *a, **k: None)
    # keyword-ONLY third parameter is not a positional third argument
    assert not _predict_torch_accepts_bands(lambda theta, times, *, bands=None: None)


def test_encode_bands_derives_labels_and_still_honours_a_fixed_list():
    """AT2017GFO carries 29 filters including HST's; a hard-coded LSST list would refuse the data."""
    from whisper_cbpf.samplers.jax.snpe_gpu import LSST_BANDS, encode_bands

    codes, labels = encode_bands(["H", "g", "H", "F160W"], labels=None)
    assert labels == ["F160W", "H", "g"]                       # sorted unique
    assert list(codes) == [1, 2, 1, 0]

    codes, labels = encode_bands(["r", "g"], labels=LSST_BANDS)
    assert labels == list(LSST_BANDS) and list(codes) == [2, 1]
    with pytest.raises(ValueError, match="not in the known label list"):
        encode_bands(["H"], labels=LSST_BANDS)


# ------------------------------------------------------------------- jax + torch (CPU is fine)
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")


def _rtol():
    """``predict`` is numpy float64, ``predict_jax`` runs at the session's JAX precision."""
    return 1e-10 if jax.config.jax_enable_x64 else 2e-6


def _toy_lc(n=24, seed=0):
    t = np.linspace(1.0, 30.0, n)
    band = np.array(["ztfg" if i % 2 else "ztfr" for i in range(n)])
    flux = 2.0 * np.exp(-t / 12.0) + 0.5
    rng = np.random.default_rng(seed)
    return wp.LightCurve(time=t, band=band, flux=flux + rng.normal(0, 0.02, n),
                         flux_err=np.full(n, 0.02), name="toy", redshift=0.01)


#: Per-band multipliers chosen far apart so a row evaluated in the WRONG filter cannot pass as noise.
_BAND_SCALE = np.array([1.0, 7.0])
_BAND_NAMES = ("ztfg", "ztfr")


def _banded_toy_model(name="snpe_gpu_toy"):
    """A 2-parameter analytic model that is genuinely BAND-DEPENDENT, with both predict slots.

    Band dependence is the point: the self-contained (flare-shaped) case cannot detect a simulator
    that drops the band axis, because for it there is no band axis to drop.
    """
    index = {b: i for i, b in enumerate(_BAND_NAMES)}

    def predict(parameters, times, bands=None):
        if bands is None:
            raise ValueError("banded toy model needs bands")
        idx = np.array([index[str(b)] for b in np.asarray(bands)])
        t = np.asarray(times, dtype=float)
        return (float(parameters["amp"]) * np.exp(-t / float(parameters["tau"]))
                * _BAND_SCALE[idx] + 0.5)

    def predict_jax(theta, times, band_idx=None):
        if band_idx is None:
            raise ValueError("banded toy model needs a band_idx array")
        scale = jnp.asarray(_BAND_SCALE)[jnp.asarray(band_idx)]
        return theta[0] * jnp.exp(-jnp.asarray(times) / theta[1]) * scale + 0.5

    predict_jax.band_index = lambda bands: np.array(
        [index[str(b)] for b in np.asarray(bands)], dtype=int)

    return wp.register_model(
        name, predict, ["amp", "tau"],
        prior=wp.Prior({"amp": wp.Uniform(0.5, 5.0), "tau": wp.Uniform(4.0, 30.0)}),
        overwrite=True, predict_jax=predict_jax)


def test_batched_simulator_matches_predict_row_by_row_including_the_band():
    """THE central correctness test: the GPU simulator must reproduce ``model.predict`` exactly.

    Rows are compared one at a time against the numpy contract, on a light curve whose two filters
    differ by a factor 7 — so a simulator that lost the band axis is off by 600%, not by round-off.
    """
    from whisper_cbpf.samplers.jax.snpe_gpu import make_predict_torch

    lc, m = _toy_lc(), _banded_toy_model()
    rows = np.array([[1.0, 6.0], [2.5, 15.0], [4.0, 28.0]])
    pt = make_predict_torch(lc, m, chunk=2)
    got = np.asarray(pt(torch.as_tensor(rows)).cpu().numpy(), dtype=float)
    assert got.shape == (3, len(lc.time))
    for i, (amp, tau) in enumerate(rows):
        want = m.predict({"amp": amp, "tau": tau}, lc.time, lc.band)
        assert np.allclose(got[i], want, rtol=_rtol()), f"row {i}"
    # ...and it is genuinely band-dependent, i.e. the test can fail
    assert not np.allclose(got[0], m.predict({"amp": 1.0, "tau": 6.0}, lc.time,
                                             np.array(["ztfg"] * len(lc.time))))


def test_batched_simulator_reports_a_zero_copy_handoff():
    """``info["sim_backend"]`` is the evidence the GPU path ran, so the flag must be real."""
    from whisper_cbpf.samplers.jax.snpe_gpu import make_predict_torch

    pt = make_predict_torch(_toy_lc(), _banded_toy_model(), chunk=2)
    assert pt.transfer is None                                # nothing claimed before a call
    pt(torch.as_tensor(np.array([[2.0, 12.0]])))
    assert pt.transfer in ("dlpack", "host")
    assert pt.chunk == 2 and pt.names == ["amp", "tau"]


def test_batched_simulator_default_chunk_is_the_re_measured_width():
    """250 simulations per compiled block, not 16. The simulator compiles in 1.3-2.5 s
    at 250, and a latency-bound model pays per block (the TDE at n_time=5000: 2.25 s against 0.73 s
    per 1000-simulation call). Not a power of two: see abc_gpu.DEFAULT_CHUNK."""
    from whisper_cbpf.samplers.jax.snpe_gpu import DEFAULT_SIM_CHUNK, make_predict_torch

    assert DEFAULT_SIM_CHUNK == 250
    pt = make_predict_torch(_toy_lc(), _banded_toy_model())
    assert pt.chunk == 250
    rows = torch.as_tensor(np.array([[1.0, 6.0], [2.5, 15.0], [4.0, 28.0]]))
    assert tuple(pt(rows).shape) == (3, 24)               # fewer rows than one block: padded


def test_batched_simulator_refuses_a_different_grid():
    """The band -> filter-index map is resolved once, outside the trace. A different band array
    would therefore evaluate the WRONG filters silently; a different epoch count, the wrong span."""
    from whisper_cbpf.samplers.jax.snpe_gpu import make_predict_torch

    lc = _toy_lc()
    pt = make_predict_torch(lc, _banded_toy_model(), chunk=2)
    theta = torch.as_tensor(np.array([[2.0, 12.0]]))
    pt(theta, lc.time, lc.band)                               # the light curve's own grid is fine
    with pytest.raises(ValueError, match="band array different"):
        pt(theta, lc.time, np.array(["ztfg"] * len(lc.time)))
    with pytest.raises(ValueError, match="epochs"):
        pt(theta, np.linspace(0, 1, 7), lc.band)


def test_parameter_columns_follow_the_prior_order_not_the_models():
    """``snpe_gpu`` orders theta by ``prior.names``; ``predict_jax`` wants ``model.parameters``.

    A user-supplied prior in another order would evaluate ``tau`` where ``amp`` belongs, converge,
    and be wrong with no error. The adapter reorders; this asserts snpe_gpu asks it to.
    """
    from whisper_cbpf.samplers.jax.snpe_gpu import make_predict_torch

    lc, m = _toy_lc(), _banded_toy_model(name="snpe_gpu_toy_rev")
    names = ["tau", "amp"]
    assert names != list(m.parameters)
    pt = make_predict_torch(lc, m, names=names, chunk=2)
    got = np.asarray(pt(torch.as_tensor(np.array([[12.0, 2.0]]))).cpu().numpy())[0]
    assert np.allclose(got, m.predict({"amp": 2.0, "tau": 12.0}, lc.time, lc.band), rtol=_rtol())


def test_context_embedding_injects_sigma_and_band_after_sbis_standardizer():
    """The optional embedding must accept sbi's flat ``(batch, n_points)`` and add real channels.

    sbi composes ``nn.Sequential(standardizing_net, embedding_net)``, so a constant channel appended
    to the INPUT is z-scored to exactly 0.0 before the network sees it. Injecting it here — inside
    the embedding — is the whole reason this class exists, so the channel count and the one-hot must
    be right.
    """
    from whisper_cbpf.samplers.jax.snpe_gpu import ContextEmbedding

    sigma = np.array([0.02, 0.05, 0.02, 0.10])
    codes = np.array([0, 1, 1, 0])
    emb = ContextEmbedding(sigma, codes, n_bands=2, spec="mlp", latent_dim=8)
    assert emb.n_channels == 2 + 2                      # value + log-sigma + one-hot per band
    out = emb(torch.zeros((5, 4)))
    assert tuple(out.shape) == (5, 8)
    # the band block really is a one-hot of `codes`, not an integer code
    assert emb.band_ch.shape == (1, 2, 4)
    assert emb.band_ch[0].argmax(dim=0).tolist() == codes.tolist()
    assert bool((emb.band_ch.sum(dim=1) == 1).all())
    # a zero magnitude error must not make the sigma channel NaN (log10(0) = -inf)
    emb0 = ContextEmbedding(np.array([0.0, 0.05]), np.array([0, 1]), n_bands=2, latent_dim=4)
    assert bool(torch.isfinite(emb0.sigma_ch).all())


def test_a_context_embedding_with_the_stacked_layout_is_refused_before_simulating():
    """``embedding_net=ContextEmbedding(...)`` is what the deprecation of ``embedding="tcn"`` points
    to, and it reads the value-only vector: with ``x_format="stacked"`` it raised ``RuntimeError:
    shape '[1, 1, 24]' is invalid for input of size 72`` at the first training batch, after the
    simulations had been paid for."""
    from whisper_cbpf.samplers.jax.snpe_gpu import ContextEmbedding, encode_bands

    lc, m = _toy_lc(), _banded_toy_model(name="snpe_gpu_toy_ctx_stacked")
    codes, labels = encode_bands(lc.band, labels=None)
    ctx = ContextEmbedding(np.asarray(lc.flux_err), codes, n_bands=len(labels), latent_dim=8)
    with pytest.raises(ValueError, match="x_format='value'"):
        wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", embedding_net=ctx,
               x_format="stacked", num_rounds=1, num_simulations=50)


def test_flare_jax_now_carries_a_predict_jax():
    """The slot was empty, which is why ``snpe_gpu`` needed a flare-shaped special case."""
    m = wp.get_model("flare_jax")
    assert m.predict_jax is not None
    theta = jnp.asarray([0.0, 0.5, 1.0, 10.0])                # log_amp, log_sigma, log_tau, t0
    times = jnp.asarray([5.0, 10.0, 20.0])
    got = np.asarray(m.predict_jax(theta, times))
    want = m.predict({"log_amp": 0.0, "log_sigma": 0.5, "log_tau": 1.0, "t0": 10.0},
                     np.asarray(times), None)
    assert np.allclose(got, want, rtol=2e-6)


def test_snpe_and_snpe_gpu_default_to_the_same_num_rounds():
    """``snpe_gpu`` wraps ``SNPESampler``, so its default must not quietly pick another algorithm.

    Regression: ``snpe`` defaulted to 2 rounds (sequential SNPE), ``snpe_gpu`` and ``fit_snpe_gpu``
    to 1 (amortised NPE), and the API reference said 3.
    """
    import inspect

    from whisper_cbpf.samplers.jax.snpe_gpu import SNPEGPUSampler, fit_snpe_gpu
    from whisper_cbpf.samplers.snpe import SNPESampler

    def default(fn):
        return inspect.signature(fn).parameters["num_rounds"].default

    assert default(SNPEGPUSampler.fit) == default(fit_snpe_gpu) == default(SNPESampler.fit)


def test_snpe_gpu_refuses_a_model_with_no_predict_jax():
    """Refusing is the honest answer: a numpy Python loop under a name containing 'gpu' is not."""
    from whisper_cbpf.samplers.jax.snpe_gpu import SNPEGPUSampler

    with pytest.raises(ValueError, match="no predict_jax"):
        SNPEGPUSampler().fit(_toy_lc(), "flare", device="cpu")


def test_predrawn_proposal_returns_the_guarded_draw_and_refuses_a_mismatch():
    """The CPU simulate path can only be given a guarded draw by presenting it as a proposal."""
    from whisper_cbpf.samplers.snpe import _PredrawnProposal

    theta = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    p = _PredrawnProposal(theta)
    assert torch.equal(p.sample((3,)), theta)
    with pytest.raises(ValueError, match="parameter rows"):
        p.sample((4,))


def test_robust_proposal_draw_escapes_to_mcmc_when_the_flow_leaks():
    """The leakage guard, at the call that used to be unguarded.

    A proposal that leaks (nothing lands in the prior support) must NOT be drawn from by rejection
    -- that loop does not raise, it simply never finishes. The guard must notice and take the draws
    from the MCMC posterior instead. Stubbed so the failure is deterministic rather than a race.
    """
    pytest.importorskip("sbi")
    from whisper_cbpf.samplers.snpe import _robust_proposal_draw

    drawn = {}

    class _Estimator:
        def sample(self, shape, condition=None):
            return torch.full((int(shape[0]), 1), 999.0)      # every probe draw is out of support

    class _LeakyPosterior:
        posterior_estimator = _Estimator()

        def sample(self, shape, **kw):                        # the loop that would stall
            raise AssertionError("rejection sampling must not be reached when the proposal leaks")

    class _Prior:
        def log_prob(self, theta):
            # the probe draws (999.0) are outside; the MCMC draws (0.0) are inside
            return torch.where(theta.reshape(theta.shape[0], -1).abs().amax(dim=-1) > 100.0,
                               torch.tensor(float("-inf")), torch.tensor(0.0))

    class _MCMC:
        def set_default_x(self, x):
            pass

        def sample(self, shape, x=None, **kw):
            drawn["n"] = int(shape[0])
            return torch.zeros((int(shape[0]), 1))

    class _Inference:
        def build_posterior(self, de_net, **kw):
            drawn["sample_with"] = kw.get("sample_with")
            return _MCMC()

    with pytest.warns(UserWarning, match="round-to-round proposal"):
        theta, method, rate = _robust_proposal_draw(
            _LeakyPosterior(), 11, _Inference(), object(), _Prior(), torch,
            torch.zeros(3), False, 1)
    assert method == "mcmc_fallback" and rate == 0.0
    assert drawn == {"sample_with": "mcmc", "n": 11} and theta.shape == (11, 1)


def test_robust_proposal_draw_leaves_a_healthy_proposal_alone():
    """The common path must be untouched: probe, accept, draw by rejection as before."""
    pytest.importorskip("sbi")
    from whisper_cbpf.samplers.snpe import _robust_proposal_draw

    class _Estimator:
        def sample(self, shape, condition=None):
            return torch.zeros((int(shape[0]), 1))

    class _HealthyPosterior:
        posterior_estimator = _Estimator()

        def sample(self, shape, **kw):
            return torch.ones((int(shape[0]), 1))

    class _Prior:
        def log_prob(self, theta):
            return torch.zeros((theta.shape[0],))

    theta, method, rate = _robust_proposal_draw(
        _HealthyPosterior(), 5, None, None, _Prior(), torch, torch.zeros(3), False, 1)
    assert method == "rejection" and rate == 1.0 and torch.equal(theta, torch.ones((5, 1)))


def test_the_prior_proposal_is_drawn_from_directly():
    """Round 0's proposal is the prior: no flow inverse, nothing to guard, no probe cost."""
    pytest.importorskip("sbi")
    from whisper_cbpf.samplers.snpe import _robust_proposal_draw

    class _Prior:
        def sample(self, shape):
            return torch.zeros((int(shape[0]), 2))

        def log_prob(self, theta):
            return torch.zeros((theta.shape[0],))

    theta, method, rate = _robust_proposal_draw(
        _Prior(), 4, None, None, _Prior(), torch, None, False, 1)
    assert method == "direct" and rate is None and theta.shape == (4, 2)


def test_a_draw_exactly_on_a_prior_bound_is_nudged_into_the_density():
    """Regression: ONE draw in 1500 killed round 2 of a real AT2017GFO fit.

    sbi accepts a draw when ``support.check`` passes — torch's ``interval`` constraint is CLOSED, so
    ``theta == high`` passes — and then asserts ``prior.log_prob`` is finite in the NPE-C atomic
    loss, where ``Uniform.log_prob`` is HALF-OPEN and ``theta == high`` is ``-inf``. The two tests
    disagree at exactly one point per bound, and the whole round dies there.
    """
    pytest.importorskip("sbi")
    from whisper_cbpf.samplers.snpe import _nudge_into_prior_density, _require_sbi, _to_torch_prior
    from sbi.utils.sbiutils import within_support

    sb = _require_sbi()
    prior = wp.Prior({"a": wp.Uniform(0.0, 1.0), "b": wp.Uniform(2.0, 5.0)})
    tp, _, _ = sb.process_prior(_to_torch_prior(prior, sb))

    theta = torch.tensor([[0.5, 3.0], [1.0, 3.0], [0.5, 5.0]])     # rows 1 and 2 sit ON a bound
    assert bool(within_support(tp, theta).all()), "sbi's accept test passes them; that is the bug"
    assert int((~torch.isfinite(tp.log_prob(theta))).sum()) == 2   # ...its loss then rejects them

    fixed, n_nudged, n_bad = _nudge_into_prior_density(theta, tp, torch)
    assert (n_nudged, n_bad) == (2, 0)
    assert bool(torch.isfinite(tp.log_prob(fixed)).all())
    assert torch.equal(fixed[0], theta[0])                          # untouched rows stay identical
    # ...and the repair is one ULP, not a re-draw
    assert abs(float(fixed[1, 0]) - 1.0) < 1e-6
    assert abs(float(fixed[2, 1]) - 5.0) < 1e-5

    # nothing on a bound -> nothing touched at all
    clean = torch.tensor([[0.5, 3.0]])
    same, n_nudged, n_bad = _nudge_into_prior_density(clean, tp, torch)
    assert (n_nudged, n_bad) == (0, 0) and same is clean


# ---------------------------------------------------------------------------- end to end (slow)
@pytest.mark.slow
def test_snpe_gpu_end_to_end_on_a_banded_toy_model():
    """A full ``wp.fit(..., sampler="snpe_gpu")`` round: dispatch, backend flag, prior support."""
    pytest.importorskip("sbi")

    lc, m = _toy_lc(n=40), _banded_toy_model(name="snpe_gpu_toy_e2e")
    res = wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", space="flux",
                 num_rounds=1, num_simulations=200, num_samples=300, seed=0,
                 sim_chunk=16, max_num_epochs=15)
    assert res.sampler == "snpe_gpu" and res.model == m.name
    assert res.samples.shape == (300, 2) and list(res.samples.columns) == ["amp", "tau"]
    assert res.info["sim_backend"] in ("jax-dlpack-torch", "jax-host-torch")
    assert res.info["predict_torch"] == "auto"
    assert res.samples["amp"].between(0.5, 5.0).all()
    assert res.samples["tau"].between(4.0, 30.0).all()
    assert res.samples_by_chain.shape == (1, 300, 2)
    import json
    json.loads(res.to_json())                                  # no torch objects leak into to_dict


@pytest.mark.slow
@pytest.mark.parametrize("embedding,estimator", [("mlp", "nsf"), ("tcn", "mdn")])
def test_snpe_gpu_embedding_and_density_estimator_options(embedding, estimator):
    """The optional embedding and the estimator choices must survive the wrapper, not just exist."""
    pytest.importorskip("sbi")

    lc, m = _toy_lc(n=32), _banded_toy_model(name=f"snpe_gpu_toy_{embedding}")
    res = wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", space="flux",
                 num_rounds=1, num_simulations=120, num_samples=150, seed=0,
                 sim_chunk=16, embedding_net=embedding, embedding_latent=8,
                 density_estimator=estimator, max_num_epochs=8)
    assert res.samples.shape == (150, 2)
    assert res.info["embedding_net"] == embedding
    assert res.info["density_estimator"] == estimator


def _tcn_channels(res):
    """Input channels of every TCN inside the trained estimator: what the network conditions on."""
    from whisper_cbpf.embeddings import TCNEmbedding

    return [mod.n_channels for mod in res.posterior.posterior_estimator.modules()
            if isinstance(mod, TCNEmbedding)]


_EMB_TINY = dict(space="flux", num_rounds=1, num_simulations=100, num_samples=100, seed=0,
                 embedding_latent=8, max_num_epochs=3)


@pytest.mark.slow
@pytest.mark.parametrize("x_format,channels", [("value", 1), ("stacked", 3)])
def test_embedding_net_names_one_network_on_both_arms(x_format, channels):
    """One keyword, one meaning: ``embedding_net="tcn"`` is the same TCN on ``snpe`` and ``snpe_gpu``.

    Regression, three ways: the GPU arm took ``embedding=``, and ``embedding_net=`` raised
    ``TypeError: multiple values``; its ``"tcn"`` was a ContextEmbedding over (value, log sigma,
    band one-hot) where the CPU's is a TCN over the ``x_format`` channels; and with
    ``x_format="stacked"`` it raised ``RuntimeError: shape '[1, 1, 29]' is invalid for input of
    size 87`` at the first training batch, after the simulations had been paid for.
    """
    pytest.importorskip("sbi")

    lc, m = _toy_lc(), _banded_toy_model(name=f"snpe_gpu_toy_emb_{x_format}")
    cpu = wp.fit(lc, m.name, sampler="snpe", embedding_net="tcn", x_format=x_format, **_EMB_TINY)
    gpu = wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", sim_chunk=16, embedding_net="tcn",
                 x_format=x_format, **_EMB_TINY)
    assert _tcn_channels(cpu) == _tcn_channels(gpu) == [channels]
    assert cpu.info["embedding_net"] == gpu.info["embedding_net"] == "tcn"


@pytest.mark.slow
def test_the_old_embedding_keyword_is_a_deprecated_alias():
    pytest.importorskip("sbi")

    lc, m = _toy_lc(), _banded_toy_model(name="snpe_gpu_toy_alias")
    for old, new in (("tcn", "tcn"), ("none", None)):
        with pytest.warns(DeprecationWarning, match="embedding_net"):
            res = wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", sim_chunk=16,
                         embedding=old, **_EMB_TINY)
        assert res.info["embedding_net"] == new
    with pytest.raises(TypeError, match="embedding_net"):
        wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", embedding="tcn", embedding_net="mlp")


@pytest.mark.slow
def test_snpe_gpu_two_rounds_terminate():
    """``num_rounds=2`` must RETURN. The between-round draw used to be unguarded rejection
    sampling and was observed at 0 of 1500 accepted after 42 s, with no error and no end."""
    pytest.importorskip("sbi")

    lc, m = _toy_lc(n=40), _banded_toy_model(name="snpe_gpu_toy_2r")
    res = wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", space="flux",
                 num_rounds=2, num_simulations=150, num_samples=200, seed=0,
                 sim_chunk=16, max_num_epochs=10)
    assert res.info["num_rounds"] == 2 and res.info["total_simulations"] == 300
    # one guarded draw per round transition, each recorded with how it was taken
    assert len(res.info["proposal_draw_methods"]) == 1
    assert res.info["proposal_draw_methods"][0] in ("rejection", "mcmc_fallback")


@pytest.mark.slow
def test_snpe_gpu_fits_the_kilonova_on_at2017gfo(at2017gfo_csv):
    """The real thing: two-component kilonova, real AT2017GFO photometry, magnitude space.

    ``set_explosion_date`` BEFORE ``select_time_window`` -- the window is in days since explosion,
    so the other order silently selects nothing.
    """
    pytest.importorskip("sbi")
    pytest.importorskip("sncosmo")

    z, dl_cm = 0.0098, 1.23e26
    lc = wp.load_lightcurve(str(at2017gfo_csv), redshift=z)
    lc = lc.select_bands(["g", "r"]).set_explosion_date(57982.529).select_time_window(0.0, 8.0)
    assert len(lc.time) > 20, "explosion date must be set before the window, or this is empty"

    m = wp.register_kilonova_two(["sdssg", "sdssr"], redshift=z, dl_cm=dl_cm, n_wave=100,
                                 band_aliases={"g": "sdssg", "r": "sdssr"},
                                 name="kilonova_two_jax_test")
    res = wp.fit(lc, m.name, sampler="snpe_gpu", device="cpu", num_rounds=1,
                 num_simulations=128, num_samples=200, seed=0, sim_chunk=16, max_num_epochs=5)
    assert res.info["space"] == "magnitude"                    # the data is photometry, not flux
    assert res.info["sim_backend"] in ("jax-dlpack-torch", "jax-host-torch")
    assert res.samples.shape == (200, len(m.parameters))
    for name in m.parameters:
        lo, hi = m.default_prior.distributions[name].bounds
        assert res.samples[name].between(lo, hi).all(), f"{name} left its prior box"
