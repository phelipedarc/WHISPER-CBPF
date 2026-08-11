"""Tests for the neural-SBI embedding nets (MLP + TCN) and the ABC/SNPE input upgrades."""
import numpy as np
import pytest

import whisper_labia as wp

torch = pytest.importorskip("torch")


def test_mlp_embedding_shape():
    from whisper_labia.embeddings import MLPEmbedding
    net = MLPEmbedding(80, latent_dim=16)
    out = net(torch.randn(7, 80))
    assert out.shape == (7, 16) and torch.isfinite(out).all()


def test_tcn_embedding_shape_and_channels():
    from whisper_labia.embeddings import TCNEmbedding
    net = TCNEmbedding(n_points=80, n_channels=4, latent_dim=16)
    out = net(torch.randn(5, 4 * 80))                    # flat sbi layout -> (B, C, L) inside
    assert out.shape == (5, 16) and torch.isfinite(out).all()


def test_tcn_causal_receptive_field_covers_input():
    # 4 levels x kernel 5: receptive field 1 + 2*4*(2^4-1) = 121 >= 80 points
    from whisper_labia.embeddings import TCNEmbedding
    net = TCNEmbedding(n_points=80, n_channels=1, latent_dim=8)
    x = torch.zeros(1, 80)
    y0 = net(x)
    x2 = x.clone(); x2[0, 0] = 5.0                       # perturb the FIRST point
    assert not torch.allclose(y0, net(x2))               # ...must reach the pooled output


def test_build_embedding_dispatch():
    from whisper_labia.embeddings import MLPEmbedding, TCNEmbedding, build_embedding
    assert isinstance(build_embedding("mlp", 50), MLPEmbedding)
    assert isinstance(build_embedding("tcn", 50, n_channels=4), TCNEmbedding)
    with pytest.raises(ValueError, match="Unknown embedding"):
        build_embedding("transformer", 50)


def test_snpe_x_format_validation():
    lc = wp.LightCurve(time=[1, 2, 3], band=["r"] * 3, flux=[1.0, 2.0, 1.0],
                       flux_err=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="x_format"):
        wp.fit_SNPE(lc, "gaussian_rise", x_format="bogus")


def test_abc_simulate_noise_reproducible_and_optional():
    from whisper_labia.models import get_model
    m = get_model("gaussian_rise")
    truth = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}
    t = np.linspace(0.1, 30, 40)
    obs = m.predict(truth, t, None) + np.random.default_rng(0).normal(0, 0.2, t.shape)
    lc = wp.LightCurve(time=t, band=["r"] * 40, flux=obs, flux_err=np.full_like(t, 0.2))
    # reproducible across n_jobs with noise on (per-index RNG streams)
    r1 = wp.fit_ABC(lc, m, n_simulations=400, quantile=0.05, n_jobs=1, seed=3)
    r2 = wp.fit_ABC(lc, m, n_simulations=400, quantile=0.05, n_jobs=2, seed=3)
    assert np.allclose(r1.samples["distance"], r2.samples["distance"])
    assert r1.info["simulate_noise"] is True
    # noisy simulations carry the irreducible ~2N distance floor; noiseless dips below it
    r0 = wp.fit_ABC(lc, m, n_simulations=400, quantile=0.05, simulate_noise=False, seed=3)
    assert r0.info["simulate_noise"] is False
    assert r0.min_distance < r1.min_distance
