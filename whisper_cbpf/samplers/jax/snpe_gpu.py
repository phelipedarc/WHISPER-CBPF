"""Arm C: per-observation SNPE on GPU, with the JAX model supplying GPU-batched simulation.

A thin wrapper around ``whisper_labia.samplers.snpe.SNPESampler`` — that sampler is used
completely unmodified. Two things are supplied from this side:

1. ``predict_torch=flare_jax.predict_torch`` — SNPESampler already has a hook for a batched,
   device-resident simulator (``_torch_simulate``); it falls back to a per-row numpy loop when
   no model provides one, and until now no whisper_labia model did. That CPU fallback is
   precisely the bottleneck this benchmark exists to remove.
2. ``embedding_net=ContextEmbedding(...)`` — adds per-point sigma and photometric band as
   extra input channels alongside the flux.

**Why not x_format="stacked".** SNPESampler's stacked layout appends (error, time) as extra
context channels to the raw input vector. For a single-object fit those channels are *constant*
across every simulation, and sbi composes ``nn.Sequential(standardizing_net, embedding_net)`` —
so its z-scoring maps them to exactly 0.0 before the network ever sees them. sbi even warns
about it ("Data has constant values in dimension(s) ..."). Measured on a 200-point curve, the
stacked layout was 4.6x WORSE on the worst parameter and 40% slower than passing flux alone:

    stacked, 2k sims/2 rounds  -> worst |median - truth| = 6.61  (131 s)
    value,   2k sims/2 rounds  -> worst |median - truth| = 1.43  ( 92 s)

This module therefore uses ``x_format="value"`` (flux only — the one channel that genuinely
varies between simulations, so z-scoring is meaningful) and injects sigma/band *inside* the
embedding, i.e. AFTER sbi's standardizing layer, where constant channels survive intact.

**Per-observation, not amortized**: every light curve gets its own freshly-trained network, so
the full simulate+train+sample cost counts toward that object's wall-clock. Real survey data
varies in cadence, noise and band coverage, so one network cannot be reused across curves.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch import nn

# whisper_labia is imported directly: whisper-GPU is cloned INTO the whisper checkout as
# `whisper_gpu/`, so the repo root that holds both is already on sys.path. The previous
# three-level dirname() walk-up hard-coded this file's nesting depth and broke on any move.

from whisper_labia.embeddings import build_embedding  # noqa: E402
from whisper_labia.samplers.snpe import SNPESampler  # noqa: E402

from ..models import flare as flare_jax  # noqa: E402

LSST_BANDS = ("u", "g", "r", "i", "z", "y")


class ContextEmbedding(nn.Module):
    """Concatenate constant per-point context (sigma, band one-hot) to the varying flux channel.

    Input from sbi: ``(batch, n_points)`` — the standardized flux vector.
    Output: ``(batch, latent_dim)``.

    Internally builds ``(batch, 1 + 1 + n_bands, n_points)``:
      * channel 0   — the flux (already z-scored by sbi's standardizing layer)
      * channel 1   — log10(sigma), centred; constant across simulations for this fit
      * channels 2+ — one-hot band code; also constant

    Band is one-hot rather than an integer code because it is categorical — integer codes
    would impose a false ordering (g < r < i) that a linear or convolutional layer reads as
    meaningful distance. Both context blocks are non-trainable ``buffer``s: they are properties
    of the observation's fixed grid, identical for the observation and every simulation drawn
    on it.

    Note this context is genuinely uninformative for a *single-object* fit (it cannot help
    discriminate between simulations, since it is the same for all of them). It is included
    because the benchmark specifies sigma and band as channels, and because it becomes
    informative the moment the network is asked to generalize across grids. The measured cost
    of carrying it is reported in BENCHMARK.md.
    """

    def __init__(self, sigma, band_codes, n_bands, spec="mlp", latent_dim=32):
        super().__init__()
        self.n_points = int(len(sigma))
        self.n_bands = int(n_bands)

        s = np.log10(np.asarray(sigma, dtype=np.float64))
        s = (s - s.mean()) / (s.std() + 1e-12)          # centre/scale so it can't dominate
        self.register_buffer("sigma_ch",
                             torch.as_tensor(s, dtype=torch.float32).view(1, 1, self.n_points))

        onehot = torch.zeros(self.n_bands, self.n_points, dtype=torch.float32)
        onehot[torch.as_tensor(band_codes, dtype=torch.long),
               torch.arange(self.n_points)] = 1.0
        self.register_buffer("band_ch", onehot.unsqueeze(0))

        self.n_channels = 2 + self.n_bands
        self.inner = build_embedding(spec, n_points=self.n_points,
                                     n_channels=self.n_channels, latent_dim=int(latent_dim))
        self.latent_dim = int(latent_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1).float()
        b = x.shape[0]
        flux = x.view(b, 1, self.n_points)
        ctx = torch.cat([self.sigma_ch.expand(b, -1, -1),
                         self.band_ch.expand(b, -1, -1)], dim=1)
        return self.inner(torch.cat([flux, ctx], dim=1).flatten(start_dim=1))


def encode_bands(band_array, labels=LSST_BANDS):
    """Map band labels to integer codes against a FIXED label list.

    Fixed rather than sorted-unique-per-curve so the channel index of a given filter means the
    same thing in every fit; a per-curve encoding would silently permute channels between
    curves.
    """
    band_array = np.asarray(band_array)
    lookup = {lab: i for i, lab in enumerate(labels)}
    missing = set(band_array.tolist()) - set(lookup)
    if missing:
        raise ValueError(f"bands {sorted(missing)} not in the known label list {labels}")
    return np.array([lookup[b] for b in band_array], dtype=int), list(labels)


def fit_snpe_gpu(lc, model, *, prior=None, num_rounds=3, num_simulations=10000,
                 embedding="none", embedding_latent=32, density_estimator="maf",
                 num_samples=10000, seed=0, device="cuda", show_progress=False,
                 **snpe_kwargs):
    """Fit one light curve with SNPE on GPU, using the JAX model for GPU-batched simulation.

    ``embedding``: ``"none"`` feeds the flux vector straight to the flow (measured best at this
    problem size); ``"mlp"``/``"tcn"`` wrap that architecture in :class:`ContextEmbedding` so
    sigma and band ride along as extra channels.

    All costs (simulation + training + sampling, every round) land in ``result.runtime_s`` —
    a genuine per-object cost, since the network is not reused.
    """
    # FLARE ONLY, and it fails loudly rather than silently.
    #
    # This function takes a `model` argument but then passes `predict_torch=flare_jax.predict_torch`
    # unconditionally (below), so whatever model is named, the simulations are FLARES. That is why
    # this sampler is deliberately absent from `whisper_gpu.GPU_SAMPLERS`: registering it would let
    # `wp.fit(lc, "kilonova_one_jax", sampler="snpe_gpu")` return flare draws carrying a kilonova's
    # parameter names -- a wrong answer with no error. Until the torch predictor is generalised, the
    # honest interface is a direct call with an explicit check.
    _mname = getattr(model, "name", model)
    if not str(_mname).startswith("flare"):
        raise NotImplementedError(
            f"fit_snpe_gpu is hard-wired to the flare model (it always uses "
            f"flare_jax.predict_torch and space='flux'), but was given model {_mname!r}. It cannot "
            f"fit anything else. Use sampler='snpe' (whisper's CPU SNPE, model-agnostic), or "
            f"generalise the predict_torch hook first.")

    n_points = int(len(lc.time))
    sigma = np.asarray(lc.flux_err, dtype=np.float64)

    torch.manual_seed(int(seed))     # before net construction: embedding init draws from this
    if embedding in (None, "none"):
        embedding_net = None
        emb_label = "none (raw flux -> flow)"
    else:
        codes, labels = encode_bands(lc.band)
        embedding_net = ContextEmbedding(sigma, codes, n_bands=len(labels),
                                          spec=embedding, latent_dim=embedding_latent)
        emb_label = f"ContextEmbedding({embedding}, {2 + len(labels)} channels)"

    result = SNPESampler().fit(
        lc, model, prior=prior,
        num_rounds=int(num_rounds), num_simulations=int(num_simulations),
        space="flux",                        # predict_torch path is flux-space only
        density_estimator=density_estimator,
        embedding_net=embedding_net,
        x_format="value",                     # see module docstring
        predict_torch=flare_jax.predict_torch,
        num_samples=int(num_samples),
        device=device, seed=int(seed), show_progress=show_progress,
        **snpe_kwargs,
    )
    result.info["embedding"] = emb_label
    result.info["sim_backend"] = "jax-dlpack-torch"
    result.info["n_points"] = n_points
    # (chains, draws, k) for the shared ESS helper; SNPE draws are iid -> one nominal chain
    result.samples_by_chain = result.samples.to_numpy()[None, :, :]
    return result
