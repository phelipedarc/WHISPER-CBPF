"""Embedding networks for neural SBI — learned compression of the light-curve input.

An embedding net maps the (possibly multi-channel) light-curve vector to a low-dimensional feature
vector *before* it conditions the density estimator. It is trained **jointly** with the estimator by
sbi, so the features are optimized for constraining the parameters. Two built-ins:

* :class:`MLPEmbedding` — a plain fully-connected compressor; cheap, structure-agnostic.
* :class:`TCNEmbedding` — a Temporal Convolutional Network (dilated causal 1-D convolutions with
  residual blocks; Bai et al. 2018): specialized for sequential data, its receptive field grows
  exponentially with depth so it sees the whole light curve while staying lightweight.

Both accept the flat ``(batch, n_channels * n_points)`` input sbi passes (the TCN reshapes it to
``(batch, n_channels, n_points)`` internally) and return ``(batch, latent_dim)`` features. Use them
via ``fit_SNPE(..., embedding_net="mlp" | "tcn")`` (built through :func:`build_embedding`) or pass any
custom ``torch.nn.Module``.

``torch`` is an optional dependency (the ``[sbi]`` extra) — import this module lazily.
"""
from __future__ import annotations

import torch
from torch import nn


class MLPEmbedding(nn.Module):
    """Fully-connected embedding: ``n_inputs -> hidden -> ... -> latent_dim`` (ReLU between layers)."""

    def __init__(self, n_inputs, latent_dim=32, hidden=(128, 128)):
        super().__init__()
        dims = [int(n_inputs), *[int(h) for h in hidden], int(latent_dim)]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers[:-1])          # no activation on the latent output
        self.latent_dim = int(latent_dim)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1).float())


class _TCNBlock(nn.Module):
    """One residual TCN block: two dilated causal convolutions + ReLU, with a 1×1 skip if needed."""

    def __init__(self, c_in, c_out, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # left-pad -> causal (no future leakage)
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, dilation=dilation)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.act(self.conv2(nn.functional.pad(y, (self.pad, 0))))
        return self.act(y + self.skip(x))


class TCNEmbedding(nn.Module):
    """Temporal Convolutional Network embedding for light curves.

    Stacked dilated causal convolutions (dilation 1, 2, 4, …) give a receptive field of
    ``1 + 2·(kernel_size−1)·(2^levels − 1)`` samples — with the defaults (kernel 5, 4 levels) that is
    121 points, covering a typical light curve — followed by global average+max pooling over time and
    a linear head to ``latent_dim`` features.

    ``forward`` accepts the flat ``(batch, n_channels * n_points)`` tensor sbi provides and reshapes
    it to ``(batch, n_channels, n_points)``.
    """

    def __init__(self, n_points, n_channels=1, latent_dim=32, hidden_channels=32,
                 levels=4, kernel_size=5):
        super().__init__()
        self.n_points, self.n_channels = int(n_points), int(n_channels)
        blocks, c_in = [], self.n_channels
        for lv in range(int(levels)):
            blocks.append(_TCNBlock(c_in, int(hidden_channels), int(kernel_size), dilation=2 ** lv))
            c_in = int(hidden_channels)
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(2 * int(hidden_channels), int(latent_dim))   # cat(avg-pool, max-pool)
        self.latent_dim = int(latent_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1).float().view(-1, self.n_channels, self.n_points)
        y = self.tcn(x)
        pooled = torch.cat([y.mean(dim=-1), y.max(dim=-1).values], dim=1)
        return self.head(pooled)


def build_embedding(spec, n_points, n_channels=1, latent_dim=32):
    """Build a built-in embedding net by name: ``"mlp"`` or ``"tcn"`` (see the module docstring)."""
    spec = str(spec).lower()
    if spec == "mlp":
        return MLPEmbedding(n_points * n_channels, latent_dim=latent_dim)
    if spec == "tcn":
        return TCNEmbedding(n_points, n_channels=n_channels, latent_dim=latent_dim)
    raise ValueError(f"Unknown embedding {spec!r}; use 'mlp', 'tcn', or pass a torch.nn.Module.")
