"""SNPE with the simulator on the GPU: ``wp.fit(lc, model, sampler="snpe_gpu", ...)``.

A thin wrapper around :class:`whisper_cbpf.samplers.snpe.SNPESampler` — that sampler does the
inference, unmodified. What this module supplies is the piece it was missing: a **batched,
device-resident forward model**, built automatically from any whisper model that carries a
``predict_jax``.

What SNPE is doing, and why the observable is shaped the way it is
-----------------------------------------------------------------
SNPE draws parameters from the prior, runs the forward model, adds the data's own noise, and trains
a density estimator ``q(theta | x)`` on those ``(theta, x)`` pairs. Conditioning that estimator on
the REAL observation then gives the posterior — after a single round, with no likelihood.

For that last step to be legitimate, a simulated ``x`` and the observed ``x`` must mean the same
thing coordinate by coordinate. They do here, and by construction: every simulation is evaluated at
the light curve's **own ``(time, band)`` grid, in the data's own row order**, so position *i* of a
simulation is the same epoch in the same filter as position *i* of the observation. Band identity is
carried **positionally**, which is why it is not (and must not be) a channel — see ``x_format`` in
:meth:`SNPESampler.fit`.

What this module adds
---------------------
1. **The GPU simulator.** :func:`make_predict_torch` composes
   :func:`whisper_cbpf.samplers.jax._adapters.make_batched_predict_jax` — which already maps the
   data's band strings to integer indices *outside* the trace, vmaps ``model.predict_jax`` over the
   parameter batch, and evaluates it in chunks — with a zero-copy ``__dlpack__`` handoff to torch.
   Parameters go torch → JAX and fluxes come back JAX → torch **without touching host memory**.
   This replaces a Python ``for`` loop that called the model's numpy ``predict`` one parameter set
   at a time on the CPU.
2. **Registration.** ``snpe_gpu`` is a real sampler: ``wp.fit(lc, model, sampler="snpe_gpu")`` works
   like every other one. Hard-wiring it to the flare would mean a
   ``predict_torch`` hook with no ``bands`` argument cannot express a photometric model. It can now.
3. **An optional context embedding.** :class:`ContextEmbedding` injects per-point sigma and a band
   one-hot as extra channels *inside* the embedding — i.e. AFTER sbi's standardizing layer, where a
   constant channel survives instead of being z-scored to exactly 0.0. Pass one as
   ``embedding_net=``; a NAME (``"mlp"``/``"tcn"``) builds the same network ``snpe`` builds, so a
   CPU/GPU pair conditions on the same channels.

**Why not ``x_format="stacked"``.** SNPESampler's stacked layout appends (error, time) to the raw
input vector. For a single-object fit those channels are *constant* across simulations, and sbi
composes ``nn.Sequential(standardizing_net, embedding_net)`` — so its z-scoring maps them to 0.0
before the network sees them (sbi even warns: "Data has constant values in dimension(s) ..."). On a
200-point flare curve the stacked layout was 4.6x WORSE on the worst parameter and 40% slower:

    stacked, 2k sims/2 rounds  -> worst |median - truth| = 6.61  (131 s)
    value,   2k sims/2 rounds  -> worst |median - truth| = 1.43  ( 92 s)

So the default here is ``x_format="value"``; context, if wanted, goes inside the embedding instead.

**float64.** The TDE and all twelve supernovae REQUIRE it — their engines raise (TDE) or overflow
(supernovae, cgs luminosities of 1e43-1e46 erg/s against float32's 3.4e38 ceiling) at the first
call. JAX's x64 flag must be set before the first array is created, which is before this module can
reach it, so set it yourself::

    import jax; jax.config.update("jax_enable_x64", True)

The dtype the parameters are handed across in follows that flag, so the handoff stays zero-copy in
either precision.

**Per-observation, not amortized**: every light curve gets its own freshly-trained network, so the
full simulate+train+sample cost counts toward that object's wall clock. Real survey data varies in
cadence, noise and band coverage, so one network cannot be reused across curves.
"""
from __future__ import annotations


import numpy as np
import torch
from torch import nn

from ...embeddings import build_embedding
from ...models import get_model
from ...samplers.base import BaseSampler, _warn_user
from ...samplers.snpe import SNPESampler, _resolve_device

LSST_BANDS = ("u", "g", "r", "i", "z", "y")

#: Parameter sets per compiled forward-model call. SNPE's batch IS a simulation count (1e3-1e5), so
#: chunking is mandatory here rather than optional (``_adapters.MAX_UNCHUNKED_BATCH`` refuses an
#: unchunked vmap at or above 64 outright); ``sim_chunk=None`` disables it and is only safe below
#: 64 simulations. It was 16, for compile time; re-measured (current models, one A6000, float64,
#: 1000 simulations per call, the chunked map compiled once), the first call
#: (compile included) takes 1.3-2.5 s at 250 against 1.4-3.9 s at 16, and every later call pays a
#: latency-bound model's per-call cost once per block: the TDE at ``n_time=5000`` 0.73 s at 250
#: against 2.25 s at 16, an arnett supernova 0.56 against 0.78 s, the kilonovae and the TDE at 500
#: 0.15-0.29 against 0.19-0.36 s. 250, as ``abc_gpu.DEFAULT_CHUNK`` (whose measurements show why a
#: power of two such as 256 is avoided).
DEFAULT_SIM_CHUNK = 250


def make_predict_torch(lc, model, *, names=None, times=None, chunk=DEFAULT_SIM_CHUNK, fixed=None):
    """Build the batched GPU simulator SNPE's ``predict_torch=`` hook takes.

    Returns ``predict_torch(theta, times=None, bands=None) -> flux``, mapping a ``(B, D)`` torch
    parameter tensor to a ``(B, n)`` torch **flux** tensor (Jy) evaluated on the light curve's own
    epochs, on the device the computation ran on.

    ``names`` is the order of ``theta``'s columns. ``fixed`` (``{name: value}``) holds parameters
    that are not columns of ``theta`` -- a prior's ``Fixed`` ones, which SNPE does not sample -- at
    their values: they are appended to every row, in the call's dtype, before the model runs.

    Examples
    --------
    >>> import numpy as np, torch
    >>> import whisper_cbpf as wp
    >>> from whisper_cbpf.samplers.jax.snpe_gpu import make_predict_torch
    >>> flare = wp.get_model("flare_jax")
    >>> t = np.linspace(1.0, 29.0, 10)
    >>> lc = wp.LightCurve(time=t, band=["r"] * 10, flux=np.ones(10), flux_err=np.ones(10))
    >>> sim = make_predict_torch(lc, flare, names=["log_amp", "log_sigma", "log_tau"],
    ...                          fixed={"t0": 8.0}, chunk=None)
    >>> out = sim(torch.tensor([[0.5, 0.3, 1.5]]))
    >>> ref = flare.predict({"log_amp": 0.5, "log_sigma": 0.3, "log_tau": 1.5, "t0": 8.0}, t, None)
    >>> bool(np.allclose(out.cpu().numpy()[0], ref, rtol=1e-5))
    True

    Three arguments, deliberately. The hook's two-argument form is exactly why no photometric model
    could ever use it: without ``bands`` there is no way to say which filter each observation is in.
    ``bands`` is accepted and **validated** rather than used, because the band→index mapping is
    resolved once at build time by
    :func:`whisper_cbpf.samplers.jax._adapters.resolve_band_index` — that lookup is a Python dict
    walk and cannot happen inside a JAX trace. Passing a *different* band array would therefore
    silently evaluate the wrong filters, so it raises instead.

    The transfer in both directions is ``__dlpack__``: torch parameters are adopted by JAX in place
    and the resulting fluxes are adopted back by torch in place, so a round of simulation never
    copies through host memory. When the two frameworks are not on the same device — a CPU-only JAX
    build next to a CUDA torch, which is what a missing ``[gpu]`` extra looks like — the handoff
    falls back to a host round trip rather than failing, and says so: the returned callable carries
    ``.transfer`` (``"dlpack"`` or ``"host"``, set on first call) and ``.device``, which the sampler
    reports as ``info["sim_transfer"]`` / ``info["sim_device"]``.
    """
    import jax
    import jax.numpy as jnp
    import torch

    from ._adapters import make_batched_predict_jax

    fixed = dict(fixed or {})
    cols = list(names) if names is not None else [p for p in model.parameters if p not in fixed]
    batched = make_batched_predict_jax(lc, model, names=cols + list(fixed), times=times,
                                       chunk=None if chunk is None else int(chunk))
    fixed_values = [float(v) for v in fixed.values()]
    # Match the SESSION's precision, not torch's. `float_dtype()` is float64 exactly when
    # jax_enable_x64 is on, and handing a float32 theta to a float64 trace would either silently
    # downcast the physics (TDE, supernovae: wrong) or force a copy (not zero-copy: slow).
    want = torch.float64 if jax.config.jax_enable_x64 else torch.float32
    ref_bands = np.asarray(lc.band)
    ref_times = np.asarray(lc.time if times is None else times, dtype=float)

    def predict_torch(theta, times=None, bands=None):
        # `times` is checked by SHAPE ONLY, deliberately. `_adapters._check_times` also compares
        # VALUES, which is right there and wrong here: SNPESampler hands this hook a float32 copy of
        # the epochs, and on MJD-valued times (~5.8e4) float32 resolves only ~0.004 d -- far coarser
        # than that check's 1e-6 * span tolerance, so a value check would reject the caller's own
        # rounding. The float64 epochs closed over by the adapter are what actually gets evaluated,
        # so this argument cannot change the answer; only its length can be wrong.
        if times is not None and tuple(np.shape(times)) != ref_times.shape:
            raise ValueError(
                f"predict_torch was built for {ref_times.shape[0]} epochs but was given "
                f"{tuple(np.shape(times))}. Rebuild the simulator for the new light curve.")
        if bands is not None:
            b = np.asarray(bands)
            if b.shape != ref_bands.shape or not np.array_equal(b.astype(str),
                                                                ref_bands.astype(str)):
                raise ValueError(
                    "predict_torch was given a band array different from the one this simulator "
                    "was built for. The band -> filter-index lookup is resolved once, outside the "
                    "JAX trace, so a different array would evaluate the wrong filters. Rebuild the "
                    "simulator for the new light curve.")
        t = theta if isinstance(theta, torch.Tensor) else torch.as_tensor(theta)
        t = t.detach().to(want)
        if fixed_values:                        # the Fixed parameters, after the sampled columns
            t = torch.cat([t, torch.tensor(fixed_values, dtype=want, device=t.device)
                           .expand(t.shape[0], -1)], dim=1)
        t = t.contiguous()
        try:
            theta_j = jnp.from_dlpack(t)
            transfer = "dlpack"
        except Exception:                       # jax and torch on different devices / no dlpack
            theta_j = jnp.asarray(t.cpu().numpy())
            transfer = "host"
        flux = batched(theta_j)                                   # (B, n) Jy, on device
        # Scrub here, not in torch: a non-finite flux from an absurd prior draw must not reach the
        # magnitude map, and jnp can do it without leaving the device.
        flux = jnp.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        flux = jax.block_until_ready(flux)
        if transfer == "dlpack":
            try:
                out = torch.from_dlpack(flux)
            except Exception:
                out, transfer = torch.as_tensor(np.asarray(flux)), "host"
        else:
            out = torch.as_tensor(np.asarray(flux))
        if predict_torch.transfer is None:      # record the FIRST call's route; it cannot change
            predict_torch.transfer = transfer
        return out

    predict_torch.transfer = None
    predict_torch.device = str(jax.devices()[0])
    predict_torch.names = list(cols)
    predict_torch.fixed = dict(fixed)
    predict_torch.chunk = None if chunk is None else int(chunk)
    predict_torch.floor_stats = batched.floor_stats     # the floored-point count, updated on every call
    return predict_torch


class ContextEmbedding(nn.Module):
    """Concatenate constant per-point context (sigma, band one-hot) to the varying value channel.

    Input from sbi: ``(batch, n_points)`` — the standardized data-space vector.
    Output: ``(batch, latent_dim)``.

    Internally builds ``(batch, 1 + 1 + n_bands, n_points)``:
      * channel 0   — the value (already z-scored by sbi's standardizing layer)
      * channel 1   — log10(sigma), centred; constant across simulations for this fit
      * channels 2+ — one-hot band code; also constant

    Band is one-hot rather than an integer code because it is categorical — an integer code would
    impose a false ordering (g < r < i) that a linear or convolutional layer reads as a meaningful
    distance. Both context blocks are non-trainable ``buffer``s: they are properties of the
    observation's fixed grid, identical for the observation and every simulation drawn on it.

    Note this context is genuinely uninformative for a *single-object* fit (it is the same for every
    simulation, so it cannot discriminate between them). It is optional for that reason, and off by
    default. It becomes informative the moment the network is asked to generalize across grids.
    """

    def __init__(self, sigma, band_codes, n_bands, spec="mlp", latent_dim=32):
        super().__init__()
        self.n_points = int(len(sigma))
        self.n_bands = int(n_bands)

        # abs() + a floor before the log: in MAGNITUDE space `sigma` is a magnitude error, which a
        # real catalogue can report as 0 for a fixed-error survey row, and log10(0) = -inf would
        # make the whole channel NaN after centring. The flare-only original took log10 of a flux
        # error and never met that case.
        s = np.log10(np.abs(np.asarray(sigma, dtype=np.float64)) + 1e-300)
        s = (s - s.mean()) / (s.std() + 1e-12)          # centre/scale so it can't dominate
        self.register_buffer("sigma_ch",
                             torch.as_tensor(s, dtype=torch.float32).view(1, 1, self.n_points))

        onehot = torch.zeros(self.n_bands, self.n_points, dtype=torch.float32)
        onehot[torch.as_tensor(np.asarray(band_codes), dtype=torch.long),
               torch.arange(self.n_points)] = 1.0
        self.register_buffer("band_ch", onehot.unsqueeze(0))

        self.n_channels = 2 + self.n_bands
        self.inner = build_embedding(spec, n_points=self.n_points,
                                     n_channels=self.n_channels, latent_dim=int(latent_dim))
        self.latent_dim = int(latent_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1).float()
        b = x.shape[0]
        value = x.view(b, 1, self.n_points)
        ctx = torch.cat([self.sigma_ch.expand(b, -1, -1),
                         self.band_ch.expand(b, -1, -1)], dim=1)
        return self.inner(torch.cat([value, ctx], dim=1).flatten(start_dim=1))


def encode_bands(band_array, labels=LSST_BANDS):
    """Map band labels to integer codes against a label list.

    ``labels`` defaults to the six LSST filters. Pass ``None`` to derive it from the array itself
    (sorted unique) — which is what :class:`SNPEGPUSampler` does, because a whisper light curve can
    carry any filter set (AT2017GFO has 29, including HST's ``F160W``) and a fixed list would refuse
    most real data.

    A FIXED list means the channel index of a given filter is the same in every fit; a derived one
    is per-curve. For SNPE that distinction costs nothing — the network is trained per object and
    never reused across curves (see the module docstring) — but pass ``labels=`` explicitly if you
    ever intend to compare embeddings across fits.
    """
    band_array = np.asarray(band_array)
    if labels is None:
        labels = sorted({str(b) for b in band_array})
    lookup = {str(lab): i for i, lab in enumerate(labels)}
    missing = sorted({str(b) for b in band_array} - set(lookup))
    if missing:
        raise ValueError(f"bands {missing} not in the known label list {list(labels)}")
    return np.array([lookup[str(b)] for b in band_array], dtype=int), list(labels)


class SNPEGPUSampler(BaseSampler):
    """SNPE/NPE with the simulator on the GPU. See the module docstring."""

    name = "snpe_gpu"

    def fit(self, lc, model, prior=None, *, predict_torch=None, space="auto", num_rounds=2,
            num_simulations=1000, density_estimator="maf", embedding_net=None, embedding_latent=32,
            num_samples=10000, sim_chunk=DEFAULT_SIM_CHUNK, device="cuda", seed=0,
            show_progress=False, embedding=None, **snpe_kwargs):
        """Fit ``lc`` with ``model`` by SNPE, simulating and training on the GPU.

        Parameters
        ----------
        prior : Prior, optional
            As :meth:`SNPESampler.fit`: ``Uniform``, ``LogUniform``, ``Normal``,
            ``TruncatedNormal``; a ``Fixed`` parameter is held at its value (not sampled, not
            counted in AIC/BIC) and appended to every simulated row by the built simulator.
        predict_torch : callable, optional
            ``predict_torch(theta, times[, bands]) -> flux`` — an explicit batched forward map.
            ``theta`` holds the prior's free parameters only, in the prior's order.
            **Optional**: when omitted it is built from ``model.predict_jax`` by
            :func:`make_predict_torch`, so any model registered through the JAX factories
            (``register_kilonova`` / ``register_kilonova_two`` / ``register_kilonova_three`` /
            ``register_tde`` / ``register_supernova``) just works. An explicitly passed callable
            always wins. A model with **no** ``predict_jax`` raises, naming ``sampler='snpe'`` —
            falling back to the numpy ``predict`` in a Python loop would be the CPU sampler with
            extra steps and a misleading name.
        space : {"auto", "flux", "magnitude"}
            The comparison space, as everywhere else in whisper. ``"auto"`` follows
            ``lc.data_mode``. The simulator returns flux; the flux→space map runs on-device and the
            noise is added AFTER it, so magnitude data is fitted in magnitude space.
        num_rounds : int
            ``2`` (default, as in ``snpe``) is sequential SNPE; the round-to-round proposal draw is
            leakage-guarded (see ``proposal_min_acceptance`` in :meth:`SNPESampler.fit`). ``1`` is
            amortized NPE — one pass, which is all the user-facing promise requires ("able to infer
            after one round on the transient data"). The default used to be 1 here and 2 on the
            CPU, so the same call ran a different algorithm on each arm.
        density_estimator : str or callable
            ``'maf'`` (default), ``'nsf'``, ``'mdn'``, ... or a pre-built ``posterior_nn(...)``
            factory. ``hidden_features`` / ``num_transforms`` / ``num_bins`` pass through.
        embedding_net : {None, "mlp", "tcn"} or torch.nn.Module
            The keyword ``snpe`` takes, with the same meaning, so one call configures both arms.
            ``None`` (default) feeds the data-space vector straight to the flow — measured best at
            this problem size. ``"mlp"`` / ``"tcn"`` build the same network ``snpe`` builds, over
            the ``x_format`` channels. Any ``torch.nn.Module`` is passed through untouched —
            e.g. a :class:`ContextEmbedding`, which adds per-point sigma and a band one-hot as
            channels inside the embedding (it reads the value-only input, so it needs
            ``x_format="value"``, and any other layout is refused before simulating).
        embedding : deprecated
            The old spelling of ``embedding_net`` (``"none"`` meaning ``None``). It warns, and its
            ``"mlp"`` / ``"tcn"`` no longer imply a :class:`ContextEmbedding`.
        sim_chunk : int or None
            Parameter sets per compiled forward-model call. See :data:`DEFAULT_SIM_CHUNK`.
        device : str
            ``'cuda'`` (default), ``'cuda:N'``, ``'gpu'``, ``'auto'`` or ``'cpu'``. This selects the
            **torch** device (training, the noise draw, posterior sampling). JAX's device is chosen
            by JAX, from ``CUDA_VISIBLE_DEVICES`` — keep them on the same card or the zero-copy
            handoff degrades to a host round trip, which is reported rather than hidden
            (``info["sim_transfer"]``).

        Extra keyword arguments pass through to :meth:`SNPESampler.fit` (``x_format``,
        ``scatter_param``, ``standardize_x``, ``num_chains``, ``max_num_epochs``, ...).
        """
        if embedding is not None:
            if embedding_net is not None:
                raise TypeError("snpe_gpu: pass embedding_net= only; embedding= is its deprecated "
                                "alias.")
            _warn_user(
                "snpe_gpu: embedding= is deprecated; use embedding_net=, the keyword snpe takes. "
                "'mlp'/'tcn' now build the same network as on snpe, not a ContextEmbedding; pass "
                "embedding_net=ContextEmbedding(...) for that.", DeprecationWarning)
            embedding_net = None if embedding == "none" else embedding
        # A ContextEmbedding reads the value-only vector (it adds sigma and the band itself), so
        # the stacked layout's 3n inputs raised "shape '[1, 1, n]' is invalid for input of size 3n"
        # at the first training batch, after the simulations had been paid for. Refuse it here.
        if (isinstance(embedding_net, ContextEmbedding)
                and snpe_kwargs.get("x_format", "value") != "value"):
            raise ValueError(
                f"snpe_gpu: a ContextEmbedding conditions on the value-only input and adds sigma "
                f"and the band itself; use x_format='value' (the default), not "
                f"{snpe_kwargs['x_format']!r}.")
        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")

        sim_source = "caller"
        if predict_torch is None:
            if getattr(model, "predict_jax", None) is None:
                raise ValueError(
                    f"snpe_gpu needs a batched, device-resident forward model, and model "
                    f"{model.name!r} carries no predict_jax to build one from. Register it through "
                    f"a JAX factory (register_kilonova / register_kilonova_two / "
                    f"register_kilonova_three / register_tde / register_supernova), pass "
                    f"predict_torch= yourself, or use sampler='snpe', whose CPU simulator works "
                    f"from the numpy predict.")
            # names=the prior's free parameters: the adapter reorders into model.parameters order
            # internally and DROPS a prior-only column, which is exactly the scatter_param case --
            # that parameter is a likelihood term and must never reach the physics. SNPE samples
            # the free parameters only; a Fixed one is appended at its value.
            fixed = dict(getattr(prior, "fixed", {}) or {})
            predict_torch = make_predict_torch(
                lc, model, names=[nm for nm in prior.names if nm not in fixed], chunk=sim_chunk,
                fixed=fixed)
            sim_source = "auto"

        resolved = _resolve_device(device, torch)
        x_format = snpe_kwargs.pop("x_format", "value")         # see the module docstring
        result = SNPESampler().fit(
            lc, model, prior=prior,
            num_rounds=int(num_rounds), num_simulations=int(num_simulations),
            space=space,
            density_estimator=density_estimator,
            embedding_net=embedding_net, embedding_latent=int(embedding_latent),
            x_format=x_format,
            predict_torch=predict_torch,
            num_samples=int(num_samples),
            device=resolved, seed=int(seed), show_progress=show_progress,
            **snpe_kwargs,
        )
        result.sampler = "snpe_gpu"
        result.info["predict_torch"] = sim_source
        transfer = getattr(predict_torch, "transfer", None)
        result.info["sim_transfer"] = transfer
        result.info["sim_device"] = getattr(predict_torch, "device", None)
        result.info["sim_chunk"] = getattr(predict_torch, "chunk", None)
        fs = getattr(predict_torch, "floor_stats", None)   # a caller's map has none
        result.info["mag_floor_stats"] = dict(fs) if fs else None
        # "jax-dlpack-torch" is the claim the docs make: simulation ran in JAX and the result was
        # adopted by torch in place. Say "jax-host-torch" when it did not, rather than letting a
        # silent host round trip be reported as the GPU path.
        result.info["sim_backend"] = ("jax-dlpack-torch" if transfer == "dlpack"
                                      else "jax-host-torch" if transfer == "host"
                                      else "torch")
        result.info["n_points"] = int(len(lc.time))
        try:
            import jax
            result.info["x64"] = bool(jax.config.jax_enable_x64)
        except Exception:                                        # pragma: no cover - jax required
            pass
        # (chains, draws, k) for the shared ESS helper; SNPE draws are iid -> one nominal chain
        result.samples_by_chain = result.samples[list(prior.names)].to_numpy()[None, :, :]
        return result


def fit_snpe_gpu(lc, model, *, prior=None, num_rounds=2, num_simulations=10000,
                 embedding_net=None, embedding_latent=32, density_estimator="maf",
                 num_samples=10000, seed=0, device="cuda", show_progress=False,
                 embedding=None, **snpe_kwargs):
    """Fit one light curve with SNPE on GPU. See :meth:`SNPEGPUSampler.fit`.

    Kept as a direct entry point because it predates registration; ``wp.fit(lc, model,
    sampler="snpe_gpu", ...)`` is the same call. It is **no longer flare-only** — the ``bands``
    argument the ``predict_torch`` hook needs is what would otherwise make that restriction
    unavoidable.
    """
    return SNPEGPUSampler().fit(
        lc, model, prior=prior, num_rounds=num_rounds, num_simulations=num_simulations,
        embedding_net=embedding_net, embedding_latent=embedding_latent,
        density_estimator=density_estimator, num_samples=num_samples, seed=seed,
        device=device, show_progress=show_progress, embedding=embedding, **snpe_kwargs)
