# PROVENANCE

Every file in this package, mapped to the repo, path and commit it was copied from.
Written during migration, not reconstructed afterwards.

## Source checkouts

| Repo | URL | Commit |
|---|---|---|
| `WHISPER_AI` | https://github.com/phelipedarc/WHISPER_AI | `8ba3843` |
| `whisper-GPU` | https://github.com/phelipedarc/whisper-GPU | `10796a0` |

Both source repos are frozen legacy and were never written to.

## Files

`sha256` is the source file's digest at copy time. Every row below is **byte-identical** to its
source at that commit; import rewrites land in later commits and are measured against this baseline.

| Destination | Source repo | Original path | Source commit | sha256 (12) |
|---|---|---|---|---|
| `CITATION.cff` | WHISPER_AI | `CITATION.cff` | `8ba3843` | `ea746de54d4f` |
| `CONTRIBUTING.md` | WHISPER_AI | `CONTRIBUTING.md` | `8ba3843` | `fe7be516df3a` |
| `LICENSE` | WHISPER_AI | `LICENSE` | `8ba3843` | `3972dc9744f6` |
| `scripts/whisper-gpu-env.sh` | whisper-GPU | `env.sh` | `10796a0` | `dab8144ca476` |
| `whisper_cbpf/__init__.py` | WHISPER_AI | `whisper_labia/__init__.py` | `8ba3843` | `90a930a16c22` |
| `whisper_cbpf/backends/_env.py` | whisper-GPU | `_env.py` | `10796a0` | `7d9fc63838d9` |
| `whisper_cbpf/backends/_registration.py` | whisper-GPU | `__init__.py` | `10796a0` | `a64777580ed6` |
| `whisper_cbpf/distance/_gpu_distances.py` | whisper-GPU | `distances.py` | `10796a0` | `e0471dea0711` |
| `whisper_cbpf/distance/_numpy_cpu_chi2.py` | WHISPER_AI | `whisper_labia/distance.py` | `8ba3843` | `5eb11dd79a20` |
| `whisper_cbpf/embeddings.py` | WHISPER_AI | `whisper_labia/embeddings.py` | `8ba3843` | `092db6cb2675` |
| `whisper_cbpf/io/__init__.py` | WHISPER_AI | `whisper_labia/io/__init__.py` | `8ba3843` | `3b0e9e478af0` |
| `whisper_cbpf/io/bands.py` | WHISPER_AI | `whisper_labia/io/bands.py` | `8ba3843` | `2b668d654176` |
| `whisper_cbpf/io/loader.py` | WHISPER_AI | `whisper_labia/io/loader.py` | `8ba3843` | `df9538453177` |
| `whisper_cbpf/io/photometry.py` | WHISPER_AI | `whisper_labia/io/photometry.py` | `8ba3843` | `ed8a897c7273` |
| `whisper_cbpf/io/schema.py` | WHISPER_AI | `whisper_labia/io/schema.py` | `8ba3843` | `a6446580ed63` |
| `whisper_cbpf/io/svo.py` | WHISPER_AI | `whisper_labia/io/svo.py` | `8ba3843` | `f2a7527e61c9` |
| `whisper_cbpf/io/units.py` | WHISPER_AI | `whisper_labia/io/units.py` | `8ba3843` | `d25a57e680fc` |
| `whisper_cbpf/likelihood.py` | WHISPER_AI | `whisper_labia/likelihood.py` | `8ba3843` | `f2e388ed5a6e` |
| `whisper_cbpf/metrics/_jax.py` | whisper-GPU | `metrics.py` | `10796a0` | `5f7ad83ae36e` |
| `whisper_cbpf/metrics/_numpy.py` | WHISPER_AI | `whisper_labia/metrics.py` | `8ba3843` | `a07731b134e5` |
| `whisper_cbpf/models/__init__.py` | WHISPER_AI | `whisper_labia/models/__init__.py` | `8ba3843` | `3552077b5e08` |
| `whisper_cbpf/models/bazin.py` | WHISPER_AI | `whisper_labia/models/bazin.py` | `8ba3843` | `486db2afd041` |
| `whisper_cbpf/models/flare.py` | WHISPER_AI | `whisper_labia/models/flare.py` | `8ba3843` | `402d067a3a13` |
| `whisper_cbpf/models/gaussian_rise.py` | WHISPER_AI | `whisper_labia/models/gaussian_rise.py` | `8ba3843` | `43451346d76d` |
| `whisper_cbpf/models/jax/_factories.py` | whisper-GPU | `models/__init__.py` | `10796a0` | `283f17fca627` |
| `whisper_cbpf/models/jax/_flare_spec.py` | whisper-GPU | `models/_flare_spec.py` | `10796a0` | `77f62265ebe9` |
| `whisper_cbpf/models/jax/flare.py` | whisper-GPU | `models/flare.py` | `10796a0` | `9051068f586e` |
| `whisper_cbpf/models/jax/kilonova.py` | whisper-GPU | `models/kilonova.py` | `10796a0` | `550686d90fc2` |
| `whisper_cbpf/models/jax/kilonova_two.py` | whisper-GPU | `models/kilonova_two.py` | `10796a0` | `95fbce628a56` |
| `whisper_cbpf/models/jax/supernova.py` | whisper-GPU | `models/supernova.py` | `10796a0` | `524e79fd14eb` |
| `whisper_cbpf/models/jax/tde.py` | whisper-GPU | `models/tde.py` | `10796a0` | `4257dd0f2926` |
| `whisper_cbpf/models/mck19.py` | WHISPER_AI | `whisper_labia/models/mck19.py` | `8ba3843` | `38186b87a7e4` |
| `whisper_cbpf/models/two_component_kilonova.py` | WHISPER_AI | `whisper_labia/models/two_component_kilonova.py` | `8ba3843` | `c1324bb64db3` |
| `whisper_cbpf/plotting.py` | WHISPER_AI | `whisper_labia/plotting.py` | `8ba3843` | `50511319d629` |
| `whisper_cbpf/priors.py` | WHISPER_AI | `whisper_labia/priors.py` | `8ba3843` | `021a7cd8c945` |
| `whisper_cbpf/samplers/__init__.py` | WHISPER_AI | `whisper_labia/samplers/__init__.py` | `8ba3843` | `5e41f631928f` |
| `whisper_cbpf/samplers/abc.py` | WHISPER_AI | `whisper_labia/samplers/abc.py` | `8ba3843` | `e983cdf50b61` |
| `whisper_cbpf/samplers/abc_smc.py` | WHISPER_AI | `whisper_labia/samplers/abc_smc.py` | `8ba3843` | `400499d229a9` |
| `whisper_cbpf/samplers/base.py` | WHISPER_AI | `whisper_labia/samplers/base.py` | `8ba3843` | `45dcb6b571e7` |
| `whisper_cbpf/samplers/jax/abc_gpu.py` | whisper-GPU | `samplers/abc_gpu.py` | `10796a0` | `be65774ff761` |
| `whisper_cbpf/samplers/jax/abc_smc_gpu.py` | whisper-GPU | `samplers/abc_smc_gpu.py` | `10796a0` | `bd760effa31a` |
| `whisper_cbpf/samplers/jax/emcee_jax.py` | whisper-GPU | `samplers/emcee_jax.py` | `10796a0` | `bc79229f4201` |
| `whisper_cbpf/samplers/jax/nuts_gpu.py` | whisper-GPU | `samplers/nuts_gpu.py` | `10796a0` | `244072ee2155` |
| `whisper_cbpf/samplers/jax/pymc_gpu.py` | whisper-GPU | `samplers/pymc_gpu.py` | `10796a0` | `8d2e92db9828` |
| `whisper_cbpf/samplers/jax/snpe_gpu.py` | whisper-GPU | `samplers/snpe_gpu.py` | `10796a0` | `8eb3fe76f286` |
| `whisper_cbpf/samplers/mcmc.py` | WHISPER_AI | `whisper_labia/samplers/mcmc.py` | `8ba3843` | `f608d6904c6c` |
| `whisper_cbpf/samplers/snpe.py` | WHISPER_AI | `whisper_labia/samplers/snpe.py` | `8ba3843` | `12e1902281c1` |
| `whisper_cbpf/validation.py` | WHISPER_AI | `whisper_labia/validation.py` | `8ba3843` | `a864918017b0` |

## Files created during the merge

Not copied from either source. Every one is registry glue, a re-export layer, or a split of a
copied file — no new numerics. Where a body came from a source file it is noted.

| File | What it is | Derived from |
|---|---|---|
| `PROVENANCE.md` | this file | — |
| `whisper_cbpf/distance/_numpy.py` | numpy implementations, no registry | `chi2_distance` verbatim from WHISPER_AI `whisper_labia/distance.py` @ `8ba3843`; `mse/rmse/mae/wmse/wmae_distance` verbatim from whisper-GPU `distances.py` @ `10796a0` |
| `whisper_cbpf/distance/_jax.py` | jnp implementations | `jnp_distance` verbatim from whisper-GPU `distances.py` @ `10796a0` |
| `whisper_cbpf/distance/__init__.py` | registry + re-exports, numpy default, `_jax` lazy | `register_distance`/`get_distance`/`list_distances` verbatim from WHISPER_AI `whisper_labia/distance.py` @ `8ba3843`, extended with `*_jax` lazy resolution |
| `whisper_cbpf/metrics/__init__.py` | re-export layer, `_jax` slot lazy | new; re-exports only |

### One duplicate collapsed — recorded, not silent

whisper-GPU's `distances.py` defined its own `chi2_distance` alongside WHISPER_AI's. Only the
WHISPER_AI one survives in `_numpy.py`, per the brief's prescribed `distance/` layout
("chi2 (from CPU) + mse rmse mae wmse wmae (from GPU)"). Evidence that this loses nothing: the
whisper-GPU copy's own docstring reads *"identical to whisper's built-in, re-exported for
symmetry"* (`distances.py:50` @ `10796a0`). Both remain in this repo's history in commit `3b12c1b`.
See `DUPLICATES.md`.

### `register_into_whisper` is intentionally not carried across

whisper-GPU's `distances.py` shipped `register_into_whisper()` to push its five numpy metrics into
whisper's registry from outside. After the merge those metrics are core and register normally in
`distance/__init__.py`, so the function has no remaining job. The CPU ABC still gains all five,
which is what keeps a CPU/GPU comparison at a fixed distance possible.

## Tests

Byte-identical at copy time; imports rewritten in the next commit.

| Destination | Source repo | Original path | Source commit | sha256 (12) |
|---|---|---|---|---|
| `tests/conftest.py` | WHISPER_AI | `tests/conftest.py` | `8ba3843` | `438f7b32f058` |
| `tests/data/at2017gfo.csv` | WHISPER_AI | `tests/data/at2017gfo.csv` | `8ba3843` | `19b610026c54` |
| `tests/data/ztf18aarlhfw.csv` | WHISPER_AI | `tests/data/ztf18aarlhfw.csv` | `8ba3843` | `3a3bf31522af` |
| `tests/goldens/filterset_t0.npz` | whisper-GPU | `tests/goldens/filterset_t0.npz` | `10796a0` | `45a62d1c80c1` |
| `tests/goldens/kilonova_one_component.npz` | whisper-GPU | `tests/goldens/kilonova_one_component.npz` | `10796a0` | `f4473d681248` |
| `tests/goldens/tde_cooling_envelope.npz` | whisper-GPU | `tests/goldens/tde_cooling_envelope.npz` | `10796a0` | `edbf426a7df2` |
| `tests/goldens/tde_cooling_envelope_magnitude.npz` | whisper-GPU | `tests/goldens/tde_cooling_envelope_magnitude.npz` | `10796a0` | `33305424b5c1` |
| `tests/t0_parity/_f32_probe.py` | whisper-GPU | `tests/t0_parity/_f32_probe.py` | `10796a0` | `45a8a281959a` |
| `tests/t0_parity/make_goldens.py` | whisper-GPU | `tests/t0_parity/make_goldens.py` | `10796a0` | `dbb3bacb3ab9` |
| `tests/t0_parity/test_conventions.py` | whisper-GPU | `tests/t0_parity/test_conventions.py` | `10796a0` | `6827f90c89a7` |
| `tests/t0_parity/test_goldens.py` | whisper-GPU | `tests/t0_parity/test_goldens.py` | `10796a0` | `750b431d3d15` |
| `tests/t2_autodiff/test_gradients.py` | whisper-GPU | `tests/t2_autodiff/test_gradients.py` | `10796a0` | `bb07dc97d720` |
| `tests/t3_inference/_stage_common.py` | whisper-GPU | `tests/t3_inference/_stage_common.py` | `10796a0` | `657a6f15a2f7` |
| `tests/t3_inference/_t3_common.py` | whisper-GPU | `tests/t3_inference/_t3_common.py` | `10796a0` | `bf1ec31b72fc` |
| `tests/t3_inference/stage_injection_recovery.py` | whisper-GPU | `tests/t3_inference/stage_injection_recovery.py` | `10796a0` | `b457399a1658` |
| `tests/t3_inference/stage_sbc.py` | whisper-GPU | `tests/t3_inference/stage_sbc.py` | `10796a0` | `991aee8c3826` |
| `tests/t3_inference/test_continuity.py` | whisper-GPU | `tests/t3_inference/test_continuity.py` | `10796a0` | `1bf403b10a76` |
| `tests/t3_inference/test_likelihood_parity.py` | whisper-GPU | `tests/t3_inference/test_likelihood_parity.py` | `10796a0` | `6d7d7779f025` |
| `tests/t3_inference/test_prior_predictive.py` | whisper-GPU | `tests/t3_inference/test_prior_predictive.py` | `10796a0` | `71bf4272d036` |
| `tests/test_abc.py` | WHISPER_AI | `tests/test_abc.py` | `8ba3843` | `cfeb41659707` |
| `tests/test_abc_gpu_scan.py` | whisper-GPU | `tests/test_abc_gpu_scan.py` | `10796a0` | `0f27d6f4c09c` |
| `tests/test_abc_smc.py` | WHISPER_AI | `tests/test_abc_smc.py` | `8ba3843` | `e3ab86812092` |
| `tests/test_abc_smc_gpu.py` | whisper-GPU | `tests/test_abc_smc_gpu.py` | `10796a0` | `e73714330b23` |
| `tests/test_bands.py` | WHISPER_AI | `tests/test_bands.py` | `8ba3843` | `07665066b54e` |
| `tests/test_benchmark_kilonova.py` | WHISPER_AI | `tests/test_benchmark_kilonova.py` | `8ba3843` | `1508562de5f4` |
| `tests/test_distance.py` | WHISPER_AI | `tests/test_distance.py` | `8ba3843` | `3a1b14fca2b8` |
| `tests/test_embeddings.py` | WHISPER_AI | `tests/test_embeddings.py` | `8ba3843` | `652abc338088` |
| `tests/test_flare_gradients.py` | whisper-GPU | `tests/test_flare_gradients.py` | `10796a0` | `750a504e6d8d` |
| `tests/test_import_and_registry.py` | whisper-GPU | `tests/test_import_and_registry.py` | `10796a0` | `517048ddc79d` |
| `tests/test_ingestion.py` | WHISPER_AI | `tests/test_ingestion.py` | `8ba3843` | `f5b008cde2f8` |
| `tests/test_kilonova_two.py` | whisper-GPU | `tests/test_kilonova_two.py` | `10796a0` | `dec26e741af2` |
| `tests/test_kilonova_vs_redback.py` | whisper-GPU | `tests/test_kilonova_vs_redback.py` | `10796a0` | `f8141c666a4b` |
| `tests/test_likelihood.py` | WHISPER_AI | `tests/test_likelihood.py` | `8ba3843` | `7098340dc888` |
| `tests/test_loader.py` | WHISPER_AI | `tests/test_loader.py` | `8ba3843` | `6c614efa9140` |
| `tests/test_mck19.py` | WHISPER_AI | `tests/test_mck19.py` | `8ba3843` | `5dcc32ada6bc` |
| `tests/test_mcmc.py` | WHISPER_AI | `tests/test_mcmc.py` | `8ba3843` | `63347f7e7ec9` |
| `tests/test_metrics.py` | WHISPER_AI | `tests/test_metrics.py` | `8ba3843` | `07bc1989a9cf` |
| `tests/test_models.py` | WHISPER_AI | `tests/test_models.py` | `8ba3843` | `3801e5f9e452` |
| `tests/test_photometry.py` | WHISPER_AI | `tests/test_photometry.py` | `8ba3843` | `3bef7d93472c` |
| `tests/test_plotting.py` | WHISPER_AI | `tests/test_plotting.py` | `8ba3843` | `b75e2028ebd8` |
| `tests/test_priors.py` | WHISPER_AI | `tests/test_priors.py` | `8ba3843` | `db5480cdfa32` |
| `tests/test_pymc_chain_method.py` | whisper-GPU | `tests/test_pymc_chain_method.py` | `10796a0` | `1587229ab499` |
| `tests/test_registries.py` | WHISPER_AI | `tests/test_registries.py` | `8ba3843` | `74f1ed3c8cad` |
| `tests/test_review_fixes.py` | WHISPER_AI | `tests/test_review_fixes.py` | `8ba3843` | `73694d5cc018` |
| `tests/test_scatter.py` | WHISPER_AI | `tests/test_scatter.py` | `8ba3843` | `12507ad0f461` |
| `tests/test_schema.py` | WHISPER_AI | `tests/test_schema.py` | `8ba3843` | `6c15e932b954` |
| `tests/test_snpe.py` | WHISPER_AI | `tests/test_snpe.py` | `8ba3843` | `38e6b104070a` |
| `tests/test_supernova_vs_redback.py` | whisper-GPU | `tests/test_supernova_vs_redback.py` | `10796a0` | `8f112ebde3ce` |
| `tests/test_svo.py` | WHISPER_AI | `tests/test_svo.py` | `8ba3843` | `42aad6396ffb` |
| `tests/test_table_lc.py` | WHISPER_AI | `tests/test_table_lc.py` | `8ba3843` | `3ec9268b8ee9` |
| `tests/test_tde_vs_redback.py` | whisper-GPU | `tests/test_tde_vs_redback.py` | `10796a0` | `4c936b677c58` |
| `tests/test_two_component_kilonova.py` | WHISPER_AI | `tests/test_two_component_kilonova.py` | `8ba3843` | `658d89f91567` |
| `tests/test_units.py` | WHISPER_AI | `tests/test_units.py` | `8ba3843` | `1d41b6b25de0` |
| `tests/test_validation.py` | WHISPER_AI | `tests/test_validation.py` | `8ba3843` | `af31239311f6` |

**Excluded by the functionality-only rule** (run output, not fixtures): `tests/t3_inference/sbc_results/` (27 files), `tests/t3_inference/injection_results/` (11), and the three tier `REPORT.md` files. Nothing reads them -- `stage_sbc.py` and `stage_injection_recovery.py` treat both directories as their `DEFAULT_OUT` and recreate them on run. `tests/goldens/*.npz` ARE kept: the t0 parity tests assert against them.

## Packaging files created during the merge

| File | What it is | Derived from |
|---|---|---|
| `pyproject.toml` | one build config for the merged package | WHISPER_AI `pyproject.toml` @ `8ba3843`, renamed to whisper-cbpf 0.1.0, plus the [gpu] and [pymc] extras |
| `conftest.py` | root pytest config | new; skips collection of JAX-import-at-module-scope tests when JAX is absent |
| `whisper_cbpf/py.typed` | typing marker | WHISPER_AI `whisper_labia/py.typed` @ `8ba3843` (byte-identical) |

`MANIFEST.in` is deliberately absent: the package reads no data files at runtime. The `.prior`
references in the JAX models are docstrings citing redback's own installed prior files, which come
from the `[models]` extra, not from this package. Only `py.typed` needs shipping, and
`[tool.setuptools.package-data]` covers it.

## Documentation, examples and notebooks

Byte-identical at copy time; imports rewritten in the next commit. Notebook **source cells only** are rewritten -- stored outputs are never touched and nothing is re-executed.

| Destination | Source repo | Original path | Source commit | sha256 (12) |
|---|---|---|---|---|
| `CHANGELOG.md` | WHISPER_AI | `CHANGELOG.md` | `8ba3843` | `ab7e0b03e421` |
| `docs/API_REFERENCE.md` | WHISPER_AI | `docs/API_REFERENCE.md` | `8ba3843` | `9bd2d8ec74cb` |
| `docs/DESIGN.md` | WHISPER_AI | `docs/DESIGN.md` | `8ba3843` | `79fd9b46244e` |
| `docs/EXTENDING.md` | WHISPER_AI | `docs/EXTENDING.md` | `8ba3843` | `ac34725d2ea7` |
| `docs/PHYSICS_NOTES.md` | whisper-GPU | `PHYSICS_NOTES.md` | `10796a0` | `a1b1a6e0aae0` |
| `docs/REPORT_at2017gfo.md` | WHISPER_AI | `docs/REPORT_at2017gfo.md` | `8ba3843` | `60fcb2a91118` |
| `docs/REPORT_ingestion_upgrade.md` | WHISPER_AI | `docs/REPORT_ingestion_upgrade.md` | `8ba3843` | `1b62a0378770` |
| `docs/TUTORIAL.md` | WHISPER_AI | `docs/TUTORIAL.md` | `8ba3843` | `6c149d7a0d59` |
| `examples/at2017gfo_quickstart.ipynb` | WHISPER_AI | `examples/at2017gfo_quickstart.ipynb` | `8ba3843` | `33d80d1a55a1` |
| `notebooks/01_lightcurves.ipynb` | WHISPER_AI | `notebooks/01_lightcurves.ipynb` | `8ba3843` | `4f650020eaa1` |
| `notebooks/02_models.ipynb` | WHISPER_AI | `notebooks/02_models.ipynb` | `8ba3843` | `8be6de455325` |
| `notebooks/03_samplers.ipynb` | WHISPER_AI | `notebooks/03_samplers.ipynb` | `8ba3843` | `df82e7bb7f9a` |
| `notebooks/04_visualizing.ipynb` | WHISPER_AI | `notebooks/04_visualizing.ipynb` | `8ba3843` | `32dd5ddb2f80` |
| `notebooks/05_bayesian_model_comparison.ipynb` | WHISPER_AI | `notebooks/05_bayesian_model_comparison.ipynb` | `8ba3843` | `281da3988b1a` |
| `notebooks/README.md` | WHISPER_AI | `notebooks/README.md` | `8ba3843` | `a92778e0e780` |
| `notebooks/_build_notebooks.py` | WHISPER_AI | `notebooks/_build_notebooks.py` | `8ba3843` | `07e8b60407fb` |
