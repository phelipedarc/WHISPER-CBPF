# Contributing to Whisper (`whisper_labia`)

Thanks for helping improve Whisper. This guide is the entry point for contributors — you should be able
to set up, test, and extend the package without reading the source.

## Development setup

Whisper is developed and tested **inside Docker** (the project does not install onto the host):

```bash
# from the repo root, inside the dev container (e.g. `docker exec <container> bash`)
pip install -e ".[dev]"          # editable install + pytest/coverage/build/astroquery
pip install -e ".[sbi]"          # optional: enables the SNPE/NPE sampler (sbi + torch)
python -m pytest tests -q        # run the suite
python -m pytest tests -q -m "not slow"   # skip the SNPE neural-net training tests
```

The core package (ingestion, plotting, ABC/ABC-SMC) needs **no compiler and no redback**. Optional
extras: `[models]` (redback physical models + priors), `[svo]` (SVO band fallback via astroquery),
`[sbi]` (SNPE).

## Testing

- Every change ships with tests. Run `pytest -q` before opening a PR.
- Mark slow tests (e.g. neural-net training) with `@pytest.mark.slow`; CI fast runs use `-m "not slow"`.
- Stochastic code must be **reproducible**: fixed `seed` ⇒ identical output, *independent of `n_jobs`*.
  Add a determinism test for any new sampler (see `tests/test_abc.py::test_abc_reproducible_and_njobs_independent`).
- Coverage: `pytest --cov=whisper_labia --cov-report=term-missing`.

## Extending Whisper (the four pluggable axes)

Each axis is a small name registry; adding one is ~1 function. See [`docs/EXTENDING.md`](docs/EXTENDING.md)
for full examples.

| Axis | Add with | Discover with |
|---|---|---|
| Model | `register_model(name, predict, parameters, prior=...)` | `list_models()` |
| Sampler | `register_sampler(name, SamplerClass)` (subclass `BaseSampler`) | `list_samplers()` |
| Likelihood | `register_likelihood(name, LikelihoodClass)` | `list_likelihoods()` |
| Distance | `register_distance(name, fn)` | `list_distances()` |

For parallel ABC (`n_jobs > 1`) the model `predict`, the prior, and the distance must be **picklable**
(module-level functions, not closures/lambdas).

## Conventions

- **Style:** clarity over cleverness; explicit over hidden magic; document every public symbol
  (purpose + parameters + returns); type-hint the public surface; no global mutable state or
  undocumented defaults. See [`docs/DESIGN.md`](docs/DESIGN.md) for the design rationale.
- **Naming:** `select_*` / `calc_*` / `add_*` for `LightCurve`; `list_*` / `register_*` / `get_*` for
  registries; `snake_case` throughout.
- **Commits:** end the message with the trailer
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` when AI-assisted.
- **Safety:** never delete files/data without confirming with a maintainer first.

## Pull requests

1. Branch from `main`.
2. Implement + tests + docs + a CHANGELOG entry under `[Unreleased]`.
3. `pytest -q` green; run `-m "not slow"` at minimum if you lack the `[sbi]` extra.
4. Open a PR describing the change and its rationale.

By contributing you agree your contributions are licensed under the project's GPL-3.0 license.
