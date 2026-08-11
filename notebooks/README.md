# Tutorials

Hands-on Jupyter notebooks for the main WHISPER functionality. Run them in order; each is
self-contained and executes in a few minutes.

| # | Notebook | Covers |
|---|---|---|
| 1 | [Light curves](01_lightcurves.ipynb) | `load_lightcurve`, SNR cuts, band selection, magnitude↔flux, phase, plotting |
| 2 | [Models](02_models.ipynb) | built-in models, `predict`, the physical kilonova model, `register_model` |
| 3 | [Samplers](03_samplers.ipynb) | `fit_ABC` / `fit_ABC_SMC` / `fit_MCMC` / `fit_SNPE`, `SamplerResult`, AIC/BIC |
| 4 | [Visualizing results](04_visualizing.ipynb) | `plot_corner`, `recovery_metrics`, `waic` |
| 5 | [Bayesian model comparison](05_bayesian_model_comparison.ipynb) | AT2017GFO: one/two/three-component kilonova × ABC/SNPE/MCMC, AIC/BIC ranking |

Notebooks 2 and 5 need the optional **redback** backend (`pip install 'whisper-labia[models]'`);
5 also benefits from **sbi** (`[sbi]`) for the SNPE fits. The source cells are generated from
`_build_notebooks.py`.
