#!/usr/bin/env python
"""Generate the WHISPER tutorial notebooks (nbformat -> valid .ipynb, no embedded outputs).

Run inside the container from the repo root:
    python notebooks/_build_notebooks.py
Then execute the light ones with `jupyter nbconvert --execute` to embed outputs.
Kept as the editable source of truth for the tutorial set.
"""
import os

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


def save(name, cells):
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    path = os.path.join(HERE, name)
    nbf.write(nb, path)
    print("wrote", path)


# ======================================================================================
# 01 — Light curves
# ======================================================================================
save("01_lightcurves.ipynb", [
    md("# 1 · Light curves\n\n"
       "`LightCurve` is WHISPER's data container: a subclass of `astropy.table.Table` with photometry "
       "helpers (SNR cuts, band grouping, magnitude↔flux, phase, plotting). `load_lightcurve` reads a "
       "CSV and auto-detects the columns.\n\n"
       "This notebook uses the bundled **AT2017GFO** (GW170817 kilonova) g/r/i data."),
    code("import numpy as np\n"
         "from pathlib import Path\n"
         "import whisper_labia as wp\n\n"
         "wp.__version__"),
    md("## Load\n\n"
       "`load_lightcurve` detects time / magnitude (or flux) / error / band / system columns, normalises "
       "band names, and resolves each band's effective wavelength + zero point. AT2017GFO sits in "
       "NGC 4993 at **z ≈ 0.0098**."),
    code("csv = next(p for p in [Path('../tests/data/at2017gfo.csv'),\n"
         "                       Path('tests/data/at2017gfo.csv')] if p.exists())\n"
         "lc = wp.load_lightcurve(csv, redshift=0.0098, bands=['g', 'r', 'i'])\n"
         "print(lc)"),
    code("print('points :', lc.n_points)\n"
         "print('bands  :', lc.bands)\n"
         "print('columns:', lc.colnames)"),
    md("## Quality cuts\n\n"
       "Per-point SNR is `lc.snr`. `select_snr` keeps detections above a threshold; slicing and `where` "
       "select by band or time window. All return a new `LightCurve`."),
    code("print('SNR range:', lc.snr.min().round(1), '->', lc.snr.max().round(1))\n\n"
         "good = lc.select_snr(min_snr=5)\n"
         "print('after SNR>5 :', good.n_points, 'of', lc.n_points)\n\n"
         "r_band = lc[lc['band'] == 'r']\n"
         "print('r-band only :', r_band.n_points, 'points')"),
    md("## Magnitude ↔ flux\n\n"
       "The data is stored in its native mode (here magnitude). `add_flux()` returns a copy with "
       "`flux` / `flux_err` columns (AB: `F = 3631 · 10^(-0.4 m)` Jy); `add_mag()` does the reverse."),
    code("lcf = lc.add_flux()\n"
         "print('first 3 mags :', np.round(lc['magnitude'][:3], 2))\n"
         "print('first 3 flux :', np.round(lcf['flux'][:3] * 1e6, 2), 'µJy')"),
    md("## Phase (days since merger)\n\n"
       "`calc_phase` returns a copy whose `time` column is days since a reference epoch — here the "
       "GW170817 merger, MJD 57982.529 — optionally rest-frame-corrected with the redshift."),
    code("ph = np.asarray(lc.calc_phase(reference=57982.529).time)\n"
         "print('phase span:', round(ph.min(), 2), '->', round(ph.max(), 2), 'days')"),
    md("## Plot\n\n"
       "`plot_light_curve` has two layouts: `report` (apparent magnitude + flux density, stacked) and "
       "`grid` (one panel per band). Detections are circles; low-SNR and upper limits get distinct "
       "markers."),
    code("fig = wp.plot_light_curve(lc, layout='report', title='AT2017GFO (g/r/i)')"),
    code("fig = wp.plot_light_curve(lc, layout='grid', quantity='apparent_mag')"),
    md("**Next:** [2 · Models](02_models.ipynb) — the forward models you fit to this data."),
])


# ======================================================================================
# 02 — Models
# ======================================================================================
save("02_models.ipynb", [
    md("# 2 · Models\n\n"
       "A **model** maps parameters + observation times/bands to predicted flux. WHISPER ships several "
       "and lets you register your own. `predict(params, times, bands) -> flux [Jy]` is the whole "
       "contract; the samplers call it."),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "import whisper_labia as wp\n\n"
         "wp.list_models()"),
    md("## Evaluate a built-in model\n\n"
       "`get_model` returns a `Model`; `.predict` evaluates it. `bazin` is an empirical supernova "
       "light-curve shape (band-independent here)."),
    code("bazin = wp.get_model('bazin')\n"
         "print('parameters:', bazin.parameters)\n\n"
         "t = np.linspace(0, 60, 120)\n"
         "theta = {'amplitude': 100.0, 't0': 15.0, 'tau_rise': 5.0, 'tau_fall': 25.0}\n"
         "flux = bazin.predict(theta, t, bands=None)\n\n"
         "plt.plot(t, flux); plt.xlabel('time [d]'); plt.ylabel('flux'); plt.title('bazin');"),
    md("## Physical models\n\n"
       "`mck19` (AGN-disk BBH flare) and `two_component_kilonova` (NS–NS merger, via the optional "
       "**redback** backend) are physically parametrised and band-dependent. The kilonova needs "
       "`bands` and returns flux density in Jy."),
    code("kn = wp.get_model('two_component_kilonova')\n"
         "print('parameters:', kn.parameters)\n\n"
         "t = np.linspace(0.5, 15, 40)\n"
         "theta = dict(mej_1=0.02, vej_1=0.25, temperature_floor_1=3000,\n"
         "             mej_2=0.05, vej_2=0.15, temperature_floor_2=1200,\n"
         "             kappa_1=0.5, kappa_2=3.0, redshift=0.0098)\n"
         "for b in ['g', 'r', 'i']:\n"
         "    f = kn.predict(theta, t, bands=[b] * len(t))\n"
         "    plt.plot(t, -2.5 * np.log10(f / 3631), label=b)\n"
         "plt.gca().invert_yaxis(); plt.xlabel('days'); plt.ylabel('AB mag'); plt.legend();"),
    md("## Register your own\n\n"
       "`register_model(name, predict, parameters, prior=...)` adds a model to the registry so every "
       "sampler can use it by name. The `predict` signature is fixed; `prior` is an optional default."),
    code("def line_flux(p, times, bands=None):\n"
         "    return p['slope'] * np.asarray(times, float) + p['intercept']\n\n"
         "wp.register_model('line', line_flux, ['slope', 'intercept'],\n"
         "                  prior=wp.Prior({'slope': wp.Uniform(-5, 5),\n"
         "                                  'intercept': wp.Uniform(-10, 10)}),\n"
         "                  overwrite=True)\n"
         "print('line' in wp.list_models(), '->', wp.get_model('line').parameters)"),
    md("**Next:** [3 · Samplers](03_samplers.ipynb) — fitting a model to data."),
])


# ======================================================================================
# 03 — Samplers
# ======================================================================================
save("03_samplers.ipynb", [
    md("# 3 · Samplers\n\n"
       "WHISPER exposes four samplers behind one interface — `fit_<NAME>(lc, model, prior=..., ...)` "
       "→ `SamplerResult`:\n\n"
       "| sampler | family | notes |\n"
       "|---|---|---|\n"
       "| `fit_ABC` | likelihood-free rejection | simple, parallel, robust |\n"
       "| `fit_ABC_SMC` | sequential ABC | importance-weighted, adaptive ε |\n"
       "| `fit_MCMC` | exact likelihood (emcee) | gold-standard posterior |\n"
       "| `fit_SNPE` | neural (sbi) | amortized; GPU-capable |\n\n"
       "We fit synthetic data from a known truth so recovery is checkable."),
    code("import numpy as np\n"
         "import whisper_labia as wp\n\n"
         "MODEL = 'gaussian_rise'\n"
         "truth = {'amplitude': 5.0, 't0': 8.0, 'sigma_rise': 3.0, 'tau_decay': 15.0}\n"
         "t = np.linspace(0.1, 30, 60)\n"
         "clean = wp.get_model(MODEL).predict(truth, t, None)\n"
         "err = np.full_like(clean, 0.1)\n"
         "obs = clean + np.random.default_rng(0).normal(0, err)\n"
         "lc = wp.LightCurve(time=t, band=['r'] * len(t), flux=obs, flux_err=err, name='synthetic')\n"
         "lc.n_points"),
    code("prior = wp.Prior({'amplitude': wp.Uniform(0, 20), 't0': wp.Uniform(0, 20),\n"
         "                  'sigma_rise': wp.Uniform(0.5, 10), 'tau_decay': wp.Uniform(1, 40)})"),
    md("## Fit with each sampler\n\n"
       "Budgets are kept small for speed. `SamplerResult` carries `.samples` (a DataFrame), `.summary` "
       "(median + credible intervals), `.best_params`, exact `.aic` / `.bic`, and `.runtime_s`."),
    code("res_abc = wp.fit_ABC(lc, MODEL, prior=prior, n_simulations=20000, quantile=0.01, seed=0)\n"
         "res_smc = wp.fit_ABC_SMC(lc, MODEL, prior=prior, n_particles=400, n_rounds=4, seed=0)\n"
         "res_mc  = wp.fit_MCMC(lc, MODEL, prior=prior, nsteps=2000, burnin=500, seed=0)\n"
         "print('done')"),
    code("import pandas as pd\n"
         "rows = []\n"
         "for name, r in [('ABC', res_abc), ('ABC-SMC', res_smc), ('MCMC', res_mc)]:\n"
         "    s = r.summary\n"
         "    rows.append(dict(sampler=name,\n"
         "                     amplitude=round(s['amplitude']['median'], 2),\n"
         "                     t0=round(s['t0']['median'], 2),\n"
         "                     AIC=round(r.aic, 1), runtime_s=round(r.runtime_s, 1)))\n"
         "pd.DataFrame(rows)"),
    md("Truth was `amplitude=5.0, t0=8.0`. All three recover it. `fit_SNPE` follows the same interface "
       "(add `device='cuda'` for GPU); it trains a neural posterior, so it costs more up front but can "
       "then condition on new data cheaply.\n\n"
       "**Next:** [4 · Visualizing results](04_visualizing.ipynb)."),
])


# ======================================================================================
# 04 — Visualizing results
# ======================================================================================
save("04_visualizing.ipynb", [
    md("# 4 · Visualizing results\n\n"
       "After fitting, WHISPER provides `plot_corner` (posterior overlays), the validation metrics "
       "(`recovery_metrics`, `waic`), and posterior-predictive checks. We reuse the synthetic fit from "
       "notebook 3."),
    code("import numpy as np\n"
         "import whisper_labia as wp\n\n"
         "MODEL = 'gaussian_rise'\n"
         "truth = {'amplitude': 5.0, 't0': 8.0, 'sigma_rise': 3.0, 'tau_decay': 15.0}\n"
         "t = np.linspace(0.1, 30, 60)\n"
         "clean = wp.get_model(MODEL).predict(truth, t, None)\n"
         "err = np.full_like(clean, 0.1)\n"
         "obs = clean + np.random.default_rng(0).normal(0, err)\n"
         "lc = wp.LightCurve(time=t, band=['r'] * len(t), flux=obs, flux_err=err, name='synthetic')\n"
         "prior = wp.Prior({'amplitude': wp.Uniform(0, 20), 't0': wp.Uniform(0, 20),\n"
         "                  'sigma_rise': wp.Uniform(0.5, 10), 'tau_decay': wp.Uniform(1, 40)})\n"
         "abc = wp.fit_ABC(lc, MODEL, prior=prior, n_simulations=20000, quantile=0.01, seed=0)\n"
         "mc  = wp.fit_MCMC(lc, MODEL, prior=prior, nsteps=2000, burnin=500, seed=0)"),
    md("## Corner plot\n\n"
       "`plot_corner` overlays any number of posteriors on shared axes, with the truth marked. Pass a "
       "list of `SamplerResult.samples` (or DataFrames) and matching labels."),
    code("fig = wp.plot_corner([abc.samples, mc.samples], labels=['ABC', 'MCMC'],\n"
         "                     parameters=['amplitude', 't0', 'sigma_rise', 'tau_decay'],\n"
         "                     truths=truth, title='gaussian_rise recovery')"),
    md("## Recovery metrics\n\n"
       "`recovery_metrics` scores a fit against a known truth: per-parameter bias, standardized "
       "z-score (|z|≲2 ⇒ recovered), and 68/95% credible-interval coverage."),
    code("import pandas as pd\n"
         "m = wp.recovery_metrics(mc, truth)\n"
         "pd.DataFrame(m).T[['median', 'bias', 'z_score', 'within_68', 'within_95']]"),
    md("## WAIC\n\n"
       "`waic` is a fully-Bayesian fit score (lower = better) usable for model comparison across "
       "samplers that share the data + likelihood."),
    code("print('MCMC WAIC:', round(wp.waic(mc, lc, MODEL)['waic'], 1))"),
    md("**Next:** [5 · Bayesian model comparison](05_bayesian_model_comparison.ipynb) — the full "
       "workflow on real kilonova data."),
])

print("\nbuilt notebooks 01-04")


# ======================================================================================
# 05 — Bayesian model comparison (AT2017GFO, 1/2/3-component kilonova x ABC/SNPE/MCMC)
# ======================================================================================
save("05_bayesian_model_comparison.ipynb", [
    md("# 5 · Bayesian model comparison\n\n"
       "The full workflow: fit **AT2017GFO** (the GW170817 kilonova) with kilonova models of increasing "
       "complexity — **one, two, and three ejecta components** — using **ABC**, **SNPE**, and **MCMC**, "
       "then compare them with information criteria to ask *how many components the data support*.\n\n"
       "Requires the optional **redback** backend. Budgets here are deliberately small so the notebook "
       "runs in minutes; scale `n_simulations` / `nsteps` up for publication-grade posteriors."),
    code("import numpy as np\n"
         "import pandas as pd\n"
         "from pathlib import Path\n"
         "import whisper_labia as wp\n"
         "from redback.model_library import all_models_dict\n"
         "from whisper_labia.models.two_component_kilonova import _redback_band\n\n"
         "wp.__version__"),
    md("## Data — preprocessed AT2017GFO\n\n"
       "The preprocessed UVOIR reduction (Swift-UVOT `uvw1` dropped, SNR > 5, one point per band per "
       "epoch). We keep four well-sampled bands and the first 15 days for speed."),
    code("csv = next(p for p in [\n"
         "    Path('../analysis/at2017gfo_villar/data/at2017gfo_full_preprocessed.csv'),\n"
         "    Path('analysis/at2017gfo_villar/data/at2017gfo_full_preprocessed.csv')] if p.exists())\n"
         "Z = 0.0098\n"
         "lc = wp.load_lightcurve(csv, redshift=Z, explosion_date=57982.529, bands=['g', 'r', 'i'])\n"
         "lc = lc[np.asarray(lc.time) <= 12.0]\n"
         "print(lc.n_points, 'points |', lc.bands)"),
    md("## Models — a one/two/three-component ladder\n\n"
       "redback exposes one-, two-, and three-component kilonova models. We wrap each as a WHISPER "
       "`predict(params, times, bands) -> flux [Jy]`, fitting **mej, vej, kappa per component** and "
       "fixing the redshift and (for 2/3-comp) the temperature floors. This gives a clean **3 / 6 / 9** "
       "parameter ladder."),
    code("AB = 3631.0\n\n"
         "class KilonovaModel:\n"
         "    \"\"\"Picklable predict(params, times, bands) -> flux [Jy] wrapping a redback kilonova model.\n\n"
         "    A class (not a closure) so it pickles for parallel ABC/MCMC (`n_jobs > 1`). It stores only\n"
         "    the redback model name + parameter lists; the model function is looked up per call.\"\"\"\n"
         "    def __init__(self, redback_name, param_names, fixed):\n"
         "        self.redback_name, self.param_names, self.fixed = redback_name, param_names, fixed\n"
         "    def __call__(self, p, times, bands):\n"
         "        fn = all_models_dict[self.redback_name]\n"
         "        t = np.clip(np.asarray(times, float), 0.1, None)\n"
         "        bands = np.asarray(bands)\n"
         "        kw = {k: float(p[k]) for k in self.param_names}\n"
         "        kw.update(self.fixed); kw['redshift'] = Z\n"
         "        out = np.empty(t.shape)\n"
         "        for b in np.unique(bands):\n"
         "            sel = bands == b\n"
         "            mag = np.asarray(fn(t[sel], output_format='magnitude',\n"
         "                                bands=[_redback_band(b)], **kw), float)\n"
         "            mag = np.nan_to_num(mag, nan=40.0, posinf=40.0, neginf=40.0)\n"
         "            out[sel] = AB * 10 ** (-0.4 * mag)\n"
         "        return out"),
    code("# priors: mej [Msun] log-uniform, vej [c] uniform, kappa [cm^2/g] uniform, per component\n"
         "def comp(i):\n"
         "    return {f'mej_{i}': wp.LogUniform(1e-3, 0.1), f'vej_{i}': wp.Uniform(0.05, 0.3),\n"
         "            f'kappa_{i}': wp.Uniform(0.5, 30.0)}\n\n"
         "FLOOR = 2500.0                              # temperature floor fixed (2/3-comp models)\n"
         "LADDER = {  # name: (redback model, free params, fixed non-redshift params)\n"
         "    'kn1': ('one_component_kilonova_model',   ['mej', 'vej', 'kappa'], {}),\n"
         "    'kn2': ('two_component_kilonova_model',\n"
         "            ['mej_1', 'vej_1', 'kappa_1', 'mej_2', 'vej_2', 'kappa_2'],\n"
         "            dict(temperature_floor_1=FLOOR, temperature_floor_2=FLOOR)),\n"
         "    'kn3': ('three_component_kilonova_model',\n"
         "            ['mej_1', 'vej_1', 'kappa_1', 'mej_2', 'vej_2', 'kappa_2',\n"
         "             'mej_3', 'vej_3', 'kappa_3'],\n"
         "            dict(temperature_floor_1=FLOOR, temperature_floor_2=FLOOR, temperature_floor_3=FLOOR)),\n"
         "}\n"
         "PRIORS = {\n"
         "    'kn1': wp.Prior({'mej': wp.LogUniform(1e-3, 0.1), 'vej': wp.Uniform(0.05, 0.3),\n"
         "                     'kappa': wp.Uniform(0.5, 30.0)}),\n"
         "    'kn2': wp.Prior({**comp(1), **comp(2)}),\n"
         "    'kn3': wp.Prior({**comp(1), **comp(2), **comp(3)}),\n"
         "}\n"
         "for name, (rb, params, fixed) in LADDER.items():\n"
         "    wp.register_model(name, KilonovaModel(rb, params, fixed), params, overwrite=True)\n"
         "print('registered:', list(LADDER))"),
    md("## Fit — 3 models × 3 samplers\n\n"
       "Same interface for every fit. ABC and SNPE are likelihood-free; MCMC uses the exact Gaussian "
       "likelihood. redback is CPU-only and semi-analytic, so we keep small budgets (`n_jobs` / "
       "`num_workers` parallelise the simulations) and fit **one model per cell** — each takes a couple "
       "of minutes."),
    code("def fit_all(name):\n"
         "    prior = PRIORS[name]\n"
         "    abc  = wp.fit_ABC(lc, name, prior=prior, n_simulations=3000, quantile=0.03,\n"
         "                      n_jobs=8, seed=0)\n"
         "    snpe = wp.fit_SNPE(lc, name, prior=prior, num_rounds=1, num_simulations=1500,\n"
         "                       num_samples=1500, num_workers=8, seed=0)\n"
         "    mcmc = wp.fit_MCMC(lc, name, prior=prior, nsteps=800, burnin=300, n_jobs=8, seed=0)\n"
         "    return {'ABC': abc, 'SNPE': snpe, 'MCMC': mcmc}\n\n"
         "results = {}"),
    code("results['kn1'] = fit_all('kn1'); print('kn1 (3 params) done')"),
    code("results['kn2'] = fit_all('kn2'); print('kn2 (6 params) done')"),
    code("results['kn3'] = fit_all('kn3'); print('kn3 (9 params) done')"),
    md("## Compare\n\n"
       "**AIC** and **BIC** (from the exact Gaussian log-likelihood at the best draw, both penalising "
       "parameter count) let us compare the complexity ladder on equal footing. Lower is better; BIC "
       "penalises extra parameters more strongly."),
    code("rows = []\n"
         "for name in LADDER:\n"
         "    k = len(LADDER[name][1])\n"
         "    for sampler, r in results[name].items():\n"
         "        rows.append(dict(model=name, k=k, sampler=sampler,\n"
         "                         AIC=round(r.aic, 1), BIC=round(r.bic, 1),\n"
         "                         runtime_s=round(r.runtime_s, 1)))\n"
         "tbl = pd.DataFrame(rows)\n"
         "tbl.pivot_table(index=['model', 'k'], columns='sampler', values='AIC')"),
    md("Read the table down each column: if AIC/BIC keeps dropping from `kn1`→`kn2`→`kn3`, the data "
       "reward the extra component; if it flattens or rises, the simpler model is preferred (added "
       "components are unconstrained). The three samplers should agree on the trend."),
    code("# best (lowest-AIC) fit overall\n"
         "best = min(rows, key=lambda d: d['AIC'])\n"
         "print('best by AIC:', best['model'], 'via', best['sampler'], '(AIC', best['AIC'], ')')\n"
         "br = results[best['model']][best['sampler']]\n"
         "br.summary"),
    md("## Posterior of the preferred model\n\n"
       "Overlay the three samplers' posteriors for the preferred model — agreement is the sign of a "
       "well-constrained fit."),
    code("name = best['model']\n"
         "wp.plot_corner([results[name][s].samples for s in ['ABC', 'SNPE', 'MCMC']],\n"
         "               labels=['ABC', 'SNPE', 'MCMC'], parameters=LADDER[name][1],\n"
         "               log_params=[p for p in LADDER[name][1] if p.startswith('mej')],\n"
         "               title=f'AT2017GFO — {name} posterior');"),
    md("## Summary\n\n"
       "One interface (`fit_<sampler>`), one registry (`register_model`) — swapping model complexity "
       "and inference method is a one-line change, and AIC/BIC give a consistent, cross-sampler model "
       "ranking. Scale the budgets up (and add the temperature floors / a Villar+17 scatter term) for a "
       "production analysis; see [`analysis/at2017gfo_villar/`](../analysis/at2017gfo_villar/) for the "
       "full study."),
])

print("built notebook 05")

