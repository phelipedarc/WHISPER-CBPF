# Report - AT2017GFO: ABC vs ABC-SMC across simple models

Test case: **AT2017GFO** (GW170817 kilonova), r-band, 88 points after `min_snr=3`; time = days since explosion (MJD 57982).

Models: `flare`, `bazin` (SN rise+fall), `gaussian_rise` (Gaussian rise + exp decay). Samplers: `ABC` (flat rejection, 200k sims, best 1%) and `ABC-SMC` (1000 particles, 8 adaptive rounds, quantile 0.4). The chi-square distance gives `AIC = chi2_min + 2k`, `BIC = chi2_min + k*ln(n)`.

## Results (ranked by AIC)

| Model | Sampler | k | chi2/dof | AIC | BIC | runtime (s) | simulations |
|---|---|---|---|---|---|---|---|
| gaussian_rise | ABC | 4 | 62.7 | 5277 | 5287 | 1.68 | 200,000 |
| gaussian_rise | ABC-SMC | 4 | 63.1 | 5310 | 5320 | 1.70 | 51,200 |
| bazin | ABC-SMC | 4 | 64.0 | 5387 | 5397 | 1.67 | 51,200 |
| bazin | ABC | 4 | 64.1 | 5390 | 5400 | 2.06 | 200,000 |
| flare | ABC | 3 | 66.8 | 5682 | 5690 | 1.35 | 200,000 |
| flare | ABC-SMC | 3 | 85.5 | 7275 | 7282 | 1.48 | 51,200 |

**Best (lowest AIC): `gaussian_rise` via ABC** (AIC=5277, chi2/dof=62.7).

![comparison](figures/at2017gfo_model_comparison.png)

## Best-fit parameters (ABC-SMC)

- **flare**: amplitude=0.00176, rise_time=1.63, decay_time=1.2
- **bazin**: amplitude=0.00183, t0=-0.0836, tau_rise=8.49, tau_fall=1.37
- **gaussian_rise**: amplitude=0.000407, t0=1.15, sigma_rise=3.1, tau_decay=1.45

## Notes

- chi2/dof is high for all models because AT2017GFO's r-band is high-SNR (tiny error bars), so any imperfection inflates chi2. The *relative* AIC ranking still identifies the best simple model.
- Read chi2 alongside the **simulations** column. Flat ABC draws every simulation from the prior (simplest, embarrassingly parallel); ABC-SMC focuses simulations near good regions over rounds. SMC's value is reaching a tight threshold with *fewer* simulations -- decisive when each simulation is expensive (e.g. redback models); for these microsecond toy models a large flat-ABC budget is already very effective.
- These are analytic toy models; physically-motivated fits come from redback models via the optional `[models]` extra.