# How whisper computes a band magnitude

A light curve is broadband photometry: every point is a flux integrated over a filter. So is every
model prediction in whisper, on the CPU and on the GPU, since version 0.1.1. This page says which
survey measurement becomes a data point, how the model's integral is taken, where the filter curves
come from, what the numbers are, and how to add a band. The code is `whisper_cbpf.io.surveys` and
`whisper_cbpf.synphot`.

**Contents**

0. [From an alert to a data point](#0-from-an-alert-to-a-data-point)
1. [The definition](#1-the-definition)
2. [From a label in your data to a filter](#2-from-a-label-in-your-data-to-a-filter)
3. [The quadrature: 16 Gauss nodes per band](#3-the-quadrature-16-gauss-nodes-per-band)
4. [Where the curves come from](#4-where-the-curves-come-from)
5. [What each model does](#5-what-each-model-does)
6. [Adding a band](#6-adding-a-band)
7. [What changed in 0.1.1, with numbers](#7-what-changed-in-011-with-numbers)

---

## 0. From an alert to a data point

`load_lightcurve(x, survey="lsst" | "ztf")` fits **difference-imaging photometry only**: the PSF flux
measured on the science image minus a reference image, which is the transient's own light with the
host subtracted. That is what every model predicts.

| | LSST (Rubin alert) | ZTF |
|---|---|---|
| detection | diaSource `psfFlux`, `psfFluxErr` [nJy] | `magpsf`, `sigmapsf` where `isdiffpos` is positive |
| AB magnitude | `m = 31.4 - 2.5 log10(psfFlux / nJy)`, `sigma_m = (2.5 / ln 10) psfFluxErr / psfFlux` | `magpsf` |
| upper limit | 5 sigma: `31.4 - 2.5 log10(5 psfFluxErr)` at a forced-photometry epoch with no diaSource | 5 sigma: `diffmaglim` of a row with no `magpsf` |
| never read | `scienceFlux` (direct image: transient plus host) | `magpsf_corr` (reference flux added back: a total magnitude for variable stars) |
| negative difference | dropped and counted (no AB magnitude) | dropped and counted |
| band label | `lsstu` … `lssty` | `ztfg`, `ztfr`, `ztfi` |

31.4 is the AB magnitude of 1 nJy (`io.surveys.LSST_ZP_NJY`). The labels are survey-prefixed, so a
model integrates the right filter without any `band_aliases=` (§2); the upper limits enter the
likelihood in flux space at `lc.meta["upper_limit_sigma"]` = 5. The field mapping in full is in
[`LSST_ALERTS.md`](LSST_ALERTS.md#field-mapping).

---

## 1. The definition

For a photon-counting filter with transmission `T_b(λ)` and a model flux density `F_ν(λ)` in the
observer frame,

```
m_b = -2.5 log10[ ∫ F_ν(λ) T_b(λ) dλ/λ  /  (3631 Jy ∫ T_b(λ) dλ/λ) ]
```

and whisper computes it as a quadrature,

```
m_b = -2.5 log10[ Σ_k w_bk F_ν(λ_bk)  /  (3631 Jy Σ_k w_bk) ]
```

with the nodes `λ_bk` and weights `w_bk` of **one `FilterSet`**. The CPU path
(`synphot.band_flux_jy`) and the JAX kernels (`filter_set=` of every factory) read the same
object, so a CPU model and a GPU model built on the same `FilterSet` compute the same number. The
AB zero point is 3631 Jy everywhere (pyphot's, for comparison, is 48.60 mag = 3630.78 Jy, 0.066 mmag
away).

## 2. From a label in your data to a filter

`whisper_cbpf.resolve_filter(label, aliases=None, default_system=None)` is the one map every model
uses. In order:

| Label | Filter | Note |
|---|---|---|
| your `band_aliases` entry | its target | applied first |
| bare `u g r i z y` | LSST's (`lsstg` …) | see below |
| `ztfg`, `zg`, `ZTF_g` | `ztfg` | survey spellings are normalised |
| an sncosmo name (`sdssr`, `bessellb`, `uvot::uvw1`) | itself | |
| a friendly name in redback's `filters.csv` (`B`, `J`, `Ks`, `w`) | its sncosmo curve | read from disk, redback is not imported |
| an SVO ID (`LSST/LSST.g`, `2MASS/2MASS.J`) | itself | the curve is fetched from SVO |
| a grouped label (`g-band`) | — | **raises**: it merges every g filter, so there is no curve |

**Bare letters mean LSST, everywhere.** A bare letter names no survey, and whisper is built for LSST
alerts. A warning names the system the first time a bare letter is read in a session. Change it:

```python
import whisper_cbpf as wp
wp.set_default_band_system("sdss")                        # this session: u g r i z are SDSS
wp.register_redback("arnett", redshift=0.01, default_system="ztf")        # one model
wp.kilonova_model(["g", "r"], 0.0098, 1.3e26, band_aliases={"g": "sdssg", "r": "sdssr"})
```

Systems: `lsst` (ugrizy), `sdss` (ugriz), `ztf` (gri), `ps1` (grizy), `des` (grizy). The session
default is mirrored in `$WHISPER_BAND_SYSTEM`, so sampler worker processes (started with `spawn`)
read bare letters as the parent does; exporting it before Python starts sets the default too. The
JAX factories resolve their `band_names` once, when the model is built. AT2017GFO's
`g r i` (`tests/data/at2017gfo.csv`) are SDSS photometry, so its fits pass `default_system="sdss"`.

**A grouped label raises in band mode.** `load_lightcurve(band_lookup=True)` collapses filters into
effective bands (`g-band`, …) for plotting and quick looks. A model cannot integrate over
"every g filter", so it refuses the label with a message naming both ways out: load without
`band_lookup`, or map the group to one filter with `band_aliases={"g-band": "ztfg"}`. Up to 0.1.0 the
redback adapter modelled a `g-band` point as SDSS g, including ZTF g data.

## 3. The quadrature: 16 Gauss nodes per band

`dμ_b = T_b(λ) dλ/λ` is a positive measure on the band, so it has a Gaussian quadrature of its own:
`synphot.gauss_rule(names, n_nodes=16)` builds it (Golub–Welsch on a fine discretisation of the
piecewise-linear curve: 8 Gauss–Legendre points on every tabulation segment). The weights are
positive, the nodes lie inside the band, `Σ w = ∫T dλ/λ` to 1e-12, and the rule is exact for every
polynomial of degree ≤ 31. Model SEDs are smooth across a band, so 16 nodes are enough.

Measured against the fine integral (`tests/test_synphot_rules.py`, every shipped filter):

| SED | Gauss-16, max \|Δm\| | Gauss-8 |
|---|---|---|
| blackbody 300 K – 1e5 K (where the flux is representable) | < 1e-8 mmag | up to 2.0 mmag (SDSS g at 300 K) |
| 500 Å Gaussian line anywhere in 3000–11000 Å | < 1e-5 mmag | up to 0.015 mmag |
| redback `CutoffBlackbody` (kink at `λ_cut` inside the band) | **0.17–1.1 mmag** | — |

The last row is the rule's known limit: redback's cutoff blackbody (`type_1a`, `slsn`,
`general_magnetar_slsn`, `tde_analytical`) multiplies the blackbody by `λ/λ_cut` below the cutoff,
a kink that no polynomial rule integrates to round-off. Gauss-32 brings it to ≤ 0.18 mmag
(`filter_set=synphot.gauss_rule(names, n_nodes=32)`). The monochromatic path it replaced was 13–181
mmag off on smooth SEDs.

**Gauss-16 stays the default.** On redback's own priors the kink costs one family more
than the 2e-5 mag every other family meets: `type_1a` (cutoff at 3000 Å rest frame) in LSST g at
z ≈ 0.35, where the kink lands at 4050 Å, inside the band. The worst of 60 prior draws is **3.6e-5
mag** with Gauss-16 and **8.9e-6 mag** with `n_nodes=32` (§7).
That family's parity test carries a documented **5e-5 mag**
budget (`tests/test_photometry_parity.py::test_type_1a_cutoff_kink_in_lsst_g_is_within_its_documented_budget`,
which also checks the Gauss-32 figure); the others are held to 2e-5. If 0.04 mmag matters to your
fit, pass a Gauss-32 `FilterSet` (`synphot.filter_set_for(names, n_nodes=32)`) to both the CPU and
the GPU model.

A build-time **self-check** compares every new rule with the fine integral on 1500–50 000 K
blackbodies and warns (naming the band and the fix, twice the nodes) above 0.02 mmag, so an odd
curve fetched from SVO is caught. Gauss-16 passes it to 1e-14 mag on every shipped filter.

An independent implementation agrees: pyphot's `Filter.get_flux` on the same curve is within
0.0015 mmag of the Gauss-16 integral on 3000–30 000 K blackbodies
(`tests/test_photometry_parity.py`), and within 1e-6 mag of a 1 Å reference band integral on every redback
family.

**The grid rule** (`synphot.grid_rule`: `make_filter_set`, `ab_weights`) is what the JAX models used
up to 0.1.0: every bandpass on one geometric 1000–30000 Å grid of `n_wave` points with exact cell
integrals. It is kept, bit for bit (the t0 goldens depend on it), and selected by passing `n_wave=`
to a factory. It agrees with Gauss-16 to ≤ 0.005 mmag and integrates over 2000 points instead of
16 per band: JAX-on-CPU `arnett` predict 3.7 → 1.5 ms for 60 observations in two bands.

## 4. Where the curves come from

| Source | Filters | When |
|---|---|---|
| shipped `FilterSet`s (`whisper_cbpf/synphot/data/*.npz`) | LSST ugrizy, ZTF gri, SDSS ugriz | always; no sncosmo needed |
| sncosmo | every sncosmo bandpass | any other sncosmo name |
| SVO Filter Profile Service | any SVO ID | via pyphot when installed, astroquery otherwise (`io.svo`) |

The shipped sets are Gauss-16 built from sncosmo's curves (`synphot.build_library()`); each band
records the sncosmo version, the sha256 of the curve and the throughput release, and
`tests/test_synphot_rules.py` rebuilds them from the installed sncosmo and compares. So the default
curves are sncosmo's, which is what redback itself integrates: LSST baseline throughputs **v1.1**
(2016), ZTF from U. Feindt (no atmosphere), SDSS from Doi et al. 2010 (airmass 1.3).

**The curve matters more than the quadrature.** The same blackbodies through SVO's
curves instead:

| | SVO − sncosmo, max over 3000–30 000 K at z = 0 and 0.35 |
|---|---|
| LSST u / g / r / i / z / y | 32 / 43 / 10 / 3.4 / 0.2 / 10 mmag |
| ZTF g / r / i | 31 / 12 / 2.5 mmag |
| SDSS g / r / i | 5.0 / 1.9 / 5.2 mmag |

SVO serves LSST throughputs **v1.5** and the ZTF team's filter+CCD curves; redback's own reference
frequencies for LSST are SVO's pivots.

**sncosmo's curves stay the default**, for parity with redback, whose band integral
uses them. The measured cost of that choice is the table above: up to **43 mmag in LSST g**, **32 in
LSST u** and **31 in ZTF g** against SVO's curves, largest for the coolest, most redshifted SED
(3000 K at z = 0.35). It is a property of the curves, not of the quadrature, whose worst case (§3,
3.6e-5 mag) is a thousand times smaller. To fit with SVO's curves instead, name them:
`band_aliases={"g": "LSST/LSST.g", ...}`.

## 5. What each model does

| Model | Band integral | Notes |
|---|---|---|
| `redback_model` / `register_redback` (CPU) | `photometry="band"` (default): redback's `flux_density` at every node of every observation, **one** redback call, each observation's nodes contiguous and the last observation last (redback sizes some grids by `time[-1]`) | `photometry="monochromatic"`: redback's SED at the band's reference frequency (0.1.0 behaviour), for physics-parity tests and for labels redback knows but no curve exists for. `filter_set=`, `default_system=`, `band_aliases=` |
| redback's double-dilation TDEs (`gaussianrise_cooling_envelope`, `bpl_cooling_envelope`, `stream_stream_tde`) | as above, redback called at `t (1+z)` | redback dilates time twice (0.5 mag at z = 0.06, 2.7 at z = 0.35). Detected from redback's source (`redback_double_dilation`), so it turns itself off when redback is fixed |
| `two_component_kilonova` (CPU) | the sum of two `one_component_kilonova_model` calls, band-integrated | redback's two-component model stops at 6 d |
| `mck19` | disk and hotspot blackbodies integrated over the filter | an unresolvable band raises (it was evaluated at 6000 Å) |
| JAX factories (`kilonova_model`, `tde_model`, `supernova_model`, …) | Gauss-16 via `filter_set=` by default; `n_wave=` for the grid rule | `default_system=` for bare letters in `band_names` |
| JAX supernovae with a free explosion time or redshift (`diffusion_grid="fixed"`) | the same Gauss-16 integral, of the SED evaluated at the observed epochs only | the diffusion runs on fixed source-frame epochs (30 per decade) and only the bolometric luminosity is interpolated to each observation: within 8.8e-4 mag of redback over 200 prior draws per family |

A non-finite flux at any node makes that observation exactly 0, as the adapter always returned for
out-of-domain epochs.

## 6. Adding a band

Any sncosmo bandpass name or SVO ID works directly:

```python
wp.register_redback("arnett", ["ztfg", "2MASS/2MASS.J"], redshift=0.01)
```

For a curve of your own, build a `FilterSet` and pass it:

```python
from whisper_cbpf.synphot import gauss_rule, filter_set_for, FilterSet
fs = gauss_rule(["mycam_r"], curves={"mycam_r": (wave_angstrom, transmission)})
fs = FilterSet.concat([filter_set_for(["ztfg"]), fs])
wp.register_redback("arnett", redshift=0.01, filter_set=fs)
wp.supernova_model("arnett", ["ztfg", "mycam_r"], 0.01, dl_cm, filter_set=fs)
fs.save("my_filters.npz")                  # FilterSet.load(...) later, offline
```

The curve must be photon-counting (as sncosmo's and SVO's are, bar a few energy-counting SVO entries).

## 7. What changed in 0.1.1, with numbers

Measured on 200 prior draws per family (ZTF, LSST and
kilonova epochs, reference = redback's SED on a 1 Å band integral):

| | before (0.1.0) | after |
|---|---|---|
| CPU redback adapter − band integral | 4.0–39 mmag median, 17–653 mmag max | ≤ 8e-6 mag max, except type_1a in LSST g at z = 0.35: 2.6e-5 median, 3.6e-5 max (its cutoff-blackbody kink, §3; 8.9e-6 with Gauss-32) |
| CPU − GPU, same model | 4–38 mmag median, up to 0.65 mag (plus port defects since fixed) | ≤ 4e-12 mag for the supernovae and the bare TDE, 8e-10 for the one-component kilonova. Two components: whisper's CPU model and the JAX model are both the sum of one-component solutions (each ≤ 6e-7 mag from that reference); redback's own two-component model, on its 6-day grid, is ≤ 1.9 mmag from both |
| gaussianrise TDE, ZTF / LSST | redback's double dilation uncorrected: CPU − JAX port 0.018 / 0.037 mag median, 1.0 / 2.2 mag max | CPU − redback at `t (1+z)` ≤ 2.1e-7 mag; CPU − JAX 1.9e-5 / 3.0e-5 median, tails 0.056 / 0.11 mag that are the port's own residual, not the photometry |
| two-component kilonova after 6 d | 99 mag (flux 4e-37 Jy) | finite, continuous across 6 d |
| two-component kilonova, 200 observations | 177 ms | 8.7 ms |
| `mck19` g peak at the test point | 25.900 mag (λ_eff) | 25.842 mag (band) |
| bare `g` through the redback adapter | SDSS g | LSST g |
| `lssty` / `y` metadata | z-band, 8679 Å | y-band, 9710 Å |
