# Report — Survey-CSV ingestion upgrade (data_mode · redshift · astropy units · SVO bands)

> **Historical snapshot (2026-06-27).** This report captures the state right after the ingestion
> upgrade (**133 tests**, and a `LightCurve` whose `lc()` returned an *enriched dataframe*). Both have
> since moved on: `LightCurve` is now an `astropy.Table` and `lc()` returns the table itself (call
> `add_flux()`/`add_mag()` to derive the other photometry column), and the suite is larger. For the
> **live** test count and the current API contract, see
> [`API_REFERENCE.md`](API_REFERENCE.md) (§8) and [`DESIGN.md`](DESIGN.md). The numbers and `lc()`
> wording below are preserved as a record of that milestone.

_Generated 2026-06-27. Covers the ingestion upgrade: what changed, the adversarial review and its
fixes, the offline demo, and the full test results (133 passing at the time)._

## 1. Summary

The `LightCurve` ingestion path gained four capabilities, all behind the existing pluggable design and
**fully backward compatible** (the 81 pre-existing tests still pass unchanged):

1. **`data_mode`** — a stored attribute in `{flux_density, magnitude, flux}` (default inferred) with an
   `output_format` property (`magnitude` / `flux_density`) — the forward-model comparison space (which
   the optional redback backend also uses). Calling the object (`lc()`) returns the **enriched**
   dataframe with the missing one of flux/magnitude derived from the per-band zero point.
2. **Redshift** — resolved `redshift=` arg > `redshift` column > unknown. Unknown does not fail (records
   `redshift_known=False` + a default `redshift_prior`, warns that `z` will be sampled). Validation:
   `z ≥ 0`; `z == 0` requires `luminosity_distance`; negative/NaN fatal.
3. **astropy units throughout** — F_ν (Jy/mJy/µJy) and F_λ (erg/s/cm²/Å) both accepted and stored
   canonically as Jy (F_λ→Jy via `u.spectral_density(lambda_eff)`); magnitude forced dimensionless AB;
   band-integrated flux dimensionality-checked; no-unit columns warn + apply a documented default.
4. **SVO band fallback** — `FILTER_LOOKUP` miss → warn → SVO Filter Profile Service
   (`astroquery.svo_fps`), cached by filter ID (offline-safe), graceful degradation + manual override.
   Optical effective bands are anchored to the LSST ugrizy zero points.

**Workflow:** implement (115 tests) → 28-agent adversarial review (21 findings, 17 confirmed) → all 17
fixed → **133 tests pass**.

## 2. What changed

**New modules**

| File | Lines | Purpose |
|---|---|---|
| `whisper_labia/io/units.py` | ~165 | astropy unit validation/conversion (`to_canonical`, `to_flux_density_jy`, `check_magnitude_unit`, …) |
| `whisper_labia/io/svo.py` | ~230 | SVO fallback — lazy/mockable network boundary, disk+memory cache, `register_manual_band`, `get_transmission_data` |
| `dev/demo_ingestion.py` | ~135 | Offline end-to-end demo of all four features |
| `tests/test_units.py` | 12 tests | unit conversions + error paths |
| `tests/test_svo.py` | 15 tests | band resolution, mocked SVO, caching, corrupt-cache resilience (×4 params), manual override |
| `tests/test_ingestion.py` | 25 tests | redshift, data_mode, `__call__`, loader unit/band integration |

**Modified** (`git diff --stat HEAD`)

```
 CHANGELOG.md                 |  26 ++-
 pyproject.toml               |   3 +-     # [svo] extra = astroquery
 whisper_labia/__init__.py    |  13 +-     # new top-level exports
 whisper_labia/io/__init__.py |  26 ++-
 whisper_labia/io/bands.py    | 103 ++++    # LSST_BAND_INFO + resolve_band/resolve_bands
 whisper_labia/io/loader.py   |  81 ++-     # redshift/units/data_mode/band plumbing
 whisper_labia/io/schema.py   | 174 +++--   # data_mode, redshift state, __call__, enrichment
 7 files changed, 397 insertions(+), 29 deletions(-)
```

## 3. Adversarial review — 17 confirmed findings, all fixed

A 6-dimension review (units, redshift, SVO, backward-compat, spec-compliance, test-gaps) ran each
finding through an independent verifier. 17 of 21 were confirmed and fixed; 4 were correctly dismissed.

### Code defects fixed (6)

| Finding | Severity | Fix |
|---|---|---|
| F_λ→Jy silently produced NaN (then dropped the row → silently empty light curve) when a band's wavelength was unresolved | medium | `to_flux_density_jy` now raises clearly, naming the unresolved point(s) |
| A corrupt-but-valid-JSON SVO cache crashed a load (`ValueError`/`KeyError`, uncaught) | medium | Validate every cache entry on read (`_valid_meta`); malformed → ignored + re-fetched; invalid SVO metadata never cached |
| `add_flux`/`add_mag` diverged from AB 3631 for non-3631 SVO/manual bands, silently rescaling the fitted flux | medium | Restored constant AB 3631 for `add_flux`/`add_mag`; per-band zero point used only in the new `lc()` enrichment |
| `add_flux`/`add_mag` treated band-integrated `flux` mode as Jy density | low | Both raise for `data_mode='flux'` |
| No-unit **magnitude** column did not warn (flux did) | low | Magnitude routed through `to_canonical` |
| Class docstring overclaimed "both flux and magnitude filled in" | low | Qualified for band-integrated flux mode |

### Test gaps closed (11) — +17 tests

per-point wavelength actually varies the result · NaN-wavelength F_λ errors · corrupt-cache resilience
(4 parametrized cases) · `get_transmission_data` mocked · empty-SVO-index-with-hint path · all-NaN
redshift column → unknown · negative redshift column → fatal · `_subset`/`_copy` preserve redshift
state without re-warning · `data_mode='flux'` end-to-end through the loader · `to_canonical('flux')`
dispatch · backward-compat zero-point regression.

### Dismissed (4, correctly)

`dex` accepted as a magnitude unit (harmless logarithmic edge case) · per-row redshift collapse to first
finite (now documented as intended) · NaN token persisted in JSON cache (now filtered on read) ·
"y-band folded into z-band wavelength" (the y→z grouping is intentional pre-existing `FILTER_LOOKUP`
behavior; AB zero points are identical; already commented).

## 4. Offline demo

`dev/demo_ingestion.py` runs end-to-end with **no network** (the SVO boundary is stubbed exactly as
the tests mock it). Selected output:

```
1. data_mode  +  __call__  -> enriched dataframe (both flux and magnitude)
   data_mode = 'magnitude'   output_format = 'magnitude'
    time band  magnitude  magnitude_err     flux  lambda_eff  zero_point       snr
     1.0    g       20.0           0.05 0.000036      4866.0      3631.0 21.71   (flux DERIVED)
     2.0    r       20.4           0.05 0.000025      6215.0      3631.0 21.71
     3.0    i       20.8           0.06 0.000017      7545.0      3631.0 18.10

2. Redshift: argument(0.12) > column(0.34) > arg-beats-column(0.99) > unknown(redshift_known=False)
   z=0.0 -> ValueError (needs luminosity_distance);  z=-0.1 / z=nan -> ValueError;  z=0 + LD=40 Mpc -> OK

3. astropy units:  1 mJy as F_lambda @6215 AA = 7.761e-16 erg/s/cm2/AA  ->  1.000000e-03 Jy  (both routes match)
   loader F_lambda -> Jy = 1.000000e-03 Jy ;  magnitude with a flux unit -> ValueError

4. Bands:  'g' -> lsst (4866 AA, 3631 Jy, no SVO call)
   'PAN-STARRS/PS1.w' -> [network] SVO queried (call #1) -> svo (6200 AA); repeat -> cache hit (still 1 call)
   SVO down -> source='unresolved'; after register_manual_band -> source='manual' (9000 AA)
```

## 5. Test results — 133 passed

`docker exec phe_sbi bash -lc 'cd …/whisper-labia && python -m pytest tests -q'` → **133 passed in ~7 s.**

| File | Tests | Status | Focus |
|---|---:|---|---|
| `test_photometry.py` | 5 | ✅ | AB zeropoint, mag↔flux, error propagation, SNR |
| `test_bands.py` | 11 | ✅ | aliases, case-sensitivity, `group_bands`/`FILTER_LOOKUP`, `unmapped_bands` |
| `test_schema.py` | 11 | ✅ | validation, subsetting, `add_*`, `snr`, explosion date, upper limits |
| `test_loader.py` | 12 | ✅ | AT2017GFO load, window/subset, grouping, `min_snr`, upper limits |
| `test_plotting.py` | 5 | ✅ | report/grid, flux/absolute-mag, redshift guard, upper-limit markers |
| `test_priors.py` | 6 | ✅ | Uniform/LogUniform/Prior sampling, log_prob, rescale, picklability |
| `test_models.py` | 8 | ✅ | flare/bazin/gaussian_rise, custom model, errors |
| `test_distance.py` | 2 | ✅ | χ² zero/known value |
| `test_abc.py` | 6 | ✅ | recovery, serial+parallel, acceptance, JSON, dispatch, custom model |
| `test_abc_smc.py` | 6 | ✅ | SMC registration, recovery, ε tightening, schedule, dispatch |
| `test_likelihood.py` | 9 | ✅ | Gaussian flux/mag, space-auto, upper-limit CDF, mixture, picklable |
| **`test_units.py`** | **12** | ✅ **new** | F_ν/F_λ→Jy, per-point λ, NaN-λ error, mag rejects flux unit, no-unit default, flux dimensionality |
| **`test_svo.py`** | **15** | ✅ **new** | FILTER_LOOKUP→SVO, mocked metadata/index, cache hit, disk cache, corrupt-cache (×4 params), graceful degrade, manual override (no spurious warn), transmission, ambiguity |
| **`test_ingestion.py`** | **25** | ✅ **new** | data_mode (+output_format), `__call__`, redshift (arg/column/unknown/0/neg/NaN/all-NaN), units, band-info, subset preservation, backward-compat ZP, flux-mode |
| **Total** | **133** | ✅ | |

<details>
<summary>Full per-test list (133 PASSED)</summary>

```
test_ingestion.py::test_data_mode_inferred .......................................... PASSED
test_ingestion.py::test_data_mode_explicit_flux_maps_to_flux_density_output ......... PASSED
test_ingestion.py::test_data_mode_invalid_raises ................................... PASSED
test_ingestion.py::test_data_mode_survives_subset ................................. PASSED
test_ingestion.py::test_call_returns_enriched_dataframe ........................... PASSED
test_ingestion.py::test_call_flux_only_fills_magnitude ........................... PASSED
test_ingestion.py::test_redshift_from_argument ................................... PASSED
test_ingestion.py::test_redshift_from_column ..................................... PASSED
test_ingestion.py::test_redshift_argument_overrides_column ....................... PASSED
test_ingestion.py::test_redshift_unknown_warns_and_sets_prior .................... PASSED
test_ingestion.py::test_redshift_zero_requires_distance .......................... PASSED
test_ingestion.py::test_redshift_negative_and_nan_are_fatal ...................... PASSED
test_ingestion.py::test_loader_no_unit_warns_and_assumes_jy ...................... PASSED
test_ingestion.py::test_loader_flambda_converts_with_band_wavelength ............. PASSED
test_ingestion.py::test_loader_flambda_without_wavelength_errors ................. PASSED
test_ingestion.py::test_loader_magnitude_given_flux_unit_errors .................. PASSED
test_ingestion.py::test_loader_known_bands_skip_svo ............................. PASSED
test_ingestion.py::test_loader_magnitude_no_unit_warns .......................... PASSED
test_ingestion.py::test_redshift_column_all_nan_degrades_to_unknown ............. PASSED
test_ingestion.py::test_redshift_column_negative_is_fatal ....................... PASSED
test_ingestion.py::test_subset_preserves_redshift_state_without_rewarning ....... PASSED
test_ingestion.py::test_add_flux_uses_constant_ab_zeropoint_but_dataframe_uses_per_band PASSED
test_ingestion.py::test_add_flux_mag_reject_band_integrated_flux ................ PASSED
test_ingestion.py::test_loader_flux_mode_band_integrated ........................ PASSED
test_ingestion.py::test_loader_flambda_unresolved_band_errors .................. PASSED
test_units.py::test_fnu_mjy_to_jy ............................................. PASSED
test_units.py::test_flux_density_roundtrip_mjy_vs_flambda ..................... PASSED
test_units.py::test_flambda_requires_lambda_eff .............................. PASSED
test_units.py::test_unrecognised_flux_density_unit_errors .................... PASSED
test_units.py::test_magnitude_rejects_flux_unit ............................. PASSED
test_units.py::test_magnitude_accepts_dimensionless_and_mag ................. PASSED
test_units.py::test_flux_band_integrated_dimensionality .................... PASSED
test_units.py::test_no_unit_metadata_warns_and_defaults .................... PASSED
test_units.py::test_unknown_data_mode_errors .............................. PASSED
test_units.py::test_flambda_uses_per_point_wavelength ..................... PASSED
test_units.py::test_flambda_nan_wavelength_with_finite_value_errors ....... PASSED
test_units.py::test_to_canonical_flux_band_integrated_dispatch ........... PASSED
test_svo.py::test_filter_lookup_hit_does_not_call_svo .................... PASSED
test_svo.py::test_absent_band_warns_then_svo_resolves ................... PASSED
test_svo.py::test_repeat_lookup_hits_cache_not_network .................. PASSED
test_svo.py::test_disk_cache_survives_fresh_memory ..................... PASSED
test_svo.py::test_svo_unavailable_degrades_then_manual_override ........ PASSED
test_svo.py::test_unknown_band_no_hint_is_graceful .................... PASSED
test_svo.py::test_ambiguous_index_warns_and_picks_closest ............ PASSED
test_svo.py::test_corrupt_cache_never_crashes_load[[1, 2, 3]] ........ PASSED
test_svo.py::test_corrupt_cache_never_crashes_load[{"X/X.r": "garbage"}] PASSED
test_svo.py::test_corrupt_cache_never_crashes_load[{"X/X.r": {"ZeroPoint": 3631.0}}] PASSED
test_svo.py::test_corrupt_cache_never_crashes_load[{ this is not json] . PASSED
test_svo.py::test_manual_band_resolves_without_warning ............. PASSED
test_svo.py::test_get_transmission_data_mocked ...................... PASSED
test_svo.py::test_empty_svo_index_with_hint_is_graceful ............ PASSED
test_svo.py::test_resolve_bands_vectorized ........................ PASSED
(+ 81 pre-existing tests in test_abc, test_abc_smc, test_bands, test_distance,
   test_likelihood, test_loader, test_models, test_photometry, test_plotting,
   test_priors, test_schema — all PASSED)
```
</details>

## 6. Open items

- **astroquery** is not installed in `phe_sbi`; the package + all tests run without it (mocked). The
  live SVO path needs the `[svo]` extra.
- Change is **uncommitted** — ready to stage.
- Unrelated, still pending: wiring the likelihood layer into the ABC/ABC-SMC samplers.
