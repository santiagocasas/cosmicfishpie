# Einstein-Boltzmann solver profiles

`precision_profiles.yaml` is the single authoritative registry for numerical and
model-level solver settings. Files in the `camb/` and `class/` subdirectories are small
selectors: they name one profile and contain no duplicated numerical configuration.

## Validated defaults (arXiv:2405.06047v1)

| Selector | Profile | Intended use |
|---|---|---|
| `camb/default.yaml` | `camb_hp` | CAMB HP for photo and spectro |
| `class/default.yaml` | `class_hp` | CLASS HP for photo |
| `class/default_spectro.yaml` | `class_uhp` | CLASS UHP for spectro |

These are the package defaults for scientific validation. The CLASS photo and spectro
profiles intentionally differ: the paper validates HP for photo and UHP for spectro.

## Experimental fast profiles

`camb/fast_photo.yaml`, `camb/fast_spectro.yaml`, `class/fast_photo.yaml`, and
`class/fast_spectro.yaml` select relaxed speed-oriented profiles. They are not validated
substitutes for the canonical defaults. Use them for explicit speed/accuracy experiments
and establish numerical convergence before scientific production use.

## Historical MontePython validation (arXiv:2303.09451v1)

| Selector | Profile | Paper setting |
|---|---|---|
| `camb/mpvalidation_p3.yaml` | `camb_mpvalidation_p3` | CAMB P3 |
| `class/mpvalidation_hp.yaml` | `class_mpvalidation_hp` | CLASS HP |
| `class/mpvalidation_dp.yaml` | `class_mpvalidation_dp` | CLASS DP comparison |

The central registry records the CAMB P1/P2/P3 relationship and the historical patched
`halofit_tol_sigma` values as provenance comments. Current CAMB does not expose that
patched tolerance, so it is not passed as a runtime parameter. Neutrino density and
temperature values that vary with the cosmological model remain the responsibility of the
CosmicFishPie parameter conversion and are not hard-coded into generic solver profiles.

## Overrides

A selector may override individual profile values by adding local mappings such as
`ACCURACY` or `COSMO_SETTINGS`; local values are merged over the selected profile. Such
files should live with the experiment that needs them, not among package defaults. The
strict CLASS photo sensitivity test is therefore stored at
`scripts/validation_configs/alternatives/class_photo_strict.yaml`.
