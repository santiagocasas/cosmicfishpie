# Einstein-Boltzmann solver profiles

Every file in `camb/` and `class/` is a plain, self-contained YAML with real settings --
there is no shared registry and no indirection layer to follow. Open any file and you see
exactly what CAMB/CLASS will be run with.

## Defaults (used automatically when nothing is specified)

| Selector | Intended use |
|---|---|
| `camb/default.yaml` | CAMB, fast/relaxed settings, used for both probes |
| `class/default.yaml` | CLASS, fast/relaxed settings, photometric probe |
| `class/default_spectro.yaml` | CLASS, fast/relaxed settings, spectroscopic probe |

These are what `cosmicfishpie.configs.config` falls back to when `camb_config_yaml` /
`class_config_yaml` is not given. They are meant for exploration and development, not for
publication-grade Fisher forecasts -- do not cite results computed with them without first
validating against the high-precision profiles below.

## Paper-validated high precision (arXiv:2405.06047v1)

| Selector | Intended use |
|---|---|
| `camb/nuvalidation_hp.yaml` | CAMB HP, used for both photo and spectro |
| `class/nuvalidation_hp.yaml` | CLASS HP, photometric probe |
| `class/nuvalidation_uhp.yaml` | CLASS UHP, spectroscopic probe (deeper neutrino-perturbation treatment) |

These reproduce the neutrino-sector validation's Appendix settings and are what the
`scripts/validation_configs/compare_run_config.env_*` paper-validation cases use. The
CLASS photo (HP) and spectro (UHP) tiers intentionally differ -- the paper validates each
probe with its own precision tier; CAMB does not need the split.

## Historical MontePython validation (arXiv:2303.09451v1)

| Selector | Paper setting |
|---|---|
| `camb/mpvalidation_p3.yaml` | CAMB P3 |
| `class/mpvalidation_hp.yaml` | CLASS HP, Appendix A.5 |
| `class/mpvalidation_dp.yaml` | CLASS DP comparison |

`camb/mpvalidation_p3.yaml` documents the CAMB P1/P2/P3 relationship and the historical
patched `halofit_tol_sigma` values as provenance comments. Current CAMB does not expose
that patched tolerance, so it is not passed as a runtime parameter. Neutrino density and
temperature values that vary with the cosmological model remain the responsibility of the
CosmicFishPie parameter conversion and are not hard-coded here.

## Overrides / one-off variants

Need a small tweak to one of the profiles above (e.g. a precision-sensitivity test)?
Copy the file and change the values you need -- there is nothing to "inherit" or point
back at. Such files should live with the experiment that needs them, not among the package
defaults. For example, the strict CLASS photo neutrino-precision sensitivity test lives at
`scripts/validation_configs/alternatives/class_photo_strict.yaml`.
