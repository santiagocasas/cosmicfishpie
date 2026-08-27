# Paper-faithful neutrino validation track (arXiv:2405.06047v1)

This document describes the `paper_mnuvalidation*` / `common_specs_paper_*` validation
track added alongside the Casas et al. w0waCDM backend validation
(`common_specs_w0waCDM.json`, `mpvalidation.yaml`, `env_01`/`env_02`). That companion
track remains the reference for the original Casas et al. fiducial
(`Omegam=.32, Omegab=.05, h=.67, ns=.96, sigma8=.815584`). This new track instead
reproduces the neutrino-sector validation convention of Euclid preparation:
"Sensitivity to neutrino parameters" (arXiv:2405.06047v1), using that paper's own
fiducial cosmology and one-massive-species mapping.

## Paper fiducial cosmology

| Parameter | Value      | Notes                                   |
|-----------|-----------|------------------------------------------|
| Omegam    | 0.314571  |                                           |
| Omegab    | 0.049199  | (~0.0492)                                 |
| h         | 0.6737    |                                           |
| ns        | 0.96605   |                                           |
| sigma8    | 0.81      |                                           |
| mnu       | 0.06      | eV, one massive species (validation convention) |
| Neff      | 3.044     |                                           |
| w0        | -1.0      |                                           |
| wa        | 0.0       |                                           |

These are the values used in all four `common_specs_paper_*.json` files.

## Parameter name mapping (paper / MontePython aliases -> CosmicFishPie canonical keys)

The reference materials this track was built from (student MontePython `.param` files
and paper appendix notation) use MontePython-style parameter labels that are **not**
valid CosmicFishPie inputs. CosmicFishPie's canonical keys are `mnu` and `Neff` only.
No `mnu_camb`, `m_nu_camb`, `N_eff_camb`, or `Omega_m_camb` aliases exist or are
accepted anywhere in this codebase (`cosmicfishpie/cosmology/cosmology.py`). The
mapping below documents the correspondence purely for readers cross-referencing the
paper/MontePython materials against this repository's configs:

| Paper / MontePython label | CosmicFishPie canonical key | Where used                                  |
|----------------------------|------------------------------|----------------------------------------------|
| `Omega_m_camb`             | `Omegam`                     | `fiducialpars` / `freepars` in common specs   |
| `omega_b` / implied `Omega_b_camb` | `Omegab`              | `fiducialpars` / `freepars` in common specs   |
| `m_nu_camb`                 | `mnu`                        | `fiducialpars` / `freepars` in common specs   |
| `N_eff_camb`                | `Neff`                       | `fiducialpars` / `freepars` in common specs   |

All four `common_specs_paper_*.json` files use only the canonical `Omegam`, `Omegab`,
`h`, `ns`, `sigma8`, `mnu`, `Neff`, `w0`, `wa` keys; the alias column above exists
purely to document provenance, not as an accepted input format.

## ShareDeltaNeff convention (one-massive-species mapping)

All four `common_specs_paper_*.json` files set `"ShareDeltaNeff": false`. This flag
controls whether the `mnu -> omnuh2` (CAMB) / `mnu -> Omega_ncdm` (CLASS) mass-mapping
factor `g_factor` scales with the *free* `Neff` parameter (`g_factor = Neff/3` when
`true`) or stays fixed at the fiducial value (`g_factor = fidNeff/3` when `false`); see
`changebasis_camb`/`changebasis_class` in `cosmicfishpie/cosmology/cosmology.py`.

**This was initially set to `true` and was found to be the root cause of the
photometric `mnu` gate failures in cases 09 and 11 (6.82% and 9.71% deviation,
against a 5% gate).** Cross-checking against the actual paper reference production
Fisher matrices in the companion `Euclid_KP_nu` repository
(`results/cosmicfish_internal/.../_specifications.dat`) showed that **every** paper
production run (`nulcdm` = LCDM+mnu+Neff, and `wCDM+mnu+Neff`, both probes, both
optimistic/pessimistic) uses `ShareDeltaNeff: False`, not `True`. Recomputing those
reference matrices with `cosmicfishpie.analysis.fisher_matrix` confirmed they stay
comfortably within the 5%/2% gates (worst photo deviation 3.83%, worst spectro 1.51%
for the `wCDM+mnu+Neff` family; 1.66% for `nulcdm` optimistic).

Diagnosis (via `scripts/compare_reference_fishers.py`, using
`fisher_matrix.get_confidence_bounds(marginal=True/False)` and
`get_fisher_inverse()`) showed this is a marginalization/correlation effect, not a
derivative bug: **unmarginalized** (diagonal) CAMB-vs-CLASS sigma deviations for case
11 were already tiny (<=1.6%, matching reference-level agreement) under
`ShareDeltaNeff=true`; only the **marginalized** sigmas blew up (mnu 9.71%, h 3.85%,
sigma8 4.03%). `ShareDeltaNeff=true` mathematically entangles the `mnu` mass-mapping
with the free `Neff` parameter via the shared `g_factor`, roughly doubling the
mnu-Neff correlation (~0.44 vs ~0.19) and raising mnu-h correlation (~0.62 vs ~0.45)
compared to `ShareDeltaNeff=false`. That extra parameter degeneracy is what amplifies
any small residual CAMB/CLASS derivative mismatch upon Fisher matrix inversion.

`ShareDeltaNeff=false` is therefore the setting that matches both the paper's actual
validation convention and this repo's own diagonal-level CAMB/CLASS agreement.

CAMB 2.0.3 deprecated `share_delta_neff` as a direct `set_cosmology()` kwarg, but the
code path used here (`camb.set_params(**cambpars)`) still honors it via its
unused-kwarg passthrough (only a harmless deprecation log message is emitted). This was
verified empirically: omitting `share_delta_neff` shifts `N_eff` from 3.044 to
3.0587 (~0.48% error) for this fiducial, so it continues to be passed explicitly and
unconditionally in `changebasis_camb`, regardless of the `ShareDeltaNeff` value.

## Four paper-fiducial model families

| Common specs JSON                          | cosmo_model | Free parameters (step)                                                                 |
|----------------------------------------------|-------------|-------------------------------------------------------------------------------------------|
| `common_specs_paper_LCDM_fixed.json`          | LCDM        | Omegam, Omegab, h, ns, sigma8 (1% each). `mnu` fixed at 60 meV (paper's plain-LCDM convention), Neff fixed. |
| `common_specs_paper_LCDM_mnu.json`            | LCDM        | above + `mnu` (10% step)                                                                  |
| `common_specs_paper_LCDM_mnu_Neff.json`       | LCDM        | above + `mnu` (10%) + `Neff` (1%)                                                          |
| `common_specs_paper_w0wa_mnu_Neff.json`       | w0waCDM     | above + `mnu` (10%) + `Neff` (1%) + `w0` (1%) + `wa` (1%)                                  |

`mnu` always uses a 10% derivative step (paper convention, ~6 meV at fiducial); all
other varied parameters use a 1% step.

## Boltzmann solver profiles

- **Photometric (nonlinear, HMcode2020):**
  `cosmicfishpie/configs/default_boltzmann_yaml_files/class/paper_mnuvalidation_photo.yaml`
  ("HP" profile: `l_max_ncdm=25`, `ncdm_fluid_trigger_tau_over_tau_k=100`,
  `hmcode_tol_sigma=1e-8`) and
  `cosmicfishpie/configs/default_boltzmann_yaml_files/camb/paper_mnuvalidation.yaml`
  (`halofit_version=mead2020`, `num_nu_massive=1`).
- **Spectroscopic (linear P_cb observable):**
  `cosmicfishpie/configs/default_boltzmann_yaml_files/class/paper_mnuvalidation_spectro.yaml`
  ("UHP" profile: `l_max_ncdm=40`, `ncdm_fluid_approximation=3`, `evolver=0`). The
  `non linear` key present in this YAML (and in the shared CAMB YAML) is **inert
  legacy configuration** for the spectroscopic observable: `GCsp_linear` in the common
  specs JSON drives the actual dewiggling/damping treatment in
  `cosmicfishpie/LSSsurvey/spectro_obs.py`, and the spectro Pk is always computed from
  the linear transfer function, never from Halofit/HMcode. This mirrors the paper's
  own treatment (Halofit is explicitly unsuitable for extended-neutrino forecasts, per
  the paper).

None of the pre-existing `nuvalidation_photo.yaml`, `nuvalidation_spectro.yaml`, or
`nuvalidation.yaml` files were modified; they remain supported fallback/package-data
profiles. Their former `env_03`/`env_04` wrappers and `common_specs_nuCDM.json` live in
`scripts/archive/validation_configs/` as historical validation inputs.

## Validation gates

- Photometric cases: max sigma-ratio deviation < **5%**
- Spectroscopic cases: max sigma-ratio deviation < **2%**

Enforced via `--sigma-threshold` in `scripts/run_fisher_compare_backends.py` (wired
through `SIGMA_THRESHOLD` in `scripts/compare_backends_report.sh` and each
`compare_run_config.env_*` file below). The gate reads `pairwise[].analysis
.param_sigma_ratio.<name>.ratio_b_over_a` from the `compare_fishers_in_dir.py` output
JSON and exits nonzero if `max(|ratio - 1| * 100) > SIGMA_THRESHOLD`.

## Running the four model families (8 cases: photo + spectro each)

```bash
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_07_camb_class_photo_papervalidation_LCDM_fixed
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_08_camb_class_spectro_papervalidation_LCDM_fixed
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_09_camb_class_photo_papervalidation_LCDM_mnu
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_10_camb_class_spectro_papervalidation_LCDM_mnu
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_11_camb_class_photo_papervalidation_LCDM_mnu_Neff
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_12_camb_class_spectro_papervalidation_LCDM_mnu_Neff
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_13_camb_class_photo_papervalidation_w0wa_mnu_Neff
scripts/compare_backends_report.sh --config scripts/validation_configs/compare_run_config.env_14_camb_class_spectro_papervalidation_w0wa_mnu_Neff
```

Each case is `CODE_A=camb` vs `CODE_B=class`, matching the convention used by the
archived historical `env_03`/`env_04` nuvalidation cases. Each run writes provenance
(backend versions, install source, VCS commit where available, git dirty state,
platform) to `run_metadata.json` in its output directory via
`scripts/run_fisher_compare_backends.py`.

**Note:** because the `ShareDeltaNeff` convention changed (see above), all cases
07-14 must be (re-)run against the corrected common-specs before their results can be
trusted; any prior results generated with `ShareDeltaNeff: true` are stale.

## Cross-checking against paper reference Fisher matrices

`scripts/compare_reference_fishers.py` is a standalone CLI (built on
`cosmicfishpie.analysis.fisher_matrix.fisher_matrix`, no hand-rolled linear algebra)
for comparing any two raw `*_fishermatrix.txt` + `.paramnames` pairs and reporting
per-parameter 1-sigma deviations:

```bash
uv run python scripts/compare_reference_fishers.py \
  --fisher-a <path>/CosmicFish_v1.0_..._camb-..._fishermatrix.txt \
  --fisher-b <path>/CosmicFish_v1.0_..._class-..._fishermatrix.txt \
  --label-a camb --label-b class \
  --threshold 5.0
```

This was used to cross-check this repo's cases against the actual paper production
Fisher matrices in the companion `Euclid_KP_nu` repository during the
`ShareDeltaNeff` root-cause investigation above.
