# Paper-faithful neutrino validation track (arXiv:2405.06047v1)

This document describes the `paper_mnuvalidation*` / `common_specs_paper_*` validation
track added alongside the Casas et al. w0waCDM backend cross-check
(`common_specs_w0waCDM.json`, `mpvalidation.yaml`, cases `01.1.*`/`01.2.*`). That companion
track remains the reference for the original Casas et al. fiducial
(`Omegam=.32, Omegab=.05, h=.67, ns=.96, sigma8=.815584`,
arXiv:2303.09451v1). This track instead reproduces the neutrino-sector validation
convention of Euclid preparation: "Sensitivity to neutrino parameters"
(arXiv:2405.06047v1), using that paper's own fiducial cosmology and one-massive-species
mapping.

## Case ID scheme

Case IDs are dotted hierarchical strings read live from
`compare_run_config.env_<ID>_<description>` filenames by
`scripts/run_selected_validations.sh` and `scripts/render_validation_dashboard.py` (see
`--help` on the former for the exact grammar and group-selection semantics). The root
groups below separate **formal paper validation** (Sec. 6 of arXiv:2405.06047v1) from
controls and stress tests that are useful diagnostics but are not part of the paper's
own validation scope:

| Group | Status | Model |
|-------|--------|-------|
| `01.*` | Cross-check (arXiv:2303.09451v1) | Casas et al. w0waCDM MontePython validation |
| `02.*` | **Formal paper model 1** [Sec. 6] | w0waCDM, `mnu`/`Neff` fixed |
| `03.*` | **Formal paper model 2** [Sec. 6] | LCDM + free `mnu`, `Neff` |
| `04.*` | **Formal paper model 3** [Sec. 6] | w0CDM (i.e. `wa` fixed) + free `mnu`, `Neff` fixed |
| `05.*` | Reduced control (not a Sec. 6 model) | LCDM, fixed `mnu`/`Neff` |
| `06.*` | Reduced control (not a Sec. 6 model) | LCDM + free `mnu`, fixed `Neff` |
| `07.*` | Stress test (not a Sec. 6 model) | w0waCDM + free `mnu`, `Neff` fixed (8 free cosmological params) |
| `08.*` | Stress test, **explicitly excluded** by Sec. 6 | w0waCDM + free `mnu`, `Neff` (9 free cosmological params) |

For cross-check/formal groups `01.*`-`04.*`, the ID is `<model>.<probe>.<scenario>`:
probe `.1` is spectroscopic and `.2` is photometric; scenario `.0` is pessimistic and
`.1` is optimistic. For example, `03.1.0` is model 2 spectroscopic pessimistic and
`03.2.1` is model 2 photometric optimistic. Reduced controls `05.*`/`06.*` use the same
three-level form but currently expose only pessimistic `.0` leaves. Stress tests `07.*`
and `08.*` now follow the same scenario convention (`07.1.0`, `07.2.0`, `08.1.0`,
`08.2.0`). An optional fourth segment identifies a variant of an otherwise identical
leaf: canonical case `07.2.0` uses P_cb, while `07.2.0.1` is variant 1 and intentionally
uses the non-paper P_mm galaxy tracer. Variant segments are unpadded (`.1`, `.2`, ...)
because the runner normalizes only the root segment; future precision or parameter-set
variants can use subsequent values.

The former `03.2.1` strict-CLASS-precision experiment is no longer a primary case. It
is retained as alternative `01.4` under `alternatives/`, where it cannot be mistaken
for the paper's optimistic photometric case.

## Why the full 9-parameter model (`08.*`) is not a formal paper model

Per `SensitivityNeutrinos.htm` (the paper's HTML source, Sec. 6, arXiv:2405.06047v1),
the paper explicitly rejects a single simultaneous 9-parameter Fisher validation
(5 base LCDM + `mnu`, `Neff`, `w0`, `wa`) due to parameter degeneracies, non-Gaussian
posteriors, and derivative-step sensitivity in that combined parameter space. It
 validates three reduced 7-parameter model families instead (`02.*`, `03.*`, `04.*`
above). `08.*` (full 9-parameter `w0waCDM+mnu+Neff`) and `07.*` (8-parameter
`w0waCDM+mnu`, `Neff` fixed) are retained in this repo as internal stress
tests/diagnostics of degeneracy amplification under marginalization (see
`CASE13_W0WA_MNU_INVESTIGATION.md` and `PHOTO_MARGINALIZATION_DIAGNOSIS.md`), not as
paper-faithful validation.

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

These are the values used in all seven `common_specs_paper_*.json` files (see table
below). The `01.*` cross-check cases use the older Casas et al. (2303.09451v1) fiducial
instead (`common_specs_w0waCDM.json`: `Omegam=.32, Omegab=.05, h=.67, ns=.96,
sigma8=.815584`).

## Galaxy tracer and intrinsic-alignment convention

All seven `common_specs_paper_*.json` files set `GCsp_Tracer` and `GCph_Tracer` to
`clustering`, which selects the CDM+baryon spectrum P_cb (called P_cc in the paper).
This follows Sec. 3 of arXiv:2405.06047v1: galaxies trace CDM+baryons rather than total
matter in massive-neutrino cosmologies. Weak lensing remains on total matter P_mm in
the photometric implementation.

This corrects the earlier validation inputs, which had both tracer settings at
`matter`. In a model-2 photometric pessimistic precursor that still varied `betaIA`,
changing to P_cb reduced the marginalized CAMB-vs-CLASS `mnu` deviation from 10.06% to
5.53%. The raw unmarginalized difference changed little; the gain came from reducing
degeneracy and Fisher-inversion amplification. The final fixed-`betaIA` batch passed all
formal and stress-test gates: model-2 photo gives 5.53% (pessimistic) and 2.60%
(optimistic), while the 8- and 9-parameter pessimistic photo stress tests give 6.34% and
5.10%, respectively. Because P_cb and fixed `betaIA` were introduced together relative
to the older stress-test matrices, those improvements cannot be attributed to
`betaIA` alone. See `PCB_NEUTRINO_DERIVATIVE_ANALYSIS.md` for the direct P_mm/P_cb
derivative comparison and the Fisher-conditioning explanation.

The dedicated paper survey YAMLs also implement the paper's nuisance convention:
`AIA` and `etaIA` vary while `betaIA=2.17` is fixed. Both arXiv:2405.06047v1 and
arXiv:2303.09451v1 state this explicitly. Some historical production `.paramnames`
files nevertheless include `betaIA`; those legacy outputs are retained as provenance,
not copied into the canonical definitions. Free-`betaIA` and P_mm variants live under
`scripts/validation_configs/alternatives/`.

## Survey scenarios

The formal models use dedicated survey definitions sourced from Tables 1 and 2:

| Scenario | Photo `lmax_GCph` / `lmax_WL` | Spectro `kmax_GCsp` |
|----------|---------------------------------|---------------------|
| Pessimistic (`*.0`) | 750 / 1500 | 0.25 |
| Optimistic (`*.1`) | 3000 / 5000 | 0.30 |

The selected survey name and survey-YAML content hash are part of each run's config
hash and metadata, preventing pessimistic and optimistic outputs from colliding or
being reused across scenarios.

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

All seven `common_specs_paper_*.json` files use only the canonical `Omegam`, `Omegab`,
`h`, `ns`, `sigma8`, `mnu`, `Neff`, `w0`, `wa` keys; the alias column above exists
purely to document provenance, not as an accepted input format.

## ShareDeltaNeff convention (one-massive-species mapping)

All seven `common_specs_paper_*.json` files set `"ShareDeltaNeff": false`. This flag
controls whether the `mnu -> omnuh2` (CAMB) / `mnu -> Omega_ncdm` (CLASS) mass-mapping
factor `g_factor` scales with the *free* `Neff` parameter (`g_factor = Neff/3` when
`true`) or stays fixed at the fiducial value (`g_factor = fidNeff/3` when `false`); see
`changebasis_camb`/`changebasis_class` in `cosmicfishpie/cosmology/cosmology.py`.

**This was initially set to `true` and was found to be the root cause of early
photometric `mnu` gate failures in the LCDM+mnu and LCDM+mnu+Neff cases (now `06.2.0` and
`03.2.0`).** Cross-checking against the actual paper reference production Fisher matrices
in the companion `Euclid_KP_nu` repository
(`results/cosmicfish_internal/.../_specifications.dat`) showed that **every** paper
production run (`nulcdm` = LCDM+mnu+Neff, and `wCDM+mnu+Neff`, both probes, both
optimistic/pessimistic) uses `ShareDeltaNeff: False`, not `True`. Recomputing those
reference matrices with `cosmicfishpie.analysis.fisher_matrix` confirmed they stay
comfortably within gate-level agreement (worst photo deviation 3.83%, worst spectro
1.51% for the `wCDM+mnu+Neff` family; 1.66% for `nulcdm` optimistic).

Diagnosis (via `scripts/compare_reference_fishers.py`, using
`fisher_matrix.get_confidence_bounds(marginal=True/False)` and
`get_fisher_inverse()`) showed this is a marginalization/correlation effect, not a
derivative bug: **unmarginalized** (diagonal) CAMB-vs-CLASS sigma deviations for the
LCDM+mnu+Neff photo case were already tiny (<=1.6%, matching reference-level agreement)
under `ShareDeltaNeff=true`; only the **marginalized** sigmas blew up (mnu 9.71%, h
3.85%, sigma8 4.03%). `ShareDeltaNeff=true` mathematically entangles the `mnu`
mass-mapping with the free `Neff` parameter via the shared `g_factor`, roughly doubling
the mnu-Neff correlation (~0.44 vs ~0.19) and raising mnu-h correlation (~0.62 vs
~0.45) compared to `ShareDeltaNeff=false`. That extra parameter degeneracy is what
amplifies any small residual CAMB/CLASS derivative mismatch upon Fisher matrix
inversion.

`ShareDeltaNeff=false` is therefore the setting that matches both the paper's actual
validation convention and this repo's own diagonal-level CAMB/CLASS agreement.

CAMB 2.0.3 deprecated `share_delta_neff` as a direct `set_cosmology()` kwarg. This
repository now assigns it directly on the constructed `CAMBparams` object via a small
`_set_camb_params()` helper in `cosmicfishpie/cosmology/cosmology.py` (see git history:
"fix for deprecated sharedeltaneff in camb"), which sets `share_delta_neff` explicitly
and unconditionally after `camb.set_params(...)` without emitting CAMB's deprecation
log. This was verified empirically: omitting `share_delta_neff` shifts `N_eff` from
3.044 to 3.0587 (~0.48% error) for this fiducial, so it continues to be applied
explicitly regardless of the `ShareDeltaNeff` value.

## Eight paper-fiducial model families

| Common specs JSON                              | cosmo_model | Group  | Free parameters (step)                                                                 |
|--------------------------------------------------|-------------|--------|-------------------------------------------------------------------------------------------|
| `common_specs_paper_LCDM_fixed.json`              | LCDM        | `05.1.0`/`05.2.0` | Omegam, Omegab, h, ns, sigma8 (1% each). `mnu` fixed at 60 meV, Neff fixed. |
| `common_specs_paper_LCDM_mnu.json`                | LCDM        | `06.1.0`/`06.2.0` | above + `mnu` (10% step)                                                     |
| `common_specs_paper_LCDM_mnu_Neff.json`           | LCDM        | `03.1.*`/`03.2.*` | above + `mnu` (10%) + `Neff` (1%) -- **formal paper model 2**       |
| `common_specs_paper_w0wa_fixed_mnu_Neff.json`     | w0waCDM     | `02.1.*`/`02.2.*` | Omegam, Omegab, h, ns, sigma8, w0, wa (1% each). `mnu`, `Neff` fixed -- **formal paper model 1** |
| `common_specs_paper_w0_mnu_fixed_Neff.json`       | w0waCDM     | `04.1.*`/`04.2.*` | Omegam, Omegab, h, ns, sigma8 (1%), mnu (10%), w0 (1%). `wa` fixed at 0, `Neff` fixed -- **formal paper model 3** |
| `common_specs_paper_w0wa_mnu_fixed_Neff.json`     | w0waCDM     | `07.1.0`/`07.2.0` | Omegam, Omegab, h, ns, sigma8 (1%), mnu (10%), w0, wa (1%). `Neff` fixed -- stress test |
| `common_specs_paper_w0wa_mnu_Neff.json`           | w0waCDM     | `08.1.0`/`08.2.0` | Omegam, Omegab, h, ns, sigma8 (1%), mnu (10%), Neff (1%), w0, wa (1%) -- stress test, excluded by Sec. 6 |
| `common_specs_paper_w0wa_mnu_fixed_Neff_Pmm.json` | w0waCDM     | `07.2.0.1` | Same free parameters as `07.2.0`, but both galaxy tracer settings intentionally use P_mm -- beyond-paper diagnostic variant |

`mnu` always uses a 10% derivative step (paper convention, ~6 meV at fiducial); all
other varied parameters use a 1% step.

Note: `common_specs_paper_w0wa_fixed_mnu_Neff.json` (model 1: `mnu` **and** `Neff`
fixed, `w0`/`wa` free) and `common_specs_paper_w0wa_mnu_fixed_Neff.json` (the `07.*`
stress test: `mnu`/`w0`/`wa` free, only `Neff` fixed) are similarly named but distinct
-- do not confuse them.

## Boltzmann solver profiles

- **Photometric (nonlinear, HMcode2020):**
  `cosmicfishpie/configs/default_boltzmann_yaml_files/class/paper_mnuvalidation_photo.yaml`
  is the paper's actual photo "HP" profile per its Appendix (`l_max_ncdm=25`,
  `ncdm_fluid_trigger_tau_over_tau_k=100`, `hmcode_tol_sigma=1e-8`) and
  `cosmicfishpie/configs/default_boltzmann_yaml_files/camb/paper_mnuvalidation.yaml`
  (`halofit_version=mead2020`, `num_nu_massive=1`) is used for every photo case
  (all `.2.*` formal cases plus the pessimistic photo controls/stress tests).
  `paper_mnuvalidation_photo_HP.yaml` (`l_max_ncdm=40`,
  `ncdm_fluid_trigger_tau_over_tau_k=90`) is a **stricter, non-paper** CLASS precision
  variant used only by alternative `01.4`, to test whether raising CLASS's ncdm precision above
  what the paper itself specifies narrows the CAMB/CLASS `mnu` marginalized deviation.
  It should not be read as "the paper's HP profile" -- `paper_mnuvalidation_photo.yaml`
  (used by all canonical photo cases) already is that.
- **Spectroscopic (linear P_cb observable):**
  `cosmicfishpie/configs/default_boltzmann_yaml_files/class/paper_mnuvalidation_spectro.yaml`
  ("UHP" profile: `l_max_ncdm=40`, `ncdm_fluid_approximation=3`, `evolver=0`), used for
  every canonical spectro case. The
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

## Validation gate

All cases (`01.*` through `08.*`) currently use `SIGMA_THRESHOLD=10` (max sigma-ratio
deviation < 10%), set per-case in each `compare_run_config.env_<ID>_*` file. Enforced
via `--sigma-threshold` in `scripts/run_fisher_compare_backends.py` (wired through
`SIGMA_THRESHOLD` in `scripts/compare_backends_report.sh` and each
`compare_run_config.env_*` file). The gate reads `pairwise[].analysis
.param_sigma_ratio.<name>.ratio_b_over_a` from the `compare_fishers_in_dir.py` output
JSON and exits nonzero if `max(|ratio - 1| * 100) > SIGMA_THRESHOLD`.

## Running the formal paper models (models 1-3, 12 probe/scenario cases)

```bash
uv run bash scripts/run_selected_validations.sh --cases 02,03,04 --omp-threads 8
```

The group command runs pessimistic and optimistic scenarios for both probes. A probe
prefix such as `--cases 03.2` runs both model-2 photometric scenarios, while an exact
leaf such as `--cases 03.2.0` runs only the pessimistic case. Use `--cases 05,06,07,08`
for controls/stress tests, or `--all` for every primary case. Alternative scenarios
are intentionally excluded from discovery; see `alternatives/README.md`.

Each case is `CODE_A=camb` vs `CODE_B=class`, matching the convention used by the
archived historical `env_03`/`env_04` nuvalidation cases. Each run writes provenance
(backend versions, install source, VCS commit where available, git dirty state,
platform) to `run_metadata.json` in its output directory via
`scripts/run_fisher_compare_backends.py`.

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
  --threshold 10.0
```

This was used to cross-check this repo's cases against the actual paper production
Fisher matrices in the companion `Euclid_KP_nu` repository during the
`ShareDeltaNeff` root-cause investigation above, and again to confirm that the old
P_mm/free-`betaIA` precursor to case `03.2.0`
(formerly numbered "case 11") reproduces the historical published `nulcdm_external`
P3-CAMB vs HP-CLASS marginalized `mnu` deviation almost exactly (10.06% vs 10.07%),
showing the paper's own original validation already exhibited this deviation -- see
`PHOTO_MARGINALIZATION_DIAGNOSIS.md` for the full root-cause analysis.
