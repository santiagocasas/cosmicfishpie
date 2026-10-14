# Scripts reference

This page covers the maintained command-line tools under `scripts/`; historical and retired tools in `scripts/archive/` are intentionally omitted. Run commands from the repository root with `uv run` for Python scripts. The **Usage** lines reproduce each CLI's help synopsis; shell and Slurm launchers show their invocation form where they do not provide a conventional help screen.

## Validation

Tools for running backend comparisons and browsing their reports. Most are also called by the selected-validation workflow.

### `run_selected_validations.sh`

Source: [`scripts/validation/run_selected_validations.sh`](../../scripts/validation/run_selected_validations.sh)

Discovers the configured validation cases and runs selected CAMB-versus-CLASS comparisons, then refreshes the dashboard. Use `--list` to inspect available cases before choosing them.

```text
Usage: bash scripts/validation/run_selected_validations.sh --cases LIST [OPTIONS]
       bash scripts/validation/run_selected_validations.sh --all [OPTIONS]
```

### `compare_backends_report.sh`

Source: [`scripts/validation/compare_backends_report.sh`](../../scripts/validation/compare_backends_report.sh)

Runs the two configured backend forecasts, comparison, plots, and report generation for one case.

```text
Usage: compare_backends_report.sh [--config path]
```

### `run_fisher_compare_backends.py`

Source: [`scripts/validation/run_fisher_compare_backends.py`](../../scripts/validation/run_fisher_compare_backends.py)

Computes two Fisher forecasts with selected backends and optionally compares their matrices and plots.

```text
Usage: run_fisher_compare_backends.py [-h] [--mode {photo,spectro}] [--accuracy ACCURACY] [--feedback FEEDBACK] [--omp-threads OMP_THREADS] [--code-a CODE_A] [--code-b CODE_B] [--yaml-a YAML_A] [--yaml-b YAML_B] [--yaml-key-a YAML_KEY_A] [--yaml-key-b YAML_KEY_B] [--outdir OUTDIR] [--compare] [--plot] [--fom-params FOM_PARAMS] [--common-specs COMMON_SPECS] [--survey-name-photo SURVEY_NAME_PHOTO] [--survey-name-spectro SURVEY_NAME_SPECTRO] [--sigma-threshold SIGMA_THRESHOLD]
```

### `compare_fishers_in_dir.py`

Source: [`scripts/validation/compare_fishers_in_dir.py`](../../scripts/validation/compare_fishers_in_dir.py)

Compares Fisher matrices in a result directory, with optional reference and parameter-pair selection.

```text
Usage: compare_fishers_in_dir.py [-h] [--ref REF] [--out OUT] [--fom-params FOM_PARAMS] [--pair SPECS_A SPECS_B] dir
```

### `plot_compare_fishers.py`

Source: [`scripts/validation/plot_compare_fishers.py`](../../scripts/validation/plot_compare_fishers.py)

Plots the Fisher comparison results stored in a JSON file.

```text
Usage: plot_compare_fishers.py [-h] [--outdir OUTDIR] [--fom-params FOM_PARAMS] json
```

### `render_compare_reports.py`

Source: [`scripts/validation/render_compare_reports.py`](../../scripts/validation/render_compare_reports.py)

Renders comparison-result folders as Markdown/HTML reports and can bundle selected outputs.

```text
Usage: render_compare_reports.py [-h] [--glob GLOB] [--index-dir INDEX_DIR] [--formats {md,html,both}] [--bundle-dir BUNDLE_DIR] [--zip] [--single-file SINGLE_FILE] [folders ...]
```

### `render_validation_dashboard.py`

Source: [`scripts/validation/render_validation_dashboard.py`](../../scripts/validation/render_validation_dashboard.py)

Builds the case overview and detailed HTML pages from validation configs and benchmark results; optionally serves them locally.

```text
Usage: render_validation_dashboard.py [-h] [--config-dir CONFIG_DIR] [--results-dir RESULTS_DIR] [--out-dir OUT_DIR] [--serve] [--host HOST] [--port PORT] [--check-completed CASE]
```

### `publish_validation_dashboard.sh`

Source: [`scripts/validation/publish_validation_dashboard.sh`](../../scripts/validation/publish_validation_dashboard.sh)

Publishes the generated validation dashboard and static landing-page assets to GitHub Pages. This changes the publication checkout and should only be run when you intend to publish.

```text
Invocation: bash scripts/validation/publish_validation_dashboard.sh
```

### `install_camb_fork_for_validation.sh`

Source: [`scripts/validation/setup/install_camb_fork_for_validation.sh`](../../scripts/validation/setup/install_camb_fork_for_validation.sh)

Installs the pinned CAMB fork into the project environment for backend validation, without changing the package's standard CAMB dependency pin.

```text
Usage: scripts/validation/setup/install_camb_fork_for_validation.sh [options]
```

## Planck diagnostics

Forecasts and comparisons against Planck chains and published covariance products.

### `run_planck_bestfit_fisher.py`

Source: [`scripts/planck/run_planck_bestfit_fisher.py`](../../scripts/planck/run_planck_bestfit_fisher.py)

Builds a CMB Fisher forecast at a selected Planck-chain best fit, with configurable parameterization, observables, noise, and solver settings.

```text
Usage: run_planck_bestfit_fisher.py [-h] [--planck-dir PLANCK_DIR] [--chain-root CHAIN_ROOT] [--bestfit-source {likestats,minimum}] [--parameterization {h,theta}] [--outdir OUTDIR] [--observables OBSERVABLES] [--lmin LMIN] [--lmax LMAX] [--accuracy ACCURACY] [--feedback FEEDBACK] [--derivatives DERIVATIVES] [--h-step-abs H_STEP_ABS] [--theta-step-abs THETA_STEP_ABS] [--fsky FSKY] [--beam-arcmin BEAM_ARCMIN] [--temp-sens TEMP_SENS] [--pol-sens POL_SENS] [--cmb-noise-model {legacy,knox}] [--ee-lowell-noise-boost EE_LOWELL_NOISE_BOOST] [--ee-lowell-max-ell EE_LOWELL_MAX_ELL] [--camb-yaml CAMB_YAML]
```

### `run_planck_diagnostics_suite.sh`

Source: [`scripts/planck/run_planck_diagnostics_suite.sh`](../../scripts/planck/run_planck_diagnostics_suite.sh)

Runs the standard set of Planck best-fit forecast variants used by the covariance/noise diagnostic notebook.

```text
Invocation: bash scripts/planck/run_planck_diagnostics_suite.sh [OPTIONS]
```

### `compare_planck_published.py`

Source: [`scripts/planck/compare_planck_published.py`](../../scripts/planck/compare_planck_published.py)

Compares a generated Fisher forecast with published Planck-chain constraints.

```text
Usage: compare_planck_published.py [-h] [--planck-dir PLANCK_DIR] [--chain-root CHAIN_ROOT] [--out OUT] fisher
```

### `plot_planck_covmat_vs_fisher.py`

Source: [`scripts/planck/plot_planck_covmat_vs_fisher.py`](../../scripts/planck/plot_planck_covmat_vs_fisher.py)

Plots Planck covariance constraints against a Fisher forecast for selected parameters.

```text
Usage: plot_planck_covmat_vs_fisher.py [-h] [--planck-dir PLANCK_DIR] [--chain-root CHAIN_ROOT] [--params PARAMS] [--outdir OUTDIR] fisher
```

## Likelihood tools

### Diagnostics

#### `run_syren_new_likelihoods.py`

Source: [`scripts/likelihood/diagnostics/run_syren_new_likelihoods.py`](../../scripts/likelihood/diagnostics/run_syren_new_likelihoods.py)

Runs the new likelihood calculations for the SYREN backend on GCsp, photometric probes, or both.

```text
Usage: run_syren_new_likelihoods.py [-h] [--probe {gcsp,photo,both}] [--symbolic-yaml SYMBOLIC_YAML]
```

#### `gcsp_likelihood_grid.py`

Source: [`scripts/likelihood/diagnostics/gcsp_likelihood_grid.py`](../../scripts/likelihood/diagnostics/gcsp_likelihood_grid.py)

Evaluates and saves a two-parameter GCsp likelihood grid around its fiducial or focused point.

```text
Usage: gcsp_likelihood_grid.py [-h] [--nx NX] [--ny NY] [--width WIDTH] [--focus FOCUS] [--outdir OUTDIR]
```

#### `gcsp_likelihood_hessian.py`

Source: [`scripts/likelihood/diagnostics/gcsp_likelihood_hessian.py`](../../scripts/likelihood/diagnostics/gcsp_likelihood_hessian.py)

Checks the GCsp likelihood curvature/Hessian using configurable finite-difference step fractions.

```text
Usage: gcsp_likelihood_hessian.py [-h] [--step-fractions STEP_FRACTIONS [STEP_FRACTIONS ...]] [--outdir OUTDIR]
```

#### `wl_likelihood_grid.py`

Source: [`scripts/likelihood/diagnostics/wl_likelihood_grid.py`](../../scripts/likelihood/diagnostics/wl_likelihood_grid.py)

Computes a two-parameter weak-lensing likelihood grid, with controls for marginal/conditional scaling and parallel workers.

```text
Usage: wl_likelihood_grid.py [-h] [--params X Y] [--free-params FREE_PARAMS] [--nx NX] [--ny NY] [--scale {marginal,conditional}] [--width WIDTH] [--focus FOCUS] [--workers WORKERS] [--outdir OUTDIR]
```

#### `verify_gcsp_covariance_sympy.py`

Source: [`scripts/likelihood/diagnostics/verify_gcsp_covariance_sympy.py`](../../scripts/likelihood/diagnostics/verify_gcsp_covariance_sympy.py)

Uses SymPy to derive and print symbolic checks of the GCsp covariance expressions. It has no CLI help parser or options; run `uv run python scripts/likelihood/diagnostics/verify_gcsp_covariance_sympy.py` to execute it.

### Nautilus sampling

#### `wl_gcsp_fisher_nautilus_demo.py`

Source: [`scripts/likelihood/nautilus/wl_gcsp_fisher_nautilus_demo.py`](../../scripts/likelihood/nautilus/wl_gcsp_fisher_nautilus_demo.py)

Runs a quick WL, GCsp, and joint Fisher-versus-Nautilus demonstration and writes comparison plots.

```text
Usage: wl_gcsp_fisher_nautilus_demo.py [-h] [--free-params FREE_PARAMS] [--n-live N_LIVE] [--n-workers N_WORKERS] [--sample-nuisances] [--nuisance-sigma NUISANCE_SIGMA] [--skip-nautilus] [--outdir OUTDIR] [--quiet-sampler]
```

#### `run_wl_gcsp_nautilus_case.py`

Source: [`scripts/likelihood/nautilus/run_wl_gcsp_nautilus_case.py`](../../scripts/likelihood/nautilus/run_wl_gcsp_nautilus_case.py)

Runs one WL, GCsp, or joint Nautilus sampling case and stores its chain and metadata in the requested run directory.

```text
Usage: run_wl_gcsp_nautilus_case.py [-h] --case {wl,gcsp,joint} --run-dir RUN_DIR [--free-params FREE_PARAMS] [--sample-nuisances] [--nuisance-sigma NUISANCE_SIGMA] --n-live N_LIVE --n-eff N_EFF --workers WORKERS [--resume] [--quiet-sampler]
```

#### `postprocess_wl_gcsp_nautilus.py`

Source: [`scripts/likelihood/nautilus/postprocess_wl_gcsp_nautilus.py`](../../scripts/likelihood/nautilus/postprocess_wl_gcsp_nautilus.py)

Summarizes a completed Nautilus run and produces chain-versus-Fisher statistics and plots.

```text
Usage: postprocess_wl_gcsp_nautilus.py [-h] --run-dir RUN_DIR
```

### Postprocessing

#### `render_likelihood_validation.py`

Source: [`scripts/likelihood/postprocess/render_likelihood_validation.py`](../../scripts/likelihood/postprocess/render_likelihood_validation.py)

Collects copied WL, GCsp, and joint HPC run artifacts into compact statistics and figures for the likelihood-validation documentation.

```text
Usage: render_likelihood_validation.py [-h] --run-dir RUN_DIR [--results-dir RESULTS_DIR] [--figure-dir FIGURE_DIR]
```

#### `plot_finished_chain.py`

Source: [`scripts/likelihood/postprocess/plot_finished_chain.py`](../../scripts/likelihood/postprocess/plot_finished_chain.py)

Plots posterior samples from one or more completed-chain metadata files, optionally selecting parameters and truth markers.

```text
Usage: plot_finished_chain.py [-h] [--output OUTPUT] [--params PARAMS] [--all-params] [--bins BINS] [--smooth SMOOTH] [--dpi DPI] [--label LABEL] [--no-truths] [--dry-run] metadata [metadata ...]
```

## HPC / Slurm

Cluster launchers require the appropriate Slurm commands and site environment. The WL/GCsp job templates use the configured JURECA/PUNCH environment.

### `submit_wl_gcsp_nautilus.sh`

Source: [`scripts/hpc/slurm/submit_wl_gcsp_nautilus.sh`](../../scripts/hpc/slurm/submit_wl_gcsp_nautilus.sh)

Submits independent WL, GCsp, and joint sampling jobs followed by dependent postprocessing. Its `--help` lists supported environment variables; a typical invocation is:

```text
Usage: RUN_DIR=<dir> [OPTIONS] scripts/hpc/slurm/submit_wl_gcsp_nautilus.sh
```

### `check_wl_gcsp_status.sh`

Source: [`scripts/hpc/slurm/check_wl_gcsp_status.sh`](../../scripts/hpc/slurm/check_wl_gcsp_status.sh)

Reads Slurm accounting and completed run statistics to report job status and chain-versus-Fisher results; it does not submit or cancel jobs.

```text
Usage: scripts/hpc/slurm/check_wl_gcsp_status.sh <RUN_DIR> <WL_JOBID> <GCSP_JOBID> <JOINT_JOBID> <POST_JOBID>
       scripts/hpc/slurm/check_wl_gcsp_status.sh --watch [SECONDS] <RUN_DIR> <WL_JOBID> <GCSP_JOBID> <JOINT_JOBID> <POST_JOBID>
```

### `wl_gcsp_nautilus_case.sbatch`

Source: [`scripts/hpc/slurm/wl_gcsp_nautilus_case.sbatch`](../../scripts/hpc/slurm/wl_gcsp_nautilus_case.sbatch)

Slurm batch template for one WL/GCsp Nautilus case; configured by the submission wrapper and intended for `sbatch`, not direct Python execution.

```text
Invocation: sbatch --export=ALL,... scripts/hpc/slurm/wl_gcsp_nautilus_case.sbatch
```

### `wl_gcsp_nautilus_postprocess.sbatch`

Source: [`scripts/hpc/slurm/wl_gcsp_nautilus_postprocess.sbatch`](../../scripts/hpc/slurm/wl_gcsp_nautilus_postprocess.sbatch)

Slurm batch template that postprocesses the dependent WL/GCsp sampling outputs.

```text
Invocation: sbatch --export=ALL,... scripts/hpc/slurm/wl_gcsp_nautilus_postprocess.sbatch
```

### `nautilus_sampler.sbatch`

Source: [`scripts/hpc/slurm/nautilus_sampler.sbatch`](../../scripts/hpc/slurm/nautilus_sampler.sbatch)

Generic Slurm launcher for the YAML-driven `sampler_scripts/run_sampler.py`; set its required `REPO_ROOT`, `CONFIG_FILE`, and `OUTPUT_DIR` environment variables.

```text
Invocation: REPO_ROOT=<repo> CONFIG_FILE=<yaml> OUTPUT_DIR=<dir> sbatch scripts/hpc/slurm/nautilus_sampler.sbatch
```

## Profiling

### `profile_photo_obs.py`

Source: [`scripts/profiling/photo_obs/profile_photo_obs.py`](../../scripts/profiling/photo_obs/profile_photo_obs.py)

Profiles photometric angular spectra and Fisher calculations across selected backends, accuracies, and derivative settings.

```text
Usage: profile_photo_obs.py [-h] [--observables OBSERVABLES [OBSERVABLES ...]] [--code {symbolic,camb,class} [{symbolic,camb,class} ...]] [--repeats REPEATS] [--accuracy ACCURACY [ACCURACY ...]] [--derivatives DERIVATIVES] --output OUTPUT [--csv CSV] [--keep-prof] [--tag TAG] [--freepars-minimal] [--no-derivs] [--specs-dir SPECS_DIR] [--fast-eff]
```

## Utilities

### `add_scalar_prior_to_fisher.py`

Source: [`scripts/utilities/fisher/add_scalar_prior_to_fisher.py`](../../scripts/utilities/fisher/add_scalar_prior_to_fisher.py)

Adds a named one-parameter Gaussian prior to a Fisher matrix and can compare the result with another forecast.

```text
Usage: add_scalar_prior_to_fisher.py [-h] [--param PARAM] [--sigma SIGMA] [--prior-name PRIOR_NAME] [--outdir OUTDIR] [--compare-to COMPARE_TO] [--params PARAMS] fisher
```

## Release maintenance

### `prepare_changelog.py`

Source: [`scripts/maintenance/release/prepare_changelog.py`](../../scripts/maintenance/release/prepare_changelog.py)

Moves the current package version from the Unreleased section into a dated `CHANGELOG.md` heading. It edits the repository changelog in place and takes no arguments.

```text
Invocation: uv run python scripts/maintenance/release/prepare_changelog.py
```

### `release_notes.py`

Source: [`scripts/maintenance/release/release_notes.py`](../../scripts/maintenance/release/release_notes.py)

Generates release notes for the tag supplied through the `TAG` environment variable; this helper is also used by the release GitHub Actions workflow.

```text
Invocation: TAG=<git-tag> uv run python scripts/maintenance/release/release_notes.py
```

### `release.sh`

Source: [`scripts/maintenance/release/release.sh`](../../scripts/maintenance/release/release.sh)

Interactive release helper that prepares the changelog, commits, creates a tag, and pushes commits and tags. Review its contents and repository state before running; it has no help mode.

```text
Invocation: bash scripts/maintenance/release/release.sh
```
