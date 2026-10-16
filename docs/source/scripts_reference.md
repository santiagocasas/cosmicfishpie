# Scripts reference

The command-line tools under `scripts/`, grouped by task. Run everything from the
repository root. Python scripts run with `uv run python <script>` and every one that
parses arguments prints its full option list with `--help`, so this page does not repeat
option lists. Shell and Slurm launchers show their invocation form instead.

Most tools here are for developers and cluster users. If you just want to run a forecast,
you do not need any of them: see {doc}`first_forecast`.

## Which script do I want?

| I want to... | Use | Section |
| --- | --- | --- |
| Compare CAMB and CLASS forecasts for the configured validation cases | `scripts/validation/run_selected_validations.sh` | [Backend validation](#backend-validation) |
| Compare two Fisher matrices on disk | `scripts/validation/compare_fishers_in_dir.py` | [Backend validation](#backend-validation) |
| Check a CMB forecast against a Planck chain | `scripts/planck/run_planck_bestfit_fisher.py` | [Planck diagnostics](#planck-diagnostics) |
| Compare Fisher and Nautilus posteriors for WL and GCsp | `scripts/likelihood/nautilus/wl_gcsp_fisher_nautilus_demo.py` | [Likelihood tools](#likelihood-tools) |
| Run long Nautilus chains on a Slurm cluster | `scripts/hpc/slurm/submit_wl_gcsp_nautilus.sh` | [Slurm launchers](#slurm-launchers) |
| Time the photometric pipeline, or compare backends and fast paths | `scripts/archive/benchmarks/photometric_benchmark.py` | [Benchmarks and profiling](#benchmarks-and-profiling) |
| Run a CMB Fisher for Planck, Simons Observatory or CMB-S4 specs | `scripts/archive/cmb/run_cmb_fisher_smoke.py` | [Benchmarks and profiling](#benchmarks-and-profiling) |
| Add a Gaussian prior to a saved Fisher matrix | `scripts/utilities/fisher/add_scalar_prior_to_fisher.py` | [Fisher utilities](#fisher-utilities) |
| Prepare a release | `scripts/maintenance/release/` | [Release maintenance](#release-maintenance) |

## Backend validation

Run CAMB-versus-CLASS comparisons and build the report dashboard; see
{doc}`backend_validation` for the workflow. Start with `run_selected_validations.sh`. It
calls most of the others.

| Script | What it does |
| --- | --- |
| `scripts/validation/run_selected_validations.sh` | Runs selected cases (`--cases LIST`) or all of them (`--all`), then refreshes the dashboard. `--help` prints the table of configured case IDs. |
| `scripts/validation/compare_backends_report.sh` | Runs both backends, the comparison, plots and report for one case (`--config path`). |
| `scripts/validation/run_fisher_compare_backends.py` | Computes two Fisher forecasts with chosen backends (`--code-a`, `--code-b`) and optionally compares and plots them (`--compare`, `--plot`). |
| `scripts/validation/compare_fishers_in_dir.py` | Compares the Fisher matrices in a result directory; positional `dir`, optional `--ref`. |
| `scripts/validation/plot_compare_fishers.py` | Plots the comparison stored in a JSON file (positional `json`). |
| `scripts/validation/render_compare_reports.py` | Renders comparison folders as Markdown or HTML reports. |
| `scripts/validation/render_validation_dashboard.py` | Builds the dashboard HTML from validation configs and results; `--serve` previews it locally. |
| `scripts/validation/publish_validation_dashboard.sh` | Publishes the dashboard to the `dashboard/` path of `gh-pages`. It leaves the `docs/` path, where `.github/workflows/docs.yml` publishes this site, untouched. It changes the publication checkout, so run it only when you intend to publish. |
| `scripts/validation/setup/install_camb_fork_for_validation.sh` | Installs the pinned CAMB fork for validation without changing the package's CAMB pin. |

## Planck diagnostics

Compare CMB Fisher forecasts with Planck chains and published covariances. The Planck
directory must hold the chain's `.likestats`, `.margestats`, `.covmat`, `.paramnames` and
`.minimum.inputparams` products.

| Script | What it does |
| --- | --- |
| `scripts/planck/run_planck_bestfit_fisher.py` | Builds a CMB Fisher at a Planck best fit, with options for parameterisation, observables, noise and solver settings. |
| `scripts/planck/run_planck_diagnostics_suite.sh` | Runs the standard set of best-fit variants used by the covariance/noise diagnostic notebook: `bash scripts/planck/run_planck_diagnostics_suite.sh [PLANCK_DIR]`. `PLANCK_DIR` defaults to a path under `Planck-Results/`. |
| `scripts/planck/compare_planck_published.py` | Compares a Fisher forecast (positional `fisher`) with published Planck constraints. |
| `scripts/planck/plot_planck_covmat_vs_fisher.py` | Plots Planck covariance constraints against a Fisher forecast. |

## Likelihood tools

For the Nautilus workflow itself, see {doc}`likelihood_sampling`.

| Script | What it does |
| --- | --- |
| `scripts/likelihood/nautilus/wl_gcsp_fisher_nautilus_demo.py` | Quick WL, GCsp and joint Fisher-versus-Nautilus demo with comparison plots. Use `--skip-nautilus` to skip sampling. |
| `scripts/likelihood/nautilus/run_wl_gcsp_nautilus_case.py` | Runs one case (`--case {wl,gcsp,joint}`) into `--run-dir`; `--n-live`, `--n-eff` and `--workers` are required; `--resume` continues a run. |
| `scripts/likelihood/nautilus/postprocess_wl_gcsp_nautilus.py` | Summarises a finished run (`--run-dir`) with chain-versus-Fisher statistics and plots. |
| `scripts/likelihood/postprocess/render_likelihood_validation.py` | Collects copied HPC run artefacts into the statistics and figures of {doc}`likelihood_validation`. |
| `scripts/likelihood/postprocess/plot_finished_chain.py` | Plots posterior samples from one or more chain metadata files. |
| `scripts/likelihood/diagnostics/run_syren_new_likelihoods.py` | Runs the SYREN likelihood checks (`--probe {gcsp,photo,both}`). |
| `scripts/likelihood/diagnostics/gcsp_likelihood_grid.py` | Evaluates and saves a two-parameter GCsp likelihood grid. |
| `scripts/likelihood/diagnostics/gcsp_likelihood_hessian.py` | Checks the GCsp likelihood curvature for several finite-difference step fractions. |
| `scripts/likelihood/diagnostics/wl_likelihood_grid.py` | Two-parameter weak-lensing likelihood grid with parallel workers. |
| `scripts/likelihood/diagnostics/verify_gcsp_covariance_sympy.py` | Prints SymPy checks of the GCsp covariance. It has no options and no `--help`. |

## Slurm launchers

For cluster runs of the WL and GCsp Nautilus cases. They need Slurm and the site
environment (JURECA/PUNCH for the WL/GCsp templates). Start with `submit_wl_gcsp_nautilus.sh`,
which submits the two WL/GCsp `.sbatch` templates for you.

| Script | Invocation |
| --- | --- |
| `scripts/hpc/slurm/submit_wl_gcsp_nautilus.sh` | `RUN_DIR=<dir> [OPTIONS] scripts/hpc/slurm/submit_wl_gcsp_nautilus.sh` submits the WL, GCsp and joint jobs plus a dependent postprocessing job. `--help` lists the environment variables. |
| `scripts/hpc/slurm/check_wl_gcsp_status.sh` | `scripts/hpc/slurm/check_wl_gcsp_status.sh [--watch [SECONDS]] <RUN_DIR> <WL_JOBID> <GCSP_JOBID> <JOINT_JOBID> <POST_JOBID>` reports job status and results; it never submits or cancels jobs. |
| `scripts/hpc/slurm/wl_gcsp_nautilus_case.sbatch` | Template for one case: `sbatch --export=ALL,... <file>`. |
| `scripts/hpc/slurm/wl_gcsp_nautilus_postprocess.sbatch` | Template for the dependent postprocessing job. |
| `scripts/hpc/slurm/nautilus_sampler.sbatch` | Generic launcher for the YAML-driven `sampler_scripts/run_sampler.py`: `REPO_ROOT=<repo> CONFIG_FILE=<yaml> OUTPUT_DIR=<dir> sbatch scripts/hpc/slurm/nautilus_sampler.sbatch`. |

## Benchmarks and profiling

| Script | What it does |
| --- | --- |
| `scripts/archive/benchmarks/photometric_benchmark.py` | Times photometric (`GCph`, `WL`) Fisher forecasts and the lensing-efficiency integrators; see below. |
| `scripts/profiling/photo_obs/profile_photo_obs.py` | Profiles photometric angular spectra and Fisher calculations across backends, accuracies and derivative settings. `--output` is required. |
| `scripts/archive/cmb/run_cmb_fisher_smoke.py` | Runs one CMB Fisher forecast from a survey-spec YAML; see below. |
| `scripts/archive/cmb/run_cmb_benchmarks.py` | Runs the Planck, Simons Observatory and CMB-S4 presets in one go; see below. |

The photometric and CMB scripts live in `scripts/archive/`, where historical tools are
kept. They still run but are not part of the maintained
validation workflow. See also {doc}`cmb_benchmarks` for the CMB specifications they use.

### Photometric benchmark

```bash
uv run python scripts/archive/benchmarks/photometric_benchmark.py \
  --code symbolic --observables WL
```

This computes a Fisher forecast with the `symbolic` backend in about 10 seconds. It
prints one timing and error summary per run and writes the matrix, a timings file and
`*_fisher_comparison.json` to `scripts/benchmark_results/` (gitignored, created
relative to the current directory, so run it from the repository root). Useful options:

- `--codes camb,class,symbolic` (or `all`) runs several backends and compares them.
- `--observables GCph,WL` (the default) selects the probes.
- `--accuracy N` scales the grid sizes; higher is slower and more precise.
- `--efficiency` also benchmarks the lensing-efficiency integrators, and `--skip-fisher`
  skips the Fisher run.
- `--fast-benchmark` runs a full Fisher with the fast code paths on and off and writes a
  JSON summary; `--fast`, `--fast-eff`, `--fast-p` and `--fast-kernel` set the fast-path
  toggles.
- `--compare-ref PREFIX_OR_JSON` compares against an earlier run.

### CMB Fisher smoke test

Run it from the repository root:

```bash
uv run python scripts/archive/cmb/run_cmb_fisher_smoke.py \
  --code camb \
  --spec-yaml cosmicfishpie/configs/default_survey_specifications/Planck.yaml \
  --observables CMB_T,CMB_E --lmax 120 \
  --outdir tmp/cmb_planck_smoke --write-enabled-yaml
```

`--spec-yaml` can also be `Simons-Observatory-PlanckLowEll.yaml` or
`CMB-Stage4-PlanckLowEll.yaml` from the same folder. `--observables` takes any of
`CMB_T`, `CMB_E`, `CMB_B`. `--write-enabled-yaml` saves a copy of the solver YAML with
CMB outputs switched on in the output directory and runs with it; keep it unless your
own YAML already enables CMB outputs.

`run_cmb_benchmarks.py --which planck,so,s4` runs the three presets with CAMB; add
`--lmax 120` for a quick test. `--boltzmann-yaml` selects another solver YAML.

## Fisher utilities

| Script | What it does |
| --- | --- |
| `scripts/utilities/fisher/add_scalar_prior_to_fisher.py` | Adds a one-parameter Gaussian prior (`--param`, `--sigma`) to a Fisher matrix (positional `fisher`) and can compare the result with another forecast (`--compare-to`). |

## Release maintenance

| Script | Invocation |
| --- | --- |
| `scripts/maintenance/release/prepare_changelog.py` | `uv run python scripts/maintenance/release/prepare_changelog.py` moves the current version from the Unreleased section into a dated `CHANGELOG.md` heading. It edits the file in place and takes no arguments. |
| `scripts/maintenance/release/release_notes.py` | `TAG=<git-tag> uv run python scripts/maintenance/release/release_notes.py` generates release notes; the release GitHub Actions workflow also uses it. |
| `scripts/maintenance/release/release.sh` | `bash scripts/maintenance/release/release.sh` is an interactive helper that prepares the changelog, commits, tags, and pushes commits and tags. It has no help mode; read it and check the repository state before running. |
