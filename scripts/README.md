# Archived Benchmarks and Maintained Workflows

Photometric benchmark and profiling scripts are archived under `archive/benchmarks/`; profiling-only runners are in `archive/profiling/`. Maintained validation, Planck, likelihood, HPC, and release workflows live in the categorized subdirectories below.

The historical entry point is `archive/benchmarks/photometric_benchmark.py`, which can:
- Run a regular Fisher benchmark for selected observables and backend
- Compare SLOW vs FAST code paths end‑to‑end (SLOW = optimizations off, FAST = optimizations on) with `--fast-benchmark`
- Optionally benchmark the lensing‑efficiency integrators with `--efficiency` (see also `archive/benchmarks/bench_efficiency.py`)

Run all commands from the repository root: results go to `scripts/benchmark_results/`
relative to the current directory.

Below you’ll find setup instructions, usage examples, and details on outputs.

### Sequential five-parameter LCDM Nautilus campaign (36 CPUs)

```bash
bash scripts/likelihood/nautilus/run_euclid_mp_nautilus.sh --dry-run
bash scripts/likelihood/nautilus/run_euclid_mp_nautilus.sh
```

Samples GCsp and photometric GCph+WL (including their cross-correlation), each
with CLASS, CAMB and SYREN-NEW, **one posterior at a time**. The five sampled
parameters are `Omegam, Omegab, h, ns, sigma8`; all nuisance parameters, mnu,
Neff, w0=-1 and wa=0 are fixed. Each backend generates its own self-consistent
noiseless fiducial data. These are conditional five-parameter posteriors.
No Fisher matrices or Fisher-derived priors are computed.

Resource defaults follow the JURECA notes in
[`cjfp-data/jobs_punch/nautilus_jureca_findings.md`](../../cjfp-data/jobs_punch/nautilus_jureca_findings.md):

| Backend | Likelihood workers | OMP threads per worker | Batch size |
|---|---:|---:|---:|
| CLASS / CAMB | 18 | 2 | 36 |
| SYREN-NEW | 36 | 1 | 72 |

BLAS threads stay at one and inherited OpenMP affinity variables are removed
to avoid pinning every spawned worker to the same core. Sampling defaults are
`n_live=1000`, `n_eff=3000`, `n_networks=8`, and `n_batch=2*workers`, with
`discard_exploration=True`. Override with `--n-live`, `--n-eff`, `--n-networks`,
`--n-batch`, `--workers` or `--omp-threads`. The launcher checks a 36-CPU budget
and any Slurm CPU allocation.

Identical uniform cosmology priors are used for all six runs:
Omega_m [0.28,0.36], Omega_b [0.04,0.06], h [0.60,0.74], n_s [0.90,1.02],
sigma8 [0.75,0.87]. Fiducials and solver profiles are the MP-validation settings
documented below, with the model changed to LCDM.

Outputs default to a new timestamped `../cjfp-data/chains/euclid_mp_lcdm_*`
directory (`--outdir` or `CFP_DATA_REPO` overrides this). Each run has its own
`run.log`, requested configuration, input hashes, fiducial likelihood check,
HDF5 checkpoint, weighted posterior text file and metadata. `campaign.json`
records exit codes and per-backend resources. Checkpoints are updated while
sampling; archive/upload only after the campaign is stopped or complete.

To resume an interrupted campaign, keep the original code and inputs and use:

```bash
bash scripts/likelihood/nautilus/run_euclid_mp_nautilus.sh \
  --outdir /path/to/original/campaign --resume
```

Repeat any original sampling overrides when resuming. Changed inputs, priors,
sampling settings or Git revision are rejected before reusing a checkpoint.
`--check-only` initializes and checks the likelihoods without sampling.

### Quick MP-validation Pessimistic campaign (36 CPUs)

```bash
bash scripts/validation/run_euclid_mp_fishers.sh --dry-run
bash scripts/validation/run_euclid_mp_fishers.sh
```

Runs six independent Fishers: GCsp and GCph+WL (including their cross-correlation,
i.e. photo 3×2pt), each with CLASS, CAMB and SYREN-NEW. Defaults to six processes,
six OpenMP threads each; BLAS threads are capped at one. Uses the existing
`common_specs_w0waCDM.json`, Paper-Pessimistic surveys, CLASS `mpvalidation_hp`,
CAMB `mpvalidation_p3`, and symbolic `syren_new_photo` profiles. The seven free
cosmological parameters are Omegam, Omegab, h, ns, sigma8, w0 and wa; mnu/Neff stay
fixed, as in the MP cases. No new parameter or emulator validation is implied.

Outputs and per-worker `run.log` files go into a new timestamped directory under
`../cjfp-data/validation/fisher/` (override with `--outdir` or `CFP_DATA_REPO`).
`campaign.json` records the code commit, thread budget and exit statuses.
Use `--jobs 3 --omp-threads 12` to run three Fishers concurrently instead.
Run inside `screen` or a Slurm allocation with at least 36 CPUs; outputs are not
uploaded automatically. Keep the checkout unchanged while the campaign runs.

---

## 1) Prerequisites

- A working installation of CosmicFishPie and its dependencies (ensure you can run the test suite).
- An activated environment, e.g.
  
  ```bash
  mamba activate cosmicfishpie
  ```

- Optional (faster runs): set thread caps for BLAS/OpenMP as needed, e.g.
  
  ```bash
  export OMP_NUM_THREADS=8
  ```

---

## 2) The Benchmark Script: `archive/benchmarks/photometric_benchmark.py`

This script focuses on photometric Fisher matrices. It supports these workflows:

- Regular Fisher benchmark (single run, with the FAST paths on by default)
- Flags benchmark (`--fast-benchmark`, SLOW vs FAST comparison)
- Lensing‑efficiency benchmark (`--efficiency`)

### Key concepts

- “FAST” refers to enabling internal optimization flags in `cosmicfishpie.LSSsurvey.photo_obs`:
  - `COSMICFISH_FAST_EFF` (O(N) lensing‑efficiency integrator)
  - `COSMICFISH_FAST_P` (vectorized Limber power spectrum)
  - `COSMICFISH_FAST_KERNEL` (vectorized WL kernel)
- “SLOW” disables all the above flags to act as a baseline.

These flags are managed by the script; you do not need to set environment variables manually.

---

## 3) Common Options

- `--accuracy N`  Integer sampling multiplier (affects z/ell grids). Higher means slower, more precise. Default: `1`.
- `--code {symbolic,class,camb}`  Cosmology backend to use. Default: `symbolic` (fastest).
- `--codes <list|all>`  Comma‑separated backends (or `all`); overrides `--code` and compares them.
- `--observables <list>`  Comma‑separated list among `GCph` and `WL`. Examples:
  - `--observables GCph,WL` (default combined photometric)
  - `--observables GCph`
  - `--observables WL`

All runs write results to `scripts/benchmark_results/` by default. Output file names start
with `Euclid-Photo-ISTF-Pess-Benchmark_<timestamp>_<hash>_`, where the hash identifies the
command‑line arguments.

---

## 4) Regular Fisher Benchmark (single run)

Runs a single Fisher for the selected observables and backend, then exports a summary comparison JSON (mainly useful when running multiple combos).

```bash
python scripts/archive/benchmarks/photometric_benchmark.py \
  --code symbolic \
  --observables GCph,WL \
  --accuracy 1
```

Outputs:
- Fisher matrix text/CSV/paramnames/specs under `scripts/benchmark_results/` with a descriptive filename that includes the backend and observables.
- Timings (`..._timings.json`) and summary (`..._fisher_comparison.json`) JSON files in `scripts/benchmark_results/`.

---

## 5) SLOW vs FAST Flags Benchmark (end‑to‑end)

Runs two full Fisher computations back‑to‑back for the selected observables/backend:
1) SLOW: all fast flags off
2) FAST: fast flags on (by default)

```bash
python scripts/archive/benchmarks/photometric_benchmark.py \
  --fast-benchmark \
  --code symbolic \
  --observables GCph,WL \
  --accuracy 1
```

Optional granular FAST toggles:
- `--fast-eff {auto,on,off}`  (default auto=on)
- `--fast-p {auto,on,off}`    (default auto=on)
- `--fast-kernel {auto,on,off}` (default auto=on)

This is useful if you want to isolate the effect of one fast path.

Outputs:
- Two Fisher matrices, clearly tagged in filenames:
  - `..._SLOW_...` and `..._FAST_...` Fisher matrix files (the tag is part of the output root)
- JSON summary: `..._flags_fisher_benchmark.json` in `scripts/benchmark_results/` with:
  - `accuracy`, `code`, `observables`
  - `runs`: SLOW/FAST timestamps, env flags actually used, exported matrix paths
  - `metrics`: per‑run (trace, Frobenius, robust logdet) and comparisons (entrywise relative max, abs diff near zero, Frobenius relative diff, diagonal ratio stats)
  - `speedup`: SLOW/FAST wall‑clock ratio
  - `options`: a snapshot of key configuration

Interpretation tips:
- `rel_max` near ~1e‑12–1e‑14 indicates numerical agreement (differences only at machine precision).
- `speedup > 1` means FAST is faster; the gain typically grows with `--accuracy`.

---

## 6) Single FAST run (for profiling)

There is no separate FAST‑only flag: a regular run (section 4) already has all FAST paths
enabled unless you turn one off with `--fast-eff off`, `--fast-p off` or
`--fast-kernel off` (or all at once with `--fast off`). Use it with profilers like
`cProfile`, `line_profiler`, or `memory_profiler`.

Profiling examples:

- cProfile:
  ```bash
  python -m cProfile -o scripts/benchmark_results/fast.prof \
    scripts/archive/benchmarks/photometric_benchmark.py --code symbolic --observables GCph,WL
  ```
- line_profiler:
  ```bash
  kernprof -l -o scripts/benchmark_results/fast.lprof \
    scripts/archive/benchmarks/photometric_benchmark.py --code symbolic --observables GCph,WL
  python -m line_profiler scripts/benchmark_results/fast.lprof
  ```

---

## 7) Lensing‑Efficiency Micro‑Benchmark (optional)

`archive/benchmarks/bench_efficiency.py` compares three internal implementations of the WL lensing‑efficiency integral:
- `memo_integral_efficiency` (legacy O(N²), corrected reference)
- `faster_integral_efficiency` (vectorized O(N²))
- `much_faster_integral_efficiency` (O(N) cumulative trapezoids)

Run:
```bash
python scripts/archive/benchmarks/bench_efficiency.py
```

It prints construction/evaluation timings and agreement metrics (ratios), and uses a `ComputeCls` compatible configuration.

---

## 8) Reproducibility & Tips

- The JSON outputs include a snapshot of key options and the effective env flags used.
- For apples‑to‑apples comparisons, keep `--accuracy`, `--code`, `--observables`, and survey specs fixed.
- If you change backends (`class`/`camb`), ensure those backends are properly installed and configured; see `cosmicfishpie/configs/default_boltzmann_yaml_files/`.
- For dense grids (`--accuracy 2` or higher), FAST paths yield larger speedups.

---

## 9) Troubleshooting

- “Module not found” (e.g., pandas): ensure your environment has all dependencies installed and activated.
- Paths look duplicated (double slashes): harmless; filenames remain valid. If desired, adjust `results_dir` without a trailing slash.
- If you toggle flags manually in a REPL, reload the modules so flags take effect:
  
  ```python
  import os, importlib
  os.environ.update(COSMICFISH_FAST_EFF="1", COSMICFISH_FAST_P="1", COSMICFISH_FAST_KERNEL="1")
  import cosmicfishpie.LSSsurvey.photo_obs as photo_obs
  importlib.reload(photo_obs)
  ```

---

## 10) Examples Cheat‑Sheet

- Regular (symbolic, combined):
  ```bash
  python scripts/archive/benchmarks/photometric_benchmark.py --code symbolic --observables GCph,WL
  ```
- Compare SLOW vs FAST (class, WL only):
  ```bash
  python scripts/archive/benchmarks/photometric_benchmark.py --fast-benchmark --code class --observables WL
  ```
- Single run for profiling (camb, GCph only):
  ```bash
  python scripts/archive/benchmarks/photometric_benchmark.py --code camb --observables GCph
  ```
- SLOW vs FAST, with only the efficiency fast path enabled in the FAST run:
  ```bash
  python scripts/archive/benchmarks/photometric_benchmark.py --fast-benchmark --fast-p off --fast-kernel off
  ```

---

## 11) Backend comparisons and reports

Use these scripts to run two backends, compare, plot, and render shareable reports.

### Maintained paper-validation workflow

The maintained validation entry point covers the Casas et al. w0waCDM cases and the
paper-faithful neutrino cases. Cases run sequentially by default; `--jobs` bounds
parallel cases, and `--omp-threads` sets the OpenMP thread count for each case:

```bash
bash scripts/validation/run_selected_validations.sh --all --jobs 2 --omp-threads 4
```

This uses up to eight OpenMP threads total. On a Slurm allocation, reserve at least
`--jobs` × `--omp-threads` CPUs. The runner checks `SLURM_CPUS_PER_TASK` or
`SLURM_CPUS_ON_NODE` when available and rejects requests larger than the allocation.
Keep the source checkout and `cjfp-data` as sibling directories. The parallel
convenience wrapper writes Fisher outputs, batch logs, and the dashboard directly to
`../cjfp-data/validation/fisher/`:

```bash
uv sync
bash scripts/validation/run_parallel_validations.sh
```

It defaults to two concurrent cases with four threads each. Set `CFP_DATA_REPO` if the
data checkout is elsewhere, or `CFP_VALIDATION_RESULTS_DIR` to choose a different
output path. Override concurrency with `CFP_VALIDATION_JOBS` and `CFP_OMP_THREADS`.
This Fisher workflow does not read or require the archived
Nautilus chains. After reviewing results, use DVC from the data checkout to register
and upload them if they should be retained remotely.

Completed cases are reused when their numerical inputs and backend provenance still
match; pass `--force` to recompute them. The command refreshes the clean dashboard at
`<results-directory>/dashboard/index.html`. Superseded one-off and batch runners are
retained for provenance under `scripts/archive/` and are not part of the maintained
workflow.

For a fresh campaign that prioritizes previously failing comparisons:

```bash
bash scripts/validation/run_parallel_validations.sh --failed-first
```

This reads saved numerical FAIL gates directly from the result files, runs those cases
first with the usual concurrency, **waits for that whole phase**, then runs all remaining
selected cases. It implies `--force`: even previously passing cases are recomputed.
Use `--cases LIST` to restrict the campaign. With no saved FAIL results, all selected
cases run in the remaining phase. The intentional P_mm failure control is included
when selected. A failed case does not prevent the second phase from running.

The campaign records its starting revision in `<batch-directory>/git_commit.txt` and
stops launching cases if HEAD changes. Keep the checkout unchanged during execution;
commit the intended numerical changes before launching a single-commit campaign.
Normal reuse is based on changes to **relevant code/inputs**, not every new Git commit:
a docs-only commit does not require fresh Fishers. `--failed-first` bypasses reuse.
The dashboard still shows the latest available completed comparisons, so if a new
attempt exits before producing results, it can continue to show older results.

To render the dashboard from whatever completed results are currently available:

```bash
uv run python scripts/validation/render_validation_dashboard.py
```

Preview it locally with:

```bash
uv run python scripts/validation/render_validation_dashboard.py --serve
```

Open `http://127.0.0.1:8000/` and press Ctrl-C once to stop the preview server.

The generated `scripts/benchmark_results/` directory is gitignored. To publish the
dashboard and landing page to the `gh-pages` branch after rendering, run:

```bash
bash scripts/validation/publish_validation_dashboard.sh
```

The publisher uses a temporary worktree, commits the generated site on `gh-pages`, and
pushes that branch. It does not add benchmark output to the feature PR. If the site must
be updated only after the feature branch is merged, run the publisher from the updated
main branch instead.

The validation runner isolates each case in its own process group. Press Ctrl-C once to
terminate the active backend comparison and stop the batch; completed cases remain
reusable on the next run.

### One-off CLI workflow

1) Run two backends and compare (writes to scripts/benchmark_results/compare_<mode>_<timestamp>/):

```bash
uv run python scripts/validation/run_fisher_compare_backends.py \
  --mode photo \
  --code-a class --code-b camb \
  --compare --plot \
  --fom-params Omegam,sigma8
```

2) Render per-run reports, index, and a single-file HTML report:

```bash
uv run python scripts/validation/render_compare_reports.py \
  scripts/benchmark_results/compare_photo_YYYYMMDD_HHMMSS \
  --single-file scripts/benchmark_results/compare_reports_single.html
```

For multiple runs, use a glob:

```bash
uv run python scripts/validation/render_compare_reports.py \
  --glob "scripts/benchmark_results/compare_*" \
  --single-file scripts/benchmark_results/compare_reports_single.html
```

Optional bundle and zip:

```bash
uv run python scripts/validation/render_compare_reports.py \
  --glob "scripts/benchmark_results/compare_*" \
  --bundle-dir scripts/benchmark_results/compare_reports_bundle \
  --zip
```

### Helper script

Create a local config file:

```bash
cp scripts/validation/configs/compare_run_config.env.example \
  scripts/validation/configs/compare_run_config.env
```

Edit `scripts/validation/configs/compare_run_config.env`, then run:

```bash
bash scripts/validation/compare_backends_report.sh --config scripts/validation/configs/compare_run_config.env
```

Notes:
- `scripts/validation/configs/compare_run_config.env` is gitignored; commit only the example file.
- Leave YAML_A/YAML_B empty to use defaults inferred from backend.
- COMMON_SPECS_JSON can point to a *_FM_specs.json or a small JSON with fiducialpars/freepars/options.
- Values in the env file override anything set in the JSON.
- If COMMON_SPECS_JSON is a relative path, it is resolved relative to the config file.
- Use REPORT_SINGLE=true/false to control the single-file output; REPORT_SINGLE_FILE is a path override.
- The single-file report inlines plots and compare JSON for easy sharing.
- The helper script always generates per-run REPORT.html and an index.html alongside the single-file output.
- Outputs are stored under a run folder that includes the config hash; the single-file report
  and index live inside that folder. A `run_config.env` file is written for provenance.
- REPORT_SINGLE_FILE/INDEX_DIR/BUNDLE_DIR must resolve inside OUTDIR.
- Relative REPORT_SINGLE_FILE/INDEX_DIR/BUNDLE_DIR values are resolved under OUTDIR.
- The helper script defaults to deterministic names using a config hash. Set USE_TIMESTAMP=true
  to append a timestamp and keep multiple runs with identical settings.
- For pipelines, create multiple config files and run in a loop, e.g.:
  `for cfg in scripts/compare_configs/*.env; do bash scripts/validation/compare_backends_report.sh --config "$cfg"; done`

Defaults:
- class: `cosmicfishpie/configs/default_boltzmann_yaml_files/class/default.yaml`
- camb: `cosmicfishpie/configs/default_boltzmann_yaml_files/camb/default.yaml`
- symbolic: `cosmicfishpie/configs/default_boltzmann_yaml_files/symbolic/default.yaml`
- REPORT_SINGLE=true writes `OUTDIR/report_single.html`; override with REPORT_SINGLE_FILE.

---

## 12) CMB Fisher smoke + benchmarks

CosmicFishPie includes a basic CMB Fisher path for `CMB_T`, `CMB_E`, `CMB_B`.

Notes/limitations:
- The current CMB implementation uses a single `lmin_CMB`/`lmax_CMB` for all spectra.
- The specs in Table form (separate TT/TE/EE ranges, and lensing spectra like `phi-phi`) are not yet implemented.

### CMB survey spec YAMLs

These live in `cosmicfishpie/configs/default_survey_specifications/`:
- `Planck.yaml`
- `Simons-Observatory-PlanckLowEll.yaml`
- `CMB-Stage4-PlanckLowEll.yaml`

The SO/S4 YAMLs use a conservative common `lmax_CMB=3000`.

### Single run (smoke)

The CMB scripts live in `scripts/archive/cmb/`. Run them from the repository root:

```bash
uv run python scripts/archive/cmb/run_cmb_fisher_smoke.py \
  --code camb \
  --spec-yaml cosmicfishpie/configs/default_survey_specifications/Planck.yaml \
  --observables CMB_T,CMB_E \
  --lmax 120 \
  --outdir tmp/cmb_planck_smoke \
  --write-enabled-yaml
```

### Run the three presets

```bash
uv run python scripts/archive/cmb/run_cmb_benchmarks.py \
  --which planck,so,s4 --observables CMB_T,CMB_E --outdir tmp/cmb_bench
```

Add `--lmax 120` for a quick test. See [CMB benchmarks](../docs/source/cmb_benchmarks.md).

---

## 13) Planck best-fit CMB Fisher (h/theta primary basis)

To run a Planck-like CMB Fisher at the Planck chain best-fit point:

- `--parameterization h`: `ombh2`, `omch2`, `h`, `tau`, `logAs`, `ns`
- `--parameterization theta`: `ombh2`, `omch2`, `theta`, `tau`, `logAs`, `ns`
- default CMB noise model in this runner: `knox`

Example (`h`-based):

```bash
uv run python scripts/planck/run_planck_bestfit_fisher.py \
  --outdir tmp/planck_bestfit_h
```

Example (`theta`-based, for direct Planck covmat direction checks):

```bash
uv run python scripts/planck/run_planck_bestfit_fisher.py \
  --parameterization theta \
  --outdir tmp/planck_bestfit_theta
```

To use the paper-consistent Knox noise model with low-ell EE noise boost:

```bash
uv run python scripts/planck/run_planck_bestfit_fisher.py \
  --parameterization theta \
  --cmb-noise-model knox \
  --ee-lowell-noise-boost 8 \
  --ee-lowell-max-ell 29 \
  --outdir tmp/planck_bestfit_theta_knox
```

What this script does:
- reads the best-fit sample from
  `Planck-Results/.../dist/base_plikHM_TTTEEE_lowl_lowE.likestats`
- uses Planck-like ell coverage (`lmin=2`, `lmax=2508` inclusive)
- writes a Planck-like CAMB YAML into the run folder for provenance
- runs a CMB Fisher for `CMB_T,CMB_E`

Then compare the resulting Fisher constraints against Planck published
`.margestats`:

```bash
uv run python scripts/planck/compare_planck_published.py \
  tmp/planck_bestfit_h/<your_fisher_file>_CMB_TCMB_E_FM.txt
```

This comparison writes `compare_planck_published.json` next to the Fisher file
and prints a compact sigma table in the terminal.

To generate all Fisher products used by
`notebooks/7-Planck-covmat-noise-diagnostics.ipynb` in one shot:

```bash
uv run bash scripts/planck/run_planck_diagnostics_suite.sh
```

Optional positional argument: custom Planck chain directory.

```bash
uv run bash scripts/planck/run_planck_diagnostics_suite.sh /path/to/plikHM_TTTEEE_lowl_lowE
```

For correlation-direction validation, overlay your theta-based Fisher against
the published Planck Gaussian approximation from `.covmat`:

```bash
uv run python scripts/planck/plot_planck_covmat_vs_fisher.py \
  tmp/planck_bestfit_theta/<your_theta_fisher>_CMB_TCMB_E_FM.txt
```

This writes:
- a converted Gaussian Fisher from Planck covmat,
- a triangle plot with both contours,
- a JSON report with correlation matrices and their differences.

Compatibility note:
- Deprecated flags `--ee-lowell-inflation` and `--ee-lowell-lmax` are still accepted,
  but should be replaced by `--ee-lowell-noise-boost` and `--ee-lowell-max-ell`.

---

Happy benchmarking!
