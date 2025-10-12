# Photometric Benchmarks and Profiling

This folder contains helper scripts to benchmark and profile the photometric pipeline in CosmicFishPie.

The main entry point is `photometric_benchmark.py`, which can:
- Run a regular Fisher benchmark for selected observables and backend
- Compare SLOW vs FAST code paths end‑to‑end (SLOW = optimizations off, FAST = optimizations on)
- Run a single FAST run for profiling (no SLOW), to use with profilers
- Optionally benchmark the lensing‑efficiency integrators (`bench_efficiency.py`)

Below you’ll find setup instructions, usage examples, and details on outputs.

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

## 2) The Benchmark Script: `photometric_benchmark.py`

This script focuses on photometric Fisher matrices. It supports three workflows:

- Regular Fisher benchmark (single run)
- Flags benchmark (SLOW vs FAST comparison)
- FAST‑only run (one FAST run, ideal for profiling)

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
- `--observables <list>`  Comma‑separated list among `GCph` and `WL`. Examples:
  - `--observables GCph,WL` (default combined photometric)
  - `--observables GCph`
  - `--observables WL`

All runs write results to `scripts/benchmark_results/` by default.

---

## 4) Regular Fisher Benchmark (single run)

Runs a single Fisher for the selected observables and backend, then exports a summary comparison JSON (mainly useful when running multiple combos).

```bash
python scripts/photometric_benchmark.py \
  --code symbolic \
  --observables GCph,WL \
  --accuracy 1
```

Outputs:
- Fisher matrix text/CSV/paramnames/specs under `scripts/benchmark_results/` with a descriptive filename that includes the backend and observables.
- Summary JSON: `scripts/benchmark_results/Euclid-Photo-ISTF-Pess-Benchmark_fisher_comparison.json`.

---

## 5) SLOW vs FAST Flags Benchmark (end‑to‑end)

Runs two full Fisher computations back‑to‑back for the selected observables/backend:
1) SLOW: all fast flags off
2) FAST: fast flags on (by default)

```bash
python scripts/photometric_benchmark.py \
  --flags-benchmark \
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
  - `..._SLOW__<obs>_fishermatrix.txt`
  - `..._FAST__<obs>_fishermatrix.txt`
- JSON summary: `scripts/benchmark_results/Euclid-Photo-ISTF-Pess-Benchmark_flags_fisher_benchmark.json` with:
  - `accuracy`, `code`, `observables`
  - `runs`: SLOW/FAST timestamps, env flags actually used, exported matrix paths
  - `metrics`: per‑run (trace, Frobenius, robust logdet) and comparisons (entrywise relative max, abs diff near zero, Frobenius relative diff, diagonal ratio stats)
  - `speedup`: SLOW/FAST wall‑clock ratio
  - `options`: a snapshot of key configuration

Interpretation tips:
- `rel_max` near ~1e‑12–1e‑14 indicates numerical agreement (differences only at machine precision).
- `speedup > 1` means FAST is faster; the gain typically grows with `--accuracy`.

---

## 6) FAST‑Only Run (single run, ideal for profiling)

Runs one Fisher with fast flags enabled and stops. Use this with profilers like `cProfile`, `line_profiler`, or `memory_profiler`.

```bash
python scripts/photometric_benchmark.py \
  --fast-only \
  --code symbolic \
  --observables GCph,WL
```

Outputs:
- Fisher matrix with `FASTONLY` tag in filename.
- JSON summary: `scripts/benchmark_results/Euclid-Photo-ISTF-Pess-Benchmark_fast_only.json` with wall‑clock time, flags used, and matrix path.

Profiling examples:

- cProfile:
  ```bash
  python -m cProfile -o scripts/benchmark_results/fast.prof \
    scripts/photometric_benchmark.py --fast-only --code symbolic --observables GCph,WL
  ```
- line_profiler:
  ```bash
  kernprof -l -o scripts/benchmark_results/fast.lprof \
    scripts/photometric_benchmark.py --fast-only --code symbolic --observables GCph,WL
  python -m line_profiler scripts/benchmark_results/fast.lprof
  ```

---

## 7) Lensing‑Efficiency Micro‑Benchmark (optional)

`bench_efficiency.py` compares three internal implementations of the WL lensing‑efficiency integral:
- `memo_integral_efficiency` (legacy O(N²), corrected reference)
- `faster_integral_efficiency` (vectorized O(N²))
- `much_faster_integral_efficiency` (O(N) cumulative trapezoids)

Run:
```bash
python scripts/bench_efficiency.py
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
  python scripts/photometric_benchmark.py --code symbolic --observables GCph,WL
  ```
- Compare SLOW vs FAST (class, WL only):
  ```bash
  python scripts/photometric_benchmark.py --flags-benchmark --code class --observables WL
  ```
- FAST‑only for profiling (camb, GCph only):
  ```bash
  python scripts/photometric_benchmark.py --fast-only --code camb --observables GCph
  ```
- FAST but only enable efficiency fast path:
  ```bash
  python scripts/photometric_benchmark.py --flags-benchmark --fast-p off --fast-kernel off
  ```

---

Happy benchmarking!

