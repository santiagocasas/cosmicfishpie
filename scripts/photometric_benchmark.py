#!/usr/bin/env python
# coding: utf-8

# # CosmicFishPie

import argparse
import importlib
import json
import os
import subprocess
import time
from pathlib import Path
from time import perf_counter

import numpy as np

from cosmicfishpie.analysis import fisher_plot_analysis as fpa
from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey.photo_obs import (
    ComputeCls,
    faster_integral_efficiency,
    memo_integral_efficiency,
    much_faster_integral_efficiency,
)

envkey = "OMP_NUM_THREADS"
print("The value of {:s} is: ".format(envkey), os.environ.get(envkey))
os.environ[envkey] = str(8)
print("The value of {:s} is: ".format(envkey), os.environ.get(envkey))


# # Cosmological parameters

fiducial = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.815584,
    "mnu": 0.06,
    "Neff": 3.044,
}


# Define the observables you are interested in
# Default: benchmark the combined photometric case
observables = [["GCph", "WL"]]
# Default to the fastest cosmology backend for benchmarks in this script
codes = ["symbolic"]
# Input options for CosmicFish (global options)
BASE_OUTROOT = "Euclid-Photo-ISTF-Pess-Benchmark_"

###############################################
# Path Resolution Helpers
###############################################


def _resolve_repo_root():
    """Attempt to find the repository root by looking for pyproject.toml upward."""
    start = Path(__file__).resolve()
    for parent in [start.parent] + list(start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: assume scripts/.. is root
    return start.parent.parent


def _resolve_configs_dir(repo_root: Path):
    """Return the configs directory (cosmicfishpie/configs)."""
    candidate = repo_root / "cosmicfishpie" / "configs"
    if candidate.is_dir():
        return candidate
    # fallback to package install
    try:
        import cosmicfishpie as _cfp_pkg  # type: ignore

        pkg_dir = Path(_cfp_pkg.__file__).resolve().parent
        candidate_pkg = pkg_dir / "configs"
        if candidate_pkg.is_dir():
            return candidate_pkg
    except Exception:
        pass
    return candidate  # even if missing


def _resolve_survey_specs_dir(configs_dir: Path):
    d = configs_dir / "default_survey_specifications"
    return d


def _resolve_boltzmann_dirs(repo_root: Path):
    boltz_root = repo_root / "default_boltzmann_yaml_files"
    return boltz_root


_repo_root = _resolve_repo_root()
_configs_dir = _resolve_configs_dir(_repo_root)
_survey_specs_dir = _resolve_survey_specs_dir(_configs_dir)
_boltz_root = _resolve_boltzmann_dirs(_configs_dir)

print(f"[benchmark] repo_root:        {_repo_root}", "is dir:", _repo_root.is_dir())
print(f"[benchmark] configs_dir:      {_configs_dir}", "is dir:", _configs_dir.is_dir())
print(f"[benchmark] survey_specs_dir: {_survey_specs_dir}", "is dir:", _survey_specs_dir.is_dir())
print(f"[benchmark] boltzmann_dir:    {_boltz_root}", "is dir:", _boltz_root.is_dir())


def build_default_options(accuracy: int = 1):
    return {
        "accuracy": accuracy,
        "outroot": BASE_OUTROOT,
        "results_dir": "scripts/benchmark_results/",
        "derivatives": "3PT",
        "feedback": 1,
        "survey_name": "Euclid",
        "specs_dir": str(_survey_specs_dir) + os.sep,
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "camb_config_yaml": str(_boltz_root / "camb" / "default.yaml"),
        "class_config_yaml": str(_boltz_root / "class" / "fast_photo.yaml"),
        "symbolic_config_yaml": str(_boltz_root / "symbolic" / "default.yaml"),
        "cosmo_model": "LCDM",
        "code": "dummy",
    }


# Sanity check: confirm target photometric YAML exists (informational only)
_photo_yaml = _survey_specs_dir / "Euclid-Photometric-ISTF-Pessimistic.yaml"
if not _photo_yaml.is_file():
    print(f"[benchmark][WARNING] Expected photometric spec file missing: {_photo_yaml}")
else:
    print(f"[benchmark] Found photometric spec file: {_photo_yaml}")

freepars = {
    "Omegam": 0.01,
    "Omegab": 0.01,
    "h": 0.01,
    "ns": 0.01,
    "sigma8": 0.01,
}


def run_fisher_benchmark(options, codes, observables):
    cosmoFM = {}
    fish_photo = {}
    strings_obses = []
    timings = {}

    for code in codes:
        options["code"] = code
        for obse in observables:
            string_obse = "-".join(obse) + "_" + code
            print("Computing Fisher matrix for ", string_obse)
            strings_obses.append(string_obse)
            options["outroot"] = BASE_OUTROOT + string_obse + "_"
            start_t = time.time()
            cosmoFM[string_obse] = cosmicfish.FisherMatrix(
                fiducialpars=fiducial,
                freepars=freepars,
                options=options,
                observables=obse,
                cosmoModel=options["cosmo_model"],
            )
            fish_photo[string_obse] = cosmoFM[string_obse].compute()
            end_t = time.time()
            elapsed = end_t - start_t
            timings[string_obse] = {"backend": code, "observables": obse, "seconds": elapsed}
            print(f"Finished {string_obse} in {elapsed:.3f} s")

    # After loop restore stable outroot
    options["outroot"] = BASE_OUTROOT

    # Write timing summary to disk (aggregate)
    results_dir = options.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    timing_outfile = os.path.join(results_dir, BASE_OUTROOT + "timings.json")
    summary = {
        "outroot": BASE_OUTROOT,
        "codes": codes,
        "observables_combinations": strings_obses,
        "timings": timings,
    }
    with open(timing_outfile, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Timing information written to {timing_outfile}")

    print("Computed Fisher matrices for: ", strings_obses)

    print("Comparing results for different codes and observables")

    fisher_list = [fish_photo[key] for key in strings_obses]
    fishanalysis = fpa.CosmicFish_FisherAnalysis(fisher_list=fisher_list)
    return fishanalysis, results_dir


def _git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


# Export Fisher comparison to JSON using new exporter
def export_fisher_comparison(fishanalysis, results_dir, options, codes, observables):
    fisher_comparison_json = os.path.join(results_dir, BASE_OUTROOT + "fisher_comparison.json")
    extra_meta = {
        "git_commit": _git_commit(),
        "codes": codes,
        "observables": observables,
        "derivatives": options.get("derivatives"),
        "cosmo_model": options.get("cosmo_model"),
        "survey_name": options.get("survey_name"),
        "survey_name_photo": options.get("survey_name_photo"),
        "specs_dir": options.get("specs_dir"),
        "configs_dir": str(_configs_dir),
        "camb_config_yaml": options.get("camb_config_yaml"),
        "class_config_yaml": options.get("class_config_yaml"),
        "symbolic_config_yaml": options.get("symbolic_config_yaml"),
        "fiducial_parameters": list(fiducial.keys()),
        "free_parameters": list(freepars.keys()),
        "benchmark_script": os.path.basename(__file__),
    }

    fishanalysis.compare_fisher_results(
        parstomarg=["Omegam", "sigma8"],
        export_json_path=fisher_comparison_json,
        append=False,
        overwrite=True,
        extra_metadata=extra_meta,
    )
    print(f"Fisher comparison JSON written to {fisher_comparison_json}")


def compare_vectors(a, b, name_a="A", name_b="B", thresh=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.abs(b) > thresh
    rel = np.zeros_like(a)
    rel[mask] = (a[mask] / b[mask]) - 1.0
    abs_rel_max = np.max(np.abs(rel[mask])) if np.any(mask) else 0.0
    abs_max_near0 = np.max(np.abs(a[~mask] - b[~mask])) if np.any(~mask) else 0.0
    return {
        "rel_max": float(abs_rel_max),
        "abs_max_near0": float(abs_max_near0),
        "threshold": float(thresh),
        "count_rel": int(mask.sum()),
        "count_abs": int((~mask).sum()),
    }


def run_efficiency_benchmark(options):
    # Build a ComputeCls consistent with options (symbolic backend for speed)
    local_opts = dict(options)
    local_opts["code"] = "symbolic"
    cosmoFM = cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        freepars={"Omegam": 0.01, "h": 0.01},
        options=local_opts,
        observables=["WL", "GCph"],
        cosmoModel=local_opts["cosmo_model"],
    )
    cosmopars = {"Omegam": 0.3, "h": 0.7}
    cls = ComputeCls(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars)

    z = cls.z
    i = cls.binrange_WL[0]

    ngal_func = cls.window.norm_ngal_photoz
    comoving_func = cls.cosmo.comoving

    # Legacy O(N^2) memo-like (constructor + evaluation)
    t0 = perf_counter()
    f_memo = memo_integral_efficiency(i, ngal_func, comoving_func, z, None, None)
    t1 = perf_counter()
    eff_memo = f_memo(z)
    t2 = perf_counter()

    # Vectorized O(N^2)
    f_fast = None
    t3 = perf_counter()
    f_fast = faster_integral_efficiency(i, ngal_func, comoving_func, z)
    t4 = perf_counter()
    eff_fast = f_fast(z)
    t5 = perf_counter()

    # O(N)
    t6 = perf_counter()
    f_much = much_faster_integral_efficiency(i, ngal_func, comoving_func, z)
    t7 = perf_counter()
    eff_much = f_much(z)
    t8 = perf_counter()

    timings = {
        "memo_integral_efficiency": {"build": t1 - t0, "eval": t2 - t1},
        "faster_integral_efficiency": {"build": t4 - t3, "eval": t5 - t4},
        "much_faster_integral_efficiency": {"build": t7 - t6, "eval": t8 - t7},
    }

    comparisons = {
        "fast_vs_memo": compare_vectors(eff_fast, eff_memo, "fast", "memo"),
        "much_vs_fast": compare_vectors(eff_much, eff_fast, "much", "fast"),
    }

    results_dir = options.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, BASE_OUTROOT + "efficiency_benchmark.json")
    with open(outpath, "w") as f:
        json.dump(
            {"timings": timings, "comparisons": comparisons, "accuracy": options["accuracy"]},
            f,
            indent=2,
        )
    print("Efficiency benchmark written to", outpath)


def main():
    parser = argparse.ArgumentParser(
        description="Photometric benchmarks: Fisher and lensing-efficiency."
    )
    parser.add_argument(
        "--accuracy", type=int, default=1, help="Accuracy level (affects grid sizes)"
    )
    parser.add_argument(
        "--skip-fisher", action="store_true", help="Skip Fisher benchmark and comparisons"
    )
    parser.add_argument(
        "--efficiency", action="store_true", help="Run the lensing efficiency benchmark"
    )
    parser.add_argument(
        "--flags-benchmark",
        action="store_true",
        help="Benchmark full Fisher with fast flags ON vs OFF",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Run a single FAST Fisher (all fast flags on by default)",
    )
    parser.add_argument(
        "--compare-ref",
        type=str,
        default=None,
        help="Path to a reference Fisher matrix .txt to compare against (single current run)",
    )
    parser.add_argument(
        "--code",
        type=str,
        choices=["symbolic", "class", "camb"],
        default="symbolic",
        help="Cosmology backend to use (default: symbolic)",
    )
    parser.add_argument(
        "--observables",
        type=str,
        default="GCph,WL",
        help="Comma-separated list among {GCph, WL}; e.g. 'WL' or 'GCph,WL'",
    )
    # Granular control for FAST run flags (SLOW always disables all)
    parser.add_argument(
        "--fast-eff",
        choices=["auto", "on", "off"],
        default="auto",
        help="FAST run: COSMICFISH_FAST_EFF toggle (auto=on)",
    )
    parser.add_argument(
        "--fast-p",
        choices=["auto", "on", "off"],
        default="auto",
        help="FAST run: COSMICFISH_FAST_P toggle (auto=on)",
    )
    parser.add_argument(
        "--fast-kernel",
        choices=["auto", "on", "off"],
        default="auto",
        help="FAST run: COSMICFISH_FAST_KERNEL toggle (auto=on)",
    )
    args = parser.parse_args()

    options = build_default_options(args.accuracy)

    def _parse_obs_arg(arg: str):
        valid = {"GCph", "WL"}
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        if not parts:
            return ["GCph", "WL"]
        for p in parts:
            if p not in valid:
                raise SystemExit(f"Invalid observable '{p}'. Valid options: {sorted(valid)}")
        # preserve order and uniqueness
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    selected_obs_list = _parse_obs_arg(args.observables)

    # If any specialized mode is requested, skip the regular Fisher run
    run_regular_fisher = (
        (not args.skip_fisher)
        and (not args.flags_benchmark)
        and (not args.fast_only)
        and (not args.compare_ref)
    )

    if run_regular_fisher:
        # Override default codes with the selected backend
        selected_codes = [args.code]
        fishanalysis, results_dir = run_fisher_benchmark(
            options, selected_codes, [selected_obs_list]
        )
        export_fisher_comparison(fishanalysis, results_dir, options, codes, observables)

    if args.efficiency:
        run_efficiency_benchmark(options)

    if args.flags_benchmark:
        run_fast_flags_fisher_benchmark(options, args, selected_obs_list)

    if args.fast_only:
        run_fast_only(options, args, selected_obs_list)

    if args.compare_ref:
        run_compare_ref(options, args, selected_obs_list)


def run_fast_flags_fisher_benchmark(options, args, obs_list):
    """Benchmark a full Fisher compute with fast flags enabled vs disabled.

    Runs with observables ["WL","GCph"] and the symbolic backend for speed.
    Results are written to the results_dir as a JSON summary.
    """
    results_dir = options.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)

    def _set_flags(eff: bool, P: bool, kernel: bool):
        os.environ["COSMICFISH_FAST_EFF"] = "1" if eff else "0"
        os.environ["COSMICFISH_FAST_P"] = "1" if P else "0"
        os.environ["COSMICFISH_FAST_KERNEL"] = "1" if kernel else "0"
        # Reload photo_obs so module-level toggles pick up env
        import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

        importlib.reload(photo_obs)
        # Ensure cosmicfish module sees the reloaded module
        import cosmicfishpie.fishermatrix.cosmicfish as _cf

        importlib.reload(_cf)

    def _run_once(tag: str):
        local_opts = dict(options)
        local_opts["code"] = args.code
        local_opts["outroot"] = f"{BASE_OUTROOT}{tag}_"
        start_t = time.time()
        fm = cosmicfish.FisherMatrix(
            fiducialpars=fiducial,
            freepars=freepars,
            options=local_opts,
            observables=obs_list,
            cosmoModel=local_opts["cosmo_model"],
        )
        res = fm.compute()
        elapsed = time.time() - start_t
        return elapsed, res

    # Run SLOW first (baseline), then FAST
    # Determine FAST run flag booleans from CLI
    fast_eff = args.fast_eff != "off"  # auto->on, on->on
    fast_P = args.fast_p != "off"
    fast_kernel = args.fast_kernel != "off"

    print("\n[flags-benchmark] Fast flags: OFF (SLOW baseline)")
    _set_flags(False, False, False)
    t_slow, res_slow = _run_once("SLOW")
    print(f"[flags-benchmark] Time (SLOW): {t_slow:.3f} s")

    print("[flags-benchmark] Fast flags: ON (FAST)")
    _set_flags(fast_eff, fast_P, fast_kernel)
    t_fast, res_fast = _run_once("FAST")
    print(f"[flags-benchmark] Time (FAST): {t_fast:.3f} s")

    # Build comparison metrics
    def _matrix_metrics(F: np.ndarray):
        F = np.asarray(F)
        # Robust logdet: use slogdet; if non-positive, compute pseudo-logdet on positive eigvals
        sign, logdet = np.linalg.slogdet(F)
        if not np.isfinite(logdet) or sign <= 0:
            w = np.linalg.eigvalsh((F + F.T) * 0.5)
            wpos = w[w > 1e-18]
            if wpos.size:
                logdet = float(np.sum(np.log(wpos)))
            else:
                logdet = float("-inf")
        tr = float(np.trace(F))
        fro = float(np.linalg.norm(F))
        return {"trace": tr, "frobenius": fro, "logdet": logdet}

    def _compare(F_fast: np.ndarray, F_slow: np.ndarray, thresh: float = 1e-12):
        A = np.asarray(F_fast)
        B = np.asarray(F_slow)
        D = A - B
        mask = np.abs(B) > thresh
        rel = np.zeros_like(A)
        rel[mask] = (A[mask] / B[mask]) - 1.0
        rel_max = float(np.max(np.abs(rel[mask]))) if np.any(mask) else 0.0
        abs_max_near0 = float(np.max(np.abs(D[~mask]))) if np.any(~mask) else 0.0
        fro_rel = float(np.linalg.norm(D) / (np.linalg.norm(B) + 1e-30))
        # Diagonal ratios stats
        dA = np.diag(A)
        dB = np.diag(B)
        dmask = np.abs(dB) > thresh
        diag_ratio_stats = {}
        if np.any(dmask):
            ratios = dA[dmask] / dB[dmask]
            diag_ratio_stats = {
                "min": float(np.min(ratios)),
                "max": float(np.max(ratios)),
                "median": float(np.median(ratios)),
            }
        return {
            "rel_max": rel_max,
            "abs_max_near0": abs_max_near0,
            "fro_rel": fro_rel,
            "diag_ratio": diag_ratio_stats,
        }

    metrics_fast = _matrix_metrics(res_fast.fisher_matrix)
    metrics_slow = _matrix_metrics(res_slow.fisher_matrix)
    cmp_metrics = _compare(res_fast.fisher_matrix, res_slow.fisher_matrix)

    # Emit small report
    print("[flags-benchmark] Comparison report:")
    summary_speed = (t_slow / t_fast) if t_fast > 0 else float("inf")
    print(f"  - speedup (slow/fast): {summary_speed:.2f}x")
    print(
        "  - rel entry max |ratio-1|:",
        f"{cmp_metrics['rel_max']:.3e}",
        "| abs max near zero:",
        f"{cmp_metrics['abs_max_near0']:.3e}",
        "| frob rel diff:",
        f"{cmp_metrics['fro_rel']:.3e}",
    )
    print(
        f"  - logdet (FAST)={metrics_fast['logdet']:.6e} vs (SLOW)={metrics_slow['logdet']:.6e} "
        f"| trace FAST={metrics_fast['trace']:.6e} SLOW={metrics_slow['trace']:.6e}"
    )
    if cmp_metrics.get("diag_ratio"):
        dr = cmp_metrics["diag_ratio"]
        print(
            f"  - diag ratio stats (FAST/SLOW): min={dr['min']:.6e}, median={dr['median']:.6e}, max={dr['max']:.6e}"
        )

    # Collect per-run metadata
    runs = [
        {
            "mode": "SLOW",
            "seconds": t_slow,
            "env_flags": {
                "COSMICFISH_FAST_EFF": False,
                "COSMICFISH_FAST_P": False,
                "COSMICFISH_FAST_KERNEL": False,
            },
            "matrix_path": getattr(res_slow, "file_name", None),
        },
        {
            "mode": "FAST",
            "seconds": t_fast,
            "env_flags": {
                "COSMICFISH_FAST_EFF": fast_eff,
                "COSMICFISH_FAST_P": fast_P,
                "COSMICFISH_FAST_KERNEL": fast_kernel,
            },
            "matrix_path": getattr(res_fast, "file_name", None),
        },
    ]

    summary = {
        "accuracy": options["accuracy"],
        "code": args.code,
        "observables": obs_list,
        "speedup": (t_slow / t_fast) if t_fast > 0 else None,
        "runs": runs,
        "metrics": {
            "fast": metrics_fast,
            "slow": metrics_slow,
            "comparison": cmp_metrics,
        },
        "options": options,
    }

    outpath = os.path.join(results_dir, BASE_OUTROOT + "flags_fisher_benchmark.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print("[flags-benchmark] Summary written to", outpath)


def run_fast_only(options, args, obs_list):
    """Run a single FAST Fisher computation with env flags enabled.

    Honors --code, --observables, and granular --fast-* toggles.
    Exports the matrix with a FASTONLY tag and writes a small JSON summary.
    """
    import importlib

    def _set_flags(eff: bool, P: bool, kernel: bool):
        os.environ["COSMICFISH_FAST_EFF"] = "1" if eff else "0"
        os.environ["COSMICFISH_FAST_P"] = "1" if P else "0"
        os.environ["COSMICFISH_FAST_KERNEL"] = "1" if kernel else "0"
        # Reload photo_obs so module-level toggles pick up env
        import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

        importlib.reload(photo_obs)
        # Ensure cosmicfish module sees the reloaded module
        import cosmicfishpie.fishermatrix.cosmicfish as _cf

        importlib.reload(_cf)

    # Determine FAST toggles
    fast_eff = args.fast_eff != "off"
    fast_P = args.fast_p != "off"
    fast_kernel = args.fast_kernel != "off"
    _set_flags(fast_eff, fast_P, fast_kernel)

    local_opts = dict(options)
    local_opts["code"] = args.code
    local_opts["outroot"] = f"{BASE_OUTROOT}FASTONLY_"

    start_t = time.time()
    fm = cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=local_opts,
        observables=obs_list,
        cosmoModel=local_opts["cosmo_model"],
    )
    res = fm.compute()
    elapsed = time.time() - start_t
    print(f"[fast-only] Time (FAST): {elapsed:.3f} s")

    results_dir = options.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, BASE_OUTROOT + "fast_only.json")
    with open(outpath, "w") as f:
        json.dump(
            {
                "accuracy": options["accuracy"],
                "code": args.code,
                "observables": obs_list,
                "seconds_fast": elapsed,
                "env_flags": {
                    "COSMICFISH_FAST_EFF": fast_eff,
                    "COSMICFISH_FAST_P": fast_P,
                    "COSMICFISH_FAST_KERNEL": fast_kernel,
                },
                "matrix_path": getattr(res, "file_name", None),
                "options": options,
            },
            f,
            indent=2,
        )
    print("[fast-only] Summary written to", outpath)


def run_compare_ref(options, args, obs_list):
    """Run a single Fisher with current settings and compare to a reference matrix.

    Honors --code, --observables and FAST toggles (defaults to FAST on).
    Writes a JSON report with alignment info and difference metrics.
    """
    import importlib

    from cosmicfishpie.analysis import fisher_matrix as fm_mod

    def _set_flags(eff: bool, P: bool, kernel: bool):
        os.environ["COSMICFISH_FAST_EFF"] = "1" if eff else "0"
        os.environ["COSMICFISH_FAST_P"] = "1" if P else "0"
        os.environ["COSMICFISH_FAST_KERNEL"] = "1" if kernel else "0"
        import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

        importlib.reload(photo_obs)
        import cosmicfishpie.fishermatrix.cosmicfish as _cf

        importlib.reload(_cf)

    # Determine FAST toggles for the current run
    fast_eff = args.fast_eff != "off"
    fast_P = args.fast_p != "off"
    fast_kernel = args.fast_kernel != "off"
    _set_flags(fast_eff, fast_P, fast_kernel)

    # Compute current Fisher
    local_opts = dict(options)
    local_opts["code"] = args.code
    local_opts["outroot"] = f"{BASE_OUTROOT}COMPARE_"
    start_t = time.time()
    fm_cur = cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=local_opts,
        observables=obs_list,
        cosmoModel=local_opts["cosmo_model"],
    ).compute()
    t_cur = time.time() - start_t

    # Load reference Fisher
    ref_path = args.compare_ref
    fm_ref = fm_mod.fisher_matrix(file_name=ref_path)

    # Align parameter order: create mapping from ref names to cur indices
    ref_names = list(fm_ref.param_names)
    cur_names = list(fm_cur.param_names)
    name_to_idx = {n: i for i, n in enumerate(cur_names)}
    missing = [n for n in ref_names if n not in name_to_idx]
    if missing:
        print("[compare-ref] WARNING: parameters missing in current run:", missing)
    order = [name_to_idx[n] for n in ref_names if n in name_to_idx]

    F_ref = np.array(fm_ref.fisher_matrix)
    F_cur_full = np.array(fm_cur.fisher_matrix)
    F_cur = F_cur_full[np.ix_(order, order)]

    # Compute metrics
    def metrics(A, B, thresh=1e-12):
        D = A - B
        mask = np.abs(B) > thresh
        rel = np.zeros_like(A)
        rel[mask] = (A[mask] / B[mask]) - 1.0
        out = {
            "rel_max": float(np.max(np.abs(rel[mask]))) if np.any(mask) else 0.0,
            "abs_max_near0": float(np.max(np.abs(D[~mask]))) if np.any(~mask) else 0.0,
            "fro_rel": float(np.linalg.norm(D) / (np.linalg.norm(B) + 1e-30)),
        }
        # diag ratio stats
        dA = np.diag(A)
        dB = np.diag(B)
        dmask = np.abs(dB) > thresh
        if np.any(dmask):
            ratios = dA[dmask] / dB[dmask]
            out["diag_ratio"] = {
                "min": float(np.min(ratios)),
                "median": float(np.median(ratios)),
                "max": float(np.max(ratios)),
            }
        else:
            out["diag_ratio"] = {}
        return out

    cmp = metrics(F_cur, F_ref)
    print("[compare-ref] Current run time:", f"{t_cur:.3f}s")
    print(
        "[compare-ref] rel_max=",
        f"{cmp['rel_max']:.3e}",
        "abs_max_near0=",
        f"{cmp['abs_max_near0']:.3e}",
        "fro_rel=",
        f"{cmp['fro_rel']:.3e}",
    )

    # Write JSON report
    results_dir = options.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, BASE_OUTROOT + "compare_ref.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "code": args.code,
                "observables": obs_list,
                "seconds_current": t_cur,
                "current_matrix_path": getattr(fm_cur, "file_name", None),
                "reference_matrix_path": ref_path,
                "param_order_reference": ref_names,
                "param_order_current": cur_names,
                "aligned_order_indices": order,
                "env_flags": {
                    "COSMICFISH_FAST_EFF": fast_eff,
                    "COSMICFISH_FAST_P": fast_P,
                    "COSMICFISH_FAST_KERNEL": fast_kernel,
                },
                "metrics": cmp,
            },
            f,
            indent=2,
        )
    print("[compare-ref] Report written to", out_json)


if __name__ == "__main__":
    main()
