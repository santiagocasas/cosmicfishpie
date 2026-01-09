#!/usr/bin/env python
# coding: utf-8

# # CosmicFishPie

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
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
BASE_OUTROOT_BASE = "Euclid-Photo-ISTF-Pess-Benchmark"
BASE_OUTROOT = BASE_OUTROOT_BASE + "_"

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
    return fishanalysis, results_dir, fish_photo, strings_obses, timings


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


def _normalize_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("_") else prefix + "_"


def _prefix_from_summary_file(filename: str) -> str:
    if filename.endswith("timings.json"):
        return filename[: -len("timings.json")]
    if filename.endswith("fisher_comparison.json"):
        return filename[: -len("fisher_comparison.json")]
    return ""


def _resolve_compare_ref(arg: str, default_results_dir: str):
    if not arg:
        return None, None
    if os.path.isdir(arg):
        candidates = []
        for name in os.listdir(arg):
            if name.endswith("timings.json") or name.endswith("fisher_comparison.json"):
                candidates.append(name)
        if not candidates:
            print(f"[compare-ref] No summary JSON found in {arg}")
            sys.exit(1)
        candidates.sort(key=lambda n: os.path.getmtime(os.path.join(arg, n)), reverse=True)
        chosen = candidates[0]
        prefix = _prefix_from_summary_file(chosen)
        return arg, _normalize_prefix(prefix)
    if os.path.isfile(arg):
        prefix = _prefix_from_summary_file(os.path.basename(arg))
        if not prefix:
            print(f"[compare-ref] Unsupported reference file: {arg}")
            sys.exit(1)
        return os.path.dirname(arg), _normalize_prefix(prefix)
    ref_dir = default_results_dir
    ref_prefix = arg
    if os.sep in arg:
        ref_dir = os.path.dirname(arg)
        ref_prefix = os.path.basename(arg)
    return ref_dir, _normalize_prefix(ref_prefix)


def _find_ref_specs(ref_dir: str, ref_prefix: str):
    if not ref_dir or not os.path.isdir(ref_dir):
        return []
    specs = []
    for name in os.listdir(ref_dir):
        if ref_prefix and not name.startswith(ref_prefix):
            continue
        if name.endswith("_FM_specs.json") or name.endswith("_specifications.json"):
            specs.append(os.path.join(ref_dir, name))
    return sorted(specs)


def _compare_matrices(fm_cur, fm_ref):
    ref_names = list(fm_ref.param_names)
    cur_names = list(fm_cur.param_names)
    name_to_idx = {n: i for i, n in enumerate(cur_names)}
    common = [n for n in ref_names if n in name_to_idx]
    missing = [n for n in ref_names if n not in name_to_idx]
    if not common:
        return {"error": "no_common_params", "missing_in_current": missing}
    ref_indices = [ref_names.index(n) for n in common]
    cur_indices = [name_to_idx[n] for n in common]
    F_ref = np.array(fm_ref.fisher_matrix)[np.ix_(ref_indices, ref_indices)]
    F_cur = np.array(fm_cur.fisher_matrix)[np.ix_(cur_indices, cur_indices)]

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

    out = metrics(F_cur, F_ref)
    out["missing_in_current"] = missing
    return out


def compare_run_to_ref(fish_photo, strings_obses, results_dir, ref_dir, ref_prefix, timings):
    from cosmicfishpie.analysis import fisher_matrix as fm_mod

    ref_timings_path = os.path.join(ref_dir, ref_prefix + "timings.json")
    ref_timings = {}
    if os.path.isfile(ref_timings_path):
        try:
            with open(ref_timings_path, "r") as f:
                ref_timings = json.load(f).get("timings", {})
        except Exception:
            ref_timings = {}

    summary = {
        "reference": {"dir": ref_dir, "prefix": ref_prefix},
        "current_prefix": BASE_OUTROOT,
        "entries": [],
    }
    for key in strings_obses:
        res = fish_photo.get(key)
        cur_txt = getattr(res, "file_name", None) if res else None
        entry = {"name": key, "current_matrix_path": cur_txt}
        if key in timings:
            entry["timing_current_seconds"] = timings[key].get("seconds")
        if key in ref_timings:
            entry["timing_reference_seconds"] = ref_timings[key].get("seconds")
        if not cur_txt:
            entry["error"] = "missing_current_matrix"
            summary["entries"].append(entry)
            continue
        cur_base = os.path.basename(cur_txt)
        if BASE_OUTROOT in cur_base:
            ref_base = cur_base.replace(BASE_OUTROOT, ref_prefix, 1)
            ref_txt = os.path.join(ref_dir, ref_base)
        else:
            ref_txt = os.path.join(ref_dir, ref_prefix + cur_base)
        entry["reference_matrix_path"] = ref_txt
        if not os.path.isfile(ref_txt):
            entry["error"] = "missing_reference_matrix"
            summary["entries"].append(entry)
            continue
        fm_ref = fm_mod.fisher_matrix(file_name=ref_txt)
        entry["metrics"] = _compare_matrices(res, fm_ref)
        summary["entries"].append(entry)

    out_json = os.path.join(results_dir, BASE_OUTROOT + "compare_ref.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("[compare-ref] Report written to", out_json)


def compare_ref_replay(ref_specs, results_dir, ref_dir, ref_prefix):
    from cosmicfishpie.analysis import fisher_matrix as fm_mod
    from cosmicfishpie.utilities import utils as _u

    print(
        f"[compare-ref] Replaying {len(ref_specs)} run(s) from {ref_dir} "
        f"(prefix='{ref_prefix or '*'}')"
    )
    for spec_path in ref_specs:
        print(f"[compare-ref]   using specs: {spec_path}")

    observables_sets = []
    for spec_path in ref_specs:
        try:
            _, snap = _u.load_fisher_from_json(spec_path)
            obs = snap.get("metadata", {}).get("observables")
            if obs is not None:
                observables_sets.append(tuple(obs))
        except Exception:
            continue
    if observables_sets and len(set(observables_sets)) > 1:
        raise SystemExit(
            "[compare-ref] Reference specs contain multiple observables sets; "
            "please compare runs with matching observables only."
        )

    summary = {
        "reference": {"dir": ref_dir, "prefix": ref_prefix},
        "current_prefix": BASE_OUTROOT,
        "mode": "replay_specs",
        "entries": [],
    }
    for spec_path in ref_specs:
        fm, snap = _u.load_fisher_from_json(spec_path)
        ref_outroot = fm.settings.get("outroot", "")
        if ref_prefix in ref_outroot:
            new_outroot = ref_outroot.replace(ref_prefix, BASE_OUTROOT, 1)
        else:
            base = os.path.basename(ref_outroot.rstrip("_")) or "RUN"
            new_outroot = BASE_OUTROOT + base + "_"
        fm.settings["outroot"] = new_outroot
        fm.settings["results_dir"] = results_dir

        t0 = time.time()
        res = fm.compute()
        t1 = time.time()

        ref_txt = (
            snap.get("metadata", {}).get("matrix_files", {}).get("txt")
            if isinstance(snap.get("metadata", {}), dict)
            else None
        )
        if ref_txt and not os.path.isfile(ref_txt):
            ref_txt = os.path.join(ref_dir, os.path.basename(ref_txt))
        entry = {
            "ref_specs_json": spec_path,
            "current_matrix_path": getattr(res, "file_name", None),
            "reference_matrix_path": ref_txt,
            "observables": snap.get("metadata", {}).get("observables"),
            "timing_current_seconds": t1 - t0,
            "timing_reference_seconds": snap.get("metadata", {}).get("totaltime_sec"),
        }
        if ref_txt and os.path.isfile(ref_txt):
            fm_ref = fm_mod.fisher_matrix(file_name=ref_txt)
            entry["metrics"] = _compare_matrices(res, fm_ref)
        else:
            entry["error"] = "missing_reference_matrix"
        summary["entries"].append(entry)

    out_json = os.path.join(results_dir, BASE_OUTROOT + "compare_ref.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("[compare-ref] Report written to", out_json)


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
        description="Photometric benchmarks: Fisher and lensing-efficiency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--fast-benchmark",
        action="store_true",
        help="Benchmark full Fisher with FAST flags ON vs OFF",
    )
    parser.add_argument(
        "--code",
        type=str,
        choices=["symbolic", "class", "camb"],
        default="symbolic",
        help="Cosmology backend to use (default: symbolic)",
    )
    parser.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Comma-separated list of backends or 'all' (overrides --code)",
    )
    parser.add_argument(
        "--observables",
        type=str,
        default="GCph,WL",
        help="Comma-separated list among {GCph, WL}; e.g. 'WL' or 'GCph,WL'",
    )
    parser.add_argument(
        "--compare-ref",
        type=str,
        default=None,
        help="Compare against a previous run prefix or summary JSON (replays settings if specs are found)",
    )
    parser.add_argument(
        "--fast",
        choices=["auto", "on", "off"],
        default="auto",
        help="Set all FAST toggles at once (auto=use individual flags)",
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
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=8,
        help="Set OMP_NUM_THREADS for the run",
    )
    args = parser.parse_args()

    def _make_run_id() -> str:
        arg_text = " ".join(sys.argv[1:]).strip() or "defaults"
        digest = hashlib.sha1(arg_text.encode("utf-8")).hexdigest()[:8]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{digest}"

    global BASE_OUTROOT
    run_id = _make_run_id()
    BASE_OUTROOT = f"{BASE_OUTROOT_BASE}_{run_id}_"

    # Default FAST flags for runs that don't explicitly toggle them later.
    if args.fast == "on":
        fast_eff = True
        fast_P = True
        fast_kernel = True
    elif args.fast == "off":
        fast_eff = False
        fast_P = False
        fast_kernel = False
    else:
        fast_eff = args.fast_eff != "off"
        fast_P = args.fast_p != "off"
        fast_kernel = args.fast_kernel != "off"
    os.environ["COSMICFISH_FAST_EFF"] = "1" if fast_eff else "0"
    os.environ["COSMICFISH_FAST_P"] = "1" if fast_P else "0"
    os.environ["COSMICFISH_FAST_KERNEL"] = "1" if fast_kernel else "0"
    # Ensure photo_obs picks up env toggles set here.
    import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

    importlib.reload(photo_obs)
    global ComputeCls, faster_integral_efficiency, memo_integral_efficiency
    global much_faster_integral_efficiency
    ComputeCls = photo_obs.ComputeCls
    faster_integral_efficiency = photo_obs.faster_integral_efficiency
    memo_integral_efficiency = photo_obs.memo_integral_efficiency
    much_faster_integral_efficiency = photo_obs.much_faster_integral_efficiency

    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    options = build_default_options(args.accuracy)
    compare_ref_dir, compare_ref_prefix = _resolve_compare_ref(
        args.compare_ref, options.get("results_dir", "results/")
    )
    ref_specs = _find_ref_specs(compare_ref_dir, compare_ref_prefix)
    if args.compare_ref and not ref_specs and compare_ref_dir and os.path.isdir(compare_ref_dir):
        # Fall back to any specs in the directory if prefix was derived from summary files.
        ref_specs = _find_ref_specs(compare_ref_dir, "")
        if ref_specs:
            compare_ref_prefix = ""
    if args.compare_ref and not ref_specs:
        raise SystemExit(
            "[compare-ref] No reference *_FM_specs.json found for prefix: "
            f"{compare_ref_prefix} in {compare_ref_dir}"
        )

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

    def _parse_codes_arg(arg: str | None) -> list[str]:
        valid = ["symbolic", "class", "camb"]
        if not arg:
            return [args.code]
        if arg.strip().lower() == "all":
            return valid
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        if not parts:
            return [args.code]
        for p in parts:
            if p not in valid:
                raise SystemExit(f"Invalid code '{p}'. Valid options: {valid} or 'all'")
        # preserve order and uniqueness
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    selected_codes = _parse_codes_arg(args.codes)

    run_regular_fisher = (
        (not args.skip_fisher) and (not args.fast_benchmark) and (not args.compare_ref)
    )

    if args.compare_ref and ref_specs:
        results_dir = options.get("results_dir", "results/")
        os.makedirs(results_dir, exist_ok=True)
        compare_ref_replay(ref_specs, results_dir, compare_ref_dir, compare_ref_prefix)

    if run_regular_fisher:
        # Override default codes with the selected backend(s)
        fishanalysis, results_dir, fish_photo, strings_obses, timings = run_fisher_benchmark(
            options, selected_codes, [selected_obs_list]
        )
        export_fisher_comparison(fishanalysis, results_dir, options, codes, observables)
        if compare_ref_prefix:
            compare_run_to_ref(
                fish_photo,
                strings_obses,
                results_dir,
                compare_ref_dir,
                compare_ref_prefix,
                timings,
            )

    if args.efficiency:
        run_efficiency_benchmark(options)

    if args.fast_benchmark:
        run_fast_flags_fisher_benchmark(options, args, selected_obs_list)


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
    import sys

    from cosmicfishpie.analysis import fisher_matrix as fm_mod

    def _set_flags(eff: bool, P: bool, kernel: bool):
        os.environ["COSMICFISH_FAST_EFF"] = "1" if eff else "0"
        os.environ["COSMICFISH_FAST_P"] = "1" if P else "0"
        os.environ["COSMICFISH_FAST_KERNEL"] = "1" if kernel else "0"
        import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

        importlib.reload(photo_obs)
        import cosmicfishpie.fishermatrix.cosmicfish as _cf

        importlib.reload(_cf)

    # Pre-check reference path(s) exist before doing any heavy compute
    ref_arg = args.compare_ref
    candidates = []
    if os.path.isfile(ref_arg):
        candidates = [ref_arg]
    else:
        candidates = [ref_arg + "_FM.txt", ref_arg + "_fishermatrix.txt"]
    if not any(os.path.isfile(p) for p in candidates):
        msg = "[compare-ref] Reference not found. Checked: " + ", ".join(candidates)
        print(msg)
        sys.exit(1)

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

    # Resolve reference root and files
    ref_arg = args.compare_ref
    # If an explicit file path exists, honor it directly for the TXT
    if os.path.isfile(ref_arg):
        if ref_arg.endswith("_FM.txt"):
            ref_txt = ref_arg
            ref_root = ref_arg[: -len("_FM.txt")]
        elif ref_arg.endswith("_fishermatrix.txt"):
            # Legacy file naming
            ref_txt = ref_arg
            ref_root = ref_arg[: -len("_fishermatrix.txt")]
        elif ref_arg.endswith(".txt"):
            # Unknown TXT suffix; treat as literal file and derive root by dropping extension
            ref_txt = ref_arg
            ref_root = ref_arg[:-4]
        elif (
            ref_arg.endswith("_FM_specs.json")
            or ref_arg.endswith("_specifications.json")
            or ref_arg.endswith(".json")
        ):
            # JSON reference provided; derive root from it
            if ref_arg.endswith("_FM_specs.json"):
                ref_root = ref_arg[: -len("_FM_specs.json")]
            elif ref_arg.endswith("_specifications.json"):
                ref_root = ref_arg[: -len("_specifications.json")]
            else:
                ref_root = ref_arg[: -len(".json")]
            # Prefer new TXT; fallback to legacy
            ref_txt = ref_root + "_FM.txt"
            if not os.path.isfile(ref_txt):
                legacy_txt = ref_root + "_fishermatrix.txt"
                if os.path.isfile(legacy_txt):
                    ref_txt = legacy_txt
        else:
            # Treat as root without known suffix
            ref_root = ref_arg
            ref_txt = ref_root + "_FM.txt"
            if not os.path.isfile(ref_txt):
                legacy_txt = ref_root + "_fishermatrix.txt"
                if os.path.isfile(legacy_txt):
                    ref_txt = legacy_txt
    else:
        # Not a file; treat as root and compose expected names
        ref_root = ref_arg
        ref_txt = ref_root + "_FM.txt"
        if not os.path.isfile(ref_txt):
            legacy_txt = ref_root + "_fishermatrix.txt"
            if os.path.isfile(legacy_txt):
                ref_txt = legacy_txt
    # Prefer new JSON name, fallback to legacy
    ref_json = ref_root + "_FM_specs.json"
    if not os.path.isfile(ref_json):
        legacy = ref_root + "_specifications.json"
        ref_json = legacy if os.path.isfile(legacy) else None

    fm_ref = fm_mod.fisher_matrix(file_name=ref_txt)

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
    # Write a per-comparison JSON (include observables and backend to avoid overwrite)
    obs_tag = "-".join(obs_list)
    out_json = os.path.join(results_dir, f"{BASE_OUTROOT}compare_ref_{obs_tag}_{args.code}.json")
    # Option-diff if JSON available
    ref_snapshot = None
    cur_snapshot = None
    try:
        import json as _json

        # load current json if present
        cur_json = snap_path_from_matrix(getattr(fm_cur, "file_name", None))
        if cur_json and os.path.isfile(cur_json):
            with open(cur_json, "r") as jf:
                cur_snapshot = _json.load(jf)
        if ref_json and os.path.isfile(ref_json):
            with open(ref_json, "r") as jf:
                ref_snapshot = _json.load(jf)
    except Exception:
        pass

    def _shallow_diff(a: dict, b: dict):
        dif = {"changed": {}, "only_in_a": [], "only_in_b": []}
        if a is None or b is None:
            return dif
        ka = set(a.keys())
        kb = set(b.keys())
        for k in sorted(ka - kb):
            dif["only_in_a"].append(k)
        for k in sorted(kb - ka):
            dif["only_in_b"].append(k)
        for k in sorted(ka & kb):
            va, vb = a.get(k), b.get(k)
            if va != vb:
                dif["changed"][k] = {"a": va, "b": vb}
        return dif

    json_diff = None
    if ref_snapshot and cur_snapshot:
        json_diff = {
            "options": _shallow_diff(
                ref_snapshot.get("options", {}), cur_snapshot.get("options", {})
            ),
            "specifications": _shallow_diff(
                ref_snapshot.get("specifications", {}), cur_snapshot.get("specifications", {})
            ),
            "fiducialpars": _shallow_diff(
                ref_snapshot.get("fiducialpars", {}), cur_snapshot.get("fiducialpars", {})
            ),
            "freepars": _shallow_diff(
                ref_snapshot.get("freepars", {}), cur_snapshot.get("freepars", {})
            ),
        }
    with open(out_json, "w") as f:
        json.dump(
            {
                "code": args.code,
                "observables": obs_list,
                "seconds_current": t_cur,
                "current_matrix_path": getattr(fm_cur, "file_name", None),
                "reference_matrix_path": ref_txt,
                "param_order_reference": ref_names,
                "param_order_current": cur_names,
                "aligned_order_indices": order,
                "env_flags": {
                    "COSMICFISH_FAST_EFF": fast_eff,
                    "COSMICFISH_FAST_P": fast_P,
                    "COSMICFISH_FAST_KERNEL": fast_kernel,
                },
                "metrics": cmp,
                "json_paths": {"ref": ref_json, "current": cur_json},
                "json_diff": json_diff,
            },
            f,
            indent=2,
        )
    print("[compare-ref] Report written to", out_json)


def snap_path_from_matrix(matrix_txt_path: str):
    if not matrix_txt_path:
        return None
    if matrix_txt_path.endswith("_FM.txt"):
        return matrix_txt_path[:-7] + "_FM_specs.json"
    # legacy fallback
    if matrix_txt_path.endswith("_fishermatrix.txt"):
        return matrix_txt_path[:-16] + "_specifications.json"
    return None


def run_replay_json(json_path):
    """Replay a run from a JSON snapshot and compare to its referenced matrix if available."""
    import os
    import sys

    import numpy as np

    from cosmicfishpie.analysis import fisher_matrix as fm_mod
    from cosmicfishpie.utilities import utils as _u

    # Resolve to a JSON file if a root was provided
    def _resolve_json(p: str) -> str:
        if os.path.isfile(p):
            return p
        # If user passed a root ending with '_FM', assume new naming
        if p.endswith("_FM") and os.path.isfile(p + "_specs.json"):
            return p + "_specs.json"
        cand = p + "_FM_specs.json"
        if os.path.isfile(cand):
            return cand
        # Legacy fallback
        cand2 = p + "_specifications.json"
        if os.path.isfile(cand2):
            return cand2
        return p

    json_path = _resolve_json(json_path)
    if not os.path.isfile(json_path):
        print(f"[replay-json] JSON snapshot not found: {json_path}")
        sys.exit(1)

    fm, snap = _u.load_fisher_from_json(json_path)
    # Compute
    import time as _time

    t0 = _time.time()
    res = fm.compute()
    t1 = _time.time()
    print(f"[replay-json] Run time: {t1 - t0:.3f}s")

    # If reference txt exists, compare
    ref_txt = snap.get("metadata", {}).get("matrix_files", {}).get("txt")
    if ref_txt and os.path.isfile(ref_txt):
        fm_ref = fm_mod.fisher_matrix(file_name=ref_txt)
        F_ref = np.array(fm_ref.fisher_matrix)
        F_cur = np.array(res.fisher_matrix)
        # Align by parameter names
        ref_names = list(fm_ref.param_names)
        cur_names = list(res.param_names)
        idx = {n: i for i, n in enumerate(cur_names)}
        order = [idx[n] for n in ref_names if n in idx]
        F_cur = F_cur[np.ix_(order, order)]
        D = F_cur - F_ref
        fro_rel = float(np.linalg.norm(D) / (np.linalg.norm(F_ref) + 1e-30))
        print(f"[replay-json] frob rel diff vs reference: {fro_rel:.3e}")
    else:
        print("[replay-json] Reference TXT not found; skipped matrix comparison.")


if __name__ == "__main__":
    main()
