#!/usr/bin/env python
"""Comprehensive profiler for photometric Cl / FisherMatrix performance.

This script is self-contained (no library code edits required) and can be copied
between branches (e.g. performance_opt and main) to produce comparable JSON summaries.

It runs Fisher computations for selected cosmology backends / observables / accuracy
settings, profiles with cProfile, then extracts cumulative times for key hotspot
functions in `photo_obs.py` and `photo_cov.py`.

Example usage:
  python scripts/profile_photo_obs.py --observables GCph WL --code symbolic \
      --repeats 2 --accuracy 1 2 --output results/profile_photo_obs_symbolic.json

Compare multiple codes:
  python scripts/profile_photo_obs.py --observables GCph WL \
      --code symbolic camb class --repeats 1 --output results/profile_multi.json

Light (no .prof retention, JSON only):
  python scripts/profile_photo_obs.py --observables GCph --code symbolic --output results/quick.json

Keep raw cProfile files (one per run):
  python scripts/profile_photo_obs.py --observables GCph WL --code symbolic --keep-prof --output results/with_prof.json

CSV export of aggregate per-function timings:
  python scripts/profile_photo_obs.py --observables GCph WL --code symbolic --csv results/summary.csv

Notes:
- Symbolic backend supports LCDM only (auto-handled here).
- Derivatives can dominate total time; you can restrict free parameters by editing `freepars` below if you want purer getcls timing.
- Function cumulative times are taken from pstats (cumulative includes time in subcalls).
- You can diff JSON outputs between branches to assess speedups.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import pstats
import statistics as st
import sys
import time

# --- Spec directory discovery (robust across branches) ---
from pathlib import Path
from pathlib import Path as _P

from cosmicfishpie.configs import config as cfg
from cosmicfishpie.fishermatrix import cosmicfish


def _resolve_repo_root():
    start = Path(__file__).resolve()
    for parent in [start.parent] + list(start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return start.parent.parent


def _resolve_configs_dir(repo_root: Path):
    candidate = repo_root / "cosmicfishpie" / "configs"
    if candidate.is_dir():
        return candidate
    # fallback to installed package location
    pkg_root = _P(cfg.__file__).resolve().parent
    return pkg_root


def _resolve_survey_specs_dir(configs_dir: Path):
    d = configs_dir / "default_survey_specifications"
    return d


def _resolve_boltzmann_dir(repo_root: Path):
    # Try both names used in repo (boltzmann_yaml_files or default_boltzmann_yaml_files)
    for name in ["boltzmann_yaml_files", "default_boltzmann_yaml_files"]:
        cand = repo_root / name
        if cand.is_dir():
            return cand
    # fallback under configs
    configs = _resolve_configs_dir(repo_root)
    for name in ["boltzmann_yaml_files", "default_boltzmann_yaml_files"]:
        cand = configs / name
        if cand.is_dir():
            return cand
    return repo_root / "boltzmann_yaml_files"


def _discover_specs_dir(user_override: str | None = None):
    if user_override:
        if os.path.isdir(user_override):
            return os.path.abspath(user_override) + os.sep
        raise FileNotFoundError(f"--specs-dir '{user_override}' not found")
    # Existing config hints
    for cand in [getattr(cfg, "specs_dir", None), getattr(cfg, "SPECSDIR", None)]:
        if isinstance(cand, str) and cand and os.path.isdir(cand):
            return cand if cand.endswith(os.sep) else cand + os.sep
    # Package local (parent already points to configs dir) -> just append directory directly
    pkg_root = _P(cfg.__file__).resolve().parent  # .../cosmicfishpie/configs
    candidate = pkg_root / "default_survey_specifications"
    if candidate.is_dir():
        return str(candidate) + os.sep
    # Repo root search
    repo_root = _resolve_repo_root()
    configs_dir = _resolve_configs_dir(repo_root)
    candidate2 = _resolve_survey_specs_dir(configs_dir)
    if candidate2.is_dir():
        return str(candidate2) + os.sep
    return ""  # last resort (will trigger internal fallbacks & warnings)


_REPO_ROOT = _resolve_repo_root()
_CONFIGS_DIR = _resolve_configs_dir(_REPO_ROOT)
_BOLTZ_DIR = _resolve_boltzmann_dir(_REPO_ROOT)
# specs dir assigned later after parsing to allow --specs-dir override
_SPECS_DIR = None

TARGET_FUNCS = [
    # photo_obs hotspots
    "compute_all",
    "compute_kernels",
    "lensing_kernel",
    "galaxy_kernel",
    "integral_efficiency",
    "faster_integral_efficiency",
    "lensing_efficiency",
    "P_limber",
    "sqrtP_limber",
    "computecls_vectorized",
    "clsintegral",
    "genwindow",
    # photo_cov hotspots
    "getcls",
    "compute_derivs",
    "compute_covmat",
]
PHOTO_OBS_FILE = "photo_obs.py"
PHOTO_COV_FILE = "photo_cov.py"


def parse_args():
    p = argparse.ArgumentParser(description="Profile photometric Cl / Fisher performance.")
    p.add_argument(
        "--observables", nargs="+", default=["GCph", "WL"], help="Subset of observables (GCph WL)"
    )
    p.add_argument(
        "--code",
        nargs="+",
        default=["symbolic"],
        choices=["symbolic", "camb", "class"],
        help="Cosmo backends to run",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="Repeats per (code, accuracy) configuration"
    )
    p.add_argument(
        "--accuracy", nargs="+", type=float, default=[1.0], help="Accuracy settings to test"
    )
    p.add_argument("--derivatives", default="3PT", help="Derivative scheme")
    p.add_argument("--output", required=True, help="Path to write JSON summary")
    p.add_argument("--csv", default=None, help="Optional CSV for aggregate per-function stats")
    p.add_argument(
        "--keep-prof", action="store_true", help="Keep individual .prof and .txt pstats dumps"
    )
    p.add_argument("--tag", default=None, help="Optional label to include in output JSON")
    p.add_argument(
        "--freepars-minimal",
        action="store_true",
        help="Use a minimal free parameter set (faster derivatives)",
    )
    p.add_argument(
        "--no-derivs",
        action="store_true",
        help="Skip derivative computation by temporarily clearing free parameters (baseline Cl timing)",
    )
    p.add_argument(
        "--specs-dir",
        default=None,
        help="Explicit path to survey specifications directory (overrides auto-detection)",
    )
    p.add_argument(
        "--fast-eff",
        action="store_true",
        help="Enable fast O(N) lensing efficiency algorithm (sets internal toggle)",
    )
    return p.parse_args()


def build_fisher(observables, code, accuracy, derivatives, freepars_minimal, no_derivs, specs_dir):
    fiducial = {
        "Omegam": 0.32,
        "Omegab": 0.05,
        "h": 0.67,
        "ns": 0.96,
        "sigma8": 0.815584,
        "w0": -1.0,
        "wa": 0.0,
        "mnu": 0.06,
        "Neff": 3.044,
    }
    if freepars_minimal:
        freepars = {"Omegam": 0.01, "h": 0.01}
    else:
        freepars = {"Omegam": 0.01, "Omegab": 0.01, "h": 0.01, "ns": 0.01}
    if no_derivs:
        freepars = {}  # Fisher derivatives skipped

    cosmo_model = "LCDM" if code == "symbolic" else "w0waCDM"
    acc_int = int(round(float(accuracy)))
    options = {
        "accuracy": acc_int,
        "outroot": "PROFILE_",
        "results_dir": "results/",
        "derivatives": ("0PT" if no_derivs else derivatives),
        "feedback": 0,
        "survey_name": "Euclid",
        "specs_dir": specs_dir,
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": cosmo_model,
        "code": code,
    }
    # Add only the YAML relevant to selected code if file exists
    yaml_candidates = {
        "camb": ("camb_config_yaml", _BOLTZ_DIR / "camb" / "default.yaml"),
        "class": ("class_config_yaml", _BOLTZ_DIR / "class" / "fast_photo.yaml"),
        "symbolic": ("symbolic_config_yaml", _BOLTZ_DIR / "symbolic" / "default.yaml"),
    }
    if code in yaml_candidates:
        key, path = yaml_candidates[code]
        if path.is_file():
            options[key] = str(path)
    fm = cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
    )
    return fm


def extract_function_times(stats: pstats.Stats) -> dict:
    out = {f: 0.0 for f in TARGET_FUNCS}
    stats_dict = getattr(stats, "stats", {})
    for (filename, line, funcname), stat in stats_dict.items():
        if funcname not in out:
            continue
        if PHOTO_OBS_FILE in filename or PHOTO_COV_FILE in filename:
            ccalls, ncalls, tt, ct, callers = stat  # per pstats format
            out[funcname] += ct  # cumulative time
    return out


def run_profile(
    observables,
    code,
    accuracy,
    derivatives,
    repeats,
    keep_prof,
    freepars_minimal,
    no_derivs,
    specs_dir,
):
    runs = []
    for r in range(repeats):
        fm = build_fisher(
            observables, code, accuracy, derivatives, freepars_minimal, no_derivs, specs_dir
        )
        # Profile entire compute
        import cProfile

        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
        fm.compute()
        pr.disable()
        t1 = time.time()
        wall = t1 - t0
        # Stats
        stats = pstats.Stats(pr)
        func_times = extract_function_times(stats)
        prof_bin = prof_txt = None
        if keep_prof:
            prefix = f"prof_{code}_acc{accuracy}_rep{r+1}"
            prof_bin = prefix + ".prof"
            prof_txt = prefix + ".txt"
            stats.dump_stats(prof_bin)
            with open(prof_txt, "w"):
                stats.sort_stats("cumulative").print_stats("photo_obs")
                stats.print_stats("photo_cov")
        runs.append(
            {
                "repeat": r + 1,
                "wall_time_sec": wall,
                "function_cum_times_sec": func_times,
                "prof_bin": prof_bin,
                "prof_txt": prof_txt,
            }
        )
    return runs


def aggregate(runs):
    agg = {}
    keys = set()
    for r in runs:
        keys.update(r["function_cum_times_sec"].keys())
    for k in sorted(keys):
        vals = [r["function_cum_times_sec"][k] for r in runs if k in r["function_cum_times_sec"]]
        if not vals:
            continue
        agg[k] = {
            "mean": st.mean(vals),
            "stdev": (st.pstdev(vals) if len(vals) > 1 else 0.0),
            "min": min(vals),
            "max": max(vals),
        }
    return agg


def main():
    args = parse_args()
    # Resolve specs dir (after args so override works)
    global _SPECS_DIR
    _SPECS_DIR = _discover_specs_dir(args.specs_dir)

    # Optionally enable fast efficiency optimization (applied after imports via monkey patch)
    if args.fast_eff:
        try:
            import cosmicfishpie.LSSsurvey.photo_obs as _photo_obs

            _photo_obs._USE_FAST_EFF = True  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            print(f"[warning] Could not enable fast efficiency: {e}")

    results = {
        "tag": args.tag,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "specs_dir": _SPECS_DIR,
            "repo_root": str(_REPO_ROOT),
            "configs_dir": str(_CONFIGS_DIR),
            "boltzmann_dir": str(_BOLTZ_DIR),
            "fast_eff_enabled": args.fast_eff,
        },
        "configs": [],
    }

    for code in args.code:
        for acc in args.accuracy:
            runs = run_profile(
                args.observables,
                code,
                acc,
                args.derivatives,
                args.repeats,
                args.keep_prof,
                args.freepars_minimal,
                args.no_derivs,
                _SPECS_DIR,
            )
            agg = aggregate(runs)
            results["configs"].append(
                {
                    "code": code,
                    "accuracy": acc,
                    "observables": args.observables,
                    "derivatives": ("0PT" if args.no_derivs else args.derivatives),
                    "repeats": args.repeats,
                    "runs": runs,
                    "aggregate_function_cum_times_sec": agg,
                    "aggregate_wall_time_sec": {
                        "mean": st.mean([r["wall_time_sec"] for r in runs]),
                        "stdev": (
                            st.pstdev([r["wall_time_sec"] for r in runs])
                            if args.repeats > 1
                            else 0.0
                        ),
                        "min": min(r["wall_time_sec"] for r in runs),
                        "max": max(r["wall_time_sec"] for r in runs),
                    },
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON summary written to {out_path}")

    if args.csv:
        # Flatten aggregate into CSV per (code,accuracy,function)
        lines = ["code,accuracy,function,mean,stdev,min,max"]
        for cfg_entry in results["configs"]:
            code = cfg_entry["code"]
            acc = cfg_entry["accuracy"]
            for fname, statsd in cfg_entry["aggregate_function_cum_times_sec"].items():
                lines.append(
                    f"{code},{acc},{fname},{statsd['mean']:.6f},{statsd['stdev']:.6f},{statsd['min']:.6f},{statsd['max']:.6f}"
                )
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("\n".join(lines) + "\n")
        print(f"CSV aggregate written to {csv_path}")

    print("\nDone. You can diff this JSON against another branch to measure speedups.")
    print("Focus on aggregate_function_cum_times_sec entries for hotspots.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
