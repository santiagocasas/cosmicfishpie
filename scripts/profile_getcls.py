#!/usr/bin/env python
"""Profile getcls / Fisher computation performance.

Usage examples:
  python scripts/profile_getcls.py --observables GCph WL --code symbolic --repeats 3 --profile
  python scripts/profile_getcls.py --observables GCph WL --code symbolic --dump-json results/getcls_profile.json

Flags:
  --observables  One or more of GCph WL (order matters for combinations)
  --code         Backend code: symbolic|camb|class (default symbolic)
  --repeats      Number of repeated timing runs (default 1; first run warms caches)
  --profile      Enable cProfile output (writes .prof + .txt next to JSON if requested)
  --dump-json    Path to write JSON summary (optional)
  --derivatives  Derivative scheme (default 3PT)
  --accuracy     Accuracy setting (default 1)
  --no-wl        Exclude WL even if passed (quick toggle)
  --no-gc        Exclude GCph even if passed

The script does not modify library code; it measures wall-clock times and (optionally) cProfile data.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import sys
import time
from pathlib import Path
from pathlib import Path as _P

from cosmicfishpie.configs import config as cfg
from cosmicfishpie.fishermatrix import cosmicfish


def _discover_specs_dir():
    # Try attribute patterns in config
    for cand in [
        getattr(cfg, "specs_dir", None),
        getattr(cfg, "SPECSDIR", None),
    ]:
        if isinstance(cand, str) and cand and os.path.isdir(cand):
            return cand
    # Fallback: relative to installed package
    pkg_root = _P(cfg.__file__).resolve().parent
    candidate = pkg_root / "configs" / "default_survey_specifications"
    if candidate.is_dir():
        return str(candidate) + os.sep
    return ""


_SPECS_DIR = _discover_specs_dir()


def parse_args():
    p = argparse.ArgumentParser(description="Profile getcls performance")
    p.add_argument("--observables", nargs="+", default=["GCph", "WL"])
    p.add_argument("--code", default="symbolic", choices=["symbolic", "camb", "class"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--dump-json", default=None)
    p.add_argument("--derivatives", default="3PT")
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--no-wl", action="store_true")
    p.add_argument("--no-gc", action="store_true")
    return p.parse_args()


def build_fisher(observables, code, accuracy, derivatives):
    # Fiducial cosmology (mirrors benchmark script)
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
    # Keep a minimal free set to reduce derivative overhead when focusing on getcls
    freepars = {"Omegam": 0.01, "h": 0.01}
    cosmo_model = "LCDM" if code == "symbolic" else "w0waCDM"
    try:
        acc_int = int(accuracy)
    except Exception:
        acc_int = 1
    options = {
        "accuracy": acc_int,
        "outroot": "PROFILE_",
        "results_dir": "results/",
        "derivatives": derivatives,
        "feedback": 0,
        "survey_name": "Euclid",
        "specs_dir": _SPECS_DIR,
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": cosmo_model,
        "code": code,
    }
    fm = cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
    )
    return fm


def main():
    args = parse_args()

    obs = []
    for o in args.observables:
        if o == "WL" and args.no_wl:
            continue
        if o == "GCph" and args.no_gc:
            continue
        obs.append(o)
    if not obs:
        print("No observables selected after filters", file=sys.stderr)
        return 2

    # Warm-up + repeats
    durations = []
    profile_files = []
    for run in range(args.repeats):
        fm = build_fisher(obs, args.code, args.accuracy, args.derivatives)
        pr = None
        if args.profile:
            import cProfile
            import pstats

            pr = cProfile.Profile()
            pr.enable()
        t0 = time.time()
        fm.compute()
        t1 = time.time()
        d = t1 - t0
        durations.append(d)
        print(f"Run {run+1}/{args.repeats}: {d:.3f} s")
        if args.profile and pr is not None:
            pr.disable()
            prefix = f"profile_getcls_run{run+1}"
            prof_bin = prefix + ".prof"
            prof_txt = prefix + ".txt"
            pr.dump_stats(prof_bin)
            with open(prof_txt, "w") as fh:
                ps = pstats.Stats(pr, stream=fh).sort_stats("cumulative")
                ps.print_stats("photo_obs")
                ps.print_stats("photo_cov")
                fh.write("\nTop 25 cumulative (overall)\n")
                ps.print_stats(25)
            profile_files.append((prof_bin, prof_txt))
            print(f"  Profile written: {prof_bin}, {prof_txt}")

    summary = {
        "observables": obs,
        "code": args.code,
        "repeats": args.repeats,
        "durations_sec": durations,
        "stats": {
            "mean": st.mean(durations),
            "stdev": st.pstdev(durations) if len(durations) > 1 else 0.0,
            "min": min(durations),
            "max": max(durations),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "derivatives": args.derivatives,
            "accuracy": args.accuracy,
            "specs_dir": _SPECS_DIR,
        },
        "profile_files": profile_files,
    }

    if args.dump_json:
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary written to {args.dump_json}")
    else:
        print("\nSummary:")
        print(json.dumps(summary, indent=2))

    print("\nTips:")
    print(
        "  Line profiling (requires line_profiler): kernprof -l scripts/profile_getcls.py --observables GCph WL --code symbolic"
    )
    print("  Then view: python -m line_profiler profile_getcls.py.lprof | less")
    print(
        "  Memory profiling (requires memory_profiler): python -m memory_profiler scripts/profile_getcls.py --observables GCph WL"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
