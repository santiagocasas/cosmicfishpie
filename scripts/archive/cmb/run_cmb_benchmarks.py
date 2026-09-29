#!/usr/bin/env python
# coding: utf-8

"""Run standard CMB experiment benchmarks (Planck / SO / S4).

This is a thin wrapper around scripts/run_cmb_fisher_smoke.py that selects a
survey-spec YAML and runs a CAMB CMB Fisher.

Example
-------
  uv run python scripts/run_cmb_benchmarks.py --outdir tmp/cmb_bench
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CMB benchmark Fisher matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Base output directory (default: scripts/benchmark_results/cmb_bench_<timestamp>)",
    )
    parser.add_argument(
        "--which",
        default="planck,so,s4",
        help="Comma-separated list: planck,so,s4",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Override lmax_CMB for all runs (default: use spec YAML value)",
    )
    parser.add_argument(
        "--observables",
        default="CMB_T,CMB_E",
        help="Comma-separated list from: CMB_T,CMB_E,CMB_B",
    )
    parser.add_argument("--feedback", type=int, default=1)
    parser.add_argument("--accuracy", type=int, default=1)
    parser.add_argument("--derivatives", default="3PT")
    args = parser.parse_args()

    repo = _repo_root()
    run_id = _ts()
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else repo / "scripts" / "benchmark_results" / f"cmb_bench_{run_id}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    spec_dir = repo / "cosmicfishpie" / "configs" / "default_survey_specifications"
    presets = {
        "planck": spec_dir / "Planck.yaml",
        "so": spec_dir / "Simons-Observatory-PlanckLowEll.yaml",
        "s4": spec_dir / "CMB-Stage4-PlanckLowEll.yaml",
    }

    which = [w.strip().lower() for w in args.which.split(",") if w.strip()]
    unknown = [w for w in which if w not in presets]
    if unknown:
        raise SystemExit(f"Unknown --which entries: {unknown} (allowed: {sorted(presets)})")

    smoke = repo / "scripts" / "run_cmb_fisher_smoke.py"
    rc = 0
    for w in which:
        spec = presets[w]
        if not spec.exists():
            raise SystemExit(f"Missing spec YAML: {spec}")
        run_out = outdir / w
        run_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(smoke),
            "--code",
            "camb",
            "--spec-yaml",
            str(spec),
            "--observables",
            args.observables,
            "--feedback",
            str(args.feedback),
            "--accuracy",
            str(args.accuracy),
            "--derivatives",
            str(args.derivatives),
            "--outdir",
            str(run_out),
            "--write-enabled-yaml",
        ]
        if args.lmax is not None:
            cmd += ["--lmax", str(args.lmax)]

        print("[cmb-bench] running:")
        print(" ", " ".join(cmd))
        r = subprocess.call(cmd)
        if r != 0:
            rc = r
            print(f"[cmb-bench] FAILED preset={w} exit={r}")
        else:
            print(f"[cmb-bench] OK preset={w}")

    print("[cmb-bench] outputs:", outdir)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
