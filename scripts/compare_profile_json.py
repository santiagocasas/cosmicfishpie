#!/usr/bin/env python
"""Compare two profile JSON outputs produced by profile_photo_obs.py.

Usage:
  python scripts/compare_profile_json.py --base results/test_profile_fixed.json \
      --new results/test_profile_fast_eff.json

Outputs a markdown-style table plus a JSON diff summary with speedups.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Compare two profiling JSON summaries")
    p.add_argument("--base", required=True, help="Baseline JSON (before optimization)")
    p.add_argument("--new", required=True, help="New JSON (after optimization)")
    p.add_argument(
        "--out-json", default=None, help="Optional path to write machine-readable diff JSON"
    )
    p.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.05,
        help="Highlight improvements over this speedup ratio",
    )
    return p.parse_args()


def load_one(path: str):
    with open(path) as f:
        data = json.load(f)
    # Assume single config for simplicity; extendable if multiple
    cfg = data["configs"][0]
    runs = cfg["aggregate_function_cum_times_sec"]
    wall = cfg["aggregate_wall_time_sec"]["mean"]
    return {
        "path": path,
        "wall": wall,
        "funcs": {k: v["mean"] for k, v in runs.items()},
        "code": cfg["code"],
        "accuracy": cfg["accuracy"],
        "observables": cfg["observables"],
    }


def compare(base, new):
    all_funcs = sorted(set(base["funcs"]) | set(new["funcs"]))
    rows = []
    diff_summary = {
        "baseline_wall": base["wall"],
        "new_wall": new["wall"],
        "wall_speedup": base["wall"] / new["wall"] if new["wall"] > 0 else math.inf,
        "functions": [],
    }
    for fn in all_funcs:
        b = base["funcs"].get(fn)
        n = new["funcs"].get(fn)
        if b is None or n is None:
            speed = None
        else:
            speed = (b / n) if n > 0 else math.inf
        rows.append((fn, b, n, speed))
        diff_summary["functions"].append(
            {"function": fn, "baseline_sec": b, "new_sec": n, "speedup": speed}
        )
    return rows, diff_summary


def format_table(rows, ratio_threshold):
    lines = []
    header = "| Function | Baseline (s) | New (s) | Speedup |"
    sep = "|---|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for fn, b, n, speed in rows:
        if b is None or n is None:
            lines.append(
                f"| {fn} | {b if b is not None else '-'} | {n if n is not None else '-'} | - |"
            )
            continue
        flag = "" if (speed is None or speed < ratio_threshold) else "✅"
        lines.append(f"| {fn} | {b:.3f} | {n:.3f} | {speed:.2f}x {flag} |")
    return "\n".join(lines)


def main():
    args = parse_args()
    base = load_one(args.base)
    new = load_one(args.new)

    if base["code"] != new["code"]:
        print(f"[warn] Different codes: {base['code']} vs {new['code']}")
    if base["accuracy"] != new["accuracy"]:
        print(f"[warn] Different accuracy: {base['accuracy']} vs {new['accuracy']}")
    if base["observables"] != new["observables"]:
        print(f"[warn] Different observables: {base['observables']} vs {new['observables']}")

    rows, diff_summary = compare(base, new)
    print("\nOverall wall time:")
    print(f"  Baseline: {base['wall']:.3f} s")
    print(f"  New:      {new['wall']:.3f} s")
    print(f"  Speedup:  {diff_summary['wall_speedup']:.2f}x")

    print("\nPer-function cumulative time (cProfile cumulative):")
    print(format_table(rows, args.ratio_threshold))

    improved = [
        d
        for d in diff_summary["functions"]
        if d["speedup"] and d["speedup"] >= args.ratio_threshold
    ]
    top = sorted(
        [d for d in diff_summary["functions"] if d["speedup"]],
        key=lambda x: x["speedup"],
        reverse=True,
    )[:5]

    print("\nFunctions exceeding speedup threshold (>= {:.2f}x):".format(args.ratio_threshold))
    for d in improved:
        print(f"  {d['function']}: {d['speedup']:.2f}x")

    print("\nTop 5 speedups:")
    for d in top:
        print(
            f"  {d['function']}: {d['baseline_sec']:.3f} -> {d['new_sec']:.3f} ({d['speedup']:.2f}x)"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(diff_summary, f, indent=2)
        print(f"\nDiff JSON written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
