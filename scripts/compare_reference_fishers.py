#!/usr/bin/env python3
"""Compare marginalized 1-sigma errors between two raw Fisher matrix files.

Thin CLI wrapper around cosmicfishpie's own
:class:`cosmicfishpie.analysis.fisher_matrix.fisher_matrix` for comparing
external reference Fisher matrices (e.g. the
``CosmicFish_v1.0_*_fishermatrix.txt`` files shipped in a companion results
repository such as ``Euclid_KP_nu``) against each other, or against this
repository's own validation outputs. Loading, ``.paramnames`` parsing, and
Fisher-matrix inversion/marginalization are all delegated to
``fisher_matrix`` (``file_name=...`` constructor + ``get_confidence_bounds``)
rather than reimplemented here.

Usage
-----
    uv run python scripts/compare_reference_fishers.py \\
        --fisher-a /path/to/..._camb-Optimistic-3PT_WLGCph_fishermatrix.txt \\
        --fisher-b /path/to/..._class-Optimistic-3PT_WLGCph_fishermatrix.txt \\
        --params Omegam,Omegab,h,ns,sigma8,mnu,Neff \\
        --label-a camb --label-b class

If --params is omitted, the intersection of both paramnames files is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cosmicfishpie.analysis.fisher_matrix import fisher_matrix


def marginalized_sigmas(fisher_path: Path) -> tuple[list[str], dict[str, float]]:
    """Load a Fisher matrix (+ its .paramnames sidecar) and return marginalized 1-sigma errors."""
    fish = fisher_matrix(file_name=str(fisher_path))
    names = fish.get_param_names()
    bounds = fish.get_confidence_bounds(confidence_level=0.6827)
    sigmas = {name: float(bounds[i]) for i, name in enumerate(names)}
    return names, sigmas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fisher-a", required=True, type=Path)
    parser.add_argument("--fisher-b", required=True, type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument(
        "--params", default=None, help="Comma-separated parameter subset (default: intersection)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional max %% deviation gate; nonzero exit if exceeded",
    )
    args = parser.parse_args(argv)

    names_a, sig_a = marginalized_sigmas(args.fisher_a)
    _, sig_b = marginalized_sigmas(args.fisher_b)

    if args.params:
        params = [p.strip() for p in args.params.split(",")]
    else:
        params = [p for p in names_a if p in sig_b]

    missing = [p for p in params if p not in sig_a or p not in sig_b]
    if missing:
        print(f"WARNING: parameters missing from one side, skipped: {missing}", file=sys.stderr)
        params = [p for p in params if p not in missing]

    print(f"A ({args.label_a}): {args.fisher_a}")
    print(f"B ({args.label_b}): {args.fisher_b}")
    print(f"{'param':10s} {'sigma_' + args.label_a:>16s} {'sigma_' + args.label_b:>16s} {'dev%':>8s}")
    worst_param, worst_dev = None, 0.0
    for p in params:
        dev = abs(sig_b[p] / sig_a[p] - 1.0) * 100.0
        print(f"{p:10s} {sig_a[p]:16.6g} {sig_b[p]:16.6g} {dev:8.2f}")
        if dev > worst_dev:
            worst_param, worst_dev = p, dev

    print("-" * 60)
    print(f"worst deviation: {worst_dev:.2f}% (param={worst_param})")

    if args.threshold is not None and worst_dev > args.threshold:
        print(f"THRESHOLD EXCEEDED: {worst_dev:.2f}% > {args.threshold:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
