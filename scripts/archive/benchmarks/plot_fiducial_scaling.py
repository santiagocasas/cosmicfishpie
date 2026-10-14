#!/usr/bin/env python3
"""Plot fiducial-construction scaling summaries from benchmark log files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BACKEND_RE = re.compile(r"^Backend: (\S+)$", re.MULTILINE)
SUMMARY_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)-(\d+\.\d+)\s+(\d+\.\d+)\s*$",
    re.MULTILINE,
)
BACKEND_ORDER = ("camb", "class", "symbolic")
COLORS = {"camb": "#0072B2", "class": "#D55E00", "symbolic": "#009E73"}
MARKERS = {"camb": "o", "class": "s", "symbolic": "^"}


def parse_log(path: Path) -> tuple[str, np.ndarray]:
    """Return the backend name and summary rows from one benchmark log."""
    text = path.read_text(encoding="utf-8")
    backend_match = BACKEND_RE.search(text)
    rows = SUMMARY_ROW_RE.findall(text)
    if backend_match is None or not rows:
        raise ValueError(f"Could not parse a benchmark summary from {path}")
    return backend_match.group(1), np.asarray(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Benchmark scheduler output files")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    args = parser.parse_args()

    datasets = dict(parse_log(path) for path in args.logs)
    missing = set(BACKEND_ORDER) - datasets.keys()
    if missing:
        raise ValueError(f"Missing benchmark logs for: {', '.join(sorted(missing))}")

    fig, (time_ax, speedup_ax) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for backend in BACKEND_ORDER:
        rows = datasets[backend]
        threads, median, minimum, maximum, speedup = rows.T
        style = {
            "color": COLORS[backend],
            "marker": MARKERS[backend],
            "linewidth": 2,
            "markersize": 6,
            "label": backend.upper() if backend != "class" else "CLASS",
        }
        time_ax.errorbar(
            threads,
            median,
            yerr=np.vstack((median - minimum, maximum - median)),
            capsize=3,
            **style,
        )
        speedup_ax.plot(threads, speedup, **style)

    threads = datasets["camb"][:, 0]
    speedup_ax.plot(threads, threads, color="0.55", linestyle="--", label="Ideal")

    time_ax.set(
        xlabel="OpenMP threads",
        ylabel="Median fiducial-construction time [s]",
        yscale="log",
        xticks=threads,
    )
    speedup_ax.set(
        xlabel="OpenMP threads",
        ylabel="Speedup relative to one thread",
        xticks=threads,
    )
    for axis in (time_ax, speedup_ax):
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle("CosmicFishPie fiducial-cosmology construction scaling")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
