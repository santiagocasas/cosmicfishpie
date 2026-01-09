#!/usr/bin/env python
# coding: utf-8

"""Plot pairwise constraint ratios from compare_fishers JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from cosmicfishpie.analysis import fisher_matrix as fm_mod
from cosmicfishpie.analysis import plot_comparison as pc


def _load(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace(":", "_")


def _pair_id(entry: dict) -> str:
    a = Path(entry.get("a", "A")).name
    b = Path(entry.get("b", "B")).name
    return f"{a}__vs__{b}"


def _code_labels(entry: dict) -> tuple[str, str]:
    specs = entry.get("specs_diff", {})
    code_change = specs.get("options", {}).get("changed", {}).get("code", {})
    code_a = code_change.get("a")
    code_b = code_change.get("b")
    if code_a and code_b:
        return f"Code: {code_a}", f"Code: {code_b}"
    return "A", "B"


def _plot_param_differences(entry: dict, outdir: Path) -> None:
    a_path = entry.get("a_matrix")
    b_path = entry.get("b_matrix")
    if not a_path or not b_path:
        return
    if not os.path.isfile(a_path) or not os.path.isfile(b_path):
        return
    fm_a = fm_mod.fisher_matrix(file_name=str(a_path))
    fm_b = fm_mod.fisher_matrix(file_name=str(b_path))
    label_a, label_b = _code_labels(entry)
    fig_title = f"{label_a} vs {label_b}"
    outpath = outdir / f"params_{_safe_name(_pair_id(entry))}.png"
    # plot_comparison treats compare_to_index=0 as False, so use index 1 and swap order
    pc.ploterrs(
        fishers_list=[fm_b, fm_a],
        fishers_name=[label_b, label_a],
        plot_style="original",
        compare_to_index=1,
        outpathfile=str(outpath),
        savefig=True,
        figure_title=fig_title,
        y_label="Discrepancy of constraints (%)",
    )


def _plot_fom(entry: dict, outdir: Path) -> None:
    analysis = entry.get("analysis", {})
    results = analysis.get("results", [])
    if len(results) < 2:
        return
    a_fom = results[0].get("FoM", {}).get("value")
    b_fom = results[1].get("FoM", {}).get("value")
    if a_fom is None and b_fom is None:
        return

    plt.figure(figsize=(4, 4))
    label_a, label_b = _code_labels(entry)
    labels = [label_a, label_b]
    values = [a_fom, b_fom]
    plt.bar(labels, values, color=["#4C78A8", "#F58518"])
    plt.ylabel("FoM")
    plt.title(f"{label_a} vs {label_b}")
    fom_params = results[0].get("FoM", {}).get("parameters")
    if fom_params:
        plt.subplots_adjust(bottom=0.25)
        plt.text(
            0.5,
            -0.18,
            f"FoM params: {', '.join(fom_params)}",
            transform=plt.gca().transAxes,
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    outpath = outdir / f"fom_{_safe_name(_pair_id(entry))}.png"
    plt.savefig(outpath, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot compare_fishers ratios.")
    parser.add_argument("json", help="compare_fishers_*.json")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <json>_plots)",
    )
    parser.add_argument(
        "--fom-params",
        default=None,
        help="Override FoM parameter labels (comma-separated, for plot annotation only)",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")
    outdir = (
        Path(args.outdir)
        if args.outdir
        else json_path.with_suffix("").with_name(json_path.stem + "_plots")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load(json_path)
    entries = data.get("pairwise", [])
    if not entries:
        raise SystemExit("No pairwise entries found in JSON.")

    for entry in entries:
        if args.fom_params:
            parts = [p.strip() for p in args.fom_params.split(",") if p.strip()]
            if parts:
                analysis = entry.get("analysis", {})
                results = analysis.get("results", [])
                if results and "FoM" in results[0]:
                    current = results[0]["FoM"].get("parameters")
                    if current and list(current) != parts:
                        raise SystemExit(
                            f"[plot] FoM parameters mismatch. JSON has {current}, "
                            f"but --fom-params requested {parts}"
                        )
                    results[0]["FoM"]["parameters"] = parts
        _plot_param_differences(entry, outdir)
        _plot_fom(entry, outdir)

    print(f"Wrote plots to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
