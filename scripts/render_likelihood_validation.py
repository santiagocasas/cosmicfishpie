#!/usr/bin/env python
"""Render compact Fisher-versus-Nautilus likelihood-validation evidence.

The input run directory is an externally stored HPC result with ``wl``,
``gcsp``, and ``joint`` subdirectories produced by
``run_wl_gcsp_nautilus_case.py``. Only the weighted summary statistics and
three figures are written to the repository. Chains and HDF5 checkpoints stay
outside version control.

Example
-------
uv run python scripts/render_likelihood_validation.py \
    --run-dir /scratch/$USER/cfp/wl-gcsp-64cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wl_gcsp_fisher_nautilus_demo as demo

from cosmicfishpie.analysis import fishconsumer as fico
from cosmicfishpie.analysis import fisher_matrix as fm

CASE_LABELS = {
    "wl": "WL",
    "gcsp": "GCsp",
    "joint": "WL+GCsp",
}

PLOT_COLORS = {
    "wl": {"fisher": "#0072B2", "nautilus": "#56B4E9"},
    "gcsp": {"fisher": "#009E73", "nautilus": "#44AA99"},
    "joint": {"fisher": "#D55E00", "nautilus": "#E69F00"},
}


def _local_artifact(case_dir: Path, metadata_path: str) -> Path:
    """Resolve a copied run artifact without retaining its original scratch path."""
    return case_dir / Path(metadata_path).name


def _read_case(run_dir: Path, case: str) -> tuple[dict, pd.DataFrame, object]:
    """Read one copied HPC case and its Fisher reference."""
    case_dir = run_dir / case
    metadata_file = case_dir / "case_metadata.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Missing {case} metadata: {metadata_file}")

    metadata = json.loads(metadata_file.read_text())
    chain_file = _local_artifact(case_dir, metadata["chain"])
    fisher_file = _local_artifact(case_dir, metadata["fisher_reference"])
    if not chain_file.is_file() or not fisher_file.is_file():
        raise FileNotFoundError(f"{case} is missing its copied chain or Fisher reference")
    return metadata, pd.read_csv(chain_file), fm.fisher_matrix(file_name=str(fisher_file))


def _check_compatible(metadata: dict[str, dict]) -> None:
    """Reject a plot assembled from unlike sampling configurations."""
    reference = metadata["wl"]
    keys = ("free_params", "sample_nuisances", "nuisance_sigma", "reference_kind")
    for case, value in metadata.items():
        mismatch = [key for key in keys if value[key] != reference[key]]
        if mismatch:
            raise ValueError(f"{case} does not match WL for {mismatch}")


def _render_case(
    case: str,
    label: str,
    fisher: object,
    chain: pd.DataFrame,
    params: list[str],
    truth_values: dict[str, float],
    reference_kind: str,
    output: Path,
) -> None:
    """Render one two-entry ChainConsumer overlay with a compact legend."""
    colors = PLOT_COLORS[case]
    figure = fico.make_triangle_plot(
        fishers=[fisher],
        chains=[chain],
        fisher_labels=[f"Fisher ({reference_kind})"],
        chain_labels=["Nautilus posterior"],
        colors=[colors["fisher"], colors["nautilus"]],
        params=params,
        truth_values=truth_values,
        shade_fisher=False,
        shade_chains=True,
        shade_alpha=0.25,
        lw_fisher=2.2,
        legend_kwargs={"fontsize": 10, "loc": "upper right", "frameon": True},
        label_font_size=14,
        tick_font_size=11,
        smooth=5,
        bins=16,
        figsize=(6.5, 6.5),
    )
    figure.suptitle(f"Euclid {label}: Fisher forecast vs. Nautilus likelihood", y=1.01, fontsize=15)
    figure.savefig(output, bbox_inches="tight", dpi=220)
    plt.close(figure)
    print(f"Saved: {output}")


def _render_joint_overview(
    cases: dict[str, tuple[dict, pd.DataFrame, object]],
    params: list[str],
    truth_values: dict[str, float],
    reference_kind: str,
    output: Path,
) -> None:
    """Render all three Fisher-Nautilus pairs without an oversized legend."""
    ordered_cases = tuple(CASE_LABELS)
    fishers = [cases[case][2] for case in ordered_cases]
    chains = [cases[case][1] for case in ordered_cases]
    figure = fico.make_triangle_plot(
        fishers=fishers,
        chains=chains,
        fisher_labels=[f"{CASE_LABELS[case]} Fisher" for case in ordered_cases],
        chain_labels=[f"{CASE_LABELS[case]} Nautilus" for case in ordered_cases],
        colors=[
            *(PLOT_COLORS[case]["fisher"] for case in ordered_cases),
            *(PLOT_COLORS[case]["nautilus"] for case in ordered_cases),
        ],
        params=params,
        truth_values=truth_values,
        shade_fisher=False,
        shade_chains=True,
        shade_alpha=0.18,
        lw_fisher=2.0,
        legend_kwargs={"fontsize": 8, "loc": "upper right", "frameon": True},
        label_font_size=13,
        tick_font_size=10,
        smooth=5,
        bins=16,
        figsize=(7.5, 7.5),
    )
    figure.suptitle(
        f"Euclid WL, GCsp, and joint: Fisher ({reference_kind}) vs. Nautilus",
        y=1.01,
        fontsize=14,
    )
    figure.savefig(output, bbox_inches="tight", dpi=220)
    plt.close(figure)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("scripts/likelihood_validation_results"),
        help="Tracked directory for the chain-free statistics JSON.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/source/_static/likelihood_validation"),
        help="Tracked documentation-asset directory for compact figures.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    cases = {case: _read_case(run_dir, case) for case in CASE_LABELS}
    metadata = {case: item[0] for case, item in cases.items()}
    _check_compatible(metadata)

    params = metadata["wl"]["free_params"]
    reference_kind = metadata["wl"]["reference_kind"]
    comparisons = {
        CASE_LABELS[case]: demo._compare_chain_to_fisher(
            cases[case][1], cases[case][2], params, demo.FIDUCIAL
        )
        for case in CASE_LABELS
    }
    statistics_file = demo._report_statistics(
        comparisons,
        params,
        reference_kind,
        args.results_dir,
    )
    stable_statistics_file = args.results_dir / "wl_gcsp_symbolic_euclid_64cpu_statistics.json"
    statistics_file.replace(stable_statistics_file)
    print(f"Saved: {stable_statistics_file}")

    for case, label in CASE_LABELS.items():
        if case == "joint":
            continue
        _render_case(
            case,
            label,
            cases[case][2],
            cases[case][1],
            params,
            demo.FIDUCIAL,
            reference_kind,
            args.figure_dir / f"wl_gcsp_symbolic_euclid_64cpu_{case}.png",
        )

    _render_joint_overview(
        cases,
        params,
        demo.FIDUCIAL,
        reference_kind,
        args.figure_dir / "wl_gcsp_symbolic_euclid_64cpu_joint.png",
    )


if __name__ == "__main__":
    main()
