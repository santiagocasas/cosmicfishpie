#!/usr/bin/env python
"""Combine completed HPC WL, GCsp, and joint Nautilus cases into plots and statistics.

All three jobs must have written ``<run-dir>/<case>/case_metadata.json``. The
post-processor refuses incompatible case settings rather than combining unlike
priors or dimensionalities.

Run after the sampling jobs, normally through the dependent Slurm job:

    uv run python scripts/postprocess_wl_gcsp_nautilus.py --run-dir /scratch/$USER/cfp/run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wl_gcsp_fisher_nautilus_demo as demo

from cosmicfishpie.analysis import fishconsumer as fico
from cosmicfishpie.analysis import fisher_matrix as fm


def _read_case(run_dir: Path, label: str) -> tuple[dict, pd.DataFrame, object]:
    case_dir = run_dir / label
    metadata_path = case_dir / "case_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing completed {label} case metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    chain_path = Path(metadata["chain"])
    reference_path = Path(metadata["fisher_reference"])
    if not chain_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(f"{label} metadata points to missing chain or Fisher file")
    return metadata, pd.read_csv(chain_path), fm.fisher_matrix(file_name=str(reference_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    wl_meta, chain_wl, fisher_wl = _read_case(run_dir, "wl")
    gcsp_meta, chain_gcsp, fisher_gcsp = _read_case(run_dir, "gcsp")
    joint_meta, chain_joint, fisher_joint = _read_case(run_dir, "joint")
    metadata = {"WL": wl_meta, "GCsp": gcsp_meta, "WL+GCsp": joint_meta}

    first = wl_meta
    keys = ("free_params", "sample_nuisances", "nuisance_sigma", "reference_kind")
    for label, value in metadata.items():
        mismatch = [key for key in keys if value[key] != first[key]]
        if mismatch:
            raise ValueError(f"{label} is incompatible with WL for {mismatch}")

    params = first["free_params"]
    reference_kind = first["reference_kind"]
    print(f"Post-processing {run_dir}; Fisher reference: {reference_kind}")
    demo._report_statistics(
        {
            "WL": demo._compare_chain_to_fisher(chain_wl, fisher_wl, params, demo.FIDUCIAL),
            "GCsp": demo._compare_chain_to_fisher(chain_gcsp, fisher_gcsp, params, demo.FIDUCIAL),
            "WL+GCsp": demo._compare_chain_to_fisher(
                chain_joint, fisher_joint, params, demo.FIDUCIAL
            ),
        },
        params,
        reference_kind,
        run_dir,
    )

    fico.make_triangle_plot(
        chains=[chain_wl, chain_gcsp, chain_joint],
        chain_labels=["WL (Nautilus)", "GCsp (Nautilus)", "WL+GCsp (Nautilus)"],
        colors=["blue", "green", "red"],
        params=params,
        truth_values=demo.FIDUCIAL,
        smooth=5,
        bins=12,
        save_plot=True,
        savepath=str(run_dir) + "/",
        plot_filename="nautilus_wl_gcsp_combined",
        file_format=".png",
    )
    fico.make_triangle_plot(
        fishers=[fisher_wl, fisher_gcsp, fisher_joint],
        chains=[chain_wl, chain_gcsp, chain_joint],
        fisher_labels=[
            f"WL (Fisher, {reference_kind})",
            f"GCsp (Fisher, {reference_kind})",
            f"WL+GCsp (Fisher, {reference_kind})",
        ],
        chain_labels=["WL (Nautilus)", "GCsp (Nautilus)", "WL+GCsp (Nautilus)"],
        colors=["blue", "green", "red", "cyan", "lime", "orange"],
        params=params,
        truth_values=demo.FIDUCIAL,
        smooth=5,
        bins=12,
        save_plot=True,
        savepath=str(run_dir) + "/",
        plot_filename="fisher_vs_nautilus_wl_gcsp",
        file_format=".png",
    )
    print(f"Saved plots and chain_statistics.json under: {run_dir}")


if __name__ == "__main__":
    main()
