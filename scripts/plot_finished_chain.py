#!/usr/bin/env python
"""Render weighted contour plots from one or more completed Nautilus chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cosmicfishpie.analysis.fishconsumer import make_triangle_plot


PREFERRED_COSMOLOGY_PARAMS = ("Omegam", "Omegab", "h", "ns", "sigma8", "w0", "wa")
NON_SAMPLE_COLUMNS = {"weight", "weights", "loglike", "log_posterior", "posterior"}
COLORS = (
    "#3a86ff",
    "#fb5607",
    "#8338ec",
    "#ffbe0b",
    "#d11149",
    "#2a9d8f",
    "#6a994e",
    "#4361ee",
    "#f72585",
    "#8d6e63",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata",
        nargs="+",
        type=Path,
        help="One or more completed Nautilus *_metadata.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path; defaults beside the first chain text file",
    )
    parser.add_argument(
        "--params",
        help="Comma-separated parameter names. Default: cosmological parameters shared by all chains.",
    )
    parser.add_argument(
        "--all-params",
        action="store_true",
        help="Plot every sampled parameter shared by all chains.",
    )
    parser.add_argument("--bins", type=int, default=12, help="Histogram bins per axis")
    parser.add_argument("--smooth", type=int, default=5, help="Contour smoothing factor")
    parser.add_argument("--dpi", type=int, default=250, help="PNG resolution")
    parser.add_argument(
        "--label",
        action="append",
        help="Legend label in metadata-file order; repeat once per chain",
    )
    parser.add_argument(
        "--no-truths",
        action="store_true",
        help="Do not draw fiducial values from the first metadata file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without rendering")
    return parser.parse_args()


def load_metadata(path: Path) -> tuple[Path, dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    if not path.name.endswith("_metadata.json"):
        raise ValueError(f"Expected a *_metadata.json file, got: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata.get("chain_file"), str):
        raise ValueError(f"Metadata has no usable chain_file: {path}")
    return path, metadata


def load_chain(metadata_path: Path, metadata: dict) -> tuple[Path, pd.DataFrame, list[str]]:
    chain_path = Path(metadata["chain_file"]).expanduser()
    if not chain_path.is_absolute():
        chain_path = metadata_path.parent / chain_path
    chain_path = chain_path.resolve()
    if not chain_path.is_file():
        raise FileNotFoundError(f"Chain file from metadata does not exist: {chain_path}")

    with chain_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
    if not header.startswith("#"):
        raise ValueError(f"Expected a commented column header in: {chain_path}")
    columns = header.lstrip("# ").split()
    if not columns:
        raise ValueError(f"No columns found in chain header: {chain_path}")

    samples = np.atleast_2d(np.loadtxt(chain_path))
    if samples.shape[1] != len(columns):
        raise ValueError(
            f"Chain has {samples.shape[1]} columns but header declares {len(columns)}: {chain_path}"
        )
    chain = pd.DataFrame(samples, columns=columns)
    if "weights" in chain.columns:
        chain = chain.rename(columns={"weights": "weight"})
    if "weight" not in chain.columns:
        raise ValueError(f"Expected weights or weight column in: {chain_path}")

    sample_columns = [column for column in chain.columns if column not in NON_SAMPLE_COLUMNS]
    finite = np.isfinite(chain[sample_columns + ["weight"]]).all(axis=1)
    chain = chain.loc[finite & (chain["weight"] > 0)].copy()
    if chain.empty:
        raise ValueError(f"No finite positive-weight samples in: {chain_path}")
    return chain_path, chain, sample_columns


def select_parameters(args: argparse.Namespace, available: list[list[str]]) -> list[str]:
    if args.params and args.all_params:
        raise ValueError("Use either --params or --all-params, not both")

    shared = [name for name in available[0] if all(name in names for names in available[1:])]
    if args.params:
        params = [name.strip() for name in args.params.split(",") if name.strip()]
    elif args.all_params:
        params = shared
    else:
        params = [name for name in PREFERRED_COSMOLOGY_PARAMS if name in shared]
        if not params:
            params = shared

    missing_by_chain = {
        index + 1: [name for name in params if name not in names]
        for index, names in enumerate(available)
        if any(name not in names for name in params)
    }
    if missing_by_chain:
        raise ValueError(f"Requested parameters are not available in every chain: {missing_by_chain}")
    if not params:
        raise ValueError("The chains have no shared parameters to plot")
    return params


def output_path(args: argparse.Namespace, chain_paths: list[Path]) -> Path:
    if args.output:
        return args.output.expanduser().resolve()
    if len(chain_paths) == 1:
        name = f"{chain_paths[0].stem}_triangle.png"
    else:
        name = f"overplot_{len(chain_paths)}_chains_triangle.png"
    return chain_paths[0].with_name(name).resolve()


def main() -> int:
    args = parse_args()
    loaded = []
    for requested_path in args.metadata:
        metadata_path, metadata = load_metadata(requested_path)
        chain_path, chain, available = load_chain(metadata_path, metadata)
        loaded.append((metadata_path, metadata, chain_path, chain, available))

    params = select_parameters(args, [item[4] for item in loaded])
    chains = [item[3][params + ["weight"]] for item in loaded]
    chain_paths = [item[2] for item in loaded]
    output = output_path(args, chain_paths)

    if args.label and len(args.label) != len(loaded):
        raise ValueError(
            f"Received {len(args.label)} --label values for {len(loaded)} chains; provide one per chain"
        )
    labels = args.label or [item[1].get("name") or item[2].stem for item in loaded]

    truth_values = None
    if not args.no_truths:
        fiducial = loaded[0][1].get("sampled_fiducial_params", {})
        truth_values = {name: float(fiducial[name]) for name in params if name in fiducial}

    parameter_list = ", ".join(params)
    for index, (metadata_path, _, chain_path, chain, _) in enumerate(loaded, start=1):
        print(f"Chain {index}: {labels[index - 1]}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Samples: {chain_path} ({len(chain)} finite positive-weight rows)")
    print(f"Parameters: {parameter_list}")
    print(f"Output: {output}")
    if truth_values:
        print(f"Truth markers: first metadata file ({loaded[0][0]})")
    if args.dry_run:
        print("Dry run: plot not rendered")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_triangle_plot(
        chains=chains,
        chain_labels=labels,
        colors=[COLORS[index % len(COLORS)] for index in range(len(chains))],
        params=params,
        truth_values=truth_values,
        shade_chains=True,
        smooth=args.smooth,
        bins=args.bins,
        figsize=(2.5 * len(params), 2.5 * len(params)),
        savefile=output,
        save_dpi=args.dpi,
    )
    plt.close(figure)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
