#!/usr/bin/env python
# coding: utf-8

"""Add a 1D Gaussian prior Fisher to a Fisher matrix.

This script builds a single-parameter Fisher matrix from
`sigma(param)=prior_sigma`, adds it to an input Fisher using
`cosmicfishpie.analysis.fisher_matrix.fisher_matrix.__add__`, and optionally
prints a comparison table against a reference Fisher (e.g. Planck covmat
converted to Gaussian Fisher).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cosmicfishpie.analysis import fisher_matrix as fm_mod


def _resolve_fisher(path_arg: str) -> Path:
    p = Path(path_arg).expanduser().resolve()
    if p.is_file():
        return p
    p_txt = Path(str(p) + ".txt")
    if p_txt.is_file():
        return p_txt
    raise FileNotFoundError(f"Fisher file not found: {p}")


def _overlap_params(a: fm_mod.fisher_matrix, b: fm_mod.fisher_matrix) -> list[str]:
    a_names = [str(x) for x in a.get_param_names()]
    b_set = {str(x) for x in b.get_param_names()}
    return [x for x in a_names if x in b_set]


def _sigma_map(fish: fm_mod.fisher_matrix) -> dict[str, float]:
    names = [str(x) for x in fish.get_param_names()]
    sig = [float(x) for x in fish.get_confidence_bounds()]
    return dict(zip(names, sig))


def _print_compare_table(
    base: fm_mod.fisher_matrix,
    with_prior: fm_mod.fisher_matrix,
    ref: fm_mod.fisher_matrix,
    params: list[str],
) -> None:
    s_base = _sigma_map(base)
    s_prior = _sigma_map(with_prior)
    s_ref = _sigma_map(ref)

    header = (
        f"{'param':<10} {'sigma_base':>12} {'sigma+prior':>12} {'sigma_ref':>12} "
        f"{'(base/ref)':>10} {'(+prior/ref)':>12}"
    )
    print(header)
    print("-" * len(header))
    for p in params:
        sb = s_base[p]
        sp = s_prior[p]
        sr = s_ref[p]
        print(f"{p:<10} {sb:12.6e} {sp:12.6e} {sr:12.6e} {(sb / sr):10.4f} {(sp / sr):12.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add 1D Gaussian prior Fisher to a Fisher matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fisher", help="Input Fisher matrix (.txt or stem)")
    parser.add_argument(
        "--param",
        default="ombh2",
        help="Parameter name to place the scalar prior on",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0e-4,
        help="1-sigma Gaussian prior width",
    )
    parser.add_argument(
        "--prior-name",
        default="bbn_prior",
        help="Name tag used for the scalar prior Fisher",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: same directory as input Fisher)",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Optional reference Fisher (.txt or stem), e.g. Planck covmat Gaussian",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Optional comma-separated params for comparison table",
    )
    args = parser.parse_args()

    fisher_path = _resolve_fisher(args.fisher)
    base = fm_mod.fisher_matrix(file_name=str(fisher_path))

    p = str(args.param)
    if p not in [str(x) for x in base.get_param_names()]:
        raise SystemExit(f"Parameter '{p}' is not present in input Fisher")

    fid = float(base.get_fiducial(p))
    inv_var = 1.0 / float(args.sigma) ** 2

    prior = fm_mod.fisher_matrix(
        fisher_matrix=np.array([[inv_var]], dtype=float),
        param_names=[p],
        param_names_latex=[base.get_param_name_latex(p)],
        fiducial=np.array([fid], dtype=float),
        name=str(args.prior_name),
    )

    combined = base + prior
    combined.name = f"{base.name}_{args.prior_name}_{p}_sigma{args.sigma:.1e}"

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else fisher_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    prior_root = outdir / f"prior_{args.prior_name}_{p}"
    combined_root = outdir / f"{combined.name}"
    prior.save_to_file(str(prior_root))
    combined.save_to_file(str(combined_root))

    print(f"[prior-add] input Fisher:  {fisher_path}")
    print(f"[prior-add] prior Fisher:  {prior_root}.txt")
    print(f"[prior-add] output Fisher: {combined_root}.txt")

    payload: dict[str, object] = {
        "input_fisher": str(fisher_path),
        "prior_fisher": str(prior_root) + ".txt",
        "output_fisher": str(combined_root) + ".txt",
        "param": p,
        "sigma": float(args.sigma),
        "inv_variance": float(inv_var),
        "fiducial": fid,
    }

    if args.compare_to:
        ref_path = _resolve_fisher(args.compare_to)
        ref = fm_mod.fisher_matrix(file_name=str(ref_path))
        if args.params:
            params = [x.strip() for x in args.params.split(",") if x.strip()]
        else:
            params = _overlap_params(combined, ref)
        if not params:
            raise SystemExit("No overlapping parameters for comparison")

        print("\n[prior-add] Comparison versus reference:")
        _print_compare_table(base, combined, ref, params)
        payload["reference_fisher"] = str(ref_path)
        payload["comparison_params"] = params

    report_path = outdir / "prior_add_report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[prior-add] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
