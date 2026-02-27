#!/usr/bin/env python
# coding: utf-8

"""Compare Planck covmat Gaussian contours against a CosmicFish Fisher.

This script:
- reads a Planck GetDist `.covmat` file,
- extracts a selected parameter subspace,
- converts that covariance to a Fisher matrix,
- overlays triangle contours against an input CosmicFish Fisher matrix.

It is intended to validate the *direction* of constraints (parameter
correlations), especially in the theta-based primary basis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cosmicfishpie.analysis import fishconsumer as fco
from cosmicfishpie.analysis import fisher_matrix as fm_mod

DEFAULT_PLANCK_REL = Path(
    "Planck-Results/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE"
)
DEFAULT_CHAIN_ROOT = "base_plikHM_TTTEEE_lowl_lowE"

CFP_TO_PLANCK = {
    "ombh2": "omegabh2",
    "omch2": "omegach2",
    "logAs": "logA",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_likestats_bestfit(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"likestats file not found: {path}")
    out: dict[str, float] = {}
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("parameter"):
            in_table = True
            continue
        if not in_table:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def _parse_paramnames(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"paramnames file not found: {path}")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
        elif len(parts) == 1:
            out[parts[0]] = parts[0]
    return out


def _read_covmat(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"covmat file not found: {path}")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"covmat file is empty: {path}")
    header = lines[0].strip()
    if not header.startswith("#"):
        raise ValueError(f"covmat header should start with '#': {path}")
    names = header.lstrip("#").strip().split()
    cov = np.loadtxt(path)
    cov = np.atleast_2d(cov).astype(float)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covmat is not square: shape={cov.shape}")
    if cov.shape[0] != len(names):
        raise ValueError(
            f"covmat size/header mismatch: n_header={len(names)} n_matrix={cov.shape[0]} ({path})"
        )
    return names, cov


def _to_planck_name(param: str) -> str:
    return CFP_TO_PLANCK.get(param, param)


def _subset_covariance(
    all_names: list[str], cov: np.ndarray, params_cfp: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    params_planck = [_to_planck_name(p) for p in params_cfp]
    index = {name: i for i, name in enumerate(all_names)}
    missing = [p for p in params_planck if p not in index]
    if missing:
        raise ValueError(f"Requested params not found in covmat header: {missing}")
    idx = [index[p] for p in params_planck]
    cov_sub = cov[np.ix_(idx, idx)]
    return cov_sub, params_cfp, params_planck


def _subset_fisher(fisher: fm_mod.fisher_matrix, params: list[str]) -> fm_mod.fisher_matrix:
    names = list(fisher.get_param_names())
    idx_map = {str(name): i for i, name in enumerate(names)}
    missing = [p for p in params if p not in idx_map]
    if missing:
        raise ValueError(f"Input Fisher missing requested parameters: {missing}")
    idx = [idx_map[p] for p in params]
    mat = np.asarray(fisher.get_fisher_matrix(), dtype=float)
    fid = np.asarray(fisher.get_param_fiducial(), dtype=float)
    latex = list(fisher.get_param_names_latex())
    sub = mat[np.ix_(idx, idx)]
    sub_fid = fid[idx]
    sub_latex = [latex[i] for i in idx]
    return fm_mod.fisher_matrix(
        fisher_matrix=sub,
        param_names=params,
        param_names_latex=sub_latex,
        fiducial=sub_fid,
        name=f"{fisher.name}_subset",
    )


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    denom = np.outer(d, d)
    corr = cov / denom
    return corr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Planck covmat Gaussian vs CosmicFish Fisher contours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fisher", help="Path to CosmicFish Fisher .txt")
    parser.add_argument(
        "--planck-dir",
        default=None,
        help="Directory containing Planck chain products for one data combo",
    )
    parser.add_argument("--chain-root", default=DEFAULT_CHAIN_ROOT)
    parser.add_argument(
        "--params",
        default="ombh2,omch2,theta,tau,logAs,ns",
        help="Comma-separated CFP parameter names for comparison",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <fisher_dir>/planck_covmat_compare)",
    )
    args = parser.parse_args()

    fisher_path = Path(args.fisher).expanduser().resolve()
    if not fisher_path.exists():
        raise SystemExit(f"Fisher file not found: {fisher_path}")

    repo = _repo_root()
    planck_dir = (
        Path(args.planck_dir).expanduser().resolve()
        if args.planck_dir
        else repo / DEFAULT_PLANCK_REL
    )
    covmat_path = planck_dir / "dist" / f"{args.chain_root}.covmat"
    paramnames_path = planck_dir / f"{args.chain_root}.paramnames"
    likestats_path = planck_dir / "dist" / f"{args.chain_root}.likestats"

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else fisher_path.parent / "planck_covmat_compare"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    if len(params) < 2:
        raise SystemExit("Need at least two parameters in --params")

    cov_names, cov = _read_covmat(covmat_path)
    cov_sub, params_cfp, params_planck = _subset_covariance(cov_names, cov, params)
    fisher_sub = np.linalg.pinv(cov_sub)

    latex_map_planck = _parse_paramnames(paramnames_path)
    bestfit_planck = _parse_likestats_bestfit(likestats_path)

    fid = []
    latex = []
    for cfp_name, pl_name in zip(params_cfp, params_planck):
        fid.append(float(bestfit_planck.get(pl_name, 0.0)))
        latex_val = latex_map_planck.get(pl_name, pl_name)
        if cfp_name == "logAs" and pl_name == "logA":
            latex_val = r"{\rm{ln}}(10^{10} A_s)"
        latex.append(latex_val)

    cov_fisher_obj = fm_mod.fisher_matrix(
        fisher_matrix=fisher_sub,
        param_names=params_cfp,
        param_names_latex=latex,
        fiducial=np.asarray(fid, dtype=float),
        name="Planck_covmat_Gaussian",
    )

    cov_fisher_root = outdir / "Planck_covmat_Gaussian"
    cov_fisher_obj.save_to_file(str(cov_fisher_root))

    cfp_fisher = fm_mod.fisher_matrix(file_name=str(fisher_path))
    cfp_sub = _subset_fisher(cfp_fisher, params_cfp)

    cov_cfp = cfp_sub.inverse_fisher_matrix()
    corr_cfp = _corr_from_cov(cov_cfp)
    corr_planck = _corr_from_cov(cov_sub)
    corr_delta = corr_cfp - corr_planck

    triangle_path = outdir / "triangle_cfp_vs_planck_covmat.png"
    fco.make_triangle_plot(
        fishers=[cfp_sub, cov_fisher_obj],
        fisher_labels=["CosmicFish theta Fisher", "Planck covmat Gaussian"],
        params=params_cfp,
        colors=["#1f77b4", "#d62728"],
        shade_fisher=False,
        ls_fisher=["-", "--"],
        lw_fisher=[2.5, 2.2],
        savefile=str(triangle_path),
    )

    report = {
        "fisher_file": str(fisher_path),
        "planck_covmat": str(covmat_path),
        "params_cfp": params_cfp,
        "params_planck": params_planck,
        "covmat_gaussian_fisher_file": str(cov_fisher_root) + ".txt",
        "triangle_plot": str(triangle_path),
        "corr_cfp": corr_cfp.tolist(),
        "corr_planck_covmat": corr_planck.tolist(),
        "corr_delta": corr_delta.tolist(),
    }
    report_path = outdir / "correlation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[covmat-compare] wrote Gaussian Fisher: {cov_fisher_root}.txt")
    print(f"[covmat-compare] wrote triangle plot: {triangle_path}")
    print(f"[covmat-compare] wrote correlation report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
