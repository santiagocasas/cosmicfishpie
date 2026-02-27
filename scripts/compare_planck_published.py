#!/usr/bin/env python
# coding: utf-8

"""Compare a CosmicFish Planck Fisher against Planck published margestats.

This script compares marginalized 1-sigma constraints between:
- a Fisher matrix produced by CosmicFishPie, and
- Planck GetDist `.margestats` published in the PLA products.

Current parameter mapping focuses on the h-based 6-parameter primary basis used
by `scripts/run_planck_bestfit_fisher.py`:

- `ombh2`  <-> `omegabh2`
- `omch2`  <-> `omegach2`
- `h`      <-> `H0*`      (with conversion `h = H0/100`)
- `tau`    <-> `tau`
- `logAs`  <-> `logA`
- `ns`     <-> `ns`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, cast

from cosmicfishpie.analysis import fisher_matrix as fm_mod

DEFAULT_PLANCK_REL = Path(
    "Planck-Results/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE"
)
DEFAULT_CHAIN_ROOT = "base_plikHM_TTTEEE_lowl_lowE"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_margestats(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"margestats file not found: {path}")
    out: dict[str, dict[str, float]] = {}
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
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            mean = float(parts[1])
            sddev = float(parts[2])
        except ValueError:
            continue
        out[name] = {"mean": mean, "sddev": sddev}
    if not out:
        raise ValueError(f"Could not parse margestats table: {path}")
    return out


def _to_identity(x: float) -> float:
    return x


def _to_h_from_h0(x: float) -> float:
    return x / 100.0


def _resolve_fisher_path(path_arg: str) -> Path:
    p = Path(path_arg).expanduser().resolve()
    if p.is_file():
        return p
    p_txt = Path(str(p) + ".txt")
    if p_txt.is_file():
        return p_txt
    raise FileNotFoundError(f"Fisher file not found: {p}")


def _collect_fisher_sigmas(fisher_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    fish = fm_mod.fisher_matrix(file_name=str(fisher_path))
    names = list(fish.get_param_names())
    sigmas = list(fish.get_confidence_bounds())
    fidus = list(fish.get_param_fiducial())
    sigma_map: dict[str, float] = {}
    fid_map: dict[str, float] = {}
    for idx, name in enumerate(names):
        sigma_map[str(name)] = float(sigmas[idx])
        fid_map[str(name)] = float(fidus[idx])
    return sigma_map, fid_map


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'param':<10} {'sigma_fisher':>14} {'sigma_planck':>14} "
        f"{'ratio(F/P)':>12} {'delta_%':>10} {'status':>16}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        sf = row.get("sigma_fisher")
        sp = row.get("sigma_planck")
        rr = row.get("ratio_fisher_over_planck")
        dp = row.get("delta_percent")
        status = row.get("status", "")

        sf_str = f"{sf:.6e}" if isinstance(sf, (int, float)) else "n/a"
        sp_str = f"{sp:.6e}" if isinstance(sp, (int, float)) else "n/a"
        rr_str = f"{rr:.4f}" if isinstance(rr, (int, float)) else "n/a"
        dp_str = f"{dp:+.2f}" if isinstance(dp, (int, float)) else "n/a"
        print(
            f"{row['parameter']:<10} {sf_str:>14} {sp_str:>14} {rr_str:>12} {dp_str:>10} {status:>16}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CosmicFish Planck Fisher constraints with Planck margestats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fisher", help="Path to Fisher matrix .txt (or base path without .txt)")
    parser.add_argument(
        "--planck-dir",
        default=None,
        help="Directory containing Planck chain products for one data combo",
    )
    parser.add_argument("--chain-root", default=DEFAULT_CHAIN_ROOT)
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <fisher_dir>/compare_planck_published.json)",
    )
    args = parser.parse_args()

    fisher_path = _resolve_fisher_path(args.fisher)
    sigma_map, fid_map = _collect_fisher_sigmas(fisher_path)

    repo = _repo_root()
    planck_dir = (
        Path(args.planck_dir).expanduser().resolve()
        if args.planck_dir
        else (repo / DEFAULT_PLANCK_REL)
    )
    marg_path = planck_dir / "dist" / f"{args.chain_root}.margestats"
    marg = _parse_margestats(marg_path)

    targets: list[dict[str, Any]] = [
        {
            "canonical": "ombh2",
            "fisher_candidates": ["ombh2", "omegabh2"],
            "planck": "omegabh2",
            "conv": _to_identity,
        },
        {
            "canonical": "omch2",
            "fisher_candidates": ["omch2", "omegach2"],
            "planck": "omegach2",
            "conv": _to_identity,
        },
        {
            "canonical": "tau",
            "fisher_candidates": ["tau"],
            "planck": "tau",
            "conv": _to_identity,
        },
        {
            "canonical": "logAs",
            "fisher_candidates": ["logAs", "logA"],
            "planck": "logA",
            "conv": _to_identity,
        },
        {
            "canonical": "ns",
            "fisher_candidates": ["ns"],
            "planck": "ns",
            "conv": _to_identity,
        },
    ]

    if "theta" in sigma_map:
        targets.insert(
            2,
            {
                "canonical": "theta",
                "fisher_candidates": ["theta"],
                "planck": "theta",
                "conv": _to_identity,
            },
        )
    else:
        targets.insert(
            2,
            {
                "canonical": "h",
                "fisher_candidates": ["h"],
                "planck": "H0*",
                "conv": _to_h_from_h0,
            },
        )

    rows: list[dict] = []
    for target in targets:
        canonical = str(target["canonical"])
        fisher_candidates = [str(name) for name in target["fisher_candidates"]]
        planck_name = str(target["planck"])
        conv = target["conv"]
        if not callable(conv):
            raise TypeError(f"Invalid conversion callable for target={canonical}")
        conv_fn = cast(Callable[[float], float], conv)

        fisher_name = next(
            (name for name in fisher_candidates if name in sigma_map), fisher_candidates[0]
        )

        row = {
            "parameter": canonical,
            "fisher_parameter_used": fisher_name if fisher_name in sigma_map else None,
            "mapped_planck_parameter": planck_name,
            "fisher_fiducial": fid_map.get(fisher_name),
            "sigma_fisher": sigma_map.get(fisher_name),
            "planck_mean": None,
            "sigma_planck": None,
            "ratio_fisher_over_planck": None,
            "delta_percent": None,
            "status": "ok",
        }

        if fisher_name not in sigma_map:
            row["status"] = "missing_in_fisher"
            rows.append(row)
            continue
        if planck_name not in marg:
            row["status"] = "missing_in_planck"
            rows.append(row)
            continue

        pmean = conv_fn(float(marg[planck_name]["mean"]))
        psig = conv_fn(float(marg[planck_name]["sddev"]))
        fsig = float(sigma_map[fisher_name])

        row["planck_mean"] = pmean
        row["sigma_planck"] = psig
        if psig > 0.0:
            ratio = fsig / psig
            row["ratio_fisher_over_planck"] = ratio
            row["delta_percent"] = 100.0 * (ratio - 1.0)
        else:
            row["status"] = "invalid_planck_sigma"
        rows.append(row)

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else fisher_path.parent / "compare_planck_published.json"
    )

    payload = {
        "fisher_file": str(fisher_path),
        "planck_dir": str(planck_dir),
        "chain_root": args.chain_root,
        "margestats_file": str(marg_path),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[planck-compare] wrote {out_path}")
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
