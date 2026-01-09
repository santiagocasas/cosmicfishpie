#!/usr/bin/env python
# coding: utf-8

"""Compare all Fisher matrices in a directory."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from cosmicfishpie.analysis import fisher_matrix as fm_mod
from cosmicfishpie.analysis import fisher_plot_analysis as cfa


def _load_specs(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _resolve_matrix_path(specs: dict, base_dir: Path, specs_path: Path) -> Path | None:
    meta = specs.get("metadata", {})
    txt = meta.get("matrix_files", {}).get("txt") if isinstance(meta, dict) else None
    if txt:
        txt_path = Path(txt)
        if txt_path.is_file():
            return txt_path
        alt = base_dir / txt_path.name
        if alt.is_file():
            return alt
    # Fallback by spec filename
    name = specs_path.name
    if name.endswith("_FM_specs.json"):
        root = name[: -len("_FM_specs.json")]
        cand = base_dir / f"{root}_FM.txt"
        if cand.is_file():
            return cand
    if name.endswith("_specifications.json"):
        root = name[: -len("_specifications.json")]
        cand = base_dir / f"{root}_fishermatrix.txt"
        if cand.is_file():
            return cand
    return None


def _metrics(A: np.ndarray, B: np.ndarray, thresh: float = 1e-12) -> dict:
    D = A - B
    mask = np.abs(B) > thresh
    rel = np.zeros_like(A)
    rel[mask] = (A[mask] / B[mask]) - 1.0
    out = {
        "rel_max": float(np.max(np.abs(rel[mask]))) if np.any(mask) else 0.0,
        "abs_max_near0": float(np.max(np.abs(D[~mask]))) if np.any(~mask) else 0.0,
        "fro_rel": float(np.linalg.norm(D) / (np.linalg.norm(B) + 1e-30)),
    }
    dA = np.diag(A)
    dB = np.diag(B)
    dmask = np.abs(dB) > thresh
    if np.any(dmask):
        ratios = dA[dmask] / dB[dmask]
        out["diag_ratio"] = {
            "min": float(np.min(ratios)),
            "median": float(np.median(ratios)),
            "max": float(np.max(ratios)),
        }
    else:
        out["diag_ratio"] = {}
    return out


def _shallow_diff(a: dict, b: dict) -> dict:
    dif = {"changed": {}, "only_in_a": [], "only_in_b": []}
    if not isinstance(a, dict) or not isinstance(b, dict):
        return dif
    ka = set(a.keys())
    kb = set(b.keys())
    for k in sorted(ka - kb):
        dif["only_in_a"].append(k)
    for k in sorted(kb - ka):
        dif["only_in_b"].append(k)
    for k in sorted(ka & kb):
        va, vb = a.get(k), b.get(k)
        if va != vb:
            dif["changed"][k] = {"a": va, "b": vb}
    return dif


def _align_matrices(fm_ref, fm_cur):
    ref_names = list(fm_ref.param_names)
    cur_names = list(fm_cur.param_names)
    name_to_idx = {n: i for i, n in enumerate(cur_names)}
    common = [n for n in ref_names if n in name_to_idx]
    missing = [n for n in ref_names if n not in name_to_idx]
    if not common:
        return None, {"error": "no_common_params", "missing_in_current": missing}
    ref_idx = [ref_names.index(n) for n in common]
    cur_idx = [name_to_idx[n] for n in common]
    A = np.array(fm_ref.fisher_matrix)[np.ix_(ref_idx, ref_idx)]
    B = np.array(fm_cur.fisher_matrix)[np.ix_(cur_idx, cur_idx)]
    return (A, B, common), {"missing_in_current": missing}


def _analysis_pair(fm_a, fm_b, parstomarg: list[str]) -> dict:
    a_names = set(fm_a.get_param_names())
    b_names = set(fm_b.get_param_names())
    missing_a = [p for p in parstomarg if p not in a_names]
    missing_b = [p for p in parstomarg if p not in b_names]
    if missing_a or missing_b:
        raise SystemExit(
            "[compare] FoM parameters not found in both Fishers. "
            f"missing_in_a={missing_a} missing_in_b={missing_b}"
        )
    analysis = cfa.CosmicFish_FisherAnalysis(fisher_list=[fm_a, fm_b])
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        results = analysis.compare_fisher_results(parstomarg=parstomarg)
    params = {}
    if len(results) >= 2:
        a_params = {p["name"]: p for p in results[0].get("parameters", [])}
        b_params = {p["name"]: p for p in results[1].get("parameters", [])}
        for name in sorted(set(a_params) & set(b_params)):
            a_sigma = a_params[name].get("sigma")
            b_sigma = b_params[name].get("sigma")
            ratio = (b_sigma / a_sigma) if a_sigma and b_sigma else None
            params[name] = {"sigma_a": a_sigma, "sigma_b": b_sigma, "ratio_b_over_a": ratio}
    return {"results": results, "param_sigma_ratio": params}


def _summarize_timing(specs: dict) -> float | None:
    meta = specs.get("metadata", {})
    if isinstance(meta, dict):
        return meta.get("totaltime_sec")
    return None


def _yaml_info(specs: dict) -> dict:
    options = specs.get("options", {}) if isinstance(specs.get("options", {}), dict) else {}
    code = options.get("code")
    key_map = {
        "class": "class_config_yaml",
        "camb": "camb_config_yaml",
        "symbolic": "symbolic_config_yaml",
    }
    yaml_key = key_map.get(code)
    path = options.get(yaml_key) if yaml_key else None
    return {"code": code, "path": path, "key": yaml_key}


def _load_yaml(path: str) -> dict | list | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _yaml_struct_diff(a_obj, b_obj) -> dict:
    if isinstance(a_obj, dict) and isinstance(b_obj, dict):
        out = {"changed": {}, "only_in_a": {}, "only_in_b": {}, "same": {}}
        ka = set(a_obj.keys())
        kb = set(b_obj.keys())
        for k in sorted(ka - kb):
            out["only_in_a"][k] = a_obj[k]
        for k in sorted(kb - ka):
            out["only_in_b"][k] = b_obj[k]
        for k in sorted(ka & kb):
            va, vb = a_obj[k], b_obj[k]
            if va == vb:
                out["same"][k] = va
                continue
            out["changed"][k] = _yaml_struct_diff(va, vb)
        return out
    if isinstance(a_obj, list) and isinstance(b_obj, list):
        if a_obj == b_obj:
            return {"same": a_obj}
        return {"a": a_obj, "b": b_obj}
    if a_obj == b_obj:
        return {"same": a_obj}
    return {"a": a_obj, "b": b_obj}


def _yaml_diff(a_path: str | None, b_path: str | None) -> dict:
    if not a_path or not b_path:
        return {"error": "missing_yaml_path"}
    if not os.path.isfile(a_path):
        return {"error": f"missing_yaml_file: {a_path}"}
    if not os.path.isfile(b_path):
        return {"error": f"missing_yaml_file: {b_path}"}
    a_obj = _load_yaml(a_path)
    b_obj = _load_yaml(b_path)
    if a_obj is None or b_obj is None:
        return {"error": "yaml_parse_failed_or_missing_pyyaml"}
    return {"diff": _yaml_struct_diff(a_obj, b_obj)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Fisher matrices in a directory.")
    parser.add_argument(
        "dir",
        help="Directory containing *_FM_specs.json and *_FM.txt files",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Reference specs JSON filename (default: first found)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON filename (default: compare_fishers_<YYYYMMDD_HHMMSS>.json)",
    )
    parser.add_argument(
        "--fom-params",
        default=None,
        help="Comma-separated list of parameters for FoM (default: first two parameters)",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.is_dir():
        raise SystemExit(f"Directory not found: {base_dir}")

    specs_files = sorted(base_dir.glob("*_FM_specs.json")) + sorted(
        base_dir.glob("*_specifications.json")
    )
    if not specs_files:
        raise SystemExit("No specs JSON files found.")

    ref_path = None
    if args.ref:
        ref_path = base_dir / args.ref
        if not ref_path.is_file():
            raise SystemExit(f"Reference specs not found: {ref_path}")
    else:
        ref_path = specs_files[0]

    ref_specs = _load_specs(ref_path)
    ref_txt = _resolve_matrix_path(ref_specs, base_dir, ref_path)
    if not ref_txt or not ref_txt.is_file():
        raise SystemExit(f"Reference matrix not found for {ref_path}")
    fm_ref = fm_mod.fisher_matrix(file_name=str(ref_txt))

    ref_time = _summarize_timing(ref_specs)

    fom_params = None
    if args.fom_params:
        parts = [p.strip() for p in args.fom_params.split(",") if p.strip()]
        if len(parts) < 2:
            raise SystemExit("--fom-params expects at least two comma-separated names")
        fom_params = parts

    results = {
        "dir": str(base_dir),
        "reference": {
            "specs": str(ref_path),
            "matrix": str(ref_txt),
            "timing_seconds": ref_time,
        },
        "fom_params": fom_params,
        "pairwise": [],
    }

    for spec_path in specs_files:
        if spec_path == ref_path:
            continue
        cur_specs = _load_specs(spec_path)
        cur_txt = _resolve_matrix_path(cur_specs, base_dir, spec_path)
        if not cur_txt or not cur_txt.is_file():
            results["pairwise"].append(
                {
                    "a": str(ref_path),
                    "b": str(spec_path),
                    "error": "missing_matrix",
                }
            )
            continue
        fm_cur = fm_mod.fisher_matrix(file_name=str(cur_txt))
        aligned, meta = _align_matrices(fm_ref, fm_cur)
        cur_time = _summarize_timing(cur_specs)
        entry = {
            "a": str(ref_path),
            "b": str(spec_path),
            "a_matrix": str(ref_txt),
            "b_matrix": str(cur_txt),
            "a_timing_seconds": ref_time,
            "b_timing_seconds": cur_time,
            "matrix_metrics": None,
            "missing_in_b": meta.get("missing_in_current", []),
            "specs_diff": None,
            "analysis": None,
            "yaml": None,
        }
        if aligned is None:
            entry["matrix_metrics"] = meta
            results["pairwise"].append(entry)
            continue
        A, B, names = aligned
        entry["matrix_metrics"] = _metrics(A, B)
        entry["analysis"] = _analysis_pair(fm_ref, fm_cur, fom_params or names[:2])
        yaml_a = _yaml_info(ref_specs)
        yaml_b = _yaml_info(cur_specs)
        entry["yaml"] = {
            "a": yaml_a,
            "b": yaml_b,
            "diff": _yaml_diff(yaml_a.get("path"), yaml_b.get("path")),
        }

        diffs = {
            "options": _shallow_diff(ref_specs.get("options", {}), cur_specs.get("options", {})),
            "specifications": _shallow_diff(
                ref_specs.get("specifications", {}), cur_specs.get("specifications", {})
            ),
            "fiducialpars": _shallow_diff(
                ref_specs.get("fiducialpars", {}), cur_specs.get("fiducialpars", {})
            ),
            "freepars": _shallow_diff(ref_specs.get("freepars", {}), cur_specs.get("freepars", {})),
            "metadata": _shallow_diff(ref_specs.get("metadata", {}), cur_specs.get("metadata", {})),
        }
        entry["specs_diff"] = diffs
        results["pairwise"].append(entry)

    # Pairwise comparisons across all specs
    for i, spec_a in enumerate(specs_files):
        for spec_b in specs_files[i + 1 :]:
            a_specs = _load_specs(spec_a)
            b_specs = _load_specs(spec_b)
            a_txt = _resolve_matrix_path(a_specs, base_dir, spec_a)
            b_txt = _resolve_matrix_path(b_specs, base_dir, spec_b)
            entry = {
                "a": str(spec_a),
                "b": str(spec_b),
                "a_matrix": str(a_txt) if a_txt else None,
                "b_matrix": str(b_txt) if b_txt else None,
                "a_timing_seconds": _summarize_timing(a_specs),
                "b_timing_seconds": _summarize_timing(b_specs),
                "matrix_metrics": None,
                "missing_in_b": [],
                "specs_diff": None,
                "analysis": None,
                "yaml": None,
            }
            if not a_txt or not a_txt.is_file() or not b_txt or not b_txt.is_file():
                entry["matrix_metrics"] = {"error": "missing_matrix"}
                results["pairwise"].append(entry)
                continue
            fm_a = fm_mod.fisher_matrix(file_name=str(a_txt))
            fm_b = fm_mod.fisher_matrix(file_name=str(b_txt))
            aligned, meta = _align_matrices(fm_a, fm_b)
            entry["missing_in_b"] = meta.get("missing_in_current", [])
            if aligned is None:
                entry["matrix_metrics"] = meta
                results["pairwise"].append(entry)
                continue
            A, B, names = aligned
            entry["matrix_metrics"] = _metrics(A, B)
            entry["analysis"] = _analysis_pair(fm_a, fm_b, fom_params or names[:2])
            yaml_a = _yaml_info(a_specs)
            yaml_b = _yaml_info(b_specs)
            entry["yaml"] = {
                "a": yaml_a,
                "b": yaml_b,
                "diff": _yaml_diff(yaml_a.get("path"), yaml_b.get("path")),
            }
            entry["specs_diff"] = {
                "options": _shallow_diff(a_specs.get("options", {}), b_specs.get("options", {})),
                "specifications": _shallow_diff(
                    a_specs.get("specifications", {}), b_specs.get("specifications", {})
                ),
                "fiducialpars": _shallow_diff(
                    a_specs.get("fiducialpars", {}), b_specs.get("fiducialpars", {})
                ),
                "freepars": _shallow_diff(a_specs.get("freepars", {}), b_specs.get("freepars", {})),
                "metadata": _shallow_diff(a_specs.get("metadata", {}), b_specs.get("metadata", {})),
            }
            results["pairwise"].append(entry)

    out_name = args.out or f"compare_fishers_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path = base_dir / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[compare] Wrote {out_path}")

    # Human-friendly summary
    print("[compare] Summary")
    print(f"[compare] Pairs compared: {len(results['pairwise'])}")
    for entry in results["pairwise"]:
        a = Path(entry.get("a", "A")).name
        b = Path(entry.get("b", "B")).name
        timing_a = entry.get("a_timing_seconds")
        timing_b = entry.get("b_timing_seconds")
        timing_ratio = None
        if isinstance(timing_a, (int, float)) and isinstance(timing_b, (int, float)):
            if timing_a != 0:
                timing_ratio = timing_b / timing_a

        yaml_diff = entry.get("yaml", {}).get("diff", {}).get("diff", {})
        yaml_changed = len(yaml_diff.get("changed", {})) if isinstance(yaml_diff, dict) else 0
        yaml_only_a = len(yaml_diff.get("only_in_a", {})) if isinstance(yaml_diff, dict) else 0
        yaml_only_b = len(yaml_diff.get("only_in_b", {})) if isinstance(yaml_diff, dict) else 0

        max_ratio = None
        max_ratio_val = None
        ratios = (
            entry.get("analysis", {}).get("param_sigma_ratio", {})
            if isinstance(entry.get("analysis", {}), dict)
            else {}
        )
        for name, vals in ratios.items():
            ratio = vals.get("ratio_b_over_a")
            if ratio is None:
                continue
            dev = abs(ratio - 1.0)
            if max_ratio is None or dev > max_ratio:
                max_ratio = dev
                max_ratio_val = ratio

        print(f"[compare] a: {a}")
        print(f"[compare] b: {b}")
        if timing_ratio is not None:
            print(f"[compare]   timing ratio (b/a): {timing_ratio:.3f}x")
        print(
            f"[compare]   boltzmann_specifications yaml keys changed/only_a/only_b: "
            f"{yaml_changed}/{yaml_only_a}/{yaml_only_b}"
        )
        specs_diff = (
            entry.get("specs_diff", {}) if isinstance(entry.get("specs_diff", {}), dict) else {}
        )
        opt_diff = (
            specs_diff.get("options", {}) if isinstance(specs_diff.get("options", {}), dict) else {}
        )
        spec_diff = (
            specs_diff.get("specifications", {})
            if isinstance(specs_diff.get("specifications", {}), dict)
            else {}
        )
        opt_changed = len(opt_diff.get("changed", {}))
        opt_only_a = len(opt_diff.get("only_in_a", []))
        opt_only_b = len(opt_diff.get("only_in_b", []))
        spec_changed = len(spec_diff.get("changed", {}))
        spec_only_a = len(spec_diff.get("only_in_a", []))
        spec_only_b = len(spec_diff.get("only_in_b", []))
        print(
            f"[compare]   general options keys changed/only_a/only_b: "
            f"{opt_changed}/{opt_only_a}/{opt_only_b}"
        )
        print(
            f"[compare]   survey specifications keys changed/only_a/only_b: "
            f"{spec_changed}/{spec_only_a}/{spec_only_b}"
        )
        if max_ratio is not None:
            percent_dev = 100.0 * max_ratio
            print(
                f"[compare]   max sigma ratio deviation: {percent_dev:.2f}% "
                f"(ratio b/a: {max_ratio_val:.3e})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
