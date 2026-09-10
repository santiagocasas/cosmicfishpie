#!/usr/bin/env python
"""Time the repeated ComputeCls build unit (what each Fisher stencil point pays).

Symbolic-backend friendly; runs N perturbed parameter points and times:
  - ComputeCls.__init__ (incl. cosmo_functions build)
  - P_limber / sqrtP_limber / compute_kernels / computecls_vectorized

Usage:
  OMP_NUM_THREADS=2 uv run python scripts/perf_getcls_parts.py --points 5
  ... --ref results/ref.npz        # dump reference Cls for correctness checks
  ... --check results/ref.npz      # compare current Cls vs reference (max rel err)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cosmicfishpie.configs import config as cfg
from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey import photo_obs


def resolve_specs_dir() -> str:
    pkg_root = Path(cfg.__file__).resolve().parent
    cand = pkg_root / "default_survey_specifications"
    if not cand.is_dir():
        raise FileNotFoundError(f"survey specs dir not found: {cand}")
    return str(cand) + "/"


def build_configuration(observables, accuracy):
    fiducial = {
        "Omegam": 0.32,
        "Omegab": 0.05,
        "h": 0.67,
        "ns": 0.96,
        "sigma8": 0.815584,
    }
    options = {
        "accuracy": int(round(accuracy)),
        "outroot": "PERFPARTS_",
        "results_dir": "results/",
        "derivatives": "3PT",
        "feedback": 0,
        "survey_name": "Euclid",
        "specs_dir": resolve_specs_dir(),
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": "symbolic",
    }
    boltz = pkg_boltzmann_yaml("symbolic")
    if boltz is not None:
        options["symbolic_config_yaml"] = str(boltz)
    return cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        freepars={"Omegam": 0.01, "h": 0.01},
        options=options,
        observables=observables,
        cosmoModel="LCDM",
    )


def pkg_boltzmann_yaml(code: str):
    pkg_root = Path(cfg.__file__).resolve().parent
    for name in ("default_boltzmann_yaml_files", "boltzmann_yaml_files"):
        cand = pkg_root / name / code / "default.yaml"
        if cand.is_file():
            return cand
    return None


def time_one_point(fm, cosmopars, label):
    t0 = time.perf_counter()
    cls = photo_obs.ComputeCls(
        dict(cosmopars),
        fm.photopars,
        fm.IApars,
        fm.photobiaspars,
        print_info_specs=False,
        configuration=fm,
    )
    t1 = time.perf_counter()
    cls.P_limber()
    t2 = time.perf_counter()
    cls.sqrtP_limber()
    t3 = time.perf_counter()
    cls.compute_kernels()
    t4 = time.perf_counter()
    result = cls.computecls_vectorized()
    t5 = time.perf_counter()
    parts = {
        "build": t1 - t0,
        "P_limber": t2 - t1,
        "sqrtP_limber": t3 - t2,
        "compute_kernels": t4 - t3,
        "computecls_vectorized": t5 - t4,
        "total": t5 - t0,
    }
    return parts, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=int, default=5)
    p.add_argument("--observables", nargs="+", default=["GCph", "WL"])
    p.add_argument("--accuracy", type=float, default=1.0)
    p.add_argument("--json", default=None, help="Write per-part timings to JSON")
    p.add_argument("--ref", default=None, help="npz path to dump fiducial-result reference")
    p.add_argument("--check", default=None, help="npz path of reference to verify against")
    args = p.parse_args()

    fm = build_configuration(args.observables, args.accuracy)
    base = dict(fm.fiducialcosmopars)

    # perturbation sequence mimics stencil diversity (no repeated parameter sets)
    steps = [0.0, 0.005, -0.005, 0.01, -0.01, 0.0025, -0.0025, 0.0075]
    rows = []
    ref_results = {}
    for i in range(args.points):
        pars = dict(base)
        pars["Omegam"] = base["Omegam"] + steps[i % len(steps)] * base["Omegam"]
        pars["h"] = base["h"] - steps[i % len(steps)] * base["h"] * 0.5
        parts, result = time_one_point(fm, pars, f"pt{i}")
        rows.append(parts)
        ref_results[i] = result
        print(f"point {i}: " + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()))

    keys = list(rows[0].keys())
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    first = {k: rows[0][k] for k in keys}
    print("\n== mean over points (s) ==")
    for k in keys:
        share = 100.0 * means[k] / means["total"] if means["total"] > 0 else 0.0
        print(f"  {k:24s} {means[k]:8.4f}  ({share:5.1f}% of total)   first: {first[k]:.4f}")

    if args.ref:
        arrays = {"ells": ref_results[0]["ells"]}
        for j, res in ref_results.items():
            for k, v in res.items():
                if k == "ells" or not isinstance(v, np.ndarray):
                    continue
                arrays[f"pt{j}::{k}"] = v
        Path(args.ref).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.ref, **arrays)
        print(f"reference dumped to {args.ref}")

    if args.check:
        ref = np.load(args.check)
        worst = 0.0
        for j, res in ref_results.items():
            for k, v in res.items():
                if k == "ells" or not isinstance(v, np.ndarray):
                    continue
                name = f"pt{j}::{k}"
                if name not in ref:
                    print(f"  MISSING in reference: {name}")
                    continue
                rv = ref[name]
                denom = np.maximum(np.abs(rv), 1e-300)
                err = float(np.max(np.abs(v - rv) / denom))
                worst = max(worst, err)
        print(f"max rel err vs reference: {worst:.3e}")

    if args.json:
        out = {
            "points": args.points,
            "observables": args.observables,
            "accuracy": args.accuracy,
            "fast_flags": {
                "eff": photo_obs._USE_FAST_EFF,
                "P": photo_obs._USE_FAST_P,
                "kernel": photo_obs._USE_FAST_KERNEL,
            },
            "mean_parts_sec": means,
            "rows": rows,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"timings written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
