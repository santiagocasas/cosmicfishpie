#!/usr/bin/env python
# coding: utf-8

"""Run a Planck-like CMB Fisher at the Planck chain best-fit point.

This script reads best-fit values from Planck chain products and runs a CMB
Fisher in a 6-parameter primary basis.

Supported parameterizations:

- `h`: `ombh2`, `omch2`, `h`, `tau`, `logAs`, `ns`
- `theta`: `ombh2`, `omch2`, `theta`, `tau`, `logAs`, `ns`

It targets the baseline 2018 likelihood tag `plikHM_TTTEEE_lowl_lowE` using
settings inferred from the provided Planck files as closely as supported by the
current CosmicFishPie CMB pipeline.

Notes
-----
- Planck chain best-fit is taken from `.likestats` by default.
- The CMB pipeline uses a single ell range for all spectra; to include
  ell=2508, this script sets internal `lmax_CMB=2509` (half-open convention).
- Full Planck nuisance-parameter likelihood modeling is not implemented here;
  this is an approximate reproduction in the reduced cosmological basis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import yaml

from cosmicfishpie.fishermatrix import cosmicfish

DEFAULT_PLANCK_REL = Path(
    "Planck-Results/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE"
)
DEFAULT_CHAIN_ROOT = "base_plikHM_TTTEEE_lowl_lowE"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _git_commit(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        head_txt = head.read_text(encoding="utf-8").strip()
        if head_txt.startswith("ref: "):
            ref = head_txt.removeprefix("ref: ")
            ref_path = repo_root / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
        return head_txt[:12]
    except Exception:
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a dict at top-level: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _build_planck_camb_yaml(base_yaml: Path, out_yaml: Path) -> Path:
    cfg = _read_yaml(base_yaml)
    cfg.setdefault("COSMO_SETTINGS", {})
    if not isinstance(cfg["COSMO_SETTINGS"], dict):
        raise ValueError(f"COSMO_SETTINGS must be a dict in {base_yaml}")

    settings = cfg["COSMO_SETTINGS"]
    settings["WantCls"] = True
    settings["Want_CMB"] = True
    settings["Want_CMB_lensing"] = True
    settings["Want_cl_2D_array"] = False
    settings["DoLensing"] = True
    settings["Reion.Reionization"] = True
    # Planck 2018 chains indicate halofit_version=5 in input params.
    settings["halofit_version"] = 5

    _write_yaml(out_yaml, cfg)
    return out_yaml


def _parse_likestats(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"likestats file not found: {path}")
    bestfit: dict[str, float] = {}
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
            value = float(parts[1])
        except ValueError:
            continue
        bestfit[name] = value
    if not bestfit:
        raise ValueError(f"Could not parse best-fit entries from {path}")
    return bestfit


def _parse_minimum(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"minimum file not found: {path}")
    bestfit: dict[str, float] = {}
    patt = re.compile(r"^\s*\d+\s+([+-]?\d*\.?\d+E[+-]\d+)\s+(\S+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = patt.match(line)
        if not match:
            continue
        value = float(match.group(1))
        name = match.group(2)
        bestfit[name] = value
    if not bestfit:
        raise ValueError(f"Could not parse best-fit entries from {path}")
    return bestfit


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
        try:
            out[parts[0]] = {"mean": float(parts[1]), "sddev": float(parts[2])}
        except ValueError:
            continue
    return out


def _parse_inputparams(path: Path) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    if not path.exists():
        return {}, {}
    step_abs: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    patt = re.compile(r"^\s*param\[(?P<name>[^\]]+)\]\s*=\s*(?P<body>.+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = patt.match(line)
        if not match:
            continue
        name = match.group("name")
        tokens = match.group("body").split()
        vals: list[float] = []
        for tok in tokens:
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if len(vals) >= 3:
            ranges[name] = (vals[1], vals[2])
        if len(vals) >= 4:
            step_abs[name] = abs(vals[3])
    return step_abs, ranges


def _get_h_from_bestfit(bestfit: dict[str, float]) -> float:
    if "H0*" in bestfit:
        return bestfit["H0*"] / 100.0
    if "H0" in bestfit:
        return bestfit["H0"] / 100.0
    raise KeyError("Could not find H0* or H0 in Planck best-fit inputs")


def _safe_relstep(step_abs: float, fid: float) -> float:
    if fid == 0.0:
        return step_abs
    return abs(step_abs / fid)


def _validate_obs(obs: list[str]) -> None:
    allowed = {"CMB_T", "CMB_E", "CMB_B"}
    bad = [o for o in obs if o not in allowed]
    if bad:
        raise ValueError(f"Invalid observables: {bad} (allowed: {sorted(allowed)})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Planck best-fit CMB Fisher in h/theta primary basis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--planck-dir",
        default=None,
        help="Directory containing Planck chain products for one data combo",
    )
    parser.add_argument("--chain-root", default=DEFAULT_CHAIN_ROOT)
    parser.add_argument(
        "--bestfit-source",
        choices=["likestats", "minimum"],
        default="likestats",
        help="Use chain best-fit sample (.likestats) or minimizer best fit (.minimum)",
    )
    parser.add_argument(
        "--parameterization",
        choices=["h", "theta"],
        default="h",
        help="Primary-parameter basis for the Fisher run",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: scripts/benchmark_results/planck_bestfit_<timestamp>)",
    )
    parser.add_argument("--observables", default="CMB_T,CMB_E")
    parser.add_argument("--lmin", type=int, default=2)
    parser.add_argument(
        "--lmax",
        type=int,
        default=2508,
        help="Maximum ell to include (inclusive). Internal lmax_CMB is set to lmax+1",
    )
    parser.add_argument("--accuracy", type=int, default=1)
    parser.add_argument("--feedback", type=int, default=1)
    parser.add_argument("--derivatives", default="3PT")
    parser.add_argument(
        "--h-step-abs",
        type=float,
        default=None,
        help="Absolute derivative step for h (default: half Planck sigma_h from margestats)",
    )
    parser.add_argument(
        "--theta-step-abs",
        type=float,
        default=None,
        help="Absolute derivative step for theta=100*theta_MC (default: from Planck inputparams)",
    )
    parser.add_argument(
        "--fsky",
        type=float,
        default=None,
        help="Override f_sky for T/E/B if set",
    )
    parser.add_argument("--beam-arcmin", type=float, default=None)
    parser.add_argument("--temp-sens", type=float, default=None, help="uK-arcmin")
    parser.add_argument("--pol-sens", type=float, default=None, help="uK-arcmin")
    parser.add_argument(
        "--cmb-noise-model",
        choices=["legacy", "knox"],
        default="knox",
        help="CMB noise model in CMBCov",
    )
    parser.add_argument(
        "--ee-lowell-noise-boost",
        type=float,
        default=1.0,
        help="Optional factor to boost EE noise at low ell (used with knox mode)",
    )
    parser.add_argument(
        "--ee-lowell-max-ell",
        type=int,
        default=29,
        help="Maximum ell where EE low-ell noise boost is applied",
    )
    parser.add_argument(
        "--ee-lowell-inflation",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ee-lowell-lmax",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camb-yaml",
        default=None,
        help="Base CAMB YAML (default: package camb/default.yaml)",
    )
    args = parser.parse_args()

    repo = _repo_root()
    planck_dir = (
        Path(args.planck_dir).expanduser().resolve()
        if args.planck_dir
        else (repo / DEFAULT_PLANCK_REL)
    )
    if not planck_dir.exists():
        raise SystemExit(f"Planck directory not found: {planck_dir}")

    run_id = _timestamp_id()
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else repo / "scripts" / "benchmark_results" / f"planck_bestfit_{run_id}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_dir = repo / "cosmicfishpie" / "configs"
    spec_planck_yaml = cfg_dir / "default_survey_specifications" / "Planck.yaml"
    if not spec_planck_yaml.exists():
        raise SystemExit(f"Planck spec YAML not found: {spec_planck_yaml}")

    base_camb_yaml = (
        Path(args.camb_yaml).expanduser().resolve()
        if args.camb_yaml
        else (cfg_dir / "default_boltzmann_yaml_files" / "camb" / "default.yaml")
    )
    if not base_camb_yaml.exists():
        raise SystemExit(f"CAMB YAML not found: {base_camb_yaml}")

    bestfit_path: Path
    if args.bestfit_source == "likestats":
        bestfit_path = planck_dir / "dist" / f"{args.chain_root}.likestats"
        bestfit_raw = _parse_likestats(bestfit_path)
    else:
        bestfit_path = planck_dir / f"{args.chain_root}.minimum"
        bestfit_raw = _parse_minimum(bestfit_path)

    margestats_path = planck_dir / "dist" / f"{args.chain_root}.margestats"
    margestats = _parse_margestats(margestats_path)
    inputparams_path = planck_dir / f"{args.chain_root}.minimum.inputparams"
    step_abs_planck, planck_ranges = _parse_inputparams(inputparams_path)

    required = ["omegabh2", "omegach2", "tau", "logA", "ns"]
    if args.parameterization == "theta":
        required += ["theta"]
    missing = [k for k in required if k not in bestfit_raw]
    if args.parameterization == "h" and "H0*" not in bestfit_raw and "H0" not in bestfit_raw:
        missing += ["H0*|H0"]
    if missing:
        raise SystemExit(f"Missing required best-fit keys in {bestfit_path}: {missing}")

    fiducial = {
        "ombh2": float(bestfit_raw["omegabh2"]),
        "omch2": float(bestfit_raw["omegach2"]),
        "tau": float(bestfit_raw["tau"]),
        "logAs": float(bestfit_raw["logA"]),
        "ns": float(bestfit_raw["ns"]),
        # Baseline assumptions listed in Planck products.
        "mnu": 0.06,
        "Neff": 3.046,
    }
    if args.parameterization == "h":
        fiducial["h"] = float(_get_h_from_bestfit(bestfit_raw))
    else:
        fiducial["theta"] = float(bestfit_raw["theta"])

    sigma_h = None
    if "H0*" in margestats and isinstance(margestats["H0*"].get("sddev"), float):
        sigma_h = margestats["H0*"]["sddev"] / 100.0

    step_abs = {
        "ombh2": float(step_abs_planck.get("omegabh2", 1.0e-4)),
        "omch2": float(step_abs_planck.get("omegach2", 1.0e-3)),
        "tau": float(step_abs_planck.get("tau", 6.0e-3)),
        "logAs": float(step_abs_planck.get("logA", 1.0e-3)),
        "ns": float(step_abs_planck.get("ns", 4.0e-3)),
    }
    if args.parameterization == "h":
        step_abs["h"] = (
            float(args.h_step_abs)
            if args.h_step_abs is not None
            else float(0.5 * sigma_h if sigma_h is not None else 3.0e-3)
        )
    else:
        step_abs["theta"] = (
            float(args.theta_step_abs)
            if args.theta_step_abs is not None
            else float(step_abs_planck.get("theta", 2.0e-4))
        )

    freepars: dict[str, float] = {
        "ombh2": _safe_relstep(step_abs["ombh2"], fiducial["ombh2"]),
        "omch2": _safe_relstep(step_abs["omch2"], fiducial["omch2"]),
        "tau": _safe_relstep(step_abs["tau"], fiducial["tau"]),
        "logAs": _safe_relstep(step_abs["logAs"], fiducial["logAs"]),
        "ns": _safe_relstep(step_abs["ns"], fiducial["ns"]),
    }
    if args.parameterization == "h":
        freepars["h"] = _safe_relstep(step_abs["h"], fiducial["h"])
    else:
        freepars["theta"] = _safe_relstep(step_abs["theta"], fiducial["theta"])

    specs_all = _read_yaml(spec_planck_yaml)
    spec = specs_all.get("specifications", specs_all)
    if not isinstance(spec, dict):
        raise SystemExit(f"Invalid spec structure in: {spec_planck_yaml}")
    spec = dict(spec)
    spec["lmin_CMB"] = int(args.lmin)
    spec["lmax_CMB"] = int(args.lmax) + 1
    if args.fsky is not None:
        fsky = float(args.fsky)
        spec["fsky_CMB_T"] = fsky
        spec["fsky_CMB_E"] = fsky
        spec["fsky_CMB_B"] = fsky
    if args.beam_arcmin is not None:
        spec["CMB_fwhm"] = [float(args.beam_arcmin)]
    if args.temp_sens is not None:
        spec["CMB_temp_sens"] = [float(args.temp_sens)]
    if args.pol_sens is not None:
        spec["CMB_pol_sens"] = [float(args.pol_sens)]
    ee_lowell_boost = float(args.ee_lowell_noise_boost)
    ee_lowell_max_ell = int(args.ee_lowell_max_ell)
    if args.ee_lowell_inflation is not None:
        ee_lowell_boost = float(args.ee_lowell_inflation)
        warnings.warn(
            "--ee-lowell-inflation is deprecated; use --ee-lowell-noise-boost.",
            DeprecationWarning,
            stacklevel=1,
        )
    if args.ee_lowell_lmax is not None:
        ee_lowell_max_ell = int(args.ee_lowell_lmax)
        warnings.warn(
            "--ee-lowell-lmax is deprecated; use --ee-lowell-max-ell.",
            DeprecationWarning,
            stacklevel=1,
        )

    spec["CMB_noise_model"] = str(args.cmb_noise_model)
    spec["CMB_EE_noise_boost_lowell"] = ee_lowell_boost
    spec["CMB_EE_noise_boost_lmax"] = ee_lowell_max_ell

    obs = [o.strip() for o in args.observables.split(",") if o.strip()]
    _validate_obs(obs)

    camb_yaml_out = outdir / "camb_planck_like.yaml"
    _build_planck_camb_yaml(base_camb_yaml, camb_yaml_out)

    options = {
        "code": "camb",
        "camb_config_yaml": str(camb_yaml_out),
        "accuracy": int(args.accuracy),
        "feedback": int(args.feedback),
        "derivatives": str(args.derivatives),
        "results_dir": str(outdir) + "/",
        "outroot": f"planck_bestfit_{args.parameterization}_primary_",
        "cosmo_model": "LCDM",
    }

    latexnames = {
        "ombh2": r"\Omega_b h^2",
        "omch2": r"\Omega_c h^2",
        "tau": r"\tau",
        "logAs": r"\ln(10^{10} A_s)",
        "ns": r"n_s",
    }
    if args.parameterization == "h":
        latexnames["h"] = r"h"
    else:
        latexnames["theta"] = r"100\theta_{\rm MC}"

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": list(sys.argv),
        "git_commit": _git_commit(repo),
        "cwd": os.getcwd(),
        "inputs": {
            "planck_dir": str(planck_dir),
            "chain_root": args.chain_root,
            "bestfit_source": args.bestfit_source,
            "parameterization": args.parameterization,
            "bestfit_path": str(bestfit_path),
            "margestats_path": str(margestats_path),
            "inputparams_path": str(inputparams_path),
            "spec_planck_yaml": str(spec_planck_yaml),
            "base_camb_yaml": str(base_camb_yaml),
            "camb_yaml_used": str(camb_yaml_out),
        },
        "resolved": {
            "observables": obs,
            "fiducial": fiducial,
            "freepars_relative": freepars,
            "freepars_abs": step_abs,
            "specifications": spec,
            "basis_parameters": list(freepars.keys()),
            "lmax_inclusive_requested": int(args.lmax),
            "lmax_exclusive_internal": spec["lmax_CMB"],
            "planck_inputparam_ranges": planck_ranges,
        },
        "notes": [
            "Best-fit from Planck chain products (.likestats by default).",
            (
                "Uses h-based primary basis: ombh2, omch2, h, tau, logAs, ns."
                if args.parameterization == "h"
                else "Uses theta-based primary basis: ombh2, omch2, theta, tau, logAs, ns."
            ),
            "Single CMB ell-range approximation in current CosmicFishPie CMB pipeline.",
            "Does not include full Planck nuisance-parameter likelihood block.",
        ],
    }
    meta_path = outdir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print("[planck-bestfit] outdir:", outdir)
    print("[planck-bestfit] bestfit source:", args.bestfit_source)
    print("[planck-bestfit] bestfit path:", bestfit_path)
    print("[planck-bestfit] parameterization:", args.parameterization)
    print("[planck-bestfit] observables:", obs)
    print("[planck-bestfit] ell range:", spec["lmin_CMB"], "..", int(spec["lmax_CMB"]) - 1)
    if args.parameterization == "h":
        print("[planck-bestfit] fiducial h:", f"{fiducial['h']:.7f}")
    else:
        print("[planck-bestfit] fiducial theta:", f"{fiducial['theta']:.6f}")
    print("[planck-bestfit] wrote metadata:", meta_path)

    try:
        fm = cosmicfish.FisherMatrix(
            fiducialpars=fiducial,
            freepars=freepars,
            options=options,
            specifications=spec,
            observables=obs,
            surveyName="Euclid",
            cosmoModel="LCDM",
            latexnames=latexnames,
        )
        fish = fm.compute()
        out_summary = {
            "fisher_name": getattr(fish, "name", None),
            "fisher_file": getattr(fish, "file_name", None),
        }
        (outdir / "run_result.json").write_text(
            json.dumps(out_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("[planck-bestfit] SUCCESS:", getattr(fish, "name", "<no name>"))
        print("[planck-bestfit] file:", getattr(fish, "file_name", "<unknown>"))
        return 0
    except Exception:
        print("[planck-bestfit] FAILED. Traceback:\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
