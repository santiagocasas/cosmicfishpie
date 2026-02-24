#!/usr/bin/env python
# coding: utf-8

"""CMB Fisher smoke runner.

This script mirrors the LSS script style: it builds a minimal config and runs a
single FisherMatrix with CMB observables to expose any integration errors.

It intentionally uses `surveyName='Euclid'` and injects the required CMB specs
via the `specifications` dict, so it does not depend on a Planck survey YAML.

Examples
--------
CAMB CMB_T+E quick smoke:

  uv run python scripts/run_cmb_fisher_smoke.py --lmax 300 --compare-dir scripts/benchmark_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from cosmicfishpie.fishermatrix import cosmicfish


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _git_commit(repo_root: Path) -> str | None:
    # Avoid subprocess; git is optional in runtime env.
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


def _infer_yaml_key(code: str) -> str:
    return {
        "camb": "camb_config_yaml",
        "class": "class_config_yaml",
        "symbolic": "symbolic_config_yaml",
    }.get(code, f"{code}_config_yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a dict at top-level: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _enable_cmb_in_boltzmann_yaml(*, code: str, yaml_in: Path, yaml_out: Path) -> Path:
    cfg = _load_yaml(yaml_in)
    cfg.setdefault("COSMO_SETTINGS", {})
    if not isinstance(cfg["COSMO_SETTINGS"], dict):
        raise ValueError(f"COSMO_SETTINGS must be a dict in {yaml_in}")

    if code == "camb":
        cfg["COSMO_SETTINGS"]["WantCls"] = True
        cfg["COSMO_SETTINGS"]["Want_CMB"] = True
        cfg["COSMO_SETTINGS"]["Want_CMB_lensing"] = False
        cfg["COSMO_SETTINGS"]["Want_cl_2D_array"] = False
    elif code == "class":
        # CLASS needs tCl output for CMB spectra.
        out = cfg["COSMO_SETTINGS"].get("output", "")
        if not isinstance(out, str):
            out = str(out)
        tokens = [t.strip() for t in out.split(",") if t.strip()]
        for need in ("tCl", "pCl"):
            if need not in tokens:
                tokens.append(need)
        cfg["COSMO_SETTINGS"]["output"] = ",".join(tokens)
    else:
        raise ValueError(f"Unsupported code for CMB YAML enable: {code}")

    _write_yaml(yaml_out, cfg)
    return yaml_out


def _write_metadata(outdir: Path, payload: dict[str, Any]) -> None:
    (outdir / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a CMB Fisher smoke test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--code", choices=["camb", "class"], default="camb")
    parser.add_argument(
        "--boltzmann-yaml",
        default=None,
        help="Path to backend YAML (default: package default for selected --code)",
    )
    parser.add_argument(
        "--spec-yaml",
        default=None,
        help="Path to a survey specifications YAML (expects a top-level 'specifications' dict)",
    )
    parser.add_argument(
        "--write-enabled-yaml",
        action="store_true",
        help="Write a copy of the YAML with CMB outputs enabled into the output dir",
    )
    parser.add_argument(
        "--lmin",
        type=int,
        default=None,
        help="Minimum ell (default: from --spec-yaml or 30)",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="Maximum ell (default: from --spec-yaml or 600)",
    )
    parser.add_argument(
        "--fsky",
        type=float,
        default=None,
        help="f_sky used for CMB_T/E/B (default: from --spec-yaml or 0.65)",
    )
    parser.add_argument(
        "--beam-arcmin",
        type=float,
        default=None,
        help="Gaussian beam FWHM in arcmin (single channel; default: from --spec-yaml or 7.1)",
    )
    parser.add_argument(
        "--temp-sens",
        type=float,
        default=None,
        help="Temperature sensitivity in uK-arcmin (default: from --spec-yaml or 43.0)",
    )
    parser.add_argument(
        "--pol-sens",
        type=float,
        default=None,
        help="Polarization sensitivity in uK-arcmin (default: from --spec-yaml or 66.0)",
    )
    parser.add_argument(
        "--observables",
        default="CMB_T,CMB_E",
        help="Comma-separated list from: CMB_T,CMB_E,CMB_B",
    )
    parser.add_argument("--accuracy", type=int, default=1)
    parser.add_argument("--feedback", type=int, default=2)
    parser.add_argument("--derivatives", default="3PT")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: scripts/benchmark_results/cmb_smoke_<timestamp>)",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    run_id = _timestamp_id()
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else (repo_root / "scripts" / "benchmark_results" / f"cmb_smoke_{run_id}")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_dir = repo_root / "cosmicfishpie" / "configs" / "default_boltzmann_yaml_files"
    default_yaml = cfg_dir / args.code / "default.yaml"
    yaml_in = (
        Path(args.boltzmann_yaml).expanduser().resolve() if args.boltzmann_yaml else default_yaml
    )
    if not yaml_in.exists():
        raise SystemExit(f"Boltzmann YAML not found: {yaml_in}")

    yaml_key = _infer_yaml_key(args.code)
    yaml_path_to_use = str(yaml_in)
    yaml_enabled_out = None
    if args.write_enabled_yaml:
        yaml_enabled_out = outdir / f"{args.code}_cmb_enabled.yaml"
        _enable_cmb_in_boltzmann_yaml(
            code=args.code,
            yaml_in=yaml_in,
            yaml_out=yaml_enabled_out,
        )
        yaml_path_to_use = str(yaml_enabled_out)

    obs = [o.strip() for o in args.observables.split(",") if o.strip()]
    allowed = {"CMB_T", "CMB_E", "CMB_B"}
    bad = [o for o in obs if o not in allowed]
    if bad:
        raise SystemExit(f"Invalid observables: {bad} (allowed: {sorted(allowed)})")

    cmb_specs: dict[str, Any] = {}
    if args.spec_yaml:
        spec_path = Path(args.spec_yaml).expanduser().resolve()
        if not spec_path.exists():
            raise SystemExit(f"Spec YAML not found: {spec_path}")
        loaded = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit(f"Spec YAML must be a dict at top-level: {spec_path}")
        cmb_specs = loaded.get("specifications", loaded)
        if not isinstance(cmb_specs, dict):
            raise SystemExit(f"Spec YAML must contain a dict 'specifications': {spec_path}")

    # Apply conservative defaults if missing, and allow CLI overrides when provided.
    if args.lmin is not None:
        cmb_specs["lmin_CMB"] = int(args.lmin)
    cmb_specs.setdefault("lmin_CMB", 30)

    if args.lmax is not None:
        cmb_specs["lmax_CMB"] = int(args.lmax)
    cmb_specs.setdefault("lmax_CMB", 600)

    if args.fsky is not None:
        fsky = float(args.fsky)
        cmb_specs["fsky_CMB_T"] = fsky
        cmb_specs["fsky_CMB_E"] = fsky
        cmb_specs["fsky_CMB_B"] = fsky
    cmb_specs.setdefault("fsky_CMB_T", 0.65)
    cmb_specs.setdefault("fsky_CMB_E", cmb_specs.get("fsky_CMB_T", 0.65))
    cmb_specs.setdefault("fsky_CMB_B", cmb_specs.get("fsky_CMB_T", 0.65))

    if args.beam_arcmin is not None:
        cmb_specs["CMB_fwhm"] = [float(args.beam_arcmin)]
    cmb_specs.setdefault("CMB_fwhm", [7.1])

    if args.temp_sens is not None:
        cmb_specs["CMB_temp_sens"] = [float(args.temp_sens)]
    cmb_specs.setdefault("CMB_temp_sens", [43.0])

    if args.pol_sens is not None:
        cmb_specs["CMB_pol_sens"] = [float(args.pol_sens)]
    cmb_specs.setdefault("CMB_pol_sens", [66.0])

    fiducial = {
        "Omegam": 0.32,
        "Omegab": 0.05,
        "h": 0.67,
        "ns": 0.96,
        "sigma8": 0.815584,
        "mnu": 0.06,
        "Neff": 3.044,
    }
    freepars = {
        "Omegam": 0.01,
        "Omegab": 0.01,
        "h": 0.01,
        "ns": 0.01,
        "sigma8": 0.01,
    }

    options = {
        "code": args.code,
        yaml_key: yaml_path_to_use,
        "feedback": int(args.feedback),
        "derivatives": str(args.derivatives),
        "results_dir": str(outdir) + "/",
        "outroot": "cmb_smoke_",
        "accuracy": int(args.accuracy),
    }

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": list(sys.argv),
        "git_commit": _git_commit(repo_root),
        "cwd": os.getcwd(),
        "args": vars(args),
        "resolved": {
            "outdir": str(outdir),
            "yaml_key": yaml_key,
            "yaml_in": str(yaml_in),
            "yaml_enabled_out": str(yaml_enabled_out) if yaml_enabled_out else None,
            "yaml_used": yaml_path_to_use,
            "observables": obs,
            "cmb_specs": cmb_specs,
        },
    }
    _write_metadata(outdir, meta)

    print("[cmb] outdir:", outdir)
    print("[cmb] code:", args.code)
    print("[cmb] yaml_key:", yaml_key)
    print("[cmb] yaml_used:", yaml_path_to_use)
    print("[cmb] observables:", obs)
    print("[cmb] ell range:", cmb_specs.get("lmin_CMB"), "..", cmb_specs.get("lmax_CMB"))
    print("[cmb] Wrote metadata:", outdir / "run_metadata.json")
    if args.write_enabled_yaml:
        print("[cmb] Wrote enabled YAML:", yaml_enabled_out)

    try:
        fm = cosmicfish.FisherMatrix(
            fiducialpars=fiducial,
            freepars=freepars,
            options=options,
            specifications=cmb_specs,
            observables=obs,
            surveyName="Euclid",
            cosmoModel="LCDM",
        )
        res = fm.compute()
        print("[cmb] SUCCESS:", getattr(res, "name", "<no name>"))
        return 0
    except Exception:
        print("[cmb] FAILED. Traceback:\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
