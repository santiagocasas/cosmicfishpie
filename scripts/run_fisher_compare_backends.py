#!/usr/bin/env python
# coding: utf-8

"""Run and compare Fisher matrices across backends and Boltzmann YAML configs.

Examples
--------
Photometric (GCph+WL), CLASS vs CAMB using nuvalidation YAMLs:

  uv run python scripts/run_fisher_compare_backends.py \
    --mode photo \
    --code-a class --yaml-a cosmicfishpie/configs/default_boltzmann_yaml_files/class/nuvalidation_photo.yaml \
    --code-b camb  --yaml-b cosmicfishpie/configs/default_boltzmann_yaml_files/camb/nuvalidation.yaml \
    --compare --plot

Spectroscopic (GCsp), CLASS vs CAMB:

  uv run python scripts/run_fisher_compare_backends.py \
    --mode spectro \
    --code-a class --yaml-a cosmicfishpie/configs/default_boltzmann_yaml_files/class/nuvalidation_spectro.yaml \
    --code-b camb  --yaml-b cosmicfishpie/configs/default_boltzmann_yaml_files/camb/nuvalidation.yaml \
    --compare

Notes
-----
- The compare step uses scripts/compare_fishers_in_dir.py which also diffs the YAML
  files referenced by each run's specs JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cosmicfishpie.fishermatrix import cosmicfish

DEFAULT_FIDUCIAL = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.815584,
    "mnu": 0.06,
    "Neff": 3.044,
}

DEFAULT_FREEPARS = {
    "Omegam": 0.01,
    "Omegab": 0.01,
    "h": 0.01,
    "ns": 0.01,
    "sigma8": 0.01,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _infer_yaml_key(code: str) -> str:
    return {
        "class": "class_config_yaml",
        "camb": "camb_config_yaml",
        "symbolic": "symbolic_config_yaml",
    }.get(code, f"{code}_config_yaml")


def _default_paths(repo_root: Path) -> dict[str, str]:
    cfg = repo_root / "cosmicfishpie" / "configs"
    return {
        "specs_dir": str(cfg / "default_survey_specifications") + "/",
        "class_yaml_photo": str(
            cfg / "default_boltzmann_yaml_files" / "class" / "nuvalidation_photo.yaml"
        ),
        "class_yaml_spectro": str(
            cfg / "default_boltzmann_yaml_files" / "class" / "nuvalidation_spectro.yaml"
        ),
        "camb_yaml": str(cfg / "default_boltzmann_yaml_files" / "camb" / "nuvalidation.yaml"),
        "symbolic_yaml": str(cfg / "default_boltzmann_yaml_files" / "symbolic" / "default.yaml"),
    }


def _make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _write_run_metadata(
    *,
    outdir: Path,
    args: argparse.Namespace,
    repo_root: Path,
    resolved: dict,
) -> Path:
    outpath = outdir / "run_metadata.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": list(sys.argv),
        "git_commit": _git_commit(repo_root),
        "python": sys.version,
        "cwd": os.getcwd(),
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "args": vars(args),
        "resolved": resolved,
    }
    outpath.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return outpath


def _load_common_specs(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read common specs JSON: {path} ({exc})")


def _build_base_options(
    *,
    code: str,
    outroot: str,
    results_dir: Path,
    specs_dir: str,
    accuracy: int,
    survey_name_photo: str | bool,
    survey_name_spectro: str | bool,
    feedback: int,
) -> dict:
    return {
        "accuracy": accuracy,
        "feedback": feedback,
        "derivatives": "3PT",
        "results_dir": str(results_dir) + "/",
        "specs_dir": specs_dir,
        "survey_name": "Euclid",
        "survey_name_photo": survey_name_photo,
        "survey_name_spectro": survey_name_spectro,
        "cosmo_model": "LCDM",
        "code": code,
        "outroot": outroot,
    }


def _run_fisher(
    *,
    options: dict,
    observables: list[str],
    fiducial: dict[str, float],
    freepars: dict[str, float],
) -> str | None:
    fm = cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options.get("cosmo_model", "LCDM"),
        surveyName=options.get("survey_name", "Euclid"),
    )
    res = fm.compute()
    return getattr(res, "file_name", None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run two Fishers and compare outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["photo", "spectro"],
        default="photo",
        help="Which pipeline to run: photo=GCph+WL, spectro=GCsp",
    )
    parser.add_argument("--accuracy", type=int, default=1)
    parser.add_argument("--feedback", type=int, default=1)
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Set OMP_NUM_THREADS for the run (default: leave unchanged)",
    )
    parser.add_argument(
        "--code-a",
        default="class",
        help="Backend name for run A (e.g. class, camb, symbolic)",
    )
    parser.add_argument(
        "--code-b",
        default="camb",
        help="Backend name for run B (e.g. class, camb, symbolic)",
    )
    parser.add_argument(
        "--yaml-a",
        default=None,
        help="Boltzmann YAML path for run A (default depends on code/mode)",
    )
    parser.add_argument(
        "--yaml-b",
        default=None,
        help="Boltzmann YAML path for run B (default depends on code/mode)",
    )
    parser.add_argument(
        "--yaml-key-a",
        default=None,
        help="Options key used to store yaml-a (default inferred from code-a)",
    )
    parser.add_argument(
        "--yaml-key-b",
        default=None,
        help="Options key used to store yaml-b (default inferred from code-b)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: scripts/benchmark_results/compare_<mode>_<timestamp>)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run scripts/compare_fishers_in_dir.py after computing both Fishers",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run scripts/plot_compare_fishers.py on the produced compare JSON (requires --compare)",
    )
    parser.add_argument(
        "--fom-params",
        default="Omegam,sigma8",
        help="FoM parameters passed to compare_fishers_in_dir.py",
    )
    parser.add_argument(
        "--common-specs",
        default=None,
        help="Path to JSON with fiducialpars/freepars/options (e.g. *_FM_specs.json)",
    )
    args = parser.parse_args()

    if args.omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    repo_root = _repo_root()
    paths = _default_paths(repo_root)
    run_id = _make_run_id()

    outdir = (
        Path(args.outdir)
        if args.outdir
        else repo_root / "scripts" / "benchmark_results" / f"compare_{args.mode}_{run_id}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    common_specs_path = (
        Path(args.common_specs).expanduser().resolve() if args.common_specs else None
    )
    common_specs = _load_common_specs(common_specs_path) if common_specs_path else None
    common_fiducial = common_specs.get("fiducialpars") if isinstance(common_specs, dict) else None
    common_freepars = common_specs.get("freepars") if isinstance(common_specs, dict) else None
    common_options = common_specs.get("options") if isinstance(common_specs, dict) else None

    fiducial = DEFAULT_FIDUCIAL.copy()
    if isinstance(common_fiducial, dict):
        fiducial.update(common_fiducial)
    freepars = DEFAULT_FREEPARS.copy()
    if isinstance(common_freepars, dict):
        freepars.update(common_freepars)

    if args.mode == "photo":
        observables = ["GCph", "WL"]
        survey_name_photo: str | bool = "Euclid-Photometric-ISTF-Pessimistic"
        survey_name_spectro: str | bool = False
    else:
        observables = ["GCsp"]
        survey_name_photo = False
        survey_name_spectro = "Euclid-Spectroscopic-ISTF-Pessimistic"

    def _default_yaml_for(code: str) -> str | None:
        if code == "class":
            return (
                paths["class_yaml_photo"] if args.mode == "photo" else paths["class_yaml_spectro"]
            )
        if code == "camb":
            return paths["camb_yaml"]
        if code == "symbolic":
            return paths["symbolic_yaml"]
        return None

    yaml_key_a = args.yaml_key_a or _infer_yaml_key(args.code_a)
    yaml_key_b = args.yaml_key_b or _infer_yaml_key(args.code_b)
    yaml_a = args.yaml_a or _default_yaml_for(args.code_a)
    yaml_b = args.yaml_b or _default_yaml_for(args.code_b)

    if yaml_a is None:
        raise SystemExit(
            f"No default yaml-a for code-a='{args.code_a}'. Provide --yaml-a explicitly."
        )
    if yaml_b is None:
        raise SystemExit(
            f"No default yaml-b for code-b='{args.code_b}'. Provide --yaml-b explicitly."
        )

    inferred_a = _infer_yaml_key(args.code_a)
    inferred_b = _infer_yaml_key(args.code_b)
    if yaml_key_a != inferred_a:
        print(
            f"[compare][WARN] yaml-key-a='{yaml_key_a}' does not match inferred '{inferred_a}' for code-a='{args.code_a}'. "
            "Backend may ignore it."
        )
    if yaml_key_b != inferred_b:
        print(
            f"[compare][WARN] yaml-key-b='{yaml_key_b}' does not match inferred '{inferred_b}' for code-b='{args.code_b}'. "
            "Backend may ignore it."
        )

    print("[compare] repo_root:", repo_root)
    print("[compare] outdir:", outdir)
    if args.omp_threads is not None:
        print("[compare] OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("[compare] specs_dir:", paths["specs_dir"])
    print("[compare] mode:", args.mode, "observables:", observables)
    print("[compare] run A:", args.code_a, "yaml_key:", yaml_key_a, "yaml:", yaml_a)
    print("[compare] run B:", args.code_b, "yaml_key:", yaml_key_b, "yaml:", yaml_b)

    meta_path = _write_run_metadata(
        outdir=outdir,
        args=args,
        repo_root=repo_root,
        resolved={
            "specs_dir": paths["specs_dir"],
            "common_specs_json": str(common_specs_path) if common_specs_path else None,
            "yaml_key_a": yaml_key_a,
            "yaml_key_b": yaml_key_b,
            "yaml_a": yaml_a,
            "yaml_b": yaml_b,
            "observables": observables,
            "survey_name_photo": survey_name_photo,
            "survey_name_spectro": survey_name_spectro,
        },
    )
    print("[compare] Wrote run metadata:", meta_path)

    prefix = f"compare_{args.mode}_{run_id}_"

    print("[compare] Running Fisher A...")
    opts_a = _build_base_options(
        code=args.code_a,
        outroot=prefix + "A_",
        results_dir=outdir,
        specs_dir=paths["specs_dir"],
        accuracy=args.accuracy,
        survey_name_photo=survey_name_photo,
        survey_name_spectro=survey_name_spectro,
        feedback=args.feedback,
    )
    if isinstance(common_options, dict):
        opts_a.update(common_options)
    opts_a["accuracy"] = args.accuracy
    opts_a["feedback"] = args.feedback
    opts_a["code"] = args.code_a
    opts_a["outroot"] = prefix + "A_"
    opts_a["results_dir"] = str(outdir) + "/"
    opts_a[yaml_key_a] = yaml_a
    a_txt = _run_fisher(
        options=opts_a,
        observables=observables,
        fiducial=fiducial,
        freepars=freepars,
    )
    print("[compare] A matrix:", a_txt)

    print("[compare] Running Fisher B...")
    opts_b = _build_base_options(
        code=args.code_b,
        outroot=prefix + "B_",
        results_dir=outdir,
        specs_dir=paths["specs_dir"],
        accuracy=args.accuracy,
        survey_name_photo=survey_name_photo,
        survey_name_spectro=survey_name_spectro,
        feedback=args.feedback,
    )
    if isinstance(common_options, dict):
        opts_b.update(common_options)
    opts_b["accuracy"] = args.accuracy
    opts_b["feedback"] = args.feedback
    opts_b["code"] = args.code_b
    opts_b["outroot"] = prefix + "B_"
    opts_b["results_dir"] = str(outdir) + "/"
    opts_b[yaml_key_b] = yaml_b
    b_txt = _run_fisher(
        options=opts_b,
        observables=observables,
        fiducial=fiducial,
        freepars=freepars,
    )
    print("[compare] B matrix:", b_txt)

    if not args.compare:
        print("[compare] Done. To compare:")
        print(
            f"  uv run python scripts/compare_fishers_in_dir.py {outdir} --fom-params {args.fom_params}"
        )
        return 0

    compare_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "compare_fishers_in_dir.py"),
        str(outdir),
        "--fom-params",
        args.fom_params,
    ]
    print("[compare] Comparing Fishers:")
    print(" ", " ".join(compare_cmd))
    subprocess.check_call(compare_cmd)

    if not args.plot:
        return 0

    compare_jsons = sorted(outdir.glob("compare_fishers_*.json"), key=lambda p: p.stat().st_mtime)
    if not compare_jsons:
        raise SystemExit(f"No compare_fishers_*.json found in {outdir}")
    latest = compare_jsons[-1]

    plot_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "plot_compare_fishers.py"),
        str(latest),
    ]
    print("[compare] Plotting comparison:")
    print(" ", " ".join(plot_cmd))
    subprocess.check_call(plot_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
