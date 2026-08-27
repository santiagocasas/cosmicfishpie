#!/usr/bin/env python
# coding: utf-8

"""Fast CLASS(HP) vs CAMB(P3) Fisher comparison restricted to {w0, wa}.

This is a cheap, high-signal re-run of Case 01 of the Euclid w0waCDM validation
suite (photometric, nonlinear, w0waCDM), but with only ``w0`` and ``wa`` left
free instead of the full 7-parameter set. All other cosmological parameters
stay pinned at their fiducial values. In the 3-point derivative scheme this
cuts the number of forward/backward Boltzmann evaluations from 2*7+1=15 down
to 2*2+1=5, which is roughly 3x faster, while still exercising the exact same
nonlinear photometric w0waCDM code path that showed a 26.67% (Omegab-driven,
propagated through the marginalized w0/wa errors) discrepancy between
default-precision (DP) CLASS and P3 CAMB in Case 01.

By default this script compares:
  - CLASS with the new HP (high-precision) YAML:
    scripts/archive/legacy_yamls/class/mpvalidation_HP.yaml
  - CAMB with the existing P3 YAML (already high precision):
    cosmicfishpie/configs/default_boltzmann_yaml_files/camb/mpvalidation.yaml

See CONFIG_DIAGNOSIS.md and EUCLID_VALIDATION_FINDINGS.md at the repo root for
the full root-cause analysis and the rationale for the HP settings.

The script also writes a metadata JSON with the repository git commit and the
best-effort provenance of the CAMB/CLASS installs (PyPI wheel version, install
path, and — if available — a VCS commit id from a ``direct_url.json``). Neither
backend is git-commit-traceable in a standard PyPI install; this is recorded
explicitly rather than fabricated.

Example
-------
Default run (CLASS HP vs CAMB P3, photo, w0waCDM, w0+wa only)::

  uv run python scripts/archive/run_fisher_compare_w0wa_fast.py --compare

Compare against the original (failing) DP CLASS config instead, to reproduce
the regression on just 2 parameters::

  uv run python scripts/archive/run_fisher_compare_w0wa_fast.py \
    --yaml-a cosmicfishpie/configs/default_boltzmann_yaml_files/class/mpvalidation.yaml \
    --compare
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from cosmicfishpie.fishermatrix import cosmicfish

# Full fiducial cosmology from the Case 01 validation config
# (scripts/validation_configs/common_specs_w0waCDM.json). All of these stay
# fixed at fiducial; only the subset named in --params is varied.
DEFAULT_FIDUCIAL: dict[str, float] = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.815584,
    "mnu": 0.06,
    "Neff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
}

# Relative step sizes for the parameters that CAN be varied. Only the ones
# selected via --params are actually passed as freepars.
DEFAULT_STEPS: dict[str, float] = {
    "Omegam": 0.01,
    "Omegab": 0.01,
    "h": 0.01,
    "ns": 0.01,
    "sigma8": 0.01,
    "w0": 0.01,
    "wa": 0.01,
}

# Same photometric w0waCDM options block as Case 01
# (scripts/validation_configs/common_specs_w0waCDM.json), reused verbatim so
# this fast run is a faithful (but cheaper) re-run of that case.
DEFAULT_OPTIONS: dict = {
    "derivatives": "3PT",
    "cosmo_model": "w0waCDM",
    "nonlinear": True,
    "nonlinear_photo": True,
    "bfs8terms": False,
    "vary_bias_str": "lnb",
    "AP_effect": True,
    "FoG_switch": True,
    "GCsp_linear": False,
    "fix_cosmo_nl_terms": True,
    "Pshot_nuisance_fiducial": 0.0,
    "SUPPRESS_WARNINGS": True,
    "GCsp_Tracer": "matter",
    "GCph_Tracer": "matter",
    "ell_sampling": "accuracy",
    "ShareDeltaNeff": True,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode("utf-8").strip())
    except Exception:
        return None


def _git_commit_full(path: Path) -> str | None:
    """Full (non-abbreviated) commit hash at an arbitrary path, or None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _git_remote_url(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=str(path),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _package_provenance(module_name: str, dist_name: str | None = None) -> dict:
    """Best-effort provenance for an installed backend package.

    Standard PyPI-wheel installs do not carry a git commit. We report the
    package version, install location, and — only if present — a VCS commit
    id recovered from ``direct_url.json`` (written by pip for VCS/editable
    installs). Absence of a commit is reported explicitly, never fabricated.
    """
    dist_name = dist_name or module_name
    info: dict = {
        "module": module_name,
        "importable": False,
        "version": None,
        "module_file": None,
        "dist_version": None,
        "install_source": None,
        "vcs_commit": None,
        "note": None,
    }
    try:
        mod = __import__(module_name)
        info["importable"] = True
        info["module_file"] = getattr(mod, "__file__", None)
        info["version"] = getattr(mod, "__version__", None)
    except Exception as exc:
        info["note"] = f"import failed: {exc}"
        return info

    try:
        dist = importlib.metadata.distribution(dist_name)
        info["dist_version"] = dist.version
    except importlib.metadata.PackageNotFoundError:
        info["note"] = f"importlib.metadata could not find distribution '{dist_name}'"
        return info

    try:
        direct_url_text = dist.read_text("direct_url.json")
    except Exception:
        direct_url_text = None

    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except Exception:
            direct_url = {}
        vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
        dir_info = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
        url = direct_url.get("url") if isinstance(direct_url, dict) else None
        if isinstance(vcs_info, dict) and vcs_info.get("commit_id"):
            # pip installed directly from a VCS URL (e.g. `pip install git+...`).
            info["install_source"] = "vcs"
            info["vcs_commit"] = vcs_info.get("commit_id")
            info["vcs_url"] = url
        elif (
            isinstance(dir_info, dict)
            and dir_info.get("editable")
            and isinstance(url, str)
            and url.startswith("file://")
        ):
            # Editable local-path install (e.g. `uv pip install -e /path/to/fork`).
            # pip does not record a commit for this case, so resolve it ourselves
            # by treating the source path as its own git repo.
            local_path = Path(url[len("file://") :])
            commit = _git_commit_full(local_path)
            info["install_source"] = "editable_local_path"
            info["local_path"] = str(local_path)
            if commit is not None:
                info["vcs_commit"] = commit
                info["vcs_dirty"] = _git_dirty(local_path)
                info["vcs_remote"] = _git_remote_url(local_path)
            else:
                info["note"] = (
                    f"Editable install from {local_path}, but it is not a git "
                    "repository (or git is unavailable); no commit recorded."
                )
        else:
            info["install_source"] = "direct_url (non-vcs, e.g. local path or URL)"
            info["note"] = "direct_url.json present but no vcs_info.commit_id"
    else:
        info["install_source"] = "pypi_wheel"
        info["note"] = "Installed from a PyPI wheel; no git commit is recorded by pip."

    return info


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
        "repo": {
            "git_commit": _git_commit(repo_root),
            "git_dirty": _git_dirty(repo_root),
        },
        "backends": {
            "camb": _package_provenance("camb"),
            "classy": _package_provenance("classy", dist_name="classy"),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "args": vars(args),
        "resolved": resolved,
    }
    outpath.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return outpath


def _infer_yaml_key(code: str) -> str:
    return {
        "class": "class_config_yaml",
        "camb": "camb_config_yaml",
        "symbolic": "symbolic_config_yaml",
    }.get(code, f"{code}_config_yaml")


def _load_common_specs(path: Path | None) -> dict | None:
    if path is None:
        return None
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
    feedback: int,
) -> dict:
    return {
        "accuracy": accuracy,
        "feedback": feedback,
        "results_dir": str(results_dir) + "/",
        "specs_dir": specs_dir,
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "survey_name_spectro": False,
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
        cosmoModel=options.get("cosmo_model", "w0waCDM"),
        surveyName=options.get("survey_name", "Euclid"),
    )
    res = fm.compute()
    return getattr(res, "file_name", None)


def main() -> int:
    repo_root = _repo_root()
    cfg_dir = repo_root / "cosmicfishpie" / "configs"
    default_specs_dir = str(cfg_dir / "default_survey_specifications") + "/"
    default_yaml_a = str(
        repo_root / "scripts" / "archive" / "legacy_yamls" / "class" / "mpvalidation_HP.yaml"
    )
    default_yaml_b = str(cfg_dir / "default_boltzmann_yaml_files" / "camb" / "mpvalidation.yaml")
    default_common_specs = str(
        repo_root / "scripts" / "validation_configs" / "common_specs_w0waCDM.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Fast 2-param (w0, wa) photometric w0waCDM Fisher comparison: "
            "CLASS-HP vs CAMB-P3, with full backend/repo provenance metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["photo", "spectro"],
        default="photo",
        help="Which pipeline to run: photo=GCph+WL, spectro=GCsp",
    )
    parser.add_argument(
        "--params",
        default="w0,wa",
        help="Comma-separated list of parameters to leave free (varied); "
        "everything else stays fixed at fiducial",
    )
    parser.add_argument("--code-a", default="class", help="Backend for run A")
    parser.add_argument("--code-b", default="camb", help="Backend for run B")
    parser.add_argument(
        "--yaml-a",
        default=default_yaml_a,
        help="Boltzmann YAML for run A (default: CLASS HP)",
    )
    parser.add_argument(
        "--yaml-b",
        default=default_yaml_b,
        help="Boltzmann YAML for run B (default: CAMB P3)",
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
        "--common-specs",
        default=default_common_specs,
        help="JSON with fiducialpars/freepars/options to merge over the built-in "
        "w0waCDM defaults (default: the exact Case 01 validation specs)",
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
        "--outdir",
        default=None,
        help="Output directory (default: scripts/benchmark_results/compare_w0wa_fast_<timestamp>)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Run scripts/compare_fishers_in_dir.py after computing both Fishers (default: on)",
    )
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        help="Skip the compare_fishers_in_dir.py step",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Pass/fail threshold in percent for max |sigma ratio - 1| on the varied params",
    )
    args = parser.parse_args()

    if args.omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    varied_params = [p.strip() for p in args.params.split(",") if p.strip()]
    if not varied_params:
        raise SystemExit("--params must list at least one parameter (e.g. 'w0,wa')")
    unknown = [p for p in varied_params if p not in DEFAULT_STEPS]
    if unknown:
        raise SystemExit(
            f"--params contains parameters with no known default step size: {unknown}. "
            f"Known: {sorted(DEFAULT_STEPS)}"
        )

    run_id = _make_run_id()
    outdir = (
        Path(args.outdir)
        if args.outdir
        else repo_root / "scripts" / "benchmark_results" / f"compare_w0wa_fast_{run_id}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    common_specs_path = (
        Path(args.common_specs).expanduser().resolve() if args.common_specs else None
    )
    if common_specs_path is not None and not common_specs_path.is_file():
        raise SystemExit(f"--common-specs file not found: {common_specs_path}")
    common_specs = _load_common_specs(common_specs_path)
    common_fiducial = common_specs.get("fiducialpars") if isinstance(common_specs, dict) else None
    common_options = common_specs.get("options") if isinstance(common_specs, dict) else None

    fiducial = DEFAULT_FIDUCIAL.copy()
    if isinstance(common_fiducial, dict):
        fiducial.update(common_fiducial)

    # Only the requested params are left free; everything else in `fiducial`
    # stays pinned. This is the key speed-up vs. the full 7-param Case 01.
    freepars = {p: DEFAULT_STEPS[p] for p in varied_params}

    if args.mode == "photo":
        observables = ["GCph", "WL"]
    else:
        observables = ["GCsp"]

    base_options = DEFAULT_OPTIONS.copy()
    if isinstance(common_options, dict):
        base_options.update(common_options)

    yaml_key_a = args.yaml_key_a or _infer_yaml_key(args.code_a)
    yaml_key_b = args.yaml_key_b or _infer_yaml_key(args.code_b)

    print("[w0wa-fast] repo_root:", repo_root)
    print("[w0wa-fast] outdir:", outdir)
    print("[w0wa-fast] mode:", args.mode, "observables:", observables)
    print("[w0wa-fast] varied params (freepars):", freepars)
    print("[w0wa-fast] fixed fiducial:", fiducial)
    print("[w0wa-fast] run A:", args.code_a, "yaml_key:", yaml_key_a, "yaml:", args.yaml_a)
    print("[w0wa-fast] run B:", args.code_b, "yaml_key:", yaml_key_b, "yaml:", args.yaml_b)

    meta_path = _write_run_metadata(
        outdir=outdir,
        args=args,
        repo_root=repo_root,
        resolved={
            "specs_dir": default_specs_dir,
            "common_specs_json": str(common_specs_path) if common_specs_path else None,
            "yaml_key_a": yaml_key_a,
            "yaml_key_b": yaml_key_b,
            "yaml_a": args.yaml_a,
            "yaml_b": args.yaml_b,
            "observables": observables,
            "fiducialpars": fiducial,
            "freepars": freepars,
            "options": base_options,
        },
    )
    print("[w0wa-fast] Wrote run metadata:", meta_path)

    prefix = f"compare_w0wa_fast_{run_id}_"

    print("[w0wa-fast] Running Fisher A (%s)..." % args.code_a)
    opts_a = _build_base_options(
        code=args.code_a,
        outroot=prefix + "A_",
        results_dir=outdir,
        specs_dir=default_specs_dir,
        accuracy=args.accuracy,
        feedback=args.feedback,
    )
    opts_a.update(base_options)
    opts_a["accuracy"] = args.accuracy
    opts_a["feedback"] = args.feedback
    opts_a["code"] = args.code_a
    opts_a["outroot"] = prefix + "A_"
    opts_a["results_dir"] = str(outdir) + "/"
    opts_a[yaml_key_a] = args.yaml_a
    t0 = time.time()
    a_txt = _run_fisher(
        options=opts_a, observables=observables, fiducial=fiducial, freepars=freepars
    )
    a_time = time.time() - t0
    print(f"[w0wa-fast] A matrix: {a_txt} ({a_time:.1f}s)")

    print("[w0wa-fast] Running Fisher B (%s)..." % args.code_b)
    opts_b = _build_base_options(
        code=args.code_b,
        outroot=prefix + "B_",
        results_dir=outdir,
        specs_dir=default_specs_dir,
        accuracy=args.accuracy,
        feedback=args.feedback,
    )
    opts_b.update(base_options)
    opts_b["accuracy"] = args.accuracy
    opts_b["feedback"] = args.feedback
    opts_b["code"] = args.code_b
    opts_b["outroot"] = prefix + "B_"
    opts_b["results_dir"] = str(outdir) + "/"
    opts_b[yaml_key_b] = args.yaml_b
    t0 = time.time()
    b_txt = _run_fisher(
        options=opts_b, observables=observables, fiducial=fiducial, freepars=freepars
    )
    b_time = time.time() - t0
    print(f"[w0wa-fast] B matrix: {b_txt} ({b_time:.1f}s)")

    if not args.compare:
        print("[w0wa-fast] Done. To compare:")
        print(
            f"  uv run python scripts/compare_fishers_in_dir.py {outdir} "
            f"--fom-params {','.join(varied_params)}"
        )
        return 0

    compare_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "compare_fishers_in_dir.py"),
        str(outdir),
        "--fom-params",
        ",".join(varied_params),
    ]
    print("[w0wa-fast] Comparing Fishers:")
    print(" ", " ".join(compare_cmd))
    subprocess.check_call(compare_cmd)

    compare_jsons = sorted(outdir.glob("compare_fishers_*.json"), key=lambda p: p.stat().st_mtime)
    if not compare_jsons:
        print("[w0wa-fast][WARN] No compare_fishers_*.json produced; skipping PASS/FAIL summary.")
        return 0
    latest = compare_jsons[-1]
    data = json.loads(latest.read_text(encoding="utf-8"))
    pairwise = data.get("pairwise") or []
    if not pairwise:
        print("[w0wa-fast][WARN] compare JSON has no pairwise entries; skipping PASS/FAIL summary.")
        return 0

    ratios = pairwise[0].get("analysis", {}).get("param_sigma_ratio", {})
    print()
    print("=" * 72)
    print(f"[w0wa-fast] PASS/FAIL summary (threshold: {args.threshold:.2f}%)")
    print("=" * 72)
    worst_dev = 0.0
    any_fail = False
    for name in varied_params:
        vals = ratios.get(name)
        if not vals or vals.get("ratio_b_over_a") is None:
            print(f"  {name:>6s}: N/A (missing sigma ratio)")
            continue
        ratio = vals["ratio_b_over_a"]
        dev_pct = 100.0 * abs(ratio - 1.0)
        worst_dev = max(worst_dev, dev_pct)
        status = "PASS" if dev_pct <= args.threshold else "FAIL"
        any_fail = any_fail or status == "FAIL"
        print(
            f"  {name:>6s}: sigma_A={vals['sigma_a']:.6g}  sigma_B={vals['sigma_b']:.6g}  "
            f"ratio(B/A)={ratio:.4f}  deviation={dev_pct:.2f}%  [{status}]"
        )
    print("-" * 72)
    overall = "FAIL" if any_fail else "PASS"
    print(f"[w0wa-fast] Overall: {overall} (worst deviation: {worst_dev:.2f}%)")
    print(f"[w0wa-fast] Timing: A={a_time:.1f}s B={b_time:.1f}s")
    print(f"[w0wa-fast] Compare JSON: {latest}")
    print(f"[w0wa-fast] Run metadata: {meta_path}")
    print("=" * 72)

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
