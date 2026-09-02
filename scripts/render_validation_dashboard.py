#!/usr/bin/env python
# coding: utf-8

"""Render a clean HTML dashboard for the CAMB-vs-CLASS backend validation cases.

Reads the case definitions from ``scripts/validation_configs/compare_run_config.env_*``
and the matching outputs from ``scripts/benchmark_results/compare_*`` (as produced by
``compare_backends_report.sh`` / ``run_fisher_compare_backends.py``), and writes:

  <out-dir>/index.html            landing page: case #, model, probe, max deviation, gate
  <out-dir>/case_<ID>.html        per-case detail: full parameter sigma table + metadata
  <out-dir>/specs/case_<ID>_*.html  readable YAML/common/generated Fisher specifications

Case IDs are dotted hierarchical strings (e.g. "01.2.0", "03.2.1"), read from
config filenames compare_run_config.env_<ID>_<description>.

Design goals (per user request):
  - The landing page never shows config hashes, folder names, or file paths.
  - Clicking a case shows every varied parameter's CAMB/CLASS sigma and deviation,
    split into cosmological parameters and nuisance parameters.
  - Further links open the exact YAML, common specs, and generated A/B Fisher specs.
  - The correct latest CAMB-vs-CLASS pair is selected explicitly (by matching the
    newest shared run timestamp between the "_A__" and "_B__" outputs), instead of
    picking the first pairwise entry in the compare JSON (which can be a stale
    same-code rerun and report a spurious 0.000% deviation).

This script is read-only with respect to the raw benchmark_results folders (except
writing into --out-dir, which defaults to a subdirectory of benchmark_results).

Pass ``--serve`` to keep a local HTTP server running after generation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

COSMO_PARAM_ORDER = ["Omegam", "Omegab", "h", "ns", "sigma8", "mnu", "Neff", "w0", "wa"]
COSMO_PARAM_SET = set(COSMO_PARAM_ORDER)

TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})_([AB])__")


def esc(s: Any) -> str:
    text = "" if s is None else str(s)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_env_value(raw: str, repo_root: Path) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    value = value.replace("${REPO_ROOT}", str(repo_root))
    return value


def _parse_env_config(path: Path, repo_root: Path) -> dict[str, str]:
    """Minimal KEY="VALUE" / KEY=VALUE parser for these env-style config files."""
    out: dict[str, str] = {}
    description = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not description and stripped.startswith("#"):
            description = stripped.lstrip("#").strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", stripped)
        if not m:
            continue
        key, raw_val = m.group(1), m.group(2)
        out[key] = _resolve_env_value(raw_val, repo_root)
    out["_description"] = description
    return out


@dataclass
class CaseDef:
    number: str
    config_path: Path
    description: str
    mode: str
    code_a: str
    code_b: str
    yaml_a: Path | None
    yaml_b: Path | None
    common_specs_json: Path | None
    sigma_threshold: float | None
    accuracy: int
    yaml_key_a: str | None
    yaml_key_b: str | None
    survey_name_photo: str | None
    survey_name_spectro: str | None


@dataclass
class ParamRow:
    name: str
    sigma_a: float | None
    sigma_b: float | None
    deviation_pct: float | None
    is_cosmo: bool


@dataclass
class CaseResult:
    case: CaseDef
    model_label: str
    outdir: Path | None = None
    run_timestamp: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    omp_threads: str | None = None
    a_time_s: float | None = None
    b_time_s: float | None = None
    provenance_a: dict[str, Any] | None = None
    provenance_b: dict[str, Any] | None = None
    fisher_specs_a: Path | None = None
    fisher_specs_b: Path | None = None
    config_mismatches: list[str] = field(default_factory=list)
    params: list[ParamRow] = field(default_factory=list)
    max_deviation_pct: float | None = None
    max_deviation_param: str | None = None
    status: str = "not_run"  # not_run | ok | no_pair | error


def _case_sort_key(number: str) -> tuple[int, ...]:
    """Natural-sort key for dotted hierarchical case IDs (e.g. "03.2.1" -> (3, 2, 1))."""
    return tuple(int(part) for part in number.split("."))


def _normalize_case_id(raw: str) -> str | None:
    """Zero-pad the leading root segment of a dotted case ID (e.g. "3.2" -> "03.2")."""
    parts = raw.strip().split(".")
    if not parts or not all(part.isdigit() for part in parts):
        return None
    parts[0] = f"{int(parts[0]):02d}"
    return ".".join(parts)


def discover_cases(config_dir: Path, repo_root: Path) -> list[CaseDef]:
    cases: list[CaseDef] = []
    seen_numbers: dict[str, Path] = {}
    for path in sorted(config_dir.glob("compare_run_config.env_*")):
        m = re.match(r"compare_run_config\.env_(\d+(?:\.\d+)*)_", path.name)
        if not m:
            continue
        number = m.group(1)
        if number in seen_numbers:
            raise SystemExit(
                f"Duplicate case number {number}: {seen_numbers[number].name} and {path.name}"
            )
        seen_numbers[number] = path
        parsed = _parse_env_config(path, repo_root)
        threshold_raw = parsed.get("SIGMA_THRESHOLD", "").strip()
        threshold = None
        if threshold_raw:
            try:
                threshold = float(threshold_raw)
            except ValueError:
                threshold = None
        yaml_a = Path(parsed["YAML_A"]) if parsed.get("YAML_A") else None
        yaml_b = Path(parsed["YAML_B"]) if parsed.get("YAML_B") else None
        common_specs = (
            Path(parsed["COMMON_SPECS_JSON"]) if parsed.get("COMMON_SPECS_JSON") else None
        )
        cases.append(
            CaseDef(
                number=number,
                config_path=path,
                description=parsed.get("_description", ""),
                mode=parsed.get("MODE", ""),
                code_a=parsed.get("CODE_A", ""),
                code_b=parsed.get("CODE_B", ""),
                yaml_a=yaml_a,
                yaml_b=yaml_b,
                common_specs_json=common_specs,
                sigma_threshold=threshold,
                accuracy=int(parsed.get("ACCURACY", "1")),
                yaml_key_a=parsed.get("YAML_KEY_A") or None,
                yaml_key_b=parsed.get("YAML_KEY_B") or None,
                survey_name_photo=parsed.get("SURVEY_NAME_PHOTO") or None,
                survey_name_spectro=parsed.get("SURVEY_NAME_SPECTRO") or None,
            )
        )
    return cases


def _model_label(case: CaseDef) -> str:
    specs = _read_json(case.common_specs_json) if case.common_specs_json else None
    if not isinstance(specs, dict):
        return case.description or f"case {case.number}"
    options = specs.get("options") or {}
    freepars = specs.get("freepars") or {}
    cosmo_model = str(options.get("cosmo_model", "?"))
    if "w0" in freepars and "wa" in freepars:
        cosmo_model = "w0waCDM"
    elif "w0" in freepars:
        cosmo_model = "w0CDM"
    extras = [p for p in ("mnu", "Neff") if p in freepars]
    label = cosmo_model
    if extras:
        label += " + " + ", ".join(extras)
    return label


def _survey_specs_path(case: CaseDef, repo_root: Path) -> Path | None:
    survey_name = case.survey_name_photo or case.survey_name_spectro
    if not survey_name:
        return None
    return (
        repo_root
        / "cosmicfishpie"
        / "configs"
        / "default_survey_specifications"
        / f"{survey_name}.yaml"
    )


def _survey_label(case: CaseDef) -> str:
    survey_name = case.survey_name_photo or case.survey_name_spectro
    if not survey_name:
        return "default"
    if survey_name.endswith("-Optimistic"):
        return "Optimistic"
    if survey_name.endswith("-Pessimistic"):
        return "Pessimistic"
    return survey_name


def _variant_label(case: CaseDef) -> str:
    specs = _read_json(case.common_specs_json) if case.common_specs_json else None
    if not isinstance(specs, dict):
        return "default"
    options = specs.get("options") or {}
    tracer_key = "GCph_Tracer" if case.mode == "photo" else "GCsp_Tracer"
    tracer = options.get(tracer_key)
    if tracer == "clustering":
        return "P_cb"
    if tracer == "matter":
        return "P_mm"
    return str(tracer or "default")


LEGACY_2405_YAML_MIGRATIONS = {
    ("camb", "paper_mnuvalidation.yaml"): ("camb", "nuvalidation_hp.yaml"),
    ("class", "paper_mnuvalidation_photo.yaml"): ("class", "nuvalidation_hp.yaml"),
    ("class", "paper_mnuvalidation_spectro.yaml"): ("class", "nuvalidation_uhp.yaml"),
}


def _yaml_path_matches(saved: object, current: str | None) -> bool:
    """Match current solver selectors and verified-equivalent 2405 legacy paths."""
    if current is None:
        return True
    if saved == current:
        return True
    if not isinstance(saved, str):
        return False

    saved_path = Path(saved)
    current_path = Path(current)
    saved_key = (saved_path.parent.name, saved_path.name)
    current_key = (current_path.parent.name, current_path.name)
    return LEGACY_2405_YAML_MIGRATIONS.get(saved_key) == current_key


def _find_case_outdir(case: CaseDef, results_dir: Path, repo_root: Path) -> Path | None:
    if not results_dir.is_dir():
        return None
    want_common = str(case.common_specs_json.resolve()) if case.common_specs_json else None
    want_yaml_a = str(case.yaml_a.resolve()) if case.yaml_a else None
    want_yaml_b = str(case.yaml_b.resolve()) if case.yaml_b else None
    best: tuple[float, Path] | None = None
    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue
        meta = _read_json(folder / "run_metadata.json")
        if not isinstance(meta, dict):
            continue
        args = meta.get("args")
        if not isinstance(args, dict):
            continue
        if args.get("mode") != case.mode:
            continue
        if args.get("code_a") != case.code_a or args.get("code_b") != case.code_b:
            continue
        if args.get("accuracy") != case.accuracy:
            continue
        if case.yaml_key_a is not None and args.get("yaml_key_a") != case.yaml_key_a:
            continue
        if case.yaml_key_b is not None and args.get("yaml_key_b") != case.yaml_key_b:
            continue
        if want_common and args.get("common_specs") != want_common:
            continue
        if not _yaml_path_matches(args.get("yaml_a"), want_yaml_a):
            continue
        if not _yaml_path_matches(args.get("yaml_b"), want_yaml_b):
            continue
        resolved = meta.get("resolved")
        if not isinstance(resolved, dict):
            continue
        if case.survey_name_photo and resolved.get("survey_name_photo") != case.survey_name_photo:
            continue
        if (
            case.survey_name_spectro
            and resolved.get("survey_name_spectro") != case.survey_name_spectro
        ):
            continue
        mtime = (folder / "run_metadata.json").stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, folder)
    return best[1] if best else None


def _find_compare_json(folder: Path) -> Path | None:
    files = [p for p in folder.glob("compare_fishers_*.json") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _select_latest_ab_pair(compare: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the pairwise entry for the newest run timestamp that has both A and B.

    Falls back to the newest well-formed pair if no clean A/B timestamp match exists.
    """
    pairwise = compare.get("pairwise")
    if not isinstance(pairwise, list) or not pairwise:
        return None

    # timestamp -> {"A": path_str, "B": path_str}
    seen: dict[str, dict[str, str]] = {}
    for entry in pairwise:
        if not isinstance(entry, dict):
            continue
        for key in ("a", "b"):
            val = entry.get(key)
            if not isinstance(val, str):
                continue
            m = TIMESTAMP_RE.search(val)
            if not m:
                continue
            ts, label = m.group(1), m.group(2)
            seen.setdefault(ts, {})[label] = val

    complete_timestamps = sorted(
        (ts for ts, labels in seen.items() if "A" in labels and "B" in labels), reverse=True
    )
    for ts in complete_timestamps:
        want_a = seen[ts]["A"]
        want_b = seen[ts]["B"]
        for entry in pairwise:
            if not isinstance(entry, dict):
                continue
            if entry.get("a") == want_a and entry.get("b") == want_b:
                mm = entry.get("matrix_metrics")
                if isinstance(mm, dict):
                    return entry

    # Fallback: newest well-formed pair (previous behavior), better than nothing.
    def _is_good(entry: dict[str, Any]) -> bool:
        mm = entry.get("matrix_metrics")
        return isinstance(mm, dict) and ("rel_max" in mm or "fro_rel" in mm)

    good = [e for e in pairwise if isinstance(e, dict) and _is_good(e)]
    return good[-1] if good else None


def _param_rows(entry: dict[str, Any]) -> list[ParamRow]:
    analysis = entry.get("analysis")
    if not isinstance(analysis, dict):
        return []
    ratios = analysis.get("param_sigma_ratio")
    if not isinstance(ratios, dict):
        return []
    rows: list[ParamRow] = []
    for name, vals in ratios.items():
        if not isinstance(vals, dict):
            continue
        sigma_a = vals.get("sigma_a")
        sigma_b = vals.get("sigma_b")
        ratio = vals.get("ratio_b_over_a")
        dev = None
        if isinstance(ratio, (int, float)):
            dev = abs(float(ratio) - 1.0) * 100.0
        rows.append(
            ParamRow(
                name=str(name),
                sigma_a=float(sigma_a) if isinstance(sigma_a, (int, float)) else None,
                sigma_b=float(sigma_b) if isinstance(sigma_b, (int, float)) else None,
                deviation_pct=dev,
                is_cosmo=str(name) in COSMO_PARAM_SET,
            )
        )

    def sort_key(row: ParamRow) -> tuple[int, int, str]:
        if row.is_cosmo and row.name in COSMO_PARAM_ORDER:
            return (0, COSMO_PARAM_ORDER.index(row.name), row.name)
        if row.is_cosmo:
            return (0, len(COSMO_PARAM_ORDER), row.name)
        return (1, 0, row.name)

    rows.sort(key=sort_key)
    return rows


def _common_specs_mismatches(common_specs: Path, fisher_specs: Path) -> list[str]:
    """Return current common-spec values that differ from the saved run snapshot."""
    current = _read_json(common_specs)
    snapshot = _read_json(fisher_specs)
    if not isinstance(current, dict) or not isinstance(snapshot, dict):
        return []

    mismatches = []
    for section in ("options", "fiducialpars", "freepars"):
        current_section = current.get(section)
        snapshot_section = snapshot.get(section)
        if not isinstance(current_section, dict) or not isinstance(snapshot_section, dict):
            continue
        for key, current_value in current_section.items():
            snapshot_value = snapshot_section.get(key)
            if snapshot_value != current_value:
                mismatches.append(
                    f"{section}.{key}: run={json.dumps(snapshot_value)}, "
                    f"current={json.dumps(current_value)}"
                )
    return mismatches


def build_case_result(case: CaseDef, results_dir: Path, repo_root: Path) -> CaseResult:
    result = CaseResult(case=case, model_label=_model_label(case))
    outdir = _find_case_outdir(case, results_dir, repo_root)
    if outdir is None:
        return result
    result.outdir = outdir

    meta = _read_json(outdir / "run_metadata.json")
    if isinstance(meta, dict):
        result.run_timestamp = meta.get("timestamp")
        result.git_commit = meta.get("git_commit")
        result.git_dirty = meta.get("git_dirty")
        env = meta.get("env")
        if isinstance(env, dict):
            result.omp_threads = env.get("OMP_NUM_THREADS")
        prov = meta.get("backend_provenance")
        if isinstance(prov, dict):
            result.provenance_a = prov.get("code_a")
            result.provenance_b = prov.get("code_b")

    compare_json = _find_compare_json(outdir)
    if compare_json is None:
        result.status = "no_pair"
        return result
    compare = _read_json(compare_json)
    if not isinstance(compare, dict):
        result.status = "error"
        return result

    entry = _select_latest_ab_pair(compare)
    if entry is None:
        result.status = "no_pair"
        return result

    a_time = entry.get("a_timing_seconds")
    b_time = entry.get("b_timing_seconds")
    result.a_time_s = float(a_time) if isinstance(a_time, (int, float)) else None
    result.b_time_s = float(b_time) if isinstance(b_time, (int, float)) else None

    for key, attr in (("a", "fisher_specs_a"), ("b", "fisher_specs_b")):
        raw_path = entry.get(key)
        if not isinstance(raw_path, str):
            continue
        specs_path = Path(raw_path)
        if not specs_path.is_absolute():
            specs_path = outdir / specs_path
        if specs_path.is_file():
            setattr(result, attr, specs_path)

    if case.common_specs_json is not None and result.fisher_specs_a is not None:
        result.config_mismatches = _common_specs_mismatches(
            case.common_specs_json, result.fisher_specs_a
        )

    result.params = _param_rows(entry)
    for row in result.params:
        if row.deviation_pct is None:
            continue
        if result.max_deviation_pct is None or row.deviation_pct > result.max_deviation_pct:
            result.max_deviation_pct = row.deviation_pct
            result.max_deviation_param = row.name

    result.status = "ok" if result.params else "no_pair"
    return result


def _gate_status(result: CaseResult) -> str | None:
    if result.config_mismatches:
        return "STALE"
    if result.case.sigma_threshold is None or result.max_deviation_pct is None:
        return None
    return "FAIL" if result.max_deviation_pct > result.case.sigma_threshold else "PASS"


def _git_output(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _relevant_paths(case: CaseDef, repo_root: Path) -> list[str]:
    paths = [
        ":(glob)cosmicfishpie/**/*.py",
        "cosmicfishpie/configs/default_survey_specifications",
        "cosmicfishpie/configs/other_survey_specifications",
        "cosmicfishpie/configs/external_data",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
        "scripts/run_fisher_compare_backends.py",
        "scripts/compare_fishers_in_dir.py",
    ]
    for input_path in (
        case.yaml_a,
        case.yaml_b,
        case.common_specs_json,
        _survey_specs_path(case, repo_root),
    ):
        if input_path is None:
            continue
        try:
            paths.append(str(input_path.relative_to(repo_root)))
        except ValueError:
            paths.append(str(input_path))
    return paths


def _backend_version_changed(result: CaseResult) -> str | None:
    for label, provenance in (("A", result.provenance_a), ("B", result.provenance_b)):
        if not isinstance(provenance, dict):
            continue
        code = provenance.get("code")
        distribution = "classy" if code == "class" else code
        recorded = provenance.get("dist_version") or provenance.get("version")
        if not isinstance(distribution, str) or not recorded:
            continue
        try:
            current = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return f"backend {label} ({distribution}) is not installed"
        if current != str(recorded):
            return f"backend {label} changed from {recorded} to {current}"
    return None


def completed_case_state(result: CaseResult, repo_root: Path) -> tuple[bool, str, str]:
    """Return whether an existing result can be reused, its gate state, and the reason."""
    if result.status != "ok":
        return False, "INCOMPLETE", f"result status is {result.status}"
    if result.config_mismatches:
        return False, "STALE", "; ".join(result.config_mismatches)
    if not result.git_commit:
        return False, "STALE", "saved run has no Git revision"

    commit_check = _git_output(repo_root, "cat-file", "-e", f"{result.git_commit}^{{commit}}")
    if commit_check.returncode != 0:
        return False, "STALE", f"saved Git revision {result.git_commit} is unavailable"

    relevant_paths = _relevant_paths(result.case, repo_root)
    committed_diff = _git_output(
        repo_root,
        "diff",
        "--quiet",
        f"{result.git_commit}..HEAD",
        "--",
        *relevant_paths,
    )
    if committed_diff.returncode == 1:
        return False, "STALE", "relevant code or inputs changed since the saved run"
    if committed_diff.returncode > 1:
        return False, "STALE", "could not compare the saved Git revision with HEAD"

    dirty = _git_output(repo_root, "status", "--porcelain", "--", *relevant_paths)
    if dirty.returncode != 0:
        return False, "STALE", "could not inspect current input changes"
    if dirty.stdout.strip():
        return False, "STALE", "relevant code or inputs have uncommitted changes"

    version_change = _backend_version_changed(result)
    if version_change:
        return False, "STALE", version_change

    gate = _gate_status(result) or "INFORMATIONAL"
    dirty_note = "; saved run was marked dirty" if result.git_dirty else ""
    return (
        True,
        gate,
        f"completed at {result.git_commit}; relevant inputs unchanged{dirty_note}",
    )


def _gate_badge(status: str | None) -> str:
    if status == "PASS":
        return "<span class='badge badge-pass'>PASS</span>"
    if status == "FAIL":
        return "<span class='badge badge-fail'>FAIL</span>"
    if status == "STALE":
        return "<span class='badge badge-warn'>STALE CONFIG</span>"
    if status == "PENDING":
        return "<span class='badge badge-na'>pending</span>"
    return "<span class='badge badge-na'>informational</span>"


def _status_badge(result: CaseResult) -> str:
    if result.config_mismatches:
        return "<span class='badge badge-warn'>rerun required</span>"
    if result.status == "ok":
        commit = result.git_commit or "unknown"
        if result.git_dirty:
            commit = f"{commit} (dirty)"
        commit_esc = esc(commit)
        return f"<code>{commit_esc}</code>"
    labels = {
        "not_run": "<span class='badge badge-na'>not run yet</span>",
        "no_pair": "<span class='badge badge-warn'>ran, no A/B pair found</span>",
        "error": "<span class='badge badge-warn'>could not read results</span>",
    }
    return labels.get(result.status, "")


CSS = """
:root { color-scheme: light; }
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  max-width:980px;margin:32px auto;padding:0 16px;line-height:1.5;color:#111827}
h1,h2,h3{line-height:1.25}
a{color:#1d4ed8;text-decoration:none}
a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;margin:16px 0}
th,td{border:1px solid #e5e7eb;padding:10px 12px;text-align:left;vertical-align:top}
th{background:#f9fafb}
tr.clickable{cursor:pointer}
tr.clickable:hover{background:#f3f4f6}
.track-row td{background:#e5e7eb;color:#374151;font-size:13px;font-weight:700;
  letter-spacing:.02em;text-transform:uppercase}
.badge{display:inline-block;border-radius:999px;padding:2px 12px;font-size:12px;font-weight:600}
.badge-pass{background:#dcfce7;color:#166534;border:1px solid #86efac}
.badge-fail{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5}
.badge-na{background:#f3f4f6;color:#6b7280;border:1px solid #e5e7eb}
.badge-warn{background:#fef9c3;color:#854d0e;border:1px solid #fde68a}
.kv{display:grid;grid-template-columns:200px 1fr;gap:6px 16px;margin:16px 0}
.kv div{padding:2px 0}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}
pre{background:#0b1021;color:#d1d5db;padding:16px;border-radius:10px;overflow-x:auto;
  white-space:pre-wrap;word-break:break-word}
.section{margin:32px 0}
.subtle{color:#6b7280;font-size:14px}
.dev-cell{font-variant-numeric:tabular-nums}
.dev-high{color:#991b1b;font-weight:700}
.back{display:inline-block;margin-bottom:12px}
.links a{display:inline-block;margin-right:16px}
"""


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}%"


def _fmt_float(x: float | None, digits: int = 6) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}g}"


def _fmt_seconds(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1f}s"


def render_index(results: list[CaseResult], out_dir: Path) -> None:
    rows = []
    current_track = None
    for r in sorted(results, key=lambda r: _case_sort_key(r.case.number)):
        root = r.case.number.split(".", 1)[0]
        if root == "01":
            track = ("2303", "arXiv:2303.09451v1 - baseline-model MontePython validation")
        elif root in {"02", "03", "04"}:
            track = ("2405", "arXiv:2405.06047v1 - sensitivity to the neutrino sector")
        else:
            track = ("beyond", "Beyond paper validation")
        if track[0] != current_track:
            rows.append(f"<tr class='track-row'><td colspan='8'>{esc(track[1])}</td></tr>")
            current_track = track[0]
        gate = _gate_status(r)
        display_gate = gate
        if display_gate is None and r.case.sigma_threshold is not None:
            display_gate = "PENDING"
        dev = (
            f"{_fmt_pct(r.max_deviation_pct)} ({esc(r.max_deviation_param)})"
            if r.max_deviation_pct is not None
            else "n/a"
        )
        dev_cls = "dev-cell dev-high" if gate == "FAIL" else "dev-cell"
        status_html = _status_badge(r)
        row_html = (
            f"<tr class='clickable' onclick=\"location.href='case_{esc(r.case.number)}.html'\">"
            f"<td>{esc(r.case.number)}</td>"
            f"<td><a href='case_{esc(r.case.number)}.html'>{esc(r.model_label)}</a></td>"
            f"<td>{esc(r.case.mode.capitalize())}</td>"
            f"<td>{esc(_survey_label(r.case))}</td>"
            f"<td>{esc(_variant_label(r.case))}</td>"
            f"<td class='{dev_cls}'>{dev}</td>"
            f"<td>{_gate_badge(display_gate)}</td>"
            f"<td>{status_html}</td>"
            "</tr>"
        )
        rows.append(row_html)

    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>CosmicFishPie Backend Validation Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<h1>Backend Validation Dashboard</h1>
<p class='subtle'>CAMB vs CLASS Fisher-matrix validation for arXiv:2303.09451v1
(baseline-model MontePython validation) and arXiv:2405.06047v1 (sensitivity to the
neutrino sector). Click a row for the full per-parameter breakdown and the exact
YAML/spec files used.</p>
<table>
<thead>
<tr><th>Case</th><th>Model</th><th>Probe</th><th>Scenario</th><th>Variant</th><th>Max deviation</th><th>Gate</th><th>State</th></tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _param_table_html(rows: list[ParamRow], threshold: float | None, cosmo: bool) -> str:
    filtered = [r for r in rows if r.is_cosmo == cosmo]
    if not filtered:
        return "<p class='subtle'>(none)</p>"
    lines = [
        "<table><thead><tr><th>Parameter</th><th>&sigma; (A)</th><th>&sigma; (B)</th>"
        "<th>Deviation</th><th>Status</th></tr></thead><tbody>"
    ]
    for row in filtered:
        dev_cls = "dev-cell"
        status = ""
        if threshold is not None and row.deviation_pct is not None:
            if row.deviation_pct > threshold:
                dev_cls += " dev-high"
                status = "<span class='badge badge-fail'>FAIL</span>"
            else:
                status = "<span class='badge badge-pass'>PASS</span>"
        lines.append(
            "<tr>"
            f"<td><code>{esc(row.name)}</code></td>"
            f"<td>{esc(_fmt_float(row.sigma_a))}</td>"
            f"<td>{esc(_fmt_float(row.sigma_b))}</td>"
            f"<td class='{dev_cls}'>{esc(_fmt_pct(row.deviation_pct))}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _provenance_html(label: str, prov: dict[str, Any] | None) -> str:
    if not isinstance(prov, dict):
        return f"<div>{esc(label)}</div><div>n/a</div>"
    version = prov.get("dist_version") or prov.get("version") or "n/a"
    source = prov.get("install_source") or "n/a"
    return f"<div>{esc(label)}</div><div><code>{esc(prov.get('code'))} {esc(version)}</code> ({esc(source)})</div>"


def render_case(result: CaseResult, out_dir: Path, specs_dir: Path) -> None:
    case = result.case
    gate = _gate_status(result)
    threshold = case.sigma_threshold

    specs_links = []
    if case.yaml_a is not None:
        rel = _write_yaml_page(specs_dir, f"case_{case.number}_yaml_a", "YAML A", case.yaml_a)
        if rel:
            specs_links.append((f"CAMB/CLASS YAML A ({esc(case.code_a)})", rel))
    if case.yaml_b is not None:
        rel = _write_yaml_page(specs_dir, f"case_{case.number}_yaml_b", "YAML B", case.yaml_b)
        if rel:
            specs_links.append((f"CAMB/CLASS YAML B ({esc(case.code_b)})", rel))
    if case.common_specs_json is not None:
        rel = _write_spec_page(
            specs_dir,
            f"case_{case.number}_common_specs",
            "Common Specs JSON",
            case.common_specs_json,
        )
        if rel:
            label = "Current common specs source"
            if result.config_mismatches:
                label += " (differs from this run)"
            specs_links.append((label, rel))
    survey_specs = _survey_specs_path(case, REPO_ROOT)
    if survey_specs is not None:
        rel = _write_yaml_page(
            specs_dir,
            f"case_{case.number}_survey_specs",
            "Survey Specs YAML",
            survey_specs,
        )
        if rel:
            specs_links.append((f"Survey specs ({esc(_survey_label(case))})", rel))
    if result.fisher_specs_a is not None:
        rel = _write_spec_page(
            specs_dir,
            f"case_{case.number}_fisher_specs_a",
            "Fisher Specs A",
            result.fisher_specs_a,
        )
        if rel:
            specs_links.append((f"Generated Fisher specs A ({esc(case.code_a)})", rel))
    if result.fisher_specs_b is not None:
        rel = _write_spec_page(
            specs_dir,
            f"case_{case.number}_fisher_specs_b",
            "Fisher Specs B",
            result.fisher_specs_b,
        )
        if rel:
            specs_links.append((f"Generated Fisher specs B ({esc(case.code_b)})", rel))

    links_html = "".join(
        f"<a href='specs/{esc(rel)}' target='_blank'>{esc(label)} &rarr;</a>"
        for label, rel in specs_links
    )

    meta_rows = [
        ("Model", esc(result.model_label)),
        ("Probe", esc(case.mode.capitalize())),
        ("Scenario", esc(_survey_label(case))),
        ("Variant", esc(_variant_label(case))),
        ("Description", esc(case.description) or "n/a"),
        ("Sigma threshold", f"{threshold:g}%" if threshold is not None else "n/a (informational)"),
        (
            "Gate",
            _gate_badge(
                gate
                or (
                    "PENDING"
                    if threshold is not None and result.max_deviation_pct is None
                    else None
                )
            ),
        ),
        ("Timestamp", esc(result.run_timestamp) or "n/a"),
        ("OMP_NUM_THREADS", esc(result.omp_threads) or "n/a"),
        ("Elapsed A / B", f"{_fmt_seconds(result.a_time_s)} / {_fmt_seconds(result.b_time_s)}"),
        (
            "Git commit",
            (esc(result.git_commit) or "n/a") + (" (dirty)" if result.git_dirty else ""),
        ),
    ]
    meta_html = "".join(f"<div>{k}</div><div>{v}</div>" for k, v in meta_rows)

    prov_html = _provenance_html("Code A", result.provenance_a) + _provenance_html(
        "Code B", result.provenance_b
    )

    body_messages = []
    if result.config_mismatches:
        mismatch_items = "".join(
            f"<li><code>{esc(item)}</code></li>" for item in result.config_mismatches
        )
        body_messages.append(
            "<div class='section'><h2>Stale configuration</h2>"
            "<p>This result was generated with different common-spec values than the current "
            "case configuration. It is shown for provenance only and requires a rerun before "
            "its gate can be evaluated against the current configuration.</p>"
            f"<ul>{mismatch_items}</ul></div>"
        )
    if result.status != "ok":
        messages = {
            "not_run": "This case has not been run yet, or its output folder could not be located.",
            "no_pair": "This case ran, but a matching CAMB-vs-CLASS output pair could not be found "
            "in the results folder (partial or interrupted run).",
            "error": "This case's results could not be read (malformed compare JSON).",
        }
        body_messages.append(
            f"<div class='section'><p class='subtle'>{esc(messages.get(result.status, ''))}</p></div>"
        )
    body_status = "".join(body_messages)

    cosmo_table = _param_table_html(result.params, threshold, cosmo=True)
    nuisance_table = _param_table_html(result.params, threshold, cosmo=False)

    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Case {esc(case.number)}: {esc(result.model_label)}</title>
<style>{CSS}</style>
</head>
<body>
<a class='back' href='index.html'>&larr; Back to all cases</a>
<h1>Case {esc(case.number)}: {esc(result.model_label)} ({esc(case.mode.capitalize())})</h1>
{body_status}
<div class='section'>
<h2>Metadata</h2>
<div class='kv'>{meta_html}</div>
<div class='kv'>{prov_html}</div>
</div>
<div class='section'>
<h2>Cosmological parameters</h2>
{cosmo_table}
</div>
<div class='section'>
<h2>Nuisance parameters</h2>
{nuisance_table}
</div>
<div class='section links'>
<h2>Inputs used</h2>
{links_html or "<p class='subtle'>No spec files recorded for this case.</p>"}
</div>
</body>
</html>
"""
    (out_dir / f"case_{case.number}.html").write_text(html, encoding="utf-8")


def _write_spec_page(
    specs_dir: Path,
    slug: str,
    kind: str,
    source: Path,
    related_links: list[tuple[str, str]] | None = None,
) -> str | None:
    if not source.is_file():
        return None
    try:
        content = source.read_text(encoding="utf-8")
    except Exception:
        return None
    links_html = ""
    if related_links:
        links = "".join(
            f"<a href='{esc(href)}' target='_blank'>{esc(label)} &rarr;</a>"
            for label, href in related_links
        )
        links_html = "<div class='section links'><h2>Referenced inputs</h2>" f"{links}</div>"
    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{esc(kind)}: {esc(source.name)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{esc(kind)}</h1>
<p class='subtle'><code>{esc(source.name)}</code></p>
{links_html}
<pre>{esc(content)}</pre>
</body>
</html>
"""
    specs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slug}.html"
    (specs_dir / filename).write_text(html, encoding="utf-8")
    return filename


def _write_yaml_page(specs_dir: Path, slug: str, kind: str, source: Path) -> str | None:
    """Write a solver/survey YAML detail page. Each leaf YAML is self-contained,
    so this is a thin alias over _write_spec_page (kept as its own name since
    call sites read more clearly as "write this YAML" than "write this spec")."""
    return _write_spec_page(specs_dir, slug, kind, source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default=str(REPO_ROOT / "scripts" / "validation_configs"),
        help="Directory with compare_run_config.env_* case definitions",
    )
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "scripts" / "benchmark_results"),
        help="Directory with compare_* output folders",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "scripts" / "benchmark_results" / "dashboard"),
        help="Directory to write index.html/case_*.html into",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the generated dashboard over HTTP until interrupted",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP server bind address used with --serve (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP server port used with --serve (default: 8000)",
    )
    parser.add_argument(
        "--check-completed",
        metavar="CASE",
        help="Check whether a case has a reusable completed result, then exit",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    specs_dir = out_dir / "specs"

    cases = discover_cases(config_dir, REPO_ROOT)
    if not cases:
        raise SystemExit(f"No case configs found in {config_dir}")

    if args.check_completed is not None:
        requested = _normalize_case_id(args.check_completed)
        if requested is None:
            print(f"Invalid case number: {args.check_completed}")
            return 2
        case = next((item for item in cases if item.number == requested), None)
        if case is None:
            print(f"Unknown case: {requested}")
            return 2
        result = build_case_result(case, results_dir, REPO_ROOT)
        reusable, gate, reason = completed_case_state(result, REPO_ROOT)
        action = "SKIP" if reusable else "RUN"
        print(f"Case {requested}: {action} gate={gate} - {reason}")
        return 0 if reusable else 1

    results = [build_case_result(case, results_dir, REPO_ROOT) for case in cases]

    out_dir.mkdir(parents=True, exist_ok=True)
    render_index(results, out_dir)
    for result in results:
        render_case(result, out_dir, specs_dir)

    print(f"Wrote dashboard: {out_dir / 'index.html'}")
    print(f"Cases: {len(results)} ({sum(1 for r in results if r.status == 'ok')} with results)")

    if args.serve:
        handler = partial(SimpleHTTPRequestHandler, directory=str(out_dir))
        server = ThreadingHTTPServer((args.host, args.port), handler)
        display_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
        print(f"Serving dashboard at http://{display_host}:{server.server_port}/")
        print("Press Ctrl-C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping dashboard server.")
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
