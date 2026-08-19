#!/usr/bin/env python
# coding: utf-8

"""Render a clean HTML dashboard for the CAMB-vs-CLASS backend validation cases.

Reads the case definitions from ``scripts/validation_configs/compare_run_config.env_*``
and the matching outputs from ``scripts/benchmark_results/compare_*`` (as produced by
``compare_backends_report.sh`` / ``run_fisher_compare_backends.py``), and writes:

  <out-dir>/index.html            landing page: case #, model, probe, max deviation, gate
  <out-dir>/case_<NN>.html        per-case detail: full parameter sigma table + metadata
  <out-dir>/specs/case_<NN>_*.html  readable YAML/common/generated Fisher specifications

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
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
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
    params: list[ParamRow] = field(default_factory=list)
    max_deviation_pct: float | None = None
    max_deviation_param: str | None = None
    status: str = "not_run"  # not_run | ok | no_pair | error


def discover_cases(config_dir: Path, repo_root: Path) -> list[CaseDef]:
    cases: list[CaseDef] = []
    for path in sorted(config_dir.glob("compare_run_config.env_*")):
        m = re.match(r"compare_run_config\.env_(\d+)_", path.name)
        if not m:
            continue
        number = m.group(1)
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
            )
        )
    return cases


def _model_label(case: CaseDef) -> str:
    specs = _read_json(case.common_specs_json) if case.common_specs_json else None
    if not isinstance(specs, dict):
        return case.description or f"case {case.number}"
    options = specs.get("options") or {}
    freepars = specs.get("freepars") or {}
    cosmo_model = options.get("cosmo_model", "?")
    extras = [p for p in ("mnu", "Neff") if p in freepars]
    label = str(cosmo_model)
    if extras:
        label += " + " + ", ".join(extras)
    elif "mnu" in (specs.get("fiducialpars") or {}) and "mnu" not in freepars:
        label += " (fixed mnu)"
    return label


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
        if want_common and args.get("common_specs") != want_common:
            continue
        if want_yaml_a and args.get("yaml_a") != want_yaml_a:
            continue
        if want_yaml_b and args.get("yaml_b") != want_yaml_b:
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
    if result.case.sigma_threshold is None or result.max_deviation_pct is None:
        return None
    return "FAIL" if result.max_deviation_pct > result.case.sigma_threshold else "PASS"


def _gate_badge(status: str | None) -> str:
    if status == "PASS":
        return "<span class='badge badge-pass'>PASS</span>"
    if status == "FAIL":
        return "<span class='badge badge-fail'>FAIL</span>"
    return "<span class='badge badge-na'>informational</span>"


def _status_badge(result: CaseResult) -> str:
    if result.status == "ok":
        return ""
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
    for r in sorted(results, key=lambda r: int(r.case.number)):
        gate = _gate_status(r)
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
            f"<td class='{dev_cls}'>{dev}</td>"
            f"<td>{_gate_badge(gate)}</td>"
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
<p class='subtle'>CAMB vs CLASS Fisher matrix validation, arXiv:2405.06047v1 (paper neutrino
validation) and the Casas et al. w0waCDM/nuCDM validation tracks. Click a row for the full
per-parameter breakdown and the exact YAML/spec files used.</p>
<table>
<thead>
<tr><th>Case</th><th>Model</th><th>Probe</th><th>Max deviation</th><th>Gate</th><th></th></tr>
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
        rel = _write_spec_page(specs_dir, f"case_{case.number}_yaml_a", "YAML A", case.yaml_a)
        if rel:
            specs_links.append((f"CAMB/CLASS YAML A ({esc(case.code_a)})", rel))
    if case.yaml_b is not None:
        rel = _write_spec_page(specs_dir, f"case_{case.number}_yaml_b", "YAML B", case.yaml_b)
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
            specs_links.append(("Common specs JSON (fiducial + free parameters)", rel))
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
        ("Description", esc(case.description) or "n/a"),
        ("Sigma threshold", f"{threshold:g}%" if threshold is not None else "n/a (informational)"),
        ("Gate", _gate_badge(gate)),
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

    body_status = ""
    if result.status != "ok":
        messages = {
            "not_run": "This case has not been run yet, or its output folder could not be located.",
            "no_pair": "This case ran, but a matching CAMB-vs-CLASS output pair could not be found "
            "in the results folder (partial or interrupted run).",
            "error": "This case's results could not be read (malformed compare JSON).",
        }
        body_status = f"<div class='section'><p class='subtle'>{esc(messages.get(result.status, ''))}</p></div>"

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


def _write_spec_page(specs_dir: Path, slug: str, kind: str, source: Path) -> str | None:
    if not source.is_file():
        return None
    try:
        content = source.read_text(encoding="utf-8")
    except Exception:
        return None
    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{esc(kind)}: {esc(source.name)}</title>
<style>{CSS}</style>
</head>
<body>
<a class='back' href='javascript:history.back()'>&larr; Back</a>
<h1>{esc(kind)}</h1>
<p class='subtle'><code>{esc(source.name)}</code></p>
<pre>{esc(content)}</pre>
</body>
</html>
"""
    specs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slug}.html"
    (specs_dir / filename).write_text(html, encoding="utf-8")
    return filename


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
    args = parser.parse_args()

    config_dir = Path(args.config_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    specs_dir = out_dir / "specs"

    cases = discover_cases(config_dir, REPO_ROOT)
    if not cases:
        raise SystemExit(f"No case configs found in {config_dir}")

    results = [build_case_result(case, results_dir, REPO_ROOT) for case in cases]

    out_dir.mkdir(parents=True, exist_ok=True)
    render_index(results, out_dir)
    for result in results:
        render_case(result, out_dir, specs_dir)

    print(f"Wrote dashboard: {out_dir / 'index.html'}")
    print(f"Cases: {len(results)} ({sum(1 for r in results if r.status == 'ok')} with results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
