#!/usr/bin/env python
# coding: utf-8

"""Render per-run comparison reports (Markdown + HTML) and an index.

This script is designed to be rerunnable and robust to partially-complete runs.
It reads, per run folder:
  - run_metadata.json (optional)
  - newest compare_fishers_*.json (optional)
  - newest compare_fishers_*_plots/ directory (optional)

It writes:
  - REPORT.md and REPORT.html inside each run folder
  - index.md and index.html in a chosen index directory
  - optional bundle directory with reports, plots, compare JSONs, and an index
  - optional single-file HTML report with inline plots
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    folder: Path
    status: str
    timestamp: str | None
    git_commit: str | None
    mode: str | None
    code_a: str | None
    code_b: str | None
    yaml_a: str | None
    yaml_b: str | None
    yaml_key_a: str | None
    yaml_key_b: str | None
    omp_threads: str | None
    compare_json: Path | None
    plots_dir: Path | None
    a_time_s: float | None
    b_time_s: float | None
    time_ratio_b_over_a: float | None
    time_delta_s: float | None
    rel_max: float | None
    fro_rel: float | None
    diag_ratio_min: float | None
    diag_ratio_median: float | None
    diag_ratio_max: float | None
    max_sigma_dev_param: str | None
    max_sigma_dev_ratio: float | None
    max_sigma_dev_percent: float | None
    yaml_top_changed: int | None
    yaml_top_only_a: int | None
    yaml_top_only_b: int | None
    yaml_top_changed_keys: list[str]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _newest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _discover_run_folders(patterns: list[str], explicit: list[str]) -> list[Path]:
    folders: set[Path] = set()
    for p in explicit:
        folders.add(Path(p).expanduser().resolve())
    for pat in patterns:
        # glob is relative to cwd
        for match in Path(".").glob(pat):
            if match.is_dir():
                folders.add(match.resolve())
    out = sorted(folders)
    return out


def _safe_relpath(target: Path, start: Path) -> str:
    try:
        return os.path.relpath(str(target), start=str(start))
    except Exception:
        return str(target)


def _slugify(text: str) -> str:
    if not text:
        return "run"
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif not out or out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "run"


def _data_uri(path: Path, mime: str) -> str | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _image_data_uri(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".svg":
        mime = "image/svg+xml"
    else:
        mime = "application/octet-stream"
    return _data_uri(path, mime)


def _json_data_uri(path: Path) -> str | None:
    return _data_uri(path, "application/json")


def _pick_main_pair(compare: dict[str, Any]) -> dict[str, Any] | None:
    pairwise = compare.get("pairwise")
    if not isinstance(pairwise, list) or not pairwise:
        return None

    ref_specs = None
    ref = compare.get("reference")
    if isinstance(ref, dict):
        ref_specs = ref.get("specs")

    def _is_good(entry: dict[str, Any]) -> bool:
        mm = entry.get("matrix_metrics")
        return isinstance(mm, dict) and ("rel_max" in mm or "fro_rel" in mm)

    # Prefer the entry comparing against the declared reference.
    if ref_specs:
        for e in pairwise:
            if not isinstance(e, dict):
                continue
            if e.get("a") == ref_specs and e.get("b") != ref_specs and _is_good(e):
                return e

    # Otherwise, pick the first well-formed pair.
    for e in pairwise:
        if isinstance(e, dict) and _is_good(e):
            return e
    return None


def _sigma_max_dev(entry: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    analysis = entry.get("analysis")
    if not isinstance(analysis, dict):
        return None, None, None
    ratios = analysis.get("param_sigma_ratio")
    if not isinstance(ratios, dict):
        return None, None, None

    best_name = None
    best_ratio = None
    best_dev = None
    for name, vals in ratios.items():
        if not isinstance(vals, dict):
            continue
        r = vals.get("ratio_b_over_a")
        if r is None:
            continue
        try:
            r_f = float(r)
        except Exception:
            continue
        dev = abs(r_f - 1.0)
        if best_dev is None or dev > best_dev:
            best_dev = dev
            best_ratio = r_f
            best_name = str(name)
    if best_dev is None or best_ratio is None:
        return None, None, None
    return best_name, best_ratio, 100.0 * best_dev


def _yaml_top_summary(
    entry: dict[str, Any]
) -> tuple[int | None, int | None, int | None, list[str]]:
    yaml_block = entry.get("yaml")
    if not isinstance(yaml_block, dict):
        return None, None, None, []
    diff = yaml_block.get("diff")
    if not isinstance(diff, dict):
        return None, None, None, []
    diff2 = diff.get("diff")
    if not isinstance(diff2, dict):
        return None, None, None, []
    changed = diff2.get("changed")
    only_a = diff2.get("only_in_a")
    only_b = diff2.get("only_in_b")
    if (
        not isinstance(changed, dict)
        or not isinstance(only_a, dict)
        or not isinstance(only_b, dict)
    ):
        return None, None, None, []
    keys = sorted(changed.keys())
    return len(changed), len(only_a), len(only_b), keys


def _find_plots_dir(folder: Path) -> Path | None:
    dirs = [p for p in folder.glob("compare_fishers_*_plots") if p.is_dir()]
    return _newest(dirs)


def _find_compare_json(folder: Path) -> Path | None:
    files = [p for p in folder.glob("compare_fishers_*.json") if p.is_file()]
    return _newest(files)


def _find_latest_reported_pair_files(folder: Path) -> tuple[list[Path], list[Path]]:
    specs = sorted(folder.glob("*_FM_specs.json")) + sorted(folder.glob("*_specifications.json"))
    mats = sorted(folder.glob("*_FM.txt")) + sorted(folder.glob("*_fishermatrix.txt"))
    return specs, mats


def summarize_folder(folder: Path) -> RunSummary:
    meta_path = folder / "run_metadata.json"
    meta = _read_json(meta_path) if meta_path.is_file() else None
    compare_json = _find_compare_json(folder)
    compare = _read_json(compare_json) if compare_json else None
    plots_dir = _find_plots_dir(folder)

    specs_files, mat_files = _find_latest_reported_pair_files(folder)

    timestamp = None
    git_commit = None
    mode = None
    code_a = None
    code_b = None
    yaml_a = None
    yaml_b = None
    yaml_key_a = None
    yaml_key_b = None
    omp_threads = None

    if isinstance(meta, dict):
        timestamp = meta.get("timestamp")
        git_commit = meta.get("git_commit")
        env = meta.get("env")
        if isinstance(env, dict):
            omp_threads = env.get("OMP_NUM_THREADS")
        args = meta.get("args")
        if isinstance(args, dict):
            mode = args.get("mode")
            code_a = args.get("code_a")
            code_b = args.get("code_b")
            yaml_a = args.get("yaml_a")
            yaml_b = args.get("yaml_b")
            yaml_key_a = args.get("yaml_key_a")
            yaml_key_b = args.get("yaml_key_b")
        resolved = meta.get("resolved")
        if isinstance(resolved, dict):
            # Prefer resolved yaml keys if args omitted them.
            yaml_key_a = yaml_key_a or resolved.get("yaml_key_a")
            yaml_key_b = yaml_key_b or resolved.get("yaml_key_b")

    a_time_s = None
    b_time_s = None
    time_ratio = None
    time_delta = None
    rel_max = None
    fro_rel = None
    dmin = None
    dmed = None
    dmax = None
    max_sigma_param = None
    max_sigma_ratio = None
    max_sigma_percent = None
    yaml_top_changed = None
    yaml_top_only_a = None
    yaml_top_only_b = None
    yaml_top_keys: list[str] = []

    status = "pending"
    if compare is None and compare_json is not None:
        status = "compare_json_unreadable"
    elif compare is None:
        # infer progress from file presence
        if specs_files or mat_files:
            status = "no_compare_yet"
        else:
            status = "no_outputs_yet"
    else:
        main = _pick_main_pair(compare)
        if main is None:
            status = "compare_missing_pairs"
        else:
            status = "ok"
            a_time_s = (
                main.get("a_timing_seconds")
                if isinstance(main.get("a_timing_seconds"), (int, float))
                else None
            )
            b_time_s = (
                main.get("b_timing_seconds")
                if isinstance(main.get("b_timing_seconds"), (int, float))
                else None
            )
            if isinstance(a_time_s, (int, float)) and isinstance(b_time_s, (int, float)):
                time_delta = float(b_time_s - a_time_s)
                time_ratio = float(b_time_s / a_time_s) if a_time_s != 0 else None

            mm = main.get("matrix_metrics")
            if isinstance(mm, dict):
                rel_max_v = mm.get("rel_max")
                rel_max = float(rel_max_v) if isinstance(rel_max_v, (int, float)) else None
                fro_rel_v = mm.get("fro_rel")
                fro_rel = float(fro_rel_v) if isinstance(fro_rel_v, (int, float)) else None
                dr = mm.get("diag_ratio")
                if isinstance(dr, dict):
                    dmin_v = dr.get("min")
                    dmin = float(dmin_v) if isinstance(dmin_v, (int, float)) else None
                    dmed_v = dr.get("median")
                    dmed = float(dmed_v) if isinstance(dmed_v, (int, float)) else None
                    dmax_v = dr.get("max")
                    dmax = float(dmax_v) if isinstance(dmax_v, (int, float)) else None

            max_sigma_param, max_sigma_ratio, max_sigma_percent = _sigma_max_dev(main)
            yaml_top_changed, yaml_top_only_a, yaml_top_only_b, yaml_top_keys = _yaml_top_summary(
                main
            )

    return RunSummary(
        folder=folder,
        status=status,
        timestamp=str(timestamp) if timestamp is not None else None,
        git_commit=str(git_commit) if git_commit is not None else None,
        mode=str(mode) if mode is not None else None,
        code_a=str(code_a) if code_a is not None else None,
        code_b=str(code_b) if code_b is not None else None,
        yaml_a=str(yaml_a) if yaml_a is not None else None,
        yaml_b=str(yaml_b) if yaml_b is not None else None,
        yaml_key_a=str(yaml_key_a) if yaml_key_a is not None else None,
        yaml_key_b=str(yaml_key_b) if yaml_key_b is not None else None,
        omp_threads=str(omp_threads) if omp_threads is not None else None,
        compare_json=compare_json,
        plots_dir=plots_dir,
        a_time_s=float(a_time_s) if isinstance(a_time_s, (int, float)) else None,
        b_time_s=float(b_time_s) if isinstance(b_time_s, (int, float)) else None,
        time_ratio_b_over_a=time_ratio,
        time_delta_s=time_delta,
        rel_max=rel_max,
        fro_rel=fro_rel,
        diag_ratio_min=dmin,
        diag_ratio_median=dmed,
        diag_ratio_max=dmax,
        max_sigma_dev_param=max_sigma_param,
        max_sigma_dev_ratio=max_sigma_ratio,
        max_sigma_dev_percent=max_sigma_percent,
        yaml_top_changed=yaml_top_changed,
        yaml_top_only_a=yaml_top_only_a,
        yaml_top_only_b=yaml_top_only_b,
        yaml_top_changed_keys=yaml_top_keys,
    )


def _fmt_float(x: float | None, fmt: str = ".6g") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), fmt)
    except Exception:
        return "n/a"


def _fmt_seconds(x: float | None) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.3f}s"
    except Exception:
        return "n/a"


def _report_css() -> str:
    return (
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.5}"
        "code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}"
        "h1,h2,h3,h4{line-height:1.2}"
        ".kv{display:grid;grid-template-columns:240px 1fr;gap:8px 16px;margin:12px 0}"
        ".kv div{padding:2px 0}"
        ".tag{display:inline-block;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:999px;padding:2px 10px;font-size:12px}"
        ".warn{background:#fff7ed;border:1px solid #fed7aa;padding:10px;border-radius:10px}"
        ".cards{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0}"
        ".card{border:1px solid #e5e7eb;border-radius:12px;padding:12px}"
        "img{max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:10px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #e5e7eb;padding:10px;vertical-align:top}"
        "th{background:#f9fafb;text-align:left}"
        "small{color:#6b7280}"
        "section{margin:32px 0;padding-top:12px;border-top:1px solid #e5e7eb}"
    )


def _report_body_html(
    summary: RunSummary,
    *,
    inline_images: bool,
    inline_json: bool,
    base_level: int = 1,
    anchor_id: str | None = None,
    include_back_to_top: bool = False,
) -> str:
    folder = summary.folder

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )

    def h(level: int, text: str, anchor: str | None = None) -> str:
        if anchor:
            return f"<h{level} id='{esc(anchor)}'>{text}</h{level}>"
        return f"<h{level}>{text}</h{level}>"

    parts: list[str] = []
    if anchor_id:
        parts.append("<section>")
        parts.append(f"<a id='{esc(anchor_id)}' name='{esc(anchor_id)}'></a>")

    parts.append(h(base_level, f"Fisher Compare Report: <code>{esc(folder.name)}</code>"))
    parts.append(f"<p><span class='tag'>status: {esc(summary.status)}</span></p>")

    parts.append(h(base_level + 1, "Run Definition"))
    parts.append("<div class='kv'>")
    parts.append(f"<div>Folder</div><div><code>{esc(str(folder))}</code></div>")
    if summary.timestamp:
        parts.append(f"<div>Timestamp</div><div><code>{esc(summary.timestamp)}</code></div>")
    if summary.git_commit:
        parts.append(f"<div>Git commit</div><div><code>{esc(summary.git_commit)}</code></div>")
    if summary.omp_threads:
        parts.append(
            f"<div>OMP_NUM_THREADS</div><div><code>{esc(summary.omp_threads)}</code></div>"
        )
    parts.append(f"<div>Mode</div><div><code>{esc(summary.mode or 'n/a')}</code></div>")
    parts.append(
        f"<div>A</div><div>code=<code>{esc(summary.code_a or 'n/a')}</code> "
        f"yaml_key=<code>{esc(summary.yaml_key_a or 'n/a')}</code><br>"
        f"yaml=<code>{esc(summary.yaml_a or 'n/a')}</code></div>"
    )
    parts.append(
        f"<div>B</div><div>code=<code>{esc(summary.code_b or 'n/a')}</code> "
        f"yaml_key=<code>{esc(summary.yaml_key_b or 'n/a')}</code><br>"
        f"yaml=<code>{esc(summary.yaml_b or 'n/a')}</code></div>"
    )
    parts.append("</div>")

    parts.append(h(base_level + 1, "Results"))
    if summary.compare_json:
        if inline_json:
            data_uri = _json_data_uri(summary.compare_json)
            if data_uri:
                name = summary.compare_json.name
                parts.append(
                    f"<p>Compare JSON: <a download='{esc(name)}' href='{data_uri}'>download</a></p>"
                )
            else:
                parts.append("<p>Compare JSON: <code>n/a</code></p>")
        else:
            rel = _safe_relpath(summary.compare_json, folder)
            parts.append(f"<p>Compare JSON: <code>{esc(rel)}</code></p>")
    else:
        parts.append("<p>Compare JSON: <code>n/a</code></p>")

    parts.append("<div class='cards'>")
    parts.append("<div class='card'>")
    parts.append(h(base_level + 2, "Timing"))
    parts.append("<div class='kv'>")
    parts.append(f"<div>A</div><div><code>{esc(_fmt_seconds(summary.a_time_s))}</code></div>")
    parts.append(f"<div>B</div><div><code>{esc(_fmt_seconds(summary.b_time_s))}</code></div>")
    parts.append(
        f"<div>Delta (B-A)</div><div><code>{esc(_fmt_seconds(summary.time_delta_s))}</code></div>"
    )
    parts.append(
        f"<div>Ratio (B/A)</div><div><code>{esc(_fmt_float(summary.time_ratio_b_over_a, '.4f'))}</code></div>"
    )
    parts.append("</div>")
    parts.append("</div>")

    parts.append("<div class='card'>")
    parts.append(h(base_level + 2, "Matrix Metrics"))
    parts.append("<div class='kv'>")
    parts.append(f"<div>rel_max</div><div><code>{esc(_fmt_float(summary.rel_max))}</code></div>")
    parts.append(f"<div>fro_rel</div><div><code>{esc(_fmt_float(summary.fro_rel))}</code></div>")
    parts.append(
        f"<div>diag_ratio min/med/max</div><div><code>{esc(_fmt_float(summary.diag_ratio_min))}</code> / "
        f"<code>{esc(_fmt_float(summary.diag_ratio_median))}</code> / <code>{esc(_fmt_float(summary.diag_ratio_max))}</code></div>"
    )
    if summary.max_sigma_dev_param and summary.max_sigma_dev_percent is not None:
        parts.append(
            f"<div>Max |sigma ratio-1|</div><div><code>{summary.max_sigma_dev_percent:.3f}%</code> "
            f"(param <code>{esc(summary.max_sigma_dev_param)}</code>, ratio <code>{esc(_fmt_float(summary.max_sigma_dev_ratio))}</code>)</div>"
        )
    else:
        parts.append("<div>Max |sigma ratio-1|</div><div><code>n/a</code></div>")
    parts.append("</div>")
    parts.append("</div>")
    parts.append("</div>")

    parts.append(h(base_level + 2, "YAML Diff Summary"))
    if summary.yaml_top_changed is not None:
        parts.append(
            f"<p>Top-level keys: changed=<code>{summary.yaml_top_changed}</code> "
            f"only_in_a=<code>{summary.yaml_top_only_a}</code> only_in_b=<code>{summary.yaml_top_only_b}</code></p>"
        )
        if summary.yaml_top_changed_keys:
            shown = summary.yaml_top_changed_keys[:12]
            rest = len(summary.yaml_top_changed_keys) - len(shown)
            txt = ", ".join(f"<code>{esc(k)}</code>" for k in shown)
            if rest > 0:
                txt += f", ... (+{rest})"
            parts.append(f"<p>Changed sections (top-level): {txt}</p>")
    else:
        parts.append("<div class='warn'>YAML diff summary not available yet.</div>")

    parts.append(h(base_level + 1, "Plots"))
    if summary.plots_dir and summary.plots_dir.is_dir():
        pngs = sorted(summary.plots_dir.glob("*.png"))
        if not pngs:
            parts.append("<p>No PNG plots found.</p>")
        else:
            for p in pngs:
                parts.append(h(base_level + 2, f"<code>{esc(p.name)}</code>"))
                if inline_images:
                    data_uri = _image_data_uri(p)
                    if data_uri:
                        parts.append(f"<p><img src='{data_uri}' alt='{esc(p.name)}'></p>")
                    else:
                        parts.append(f"<p><img src='{esc(str(p))}' alt='{esc(p.name)}'></p>")
                else:
                    rel = _safe_relpath(p, folder)
                    parts.append(f"<p><img src='{esc(rel)}' alt='{esc(p.name)}'></p>")
    else:
        parts.append("<p>No plots directory found yet.</p>")

    if include_back_to_top:
        parts.append("<p><a href='#top'>Back to index</a></p>")

    if anchor_id:
        parts.append("</section>")

    return "\n".join(parts)


def write_report_md(summary: RunSummary) -> None:
    folder = summary.folder
    lines: list[str] = []
    lines.append(f"# Fisher Compare Report: `{folder.name}`")
    lines.append("")
    lines.append(f"- Folder: `{folder}`")
    lines.append(f"- Status: `{summary.status}`")
    if summary.timestamp:
        lines.append(f"- Timestamp: `{summary.timestamp}`")
    if summary.git_commit:
        lines.append(f"- Git commit: `{summary.git_commit}`")
    if summary.omp_threads:
        lines.append(f"- OMP_NUM_THREADS: `{summary.omp_threads}`")
    lines.append("")
    lines.append("## Run Definition")
    lines.append("")
    lines.append(f"- Mode: `{summary.mode or 'n/a'}`")
    lines.append(
        f"- A: code=`{summary.code_a or 'n/a'}` yaml_key=`{summary.yaml_key_a or 'n/a'}` yaml=`{summary.yaml_a or 'n/a'}`"
    )
    lines.append(
        f"- B: code=`{summary.code_b or 'n/a'}` yaml_key=`{summary.yaml_key_b or 'n/a'}` yaml=`{summary.yaml_b or 'n/a'}`"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    if summary.compare_json:
        lines.append(f"- Compare JSON: `{_safe_relpath(summary.compare_json, folder)}`")
    else:
        lines.append("- Compare JSON: `n/a`")
    lines.append(f"- Timing A: `{_fmt_seconds(summary.a_time_s)}`")
    lines.append(f"- Timing B: `{_fmt_seconds(summary.b_time_s)}`")
    lines.append(f"- Timing delta (B-A): `{_fmt_seconds(summary.time_delta_s)}`")
    lines.append(f"- Timing ratio (B/A): `{_fmt_float(summary.time_ratio_b_over_a, '.4f')}`")
    lines.append("")
    lines.append("### Fisher Matrix Metrics")
    lines.append("")
    lines.append(f"- rel_max: `{_fmt_float(summary.rel_max, '.6g')}`")
    lines.append(f"- fro_rel: `{_fmt_float(summary.fro_rel, '.6g')}`")
    lines.append(
        "- diag_ratio (B/A): "
        f"min=`{_fmt_float(summary.diag_ratio_min, '.6g')}` "
        f"median=`{_fmt_float(summary.diag_ratio_median, '.6g')}` "
        f"max=`{_fmt_float(summary.diag_ratio_max, '.6g')}`"
    )
    lines.append("")
    lines.append("### Constraint Delta")
    lines.append("")
    if summary.max_sigma_dev_param and summary.max_sigma_dev_percent is not None:
        lines.append(
            f"- Max |sigma ratio - 1|: `{summary.max_sigma_dev_percent:.3f}%` "
            f"(param `{summary.max_sigma_dev_param}`, ratio B/A `{_fmt_float(summary.max_sigma_dev_ratio, '.6g')}`)"
        )
    else:
        lines.append("- Max |sigma ratio - 1|: `n/a`")
    lines.append("")
    lines.append("### YAML Diff Summary")
    lines.append("")
    if summary.yaml_top_changed is not None:
        lines.append(
            f"- Top-level keys: changed=`{summary.yaml_top_changed}` only_in_a=`{summary.yaml_top_only_a}` only_in_b=`{summary.yaml_top_only_b}`"
        )
        if summary.yaml_top_changed_keys:
            keys = ", ".join(f"`{k}`" for k in summary.yaml_top_changed_keys[:12])
            if len(summary.yaml_top_changed_keys) > 12:
                keys += ", ..."
            lines.append(f"- Changed sections (top-level, truncated): {keys}")
    else:
        lines.append("- YAML diff: `n/a`")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    if summary.plots_dir and summary.plots_dir.is_dir():
        pngs = sorted(summary.plots_dir.glob("*.png"))
        if not pngs:
            lines.append("(No PNG plots found.)")
        else:
            for p in pngs:
                rel = _safe_relpath(p, folder)
                lines.append(f"### `{p.name}`")
                lines.append("")
                lines.append(f"![]({rel})")
                lines.append("")
    else:
        lines.append("(No plots directory found yet.)")
        lines.append("")

    (folder / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_html(summary: RunSummary) -> None:
    folder = summary.folder

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>Fisher Compare Report: {esc(folder.name)}</title>")
    parts.append(f"<style>{_report_css()}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append(
        _report_body_html(
            summary,
            inline_images=False,
            inline_json=False,
            base_level=1,
        )
    )
    parts.append("</body></html>")
    (folder / "REPORT.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def _sort_key(summary: RunSummary) -> tuple:
    # Favor newer timestamp if parseable.
    ts = summary.timestamp
    dt = None
    if ts:
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = None
    return (dt or datetime.min, summary.folder.name)


def write_index(index_dir: Path, summaries: list[RunSummary]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    # Sort newest first
    ordered = sorted(summaries, key=_sort_key, reverse=True)

    # HTML
    rows_html: list[str] = []

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )

    for s in ordered:
        rel_report = _safe_relpath(s.folder / "REPORT.html", index_dir)
        rel_folder = _safe_relpath(s.folder, index_dir)
        label = s.folder.name
        a_code = s.code_a or "n/a"
        b_code = s.code_b or "n/a"
        codes_html = f"<code>A: {esc(a_code)}</code><br><code>B: {esc(b_code)}</code>"
        yaml_a_bn = Path(s.yaml_a).name if s.yaml_a else "n/a"
        yaml_b_bn = Path(s.yaml_b).name if s.yaml_b else "n/a"
        yamls_html = f"<code>A: {esc(yaml_a_bn)}</code><br><code>B: {esc(yaml_b_bn)}</code>"
        timing = (
            f"{_fmt_seconds(s.a_time_s)} / {_fmt_seconds(s.b_time_s)} (B/A {_fmt_float(s.time_ratio_b_over_a, '.3f')})"
            if s.a_time_s is not None or s.b_time_s is not None
            else "n/a"
        )
        dev = (
            f"{s.max_sigma_dev_percent:.2f}% ({s.max_sigma_dev_param})"
            if s.max_sigma_dev_percent is not None and s.max_sigma_dev_param
            else "n/a"
        )
        rows_html.append(
            "<tr>"
            f"<td><a href='{esc(rel_report)}'>{esc(label)}</a><br><small><code>{esc(rel_folder)}</code></small></td>"
            f"<td><code>{s.status}</code></td>"
            f"<td><code>{s.timestamp or 'n/a'}</code></td>"
            f"<td><code>{s.mode or 'n/a'}</code></td>"
            f"<td>{codes_html}</td>"
            f"<td>{yamls_html}</td>"
            f"<td><code>{timing}</code></td>"
            f"<td><code>{dev}</code></td>"
            "</tr>"
        )

    html = """<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Fisher Compare Reports</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.5}
    code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #e5e7eb;padding:10px;vertical-align:top}
    th{background:#f9fafb;text-align:left}
    small{color:#6b7280}
  </style>
</head>
<body>
  <h1>Fisher Compare Reports</h1>
  <p>Generated by <code>scripts/render_compare_reports.py</code></p>
  <table>
    <thead>
      <tr>
        <th>Run</th>
        <th>Status</th>
        <th>Timestamp</th>
        <th>Mode</th>
        <th>Codes</th>
        <th>YAMLs</th>
        <th>Timings (A/B)</th>
        <th>Max σ dev</th>
      </tr>
    </thead>
    <tbody>
"""
    html += "\n".join(rows_html)
    html += """
    </tbody>
  </table>
</body>
</html>
"""
    (index_dir / "index.html").write_text(html, encoding="utf-8")

    # Markdown
    md_lines: list[str] = []
    md_lines.append("# Fisher Compare Reports")
    md_lines.append("")
    md_lines.append(
        "| Run | Status | Timestamp | Mode | Codes | YAMLs | Timing (A/B) | Max σ dev |"
    )
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for s in ordered:
        rel_report = _safe_relpath(s.folder / "REPORT.md", index_dir)
        label = s.folder.name
        a_code = s.code_a or "n/a"
        b_code = s.code_b or "n/a"
        codes = f"A: {a_code}; B: {b_code}"
        yaml_a_bn = Path(s.yaml_a).name if s.yaml_a else "n/a"
        yaml_b_bn = Path(s.yaml_b).name if s.yaml_b else "n/a"
        yamls = f"A: {yaml_a_bn}; B: {yaml_b_bn}"
        timing = (
            f"{_fmt_seconds(s.a_time_s)} / {_fmt_seconds(s.b_time_s)} (B/A {_fmt_float(s.time_ratio_b_over_a, '.3f')})"
            if s.a_time_s is not None or s.b_time_s is not None
            else "n/a"
        )
        dev = (
            f"{s.max_sigma_dev_percent:.2f}% ({s.max_sigma_dev_param})"
            if s.max_sigma_dev_percent is not None and s.max_sigma_dev_param
            else "n/a"
        )
        md_lines.append(
            "| "
            f"[{label}]({rel_report}) | "
            f"`{s.status}` | "
            f"`{s.timestamp or 'n/a'}` | "
            f"`{s.mode or 'n/a'}` | "
            f"`{codes}` | "
            f"`{yamls}` | "
            f"`{timing}` | "
            f"`{dev}` |"
        )
    (index_dir / "index.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _bundle_run_folder(summary: RunSummary, bundle_dir: Path) -> RunSummary:
    dest = bundle_dir / summary.folder.name
    dest.mkdir(parents=True, exist_ok=True)

    report_html = summary.folder / "REPORT.html"
    report_md = summary.folder / "REPORT.md"
    if report_html.is_file():
        _copy_file(report_html, dest / report_html.name)
    if report_md.is_file():
        _copy_file(report_md, dest / report_md.name)

    for json_path in sorted(summary.folder.glob("compare_fishers_*.json")):
        if json_path.is_file():
            _copy_file(json_path, dest / json_path.name)

    if summary.plots_dir and summary.plots_dir.is_dir():
        _copy_tree(summary.plots_dir, dest / summary.plots_dir.name)

    compare_json = None
    if summary.compare_json:
        compare_json = dest / summary.compare_json.name
    plots_dir = None
    if summary.plots_dir:
        plots_dir = dest / summary.plots_dir.name

    return replace(summary, folder=dest, compare_json=compare_json, plots_dir=plots_dir)


def bundle_reports(bundle_dir: Path, summaries: list[RunSummary]) -> list[RunSummary]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundled: list[RunSummary] = []
    for summary in summaries:
        bundled.append(_bundle_run_folder(summary, bundle_dir))
    write_index(bundle_dir, bundled)
    return bundled


def _zip_bundle(bundle_dir: Path) -> Path:
    zip_path = bundle_dir.with_suffix(".zip")
    base = bundle_dir.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle_dir.rglob("*"):
            if not path.is_file():
                continue
            zf.write(path, path.relative_to(base))
    return zip_path


def write_single_file(path: Path, summaries: list[RunSummary]) -> None:
    ordered = sorted(summaries, key=_sort_key, reverse=True)

    seen: dict[str, int] = {}
    entries: list[tuple[RunSummary, str]] = []
    for summary in ordered:
        slug = _slugify(summary.folder.name)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchor = slug if count == 0 else f"{slug}-{count + 1}"
        entries.append((summary, anchor))

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )

    rows_html: list[str] = []
    for summary, anchor in entries:
        label = summary.folder.name
        a_code = summary.code_a or "n/a"
        b_code = summary.code_b or "n/a"
        codes_html = f"<code>A: {esc(a_code)}</code><br><code>B: {esc(b_code)}</code>"
        yaml_a_bn = Path(summary.yaml_a).name if summary.yaml_a else "n/a"
        yaml_b_bn = Path(summary.yaml_b).name if summary.yaml_b else "n/a"
        yamls_html = f"<code>A: {esc(yaml_a_bn)}</code><br><code>B: {esc(yaml_b_bn)}</code>"
        timing = (
            f"{_fmt_seconds(summary.a_time_s)} / {_fmt_seconds(summary.b_time_s)} (B/A {_fmt_float(summary.time_ratio_b_over_a, '.3f')})"
            if summary.a_time_s is not None or summary.b_time_s is not None
            else "n/a"
        )
        dev = (
            f"{summary.max_sigma_dev_percent:.2f}% ({summary.max_sigma_dev_param})"
            if summary.max_sigma_dev_percent is not None and summary.max_sigma_dev_param
            else "n/a"
        )
        rows_html.append(
            "<tr>"
            f"<td><a href='#{esc(anchor)}'>{esc(label)}</a><br><small><code>{esc(str(summary.folder))}</code></small></td>"
            f"<td><code>{summary.status}</code></td>"
            f"<td><code>{summary.timestamp or 'n/a'}</code></td>"
            f"<td><code>{summary.mode or 'n/a'}</code></td>"
            f"<td>{codes_html}</td>"
            f"<td>{yamls_html}</td>"
            f"<td><code>{timing}</code></td>"
            f"<td><code>{dev}</code></td>"
            "</tr>"
        )

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Fisher Compare Reports (Single File)</title>")
    parts.append(f"<style>{_report_css()}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<a id='top'></a>")
    parts.append("<h1>Fisher Compare Reports</h1>")
    parts.append("<p>Generated by <code>scripts/render_compare_reports.py</code></p>")
    parts.append("<table>")
    parts.append("<thead>")
    parts.append(
        "<tr><th>Run</th><th>Status</th><th>Timestamp</th><th>Mode</th><th>Codes</th><th>YAMLs</th><th>Timings (A/B)</th><th>Max sigma dev</th></tr>"
    )
    parts.append("</thead>")
    parts.append("<tbody>")
    parts.append("\n".join(rows_html))
    parts.append("</tbody>")
    parts.append("</table>")

    for summary, anchor in entries:
        parts.append(
            _report_body_html(
                summary,
                inline_images=True,
                inline_json=True,
                base_level=2,
                anchor_id=anchor,
                include_back_to_top=True,
            )
        )

    parts.append(
        "<script>"
        "document.addEventListener('click',function(e){"
        "var link=e.target.closest('a[href^=\"#\"]');"
        "if(!link){return;}"
        "var href=link.getAttribute('href');"
        "if(!href||href.length<2){return;}"
        "var id=decodeURIComponent(href.slice(1));"
        'var target=document.getElementById(id)||document.querySelector("a[name=\'"+id+"\']");'
        "if(!target){return;}"
        "e.preventDefault();"
        "target.scrollIntoView({behavior:'smooth',block:'start'});"
        "history.replaceState(null,'',href);"
        "});"
        "</script>"
    )

    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render REPORT.md/REPORT.html for compare_* folders and an index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="Run folders to include (each containing run_metadata.json and/or compare_fishers_*.json)",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern(s) to discover run folders (relative to cwd)",
    )
    parser.add_argument(
        "--index-dir",
        default="scripts/benchmark_results/compare_reports",
        help="Directory where index.html/index.md will be written",
    )
    parser.add_argument(
        "--formats",
        choices=["md", "html", "both"],
        default="both",
        help="Which report formats to generate per folder",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Directory to write a shareable bundle (REPORTs, plots, compare JSONs, index)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a zip file from the bundle directory",
    )
    parser.add_argument(
        "--single-file",
        default=None,
        help="Write a single self-contained HTML report (index + all runs)",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    bundle_dir = None
    if args.bundle_dir or args.zip:
        bundle_dir = (
            Path(args.bundle_dir)
            if args.bundle_dir
            else index_dir.parent / f"{index_dir.name}_bundle"
        )
        bundle_dir = bundle_dir.expanduser().resolve()
    single_file_path = Path(args.single_file).expanduser().resolve() if args.single_file else None

    folders = _discover_run_folders(args.glob, args.folders)
    if not folders:
        raise SystemExit("No run folders found. Provide folders or --glob patterns.")

    # Never treat the index output directory as an input run folder.
    folders = [f for f in folders if f.resolve() != index_dir]

    summaries: list[RunSummary] = []
    want_md = args.formats in ("md", "both")
    want_html = args.formats in ("html", "both") or bundle_dir is not None
    for f in folders:
        if not f.is_dir():
            continue
        s = summarize_folder(f)
        summaries.append(s)
        if want_md:
            write_report_md(s)
        if want_html:
            write_report_html(s)

    write_index(index_dir, summaries)
    print(f"Wrote index: {index_dir / 'index.html'}")

    report_kinds: list[str] = []
    if want_md:
        report_kinds.append("REPORT.md")
    if want_html:
        report_kinds.append("REPORT.html")

    if single_file_path:
        write_single_file(single_file_path, summaries)
        print(f"Wrote single-file report: {single_file_path}")
        if report_kinds:
            reports = ", ".join(report_kinds)
            print(f"Note: per-run {reports} and index were also generated.")

    if bundle_dir:
        bundle_reports(bundle_dir, summaries)
        print(f"Wrote bundle: {bundle_dir}")
        if args.zip:
            zip_path = _zip_bundle(bundle_dir)
            print(f"Wrote zip: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
