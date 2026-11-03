#!/usr/bin/env python3
"""Render ara/trace/exploration_tree.yaml to one self-contained HTML file.

Needs PyYAML and Graphviz (`dot`). Usage, from the cfp-dev root:

    python3 ara/tools/render_tree.py            # writes ara/trace/exploration_tree.html
    python3 ara/tools/render_tree.py out.html

Nodes are coloured by type, boxed by provenance, grouped by day (one cluster
per date). Hover a node for its title, result/choice text. Solid arrows are
`also_depends_on` edges, dashed arrows are parent -> child nesting.
"""
from __future__ import annotations

import html
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "trace" / "exploration_tree.yaml"

FILL = {
    "experiment": "#cfe8ff",
    "decision": "#ffe9b3",
    "question": "#e3d4f5",
    "dead_end": "#f6b8b8",
    "pivot": "#c9efc9",
}
PEN = {"user": "#1b7f3b", "ai-executed": "#555555", "ai-suggested": "#c0392b", "user-revised": "#1b7f3b"}


def collect(items, parent=None, nodes=None, edges=None):
    nodes = {} if nodes is None else nodes
    edges = [] if edges is None else edges
    for node in items:
        if not isinstance(node, dict) or "id" not in node:
            continue
        nodes[node["id"]] = node
        if parent:
            edges.append((parent, node["id"], "child"))
        for dep in node.get("also_depends_on") or []:
            edges.append((dep, node["id"], "dep"))
        collect(node.get("children") or [], node["id"], nodes, edges)
    return nodes, edges


def short(text: str, width: int = 28, lines: int = 3) -> str:
    words, out, cur = str(text).split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width and cur:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    out.append(cur)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: width - 1] + "..."
    return "\\n".join(o.replace('"', "'") for o in out)


def tooltip(node) -> str:
    body = node.get("result") or node.get("choice") or node.get("description") or ""
    text = f"{node['id']} [{node.get('type')}] {node.get('provenance')} {node.get('timestamp')}\n{node.get('title')}\n\n{body}"
    return text.replace('"', "'")[:900]


def build_dot(nodes, edges) -> str:
    by_day = defaultdict(list)
    for node in nodes.values():
        by_day[str(node.get("timestamp"))[:10]].append(node)
    out = ["digraph G {", "rankdir=LR; compound=true; nodesep=0.15; ranksep=0.5;",
           'node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
           'edge [color="#888888", arrowsize=0.6];']
    for i, day in enumerate(sorted(by_day)):
        out.append(f'subgraph cluster_{i} {{ label="{day}"; fontname="Helvetica"; color="#bbbbbb"; style=rounded;')
        for node in by_day[day]:
            fill = FILL.get(node.get("type"), "#eeeeee")
            pen = PEN.get(node.get("provenance"), "#555555")
            label = f"{node['id']}\\n{short(node.get('title', ''))}"
            out.append(
                f'"{node["id"]}" [label="{label}", fillcolor="{fill}", color="{pen}", penwidth=2, '
                f'tooltip="{html.escape(tooltip(node), quote=True).replace(chr(10), "&#10;")}"];'
            )
        out.append("}")
    for src, dst, kind in edges:
        if src in nodes and dst in nodes:
            style = "dashed" if kind == "child" else "solid"
            out.append(f'"{src}" -> "{dst}" [style={style}];')
    out.append("}")
    return "\n".join(out)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "trace" / "exploration_tree.html"
    data = yaml.safe_load(SRC.read_text())
    nodes, edges = collect(data["tree"])
    svg = subprocess.run(["dot", "-Tsvg"], input=build_dot(nodes, edges), capture_output=True, text=True, check=True).stdout
    svg = svg[svg.index("<svg"):]
    legend = " ".join(
        f'<span style="background:{c};padding:2px 8px;border-radius:4px;margin-right:6px">{t}</span>'
        for t, c in FILL.items()
    )
    page = (
        "<!doctype html><meta charset='utf-8'><title>ARA exploration tree</title>"
        "<body style='font-family:Helvetica,Arial,sans-serif;margin:12px'>"
        f"<h3>ARA exploration tree: {len(nodes)} nodes, {len(edges)} edges</h3>"
        f"<p>{legend}<br>Border: green = user, grey = ai-executed, red = ai-suggested. "
        "Solid arrow = depends on, dashed = parent/child. Hover a node for details; scroll sideways, time runs left to right by day.</p>"
        f"<div style='overflow:auto;border:1px solid #ddd'>{svg}</div></body>"
    )
    target.write_text(page)
    print(f"wrote {target} ({len(nodes)} nodes, {len(edges)} edges)")


if __name__ == "__main__":
    main()
