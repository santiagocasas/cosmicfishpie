#!/usr/bin/env python
# coding: utf-8

"""Package benchmark outputs into a single tar.gz archive."""

from __future__ import annotations

import argparse
import io
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def _git_cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _git_meta() -> dict[str, str]:
    root = _git_cmd(["git", "rev-parse", "--show-toplevel"])
    if not root:
        return {}
    branch = _git_cmd(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git_cmd(["git", "-C", root, "rev-parse", "--short", "HEAD"])
    dirty = _git_cmd(["git", "-C", root, "status", "--porcelain"])
    meta = {}
    if branch:
        meta["git_branch"] = branch
    if commit:
        meta["git_commit"] = commit
    meta["git_dirty"] = "yes" if dirty else "no"
    return meta


def _collect_files(results_dir: Path) -> list[Path]:
    return sorted([p for p in results_dir.iterdir() if p.is_file()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Package benchmark outputs.")
    parser.add_argument(
        "--results-dir",
        default="scripts/benchmark_results",
        help="Directory containing benchmark outputs",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output tar.gz path (default: benchmarks_<branch>_<commit>_<YYYYMMDD>.tar.gz)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results dir not found: {results_dir}")

    meta = _git_meta()
    branch = meta.get("git_branch", "unknown")
    commit = meta.get("git_commit", "unknown")
    stamp = time.strftime("%Y%m%d")
    outpath = args.out or f"benchmarks_{branch}_{commit}_{stamp}.tar.gz"
    outpath = Path(outpath)

    files = _collect_files(results_dir)
    if not files:
        raise SystemExit("No files found to package.")

    manifest_lines = [
        f"results_dir={results_dir}",
        f"created_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
    ]
    manifest_lines += [f"{k}={v}" for k, v in meta.items()]
    manifest_lines += ["files:"] + [str(p) for p in files]
    manifest = "\n".join(manifest_lines) + "\n"

    with tarfile.open(outpath, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p))
        info = tarfile.TarInfo("MANIFEST.txt")
        info.size = len(manifest.encode("utf-8"))
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest.encode("utf-8")))

    print(f"Packaged {len(files)} files to {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
