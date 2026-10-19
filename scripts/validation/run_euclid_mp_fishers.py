#!/usr/bin/env python
"""Six independent MP-validation Fishers: Pessimistic GCsp and photo 3x2pt."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "scripts/validation/configs/common_specs_w0waCDM.json"
JOBS = [(mode, code) for mode in ("spectro", "photo") for code in ("class", "camb", "symbolic")]


def run_specs(mode: str, code: str, outdir: Path) -> dict:
    """Use the existing MP case inputs, changing only runtime/output selection."""
    common = json.loads(COMMON.read_text())
    profiles = ROOT / "cosmicfishpie/configs/default_boltzmann_yaml_files"
    profile = {
        "class": "mpvalidation_hp",
        "camb": "mpvalidation_p3",
        "symbolic": "syren_new_photo",
    }[code]
    options = dict(common["options"])
    options.update(
        code=code,
        outroot=f"mp_pessimistic_{mode}_{code}",
        results_dir=str(outdir) + "/",
        specs_dir=str(ROOT / "cosmicfishpie/configs/default_survey_specifications") + "/",
        survey_name_photo="Euclid-Photometric-Paper-Pessimistic" if mode == "photo" else False,
        survey_name_spectro=(
            "Euclid-Spectroscopic-Paper-Pessimistic" if mode == "spectro" else False
        ),
    )
    options[f"{code}_config_yaml"] = str(profiles / code / f"{profile}.yaml")
    return {
        "options": options,
        "fiducialpars": common["fiducialpars"],
        "freepars": common["freepars"],
        # GCph+WL covariance includes GCph x WL; it is not a third observable token.
        "observables": ["GCph", "WL"] if mode == "photo" else ["GCsp"],
    }


def worker(mode: str, code: str, outdir: Path) -> int:
    from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix

    specs = run_specs(mode, code, outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "requested_specs.json").write_text(json.dumps(specs, indent=2))
    print(json.dumps(specs, indent=2), flush=True)
    matrix = FisherMatrix(
        **specs, cosmoModel=specs["options"]["cosmo_model"], surveyName="Euclid", parallel=False
    )
    start = time.monotonic()
    result = matrix.compute()
    if result is None:
        raise RuntimeError("Fisher computation returned no result")
    (outdir / "result.json").write_text(
        json.dumps({"matrix": str(result.file_name), "seconds": time.monotonic() - start}, indent=2)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6, help="Concurrent Fishers (default 6)")
    parser.add_argument(
        "--omp-threads", type=int, default=6, help="OMP threads per Fisher (default 6)"
    )
    parser.add_argument("--outdir", type=Path, help="New campaign directory; must not exist")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print all six inputs without running"
    )
    parser.add_argument("--worker", nargs=2, choices=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(*args.worker, args.outdir)
    if args.jobs < 1 or args.jobs > 6 or args.omp_threads < 1 or args.jobs * args.omp_threads > 36:
        parser.error("Use 1..6 jobs and positive OMP threads, with jobs * omp-threads <= 36")
    default_data = Path(os.environ.get("CFP_DATA_REPO", str(ROOT.parent / "cjfp-data")))
    outdir = (
        args.outdir
        or default_data
        / "validation/fisher"
        / f"euclid_mp_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    ).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {f"{m}_{c}": run_specs(m, c, outdir / f"{m}_{c}") for m, c in JOBS}, indent=2
            )
        )
        return 0
    allocated = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if allocated and args.jobs * args.omp_threads > int(allocated):
        parser.error(
            f"Requested {args.jobs * args.omp_threads} threads, but Slurm allocated {allocated}"
        )
    outdir.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"commit": commit, "jobs": args.jobs, "omp_threads": args.omp_threads, "runs": {}}
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(args.omp_threads),
        OMP_DYNAMIC="FALSE",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    pending = list(JOBS)
    active = []
    print(
        f"Six MP w0waCDM Fishers; {args.jobs} workers x {args.omp_threads} OMP threads", flush=True
    )
    print(f"Outputs: {outdir}\nCommit: {commit}", flush=True)
    try:
        while pending or active:
            while pending and len(active) < args.jobs:
                mode, code = pending.pop(0)
                name = f"{mode}_{code}"
                folder = outdir / name
                folder.mkdir()
                log = (folder / "run.log").open("w")
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    code,
                    "--outdir",
                    str(folder),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log.close()
                active.append((name, process))
                print(f"[RUN] {name}; PID {process.pid}; log {folder / 'run.log'}", flush=True)
            for name, process in active[:]:
                status = process.poll()
                if status is not None:
                    active.remove((name, process))
                    manifest["runs"][name] = {"exit_code": status}
                    (outdir / "campaign.json").write_text(json.dumps(manifest, indent=2))
                    print(f"[{'OK' if status == 0 else 'FAIL'}] {name}: exit {status}", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping active Fisher workers...", flush=True)
        return 130
    finally:
        for _, process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    print(f"Finished. Summary: {outdir / 'campaign.json'}", flush=True)
    return int(any(run["exit_code"] != 0 for run in manifest["runs"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
