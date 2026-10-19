#!/usr/bin/env python
"""Six five-parameter LCDM Nautilus runs: MP Pessimistic GCsp and photo 3x2pt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JOBS = [(mode, code) for mode in ("spectro", "photo") for code in ("class", "camb", "symbolic")]
PRIORS = {
    "Omegam": [0.28, 0.36],
    "Omegab": [0.04, 0.06],
    "h": [0.60, 0.74],
    "ns": [0.90, 1.02],
    "sigma8": [0.75, 0.87],
}


def resources(code: str, args) -> tuple[int, int]:
    """Use the JURECA pool/OMP recipe adapted to the 36-CPU workstation."""
    omp = args.omp_threads if args.omp_threads is not None else (1 if code == "symbolic" else 2)
    workers = args.workers if args.workers is not None else 36 // omp
    return workers, omp


def run_config(mode: str, code: str, outdir: Path, args) -> dict:
    """Use MP validation inputs with LCDM, five common priors and fixed nuisances."""
    common = json.loads((ROOT / "scripts/validation/configs/common_specs_w0waCDM.json").read_text())
    profile = {
        "class": "mpvalidation_hp",
        "camb": "mpvalidation_p3",
        "symbolic": "syren_new_photo",
    }[code]
    options = dict(common["options"])
    options.update(
        code=code,
        cosmo_model="LCDM",
        survey_name="Euclid",
        survey_name_photo="Euclid-Photometric-Paper-Pessimistic" if mode == "photo" else False,
        survey_name_spectro=(
            "Euclid-Spectroscopic-Paper-Pessimistic" if mode == "spectro" else False
        ),
        specs_dir=str(ROOT / "cosmicfishpie/configs/default_survey_specifications") + "/",
        results_dir=str(outdir / "runtime") + "/",
        feedback=0,
    )
    options[f"{code}_config_yaml"] = str(
        ROOT / "cosmicfishpie/configs/default_boltzmann_yaml_files" / code / f"{profile}.yaml"
    )
    fiducial = dict(common["fiducialpars"], w0=-1.0, wa=0.0)
    workers, _ = resources(code, args)
    return {
        "name": f"mp_lcdm_{mode}_{code}",
        "fiducial": fiducial,
        "observables": ["GCph", "WL"] if mode == "photo" else ["GCsp"],
        "options": options,
        "priors": PRIORS,
        "use_nuisance": False,
        "output_dir": str(outdir),
        "sampler_settings": {
            "pool": workers,
            "n_live": args.n_live,
            "n_eff": args.n_eff,
            "n_networks": args.n_networks,
            "n_batch": args.n_batch or 2 * workers,
            "discard_exploration": True,
            "verbose": True,
        },
    }


def worker(mode: str, code: str, outdir: Path, args) -> int:
    from cosmicfishpie.likelihood.sampler import NautilusSampler

    config = run_config(mode, code, outdir, args)
    survey = config["options"][f"survey_name_{mode}"]
    inputs = [
        Path(config["options"][f"{code}_config_yaml"]),
        Path(config["options"]["specs_dir"]) / f"{survey}.yaml",
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record = {
        "config": config,
        "omp_threads": resources(code, args)[1],
        "commit": commit,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "data": "self-consistent noiseless fiducial data for this backend",
        "nuisances": "fixed to survey fiducials; conditional five-parameter posterior",
    }
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "requested_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"Existing run: {outdir}; use --resume for identical inputs")
        if json.loads(config_path.read_text()) != record:
            raise ValueError("Cannot resume: configuration, input files or Git revision changed")
    else:
        config_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    sampler = NautilusSampler(config)
    if list(sampler.prior_chosen.keys) != list(PRIORS):
        raise RuntimeError(f"Expected exactly five parameters, got {sampler.prior_chosen.keys}")
    loglike = float(
        sampler.likelihood.loglike(param_dict={p: config["fiducial"][p] for p in PRIORS})
    )
    if not (-1e-6 < loglike < 1e-6):
        raise RuntimeError(f"Fiducial self-consistency failed: loglike={loglike}")
    (outdir / "preflight.json").write_text(
        json.dumps({"sampled_parameters": list(PRIORS), "fiducial_loglike": loglike}, indent=2)
    )
    print(f"Fiducial loglike: {loglike}; sampling {list(PRIORS)}", flush=True)
    if not args.check_only:
        sampler.run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, help="Override automatic 18/36 likelihood workers")
    parser.add_argument("--omp-threads", type=int, help="Override automatic backend OMP=2/1")
    parser.add_argument("--n-live", type=int, default=1000)
    parser.add_argument("--n-eff", type=int, default=3000)
    parser.add_argument("--n-networks", type=int, default=8)
    parser.add_argument("--n-batch", type=int, help="Default: twice the likelihood worker count")
    parser.add_argument("--outdir", type=Path, help="Campaign directory (required with --resume)")
    parser.add_argument("--resume", action="store_true", help="Resume unchanged checkpoints")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print six configs without computing"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Initialize likelihoods, do not sample"
    )
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.n_eff, args.n_networks) < 1
        or args.n_live < 2
        or any(
            value is not None and value < 1
            for value in (args.workers, args.omp_threads, args.n_batch)
        )
    ):
        parser.error("Require positive workers/OMP/n-eff/networks/batch and n-live >= 2")
    allocation = int(
        os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or 36
    )
    for _, code in JOBS:
        workers, omp = resources(code, args)
        if workers * omp > min(36, allocation):
            parser.error(
                f"{code}: {workers} workers x {omp} OMP exceeds {min(36, allocation)} CPUs"
            )
    if args.resume and args.outdir is None:
        parser.error("--resume requires the original --outdir")
    if args.worker:
        if tuple(args.worker) not in JOBS or args.outdir is None:
            parser.error("Invalid worker selection or missing --outdir")
        return worker(*args.worker, args.outdir.resolve(), args)
    data_repo = Path(os.environ.get("CFP_DATA_REPO", str(ROOT.parent / "cjfp-data")))
    outdir = (
        args.outdir
        or data_repo / "chains" / f"euclid_mp_lcdm_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    ).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    f"{m}_{c}": {
                        "config": run_config(m, c, outdir / f"{m}_{c}", args),
                        "omp_threads": resources(c, args)[1],
                    }
                    for m, c in JOBS
                },
                indent=2,
            )
        )
        return 0
    if args.resume and not outdir.is_dir():
        parser.error("Resume directory does not exist")
    outdir.mkdir(parents=True, exist_ok=args.resume)
    manifest = {"execution": "sequential", "runs": {}}
    active = None
    print(f"Six sequential LCDM Nautilus runs\nOutputs: {outdir}", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for mode, code in JOBS:
            workers, omp = resources(code, args)
            env = dict(
                os.environ,
                OMP_NUM_THREADS=str(omp),
                OMP_DYNAMIC="FALSE",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
                FLEXIBLAS_NUM_THREADS="1",
            )
            # JURECA finding: libgomp can pin every spawned worker to the same core.
            for key in ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"):
                env.pop(key, None)
            name = f"{mode}_{code}"
            folder = outdir / name
            folder.mkdir(exist_ok=args.resume)
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                code,
                "--outdir",
                str(folder),
                "--workers",
                str(workers),
                "--omp-threads",
                str(omp),
                "--n-live",
                str(args.n_live),
                "--n-eff",
                str(args.n_eff),
                "--n-networks",
                str(args.n_networks),
            ]
            if args.n_batch:
                command.extend(["--n-batch", str(args.n_batch)])
            if args.resume:
                command.append("--resume")
            if args.check_only:
                command.append("--check-only")
            print(f"[RUN] {name}: {workers} workers x {omp} OMP; {folder / 'run.log'}", flush=True)
            with (folder / "run.log").open("a" if args.resume else "w") as log:
                active = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            status = active.wait()  # Finish this posterior before starting the next.
            active = None
            manifest["runs"][name] = {"exit_code": status, "workers": workers, "omp_threads": omp}
            (outdir / "campaign.json").write_text(json.dumps(manifest, indent=2))
            print(f"[{'OK' if status == 0 else 'FAIL'}] {name}: exit {status}", flush=True)
    except KeyboardInterrupt:
        print("Stopping likelihood workers; HDF5 checkpoints can be resumed.", flush=True)
        return 130
    finally:
        if active is not None:
            if active.poll() is None:
                os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
    return int(any(run["exit_code"] != 0 for run in manifest["runs"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
