#!/usr/bin/env python
"""Measure fiducial-cosmology scaling for CosmicFishPie backends.

Each trial starts a fresh Python process. This is necessary because CAMB's
OpenMP runtime reads ``OMP_NUM_THREADS`` when the process starts. The measured
``fiducial_cosmology_seconds`` value is the same interval reported by
``config.init()`` as ``Cosmological functions obtained in``; no Fisher matrix,
likelihood, or sampler is constructed.

The default backend is CAMB with the ``fresh_lcdm_camb_safe_1x8`` fiducial
configuration and the package's relaxed-precision development YAML. CLASS uses
the corresponding package default YAML. Symbolic uses an LCDM-compatible
five-parameter fiducial; its OpenMP sweep is a control measurement because the
emulator is not expected to benefit from CAMB/CLASS-style OpenMP parallelism.

Example on a Slurm allocation with 12 CPUs available to one task::

    srun --cpu-bind=cores --cpus-per-task=12 \
      uv run python scripts/benchmark_fiducial_scaling.py --backend camb

Results, per-trial child logs, and CSV/JSON summaries are written below
``results/fiducial_scaling/`` and must not be committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_THREADS = (1, 2, 4, 8, 12)
BACKENDS = ("camb", "class", "symbolic")
BOLTZMANN_FIDUCIAL = {
    "Omegam": 0.3191,
    "Omegab": 0.049795,
    "w0": -1.0,
    "wa": 0.0,
    "h": 0.6674,
    "ns": 0.96605,
    "sigma8": 0.81,
    "mnu": 0.06,
    "Neff": 3.044,
}
CAMB_SAFE_OBSERVABLES = ["GCph", "WL"]
SYMBOLIC_FIDUCIAL = {
    "Omegam": 0.3191,
    "Omegab": 0.049795,
    "h": 0.6674,
    "ns": 0.96605,
    "sigma8": 0.81,
}


def default_backend_yaml(backend: str) -> Path:
    """Return the package's fast development YAML for ``backend``."""
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    return (
        Path(__file__).resolve().parents[1]
        / "cosmicfishpie"
        / "configs"
        / "default_boltzmann_yaml_files"
        / backend
        / "default.yaml"
    )


def backend_fiducial(backend: str) -> dict[str, float | int]:
    """Return parameters supported by the selected fiducial backend."""
    if backend == "symbolic":
        return dict(SYMBOLIC_FIDUCIAL)
    return dict(BOLTZMANN_FIDUCIAL)


def backend_version(backend: str) -> str:
    """Return installed backend package metadata when available."""
    from importlib.metadata import PackageNotFoundError, version

    package_name = {"camb": "camb", "class": "classy", "symbolic": "symbolic-pofk"}[backend]
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def yaml_sha256(path: Path) -> str:
    """Return the content hash used to identify a solver configuration."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_child(result_file: Path, backend: str, backend_yaml: Path) -> None:
    """Construct one backend fiducial once and write timing metadata.

    The parent process sets all thread environment variables before this child
    starts. ``config.time`` is locally replaced because it is used exclusively
    for the two timestamps surrounding ``cosmo_functions`` in ``config.init``.
    """
    from cosmicfishpie.configs import config as cfg

    timestamps: list[float] = []

    def config_timer() -> float:
        value = time.perf_counter()
        timestamps.append(value)
        return value

    cfg.time = config_timer
    start = time.perf_counter()
    cfg.init(
        options={
            "accuracy": 1,
            "code": backend,
            "feedback": 0,
            f"{backend}_config_yaml": str(backend_yaml),
            "results_dir": str(result_file.parent) + "/",
            "outroot": f"{backend}_fiducial_scaling_",
            "survey_name": "Euclid",
            "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
            "survey_name_spectro": False,
            "cosmo_model": "LCDM",
        },
        observables=CAMB_SAFE_OBSERVABLES,
        fiducialpars=backend_fiducial(backend),
        surveyName="Euclid",
        cosmoModel="LCDM",
    )
    configuration_seconds = time.perf_counter() - start
    if len(timestamps) != 2:
        raise RuntimeError(
            "Expected config.init() to take exactly two fiducial-cosmology timestamps; "
            f"received {len(timestamps)}."
        )

    payload = {
        "backend": backend,
        "backend_yaml": str(backend_yaml),
        "omp_num_threads": int(os.environ["OMP_NUM_THREADS"]),
        "fiducial_cosmology_seconds": timestamps[1] - timestamps[0],
        "configuration_seconds": configuration_seconds,
        "backend_version": backend_version(backend),
        "observables": CAMB_SAFE_OBSERVABLES,
        "fiducial_parameters": backend_fiducial(backend),
        "python_version": sys.version,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    result_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for both parent and isolated child modes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="camb",
        help="Cosmology backend to benchmark (default: camb).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=DEFAULT_THREADS,
        help=f"OpenMP thread counts to test (default: {' '.join(map(str, DEFAULT_THREADS))}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed fresh-process trials per thread count (default: 3).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Untimed fresh-process trials per thread count (default: 1).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/fiducial_scaling"),
        help="Directory for this run's timestamped results (default: results/fiducial_scaling).",
    )
    parser.add_argument(
        "--backend-yaml",
        "--camb-yaml",
        dest="backend_yaml",
        type=Path,
        default=None,
        help="Solver settings YAML (default: package default for --backend).",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.backend_yaml is None:
        args.backend_yaml = default_backend_yaml(args.backend)

    if args.child and args.result_file is None:
        parser.error("--child requires --result-file")
    if not args.child:
        if not args.backend_yaml.is_file():
            parser.error(f"Backend YAML not found: {args.backend_yaml}")
        if any(count < 1 for count in args.threads):
            parser.error("--threads values must be positive")
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        if args.warmups < 0:
            parser.error("--warmups cannot be negative")
    return args


def child_environment(thread_count: int) -> dict[str, str]:
    """Set backend OpenMP parallelism while pinning library thread pools."""
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(thread_count),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def launch_trial(
    *,
    thread_count: int,
    trial_kind: str,
    trial_index: int,
    run_dir: Path,
    backend: str,
    backend_yaml: Path,
) -> dict[str, object]:
    """Run one isolated fiducial construction and return its timings."""
    trial_dir = run_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    stem = f"omp{thread_count:02d}_{trial_kind}{trial_index:02d}"
    result_file = trial_dir / f"{stem}.json"
    log_file = trial_dir / f"{stem}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--backend",
        backend,
        "--result-file",
        str(result_file),
        "--backend-yaml",
        str(backend_yaml),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        env=child_environment(thread_count),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    process_seconds = time.perf_counter() - started
    log_file.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} trial failed for OMP_NUM_THREADS={thread_count}; see {log_file}"
        )
    if not result_file.is_file():
        raise RuntimeError(f"{backend} trial did not create {result_file}")

    result = json.loads(result_file.read_text(encoding="utf-8"))
    result.update(
        {
            "trial_kind": trial_kind,
            "trial_index": trial_index,
            "process_seconds": process_seconds,
            "result_file": str(result_file),
            "log_file": str(log_file),
        }
    )
    return result


def write_csv(path: Path, trials: list[dict[str, object]]) -> None:
    """Write raw trial timings in a spreadsheet-friendly form."""
    fieldnames = [
        "omp_num_threads",
        "trial_kind",
        "trial_index",
        "fiducial_cosmology_seconds",
        "configuration_seconds",
        "process_seconds",
        "backend",
        "backend_version",
        "result_file",
        "log_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: trial.get(key) for key in fieldnames} for trial in trials)


def summarise(timed_trials: list[dict[str, object]]) -> list[dict[str, float | int]]:
    """Return median, minimum, and maximum fiducial timings per thread count."""
    results = []
    for thread_count in sorted({int(trial["omp_num_threads"]) for trial in timed_trials}):
        selected = [
            float(trial["fiducial_cosmology_seconds"])
            for trial in timed_trials
            if int(trial["omp_num_threads"]) == thread_count
        ]
        results.append(
            {
                "omp_num_threads": thread_count,
                "median_fiducial_seconds": statistics.median(selected),
                "min_fiducial_seconds": min(selected),
                "max_fiducial_seconds": max(selected),
            }
        )
    baseline = results[0]["median_fiducial_seconds"]
    for result in results:
        result["speedup_vs_one_thread"] = baseline / result["median_fiducial_seconds"]
    return results


def main() -> int:
    """Run the requested scaling experiment or one isolated child trial."""
    args = parse_args()
    if args.child:
        run_child(args.result_file, args.backend, args.backend_yaml.resolve())
        return 0

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    backend_yaml = args.backend_yaml.resolve()
    metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": args.backend,
        "backend_yaml": str(backend_yaml),
        "backend_yaml_sha256": yaml_sha256(backend_yaml),
        "fiducial_parameters": backend_fiducial(args.backend),
        "observables": CAMB_SAFE_OBSERVABLES,
        "threads": args.threads,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "platform": platform.platform(),
        "parent_python": sys.version,
        "thread_environment": child_environment(args.threads[0]),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Backend: {args.backend}")
    print(f"Backend YAML: {backend_yaml}")
    print(f"Results: {run_dir}")
    print("Each trial is a fresh process; only the requested OpenMP thread count varies.")
    all_trials: list[dict[str, object]] = []
    for thread_count in args.threads:
        print(f"\nOMP_NUM_THREADS={thread_count}", flush=True)
        for trial_kind, repetitions in (("warmup", args.warmups), ("timed", args.repeats)):
            for trial_index in range(1, repetitions + 1):
                result = launch_trial(
                    thread_count=thread_count,
                    trial_kind=trial_kind,
                    trial_index=trial_index,
                    run_dir=run_dir,
                    backend=args.backend,
                    backend_yaml=backend_yaml,
                )
                all_trials.append(result)
                print(
                    f"  {trial_kind} {trial_index}/{repetitions}: "
                    f"fiducial={result['fiducial_cosmology_seconds']:.3f} s, "
                    f"config={result['configuration_seconds']:.3f} s, "
                    f"process={result['process_seconds']:.3f} s",
                    flush=True,
                )

    timed_trials = [trial for trial in all_trials if trial["trial_kind"] == "timed"]
    summary = summarise(timed_trials)
    write_csv(run_dir / "trials.csv", all_trials)
    (run_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary, "trials": all_trials}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"\nMedian {args.backend} fiducial-cosmology time")
    print(f"{'OMP threads':>11s} {'median [s]':>12s} {'min-max [s]':>23s} {'speedup':>10s}")
    for result in summary:
        print(
            f"{result['omp_num_threads']:11d} {result['median_fiducial_seconds']:12.3f} "
            f"{result['min_fiducial_seconds']:.3f}-{result['max_fiducial_seconds']:.3f} "
            f"{result['speedup_vs_one_thread']:10.2f}"
        )
    print(f"\nWrote {run_dir / 'summary.json'} and {run_dir / 'trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
