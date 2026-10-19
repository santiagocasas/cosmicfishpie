import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "archive" / "cmb" / "run_cmb_benchmarks.py"
)
SPEC = importlib.util.spec_from_file_location("run_cmb_benchmarks", SCRIPT)
run_cmb_benchmarks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_cmb_benchmarks)


def test_benchmark_forwards_boltzmann_yaml(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--which",
            "planck",
            "--outdir",
            str(tmp_path),
            "--boltzmann-yaml",
            "/tmp/custom-solver.yaml",
        ],
    )
    monkeypatch.setattr(
        run_cmb_benchmarks.subprocess,
        "call",
        lambda command: captured.append(command) or 0,
    )

    assert run_cmb_benchmarks.main() == 0
    assert captured[0][-2:] == ["--boltzmann-yaml", "/tmp/custom-solver.yaml"]
