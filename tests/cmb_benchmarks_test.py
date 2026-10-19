import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "archive" / "cmb" / "run_cmb_benchmarks.py"
)
SPEC = importlib.util.spec_from_file_location("run_cmb_benchmarks", SCRIPT)
run_cmb_benchmarks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_cmb_benchmarks)


def _run_benchmark(tmp_path, monkeypatch, extra_args):
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
            *extra_args,
        ],
    )
    monkeypatch.setattr(
        run_cmb_benchmarks.subprocess,
        "call",
        lambda command: captured.append(command) or 0,
    )

    assert run_cmb_benchmarks.main() == 0
    return captured[0]


def test_benchmark_forwards_boltzmann_yaml(tmp_path, monkeypatch):
    command = _run_benchmark(
        tmp_path,
        monkeypatch,
        ["--boltzmann-yaml", "/tmp/custom-solver.yaml"],
    )

    option_index = command.index("--boltzmann-yaml")
    assert command[option_index + 1] == "/tmp/custom-solver.yaml"


def test_benchmark_omits_boltzmann_yaml_by_default(tmp_path, monkeypatch):
    command = _run_benchmark(tmp_path, monkeypatch, [])

    assert "--boltzmann-yaml" not in command
