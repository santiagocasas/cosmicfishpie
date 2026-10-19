"""Verify the launcher's six inputs without computing expensive Fishers."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validation/run_euclid_mp_fishers.py"
spec = importlib.util.spec_from_file_location("euclid_mp_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("mode,code", runner.JOBS)
def test_worker_preserves_mp_cosmology_and_uses_correct_probe_and_backend(
    tmp_path, monkeypatch, mode, code
):
    received = []

    class Matrix:
        def __init__(self, **kwargs):
            received.append(kwargs)

        def compute(self):
            return SimpleNamespace(file_name="mock_FM.txt")

    monkeypatch.setitem(
        sys.modules, "cosmicfishpie.fishermatrix.cosmicfish", SimpleNamespace(FisherMatrix=Matrix)
    )
    assert runner.worker(mode, code, tmp_path) == 0
    args = received[0]
    common = json.loads(runner.COMMON.read_text())
    assert args["fiducialpars"] == common["fiducialpars"]
    assert args["freepars"] == common["freepars"]
    assert args["parallel"] is False
    assert args["observables"] == (["GCph", "WL"] if mode == "photo" else ["GCsp"])
    for key, value in common["options"].items():
        assert args["options"][key] == value
    options = args["options"]
    profile = Path(options[f"{code}_config_yaml"])
    assert profile.is_file()
    if code == "symbolic":
        assert yaml.safe_load(profile.read_text())["ACCURACY"]["provider"] == "syren_new"
    survey = options[f"survey_name_{mode if mode == 'photo' else 'spectro'}"]
    assert survey.endswith("Paper-Pessimistic")
    assert (Path(options["specs_dir"]) / f"{survey}.yaml").is_file()
    assert json.loads((tmp_path / "result.json").read_text())["matrix"] == "mock_FM.txt"


def test_dry_run_creates_no_outputs_and_lists_all_six_runs(tmp_path):
    output = tmp_path / "campaign"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--outdir", str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert set(json.loads(result.stdout)) == {f"{mode}_{code}" for mode, code in runner.JOBS}
    assert not output.exists()


def test_rejects_oversubscribed_budget_before_creating_outputs(tmp_path):
    output = tmp_path / "campaign"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--jobs", "6", "--omp-threads", "7", "--outdir", str(output)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert not output.exists()
