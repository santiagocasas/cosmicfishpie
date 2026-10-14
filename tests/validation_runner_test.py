"""Exercise campaign scheduling without starting scientific calculations."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("saved_failures", ["01.1.1\n01.1.2", ""])
def test_failed_first_forces_all_cases_and_waits_for_first_phase(tmp_path, saved_failures):
    root = Path(__file__).resolve().parents[1]
    script_dir = tmp_path / "scripts" / "validation"
    config_dir = script_dir / "configs"
    config_dir.mkdir(parents=True)
    shutil.copy(root / "scripts/validation/run_selected_validations.sh", script_dir)
    for case in ("01.1.0", "01.1.1", "01.1.2"):
        (config_dir / f"compare_run_config.env_{case}_test").write_text("# Test case\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/bash\nprintf 'test-commit\\n'\n")
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/bash\n"
        'for arg in "$@"; do\n'
        "  case $arg in\n"
        "    --list-failed) [[ -z $SAVED_FAILURES ]] || printf '%s\\n' \"$SAVED_FAILURES\"; exit 0 ;;\n"
        '    --check-completed) echo reuse >> "$EVENT_LOG"; exit 0 ;;\n'
        "  esac\n"
        "done\n"
        'echo render >> "$EVENT_LOG"\n'
    )
    for executable in (git, uv):
        executable.chmod(0o755)

    (script_dir / "compare_backends_report.sh").write_text(
        "#!/bin/bash\n"
        'case_id=$(basename "$2"); case_id=${case_id#compare_run_config.env_}; '
        "case_id=${case_id%_test}\n"
        'echo "start $case_id" >> "$EVENT_LOG"\n'
        "sleep 0.2\n"
        'echo "end $case_id" >> "$EVENT_LOG"\n'
        # One failing first-phase worker must not prevent the remaining phase.
        "[[ $case_id != 01.1.1 ]]\n"
    )
    env = dict(os.environ)
    env.pop("SLURM_CPUS_PER_TASK", None)
    env.pop("SLURM_CPUS_ON_NODE", None)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        CFP_VALIDATION_RESULTS_DIR=str(tmp_path / "results"),
        EVENT_LOG=str(tmp_path / "events"),
        SAVED_FAILURES=saved_failures,
    )
    run = subprocess.run(
        [
            "bash",
            str(script_dir / "run_selected_validations.sh"),
            "--all",
            "--failed-first",
            "--jobs",
            "2",
            "--omp-threads",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    events = (tmp_path / "events").read_text().splitlines()
    for case in ("01.1.0", "01.1.1", "01.1.2"):
        assert events.count(f"start {case}") == 1
        assert events.count(f"end {case}") == 1
    if saved_failures:
        assert events.index("start 01.1.0") > events.index("end 01.1.1")
        assert events.index("start 01.1.0") > events.index("end 01.1.2")
    assert events[-1] == "render"
    assert "Skipped unchanged completed cases: 0" in run.stdout
    assert "reuse" not in events
    batch = next((tmp_path / "results").glob("selected_validation_*"))
    assert (batch / "git_commit.txt").read_text().strip() == "test-commit"


def test_comparison_setup_uses_project_python_not_shell_python(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script_dir = tmp_path / "scripts/validation"
    script_dir.mkdir(parents=True)
    shutil.copy(root / "scripts/validation/compare_backends_report.sh", script_dir)
    survey_dir = tmp_path / "cosmicfishpie/configs/default_survey_specifications"
    survey_dir.mkdir(parents=True)
    (survey_dir / "Euclid-Photometric-ISTF-Pessimistic.yaml").write_text("name: test\n")
    yaml_file = tmp_path / "backend.yaml"
    yaml_file.write_text("ACCURACY: {}\n")
    config = tmp_path / "case.env"
    config.write_text(f'YAML_A="{yaml_file}"\nYAML_B="{yaml_file}"\n')

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Simulate a base environment where bare python cannot run the YAML helper.
    (bin_dir / "python").write_text("#!/bin/bash\necho 'Wrong Python' >&2\nexit 99\n")
    (bin_dir / "uv").write_text(
        "#!/bin/bash\n"
        '[[ $1 == run && $2 == --project && $3 == "$TEST_PROJECT" '
        "&& $4 == python ]] || exit 98\n"
        "shift 4\n"
        'if [[ $1 == - ]]; then exec "$TEST_PYTHON" "$@"; fi\n'
        # Stop at mocked scientific entry points; setup/hashing runs for real.
        'echo "$(basename "$1")" >> "$EVENT_LOG"\n'
    )
    for executable in bin_dir.iterdir():
        executable.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        TEST_PROJECT=str(tmp_path),
        TEST_PYTHON=sys.executable,
        EVENT_LOG=str(tmp_path / "events"),
        CFP_VALIDATION_RESULTS_DIR=str(tmp_path / "results"),
    )
    env.pop("OUTDIR", None)
    run = subprocess.run(
        ["bash", str(script_dir / "compare_backends_report.sh"), "--config", str(config)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / "events").read_text().splitlines() == [
        "run_fisher_compare_backends.py",
        "render_compare_reports.py",
    ]
