"""Posterior dimension and checkpoint-safety contracts without running chains."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/likelihood/nautilus/run_euclid_mp_nautilus.py"
)
spec = importlib.util.spec_from_file_location("mp_nautilus_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def settings(**changes):
    values = dict(
        workers=None,
        omp_threads=None,
        n_live=1000,
        n_eff=3000,
        n_networks=8,
        n_batch=None,
        resume=False,
        check_only=False,
    )
    return SimpleNamespace(**(values | changes))


def test_six_lcdm_configs_are_exactly_five_dimensional(tmp_path):
    for mode, code in runner.JOBS:
        config = runner.run_config(mode, code, tmp_path, settings())
        assert list(config["priors"]) == ["Omegam", "Omegab", "h", "ns", "sigma8"]
        assert config["use_nuisance"] is False
        assert config["options"]["cosmo_model"] == "LCDM"
        assert config["fiducial"]["w0"] == -1 and config["fiducial"]["wa"] == 0
        assert config["fiducial"]["mnu"] == 0.06
        assert config["observables"] == (["GCph", "WL"] if mode == "photo" else ["GCsp"])
        assert Path(config["options"][f"{code}_config_yaml"]).is_file()
        assert config["sampler_settings"]["pool"] == (36 if code == "symbolic" else 18)
        assert config["sampler_settings"]["n_batch"] == (72 if code == "symbolic" else 36)
        assert all(
            low < config["fiducial"][p] < high for p, (low, high) in config["priors"].items()
        )


def test_worker_samples_only_five_parameters_and_guards_resume(tmp_path, monkeypatch):
    calls = []

    class FakeSampler:
        def __init__(self, config):
            calls.append(config)
            self.prior_chosen = SimpleNamespace(keys=list(config["priors"]))
            self.likelihood = SimpleNamespace(loglike=lambda param_dict: 0.0)

        def run(self):
            calls.append("sampled")

    monkeypatch.setitem(
        sys.modules,
        "cosmicfishpie.likelihood.sampler",
        SimpleNamespace(NautilusSampler=FakeSampler),
    )
    args = settings()
    assert runner.worker("photo", "symbolic", tmp_path, args) == 0
    assert calls[-1] == "sampled"
    assert len(json.loads((tmp_path / "preflight.json").read_text())["sampled_parameters"]) == 5
    with pytest.raises(FileExistsError):
        runner.worker("photo", "symbolic", tmp_path, args)
    args.resume = True
    args.n_live += 1
    with pytest.raises(ValueError, match="Cannot resume"):
        runner.worker("photo", "symbolic", tmp_path, args)
    assert len(calls) == 2  # Rejected before constructing any likelihood.


def test_worker_refuses_unintended_extra_parameter(tmp_path, monkeypatch):
    def fake_sampler(config):
        return SimpleNamespace(prior_chosen=SimpleNamespace(keys=list(config["priors"]) + ["AIA"]))

    monkeypatch.setitem(
        sys.modules,
        "cosmicfishpie.likelihood.sampler",
        SimpleNamespace(NautilusSampler=fake_sampler),
    )
    with pytest.raises(RuntimeError, match="Expected exactly five"):
        runner.worker("spectro", "symbolic", tmp_path, settings())


def test_campaign_waits_and_sets_backend_threads_without_cpu_pinning(tmp_path, monkeypatch):
    processes = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            assert not processes or processes[-1].done
            self.done = False
            self.code = command[command.index("--worker") + 2]
            self.env = kwargs["env"]
            self.workers = int(command[command.index("--workers") + 1])
            assert not any(k in self.env for k in ("OMP_PLACES", "OMP_PROC_BIND"))
            processes.append(self)

        def wait(self):
            self.done = True
            return 1 if len(processes) == 1 else 0

    monkeypatch.setenv("OMP_PLACES", "cores")
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--outdir", str(tmp_path / "campaign")])
    assert runner.main() == 1
    assert len(processes) == 6  # Later posteriors still run if one fails.
    for process in processes:
        assert process.workers == (36 if process.code == "symbolic" else 18)
        assert process.env["OMP_NUM_THREADS"] == ("1" if process.code == "symbolic" else "2")
        assert process.env["OPENBLAS_NUM_THREADS"] == "1"
