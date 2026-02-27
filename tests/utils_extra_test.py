import json

import numpy as np
import pytest

from cosmicfishpie.utilities import utils


def test_round_decimals_up_old_mode_branches():
    old_flag = utils.numerics.old_round_decimals_up
    try:
        utils.numerics.old_round_decimals_up = True
        assert utils.numerics.round_decimals_up(1.21, decimals=1) == pytest.approx(1.3)
        assert utils.numerics.round_decimals_up(0.05, decimals=1) == pytest.approx(0.05)
        assert utils.numerics.round_decimals_up(0.005, decimals=1) == pytest.approx(0.005)
        assert utils.numerics.round_decimals_up(1.2, decimals=0) == pytest.approx(2.0)
    finally:
        utils.numerics.old_round_decimals_up = old_flag


def test_bisection_boundary_cases():
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    assert utils.numerics.bisection(arr, -1.0) == 0
    assert utils.numerics.bisection(arr, 0.0) == 0
    assert utils.numerics.bisection(arr, 3.0) == len(arr) - 2
    assert utils.numerics.bisection(arr, 9.0) == len(arr) - 2


def test_git_version_oserror_returns_unknown(monkeypatch):
    class DummyPopen:
        def __init__(self, *args, **kwargs):
            raise OSError("boom")

    monkeypatch.setattr(utils.subprocess, "Popen", DummyPopen)
    assert utils.filesystem.git_version() == "Unknown"


def test_git_version_generic_exception_returns_not_repo(monkeypatch):
    class DummyPopen:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("generic")

    monkeypatch.setattr(utils.subprocess, "Popen", DummyPopen)
    assert utils.filesystem.git_version() == "Not a git repository"


def test_load_fisher_from_json_uses_defaults(monkeypatch, tmp_path):
    snap = {
        "options": {},
        "specifications": {"x": 1},
        "fiducialpars": {"Omegam": 0.3},
        "freepars": {"Omegam": 0.01},
        "metadata": {},
    }
    json_path = tmp_path / "snapshot.json"
    json_path.write_text(json.dumps(snap), encoding="utf-8")

    captured = {}

    class DummyFM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "cosmicfishpie.fishermatrix.cosmicfish.FisherMatrix",
        DummyFM,
        raising=True,
    )

    fm, loaded = utils.load_fisher_from_json(str(json_path))
    assert isinstance(fm, DummyFM)
    assert loaded["specifications"]["x"] == 1
    assert captured["observables"] == ["GCph", "WL"]
    assert captured["cosmoModel"] == "LCDM"
    assert captured["surveyName"] == "Euclid"
