"""Tests for the run-local AnalysisContext compatibility layer."""

from types import ModuleType
from typing import Any, cast

import pytest

from cosmicfishpie.configs import context


def _legacy_config() -> ModuleType:
    legacy = ModuleType("legacy_config")
    legacy.__dict__.update(
        settings={"feedback": 0, "code": "symbolic"},
        specs={"survey": "test", "nested": {"bins": [1, 2]}},
        obs=["GCph"],
        freeparams={"Omegam": 0.01},
        fiducialparams={"Omegam": 0.3},
        fiducialcosmo=object(),
        photoparams={"z0": 0.9},
        Photobiasparams={"bias_model": "binned", "b1": 1.0},
        IAparams={"AIA": 1.0},
        Spectrobiasparams={"bg_1": 1.0},
        Spectrononlinearparams={"sigmap_0": 1.0},
        IMbiasparams={"bHI": 1.0},
        PShotparams={"Ps_1": 0.0},
        latex_names={"Omegam": r"\\Omega_m"},
        input_type="symbolic",
        boltzmann_symbolicpars={"nonlinear": False},
        external=None,
    )
    return legacy


def test_analysis_context_snapshots_legacy_configuration():
    legacy = _legacy_config()

    analysis = context.AnalysisContext.from_legacy_config(legacy)

    legacy.settings["feedback"] = 3
    legacy.specs["survey"] = "different"
    legacy.freeparams["h"] = 0.01

    assert analysis.settings == {"feedback": 0, "code": "symbolic"}
    assert analysis.specs == {"survey": "test", "nested": {"bins": (1, 2)}}
    assert analysis.freeparams == {"Omegam": 0.01}
    assert analysis.backend_parameters == {"nonlinear": False}
    assert analysis.external is None
    assert analysis.fiducialcosmo is legacy.fiducialcosmo

    with pytest.raises(TypeError):
        cast(Any, analysis.settings)["feedback"] = 2
    with pytest.raises(TypeError):
        cast(Any, analysis.specs["nested"])["bins"] = ()


def test_analysis_context_exposes_legacy_aliases_and_combined_parameters():
    analysis = context.AnalysisContext.from_legacy_config(_legacy_config())

    assert analysis.obs == ("GCph",)
    assert analysis.fiducialcosmopars == {"Omegam": 0.3}
    assert analysis.photopars == {"z0": 0.9}
    assert analysis.photobiaspars["b1"] == 1.0
    assert analysis.IApars == {"AIA": 1.0}
    assert analysis.Spectrobiaspars == {"bg_1": 1.0}
    assert analysis.PShotpars == {"Ps_1": 0.0}
    assert analysis.Spectrononlinpars == {"sigmap_0": 1.0}
    assert analysis.allparams_fidus == analysis.allparams
    assert analysis.allparams == {
        "Omegam": 0.3,
        "z0": 0.9,
        "bias_model": "binned",
        "b1": 1.0,
        "AIA": 1.0,
        "bg_1": 1.0,
        "sigmap_0": 1.0,
        "bHI": 1.0,
        "Ps_1": 0.0,
    }


def test_build_analysis_context_copies_options_before_legacy_initialization(monkeypatch):
    legacy = _legacy_config()
    received = {}

    def fake_init(**kwargs):
        received.update(kwargs)
        kwargs["options"]["mutated_by_legacy_init"] = True

    legacy.__dict__["init"] = fake_init
    monkeypatch.setattr(context, "_load_legacy_config", lambda: legacy)
    options = {"feedback": 0}

    analysis = context.build_analysis_context(
        options=options,
        observables=["GCph"],
        survey_name="TestSurvey",
        cosmo_model="LCDM",
    )

    assert options == {"feedback": 0}
    assert received["options"] == {"feedback": 0, "mutated_by_legacy_init": True}
    assert received["surveyName"] == "TestSurvey"
    assert received["cosmoModel"] == "LCDM"
    assert analysis.observables == ("GCph",)
