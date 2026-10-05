"""Focused tests for Boltzmann-code parameter translations."""

import importlib
import sys
from pathlib import Path

import camb
import numpy as np
import pytest

from cosmicfishpie.cosmology.cosmology import (
    _class_pk_grid,
    _normalize_camb_import_path,
    boltzmann_code,
)
from cosmicfishpie.fishermatrix import cosmicfish as cff


def _translator(cosmo_model):
    translator = object.__new__(boltzmann_code)
    translator.settings = {"ShareDeltaNeff": True, "cosmo_model": cosmo_model}
    return translator


def _parameters():
    return {
        "Omegam": 0.314571,
        "Omegab": 0.049199,
        "h": 0.6737,
        "ns": 0.96605,
        "mnu": 0.06,
        "Neff": 3.044,
        "w0": -1.0,
        "wa": 0.0,
    }


def test_class_lcdm_drops_dark_energy_evolution_parameters():
    translated = _translator("LCDM").changebasis_class(_parameters())

    assert "w0" not in translated
    assert "wa" not in translated
    assert "w0_fld" not in translated
    assert "wa_fld" not in translated


def test_class_w0wa_keeps_dark_energy_evolution_parameters():
    translated = _translator("w0waCDM").changebasis_class(_parameters())

    assert translated["w0_fld"] == -1.0
    assert translated["wa_fld"] == 0.0


def test_camb_package_directory_uses_parent_as_import_root(monkeypatch):
    package_directory = Path(camb.__file__).parent
    monkeypatch.delitem(sys.modules, "camb")
    monkeypatch.setattr(sys, "path", [str(package_directory), *sys.path])

    import_root = _normalize_camb_import_path(str(package_directory))
    sys.path.insert(0, import_root)
    reloaded_camb = importlib.import_module("camb")

    assert import_root == str(package_directory.parent)
    assert Path(reloaded_camb.__file__).parent == package_directory


def test_class_pk_grid_uses_explicit_array_samples():
    class FakeClass:
        def get_pk_array(self, k, z, k_size, z_size, nonlinear):
            assert (k_size, z_size, nonlinear) == (2, 3, True)
            return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = _class_pk_grid(
        FakeClass(), np.array([0.1, 1.0]), np.array([0.0, 1.0, 2.0]), nonlinear=True
    )

    np.testing.assert_array_equal(result, np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]))


FULL_FIDUCIAL = {
    "Omegam": 0.32,
    "Omegab": 0.049,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.81,
    "mnu": 0.06,
    "Neff": 3.044,
}


def _spectro_backend_fisher_matrix(code, *, nonlinear):
    options = {
        "accuracy": 1,
        "outroot": f"test_backend_init_{code}",
        "results_dir": "results/",
        "derivatives": "3PT",
        "nonlinear": nonlinear,
        "feedback": 0,
        "specs_dir": "cosmicfishpie/configs/default_survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": code,
        "vary_bias_str": "b",
        "bfs8terms": False,
    }
    return cff.FisherMatrix(
        fiducialpars=dict(FULL_FIDUCIAL),
        freepars={"h": 0.01},
        options=options,
        observables=["GCsp"],
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )


@pytest.fixture(scope="module")
def camb_fisher_matrix():
    return _spectro_backend_fisher_matrix("camb", nonlinear=False)


@pytest.fixture(scope="module")
def class_fisher_matrix():
    return _spectro_backend_fisher_matrix("class", nonlinear=True)


def test_camb_plain_directory_used_directly_as_import_root(tmp_path):
    plain_directory = tmp_path / "camb_root"
    plain_directory.mkdir()

    assert _normalize_camb_import_path(str(plain_directory)) == str(plain_directory.resolve())


def test_camb_backend_init(camb_fisher_matrix):
    fm = camb_fisher_matrix

    assert fm.settings["code"] == "camb"
    assert fm.fiducialcosmo.cambcosmopars
    expected_import_root = str(Path(camb.__file__).parent.parent)
    assert expected_import_root in sys.path


def test_class_backend_init(class_fisher_matrix):
    fm = class_fisher_matrix

    assert fm.settings["code"] == "class"
    assert fm.fiducialcosmo.Classres is not None
    results = fm.fiducialcosmo.results
    assert np.isfinite(results.Pk_nl(0.5, 1e-2))
    assert np.isfinite(results.Pk_l(0.5, 1e-2))


def test_symbolic_backend_none_colossus_persistence():
    options = {
        "accuracy": 1,
        "outroot": "test_symbolic_none_persistence",
        "results_dir": "results/",
        "derivatives": "3PT",
        "nonlinear": False,
        "feedback": 0,
        "specs_dir": "cosmicfishpie/configs/default_survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": "symbolic",
        "colossus_persistence": None,
        "vary_bias_str": "b",
        "bfs8terms": False,
    }
    fm = cff.FisherMatrix(
        fiducialpars=dict(FULL_FIDUCIAL),
        freepars={"h": 0.01},
        options=options,
        observables=["GCsp"],
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )

    assert fm.settings["colossus_persistence"] is None
    assert fm.fiducialcosmo.results is not None
