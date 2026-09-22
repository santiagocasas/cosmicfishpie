"""Focused tests for Boltzmann-code parameter translations."""

import importlib
import sys
from pathlib import Path

import camb

from cosmicfishpie.cosmology.cosmology import (
    _normalize_camb_import_path,
    boltzmann_code,
)


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
