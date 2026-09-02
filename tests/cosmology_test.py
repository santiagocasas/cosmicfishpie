"""Focused tests for Boltzmann-code parameter translations."""

from cosmicfishpie.configs import config as cfg
from cosmicfishpie.cosmology.cosmology import boltzmann_code


def _translator(monkeypatch, cosmo_model):
    monkeypatch.setattr(cfg, "settings", {"ShareDeltaNeff": True}, raising=False)
    translator = object.__new__(boltzmann_code)
    translator.settings = {"cosmo_model": cosmo_model}
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


def test_class_lcdm_drops_dark_energy_evolution_parameters(monkeypatch):
    translated = _translator(monkeypatch, "LCDM").changebasis_class(_parameters())

    assert "w0" not in translated
    assert "wa" not in translated
    assert "w0_fld" not in translated
    assert "wa_fld" not in translated


def test_class_w0wa_keeps_dark_energy_evolution_parameters(monkeypatch):
    translated = _translator(monkeypatch, "w0waCDM").changebasis_class(_parameters())

    assert translated["w0_fld"] == -1.0
    assert translated["wa_fld"] == 0.0
