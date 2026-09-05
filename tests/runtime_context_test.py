"""Regression tests for explicit cosmology runtime configuration ownership."""

from types import SimpleNamespace

import pytest

from cosmicfishpie.cosmology import cosmology
from cosmicfishpie.cosmology.cosmology import boltzmann_code, cosmo_functions


def _context(*, input_type="symbolic", omegam=0.31, nonlinear=False):
    return SimpleNamespace(
        settings={
            "feedback": 0,
            "SUPPRESS_WARNINGS": True,
            "ShareDeltaNeff": True,
            "nonlinear": nonlinear,
            "cosmo_model": "LCDM",
        },
        fiducialcosmopars={"Omegam": omegam, "h": 0.68},
        input_type=input_type,
        backend_parameters={"NUMERICS": {"k_samples": 3}},
        external=None,
    )


def test_cosmo_functions_forwards_explicit_context(monkeypatch):
    calls = []

    class FakeBackend:
        def __init__(self, cosmopars, code, *, configuration):
            calls.append((cosmopars, code, configuration))
            self.cosmopars = cosmopars
            self.symbcosmopars = dict(cosmopars)
            self.results = SimpleNamespace(kgrid=[0.1])

    context = _context()
    monkeypatch.setattr(cosmology, "boltzmann_code", FakeBackend)

    runtime = cosmo_functions({"Omegam": 0.31, "h": 0.68}, configuration=context)

    assert calls == [({"Omegam": 0.31, "h": 0.68}, "symbolic", context)]
    assert runtime.configuration is context
    assert runtime.settings is context.settings
    assert runtime.fiducialcosmopars is context.fiducialcosmopars


def test_explicit_context_rejects_backend_mismatch():
    context = _context(input_type="symbolic")

    with pytest.raises(ValueError, match="resolves backend 'symbolic'"):
        cosmology._backend_parameters(context, "camb")


def test_class_translation_uses_instance_owned_share_delta_neff():
    translator = object.__new__(boltzmann_code)
    translator.settings = {"ShareDeltaNeff": False, "cosmo_model": "LCDM"}

    translated = translator.changebasis_class(
        {
            "Omegam": 0.31,
            "Omegab": 0.05,
            "h": 0.68,
            "mnu": 0.06,
            "Neff": 3.2,
        }
    )

    assert translated["N_ur"] == pytest.approx(3.2 - boltzmann_code.hardcoded_Neff / 3)


def test_scoped_colossus_settings_restores_after_exception():
    settings = {"colossus_base_dir": "/tmp/context-a", "colossus_persistence": "rw"}
    colossus_settings = SimpleNamespace(BASE_DIR="/original", PERSISTENCE="r")

    with pytest.raises(RuntimeError):
        with cosmology._scoped_colossus_settings(settings, colossus_settings):
            assert colossus_settings.BASE_DIR == "/tmp/context-a"
            assert colossus_settings.PERSISTENCE == "rw"
            raise RuntimeError("intentional")

    assert colossus_settings.BASE_DIR == "/original"
    assert colossus_settings.PERSISTENCE == "r"
