"""Amplitude translation and file contracts; never compute a power spectrum."""

from types import SimpleNamespace

import numpy as np
import pytest

from cosmicfishpie.cosmology.cosmology import boltzmann_code
from cosmicfishpie.cosmology.symbolic_new import SyrenNewProvider
from cosmicfishpie.fishermatrix import cosmicfish as cff


def backend_stub():
    backend = object.__new__(boltzmann_code)
    backend.settings = {"ShareDeltaNeff": False, "cosmo_model": "w0waCDM"}
    return backend


@pytest.mark.parametrize("amplitude", [1.7e-9, 2.1e-9, 2.5e-9])
@pytest.mark.parametrize("ee2", [False, True])
@pytest.mark.parametrize("amplitude_key", ["As", "10^9As"])
def test_public_as_reaches_every_backend_in_correct_units(amplitude, ee2, amplitude_key):
    inputs = {
        amplitude_key: amplitude if amplitude_key == "As" else amplitude * 1e9,
        "h": 0.6737,
        "Omegam": 0.314571,
        "Omegab": 0.049199,
        "ns": 0.96605,
        "mnu": 0.06,
        "Neff": 3.044,
        "w0": -1.0,
        "wa": 0.0,
    }
    original = dict(inputs)
    backend = backend_stub()
    if ee2:
        backend.boltzmann_classpars = {
            "PARAMETER_TRANSLATION": {
                "neutrino_scheme": "three_degenerate",
                "N_ur": 0.00641,
                "neutrino_mass_fac": 93.14,
            }
        }
    translated = backend.changebasis_class(inputs)
    assert translated["A_s"] == pytest.approx(amplitude, rel=1e-14, abs=0)
    assert "10^9As" not in translated
    assert "As" not in translated
    assert "sigma8" not in translated
    assert inputs == original

    # CAMB parameter construction is stubbed: no solver initialization or compute.
    seen = {}
    backend._set_camb_params = lambda camb, pars: seen.update(pars)
    camb_inputs = dict(inputs, num_nu_massive=1, num_nu_massless=2.0306666667)
    camb_original = dict(camb_inputs)
    translated = backend.changebasis_camb(
        camb_inputs, SimpleNamespace(CAMBUnknownArgumentError=RuntimeError)
    )
    assert translated["As"] == pytest.approx(amplitude, rel=1e-14, abs=0)
    assert seen["As"] == pytest.approx(amplitude, rel=1e-14, abs=0)
    assert "10^9As" not in translated
    assert "sigma8" not in translated
    assert camb_inputs == camb_original
    resolved = SyrenNewProvider._resolve_parameters(inputs, "w0waCDM")
    assert resolved["As"] == pytest.approx(amplitude * 1e9, rel=1e-14, abs=0)


@pytest.mark.parametrize("inputs", [{"logAs": np.log(21.0)}, {"10^9As": 2.1}])
def test_existing_class_amplitude_aliases_are_preserved(inputs):
    backend = backend_stub()
    translated = backend.changebasis_class(dict(h=0.67, **inputs))
    assert translated["A_s"] == pytest.approx(2.1e-9, rel=1e-14, abs=0)
    assert not set(inputs).intersection(translated)


@pytest.mark.parametrize(
    "parameter, fiducial, information", [("As", 2.1e-9, 1e20), ("10^9As", 2.1, 100.0)]
)
def test_fisher_export_preserves_amplitude_coordinate(
    tmp_path, monkeypatch, parameter, fiducial, information
):
    # Exercise the real exporter, with a supplied matrix and no Fisher computation.
    fisher = object.__new__(cff.FisherMatrix)
    fisher.settings = {
        "outroot": "as_precision",
        "results_dir": str(tmp_path),
        "fishermatrix_file_extension": ".txt",
        "feedback": 0,
    }
    fisher.observables = ["GCsp"]
    fisher.freeparams = {parameter: 0.01}
    fisher.allparams = {parameter: fiducial}
    fisher.fiducialcosmopars = dict(fisher.allparams)
    fisher.specs = {}
    fisher.cf_version = "test"
    fisher.feed_lvl = 0
    monkeypatch.setattr(cff.cfg, "latex_names", {parameter: parameter}, raising=False)
    monkeypatch.setattr(cff.ufs, "git_version", lambda: "synthetic-test")
    result = fisher.export_fisher(np.array([[information]]))
    assert result.param_names == [parameter]
    assert result.param_fiducial[0] == pytest.approx(fiducial, rel=1e-14, abs=0)
