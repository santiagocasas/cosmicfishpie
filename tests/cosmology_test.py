"""Focused tests for Boltzmann-code parameter translations."""

import importlib
from importlib.util import find_spec
import sys
from pathlib import Path
from types import SimpleNamespace

import camb
import numpy as np
import pytest
import yaml

from cosmicfishpie.cosmology.cosmology import (
    _class_pk_grid,
    _normalize_camb_import_path,
    boltzmann_code,
)
from cosmicfishpie.cosmology.ee2 import class_ee2_boost, class_ee2_parameters
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


def test_class_three_degenerate_translation_matches_syren_example():
    translator = _translator("LCDM")
    translator.boltzmann_classpars = {
        "PARAMETER_TRANSLATION": {
            "neutrino_scheme": "three_degenerate",
            "N_ur": 0.00641,
            "neutrino_mass_fac": 93.14,
        }
    }

    translated = translator.changebasis_class(_parameters())

    assert translated["N_ur"] == 0.00641
    assert translated["m_ncdm"] == "0.02,0.02,0.02"
    assert "T_ncdm" not in translated
    assert "Omega_ncdm" not in translated
    assert translated["Omega_cdm"] == pytest.approx(0.314571 - 0.049199 - 0.06 / 93.14 / 0.6737**2)


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


def test_class_pk_grid_supports_cb_array():
    class FakeClassWithCb:
        def get_pk_cb_array(self, k, z, k_size, z_size, nonlinear):
            assert (k_size, z_size, nonlinear) == (2, 2, True)
            return np.array([10.0, 20.0, 30.0, 40.0])

        def get_pk_array(self, k, z, k_size, z_size, nonlinear):
            raise AssertionError(
                "get_pk_array should not be called when cb=True and get_pk_cb_array is available"
            )

    result = _class_pk_grid(
        FakeClassWithCb(), np.array([0.1, 1.0]), np.array([0.0, 1.0]), nonlinear=True, cb=True
    )
    np.testing.assert_array_equal(result, np.array([[10.0, 30.0], [20.0, 40.0]]))


def test_class_nonlinear_pk_grid_excludes_class_padding():
    from cosmicfishpie.cosmology.cosmology import _class_nonlinear_pk_grid

    class FakeClass:
        pars = {"P_k_max_1/Mpc": 1.0, "z_max_pk": 2.0}

        def get_pk_array(self, k, z, k_size, z_size, nonlinear):
            np.testing.assert_array_equal(k, [0.1, 1.0])
            np.testing.assert_array_equal(z, [0.0, 1.0, 2.0])
            assert (k_size, z_size, nonlinear) == (2, 3, True)
            return np.arange(1.0, 7.0)

    grid, k, z = _class_nonlinear_pk_grid(
        FakeClass(),
        np.array([0.1, 1.0, 1.2]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )

    np.testing.assert_array_equal(k, [0.1, 1.0])
    np.testing.assert_array_equal(z, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(grid, np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]))


def test_class_nonlinear_pk_grid_rejects_nonfinite_supported_values():
    from cosmicfishpie.cosmology.cosmology import _class_nonlinear_pk_grid

    class FakeClass:
        pars = {"P_k_max_1/Mpc": 1.0, "z_max_pk": 2.0}

        def get_pk_array(self, k, z, k_size, z_size, nonlinear):
            return np.array([1.0, 2.0, 3.0, np.inf])

    with pytest.raises(FloatingPointError, match="1 non-finite P_m"):
        _class_nonlinear_pk_grid(FakeClass(), np.array([0.1, 1.0]), np.array([0.0, 2.0]))


def test_cosmo_functions_pmm_pcb_nonnegative_clamping():
    from cosmicfishpie.cosmology.cosmology import cosmo_functions

    cf = object.__new__(cosmo_functions)
    cf.code = "class"
    cf.results = SimpleNamespace(
        Pk_nl=lambda z, k, grid=False: np.array([-5.0, 10.0]),
        Pk_l=lambda z, k, grid=False: np.array([-2.0, 4.0]),
        Pk_cb_nl=lambda z, k, grid=False: np.array([-1.0, 8.0]),
        Pk_cb_l=lambda z, k, grid=False: np.array([-0.5, 3.0]),
    )
    np.testing.assert_array_equal(cf.Pmm(0.5, [0.1, 1.0], nonlinear=True), [0.0, 10.0])
    np.testing.assert_array_equal(cf.Pmm(0.5, [0.1, 1.0], nonlinear=False), [0.0, 4.0])
    np.testing.assert_array_equal(cf.Pcb(0.5, [0.1, 1.0], nonlinear=True), [0.0, 8.0])
    np.testing.assert_array_equal(cf.Pcb(0.5, [0.1, 1.0], nonlinear=False), [0.0, 3.0])


def test_class_ee2_boost_converts_units_preserves_order_and_checks_range(monkeypatch):
    class FakeClass:
        pars = {"m_ncdm": "0.02,0.02,0.02", "n_s": 0.96}

        def h(self):
            return 0.5

        def Omega_b(self):
            return 0.05

        def Omega_m(self):
            return 0.3

        def get_primordial(self):
            return {"k [1/Mpc]": [0.01, 0.05, 0.1], "P_scalar(k)": [2e-9] * 3}

    class FakeEmulator:
        def get_boost(self, parameters, redshifts):
            assert parameters["Omega_m"] == 0.3
            return np.array([0.01, 1.0, 2.0]), {
                index: np.array([1.0, 1.0 + redshift, 2.0 + redshift])
                for index, redshift in enumerate(redshifts)
            }

    monkeypatch.setitem(sys.modules, "euclidemu2", SimpleNamespace(PyEuclidEmulator=FakeEmulator))
    classres = FakeClass()
    parameters = class_ee2_parameters(classres)
    assert parameters["A_s"] == pytest.approx(2e-9)
    assert parameters["m_ncdm"] == pytest.approx(0.06)
    boost = class_ee2_boost(classres, [0.001, 0.5], [2.0, 0.0])
    np.testing.assert_allclose(boost, [[1.0, 1.0], [3.0, 1.0]])
    with pytest.raises(ValueError, match="exceeds EE2 maximum"):
        class_ee2_boost(classres, [1.1], [0.0])


@pytest.mark.parametrize("amplitude", [{"10^9As": 2.1}, {"sigma8": 0.81}])
def test_class_ee2_profile_multiplies_linear_spectra(amplitude):
    pytest.importorskip("classy")
    pytest.importorskip("euclidemu2")
    profile = Path(__file__).resolve().parents[1] / (
        "cosmicfishpie/configs/default_boltzmann_yaml_files/class/ee2_boost.yaml"
    )
    with profile.open() as config_file:
        parameters = yaml.safe_load(config_file)
    settings = {
        "feedback": 0,
        "SUPPRESS_WARNINGS": False,
        "cosmo_model": "LCDM",
        "ShareDeltaNeff": True,
        "nonlinear": True,
    }
    configuration = SimpleNamespace(
        settings=settings, input_type="class", backend_parameters=parameters
    )
    cosmo = boltzmann_code(
        {
            "h": 0.6737,
            "Omegam": 0.314571,
            "Omegab": 0.049199,
            "ns": 0.96605,
            "mnu": 0.06,
            **amplitude,
        },
        code="class",
        configuration=configuration,
    )
    assert cosmo.nonlinear_model == "ee2"
    assert "NONLINEAR" not in cosmo.classcosmopars
    for power in ("Pk", "Pk_cb"):
        linear = getattr(cosmo.results, f"{power}_l")
        nonlinear = getattr(cosmo.results, f"{power}_nl")
        assert np.isfinite(nonlinear(0.5, 0.2))
        assert nonlinear(0.5, 0.2)[0, 0] > linear(0.5, 0.2)[0, 0]


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
    assert "As" in fm.fiducialcosmo.symbcosmopars
    assert not hasattr(fm.fiducialcosmo.results, "Pk_cb_l")
    assert np.isfinite(fm.fiducialcosmo.results.Pk_nl(0.5, 1e-2))


def test_class_changebasis_cosmicfishpie_neutrino_scheme():
    translator = _translator("LCDM")
    translator.configuration = SimpleNamespace(
        backend_parameters={
            "PARAMETER_TRANSLATION": {
                "neutrino_scheme": "cosmicfishpie",
                "neutrino_mass_fac": 93.14,
            }
        }
    )
    pars = translator.changebasis_class({"h": 0.67, "Omegab": 0.05, "Omegam": 0.3, "mnu": 0.06})
    assert "Omega_ncdm" in pars
    assert pars["Omega_ncdm"] > 0


def test_class_hmcode_pk_cb_nonnegative():
    pytest.importorskip("classy")
    profile = Path(__file__).resolve().parents[1] / (
        "cosmicfishpie/configs/default_boltzmann_yaml_files/class/syren_new_minimal_matched.yaml"
    )
    with profile.open() as config_file:
        parameters = yaml.safe_load(config_file)
    settings = {
        "feedback": 0,
        "SUPPRESS_WARNINGS": False,
        "cosmo_model": "w0waCDM",
        "ShareDeltaNeff": True,
        "nonlinear": True,
    }
    configuration = SimpleNamespace(
        settings=settings, input_type="class", backend_parameters=parameters
    )
    cosmo = boltzmann_code(
        {
            "h": 0.67,
            "Omegam": 0.32,
            "Omegab": 0.049,
            "ns": 0.96,
            "10^9As": 2.1,
            "mnu": 0.06,
            "w0": -1.0,
            "wa": -0.05,
        },
        code="class",
        configuration=configuration,
    )
    k = np.geomspace(1e-4, 30.0, 30)
    for z in [0.0, 1.0, 2.0]:
        pk_cb = cosmo.results.Pk_cb_nl(z, k, grid=False)
        pk_m = cosmo.results.Pk_nl(z, k, grid=False)
        assert np.all(pk_cb >= 0.0)
        assert np.all(pk_m >= 0.0)


def test_photometric_yaml_profiles_load_and_run():
    from cosmicfishpie.configs.context import build_analysis_context
    from cosmicfishpie.cosmology.cosmology import cosmo_functions

    fiducial = dict(FULL_FIDUCIAL)
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "cosmicfishpie/configs/default_boltzmann_yaml_files"

    profiles = [
        ("symbolic", "symbolic_config_yaml", base_dir / "symbolic/syren_new_photo.yaml"),
        ("class", "class_config_yaml", base_dir / "class/hmcode2020_photo.yaml"),
    ]
    if find_spec("euclidemu2") is not None:
        profiles.append(("class", "class_config_yaml", base_dir / "class/ee2_boost_photo.yaml"))

    for code, key, path in profiles:
        assert path.is_file(), f"Missing profile {path}"
        ctx = build_analysis_context(
            options={
                "code": code,
                "cosmo_model": "LCDM",
                "accuracy": 1,
                "nonlinear": True,
                "feedback": 0,
                key: str(path),
            },
            observables=["GCph", "WL"],
            freepars={},
            fiducialpars=fiducial,
            survey_name="Euclid",
            cosmo_model="LCDM",
        )
        cosmo = cosmo_functions(fiducial, configuration=ctx)
        p_nl = cosmo.Pmm(0.5, 1.0, nonlinear=True)
        assert np.isfinite(p_nl)
        assert p_nl > 0.0
        assert cosmo.results.kgrid[-1] > 5.0  # extended optimistic photo reach
