from types import SimpleNamespace

import numpy as np
import pytest

from cosmicfishpie.cosmology.nuisance import Nuisance


def _base_specs():
    return {
        "survey_name": "TestSurvey",
        "z_bins_WL": [0.0, 0.5, 1.0],
        "z_bins_GCph": [0.0, 0.5, 1.0],
        "binrange_GCph": [1, 2],
        "z_bins_sp": {1: np.array([0.9, 1.1]), 2: np.array([1.1, 1.3])},
        "dndOmegadz": {1: 10.0, 2: 20.0},
        "sp_bias_sample": "g",
        "sp_bias_root": "b",
        "sp_bias_model": "linear",
        "sp_bias_parametrization": "bins",
        "z_bins_IM": {1: np.array([0.1, 0.3]), 2: np.array([0.3, 0.5])},
        "IM_bias_sample": "I",
        "IM_bias_root": "bI",
        "IM_bias_model": "fitting",
        "IM_bias_parametrization": "simple",
        "THI_sys_noise": {
            "z_vals_THI": np.array([0.0, 1.0, 2.0, 3.0]),
            "THI_sys_noise": np.array([1.0, 2.0, 3.0, 4.0]),
        },
    }


def _make_config(observables):
    return SimpleNamespace(
        obs=observables,
        specs=_base_specs(),
        settings={
            "specs_dir": ".",
            "external_data_dir": "/definitely/nonexistent/path",
            "accuracy": 1,
            "pivot_z_IA": 0.62,
            "Pshot_nuisance_fiducial": 0,
        },
        Spectrobiasparams={"bg_1": 1.5, "bg_2": 1.7},
        Spectrononlinearparams={"sigmap_1": 1.8},
        IMbiasparams={"bI_c1": 0.8, "bI_c2": 0.4},
    )


def test_luminosity_ratio_missing_file_returns_unity():
    nuisance = Nuisance(configuration=_make_config(["WL"]))
    assert nuisance.lumratio(0.5) == pytest.approx(1.0)
    np.testing.assert_allclose(nuisance.lumratio(np.array([0.2, 0.4])), np.ones(2))


def test_gcph_bias_binned_model_clips_out_of_range():
    nuisance = Nuisance(configuration=_make_config(["GCph"]))
    bfunc = nuisance.gcph_bias({"bias_model": "binned", "b1": 1.2, "b2": 1.8})
    vals = bfunc(np.array([-1.0, 0.1, 0.8, 10.0]))
    np.testing.assert_allclose(vals, np.array([1.2, 1.2, 1.8, 1.8]))


def test_gcsp_zvalue_to_zindex_clamps_bounds():
    nuisance = Nuisance(configuration=_make_config(["GCsp"]))
    assert nuisance.gcsp_zvalue_to_zindex(0.01) == 1
    assert nuisance.gcsp_zvalue_to_zindex(5.0) == 2


def test_gcsp_rescale_sigmapv_default_and_named_key():
    nuisance = Nuisance(configuration=_make_config(["GCsp"]))
    assert nuisance.gcsp_rescale_sigmapv_at_z(0.95, sigma_key="sigmap") == pytest.approx(1.8)
    # second bin key missing -> default 1.0
    assert nuisance.gcsp_rescale_sigmapv_at_z(1.25, sigma_key="sigmap") == pytest.approx(1.0)


def test_ia_enla_returns_callable_spline():
    nuisance = Nuisance(configuration=_make_config(["WL"]))

    class DummyCosmo:
        @staticmethod
        def Omegam_of_z(_z):
            return 0.3

        @staticmethod
        def growth(z):
            return np.ones_like(np.atleast_1d(z), dtype=float)

    iaf = nuisance.IA(
        {"IA_model": "eNLA", "AIA": 1.0, "etaIA": 0.0, "betaIA": 0.0},
        DummyCosmo(),
    )
    vals = iaf(np.array([0.2, 0.7]))
    assert np.all(np.isfinite(vals))


def test_im_bias_fitting_and_thi_noise_interp():
    nuisance = Nuisance(configuration=_make_config(["IM"]))
    assert nuisance.IM_bias_at_z(1.0) == pytest.approx(0.8 * (1 + 1.0) + 0.4)

    tsys = nuisance.IM_THI_noise()
    assert tsys(1.0) == pytest.approx(2.0, rel=1e-6)
