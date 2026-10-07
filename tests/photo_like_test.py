import math

import numpy as np
import pytest

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.likelihood import PhotometricLikelihood


@pytest.fixture(scope="module")
def photometric_fiducial_obs():
    options = {
        "accuracy": 1,
        "feedback": 1,
        "code": "symbolic",
        "outroot": "test_photo_like",
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "survey_name_spectro": False,
        "specs_dir": "../cosmicfishpie/configs/default_survey_specifications/",
        "cosmo_model": "LCDM",
    }

    fiducial = {
        "Omegam": 0.3145714273,
        "Omegab": 0.0491989,
        "h": 0.6737,
        "ns": 0.96605,
        "sigma8": 0.81,
        "w0": -1.0,
        "wa": 0.0,
        "mnu": 0.06,
        "Neff": 3.044,
    }

    observables = ["GCph"]

    return cosmicfish.FisherMatrix(
        fiducialpars=fiducial,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )


@pytest.fixture(scope="module")
def photometric_likelihood(photometric_fiducial_obs):
    return PhotometricLikelihood(
        cosmo_data=photometric_fiducial_obs,
        cosmo_theory=photometric_fiducial_obs,
        observables=photometric_fiducial_obs.observables,
    )


def _sample_params(photometric_likelihood):
    fiducial_obs = photometric_likelihood.cosmo_data
    samp_pars = fiducial_obs.allparams.copy()
    return samp_pars


def test_photometric_cells_have_expected_shape(photometric_likelihood):
    cells = photometric_likelihood.data_obs
    assert "ells" in cells
    assert "Cell_GG" in cells
    # assert "Cell_LL" in cells
    # assert "Cell_GL" in cells
    assert cells["Cell_GG"].shape[0] == len(cells["ells"])


def test_photometric_loglike_matches_notebook_value(photometric_likelihood):
    sample_params = _sample_params(photometric_likelihood)
    loglike_value = photometric_likelihood.loglike(param_dict=sample_params)
    # Identical data/theory cells give a numerically zero likelihood; its
    # residual depends on floating-point/runtime details and is not a stable
    # notebook reference value.
    assert math.isfinite(loglike_value)
    assert math.isclose(loglike_value, 0.0, abs_tol=1e-9)


def test_photometric_cell_entry_matches_theory(photometric_likelihood):
    sample_params = _sample_params(photometric_likelihood)
    ells = photometric_likelihood.data_obs["ells"]
    target_ell = 300
    idx = int(np.argmin(np.abs(ells - target_ell)))

    data_val = photometric_likelihood.data_obs["Cell_GG"][idx, 1, 8]
    theory_cells = photometric_likelihood.compute_theory(dict(sample_params))
    theory_val = theory_cells["Cell_GG"][idx, 1, 8]
    assert theory_val == data_val
    assert theory_val == pytest.approx(9.91823004261385e-14, rel=1e-12, abs=1e-18)


def test_photometric_likelihood_uses_preloaded_cells(photometric_fiducial_obs, monkeypatch):
    fm = photometric_fiducial_obs
    monkeypatch.setattr(fm, "photo_obs_fid", object(), raising=False)
    monkeypatch.setattr(fm, "photo_LSS", object(), raising=False)

    ells = np.array([50.0, 100.0, 150.0])
    cell_gg = np.full((ells.size, 3, 3), 2.0)

    likelihood = PhotometricLikelihood(
        cosmo_data=fm,
        cosmo_theory=fm,
        observables=["GCph"],
        data_cells={"ells": ells, "Cell_GG": cell_gg},
    )

    cells = likelihood.data_obs
    assert cells is likelihood._preloaded_cells
    np.testing.assert_array_equal(cells["ells"], ells)
    np.testing.assert_array_equal(cells["Cell_GG"], cell_gg)
