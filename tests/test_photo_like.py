import math
from pathlib import Path

import numpy as np
import pytest

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.likelihood import PhotometricLikelihood


SPEC_DIR = Path(__file__).resolve().parents[1] / "cosmicfishpie" / "configs" / "default_survey_specifications"


@pytest.fixture(scope="module")
def photometric_fisher_matrix():
    options = {
        "accuracy": 1,
        "feedback": 1,
        "code": "symbolic",
        "outroot": "test_photo_like",
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "survey_name_spectro": False,
        "specs_dir": "../cosmicfishpie/configs/default_survey_specifications/",
        #"specs_dir": str(SPEC_DIR) + "/",
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
def photometric_likelihood(photometric_fisher_matrix):
    return PhotometricLikelihood(
        cosmoFM_data=photometric_fisher_matrix,
        cosmoFM_theory=photometric_fisher_matrix,
        observables=photometric_fisher_matrix.observables,
    )


def _sample_params():
    samp_pars = {
    'Omegam': 0.3145714273,
    'Omegab': 0.0491989,
    'h': 0.6737,
    'ns': 0.96605,
    'sigma8': 0.81,
    'w0': -1.0,
    'wa': 0.0,
    'mnu': 0.06,
    'Neff': 3.044,
    'bias_model': 'binned',
    'b1': 1.0997727037892875,
    'b2': 1.220245876862528,
    'b3': 1.2723993083933989,
    'b4': 1.316624471897739,
    'b5': 1.35812370570578,
    'b6': 1.3998214171814918,
    'b7': 1.4446452851824907,
    'b8': 1.4964959071110084,
    'b9': 1.5652475842498528,
    'b10': 1.7429859437184225,
    'fout': 0.1,
    'co': 1,
    'cb': 1,
    'sigma_o': 0.05,
    'sigma_b': 0.05,
    'zo': 0.1,
    'zb': 0.0,
    'IA_model': 'eNLA',
    'AIA': 1.72,
    'betaIA': 2.17,
    'etaIA': -0.41*1.1
    }
    return samp_pars


def test_photometric_cells_have_expected_shape(photometric_likelihood):
    cells = photometric_likelihood.data_obs
    assert "ells" in cells
    assert "Cell_GG" in cells
    #assert "Cell_LL" in cells
    #assert "Cell_GL" in cells
    assert cells["Cell_GG"].shape[0] == len(cells["ells"])


def test_photometric_loglike_matches_notebook_value(photometric_likelihood):
    sample_params = _sample_params()
    loglike_value = photometric_likelihood.loglike(param_dict=sample_params)
    expected = 4.038266295364123e-11
    assert math.isclose(loglike_value, expected, rel_tol=1e-3, abs_tol=1e-12)


def test_photometric_cell_entry_matches_theory(photometric_likelihood):
    sample_params = _sample_params()
    ells = photometric_likelihood.data_obs["ells"]
    target_ell = 300
    idx = int(np.argmin(np.abs(ells - target_ell)))

    data_val = photometric_likelihood.data_obs["Cell_GG"][idx, 1, 8]
    theory_cells = photometric_likelihood.compute_theory(dict(sample_params))
    theory_val = theory_cells["Cell_GG"][idx, 1, 8]
    nb_val = 9.974414863747675e-14
    assert theory_val == pytest.approx(nb_val, rel=1e-12, abs=1e-18)
