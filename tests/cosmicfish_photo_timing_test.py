"""Tests for the photometric branch of ``FisherMatrix.compute`` with timing enabled."""

import numpy as np

from cosmicfishpie.fishermatrix import cosmicfish as cff


def test_photometric_fisher_compute_with_timing():
    options = {
        "accuracy": 1,
        "outroot": "test_photo_timing",
        "results_dir": "results/",
        "derivatives": "3PT",
        "ell_sampling": 25,
        "nonlinear": True,
        "feedback": 0,
        "timing": True,
        "specs_dir": "cosmicfishpie/configs/default_survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": "symbolic",
    }

    fiducial = {
        "Omegam": 0.32,
        "h": 0.67,
    }

    freepars = {
        "Omegam": 0.01,
        "h": 0.01,
    }

    fisher = cff.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=["WL", "GCph"],
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )

    assert fisher.settings["timing"] is True

    fishanalysis = fisher.compute()
    fm = fishanalysis.fisher_matrix

    assert fm.shape[0] == fm.shape[1]
    assert fm.shape[0] > 0
    assert np.all(np.isfinite(fm))
    np.testing.assert_allclose(fm, fm.T, rtol=1e-8, atol=0.0)
    assert np.all(np.diag(fm) > 0.0)
