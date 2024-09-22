import pytest

from cosmicfishpie.fishermatrix import cosmicfish as cff

code_to_use = "camb"


@pytest.fixture(scope="module")
def photo_fisher_matrix():
    # These are typical options that you can pass to Cosmicfishpie
    options = {
        "accuracy": 1,
        "outroot": "test_installation_test_EBS",
        "results_dir": "results/",
        "derivatives": "3PT",
        "nonlinear": True,
        "feedback": 2,
        "specs_dir": "cosmicfishpie/configs/default_survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": code_to_use,
    }

    # Internally CosmicFish converts these parameters to the corresponding parameters in CAMB or CLASS
    fiducial = {
        "Omegam": 0.32,
        "h": 0.67,
    }

    # Parameters to be varied and analyzed and their percentage variation for numerical derivatives
    freepars = {
        "Omegam": 0.01,
        "h": 0.01,
    }

    observables = ["WL", "GCph"]

    cosmoFM = cff.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )

    return cosmoFM


@pytest.fixture(scope="module")
def spectro_fisher_matrix():
    # These are typical options that you can pass to Cosmicfishpie
    options = {
        "accuracy": 1,
        "outroot": "test_installation_test_EBS",
        "results_dir": "results/",
        "derivatives": "3PT",
        "nonlinear": False,
        "feedback": 2,
        "specs_dir": "survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": code_to_use,
    }

    options["vary_bias_str"] = "b"
    options["bfs8terms"] = False
    # Internally CosmicFish converts these parameters to the corresponding parameters in CAMB or CLASS
    fiducial = {
        "Omegam": 0.32,
        "h": 0.67,
    }

    # Parameters to be varied and analyzed and their percentage variation for numerical derivatives
    freepars = {
        "Omegam": 0.01,
        "h": 0.01,
    }

    observables = ["GCsp"]

    cosmoFM = cff.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )

    return cosmoFM