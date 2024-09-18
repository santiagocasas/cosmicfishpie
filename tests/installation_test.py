from cosmicfishpie.fishermatrix import cosmicfish as cff
from cosmicfishpie.utilities.utils import printing as cpr

def test_installation():
    # These are typical options that you can pass to Cosmicfishpie
    options = {
        "accuracy": 1,
        "outroot": "test_installation_test_EBS",
        "results_dir": "results/",
        "derivatives": "3PT",
        "nonlinear": True,
        "feedback": 2,
        "specs_dir": "survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": "symbolic",
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

    observables = ["GCsp"]
    cpr.debug = True
    cosmoFM = cff.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )

    cpr.debug = False
    fish = cosmoFM.compute(max_z_bins=1)
    print("Fisher name: ", fish.name)
    print("Fisher parameters: ", fish.get_param_names())
    print("Fisher fiducial values: ", fish.get_param_fiducial())
    print("Fisher confidence bounds: ", fish.get_confidence_bounds())
    print("Fisher covariance matrix: ", fish.fisher_matrix_inv)
