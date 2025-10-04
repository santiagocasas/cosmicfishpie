import pytest

from cosmicfishpie.fishermatrix import cosmicfish as cff
from cosmicfishpie.utilities.utils import printing as upt

upt.debug = True
code_to_use = "symbolic"
upt.debug = True


@pytest.fixture(scope="session")
def photo_fisher_matrix():
    # These are typical options that you can pass to Cosmicfishpie
    options = {
        "accuracy": 1,
        "outroot": "test_photo_low_ellsamp",
        "results_dir": "results/",
        "derivatives": "3PT",
        "ell_sampling": 25,
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


@pytest.fixture(scope="session")
def computecls_fid(photo_fisher_matrix):
    """Module-scoped fiducial ComputeCls instance reused across photo tests to avoid recomputation."""
    from cosmicfishpie.LSSsurvey.photo_obs import ComputeCls

    cosmoFM = photo_fisher_matrix
    cosmopars = {"Omegam": 0.3, "h": 0.7}
    cls = ComputeCls(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars)
    cls.compute_all()
    return cosmopars, cls, cosmoFM


@pytest.fixture(scope="session")
def photo_cov_cached(computecls_fid):
    """Session-scoped PhotoCov object with precomputed covariance to avoid recomputation across tests.

    Derivatives are intentionally not precomputed here (and will be stubbed in the derivative test).
    """
    from cosmicfishpie.LSSsurvey.photo_cov import PhotoCov

    cosmopars, fid_cls, cosmoFM = computecls_fid
    pc = PhotoCov(
        cosmopars,
        cosmoFM.photopars,
        cosmoFM.IApars,
        cosmoFM.photobiaspars,
        fiducial_Cls=fid_cls,
    )
    # Precompute covariance once (counts toward coverage while saving later duplication)
    pc.compute_covmat()
    return pc


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
        "specs_dir": "cosmicfishpie/configs/default_survey_specifications/",
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
    # for testing purposes, we use a smaller sigma_dz
    specifs = {
        "spec_sigma_dz": 0.001,
    }

    cosmoFM = cff.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
        specifications=specifs,
    )

    return cosmoFM
