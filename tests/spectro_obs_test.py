from cosmicfishpie.LSSsurvey.spectro_obs import ComputeGalSpectro
from cosmicfishpie.cosmology.nuisance import Nuisance
import cosmicfishpie.configs.config as cfg
import pytest
import numpy as np


@pytest.fixture
def spectro_obs(spectro_fisher_matrix):
    cosmopars = {"Omegam": 0.3, "h": 0.7}  # Add more parameters as needed
    cosmoFM = spectro_fisher_matrix
    spectro_obs = ComputeGalSpectro(
        cosmopars=cosmopars,
        fiducial_cosmopars=cosmoFM.fiducialcosmopars,
        spectrobiaspars=cosmoFM.Spectrobiaspars,
        spectrononlinearpars=cosmoFM.Spectrononlinpars,
        PShotpars=cosmoFM.PShotpars,
        fiducial_cosmo=cosmoFM.fiducialcosmo,
        bias_samples=["g", "g"],
        use_bias_funcs=False,
    )
    return spectro_obs


def test_spectro_obs_initialization(spectro_obs):
    cosmopars = {"Omegam": 0.3, "h": 0.7}  # Add more parameters as needed
    assert isinstance(spectro_obs, ComputeGalSpectro)
    assert spectro_obs.cosmopars == cosmopars


def test_bterm_fid(spectro_obs):
    z = 1.2  # 1.2 is central value of second bin
    bin_ind = 2
    print(spectro_obs.spectrobiaspars)
    print(spectro_obs.bterm_fid(z, bias_sample="g"))
    nuisance = Nuisance()
    print(spectro_obs.gcsp_z_bin_mids)
    bkey, bval = nuisance.bterm_z_key(
        bin_ind, spectro_obs.gcsp_z_bin_mids, spectro_obs.fiducialcosmo, bias_sample="g"
    )
    bterm = spectro_obs.bterm_fid(z, bias_sample="g")
    assert isinstance(bterm, float)
    assert np.isclose(bterm, 1.6060949)
    assert np.isclose(bval, bterm)
    assert bkey == "bg_2"


def test_qparallel(spectro_obs):
    z = 1.0
    qpar = spectro_obs.qparallel(z)
    assert isinstance(qpar, float)
    assert qpar > 0  # qparallel should always be positive


def test_qperpendicular(spectro_obs):
    z = 1.0
    qperp = spectro_obs.qperpendicular(z)
    assert isinstance(qperp, float)
    assert qperp > 0  # qperpendicular should always be positive
