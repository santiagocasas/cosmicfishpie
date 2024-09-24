import numpy as np
import pytest

from cosmicfishpie.cosmology.nuisance import Nuisance
from cosmicfishpie.LSSsurvey.spectro_obs import ComputeGalSpectro


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
    nuisance = Nuisance(spectrobiasparams=spectro_obs.spectrobiaspars)
    bterm = nuisance.gscp_bias_at_z(z)
    bterm_2 = nuisance.gscp_bias_at_zi(bin_ind)
    assert isinstance(bterm, float)
    assert isinstance(bterm_2, float)
    assert np.isclose(bterm, 1.6060949)
    assert np.isclose(bterm_2, bterm)


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
