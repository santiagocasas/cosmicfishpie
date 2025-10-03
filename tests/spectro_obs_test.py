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
        configuration=cosmoFM,
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
    bterm = nuisance.gcsp_bias_at_z(z)
    bterm_2 = nuisance.gcsp_bias_at_zi(bin_ind)
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


def test_kpar_calculation(spectro_obs):
    z, k, mu = 1.0, 0.1, 0.5
    kpar = spectro_obs.kpar(z, k, mu)
    assert isinstance(kpar, (float, np.ndarray))
    assert kpar > 0
    # Should be approximately k * mu / qpar
    expected = k * mu / spectro_obs.qparallel(z)
    assert np.isclose(kpar, expected)


def test_kper_calculation(spectro_obs):
    z, k, mu = 1.0, 0.1, 0.5
    kper = spectro_obs.kper(z, k, mu)
    assert isinstance(kper, (float, np.ndarray))
    assert kper > 0
    # Should be approximately k * sqrt(1-mu^2) / qperp
    expected = k * (1 - mu**2) ** 0.5 / spectro_obs.qperpendicular(z)
    assert np.isclose(kper, expected)


def test_spec_err_z(spectro_obs):
    z, k, mu = 1.0, 0.1, 0.5
    err = spectro_obs.spec_err_z(z, k, mu)
    assert isinstance(err, (float, np.ndarray))
    assert 0 < err <= 1  # Error suppression should be between 0 and 1


def test_BAO_term_no_ap(spectro_obs):
    spectro_obs.APbool = False
    z = 1.0
    bao = spectro_obs.BAO_term(z)
    assert bao == 1  # Should be 1 when AP is disabled


def test_BAO_term_with_ap(spectro_obs):
    spectro_obs.APbool = True
    z = 1.0
    bao = spectro_obs.BAO_term(z)
    assert isinstance(bao, (float, np.ndarray))
    assert bao > 0


def test_kaiser_term(spectro_obs):
    z, k, mu = 1.0, 0.1, 0.5
    kaiser = spectro_obs.kaiserTerm(z, k, mu, bias_sample="g")
    assert isinstance(kaiser, (float, np.ndarray))
    assert kaiser > 0


def test_fingers_of_god(spectro_obs):
    spectro_obs.FoG_switch = True
    spectro_obs.linear_switch = False
    z, k, mu = 1.0, 0.1, 0.5
    fog = spectro_obs.FingersOfGod(z, k, mu, mode="Lorentz")
    assert isinstance(fog, (float, np.ndarray))
    assert 0 < fog <= 1  # FoG should suppress power


def test_observed_Pgg(spectro_obs):
    z, k, mu = 1.0, 0.1, 0.5
    p_obs = spectro_obs.observed_Pgg(z, k, mu)
    assert isinstance(p_obs, (float, np.ndarray))
    assert p_obs > 0  # Power spectrum should be positive


def test_lnpobs_gg(spectro_obs):
    z, k, mu = 1.0, 0.1, 0.5
    ln_p_obs = spectro_obs.lnpobs_gg(z, k, mu)
    assert isinstance(ln_p_obs, (float, np.ndarray))
    # Check if log value is reasonable (not too negative)
    assert ln_p_obs > -50  # Should not be extremely negative


def test_k_units_change(spectro_obs):
    spectro_obs.kh_rescaling_bug = False
    k_input = 0.1
    k_output = spectro_obs.k_units_change(k_input)
    assert k_output == k_input  # Should be unchanged when bug is disabled


def test_normalized_pdd(spectro_obs):
    z, k = 1.0, 0.1
    p_dd = spectro_obs.normalized_pdd(z, k)
    assert isinstance(p_dd, (float, np.ndarray))
    assert p_dd > 0


def test_dewiggled_pdd(spectro_obs):
    z, k, mu = 1.0, 0.1, 0.5
    p_dw = spectro_obs.dewiggled_pdd(z, k, mu)
    assert isinstance(p_dw, (float, np.ndarray))
    assert p_dw > 0


def test_P_ThetaTheta_Moments(spectro_obs):
    z = 1.0
    for moment in [0, 1, 2]:
        ptt = spectro_obs.P_ThetaTheta_Moments(z, moment=moment)
        assert isinstance(ptt, (float, np.ndarray))
        assert ptt >= 0  # Should be non-negative


def test_sigmapNL_linear_switch(spectro_obs):
    spectro_obs.linear_switch = True
    z = 1.0
    sigma_p = spectro_obs.sigmapNL(z)
    assert sigma_p == 0  # Should be 0 in linear regime


def test_sigmavNL_linear_switch(spectro_obs):
    spectro_obs.linear_switch = True
    z, mu = 1.0, 0.5
    sigma_v = spectro_obs.sigmavNL(z, mu)
    assert sigma_v == 0  # Should be 0 in linear regime
