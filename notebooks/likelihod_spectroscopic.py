#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from nautilus import Prior, Sampler
from scipy.integrate import simpson
from scipy.interpolate import make_interp_spline
from scipy.special import eval_legendre

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey import spectro_cov as spcov
from cosmicfishpie.LSSsurvey import spectro_obs as spobs
from cosmicfishpie.utilities import legendre_tools as lgt
from cosmicfishpie.utilities.utils import printing as upr

# In[2]:


# In[3]:


# In[4]:


def is_indexable_iterable(var):
    return isinstance(var, (list, np.ndarray, Sequence)) and not isinstance(var, (str, bytes))


# In[5]:


logger = logging.getLogger("cosmicfishpie.cosmology.nuisance")
logger.setLevel(logging.INFO)


# In[6]:


upr.debug = False


# In[7]:


upr.debug_print("test")


# In[8]:


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
observables = ["GCsp"]


# In[ ]:


options = {
    "accuracy": 1,
    "feedback": 1,
    "code": "symbolic",
    "outroot": "GCsp",
    "survey_name": "Euclid",
    # "survey_name_spectro": "SKAO-Spectroscopic-Redbook",
    "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
    "survey_name_photo": False,
    "survey_name_radio_IM": "SKAO-IM-Redbook",
    #'specs_dir': '../cosmicfishpie/configs/other_survey_specifications/',
    "cosmo_model": "LCDM",
    "bfs8terms": False,
}
cosmoFM_fid = cosmicfish.FisherMatrix(
    fiducialpars=fiducial,
    options=options,
    observables=observables,
    cosmoModel=options["cosmo_model"],
    surveyName=options["survey_name"],
)


# In[ ]:


print(cosmoFM_fid.Spectrobiaspars)


# In[ ]:


print(cosmoFM_fid.freeparams)


# In[12]:


cosmoFM_fid.set_pk_settings()

spectro_fid = spobs.ComputeGalSpectro(
    cosmoFM_fid.fiducialcosmopars,
    spectrobiaspars=cosmoFM_fid.Spectrobiaspars,
    spectrononlinearpars=cosmoFM_fid.Spectrononlinpars,
    PShotpars=cosmoFM_fid.PShotpars,
    IMbiaspars=cosmoFM_fid.IMbiaspars,
    fiducial_cosmo=cosmoFM_fid.fiducialcosmo,
    use_bias_funcs=False,
    configuration=cosmoFM_fid,
)

spectro_cov_fid = spcov.SpectroCov(
    fiducialpars=cosmoFM_fid.fiducialcosmopars, configuration=cosmoFM_fid
)


volume_surveys = np.array(
    [spectro_cov_fid.volume_survey(ii) for ii in range(len(spectro_cov_fid.global_z_bin_mids))]
)


# In[13]:


def observable_Pgg(vary_spectro, nuisance_shot=None):
    z_bins = spectro_cov_fid.global_z_bin_mids
    n_bins = len(z_bins)
    si, sj = spectro_fid.obs_spectrum
    if nuisance_shot is None:
        nuisance_shot = np.zeros_like(z_bins)
    obs_Pgg = np.zeros((n_bins, cosmoFM_fid.Pk_musamp, cosmoFM_fid.Pk_ksamp))
    for ii in range(n_bins):
        obs_Pgg[ii, :, :] = (
            vary_spectro.noisy_P_ij(
                z_bins[ii], cosmoFM_fid.Pk_kmesh, cosmoFM_fid.Pk_mumesh, si=si, sj=sj
            )
            + nuisance_shot[ii]
        )
    return obs_Pgg


multipole_order = 4
leg_ells = np.arange(0, multipole_order + 1, 2, dtype="int")


def legendre_Pgg(obs_Pgg):
    legendre_vals = eval_legendre(leg_ells[None, :], cosmoFM_fid.Pk_mugrid[:, None])
    legendre_vals = legendre_vals[None, :, None, :]
    obsPgg_forleg = obs_Pgg[:, :, :, None]
    P_ell = ((2.0 * leg_ells[None, None, :] + 1)) * simpson(
        legendre_vals * obsPgg_forleg, x=cosmoFM_fid.Pk_mugrid, axis=1
    )
    P_ell = P_ell.transpose(1, 0, 2)
    return P_ell


# In[ ]:


obsPgg_fid = observable_Pgg(spectro_cov_fid)
obsPgg_fid.shape


# In[ ]:


P_ell_fid = legendre_Pgg(obsPgg_fid)
P_ell_fid.shape


# In[16]:


def compute_covariance_legendre(P_ell, volume_survey):
    """
    Compute covariance matrix for power spectrum multipoles
    P_ell shape: (n_k, n_z, n_ell)
    """
    k_grid = spectro_cov_fid.config.Pk_kgrid
    P_ell_broad = P_ell[:, :, :, None, None]
    n_k, n_z, n_ell = P_ell.shape

    # Compute auto and cross terms
    P00 = (P_ell_broad[:, :, 0]) ** 2 * lgt.m00[None, None, :, :]
    P22 = (P_ell_broad[:, :, 1]) ** 2 * lgt.m22[None, None, :, :]
    P44 = (P_ell_broad[:, :, 2]) ** 2 * lgt.m44[None, None, :, :]

    P02 = P_ell_broad[:, :, 0] * P_ell_broad[:, :, 1] * lgt.m02[None, None, :, :]
    P24 = P_ell_broad[:, :, 1] * P_ell_broad[:, :, 2] * lgt.m24[None, None, :, :]
    P04 = P_ell_broad[:, :, 0] * P_ell_broad[:, :, 2] * lgt.m04[None, None, :, :]

    # Compute mode-by-mode covariance
    mode_by_mode_covariance = (
        2
        * (2 * leg_ells[None, None, :, None] + 1)
        * (2 * leg_ells[None, None, None, :] + 1)
        / volume_survey[None, :, None, None]
        * (P00 + P22 + P44 + 2 * P02 + 2 * P24 + 2 * P04)
    )
    ln_k_grid = np.log(k_grid)
    k_size = len(k_grid)
    if n_k != k_size:
        raise ValueError("k_grid and P_ell have different lengths in k")
    ln_k_bin_edges = np.zeros((k_size + 1), "float64")
    ln_k_bin_edges[1:-1] = (ln_k_grid[1:] + ln_k_grid[:-1]) / 2.0
    ln_k_bin_edges[0] = ln_k_bin_edges[1] - (ln_k_grid[1] - ln_k_grid[0])
    ln_k_bin_edges[-1] = ln_k_bin_edges[-2] + (ln_k_grid[-1] - ln_k_grid[-2])
    k_bin_volume = 4 * np.pi / 3.0 * np.diff(np.exp(3 * ln_k_bin_edges))
    # Multiply by k^3 factor
    covariance_integrand = mode_by_mode_covariance * (k_grid[:, None, None, None]) ** 3
    covariance_integrand_spline = make_interp_spline(ln_k_grid, covariance_integrand, k=1, axis=0)

    # We create an array into which we fill the values for the covmat. We integrate over the ln_k_bin_edges
    covariance = np.zeros((k_size, n_z, n_ell, n_ell), "float64")
    for index_k in range(k_size):
        covariance[index_k, :, :, :] = covariance_integrand_spline.integrate(
            ln_k_bin_edges[index_k], ln_k_bin_edges[index_k + 1]
        )

    # We multiply the covariance matrix by this factor
    covariance *= 2 * (2 * np.pi) ** 4 / ((k_bin_volume[:, None, None, None]) ** 2)

    # We need the inverse of the covariance matrix for the chi2 computation
    inv_covariance = np.linalg.inv(covariance)

    return covariance, inv_covariance


# In[17]:


covariance_leg, inv_covariance_leg = compute_covariance_legendre(
    P_ell=P_ell_fid, volume_survey=volume_surveys  # From legendre_Pgg function
)


# In[18]:


def compute_chi2_legendre(P_ell_fid, P_ell, inv_covariance):
    """
    Compute χ² using broadcasting

    P_ell_fid: shape (n_k, n_z, n_ell)
    P_ell: shape (n_k, n_z, n_ell)
    inv_covariance: shape (n_k, n_z, n_ell, n_ell)
    """

    chi2 = np.sum(
        (P_ell_fid[..., None] - P_ell[..., None])
        * inv_covariance
        * (P_ell_fid[..., None, :] - P_ell[..., None, :])
    )

    return chi2


# In[19]:


def compute_wedge_chi2(P_obs_fid, P_obs, volume_survey):
    """
    Compute χ² for wedges using fully vectorized operations.
    Matches the loop implementation exactly.

    Parameters:
    ----------
    P_obs_fid : array_like
        Fiducial power spectrum (n_z, n_mu, n_k)
    P_obs : array_like
        Observed power spectrum (n_z, n_mu, n_k)
    volume_survey : array_like
        Fiducial volumes (volume_surveys)

    Returns:
    -------
    float
        χ² value
    """

    k_grid = spectro_cov_fid.config.Pk_kgrid
    mu_grid = spectro_cov_fid.config.Pk_mugrid
    prefactor = 8 * np.pi**2

    # Compute delta (n_z, n_mu, n_k)
    delta = P_obs - P_obs_fid

    # Prepare terms for broadcasting:
    k_term = k_grid[None, None, :] ** 2  # (1, 1, n_k)
    V_term = volume_survey[:, None, None]  # (n_z, 1, 1)

    # Compute covariance (n_z, n_mu, n_k)
    covariance = (prefactor / (k_term * V_term)) * P_obs_fid**2

    # Compute k_integrand for all points simultaneously
    k_integrand = delta**2 / covariance

    # Integrate over k for all z and mu (n_z, n_mu)
    mu_integrand = 2 * simpson(k_integrand, x=k_grid, axis=2)  # Factor 2 for mu-symmetry

    # Integrate over mu for all z (n_z)
    z_integrand = simpson(mu_integrand, x=mu_grid, axis=1)

    # Sum over z bins
    chi2 = np.sum(z_integrand)

    # print(f'Max of k_integrand: {np.max(k_integrand)}')
    # print(f'Min of k_integrand: {np.min(k_integrand)}')

    return chi2


def loglike(param_vec, prior=None, leg_flag="wedges"):

    z_bins = spectro_cov_fid.global_z_bin_mids

    if isinstance(param_vec, dict):
        param_dict = deepcopy(param_vec)
    elif is_indexable_iterable(param_vec) and prior is not None:
        # print(f'Loading prior with keys: {prior.keys}')
        param_dict = {key: param_vec[i] for i, key in enumerate(prior.keys)}

    nuisance_shot = np.zeros(len(z_bins))
    pshotpars = deepcopy(cosmoFM_fid.PShotpars)
    for ii, pp in enumerate(pshotpars.keys()):
        nuisance_shot[ii] = param_dict.pop(pp, cosmoFM_fid.PShotpars[pp])

    spectrobiaspars = deepcopy(cosmoFM_fid.Spectrobiaspars)
    for ii, pp in enumerate(spectrobiaspars.keys()):
        spectrobiaspars[pp] = param_dict.pop(pp, cosmoFM_fid.Spectrobiaspars[pp])

    spectrononlinearpars = deepcopy(cosmoFM_fid.Spectrononlinpars)
    for ii, pp in enumerate(spectrononlinearpars.keys()):
        spectrononlinearpars[pp] = param_dict.pop(pp, cosmoFM_fid.Spectrononlinpars[pp])

    IMbiaspars = deepcopy(cosmoFM_fid.IMbiasparams)
    for pp in IMbiaspars.keys():
        IMbiaspars[pp] = param_dict.pop(pp, cosmoFM_fid.IMbiasparams[pp])

    spectro_vary = spobs.ComputeGalSpectro(
        param_dict,
        spectrobiaspars=spectrobiaspars,
        spectrononlinearpars=spectrononlinearpars,
        PShotpars=cosmoFM_fid.PShotpars,
        fiducial_cosmo=cosmoFM_fid.fiducialcosmo,
        IMbiaspars=IMbiaspars,
        use_bias_funcs=False,
        configuration=cosmoFM_fid,
    )
    spectro_cov_vary = spcov.SpectroCov(
        fiducialpars=param_dict, configuration=cosmoFM_fid, fiducial_specobs=spectro_vary
    )
    obsPgg_vary = observable_Pgg(spectro_cov_vary, nuisance_shot=nuisance_shot)

    if leg_flag == "wedges":
        chi2 = compute_wedge_chi2(
            P_obs_fid=obsPgg_fid, P_obs=obsPgg_vary, volume_survey=volume_surveys
        )
    elif leg_flag == "legendre":
        P_ell_vary = legendre_Pgg(obsPgg_vary)
        covariance_leg, inv_covariance_leg = compute_covariance_legendre(
            P_ell=P_ell_fid, volume_survey=volume_surveys  # Test later with P_ell_vary
        )
        chi2 = compute_chi2_legendre(P_ell_fid, P_ell_vary, inv_covariance_leg)
    return -0.5 * chi2


# In[ ]:


fiducial


# In[ ]:


cosmoFM_fid.freeparams


# In[ ]:


fid_truth = {par: cosmoFM_fid.allparams[par] for par in cosmoFM_fid.freeparams.keys()}
fid_truth


# In[ ]:


samp1dic = {
    "Omegam": 0.31,
    "Omegab": 0.05,
    "h": 0.68,
    "ns": 0.96,
    "sigma8": 0.82,
    "w0": -1.01,
    "wa": 0.2,
    "Ps_0": 0.0,
    "Ps_1": 0.0,
    "Ps_2": 0.0,
    "Ps_3": 10.0,
    "lnbg_1": 0.37944989,
    "lnbg_2": 0.4738057,
    "lnbg_3": 0.55760176,
    "lnbg_4": 0.6,
    "sigmap_1": 1.0,
    "sigmap_2": 0.0,
    "sigmap_3": 1.0,
    "sigmap_4": 1.0,
    "sigmav_1": 1.0,
    "sigmav_2": 1.0,
    "sigmav_3": 0.0,
    "sigmav_4": 10,
    "bI_c1": -0.3,
    "bI_c2": +0.6,
}
print("Wedges loglike: ", loglike(samp1dic))
print("Multipoles loglike: ", loglike(samp1dic, leg_flag="legendre"))


# In[25]:


prior = Prior()
prior_noshot = Prior()
prior_noshot_no_bi = Prior()
prior_nosigma = Prior()


# In[26]:


prior_dict = {
    "Omegam": [0.24, 0.4],
    "Omegab": [0.04, 0.06],
    "h": [0.61, 0.73],
    "ns": [0.92, 1.00],
    "sigma8": [0.79, 0.83],
    "Ps_0": [-10, 10],
    "Ps_1": [-10, 10],
    "Ps_2": [-10, 10],
    "Ps_3": [-10, 10],
    "lnbg_1": [0, 1],
    "lnbg_2": [0, 1],
    "lnbg_3": [0, 1],
    "lnbg_4": [0, 1],
    "sigmap_1": [0, 10],
    "sigmap_2": [0, 10],
    "sigmap_3": [0, 10],
    "sigmap_4": [0, 10],
    "sigmav_1": [0, 10],
    "sigmav_2": [0, 10],
    "sigmav_3": [0, 10],
    "sigmav_4": [0, 10],
    "bI_c1": [-5, 5],
    "bI_c2": [-5, 5],
}


# In[ ]:


cosmoFM_fid.freeparams


# In[28]:


for par in prior_dict.keys():
    if par in cosmoFM_fid.freeparams.keys():
        dist_prior = (prior_dict[par][0], prior_dict[par][1])
        prior.add_parameter(par, dist_prior)
        if "sigmap" not in par and "sigmav" not in par:
            prior_nosigma.add_parameter(par, dist_prior)
        if "Ps" not in par and ("sigmap" not in par and "sigmav" not in par):
            prior_noshot.add_parameter(par, dist_prior)
        if "Ps" not in par and "lnbg" not in par and ("sigmap" not in par and "sigmav" not in par):
            prior_noshot_no_bi.add_parameter(par, dist_prior)


# In[ ]:


print("Loaded prior into Nautilus with dimension", prior.dimensionality())
print("Loaded prior into Nautilus with dimension", prior_noshot.dimensionality())
print("Loaded prior into Nautilus with dimension", prior_noshot_no_bi.dimensionality())
print("Loaded prior into Nautilus with dimension", prior_nosigma.dimensionality())
print("Prior keys: ", prior.keys)
print("Prior keys noshot: ", prior_noshot.keys)
print("Prior keys noshot no bi: ", prior_noshot_no_bi.keys)
print("Prior keys nosigma: ", prior_nosigma.keys)


# In[32]:


outfile_chain_path = "cosmicjellyfish_eucid_spectro_nonuis"
prior_to_use = prior_noshot_no_bi


# In[ ]:


print("Loading prior with keys: ", prior_to_use.keys)
sampler = Sampler(
    prior_to_use,
    loglike,
    n_live=1000,
    n_networks=4,
    n_batch=256,
    pool=8,
    pass_dict=False,
    filepath=outfile_chain_path + ".hdf5",
    resume=True,
    likelihood_kwargs={"leg_flag": "wedges", "prior": prior_to_use},
)
sampler.run(verbose=True, discard_exploration=True)
log_z_all = sampler.evidence()
print("Evidence:", log_z_all)
print("Sampling posterior")
points_all, log_w_all, log_l_all = sampler.posterior()


# In[82]:


sample_wghlkl = np.vstack((points_all.T, np.exp(log_w_all), log_l_all)).T


# In[83]:


print(f"Saving chain to text file {outfile_chain_path}.txt")
np.savetxt(outfile_chain_path + ".txt", sample_wghlkl)
