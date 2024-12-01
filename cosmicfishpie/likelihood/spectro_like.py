import logging
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import make_interp_spline
from scipy.special import eval_legendre

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey import spectro_cov as spcov
from cosmicfishpie.LSSsurvey import spectro_obs as spobs
from cosmicfishpie.utilities import legendre_tools as lgt
from cosmicfishpie.utilities.utils import printing as upr

logger = logging.getLogger("cosmicfishpie.cosmology.nuisance")
logger.setLevel(logging.INFO)
upr.debug = False


def is_indexable_iterable(var):
    return isinstance(var, (list, np.ndarray, Sequence)) and not isinstance(var, (str, bytes))


def observable_Pgg(theory_spectro, cosmoFM: cosmicfish.FisherMatrix, nuisance_shot=None):
    """
    cosmoFM: cosmicfish.FisherMatrix object
             must have been computed fully before calling this function
    """
    z_bins = cosmoFM.pk_cov.global_z_bin_mids
    n_bins = len(z_bins)
    si, sj = cosmoFM.obs_spectrum
    if nuisance_shot is None:
        nuisance_shot = np.zeros_like(z_bins)
    obs_Pgg = np.zeros((n_bins, cosmoFM.Pk_musamp, cosmoFM.Pk_ksamp))
    for ii in range(n_bins):
        obs_Pgg[ii, :, :] = (
            theory_spectro.noisy_P_ij(z_bins[ii], cosmoFM.Pk_kmesh, cosmoFM.Pk_mumesh, si=si, sj=sj)
            + nuisance_shot[ii]
        )
    return obs_Pgg


multipole_order = 4
leg_ells = np.arange(0, multipole_order + 1, 2, dtype="int")


def legendre_Pgg(obs_Pgg, cosmoFM: cosmicfish.FisherMatrix):
    legendre_vals = eval_legendre(leg_ells[None, :], cosmoFM.Pk_mugrid[:, None])
    legendre_vals = legendre_vals[None, :, None, :]
    obsPgg_forleg = obs_Pgg[:, :, :, None]
    P_ell = ((2.0 * leg_ells[None, None, :] + 1)) * simpson(
        legendre_vals * obsPgg_forleg, x=cosmoFM.Pk_mugrid, axis=1
    )
    P_ell = P_ell.transpose(1, 0, 2)
    return P_ell


def compute_covariance_legendre(P_ell, cosmoFM: cosmicfish.FisherMatrix):
    """
    Compute covariance matrix for power spectrum multipoles
    P_ell shape: (n_k, n_z, n_ell)
    """
    k_grid = cosmoFM.Pk_kgrid
    P_ell_broad = P_ell[:, :, :, None, None]
    n_k, n_z, n_ell = P_ell.shape
    volume_survey = cosmoFM.pk_cov.volume_survey_array

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


def compute_chi2_legendre(P_ell_data, P_ell_theory, inv_covariance):
    """
    Compute χ² using broadcasting

    P_ell_data: shape (n_k, n_z, n_ell)
    P_ell_theory: shape (n_k, n_z, n_ell)
    inv_covariance: shape (n_k, n_z, n_ell, n_ell)
    """

    chi2 = np.sum(
        (P_ell_data[..., None] - P_ell_theory[..., None])
        * inv_covariance
        * (P_ell_data[..., None, :] - P_ell_theory[..., None, :])
    )

    return chi2


def compute_wedge_chi2(P_obs_data, P_obs_theory, cosmoFM_data: cosmicfish.FisherMatrix):
    """
    Compute χ² for wedges using fully vectorized operations.
    Matches the loop implementation exactly.

    Parameters:
    ----------
    P_obs_data : array_like
        Synthetic data power spectrum (n_z, n_mu, n_k)
    P_obs_theory : array_like
        Theory power spectrum (n_z, n_mu, n_k)
    Returns:
    -------
    float
        χ² value
    """

    k_grid = cosmoFM_data.Pk_kgrid
    mu_grid = cosmoFM_data.Pk_mugrid
    prefactor = 8 * np.pi**2

    # Compute delta (n_z, n_mu, n_k)
    delta = P_obs_data - P_obs_theory

    # Prepare terms for broadcasting:
    k_term = k_grid[None, None, :] ** 2  # (1, 1, n_k)
    V_term = cosmoFM_data.pk_cov.volume_survey_array[:, None, None]  # (n_z, 1, 1)

    # Compute covariance (n_z, n_mu, n_k)
    covariance = (prefactor / (k_term * V_term)) * P_obs_data**2

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


def compute_theory_spectro(param_dict, cosmoFM_theory: cosmicfish.FisherMatrix, leg_flag="wedges"):

    z_bins = cosmoFM_theory.pk_cov.global_z_bin_mids

    nuisance_shot = np.zeros(len(z_bins))
    pshotpars = deepcopy(cosmoFM_theory.PShotpars)
    for ii, pp in enumerate(pshotpars.keys()):
        nuisance_shot[ii] = param_dict.pop(pp, cosmoFM_theory.PShotpars[pp])

    spectrobiaspars = deepcopy(cosmoFM_theory.Spectrobiaspars)
    for ii, pp in enumerate(spectrobiaspars.keys()):
        spectrobiaspars[pp] = param_dict.pop(pp, cosmoFM_theory.Spectrobiaspars[pp])

    spectrononlinearpars = deepcopy(cosmoFM_theory.Spectrononlinpars)
    for ii, pp in enumerate(spectrononlinearpars.keys()):
        spectrononlinearpars[pp] = param_dict.pop(pp, cosmoFM_theory.Spectrononlinpars[pp])

    IMbiaspars = deepcopy(cosmoFM_theory.IMbiasparams)
    for pp in IMbiaspars.keys():
        IMbiaspars[pp] = param_dict.pop(pp, cosmoFM_theory.IMbiasparams[pp])

    spectro_vary = spobs.ComputeGalSpectro(
        param_dict,
        spectrobiaspars=spectrobiaspars,
        spectrononlinearpars=spectrononlinearpars,
        PShotpars=cosmoFM_theory.PShotpars,
        fiducial_cosmo=cosmoFM_theory.fiducialcosmo,
        IMbiaspars=IMbiaspars,
        use_bias_funcs=False,
        configuration=cosmoFM_theory,
    )
    spectro_cov_vary = spcov.SpectroCov(
        fiducialpars=param_dict, configuration=cosmoFM_theory, fiducial_specobs=spectro_vary
    )
    obsPgg_vary = observable_Pgg(spectro_cov_vary, cosmoFM_theory, nuisance_shot=nuisance_shot)
    if leg_flag == "wedges":
        return obsPgg_vary
    elif leg_flag == "legendre":
        P_ell_vary = legendre_Pgg(obsPgg_vary, cosmoFM_theory)
        return P_ell_vary


def loglike(
    param_vec=None,
    theory_obsPgg=None,
    prior=None,
    leg_flag="wedges",
    cosmoFM_theory: cosmicfish.FisherMatrix = None,
    cosmoFM_data: cosmicfish.FisherMatrix = None,
    data_obsPgg: np.ndarray = None,
):
    if theory_obsPgg is None and param_vec is not None:
        if isinstance(param_vec, dict):
            param_dict = deepcopy(param_vec)
        elif is_indexable_iterable(param_vec) and prior is not None:
            # print(f'Loading prior with keys: {prior.keys}')
            param_dict = {key: param_vec[i] for i, key in enumerate(prior.keys)}
        theory_obsPgg = compute_theory_spectro(param_dict, cosmoFM_theory, leg_flag)
    else:
        upr.debug_print("No theory_obsPgg provided and no param_vec provided")
        return -np.inf
    if leg_flag == "wedges":
        chi2 = compute_wedge_chi2(
            P_obs_data=data_obsPgg, P_obs_theory=theory_obsPgg, cosmoFM_data=cosmoFM_data
        )
    elif leg_flag == "legendre":
        P_ell_data = legendre_Pgg(data_obsPgg, cosmoFM_data)
        covariance_leg, inv_covariance_leg = compute_covariance_legendre(
            P_ell=P_ell_data, cosmoFM=cosmoFM_data
        )
        chi2 = compute_chi2_legendre(P_ell_data, theory_obsPgg, inv_covariance_leg)
    return -0.5 * chi2
