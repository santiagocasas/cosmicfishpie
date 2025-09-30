from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import make_interp_spline
from scipy.special import eval_legendre

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey import spectro_cov as spcov
from cosmicfishpie.LSSsurvey import spectro_obs as spobs
from cosmicfishpie.utilities import legendre_tools as lgt
from cosmicfishpie.utilities.utils import printing as upr

from .base import Likelihood, NautilusMixin

logger = logging.getLogger("cosmicfishpie.cosmology.nuisance")
logger.setLevel(logging.INFO)
upr.debug = False


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


def _dict_with_updates(template: Dict[str, Any], pool: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deepcopy of ``template`` with any matching keys updated from ``pool``."""

    updated = deepcopy(template)
    for key in template:
        if key in pool:
            updated[key] = pool.pop(key)
    return updated


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


def compute_theory_spectro(
    param_dict: Dict[str, Any],
    cosmoFM_theory: cosmicfish.FisherMatrix,
    leg_flag: str = "wedges",
) -> np.ndarray:

    params = deepcopy(param_dict)
    z_bins = cosmoFM_theory.pk_cov.global_z_bin_mids

    nuisance_shot = np.zeros(len(z_bins))
    for index, key in enumerate(cosmoFM_theory.PShotpars.keys()):
        nuisance_shot[index] = params.pop(key, cosmoFM_theory.PShotpars[key])

    spectrobiaspars = _dict_with_updates(cosmoFM_theory.Spectrobiaspars, params)
    spectrononlinearpars = _dict_with_updates(cosmoFM_theory.Spectrononlinpars, params)
    IMbiaspars = _dict_with_updates(cosmoFM_theory.IMbiasparams, params)
    cosmological = _dict_with_updates(cosmoFM_theory.fiducialcosmopars, params)

    if params:
        logger.debug(
            "SpectroLikelihood received unused parameters: %s", ", ".join(sorted(params.keys()))
        )

    spectro_vary = spobs.ComputeGalSpectro(
        cosmological,
        spectrobiaspars=spectrobiaspars,
        spectrononlinearpars=spectrononlinearpars,
        PShotpars=cosmoFM_theory.PShotpars,
        fiducial_cosmo=cosmoFM_theory.fiducialcosmo,
        IMbiaspars=IMbiaspars,
        use_bias_funcs=False,
        configuration=cosmoFM_theory,
    )
    spectro_cov_vary = spcov.SpectroCov(
        fiducialpars=cosmological, configuration=cosmoFM_theory, fiducial_specobs=spectro_vary
    )
    obsPgg_vary = observable_Pgg(spectro_cov_vary, cosmoFM_theory, nuisance_shot=nuisance_shot)
    if leg_flag == "wedges":
        return obsPgg_vary
    if leg_flag == "legendre":
        return legendre_Pgg(obsPgg_vary, cosmoFM_theory)
    raise ValueError(f"Unknown leg_flag '{leg_flag}'. Use 'wedges' or 'legendre'.")


class SpectroLikelihood(Likelihood):
    """Likelihood for spectroscopic clustering using CosmicFish Fisher matrices."""

    def __init__(
        self,
        *,
        cosmoFM_data: cosmicfish.FisherMatrix,
        cosmoFM_theory: Optional[cosmicfish.FisherMatrix] = None,
        leg_flag: str = "wedges",
        data_obs: Optional[np.ndarray] = None,
        nuisance_shot: Optional[Iterable[float]] = None,
    ) -> None:
        self._preloaded_data = None if data_obs is None else np.array(data_obs)
        self._nuisance_shot = (
            None if nuisance_shot is None else np.array(nuisance_shot, dtype=float)
        )
        self._inv_cov_legendre = None
        self._data_wedges = None
        super().__init__(cosmo_data=cosmoFM_data, cosmo_theory=cosmoFM_theory, leg_flag=leg_flag)

    @property
    def data_wedges(self) -> Optional[np.ndarray]:
        """Return wedge data even when the likelihood operates in Legendre space."""

        return self._data_wedges

    def compute_data(self) -> np.ndarray:
        if self._preloaded_data is not None:
            data = np.array(self._preloaded_data, copy=False)
            if self.leg_flag == "legendre":
                _, self._inv_cov_legendre = compute_covariance_legendre(data, self.cosmoFM_data)
            else:
                self._data_wedges = data
            return data

        if not hasattr(self.cosmoFM_data, "pk_cov") or self.cosmoFM_data.pk_cov is None:
            raise AttributeError(
                "cosmoFM_data.pk_cov is not available. Ensure the FisherMatrix was initialised for spectroscopic probes."
            )

        obsPgg = observable_Pgg(
            self.cosmoFM_data.pk_cov,
            self.cosmoFM_data,
            nuisance_shot=self._nuisance_shot,
        )
        self._data_wedges = obsPgg
        if self.leg_flag == "legendre":
            p_ell = legendre_Pgg(obsPgg, self.cosmoFM_data)
            _, self._inv_cov_legendre = compute_covariance_legendre(p_ell, self.cosmoFM_data)
            return p_ell
        return obsPgg

    def compute_theory(self, param_dict: Dict[str, Any]) -> np.ndarray:
        return compute_theory_spectro(param_dict, self.cosmoFM_theory, self.leg_flag)

    def compute_chi2(self, theory_obs: np.ndarray) -> float:
        if self.leg_flag == "wedges":
            return compute_wedge_chi2(self.data_obs, theory_obs, self.cosmoFM_data)

        if self._inv_cov_legendre is None:
            _, self._inv_cov_legendre = compute_covariance_legendre(
                self.data_obs, self.cosmoFM_data
            )
        return compute_chi2_legendre(self.data_obs, theory_obs, self._inv_cov_legendre)


def loglike(
    param_vec=None,
    theory_obsPgg=None,
    prior=None,
    leg_flag="wedges",
    cosmoFM_theory: cosmicfish.FisherMatrix = None,
    cosmoFM_data: cosmicfish.FisherMatrix = None,
    data_obsPgg: np.ndarray = None,
):
    if cosmoFM_data is None:
        return -np.inf

    likelihood = SpectroLikelihood(
        cosmoFM_data=cosmoFM_data,
        cosmoFM_theory=cosmoFM_theory,
        leg_flag=leg_flag,
        data_obs=data_obsPgg,
    )

    if theory_obsPgg is not None:
        return -0.5 * likelihood.compute_chi2(theory_obsPgg)

    try:
        if isinstance(param_vec, dict):
            params = dict(param_vec)
        else:
            params = likelihood.build_param_dict(param_vec=param_vec, prior=prior)
    except (TypeError, ValueError, AttributeError):
        return -np.inf

    return likelihood.loglike(param_dict=params)
