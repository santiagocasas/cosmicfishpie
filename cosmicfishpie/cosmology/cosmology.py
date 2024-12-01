# -*- coding: utf-8 -*-
"""
COSMOLOGY.

This module contains useful cosmological functions.

"""

import os
import sys
import types
from copy import deepcopy
from glob import glob
from time import time
from warnings import warn

import numpy as np
import scipy.constants as sconst
from joblib import Memory
from scipy import integrate
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
    UnivariateSpline,
)
from scipy.signal import savgol_filter

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upr

cachedir = "memory_cache"
memory = Memory(cachedir, verbose=0)


def _dcom_func_trapz(zi, interpolfunc):
    """
    Calculate the comoving distance using the trapezoidal rule.

    Parameters
    ----------
    zi : float
        Redshift value
    interpolfunc : callable
        Interpolation function for H(z)

    Returns
    -------
    float
        Comoving distance
    """
    zt = np.linspace(0.0, zi, 100)
    Hzt = interpolfunc(zt)
    dcom = integrate.trapezoid(1 / Hzt, zt)
    return dcom


@memory.cache
def memorize_external_input(cosmopars, fiducialcosmopars, external, extra_settings):
    """
    Memorize external input for caching purposes.

    Parameters
    ----------
    cosmopars : dict
        Cosmological parameters
    fiducialcosmopars : dict
        Fiducial cosmological parameters
    external : dict
        External input settings
    extra_settings : dict
        Additional settings

    Returns
    -------
    external_input
        Memorized external input
    """
    externalinput = external_input(
        cosmopars,
        fiducialcosmopars=fiducialcosmopars,
        external=external,
        extra_settings=extra_settings,
    )
    return externalinput


class boltzmann_code:
    hardcoded_Neff = 3.044
    hardcoded_neutrino_mass_fac = 94.07
    hardcoded_mnu_massive_min = 0.001

    def __init__(self, cosmopars, code="camb"):
        """
        Initialize the boltzmann_code class.

        Parameters
        ----------
        cosmopars : dict
            The cosmological parameters object to be copied.
        code : str, optional
            The Boltzmann code to be used (default is 'camb').

        Raises
        ------
        ValueError
            If an unsupported Boltzmann code is specified.
        """
        self.cosmopars = deepcopy(cosmopars)
        self.feed_lvl = cfg.settings["feedback"]
        self.settings = cfg.settings
        self.set_cosmicfish_defaults()
        if code == "camb":
            camb_path = os.path.realpath(os.path.join(os.getcwd(), self.settings["camb_path"]))
            sys.path.insert(0, camb_path)
            import camb as camb

            self.boltzmann_cambpars = cfg.boltzmann_cambpars
            self.camb_setparams(self.cosmopars, camb)
            self.camb_results(camb)
        elif code == "class":
            self.boltzmann_classpars = cfg.boltzmann_classpars
            from classy import Class

            self.class_setparams(self.cosmopars)
            self.class_results(Class)
        elif code == "symbolic":
            try:
                import colossus.cosmology as colmo
                import symbolic_pofk.linear as symblin
                import symbolic_pofk.syrenhalofit as symbfit

                self.colmo = colmo
                self.symblin = symblin
                self.symbfit = symbfit
            except ImportError:
                print("Module symbolic_pofk not properly installed. Aborting")
                sys.exit()
            self.boltzmann_symbolicpars = cfg.boltzmann_symbolicpars
            self.halofit_version = self.boltzmann_symbolicpars["COSMO_SETTINGS"][
                "halofit_version"
            ]  # 'syren' or 'halofit+' or 'takahashi'
            if self.halofit_version == "takahashi":
                self.which_params = "Takahashi"
                self.add_correction = False
            elif self.halofit_version == "halofit+":
                self.which_params = "Bartlett"
                self.add_correction = False
            elif self.halofit_version == "syren":
                self.which_params = "Bartlett"
                self.add_correction = True
            self.extrapolate = self.boltzmann_symbolicpars["NUMERICS"]["extrapolate"]
            self.emulator_precision = self.boltzmann_symbolicpars["ACCURACY"][
                "emulator_precision"
            ]  # 'max_precision' or 'fiducial'
            self.symbolic_setparams()
            self.symbolic_results()
        else:
            print("other Boltzmann code not implemented yet")
            exit()

    def set_cosmicfish_defaults(self):
        """
        Fill up default values in the cosmopars dictionary if the values are not found.

        This method sets default values for various cosmological parameters if they
        are not already present in the cosmopars dictionary.
        """

        # Set default value for Omegam if neither Omegam or omch2 (camb) or omega_cdm (class) are passed
        if not any(par in self.cosmopars for par in ["Omegam", "omch2", "omega_cdm", "Omega_cdm"]):
            self.cosmopars["Omegam"] = 0.32

        # Set default value for Omegab if neither Omegab or ombh2 or omega_b or Omega_b or 100omega_b are passed
        if not any(
            par in self.cosmopars for par in ["Omegab", "ombh2", "omega_b", "Omega_b", "100omega_b"]
        ):
            self.cosmopars["Omegab"] = 0.05

        # Set default value for h if neither H0 or h are passed
        if not any(par in self.cosmopars for par in ["H0", "h"]):
            self.cosmopars["h"] = 0.67

        # Set default value for ns if it is not found in cosmopars
        if not any(par in self.cosmopars for par in ["ns", "n_s"]):
            self.cosmopars["ns"] = self.cosmopars.get("ns", 0.96)

        # Set default value for sigma8 if neither sigma8 or As or logAs or 10^9As are passed
        if not any(
            par in self.cosmopars for par in ["sigma8", "As", "logAs", "10^9As", "ln_A_s_1e10"]
        ):
            self.cosmopars["sigma8"] = 0.815583

        # Set default values for w0 and wa if cosmo_model is 'w0waCDM'
        if self.settings["cosmo_model"] == "w0waCDM":
            if not any(par in self.cosmopars for par in ["w", "w0_fld"]):
                self.cosmopars["w0"] = self.cosmopars.get("w0", -1.0)
            if not any(par in self.cosmopars for par in ["wa", "wa_fld"]):
                self.cosmopars["wa"] = self.cosmopars.get("wa", 0.0)

        # Set default value for mnu if Omeganu or omnuh2 or mnu is not found in cosmopars
        if not any(par in self.cosmopars for par in ["Omeganu", "omnuh2", "mnu"]):
            self.cosmopars["mnu"] = self.cosmopars.get(
                "mnu", self.cosmopars.get("m_nu", self.cosmopars.get("M_nu", 0.06))
            )

        # Set default value for Neff if it is not found in cosmopars
        # self.cosmopars["Neff"] = self.cosmopars.get("Neff", self.cosmopars.get("N_eff", 3.046))

        # Set default value for gamma, if it is not found in cosmopars
        # gamma is not used in many places, therefore not needed to add back in cosmopars
        self.gamma = self.cosmopars.get("gamma", 0.545)

    @staticmethod
    def print_cosmo_params(cosmopars, feedback=1, text="---Cosmo pars---"):
        """
        Print cosmological parameters.

        Parameters
        ----------
        cosmopars : dict
            Dictionary of cosmological parameters to print.
        feedback : int, optional
            Feedback level determining whether to print (default is 1).
        text : str, optional
            Header text for the parameter list (default is "---Cosmo pars---").
        """
        if feedback > 2:
            print("")
            print(text)
            for key in cosmopars:
                print(key + "=" + str(cosmopars[key]))

    @staticmethod
    def f_deriv(D_growth_zk, z_array, k_array, k_fix=False, fixed_k=1e-3):
        """
        Calculate the growth rate f(z,k).

        Parameters
        ----------
        D_growth_zk : callable
            Function that returns the growth factor D(z,k).
        z_array : array_like
            Array of redshift values.
        k_array : array_like
            Array of wavenumbers.
        k_fix : bool, optional
            If True, use a fixed k value (default is False).
        fixed_k : float, optional
            Fixed k value to use if k_fix is True (default is 1e-3).

        Returns
        -------
        tuple
            Growth rate f(z,k) and the corresponding z array.
        """
        if k_fix:
            k_array = np.full((len(k_array)), fixed_k)
        ## Generates interpolators D(z) for varying k values
        D_z = np.array([UnivariateSpline(z_array, D_growth_zk(z_array, kk), s=0) for kk in k_array])
        ## Generates arrays f(z) for varying k values
        f_z = np.array(
            [-(1 + z_array) / D_zk(z_array) * (D_zk.derivative())(z_array) for D_zk in D_z]
        )
        return f_z, z_array

    @staticmethod
    def compute_sigma8(z_range, pk_interpolator, h_value, k_range):
        """
        Calculate sigma8 over a range of redshifts using a given power spectrum interpolator.

        This function computes sigma8, which is the RMS matter fluctuation in spheres of 8 h^-1 Mpc radius,
        for a range of redshifts. It uses a provided power spectrum interpolator to perform the calculation.

        Parameters
        ----------
        z_range : numpy.ndarray
            Array of redshift values at which to calculate sigma8.
        pk_interpolator : callable
            A function that takes (z, k) as arguments and returns
            the power spectrum P(k,z). It should be able to handle
            array inputs for both z and k.

        Returns
        -------
        scipy.interpolate.UnivariateSpline
            A 1-D interpolation function sigma8(z) that can be
            used to obtain sigma8 values for any redshift within
            the input range.
        """
        R = 8.0 / (h_value)
        kmin = np.min(k_range)
        kmax = np.max(k_range)
        # resampling the k range to 10000 points
        k = np.linspace(kmin, kmax, 10000)
        sigma_z = np.empty_like(z_range)
        pk_z = pk_interpolator(z_range, k)

        for i in range(len(sigma_z)):
            integrand = (
                9
                * (k * R * np.cos(k * R) - np.sin(k * R)) ** 2
                * pk_z[i]
                / k**4
                / R**6
                / 2
                / np.pi**2
            )
            sigma_z[i] = np.sqrt(integrate.trapezoid(integrand, k))

        sigma8_z_interp = UnivariateSpline(z_range, sigma_z, s=0)
        return sigma8_z_interp

    def camb_setparams(self, cosmopars, camb):
        """
        Set the parameters for CAMB computation.

        Parameters
        ----------
        cosmopars : dict
            Dictionary containing the cosmological parameters.
        camb : module
            The CAMB module.

        Notes
        -----
        This method sets up CAMB parameters and prepares for power spectrum computation.
        """
        # Adding hard coded CAMB options
        self.cosmopars = deepcopy(cosmopars)
        self.gamma = self.cosmopars.get(
            "gamma", 0.545
        )  # default gamma value 0.545, set if gamma not in self.cosmopars
        ## gamma not used in many places, therefore not needed to add back in self.cosmopars
        tini_basis = time()
        self.cambcosmopars = {
            **self.boltzmann_cambpars["ACCURACY"],
            **self.boltzmann_cambpars["COSMO_SETTINGS"],
            **self.boltzmann_cambpars[self.settings["cosmo_model"]],
        }.copy()
        self.kmax_pk = self.cambcosmopars["kmax"]
        self.kmin_pk = 1e-4
        self.zmax_pk = self.boltzmann_cambpars["NUMERICS"]["zmax"]
        self.z_samples = self.boltzmann_cambpars["NUMERICS"]["zsamples"]
        upr.debug_print(self.cambcosmopars)
        self.cambcosmopars.update(self.cosmopars)
        self.cambcosmopars = self.changebasis_camb(self.cambcosmopars, camb)
        upr.debug_print(self.cambcosmopars)
        tend_basis = time()
        if self.feed_lvl > 2:
            print("Basis change took {:.2f} s".format(tend_basis - tini_basis))
        self.print_cosmo_params(
            self.cambcosmopars, feedback=self.feed_lvl, text="---CAMB parameters---"
        )
        self.cambclasspars = camb.set_params(**self.cambcosmopars)

        self.camb_zarray = np.linspace(0.0, self.zmax_pk, self.z_samples)[::-1]
        self.cambclasspars.set_matter_power(
            redshifts=self.camb_zarray,
            k_per_logint=self.cambcosmopars["k_per_logint"],
            kmax=self.cambcosmopars["kmax"],
            accurate_massive_neutrino_transfers=self.cambcosmopars[
                "accurate_massive_neutrino_transfers"
            ],
        )
        # TODO: nonlinear options to be selectable
        self.cambclasspars.NonLinear = camb.model.NonLinear_both
        self.cambclasspars.set_for_lmax(4000, lens_potential_accuracy=1)

    def changebasis_camb(self, cosmopars, camb):
        """
        Convert cosmological parameters to CAMB format.

        Parameters
        ----------
        cosmopars : dict
            Dictionary of cosmological parameters.
        camb : module
            The CAMB module.

        Returns
        -------
        dict
            Dictionary of CAMB-formatted cosmological parameters.
        """
        cambpars = deepcopy(cosmopars)

        if "h" in cambpars:
            cambpars["H0"] = cambpars.pop("h") * 100
        if "H0" in cambpars:
            self.h_now = cambpars["H0"] / 100
        if "Omegab" in cambpars:
            cambpars["ombh2"] = cambpars.pop("Omegab") * (cambpars["H0"] / 100) ** 2
        if "Omegak" in cambpars:
            cambpars["omk"] = cambpars.pop("Omegak")
        if "w0" in cambpars:
            cambpars["w"] = cambpars.pop("w0")
        if "logAs" in cambpars:
            cambpars["As"] = np.exp(cambpars.pop("logAs")) * 1.0e-10

        upr.debug_print("DEBUG:  --> ", cosmopars)
        shareDeltaNeff = cfg.settings["ShareDeltaNeff"]
        cambpars["share_delta_neff"] = shareDeltaNeff
        fidNeff = boltzmann_code.hardcoded_Neff
        minmassmnu = boltzmann_code.hardcoded_mnu_massive_min
        if "mnu" in cambpars:
            if cambpars["mnu"] < minmassmnu and cambpars["num_nu_massive"] > 0:
                raise ValueError(
                    f"mnu is less than {minmassmnu} and "
                    f"num_nu_massive is greater than 0. "
                    f"Check your yaml file."
                )
        if "Neff" in cambpars:
            Neff = cambpars.pop("Neff")
            if shareDeltaNeff:
                cambpars["num_nu_massless"] = Neff - cambpars["num_nu_massive"]
            else:
                cambpars["num_nu_massless"] = Neff - fidNeff / 3

        else:
            Neff = cambpars["num_nu_massive"] + cambpars["num_nu_massless"]

        if shareDeltaNeff:
            g_factor = Neff / 3
        else:
            g_factor = fidNeff / 3

        neutrino_mass_fac = boltzmann_code.hardcoded_neutrino_mass_fac
        h2 = self.h_now**2

        if "mnu" in cambpars:
            Onu = cambpars["mnu"] / neutrino_mass_fac * (g_factor) ** 0.75 / h2
            onuh2 = Onu * h2
            cambpars["omnuh2"] = onuh2
        elif "Omeganu" in cambpars:
            cambpars["omnuh2"] = cambpars.pop("Omeganu") * h2
            onuh2 = cambpars["omnuh2"]
        elif "omnuh2" in cambpars:
            onuh2 = cambpars["omnuh2"]

        if "Omegam" in cambpars:  # TO BE GENERALIZED
            cambpars["omch2"] = cambpars.pop("Omegam") * h2 - cambpars["ombh2"] - onuh2

        rescaleAs = False
        if "sigma8" in cambpars:
            insigma8 = cambpars["sigma8"]
            cambpars["As"] = self.settings.get("rescale_ini_As", 2.1e-9)
            cambpars.pop("sigma8")
            rescaleAs = True

        try:
            camb.set_params(**cambpars)  # to see which methods are being called: verbose=True
        except camb.CAMBUnknownArgumentError as argument:
            upr.debug_print("Remove parameter from cambparams: ", str(argument))

            # pars= camb.set_params(redshifts=[0.], kmax=50.0,accurate_massive_neutrino_transfers=True,lmax=1000, lens_potential_accuracy=1,**cambpars)

        self.extrap_kmax = cambpars.pop("extrap_kmax", 100)

        if rescaleAs is True:
            cambpars["As"] = self.rescale_LP(cambpars, camb, insigma8)

        return cambpars

    def rescale_LP(self, cambpars, camb, insigma8):
        """
        Rescale As to match a target sigma8 value.

        Parameters
        ----------
        cambpars : dict
            CAMB parameters.
        camb : module
            The CAMB module.
        insigma8 : float
            Target sigma8 value.

        Returns
        -------
        float
            Rescaled As value.
        """
        cambpars_LP = cambpars.copy()
        ini_As = self.settings.get("rescale_ini_As", 2.1e-9)
        boost = self.settings.get("rescale_boost", 1)
        cambpars_LP["AccuracyBoost"] = boost
        cambpars_LP["lAccuracyBoost"] = boost
        cambpars_LP["lSampleBoost"] = boost
        cambpars_LP["kmax"] = 20
        pars = camb.set_params(redshifts=[0.0], **cambpars_LP)
        results = camb.get_results(pars)
        test_sig8 = np.array(results.get_sigma8())
        final_As = ini_As * (insigma8 / test_sig8[-1]) ** 2.0
        get_rescaled_s8 = self.settings.get("get_rescaled_s8", False)
        if get_rescaled_s8:
            cambpars_rs = cambpars_LP.copy()
            cambpars_rs["As"] = final_As
            pars2 = camb.set_params(redshifts=[0.0], **cambpars_rs)
            results2 = camb.get_results(pars2)
            final_sig8 = np.array(results2.get_sigma8())[-1]
        if self.feed_lvl > 2:
            print("AccuracyBoost input = ", cambpars["AccuracyBoost"])
            print("AccuracyBoost rescaling = ", cambpars_LP["lAccuracyBoost"])
            print("Goal sig8 = ", insigma8)
            print("Reference As = ", ini_As)
            print("Reference sig8 = ", test_sig8)
            print("Rescaled As  = ", final_As)
            if get_rescaled_s8:
                print("Rescaled sig8 = ", final_sig8)
        return final_As

    def camb_results(self, camb):
        """
        Compute and store CAMB results.

        Parameters
        ----------
        camb : module
            The CAMB module.

        Notes
        -----
        This method computes various cosmological quantities using CAMB and stores
        them in the results attribute.
        """
        tini_camb = time()
        self.results = types.SimpleNamespace()
        cambres = camb.get_results(self.cambclasspars)
        if self.feed_lvl > 2:
            tres = time()
            print("Time for Results = ", tres - tini_camb)
        Pk_l, self.results.zgrid, self.results.kgrid = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_tot",
            var2="delta_tot",
            nonlinear=False,
            extrap_kmax=self.extrap_kmax,
            return_z_k=True,
        )
        Pk_nl, zgrid, kgrid = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_tot",
            var2="delta_tot",
            nonlinear=True,
            extrap_kmax=self.extrap_kmax,
            return_z_k=True,
        )
        Pk_cb_l, zgrid, kgrid = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_nonu",
            var2="delta_nonu",
            nonlinear=False,
            extrap_kmax=self.extrap_kmax,
            return_z_k=True,
        )

        self.results.Pk_l = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, Pk_l.P(self.results.zgrid, self.results.kgrid)
        )
        self.results.Pk_nl = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, Pk_nl.P(self.results.zgrid, self.results.kgrid)
        )
        self.results.Pk_cb_l = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid,
            Pk_cb_l.P(self.results.zgrid, self.results.kgrid),
        )
        self.results.h_of_z = InterpolatedUnivariateSpline(
            self.results.zgrid, cambres.h_of_z(self.results.zgrid)
        )
        self.results.ang_dist = InterpolatedUnivariateSpline(
            self.results.zgrid, cambres.angular_diameter_distance(self.results.zgrid)
        )
        self.results.com_dist = InterpolatedUnivariateSpline(
            self.results.zgrid, cambres.comoving_radial_distance(self.results.zgrid)
        )
        self.results.Om_m = InterpolatedUnivariateSpline(
            self.results.zgrid,
            (
                cambres.get_Omega("cdm", z=self.results.zgrid)
                + cambres.get_Omega("baryon", z=self.results.zgrid)
                + cambres.get_Omega("nu", z=self.results.zgrid)
            ),
        )

        # Calculate the Non linear cb power spectrum using Gabrieles Approximation
        f_cdm = cambres.get_Omega("cdm", z=0) / self.results.Om_m(0)
        f_b = cambres.get_Omega("baryon", z=0) / self.results.Om_m(0)
        f_cb = f_cdm + f_b
        f_nu = 1 - f_cb
        Pk_cross_l = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_nonu",
            var2="delta_nu",
            nonlinear=False,
            extrap_kmax=self.extrap_kmax,
            return_z_k=False,
        )
        Pk_nunu_l = cambres.get_matter_power_interpolator(
            hubble_units=False,
            k_hunit=False,
            var1="delta_nu",
            var2="delta_nu",
            nonlinear=False,
            extrap_kmax=self.extrap_kmax,
            return_z_k=False,
        )
        Pk_cb_nl = (
            1
            / f_cb**2
            * (
                Pk_nl.P(self.results.zgrid, self.results.kgrid)
                - 2 * Pk_cross_l.P(self.results.zgrid, self.results.kgrid) * f_cb * f_nu
                - Pk_nunu_l.P(self.results.zgrid, self.results.kgrid) * f_nu**2
            )
        )
        self.results.Pk_cb_nl = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, Pk_cb_nl
        )

        if self.feed_lvl > 2:
            tPk = time()
            print("Time for lin+nonlin Pk = ", tPk - tres)

        P_kz_0 = self.results.Pk_l(0.0, self.results.kgrid)
        D_g_norm_kz = np.sqrt(self.results.Pk_l(self.results.zgrid, self.results.kgrid) / P_kz_0)

        self.results.D_growth_zk = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, (D_g_norm_kz), kx=3, ky=3
        )

        P_cb_kz_0 = self.results.Pk_cb_l(0.0, self.results.kgrid)
        D_g_cb_norm_kz = np.sqrt(
            self.results.Pk_cb_l(self.results.zgrid, self.results.kgrid) / P_cb_kz_0
        )
        self.results.D_growth_cb_zk = RectBivariateSpline(
            self.results.zgrid, self.results.kgrid, (D_g_cb_norm_kz), kx=3, ky=3
        )

        if self.feed_lvl > 2:
            tDzk = time()
            print("Time for Growth factor = ", tDzk - tPk)

        f_z_k_array, z_array = self.f_deriv(
            self.results.D_growth_zk, self.results.zgrid, self.results.kgrid
        )
        self.results.f_growthrate_zk = RectBivariateSpline(
            z_array, self.results.kgrid, f_z_k_array.T
        )

        f_cb_z_k_array, z_array = self.f_deriv(
            self.results.D_growth_cb_zk, self.results.zgrid, self.results.kgrid
        )
        self.results.f_growthrate_cb_zk = RectBivariateSpline(
            z_array, self.results.kgrid, f_cb_z_k_array.T
        )

        if self.feed_lvl > 2:
            tfzk = time()
            print("Time for Growth factor = ", tfzk - tDzk)

        self.results.s8_cb_of_z = self.compute_sigma8(
            self.results.zgrid, self.results.Pk_cb_l, self.h_now, self.results.kgrid
        )
        self.results.s8_of_z = self.compute_sigma8(
            self.results.zgrid, self.results.Pk_l, self.h_now, self.results.kgrid
        )

        if self.feed_lvl > 2:
            ts8 = time()
            print("Time for Growth factor = ", ts8 - tfzk)

        if self.cambcosmopars["Want_CMB"]:
            powers = cambres.get_cmb_power_spectra(CMB_unit="muK")
            self.results.camb_cmb = powers["total"]
        tend_camb = time()
        if self.feed_lvl > 2:
            print("Time for CMB = ", tend_camb - ts8)
        if self.feed_lvl > 1:
            print("Cosmology computation took {:.2f} s".format(tend_camb - tini_camb))

    def class_setparams(self, cosmopars):
        """
        Set the parameters for CLASS computation.

        Parameters
        ----------
        cosmopars : dict
            Dictionary containing the cosmological parameters.

        Notes
        -----
        This method sets up CLASS parameters and prepares for power spectrum computation.
        """
        tini_basis = time()
        self.classcosmopars = {
            **self.boltzmann_classpars["ACCURACY"],
            **self.boltzmann_classpars["COSMO_SETTINGS"],
            **self.boltzmann_classpars[self.settings["cosmo_model"]],
        }.copy()
        upr.debug_print(self.classcosmopars)
        upr.debug_print(cosmopars)
        self.classcosmopars.update(self.changebasis_class(self.cosmopars))
        upr.debug_print(self.classcosmopars)
        self.kmax_pk = self.classcosmopars["P_k_max_1/Mpc"]
        self.kmin_pk = 1e-4
        self.zmax_pk = self.classcosmopars["z_max_pk"]
        tend_basis = time()
        if self.feed_lvl > 2:
            print("Basis change took {:.2f} s".format(tend_basis - tini_basis))
        self.print_cosmo_params(
            self.classcosmopars, feedback=self.feed_lvl, text="---CLASS parameters---"
        )

    def changebasis_class(self, cosmopars):
        """
        Convert cosmological parameters to CLASS format.

        Parameters
        ----------
        cosmopars : dict
            Dictionary of cosmological parameters.

        Returns
        -------
        dict
            Dictionary of CLASS-formatted cosmological parameters.
        """
        classpars = deepcopy(cosmopars)
        if "h" in classpars:
            classpars["h"] = classpars.pop("h")
            h = classpars["h"]
        if "H0" in classpars:
            classpars["H0"] = classpars.pop("H0")
            h = classpars["H0"] / 100.0

        shareDeltaNeff = cfg.settings["ShareDeltaNeff"]
        fidNeff = boltzmann_code.hardcoded_Neff
        Neff = classpars.pop("Neff", fidNeff)

        if shareDeltaNeff:
            classpars["N_ur"] = (
                2.0 / 3.0 * Neff
            )  # This version does not have the discontinuity at Nur = 1.99
            g_factor = Neff / 3.0
        else:
            classpars["N_ur"] = Neff - fidNeff / 3.0
            g_factor = fidNeff / 3.0

        neutrino_mass_fac = boltzmann_code.hardcoded_neutrino_mass_fac

        if "mnu" in classpars:
            classpars["T_ncdm"] = (4.0 / 11.0) ** (1.0 / 3.0) * g_factor ** (1.0 / 4.0)
            classpars["Omega_ncdm"] = (
                classpars["mnu"] * g_factor ** (0.75) / neutrino_mass_fac / h**2
            )
            classpars.pop("mnu")
            # classpars['m_ncdm'] = classpars.pop('mnu')
            # Om_ncdm = classpars['m_ncdm'] / 93.13858 /h/h
        elif "Omeganu" in classpars:
            classpars["Omega_ncdm"] = classpars.pop("Omeganu")

        if "100omega_b" in classpars:
            classpars["omega_b"] = (1 / 100) * classpars.pop("100omega_b")
        if "Omegab" in classpars:
            classpars["Omega_b"] = classpars.pop("Omegab")
        if "Omegam" in classpars:
            classpars["Omega_cdm"] = (
                classpars.pop("Omegam") - classpars["Omega_b"] - classpars["Omega_ncdm"]
            )

        if "w0" in classpars:
            classpars["w0_fld"] = classpars.pop("w0")
        if "wa" in classpars:
            classpars["wa_fld"] = classpars.pop("wa")
        if "logAs" in classpars:
            classpars["A_s"] = np.exp(classpars.pop("logAs")) * 1.0e-10
        if "10^9As" in classpars:
            classpars["A_s"] = classpars.pop("10^9As") * 1.0e-9
        if "ns" in classpars:
            classpars["n_s"] = classpars.pop("ns")

        return classpars

    def class_results(self, Class):
        """
        Compute and store CLASS results.

        Parameters
        ----------
        Class : class
            The CLASS class.

        Notes
        -----
        This method computes various cosmological quantities using CLASS and stores
        them in the results attribute.
        """
        self.results = types.SimpleNamespace()
        classres = Class()
        classres.set(self.classcosmopars)
        classres.compute()
        self.Classres = classres
        self.results.h_of_z = np.vectorize(classres.Hubble)
        self.results.ang_dist = np.vectorize(classres.angular_distance)
        self.results.com_dist = np.vectorize(classres.comoving_distance)
        h = classres.h()
        self.results.s8_of_z = np.vectorize(lambda zz: classres.sigma(R=8 / h, z=zz))
        self.results.s8_cb_of_z = np.vectorize(lambda zz: classres.sigma_cb(R=8 / h, z=zz))
        self.results.Om_m = np.vectorize(classres.Om_m)

        # Calculate the Matter fractions for CB Powerspectrum
        f_cdm = classres.Omega0_cdm() / classres.Omega_m()
        f_b = classres.Omega_b() / classres.Omega_m()
        f_cb = f_cdm + f_b
        f_nu = 1 - f_cb

        ## rows are k, and columns are z
        ## interpolating function Pk_l (k,z)
        Pk_l, k, z = classres.get_pk_and_k_and_z(nonlinear=False)
        Pk_cb_l, k, z = classres.get_pk_and_k_and_z(only_clustering_species=True, nonlinear=False)
        self.results.Pk_l = RectBivariateSpline(z[::-1], k, (np.flip(Pk_l, axis=1)).transpose())
        # self.results.Pk_l = lambda z,k: [np.array([classres.pk_lin(kval,z) for kval in k])]
        self.results.Pk_cb_l = RectBivariateSpline(
            z[::-1], k, (np.flip(Pk_cb_l, axis=1)).transpose()
        )
        # self.results.Pk_cb_l = lambda z,k: [np.array([classres.pk_cb_lin(kval,z) for kval in k])]

        self.results.kgrid = k
        self.results.zgrid = z[::-1]

        ## interpolating function Pk_nl (k,z)
        Pk_nl, k, z = classres.get_pk_and_k_and_z(nonlinear=cfg.settings["nonlinear"])
        self.results.Pk_nl = RectBivariateSpline(z[::-1], k, (np.flip(Pk_nl, axis=1)).transpose())

        tk, k, z = classres.get_transfer_and_k_and_z()
        T_cb = (f_b * tk["d_b"] + f_cdm * tk["d_cdm"]) / f_cb
        T_nu = tk["d_ncdm[0]"]

        pm = classres.get_primordial()
        pk_prim = (
            UnivariateSpline(pm["k [1/Mpc]"], pm["P_scalar(k)"])(k)
            * (2.0 * np.pi**2)
            / np.power(k, 3)
        )

        pk_cnu = T_nu * T_cb * pk_prim[:, None]
        pk_nunu = T_nu * T_nu * pk_prim[:, None]
        Pk_cb_nl = 1.0 / f_cb**2 * (Pk_nl - 2 * pk_cnu * f_nu * f_cb - pk_nunu * f_nu * f_nu)

        self.results.Pk_cb_nl = RectBivariateSpline(
            z[::-1], k, (np.flip(Pk_cb_nl, axis=1)).transpose()
        )

        def create_growth():
            z_ = self.results.zgrid
            pk_flipped = np.flip(Pk_l, axis=1).T
            D_growth_zk = RectBivariateSpline(z_, k, np.sqrt(pk_flipped / pk_flipped[0, :]))
            return D_growth_zk

        self.results.D_growth_zk = create_growth()

        f_z_k_array, z_array = self.f_deriv(
            self.results.D_growth_zk, self.results.zgrid, self.results.kgrid
        )
        f_g_kz = RectBivariateSpline(z_array, self.results.kgrid, f_z_k_array.T)
        self.results.f_growthrate_zk = f_g_kz

        def create_growth_cb():
            z_ = self.results.zgrid
            pk_flipped = np.flip(Pk_cb_l, axis=1).T
            D_growth_zk = RectBivariateSpline(z_, k, np.sqrt(pk_flipped / pk_flipped[0, :]))
            return D_growth_zk

        self.results.D_growth_cb_zk = create_growth_cb()

        def f_cb_deriv(k_array, k_fix=False, fixed_k=1e-3):
            z_array = np.linspace(0, classres.pars["z_max_pk"], 100)
            if k_fix:
                k_array = np.full((len(k_array)), fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_cb_z = np.array(
                [
                    UnivariateSpline(z_array, self.results.D_growth_cb_zk(z_array, kk), s=0)
                    for kk in k_array
                ]
            )

            ## Generates arrays f(z) for varying k values
            f_cb_z = np.array(
                [
                    -(1 + z_array) / D_cb_zk(z_array) * (D_cb_zk.derivative())(z_array)
                    for D_cb_zk in D_cb_z
                ]
            )
            return f_cb_z, z_array

        f_cb_z_k_array, z_array = f_cb_deriv(self.results.kgrid)
        f_g_cb_kz = RectBivariateSpline(z_array, self.results.kgrid, f_cb_z_k_array.T)
        self.results.f_growthrate_cb_zk = f_g_cb_kz

    def changebasis_symb(self, cosmopars):
        """
        Convert and adjust cosmological parameters for symbolic computation.

        This method takes the input cosmological parameters and converts them
        to the format required for symbolic computation. It handles unit
        conversions, derives some parameters, and ensures consistency between
        different parameterizations (e.g., sigma8 and As).

        Parameters
        ----------
        cosmopars : dict
            Dictionary containing the input cosmological parameters.

        Returns
        -------
        dict
            A new dictionary with the converted and adjusted parameters suitable
            for symbolic computation.

        Raises
        ------
        ValueError
            If there's an issue with the sigma8 to As conversion or vice versa.
        KeyError
            If required parameters are missing from the input.

        Notes
        -----
        - The method performs conversions between h and H0, ombh2 and Omegab,
          and handles various representations of the primordial power spectrum
          amplitude (As, sigma8).
        - Debug information is printed if the upr.debug flag is set.
        """
        upr.debug_print("DEBUG: cosmopars = ", cosmopars)
        symbpars = deepcopy(cosmopars)
        if "h" in symbpars:
            symbpars["h"] = symbpars.pop("h")
        if "H0" in symbpars:
            H0 = symbpars.pop("H0")
            symbpars["h"] = H0 / 100.0
        if "ombh2" in symbpars:
            symbpars["Omegab"] = symbpars.pop("ombh2") / (symbpars["h"] ** 2)
        if "omch2" in symbpars:
            symbpars["Omegam"] = (symbpars.pop("omch2") / symbpars["h"] ** 2) + symbpars["Omegab"]
            # Omegam = Omegac + Omegab
        # ["sigma8", "As", "logAs", "10^9As", "ln_A_s_1e10"]
        try:
            if "As" in symbpars:
                symbpars["10^9As"] = 10**9 * symbpars.pop("As")
            if "logAs" in symbpars:
                symbpars["10^9As"] = 10**9 * (np.exp(symbpars.pop("logAs")) * 1.0e-10)
            try:
                As_value = symbpars.get("10^9As")
                upr.debug_print("DEBUG: symbpars['10^9As'] = ", As_value)
                if np.isscalar(As_value):
                    try:
                        symbpars["sigma8"] = self.symblin.As_to_sigma8(
                            symbpars["10^9As"],
                            symbpars["Omegam"],
                            symbpars["Omegab"],
                            symbpars["h"],
                            symbpars["ns"],
                        )
                        upr.debug_print("DEBUG: symbpars['sigma8'] = ", symbpars["sigma8"])
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        print("As to sigma8 conversion failed")
                        raise ValueError
            except KeyError:
                pass
            try:
                sigma8_value = symbpars.get("sigma8")
                if np.isscalar(sigma8_value):
                    try:
                        As_n = self.symblin.sigma8_to_As(
                            symbpars["sigma8"],
                            symbpars["Omegam"],
                            symbpars["Omegab"],
                            symbpars["h"],
                            symbpars["ns"],
                        )
                        upr.debug_print("DEBUG: As_n = ", As_n)
                        symbpars["10^9As"] = As_n
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        print("sigma8 to As conversion failed")
                        raise ValueError
                else:
                    print("sigma8 value not scalar")
                    upr.debug_print("DEBUG: symbpars = ", symbpars)
            except KeyError:
                print("sigma8 or 10^9As not in symbpars")
                raise KeyError
        except Exception as e:
            print(f"An error occurred: {e}")
            print("sigma8<->As conversion failed for other reasons")
            upr.debug_print("DEBUG: symbpars = ", symbpars)
            raise
        return symbpars

    def symbolic_setparams(self):
        """
        Set up parameters for symbolic computation.

        Notes
        -----
        This method prepares the cosmological parameters and numerical settings
        for symbolic power spectrum computation.

        Raises
        ------
        ValueError
            If the cosmological model is not supported by the symbolic code.
        """
        tini_basis = time()
        if cfg.settings["cosmo_model"] != "LCDM":
            print("Symbolic_pofk only supports LCDM at the moment")
            raise ValueError("Cosmo model not supported by cosmo code")
        self.symbcosmopars = dict()
        self.symbcosmopars.update(self.changebasis_symb(self.cosmopars))
        self.kmax_pk = self.boltzmann_symbolicpars["NUMERICS"]["kmax_pk"]
        self.kmin_pk = self.boltzmann_symbolicpars["NUMERICS"]["kmin_pk"]
        self.zmax_pk = self.boltzmann_symbolicpars["NUMERICS"]["zmax_pk"]
        self.zmin_pk = self.boltzmann_symbolicpars["NUMERICS"]["zmin_pk"]
        self.z_samples = self.boltzmann_symbolicpars["NUMERICS"]["z_samples"]
        self.zgrid = np.linspace(self.zmin_pk, self.zmax_pk, self.z_samples)
        self.k_samples = self.boltzmann_symbolicpars["NUMERICS"]["k_samples"]
        self.kgrid_1Mpc = np.logspace(
            np.log10(self.kmin_pk), np.log10(self.kmax_pk), self.k_samples
        )
        tend_basis = time()
        if self.feed_lvl > 2:
            print("Basis change took {:.2f} s".format(tend_basis - tini_basis))
        self.print_cosmo_params(
            self.symbcosmopars, feedback=self.feed_lvl, text="--- Symbolic Cosmo parameters ---"
        )

    def symbolic_results(self):
        """
        Compute and store results using symbolic computation.

        Returns
        -------
        types.SimpleNamespace
            Namespace containing the computed cosmological quantities.

        Notes
        -----
        This method computes various cosmological quantities using symbolic
        computation and stores them in the results attribute.
        """
        tini_results = time()
        self.results = types.SimpleNamespace()
        symb_colmo_pars = {
            "flat": True,
            "sigma8": self.symbcosmopars["sigma8"],
            "Om0": self.symbcosmopars["Omegam"],
            "Ob0": self.symbcosmopars["Omegab"],
            "H0": self.symbcosmopars["h"] * 100,
            "ns": self.symbcosmopars["ns"],
        }
        self.results.zgrid = self.zgrid
        self.h_now = self.symbcosmopars["h"]
        self.kgrid_hMpc = self.kgrid_1Mpc / self.h_now
        self.results.kgrid = self.kgrid_1Mpc  # results kgrid is in units of 1/Mpc
        self.symbcosmo = self.colmo.cosmology.setCosmology("colmo", **symb_colmo_pars)
        self.results.h_of_z = np.vectorize(
            lambda zz: self.symbcosmo.Hz(zz) / cosmo_functions.c
        )  # H(z) in 1/Mpc
        self.results.ang_dist = np.vectorize(
            lambda zz: self.symbcosmo.angularDiameterDistance(zz) / self.h_now
        )
        self.results.com_dist = np.vectorize(
            lambda zz: self.symbcosmo.comovingDistance(z_min=0.0, z_max=zz, transverse=True)
            / self.h_now
        )
        self.results.s8_of_z = np.vectorize(lambda zz: self.symbcosmo.sigma(8, z=zz))
        self.results.Om_m = self.symbcosmo.Om
        D_kz = np.array([self.symbcosmo.growthFactor(self.zgrid) for kk in self.kgrid_1Mpc]).T
        self.results.D_growth_zk = RectBivariateSpline(self.zgrid, self.kgrid_1Mpc, D_kz)

        self.results.Pk_l_0 = ((1 / self.h_now) ** 3) * self.symblin.plin_emulated(
            self.kgrid_hMpc,
            self.symbcosmopars["sigma8"],
            self.symbcosmopars["Omegam"],
            self.symbcosmopars["Omegab"],
            self.h_now,
            self.symbcosmopars["ns"],
            emulator=self.emulator_precision,
            extrapolate=self.extrapolate,
        )
        # symbfit plin_emulated returns P_l(k,z=0) in 1/Mpc^3, requests kgrid in h/Mpc
        Pk_at_z = (D_kz**2) * self.results.Pk_l_0
        self.results.Pk_l = RectBivariateSpline(
            self.zgrid, self.kgrid_1Mpc, Pk_at_z
        )  # P_l(k,z) in 1/Mpc^3

        f_z_k_array, z_array = self.f_deriv(
            self.results.D_growth_zk, self.results.zgrid, self.results.kgrid
        )
        self.results.f_growthrate_zk = RectBivariateSpline(z_array, self.kgrid_1Mpc, f_z_k_array.T)

        def vectorized_halofit(z):
            return ((1 / self.h_now) ** 3) * self.symbfit.run_halofit(
                self.kgrid_hMpc,
                self.symbcosmopars["sigma8"],
                self.symbcosmopars["Omegam"],
                self.symbcosmopars["Omegab"],
                self.h_now,
                self.symbcosmopars["ns"],
                a=cosmo_functions.scale_factor(z),
                emulator=self.emulator_precision,
                extrapolate=self.extrapolate,
                which_params=self.which_params,
                add_correction=self.add_correction,
            )

        # symbfit run_halofit returns P_nl(k,z) in Mpc^3/h^3, requests kgrid in h/Mpc
        vectorized_run_halofit = np.vectorize(vectorized_halofit, signature="()->(n)")
        Pk_nl = vectorized_run_halofit(self.zgrid)
        self.results.Pk_nl = RectBivariateSpline(self.zgrid, self.kgrid_1Mpc, Pk_nl)
        # Pk_nl in 1/Mpc^3
        if self.feed_lvl > 2:
            print("Symbolic results took {:.2f} s".format(time() - tini_results))
        return self.results


class external_input:
    def __init__(self, cosmopars, fiducialcosmopars=dict(), external=dict(), extra_settings=dict()):
        """
        Initialize the external_input class.

        Parameters
        ----------
        cosmopars : dict
            The cosmological parameters object to be copied.
        fiducialcosmopars : dict, optional
            Fiducial cosmological parameter dict (default is an empty dict).
        external : dict, optional
            Dictionary of attributes for external files (default is an empty dict).
        extra_settings : dict, optional
            Dictionary of settings relevant for external input (default is an empty dict).

        Raises
        ------
        ValueError
            If the external_input class has been initialized wrongly.
        """
        self.cosmopars = cosmopars
        self.fiducialpars = fiducialcosmopars
        self.feed_lvl = cfg.settings["feedback"]
        self.external = external  # cfg.external
        self.settings = extra_settings  # cfg.settings
        self.activate_MG = None
        if (
            self.settings["external_activateMG"] is True
            or self.settings["activateMG"] == "external"
        ):
            self.activate_MG = "external"
            upr.debug_print("********EXTERNAL: activateMG = 'external'")
        if not fiducialcosmopars or not external or not extra_settings:
            raise ValueError("The external_input class has been initialized wrongly")
        self.directory = self.external["directory"]  ##['baseline_cosmo/']
        self.param_names = self.external[
            "paramnames"
        ]  ## ['Om', 'Ob', ....] ##have to be the same as keys in cosmopars dict
        self.folder_param_names = self.external[
            "folder_paramnames"
        ]  ## ['Om', 'Ob', ....] ##have to be the same as keys in cosmopars dict
        self.folder_param_dict = dict(zip(self.param_names, self.folder_param_names))
        self.epsilon_values = self.external["eps_values"]  ### [0.01, 0.1,...]
        self.epsilon_names = "eps_"
        self.signstrings = {-1.0: "mn", 1.0: "pl"}
        param_folder_string = self.get_param_string_from_value(cosmopars)
        upr.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="-**-> File folder used: {:s}".format(param_folder_string),
        )
        self.cb_files_on = False
        if (
            cfg.settings["GCsp_Tracer"] == "clustering"
            or cfg.settings["GCph_Tracer"] == "clustering"
        ):
            self.cb_files_on = True
        self.load_txt_files(parameter_string=param_folder_string)
        self.calculate_interpol_results(parameter_string=param_folder_string)

    def load_txt_files(self, parameter_string="fiducial_eps_0"):
        """
        Load text files containing cosmological data.

        Parameters
        ----------
        parameter_string : str, optional
            String representing the parameter set to load (default is "fiducial_eps_0").
        """
        self.input_arrays = dict()
        z_grid_filename = "z_values_list.txt"  # just as a safeguard value here
        # check if z_list exists
        z_arr_file = self.external["file_names"]["z_arr"]
        k_arr_file = self.external["file_names"]["k_arr"]
        self.k_arr_special_file = self.external["file_names"].get("k_arr_special", None)
        self.k_arr_nonlin_file = self.external["file_names"].get("k_arr_nonlin", None)
        H_z_file = self.external["file_names"]["H_z"]
        s8_z_file = self.external["file_names"]["s8_z"]
        D_zk_file = self.external["file_names"]["D_zk"]
        f_zk_file = self.external["file_names"]["f_zk"]
        Pl_zk_file = self.external["file_names"]["Pl_zk"]
        Pnl_zk_file = self.external["file_names"]["Pnl_zk"]
        SigWL_zk_file = self.external["file_names"].get("SigmaWL", None)

        upr.debug_print(os.path.join(self.directory, parameter_string, z_arr_file + ".*"))
        try:
            z_grid_filename = glob(
                os.path.join(self.directory, parameter_string, z_arr_file + ".*")
            )[0]
        except IndexError:
            print(
                "Folder or path not correctly specified: "
                + str(os.path.join(self.directory, parameter_string, z_arr_file + ".*"))
            )
            raise
        if os.path.isfile(z_grid_filename):
            self.input_arrays[("z_grid", parameter_string)] = np.loadtxt(z_grid_filename)
        else:
            z_grid_filename = glob(
                os.path.join(self.directory, "fiducial_eps_0", z_arr_file + ".*")
            )[0]
            self.input_arrays[("z_grid", parameter_string)] = np.loadtxt(z_grid_filename)
        # check if k_list exists
        k_grid_filename = glob(os.path.join(self.directory, parameter_string, k_arr_file + ".*"))[0]
        if os.path.isfile(k_grid_filename):
            self.input_arrays[("k_grid", parameter_string)] = np.loadtxt(k_grid_filename)
        else:
            k_grid_filename = glob(
                os.path.join(self.directory, "fiducial_eps_0", k_arr_file + ".*")
            )[0]
            self.input_arrays[("k_grid", parameter_string)] = np.loadtxt(k_grid_filename)

        if self.k_arr_special_file is not None:
            k_grid_special_filename = glob(
                os.path.join(self.directory, "fiducial_eps_0", self.k_arr_special_file + ".*")
            )[0]
            self.input_arrays[("k_grid_special", parameter_string)] = np.loadtxt(
                k_grid_special_filename
            )

        if self.k_arr_nonlin_file is not None:
            k_grid_nonlin_filename = glob(
                os.path.join(self.directory, parameter_string, self.k_arr_nonlin_file + ".*")
            )[0]
            if os.path.isfile(k_grid_nonlin_filename):
                self.input_arrays[("k_grid_nonlin", parameter_string)] = np.loadtxt(
                    k_grid_nonlin_filename
                )
            else:
                k_grid_nonlin_filename = glob(
                    os.path.join(self.directory, "fiducial_eps_0", self.k_arr_nonlin_file + ".*")
                )[0]
                self.input_arrays[("k_grid_nonlin", parameter_string)] = np.loadtxt(
                    k_grid_nonlin_filename
                )
        # check if background_Hz list exists, if not, take fiducial one
        # (to allow for easier import of parameters that do not affect background)
        Hz_filename = glob(os.path.join(self.directory, parameter_string, H_z_file + ".*"))[0]
        if os.path.isfile(Hz_filename):
            self.input_arrays[("H_z", parameter_string)] = np.loadtxt(Hz_filename)
        else:
            Hz_filename = glob(os.path.join(self.directory, "fiducial_eps_0", H_z_file + ".*"))[0]
            self.input_arrays[("H_z", parameter_string)] = np.loadtxt(Hz_filename)

        self.input_arrays[("s8_z", parameter_string)] = np.loadtxt(
            glob(os.path.join(self.directory, parameter_string, s8_z_file + ".*"))[0]
        )
        self.input_arrays[("D_zk", parameter_string)] = np.loadtxt(
            glob(os.path.join(self.directory, parameter_string, D_zk_file + ".*"))[0]
        )
        self.input_arrays[("f_zk", parameter_string)] = np.loadtxt(
            glob(os.path.join(self.directory, parameter_string, f_zk_file + ".*"))[0]
        )
        self.input_arrays[("Pkl_zk", parameter_string)] = np.loadtxt(
            glob(os.path.join(self.directory, parameter_string, Pl_zk_file + ".*"))[0]
        )
        self.input_arrays[("Pknl_zk", parameter_string)] = np.loadtxt(
            glob(os.path.join(self.directory, parameter_string, Pnl_zk_file + ".*"))[0]
        )
        if SigWL_zk_file is not None:
            self.input_arrays[("SigWL_zk", parameter_string)] = np.loadtxt(
                glob(os.path.join(self.directory, parameter_string, SigWL_zk_file + ".*"))[0]
            )
        if self.cb_files_on:
            s8cb_z_file = self.external["file_names"]["s8cb_z"]
            fcb_zk_file = self.external["file_names"]["fcb_zk"]
            Plcb_zk_file = self.external["file_names"]["Plcb_zk"]
            Pnlcb_zk_file = self.external["file_names"]["Pnlcb_zk"]
            self.input_arrays[("s8cb_z", parameter_string)] = np.loadtxt(
                glob(os.path.join(self.directory, parameter_string, s8cb_z_file + ".*"))[0]
            )
            self.input_arrays[("fcb_zk", parameter_string)] = np.loadtxt(
                glob(os.path.join(self.directory, parameter_string, fcb_zk_file + ".*"))[0]
            )
            self.input_arrays[("Pklcb_zk", parameter_string)] = np.loadtxt(
                glob(os.path.join(self.directory, parameter_string, Plcb_zk_file + ".*"))[0]
            )
            self.input_arrays[("Pknlcb_zk", parameter_string)] = np.loadtxt(
                glob(os.path.join(self.directory, parameter_string, Pnlcb_zk_file + ".*"))[0]
            )
        if upr.debug:
            for stri in ["z_grid", "k_grid", "H_z", "s8_z", "D_zk", "f_zk", "Pkl_zk", "Pknl_zk"]:
                upr.debug_print(
                    stri + " grid shape : " + str(self.input_arrays[(stri, parameter_string)].shape)
                )

    def get_param_string_from_value(self, cosmopars):
        """
        Get the parameter string from the cosmological parameters.

        Parameters
        -----------
        cosmopars : dict
            Dictionary of cosmological parameters.

        Returns
        --------
        str
            Parameter string representing the input cosmological parameters.
        """
        rel_tol = 1e-5
        for parname in self.param_names:
            if not np.isclose(cosmopars[parname], self.fiducialpars[parname], rtol=rel_tol):
                if self.fiducialpars[parname] != 0:
                    delta_eps = (cosmopars[parname] / self.fiducialpars[parname]) - 1.0
                elif self.fiducialpars[parname] == 0.0:
                    delta_eps = cosmopars[parname] - self.fiducialpars[parname]
                eps_sign = np.sign(delta_eps)
                sign_string = self.signstrings[eps_sign]
                # print('delta_eps before = {:.16f}'.format(delta_eps))
                eps_vals = np.array(self.external["eps_values"])
                allowed_eps_vals = np.concatenate((eps_vals, -eps_vals, np.array([0])))
                # print('delta_eps before = {:.6f}'.format(delta_eps))
                delta_eps = unu.closest(allowed_eps_vals, delta_eps)
                delta_eps = unu.round_decimals_up(delta_eps)
                # print('delta_eps after = {:.6f}'.format(delta_eps))
                eps_string = "{:.1E}".format(abs(delta_eps))
                eps_string = eps_string.replace(".", "p")
                if self.external["E-00"] is False:
                    eps_string = eps_string.replace("E-0", "E-")
                folder_parname = self.folder_param_dict[parname]
                param_folder_string = (
                    folder_parname + "_" + sign_string + "_" + "eps" + "_" + eps_string
                )
                break
            else:
                param_folder_string = "fiducial_eps_0"
        return param_folder_string

    def calculate_interpol_results(self, parameter_string="fiducial_eps_0"):
        """
        Calculate interpolated results from the input data.

        Parameters
        -----------
        parameter_string : str, optional
            String representing the parameter set to use (default is "fiducial_eps_0").
        """
        pk_units_factor = 1  ## units in all the code are in Mpc or Mpc^-1
        r_units_factor = 1
        k_units_factor = 1
        if self.external is not None:
            k_units = self.external["k-units"]
            r_units = self.external["r-units"]
            if k_units == "h/Mpc":
                pk_units_factor = (1 / self.cosmopars["h"]) ** 3
                k_units_factor = self.cosmopars["h"]
            if r_units == "Mpc/h":
                r_units_factor = 1 / self.cosmopars["h"]
            elif r_units == "km/s/Mpc":
                r_units_factor = cosmo_functions.c

        self.results = types.SimpleNamespace()
        self.results.zgrid = self.input_arrays[("z_grid", parameter_string)].flatten()
        self.results.kgrid = (k_units_factor) * self.input_arrays[
            ("k_grid", parameter_string)
        ].flatten()
        if self.k_arr_special_file is not None:
            self.results.kgrid_special = (k_units_factor) * self.input_arrays[
                ("k_grid_special", parameter_string)
            ].flatten()
        else:
            self.results.kgrid_special = self.results.kgrid
        if self.k_arr_nonlin_file is not None:
            self.results.kgrid_nonlin = (k_units_factor) * self.input_arrays[
                ("k_grid_nonlin", parameter_string)
            ].flatten()
        else:
            self.results.kgrid_nonlin = self.results.kgrid
        self.results.h_of_z = InterpolatedUnivariateSpline(
            self.results.zgrid,
            (1 / r_units_factor) * self.input_arrays[("H_z", parameter_string)].flatten(),
        )
        dcom_arr = np.array(
            [_dcom_func_trapz(zii, self.results.h_of_z) for zii in self.results.zgrid]
        )
        self.results.com_dist = InterpolatedUnivariateSpline(self.results.zgrid, dcom_arr)
        self.results.ang_dist = InterpolatedUnivariateSpline(
            self.results.zgrid, dcom_arr / (1 + self.results.zgrid)
        )

        ky_ord = 3
        kx_ord = 3
        self.results.D_growth_zk = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid_special,
            self.input_arrays[("D_zk", parameter_string)],
            kx=kx_ord,
            ky=ky_ord,
        )
        self.results.f_growthrate_zk = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid_special,
            self.input_arrays[("f_zk", parameter_string)],
            kx=kx_ord,
            ky=ky_ord,
        )
        self.results.s8_of_z = InterpolatedUnivariateSpline(
            self.results.zgrid, self.input_arrays[("s8_z", parameter_string)].flatten()
        )
        self.results.Pk_l = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid,
            pk_units_factor * (self.input_arrays[("Pkl_zk", parameter_string)]),
            kx=kx_ord,
            ky=ky_ord,
        )
        self.results.Pk_nl = RectBivariateSpline(
            self.results.zgrid,
            self.results.kgrid_nonlin,
            pk_units_factor * (self.input_arrays[("Pknl_zk", parameter_string)]),
            kx=kx_ord,
            ky=ky_ord,
        )
        if self.cb_files_on:
            self.results.f_growthrate_cb_zk = RectBivariateSpline(
                self.results.zgrid,
                self.results.kgrid_special,
                self.input_arrays[("fcb_zk", parameter_string)],
                kx=kx_ord,
                ky=ky_ord,
            )
            self.results.s8_cb_of_z = InterpolatedUnivariateSpline(
                self.results.zgrid, self.input_arrays[("s8cb_z", parameter_string)].flatten()
            )
            self.results.Pk_cb_l = RectBivariateSpline(
                self.results.zgrid,
                self.results.kgrid,
                pk_units_factor * (self.input_arrays[("Pklcb_zk", parameter_string)]),
                kx=kx_ord,
                ky=ky_ord,
            )
            self.results.Pk_cb_nl = RectBivariateSpline(
                self.results.zgrid,
                self.results.kgrid_nonlin,
                pk_units_factor * (self.input_arrays[("Pknlcb_zk", parameter_string)]),
                kx=kx_ord,
                ky=ky_ord,
            )
        if self.activate_MG == "external":
            self.results.SigWL_zk = RectBivariateSpline(
                self.results.zgrid,
                self.results.kgrid,
                (self.input_arrays[("SigWL_zk", parameter_string)]),
                kx=kx_ord,
                ky=ky_ord,
            )
        ### Reset the zgrid and kgrid to the fiducial ones, in case one parameter variation had a different one
        # self.results.zgrid = self.input_arrays[('z_grid','fiducial_eps_0')].flatten()
        # self.results.kgrid = self.input_arrays[('k_grid','fiducial_eps_0')].flatten()


class cosmo_functions:
    c = sconst.speed_of_light / 1000  ##speed of light in km/s

    def __init__(self, cosmopars, input=None):
        """
        Initialize the cosmo_functions class.

        Parameters
        ----------
        cosmopars : dict
            Dictionary containing the cosmological parameters.
        input : str, optional
            Input type for the cosmological code (default is None).

        Raises
        ------
        ValueError
            If an unsupported input type is specified.
        """
        self.settings = cfg.settings
        self.fiducialcosmopars = cfg.fiducialparams
        self.input = input
        if input is None:
            input = cfg.input_type
        if input == "camb":
            cambresults = boltzmann_code(cosmopars, code="camb")
            self.code = "camb"
            self.results = cambresults.results
            self.kgrid = cambresults.results.kgrid
            self.cosmopars = cambresults.cosmopars
            self.cambcosmopars = cambresults.cambclasspars
        elif input == "external":
            self.external = cfg.external
            ## filter settings which contain the word 'external_'
            # extra_settings = dict([[kk, self.settings[kk]] for kk in self.settings.keys() if 'external_' in kk])
            # externalinput = external_input(cosmopars,  fiducialcosmopars=self.fiducialcosmopars,
            #                                external=self.external, extra_settings=extra_settings)
            if self.settings["memorize_cosmo"]:
                externalinput = memorize_external_input(
                    cosmopars, self.fiducialcosmopars, self.external, self.settings
                )
            else:
                externalinput = external_input(
                    cosmopars,
                    fiducialcosmopars=self.fiducialcosmopars,
                    external=self.external,
                    extra_settings=self.settings,
                )
            self.code = "external"
            self.results = externalinput.results
            self.kgrid = externalinput.results.kgrid
            self.cosmopars = externalinput.cosmopars
        elif input == "class":
            classresults = boltzmann_code(cosmopars, code="class")
            self.code = "class"
            self.results = classresults.results
            self.Classres = classresults.Classres
            self.kgrid = classresults.results.kgrid
            self.cosmopars = classresults.cosmopars
            self.classcosmopars = classresults.classcosmopars
        elif input == "symbolic":
            symbresults = boltzmann_code(cosmopars, code="symbolic")
            self.code = "symbolic"
            self.results = symbresults.results
            self.kgrid = symbresults.results.kgrid
            self.cosmopars = symbresults.cosmopars
            self.symbcosmopars = symbresults.symbcosmopars
        else:
            print(input, ":  This input type is not implemented yet")

    def Hubble(self, z, physical=False):
        """
        Hubble function

        Parameters
        ----------
        z     : float
                redshift

        physical: bool
                Default False, if True, return H(z) in (km/s/Mpc).
        Returns
        -------
        float
            Hubble function values (Mpc^-1) at the redshifts of the input redshift

        """
        prefactor = 1
        if physical:
            prefactor = self.c

        hubble = prefactor * self.results.h_of_z(z)

        return hubble

    def E_hubble(self, z):
        """
        E(z) dimensionless Hubble function

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Dimensionless E(z) Hubble function values at the redshifts of the input redshift

        """

        H0 = self.Hubble(0.0)
        Eofz = self.Hubble(z) / H0

        return Eofz

    def angdist(self, z):
        """
        Angular diameter distance

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Angular diameter distance values at the redshifts of the input redshift

        """

        dA = self.results.ang_dist(z)

        return dA

    def matpow(self, z, k, nonlinear=False, tracer="matter"):
        """
        Calculates the power spectrum of a given tracer quantity at a specific redshift and wavenumber.

        Parameters
        ----------
        z : float
            The redshift of interest.

        k : array_like
            An array of wavenumbers at which to compute the power spectrum. These must be in units of 1/Mpc and
            should be sorted in increasing order.

        nonlinear : bool, optional
            A boolean indicating whether or not to include nonlinear corrections to the matter power spectrum. The default
            value is False.

        tracer : str, optional
            A string indicating which trace quantity to use for computing the power spectrum. If this argument is "matter"
            or anything other than "clustering", the power spectrum functions `Pmm` will be used to compute the power
            spectrum. If the argument is "clustering", the power spectrum function `Pcb` will be used instead. The default
            value is "matter".

        Returns
        -------
        np.ndarray:
            Array containing the calculated power spectrum values.

        Warnings
        --------
        If `tracer` is not "matter" or "clustering", a warning message is printed to the console saying the provided tracer was not
        recognized and the function defaults to using `Pmm` to calculate the power spectrum of matter.
        """
        if tracer == "clustering":
            return self.Pcb(z, k, nonlinear=nonlinear)

        if tracer != "matter":
            warn("Did not recognize tracer: reverted to matter")
        return self.Pmm(z, k, nonlinear=nonlinear)

    def Pmm(self, z, k, nonlinear=False):
        """
        Compute the power spectrum of the total matter species  (MM) at a given redshift and wavenumber.

        Args:
            self: An instance of the current class.
            z: The redshift at which to compute the MM power spectrum.
            k: The wavenumber at which to compute the MM power spectrum in 1/Mpc.
            nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

        Returns:
            float: The value of the MM power spectrum at the given redshift and wavenumber.
        """
        if nonlinear is True:
            power = self.results.Pk_nl(z, k, grid=False)
        elif nonlinear is False:
            power = self.results.Pk_l(z, k, grid=False)
        return power  # type: ignore

    def Pcb(self, z, k, nonlinear=False):
        """
        Compute the power spectrum of the clustering matter species  (CB) at a given redshift and wavenumber.

        Args:
            self: An instance of the current class.
            z: The redshift at which to compute the CB power spectrum.
            k: The wavenumber at which to compute the CB power spectrum in 1/Mpc.
            nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

        Returns:
            The value of the CB power spectrum at the given redshift and wavenumber.
        """
        if nonlinear is True:
            power = self.results.Pk_cb_nl(z, k, grid=False)
        elif nonlinear is False:
            power = self.results.Pk_cb_l(z, k, grid=False)
        return power

    def nonwiggle_pow(self, z, k, nonlinear=False, tracer="matter"):
        """
        Calculate the power spectrum at a specific redshift and wavenumber,
        after smoothing to remove baryonic acoustic oscillations (BAO).

        Args:
            z: The redshift of interest.
            k: An array of wavenumbers at which to compute the power
                spectrum. Must be in units of Mpc^-1/h. Should be sorted in
                increasing order.
            nonlinear: Whether to include nonlinear corrections
                to the matter power spectrum. Default is False.
            tracer: Which perturbations to use for computing
                the power spectrum. Options are 'matter' or 'clustering'.
                Default is 'matter'.

        Returns:
            An array of power spectrum values corresponding to the
            input wavenumbers. Units are (Mpc/h)^3.

        Note:
            This function computes the power spectrum of a given tracer quantity
            at a specific redshift, using the matter power spectrum function
            `matpow`. It then applies a Savitzky-Golay filter to smooth out the
            BAO features in the power spectrum. This is done by first taking the
            natural logarithm of the power spectrum values at a set of logarithmic
            wavenumbers spanning from `kmin_loc` to `kmax_loc`. The smoothed power
            spectrum is then returned on a linear (not logarithmic) grid of
            wavenumbers given by the input array `k`.
        """
        unitsf = self.cosmopars["h"]
        kmin_loc = unitsf * self.settings["savgol_internalkmin"]
        kmax_loc = unitsf * np.max(self.kgrid)
        loc_samples = self.settings["savgol_internalsamples"]
        log_kgrid_loc = np.linspace(np.log(kmin_loc), np.log(kmax_loc), loc_samples)
        poly_order = self.settings["savgol_polyorder"]
        dlnk_loc = np.mean(log_kgrid_loc[1:] - log_kgrid_loc[0:-1])
        savgol_width = self.settings["savgol_width"]
        n_savgol = int(np.round(savgol_width / np.log(1 + dlnk_loc)))
        upr.debug_print(n_savgol)
        upr.debug_print(savgol_width)
        upr.debug_print(dlnk_loc)
        upr.debug_print(kmin_loc)
        upr.debug_print(kmax_loc)
        intp_p = InterpolatedUnivariateSpline(
            log_kgrid_loc,
            np.log(
                self.matpow(z, np.exp(log_kgrid_loc), nonlinear=nonlinear, tracer=tracer).flatten()
            ),
            k=1,
        )
        pow_sg = savgol_filter(intp_p(log_kgrid_loc), n_savgol, poly_order)
        intp_pnw = InterpolatedUnivariateSpline(np.exp(log_kgrid_loc), np.exp(pow_sg), k=1)
        return intp_pnw(k)

    def comoving(self, z):
        """
        Comoving distance

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Comoving distance values at the redshifts of the input redshift

        """

        chi = self.results.com_dist(z)

        return chi

    def sigma8_of_z(self, z, tracer="matter"):
        """
        sigma_8

        Parameters
        ----------
        z     : float
                redshift
        tracer: String
                either 'matter' if you want sigma_8 calculated from the total matter power spectrum or 'clustering' if you want it from the Powerspectrum with massive neutrinos substracted
        Returns
        -------
        float
            The Variance of the matter perturbation smoothed over a scale of 8 Mpc/h

        """
        if tracer == "clustering":
            return self.results.s8_cb_of_z(z)
        elif tracer == "matter":
            return self.results.s8_of_z(z)
        else:
            warn("Did not recognize tracer: reverted to matter")
            return self.results.s8_of_z(z)

    def growth(self, z, k=None):
        """
        Growth factor

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth factor values at the redshifts of the input redshift

        """
        if k is None:
            k = 0.0001
        Dg = self.results.D_growth_zk(z, k, grid=False)

        return Dg

    def Omegam_of_z(self, z):
        """
        Omega matter fraction as a function of redshift

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Omega matter (total) at the redshifts of the input redshift `z`


        Note
        -----
        Assumes standard matter evolution
        Implements the following equation:

        .. math::
            Omega_m(z) = Omega_{m,0}*(1+z)^3 / E^2(z)
        """
        omz = 0
        if self.input == "external":
            omz = (self.cosmopars["Omegam"] * (1 + z) ** 3) / self.E_hubble(z) ** 2
        else:
            omz = self.results.Om_m(z)

        return omz

    def f_growthrate(self, z, k=None, gamma=False, tracer="matter"):
        r"""
        Growth rate in LCDM gamma approximation

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth rate values at the redshifts of the input redshift,
            using self.gamma as gamma value.


        Note
        -----
        Implements the following equation:

        .. math::
            f(z) = Omega_m(z)^{\gamma}
        """
        if k is None:
            k = 0.0001

        if tracer == "clustering":
            fg = self.results.f_growthrate_cb_zk(z, k, grid=False)
            return fg
        if tracer != "matter":
            warn("Did not recognize tracer: reverted to matter")

        if gamma is False:
            fg = self.results.f_growthrate_zk(z, k, grid=False)
        else:
            # Assumes standard Omega_matter evolution in z
            fg = np.power(self.Omegam_of_z(z), self.gamma)

        return fg

    def fsigma8_of_z(self, z, k=None, gamma=False, tracer="matter"):
        r"""
        Growth rate in LCDM gamma approximation

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth rate values at the redshifts of the input redshift,
            using self.gamma as gamma value.


        Note
        -----
        Implements the following equation:

        .. math::
            f(z) = Omega_m(z)^{\gamma}
        """
        # Assumes standard Omega_matter evolution in z
        fs8 = self.f_growthrate(z, k, gamma, tracer=tracer) * self.sigma8_of_z(z, tracer=tracer)

        return fs8

    def SigmaMG(self, z, k):
        """
        Compute the modified growth rate Sigma for modified gravity models.

        Parameters
        ----------
        z : float
            Redshift
        k : float
            Wavenumber

        Returns
        -------
        float
            Modified growth rate Sigma
        """
        Sigma = np.array(1)
        if self.settings["activateMG"] == "late-time":
            E11 = self.cosmopars["E11"]
            E22 = self.cosmopars["E22"]
            # TODO: Fix for non-flat models
            Omega_DE = 1 - self.Omegam_of_z(z)
            mu = 1 + E11 * Omega_DE
            eta = 1 + E22 * Omega_DE
            Sigma = (mu / 2) * (1 + eta)
        elif (
            self.settings["external_activateMG"] is True
            or self.settings["activateMG"] == "external"
        ):
            Sigma = self.results.SigWL_zk(z, k, grid=False)

        return Sigma

    def cmb_power(self, lmin, lmax, obs1, obs2):
        """
        Compute the CMB power spectrum.

        Parameters
        ----------
        lmin : int
            Minimum multipole moment
        lmax : int
            Maximum multipole moment
        obs1 : str
            First observable (e.g., 'CMB_TCMB_T')
        obs2 : str
            Second observable (e.g., 'CMB_ECMB_E')

        Returns
        -------
        np.ndarray
            Array of CMB power spectrum values
        """
        if self.code == "camb":
            if self.cambcosmopars.Want_CMB:
                print("CMB Spectrum not computed")
                return
        elif self.code == "class":
            if "tCl" not in self.classcosmopars["output"]:
                print("CMB Spectrum not computed")
                return
        else:
            ells = np.arange(lmin, lmax)

            norm_fac = 2 * np.pi / (ells * (ells + 1))

            if obs1 + obs2 == "CMB_TCMB_T":
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 0]
            elif obs1 + obs2 == "CMB_ECMB_E":
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 1]
            elif obs1 + obs2 == "CMB_BCMB_B":
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 2]
            elif (obs1 + obs2 == "CMB_TCMB_E") or (obs1 + obs2 == "CMB_ECMB_T"):
                cls = norm_fac * self.results.camb_cmb[lmin:lmax, 3]
            else:
                cls = np.array([0.0] * len(ells))

            return cls

    @staticmethod
    def scale_factor(z):
        """
        Compute the scale factor a from redshift z.

        Parameters
        ----------
        z : float or array-like
            Redshift

        Returns
        -------
        float or array-like
            Scale factor a
        """
        return 1.0 / (1.0 + z)
