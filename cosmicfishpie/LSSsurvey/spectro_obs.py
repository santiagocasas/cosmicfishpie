# -*- coding: utf-8 -*-
"""CLS

This module contains cls calculations (only LSS atm).

"""
import warnings
from copy import deepcopy
from time import time

import numpy as np
import scipy.integrate as integrate

import cosmicfishpie.configs.config as cfg
import cosmicfishpie.cosmology.cosmology as cosmology
import cosmicfishpie.cosmology.nuisance as nuisance
from cosmicfishpie.utilities.utils import printing as upt


class ComputeGalSpectro:
    # Class attributes shared among all class instances

    def __init__(
        self,
        cosmopars,
        fiducial_cosmopars=None,
        spectrobiaspars=None,
        IMbiaspars=None,
        spectrononlinearpars=None,
        PShotpars=None,
        fiducial_cosmo=None,
        bias_samples=None,
        use_bias_funcs=False,
        configuration=None,
    ):
        """Class to compute the observed power spectrum for spectroscopic galaxy clustering and intensity mapping experiments.

        Parameters
        ----------
        cosmopars            : dict
                              A dictionary containing the cosmological parameters of the sample cosmology
                              A dictionary containing the cosmological parameters of the sample cosmology
        fiducial_cosmopars   : dict, optional
                              A dictionary containing the cosmological parameters of the fiducial/reference cosmology
        spectrobiaspars      : dict, optional
                              A dictionary containing the specifications for the galaxy biases
        IMbiaspars          : dict, optional
                              A dictionary containing the specifications for the intensity mapping biases
                              A dictionary containing the cosmological parameters of the fiducial/reference cosmology
        spectrobiaspars      : dict, optional
                              A dictionary containing the specifications for the galaxy biases
        IMbiaspars          : dict, optional
                              A dictionary containing the specifications for the intensity mapping biases
        spectrononlinearpars : dict, optional
                              A dictionary containing the values of the non linear modeling parameters entering FOG and the dewiggling weight per bin
        PShotpars           : dict, optional
                              A dictionary containing the values of the additional shot noise per bin
        fiducial_cosmo      : cosmicfishpie.cosmology.cosmology.cosmo_functions, optional
                              An instance of `cosmo_functions` of the fiducial cosmology
        bias_samples        : list, optional
                              A list of two strings specifying if galaxy clustering, intensity mapping or cross correlation power spectrum should be computed.
                              Use "g" for galaxy and "I" for intensity mapping. Default: ["g", "g"]
        use_bias_funcs     : bool, optional
                              If True will compute the bias function by constructing it from the specification file.
                              If False it will be recomputed from bias parameters. Default: False
        configuration      : object, optional
                              Configuration object containing settings and parameters. If None, uses default config

        Attributes
        ----------
        feed_lvl                      : int
                                       Number indicating the verbosity of the output. Higher numbers mean more output
                                       Number indicating the verbosity of the output. Higher numbers mean more output
        observables                   : list
                                       A list of the observables that the observed power spectrum is computed for
                                       A list of the observables that the observed power spectrum is computed for
        s8terms                       : bool
                                       If True will expand the observed power spectrum with :math:`\\sigma_8` to match the IST:F recipe
                                       If True will expand the observed power spectrum with :math:`\\sigma_8` to match the IST:F recipe
        tracer                        : str
                                       What Power spectrum should be used when calculating the power spectrum. Either "matter" or "clustering"
                                       What Power spectrum should be used when calculating the power spectrum. Either "matter" or "clustering"
        cosmo                         : cosmicfishpie.cosmology.cosmology.cosmo_functions
                                       An instance of `cosmo_functions` of the sample cosmology
                                       An instance of `cosmo_functions` of the sample cosmology
        nuisance                      : cosmicfishpie.cosmology.Nuisance.Nuisance
                                       An instance of `nuisance` that contains the relevant modeling of nuisance parameters
                                       An instance of `nuisance` that contains the relevant modeling of nuisance parameters
        extraPshot                    : dict
                                       A dictionary containing the values of the additional shot noise per bin
        gcsp_z_bin_mids              : numpy.ndarray
                                       Lists the redshift bin centers for galaxy clustering
        IM_z_bin_mids                : numpy.ndarray
                                       Lists the redshift bin centers for intensity mapping
                                       A dictionary containing the values of the additional shot noise per bin
        gcsp_z_bin_mids              : numpy.ndarray
                                       Lists the redshift bin centers for galaxy clustering
        IM_z_bin_mids                : numpy.ndarray
                                       Lists the redshift bin centers for intensity mapping
        k_grid                        : numpy.ndarray
                                       Lists all wavenumbers used in the internal calculations
                                       Lists all wavenumbers used in the internal calculations
        dk_grid                       : numpy.ndarray
                                       Lists the numerical distance between all wavenumbers used in the internal calculations
                                       Lists the numerical distance between all wavenumbers used in the internal calculations
        linear_switch                 : bool
                                       If False all nonlinear effects will be included in the computation
                                       If False all nonlinear effects will be included in the computation
        FoG_switch                    : bool
                                       If True and `linear_switch` is True, then the finger of god effect will be modelled
                                       If True and `linear_switch` is True, then the finger of god effect will be modelled
        APbool                        : bool
                                       If True and `linear_switch` is True, then the Alcock-Paczynski effect will be considered
                                       If True and `linear_switch` is True, then the Alcock-Paczynski effect will be considered
        fix_cosmo_nl_terms            : bool
                                       If True and the nonlinear modeling parameters are not varied, then they will be fixed to the fiducial cosmology values
                                       If True and the nonlinear modeling parameters are not varied, then they will be fixed to the fiducial cosmology values
        dz_err                        : float
                                       Value of the spectroscopic redshift error
        Dd                            : float
                                       Dish diameter for intensity mapping in meters
        lambda_21                     : float
                                       21cm wavelength in meters
        fsky_IM                       : float
                                       Sky fraction for intensity mapping
        t_tot                         : float
                                       Total observation time in seconds
        N_d                           : int
                                       Number of dishes for intensity mapping

        Notes
        -----
        This class can compute:
        - Galaxy clustering power spectrum
        - HI intensity mapping power spectrum
        - Cross-correlation between galaxy clustering and intensity mapping

        The type of power spectrum is determined by the `bias_samples` parameter:
        - ["g", "g"]: Galaxy auto-correlation
        - ["I", "I"]: Intensity mapping auto-correlation
        - ["g", "I"] or ["I", "g"]: Cross-correlation
        """
        tini = time()
        if configuration is None:
            self.config = cfg
        else:
            self.config = configuration

        upt.debug_print("Initializing ComputeGalSpectro with the following configuration:")
        upt.debug_print(self.config.__dict__)

        self.feed_lvl = self.config.settings["feedback"]
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="Entered ComputeGalSpectro",
            instance=self,
        )

        self.observables = deepcopy(self.config.obs)
        self.specs = deepcopy(self.config.specs)

        self.s8terms = deepcopy(self.config.settings["bfs8terms"])
        self.tracer = deepcopy(self.config.settings["GCsp_Tracer"])

        self.set_fiducial_cosmology(
            fiducial_cosmopars=fiducial_cosmopars, fiducial_cosmo=fiducial_cosmo
        )
        self.cosmopars = cosmopars
        if self.cosmopars == self.fiducial_cosmopars:
            self.cosmo = self.fiducialcosmo
        else:
            self.cosmo = cosmology.cosmo_functions(cosmopars, self.config.input_type)

        # Load the Nuisance Parameters
        self.fiducial_spectrobiaspars = deepcopy(self.config.Spectrobiasparams)
        if spectrobiaspars is None:
            self.spectrobiaspars = self.fiducial_spectrobiaspars
        else:
            self.spectrobiaspars = spectrobiaspars
        # Load the Non Linear Nuisance Parameters
        self.fiducial_spectrononlinearpars = deepcopy(self.config.Spectrononlinearparams)
        if spectrononlinearpars is None:
            spectrononlinearpars = self.fiducial_spectrononlinearpars
        self.spectrononlinearpars = spectrononlinearpars

        self.fiducial_IMbiaspars = deepcopy(self.config.IMbiasparams)
        if IMbiaspars is None:
            self.IMbiaspars = self.fiducial_IMbiaspars
        else:
            self.IMbiaspars = IMbiaspars

        self.nuisance = nuisance.Nuisance(
            configuration=self.config,
            spectrobiasparams=self.spectrobiaspars,
            spectrononlinearpars=self.spectrononlinearpars,
            IMbiasparams=self.IMbiaspars,
        )
        self.extraPshot = self.nuisance.extra_Pshot_noise()
        self.gcsp_z_bin_mids = self.nuisance.gcsp_zbins_mids()

        self.fiducial_PShotpars = deepcopy(self.config.PShotparams)
        if PShotpars is None:
            PShotpars = self.fiducial_PShotpars
        self.PShotpars = PShotpars

        self.allpars = {
            **self.cosmopars,
            **self.spectrobiaspars,
            **self.IMbiaspars,
            **self.IMbiaspars,
            **self.PShotpars,
            **self.spectrononlinearpars,
        }
        self.fiducial_allpars = {
            **self.fiducial_cosmopars,
            **self.fiducial_spectrobiaspars,
            **self.fiducial_IMbiaspars,
            **self.fiducial_IMbiaspars,
            **self.fiducial_PShotpars,
            **self.fiducial_spectrononlinearpars,
        }

        if "IM" in self.observables and "GCsp" in self.observables:
            self.obs_spectrum = ["I", "g"]
        elif "IM" in self.observables:  # compute  in the case of IM only
            self.obs_spectrum = ["I", "I"]
        elif "GCsp" in self.observables:
            self.obs_spectrum = ["g", "g"]
        if "IM" in self.observables and "GCsp" in self.observables:
            self.obs_spectrum = ["I", "g"]
        elif "IM" in self.observables:  # compute  in the case of IM only
            self.obs_spectrum = ["I", "I"]
        elif "GCsp" in self.observables:
            self.obs_spectrum = ["g", "g"]
        self.set_internal_kgrid()
        self.activate_terms()
        self.set_spectro_dz_specs()
        self.bias_samples = bias_samples
        self.use_bias_funcs = use_bias_funcs
        self.set_spectro_bias_specs()
        if "IM" in self.observables:
            self.set_IM_specs()
            self.set_IM_bias_specs()

        tend = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="GalSpec initialization done in: ",
            time_ini=tini,
            time_fin=tend,
            instance=self,
        )

    def set_internal_kgrid(self):
        """Updates the internal grid of wavenumbers used in the computation"""
        self.specs["kmax"] = self.specs["kmax_GCsp"]
        self.specs["kmin"] = self.specs["kmin_GCsp"]
        kmin_int = 0.001
        kmax_int = 5
        self.k_grid = np.logspace(np.log10(kmin_int), np.log10(kmax_int), 1024)
        self.dk_grid = np.diff(self.k_grid)[0]

    def activate_terms(self):
        """Update which modelling effects should be taken into consideration"""
        self.linear_switch = deepcopy(self.config.settings["GCsp_linear"])
        self.FoG_switch = deepcopy(self.config.settings["FoG_switch"])
        self.APbool = deepcopy(self.config.settings["AP_effect"])
        self.fix_cosmo_nl_terms = deepcopy(self.config.settings["fix_cosmo_nl_terms"])
        self.nonlinear_model = deepcopy(self.specs.get("nonlinear_model", "default"))
        self.nonlinear_parametrization = deepcopy(
            self.specs.get("nonlinear_parametrization", {"default": ""})
        )
        self.vary_sigmap = self.nonlinear_parametrization.get("vary_sigmap", False)
        self.vary_sigmav = self.nonlinear_parametrization.get("vary_sigmav", False)

    def set_spectro_dz_specs(self):
        """Updates the spectroscopic redshift error"""
        self.dz_err = self.specs["spec_sigma_dz"]
        self.dz_type = self.specs["spec_sigma_dz_type"]
        ## constant, z-dependent
        # These bugs are intentionally left in, in order to reproduce old results.
        # The reallity is that they are not to be here.
        self.kh_rescaling_bug = deepcopy(self.config.settings["kh_rescaling_bug"])
        self.kh_rescaling_beforespecerr_bug = deepcopy(
            self.config.settings["kh_rescaling_beforespecerr_bug"]
        )

    def set_spectro_bias_specs(self):
        """Updates the spectroscopic bias choices"""
        self.sp_bias_model = self.specs["sp_bias_model"]
        self.sp_bias_root = self.specs["sp_bias_root"]
        self.sp_bias_sample = self.specs["sp_bias_sample"]

    def set_fiducial_cosmology(self, fiducial_cosmopars=None, fiducial_cosmo=None):
        if fiducial_cosmopars is None:
            self.fiducial_cosmopars = deepcopy(self.config.fiducialparams)
        else:
            self.fiducial_cosmopars = deepcopy(fiducial_cosmopars)
        if self.fiducial_cosmopars == self.config.fiducialparams:
            try:
                try:
                    self.fiducialcosmo = self.config.fiducialcosmo
                    upt.time_print(
                        feedback_level=self.feed_lvl,
                        min_level=3,
                        text="Fiducial cosmology parameters: {}".format(
                            self.fiducialcosmo.cosmopars
                        ),
                        instance=self,
                    )
                except BaseException:
                    upt.debug_print("Fiducial cosmology from config.py raised an error")
                    # raise
                    try:
                        self.fiducialcosmo = fiducial_cosmo
                        upt.time_print(
                            feedback_level=self.feed_lvl,
                            min_level=3,
                            text="Fiducial cosmology parameters: {}".format(
                                self.fiducialcosmo.cosmopars
                            ),
                            instance=self,
                        )
                    except BaseException:
                        upt.debug_print("Fiducial cosmology from input arguments raised an error")
                        raise
            except BaseException:
                print(" >>>>> Fiducial cosmology could not be loaded, recomputing....")
                print(" **** In ComputeGalSpectro: Calculating fiducial cosmology...")
                self.fiducialcosmo = cosmology.cosmo_functions(
                    self.fiducial_cosmopars, self.config.input_type
                )
        else:
            print(
                "Error: In ComputeGalSpectro fiducial_cosmopars not equal to self.config.fiducialparams"
            )
            raise AttributeError

    def qparallel(self, z):
        """Function implementing q parallel of the Alcock-Paczynski effect

        Parameters
        ----------
        z : numpy.ndarray
            list of redshifts for which the q parallel should be computed

        Returns
        -------
        numpy.ndarray
            redshift dependant value of q parallel
        """
        qpar = self.fiducialcosmo.Hubble(z) / self.cosmo.Hubble(z)
        return qpar

    def qperpendicular(self, z):
        """Function implementing q perpendicular of the Alcock-Paczynski effect

        Parameters
        ----------
        z : numpy.ndarray
            list of redshifts for which the q perpendicular should be computed

        Returns
        -------
        numpy.ndarray
            redshift dependant value of q perpendicular
        """
        qper = self.cosmo.angdist(z) / self.fiducialcosmo.angdist(z)
        return qper

    def kpar(self, z, k, mu):
        """Computes the parallel projection of a wavevector. Takes into acount AP-effect

        Parameters
        ----------
        z  : float, numpy.ndarray
             The redshift of interest.
        k  : float, numpy.ndarray
             wavenumbers at which to compute the power spectrum. Must be in units of Mpc^-1/h.
        mu : float, numpy.ndarray
             cosine of the angel between the line of sight and the wavevector

        Returns
        -------
            Observed parallel projection of wavevector onto the line of sight with AP-effect corrected for
        """

        k_par = k * mu * (1 / self.qparallel(z))
        return k_par

    def kper(self, z, k, mu):
        """Computes the perpendicular projection of a wavevector. Takes into acount AP-effect

        Parameters
        ----------
        z  : float, numpy.ndarray
             The redshift of interest.
        k  : float, numpy.ndarray
             wavenumbers at which to compute the power spectrum. Must be in units of Mpc^-1/h.
        mu : float, numpy.ndarray
             cosine of the angel between the line of sight and the wavevector

        Returns
        -------
            Observed perpendicular projection of wavevector onto the line of sight with AP-effect corrected for
        """

        k_per = k * np.sqrt(1 - mu**2) * (1 / self.qperpendicular(z))
        return k_per

    def k_units_change(self, k):
        """
        Function that rescales the k-array, when asked for.
        The code is defined everywhere in 0/Mpc so a rescaling would be wrong.

        Parameters
        ----------
        k : float, numpy.ndarray
            wavenumbers in units of h sample/Mpc to be rescaled

        Returns
        -------
        float, numpy.ndarray
            wavenumbers in un units of h ref/Mpc
        """
        if self.kh_rescaling_bug:
            warnings.warn(
                "You requested to do an additional unphysical rescaling of the wavenumbers (h-bug).",
                category=RuntimeWarning,
                stacklevel=2,
            )
            h_change = self.cosmo.cosmopars["h"] / self.fiducialcosmo.cosmopars["h"]
            kh = k * h_change
        else:
            kh = k
        return kh

    def kmu_alc_pac(self, z, k, mu):
        """Function rescaling k and mu with the Alcock-Paczynski effect

        Parameters
        ----------
        z     : numpy.ndarray, float
                redshift
        k     : numpy.ndarray, float
                wavevector
        mu    : numpy.ndarray, float
                cosine of angle between line of sight and the wavevector

        Returns
        -------
        numpy.ndarray, float

        Note
        -----
        Implements the following equation:

        .. math::
            k^{obs} = k\\, \\sqrt{\\left(q_\\| \\mu \\right)^2 + \\left(1-\\mu^2\\right)q_\\perp^2}

            \\mu^{obs} = \\mu\\,q_\\|\\, \\sqrt{\\left(q_\\| \\mu \\right)^2 + \\left(1-\\mu^2\\right)q_\\perp^2}^{-1}

        """

        if not self.APbool:
            return k, mu
        elif self.APbool:
            sum = self.kpar(z, k, mu) ** 2 + self.kper(z, k, mu) ** 2
            kap = np.sqrt(sum)
            muap = self.kpar(z, k, mu) / kap
            return kap, muap

    def spec_err_z(self, z, k, mu):
        """Function to compute the scale dependant suppression of the observed power spectrum due to the spectroscopic redshift error

        Parameters
        ----------
        z  : float, numpy.ndarray
             The redshifts of interest.
        k  : float, numpy.ndarray
             wavenumbers at which to compute the power spectrum suppression.
        mu : float, numpy.ndarray
             cosine of the angel between the line of sight and the wavevector.

        Returns
        -------
        float, numpy.ndarray
            Suppression of the observed power spectrum due to the error on spectroscopic redshift determination.

        Note
        -----
        Implements the following equation:

        .. math::
            \\mathrm{Err} = \\exp\\left[-\\sigma^2_\\|\\, k^2\\, \\mu^2 -\\sigma_\\perp^2 \\,k^2\\,\\left(1- \\mu^2\\right)\\right].

        """
        if self.dz_type == "constant":
            spec_dz_err = self.dz_err
        elif self.dz_type == "z-dependent":
            spec_dz_err = self.dz_err * (1 + z)
        err = spec_dz_err * (1 / self.cosmo.Hubble(z)) * self.kpar(z, k, mu)
        return np.exp(-(1 / 2) * err**2)

    def BAO_term(self, z):
        """Calculates the BAO term. This is the rescaling of the Fourier volume by the  AP-effect

        Parameters
        ----------
        z     : float, numpy.ndarray
                The redshifts of interest

        Returns
        -------
        float, numpy.ndarray
            Value of BAO term at redshifts z

        Note
        -----
        Implements the following equation:

        .. math::

            \\mathrm{BAO} = q_\\perp^2\\,q_\\|

        """
        if not self.APbool:
            bao = 1
        else:
            bao = 1 / (self.qperpendicular(z) ** 2 * self.qparallel(z))

        return bao

    def bterm_fid(self, z, k=None, bias_sample="g"):
        """Calculate the fiducial bias term for galaxies or intensity mapping.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift value(s) at which to evaluate the bias term.
        k : float or numpy.ndarray, optional
            Wavenumber(s) at which to evaluate the bias term.
        bias_sample : str, optional
            Type of bias to compute. Must be either:
            - 'g' for galaxy clustering bias (default)
            - 'I' for intensity mapping bias

        Returns
        -------
        float or numpy.ndarray
            The computed bias term at the specified z and k values.

        Note
        -----
        The total bias is computed as the product of a redshift-dependent term and
        a scale-dependent term: b_total = b(z) * b(k)

        """
        if bias_sample == "g":
            if bias_sample != self.sp_bias_sample:
                raise ValueError(
                    f"Bias sample {bias_sample} not found. "
                    f"Please use {self.sp_bias_sample} bias sample."
                )
            if self.use_bias_funcs:
                bfunc_of_z = self.nuisance.gcsp_bias_interp()
                bterm_z = bfunc_of_z(z)
            else:
                bterm_z = self.nuisance.vectorized_gcsp_bias_at_z(z)
        elif bias_sample == "I":
            if bias_sample != self.IM_bias_sample:
                raise ValueError(
                    f"Bias sample {bias_sample} not found. "
                    f"Please use {self.IM_bias_sample} bias sample."
                )
            bterm_z = self.nuisance.IM_bias_at_z(z)
        bterm_k = self.nuisance.gcsp_bias_kscale(k)
        bterm_zk = bterm_z * bterm_k
        return bterm_zk

    def kaiserTerm(self, z, k, mu, b_i=None, just_rsd=False, bias_sample="g"):
        """
        Computes the Kaiser redshift-space distortion term.

        Parameters
        ----------
            z           : float, numpy.ndarray
                          Redshifts of interest
            k           : float, numpy.ndarray
                          Wave numbers at which to calculate the linear RSD
            mu          : float, numpy.ndarray
                          cosine of angles between line of sight and the wavevector.
            b_i         : float, numpy.ndarray, optional
                          galaxy bias at Redshifts z
            just_rsd    : bool, optional
                          If True, returns only the RSD term. Otherwise, computes the full Kaiser term. Defaults to False.
            bias_sample : str, optional
                          Bias term to use from self.bterm_fid(). Possible values: 'g': galaxies, 'I': intensity mapping. Defaults to 'g'.

        Returns
        -------
            The computed Kaiser term for redshift space distortions.

        Note
        -----
        Implements the following equation:

        .. math::

            \\mathrm{RSD} = \\left(b_i+f\\,\\mu^2\\right)

        """
        bterm = b_i  # get bs8 as an external parameter, unless it is none, then get it from cosmo
        if b_i is None:
            try:
                bterm = self.bterm_fid(z, k=k, bias_sample=bias_sample)
            except KeyError as ke:
                print(
                    " The key {} is not in dictionary. Check observables and parameters being used".format(
                        ke
                    )
                )
                raise ke
        if self.s8terms:
            fterm = self.cosmo.fsigma8_of_z(z, k, tracer=self.tracer)
        else:
            fterm = self.cosmo.f_growthrate(z, k, tracer=self.tracer)

        if not just_rsd:
            kaiser = bterm + fterm * mu**2
        elif just_rsd:
            kaiser = 1 + (fterm / bterm) * mu**2

        return kaiser

    def FingersOfGod(self, z, k, mu, mode="Lorentz"):
        """
        Calculates the Fingers of God effect in redshift-space power spectra.

        Parameters
        ----------
        z    : float, numpy.ndarray
               The redshifts values of interest.
        k    : float
               The wavenumber for which the suppression should be computed
        mu   : float, numpy.ndarray
               The cosine of angle between the wavevector and the line-of-sight direction.
        mode : str, optional
               A string parameter indicating the model to use. Defaults to 'Lorentz'.

        Returns
        -------
        float, numpy.ndarray
            The calculated FoG term, which is 1 if either FoG_switch is False or linear_switch is True.
            Otherwise, it depends on the specified mode.

        Note
        -----
        If mode is "Lorentz" this implements following equation

        ..math::

            \\mathrm{FoG} = \\frac{1}{1+\\left[f\\,\\sigma_p\\,\\mu^2\\right]^2}

        """
        if (self.FoG_switch is False) or (self.linear_switch):
            fog = 1
        elif mode == "Lorentz":
            fog = 1 / (1 + (k * mu * self.sigmapNL(z)) ** 2)
        else:
            print("FoG mode not implemented")
            fog = 1
        return fog

    def sigmapNL(self, zz):
        """Function to calculate the variance of the velocity dispersion

        Parameters
        ----------
            zz : float
                 The redshift value at which to calculate the variance.
        Returns
        -------
        float
            Calculates the variance of the pairwise velocity dispersion. Enters into the FOG effect.

        """
        if self.linear_switch:
            sp = 0
        else:
            sp = np.sqrt(self.P_ThetaTheta_Moments(zz, 2))
            if self.vary_sigmap:
                sp *= self.nuisance.vectorized_gcsp_rescale_sigmapv_at_z(zz, sigma_key="sigmap")
        return sp

    def sigmavNL(self, zz, mu):
        """Function to calculate the variance of the displacement field

        Parameters
        ----------
            zz : float
                 The redshift value at which to calculate the variance.

        Returns
        -------
        float
            Calculates the variance of the displacement field. Enters into the dewiggling weight to obtain the mildly nonlinear power spectrum

        """
        if self.linear_switch:
            sv = 0
        else:
            f0 = self.P_ThetaTheta_Moments(zz, 0)
            f1 = self.P_ThetaTheta_Moments(zz, 1)
            f2 = self.P_ThetaTheta_Moments(zz, 2)
            sv = np.sqrt(f0 + 2 * mu**2 * f1 + mu**2 * f2)
            if self.vary_sigmav:
                sv *= self.nuisance.vectorized_gcsp_rescale_sigmapv_at_z(zz, sigma_key="sigmav")
        return sv

    def P_ThetaTheta_Moments(self, zz, moment=0):
        """
        Calculates the angular power spectrum moments of the velocity divergence field, also known as the Theta field.

        Parameters
        ----------
        zz     : float
                 The redshift value at which to calculate the power spectrum.
        moment : int
                 An integer indicating the order of the moment to calculate. Default is 0.

        Returns
        -------
        float
            The power spectrum moment of the velocity divergence field.
        """
        # TODO: can be optimized by returning interpolating function in z and
        # doing it one time only
        if self.fix_cosmo_nl_terms:
            cosmoF = self.fiducialcosmo
        else:
            cosmoF = self.cosmo

        def f_mom(k):
            return cosmoF.f_growthrate(zz, k) ** moment

        ff = f_mom(self.k_grid).flatten()
        pp = cosmoF.matpow(zz, self.k_grid).flatten()
        integrand = pp * ff
        Int = integrate.trapezoid(integrand, x=self.k_grid)
        ptt = (1 / (6 * np.pi**2)) * Int
        return ptt

    def normalized_pdd(self, z, k):
        """This function normalizes the power spectrum to have a variance smoothed over 8 Mpc/h of 1. This is to cancel out possible terms with :math:`\\sigma_8` in the RSD.
        Parameters
        ----------
        z : float, numpy.ndarray
           The redshift at which to compute the normalized power spectrum
        k : float, numpy.ndarray
            Wavenumber at which to compute the normalized power spectrum

        Returns
        -------
        float, numpy.ndarray
            The Normalized power spectrum

        Note
        -----
        This is not really a normalisation if there is no :math:`\\sigma_8` terms inside of the RSD (Kaiserterm). It is then canceled out automatically

        """
        s8_denominator = 1
        if self.s8terms:
            s8_denominator = self.cosmo.sigma8_of_z(z, tracer=self.tracer) ** 2

        p_dd = self.cosmo.matpow(z, k, tracer=self.tracer)  # P_{delta,delta}
        self.p_dd = p_dd / s8_denominator
        return self.p_dd

    def normalized_pnw(self, z, k):
        """
        This function normalizes the power spectrum with the BAO wiggles subtracted from to have a variance smoothed over 8 Mpc/h of roughly 1. This is to cancel out possible terms with :math:`\\sigma_8` in the RSD.
        Parameters
        ----------
        z : float, numpy.ndarray
           The redshift at which to compute the normalized 'no-wiggle' power spectrum
        k : float, numpy.ndarray
            Wavenumber at which to compute the normalized 'no-wiggle' power spectrum

        Returns
        -------
        float, numpy.ndarray
            The Normalized 'no-wiggle' power spectrum

        Note
        -----
        This is not really a normalisation if there is no :math:`\\sigma_8` terms inside of the RSD (Kaiserterm). It is then canceled out automatically
        """
        s8_denominator = 1
        if self.s8terms:
            s8_denominator = self.cosmo.sigma8_of_z(z, tracer=self.tracer) ** 2

        p_nw = self.cosmo.nonwiggle_pow(z, k, tracer=self.tracer)  # P_{delta,delta}
        self.p_nw = p_nw / s8_denominator
        return self.p_nw

    def dewiggled_pdd(self, z, k, mu):
        """
        This function calculates the normalized dewiggled power spectrum.

        Parameters
        ----------
        z  : float, numpy.ndarray
             The redshifts values of interest.
        k  : float
             The wavenumber for which the power spectrum should be computed
        mu : float, numpy.ndarray
             The cosine of angle between the wavevector and the line-of-sight direction.

        Returns
        -------
        float, numpy.ndarray
            The the mildly non-linear (dewiggled) power spectrum.
        Note
        ----
            If the config asks for only linear spectrum this just returns the power spectrum normalized with either 1 or 1/sigma8^2
        """

        if self.linear_switch:
            gmudamping = 0
        else:
            gmudamping = self.sigmavNL(z, mu) ** 2

        self.p_dd = self.normalized_pdd(z, k)
        self.p_dd_NW = self.normalized_pnw(z, k)
        self.p_dd_DW = self.p_dd * np.exp(-gmudamping * k**2) + self.p_dd_NW * (
            1 - np.exp(-gmudamping * k**2)
        )
        return self.p_dd_DW

    def observed_Pgg(self, z, k, mu, b_i=None):
        """
        This function calculates the observed galaxy power spectrum.

        Parameters
        ----------
        z   : float, numpy.ndarray
              The redshifts values of interest.
        k   : float
              The wavenumber for which the power spectrum should be computed
        mu  : float, numpy.ndarray
              The cosine of angle between the wavevector and the line-of-sight direction.
        b_i : float, numpy.ndarray, optional
              Redshift dependant values of the galaxy bias

        Returns
        -------
        float, numpy.ndarray
            The observed galaxy power spectrum

        Note
        -----
        In presence of all modeling terms, this function implements the following equation:

        .. math::

            P^{obs}_{gg} = q_\\perp^2 \\, q_\\| \\, \\mathrm{RSD}^2 \\, \\mathrm{FoG}\\, \\frac{P_{dw}^{obs}}{\\sigma_8^2} \\, \\mathrm{Err} + P_{shot}

        """
        if self.feed_lvl > 1:
            print("")
        if self.feed_lvl > 1:
            print("    Computing Pgg for {}".format(self.observables))
        tstart = time()

        if self.kh_rescaling_beforespecerr_bug:
            # In this case the h-bug is only applied before computing the resolution suppression
            # This changes the scale off suppression as well.
            # Still the additional rescaling is unphysical
            k = self.k_units_change(k)
            error_z = self.spec_err_z(z, k, mu)
        else:
            # In this case the h-bug is only applied after computing the resolution suppression
            # This fixes the scale of suppression but still the additional rescaling is unphysical
            error_z = self.spec_err_z(z, k, mu)
            k = self.k_units_change(k)
        k, mu = self.kmu_alc_pac(z, k, mu)

        baoterm = self.BAO_term(z)
        kaiser = self.kaiserTerm(z, k, mu, b_i, bias_sample="g")

        extra_shotnoise = self.extraPshot
        lorentzFoG = self.FingersOfGod(z, k, mu, mode="Lorentz")
        p_dd_DW = self.dewiggled_pdd(z, k, mu)

        pgg_obs = baoterm * (kaiser**2) * p_dd_DW * lorentzFoG * (error_z**2) + extra_shotnoise

        tend = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="observed P_gg computation took: ",
            time_ini=tstart,
            time_fin=tend,
            instance=self,
        )
        return pgg_obs

    def lnpobs_gg(self, z, k, mu, b_i=None):
        """This function calculates the natural logarithm of the observed galaxy power spectrum.

        Parameters
        ----------
        z   : float, numpy.ndarray
              The redshifts values of interest.
        k   : float
              The wavenumber for which the power spectrum should be computed
        mu  : float, numpy.ndarray
              The cosine of angle between the wavevector and the line-of-sight direction.
        b_i : float, numpy.ndarray, optional
              Redshift dependant values of the galaxy bias

        Returns
        -------
        float, numpy.ndarray
            The observed galaxy power spectrums natural logarithm
        """
        pobs = self.observed_Pgg(z, k, mu, b_i=b_i)
        return np.log(pobs)

    def set_IM_bias_specs(self):
        """Updates the IM bias choices"""
        self.IM_bias_model = self.specs["IM_bias_model"]
        self.IM_bias_root = self.specs["IM_bias_root"]
        self.IM_bias_sample = self.specs["IM_bias_sample"]
        self.IM_bias_of_z = self.nuisance.IM_bias_at_z
        self.IM_z_bin_mids = self.nuisance.IM_zbins_mids

    def set_IM_specs(self):
        self.Dd = self.config.specs["D_dish"]  # Dish diameter in m
        self.lambda_21 = 21 / 100  # 21cm in m
        self.fsky_IM = self.config.specs["fsky_IM"]  # sky fraction for IM
        self.t_tot = self.config.specs["time_tot"] * 3600  # * 3600s -> in s
        self.N_d = self.config.specs["N_dish"]
        # self.cosmo.c is in km/s
        # HZ, for MHz: MHz /1e6
        self.f_21 = (self.cosmo.c * 1000) / self.lambda_21

    def Omega_HI(self, z):
        o = 4 * np.power((1 + z), 0.6) * 1e-4
        return o

    def Temperature(self, z, fixed_cosmo=True):
        """obtaining the temperature (T^2(z)) for the Power Spectrum (PHI(z))"""
        if fixed_cosmo:
            cocosmo = self.fiducialcosmo
        else:
            cocosmo = self.cosmo
        h = cocosmo.cosmopars["h"]
        H0 = cocosmo.Hubble(0.0)
        temp = 189 * h * (1 + z) ** 2 * (H0 / cocosmo.Hubble(z)) * self.Omega_HI(z)
        # temperature in mK
        return temp

    def theta_b(self, zz):
        tt = 1.22 * self.lambda_21 * (1 + zz) / self.Dd
        return tt

    def alpha_SD(self):
        return 1 / self.N_d

    def beta_SD(self, z, k, mu):
        tol = 1.0e-12
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        expo = k**2 * (1 - mu**2) * self.fiducialcosmo.comoving(z) ** 2 * self.theta_b(z) ** 2
        bet = np.exp(-expo / (16.0 * np.log(2.0)))
        bet[np.abs(bet) < tol] = tol
        return bet

    def observed_P_ij(self, z, k, mu, bsi_z=None, bsj_z=None, si="I", sj="g"):
        error_z = self.spec_err_z(z, k, mu)
        beam_damping_term_si = self.beta_SD(z, k, mu) if si == "I" else 1
        beam_damping_term_sj = self.beta_SD(z, k, mu) if sj == "I" else 1
        k = self.k_units_change(
            k
        )  # h-bug set to False by default, leaving here for cross-check of old cases
        k, mu = self.kmu_alc_pac(z, k, mu)
        # if self.bias_samples is not None:
        #    si = self.bias_samples[0]
        #    sj = self.bias_samples[1]
        baoterm = self.BAO_term(z)
        kaiser_bsi = self.kaiserTerm(z, k, mu, bsi_z, bias_sample=si)
        kaiser_bsj = self.kaiserTerm(z, k, mu, bsj_z, bias_sample=sj)
        T_HI = self.Temperature(z)
        extra_shotnoise = 0.0  # Set to identically zero for the moment, otherwise self.extraPshot
        lorentzFoG = self.FingersOfGod(z, k, mu, mode="Lorentz")
        p_dd_DW = self.dewiggled_pdd(z, k, mu)
        extra_shotnoise_si = np.sqrt(extra_shotnoise) if si == "g" else 0
        extra_shotnoise_sj = np.sqrt(extra_shotnoise) if sj == "g" else 0
        error_z_si = error_z if si == "g" else 1
        error_z_sj = error_z if sj == "g" else 1
        temp_HI_si = T_HI if si == "I" else 1
        temp_HI_sj = T_HI if sj == "I" else 1

        factors_si = (
            kaiser_bsi * beam_damping_term_si * error_z_si * temp_HI_si + extra_shotnoise_si
        )
        factors_sj = (
            kaiser_bsj * beam_damping_term_sj * error_z_sj * temp_HI_sj + extra_shotnoise_sj
        )

        p_obs = baoterm * lorentzFoG * p_dd_DW * factors_si * factors_sj

        return p_obs

    def lnpobs_ij(self, z, k, mu, bsi_z=None, bsj_z=None, si="I", sj="g"):
        pobs = self.observed_P_ij(z, k, mu, bsi_z=bsi_z, bsj_z=bsj_z, si=si, sj=sj)
        return np.log(pobs)
