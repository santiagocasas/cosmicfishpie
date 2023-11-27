# -*- coding: utf-8 -*-
"""CLS

This module contains cls calculations (only LSS atm).

"""
import copy
from time import time

import numpy as np

import cosmicfishpie.fishermatrix.config as cfg
import cosmicfishpie.LSSsurvey.spectro_obs as spec_obs
from cosmicfishpie.fishermatrix.derivatives import derivatives
from cosmicfishpie.utilities.utils import printing as upt


class SpectroCov:
    def __init__(self, fiducialpars, fiducial_specobs=None, bias_samples=["g", "g"]):
        """
        Initializes an object with specified fiducial parameters and computes
        various power spectra
        (IM, XC, and gg) using those parameters depending on which observables
        are present.

        Parameters
        ----------
        fiducialpars : dict
                       A dictionary containing the cosmological parameters of the fiducial/reference cosmology

        fiducial_specobs : cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalSpectro, cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalIM, optional
                           An optional fiducial spectroscopic observation.

        bias_samples : list
                       A list of two strings specifying if galaxy clustering, intensity mapping or corss correlation power spectrum should be computed. Use "g" for galaxy and "I" for intensity mapping. (default ['g', 'g'])

        Attributes
        ----------
        pk_obs            : cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalSpectro, cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalIM
                            Fiducial instance of the oberservable of the spectroscopic probe. Either Galaxy Clustering, Intensity  mapping or cross correlation.
        pk_obs_gg         : cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalSpectro, cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalIM
                            Fiducial instance of the galaxy clustering autocorrelation oberservable of the spectroscopic probe if corss correlation is asked for.
        pk_obs_II         : cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalSpectro, cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalIM
                            Fiducial instance of the intensity mapping autocorrelation oberservable of the spectroscopic probe if corss correlation is asked for.
        area_survey       : float
                            Size of the survey sky coverage in square arc minutes
        dnz               : float, list
                            Galaxies per sqauare arc minute per redshift bin
        z_bins            : list
                            Redshift bin edges
        z_bin_mids        : list
                            Redshift bin centers
        dz_bins           : list
                            Redshift sizes
        global_z_bin_mids : list
                            Redshift bin centers
        global_z_bins     : list
                            Redshift bin edges
        """
        # initializing the class only with fiducial parameters
        # if fiducial_specobs is None:
        if "IM" in cfg.obs and "GCsp" in cfg.obs:
            bias_samples = ["I", "g"]
            print("Entering Cov cross XC IM,g term")
            self.pk_obs = spec_obs.ComputeGalIM(
                fiducialpars, fiducialpars, bias_samples=bias_samples
            )
            self.pk_obs_gg = spec_obs.ComputeGalIM(
                fiducialpars, fiducialpars, bias_samples=["g", "g"]
            )
            self.pk_obs_II = spec_obs.ComputeGalIM(
                fiducialpars, fiducialpars, bias_samples=["I", "I"]
            )
        elif "IM" in cfg.obs and "I" in bias_samples:
            bias_samples = ["I", "I"]
            print("Entering Cov IM term")
            self.pk_obs = spec_obs.ComputeGalIM(
                fiducialpars, fiducialpars, bias_samples=bias_samples
            )
        elif "GCsp" in cfg.obs and "g" in bias_samples:
            bias_samples = ["g", "g"]
            print("Entering Cov gg term")
            self.pk_obs = spec_obs.ComputeGalSpectro(
                fiducialpars, fiducialpars, bias_samples=bias_samples
            )
        else:
            self.pk_obs = fiducial_specobs

        self.area_survey = cfg.specs["area_survey"]
        if "GCsp" in self.pk_obs.observables:
            self.dnz = self.pk_obs.nuisance.gcsp_dndz()
            self.z_bins = self.pk_obs.nuisance.gcsp_zbins()
            self.z_bin_mids = self.pk_obs.nuisance.gcsp_zbins_mids()
            self.dz_bins = np.diff(self.z_bins)
            self.global_z_bin_mids = self.z_bin_mids
            self.global_z_bins = self.z_bins
        if "IM" in self.pk_obs.observables:
            self.IM_z_bins = self.pk_obs.nuisance.IM_zbins()
            self.IM_z_bin_mids = self.pk_obs.nuisance.IM_zbins_mids()
            self.Tsys_interp = self.pk_obs.nuisance.IM_THI_noise()
            self.global_z_bin_mids = self.IM_z_bin_mids
            self.global_z_bins = self.IM_z_bins
        # Choose longest zbins array to loop in Fisher matrix
        if "GCsp" in self.pk_obs.observables and "IM" in self.pk_obs.observables:
            if len(self.z_bin_mids) >= len(self.IM_z_bin_mids):
                self.global_z_bin_mids = self.z_bin_mids
                self.global_z_bins = self.z_bins
            else:
                self.global_z_bin_mids = self.IM_z_bin_mids
                self.global_z_bins = self.IM_z_bins

    def Tsys_func(self, z):
        """Calculates Tsys in mK

        Parameters
        ----------
        z : float, numpy.ndarray
            Redshift at which to compute Tsys

        Returns
        -------
        float, numpy.ndarray
            Tsys at z in milli Kelvin
        """
        units = 1000  # convert from K to mK
        Tsys_mK = units * self.Tsys_interp(z)
        return Tsys_mK

    def volume_bin(self, zi, zj):
        """Calculates the comoving volume of a spherical shell

        Parameters
        ----------
        zi : float, nunpy.ndarray
             Redshift of the inner sphere
        zj : float, numpy.ndarray
             Redshift of the outer sphere

        Returns
        -------
        float, numpy.ndarray
            Volume of the comoving spherical shell between zj and zi
        """
        rad_to_area = 1 / (4 * np.pi * np.power(180 / np.pi, 2))
        d1 = self.pk_obs.cosmo.angdist(zi)
        d2 = self.pk_obs.cosmo.angdist(zj)
        sphere_vol = (4 * np.pi / 3) * (pow((1 + zj) * d2, 3) - pow((1 + zi) * d1, 3))
        vol = sphere_vol * rad_to_area
        return vol

    def d_volume(self, ibin):
        """Calculates the comoving volume of a redshift bin

        Parameters
        ----------
        i : int
            Index of the survey redshift bin

        Returns
        -------
        float
            Comoving volume of the redshift bin
        """
        return self.volume_bin(self.global_z_bins[ibin], self.global_z_bins[ibin + 1])

    def volume_survey(self, ibin):
        """Calculates the survey volume of a redshift bin

        Parameters
        ----------
        i : int
            Index of the survey redshift bin

        Returns
        -------
        float
            survey volume of the redshift bin
        """
        vol = self.area_survey * self.d_volume(ibin)
        return vol

    def n_density(self, ibin):
        """ calculate the comoving number density of the probe

        Parameters
        ----------
        i : int
            Index of the survey redshift bin

        Returns
        -------
        float
            comoving number density of the probe
        """
        ndens = self.dnz[ibin] * self.dz_bins[ibin] / self.d_volume(ibin)
        return ndens

    def veff(self, ibin, k, mu):
        """ calculate the effective volume entering the covariance of the galaxy clustering probe

        Parameters
        ----------
        i  : int
             Index of the survey redshift bin
        k  : float, numpy.ndarray
             wave number at which the effective volume should be calculated
        mu : float, numpy.ndarray
             The cosine of angle between the wavevector and the line-of-sight direction.

        Returns
        float, numpy.ndarray
            The effective volume for a given wavenumber, angle and redshift
        """
        zi = self.z_bin_mids[ibin]
        npobs = self.n_density(ibin) * self.pk_obs.observed_Pgg(zi, k, mu)
        prefactor = 1 / (8 * (np.pi**2))
        covterm = prefactor * (npobs / (1 + npobs)) ** 2
        return covterm

    def cov(self, ibin, k, mu):
        """Function to calculate the covariance the galaxy clustering probe

        Parameters
        ----------
        ibin : int
               Index of the redshift bin for which the covarance is to be computed
        k    : float, numpy.ndarray
               Wavenumber
        mu   : float, numpy.ndarray
               The cosine of angle between the wavevector and the line-of-sight direction.

        Returns
        -------
        float, numpy.ndarray
            Covariance of the Galaxy clustering probe
        """
        zmid = self.z_bin_mids[ibin]
        veffS = self.veff(ibin, k, mu) * self.volume_survey(ibin)
        pobs = self.pk_obs.observed_Pgg(zmid, k, mu)
        prefactor = 2 * (2 * np.pi) ** 3
        cov = (prefactor / veffS) * (pobs) ** 2 * (1 / k) ** 3
        return cov

    def P_noise_21(self, z, k, mu, temp_dim=True, beam_term=False):
        """Compute the shotnoise of the 21 centimeter intensity mapping probe

        Parameters
        ----------
        z        : float, numpy.ndarray
                   Redshift at which the noise is to be computed
        k        : float, numpy.ndarray
                   Wavenumber
        mu       : float, numpy.ndarray
                   The cosine of angle between the wavevector and the line-of-sight direction.
        temp_dim : bool
                   If true the Temperature terms is in units of Kelvin^2
        beam_term: bool
                   If true will add the beam term to the computation of the power spectrum

        Returns
        -------
        float, numpy.ndarray
            Additional shotnoise of the 21 cm intensity mapping
        """
        if not temp_dim:
            temp = self.pk_obs.Temperature(z)
        elif temp_dim:
            temp = 1
        pref = (2 * np.pi * self.pk_obs.fsky_IM) / (self.pk_obs.f_21 * self.pk_obs.t_tot)
        cosmo = ((1 + z) ** 2 * self.pk_obs.cosmo.comoving(z) ** 2) / self.pk_obs.cosmo.Hubble(z)
        T_term = (self.Tsys_func(z) / temp) ** 2  # in K
        alpha = self.pk_obs.alpha_SD()
        if beam_term:
            beta = self.pk_obs.beta_SD(z, k, mu)
        else:
            beta = np.ones_like(k)
        noise = pref * cosmo * T_term * (alpha / beta**2)
        return noise

    def veff_21cm(self, ibin, k, mu):
        """ calculate the effective volume entering the covariance of the line intensity mapping probe

        Parameters
        ----------
        i  : int
             Index of the survey redshift bin
        k  : float, numpy.ndarray
             wave number at which the effective volume should be calculated
        mu : float, numpy.ndarray
             The cosine of angle between the wavevector and the line-of-sight direction.

        Returns
        float, numpy.ndarray
            The effective volume for a given wavenumber, angle and redshift
        """
        zi = self.IM_z_bin_mids[ibin]
        pobs = self.pk_obs.observed_P_HI(zi, k, mu)
        pnoisy = pobs + self.P_noise_21(zi, k, mu)
        prefactor = 1 / (8 * (np.pi**2))
        covterm = prefactor * (pobs / pnoisy) ** 2
        return covterm

    def veff_XC(self, ibin, k, mu):
        """ calculate the effective volume entering the covariance of the cross correlation of galaxy clustering and intensity mapping

        Parameters
        ----------
        i  : int
             Index of the survey redshift bin
        k  : float, numpy.ndarray
             wave number at which the effective volume should be calculated
        mu : float, numpy.ndarray
             The cosine of angle between the wavevector and the line-of-sight direction.

        Returns
        float, numpy.ndarray
            The effective volume for a given wavenumber, angle and redshift
        """
        print("Entering veff_XC term")
        zi = self.IM_z_bin_mids[ibin]
        # when calling this function, this is the XC spectrum
        pobs_Ig = self.pk_obs.observed_P_HI(zi, k, mu)
        pobs_gg = self.pk_obs_gg.observed_P_HI(zi, k, mu)
        pobs_II = self.pk_obs_II.observed_P_HI(zi, k, mu)
        pnoisy_Ig = pobs_Ig
        pnoisy_II = pobs_II + self.P_noise_21(zi, k, mu)
        pnoisy_gg = pobs_gg + self.n_density(ibin)
        covterm = pobs_Ig**2 / (pnoisy_gg * pnoisy_II + pnoisy_Ig * pnoisy_Ig)
        prefactor = 1 / (4 * (np.pi**2))
        covterm = prefactor * covterm
        return covterm


class SpectroDerivs:
    def __init__(self, z_array, pk_kmesh, pk_mumesh, fiducial_spectro_obj, bias_samples=["g", "g"]):
        """Main derivative Engine for the Spectroscopic probes.

        Parameters
        ----------
        z_array              : numpy.ndarray
                               List of the redshift bin centers at which the derivatives should be computed at
        pk_kmesh             : numpy.ndarray
                               List of wavenumbers at which the derivatives should be computed at
        pk_mumesh            : numpy.ndarray
                               List of the cosines of angle between the wavevector and the line-of-sight direction.
        fiducial_spectro_obj : cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalSpectro, cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalIM
                               Fiducial instance of the oberservable of the spectroscopic probe. Either Galaxy Clustering, Intensity  mapping or cross correlation.
        bias_samples         : list
                               A list of two strings specifying if galaxy clustering, intensity mapping or corss correlation power spectrum should be computed. Use "g" for galaxy and "I" for intensity mapping. (default ['g', 'g'])

        Attributes
        ----------
        observables                   : list
                                        A list of the observables that the observed power spectrum is computed for
        bias_samples                  : list
                                        A list of two strings specifying if galaxy clustering, intensity mapping or corss correlation power spectrum should be computed. Use "g" for galaxy and "I" for intensity mapping. (default ['g', 'g'])
        fiducial_cosmopars            : dict
                                        A dictionary containing the cosmological parameters of the fiducial/reference cosmology
        fiducial_spectrobiaspars      : dict
                                        A dictionary containing the specifications for the galaxy biases
        fiducial_IMbiaspars           : dict
                                        A dictionary containing the specifications for the intensity mapping biases
        fiducial_PShotpars            : dict
                                        A dictionary containing the values of the additional shot noise per bin
        fiducial_allpars              : dict
                                        A dictionary containing all relevant fiducial parameters to compute the observed power spectrum
        fiducial_spectrononlinearpars : dict
                                        A dictionary containing the fiducial values of the non linear modeling parameters entering FOG and the dewiggling weight per bin
        fiducial_cosmo                : cosmicfishpie.cosmology.cosmology.cosmo_functions
                                        An instance of `cosmo_functions` of the fiducial cosmology.
        z_array                       : numpy.ndarray
                                        List of the redshift bin centers at which the derivatives should be computed at
        cosmology_variations_dict     : dict
                                        A dictionary containing the values for all relevant parameters to compute the observed power spectrum for each varied cosmology
        pk_kmesh                      : numpy.ndarray
                                        List of wavenumbers at which the derivatives should be computed at
        pk_mumesh                     : numpy.ndarray
                                        List of the cosines of angle between the wavevector and the line-of-sight direction.
        freeparams                    : dict
                                        A dictionary with all vaired parameters and their stepsizes
        feed_lvl                      : int
                                        number indicating the verbosity of the output. Higher numbers mean more output

        """
        print("Computing derivatives of Galaxy Clustering Spectro")
        self.observables = fiducial_spectro_obj.observables
        self.bias_samples = bias_samples
        self.fiducial_cosmopars = fiducial_spectro_obj.fiducial_cosmopars
        self.fiducial_spectrobiaspars = fiducial_spectro_obj.fiducial_spectrobiaspars
        if "IM" in self.observables:
            self.fiducial_IMbiaspars = fiducial_spectro_obj.fiducial_IMbiaspars
        self.fiducial_PShotpars = fiducial_spectro_obj.PShotpars
        self.fiducial_allpars = fiducial_spectro_obj.fiducial_allpars
        self.fiducial_spectrononlinearpars = fiducial_spectro_obj.fiducial_spectrononlinearpars
        self.fiducial_cosmo = fiducial_spectro_obj.fiducialcosmo
        self.z_array = z_array
        self.cosmology_variations_dict = dict()
        self.pk_kmesh = pk_kmesh
        self.pk_mumesh = pk_mumesh
        self.freeparams = None  # cfg.freeparams
        self.feed_lvl = 1  # cfg.settings['feedback']
        # self.get_obs = memory.cache(self.getobs)

    def initialize_obs(self, allpars):
        cosmopars = dict((k, allpars[k]) for k in self.fiducial_cosmopars)
        spectrobiaspars = dict((k, allpars[k]) for k in self.fiducial_spectrobiaspars)
        PShotpars = dict((k, allpars[k]) for k in self.fiducial_PShotpars)
        spectrononlinearpars = dict((k, allpars[k]) for k in self.fiducial_spectrononlinearpars)

        if "I" in self.bias_samples:
            IMbiaspars = dict((k, allpars[k]) for k in self.fiducial_IMbiaspars)

            self.pobs = spec_obs.ComputeGalIM(
                cosmopars=cosmopars,
                fiducial_cosmopars=self.fiducial_cosmopars,
                spectrobiaspars=spectrobiaspars,
                IMbiaspars=IMbiaspars,
                PShotpars=PShotpars,
                fiducial_cosmo=self.fiducial_cosmo,
                bias_samples=self.bias_samples,
            )
        elif "g" in self.bias_samples:
            self.pobs = spec_obs.ComputeGalSpectro(
                cosmopars=cosmopars,
                fiducial_cosmopars=self.fiducial_cosmopars,
                spectrobiaspars=spectrobiaspars,
                spectrononlinearpars=spectrononlinearpars,
                PShotpars=PShotpars,
                fiducial_cosmo=self.fiducial_cosmo,
                bias_samples=self.bias_samples,
            )
        strdic = str(sorted(cosmopars.items()))
        hh = hash(strdic)
        self.cosmology_variations_dict[hh] = self.pobs.cosmo
        self.cosmology_variations_dict["hash_" + str(hh)] = strdic

    def get_obs(self, allpars):
        """function to obtain the power spectrum of the observable

        Parameters
        ----------
        allpars : dict
                  A dictionary containing all relevant parameters to compute the observed power spectrum

        Returns
        -------
        dict
            A dictionary containing the observed power spectrum on the k and mu grid for all redshift bins
        """
        self.initialize_obs(allpars)
        result_array = dict()
        result_array["z_bins"] = self.z_array
        for ii, zzi in enumerate(self.z_array):
            if "I" in self.bias_samples:
                result_array[ii] = self.pobs.lnpobs_IM(zzi, self.pk_kmesh, self.pk_mumesh)
            elif "g" in self.bias_samples:
                result_array[ii] = self.pobs.lnpobs_gg(zzi, self.pk_kmesh, self.pk_mumesh)
        return result_array

    def exact_derivs(self, par):
        """ Compute the exact log derivative of the Power spectrum with respect to the shotnoise using chain rule

        Parameters
        ----------
        par : str
              String name of the Shotnoise parameter for which the exact derivative should be computed from

        Returns
        -------
        dict
            A dictionary containing the exact log derivative with respect to the shotnoise parameter
        """
        if "Ps" in par:
            deriv = dict()
            for ii, zzi in enumerate(self.z_array):
                pgg_obs = self.pobs.observed_Pgg(zzi, self.pk_kmesh, self.pk_mumesh)
                z_bin_mids = self.pobs.gcsp_z_bin_mids
                kron = self.kronecker_bins(par, z_bin_mids, zzi)
                deriv_i = 1 / pgg_obs
                deriv[ii] = kron * deriv_i
            return deriv
        else:
            return None

    def kronecker_bins(self, par, zmids, zi):
        """function to figure out what bin a parameter corresponds and compares to the redshift passed

        Parameters
        ----------
        par  : str
               String name of a parameter. The name should end with '_i' where i marks the bin it corresponds to
        zmid : numpy.ndarray
               List of the centers for the redshift bin
        zi   : float
               redshift for which we want to find the bin

        Returns
        -------
        int
            returns 1 if passed redshift is in the bin corresponding to the parameter. Returns 0 elsewise
        """
        ii = np.where(np.isclose(zmids, zi))
        ii = ii[0][0] + 1
        pi = par.split("_")
        pi = int(pi[-1])
        if ii == pi:
            kron_delta = 1
        else:
            kron_delta = 0
        return kron_delta

    def compute_derivs(self, freeparams=dict()):
        """ Calls the common derivative engine to compute the derivatives of the observed power spectrum

        Parameters
        ----------
        freeparam : dict, optional
                    A dictionary containing the names and stepsizes for all parameters you want to vary. Will default to the global free params if not passed.

        Returns
        -------
        dictionary containg lists of derivatives of the observed power spectrum for each redshift bin and parameter
        """
        derivs = dict()
        if freeparams != dict():
            self.freeparams = freeparams
        compute_derivs = True
        if compute_derivs:
            tder1 = time()
            print(">> Computing Derivs >>")
            deriv_engine = derivatives(
                observable=self.get_obs,
                fiducial=self.fiducial_allpars,
                special_deriv_function=self.exact_derivs,
                freeparams=self.freeparams,
            )
            derivs = deriv_engine.result
            tder2 = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="-->> Derivatives computed in ",
                time_ini=tder1,
                time_fin=tder2,
                instance=self,
            )

        self.derivs = derivs
        return self.derivs

    def dlnpobs_dp(self, zi, k, mu, par):
        """ This is a deprecated function! It was used to compute the compute the derivatives of the power spectrum internaly in the cosmicfishpie.LSSsurvey.spectro_cov.SpectroDerivs . Use now the common derivative engine at cosmicfishpie.fishermatrix.derivatives.derivatives

        Parameters
        ----------
        zi  : float, numpy.ndarray
              The redshifts values of interest.
        k   : float
              The wavenumber for which the power spectrum should be computed
        mu  : float, numpy.ndarray
              The cosine of angle between the wavevector and the line-of-sight direction.
        par : str
              Name of the parameter with regards to which the derivative should be computed for

        Returns
        -------
        float, numpy.ndarray
            Derivative of the observed power spectrum with regard to the passed parameter
        """
        if par in self.freepars.keys():
            return self.dlnpobs_dcosmop(zi, k, mu, par)
        elif par in self.biaspars.keys():
            return self.dlnpobs_dnuisp(zi, k, mu, par)
        elif par in self.Pspars.keys():
            return self.dlnpobs_dnuisp(zi, k, mu, par)
        else:
            print("WARNING: parameter not contained in specgal instance definition")
            return np.zeros_like(k)

    def dlnpobs_dcosmop(self, zi, k, mu, par):
        """ This is a deprecated function! It was used to compute the compute the derivatives of the power spectrum with respect to the cosmological parameters internaly in the cosmicfishpie.LSSsurvey.spectro_cov.SpectroDerivs . Use now the common derivative engine at cosmicfishpie.fishermatrix.derivatives.derivatives

        Parameters
        ----------
        zi  : float, numpy.ndarray
              The redshifts values of interest.
        k   : float
              The wavenumber for which the power spectrum should be computed
        mu  : float, numpy.ndarray
              The cosine of angle between the wavevector and the line-of-sight direction.
        par : str
              Name of the parameter with regards to which the derivative should be computed for

        Returns
        -------
        float, numpy.ndarray
            Derivative of the observed power spectrum with regard to the passed parameter
        """
        if self.fiducialpars[par] != 0.0:
            stepsize = self.fiducialpars[par] * self.freepars[par]
        else:
            stepsize = self.freepars[par]

        # Doing forward step
        fwd = copy.deepcopy(self.fiducialpars)
        fwd[par] = fwd[par] + stepsize
        galspec = self.get_obs(fwd)
        pgg_pl = galspec.lnpobs(zi, k, mu)
        # Doing backward step
        bwd = copy.deepcopy(self.fiducialpars)
        bwd[par] = bwd[par] - stepsize
        galspec = self.get_obs(bwd)
        pgg_mn = galspec.lnpobs(zi, k, mu)

        deriv = (pgg_pl - pgg_mn) / (2 * stepsize)
        return deriv

    def dlnpobs_dnuisp(self, zi, k, mu, par):
        """ This is a deprecated function! It was used to compute the compute the derivatives of the power spectrum with respect to nuicance parameters in the cosmicfishpie.LSSsurvey.spectro_cov.SpectroDerivs . Use now the common derivative engine at cosmicfishpie.fishermatrix.derivatives.derivatives

        Parameters
        ----------
        zi  : float, numpy.ndarray
              The redshifts values of interest.
        k   : float
              The wavenumber for which the power spectrum should be computed
        mu  : float, numpy.ndarray
              The cosine of angle between the wavevector and the line-of-sight direction.
        par : str
              Name of the parameter with regards to which the derivative should be computed for

        Returns
        -------
        float, numpy.ndarray
            Derivative of the observed power spectrum with regard to the passed parameter
        """
        galspec = self.galspec_fiducial
        if "lnb" in par:
            bterm = galspec.bterm_fid(zi, bias_sample="g")
            lnb_pl = np.power(bterm, 1 + self.eps_nuis)
            lnb_mn = np.power(bterm, 1 - self.eps_nuis)
            lnb = np.log(bterm)
            lnpobs_pl = galspec.lnpobs(zi, k, mu, b_i=lnb_pl)
            lnpobs_mn = galspec.lnpobs(zi, k, mu, b_i=lnb_mn)
            deriv = (lnpobs_pl - lnpobs_mn) / (2 * self.eps_nuis * lnb)
        elif "Ps" in par:
            pobs = galspec.observed_Pgg(zi, k, mu)
            deriv = 1 / pobs
        else:
            print("Error: Param name not supported in nuisance parameters")
            deriv = 0
        # get index in bin
        ii = np.where(np.isclose(self.z_bin_mids, zi))
        ii = ii[0][0] + 1
        pi = par.split("_")
        pi = int(pi[-1])
        if ii == pi:
            kron_delta = 1
        else:
            kron_delta = 0
        deriv = kron_delta * deriv
        return deriv
