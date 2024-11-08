# -*- coding: utf-8 -*-
"""MAIN

This is the main engine of CosmicFish.

"""
import os
import sys
from copy import copy, deepcopy
from itertools import product
from time import time

import numpy as np
import pandas as pd
from scipy.integrate import simpson

from cosmicfishpie.analysis import fisher_matrix as fm
from cosmicfishpie.CMBsurvey import CMB_cov
from cosmicfishpie.configs import config as cfg
from cosmicfishpie.LSSsurvey import photo_cov, photo_obs, spectro_cov, spectro_obs
from cosmicfishpie.utilities.utils import filesystem as ufs
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upt
from cosmicfishpie.version import VERSION


def ray_session(num_cpus=4, restart=True, shutdown=False):
    import ray

    if shutdown:
        if ray.is_initialized():
            ray.shutdown()
            return None
    elif restart:
        if ray.is_initialized():
            print("ray is shutting down...")
            ray.shutdown()
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
        return None


class FisherMatrix:
    # update version at every release
    cf_version = f"CosmicFish_v{VERSION}"

    def __init__(
        self,
        options=dict(),
        specifications=dict(),
        observables=None,
        freepars=None,
        extfiles=None,
        fiducialpars=None,
        photobiaspars=None,
        photopars=None,
        spectrononlinearpars=None,
        spectrobiaspars=None,
        IApars=None,
        IMbiaspars=None,
        PShotpars=None,
        surveyName="Euclid",
        cosmoModel="w0waCDM",
        latexnames=None,
        parallel=False,
        parallel_cpus=4,
    ):
        """This is the main class of cosmicfishpie that is used to do a Fisher matrix forecast for current and upcoming experiments in cosmology. If you want to compute the Fisher matrix you can pass all specifications and options here.

        Parameters
        ----------
        options              : dict, optional
                               A dictionary that contains the global options for the calculation of the fishermatrix. A list of all possible keys are found in the documentation of cosmicfishpie.configs.config
        specifications       : dict, optional
                               A dictionary containing the survey specifications. Defaults to the specifications in the `.yaml` of the survey specifications
        observables          : list, optional
                               A list of strings for the different observables
        freepars             : dict, optional
                               A dictionary containing all cosmological parameters to be varied and their corresponding rel. step sizes
        extfiles             : dict, optional
                               A dictionary containing the path to the external files as well as how all the names of the files in the folder correspond to the cosmological quantities, units etc.
        fiducialpars         : dict, optional
                               A dictionary containing the fiducial cosmological parameters
        photobiaspars             : dict, optional
                               A dictionary containing the specifications for the galaxy biases of the photometric probe
        photopars            : dict, optional
                               A dictionary containing specifications for the window function's galaxy distribution
        spectrononlinearpars : dict, optional
                               A dictionary containing the values of the non linear modeling parameters of the spectroscopic probe
        spectrobiaspars     : dict, optional
                               A dictionary containing the specifications for the galaxy biases of the spectroscopic probe
        IApars               : dict, optional
                               A dictionary containing the specifications for the galaxy biases of the spectroscopic intensity mapping probe
        surveyName           : str, optional
                               String of the name of the survey for which the forecast is done. Defaults to Euclid with optimistic specifications
        cosmoModel           : str, optional
                               A string of the name of the cosmological model used in the calculations. Defaults to flat "w0waCDM" cosmology
        latexnames           : dict, optional
                               A dictionary that contains the Latex names of the cosmological parameters
        parallel             : bool, optional
                               If True will compute the Fisher matrix using ray parallelization. Defaults to False
        parallel_cpus        : int, optional
                               Number of CPUs that should be used when computing the results using ray parallelization.

        Attributes
        ----------
        fiducialcosmopars : dict
                            A dictionary containing all fiducial values for the cosmological parameters
        photopars         : dict
                            A dictionary containing specifications for the window function's galaxy distribution of the photometric probe
        IApars            : dict
                             A dictionary containing the specifications for the intrinsic alignment effect in cosmic shear of the photometric probe
        biaspars          : dict
                             a dictionary containing the specifications for the galaxy biases of the photometric probe
        Spectrobiaspars   : dict
                             A dictionary containing the specifications for the galaxy biases of the spectroscopic probe
        Spectrononlinpars : dict
                             A dictionary containing the values of the non linear modeling parameters entering FOG and the dewiggling weight per bin for the spectroscopic probe
        IMbiaspars        : dict
                             A dictionary containing the specifications for the line intensity biases of the spectroscopic probe
        PShotpars         : dict
                             A dictionary containing the values of the additional shot noise per bin dictionary containing the values of the additional shot noise per bin for the spectroscopic probe
        observables       : list
                             A list of strings for the different observables
        freeparams        : dict
                             A dictionary containing all names and the corresponding rel. step size for all parameters
        allparams_fidus   : dict
                            A dictionary that contains all fiducial cosmological and nuisance parameters needed to compute the observable of all probes.
        """
        if options["feedback"] > 0:
            print("****************************************************************")
            print("   _____               _     _____     __  ")
            print("  / ___/__  ___ __ _  (_)___/ __(_)__ / /  ")
            print(" / /__/ _ \\(_-</  ' \\/ / __/ _// (_-</ _ \\ ")
            print(" \\___/\\___/___/_/_/_/_/\\__/_/ /_/___/_//_/ ")
            print("")
            print("****************************************************************")
            print(" This is the new Python version of the CosmicFish code.")
            print("****************************************************************")
            sys.stdout.flush()

        cfg.init(
            options=options,
            specifications=specifications,
            observables=observables,
            freepars=freepars,
            extfiles=extfiles,
            fiducialpars=fiducialpars,
            photobiaspars=photobiaspars,
            photopars=photopars,
            IApars=IApars,
            spectrononlinearpars=spectrononlinearpars,
            spectrobiaspars=spectrobiaspars,
            IMbiaspars=IMbiaspars,
            PShotpars=PShotpars,
            surveyName=surveyName,
            cosmoModel=cosmoModel,
            latexnames=latexnames,
        )
        self.settings = deepcopy(cfg.settings)
        self.specs = deepcopy(cfg.specs)
        self.fiducialcosmopars = deepcopy(cfg.fiducialparams)
        self.fiducialparams = self.fiducialcosmopars  ## for compatibility
        self.fiducialcosmo = copy(cfg.fiducialcosmo)
        self.photopars = deepcopy(cfg.photoparams)
        self.photobiaspars = deepcopy(cfg.Photobiasparams)
        self.IApars = deepcopy(cfg.IAparams)
        self.Spectrobiaspars = deepcopy(cfg.Spectrobiasparams)
        self.Spectrobiasparams = self.Spectrobiaspars  ## for compatibility
        self.Spectrononlinpars = deepcopy(cfg.Spectrononlinearparams)
        self.Spectrononlinearparams = self.Spectrononlinpars  ## for compatibility
        self.IMbiaspars = deepcopy(cfg.IMbiasparams)
        self.IMbiasparams = self.IMbiaspars  ## for compatibility
        self.PShotpars = deepcopy(cfg.PShotparams)
        self.PShotparams = self.PShotpars  ## for compatibility
        self.observables = deepcopy(cfg.obs)
        self.obs = self.observables  ## for compatibility
        self.input_type = deepcopy(cfg.input_type)  ## for compatibility
        self.freeparams = deepcopy(cfg.freeparams)
        self.allparams_fidus = {
            **self.fiducialcosmopars,
            **self.photopars,
            **self.photobiaspars,
            **self.IApars,
            **self.Spectrobiaspars,
            **self.Spectrononlinpars,
            **self.IMbiaspars,
            **self.PShotpars,
        }
        self.parallel = parallel
        self.feed_lvl = self.settings["feedback"]
        allpars = {}
        allpars.update(self.fiducialcosmopars)
        allpars.update(self.photobiaspars)
        allpars.update(self.photopars)
        allpars.update(self.IApars)
        allpars.update(self.Spectrobiaspars)
        allpars.update(self.Spectrononlinpars)
        allpars.update(self.IMbiaspars)
        allpars.update(self.PShotpars)
        self.allparams = allpars

        # self.parallel=True
        self.recap_options()
        if self.parallel:
            ray_session(num_cpus=parallel_cpus, restart=True, shutdown=False)

    def compute(self, max_z_bins=None):
        """This function will compute the Fisher information matrix and export using the specified settings.

        Arguments
        ---------
        max_z_bins : int
                     For the spectroscopic probe what the highest redshift bin that should be considered in the computation of the Fisher

        Returns
        -------
        cosmicfishpie.analysis.fisher_matrix
            A instance of fisher_matrix containing the calculated Fisher matrix as well as parameter names, settings, etc
        """
        # This switches between the possible Fisher matrices

        tfishstart = time()
        if "GCph" in self.observables or "WL" in self.observables:
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="----> Computing photo Fisher matrix",
                instance=self,
            )
            self.photo_obs_fid = photo_obs.ComputeCls(
                self.fiducialcosmopars,
                self.photopars,
                self.IApars,
                self.photobiaspars,
                print_info_specs=True,
            )
            self.photo_LSS = photo_cov.PhotoCov(
                self.fiducialcosmopars,
                self.photopars,
                self.IApars,
                self.photobiaspars,
                fiducial_Cls=self.photo_obs_fid,
            )
            noisy_cls, covmat = self.photo_LSS.compute_covmat()
            self.photo_derivs = self.photo_LSS.compute_derivs()
            photoFM = self.photo_LSS_fishermatrix_einsum(
                noisy_cls=noisy_cls, covmat=covmat, derivs=self.photo_derivs
            )
            finalFisher = deepcopy(photoFM)

        elif "GCsp" in self.observables or "IM" in self.observables:
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="----> Computing Pk-spectro Fisher matrix",
                instance=self,
            )
            self.set_pk_settings()
            if "IM" in self.observables and "GCsp" in self.observables:
                self.obs_spectrum = ["I", "g"]
            elif "IM" in self.observables:  # compute  in the case of IM only
                self.obs_spectrum = ["I", "I"]
            elif "GCsp" in self.observables:
                self.obs_spectrum = ["g", "g"]
            #    self.obs_spectrum = ["I", "g"]
            self.pk_obs_fid = spectro_obs.ComputeGalSpectro(
                cosmopars=self.fiducialcosmopars,
                fiducial_cosmopars=self.fiducialcosmopars,
                spectrobiaspars=self.Spectrobiaspars,
                spectrononlinearpars=self.Spectrononlinpars,
                IMbiaspars=self.IMbiaspars,
                PShotpars=self.PShotpars,
            #    bias_samples=self.obs_spectrum,
                configuration=self,
            )
            self.pk_cov = spectro_cov.SpectroCov(
                self.fiducialcosmopars,
                fiducial_specobs=self.pk_obs_fid,
                bias_samples=self.obs_spectrum,
                configuration=self,
            )
            self.zmids = self.pk_cov.inter_z_bin_mids
            nbins = len(self.zmids)
            if max_z_bins is not None and type(max_z_bins) == int:
                cutnbins = max_z_bins
                self.eliminate_zbinned_freepars(cutnbins)
                self.zmids = self.zmids[0:cutnbins]
                nbins = len(self.zmids)
            tini = time()
            self.derivs_dict = dict()
            # if self.parallel == False:
            k_mesh = self.Pk_kmesh
            mu_mesh = self.Pk_mumesh
            self.veff_arr = []
            tvi = time()
            self.fish_z_arr = np.array(self.zmids)
            for zi in self.zmids:
                if "IM" in self.observables and "GCsp" in self.observables:
                    self.veff_arr.append(self.pk_cov.veff_Ig(zi, k_mesh, mu_mesh))
                elif "IM" in self.observables:  # compute  in the case of IM only
                    self.veff_arr.append(self.pk_cov.veff_II(zi, k_mesh, mu_mesh))
                elif "GCsp" in self.observables:
                    self.veff_arr.append(self.pk_cov.veff(zi, k_mesh, mu_mesh))
            tvf = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=2,
                text="+++ Volumes computation computed in: ",
                time_ini=tvi,
                time_fin=tvf,
                instance=self,
            )
            self.derivs_dict = self.compute_binned_derivs(self.fish_z_arr)
            tend = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=2,
                text="+++ Derivatives computation done in: ",
                time_ini=tini,
                time_fin=tend,
                instance=self,
            )
            self.allparams = self.pk_deriv.fiducial_allpars
            pk_Fish = self.pk_LSS_Fisher(nbins=nbins)
            specFM = np.sum(pk_Fish, axis=0)
            finalFisher = deepcopy(specFM)

        elif (
            "CMB_T" in self.observables
            or "CMB_E" in self.observables
            or "CMB_B" in self.observables
        ):
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="----> Computing CMB Fisher matrix",
                instance=self,
            )
            CMB = CMB_cov.CMBCov(self.fiducialcosmopars, print_info_specs=True)
            noisy_cls, covmat = CMB.compute_covmat()
            derivs = CMB.compute_derivs()
            CMB_FM = self.CMB_fishermatrix(noisy_cls=noisy_cls, covmat=covmat, derivs=derivs)
            finalFisher = deepcopy(CMB_FM)
            # return CMB_FM
        else:
            raise AttributeError("Observables list not defined properly")
        tfishend = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
            text="Fisher matrix calculation finished in ",
            time_ini=tfishstart,
            time_fin=tfishend,
        )
        fishMat_object = self.export_fisher(finalFisher, totaltime=tfishend - tfishstart)

        if self.parallel:
            ray_session(restart=False, shutdown=True)

        return fishMat_object

    def set_pk_settings(self):
        """Function to define grids of the internal wavenumber and observation angle

        Note
        ----
        This is not a setter function. Rather the internal grids are constructed from the passed options and then they are stored in the attributes `Pk_kgrid` and `Pk_mugrid` as well as the combined meshes `Pk_kmesh` and `Pk_mumesh`.
        """
        # kmin and kmax GCsp values are always indicated in h/Mpc
        k_units_factor = self.fiducialcosmopars["h"]
        self.kmin_fish = self.specs["kmin_GCsp"] * k_units_factor
        self.kmax_fish = self.specs["kmax_GCsp"] * k_units_factor
        self.Pk_ksamp = 513 * self.settings["accuracy"]
        self.Pk_musamp = 9 * self.settings["accuracy"]
        self.Pk_kgrid = np.linspace(self.kmin_fish, self.kmax_fish, self.Pk_ksamp)
        self.Pk_mugrid = np.linspace(0.0, 1.0, self.Pk_musamp)
        self.Pk_kmesh, self.Pk_mumesh = np.meshgrid(self.Pk_kgrid, self.Pk_mugrid)

    def compute_binned_derivs(self, z_arr):
        """This computes the derivatives of the observed power spectrum for all redshift bins

        Parameters
        ----------
        z_arr : list, numpy.ndarray
                List of the redshift bin centers of the probe

        Returns
        -------
        dict
            A dictionary containing lists of derivatives of the observed power spectrum for each redshift bin and varied parameter
        """
        self.pk_deriv = spectro_cov.SpectroDerivs(
            z_arr,
            self.Pk_kmesh,
            self.Pk_mumesh,
            fiducial_spectro_obj=self.pk_obs_fid,
            bias_samples=self.obs_spectrum,
            configuration=self,
        )
        allpars_deriv = self.pk_deriv.compute_derivs(freeparams=self.freeparams)
        return allpars_deriv

    def pk_LSS_Fisher(self, nbins):
        """This computes the Fisher matrix of a spectroscopic probe for all redshift bins

        Parameters
        ----------
        nbins : int
                    number of redshift bins the spectroscopic probe has

        Returns
        -------
        numpy.ndarray
            A list of Fisher matrices for each redshift bin
        """
        self.parslist = list(self.freeparams.keys())
        fisherMatrix = np.zeros((nbins, len(self.parslist), len(self.parslist)))
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
            text="Computing Pk Fisher matrix, shape: " + str(fisherMatrix.shape),
            instance=self,
        )
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="k_min = {:.3f}Mpc^-1, k_max = {:.3f}Mpc^-1,".format(
                self.kmin_fish, self.kmax_fish
            ),
            instance=self,
        )
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
            text="Fisher Matrix shape: {}".format(fisherMatrix.shape),
            instance=self,
        )
        fisherMatrix = np.array([self.fisher_per_bin(ibin) for ibin in range(nbins)])
        return fisherMatrix

    def fisher_per_bin(self, ibin):
        """This helper function contains the Fisher matrix of a spectroscopic probe for a single redshift bin

        Arguments
        ---------
        ibin : int
               index of the redshift bin

        Returns
        -------
        numpy.ndarray
            Fisher matrix of a spectroscopic probe for a single redshift bin"""
        fish_bi = np.zeros((len(self.parslist), len(self.parslist)))
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="Fisher Spectro bin number: {:d}".format(ibin + 1),
            instance=self,
        )
        for i, pi in enumerate(self.parslist):
            fish_bi[i, i] = self.fisher_calculation(ibin, pi, pi)
            for j in range(i):
                pj = self.parslist[j]
                fish_bi[i, j] = self.fisher_calculation(ibin, pi, pj)
                fish_bi[j, i] = fish_bi[i, j]
        return fish_bi

    def fisher_calculation(self, zi, p_i, p_j):
        """This helper function calculates a singular element of the fisher matrix

        Parameters
        ----------
        zi  : int
              index of the redshift bin
        p_i : str
              name of the first parameter
        p_j : str
              name of the second parameter

        Returns
        -------
        float
            The Fisher matrix element computed from the derivative with respect to two parameters and a given redshift bin for a spectroscopic probe
        """
        k_mesh = self.Pk_kmesh
        mu_mesh = self.Pk_mumesh
        integrand = self.fish_integrand(zi, k_mesh, mu_mesh, p_i, p_j)
        fish_element = self.fisher_integral(integrand)
        return fish_element

    def fisher_integral(self, integrand):
        """Function to calculate the integral that enters the computation of the Fisher elements. Uses a simpson integration on the mu and k grid

        Arguments
        ---------
        integrand : numpy.ndarray
                    Integrant that enters the Fisher matrix element computation. The shape needs to match ( shape of the kmesh ) x ( shape of mumesh)

        Returns
        -------
        float
            The Fisher matrix element computed from the derivative with respect to two parameters at a given redshift bin for a spectroscopic probe
        """
        k_mesh = self.Pk_kmesh
        mu_mesh = self.Pk_mumesh
        mu_integral = simpson(integrand, x=mu_mesh[:, 0], axis=0)
        kmu_integral = simpson(mu_integral, x=k_mesh[0, :])
        return kmu_integral

    def fish_integrand(self, zi, k, mu, pi, pj):
        """Helper function to calculate the integrand that enters the computation of the Fisher elements.

        Arguments
        ---------
        zi : int
             index of the redshift bin
        k  : numpy.ndarray
             grid of wavenumbers used in the calculation. Has to be the internal kmesh
        mu : numpy.ndarray
             grid of observation angles used in the calculation. Has to be the internal mumesh
        pi : str
             name of the first parameter
        pj : str
             name of the second parameter

        Returns
        -------
        numpy.ndarray
            Integrant that enters the Fisher matrix element computation. The shape matches ( shape of the kmesh ) x ( shape of mumesh)
        """
        volsurv = self.pk_cov.volume_survey(zi)
        # pref = 2/(8*(np.pi**2))
        pref = 2  # prefactor due to \mu symmetry
        # zmid = self.pk_cov.global_z_bin_mids[zi]
        dPdpi = self.derivs_dict[pi][zi]
        dPdpj = self.derivs_dict[pj][zi]
        intg = k**2 * pref * volsurv * self.veff_arr[zi] * dPdpi * dPdpj
        return intg

    def photo_LSS_fishermatrix(self, noisy_cls=None, covmat=None, derivs=None, lss_obj=None):
        """Compute the Fisher matrix of a photometric probe.

        Arguments
        ---------
        noisy_cls : dict, optional
                    a dictionary with all the auto and cross correlation fiducial angular power spectra with noise added to it. Will recompute from lss_obj when not passed
        covmat    : list, optional
                    A list of pandas.DataFrame objects that store the covariance matrix for each multipole. Will recompute from lss_obj when not passed
        derivs    : dict, optional
                    A dictionary containing the derivatives of the angular power spectrum at the fiducial for all free parameters. Will recompute from lss_obj when not passed
        lss_obj   : cosmicfishpie.LSSsurvey.photo_cov.PhotoCov, optional
                    This object is used to compute the ingredients of the Fisher matrix if they were not passed

        Returns
        -------
        numpy.ndarray
            The full fisher matrix for the photometric probe
        """
        self.ph_z_bins = deepcopy(self.specs["z_bins_ph"])
        self.ph_num_z_bins = len(self.ph_z_bins) - 1
        if covmat is None and lss_obj is not None:
            noisy_cls, covmat = lss_obj.compute_covmat()
        if derivs is None and lss_obj is not None:
            derivs = lss_obj.compute_derivs()
        tini = time()
        upt.time_print(
            feedback_level=self.feed_lvl, min_level=0, text="Computing Fisher matrix", instance=self
        )
        # compute fisher matrix
        lvec = noisy_cls["ells"]
        lvec_ave = unu.moving_average(lvec, 2)  # computing center of bins
        delta_ell = np.diff(lvec)  # compute delta_ell between bin edges
        FisherV = np.zeros((len(lvec_ave), len(self.freeparams), len(self.freeparams)))

        cols = []
        for o in self.observables:
            for ind in range(self.ph_num_z_bins):
                cols.append(o + " " + str(ind + 1))

        covarr = np.zeros(((len(lvec_ave)), len(cols), len(cols)))
        der1 = np.zeros((len(cols), len(cols)))
        der2 = np.zeros((len(cols), len(cols)))

        for i_ell in range(len(lvec_ave)):
            covarr[i_ell, :, :] = covmat[i_ell]
            covdf = covarr[i_ell, :, :]
            invMat = np.linalg.pinv(covdf)
            if i_ell == 1:
                tparstart = time()
            for ind1, par1 in enumerate(self.freeparams):
                for ind2, par2 in enumerate(self.freeparams):
                    if ind1 > ind2:
                        FisherV[i_ell, ind1, ind2] = FisherV[i_ell, ind2, ind1]
                        continue
                    else:
                        for ia, aa in enumerate(cols):
                            for ib, bb in enumerate(cols):
                                der1[ia, ib] = derivs[par1][aa + "x" + bb][i_ell]
                                der2[ia, ib] = derivs[par2][aa + "x" + bb][i_ell]
                        mat1 = der1.dot(invMat)
                        mat2 = invMat.dot(mat1)
                        mat3 = der2.dot(mat2)
                        trace = np.trace(mat3)
                        FisherV[i_ell, ind1, ind2] = (
                            trace * (lvec_ave[i_ell] + 0.5) * delta_ell[i_ell]
                        )
                    if i_ell == 1:
                        tparend = time()
                        upt.time_print(
                            feedback_level=self.feed_lvl,
                            min_level=3,
                            text="FisherV entry ({}, {}) for ell index {} done in ".format(
                                par1, par2, i_ell
                            ),
                            time_ini=tparstart,
                            time_fin=tparend,
                        )
        FisherVV = np.sum(FisherV, axis=0)

        tfin = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=0,
            text="Finished calculation of Fisher Matrix for {} in: ".format(self.observables),
            time_ini=tini,
            time_fin=tfin,
        )
        return FisherVV

    def photo_LSS_fishermatrix_einsum(self, noisy_cls=None, covmat=None, derivs=None, lss_obj=None):
        """Compute the Fisher matrix of a photometric probe.

        Arguments
        ---------
        noisy_cls : dict, optional
                    a dictionary with all the auto and cross correlation fiducial angular power spectra with noise added to it. Will recompute from lss_obj when not passed
        covmat    : list, optional
                    A list of pandas.DataFrame objects that store the covariance matrix for each multipole. Will recompute from lss_obj when not passed
        derivs    : dict, optional
                    A dictionary containing the derivatives of the angular power spectrum at the fiducial for all free parameters. Will recompute from lss_obj when not passed
        lss_obj   : cosmicfishpie.LSSsurvey.photo_cov.PhotoCov, optional
                    This object is used to compute the ingredients of the Fisher matrix if they were not passed

        Returns
        -------
        numpy.ndarray
            The full fisher matrix for the photometric probe
        """
        self.ph_z_bins = deepcopy(self.specs["z_bins_ph"])
        self.ph_num_z_bins = len(self.ph_z_bins) - 1
        if covmat is None and lss_obj is not None:
            noisy_cls, covmat = lss_obj.compute_covmat()
        if derivs is None and lss_obj is not None:
            derivs = lss_obj.compute_derivs()

        tini = time()
        upt.time_print(
            feedback_level=self.feed_lvl, min_level=0, text="Computing Fisher matrix", instance=self
        )

        # compute fisher matrix
        lvec = noisy_cls["ells"]
        lvec_ave = unu.moving_average(lvec, 2)  # computing center of bins
        delta_ell = np.diff(lvec)  # compute delta_ell between bin edges

        cols = [f"{o} {ind+1}" for o in self.observables for ind in range(self.ph_num_z_bins)]

        # Precompute covariance matrices and their inverses
        covarr = np.array(covmat)
        inv_covarr = np.linalg.pinv(covarr)

        # Precompute derivatives
        der_array = np.array(
            [[derivs[par][f"{a}x{b}"] for a in cols for b in cols] for par in self.freeparams]
        )
        der_array = der_array.reshape(len(self.freeparams), len(cols), len(cols), -1)

        # Precompute the factor for FisherV
        factor = (lvec_ave + 0.5) * delta_ell

        # Compute FisherV using vectorized operations
        FisherV = np.zeros((len(lvec_ave), len(self.freeparams), len(self.freeparams)))
        for l_ind, ell in enumerate(lvec_ave):
            der = der_array[:, :, :, l_ind]  # Shape: (n_freeparams, n_cols, n_cols)

            # mat1: multiply der by inv_covarr
            # 'p' is freeparam index, 'i' and 'j' are column indices
            mat1 = np.einsum("pij,jk->pik", der, inv_covarr[l_ind])

            # mat2: multiply inv_covarr by mat1
            # 'p' is still freeparam index
            mat2 = np.einsum("ij,pjk->pik", inv_covarr[l_ind], mat1)

            # mat3: final matrix multiplication
            # 'p' and 'q' are indices for the two freeparam dimensions
            mat3 = np.einsum("pij,qji->pq", der, mat2)

            FisherV[l_ind] = mat3 * factor[l_ind]

            if l_ind == 1:
                upt.time_print(
                    feedback_level=self.feed_lvl,
                    min_level=3,
                    text=f"FisherV entries for ell={ell} at index {l_ind} done",
                    time_ini=tini,
                    time_fin=time(),
                )
        # Sum up FisherV along the ell dimension
        FisherVV = np.sum(FisherV, axis=0)
        tfin = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=0,
            text=f"Finished calculation of Fisher Matrix for {self.observables} in: ",
            time_ini=tini,
            time_fin=tfin,
        )
        return FisherVV

    def CMB_fishermatrix(self, noisy_cls=None, covmat=None, derivs=None, cmb_obj=None):
        if covmat is None and cmb_obj is not None:
            covmat, noisy_cls = cmb_obj.compute_covmat()
        if derivs is None and cmb_obj is not None:
            derivs = cmb_obj.compute_derivs()

        upt.time_print(
            feedback_level=self.feed_lvl, min_level=0, text="Computing Fisher matrix", instance=self
        )
        # compute fisher matrix
        lvec = noisy_cls["ells"]
        Fisher = np.zeros((len(self.freeparams), len(self.freeparams)))

        # TBA: THIS MUST BE MADE MUCH NICER AND FASTER!!!
        tfishstart = time()
        for ind1, par1 in enumerate(self.freeparams):
            for ind2, par2 in enumerate(self.freeparams):
                if ind1 > ind2:
                    Fisher[ind1, ind2] = Fisher[ind2, ind1]
                    upt.time_print(
                        feedback_level=self.feed_lvl,
                        min_level=2,
                        text="symmetric Fisher matrix element, skipped",
                    )
                    continue
                else:
                    tparstart = time()
                    for i_ell in range(len(lvec)):
                        # Repacking derivatives as matrix
                        cols = []
                        for o in self.observables:
                            cols.append(o)

                        der1 = pd.DataFrame(index=cols, columns=cols)
                        der1 = der1.fillna(0.0)  # with 0s rather than NaNs

                        der2 = pd.DataFrame(index=cols, columns=cols)
                        der2 = der2.fillna(0.0)  # with 0s rather than NaNs

                        for obs1, obs2 in product(self.observables, self.observables):
                            der1.at[obs1, obs2] = derivs[par1][obs1 + "x" + obs2][i_ell]
                            der2.at[obs1, obs2] = derivs[par2][obs1 + "x" + obs2][i_ell]

                        covdf = covmat[i_ell]

                        invMat = pd.DataFrame(
                            np.linalg.pinv(covdf.values), covdf.columns, covdf.index
                        )
                        mat1 = der1.dot(invMat)
                        mat2 = invMat.dot(mat1)
                        mat3 = der2.dot(mat2)
                        trace = np.trace(mat3.values)

                        Fisher[ind1, ind2] = Fisher[ind1, ind2] + (trace * (lvec[i_ell] + 0.5))

                    tparend = time()
                    upt.time_print(
                        feedback_level=self.feed_lvl,
                        min_level=1,
                        text="Fisher entry ({}, {}) done in ".format(par1, par2),
                        time_ini=tparstart,
                        time_fin=tparend,
                    )

        tfishend = time()

        upt.time_print(
            feedback_level=self.feed_lvl, min_level=0, text="Fisher matrix computed", instance=self
        )
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
            text="Fisher matrix calculation finished in ",
            time_ini=tfishstart,
            time_fin=tfishend,
        )
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=0,
            text="Finished calculation of Fisher Matrix for {}".format(self.observables),
        )

        return Fisher

    def eliminate_zbinned_freepars(self, nmax):
        """This helper function is there to find any parameters passed that correspond to no zbin. It will then remove them from the varied parameters list

        Arguments
        ---------
        nmax : int
               Index of the highest redshift bin that should be considered
        """
        freepars = deepcopy(self.freeparams)
        for key in freepars.keys():
            if "_" in key:
                # convention for z-binned pars: par_ibin
                keysplit = key.split("_")
                try:
                    ibin = int(keysplit[1])
                except ValueError:
                    continue
                if type(ibin) == int and ibin > nmax:
                    removedpar = self.freeparams.pop(key)
                    upt.time_print(
                        feedback_level=self.feed_lvl,
                        min_level=1,
                        text="Removed free param: {:s} = {:f}".format(key, removedpar),
                        instance=self,
                    )
                else:
                    continue

    def export_fisher(self, fishmat, totaltime=None):
        """Will print the fisher matrix as well as specifications and the parameter names to files. To change location and filenames for this check the settings dictionary of  cosmicfishpie.configs.config

        Arguments
        ---------
        fishmat   : numpy.ndarray
                    Computed Fisher matrix
        totaltime : float, optional
                    Total time needed to compute the fisher. Will print time information if passed

        Returns
        -------
        cosmicfishpie.analysis.fisher_matrix
            A instance of fisher_matrix containing the calculated Fisher matrix as well as parameter names, settings, etc
        """
        # If an output root is provided, we write the Fisher matrix on file
        if self.settings["outroot"] != "":
            obstring = ""
            for obs in self.observables:
                obstring = obstring + obs
            cols = [key for key in self.freeparams]
            header = "#"
            for col in cols:
                header = header + " " + col
            FM = pd.DataFrame(fishmat, columns=cols, index=cols)
            if not os.path.exists(self.settings["results_dir"]):
                os.makedirs(self.settings["results_dir"])
            filename = (
                self.settings["results_dir"]
                + "/"
                + self.cf_version
                + "_"
                + self.settings["outroot"]
                + "_"
                + obstring
            )
            filename = filename + "_fishermatrix"
            extension = self.settings["fishermatrix_file_extension"]
            FM.to_csv(filename + ".csv")
            with open(filename + extension, "w") as f:
                f.write(header + "\n")
                f.write(FM.to_csv(header=False, index=False, sep="\t"))
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=0,
                text="Fisher matrix exported: {:s}".format(filename + extension),
            )
            with open(filename + ".paramnames", "w") as f:
                f.write("#\n")
                f.write("#\n")
                f.write("# This file contains the parameter names for a derived Fisher matrix.\n")
                f.write("#\n")
                for par in self.freeparams.keys():
                    f.write(
                        str(par)
                        + "    "
                        + str(cfg.latex_names.get(par, str(par)))
                        + "    "
                        + "{:.6f}".format(self.allparams[par])
                    )
                    f.write("\n")
            fishMat_obj = fm.fisher_matrix(file_name=filename + ".txt")
            # Write specifications to file:
            specs_name = filename + "_specifications.dat"
            with open(specs_name, "w") as f:
                f.write("**Git version commit: %s \n" % (ufs.git_version()))
                if totaltime is not None:
                    f.write(
                        "**Time needed for computation of Fisher matrix: {:.3f} seconds \n".format(
                            totaltime
                        )
                    )
                f.write("**Observables: %s \n" % (obstring))
                f.write("### CosmicFish settings ###\n")
                for key, value in sorted(self.settings.items()):
                    f.write("%s:%s\n" % (key, value))
                f.write("### Fiducial parameters ###\n")
                for key, value in sorted(self.fiducialcosmopars.items()):
                    f.write("%s:%s\n" % (key, value))
                f.write("### Free parameters ###\n")
                for key, value in sorted(self.freeparams.items()):
                    f.write("%s:%s\n" % (key, value))
                f.write("### Survey specifications ###\n")
                for key, value in sorted(self.specs.items()):
                    f.write("%s:%s\n" % (key, value))
            print()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="CosmicFish settings and Survey specifications exported: {:s}".format(
                    specs_name
                ),
                instance=self,
            )
            return fishMat_obj

    def recap_options(self):
        """This will print all the selected options into the standard output"""
        if self.settings["feedback"] > 1:
            print("")
            print("----------RECAP OF SELECTED OPTIONS--------")
            print("")
            print("Settings:")
            for key in self.settings:
                print("   " + key + ": {}".format(self.settings[key]))
            print("")
            print("Specifications:")
            for key in self.specs:
                print("   " + key + ": {}".format(self.specs[key]))
            if self.feed_lvl > 1:
                print("Cosmological parameters:")
                for key in self.fiducialcosmopars:
                    print("   " + key + ": {}".format(self.fiducialcosmopars[key]))
                print("Photometric parameters:")
                for key in self.photopars:
                    print("   " + key + ": {}".format(self.photopars[key]))
                print("IA parameters:")
                for key in self.IApars:
                    print("   " + key + ": {}".format(self.IApars[key]))
                print("Bias parameters:")
                for key in self.photobiaspars:
                    print("   " + key + ": {}".format(self.photobiaspars[key]))
                print("SpectroBias parameters:")
                for key in self.Spectrobiaspars:
                    print("   " + key + ": {}".format(self.Spectrobiaspars[key]))
                print("SpectroNonlinear parameters:")
                for key in self.Spectrononlinpars:
                    print("   " + key + ": {}".format(self.Spectrononlinpars[key]))
                print("IMBias parameters:")
                for key in self.IMbiaspars:
                    print("   " + key + ": {}".format(self.IMbiaspars[key]))
                print("PShot parameters:")
                for key in self.PShotpars:
                    print("   " + key + ": {}".format(self.PShotpars[key]))
                print("Free parameters:")
                for par in self.freeparams:
                    print("   " + par)
