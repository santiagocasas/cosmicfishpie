# -*- coding: utf-8 -*-
"""MAIN

This is the main engine of CosmicFish.

"""
import os
import sys
from copy import deepcopy
from itertools import product
from time import time

import numpy as np
import pandas as pd
from scipy.integrate import simps

import cosmicfishpie.CMBsurvey.CMB_cov as CMB_cov
import cosmicfishpie.fishermatrix.config as cfg
import cosmicfishpie.LSSsurvey.photo_cov as photo_cov
import cosmicfishpie.LSSsurvey.photo_obs as photo_obs
import cosmicfishpie.LSSsurvey.spectro_cov as spec_cov
import cosmicfishpie.LSSsurvey.spectro_obs as spec_obs
from cosmicfishpie.analysis import fisher_matrix as fm

# from cosmicfishpie.cosmology.cosmology import *
from cosmicfishpie.utilities.utils import filesystem as ufs
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upt


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
    cf_version = "CosmicFish_v1.0"

    def __init__(
        self,
        options=dict(),
        specifications=dict(),
        observables=None,
        freepars=None,
        extfiles=None,
        fiducialpars=None,
        biaspars=None,
        photopars=None,
        spectrononlinearpars=None,
        IApars=None,
        surveyName="Euclid",
        cosmoModel="w0waCDM",
        latexnames=None,
        parallel="False",
        parallel_cpus=4,
    ):
        print("**************************************************************")
        print("   _____               _     _____     __  ")
        print("  / ___/__  ___ __ _  (_)___/ __(_)__ / /  ")
        print(" / /__/ _ \\(_-</  ' \\/ / __/ _// (_-</ _ \\ ")
        print(" \\___/\\___/___/_/_/_/_/\\__/_/ /_/___/_//_/ ")
        print("")
        print("**************************************************************")
        print(" This is the new Python version of the CosmicFish code.")
        print("**************************************************************")
        sys.stdout.flush()

        cfg.init(
            options=options,
            specifications=specifications,
            observables=observables,
            freepars=freepars,
            extfiles=extfiles,
            fiducialpars=fiducialpars,
            biaspars=biaspars,
            photopars=photopars,
            IApars=IApars,
            surveyName=surveyName,
            cosmoModel=cosmoModel,
            latexnames=latexnames,
            spectrononlinearpars=spectrononlinearpars,
        )

        self.fiducialcosmopars = cfg.fiducialparams
        self.photopars = cfg.photoparams
        self.IApars = cfg.IAparams
        self.biaspars = cfg.biasparams
        self.Spectrobiaspars = cfg.Spectrobiasparams
        self.Spectrononlinpars = cfg.Spectrononlinearparams
        self.IMbiaspars = cfg.IMbiasparams
        self.PShotpars = cfg.PShotparams
        self.observables = cfg.obs
        self.freeparams = cfg.freeparams
        self.allparams_fidus = {
            **self.fiducialcosmopars,
            **self.photopars,
            **self.IApars,
            **self.biaspars,
            **self.Spectrobiaspars,
            **self.IMbiaspars,
            **self.PShotpars,
        }
        self.parallel = parallel
        # MMmod: avoids code to crush if specs don't have redshift bins
        if "z_bins" in cfg.specs:
            self.z_bins = cfg.specs["z_bins"]
            self.num_z_bins = len(cfg.specs["z_bins"]) - 1
            self.binrange = cfg.specs["binrange"]
        self.feed_lvl = cfg.settings["feedback"]
        allpars = {}
        allpars.update(self.fiducialcosmopars)
        allpars.update(self.IApars)
        allpars.update(self.biaspars)
        allpars.update(self.Spectrobiaspars)
        allpars.update(self.IMbiaspars)
        allpars.update(self.PShotpars)
        self.allparams = allpars

        # self.parallel=True
        self.recap_options()
        if self.parallel:
            ray_session(num_cpus=parallel_cpus, restart=True, shutdown=False)

    def compute(self, max_z_bins=None):
        # This switches between the possible Fisher matrices

        tfishstart = time()
        if "GCph" in self.observables or "WL" in self.observables:
            self.photo_obs_fid = photo_obs.ComputeCls(
                self.fiducialcosmopars,
                self.photopars,
                self.IApars,
                self.biaspars,
                print_info_specs=True,
            )
            self.photo_LSS = photo_cov.PhotoCov(
                self.fiducialcosmopars,
                self.photopars,
                self.IApars,
                self.biaspars,
                fiducial_Cls=self.photo_obs_fid,
            )
            noisy_cls, covmat = self.photo_LSS.compute_covmat()
            derivs = self.photo_LSS.compute_derivs()
            photoFM = self.photo_LSS_fishermatrix(noisy_cls=noisy_cls, covmat=covmat, derivs=derivs)
            finalFisher = deepcopy(photoFM)

        elif "GCsp" in self.observables or "IM" in self.observables:
            self.set_pk_settings()
            if "IM" in self.observables and "GCsp" in self.observables:
                self.obs_spectrum = ["I", "g"]
                self.pk_obs = spec_obs.ComputeGalIM(
                    cosmopars=self.fiducialcosmopars,
                    fiducial_cosmopars=self.fiducialcosmopars,
                    bias_samples=self.obs_spectrum,
                )

            elif "IM" in self.observables:
                self.obs_spectrum = ["I", "I"]
                self.pk_obs = spec_obs.ComputeGalIM(
                    cosmopars=self.fiducialcosmopars,
                    fiducial_cosmopars=self.fiducialcosmopars,
                    bias_samples=self.obs_spectrum,
                )
            elif "GCsp" in self.observables:
                self.obs_spectrum = ["g", "g"]
                self.pk_obs = spec_obs.ComputeGalSpectro(
                    cosmopars=self.fiducialcosmopars,
                    fiducial_cosmopars=self.fiducialcosmopars,
                    bias_samples=self.obs_spectrum,
                    spectrononlinearpars=self.Spectrononlinpars,
                )

            self.pk_cov = spec_cov.SpectroCov(
                self.fiducialcosmopars, fiducial_specobs=self.pk_obs, bias_samples=self.obs_spectrum
            )
            self.zmids = self.pk_cov.global_z_bin_mids
            nbins = len(self.zmids)
            if max_z_bins is None:
                max_z_bins = nbins
            self.eliminate_zbinned_freepars(max_z_bins)
            tini = time()
            self.derivs_dict = dict()
            # if self.parallel == False:
            k_mesh = self.Pk_kmesh
            mu_mesh = self.Pk_mumesh
            self.veff_arr = []
            tvi = time()
            self.fish_z_arr = np.empty(max_z_bins)
            for ibi in range(max_z_bins):
                self.fish_z_arr[ibi] = self.zmids[ibi]
                if "IM" in self.observables and "GCsp" in self.observables:
                    # compute  in the case of IMxGC
                    self.veff_arr.append(self.pk_cov.veff_XC(ibi, k_mesh, mu_mesh))
                elif "IM" in self.observables:  # compute  in the case of IM only
                    self.veff_arr.append(self.pk_cov.veff_21cm(ibi, k_mesh, mu_mesh))
                elif "GCsp" in self.observables:
                    self.veff_arr.append(self.pk_cov.veff(ibi, k_mesh, mu_mesh))
            tvf = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="+++ Volumes computation computed in: ",
                time_ini=tvi,
                time_fin=tvf,
                instance=self,
            )
            self.derivs_dict = self.compute_binned_derivs(self.fish_z_arr)
            tend = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="+++ Derivatives computation done in: ",
                time_ini=tini,
                time_fin=tend,
                instance=self,
            )
            self.allparams = self.pk_deriv.fiducial_allpars
            pk_Fish = self.pk_LSS_Fisher(nbins=max_z_bins)
            print("Redshift binned Fisher shape: ", pk_Fish.shape)
            specFM = np.sum(pk_Fish, axis=0)
            finalFisher = deepcopy(specFM)

        elif (
            "CMB_T" in self.observables
            or "CMB_E" in self.observables
            or "CMB_B" in self.observables
        ):
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
        # kmin and kmax GCsp values are always indicated in h/Mpc
        k_units_factor = self.fiducialcosmopars["h"]
        self.kmin_fish = cfg.specs["kmin_GCsp"] * k_units_factor
        self.kmax_fish = cfg.specs["kmax_GCsp"] * k_units_factor
        self.Pk_ksamp = 2049 * cfg.settings["accuracy"]
        self.Pk_musamp = 129 * cfg.settings["accuracy"]
        self.Pk_kgrid = np.linspace(self.kmin_fish, self.kmax_fish, self.Pk_ksamp)
        self.Pk_mugrid = np.linspace(0.0, 1.0, self.Pk_musamp)
        self.Pk_kmesh, self.Pk_mumesh = np.meshgrid(self.Pk_kgrid, self.Pk_mugrid)

    def compute_binned_derivs(self, z_arr):
        self.pk_deriv = spec_cov.SpectroDerivs(
            z_arr,
            self.Pk_kmesh,
            self.Pk_mumesh,
            fiducial_spectro_obj=self.pk_obs,
            bias_samples=self.obs_spectrum,
        )
        allpars_deriv = self.pk_deriv.compute_derivs(freeparams=self.freeparams)
        return allpars_deriv

    def pk_LSS_Fisher(self, nbins):
        self.parslist = list(self.freeparams.keys())
        fisherMatrix = np.zeros((nbins, len(self.parslist), len(self.parslist)))
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
            text="Computing Pk Fisher matrix, shape: " + str(fisherMatrix.shape),
            instance=self,
        )
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
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
        k_mesh = self.Pk_kmesh
        mu_mesh = self.Pk_mumesh
        integrand = self.fish_integrand(zi, k_mesh, mu_mesh, p_i, p_j)
        fish_element = self.fisher_integral(integrand)
        return fish_element

    def fisher_integral(self, integrand):
        k_mesh = self.Pk_kmesh
        mu_mesh = self.Pk_mumesh
        mu_integral = simps(integrand, x=mu_mesh[:, 0], axis=0)
        kmu_integral = simps(mu_integral, x=k_mesh[0, :])
        return kmu_integral

    def fish_integrand(self, zi, k, mu, pi, pj):
        volsurv = self.pk_cov.volume_survey(zi)
        # pref = 2/(8*(np.pi**2))
        pref = 2  # prefactor due to \mu symmetry
        # zmid = self.pk_cov.global_z_bin_mids[zi]
        dPdpi = self.derivs_dict[pi][zi]
        dPdpj = self.derivs_dict[pj][zi]
        intg = k**2 * pref * volsurv * self.veff_arr[zi] * dPdpi * dPdpj
        return intg

    def photo_LSS_fishermatrix(self, noisy_cls=None, covmat=None, derivs=None, lss_obj=None):
        if covmat is None and lss_obj is not None:
            noisy_cls, covmat = self.photo_LSS.compute_covmat()
        if derivs is None and lss_obj is not None:
            derivs = self.photo_LSS.compute_derivs()
        tini = time()
        upt.time_print(
            feedback_level=self.feed_lvl, min_level=0, text="Computing Fisher matrix", instance=self
        )
        # compute fisher matrix
        lvec = noisy_cls["ells"]
        lvec_ave = unu.moving_average(lvec, 2)  # computing center of bins
        delta_ell = np.diff(lvec)  # compute delta_ell between bin edges
        FisherV = np.zeros((len(lvec_ave), len(self.freeparams), len(self.freeparams)))
        numbins = self.num_z_bins

        cols = []
        for o in self.observables:
            for ind in range(numbins):
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
                            min_level=1,
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
        freepars = deepcopy(self.freeparams)
        for key in freepars.keys():
            if "_" in key:
                # convention for z-binned pars: par_ibin
                keysplit = key.split("_")
                ibin = int(keysplit[1])
                if ibin > nmax:
                    removedpar = self.freeparams.pop(key)
                    upt.time_print(
                        feedback_level=self.feed_lvl,
                        min_level=1,
                        text="Removed free param: {:s} = {:s}".format(key, removedpar),
                        instance=self,
                    )
                else:
                    continue

    def export_fisher(self, fishmat, totaltime=None):
        # If an output root is provided, we write the Fisher matrix on file
        if cfg.settings["outroot"] != "":
            obstring = ""
            for obs in self.observables:
                obstring = obstring + obs
            cols = [key for key in self.freeparams]
            header = "#"
            for col in cols:
                header = header + " " + col
            FM = pd.DataFrame(fishmat, columns=cols, index=cols)
            if not os.path.exists(cfg.settings["results_dir"]):
                os.makedirs(cfg.settings["results_dir"])
            filename = (
                cfg.settings["results_dir"]
                + "/"
                + FisherMatrix.cf_version
                + "_"
                + cfg.settings["outroot"]
                + "_"
                + obstring
            )
            # if cfg.settings['nonlinear']: filename = filename+'_nonlinear'
            filename = filename + "_fishermatrix"
            extension = cfg.settings["fishermatrix_file_extension"]
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
                for key, value in sorted(cfg.settings.items()):
                    f.write("%s:%s\n" % (key, value))
                f.write("### Fiducial parameters ###\n")
                for key, value in sorted(self.fiducialcosmopars.items()):
                    f.write("%s:%s\n" % (key, value))
                f.write("### Free parameters ###\n")
                for key, value in sorted(self.freeparams.items()):
                    f.write("%s:%s\n" % (key, value))
                f.write("### Survey specifications ###\n")
                for key, value in sorted(cfg.specs.items()):
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
        if cfg.settings["feedback"] > 0:
            print("")
            print("----------RECAP OF SELECTED OPTIONS--------")
            print("")
            print("Settings:")
            for key in cfg.settings:
                print("   " + key + ": {}".format(cfg.settings[key]))
            print("")
            print("Specifications:")
            for key in cfg.specs:
                print("   " + key + ": {}".format(cfg.specs[key]))
            if cfg.settings["feedback"] > 1:
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
                for key in self.biaspars:
                    print("   " + key + ": {}".format(self.biaspars[key]))
                print("SpectroBias parameters:")
                for key in self.Spectrobiaspars:
                    print("   " + key + ": {}".format(self.Spectrobiaspars[key]))
                print("IMBias parameters:")
                for key in self.IMbiaspars:
                    print("   " + key + ": {}".format(self.IMbiaspars[key]))
                print("PShot parameters:")
                for key in self.PShotpars:
                    print("   " + key + ": {}".format(self.PShotpars[key]))
                print("Free parameters:")
                for par in self.freeparams:
                    print("   " + par)
