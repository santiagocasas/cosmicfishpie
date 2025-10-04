# -*- coding: utf-8 -*-
"""LSS MAIN

This is the main engine for LSS Fisher Matrix.

"""
import datetime
from collections import OrderedDict
from itertools import product
from time import time

import numpy as np
import pandas as pd

import cosmicfishpie.configs.config as cfg
import cosmicfishpie.fishermatrix.derivatives as fishderiv
import cosmicfishpie.LSSsurvey.photo_obs as photo_obs
from cosmicfishpie.utilities.utils import printing as upt

pd.set_option("future.no_silent_downcasting", True)  # to avoid future pandas warning
pd.set_option("display.float_format", "{:.9E}".format)


class PhotoCov:
    def __init__(
        self,
        cosmopars,
        photopars,
        IApars,
        biaspars,
        fiducial_Cls=None,
        enable_cache=True,
        cache_size=32,
    ):
        """Main class to obtain the ingredients for the Fisher matrix of a photometric probe

        Parameters
        ----------
        cosmopars    : dict
                       a dictionary containing the fiducial cosmological parameters
        photopars    : dict
                       a dictionary containing specifications for the window function's galaxy distribution
        IApars       : dict
                       a dictionary containing the fiducial specifications for the intrinsic alignment effect in cosmic shear
        biaspars     : dict
                       a dictionary containing the fiducial specifications for the galaxy biases
        fiducial_Cls : cosmicfishpie.LSSsurvey.photo_obs.ComputeCls, optional
                       an instance of `ComputeCLs` containing the fiducial angular power spectrum. Will recompute if not passed

        Attributes
        ----------
        cosmopars    : dict
                       a dictionary containing the fiducial cosmological parameters
        photopars    : dict
                       a dictionary containing specifications for the window function's galaxy distribution
        IApars       : dict
                       a dictionary containing the fiducial specifications for the intrinsic alignment effect in cosmic shear
        biaspars     : dict
                       a dictionary containing the fiducial specifications for the galaxy biases
        fiducial_Cls : cosmicfishpie.LSSsurvey.photo_obs
                       an instance of `photo_obs` containing the fiducial angular power spectrum
        allparsfid   : dict
                       a dictionary with all fiducial parameters needed to compute the angular power spectrum
        observables  : list
                       a list of the observables that the angular power spectrum is computed for
        binrange     : range
                       a range with all the bins asked for in the survey specifications
        feed_lvl     : int
                       number indicating the verbosity of the output. Higher numbers mean more output

        """
        self.cosmopars = cosmopars
        self.photopars = photopars
        self.IApars = IApars
        self.biaspars = biaspars
        self.allparsfid = dict()
        self.allparsfid.update(self.cosmopars)
        self.allparsfid.update(self.IApars)
        self.allparsfid.update(self.biaspars)
        self.allparsfid.update(self.photopars)
        self.fiducial_Cls = fiducial_Cls
        self.observables = []
        self.binrange = {}
        for key in cfg.obs:
            if key in ["GCph", "WL"]:
                self.observables.append(key)
                if key == "GCph":
                    self.binrange[key] = cfg.specs["binrange_GCph"]
                elif key == "WL":
                    self.binrange[key] = cfg.specs["binrange_WL"]

        self.binrange_WL = cfg.specs["binrange_WL"]
        self.binrange_GCph = cfg.specs["binrange_GCph"]
        self.feed_lvl = cfg.settings["feedback"]
        self.fsky_WL = cfg.specs.get("fsky_WL")
        self.fsky_GCph = cfg.specs.get("fsky_GCph")
        self.ngalbin_WL = np.array(cfg.specs["ngalbin_WL"])
        self.ngalbin_GCph = np.array(cfg.specs["ngalbin_GCph"])
        self.numbins_WL = len(cfg.specs["z_bins_WL"]) - 1
        self.numbins_GCph = len(cfg.specs["z_bins_GCph"]) - 1
        self.ellipt_error = cfg.specs["ellipt_error"]

        # Performance tuning options
        self._enable_cache = enable_cache
        self._cls_cache = OrderedDict()
        self._cls_cache_max = max(1, cache_size)
        # Precompute static noise vectors (used in getclsnoise)
        self._wl_shape_noise = None
        if self.numbins_WL > 0:
            # sigma_e^2 / n_g
            self._wl_shape_noise = (self.ellipt_error**2.0) / self.ngalbin_WL
        self._gcph_shot_noise = None
        if self.numbins_GCph > 0:
            self._gcph_shot_noise = 1.0 / self.ngalbin_GCph

    def getcls(self, allpars):
        """Function to calculate the angular power spectrum

        Parameters
        ----------
        allpars : dict
                  a dictionary with all parameters needed to compute the angular power spectrum

        Returns
        -------
        dict
            a dictionary with all the auto and cross correlation angular power spectra
        """
        cosmopars = {k: allpars[k] for k in self.cosmopars}
        photopars = {k: allpars[k] for k in self.photopars}
        IApars = {k: allpars[k] for k in self.IApars}
        biaspars = {k: allpars[k] for k in self.biaspars}

        cache_key = None
        if self._enable_cache:
            # Immutable sorted tuple of all parameter key-value pairs across groups
            cache_key = (
                tuple(sorted(cosmopars.items()))
                + tuple(sorted(photopars.items()))
                + tuple(sorted(IApars.items()))
                + tuple(sorted(biaspars.items()))
            )
            cached = self._cls_cache.get(cache_key)
            if cached is not None:
                # LRU update
                self._cls_cache.move_to_end(cache_key)
                return cached
        if cosmopars == self.cosmopars and self.fiducial_Cls is not None:
            cls = photo_obs.ComputeCls(
                cosmopars, photopars, IApars, biaspars, fiducial_cosmo=self.fiducial_Cls.cosmo
            )
        else:
            cls = photo_obs.ComputeCls(cosmopars, photopars, IApars, biaspars, fiducial_cosmo=None)

        cls.compute_all()
        LSScls = cls.result
        if self._enable_cache and cache_key is not None:
            self._cls_cache[cache_key] = LSScls
            if len(self._cls_cache) > self._cls_cache_max:
                # pop least recently used
                self._cls_cache.popitem(last=False)
        return LSScls

    def getclsnoise(self, cls):
        """Obtain the angular power spectrum with noise

        Parameters
        ----------
        cls : dict
              a dictionary with all the auto and cross correlation angular power spectra

        Returns
        -------
        dict
              a dictionary with all the auto and cross correlation angular power spectra with noise added to it

        Note
        ----
        Implements the following equation:

        .. math::
            N_{ij}^{YX}(\\ell) = \\frac{\\sigma_X^2}{\\bar{n}_i}\\,\\delta_{XY}\\,\\delta_{ij}

            \\sigma_L = \\sigma_\\epsilon \\quad , \\quad \\sigma_G = 1
        """
        # Shallow copy: we reassign modified arrays so originals aren't altered.
        noisy_cls = dict(cls)
        if "GCph" in self.observables and self._gcph_shot_noise is not None:
            for ind in self.binrange_GCph:
                k = f"GCph {ind}xGCph {ind}"
                noisy_cls[k] = noisy_cls[k] + self._gcph_shot_noise[ind - 1]
        if "WL" in self.observables and self._wl_shape_noise is not None:
            for ind in self.binrange_WL:
                k = f"WL {ind}xWL {ind}"
                noisy_cls[k] = noisy_cls[k] + self._wl_shape_noise[ind - 1]
        return noisy_cls

    def get_covmat(self, noisy_cls):
        """Computes the covariance matrix from the noisy angular power spectrum

        Parameters
        ----------
        noisy_cls : dict
                    a dictionary with all the auto and cross correlation angular power spectra with noise added to it

        Returns
        -------
        list
            A list of pandas.DataFrame objects that store the covariance matrix for each multipole

        """

        pd.set_option("display.float_format", "{:.9E}".format)

        covvec = []

        # Create indexes for data frame
        cols = []
        for o in self.observables:
            if o == "WL":
                for ind in range(self.numbins_WL):
                    cols.append(o + " " + str(ind + 1))
            elif o == "GCph":
                for ind in range(self.numbins_GCph):
                    cols.append(o + " " + str(ind + 1))

        # Precompute simple fsky factors (original code used sqrt(sqrt(fsky*fsky)) == sqrt(fsky))
        fsky_factor = {o: np.sqrt(getattr(self, "fsky_" + o)) for o in self.observables}

        for ind, ell in enumerate(noisy_cls["ells"]):
            covdf = pd.DataFrame(index=cols, columns=cols)
            covdf = covdf.fillna(0.0).infer_objects(copy=False)

            for obs1, obs2 in product(self.observables, self.observables):
                for bin1, bin2 in product(
                    self.binrange[obs1], self.binrange[obs2]
                ):  # MMmod: BEWARE!!! THIS IS VERY UGLY!
                    covdf.at[obs1 + " " + str(bin1), obs2 + " " + str(bin2)] = (
                        noisy_cls[obs1 + " " + str(bin1) + "x" + obs2 + " " + str(bin2)][ind]
                        / fsky_factor[obs1]
                    )

            covvec.append(covdf)

        return covvec

    def compute_covmat(self):
        """
        Computes the fiducial covariance matrix for the Fisher matrix.

        Returns
        -------
        dict
            a dictionary with all the auto and cross correlation fiducial angular power spectra with noise added to it
        list
            A list of pandas.DataFrame objects that store the covariance matrix for each multipole
        """
        tini = datetime.datetime.now().timestamp()

        obstring = ""
        for obs in self.observables:
            obstring = obstring + obs

        # Check free pars are in the fiducial
        for key in cfg.freeparams:
            if key not in self.allparsfid:
                print("ERROR: free param " + key + " does not have a fiducial value!")
                return None

        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Computing fiducial")
        t1 = datetime.datetime.now().timestamp()

        cls = self.getcls(self.allparsfid)

        t2 = datetime.datetime.now().timestamp()
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Fiducial generated in {:.2f} s".format(t2 - t1))

        t1 = datetime.datetime.now().timestamp()
        # add noise
        noisy_cls = self.getclsnoise(cls)
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Noise added to fiducial")
        t2 = datetime.datetime.now().timestamp()
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Noisy Cls generated in {:.2f} s".format(t2 - t1))

        # build covmat
        self.cls = cls
        self.noisy_cls = noisy_cls
        t1 = datetime.datetime.now().timestamp()
        self.covmat = self.get_covmat(noisy_cls)
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Computed covariance matrix")
        t2 = datetime.datetime.now().timestamp()
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Covmat of Cls generated in {:.2f} s".format(t2 - t1))

        tfin = datetime.datetime.now().timestamp()
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Total calculation in {:.2f} s".format(tfin - tini))

        return self.noisy_cls, self.covmat

    def compute_derivs(self, print_info_specs=False):
        """computes the derivatives of the angular power spectrum needed to compute the Fisher matrix

        Returns
        -------
        dict
            a dictionary containing the derivatives of the angular power spectrum at the fiducial for all free parameters

        """
        # compute and save derivatives-----------------------------------------

        derivs = dict()
        compute_derivs = True
        if compute_derivs:
            tder1 = time()
            print(">> computing derivs >>")
            deriv_engine = fishderiv.derivatives(self.getcls, self.allparsfid)
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
