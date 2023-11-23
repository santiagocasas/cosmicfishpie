# -*- coding: utf-8 -*-
"""LSS MAIN

This is the main engine for LSS Fisher Matrix.

"""
import copy
import datetime
from itertools import product
from time import time

import numpy as np
import pandas as pd

import cosmicfishpie.fishermatrix.config as cfg
import cosmicfishpie.fishermatrix.derivatives as fishderiv
import cosmicfishpie.LSSsurvey.photo_obs as photo_obs
from cosmicfishpie.utilities.utils import printing as upt

pd.set_option("display.float_format", "{:.9E}".format)


class PhotoCov:
    def __init__(self, cosmopars, photopars, IApars, biaspars, fiducial_Cls=None):
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
        for key in cfg.obs:
            if key in ["GCph", "WL"]:
                self.observables.append(key)
        self.binrange = cfg.specs["binrange"]
        self.feed_lvl = cfg.settings["feedback"]

    def getcls(self, allpars):
        # Here call to functions getting windows and then do cls
        # Splitting the dictionary of full parameters
        cosmopars = dict((k, allpars[k]) for k in self.cosmopars)
        photopars = dict((k, allpars[k]) for k in self.photopars)
        IApars = dict((k, allpars[k]) for k in self.IApars)
        biaspars = dict((k, allpars[k]) for k in self.biaspars)
        if cosmopars == self.cosmopars and self.fiducial_Cls is not None:
            cls = photo_obs.ComputeCls(
                cosmopars, photopars, IApars, biaspars, fiducial_cosmo=self.fiducial_Cls.cosmo
            )
        else:
            cls = photo_obs.ComputeCls(cosmopars, photopars, IApars, biaspars, fiducial_cosmo=None)

        cls.compute_all()
        LSScls = cls.result

        return LSScls

    def getclsnoise(self, cls):
        """Noise value

        Parameters
        ----------
        cls   : dict
                cls dictionary

        Returns
        -------
        array
            Noise value[ell,bin]

        Notes
        -----
        Implements the following equation:

        .. math::
            N_\\ell^X = ...

        """

        # TBA: reading the number directly. This should be computed
        ngalbin = np.array(cfg.specs["ngalbin"])
        ellipt_error = cfg.specs["ellipt_error"]

        noisy_cls = copy.deepcopy(cls)

        for ind in self.binrange:
            for obs in self.observables:
                if obs == "GCph":
                    # TBA: fix this
                    noisy_cls[obs + " " + str(ind) + "x" + obs + " " + str(ind)] += (
                        1.0 / ngalbin[ind - 1]
                    )
                elif obs == "WL":
                    noisy_cls[obs + " " + str(ind) + "x" + obs + " " + str(ind)] += (
                        ellipt_error**2.0
                    ) / ngalbin[ind - 1]

        return noisy_cls

    def get_covmat(self, noisy_cls):
        """Data covariance

        Parameters
        ----------
        noisy_cls   : dict
                      dictionary containing cls with noise

        Returns
        -------
        list
        data covariance matrix

        """

        pd.set_option("display.float_format", "{:.9E}".format)

        numbins = len(cfg.specs["z_bins"]) - 1
        covvec = []

        # Create indexes for data frame
        cols = []
        for o in self.observables:
            for ind in range(numbins):
                cols.append(o + " " + str(ind + 1))

        for ind, ell in enumerate(noisy_cls["ells"]):
            covdf = pd.DataFrame(index=cols, columns=cols)
            covdf = covdf.fillna(0.0)

            for obs1, obs2, bin1, bin2 in product(
                self.observables, self.observables, self.binrange, self.binrange
            ):
                covdf.at[obs1 + " " + str(bin1), obs2 + " " + str(bin2)] = noisy_cls[
                    obs1 + " " + str(bin1) + "x" + obs2 + " " + str(bin2)
                ][ind] / np.sqrt(np.sqrt(cfg.specs["fsky_" + obs1] * cfg.specs["fsky_" + obs2]))

            covvec.append(covdf)

        return covvec

    def compute_covmat(self):
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
