# -*- coding: utf-8 -*-
"""CMB MAIN

This is the main engine for CMB Fisher Matrix.

"""
import copy
import datetime
import os
from itertools import product
from time import time

import numpy as np
import pandas as pd

import cosmicfishpie.CMBsurvey.CMB_obs as CMB_obs
import cosmicfishpie.fishermatrix.config as cfg
import cosmicfishpie.fishermatrix.derivatives as fishderiv
from cosmicfishpie.utilities.utils import printing as upt

pd.set_option("display.float_format", "{:.9E}".format)


class CMBCov:
    def __init__(self, cosmopars, print_info_specs=False):
        self.cosmopars = cosmopars
        self.observables = []
        for key in cfg.obs:  # LENSING TO BE ADDED
            if key in ["CMB_T", "CMB_E", "CMB_B"]:
                self.observables.append(key)

        self.print_info = print_info_specs
        self.feed_lvl = cfg.settings["feedback"]

    def getcls(self, allpars):
        # Here call to functions getting windows and then do cls

        # Splitting the dictionary of full parameters
        pars = dict((k, allpars[k]) for k in self.cosmopars)

        cls = CMB_obs.ComputeCls(pars, print_info_specs=self.print_info)

        cls.compute_all()
        CMBcls = cls.result

        return CMBcls

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

        Note
        -----
        Implements the following equation:

        .. math::
            N_\\ell^X = ...

        """

        noisy_cls = copy.deepcopy(cls)

        # MM: kind of copy pasting the original cosmicfish stuff
        #        temp1 = self.cosmopars['TCMB']*np.pi/180./60.
        #        temp2 = 180.*60.*np.sqrt(8.*np.log(2.))/np.pi

        #        noisevar_temp = (np.array(cfg.specs['CMB_temp_sens'])*np.array(cfg.specs['CMB_fwhm'])*temp1)**2.
        #        noisevar_pol  = (np.array(cfg.specs['CMB_pol_sens'])*np.array(cfg.specs['CMB_fwhm'])*temp1)**2.
        #        sigma2        = -1.*(np.array(cfg.specs['CMB_fwhm'])/temp2 )**2

        #        TT_Noise  = np.zeros((len(noisy_cls['ells'])))
        #        pol_Noise = np.zeros((len(noisy_cls['ells'])))
        #        for ind,sig2 in enumerate(sigma2):
        #            TT_Noise  += np.exp(noisy_cls['ells']*(noisy_cls['ells']+1)*sig2)/noisevar_temp[ind]
        #            pol_Noise += np.exp(noisy_cls['ells']*(noisy_cls['ells']+1)*sig2)/noisevar_pol[ind]

        #        for obs in self.observables:
        #            if obs == 'CMB_T':
        #                noisy_cls[obs+'x'+obs] += 1/TT_Noise
        #            elif obs == 'CMB_E' or obs == 'CMB_B':
        #                noisy_cls[obs+'x'+obs] += 1/pol_Noise

        arcmin_to_rad = 0.000290888
        # norm          = ls*(ls+1)/(2.*np.pi)

        thetab = [
            arcmin_to_rad * beam / np.sqrt(8.0 * np.log(2.0)) for beam in cfg.specs["CMB_fwhm"]
        ]

        Bell = [
            np.exp(ang**2.0 * noisy_cls["ells"] * (noisy_cls["ells"] + 1) / 2.0) for ang in thetab
        ]

        wtemp = [
            (arcmin_to_rad * cfg.specs["CMB_fwhm"][ind] * cfg.specs["CMB_temp_sens"][ind]) ** (-2.0)
            for ind in range(len(cfg.specs["CMB_fwhm"]))
        ]
        wpol = [
            (arcmin_to_rad * cfg.specs["CMB_fwhm"][ind] * cfg.specs["CMB_pol_sens"][ind]) ** (-2.0)
            for ind in range(len(cfg.specs["CMB_fwhm"]))
        ]

        TTnoise_chan = np.zeros((len(cfg.specs["CMB_fwhm"]), len(noisy_cls["ells"])))
        polnoise_chan = np.zeros((len(cfg.specs["CMB_fwhm"]), len(noisy_cls["ells"])))

        for ind in range(len(cfg.specs["CMB_fwhm"])):
            TTnoise_chan[ind, :] = Bell[ind][:] / wtemp[ind]
            polnoise_chan[ind, :] = Bell[ind][:] / wpol[ind]

        if len(cfg.specs["CMB_fwhm"]) > 1:
            TTnoise = np.array(
                [
                    self.sum_inv_squares(TTnoise_chan[:, ind])
                    for ind in range(len(noisy_cls["ells"]))
                ]
            )
            polnoise = np.array(
                [
                    self.sum_inv_squares(polnoise_chan[:, ind])
                    for ind in range(len(noisy_cls["ells"]))
                ]
            )
        else:
            TTnoise = TTnoise_chan[0, :]
            polnoise = polnoise_chan[0, :]

        for obs in self.observables:
            if obs == "CMB_T":
                noisy_cls[obs + "x" + obs] += TTnoise
            elif obs == "CMB_E" or obs == "CMB_B":
                noisy_cls[obs + "x" + obs] += polnoise

        return noisy_cls

    def sum_inv_squares(self, arr):
        res = np.sqrt(sum([i ** (-2) for i in arr]))

        return 1 / res

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

        covvec = []

        # Create indexes for data frame
        cols = []
        for o in self.observables:
            cols.append(o)

        for ind, ell in enumerate(noisy_cls["ells"]):
            covdf = pd.DataFrame(index=cols, columns=cols)
            covdf = covdf.fillna(0.0)

            for obs1, obs2 in product(self.observables, self.observables):
                covdf.at[obs1, obs2] = noisy_cls[obs1 + "x" + obs2][ind] / np.sqrt(
                    np.sqrt(cfg.specs["fsky_" + obs1] * cfg.specs["fsky_" + obs2])
                )

            covvec.append(covdf)

        return covvec

    def compute_covmat(self):
        tini = datetime.datetime.now().timestamp()
        allpars = {}
        allpars.update(self.cosmopars)

        obstring = ""
        for obs in self.observables:
            obstring = obstring + obs

        # Check free pars are in the fiducial
        for key in cfg.freeparams:
            if key not in allpars:
                print("ERROR: free param " + key + " does not have a fiducial value!")
                return None

        if not os.path.exists("./raw_results"):
            os.makedirs("./raw_results")

        # compute fiducial
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Computing fiducial")
        t1 = datetime.datetime.now().timestamp()

        # fiducial_csv = './raw_results/'+cfg.settings['outroot']+'_'+obstring+'_fiducial.csv'
        # if os.path.isfile(fiducial_csv):
        #    cls = pd.read_csv(fiducial_csv, index_col=0)
        # if True:  ## TODO: needs to be fixed to properly handle Pandas!
        cls = self.getcls(allpars)
        # fiducial = pd.DataFrame(cls, columns=cls.keys())
        # fiducial.to_csv(fiducial_csv, index=False)

        t2 = datetime.datetime.now().timestamp()
        if cfg.settings["feedback"] > 0:
            print("")
        if cfg.settings["feedback"] > 0:
            print("Fiducial generated in {:.2f} s".format(t2 - t1))

        # with open('./raw_results/'+cfg.settings['outroot']+'_'+obstring+'_fiducial.txt', 'w') as f:
        #    f.write(fiducial.to_string(header = True, index = False))

        # if cfg.settings['derivatives'] == '3PT': numcomp = 2
        # if cfg.settings['derivatives'] == 'STEM': numcomp = 11
        # estimate = datetime.timedelta(seconds=numcomp*(t2-t1)*len(cfg.freeparams))
        # if cfg.settings['feedback'] > 0: print('')
        # if cfg.settings['feedback'] > 0: print('Estimated time to finish {}'.format(estimate))
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

        # TODO: fix pandas stuff
        # noisy_fiducial = pd.DataFrame(noisy_cls, columns=noisy_cls.keys())
        # noisy_fiducial_csv = './raw_results/'+cfg.settings['outroot']+'_'+obstring+'_fiducial_noisy.csv'
        # noisy_fiducial.to_csv(noisy_fiducial_csv, index=False)
        # with open('./raw_results/'+cfg.settings['outroot']+'_'+obstring+'_fiducial_noisy.txt', 'w') as f:
        #    f.write(noisy_fiducial.to_string(header = True, index = False))

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
        allpars = {}
        allpars.update(self.cosmopars)

        derivs = dict()
        # needs to be fixed: TODO: not treating things properly as Pandas!
        # for par in cfg.freeparams:
        #    deriv_csv = './raw_results/'+cfg.settings['outroot']+'_'+obstring+'_derivative_'+par+'.csv'
        #    if os.path.isfile(deriv_csv):
        #        derivs[par] = pd.read_csv(deriv_csv, index_col=0)
        #        compute_derivs=False
        #    else:
        #        compute_derivs=True
        #        break
        self.print_info = print_info_specs  # Avoid printing info at each step
        compute_derivs = True
        if compute_derivs:
            tder1 = time()
            print(">> computing derivs >>")
            deriv_engine = fishderiv.derivatives(self.getcls, allpars)
            derivs = deriv_engine.result
            tder2 = time()
            # TODO: fix pandas stuff!!
            # for par in cfg.freeparams:
            #    deriv_csv = './raw_results/'+cfg.settings['outroot']+'_'+obstring+'_derivative_'+par+'.csv'
            #    derivative = pd.DataFrame(derivs[par], columns=derivs[par].keys())
            #    derivative.to_csv(deriv_csv, index=False)

            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="-->> Derivatives computed in ",
                time_ini=tder1,
                time_fin=tder2,
                instance=self,
            )

            # with open('./raw_results/'+cfg.settings['outroot']+'_'+obstring+'_derivative_'+par+'.txt', 'w') as f:
            #    f.write(derivative.to_string(header = True, index = False))
        self.derivs = derivs
        return self.derivs
