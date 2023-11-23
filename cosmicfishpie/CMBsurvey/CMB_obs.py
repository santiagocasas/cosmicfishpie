# -*- coding: utf-8 -*-
"""CLS

This module contains cls calculations (only LSS atm).

"""
from itertools import product
from time import time

import numpy as np
from joblib import Memory

import cosmicfishpie.cosmology.cosmology as cosmology
import cosmicfishpie.fishermatrix.config as cfg
from cosmicfishpie.utilities.utils import printing as upt

cachedir = './cache'
memory = Memory(cachedir, verbose=0)


class ComputeCls:

    def __init__(self, cosmopars, print_info_specs=False):

        self.feed_lvl = cfg.settings['feedback']

        upt.time_print(feedback_level=self.feed_lvl, min_level=0,
                       text='-> Started Cls calculation',
                       instance=self)

        tcosmo1 = time()
        self.cosmopars = cosmopars
        self.cosmo = cosmology.cosmo_functions(cosmopars, cfg.input_type)
        tcosmo2 = time()
        upt.time_print(feedback_level=self.feed_lvl, min_level=1,
                       text='---> Cosmological functions obtained in ',
                       instance=self, time_ini=tcosmo1, time_fin=tcosmo2)

        # MM: CMB lensing to be added (optional?)
        self.observables = []
        for key in cfg.obs:  # LENSING TO BE ADDED
            if key in ['CMB_T', 'CMB_E', 'CMB_B']:
                self.observables.append(key)

        cfg.specs['ellmax'] = cfg.specs['lmax_CMB']
        cfg.specs['ellmin'] = cfg.specs['lmin_CMB']

        if print_info_specs:
            self.print_numerical_specs()

    def compute_all(self):

        tini = time()
        upt.time_print(feedback_level=self.feed_lvl, min_level=0,
                       text='-> Computing CMB spectra ',
                       instance=self)

        self.result = self.computecls()

        tend = time()

        upt.time_print(feedback_level=self.feed_lvl, min_level=1,
                       text='--> Total Cls computation performed in : ',
                       time_ini=tini, time_fin=tend, instance=self)

    def print_numerical_specs(self):
        print("***")
        print("Numerical specifications: ")
        for key in cfg.specs:
            print(key + ' = ' + str(cfg.specs[key]))
        print("***")

    def computecls(self):
        """Cls computation

        Parameters
        ----------
        ell   : float
                multipole
        X     : str
                first observable
        Y     : str
                second observable
        i     : int
                first bin
        j     : int
                second bin

        Returns
        -------
        float
            Value of Cl


        """

        cls = {'ells': np.arange(cfg.specs['ellmin'], cfg.specs['ellmax'])}

        # PYTHONIZE THIS HORRIBLE THING
        for obs1, obs2 in product(self.observables, self.observables):

            cls[obs1 + 'x' + obs2] = self.cosmo.cmb_power(
                cfg.specs['ellmin'], cfg.specs['ellmax'], obs1, obs2)

        return cls
