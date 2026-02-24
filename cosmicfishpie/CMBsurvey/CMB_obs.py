# -*- coding: utf-8 -*-
"""CMB angular power spectra (C_ell).

This module provides a small wrapper that computes CMB angular power spectra
(`CMB_T`, `CMB_E`, `CMB_B`) using the configured cosmology backend.

The numerical settings are taken from `cosmicfishpie.configs.config`:

- `cfg.specs["lmin_CMB"]`, `cfg.specs["lmax_CMB"]`
- beam/noise settings are handled in `cosmicfishpie.CMBsurvey.CMB_cov`
"""

from itertools import product
from time import time

import numpy as np
from joblib import Memory

import cosmicfishpie.configs.config as cfg
import cosmicfishpie.cosmology.cosmology as cosmology
from cosmicfishpie.utilities.utils import printing as upt

cachedir = "./cache"
memory = Memory(cachedir, verbose=0)


class ComputeCls:
    """Compute CMB angular power spectra for the configured observables.

    Parameters
    ----------
    cosmopars : dict
        Cosmological parameters in CosmicFishPie basis (e.g. `Omegam`, `h`, ...).
    print_info_specs : bool, optional
        If True, print the numerical specifications currently stored in
        `cfg.specs`.

    Notes
    -----
    Results are stored in the `result` attribute after calling `compute_all()`.
    The result is a dictionary with at least:

    - `ells`: 1D array of multipoles
    - `<obs1>x<obs2>` arrays for each requested combination (e.g. `CMB_TxCMB_T`)
    """

    def __init__(self, cosmopars, print_info_specs=False):
        self.feed_lvl = cfg.settings["feedback"]

        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=0,
            text="-> Started Cls calculation",
            instance=self,
        )

        tcosmo1 = time()
        self.cosmopars = cosmopars
        self.cosmo = cosmology.cosmo_functions(cosmopars, cfg.input_type)
        tcosmo2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
            text="---> Cosmological functions obtained in ",
            instance=self,
            time_ini=tcosmo1,
            time_fin=tcosmo2,
        )

        # MM: CMB lensing to be added (optional?)
        self.observables = []
        for key in cfg.obs:  # LENSING TO BE ADDED
            if key in ["CMB_T", "CMB_E", "CMB_B"]:
                self.observables.append(key)

        cfg.specs["ellmax"] = cfg.specs["lmax_CMB"]
        cfg.specs["ellmin"] = cfg.specs["lmin_CMB"]

        if print_info_specs:
            self.print_numerical_specs()

    def compute_all(self):
        """Compute all requested CMB spectra and store them in `self.result`."""
        tini = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=0,
            text="-> Computing CMB spectra ",
            instance=self,
        )

        self.result = self.computecls()

        tend = time()

        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
            text="--> Total Cls computation performed in : ",
            time_ini=tini,
            time_fin=tend,
            instance=self,
        )

    def print_numerical_specs(self):
        """Print the contents of `cfg.specs` (debug helper)."""
        print("***")
        print("Numerical specifications: ")
        for key in cfg.specs:
            print(key + " = " + str(cfg.specs[key]))
        print("***")

    def computecls(self):
        """Return a dictionary of CMB C_ell arrays.

        Returns
        -------
        dict
            Dictionary with `ells` and `obs1xobs2` arrays.
        """

        cls = {"ells": np.arange(cfg.specs["ellmin"], cfg.specs["ellmax"])}

        # Compute all spectra combinations requested by cfg.obs.
        for obs1, obs2 in product(self.observables, self.observables):
            cls[obs1 + "x" + obs2] = self.cosmo.cmb_power(
                cfg.specs["ellmin"], cfg.specs["ellmax"], obs1, obs2
            )

        return cls
