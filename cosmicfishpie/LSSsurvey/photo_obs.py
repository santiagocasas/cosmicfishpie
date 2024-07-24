# -*- coding: utf-8 -*-
"""CLS

This module contains cls calculations (only LSS atm).

"""
from itertools import product
from time import time

import numpy as np
from joblib import Memory
from scipy import integrate
from scipy.interpolate import interp1d

import cosmicfishpie.configs.config as cfg
import cosmicfishpie.cosmology.cosmology as cosmology
import cosmicfishpie.cosmology.nuisance as nuisance
import cosmicfishpie.LSSsurvey.photo_window as photo_window
from cosmicfishpie.utilities.utils import printing as upt

cachedir = "./cache"
memory = Memory(cachedir, verbose=0)


# @memory.cache


def memo_integral_efficiency(i, ngal_func, comoving_func, z, zint_mat, diffz):
    """function to do the integration over redshift that shows up in the lensing kernel

    Parameters
    ----------
    i             : int
                    index of the redshift bin for which the lensing kernel should be computed
    ngal_func     : callable
                    callable function that returns the number density distribution of galaxies. Should be a function of the redshift bin index i as an int and the redshift z as a numpy.ndarray
    comoving_func : callable
                    callable function that returns the comoving distance. Should be a function of the redshift z as a numpy.ndarray
    z             : numpy.ndarray
                    1d array of all redshifts z that the result of the integral should use as interpolated over.
    zint_mat      : numpy.ndarray
                    2d array of redshifts z that the integral should use as integration points. The first row must coincide with z.
    diffz         : numpy.ndarray
                    2d array of the separation of integration points

    Returns
    -------
    callable
        callable function that receives a numpy.ndarray of requested redshifts and returns the lensing efficiency for the i-th bin as a numpy.ndarray

    Note
    ----
    This function is deprecated. Use `faster_integral_efficiency` instead.
    """
    intg_mat = np.array(
        [
            (
                ngal_func(zint_mat[zii], i, 'WL')
                * (1 - comoving_func(zint_mat[zii, 0]) / comoving_func(zint_mat[zii]))
            )
            for zii in range(len(zint_mat))
        ]
    )
    integral_really = integrate.trapezoid(intg_mat, dx=diffz, axis=1)
    intp = interp1d(z, integral_really, kind="cubic")
    return intp


def faster_integral_efficiency(i, ngal_func, comoving_func, zarr):
    """function to do the integration over redshift that shows up in the lensing kernel

    Parameters
    ----------
    i             : int
                    index of the redshift bin for which the lensing kernel should be computed
    ngal_func     : callable
                    callable function that returns the number density distribution of galaxies. Should be a function of the redshift bin index i as an int and the redshift z as a numpy.ndarray
    comoving_func : callable
                    callable function that returns the comoving distance. Should beb a function of the redshift z as a numpy.ndarray
    zarr         : numpy.ndarray
                    1d array of all redshifts z that should be used as integration points.

    Returns
    -------
    callable
        callable function that receives a numpy.ndarray of requested redshifts and returns the lensing efficiency for the i-th bin as a numpy.ndarray
    """
    zprime = zarr[:, None]
    wintgd = ngal_func(zprime, i, 'WL') * (1.0 - comoving_func(zarr) / comoving_func(zprime))
    witri = np.tril(wintgd)
    wint = integrate.trapezoid(witri, zarr, axis=0)
    intp = interp1d(zarr, wint, kind="cubic")
    return intp


class ComputeCls:
    def __init__(
        self, cosmopars, photopars, IApars, biaspars, print_info_specs=False, fiducial_cosmo=None
    ):
        """Main class to obtain the angular power spectrum of the photometric probe.

        Parameters
        ----------
        cosmopars        : dict
                           a dictionary containing the cosmological parameters
        photopars        : dict
                           a dictionary containing specifications for the window function's galaxy distribution
        IApars           : dict
                           a dictionary containing the specifications for the intrinsic alignment effect in cosmic shear
        biaspars         : dict
                           a dictionary containing the specifications for the galaxy biases
        print_info_specs : bool
                           If True will print the numerical specifications of the computation. Defaults to False
        fiducial_cosmo   : cosmicfishpie.cosmology.cosmology.cosmo_functions, optional
                           An instance of `cosmo_functions` of the fiducial cosmology.

        Attributes
        ----------
        cosmopars        : dict
                           a dictionary containing the cosmological parameters
        photopars        : dict
                           a dictionary containing specifications for the window function's galaxy distribution
        IApars           : dict
                           a dictionary containing the specifications for the intrinsic alignment effect in cosmic shear
        biaspars         : dict
                           a dictionary containing the specifications for the galaxy biases
        cosmo            : cosmicfishpie.cosmology.cosmo_functions
                           An instance of `cosmo_functions`. Will either contain the fiducial cosmology if passed or compute from the cosmopars dict
        observables      : list
                           a list of the observables that the angular power spectrum is computed for
        nuisance         : cosmicfishpie.cosmology.nuisance.Nuisance
                           An instance of `Nuisance` that contains the relevant modeling of nuisance parameters
        window           : cosmicfishpie.LSSsurvey.photo_window.GalaxyPhotoDist
                           An instance of `GalaxyPhotoDist` containing the galaxy distribution needed to compute the window functions
        tracer           : str
                           What Power spectrum should be used when calculating the angular power spectrum of galaxy clustering. Either "matter" or "clustering"
        binrange         : range
                           a range with all the bins asked for in the survey specifications
        zsamp            : int
                           how many individual values of the redshifts are used in the internal calculations
        ellsamp          : int
                           how many individual values of the multipoles are used in the internal calculations
        ell              : numpy.ndarray
                           array containing the multipoles for which the angular power spectrum is computed.
        z_min            : float
                           minimum redshift of the probes
        z_max            : float
                           maximum redshift of the probes
        z                : numpy.ndarray
                           array containing the redshifts used in the internal calculations
        dz               : numpy.ndarray
                           array containing the numerical distance of the redshifts in z
        """
        self.feed_lvl = cfg.settings["feedback"]
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=0,
            text="-> Started Cls calculation",
            instance=self,
        )

        tcosmo1 = time()
        self.cosmopars = cosmopars
        if fiducial_cosmo is None:
            self.cosmo = cosmology.cosmo_functions(cosmopars, cfg.input_type)
        else:
            self.cosmo = fiducial_cosmo
        tcosmo2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=1,
            text="---> Cosmological functions obtained in ",
            instance=self,
            time_ini=tcosmo1,
            time_fin=tcosmo2,
        )
        self.observables = []
        self.binrange     = []
        for key in cfg.obs:
            if key in ["GCph", "WL"]:
                self.observables.append(key)
                if key == 'GCph':
                    self.binrange.append(cfg.specs["binrange_GCph"])
                elif key == 'WL':
                    self.binrange.append(cfg.specs["binrange_WL"])

        self.binrange_WL = cfg.specs["binrange_WL"]
        self.binrange_GCph = cfg.specs["binrange_GCph"]

        tnuis1 = time()
        self.biaspars = biaspars
        self.IApars = IApars
        self.nuisance = nuisance.Nuisance()
        # if 'GCph' in self.observables: self.bfunction = self.nuisance.bias(self.biaspars)
        if "WL" in self.observables:
            self.IAvalue = self.nuisance.IA(self.IApars, self.cosmo)
        tnuis2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="---> Nuisance functions obtained in ",
            instance=self,
            time_ini=tnuis1,
            time_fin=tnuis2,
        )

        tngal1 = time()
        self.photopars = photopars
        self.window = photo_window.GalaxyPhotoDist(self.photopars)
        tngal2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="---> Galaxy Photometric distributions obtained in ",
            instance=self,
            time_ini=tngal1,
            time_fin=tngal2,
        )

        if "GCph" in self.observables and "WL" in self.observables:
            cfg.specs["ellmax"] = max(cfg.specs["lmax_GCph"], cfg.specs["lmax_WL"])
            cfg.specs["ellmin"] = min(cfg.specs["lmin_GCph"], cfg.specs["lmin_WL"])
        elif "GCph" in self.observables:
            cfg.specs["ellmax"] = cfg.specs["lmax_GCph"]
            cfg.specs["ellmin"] = cfg.specs["lmin_GCph"]
        elif "WL" in self.observables:
            cfg.specs["ellmax"] = cfg.specs["lmax_WL"]
            cfg.specs["ellmin"] = cfg.specs["lmin_WL"]
        else:
            raise ValueError("Observables not specified correctly")

        self.tracer = cfg.settings["GCph_Tracer"]
#        self.binrange = cfg.specs["binrange"]
        self.zsamp = int(round(200 * cfg.settings["accuracy"]))
        self.ellsamp = int(round(100 * cfg.settings["accuracy"]))
        self.ell = np.logspace(
            np.log10(cfg.specs["ellmin"]), np.log10(cfg.specs["ellmax"] + 10), num=self.ellsamp
        )

        #MMmod: DR1
        if cfg.input_type == "camb":
            self.z_min = np.min([cfg.specs["z_bins_GCph"][0],cfg.specs["z_bins_WL"][0]])
            self.z_max = np.max([cfg.specs["z_bins_GCph"][-1],cfg.specs["z_bins_WL"][-1]])  # +0.5
        if cfg.input_type == "class":
            self.z_min = np.min([cfg.specs["z_bins_GCph"][0],cfg.specs["z_bins_WL"][0]])
            self.z_max = np.max([cfg.specs["z_bins_GCph"][-1],cfg.specs["z_bins_WL"][-1]])
        elif cfg.input_type == "external":
            self.z_min = np.max([cfg.specs["z_bins_GCph"][0],cfg.specs["z_bins_WL"][0], self.cosmo.results.zgrid[0]])
            self.z_max = np.min([cfg.specs["z_bins_GCph"][-1],cfg.specs["z_bins_WL"][-1], self.cosmo.results.zgrid[-1]])
        # +1 to go beyond the bin limit
        self.z = np.linspace(self.z_min, self.z_max, self.zsamp)
        self.dz = np.mean(np.diff(self.z))

        if print_info_specs:
            self.print_numerical_specs()

    def compute_all(self):
        """Main function to compute the angular power spectrum. Will first compute the limber approximated angular power spectrum and the window functions for both probes. From that it will obtain the angular power spectrum.

        Note
        -----
        This function does not return the calculated quantities. Rather they are found in new attributes of the object.

        The result of the Limber approximated power spectra are found in the new attributes `Pell` and `sqrtPell`

        The window functions are found in the new attributes `GC` if galaxy clustering is asked for, and `WL` if cosmic sheer is asked for

        The angular power spectrum is found in the new attribute `result`
        """
        tini = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=0,
            text="-> Computing power spectra and kernels ",
            instance=self,
        )

        tplim1 = time()
        self.P_limber()
        self.sqrtP_limber()
        tplim2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="---> Computed P_limber in ",
            time_ini=tplim1,
            time_fin=tplim2,
            instance=self,
        )

        tkern1 = time()
        self.compute_kernels()
        tkern2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="---> Computed Kernels in: ",
            time_ini=tkern1,
            time_fin=tkern2,
            instance=self,
        )

        tcls1 = time()
        self.result = self.computecls()
        tcls2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="---> Computed Cls in: ",
            time_ini=tcls1,
            time_fin=tcls2,
            instance=self,
        )
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
        """prints the numerical specifications of the internal computations"""
        print("***")
        print("Numerical specifications: ")
        print("WL ell max = ", str(cfg.specs["lmax_WL"]))
        print("GCph ell max = ", str(cfg.specs["lmax_GCph"]))
        print("ell min = ", str(cfg.specs["ellmin"]))
        print("ell max = ", str(cfg.specs["ellmax"]))
        print("ell sampling: ", str(self.ellsamp))
        print("z sampling: ", str(self.zsamp))
        print("z_min : ", str(self.z_min))
        print("z_max : ", str(self.z_max))
        print("z_max : ", str(self.z_max))
        print("delta_z : ", str(self.dz))
        print("***")

    def P_limber(self):
        """
        Calculates the Limber-approximated power spectrum.
        This is done for a range of redshift values and multipoles.
        Will add MG effects to WL and(!) GCph if asked for.

        Note
        -----
        This does not return the power spectra. The results are stored in the attribute `Pell`
        """
        self.Pell = np.zeros((self.zsamp, self.ellsamp))
        # TO BE PYTHONIZED
        chi = self.cosmo.comoving(self.z)
        # PYTHONIZE THIS HORRIBLE THING
        for ell, zin in product(range(self.ellsamp), range(self.zsamp)):
            kappa = (self.ell[ell] + 0.5) / chi[zin]
            if kappa <= cfg.specs["kmax"]:
                self.Pell[zin, ell] = (
                    (self.cosmo.SigmaMG(self.z[zin], kappa) ** 2)
                    * self.cosmo.matpow(
                        self.z[zin], kappa, nonlinear=cfg.settings["nonlinear_photo"]
                    )
                    / (chi[zin] ** 2.0)
                )
            else:
                self.Pell[zin, ell] = 0.0
        return None

    def sqrtP_limber(self):
        """
        Calculates the square root of the Limber-approximated power spectrum
        for weak lensing (WL) and galaxy clustering photometric (GCph) observables.
        This is done for a range of redshift values and multipoles.
        Depending on the tracer asked for in 'GCph_Tracer' will use 'matter' or 'clustering' observable GCph.
        Will add MG effects to WL if asked for.

        Note
        -----
        This does not return the power spectra. The results are stored in the attribute `sqrtPell`
        """
        # Sakr Fix June 2023
        # self.sqrtPell = {'WL'  :np.zeros((self.zsamp,self.ellsamp)),
        #             'WL_IA'  :np.zeros((self.zsamp,self.ellsamp)),
        #             'GCph':np.zeros((self.zsamp,self.ellsamp)),
        #             'GCph_IA':np.zeros((self.zsamp,self.ellsamp))}
        self.sqrtPell = {
            "WL": np.zeros((self.zsamp, self.ellsamp)),
            "WL_IA": np.zeros((self.zsamp, self.ellsamp)),
            "GCph": np.zeros((self.zsamp, self.ellsamp)),
            "GCph_IA": np.zeros((self.zsamp, self.ellsamp)),
        }

        chi = self.cosmo.comoving(self.z)
        kappa = (self.ell[:, None] + 1 / 2) / chi
        index_pknn = np.array(np.where(kappa < cfg.specs["kmax"])).T

        for ell, zin in index_pknn:
            self.sqrtPell["WL"][zin, ell] = (
                self.cosmo.SigmaMG(self.z[zin], kappa[ell, zin])
                * np.sqrt(
                    self.cosmo.matpow(
                        self.z[zin], kappa[ell, zin], nonlinear=cfg.settings["nonlinear"]
                    )
                )
            ) / chi[zin]
            # Sakr Fix June 2023
            self.sqrtPell["WL_IA"][zin, ell] = (
                1.0
                * np.sqrt(
                    self.cosmo.matpow(
                        self.z[zin], kappa[ell, zin], nonlinear=cfg.settings["nonlinear"]
                    )
                )
                / chi[zin]
            )
            self.sqrtPell["GCph"][zin, ell] = (
                np.sqrt(
                    self.cosmo.matpow(
                        self.z[zin],
                        kappa[ell, zin],
                        nonlinear=cfg.settings["nonlinear"],
                        tracer=self.tracer,
                    )
                )
                / chi[zin]
            )
        return None

    def galaxy_kernel(self, z, i):
        """Calculates the GCph kernel function

        Parameters
        ----------
        z : numpy.ndarray
            array containing the redshifts for which the kernel should be computed for
        i : int
            index of the redshift bin

        Returns
        -------
        numpy.ndarray
            Value of the galaxy clustering kernel at redshift z for bin i

        Note
        -----
        Implements the following equation:

        .. math::

            W_i^{GCph} = b(z) \\frac{n_i(z)}{\\bar{n}(z)} H(z)

        """
        tgcstart = time()
        # Wgc = self.window.norm_ngal_photoz(z,i) * np.array([self.nuisance.bias(self.biaspars, i)(z) * \
        # Wgc = self.window.norm_ngal_photoz(z,i) *
        # np.array([self.biaspars['b{:d}'.format(i)] * \
        Wgc = self.window.norm_ngal_photoz(z, i, 'GCph') * self.cosmo.Hubble(z)
        # Wgc = self.window.norm_ngal_photoz(z,i) * self.nuisance.bias(self.biaspars, i)(z) * \
        #                                          self.cosmo.Hubble(z)

        tgcend = time()
        if self.feed_lvl >= 3:
            print("    ...done bin {} for GCph in {:.2f} s".format(i, tgcend - tgcstart))

        return Wgc

    def lensing_kernel(self, z, i):
        """Computes the photometric cosmic shear kernel function

        Parameters
        ----------
        z : numpy.ndarray
            array containing the redshifts for which the kernel should be computed for
        i : int
            index of the redshift bin

        Returns
        -------
        numpy.ndarray
            Value of the cosmic shearing kernel at redshift z for bin i

        Note
        -----
        Implements the following equation:

        .. math::
         W_i^{WL} = W_i^{IA}+\\frac{3}{2}\left(\\frac{H_0}{c}\\right)^2\Omega_{m,0}(1+z)r(z)
         \int_z^{z_{\\rm max}}{dz' \\frac{n_i(z')}{\\bar{n}(z)}\left[1-\\frac{r(z)}{r(z')}\\right]}
        """

        twlstart = time()
        # computing lensing kernel integral
        prefac = (3.0 / 2.0) * self.cosmo.Hubble(0.0) ** 2.0 * self.cosmo.Omegam_of_z(0.0)
        # WARNING! this prefactor needs to be generalized (Sigma, non CDM...)
        Wwl = np.array(
            [prefac * (1 + zi) * self.cosmo.comoving(zi) * self.efficiency[i](zi) for zi in z]
        )

        # Adding Intrinsic alignment
        WIA = self.window.norm_ngal_photoz(z, i, 'WL') * np.array(
            [self.IAvalue(zi) * self.cosmo.Hubble(zi) for zi in z]
        )
        # Sakr Fix June 2023
        # kernel = Wwl+WIA

        twlend = time()

        if self.feed_lvl >= 3:
            print("    ...done bin {} for WL in {:.2f} s".format(i, twlend - twlstart))

        # Sakr Fix June 2023
        # return Wwl
        return Wwl, WIA

    def integral_efficiency(self, i):
        """computes the integral that enters the lensing kernel for a given redshift bin

        Parameters
        ----------
        i : int
            index of the redshift bin for which the lensing efficiency should be calculated

        Returns
        -------
        callable
            callable function that receives a numpy.ndarray of requested redshifts and returns the lensing efficiency for the i-th bin as a numpy.ndarray
        """
        # expensive calculation doesn't need to
        # be performed if cosmopars and photopars are the same
        zint_mat = np.linspace(self.z, self.z[-1], self.zsamp)
        zint_mat = zint_mat.T
        z = self.z
        ngal_func = self.window.norm_ngal_photoz
        comoving_func = self.cosmo.comoving
        intp = faster_integral_efficiency(i, ngal_func, comoving_func, z)
        return intp

    def lensing_efficiency(self):
        """computes the integral that enters the lensing kernel for all redshift bins

        Returns
        -------
        list
            list of callable functions that give the lensing efficiency for each bin
        """
        teffstart = time()
        efficiency = [self.integral_efficiency(i) for i in self.binrange_WL]
        efficiency.insert(0, None)
        teffend = time()
        if self.feed_lvl >= 3:
            print("    ...lensing efficiency computation took {:.2f} s".format(teffend - teffstart))
        return efficiency

    def compute_kernels(self):
        """function to compute all the needed window functions.

        Note
        -----
        This does not return the window functions. They are found in the new attributes `GC` if galaxy clustering is asked for, and `WL` if cosmic sheer is asked for
        """
        if "GCph" in self.observables:
            self.GC = [
                interp1d(self.z, self.galaxy_kernel(self.z, ind), kind="cubic")
                for ind in self.binrange_GCph
            ]
            self.GC.insert(0, None)
        if "WL" in self.observables:
            self.efficiency = self.lensing_efficiency()
            # Sakr Fix June 2023
            # self.WL         = [interp1d(self.z,self.lensing_kernel(self.z,ind), kind='cubic') for ind in self.binrange]
            self.WL = [
                interp1d(self.z, self.lensing_kernel(self.z, ind)[0], kind="cubic")
                for ind in self.binrange_WL
            ]
            self.WL.insert(0, None)
            # Sakr Fix June 2023
            self.WL_IA = [
                interp1d(self.z, self.lensing_kernel(self.z, ind)[1], kind="cubic")
                for ind in self.binrange_WL
            ]
            self.WL_IA.insert(0, None)
        return None

    def genwindow(self, z, obs, i):
        """generic function to obtain the window functions for the different observables.

        Parameters
        ----------
            z     : numpy.ndarray
                    array of redshifts for which the window function should be computed
            obs   : str
                    name of the observable (GC or WL)
            i     : int
                    integer of the redshift bin the window should be computed for

        Returns
        -------
        numpy.ndarray
            Values of the window function for the observable obs, for redshift bin i, and at redshifts z.
        """

        win = []
        win_IA = []

        if obs == "GCph":
            bz = self.nuisance.gcph_bias(self.biaspars, i)(z)
            win = self.GC[i](z) * bz
            win_IA = self.GC[i](z) * 0.0
        elif obs == "WL":
            win = self.WL[i](z)
            win_IA = self.WL_IA[i](z)

        return win, win_IA

    def computecls(self):
        """Function to compute the angular power spectrum for all observables, redshift bins and multipoles.

        Returns
        -------
        dict
            a dictionary containing all auto and cross correlation angular power spectra. Its keys are formatted to have an array of the power spectra in 'X ixY j'. Also the multipoles for which the angular power spectra were computed for are found in 'ells'.

        Note
        -----
        Implements the following equation:

            .. math::
                C_{i,j}^{X,Y}(\\ell) = c \\int \\mathrm{d}z \\frac{W_i^X (z)W_j^Y (z)}{ H(z) r^2(z)}
                P_{\\delta \\delta} \\big[\\frac{\\ell + 1/2}{r(z)} , z \\big]
        """

        if self.feed_lvl > 1:
            print("")
        if self.feed_lvl > 1:
            print("    Computing Cls integral for {}".format(self.observables))
        # full_ell = np.linspace(cfg.specs['ellmin'],cfg.specs['ellmax'],cfg.specs['ellmax']-cfg.specs['ellmin'])
        # full_ell = np.round(full_ell, 0)
        full_ell = self.ell

        cls = {"ells": full_ell}

        hub = self.cosmo.Hubble(self.z)

        tstart = time()
        tcell = time()

        # PYTHONIZE THIS HORRIBLE THING
        for obs1, obs2, bin1, bin2 in product(
                self.observables, self.observables, self.binrange[0], self.binrange[1] #MMmod: BEWARE! THIS IS UGLY!
        ):
            clinterp = self.clsintegral(obs1, obs2, bin1, bin2, hub)

            finalcls = np.zeros((len(full_ell)))
            for ind, lval in enumerate(full_ell):
                if (cfg.specs["lmin_" + obs1] <= lval <= cfg.specs["lmax_" + obs1]) and (
                    cfg.specs["lmin_" + obs2] <= lval <= cfg.specs["lmax_" + obs2]
                ):
                    finalcls[ind] = clinterp(lval)

            cls[obs1 + " " + str(bin1) + "x" + obs2 + " " + str(bin2)] = finalcls

            tbin = time()
            if self.feed_lvl > 2:
                print("")
            if self.feed_lvl > 2:
                print(
                    "    ...{} {} x {} {} done in {:.2f} s".format(
                        obs1, bin1, obs2, bin2, tbin - tcell
                    )
                )
            tcell = tbin

        tend = time()
        if self.feed_lvl > 1:
            print("")
        if self.feed_lvl > 1:
            print("    Cls integral computation took {:.2f} s".format(tend - tstart))

        return cls

    def clsintegral(self, obs1, obs2, bin1, bin2, hub):
        """function to obtain the angular power spectrum as an function of the multipole.

        Parameters
        ----------
        obs1 : str
               name of the observable of the redshift bin bin1 (GC or WL)
        obs2 : str
               name of the observable of the redshift bin bin2 (GC or WL)
        bin1 : int
               index of the redshift bin at which the observable obs1 is measured
        bin2 : int
               index of the redshift bin at which the observable obs2 is measured
        hub  : callable
               function that should be passed a numpy.ndarray of redshifts returns the hubble expansion rate at these redshifts.

        Returns
        -------
        callable
            Function that receives an array multipole and returns the angular power spectrum for these multipoles.
        """
        mask1 = (self.ell >= cfg.specs["lmin_" + obs1]) & (self.ell <= cfg.specs["lmax_" + obs1])
        mask2 = (self.ell >= cfg.specs["lmin_" + obs2]) & (self.ell <= cfg.specs["lmax_" + obs2])

        # Sakr Fix June 2023
        # pz_arr = self.genwindow(self.z,obs1,bin1)*self.genwindow(self.z,obs2,bin2)/hub
        # intgn  = self.sqrtPell[obs1]* self.sqrtPell[obs2] * pz_arr[:,np.newaxis]
        intgn = (
            self.sqrtPell[obs1]
            * self.genwindow(self.z, obs1, bin1)[0][:, np.newaxis]
            / np.sqrt(hub)[:, np.newaxis]
            + self.sqrtPell[obs1 + "_IA"]
            * self.genwindow(self.z, obs1, bin1)[1][:, np.newaxis]
            / np.sqrt(hub)[:, np.newaxis]
        ) * (
            self.sqrtPell[obs2]
            * self.genwindow(self.z, obs2, bin2)[0][:, np.newaxis]
            / np.sqrt(hub)[:, np.newaxis]
            + self.sqrtPell[obs2 + "_IA"]
            * self.genwindow(self.z, obs2, bin2)[1][:, np.newaxis]
            / np.sqrt(hub)[:, np.newaxis]
        )

        clint = integrate.trapezoid(intgn, dx=self.dz, axis=0)

        clint[~mask1] = 0
        clint[~mask2] = 0

        clinterp = interp1d(self.ell, clint, kind="linear")  # 'cubic')

        return clinterp
