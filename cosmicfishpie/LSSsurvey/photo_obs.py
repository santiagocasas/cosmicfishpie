# -*- coding: utf-8 -*-
"""CLS

This module contains cls calculations (only LSS atm).

"""
import os
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


# Experimental optimization toggles (default: enabled). Users can disable via env.
def _env_flag(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val not in ("0", "false", "False")


_USE_FAST_EFF = _env_flag("COSMICFISH_FAST_EFF", True)
_USE_FAST_P = _env_flag("COSMICFISH_FAST_P", True)
_USE_FAST_KERNEL = _env_flag("COSMICFISH_FAST_KERNEL", True)


# @memory.cache


def memo_integral_efficiency(i, ngal_func, comoving_func, z, zint_mat, diffz):
    """Legacy O(N^2) implementation of the efficiency integral.

    Notes
    -----
    - Deprecated: prefer `faster_integral_efficiency` (vectorized) or
      `much_faster_integral_efficiency` (O(N)).
    - This version performs a per-row trapezoidal integration on the original
      redshift grid to ensure numerically correct spacing. The `zint_mat` and
      `diffz` arguments are ignored for correctness (they were prone to
      inconsistent spacing per row).

    Parameters
    ----------
    i : int
        Bin index.
    ngal_func : callable
        Function returning n(z') given (z', i, kind).
    comoving_func : callable
        Function returning r(z).
    z : numpy.ndarray
        1D redshift grid.
    zint_mat : numpy.ndarray
        Ignored (kept for backward compatibility).
    diffz : numpy.ndarray or float
        Ignored (kept for backward compatibility).

    Returns
    -------
    callable
        Interpolating function over z for the efficiency.
    """
    chi = comoving_func(z)
    integral = np.empty_like(z)
    # For each z_k integrate from z_k to z_max on the native grid
    for k in range(z.size):
        z_row = z[k:]
        # integrand: n(z') * (1 - r(z_k)/r(z'))
        y_row = ngal_func(z_row, i, "WL") * (1.0 - chi[k] / comoving_func(z_row))
        # integrate with correct x per row
        integral[k] = integrate.trapezoid(y_row, x=z_row)
    return interp1d(z, integral, kind="cubic")


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
    wintgd = ngal_func(zprime, i, "WL") * (1.0 - comoving_func(zarr) / comoving_func(zprime))
    witri = np.tril(wintgd)
    wint = integrate.trapezoid(witri, zarr, axis=0)
    intp = interp1d(zarr, wint, kind="cubic")
    return intp


def much_faster_integral_efficiency(i, ngal_func, comoving_func, zarr):
    """O(N) algorithm for the lensing efficiency integral using backward
    cumulative trapezoid integrals.

    Parameters
    ----------
    i             : int
                    index of the redshift bin for which the lensing kernel should be computed
    ngal_func     : callable
                    callable function that returns the number density distribution of galaxies. Should be a
                    function of the redshift bin index i as an int and the redshift z as a numpy.ndarray
    comoving_func : callable
                    callable function that returns the comoving distance. Should be a function of the redshift z
                    as a numpy.ndarray
    zarr          : numpy.ndarray
                    1d array of all redshifts z that should be used as integration points.

    Returns
    -------
    callable
        callable function that receives a numpy.ndarray of requested redshifts and returns the lensing efficiency
        for the i-th bin as a numpy.ndarray
    """
    # Grid and step
    z = zarr
    if z.size < 2:
        return interp1d(z, np.zeros_like(z), kind="cubic", fill_value="extrapolate")
    dz = float(np.mean(np.diff(z)))

    # Required functions on grid
    chi = comoving_func(z)
    ngal = ngal_func(z, i, "WL")
    ngal_over_chi = ngal / chi

    # Backward trapezoid cumulative integral: I[k] = ∫_{z_k}^{z_max} f(z') dz'
    def backward_trapz(y, dx):
        if y.size < 2:
            return np.zeros_like(y)
        yr = y[::-1]
        seg = 0.5 * (yr[:-1] + yr[1:]) * dx
        cumsum = np.concatenate([[0.0], np.cumsum(seg)])
        return cumsum[::-1]

    I1 = backward_trapz(ngal, dz)
    I2 = backward_trapz(ngal_over_chi, dz)
    eff = I1 - chi * I2

    return interp1d(z, eff, kind="cubic")


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
            min_level=2,
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
            min_level=2,
            text="---> Cosmological functions obtained in ",
            instance=self,
            time_ini=tcosmo1,
            time_fin=tcosmo2,
        )
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
            min_level=3,
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
            min_level=3,
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
        if cfg.settings["ell_sampling"] == "accuracy":
            self.ellsamp = int(round(100 * cfg.settings["accuracy"]))
        else:
            if isinstance(cfg.settings["ell_sampling"], int):
                self.ellsamp = cfg.settings["ell_sampling"]
            else:
                raise ValueError("ell_sampling should be an integer or 'accuracy'")
        self.ell = np.logspace(
            np.log10(cfg.specs["ellmin"]), np.log10(cfg.specs["ellmax"] + 10), num=self.ellsamp
        )

        self.z_min = np.min([cfg.specs["z_bins_GCph"][0], cfg.specs["z_bins_WL"][0]])
        self.z_max = np.max([cfg.specs["z_bins_GCph"][-1], cfg.specs["z_bins_WL"][-1]])
        self.z = np.linspace(self.z_min, self.z_max, self.zsamp)
        self.dz = float(np.mean(np.diff(self.z)))
        # Precompute comoving distance (major hotspot) once; reuse everywhere
        self._chi_z = self.cosmo.comoving(self.z)
        # Lazy caches for other frequently used cosmological functions (filled on demand)
        self._hubble_z = None
        self._IA_z = None

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
            min_level=2,
            text="-> Computing power spectra and kernels ",
            instance=self,
        )

        tplim1 = time()
        self.P_limber()
        self.sqrtP_limber()
        tplim2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
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
            min_level=3,
            text="---> Computed Kernels in: ",
            time_ini=tkern1,
            time_fin=tkern2,
            instance=self,
        )

        tcls1 = time()
        self.result = self.computecls_vectorized()
        tcls2 = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
            text="---> Computed Cls in: ",
            time_ini=tcls1,
            time_fin=tcls2,
            instance=self,
        )
        tend = time()

        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="--> Total Cls computation performed in : ",
            time_ini=tini,
            time_fin=tend,
            instance=self,
        )

    def print_numerical_specs(self):
        """prints the numerical specifications of the internal computations"""
        if self.feed_lvl >= 2:
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
        if _USE_FAST_P:
            chi = self._chi_z  # reuse precomputed
            ell_grid = self.ell[None, :] + 0.5
            kappa = ell_grid / chi[:, None]
            mask = kappa <= cfg.specs["kmax"]
            if np.any(mask):
                for iz, chi_val in enumerate(chi):
                    row_mask = mask[iz]
                    if not np.any(row_mask):
                        continue
                    kappas_row = kappa[iz, row_mask]
                    zval = self.z[iz]
                    self.Pell[iz, row_mask] = (
                        (self.cosmo.SigmaMG(zval, kappas_row) ** 2)
                        * self.cosmo.matpow(
                            zval, kappas_row, nonlinear=cfg.settings["nonlinear_photo"]
                        )
                        / (chi_val**2.0)
                    )
        else:
            chi = self.cosmo.comoving(self.z)
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
        chi = self._chi_z  # reuse cached
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
            self.sqrtPell["WL_IA"][zin, ell] = (
                np.sqrt(
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
        Wgc = self.window.norm_ngal_photoz(z, i, "GCph") * self.cosmo.Hubble(z)
        # Wgc = self.window.norm_ngal_photoz(z,i) * self.nuisance.bias(self.biaspars, i)(z) * \
        #                                          self.cosmo.Hubble(z)

        tgcend = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
            text="    ...done bin {} for GCph in: ",
            instance=self,
            time_ini=tgcstart,
            time_fin=tgcend,
        )

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
         W_i^{WL} = W_i^{IA}+\\frac{3}{2}\\left(\\frac{H_0}{c}\\right)^2\\Omega_{m,0}(1+z)r(z)
         \\int_z^{z_{\\rm max}}{dz' \\frac{n_i(z')}{\\bar{n}(z)}\\left[1-\\frac{r(z)}{r(z')}\\right]}
        """

        twlstart = time()
        if _USE_FAST_KERNEL:
            prefac = (3.0 / 2.0) * self.cosmo.Hubble(0.0) ** 2.0 * self.cosmo.Omegam_of_z(0.0)
            chi_z = self.cosmo.comoving(z) if z is not self.z else self._chi_z
            eff_z = self.efficiency[i](z)
            Wwl = prefac * (1.0 + z) * chi_z * eff_z
            IA_val = self.IAvalue(z)
            hub_z = self.cosmo.Hubble(z)
            WIA = self.window.norm_ngal_photoz(z, i, "WL") * IA_val * hub_z
        else:
            prefac = (3.0 / 2.0) * self.cosmo.Hubble(0.0) ** 2.0 * self.cosmo.Omegam_of_z(0.0)
            Wwl = np.array(
                [prefac * (1 + zi) * self.cosmo.comoving(zi) * self.efficiency[i](zi) for zi in z]
            )
            WIA = self.window.norm_ngal_photoz(z, i, "WL") * np.array(
                [self.IAvalue(zi) * self.cosmo.Hubble(zi) for zi in z]
            )
        # Sakr Fix June 2023
        # kernel = Wwl+WIA

        twlend = time()

        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
            text="    ...done bin {} for WL in:",
            instance=self,
            time_ini=twlstart,
            time_fin=twlend,
        )

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
        z = self.z
        ngal_func = self.window.norm_ngal_photoz
        comoving_func = self.cosmo.comoving
        if not _USE_FAST_EFF:
            # Fallback to vectorized O(N^2) implementation
            return faster_integral_efficiency(i, ngal_func, comoving_func, z)
        # New O(N) implementation outsourced to helper
        return much_faster_integral_efficiency(i, ngal_func, comoving_func, z)

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
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
            text="    ...lensing efficiency computation done in:",
            instance=self,
            time_ini=teffstart,
            time_fin=teffend,
        )
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

        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="Computing Cls integral for {}".format(self.observables),
            instance=self,
        )
        # full_ell = np.linspace(cfg.specs['ellmin'],cfg.specs['ellmax'],cfg.specs['ellmax']-cfg.specs['ellmin'])
        # full_ell = np.round(full_ell, 0)
        full_ell = self.ell

        cls = {"ells": full_ell}

        hub = self.cosmo.Hubble(self.z)

        tstart = time()
        tcell = time()

        # PYTHONIZE THIS HORRIBLE THING
        # for obs1, obs2, bin1, bin2 in product(
        #        self.observables, self.observables, self.binrange[0], self.binrange[1] #MMmod: BEWARE! THIS IS UGLY!
        # ):
        for obs1, obs2 in product(self.observables, self.observables):
            for bin1, bin2 in product(self.binrange[obs1], self.binrange[obs2]):
                clinterp = self.clsintegral(obs1, obs2, bin1, bin2, hub)

                finalcls = np.zeros((len(full_ell)))
                for ind, lval in enumerate(full_ell):
                    if (cfg.specs["lmin_" + obs1] <= lval <= cfg.specs["lmax_" + obs1]) and (
                        cfg.specs["lmin_" + obs2] <= lval <= cfg.specs["lmax_" + obs2]
                    ):
                        finalcls[ind] = clinterp(lval)

                cls[obs1 + " " + str(bin1) + "x" + obs2 + " " + str(bin2)] = finalcls

            tbin = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=3,
                text="    ...{} {} x {} {} done in: ".format(obs1, bin1, obs2, bin2),
                instance=self,
                time_ini=tcell,
                time_fin=tbin,
            )
            tcell = tbin

        tend = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
            text="Cls integral computation done in: ",
            instance=self,
            time_ini=tstart,
            time_fin=tend,
        )

        return cls

    def computecls_vectorized(self):
        """Vectorized function to compute the angular power spectrum for all observables, redshift bins and multipoles.

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
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=2,
            text="Computing Cls integral for {}".format(self.observables),
            instance=self,
        )

        full_ell = self.ell
        cls = {"ells": full_ell}
        hub = self.cosmo.Hubble(self.z)

        tstart = time()

        # Pre-compute all clsintegrals
        clinterps = {}
        for obs1, obs2 in product(self.observables, self.observables):
            # Get the correct binrange for each observable
            bins1 = self.binrange[obs1]
            bins2 = self.binrange[obs2]

            for bin1, bin2 in product(bins1, bins2):
                clinterps[(obs1, obs2, bin1, bin2)] = self.clsintegral(obs1, obs2, bin1, bin2, hub)

        # Vectorized computation of finalcls
        for (obs1, obs2, bin1, bin2), clinterp in clinterps.items():
            lmin1, lmax1 = cfg.specs[f"lmin_{obs1}"], cfg.specs[f"lmax_{obs1}"]
            lmin2, lmax2 = cfg.specs[f"lmin_{obs2}"], cfg.specs[f"lmax_{obs2}"]

            mask = (
                (full_ell >= lmin1)
                & (full_ell <= lmax1)
                & (full_ell >= lmin2)
                & (full_ell <= lmax2)
            )

            finalcls = np.zeros(len(full_ell))
            finalcls[mask] = clinterp(full_ell[mask])

            key = f"{obs1} {bin1}x{obs2} {bin2}"
            cls[key] = finalcls

            tbin = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=3,
                text=f"    ...{obs1} {bin1} x {obs2} {bin2} done in: ",
                instance=self,
                time_ini=tstart,
                time_fin=tbin,
            )
            tstart = tbin

        tend = time()
        upt.time_print(
            feedback_level=self.feed_lvl,
            min_level=3,
            text="Cls integral computation done in: ",
            instance=self,
            time_ini=tstart,
            time_fin=tend,
        )

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
