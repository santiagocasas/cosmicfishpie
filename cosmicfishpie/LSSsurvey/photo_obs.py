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

import cosmicfishpie.cosmology.cosmology as cosmology
import cosmicfishpie.cosmology.nuisance as nuisance
import cosmicfishpie.fishermatrix.config as cfg
import cosmicfishpie.LSSsurvey.photo_window as photo_window
from cosmicfishpie.utilities.utils import printing as upt

cachedir = "./cache"
memory = Memory(cachedir, verbose=0)


# @memory.cache


def memo_integral_efficiency(i, ngal_func, comoving_func, z, zint_mat, diffz):
    intg_mat = np.array(
        [
            (
                ngal_func(zint_mat[zii], i)
                * (1 - comoving_func(zint_mat[zii, 0]) / comoving_func(zint_mat[zii]))
            )
            for zii in range(len(zint_mat))
        ]
    )
    integral_really = integrate.trapz(intg_mat, dx=diffz, axis=1)
    intp = interp1d(z, integral_really, kind="cubic")
    return intp


def faster_integral_efficiency(i, ngal_func, comoving_func, zarr):
    zprime = zarr[:, None]
    wintgd = ngal_func(zprime, i) * (1.0 - comoving_func(zarr) / comoving_func(zprime))
    witri = np.tril(wintgd)
    wint = np.trapz(witri, zarr, axis=0)
    intp = interp1d(zarr, wint, kind="cubic")
    return intp


class ComputeCls:
    def __init__(
        self, cosmopars, photopars, IApars, biaspars, print_info_specs=False, fiducial_cosmo=None
    ):
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
        for key in cfg.obs:
            if key in ["GCph", "WL"]:
                self.observables.append(key)

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
        self.binrange = cfg.specs["binrange"]
        self.zsamp = int(round(200 * cfg.settings["accuracy"]))
        self.ellsamp = int(round(100 * cfg.settings["accuracy"]))
        self.ell = np.logspace(
            np.log10(cfg.specs["ellmin"]), np.log10(cfg.specs["ellmax"] + 10), num=self.ellsamp
        )
        # self.ell      = np.linspace(cfg.specs['ellmin'],cfg.specs['ellmax']+10,num=self.ellsamp)

        if cfg.input_type == "camb":
            self.z_min = cfg.specs["z_bins"][0]
            self.z_max = cfg.specs["z_bins"][-1]  # +0.5
        if cfg.input_type == "class":
            self.z_min = cfg.specs["z_bins"][0]
            self.z_max = cfg.specs["z_bins"][-1]  # +0.5
        elif cfg.input_type == "external":
            self.z_min = np.max([cfg.specs["z_bins"][0], self.cosmo.results.zgrid[0]])
            # self.z_max = np.min([cfg.specs['z_bins'][-1]+0.5, self.cosmo.results.zgrid[-1]])
            self.z_max = np.min([cfg.specs["z_bins"][-1], self.cosmo.results.zgrid[-1]])
        # +1 to go beyond the bin limit
        self.z = np.linspace(self.z_min, self.z_max, self.zsamp)
        self.dz = np.mean(np.diff(self.z))

        if print_info_specs:
            self.print_numerical_specs()

    def compute_all(self):
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
        Calculates the Limber-approximated power spectrum
        This is done for a range of redshift values and multipoles.
        Will add MG effects to WL and(!) GCph if asked for.

        Returns:
            None
                The results are stored in the class object self.Pell
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

        Returns:
            None
                The results are stored in the class object self.sqrtPell
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
        """GCph kernel function

        Parameters:
            z     : array
                    redshift
            i     : int
                    bin index
        Returns:
            float
                Value of GCph at redshift z for bin i
        Notes:
            Implements the following equation:

            .. math::
                W_i^{GCph} = b(z) \\frac{n_i(z)}{\\bar{n}(z)} H(z)

        """
        tgcstart = time()
        # Wgc = self.window.norm_ngal_photoz(z,i) * np.array([self.nuisance.bias(self.biaspars, i)(z) * \
        # Wgc = self.window.norm_ngal_photoz(z,i) *
        # np.array([self.biaspars['b{:d}'.format(i)] * \
        Wgc = self.window.norm_ngal_photoz(z, i) * self.cosmo.Hubble(z)
        # Wgc = self.window.norm_ngal_photoz(z,i) * self.nuisance.bias(self.biaspars, i)(z) * \
        #                                          self.cosmo.Hubble(z)

        tgcend = time()
        if self.feed_lvl >= 3:
            print("    ...done bin {} for GCph in {:.2f} s".format(i, tgcend - tgcstart))

        return Wgc

    def lensing_kernel(self, z, i):
        """WL kernel function
        Parameters:
            z     : array
                    redshift
            i     : int
                    bin index
        Returns:
            float:  Value of WL at redshift z for bin i
        Notes:
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
        WIA = self.window.norm_ngal_photoz(z, i) * np.array(
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
        teffstart = time()
        efficiency = [self.integral_efficiency(i) for i in self.binrange]
        efficiency.insert(0, None)
        teffend = time()
        if self.feed_lvl >= 3:
            print("    ...lensing efficiency computation took {:.2f} s".format(teffend - teffstart))
        return efficiency

    def compute_kernels(self):
        if "GCph" in self.observables:
            self.GC = [
                interp1d(self.z, self.galaxy_kernel(self.z, ind), kind="cubic")
                for ind in self.binrange
            ]
            self.GC.insert(0, None)
        if "WL" in self.observables:
            self.efficiency = self.lensing_efficiency()
            # Sakr Fix June 2023
            # self.WL         = [interp1d(self.z,self.lensing_kernel(self.z,ind), kind='cubic') for ind in self.binrange]
            self.WL = [
                interp1d(self.z, self.lensing_kernel(self.z, ind)[0], kind="cubic")
                for ind in self.binrange
            ]
            self.WL.insert(0, None)
            # Sakr Fix June 2023
            self.WL_IA = [
                interp1d(self.z, self.lensing_kernel(self.z, ind)[1], kind="cubic")
                for ind in self.binrange
            ]
            self.WL_IA.insert(0, None)
        return None

    def genwindow(self, z, obs, i):
        """generic kernel function

        Parameters
        ----------
        z     : array
                redshift
        obs   : str
                observable (GC or WL)
        i     : int
                bin index

        Returns
        -------
        float
            Value of WL at redshift z for bin i

        Notes
        -----

        """

        win = []
        # Sakr Fix June 2023
        win_IA = []

        if obs == "GCph":
            bz = self.nuisance.gcph_bias(self.biaspars, i)(z)
            win = self.GC[i](z) * bz
            # Sakr Fix June 2023
            # just filling it with zeros to keep the vectorisation possible
            # later
            win_IA = self.GC[i](z) * 0.0
        elif obs == "WL":
            win = self.WL[i](z)
            # Sakr Fix June 2023
            win_IA = self.WL_IA[i](z)

        # Sakr Fix June 2023
        return win, win_IA

    def computecls(self):
        """
        Parameters:
            ell   : float multipole
            X     : str first observable
            Y     : str second observable
            i     : int first bin
            j     : int second bin
        Returns:
            float
                Value of Cl
        Notes:
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
            self.observables, self.observables, self.binrange, self.binrange
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

        clint = integrate.trapz(intgn, dx=self.dz, axis=0)

        clint[~mask1] = 0
        clint[~mask2] = 0

        clinterp = interp1d(self.ell, clint, kind="linear")  # 'cubic')

        return clinterp
