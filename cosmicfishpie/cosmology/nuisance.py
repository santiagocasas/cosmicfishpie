# -*- coding: utf-8 -*-
"""Nuisance

This module contains nuisance parameter functions.

"""

import logging
import os
from copy import deepcopy

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if handlers are already added to prevent duplicate logs in interactive environments
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Define a formatter that includes timestamps and caller information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - Line %(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Nuisance:
    def __init__(self, configuration=None, spectrobiasparams=None, spectrononlinearpars=None):
        if configuration is None:
            self.config = cfg
        else:
            self.config = configuration
        self.observables = self.config.obs
        self.specs = self.config.specs
        self.settings = self.config.settings
        self.specsdir = self.settings["specs_dir"]
        self.surveyname = self.specs["survey_name"]
        if "GCsp" in self.observables:
            self.sp_zbins = self.gcsp_zbins()
            self.sp_dndz = self.gcsp_dndz()
            self.sp_zbins_mids = self.gcsp_zbins_mids()
            self.sp_bias_sample = self.specs["sp_bias_sample"]
            self.sp_bias_root = self.specs["sp_bias_root"]
            self.sp_bias_model = self.specs["sp_bias_model"]
            self.sp_bias_prtz = self.specs["sp_bias_parametrization"]
            if spectrobiasparams is None:
                self.Spectrobiasparams = deepcopy(self.config.Spectrobiasparams)
            else:
                self.Spectrobiasparams = spectrobiasparams
            if spectrononlinearpars is None:
                self.spectrononlinearpars = deepcopy(self.config.Spectrononlinearparams)
            else:
                self.spectrononlinearpars = spectrononlinearpars
            self._vectorized_gcsp_bias_at_z = np.vectorize(self.gcsp_bias_at_z)
            self._vectorized_gcsp_rescale_sigmapv_at_z = np.vectorize(
                self.gcsp_rescale_sigmapv_at_z, excluded=["sigma_key"]
            )

        if "IM" in self.observables:
            filename_THI_noise = self.specs["IM_THI_noise_file"]
            self.Tsys_arr = np.loadtxt(os.path.join(self.specsdir, filename_THI_noise))
            filename_IM = self.specs["IM_bins_file"]
            self.im_table = np.loadtxt(os.path.join(self.specsdir, filename_IM))

        if "WL" in self.observables or "GCph" in self.observables:
            self.z = np.linspace(
                min(self.specs["z_bins_WL"][0], self.specs["z_bins_GCph"][0]),
                max(self.specs["z_bins_WL"][-1], self.specs["z_bins_GCph"][-1]) + 1,
                50 * self.settings["accuracy"],
            )
        if "WL" in self.observables:
            self.lumratio = self.luminosity_ratio()

    def gcph_bias(self, biaspars, ibin=1):
        """Galaxy bias

        Parameters
        ----------
        z     : array
                redshift

        Returns
        -------
        float
            Value galaxy bias at redshift z

        """
        self.biaspars = biaspars

        z = self.z

        # TBA: NEED TO INCLUDE CHECK OF THE BIASPARS PASSED

        if self.biaspars["bias_model"] == "sqrt":
            b = self.biaspars["b0"] * np.sqrt(1 + z)
            return interp1d(z, b, kind="linear")

        elif self.biaspars["bias_model"] == "binned":
            zb = self.specs["z_bins_GCph"]
            zba = np.array(zb)
            brang = self.specs["binrange_GCph"]
            last_bin_num = brang[-1]

            def binbis(zz):
                lowi = np.where(zba <= zz)[0][-1]
                upr.debug_print(zz)
                upr.debug_print(lowi)
                # iin[ind] = binrange[lowi]
                if zz >= zba[-1] and lowi == last_bin_num:
                    bii = self.biaspars["b" + str(last_bin_num)]
                else:
                    bii = self.biaspars["b" + str(lowi + 1)]
                return bii

            vbinbis = np.vectorize(binbis)
            return vbinbis
        elif self.biaspars["bias_model"] == "binned_constant":

            def binned_func(z):
                bi = self.biaspars["b" + str(ibin)]
                return bi

            vbinned_func = np.vectorize(binned_func)
            return vbinned_func
        elif self.biaspars["bias_model"] == "flagship":
            b = self.biaspars["A"] + self.biaspars["B"] / (
                1 + np.exp(-self.biaspars["C"] * (z - self.biaspars["D"]))
            )
            return interp1d(z, b, kind="linear")
        else:
            print("ERROR: unknown galaxy bias model!")
            print("Available models are: sqrt, binned and flagship")

    def IA(self, IApars, cosmo):
        r"""Intrinsic Alignment

        :param z: float
            redshift

        :return:
            - float: Value of IA window at redshift z

        :notes:
            Implements the following equation:

            .. math::
                W_i^{IA} = -\frac{\mathcal{A}_{\rm IA}C_{\rm IA}\Omega_m\mathcal{F}_{\rm IA}}{D(z)}
                \frac{n_i(z)}{\bar{n}}\frac{H(z)}{c}
        """

        self.IApars = IApars
        self.cosmo = cosmo
        self.Omegam = self.cosmo.Omegam_of_z(0.0)
        pivot_z_IA = self.settings["pivot_z_IA"]
        z = self.z
        if self.IApars["IA_model"] == "eNLA":
            CIA = 0.0134 * (1 + pivot_z_IA)
            fac = -self.IApars["AIA"] * CIA * self.Omegam
            z_dep = (1 + z) / (1 + pivot_z_IA)
            IAwin = (
                fac
                * z_dep ** self.IApars["etaIA"]
                * ((self.lumratio(z) ** self.IApars["betaIA"]) / (self.cosmo.growth(z).flatten()))
            )
        else:
            print("I only now eNLA model for Intrinsic Alignments, give me a break!")
            exit()
        IAwinfunc = InterpolatedUnivariateSpline(z, IAwin, k=1)

        return IAwinfunc

    def luminosity_ratio(self):
        """Luminosity ratio function used for Intrinsic Alignment eNLA model.

        This function reads and interpolates the luminosity ratio <L(z)>/L*(z) from a file.
        The ratio is used in the extended nonlinear alignment (eNLA) model for intrinsic alignments.

        Parameters
        ----------
        None

        Returns
        -------
        callable
            An interpolation function that takes redshift z as input and returns the luminosity ratio.
            If the luminosity ratio file cannot be loaded, returns a function that always returns 1.0.

        Notes
        -----
        - Attempts to read the luminosity ratio from 'lumratio_file.dat' in the specs directory
        - The file should contain two columns: redshift and luminosity ratio values
        - The luminosity ratio represents <L(z)>/L*(z), where:
            - <L(z)> is the mean luminosity at redshift z
            - L*(z) is the characteristic luminosity at redshift z
        - Uses linear interpolation between data points
        """
        try:
            # Lumratio file for IA
            lum = np.loadtxt(os.path.join(self.specsdir, "lumratio_file.dat"))
            lumratio = interp1d(lum[:, 0], lum[:, 1], kind="linear")
            logger.info("Successfully loaded luminosity ratio file")
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Could not load luminosity ratio file: {e}. Using default value of 1.0")

            def default_lumratio(z):
                return np.ones_like(z) if hasattr(z, "__len__") else 1.0

            lumratio = default_lumratio

        return lumratio

    def gcsp_zbins(self):
        """
        Reads from file for a given survey
        """
        zbins = []
        zbin_inds = []
        for key, val in self.specs["z_bins_sp"].items():
            zbins.append(val)
            zbin_inds.append(key)
        zbins = np.unique(np.concatenate(zbins))
        self.sp_zbins_inds = zbin_inds
        return zbins

    def gcsp_zbins_mids(self):
        sp_zbins_mids = unu.moving_average(self.sp_zbins)
        return sp_zbins_mids

    def gcsp_bias_at_zm(self):
        b_arr = np.ones(len(self.sp_zbins_mids))
        if "linear" in self.sp_bias_model:
            for ii, z_ind in enumerate(self.sp_zbins_inds):
                bkey = self.sp_bias_root + self.sp_bias_sample + "_" + str(z_ind)
                b_arr[ii] = self.Spectrobiasparams[bkey]
            if self.sp_bias_model == "linear_log":
                b_arr = np.exp(b_arr)
        return b_arr

    def gcsp_zvalue_to_zindex(self, z):
        bin_arr_ind = unu.bisection(self.sp_zbins, z)
        bin_num = bin_arr_ind + 1
        if bin_num < self.sp_zbins_inds[0]:
            bin_num = self.sp_zbins_inds[0]
        if bin_num > self.sp_zbins_inds[-1]:
            bin_num = self.sp_zbins_inds[-1]
        return bin_num

    def gcsp_bias_at_zi(self, zi):
        """
        Parameters
        ----------
        zi : int
            Redshift bin index

        Returns
        -------
        float
            Bias at the redshift bin index zi
        """
        bias_at_zmids = self.gcsp_bias_at_zm()
        arr_ind = zi - 1
        bias_at_zi = bias_at_zmids[arr_ind]
        return bias_at_zi

    def gcsp_bias_at_z(self, z):
        """
        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Bias at the redshift z
        """
        bin_num = self.gcsp_zvalue_to_zindex(z)
        logger.debug(f"bin_num: {bin_num} with z: {z}")
        bias_at_zzi = self.gcsp_bias_at_zi(bin_num)
        logger.debug(f"bias_at_zzi: {bias_at_zzi} with zi: {bin_num}")
        return bias_at_zzi

    def vectorized_gcsp_bias_at_z(self, z):
        return self._vectorized_gcsp_bias_at_z(z)

    def gcsp_bias_kscale(self, k, z=None):
        bterm_k = 1
        if k is not None:
            if self.sp_bias_model == "linear_Qbias":
                default_A1 = 0.0
                default_A2 = 0.0
                try:
                    bterm_k = (1 + k**2 * self.Spectrobiasparams.get("A2", default_A2)) / (
                        1 + k * self.Spectrobiasparams.get("A1", default_A1)
                    )
                except KeyError as ke:
                    print(
                        f"The key {ke} is not in dictionary."
                        f"Check observables and parameters being used"
                    )
                    print("Is spectriobiaspars a dictionary?")
                    print(isinstance(self.Spectrobiasparams, dict))
                    print(self.Spectrobiasparams)
                    raise ke
        return bterm_k

    def gcsp_bias_interp(self):
        """Galaxy bias for the galaxies used in spectroscopic Galaxy Clustering
        Parameters
        ----------

        Returns
        --------
        object
        Interpolating function of bias for the redshift bins given

        Note
        --------
        Reads from file and interpolates for a given survey
        """
        bias_at_zmids = self.gcsp_bias_at_zm()
        z_mids = self.gcsp_zbins_mids()
        bofz_spec = InterpolatedUnivariateSpline(z_mids, bias_at_zmids, k=1)
        return bofz_spec

    def gcsp_rescale_sigmapv_at_z(self, z, sigma_key="sigmap"):
        bin_num = self.gcsp_zvalue_to_zindex(z)
        sigma_key = sigma_key + "_" + str(bin_num)
        sigma_pv_value = self.spectrononlinearpars.get(sigma_key, 1.0)
        return sigma_pv_value

    def vectorized_gcsp_rescale_sigmapv_at_z(self, z, sigma_key="sigmap"):
        return self._vectorized_gcsp_rescale_sigmapv_at_z(z, sigma_key=sigma_key)

    def gcsp_dndz(self):
        """
        Reads from file for a given survey
        """
        dndz = []
        for ii in self.sp_zbins_inds:
            dndz.append(self.specs["dndOmegadz"][ii])
        return np.array(dndz)

    def extra_Pshot_noise(self):
        Psfid = self.settings["Pshot_nuisance_fiducial"] = 0
        return Psfid

    def IM_bias(self, z):
        """
        IM 21cm HI bias function from http://arxiv.org/abs/2006.05996
        """
        bb = 0.3 * (1 + z) + 0.6
        return bb

    def IM_zbins(self):
        """
        Reads from file for a given survey
        """
        # this dict can be read from a file
        zbins = np.unique(np.concatenate((self.im_table[:, 0], self.im_table[:, 2])))
        return zbins

    def IM_zbins_mids(self):
        z_bins = self.IM_zbins()
        z_bin_mids = unu.moving_average(z_bins)
        return z_bin_mids

    def IM_bias_at_zm(self):
        bfunc = self.IM_bias
        zmids = self.IM_zbins_mids()
        b_arr = bfunc(zmids)
        return b_arr

    def IM_THI_noise(self):
        """ "
        Reads from file for a given survey
        """
        Tsys_interp = UnivariateSpline(self.Tsys_arr[:, 0], self.Tsys_arr[:, 1])
        return Tsys_interp
