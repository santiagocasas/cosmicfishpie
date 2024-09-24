# -*- coding: utf-8 -*-
"""Nuisance

This module contains nuisance parameter functions.

"""

import os
from copy import deepcopy

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upr


class Nuisance:
    def __init__(self, spectrobiasparams=None):
        self.observables = cfg.obs
        self.specs = cfg.specs
        self.settings = cfg.settings
        self.specsdir = self.settings["specs_dir"]
        self.surveyname = cfg.specs["survey_name"]
        if "GCsp" in self.observables:
            # gc_survey_dict = cfg.specs["gc_specs_files_dict"]
            # gc_surveyname = cfg.survey_equivalence(self.surveyname)
            # default_filename = gc_survey_dict["default"]
            # gc_filename = gc_survey_dict.get(gc_surveyname, default_filename)
            # self.gc_table = np.loadtxt(os.path.join(self.specsdir, gc_filename))
            self.sp_zbins = self.gcsp_zbins()
            self.sp_dndz = self.gcsp_dndz()
            self.sp_zbins_mids = self.gcsp_zbins_mids()
            if spectrobiasparams is None:
                self.Spectrobiasparams = deepcopy(cfg.Spectrobiasparams)
            else:
                self.Spectrobiasparams = spectrobiasparams
        if "IM" in self.observables:
            filename_THI_noise = self.specs["IM_THI_noise_file"]
            self.Tsys_arr = np.loadtxt(os.path.join(self.specsdir, filename_THI_noise))
        if "WL" in self.observables:
            self.lumratio = self.luminosity_ratio()
        if "GCph" in self.observables or "WL" in self.observables:
            self.z_bins_ph = self.specs["z_bins_ph"]
            self.z_ph = np.linspace(
                self.z_bins_ph[0], self.z_bins_ph[-1] + 1, 50 * self.settings["accuracy"]
            )

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

        z = self.z_ph

        # TBA: NEED TO INCLUDE CHECK OF THE BIASPARS PASSED

        if self.biaspars["bias_model"] == "sqrt":
            b = self.biaspars["b0"] * np.sqrt(1 + z)
            return interp1d(z, b, kind="linear")

        elif self.biaspars["bias_model"] == "binned":
            zb = self.z_bins_ph
            zba = np.array(zb)
            brang = self.specs["binrange"]
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
        z = self.z_ph
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
        Parameters
        ----------
        Returns
        -------
        float
            Value of the luminosity ratio

        Note
        -----
        Reads from file and interpolates the following quantity:

        .. math::
            \\frac{<L(z)>}{L_*(z)}
        """

        # Lumratio file for IA
        lum = np.loadtxt(os.path.join(self.specsdir, "lumratio_file.dat"))
        # ,fill_value='extrapolate')
        lumratio = interp1d(lum[:, 0], lum[:, 1], kind="linear")
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
        self.sp_bias_sample = self.specs["sp_bias_sample"]
        self.sp_bias_root = self.specs["sp_bias_root"]
        self.sp_bias_model = self.specs["sp_bias_model"]
        self.sp_bias_prtz = self.specs["sp_bias_parametrization"]
        b_arr = np.ones(len(self.sp_zbins_mids))
        if self.sp_bias_model == "linear" or self.sp_bias_model == "linear_log":
            for ii, z_ind in enumerate(self.sp_zbins_inds):
                bkey = self.sp_bias_root + self.sp_bias_sample + "_" + str(z_ind)
                b_arr[ii] = self.Spectrobiasparams[bkey]
            if self.sp_bias_model == "linear_log":
                b_arr = np.exp(b_arr)
        return b_arr

    def gscp_bias_at_zi(self, zi):
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
        if zi < self.sp_zbins_inds[0]:
            zi = self.sp_zbins_inds[0]
        if zi > self.sp_zbins_inds[-1]:
            zi = self.sp_zbins_inds[-1]
        bias_at_zmids = self.gcsp_bias_at_zm()
        arr_ind = zi - 1
        bias_at_zi = bias_at_zmids[arr_ind]
        return bias_at_zi

    def gscp_bias_at_z(self, z):
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
        bin_arr_ind = unu.bisection(self.sp_zbins, z)
        bin_num = bin_arr_ind + 1
        bias_at_zzi = self.gscp_bias_at_zi(bin_num)
        return bias_at_zzi

    def vectorized_gscp_bias_at_z(self, z):
        bias_at_z = np.vectorize(self.gscp_bias_at_z)
        return bias_at_z(z)

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

    def bterm_z_key(self, z_ind, z_mids, fiducosmo, bias_sample="g"):
        if bias_sample == "g":
            bi_at_z_mids = self.gcsp_bias_at_zm()
        if bias_sample == "I":
            bi_at_z_mids = self.IM_bias_at_zm()
        bstring = self.settings["vary_bias_str"]
        bstring = bstring + bias_sample
        b_i = bi_at_z_mids[z_ind - 1]
        if self.settings["bfs8terms"]:
            bstring = bstring + "s8"
            b_i = b_i * fiducosmo.sigma8_of_z(
                z_mids[z_ind - 1], tracer=self.settings["GCsp_Tracer"]
            )
        bstring = bstring + "_"
        bstring = bstring + str(z_ind)
        if "ln" in bstring:
            b_i = np.log(b_i)
        b_i = b_i.item()  # Convert 1-element array to scalar
        return bstring, b_i
