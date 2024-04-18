# -*- coding: utf-8 -*-
"""LSS_window

This module returns the window functions for LSS surveys.

"""

import numpy as np
from scipy.integrate import trapz
from scipy.special import erf

import cosmicfishpie.fishermatrix.config as cfg


class GalaxyPhotoDist:
    def __init__(self, photopars):
        """Class to obtain the survey specific ingredients of the window function

        Parameters
        ----------
        photopars : dict
                    a dictionary containing specifications for the window function's galaxy distribution

        Attributes
        ----------
        z_bins  : list
                  list of the surveys redshift bin edges
        n_bins  : int
                  number of redshift bins
        z0      : float
                  parameter for the width of the galaxy redshift distribution
        z0_p    : float
                  parameter for the center of the galaxy redshift distribution
        ngamma  : float
                  parameter for the power law cutoff of the galaxy redshift distribution
        photo   : dict
                  a dictionary containing specifications for the window function's galaxy distribution
        z_min   : float
                  minimum redshift of the probes
        z_max   : float
                  maximum redshift of the probes
        norm    : callable
                  callable function that when given the redshift bin and a redshift, returns the normalization of the galaxy redshift distribution
        n_i_vec : callable
                  callable function that receives the index of a redshift bin and a numpy.ndarray of redshifts and gives back the binned galaxy redshift distribution without photometric redshift errors
        """
        self.z_bins = cfg.specs["z_bins"]
        self.n_bins = len(self.z_bins)
        self.z0 = cfg.specs["z0"]
        self.z0_p = cfg.specs["z0_p"]
        self.ngamma = cfg.specs["ngamma"]
        self.photo = photopars
        self.z_min = self.z_bins[0]
        self.z_max = self.z_bins[-1]
        self.normalization = self.norm()
        self.n_i_vec = np.vectorize(self.n_i)

    def dNdz(self, z):
        """unnormalized dN/dz(z)

        Parameters
        ----------
        z : numpy.ndarray
            array of redshifts at which to compute the galaxy distribution

        Returns
        -------
        numpy.ndarray
            unnormalized theoretical galaxy redshift distribution

        Note
        -----
        Implements the following parametrization:

        .. math::
            \\frac{{\\rm d} N}{{\\rm d} z} = \\left(\\frac{z}{z_b}\\right)^2 \\, \\exp \\left[-\\left(\\frac{z}{z_0}\\right)^{n_\\gamma} \\right]
        """
        pref = z / self.z0_p
        expo = z / self.z0

        return pref ** 2 * np.exp(-(expo ** self.ngamma))

    def n_i(self, z, i):
        """Function to compute the unnormalized dN/dz(z) with a window picking function applied to it

        Parameters
        ----------
        z : float
            Redshift
        i : int
            index of the redshift bin

        Returns
        -------
        float
            binned distribution without photometric redshift errors

        """
        z = np.atleast_1d(z)
        dNdz_at_z = self.dNdz(z)
        if i == 0 or i > self.n_bins:
            return None

        mask = (z <= self.z_bins[i]) & (z >= self.z_bins[i - 1])
        dNdz_at_z[~mask] = 0.0
        return dNdz_at_z

    def ngal_photoz(self, z, i):
        """Function to compute the binned galaxy redshift distribution convolved with photometric redshift errors n^{ph}_i(z)

        Parameters
        ----------
        z : float
            redshift at which to compute the distribution
        i : int
            index of the redshift bin

        Returns
        -------
        float
            binned galaxy distribution convolved with photometric redshift errors

        Note
        -----
        Implements the following equation:

        .. math::
            p_{ph}(z_p|z) = \\frac{1-f_{out}}{\\sqrt{2\\pi}\\sigma_b(1+z)} \\exp\\left\\{-\\frac{1}{2}\\left[\\frac{z-c_bz_p-z_b}{\\sigma_b(1+z)}\\right]^2\\right\\} \\ + \\frac{f_{out}}{\\sqrt{2\\pi}\\sigma_0(1+z)} \\exp\\left\\{-\\frac{1}{2}\\left[\\frac{z-c_0z_p-z_0}{\\sigma_0(1+z)}\\right]^2\\right\\}
        """

        if i == 0 or i >= 11:
            return None

        term1 = (
            self.photo["cb"]
            * self.photo["fout"]
            * erf(
                (np.sqrt(1 / 2) * (z - self.photo["zo"] - self.photo["co"] * self.z_bins[i - 1]))
                / (self.photo["sigma_o"] * (1 + z))
            )
        )
        term2 = (
            -self.photo["cb"]
            * self.photo["fout"]
            * erf(
                (np.sqrt(1 / 2) * (z - self.photo["zo"] - self.photo["co"] * self.z_bins[i]))
                / (self.photo["sigma_o"] * (1 + z))
            )
        )
        term3 = (
            self.photo["co"]
            * (1 - self.photo["fout"])
            * erf(
                (np.sqrt(1 / 2) * (z - self.photo["zb"] - self.photo["cb"] * self.z_bins[i - 1]))
                / (self.photo["sigma_b"] * (1 + z))
            )
        )
        term4 = (
            -self.photo["co"]
            * (1 - self.photo["fout"])
            * erf(
                (np.sqrt(1 / 2) * (z - self.photo["zb"] - self.photo["cb"] * self.z_bins[i]))
                / (self.photo["sigma_b"] * (1 + z))
            )
        )

        return (
            self.dNdz(z)
            * (term1 + term2 + term3 + term4)
            / (2 * self.photo["co"] * self.photo["cb"])
        )

    def norm(self):
        """n^{ph}_i(z)

        Parameters
        ----------
        z : float
            redshift at which to compute the function
        i : integer
            redshift bin index

        Returns
        -------
        float
           normalized binned galaxy distribution convolved with photoz errors

        """

        # norm = romberg(self.ngal_photoz, self.z_min, self.z_max, args=(i,))
        # Using this as romberg was giving crazy normalizations for the first 2
        # bins
        zint = np.linspace(0.0, self.z_max, 1000)
        dz = self.z_max / 1000

        norm = [
            trapz([self.ngal_photoz(z, i) for z in zint], dx=dz) for i in range(1, len(self.z_bins))
        ]
        norm.insert(0, None)

        return norm

    def norm_ngal_photoz(self, z, i):
        """n^{ph}_i(z)

        Parameters
        ----------
        z : array
            redshift at which to compute the function
        i : integer
            redshift bin index

        Returns
        -------
        float
           normalized binned galaxy distribution convolved with photoz errors

        """

        # norm = romberg(self.ngal_photoz, self.z_min, self.z_max, args=(i,))

        # Using this as romberg was giving crazy normalizations for the first 2
        # bins

        return np.array([self.ngal_photoz(zi, i) for zi in z]) / self.normalization[i]
