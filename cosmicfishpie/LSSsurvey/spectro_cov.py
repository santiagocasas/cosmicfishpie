# -*- coding: utf-8 -*-
"""CLS

This module contains cls calculations (only LSS atm).

"""
import numpy as np
from itertools import product
from copy import deepcopy

import cosmicfishpie.cosmology.cosmology as cosmology
import cosmicfishpie.cosmology.nuisance as nuisance
import cosmicfishpie.fishermatrix.config as cfg
import cosmicfishpie.LSSsurvey.spectro_obs as spec_obs
from cosmicfishpie.fishermatrix.derivatives import derivatives

from time import time
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upt
import copy
import os


class SpectroCov:

    def __init__(self, fiducialpars, fiducial_specobs=None,
                 bias_samples=['g','g']):
        """
        Initializes an object with specified fiducial parameters and computes 
        various power spectra
        (IM, XC, and gg) using those parameters depending on which observables 
        are present.
        
        :param fiducialpars: The fiducial parameter values used to compute 
                             power spectra.
        :type fiducialpars: list or array
        
        :param fiducial_specobs: An optional fiducial spectroscopic observation. 
                                 Defaults to None if not provided.
        :type fiducial_specobs: object
        
        :param bias_samples: The bias samples to use when computing 
                             power spectra. 
                             Defaults to ['g', 'g'] if not otherwise specified.
        :type bias_samples: list or array
        
        :return: None
            
        """
        #initializing the class only with fiducial parameters
        #if fiducial_specobs is None:
        if 'IM' in cfg.obs and 'GCsp' in cfg.obs:
            bias_samples = ['I', 'g']
            print("Entering Cov cross XC IM,g term")
            self.pk_obs = spec_obs.ComputeGalIM(fiducialpars,
                                                fiducialpars,
                                                bias_samples=bias_samples)
            self.pk_obs_gg = spec_obs.ComputeGalIM(fiducialpars,
                                                fiducialpars,
                                                bias_samples=['g', 'g'])
            self.pk_obs_II = spec_obs.ComputeGalIM(fiducialpars,
                                                fiducialpars,
                                                bias_samples=['I', 'I'])
        elif 'IM' in cfg.obs and 'I' in bias_samples:
            bias_samples = ['I', 'I']
            print("Entering Cov IM term")
            self.pk_obs = spec_obs.ComputeGalIM(fiducialpars,
                                                fiducialpars,
                                                bias_samples=bias_samples)
        elif 'GCsp' in cfg.obs and 'g' in bias_samples:
            bias_samples = ['g', 'g']
            print("Entering Cov gg term")
            self.pk_obs = spec_obs.ComputeGalSpectro(fiducialpars,
                                                     fiducialpars, 
                                                      bias_samples=bias_samples)
        else:
            self.pk_obs = fiducial_specobs
        
        self.area_survey = cfg.specs['area_survey']
        if 'GCsp' in self.pk_obs.observables: 
            self.dnz = self.pk_obs.nuisance.gcsp_dndz()
            self.z_bins = self.pk_obs.nuisance.gcsp_zbins()
            self.z_bin_mids = self.pk_obs.nuisance.gcsp_zbins_mids()
            self.dz_bins = np.diff(self.z_bins)
            self.global_z_bin_mids = self.z_bin_mids
            self.global_z_bins = self.z_bins
        if 'IM' in self.pk_obs.observables:
            self.IM_z_bins = self.pk_obs.nuisance.IM_zbins()
            self.IM_z_bin_mids = self.pk_obs.nuisance.IM_zbins_mids()
            self.Tsys_interp = self.pk_obs.nuisance.IM_THI_noise()
            self.global_z_bin_mids = self.IM_z_bin_mids
            self.global_z_bins = self.IM_z_bins
        ## Choose longest zbins array to loop in Fisher matrix
        if 'GCsp' in self.pk_obs.observables and 'IM' in self.pk_obs.observables:
            if len(self.z_bin_mids) >= len(self.IM_z_bin_mids):
                self.global_z_bin_mids = self.z_bin_mids
                self.global_z_bins = self.z_bins
            else:
                self.global_z_bin_mids = self.IM_z_bin_mids
                self.global_z_bins = self.IM_z_bins




    def Tsys_func(self, z):
        units = 1000 ##convert from K to mK
        Tsys_mK = units * self.Tsys_interp(z)
        return Tsys_mK

    def volume_bin(self, zi, zj):
        rad_to_area = 1/(4*np.pi*np.power(180/np.pi,2))
        d1 = self.pk_obs.cosmo.angdist(zi)
        d2 = self.pk_obs.cosmo.angdist(zj)
        sphere_vol = (4*np.pi/3)*(pow((1+zj)*d2,3)-pow((1+zi)*d1,3))
        vol = sphere_vol*rad_to_area
        return vol

    def d_volume(self, ibin):
        return self.volume_bin(self.global_z_bins[ibin],  self.global_z_bins[ibin+1])

    def volume_survey(self, ibin):
        vol = self.area_survey*self.d_volume(ibin)
        return vol

    def n_density(self, ibin):
        ndens = self.dnz[ibin]*self.dz_bins[ibin]/self.d_volume(ibin)
        return ndens

    def veff(self, ibin, k, mu):
        zi = self.z_bin_mids[ibin]
        npobs = self.n_density(ibin)*self.pk_obs.observed_Pgg(zi, k, mu)
        prefactor = 1/(8*(np.pi**2))
        covterm = prefactor * (npobs/(1+npobs))**2
        return covterm 

    def cov(self, ibin, k, mu):
        zmid = self.z_bin_mids[ibin]
        veffS = self.veff(ibin, k, mu)*self.volume_survey(ibin)
        pobs = self.pk_obs.observed_Pgg(zmid, k, mu)
        prefactor = 2*(2*np.pi)**3
        cov = (prefactor/veffS) * (pobs)**2 * (1/k)**3
        return cov

    def P_noise_21(self, z, k, mu, temp_dim=True, beam_term=False):
        if temp_dim==False:
            temp = self.pk_obs.Temperature(z)
        elif temp_dim==True:
            temp = 1
        pref = ((2*np.pi*self.pk_obs.fsky_IM)/(self.pk_obs.f_21*self.pk_obs.t_tot))
        cosmo = ((1+z)**2 * self.pk_obs.cosmo.comoving(z)**2 )/ self.pk_obs.cosmo.Hubble(z)
        T_term = (self.Tsys_func(z)/temp)**2  ##in K
        alpha = self.pk_obs.alpha_SD()
        if beam_term==True:
            beta = self.pk_obs.beta_SD(z, k, mu)
        else:
            beta = np.ones_like(k)
        noise = pref*cosmo*T_term*(alpha/beta**2)
        return noise

    def veff_21cm(self, ibin, k, mu):
        zi = self.IM_z_bin_mids[ibin]
        pobs = self.pk_obs.observed_P_HI(zi, k, mu)
        pnoisy = pobs+self.P_noise_21(zi, k, mu)
        prefactor = 1/(8*(np.pi**2))
        covterm = prefactor *(pobs/pnoisy)**2
        return covterm 
    
    def veff_XC(self, ibin, k, mu):
        print("Entering veff_XC term")
        zi = self.IM_z_bin_mids[ibin]
        pobs_Ig = self.pk_obs.observed_P_HI(zi, k, mu)   #when calling this function, this is the XC spectrum
        pobs_gg = self.pk_obs_gg.observed_P_HI(zi, k, mu)
        pobs_II = self.pk_obs_II.observed_P_HI(zi, k, mu)
        pnoisy_Ig = pobs_Ig
        pnoisy_II = pobs_II + self.P_noise_21(zi, k, mu)
        pnoisy_gg = pobs_gg + self.n_density(ibin)
        covterm = (pobs_Ig**2 / (pnoisy_gg * pnoisy_II + pnoisy_Ig*pnoisy_Ig))
        prefactor = 1/(4*(np.pi**2))
        covterm = prefactor*covterm
        return covterm


class SpectroDerivs:

    def __init__(self, z_array, pk_kmesh, pk_mumesh,
                 fiducial_spectro_obj, bias_samples=['g','g']):
        print("Computing derivatives of Galaxy Clustering Spectro")
        self.observables = fiducial_spectro_obj.observables
        self.bias_samples = bias_samples
        self.fiducial_cosmopars = fiducial_spectro_obj.fiducial_cosmopars
        self.fiducial_spectrobiaspars = fiducial_spectro_obj.fiducial_spectrobiaspars
        if 'IM' in self.observables:
            self.fiducial_IMbiaspars = fiducial_spectro_obj.fiducial_IMbiaspars 
        self.fiducial_PShotpars = fiducial_spectro_obj.PShotpars
        self.fiducial_allpars = fiducial_spectro_obj.fiducial_allpars
        self.fiducial_spectrononlinearpars = fiducial_spectro_obj.fiducial_spectrononlinearpars
        self.fiducial_cosmo = fiducial_spectro_obj.fiducialcosmo
        self.z_array = z_array
        self.cosmology_variations_dict = dict()
        self.pk_kmesh = pk_kmesh
        self.pk_mumesh = pk_mumesh
        self.freeparams = None#cfg.freeparams
        self.feed_lvl = 1 #cfg.settings['feedback']
        #self.get_obs = memory.cache(self.getobs)

    def initialize_obs(self, allpars):
        cosmopars = dict((k, allpars[k]) for k in self.fiducial_cosmopars)
        spectrobiaspars = dict((k, allpars[k]) for k in self.fiducial_spectrobiaspars)
        PShotpars = dict((k, allpars[k]) for k in self.fiducial_PShotpars)
        spectrononlinearpars = dict((k, allpars[k]) for k in self.fiducial_spectrononlinearpars)

        if 'I' in self.bias_samples:
            IMbiaspars = dict((k, allpars[k]) for k in self.fiducial_IMbiaspars)
        if 'I' in self.bias_samples:
            self.pobs = spec_obs.ComputeGalIM(cosmopars=cosmopars,
                                              fiducial_cosmopars=self.fiducial_cosmopars,
                                              spectrobiaspars=spectrobiaspars,
                                              IMbiaspars=IMbiaspars,
                                              PShotpars=PShotpars,
                                              fiducial_cosmo = self.fiducial_cosmo,
                                              bias_samples=self.bias_samples)
        elif 'g' in self.bias_samples:
            self.pobs = spec_obs.ComputeGalSpectro(cosmopars=cosmopars,
                                              fiducial_cosmopars=self.fiducial_cosmopars,
                                              spectrobiaspars=spectrobiaspars,
                                              spectrononlinearpars=spectrononlinearpars,
                                              PShotpars=PShotpars,
                                              fiducial_cosmo = self.fiducial_cosmo,
                                              bias_samples=self.bias_samples)
        strdic = str(sorted(cosmopars.items()))
        hh = hash(strdic)
        self.cosmology_variations_dict[hh] = self.pobs.cosmo
        self.cosmology_variations_dict['hash_'+str(hh)] = strdic
    def get_obs(self, allpars):
        self.initialize_obs(allpars)
        result_array = dict()
        result_array['z_bins'] = self.z_array
        for ii, zzi in enumerate(self.z_array):
            if 'I' in self.bias_samples:
                result_array[ii] = self.pobs.lnpobs_IM(zzi, self.pk_kmesh, self.pk_mumesh)
            elif 'g' in self.bias_samples:
                result_array[ii] = self.pobs.lnpobs_gg(zzi, self.pk_kmesh, self.pk_mumesh)
        return result_array

    def exact_derivs(self, par):
        if 'Ps' in par:
            deriv = dict()
            for ii, zzi in enumerate(self.z_array):
                pgg_obs = self.pobs.observed_Pgg(zzi, self.pk_kmesh, self.pk_mumesh)
                z_bin_mids = self.pobs.gcsp_z_bin_mids
                kron = self.kronecker_bins(par, z_bin_mids, zzi)
                deriv_i = 1/pgg_obs
                deriv[ii] = kron*deriv_i
            return deriv
        else:
            return None

    def kronecker_bins(self, par, zmids, zi):
        ii = np.where(np.isclose(zmids, zi))
        ii = ii[0][0]+1
        pi = par.split('_')
        pi = int(pi[-1])
        if ii==pi:
            kron_delta = 1
        else:
            kron_delta = 0
        return kron_delta 

    def compute_derivs(self, freeparams=dict()):
        derivs=dict()
        if freeparams != dict():
            self.freeparams = freeparams   
        compute_derivs=True
        if compute_derivs:
            tder1 = time()
            print(">> Computing Derivs >>")
            deriv_engine = derivatives(observable = self.get_obs, 
                                       fiducial = self.fiducial_allpars,
                                       special_deriv_function = self.exact_derivs,
                                       freeparams = self.freeparams)
            derivs       = deriv_engine.result
            tder2 = time()
            upt.time_print(feedback_level=self.feed_lvl, min_level=1,
                       text='-->> Derivatives computed in ',
                       time_ini=tder1,
                       time_fin=tder2, instance=self)

        self.derivs = derivs
        return self.derivs
    
    def dlnpobs_dp(self, zi, k, mu, par):
        if par in self.freepars.keys():
            return self.dlnpobs_dcosmop(zi, k, mu, par)
        elif par in self.biaspars.keys():
            return self.dlnpobs_dnuisp(zi, k, mu, par)
        elif par in self.Pspars.keys():
            return self.dlnpobs_dnuisp(zi, k, mu, par)
        else:
            print("WARNING: parameter not contained in specgal instance definition")
            return np.zeros_like(k)

    def dlnpobs_dcosmop(self, zi, k, mu, par):
        if self.fiducialpars[par] != 0.:
            stepsize = self.fiducialpars[par]*self.freepars[par]
        else:
            stepsize = self.freepars[par]

        #Doing forward step
        fwd = copy.deepcopy(self.fiducialpars)
        fwd[par] = fwd[par]+stepsize
        galspec = self.get_obs(fwd)
        pgg_pl = galspec.lnpobs(zi, k, mu)
        #Doing backward step
        bwd = copy.deepcopy(self.fiducialpars)
        bwd[par] = bwd[par]-stepsize
        galspec = self.get_obs(bwd)
        pgg_mn = galspec.lnpobs(zi, k, mu)

        deriv = (pgg_pl - pgg_mn)/(2*stepsize)
        return deriv

    def dlnpobs_dnuisp(self, zi, k, mu, par):
        galspec = self.galspec_fiducial
        if 'lnb' in par:
            bterm = galspec.bterm_fid(zi, bias_sample='g')
            lnb_pl = np.power(bterm, 1+self.eps_nuis)
            lnb_mn = np.power(bterm, 1-self.eps_nuis)
            lnb    = np.log(bterm)
            lnpobs_pl = galspec.lnpobs(zi, k, mu, b_i = lnb_pl)
            lnpobs_mn = galspec.lnpobs(zi, k, mu, b_i = lnb_mn)
            deriv = (lnpobs_pl - lnpobs_mn)/(2*self.eps_nuis*lnb)
        elif 'Ps' in par:
            pobs = galspec.observed_Pgg(zi, k, mu)
            deriv = 1/pobs
        else:
            print("Error: Param name not supported in nuisance parameters")
            deriv = 0
        ## get index in bin
        ii = np.where(np.isclose(self.z_bin_mids, zi))
        ii = ii[0][0]+1
        pi = par.split('_')
        pi = int(pi[-1])
        if ii==pi:
            kron_delta = 1
        else:
            kron_delta = 0
        deriv = kron_delta * deriv
        return deriv

