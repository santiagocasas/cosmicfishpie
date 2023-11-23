# -*- coding: utf-8 -*-
"""CLS

This module contains cls calculations (only LSS atm).

"""
import numpy as np

from copy import deepcopy

import cosmicfishpie.cosmology.cosmology as cosmology
import cosmicfishpie.cosmology.nuisance as nuisance
import cosmicfishpie.LSSsurvey.photo_window as photo_window
import cosmicfishpie.fishermatrix.config as cfg

from scipy.interpolate import CubicSpline

from time import time
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upt

class ComputeGalSpectro:
    ## Class attributes shared among all class instances
    
    def __init__(self, cosmopars, fiducial_cosmopars=None, 
                       spectrobiaspars=None,
                       spectrononlinearpars=None,
                       PShotpars=None, fiducial_cosmo=None, 
                       bias_samples=['g','g'],
                       use_bias_funcs=True):

        tini = time()
        self.feed_lvl = cfg.settings['feedback']
        upt.time_print(feedback_level=self.feed_lvl, min_level=1,
                       text='Entered ComputeGalSpectro',
                       instance=self)

        self.observables = cfg.obs
        #if 'GCsp' not in self.observables:
        #    raise AttributeError("Observables list not defined properly")
        
        self.s8terms = cfg.settings['bfs8terms']
        self.tracer = cfg.settings['GCsp_Tracer']

        if fiducial_cosmopars is None:
            self.fiducial_cosmopars = deepcopy(cfg.fiducialparams)
        else:
            self.fiducial_cosmopars = deepcopy(fiducial_cosmopars)
        if self.fiducial_cosmopars == cfg.fiducialparams:
            try:
                try:
                    self.fiducialcosmo = cfg.fiducialcosmo
                    upt.time_print(feedback_level=self.feed_lvl, min_level=3,
                       text="Fiducial cosmology parameters: {}".format(self.fiducialcosmo.cosmopars),
                       instance=self)
                except:
                    upt.debug_print("Fiducial cosmology from config.py raised an error")
                    #raise
                    try:
                        self.fiducialcosmo = fiducial_cosmo
                        upt.time_print(feedback_level=self.feed_lvl, min_level=3,
                        text="Fiducial cosmology parameters: {}".format(self.fiducialcosmo.cosmopars),
                        instance=self)
                    except:
                        upt.debug_print("Fiducial cosmology from input arguments raised an error")
                        raise
            except:
                print(" >>>>> Fiducial cosmology could not be loaded, recomputing....")
                print(" **** In ComputeGalSpectro: Calculating fiducial cosmology...")
                self.fiducialcosmo = cosmology.cosmo_functions(self.fiducial_cosmopars, cfg.input_type)
        else:
            print("Error: In ComputeGalSpectro fiducial_cosmopars not equal to cfg.fiducialparams")
            raise AttributeError

        self.cosmopars = cosmopars
        if self.cosmopars ==  self.fiducial_cosmopars:
            self.cosmo = self.fiducialcosmo
        else: 
            self.cosmo = cosmology.cosmo_functions(cosmopars, cfg.input_type)

        self.nuisance = nuisance.Nuisance()
        self.gcsp_bias_of_z = self.nuisance.gcsp_bias()
        self.extraPshot = self.nuisance.extra_Pshot_noise()
        self.bias_samples=bias_samples
        self.gcsp_z_bin_mids = self.nuisance.gcsp_zbins_mids()

        # Load the Nuiscance Parameters
        self.fiducial_spectrobiaspars = cfg.Spectrobiasparams
        self.use_bias_funcs = use_bias_funcs 
        if spectrobiaspars is None:
            spectrobiaspars = self.fiducial_spectrobiaspars
        else: 
            self.use_bias_funcs = False  ##If spectrobiaspars are not passed explicitly, use interpolated bias funcs
        self.spectrobiaspars = spectrobiaspars        

        self.fiducial_PShotpars = cfg.PShotparams        
        if PShotpars is None:
            PShotpars = self.fiducial_PShotpars
        self.PShotpars = PShotpars

        self.fiducial_spectrononlinearpars = cfg.Spectrononlinearparams
        if spectrononlinearpars is None:
            spectrononlinearpars = self.fiducial_spectrononlinearpars
        self.spectrononlinearpars = spectrononlinearpars
        
        # create interpolators, for if the asked z is not the z of the zbins (maybe this is stupid)
        if not self.spectrononlinearpars == dict():
            sigmap_input = [self.spectrononlinearpars['sigmap_{}'.format(i)] for i in range(len(self.nuisance.gcsp_zbins_mids()))]
            sigmav_input = [self.spectrononlinearpars['sigmav_{}'.format(i)] for i in range(len(self.nuisance.gcsp_zbins_mids()))]  
            self.sigmap_inter = CubicSpline(self.gcsp_z_bin_mids,sigmap_input)
            self.sigmav_inter = CubicSpline(self.gcsp_z_bin_mids,sigmav_input)

        self.allpars = {**self.cosmopars, **self.spectrobiaspars, **self.PShotpars,**self.spectrononlinearpars}
        self.fiducial_allpars = {**self.cosmopars, **self.spectrobiaspars, **self.PShotpars,**self.spectrononlinearpars} # Seems wrong but is never used. should get it from cfg 

        self.set_internal_kgrid()
        self.activate_terms()
        self.set_spectro_specs()
        tend = time()
        upt.time_print(feedback_level=self.feed_lvl, min_level=1,
                       text='GalSpec initialization done in: ',
                       time_ini=tini, time_fin=tend, instance=self)

    def set_internal_kgrid(self):
        cfg.specs['kmax'] = cfg.specs['kmax_GCsp']
        cfg.specs['kmin'] = cfg.specs['kmin_GCsp']
        kmin_int = 0.001 
        kmax_int = 5  
        self.k_grid = np.logspace(np.log10(kmin_int), np.log10(kmax_int), 1024)
        self.dk_grid = np.diff(self.k_grid)[0]

    def activate_terms(self):
        self.linear_switch = cfg.settings['GCsp_linear']
        self.FoG_switch = cfg.settings['FoG_switch']
        self.APbool = cfg.settings['AP_effect']
        self.fix_cosmo_nl_terms = cfg.settings['fix_cosmo_nl_terms']

    def set_spectro_specs(self):
        self.dz_err = cfg.specs['spec_sigma_dz']

    def qparallel(self,z):
        """
        Function implementing q parallel
        """
        qpar = self.fiducialcosmo.Hubble(z)/self.cosmo.Hubble(z)
        return qpar

    def qperpendicular(self,z):
        """
        Function implementing q parallel
        """
        qper = self.cosmo.angdist(z)/self.fiducialcosmo.angdist(z)
        return qper

    def kpar(self, z, k, mu):
        """
        Args:
        z : The redshift of interest.
        k : wavenumbers at which to compute the power
            spectrum. Must be in units of Mpc^-1/h.
        mu: cosine of the angel between the line of sight and the wavevector
        
        Returns:
            Observed parrallel projection of wavevector onto the line of sight with AP-effect corrected for
        """
        k_par = k*mu*(1/self.qparallel(z))
        return k_par

    def kper(self, z, k, mu):
        """
        Args:
        z : The redshift of interest.
        k : wavenumbers at which to compute the power
            spectrum. Must be in units of Mpc^-1/h.
        mu: cosine of the angel between the line of sight and the wavevector
        
        Returns:
            Observed orthogonal projection of wavevector onto the line of sight with AP-effect corrected for.
        """
        k_per = k*np.sqrt(1-mu**2)*(1/self.qperpendicular(z))
        return k_per

    def k_units_change(self, k):
        """
        Function that rescales the k-array, when the kmax-kmin integration units are fixed in h/Mpc, 
        while the rest of the code is defined in 1/Mpc.
        """
        h_change = self.cosmo.cosmopars['h']/self.fiducialcosmo.cosmopars['h']
        kh = k*h_change
        return kh

    def kmu_alc_pac(self,z,k,mu):
        """Function rescaling k and mu with the Alcock-Paczynski effect

        Parameters
        ----------
        z     : float (numpy array)
                redshift
        k     : float (numpy array)
                wavevector
        mu    : float (numpy array)
                cosine of angle between line of sight and the wavevektor

        Returns
        -------
        array
            [k,mu] rescaled array

        Notes
        -----
        Implements the following equation:

        .. math::


        """
        if self.APbool==False:
            return k, mu
        elif self.APbool==True:
            sum = self.kpar(z,k,mu)**2 + self.kper(z,k,mu)**2
            kap = np.sqrt(sum)
            muap = self.kpar(z,k,mu)/kap
            return kap, muap

    def spec_err_z(self, z, k, mu):
        """
        Args:
        z : The redshift of interest.
        k : wavenumbers at which to compute the power
            spectrum. Must be in units of Mpc^-1.
        mu: cosine of the angel between the line of sight and the wavevector
        
        Returns:
            Calculates the supression of the observed powerspectrum due to the error on spectroscopic redshift determination. 
        """
        err=self.dz_err*(1+z)*(1/self.cosmo.Hubble(z))*self.kpar(z,k,mu)
        return np.exp(-(1/2)*err**2)  ##Gaussian

    def BAO_term(self,z):
        """BAO term

        Parameters
        ----------
        z     : array
                redshift

        Returns
        -------
        float
            Value of BAO term at redshift z

        Notes
        -----
        Implements the following equation:

        .. math::

        """
        if self.APbool==False:
            bao = 1
        else:
            bao = 1/(self.qperpendicular(z)**2 * self.qparallel(z))

        return bao

    # def bterm_key(self):
    #     bstr = self.vary_bias_str
    #     if 'GCsp' in self.observables:
    #         bstr = bstr+'g'
    #     elif 'IM' in self.observables:
    #         bstr = bstr+'IM'
    #     if self.s8terms:
    #         bstr = bstr+'s8'
    #     bstr = bstr+'_'
    #     return bstr


    def bterm_fid(self, z, bias_sample='g'):
        """
        Calculates the fiducial bias term at a given redshift `z`, of either galaxies or intensity mapping.

        Parameters:
        -----------
        z: float
            The redshift value at which to evaluate the bias term.
        bias_sample: str, optional (default='g')
            Specifies whether to compute the galaxy ('g') or intensity mapping ('I') bias term. 

        Returns:
        --------
        float
        The value of the bias term at `z`.
        """
        if bias_sample=='g':
            bfun = self.gcsp_bias_of_z  ##This attribute is created when ComputeGalSpectro is called
            zmidsbins = self.gcsp_z_bin_mids
            bdict = self.spectrobiaspars
        elif bias_sample=='I':
            bfun = self.IM_bias_of_z    ##This attribute is created when ComputeGalIM is called
            zmidsbins = self.IM_z_bin_mids
            bdict = self.IMbiaspars
        if self.use_bias_funcs == True:
            if self.s8terms:
                bterm = bfun(z)*self.fiducialcosmo.sigma8_of_z(z,tracer=self.tracer)
            else:
                bterm = bfun(z)
        elif self.use_bias_funcs == False:
            zii = unu.bisection(zmidsbins, z) ##returns len(zmids)-1 if z above max(zmids)
                                              ## returns 0 if z below min(zmids) 
            zii += 1  ## get bin index from 1 to len(Nbins)
            bkey, bval = self.nuisance.bterm_z_key(zii, zmidsbins, 
                                                   self.fiducialcosmo, 
                                                   bias_sample=bias_sample)
            
            bterm = bdict[bkey]
            if 'ln' in bkey:
                bterm = np.exp(bterm)  ##Exponentiate bias term which is in ln()
        return bterm

    def kaiserTerm(self, z, k, mu, b_i=None, 
                   just_rsd=False, bias_sample='g'):
        """
        Computes the Kaiser redshift-space distortion term.

        Parameters:
            z: Redshift.
            k: Wave number in Mpc^-1.
            mu : cosine of angle between line of sight and the wavevector.
            b_i: External bias term. Defaults to None.
            just_rsd (bool, optional): If True, returns only the RSD term. Otherwise, computes the full Kaiser term. Defaults to False.
            bias_sample (str, optional): Bias term to use from self.bterm_fid(). Possible values: 'g': galaxies, 'I': intensity mapping. Defaults to 'g'.

        Returns:
            The computed Kaiser term.
        """
        bterm = b_i   ##get bs8 as an external parameter, unless it is none, then get it from cosmo
        if b_i is None:
            try:
                bterm = self.bterm_fid(z, bias_sample=bias_sample)
            except KeyError as ke:
                print(" The key {} is not in dictionary. Check observables and parameters being used".format(ke))
                raise ke
        if self.s8terms==True:
            fterm = self.cosmo.fsigma8_of_z(z,k,tracer=self.tracer)
        else:
            fterm = self.cosmo.f_growthrate(z,k,tracer=self.tracer)

        if just_rsd==False:
            kaiser = (bterm+fterm*mu**2)
        elif just_rsd==True:
            kaiser = (1+(fterm/bterm)*mu**2)

        return kaiser


    def FingersOfGod(self,z,k,mu, mode='Lorentz'):
        """
        Calculates the Fingers of God effect in redshift-space power spectra.

        Parameters
        ----------
        z : float
            The redshift value.
        k : float
            The wavenumber in Mpc^-1.
        mu : float
            The cosine of angle between the wavevector and the line-of-sight direction.
        mode : str, optional
            A string parameter indicating the model to use. Defaults to 'Lorentz'.

        Returns
        -------
        float
            The calculated FoG term, which is 1 if either FoG_switch is False or linear_switch is True.
            Otherwise, it depends on the specified mode.
        """
        if (self.FoG_switch == False) or (self.linear_switch == True):
            fog = 1
        elif mode=='Lorentz':
            fog = 1/(1+ (k*mu*self.sigmapNL(z))**2)
        else:
            print("FoG mode not implemented")
            fog=1
        return fog

    def sigmapNL(self, zz):
        """
        Args:
            zz: The redshift value at which to calculate the power spectrum.
        Returns:
            Calculates the variance of the pairwise velocity dispersion. Enters into the FOG effect.

        """
        if self.linear_switch:
            sp = 0
        else:
            sp = np.sqrt(self.P_ThetaTheta_Moments(zz, 2))
            if not self.spectrononlinearpars == dict():
                sp *= self.sigmap_inter(zz)/ np.sqrt(self.P_ThetaTheta_Moments(zz, 0))
        return sp

    def sigmavNL(self,zz, mu):
        """
        Args:
            zz: The redshift value at which to calculate the power spectrum.

        Returns:
            calculates the variance of the displacement field. Enters into the dewigling weight to obtain the (slightly) nonlinear Powerspectrum

        """
        if self.linear_switch:
            sv = 0
        else:
            f0 = self.P_ThetaTheta_Moments(zz, 0)
            f1 = self.P_ThetaTheta_Moments(zz, 1)
            f2 = self.P_ThetaTheta_Moments(zz, 2)
            sv = np.sqrt(f0 + 2*mu**2*f1 + mu**2*f2)
            if not self.spectrononlinearpars == dict():
                sv *= self.sigmav_inter(zz)/ np.sqrt(self.P_ThetaTheta_Moments(zz, 0))
        return sv

    def P_ThetaTheta_Moments(self, zz, moment=0):
        """
        Calculates the angular power spectrum moments of the velocity divergence field, also known as the Theta field.

        Args:
            zz: The redshift value at which to calculate the power spectrum.
            moment: An integer indicating the order of the moment to calculate. Default is 0.

        Returns:
            ptt (float): The power spectrum moment of the velocity divergence field.
        """
        ## TODO: can be optimized by returning interpolating function in z and doing it one time only
        if self.fix_cosmo_nl_terms:
            cosmoF = self.fiducialcosmo
        else:
            cosmoF = self.cosmo
        f_mom = lambda k: (cosmoF.f_growthrate(zz, k)**moment)
        ff = f_mom(self.k_grid).flatten()
        pp = cosmoF.matpow(zz, self.k_grid).flatten()
        integrand = pp*ff
        Int = np.trapz(integrand, x=self.k_grid)
        ptt = (1/(6*np.pi**2)) * Int
        return ptt

    def normalized_pdd(self, z, k):
        """
        Args:
            z: The redshift at which to compute the normalized Powerspectrum
            k: Wavenumber at which to compute the normalized Powerspectrum
        
        Returns:
            The Normalized Powerspectrum
        
        Note: 
            This is not really a normalisation it is solely here to cancel possible sigma8 terms inside of the Kaiserterm (BAO). If the user passes for
            example b times sigma8 as bias then here the sigma8 is canceled out.
        """
        s8_denominator = 1
        if self.s8terms==True:
            s8_denominator = self.cosmo.sigma8_of_z(z,tracer=self.tracer)**2
        
        p_dd = self.cosmo.matpow(z, k,tracer=self.tracer)  ## P_{delta,delta}
        self.p_dd = p_dd/s8_denominator
        return self.p_dd

    def normalized_pnw(self, z, k):
        """
        Args:
            z: The redshift at which to compute the normalized Powerspectrum
            k: Wavenumber at which to compute the normalized Powerspectrum
        
        Returns:
            The Normalized "no-wiggle" Powerspectrum
        
        Note: 
            This is not really a normalisation it is solely here to cancel possible sigma8 terms inside of the Kaiserterm (BAO). If the user passes for
            example b times sigma8 as bias then here the sigma8 is canceled out.
        """
        s8_denominator = 1
        if self.s8terms==True:
            s8_denominator = self.cosmo.sigma8_of_z(z,tracer=self.tracer)**2
        
        p_nw = self.cosmo.nonwiggle_pow(z, k,tracer=self.tracer)  ## P_{delta,delta}
        self.p_nw = p_nw/s8_denominator
        return self.p_nw

    def dewiggled_pdd(self, z, k, mu):
        """"
        Calculates the normalized dewiggled powerspectrum
        
        Args:
        z : float
            The redshift value.
        k : float
            The wavenumber in Mpc^-1.
        mu : float
            The cosine of angle between the wavevector and the line-of-sight direction.
        
        Retruns:
            The dewiggled powerspectrum calculated with the Zeldovic approximation. 
            If the config asks for only linear spectrum this just returns the powerspectrum normalized with either 1 or 1/sigma8^2
        """
    
        if self.linear_switch == True:
            gmudamping=0
        else:
            gmudamping = self.sigmavNL(z, mu)**2
        
        self.p_dd    = self.normalized_pdd(z,k)
        self.p_dd_NW = self.normalized_pnw(z,k)
        self.p_dd_DW = (self.p_dd   * np.exp(-gmudamping   * k**2) +
                        self.p_dd_NW*(1-np.exp(-gmudamping * k**2)))
        return self.p_dd_DW

    def observed_Pgg(self, z, k, mu, b_i=None):
        """galaxy-galaxy spectro power spectrum computation

        Parameters
        ----------
        z   : float
                multipole

        Returns
        -------
        float
            Value of P_gg

        Notes
        -----
        Implements the following equation:

        .. math::

        """
        if self.feed_lvl > 1: print('')
        if self.feed_lvl > 1: print('    Computing Pgg for {}'.format(self.observables))
        tstart = time()

        k = self.k_units_change(k)   #has to be done before spec_err and AP 
        error_z = self.spec_err_z(z,k,mu)  ##before rescaling of k,mu by AP
        k,mu = self.kmu_alc_pac(z,k,mu)

        baoterm = self.BAO_term(z)
        kaiser = self.kaiserTerm(z, k, mu, b_i, bias_sample='g')

        extra_shotnoise = self.extraPshot
        lorentzFoG = self.FingersOfGod(z, k, mu, mode='Lorentz')
        p_dd_DW = self.dewiggled_pdd(z, k, mu)

        pgg_obs = baoterm*(kaiser**2)*p_dd_DW*lorentzFoG*(error_z**2)+extra_shotnoise

        tend = time()
        upt.time_print(feedback_level=self.feed_lvl, min_level=1,
                       text='observed P_gg computation took: ',
                       time_ini=tstart, time_fin=tend, instance=self)
        return pgg_obs

    def lnpobs_gg(self, z, k, mu, b_i=None):
        pobs = self.observed_Pgg(z, k, mu, b_i=b_i)
        return np.log(pobs)


class ComputeGalIM(ComputeGalSpectro):

    def __init__(self, cosmopars, fiducial_cosmopars=None, 
                       spectrobiaspars=None, IMbiaspars=None,
                       PShotpars=None, fiducial_cosmo=None,
                       use_bias_funcs=True, bias_samples=['I','I']):

        super().__init__(cosmopars, fiducial_cosmopars=fiducial_cosmopars, 
                         spectrobiaspars=spectrobiaspars,
                         PShotpars=PShotpars, 
                         fiducial_cosmo=fiducial_cosmo, 
                         use_bias_funcs=True, bias_samples=bias_samples)
        
        tini = time()
        self.feed_lvl = cfg.settings['feedback']
        print('Entered ComputeGalIM')

        if 'IM' not in self.observables:
            raise AttributeError("Observables list not defined properly")
        self.fiducial_IMbiaspars = cfg.IMbiasparams
        self.use_bias_funcs = use_bias_funcs
        if IMbiaspars is None:
            IMbiaspars = self.fiducial_IMbiaspars
        else:
            self.use_bias_funcs = False  ##If IMbiaspars are not passed explicitly, use interpolated bias funcs
        self.IMbiaspars = IMbiaspars
        self.set_IM_specs()
        self.IM_bias_of_z = self.nuisance.IM_bias
        self.IM_z_bin_mids = self.nuisance.IM_zbins_mids()
        print("Bias samples",  self.bias_samples)
        self.allpars = {**self.cosmopars, **self.spectrobiaspars, 
                        **self.IMbiaspars, **self.PShotpars}
        self.fiducial_allpars = {**self.cosmopars, **self.spectrobiaspars, 
                                 **self.fiducial_IMbiaspars, **self.PShotpars}

        tend = time()
        upt.time_print(feedback_level=self.feed_lvl, min_level=1,
                       text='GalIM initialization done in: ',
                       time_ini=tini, time_fin=tend, instance=self)

    def set_IM_specs(self):
        self.Dd = cfg.specs['D_dish']   #Dish diameter in m
        self.lambda_21 = 21/100  ## 21cm in m
        self.fsky_IM =  cfg.specs['fsky_IM'] ## sky fraction for IM
        self.t_tot = cfg.specs['time_tot']*3600 ## * 3600s -> in s
        self.N_d = cfg.specs['N_dish']
        # self.cosmo.c is in km/s
        self.f_21 = ((self.cosmo.c*1000)/self.lambda_21) ##HZ, for MHz: MHz /1e6

    # def IM_bias(self, z):
    #     """
    #         b(z) for HI 21cm IM from nuisance module
    #     """ 
    #     bb = self.nuisance.IM_bias(z)
    #     return bb

    def Omega_HI(self, z):
        o = 4*np.power((1+z), 0.6) * 1e-4
        return o

    def Temperature(self, z):
        """obtaining the temperature (T^2(z)) for the Power Spectrum (PHI(z))"""
        h = self.cosmopars['h']
        H0 = self.cosmo.Hubble(0.)
        temp = 189 * h * (1+z)**2 * (H0/self.cosmo.Hubble(z)) * self.Omega_HI(z)
        ## temperature in mK
        return temp

    def theta_b(self, zz):
        tt = 1.22 * self.lambda_21 * (1+zz)/self.Dd
        return tt

    def alpha_SD(self):
          return 1/self.N_d

    def beta_SD(self, z, k, mu):
        tol = 1.0e-12
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)
        expo = k**2 *(1-mu**2) * self.fiducialcosmo.comoving(z)**2 * self.theta_b(z)**2
        bet = np.exp(-expo/(16.0*np.log(2.0)))
        bet[np.abs(bet) < tol] = tol
        return bet

    def observed_P_HI(self, z, k, mu, bsi_z=None, bsj_z=None, si='I', sj='I'):
        k = self.k_units_change(k)   #has to be done before spec_err and AP 
        error_z = self.spec_err_z(z,k,mu)  ##before rescaling of k,mu by AP
        k,mu = self.kmu_alc_pac(z,k,mu)
        if self.bias_samples is not None:
            si = self.bias_samples[0]
            sj = self.bias_samples[1]
        baoterm = self.BAO_term(z)
        kaiser_bsi = self.kaiserTerm(z, k, mu, bsi_z, bias_sample=si)
        kaiser_bsj = self.kaiserTerm(z, k, mu, bsj_z, bias_sample=sj)
        
        T_HI = self.Temperature(z)
        extra_shotnoise = 0. ##Set to identically zero for the moment, otherwise self.extraPshot
        
        lorentzFoG = self.FingersOfGod(z, k, mu, mode='Lorentz')
        p_dd = self.normalized_pdd(z, k)
        p_dd_DW = self.dewiggled_pdd(z, k, mu)
        
        beam_damping_term_si = self.beta_SD(z, k, mu) if si=='I' else 1
        beam_damping_term_sj = self.beta_SD(z, k, mu) if sj=='I' else 1
        extra_shotnoise_si = np.sqrt(extra_shotnoise) if si=='g' else 0
        extra_shotnoise_sj = np.sqrt(extra_shotnoise) if sj=='g' else 0
        error_z_si = error_z if si=='g' else 1
        error_z_sj = error_z if sj=='g' else 1
        temp_HI_si = T_HI if si=='I' else 1 
        temp_HI_sj = T_HI if sj=='I' else 1

        factors_si = kaiser_bsi*beam_damping_term_si*error_z_si*temp_HI_si+extra_shotnoise_si 
        factors_sj = kaiser_bsj*beam_damping_term_sj*error_z_sj*temp_HI_sj+extra_shotnoise_sj

        p_obs = baoterm*lorentzFoG*p_dd_DW*factors_si*factors_sj

        return p_obs

    def lnpobs_IM(self, z, k, mu, bsi_z=None, bsj_z=None):
        pobs = self.observed_P_HI(z, k, mu, bsi_z=bsi_z, bsj_z=bsj_z)
        return np.log(pobs)
