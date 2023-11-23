# -*- coding: utf-8 -*-
"""COSMOLOGY

This module contains useful cosmological functions.

"""

import os
import sys
import numpy as np
import types
import cosmicfishpie.fishermatrix.config as cfg
from glob import glob
from copy import deepcopy
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from scipy import integrate
from cosmicfishpie.utilities.utils import numerics as unu
from cosmicfishpie.utilities.utils import printing as upr
from warnings import warn
import scipy.constants as sconst
from time import time

from joblib import Memory
cachedir = 'memory_cache'
memory = Memory(cachedir, verbose=0)



def _dcom_func_trapz(zi, interpolfunc):
    zt = np.linspace(0.,zi,100)
    Hzt = interpolfunc(zt)
    dcom = integrate.trapz(1/Hzt, zt)
    return dcom

@memory.cache
def memorize_external_input(cosmopars, fiducialcosmopars, external, extra_settings):
    externalinput = external_input(cosmopars, fiducialcosmopars=fiducialcosmopars, external=external, extra_settings=extra_settings)
    return externalinput
class boltzmann_code:
    hardcoded_Neff = 3.044
    hardcoded_neutrino_mass_fac = 94.07
    def __init__(self,cosmopars,code='camb'):
        """
        Constructor method for the class.
        
        Parameters:
        - cosmopars: The cosmological parameters object to be copied.
        - code: The code to be used (default value is 'camb').
        """
        self.cosmopars = deepcopy(cosmopars)
        self.feed_lvl = cfg.settings['feedback']
        self.settings = cfg.settings
        self.set_cosmicfish_defaults()
        if code=='camb':
            camb_path = os.path.realpath(os.path.join(os.getcwd(),self.settings['camb_path']))
            sys.path.insert(0,camb_path)
            import camb as camb
            self.boltzmann_cambpars = cfg.boltzmann_cambpars
            self.camb_setparams(self.cosmopars,camb)
            self.camb_results(camb)
        elif code=='class' :
            self.boltzmann_classpars = cfg.boltzmann_classpars
            from classy import Class
            self.class_setparams(self.cosmopars,Class)
            self.class_results(Class)
        else:
            print("other Boltzmann code not implemented yet")
            exit()

    def set_cosmicfish_defaults(self):
        """
        Fills up default values in the cosmopars dictionary if the values are not found.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """

        # filling up default values, if value not found in dictionary, then fill it with default value

        # Set default value for Omegam if neither Omegam or omch2 (camb) or omega_cdm (class) are passed
        if not any(par in self.cosmopars for par in ['Omegam','omch2','omega_cdm','Omega_cdm']):
            self.cosmopars['Omegam'] = 0.32

        # Set default value for Omegab if neither Omegab or ombh2 or omega_b or Omega_b or 100omega_b are passed
        if not any(par in self.cosmopars for par in ['Omegab','ombh2','omega_b','Omega_b', '100omega_b']):
            self.cosmopars['Omegab'] = 0.05

        # Set default value for h if neither H0 or h are passed
        if not any(par in self.cosmopars for par in ['H0','h']):
            self.cosmopars['h'] = 0.67

        # Set default value for ns if it is not found in cosmopars
        if not any(par in self.cosmopars for par in ['ns', 'n_s']):
            self.cosmopars['ns'] = self.cosmopars.get('ns', 0.96)

        # Set default value for sigma8 if neither sigma8 or As or logAs or 10^9As are passed
        if not any(par in self.cosmopars for par in ['sigma8','As','logAs', '10^9As', 'ln_A_s_1e10']):
            self.cosmopars['sigma8'] = 0.815583

        # Set default values for w0 and wa if cosmo_model is 'w0waCDM'
        if self.settings['cosmo_model'] == 'w0waCDM':
            if not any(par in self.cosmopars for par in ['w', 'w0_fld']):
                self.cosmopars['w0'] = self.cosmopars.get('w0', -1.0)
            if not any(par in self.cosmopars for par in ['wa', 'wa_fld']):
                self.cosmopars['wa'] = self.cosmopars.get('wa', 0.)

        # Set default value for mnu if Omeganu or omnuh2 or mnu is not found in cosmopars
        if not any(par in self.cosmopars for par in ['Omeganu', 'omnuh2', 'mnu']):
            self.cosmopars['mnu'] = self.cosmopars.get('mnu', 
                                                       self.cosmopars.get('m_nu', 
                                                                          self.cosmopars.get('M_nu', 0.06)))
            
        # Set default value for Neff if it is not found in cosmopars
        self.cosmopars['Neff'] = self.cosmopars.get('Neff', self.cosmopars.get('N_eff', 3.046))
        
        # Set default value for gamma, if it is not found in cosmopars
        # gamma is not used in many places, therefore not needed to add back in cosmopars
        self.gamma = self.cosmopars.get('gamma', 0.545)

    
    def camb_setparams(self,cosmopars,camb):
        """
        Sets the parameters for the CAMB (Code for Anisotropies in the Microwave Background) computation.

        Parameters:
            cosmopars (dict): A dictionary containing the cosmological parameters.
            camb: The CAMB object.

        Returns:
            None

        The function sets the CAMB parameters based on the provided input and performs a basis change if necessary. 
        It then sets the matter power spectrum and additional options for the CAMB computation.

        Note:
            This function assumes that the boltzmann_cambpars and settings dictionaries are available.

        Example usage:
            cosmopars = {'gamma': 0.545, 'k_per_logint': 0.1, 'kmax': 10, 'accurate_massive_neutrino_transfers': True}
            camb = CAMB()
            camb_set_params(cosmopars, camb)
        """
        #Adding hard coded CAMB options
        self.cosmopars=deepcopy(cosmopars)
        self.gamma = self.cosmopars.get('gamma', 0.545)  #default gamma value 0.545, set if gamma not in self.cosmopars
                                                    ## gamma not used in many places, therefore not needed to add back in self.cosmopars
        tini_basis = time()
        self.cambcosmopars = {**self.boltzmann_cambpars['ACCURACY'],**self.boltzmann_cambpars['COSMO_SETTINGS'], **self.boltzmann_cambpars[self.settings['cosmo_model']]}.copy()
        self.kmax_pk=self.cambcosmopars['kmax']
        self.kmin_pk = 1e-4
        self.zmax_pk = self.boltzmann_cambpars['NUMERICS']['zmax']
        self.z_samples = self.boltzmann_cambpars['NUMERICS']['zsamples']
        upr.debug_print(self.cambcosmopars)
        self.cambcosmopars.update(self.cosmopars)
        self.cambcosmopars = self.changebasis_camb(self.cambcosmopars, camb)
        upr.debug_print(self.cambcosmopars)
        tend_basis = time()
        if self.settings['feedback'] > 2: 
            print('')
        if self.settings['feedback'] > 2: 
            print('Basis change took {:.2f} s'.format(tend_basis-tini_basis))
        self.print_camb_params(self.cambcosmopars, feedback=self.settings['feedback'])
        
        self.cambclasspars = camb.set_params(**self.cambcosmopars)
        
        self.camb_zarray = (np.linspace(0. , self.zmax_pk, self.z_samples)[::-1])
        self.cambclasspars.set_matter_power(redshifts=self.camb_zarray, 
                                            k_per_logint=self.cambcosmopars['k_per_logint'], 
                                            kmax=self.cambcosmopars['kmax'],
                                            accurate_massive_neutrino_transfers=self.cambcosmopars['accurate_massive_neutrino_transfers']
                                            )
        #TODO: nonlinear options to be selectable
        self.cambclasspars.NonLinear = camb.model.NonLinear_both
        self.cambclasspars.set_for_lmax(4000, lens_potential_accuracy=1)

    def class_setparams(self,cosmopars,Class):
        tini_basis = time()
        self.classcosmopars  = {**self.boltzmann_classpars['ACCURACY'],**self.boltzmann_classpars['COSMO_SETTINGS'], **self.boltzmann_classpars[self.settings['cosmo_model']]}.copy()
        upr.debug_print(self.classcosmopars)
        upr.debug_print(cosmopars)
        self.classcosmopars.update(self.changebasis_class(self.cosmopars, Class))
        upr.debug_print(self.classcosmopars)
        self.kmax_pk=self.classcosmopars['P_k_max_1/Mpc']
        self.kmin_pk = 1e-4
        self.zmax_pk =self.classcosmopars['z_max_pk']
        tend_basis = time()
        if self.settings['feedback'] > 2: 
            print('')
        if self.settings['feedback'] > 2: 
            print('Basis change took {:.2f} s'.format(tend_basis-tini_basis))
        self.print_class_params(self.classcosmopars, feedback=self.settings['feedback'])

    def changebasis_camb(self,cosmopars, camb):

        cambpars = deepcopy(cosmopars)

        if 'h'      in cambpars: 
            cambpars['H0']    = cambpars.pop('h')*100
        if 'Omegab' in cambpars: 
            cambpars['ombh2'] = cambpars.pop('Omegab')*(cambpars['H0']/100)**2
        if 'Omegak' in cambpars: 
            cambpars['omk']   = cambpars.pop('Omegak')
        if 'w0'     in cambpars: 
            cambpars['w']     = cambpars.pop('w0')
        if 'logAs'  in cambpars: 
            cambpars['As']    = np.exp(cambpars.pop('logAs'))*1.e-10

        upr.debug_print("DEBUG:  --> ", cosmopars)
        shareDeltaNeff = cfg.settings['ShareDeltaNeff']
        cambpars['share_delta_neff']=shareDeltaNeff
        fidNeff = boltzmann_code.hardcoded_Neff

        if 'Neff' in cambpars:
            Neff = cambpars.pop('Neff')
            if shareDeltaNeff:
                cambpars['num_nu_massless'] = Neff - cambpars['num_nu_massive']
            else:
                cambpars['num_nu_massless'] = Neff - fidNeff/3
        
        else:
            Neff = cambpars['num_nu_massive']+cambpars['num_nu_massless']
        
        if shareDeltaNeff:
            g_factor = Neff/3
        else:
            g_factor = fidNeff/3
        
        neutrino_mass_fac  = 94.07
        neutrino_mass_fac  = boltzmann_code.hardcoded_neutrino_mass_fac
        h2 = (cambpars['H0']/100)**2

        if 'mnu' in cambpars: 
            Onu                = cambpars['mnu']/neutrino_mass_fac*(g_factor)** 0.75/h2
            onuh2  = Onu*h2
            cambpars['omnuh2'] = onuh2
        elif 'Omeganu' in cambpars:
            cambpars['omnuh2'] = cambpars.pop('Omeganu')*h2
            onuh2 = cambpars['omnuh2']
        elif 'omnuh2' in cambpars:
            onuh2 = cambpars['omnuh2']

        if 'Omegam' in cambpars: #TO BE GENERALIZED
            cambpars['omch2']  = cambpars.pop('Omegam')*h2-cambpars['ombh2']-onuh2

        rescaleAs = False
        if 'sigma8' in cambpars:
            insigma8 = cambpars['sigma8']
            cambpars['As'] = self.settings.get('rescale_ini_As',2.1e-9)
            cambpars.pop('sigma8')
            rescaleAs = True

        try:
            camb.set_params(**cambpars) # to see which methods are being called: verbose=True
        except camb.CAMBUnknownArgumentError as argument:
            print("Remove parameter from cambparams: ", str(argument))

            #pars= camb.set_params(redshifts=[0.], kmax=50.0,accurate_massive_neutrino_transfers=True,lmax=1000, lens_potential_accuracy=1,**cambpars)

        if rescaleAs is True:
            cambpars['As'] = self.rescale_LP(cambpars,camb,insigma8)

        cambpars['MassiveNuMethod']= 0
        return cambpars


    def rescale_LP(self,cambpars,camb,insigma8) :
        cambpars_LP = cambpars.copy()
        ini_As = self.settings.get('rescale_ini_As',2.1e-9)
        boost  = self.settings.get('rescale_boost',1)
        cambpars_LP['AccuracyBoost'] = boost
        cambpars_LP['lAccuracyBoost'] = boost
        cambpars_LP['lSampleBoost'] = boost
        cambpars_LP['kmax'] = 20
        pars = camb.set_params(redshifts=[0.],**cambpars_LP)
        results = camb.get_results(pars)
        test_sig8=np.array(results.get_sigma8())
        final_As = ini_As*(insigma8/test_sig8[-1])**2.
        get_rescaled_s8 = self.settings.get('get_rescaled_s8',False)
        if get_rescaled_s8 :
            cambpars_rs = cambpars_LP.copy()
            cambpars_rs['As'] = final_As
            pars2 = camb.set_params(redshifts=[0.],**cambpars_rs)
            results2 = camb.get_results(pars2)
            final_sig8 = np.array(results2.get_sigma8())[-1]
        if self.settings['feedback'] > 2 :
            print('AccuracyBoost input = ', cambpars['AccuracyBoost'])
            print('AccuracyBoost rescaling = ',cambpars_LP['lAccuracyBoost'] )
            print('Goal sig8 = ', insigma8)
            print('Reference As = ', ini_As)
            print('Reference sig8 = ', test_sig8)
            print('Rescaled As  = ', final_As)
            if get_rescaled_s8:
                print('Rescaled sig8 = ', final_sig8)
        return final_As

    def changebasis_class(self,cosmopars, Class):

            classpars = deepcopy(cosmopars)
            if 'h'      in classpars: 
                classpars['h']    = classpars.pop('h') 
                h = classpars['h']
            if 'H0'     in classpars: 
                classpars['H0']   = classpars.pop('H0')
                h = classpars['H0'] / 100.

            shareDeltaNeff = cfg.settings['ShareDeltaNeff']
            Neff = classpars.pop('Neff')
            # fidNeff = cfg.fiducialparams['Neff']
            fidNeff = boltzmann_code.hardcoded_Neff

            if shareDeltaNeff:
                classpars['N_ur'] = 2./3.*Neff #This version does not have the discontinuity at Nur = 1.99
                g_factor = Neff/3.
            else:
                classpars['N_ur'] = Neff - fidNeff/3.
                g_factor = fidNeff/3.

            neutrino_mass_fac  = 94.07

            if 'mnu' in classpars: 
                classpars['T_ncdm'] = (4./11.)**(1./3.) * g_factor**(1./4.)
                classpars['Omega_ncdm'] = classpars['mnu'] * g_factor**(0.75) / neutrino_mass_fac / h**2
                classpars.pop('mnu')
                #classpars['m_ncdm'] = classpars.pop('mnu')
                #Om_ncdm = classpars['m_ncdm'] / 93.13858 /h/h
            elif 'Omeganu' in classpars:
                classpars['Omega_ncdm'] = classpars.pop('Omeganu')

            if '100omega_b' in classpars: 
                classpars['omega_b'] = (1/100)*classpars.pop('100omega_b')
            if 'Omegab' in classpars: 
                classpars['Omega_b'] = classpars.pop('Omegab')
            if 'Omegam' in classpars:
                 classpars['Omega_cdm'] = classpars.pop('Omegam') - classpars['Omega_b'] - classpars['Omega_ncdm']

            if 'w0'     in classpars: 
                classpars['w0_fld'] = classpars.pop('w0')
            if 'wa'     in classpars: 
                classpars['wa_fld'] = classpars.pop('wa')
            if 'logAs'   in classpars: 
                classpars['A_s']    = np.exp(classpars.pop('logAs'))*1.e-10
            if '10^9As'   in classpars: 
                classpars['A_s']    = classpars.pop('10^9As')*1.e-9
            if 'ns'     in classpars: 
                classpars['n_s'] = classpars.pop('ns')


            return classpars

    @staticmethod
    def print_camb_params(cambpars, feedback=1):
        if feedback > 2:
            print('')
            print('----CAMB parameters----')
            for key in cambpars:
                print(key+'='+str(cambpars[key]))

    @staticmethod
    def print_class_params(classpars, feedback=1):
        if feedback > 2:
            print('')
            print('----CLASS parameters----')
            for key in classpars:
                print(key+'='+str(classpars[key]))

    def camb_results(self, camb):
        tini_camb = time()
        self.results = types.SimpleNamespace()
        cambres = camb.get_results(self.cambclasspars)
        if self.settings['feedback'] > 2 : 
            tres = time()
        print('Time for Results = ',tres-tini_camb)
        Pk_l, self.results.zgrid, self.results.kgrid = cambres.get_matter_power_interpolator(
                                                     hubble_units=False,
                                                     k_hunit=False,
                                                     var1='delta_tot',
                                                     var2='delta_tot',
                                                     nonlinear=False,
                                                     extrap_kmax=100,
                                                     return_z_k=True)
        Pk_nl, zgrid, kgrid = cambres.get_matter_power_interpolator(
                                                     hubble_units=False,
                                                     k_hunit=False,
                                                     var1='delta_tot',
                                                     var2='delta_tot',
                                                     nonlinear=True,
                                                     extrap_kmax=100,
                                                     return_z_k=True)
        Pk_cb_l, zgrid, kgrid = cambres.get_matter_power_interpolator(
                                                     hubble_units=False,
                                                     k_hunit=False,
                                                     var1='delta_nonu',
                                                     var2='delta_nonu',
                                                     nonlinear=False,
                                                     extrap_kmax=100,
                                                     return_z_k=True)

        self.results.Pk_l = RectBivariateSpline(self.results.zgrid, self.results.kgrid, Pk_l.P(self.results.zgrid, self.results.kgrid))
        self.results.Pk_nl = RectBivariateSpline(self.results.zgrid, self.results.kgrid, Pk_nl.P(self.results.zgrid, self.results.kgrid))
        self.results.Pk_cb_l = RectBivariateSpline(self.results.zgrid, self.results.kgrid, Pk_cb_l.P(self.results.zgrid, self.results.kgrid))
        self.results.h_of_z = InterpolatedUnivariateSpline(self.results.zgrid, cambres.h_of_z(self.results.zgrid))
        self.results.ang_dist = InterpolatedUnivariateSpline(self.results.zgrid, cambres.angular_diameter_distance(self.results.zgrid))
        self.results.com_dist = InterpolatedUnivariateSpline(self.results.zgrid, cambres.comoving_radial_distance(self.results.zgrid))
        self.results.Om_m = InterpolatedUnivariateSpline(self.results.zgrid,
                                 (cambres.get_Omega('cdm', z=self.results.zgrid)+
                                     cambres.get_Omega('baryon', z=self.results.zgrid)+
                                     cambres.get_Omega('nu', z=self.results.zgrid)))
        
        #Calculate the Non linear cb powerspectrum using Gabrieles Approximation
        f_cdm=cambres.get_Omega('cdm',z=0)/self.results.Om_m(0)
        f_b  =cambres.get_Omega('baryon',z=0)/self.results.Om_m(0)
        f_cb =f_cdm+f_b
        f_nu =1-f_cb
        Pk_cross_l = cambres.get_matter_power_interpolator(
                                                     hubble_units=False,
                                                     k_hunit=False,
                                                     var1='delta_nonu',
                                                     var2='delta_nu',
                                                     nonlinear=False,
                                                     extrap_kmax=100,
                                                     return_z_k=False)
        Pk_nunu_l = cambres.get_matter_power_interpolator(
                                                     hubble_units=False,
                                                     k_hunit=False,
                                                     var1='delta_nu',
                                                     var2='delta_nu',
                                                     nonlinear=False,
                                                     extrap_kmax=100,
                                                     return_z_k=False)
        Pk_cb_nl=1/f_cb**2 * (Pk_nl.P(self.results.zgrid, self.results.kgrid)-2 * Pk_cross_l.P(self.results.zgrid, self.results.kgrid)*f_cb*f_nu - Pk_nunu_l.P(self.results.zgrid, self.results.kgrid) * f_nu**2)
        self.results.Pk_cb_nl = RectBivariateSpline(self.results.zgrid, self.results.kgrid, Pk_cb_nl)
        
        if self.settings['feedback'] > 2 : 
            tPk = time()
            print('Time for lin+nonlin Pk = ',tPk - tres)

        P_kz_0 = self.results.Pk_l(0., self.results.kgrid)
        D_g_norm_kz = np.sqrt(self.results.Pk_l(self.results.zgrid, self.results.kgrid)/P_kz_0)

        self.results.D_growth_zk=RectBivariateSpline(self.results.zgrid, self.results.kgrid,
                                          (D_g_norm_kz),
                                          kx=3, ky=3)

        P_cb_kz_0 = self.results.Pk_cb_l(0., self.results.kgrid)
        D_g_cb_norm_kz = np.sqrt(self.results.Pk_cb_l(self.results.zgrid, self.results.kgrid)/P_cb_kz_0)
        self.results.D_growth_cb_zk=RectBivariateSpline(self.results.zgrid, self.results.kgrid,
                                          (D_g_cb_norm_kz),
                                          kx=3, ky=3)

        if self.settings['feedback'] > 2 : 
            tDzk = time()
            print('Time for Growth factor = ',tDzk - tPk)
        
        def f_deriv(k_array,k_fix = False,fixed_k = 1e-3):
            z_array = self.results.zgrid
            if k_fix :
                k_array = np.full((len(k_array)),fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_z = np.array([UnivariateSpline(z_array,self.results.D_growth_zk(z_array,kk),s=0) for kk in k_array]) 
            
            ## Generates arrays f(z) for varying k values
            f_z = np.array([ -(1+z_array)/D_zk(z_array)  * (D_zk.derivative())(z_array) for D_zk in D_z   ])
            return f_z, z_array

        f_z_k_array,z_array = f_deriv(self.results.kgrid)
        self.results.f_growthrate_zk = RectBivariateSpline(z_array,self.results.kgrid,f_z_k_array.T)

        def f_cb_deriv(k_array,k_fix = False,fixed_k = 1e-3):
            z_array = self.results.zgrid
            if k_fix :
                k_array = np.full((len(k_array)),fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_cb_z = np.array([UnivariateSpline(z_array,self.results.D_growth_cb_zk(z_array,kk),s=0) for kk in k_array]) 
            
            ## Generates arrays f(z) for varying k values
            f_cb_z = np.array([ -(1+z_array)/D_cb_zk(z_array)  * (D_cb_zk.derivative())(z_array) for D_cb_zk in D_cb_z   ])
            return f_cb_z, z_array

        f_cb_z_k_array,z_array = f_cb_deriv(self.results.kgrid)
        self.results.f_growthrate_cb_zk = RectBivariateSpline(z_array,self.results.kgrid,f_cb_z_k_array.T)

        if self.settings['feedback'] > 2 : 
            tfzk = time() 
            print('Time for Growth factor = ',tfzk - tDzk)

        def get_sigma8(z_range):
            R=8./ (cambres.Params.H0 / 100.0)
            k = np.linspace(self.kmin_pk,self.kmax_pk,10000)
            sigma_z = np.empty_like(z_range)
            pkz = self.results.Pk_l(z_range,k)
            for i in range(len(sigma_z)):
                integrand = 9*(k*R*np.cos(k*R) - np.sin(k*R))**2 * pkz[i] / k**4 / R**6 / 2 / np.pi**2
                sigma_z[i] = np.sqrt(np.trapz(integrand,k))
            sigm8_z_interp = UnivariateSpline(z_range,sigma_z,s=0)
            return sigm8_z_interp

        def get_sigma8_cb(z_range):
            R=8./ (cambres.Params.H0 / 100.0)
            k = np.linspace(self.kmin_pk,self.kmax_pk,10000)
            sigma_cb_z = np.empty_like(z_range)
            pk_cb_z = self.results.Pk_cb_l(z_range,k)
            for i in range(len(sigma_cb_z)):
                integrand = 9*(k*R*np.cos(k*R) - np.sin(k*R))**2 * pk_cb_z[i] / k**4 / R**6 / 2 / np.pi**2
                sigma_cb_z[i] = np.sqrt(np.trapz(integrand,k))
            sigm8_cb_z_interp = UnivariateSpline(z_range,sigma_cb_z,s=0)  
            return sigm8_cb_z_interp

        self.results.s8_cb_of_z = get_sigma8_cb(self.results.zgrid)
        self.results.s8_of_z = get_sigma8(self.results.zgrid)

        if self.settings['feedback'] > 2: 
            ts8 = time()
            print('Time for Growth factor = ',ts8 - tfzk)
        
        if self.cambcosmopars['Want_CMB'] :
            powers = cambres.get_cmb_power_spectra(CMB_unit='muK')
            self.results.camb_cmb = powers['total']
        tend_camb = time()
        if self.settings['feedback'] > 2: 
            print('Time for CMB = ',tend_camb - ts8)
        if self.settings['feedback'] > 1: 
            print('')
        if self.settings['feedback'] > 1: 
            print('Cosmology computation took {:.2f} s'.format(tend_camb-tini_camb))

    def class_results(self, Class): # Get your CLASS results from here
        self.results = types.SimpleNamespace()
        classres = Class()
        classres.set(self.classcosmopars)
        classres.compute()
        self.Classres = classres
        self.results.h_of_z = np.vectorize(classres.Hubble)
        self.results.ang_dist = np.vectorize(classres.angular_distance)
        self.results.com_dist = np.vectorize(classres.comoving_distance)
        h = classres.h()
        self.results.s8_of_z = np.vectorize(lambda zz: classres.sigma(R=8/h, z=zz))
        self.results.s8_cb_of_z = np.vectorize(lambda zz: classres.sigma_cb(R=8/h, z=zz))
        self.results.Om_m = np.vectorize(classres.Om_m)
        
        #Calculate the Matter fractions for CB Powerspectrum
        f_cdm = classres.Omega0_cdm()/classres.Omega_m()
        f_b = classres.Omega_b()/classres.Omega_m()
        f_cb = f_cdm+f_b
        f_nu = 1-f_cb

        ## rows are k, and columns are z
        ## interpolating function Pk_l (k,z)
        Pk_l, k, z = classres.get_pk_and_k_and_z(nonlinear=False)
        Pk_cb_l, k, z = classres.get_pk_and_k_and_z(only_clustering_species=True,nonlinear=False)        
        self.results.Pk_l = RectBivariateSpline(z[::-1],k,(np.flip(Pk_l,axis=1)).transpose())
        #self.results.Pk_l = lambda z,k: [np.array([classres.pk_lin(kval,z) for kval in k])]
        self.results.Pk_cb_l = RectBivariateSpline(z[::-1],k,(np.flip(Pk_cb_l,axis=1)).transpose())
        #self.results.Pk_cb_l = lambda z,k: [np.array([classres.pk_cb_lin(kval,z) for kval in k])]


        self.results.kgrid = k
        self.results.zgrid = z[::-1]
        
        ## interpolating function Pk_nl (k,z)
        Pk_nl, k, z = classres.get_pk_and_k_and_z(nonlinear=cfg.settings['nonlinear'])
        self.results.Pk_nl = RectBivariateSpline(z[::-1],k,(np.flip(Pk_nl,axis=1)).transpose())

        tk, k ,z = classres.get_transfer_and_k_and_z()
        T_cb = (f_b*tk['d_b']+f_cdm*tk['d_cdm'])/f_cb
        T_nu = tk['d_ncdm[0]']

        pm = classres.get_primordial()
        pk_prim = UnivariateSpline(pm['k [1/Mpc]'],pm['P_scalar(k)'])(k)*(2.*np.pi**2)/np.power(k,3)
        
        pk_cnu  = T_nu * T_cb * pk_prim[:,None]
        pk_nunu = T_nu * T_nu * pk_prim[:,None]
        Pk_cb_nl= 1./f_cb**2 * (Pk_nl-2*pk_cnu*f_nu*f_cb-pk_nunu*f_nu*f_nu)

        self.results.Pk_cb_nl = RectBivariateSpline(z[::-1],k,(np.flip(Pk_cb_nl,axis=1)).transpose())

        def create_growth() :
            z_ = self.results.zgrid
            pk_flipped = np.flip(Pk_l,axis=1).T
            D_growth_zk = RectBivariateSpline(z_,k,np.sqrt(pk_flipped/pk_flipped[0,:]))
            return D_growth_zk

        self.results.D_growth_zk = create_growth()

        def f_deriv(k_array,k_fix = False,fixed_k = 1e-3):
            z_array = np.linspace(0,classres.pars['z_max_pk'],100)
            if k_fix :
                k_array = np.full((len(k_array)),fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_z = np.array([UnivariateSpline(z_array,self.results.D_growth_zk(z_array,kk),s=0) for kk in k_array]) 
            
            ## Generates arrays f(z) for varying k values
            f_z = np.array([ -(1+z_array)/D_zk(z_array)  * (D_zk.derivative())(z_array) for D_zk in D_z   ])
            return f_z, z_array

        f_z_k_array,z_array = f_deriv(self.results.kgrid)
        f_g_kz = RectBivariateSpline(z_array,self.results.kgrid,f_z_k_array.T)
        self.results.f_growthrate_zk = f_g_kz

        def create_growth_cb() :
            z_ = self.results.zgrid
            pk_flipped = np.flip(Pk_cb_l,axis=1).T
            D_growth_zk = RectBivariateSpline(z_,k,np.sqrt(pk_flipped/pk_flipped[0,:]))
            return D_growth_zk

        self.results.D_growth_cb_zk = create_growth_cb()

        def f_cb_deriv(k_array,k_fix = False,fixed_k = 1e-3):
            z_array = np.linspace(0,classres.pars['z_max_pk'],100)
            if k_fix :
                k_array = np.full((len(k_array)),fixed_k)
            ## Generates interpolaters D(z) for varying k values
            D_cb_z = np.array([UnivariateSpline(z_array,self.results.D_growth_cb_zk(z_array,kk),s=0) for kk in k_array]) 
            
            ## Generates arrays f(z) for varying k values
            f_cb_z = np.array([ -(1+z_array)/D_cb_zk(z_array)  * (D_cb_zk.derivative())(z_array) for D_cb_zk in D_cb_z   ])
            return f_cb_z, z_array

        f_cb_z_k_array,z_array = f_cb_deriv(self.results.kgrid)
        f_g_cb_kz = RectBivariateSpline(z_array,self.results.kgrid,f_cb_z_k_array.T)
        self.results.f_growthrate_cb_zk = f_g_cb_kz


class external_input:
    def __init__(self, cosmopars, fiducialcosmopars=dict(), 
                                  external=dict(), extra_settings=dict()):
        self.cosmopars = cosmopars
        self.fiducialpars = fiducialcosmopars
        self.feed_lvl = cfg.settings['feedback']
        self.external = external #cfg.external
        self.settings = extra_settings #cfg.settings
        self.activate_MG = None
        if self.settings['external_activateMG'] is True or self.settings['activateMG']=='external':
            self.activate_MG = 'external'
            upr.debug_print("********EXTERNAL: activateMG = 'external'")
        if not fiducialcosmopars or not external or not extra_settings:
            raise ValueError("The external_input class has been initialized wrongly")
        self.directory = self.external['directory']  ##['baseline_cosmo/']
        self.param_names = self.external['paramnames']  ## ['Om', 'Ob', ....] ##have to be the same as keys in cosmopars dict
        self.folder_param_names = self.external['folder_paramnames']  ## ['Om', 'Ob', ....] ##have to be the same as keys in cosmopars dict
        self.folder_param_dict = dict(zip(self.param_names, self.folder_param_names))
        self.epsilon_values = self.external['eps_values'] ### [0.01, 0.1,...]
        self.epsilon_names = 'eps_'
        self.signstrings = {-1.0:'mn', 1.0:'pl'}
        param_folder_string=self.get_param_string_from_value(cosmopars)
        upr.time_print(feedback_level=self.feed_lvl, min_level=2,
                    text='-**-> File folder used: {:s}'.format(param_folder_string))
        self.cb_files_on = False
        if cfg.settings['GCsp_Tracer'] == 'clustering' or cfg.settings['GCph_Tracer'] == 'clustering':
            self.cb_files_on = True
        self.load_txt_files(parameter_string=param_folder_string)
        self.calculate_interpol_results(parameter_string=param_folder_string)

    def load_txt_files(self,parameter_string='fiducial_eps_0'):
        self.input_arrays=dict()
        z_grid_filename='z_values_list.txt'  #just as a safeguard value here
        # check if z_list exists
        z_arr_file = self.external['file_names']['z_arr']
        k_arr_file = self.external['file_names']['k_arr']
        self.k_arr_special_file = self.external['file_names'].get('k_arr_special', None)
        H_z_file = self.external['file_names']['H_z']
        s8_z_file = self.external['file_names']['s8_z']
        D_zk_file = self.external['file_names']['D_zk']
        f_zk_file = self.external['file_names']['f_zk']
        Pl_zk_file = self.external['file_names']['Pl_zk']
        Pnl_zk_file = self.external['file_names']['Pnl_zk']
        SigWL_zk_file = self.external['file_names'].get('SigmaWL', None)


        upr.debug_print(os.path.join(self.directory,parameter_string,z_arr_file+'.*'))
        try:
            z_grid_filename = glob(os.path.join(self.directory,parameter_string,z_arr_file+'.*'))[0]
        except IndexError:
            print("Folder or path not correctly specified: "+str(os.path.join(self.directory,parameter_string,z_arr_file+'.*')))
            raise
        if os.path.isfile(z_grid_filename):
            self.input_arrays[('z_grid',parameter_string)]=  np.loadtxt(z_grid_filename)
        else:
            z_grid_filename = glob(os.path.join(self.directory,'fiducial_eps_0',z_arr_file+'.*'))[0]
            self.input_arrays[('z_grid',parameter_string)]=  np.loadtxt(z_grid_filename)
        # check if k_list exists
        k_grid_filename = glob(os.path.join(self.directory,parameter_string,k_arr_file+'.*'))[0]
        if os.path.isfile(k_grid_filename):
            self.input_arrays[('k_grid',parameter_string)]=  np.loadtxt(k_grid_filename)
        else:
            k_grid_filename = glob(os.path.join(self.directory,'fiducial_eps_0',k_arr_file+'.*'))[0]
            self.input_arrays[('k_grid',parameter_string)]=  np.loadtxt(k_grid_filename)
        if self.k_arr_special_file is not None:
            k_grid_special_filename = glob(os.path.join(self.directory,'fiducial_eps_0',self.k_arr_special_file+'.*'))[0]
            self.input_arrays[('k_grid_special', parameter_string)]=  np.loadtxt(k_grid_special_filename)
        # check if background_Hz list exists, if not, take fiducial one
        # (to allow for easier import of parameters that do not affect background)
        Hz_filename = glob(os.path.join(self.directory,parameter_string,H_z_file+'.*'))[0]
        if os.path.isfile(Hz_filename):
            self.input_arrays[('H_z',parameter_string)] =  np.loadtxt(Hz_filename)
        else:
            Hz_filename = glob(os.path.join(self.directory,'fiducial_eps_0',H_z_file+'.*'))[0]
            self.input_arrays[('H_z',parameter_string)] =  np.loadtxt(Hz_filename)

        self.input_arrays[('s8_z',parameter_string)]=    np.loadtxt(glob(os.path.join(self.directory,parameter_string,s8_z_file+'.*'  ))[0])
        self.input_arrays[('D_zk',parameter_string)]=    np.loadtxt(glob(os.path.join(self.directory,parameter_string,D_zk_file+'.*'))[0])
        self.input_arrays[('f_zk',parameter_string)]=    np.loadtxt(glob(os.path.join(self.directory,parameter_string,f_zk_file+'.*'))[0])
        self.input_arrays[('Pkl_zk',parameter_string)]=  np.loadtxt(glob(os.path.join(self.directory,parameter_string,Pl_zk_file+'.*' ))[0])
        self.input_arrays[('Pknl_zk',parameter_string)]= np.loadtxt(glob(os.path.join(self.directory,parameter_string,Pnl_zk_file+'.*'))[0])
        if SigWL_zk_file is not None:
            self.input_arrays[('SigWL_zk',parameter_string)]= np.loadtxt(glob(os.path.join(self.directory,parameter_string,SigWL_zk_file+'.*'))[0])
        if self.cb_files_on:
            s8cb_z_file = self.external['file_names']['s8cb_z']
            fcb_zk_file = self.external['file_names']['fcb_zk']
            Plcb_zk_file = self.external['file_names']['Plcb_zk']
            Pnlcb_zk_file = self.external['file_names']['Pnlcb_zk']
            self.input_arrays[('s8cb_z',parameter_string)]=    np.loadtxt(glob(os.path.join(self.directory,parameter_string, s8cb_z_file+'.*'  ))[0])
            self.input_arrays[('fcb_zk',parameter_string)]=    np.loadtxt(glob(os.path.join(self.directory,parameter_string, fcb_zk_file+'.*'))[0])
            self.input_arrays[('Pklcb_zk',parameter_string)]=  np.loadtxt(glob(os.path.join(self.directory,parameter_string, Plcb_zk_file+'.*' ))[0])
            self.input_arrays[('Pknlcb_zk',parameter_string)]= np.loadtxt(glob(os.path.join(self.directory,parameter_string, Pnlcb_zk_file+'.*'))[0])
        if upr.debug:
            for stri in ['z_grid', 'k_grid', 'H_z', 's8_z', 'D_zk', 'f_zk', 'Pkl_zk', 'Pknl_zk']:
                upr.debug_print(stri+' grid shape : '+str(self.input_arrays[(stri,parameter_string)].shape) )



    def get_param_string_from_value(self,cosmopars):
        rel_tol = 1e-5
        for parname in self.param_names:
            if np.isclose(cosmopars[parname],self.fiducialpars[parname], rtol=rel_tol) is False:
                if self.fiducialpars[parname] != 0:
                    delta_eps = (cosmopars[parname]/self.fiducialpars[parname])-1.0
                elif self.fiducialpars[parname] == 0.:
                    delta_eps = (cosmopars[parname]-self.fiducialpars[parname])
                eps_sign = np.sign(delta_eps)
                sign_string = self.signstrings[eps_sign]
                #print('delta_eps before = {:.16f}'.format(delta_eps))
                eps_vals = np.array(self.external['eps_values'])
                allowed_eps_vals = np.concatenate((eps_vals, -eps_vals, np.array([0])))
                #print('delta_eps before = {:.6f}'.format(delta_eps))
                delta_eps = unu.round_decimals_up(delta_eps)
                delta_eps = unu.closest(allowed_eps_vals, delta_eps)
                #print('delta_eps after = {:.6f}'.format(delta_eps))
                eps_string='{:.1E}'.format(abs(delta_eps))
                eps_string = eps_string.replace(".","p")
                if self.external['E-00'] is False:
                    eps_string = eps_string.replace("E-0","E-")
                folder_parname = self.folder_param_dict[parname]
                param_folder_string = folder_parname+'_'+sign_string+'_'+'eps'+'_'+eps_string
                break
            else:
                param_folder_string = 'fiducial_eps_0'
        return param_folder_string

    def calculate_interpol_results(self, parameter_string='fiducial_eps_0'):
        pk_units_factor = 1 ## units in all the code are in Mpc or Mpc^-1
        r_units_factor = 1
        k_units_factor = 1
        c = sconst.speed_of_light/1000
        if self.external is not None:
            k_units = self.external['k-units']
            r_units = self.external['r-units']
            if k_units=='h/Mpc':
                pk_units_factor = (1/self.cosmopars['h'])**3
                k_units_factor = self.cosmopars['h']
            if r_units=='Mpc/h':
                r_units_factor = (1/self.cosmopars['h'])
            elif r_units=='km/s/Mpc':
                r_units_factor = c

        self.results = types.SimpleNamespace()
        self.results.zgrid = self.input_arrays[('z_grid',parameter_string)].flatten()
        self.results.kgrid = (k_units_factor)*self.input_arrays[('k_grid',parameter_string)].flatten()
        if self.k_arr_special_file is not None:
            self.results.kgrid_special = (k_units_factor)*self.input_arrays[('k_grid_special', parameter_string)].flatten()
        else:
            self.results.kgrid_special = self.results.kgrid
        self.results.h_of_z = InterpolatedUnivariateSpline(self.results.zgrid,
                                               (1/r_units_factor)*
                                               self.input_arrays[('H_z',
                                               parameter_string)].flatten())
        dcom_arr = np.array([_dcom_func_trapz(zii, self.results.h_of_z) for zii in self.results.zgrid])
        self.results.com_dist = InterpolatedUnivariateSpline(self.results.zgrid, dcom_arr)
        self.results.ang_dist = InterpolatedUnivariateSpline(self.results.zgrid,
                                                 dcom_arr/(1+self.results.zgrid))

        ky_ord = 3
        kx_ord = 3
        self.results.D_growth_zk = RectBivariateSpline(self.results.zgrid,
                                          self.results.kgrid_special,
                                       self.input_arrays[('D_zk', parameter_string)],
                                       kx=kx_ord, ky=ky_ord)
        self.results.f_growthrate_zk = RectBivariateSpline(self.results.zgrid,
                                          self.results.kgrid_special,
                                       self.input_arrays[('f_zk', parameter_string)],
                                       kx=kx_ord, ky=ky_ord)
        self.results.s8_of_z = InterpolatedUnivariateSpline(self.results.zgrid,
                                                self.input_arrays[('s8_z',
                                                parameter_string)].flatten())
        self.results.Pk_l = RectBivariateSpline(self.results.zgrid,
                                          self.results.kgrid,
                                       pk_units_factor*(self.input_arrays[('Pkl_zk',
                                       parameter_string)]),
                                       kx=kx_ord, ky=ky_ord)
        self.results.Pk_nl = RectBivariateSpline(self.results.zgrid,
                                          self.results.kgrid,
                                       pk_units_factor*(self.input_arrays[('Pknl_zk',
                                       parameter_string)]),
                                       kx=kx_ord, ky=ky_ord)
        if self.cb_files_on:
            self.results.f_growthrate_cb_zk = RectBivariateSpline(self.results.zgrid,
                                            self.results.kgrid_special,
                                        self.input_arrays[('fcb_zk', parameter_string)],
                                        kx=kx_ord, ky=ky_ord)
            self.results.s8_cb_of_z = InterpolatedUnivariateSpline(self.results.zgrid,
                                                    self.input_arrays[('s8cb_z',
                                                    parameter_string)].flatten())
            self.results.Pk_cb_l = RectBivariateSpline(self.results.zgrid,
                                            self.results.kgrid,
                                        pk_units_factor*(self.input_arrays[('Pklcb_zk',
                                        parameter_string)]),
                                        kx=kx_ord, ky=ky_ord)
            self.results.Pk_cb_nl = RectBivariateSpline(self.results.zgrid,
                                            self.results.kgrid,
                                        pk_units_factor*(self.input_arrays[('Pknlcb_zk',
                                        parameter_string)]),
                                        kx=kx_ord, ky=ky_ord)
        if self.activate_MG == 'external':
            self.results.SigWL_zk = RectBivariateSpline(self.results.zgrid,
                                        self.results.kgrid,
                                        (self.input_arrays[('SigWL_zk',
                                        parameter_string)]),
                                        kx=kx_ord, ky=ky_ord)
        ### Reset the zgrid and kgrid to the fiducial ones, in case one parameter variation had a different one
        #self.results.zgrid = self.input_arrays[('z_grid','fiducial_eps_0')].flatten()
        #self.results.kgrid = self.input_arrays[('k_grid','fiducial_eps_0')].flatten()

class cosmo_functions:

    c = sconst.speed_of_light/1000  ##speed of light in km/s
    def __init__(self, cosmopars, input=None):
        self.settings = cfg.settings
        self.fiducialcosmopars = cfg.fiducialparams
        self.input = input
        if input is None:
            input = cfg.input_type
        if input=='camb':
            cambresults = boltzmann_code(cosmopars,code='camb')
            self.code = 'camb'
            self.results = cambresults.results
            self.kgrid = cambresults.results.kgrid
            self.cosmopars = cambresults.cosmopars
            self.cambcosmopars = cambresults.cambclasspars
        elif input=='external':
            self.external = cfg.external
            ## filter settings which contain the word 'external_'
            #extra_settings = dict([[kk, self.settings[kk]] for kk in self.settings.keys() if 'external_' in kk])
            #externalinput = external_input(cosmopars,  fiducialcosmopars=self.fiducialcosmopars, 
            #                                external=self.external, extra_settings=extra_settings)
            if self.settings['memorize_cosmo']:
                externalinput = memorize_external_input(cosmopars, self.fiducialcosmopars, 
                                                     self.external, self.settings)
            else:
                externalinput = external_input(cosmopars, 
                                               fiducialcosmopars=self.fiducialcosmopars, 
                                               external=self.external, 
                                               extra_settings=self.settings)
            self.code = 'external'
            self.results = externalinput.results
            self.kgrid = externalinput.results.kgrid
            self.cosmopars = externalinput.cosmopars
        elif input=='class':
            classresults = boltzmann_code(cosmopars,code='class')
            self.code = 'class'
            self.results = classresults.results
            self.Classres = classresults.Classres
            self.kgrid = classresults.results.kgrid
            self.cosmopars = classresults.cosmopars
            self.classcosmopars = classresults.classcosmopars
        else:
            print(input, ":  This input type is not implemented yet")

    def Hubble(self,z, physical=False):
        """Hubble function

        Parameters
        ----------
        z     : float
                redshift

        physical: bool
                Default False, if True, return H(z) in (km/s/Mpc).
        Returns
        -------
        float
            Hubble function values (Mpc^-1) at the redshifts of the input redshift

        """
        prefactor=1
        if physical:
            prefactor = self.c


        hubble = prefactor*self.results.h_of_z(z)

        return hubble

    def E_hubble(self,z):
        """E(z) dimensionless Hubble function

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Dimensionless E(z) Hubble function values at the redshifts of the input redshift

        """

        H0  = self.Hubble(0.)
        Eofz = self.Hubble(z)/H0

        return Eofz

    def angdist(self,z):
        """Angular diameter distance

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Angular diameter distance values at the redshifts of the input redshift

        """

        dA = self.results.ang_dist(z)

        return dA

    def matpow(self, z, k, nonlinear=False, tracer='matter'):
        """Calculates the power spectrum of a given tracer quantity at a specific redshift and wavenumber.

    Parameters
    ----------
    z : float
        The redshift of interest.
        
    k : array_like
        An array of wavenumbers at which to compute the power spectrum. These must be in units of 1/Mpc and 
        should be sorted in increasing order.
        
    nonlinear : bool, optional
        A boolean indicating whether or not to include nonlinear corrections to the matter power spectrum. The default
        value is False.
        
    tracer : str, optional
        A string indicating which trace quantity to use for computing the power spectrum. If this argument is "matter" 
        or anything other than "clustering", the power spectrum functions `Pmm` will be used to compute the power 
        spectrum. If the argument is "clustering", the power spectrum function `Pcb` will be used instead. The default
        value is "matter".
    
    Returns
    -------
    np.ndarray:
        Array containing the calculated power spectrum values. 
    
    Warnings
    --------
    If `tracer` is not "matter" or "clustering", a warning message is printed to the console saying the provided tracer was not 
    recognized and the function defaults to using `Pmm` to calculate the power spectrum of matter.
    """
        if tracer == 'clustering':
            return self.Pcb(z,k,nonlinear=nonlinear)
        
        if tracer != 'matter':
            warn('Did not recognize tracer: reverted to matter')
        return self.Pmm(z,k,nonlinear=nonlinear)

    def Pmm(self, z, k, nonlinear = False):
        """ Compute the power spectrum of the total matter species  (MM) at a given redshift and wavenumber.

    Args:
        self: An instance of the current class.
        z: The redshift at which to compute the MM power spectrum.
        k: The wavenumber at which to compute the MM power spectrum in 1/Mpc.
        nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

    Returns:
        float: The value of the MM power spectrum at the given redshift and wavenumber.
    """     
        if nonlinear is True:
            power = self.results.Pk_nl(z, k, grid=False)
        elif nonlinear is False:
            power = self.results.Pk_l(z, k, grid=False)
        return power        

    def Pcb(self, z, k, nonlinear = False):
        """ Compute the power spectrum of the clustering matter species  (CB) at a given redshift and wavenumber.

    Args:
        self: An instance of the current class.
        z: The redshift at which to compute the CB power spectrum.
        k: The wavenumber at which to compute the CB power spectrum in 1/Mpc.
        nonlinear (bool, optional): If True, include nonlinear effects in the computation. Default is False.

    Returns:
        The value of the CB power spectrum at the given redshift and wavenumber.
    """        
        if nonlinear is True:
            power = self.results.Pk_cb_nl(z, k, grid=False)
        elif nonlinear is False:
            power = self.results.Pk_cb_l(z, k, grid=False)
        return power

    def nonwiggle_pow(self, z, k, nonlinear = False,
                  tracer = 'matter'):
        """Calculate the power spectrum at a specific redshift and wavenumber,
    after smoothing to remove baryonic acoustic oscillations (BAO).

    Args:
        z: The redshift of interest.
        k: An array of wavenumbers at which to compute the power
            spectrum. Must be in units of Mpc^-1/h. Should be sorted in
            increasing order.
        nonlinear: Whether to include nonlinear corrections
            to the matter power spectrum. Default is False.
        tracer: Which perturbations to use for computing
            the power spectrum. Options are 'matter' or 'clustering'.
            Default is 'matter'.

    Returns:
        An array of power spectrum values corresponding to the
        input wavenumbers. Units are (Mpc/h)^3.

    Notes:
        This function computes the power spectrum of a given tracer quantity
        at a specific redshift, using the matter power spectrum function
        `matpow`. It then applies a Savitzky-Golay filter to smooth out the
        BAO features in the power spectrum. This is done by first taking the
        natural logarithm of the power spectrum values at a set of logarithmic
        wavenumbers spanning from `kmin_loc` to `kmax_loc`. The smoothed power
        spectrum is then returned on a linear (not logarithmic) grid of
        wavenumbers given by the input array `k`.
    """        
        unitsf = self.cosmopars['h']
        kmin_loc = unitsf*self.settings['savgol_internalkmin']
        kmax_loc  = unitsf*np.max(self.kgrid)
        loc_samples = self.settings['savgol_internalsamples']
        log_kgrid_loc = np.linspace(np.log(kmin_loc),np.log(kmax_loc), loc_samples) 
        poly_order = self.settings['savgol_polyorder']
        dlnk_loc = np.mean(log_kgrid_loc[1:]-log_kgrid_loc[0:-1])
        savgol_width = self.settings['savgol_width']
        n_savgol = int(np.round(savgol_width/np.log(1+dlnk_loc)))
        upr.debug_print(n_savgol)
        upr.debug_print(savgol_width)
        upr.debug_print(dlnk_loc)
        upr.debug_print(kmin_loc)
        upr.debug_print(kmax_loc)
        intp_p=InterpolatedUnivariateSpline(log_kgrid_loc,
                        np.log(self.matpow(z,np.exp(log_kgrid_loc),
                                           nonlinear=nonlinear,tracer=tracer).flatten()),
                        k=1)
        pow_sg = savgol_filter(intp_p(log_kgrid_loc), n_savgol, poly_order)
        intp_pnw = InterpolatedUnivariateSpline(np.exp(log_kgrid_loc),
                                                np.exp(pow_sg), k=1)
        return intp_pnw(k)

    def comoving(self,z):
        """Comoving distance

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Comoving distance values at the redshifts of the input redshift

        """

        chi = self.results.com_dist(z)

        return chi

    def sigma8_of_z(self, z,tracer='matter'):
        """sigma_8

        Parameters
        ----------
        z     : float
                redshift
        tracer: String
                either 'matter' if you want sigma_8 calculated from the total matter powerspectrum or 'clustering' if you want it from the Powerspectrum with massive neutrinos substracted 
        Returns
        -------
        float
            The Variance of the matter perturbation smoothed over a scale of 8 Mpc/h

        """
        if tracer == 'clustering':
            return self.results.s8_cb_of_z(z)
        if tracer != 'matter':
            warn('Did not recognize tracer: reverted to matter')
        
        return self.results.s8_of_z(z)

    def growth(self,z,k=None):
        """Growth factor

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth factor values at the redshifts of the input redshift

        """
        if k is None:
            k=0.0001
        Dg  = self.results.D_growth_zk(z, k, grid=False)

        return Dg

    def Omegam_of_z(self,z):
        """Omega matter fraction as a function of redshift

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Omega matter (total) at the redshifts of the input redshift `z`


        Notes
        -----
        Assumes standard matter evolution
        Implements the following equation:

        .. math::
            Omega_m(z) = Omega_{m,0}*(1+z)^3 / E^2(z)
        """
        omz = 0
        if self.input=='external':
            omz = (self.cosmopars['Omegam']*(1+z)**3)/self.E_hubble(z)**2
        else:
            omz = self.results.Om_m(z)
            
        return omz

    def f_growthrate(self,z, k=None, gamma=False,tracer='matter'):
        """Growth rate in LCDM gamma approximation

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth rate values at the redshifts of the input redshift,
            using self.gamma as gamma value.


        Notes
        -----
        Implements the following equation:

        .. math::
            f(z) = Omega_m(z)^{\gamma}
        """
        if k is None:
            k=0.0001

        if tracer == 'clustering':
            fg  = self.results.f_growthrate_cb_zk(z, k, grid=False)
            return fg
        if tracer != 'matter':
            warn('Did not recognize tracer: reverted to matter')

        if gamma is False:
            fg  = self.results.f_growthrate_zk(z, k, grid=False)
        else:
            #Assumes standard Omega_matter evolution in z 
            fg = np.power(self.Omegam_of_z(z), self.gamma)

        return fg

    def fsigma8_of_z(self, z, k=None, gamma=False,tracer='matter'):
        """Growth rate in LCDM gamma approximation

        Parameters
        ----------
        z     : float
                redshift

        Returns
        -------
        float
            Growth rate values at the redshifts of the input redshift,
            using self.gamma as gamma value.


        Notes
        -----
        Implements the following equation:

        .. math::
            f(z) = Omega_m(z)^{\gamma}
        """
        #Assumes standard Omega_matter evolution in z
        fs8  = self.f_growthrate(z, k, gamma,tracer=tracer)*self.sigma8_of_z(z,tracer=tracer)

        return fs8


    def SigmaMG(self, z, k):
        Sigma = np.array(1)
        if self.settings['activateMG']=='late-time':
            E11 = self.cosmopars['E11']
            E22 = self.cosmopars['E22']
            # TODO: Fix for non-flat models
            Omega_DE = (1-self.Omegam_of_z(z))
            mu = 1 + E11*Omega_DE
            eta = 1 + E22*Omega_DE
            Sigma = (mu/2)*(1+eta)
        elif self.settings['external_activateMG'] is True or self.settings['activateMG']=='external':
            Sigma = self.results.SigWL_zk(z, k, grid=False)

        return Sigma

    def cmb_power(self,lmin,lmax,obs1,obs2):

        if self.code == 'camb' :
            if self.cambcosmopars.Want_CMB :
                print('CMB Spectrum not computed')
                return
        elif self.code == 'class' :
            if 'tCl' in self.classcosmopars['output'] :
                print('CMB Spectrum not computed')
                return                
        else :
            ells = np.arange(lmin,lmax)

            norm_fac = 2*np.pi/(ells*(ells+1))

            if obs1+obs2 == 'CMB_TCMB_T':
                cls = norm_fac*self.results.camb_cmb[lmin:lmax,0]
            elif obs1+obs2 == 'CMB_ECMB_E':
                cls = norm_fac*self.results.camb_cmb[lmin:lmax,1]
            elif obs1+obs2 == 'CMB_BCMB_B':
                cls = norm_fac*self.results.camb_cmb[lmin:lmax,2]
            elif (obs1+obs2 == 'CMB_TCMB_E') or (obs1+obs2 == 'CMB_ECMB_T'):
                cls = norm_fac*self.results.camb_cmb[lmin:lmax,3]
            else:
                cls = np.array([0.]*len(ells))


            return cls
