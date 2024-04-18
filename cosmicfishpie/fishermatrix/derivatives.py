# -*- coding: utf-8 -*-
"""DERIVATIVES

This is the derivatives engine of CosmicFish.

"""

import copy
from time import time

import numpy as np

import cosmicfishpie.fishermatrix.config as cfg
from cosmicfishpie.utilities.utils import printing as upt


class derivatives:
    def __init__(
        self,
        observable,
        fiducial,
        special_deriv_function=None,
        freeparams=dict(),
    ):
        """This class is the main derivative engine for the different observables. It gives access to different derivative methods. After the constructor of this class is called the resulting dictionary with the derivatives are is found in it's `results` attribute.

        Arguments
        ---------
        observable : callable
                     A callable function that when passed a dictionary of all cosmological and nuisance parameters will return the observable of a probe
        fiducial : dict
                   A dictionary containing the fiducial values of all parameters
        special_deriv_function : callable, optional
                                 callable function that receives a parameter name and calculates the exact derivative of the observable for that parameter
        freeparams : dict, optional
                     A dictionary for the parameters that should be varied. Will vary all parameters if not passed

        Attributes
        ----------
        observable        : callable
                            A callable function that when passed a dictionary of all cosmological and nuisance parameters will return the observable of a probe
        fiducial          : dict
                            A dictionary containing the fiducial values of all parameters
        special           : callable, None
                            If it exists, it is a callable function that receives a parameter name and calculates the exact derivative of the observable for that parameter
        feed_lvl          : int
                            Number indicating the verbosity of the output. Higher numbers generally mean more output. Defaults to 2
        observables       : callable
                            A callable that is passed all cosmological and nuisance parameters in a dict that is returning the computed observable of a given probe
        external_settings : dict
                            A dictionary containing all paths to the external files, how all the names of the files in the folder correspond to the cosmological quantities, the units etc. Will be none if code runs in internal mode
        result            : dict
                            A dictionary with all derivatives of the observable with respect to the parameter corresponding to the key name
        """
        if freeparams != dict():
            self.freeparams = freeparams
        else:
            self.freeparams = cfg.freeparams
        self.observable = observable
        self.fiducial = fiducial
        self.special = special_deriv_function
        self.feed_lvl = cfg.settings["feedback"]
        self.observables = cfg.obs
        self.external_settings = cfg.external
        if cfg.settings["derivatives"] == "3PT":
            self.result = self.derivative_3pt()
        elif cfg.settings["derivatives"] == "STEM":
            self.result = self.derivative_stem()
        elif cfg.settings["derivatives"] == "POLY":
            self.result = self.derivative_poly()
        elif cfg.settings["derivatives"] == "4PT_FWD":
            self.result = self.derivative_forward_4pt()
        else:
            print("ERROR: I don't know this derivative type!!!")

    def der_3pt_stencil(self, fwd, bwd, step):
        """Helper function to compute the 3PT symmetrical finite step size derivative

        Arguments
        ---------
        fwd  : float, numpy.ndarray
               Observable computed at the forward step
        fwd  : float, numpy.ndarray
               Observable computed at the backwards step
        step : float
               Absolute step size of the numerical derivative

        Returns
        -------
        float, numpy.ndarray
            Numerical derivative using the a 3 point stencil
        """
        der = (fwd - bwd) / (2 * step)
        return der

    def derivative_3pt(self):
        """One of the possible derivative methods. Computes the numerical derivative using a finite differences 3 point symmetrical derivative

        Returns
        -------
        dict
            A dictionary containing the derivative of the observable for each varied parameter.

        Note
        -----
        Implements the following equation:

        .. math::

            \\frac{\\mathrm{d} \\mathcal{O}}{\\mathrm{d} \\theta} = \\frac{\\mathcal{O}(\\theta+h)-\\mathcal{O}(\\theta-h)}{2\\,h}
        """
        deriv_dict = {}

        for par in self.freeparams:
            if self.special is not None:
                special_deriv = self.special(par)
                if special_deriv is not None:
                    print("ððð Obtaining analytical derivative for parameter: ", str(par))
                    deriv_dict[par] = special_deriv
                    continue
                elif special_deriv is None:
                    pass
            if self.fiducial[par] != 0.0:
                stepsize = self.fiducial[par] * self.freeparams[par]
            else:
                stepsize = self.freeparams[par]

            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="+++ Computing derivative on {}".format(par),
            )
            tini = time()

            # doing forward step
            fwd = copy.deepcopy(self.fiducial)

            fwd[par] = fwd[par] + stepsize

            obs_fwd = self.observable(fwd)

            # Doing backward step
            bwd = copy.deepcopy(self.fiducial)
            bwd[par] = bwd[par] - stepsize

            obs_bwd = self.observable(bwd)

            if "GCph" in self.observables or "WL" in self.observables:
                dpar = {}
                for key in obs_fwd:
                    if key == "ells":
                        dpar[key] = obs_fwd[key]
                    else:
                        dpar[key] = self.der_3pt_stencil(obs_fwd[key], obs_bwd[key], stepsize)
            if "GCsp" in self.observables or "IM" in self.observables:
                dpar = {}
                for key in obs_fwd:
                    if key == "z_bins":
                        dpar[key] = obs_fwd[key]
                    else:
                        dpar[key] = self.der_3pt_stencil(obs_fwd[key], obs_bwd[key], stepsize)

            tend = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="Derivative on {} done! in :".format(par),
                time_ini=tini,
                time_fin=tend,
                instance=self,
            )

            deriv_dict[par] = dpar

        return deriv_dict

    def der_fwd_4pt(self, fwdi, step):
        """Helper function to compute the 4PT forward finite step size derivative

        Arguments
        ---------
        fwdi : list, numpy.ndarray
               Observable computed at the fiducial and at equally spaced points in the forward direction
        step : float
               Absolute distance between the size of the numerical derivative

        Returns
        -------
        float, numpy.ndarray
            Numerical derivative using the a 4 point forward stencil
        """
        der = (-11 * fwdi[0] + 18 * fwdi[1] - 9 * fwdi[2] + 2 * fwdi[3]) / (6 * step ** 1)
        return der

    def derivative_forward_4pt(self):
        r"""One of the possible derivative methods. Computes the numerical derivative using a finite differences one-sided 4 point forward derivative.
        Taken from:
        https://web.media.mit.edu/~crtaylor/calculator.html
        @misc{fdcc,
        title={Finite Difference Coefficients Calculator},
        author={Taylor, Cameron R.},
        year={2016},
        howpublished="\url{https://web.media.mit.edu/~crtaylor/calculator.html}"
        }

        Returns
        -------
        dict
            A dictionary containing the derivative of the observable for each varied parameter.

        Note
        -----
        Implements the following equation:

        .. math::

            \\frac{\\mathrm{d} \\mathcal{O}}{\\mathrm{d} \\theta} = \\frac{-11\\,\\mathcal{O}(\\theta)+18\\,\\mathcal{O}(\\theta+h)-9\\,\\mathcal{O}(\\theta+2\\,)+2\\,\\mathcal{O}(\\theta+3\\,h)}{6\\,h}

        """
        deriv_dict = {}

        for par in self.freeparams:
            if self.special is not None:
                special_deriv = self.special(par)
                if special_deriv is not None:
                    upt.time_print(
                        feedback_level=self.feed_lvl,
                        min_level=2,
                        text='ððð "Obtaining analytical derivative for parameter: {:s}'.format(par),
                    )
                    deriv_dict[par] = special_deriv
                    continue
                elif special_deriv is None:
                    pass
            if self.fiducial[par] != 0.0:
                stepsize = self.fiducial[par] * self.freeparams[par]
            else:
                stepsize = self.freeparams[par]

            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="+++ Computing 4pt forward derivative on {}".format(par),
            )
            tini = time()

            # doing forward step
            fwd_0 = copy.deepcopy(self.fiducial)
            fwd_1 = copy.deepcopy(self.fiducial)
            fwd_2 = copy.deepcopy(self.fiducial)
            fwd_3 = copy.deepcopy(self.fiducial)

            fwd_0[par] = fwd_0[par]
            fwd_1[par] = fwd_1[par] + 1 * stepsize
            fwd_2[par] = fwd_2[par] + 2 * stepsize
            fwd_3[par] = fwd_3[par] + 3 * stepsize
            fwdlist = [fwd_0, fwd_1, fwd_2, fwd_3]
            Nsteps_fwd = len(fwdlist)

            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="++++ Computing observables at 4 steps",
            )
            obs_fwd_list = []
            for ffstep in fwdlist:
                upt.time_print(
                    feedback_level=self.feed_lvl,
                    min_level=2,
                    text="^^^ Computing observable at parameter {:s} with value: {:.6f} and stepsize: {:.4f}".format(
                        par, ffstep[par], (ffstep[par] - fwd_0[par])
                    ),
                )
                obs_at_step = self.observable(ffstep)
                obs_fwd_list.append(obs_at_step)
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=2,
                text="++^^++ Size of obs_fwd_list : {:d}".format(len(obs_fwd_list)),
            )

            if "GCph" in self.observables or "WL" in self.observables:
                dpar = {}
                for key in obs_fwd_list[0]:
                    if key == "ells":
                        dpar[key] = obs_fwd_list[0][key]
                    else:
                        obs_fwd_list_at_key = [obs_fwd_list[sti][key] for sti in range(Nsteps_fwd)]
                        dpar[key] = self.der_fwd_4pt(obs_fwd_list_at_key, stepsize)
            if "GCsp" in self.observables or "IM" in self.observables:
                dpar = {}
                for key in obs_fwd_list[0]:
                    if key == "z_bins":
                        dpar[key] = obs_fwd_list[0][key]
                    else:
                        obs_fwd_list_at_key = [obs_fwd_list[sti][key] for sti in range(Nsteps_fwd)]
                        dpar[key] = self.der_fwd_4pt(obs_fwd_list_at_key, stepsize)

            tend = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="Derivative on {} done! in :".format(par),
                time_ini=tini,
                time_fin=tend,
                instance=self,
            )

            deriv_dict[par] = dpar

        return deriv_dict

    def derivative_stem(self):
        """One of the possible derivative methods. Computes the numerical derivative using the SteM derivative method

        Returns
        -------
        dict
            A dictionary containing the derivative of the observable for each varied parameter.
        """
        numstem = 11
        mult_eps_factor = 5

        def adaptive_eps(param_eps):
            if self.external_settings is not None:
                eps_v = np.array(self.external_settings["eps_values"])
                d_eps = np.concatenate([-eps_v[::-1], eps_v])
            else:
                d_eps = np.linspace(
                    -param_eps * mult_eps_factor, param_eps * mult_eps_factor, numstem
                )
                # eps_arr = np.linspace(-self.freeparams[par], self.freeparams[par], numstem)
            return d_eps

        threshold = 1.0e-3

        deriv_dict = {}

        for par in self.freeparams:
            if self.fiducial[par] != 0.0:
                stepsize = self.fiducial[par] * adaptive_eps(self.freeparams[par])
            else:
                stepsize = adaptive_eps(self.freeparams[par])

            dpar = {}

            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="+++ Computing STEM derivative on {}".format(par),
            )
            tini = time()

            obs_mod = []

            for step in stepsize:
                modpars = copy.deepcopy(self.fiducial)

                modpars[par] = modpars[par] + step

                obs_mod.append(self.observable(modpars))

            if "GCph" in self.observables or "WL" in self.observables:
                for key in obs_mod[0]:
                    # WARNING: THIS WORKS FOR NOW, BUT OTHER THINGS HAVE TO BE
                    # ADDED TO THE IF FOR OTHER OBS
                    if key == "ells":
                        dpar[key] = obs_mod[0][key]
                    else:
                        temp = []
                        for ind in range(len(dpar["ells"])):
                            residuals = 1000
                            counter = 0
                            tempstep = stepsize
                            while residuals > threshold:
                                fit = np.polyfit(
                                    tempstep,
                                    [obs_mod[step][key][ind] for step in range(len(tempstep))],
                                    1,
                                    full=True,
                                )
                                residuals = fit[1]
                                if residuals > threshold:
                                    tempstep = tempstep[1:-1]
                                counter += 1
                                if numstem - counter < 3:
                                    print("ERROR: {} derivative could not converge!".format(par))
                                    exit()
                            temp.append(fit[0][0])
                        dpar[key] = np.array(temp)
            else:
                raise ValueError("STEM derivative not availabe for spectropscopic probes yet!")

            tend = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="Derivative on {} done! in :".format(par),
                time_ini=tini,
                time_fin=tend,
                instance=self,
            )

            deriv_dict[par] = dpar

        return deriv_dict

    def derivative_poly(self):
        """One of the possible derivative methods. Computes the numerical derivative using a polynomial derivative method

        Returns
        -------
        dict
            A dictionary containing the derivative of the observable for each varied parameter."""
        numpoints = 10  # HARD CODED?

        deriv_dict = {}

        for par in self.freeparams:
            if self.fiducial[par] != 0.0:
                stepsize = np.linspace(
                    -self.fiducial[par] * self.freeparams[par],
                    self.fiducial[par] * self.freeparams[par],
                    numpoints,
                )
            else:
                stepsize = np.linspace(-self.freeparams[par], self.freeparams[par], numpoints)

            dpar = {}

            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="+++ Computing poly derivative on {}".format(par),
            )
            tini = time()

            fidpar = self.fiducial[par]

            obs_mod = []

            for step in stepsize:
                modpars = copy.deepcopy(self.fiducial)

                modpars[par] = modpars[par] + step

                obs_mod.append(self.observable(modpars))

            for key in obs_mod[0]:
                # WARNING: THIS WORKS FOR NOW, BUT OTHER THINGS HAVE TO BE
                # ADDED TO THE IF FOR OTHER OBS
                if key == "ells":
                    dpar[key] = obs_mod[0][key]
                else:
                    temp = []
                    for ind in range(len(dpar["ells"])):
                        fit = np.polyfit(
                            stepsize, [obs_mod[step][key][ind] for step in range(len(stepsize))], 4
                        )
                        temp.append(
                            4 * fit[0] * fidpar ** 3 * +3 * fit[2] * fidpar ** 2
                            + 2 * fit[3] * fidpar
                            + fit[4]
                        )

                    dpar[key] = np.array(temp)

            tend = time()
            upt.time_print(
                feedback_level=self.feed_lvl,
                min_level=1,
                text="Derivative on {} done! in :".format(par),
                time_ini=tini,
                time_fin=tend,
                instance=self,
            )

            deriv_dict[par] = dpar

        return deriv_dict
