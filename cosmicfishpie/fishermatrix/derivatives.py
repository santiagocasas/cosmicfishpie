# -*- coding: utf-8 -*-

"""
DERIVATIVES

This is the derivatives engine of CosmicFish.
"""

import copy
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from time import time
from typing import Callable, Mapping, Protocol

import numpy as np

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.utilities.utils import printing as upt

_UNSET = object()


def _thaw(value):
    """Recursively copy frozen configuration values into provider-owned values."""
    if isinstance(value, MappingABC):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class DerivativeRequest:
    """Backend-neutral inputs required to compute observable derivatives.

    Concrete providers may use finite differences, analytical formulae, or a
    future autodiff implementation. Providers must return the existing
    ``{parameter: derivative}`` result shape.
    """

    observable: Callable
    fiducial: Mapping[str, object]
    freeparams: Mapping[str, float]
    observables_type: tuple[str, ...]
    external_settings: object
    feed_lvl: int
    method: str
    special_deriv_function: Callable | None
    configuration: object


class DerivativeProvider(Protocol):
    """Protocol implemented by interchangeable derivative backends."""

    def compute(self, request: DerivativeRequest) -> dict[str, object]:
        """Compute all derivatives described by ``request``."""
        ...


class derivatives:
    """This class is the main derivative engine for the different observables. It gives access to different derivative methods. After the constructor of this class is called the resulting dictionary with the derivatives are is found in it's `results` attribute."""

    def __init__(
        self,
        observable,
        fiducial,
        special_deriv_function=None,
        derivatives_type=None,
        freeparams=None,
        observables_type=None,
        external_settings=_UNSET,
        feed_lvl=None,
        *,
        configuration=None,
    ):
        """
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
        observables_type : tuple[str, ...], optional
                           Tuple of observable types (e.g. ("GCph", "WL"))
        external_settings : dict or _UNSET, optional
                            A dictionary containing all paths to the external files, how all the names of the files in the folder correspond to the cosmological quantities, the units etc. Will be none if code runs in internal mode
        feed_lvl : int, optional
                   Number indicating the verbosity of the output. Higher numbers generally mean more output. Defaults to 2
        derivatives_type : str, optional
                           One of "3PT", "STEM", "POLY", "4PT_FWD". If None, taken from configuration.
        configuration : object, optional
                        Configuration object providing defaults for missing arguments. If None, uses the legacy global cfg.
        """
        self.configuration = cfg if configuration is None else configuration

        # Resolve freeparams: if None, take a thawed copy from configuration; if provided, use as-is (but still thaw if it's a mapping? we assume user provides a mutable dict)
        if freeparams is None:
            self.freeparams = _thaw(self.configuration.freeparams)
        else:
            # User-provided freeparams should be used directly; we still copy to avoid mutating user input?
            self.freeparams = copy.deepcopy(freeparams)

        self.observable = observable
        self.fiducial = fiducial
        self.special = special_deriv_function

        self.feed_lvl = self.configuration.settings["feedback"] if feed_lvl is None else feed_lvl

        self.observables_type = (
            tuple(self.configuration.obs) if observables_type is None else tuple(observables_type)
        )

        if external_settings is _UNSET:
            self.external_settings = self.configuration.external
        else:
            self.external_settings = external_settings

        self.method = (
            self.configuration.settings["derivatives"]
            if derivatives_type is None
            else derivatives_type
        )
        self.derivatives_type = self.method

        # Dispatch to the correct derivative method
        if self.method == "3PT":
            self.result = self.derivative_3pt()
        elif self.method == "STEM":
            self.result = self.derivative_stem()
        elif self.method == "POLY":
            self.result = self.derivative_poly()
        elif self.method == "4PT_FWD":
            self.result = self.derivative_forward_4pt()
        else:
            raise ValueError(f"ERROR: I don't know this derivative type!!! ({self.method})")

    # --- 3PT -------------------------------------------------------------
    def der_3pt_stencil(self, fwd, bwd, step):
        """Helper function to compute the 3PT symmetrical finite step size derivative

        Arguments
        ---------
        fwd  : float, numpy.ndarray
               Observable computed at the forward step
        bwd  : float, numpy.ndarray
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
                    upt.time_print(
                        feedback_level=self.feed_lvl,
                        min_level=1,
                        text="ððð Obtaining analytical derivative for parameter: {}".format(
                            str(par)
                        ),
                    )
                    deriv_dict[par] = special_deriv
                    continue
                # if special_deriv is None, fall back to numerical

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

            observables_type = self.observables_type or []

            if "GCph" in observables_type or "WL" in observables_type:
                dpar = {}
                for key in obs_fwd:
                    if key == "ells":
                        dpar[key] = obs_fwd[key]
                    else:
                        dpar[key] = self.der_3pt_stencil(obs_fwd[key], obs_bwd[key], stepsize)
            elif "GCsp" in observables_type or "IM" in observables_type:
                dpar = {}
                for key in obs_fwd:
                    if key == "z_bins":
                        dpar[key] = obs_fwd[key]
                    else:
                        dpar[key] = self.der_3pt_stencil(obs_fwd[key], obs_bwd[key], stepsize)
            elif any(o in observables_type for o in ("CMB_T", "CMB_E", "CMB_B")):
                dpar = {}
                for key in obs_fwd:
                    if key == "ells":
                        dpar[key] = obs_fwd[key]
                    else:
                        dpar[key] = self.der_3pt_stencil(obs_fwd[key], obs_bwd[key], stepsize)
            elif "plain" in observables_type:
                dpar = self.der_3pt_stencil(obs_fwd, obs_bwd, stepsize)
            else:
                raise ValueError(
                    f"Unable to compute derivatives: unsupported observables_type={observables_type!r}."
                )

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

    # --- 4PT forward -----------------------------------------------------
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
        der = (-11 * fwdi[0] + 18 * fwdi[1] - 9 * fwdi[2] + 2 * fwdi[3]) / (6 * step**1)
        return der

    def derivative_forward_4pt(self):
        """One of the possible derivative methods. Computes the numerical derivative using a finite differences one-sided 4 point forward derivative.
        Taken from:
        https://web.media.mit.edu/~crtaylor/calculator.html
        @misc{fdcc,
          title={Finite Difference Coefficients Calculator},
          author={Taylor, Cameron R.},
          year={2016},
          howpublished="\\url{https://web.media.mit.edu/~crtaylor/calculator.html}"
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
                        text='ððð "Obtaining analytical derivative for parameter: {:s}".format(par)',
                    )
                    deriv_dict[par] = special_deriv
                    continue
                # if special_deriv is None, fall back to numerical

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
                min_level=2,
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

            if "GCph" in self.observables_type or "WL" in self.observables_type:
                dpar = {}
                for key in obs_fwd_list[0]:
                    if key == "ells":
                        dpar[key] = obs_fwd_list[0][key]
                    else:
                        obs_fwd_list_at_key = [obs_fwd_list[sti][key] for sti in range(Nsteps_fwd)]
                        dpar[key] = self.der_fwd_4pt(obs_fwd_list_at_key, stepsize)
            if "GCsp" in self.observables_type or "IM" in self.observables_type:
                dpar = {}
                for key in obs_fwd_list[0]:
                    if key == "z_bins":
                        dpar[key] = obs_fwd_list[0][key]
                    else:
                        obs_fwd_list_at_key = [obs_fwd_list[sti][key] for sti in range(Nsteps_fwd)]
                        dpar[key] = self.der_fwd_4pt(obs_fwd_list_at_key, stepsize)
            if "plain" in self.observables_type:
                dpar = self.der_fwd_4pt(obs_fwd_list, stepsize)

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

    # --- STEM ------------------------------------------------------------
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

            if "GCph" in self.observables_type or "WL" in self.observables_type:
                for key in obs_mod[0]:
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
                raise ValueError("STEM derivative not availabe")

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

    # --- Polynomial ------------------------------------------------------
    def derivative_poly(self):
        """One of the possible derivative methods. Computes the numerical derivative using a polynomial derivative method

        Returns
        -------
        dict
            A dictionary containing the derivative of the observable for each varied parameter.
        """
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
                if key == "ells":
                    dpar[key] = obs_mod[0][key]
                else:
                    temp = []
                    for ind in range(len(dpar["ells"])):
                        fit = np.polyfit(
                            stepsize, [obs_mod[step][key][ind] for step in range(len(stepsize))], 4
                        )
                        temp.append(
                            4 * fit[0] * fidpar**3
                            + 3 * fit[2] * fidpar**2
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


class FiniteDifferenceProvider:
    """Default provider adapting the existing finite-difference engine."""

    def compute(self, request: DerivativeRequest) -> dict[str, object]:
        return derivatives(
            observable=request.observable,
            fiducial=request.fiducial,
            special_deriv_function=request.special_deriv_function,
            derivatives_type=request.method,
            freeparams=request.freeparams,
            observables_type=request.observables_type,
            external_settings=request.external_settings,
            feed_lvl=request.feed_lvl,
            configuration=request.configuration,
        ).result


def compute_derivatives(
    observable,
    fiducial,
    *,
    configuration=None,
    provider: DerivativeProvider | None = None,
    special_deriv_function=None,
    derivatives_type=None,
    freeparams=None,
    observables_type=None,
    external_settings=_UNSET,
    feed_lvl=None,
):
    """Compute derivatives through an interchangeable backend-neutral provider.

    The default provider preserves the existing finite-difference formulas. A
    differentiable backend or emulator can later inject an autodiff provider
    without coupling probe covariance code to JAX.
    """
    resolved_configuration = cfg if configuration is None else configuration

    # Normalize external_settings sentinel to None for internal logic (the derivatives class expects _UNSET or a dict)
    # but we will pass through as-is; the derivatives class treats _UNSET as "take from configuration".
    # We'll keep the sentinel.

    request = DerivativeRequest(
        observable=observable,
        fiducial=fiducial,
        freeparams=dict(resolved_configuration.freeparams if freeparams is None else freeparams),
        observables_type=tuple(
            resolved_configuration.obs if observables_type is None else observables_type
        ),
        external_settings=(
            resolved_configuration.external if external_settings is _UNSET else external_settings
        ),
        feed_lvl=(resolved_configuration.settings["feedback"] if feed_lvl is None else feed_lvl),
        method=(
            resolved_configuration.settings["derivatives"]
            if derivatives_type is None
            else derivatives_type
        ),
        special_deriv_function=special_deriv_function,
        configuration=resolved_configuration,
    )
    active_provider = FiniteDifferenceProvider() if provider is None else provider
    return active_provider.compute(request)
