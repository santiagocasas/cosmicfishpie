"""Base infrastructure for likelihood modules in Cosmicfishpie.

This module provides the core interface and utilities for implementing
likelihood calculations in Cosmicfishpie, particularly for cosmological
parameter estimation using Fisher matrices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Iterable, Optional

import numpy as np
from nautilus import Prior, Sampler

from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix


def is_indexable_iterable(var: Any) -> bool:
    """Check if a variable is an indexable iterable.

    Args:
        var: The variable to check

    Returns:
        bool: True if the variable is an indexable iterable (list, numpy array, or Sequence)
              and not a string or bytes object, False otherwise.
    """
    return isinstance(var, (list, np.ndarray, Sequence)) and not isinstance(var, (str, bytes))


class Likelihood(ABC):
    """Common interface for likelihood evaluations used in Cosmicfishpie.

    This abstract base class defines the core interface for likelihood calculations
    in Cosmicfishpie. It provides methods for computing data representations,
    theory predictions, and chi-squared values, as well as utilities for parameter
    handling and running Nautilus samplers.
    """

    def __init__(
        self,
        *,
        cosmo_data: FisherMatrix,
        cosmo_theory: Optional[FisherMatrix] = None,
        leg_flag: str = "wedges",
    ) -> None:
        """Initialize the Likelihood object with Fisher matrices.

        Args:
            cosmo_data: FisherMatrix object containing the observed data
            cosmo_theory: Optional FisherMatrix object for theory predictions.
                           If None, uses cosmoFM_data
            leg_flag: Flag indicating the type of data representation to use
                     ("wedges" or other supported types)
        """
        self.cosmo_data = cosmo_data
        self.cosmo_theory = cosmo_theory or cosmo_data
        self.leg_flag = leg_flag
        self.data_obs = self.compute_data()

    @abstractmethod
    def compute_data(self) -> Any:
        """Compute and return the observed data representation.

        This method should be implemented by subclasses to return the data
        in the appropriate format (e.g., wedges or multipoles) for likelihood
        calculations.

        Returns:
            The observed data representation
        """

    @abstractmethod
    def compute_theory(self, param_dict: Dict[str, Any]) -> Any:
        """Compute and return the theory prediction.

        This method should be implemented by subclasses to return the theory
        prediction in the same representation as the observed data.

        Args:
            param_dict: Dictionary of cosmological parameters

        Returns:
            The theory prediction in the same format as the observed data
        """

    @abstractmethod
    def compute_chi2(self, theory_obs: Any) -> float:
        """Compute and return the chi-squared value.

        This method should be implemented by subclasses to compute the chi-squared
        value between the observed data and theory prediction.

        Args:
            theory_obs: Theory prediction in the same format as observed data

        Returns:
            The chi-squared value
        """

    def build_param_dict(
        self,
        *,
        param_vec: Optional[Iterable[float]] = None,
        param_dict: Optional[Dict[str, Any]] = None,
        prior: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Create a parameter dictionary from vector inputs when needed.

        This method converts a parameter vector to a dictionary using the prior
        information, or returns the provided parameter dictionary directly.

        Args:
            param_vec: Optional iterable of parameter values
            param_dict: Optional dictionary of parameters
            prior: Optional prior information for parameter mapping

        Returns:
            Dictionary of parameters

        Raises:
            ValueError: If neither param_dict nor (param_vec and prior) are provided
            TypeError: If param_vec is not an indexable iterable
            AttributeError: If prior object doesn't expose an ordered 'keys' attribute
        """
        if param_dict is not None:
            return dict(param_dict)

        if param_vec is None or prior is None:
            raise ValueError(
                "Provide either param_dict or (param_vec and prior) to build the parameter mapping"
            )

        if not is_indexable_iterable(param_vec):
            raise TypeError(
                "param_vec must be an indexable iterable when no param_dict is supplied"
            )

        prior_keys = getattr(prior, "keys", None)
        if prior_keys is None:
            raise AttributeError(
                "prior object must expose an ordered 'keys' attribute to map the vector to parameters"
            )

        return {key: param_vec[i] for i, key in enumerate(prior_keys)}

    def loglike(
        self,
        param_vec: Optional[Iterable[float]] = None,
        *,
        param_dict: Optional[Dict[str, Any]] = None,
        prior: Optional[Any] = None,
    ) -> float:
        """Compute the log-likelihood value.

        This method computes the log-likelihood value (-0.5 * χ²) for the
        supplied parameters. It can accept either a parameter vector or a
        parameter dictionary.

        Args:
            param_vec: Optional iterable of parameter values
            param_dict: Optional dictionary of parameters
            prior: Optional prior information for parameter mapping

        Returns:
            The log-likelihood value (-0.5 * χ²)
        """
        params = self.build_param_dict(param_vec=param_vec, param_dict=param_dict, prior=prior)
        theory_obs = self.compute_theory(params)
        chi2 = self.compute_chi2(theory_obs)
        return -0.5 * chi2


class NautilusMixin:
    """Mixin class for running Nautilus samplers."""

    def create_nautilus_prior(self, prior_dict: Dict[str, Any]) -> Prior:
        """Create a Nautilus prior object from a dictionary of parameter names and their prior ranges.

        Args:
            prior_dict: Dictionary of parameter names and their prior ranges

        Returns:
            Nautilus Prior object
        """

        prior = Prior()
        for par, (lower, upper) in prior_dict.items():
            prior.add_parameter(par, (lower, upper))
        return prior

    def run_nautilus(
        self,
        *,
        prior: Any,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Sampler:
        """Convenience wrapper to launch a Nautilus sampler using this likelihood.

        This method provides a convenient interface to run Nautilus samplers
        with the current likelihood function.

        Args:
            prior: Prior information for the parameters
            sampler_kwargs: Optional dictionary of keyword arguments for the Sampler
            run_kwargs: Optional dictionary of keyword arguments for the run method

        Returns:
            The Nautilus Sampler object
        """

        sampler_kwargs = dict(sampler_kwargs or {})
        run_kwargs = dict(run_kwargs or {})

        sampler = Sampler(prior, self.loglike, **sampler_kwargs, likelihood_kwargs={"prior": prior})
        sampler.run(**run_kwargs)
        return sampler
