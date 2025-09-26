"""Base infrastructure for likelihood modules in Cosmicfishpie."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Iterable, Optional

import numpy as np

from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix


def is_indexable_iterable(var: Any) -> bool:
    """Return True for non-string iterables that support numeric indexing."""

    return isinstance(var, (list, np.ndarray, Sequence)) and not isinstance(var, (str, bytes))


class Likelihood(ABC):
    """Common interface for likelihood evaluations used in Cosmicfishpie."""

    def __init__(
        self,
        *,
        cosmoFM_data: FisherMatrix,
        cosmoFM_theory: Optional[FisherMatrix] = None,
        leg_flag: str = "wedges",
    ) -> None:
        """Store Fisher matrices and pre-compute the observed data representation."""

        self.cosmoFM_data = cosmoFM_data
        self.cosmoFM_theory = cosmoFM_theory or cosmoFM_data
        self.leg_flag = leg_flag
        self.data_obs = self.compute_data()

    @abstractmethod
    def compute_data(self) -> Any:
        """Return the data representation (e.g., wedges or multipoles)."""

    @abstractmethod
    def compute_theory(self, param_dict: Dict[str, Any]) -> Any:
        """Return the theory prediction in the same representation as the data."""

    @abstractmethod
    def compute_chi2(self, theory_obs: Any) -> float:
        """Return χ² between ``self.data_obs`` and ``theory_obs``."""

    def build_param_dict(
        self,
        *,
        param_vec: Optional[Iterable[float]] = None,
        param_dict: Optional[Dict[str, Any]] = None,
        prior: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Create a parameter dictionary from vector inputs when needed."""

        if param_dict is not None:
            return dict(param_dict)

        if param_vec is None or prior is None:
            raise ValueError("Provide either param_dict or (param_vec and prior) to build the parameter mapping")

        if not is_indexable_iterable(param_vec):
            raise TypeError("param_vec must be an indexable iterable when no param_dict is supplied")

        prior_keys = getattr(prior, "keys", None)
        if prior_keys is None:
            raise AttributeError("prior object must expose an ordered 'keys' attribute to map the vector to parameters")

        return {key: param_vec[i] for i, key in enumerate(prior_keys)}

    def loglike(
        self,
        *,
        param_vec: Optional[Iterable[float]] = None,
        param_dict: Optional[Dict[str, Any]] = None,
        prior: Optional[Any] = None,
    ) -> float:
        """Compute ``-0.5 * χ²`` for the supplied parameters."""

        params = self.build_param_dict(param_vec=param_vec, param_dict=param_dict, prior=prior)
        theory_obs = self.compute_theory(params)
        chi2 = self.compute_chi2(theory_obs)
        return -0.5 * chi2

    def run_nautilus(
        self,
        *,
        prior: Any,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Sampler":
        """Convenience wrapper to launch a Nautilus sampler using this likelihood."""

        from nautilus import Sampler

        sampler_kwargs = dict(sampler_kwargs or {})
        run_kwargs = dict(run_kwargs or {})

        def wrapper(theta: Iterable[float]) -> float:
            return self.loglike(param_vec=theta, prior=prior)

        sampler = Sampler(prior, wrapper, **sampler_kwargs)
        sampler.run(**run_kwargs)
        return sampler
