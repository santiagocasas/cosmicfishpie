r"""Derivative providers for the JAX-native observables of the ``jaxcosmo`` backend.

Two providers implement
:class:`cosmicfishpie.fishermatrix.derivatives.DerivativeProvider`:

* :class:`JaxAutodiffProvider` differentiates the JAX observable with ``jax.jacfwd``;
* :class:`JaxFiniteDifferenceProvider` applies the *same* finite-difference engine used by the
  standard CosmicFishPie pipeline to the *same* JAX observable.

Having both is what makes a controlled validation possible: autodiff against finite differences
of one implementation isolates the derivative method, while either of them against the legacy
NumPy observable isolates the observable implementation.

Usage
-----
>>> from cosmicfishpie.fishermatrix.jaxcosmo_derivatives import JaxAutodiffProvider
>>> # FisherMatrix(..., derivative_provider=JaxAutodiffProvider())  # doctest: +SKIP

The providers are stateful on purpose: the JAX observable and its compiled Jacobians are cached
between calls (and are *not* picklable), so a fresh provider must be created in every worker
process.  Only the photometric probes ``GCph`` and ``WL`` are supported.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from cosmicfishpie.fishermatrix.derivatives import DerivativeRequest, FiniteDifferenceProvider


class _JaxObservableProvider:
    """Shared lifecycle for providers backed by :class:`JaxPhotoObservable`."""

    def __init__(self) -> None:
        self._observable = None
        self._fiducial: dict[str, Any] | None = None

    # Compiled functions and device arrays must never be shipped between processes.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_observable"] = None
        state["_fiducial"] = None
        return state

    def _get_observable(self, request: DerivativeRequest):
        fiducial = dict(request.fiducial)
        if self._observable is not None and self._same_fiducial(fiducial):
            return self._observable

        configuration = request.configuration
        if configuration is None:
            from cosmicfishpie.configs import config as configuration  # legacy global state
        code = configuration.settings.get("code") if hasattr(configuration, "settings") else None
        if code not in (None, "jaxcosmo"):
            raise ValueError(
                f"{type(self).__name__} requires options['code'] = 'jaxcosmo' (got '{code}')."
            )
        observables = tuple(request.observables_type)
        if "GCsp" in observables:
            from cosmicfishpie.LSSsurvey.jaxcosmo_spectro import JaxSpectroObservable

            self._observable = JaxSpectroObservable(configuration, fiducial)
        else:
            from cosmicfishpie.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable

            self._observable = JaxPhotoObservable(configuration, fiducial)
        self._fiducial = fiducial
        return self._observable

    def _same_fiducial(self, fiducial: dict[str, Any]) -> bool:
        old = self._fiducial
        if old is None or old.keys() != fiducial.keys():
            return False
        return all(
            np.array_equal(old[k], fiducial[k])
            if not isinstance(old[k], str)
            else old[k] == fiducial[k]
            for k in old
        )

    @staticmethod
    def _special_derivs(request: DerivativeRequest) -> dict[str, object]:
        """Derivatives CosmicFishPie already knows analytically (e.g. shot noise)."""
        special = request.special_deriv_function
        if special is None:
            return {}
        found: dict[str, object] = {}
        primed = False
        for name in request.freeparams:
            try:
                value = special(name)
            except AttributeError:
                # The legacy spectroscopic engine caches its fiducial observable on the
                # first call to ``get_obs``; prime it so its analytic shot-noise
                # derivatives are available without a legacy evaluation loop.
                if primed:  # pragma: no cover - defensive
                    raise
                request.observable(dict(request.fiducial))
                primed = True
                value = special(name)
            if value is not None:
                found[name] = value
        return found


class JaxAutodiffProvider(_JaxObservableProvider):
    """Exact forward-mode derivatives of the photometric ``C_ell`` (``jaxcosmo`` backend)."""

    def compute(self, request: DerivativeRequest) -> dict[str, object]:
        """Return ``{parameter: derivative}`` for ``request.freeparams``.

        Parameters with an analytic derivative in ``request.special_deriv_function`` keep it,
        exactly as the finite-difference engine does.
        """
        observable = self._get_observable(request)
        result = self._special_derivs(request)
        names = [name for name in request.freeparams if name not in result]
        if names:
            result.update(dict(observable.jacobian(tuple(names))))
        return {name: result[name] for name in request.freeparams}


class JaxFiniteDifferenceProvider(_JaxObservableProvider):
    """Finite differences of the JAX observable, using the standard CosmicFishPie engine.

    The step sizes and the stencil (``options["derivatives"]``) are taken from the same
    :class:`~cosmicfishpie.fishermatrix.derivatives.DerivativeRequest` used by the reference
    pipeline, so only the observable implementation differs.
    """

    def compute(self, request: DerivativeRequest) -> dict[str, object]:
        """Return ``{parameter: derivative}`` by finite differences of the JAX observable."""
        if request.special_deriv_function is not None:
            # The legacy spectroscopic engine keeps its fiducial observable in an attribute
            # that ``get_obs`` fills in; prime it so the analytic shot-noise derivatives the
            # finite-difference engine asks for are available.
            request.observable(dict(request.fiducial))
        observable = self._get_observable(request)
        jax_request = replace(request, observable=observable)
        return FiniteDifferenceProvider().compute(jax_request)
