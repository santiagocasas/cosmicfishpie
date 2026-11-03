r"""Derivative providers for the JAX-native observables of the ``jaxcosmo`` backend.

Two providers implement
:class:`cosmicjellyfish.fishermatrix.derivatives.DerivativeProvider`:

* :class:`JaxAutodiffProvider` differentiates the JAX observable with ``jax.jacfwd``;
* :class:`JaxFiniteDifferenceProvider` applies the *same* finite-difference engine used by the
  standard CosmicJellyfish pipeline to the *same* JAX observable.

Having both is what makes a controlled validation possible: autodiff against finite differences
of one implementation isolates the derivative method, while either of them against the legacy
NumPy observable isolates the observable implementation.

Usage
-----
>>> from cosmicjellyfish.fishermatrix.jaxcosmo_derivatives import JaxAutodiffProvider
>>> # FisherMatrix(..., derivative_provider=JaxAutodiffProvider())  # doctest: +SKIP

The providers are stateful on purpose: the JAX observable and its compiled Jacobians are cached
between calls (and are *not* picklable), so a fresh provider must be created in every worker
process.  The probes ``GCph`` + ``WL`` (photometric) and ``GCsp`` (spectroscopic) are supported;
the provider picks the JAX observable from ``request.observables_type``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np

from cosmicjellyfish.fishermatrix.derivatives import (
    DerivativeRequest,
    FiniteDifferenceProvider,
)


def _json_default(obj: Any) -> Any:
    """Serialise the odd objects found in ``settings`` / ``specs`` for fingerprinting."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(map(repr, obj))
    return repr(obj)


def _fingerprint(*objects: Any) -> str:
    """Content hash of nested settings/specs/fiducial dictionaries (order independent)."""
    payload = json.dumps(objects, default=_json_default, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _JaxObservableProvider:
    """Shared lifecycle for providers backed by the JAX photometric / spectroscopic observables.

    The observable (and its compiled Jacobians) is cached between calls.  The cache key is a
    *content* fingerprint of the fiducial point, the analysis settings, the survey
    specifications, the requested probes and the backend parameters, so one provider instance
    can safely be reused across analyses: any change rebuilds the observable.  Object identity
    would not be enough, because the legacy ``config`` module is mutated in place.
    """

    def __init__(self) -> None:
        self._observable = None
        self._cache_key: str | None = None

    # Compiled functions and device arrays must never be shipped between processes.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_observable"] = None
        state["_cache_key"] = None
        return state

    @staticmethod
    def _resolve_configuration(request: DerivativeRequest):
        configuration = request.configuration
        if configuration is None:
            from cosmicjellyfish.configs import (
                config as configuration,  # legacy global state
            )
        return configuration

    def _make_cache_key(self, request: DerivativeRequest, configuration) -> str:
        backend = getattr(configuration, "backend_parameters", None)
        if not backend:
            from cosmicjellyfish.configs import config as legacy

            backend = getattr(legacy, "boltzmann_jaxcosmopars", None)
        return _fingerprint(
            dict(request.fiducial),
            getattr(configuration, "settings", None),
            getattr(configuration, "specs", None),
            tuple(getattr(configuration, "obs", ())),
            tuple(request.observables_type),
            backend,
        )

    def _get_observable(self, request: DerivativeRequest):
        configuration = self._resolve_configuration(request)
        key = self._make_cache_key(request, configuration)
        if self._observable is not None and key == self._cache_key:
            return self._observable

        code = configuration.settings.get("code") if hasattr(configuration, "settings") else None
        if code not in (None, "jaxcosmo"):
            raise ValueError(
                f"{type(self).__name__} requires options['code'] = 'jaxcosmo' (got '{code}')."
            )
        fiducial = dict(request.fiducial)
        observables = tuple(request.observables_type)
        unsupported = sorted(set(observables) - {"GCph", "WL", "GCsp"})
        if unsupported or ("GCsp" in observables and {"GCph", "WL"} & set(observables)):
            raise ValueError(
                f"{type(self).__name__} supports GCph+WL or GCsp per request; "
                f"got {observables!r}."
            )
        observable: Any
        if "GCsp" in observables:
            from cosmicjellyfish.LSSsurvey.jaxcosmo_spectro import JaxSpectroObservable

            observable = JaxSpectroObservable(configuration, fiducial, observables=observables)
        else:
            from cosmicjellyfish.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable

            observable = JaxPhotoObservable(configuration, fiducial, observables=observables)
        self._observable, self._cache_key = observable, key
        return observable

    @staticmethod
    def _special_derivs(request: DerivativeRequest) -> dict[str, object]:
        """Derivatives CosmicJellyfish already knows analytically (e.g. shot noise)."""
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
    """Exact forward-mode derivatives of the JAX observables (``jaxcosmo`` backend)."""

    def compute(self, request: DerivativeRequest) -> dict[str, object]:
        """Return ``{parameter: derivative}`` for ``request.freeparams``.

        Parameters with an analytic derivative in ``request.special_deriv_function`` keep it,
        exactly as the finite-difference engine does.  The ``Ps_`` shot-noise parameters are
        the exception: the legacy closed form differentiates the legacy power spectrum, while
        this provider must differentiate the JAX one, so they always use the JAX jacobian.
        """
        observable = self._get_observable(request)
        result = self._special_derivs(request)
        names = [
            name for name in request.freeparams if name not in result or name.startswith("Ps_")
        ]
        if names:
            result.update(dict(observable.jacobian(tuple(names))))
        return {name: result[name] for name in request.freeparams}


class JaxFiniteDifferenceProvider(_JaxObservableProvider):
    """Finite differences of the JAX observable, using the standard CosmicJellyfish engine.

    The step sizes and the stencil (``options["derivatives"]``) are taken from the same
    :class:`~cosmicjellyfish.fishermatrix.derivatives.DerivativeRequest` used by the reference
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
        if request.special_deriv_function is not None:
            special = request.special_deriv_function

            def jax_special(name: str):
                # ``Ps_*``: the legacy closed form differentiates the legacy spectrum; let the
                # engine finite-difference the JAX observable for these instead.
                return None if name.startswith("Ps_") else special(name)

            jax_request = replace(jax_request, special_deriv_function=jax_special)
        return FiniteDifferenceProvider().compute(jax_request)
