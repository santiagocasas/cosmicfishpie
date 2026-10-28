r"""Automatic-differentiation derivative provider for the ``jaxcosmo`` backend.

:class:`JaxAutodiffProvider` implements the
:class:`cosmicfishpie.fishermatrix.derivatives.DerivativeProvider` protocol.  Instead of
finite differences of the NumPy observable it differentiates the JAX-native photometric
observable :class:`cosmicfishpie.LSSsurvey.jaxcosmo_photo.JaxPhotoObservable` with
``jax.jacfwd``.

Usage
-----
>>> from cosmicfishpie.fishermatrix.jaxcosmo_autodiff import JaxAutodiffProvider
>>> # FisherMatrix(..., derivative_provider=JaxAutodiffProvider())  # doctest: +SKIP

The provider is stateful on purpose: the JAX observable and its compiled Jacobians are cached
between calls (and are *not* picklable), so a fresh provider must be created in every worker
process.  Only the photometric probes ``GCph`` and ``WL`` are supported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cosmicfishpie.fishermatrix.derivatives import DerivativeRequest


class JaxAutodiffProvider:
    """Exact forward-mode derivatives of the photometric ``C_ell`` (``jaxcosmo`` backend)."""

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
        from cosmicfishpie.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable

        configuration = request.configuration
        if configuration is None:
            from cosmicfishpie.configs import config as configuration  # legacy global state
        code = configuration.settings.get("code") if hasattr(configuration, "settings") else None
        if code not in (None, "jaxcosmo"):
            raise ValueError(
                f"JaxAutodiffProvider requires options['code'] = 'jaxcosmo' (got '{code}')."
            )
        self._observable = JaxPhotoObservable(configuration, fiducial)
        self._fiducial = fiducial
        return self._observable

    def _same_fiducial(self, fiducial: dict[str, Any]) -> bool:
        old = self._fiducial
        if old is None or old.keys() != fiducial.keys():
            return False
        return all(
            np.array_equal(old[k], fiducial[k]) if not isinstance(old[k], str) else old[k] == fiducial[k]
            for k in old
        )

    def compute(self, request: DerivativeRequest) -> dict[str, object]:
        """Return ``{parameter: {Cl key: dC/dtheta, 'ells': ells}}`` for ``request.freeparams``."""
        observable = self._get_observable(request)
        names = tuple(request.freeparams)
        if request.special_deriv_function is not None:
            raise NotImplementedError(
                "special_deriv_function is not supported by the autodiff provider."
            )
        return dict(observable.jacobian(names))
