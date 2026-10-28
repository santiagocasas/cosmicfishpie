r"""Backend-neutral, differentiable theory interface for JAX observables.

CosmicFishPie separates the *theory* backend (which produces distances, growth and matter
power spectra) from the *observables* (which turn those quantities into angular power spectra
or cluster counts).  The same separation is useful for the JAX/differentiable path: a JAX
observable should not know which theory backend produced its inputs.

This module defines that small contract:

* :class:`TheoryTables` is the bundle of quantities an observable needs, always in
  CosmicFishPie units (``chi`` in Mpc, ``h_over_c`` in 1/Mpc, ``P`` in Mpc^3);
* :class:`JaxTheory` is the protocol a differentiable backend implements;
* :func:`get_jax_theory` resolves the backend selected in ``options["code"]``.

Only backends whose computations are written with ``jax.numpy`` can provide exact automatic
derivatives.  A NumPy/SciPy backend can still drive the plain CosmicFishPie observables through
finite differences, but it cannot be used here.

Nothing in this module imports JAX at module level, so it stays importable without the optional
``jaxcosmo`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass
class TheoryTables:
    """Differentiable theory quantities on a redshift/wavenumber grid.

    Attributes
    ----------
    z : array, shape (nz,)
        Redshifts the quantities are evaluated at.
    k_hmpc : array, shape (nk,)
        Wavenumbers in ``h/Mpc``; the power spectra' second axis follows this order.
    chi : array, shape (nz,)
        Comoving distance in Mpc.
    h_over_c : array, shape (nz,)
        Hubble rate divided by the speed of light, in ``1/Mpc``.
    growth : array, shape (nz,)
        Linear growth factor normalised to ``D(0) = 1``.
    omega_m : array, shape (nz,)
        Matter fraction ``Omega_m(z)``.
    pk_linear, pk_nonlinear : array, shape (nz, nk)
        Matter power spectra in ``Mpc^3``.
    h0_over_c : scalar
        Present-day ``H0 / c`` in ``1/Mpc``.
    omega_m0 : scalar
        Present-day matter fraction.
    """

    z: Any
    k_hmpc: Any
    chi: Any
    h_over_c: Any
    growth: Any
    omega_m: Any
    pk_linear: Any
    pk_nonlinear: Any
    h0_over_c: Any
    omega_m0: Any


@runtime_checkable
class JaxTheory(Protocol):
    """A theory backend that can be traced and differentiated by JAX.

    Implementations must be *pure functions* of the parameters: no NumPy conversion and no
    SciPy splines on the path between parameters and returned arrays.
    """

    #: Canonical cosmological parameter names expected in the ``params`` mapping.
    cosmo_names: tuple[str, ...]
    #: Mapping from CosmicFishPie parameter names to this backend's canonical names.
    free_name_map: Mapping[str, str]

    def resolve_cosmology(self, params: Mapping[str, Any]) -> dict[str, float]:
        """Validate a CosmicFishPie cosmology mapping and return canonical float values."""

    def tables(
        self,
        params: Mapping[str, Any],
        z: Any,
        k_hmpc: Any,
        nonlinear: bool = True,
    ) -> TheoryTables:
        """Evaluate :class:`TheoryTables` at ``z`` and ``k_hmpc`` for ``params``."""


def get_jax_theory(
    code: str | None,
    backend_parameters: Mapping[str, Any] | None = None,
    cosmo_model: str = "LCDM",
) -> JaxTheory:
    """Return the differentiable theory implementation selected by ``options["code"]``.

    Parameters
    ----------
    code : str
        Value of ``options["code"]``.  Currently only ``"jaxcosmo"`` is available.
    backend_parameters : mapping, optional
        Parsed ``default_boltzmann_yaml_files/<code>/default.yaml``.
    cosmo_model : str
        ``"LCDM"`` or ``"w0waCDM"``; fixed-w models are rejected by the backend.

    Raises
    ------
    NotImplementedError
        If the selected backend has no differentiable implementation.
    """
    if code == "jaxcosmo":
        from cosmicfishpie.cosmology.jaxcosmo_theory import JaxCosmoTheory

        return JaxCosmoTheory(backend_parameters, cosmo_model=cosmo_model)

    raise NotImplementedError(
        f"The backend '{code}' does not provide a differentiable JAX theory implementation. "
        "Set options['code'] = 'jaxcosmo' to use the autodiff path."
    )
