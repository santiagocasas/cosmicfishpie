r"""Pure-JAX theory layer of the ``jaxcosmo`` backend.

Everything in this module is a *pure function* of the cosmological parameters, written with
``jax.numpy`` only, so that it can be traced by ``jax.jit`` / ``jax.jacfwd``.  No NumPy/SciPy
objects (splines, ``np.vectorize`` ...) are used here; the conversion to the spline-like objects
expected by :class:`cosmicjellyfish.cosmology.cosmology.cosmo_functions` happens at the outer
boundary, in :mod:`cosmicjellyfish.cosmology.jaxcosmo_provider`.

Model
-----
* Flat universe, massless neutrinos only (``mnu = 0``), no radiation in the background
  (this is the ``jax_cosmo`` convention: ``E^2 = Om a^-3 + Ode f(a)``).
* Linear power: Eisenstein & Hu transfer function, normalised to ``sigma8``.
* ``sigma8`` is *derived* from ``As`` with the JAX port of the ``symbolic_pofk`` fit
  (:mod:`cosmicjellyfish.cosmology.jaxcosmo_amplitude`), inside the traced function, so that the
  derivative with respect to ``As`` is exact with respect to the fit.
* Non-linear power: Halofit (``takahashi2012`` or ``smith2003``) as implemented in ``jax_cosmo``.

Units
-----
Public functions of this module return the *CosmicJellyfish* conventions:
distances in ``Mpc``, ``H/c`` in ``1/Mpc``, wavenumbers in ``h/Mpc`` and power spectra in
``Mpc^3``.  Internally ``jax_cosmo`` works in ``Mpc/h``, ``h/Mpc`` and ``(Mpc/h)^3``; the
conversions are done in :func:`compute_tables` only.

Notes
-----
``jax_cosmo`` caches tabulated background quantities in ``cosmo._workspace`` (mutable state).
A fresh :class:`jax_cosmo.Cosmology` is therefore created *inside* every traced function and the
caches are primed before any ``vmap``, so no tracer can leak out of a transformation.

``jax`` must run in double precision for Fisher work, call :func:`enable_x64` before creating
any array.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping

try:  # pragma: no cover - exercised only when the optional extra is missing
    import jax
    import jax.numpy as jnp
    import jax_cosmo as jc
    from jax_cosmo import background as jc_background
    from jax_cosmo import power as jc_power
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "The 'jaxcosmo' backend requires the optional dependencies jax and jax-cosmo "
        "(and setuptools<81, which provides pkg_resources for jax-cosmo 0.1.0). "
        "Install them with: pip install 'cosmicjellyfish[jaxcosmo]'"
    ) from err

from cosmicjellyfish.cosmology.jax_theory import TheoryTables
from cosmicjellyfish.cosmology.jaxcosmo_amplitude import As_to_sigma8
from cosmicjellyfish.cosmology.jaxcosmo_halofit import halofit as _smooth_halofit

SPEED_OF_LIGHT_KM_S = 299792.458

#: Order of the cosmological parameters used by the vector interface of the autodiff layer.
#: ``As`` is ``10^9 As``.
PARAM_ORDER: tuple[str, ...] = ("As", "Omegam", "Omegab", "h", "ns", "w0", "wa")

#: Mapping from CosmicJellyfish parameter names to the canonical names of this backend.
FREE_NAME_MAP: dict[str, str] = {
    "10^9As": "As",
    "Omegam": "Omegam",
    "Omegab": "Omegab",
    "h": "h",
    "ns": "ns",
    "w0": "w0",
    "wa": "wa",
}

_SUPPORTED_TRANSFER = ("eisenhu_osc", "eisenhu")
_SUPPORTED_HALOFIT = ("takahashi2012", "smith2003")


def enable_x64() -> None:
    """Switch JAX to double precision (required for Fisher matrices)."""
    jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class TheoryConfig:
    """Static (non-differentiable) numerical choices of the JAX theory.

    All entries are hashable so that the object can be a ``static_argnums`` of ``jax.jit``.
    """

    transfer: str = "eisenhu_osc"
    halofit: str = "takahashi2012"
    distance_steps: int = 256

    def __post_init__(self) -> None:
        if self.transfer not in _SUPPORTED_TRANSFER:
            raise ValueError(
                f"Unsupported transfer function '{self.transfer}'. "
                f"Choose one of {_SUPPORTED_TRANSFER}."
            )
        if self.halofit not in _SUPPORTED_HALOFIT:
            raise ValueError(
                f"Unsupported halofit prescription '{self.halofit}'. "
                f"Choose one of {_SUPPORTED_HALOFIT}."
            )
        steps = int(self.distance_steps)
        if steps != self.distance_steps or steps < 16:
            raise ValueError("distance_steps must be an integer >= 16.")
        object.__setattr__(self, "distance_steps", steps)

    @classmethod
    def from_backend_parameters(cls, backend_parameters: Mapping | None) -> "TheoryConfig":
        """Build the config from the ``ACCURACY`` block of the backend yaml."""
        accuracy = dict((backend_parameters or {}).get("ACCURACY", {}) or {})
        defaults = cls()
        raw_steps = accuracy.get("distance_steps", defaults.distance_steps)
        steps = int(raw_steps)
        if steps != raw_steps:
            raise ValueError("distance_steps must be an integer >= 16.")
        return cls(
            transfer=str(accuracy.get("transfer", defaults.transfer)),
            halofit=str(accuracy.get("halofit", defaults.halofit)),
            distance_steps=steps,
        )


def make_cosmology(params: Mapping) -> "jc.Cosmology":
    """Create a fresh ``jax_cosmo.Cosmology`` from CosmicJellyfish-style parameters.

    Parameters
    ----------
    params : mapping
        Keys ``As`` (this is ``10^9 As``), ``Omegam``, ``Omegab``, ``h``, ``ns``, ``w0``, ``wa``.
        Values may be JAX tracers.

    Returns
    -------
    jax_cosmo.Cosmology
        Flat cosmology with ``Omega_c = Omegam - Omegab`` and ``sigma8`` obtained from ``As``.
    """
    om = params["Omegam"]
    ob = params["Omegab"]
    h = params["h"]
    ns = params["ns"]
    w0 = params["w0"]
    wa = params["wa"]
    sigma8 = As_to_sigma8(params["As"], om, ob, h, ns, 0.0, w0, wa)
    return jc.Cosmology(
        Omega_c=om - ob,
        Omega_b=ob,
        h=h,
        n_s=ns,
        sigma8=sigma8,
        Omega_k=0.0,
        w0=w0,
        wa=wa,
    )


def _transfer_fn(config: TheoryConfig):
    return partial(jc.transfer.Eisenstein_Hu, type=config.transfer)


def compute_tables(
    params: Mapping,
    z: jnp.ndarray,
    k_hmpc: jnp.ndarray,
    config: TheoryConfig,
    nonlinear: bool = True,
) -> dict[str, jnp.ndarray]:
    """Evaluate all theory quantities on a ``(z, k)`` grid, in CosmicJellyfish units.

    Parameters
    ----------
    params : mapping
        See :func:`make_cosmology`.
    z : array, shape (nz,)
        Redshifts.
    k_hmpc : array, shape (nk,)
        Wavenumbers in ``h/Mpc``.
    config : TheoryConfig
        Static numerical choices.
    nonlinear : bool
        If ``False`` the (expensive) Halofit table is skipped and ``Pnl`` is the linear one.

    Returns
    -------
    dict
        ``H_over_c`` [1/Mpc], ``chi`` [Mpc], ``d_A`` [Mpc], ``Omega_m_z``, ``D`` (D(0)=1),
        ``f`` (dlnD/dlna), ``sigma8_z``, ``Plin`` and ``Pnl`` with shape ``(nz, nk)`` in
        ``Mpc^3``.  The wavenumber axis corresponds to ``k = k_hmpc * h`` [1/Mpc].
    """
    z = jnp.atleast_1d(jnp.asarray(z))
    k_hmpc = jnp.atleast_1d(jnp.asarray(k_hmpc))
    cosmo = make_cosmology(params)
    h = cosmo.h
    a = 1.0 / (1.0 + z)
    transfer = _transfer_fn(config)

    # Prime the cached tables *outside* any vmap so that no tracer is stored in the workspace.
    a_prime = jnp.atleast_1d(1.0)
    jc_background.radial_comoving_distance(cosmo, a_prime, steps=config.distance_steps)
    jc_background.growth_factor(cosmo, a_prime)

    # Background (jax_cosmo: Mpc/h and km/s/(Mpc/h)) -> Mpc and 1/Mpc.
    chi = jc_background.radial_comoving_distance(cosmo, a, steps=config.distance_steps) / h
    d_a = jc_background.angular_diameter_distance(cosmo, a) / h
    e_z = jnp.sqrt(jc_background.Esqr(cosmo, a))
    h_over_c = 100.0 * h * e_z / SPEED_OF_LIGHT_KM_S
    omega_m_z = jc_background.Omega_m_a(cosmo, a)

    # Growth.
    growth = jnp.atleast_1d(jc_background.growth_factor(cosmo, a))
    growth_rate = jnp.atleast_1d(jc_background.growth_rate(cosmo, a))

    # Linear power at z=0 normalised to sigma8, then scaled by D(z)^2 (scale independent growth).
    transfer_k = transfer(cosmo, k_hmpc)
    pknorm = cosmo.sigma8**2 / jc_power.sigmasqr(cosmo, 8.0, transfer)
    p0 = jc_power.primordial_matter_power(cosmo, k_hmpc) * transfer_k**2 * pknorm
    plin = growth[:, None] ** 2 * p0[None, :]

    if nonlinear:

        def _pnl_one(a_i):
            return _smooth_halofit(cosmo, k_hmpc, a_i, transfer, prescription=config.halofit)

        pnl = jax.vmap(_pnl_one)(a)
    else:
        pnl = plin

    h3 = h**3
    return {
        "H_over_c": h_over_c,
        "chi": chi,
        "d_A": d_a,
        "Omega_m_z": omega_m_z,
        "D": growth,
        "f": growth_rate,
        "sigma8_z": cosmo.sigma8 * growth,
        "Plin": plin / h3,
        "Pnl": pnl / h3,
    }


def params_from_vector(vector: jnp.ndarray, names: tuple[str, ...] = PARAM_ORDER) -> dict:
    """Unpack a parameter vector (ordered as ``names``) into the dict used by this module."""
    return {name: vector[i] for i, name in enumerate(names)}


class JaxCosmoTheory:
    """``jax_cosmo`` implementation of :class:`cosmicjellyfish.cosmology.jax_theory.JaxTheory`.

    This is the adapter that lets backend-agnostic JAX observables request theory tables
    without importing ``jax_cosmo`` themselves.

    Parameters
    ----------
    backend_parameters : mapping, optional
        Parsed ``default_boltzmann_yaml_files/jaxcosmo/default.yaml``.  The ``ACCURACY`` block
        configures the transfer function, the Halofit prescription and the distance ODE, and
        ``NUMERICS['x64']`` enables double precision.
    cosmo_model : str
        ``"LCDM"`` or ``"w0waCDM"``.  ``LCDM`` fixes ``w0 = -1`` and ``wa = 0``.
    """

    cosmo_names: tuple[str, ...] = PARAM_ORDER
    free_name_map: Mapping[str, str] = FREE_NAME_MAP

    def __init__(
        self,
        backend_parameters: Mapping | None = None,
        cosmo_model: str = "LCDM",
    ) -> None:
        self.backend_parameters = dict(backend_parameters or {})
        numerics = dict(self.backend_parameters.get("NUMERICS", {}) or {})
        if numerics.get("x64", True):
            enable_x64()
        self.config = TheoryConfig.from_backend_parameters(self.backend_parameters)
        self.cosmo_model = cosmo_model

    def resolve_cosmology(self, params: Mapping) -> dict[str, float]:
        """Resolve CosmicJellyfish parameter names into canonical floats (massless neutrinos)."""
        from cosmicjellyfish.cosmology.jaxcosmo_provider import resolve_parameters

        return resolve_parameters(params, self.cosmo_model)

    def _model_params(self, params: Mapping) -> dict:
        """Pin ``w0``/``wa`` for ``LCDM`` so direct calls cannot evaluate perturbed DE."""
        if self.cosmo_model != "LCDM":
            return dict(params)
        out = dict(params)
        for key, expected in (("w0", -1.0), ("wa", 0.0)):
            if key not in out:
                continue
            try:
                concrete = float(jnp.asarray(out[key]).reshape(-1)[0])
            except Exception:
                concrete = None
            if concrete is not None and concrete != expected:
                raise ValueError(
                    "cosmo_model='LCDM' fixes w0=-1 and wa=0, but got "
                    f"{key}={concrete}. Use cosmo_model='w0waCDM' instead."
                )
            out[key] = expected
        return out

    def tables(
        self,
        params: Mapping,
        z,
        k_hmpc,
        nonlinear: bool = True,
    ) -> TheoryTables:
        """Evaluate the ``jax_cosmo`` theory on ``(z, k_hmpc)``, in CosmicJellyfish units."""
        params = self._model_params(params)
        raw = compute_tables(params, z, k_hmpc, self.config, nonlinear)
        h = params["h"]
        # jax_cosmo's background has no radiation, so H(0)/c = 100 h / c exactly.
        h0_over_c = 100.0 * h / SPEED_OF_LIGHT_KM_S
        return TheoryTables(
            z=z,
            k_hmpc=k_hmpc,
            chi=raw["chi"],
            h_over_c=raw["H_over_c"],
            growth=raw["D"],
            growth_rate=raw["f"],
            omega_m=raw["Omega_m_z"],
            pk_linear=raw["Plin"],
            pk_nonlinear=raw["Pnl"],
            h0_over_c=h0_over_c,
            omega_m0=params["Omegam"],
            sigma8=raw["sigma8_z"],
        )

    def power(
        self,
        params: Mapping,
        z,
        k_hmpc,
        nonlinear: bool = True,
    ) -> jnp.ndarray:
        """Matter power at *per-redshift* wavenumbers, in ``Mpc^3``.

        Unlike :meth:`tables`, this evaluates the transfer function and Halofit directly at the
        requested ``k`` values instead of interpolating a fixed wavenumber grid.  That removes
        the slope kinks of the interpolation, which is what makes finite differences of the
        projected spectra converge slowly.
        """
        params = self._model_params(params)
        z = jnp.atleast_1d(jnp.asarray(z))
        k = jnp.asarray(k_hmpc)
        if k.ndim == 1:
            k = jnp.broadcast_to(k[None, :], (z.shape[0], k.shape[0]))

        cosmo = make_cosmology(params)
        h = cosmo.h
        a = 1.0 / (1.0 + z)
        transfer = _transfer_fn(self.config)

        # Prime the cached tables outside the transformation (see ``compute_tables``).
        a_prime = jnp.atleast_1d(1.0)
        jc_background.radial_comoving_distance(cosmo, a_prime, steps=self.config.distance_steps)
        jc_background.growth_factor(cosmo, a_prime)

        if nonlinear:

            def _one(a_i, k_i):
                return _smooth_halofit(cosmo, k_i, a_i, transfer, prescription=self.config.halofit)

            pk = jax.vmap(_one)(a, k)
        else:
            transfer_k = transfer(cosmo, k)
            pknorm = cosmo.sigma8**2 / jc_power.sigmasqr(cosmo, 8.0, transfer)
            p0 = jc_power.primordial_matter_power(cosmo, k) * transfer_k**2 * pknorm
            growth = jnp.atleast_1d(jc_background.growth_factor(cosmo, a))
            pk = growth[:, None] ** 2 * p0

        return pk / h**3
