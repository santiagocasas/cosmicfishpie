"""CosmicFishPie-facing provider of the differentiable ``jaxcosmo`` backend.

The heavy lifting is done by the pure-JAX functions of
:mod:`cosmicfishpie.cosmology.jaxcosmo_theory`.  This module is the *outer boundary*: it
resolves CosmicFishPie parameter conventions, evaluates the (jitted) theory on the configured
``(z, k)`` grids and wraps the results into the SciPy-spline-like objects expected by
:class:`cosmicfishpie.cosmology.cosmology.cosmo_functions`.  Gradients do not flow through this
module; the autodiff path uses the theory layer directly.

Supported model: flat ``LCDM`` / ``w0waCDM`` with massless neutrinos, total-matter power only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

from cosmicfishpie.cosmology import jaxcosmo_theory as theory
from cosmicfishpie.cosmology.jaxcosmo_amplitude import sigma8_to_As

_MNU_TOLERANCE = 1.0e-12
_AMPLITUDE_KEYS = ("10^9As", "As", "logAs", "ln_A_s_1e10")

# One jitted callable per process: compiled functions are never pickled, so spawned Nautilus
# workers simply build their own on first use.
_TABLES_JIT = None


def _tables_jit():
    global _TABLES_JIT
    if _TABLES_JIT is None:
        _TABLES_JIT = theory.jax.jit(
            theory.compute_tables, static_argnames=("config", "nonlinear")
        )
    return _TABLES_JIT


def _as_scalar(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def resolve_parameters(cosmopars: Mapping, cosmo_model: str) -> dict[str, float]:
    """Map CosmicFishPie parameters to the ``jaxcosmo`` parameter dict.

    Returns the keys of :data:`cosmicfishpie.cosmology.jaxcosmo_theory.PARAM_ORDER`
    (``As`` is ``10^9 As``).

    Raises
    ------
    ValueError
        For unsupported models, massive neutrinos, curvature, ambiguous or unphysical input.
    """
    if cosmo_model not in ("LCDM", "w0waCDM"):
        raise ValueError(f"The jaxcosmo backend does not support cosmo_model={cosmo_model!r}.")
    pars = dict(cosmopars)

    mnu = _as_scalar(pars.get("mnu", pars.get("m_nu", 0.0)))
    if abs(mnu) > _MNU_TOLERANCE:
        raise ValueError(
            "The jaxcosmo backend supports massless neutrinos only; "
            f"got mnu={mnu}. Set mnu=0 in the fiducial cosmology."
        )
    for key in ("Omegak", "omk", "Omega_k"):
        if key in pars and abs(_as_scalar(pars[key])) > _MNU_TOLERANCE:
            raise ValueError("The jaxcosmo backend supports flat cosmologies only.")

    h = _as_scalar(pars["h"]) if "h" in pars else _as_scalar(pars.get("H0", 67.0)) / 100.0
    omega_b = _as_scalar(pars["Omegab"]) if "Omegab" in pars else None
    if omega_b is None:
        omega_b = _as_scalar(pars.get("ombh2", 0.05 * h**2)) / h**2
    if "Omegam" in pars:
        omega_m = _as_scalar(pars["Omegam"])
    elif "omch2" in pars:
        omega_m = _as_scalar(pars["omch2"]) / h**2 + omega_b
    else:
        omega_m = 0.32
    ns = _as_scalar(pars.get("ns", pars.get("n_s", 0.96)))
    if cosmo_model == "LCDM":
        w0, wa = -1.0, 0.0
    else:
        w0 = _as_scalar(pars.get("w0", pars.get("w0_fld", -1.0)))
        wa = _as_scalar(pars.get("wa", pars.get("wa_fld", 0.0)))

    if omega_b <= 0 or omega_m <= omega_b:
        raise ValueError("The jaxcosmo backend requires Omegam > Omegab > 0.")

    given = [key for key in (*_AMPLITUDE_KEYS, "sigma8") if key in pars]
    if "sigma8" in given and len(given) > 1:
        raise ValueError(
            f"Ambiguous primordial amplitude: both sigma8 and {given} were given. Pass one."
        )
    if "10^9As" in pars:
        as_value = _as_scalar(pars["10^9As"])
    elif "As" in pars:
        as_value = _as_scalar(pars["As"]) * 1.0e9
    elif "logAs" in pars:
        as_value = float(np.exp(_as_scalar(pars["logAs"])) * 0.1)
    elif "ln_A_s_1e10" in pars:
        as_value = float(np.exp(_as_scalar(pars["ln_A_s_1e10"])) * 0.1)
    else:
        sigma8 = _as_scalar(pars.get("sigma8", 0.815583))
        as_value = float(sigma8_to_As(sigma8, omega_m, omega_b, h, ns, 0.0, w0, wa))

    return {
        "As": as_value,
        "Omegam": omega_m,
        "Omegab": omega_b,
        "h": h,
        "ns": ns,
        "w0": w0,
        "wa": wa,
    }


class JaxCosmoProvider:
    """Build total-matter background and power-spectrum interpolators with ``jax-cosmo``.

    Parameters
    ----------
    cosmopars : dict
        Cosmological parameters in CosmicFishPie conventions.
    backend_parameters : dict
        Content of ``jaxcosmo/default.yaml`` (``ACCURACY`` and ``NUMERICS`` blocks).
    cosmo_model : str
        ``"LCDM"`` or ``"w0waCDM"``.
    settings : dict, optional
        Global ``cfg.settings`` (currently unused, kept for interface symmetry).
    """

    def __init__(
        self,
        cosmopars: Mapping,
        backend_parameters: Mapping,
        cosmo_model: str,
        settings: Mapping | None = None,
    ):
        self.numerics = dict(backend_parameters["NUMERICS"])
        self.config = theory.TheoryConfig.from_backend_parameters(backend_parameters)
        self.cosmo_model = cosmo_model
        self.settings = settings
        if self.numerics.get("x64", True):
            theory.enable_x64()
        self.cosmopars = resolve_parameters(cosmopars, cosmo_model)

    def _theory_params(self) -> dict[str, float]:
        return {name: self.cosmopars[name] for name in theory.PARAM_ORDER}

    def build(self) -> SimpleNamespace:
        """Evaluate the theory and return the ``results`` namespace used by ``cosmo_functions``."""
        numerics = self.numerics
        h = self.cosmopars["h"]
        params = self._theory_params()
        tables_jit = _tables_jit()

        zgrid = np.linspace(numerics["zmin_pk"], numerics["zmax_pk"], numerics["z_samples"])
        kgrid_hmpc = np.logspace(
            np.log10(numerics["kmin_pk"]), np.log10(numerics["kmax_pk"]), numerics["k_samples"]
        )
        kgrid = kgrid_hmpc * h  # 1/Mpc

        zmax_bg = max(float(numerics.get("zmax_background", 10.0)), float(numerics["zmax_pk"]))
        zbg = np.linspace(0.0, zmax_bg, int(numerics.get("z_samples_background", 512)))

        # Background on a dense, separate grid: cheap, linear power only on a tiny k grid.
        bg = self._to_numpy(
            tables_jit(params, zbg, kgrid_hmpc[:4], config=self.config, nonlinear=False)
        )
        pk = self._to_numpy(
            tables_jit(params, zgrid, kgrid_hmpc, config=self.config, nonlinear=True)
        )

        def _spline(values: np.ndarray) -> InterpolatedUnivariateSpline:
            return InterpolatedUnivariateSpline(zbg, values, k=3)

        results = SimpleNamespace()
        results.zgrid = zgrid
        results.kgrid = kgrid
        results.h_of_z = _spline(bg["H_over_c"])
        results.ang_dist = _spline(bg["d_A"])
        results.com_dist = _spline(bg["chi"])
        results.Om_m = _spline(bg["Omega_m_z"])
        results.Pk_l = RectBivariateSpline(zgrid, kgrid, pk["Plin"])
        results.Pk_nl = RectBivariateSpline(zgrid, kgrid, pk["Pnl"])
        # Growth is scale independent in this backend: broadcast over k.
        results.D_growth_zk = RectBivariateSpline(
            zgrid, kgrid, np.repeat(pk["D"][:, None], kgrid.size, axis=1)
        )
        results.f_growthrate_zk = RectBivariateSpline(
            zgrid, kgrid, np.repeat(pk["f"][:, None], kgrid.size, axis=1)
        )
        results.s8_of_z = InterpolatedUnivariateSpline(zgrid, pk["sigma8_z"])
        return results

    @staticmethod
    def _to_numpy(tables: Mapping) -> dict[str, np.ndarray]:
        return {name: np.asarray(value, dtype=float) for name, value in tables.items()}
