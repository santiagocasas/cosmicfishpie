"""Total-matter SYREN-NEW cosmology provider.

The public :mod:`cosmology` interface uses physical ``1/Mpc`` and ``Mpc^3``
units.  SYREN-NEW uses ``h/Mpc`` and ``(Mpc/h)^3``; conversion happens only in
this module.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

SPEED_OF_LIGHT_KM_S = 299792.458


def _as_scalar(value: float | np.ndarray) -> float:
    return float(np.asarray(value))


class SyrenNewProvider:
    """Build total-matter background and power-spectrum interpolators.

    Parameters are accepted in CosmicFishPie conventions.  The provider is
    intentionally total-matter-only because SYREN-NEW does not provide a cb
    spectrum.
    """

    def __init__(self, cosmopars: dict, backend_parameters: dict, cosmo_model: str):
        self.cosmopars = self._resolve_parameters(cosmopars, cosmo_model)
        self.backend_parameters = backend_parameters
        self.numerics = backend_parameters["NUMERICS"]
        self.cosmo_model = cosmo_model

    @staticmethod
    def _resolve_parameters(cosmopars: dict, cosmo_model: str) -> dict[str, float]:
        parameters = dict(cosmopars)
        h = _as_scalar(parameters.get("h", parameters.get("H0", 67.0) / 100.0))
        omega_b = _as_scalar(parameters.get("Omegab", parameters.get("ombh2", 0.05 * h**2) / h**2))
        mnu = _as_scalar(parameters.get("mnu", parameters.get("m_nu", 0.06)))
        omega_nu = mnu / 94.07 / h**2
        if "Omegam" in parameters:
            omega_m = _as_scalar(parameters["Omegam"])
        elif "omch2" in parameters:
            omega_m = _as_scalar(parameters["omch2"]) / h**2 + omega_b + omega_nu
        else:
            omega_m = 0.32
        ns = _as_scalar(parameters.get("ns", parameters.get("n_s", 0.96)))
        w0 = _as_scalar(parameters.get("w0", parameters.get("w0_fld", -1.0)))
        wa = _as_scalar(parameters.get("wa", parameters.get("wa_fld", 0.0)))

        if cosmo_model == "LCDM":
            w0, wa = -1.0, 0.0
        elif cosmo_model != "w0waCDM":
            raise ValueError(f"SYREN-NEW does not support cosmo_model={cosmo_model!r}.")

        as_value = parameters.get("10^9As")
        if as_value is None and "As" in parameters:
            as_value = _as_scalar(parameters["As"]) * 1.0e9
        if as_value is None and "logAs" in parameters:
            as_value = np.exp(_as_scalar(parameters["logAs"])) * 0.1

        if as_value is None:
            from symbolic_pofk.linear_new import sigma8_to_As_max_precision

            as_value = sigma8_to_As_max_precision(
                _as_scalar(parameters.get("sigma8", 0.815583)),
                omega_m,
                omega_b,
                h,
                ns,
                mnu,
                w0,
                wa,
            )

        resolved = {
            "As": _as_scalar(as_value),
            "Omegam": omega_m,
            "Omegab": omega_b,
            "h": h,
            "ns": ns,
            "mnu": mnu,
            "w0": w0,
            "wa": wa,
        }
        if omega_b <= 0 or omega_m <= omega_b + omega_nu:
            raise ValueError("SYREN-NEW requires Omegam > Omegab + Omega_nu and Omegab > 0.")
        return resolved

    def build(self) -> SimpleNamespace:
        from symbolic_pofk.linear_new import plin_new_emulated
        from symbolic_pofk.syren_new import pnl_new_emulated

        h = self.cosmopars["h"]
        zgrid = np.linspace(
            self.numerics["zmin_pk"], self.numerics["zmax_pk"], self.numerics["z_samples"]
        )
        kgrid_hmpc = np.logspace(
            np.log10(self.numerics["kmin_pk"]),
            np.log10(self.numerics["kmax_pk"]),
            self.numerics["k_samples"],
        )
        kgrid = kgrid_hmpc * h
        args = (
            self.cosmopars["As"],
            self.cosmopars["Omegam"],
            self.cosmopars["Omegab"],
            h,
            self.cosmopars["ns"],
            self.cosmopars["mnu"],
            self.cosmopars["w0"],
            self.cosmopars["wa"],
        )
        scale_factors = 1.0 / (1.0 + zgrid)
        pk_linear = (
            np.array(
                [
                    plin_new_emulated(kgrid_hmpc, *args, a=scale_factor)
                    for scale_factor in scale_factors
                ]
            )
            / h**3
        )
        pk_nonlinear = (
            np.array(
                [
                    pnl_new_emulated(kgrid_hmpc, *args, a=scale_factor)
                    for scale_factor in scale_factors
                ]
            )
            / h**3
        )

        cosmology = self._background()
        growth = np.sqrt(pk_linear / pk_linear[0])
        growth_spline = RectBivariateSpline(zgrid, kgrid, growth)
        growth_rate = -(1.0 + zgrid[:, None]) * np.gradient(
            np.log(growth), zgrid, axis=0, edge_order=2
        )

        results = SimpleNamespace()
        results.zgrid = zgrid
        results.kgrid = kgrid
        results.h_of_z = np.vectorize(
            lambda z: _as_scalar(cosmology.H(z).value) / SPEED_OF_LIGHT_KM_S
        )
        results.ang_dist = np.vectorize(
            lambda z: _as_scalar(cosmology.angular_diameter_distance(z).value)
        )
        results.com_dist = np.vectorize(lambda z: _as_scalar(cosmology.comoving_distance(z).value))
        results.Om_m = np.vectorize(lambda z: _as_scalar(cosmology.Om(z)))
        results.Pk_l = RectBivariateSpline(zgrid, kgrid, pk_linear)
        results.Pk_nl = RectBivariateSpline(zgrid, kgrid, pk_nonlinear)
        results.D_growth_zk = growth_spline
        results.f_growthrate_zk = RectBivariateSpline(zgrid, kgrid, growth_rate)
        results.s8_of_z = InterpolatedUnivariateSpline(
            zgrid, [self._sigma8(kgrid, power, h) for power in pk_linear]
        )
        return results

    def _background(self):
        kwargs = {
            "H0": self.cosmopars["h"] * 100,
            "Om0": self.cosmopars["Omegam"],
            "Ob0": self.cosmopars["Omegab"],
            "Tcmb0": 2.7255,
            "m_nu": self.cosmopars["mnu"] / 3.0,
        }
        if self.cosmo_model == "LCDM":
            return FlatLambdaCDM(**kwargs)
        return Flatw0waCDM(w0=self.cosmopars["w0"], wa=self.cosmopars["wa"], **kwargs)

    @staticmethod
    def _sigma8(kgrid: np.ndarray, power: np.ndarray, h: float) -> float:
        radius = 8.0 / h
        kr = kgrid * radius
        window = np.where(
            kr == 0,
            1.0,
            3.0 * (np.sin(kr) - kr * np.cos(kr)) / kr**3,
        )
        variance = simpson(kgrid**3 * power * window**2, x=np.log(kgrid)) / (2.0 * np.pi**2)
        return float(np.sqrt(variance))
