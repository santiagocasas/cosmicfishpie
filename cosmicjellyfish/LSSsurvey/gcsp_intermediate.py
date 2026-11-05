r"""Intermediate-parameter forecast for spectroscopic galaxy clustering.

The Euclid forecast-validation paper (Euclid preparation VII, section 4.2.1) splits the
galaxy-clustering Fisher analysis into two stages.  First it differentiates the observed
power spectrum with respect to an intermediate parameter set made of

* four shape parameters :math:`\omega_{b,0}, h, \omega_{m,0}, n_s`,
* two background parameters per redshift bin, :math:`\ln D_A(z_i)` and :math:`\ln H(z_i)`,
* two growth parameters per bin, :math:`\ln f\sigma_8(z_i)` and :math:`\ln b\sigma_8(z_i)`,
* one shot-noise amplitude per bin, :math:`P_s(z_i)`,

and then projects the resulting Fisher matrix onto the final cosmological parameters with
the Jacobian of the intermediate-to-final map.

This module implements that recipe on top of the validated
:class:`~cosmicjellyfish.LSSsurvey.jaxcosmo_spectro.JaxSpectroObservable`.  The observable
is written in the :math:`f\sigma_8`/ :math:`b\sigma_8` form and is algebraically identical
to the validated ``bfs8terms=False`` observable at the fiducial point.  Both paper-style
finite differences and JAX automatic differentiation are available on the *same* function,
so the two derivative engines can be compared directly.

The projection is the Fisher-space chain rule,

.. math::

    F_{\alpha\beta} = J^{T}_{\alpha i} F_{ij} J_{j\beta},
    \qquad J_{i\beta} = \frac{\partial \theta_i^{\rm int}}{\partial \theta_\beta^{\rm final}},

where the intermediate Fisher matrix has already been marginalised over the observational
nuisances :math:`b\sigma_8(z_i)` and :math:`P_s(z_i)`.  This is *not* the covariance
pull-back :math:`A^{T} F^{-1} A` implemented by
:class:`~cosmicjellyfish.analysis.fisher_derived.fisher_derived`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import jax
    import jax.numpy as jnp
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "The intermediate GCsp forecast requires jax. "
        "Install it with: pip install 'cosmicjellyfish[jaxcosmo]'"
    ) from err

from cosmicjellyfish.LSSsurvey.jaxcosmo_spectro import JaxSpectroObservable

_SHAPE_NAMES = ("omegab", "h", "omegam", "ns")
_FINAL_CANONICAL = {
    "10^9As": "As",
    "As": "As",
    "Omegam": "Omegam",
    "Omegab": "Omegab",
    "h": "h",
    "ns": "ns",
    "w0": "w0",
    "wa": "wa",
}
_FINAL_SCALE = {
    "10^9As": 1.0,
    "As": 1.0e9,
    "Omegam": 1.0,
    "Omegab": 1.0,
    "h": 1.0,
    "ns": 1.0,
    "w0": 1.0,
    "wa": 1.0,
}
_AMPLITUDE_KEYS = ("10^9As", "As", "logAs", "ln_A_s_1e10", "sigma8")
_DEFAULT_FIDUCIAL = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "w0": -1.0,
    "wa": 0.0,
    "h": 0.67,
    "ns": 0.96,
}
_NUMERIC_COSMOLOGY_KEYS = set(_DEFAULT_FIDUCIAL) | {
    "10^9As",
    "As",
    "logAs",
    "ln_A_s_1e10",
    "sigma8",
    "mnu",
    "m_nu",
    "Omeganu",
    "omnuh2",
    "Omegak",
    "omk",
    "Omega_k",
    "Neff",
    "N_eff",
    "tau",
    "Tcmb",
    "Omch2",
    "ombh2",
    "H0",
    "n_s",
    "w0_fld",
    "wa_fld",
}


def _resolve_fiducial(fiducialpars: Mapping | None) -> dict[str, float]:
    """Return a fiducial dictionary compatible with the massless jaxcosmo backend."""
    fiducial = dict(_DEFAULT_FIDUCIAL)
    if fiducialpars is not None:
        fiducial.update(dict(fiducialpars))
    numeric_fiducial = {}
    for key, value in fiducial.items():
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as err:
                if key in _NUMERIC_COSMOLOGY_KEYS:
                    raise ValueError(
                        f"Fiducial cosmology parameter '{key}' must be numeric; got {value!r}."
                    ) from err
                continue
        try:
            numeric_fiducial[key] = float(value)
        except (TypeError, ValueError):
            if key in _NUMERIC_COSMOLOGY_KEYS:
                raise ValueError(f"Fiducial cosmology parameter '{key}' must be numeric.")
    fiducial = numeric_fiducial
    for key in ("mnu", "m_nu", "Omeganu", "omnuh2"):
        if key in fiducial and abs(float(fiducial[key])) > 1.0e-12:
            raise ValueError(
                "The jaxcosmo backend supports massless neutrinos only; "
                f"set {key}=0 in fiducialpars."
            )
        fiducial.pop(key, None)
    fiducial["mnu"] = 0.0
    fiducial = {key: value for key, value in fiducial.items() if not isinstance(value, str)}
    if not any(key in fiducial for key in _AMPLITUDE_KEYS):
        fiducial["10^9As"] = 2.1
    return fiducial


def _central_3pt(function, point: np.ndarray, index: int, step: float) -> np.ndarray:
    """Central three-point derivative of ``function`` with respect to ``point[index]``."""
    plus = np.array(point, copy=True)
    minus = np.array(point, copy=True)
    plus[index] += step
    minus[index] -= step
    return (function(plus) - function(minus)) / (2.0 * step)


def _central_5pt(function, point: np.ndarray, index: int, step: float) -> np.ndarray:
    """Central five-point derivative of ``function`` with respect to ``point[index]``."""
    plus = np.array(point, copy=True)
    minus = np.array(point, copy=True)
    plus2 = np.array(point, copy=True)
    minus2 = np.array(point, copy=True)
    plus[index] += step
    minus[index] -= step
    plus2[index] += 2.0 * step
    minus2[index] -= 2.0 * step
    return (8.0 * (function(plus) - function(minus)) - (function(plus2) - function(minus2))) / (
        12.0 * step
    )


class IntermediateSpectroObservable:
    r"""Spectroscopic galaxy-clustering observable in intermediate parameters.

    Parameters
    ----------
    base : JaxSpectroObservable
        Validated differentiable observable.  Its grids, theory object, fiducial
        background quantities, bias model and no-wiggle operator are reused.
    include_nonlinear : bool, optional
        If true, add the de-wiggling and fingers-of-god velocity dispersions per bin as
        free log-parameters.  The default is false, matching the fiducial-moment
        approximation of the validated observable.

    Notes
    -----
    The intermediate parameter vector is ordered as the four shape parameters
    ``(omegab, h, omegam, ns)``, with :math:`\omega_i = \Omega_i h^2`, followed by
    the redshift-bin blocks ``[lnDA_i, lnH_i, lnfs8_i, lnbs8_i, Ps_i]`` and, when enabled,
    ``[lnsigmav_i, lnsigmap_i]``.  Shot noise is always present as an intermediate
    parameter; its fiducial value is read from the base observable when available and is
    zero otherwise.
    """

    def __init__(
        self,
        base: JaxSpectroObservable,
        include_nonlinear: bool = False,
    ):
        if base.bias_model == "linear_Qbias":
            raise NotImplementedError(
                "The intermediate observable fixes the Q-bias shape parameters A1 and A2; "
                "use the default linear or linear_log bias model."
            )
        self.base = base
        self.include_nonlinear = bool(include_nonlinear)
        self.z = np.asarray(base.z)
        self.nbins = int(base.nbins)
        self.nmu, self.nk = base.shape_kmu
        self._base_cosmo = {name: float(base.base_values[name]) for name in base.cosmo_names}

        params = {name: self._base_cosmo[name] for name in base.cosmo_names}
        z = jnp.asarray(self.z)
        table = base.theory.tables(params, z, base.k_hmpc[:2], False)
        self.sigma8_fid = np.asarray(table.sigma8)
        self.f_fid = np.asarray(table.growth_rate)
        self.lnDA_fid = np.log(np.asarray(base.da_fid))
        self.lnH_fid = np.log(np.asarray(base.hub_fid))
        bias_fid = np.asarray(base._bias(base.base_values))
        self.fs8_fid = self.f_fid * self.sigma8_fid
        self.bs8_fid = bias_fid * self.sigma8_fid
        self.lnfs8_fid = np.log(self.fs8_fid)
        self.lnbs8_fid = np.log(self.bs8_fid)
        self.ps_fid = np.asarray(
            [float(base.fiducial.get(f"Ps_{ibin + 1}", 0.0)) for ibin in range(self.nbins)]
        )
        self.sigmav_fid = np.sqrt(np.asarray(base.theta_moments[0]))
        self.sigmap_fid = np.asarray(base.sigmap)
        self._local_stride = 7 if self.include_nonlinear else 5
        self._names = self._build_names()
        self._name_to_index = {name: index for index, name in enumerate(self._names)}
        self._lnp_jit = jax.jit(self._lnp_from_vector)

    def _build_names(self) -> tuple[str, ...]:
        names = list(_SHAPE_NAMES)
        for ibin in range(self.nbins):
            suffix = ibin + 1
            names.extend(
                [
                    f"lnDA_{suffix}",
                    f"lnH_{suffix}",
                    f"lnfs8_{suffix}",
                    f"lnbs8_{suffix}",
                    f"Ps_{suffix}",
                ]
            )
            if self.include_nonlinear:
                names.extend([f"lnsigmav_{suffix}", f"lnsigmap_{suffix}"])
        return tuple(names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the intermediate parameters, in vector order."""
        return self._names

    def fiducial_params(self) -> dict[str, float]:
        """Fiducial intermediate parameters derived from the base cosmology."""
        fiducial = {
            "omegab": float(self._base_cosmo["Omegab"] * self._base_cosmo["h"] ** 2),
            "h": float(self._base_cosmo["h"]),
            "omegam": float(self._base_cosmo["Omegam"] * self._base_cosmo["h"] ** 2),
            "ns": float(self._base_cosmo["ns"]),
        }
        for ibin in range(self.nbins):
            suffix = ibin + 1
            fiducial[f"lnDA_{suffix}"] = float(self.lnDA_fid[ibin])
            fiducial[f"lnH_{suffix}"] = float(self.lnH_fid[ibin])
            fiducial[f"lnfs8_{suffix}"] = float(self.lnfs8_fid[ibin])
            fiducial[f"lnbs8_{suffix}"] = float(self.lnbs8_fid[ibin])
            fiducial[f"Ps_{suffix}"] = float(self.ps_fid[ibin])
            if self.include_nonlinear:
                fiducial[f"lnsigmav_{suffix}"] = float(np.log(self.sigmav_fid[ibin]))
                fiducial[f"lnsigmap_{suffix}"] = float(np.log(self.sigmap_fid[ibin]))
        return fiducial

    def fiducial_vector(self) -> np.ndarray:
        """Fiducial intermediate parameter vector."""
        fiducial = self.fiducial_params()
        return np.array([fiducial[name] for name in self._names], dtype=float)

    def to_vector(self, overrides: Mapping[str, float] | None = None) -> np.ndarray:
        """Return the fiducial vector with a subset of parameters replaced."""
        vector = self.fiducial_vector()
        for name, value in (overrides or {}).items():
            if name not in self._name_to_index:
                raise ValueError(f"Unknown intermediate parameter '{name}'.")
            vector[self._name_to_index[name]] = float(value)
        return vector

    def _lnp_from_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        base = self.base
        z = jnp.asarray(self.z)
        shape = (self.nbins, self.nmu, self.nk)
        km = jnp.broadcast_to(jnp.asarray(base.km), shape)
        mum = jnp.broadcast_to(jnp.asarray(base.mum), shape)

        cosmo = dict(self._base_cosmo)
        cosmo["h"] = vector[1]
        cosmo["Omegab"] = vector[0] / vector[1] ** 2
        cosmo["Omegam"] = vector[2] / vector[1] ** 2
        cosmo["ns"] = vector[3]

        local = vector[4:].reshape(self.nbins, self._local_stride)
        lnDA = local[:, 0]
        lnH = local[:, 1]
        lnfs8 = local[:, 2]
        lnbs8 = local[:, 3]
        ps = local[:, 4]
        if self.include_nonlinear:
            lnsigmav = local[:, 5]
            lnsigmap = local[:, 6]
            sigmav = jnp.exp(lnsigmav)
            sigmap = jnp.exp(lnsigmap)
        else:
            sigmav = jnp.asarray(self.sigmav_fid)
            sigmap = jnp.asarray(self.sigmap_fid)

        table = base.theory.tables(cosmo, z, base.k_hmpc[:2], False)
        sigma8 = table.sigma8

        qpar = jnp.exp(self.lnH_fid - lnH)[:, None, None]
        if base.kh_rescaling_bug:
            k_h = km * (cosmo["h"] / base.h_fid)
            k_err = k_h if base.kh_bug_before_err else km
        else:
            k_h = km
            k_err = km

        if base.dz_type == "z-dependent":
            dz_err = (base.dz_err * (1.0 + z))[:, None, None]
        else:
            dz_err = base.dz_err
        err = jnp.exp(-0.5 * (dz_err * jnp.exp(-lnH)[:, None, None] * (k_err * mum / qpar)) ** 2)

        if base.ap:
            qper = jnp.exp(lnDA - self.lnDA_fid)[:, None, None]
            kpar = k_h * mum / qpar
            kper = k_h * jnp.sqrt(1.0 - mum**2) / qper
            kap = jnp.sqrt(kpar**2 + kper**2)
            muap = kpar / kap
            bao = 1.0 / (qper**2 * qpar)
        else:
            kap = k_h
            muap = mum
            bao = 1.0

        kscale = base._bias_kscale(kap, base.base_values)
        kaiser = jnp.exp(lnbs8)[:, None, None] * kscale + jnp.exp(lnfs8)[:, None, None] * muap**2

        if base.linear_switch or not base.fog_switch:
            fog = 1.0
        else:
            fog = 1.0 / (1.0 + (kap * muap * sigmap[:, None, None]) ** 2)

        theta0, theta1, theta2 = (
            jnp.asarray(base.theta_moments[i])[:, None, None] for i in range(3)
        )
        if base.linear_switch:
            gmu = 0.0
        else:
            gmu = (sigmav**2 / jnp.asarray(self.sigmav_fid) ** 2)[:, None, None] * (
                theta0 + 2.0 * muap**2 * theta1 + muap**2 * theta2
            )

        k_eval = kap
        k_hmpc = (k_eval / cosmo["h"]).reshape(self.nbins, -1)
        if base.legacy_pk_clamp:
            k_hmpc = jnp.clip(k_hmpc, base.kmin_pk, base.kmax_pk)
        p_dd = base.theory.power(cosmo, z, k_hmpc, False).reshape(kap.shape)
        p_nw = base._nowiggle_power(cosmo, k_eval.reshape(self.nbins, -1)).reshape(kap.shape)
        damp = jnp.exp(-gmu * kap**2)
        p_dw = p_dd * damp + p_nw * (1.0 - damp)

        normalized = p_dw / sigma8[:, None, None] ** 2
        pobs = bao * kaiser**2 * normalized * fog * err**2
        pobs = pobs + base.extra_shot + ps[:, None, None]
        return jnp.log(pobs)

    def lnp_vector(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        """Log observed power spectrum for an intermediate parameter vector."""
        return np.asarray(self._lnp_jit(jnp.asarray(vector, dtype=float)))

    def lnp(self, overrides: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        """Log observed power spectrum as the base observable's per-bin dictionary."""
        return self.base.as_dict(self.lnp_vector(self.to_vector(overrides)))

    def __call__(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        """Alias of :meth:`lnp_vector`."""
        return self.lnp_vector(vector)

    def _jacobian_all(self) -> np.ndarray:
        jacobian = jax.jit(jax.jacfwd(self._lnp_from_vector))(jnp.asarray(self.fiducial_vector()))
        return np.moveaxis(np.asarray(jacobian), -1, 0)

    def jacobian_ad(self, names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """Derivatives of ``ln P`` from JAX reverse/forward autodiff."""
        all_names = self._names if names is None else tuple(names)
        jacobian = self._jacobian_all()
        result = {}
        for name in all_names:
            if name not in self._name_to_index:
                raise ValueError(f"Unknown intermediate parameter '{name}'.")
            result[name] = jacobian[self._name_to_index[name]]
        return result

    def jacobian_fd(
        self,
        names: Iterable[str] | None = None,
        step: float = 5.0e-6,
        background_step: float = 5.0e-6,
        analytic_shot_noise: bool = True,
    ) -> dict[str, np.ndarray]:
        r"""Paper-style finite-difference derivatives of ``ln P``.

        Three-point central differences are used for the shape parameters (with a
        relative step, :math:`\theta(1\pm\epsilon)`) and for the log growth, log bias
        and log velocity-dispersion parameters (with an absolute step in the logarithm).  The background parameters :math:`\ln D_A` and
        :math:`\ln H` use the paper's five-point rule.  The shot-noise derivative is
        analytic, :math:`\partial \ln P_{\rm obs}/\partial P_s = 1/P_{\rm obs}`.

        The no-wiggle spectrum is interpolated linearly in :math:`k`, so the step is much
        smaller than the paper's recommended range: a step of :math:`10^{-3}` crosses
        interpolation nodes and biases the derivatives at the per-mille level.  With the
        defaults used here the finite-difference and autodiff derivatives agree to about
        :math:`10^{-10}` in relative terms on the validation grid.
        """
        all_names = self._names if names is None else tuple(names)
        point = self.fiducial_vector()
        fiducial_lnp = self.lnp_vector(point)
        result = {}
        for name in all_names:
            if name not in self._name_to_index:
                raise ValueError(f"Unknown intermediate parameter '{name}'.")
            index = self._name_to_index[name]
            if analytic_shot_noise and name.startswith("Ps_"):
                derivative = np.zeros((self.nbins, self.nmu, self.nk), dtype=float)
                ibin = int(name.rsplit("_", 1)[-1]) - 1
                derivative[ibin] = 1.0 / np.exp(fiducial_lnp[ibin])
            elif name.startswith(("lnDA_", "lnH_")):
                derivative = _central_5pt(self.lnp_vector, point, index, background_step)
            elif name in _SHAPE_NAMES:
                # Relative step, as in the paper: theta (1 +/- eps).
                derivative = _central_3pt(self.lnp_vector, point, index, step * abs(point[index]))
            else:
                derivative = _central_3pt(self.lnp_vector, point, index, step)
            result[name] = derivative
        return result


class GCspIntermediateForecast:
    r"""End-to-end intermediate-parameter Fisher forecast for GCsp.

    Parameters
    ----------
    options : dict, optional
        CosmicJellyfish options.  ``code`` is forced to ``"jaxcosmo"`` and the default is
        ``{"code": "jaxcosmo", "nonlinear": True, "feedback": 0}``.
    fiducialpars : dict, optional
        Fiducial cosmological parameters passed to
        :class:`~cosmicjellyfish.fishermatrix.cosmicfish.FisherMatrix`.
    freepars : dict, optional
        Free final parameters and relative steps.  The default is
        ``{"10^9As": 0.01, "Omegam": 0.01, "Omegab": 0.01, "h": 0.01, "ns": 0.01}``,
        plus ``w0`` and ``wa`` for ``cosmoModel="w0waCDM"``.
    surveyName : str, optional
        Survey specification name.  The default is ``"Euclid"``.
    cosmoModel : str, optional
        ``"LCDM"`` or ``"w0waCDM"``.  The default is ``"LCDM"``.
    specifications : dict, optional
        Survey specification overrides.
    include_nonlinear : bool, optional
        Whether to keep the velocity dispersions as free intermediate parameters.
    fisher : FisherMatrix, optional
        Pre-built Fisher matrix configuration.  When omitted one is constructed from the
        arguments above.

    Attributes
    ----------
    base : JaxSpectroObservable
        Validated observable used as the reference at the fiducial point.
    intermediate : IntermediateSpectroObservable
        Observable expressed in intermediate parameters.
    fisher : FisherMatrix
        Configuration object providing grids, specifications and the fiducial cosmology.
    pk_cov : SpectroCov
        Spectroscopic covariance used for the effective volume.
    """

    def __init__(
        self,
        options: Mapping | None = None,
        fiducialpars: Mapping | None = None,
        freepars: Mapping | None = None,
        surveyName: str = "Euclid",
        cosmoModel: str = "LCDM",
        specifications: Mapping | None = None,
        include_nonlinear: bool = False,
        fisher=None,
    ):
        if cosmoModel not in ("LCDM", "w0waCDM"):
            raise ValueError("cosmoModel must be either 'LCDM' or 'w0waCDM'.")
        self.include_nonlinear = bool(include_nonlinear)
        self.cosmo_model = cosmoModel
        self.final_names = (
            tuple(dict(freepars)) if freepars is not None else self._default_final_names(cosmoModel)
        )

        if fisher is None:
            resolved_options = {"code": "jaxcosmo", "nonlinear": True, "feedback": 0}
            resolved_options.update(dict(options or {}))
            resolved_options["code"] = "jaxcosmo"
            resolved_freepars = (
                dict(freepars)
                if freepars is not None
                else {
                    "10^9As": 0.01,
                    "Omegam": 0.01,
                    "Omegab": 0.01,
                    "h": 0.01,
                    "ns": 0.01,
                }
            )
            if cosmoModel == "w0waCDM":
                resolved_freepars.setdefault("w0", 0.01)
                resolved_freepars.setdefault("wa", 0.01)
            from cosmicjellyfish.fishermatrix.cosmicfish import FisherMatrix

            resolved_fiducial = _resolve_fiducial(fiducialpars)
            fisher = FisherMatrix(
                options=resolved_options,
                specifications=dict(specifications or {}),
                observables=["GCsp"],
                freepars=resolved_freepars,
                fiducialpars=resolved_fiducial,
                surveyName=surveyName,
                cosmoModel=cosmoModel,
            )
            self._options = resolved_options
            self._specifications = dict(specifications or {})
            self._freepars = resolved_freepars
            self._fiducialpars = resolved_fiducial
            self._survey_name = surveyName
        else:
            self._options = dict(getattr(fisher, "settings", {}))
            self._specifications = dict(getattr(fisher, "specs", {}))
            self._freepars = dict(getattr(fisher, "freeparams", {}))
            self._fiducialpars = dict(getattr(fisher, "fiducialcosmopars", {}))
            self._survey_name = surveyName
            if self._options.get("code") != "jaxcosmo":
                raise ValueError("The pre-built FisherMatrix must use code='jaxcosmo'.")

        self.fisher = fisher
        self.fisher.observables = ["GCsp"]
        self.fisher.obs = ["GCsp"]
        self.fisher.obs_spectrum = ["g", "g"]
        self.fisher.set_pk_settings()

        self.base_fiducial = {
            **self.fisher.fiducialcosmopars,
            **self.fisher.Spectrobiaspars,
            **self.fisher.Spectrononlinpars,
            **self.fisher.PShotpars,
        }
        self.base = JaxSpectroObservable(
            self.fisher,
            self.base_fiducial,
            backend_parameters=getattr(self.fisher, "backend_parameters", None),
            observables=["GCsp"],
        )
        self.intermediate = IntermediateSpectroObservable(
            self.base, include_nonlinear=self.include_nonlinear
        )
        self.zmids = np.asarray(self.base.z)
        self._setup_covariance()

    @staticmethod
    def _default_final_names(cosmo_model: str) -> tuple[str, ...]:
        names = ["10^9As", "Omegam", "Omegab", "h", "ns"]
        if cosmo_model != "LCDM":
            names.extend(["w0", "wa"])
        return tuple(names)

    def _setup_covariance(self) -> None:
        from cosmicjellyfish.LSSsurvey import spectro_cov, spectro_obs

        self.pk_obs_fid = spectro_obs.ComputeGalSpectro(
            cosmopars=self.fisher.fiducialcosmopars,
            fiducial_cosmopars=self.fisher.fiducialcosmopars,
            spectrobiaspars=self.fisher.Spectrobiaspars,
            spectrononlinearpars=self.fisher.Spectrononlinpars,
            IMbiaspars=self.fisher.IMbiaspars,
            PShotpars=self.fisher.PShotpars,
            configuration=self.fisher,
        )
        self.pk_cov = spectro_cov.SpectroCov(
            self.fisher.fiducialcosmopars,
            fiducial_specobs=self.pk_obs_fid,
            bias_samples=["g", "g"],
            configuration=self.fisher,
        )
        if not np.allclose(self.zmids, self.pk_cov.inter_z_bin_mids):
            raise ValueError("Intermediate-bin redshifts do not match the covariance bins.")
        self.veff = np.asarray(
            [
                self.pk_cov.veff(float(zi), self.fisher.Pk_kmesh, self.fisher.Pk_mumesh)
                for zi in self.zmids
            ]
        )
        self.volume = np.asarray(
            [self.pk_cov.volume_survey(ibin) for ibin in range(self.base.nbins)]
        )
        self._quadrature_weights = self._build_quadrature_weights()

    def _build_quadrature_weights(self) -> np.ndarray:
        from scipy.integrate import simpson

        k = self.fisher.Pk_kmesh[0, :]
        mu = self.fisher.Pk_mumesh[:, 0]
        w_k = simpson(np.eye(k.size), x=k, axis=0)
        w_mu = simpson(np.eye(mu.size), x=mu, axis=0)
        return np.outer(w_mu, w_k)

    def fisher_intermediate(
        self,
        method: str = "ad",
        derivatives: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Assemble the Fisher matrix of the intermediate parameters.

        Parameters
        ----------
        method : {"ad", "fd"}, optional
            Derivative engine used when ``derivatives`` is not supplied.
        derivatives : dict, optional
            Pre-computed derivatives keyed by intermediate parameter name.

        Returns
        -------
        numpy.ndarray
            Fisher matrix in the order of :attr:`IntermediateSpectroObservable.parameter_names`.
        """
        if derivatives is None:
            if method == "ad":
                derivatives = self.intermediate.jacobian_ad()
            elif method == "fd":
                derivatives = self.intermediate.jacobian_fd()
            else:
                raise ValueError(f"Unknown derivative method '{method}'.")
        names = self.intermediate.parameter_names
        n_params = len(names)
        fisher = np.zeros((n_params, n_params), dtype=float)
        shape_indices = np.arange(len(_SHAPE_NAMES))
        local_start = len(_SHAPE_NAMES)
        stride = self.intermediate._local_stride

        for ibin in range(self.base.nbins):
            local_indices = local_start + ibin * stride + np.arange(stride)
            indices = np.concatenate([shape_indices, local_indices])
            block = np.stack([derivatives[names[index]][ibin].ravel() for index in indices])
            weight = (
                np.asarray(self.fisher.Pk_kmesh) ** 2
                * (2.0 * self.volume[ibin] * self.veff[ibin])
                * self._quadrature_weights
            )
            fisher[np.ix_(indices, indices)] += (block * weight.ravel()).dot(block.T)
        return fisher

    def nuisance_indices(self) -> list[int]:
        """Indices of the intermediate parameters marginalised by default."""
        names = self.intermediate.parameter_names
        indices = []
        for index, name in enumerate(names):
            if name.startswith(("lnbs8_", "Ps_")):
                indices.append(index)
            elif self.include_nonlinear and name.startswith(("lnsigmav_", "lnsigmap_")):
                indices.append(index)
        return indices

    @staticmethod
    def marginalize(fisher: np.ndarray, indices: Iterable[int]) -> tuple[np.ndarray, list[int]]:
        """Marginalise a Fisher matrix over ``indices`` via the Schur complement."""
        removed = sorted(set(int(index) for index in indices))
        keep = [index for index in range(fisher.shape[0]) if index not in removed]
        if not removed:
            return np.array(fisher, copy=True), keep
        f_keep = fisher[np.ix_(keep, keep)]
        f_cross = fisher[np.ix_(keep, removed)]
        f_removed = fisher[np.ix_(removed, removed)]
        reduced = f_keep - f_cross @ np.linalg.pinv(f_removed) @ f_cross.T
        return reduced, keep

    def _kept_names(self, keep: Sequence[int]) -> tuple[str, ...]:
        names = self.intermediate.parameter_names
        return tuple(names[index] for index in keep)

    def reduced_fisher(
        self, method: str = "ad", fisher: np.ndarray | None = None
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Intermediate Fisher matrix marginalised over ``b sigma8`` and shot noise."""
        if fisher is None:
            fisher = self.fisher_intermediate(method=method)
        reduced, keep = self.marginalize(fisher, self.nuisance_indices())
        return reduced, self._kept_names(keep)

    def _final_to_kept(
        self,
        theta: jnp.ndarray,
        kept_names: Sequence[str],
        final_names: Sequence[str],
    ) -> jnp.ndarray:
        cosmo = dict(self._base_cosmo_for_jacobian())
        for index, name in enumerate(final_names):
            cosmo[_FINAL_CANONICAL[name]] = theta[index] * _FINAL_SCALE[name]
        z = jnp.asarray(self.intermediate.z)
        table = self.base.theory.tables(cosmo, z, self.base.k_hmpc[:2], False)
        values = {
            "omegab": cosmo["Omegab"] * cosmo["h"] ** 2,
            "h": cosmo["h"],
            "omegam": cosmo["Omegam"] * cosmo["h"] ** 2,
            "ns": cosmo["ns"],
        }
        for ibin, zi in enumerate(self.intermediate.z):
            suffix = ibin + 1
            values[f"lnDA_{suffix}"] = jnp.log(table.chi[ibin] / (1.0 + zi))
            values[f"lnH_{suffix}"] = jnp.log(table.h_over_c[ibin])
            values[f"lnfs8_{suffix}"] = jnp.log(table.growth_rate[ibin] * table.sigma8[ibin])
        return jnp.stack([values[name] for name in kept_names])

    def _base_cosmo_for_jacobian(self) -> dict[str, float]:
        return dict(self.intermediate._base_cosmo)

    def final_fiducial_vector(self, final_names: Sequence[str] | None = None) -> np.ndarray:
        """Fiducial values of the final parameters in CosmicJellyfish spelling."""
        names = self.final_names if final_names is None else tuple(final_names)
        canonical = [_FINAL_CANONICAL.get(name) for name in names]
        duplicates = sorted(
            {key for key in canonical if key is not None and canonical.count(key) > 1}
        )
        if duplicates:
            aliases = {
                key: [name for name in names if _FINAL_CANONICAL.get(name) == key]
                for key in duplicates
            }
            raise ValueError(
                "Final parameter names cannot contain aliases of the same parameter; "
                f"use only one of each alias group: {aliases}."
            )
        values = []
        for name in names:
            if name not in _FINAL_CANONICAL:
                raise ValueError(f"Parameter '{name}' is not supported by the jaxcosmo backend.")
            values.append(
                self.intermediate._base_cosmo[_FINAL_CANONICAL[name]] / _FINAL_SCALE[name]
            )
        return np.array(values, dtype=float)

    def final_kept_jacobian(
        self,
        kept_names: Sequence[str],
        final_names: Sequence[str] | None = None,
        method: str = "ad",
        step: float = 1.0e-5,
    ) -> np.ndarray:
        """Jacobian ``d(kept intermediate)/d(final)``.

        Parameters
        ----------
        kept_names : sequence of str
            Intermediate parameters surviving the marginalisation, in output order.
        final_names : sequence of str, optional
            Final parameters.  Defaults to :attr:`final_names`.
        method : {"ad", "fd"}, optional
            Use JAX autodiff or central finite differences.
        step : float, optional
            Relative step for the finite-difference Jacobian.
        """
        names = self.final_names if final_names is None else tuple(final_names)
        reference = self.final_fiducial_vector(names)

        def function(theta):
            return self._final_to_kept(theta, kept_names, names)

        if method == "ad":
            jacobian = jax.jacfwd(function)(jnp.asarray(reference))
            return np.asarray(jacobian)
        if method == "fd":
            columns = []
            for index, value in enumerate(reference):
                delta = step * max(abs(value), 1.0)
                plus = np.array(reference, copy=True)
                minus = np.array(reference, copy=True)
                plus[index] += delta
                minus[index] -= delta
                columns.append(
                    np.asarray(
                        (function(jnp.asarray(plus)) - function(jnp.asarray(minus))) / (2.0 * delta)
                    )
                )
            return np.stack(columns, axis=1)
        raise ValueError(f"Unknown Jacobian method '{method}'.")

    def project(
        self,
        reduced: np.ndarray,
        kept_names: Sequence[str],
        final_names: Sequence[str] | None = None,
        jacobian_method: str = "ad",
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Project an intermediate Fisher matrix with the Fisher-space chain rule.

        The projection is delegated to
        :func:`cosmicjellyfish.analysis.fisher_operations.reparametrise`, which computes
        :math:`J^T F J` with :math:`J = \\partial(\\text{intermediate})/\\partial(\\text{final})`.
        """
        names = self.final_names if final_names is None else tuple(final_names)
        jacobian = self.final_kept_jacobian(kept_names, names, method=jacobian_method)
        projected = self.projected_fisher_matrix(reduced, kept_names, jacobian, names)
        return np.asarray(projected.fisher_matrix), jacobian, names

    def projected_fisher_matrix(
        self,
        reduced: np.ndarray,
        kept_names: Sequence[str],
        jacobian: np.ndarray,
        final_names: Sequence[str],
    ):
        """Wrap the reduced Fisher matrix and project it with ``reparametrise``.

        Returns
        -------
        cosmicjellyfish.analysis.fisher_matrix.fisher_matrix
            Fisher matrix of the final parameters, with names and fiducial values set.
        """
        from cosmicjellyfish.analysis import fisher_matrix as fm
        from cosmicjellyfish.analysis.fisher_operations import reparametrise

        fiducial = self.intermediate.fiducial_params()
        intermediate = fm.fisher_matrix(
            fisher_matrix=np.asarray(reduced),
            param_names=list(kept_names),
            param_names_latex=list(kept_names),
            fiducial=[fiducial[name] for name in kept_names],
            name="GCsp_intermediate_reduced",
        )
        return reparametrise(
            intermediate,
            jacobian,
            list(final_names),
            new_fiducial=list(self.final_fiducial_vector(final_names)),
            name="GCsp_intermediate_projected",
        )

    def binned_constraints(
        self, reduced: np.ndarray, kept_names: Sequence[str]
    ) -> dict[str, dict[str, np.ndarray]]:
        r"""Marginalised absolute errors on :math:`H(z_i)`, :math:`D_A(z_i)` and :math:`f\sigma_8(z_i)`.

        The reduced Fisher matrix constrains logarithms, so
        :math:`\sigma(X_i) = X_i\,\sigma(\ln X_i)` at linear order.

        Returns
        -------
        dict
            Keys ``"H"`` (km/s/Mpc), ``"DA"`` (Mpc) and ``"fs8"``; each holds ``z``,
            ``fiducial`` and ``sigma`` arrays over the redshift bins. An unconstrained
            parameter (a zero-information Fisher row, as for ``lnDA_i`` when AP effects
            are disabled) is reported with an infinite uncertainty.
        """
        speed_of_light = 299792.458
        reduced = np.asarray(reduced, dtype=float)
        diagonal = np.diag(reduced)
        scale = max(float(np.max(np.abs(diagonal))), 1.0)
        informative = np.flatnonzero(diagonal > np.finfo(float).eps * scale)
        covariance = np.zeros_like(reduced)
        if informative.size:
            covariance[np.ix_(informative, informative)] = np.linalg.pinv(
                reduced[np.ix_(informative, informative)]
            )
        sigma = dict(zip(kept_names, np.sqrt(np.maximum(np.diag(covariance), 0.0))))
        for index, name in enumerate(kept_names):
            if index not in informative:
                sigma[name] = np.inf
        fiducial = {
            "H": np.exp(self.intermediate.lnH_fid) * speed_of_light,
            "DA": np.exp(self.intermediate.lnDA_fid),
            "fs8": np.exp(self.intermediate.lnfs8_fid),
        }
        prefix = {"H": "lnH", "DA": "lnDA", "fs8": "lnfs8"}
        result = {}
        for key, root in prefix.items():
            log_sigma = np.array(
                [sigma[f"{root}_{ibin + 1}"] for ibin in range(self.intermediate.nbins)]
            )
            result[key] = {
                "z": np.asarray(self.intermediate.z),
                "fiducial": fiducial[key],
                "sigma": fiducial[key] * log_sigma,
            }
        return result

    def run(
        self,
        method: str = "ad",
        final_names: Sequence[str] | None = None,
        jacobian_method: str = "ad",
    ) -> dict:
        """Run the full intermediate-parameter forecast.

        Returns
        -------
        dict
            Contains the intermediate Fisher matrix, the reduced Fisher matrix and its
            parameter names, the final Fisher matrix, the Jacobian and the final names.
        """
        full = self.fisher_intermediate(method=method)
        reduced, kept_names = self.reduced_fisher(method=method, fisher=full)
        final, jacobian, names = self.project(
            reduced, kept_names, final_names=final_names, jacobian_method=jacobian_method
        )
        return {
            "fisher_intermediate": full,
            "fisher_reduced": reduced,
            "kept_names": kept_names,
            "jacobian": jacobian,
            "fisher_final": final,
            "final_names": names,
        }

    def direct_fisher(
        self,
        method: str = "ad",
        final_names: Sequence[str] | None = None,
        freepars: Mapping | None = None,
    ):
        """Run the validated direct Fisher recipe and marginalise to the final parameters."""
        from cosmicjellyfish.analysis.fisher_operations import marginalise
        from cosmicjellyfish.fishermatrix.cosmicfish import FisherMatrix
        from cosmicjellyfish.fishermatrix.jaxcosmo_derivatives import (
            JaxAutodiffProvider,
            JaxFiniteDifferenceProvider,
        )

        names = self.final_names if final_names is None else tuple(final_names)
        if "As" in names:
            raise ValueError(
                "Direct jaxcosmo Fisher comparisons support '10^9As', not 'As'; "
                "use '10^9As' in final_names."
            )
        provider: JaxAutodiffProvider | JaxFiniteDifferenceProvider
        if method == "ad":
            provider = JaxAutodiffProvider()
        elif method == "fd":
            provider = JaxFiniteDifferenceProvider()
        else:
            raise ValueError(f"Unknown direct derivative method '{method}'.")
        options = dict(self._options)
        options["code"] = "jaxcosmo"
        if freepars is None:
            freepars = dict(self._freepars)
            freepars = {name: step for name, step in freepars.items() if name in names}
        fisher = FisherMatrix(
            options=options,
            specifications=dict(self._specifications),
            observables=["GCsp"],
            freepars=dict(freepars),
            fiducialpars=dict(self._fiducialpars) if self._fiducialpars else None,
            surveyName=self._survey_name,
            cosmoModel=self.cosmo_model,
            derivative_provider=provider,
        )
        matrix = fisher.compute()
        return marginalise(matrix, list(names))

    @staticmethod
    def covariance(fisher: np.ndarray) -> np.ndarray:
        """Inverse of a Fisher matrix."""
        return np.linalg.pinv(fisher)

    def compare(
        self,
        method: str = "ad",
        final_names: Sequence[str] | None = None,
        jacobian_method: str = "ad",
        direct_method: str = "ad",
    ) -> dict:
        """Compare the intermediate projection with the validated direct Fisher recipe."""
        names = list(self.final_names if final_names is None else tuple(final_names))
        if "As" in names:
            raise ValueError(
                "Direct jaxcosmo Fisher comparisons support '10^9As', not 'As'; "
                "use '10^9As' in final_names."
            )
        result = self.run(method=method, final_names=final_names, jacobian_method=jacobian_method)
        direct = self.direct_fisher(method=direct_method, final_names=result["final_names"])
        names = list(result["final_names"])
        direct_matrix = np.asarray(direct.fisher_matrix)
        direct_names = list(direct.param_names)
        order = [direct_names.index(name) for name in names]
        direct_matrix = direct_matrix[np.ix_(order, order)]

        intermediate_cov = self.covariance(result["fisher_final"])
        direct_cov = self.covariance(direct_matrix)
        intermediate_sigma = np.sqrt(np.diag(intermediate_cov))
        direct_sigma = np.sqrt(np.diag(direct_cov))
        scale = np.maximum(np.abs(direct_matrix), 1.0e-30)
        return {
            **result,
            "fisher_direct": direct_matrix,
            "direct_names": tuple(direct_names[index] for index in order),
            "sigma_intermediate": intermediate_sigma,
            "sigma_direct": direct_sigma,
            "sigma_ratio": intermediate_sigma / direct_sigma,
            "max_abs_fisher_difference": float(
                np.max(np.abs(result["fisher_final"] - direct_matrix))
            ),
            "max_relative_fisher_difference": float(
                np.max(np.abs(result["fisher_final"] - direct_matrix) / scale)
            ),
        }
