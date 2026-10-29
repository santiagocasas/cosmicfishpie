r"""Differentiable spectroscopic galaxy-clustering power spectrum for ``jaxcosmo``.

This is the spectroscopic counterpart of
:class:`~cosmicfishpie.LSSsurvey.jaxcosmo_photo.JaxPhotoObservable`: a ``jax.numpy``
reimplementation of :meth:`cosmicfishpie.LSSsurvey.spectro_obs.ComputeGalSpectro.lnpobs_gg`
that follows the legacy NumPy code as closely as possible, so that

* finite differences of this observable can be compared with the legacy pipeline
  (isolating the observable), and
* ``jax.jacfwd`` of this observable can be compared with finite differences of *the same*
  observable (isolating the derivative method).

The implemented configuration is the one CosmicFishPie uses by default for galaxy
clustering: Alcock-Paczynski rescaling, the Kaiser RSD term, the de-wiggling recipe with
Savitzky-Golay no-wiggle power, the Lorentzian fingers-of-god suppression, the spectroscopic
redshift-error damping and an optional extra shot noise.  The mildly nonlinear moments that
enter the de-wiggling and FoG terms are held at their fiducial-cosmology values, matching
``options["fix_cosmo_nl_terms"] = True``; intensity mapping, cross correlations, the
``sigma8``-expanded (IST:F) variant and free nonlinear-model parameters are not implemented
yet and raise :class:`NotImplementedError`.

Only the linear matter power spectrum enters this observable, exactly as in the legacy code
(``dewiggled_pdd`` uses ``cosmo.matpow`` with its default ``nonlinear=False``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import jax
    import jax.numpy as jnp
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "The jaxcosmo spectroscopic observable requires jax. "
        "Install it with: pip install 'cosmicfishpie[jaxcosmo]'"
    ) from err

from cosmicfishpie.LSSsurvey.jaxcosmo_photo import _COSMO_KEYS, _same_value
from cosmicfishpie.cosmology.jax_theory import get_jax_theory

_SPECTRO_OBS = ("GCsp",)


def _linear_interp_rows(x, xk, yk):
    """Piecewise-linear interpolation of ``yk`` (rows over ``xk``) at ``x`` (same shape).

    Reproduces ``scipy.interpolate.InterpolatedUnivariateSpline(..., k=1)``, including its
    linear extrapolation from the first/last interval.
    """
    idx = jnp.clip(jnp.searchsorted(xk, x, side="right") - 1, 0, xk.shape[0] - 2)
    x0 = xk[idx]
    x1 = xk[idx + 1]
    y0 = jnp.take_along_axis(yk, idx, axis=-1)
    y1 = jnp.take_along_axis(yk, idx + 1, axis=-1)
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def _savgol_operator(n_samples: int, window: int, order: int) -> np.ndarray:
    """Matrix ``A`` with ``A @ y == savgol_filter(y, window, order)`` (``mode='interp'``).

    The filter is linear in ``y``, so column ``i`` of ``A`` is the filter applied to the unit
    vector ``e_i``.  The edge treatment of ``mode='interp'`` is not symmetric, hence ``A`` is
    *not* equal to its transpose and the orientation matters at the boundaries.
    """
    from scipy.signal import savgol_filter

    return savgol_filter(np.eye(n_samples), window, order, axis=0)


class JaxSpectroObservable:
    """Differentiable galaxy-clustering :math:`\\ln P(k,\\mu)` in CosmicFishPie units.

    Parameters
    ----------
    configuration : object
        Analysis configuration exposing ``settings``, ``specs`` and ``obs``.
    fiducial : mapping
        Merged fiducial dictionary (cosmology + spectroscopic bias + shot noise + nonlinear
        modelling parameters), i.e. ``SpectroDerivs.fiducial_allpars``.
    backend_parameters : mapping, optional
        Content of ``jaxcosmo/default.yaml``.  Defaults to the one carried by
        ``configuration``.
    """

    def __init__(
        self,
        configuration: Any,
        fiducial: Mapping[str, Any],
        backend_parameters: Mapping[str, Any] | None = None,
    ):
        self.configuration = configuration
        self.settings = configuration.settings
        self.specs = configuration.specs
        self.fiducial = dict(fiducial)

        self.observables = [o for o in configuration.obs if o in _SPECTRO_OBS]
        unsupported = [o for o in configuration.obs if o not in _SPECTRO_OBS]
        if unsupported or not self.observables:
            raise NotImplementedError(
                "The jaxcosmo spectroscopic autodiff path supports only the 'GCsp' probe; "
                f"got observables {tuple(configuration.obs)}."
            )
        s = self.settings
        if s.get("GCsp_Tracer", "matter") != "matter":
            raise NotImplementedError(
                "The jaxcosmo backend provides total-matter power only; "
                "GCsp_Tracer='clustering' is unsupported."
            )
        if s.get("bfs8terms", False):
            raise NotImplementedError(
                "The jaxcosmo spectroscopic autodiff path does not implement the "
                "sigma8-expanded (bfs8terms) power spectrum."
            )
        if not s.get("fix_cosmo_nl_terms", True):
            raise NotImplementedError(
                "The jaxcosmo spectroscopic autodiff path holds the de-wiggling and "
                "fingers-of-god moments fixed to the fiducial cosmology; set "
                "options['fix_cosmo_nl_terms'] = True."
            )
        self.linear_switch = bool(s.get("GCsp_linear", False))
        self.fog_switch = bool(s.get("FoG_switch", True))
        self.ap = bool(s.get("AP_effect", True))
        self.kh_rescaling_bug = bool(s.get("kh_rescaling_bug", False))
        self.kh_bug_before_err = bool(s.get("kh_rescaling_beforespecerr_bug", False))

        if backend_parameters is None:
            backend_parameters = self._default_backend_parameters(configuration)
        self.backend_parameters = backend_parameters
        numerics = backend_parameters["NUMERICS"]
        self.theory = get_jax_theory(
            s.get("code", "jaxcosmo"), backend_parameters, cosmo_model=s.get("cosmo_model", "LCDM")
        )
        self.free_name_map = dict(self.theory.free_name_map)
        self.k_hmpc = np.logspace(
            np.log10(numerics["kmin_pk"]), np.log10(numerics["kmax_pk"]), int(numerics["k_samples"])
        )
        self.kmin_pk = float(numerics["kmin_pk"])
        self.kmax_pk = float(numerics["kmax_pk"])
        self.legacy_pk_clamp = bool(numerics.get("legacy_pk_clamp", False))
        self.h_fid = float(fiducial.get("h", 0.67))

        self._split_fiducial()
        self._setup_grids()
        self._setup_bins()
        self._setup_bias()
        self._setup_no_wiggle()
        self._setup_fiducial_terms()
        self._jac_cache: dict[tuple, Any] = {}
        self._lnp_jit = jax.jit(self._lnp_from_values)

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def _default_backend_parameters(configuration: Any) -> Mapping[str, Any]:
        params = getattr(configuration, "backend_parameters", None)
        if params:
            return params
        from cosmicfishpie.configs import config as cfg

        return cfg.boltzmann_jaxcosmopars

    def _split_fiducial(self) -> None:
        fid = self.fiducial
        cosmo_in = {k: v for k, v in fid.items() if k in _COSMO_KEYS}
        base = self.theory.resolve_cosmology(cosmo_in)
        self.bias_model = self.specs["sp_bias_model"]
        self.bias_sample = self.specs["sp_bias_sample"]
        self.bias_root = self.specs["sp_bias_root"]
        if self.bias_model not in ("linear", "linear_log", "linear_Qbias"):
            raise NotImplementedError(
                f"Spectroscopic bias model '{self.bias_model}' is not implemented."
            )
        numeric = {
            k: float(v)
            for k, v in fid.items()
            if k not in _COSMO_KEYS and np.ndim(v) == 0
        }
        self.cosmo_names = tuple(self.theory.cosmo_names)
        self.nuisance_names = tuple(numeric)
        self.base_values: dict[str, float] = {**base, **numeric}

    def _setup_grids(self) -> None:
        s, sp = self.settings, self.specs
        h_fid = float(self.fiducial.get("h", 0.67))
        kmin = sp["kmin_GCsp"] * h_fid
        kmax = sp["kmax_GCsp"] * h_fid
        self.k_1mpc = np.linspace(kmin, kmax, int(s["spectro_Pk_k_samples"]))
        self.mu = np.linspace(0.0, 1.0, int(s["spectro_Pk_mu_samples"]))
        self.km = self.k_1mpc[None, :] * np.ones((self.mu.size, 1))
        self.mum = self.mu[:, None] * np.ones((1, self.k_1mpc.size))
        self.shape_kmu = self.km.shape

    def _setup_bins(self) -> None:
        edges = []
        mids = []
        for _, interval in self.specs["z_bins_sp"].items():
            edges.append(interval)
            mids.append(0.5 * (interval[0] + interval[1]))
        self.bin_edges = np.array(sorted(edges))
        self.z = np.array(mids)
        self.nbins = self.z.size
        self.dz_err = float(self.specs["spec_sigma_dz"])
        self.dz_type = self.specs["spec_sigma_dz_type"]
        if self.dz_type not in ("constant", "z-dependent"):
            raise NotImplementedError(f"Unknown spectroscopic redshift-error type {self.dz_type!r}.")

    def _setup_bias(self) -> None:
        model = self.bias_model
        keys = []
        for i in range(1, self.nbins + 1):
            keys.append(f"{self.bias_root}{self.bias_sample}_{i}")
        missing = [k for k in keys if k not in self.fiducial]
        if missing:
            raise KeyError(f"Missing spectroscopic bias parameters: {missing}")
        self.bias_keys = tuple(keys)
        self.bias_pars = {k: float(self.fiducial[k]) for k in keys}
        self.bias_edges = self.bin_edges[:, 0]

    def _setup_no_wiggle(self) -> None:
        """Savitzky-Golay operator of ``cosmo_functions.nonwiggle_pow``.

        The smoothing always runs on ``savgol_internalsamples`` points in ``ln k``, so its
        weights are a constant matrix no matter where the grid ends; only the window length
        is fixed here (from the fiducial ``h``), exactly as the legacy code derives it.
        """
        s = self.settings
        h = float(self.fiducial.get("h", 0.67))
        self.savgol_kmin_factor = float(s["savgol_internalkmin"])
        self.savgol_samples = int(s["savgol_internalsamples"])
        # ``nonwiggle_pow`` spans ``h * savgol_internalkmin`` to ``h * kmax_pk * h`` (in
        # 1/Mpc, see ``_nowiggle_grid``); derive the window on the fiducial grid.
        kmin_loc = h * self.savgol_kmin_factor
        kmax_loc = h * self.kmax_pk * h
        dlnk = (np.log(kmax_loc) - np.log(kmin_loc)) / (self.savgol_samples - 1)
        window = int(round(float(s["savgol_width"]) / np.log(1.0 + dlnk)))
        order = int(s["savgol_polyorder"])
        if window % 2 == 0:
            window += 1
        self.savgol_operator = _savgol_operator(self.savgol_samples, window, order).astype(
            np.float64
        )
        self.savgol_window = window

    def _nowiggle_grid(self, h):
        """``ln k`` nodes (k in 1/Mpc) of the legacy no-wiggle smoothing, for a sample ``h``."""
        kmin_loc = h * self.savgol_kmin_factor
        # ``nonwiggle_pow`` rescales the provider's maximum wavenumber (already in 1/Mpc)
        # by the sample ``h`` again and treats the result as 1/Mpc; reproduce that quirk.
        kmax_loc = h * self.kmax_pk * h
        return jnp.linspace(jnp.log(kmin_loc), jnp.log(kmax_loc), self.savgol_samples)

    def _setup_fiducial_terms(self) -> None:
        """Fiducial background, de-wiggling moments and shot noise (parameter independent)."""
        params = {n: self.base_values[n] for n in self.cosmo_names}
        z = jnp.asarray(self.z)
        tab = self.theory.tables(params, z, self.k_hmpc[:2], False)
        self.hub_fid = np.asarray(tab.h_over_c)
        self.da_fid = np.asarray(tab.chi) / (1.0 + self.z)

        k_mom = np.logspace(np.log10(1e-3), np.log10(5.0), 1024)
        p_mom = np.asarray(self.theory.power(params, z, k_mom / params["h"], False))
        f_mom = np.asarray(tab.growth_rate)  # scale independent
        moments = []
        for moment in (0, 1, 2):
            integrand = p_mom * f_mom[:, None] ** moment
            moments.append(np.trapezoid(integrand, x=k_mom, axis=1) / (6.0 * np.pi**2))
        self.sigmap = np.sqrt(moments[2])
        self.theta_moments = np.stack(moments)
        self.extra_shot = float(self.settings.get("Pshot_nuisance_fiducial", 0.0))
        self.pshot_keys = tuple(k for k in self.fiducial if k.startswith("Ps_"))

    # ------------------------------------------------------------------ equations
    def _bias(self, v: Mapping) -> jnp.ndarray:
        raw = jnp.asarray([v[k] for k in self.bias_keys])
        binned = jnp.exp(raw) if self.bias_model == "linear_log" else raw
        edges = jnp.asarray(self.bias_edges)
        idx = jnp.clip(jnp.searchsorted(edges, jnp.asarray(self.z), side="right") - 1, 0, self.nbins - 1)
        return binned[idx]

    def _bias_kscale(self, k, v: Mapping):
        if self.bias_model != "linear_Qbias":
            return 1.0
        a1 = v.get("A1", 0.0)
        a2 = v.get("A2", 0.0)
        return (1.0 + k**2 * a2) / (1.0 + k * a1)

    def _lnp_from_values(self, v: Mapping) -> jnp.ndarray:
        """``ln P_gg(k, mu)`` for every redshift bin, shape ``(nbins, nmu, nk)``."""
        cosmo = {name: v[name] for name in self.cosmo_names}
        z = jnp.asarray(self.z)
        tab = self.theory.tables(cosmo, z, self.k_hmpc[:2], False)
        hub, chi = tab.h_over_c, tab.chi
        growth_rate = tab.growth_rate

        shape = (z.shape[0], *self.shape_kmu)  # (nz, nmu, nk)
        km = jnp.broadcast_to(jnp.asarray(self.km), shape)
        mum = jnp.broadcast_to(jnp.asarray(self.mum), shape)
        qpar = (self.hub_fid / hub)[:, None, None]

        # Optional unphysical h-rescaling of the wavenumbers (legacy ``kh_rescaling_bug``).
        # It is applied before or after the redshift-error damping, as in the legacy
        # ``observed_Pgg``: the damping sees ``km`` or ``k_h`` accordingly.
        if self.kh_rescaling_bug:
            k_h = km * (cosmo["h"] / self.h_fid)
            k_err = k_h if self.kh_bug_before_err else km
        else:
            k_h = km
            k_err = km

        # Spectroscopic redshift error.  The legacy ``spec_err_z`` always uses the
        # AP-corrected parallel wavenumber ``k mu / q_par`` (even if AP_effect is off).
        dz_err = self.dz_err * (1.0 + z) if self.dz_type == "z-dependent" else self.dz_err
        dz_err = dz_err[:, None, None] if jnp.ndim(dz_err) else dz_err
        err = jnp.exp(-0.5 * (dz_err / hub[:, None, None] * (k_err * mum / qpar)) ** 2)

        # Alcock-Paczynski rescaling (spectro_obs.ComputeGalSpectro.kmu_alc_pac).
        if self.ap:
            qper = ((chi / (1.0 + z)) / self.da_fid)[:, None, None]
            kpar = k_h * mum / qpar
            kper = k_h * jnp.sqrt(1.0 - mum**2) / qper
            kap = jnp.sqrt(kpar**2 + kper**2)
            muap = kpar / kap
            bao = 1.0 / (qper**2 * qpar)
        else:
            kap = k_h
            muap = mum
            bao = 1.0

        # Kaiser RSD term with the (possibly scale dependent) galaxy bias.
        bias = self._bias(v)
        bterm = bias[:, None, None] * self._bias_kscale(kap, v)
        kaiser = bterm + growth_rate[:, None, None] * muap**2

        # Fingers of god with the fiducial velocity dispersion.
        if self.linear_switch or not self.fog_switch:
            fog = 1.0
        else:
            fog = 1.0 / (1.0 + (kap * muap * self.sigmap[:, None, None]) ** 2)

        # De-wiggled linear power.
        if self.linear_switch:
            gmu = 0.0
        else:
            f0, f1, f2 = (self.theta_moments[i][:, None, None] for i in range(3))
            gmu = f0 + 2.0 * muap**2 * f1 + muap**2 * f2

        nz = z.shape[0]
        # Linear power at the AP-rescaled wavenumbers.  The legacy provider interpolates the
        # power on a table spanning [kmin_pk, kmax_pk] (h/Mpc) with a SciPy
        # ``RectBivariateSpline``, which *clamps* to the edge value outside that range.  Here
        # the theory is evaluated directly (exact, smooth) unless ``legacy_pk_clamp`` is set
        # in the backend NUMERICS block, which reproduces the clamp for parity studies.  Note
        # that the clamp makes the derivative discontinuous exactly where the fiducial grid
        # touches the table edge (first k at mu = 1).
        k_eval = kap
        k_hmpc = (k_eval / cosmo["h"]).reshape(nz, -1)
        if self.legacy_pk_clamp:
            k_hmpc = jnp.clip(k_hmpc, self.kmin_pk, self.kmax_pk)
        p_dd = self.theory.power(cosmo, z, k_hmpc, False).reshape(kap.shape)
        p_nw = self._nowiggle_power(cosmo, k_eval.reshape(nz, -1)).reshape(kap.shape)
        damp = jnp.exp(-gmu * kap**2)
        p_dw = p_dd * damp + p_nw * (1.0 - damp)

        pobs = bao * kaiser**2 * p_dw * fog * err**2 + self.extra_shot
        # Shot-noise parameters enter the observed power spectrum additively, exactly as the
        # analytic derivative in SpectroDerivs.exact_derivs assumes.
        if self.pshot_keys:
            pshot = jnp.asarray([v[k] for k in self.pshot_keys])
            pobs = pobs + pshot[:, None, None]
        return jnp.log(pobs)

    def _nowiggle_power(self, cosmo: Mapping, k_1mpc: jnp.ndarray) -> jnp.ndarray:
        """Savitzky-Golay smoothed linear power at ``k_1mpc`` (``cosmo_functions.nonwiggle_pow``).

        ``k_1mpc`` has shape ``(nz, n)`` in 1/Mpc and the result follows it.  The smoothing runs
        on the fixed fine grid the legacy code uses, so the Savitzky-Golay weights become a
        constant matrix and stay differentiable.
        """
        z = jnp.asarray(self.z)
        k = jnp.exp(self._nowiggle_grid(cosmo["h"]))
        p = self.theory.power(cosmo, z, k / cosmo["h"], False)
        # rows of ``log p`` are samples: (A @ y) for each row y is ``log p @ A.T``
        smooth = jnp.log(p) @ jnp.asarray(self.savgol_operator).T
        return _linear_interp_rows(k_1mpc, k, jnp.exp(smooth))

    # ------------------------------------------------------------------ public API
    def canonical_name(self, name: str) -> str:
        """Map a CosmicFishPie parameter name to the name used inside the traced function."""
        if name in self.free_name_map:
            canon = self.free_name_map[name]
            if canon not in self.cosmo_names:  # pragma: no cover - defensive
                raise ValueError(f"Parameter '{name}' is not part of the jaxcosmo model.")
            return canon
        if name in self.nuisance_names:
            return name
        if name in _COSMO_KEYS:
            raise ValueError(
                f"Cosmological parameter '{name}' cannot be differentiated by the jaxcosmo "
                f"autodiff provider; supported: {sorted(self.free_name_map)}."
            )
        raise ValueError(f"Unknown parameter '{name}' for the jaxcosmo spectroscopic observable.")

    def _values(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        values = dict(self.base_values)
        for name, val in (overrides or {}).items():
            if name in self.free_name_map:
                values[self.free_name_map[name]] = val
            elif name in self.nuisance_names:
                values[name] = val
            else:
                fid = self.fiducial.get(name)
                if fid is not None and _same_value(val, fid):
                    continue
                if name in _COSMO_KEYS:
                    raise ValueError(
                        f"Cosmological parameter '{name}' is fixed by the jaxcosmo model and "
                        f"cannot be varied; supported: {sorted(self.free_name_map)}."
                    )
                raise ValueError(
                    f"Unknown parameter '{name}' for the jaxcosmo spectroscopic observable."
                )
        return values

    def as_dict(self, array: Any) -> dict[str, np.ndarray]:
        """Convert a ``(nbins, nmu, nk)`` array into the CosmicFishPie ``P(k, mu)`` dictionary."""
        arr = np.asarray(array)
        out: dict[Any, np.ndarray] = {"z_bins": np.array(self.z)}
        for i in range(self.nbins):
            out[i] = arr[i]
        return out

    def lnp(self, overrides: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """``ln P_gg`` at the fiducial point (optionally with parameter overrides)."""
        return self.as_dict(self._lnp_jit(self._values(overrides)))

    def __call__(self, allpars: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Evaluate ``ln P_gg`` for a full CosmicFishPie parameter dictionary."""
        return self.lnp(allpars)

    def jacobian(self, names: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
        """Exact derivatives ``d ln P / d theta`` for each name."""
        names = tuple(names)
        canon = [self.canonical_name(n) for n in names]
        cosmo_idx = [i for i, c in enumerate(canon) if c in self.cosmo_names]
        nuis_idx = [i for i, c in enumerate(canon) if c not in self.cosmo_names]
        result: dict[str, dict[str, np.ndarray]] = {}
        for idx in (cosmo_idx, nuis_idx):
            if not idx:
                continue
            group = tuple(canon[i] for i in idx)
            jac = np.asarray(self._jacobian_fn(group)(self._group_point(group)))
            for col, i in enumerate(idx):
                result[names[i]] = self.as_dict(jac[..., col])
        return {n: result[n] for n in names}

    def _group_point(self, group: tuple[str, ...]):
        return jnp.asarray([self.base_values[g] for g in group], dtype=float)

    def _jacobian_fn(self, group: tuple[str, ...]):
        if group not in self._jac_cache:
            base = dict(self.base_values)

            def fn(vec):
                values = dict(base)
                for n, name in enumerate(group):
                    values[name] = vec[n]
                return self._lnp_from_values(values)

            self._jac_cache[group] = jax.jit(jax.jacfwd(fn))
        return self._jac_cache[group]
