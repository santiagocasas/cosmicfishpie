r"""JAX-native photometric (``GCph`` + ``WL``) angular power spectra for the ``jaxcosmo`` backend.

This module re-implements, with ``jax.numpy`` only, the Limber computation done by
:class:`cosmicfishpie.LSSsurvey.photo_obs.ComputeCls` so that the full chain

``(cosmology, bias, IA, photo-z) -> C_ell``

is traceable by ``jax.jit`` / ``jax.jacfwd``.  The existing NumPy/SciPy pipeline is left
untouched; this is a *parallel* observable used by
:class:`cosmicfishpie.fishermatrix.jaxcosmo_autodiff.JaxAutodiffProvider`.

Scope (explicitly rejected otherwise)
-------------------------------------
* observables: ``GCph`` and ``WL`` only (no spectroscopic, no CMB);
* total-matter tracer (``GCph_Tracer = "matter"``), no modified gravity;
* flat, massless-neutrino cosmology (see :mod:`cosmicfishpie.cosmology.jaxcosmo_theory`);
* intrinsic alignment model ``eNLA``; bias models ``sqrt``, ``binned``, ``binned_constant``,
  ``flagship``.

Differences with respect to ``ComputeCls`` (all numerical, not conceptual)
-------------------------------------------------------------------------
* ``P(k, z)`` is evaluated with Halofit directly on the Limber redshift grid and interpolated
  linearly in ``(ln k, ln P)`` (``ComputeCls`` uses a bicubic spline on a coarser ``z`` grid).
  Outside the tabulated ``k`` range the value is clamped, like the spline does.
* The luminosity-ratio table is interpolated with ``numpy.interp`` (clamped), whereas the
  original code uses ``scipy.interpolate.interp1d`` (raises outside the table).

Everything else (``z`` grid, ``ell`` grid, kernels, trapezoid rules, masks, dictionary keys and
their order) follows ``ComputeCls`` literally.
"""

from __future__ import annotations

import os
import re
import warnings
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erf
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "The jaxcosmo photometric observable requires jax. "
        "Install it with: pip install 'cosmicfishpie[jaxcosmo]'"
    ) from err

from cosmicfishpie.cosmology import jaxcosmo_theory as theory
from cosmicfishpie.cosmology.jaxcosmo_provider import resolve_parameters

#: CosmicFishPie names of the cosmological parameters that can be differentiated.
COSMO_FREE_NAMES: dict[str, str] = {
    "10^9As": "As",
    "Omegam": "Omegam",
    "Omegab": "Omegab",
    "h": "h",
    "ns": "ns",
    "w0": "w0",
    "wa": "wa",
}
#: Every key that describes the background cosmology in a CosmicFishPie fiducial dictionary.
_COSMO_KEYS = frozenset(
    {
        "10^9As", "As", "logAs", "ln_A_s_1e10", "sigma8", "Omegam", "omch2", "Omegab", "ombh2",
        "h", "H0", "ns", "n_s", "mnu", "m_nu", "w0", "w0_fld", "wa", "wa_fld", "Omegak", "omk",
        "Omega_k", "Neff", "N_eff", "tau", "Tcmb",
    }
)  # fmt: skip
PHOTO_KEYS = ("cb", "co", "fout", "zo", "zb", "sigma_o", "sigma_b")
IA_KEYS = ("AIA", "betaIA", "etaIA")
_STRING_KEYS = ("bias_model", "IA_model")
_PHOTO_OBS = ("GCph", "WL")


def _backward_trapezoid(y: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Cumulative trapezoid integral from ``z_k`` to the last grid point (``photo_obs`` rule)."""
    yr = y[::-1]
    seg = 0.5 * (yr[:-1] + yr[1:]) * dx
    cum = jnp.concatenate([jnp.zeros(1, dtype=y.dtype), jnp.cumsum(seg)])
    return cum[::-1]


class JaxPhotoObservable:
    """Differentiable photometric angular power spectra.

    Parameters
    ----------
    configuration : object
        Analysis configuration exposing ``settings``, ``specs`` and ``obs`` (a
        :class:`~cosmicfishpie.configs.context.AnalysisContext` or the legacy ``cfg`` module).
    fiducial : mapping
        Merged fiducial dictionary (cosmology + intrinsic alignment + bias + photo-z), the same
        object that ``PhotoCov`` passes as ``allparsfid``.
    backend_parameters : mapping, optional
        Content of ``jaxcosmo/default.yaml``.  Defaults to the one carried by ``configuration``.
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

        self.observables = [o for o in configuration.obs if o in _PHOTO_OBS]
        unsupported = [o for o in configuration.obs if o not in _PHOTO_OBS]
        if unsupported or not self.observables:
            raise NotImplementedError(
                "The jaxcosmo autodiff path supports only the photometric probes "
                f"'GCph' and 'WL'; got observables {tuple(configuration.obs)}."
            )
        if self.settings.get("GCph_Tracer", "matter") != "matter":
            raise NotImplementedError(
                "The jaxcosmo backend provides total-matter power only; "
                "GCph_Tracer='clustering' is unsupported."
            )

        if backend_parameters is None:
            backend_parameters = self._default_backend_parameters(configuration)
        self.backend_parameters = backend_parameters
        numerics = backend_parameters["NUMERICS"]
        if numerics.get("x64", True):
            theory.enable_x64()
        self.theory_config = theory.TheoryConfig.from_backend_parameters(backend_parameters)
        self.k_hmpc = np.logspace(
            np.log10(numerics["kmin_pk"]), np.log10(numerics["kmax_pk"]), int(numerics["k_samples"])
        )
        self.nonlinear = bool(self.settings.get("nonlinear", True))

        self._split_fiducial()
        self._setup_grids()
        self._setup_bins()
        self._jac_cache: dict[tuple, Any] = {}
        self._cls_jit = jax.jit(self._cls_from_values)

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
        cosmo_model = self.settings.get("cosmo_model", "LCDM")
        base = resolve_parameters(cosmo_in, cosmo_model)  # canonical names, floats
        self.bias_model = fid.get("bias_model", "binned")
        self.ia_model = fid.get("IA_model", "eNLA")
        if self.ia_model != "eNLA" and "WL" in self.observables:
            raise NotImplementedError("Only the 'eNLA' intrinsic alignment model is supported.")
        numeric = {
            k: float(v)
            for k, v in fid.items()
            if k not in _COSMO_KEYS and k not in _STRING_KEYS and np.ndim(v) == 0
        }
        self.cosmo_names = tuple(base)
        self.nuisance_names = tuple(numeric)
        self.base_values: dict[str, float] = {**base, **numeric}

    def _setup_grids(self) -> None:
        s, sp = self.settings, self.specs
        obs = self.observables
        lmax = [sp[f"lmax_{o}"] for o in obs]
        lmin = [sp[f"lmin_{o}"] for o in obs]
        ellmax, ellmin = max(lmax), min(lmin)
        accuracy = s["accuracy"]
        zsamp = int(round(200 * accuracy))
        ellsamp = (
            int(round(100 * accuracy)) if s["ell_sampling"] == "accuracy" else int(s["ell_sampling"])
        )
        self.ell = np.logspace(np.log10(ellmin), np.log10(ellmax + 10), num=ellsamp)
        z_min = min(sp["z_bins_GCph"][0], sp["z_bins_WL"][0])
        z_max = max(sp["z_bins_GCph"][-1], sp["z_bins_WL"][-1])
        self.z_max = float(z_max)
        self.z = np.linspace(z_min, z_max, zsamp)
        self.dz = float(np.mean(np.diff(self.z)))
        self.kmax = float(sp["kmax"])
        # trapezoid weights on the redshift grid
        w = np.full(zsamp, self.dz)
        w[0] *= 0.5
        w[-1] *= 0.5
        self.z_weights = w
        self.ell_mask = {
            o: (self.ell >= sp[f"lmin_{o}"]) & (self.ell <= sp[f"lmax_{o}"]) for o in obs
        }
        # coarse redshift grid used by Nuisance (bias, IA)
        self.z_nuis = np.linspace(
            min(sp["z_bins_WL"][0], sp["z_bins_GCph"][0]),
            max(sp["z_bins_WL"][-1], sp["z_bins_GCph"][-1]) + 1,
            int(50 * accuracy),
        )
        self.z_norm = np.linspace(0.0, self.z_max, 1000)
        self.dz_norm = self.z_max / 1000
        self.pivot_z_IA = float(s.get("pivot_z_IA", 0.0))
        self.lumratio_nuis = self._load_lumratio() if "WL" in self.observables else None

    def _load_lumratio(self) -> np.ndarray:
        path = os.path.join(self.settings["external_data_dir"], "lumratio_file.dat")
        try:
            table = np.loadtxt(path)
            return np.interp(self.z_nuis, table[:, 0], table[:, 1])
        except (FileNotFoundError, OSError):
            warnings.warn(
                "lumratio_file.dat not found: using <L>/L* = 1 for the intrinsic alignments.",
                stacklevel=2,
            )
            return np.ones_like(self.z_nuis)

    def _setup_bins(self) -> None:
        sp = self.specs
        self.binrange = {o: list(sp[f"binrange_{o}"]) for o in self.observables}
        self.z_bins = {o: np.asarray(sp[f"z_bins_{o}"], dtype=float) for o in self.observables}
        # stack order: (obs, bin) for obs in observables for bin in binrange[obs]
        self.stack = [(o, b) for o in self.observables for b in self.binrange[o]]
        index = {pair: n for n, pair in enumerate(self.stack)}
        keys: list[str] = []
        ia: list[int] = []
        ib: list[int] = []
        masks: list[np.ndarray] = []
        for o1, o2 in product(self.observables, self.observables):
            for b1, b2 in product(self.binrange[o1], self.binrange[o2]):
                keys.append(f"{o1} {b1}x{o2} {b2}")
                ia.append(index[(o1, b1)])
                ib.append(index[(o2, b2)])
                masks.append(self.ell_mask[o1] & self.ell_mask[o2])
        self.keys = tuple(keys)
        self._pair_a = np.asarray(ia)
        self._pair_b = np.asarray(ib)
        self._pair_mask = np.asarray(masks)
        if "GCph" in self.observables and self.bias_model == "binned":
            edges = self.z_bins["GCph"]
            idx = np.searchsorted(edges, self.z, side="right") - 1
            self._binned_idx = np.clip(idx, 0, self.binrange["GCph"][-1] - 1)

    # ------------------------------------------------------------------ physics (traced)
    def _ngal(self, z, obs: str, ibin: int, v: Mapping) -> jnp.ndarray:
        """Photo-z convolved, un-normalised ``n_i(z)`` (``GalaxyPhotoDist.ngal_photoz``)."""
        sp = self.specs
        edges = self.z_bins[obs]
        lo, hi = edges[ibin - 1], edges[ibin]
        cb, co, fout = v["cb"], v["co"], v["fout"]
        zo, zb, so, sb = v["zo"], v["zb"], v["sigma_o"], v["sigma_b"]
        dndz = (z / sp["z0_p"]) ** 2 * jnp.exp(-((z / sp["z0"]) ** sp["ngamma"]))
        r = np.sqrt(0.5)
        t1 = cb * fout * erf(r * (z - zo - co * lo) / (so * (1 + z)))
        t2 = -cb * fout * erf(r * (z - zo - co * hi) / (so * (1 + z)))
        t3 = co * (1 - fout) * erf(r * (z - zb - cb * lo) / (sb * (1 + z)))
        t4 = -co * (1 - fout) * erf(r * (z - zb - cb * hi) / (sb * (1 + z)))
        return dndz * (t1 + t2 + t3 + t4) / (2 * co * cb)

    def _ngal_norm(self, obs: str, ibin: int, v: Mapping) -> jnp.ndarray:
        norm = jnp.trapezoid(self._ngal(self.z_norm, obs, ibin, v), dx=self.dz_norm)
        return self._ngal(self.z, obs, ibin, v) / norm

    def _bias(self, ibin: int, v: Mapping) -> jnp.ndarray:
        model, z, zc = self.bias_model, self.z, self.z_nuis
        if model == "sqrt":
            return jnp.interp(z, zc, v["b0"] * jnp.sqrt(1 + zc))
        if model == "binned":
            table = jnp.stack([v[f"b{k}"] for k in self.binrange["GCph"]])
            return table[self._binned_idx]
        if model == "binned_constant":
            return v[f"b{ibin}"] * jnp.ones_like(z)
        if model == "flagship":
            b = v["A"] + v["B"] / (1 + jnp.exp(-v["C"] * (zc - v["D"])))
            return jnp.interp(z, zc, b)
        raise ValueError(f"Unknown galaxy bias model '{model}'.")

    def _ia_window(self, v: Mapping, omegam0, growth_nuis) -> jnp.ndarray:
        """eNLA intrinsic alignment amplitude on the Limber grid (``Nuisance.IA``)."""
        cia = 0.0134 * (1 + self.pivot_z_IA)
        fac = -v["AIA"] * cia * omegam0
        z_dep = (1 + self.z_nuis) / (1 + self.pivot_z_IA)
        win = fac * z_dep ** v["etaIA"] * (self.lumratio_nuis ** v["betaIA"] / growth_nuis)
        return jnp.interp(self.z, self.z_nuis, win)

    def _cls_from_values(self, v: Mapping) -> jnp.ndarray:
        """All requested ``C_ell`` as an array of shape ``(len(self.keys), len(self.ell))``."""
        cosmo = {name: v[name] for name in self.cosmo_names}
        z = jnp.asarray(self.z)
        tab = theory.compute_tables(cosmo, z, self.k_hmpc, self.theory_config, self.nonlinear)
        chi, hub = tab["chi"], tab["H_over_c"]
        h = cosmo["h"]
        hub0 = 100.0 * h / theory.SPEED_OF_LIGHT_KM_S
        omegam0 = cosmo["Omegam"]

        # Limber: P(kappa_l(z), z) / chi^2, zero above kmax.   shapes (nz, nell)
        kappa = (jnp.asarray(self.ell)[None, :] + 0.5) / chi[:, None]
        logk = jnp.log(jnp.asarray(self.k_hmpc) * h)
        logp = jnp.log(tab["Pnl"])
        p_kappa = jnp.exp(jax.vmap(jnp.interp, in_axes=(0, None, 0))(jnp.log(kappa), logk, logp))
        q = jnp.where(kappa < self.kmax, p_kappa / chi[:, None] ** 2, 0.0)

        growth_nuis = None
        if "WL" in self.observables:
            tab_nuis = theory.compute_tables(
                cosmo, jnp.asarray(self.z_nuis), self.k_hmpc[:2], self.theory_config, False
            )
            growth_nuis = tab_nuis["D"]
            ia_win = self._ia_window(v, omegam0, growth_nuis)
            prefac = 1.5 * hub0**2 * omegam0

        windows = []
        for obs, ibin in self.stack:
            n_i = self._ngal_norm(obs, ibin, v)
            if obs == "GCph":
                windows.append(n_i * hub * self._bias(ibin, v))
            else:
                i1 = _backward_trapezoid(n_i, self.dz)
                i2 = _backward_trapezoid(n_i / chi, self.dz)
                w_wl = prefac * (1 + z) * chi * (i1 - chi * i2)
                windows.append(w_wl + n_i * ia_win * hub)
        w_mat = jnp.stack(windows) * jnp.sqrt(jnp.asarray(self.z_weights) / hub)[None, :]
        cl_all = jnp.einsum("az,bz,zl->abl", w_mat, w_mat, q)
        cls = cl_all[self._pair_a, self._pair_b, :]
        return jnp.where(jnp.asarray(self._pair_mask), cls, 0.0)

    # ------------------------------------------------------------------ public API
    def canonical_name(self, name: str) -> str:
        """Map a CosmicFishPie parameter name to the name used inside the traced function."""
        if name in COSMO_FREE_NAMES:
            canon = COSMO_FREE_NAMES[name]
            if canon not in self.cosmo_names:  # pragma: no cover - defensive
                raise ValueError(f"Parameter '{name}' is not part of the jaxcosmo model.")
            return canon
        if name in self.nuisance_names:
            return name
        if name in _COSMO_KEYS:
            raise ValueError(
                f"Cosmological parameter '{name}' cannot be differentiated by the jaxcosmo "
                f"autodiff provider; supported: {sorted(COSMO_FREE_NAMES)}."
            )
        raise ValueError(f"Unknown parameter '{name}' for the jaxcosmo photometric observable.")

    def _values(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        values = dict(self.base_values)
        for name, val in (overrides or {}).items():
            values[self.canonical_name(name)] = val
        return values

    def as_dict(self, array: Any) -> dict[str, np.ndarray]:
        """Convert a ``(nkeys, nell)`` array into the CosmicFishPie ``C_ell`` dictionary."""
        out: dict[str, np.ndarray] = {"ells": np.array(self.ell)}
        arr = np.asarray(array)
        for n, key in enumerate(self.keys):
            out[key] = arr[n]
        return out

    def cls(self, overrides: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """Angular power spectra at the fiducial point (optionally with parameter overrides)."""
        return self.as_dict(self._cls_jit(self._values(overrides)))

    def jacobian(self, names: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
        """Exact derivatives ``dC_ell/dtheta`` for each name, as ``{name: Cl dictionary}``.

        Cosmological and nuisance parameters are differentiated in two separate forward-mode
        passes so that the (expensive) theory tables carry tangents only when needed.
        """
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
                return self._cls_from_values(values)

            self._jac_cache[group] = jax.jit(jax.jacfwd(fn))
        return self._jac_cache[group]


def is_bias_key(name: str) -> bool:
    """Whether ``name`` looks like a galaxy-bias nuisance parameter (``b1``, ``b0``, ...)."""
    return re.fullmatch(r"b\d+", name) is not None
