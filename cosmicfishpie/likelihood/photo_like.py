"""Photometric likelihood implementation for Cosmicfishpie."""

from __future__ import annotations

import logging
from copy import copy, deepcopy
from itertools import product
from typing import Any, Dict, Iterable, Optional

import numpy as np

from cosmicfishpie.LSSsurvey import photo_cov as pcov
from cosmicfishpie.LSSsurvey import photo_obs as pobs

from .base import Likelihood

logger = logging.getLogger("cosmicfishpie.likelihood.photo")


def _dict_with_updates(template: Dict[str, Any], pool: Dict[str, Any]) -> Dict[str, Any]:
    updated = deepcopy(template)
    for key in template:
        if key in pool:
            updated[key] = pool.pop(key)
    return updated


def _cells_from_cls(
    photo_cls: pobs.ComputeCls,
    photo_cov: pcov.PhotoCov,
    observables: Iterable[str],
) -> Dict[str, np.ndarray]:
    photo_cls.compute_all()

    observables = tuple(observables)
    ells = np.array(photo_cls.result["ells"], copy=True)
    data: Dict[str, np.ndarray] = {"ells": ells}

    if "WL" in observables:
        wl_bins = list(photo_cls.binrange_WL)
        n_wl = len(wl_bins)
        cell_ll = np.empty((len(ells), n_wl, n_wl), dtype=np.float64)
        for i, j in product(wl_bins, repeat=2):
            base = photo_cls.result[f"WL {i}xWL {j}"]
            noise = 0.0
            if hasattr(photo_cov, "ellipt_error") and hasattr(photo_cov, "ngalbin_WL"):
                noise = (
                    (photo_cov.ellipt_error**2.0) / photo_cov.ngalbin_WL[i - 1] if i == j else 0.0
                )
            cell_ll[:, i - 1, j - 1] = base + noise
        data["Cell_LL"] = cell_ll
    else:
        wl_bins = []

    if "GCph" in observables:
        gc_bins = list(photo_cls.binrange_GCph)
        n_gc = len(gc_bins)
        cell_gg = np.empty((len(ells), n_gc, n_gc), dtype=np.float64)
        for i, j in product(gc_bins, repeat=2):
            base = photo_cls.result[f"GCph {i}xGCph {j}"]
            noise = 0.0
            if hasattr(photo_cov, "ngalbin_GCph"):
                noise = (1.0 / photo_cov.ngalbin_GCph[i - 1]) if i == j else 0.0
            cell_gg[:, i - 1, j - 1] = base + noise
        data["Cell_GG"] = cell_gg
    else:
        gc_bins = []

    if "WL" in observables and "GCph" in observables:
        cell_gl = np.empty((len(ells), len(gc_bins), len(wl_bins)), dtype=np.float64)
        for i, j in product(gc_bins, wl_bins):
            cell_gl[:, i - 1, j - 1] = photo_cls.result[f"GCph {i}xWL {j}"]
        data["Cell_GL"] = cell_gl

    return data


def _chi2_per_obs(
    cell_fid: np.ndarray, cell_th: np.ndarray, ells: np.ndarray, dells: np.ndarray
) -> float:
    dfid = np.linalg.det(cell_fid)
    dth = np.linalg.det(cell_th)

    dmix = 0.0
    for idx in range(cell_fid.shape[-1]):
        mix = copy(cell_th)
        mix[:, idx, :] = cell_fid[:, idx, :]
        dmix += np.linalg.det(mix)

    integrand = (2 * ells + 1) * (dmix / dth + np.log(dth / dfid) - cell_fid.shape[-1])
    integrand = np.array(integrand, copy=False)
    result = np.sum(
        np.concatenate(
            [((integrand[1:] + integrand[:-1]) / 2) * dells[:-1], integrand[-1:] * dells[-1:]]
        )
    )
    return float(result)


class PhotometricLikelihood(Likelihood):
    """Likelihood built from photometric clusterings (WL / GCph)."""

    def __init__(
        self,
        *,
        cosmo_data,
        cosmo_theory=None,
        observables: Optional[Iterable[str]] = None,
        data_cells: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.observables = tuple(observables or cosmo_data.observables)
        self._preloaded_cells = (
            None if data_cells is None else {k: np.array(v) for k, v in data_cells.items()}
        )
        self.photo_cov_data: Optional[pcov.PhotoCov] = None
        self._ells = None
        self._ellmax_WL = None
        self._ellmax_GC = None
        self._ellmax_XC = None
        super().__init__(cosmo_data=cosmo_data, cosmo_theory=cosmo_theory, leg_flag="cells")

    def compute_data(self) -> Dict[str, np.ndarray]:
        if self._preloaded_cells is not None:
            self._ells = np.array(self._preloaded_cells.get("ells"), copy=True)
            return self._preloaded_cells

        photo_cls = getattr(self.cosmo_data, "photo_obs_fid", None)
        if photo_cls is None:
            photo_cls = pobs.ComputeCls(
                cosmopars=self.cosmo_data.fiducialcosmopars,
                photopars=self.cosmo_data.photopars,
                IApars=self.cosmo_data.IApars,
                biaspars=self.cosmo_data.photobiaspars,
                fiducial_cosmo=self.cosmo_data.fiducialcosmo,
            )

        self.photo_cov_data = getattr(self.cosmo_data, "photo_LSS", None)
        if self.photo_cov_data is None:
            self.photo_cov_data = pcov.PhotoCov(
                cosmopars=self.cosmo_data.fiducialcosmopars,
                photopars=self.cosmo_data.photopars,
                IApars=self.cosmo_data.IApars,
                biaspars=self.cosmo_data.photobiaspars,
                fiducial_Cls=photo_cls,
            )

        cells = _cells_from_cls(photo_cls, self.photo_cov_data, self.observables)
        self._ells = cells["ells"]
        self._ellmax_WL = self.cosmo_data.specs.get("lmax_WL")
        self._ellmax_GC = self.cosmo_data.specs.get("lmax_GCph")
        if self._ellmax_WL is not None and self._ellmax_GC is not None:
            self._ellmax_XC = min(self._ellmax_WL, self._ellmax_GC)
        else:
            self._ellmax_XC = None
        return cells

    def compute_theory(self, param_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        params = deepcopy(param_dict)

        cosmopars = _dict_with_updates(self.cosmo_theory.fiducialcosmopars, params)
        photopars = _dict_with_updates(self.cosmo_theory.photopars, params)
        IApars = _dict_with_updates(self.cosmo_theory.IApars, params)
        photobias = _dict_with_updates(self.cosmo_theory.photobiaspars, params)

        if params:
            logger.debug(
                "PhotometricLikelihood received unused parameters: %s",
                ", ".join(sorted(params.keys())),
            )

        photo_cls = pobs.ComputeCls(
            cosmopars=cosmopars,
            photopars=photopars,
            IApars=IApars,
            biaspars=photobias,
            fiducial_cosmo=None,
        )
        return _cells_from_cls(photo_cls, self.photo_cov_data, self.observables)

    def compute_chi2(self, theory_obs: Dict[str, np.ndarray]) -> float:
        if self._ells is None or self.photo_cov_data is None:
            raise RuntimeError("Data ells not initialised")

        ells = self._ells
        chi2 = 0.0
        fsky_wl = getattr(self.photo_cov_data, "fsky_WL", 1.0)
        fsky_gc = getattr(self.photo_cov_data, "fsky_GCph", 1.0)

        def ell_subset(limit: Optional[int]) -> tuple[int, np.ndarray, np.ndarray]:
            working = np.array(ells, copy=True)
            if limit is not None:
                insert_idx = np.searchsorted(working, limit)
                working = np.insert(working, insert_idx, limit)
            else:
                insert_idx = len(working)

            deltas = np.diff(working)
            if deltas.size < insert_idx:
                last = deltas[-1] if deltas.size else 1.0
                deltas = np.concatenate([deltas, [last]])
            else:
                deltas = deltas[:insert_idx]

            return insert_idx, working[:insert_idx], deltas

        if "Cell_LL" in self.data_obs and "Cell_LL" in theory_obs:
            n_wl, ell_wl, d_ell_wl = ell_subset(self._ellmax_WL)
            chi2 += fsky_wl * _chi2_per_obs(
                self.data_obs["Cell_LL"][:n_wl], theory_obs["Cell_LL"][:n_wl], ell_wl, d_ell_wl
            )

        if "Cell_GG" in self.data_obs and "Cell_GG" in theory_obs:
            n_gc, ell_gc, d_ell_gc = ell_subset(self._ellmax_GC)
            chi2 += fsky_gc * _chi2_per_obs(
                self.data_obs["Cell_GG"][:n_gc], theory_obs["Cell_GG"][:n_gc], ell_gc, d_ell_gc
            )

        if (
            "Cell_GL" in self.data_obs
            and "Cell_GL" in theory_obs
            and "Cell_GG" in theory_obs
            and "Cell_LL" in theory_obs
        ):
            n_xc, ell_xc, d_ell_xc = ell_subset(self._ellmax_XC)
            big_th = np.block(
                [
                    [
                        theory_obs["Cell_LL"][:n_xc],
                        np.transpose(theory_obs["Cell_GL"], (0, 2, 1))[:n_xc],
                    ],
                    [
                        theory_obs["Cell_GL"][:n_xc],
                        theory_obs["Cell_GG"][:n_xc],
                    ],
                ]
            )
            big_fid = np.block(
                [
                    [
                        self.data_obs["Cell_LL"][:n_xc],
                        np.transpose(self.data_obs["Cell_GL"], (0, 2, 1))[:n_xc],
                    ],
                    [
                        self.data_obs["Cell_GL"][:n_xc],
                        self.data_obs["Cell_GG"][:n_xc],
                    ],
                ]
            )
            chi2 += np.sqrt(
                self.photo_cov_data.fsky_WL * self.photo_cov_data.fsky_GCph
            ) * _chi2_per_obs(big_fid, big_th, ell_xc, d_ell_xc)
            chi2 += fsky_wl * _chi2_per_obs(
                self.data_obs["Cell_LL"][:n_xc], theory_obs["Cell_LL"][:n_xc], ell_xc, d_ell_xc
            )

        return float(chi2)
