from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.configs.context import AnalysisContext
from cosmicfishpie.LSSsurvey.photo_cov import PhotoCov
from cosmicfishpie.LSSsurvey.photo_obs import ComputeCls


def test_computecls_uses_explicit_context_without_mutation(photo_fisher_matrix):
    """Explicit contexts must drive photo setup without adding derived spec keys."""
    context = AnalysisContext.from_legacy_config(cfg)
    original_specs = dict(context.specs)

    cls = ComputeCls(
        {"Omegam": 0.3, "h": 0.7},
        context.photopars,
        context.IApars,
        context.photobiaspars,
        fiducial_cosmo=context.fiducialcosmo,
        configuration=context,
    )

    assert tuple(cls.observables) == context.obs
    assert cls.ellsamp == context.settings["ell_sampling"]
    assert cls.ellmin == min(context.specs["lmin_GCph"], context.specs["lmin_WL"])
    assert cls.ellmax == max(context.specs["lmax_GCph"], context.specs["lmax_WL"])
    assert dict(context.specs) == original_specs
    assert "ellmin" not in context.specs
    assert "ellmax" not in context.specs


def test_photocov_uses_explicit_context(photo_fisher_matrix):
    """Covariance setup derives its observable/bin state from the supplied context."""
    context = AnalysisContext.from_legacy_config(cfg)
    cls = ComputeCls(
        {"Omegam": 0.3, "h": 0.7},
        context.photopars,
        context.IApars,
        context.photobiaspars,
        fiducial_cosmo=context.fiducialcosmo,
        configuration=context,
    )
    cov = PhotoCov(
        {"Omegam": 0.3, "h": 0.7},
        context.photopars,
        context.IApars,
        context.photobiaspars,
        fiducial_Cls=cls,
        configuration=context,
    )

    assert tuple(cov.observables) == context.obs
    assert cov.binrange_WL == context.specs["binrange_WL"]
    assert cov.binrange_GCph == context.specs["binrange_GCph"]


def test_photocov_derivatives_remain_bound_to_context_a(monkeypatch):
    context_a = SimpleNamespace(
        freeparams={"x": 0.1},
        settings={"feedback": 0, "derivatives": "3PT"},
        obs=("WL",),
        external=None,
    )
    monkeypatch.setattr(cfg, "freeparams", {"wrong": 1.0}, raising=False)
    monkeypatch.setattr(cfg, "settings", {"feedback": 0, "derivatives": "UNKNOWN"}, raising=False)
    monkeypatch.setattr(cfg, "obs", ("UNKNOWN",), raising=False)

    cov = object.__new__(PhotoCov)
    cov.configuration = context_a
    cov.allparsfid = {"x": 2.0}
    cov.feed_lvl = 0
    cov.getcls = lambda pars: {
        "ells": np.array([10.0]),
        "WL 1xWL 1": np.array([pars["x"] ** 2]),
    }

    result = cov.compute_derivs()

    np.testing.assert_allclose(result["x"]["WL 1xWL 1"], [4.0])
