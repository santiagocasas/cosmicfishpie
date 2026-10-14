"""CMB explicit-context ownership regressions."""

from types import SimpleNamespace

import numpy as np

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.CMBsurvey import CMB_obs
from cosmicfishpie.CMBsurvey.CMB_cov import CMBCov
from cosmicfishpie.fishermatrix import cosmicfish as cff


def test_cmb_computecls_uses_explicit_context_without_mutation(monkeypatch):
    calls = []

    class FakeCosmology:
        def __init__(self, cosmopars, input_type, *, configuration):
            calls.append((cosmopars, input_type, configuration))

        def cmb_power(self, ellmin, ellmax, obs1, obs2):
            return np.full(ellmax - ellmin, 7.0)

    context = SimpleNamespace(
        settings={"feedback": 0},
        specs={"lmin_CMB": 4, "lmax_CMB": 9},
        input_type="symbolic",
        obs=("CMB_T", "CMB_E", "WL"),
    )
    original_specs = dict(context.specs)
    monkeypatch.setattr(CMB_obs.cosmology, "cosmo_functions", FakeCosmology)

    cls = CMB_obs.ComputeCls({"Omegam": 0.31}, configuration=context)
    result = cls.computecls()

    assert calls == [({"Omegam": 0.31}, "symbolic", context)]
    assert cls.observables == ["CMB_T", "CMB_E"]
    assert np.array_equal(result["ells"], np.arange(4, 9))
    assert np.array_equal(result["CMB_TxCMB_E"], np.full(5, 7.0))
    assert context.specs == original_specs
    assert "ellmin" not in context.specs
    assert "ellmax" not in context.specs


def test_cmbcov_derivatives_remain_bound_to_context_a(monkeypatch):
    context_a = SimpleNamespace(
        freeparams={"x": 0.1},
        settings={"feedback": 0, "derivatives": "3PT"},
        obs=("CMB_T",),
        external=None,
    )
    monkeypatch.setattr(cfg, "freeparams", {"wrong": 1.0}, raising=False)
    monkeypatch.setattr(cfg, "settings", {"feedback": 0, "derivatives": "UNKNOWN"}, raising=False)
    monkeypatch.setattr(cfg, "obs", ("UNKNOWN",), raising=False)

    cov = object.__new__(CMBCov)
    cov.config = context_a
    cov.cosmopars = {"x": 2.0}
    cov.feed_lvl = 0
    cov.getcls = lambda allpars: {
        "ells": np.array([2.0]),
        "CMB_TxCMB_T": np.array([allpars["x"] ** 2]),
    }

    result = cov.compute_derivs()

    np.testing.assert_allclose(result["x"]["CMB_TxCMB_T"], [4.0])


def test_fishermatrix_exposes_cmb_derivs_dict(monkeypatch):
    derivatives = {"x": {"CMB_TxCMB_T": np.array([4.0])}}

    class StubCMB:
        def __init__(self, cosmopars, *, print_info_specs, configuration):
            assert cosmopars == {"x": 2.0}

        def compute_covmat(self):
            return {}, {}

        def compute_derivs(self, *, derivative_provider):
            return derivatives

    monkeypatch.setattr(cff.CMB_cov, "CMBCov", StubCMB)
    fisher = object.__new__(cff.FisherMatrix)
    fisher.observables = ["CMB_T"]
    fisher.fiducialcosmopars = {"x": 2.0}
    fisher.derivative_provider = None
    fisher.feed_lvl = 0
    fisher.parallel = False
    fisher.CMB_fishermatrix = lambda **kwargs: np.array([[1.0]])
    fisher.export_fisher = lambda matrix, **kwargs: matrix

    result = fisher.compute()

    assert fisher.derivs_dict is derivatives
    np.testing.assert_array_equal(result, [[1.0]])
