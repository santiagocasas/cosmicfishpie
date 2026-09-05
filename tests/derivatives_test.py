from types import SimpleNamespace

import numpy as np
import pytest

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.fishermatrix.derivatives import compute_derivatives, derivatives


def test_invalid_derivative_type_raises():
    with pytest.raises(ValueError):
        derivatives(
            observable=lambda p: p["x"],
            fiducial={"x": 1.0},
            freeparams={"x": 0.1},
            observables_type=["plain"],
            derivatives_type="UNKNOWN",
            external_settings={},
            feed_lvl=0,
        )


def test_der_3pt_stencil_scalar():
    der = derivatives(
        observable=lambda p: p["x"],
        fiducial={"x": 1.0},
        freeparams={"x": 0.1},
        observables_type=["plain"],
        derivatives_type="3PT",
        external_settings={},
        feed_lvl=0,
    )
    assert der.der_3pt_stencil(3.0, 1.0, 0.5) == pytest.approx(2.0)


def test_derivative_3pt_plain_linear_exact():
    # f(x)=2x+1 -> f'(x)=2 exactly
    d = derivatives(
        observable=lambda p: 2.0 * p["x"] + 1.0,
        fiducial={"x": 1.5},
        freeparams={"x": 0.1},
        observables_type=["plain"],
        derivatives_type="3PT",
        external_settings={},
        feed_lvl=0,
    )
    assert d.result["x"] == pytest.approx(2.0, rel=1e-12)


def test_derivative_3pt_cmb_branch_keeps_ells():
    def cmb_obs(pars):
        x = pars["x"]
        return {
            "ells": np.array([2, 3, 4], dtype=float),
            "CMB_TxCMB_T": np.array([1.0, 2.0, 3.0], dtype=float) * x,
            "CMB_ExCMB_E": np.array([0.5, 1.5, 2.5], dtype=float) * x,
        }

    d = derivatives(
        observable=cmb_obs,
        fiducial={"x": 1.0},
        freeparams={"x": 0.1},
        observables_type=["CMB_T", "CMB_E"],
        derivatives_type="3PT",
        external_settings={},
        feed_lvl=0,
    )

    out = d.result["x"]
    np.testing.assert_allclose(out["ells"], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(out["CMB_TxCMB_T"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out["CMB_ExCMB_E"], np.array([0.5, 1.5, 2.5]))


def test_derivative_3pt_special_derivative_short_circuit():
    def special(par):
        if par == "x":
            return {"analytical": 123.0}
        return None

    d = derivatives(
        observable=lambda p: p["x"],
        fiducial={"x": 1.0},
        freeparams={"x": 0.1},
        observables_type=["plain"],
        derivatives_type="3PT",
        special_deriv_function=special,
        external_settings={},
        feed_lvl=0,
    )
    assert d.result["x"] == {"analytical": 123.0}


def test_derivative_3pt_unsupported_observable_raises():
    with pytest.raises(ValueError, match="unsupported observables_type"):
        derivatives(
            observable=lambda p: p["x"],
            fiducial={"x": 1.0},
            freeparams={"x": 0.1},
            observables_type=["UNKNOWN"],
            derivatives_type="3PT",
            external_settings={},
            feed_lvl=0,
        )


def test_derivative_forward_4pt_plain_quadratic_smoke():
    # f(x)=x^2, derivative at x=1 is 2
    d = derivatives(
        observable=lambda p: p["x"] ** 2,
        fiducial={"x": 1.0},
        freeparams={"x": 0.05},
        observables_type=["plain"],
        derivatives_type="4PT_FWD",
        external_settings={},
        feed_lvl=0,
    )
    assert d.result["x"] == pytest.approx(2.0, rel=1e-6)


def test_derivative_stem_unsupported_for_spectro_raises():
    def spectro_obs(pars):
        x = pars["x"]
        return {"z_bins": np.array([0.5, 1.0]), "Pgg": np.array([x, 2.0 * x])}

    with pytest.raises(ValueError, match="STEM derivative not availabe"):
        derivatives(
            observable=spectro_obs,
            fiducial={"x": 1.0},
            freeparams={"x": 0.05},
            observables_type=["GCsp"],
            derivatives_type="STEM",
            external_settings={"eps_values": [0.01, 0.02, 0.03]},
            feed_lvl=0,
        )


def test_explicit_configuration_owns_all_default_derivative_inputs(monkeypatch):
    context_a = SimpleNamespace(
        freeparams={"x": 0.25},
        settings={"feedback": 0, "derivatives": "3PT"},
        obs=("plain",),
        external={"owner": "A"},
    )
    monkeypatch.setattr(cfg, "freeparams", {"wrong": 10.0}, raising=False)
    monkeypatch.setattr(cfg, "settings", {"feedback": 99, "derivatives": "UNKNOWN"}, raising=False)
    monkeypatch.setattr(cfg, "obs", ("UNKNOWN",), raising=False)
    monkeypatch.setattr(cfg, "external", {"owner": "B"}, raising=False)

    result = derivatives(
        observable=lambda pars: pars["x"] ** 2,
        fiducial={"x": 2.0},
        configuration=context_a,
    )

    assert result.freeparams == {"x": 0.25}
    assert result.feed_lvl == 0
    assert result.observables_type == ("plain",)
    assert result.external_settings == {"owner": "A"}
    assert result.derivatives_type == "3PT"
    assert result.result["x"] == pytest.approx(4.0)


def test_compute_derivatives_accepts_backend_neutral_provider():
    context = SimpleNamespace(
        freeparams={"x": 0.1},
        settings={"feedback": 0, "derivatives": "3PT"},
        obs=("plain",),
        external=None,
    )
    captured = []

    class FakeAutodiffProvider:
        def compute(self, request):
            captured.append(request)
            return {"x": "jacobian"}

    result = compute_derivatives(
        observable=lambda pars: pars["x"],
        fiducial={"x": 1.0},
        configuration=context,
        provider=FakeAutodiffProvider(),
    )

    assert result == {"x": "jacobian"}
    assert captured[0].configuration is context
    assert captured[0].freeparams == {"x": 0.1}
    assert captured[0].method == "3PT"
