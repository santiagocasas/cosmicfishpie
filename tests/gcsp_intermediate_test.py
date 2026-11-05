from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("jax_cosmo")

from cosmicjellyfish.LSSsurvey.gcsp_intermediate import (  # noqa: E402
    GCspIntermediateForecast,
    IntermediateSpectroObservable,
    _resolve_fiducial,
)

OPTIONS = {
    "code": "jaxcosmo",
    "nonlinear": True,
    "feedback": 0,
    "spectro_Pk_k_samples": 16,
    "spectro_Pk_mu_samples": 8,
}
SPECIFICATIONS = {
    "kmin_GCsp": 0.01,
    "kmax_GCsp": 0.20,
    "spec_sigma_dz": 0.001,
}


@pytest.fixture(scope="module")
def forecast(tmp_path_factory):
    results_dir = tmp_path_factory.mktemp("intermediate_results")
    options = dict(OPTIONS)
    options["results_dir"] = str(results_dir)
    options["outroot"] = "intermediate_test"
    return GCspIntermediateForecast(
        options=options,
        specifications=SPECIFICATIONS,
        cosmoModel="LCDM",
    )


def test_observable_matches_base(forecast):
    base = forecast.base.lnp(forecast.base.fiducial)
    intermediate = forecast.intermediate.lnp()
    for ibin in range(forecast.base.nbins):
        np.testing.assert_allclose(intermediate[ibin], base[ibin], rtol=0.0, atol=1.0e-12)


def test_ad_and_fd_derivatives_agree(forecast):
    names = forecast.intermediate.parameter_names
    ad = forecast.intermediate.jacobian_ad(names)
    fd = forecast.intermediate.jacobian_fd(names)
    for name in names:
        scale = max(np.max(np.abs(ad[name])), 1.0e-30)
        error = np.max(np.abs(ad[name] - fd[name])) / scale
        assert error < 1.0e-6, f"{name}: {error:.3e}"


def test_intermediate_fisher_is_symmetric_and_positive(forecast):
    fisher = forecast.fisher_intermediate(method="ad")
    np.testing.assert_allclose(fisher, fisher.T, rtol=1.0e-12, atol=0.0)
    assert np.all(np.diag(fisher) > 0.0)


def test_schur_marginalisation_matches_explicit_formula():
    fisher = np.array(
        [
            [4.0, 1.0, 0.5],
            [1.0, 3.0, 0.2],
            [0.5, 0.2, 2.0],
        ]
    )
    reduced, keep = GCspIntermediateForecast.marginalize(fisher, [2])
    expected = fisher[:2, :2] - np.outer(fisher[:2, 2], fisher[2, :2]) / fisher[2, 2]
    np.testing.assert_allclose(reduced, expected)
    assert keep == [0, 1]


def test_projection_matches_direct_ad(forecast):
    comparison = forecast.compare(method="ad", direct_method="ad")
    assert np.max(np.abs(comparison["sigma_ratio"] - 1.0)) < 1.0e-6
    assert comparison["max_relative_fisher_difference"] < 1.0e-10


def test_fd_projection_matches_ad(forecast):
    ad = forecast.run(method="ad", jacobian_method="ad")
    fd = forecast.run(method="fd", jacobian_method="fd")
    sigma_ad = np.sqrt(np.diag(np.linalg.pinv(ad["fisher_final"])))
    sigma_fd = np.sqrt(np.diag(np.linalg.pinv(fd["fisher_final"])))
    np.testing.assert_allclose(sigma_fd, sigma_ad, rtol=1.0e-4)
    fisher_scale = np.max(np.abs(ad["fisher_final"]))
    np.testing.assert_allclose(
        fd["fisher_final"], ad["fisher_final"], rtol=0.0, atol=1.0e-6 * fisher_scale
    )


def test_final_parameter_guard(forecast):
    with pytest.raises(ValueError):
        forecast.final_fiducial_vector(["sigma8"])
    with pytest.raises(ValueError):
        forecast.fisher_intermediate(method="not-a-method")
    with pytest.raises(ValueError, match="use only one"):
        forecast.final_fiducial_vector(["As", "10^9As"])
    with pytest.raises(ValueError, match="use '10"):
        forecast.compare(final_names=["As"])


def test_resolve_fiducial_handles_string_values():
    assert _resolve_fiducial({"w0": "-0.9"})["w0"] == -0.9
    with pytest.raises(ValueError, match="'w0' must be numeric"):
        _resolve_fiducial({"w0": "not-a-number"})
    assert "dark_energy_model" not in _resolve_fiducial({"dark_energy_model": "CPL"})


def test_nonlinear_intermediate_parameters(forecast):
    observable = IntermediateSpectroObservable(forecast.base, include_nonlinear=True)
    names = observable.parameter_names
    assert "lnsigmav_1" in names
    assert "lnsigmap_1" in names
    fiducial = observable.fiducial_params()
    assert all(np.isfinite(value) for value in fiducial.values())
    assert np.all(np.isfinite(observable.lnp_vector(observable.fiducial_vector())))
    derivatives = observable.jacobian_ad(["lnsigmav_1", "lnsigmap_1"])
    for name, derivative in derivatives.items():
        assert np.all(np.isfinite(derivative))
        assert np.any(derivative != 0.0), name


def test_projection_uses_reparametrise_and_binned_constraints(forecast):
    result = forecast.run(method="ad")
    projected = forecast.projected_fisher_matrix(
        result["fisher_reduced"], result["kept_names"], result["jacobian"], result["final_names"]
    )
    assert list(projected.param_names) == list(result["final_names"])
    np.testing.assert_allclose(projected.fisher_matrix, result["fisher_final"], rtol=1.0e-12)
    np.testing.assert_allclose(
        projected.param_fiducial, forecast.final_fiducial_vector(), rtol=1.0e-12
    )
    binned = forecast.binned_constraints(result["fisher_reduced"], result["kept_names"])
    for key in ("H", "DA", "fs8"):
        assert binned[key]["sigma"].shape == (forecast.base.nbins,)
        assert np.all(binned[key]["sigma"] > 0.0)
        assert np.all(binned[key]["sigma"] < binned[key]["fiducial"])


def test_binned_constraints_reports_unconstrained_parameters(forecast):
    result = forecast.run(method="ad")
    reduced = np.array(result["fisher_reduced"], copy=True)
    index = result["kept_names"].index("lnDA_1")
    reduced[index, :] = 0.0
    reduced[:, index] = 0.0
    binned = forecast.binned_constraints(reduced, result["kept_names"])
    assert np.isinf(binned["DA"]["sigma"][0])
    assert np.all(np.isfinite(binned["H"]["sigma"]))
    assert np.all(np.isfinite(binned["fs8"]["sigma"]))
