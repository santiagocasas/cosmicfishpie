from __future__ import annotations

import numpy as np
import pytest

from cosmicjellyfish.analysis import fisher_matrix as fm
from cosmicjellyfish.analysis.fisher_operations import (
    jacobian_from_function,
    marginalise,
    reparametrise,
)


def _toy_fisher():
    matrix = np.array(
        [
            [40.0, 3.0, -2.0],
            [3.0, 25.0, 1.5],
            [-2.0, 1.5, 10.0],
        ]
    )
    return fm.fisher_matrix(
        fisher_matrix=matrix,
        param_names=["a", "b", "c"],
        param_names_latex=["a", "b", "c"],
        fiducial=[1.0, 2.0, 0.5],
    )


def test_linear_reparametrisation_matches_chain_rule():
    fisher = _toy_fisher()
    jacobian = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    new = reparametrise(fisher, jacobian, ["x", "y"], new_fiducial=[0.1, 0.2])
    expected = jacobian.T @ fisher.get_fisher_matrix() @ jacobian
    np.testing.assert_allclose(new.get_fisher_matrix(), expected)
    assert new.get_param_names() == ["x", "y"]
    np.testing.assert_allclose(new.get_param_fiducial(), [0.1, 0.2])


def test_invertible_round_trip_recovers_input():
    fisher = _toy_fisher()
    jacobian = np.array([[2.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, -1.0, 4.0]])
    forward = reparametrise(fisher, jacobian, ["x", "y", "z"])
    back = reparametrise(forward, np.linalg.inv(jacobian), ["a", "b", "c"])
    np.testing.assert_allclose(back.get_fisher_matrix(), fisher.get_fisher_matrix())


def test_invertible_map_propagates_covariance():
    fisher = _toy_fisher()
    jacobian = np.array([[2.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, -1.0, 4.0]])
    new = reparametrise(fisher, jacobian, ["x", "y", "z"])
    inverse = np.linalg.inv(jacobian)
    expected_cov = inverse @ fisher.get_fisher_inverse() @ inverse.T
    np.testing.assert_allclose(new.get_fisher_inverse(), expected_cov, rtol=1e-10)


def test_reparametrise_after_marginalisation():
    fisher = _toy_fisher()
    reduced = marginalise(fisher, ["a", "b"])
    jacobian = np.array([[1.0], [3.0]])
    new = reparametrise(reduced, jacobian, ["x"])
    expected = jacobian.T @ reduced.get_fisher_matrix() @ jacobian
    np.testing.assert_allclose(new.get_fisher_matrix(), expected)


def test_jacobian_from_function_fd():
    def function(x):
        return np.array([x[0] * x[1], x[1] ** 2, np.exp(x[0])])

    point = np.array([0.3, 1.7])
    expected = np.array([[point[1], point[0]], [0.0, 2.0 * point[1]], [np.exp(point[0]), 0.0]])
    np.testing.assert_allclose(jacobian_from_function(function, point), expected, rtol=1e-8)


def test_jacobian_from_function_jax_matches_fd():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def function(x):
        return jnp.stack([x[0] * x[1], x[1] ** 2, jnp.exp(x[0])])

    point = np.array([0.3, 1.7])
    np.testing.assert_allclose(
        jacobian_from_function(function, point, method="jax"),
        jacobian_from_function(lambda x: np.asarray(function(jnp.asarray(x))), point),
        rtol=1e-7,
    )


def test_reparametrise_validates_shapes():
    fisher = _toy_fisher()
    with pytest.raises(ValueError):
        reparametrise(fisher, np.ones((2, 2)), ["x", "y"])
    with pytest.raises(ValueError):
        reparametrise(fisher, np.ones((3, 2)), ["x"])
    with pytest.raises(ValueError):
        reparametrise(fisher.get_fisher_matrix(), np.ones((3, 2)), ["x", "y"])
    with pytest.raises(ValueError):
        jacobian_from_function(lambda x: x, [1.0], method="bogus")


@pytest.mark.parametrize("step", [0.0, np.nan, np.inf, -1.0e-6])
def test_jacobian_from_function_rejects_invalid_steps(step):
    with pytest.raises(ValueError, match="step must be finite and positive"):
        jacobian_from_function(lambda x: x**2, [1.0], step=step)


def test_jacobian_from_function_rejects_non_finite_columns():
    with pytest.raises(ValueError, match="non-finite finite-difference derivative"):
        jacobian_from_function(lambda x: np.log(x), [0.5, 1.0], step=1.0)


def test_reparametrise_rejects_duplicate_names():
    with pytest.raises(ValueError, match="names must be unique"):
        reparametrise(_toy_fisher(), np.ones((3, 2)), ["x", "x"])
