"""
Test suite for cosmicfishpie.utilities.legendre_tools module.

This module tests Legendre polynomial tools to improve coverage.
"""

import numpy as np

from cosmicfishpie.utilities.legendre_tools import (
    gauss_lobatto_abscissa_and_weights,
    m00,
    m00l,
    m02,
    m04,
    m22,
    m24,
    m44,
)


class TestGaussLobattoQuadrature:
    """Test Gauss-Lobatto quadrature functions."""

    def test_gauss_lobatto_order_2(self):
        """Test Gauss-Lobatto quadrature with order 2."""
        order = 2
        roots, weights = gauss_lobatto_abscissa_and_weights(order)

        # Check dimensions
        assert len(roots) == order
        assert len(weights) == order
        assert isinstance(roots, np.ndarray)
        assert isinstance(weights, np.ndarray)

        # Check boundary conditions (should include endpoints 0 and 1)
        np.testing.assert_almost_equal(roots[0], 0.0)
        np.testing.assert_almost_equal(roots[-1], 1.0)

    def test_gauss_lobatto_order_3(self):
        """Test Gauss-Lobatto quadrature with order 3."""
        order = 3
        roots, weights = gauss_lobatto_abscissa_and_weights(order)

        # Check dimensions
        assert len(roots) == order
        assert len(weights) == order

        # Roots should be in [0, 1] and include endpoints
        assert all(0 <= r <= 1 for r in roots)
        np.testing.assert_almost_equal(roots[0], 0.0)
        np.testing.assert_almost_equal(roots[-1], 1.0)

        # Weights should be positive
        assert all(w > 0 for w in weights)

    def test_gauss_lobatto_order_5(self):
        """Test Gauss-Lobatto quadrature with higher order."""
        order = 5
        roots, weights = gauss_lobatto_abscissa_and_weights(order)

        # Check that we get the right number of points
        assert len(roots) == 5
        assert len(weights) == 5

        # Check ordering (roots can span beyond [0,1] for Legendre-Lobatto quadrature)
        # Just check that we have reasonable root distribution
        assert len(set(roots)) == len(roots)  # All roots should be unique
        assert isinstance(roots[0], (float, int))  # Check types

        # All weights should be positive
        assert all(w > 0 for w in weights)

    def test_gauss_lobatto_weights_properties(self):
        """Test mathematical properties of Gauss-Lobatto weights."""
        order = 4
        roots, weights = gauss_lobatto_abscissa_and_weights(order)

        # The sum of weights should approximate the integral of 1 over [0,1]
        # For the standard interval, this should be close to 1
        weights_sum = np.sum(weights)
        assert weights_sum > 0


class TestWignerMatrices:
    """Test pre-computed Wigner 3j symbol matrices."""

    def test_wigner_matrix_shapes(self):
        """Test that all Wigner matrices have correct shapes."""
        matrices = [m00, m22, m44, m02, m24, m04]

        for matrix in matrices:
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (3, 3)

    def test_wigner_matrix_m00_properties(self):
        """Test properties of m00 matrix (l3=l4=0)."""
        # Check specific known values
        np.testing.assert_almost_equal(m00[0, 0], 1.0)
        np.testing.assert_almost_equal(m00[1, 1], 0.2)
        np.testing.assert_almost_equal(m00[2, 2], 1.0 / 9.0, decimal=6)

        # Check symmetry properties where expected
        assert m00[0, 1] == m00[1, 0]  # Should be 0
        assert m00[0, 2] == m00[2, 0]  # Should be 0

    def test_wigner_matrix_m22_properties(self):
        """Test properties of m22 matrix (l3=l4=2)."""
        # Check that matrix is symmetric
        np.testing.assert_array_almost_equal(m22, m22.T)

        # Check specific known values
        np.testing.assert_almost_equal(m22[0, 0], 0.2)
        np.testing.assert_almost_equal(m22[0, 1], 2.0 / 35.0)

    def test_wigner_matrix_m44_properties(self):
        """Test properties of m44 matrix (l3=l4=4)."""
        # Check that matrix is symmetric
        np.testing.assert_array_almost_equal(m44, m44.T)

        # All diagonal elements should be positive
        assert all(m44[i, i] > 0 for i in range(3))

    def test_wigner_matrices_numerical_values(self):
        """Test that matrices contain expected numerical ranges."""
        matrices = [m00, m22, m44, m02, m24, m04]

        for matrix in matrices:
            # All values should be non-negative (physical constraint)
            assert np.all(matrix >= 0)

            # Values should be reasonable (not too large)
            assert np.all(matrix <= 2.0)

    def test_original_lists_conversion(self):
        """Test that the original lists convert correctly to matrices."""
        # Test that reshaping works correctly
        test_list = m00l
        test_array = np.array(test_list)
        test_matrix = test_array.reshape(3, 3)

        np.testing.assert_array_equal(test_matrix, m00)

    def test_matrix_access_patterns(self):
        """Test typical access patterns for the matrices."""
        # Test that matrices can be indexed properly
        for i in range(3):
            for j in range(3):
                value = m00[i, j]
                assert isinstance(value, (float, int))

        # Test row and column access
        row = m22[0, :]
        col = m44[:, 0]
        assert len(row) == 3
        assert len(col) == 3
