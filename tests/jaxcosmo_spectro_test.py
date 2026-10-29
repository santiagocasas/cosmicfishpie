"""Regression tests for the JAX spectroscopic observable helpers (optional ``jaxcosmo`` extra)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_cosmo")

from scipy.signal import savgol_filter  # noqa: E402

from cosmicfishpie.LSSsurvey.jaxcosmo_spectro import (  # noqa: E402
    _linear_interp_rows,
    _savgol_operator,
)


@pytest.mark.parametrize("window,order", [(7, 3), (125, 3), (31, 2)])
def test_savgol_operator_matches_scipy_including_edges(window, order):
    rng = np.random.default_rng(1)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    # smooth trend plus noise: edge handling of mode='interp' is not symmetric
    y = np.sin(3 * x) + 0.1 * rng.standard_normal(n) + x**4
    operator = _savgol_operator(n, window, order)
    expected = savgol_filter(y, window, order)
    np.testing.assert_allclose(operator @ y, expected, rtol=1e-10, atol=1e-10)
    # the operator is not symmetric at the edges: a transposed application must differ
    assert np.max(np.abs(operator.T @ y - expected)) > 1e-6


def test_linear_interp_rows_matches_numpy_and_extrapolates_linearly():
    xk = np.array([0.0, 1.0, 2.0, 4.0])
    yk = np.array([[0.0, 1.0, 4.0, 16.0], [1.0, 1.0, 1.0, 1.0]])
    x = np.array([[0.5, 3.0, 5.0], [-1.0, 2.0, 6.0]])
    out = np.asarray(_linear_interp_rows(x, xk, yk))
    # inside the table: identical to numpy.interp
    assert out[0, 0] == pytest.approx(np.interp(0.5, xk, yk[0]))
    assert out[0, 1] == pytest.approx(np.interp(3.0, xk, yk[0]))
    # outside: continue the first/last segment (slope 6 on [2, 4])
    assert out[0, 2] == pytest.approx(16.0 + 6.0 * 1.0)
    assert out[1, 0] == pytest.approx(1.0)
