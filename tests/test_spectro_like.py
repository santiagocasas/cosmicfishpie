import numpy as np
import pytest

from cosmicfishpie.likelihood.spectro_like import (
    compute_chi2_legendre,
    compute_wedge_chi2,
    legendre_Pgg,
    loglike,
)


class DummyFMLegendre:
    def __init__(self, mu_grid, ksamp):
        # mu_grid array over which to integrate
        self.Pk_mugrid = np.array(mu_grid)
        self.Pk_musamp = len(mu_grid)
        # number of k-samples (unused except for shape checks)
        self.Pk_ksamp = ksamp


def test_legendre_Pgg_constant_field():
    # Create dummy FisherMatrix with mu grid symmetric [-1, 1]
    n_mu = 101
    n_k = 3
    mu = np.linspace(-1.0, 1.0, n_mu)
    fm = DummyFMLegendre(mu_grid=mu, ksamp=n_k)

    # constant field C=1 for two redshift bins
    n_z = 2
    obs_Pgg = np.ones((n_z, n_mu, n_k))
    # Compute multipoles
    P_ell = legendre_Pgg(obs_Pgg, fm)
    # Expect P0 = (2*0+1) * ∫1 dµ = 2, P2 and P4 ~ 0
    # P_ell shape: (n_k, n_z, n_ell=3)
    assert P_ell.shape == (n_k, n_z, 3)
    # Monopole ~2, quadrupole and hexadecapole ~0
    assert np.allclose(P_ell[:, :, 0], 2.0, atol=1e-3)
    assert np.allclose(P_ell[:, :, 1:], 0.0, atol=1e-3)


def test_compute_chi2_legendre_simple():
    # Single k, single z, single ell
    P_data = np.array([[[2.0]]])
    P_th = np.array([[[1.0]]])
    inv_cov = np.array([[[[0.25]]]])  # variance = 4
    chi2 = compute_chi2_legendre(P_data, P_th, inv_cov)
    # (2-1)^2 * 0.25 = 0.25
    assert chi2 == pytest.approx(0.25)


class DummyFMWedge:
    def __init__(self, k_grid, mu_grid, vol_array):
        # k and mu grids for integration
        self.Pk_kgrid = np.array(k_grid)
        self.Pk_mugrid = np.array(mu_grid)
        # volume_survey_array indexed by redshift bins
        self.pk_cov = type("pc", (), {"volume_survey_array": np.array(vol_array)})


def test_compute_wedge_chi2_constant_simple():
    # Set up simple case with one redshift bin, two mu and two k samples
    k_grid = np.array([1.0, 2.0])
    mu_grid = np.array([0.0, 1.0])
    vol = np.array([2.0])  # one redshift bin
    fm = DummyFMWedge(k_grid=k_grid, mu_grid=mu_grid, vol_array=vol)

    # Observed = 1, Theory = 0
    data = np.ones((1, len(mu_grid), len(k_grid)))
    theory = np.zeros_like(data)
    chi2 = compute_wedge_chi2(data, theory, fm)
    # Analytical result: sum_z ∫_mu ∫_k delta^2 / sigma = 5/(4*pi^2)
    expected = 5.0 / (4.0 * np.pi**2)
    assert chi2 == pytest.approx(expected, rel=1e-3)


def test_loglike_no_inputs_returns_minus_inf():
    # loglike with no theory and no params gives -inf
    ll = loglike()
    assert ll == -np.inf
