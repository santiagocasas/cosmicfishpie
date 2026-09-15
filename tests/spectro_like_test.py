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


class DummyPkCov:
    def __init__(self, vol_array):
        self._vol = np.array(vol_array, dtype=float)
        # SpectroCov exposes one bin midpoint per redshift bin
        self.global_z_bin_mids = np.arange(len(self._vol), dtype=float)

    def volume_survey(self, ibin):
        return self._vol[ibin]


class DummyFMWedge:
    def __init__(self, k_grid, mu_grid, vol_array):
        # k and mu grids for integration
        self.Pk_kgrid = np.array(k_grid)
        self.Pk_mugrid = np.array(mu_grid)
        self.pk_cov = DummyPkCov(vol_array)


def _wedge_fixture():
    # Simple case with one redshift bin, two mu and two k samples
    return DummyFMWedge(
        k_grid=np.array([1.0, 2.0]),
        mu_grid=np.array([0.0, 1.0]),
        vol_array=np.array([2.0]),  # one redshift bin
    )


def test_compute_wedge_chi2_theory_covariance():
    # Default convention: the covariance is built from the theory spectrum.
    fm = _wedge_fixture()
    shape = (1, len(fm.Pk_mugrid), len(fm.Pk_kgrid))
    theory = np.ones(shape)  # covariance spectrum = 1
    data = 2.0 * np.ones(shape)  # delta = 1

    chi2 = compute_wedge_chi2(data, theory, fm)
    # cov = 8 pi^2 / (k^2 V) * 1, delta^2 = 1, V = 2
    # sum_z int_mu 2 int_k delta^2 / cov = 5/(4 pi^2)
    expected = 5.0 / (4.0 * np.pi**2)
    assert chi2 == pytest.approx(expected, rel=1e-3)


def test_compute_wedge_chi2_frozen_data_covariance():
    # Legacy convention: the covariance is frozen at the data spectrum.
    fm = _wedge_fixture()
    shape = (1, len(fm.Pk_mugrid), len(fm.Pk_kgrid))
    data = np.ones(shape)
    theory = np.zeros_like(data)

    chi2 = compute_wedge_chi2(data, theory, fm, covariance_spectrum=data)
    expected = 5.0 / (4.0 * np.pi**2)
    assert chi2 == pytest.approx(expected, rel=1e-3)


def test_compute_wedge_chi2_vanishes_at_fiducial():
    fm = _wedge_fixture()
    shape = (1, len(fm.Pk_mugrid), len(fm.Pk_kgrid))
    data = 3.0 * np.ones(shape)

    assert compute_wedge_chi2(data, data, fm) == pytest.approx(0.0, abs=1e-12)


def test_loglike_no_inputs_returns_minus_inf():
    # loglike with no theory and no params gives -inf
    ll = loglike()
    assert ll == -np.inf
