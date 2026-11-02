import numpy as np
import pytest

from cosmicfishpie.likelihood import spectro_like as spectro_like_mod
from cosmicfishpie.likelihood.spectro_like import (
    SpectroLikelihood,
    _dict_with_updates,
    compute_chi2_legendre,
    compute_covariance_legendre,
    compute_theory_spectro,
    compute_wedge_chi2,
    legendre_Pgg,
    loglike,
    observable_Pgg,
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


# --- observable_Pgg -----------------------------------------------------


class DummyPkCovBins:
    def __init__(self, z_bins):
        self.global_z_bin_mids = np.array(z_bins, dtype=float)


class DummyFMObsPgg:
    def __init__(self, z_bins, musamp, ksamp):
        self.pk_cov = DummyPkCovBins(z_bins)
        self.obs_spectrum = ("g", "g")
        self.Pk_musamp = musamp
        self.Pk_ksamp = ksamp
        self.Pk_kmesh = np.ones((musamp, ksamp))
        self.Pk_mumesh = np.zeros((musamp, ksamp))


class DummyTheorySpectro:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def noisy_P_ij(self, z, kmesh, mumesh, si, sj):
        self.calls.append((z, si, sj))
        return np.full(kmesh.shape, self.value)


def test_observable_pgg_sums_theory_and_shot_noise():
    fm = DummyFMObsPgg(z_bins=[0.5, 1.0], musamp=2, ksamp=3)
    theory = DummyTheorySpectro(value=5.0)
    shot = np.array([1.0, 2.0])

    result = observable_Pgg(theory, fm, nuisance_shot=shot)

    assert result.shape == (2, 2, 3)
    assert np.allclose(result[0], 6.0)
    assert np.allclose(result[1], 7.0)
    assert theory.calls == [(0.5, "g", "g"), (1.0, "g", "g")]


def test_observable_pgg_defaults_nuisance_shot_to_zero():
    fm = DummyFMObsPgg(z_bins=[0.5], musamp=1, ksamp=1)
    theory = DummyTheorySpectro(value=3.0)

    result = observable_Pgg(theory, fm, nuisance_shot=None)

    assert np.allclose(result, 3.0)


# --- _dict_with_updates ---------------------------------------------------


def test_dict_with_updates_pops_matching_keys_from_pool():
    template = {"a": 1, "b": 2}
    pool = {"a": 10, "c": 30}

    result = _dict_with_updates(template, pool)

    assert result == {"a": 10, "b": 2}
    # matched key was popped out of the pool; unmatched key remains
    assert pool == {"c": 30}


# --- compute_covariance_legendre ------------------------------------------


class DummyPkCovLegendreCov:
    def __init__(self, vol_array):
        self._vol = np.array(vol_array, dtype=float)

    def volume_survey(self, ibin):
        return self._vol[ibin]


class DummyFMCovLegendre:
    def __init__(self, k_grid, vol_array):
        self.Pk_kgrid = np.array(k_grid, dtype=float)
        self.pk_cov = DummyPkCovLegendreCov(vol_array)


def test_compute_covariance_legendre_returns_matrix_inverse():
    k_grid = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    n_k = len(k_grid)
    n_z = 2
    n_ell = 3
    fm = DummyFMCovLegendre(k_grid=k_grid, vol_array=[1.0, 2.0])
    rng = np.random.default_rng(0)
    P_ell = 1.0 + 0.1 * rng.standard_normal((n_k, n_z, n_ell))

    covariance, inv_covariance = compute_covariance_legendre(P_ell, fm)

    assert covariance.shape == (n_k, n_z, n_ell, n_ell)
    assert inv_covariance.shape == (n_k, n_z, n_ell, n_ell)
    identity = np.eye(n_ell)
    for ik in range(n_k):
        for iz in range(n_z):
            product = covariance[ik, iz] @ inv_covariance[ik, iz]
            assert np.allclose(product, identity, atol=1e-6)


def test_compute_covariance_legendre_raises_on_k_length_mismatch():
    fm = DummyFMCovLegendre(k_grid=[0.1, 0.2, 0.3], vol_array=[1.0])
    P_ell = np.ones((2, 1, 3))  # n_k=2 does not match Pk_kgrid length 3

    with pytest.raises(ValueError):
        compute_covariance_legendre(P_ell, fm)


# --- compute_theory_spectro -------------------------------------------------


class DummyFMTheorySpectro:
    def __init__(self):
        self.pk_cov = DummyPkCovBins([0.5, 1.0])
        self.PShotpars = {"Ps_1": 10.0, "Ps_2": 20.0}
        self.Spectrobiaspars = {"b1": 1.0}
        self.Spectrononlinpars = {"sigma_p": 5.0}
        self.IMbiasparams = {}
        self.fiducialcosmopars = {"Omegam": 0.3}
        self.fiducialcosmo = object()
        self.Pk_mugrid = np.linspace(-1.0, 1.0, 3)


def test_compute_theory_spectro_wedges_branch(monkeypatch):
    calls = {}

    def fake_compute_gal_spectro(cosmological, **kwargs):
        calls["cosmological"] = cosmological
        calls["kwargs"] = kwargs
        return "fake-spectro-vary"

    def fake_spectro_cov(fiducialpars, configuration, fiducial_specobs):
        calls["cov_args"] = (fiducialpars, configuration, fiducial_specobs)
        return "fake-cov-vary"

    def fake_observable_pgg(theory_spectro, cosmoFM, nuisance_shot=None):
        calls["obs_args"] = (theory_spectro, nuisance_shot)
        return np.ones((2, 3, 3))

    monkeypatch.setattr(spectro_like_mod.spobs, "ComputeGalSpectro", fake_compute_gal_spectro)
    monkeypatch.setattr(spectro_like_mod.spcov, "SpectroCov", fake_spectro_cov)
    monkeypatch.setattr(spectro_like_mod, "observable_Pgg", fake_observable_pgg)

    fm = DummyFMTheorySpectro()
    param_dict = {"Ps_1": 99.0, "b1": 2.0, "Omegam": 0.31, "extra_unused": 42}

    result = compute_theory_spectro(param_dict, fm, leg_flag="wedges")

    assert result.shape == (2, 3, 3)
    _, nuisance_shot = calls["obs_args"]
    assert np.allclose(nuisance_shot, [99.0, 20.0])  # Ps_1 popped, Ps_2 defaulted
    assert calls["kwargs"]["spectrobiaspars"] == {"b1": 2.0}
    assert calls["cosmological"] == {"Omegam": 0.31}
    # original param_dict must be untouched (deepcopy inside the function)
    assert param_dict == {"Ps_1": 99.0, "b1": 2.0, "Omegam": 0.31, "extra_unused": 42}


def test_compute_theory_spectro_legendre_branch(monkeypatch):
    monkeypatch.setattr(
        spectro_like_mod.spobs, "ComputeGalSpectro", lambda cosmological, **kw: "spectro"
    )
    monkeypatch.setattr(
        spectro_like_mod.spcov,
        "SpectroCov",
        lambda fiducialpars, configuration, fiducial_specobs: "cov",
    )
    monkeypatch.setattr(
        spectro_like_mod,
        "observable_Pgg",
        lambda theory, fm, nuisance_shot=None: np.ones((2, 3, 3)),
    )

    fm = DummyFMTheorySpectro()
    result = compute_theory_spectro({"b1": 1.0}, fm, leg_flag="legendre")

    # legendre_Pgg transposes (n_z, n_mu, n_ell) -> (n_ell, n_z, n_mu)-ish; just check shape sanity
    assert result.shape[1] == 2  # n_z preserved


def test_compute_theory_spectro_rejects_unknown_leg_flag(monkeypatch):
    monkeypatch.setattr(
        spectro_like_mod.spobs, "ComputeGalSpectro", lambda cosmological, **kw: "spectro"
    )
    monkeypatch.setattr(
        spectro_like_mod.spcov,
        "SpectroCov",
        lambda fiducialpars, configuration, fiducial_specobs: "cov",
    )
    monkeypatch.setattr(
        spectro_like_mod,
        "observable_Pgg",
        lambda theory, fm, nuisance_shot=None: np.ones((2, 3, 3)),
    )

    fm = DummyFMTheorySpectro()
    with pytest.raises(ValueError):
        compute_theory_spectro({}, fm, leg_flag="bogus")


# --- SpectroLikelihood ------------------------------------------------------


class DummyCosmoFMData:
    def __init__(self, observables, has_pk_cov=False, with_set_pk_settings=True):
        self.observables = observables
        self.pk_cov = "existing-pk-cov" if has_pk_cov else None
        self.fiducialcosmopars = {"Omegam": 0.3}
        self.Spectrobiaspars = {"b1": 1.0}
        self.Spectrononlinpars = {"sigma_p": 5.0}
        self.IMbiaspars = {}
        self.PShotpars = {"Ps_1": 10.0}
        self.set_pk_settings_called = False
        if with_set_pk_settings:
            self.set_pk_settings = self._set_pk_settings

    def _set_pk_settings(self):
        self.set_pk_settings_called = True
        self.pk_cov = "computed-pk-cov"


def test_spectro_likelihood_rejects_invalid_covariance_from():
    with pytest.raises(ValueError):
        SpectroLikelihood(cosmoFM_data=object(), covariance_from="bogus")


def test_spectro_likelihood_preloaded_wedges_sets_data_wedges():
    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=True)
    data = np.ones((2, 3, 3))

    likelihood = SpectroLikelihood(cosmoFM_data=fm, leg_flag="wedges", data_obs=data)

    assert np.array_equal(likelihood.data_obs, data)
    assert np.array_equal(likelihood.data_wedges, data)


def test_spectro_likelihood_preloaded_legendre_builds_inv_covariance(monkeypatch):
    calls = []

    def fake_compute_covariance_legendre(data, cosmoFM):
        calls.append(data)
        return ("fake-cov", "fake-inv-cov")

    monkeypatch.setattr(
        spectro_like_mod, "compute_covariance_legendre", fake_compute_covariance_legendre
    )

    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=True)
    data = np.ones((2, 3, 3))

    likelihood = SpectroLikelihood(cosmoFM_data=fm, leg_flag="legendre", data_obs=data)

    assert likelihood._inv_cov_legendre == "fake-inv-cov"
    assert len(calls) == 1


def test_spectro_likelihood_compute_data_without_preload_runs_ensure_state(monkeypatch):
    monkeypatch.setattr(spectro_like_mod.spobs, "ComputeGalSpectro", lambda **kw: "fake-pk-obs-fid")
    monkeypatch.setattr(spectro_like_mod.spcov, "SpectroCov", lambda *a, **kw: "fake-pk-cov")
    monkeypatch.setattr(
        spectro_like_mod,
        "observable_Pgg",
        lambda theory, fm, nuisance_shot=None: np.full((1, 2, 2), 7.0),
    )

    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=False)

    likelihood = SpectroLikelihood(cosmoFM_data=fm, leg_flag="wedges")

    assert fm.set_pk_settings_called is True
    assert fm.obs_spectrum == ["g", "g"]
    assert np.allclose(likelihood.data_wedges, 7.0)


def test_spectro_likelihood_compute_data_without_preload_legendre(monkeypatch):
    monkeypatch.setattr(spectro_like_mod.spobs, "ComputeGalSpectro", lambda **kw: "fake-pk-obs-fid")
    monkeypatch.setattr(spectro_like_mod.spcov, "SpectroCov", lambda *a, **kw: "fake-pk-cov")
    monkeypatch.setattr(
        spectro_like_mod,
        "observable_Pgg",
        lambda theory, fm, nuisance_shot=None: np.ones((1, 1, 1)),
    )

    legendre_calls = []

    def fake_legendre_pgg(obsPgg, cosmoFM):
        legendre_calls.append(obsPgg)
        return "fake-p-ell"

    cov_calls = []

    def fake_compute_covariance_legendre(p_ell, cosmoFM):
        cov_calls.append(p_ell)
        return ("cov", "inv-cov-legendre")

    monkeypatch.setattr(spectro_like_mod, "legendre_Pgg", fake_legendre_pgg)
    monkeypatch.setattr(
        spectro_like_mod, "compute_covariance_legendre", fake_compute_covariance_legendre
    )

    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=False)

    likelihood = SpectroLikelihood(cosmoFM_data=fm, leg_flag="legendre")

    assert likelihood.data_obs == "fake-p-ell"
    assert likelihood._inv_cov_legendre == "inv-cov-legendre"
    assert len(legendre_calls) == 1
    assert len(cov_calls) == 1


@pytest.mark.parametrize(
    "observables,expected_spectrum",
    [
        (["IM", "GCsp"], ["I", "g"]),
        (["IM"], ["I", "I"]),
        (["GCsp"], ["g", "g"]),
    ],
)
def test_spectro_likelihood_ensure_runtime_state_obs_spectrum_branches(
    monkeypatch, observables, expected_spectrum
):
    monkeypatch.setattr(spectro_like_mod.spobs, "ComputeGalSpectro", lambda **kw: "fake-pk-obs-fid")
    monkeypatch.setattr(spectro_like_mod.spcov, "SpectroCov", lambda *a, **kw: "fake-pk-cov")
    monkeypatch.setattr(
        spectro_like_mod,
        "observable_Pgg",
        lambda theory, fm, nuisance_shot=None: np.ones((1, 1, 1)),
    )

    fm = DummyCosmoFMData(observables=observables, has_pk_cov=False)
    SpectroLikelihood(cosmoFM_data=fm, leg_flag="wedges")

    assert fm.obs_spectrum == expected_spectrum


def test_spectro_likelihood_ensure_runtime_state_raises_without_set_pk_settings():
    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=False, with_set_pk_settings=False)

    with pytest.raises(AttributeError):
        SpectroLikelihood(cosmoFM_data=fm, leg_flag="wedges", data_obs=np.ones((1, 1, 1)))


def test_spectro_likelihood_compute_theory_delegates(monkeypatch):
    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=True)
    likelihood = SpectroLikelihood(cosmoFM_data=fm, leg_flag="wedges", data_obs=np.ones((1, 1, 1)))

    recorded = {}

    def fake_compute_theory_spectro(param_dict, cosmoFM_theory, leg_flag):
        recorded["args"] = (param_dict, cosmoFM_theory, leg_flag)
        return "theory-result"

    monkeypatch.setattr(spectro_like_mod, "compute_theory_spectro", fake_compute_theory_spectro)

    result = likelihood.compute_theory({"b1": 1.0})

    assert result == "theory-result"
    assert recorded["args"] == ({"b1": 1.0}, likelihood.cosmo_theory, "wedges")


def test_spectro_likelihood_compute_chi2_wedges_covariance_selection(monkeypatch):
    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=True)
    data = np.ones((1, 1, 1))

    recorded = []

    def fake_compute_wedge_chi2(data_obs, theory_obs, cosmoFM, covariance_spectrum=None):
        recorded.append(covariance_spectrum)
        return 2.0

    monkeypatch.setattr(spectro_like_mod, "compute_wedge_chi2", fake_compute_wedge_chi2)

    likelihood = SpectroLikelihood(
        cosmoFM_data=fm, leg_flag="wedges", data_obs=data, covariance_from="theory"
    )
    assert likelihood.compute_chi2(data) == 2.0
    assert recorded[-1] is None  # theory covariance -> None passed through

    likelihood.covariance_from = "data"
    assert likelihood.compute_chi2(data) == 2.0
    assert recorded[-1] is likelihood.data_obs  # data covariance -> frozen at data


def test_spectro_likelihood_compute_chi2_legendre_theory_covariance_rebuilds(monkeypatch):
    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=True)
    data = np.ones((1, 1, 3))
    likelihood = SpectroLikelihood(
        cosmoFM_data=fm, leg_flag="wedges", data_obs=data, covariance_from="theory"
    )
    likelihood.leg_flag = "legendre"

    cov_calls = []

    def fake_compute_covariance_legendre(theory_obs, cosmoFM):
        cov_calls.append(theory_obs)
        return ("cov", "inv-cov")

    def fake_compute_chi2_legendre(data_obs, theory_obs, inv_covariance):
        assert inv_covariance == "inv-cov"
        return 9.0

    monkeypatch.setattr(
        spectro_like_mod, "compute_covariance_legendre", fake_compute_covariance_legendre
    )
    monkeypatch.setattr(spectro_like_mod, "compute_chi2_legendre", fake_compute_chi2_legendre)

    assert likelihood.compute_chi2(data) == 9.0
    assert likelihood.compute_chi2(data) == 9.0
    # theory covariance path rebuilds every call, never caches
    assert len(cov_calls) == 2


def test_spectro_likelihood_compute_chi2_legendre_data_covariance_is_cached(monkeypatch):
    fm = DummyCosmoFMData(observables=["GCsp"], has_pk_cov=True)
    data = np.ones((1, 1, 3))
    likelihood = SpectroLikelihood(
        cosmoFM_data=fm, leg_flag="wedges", data_obs=data, covariance_from="data"
    )
    likelihood.leg_flag = "legendre"
    assert likelihood._inv_cov_legendre is None

    cov_calls = []

    def fake_compute_covariance_legendre(data_obs, cosmoFM):
        cov_calls.append(data_obs)
        return ("cov", "inv-cov-cached")

    def fake_compute_chi2_legendre(data_obs, theory_obs, inv_covariance):
        assert inv_covariance == "inv-cov-cached"
        return 3.0

    monkeypatch.setattr(
        spectro_like_mod, "compute_covariance_legendre", fake_compute_covariance_legendre
    )
    monkeypatch.setattr(spectro_like_mod, "compute_chi2_legendre", fake_compute_chi2_legendre)

    assert likelihood.compute_chi2(data) == 3.0
    assert likelihood.compute_chi2(data) == 3.0
    # data covariance path is built once and reused
    assert len(cov_calls) == 1


# --- loglike orchestration --------------------------------------------------


class _FakeSpectroLikelihoodForLoglike:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def compute_chi2(self, theory_obsPgg):
        return 4.0

    def build_param_dict(self, param_vec=None, prior=None):
        if prior is None:
            raise AttributeError("prior required")
        return {"p": param_vec[0]}

    def loglike(self, param_dict=None):
        return -0.5 * param_dict["p"]


def test_loglike_with_theory_obs_returns_half_chi2(monkeypatch):
    monkeypatch.setattr(spectro_like_mod, "SpectroLikelihood", _FakeSpectroLikelihoodForLoglike)

    result = loglike(cosmoFM_data=object(), theory_obsPgg=np.ones((1, 1, 1)))

    assert result == pytest.approx(-2.0)


def test_loglike_with_dict_param_vec(monkeypatch):
    monkeypatch.setattr(spectro_like_mod, "SpectroLikelihood", _FakeSpectroLikelihoodForLoglike)

    result = loglike(cosmoFM_data=object(), param_vec={"p": 6.0})

    assert result == pytest.approx(-3.0)


def test_loglike_returns_minus_inf_on_param_build_failure(monkeypatch):
    monkeypatch.setattr(spectro_like_mod, "SpectroLikelihood", _FakeSpectroLikelihoodForLoglike)

    result = loglike(cosmoFM_data=object(), param_vec=[1, 2, 3], prior=None)

    assert result == -np.inf


def test_loglike_with_param_vec_and_prior_builds_params(monkeypatch):
    monkeypatch.setattr(spectro_like_mod, "SpectroLikelihood", _FakeSpectroLikelihoodForLoglike)

    result = loglike(cosmoFM_data=object(), param_vec=[7.0], prior="some-prior")

    assert result == pytest.approx(-3.5)
