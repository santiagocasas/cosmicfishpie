import numpy as np

import cosmicfishpie.configs.config as cfg
from cosmicfishpie.LSSsurvey.spectro_cov import SpectroCov, SpectroDerivs


def test_spectro_cov_initialization(spectro_fisher_matrix):
    cosmoFM = spectro_fisher_matrix
    # reuse fiducial cosmological parameters subset for speed
    fiducialpars = dict(cosmoFM.fiducialcosmopars)
    spec_cov = SpectroCov(fiducialpars, configuration=cosmoFM)
    assert isinstance(spec_cov, SpectroCov)
    # basic attributes set
    assert np.isclose(
        spec_cov.fsky_spectro, cosmoFM.specs.get("fsky_spectro", spec_cov.fsky_spectro)
    )


def test_spectro_cov_volume_and_density(spectro_fisher_matrix):
    cosmoFM = spectro_fisher_matrix
    fiducialpars = dict(cosmoFM.fiducialcosmopars)
    spec_cov = SpectroCov(fiducialpars, configuration=cosmoFM)
    # pick first bin
    vol_bin0 = spec_cov.d_volume(0)
    assert vol_bin0 > 0
    survey_vol0 = spec_cov.volume_survey(0)
    assert np.isclose(survey_vol0, spec_cov.fsky_spectro * vol_bin0)
    # number density at mid of first bin
    z0 = spec_cov.global_z_bin_mids[0]
    nd = spec_cov.n_density(z0)
    assert nd >= 0


def test_spectro_cov_noise_and_noisy_pk(spectro_fisher_matrix):
    cosmoFM = spectro_fisher_matrix
    fiducialpars = dict(cosmoFM.fiducialcosmopars)
    spec_cov = SpectroCov(fiducialpars, configuration=cosmoFM)
    # simple k, mu arrays
    k = np.array([0.1])
    mu = np.array([0.5])
    z = float(spec_cov.global_z_bin_mids[0])
    # Only call the 21cm noise routine if the Intensity Mapping observable is present.
    # The default fixture uses only GCsp so ComputeGalSpectro does not define fsky_IM.
    if "IM" in spec_cov.pk_obs.observables:
        noise_21 = spec_cov.P_noise_21(z, k, mu)
        assert np.all(noise_21 >= 0)
    # galaxy shot noise branch in noisy_P_ij
    pnoisy_gg = spec_cov.noisy_P_ij(z, k, mu, si="g", sj="g")
    pnoisy_II = (
        spec_cov.noisy_P_ij(z, k, mu, si="I", sj="I")
        if "IM" in spec_cov.pk_obs.observables
        else None
    )
    assert np.all(pnoisy_gg > 0)
    if pnoisy_II is not None:
        assert np.all(pnoisy_II > 0)


def test_spectro_cov_effective_volumes_basic(spectro_fisher_matrix):
    cosmoFM = spectro_fisher_matrix
    fiducialpars = dict(cosmoFM.fiducialcosmopars)
    spec_cov = SpectroCov(fiducialpars, configuration=cosmoFM)
    k = np.array([0.1])
    mu = np.array([0.5])
    z = float(spec_cov.global_z_bin_mids[0])
    if spec_cov.pk_obs.observables == ["GCsp"]:
        # only galaxy clustering; veff uses n_density path indirectly
        veff_like = spec_cov.veff(z, k, mu)
        # ensure non-negative: coerce to numpy array
        veff_arr = np.array(veff_like, copy=False)
        assert np.all(veff_arr >= 0)
    # we always can call noisy_P_ij for gg and use it within a simple covariance style expression
    pnoisy = spec_cov.noisy_P_ij(z, k, mu, si="g", sj="g")
    assert pnoisy.shape == (1,)


def test_spectro_derivs_wrapper_minimal(spectro_fisher_matrix):
    cosmoFM = spectro_fisher_matrix
    fiducialpars = dict(cosmoFM.fiducialcosmopars)
    spec_cov = SpectroCov(fiducialpars, configuration=cosmoFM)
    # small meshes to keep fast
    kmesh = np.array([0.1])
    mumesh = np.array([0.0, 0.5])
    z_array = np.array(spec_cov.global_z_bin_mids[:1])  # just first bin
    deriv_engine = SpectroDerivs(
        z_array=z_array,
        pk_kmesh=kmesh,
        pk_mumesh=mumesh,
        fiducial_spectro_obj=spec_cov.pk_obs,
        bias_samples=["g", "g"],
        configuration=cosmoFM,
    )
    # provide simple freeparams for one cosmological parameter
    freeparams = {"Omegam": 0.01}
    derivs = deriv_engine.compute_derivs(freeparams=freeparams)
    assert isinstance(derivs, dict)
    # structure: parameter -> redshift index -> array
    if "Omegam" in derivs:
        first_par = derivs["Omegam"]
        assert 0 in first_par


def test_spectro_derivatives_remain_bound_to_context_a(spectro_fisher_matrix, monkeypatch):
    context_a = spectro_fisher_matrix
    spec_cov = SpectroCov(dict(context_a.fiducialcosmopars), configuration=context_a)
    deriv_engine = SpectroDerivs(
        z_array=np.array(spec_cov.global_z_bin_mids[:1]),
        pk_kmesh=np.array([0.1]),
        pk_mumesh=np.array([0.0, 0.5]),
        fiducial_spectro_obj=spec_cov.pk_obs,
        bias_samples=["g", "g"],
        configuration=context_a,
    )

    monkeypatch.setattr(cfg, "freeparams", {"wrong": 99.0})
    monkeypatch.setattr(cfg, "settings", {"feedback": 99, "derivatives": "UNKNOWN"})
    monkeypatch.setattr(cfg, "obs", ["UNKNOWN"])
    monkeypatch.setattr(cfg, "external", {"wrong": True})

    class CaptureProvider:
        def compute(self, request):
            assert request.configuration is context_a
            assert request.freeparams == {"Omegam": 0.01}
            assert request.observables_type == tuple(context_a.observables)
            assert request.external_settings == context_a.external
            assert request.method == context_a.settings["derivatives"]
            return {"spectro": "context-a"}

    result = deriv_engine.compute_derivs(
        freeparams={"Omegam": 0.01}, derivative_provider=CaptureProvider()
    )

    assert result == {"spectro": "context-a"}
