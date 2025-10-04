import pytest

from cosmicfishpie.LSSsurvey.photo_cov import PhotoCov


@pytest.fixture(scope="module")
def fast_photo_cov_setup(computecls_fid):
    """Reuse module-scoped ComputeCls and provide its components.

    computecls_fid already executed compute_all; just return for covariance/derivative tests.
    """
    cosmopars, fid_cls, cosmoFM = computecls_fid
    return cosmopars, fid_cls, cosmoFM


# You might need to add more imports depending on what you're testing


def test_photo_cov_initialization(fast_photo_cov_setup):
    cosmopars, fid_cls, cosmoFM = fast_photo_cov_setup
    photo_cov = PhotoCov(
        cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars, fiducial_Cls=fid_cls
    )
    assert isinstance(photo_cov, PhotoCov)
    assert photo_cov.cosmopars == cosmopars


def test_get_cls(fast_photo_cov_setup):
    cosmopars, fid_cls, cosmoFM = fast_photo_cov_setup
    photo_cov = PhotoCov(
        cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars, fiducial_Cls=fid_cls
    )
    allparsfid = dict()
    allparsfid.update(cosmopars)
    allparsfid.update(cosmoFM.IApars)
    allparsfid.update(cosmoFM.photobiaspars)
    allparsfid.update(cosmoFM.photopars)
    cls = photo_cov.getcls(allparsfid)
    assert isinstance(cls, dict)
    # Add more specific assertions based on what you expect in the result


def test_get_cls_noise(fast_photo_cov_setup):
    cosmopars, fid_cls, cosmoFM = fast_photo_cov_setup
    photo_cov = PhotoCov(
        cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars, fiducial_Cls=fid_cls
    )
    cls = photo_cov.getcls(photo_cov.allparsfid)
    noisy_cls = photo_cov.getclsnoise(cls)
    assert isinstance(noisy_cls, dict)
    # Add more specific assertions based on what you expect in the result


def test_photo_cov_compute_covmat(fast_photo_cov_setup):
    cosmopars, fid_cls, cosmoFM = fast_photo_cov_setup
    photo_cov = PhotoCov(
        cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars, fiducial_Cls=fid_cls
    )
    result = photo_cov.compute_covmat()
    # compute_covmat should return a tuple (noisy_cls, covmat); if None returned, fail explicitly
    assert result is not None, "compute_covmat returned None unexpectedly"
    noisy_cls, covmat = result
    assert isinstance(noisy_cls, dict)
    assert isinstance(covmat, list)
    if covmat:
        first = covmat[0]
        # Duck-typing for DataFrame: should have columns and index attributes
        assert hasattr(first, "columns") and hasattr(first, "index")


def test_photo_cov_compute_derivs(fast_photo_cov_setup):
    import cosmicfishpie.configs.config as cfg

    cosmopars, fid_cls, cosmoFM = fast_photo_cov_setup
    photo_cov = PhotoCov(
        cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars, fiducial_Cls=fid_cls
    )
    original_free = dict(cfg.freeparams)
    # choose fast subset (bias if available; else Omegam)
    subset_key = next((k for k in original_free if k.startswith("b")), None)
    if subset_key is None:
        subset_key = "Omegam" if "Omegam" in original_free else list(original_free.keys())[0]
    try:
        cfg.freeparams = {subset_key: original_free[subset_key]}
        derivs = photo_cov.compute_derivs()
        assert isinstance(derivs, dict)
        assert subset_key in derivs
    finally:
        cfg.freeparams = original_free
