import pytest


def test_photo_cov_initialization(photo_cov_cached):
    # Basic attribute presence
    assert photo_cov_cached.cosmopars
    assert hasattr(photo_cov_cached, "allparsfid")


@pytest.mark.parametrize("with_noise", [False, True])
def test_photo_cov_cls_and_noise(photo_cov_cached, with_noise):
    base_cls = photo_cov_cached.getcls(photo_cov_cached.allparsfid)
    assert isinstance(base_cls, dict)
    if with_noise:
        noisy = photo_cov_cached.getclsnoise(base_cls)
        assert isinstance(noisy, dict)
        # check at least one auto term got increased (heuristic)
        some_key = next(k for k in noisy if "x" in k and k.split("x")[0] == k.split("x")[1])
        assert noisy[some_key][0] >= base_cls[some_key][0]


def test_photo_cov_covmat_cached(photo_cov_cached):
    # compute_covmat already run in fixture; covmat & noisy_cls should exist
    assert hasattr(photo_cov_cached, "covmat")
    assert hasattr(photo_cov_cached, "noisy_cls")
    assert isinstance(photo_cov_cached.noisy_cls, dict)
    assert isinstance(photo_cov_cached.covmat, list)
    if photo_cov_cached.covmat:
        first = photo_cov_cached.covmat[0]
        assert hasattr(first, "columns") and hasattr(first, "index")


def test_photo_cov_compute_derivs_stub(monkeypatch, photo_cov_cached):
    """Stub derivative engine so we only test integration & shape, not heavy recomputation."""
    import cosmicfishpie.configs.config as cfg
    import cosmicfishpie.fishermatrix.derivatives as fishderiv

    freeparams_obj = cfg.freeparams or {}
    subset_key = next((k for k in freeparams_obj if k.startswith("b")), None)
    if subset_key is None:
        subset_key = "Omegam" if "Omegam" in freeparams_obj else list(freeparams_obj.keys())[0]

    class DummyDeriv:
        def __init__(self, *args, **kwargs):
            self.result = {subset_key: {"dummy": 0.0}}

    monkeypatch.setattr(fishderiv, "derivatives", DummyDeriv)
    derivs = photo_cov_cached.compute_derivs()
    assert subset_key in derivs
