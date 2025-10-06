from cosmicfishpie.LSSsurvey.photo_obs import ComputeCls


def test_compute_cls_initialization(computecls_fid):
    cosmopars, photo_Cls, _ = computecls_fid
    assert isinstance(photo_Cls, ComputeCls)
    assert photo_Cls.cosmopars == cosmopars


def test_getcls(computecls_fid):
    _, photo_Cls, _ = computecls_fid
    # compute_all already called in fixture
    assert isinstance(photo_Cls.result, dict)
