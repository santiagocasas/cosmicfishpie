import numpy as np

from cosmicfishpie.LSSsurvey.photo_obs import ComputeCls

def test_compute_cls_initialization(computecls_fid):
    cosmopars, photo_Cls, _ = computecls_fid
    assert isinstance(photo_Cls, ComputeCls)
    assert photo_Cls.cosmopars == cosmopars

def test_getcls(computecls_fid):
    _, photo_Cls, _ = computecls_fid
    # compute_all already called in fixture
    assert isinstance(photo_Cls.result, dict)

def test_wl_kernel_fast_kernel_equivalence(computecls_fid, monkeypatch):
    """Check that enabling/disabling _USE_FAST_KERNEL yields the same WL kernel.

    We toggle the module-level flag and call `lensing_kernel` on the same
    ComputeCls instance to compare both the WL and IA components.
    """
    # Import module to toggle the flag reliably
    import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

    _, cls, _ = computecls_fid

    # Use the class' native z-grid and a representative WL bin
    z = cls.z
    i = cls.binrange_WL[0]

    # Ensure efficiency is available (compute_all in fixture should have done this)
    assert cls.efficiency is not None and callable(cls.efficiency[i])

    # Compute with fast kernel
    monkeypatch.setattr(photo_obs, "_USE_FAST_KERNEL", True, raising=False)
    Wwl_fast, WIA_fast = cls.lensing_kernel(z, i)

    # Compute with slow kernel
    monkeypatch.setattr(photo_obs, "_USE_FAST_KERNEL", False, raising=False)
    Wwl_slow, WIA_slow = cls.lensing_kernel(z, i)

    # Compare results
    assert np.allclose(Wwl_fast, Wwl_slow, rtol=1e-10, atol=0.0)
    assert np.allclose(WIA_fast, WIA_slow, rtol=1e-12, atol=0.0)


def test_wl_efficiency_fast_eff_equivalence(computecls_fid, monkeypatch):
    """Check that enabling/disabling _USE_FAST_EFF yields the same lensing efficiency."""
    import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

    _, cls, _ = computecls_fid

    z = cls.z
    i = cls.binrange_WL[0]

    # Compute efficiency with fast implementation
    monkeypatch.setattr(photo_obs, "_USE_FAST_EFF", True, raising=False)
    eff_fast_fn = cls.integral_efficiency(i)
    eff_fast = eff_fast_fn(z)

    # Compute efficiency with fallback implementation
    monkeypatch.setattr(photo_obs, "_USE_FAST_EFF", False, raising=False)
    eff_slow_fn = cls.integral_efficiency(i)
    eff_slow = eff_slow_fn(z)

    # Use mixed tolerance: relative where signal is appreciable, absolute near zeros
    thresh = 1e-12
    mask = np.abs(eff_slow) > thresh
    assert np.allclose(eff_fast[mask], eff_slow[mask], rtol=1e-8, atol=1e-12)
    assert np.allclose(eff_fast[~mask], eff_slow[~mask], rtol=0.0, atol=1e-12)


def test_P_limber_fast_P_equivalence(computecls_fid, monkeypatch):
    """Check that enabling/disabling _USE_FAST_P yields identical Pell."""
    import cosmicfishpie.LSSsurvey.photo_obs as photo_obs

    _, cls, _ = computecls_fid

    # Fast vectorized Pell
    monkeypatch.setattr(photo_obs, "_USE_FAST_P", True, raising=False)
    cls.P_limber()
    pell_fast = cls.Pell.copy()

    # Slow nested-loop Pell
    monkeypatch.setattr(photo_obs, "_USE_FAST_P", False, raising=False)
    cls.P_limber()
    pell_slow = cls.Pell.copy()

    assert np.allclose(pell_fast, pell_slow, rtol=1e-10, atol=0.0)
