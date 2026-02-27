import numpy as np
import pytest

import cosmicfishpie.configs.config as cfg


def test_getcls_cache_hit_returns_same_object(photo_cov_cached):
    cls1 = photo_cov_cached.getcls(photo_cov_cached.allparsfid)
    cls2 = photo_cov_cached.getcls(photo_cov_cached.allparsfid)
    assert cls1 is cls2


def test_getclsnoise_adds_only_auto_noise(photo_cov_cached):
    base = photo_cov_cached.getcls(photo_cov_cached.allparsfid)
    noisy = photo_cov_cached.getclsnoise(base)

    auto_key = next(k for k in base if "x" in k and k.split("x")[0] == k.split("x")[1])
    assert np.all(noisy[auto_key] >= base[auto_key])

    cross_key = next(k for k in base if "x" in k and k.split("x")[0] != k.split("x")[1])
    np.testing.assert_allclose(noisy[cross_key], base[cross_key])


def test_compute_covmat_fills_missing_omegab_fallback(monkeypatch, photo_cov_cached):
    monkeypatch.setitem(cfg.settings, "feedback", 0)
    monkeypatch.setattr(cfg, "freeparams", {"Omegam": 0.01, "h": 0.01, "Omegab": 0.01})

    # Force missing Omegab in local fiducials
    photo_cov_cached.allparsfid.pop("Omegab", None)

    fake_cls = {
        "ells": np.array([2, 3], dtype=float),
        "GCph 1xGCph 1": np.array([1.0, 1.1], dtype=float),
    }

    monkeypatch.setattr(photo_cov_cached, "getcls", lambda _: fake_cls)
    monkeypatch.setattr(photo_cov_cached, "getclsnoise", lambda cls: cls)
    monkeypatch.setattr(photo_cov_cached, "get_covmat", lambda noisy: [])

    noisy, cov = photo_cov_cached.compute_covmat()
    assert noisy["ells"][0] == 2
    assert cov == []
    assert photo_cov_cached.allparsfid["Omegab"] == pytest.approx(0.05)


def test_compute_covmat_resolves_missing_from_cfg_fiducialparams(monkeypatch, photo_cov_cached):
    monkeypatch.setitem(cfg.settings, "feedback", 0)
    monkeypatch.setattr(cfg, "freeparams", {"test_param": 0.01})
    monkeypatch.setattr(cfg, "fiducialparams", {"test_param": 1.23}, raising=False)

    photo_cov_cached.allparsfid.pop("test_param", None)

    fake_cls = {
        "ells": np.array([2, 3], dtype=float),
        "GCph 1xGCph 1": np.array([1.0, 1.1], dtype=float),
    }
    monkeypatch.setattr(photo_cov_cached, "getcls", lambda _: fake_cls)
    monkeypatch.setattr(photo_cov_cached, "getclsnoise", lambda cls: cls)
    monkeypatch.setattr(photo_cov_cached, "get_covmat", lambda noisy: [])

    photo_cov_cached.compute_covmat()
    assert photo_cov_cached.allparsfid["test_param"] == pytest.approx(1.23)


def test_compute_covmat_raises_for_unresolved_missing_freeparam(monkeypatch, photo_cov_cached):
    monkeypatch.setitem(cfg.settings, "feedback", 0)
    monkeypatch.setattr(cfg, "freeparams", {"definitely_missing_param": 0.01})
    monkeypatch.setattr(cfg, "fiducialparams", {}, raising=False)

    with pytest.raises(ValueError, match="Fiducial values missing for free parameters"):
        photo_cov_cached.compute_covmat()
