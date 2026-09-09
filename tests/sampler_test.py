import numpy as np
import pytest

from cosmicfishpie.likelihood import sampler as sampler_mod


class _FakeContext:
    """Minimal stand-in for an AnalysisContext, enough for NautilusSampler."""

    def __init__(self):
        self.freeparams = {}
        self.allparams = {}


class _FakePhotometricLikelihood:
    def __init__(self, cosmo_data, cosmo_theory):
        self.loglike = lambda x: 0.0


class _FakeNautilusSampler:
    """Stand-in for nautilus.Sampler with a configurable run() outcome."""

    def __init__(self, run_return, **kwargs):
        self._run_return = run_return
        self.log_z = -1.23

    def run(self, **kwargs):
        return self._run_return

    def posterior(self):
        points = np.zeros((1, 1))
        log_w = np.zeros(1)
        log_l = np.zeros(1)
        return points, log_w, log_l


def _make_sampler(tmp_path, monkeypatch, run_return):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sampler_mod, "build_analysis_context", lambda **kwargs: _FakeContext())
    monkeypatch.setattr(sampler_mod, "PhotometricLikelihood", _FakePhotometricLikelihood)
    monkeypatch.setattr(
        sampler_mod, "Sampler", lambda **kwargs: _FakeNautilusSampler(run_return, **kwargs)
    )

    config = {
        "name": "unit_test",
        "fiducial": {},
        "observables": [],
        "options": {
            "code": "symbolic",
            "survey_name_photo": "Euclid",
            "survey_name_spectro": None,
            "cosmo_model": "LCDM",
            "survey_name": "Euclid",
        },
        "priors": {},
        "sampler_settings": {
            "pool": 1,
            "n_live": 10,
            "n_networks": 1,
            "n_batch": 1,
        },
    }
    return sampler_mod.NautilusSampler(config)


def test_incomplete_run_raises_and_does_not_save_chain(tmp_path, monkeypatch):
    naut_sampler = _make_sampler(tmp_path, monkeypatch, run_return=False)

    with pytest.raises(RuntimeError, match="stopped before convergence"):
        naut_sampler.run()

    assert not (tmp_path / naut_sampler.chain_file).exists()


def test_completed_run_saves_chain_and_metadata(tmp_path, monkeypatch):
    naut_sampler = _make_sampler(tmp_path, monkeypatch, run_return=True)

    naut_sampler.run()

    assert (tmp_path / naut_sampler.chain_file).exists()
    assert (tmp_path / (naut_sampler.outroot + "_metadata.json")).exists()
