import numpy as np
import pytest

from cosmicfishpie.likelihood import sampler as sampler_mod


class _FakeContext:
    """Minimal stand-in for an AnalysisContext, enough for NautilusSampler."""

    def __init__(self):
        self.freeparams = {}
        self.allparams = {}
        self.observables = ("WL",)


class _FakePhotometricLikelihood:
    def __init__(self, cosmo_data, cosmo_theory, observables=None, data_cells=None):
        self.data_obs = data_cells if data_cells is not None else {"ells": np.array([1.0])}

    def loglike(self, *args, **kwargs):
        return 0.0


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
        "observables": ["WL"],
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


class _FakePool:
    def __init__(self):
        self.closed = False
        self.terminated = False
        self.joined = False

    def close(self):
        self.closed = True

    def terminate(self):
        self.terminated = True

    def join(self):
        self.joined = True


class _FakeSpawnContext:
    def __init__(self, pool):
        self.pool_instance = pool
        self.pool_call = None

    def Pool(self, size, initializer, initargs):
        self.pool_call = (size, initializer, initargs)
        return self.pool_instance


def test_worker_initializer_builds_process_local_likelihood(monkeypatch):
    data_cells = {"ells": np.array([1.0, 2.0])}
    sampler_mod._WORKER_LIKELIHOOD = None
    monkeypatch.setattr(
        sampler_mod, "_build_context", lambda config, observables=None: _FakeContext()
    )
    monkeypatch.setattr(sampler_mod, "PhotometricLikelihood", _FakePhotometricLikelihood)

    sampler_mod._initialize_likelihood_worker(
        {"observables": ["WL"]},
        [{"type": "photometric", "settings": {}}],
        [data_cells],
    )

    assert sampler_mod._WORKER_LIKELIHOOD.data_obs is data_cells
    assert sampler_mod._worker_loglike({"Omegam": 0.3}) == 0.0


def test_parallel_run_uses_external_spawn_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sampler_mod, "build_analysis_context", lambda **kwargs: _FakeContext())
    monkeypatch.setattr(sampler_mod, "PhotometricLikelihood", _FakePhotometricLikelihood)

    fake_pool = _FakePool()
    fake_context = _FakeSpawnContext(fake_pool)
    monkeypatch.setattr(sampler_mod.multiprocessing, "get_context", lambda method: fake_context)

    captured = {}

    def make_nautilus(**kwargs):
        captured.update(kwargs)
        return _FakeNautilusSampler(True, **kwargs)

    monkeypatch.setattr(sampler_mod, "Sampler", make_nautilus)
    config = {
        "name": "parallel_test",
        "fiducial": {},
        "observables": ["WL"],
        "options": {
            "code": "symbolic",
            "survey_name_photo": "Euclid",
            "survey_name_spectro": None,
            "cosmo_model": "LCDM",
            "survey_name": "Euclid",
        },
        "priors": {},
        "sampler_settings": {"pool": 2, "n_live": 10, "n_networks": 1, "n_batch": 2},
    }

    naut_sampler = sampler_mod.NautilusSampler(config)
    naut_sampler.run()

    assert fake_context.pool_call[0] == 2
    assert fake_context.pool_call[1] is sampler_mod._initialize_likelihood_worker
    assert fake_context.pool_call[2][1] == naut_sampler.likelihood_specs
    assert fake_context.pool_call[2][2] == naut_sampler.likelihood_payloads
    assert captured["pool"] is fake_pool
    assert captured["likelihood"] is sampler_mod._worker_loglike
    assert captured["pass_dict"] is True
    assert "likelihood_kwargs" not in captured
    assert fake_pool.closed and fake_pool.joined
    assert not fake_pool.terminated


def test_mixed_probes_require_explicit_independence_declaration():
    config = {"observables": ["WL", "GCsp"]}

    with pytest.raises(ValueError, match="statistically independent"):
        sampler_mod._likelihood_specs(config)


def test_explicit_components_build_composite_and_preserve_settings(monkeypatch):
    builds = []

    class FakeLikelihood:
        def __init__(self, value):
            self.value = value
            self.data_obs = np.array([value])

        def loglike(self, *args, **kwargs):
            return self.value

    class FakeComponent:
        def build(self, config, settings, data=None, context=None):
            builds.append((settings, data, context))
            value = settings["value"] if data is None else float(data[0])
            return FakeLikelihood(value)

        def data_payload(self, likelihood):
            return likelihood.data_obs

    monkeypatch.setitem(sampler_mod.LIKELIHOOD_COMPONENTS, "fake", FakeComponent())
    config = {
        "observables": ["CMB"],
        "likelihoods": [
            {"type": "fake", "value": 1.5},
            {"type": "fake", "value": 2.5},
        ],
    }

    specs = sampler_mod._likelihood_specs(config)
    likelihood = sampler_mod._build_likelihood(specs, config)
    payloads = sampler_mod._likelihood_payloads(specs, likelihood)
    rebuilt = sampler_mod._build_likelihood(specs, config, payloads)

    assert likelihood.loglike(param_dict={}) == pytest.approx(4.0)
    assert rebuilt.loglike(param_dict={}) == pytest.approx(4.0)
    assert [build[0]["value"] for build in builds[:2]] == [1.5, 2.5]
    assert all(build[2] is None for build in builds)


def test_spectroscopic_component_uses_mutable_fisher_context(monkeypatch):
    context = object()
    captured = {}

    class FakeSpectroLikelihood:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.data_obs = kwargs["data_obs"]

    monkeypatch.setattr(sampler_mod, "_build_fisher_context", lambda config, obs: context)
    monkeypatch.setattr(sampler_mod, "SpectroLikelihood", FakeSpectroLikelihood)
    payload = np.array([1.0, 2.0])
    config = {"observables": ["GCsp"]}
    specs = sampler_mod._likelihood_specs(config)

    sampler_mod._build_likelihood(specs, config, [payload])

    assert captured["cosmoFM_data"] is context
    assert captured["cosmoFM_theory"] is context
    assert captured["data_obs"] is payload
    assert captured["leg_flag"] == "wedges"
