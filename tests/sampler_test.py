import json
import os

import numpy as np
import pytest

from cosmicfishpie.likelihood import sampler as sampler_mod


class _FakeContext:
    """Minimal stand-in for an AnalysisContext, enough for NautilusSampler."""

    def __init__(self, freeparams=None, allparams=None):
        self.freeparams = freeparams if freeparams is not None else {}
        self.allparams = allparams if allparams is not None else {}
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

    def evidence(self):
        return self.log_z

    def posterior(self):
        points = np.zeros((1, 1))
        log_w = np.zeros(1)
        log_l = np.zeros(1)
        return points, log_w, log_l


def _make_sampler(tmp_path, monkeypatch, run_return, output_dir=None):
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
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
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


def test_output_dir_places_sampler_artifacts_outside_working_directory(tmp_path, monkeypatch):
    output_dir = tmp_path / "scratch"
    naut_sampler = _make_sampler(tmp_path, monkeypatch, run_return=True, output_dir=output_dir)

    naut_sampler.run()

    assert (output_dir / "chains_unit_test").exists()
    assert not (tmp_path / "chains").exists()


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


def test_spectroscopic_component_data_payload_returns_data_obs():
    component = sampler_mod._SpectroscopicComponent()

    class FakeLikelihood:
        data_obs = "payload"

    assert component.data_payload(FakeLikelihood()) == "payload"


def test_build_fisher_context_constructs_fisher_matrix(monkeypatch):
    captured = {}

    class FakeFisherMatrix:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(sampler_mod, "FisherMatrix", FakeFisherMatrix)
    config = {
        "options": {"survey_name": "Euclid", "cosmo_model": "LCDM"},
        "fiducial": {"Omegam": 0.3},
    }

    sampler_mod._build_fisher_context(config, ["GCsp"])

    assert captured["surveyName"] == "Euclid"
    assert captured["cosmoModel"] == "LCDM"
    assert captured["observables"] == ["GCsp"]
    assert captured["fiducialpars"] == {"Omegam": 0.3}


def test_worker_initializer_verbose_for_worker_one(monkeypatch, capsys):
    sampler_mod._WORKER_LIKELIHOOD = None

    class FakeProcess:
        _identity = (1,)

    monkeypatch.setattr(sampler_mod.multiprocessing, "current_process", lambda: FakeProcess())
    monkeypatch.setattr(
        sampler_mod, "_build_context", lambda config, observables=None: _FakeContext()
    )
    monkeypatch.setattr(sampler_mod, "PhotometricLikelihood", _FakePhotometricLikelihood)

    sampler_mod._initialize_likelihood_worker(
        {"observables": ["WL"]},
        [{"type": "photometric", "settings": {}}],
        [{"ells": np.array([1.0])}],
    )

    captured = capsys.readouterr()
    assert "[worker 1] pid=" in captured.out
    assert "[worker 1] likelihood built" in captured.out


def test_worker_loglike_requires_initialization(monkeypatch):
    monkeypatch.setattr(sampler_mod, "_WORKER_LIKELIHOOD", None)

    with pytest.raises(RuntimeError, match="not initialized"):
        sampler_mod._worker_loglike({})


def test_register_likelihood_component_validates_name():
    with pytest.raises(ValueError, match="non-empty strings"):
        sampler_mod.register_likelihood_component("", object())


def test_register_likelihood_component_validates_interface():
    class Incomplete:
        pass

    with pytest.raises(TypeError, match=r"build\(\) and data_payload\(\)"):
        sampler_mod.register_likelihood_component("incomplete", Incomplete())


def test_register_likelihood_component_adds_valid_component():
    class Good:
        def build(self, config, settings, data=None, context=None):
            return None

        def data_payload(self, likelihood):
            return None

    sampler_mod.register_likelihood_component("custom_test_component", Good())
    try:
        assert "custom_test_component" in sampler_mod.LIKELIHOOD_COMPONENTS
    finally:
        del sampler_mod.LIKELIHOOD_COMPONENTS["custom_test_component"]


def test_likelihood_specs_raises_when_no_probe_matches():
    with pytest.raises(ValueError, match="Could not infer"):
        sampler_mod._likelihood_specs({"observables": ["CMB"]})


def test_likelihood_specs_accepts_single_dict():
    config = {"observables": ["WL"], "likelihoods": {"type": "photometric"}}

    specs = sampler_mod._likelihood_specs(config)

    assert specs == [{"type": "photometric", "settings": {}}]


def test_likelihood_specs_accepts_single_string():
    config = {"observables": ["WL"], "likelihoods": "photometric"}

    specs = sampler_mod._likelihood_specs(config)

    assert specs[0]["type"] == "photometric"


def test_likelihood_specs_rejects_empty_list():
    with pytest.raises(ValueError, match="cannot be empty"):
        sampler_mod._likelihood_specs({"observables": ["WL"], "likelihoods": []})


def test_likelihood_specs_rejects_invalid_entry_type():
    with pytest.raises(TypeError, match="component name or mapping"):
        sampler_mod._likelihood_specs({"observables": ["WL"], "likelihoods": [123]})


def test_likelihood_specs_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown likelihood component"):
        sampler_mod._likelihood_specs({"observables": ["WL"], "likelihoods": ["nonexistent"]})


def test_build_likelihood_rejects_payload_count_mismatch():
    specs = [{"type": "photometric", "settings": {}}]

    with pytest.raises(ValueError, match="counts do not match"):
        sampler_mod._build_likelihood(specs, {}, payloads=[1, 2])


@pytest.mark.parametrize(
    "param_name,expected",
    [
        ("Omegam", r"$\Omega_{m}$"),
        ("Omega", r"$\Omega$"),
        ("omegab", r"$\omega_{b}$"),
        ("omega", r"$\omega$"),
        ("b1", r"$b_{1}$"),
        ("bx", "bx"),
        ("sigma8", r"$\sigma_8$"),
        ("unknown_param", "unknown_param"),
    ],
)
def test_format_param_label_variants(param_name, expected):
    assert sampler_mod._format_param_label(param_name) == expected


def test_load_chain_metadata_discovers_single_file(tmp_path):
    metadata = {
        "chain_file": "run.txt",
        "sampled_fiducial_params": {"Omegam": 0.3, "sigma8": 0.8},
    }
    (tmp_path / "test_metadata.json").write_text(json.dumps(metadata))

    chain_file, sampled_fiducial, meta, labels = sampler_mod.load_chain_metadata(str(tmp_path))

    assert chain_file == os.path.join(str(tmp_path), "run.txt")
    assert sampled_fiducial == {"Omegam": 0.3, "sigma8": 0.8}
    assert labels["Omegam"] == r"$\Omega_{m}$"
    assert labels["sigma8"] == r"$\sigma_8$"
    assert meta == metadata


def test_load_chain_metadata_requires_exactly_one_candidate(tmp_path):
    with pytest.raises(ValueError, match="Expected exactly one"):
        sampler_mod.load_chain_metadata(str(tmp_path))

    (tmp_path / "a_metadata.json").write_text("{}")
    (tmp_path / "b_metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="Expected exactly one"):
        sampler_mod.load_chain_metadata(str(tmp_path))


def test_load_chain_metadata_falls_back_to_outroot(tmp_path):
    outroot = str(tmp_path / "outroot")
    metadata = {"outroot path": outroot, "sampled_fiducial_params": {}}
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata))

    chain_file, *_ = sampler_mod.load_chain_metadata(str(tmp_path))

    assert chain_file == outroot + ".txt"


def test_load_chain_metadata_raises_when_chain_file_missing(tmp_path):
    metadata = {"sampled_fiducial_params": {}}
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Chain file not found"):
        sampler_mod.load_chain_metadata(str(tmp_path))


def test_load_chain_metadata_with_explicit_filename_and_label_overrides(tmp_path):
    metadata = {"chain_file": "abs.txt", "sampled_fiducial_params": {"h": 0.7}}
    (tmp_path / "custom.json").write_text(json.dumps(metadata))

    chain_file, sampled, meta, labels = sampler_mod.load_chain_metadata(
        str(tmp_path), metadata_filename="custom.json", label_overrides={"h": "Hubble"}
    )

    assert labels["h"] == "Hubble"
    assert sampled == {"h": 0.7}


def test_load_chain_metadata_preserves_absolute_chain_file(tmp_path):
    abs_path = str(tmp_path / "elsewhere" / "chain.txt")
    metadata = {"chain_file": abs_path, "sampled_fiducial_params": {}}
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata))

    chain_file, *_ = sampler_mod.load_chain_metadata(str(tmp_path))

    assert chain_file == abs_path


def test_setup_priors_handles_gaussian_and_tuple_priors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_context = _FakeContext(
        freeparams={"Omegam": None, "sigma8": None},
        allparams={"Omegam": 0.32, "sigma8": 0.8},
    )
    monkeypatch.setattr(sampler_mod, "build_analysis_context", lambda **kwargs: fake_context)
    monkeypatch.setattr(sampler_mod, "PhotometricLikelihood", _FakePhotometricLikelihood)
    monkeypatch.setattr(
        sampler_mod, "Sampler", lambda **kwargs: _FakeNautilusSampler(True, **kwargs)
    )

    config = {
        "name": "gauss_test",
        "fiducial": {},
        "observables": ["WL"],
        "options": {
            "code": "symbolic",
            "survey_name_photo": "Euclid",
            "survey_name_spectro": None,
            "cosmo_model": "LCDM",
            "survey_name": "Euclid",
        },
        "priors": {
            "Omegam": {"type": "gaussian", "loc": 0.3, "scale": 0.01},
            "sigma8": (0.6, 1.0),
            "not_free": (0, 1),
        },
        "sampler_settings": {
            "pool": 1,
            "n_live": 10,
            "n_networks": 1,
            "n_batch": 1,
        },
    }

    naut_sampler = sampler_mod.NautilusSampler(config)
    naut_sampler.run()

    assert set(naut_sampler.prior_chosen.keys) == {"Omegam", "sigma8"}
    with open(naut_sampler.outroot + "_metadata.json") as f:
        meta = json.load(f)
    assert meta["sampled_fiducial_params"] == {"Omegam": 0.32, "sigma8": 0.8}


def test_run_reports_existing_completed_chain_and_updates_metadata(tmp_path, monkeypatch):
    naut_sampler = _make_sampler(tmp_path, monkeypatch, run_return=True)
    chain_file = naut_sampler.outroot + ".txt"
    chain_hdf5 = naut_sampler.outroot + ".hdf5"
    open(chain_file, "w").close()
    open(chain_hdf5, "w").close()

    naut_sampler.run()

    assert os.path.exists(naut_sampler.outroot + "_metadata.json")
    with open(naut_sampler.outroot + "_metadata.json") as f:
        meta = json.load(f)
    assert meta["evidence_log_z"] == pytest.approx(-1.23)


def test_run_reports_existing_chain_without_hdf5_updates_metadata_only(tmp_path, monkeypatch):
    naut_sampler = _make_sampler(tmp_path, monkeypatch, run_return=True)
    chain_file = naut_sampler.outroot + ".txt"
    open(chain_file, "w").close()

    naut_sampler.run()

    assert os.path.exists(naut_sampler.outroot + "_metadata.json")
    with open(naut_sampler.outroot + "_metadata.json") as f:
        meta = json.load(f)
    assert meta["evidence_log_z"] is None


def test_run_handles_error_loading_existing_run(tmp_path, monkeypatch, capsys):
    naut_sampler = _make_sampler(tmp_path, monkeypatch, run_return=True)
    chain_file = naut_sampler.outroot + ".txt"
    chain_hdf5 = naut_sampler.outroot + ".hdf5"
    open(chain_file, "w").close()
    open(chain_hdf5, "w").close()

    def broken_sampler(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sampler_mod, "Sampler", broken_sampler)

    naut_sampler.run()

    captured = capsys.readouterr()
    assert "WARNING: Could not load existing run" in captured.out


def test_start_sampler_failure_terminates_worker_pool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sampler_mod, "build_analysis_context", lambda **kwargs: _FakeContext())
    monkeypatch.setattr(sampler_mod, "PhotometricLikelihood", _FakePhotometricLikelihood)

    fake_pool = _FakePool()
    fake_context = _FakeSpawnContext(fake_pool)
    monkeypatch.setattr(sampler_mod.multiprocessing, "get_context", lambda method: fake_context)

    def broken_sampler(**kwargs):
        raise RuntimeError("sampler init failed")

    monkeypatch.setattr(sampler_mod, "Sampler", broken_sampler)

    config = {
        "name": "failure_test",
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

    with pytest.raises(RuntimeError, match="sampler init failed"):
        naut_sampler.run()

    assert fake_pool.terminated
    assert fake_pool.joined


def test_run_forwards_optional_sampler_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sampler_mod, "build_analysis_context", lambda **kwargs: _FakeContext())
    monkeypatch.setattr(sampler_mod, "PhotometricLikelihood", _FakePhotometricLikelihood)

    captured_run_kwargs = {}

    class CapturingSampler(_FakeNautilusSampler):
        def run(self, **kwargs):
            captured_run_kwargs.update(kwargs)
            return super().run(**kwargs)

    monkeypatch.setattr(sampler_mod, "Sampler", lambda **kwargs: CapturingSampler(True, **kwargs))

    config = {
        "name": "optional_settings_test",
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
            "n_eff": 500,
        },
    }

    naut_sampler = sampler_mod.NautilusSampler(config)
    naut_sampler.run()

    assert captured_run_kwargs["n_eff"] == 500
