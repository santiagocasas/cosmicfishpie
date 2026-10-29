"""Tests for the cache and pickling behaviour of the jaxcosmo derivative providers."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_cosmo")

from cosmicfishpie.fishermatrix import jaxcosmo_derivatives as jd  # noqa: E402
from cosmicfishpie.LSSsurvey import jaxcosmo_photo  # noqa: E402


class _StubObservable:
    built = 0

    def __init__(self, configuration, fiducial):
        type(self).built += 1
        self.configuration = configuration
        self.fiducial = fiducial


@pytest.fixture
def stub(monkeypatch):
    _StubObservable.built = 0
    monkeypatch.setattr(jaxcosmo_photo, "JaxPhotoObservable", _StubObservable)
    return _StubObservable


def _request(fiducial=None, settings=None, specs=None, observables=("GCph", "WL")):
    configuration = SimpleNamespace(
        settings=settings or {"code": "jaxcosmo", "accuracy": 1, "nonlinear": True},
        specs=specs or {"kmax": 30.0, "z_bins_WL": np.array([0.0, 1.0, 2.0])},
        obs=list(observables),
    )
    return SimpleNamespace(
        fiducial=fiducial or {"h": 0.67, "Omegam": 0.32, "bias_model": "binned"},
        configuration=configuration,
        observables_type=tuple(observables),
    )


def test_observable_is_reused_for_identical_requests(stub):
    provider = jd.JaxAutodiffProvider()
    first = provider._get_observable(_request())
    second = provider._get_observable(_request())
    assert first is second
    assert stub.built == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"fiducial": {"h": 0.70, "Omegam": 0.32, "bias_model": "binned"}},
        {"settings": {"code": "jaxcosmo", "accuracy": 2, "nonlinear": True}},
        {"settings": {"code": "jaxcosmo", "accuracy": 1, "nonlinear": False}},
        {"specs": {"kmax": 10.0, "z_bins_WL": np.array([0.0, 1.0, 2.0])}},
        {"specs": {"kmax": 30.0, "z_bins_WL": np.array([0.0, 1.0, 3.0])}},
        {"observables": ("WL",)},
    ],
)
def test_observable_is_rebuilt_when_anything_relevant_changes(stub, changed):
    provider = jd.JaxFiniteDifferenceProvider()
    first = provider._get_observable(_request())
    second = provider._get_observable(_request(**changed))
    assert first is not second
    assert stub.built == 2


def test_cache_sees_in_place_mutation_of_the_same_configuration_object(stub):
    provider = jd.JaxAutodiffProvider()
    request = _request()
    first = provider._get_observable(request)
    request.configuration.settings["accuracy"] = 3  # legacy cfg module is mutated in place
    assert provider._get_observable(request) is not first


def test_wrong_backend_code_is_rejected(stub):
    provider = jd.JaxAutodiffProvider()
    with pytest.raises(ValueError, match="jaxcosmo"):
        provider._get_observable(_request(settings={"code": "camb"}))


def test_provider_pickles_without_compiled_state(stub):
    provider = jd.JaxAutodiffProvider()
    provider._get_observable(_request())
    clone = pickle.loads(pickle.dumps(provider))
    assert clone._observable is None
    assert clone._cache_key is None
