"""Tests for the derivative results, cache and pickling behaviour of the jaxcosmo providers."""

from __future__ import annotations

import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_cosmo")

from cosmicfishpie.fishermatrix import jaxcosmo_derivatives as jd  # noqa: E402
from cosmicfishpie.LSSsurvey import jaxcosmo_photo  # noqa: E402


class _StubObservable:
    built = 0

    def __init__(self, configuration, fiducial, observables=None):
        type(self).built += 1
        self.configuration = configuration
        self.fiducial = fiducial
        self.observables = observables

    def jacobian(self, names):
        return {name: f"d({name})" for name in names}


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


def _derivative_request(configuration, freeparams, special=None):
    return jd.DerivativeRequest(
        observable=lambda fiducial: None,
        fiducial={"h": 0.67, "Omegam": 0.32, "bias_model": "binned"},
        freeparams=freeparams,
        observables_type=("GCph", "WL"),
        external_settings=None,
        feed_lvl=0,
        method="jax",
        special_deriv_function=special,
        configuration=configuration,
    )


def test_observable_gets_the_requested_probe_subset(stub):
    configuration = _request().configuration
    request = replace(_derivative_request(configuration, ("h",)), observables_type=("WL",))
    observable = jd.JaxAutodiffProvider()._get_observable(request)
    assert observable.observables == ("WL",)


def test_autodiff_compute_keeps_special_derivatives_and_adds_the_jacobian(stub):
    provider = jd.JaxAutodiffProvider()
    configuration = _request().configuration
    request = _derivative_request(
        configuration, ("b1", "h"), lambda name: "analytic-b1" if name == "b1" else None
    )
    result = provider.compute(request)
    assert list(result) == ["b1", "h"]
    assert result["b1"] == "analytic-b1"
    assert result["h"] == "d(h)"


def test_finite_difference_compute_delegates_to_the_standard_engine(stub, monkeypatch):
    seen = {}

    class _Engine:
        def compute(self, request):
            seen["request"] = request
            return {name: ("fd", name) for name in request.freeparams}

    monkeypatch.setattr(jd, "FiniteDifferenceProvider", _Engine)
    provider = jd.JaxFiniteDifferenceProvider()
    configuration = _request().configuration
    request = _derivative_request(configuration, ("h",))
    result = provider.compute(request)
    assert result == {"h": ("fd", "h")}
    assert seen["request"].observable is provider._observable
