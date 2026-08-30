import pytest

from cosmicfishpie.configs.config import _add_free_ia_parameters


IA_PARAMS = {"IA_model": "eNLA", "AIA": 1.72, "betaIA": 2.17, "etaIA": -0.41}


def test_scalar_ia_variation_preserves_legacy_behavior():
    freeparams = {}

    _add_free_ia_parameters(freeparams, IA_PARAMS, 0.035)

    assert freeparams == {"AIA": 0.035, "betaIA": 0.035, "etaIA": 0.035}


def test_mapping_ia_variation_can_fix_betaia():
    freeparams = {}

    _add_free_ia_parameters(freeparams, IA_PARAMS, {"AIA": 0.035, "etaIA": 0.035})

    assert freeparams == {"AIA": 0.035, "etaIA": 0.035}


def test_mapping_ia_variation_rejects_unknown_parameters():
    with pytest.raises(ValueError, match="Unknown intrinsic-alignment parameter"):
        _add_free_ia_parameters({}, IA_PARAMS, {"not_an_ia_parameter": 0.01})
