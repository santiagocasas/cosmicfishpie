"""Legacy-vs-JAX parity of the photometric ``C_ell`` for ``GCph`` + ``WL`` (jaxcosmo extra).

:mod:`cosmicfishpie.LSSsurvey.jaxcosmo_photo` is a ``jax.numpy`` re-implementation of
:class:`cosmicfishpie.LSSsurvey.photo_obs.ComputeCls`.  Two differences are deliberate and
documented in the module: ``P(k, z)`` is evaluated directly at the Limber wavenumbers instead of
being read off a coarse bicubic table, and the luminosity-ratio table is clamped instead of
raising outside its range.  Everything else (``z``/``ell`` grids, photo-z convolution, kernels,
trapezoid rules, masks, dictionary keys and their order) follows ``ComputeCls`` literally, so
the two spectra must agree up to the difference between the two power-spectrum routes.

The parity is checked at the fiducial point, at a shifted cosmology (which moves ``chi``, the
growth factor and the Limber wavenumbers at once) and at shifted bias / intrinsic-alignment
amplitudes.  The fiducial point alone would hide errors in those branches.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_cosmo")

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compare_jax_photo_fishers", ROOT / "scripts/validation/compare_jax_photo_fishers.py"
)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)

#: max |dC| / max|C| over every spectrum (dominated by the power-spectrum route difference).
ATOL = 1.0e-3
#: max |dC| / |C| where |C| > 1% of max|C| (the observed agreement is ~5e-5).
RTOL = 1.0e-3

#: Multiplicative shifts applied to the merged fiducial dictionary of each case.
CASES = {
    "fiducial": {},
    "cosmology": {"10^9As": 1.05, "Omegam": 1.06, "Omegab": 1.03, "h": 1.04, "ns": 1.02},
}

#: Multiplicative shifts of the bias and intrinsic-alignment amplitudes.  These fiducials come
#: from the survey specification rather than from ``fiducialpars``, so the legacy observable has
#: to be given modified dictionaries (see ``compare._legacy_observable``).
NUISANCE_SHIFTS = {"b1": 1.20, "AIA": 0.70}


def _assert_cls_matches_legacy(legacy: dict, jax_cls: dict) -> None:
    """Compare every spectrum of the legacy and JAX dictionaries."""
    assert list(jax_cls) == list(legacy), "spectrum dictionary keys or their order differ"
    np.testing.assert_allclose(jax_cls["ells"], legacy["ells"], rtol=0.0, atol=0.0)

    keys = [key for key in legacy if key != "ells"]
    reference = np.concatenate([np.ravel(legacy[key]) for key in keys])
    candidate = np.concatenate([np.ravel(jax_cls[key]) for key in keys])
    assert reference.shape == candidate.shape

    scale = float(np.max(np.abs(reference)))
    assert scale > 0.0
    assert float(np.max(np.abs(candidate - reference))) < ATOL * scale

    significant = np.abs(reference) > 1.0e-2 * scale
    assert significant.any()
    relative = np.abs(candidate[significant] - reference[significant]) / np.abs(
        reference[significant]
    )
    assert float(np.max(relative)) < RTOL


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Legacy observable and JAX observable at the fiducial point (built once)."""
    from cosmicfishpie.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable

    tmp_path = tmp_path_factory.mktemp("photo_parity")
    calc = compare._make_calc(tmp_path, "fiducial")
    allpars = dict(calc.allparams_fidus)
    return calc, compare._legacy_observable(calc), allpars, JaxPhotoObservable(calc, allpars)


@pytest.mark.parametrize("name", list(CASES))
def test_cls_matches_legacy(name, reference, tmp_path):
    from cosmicfishpie.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable

    base_calc, base_legacy, base_allpars, _ = reference
    shifts = CASES[name]
    fiducial = {
        key: (value * shifts[key] if key in shifts else value)
        for key, value in base_allpars.items()
    }
    if not shifts:
        calc, legacy = base_calc, base_legacy
    else:
        calc = compare._make_calc(tmp_path, name, fiducial=fiducial)
        legacy = compare._legacy_observable(calc)

    _assert_cls_matches_legacy(legacy.result, JaxPhotoObservable(calc, fiducial).cls())


def test_cls_matches_legacy_at_shifted_bias_and_ia(reference, tmp_path):
    from cosmicfishpie.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable

    base_calc, _, base_allpars, _ = reference
    calc = compare._make_calc(tmp_path, "nuisance")
    photobiaspars = dict(calc.photobiaspars)
    iapars = dict(calc.IApars)
    for key, shift in NUISANCE_SHIFTS.items():
        target = photobiaspars if key in photobiaspars else iapars
        assert key in target, f"fiducial dictionary does not carry '{key}'"
        target[key] = target[key] * shift
    legacy = compare._legacy_observable(calc, photobiaspars=photobiaspars, iapars=iapars)

    fiducial = {
        key: (value * NUISANCE_SHIFTS[key] if key in NUISANCE_SHIFTS else value)
        for key, value in base_allpars.items()
    }
    _assert_cls_matches_legacy(legacy.result, JaxPhotoObservable(calc, fiducial).cls())


def test_call_replays_the_full_fiducial_dictionary(reference):
    """The finite-difference engine calls the observable with the whole dictionary."""
    _, _, allpars, observable = reference
    direct = observable.cls()
    replayed = observable(dict(allpars))
    assert list(replayed) == list(direct)
    for key in direct:
        np.testing.assert_allclose(replayed[key], direct[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("name", ["Omegam", "b1"])
def test_jacobian_matches_finite_differences(name, reference):
    """The autodiff derivatives agree with central differences of the same observable."""
    _, _, allpars, observable = reference
    value = float(allpars[name])
    step = 1.0e-4 * value

    jac = observable.jacobian([name])[name]
    plus = observable.cls({name: value + step})
    minus = observable.cls({name: value - step})
    keys = [key for key in plus if key != "ells"]
    scale = max(float(np.max(np.abs(plus[key] - minus[key]))) for key in keys)
    assert scale > 0.0
    for key in keys:
        finite_difference = (plus[key] - minus[key]) / (2.0 * step)
        residual = float(np.max(np.abs(jac[key] - finite_difference)))
        assert residual < 1.0e-4 * scale / (2.0 * step)
