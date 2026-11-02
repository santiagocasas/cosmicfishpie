"""Legacy-vs-JAX parity of the spectroscopic ``ln P`` for the option branches (jaxcosmo extra).

Each case builds the legacy ``ComputeGalSpectro`` and the JAX observable on the same options and
compares ``ln P`` at the fiducial point *and* at a perturbed cosmology (the fiducial point alone
hides errors in the Alcock-Paczynski and h-rescaling branches).  The legacy power spectrum comes
from a SciPy spline that clamps at the edge of its table, so the JAX observable is run with
``legacy_pk_clamp`` for the cases where the perturbed grid leaves that table.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("jax")
pytest.importorskip("jax_cosmo")

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compare_jax_spectro_fishers", ROOT / "scripts/validation/compare_jax_spectro_fishers.py"
)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)

# max |ln P_jax - ln P_legacy| (absolute); the observed agreement is ~3e-5.
ATOL = 2.0e-4

CASES = {
    "default": {},
    "ap_off": {"AP_effect": False},
    "h_bug_after_error": {"kh_rescaling_bug": True},
    "h_bug_before_error": {"kh_rescaling_bug": True, "kh_rescaling_beforespecerr_bug": True},
    "linear": {"GCsp_linear": True},
    "fog_off": {"FoG_switch": False},
}


@pytest.fixture(scope="module")
def clamp_yaml(tmp_path_factory):
    import cosmicfishpie

    src = Path(cosmicfishpie.__file__).parent / "configs/default_boltzmann_yaml_files"
    params = yaml.safe_load((src / "jaxcosmo/default.yaml").read_text())
    params["NUMERICS"]["legacy_pk_clamp"] = True
    path = tmp_path_factory.mktemp("yaml") / "jaxcosmo_clamp.yaml"
    path.write_text(yaml.safe_dump(params))
    return str(path)


@pytest.mark.parametrize("name", list(CASES))
def test_lnp_matches_legacy_at_fiducial_and_perturbed_cosmology(name, clamp_yaml, tmp_path):
    from cosmicfishpie.LSSsurvey.jaxcosmo_spectro import JaxSpectroObservable

    extra = dict(CASES[name], jaxcosmo_config_yaml=clamp_yaml)
    calc = compare._make_calc(tmp_path, name, extra_options=extra)
    engine = compare._legacy_engine(calc)
    fiducial = dict(calc.pk_obs_fid.fiducial_allpars)
    observable = JaxSpectroObservable(calc, fiducial)
    assert observable.legacy_pk_clamp

    perturbed = dict(fiducial)
    for key, factor in (("Omegam", 1.06), ("h", 1.04), ("ns", 1.02), ("10^9As", 1.05)):
        perturbed[key] *= factor

    for point in (fiducial, perturbed):
        legacy = engine.get_obs(point)
        jax_out = observable.lnp(point)
        for ibin in range(observable.nbins):
            assert np.all(np.isfinite(jax_out[ibin]))
            assert np.max(np.abs(jax_out[ibin] - legacy[ibin])) < ATOL


def test_unclamped_default_agrees_where_the_grid_stays_inside_the_table(tmp_path):
    """Direct evaluation (the default) equals the legacy value away from the table edge."""
    from cosmicfishpie.LSSsurvey.jaxcosmo_spectro import JaxSpectroObservable

    calc = compare._make_calc(tmp_path, "direct")
    engine = compare._legacy_engine(calc)
    fiducial = dict(calc.pk_obs_fid.fiducial_allpars)
    observable = JaxSpectroObservable(calc, fiducial)
    assert not observable.legacy_pk_clamp
    legacy = engine.get_obs(fiducial)
    jax_out = observable.lnp(fiducial)
    for ibin in range(observable.nbins):
        assert np.max(np.abs(jax_out[ibin] - legacy[ibin])) < ATOL
