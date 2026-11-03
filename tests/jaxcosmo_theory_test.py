"""Tests for the differentiable ``jaxcosmo`` theory layer (optional ``jaxcosmo`` extra)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_cosmo")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_cosmo as jc  # noqa: E402
from jax_cosmo import power as jc_power  # noqa: E402

from cosmicfishpie.cosmology import jaxcosmo_halofit as halofit  # noqa: E402
from cosmicfishpie.cosmology import jaxcosmo_theory as theory  # noqa: E402
from cosmicfishpie.cosmology.jaxcosmo_amplitude import (  # noqa: E402
    As_to_sigma8,
    sigma8_to_As,
)

theory.enable_x64()

FIDUCIAL = {"As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96, "w0": -1.0, "wa": 0.0}
# corners of the box covered by the validation campaigns
LOW = {"As": 1.7, "Omegam": 0.24, "Omegab": 0.06, "h": 0.73, "ns": 0.92, "w0": -1.3, "wa": -0.7}
HIGH = {"As": 2.5, "Omegam": 0.40, "Omegab": 0.04, "h": 0.61, "ns": 1.00, "w0": -0.7, "wa": 0.5}
TRANSFER = jc.transfer.Eisenstein_Hu


# ------------------------------------------------------------------------------- amplitude
def test_amplitude_roundtrip_and_symbolic_pofk_agreement():
    symbolic = pytest.importorskip("symbolic_pofk.linear_new")
    for p in (FIDUCIAL, LOW, HIGH):
        args = (p["Omegam"], p["Omegab"], p["h"], p["ns"], 0.0, p["w0"], p["wa"])
        s8 = float(As_to_sigma8(p["As"], *args))
        expected = float(symbolic.As_to_sigma8_max_precision(p["As"], *args))
        assert s8 == pytest.approx(expected, rel=1e-12)
        assert float(sigma8_to_As(s8, *args)) == pytest.approx(p["As"], rel=1e-9)


def test_amplitude_gradient_matches_finite_difference():
    def s8(om):
        return As_to_sigma8(2.1, om, 0.05, 0.67, 0.96, 0.0, -1.0, 0.0)

    eps = 1e-6
    fd = (s8(0.32 + eps) - s8(0.32 - eps)) / (2 * eps)
    assert float(jax.grad(s8)(0.32)) == pytest.approx(float(fd), rel=1e-6)


# ---------------------------------------------------------------------------------- halofit
@pytest.mark.parametrize("params", [FIDUCIAL, LOW, HIGH], ids=["fiducial", "low", "high"])
@pytest.mark.parametrize("z", [0.0, 3.0])
def test_newton_solve_converges_inside_the_bracket(params, z):
    cosmo = theory.make_cosmology(params)
    residual, inside = halofit.nonlinear_scale_residual(cosmo, 1.0 / (1.0 + z), TRANSFER)
    assert bool(inside)
    assert float(residual) < 1e-10


def test_halofit_agrees_with_jax_cosmo_to_the_accuracy_of_its_root_find():
    cosmo = theory.make_cosmology(FIDUCIAL)
    k = jnp.geomspace(1e-3, 9.0, 40)
    for z in (0.0, 1.0):
        a = 1.0 / (1.0 + z)
        ours = halofit.halofit(cosmo, k, a, TRANSFER)
        reference = jc_power.halofit(cosmo, k, a, TRANSFER)
        assert float(jnp.max(jnp.abs(ours / reference - 1.0))) < 3e-3


def test_halofit_autodiff_matches_finite_difference_quadratically():
    """AD equals the central difference at every step size on a fine redshift grid."""
    k = jnp.array([0.5, 2.0])
    z = jnp.linspace(0.0, 2.5, 12)

    def pnl(omegam):
        cosmo = theory.make_cosmology({**FIDUCIAL, "Omegam": omegam})
        return jax.vmap(lambda a: halofit.halofit(cosmo, k, a, TRANSFER))(1.0 / (1.0 + z))

    pnl = jax.jit(pnl)
    grad = np.asarray(jax.jit(jax.jacfwd(pnl))(0.32))
    for eps in (1e-2, 1e-3):
        fd = np.asarray((pnl(0.32 + eps) - pnl(0.32 - eps)) / (2 * eps))
        # the error must shrink like eps^2 (quadratic truncation), not saturate
        assert np.max(np.abs(grad / fd - 1.0)) < 20 * eps**2 + 1e-8


# ------------------------------------------------------------------ provider parameter maps
def test_resolve_parameters_rejects_theta_like_expansion_parameters():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "ns": 0.96}
    for key in ("theta", "cosmomc_theta", "thetastar"):
        with pytest.raises(ValueError, match=key):
            provider.resolve_parameters(dict(base, **{key: 1.04}), "LCDM")


def test_resolve_parameters_rejects_conflicting_amplitude_aliases():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"Omegam": 0.32, "Omegab": 0.05, "ns": 0.96}
    for aliases in ({"10^9As": 2.1, "As": 2.1e-9}, {"10^9As": 2.1, "sigma8": 0.8}):
        with pytest.raises(ValueError, match="Ambiguous primordial amplitude"):
            provider.resolve_parameters(dict(base, **aliases), "LCDM")


def test_resolve_parameters_rejects_non_positive_sigma8():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    with pytest.raises(ValueError, match="sigma8 > 0"):
        provider.resolve_parameters({"sigma8": -0.8, "Omegam": 0.32, "Omegab": 0.05}, "LCDM")


def test_resolve_parameters_accepts_a_single_amplitude_alias():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    resolved = provider.resolve_parameters({"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05}, "LCDM")
    assert resolved["As"] == pytest.approx(2.1)


# ---------------------------------------------------------------------------- theory tables
def test_halofit_keeps_a_single_wavenumber_axis():
    tab = theory.compute_tables(
        FIDUCIAL, jnp.array([0.0, 0.5]), jnp.array([0.3]), theory.TheoryConfig(), True
    )
    assert tab["Pnl"].shape == (2, 1)


def test_tables_units_shapes_and_normalisation():
    z = jnp.array([0.0, 0.5, 1.0, 2.0])
    k = jnp.geomspace(1e-3, 5.0, 16)
    tab = theory.compute_tables(FIDUCIAL, z, k, theory.TheoryConfig(), True)
    c = theory.SPEED_OF_LIGHT_KM_S
    assert float(tab["H_over_c"][0]) == pytest.approx(100.0 * FIDUCIAL["h"] / c, rel=1e-12)
    assert float(tab["D"][0]) == pytest.approx(1.0, abs=1e-12)
    assert float(tab["chi"][0]) == 0.0
    assert tab["Pnl"].shape == tab["Plin"].shape == (4, 16)
    ratio = tab["Pnl"] / tab["Plin"]
    assert bool(jnp.all(jnp.abs(ratio[:, 0] - 1.0) < 1e-2))  # linear on large scales
    assert bool(jnp.all(ratio[:, -1] > 1.5))  # strongly nonlinear at k = 5 h/Mpc
    assert float(ratio.min()) > 0.97  # the known ~1% Halofit dip at quasi-linear scales
    # comoving distance against astropy (jax-cosmo tabulates distances; ~2e-4 accuracy)
    cosmology = pytest.importorskip("astropy.cosmology")
    ref = cosmology.Flatw0waCDM(
        H0=100 * FIDUCIAL["h"], Om0=FIDUCIAL["Omegam"], w0=-1.0, wa=0.0, Tcmb0=0.0
    )
    chi_ref = ref.comoving_distance(np.asarray(z[1:])).value
    np.testing.assert_allclose(np.asarray(tab["chi"][1:]), chi_ref, rtol=5e-4)


# ------------------------------------------------------- provider guards added in review
def test_resolve_parameters_rejects_every_massive_neutrino_alias():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "ns": 0.96}
    for key in ("mnu", "m_nu", "Omeganu", "omnuh2"):
        with pytest.raises(ValueError, match="massless neutrinos"):
            provider.resolve_parameters(dict(base, **{key: 0.1}), "LCDM")
    # A zero in one alias must not hide a massive value in another.
    with pytest.raises(ValueError, match="massless neutrinos"):
        provider.resolve_parameters(dict(base, mnu=0.0, m_nu=0.1), "LCDM")


def test_resolve_parameters_rejects_ignored_density_parameterizations():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1, "Omegab": 0.05}
    for key in ("omega_cdm", "Omega_cdm"):
        with pytest.raises(ValueError, match="does not read"):
            provider.resolve_parameters(dict(base, **{key: 0.12}), "LCDM")
    base = {"10^9As": 2.1, "Omegam": 0.32}
    for key in ("omega_b", "Omega_b", "100omega_b"):
        with pytest.raises(ValueError, match="does not read"):
            provider.resolve_parameters(dict(base, **{key: 0.022}), "LCDM")
    resolved = provider.resolve_parameters(
        {"10^9As": 2.1, "omch2": 0.12, "Omegab": 0.05, "h": 0.67}, "LCDM"
    )
    assert resolved["Omegam"] == pytest.approx(0.12 / 0.67**2 + 0.05)


def test_resolve_parameters_rejects_conflicting_density_and_expansion_aliases():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67}
    with pytest.raises(ValueError, match="Conflicting values for Omegab, ombh2"):
        provider.resolve_parameters(dict(base, ombh2=0.022), "LCDM")
    with pytest.raises(ValueError, match="Conflicting values for Omegam, omch2"):
        provider.resolve_parameters(dict(base, omch2=0.12), "LCDM")
    consistent = provider.resolve_parameters(dict(base, ombh2=0.05 * 0.67**2), "LCDM")
    assert consistent["Omegab"] == pytest.approx(0.05)
    consistent = provider.resolve_parameters(dict(base, omch2=(0.32 - 0.05) * 0.67**2), "LCDM")
    assert consistent["Omegam"] == pytest.approx(0.32)
    for alias in ("omega_b", "Omega_b", "100omega_b", "omega_cdm", "Omega_cdm"):
        with pytest.raises(ValueError, match="does not read"):
            provider.resolve_parameters(dict(base, **{alias: 0.02}), "LCDM")
    with pytest.raises(ValueError, match="Conflicting values for h and H0"):
        provider.resolve_parameters(dict(base, H0=70.0), "LCDM")


def test_resolve_parameters_rejects_invalid_h_and_h0():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05}
    for key, value in (("h", 0.0), ("h", float("nan")), ("H0", float("inf"))):
        with pytest.raises(ValueError, match="positive, finite h"):
            provider.resolve_parameters(dict(base, **{key: value}), "LCDM")


def test_resolve_parameters_rejects_conflicting_w0wa_aliases():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05}
    for aliases in ({"w0": -0.8, "w0_fld": -0.9}, {"wa": 0.2, "wa_fld": 0.3}):
        with pytest.raises(ValueError, match="Conflicting values"):
            provider.resolve_parameters(dict(base, **aliases), "w0waCDM")


@pytest.mark.parametrize("key", ["Omegab", "Omegam", "ombh2", "omch2"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_resolve_parameters_rejects_non_finite_density(key, value):
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    pars = {"10^9As": 2.1, key: value}
    with pytest.raises(ValueError, match="Omegam > Omegab > 0"):
        provider.resolve_parameters(pars, "LCDM")


def test_resolve_parameters_checks_spectral_index_aliases():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1}
    with pytest.raises(ValueError, match="Conflicting values for ns and n_s"):
        provider.resolve_parameters(dict(base, ns=0.96, n_s=0.98), "LCDM")
    for aliases in ({"ns": 0.97}, {"n_s": 0.97}, {"ns": 0.97, "n_s": 0.97}):
        assert provider.resolve_parameters(dict(base, **aliases), "LCDM")["ns"] == 0.97


@pytest.mark.parametrize("key", ["w0", "wa", "w0_fld", "wa_fld"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_resolve_parameters_rejects_non_finite_dark_energy(key, value):
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    with pytest.raises(ValueError, match="requires a finite"):
        provider.resolve_parameters({"10^9As": 2.1, key: value}, "w0waCDM")


def test_resolve_parameters_rejects_non_finite_amplitude():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"Omegam": 0.32, "Omegab": 0.05}
    for weak in ({"10^9As": float("nan")}, {"10^9As": -1.0}, {"logAs": 1000.0}):
        with pytest.raises(ValueError, match="positive, finite"):
            provider.resolve_parameters(dict(base, **weak), "LCDM")


def test_resolve_parameters_rejects_conflicting_lcdm_dark_energy():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    base = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05}
    for key, value in (("w0", -0.8), ("wa", 0.3), ("w0_fld", -0.9), ("wa_fld", 0.2)):
        with pytest.raises(ValueError, match="cosmo_model='LCDM'"):
            provider.resolve_parameters(dict(base, **{key: value}), "LCDM")
    resolved = provider.resolve_parameters(dict(base, w0=-1.0, wa=0.0), "LCDM")
    assert (resolved["w0"], resolved["wa"]) == (-1.0, 0.0)


def test_theory_tables_reject_conflicting_lcdm_dark_energy():
    params = dict(FIDUCIAL, w0=-0.8)
    cosmology = theory.JaxCosmoTheory(cosmo_model="LCDM")
    with pytest.raises(ValueError, match="cosmo_model='LCDM'"):
        cosmology.tables(params, jnp.array([0.0]), jnp.array([0.1]))
    with pytest.raises(ValueError, match="cosmo_model='LCDM'"):
        cosmology.power(params, jnp.array([0.0]), jnp.array([0.1]))


def test_distance_steps_must_be_an_integer_in_the_accuracy_block():
    with pytest.raises(ValueError, match="distance_steps"):
        theory.TheoryConfig.from_backend_parameters({"ACCURACY": {"distance_steps": 256.9}})
    config = theory.TheoryConfig.from_backend_parameters({"ACCURACY": {"distance_steps": 64}})
    assert config.distance_steps == 64


def _backend(numerics: dict) -> dict:
    base = {
        "kmin_pk": 1.0e-3,
        "kmax_pk": 9.0,
        "zmin_pk": 0.0,
        "zmax_pk": 3.0,
        "z_samples": 100,
        "k_samples": 400,
        "x64": True,
    }
    return {"ACCURACY": {}, "NUMERICS": dict(base, **numerics)}


def test_numerics_grids_need_enough_nodes_for_cubic_splines():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    cosmopars = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96}
    for key in ("z_samples", "k_samples", "z_samples_background"):
        with pytest.raises(ValueError, match=key):
            provider.JaxCosmoProvider(cosmopars, _backend({key: 3}), "LCDM")


def test_numerics_sample_counts_must_be_integers():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    cosmopars = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96}
    for key in ("z_samples", "k_samples", "z_samples_background"):
        with pytest.raises(ValueError, match="integer >= 4"):
            provider.JaxCosmoProvider(cosmopars, _backend({key: 12.5}), "LCDM")


def test_integral_float_sample_counts_are_normalized_for_numpy():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    keys = ("z_samples", "k_samples", "z_samples_background")
    backend = _backend(dict.fromkeys(keys, 4.0))
    cosmology = provider.JaxCosmoProvider({"10^9As": 2.1}, backend, "LCDM")
    for key in keys:
        assert isinstance(cosmology.numerics[key], int)
        assert np.linspace(0.0, 1.0, cosmology.numerics[key]).shape == (4,)
    assert backend["NUMERICS"]["z_samples"] == 4.0


def test_provider_rejects_a_cosmology_whose_halofit_root_is_outside_the_bracket():
    from cosmicfishpie.cosmology import jaxcosmo_provider as provider

    cosmology = theory.make_cosmology(dict(FIDUCIAL, As=1.0e-4))
    residual, inside = halofit.nonlinear_scale_residual(cosmology, 1.0, TRANSFER)
    assert not bool(inside)
    assert not np.isfinite(float(residual))

    cosmopars = {"10^9As": 1.0e-4, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96}
    with pytest.raises(ValueError, match="not finite"):
        provider.JaxCosmoProvider(cosmopars, _backend({}), "LCDM").build()
