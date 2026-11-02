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
