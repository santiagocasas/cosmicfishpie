"""Numerical contracts for backend units, tracer recipes and optional emulators."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM

from cosmicfishpie.cosmology.cosmology import (
    _class_cb_subtraction_grid,
    cosmo_functions,
)
from cosmicfishpie.cosmology.ee2 import class_ee2_boost, class_ee2_parameters
from cosmicfishpie.cosmology.symbolic_new import SyrenNewProvider


@pytest.mark.parametrize("f_nu", [0.0, 0.02])
def test_cb_reconstruction_recovers_known_spectrum_on_a_different_grid(f_nu):
    """Check subtraction, density weighting and descending CLASS redshift ordering."""
    kt = np.array([0.01, 0.1, 0.2, 0.4])
    zt = np.array([2.0, 1.2, 0.6, 0.0])
    k = np.array([0.07, 0.15, 0.3])
    z = np.array([1.5, 0.4])
    f_b, f_cdm = 0.2, 0.8 - f_nu
    f_cb = 1.0 - f_nu

    def transfers(k, z):
        db = 2.0 + k[:, None] + z[None, :]
        dc = 1.0 + 0.5 * k[:, None] + 0.3 * z[None, :]
        dn = 0.2 + 0.1 * k[:, None] + 0.05 * z[None, :]
        return db, dc, dn

    class AnalyticClass:
        def Omega_b(self):
            return 0.3 * f_b

        def Omega0_cdm(self):
            return 0.3 * f_cdm

        def Omega_m(self):
            return 0.3

        def get_transfer_and_k_and_z(self):
            db, dc, dn = transfers(kt, zt)
            return {"d_b": db, "d_cdm": dc, "d_ncdm[0]": dn}, kt, zt

        def get_primordial(self):
            # Dimensionless primordial power giving 2*pi^2*P_prim/k^3 = 1.
            return {"k [1/Mpc]": kt, "P_scalar(k)": kt**3 / (2 * np.pi**2)}

    expected_cb = 100.0 + 3.0 * k[:, None] + 2.0 * z[None, :]
    db, dc, dn = transfers(k, z)
    cb_transfer = (f_b * db + f_cdm * dc) / f_cb
    total_power = f_cb**2 * expected_cb + 2 * f_cb * f_nu * cb_transfer * dn + f_nu**2 * dn**2
    recovered = _class_cb_subtraction_grid(AnalyticClass(), total_power, k, z)
    np.testing.assert_allclose(recovered, expected_cb, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("amplitude", [{"10^9As": 2.1}, {"As": 2.1e-9}, {"logAs": np.log(21)}])
def test_syren_amplitude_and_density_aliases_are_equivalent(amplitude):
    inputs = dict(H0=67.0, ombh2=0.022, omch2=0.12, m_nu=0.06, n_s=0.965, **amplitude)
    original = dict(inputs)
    resolved = SyrenNewProvider._resolve_parameters(inputs, "LCDM")
    assert inputs == original
    assert resolved["As"] == pytest.approx(2.1)
    assert resolved["h"] == 0.67
    assert resolved["Omegab"] == pytest.approx(0.022 / 0.67**2)
    assert resolved["Omegam"] == pytest.approx((0.022 + 0.12 + 0.06 / 94.07) / 0.67**2)
    assert resolved["ns"] == 0.965


def test_syren_explicit_amplitude_takes_precedence_over_sigma8():
    resolved = SyrenNewProvider._resolve_parameters({"10^9As": 2.1, "sigma8": 0.1}, "LCDM")
    assert resolved["As"] == 2.1
    assert resolved["Omegam"] == 0.32


@pytest.mark.parametrize("densities", [{"Omegab": 0.0}, {"Omegam": 0.01}, {"Omegab": -0.01}])
def test_syren_rejects_unphysical_density_budget(densities):
    with pytest.raises(ValueError, match="Omegam > Omegab"):
        SyrenNewProvider._resolve_parameters({"As": 2.1e-9, **densities}, "LCDM")


def test_syren_rejects_unsupported_cosmological_models():
    with pytest.raises(ValueError, match="does not support"):
        SyrenNewProvider._resolve_parameters({"As": 2.1e-9}, "curvedLCDM")


@pytest.mark.parametrize("model", ["LCDM", "w0waCDM"])
def test_syren_background_uses_resolved_dark_energy_and_hubble_units(model):
    provider = SyrenNewProvider(
        {"As": 2.1e-9, "w0_fld": -0.9, "wa_fld": 0.2}, {"NUMERICS": {}}, model
    )
    background = provider._background()
    assert background.H0.value == pytest.approx(67.0)
    if model == "LCDM":
        assert isinstance(background, FlatLambdaCDM)
        assert (provider.cosmopars["w0"], provider.cosmopars["wa"]) == (-1.0, 0.0)
    else:
        assert isinstance(background, Flatw0waCDM)
        assert (background.w0, background.wa) == (-0.9, 0.2)


def _emulator_class():
    return SimpleNamespace(
        pars={"m_ncdm": [0.01, 0.02, 0.03], "n_s": 0.96, "w0_fld": -0.9, "wa_fld": 0.1},
        h=lambda: 0.5,
        Omega_b=lambda: 0.05,
        Omega_m=lambda: 0.3,
        get_primordial=lambda: {
            "k [1/Mpc]": [0.01, 0.05, 0.1],
            "P_scalar(k)": [1e-9, 2e-9, 3e-9],
        },
    )


def test_ee2_reads_computed_amplitude_summed_mass_and_dark_energy():
    parameters = class_ee2_parameters(_emulator_class())
    assert parameters["A_s"] == pytest.approx(2e-9)
    assert parameters["m_ncdm"] == pytest.approx(0.06)
    assert parameters["w0_fld"] == -0.9
    assert parameters["wa_fld"] == 0.1
    classres = _emulator_class()
    classres.pars["k_pivot"] = 0.2
    with pytest.raises(ValueError, match="amplitude pivot"):
        class_ee2_parameters(classres)


def test_ee2_missing_extra_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "euclidemu2", None)
    with pytest.raises(ImportError, match="optional 'emulators' extra"):
        class_ee2_boost(_emulator_class(), [0.01], [0.0])


@pytest.mark.parametrize(
    "k,z,message",
    [
        ([[0.01]], [0.0], "one-dimensional"),
        ([np.nan], [0.0], "finite"),
        ([0.01], [np.inf], "finite"),
        ([0.01], [-0.01], "redshifts from 0 to 10"),
        ([0.01], [10.01], "redshifts from 0 to 10"),
    ],
)
def test_ee2_rejects_invalid_grids_before_emulation(monkeypatch, k, z, message):
    def unexpected_emulation():
        pytest.fail("Invalid input must be rejected before constructing the emulator")

    monkeypatch.setitem(
        sys.modules, "euclidemu2", SimpleNamespace(PyEuclidEmulator=unexpected_emulation)
    )
    with pytest.raises(ValueError, match=message):
        class_ee2_boost(_emulator_class(), k, z)


def test_ee2_chunks_redshifts_without_losing_order_or_unit_conversion(monkeypatch):
    calls = []

    class Emulator:
        def get_boost(self, parameters, redshifts):
            calls.append(redshifts)
            return [0.02, 0.2, 2.0], {i: np.full(3, 1.0 + z) for i, z in enumerate(redshifts)}

    monkeypatch.setitem(sys.modules, "euclidemu2", SimpleNamespace(PyEuclidEmulator=Emulator))
    z = np.linspace(10.0, 0.0, 205)
    boost = class_ee2_boost(_emulator_class(), [0.001, 0.1, 1.0], z)
    assert [len(chunk) for chunk in calls] == [100, 100, 5]
    np.testing.assert_allclose(np.concatenate(calls), z)
    np.testing.assert_allclose(boost[0], 1.0)
    np.testing.assert_allclose(boost[1:], np.tile(1.0 + z, (2, 1)))


@pytest.mark.parametrize(
    "obs1,obs2,column",
    [
        ("CMB_T", "CMB_T", 0),
        ("CMB_E", "CMB_E", 1),
        ("CMB_B", "CMB_B", 2),
        ("CMB_T", "CMB_E", 3),
        ("CMB_E", "CMB_T", 3),
    ],
)
@pytest.mark.parametrize("flags", [{"WantCls": True}, SimpleNamespace(Want_CMB=True)])
def test_camb_cmb_spectra_convert_dl_to_cl_and_select_correct_column(obs1, obs2, column, flags):
    cosmo = object.__new__(cosmo_functions)
    cosmo.code = "camb"
    cosmo.cambcosmopars = flags
    dl = np.arange(48.0).reshape(12, 4)
    cosmo.results = SimpleNamespace(camb_cmb=dl)
    ell = np.arange(2, 10)
    expected = dl[2:10, column] * 2 * np.pi / (ell * (ell + 1))
    np.testing.assert_allclose(cosmo.cmb_power(2, 10, obs1, obs2), expected)


def test_cmb_missing_spectra_and_unsupported_backends_fail_explicitly():
    cosmo = object.__new__(cosmo_functions)
    cosmo.code = "camb"
    cosmo.cambcosmopars = None
    cosmo.results = SimpleNamespace()
    with pytest.raises(ValueError, match="Enable WantCls"):
        cosmo.cmb_power(2, 10, "CMB_T", "CMB_T")
    cosmo.cambcosmopars = {"WantCls": True}
    with pytest.raises(AttributeError, match="spectrum missing"):
        cosmo.cmb_power(2, 10, "CMB_T", "CMB_T")
    for code in ("class", "symbolic"):
        cosmo.code = code
        with pytest.raises(NotImplementedError, match="Use code='camb'"):
            cosmo.cmb_power(2, 10, "CMB_T", "CMB_T")


@pytest.mark.parametrize("tracer,expected", [("matter", 0.8 * 0.7), ("clustering", 0.85 * 0.75)])
def test_fsigma8_uses_consistent_growth_and_variance_tracer(tracer, expected):
    cosmo = object.__new__(cosmo_functions)

    def growth_rate(value):
        def evaluate(z, k, grid):
            assert (z, k, grid) == (0.5, 0.1, False)
            return value

        return evaluate

    cosmo.results = SimpleNamespace(
        s8_of_z=lambda z: 0.7,
        s8_cb_of_z=lambda z: 0.75,
        f_growthrate_zk=growth_rate(0.8),
        f_growthrate_cb_zk=growth_rate(0.85),
    )
    assert cosmo.fsigma8_of_z(0.5, k=0.1, tracer=tracer) == pytest.approx(expected)


def test_growth_gamma_approximation_and_unknown_tracer_fallback():
    cosmo = object.__new__(cosmo_functions)
    cosmo.gamma = 0.55
    cosmo.input = "class"
    cosmo.results = SimpleNamespace(
        Om_m=lambda z: 0.6,
        s8_of_z=lambda z: 0.7,
        f_growthrate_zk=lambda z, k, grid: 0.8 if k == 0.0001 else 0.9,
    )
    assert cosmo.fsigma8_of_z(0.5, gamma=True) == pytest.approx(0.6**0.55 * 0.7)
    with pytest.warns(UserWarning, match="reverted to matter"):
        assert cosmo.sigma8_of_z(0.5, tracer="unknown") == 0.7
    with pytest.warns(UserWarning, match="reverted to matter"):
        assert cosmo.f_growthrate(0.5, tracer="unknown") == 0.8
