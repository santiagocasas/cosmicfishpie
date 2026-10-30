"""Explain the SYREN-NEW versus JAXcosmo amplitude (``10^9As``) error difference.

    uv run --extra jaxcosmo python scripts/validation/compare_theory_amplitude_response.py

The two backends agree on everything except the matter power model: SYREN-NEW uses its own
emulator, JAXcosmo uses Eisenstein-Hu plus Halofit. This script separates the possible
causes of the ~10% difference in the marginalized ``10^9As`` error:

1. Theory response -- compare ``dlnP_nl/dlnAs`` between the two models.
2. Fisher decomposition -- marginalized versus unmarginalized amplitude error, and the
   error after fixing or marginalizing subsets of the nuisance parameters. This shows
   whether the covariance, the derivatives or the marginalization drives the difference.
3. Hybrid test -- put the SYREN-NEW power spectrum inside the *JAX* photometric
   observable. If the hybrid reproduces the SYREN error, the observable implementation,
   the Fisher assembly and the autodiff are all exonerated and the difference is a genuine
   theory difference.
4. Fiducial stability -- repeat the marginalized comparison at a second cosmology.

The JAXcosmo finite-difference control (same JAX theory and observable, finite-difference
derivatives) is included throughout; autodiff must reproduce it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Keep CPU thread counts reproducible; respect values supplied by the caller.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cosmicfishpie.cosmology.jax_theory import get_jax_theory  # noqa: E402
from cosmicfishpie.cosmology.jaxcosmo_halofit import halofit  # noqa: E402
from cosmicfishpie.cosmology.jaxcosmo_theory import make_cosmology  # noqa: E402
from cosmicfishpie.cosmology.symbolic_new import SyrenNewProvider  # noqa: E402
from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix  # noqa: E402
from cosmicfishpie.fishermatrix.derivatives import compute_derivatives  # noqa: E402
from cosmicfishpie.fishermatrix.jaxcosmo_derivatives import (  # noqa: E402
    JaxAutodiffProvider,
    JaxFiniteDifferenceProvider,
)
from cosmicfishpie.LSSsurvey import photo_cov, photo_obs  # noqa: E402
from cosmicfishpie.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable  # noqa: E402

jax.config.update("jax_enable_x64", True)

FREE = {name: 0.01 for name in ("10^9As", "Omegam", "Omegab", "h")}
FIDUCIALS = (
    ("fiducial-1", {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96}),
    ("fiducial-2", {"10^9As": 2.15, "Omegam": 0.31, "Omegab": 0.049, "h": 0.68, "ns": 0.96}),
)
BIAS = tuple(f"b{i}" for i in range(1, 11))
IA = ("AIA", "betaIA", "etaIA")
COSMO = tuple(FREE)


def load_yaml(relative: str) -> dict:
    with open(ROOT / "cosmicfishpie/configs/default_boltzmann_yaml_files" / relative) as handle:
        return yaml.safe_load(handle)


SYREN_PARS = load_yaml("symbolic/syren_new_photo.yaml")
JAX_PARS = load_yaml("jaxcosmo/default.yaml")


def fiducial_parameters(fiducial: dict) -> dict:
    pars = dict(fiducial)
    pars.setdefault("mnu", 0.0)
    return pars


def make_calc(code: str, fiducial: dict, provider=None, suffix: str = "") -> FisherMatrix:
    options = {
        "code": code,
        "accuracy": 1,
        "derivatives": "3PT",
        "feedback": 0,
        "nonlinear": True,
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "specs_dir": str(ROOT / "cosmicfishpie/configs/default_survey_specifications") + "/",
        "results_dir": str(ROOT / "results/theory_amplitude_response" / (code + suffix)) + "/",
    }
    if code == "symbolic":
        options["symbolic_config_yaml"] = str(
            ROOT
            / "cosmicfishpie/configs/default_boltzmann_yaml_files/symbolic/syren_new_photo.yaml"
        )
    return FisherMatrix(
        options=options,
        fiducialpars=fiducial_parameters(fiducial),
        freepars=dict(FREE),
        observables=["GCph", "WL"],
        cosmoModel="LCDM",
        surveyName="Euclid",
        derivative_provider=provider,
    )


def make_covariance(calc: FisherMatrix) -> photo_cov.PhotoCov:
    fiducial_cls = photo_obs.ComputeCls(
        calc.fiducialcosmopars,
        calc.photopars,
        calc.IApars,
        calc.photobiaspars,
        configuration=calc,
    )
    return photo_cov.PhotoCov(
        calc.fiducialcosmopars,
        calc.photopars,
        calc.IApars,
        calc.photobiaspars,
        fiducial_Cls=fiducial_cls,
        configuration=calc,
    )


def run_forecast(code: str, fiducial: dict, provider=None, suffix: str = ""):
    calc = make_calc(code, fiducial, provider, suffix)
    covariance = make_covariance(calc)
    cls = covariance.getcls(covariance.allparsfid)
    noisy_cls = covariance.getclsnoise(cls)
    covmat = covariance.get_covmat(noisy_cls)
    derivs = covariance.compute_derivs(derivative_provider=provider)
    matrix = calc.photo_LSS_fishermatrix_einsum(noisy_cls=noisy_cls, covmat=covmat, derivs=derivs)
    return {
        "calc": calc,
        "covariance": covariance,
        "cls": cls,
        "noisy_cls": noisy_cls,
        "covmat": covmat,
        "derivs": derivs,
        "matrix": matrix,
        "names": list(calc.freeparams),
    }


def sigma_of(matrix: np.ndarray, names: list[str], subset: tuple[str, ...] | None = None):
    if subset is None:
        indices = list(range(len(names)))
    else:
        indices = [names.index(name) for name in subset]
    submatrix = np.asarray(matrix)[np.ix_(indices, indices)]
    return float(np.sqrt(np.diag(np.linalg.inv(submatrix)))[names.index("10^9As")])


def correlation_of(matrix: np.ndarray, names: list[str], other: str) -> float:
    covariance = np.linalg.inv(np.asarray(matrix))
    i = names.index("10^9As")
    j = names.index(other)
    return float(covariance[i, j] / np.sqrt(covariance[i, i] * covariance[j, j]))


class SyrenPowerShim:
    """JAX background and growth with the SYREN-NEW total-matter power spectrum."""

    def __init__(self, base):
        self._base = base
        self.cosmo_names = base.cosmo_names
        self.free_name_map = base.free_name_map

    def resolve_cosmology(self, params):
        return self._base.resolve_cosmology(params)

    def tables(self, params, z, k_hmpc, nonlinear=True):
        return self._base.tables(params, z, k_hmpc, nonlinear)

    def power(self, params, z, k_hmpc, nonlinear=True):
        z = np.asarray(z, dtype=float)
        k = np.asarray(k_hmpc, dtype=float)
        cosmopars = {
            "10^9As": float(params["As"]),
            "Omegam": float(params["Omegam"]),
            "Omegab": float(params["Omegab"]),
            "h": float(params["h"]),
            "ns": float(params["ns"]),
            "mnu": 0.0,
            "w0": float(params["w0"]),
            "wa": float(params["wa"]),
        }
        results = SyrenNewProvider(cosmopars, SYREN_PARS, "LCDM").build()
        power = np.empty_like(k)
        for iz, redshift in enumerate(z):
            # The SYREN spline is defined on k in 1/Mpc and returns Mpc^3.
            power[iz] = results.Pk_nl(redshift, k[iz] * cosmopars["h"], grid=False)
        return power


class HalofitOnSyrenLinear:
    """JAX background, growth and Halofit, fed with the SYREN-NEW *linear* shape.

    Both backends are often described as "Eisenstein-Hu". They are not the same formula:
    ``jax_cosmo`` evaluates the Eisenstein-Hu 1998 fit including baryon wiggles, while
    SYREN-NEW multiplies the *no-wiggle* Eisenstein-Hu fit by a fitted correction to the
    true linear spectrum (Bartlett et al. 2023), a growth correction and a further
    ``S(k)`` correction. This shim keeps Halofit's nonlinear prescription and only swaps
    the linear input, which measures how much of the Fisher difference the linear shape
    carries.
    """

    def __init__(self, base):
        self._base = base
        self.cosmo_names = base.cosmo_names
        self.free_name_map = base.free_name_map

    def resolve_cosmology(self, params):
        return self._base.resolve_cosmology(params)

    def tables(self, params, z, k_hmpc, nonlinear=True):
        return self._base.tables(params, z, k_hmpc, nonlinear)

    def power(self, params, z, k_hmpc, nonlinear=True):
        if not nonlinear:
            return self._base.power(params, z, k_hmpc, nonlinear)
        z = np.asarray(z, dtype=float)
        k = np.asarray(k_hmpc, dtype=float)
        cosmopars = {
            "10^9As": float(params["As"]),
            "Omegam": float(params["Omegam"]),
            "Omegab": float(params["Omegab"]),
            "h": float(params["h"]),
            "ns": float(params["ns"]),
            "mnu": 0.0,
            "w0": float(params["w0"]),
            "wa": float(params["wa"]),
        }
        h = cosmopars["h"]
        results = SyrenNewProvider(cosmopars, SYREN_PARS, "LCDM").build()
        grid = np.logspace(-4.0, 3.0, 600)
        linear = np.array([results.Pk_l(0.0, value * h, grid=False) for value in grid]) * h**3
        log_grid = jnp.asarray(np.log10(grid))
        log_power = jnp.asarray(np.log(linear))

        def transfer(cosmo, wavenumber):
            wavenumber = jnp.asarray(wavenumber)
            shape = jnp.exp(jnp.interp(jnp.log10(wavenumber), log_grid, log_power))
            return jnp.sqrt(shape / wavenumber**cosmo.n_s)

        cosmology = make_cosmology({name: float(value) for name, value in params.items()})
        halofit_grid = jax.jit(
            jax.vmap(lambda scale, wavenumber: halofit(cosmology, wavenumber, scale, transfer))
        )
        return np.asarray(halofit_grid(jnp.asarray(1.0 / (1.0 + z)), jnp.asarray(k))) / h**3


def linear_shape_table(fiducial: dict) -> dict:
    """Ratio of the two linear spectra at equal As, plus the growth-factor ratio."""
    parameters = fiducial_parameters(fiducial)
    theory = get_jax_theory("jaxcosmo", JAX_PARS, "LCDM")
    cosmology = theory.resolve_cosmology(parameters)
    results = SyrenNewProvider(parameters, SYREN_PARS, "LCDM").build()
    h = parameters["h"]

    k_hmpc = np.logspace(-3.0, np.log10(3.0), 120)
    redshifts = np.array([0.0, 0.5, 1.0, 2.0])
    tables = theory.tables(cosmology, redshifts, k_hmpc, False)
    linear_jax = np.asarray(tables.pk_linear)
    linear_syren = np.array([results.Pk_l(z, k_hmpc * h, grid=False) for z in redshifts])
    growth_jax = np.asarray(tables.growth)
    growth_syren = np.array([results.D_growth_zk(z, 1e-4, grid=False) for z in redshifts])

    bands = (
        (1e-3, 5e-3),
        (5e-3, 2e-2),
        (2e-2, 5e-2),
        (5e-2, 0.1),
        (0.1, 0.2),
        (0.2, 0.5),
        (0.5, 1.0),
    )
    band_ratios = {}
    for low, high in bands:
        mask = (k_hmpc >= low) & (k_hmpc < high)
        band_ratios[f"{low:g}-{high:g}"] = float(
            np.mean(linear_syren[0, mask] / linear_jax[0, mask])
        )
    samples = {}
    for value in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        index = int(np.argmin(np.abs(k_hmpc - value)))
        samples[f"{k_hmpc[index]:.4f}"] = float(linear_syren[0, index] / linear_jax[0, index])
    return {
        "bands_z0": band_ratios,
        "samples_z0": samples,
        "growth_ratio": {
            float(z): float(syren / jax)
            for z, syren, jax in zip(redshifts, growth_syren, growth_jax)
        },
    }


def halofit_syren_forecast(jax_run: dict) -> dict:
    """Run the JAX observable with Halofit applied to the SYREN-NEW linear shape."""
    calc = jax_run["calc"]
    covariance = jax_run["covariance"]
    observable = JaxPhotoObservable(calc, dict(covariance.allparsfid))
    observable.theory = HalofitOnSyrenLinear(observable.theory)

    def hybrid_cls(allpars):
        return observable.as_dict(observable._cls_from_values(observable._values(allpars)))

    cls = hybrid_cls(covariance.allparsfid)
    noisy_cls = covariance.getclsnoise(cls)
    covmat = covariance.get_covmat(noisy_cls)
    derivs = compute_derivatives(hybrid_cls, covariance.allparsfid, configuration=calc)
    matrix = calc.photo_LSS_fishermatrix_einsum(noisy_cls=noisy_cls, covmat=covmat, derivs=derivs)
    return {"matrix": matrix, "cls": cls}


def theory_response(fiducial: dict) -> dict:
    """dlnP_nl/dlnAs on the same wavenumbers for both matter power models."""
    parameters = fiducial_parameters(fiducial)
    theory = get_jax_theory("jaxcosmo", JAX_PARS, "LCDM")
    jax_cosmology = theory.resolve_cosmology(parameters)

    k_hmpc = np.logspace(np.log10(1e-3), np.log10(1.0), 60)
    redshifts = np.array([0.0, 0.5, 1.0, 2.0])
    step = 0.01

    def jax_power(amplitude: float) -> np.ndarray:
        cosmology = dict(jax_cosmology, As=amplitude)
        return np.asarray(theory.tables(cosmology, redshifts, k_hmpc, True).pk_nonlinear)

    def syren_power(amplitude: float) -> np.ndarray:
        cosmology = dict(parameters, **{"10^9As": amplitude})
        results = SyrenNewProvider(cosmology, SYREN_PARS, "LCDM").build()
        k_1mpc = k_hmpc * cosmology["h"]
        return np.asarray([results.Pk_nl(z, k_1mpc, grid=False) for z in redshifts])

    amplitude = parameters["10^9As"]
    jax_up, jax_down = jax_power(amplitude * (1 + step)), jax_power(amplitude * (1 - step))
    syren_up = syren_power(amplitude * (1 + step))
    syren_down = syren_power(amplitude * (1 - step))

    response = {
        "jax": (np.log(jax_up) - np.log(jax_down)) / (np.log1p(step) - np.log1p(-step)),
        "syren": (np.log(syren_up) - np.log(syren_down)) / (np.log1p(step) - np.log1p(-step)),
    }

    # Exact autodiff response as a check on the finite-difference step.
    def jitted_log_power(vector):
        cosmology = dict(jax_cosmology, As=jnp.exp(vector[0]))
        return jnp.log(theory.tables(cosmology, redshifts, k_hmpc, True).pk_nonlinear)

    autodiff = np.asarray(jax.jit(jax.jacfwd(jitted_log_power))(jnp.array([np.log(amplitude)])))[
        ..., 0
    ]

    out = {}
    for iz, redshift in enumerate(redshifts):
        ratio = response["syren"][iz] / response["jax"][iz]
        out[float(redshift)] = {
            "jax_mean": float(np.mean(response["jax"][iz])),
            "syren_mean": float(np.mean(response["syren"][iz])),
            "ratio_mean": float(np.mean(ratio)),
            "ratio_min": float(np.min(ratio)),
            "ratio_max": float(np.max(ratio)),
        }
    autodiff_error = float(
        np.max(np.abs(autodiff - response["jax"])) / np.max(np.abs(response["jax"]))
    )
    return {
        "redshifts": out,
        "autodiff_vs_finite_difference": autodiff_error,
        "k_hmpc": k_hmpc.tolist(),
    }


def hybrid_forecast(jax_run: dict, syren_cls: dict) -> dict:
    calc = jax_run["calc"]
    covariance = jax_run["covariance"]
    observable = JaxPhotoObservable(calc, dict(covariance.allparsfid))
    observable.theory = SyrenPowerShim(observable.theory)

    def hybrid_cls(allpars):
        return observable.as_dict(observable._cls_from_values(observable._values(allpars)))

    cls = hybrid_cls(covariance.allparsfid)
    noisy_cls = covariance.getclsnoise(cls)
    covmat = covariance.get_covmat(noisy_cls)
    derivs = compute_derivatives(hybrid_cls, covariance.allparsfid, configuration=calc)
    matrix = calc.photo_LSS_fishermatrix_einsum(noisy_cls=noisy_cls, covmat=covmat, derivs=derivs)
    scale = max(np.max(np.abs(value)) for key, value in cls.items() if key != "ells")
    worst = max(
        np.max(np.abs(np.asarray(cls[key]) - np.asarray(syren_cls[key]))) / scale
        for key in cls
        if key != "ells"
    )
    return {"matrix": matrix, "cls_difference": float(worst)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=None, help="write summary.json and REPORT.md"
    )
    arguments = parser.parse_args()

    summary: dict = {"fiducials": {}, "theory": theory_response(FIDUCIALS[0][1])}

    print("Theory response dlnP_nl/dlnAs, SYREN-NEW versus Eisenstein-Hu + Halofit")
    print(f"{'z':>6} {'jax mean':>10} {'syren mean':>11} {'ratio mean':>11} {'ratio range':>16}")
    for redshift, values in summary["theory"]["redshifts"].items():
        print(
            f"{redshift:6.2f} {values['jax_mean']:10.4f} {values['syren_mean']:11.4f} "
            f"{values['ratio_mean']:11.4f} "
            f"{values['ratio_min']:7.4f} - {values['ratio_max']:7.4f}"
        )
    print(
        "autodiff versus finite-difference response (JAX): "
        f"{summary['theory']['autodiff_vs_finite_difference']:.2e}\n"
    )

    matrices: dict[str, dict] = {}
    for label, fiducial in FIDUCIALS:
        print(f"Running forecasts at {label}", flush=True)
        syren = run_forecast("symbolic", fiducial)
        jax_ad = run_forecast("jaxcosmo", fiducial, JaxAutodiffProvider())
        jax_fd = run_forecast("jaxcosmo", fiducial, JaxFiniteDifferenceProvider())
        entry = {"syren": syren, "jax_ad": jax_ad, "jax_fd": jax_fd}
        matrices[label] = entry

        names = syren["names"]
        assert names == jax_ad["names"] == jax_fd["names"]
        sigma_style = {
            "marginalized_all": None,
            "cosmology_only": COSMO,
            "cosmology_bias": COSMO + BIAS,
            "cosmology_bias_ia": COSMO + BIAS + IA,
        }
        fiducial_summary = {"sigma": {}, "unmarginalized": {}, "correlations": {}}
        for style, subset in sigma_style.items():
            fiducial_summary["sigma"][style] = {
                "syren": sigma_of(syren["matrix"], names, subset),
                "jax_ad": sigma_of(jax_ad["matrix"], names, subset),
                "jax_fd": sigma_of(jax_fd["matrix"], names, subset),
            }
        for run_name, run in (("syren", syren), ("jax_ad", jax_ad), ("jax_fd", jax_fd)):
            matrix = np.asarray(run["matrix"])
            index = names.index("10^9As")
            fiducial_summary["unmarginalized"][run_name] = float(
                1.0 / np.sqrt(matrix[index, index])
            )
            fiducial_summary["correlations"][run_name] = {
                other: correlation_of(run["matrix"], names, other)
                for other in ("Omegam", "Omegab", "h", "b1", "b8", "b10", "AIA")
            }
        summary["fiducials"][label] = fiducial_summary

        print(
            f"  marginalized 10^9As: SYREN {fiducial_summary['sigma']['marginalized_all']['syren']:.6g}  "
            f"JAX AD {fiducial_summary['sigma']['marginalized_all']['jax_ad']:.6g}  "
            f"JAX FD {fiducial_summary['sigma']['marginalized_all']['jax_fd']:.6g}",
            flush=True,
        )

    labels = [label for label, _ in FIDUCIALS]
    first = summary["fiducials"][labels[0]]
    print("\nAmplitude error versus nuisance treatment at " f"{labels[0]} (10^9As)")
    print(f"{'case':<24} {'SYREN':>12} {'JAX AD':>12} {'JAX FD':>12} {'AD/SYREN':>10}")
    for style in ("cosmology_only", "cosmology_bias", "cosmology_bias_ia", "marginalized_all"):
        values = first["sigma"][style]
        print(
            f"{style:<24} {values['syren']:12.6g} {values['jax_ad']:12.6g} "
            f"{values['jax_fd']:12.6g} {100 * (values['jax_ad'] / values['syren'] - 1):+9.2f}%"
        )
    print(
        "\nUnmarginalized 10^9As: "
        f"SYREN {first['unmarginalized']['syren']:.6g}  "
        f"JAX AD {first['unmarginalized']['jax_ad']:.6g}  "
        f"JAX FD {first['unmarginalized']['jax_fd']:.6g}"
    )
    print("\n10^9As correlations")
    print(f"{'with':<8} {'SYREN':>10} {'JAX AD':>10} {'JAX FD':>10}")
    for other in ("Omegam", "Omegab", "h", "b1", "b8", "b10", "AIA"):
        print(
            f"{other:<8} {first['correlations']['syren'][other]:10.4f} "
            f"{first['correlations']['jax_ad'][other]:10.4f} "
            f"{first['correlations']['jax_fd'][other]:10.4f}"
        )

    hybrid = hybrid_forecast(matrices[labels[0]]["jax_ad"], matrices[labels[0]]["syren"]["cls"])
    names = matrices[labels[0]]["syren"]["names"]
    hybrid_sigma = sigma_of(hybrid["matrix"], names)
    summary["hybrid"] = {
        "sigma": hybrid_sigma,
        "syren_sigma": first["sigma"]["marginalized_all"]["syren"],
        "jax_sigma": first["sigma"]["marginalized_all"]["jax_ad"],
        "cls_difference": hybrid["cls_difference"],
    }
    print(
        "\nHybrid test (SYREN-NEW power inside the JAX photometric observable)"
        f"\n  10^9As: SYREN {first['sigma']['marginalized_all']['syren']:.6g}  "
        f"JAX {first['sigma']['marginalized_all']['jax_ad']:.6g}  "
        f"hybrid {hybrid_sigma:.6g}"
        f"\n  hybrid versus SYREN fiducial C_ell: max|dC|/max|C| = {hybrid['cls_difference']:.3e}"
    )

    linear = linear_shape_table(FIDUCIALS[0][1])
    summary["linear_shape"] = linear
    print(
        "\nLinear power at equal As (SYREN-NEW / JAX Eisenstein-Hu), z=0"
        "\n  SYREN-NEW starts from the *no-wiggle* Eisenstein-Hu fit and multiplies it by a"
        "\n  fitted correction to the true linear spectrum, so the two are not the same formula."
    )
    print(f"{'band [h/Mpc]':>16} {'ratio':>8}")
    for band, value in linear["bands_z0"].items():
        print(f"{band:>16} {value:8.5f}")
    print(
        "  growth-factor ratio: "
        + ", ".join(f"z={z:g} {value:.5f}" for z, value in linear["growth_ratio"].items())
    )

    halofit_syren = halofit_syren_forecast(matrices[labels[0]]["jax_ad"])
    halofit_sigma = sigma_of(halofit_syren["matrix"], names)
    summary["halofit_on_syren_linear"] = {
        "sigma": halofit_sigma,
        "syren_sigma": first["sigma"]["marginalized_all"]["syren"],
        "jax_sigma": first["sigma"]["marginalized_all"]["jax_ad"],
    }
    print(
        "\nHalofit fed with the SYREN-NEW linear shape, same JAX observable"
        f"\n  10^9As: SYREN {first['sigma']['marginalized_all']['syren']:.6g}  "
        f"JAX {first['sigma']['marginalized_all']['jax_ad']:.6g}  "
        f"Halofit-on-SYREN-linear {halofit_sigma:.6g}"
    )

    print("\nFiducial stability of the marginalized 10^9As error:")
    print(f"{'fiducial':<14} {'SYREN':>11} {'JAX AD':>11} {'AD/SYREN':>10}")
    for label in labels:
        values = summary["fiducials"][label]["sigma"]["marginalized_all"]
        print(
            f"{label:<14} {values['syren']:11.6g} {values['jax_ad']:11.6g} "
            f"{100 * (values['jax_ad'] / values['syren'] - 1):+9.2f}%"
        )
    print(
        "\nThe JAX AD/FD columns differ only in the derivative method. The SYREN/JAX "
        "difference is the matter power model."
    )

    summary["ad_vs_fd_max_fisher"] = {
        label: float(
            np.max(
                np.abs(matrices[label]["jax_ad"]["matrix"] - matrices[label]["jax_fd"]["matrix"])
            )
            / np.max(np.abs(matrices[label]["jax_fd"]["matrix"]))
        )
        for label in labels
    }
    print(
        "max|F_AD - F_FD| / max|F_FD|: "
        + ", ".join(
            f"{label} {value:.2e}" for label, value in summary["ad_vs_fd_max_fisher"].items()
        )
    )

    if arguments.outdir is not None:
        outdir = arguments.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
