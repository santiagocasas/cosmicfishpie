"""Time SYREN FD and JAXcosmo AD/FD photometric forecasts at two fiducials.

    uv run --extra jaxcosmo python scripts/validation/time_syren_vs_jaxcosmo.py

Straight-line example: no helper functions. ns and mnu are fixed, nuisance parameters
are marginalized. The first AD call includes compilation AND execution, not pure
compilation time. The second call recomputes the same Jacobian using compiled code.
The second fiducial is a genuinely new forecast: it shows whether that compilation can be
reused when the fiducial changes.

The JAXcosmo finite-difference column is the control for the autodiff column: both
differentiate the same JAX theory and observable, so they should agree very closely. Any
larger difference against SYREN is a difference between the two matter power models.
"""

import os
import sys
from pathlib import Path
from time import perf_counter

# Keep CPU thread counts reproducible; respect values supplied by the caller.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

start = perf_counter()
import jax  # noqa: E402
import numpy as np  # noqa: E402

from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix  # noqa: E402
from cosmicfishpie.fishermatrix.jaxcosmo_derivatives import (  # noqa: E402
    JaxAutodiffProvider,
    JaxFiniteDifferenceProvider,
)
from cosmicfishpie.LSSsurvey import photo_cov, photo_obs  # noqa: E402

jax.config.update("jax_enable_x64", True)
devices = jax.devices()
print(f"Shared imports + JAX device startup: {perf_counter() - start:.3f} s", flush=True)
print(f"JAX devices: {devices}", flush=True)
print("Euclid GCph + WL, LCDM; ns=0.96 and mnu=0 fixed; default nuisance parameters.")
print("SYREN-NEW versus Eisenstein-Hu + Halofit; JAX AD versus JAX FD isolates the method.\n")

FIDUCIALS = [
    (
        "fiducial-1",
        {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96, "mnu": 0.0},
    ),
    (
        "fiducial-2",
        {"10^9As": 2.15, "Omegam": 0.31, "Omegab": 0.049, "h": 0.68, "ns": 0.96, "mnu": 0.0},
    ),
]
BACKENDS = (
    ("SYREN FD", "symbolic", None),
    ("JAXcosmo AD", "jaxcosmo", JaxAutodiffProvider()),
    ("JAXcosmo FD", "jaxcosmo", JaxFiniteDifferenceProvider()),
)
FREE = {name: 0.01 for name in ("10^9As", "Omegam", "Omegab", "h")}
results = {}

for fiducial_label, fiducial in FIDUCIALS:
    for label, code, provider in BACKENDS:
        run_label = f"{label} @ {fiducial_label}"
        print(f"--- {run_label} ---", flush=True)
        total_start = perf_counter()
        start = perf_counter()
        options = {
            "code": code,
            "accuracy": 1,
            "derivatives": "3PT",
            "feedback": 0,
            "nonlinear": True,
            "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
            "specs_dir": str(ROOT / "cosmicfishpie/configs/default_survey_specifications") + "/",
            "results_dir": str(ROOT / "results/timing_syren_vs_jaxcosmo" / code) + "/",
        }
        if code == "symbolic":
            options["symbolic_config_yaml"] = str(
                ROOT
                / "cosmicfishpie/configs/default_boltzmann_yaml_files/symbolic/syren_new_photo.yaml"
            )
        observable_before = provider._observable if code == "jaxcosmo" else None
        calc = FisherMatrix(
            options=options,
            fiducialpars=dict(fiducial),
            freepars=dict(FREE),
            observables=["GCph", "WL"],
            cosmoModel="LCDM",
            surveyName="Euclid",
            derivative_provider=provider,
        )
        fiducial_cls = photo_obs.ComputeCls(
            calc.fiducialcosmopars,
            calc.photopars,
            calc.IApars,
            calc.photobiaspars,
            configuration=calc,
        )
        covariance = photo_cov.PhotoCov(
            calc.fiducialcosmopars,
            calc.photopars,
            calc.IApars,
            calc.photobiaspars,
            fiducial_Cls=fiducial_cls,
            configuration=calc,
        )
        setup_seconds = perf_counter() - start
        print(f"  Setup/background (includes any theory JIT): {setup_seconds:.3f} s", flush=True)

        start = perf_counter()
        cls = covariance.getcls(covariance.allparsfid)
        cls_seconds = perf_counter() - start
        print(f"  Fiducial spectra:                           {cls_seconds:.3f} s", flush=True)

        start = perf_counter()
        noisy_cls = covariance.getclsnoise(cls)
        covmat = covariance.get_covmat(noisy_cls)
        covariance_seconds = perf_counter() - start
        print(
            f"  Noise + covariance:                         {covariance_seconds:.3f} s", flush=True
        )

        start = perf_counter()
        derivs = covariance.compute_derivs(derivative_provider=provider)
        # Providers return NumPy arrays: device work is synchronized before this timer stops.
        derivative_seconds = perf_counter() - start
        print(
            f"  Derivatives (first call):                   {derivative_seconds:.3f} s", flush=True
        )
        if isinstance(provider, JaxAutodiffProvider):
            rebuilt = (
                observable_before is not None and provider._observable is not observable_before
            )
            print(f"  Observable rebuilt for this fiducial:       {rebuilt}", flush=True)

        start = perf_counter()
        matrix = calc.photo_LSS_fishermatrix_einsum(
            noisy_cls=noisy_cls, covmat=covmat, derivs=derivs
        )
        assembly_seconds = perf_counter() - start
        print(f"  Fisher assembly:                            {assembly_seconds:.3f} s", flush=True)

        start = perf_counter()
        np.linalg.cholesky(matrix)
        sigma = np.sqrt(np.diag(np.linalg.inv(matrix)))
        inversion_seconds = perf_counter() - start
        cold_total = perf_counter() - total_start
        print(
            f"  Cholesky + inversion:                       {inversion_seconds:.3f} s", flush=True
        )
        print(f"  First forecast total (excludes imports):    {cold_total:.3f} s", flush=True)

        warm_seconds = None
        if isinstance(provider, JaxAutodiffProvider):
            start = perf_counter()
            warm_derivs = covariance.compute_derivs(derivative_provider=provider)
            warm_seconds = perf_counter() - start
            print(f"  AD derivatives, compiled/warm:              {warm_seconds:.3f} s", flush=True)
            print("  First AD call = tracing/compilation + execution; warm call = execution.")
            # Ensure this is a repeat evaluation of the same Jacobian.
            for name in calc.freeparams:
                for key in derivs[name]:
                    np.testing.assert_allclose(warm_derivs[name][key], derivs[name][key])

        results[run_label] = {
            "names": list(calc.freeparams),
            "matrix": matrix,
            "sigma": sigma,
            "derivative_seconds": derivative_seconds,
            "warm_seconds": warm_seconds,
            "cold_total": cold_total,
        }
        print()

print("Marginalized 1-sigma errors (sqrt(diag(F^-1)), including nuisance marginalization):")
for fiducial_label, _ in FIDUCIALS:
    syren = results[f"SYREN FD @ {fiducial_label}"]
    ad = results[f"JAXcosmo AD @ {fiducial_label}"]
    jaxfd = results[f"JAXcosmo FD @ {fiducial_label}"]
    assert syren["names"] == ad["names"] == jaxfd["names"], "Parameter ordering differs."
    print(f"[{fiducial_label}]")
    print(
        f"{'parameter':<16} {'SYREN FD':>12} {'JAX AD':>12} {'JAX FD':>12} "
        f"{'AD/SYREN':>10} {'AD/FD':>10}"
    )
    for name, s, a, f in zip(syren["names"], syren["sigma"], ad["sigma"], jaxfd["sigma"]):
        print(
            f"{name:<16} {s:12.6g} {a:12.6g} {f:12.6g} "
            f"{100 * (a / s - 1):+9.2f}% {100 * (a / f - 1):+9.2f}%"
        )
    print(
        f"max|F_AD - F_SYREN| / max|F_SYREN| = "
        f"{np.max(np.abs(ad['matrix'] - syren['matrix'])) / np.max(np.abs(syren['matrix'])):.3e}"
    )
    print(
        f"max|F_AD - F_JAXFD| / max|F_JAXFD| = "
        f"{np.max(np.abs(ad['matrix'] - jaxfd['matrix'])) / np.max(np.abs(jaxfd['matrix'])):.3e}"
    )
    print(
        f"SYREN FD derivatives / warm AD derivatives: "
        f"{syren['derivative_seconds'] / ad['warm_seconds']:.2f}x"
    )
    print()

print("Forecast wall time by fiducial (excludes imports):")
print(f"{'fiducial':<14} {'SYREN FD':>10} {'JAX AD cold':>13} {'JAX FD cold':>13} {'AD warm':>10}")
for fiducial_label, _ in FIDUCIALS:
    syren = results[f"SYREN FD @ {fiducial_label}"]
    ad = results[f"JAXcosmo AD @ {fiducial_label}"]
    jaxfd = results[f"JAXcosmo FD @ {fiducial_label}"]
    print(
        f"{fiducial_label:<14} {syren['cold_total']:10.3f} {ad['cold_total']:13.3f} "
        f"{jaxfd['cold_total']:13.3f} {ad['warm_seconds']:10.3f}"
    )
print("\nThe warm ratio excludes JIT startup. No accuracy ranking implied between theories.")
