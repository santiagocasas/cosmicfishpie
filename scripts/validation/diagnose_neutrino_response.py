"""Isolate CLASS cb prescription and neutrino-density effects without full Fishers.

Run from the repository root; supply a saved Fisher specs JSON to preserve the
resolved settings. Outputs retain all sampled spectra and translated parameters.
"""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from cosmicfishpie.cosmology.cosmology import boltzmann_code


def subtraction_spline(runtime):
    """Historical CLASS cb recipe on the new, supported nonlinear grid."""
    solver = runtime.Classres
    transfer, k, z = solver.get_transfer_and_k_and_z()
    fb = solver.Omega_b() / solver.Omega_m()
    fc = solver.Omega0_cdm() / solver.Omega_m()
    fcb = fb + fc
    fnu = 1 - fcb
    tcb = (fb * transfer["d_b"] + fc * transfer["d_cdm"]) / fcb
    tnu = transfer["d_ncdm[0]"]
    primordial = solver.get_primordial()
    prim = UnivariateSpline(primordial["k [1/Mpc]"], primordial["P_scalar(k)"])(k)
    prim *= 2 * np.pi**2 / k**3
    correction = (2 * fnu * fcb * tnu * tcb + fnu**2 * tnu**2) * prim[:, None]
    correction = RectBivariateSpline(z[::-1], k, correction[:, ::-1].T)
    return lambda zz, kk: (runtime.results.Pk_nl(zz, kk) - correction(zz, kk)) / fcb**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    specs = json.loads(args.specs.read_text())
    repo = Path(__file__).resolve().parents[2]
    profiles = repo / "cosmicfishpie/configs/default_boltzmann_yaml_files"
    k = np.geomspace(1e-3, 10, 60)
    z = np.array([0.5, 1.0, 1.5, 2.0])
    masses = np.array([0.054, 0.060, 0.066])
    arrays = {"k": k, "z": z, "mnu": masses}
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "specs": str(args.specs.resolve()),
        "specs_sha256": hashlib.sha256(args.specs.read_bytes()).hexdigest(),
        "fiducialpars": specs["fiducialpars"],
        "options": specs["options"],
        "versions": {package: version(package) for package in ("camb", "classy", "numpy", "scipy")},
        "cosmology_source_sha256": hashlib.sha256(
            (repo / "cosmicfishpie/cosmology/cosmology.py").read_bytes()
        ).hexdigest(),
        "profile_sha256": {
            code: hashlib.sha256(
                (profiles / code / "nuvalidation_hp.yaml").read_bytes()
            ).hexdigest()
            for code in ("camb", "class")
        },
        "runs": [],
    }
    for code, factor in [("camb", 94.07), ("class", 93.14), ("class", 94.07)]:
        profile = yaml.safe_load((profiles / code / "nuvalidation_hp.yaml").read_text())
        if code == "class":
            profile.setdefault("PARAMETER_TRANSLATION", {})["neutrino_mass_fac"] = factor
            profile.setdefault("NONLINEAR", {})["cb_prescription"] = "class_native"
        settings = dict(specs["options"], feedback=0, camb_path="", memorize_cosmo=False)
        configuration = SimpleNamespace(
            settings=settings, input_type=code, backend_parameters=profile
        )
        tag = f"{code}_{factor}"
        spectra = {"mm": [], "cb": []}
        if code == "class":
            spectra["cb_subtraction"] = []
        for mass in masses:
            pars = dict(specs["fiducialpars"], mnu=float(mass))
            print(f"{tag}: mnu={mass}", flush=True)
            runtime = boltzmann_code(pars, code=code, configuration=configuration)
            spectra["mm"].append(runtime.results.Pk_nl(z, k))
            spectra["cb"].append(runtime.results.Pk_cb_nl(z, k))
            if code == "class":
                spectra["cb_subtraction"].append(subtraction_spline(runtime)(z, k))
                provenance["runs"].append(
                    {"tag": tag, "mnu": float(mass), "class_parameters": runtime.classcosmopars}
                )
                runtime.Classres.struct_cleanup()
                runtime.Classres.empty()
        for spectrum, values in spectra.items():
            values = np.array(values)
            if not np.isfinite(values).all() or not (values > 0).all():
                raise ValueError(f"Invalid spectrum: {tag}/{spectrum}")
            arrays[f"{tag}_{spectrum}"] = values
    reference = arrays["camb_94.07_cb"]
    derivative = (np.log(reference[2]) - np.log(reference[0])) / (masses[2] - masses[0])
    summary = {}
    for factor in (93.14, 94.07):
        for spectrum in ("cb", "cb_subtraction"):
            key = f"class_{factor}_{spectrum}"
            values = arrays[key]
            response = (np.log(values[2]) - np.log(values[0])) / (masses[2] - masses[0])
            summary[key] = {
                "median_fiducial_difference_percent": float(
                    np.median(abs(values[1] / reference[1] - 1)) * 100
                ),
                "median_response_difference_percent": float(
                    np.median(abs((response - derivative) / derivative)) * 100
                ),
                "max_response_difference_percent": float(
                    np.max(abs((response - derivative) / derivative)) * 100
                ),
                "median_absolute_response_difference_per_eV": float(
                    np.median(abs(response - derivative))
                ),
            }
    np.savez_compressed(args.outdir / "spectra.npz", **arrays)
    (args.outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
