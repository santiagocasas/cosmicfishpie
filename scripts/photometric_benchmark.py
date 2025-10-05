#!/usr/bin/env python
# coding: utf-8

# # CosmicFishPie

import json
import os
import subprocess
import time
from pathlib import Path

from cosmicfishpie.analysis import fisher_plot_analysis as fpa
from cosmicfishpie.fishermatrix import cosmicfish

envkey = "OMP_NUM_THREADS"
print("The value of {:s} is: ".format(envkey), os.environ.get(envkey))
os.environ[envkey] = str(8)
print("The value of {:s} is: ".format(envkey), os.environ.get(envkey))


# # Cosmological parameters

fiducial = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.815584,
    "mnu": 0.06,
    "Neff": 3.044,
}


# Define the observables you are interested in
observables = [["GCph"], ["WL"], ["GCph", "WL"]]
codes = ["class", "symbolic", "camb"]
# Input options for CosmicFish (global options)
BASE_OUTROOT = "Euclid-Photo-ISTF-Pess-Benchmark_"

###############################################
# Path Resolution Helpers
###############################################


def _resolve_repo_root():
    """Attempt to find the repository root by looking for pyproject.toml upward."""
    start = Path(__file__).resolve()
    for parent in [start.parent] + list(start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: assume scripts/.. is root
    return start.parent.parent


def _resolve_configs_dir(repo_root: Path):
    """Return the configs directory (cosmicfishpie/configs)."""
    candidate = repo_root / "cosmicfishpie" / "configs"
    if candidate.is_dir():
        return candidate
    # fallback to package install
    try:
        import cosmicfishpie as _cfp_pkg  # type: ignore

        pkg_dir = Path(_cfp_pkg.__file__).resolve().parent
        candidate_pkg = pkg_dir / "configs"
        if candidate_pkg.is_dir():
            return candidate_pkg
    except Exception:
        pass
    return candidate  # even if missing


def _resolve_survey_specs_dir(configs_dir: Path):
    d = configs_dir / "default_survey_specifications"
    return d


def _resolve_boltzmann_dirs(repo_root: Path):
    boltz_root = repo_root / "default_boltzmann_yaml_files"
    return boltz_root


_repo_root = _resolve_repo_root()
_configs_dir = _resolve_configs_dir(_repo_root)
_survey_specs_dir = _resolve_survey_specs_dir(_configs_dir)
_boltz_root = _resolve_boltzmann_dirs(_configs_dir)

print(f"[benchmark] repo_root:        {_repo_root}", "is dir:", _repo_root.is_dir())
print(f"[benchmark] configs_dir:      {_configs_dir}", "is dir:", _configs_dir.is_dir())
print(f"[benchmark] survey_specs_dir: {_survey_specs_dir}", "is dir:", _survey_specs_dir.is_dir())
print(f"[benchmark] boltzmann_dir:    {_boltz_root}", "is dir:", _boltz_root.is_dir())


options = {
    "accuracy": 1,
    "outroot": BASE_OUTROOT,
    "results_dir": "scripts/benchmark_results/",
    "derivatives": "3PT",
    "feedback": 1,
    "survey_name": "Euclid",
    "specs_dir": str(_survey_specs_dir) + os.sep,
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "camb_config_yaml": str(_boltz_root / "camb" / "default.yaml"),
    "class_config_yaml": str(_boltz_root / "class" / "fast_photo.yaml"),
    "symbolic_config_yaml": str(_boltz_root / "symbolic" / "default.yaml"),
    "cosmo_model": "LCDM",
    "code": "dummy",
}

# Sanity check: confirm target photometric YAML exists (informational only)
_photo_yaml = _survey_specs_dir / "Euclid-Photometric-ISTF-Pessimistic.yaml"
if not _photo_yaml.is_file():
    print(f"[benchmark][WARNING] Expected photometric spec file missing: {_photo_yaml}")
else:
    print(f"[benchmark] Found photometric spec file: {_photo_yaml}")

freepars = {
    "Omegam": 0.01,
    "Omegab": 0.01,
    "h": 0.01,
    "ns": 0.01,
    "sigma8": 0.01,
}


cosmoFM = {}
fish_photo = {}
strings_obses = []
timings = {}

for code in codes:
    options["code"] = code
    for obse in observables:
        string_obse = "-".join(obse) + "_" + code  # Create a string with the name of the observable
        print("Computing Fisher matrix for ", string_obse)
        strings_obses.append(string_obse)
        options["outroot"] = "Euclid-Photo-ISTF-Pess-Benchmark_" + string_obse + "_"
        start_t = time.time()
        cosmoFM[string_obse] = cosmicfish.FisherMatrix(
            fiducialpars=fiducial,
            freepars=freepars,
            options=options,
            observables=obse,
            cosmoModel=options["cosmo_model"],
        )
        fish_photo[string_obse] = cosmoFM[string_obse].compute()
        end_t = time.time()
        elapsed = end_t - start_t
        timings[string_obse] = {"backend": code, "observables": obse, "seconds": elapsed}
        print(f"Finished {string_obse} in {elapsed:.3f} s")

# After loop restore a stable outroot for aggregate summary outputs
options["outroot"] = BASE_OUTROOT

# Write timing summary to disk (aggregate)
results_dir = options.get("results_dir", "results/")
os.makedirs(results_dir, exist_ok=True)
timing_outfile = os.path.join(results_dir, BASE_OUTROOT + "timings.json")
summary = {
    "outroot": BASE_OUTROOT,
    "codes": codes,
    "observables_combinations": strings_obses,
    "timings": timings,
}
with open(timing_outfile, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Timing information written to {timing_outfile}")

print("Computed Fisher matrices for: ", strings_obses)

print("Comparing results for different codes and observables")


fisher_list = [fish_photo[key] for key in strings_obses]

fishanalysis = fpa.CosmicFish_FisherAnalysis(fisher_list=fisher_list)


def _git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


# Export Fisher comparison to JSON using new exporter
fisher_comparison_json = os.path.join(results_dir, BASE_OUTROOT + "fisher_comparison.json")
extra_meta = {
    "git_commit": _git_commit(),
    "codes": codes,
    "observables": observables,
    "derivatives": options.get("derivatives"),
    "cosmo_model": options.get("cosmo_model"),
    "survey_name": options.get("survey_name"),
    "survey_name_photo": options.get("survey_name_photo"),
    "specs_dir": options.get("specs_dir"),
    "configs_dir": str(_configs_dir),
    "camb_config_yaml": options.get("camb_config_yaml"),
    "class_config_yaml": options.get("class_config_yaml"),
    "symbolic_config_yaml": options.get("symbolic_config_yaml"),
    "fiducial_parameters": list(fiducial.keys()),
    "free_parameters": list(freepars.keys()),
    "num_fishers": len(fisher_list),
    "benchmark_script": os.path.basename(__file__),
}

fishanalysis.compare_fisher_results(
    parstomarg=["Omegam", "sigma8"],
    export_json_path=fisher_comparison_json,
    append=False,
    overwrite=True,
    extra_metadata=extra_meta,
)
print(f"Fisher comparison JSON written to {fisher_comparison_json}")
