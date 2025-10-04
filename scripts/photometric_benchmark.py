#!/usr/bin/env python
# coding: utf-8

# # CosmicFishPie

import json
import os
import subprocess
import time

import seaborn as sns

from cosmicfishpie.analysis import fisher_plot_analysis as fpa
from cosmicfishpie.fishermatrix import cosmicfish

snscolors = sns.color_palette("colorblind")
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
    "w0": -1.0,
    "wa": 0.0,
    "mnu": 0.06,
    "Neff": 3.044,
}


# Define the observables you are interested in
observables = [["GCph"], ["WL"], ["GCph", "WL"]]
codes = ["symbolic", "camb", "class"]
# Input options for CosmicFish (global options)
BASE_OUTROOT = "Euclid-Photo-ISTF-Pess-Benchmark_"
options = {
    "accuracy": 1,
    "outroot": BASE_OUTROOT,
    "results_dir": "results/",
    "derivatives": "3PT",
    "feedback": 1,
    "survey_name": "Euclid",
    "specs_dir": "../cosmicfishpie/configs/default_survey_specifications/",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "cosmo_model": "LCDM",
    "code": "dummy",
}

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
    "specs_dir": options.get("specs_dir"),
    "survey_name_photo": options.get("survey_name_photo"),
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
