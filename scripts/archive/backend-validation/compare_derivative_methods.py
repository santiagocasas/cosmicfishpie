#!/usr/bin/env python3
"""Compare Fisher matrices computed with different derivative methods.

Builds a WL-only Euclid Fisher matrix once per derivative method ("3PT",
"4PT_FWD", "POLY", "STEM"), then reports the marginalized 1-sigma error on
each free cosmological parameter for every method side by side, using "3PT"
as the reference baseline. Two backends are supported:

- ``symbolic`` (default): a fast, smooth, closed-form/emulator backend with
  essentially no numerical noise. Useful to check that the derivative
  methods agree in the noise-free limit.
- ``camb``: the real CAMB Boltzmann backend using the repository's fast
  default settings (``camb/default.yaml``), with ``halofit_version``
  overridden to ``takahashi``. Real Boltzmann solvers have an irreducible
  numerical noise floor (ODE-integration tolerances, line-of-sight
  quadrature, nonlinear halofit), which is the regime the adaptive "STEM"
  method is designed to be robust against.

This is meant as a manual validation/regression check for the derivative
engine in ``cosmicfishpie/fishermatrix/derivatives.py`` -- in particular to
confirm that the POLY method (fixed in this branch to correctly evaluate the
polynomial fit at the fiducial point) now agrees with the well-tested 3PT
central-difference method to good precision, in both the noise-free
(symbolic) and noisy (camb) regimes.

Usage
-----
    uv run python scripts/compare_derivative_methods.py
    uv run python scripts/compare_derivative_methods.py --methods 3PT,POLY
    uv run python scripts/compare_derivative_methods.py --backend camb
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

from cosmicfishpie.fishermatrix import cosmicfish as cff

ALL_METHODS = ["3PT", "4PT_FWD", "POLY", "STEM"]

# Both backends used here (symbolic, and CAMB in LCDM mode) support only the
# standard LCDM parameter set, so we restrict the free parameters to that
# set (no w0/wa/mnu here) to keep the comparison apples-to-apples.
FIDUCIAL = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.815,
}
FREEPARS = {name: 0.01 for name in FIDUCIAL}

# cosmicfishpie.configs.config.init() uses a caller-provided ``fiducialpars``
# dict wholesale (it only falls back to its own built-in defaults when
# ``fiducialpars is None``), so passing a minimal 5-key dict drops keys that
# CAMB's parameter-basis conversion needs (num_nu_massless in particular,
# see cosmology.changebasis_camb). The symbolic backend does not need these,
# so they are only added for the camb backend, matching config.py's own
# built-in fiducial defaults.
CAMB_EXTRA_FIDUCIAL = {
    "mnu": 0.06,
    "tau": 0.058,
    "num_nu_massive": 1,
    "num_nu_massless": 2.046,
}

RESULTS_DIR = "scripts/benchmark_results/compare_derivative_methods/"
DEFAULT_CAMB_YAML = "cosmicfishpie/configs/default_boltzmann_yaml_files/camb/default.yaml"


def _camb_yaml_with_halofit(halofit_version: str) -> str:
    """Write a temp copy of the fast default CAMB yaml with halofit_version overridden.

    The CAMB solver settings are only loadable from a YAML file path (see
    ``cosmicfishpie.configs.config._load_boltzmann_yaml``); there is no way to
    override an individual key via ``options``. So we copy the repository's
    fast default profile and only change ``halofit_version``, writing the
    result under the (gitignored) results directory.
    """
    with open(DEFAULT_CAMB_YAML, "r") as f:
        parsed = yaml.safe_load(f)
    parsed["COSMO_SETTINGS"]["halofit_version"] = halofit_version
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"camb_{halofit_version}.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(parsed, f)
    return out_path


def build_options(derivatives_method: str, backend: str) -> dict:
    options = {
        "accuracy": 1,
        "outroot": f"compare_deriv_{backend}_{derivatives_method}",
        "results_dir": RESULTS_DIR,
        "derivatives": derivatives_method,
        "ell_sampling": 25,
        "nonlinear": True,
        "feedback": 1,
        "specs_dir": "cosmicfishpie/configs/default_survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": backend,
    }
    if backend == "camb":
        options["camb_config_yaml"] = _camb_yaml_with_halofit("takahashi")
    return options


def compute_one(derivatives_method: str, backend: str):
    """Build a fresh WL-only FisherMatrix and compute it with the given method."""
    options = build_options(derivatives_method, backend)
    fiducialpars = dict(FIDUCIAL)
    if backend == "camb":
        fiducialpars.update(CAMB_EXTRA_FIDUCIAL)
    cosmoFM = cff.FisherMatrix(
        fiducialpars=fiducialpars,
        freepars=dict(FREEPARS),
        options=options,
        observables=["WL"],
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )
    t0 = time.time()
    fish = cosmoFM.compute()
    elapsed = time.time() - t0
    return fish, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help=f"Comma-separated derivative methods to compare (default: {','.join(ALL_METHODS)})",
    )
    parser.add_argument(
        "--backend",
        default="symbolic",
        choices=["symbolic", "camb"],
        help=(
            "Boltzmann backend to use (default: symbolic). 'camb' uses the repository's "
            "fast default settings with halofit_version=takahashi."
        ),
    )
    args = parser.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",")]
    backend = args.backend

    baseline_method = "3PT"
    if baseline_method not in methods:
        methods = [baseline_method] + methods

    results = {}
    timings = {}
    for method in methods:
        print(
            f"--- Computing WL-only Euclid Fisher matrix with backend={backend}, "
            f"derivatives={method} ---"
        )
        try:
            fish, elapsed = compute_one(method, backend)
        except SystemExit:
            print(f"WARNING: {method} derivative failed to converge (hit hard exit), skipping.")
            continue
        names = fish.get_param_names()
        bounds = fish.get_confidence_bounds(confidence_level=0.6827)
        sigmas = {name: float(bounds[i]) for i, name in enumerate(names)}
        results[method] = sigmas
        timings[method] = elapsed
        print(f"    done in {elapsed:.1f}s")

    if baseline_method not in results:
        print("ERROR: baseline method 3PT did not produce a result.", file=sys.stderr)
        return 1

    baseline = results[baseline_method]
    params = list(FIDUCIAL.keys())
    other_methods = [m for m in results if m != baseline_method]

    header = f"{'param':10s} {'sigma_3PT':>14s}"
    for m in other_methods:
        header += f" {'sigma_' + m:>14s} {'dev%_' + m:>10s}"
    print()
    print(header)
    print("-" * len(header))

    worst = {m: (None, 0.0) for m in other_methods}
    for p in params:
        row = f"{p:10s} {baseline[p]:14.6g}"
        for m in other_methods:
            sig = results[m][p]
            dev = abs(sig / baseline[p] - 1.0) * 100.0
            row += f" {sig:14.6g} {dev:10.3f}"
            if dev > worst[m][1]:
                worst[m] = (p, dev)
        print(row)

    print("-" * len(header))
    for m in other_methods:
        p, dev = worst[m]
        print(f"worst deviation vs 3PT for {m}: {dev:.3f}% (param={p})")

    print()
    print("timings (s):", {m: round(t, 1) for m, t in timings.items()})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
