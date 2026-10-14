#!/usr/bin/env python
"""Evaluate fiducial GCsp and coupled WL+GCph likelihoods with SYREN-NEW.

Examples
--------
uv run python scripts/likelihood/diagnostics/run_syren_new_likelihoods.py --probe gcsp
uv run python scripts/likelihood/diagnostics/run_syren_new_likelihoods.py --probe photo
uv run python scripts/likelihood/diagnostics/run_syren_new_likelihoods.py --probe both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix
from cosmicfishpie.likelihood.photo_like import PhotometricLikelihood
from cosmicfishpie.likelihood.spectro_like import SpectroLikelihood

FIDUCIAL = {
    "10^9As": 2.1,
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "mnu": 0.06,
}


def build_runtime(observables: list[str], symbolic_yaml: Path) -> FisherMatrix:
    """Build the backend and probe state needed by one likelihood evaluation."""
    return FisherMatrix(
        fiducialpars=FIDUCIAL,
        freepars={},
        options={
            "code": "symbolic",
            "symbolic_config_yaml": str(symbolic_yaml),
            "cosmo_model": "LCDM",
            "nonlinear": True,
            "feedback": 0,
            "survey_name": "Euclid",
            "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
            "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
        },
        observables=observables,
        cosmoModel="LCDM",
        surveyName="Euclid",
    )


def evaluate_gcsp(symbolic_yaml: Path) -> float:
    runtime = build_runtime(["GCsp"], symbolic_yaml)
    likelihood = SpectroLikelihood(cosmoFM_data=runtime, cosmoFM_theory=runtime)
    return float(likelihood.loglike(param_dict={}))


def evaluate_photo(symbolic_yaml: Path) -> float:
    runtime = build_runtime(["WL", "GCph"], symbolic_yaml)
    likelihood = PhotometricLikelihood(
        cosmo_data=runtime,
        cosmo_theory=runtime,
        observables=["WL", "GCph"],
    )
    return float(likelihood.loglike(param_dict={}))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("gcsp", "photo", "both"), default="both")
    parser.add_argument(
        "--symbolic-yaml",
        type=Path,
        default=root / "cosmicfishpie/configs/default_boltzmann_yaml_files/symbolic/default.yaml",
        help="SYREN-NEW symbolic backend profile.",
    )
    args = parser.parse_args()
    symbolic_yaml = args.symbolic_yaml.resolve()

    if not symbolic_yaml.is_file():
        parser.error(f"SYREN-NEW YAML does not exist: {symbolic_yaml}")

    evaluations = {}
    if args.probe in ("gcsp", "both"):
        evaluations["GCsp"] = evaluate_gcsp(symbolic_yaml)
    if args.probe in ("photo", "both"):
        evaluations["WL+GCph"] = evaluate_photo(symbolic_yaml)

    for label, loglike in evaluations.items():
        if not np.isfinite(loglike):
            raise RuntimeError(f"{label} returned a non-finite log-likelihood: {loglike}")
        print(f"{label}: log L(fiducial) = {loglike:.12g}")


if __name__ == "__main__":
    main()
