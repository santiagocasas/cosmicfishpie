"""Compare the actual GCsp negative-log-likelihood Hessian with its Fisher matrix.

Uses the demo's symbolic LCDM, Euclid pessimistic setup; only Omegam and
sigma8 vary. All other cosmological and nuisance parameters remain fixed.
Run from the worktree root: uv run python scripts/gcsp_likelihood_hessian.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

PARAMS = ("Omegam", "sigma8")
FIDUCIAL = dict(Omegam=0.32, Omegab=0.05, h=0.67, ns=0.96, sigma8=0.815584)


def central_hessian(
    function: Callable[[np.ndarray], float], x: np.ndarray, steps: np.ndarray
) -> np.ndarray:
    """Return the O(h^2) central finite-difference Hessian of a scalar function.

    ``steps`` contains positive ABSOLUTE increments, one per parameter.
    The callable receives a fresh 1D array and must return a finite scalar.
    No cosmology dependencies: this function can be moved to derivatives.py.
    Requires 1 + 2*d**2 evaluations. No regularization or eigenvalue clipping.
    """
    x = np.asarray(x, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if x.ndim != 1 or x.size == 0 or steps.shape != x.shape:
        raise ValueError("x and steps must be nonempty 1D arrays of equal shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("x must be finite and steps must be finite and positive")

    def evaluate(point):
        value = float(function(point.copy()))
        if not np.isfinite(value):
            raise ValueError(f"Non-finite function at {point}: {value}")
        return value

    f0 = evaluate(x)
    result = np.empty((x.size, x.size))
    for i in range(x.size):
        ei = np.zeros_like(x)
        ei[i] = steps[i]
        result[i, i] = ((evaluate(x + ei) - f0) + (evaluate(x - ei) - f0)) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros_like(x)
            ej[j] = steps[j]
            value = (
                (evaluate(x + ei + ej) - evaluate(x + ei - ej))
                - (evaluate(x - ei + ej) - evaluate(x - ei - ej))
            ) / (4 * steps[i] * steps[j])
            result[i, j] = result[j, i] = value
    return result


def build_problem(outdir: Path):
    """Return a cached -log L callable and the name-aligned raw Fisher matrix."""
    from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix
    from cosmicfishpie.likelihood.spectro_like import SpectroLikelihood

    outdir.mkdir(parents=True, exist_ok=True)
    options = dict(
        accuracy=1,
        feedback=0,
        code="symbolic",
        nonlinear=True,
        cosmo_model="LCDM",
        survey_name="Euclid",
        derivatives="3PT",
        survey_name_photo="Euclid-Photometric-ISTF-Pessimistic",
        survey_name_spectro="Euclid-Spectroscopic-ISTF-Pessimistic",
        results_dir=str(outdir) + "/",
        outroot="GCsp_diagnostic_",
    )
    fm = FisherMatrix(
        fiducialpars=FIDUCIAL.copy(),
        freepars={p: 0.01 for p in PARAMS},
        options=options,
        observables=["GCsp"],
        cosmoModel="LCDM",
        surveyName="Euclid",
    )
    fisher = fm.compute()
    names = list(fisher.param_names)
    missing = [p for p in PARAMS if p not in names]
    if missing:
        raise ValueError(f"Fisher matrix is missing {missing}; got {names}")

    # CosmicFishPie automatically adds spectroscopic nuisances (lnbg_i, Ps_i).
    # The likelihood below varies only PARAMS and keeps every nuisance at its
    # fiducial value, so the matching Fisher prediction is the CONDITIONAL
    # submatrix of the Fisher matrix itself -- not the marginal submatrix of
    # its inverse. Both are returned because the difference between them is
    # exactly the galaxy-bias / sigma8 degeneracy under investigation.
    indices = [names.index(p) for p in PARAMS]
    full = np.asarray(fisher.fisher_matrix)
    conditional = full[np.ix_(indices, indices)]
    marginal = np.linalg.inv(np.linalg.inv(full)[np.ix_(indices, indices)])
    fishers = {
        "conditional": conditional,
        "marginal": marginal,
        "full": full,
        "param_names": names,
        "nuisance_names": [n for n in names if n not in PARAMS],
    }
    likelihood = SpectroLikelihood(cosmoFM_data=fm, cosmoFM_theory=fm, leg_flag="wedges")
    cache = {}

    def nll(point):
        key = tuple(float(v) for v in point)
        if key not in cache:
            value = -float(likelihood.loglike(param_dict=dict(zip(PARAMS, key))))
            if not np.isfinite(value):
                raise ValueError(f"Non-finite -log L at {key}: {value}")
            cache[key] = value
        return cache[key]

    return nll, fishers, np.array([FIDUCIAL[p] for p in PARAMS])


def covariance_summary(matrix):
    """Report widths/correlation only for positive-definite precision matrices."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    result = {"precision_eigenvalues": eigenvalues.tolist()}
    if np.all(eigenvalues > 0):
        covariance = np.linalg.inv(matrix)
        sigmas = np.sqrt(np.diag(covariance))
        result.update(sigmas=sigmas.tolist(), correlation=float(covariance[0, 1] / np.prod(sigmas)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-fractions",
        type=float,
        nargs="+",
        default=[0.2, 0.1, 0.05, 0.025],
        help="Absolute steps = fraction times Fisher marginal sigma",
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/gcsp_diagnostics/hessian"))
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in args.step_fractions):
        parser.error("Step fractions must be finite and positive")
    nll, fishers, center = build_problem(args.outdir)
    # The likelihood holds all nuisances fixed, so CONDITIONAL is the like-for-like
    # reference. MARGINAL is reported alongside it because that is what the triangle
    # plots show, and the gap between the two is the bias/sigma8 degeneracy.
    fisher = fishers["conditional"]
    marginal = fishers["marginal"]
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    if np.any(eigenvalues <= 0):
        raise ValueError("Fisher is not positive definite; cannot define diagnostic scales")
    sigma = np.sqrt(np.diag(np.linalg.inv(fisher)))
    inverse_sqrt = (eigenvectors * eigenvalues**-0.5) @ eigenvectors.T
    f0 = nll(center)
    conditional_summary = covariance_summary(fisher)
    marginal_summary = covariance_summary(marginal)
    report = dict(
        parameters=PARAMS,
        fiducial=center.tolist(),
        nll_fiducial=f0,
        nuisance_names=fishers["nuisance_names"],
        fisher_conditional=fisher.tolist(),
        fisher_marginal=marginal.tolist(),
        fisher_conditional_summary=conditional_summary,
        fisher_marginal_summary=marginal_summary,
        runs=[],
    )
    print("Nuisances held fixed:", fishers["nuisance_names"])
    print("Fisher (conditional, matches the likelihood):\n", fisher)
    print(
        "  sigmas:",
        conditional_summary.get("sigmas"),
        "correlation:",
        conditional_summary.get("correlation"),
    )
    print("Fisher (marginal, matches the triangle plots):\n", marginal)
    print(
        "  sigmas:",
        marginal_summary.get("sigmas"),
        "correlation:",
        marginal_summary.get("correlation"),
    )
    print("Fiducial -log L:", f0, flush=True)
    previous = None
    for fraction in args.step_fractions:
        steps = fraction * sigma
        hessian = central_hessian(nll, center, steps)
        whitened = inverse_sqrt @ hessian @ inverse_sqrt
        gradient = np.array(
            [
                (nll(center + np.eye(2)[i] * steps[i]) - nll(center - np.eye(2)[i] * steps[i]))
                / (2 * steps[i])
                for i in range(2)
            ]
        )
        run = dict(
            step_fraction=fraction,
            absolute_steps=steps.tolist(),
            hessian=hessian.tolist(),
            gradient=gradient.tolist(),
            hessian_summary=covariance_summary(hessian),
            fisher_whitened_eigenvalues=np.linalg.eigvalsh(whitened).tolist(),
            whitened_residual_norm=float(np.linalg.norm(whitened - np.eye(2))),
        )
        if previous is not None:
            run["whitened_step_change_norm"] = float(np.linalg.norm(whitened - previous))
        previous = whitened
        report["runs"].append(run)
        print(json.dumps(run, indent=2), flush=True)
        # Save after each scale, so completed results survive an interruption.
        (args.outdir / "hessian_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Agreement means stable step results and whitened eigenvalues near [1, 1].")
    print("A nonzero gradient/fiducial residual must be investigated before interpreting widths.")


if __name__ == "__main__":
    main()
