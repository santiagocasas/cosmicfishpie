#!/usr/bin/env python
"""Quick WL + GCsp Fisher vs. Nautilus demo (symbolic backend, fast).

Computes:
  1. A Euclid WL Fisher forecast (symbolic backend).
  2. A Euclid GCsp Fisher forecast (symbolic backend).
  3. Their sum (WL + GCsp), combined via the `+` operator on
     ``cosmicfishpie.analysis.fisher_matrix.fisher_matrix`` (matches by
     parameter name, so no double counting of shared cosmological params).
   4. A triangle plot comparing the three Fisher forecasts.
   5. A triangle plot contrasting MARGINAL and CONDITIONAL Fisher contours.

Then repeats the same three cases (WL, GCsp, WL+GCsp) using fast/low-precision
Nautilus nested-sampling runs (few live points) instead of the Fisher
approximation, and produces:
   6. A triangle plot comparing the three Nautilus posteriors.
   7. A combined triangle plot overlaying Fisher ellipses and Nautilus
      posteriors together, for a direct Fisher-vs-Nautilus comparison.

Marginal vs. conditional
------------------------
CosmicFishPie adds nuisance parameters to the Fisher matrices automatically
(GCsp: ``lnbg_i``/``Ps_i`` per redshift bin; WL: intrinsic alignment). By
default this script samples only cosmology, holding the nuisances fixed, so
plot 7 uses the CONDITIONAL Fisher as the like-for-like reference. Pass
``--sample-nuisances`` to sample them too, in which case the MARGINAL Fisher
becomes the correct reference. Comparing a nuisance-fixed posterior against a
marginalised Fisher ellipse is an apples-to-oranges comparison.

Everything uses the ``symbolic`` boltzmann/power-spectrum backend for speed;
because of that, only flat LCDM (Omegam, Omegab, h, ns, sigma8) is available
(no w0/wa/mnu/Neff).

This is a demo/benchmark script, not part of the library. Outputs (plots) are
written to --outdir (default: scripts/wl_gcsp_demo_results/), which is not
committed to the repo.

Usage
-----
    uv run python scripts/wl_gcsp_fisher_nautilus_demo.py
    uv run python scripts/wl_gcsp_fisher_nautilus_demo.py --n-live 30 --skip-nautilus
    uv run python scripts/wl_gcsp_fisher_nautilus_demo.py --free-params Omegam,sigma8,h
    uv run python scripts/wl_gcsp_fisher_nautilus_demo.py --sample-nuisances --n-workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: we only ever save figures, never show()

import numpy as np
import pandas as pd
from nautilus import Sampler

import cosmicfishpie.analysis.fishconsumer as fico
import cosmicfishpie.analysis.fisher_operations as cfop
from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix
from cosmicfishpie.likelihood.photo_like import PhotometricLikelihood
from cosmicfishpie.likelihood.spectro_like import SpectroLikelihood

# ---------------------------------------------------------------------------
# Fiducial cosmology (symbolic backend only supports flat LCDM)
# ---------------------------------------------------------------------------

FIDUCIAL = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.815584,
}

# Prior ranges used both for the Nautilus runs. Only entries matching
# --free-params are actually used.
PRIOR_RANGES = {
    "Omegam": (0.28, 0.36),
    "Omegab": (0.04, 0.06),
    "h": (0.60, 0.74),
    "ns": (0.90, 1.02),
    "sigma8": (0.75, 0.87),
}

COMMON_OPTIONS = {
    "accuracy": 1,
    "feedback": 1,
    "code": "symbolic",
    "nonlinear": True,
    "cosmo_model": "LCDM",
    "survey_name": "Euclid",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
    "derivatives": "3PT",
    "results_dir": "results/",
}


def _options(outroot: str, **overrides) -> dict:
    opts = dict(COMMON_OPTIONS)
    opts["outroot"] = outroot
    opts.update(overrides)
    return opts


# ---------------------------------------------------------------------------
# Marginal vs. conditional Fisher helpers
# ---------------------------------------------------------------------------
#
# CosmicFishPie automatically adds nuisance parameters to the Fisher matrix
# (GCsp: lnbg_i and Ps_i per redshift bin; WL: intrinsic-alignment parameters).
# That makes two very different 2D contours available for the same forecast:
#
#   * MARGINAL   -- integrate the nuisances out. This is what `make_triangle_plot`
#                   shows by default, because it slices `fisher_matrix_inv`.
#   * CONDITIONAL-- hold the nuisances fixed at their fiducial values. This is
#                   the sub-block of the Fisher matrix itself.
#
# A Nautilus run that samples only cosmology (nuisances fixed) must be compared
# against the CONDITIONAL Fisher. Comparing it against the marginal one is an
# apples-to-oranges comparison: marginalising over galaxy bias inflates
# sigma(sigma8) by almost an order of magnitude for GCsp and rotates the
# degeneracy direction.


def conditional_fisher(fisher, params: list[str]):
    """Fisher sub-block with all other parameters held FIXED (not marginalised)."""
    return cfop.reshuffle(fisher, params, update_names=False)


def marginal_fisher(fisher, params: list[str]):
    """Fisher marginalised over every parameter not in ``params``."""
    return cfop.marginalise(fisher, params, update_names=False)


def nuisance_prior_ranges(fisher, free_params: list[str], n_sigma: float = 5.0) -> dict:
    """Build Nautilus prior ranges for every parameter in ``fisher``.

    Cosmological parameters use the hand-picked ``PRIOR_RANGES``. Nuisance
    parameters have no natural range here, so they get a box of
    ``fiducial +/- n_sigma * sigma_marginal`` taken from the Fisher forecast
    itself. That is wide enough not to truncate the posterior while keeping the
    prior volume small enough for a cheap nested-sampling run.
    """
    ranges = {p: PRIOR_RANGES[p] for p in free_params}
    names = list(fisher.param_names)
    fiducials = np.asarray(fisher.param_fiducial, dtype=float)
    sigmas = np.sqrt(np.diag(np.asarray(fisher.fisher_matrix_inv, dtype=float)))
    for index, name in enumerate(names):
        if name in ranges:
            continue
        half_width = n_sigma * sigmas[index]
        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError(f"Non-finite Fisher error for nuisance parameter '{name}'")
        ranges[name] = (fiducials[index] - half_width, fiducials[index] + half_width)
    return ranges


def compute_fisher_wl(free_params: list[str]):
    """Build and compute a WL-only Fisher matrix with the symbolic backend.

    Both survey_name_photo/spectro stay set to real spec files (matching the
    tested notebook pattern); only `observables=["WL"]` decides which branch
    of FisherMatrix.compute() actually runs (see cosmicfish.py's if/elif).
    """
    options = _options("WL_symb_demo_")
    freepars = {p: 0.01 for p in free_params}
    fm_wl = FisherMatrix(
        fiducialpars=FIDUCIAL,
        freepars=freepars,
        options=options,
        observables=["WL"],
        cosmoModel="LCDM",
        surveyName="Euclid",
    )
    fisher_wl = fm_wl.compute()
    return fm_wl, fisher_wl


def compute_fisher_gcsp(free_params: list[str]):
    """Build and compute a GCsp-only Fisher matrix with the symbolic backend."""
    options = _options("GCsp_symb_demo_")
    freepars = {p: 0.01 for p in free_params}
    fm_gcsp = FisherMatrix(
        fiducialpars=FIDUCIAL,
        freepars=freepars,
        options=options,
        observables=["GCsp"],
        cosmoModel="LCDM",
        surveyName="Euclid",
    )
    fisher_gcsp = fm_gcsp.compute()
    return fm_gcsp, fisher_gcsp


def _posterior_dataframe(sampler: Sampler, prior) -> pd.DataFrame:
    """Convert a run Nautilus Sampler's posterior into a chainconsumer-ready DataFrame."""
    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    data = np.column_stack([points, weights, log_l])
    columns = list(prior.keys) + ["weight", "posterior"]
    return pd.DataFrame(data, columns=columns)


def _stage_banner(index: str, description: str, ranges: dict, n_live: int) -> None:
    """Announce which Nautilus stage is about to start.

    Nautilus prints its own progress table but never says *what* it is
    sampling, so consecutive runs are indistinguishable in the log.
    """
    names = list(ranges)
    print()
    print("-" * 70)
    print(f"Nautilus stage {index}: {description}")
    print(f"  Dimensions: {len(names)}   n_live: {n_live}")
    print(f"  Parameters: {names}")
    print("-" * 70)


def _checkpoint_name(label: str, n_live: int, prior_ranges: dict) -> str:
    """Build a checkpoint filename that encodes the run configuration.

    Nautilus resumes from ``filepath`` whenever the file exists, but it does not
    restore or validate ``n_live``: the stored value is written and never read
    back. Resuming with a different ``n_live`` therefore silently reuses the
    bounds built by the *previous* run and skips the exploration phase, which is
    exactly the stage that determines how well a thin degeneracy is tracked.

    Encoding ``n_live`` and the sampled dimensionality in the name makes that
    failure mode impossible: a changed configuration simply starts a new file,
    while re-running an identical configuration still resumes as intended.
    """
    return f"nautilus_{label}_nlive{n_live}_dim{len(prior_ranges)}.hdf5"


def _save_chain(chain: pd.DataFrame, outdir: Path, label: str) -> Path:
    """Persist a finished chain immediately, so a later crash cannot lose it.

    Nested sampling runs here take tens of minutes to hours. Keeping the chain
    only in memory until the plotting stage means an error in a *later* run
    discards everything computed so far.
    """
    path = outdir / f"chain_{label}.csv"
    chain.to_csv(path, index=False)
    print(f"  Chain saved to: {path}")
    return path


def _effective_sample_size(chain: pd.DataFrame) -> float:
    """Kish effective sample size of a weighted chain."""
    w = np.asarray(chain["weight"], dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def _chain_statistics(chain: pd.DataFrame, params: list[str]) -> dict:
    """Weighted mean, covariance, sigmas and correlations of a Nautilus chain.

    Nautilus returns *weighted* posterior samples, so every moment must use the
    weights. The Kish effective sample size sets the Monte-Carlo precision:
    the error on each mean is ``sigma / sqrt(ESS)`` and the fractional error on
    each sigma is roughly ``1 / sqrt(2 * ESS)``.
    """
    missing = [p for p in params if p not in chain.columns]
    if missing:
        raise ValueError(f"Chain is missing parameters {missing}")

    weights = np.asarray(chain["weight"], dtype=float)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("Chain weights must be finite and sum to a positive value")
    weights = weights / weights.sum()

    samples = np.asarray(chain[params], dtype=float)
    mean = weights @ samples
    cov = np.atleast_2d(np.cov(samples.T, aweights=weights, ddof=0))
    sigma = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(sigma, sigma)
    ess = float(1.0 / np.sum(weights**2))

    return {
        "parameters": list(params),
        "effective_sample_size": ess,
        "mean": mean,
        "covariance": cov,
        "sigma": sigma,
        "correlation": correlation,
        "mean_error": sigma / np.sqrt(ess),
        "sigma_fractional_error": float(1.0 / np.sqrt(2.0 * ess)),
    }


def _fisher_covariance(fisher, params: list[str]) -> np.ndarray:
    """Covariance sub-block of a Fisher object, selected by parameter name."""
    names = list(fisher.param_names)
    missing = [p for p in params if p not in names]
    if missing:
        raise ValueError(f"Fisher matrix is missing parameters {missing}")
    idx = [names.index(p) for p in params]
    return np.asarray(fisher.fisher_matrix_inv)[np.ix_(idx, idx)]


def _compare_chain_to_fisher(chain, fisher, params: list[str], fiducial: dict) -> dict:
    """Tabulate chain moments against the reference Fisher forecast.

    ``sigma_ratio`` is chain/Fisher: values above one mean the sampled posterior
    is *wider* than the Gaussian forecast. ``bias`` is the offset of the
    posterior mean from the fiducial in units of the chain sigma; it should be
    consistent with zero to within the Monte-Carlo error ``1 / sqrt(ESS)``.
    """
    stats = _chain_statistics(chain, params)
    cov_fisher = _fisher_covariance(fisher, params)
    sigma_fisher = np.sqrt(np.diag(cov_fisher))
    corr_fisher = cov_fisher / np.outer(sigma_fisher, sigma_fisher)

    truth = np.array([fiducial[p] for p in params], dtype=float)
    bias = (stats["mean"] - truth) / stats["sigma"]

    return {
        **stats,
        "fiducial": truth,
        "sigma_fisher": sigma_fisher,
        "correlation_fisher": corr_fisher,
        "sigma_ratio": stats["sigma"] / sigma_fisher,
        "bias": bias,
    }


def _report_statistics(
    comparisons: dict[str, dict],
    params: list[str],
    reference_kind: str,
    outdir: Path,
) -> Path:
    """Print the chain-vs-Fisher comparison and persist it as JSON."""
    print()
    print("=" * 70)
    print(f"Chain statistics vs. {reference_kind} Fisher forecast")
    print("=" * 70)

    for label, comp in comparisons.items():
        ess = comp["effective_sample_size"]
        precision = 100 * comp["sigma_fractional_error"]
        print(f"\n{label}  (ESS {ess:.0f}, sigma precision +-{precision:.0f}%)")
        print(f"  {'param':<10} {'mean':>12} {'sigma':>12} {'Fisher':>12} {'ratio':>8} {'bias':>8}")
        for i, name in enumerate(params):
            print(
                f"  {name:<10} {comp['mean'][i]:>12.6f} {comp['sigma'][i]:>12.6f} "
                f"{comp['sigma_fisher'][i]:>12.6f} {comp['sigma_ratio'][i]:>8.3f} "
                f"{comp['bias'][i]:>+8.2f}"
            )
        if len(params) == 2:
            print(
                f"  correlation: {comp['correlation'][0, 1]:+.4f} (chain) vs "
                f"{comp['correlation_fisher'][0, 1]:+.4f} (Fisher)"
            )

    payload = {
        "parameters": params,
        "reference_kind": reference_kind,
        "probes": {
            label: {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in comp.items()
            }
            for label, comp in comparisons.items()
        },
    }
    path = outdir / "chain_statistics.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nStatistics saved to: {path}")
    return path


def _n_batch(n_live: int, n_workers: int) -> int:
    """Pick n_batch as the smallest multiple of n_workers that is >= max(n_live, 4).

    Nautilus recommends n_batch be a multiple of the pool size so every worker
    gets an equal share of likelihood evaluations each step (see `pool`/`n_batch`
    in nautilus.Sampler's docstring).
    """
    minimum = max(n_live, 4, n_workers)
    return -(-minimum // n_workers) * n_workers  # ceil(minimum / n_workers) * n_workers


def run_nautilus_wl(
    fm_wl: FisherMatrix,
    prior_ranges: dict,
    n_live: int,
    n_workers: int = 1,
    outdir: Path | None = None,
    verbose: bool = True,
):
    """Fast Nautilus run for WL alone, reusing the already-computed fm_wl."""
    photo_like = PhotometricLikelihood(cosmo_data=fm_wl, cosmo_theory=fm_wl)
    prior = photo_like.create_nautilus_prior(prior_ranges)
    n_batch = _n_batch(n_live, n_workers)
    sampler_kwargs = {
        "n_live": n_live,
        "n_networks": 1,
        "n_batch": n_batch,
        "pool": n_workers,
    }
    if outdir is not None:
        sampler_kwargs["filepath"] = str(outdir / _checkpoint_name("wl", n_live, prior_ranges))
        sampler_kwargs["resume"] = True
    sampler = photo_like.run_nautilus(
        prior=prior,
        sampler_kwargs=sampler_kwargs,
        run_kwargs={
            "n_eff": max(2 * n_live, 20),
            "verbose": verbose,
            "discard_exploration": True,
        },
    )
    return _posterior_dataframe(sampler, prior), photo_like


def run_nautilus_gcsp(
    fm_gcsp: FisherMatrix,
    prior_ranges: dict,
    n_live: int,
    n_workers: int = 1,
    outdir: Path | None = None,
    verbose: bool = True,
):
    """Fast Nautilus run for GCsp alone, reusing the already-computed fm_gcsp."""
    spectro_like = SpectroLikelihood(cosmoFM_data=fm_gcsp, cosmoFM_theory=fm_gcsp)
    prior = spectro_like.create_nautilus_prior(prior_ranges)
    n_batch = _n_batch(n_live, n_workers)
    sampler_kwargs = {
        "n_live": n_live,
        "n_networks": 1,
        "n_batch": n_batch,
        "pool": n_workers,
    }
    if outdir is not None:
        sampler_kwargs["filepath"] = str(outdir / _checkpoint_name("gcsp", n_live, prior_ranges))
        sampler_kwargs["resume"] = True
    sampler = spectro_like.run_nautilus(
        prior=prior,
        sampler_kwargs=sampler_kwargs,
        run_kwargs={
            "n_eff": max(2 * n_live, 20),
            "verbose": verbose,
            "discard_exploration": True,
        },
    )
    return _posterior_dataframe(sampler, prior), spectro_like


def run_nautilus_joint(
    photo_like: PhotometricLikelihood,
    spectro_like: SpectroLikelihood,
    prior_ranges: dict,
    n_live: int,
    n_workers: int = 1,
    outdir: Path | None = None,
    verbose: bool = True,
):
    """Joint WL+GCsp Nautilus run: sums both loglikes over the shared parameters.

    This is the correct "no double counting" analogue of the Fisher `+`
    combination: a single nested-sampling run over one prior, with a
    log-likelihood equal to the sum of the WL and GCsp log-likelihoods
    evaluated at the same point.
    """
    prior = photo_like.create_nautilus_prior(prior_ranges)

    def joint_loglike(param_dict):
        # nautilus defaults pass_dict=True whenever `prior` is a nautilus.Prior
        # instance, so the sampler already hands us a dict keyed by param name.
        return photo_like.loglike(param_dict=param_dict) + spectro_like.loglike(
            param_dict=param_dict
        )

    n_batch = _n_batch(n_live, n_workers)
    extra = {}
    if outdir is not None:
        extra["filepath"] = str(outdir / _checkpoint_name("joint", n_live, prior_ranges))
        extra["resume"] = True
    sampler = Sampler(
        prior,
        joint_loglike,
        n_live=n_live,
        n_networks=1,
        n_batch=n_batch,
        pool=n_workers,
        pass_dict=True,
        **extra,
    )
    sampler.run(n_eff=max(2 * n_live, 20), verbose=verbose, discard_exploration=True)
    return _posterior_dataframe(sampler, prior)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--free-params",
        type=str,
        default="Omegam,sigma8",
        help="Comma-separated free parameters (default: Omegam,sigma8). "
        "Must be a subset of %s" % list(PRIOR_RANGES),
    )
    parser.add_argument(
        "--n-live", type=int, default=50, help="Nautilus live points (few = fast, less precise)"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for Nautilus likelihood evaluations "
        "(passed as nautilus.Sampler's `pool`). Use e.g. `nproc` cores minus a couple "
        "(e.g. --n-workers 8 on a 14-core machine) to leave headroom for the OS/other work.",
    )
    parser.add_argument(
        "--sample-nuisances",
        action="store_true",
        help="Sample the nuisance parameters (GCsp: lnbg_i/Ps_i, WL: intrinsic alignment) "
        "together with cosmology, instead of holding them fixed at their fiducial values. "
        "This is the scientifically correct comparison against the MARGINAL Fisher contours, "
        "but it raises the dimensionality (2 -> ~10+) and is much slower.",
    )
    parser.add_argument(
        "--nuisance-sigma",
        type=float,
        default=5.0,
        help="Half-width of the nuisance priors in units of the marginal Fisher error "
        "(only used with --sample-nuisances).",
    )
    parser.add_argument(
        "--skip-nautilus", action="store_true", help="Only run the Fisher forecasts and plot"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="scripts/wl_gcsp_demo_results",
        help="Directory to save plots to (not committed to the repo)",
    )
    parser.add_argument(
        "--quiet-sampler",
        action="store_true",
        help="Silence Nautilus progress output (it is shown by default, since the "
        "high-dimensional runs take hours)",
    )
    args = parser.parse_args()

    free_params = [p.strip() for p in args.free_params.split(",") if p.strip()]
    unknown = [p for p in free_params if p not in PRIOR_RANGES]
    if unknown:
        parser.error(f"Unknown free parameter(s) {unknown}; choose from {list(PRIOR_RANGES)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Free parameters: {free_params}")
    print("Step 1/2: Fisher forecasts (symbolic backend, Euclid WL + GCsp)")
    print("=" * 70)

    t0 = time.time()
    fm_wl, fisher_wl = compute_fisher_wl(free_params)
    print(f"  WL Fisher done in {time.time() - t0:.1f}s")

    t0 = time.time()
    fm_gcsp, fisher_gcsp = compute_fisher_gcsp(free_params)
    print(f"  GCsp Fisher done in {time.time() - t0:.1f}s")

    fisher_combined = fisher_wl + fisher_gcsp

    fisher_marg = {
        "WL": marginal_fisher(fisher_wl, free_params),
        "GCsp": marginal_fisher(fisher_gcsp, free_params),
        "WL+GCsp": marginal_fisher(fisher_combined, free_params),
    }
    fisher_cond = {
        "WL": conditional_fisher(fisher_wl, free_params),
        "GCsp": conditional_fisher(fisher_gcsp, free_params),
        "WL+GCsp": conditional_fisher(fisher_combined, free_params),
    }

    nuisances = [p for p in fisher_combined.param_names if p not in free_params]
    print(f"  Nuisance parameters in the Fisher matrices: {nuisances}")
    for label in ("WL", "GCsp", "WL+GCsp"):
        sig_m = np.sqrt(np.diag(fisher_marg[label].fisher_matrix_inv))
        sig_c = np.sqrt(np.diag(fisher_cond[label].fisher_matrix_inv))
        widths = ", ".join(
            f"{p}: {m:.4g} (marg) vs {c:.4g} (cond)" for p, m, c in zip(free_params, sig_m, sig_c)
        )
        print(f"    {label:>8}: {widths}")

    fico.make_triangle_plot(
        fishers=[fisher_wl, fisher_gcsp, fisher_combined],
        fisher_labels=["WL (Fisher)", "GCsp (Fisher)", "WL+GCsp (Fisher)"],
        colors=["blue", "green", "red"],
        params=free_params,
        truth_values=FIDUCIAL,
        save_plot=True,
        savepath=str(outdir) + "/",
        plot_filename="fisher_wl_gcsp_combined",
        file_format=".png",
    )
    print(f"Saved: {outdir}/fisher_wl_gcsp_combined.png")

    # Make the marginal-vs-conditional difference explicit: the same Fisher
    # matrices, once with nuisances marginalised and once with them fixed.
    fico.make_triangle_plot(
        fishers=[
            fisher_marg["WL"],
            fisher_marg["GCsp"],
            fisher_marg["WL+GCsp"],
            fisher_cond["WL"],
            fisher_cond["GCsp"],
            fisher_cond["WL+GCsp"],
        ],
        fisher_labels=[
            "WL (marginal)",
            "GCsp (marginal)",
            "WL+GCsp (marginal)",
            "WL (conditional)",
            "GCsp (conditional)",
            "WL+GCsp (conditional)",
        ],
        colors=["blue", "green", "red", "cyan", "lime", "orange"],
        params=free_params,
        truth_values=FIDUCIAL,
        save_plot=True,
        savepath=str(outdir) + "/",
        plot_filename="fisher_marginal_vs_conditional",
        file_format=".png",
    )
    print(f"Saved: {outdir}/fisher_marginal_vs_conditional.png")

    if args.skip_nautilus:
        print("--skip-nautilus set, stopping here.")
        return

    # Prior ranges decide what the sampler actually explores, and therefore
    # which Fisher contour is the like-for-like reference.
    if args.sample_nuisances:
        ranges_wl = nuisance_prior_ranges(fisher_wl, free_params, args.nuisance_sigma)
        ranges_gcsp = nuisance_prior_ranges(fisher_gcsp, free_params, args.nuisance_sigma)
        ranges_joint = nuisance_prior_ranges(fisher_combined, free_params, args.nuisance_sigma)
        reference = fisher_marg
        reference_kind = "marginal"
    else:
        ranges_wl = ranges_gcsp = ranges_joint = {p: PRIOR_RANGES[p] for p in free_params}
        reference = fisher_cond
        reference_kind = "conditional"

    print("=" * 70)
    print(f"Step 2/2: Nautilus nested sampling (n_live={args.n_live}, fast/low-precision)")
    print(f"  Like-for-like Fisher reference: {reference_kind}")
    print("  Stages to run: 1/3 WL, 2/3 GCsp, 3/3 WL+GCsp joint")
    print("=" * 70)

    verbose = not args.quiet_sampler

    _stage_banner("1/3", "WL (photometric weak lensing)", ranges_wl, args.n_live)
    t0 = time.time()
    chain_wl, photo_like = run_nautilus_wl(
        fm_wl, ranges_wl, args.n_live, args.n_workers, outdir=outdir, verbose=verbose
    )
    print(
        f"  WL Nautilus done in {time.time() - t0:.1f}s ({len(chain_wl)} samples, "
        f"ESS {_effective_sample_size(chain_wl):.0f})"
    )
    _save_chain(chain_wl, outdir, "wl")

    _stage_banner("2/3", "GCsp (spectroscopic galaxy clustering)", ranges_gcsp, args.n_live)
    t0 = time.time()
    chain_gcsp, spectro_like = run_nautilus_gcsp(
        fm_gcsp, ranges_gcsp, args.n_live, args.n_workers, outdir=outdir, verbose=verbose
    )
    print(
        f"  GCsp Nautilus done in {time.time() - t0:.1f}s ({len(chain_gcsp)} samples, "
        f"ESS {_effective_sample_size(chain_gcsp):.0f})"
    )
    _save_chain(chain_gcsp, outdir, "gcsp")

    _stage_banner("3/3", "WL+GCsp joint", ranges_joint, args.n_live)
    t0 = time.time()
    chain_joint = run_nautilus_joint(
        photo_like,
        spectro_like,
        ranges_joint,
        args.n_live,
        args.n_workers,
        outdir=outdir,
        verbose=verbose,
    )
    print(
        f"  WL+GCsp joint Nautilus done in {time.time() - t0:.1f}s "
        f"({len(chain_joint)} samples, ESS {_effective_sample_size(chain_joint):.0f})"
    )
    _save_chain(chain_joint, outdir, "joint")

    _report_statistics(
        {
            "WL": _compare_chain_to_fisher(chain_wl, reference["WL"], free_params, FIDUCIAL),
            "GCsp": _compare_chain_to_fisher(chain_gcsp, reference["GCsp"], free_params, FIDUCIAL),
            "WL+GCsp": _compare_chain_to_fisher(
                chain_joint, reference["WL+GCsp"], free_params, FIDUCIAL
            ),
        },
        free_params,
        reference_kind,
        outdir,
    )

    fico.make_triangle_plot(
        chains=[chain_wl, chain_gcsp, chain_joint],
        chain_labels=["WL (Nautilus)", "GCsp (Nautilus)", "WL+GCsp (Nautilus)"],
        colors=["blue", "green", "red"],
        params=free_params,
        truth_values=FIDUCIAL,
        smooth=5,
        bins=12,
        save_plot=True,
        savepath=str(outdir) + "/",
        plot_filename="nautilus_wl_gcsp_combined",
        file_format=".png",
    )
    print(f"Saved: {outdir}/nautilus_wl_gcsp_combined.png")

    # Overlay Fisher ellipses and Nautilus posteriors together for direct comparison.
    # The Fisher matrices used here match what the sampler actually explored:
    # conditional when the nuisances are fixed, marginal when they are sampled.
    fico.make_triangle_plot(
        fishers=[reference["WL"], reference["GCsp"], reference["WL+GCsp"]],
        chains=[chain_wl, chain_gcsp, chain_joint],
        fisher_labels=[
            f"WL (Fisher, {reference_kind})",
            f"GCsp (Fisher, {reference_kind})",
            f"WL+GCsp (Fisher, {reference_kind})",
        ],
        chain_labels=["WL (Nautilus)", "GCsp (Nautilus)", "WL+GCsp (Nautilus)"],
        colors=["blue", "green", "red", "cyan", "lime", "orange"],
        params=free_params,
        truth_values=FIDUCIAL,
        smooth=5,
        bins=12,
        save_plot=True,
        savepath=str(outdir) + "/",
        plot_filename="fisher_vs_nautilus_wl_gcsp",
        file_format=".png",
    )
    print(f"Saved: {outdir}/fisher_vs_nautilus_wl_gcsp.png")


if __name__ == "__main__":
    main()
