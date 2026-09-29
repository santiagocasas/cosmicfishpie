#!/usr/bin/env python
"""Run one checkpointable WL, GCsp, or joint Nautilus case for batch systems.

Unlike ``wl_gcsp_fisher_nautilus_demo.py``, this runner executes exactly one
case, makes the target effective sample size explicit, and writes all durable
artifacts beneath ``--run-dir/<case>/``. It is intended for Slurm jobs; run the
post-processor only after all three cases have completed successfully.

Examples
--------
    uv run python scripts/run_wl_gcsp_nautilus_case.py \
        --case gcsp --sample-nuisances --n-live 500 --n-eff 2000 \
        --workers 16 --run-dir /scratch/$USER/cfp/wl-gcsp-500

    # Resume only an identical interrupted case.
    uv run python scripts/run_wl_gcsp_nautilus_case.py \
        --case gcsp --sample-nuisances --n-live 500 --n-eff 2000 \
        --workers 16 --run-dir /scratch/$USER/cfp/wl-gcsp-500 --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import wl_gcsp_fisher_nautilus_demo as demo
from nautilus import Sampler

from cosmicfishpie.likelihood.photo_like import PhotometricLikelihood
from cosmicfishpie.likelihood.spectro_like import SpectroLikelihood


def _git_revision() -> str:
    """Return the checked-out revision, or ``unknown`` outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_free_params(raw: str, parser: argparse.ArgumentParser) -> list[str]:
    free_params = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in free_params if name not in demo.PRIOR_RANGES]
    if not free_params or unknown:
        parser.error(
            f"--free-params must be a nonempty subset of {list(demo.PRIOR_RANGES)}; "
            f"got {free_params}"
        )
    return free_params


def _checkpoint_path(case_dir: Path, case: str, n_live: int, n_eff: int, ranges: dict) -> Path:
    """Make changed sampling targets start a fresh checkpoint, never stale bounds."""
    return case_dir / f"nautilus_{case}_nlive{n_live}_neff{n_eff}_dim{len(ranges)}.hdf5"


def _run_wl(
    fm,
    ranges: dict,
    checkpoint: Path,
    n_live: int,
    n_eff: int,
    workers: int,
    resume: bool,
    verbose: bool,
):
    likelihood = PhotometricLikelihood(cosmo_data=fm, cosmo_theory=fm)
    prior = likelihood.create_nautilus_prior(ranges)
    sampler = likelihood.run_nautilus(
        prior=prior,
        sampler_kwargs={
            "n_live": n_live,
            "n_networks": 1,
            "n_batch": demo._n_batch(n_live, workers),
            "pool": workers,
            "filepath": str(checkpoint),
            "resume": resume,
        },
        run_kwargs={"n_eff": n_eff, "verbose": verbose, "discard_exploration": True},
    )
    return demo._posterior_dataframe(sampler, prior)


def _run_gcsp(
    fm,
    ranges: dict,
    checkpoint: Path,
    n_live: int,
    n_eff: int,
    workers: int,
    resume: bool,
    verbose: bool,
):
    likelihood = SpectroLikelihood(cosmoFM_data=fm, cosmoFM_theory=fm)
    prior = likelihood.create_nautilus_prior(ranges)
    sampler = likelihood.run_nautilus(
        prior=prior,
        sampler_kwargs={
            "n_live": n_live,
            "n_networks": 1,
            "n_batch": demo._n_batch(n_live, workers),
            "pool": workers,
            "filepath": str(checkpoint),
            "resume": resume,
        },
        run_kwargs={"n_eff": n_eff, "verbose": verbose, "discard_exploration": True},
    )
    return demo._posterior_dataframe(sampler, prior)


def _run_joint(
    fm_wl,
    fm_gcsp,
    ranges: dict,
    checkpoint: Path,
    n_live: int,
    n_eff: int,
    workers: int,
    resume: bool,
    verbose: bool,
):
    """Run one shared-prior sample with the sum of the two probe log-likelihoods."""
    photo_like = PhotometricLikelihood(cosmo_data=fm_wl, cosmo_theory=fm_wl)
    spectro_like = SpectroLikelihood(cosmoFM_data=fm_gcsp, cosmoFM_theory=fm_gcsp)
    prior = photo_like.create_nautilus_prior(ranges)

    def joint_loglike(param_dict):
        return photo_like.loglike(param_dict=param_dict) + spectro_like.loglike(
            param_dict=param_dict
        )

    sampler = Sampler(
        prior,
        joint_loglike,
        n_live=n_live,
        n_networks=1,
        n_batch=demo._n_batch(n_live, workers),
        pool=workers,
        pass_dict=True,
        filepath=str(checkpoint),
        resume=resume,
    )
    sampler.run(n_eff=n_eff, verbose=verbose, discard_exploration=True)
    return demo._posterior_dataframe(sampler, prior)


def _save_fisher(fisher, case_dir: Path, label: str) -> Path:
    """Persist a Fisher object with parameter names for the dependent plot job."""
    stem = case_dir / f"fisher_{label}"
    fisher.save_to_file(stem)
    return stem.with_suffix(".txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("wl", "gcsp", "joint"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--free-params", default="Omegam,sigma8")
    parser.add_argument("--sample-nuisances", action="store_true")
    parser.add_argument("--nuisance-sigma", type=float, default=5.0)
    parser.add_argument("--n-live", type=int, required=True)
    parser.add_argument("--n-eff", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume this exact checkpoint.")
    parser.add_argument("--quiet-sampler", action="store_true")
    args = parser.parse_args()

    if args.n_live < 2 or args.n_eff < 1 or args.workers < 1:
        parser.error("--n-live must be >= 2; --n-eff and --workers must be positive")
    if not np.isfinite(args.nuisance_sigma) or args.nuisance_sigma <= 0:
        parser.error("--nuisance-sigma must be finite and positive")

    free_params = _parse_free_params(args.free_params, parser)
    case_dir = args.run_dir.resolve() / args.case
    case_dir.mkdir(parents=True, exist_ok=True)

    # Keep concurrent Slurm cases from competing for the demo's local results/ path.
    demo.COMMON_OPTIONS["results_dir"] = str(case_dir / "fisher") + "/"
    print(f"Case: {args.case}; run directory: {case_dir}")
    print(f"Resources: {args.workers} Nautilus workers; n_live={args.n_live}; n_eff={args.n_eff}")

    fm_wl = fisher_wl = fm_gcsp = fisher_gcsp = None
    if args.case in ("wl", "joint"):
        fm_wl, fisher_wl = demo.compute_fisher_wl(free_params)
    if args.case in ("gcsp", "joint"):
        fm_gcsp, fisher_gcsp = demo.compute_fisher_gcsp(free_params)

    if args.case == "wl":
        raw_fisher = fisher_wl
        reference = (
            demo.marginal_fisher(raw_fisher, free_params)
            if args.sample_nuisances
            else demo.conditional_fisher(raw_fisher, free_params)
        )
        ranges = (
            demo.nuisance_prior_ranges(raw_fisher, free_params, args.nuisance_sigma)
            if args.sample_nuisances
            else {name: demo.PRIOR_RANGES[name] for name in free_params}
        )
    elif args.case == "gcsp":
        raw_fisher = fisher_gcsp
        reference = (
            demo.marginal_fisher(raw_fisher, free_params)
            if args.sample_nuisances
            else demo.conditional_fisher(raw_fisher, free_params)
        )
        ranges = (
            demo.nuisance_prior_ranges(raw_fisher, free_params, args.nuisance_sigma)
            if args.sample_nuisances
            else {name: demo.PRIOR_RANGES[name] for name in free_params}
        )
    else:
        raw_fisher = fisher_wl + fisher_gcsp
        reference = (
            demo.marginal_fisher(raw_fisher, free_params)
            if args.sample_nuisances
            else demo.conditional_fisher(raw_fisher, free_params)
        )
        ranges = (
            demo.nuisance_prior_ranges(raw_fisher, free_params, args.nuisance_sigma)
            if args.sample_nuisances
            else {name: demo.PRIOR_RANGES[name] for name in free_params}
        )

    checkpoint = _checkpoint_path(case_dir, args.case, args.n_live, args.n_eff, ranges)
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint}. Re-run with --resume only if this "
            "case, priors, n_live, and n_eff are unchanged."
        )

    started = time.time()
    if args.case == "wl":
        chain = _run_wl(
            fm_wl,
            ranges,
            checkpoint,
            args.n_live,
            args.n_eff,
            args.workers,
            args.resume,
            not args.quiet_sampler,
        )
    elif args.case == "gcsp":
        chain = _run_gcsp(
            fm_gcsp,
            ranges,
            checkpoint,
            args.n_live,
            args.n_eff,
            args.workers,
            args.resume,
            not args.quiet_sampler,
        )
    else:
        chain = _run_joint(
            fm_wl,
            fm_gcsp,
            ranges,
            checkpoint,
            args.n_live,
            args.n_eff,
            args.workers,
            args.resume,
            not args.quiet_sampler,
        )

    chain_path = demo._save_chain(chain, case_dir, args.case)
    reference_path = _save_fisher(reference, case_dir, "reference")
    raw_path = _save_fisher(raw_fisher, case_dir, "raw")
    metadata = {
        "case": args.case,
        "git_revision": _git_revision(),
        "free_params": free_params,
        "sample_nuisances": args.sample_nuisances,
        "nuisance_sigma": args.nuisance_sigma,
        "reference_kind": "marginal" if args.sample_nuisances else "conditional",
        "sampled_parameters": list(ranges),
        "prior_ranges": {name: list(bounds) for name, bounds in ranges.items()},
        "n_live": args.n_live,
        "n_eff": args.n_eff,
        "workers": args.workers,
        "checkpoint": str(checkpoint),
        "chain": str(chain_path),
        "fisher_reference": str(reference_path),
        "fisher_raw": str(raw_path),
        "elapsed_seconds": time.time() - started,
        "effective_sample_size": demo._effective_sample_size(chain),
    }
    metadata_path = case_dir / "case_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Case complete in {metadata['elapsed_seconds']:.1f}s; ESS {metadata['effective_sample_size']:.0f}"
    )
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
