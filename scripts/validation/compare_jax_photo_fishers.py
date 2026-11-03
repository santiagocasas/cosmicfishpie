"""Compare the legacy, JAX finite-difference and JAX autodiff photometric forecasts.

The three forecasts share everything except the derivative route:

``legacy-fd``
    The standard NumPy/SciPy ``GCph``+``WL`` observable with the finite-difference engine.
``jax-fd``
    The JAX-native observable (:mod:`cosmicjellyfish.LSSsurvey.jaxcosmo_photo`) with the *same*
    finite-difference engine.
``jax-ad``
    The same JAX-native observable differentiated with ``jax.jacfwd``.

``jax-fd`` versus ``jax-ad`` isolates the derivative method; either versus ``legacy-fd``
isolates the observable implementation.

Usage
-----
    uv run --no-sync python scripts/validation/compare_jax_photo_fishers.py
    uv run --no-sync python scripts/validation/compare_jax_photo_fishers.py --outdir DIR

Nothing is uploaded; outputs are written under ``--outdir`` (default: ``/tmp``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FIDUCIAL = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96, "mnu": 0.0}
FREE = {name: 0.01 for name in ("10^9As", "Omegam", "Omegab", "h", "ns")}


def _options(outdir: Path, tag: str) -> dict:
    results = outdir / tag
    results.mkdir(parents=True, exist_ok=True)
    return {
        "accuracy": 1,
        "derivatives": "3PT",
        "feedback": 0,
        "nonlinear": True,
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "specs_dir": str(ROOT / "cosmicjellyfish/configs/default_survey_specifications") + "/",
        "cosmo_model": "LCDM",
        "code": "jaxcosmo",
        "results_dir": str(results) + "/",
        "outroot": f"jaxcosmo_parity_{tag}",
    }


def _make_calc(outdir: Path, tag: str, provider=None, fiducial=None):
    from cosmicjellyfish.fishermatrix.cosmicfish import FisherMatrix

    return FisherMatrix(
        options=_options(outdir, tag),
        fiducialpars=dict(FIDUCIAL if fiducial is None else fiducial),
        freepars=dict(FREE),
        observables=["GCph", "WL"],
        cosmoModel="LCDM",
        surveyName="Euclid",
        derivative_provider=provider,
    )


def _legacy_observable(calc, photobiaspars=None, iapars=None):
    """The legacy ``ComputeCls`` observable for a :class:`FisherMatrix` configuration.

    ``photobiaspars`` and ``iapars`` default to the configuration's values; pass a modified copy
    to compare at a shifted nuisance point (the bias and intrinsic-alignment fiducials come from
    the survey specification, not from ``fiducialpars``).
    """
    from cosmicjellyfish.LSSsurvey import photo_obs

    observable = photo_obs.ComputeCls(
        calc.fiducialcosmopars,
        calc.photopars,
        calc.IApars if iapars is None else iapars,
        calc.photobiaspars if photobiaspars is None else photobiaspars,
        configuration=calc,
    )
    observable.compute_all()
    return observable


def _relative_stats(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Difference statistics normalised by the largest absolute reference value."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    diff = np.abs(candidate - reference)
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    if scale == 0.0:
        return {
            "max": float(diff.max(initial=0.0)),
            "mean": float(diff.mean() if diff.size else 0.0),
        }
    significant = np.abs(reference) > 1.0e-2 * scale
    rel = diff / scale
    rel_sig = (diff[significant] / np.abs(reference[significant])) if significant.any() else rel
    return {
        "max": float(rel.max(initial=0.0)),
        "mean": float(rel.mean() if rel.size else 0.0),
        "max_significant": float(rel_sig.max(initial=0.0)),
        "n_significant": int(significant.sum()),
    }


def _compare_dicts(reference: dict, candidate: dict) -> dict:
    keys = [k for k in reference if k != "ells" and k in candidate]
    missing = [k for k in reference if k != "ells" and k not in candidate]
    if not keys:
        return {"n_keys": 0, "missing_keys": missing}
    ref = np.concatenate([np.ravel(reference[k]) for k in keys])
    new = np.concatenate([np.ravel(candidate[k]) for k in keys])
    stats = _relative_stats(ref, new)
    stats["n_keys"] = len(keys)
    stats["missing_keys"] = missing

    per_key = {}
    for key in keys:
        per_key[key] = _relative_stats(np.ravel(reference[key]), np.ravel(candidate[key]))["max"]
    stats["worst_keys"] = sorted(per_key.items(), key=lambda item: item[1], reverse=True)[:5]
    return stats


def _sigma(fisher) -> dict[str, float]:
    names = fisher.get_param_names()
    covariance = fisher.get_fisher_inverse()
    return {name: float(value) for name, value in zip(names, np.sqrt(np.diag(covariance)))}


def _matrix(fisher) -> np.ndarray:
    return np.asarray(fisher.get_fisher_matrix(), dtype=float)


def _nonzero_finite_float(text: str) -> float:
    """Parse a finite, non-zero relative step for the central difference."""
    value = float(text)
    if not math.isfinite(value) or value == 0.0:
        raise argparse.ArgumentTypeError("must be a finite, non-zero number")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("/tmp/opencode/jax_parity"))
    parser.add_argument(
        "--fd-step",
        type=_nonzero_finite_float,
        default=0.01,
        help="relative finite-difference step (default 0.01, the CosmicJellyfish default)",
    )
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.fd_step != 0.01:
        FREE.update({name: args.fd_step for name in FREE})

    from cosmicjellyfish.analysis.fisher_matrix import fisher_matrix
    from cosmicjellyfish.fishermatrix.jaxcosmo_derivatives import (
        JaxAutodiffProvider,
        JaxFiniteDifferenceProvider,
    )
    from cosmicjellyfish.LSSsurvey import photo_cov
    from cosmicjellyfish.LSSsurvey.jaxcosmo_photo import JaxPhotoObservable

    summary: dict = {"fiducial": FIDUCIAL, "freeparams": FREE, "variants": {}}

    # ---------------------------------------------------------------- fiducial spectra
    reference = _make_calc(outdir, "reference")
    allpars = dict(reference.allparams_fidus)
    legacy_obs = _legacy_observable(reference)
    jax_observable = JaxPhotoObservable(reference, allpars)
    t0 = time.time()
    jax_cls = jax_observable.cls()
    summary["jax_cls_seconds"] = time.time() - t0
    summary["cls"] = _compare_dicts(legacy_obs.result, jax_cls)

    # ---------------------------------------------------------------- derivatives
    covariance = photo_cov.PhotoCov(
        reference.fiducialcosmopars,
        reference.photopars,
        reference.IApars,
        reference.photobiaspars,
        fiducial_Cls=legacy_obs,
        configuration=reference,
    )
    t0 = time.time()
    derivs_legacy = covariance.compute_derivs()
    summary["derivs_seconds"] = {"legacy-fd": time.time() - t0}
    t0 = time.time()
    derivs_jax_fd = covariance.compute_derivs(derivative_provider=JaxFiniteDifferenceProvider())
    summary["derivs_seconds"]["jax-fd"] = time.time() - t0
    t0 = time.time()
    derivs_jax_ad = covariance.compute_derivs(derivative_provider=JaxAutodiffProvider())
    summary["derivs_seconds"]["jax-ad"] = time.time() - t0

    comparisons = {
        "jax-fd_vs_legacy-fd": (derivs_legacy, derivs_jax_fd),
        "jax-ad_vs_jax-fd": (derivs_jax_fd, derivs_jax_ad),
        "jax-ad_vs_legacy-fd": (derivs_legacy, derivs_jax_ad),
    }
    summary["derivatives"] = {}
    for label, (ref, cand) in comparisons.items():
        per_param = {name: _compare_dicts(ref[name], cand[name]) for name in FREE if name in ref}
        summary["derivatives"][label] = per_param

    # ---------------------------------------------------------------- Fisher matrices
    providers = {
        "legacy-fd": None,
        "jax-fd": JaxFiniteDifferenceProvider,
        "jax-ad": JaxAutodiffProvider,
    }
    matrices = {}
    for label, provider_factory in providers.items():
        provider = provider_factory() if provider_factory else None
        calc = _make_calc(outdir, label, provider=provider)
        t0 = time.time()
        result = calc.compute()
        elapsed = time.time() - t0
        fisher = fisher_matrix(file_name=str(result.file_name))
        matrices[label] = _matrix(fisher)
        summary["variants"][label] = {
            "seconds": elapsed,
            "file": str(result.file_name),
            "sigma": _sigma(fisher),
        }
        print(f"[{label}] {elapsed:.1f}s  -> {result.file_name}")

    base = matrices["legacy-fd"]
    for label in ("jax-fd", "jax-ad"):
        summary["variants"][label]["fisher_vs_legacy-fd"] = _relative_stats(base, matrices[label])
    summary["variants"]["jax-ad"]["fisher_vs_jax-fd"] = _relative_stats(
        matrices["jax-fd"], matrices["jax-ad"]
    )

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (outdir / "REPORT.md").write_text(_report(summary))
    print(json.dumps({k: v["sigma"] for k, v in summary["variants"].items()}, indent=2))
    print(f"Wrote {outdir / 'summary.json'} and {outdir / 'REPORT.md'}")
    missing = bool(summary["cls"].get("missing_keys")) or any(
        stats.get("missing_keys")
        for per_param in summary["derivatives"].values()
        for stats in per_param.values()
    )
    return 1 if missing else 0


def _report(summary: dict) -> str:
    lines = ["# jaxcosmo photometric parity", ""]
    lines.append("Fiducial: `" + json.dumps(summary["fiducial"]) + "`")
    lines.append("")
    lines.append("## C_ell at the fiducial point")
    cls = summary["cls"]
    lines.append(
        f"- {cls.get('n_keys', 0)} spectra; max |dC|/max|C| = {cls.get('max', float('nan')):.3e}, "
        f"mean = {cls.get('mean', float('nan')):.3e}; "
        f"max relative where |C| > 1% of max = {cls.get('max_significant', float('nan')):.3e}"
    )
    worst = ", ".join(f"`{k}` {v:.2e}" for k, v in cls.get("worst_keys", []))
    lines.append(f"- worst spectra: {worst}")
    lines.append(f"- missing reference spectra: {cls.get('missing_keys', [])}")
    lines.append("")
    lines.append("## Derivatives")
    for label, per_param in summary["derivatives"].items():
        worst = max(per_param.items(), key=lambda kv: kv[1].get("max", 0.0), default=(None, {}))
        lines.append(
            f"- **{label}**: worst parameter `{worst[0]}` with "
            f"max |dC|/max|dC| = {worst[1].get('max', float('nan')):.3e}"
        )
        for name, stats in per_param.items():
            lines.append(
                f"  - `{name}`: max {stats.get('max', float('nan')):.3e}, "
                f"max-significant {stats.get('max_significant', float('nan')):.3e}"
            )
    lines.append("")
    lines.append("## Fisher matrices")
    lines.append("")
    lines.append("| variant | seconds | " + " | ".join(FREE) + " |")
    lines.append("|---|---|" + "---|" * len(FREE))
    for label, data in summary["variants"].items():
        sigmas = " | ".join(f"{data['sigma'][name]:.6g}" for name in FREE)
        lines.append(f"| {label} | {data['seconds']:.1f} | {sigmas} |")
    lines.append("")
    for label in ("jax-fd", "jax-ad"):
        stats = summary["variants"][label]["fisher_vs_legacy-fd"]
        lines.append(
            f"- `{label}` vs `legacy-fd`: Fisher max |dF|/max|F| = {stats['max']:.3e}, "
            f"mean = {stats['mean']:.3e}"
        )
    stats = summary["variants"]["jax-ad"]["fisher_vs_jax-fd"]
    lines.append(
        f"- `jax-ad` vs `jax-fd`: Fisher max |dF|/max|F| = {stats['max']:.3e}, "
        f"mean = {stats['mean']:.3e}"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
