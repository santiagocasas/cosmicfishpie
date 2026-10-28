"""Compare the legacy, JAX finite-difference and JAX autodiff spectroscopic forecasts.

The three forecasts share everything except the derivative route:

``legacy-fd``
    The standard NumPy/SciPy galaxy-clustering ``GCsp`` observable with the finite-difference
    engine.
``jax-fd``
    The JAX-native observable (:mod:`cosmicfishpie.LSSsurvey.jaxcosmo_spectro`) with the *same*
    finite-difference engine.
``jax-ad``
    The same JAX-native observable differentiated with ``jax.jacfwd``.

``jax-fd`` versus ``jax-ad`` isolates the derivative method; either versus ``legacy-fd``
isolates the observable implementation.

Usage
-----
    uv run --no-sync python scripts/validation/compare_jax_spectro_fishers.py
    uv run --no-sync python scripts/validation/compare_jax_spectro_fishers.py --outdir DIR

Nothing is uploaded; outputs are written under ``--outdir`` (default: ``/tmp``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FIDUCIAL = {
    "10^9As": 2.1,
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "mnu": 0.0,
}
FREE = {name: 0.01 for name in ("10^9As", "Omegam", "Omegab", "h", "ns")}
FREE.update({"lnbg_1": 0.01, "lnbg_2": 0.01, "lnbg_3": 0.01, "lnbg_4": 0.01})

_OBS_KEYS = ("z_bins", "ells")


def _options(outdir: Path, tag: str) -> dict:
    results = outdir / tag
    results.mkdir(parents=True, exist_ok=True)
    return {
        "accuracy": 1,
        "derivatives": "3PT",
        "feedback": 0,
        "nonlinear": True,
        "survey_name": "Euclid",
        "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
        "specs_dir": str(ROOT / "cosmicfishpie/configs/default_survey_specifications") + "/",
        "cosmo_model": "LCDM",
        "code": "jaxcosmo",
        "results_dir": str(results) + "/",
        "outroot": f"jaxcosmo_spectro_parity_{tag}",
    }


def _make_calc(outdir: Path, tag: str, provider=None):
    from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix

    return FisherMatrix(
        options=_options(outdir, tag),
        fiducialpars=dict(FIDUCIAL),
        freepars=dict(FREE),
        observables=["GCsp"],
        cosmoModel="LCDM",
        surveyName="Euclid",
        derivative_provider=provider,
    )


def _legacy_engine(calc):
    """Build the legacy fiducial observable and the ``SpectroDerivs`` engine of ``calc``."""
    from cosmicfishpie.LSSsurvey import spectro_cov, spectro_obs

    calc.set_pk_settings()
    calc.obs_spectrum = ["g", "g"]
    calc.pk_obs_fid = spectro_obs.ComputeGalSpectro(
        cosmopars=calc.fiducialcosmopars,
        fiducial_cosmopars=calc.fiducialcosmopars,
        spectrobiaspars=calc.Spectrobiaspars,
        spectrononlinearpars=calc.Spectrononlinpars,
        IMbiaspars=calc.IMbiaspars,
        PShotpars=calc.PShotpars,
        configuration=calc,
    )
    pk_cov = spectro_cov.SpectroCov(
        calc.fiducialcosmopars,
        fiducial_specobs=calc.pk_obs_fid,
        bias_samples=calc.obs_spectrum,
        configuration=calc,
    )
    calc.zmids = pk_cov.inter_z_bin_mids
    calc.fish_z_arr = np.array(calc.zmids)
    calc.pk_deriv = spectro_cov.SpectroDerivs(
        calc.fish_z_arr,
        calc.Pk_kmesh,
        calc.Pk_mumesh,
        fiducial_spectro_obj=calc.pk_obs_fid,
        bias_samples=calc.obs_spectrum,
        configuration=calc,
    )
    return calc.pk_deriv


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
    keys = [k for k in reference if k not in _OBS_KEYS and k in candidate]
    if not keys:
        return {"n_keys": 0}
    ref = np.concatenate([np.ravel(reference[k]) for k in keys])
    new = np.concatenate([np.ravel(candidate[k]) for k in keys])
    stats = _relative_stats(ref, new)
    stats["n_keys"] = len(keys)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("/tmp/opencode/jax_spectro_parity"))
    parser.add_argument(
        "--fd-step",
        type=float,
        default=0.01,
        help="relative finite-difference step (default 0.01, the CosmicFishPie default)",
    )
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.fd_step != 0.01:
        FREE.update({name: args.fd_step for name in FREE})

    from cosmicfishpie.analysis.fisher_matrix import fisher_matrix
    from cosmicfishpie.fishermatrix.jaxcosmo_derivatives import (
        JaxAutodiffProvider,
        JaxFiniteDifferenceProvider,
    )
    from cosmicfishpie.LSSsurvey.jaxcosmo_spectro import JaxSpectroObservable

    summary: dict = {"fiducial": FIDUCIAL, "freeparams": FREE, "variants": {}}

    reference = _make_calc(outdir, "reference")
    legacy_derivs_engine = _legacy_engine(reference)
    fiducial_allpars = dict(reference.pk_obs_fid.fiducial_allpars)
    summary["fiducial_allpars"] = {k: v for k, v in fiducial_allpars.items() if np.ndim(v) == 0}

    # ---------------------------------------------------------------- fiducial ln P
    legacy_lnp = legacy_derivs_engine.get_obs(fiducial_allpars)
    jax_observable = JaxSpectroObservable(reference, fiducial_allpars)
    t0 = time.time()
    jax_lnp = jax_observable.lnp()
    summary["jax_lnp_seconds"] = time.time() - t0
    summary["lnp"] = _compare_dicts(legacy_lnp, jax_lnp)

    # ---------------------------------------------------------------- derivatives
    t0 = time.time()
    derivs_legacy = legacy_derivs_engine.compute_derivs(
        freeparams=reference.freeparams,
    )
    summary["derivs_seconds"] = {"legacy-fd": time.time() - t0}
    t0 = time.time()
    derivs_jax_fd = legacy_derivs_engine.compute_derivs(
        freeparams=reference.freeparams,
        derivative_provider=JaxFiniteDifferenceProvider(),
    )
    summary["derivs_seconds"]["jax-fd"] = time.time() - t0
    t0 = time.time()
    derivs_jax_ad = legacy_derivs_engine.compute_derivs(
        freeparams=reference.freeparams,
        derivative_provider=JaxAutodiffProvider(),
    )
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
    summary["variants"]["jax-fd"]["sigma_vs_legacy-fd"] = {
        name: summary["variants"]["jax-fd"]["sigma"][name]
        / summary["variants"]["legacy-fd"]["sigma"][name]
        - 1.0
        for name in summary["variants"]["legacy-fd"]["sigma"]
    }
    summary["variants"]["jax-ad"]["sigma_vs_legacy-fd"] = {
        name: summary["variants"]["jax-ad"]["sigma"][name]
        / summary["variants"]["legacy-fd"]["sigma"][name]
        - 1.0
        for name in summary["variants"]["legacy-fd"]["sigma"]
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (outdir / "REPORT.md").write_text(_report(summary))
    print(json.dumps({k: v["sigma"] for k, v in summary["variants"].items()}, indent=2))
    print(f"Wrote {outdir / 'summary.json'} and {outdir / 'REPORT.md'}")
    return 0


def _report(summary: dict) -> str:
    lines = ["# jaxcosmo spectroscopic parity", ""]
    lines.append("Fiducial: `" + json.dumps(summary["fiducial"]) + "`")
    lines.append("")
    lines.append("## ln P(k, mu) at the fiducial point")
    lnp = summary["lnp"]
    lines.append(
        f"- {lnp.get('n_keys', 0)} redshift bins; max |dlnP|/max|lnP| = "
        f"{lnp.get('max', float('nan')):.3e}, mean = {lnp.get('mean', float('nan')):.3e}; "
        f"max relative where |lnP| > 1% of max = {lnp.get('max_significant', float('nan')):.3e}"
    )
    worst = ", ".join(f"`{k}` {v:.2e}" for k, v in lnp.get("worst_keys", []))
    lines.append(f"- worst bins: {worst}")
    lines.append("")
    lines.append("## Derivatives")
    for label, per_param in summary["derivatives"].items():
        worst = max(per_param.items(), key=lambda kv: kv[1].get("max", 0.0), default=(None, {}))
        lines.append(
            f"- **{label}**: worst parameter `{worst[0]}` with "
            f"max |dP|/max|dP| = {worst[1].get('max', float('nan')):.3e}"
        )
        for name, stats in per_param.items():
            lines.append(
                f"  - `{name}`: max {stats.get('max', float('nan')):.3e}, "
                f"max-significant {stats.get('max_significant', float('nan')):.3e}"
            )
    lines.append("")
    lines.append("## Fisher matrices")
    lines.append("")
    params = list(FREE)
    lines.append("| variant | seconds | " + " | ".join(params) + " |")
    lines.append("|---|---|" + "---|" * len(params))
    for label, data in summary["variants"].items():
        sigmas = " | ".join(f"{data['sigma'].get(name, float('nan')):.6g}" for name in params)
        lines.append(f"| {label} | {data['seconds']:.1f} | {sigmas} |")
    lines.append("")
    lines.append("| parameter | jax-fd sigma vs legacy-fd | jax-ad sigma vs legacy-fd |")
    lines.append("|---|---|---|")
    for name in params:
        lines.append(
            f"| `{name}` | "
            f"{100.0 * summary['variants']['jax-fd']['sigma_vs_legacy-fd'][name]:+.3f}% | "
            f"{100.0 * summary['variants']['jax-ad']['sigma_vs_legacy-fd'][name]:+.3f}% |"
        )
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
