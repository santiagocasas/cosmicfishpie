#!/usr/bin/env python3
"""Compare fixed-As ns responses of total-matter Plin and Pnl through CFP's public API.

Run from the repository root with the environment containing CLASS, EuclidEmulator2,
and symbolic_pofk. Defaults reproduce the photometric notebook's profiles and ns step.
Use --plot-only to redraw saved arrays without initializing any cosmology backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Set before importing numerical libraries; explicit user environment values win.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/source/_static/ns_response"
PROFILE_ROOT = ROOT / "cosmicfishpie/configs/default_boltzmann_yaml_files"
CASES = (
    ("CLASS + EE2", "class", "class_config_yaml", "class/ee2_boost_photo.yaml"),
    ("CLASS + HMcode2020", "class", "class_config_yaml", "class/hmcode2020_photo.yaml"),
    ("SYREN-NEW", "symbolic", "symbolic_config_yaml", "symbolic/syren_new_photo.yaml"),
)
FIDUCIAL = {"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96, "mnu": 0.06}
COLORS = ("#0072B2", "#009E73", "#D55E00")
STYLES = ("-", "-.", "--")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ns-step", type=float, default=0.0096, help="Absolute ns half-step")
    parser.add_argument("--redshifts", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--k-min", type=float, default=0.01, help="h/Mpc")
    parser.add_argument("--k-max", type=float, default=3.0, help="h/Mpc")
    parser.add_argument("--k-samples", type=int, default=240)
    parser.add_argument("--cosmo-model", choices=("LCDM", "w0waCDM"), default="LCDM")
    parser.add_argument("--plot-only", action="store_true", help="Read ns_response.npz; no solvers")
    args = parser.parse_args()
    if not args.plot_only:
        if not 0 < args.ns_step < 0.04:
            parser.error("--ns-step must be between 0 and 0.04 (EE2 ns calibration range).")
        if not 0.01 <= args.k_min < args.k_max <= 3.0 or args.k_samples < 4:
            parser.error("Use 0.01 <= k-min < k-max <= 3 h/Mpc and at least four samples.")
        if not args.redshifts or any(not 0 <= z <= 2 for z in args.redshifts):
            parser.error("Redshifts must lie within [0, 2], safely inside the photo spline grids.")
    return args


def file_record(path: Path) -> dict:
    """Record source/profile identity alongside the numerical artifact."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sample_power(cosmology, k: np.ndarray, redshifts: np.ndarray, label: str) -> np.ndarray:
    """Return [linear/nonlinear, z, k] power in (Mpc/h)^3, rejecting extrapolation."""
    h = FIDUCIAL["h"]
    physical_k = k * h
    powers = []
    for nonlinear, name in ((False, "Pk_l"), (True, "Pk_nl")):
        zknots, kknots = getattr(cosmology.results, name).get_knots()
        if (
            physical_k.min() < kknots[0]
            or physical_k.max() > kknots[-1]
            or redshifts.min() < zknots[0]
            or redshifts.max() > zknots[-1]
        ):
            raise ValueError(f"{label} {name}: requested grid exceeds interpolation support.")
        power = np.array(
            [cosmology.Pmm(z, physical_k, nonlinear=nonlinear) * h**3 for z in redshifts]
        )
        if not np.all(np.isfinite(power) & (power > 0)):
            raise FloatingPointError(f"{label} {name}: non-positive or non-finite power.")
        powers.append(power)
    return np.array(powers)


def compute(args: argparse.Namespace) -> tuple[dict, dict]:
    # Lazy imports ensure --plot-only needs just NumPy and Matplotlib.
    import cosmicfishpie
    from cosmicfishpie.configs.context import build_analysis_context
    from cosmicfishpie.cosmology.cosmology import cosmo_functions

    if Path(cosmicfishpie.__file__).resolve().parents[1] != ROOT:
        raise RuntimeError("The interpreter imports a different CosmicFishPie checkout.")
    k = np.geomspace(args.k_min, args.k_max, args.k_samples)
    anchors = np.array([0.2, 0.5, 1.0])
    k = np.unique(np.r_[k, anchors[(anchors >= args.k_min) & (anchors <= args.k_max)]])
    redshifts = np.array(sorted(set(args.redshifts)))
    fiducial = dict(FIDUCIAL)
    if args.cosmo_model == "w0waCDM":
        fiducial.update(w0=-1.0, wa=0.0)
    powers, profiles, resolved_parameters = [], {}, {}
    for label, code, yaml_key, relative_profile in CASES:
        path = PROFILE_ROOT / relative_profile
        print(f"{label}: fiducial", flush=True)
        context = build_analysis_context(
            options={
                "code": code,
                yaml_key: str(path),
                "cosmo_model": args.cosmo_model,
                "nonlinear": True,
                "feedback": 0,
                "specs_dir": str(ROOT / "cosmicfishpie/configs/default_survey_specifications")
                + "/",
            },
            observables=["GCph", "WL"],
            freepars={},
            fiducialpars=fiducial,
            survey_name="Euclid",
            cosmo_model=args.cosmo_model,
        )
        # Context construction already builds this spectrum; do not compute it twice.
        fid = sample_power(context.fiducialcosmo, k, redshifts, label)
        variations = []
        for sign in (-1, 1):
            print(f"{label}: ns {sign:+d} x {args.ns_step:g}", flush=True)
            parameters = {**fiducial, "ns": fiducial["ns"] + sign * args.ns_step}
            cosmology = cosmo_functions(parameters, configuration=context)
            variations.append(sample_power(cosmology, k, redshifts, label))
            del cosmology
        powers.append(np.array([variations[0], fid, variations[1]]))
        profiles[label] = {**file_record(path), "contents": path.read_text()}
        if code == "class":
            resolved_parameters[label] = dict(context.fiducialcosmo.classcosmopars)
        else:
            resolved_parameters[label] = dict(context.fiducialcosmo.symbcosmopars)

    power = np.array(powers)  # backend, minus/fiducial/plus, linear/nonlinear, z, k
    response = (np.log(power[:, 2]) - np.log(power[:, 0])) / (2 * args.ns_step)
    data = {
        "k_hmpc": k,
        "redshifts": redshifts,
        "labels": np.array([case[0] for case in CASES]),
        "power": power,
        "response": response,
        "dP_dns": (power[:, 2] - power[:, 0]) / (2 * args.ns_step),
        "boost_response": response[:, 1] - response[:, 0],
        "ns_step": np.array(args.ns_step),
        "h": np.array(FIDUCIAL["h"]),
        "cosmo_model": np.array(args.cosmo_model),
    }
    packages = {}
    for name in ("cosmicfishpie", "classy", "euclidemu2", "symbolic_pofk", "numpy", "scipy"):
        module = importlib.import_module(name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        packages[name] = {"version": version, "module_path": module.__file__}
    sources = {
        name: file_record(Path(importlib.import_module(name).__file__))
        for name in (
            "cosmicfishpie.cosmology.cosmology",
            "cosmicfishpie.cosmology.symbolic_new",
            "cosmicfishpie.cosmology.ee2",
            "symbolic_pofk.syren_new",
            "symbolic_pofk.linear_new",
        )
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": file_record(Path(__file__)),
        "fiducial": fiducial,
        "cosmo_model": args.cosmo_model,
        "ns_step": args.ns_step,
        "redshifts": redshifts.tolist(),
        "k_hmpc": {"min": float(k.min()), "max": float(k.max()), "samples": len(k)},
        "power_axes": ["backend", "minus/fiducial/plus", "linear/nonlinear", "z", "k"],
        "power_units": "(Mpc/h)^3",
        "response_definition": "[ln P(ns+step) - ln P(ns-step)] / (2 step), fixed As",
        "profiles": profiles,
        "resolved_fiducial_backend_parameters": resolved_parameters,
        "packages": packages,
        "sources": sources,
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
    }
    return data, metadata


def plot_response(data: dict, output: Path, kind: str) -> None:
    """Draw responses and additive EE2 residuals, including through zero crossings."""
    k, zs = data["k_hmpc"], data["redshifts"]
    if kind == "boost":
        values = data["boost_response"]
        title, ylabel = "Nonlinear boost", r"$\partial\ln(P_{\rm nl}/P_{\rm lin})/\partial n_s$"
    else:
        index = 0 if kind == "linear" else 1
        values = data["response"][:, index]
        title = "Linear power" if index == 0 else "Nonlinear power"
        ylabel = r"$\partial\ln P/\partial n_s$"
    fig, axes = plt.subplots(2, len(zs), figsize=(4.4 * len(zs), 6.4), squeeze=False)
    for j, z in enumerate(zs):
        for i, label in enumerate(data["labels"]):
            style = {"color": COLORS[i], "ls": STYLES[i], "lw": 1.9, "label": label}
            axes[0, j].semilogx(k, values[i, j], **style)
            axes[1, j].semilogx(k, values[i, j] - values[0, j], **style)
        if kind == "linear":
            axes[0, j].semilogx(
                k,
                np.log(k * float(data["h"]) / 0.05),
                ":",
                color="0.2",
                lw=1.1,
                label=r"Exact tilt $\ln(kh/0.05)$",
            )
        axes[0, j].set_title(f"z = {z:g}")
        axes[1, j].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        for ax in axes[:, j]:
            ax.axhline(0, color="0.5", lw=0.65)
            ax.grid(alpha=0.2)
            ax.set_xlim(k[0], k[-1])
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(r"$R_{n_s}^{\rm model}-R_{n_s}^{\rm CLASS+EE2}$")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{title}: fixed-$A_s$ tilt response | {str(data['cosmo_model'])} | "
        rf"$\Delta n_s={float(data['ns_step']):g}$"
    )
    fig.tight_layout()
    fig.savefig(output / f"ns_response_{kind}.png", dpi=170)
    plt.close(fig)


def plot_spectra(data: dict, output: Path) -> None:
    """Show fiducial spectrum agreement beside the derivative comparison."""
    k, zs = data["k_hmpc"], data["redshifts"]
    fid = data["power"][:, 1]
    fig, axes = plt.subplots(2, len(zs), figsize=(4.4 * len(zs), 6.4), squeeze=False)
    for j, z in enumerate(zs):
        for i, label in enumerate(data["labels"]):
            for t, kind in enumerate(("linear", "nonlinear")):
                axes[t, j].semilogx(
                    k,
                    100 * (fid[i, t, j] / fid[0, t, j] - 1),
                    color=COLORS[i],
                    ls=STYLES[i],
                    lw=1.9,
                    label=label,
                )
                axes[t, j].set_title(f"{kind.capitalize()}, z = {z:g}")
        for ax in axes[:, j]:
            ax.axhline(0, color="0.5", lw=0.65)
            ax.grid(alpha=0.2)
            ax.set_xlim(k[0], k[-1])
        axes[1, j].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$100\,(P_{\rm model}/P_{\rm CLASS+EE2}-1)$ [%]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Fiducial total-matter spectra | {str(data['cosmo_model'])}")
    fig.tight_layout()
    fig.savefig(output / "ns_fiducial_spectra.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "ns_response.npz"
    if args.plot_only:
        with np.load(archive, allow_pickle=False) as saved:
            data = {name: saved[name] for name in saved.files}
    else:
        data, metadata = compute(args)
        # Serialize metadata before saving either artifact so unsupported types fail early.
        text = json.dumps(metadata, indent=2, allow_nan=False)
        np.savez_compressed(archive, **data)
        (output / "ns_response_metadata.json").write_text(text + "\n")
    for kind in ("linear", "nonlinear", "boost"):
        plot_response(data, output, kind)
    plot_spectra(data, output)
    for j, z in enumerate(data["redshifts"]):
        for anchor in (0.2, 0.5, 1.0):
            matches = np.flatnonzero(np.isclose(data["k_hmpc"], anchor, rtol=1e-12))
            if matches.size:
                vals = data["response"][:, 1, j, matches[0]]
                print(f"z={z:g}, k={anchor:g}: Pnl ns responses (EE2, HMcode, SYREN) = {vals}")
    print(f"Saved four figures and numerical artifacts in {output}")


if __name__ == "__main__":
    main()
