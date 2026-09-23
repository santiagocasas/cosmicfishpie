"""Plot a direct 2D GCsp likelihood grid against the Fisher quadratic form.

Default: 41 x 81 points, +/-4 Fisher marginal sigmas, sinh-spaced to resolve
the narrow sigma8 contour. All other parameters are fixed, as in the demo.
Contours are likelihood-ratio levels, NOT integrated posterior credible regions.
Run: uv run python scripts/gcsp_likelihood_grid.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gcsp_likelihood_hessian import PARAMS, build_problem
from matplotlib.lines import Line2D


def focused_axis(center, sigma, size, width, focus):
    """Symmetric nonuniform axis; odd size includes the fiducial exactly."""
    t = np.linspace(-1, 1, size)
    offsets = t if focus == 0 else np.sinh(focus * t) / np.sinh(focus)
    return center + width * sigma * offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=41)
    parser.add_argument("--ny", type=int, default=81)
    parser.add_argument("--width", type=float, default=4.0, help="Extent in Fisher marginal sigmas")
    parser.add_argument(
        "--focus", type=float, default=3.0, help="Central concentration; 0 = uniform"
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/gcsp_diagnostics/grid"))
    args = parser.parse_args()
    if any(n < 5 or n % 2 == 0 for n in (args.nx, args.ny)):
        parser.error("nx and ny must be odd integers >= 5")
    if not np.isfinite(args.width) or args.width <= 0:
        parser.error("width must be finite and positive")
    if not np.isfinite(args.focus) or not 0 <= args.focus <= 10:
        parser.error("focus must be between 0 and 10")

    nll, fishers, center = build_problem(args.outdir)
    # The grid varies only PARAMS and keeps nuisances fixed, so CONDITIONAL is the
    # like-for-like Fisher. MARGINAL is drawn too because it is what the triangle
    # plots show; the gap between them is the bias/sigma8 degeneracy.
    fisher = fishers["conditional"]
    marginal = fishers["marginal"]
    if np.any(np.linalg.eigvalsh(fisher) <= 0):
        raise ValueError("Fisher must be positive definite to define grid extents")
    # Size the grid with the wider (marginal) errors so both contours fit inside it.
    sigma = np.sqrt(np.diag(np.linalg.inv(marginal)))
    x = focused_axis(center[0], sigma[0], args.nx, args.width, args.focus)
    y = focused_axis(center[1], sigma[1], args.ny, args.width, args.focus)
    # Keep comparison inside the demo's uniform prior; never silently clip it.
    if x[0] < 0.28 or x[-1] > 0.36 or y[0] < 0.75 or y[-1] > 0.87:
        raise ValueError("Grid exceeds demo priors; reduce --width")
    xx, yy = np.meshgrid(x, y)
    offsets = np.stack([xx - center[0], yy - center[1]], axis=-1)
    fisher_chi2 = np.einsum("...i,ij,...j->...", offsets, fisher, offsets)
    marginal_chi2 = np.einsum("...i,ij,...j->...", offsets, marginal, offsets)
    values = np.full(xx.shape, np.nan)
    f0 = nll(center)
    metadata = dict(
        parameters=PARAMS,
        fiducial=center.tolist(),
        nx=args.nx,
        ny=args.ny,
        width=args.width,
        focus=args.focus,
        nll_fiducial=f0,
        central_grid_steps=[
            float(x[args.nx // 2 + 1] - center[0]),
            float(y[args.ny // 2 + 1] - center[1]),
        ],
    )
    (args.outdir / "grid_settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Evaluating {args.nx * args.ny} points, sequentially.", flush=True)
    start = time.monotonic()
    for iy, yvalue in enumerate(y):
        for ix, xvalue in enumerate(x):
            values[iy, ix] = nll(np.array([xvalue, yvalue]))
        np.savez_compressed(
            args.outdir / "likelihood_grid.npz",
            x=x,
            y=y,
            nll=values,
            fisher=fisher,
            fisher_chi2=fisher_chi2,
            fisher_marginal=marginal,
            marginal_chi2=marginal_chi2,
            center=center,
            nll_fiducial=f0,
            parameters=np.array(PARAMS),
        )
        elapsed = time.monotonic() - start
        remaining = elapsed / (iy + 1) * (args.ny - iy - 1)
        print(f"Row {iy + 1}/{args.ny}: elapsed {elapsed:.1f}s, ETA {remaining:.1f}s", flush=True)

    # Reference to the fiducial intentionally exposes offsets; do not hide them
    # by automatically subtracting the sampled grid minimum.
    delta_chi2 = 2 * (values - f0)
    minimum = np.unravel_index(np.argmin(values), values.shape)
    print(
        "Grid minimum:",
        (x[minimum[1]], y[minimum[0]]),
        "delta chi2 relative to fiducial:",
        delta_chi2[minimum],
        flush=True,
    )
    levels = [2.30, 6.18]  # 68.3%, 95.4% enclosed for a Gaussian in TWO dimensions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, delta_chi2, levels=levels, colors="tab:orange", linewidths=2)
    ax.contour(
        xx, yy, fisher_chi2, levels=levels, colors="tab:green", linewidths=2, linestyles="--"
    )
    ax.contour(
        xx, yy, marginal_chi2, levels=levels, colors="tab:blue", linewidths=2, linestyles=":"
    )
    ax.axvline(center[0], color="black", linewidth=0.7)
    ax.axhline(center[1], color="black", linewidth=0.7)
    ax.set(
        xlabel=r"$\Omega_m$",
        ylabel=r"$\sigma_8$",
        title=r"GCsp: $2[-\log L(\theta)+\log L(\theta_0)]$ vs Fisher",
    )
    ax.legend(
        handles=[
            Line2D([], [], color="tab:orange", label="Likelihood (nuisances fixed)"),
            Line2D([], [], color="tab:green", linestyle="--", label="Fisher conditional"),
            Line2D([], [], color="tab:blue", linestyle=":", label="Fisher marginal"),
        ]
    )
    fig.tight_layout()
    fig.savefig(args.outdir / "grid_vs_fisher.png", dpi=180)
    plt.close(fig)
    print("Saved", args.outdir / "grid_vs_fisher.png")
    print("These are likelihood-ratio contours; they are not posterior credible contours.")
    print("Inspect grid resolution and boundary coverage before interpreting the contour shape.")


if __name__ == "__main__":
    main()
