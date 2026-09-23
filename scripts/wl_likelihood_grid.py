"""Scan the Euclid WL likelihood on a 2D grid and compare it against the Fisher forecast.

Motivation
----------
The ``--sample-nuisances`` Nautilus run reproduces the marginal Fisher forecast for
GCsp (sigma ratios 1.00 / 1.01) but not for WL, where the sampled posterior is
~30% narrower than the Fisher ellipse and displaced by more than one sigma along
the Omegam-sigma8 degeneracy.  Prior truncation was ruled out: no boundary carries
posterior weight, and the box sits at ~8.4 sigma along the degeneracy major axis.

What remains is the hypothesis that the WL likelihood is genuinely non-Gaussian,
with the distortion concentrated in the intrinsic-alignment sector (the chain mean
of ``AIA`` sits ~2.6 sigma from its fiducial value).  This script evaluates the
likelihood directly on a grid, so the answer does not depend on any sampler.

Scope and caveats
-----------------
This is a *conditional* slice: the parameters that are not scanned are held at
their fiducial values.  It therefore shows the shape of the likelihood surface,
and is compared against the *conditional* Fisher.  The marginal Fisher is also
drawn for reference, but a 2D slice cannot reproduce a marginalisation over the
remaining dimensions - only a sampler or a profile can do that.

The contours are likelihood-*ratio* levels (Delta chi^2 = 2.30, 6.18), not
integrated posterior credible regions.  They coincide only in the Gaussian limit,
which is precisely what is being tested here.

Examples
--------
Scan the intrinsic-alignment plane where the distortion is expected::

    uv run python scripts/wl_likelihood_grid.py --params Omegam AIA --nx 31 --ny 31 --workers 8

Scan the cosmological plane for comparison::

    uv run python scripts/wl_likelihood_grid.py --params Omegam sigma8 --nx 31 --ny 31 --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Cosmological fiducial for the symbolic backend.  Only flat LCDM is supported,
# so mnu/Neff/w0/wa are absent by construction.
FIDUCIAL_COSMO = {
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "sigma8": 0.815584,
}

# Hand-picked boxes used by the demo script.  Only the cosmological parameters
# have them; nuisance ranges are derived from the Fisher forecast at run time.
COSMO_PRIOR_RANGES = {
    "Omegam": (0.28, 0.36),
    "sigma8": (0.75, 0.87),
}

# Set once in the parent process before the worker pool forks.  Workers inherit
# it through copy-on-write, which avoids pickling the likelihood object.
_NLL = None


def build_problem(outdir: Path, free_params: tuple[str, ...]):
    """Build the WL Fisher matrix and a callable that returns -log L.

    The Fisher matrix is computed with ``free_params`` varied, which makes
    CosmicFishPie add its intrinsic-alignment nuisances automatically.  The
    returned likelihood reuses the ``FisherMatrix`` instance, so the cached
    ``photo_obs_fid`` and ``photo_LSS`` attributes are not recomputed.

    Parameters
    ----------
    outdir : Path
        Directory for CosmicFishPie's own exported Fisher files.
    free_params : tuple of str
        Parameters varied when building the Fisher matrix.

    Returns
    -------
    nll : callable
        Maps a dictionary of parameter values to ``-log L``.  Memoised.
    fishers : dict
        ``full`` matrix, ``param_names``, and the fiducial values of every
        parameter the Fisher matrix knows about.
    """
    from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix
    from cosmicfishpie.likelihood.photo_like import PhotometricLikelihood

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
        outroot="WL_grid_",
    )

    fm = FisherMatrix(
        fiducialpars=FIDUCIAL_COSMO.copy(),
        freepars={p: 0.01 for p in free_params},
        options=options,
        observables=["WL"],
        cosmoModel="LCDM",
        surveyName="Euclid",
    )
    fisher = fm.compute()

    names = list(fisher.param_names)
    fiducial = {name: float(value) for name, value in zip(names, fisher.param_fiducial)}
    fishers = {
        "full": np.asarray(fisher.fisher_matrix),
        "param_names": names,
        "fiducial": fiducial,
    }

    # PhotometricLikelihood uses cosmo_data/cosmo_theory, unlike SpectroLikelihood.
    likelihood = PhotometricLikelihood(cosmo_data=fm, cosmo_theory=fm)

    cache: dict[tuple, float] = {}

    def nll(params: dict) -> float:
        key = tuple(sorted((k, float(v)) for k, v in params.items()))
        if key not in cache:
            value = -float(likelihood.loglike(param_dict=dict(params)))
            if not np.isfinite(value):
                raise ValueError(f"Non-finite -log L at {params}: {value}")
            cache[key] = value
        return cache[key]

    return nll, fishers


def submatrix(matrix: np.ndarray, names: list[str], wanted: tuple[str, ...]) -> np.ndarray:
    """Return the conditional (non-marginalised) sub-block for ``wanted``."""
    missing = [p for p in wanted if p not in names]
    if missing:
        raise ValueError(f"Fisher matrix is missing {missing}; it has {names}")
    index = [names.index(p) for p in wanted]
    return matrix[np.ix_(index, index)]


def marginal_submatrix(matrix: np.ndarray, names: list[str], wanted: tuple[str, ...]) -> np.ndarray:
    """Return the marginalised Fisher for ``wanted``, inverting twice."""
    index = [names.index(p) for p in wanted]
    covariance = np.linalg.inv(matrix)[np.ix_(index, index)]
    return np.linalg.inv(covariance)


def focused_axis(center: float, sigma: float, size: int, width: float, focus: float) -> np.ndarray:
    """Build a symmetric axis, optionally denser near the centre.

    ``focus=0`` gives a uniform axis.  Larger values concentrate points near
    ``center`` via a sinh mapping, which resolves a narrow core without losing
    the tails.  An odd ``size`` always includes ``center`` exactly.
    """
    t = np.linspace(-1.0, 1.0, size)
    offsets = t if focus == 0 else np.sinh(focus * t) / np.sinh(focus)
    return center + width * sigma * offsets


def _evaluate(task):
    """Worker entry point: evaluate ``-log L`` at one grid point."""
    iy, ix, params = task
    return iy, ix, _NLL(params)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan the WL likelihood on a 2D grid and compare against the Fisher forecast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--params",
        nargs=2,
        metavar=("X", "Y"),
        default=["Omegam", "AIA"],
        help="The two parameters to scan. Any of Omegam, sigma8, AIA, betaIA, etaIA.",
    )
    parser.add_argument(
        "--free-params",
        default="Omegam,sigma8",
        help="Cosmological parameters varied when building the Fisher matrix.",
    )
    parser.add_argument("--nx", type=int, default=31, help="Grid points along X (odd, >=5).")
    parser.add_argument("--ny", type=int, default=31, help="Grid points along Y (odd, >=5).")
    parser.add_argument(
        "--scale",
        choices=("marginal", "conditional"),
        default="conditional",
        help=(
            "Which Fisher sigmas set the grid extent. 'conditional' matches what "
            "a fixed-nuisance slice actually probes; 'marginal' shows the full "
            "forecast ellipse but may leave the likelihood contour very thin."
        ),
    )
    parser.add_argument(
        "--width",
        type=float,
        default=4.0,
        help="Half-width of the grid, in units of the sigmas chosen by --scale.",
    )
    parser.add_argument(
        "--focus",
        type=float,
        default=2.0,
        help="Concentration of points near the centre. 0 gives a uniform grid.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel processes for the grid evaluation. The grid is embarrassingly parallel.",
    )
    parser.add_argument(
        "--outdir",
        default="results/wl_diagnostics/grid",
        help="Directory for the grid data, settings, and figure.",
    )
    args = parser.parse_args()

    for label, size in (("--nx", args.nx), ("--ny", args.ny)):
        if size < 5 or size % 2 == 0:
            parser.error(f"{label} must be odd and at least 5, got {size}")
    if not np.isfinite(args.width) or args.width <= 0:
        parser.error(f"--width must be finite and positive, got {args.width}")
    if not 0 <= args.focus <= 10:
        parser.error(f"--focus must lie in [0, 10], got {args.focus}")
    if args.workers < 1:
        parser.error(f"--workers must be at least 1, got {args.workers}")

    scan = tuple(args.params)
    if scan[0] == scan[1]:
        parser.error(f"--params must name two different parameters, got {scan}")

    free_params = tuple(p.strip() for p in args.free_params.split(",") if p.strip())
    unknown = [p for p in free_params if p not in FIDUCIAL_COSMO]
    if unknown:
        parser.error(f"Unknown free parameters {unknown}; choose from {sorted(FIDUCIAL_COSMO)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"WL likelihood grid: {scan[0]} vs {scan[1]}")
    print("=" * 70)

    nll, fishers = build_problem(outdir, free_params)
    names = fishers["param_names"]
    fiducial = fishers["fiducial"]
    full = fishers["full"]

    print(f"  Fisher parameters: {names}")

    conditional = submatrix(full, names, scan)
    marginal = marginal_submatrix(full, names, scan)
    if np.any(np.linalg.eigvalsh(conditional) <= 0):
        raise ValueError("Conditional Fisher block is not positive definite; cannot size the grid.")

    center = np.array([fiducial[p] for p in scan])
    sigma_marginal = np.sqrt(np.diag(np.linalg.inv(marginal)))
    sigma_conditional = np.sqrt(np.diag(np.linalg.inv(conditional)))
    for i, name in enumerate(scan):
        print(
            f"  {name}: fiducial {center[i]:.6g}, "
            f"sigma {sigma_marginal[i]:.4g} (marg) vs {sigma_conditional[i]:.4g} (cond)"
        )

    # A conditional slice only probes the conditional width, which can be far
    # narrower than the marginal one (for WL the ratio reaches ~30 in the IA
    # sector). Sizing the grid on the marginal sigmas then wastes almost every
    # evaluation on a region where the sliced likelihood is already negligible,
    # so the scale is configurable.
    sigma_scale = sigma_marginal if args.scale == "marginal" else sigma_conditional
    print(f"  Grid sized on the {args.scale} sigmas, +-{args.width} of them per axis")

    x = focused_axis(center[0], sigma_scale[0], args.nx, args.width, args.focus)
    y = focused_axis(center[1], sigma_scale[1], args.ny, args.width, args.focus)

    # The demo's hand-picked boxes only cover the cosmological parameters; a grid
    # reaching outside them would not be comparable to the sampled posterior.
    for axis, name in ((x, scan[0]), (y, scan[1])):
        bounds = COSMO_PRIOR_RANGES.get(name)
        if bounds is not None and (axis[0] < bounds[0] or axis[-1] > bounds[1]):
            raise ValueError(
                f"Grid on {name} spans [{axis[0]:.4g}, {axis[-1]:.4g}], "
                f"outside the demo prior {bounds}. Reduce --width."
            )

    xx, yy = np.meshgrid(x, y)
    offsets = np.stack([xx - center[0], yy - center[1]], axis=-1)
    conditional_chi2 = np.einsum("...i,ij,...j->...", offsets, conditional, offsets)
    marginal_chi2 = np.einsum("...i,ij,...j->...", offsets, marginal, offsets)

    f0 = nll({scan[0]: center[0], scan[1]: center[1]})
    print(f"  Fiducial -log L: {f0:.6g}")
    print(f"  Grid: {args.nx} x {args.ny} = {args.nx * args.ny} evaluations")
    print(
        f"  Central spacing: {x[args.nx // 2 + 1] - x[args.nx // 2]:.4g} "
        f"and {y[args.ny // 2 + 1] - y[args.ny // 2]:.4g}"
    )

    settings = {
        "scanned": list(scan),
        "free_params": list(free_params),
        "fisher_parameters": names,
        "fiducial": fiducial,
        "nx": args.nx,
        "ny": args.ny,
        "scale": args.scale,
        "width": args.width,
        "focus": args.focus,
        "workers": args.workers,
        "nll_fiducial": f0,
    }
    (outdir / "grid_settings.json").write_text(json.dumps(settings, indent=2))

    tasks = [
        (iy, ix, {scan[0]: float(x[ix]), scan[1]: float(y[iy])})
        for iy in range(args.ny)
        for ix in range(args.nx)
    ]
    values = np.full((args.ny, args.nx), np.nan)

    def save() -> None:
        np.savez_compressed(
            outdir / "likelihood_grid.npz",
            x=x,
            y=y,
            nll=values,
            conditional=conditional,
            marginal=marginal,
            conditional_chi2=conditional_chi2,
            marginal_chi2=marginal_chi2,
            center=center,
            nll_fiducial=f0,
            scanned=np.array(scan),
        )

    start = time.time()
    global _NLL
    _NLL = nll

    if args.workers == 1:
        for done, (iy, ix, params) in enumerate(tasks, start=1):
            values[iy, ix] = nll(params)
            if done % args.nx == 0:
                elapsed = time.time() - start
                eta = elapsed / done * (len(tasks) - done)
                save()
                print(
                    f"  Row {done // args.nx}/{args.ny}: elapsed {elapsed:.0f}s, ETA {eta:.0f}s",
                    flush=True,
                )
    else:
        # fork lets the workers inherit the likelihood without pickling it.
        context = mp.get_context("fork")
        with context.Pool(processes=args.workers) as pool:
            for done, (iy, ix, value) in enumerate(
                pool.imap_unordered(_evaluate, tasks, chunksize=4), start=1
            ):
                values[iy, ix] = value
                if done % args.nx == 0:
                    elapsed = time.time() - start
                    eta = elapsed / done * (len(tasks) - done)
                    print(
                        f"  {done}/{len(tasks)} points: elapsed {elapsed:.0f}s, ETA {eta:.0f}s",
                        flush=True,
                    )
        save()

    save()
    print(f"  Grid finished in {time.time() - start:.1f}s")

    # Referenced to the fiducial rather than the sampled minimum, so that any
    # offset of the likelihood peak stays visible instead of being absorbed.
    delta_chi2 = 2.0 * (values - f0)
    iy, ix = np.unravel_index(np.argmin(values), values.shape)
    print(
        f"  Grid minimum at {scan[0]}={x[ix]:.6g}, {scan[1]}={y[iy]:.6g} "
        f"with delta chi2 = {delta_chi2[iy, ix]:.4g}"
    )
    print(
        f"  Offset from fiducial: {(x[ix] - center[0]) / sigma_marginal[0]:+.2f} sigma, "
        f"{(y[iy] - center[1]) / sigma_marginal[1]:+.2f} sigma (marginal units)"
    )

    levels = [2.30, 6.18]  # 68.3% and 95.4% for a 2D Gaussian
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.contour(xx, yy, delta_chi2, levels=levels, colors="tab:orange", linewidths=2.0)
    ax.contour(xx, yy, conditional_chi2, levels=levels, colors="tab:green", linestyles="dashed")
    ax.contour(xx, yy, marginal_chi2, levels=levels, colors="tab:blue", linestyles="dotted")
    ax.axvline(center[0], color="grey", lw=0.8, ls=":")
    ax.axhline(center[1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel(scan[0])
    ax.set_ylabel(scan[1])
    ax.set_title(f"WL likelihood vs Fisher: {scan[0]} - {scan[1]}")
    ax.legend(
        handles=[
            Line2D([], [], color="tab:orange", lw=2.0, label="Likelihood (others fixed)"),
            Line2D([], [], color="tab:green", ls="dashed", label="Fisher conditional"),
            Line2D([], [], color="tab:blue", ls="dotted", label="Fisher marginal"),
        ],
        loc="best",
    )
    fig.tight_layout()
    figure_path = outdir / f"grid_{scan[0]}_{scan[1]}.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"  Saved: {outdir / 'likelihood_grid.npz'}")
    print(f"  Saved: {figure_path}")
    print()
    print("  Note: these are likelihood-ratio contours on a conditional slice.")
    print("  The remaining parameters are fixed at their fiducial values, so this")
    print("  shows the shape of the surface, not a marginalised posterior.")


if __name__ == "__main__":
    main()
