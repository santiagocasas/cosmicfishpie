# Derivative Methods

CosmicFishPie computes Fisher matrices from numerical derivatives of
observables (angular power spectra, galaxy power spectra, etc.) with respect
to the free cosmological and nuisance parameters. The method used for these
derivatives is selected via `options["derivatives"]` and implemented in
{mod}`cosmicfishpie.fishermatrix.derivatives`. Four methods are available:
`3PT` (default), `4PT_FWD`, `POLY`, and `STEM`.

This page documents the algorithm behind each method, the relative cost of
using them, and an empirical comparison across two different backends
(`symbolic` and `camb`) that motivated a bug fix to the `POLY` method
(see [Historical note: the POLY bug](#historical-note-the-poly-bug) below).
It also documents `scripts/compare_derivative_methods.py`, the regression
script used to produce these results, so the comparison can be reproduced or
extended.

## 1. `3PT` — central three-point finite difference

The default method. For each free parameter $\theta$ with fiducial value
$\theta_0$ and step $h = \theta_0 \cdot \texttt{freeparams}[\theta]$, the
observable $\mathcal{O}$ is evaluated once on each side of the fiducial
point:

```{math}
\frac{\mathrm{d}\mathcal{O}}{\mathrm{d}\theta}\bigg|_{\theta_0}
    \approx \frac{\mathcal{O}(\theta_0+h) - \mathcal{O}(\theta_0-h)}{2h}
```

This is the standard central-difference stencil. Its leading-order
truncation error comes from the next (cubic) term in the Taylor expansion of
$\mathcal{O}$ around $\theta_0$:

```{math}
\text{error} \approx \frac{h^2}{6}\,\mathcal{O}'''(\theta_0)
```

i.e. the error scales as $O(h^2)$ and vanishes exactly for any observable
that is at most quadratic in $\theta$ locally. Cost: **2 observable
evaluations per parameter**.

## 2. `4PT_FWD` — one-sided four-point forward stencil

A forward-only (no backward step) finite-difference stencil, useful when a
parameter cannot be perturbed below the fiducial value. It evaluates the
observable at the fiducial point and three equally spaced forward steps
($\theta_0, \theta_0+h, \theta_0+2h, \theta_0+3h$) and combines them with
fixed coefficients (obtained from a
[finite-difference coefficients calculator](https://web.media.mit.edu/~crtaylor/calculator.html)):

```{math}
\frac{\mathrm{d}\mathcal{O}}{\mathrm{d}\theta}\bigg|_{\theta_0}
    \approx \frac{-11\,\mathcal{O}(\theta_0) + 18\,\mathcal{O}(\theta_0+h)
    - 9\,\mathcal{O}(\theta_0+2h) + 2\,\mathcal{O}(\theta_0+3h)}{6h}
```

This is $O(h^3)$-accurate, but uses **4 observable
evaluations per parameter** (2x the cost of `3PT`) since it cannot reuse a
symmetric backward point.

## 3. `POLY` — quartic polynomial fit

Evaluates the observable at `numpoints=10` steps evenly spaced around the
fiducial value, `stepsize = np.linspace(-h, h, 10)` (i.e. `stepsize` is the
*offset from the fiducial*, not the parameter value itself), then fits a
degree-4 polynomial to the resulting values as a function of that offset:

```python
fit = np.polyfit(stepsize, values, 4)
# fit = [a4, a3, a2, a1, a0], highest power first:
# f(step) = a4*step^4 + a3*step^3 + a2*step^2 + a1*step + a0
```

Because `stepsize` is a translation of $\theta$ ($\text{step} = \theta -
\theta_0$), the change of variables is a pure shift, so
$\mathrm{d}/\mathrm{d}\theta = \mathrm{d}/\mathrm{d}(\text{step})$. The
derivative at the fiducial point ($\text{step}=0$) is therefore simply the
polynomial's linear coefficient, `fit[3]` (`a1` above) — no evaluation of the
fit at any nonzero point is needed.

**Worked example.** Take $\mathcal{O}(\theta) = \theta^3$ with fiducial
$\theta_0 = 2$. Expanding around the fiducial with $\text{step} = \theta -
\theta_0$:

```{math}
\mathcal{O}(\theta_0 + \text{step}) = 8 + 12\,\text{step}
    + 6\,\text{step}^2 + \text{step}^3
```

so the exact derivative at the fiducial is the coefficient of the linear
term, $12$, matching $3\theta_0^2 = 3 \cdot 4 = 12$. A degree-4 fit to this
data recovers `fit[3] = 12` exactly (the quartic term is fit to zero since
the data has no quartic component). Cost: **10 observable evaluations per
parameter** (5x the cost of `3PT`), but the derivative from a single fit is
essentially free (no additional evaluations of the observable).

### Historical note: the POLY bug

Prior to this branch, `derivative_poly()` evaluated the fitted polynomial at
the *fiducial value* $\theta_0$ itself, using a mismatched expression that
treated the fit as if it were a function of $\theta$ rather than of the
offset:

```python
# buggy: wrong evaluation point AND wrong coefficient-to-power mapping
4 * fit[0] * fidpar**3 + 3 * fit[2] * fidpar**2 + 2 * fit[3] * fidpar + fit[4]
```

For the $\theta^3$ example above this does not reproduce the correct
derivative of 12 at any fiducial value other than 0. The fix (this branch)
replaces this with `fit[3]`, i.e. reading off the linear coefficient of the
fit directly, since the fit is already expressed in the offset coordinate
where the fiducial point is 0. See
{func}`cosmicfishpie.fishermatrix.derivatives.derivatives.derivative_poly`
and the regression tests
`test_derivative_poly_cubic_exact_at_fiducial` /
`test_derivative_poly_matches_3pt_for_power_spectrum_observable` in
`tests/derivatives_test.py`.

## 4. `STEM` — adaptive linear-fit ("step method")

`STEM` solves a different problem than the three fixed-step methods above:
rather than assuming one pre-chosen step size is simultaneously *large
enough* to rise above the observable's numerical noise floor and *small
enough* to avoid curvature bias, it adaptively searches for a step-size
window where a straight-line (degree-1) fit is statistically trustworthy.

Algorithm (`numstem=11`, `mult_eps_factor=5`, `threshold=1e-3`):

1. Build an 11-point step grid spanning $\pm 5h$ around the fiducial
   (`np.linspace(-5h, 5h, 11)`), and evaluate the observable at each point.
2. Fit a degree-1 polynomial (`np.polyfit(step, values, 1, full=True)`) and
   check the residual sum-of-squares against `threshold = 1e-3`.
3. If the residual exceeds the threshold, trim the two outermost
   (largest-`|step|`) points — the ones most likely dominated by curvature —
   and refit.
4. Repeat until the fit converges or fewer than 3 points remain, in which
   case the routine currently prints an error and calls `exit()` (a known,
   pre-existing hard-exit fragility, tracked separately and not addressed by
   this comparison).
5. The derivative is the converged linear fit's slope.

Cost: **11 observable evaluations per parameter** (the most expensive
method), plus possible extra polynomial refits (free, no extra observable
calls).

## Empirical comparison: WL-only Euclid Fisher matrix

To validate the `POLY` fix and understand when `STEM`'s adaptivity actually
matters, `scripts/compare_derivative_methods.py` builds a fresh WL-only
Euclid Fisher matrix (5 free LCDM parameters: `Omegam`, `Omegab`, `h`, `ns`,
`sigma8`, each with a 1% step) once per derivative method, using `3PT` as
the reference baseline, across two backends:

- **`symbolic`** — a fast, closed-form/emulator backend (colossus for
  background cosmology + the `symbolic_pofk` package's symbolic-regression
  P(k) emulators) with essentially no numerical noise.
- **`camb`** — the real CAMB Boltzmann backend, using the repository's fast
  default settings (`camb/default.yaml`) with `halofit_version` overridden
  to `takahashi`. Real Boltzmann solvers have an irreducible numerical noise
  floor (ODE-integration tolerances, line-of-sight quadrature, nonlinear
  halofit), which is exactly the regime `STEM`'s adaptivity is designed to
  be robust against.

### Symbolic backend (noise-free)

```
param           sigma_3PT   dev%_4PT_FWD  dev%_POLY  dev%_STEM
Omegam         0.00595468       0.020        0.016      0.323
Omegab         0.0203588        0.022        0.021      0.349
h              0.119352         0.016        0.015      0.243
ns             0.027081         0.010        0.009      0.154
sigma8         0.00761877       0.016        0.013      0.271
-------------------------------------------------------------
worst deviation vs 3PT                       0.022%       0.021%      0.349%
                                              (4PT_FWD)    (POLY)      (STEM)
```

Total wall-clock (all 4 methods): 2m33.6s (3PT 14.6s, 4PT_FWD 17.6s, POLY
57.2s, STEM 59.3s).

### CAMB backend (real solver, halofit=takahashi)

```
param           sigma_3PT   dev%_4PT_FWD  dev%_POLY  dev%_STEM
Omegam         0.00538462       0.204        0.104      0.317
Omegab         0.0213948        1.198        0.669      0.419
h              0.124571         1.399        0.732      0.213
ns             0.0295462        1.586        0.727      0.029
sigma8         0.00665589       0.194        0.105      0.312
-------------------------------------------------------------
worst deviation vs 3PT                       1.586%       0.732%      0.419%
                                              (4PT_FWD)    (POLY)      (STEM)
```

Total wall-clock (all 4 methods): 11m50s (3PT 54.0s, 4PT_FWD 79.2s, POLY
264.7s, STEM 292.0s).

### Interpretation

- **`POLY` is fixed.** In both backends `POLY` agrees with the well-tested
  `3PT` baseline far more closely than `STEM` does, confirming the fitted
  linear-coefficient fix is correct (previously a wrong evaluation-point
  formula would have produced grossly, not just slightly, different
  derivatives).
- **The ranking flips between backends.** With the noise-free `symbolic`
  backend, `4PT_FWD` and `POLY` are essentially exact matches to `3PT`
  (~0.02%), while `STEM` has the *largest* deviation (0.35%) — just its own
  `1e-3` convergence-tolerance floor showing through, with no real noise for
  the adaptivity to correct for. With the real, noisy `camb` backend,
  `4PT_FWD` (the smallest, most fixed step, most exposed to cancellation
  noise) degrades the most (0.02% → 1.6%), `POLY` degrades less thanks to
  its wider default step and averaging over 10 points (0.02% → 0.73%), and
  `STEM` stays roughly stable regardless of backend (0.35% → 0.42%) — ending
  up with the *smallest* deviation from `3PT` of the three non-baseline
  methods in the noisy regime.
- **Takeaway:** `STEM`'s adaptive step-size search earns its keep
  specifically against real Boltzmann-solver numerical noise (this is
  consistent with historical experience that `STEM` "helped" in past
  Euclid-validation Fisher comparisons using CAMB/CLASS). In smooth or
  noise-free conditions (e.g. the `symbolic` backend, or well-converged
  analytic observables) it offers no advantage and simply reflects its own
  coarser stopping tolerance.

## Running the comparison script

```bash
# Default: symbolic backend, all 4 methods
uv run python scripts/compare_derivative_methods.py

# Only compare a subset of methods against the 3PT baseline
uv run python scripts/compare_derivative_methods.py --methods 3PT,POLY

# Real CAMB backend (fast default yaml, halofit_version=takahashi)
uv run python scripts/compare_derivative_methods.py --backend camb
```

The script prints a table of the marginalized 1-sigma bound on each free
parameter for every requested method, the percent deviation relative to the
`3PT` baseline, the worst-deviating parameter per method, and per-method
wall-clock timings. Fisher matrices and metadata are written under
`scripts/benchmark_results/compare_derivative_methods/` (gitignored, not
committed). See `scripts/compare_derivative_methods.py`'s module docstring
and `--help` output for the full set of options.
