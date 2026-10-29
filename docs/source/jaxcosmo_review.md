# `jaxcosmo` backend: review log

Living record of the review of the `jaxcosmo-autodiff` branch (draft PR #231): what was built,
what was found, what was fixed, and what is still open. Each finding has a status and the commit
that closed it.

Status legend: **open**, **fixed** (with commit), **accepted** (documented limitation).

## 1. What the branch contains

| layer | module | role |
|---|---|---|
| backend registration | `configs/config.py`, `configs/context.py`, `cosmology/cosmology.py`, `fishermatrix/cosmicfish.py` | `options["code"] = "jaxcosmo"`, yaml in `default_boltzmann_yaml_files/jaxcosmo/` |
| theory (traceable) | `cosmology/jaxcosmo_theory.py`, `jaxcosmo_halofit.py`, `jaxcosmo_amplitude.py`, `jax_theory.py` | distances, growth, Eisenstein-Hu + Halofit, As -> sigma8 fit, backend-neutral `JaxTheory` protocol |
| theory (legacy adapter) | `cosmology/jaxcosmo_provider.py` | NumPy/SciPy splines so the *existing* observables can use the backend with finite differences |
| observables (traceable) | `LSSsurvey/jaxcosmo_photo.py`, `jaxcosmo_spectro.py` | JAX ports of `ComputeCls` (GCph+WL) and `ComputeGalSpectro.lnpobs_gg` (GCsp) |
| derivatives | `fishermatrix/jaxcosmo_derivatives.py` | `JaxAutodiffProvider` (`jacfwd`) and `JaxFiniteDifferenceProvider` (legacy FD engine on the same JAX observable) |
| evidence | `scripts/validation/compare_jax_{photo,spectro}_fishers.py` | three-way parity: legacy-FD / JAX-FD / JAX-AD |

Design rationale: AD vs JAX-FD isolates the derivative method; JAX-FD vs legacy-FD isolates the
observable port.

Scope: flat LCDM/w0waCDM, massless neutrinos, total-matter tracer, photometric GCph+WL and
spectroscopic GCsp (default modelling path only). No CMB, no clustering (CB) tracer.

## 2. Findings

| # | severity | finding | status |
|---|---|---|---|
| F1 | high | Savitzky-Golay operator applied transposed in `jaxcosmo_spectro._nowiggle_power`; differs from `scipy.signal.savgol_filter` at the edges | fixed (see 3.1) |
| F2 | high | Derivative-provider cache keyed on the fiducial dictionary only; reuse across configurations returns stale observable | fixed (see 3.2) |
| F3 | high | `AP_effect=False` branch has wrong array rank and a wrong redshift-error `k_par` (legacy always uses `k mu / q_par`) | fixed (see 3.3) |
| F4 | medium | `kh_rescaling_bug` / `kh_rescaling_beforespecerr_bug` read but not implemented | fixed (see 3.3) |
| F5 | medium | Claim "legacy spline extrapolates" was wrong: `RectBivariateSpline` clamps at the boundary. Direct Halofit evaluation outside the table is a deliberate difference, not a reproduction | fixed: documented + `legacy_pk_clamp` switch (see 3.3) |
| F6 | medium | ARA trace: duplicate top-level YAML keys (`tree:`, `observations:`); observation O19 states the wrong spline claim | fixed (see 3.4; trace lives in `cfp-dev`, not in this repo) |
| F7 | medium | No automated tests for the jaxcosmo path | partly fixed (31 tests added; 474 pass in the full suite): theory, Halofit, amplitude, GCsp options, providers (3.2-3.5); photometric observable and Nautilus still untested |
| F8 | medium | Halofit Newton solve has no convergence diagnostic; module-level constants built with JAX before x64 is enabled | fixed (see 3.5) |
| F9 | low | Documentation drift (provider docstring says photometric only; photo docstring describes an interpolation no longer used) | fixed (see 3.6) |
| F10 | info | Parity evidence covers one LCDM massless-nu Euclid setup at a 0.3% step; max differences are normalised by the largest matrix element | accepted, see section 4 |

## 3. Fix log

### 3.1 F1: Savitzky-Golay orientation

The operator was built row-wise (`A^T`) and then applied as `A^T^T`, i.e. with the wrong
orientation. The `mode='interp'` edge handling is not symmetric, so the two differ only at the
boundary of the 800-point smoothing grid. Now `A = savgol_filter(eye(n), axis=0)` and
`tests/jaxcosmo_spectro_test.py` checks `A @ y` against SciPy (interior and edges).

Effect, same Euclid spectroscopic setup, 0.3% step (before -> after):

* fiducial `ln P` max |dlnP|/max|lnP| vs legacy: 7.9e-5 -> **2.2e-6**
* Fisher max |dF|/max|F|, JAX-FD vs legacy: 1.5e-6 -> 1.5e-6 (unchanged: the edge points carry
  little weight)

New observation: with the corrected smoothing, one point stands out in the `Omegam` derivative
of every redshift bin: the first wavenumber at `mu = 1`, where `k_ap = k / q_par` sits exactly
on the lower edge of the tabulated power spectrum (`kmin_pk * h`). Legacy clamps the spline
there, JAX evaluates directly. See F5.

### 3.2 F2: configuration-aware observable cache

`_JaxObservableProvider` now keys its cached observable on a SHA-256 fingerprint of the
fiducial point, `settings`, `specs`, `obs`, the requested probes and the backend parameters.
Object identity is deliberately not used: the legacy `config` module is mutated in place, so
the same object can describe two different analyses. The fingerprint is dropped when the
provider is pickled, as before. `tests/jaxcosmo_derivatives_test.py` covers reuse, rebuild on
each kind of change (including in-place mutation), wrong `code`, and pickling.

### 3.3 F3 + F4 + F5: option branches and the table-edge behaviour

`JaxSpectroObservable._lnp_from_values` now follows the order of operations of
`ComputeGalSpectro.observed_Pgg`:

* the optional unphysical h-rescaling of `k` (`kh_rescaling_bug`) is applied either before
  (`kh_rescaling_beforespecerr_bug`) or after the spectroscopic-redshift damping;
* the damping always uses `k mu / q_par` (legacy `spec_err_z`), also with `AP_effect=False`;
* everything is broadcast to `(nz, nmu, nk)`, which also removes the rank error of the AP-off
  branch for multi-bin surveys.

**F5 correction.** The earlier statement "the legacy spline extrapolates" was wrong:
`RectBivariateSpline` returns the edge value outside its table (checked directly), while
`nonwiggle_pow` (a `k=1` `InterpolatedUnivariateSpline`) does extrapolate linearly. The JAX
observable evaluates the linear power directly (exact and smooth) by default. A new backend
option `NUMERICS.legacy_pk_clamp` clamps `k` to `[kmin_pk, kmax_pk]` (h/Mpc) and reproduces the
legacy behaviour for parity studies; it makes the derivative discontinuous where the fiducial
grid touches the table edge (first `k` at `mu = 1`).

Evidence (`tests/jaxcosmo_spectro_parity_test.py`, perturbed cosmology
`Omegam +6%, h +4%, ns +2%, As +5%`, absolute `|dlnP|`, legacy vs JAX):

| case | max abs diff |
|---|---|
| default | 2.5e-5 |
| AP off, **without** clamp | **3.8e-2** (grid below `kmin_pk h` once `h` grows) |
| AP off, with clamp | 2.5e-5 |
| h-bug before / after damping | 2.5e-5 (both shift `ln P` by 0.106, agreeing to 1e-7) |
| linear | 3.0e-5 |

So the AP-off "failure" is entirely the table-edge clamp, not a modelling error. The same effect
explains the single-point `Omegam` derivative mismatch noted in 3.1.

### 3.4 F6: ARA trace

`ara/trace/exploration_tree.yaml` and `ara/staging/observations.yaml` started with a repeated
top-level key (`tree:` / `observations:`). A permissive loader accepts that and silently keeps
the last one, which is why "all YAML validated" was misleading. The repeated line was removed
(parsed content verified identical before and after) and a duplicate-key-detecting loader over
all of `ara/` now reports none. Observation O19 keeps its original text and gets a dated
`correction:` field pointing to F5 instead of being rewritten.

### 3.5 F8 (+ part of F7): Halofit convergence and tests

Measured over 24 random cosmologies from the validation box (`As 1.7-2.5, Omegam 0.24-0.40,
Omegab 0.04-0.06, h 0.61-0.73, ns 0.92-1.0, w0 -1.3..-0.7, wa -0.7..0.5`) at
`z = 0, 0.5, 1.5, 3`:

* worst Newton residual `|growth^2 sigma^2(R_nl) - 1|` = **4.4e-16**, and the solution never
  touches the bracket `[-20, 10]` in `ln R`;
* maximum relative difference to `jax_cosmo.power.halofit` over `k = 1e-3..9` h/Mpc: **0.59%**
  (0.05% at the fiducial point). This is jax-cosmo's interpolation error in its own root, not
  ours: the 256-interval Simpson integral reproduces `k_nl` of a 4096-interval reference to
  1.7e-6 (z=0) and 3.1e-5 (z=1);
* `nonlinear_scale_residual()` is a new public diagnostic (the traced code cannot branch on a
  failed solve); the import-time `ln k` limits now use `math.log`, so they no longer depend on
  whether x64 was enabled before the import.

`tests/jaxcosmo_theory_test.py` (11 tests, ~55 s) covers convergence at the box corners,
agreement with jax-cosmo, AD-vs-central-difference at two step sizes (the error must shrink
quadratically), the amplitude fit against `symbolic_pofk` (values, inverse, gradient) and the
unit/normalisation conventions of `compute_tables` (distances checked against astropy).

### 3.6 F9 + lint

Provider and photo docstrings now describe what the code does. `black`, `isort` and `ruff` are
clean on every new file (one unused variable removed). `mypy` still reports
`config.boltzmann_jaxcosmopars` as a missing module attribute: this is the same dynamic
`cfg`-module pattern the other backends use (`boltzmann_symbolicpars`, ...) and the project has
~280 such pre-existing findings, so it was not "fixed" here.

### 3.7 Final parity after all fixes (commit `bd77f26` + script flag)

Photometric (GCph+WL, 1% step): fiducial `C_ell` max |dC|/max|C| 2.1e-5; Fisher max
|dF|/max|F|: JAX-FD vs legacy 1.2e-3, JAX-AD vs legacy 1.2e-3, **AD vs JAX-FD 1.8e-5**.

Spectroscopic (GCsp, 0.3% step):

| quantity | default (direct) | `--legacy-pk-clamp` |
|---|---|---|
| fiducial `ln P`, max rel | 2.2e-6 | 2.2e-6 |
| `d ln P/d Omegam`, JAX-FD vs legacy-FD, max / max-significant | 1.1e-1 / 2.4e-1 | **5.9e-5** / 1.7e-3 |
| `d ln P/d Omegam`, AD vs JAX-FD | 5.5e-3 | 1.0e-1 (kink at the clamp) |
| Fisher max \|dF\|/max\|F\|, JAX-FD vs legacy | 1.5e-6 | 1.5e-6 |
| Fisher, JAX-AD vs legacy | 1.4e-5 | 1.4e-5 |

Reading: the one grid point where the derivative differs from legacy (first `k`, `mu = 1`) is
explained completely by the legacy clamp: switching the clamp on removes the gap to the legacy
derivative (0.107 -> 5.9e-5) and moves the discontinuity into AD-vs-FD instead, exactly as a
kink predicts. The Fisher matrix does not care (one point in ~17k per bin). The default is the
direct evaluation, which is the smooth, physical choice.

### 3.8 Improvement: implicit-function Newton step (speed, same numbers)

The Newton iteration was unrolled 40 times, so forward-mode AD carried tangents through every
iteration. It now runs as a `lax.fori_loop` on gradient-free copies of the inputs, followed by
one differentiable Newton step at the converged root, which gives exactly the implicit-function
derivative `dR/dp = -(d sigma^2/dp)/(d sigma^2/dR)`.

* values: max rel difference to the unrolled version 1.6e-15; Jacobian 4.2e-14;
  AD vs central FD 1e-9;
* cost of the Halofit Jacobian (6 redshifts, 4 wavenumbers, 3 parameters, incl. compile):
  24.5 s -> 8.8 s; end-to-end photometric `jax-ad` Fisher: 37.6 s -> 18.3 s, with identical
  sigmas to all printed digits;
* the 11 theory tests pass unchanged.

## 4. Limits of the evidence

* The parity numbers below are for the Euclid pessimistic setups with fiducial
  `10^9As=2.1, Omegam=0.32, Omegab=0.05, h=0.67, ns=0.96`, massless neutrinos, LCDM, five
  cosmological parameters (+ four `lnbg_i` for GCsp).
* "max |dF|/max|F|" hides larger relative differences in small matrix entries. Use the
  entry-wise and marginalised-sigma comparisons as well.
* The large cross-backend Fisher campaign is deliberately left for a separate workstation run.

## 5. Upstreaming the smooth Halofit to jax-cosmo

`jax_cosmo.power._halofit_parameters` finds the nonlinear scale with a linear `interp` on a
fixed 256-node `ln R` grid (the code itself carries a `TODO: implement a better root finding
algorithm`), so d P/d theta is discontinuous between nodes.

Status: a patch is **prepared and committed locally, not pushed, no PR opened**, on branch
`smooth-halofit-nonlinear-scale` of `~/CosmoProjects/differentiablestuff/jax_cosmo`
(commit `d193ebf`, parent `c4f249f`). It changes only `jax_cosmo/power.py` (and drops the now
unused `interp` import) and adds `tests/test_halofit_smooth.py`.

Evidence, using the same `pyccl`-enabled environment their suite needs:

* their full suite (including the CCL comparisons): **25 passed** on the patched tree;
* the new derivative test **fails on master and passes with the patch**. 12 redshifts,
  `d P_halofit / d Omega_c`, error relative to the derivative's scale:

  | step | master | patch |
  |---|---|---|
  | 1% | 2.4e-1 (`k_nl` alone: 3.0e-2) | 1.0e-3 (`k_nl`: 2.7e-4) |
  | 0.1% | 1.6e-5 | 1.0e-5 |
  | 0.01% | 1.6e-7 | 1.0e-7 |

  Both converge quadratically to the same derivative, so master's problem is specific to the
  finite steps used in Fisher forecasts, where the root hops between nodes;
* the new test also checks `growth^2 sigma^2(R_nl) = 1` against a 40001-point integral
  (rtol 1e-4), which master satisfies too: the issue is smoothness, not the root's accuracy.

Open points for whoever proposes it: the project may prefer `custom_jvp` to the
stop-gradient-plus-one-step form; the Newton bracket `ln R in [-20, 10]` is wider than the old
`[ln 1e-4, ln 10]`; and a short note on behaviour change (values move by up to 0.6% over the
CFP validation box, which is jax-cosmo's own interpolation error) belongs in the PR text.
CosmicFishPie keeps its vendored `jaxcosmo_halofit.py` until a jax-cosmo release contains the fix.
