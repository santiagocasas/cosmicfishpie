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
| F3 | high | `AP_effect=False` branch has wrong array rank and a wrong redshift-error `k_par` (legacy always uses `k mu / q_par`) | open |
| F4 | medium | `kh_rescaling_bug` / `kh_rescaling_beforespecerr_bug` read but not implemented | open |
| F5 | medium | Claim "legacy spline extrapolates" was wrong: `RectBivariateSpline` clamps at the boundary. Direct Halofit evaluation outside the table is a deliberate difference, not a reproduction | open |
| F6 | medium | ARA trace: duplicate top-level YAML keys (`tree:`, `observations:`); observation O19 states the wrong spline claim | open |
| F7 | medium | No automated tests for the jaxcosmo path | open |
| F8 | medium | Halofit Newton solve has no convergence diagnostic; module-level constants built with JAX before x64 is enabled | open |
| F9 | low | Documentation drift (provider docstring says photometric only; photo docstring describes an interpolation no longer used) | open |
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

## 4. Limits of the evidence

* The parity numbers in the CHANGELOG and PR are for the Euclid pessimistic setups with fiducial
  `10^9As=2.1, Omegam=0.32, Omegab=0.05, h=0.67, ns=0.96`, massless neutrinos.
* "max |dF|/max|F|" hides larger relative differences in small matrix entries. Use the
  entry-wise and marginalised-sigma comparisons as well.
* The large cross-backend Fisher campaign is deliberately left for a separate workstation run.

## 5. Upstreaming the smooth Halofit to jax-cosmo

`jax_cosmo.power._halofit_parameters` finds the nonlinear scale with a linear `interp` on a
fixed 256-node `ln R` grid, so d P/d theta is discontinuous between nodes. `jaxcosmo_halofit.py`
replaces that with a fixed-count Newton iteration. Planned steps: validate across the parameter
domain here (F8), then prepare a minimal patch plus tests against the local jax_cosmo clone.
No pull request is opened without maintainer (user) approval.
