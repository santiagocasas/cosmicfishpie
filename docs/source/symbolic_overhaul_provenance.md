# Symbolic overhaul: implementation provenance

> This page preserves the original design, decisions, implementation notes, and
> remaining validation work for the symbolic-backend overhaul. The status below is
> a project record, not a claim that every roadmap item or scientific comparison is
> complete.

Objective
- Integrate SYREN-NEW as the default symbolic backend in CosmicFishPie (cosmicfishpie-symbolic_overhaul branch), add CLASS+EuclidEmu2 as a composable nonlinear boost, and validate through CLASS-comparison forecasts (GCsp and WL+GCph Fisher matrices).
Important Details
- User decisions: total-matter first (cb requests rejected explicitly); validate through spectra→derivatives→Fisher forecasts; new symbolic is default, explicit legacy mode kept; wider_syren and baryonic corrections are later work.
- euclidemu2==1.4.3 is already declared as optional `emulators` extra and installed in this branch's .venv. Its native grid spans approximately 8.73e-3–9.41 h/Mpc; calls accept at most 100 redshifts.
- CLASS+EE2 uses P_nl = P_lin_CLASS × B_EE2 and the same total-matter boost for cb as an explicit approximation. The top-level NONLINEAR.model YAML selector defaults to class_native; it is never passed to CLASS.
- EE2 k uses h/Mpc (CLASS k in 1/Mpc divided by h). Below the native grid the boost is unity; above the grid the adapter raises ValueError. CLASS's padded redshift grid must fit EE2's z=0–10 range.
- class/syren_new_minimal_matched.yaml requires output: mPk,mTk; changebasis_class() has opt-in three_degenerate neutrino translation via PARAMETER_TRANSLATION.
- build_analysis_context is the spectrum-only configuration API; FisherMatrix is for derivatives/Fisher computations. Call likelihood.loglike(param_dict={}) by keyword.
- Do not execute notebooks; do not commit; do not kill the Nautilus sampler (if running).
- Keep structural checks after edits; additional backend experiments require an explicit request.
Work State
Completed
- SyrenNewProvider implemented in cosmicfishpie/cosmology/symbolic_new.py; wired into cosmology.py via input_type='symbolic' dispatch; uses Astropy backgrounds, physical-unit Pmm, total-matter only.
- SYREN-NEW GCsp and WL Fisher matrices produced: results/CosmicFish_v1.3.1_GCsp_symb_demo__GCsp_FM.txt, results/CosmicFish_v1.3.1_WL_symb_demo__WL_FM.txt.
- Matched CLASS and symbolic YAML profiles, three_degenerate CLASS neutrino scheme, unexecuted GCsp/Photo comparison notebooks, and aggregate derivative plotting are in place.
- Real 3-point GCsp derivatives run through derivatives.py engine (via temp script /tmp/opencode/run_matched_gcsp_derivatives.py) for all 4 configs (CLASS LCDM/w0wa, SYREN LCDM/w0wa); all derivatives finite; artifacts in /tmp/opencode/matched_gcsp_derivative_check/.
- scripts/run_syren_new_likelihoods.py created; fiducial loglike(param_dict={}) verified: GCsp=-0, WL+GCph≈1.4e-10; parameter responsiveness confirmed.
- CLASS+EE2 implemented in cosmicfishpie/cosmology/ee2.py and cosmology.py, with a new class/ee2_boost.yaml profile. CLASS linear spectra are multiplied by the emulator boost when nonlinear is enabled; native CLASS remains the default.
- Focused adapter and real CLASS+EE2 integration tests passed for both A_s and sigma8 normalization (3 passed); ruff, isort and Black checks passed on touched Python files.
Active
- Review the CLASS+EE2 integration in representative spectra/derivatives/forecasts as the broader comparison work proceeds; no notebook or sampler was executed during this implementation.
Blocked
- None for CLASS+EE2; the optional dependency is installed and the focused integration test passed.
Next Move
1. Select `class_config_yaml=cosmicfishpie/configs/default_boltzmann_yaml_files/class/ee2_boost.yaml` with `code=class` and `nonlinear=True` for a CLASS+EE2 comparison.
2. Compare spectra, derivatives and forecasts against the matched CLASS HMcode and SYREN-NEW configurations when that validation is requested.
Relevant Files
- cosmicfishpie/cosmology/ee2.py: CLASS→EE2 parameter mapping and boost grid adapter.
- cosmicfishpie/cosmology/cosmology.py: class_setparams() selector and class_results() multiplication.
- cosmicfishpie/configs/default_boltzmann_yaml_files/class/ee2_boost.yaml: opt-in CLASS+EE2 profile.
- tests/cosmology_test.py: unit and integration checks.
- docs/source/symbolic_overhaul_provenance.md: full design below this current status.






**The plan is to make SYREN-NEW the default symbolic backend, separate nonlinear modelling from the underlying cosmology calculation, and validate the result against CLASS-based references through representative forecasts.**

Your decisions are incorporated:
- **Total matter first**; unsupported CDM+baryon (`clustering`) requests will fail explicitly.
- **Validation through spectra, derivatives, observables, and Fisher forecasts.**
- **New symbolic by default**, with an explicit legacy option. Legacy needs basic operational coverage, not an exact-output preservation campaign.
- **Wider SYREN and baryonic corrections come last**, without a separate scientific-validation campaign.

The sections below describe the wider roadmap for `cosmicfishpie-symbolic_overhaul`; the current implementation status is tracked above.

## 1. Design: cosmology provider + nonlinear prescription

Keep the public cosmology interface—`Pmm`, `Hubble`, distances, growth, and so on—but separate its internal responsibilities:

```text
Cosmological parameters
        │
        ▼
Background + linear-spectrum provider
        │
        ▼
Nonlinear prescription
  ├─ native calculation
  ├─ direct nonlinear emulator
  └─ linear power × nonlinear boost
        │
        ▼
Optional baryonic correction
        │
        ▼
Existing CosmicFishPie observables and forecasts
```

This accommodates the requested combinations:

| Configuration | Linear power | Nonlinear power |
|---|---|---|
| New symbolic default | `linear_new` | `syren_new` |
| CLASS + EuclidEmu2 | CLASS | CLASS linear × EE2 boost |
| Legacy symbolic | Existing implementation | Existing legacy choices |
| Later: wider symbolic | `wider_syren.linear` | `wider_syren.halofit` |
| Later: baryonic symbolic | `linear_new` | SYREN-NEW × baryonic correction |
| Future: MG combination | MGCLASS | Compatible MG linear power × MG boost |

**Each boost must declare what it multiplies.** A nonlinear-to-linear boost and an MG-to-GR nonlinear ratio are different objects. Recording that distinction now enables future MG combinations without assuming every emulator can multiply every spectrum.

SYREN-NEW remains a **direct nonlinear-spectrum provider**: its implementation evaluates its own linear model internally. We should not pretend it accepts arbitrary CLASS linear power.

## 2. Replace the symbolic calculation completely

The main integration point is `cosmicfishpie/cosmology/cosmology.py`, especially `changebasis_symb`, `symbolic_setparams`, and `symbolic_results`.

### Parameter handling

Resolve a consistent parameter set:

\[
\{10^9A_s,\Omega_m,\Omega_b,h,n_s,\Sigma m_\nu,w_0,w_a\}.
\]

Changes include:

- Support flat ΛCDM and \(w_0w_a\)CDM within the emulator’s supported domain.
- Include neutrinos when converting between total matter and CDM densities.
- Use the new amplitude conversions, including their \(m_\nu,w_0,w_a\) dependence.
- Preserve a supplied \(A_s\); avoid the current unnecessary \(A_s\to\sigma_8\to A_s\) round trip.
- Define how conflicting amplitude inputs are handled.
- Reject unsupported cosmological freedom rather than silently ignoring it.

### Spectra and units

Evaluate the new linear spectrum **at each redshift**, rather than scaling a \(z=0\) spectrum with the old scale-independent Colossus growth.

Preserve CosmicFishPie’s physical-unit interface:

\[
k_{\rm symbolic}=k_{\rm CFP}/h,\qquad
P_{\rm CFP}=P_{\rm symbolic}/h^3.
\]

Internally, use explicit `(n_z, n_k)` grids and retain the existing interpolation interface.

### Background, growth, and normalization

My recommendation is:

- **Astropy backgrounds**, already a dependency, for \(H(z)\), distances, and density evolution, tabulated once per cosmology.
- Configure radiation and massive neutrinos explicitly; account for Astropy’s distinction between matter and neutrino densities.
- Derive growth from the complete new linear spectrum:
  \[
  D(k,z)=\sqrt{\frac{P_{\rm lin}(k,z)}{P_{\rm lin}(k,0)}},
  \qquad
  f(k,z)=-(1+z)\frac{\partial\ln D(k,z)}{\partial z}.
  \]
- Compute \(\sigma_8(z)\) from the linear spectrum, with controlled integration limits and normalization checks.

This keeps growth and normalization consistent with the emulator’s scale dependence. Its internal approximate growth factor alone is insufficient because the final spectrum includes additional redshift-dependent corrections.

### Colossus compatibility

The new path will stop creating Colossus cosmologies or using its background/growth routines.

One source detail matters: `linear_new.py` imports a function from `linear.py`, and that module imports Colossus at module level. Upstream also declares Colossus as a dependency. Therefore, **inactive but still installed** is the appropriate first implementation, matching your compatibility preference.

## 3. Add EuclidEmu2 as a reusable nonlinear boost

Implement a small adapter around:

```python
ee2.get_boost(parameters, redshifts)
```

Then construct:

\[
P_{\rm nl}(k,z)=B_{\rm EE2}(k,z)\,P_{\rm lin}(k,z).
\]

For CLASS+EE2:

- Use CosmicFishPie’s existing CLASS linear calculation.
- Avoid computing native CLASS nonlinear power unnecessarily.
- Keep CLASS backgrounds and linear growth.
- Obtain \(A_s\), densities, and neutrino parameters from the same resolved cosmology.
- Expose total-matter nonlinear results consistently; do not leave native CLASS cb nonlinear results attached to an EE2-labelled calculation.
- Import EE2 lazily and provide it as an optional dependency.

The installed EE2 wrapper warrants a little care: it extrapolates custom \(k\) inputs and has fixed-size native redshift storage. I recommend requesting its native \(k\) grid, handling interpolation and range policies in our adapter, and chunking redshifts within its capacity.

All provider choices must travel with the existing immutable run configuration, including metadata and any cache identity.

## 4. Set the numerical domain before running forecasts

This is the main scientific constraint to resolve early.

The SYREN-NEW notebook exercises:

- \(0\le z\le3\);
- \(0.009\le k\le9\,h\,{\rm Mpc}^{-1}\);
- bounded cosmological parameters, including \(0\le\Sigma m_\nu\le0.15\) eV.

The installed EE2 boost covers approximately \(0.00873\)–\(9.41\,h\,{\rm Mpc}^{-1}\), while some existing photometric configurations request substantially higher \(k\).

The implementation should therefore:

1. Distinguish documented emulator validity from the notebook’s sampled test range.
2. Check actual requested scales—including growth defaults, smoothing grids, and \(\sigma_8\) integration.
3. Define explicit low-/high-\(k\) behaviour.
4. Validate the **whole derivative stencil**, not just the fiducial cosmology.
5. Use identical physical scale cuts for both sides of each forecast comparison.

For initial forecast acceptance, I recommend a clearly labelled common-domain survey configuration. Full survey settings extending outside that domain require a separately justified extension policy; spline extrapolation must not silently supply it.

## 5. CLASS-only validation, in successive stages

### Stage A — Reproduce the notebook reference

Start with a fiducial and a small deterministic set of cosmologies, then extend coverage.

The CLASS settings need a dedicated profile: the notebook uses three degenerate massive neutrinos and a density convention that differs from CosmicFishPie’s current CLASS mapping. Match those conventions deliberately before attributing residuals to the emulator.

### Stage B — Validate cosmology outputs

Compare:

- linear \(P(k,z)\);
- \(H(z)\), comoving and angular-diameter distances;
- \(\sigma_8(z)\);
- \(D(k,z)\) and \(f(k,z)\).

Separate raw emulator discrepancies from interpolation, integration, and parameter-conversion errors.

### Stage C — Validate nonlinear power

Use the notebook’s reference:

\[
\boxed{P_{\rm nl}^{\rm reference}
=P_{\rm lin}^{\rm CLASS}\,B_{\rm EE2}}
\]

Native CLASS HMcode/Halofit can be a separately labelled comparison if useful, but it should not be the sole pass/fail reference for an emulator targeting EE2.

### Stage D — Validate derivatives

Check parameter responses and finite-difference stability, especially for \(m_\nu,w_0,w_a\) and amplitude normalization.

Use absolute or scaled residuals where derivatives approach zero. Agreement in spectra does not guarantee agreement in their derivatives.

### Stage E — Validate representative forecasts

Run matched, total-matter cases for:

- photometric observables and Fisher matrices;
- spectroscopic observables and Fisher matrices.

Reuse `scripts/run_fisher_compare_backends.py` and the current reporting pipeline. Compare observable derivatives before interpreting differences in marginalized errors.

Define acceptance thresholds separately for each stage before the formal campaign. Any slightly different CLASS settings and survey cuts should be recorded alongside the results.

**The existing CAMB–CLASS validation remains supporting evidence.** The new campaign establishes consistency with the chosen CLASS-based reference; it does not automatically establish agreement for every earlier nonlinear prescription or parameter setting.

## 6. Dashboard, tests, and dependency integration

Extend the current validation machinery to record:

- background and linear provider;
- nonlinear prescription and optional correction;
- tracer, amplitude basis, neutrino conventions, and domain policy;
- package revisions and resolved numerical settings;
- spectrum, derivative, observable, and Fisher metrics.

Relevant files include:

- `cosmicfishpie/configs/config.py` and `configs/context.py`;
- `configs/default_boltzmann_yaml_files/symbolic/`;
- `scripts/run_fisher_compare_backends.py`;
- `scripts/render_validation_dashboard.py`;
- `scripts/validation_configs/`;
- focused cosmology and runtime-context tests.

Software tests should cover units, shape/order handling, nonlinear toggles, unsupported cb requests, amplitude conversion, range handling, and configuration isolation.

A useful finding: the current project pin already matches the local upstream commit containing these new modules. We should verify packaging and align dependency declarations rather than assume an upstream version bump is necessary.

## 7. Final optional additions

### Wider SYREN

Add it as an explicit **ΛCDM-only** option, with its own parameter/domain metadata, linear spectrum, and Halofit calculation. Its wider priors do not extend SYREN-NEW’s massive-neutrino or dynamical-dark-energy support.

### Baryonic corrections

Add a final correction stage:

\[
P_{\rm nl,baryonic}=S_{\rm baryon}\,P_{\rm nl,DMO}.
\]

Expose the hydro-model choice and feedback parameters. Keep the supplied scatter estimate separate from the deterministic spectrum; incorporating it into forecast covariance would be a further feature.

As requested, these final options receive integration checks, not another scientific-validation campaign.

---

## Implementation order

1. **Parameter, unit, capability, and domain contracts.**
2. **New symbolic spectra plus Colossus-free background/growth calculation.**
3. **Composable nonlinear interface and CLASS+EE2 adapter.**
4. **CLASS validation: cosmology outputs → derivatives → forecasts.**
5. **Dashboard integration and default/legacy documentation.**
6. **Wider SYREN, then baryonic corrections.**

The most important early milestone is a complete new symbolic cosmology object—not merely matching \(P(k)\)—with consistent backgrounds, growth, normalization, and explicit numerical coverage. Once that is established, EE2 and the later corrections fit naturally into the same design.
