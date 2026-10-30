# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Document that the roughly 10% difference in the marginalized `10^9As` error between
  SYREN-NEW and JAXcosmo is a matter-power-model difference, not an implementation or
  differentiation artefact: unmarginalized errors agree to 2.6%, the gap appears through
  galaxy-bias marginalization, and putting the SYREN-NEW power spectrum inside the JAX
  observable reproduces the SYREN result to 0.08% (`scripts/validation/compare_theory_amplitude_response.py`).

- Document the JAXcosmo backend, AD/FD providers and configuration in the user guide,
  with a downloadable photometric comparison notebook against SYREN-NEW. Its Fisher
  comparison keeps `ns` fixed while retaining the spectral-index response diagnostic.

- Add the optional, differentiable `jaxcosmo` backend (`options["code"] = "jaxcosmo"`,
  install with `pip install 'cosmicfishpie[jaxcosmo]'`): `jax-cosmo` background,
  Eisenstein-Hu + Halofit total-matter power (flat, massless neutrinos), and a JAX port
  of the `symbolic_pofk` `10^9As -> sigma8` fit. A backend-neutral differentiable theory
  interface (`cosmology/jax_theory.py`) lets JAX observables stay independent of the
  selected backend. JAX-native photometric (`GCph`+`WL`, `LSSsurvey/jaxcosmo_photo.py`)
  and spectroscopic (`GCsp`, `LSSsurvey/jaxcosmo_spectro.py`) observables reproduce the
  standard Limber and P(k, mu) calculations, with `fishermatrix.jaxcosmo_derivatives`
  providers: `JaxAutodiffProvider` gives exact forward-mode derivatives and
  `JaxFiniteDifferenceProvider` applies the standard finite-difference engine to the same
  observable, so autodiff and finite differences can be validated against each other and
  against the legacy pipeline (`scripts/validation/compare_jax_*_fishers.py`).
  Both are wired through `FisherMatrix(derivative_provider=...)`. CMB, clustering/CB
  tracers and massive neutrinos are explicitly rejected.

- Use a smooth Halofit root find in the `jaxcosmo` theory layer: `jax-cosmo` locates the
  nonlinear scale by interpolating `sigma(R) = 1` on a fixed 256-node grid, which makes the
  power spectrum jump between nodes and spoils finite differences even at small steps. The
  replacement integrates `sigma^2(R)` on a fine grid and solves it with a fixed-count Newton
  iteration, which converges to machine precision. The values stay within 5e-4 of `jax-cosmo`
  at the fiducial point and within 0.6% over the validation parameter box (the difference is
  `jax-cosmo`'s own interpolation error in the nonlinear scale), while being differentiable.

- Review fixes for the `jaxcosmo` branch (see `docs/source/jaxcosmo_review.md`): correct
  orientation of the Savitzky-Golay operator in the JAX spectroscopic observable (fiducial
  `ln P` agreement with the legacy code improves from 8e-5 to 2e-6); observable cache of
  the derivative providers keyed on the full analysis configuration; `kh_rescaling_bug`
  options and `AP_effect=False` handled like the legacy `observed_Pgg`; new
  `NUMERICS.legacy_pk_clamp` option reproducing the table-edge clamp of the legacy power
  spline; tests for the theory layer, option branches and providers.

- Support NumPy 2: widen the dependency pin from `numpy<2` to `numpy>=2.1,<2.3`
  (2.2.x is the newest series supporting Python 3.10), relock, and compute the
  legacy `round_decimals_up` branch in float64 to keep results identical under
  NEP 50 promotion rules. Full Fisher backend validation against the NumPy 1
  results is run separately.

- Add `fishconsumer.make_fisher_chain_plot` for paired Fisher/posterior overlays,
  with marginalization to plotted parameters, matched colors for missing-chain
  cases, independent `show_fisher`/`show_chains` switches, and per-call limits
  using Gaussian-equivalent weighted posterior quantiles (3 sigma by default).

- Build sampler analysis and worker contexts with the configured prior parameters,
  preventing nondefault coordinates such as `10^9As` from being dropped.

- Translate public dimensionless `As` to CLASS `A_s` and order-one `10^9As`
  to CAMB `As` (CLASS/SYREN-NEW already support it), label amplitude coordinates
  consistently, and preserve small fiducials in Fisher parameter-file exports.
  EE2 now reads the exact computed CLASS amplitude instead of interpolating
  primordial-grid values near its hard amplitude bounds.

- Restore matched CAMB/CLASS nonlinear cb subtraction and the shared default
  neutrino-density factor, retaining nonlinear-grid hardening. Native CLASS cb
  remains opt-in; see `docs/source/neutrino_cb_regression.md` for local evidence.

- Migrate the documentation to the PyData Sphinx Theme with a card-based landing
  page, Getting started / User guide / Tutorials / API / Development sections,
  autosummary-generated per-module API pages, a new `docs` extra
  (`uv sync --extra docs`), a build-only docs CI job, and warnings-as-errors
  builds. Fix docstrings that broke the API build.
- Publish the documentation to the `docs/` folder of `gh-pages` with a new
  `docs.yml` GitHub Actions workflow (on pushes to `main` and version tags) and
  retire Read the Docs (`.readthedocs.yaml` removed).
- Vectorize the photometric Cl pipeline (row-wise `sqrtP_limber`, cached kernels
  and window factors) for up to ~6x faster Fisher matrix computation with the
  symbolic backend; see `docs/source/photometric_optimization_results.md`.
- Generalize the Nautilus sampler to a registered-likelihood-component
  architecture, fix spawn-safety for multiprocessing worker pools, and add
  Slurm launch scripts for HPC runs.
- Validate WL/GCsp/joint Nautilus likelihoods against Fisher forecasts; see
  `docs/source/likelihood_validation.md`.
- Fix a CLASS/HMcode nonlinear power spectrum error at z=6, normalize CAMB
  input paths, and various spectro/base likelihood and job-handling fixes.
- Remove the unused Ray options `parallel` and `parallel_cpus` of `FisherMatrix`
  and the `ray_session` helper.
- Default `survey_name_radio_IM` to the shipped `SKAO-IM-Redbook`, and look up
  survey specification files in `other_survey_specifications` when they are not
  found in `specs_dir`.
- Use the standard `Neff = 3.044` for CAMB when the fiducial cosmology has no `Neff`.
- Warn when the `POLY` or `STEM` derivative methods are used with non-photometric
  probes, or `4PT_FWD` with CMB probes.
- Nautilus sampler: do not modify the caller's config, write the real package
  version to the metadata, store an evidence of 0.0, warn about priors that are
  ignored, raise a clear error with fewer than two sampled parameters, add an
  `overwrite` option to resample an existing run, and export `load_chain_metadata`.
- Parse MontePython parameter files with `ast.literal_eval` instead of `eval` and
  fix the `FishConsumer` chain-loading docstrings.
- Point the validation dashboard links to `https://www.santicasas.space/cosmicfishpie/dashboard/`.
- Add explicit analysis context runtime isolation and refresh the comprehensive guide.
- Symbolic backend overhaul to the new syren emulators, pinning the `symbolic-pofk`
  GitHub tarball in CI instead of a local editable path.
- Expose derivatives on `FisherMatrix`, add CLASS EE2 profiles, and harden nonlinear
  spectra handling.
- Reorganize `scripts/` into category subfolders and notebooks into a numbered workflow
  (legacy notebooks archived); add backend validation and symbolic provenance docs pages.

## 1.3.1

- Add the CAMB/CLASS validation suite, threshold-aware dashboard, and manual
  GitHub Pages publication workflow.

1.3.0 : Performance optimization in Photo module and benchmarking scripts
1.2.8 : Added compehensive test catalogue.
1.2.7 : Added test for analysis module
1.2.6 : New demo notebook GCsp
1.2.5 : Likelihood modules tested for photo and spectro
1.2.4 : Discontinued support for python3.8. Fixed the style for likelihood scripts.
1.2.3 : Added a new likelihood function for photometric and spectroscopic surveys.
1.2.2 : Galaxy sample split for sources and lenses. Feedback prints more consistent.
1.2.1 : Updating configs of other surveys: SKAO, DESI, LSST to work in new config file structure
1.2.0 : Big update of configuration, specification yamls and nuisance parameter interface. No backwards compatibility of yamls!
1.1.4 : Coverage reports with Codecov
1.1.3 : New split up demonstration notebooks
1.1.2 : Coverage badge
1.1.1 : Resolved gcsp bias bug, new test suite.
1.1.0 : Implemented symbolic as a new boltzmann solver class. Fixes of feedback prints. More docstrings.
1.0.8 : Overview documentation and initial tests
1.0.7 : Added a new installation script to download external data from the github

1.0.6 : Restructured default parameters for EBS and Survey Specifications

1.0.5 : Removed a backwards compatibility issue with older version of numpy.
        Switched Integration to use scipy now

1.0.4 : Merged changes from the old repo to the release repo.
        Mainly changes to the plotting routine
        Added new external file for nonlinear power spectra
        New options for spectroscopic error. Removed infamous h-bug

1.0.3 : Added documentation for the cosmicfishpie.fishermatrix submodule

1.0.2 : Added documentation for the cosmicfishpie.LSSsurvey submodule

1.0.1 : Bugfix for external file reading for release notebooks.
        Change `is False` statements with a normal `not` to catch False-equivalents.
