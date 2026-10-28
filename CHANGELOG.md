# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
