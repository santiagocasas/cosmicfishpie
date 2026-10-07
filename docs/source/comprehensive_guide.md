# Comprehensive Guide

This guide introduces the supported workflow for building and inspecting a
Cosmicfishpie Fisher forecast. It uses the built-in `symbolic` backend so that
the example can run without installing or configuring an external
Einstein--Boltzmann solver. Use `camb` or `class` for scientific forecasts once
you have validated the corresponding solver setup.

For installation instructions, see the [repository README](https://github.com/santiagocasas/cosmicfishpie#readme)
and the [installation guide](installation.md).

## Core concepts

Cosmicfishpie forecasts parameter constraints with a Fisher information
matrix. A forecast combines:

- a **fiducial cosmology**, the parameter values at which the observable is evaluated;
- **free parameters**, with finite-difference step sizes used to calculate derivatives;
- one or more supported **observables**; and
- a survey specification loaded from the packaged YAML files.

The examples below use Euclid photometric observables:

- `GCph`: photometric galaxy clustering;
- `WL`: weak lensing (cosmic shear).

## A verified first forecast

A complete, runnable photometric example lives in the tutorial
{ref}`A verified first forecast <first-forecast>`.

## Using CAMB or CLASS

The `camb` and `class` backends are selected with `options["code"]`. They need
their own installed solver and configuration. Cosmicfishpie supplies default
YAML configurations below
`cosmicfishpie/configs/default_boltzmann_yaml_files/`; omit an explicit YAML
path to use the backend default.

Before using a solver for production work, run a small forecast and inspect its
output. Solver settings, numerical precision, survey assumptions, and derivative
step sizes can all affect a Fisher forecast.

(observables)=
## Supported observables and survey settings

### Euclid photometric and spectroscopic forecasts

Euclid photometric forecasts use `GCph`, `WL`, or both. Euclid spectroscopic
galaxy-clustering forecasts use `GCsp`. A `FisherMatrix.compute()` call chooses
the photometric branch when `GCph` or `WL` is requested and the spectroscopic
branch when `GCsp` is requested. Run these as separate forecasts, then combine
the resulting Fisher objects if their assumptions make that appropriate.

For the packaged Euclid survey files, set `specs_dir` to
`cosmicfishpie/configs/default_survey_specifications` when running from the
repository root. The available filenames define the values accepted by
`survey_name_photo` and `survey_name_spectro`.

### CMB forecasts

The current CMB implementation supports the observable tokens `CMB_T`,
`CMB_E`, and `CMB_B`. For the packaged Planck specification, use
`surveyName="Planck"` when constructing `FisherMatrix`. Do not use `TT`, `EE`,
`BB`, `TE`, `survey_name="CMB"`, or `survey_name_cmb`: they are not current
Cosmicfishpie configuration names.

### Intensity mapping

Intensity-mapping forecasts use the observable token `IM`. The configuration
key for the radio intensity-mapping survey specification is
`survey_name_radio_IM`. This is distinct from Euclid galaxy clustering and is
not configured through `survey_name_radio`.

For a custom survey, pass a complete `specifications` dictionary to
`FisherMatrix`. Adding a YAML file alone does not register an arbitrary survey
name; survey-file loading currently has explicit branches for the supported
survey families.

## Working with the returned Fisher object

The returned object contains the numeric matrix and its parameter metadata.
For example, independent Fisher objects with compatible parameter definitions
can be added using the object addition operator:

```python
combined = fisher_a + fisher_b
print(combined.get_param_names())
```

The `cosmicfishpie.analysis.fisher_operations` module provides operations named
`marginalise`, `marginalise_over`, and `reshuffle`. For example, to retain only
`Omegam` and `h` while marginalising over all other parameters:

```python
from cosmicfishpie.analysis.fisher_operations import marginalise

reduced = marginalise(fisher, ["Omegam", "h"])
```

Do not use the undocumented names `add_fishers`, `marginalize`,
`conditionalize`, or `resample_basis`.

## Derivatives and nuisance parameters

The supported finite-difference methods are `3PT`, `STEM`, `POLY`, and
`4PT_FWD`; select one with `options["derivatives"]`. The default is `3PT`.

Nuisance parameters are survey- and model-specific. For example, the packaged
Euclid spectroscopic pessimistic specification uses names such as `lnbg_1`
through `lnbg_4` for its default logarithmic galaxy-bias model. Consult the
selected survey YAML and configuration before adding nuisance parameters to
`freepars`.

Intermediate attributes are probe-specific and are not a stable public API.
For debugging, a spectroscopic forecast currently stores derivatives in
`forecast.derivs_dict`, while a photometric forecast uses
`forecast.photo_derivs`. Prefer the returned Fisher object for ordinary
post-processing.

## Performance settings

`COSMICFISH_FAST_EFF`, `COSMICFISH_FAST_P`, and `COSMICFISH_FAST_KERNEL`
control optional fast implementations used in photometric calculations. They
are enabled by default; set a variable to `0` to disable its fast path when you
need a comparison against the slower implementation. Validate any scientific
result with the settings appropriate to your analysis.

## Project status

Cosmicfishpie has a backend-neutral derivative-provider interface. It allows a
future differentiable solver or emulator to provide derivatives without coupling
probe covariance code to a particular automatic-differentiation library. JAX is
not currently an implemented Cosmicfishpie backend.

## Citation

Use the repository's [CITATION.cff](https://github.com/santiagocasas/cosmicfishpie/blob/main/CITATION.cff)
file as the citation source. It is the maintained project metadata; this guide
does not duplicate a BibTeX record that could drift from it.

## Next steps

- Start from the verified photometric example and change one option at a time.
- Use the existing scripts in `scripts/` for benchmark and backend-comparison
  workflows.
- Read the API documentation for configuration and analysis classes before
  adding a new survey model or parameterisation.
