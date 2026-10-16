# Comprehensive guide

This guide covers the options you can choose when building a Cosmicfishpie
Fisher forecast, and what you can do with the result. For a runnable end-to-end
example, start with {ref}`A verified first forecast <first-forecast>`. For
installation, see the [installation guide](installation.md).

## Core concepts

A forecast combines:

- a **fiducial cosmology**: the parameter values at which observables are evaluated;
- **free parameters**, each with a finite-difference step size used for derivatives;
- one or more **observables**, such as `GCph` (photometric galaxy clustering) and
  `WL` (weak lensing); and
- a **survey specification**, loaded from the packaged YAML files.

## Choose a cosmology backend

Set `options["code"]` to choose where the cosmology comes from:

| `code` | Source |
| --- | --- |
| `camb` | CAMB (the default if `code` is not set). |
| `class` | CLASS. |
| `symbolic` | Built-in backend; no Boltzmann solver is run, so it is much faster. |
| `external` | Precomputed tables read from files (see {ref}`below <external-backend>`). |

CAMB and CLASS are installed together with Cosmicfishpie (as the `camb` and
`classy` packages), so `camb` and `class` work without extra setup. They run a
Boltzmann solver for every derivative step and are therefore slower than
`symbolic`. Omit the YAML path to use the default configurations in
`cosmicfishpie/configs/default_boltzmann_yaml_files/`.

With `code="camb"`, the fiducial cosmology must include `Neff` (for example
`"Neff": 3.044`); without it the CAMB setup stops with
`KeyError: 'num_nu_massless'`. CLASS does not need it.

The examples in this guide use `symbolic` so that they run quickly. To use
CAMB or CLASS, change `code` (and add `Neff` for CAMB); check a small forecast
against your solver setup before using it for science.

(observables)=
## Observables and survey settings

| Observable | Meaning | Notes |
| --- | --- | --- |
| `GCph`, `WL` | Euclid photometric clustering and shear | Run together or separately. |
| `GCsp` | Euclid spectroscopic clustering | Run as a separate forecast from `GCph`/`WL`. |
| `CMB_T`, `CMB_E`, `CMB_B` | CMB temperature and polarisation | Use `surveyName="Planck"`. |
| `IM` | HI intensity mapping | Needs `survey_name_radio_IM` (see below). |

`FisherMatrix.compute()` takes the photometric branch if `GCph` or `WL` is
requested and the spectroscopic branch if `GCsp` is requested, so run these as
separate forecasts and add the resulting Fisher objects if appropriate.

For the CMB, do not use `TT`, `EE`, `BB`, `TE`, `survey_name="CMB"` or
`survey_name_cmb`; they are not valid configuration names. See
[CMB benchmarks](cmb_benchmarks.md) for the Planck, Simons Observatory and
Stage-IV specs, the single-ell-range limitation, and smoke-test scripts.

For a custom survey, pass a complete `specifications` dictionary to
`FisherMatrix`. Adding a YAML file alone does not register a new survey name.

(survey-files)=
### Shipped survey files

Pass a file name (without `.yaml`) as `survey_name_photo`, `survey_name_spectro`
or `survey_name_radio_IM`, and set `specs_dir` to the folder that contains it.
Both folders live in `cosmicfishpie/configs/`; `default_survey_specifications`
is the default, while `other_survey_specifications` must be set explicitly.

`default_survey_specifications/`

| File | Description |
| --- | --- |
| `Euclid-Photometric-ISTF-Pessimistic` | `GCph`/`WL`, IST:F pessimistic: `lmax` 750/1500, `fsky` 0.35. Default photometric survey. |
| `Euclid-Photometric-Paper-Pessimistic` | Same scales, with `AIA`, `etaIA` varied (arXiv:2405.06047). |
| `Euclid-Photometric-Paper-Optimistic` | As above with `lmax` 3000/5000. |
| `Euclid-Spectroscopic-ISTF-Pessimistic` | `GCsp`, IST:F pessimistic: 15000 deg², `kmax` 0.25 h/Mpc. Default spectroscopic survey. |
| `Euclid-Spectroscopic-Paper-Pessimistic` | Same setup, from arXiv:2405.06047. |
| `Euclid-Spectroscopic-Paper-Optimistic` | As above with `kmax` 0.30 h/Mpc. |
| `Planck` | CMB: `fsky` 0.7, `lmax` 1500. Loaded with `surveyName="Planck"`. |
| `Simons-Observatory-PlanckLowEll`, `CMB-Stage4-PlanckLowEll` | CMB with Planck low-ell: `fsky` 0.4, `lmax` 3000. Not selected by `surveyName` (see [CMB benchmarks](cmb_benchmarks.md)). |

`other_survey_specifications/`

| File | Description |
| --- | --- |
| `Euclid-Photometric-DeboleR1` | Photometric, different bins, 2500 deg². |
| `Euclid-Photometric-ISTF-GW` | IST:F photometric with `lmax_GCph` 1000 and a `sqrt` bias model. |
| `Euclid-Spectroscopic-DR1-Pessimistic` | DR1-like `GCsp`: 2500 deg², `kmax` 0.15 h/Mpc. |
| `Euclid-Spectroscopic-ISTF-Pessimistic-sigma_pv` | IST:F `GCsp` with the `rescale_sigma_pv` nonlinear model. |
| `Euclid-Spectroscopic-ISTF-Pessimistic-Qbias`, `-Qbias-sigma_pv`, `-Qbias-1par` | IST:F `GCsp` with the `linear_Qbias` scale-dependent bias (optionally with `rescale_sigma_pv`, or with only `A2`). |
| `SKAO-Photometric-Redbook` | SKAO photometric: `fsky` 0.1166. |
| `SKAO-Spectroscopic-Redbook` | SKAO HI `GCsp`: 5000 deg², `kmax` 0.25 h/Mpc. |
| `SKAO-IM-Redbook` | SKAO HI intensity mapping (`IM`): 5000 deg². |
| `MeerKlass-IM-Ze` | MeerKLASS `IM`: 7500 deg², 60 dishes. |

Always set `survey_name_radio_IM` explicitly for `IM` forecasts. Its default
name (`SKA1-IM-Redbook-Optimistic`) matches no shipped file: Cosmicfishpie warns,
falls back to a Euclid spectroscopic file, and stops with
`KeyError: 'IM_bias_model'`.

```python
options = {
    "specs_dir": "cosmicfishpie/configs/other_survey_specifications",
    "survey_name_radio_IM": "SKAO-IM-Redbook",
    "survey_name_photo": False,
    "survey_name_spectro": False,
}
forecast = cff.FisherMatrix(..., options=options, observables=["IM"], surveyName="SKAO")
```

```{note}
No Rubin or DESI specification file is shipped. The `Rubin` and `DESI` branches in
`config.py` load whatever file you name in `survey_name_photo` or
`survey_name_spectro`.
```

## Derivatives and nuisance parameters

Select the finite-difference method with `options["derivatives"]`: `3PT`
(default), `STEM`, `POLY` or `4PT_FWD`. Not every method supports every
observable; see {ref}`Observable support by method <observable-support-by-method>`
before using anything other than `3PT` for CMB, `GCsp` or `IM`.

Nuisance parameters are survey- and model-specific. For example, the default
Euclid spectroscopic specification uses `lnbg_1` to `lnbg_4` for its logarithmic
galaxy bias. Check the survey YAML before adding nuisance parameters to
`freepars`. The `Nuisance` class (`cosmicfishpie.cosmology.nuisance`) builds the
bias, shot-noise and intrinsic-alignment functions from these settings; you
rarely need to call it directly.

## Advanced options

These keys go in `options`; the defaults suit most forecasts.

| Option | Default | Meaning |
| --- | --- | --- |
| `GCsp_Tracer`, `GCph_Tracer` | `"matter"` | Power spectrum traced by clustering: `"matter"` or `"clustering"` (CDM + baryons). |
| `activateMG` | `False` | Modified-gravity lensing term: `"late-time"` (needs `E11`, `E22` in the cosmological parameters) or `"external"` (reads the `SigmaWL` file). |
| `ShareDeltaNeff` | `False` | If `True`, a varied `Neff` rescales the neutrino decoupling temperature; otherwise extra `Neff` is added as massless relics. |
| `memorize_cosmo` | `False` | Cache cosmologies in `memory_cache/` (only used by `code="external"`). |

Two legacy switches, `kh_rescaling_bug` and `kh_rescaling_beforespecerr_bug`
(default `False`), reproduce an old, unphysical `h/h_fid` rescaling of `k` in the
spectroscopic power spectrum. Leave them off unless you must reproduce older
results.

## Work with the Fisher object

The returned object holds the matrix and its parameter metadata. Add
compatible Fisher objects with `+`:

```python
combined = fisher_a + fisher_b
print(combined.get_param_names())
```

The module `cosmicfishpie.analysis.fisher_operations` provides `marginalise`,
`marginalise_over`, `reshuffle` and `information_gain` (the Fisher approximation
of the Kullback-Leibler divergence between two experiments, given a prior).
To keep `Omegam` and `h` and marginalise over everything else:

```python
from cosmicfishpie.analysis.fisher_operations import marginalise

reduced = marginalise(fisher, ["Omegam", "h"])
```

To fix parameters at their fiducial values instead (dropping their rows and
columns without marginalising), use `eliminate_parameters(fisher, names)`, or
`eliminate_columns_rows(fisher, indexes)` with zero-based positions.

A forecast with a non-empty `outroot` also writes a `*_specs.json` snapshot next
to the Fisher matrix. `cosmicfishpie.utilities.utils.load_fisher_from_json(path)`
rebuilds the `FisherMatrix` from it and returns `(forecast, snapshot)`; call
`forecast.compute()` to rerun it.

For debugging, every forecast keeps its raw derivatives in `forecast.derivs_dict`
(photometric forecasts also expose them as `forecast.photo_derivs`). Other
intermediate attributes are not a stable API; prefer the returned Fisher object.
The names `add_fishers`, `marginalize`, `conditionalize` and `resample_basis`
do not exist.

## Plot and compare Fisher matrices

- `CosmicFish_FisherAnalysis` (`cosmicfishpie.analysis.fisher_plot_analysis`)
  holds a list of Fisher matrices, or loads them from files, and applies
  `marginalise` / `reshuffle` to all of them at once.
- `fisher_plotting` (`cosmicfishpie.analysis.fisher_plotting`) draws comparison
  contour plots for such a group.
- `cosmicfishpie.analysis.fishconsumer` provides `make_triangle_plot` (Fisher
  matrices and MCMC chains), `fishtable_to_pandas` for error tables, and chain
  loaders; see {doc}`likelihood_sampling`.

```python
from cosmicfishpie.analysis.fisher_plot_analysis import CosmicFish_FisherAnalysis
from cosmicfishpie.analysis.fishconsumer import make_triangle_plot

fisher.name = "WL+GCph"
analysis = CosmicFish_FisherAnalysis(fisher_list=[fisher])
small = analysis.marginalise(["Omegam", "h"])  # applied to every Fisher

fig = make_triangle_plot(fishers=[fisher], fisher_labels=["WL+GCph"], params=["Omegam", "h"])
```

## Advanced topics

(external-backend)=
### Precomputed cosmologies (`code="external"`)

With `code="external"`, Cosmicfishpie reads background, growth and power-spectrum
tables from text files instead of running a solver. Pass `extfiles` to
`FisherMatrix`, with a `directory` holding one folder for the fiducial cosmology
and one for each finite-difference step:

```text
my_cosmologies/
  fiducial_eps_0/
  Omegam_pl_eps_1p0E-02/     # <folder name>_<pl|mn>_eps_<relative step>
  Omegam_mn_eps_1p0E-02/
  h_pl_eps_1p0E-02/
  h_mn_eps_1p0E-02/
```

Each folder contains the same files (any extension, read with `numpy.loadtxt`):
`z_values_list`, `k_values_list`, `background_Hz`, `sigma8-z`, and the
`(n_z, n_k)` tables `D_Growth-zk`, `f_GrowthRate-zk`, `Plin-zk` and `Pnonlin-zk`.
Different names can be set in `extfiles["file_names"]`. `GCsp_Tracer="clustering"`
additionally needs the `*cb*` files.

```python
extfiles = {
    "directory": "my_cosmologies",
    "fiducial_folder": "fiducial_eps_0",
    "paramnames": ["Omegam", "h"],         # keys of the cosmological parameters
    "folder_paramnames": ["Omegam", "h"],  # folder prefixes, same order
    "eps_values": [0.01],                  # relative steps available on disk
    "k-units": "1/Mpc",                    # or "h/Mpc"
    "r-units": "1/Mpc",                    # H(z) units; or "Mpc/h", "km/s/Mpc"
}
forecast = cff.FisherMatrix(..., options={"code": "external"}, extfiles=extfiles)
```

Only parameters listed in `folder_paramnames` can be varied.

### Cosmology calculator

`cosmo_functions` gives direct access to distances, growth and power spectra
without running a Fisher calculation. It reads the global configuration, so call
`config.init` first:

```python
from cosmicfishpie.configs import config as cfg
from cosmicfishpie.cosmology.cosmology import cosmo_functions

cfg.init(
    options={"code": "symbolic", "feedback": 0},
    observables=["GCph"],
    fiducialpars={"Omegam": 0.32, "h": 0.67},
    freepars={"Omegam": 0.01, "h": 0.01},
)

cosmo = cosmo_functions(cfg.fiducialparams)
cosmo.Hubble(0.5, physical=True)        # H(z) in km/s/Mpc
cosmo.comoving(0.5)                     # comoving distance, Mpc
cosmo.Pmm(0.5, 0.1, nonlinear=False)    # P(k) at z=0.5, k=0.1 1/Mpc
cosmo.sigma8_of_z(0.5)                  # sigma_8(z)
```

To avoid global state, `cosmicfishpie.configs.context.build_analysis_context`
takes the same arguments as `config.init` and returns an immutable
`AnalysisContext`. Pass it as
`cosmo_functions(dict(ctx.fiducialparams), configuration=ctx)`.

### Performance

`COSMICFISH_FAST_EFF`, `COSMICFISH_FAST_P` and `COSMICFISH_FAST_KERNEL` enable
fast implementations in the photometric calculations. They are on by default;
set one to `0` to compare against the slower path.

`FisherMatrix(..., parallel=True, parallel_cpus=8)` (experimental) starts a
[Ray](https://docs.ray.io/en/latest/) session with `parallel_cpus` CPUs
(default 4) and shuts it down at the end of `compute()`. Ray is not a
dependency; install it with `pip install ray`. Currently no part of the
calculation is sent to Ray, so expect no speed-up.

## Project status and citation

Cosmicfishpie has a backend-neutral derivative-provider interface, so a future
differentiable solver or emulator can supply derivatives. JAX is not currently
an implemented backend.

To cite the project, use the repository's
[CITATION.cff](https://github.com/santiagocasas/cosmicfishpie/blob/main/CITATION.cff).

## Next steps

- Start from the verified photometric example and change one option at a time.
- Browse the [scripts reference](scripts_reference.md) for benchmark, validation
  and backend-comparison workflows.
- Read the API documentation before adding a new survey model or parameterisation.
