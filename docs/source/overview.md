# Overview

`cosmicfishpie` computes Fisher-matrix forecasts for upcoming surveys such as Euclid,
DESI, SKAO and CMB experiments. It evaluates observables (galaxy clustering, weak
lensing, CMB spectra, intensity mapping), differentiates them numerically with respect
to the parameters you choose, and returns a Fisher matrix you can marginalise, combine
and plot. Cosmology comes from CAMB, CLASS, the built-in `symbolic` backend or
precomputed tables.

This page explains the pieces of a forecast and shows how to plot one. To run the code
first, start with {doc}`first_forecast`.

## The four inputs

Every forecast, including the one in {doc}`first_forecast`, is built from four inputs
that you pass to `FisherMatrix`:

| Input | What it is |
| --- | --- |
| `options` | A dictionary of settings: backend (`code`), survey names, derivative scheme, accuracy, output location. The full list is in {mod}`cosmicfishpie.configs.config`. |
| `observables` | Which observables to forecast, for example `["WL", "GCph"]` or `["GCsp"]`. |
| `fiducialpars` | The cosmology where the forecast is evaluated. Parameters you leave out take their code defaults. |
| `freepars` | The parameters to vary, each with its relative finite-difference step size. |

`FisherMatrix(...).compute()` runs the numerical derivatives and returns a
`cosmicfishpie.analysis.fisher_matrix` object, which holds the matrix, the parameter
names and a record of the settings. See {ref}`Supported observables and survey settings
<observables>` for the available observables and survey files.

## Plot a result

Pass the Fisher matrix from {doc}`first_forecast` (the `fisher` variable) to
`fisher_plotting` to draw a triangle plot of the free parameters:

```python
import cosmicfishpie.analysis.fisher_plotting as cfp

plot_options = {
    "fishers_list": [fisher],
    "fish_labels": ["Euclid photometric (WL + GCph)"],
    "plot_pars": ["Omegam", "h"],
    "axis_custom_factors": {"all": 7},  # axis range in sigma of the first Fisher matrix
    "plot_method": "Gaussian",
    "file_format": ".pdf",
    "outpath": "./plots/",  # created if it does not exist
    "outroot": "first_forecast_plot",
    "colors": ["#000000"],
}

plotter = cfp.fisher_plotting(**plot_options)
plotter.plot_fisher(filled=[False])
```

This writes `plots/first_forecast_plot_contours.pdf`. To compare several surveys or
settings, add more Fisher matrices to `fishers_list` and matching entries to
`fish_labels` and `colors`; the other options are in the
{mod}`cosmicfishpie.analysis.fisher_plotting` API reference.

## Where to go next

- {doc}`comprehensive_guide`: backends, observables, survey files, and working with the
  Fisher object (marginalise, combine, reshuffle).
- {doc}`derivatives`: the finite-difference schemes and how to choose one.
- {doc}`likelihood_sampling`: posterior sampling with Nautilus.
