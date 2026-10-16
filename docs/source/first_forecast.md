(first-forecast)=
# A verified first forecast

This page runs a complete photometric Fisher forecast (weak lensing plus
photometric galaxy clustering) for two cosmological parameters. It uses the
built-in `symbolic` backend, so you need no Einstein-Boltzmann solver setup, and
it finishes in well under a minute. It is the same compact configuration the
test suite exercises.

## Run it

Save the script as `first_fisher.py` and run it with `python first_fisher.py`
(inside a source checkout, `uv run python first_fisher.py`). Any working
directory will do; the script creates `results/` there.

```python
from cosmicfishpie.fishermatrix import cosmicfish as cff

options = {
    "accuracy": 1,
    "feedback": 1,
    "code": "symbolic",
    "cosmo_model": "LCDM",
    "derivatives": "3PT",
    "ell_sampling": 25,  # coarse ell grid: fast, but not production accuracy
    "nonlinear": True,
    "outroot": "first_forecast",
    "results_dir": "results",
    "survey_name": "Euclid",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "survey_name_spectro": False,
}

fiducial = {"Omegam": 0.32, "h": 0.67}

# Relative finite-difference step size for each free parameter.
freepars = {"Omegam": 0.01, "h": 0.01}

forecast = cff.FisherMatrix(
    fiducialpars=fiducial,
    freepars=freepars,
    options=options,
    observables=["WL", "GCph"],
    cosmoModel=options["cosmo_model"],
    surveyName=options["survey_name"],
)

fisher = forecast.compute()
print("Fisher matrix shape:", fisher.fisher_matrix.shape)
print("Parameter names:", fisher.get_param_names())
```

The output is a `fisher_matrix` object (not a dictionary). The matrix itself is
`fisher.fisher_matrix`, and `fisher.get_param_names()` lists its parameters. Besides
`Omegam` and `h` the list includes nuisance parameters (galaxy bias, intrinsic
alignment) that the survey specification adds automatically, so the matrix is
larger than 2 x 2.

Because `outroot` is non-empty, `compute()` also writes a Fisher matrix text file, a
`.paramnames` file and run-specification files to `results_dir`. File names
contain the package version, the output root and the observables.

## Use CAMB or CLASS instead

Change `"code"` to `"class"` and nothing else needs to change. For `"camb"`, also add
`"Neff": 3.044` to `fiducial`; without it the CAMB setup stops with a
`KeyError: 'num_nu_massless'`. Both are slower than `symbolic`, because they run a
Boltzmann solver for every derivative step. `CAMB` and `CLASS` are installed with the
package.

## Next steps

- Plot and compare the result: {doc}`overview`.
- Choose other observables, surveys and options: {doc}`comprehensive_guide`.
- Sample the full posterior instead of the Gaussian approximation:
  {doc}`likelihood_sampling`.
