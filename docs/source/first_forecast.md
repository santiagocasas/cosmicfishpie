(first-forecast)=
# A verified first forecast


Run the following script from the repository root. It uses the same compact
configuration exercised by the test suite. `ell_sampling` keeps the tutorial
fast; it is not a production-accuracy setting.

```python
from pathlib import Path

from cosmicfishpie.fishermatrix import cosmicfish as cff

options = {
    "accuracy": 1,
    "feedback": 1,
    "code": "symbolic",
    "cosmo_model": "LCDM",
    "derivatives": "3PT",
    "ell_sampling": 25,
    "nonlinear": True,
    "outroot": "first_forecast",
    "results_dir": "results",
    "specs_dir": "cosmicfishpie/configs/default_survey_specifications",
    "survey_name": "Euclid",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "survey_name_spectro": False,
}

fiducial = {
    "Omegam": 0.32,
    "h": 0.67,
}

# Values are relative finite-difference step sizes for non-zero parameters.
freepars = {
    "Omegam": 0.01,
    "h": 0.01,
}

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

# `compute()` creates the output directory when outroot is non-empty.
print("Results directory:", Path(options["results_dir"]).resolve())
```

Run it with:

```bash
uv run python first_fisher.py
```

`compute()` returns a `fisher_matrix` object, not a dictionary. Its matrix is
available as `fisher.fisher_matrix`; use methods such as
`fisher.get_param_names()` to inspect its metadata. Depending on the selected
survey specification, the result can also include automatically configured
nuisance parameters in addition to the two cosmological parameters above.

With a non-empty `outroot`, Cosmicfishpie writes a versioned Fisher matrix text
file, a `.paramnames` sidecar, and run-specification files below `results_dir`.
The exact filename includes the package version, output root, and observables.
