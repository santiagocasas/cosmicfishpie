# cosmicfishpie

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![Documentation](https://github.com/santiagocasas/cosmicfishpie/actions/workflows/docs.yml/badge.svg)](https://www.santicasas.space/cosmicfishpie/docs/)
[![codecov](https://codecov.io/github/santiagocasas/cosmicfishpie/graph/badge.svg?token=BXTVDPXPUO)](https://codecov.io/github/santiagocasas/cosmicfishpie)
[![37.50 % FAIR](https://img.shields.io/badge/FAIR_assessment-37.50_%25-red)](https://fair-checker.france-bioinformatique.fr/assessment/68da9e3cc49e421b3e2cf501)
[![CI](https://github.com/santiagocasas/cosmicfishpie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/santiagocasas/cosmicfishpie/actions/workflows/ci.yml)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/santiagocasas/cosmicfishpie)

`cosmicfishpie` is a Python package for cosmological Fisher forecasts. It helps prepare and compare forecasts for surveys such as Euclid, DESI, SKAO and CMB experiments.

<div align="center">
  <img src="https://github.com/santiagocasas/cosmicfishpie/assets/6987716/1816b3b7-0920-4a2c-aafd-9c4ba1dc3e2b" width="280">
</div>

## Features

- **Observables.** Photometric (`GCph`, `WL`) and spectroscopic (`GCsp`) galaxy clustering and weak lensing, CMB (`CMB_T`, `CMB_E`, `CMB_B`) and HI intensity mapping (`IM`).
- **Backends.** CAMB, CLASS, a built-in `symbolic` backend that needs no Einstein-Boltzmann solver, or precomputed tables.
- **Derivatives.** Four finite-difference schemes (`3PT`, `STEM`, `POLY`, `4PT_FWD`), chosen with one option.
- **Analysis and plots.** Marginalise, reshuffle and combine Fisher matrices, and draw triangle plots to compare surveys and settings.
- **Likelihood sampling.** Sample the full posterior of the photometric and spectroscopic likelihoods with [Nautilus](https://nautilus-sampler.readthedocs.io/) and compare it with the Fisher forecast.

## Installation

`cosmicfishpie` supports Python `>=3.10,<3.13`.

```bash
pip install cosmicfishpie
```

To work from a source checkout, use [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/santiagocasas/cosmicfishpie.git
cd cosmicfishpie
uv sync --extra dev
```

Check the installation with

```bash
python -c "import cosmicfishpie; print('cosmicfishpie import OK')"
```

and, from a source checkout, run the tests with `uv run pytest`. Conda setup, optional extras and more are in the [installation guide](https://www.santicasas.space/cosmicfishpie/docs/installation.html).

## Minimal example

A photometric forecast with the `symbolic` backend. It needs no solver setup and runs in a few seconds from any directory, writing its output to `results/`:

```python
from cosmicfishpie.fishermatrix import cosmicfish as cff

options = {
    "accuracy": 1,
    "feedback": 1,
    "code": "symbolic",  # or "class" / "camb"
    "cosmo_model": "LCDM",
    "derivatives": "3PT",
    "ell_sampling": 25,  # coarse grid for speed, not production accuracy
    "nonlinear": True,
    "outroot": "first_forecast",
    "results_dir": "results",
    "survey_name": "Euclid",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "survey_name_spectro": False,
}

fiducial = {"Omegam": 0.32, "h": 0.67}
freepars = {"Omegam": 0.01, "h": 0.01}  # relative finite-difference steps

forecast = cff.FisherMatrix(
    fiducialpars=fiducial,
    freepars=freepars,
    options=options,
    observables=["WL", "GCph"],
    cosmoModel=options["cosmo_model"],
    surveyName=options["survey_name"],
)

fisher = forecast.compute()
print(fisher.fisher_matrix.shape, fisher.get_param_names())
```

## Documentation

- [Documentation](https://www.santicasas.space/cosmicfishpie/docs/): getting started, user guide, tutorials, API reference and development notes.
- [Validation dashboard](https://www.santicasas.space/cosmicfishpie/dashboard/)
- [Citation metadata](CITATION.cff) (`CITATION.cff`)

To build the documentation locally:

```bash
uv sync --extra docs
make docs   # live-reloading server at http://127.0.0.1:8000
```

CI builds the docs with warnings treated as errors:

```bash
uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/build/html
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sefa76"><img src="https://avatars.githubusercontent.com/u/19888927?v=4?s=100" width="100px;" alt="Sefa Pamuk"/><br /><sub><b>Sefa Pamuk</b></sub></a><br /><a href="https://github.com/santiagocasas/cosmicfishpie/commits?author=Sefa76" title="Code">💻</a> <a href="#design-Sefa76" title="Design">🎨</a> <a href="#ideas-Sefa76" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/santiagocasas/cosmicfishpie/commits?author=Sefa76" title="Tests">⚠️</a> <a href="#research-Sefa76" title="Research">🔬</a> <a href="https://github.com/santiagocasas/cosmicfishpie/pulls?q=is%3Apr+reviewed-by%3ASefa76" title="Reviewed Pull Requests">👀</a> <a href="#userTesting-Sefa76" title="User Testing">📓</a> <a href="#example-Sefa76" title="Examples">💡</a> <a href="https://github.com/santiagocasas/cosmicfishpie/commits?author=Sefa76" title="Documentation">📖</a> <a href="#content-Sefa76" title="Content">🖋</a> <a href="https://github.com/santiagocasas/cosmicfishpie/issues?q=author%3ASefa76" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.santicasas.xyz"><img src="https://avatars.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="https://github.com/santiagocasas/cosmicfishpie/issues?q=author%3Asantiagocasas" title="Bug reports">🐛</a> <a href="#content-santiagocasas" title="Content">🖋</a> <a href="#data-santiagocasas" title="Data">🔣</a> <a href="https://github.com/santiagocasas/cosmicfishpie/commits?author=santiagocasas" title="Documentation">📖</a> <a href="#example-santiagocasas" title="Examples">💡</a> <a href="#ideas-santiagocasas" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-santiagocasas" title="Maintenance">🚧</a> <a href="#projectManagement-santiagocasas" title="Project Management">📆</a> <a href="#research-santiagocasas" title="Research">🔬</a> <a href="https://github.com/santiagocasas/cosmicfishpie/pulls?q=is%3Apr+reviewed-by%3Asantiagocasas" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/santiagocasas/cosmicfishpie/commits?author=santiagocasas" title="Tests">⚠️</a> <a href="#tutorial-santiagocasas" title="Tutorials">✅</a> <a href="#userTesting-santiagocasas" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/matmartinelli"><img src="https://avatars.githubusercontent.com/u/17426508?v=4?s=100" width="100px;" alt="Matteo Martinelli"/><br /><sub><b>Matteo Martinelli</b></sub></a><br /><a href="https://github.com/santiagocasas/cosmicfishpie/commits?author=matmartinelli" title="Code">💻</a> <a href="#design-matmartinelli" title="Design">🎨</a> <a href="#ideas-matmartinelli" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome.
