# cosmicfishpie

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![Documentation Status](https://readthedocs.org/projects/cosmicfishpie/badge/?version=latest)](https://cosmicfishpie.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/santiagocasas/cosmicfishpie/graph/badge.svg?token=BXTVDPXPUO)](https://codecov.io/github/santiagocasas/cosmicfishpie)
[![37.50 % FAIR](https://img.shields.io/badge/FAIR_assessment-37.50_%25-red)](https://fair-checker.france-bioinformatique.fr/assessment/68da9e3cc49e421b3e2cf501)
[![CI](https://github.com/santiagocasas/cosmicfishpie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/santiagocasas/cosmicfishpie/actions/workflows/ci.yml)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/santiagocasas/cosmicfishpie)

`cosmicfishpie` is a Python package for cosmological Fisher forecasts. It is designed to help prepare and compare forecasts for surveys such as Euclid, DESI, SKAO, and CMB experiments, using flexible pipelines that combine cosmology backends, survey specs, and post-processing tools.

<div align="center">
  <img src="https://github.com/santiagocasas/cosmicfishpie/assets/6987716/1816b3b7-0920-4a2c-aafd-9c4ba1dc3e2b" width="280">
</div>

## What you can do with cosmicfishpie

- Compute Fisher matrices for multiple observables (e.g. spectroscopic/photometric galaxy probes and CMB spectra).
- Use established Einstein-Boltzmann solvers such as CAMB (and configurable backend workflows).
- Choose finite-difference strategies and numerical settings for derivatives and stability.
- Run likelihood-based Bayesian sampling with Nautilus (nested sampling / MCMC-style posterior analysis).
- Combine, reparameterize, and post-process Fisher matrices after the initial run.
- Generate publication-ready comparison and corner plots.

## Installation

`cosmicfishpie` supports Python `>=3.10,<3.13`.

### Recommended: uv workflow

Create a virtual environment and install from source:

```bash
git clone https://github.com/santiagocasas/cosmicfishpie.git
cd cosmicfishpie
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
```

For a lighter install (without the dev extras):

```bash
uv sync
```

### Alternative: mamba/conda-forge workflow

If you prefer Conda-style environments, create one with `mamba` (or `conda`) and install from source in editable mode:

```bash
mamba create -n cosmicfishpie -c conda-forge python=3.11 pip
mamba activate cosmicfishpie
git clone https://github.com/santiagocasas/cosmicfishpie.git
cd cosmicfishpie
pip install -e .
```

This setup lets you manage the environment with conda-forge while using the latest repository version.

## Quick check

After installation, run:

```bash
python -c "import cosmicfishpie; print('cosmicfishpie import OK')"
```

If you installed from source (dev setup), you can also run tests:

```bash
uv run pytest
```

## Minimal usage example

```python
import cosmicfishpie.fishermatrix.cosmicfish as cff

options = {
    "accuracy": 1,
    "feedback": 2,
    "outroot": "example_run",
    "results_dir": "results/",
    "derivatives": "3PT",
    "nonlinear": True,
    "cosmo_model": "LCDM",
    "code": "camb",
    "survey_name": "Euclid",
    "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
}

observables = ["GCsp"]
fiducial = {"Omegam": 0.32, "h": 0.67}
freepars = {"Omegam": 0.01, "h": 0.01}

fm = cff.FisherMatrix(
    fiducialpars=fiducial,
    freepars=freepars,
    options=options,
    observables=observables,
    cosmoModel=options["cosmo_model"],
    surveyName=options["survey_name"],
)

fisher_matrix = fm.compute()
```

For fuller examples and configuration details, see the documentation.

## Documentation and citation

- Docs: https://cosmicfishpie.readthedocs.io/
- Repository: https://github.com/santiagocasas/cosmicfishpie
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Nautilus sampler docs (and citation info): https://nautilus-sampler.readthedocs.io/

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
