---
title: cosmicfishpie
html_theme.sidebar_primary.remove: true
---

```{toctree}
:hidden:

getting_started
user_guide
tutorials
api/index
development
```

# cosmicfishpie

::::{div} cfp-hero

:::{div} cfp-tagline
Fisher-matrix forecasts for galaxy clustering, weak lensing and CMB surveys,
with CAMB, CLASS or a built-in symbolic backend.
:::

```bash
pip install cosmicfishpie
```

:::{div} cfp-buttons
```{button-ref} getting_started
:ref-type: doc
:color: primary
:class: sd-rounded-pill

Get started
```

```{button-ref} api/index
:ref-type: doc
:color: secondary
:outline:
:class: sd-rounded-pill

API reference
```
:::

::::

:::::{grid} 1 2 2 3
:gutter: 3
:class-container: cfp-cards

::::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Install from PyPI or from source, and fetch the optional external survey data.
::::

::::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` A verified first forecast
:link: first-forecast
:link-type: ref

Run a compact photometric forecast end to end with the built-in `symbolic` backend.
::::

::::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` User guide
:link: user_guide
:link-type: doc

Core concepts, derivative schemes, benchmarks and backend background.
::::

::::{grid-item-card} {octicon}`telescope;1.5em;sd-mr-1` Observables and probes
:link: observables
:link-type: ref

Euclid photometric and spectroscopic clustering, weak lensing, CMB and intensity mapping.
::::

::::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API reference
:link: api/index
:link-type: doc

Modules, classes and functions generated from the package docstrings.
::::

::::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Contributing and citing
:link: development
:link-type: doc

Contribution guidelines, changelog, validation reports and citation information.
::::

:::::

## Features

- **Multiple backends.** Use CAMB, CLASS, or the built-in `symbolic` backend that needs no
  external Einstein--Boltzmann solver.
- **Surveys and observables.** Photometric (`GCph`, `WL`) and spectroscopic (`GCsp`) Euclid
  forecasts, CMB (`CMB_T`, `CMB_E`, `CMB_B`) and intensity mapping (`IM`).
- **Finite-difference derivatives.** `3PT`, `STEM`, `POLY` and `4PT_FWD` schemes, selected with
  a single option.
- **Analysis and plotting.** Marginalise, reshuffle and combine Fisher matrices, and draw
  triangle plots to compare surveys and settings.
- **Validated.** CAMB-versus-CLASS Fisher comparisons and likelihood checks are documented and
  reproducible from the repository scripts.

## Cite us

Please cite the software if you use it. The repository's
[`CITATION.cff`](https://github.com/santiagocasas/cosmicfishpie/blob/main/CITATION.cff) is the
maintained source of citation metadata; a minimal BibTeX entry is:

```bibtex
@software{cosmicfishpie,
  author = {Casas, Santiago and Martinelli, Matteo and Pamuk, Sefa and Sabarish, V. M.},
  title  = {cosmicfishpie: Fisher-matrix forecasts for cosmological surveys},
  url    = {https://github.com/santiagocasas/cosmicfishpie}
}
```
