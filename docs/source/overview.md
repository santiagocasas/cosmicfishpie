# Cosmicfishpie

## A Comprehensive Tool for Fisher Forecasts in Cosmology

### Introduction

`Cosmcfishpie` is a cosmological code to help researchers prepare for upcoming missions like Euclid, DESI, or SKAO. It contains a vast tool set to  produce Fisher forecasts for different applications. This code offers you easy access to cosmological quantities from leading Einstein-Boltzmann solvers `Class` and `CAMB` or the ability to directly read precomputed cosmologies from files.

`Cosmicfishpie` allows the user to compute different cosmological observables like 3D power spectra of galaxies, angular power spectra of the cosmic microwave background, galaxy clustering or cosmic shear measurements. The code provides flexibility with different finite distance differentiation schemes for calculating the Fisher information matrix.

Additionally, `Cosmicfishpie` excels in analysis and visualization. For this users can fix parameters in retrospect, combine information from independent experiments, and resample the results in new parameter bases. The code also allows comparisons of constraints across different surveys or settings, generating high-quality plots to effectively communicate findings.

Whether you're preparing for future cosmological surveys or analyzing existing data, Cosmicfish offers a comprehensive and user-friendly approach to cosmological forecasting. In the following we will present a quick users guide to some of these functionalities.

### Getting Started

To install `cosmicfishpie` please refer to our [installation page](https://cosmicfishpie.readthedocs.io/en/latest/installation.html)
You can install the code using `pip`

~~~
$ pip install cosmicfishpie
~~~

Any additional requirements will be installed and can be found in the requirements files.

You can test your installation by running the `pytest` pipeline. For this please make sure that you have pytest installed and then simply run it in your console

~~~
$ pytest
~~~

This should compute a two-parameter Fisher and create a corner plot that you can check out in your results or plots folder.


