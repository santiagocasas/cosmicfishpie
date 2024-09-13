# Cosmicfishpie

## A Comprehensive Tool for Fisher Forecasts in Cosmology

### Introduction

`Cosmcfishpie` is a cosmological code to help researchers prepare for upcoming missions like Euclid, DESI, or SKAO. It contains a vast tool set to  produce Fisher forecasts for different applications. This code offers you easy access to cosmological quantities from leading Einstein-Boltzmann solvers `Class` and `CAMB` or the ability to directly read precomputed cosmologies from files.

`Cosmicfishpie` allows the user to compute different cosmological observables like 3D power spectra of galaxies, angular power spectra of the cosmic microwave background, galaxy clustering or cosmic shear measurements. The code provides flexibility with different finite distance differentiation schemes for calculating the Fisher information matrix.

Additionally, `Cosmicfishpie` excels in analysis and visualization. For this users can fix parameters in retrospect, combine information from independent experiments, and resample the results in new parameter bases. The code also allows comparisons of constraints across different surveys or settings, generating high-quality plots to effectively communicate findings.

Whether you're preparing for future cosmological surveys or analyzing existing data, Cosmicfish offers a comprehensive and user-friendly approach to cosmological forecasting. In the following we will present a quick users guide to some of these functionalities.

### Getting Started

#### Installing Cosmicfishpie
To install `cosmicfishpie` please refer to our [installation page](https://cosmicfishpie.readthedocs.io/en/latest/installation.html)
You can install the code using `pip`

~~~shell
$ pip install cosmicfishpie
~~~

Any additional requirements will be installed and can be found in the requirements files.

To run your first tests you should install a Einstein-Boltzmann solver. For example you can install `CAMB` using pip. The next section will assume that you did so. If you choose to use `Class` please mind that you might need to change some things.
~~~shell
$ pip install camb
~~~

You can test your installation by running the `pytest` pipeline. For this please make sure that you have pytest installed and then simply run it in your console

~~~shell
$ pytest
~~~

This should compute a two-parameter Fisher and create a corner plot that you can check out in your results or plots folder.

Now that you have `Cosmicfishpie` installed and running, you can dive deeper into its functionality:

#### Create your first Fisher matrix

To compute a fisher matrix you first have to specify your model and your experiment. A typical script of `Cosmicfishpie` starts like this

~~~python
options = {
    'accuracy': 1, # Internal Boost to CF numerical accuracy
    'feedback': 2, # Verbosity of CF outputs
    'outroot': 'Euclid_Pessimistic', # The name of results will contain this String
    'results_dir': 'results/', # Path to where the results will be saved
    'derivatives': '3PT', # Method to compute finite differentiation

    'nonlinear': True, # If the nonlinear power spectrum should be computed
    'cosmo_model' : 'LCDM', # Cosmological model used
    'code': 'camb', # Code to be used
    'class_config_yaml':'/boltzmann_yaml_files/camb/default.yaml', # Settings for the Einstein Boltzmann solver

    'survey_name': 'Euclid', # Survey Name
    'specs_dir' : 'survey_specifications/', # Path of survey specification
    'survey_name_spectro': 'Euclid-Spectroscopic-ISTF-Pessimistic', # Survey specification for the Euclid spectroscopic observables
    'survey_name_photo': 'Euclid-Photometric-ISTF-Pessimistic', # Survey specification for the Euclid photometric observables
}

#Observables that should be considered for the forecast.
observables = ['GCsp']

#The Cosmicfishpie parameter basis is internally converted to CAMB or CLASS basis
fiducial = {"Omegam":0.32,
            "h":0.67,
}

#Parameters to be varied and the relative step size for numerical derivatives
freepars = {'Omegam': 0.01,
            'h': 0.01,
}

~~~

The first dictionary contains the basic options that are needed to run compute the Fisher information. The first set of options are specific to the output of `Cosmicfishpie` and the internal numerics. The second block specifies the cosmology model and options for the Einstein-Boltzmann solver. Here we use `CAMB`but you can also use `Class` if you have it installed. Finally, the third block specifies the survey. You can look at additional options in the documentation of the configurations module.

With observables you can choose which observables and which combination of observables should the forecast done for.
With the next dictionary you can choose the fiducial cosmology parameters. Other parameters are set to default values within the code.

The last dictionary specifies the relative step sizes to compute the Fisher matrix. When computing the Fisher these will be the cosmological parameters present.
We can pass this directly to the main `cosmicfishpie` module:

~~~python
import cosmicfishpie.fishermatrix.cosmicfish as cff

cosmoFM = cff.FisherMatrix(
    fiducialpars=fiducial,
    freepars=freepars,
    options=options,
    observables=observables,
    cosmoModel=options['cosmo_model'],
    surveyName=options['survey_name'])

FA = cosmoFM.compute()
~~~

We can pass all parameters to the constructor of the `FisherMatrix`. You then just have to call `compute` and see your numerical derivatives getting computed. The results, a list of varied parameters and nuisance parameter and a summery of all options passed to CF and the EBS. Enjoy the simplicity!

#### Create your first plots

Creating a high-quality plot is very easy with `Cosmicfishpie`. We can use the `fisher_ploting` module and pass to it similarly a set of options for the plot. Additional options can be found in the documentation of the module.

~~~python
import cosmicfishpie.analysis.fisher_plotting as cfp

plot_options = {
    'fishers_list': [FA],
    'fish_labels': ['Euclid Spectroscopic pessimistic'],
    'plot_pars': list(freepars.keys()),
    'axis_custom_factors': {'all':7}, # Axis limits cover 3-sigma bounds of first Fisher matrix
    'plot_method': 'Gaussian',
    'file_format': '.pdf', # file format for all the plots
    'outpath' : './plots/', # directory where to store the files, if non-existent, it will be created
    'outroot':'test_installation_test_plot', # file name root for all the plots, extra names can be added individually
    'colors':["#000000"],
}

fish_plotter = cfp.fisher_plotting(**plot_options)
fish_plotter.plot_fisher(filled=[False])
~~~

This creates a nice looking triangle plot of the two parameters that were varied. The `fisher_plotting` class has additional plotting capabilities, for example comparing different constraints, can be found in the documentation as well.