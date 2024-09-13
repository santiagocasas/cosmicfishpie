import cosmicfishpie.analysis.fisher_plotting as cfp
import cosmicfishpie.fishermatrix.cosmicfish as cff


def test_installation():
    # These are typical options that you can pass to Cosmicfishpie
    options = {
        "accuracy": 1,
        "outroot": "test_installation_test_EBS",
        "results_dir": "results/",
        "derivatives": "3PT",
        "nonlinear": True,
        "feedback": 2,
        "specs_dir": "survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": "camb",
        "class_config_yaml": "../boltzmann_yaml_files/camb/default.yaml",
    }

    # Internally CosmicFish converts these parameters to the corresponding parameters in CAMB or CLASS
    fiducial = {
        "Omegam": 0.32,
        "h": 0.67,
    }

    # Parameters to be varied and analyzed and their percentage variation for numerical derivatives
    freepars = {
        "Omegam": 0.01,
        "h": 0.01,
    }

    observables = ["GCsp"]

    cosmoFM = cff.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )

    FA = cosmoFM.compute()

    plot_options = {
        "fishers_list": [FA],
        "fish_labels": ["Euclid Spectroscopic pessimistic"],
        "plot_pars": list(freepars.keys()),
        "axis_custom_factors": {
            "all": 7
        },  ## Axis limits cover 3-sigma bounds of first Fisher matrix
        "plot_method": "Gaussian",
        "file_format": ".pdf",  ##file format for all the plots
        "outpath": "./plots/",  ## directory where to store the files, if non-existent, it will be created
        "outroot": "test_installation_test_plot",  ## file name root for all the plots, extra names can be added individually
        "colors": ["#000000"],
    }

    fish_plotter = cfp.fisher_plotting(**plot_options)
    fish_plotter.plot_fisher(filled=[False])
