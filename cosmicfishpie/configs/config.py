# This class defines global variables that will not change through
# the computation of one fisher matrix

import glob
import os
from copy import deepcopy
from time import time

import numpy as np
import yaml

import cosmicfishpie.cosmology.cosmology as cosmology
from cosmicfishpie.utilities.utils import misc as ums
from cosmicfishpie.utilities.utils import physmath as upm
from cosmicfishpie.utilities.utils import printing as upt


def init(
    options=dict(),
    specifications=dict(),
    observables=None,
    freepars=None,
    extfiles=None,
    fiducialpars=None,
    photobiaspars=None,
    photopars=None,
    IApars=None,
    PShotpars=None,
    spectrobiaspars=None,
    spectrononlinearpars=None,
    IMbiaspars=None,
    surveyName="Euclid",
    cosmoModel="w0waCDM",
    latexnames=None,
):
    """This class is to handle the configuration for the fishermatrix computation as well as the fiducial parameters. It then gives access to all global variables

    Parameters
    ----------
    options              : dict, optional
                           A dictionary that contains the global options for the calculation of the fishermatrix. A list of all possible keys are found below
    specifications       : dict, optional
                           A dictionary containing the survey specifications. Defaults to the specifications in the `.yaml` of the survey specifications
    observables          : list, optional
                           A list of strings for the different observables
    freepars             : dict, optional
                           A dictionary containing all cosmological parameters to be varied and their corresponding rel. step sizes
    extfiles             : dict, optional
                           A dictionary containing the path to the external files as well as how all the names of the files in the folder correspond to the cosmological quantities, units etc.
    fiducialpars         : dict, optional
                           A dictionary containing the fiducial cosmological parameters
    photobiaspars             : dict, optional
                           A dictionary containing the specifications for the galaxy biases of the photometric probe
    photopars            : dict, optional
                           A dictionary containing specifications for the window function's galaxy distribution
    IApars               : dict, optional
                           A dictionary containing the specifications for the intrinsic alignment effect in cosmic shear
    PShotpars            : dict, optional
                           A dictionary containing the values of additional shotnoise of the spectroscopic probes
    spectrobiaspars      : dict, optional
                           A dictionary containing the specifications for the galaxy biases of the spectroscopic probe
    spectrononlinearpars : dict, optional
                           A dictionary containing the values of the non linear modeling parameters of the spectroscopic probe
    IMbiaspars           : dict, optional
                           A dictionary containing the specifications for the galaxy biases of the spectroscopic intensity mapping probe
    surveyName           : str, optional
                           String of the name of the survey for which the forecast is done. Defaults to Euclid with optimistic specifications
    cosmoModel           : str, optional
                           A string of the name of the cosmological model used in the calculations. Defaults to flat "w0waCDM" cosmology
    latexnames           : dict, optional
                           A dictionary that contains the Latex names of the cosmological parameters

    Options
    -------
    camb_path                   : str
                                  Path to camb. Defaults to the camb in your current environment
    specs_dir                   : str
                                  Path to the survey specifications. Defaults to the `specs_dir_default`
    specs_dir_default           : str
                                  Path to the default survey specifications. Defaults to the `survey_specifications` folder in the config directory of cosmicfishpie
    survey_name                 : str
                                  String of the names of the survey. Defaults to the name passed in the parameter `surveyName`
    survey_name_photo           : str
                                  Name of the survey specifications file for a photometric probe
    survey_name_spectro         : str
                                  Name of the survey specifications file for a spectrocopic probe
    survey_name_radio_IM        : str
                                  Name of the survey specifications file for a line intensity mapping probe
    derivatives                 : str
                                  String of the name of the derivative method. Either `3PT`, `4PT_FWD`, `STEM` or `POLY`. Defaults to `3PT`
    nonlinear                   : bool
                                  If True will do nonlinear corrections in the computation of the different observables. Defaults to True
    nonlinear_photo             : bool
                                  If True will use the nonlinear power spectrum when calculation the angular power spectrum of the photometric probe. Defaults to True
    bfs8terms                   : bool
                                  If True will expand the observed power spectrum with :math:`\\sigma_8` to match the IST:F recipe. Defaults to True
    vary_bias_str               : str
                                  The root of the name of the bias parameters that should be varied in the spectroscopic probe. Defaults to 'lnb'
    AP_effect                   : bool
                                  If True the Alcock-Paczynsk effect will be considered in the spectroscopic probe. Defaults to True
    FoG_switch                  : bool
                                  If True the finger of god effect will be modelled in the observed power spectrum of the spectroscopic probe. Defaults to True
    GCsp_linear                 : bool
                                  If True there will be no nonlinear modelling used in the spectroscopic probe. Defaults to False
    fix_cosmo_nl                : bool
                                  If True and the nonlinear modeling parameters are not varied, then they will be fixed to the values computed in the fiducial cosmology. Else they will be recomputed in each sample cosmology. Defaults to True
    Pshot_nuisance_fiducial     : float
                                  Value of the fiducial additional shotnoise of the spectroscopic probe. Defaults to 0.0
    pivot_z_IA                  : float
                                  Redshift on which the power law dependance in the eNLA model of intrinsic alignment should be normalized to. Defaults to 0.0
    accuracy                    : int
                                  Global rescaling of the amount of points that are used in internal calculations or interpolations for the probes. Defaults to 1
    feedback                    : int
                                  Number indicating the verbosity of the output. Higher numbers generally mean more output. Defaults to 2
    activateMG                  : bool
                                  If True will also consider modified gravity in the calculations of the observables. Defaults to False
    external_activateMG         : bool
                                  If True while reading the external Files will also look for files specific to modified gravity models
    cosmo_model                 : str
                                  A string of the name of the cosmological model used in the calculations. Defaults to what was passed in the parameter `cosmoModel`
    outroot                     : str
                                  The name of the output files are always starting with CosmicFish_Version number_outroot. Defaults to 'default_run'
    code                        : str
                                  String of the method to obtain the cosmological functions such as the power spectrum. Either 'camb', 'class' or 'external'. Defaults to 'camb'
    memorize_cosmo              : bool
                                  If True will save and load cosmologies that have already been computed in the cache. Defaults to False
    results_dir                 : str
                                  Name of the folder in which the results should be saved. If the folder does not exist will create a new one. Defaults to './results'
    boltzmann_yaml_path         : str
                                  Path to the configurations for the Einstein-Boltzmann solvers. Defaults to the `boltzmann_yaml_files` folder in the home directory of cosmicfishpie
    class_config_yaml           : str
                                  Path to the configurations for class. Defaults to 'boltzmann_yaml_path/class/default.yaml'
    camb_config_yaml            : str
                                  Path to the configurations for camb. Defaults to 'boltzmann_yaml_path/camb/default.yaml'
    fishermatrix_file_extension : str
                                  Specifies in what kind of file the result Fisher matrix should be saved. Defaults to '.txt'
    savgol_polyorder            : float
                                  Order of the Savitzky-Golay filter. Defaults to 3 matching the IST:F recipe
    savgol_width                : float
                                  Width of the Savitzky-Golay filter. Defaults to ~1.359
    savgol_internalsamples      : float
                                  How many points on a logarithmic k axis should be taken to apply the Savitzky-Golay filter to. Defaults to 800
    savgol_internalkmin         : float
                                  Lowest wavenumber that should be used when calculating when applying the Savitzky-Golay filter. Defaults to 1e-3. Together with the other defaults this is matching the IST:F recipe
    eps_cosmopars               : float
                                  The default rel. step size of the cosmological parameters if none have been passed. Defaults to 1e-2
    eps_gal_nuispars            : float
                                   The default rel. step size of the bias parameters if none have been passed. Defaults to 1e-4
    GCsp_Tracer                 : str
                                  What power spectrum should be used as the underlying power spectrum the spectroscopic probe's galaxy clustering traces. Either 'matter' for the total matter spectrum or 'clustering' for CDM+baryons. Defaults to 'matter'
    GCph_Tracer                 : str
                                  What power spectrum should be used as the underlying power spectrum the photometric probe's galaxy clustering traces. Either 'matter' for the total matter spectrum or 'clustering' for CDM+baryons. Defaults to 'matter'
    ShareDeltaNeff              : bool
                                  If True, the variation of the cosmological parameter Neff is understood as a rescaling of the decoupling temperature of neutrinos. If False any additional Neff is accounted for as additional massless relics.
    kh_rescaling_bug            : bool
                                  If true, the internal wavenumber in the computation of the spectroscopic probe's power spectrum computation will be rescaled by an adtional factor h/hfid. Default False
    kh_rescaling_beforespecerr_bug : bool
                                     If True, the internal scales in the computation of the spectroscopic probe's resolution error will be rescaled by an adtional factor h/hfid. Default False

    Attributes
    ----------
    settings               : dict, global
                             A dictionary containing all the global options passed as well as the default values of the ones not passed
    external               : dict, global
                             A dictionary containing all paths to the external files, how all the names of the files in the folder correspond to the cosmological quantities, the units etc. Will be None if no external files are given
    input_type             : str, global
                             String of the method to obtain the cosmological functions such as the power spectrum. Either 'camb', 'class' or 'external'
    specs                  : dict, global
                             A dictionary containing the survey specifications
    boltzmann_classpars    : dict, global
                             A dictionary containing the configuration, precision parameters, and fixed cosmological parameters for class
    boltzmann_cambpars     : dict, global
                             A dictionary containing the configuration, precision parameters, and fixed cosmological parameters for camb
    survey_equivalence     : callable, global
                             Part of the Parser, will correspond the passed survey to the name of a specifications file
    obs                    : list, global
                             A list of strings for the different observables
    freeparams             : dict, global
                             A dictionary containing all names and the corresponding rel. step size for all parameters
    fiducialparams         : dict, global
                             A dictionary containing all fiducial values for the cosmological parameters
    fiducialcosmo          : cosmicfishpie.cosmology.cosmology.cosmo_functions, global
                             An instance of `cosmo_functions` of the fiducial cosmology, this contains all the cosmological functions and quantities computed from them
    biasparams             : dict, global
                             a dictionary containing the specifications for the galaxy biases of the photometric probe
    photoparams            : dict, global
                             A dictionary containing specifications for the window function's galaxy distribution of the photometric probe
    IAparams               : dict, global
                             A dictionary containing the specifications for the intrinsic alignment effect in cosmic shear of the photometric probe
    PShotparams            : dict, global
                             A dictionary containing the values of the additional shot noise per bin dictionary containing the values of the additional shot noise per bin for the spectroscopic probe
    Spectrobiasparams      : dict, global
                             A dictionary containing the specifications for the galaxy biases of the spectroscopic probe
    Spectrononlinearparams : dict, global
                             A dictionary containing the values of the non linear modeling parameters entering FOG and the dewiggling weight per bin for the spectroscopic probe
    IMbiasparams           : dict, global
                             A dictionary containing the specifications for the line intensity biases of the spectroscopic probe
    latex_names            : dict, global
                             A dictionary with all cosmological + nuisance parameters and their corresponding name for the LaTeX labels.
    """
    global settings
    settings = options
    # Set defaults if not contained previously in options
    settings.setdefault(
        "specs_dir_default",
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "default_survey_specifications",
        ),
    )
    settings.setdefault("specs_dir", settings["specs_dir_default"])
    settings.setdefault("external_data_dir", os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                                          "external_data"))
    settings.setdefault("survey_name", surveyName)
    settings.setdefault("survey_specs", "ISTF-Optimistic")
    settings.setdefault("survey_name_photo", "Euclid-Photometric-ISTF-Pessimistic")
    settings.setdefault("survey_name_spectro", "Euclid-Spectroscopic-ISTF-Pessimistic")
    # settings.setdefault("survey_name_radio_photo", "SKA1-Photometric-Redbook-Optimistic")
    # settings.setdefault("survey_name_radio_spectro", "SKA1-Spectroscopic-Redbook-Optimistic")
    settings.setdefault("survey_name_radio_IM", "SKA1-IM-Redbook-Optimistic")
    settings.setdefault("fail_on_specs_not_found", False)
    settings.setdefault("derivatives", "3PT")
    settings.setdefault("nonlinear", True)
    settings.setdefault("nonlinear_photo", True)
    settings.setdefault("bfs8terms", False)
    settings.setdefault("vary_bias_str", "lnb")
    settings.setdefault("AP_effect", True)
    settings.setdefault("FoG_switch", True)
    settings.setdefault("GCsp_linear", False)
    settings.setdefault("fix_cosmo_nl_terms", True)
    settings.setdefault("Pshot_nuisance_fiducial", 0.0)
    settings.setdefault("pivot_z_IA", 0.0)
    settings.setdefault("accuracy", 1)
    settings.setdefault("spectro_Pk_k_samples", 1025)
    settings.setdefault("spectro_Pk_mu_samples", 17)
    settings.setdefault("feedback", 2)
    settings.setdefault("activateMG", False)
    settings.setdefault("external_activateMG", False)
    settings.setdefault("cosmo_model", cosmoModel)
    settings.setdefault("outroot", "default_run")
    settings.setdefault("code", "camb")
    settings.setdefault("memorize_cosmo", False)
    settings.setdefault("results_dir", "./results")
    settings.setdefault("SUPPRESS_WARNINGS", True)
    settings.setdefault(
        "boltzmann_yaml_path",
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "default_boltzmann_yaml_files"),
    )
    settings.setdefault(
        "class_config_yaml", os.path.join(settings["boltzmann_yaml_path"], "class", "default.yaml")
    )
    settings.setdefault(
        "camb_config_yaml", os.path.join(settings["boltzmann_yaml_path"], "camb", "default.yaml")
    )
    settings.setdefault(
        "symbolic_config_yaml",
        os.path.join(settings["boltzmann_yaml_path"], "symbolic", "default.yaml"),
    )
    settings.setdefault("fishermatrix_file_extension", ".txt")
    settings.setdefault("savgol_window", 101)
    settings.setdefault("savgol_polyorder", 3)
    settings.setdefault("savgol_width", 1.358528901113328)
    settings.setdefault("savgol_internalsamples", 800)
    settings.setdefault("savgol_internalkmin", 0.001)
    settings.setdefault("eps_cosmopars", 0.01)
    settings.setdefault("eps_gal_nuispars", 0.0001)
    settings.setdefault("eps_gal_nonlinpars", 0.01)
    settings.setdefault("GCsp_Tracer", "matter")
    settings.setdefault("GCph_Tracer", "matter")
    settings.setdefault("ell_sampling", "accuracy")
    settings.setdefault("ShareDeltaNeff", False)
    settings.setdefault("kh_rescaling_bug", False)
    settings.setdefault("kh_rescaling_beforespecerr_bug", False)

    feed_lvl = settings["feedback"]

    global external
    global input_type
    if extfiles is not None and settings["code"] == "external":
        input_type = settings["code"]
        extfiles_default = {
            "file_names": {
                "H_z": "background_Hz",
                "D_zk": "D_Growth-zk",
                "f_zk": "f_GrowthRate-zk",
                "fcb_zk": "fcb_GrowthRate-zk",
                "Pl_zk": "Plin-zk",
                "Plcb_zk": "Plincb-zk",
                "Pnl_zk": "Pnonlin-zk",
                "Pnlcb_zk": "Pnonlincb-zk",
                "s8_z": "sigma8-z",
                "s8cb_z": "sigma8cb-z",
                "SigmaWL": None,  # Default name: 'sigmaWL-zk'
                "z_arr": "z_values_list",
                "k_arr": "k_values_list",
                "k_arr_special": None,
            },
            "E-00": True,
            "fiducial_folder": "fiducial_eps_0",
            "folder_paramnames": [],
        }
        external = extfiles_default.copy()
        ums.deepupdate(external, extfiles)

        if os.path.isdir(external["directory"]):
            ff = external["fiducial_folder"]
            dii = external["directory"]
            upt.time_print(
                feedback_level=feed_lvl,
                min_level=0,
                text=f"-> Using input files for cosmology observables: {dii}",
            )
            fidudir = glob.glob(os.path.join(external["directory"], ff + "*"))
            lendir = len(fidudir)
            if lendir < 1:
                raise ValueError("External directory does not contain fiducial folder ")
            for dd in external["folder_paramnames"]:
                subdirs = glob.glob(os.path.join(external["directory"], dd + "*"))
                lensub = len(subdirs)
                if lensub < 1:
                    raise ValueError(
                        "External directory does not contain appropriate subfolders for:  {:s} ".format(
                            dd
                        )
                    )
                else:
                    upt.time_print(
                        feedback_level=feed_lvl,
                        min_level=1,
                        text=f"-> {lensub} folders for parameter {dd}",
                    )
        else:
            raise ValueError("External directory does not exist")

    elif settings["code"] == "class":
        input_type = settings["code"]
        global boltzmann_classpars
        boltzmann_yaml_file = open(settings["class_config_yaml"])
        parsed_boltzmann = yaml.load(boltzmann_yaml_file, Loader=yaml.FullLoader)
        boltzmann_classpars = parsed_boltzmann
        external = None
    elif settings["code"] == "camb":
        if "camb_path" not in settings:
            import camb

            cambpath = os.path.dirname(camb.__file__)
            settings["camb_path"] = cambpath
        input_type = settings["code"]
        global boltzmann_cambpars
        boltzmann_yaml_file = open(settings["camb_config_yaml"])
        parsed_boltzmann = yaml.load(boltzmann_yaml_file, Loader=yaml.FullLoader)
        boltzmann_cambpars = parsed_boltzmann
        external = None
    elif settings["code"] == "symbolic":
        input_type = settings["code"]
        global boltzmann_symbolicpars
        boltzmann_yaml_file = open(settings["symbolic_config_yaml"])
        parsed_boltzmann = yaml.load(boltzmann_yaml_file, Loader=yaml.FullLoader)
        boltzmann_symbolicpars = parsed_boltzmann
        external = None
    else:
        print("No external input files used in this calculation.")
        print("No Einstein-Boltzmann-Solver (EBS) specified.")
        print("No symbolic_pofk specified.")
        # settings['code'] = 'camb'
        external = None
        raise ValueError("No cosmology calculator specified")

    def ngal_per_bin(ngal_sqarmin, zbins):
        # compute num galaxies per bin for whole sky area
        nbins = len(zbins[:-1])
        ones = np.ones_like(zbins[:-1])
        ngal_sqdeg = ngal_sqarmin * 3600
        ngal_bin = (ngal_sqdeg / nbins) * upm.sr
        numgal = ngal_bin * ones
        return numgal

    ##############################
    # Load Survey Specifications #
    ##############################

    # Add additional surveys here
    available_survey_names = ["Euclid", "SKAO", "DESI", "Planck", "Rubin"]
    available_survey_names = ["Euclid", "SKAO", "DESI", "Planck", "Rubin"]

    def create_ph_dict(foldername, filename):
        photo_dict = dict()
        if not filename:
            upt.time_print(
                feedback_level=feed_lvl,
                min_level=1,
                text="-> No photo survey passed, returning empty dict",
            )
            return photo_dict
        try:
            ph_file_path = os.path.join(foldername, filename + ".yaml")
            if not os.path.isfile(ph_file_path):
                raise FileNotFoundError(f"specifications file : {ph_file_path} not found!")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            if settings["fail_on_specs_not_found"]:
                raise FileNotFoundError(
                    f"specifications file : {ph_file_path} not found! Exiting..."
                )
            else:
                ph_file_path = os.path.join(
                    settings["specs_dir_default"], specs_default_photo + ".yaml"
                )
                print(f"Using default specifications for photo: {ph_file_path}")

        ph_yaml_fs = open(ph_file_path, "r")
        ph_yaml_content = yaml.load(ph_yaml_fs, Loader=yaml.FullLoader)
        ph_yaml_fs.close()

        photo_dict = ph_yaml_content["specifications"]
        # Load WL bins and num galaxies per bin make it backwards compatible
        z_bins_WL = photo_dict.get("z_bins_WL", photo_dict.get("z_bins_ph"))
        photo_dict["z_bins_WL"] = np.array(z_bins_WL)
        photo_dict["ngal_sqarmin_WL"] = photo_dict.get(
            "ngal_sqarmin_WL", photo_dict.get("ngal_sqarmin")
        )
        photo_dict["ngalbin_WL"] = ngal_per_bin(
            photo_dict["ngal_sqarmin_WL"], photo_dict["z_bins_WL"]
        )
        photo_dict["binrange_WL"] = range(1, len(photo_dict["z_bins_WL"]))
        # Load GCph bins and num galaxies per bin make it backwards compatible
        photo_dict["z_bins_GCph"] = photo_dict.get("z_bins_GCph", photo_dict.get("z_bins_ph"))
        photo_dict["ngal_sqarmin_GCph"] = photo_dict.get(
            "ngal_sqarmin_GCph", photo_dict.get("ngal_sqarmin")
        )
        photo_dict["ngalbin_GCph"] = ngal_per_bin(
            photo_dict["ngal_sqarmin_GCph"], photo_dict["z_bins_GCph"]
        )
        photo_dict["binrange_GCph"] = range(1, len(photo_dict["z_bins_GCph"]))
        # Load theoretical n(z) parameters
        photo_dict["z0"] = photo_dict["zm"] / np.sqrt(2)
        photo_dict["z0_p"] = photo_dict["z0"]

        return photo_dict

    def create_sp_dict(foldername, filename, type="spectro"):
        spec_dict = dict()
        if not filename:
            upt.time_print(
                feedback_level=feed_lvl,
                min_level=1,
                text=f"-> No {type} survey passed, returning empty dict",
            )
            return spec_dict
        try:
            sp_file_path = os.path.join(foldername, filename + ".yaml")
            if not os.path.isfile(sp_file_path):
                raise FileNotFoundError(f"specifications file : {sp_file_path} not found!")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            if settings["fail_on_specs_not_found"]:
                raise FileNotFoundError(
                    f"specifications file : {sp_file_path} not found! Exiting..."
                )
            else:
                sp_file_path = os.path.join(
                    settings["specs_dir_default"], specs_default_spectro + ".yaml"
                )
                print(f"Using default specifications for spectroscopic: {sp_file_path}")

        sp_yaml_fs = open(sp_file_path, "r")
        sp_yaml_content = yaml.load(sp_yaml_fs, Loader=yaml.FullLoader)
        sp_yaml_fs.close()
        spec_dict = sp_yaml_content["specifications"]
        return spec_dict

    # Load the default Euclid cases
    specs_defaults = {}
    specs_default_spectro = "Euclid-Spectroscopic-ISTF-Pessimistic"
    specs_default_photo = "Euclid-Photometric-ISTF-Pessimistic"
    specs_defaults.update(create_ph_dict(settings["specs_dir_default"], specs_default_photo))
    specs_defaults.update(create_sp_dict(settings["specs_dir_default"], specs_default_spectro))

    global specs
    specs = specs_defaults.copy()  # Start with default dict
    spectroTaken = False
    photoTaken = False
    specificationsf = dict()
    spectroTaken = False
    photoTaken = False
    specificationsf = dict()

    if "Euclid" in surveyName:
        surveyNameSpectro = settings.get("survey_name_spectro")
        if surveyNameSpectro:
            specificationsf1 = create_sp_dict(settings["specs_dir"], surveyNameSpectro)
            specificationsf.update(specificationsf1)
            spectroTaken = True
            upt.time_print(
                feedback_level=feed_lvl, min_level=1, text=f"-> Survey loaded:  {surveyNameSpectro}"
            )
        surveyNamePhoto = settings.get("survey_name_photo")
        if surveyNamePhoto:
            specificationsf2 = create_ph_dict(settings["specs_dir"], surveyNamePhoto)
            specificationsf.update(specificationsf2)
            photoTaken = True
            upt.time_print(
                feedback_level=feed_lvl, min_level=1, text=f"-> Survey loaded:  {surveyNamePhoto}"
            )
    if "SKA" in surveyName:
        surveyNameRadioIM = settings.get("survey_name_radio_IM")
        specificationsf3 = create_sp_dict(settings["specs_dir"], surveyNameRadioIM, type="IM")
        specificationsf.update(specificationsf3)
        upt.time_print(
            feedback_level=feed_lvl, min_level=1, text=f"-> Survey loaded:  {surveyNameRadioIM}"
        )
        if not spectroTaken:
            surveyNameSpectro = settings.get("survey_name_spectro")
            specificationsf4 = create_sp_dict(settings["specs_dir"], surveyNameSpectro)
            specificationsf.update(specificationsf4)
            spectroTaken = True
            upt.time_print(
                feedback_level=feed_lvl, min_level=1, text=f"-> Survey loaded:  {surveyNameSpectro}"
            )
        if not photoTaken:
            surveyNamePhoto = settings.get("survey_name_photo")
            specificationsf5 = create_ph_dict(settings["specs_dir"], surveyNamePhoto)
            specificationsf.update(specificationsf5)
            photoTaken = True
            upt.time_print(
                feedback_level=feed_lvl, min_level=1, text=f"-> Survey loaded:  {surveyNamePhoto}"
            )
    if "Rubin" in surveyName:
        if not photoTaken:
            surveyNamePhoto = settings.get("survey_name_photo")
            specificationsf6 = create_ph_dict(settings["specs_dir"], surveyNamePhoto)
            specificationsf.update(specificationsf6)
            photoTaken = True
            upt.time_print(
                feedback_level=feed_lvl, min_level=1, text=f"-> Survey loaded:  {surveyNamePhoto}"
            )
    if "DESI" in surveyName:
        if not spectroTaken:
            surveyNameSpectro = settings.get("survey_name_spectro")
            specificationsf7 = create_sp_dict(settings["specs_dir"], surveyNameSpectro)
            specificationsf.update(specificationsf7)
            spectroTaken = True
            upt.time_print(
                feedback_level=feed_lvl, min_level=1, text=f"-> Survey loaded:  {surveyNameSpectro}"
            )
    if "Planck" in surveyName:
        yaml_file = open(os.path.join(settings["specs_dir"], "Planck.yaml"))
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        specificationsfPlanck = parsed_yaml_file["specifications"]
        specificationsf.update(specificationsfPlanck)
        upt.time_print(feedback_level=feed_lvl, min_level=1, text="-> Survey loaded:  Planck")
    if surveyName not in available_survey_names:
        print("Survey name passed: ", surveyName)
        print(
            "Survey name not found in available survey names.",
            "Please pass your full custom specifications as a dictionary.",
        )
    upt.debug_print("Files specifications: ", specificationsf)
    upt.debug_print("Default specifications: ", specs)
    # ums.deepupdate(specs, specificationsf)  # deep update keys if present in files
    # no more deepupdate because it causes problems with duplicated keys in bias parameters when using different bias samples
    specs.update(specificationsf)
    upt.debug_print("Updated specifications: ", specs)
    upt.debug_print("Files specifications: ", specificationsf)
    upt.debug_print("Default specifications: ", specs)
    # ums.deepupdate(specs, specificationsf)  # deep update keys if present in files
    # no more deepupdate because it causes problems with duplicated keys in bias parameters when using different bias samples
    specs.update(specificationsf)
    upt.debug_print("Updated specifications: ", specs)
    specs["fsky_GCph"] = specificationsf.get(
        "fsky_GCph", upm.sqdegtofsky(specificationsf.get("area_survey_GCph", 0.0))
    )
    specs["fsky_WL"] = specificationsf.get(
        "fsky_WL", upm.sqdegtofsky(specificationsf.get("area_survey_WL", 0.0))
    )
    specs["fsky_spectro"] = specificationsf.get(
        "fsky_spectro", upm.sqdegtofsky(specificationsf.get("area_survey_spectro", 0.0))
    )
    specs["fsky_IM"] = specificationsf.get(
        "fsky_IM", upm.sqdegtofsky(specificationsf.get("area_survey_IM", 0.0))
    )
    # ums.deepupdate(specs, specifications)  # deep update keys if passed by users
    specs.update(specifications)
    specs["survey_name"] = surveyName
    specs["specs_dir"] = settings["specs_dir"]  # Path for additional files like luminosity

    if observables is None:
        observables = ["GCph", "WL"]
    global obs
    obs = observables

    global freeparams
    if freepars is None:
        eps_cp = settings["eps_cosmopars"]
        if settings["cosmo_model"] == "w0waCDM":
            freepars = {
                "Omegam": eps_cp,
                "Omegab": eps_cp,
                "w0": eps_cp,
                "wa": eps_cp,
                "h": eps_cp,
                "ns": eps_cp,
                "sigma8": eps_cp,
            }
        elif settings["cosmo_model"] == "LCDM":
            freepars = {
                "Omegam": eps_cp,
                "Omegab": eps_cp,
                "h": eps_cp,
                "ns": eps_cp,
                "sigma8": eps_cp,
            }
        else:
            print(
                "Other cosmological models not implemented yet.",
                "Please pass your free parameters and their respective epsilons as a dictionary.",
            )
    freeparams = deepcopy(freepars)

    global fiducialparams
    if fiducialpars is None:
        fiducialpars = {
            "Omegam": 0.32,
            "Omegab": 0.05,
            "w0": -1.0,
            "wa": 0.0,
            "h": 0.67,
            "ns": 0.96,
            "sigma8": 0.815583,
            "Omegak": 0.0,
            "mnu": 0.06,
            "tau": 0.058,
            "num_nu_massive": 1,
            "num_nu_massless": 2.046,
            "dark_energy_model": "ppf",
        }
    else:
        upt.time_print(
            feedback_level=feed_lvl, min_level=2, text="-> Custom fiducial parameters loaded"
        )
    fiducialparams = deepcopy(fiducialpars)

    global fiducialcosmo
    upt.time_print(
        feedback_level=feed_lvl,
        min_level=1,
        text="-> Computing cosmology at the fiducial point",
    )
    tcosmo1 = time()
    fiducialcosmo = cosmology.cosmo_functions(fiducialparams, input_type)
    tcosmo2 = time()
    upt.time_print(
        feedback_level=feed_lvl,
        min_level=1,
        text="---> Cosmological functions obtained in: ",
        time_ini=tcosmo1,
        time_fin=tcosmo2,
    )

    global Photobiasparams
    if "GCph" in obs:
        if isinstance(photobiaspars, str):
            biasmodel = photobiaspars
        if isinstance(photobiaspars, dict):
            biasmodel = photobiaspars["bias_model"]
            print("using bias model: ", biasmodel)
            print("with bias keys: ", list(photobiaspars.keys())[1:])
        if photobiaspars is None:
            biasmodel = specs["ph_bias_model"]
            bias_prtz = specs["ph_bias_parametrization"]
            photobiaspars = dict()
            photobiaspars["bias_model"] = biasmodel
            if biasmodel == "binned" or biasmodel == "binned_constant":
                generate_bias_keys = bias_prtz[biasmodel]["generate_bias_keys"]
                if generate_bias_keys:
                    zbins = specs.get("z_bins_GCph", specs.get("z_bins_ph"))
                    for ind in range(1, len(zbins)):
                        keystr = bias_prtz[biasmodel]["keystr"]
                        key = keystr + str(ind)
                        photobiaspars[key] = np.sqrt(1 + 0.5 * (zbins[ind] + zbins[ind - 1]))
                else:
                    for key in bias_prtz[biasmodel].keys():
                        if key != "generate_bias_keys" and key != "keystr":
                            photobiaspars[key] = bias_prtz[biasmodel][key]
            else:
                for key in bias_prtz[biasmodel].keys():
                    photobiaspars[key] = bias_prtz[biasmodel][key]
    else:
        photobiaspars = {}
    Photobiasparams = deepcopy(photobiaspars)

    if "GCph" in obs:
        if specs["vary_ph_bias"] is not None:
            default_eps_ph_bias = specs["vary_ph_bias"]
            for key in photobiaspars.keys():
                if key != "bias_model":
                    # Only add the free parameters that are not already in the dictionary
                    freeparams.setdefault(key, default_eps_ph_bias)

    global photoparams
    if photopars is None:
        photopars = specs["photo_z_params"]
    photoparams = deepcopy(photopars)

    global IAparams
    if IApars is None:
        IApars = specs["IA_params"]
    IAparams = deepcopy(IApars)
    if "WL" in obs:
        if specs["vary_IA_pars"] is not None:
            default_eps_IA = specs["vary_IA_pars"]
            for key in IAparams.keys():
                if key != "IA_model":
                    # Only add the free parameters that are not already in the dictionary
                    freeparams.setdefault(key, default_eps_IA)

    global Spectrononlinearparams
    Spectrononlinearparams = dict()
    if "GCsp" in obs or "IM" in obs:
        gscp_nonlin_model = specs.get("nonlinear_model", "default")
        if gscp_nonlin_model == "default":
            Spectrononlinearparams = {}
        elif gscp_nonlin_model == "rescale_sigma_pv":
            nonlin_prtz = specs["nonlinear_parametrization"]
            nonlin_prmod = nonlin_prtz[gscp_nonlin_model]
            for key in nonlin_prmod.keys():
                Spectrononlinearparams[key] = nonlin_prmod[key]

    global Spectrobiasparams
    Spectrobiasparams = dict()
    if spectrobiaspars is not None:
        Spectrobiasparams = deepcopy(spectrobiaspars)
    global PShotparams
    PShotparams = dict()
    if PShotpars is not None:
        PShotparams = deepcopy(PShotpars)
    else:
        if "GCsp" in obs or "IM" in obs:
            bias_model = specs["sp_bias_model"]
            bias_sample = specs["sp_bias_sample"]
            bias_prtz = specs["sp_bias_parametrization"]
            bias_prmod = deepcopy(bias_prtz[bias_model])
            for key in bias_prmod.keys():
                Spectrobiasparams[key] = bias_prmod[key]
            # sanity check
            if bias_sample not in list(Spectrobiasparams.keys())[0]:
                print("Warning: bias_sample not found in bias_parameter keys")
            shot_noise_model = specs["shot_noise_model"]
            shot_noise_prtz = specs["shot_noise_parametrization"]
            try:
                Pskeys = shot_noise_prtz[shot_noise_model].keys()
            except AttributeError:
                Pskeys = []
            for key in Pskeys:
                PShotparams[key] = shot_noise_prtz[shot_noise_model][key]

    global IMbiasparams
    IMbiasparams = dict()
    if IMbiaspars is not None:
        IMbiasparams = deepcopy(IMbiaspars)
    else:
        if "IM" in obs:
            bias_model = specs["IM_bias_model"]
            bias_sample = specs["IM_bias_sample"]
            bias_prtz = specs["IM_bias_parametrization"]
            bias_prmod = deepcopy(bias_prtz[bias_model])
            for key in bias_prmod.keys():
                IMbiasparams[key] = bias_prmod[key]
            bias_model = specs["IM_bias_model"]
            bias_sample = specs["IM_bias_sample"]
            bias_prtz = specs["IM_bias_parametrization"]
            bias_prmod = deepcopy(bias_prtz[bias_model])
            for key in bias_prmod.keys():
                IMbiasparams[key] = bias_prmod[key]

    # Set the default free parameters for the spectro nuisance parameters
    default_eps_gc_nuis = settings["eps_gal_nuispars"]
    default_eps_gc_nonlin = settings["eps_gal_nonlinpars"]
    if "GCsp" in obs:
        for key in Spectrobiasparams:
            freeparams.setdefault(key, default_eps_gc_nuis)
    if "IM" in obs:
        for key in IMbiasparams:
            freeparams.setdefault(key, default_eps_gc_nuis)
    if "GCsp" in obs or "IM" in obs:
        for key in PShotparams:
            freeparams.setdefault(key, default_eps_gc_nuis)
        for key in Spectrononlinearparams:
            freeparams.setdefault(key, default_eps_gc_nonlin)
    # Only add the free parameters that are not already in the dictionary
    upt.debug_print("Final dict of free parameters in config.py:")
    upt.debug_print(freeparams)

    global latex_names
    latex_names_def = {
        "Omegam": r"\Omega_{{\rm m}, 0}",
        "Omegab": r"\Omega_{{\rm b}, 0}",
        "Om": r"\Omega_{{\rm m}, 0}",
        "Ob": r"\Omega_{{\rm b}, 0}",
        "h": r"h",
        "ns": r"n_{\rm s}",
        "tau": r"\tau",
        "sigma8": r"\sigma_8",
        "s8": r"\sigma_8",
        "Mnu": r"M_\nu",
        "mnu": r"\Sigma m_\nu",
        "Neff": r"N_\mathrm{eff}",
        "lnbgs8_1": r"\ln(b_g \sigma_8)_1",
        "lnbgs8_2": r"\ln(b_g \sigma_8)_2",
        "lnbgs8_3": r"\ln(b_g \sigma_8)_3",
        "lnbgs8_4": r"\ln(b_g \sigma_8)_4",
        "lnbgs8_5": r"\ln(b_g \sigma_8)_5",
        "lnbgs8_6": r"\ln(b_g \sigma_8)_6",
        "lnbgs8_7": r"\ln(b_g \sigma_8)_7",
        "lnbgs8_8": r"\ln(b_g \sigma_8)_8",
        "lnbgs8_9": r"\ln(b_g \sigma_8)_9",
        "lnbgs8_10": r"\ln(b_g \sigma_8)_10",
        "lnbgs8_11": r"\ln(b_g \sigma_8)_11",
        "lnbIs8_1": r"\ln(b_{IM} \sigma_8)_1",
        "lnbIs8_2": r"\ln(b_{IM} \sigma_8)_2",
        "lnbIs8_3": r"\ln(b_{IM} \sigma_8)_3",
        "lnbIs8_4": r"\ln(b_{IM} \sigma_8)_4",
        "lnbIs8_5": r"\ln(b_{IM} \sigma_8)_5",
        "lnbIs8_6": r"\ln(b_{IM} \sigma_8)_6",
        "lnbIs8_7": r"\ln(b_{IM} \sigma_8)_7",
        "lnbIs8_8": r"\ln(b_{IM} \sigma_8)_8",
        "lnbIs8_9": r"\ln(b_{IM} \sigma_8)_9",
        "lnbIs8_10": r"\ln(b_{IM} \sigma_8)_10",
        "lnbIs8_11": r"\ln(b_{IM} \sigma_8)_11",
        "Ps_1": r"P_{S1}",
        "Ps_2": r"P_{S2}",
        "Ps_3": r"P_{S3}",
        "Ps_4": r"P_{S4}",
        "E11": r"E_{11}",
        "E22": r"E_{22}",
        "bg_1": r"b_{g1}",
        "bg_2": r"b_{g2}",
        "bg_3": r"b_{g3}",
        "bg_4": r"b_{g4}",
        "bg_5": r"b_{g5}",
        "bg_6": r"b_{g6}",
        "bg_7": r"b_{g7}",
        "bg_8": r"b_{g8}",
        "bg_9": r"b_{g9}",
        "bg_10": r"b_{g10}",
        "bg_11": r"b_{g11}",
        "AIA": r"A_{IA}",
        "betaIA": r"\beta_{IA}",
        "etaIA": r"\eta_{IA}",
        "w0": r"w_0",
        "wa": r"w_a",
        "sigmap_0": r"\sigma_{p\,0}",
        "sigmap_1": r"\sigma_{p\,1}",
        "sigmap_2": r"\sigma_{p\,2}",
        "sigmap_3": r"\sigma_{p\,3}",
        "sigmav_0": r"\sigma_{v\,0}",
        "sigmav_1": r"\sigma_{v\,1}",
        "sigmav_2": r"\sigma_{v\,2}",
        "sigmav_3": r"\sigma_{v\,3}",
    }
    latex_names = deepcopy(latex_names_def)
    if isinstance(latexnames, dict):
        latex_names.update(latexnames)
