# This class defines global variables that will not change through
# the computation of one fisher matrix

import glob
import os
from copy import deepcopy
from time import time

import numpy as np
import yaml

import cosmicfishpie.cosmology.cosmology as cosmology
from cosmicfishpie.cosmology.nuisance import Nuisance
from cosmicfishpie.utilities.utils import printing as upt


def init(
    options=dict(),
    specifications=dict(),
    observables=None,
    freepars=None,
    extfiles=None,
    fiducialpars=None,
    biaspars=None,
    photopars=None,
    IApars=None,
    PShotpars=None,
    Spectrobiaspars=None,
    spectrononlinearpars=None,
    IMbiaspars=None,
    surveyName="Euclid",
    cosmoModel="w0waCDM",
    latexnames=None,
):
    global settings
    settings = options
    if "camb_path" not in settings:
        import camb

        cambpath = os.path.dirname(camb.__file__)
        settings["camb_path"] = cambpath
    # Set defaults if not contained previously in options
    settings.setdefault(
        "specs_dir",
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "survey_specifications"
        ),
    )
    settings.setdefault("survey_name", surveyName)
    settings.setdefault("derivatives", "3PT")
    settings.setdefault("nonlinear", True)
    settings.setdefault("nonlinear_photo", True)
    settings.setdefault("bfs8terms", True)
    settings.setdefault("vary_bias_str", "lnb")
    settings.setdefault("AP_effect", True)
    settings.setdefault("FoG_switch", True)
    settings.setdefault("GCsp_linear", False)
    settings.setdefault("fix_cosmo_nl_terms", True)
    settings.setdefault("Pshot_nuisance_fiducial", 0.0)
    settings.setdefault("pivot_z_IA", 0.0)
    settings.setdefault("accuracy", 1)
    settings.setdefault("feedback", 2)
    settings.setdefault("activateMG", False)
    settings.setdefault("external_activateMG", False)
    settings.setdefault("cosmo_model", cosmoModel)
    settings.setdefault("outroot", "default_run")
    settings.setdefault("code", "camb")
    settings.setdefault("memorize_cosmo", False)
    settings.setdefault("results_dir", "./results")
    settings.setdefault(
        "boltzmann_yaml_path",
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "boltzmann_yaml_files"
        ),
    )
    settings.setdefault(
        "class_config_yaml", os.path.join(settings["boltzmann_yaml_path"], "class", "default.yaml")
    )
    settings.setdefault(
        "camb_config_yaml", os.path.join(settings["boltzmann_yaml_path"], "camb", "default.yaml")
    )
    settings.setdefault("fishermatrix_file_extension", ".txt")
    settings.setdefault("savgol_window", 101)
    settings.setdefault("savgol_polyorder", 3)
    settings.setdefault("savgol_width", 1.358528901113328)
    settings.setdefault("savgol_internalsamples", 800)
    settings.setdefault("savgol_internalkmin", 0.001)
    settings.setdefault("eps_cosmopars", 0.01)
    settings.setdefault("eps_gal_nuispars", 0.0001)
    settings.setdefault("GCsp_Tracer", "matter")
    settings.setdefault("GCph_Tracer", "matter")
    settings.setdefault("ShareDeltaNeff", False)

    global external
    global input_type
    if extfiles is not None and settings["code"] == "external":
        print("Using input files for cosmology observables.")
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
        external.update(extfiles)

        if os.path.isdir(external["directory"]):
            ff = external["fiducial_folder"]
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
                    print("External directory: ", external["directory"])
                    print("{:d} subfolders for parameter {:s}".format(lensub, dd))
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
        input_type = settings["code"]
        global boltzmann_cambpars
        boltzmann_yaml_file = open(settings["camb_config_yaml"])
        parsed_boltzmann = yaml.load(boltzmann_yaml_file, Loader=yaml.FullLoader)
        boltzmann_cambpars = parsed_boltzmann
        external = None
    else:
        print("No external input files used in this calculation.")
        print("No Einstein-Boltzmann-Solver (EBS) specified.")
        print("Defaulting to EBS camb")
        # settings['code'] = 'camb'
        input_type = settings["code"]
        external = None

    def ngal_per_bin(ngal_sqarmin, zbins):
        # compute num galaxies per bin for whole sky area
        nbins = len(zbins[:-1])
        ones = np.ones_like(zbins[:-1])
        ngal_bin = (ngal_sqarmin / nbins) * 3600 * (180 / np.pi) ** 2
        numgal = ngal_bin * ones
        return numgal

    global specs

    specs_defaults = {}
    specs_defaults["specs_dir"] = settings["specs_dir"]
    specs = specs_defaults.copy()  # start with default dict

    global survey_equivalence

    def survey_equivalence(surv_str):
        # def_surv = 'Euclid'
        if "Euclid" in surv_str:
            surv = "Euclid"
        if "Rubin" in surv_str:
            surv = "Rubin"
        elif surv_str == "DESI_E":
            surv = "DESI_ELG"
        elif surv_str == "DESI_B":
            surv = "DESI_BGS"
        elif "SKA1" in surv_str:
            surv = "SKA1"
        elif "SKAO" in surv_str:
            surv = "SKA1"
        elif "SKA2" in surv_str:
            surv = "SKA2"
        else:
            surv = surv_str
        return surv

    specs["gc_specs_files_dict"] = {
        "default": "Euclid_GCsp_IST.dat",
        "Euclid": "Euclid_GCsp_IST.dat",
        "SKA1": "SKA1_GCsp_MDB2_Redbook.dat",
        "SKA2": "SKA2_GCsp.dat",
        "DESI_ELG": "DESI_ELG_GCsp.dat",
        "DESI_ELG_4bins": "DESI_4bins_ELG_GCsp.dat",
        "SKA1 x DESI_ELG": "DESI_ELG_GCsp.dat",
        "DESI_BGS": "DESI_BGS_GCsp.dat",
        "DESI_BGS_2bins": "DESI_2bins_BGS_GCsp.dat",
    }
    if "Euclid" in surveyName:
        if surveyName == "Euclid":
            surveyName = "Euclid-ISTF-Optimistic"
        yaml_file = open(os.path.join(settings["specs_dir"], surveyName + ".yaml"))
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        specificationsf = parsed_yaml_file["specifications"]
        z_bins = specificationsf["z_bins"]
        specificationsf["z_bins"] = np.array(z_bins)
        specificationsf["ngalbin"] = ngal_per_bin(
            specificationsf["ngal_sqarmin"], specificationsf["z_bins"]
        )
        specificationsf["z0"] = specificationsf["zm"] / np.sqrt(2)
        specificationsf["z0_p"] = specificationsf["z0"]
        specificationsf["binrange"] = range(1, len(specificationsf["z_bins"]))
        specificationsf["survey_name"] = surveyName
    elif "SKA1" in surveyName:
        yaml_file = open(os.path.join(settings["specs_dir"], "SKA1-Redbook-Optimistic.yaml"))
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        specificationsf = parsed_yaml_file["specifications"]
        z_bins = specificationsf["z_bins"]
        specificationsf["z_bins"] = np.array(z_bins)
        specificationsf["ngalbin"] = ngal_per_bin(
            specificationsf["ngal_sqarmin"], specificationsf["z_bins"]
        )
        # numgal =  specificationsf['ngal_per_bin']*np.ones_like(z_bins[:-1])
        specificationsf["z0"] = specificationsf["zm"] / np.sqrt(2)
        specificationsf["z0_p"] = 1.0
        specificationsf["IM_bins_file"] = "SKA1_IM_MDB1_Redbook.dat"
        specificationsf["IM_THI_noise_file"] = "SKA1_THI_sys_noise.txt"
        # MMmod:
        specificationsf["binrange"] = range(1, len(specificationsf["z_bins"]))
        specificationsf["survey_name"] = surveyName
    elif (
        surveyName == "SKA1-pessimistic"
    ):  # needs to be modified to account for pessimistic and cross
        yaml_file = open(os.path.join(settings["specs_dir"], "SKA1-Redbook-Pessimistic.yaml"))
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        specificationsf = parsed_yaml_file["specifications"]
        z_bins = specificationsf["z_bins"]
        specificationsf["z_bins"] = np.array(z_bins)
        specificationsf["ngalbin"] = ngal_per_bin(
            specificationsf["ngal_sqarmin"], specificationsf["z_bins"]
        )
        specificationsf["z0"] = specificationsf["zm"] / np.sqrt(2)
        specificationsf["z0_p"] = 1.0
        specificationsf["binrange"] = range(1, len(specificationsf["z_bins"]))
        specificationsf["survey_name"] = surveyName
    # if 'SKA' in surveyName:
    #     specificationsf['IM_bins_file'] = 'SKA1_IM_MDB1_Redbook.dat'
    #     specificationsf['IM_THI_noise_file'] = 'SKA1_THI_sys_noise.txt'
    elif "Rubin" in surveyName:
        if surveyName == "Rubin":
            surveyName = "Rubin-Optimistic"
        yaml_file = open(os.path.join(settings["specs_dir"], surveyName + ".yaml"))
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        specificationsf = parsed_yaml_file["specifications"]
        z_bins = specificationsf["z_bins"]
        specificationsf["z_bins"] = np.array(z_bins)
        specificationsf["ngalbin"] = ngal_per_bin(
            specificationsf["ngal_sqarmin"], specificationsf["z_bins"]
        )
        specificationsf["z0"] = specificationsf["zm"] / np.sqrt(2)
        specificationsf["z0_p"] = 1.0
        specificationsf["binrange"] = range(1, len(specificationsf["z_bins"]))
        specificationsf["survey_name"] = surveyName
        print("Survey loaded:  ", surveyName)
    elif "DESI" in surveyName:
        yaml_file = open(os.path.join(settings["specs_dir"], "DESI-Optimistic.yaml"))
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        specificationsf = parsed_yaml_file["specifications"]
        z_bins = specificationsf["z_bins"]
        specificationsf["z_bins"] = np.array(z_bins)
        # numgal =  specificationsf['ngal_per_bin']*np.ones_like(z_bins[:-1])
        # specificationsf['ngalbin'] = numgal
        specificationsf["ngalbin"] = ngal_per_bin(
            specificationsf["ngal_sqarmin"], specificationsf["z_bins"]
        )
        specificationsf["z0"] = specificationsf["zm"] / np.sqrt(2)
        specificationsf["z0_p"] = 1.0
        # SC: These specifications are loaded, but are actually not used, since they are not for spectro
        # Only needed specs are loaded in nuisance.py
        specificationsf["binrange"] = range(1, len(specificationsf["z_bins"]))
        specificationsf["survey_name"] = surveyName
    elif surveyName == "Planck":
        yaml_file = open(os.path.join(settings["specs_dir"], "Planck.yaml"))
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
        specificationsf = parsed_yaml_file["specifications"]
    else:
        print("Survey name passed: ", surveyName)
        print(
            "Other named survey specifications not implemented yet.",
            "Please pass your custom specifications as a dictionary.",
        )

    specs.update(specificationsf)  # update keys if present in files
    specs.update(specifications)  # update keys if passed by users

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
    freeparams = freepars

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
        print("Custom fiducial parameters loaded")
    fiducialparams = fiducialpars

    global fiducialcosmo
    feed_lvl = settings["feedback"]
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

    global biasparams
    if "GCph" in obs:
        if biaspars is None:
            biasmodel = specs["bias_model"]
        elif isinstance(biaspars, str):
            biasmodel = biaspars
        if not isinstance(biaspars, dict):
            if biasmodel == "sqrt":
                biaspars = {"bias_model": "sqrt", "b0": 1.0}
            elif biasmodel == "flagship":
                biaspars = {"bias_model": "flagship", "A": 1.0, "B": 2.5, "C": 2.8, "D": 1.6}
            elif biasmodel == "binned" or biasmodel == "binned_constant":
                biaspars = {"bias_model": biasmodel}
                zbins = specs["z_bins"]
                for ind in range(1, len(zbins)):
                    key = "b" + str(ind)
                    biaspars[key] = np.sqrt(1 + 0.5 * (zbins[ind] + zbins[ind - 1]))
            else:
                print("Bias model not implemented yet or not correct.")
    else:
        biaspars = {}

    biasparams = biaspars
    if "GCph" in obs:
        if specs["vary_ph_bias"] is not None:
            for key in biaspars.keys():
                if key != "bias_model":
                    freeparams[key] = specs["vary_ph_bias"]

    global photoparams
    if photopars is None:
        print("No photo-z parameters specified. Using default: Euclid-like")
        photopars = {
            "fout": 0.1,  # does this need to be updated for SKA1??
            "co": 1,
            "cb": 1,
            "sigma_o": 0.05,
            "sigma_b": 0.05,
            "zo": 0.1,
            "zb": 0.0,
        }
    photoparams = photopars

    global IAparams
    if IApars is None:
        print("No IA specified. Using default: eNLA")
        IApars = {"IA_model": "eNLA", "AIA": 1.72, "betaIA": 2.17, "etaIA": -0.41}
    IAparams = IApars
    if "WL" in obs:
        if specs["vary_IA_pars"] is not None:
            for key in IAparams.keys():
                if key != "IA_model":
                    freeparams[key] = specs["vary_IA_pars"]

    nuis = Nuisance()

    # def bterm_z_key(z_ind, z_mids, bias_sample='g'):
    #     if bias_sample=='g':
    #         bi_at_z_mids = nuis.gcsp_bias_at_zm()
    #     if bias_sample=='I':
    #         bi_at_z_mids = nuis.IM_bias_at_zm()
    #     bstring = settings['vary_bias_str']
    #     bstring = bstring+bias_sample
    #     b_i = bi_at_z_mids[z_ind-1]
    #     if settings['bfs8terms'] ==  True:
    #         bstring = bstring + 's8'
    #         b_i = b_i * fiducialcosmo.sigma8_of_z(z_mids[z_ind-1])
    #     bstring = bstring + '_'
    #     bstring = bstring + str(z_ind)
    #     if 'ln' in bstring:
    #         b_i = np.log(b_i)
    #     b_i = b_i.item()    ## Convert 1-element array to scalar
    #     return bstring, b_i

    global PShotparams
    PShotparams = dict()

    global Spectrobiasparams
    Spectrobiasparams = dict()

    global Spectrononlinearparams
    Spectrononlinearparams = dict()

    global IMbiasparams
    IMbiasparams = dict()

    if spectrononlinearpars is not None:
        Spectrononlinearparams = spectrononlinearpars
        if "GCsp" in obs:
            if "vary_GCsp_nonlinear_pars" in specs.keys():
                for key in Spectrononlinearparams.keys():
                    freeparams[key] = specs["vary_GCsp_nonlinear_pars"]

    if Spectrobiaspars is not None:
        Spectrobiasparams = Spectrobiaspars
    else:
        if "GCsp" in obs:
            bias_sample = "g"
            z_mids = nuis.gcsp_zbins_mids()
            Nbin = len(z_mids)
            for ii in range(1, Nbin + 1):
                bstr_i, b_i = nuis.bterm_z_key(ii, z_mids, fiducialcosmo, bias_sample)
                Spectrobiasparams[bstr_i] = b_i
                PShotparams["Ps_" + str(ii)] = nuis.extra_Pshot_noise()

    if IMbiaspars is not None:
        IMbiasparams = IMbiaspars
    else:
        if "IM" in obs:
            bias_sample = "I"
            z_mids = nuis.IM_zbins_mids()
            Nbin = len(z_mids)
            for ii in range(1, Nbin + 1):
                bstr_i, b_i = nuis.bterm_z_key(ii, z_mids, fiducialcosmo, bias_sample)
                IMbiasparams[bstr_i] = b_i

    if PShotpars is not None:
        PShotparams = PShotpars

    if "GCsp" in obs:
        for key in Spectrobiasparams:
            freeparams[key] = settings["eps_gal_nuispars"]
        if "IM" not in obs:
            for key in PShotparams:
                freeparams[key] = settings["eps_gal_nuispars"]
    if "IM" in obs:
        for key in IMbiasparams:
            freeparams[key] = settings["eps_gal_nuispars"]
    print("*** Dictionary of varied parameters in this Fisher Matrix run: ")
    print(freeparams)
    print("                                                            ***")

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
        "bg_1": r"\ln(b_g)_1",
        "bg_2": r"\ln(b_g)_2",
        "bg_3": r"\ln(b_g)_3",
        "bg_4": r"\ln(b_g)_4",
        "bg_5": r"\ln(b_g)_5",
        "bg_6": r"\ln(b_g)_6",
        "bg_7": r"\ln(b_g)_7",
        "bg_8": r"\ln(b_g)_8",
        "bg_9": r"\ln(b_g)_9",
        "bg_10": r"\ln(b_g)_10",
        "bg_11": r"\ln(b_g)_11",
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
