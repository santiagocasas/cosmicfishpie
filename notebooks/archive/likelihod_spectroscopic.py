#!/usr/bin/env python
# coding: utf-8

import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from nautilus import Prior, Sampler

import cosmicfishpie.analysis.fishconsumer as fco
from cosmicfishpie.analysis import fisher_operations as cfop
from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.likelihood.spectro_like import legendre_Pgg, loglike, observable_Pgg
from cosmicfishpie.LSSsurvey import spectro_cov as spcov
from cosmicfishpie.LSSsurvey import spectro_obs as spobs

os.environ["OMP_NUM_THREADS"] = "6"


# # Compute fiducial data and theory

# fiducial = {
#     "Omegam": 0.3145714273,
#     "Omegab": 0.0491989,
#     "h": 0.6737,
#     "ns": 0.96605,
#     "sigma8": 0.81,
#     "w0": -1.0,
#     "wa": 0.0,
#     "mnu": 0.06,
#     "Neff": 3.044,
# }

fiducial = {
    "Omegam": 0.3145714273,
    "Omegab": 0.0491989,
    "h": 0.6737,
    "ns": 0.96605,
    "sigma8": 0.81,
    # "Omegab":0.04897,
    # "h":0.6766,
    # "ns":0.9665,
    # "sigma8":0.815584,
    # "As" : 2.1098301595078886e-09,
    "w0": -1.0,
    "wa": 0.0,
    "mnu": 0.0,
    # "Neff":3.044,
    "num_massive_neutrinos": 0,
    "num_nu_massive": 0,
    "num_nu_massless": 3.044,
    "tau": 0.0561,
    "kmax": 10,
    "extrap_kmax": None,
}
observables = ["GCsp"]


# ## Compute symbolic Observable

options = {
    "accuracy": 1,
    "feedback": 1,
    "code": "symbolic",
    "outroot": "Euc-ISTF-Pess-symb",
    "survey_name": "Euclid",
    # "survey_name_spectro": "SKAO-Spectroscopic-Redbook",
    "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
    "survey_name_photo": False,
    # "survey_name_radio_IM": "SKAO-IM-Redbook",
    #'specs_dir': '../cosmicfishpie/configs/other_survey_specifications/',
    "cosmo_model": "LCDM",
    "bfs8terms": False,
}
cosmoFM_theory = cosmicfish.FisherMatrix(
    fiducialpars=fiducial,
    options=options,
    observables=observables,
    cosmoModel=options["cosmo_model"],
    surveyName=options["survey_name"],
)
print(cosmoFM_theory.Spectrobiaspars)
print(cosmoFM_theory.freeparams)
theory_fish = cosmoFM_theory.compute()


# cosmoFM_fid.set_pk_settings()
print(cosmoFM_theory.pk_cov.global_z_bin_mids)
print(cosmoFM_theory.Pk_kgrid.shape)
print(cosmoFM_theory.obs_spectrum)
print(cosmoFM_theory.Pk_musamp)
print(cosmoFM_theory.Pk_ksamp)
print(cosmoFM_theory.Pk_mumesh.shape)
print(cosmoFM_theory.Pk_kgrid.shape)
# spectro_fid = spobs.ComputeGalSpectro(cosmoFM_fid.fiducialcosmopars,
#                                        spectrobiaspars=cosmoFM_fid.Spectrobiaspars,
#                                        spectrononlinearpars=cosmoFM_fid.Spectrononlinpars,
#                                        PShotpars=cosmoFM_fid.PShotpars,
#                                        IMbiaspars=cosmoFM_fid.IMbiaspars,
#                                        fiducial_cosmo=cosmoFM_fid.fiducialcosmo,
#                                        use_bias_funcs=False,
#                                        configuration=cosmoFM_fid)

# spectro_cov_fid = spcov.SpectroCov(fiducialpars=cosmoFM_fid.fiducialcosmopars,
#                                    configuration=cosmoFM_fid)


# volume_surveys = np.array([spectro_cov_fid.volume_survey(ii) for ii in range(len(spectro_cov_fid.global_z_bin_mids))])


# ## Compute camb Observable

options = {
    "accuracy": 1,
    "feedback": 1,
    "code": "camb",
    "outroot": "Euc-ISTF-Pess-camb",
    "survey_name": "Euclid",
    # "survey_name_spectro": "SKAO-Spectroscopic-Redbook",
    "survey_name_spectro": "Euclid-Spectroscopic-ISTF-Pessimistic",
    "survey_name_photo": False,
    # "survey_name_radio_IM": "SKAO-IM-Redbook",
    #'specs_dir': '../cosmicfishpie/configs/other_survey_specifications/',
    "cosmo_model": "LCDM",
    "bfs8terms": False,
}
cosmoFM_data = cosmicfish.FisherMatrix(
    fiducialpars=fiducial,
    options=options,
    observables=observables,
    cosmoModel=options["cosmo_model"],
    surveyName=options["survey_name"],
)
print(cosmoFM_data.Spectrobiaspars)
print(cosmoFM_data.freeparams)
data_fish = cosmoFM_data.compute()


# cosmoFM_fid.set_pk_settings()
print(cosmoFM_data.pk_cov.global_z_bin_mids)
print(cosmoFM_data.Pk_kgrid.shape)
print(cosmoFM_data.obs_spectrum)
print(cosmoFM_data.Pk_musamp)
print(cosmoFM_data.Pk_ksamp)
print(cosmoFM_data.Pk_mumesh.shape)
print(cosmoFM_data.Pk_kgrid.shape)
spectro_data = spobs.ComputeGalSpectro(
    cosmoFM_data.fiducialcosmopars,
    spectrobiaspars=cosmoFM_data.Spectrobiaspars,
    spectrononlinearpars=cosmoFM_data.Spectrononlinpars,
    PShotpars=cosmoFM_data.PShotpars,
    IMbiaspars=cosmoFM_data.IMbiaspars,
    fiducial_cosmo=cosmoFM_data.fiducialcosmo,
    use_bias_funcs=False,
    configuration=cosmoFM_data,
)

spectro_cov_data = spcov.SpectroCov(
    fiducialpars=cosmoFM_data.fiducialcosmopars, configuration=cosmoFM_data
)


min(cosmoFM_theory.Pk_kgrid)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
pktheo = cosmoFM_theory.pk_obs_fid.observed_Pgg(1.0, cosmoFM_theory.Pk_kgrid, 0.66)
pkdata = cosmoFM_data.pk_obs_fid.observed_Pgg(1.0, cosmoFM_data.Pk_kgrid, 0.66)
axs[0].loglog(cosmoFM_theory.Pk_kgrid, pktheo)
axs[0].loglog(cosmoFM_data.Pk_kgrid, pkdata)
axs[0].legend(["symbolicPk Theory", "CAMB Data"])
axs[1].semilogx(cosmoFM_theory.Pk_kgrid, (pktheo / pkdata - 1))
axs[0].set_xlim(1e-3, 0.2)
axs[1].set_xlim(1e-3, 0.2)
axs[1].set_ylim(-0.01, 0.01)
axs[0].set_ylabel(r"$P_{gg}(k,z, \mu)$")
axs[1].set_ylabel(r"$P_{th}/P_{dat} - 1$")
axs[1].set_xlabel(r"$k$ [1/Mpc]")
axs[1].legend("Ratio Theory to Data")


parss = theory_fish.get_param_names()[0:6]
parss


truth_fish = dict(zip(parss, theory_fish.get_param_fiducial()[0:6]))
truth_fish


parss_nonui = parss[0:5]
parss_nonui


theory_fish_nonui = cfop.reshuffle(theory_fish, names=parss_nonui)
data_fish_nonui = cfop.reshuffle(data_fish, names=parss_nonui)


euclid_sp_all = fco.make_triangle_plot(
    fishers=[theory_fish, theory_fish_nonui, data_fish, data_fish_nonui],
    colors=["red", "orange", "blue", "green"],
    fisher_labels=[
        "Euclid Fisher GCsp symbolic",
        "Euclid Fisher GCsp no-nui symbolic",
        "Euclid Fisher GCsp camb",
        "Euclid Fisher GCsp no-nui camb",
    ],
    truth_values=truth_fish,
    params=["h", "ns", "sigma8"],
)


# ## Choose data vector

obsPgg_data = observable_Pgg(spectro_cov_data, cosmoFM_data)
obsPgg_data.shape


P_ell_data = legendre_Pgg(obsPgg_data, cosmoFM_data)
P_ell_data.shape


fid_truth = {par: cosmoFM_data.allparams[par] for par in cosmoFM_data.freeparams.keys()}
fid_truth


# ## Compute example loglike

samp1dic = {
    "Omegam": 0.31,
    "Omegab": 0.05,
    "h": 0.68,
    "ns": 0.96,
    "sigma8": 0.82,
    "w0": -1.01,
    "wa": 0.2,
    "Ps_0": 0.0,
    "Ps_1": 0.0,
    "Ps_2": 0.0,
    "Ps_3": 10.0,
    "lnbg_1": 0.37944989,
    "lnbg_2": 0.4738057,
    "lnbg_3": 0.55760176,
    "lnbg_4": 0.6,
    "sigmap_1": 1.0,
    "sigmap_2": 0.0,
    "sigmap_3": 1.0,
    "sigmap_4": 1.0,
    "sigmav_1": 1.0,
    "sigmav_2": 1.0,
    "sigmav_3": 0.0,
    "sigmav_4": 10,
    "bI_c1": -0.3,
    "bI_c2": +0.6,
}
print(
    "Wedges loglike: ",
    loglike(
        samp1dic,
        leg_flag="wedges",
        cosmoFM_theory=cosmoFM_theory,
        cosmoFM_data=cosmoFM_data,
        data_obsPgg=obsPgg_data,
    ),
)
print(
    "Multipoles loglike: ",
    loglike(
        samp1dic,
        leg_flag="legendre",
        cosmoFM_theory=cosmoFM_theory,
        cosmoFM_data=cosmoFM_data,
        data_obsPgg=obsPgg_data,
    ),
)

# ## Initialize Nautilus sampler

prior = Prior()
prior_noshot = Prior()
prior_noshot_no_bi = Prior()
prior_nosigma = Prior()


prior_dict = {
    "Omegam": [0.24, 0.4],
    "Omegab": [0.04, 0.06],
    "h": [0.61, 0.73],
    "ns": [0.92, 1.00],
    "sigma8": [0.79, 0.83],
    "Ps_0": [-10, 10],
    "Ps_1": [-10, 10],
    "Ps_2": [-10, 10],
    "Ps_3": [-10, 10],
    "lnbg_1": [0, 1],
    "lnbg_2": [0, 1],
    "lnbg_3": [0, 1],
    "lnbg_4": [0, 1],
    "sigmap_1": [0, 10],
    "sigmap_2": [0, 10],
    "sigmap_3": [0, 10],
    "sigmap_4": [0, 10],
    "sigmav_1": [0, 10],
    "sigmav_2": [0, 10],
    "sigmav_3": [0, 10],
    "sigmav_4": [0, 10],
    "bI_c1": [-5, 5],
    "bI_c2": [-5, 5],
}


cosmoFM_theory.freeparams


for par in prior_dict.keys():
    if par in cosmoFM_theory.freeparams.keys():
        dist_prior = (prior_dict[par][0], prior_dict[par][1])
        prior.add_parameter(par, dist_prior)
        if "sigmap" not in par and "sigmav" not in par:
            prior_nosigma.add_parameter(par, dist_prior)
        if "Ps" not in par and ("sigmap" not in par and "sigmav" not in par):
            prior_noshot.add_parameter(par, dist_prior)
        if "Ps" not in par and "lnbg" not in par and ("sigmap" not in par and "sigmav" not in par):
            prior_noshot_no_bi.add_parameter(par, dist_prior)


print("Loaded prior into Nautilus with dimension", prior.dimensionality())
print("Loaded prior into Nautilus with dimension", prior_noshot.dimensionality())
print("Loaded prior into Nautilus with dimension", prior_noshot_no_bi.dimensionality())
print("Loaded prior into Nautilus with dimension", prior_nosigma.dimensionality())
print("Prior keys: ", prior.keys)
print("Prior keys noshot: ", prior_noshot.keys)
print("Prior keys noshot no bi: ", prior_noshot_no_bi.keys)
print("Prior keys nosigma: ", prior_nosigma.keys)


outfile_chain_path = "cosmicjellyfish_eucid_spectro_wedges_nonuis_vs_camb"
prior_to_use = prior_noshot_no_bi
print("Loading prior with keys: ", prior_to_use.keys)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    sampler = Sampler(
        prior_to_use,
        loglike,
        n_live=1000,
        n_networks=4,
        n_batch=256,
        pool=8,
        pass_dict=False,
        filepath=outfile_chain_path + ".hdf5",
        resume=True,
        likelihood_kwargs={
            "leg_flag": "wedges",
            "prior": prior_to_use,
            "cosmoFM_theory": cosmoFM_theory,
            "cosmoFM_data": cosmoFM_data,
            "data_obsPgg": obsPgg_data,
        },
    )
    sampler.run(verbose=True, discard_exploration=True)
    log_z_all = sampler.evidence()
print("Evidence:", log_z_all)
print("Sampling posterior")
points_all, log_w_all, log_l_all = sampler.posterior()


sample_wghlkl = np.vstack((points_all.T, np.exp(log_w_all), log_l_all)).T


outfile_new = outfile_chain_path
print(f"Saving chain to text file {outfile_new}.txt")


np.savetxt(outfile_new + ".txt", sample_wghlkl)


maindir = "./"


cjelly_spectro_nonui = maindir + "cosmicjellyfish_eucid_spectro_nonuis.txt"
spectro_chain_nonui = fco.load_Nautilus_chains_from_txt(
    cjelly_spectro_nonui, param_cols=parss_nonui
)


cjelly_spectro_nonui_vs_camb = maindir + "cosmicjellyfish_eucid_spectro_wedges_nonuis_vs_camb.txt"
spectro_chain_nonui_vs_camb = fco.load_Nautilus_chains_from_txt(
    cjelly_spectro_nonui_vs_camb, param_cols=parss_nonui
)


euclid_sp_nonui = fco.make_triangle_plot(
    chains=[spectro_chain_nonui, spectro_chain_nonui_vs_camb],
    fishers=[
        theory_fish_nonui,
        # data_fish_nonui
    ],
    colors=[
        "purple",
        #'red',
        "green",
        "orange",
    ],
    chain_labels=["Euclid MCMC GCsp no-nui symb", "Euclid MCMC GCsp no-nui symb vs camb"],
    fisher_labels=[
        "Euclid Fisher GCsp no-nui symb",
        #'Euclid Fisher GCsp no-nui camb',
    ],
    truth_values=truth_fish,
    params=parss_nonui,
    smooth=15,
    bins=15,
)
